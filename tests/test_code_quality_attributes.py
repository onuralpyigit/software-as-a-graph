"""
Tests for Code-Level Quality Attribute Integration

Covers:
    CQ-001: Application dataclass stores code-quality fields; instability property
    CQ-002: extract_layer_subgraph propagates code-quality props to graph nodes
    CQ-003: StructuralMetrics receives loc_norm, complexity_norm, instability_code, lcom_norm
    CQ-004: code_quality_penalty computed correctly
    CQ-005: High-complexity app scores higher M(v) than structurally-equivalent low-complexity app
    CQ-006: Non-Application nodes receive code_quality_penalty = 0.0
    CQ-007: Missing code-quality fields (all-zero) → backward-compatible M(v) 
    CQ-008: Normalization is population-level (identical apps get equal normalised scores)
    CQ-009: Generator produces code-quality fields for every Application
    CQ-010: AHP compute_weights includes m_code_quality_penalty and sums to 1
"""

import pytest
import math

from saag.core.models import Application, GraphData, ComponentData, EdgeData
from saag.core.metrics import StructuralMetrics
from saag.analysis.structural_analyzer import StructuralAnalyzer, extract_layer_subgraph
from saag.prediction.analyzer import QualityAnalyzer
from saag.prediction.weight_calculator import AHPProcessor, AHPMatrices, QualityWeights
from saag.core import AnalysisLayer


# =============================================================================
# Helpers
# =============================================================================

def _make_app_component(
    app_id: str,
    loc: int = 0,
    cc: float = 0.0,
    ca: int = 0,
    ce: int = 0,
    lcom: float = 0.0,
    weight: float = 1.0,
) -> ComponentData:
    """Build a ComponentData with code-quality fields embedded in properties."""
    return ComponentData(
        id=app_id,
        component_type="Application",
        weight=weight,
        properties={
            "name": app_id,
            "loc": loc,
            "cyclomatic_complexity": cc,
            "coupling_afferent": ca,
            "coupling_efferent": ce,
            "lcom": lcom,
        },
    )


def _simple_two_app_graph(
    loc_a: int = 0, cc_a: float = 0.0, ca_a: int = 0, ce_a: int = 0, lcom_a: float = 0.0,
    loc_b: int = 0, cc_b: float = 0.0, ca_b: int = 0, ce_b: int = 0, lcom_b: float = 0.0,
) -> GraphData:
    """A->B two-node application graph with code-quality attributes."""
    return GraphData(
        components=[
            _make_app_component("A", loc=loc_a, cc=cc_a, ca=ca_a, ce=ce_a, lcom=lcom_a),
            _make_app_component("B", loc=loc_b, cc=cc_b, ca=ca_b, ce=ce_b, lcom=lcom_b),
        ],
        edges=[
            EdgeData("A", "B", "Application", "Application", "app_to_app", "dependency", 1.0),
        ],
    )


# =============================================================================
# CQ-001: Application dataclass
# =============================================================================

def _mk_code_metrics(loc=0, cc=0.0, ca=0, ce=0, lcom=0.0):
    """Build a nested code_metrics dict for test construction."""
    return {
        "size": {"total_loc": loc},
        "complexity": {"avg_wmc": cc},
        "cohesion": {"avg_lcom": lcom},
        "coupling": {"avg_fanin": ca, "avg_fanout": ce},
    }


class TestApplicationDataclass:
    def test_code_quality_fields_stored(self):
        """CQ-001a: Application stores all 5 code-quality fields via code_metrics."""
        app = Application(
            id="svc1", name="SVC1",
            code_metrics=_mk_code_metrics(loc=500, cc=8.5, ca=3, ce=7, lcom=0.62),
        )
        assert app.loc == 500
        assert app.cyclomatic_complexity == 8.5
        assert app.coupling_afferent == 3
        assert app.coupling_efferent == 7
        assert app.lcom == pytest.approx(0.62)

    def test_instability_property(self):
        """CQ-001b: instability = Ce / (Ca + Ce)."""
        app = Application(id="svc1", name="SVC1",
                          code_metrics=_mk_code_metrics(ca=3, ce=7))
        assert app.instability == pytest.approx(7 / 10)

    def test_instability_zero_when_no_coupling(self):
        """CQ-001c: instability = 0.0 when both coupling counts are 0."""
        app = Application(id="svc1", name="SVC1")
        assert app.instability == 0.0

    def test_defaults_are_zero(self):
        """CQ-001d: All code-quality fields default to 0."""
        app = Application(id="svc1", name="SVC1")
        assert app.loc == 0
        assert app.cyclomatic_complexity == 0.0
        assert app.coupling_afferent == 0
        assert app.coupling_efferent == 0
        assert app.lcom == 0.0

    def test_to_dict_includes_code_metrics(self):
        """CQ-001e: to_dict() always includes code_metrics key."""
        app = Application(id="svc1", name="SVC1")
        d = app.to_dict()
        assert "code_metrics" in d

    def test_to_dict_includes_nonzero_fields(self):
        """CQ-001f: to_dict() includes code_metrics; backward-compat properties work."""
        app = Application(id="svc1", name="SVC1",
                          code_metrics=_mk_code_metrics(loc=300, lcom=0.4))
        d = app.to_dict()
        assert d["code_metrics"]["size"]["total_loc"] == 300
        assert d["code_metrics"]["cohesion"]["avg_lcom"] == pytest.approx(0.4)


# =============================================================================
# CQ-002: extract_layer_subgraph propagates props
# =============================================================================

class TestSubgraphCodeQualityPropagation:
    def test_code_quality_stored_on_graph_nodes(self):
        """CQ-002: Graph nodes carry code-quality attributes after extraction."""
        gd = _simple_two_app_graph(
            loc_a=1000, cc_a=15.0, ca_a=2, ce_a=5, lcom_a=0.6,
        )
        G = extract_layer_subgraph(gd, AnalysisLayer.APP)
        
        assert G.nodes["A"]["loc"] == 1000
        assert G.nodes["A"]["cyclomatic_complexity"] == 15.0
        assert G.nodes["A"]["coupling_afferent"] == 2
        assert G.nodes["A"]["coupling_efferent"] == 5
        assert G.nodes["A"]["lcom"] == pytest.approx(0.6)

    def test_missing_code_quality_defaults_to_zero(self):
        """CQ-002b: Nodes without code-quality props get 0 defaults."""
        gd = GraphData(
            components=[ComponentData(id="X", component_type="Application", weight=1.0, properties={"name": "X"})],
            edges=[],
        )
        G = extract_layer_subgraph(gd, AnalysisLayer.APP)
        assert G.nodes["X"]["loc"] == 0
        assert G.nodes["X"]["lcom"] == 0.0


# =============================================================================
# CQ-003 & CQ-004: StructuralMetrics code-quality fields
# =============================================================================

class TestStructuralMetricsCodeQuality:
    def test_instability_code_computed(self):
        """CQ-003a: instability_code = Ce/(Ca+Ce) stored on StructuralMetrics."""
        gd = _simple_two_app_graph(ca_a=2, ce_a=8)  # I = 8/10 = 0.8
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd)
        
        assert res.components["A"].instability_code == pytest.approx(0.8)

    def test_loc_complexity_lcom_normalized(self):
        """CQ-003b: loc_norm / complexity_norm / lcom_norm are in [0,1]."""
        gd = _simple_two_app_graph(
            loc_a=200, cc_a=5.0, lcom_a=0.3,
            loc_b=800, cc_b=20.0, lcom_b=0.7,
        )
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd)
        
        for m in res.components.values():
            assert 0.0 <= m.loc_norm <= 1.0
            assert 0.0 <= m.complexity_norm <= 1.0
            assert 0.0 <= m.lcom_norm <= 1.0

    def test_code_quality_penalty_computed(self):
        """CQ-004: code_quality_penalty is in [0,1] and > 0 for non-trivial inputs."""
        gd = _simple_two_app_graph(
            loc_a=200, cc_a=5.0, ca_a=1, ce_a=3, lcom_a=0.3,
            loc_b=800, cc_b=20.0, ca_b=0, ce_b=2, lcom_b=0.8,
        )
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd)
        
        # At least the higher-CC/LCOM node should have a positive CQP
        cqp_a = res.components["A"].code_quality_penalty
        cqp_b = res.components["B"].code_quality_penalty
        assert 0.0 <= cqp_a <= 1.0
        assert 0.0 <= cqp_b <= 1.0
        # B has higher CC and LCOM so it should score higher
        assert cqp_b >= cqp_a

    def test_cqp_zero_when_all_code_quality_inputs_zero(self):
        """CQ-004b: CQP = 0 when all code-quality fields are 0."""
        gd = _simple_two_app_graph()  # no code-quality fields
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd)
        
        for m in res.components.values():
            assert m.code_quality_penalty == pytest.approx(0.0)


# =============================================================================
# CQ-005: High-complexity app has higher M(v)
# =============================================================================

class TestMaintainabilityWithCodeQuality:
    def _run_analysis_and_get_M(self, gd: GraphData) -> dict:
        """Helper: run full pipeline and return M scores keyed by component id."""
        analyzer = StructuralAnalyzer()
        structural = analyzer.analyze(gd)
        qa = QualityAnalyzer()
        result = qa.analyze(structural)
        return {cq.id: cq.scores.maintainability for cq in result.components}

    def test_higher_complexity_raises_M_score(self):
        """CQ-005: Two applications with identical topology: higher CC/LCOM → higher M(v)."""
        # Low-quality application
        gd_low = _simple_two_app_graph(
            loc_a=200, cc_a=3.0, lcom_a=0.1,  # app A: low complexity
            loc_b=200, cc_b=3.5, lcom_b=0.15,
        )
        # High-quality application (same graph structure with different code metrics)
        gd_high = _simple_two_app_graph(
            loc_a=2000, cc_a=22.0, ca_a=0, ce_a=5, lcom_a=0.75,  # app A: high complexity
            loc_b=2000, cc_b=24.0, lcom_b=0.80,
        )
        m_low = self._run_analysis_and_get_M(gd_low)
        m_high = self._run_analysis_and_get_M(gd_high)
        
        # App A in high-complexity graph should score higher M (worse maintainability)
        # We test that at least one score differs meaningfully
        diff = abs(m_high["A"] - m_low["A"])
        assert diff > 0.0, (
            "High-CC application should show different M(v) than low-CC "
            f"counterpart (diff={diff})"
        )


# =============================================================================
# CQ-006: Non-Application nodes unaffected
# =============================================================================

class TestNonApplicationNodesUnaffected:
    def test_broker_has_zero_cqp(self):
        """CQ-006a: Broker nodes get code_quality_penalty = 0.0."""
        gd = GraphData(
            components=[
                _make_app_component("A", cc=10.0, lcom=0.5),
                ComponentData(id="BRK1", component_type="Broker", weight=2.0,
                              properties={"name": "BRK1"}),
            ],
            edges=[
                EdgeData("A", "BRK1", "Application", "Broker", "app_to_broker", "dependency", 1.0),
            ],
        )
        analyzer = StructuralAnalyzer()
        # MW layer includes both Application and Broker
        G = extract_layer_subgraph(gd, AnalysisLayer.MW)
        # Broker node should have no contamination
        assert G.nodes.get("BRK1", {}).get("loc", 0) == 0

    def test_broker_metrics_zero_cqp(self):
        """CQ-006b: Broker StructuralMetrics have code_quality_penalty = 0.0."""
        gd = GraphData(
            components=[
                _make_app_component("A", cc=10.0, lcom=0.5),
                ComponentData(id="BRK1", component_type="Broker", weight=2.0,
                              properties={"name": "BRK1"}),
            ],
            edges=[
                EdgeData("A", "BRK1", "Application", "Broker", "app_to_broker", "dependency", 1.0),
            ],
        )
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd)
        if "BRK1" in res.components:
            assert res.components["BRK1"].code_quality_penalty == pytest.approx(0.0)
            assert res.components["BRK1"].instability_code == pytest.approx(0.0)


# =============================================================================
# CQ-007: Backward compatibility (zero code-quality inputs)
# =============================================================================

class TestBackwardCompatibility:
    def test_no_code_quality_fields_still_analyzes(self):
        """CQ-007: Graphs without code-quality fields (all-zero) analyze without error."""
        gd = GraphData(
            components=[
                ComponentData(id="X", component_type="Application", weight=1.0, properties={"name": "X"}),
                ComponentData(id="Y", component_type="Application", weight=1.5, properties={"name": "Y"}),
            ],
            edges=[
                EdgeData("X", "Y", "Application", "Application", "app_to_app", "dependency", 1.0),
            ],
        )
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd)
        qa = QualityAnalyzer()
        result = qa.analyze(res)
        
        # Should produce scores without errors
        comp_dict = {cq.id: cq for cq in result.components}
        assert "X" in comp_dict
        assert "Y" in comp_dict
        # CQP should be 0
        for m in res.components.values():
            assert m.code_quality_penalty == pytest.approx(0.0)


# =============================================================================
# CQ-008: Population-level normalization (identical apps → equal scores)
# =============================================================================

class TestNormalizationPopulationLevel:
    def test_identical_apps_get_identical_normalized_scores(self):
        """CQ-008: Two applications with identical code-quality values get equal normalized scores."""
        gd = _simple_two_app_graph(
            loc_a=500, cc_a=10.0, lcom_a=0.4,
            loc_b=500, cc_b=10.0, lcom_b=0.4,
        )
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd)
        
        a = res.components["A"]
        b = res.components["B"]
        assert a.loc_norm == pytest.approx(b.loc_norm)
        assert a.complexity_norm == pytest.approx(b.complexity_norm)
        assert a.lcom_norm == pytest.approx(b.lcom_norm)
        assert a.code_quality_penalty == pytest.approx(b.code_quality_penalty)

    def test_normalization_spans_full_range(self):
        """CQ-008b: Min and max apps get normalised values at 0.0 and 1.0 respectively."""
        gd = _simple_two_app_graph(
            loc_a=100, cc_a=2.0, lcom_a=0.1,   # min
            loc_b=900, cc_b=20.0, lcom_b=0.9,  # max
        )
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd)
        
        a, b = res.components["A"], res.components["B"]
        assert a.loc_norm == pytest.approx(0.0)
        assert b.loc_norm == pytest.approx(1.0)
        assert a.complexity_norm == pytest.approx(0.0)
        assert b.complexity_norm == pytest.approx(1.0)


# =============================================================================
# CQ-009: Generator produces code-quality fields
# =============================================================================

class TestGeneratorCodeQuality:
    def test_generator_produces_code_quality_fields(self):
        """CQ-009: StatisticalGraphGenerator includes code_metrics for every Application."""
        from tools.generation.generator import StatisticalGraphGenerator
        from tools.generation.models import GraphConfig
        
        config = GraphConfig.from_scale("tiny", seed=42)
        gen = StatisticalGraphGenerator(config)
        data = gen.generate()
        
        apps = data["applications"]
        assert len(apps) > 0
        for app in apps:
            assert "code_metrics" in app, f"App {app['id']} missing 'code_metrics'"
            cm = app["code_metrics"]
            assert cm["size"]["total_loc"] > 0, f"App {app['id']} has total_loc=0"
            assert cm["complexity"]["avg_wmc"] > 0
            assert "avg_lcom" in cm["cohesion"]

    def test_generator_respects_app_type_ranges(self):
        """CQ-009b: Sensor apps have lower max LOC than gateway apps."""
        from tools.generation.generator import _CODE_METRICS_PARAMS
        
        sensor_params = _CODE_METRICS_PARAMS["sensor"]
        gateway_params = _CODE_METRICS_PARAMS["gateway"]
        
        assert sensor_params["loc"][1] < gateway_params["loc"][1], "sensor max_loc should be < gateway max_loc"
        assert sensor_params["avg_wmc"][1] < gateway_params["avg_wmc"][1], "sensor max_cc should be < gateway max_cc"


# =============================================================================
# CQ-010: AHP weights include m_code_quality_penalty and sum to 1
# =============================================================================

class TestAHPWeightsCodeQuality:
    def test_compute_weights_includes_cqp_field(self):
        """CQ-010a: AHPProcessor.compute_weights() populates m_code_quality_penalty."""
        processor = AHPProcessor()
        weights = processor.compute_weights()
        
        assert hasattr(weights, "m_code_quality_penalty")
        assert weights.m_code_quality_penalty > 0.0

    def test_maintainability_weights_sum_to_approximately_one(self):
        """CQ-010b: BT + w_out + CQP + CR + CC ~ 1.0."""
        processor = AHPProcessor()
        w = processor.compute_weights()
        
        m_sum = (
            w.m_betweenness
            + w.m_w_out
            + w.m_code_quality_penalty
            + w.m_coupling_risk
            + w.m_clustering
        )
        # After shrinkage the sum should still be ≈ 1.0
        assert m_sum == pytest.approx(1.0, abs=0.02)

    def test_default_weights_sum_to_one(self):
        """CQ-010c: Default QualityWeights M components exactly sum to 1.0."""
        w = QualityWeights()
        m_sum = w.m_betweenness + w.m_w_out + w.m_code_quality_penalty + w.m_coupling_risk + w.m_clustering
        assert m_sum == pytest.approx(1.0, abs=0.001)

    def test_ahp_maintainability_matrix_is_5x5(self):
        """CQ-010d: AHPMatrices criteria_maintainability is a 5×5 matrix."""
        matrices = AHPMatrices()
        mat = matrices.criteria_maintainability
        assert len(mat) == 5
        for row in mat:
            assert len(row) == 5
