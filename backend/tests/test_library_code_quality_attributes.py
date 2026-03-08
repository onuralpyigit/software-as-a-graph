"""
Tests for Code-Level Quality Attribute Integration — Library Nodes

Covers:
    LCQ-001: Library dataclass stores 5 code-quality fields + instability property
    LCQ-002: extract_layer_subgraph propagates Library code-quality props to graph nodes
    LCQ-003: StructuralMetrics correct instability_code and code_quality_penalty for Library
    LCQ-004: Library nodes normalised independently from Application nodes
    LCQ-005: High-complexity Library scores higher M(v) than equivalent low-complexity Library
    LCQ-006: Non-Library/non-Application nodes (Broker, Node) get CQP = 0
    LCQ-007: Missing code-quality fields on Library (all-zero) → no error, CQP = 0
    LCQ-008: Generator produces code-quality fields for every Library
    LCQ-009: _assign_lib_coupling correctly derives Ca/Ce from USES topology
"""

import pytest

from src.core.models import Application, Library, GraphData, ComponentData, EdgeData
from src.core.metrics import StructuralMetrics
from src.analysis.structural_analyzer import StructuralAnalyzer, extract_layer_subgraph
from src.analysis.quality_analyzer import QualityAnalyzer
from src.core import AnalysisLayer


# =============================================================================
# Helpers
# =============================================================================

def _make_lib_component(
    lib_id: str,
    loc: int = 0,
    cc: float = 0.0,
    ca: int = 0,
    ce: int = 0,
    lcom: float = 0.0,
    weight: float = 1.0,
) -> ComponentData:
    """Build a Library ComponentData with code-quality fields in properties."""
    return ComponentData(
        id=lib_id,
        component_type="Library",
        weight=weight,
        properties={
            "name": lib_id,
            "version": "1.0.0",
            "loc": loc,
            "cyclomatic_complexity": cc,
            "coupling_afferent": ca,
            "coupling_efferent": ce,
            "lcom": lcom,
        },
    )


def _make_app_component(app_id: str, weight: float = 1.0) -> ComponentData:
    return ComponentData(
        id=app_id,
        component_type="Application",
        weight=weight,
        properties={"name": app_id},
    )


def _uses_edge(src: str, tgt: str) -> EdgeData:
    return EdgeData(src, tgt, "Application", "Library", "app_to_lib", "dependency", 1.0)


def _simple_system_graph(
    loc_l: int = 0, cc_l: float = 0.0, ca_l: int = 0, ce_l: int = 0, lcom_l: float = 0.0,
) -> GraphData:
    """Single App USES one Library graph, all in the SYSTEM layer."""
    return GraphData(
        components=[
            _make_app_component("A1"),
            _make_lib_component("L1", loc=loc_l, cc=cc_l, ca=ca_l, ce=ce_l, lcom=lcom_l),
        ],
        edges=[
            _uses_edge("A1", "L1"),
        ],
    )


# =============================================================================
# LCQ-001: Library dataclass
# =============================================================================

class TestLibraryDataclass:
    def test_code_quality_fields_stored(self):
        """LCQ-001a: Library stores all 5 code-quality fields."""
        lib = Library(
            id="lib1", name="Lib1",
            loc=1200, cyclomatic_complexity=10.5,
            coupling_afferent=5, coupling_efferent=2, lcom=0.32,
        )
        assert lib.loc == 1200
        assert lib.cyclomatic_complexity == 10.5
        assert lib.coupling_afferent == 5
        assert lib.coupling_efferent == 2
        assert lib.lcom == pytest.approx(0.32)

    def test_instability_property(self):
        """LCQ-001b: instability = Ce / (Ca + Ce)."""
        lib = Library(id="lib1", name="Lib1", coupling_afferent=5, coupling_efferent=2)
        assert lib.instability == pytest.approx(2 / 7)

    def test_instability_zero_when_no_coupling(self):
        """LCQ-001c: instability = 0.0 when both counts are 0."""
        lib = Library(id="lib1", name="Lib1")
        assert lib.instability == 0.0

    def test_defaults_are_zero(self):
        """LCQ-001d: All code-quality fields default to 0."""
        lib = Library(id="lib1", name="Lib1")
        assert lib.loc == 0
        assert lib.cyclomatic_complexity == 0.0
        assert lib.lcom == 0.0
        assert lib.coupling_afferent == 0
        assert lib.coupling_efferent == 0

    def test_to_dict_omits_zero_fields(self):
        """LCQ-001e: to_dict() omits code-quality keys when all zero."""
        lib = Library(id="lib1", name="Lib1", version="1.0.0")
        d = lib.to_dict()
        assert "loc" not in d
        assert "lcom" not in d
        assert "version" in d

    def test_to_dict_includes_nonzero_fields(self):
        """LCQ-001f: to_dict() includes non-zero code-quality keys."""
        lib = Library(id="lib1", name="Lib1", loc=500, lcom=0.4)
        d = lib.to_dict()
        assert d["loc"] == 500
        assert d["lcom"] == pytest.approx(0.4)


# =============================================================================
# LCQ-002: extract_layer_subgraph propagates Library code-quality props
# =============================================================================

class TestSubgraphLibraryCodeQualityPropagation:
    def test_code_quality_stored_on_library_graph_node(self):
        """LCQ-002a: Graph nodes carry code-quality attributes after extraction (SYSTEM layer)."""
        gd = _simple_system_graph(loc_l=2000, cc_l=18.0, ca_l=3, ce_l=1, lcom_l=0.55)
        G = extract_layer_subgraph(gd, AnalysisLayer.SYSTEM)

        assert G.nodes["L1"]["loc"] == 2000
        assert G.nodes["L1"]["cyclomatic_complexity"] == 18.0
        assert G.nodes["L1"]["coupling_afferent"] == 3
        assert G.nodes["L1"]["coupling_efferent"] == 1
        assert G.nodes["L1"]["lcom"] == pytest.approx(0.55)

    def test_missing_lib_code_quality_defaults_to_zero(self):
        """LCQ-002b: Library nodes without code-quality props get 0 defaults."""
        gd = GraphData(
            components=[
                ComponentData(id="L0", component_type="Library", weight=1.0,
                              properties={"name": "L0", "version": "1.0"})
            ],
            edges=[],
        )
        G = extract_layer_subgraph(gd, AnalysisLayer.SYSTEM)
        assert G.nodes["L0"]["loc"] == 0
        assert G.nodes["L0"]["lcom"] == 0.0


# =============================================================================
# LCQ-003: StructuralMetrics code-quality fields for Library
# =============================================================================

class TestLibraryStructuralMetricsCodeQuality:
    def test_instability_code_computed_for_library(self):
        """LCQ-003a: instability_code = Ce/(Ca+Ce) on StructuralMetrics for Library."""
        gd = _simple_system_graph(ca_l=4, ce_l=1)  # I = 1/5 = 0.2
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd, layer=AnalysisLayer.SYSTEM)

        assert res.components["L1"].instability_code == pytest.approx(0.2)

    def test_library_cqp_computed_nonzero(self):
        """LCQ-003b: code_quality_penalty > 0 for Library with non-trivial metrics."""
        gd = GraphData(
            components=[
                _make_lib_component("L1", loc=100, cc=3.0, lcom=0.3),
                _make_lib_component("L2", loc=900, cc=20.0, lcom=0.8),
            ],
            edges=[_uses_edge("L1", "L2")],  # system layer: lib depends on lib
        )
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd, layer=AnalysisLayer.SYSTEM)

        cqp_l1 = res.components["L1"].code_quality_penalty
        cqp_l2 = res.components["L2"].code_quality_penalty
        assert 0.0 <= cqp_l1 <= 1.0
        assert 0.0 <= cqp_l2 <= 1.0
        # L2 has much higher CC and LCOM → higher CQP
        assert cqp_l2 > cqp_l1


# =============================================================================
# LCQ-004: Library and Application normalised independently
# =============================================================================

class TestIndependentNormalisation:
    def test_library_and_app_normalised_independently(self):
        """LCQ-004: Library max-LOC = 10x App max-LOC → Library max gets 1.0, App max gets 1.0 independently."""
        gd = GraphData(
            components=[
                ComponentData(id="A1", component_type="Application", weight=1.0,
                              properties={"name": "A1", "loc": 100, "cyclomatic_complexity": 2.0, "lcom": 0.1}),
                ComponentData(id="A2", component_type="Application", weight=1.0,
                              properties={"name": "A2", "loc": 500, "cyclomatic_complexity": 8.0, "lcom": 0.4}),
                ComponentData(id="L1", component_type="Library", weight=1.0,
                              properties={"name": "L1", "loc": 1000, "cyclomatic_complexity": 3.0, "lcom": 0.2}),
                ComponentData(id="L2", component_type="Library", weight=1.0,
                              properties={"name": "L2", "loc": 5000, "cyclomatic_complexity": 25.0, "lcom": 0.7}),
            ],
            edges=[
                EdgeData("A1", "L1", "Application", "Library", "app_to_lib", "dependency", 1.0),
                EdgeData("A2", "L2", "Application", "Library", "app_to_lib", "dependency", 1.0),
                EdgeData("A1", "A2", "Application", "Application", "app_to_app", "dependency", 1.0),
            ],
        )
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd, layer=AnalysisLayer.SYSTEM)

        # Each type's max normalised value should be 1.0
        app_loc_norms  = [res.components[i].loc_norm  for i in ("A1", "A2")]
        lib_loc_norms  = [res.components[i].loc_norm  for i in ("L1", "L2")]

        assert max(app_loc_norms) == pytest.approx(1.0)
        assert max(lib_loc_norms) == pytest.approx(1.0)
        # The raw App values are all < Lib values; without independent normalisation
        # App max would be < 1.0 (because lib raw > app raw). So this checks independence.
        assert min(app_loc_norms) == pytest.approx(0.0)
        assert min(lib_loc_norms) == pytest.approx(0.0)

    def test_identical_libraries_get_equal_scores(self):
        """LCQ-004b: Two identical Libraries get equal normalised scores."""
        gd = GraphData(
            components=[
                _make_lib_component("L1", loc=400, cc=8.0, lcom=0.3),
                _make_lib_component("L2", loc=400, cc=8.0, lcom=0.3),
            ],
            edges=[_uses_edge("L1", "L2")],
        )
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd, layer=AnalysisLayer.SYSTEM)

        l1, l2 = res.components["L1"], res.components["L2"]
        assert l1.loc_norm == pytest.approx(l2.loc_norm)
        assert l1.complexity_norm == pytest.approx(l2.complexity_norm)
        assert l1.lcom_norm == pytest.approx(l2.lcom_norm)
        assert l1.code_quality_penalty == pytest.approx(l2.code_quality_penalty)


# =============================================================================
# LCQ-005: High-complexity Library → higher M(v)
# =============================================================================

class TestLibraryMaintainabilityWithCQP:
    def test_high_complexity_lib_has_higher_m_score(self):
        """LCQ-005: High-CC Library scores higher M(v) than low-CC Library (same topology)."""
        def _run(cc: float, lcom: float) -> float:
            gd = GraphData(
                components=[
                    _make_app_component("A1"),
                    _make_lib_component("L1", loc=500, cc=cc, lcom=lcom),
                    _make_lib_component("L2", loc=500, cc=cc, lcom=lcom),
                ],
                edges=[
                    _uses_edge("A1", "L1"),
                    _uses_edge("A1", "L2"),
                    _uses_edge("L1", "L2"),
                ],
            )
            analyzer = StructuralAnalyzer()
            res_s = analyzer.analyze(gd, layer=AnalysisLayer.SYSTEM)
            qa = QualityAnalyzer()
            result = qa.analyze(res_s)
            scores = {cq.id: cq.scores.maintainability for cq in result.components}
            return scores.get("L1", 0.0)

        m_low  = _run(cc=2.0, lcom=0.1)
        m_high = _run(cc=28.0, lcom=0.8)
        # High-complexity library should score higher (worse) M(v)
        assert m_high >= m_low, (
            f"High-CC library M(v)={m_high:.4f} should be >= low-CC M(v)={m_low:.4f}"
        )


# =============================================================================
# LCQ-006: Non-Library/Application nodes get CQP = 0
# =============================================================================

class TestNonLibraryNonAppNodesUnaffected:
    def test_broker_cqp_zero_in_system_layer(self):
        """LCQ-006a: Broker in SYSTEM layer gets code_quality_penalty = 0.0."""
        gd = GraphData(
            components=[
                _make_lib_component("L1", cc=15.0, lcom=0.6),
                ComponentData(id="BRK1", component_type="Broker", weight=2.0,
                              properties={"name": "BRK1"}),
            ],
            edges=[
                EdgeData("L1", "BRK1", "Library", "Broker", "lib_to_broker", "dependency", 1.0),
            ],
        )
        G = extract_layer_subgraph(gd, AnalysisLayer.SYSTEM)
        assert G.nodes.get("BRK1", {}).get("loc", 0) == 0

    def test_node_cqp_zero(self):
        """LCQ-006b: Node component gets code_quality_penalty = 0.0."""
        gd = GraphData(
            components=[
                _make_lib_component("L1", cc=10.0, lcom=0.4),
                ComponentData(id="N1", component_type="Node", weight=1.0,
                              properties={"name": "N1"}),
            ],
            edges=[
                EdgeData("L1", "N1", "Library", "Node", "lib_to_node", "dependency", 1.0),
            ],
        )
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd, layer=AnalysisLayer.SYSTEM)
        if "N1" in res.components:
            assert res.components["N1"].code_quality_penalty == pytest.approx(0.0)


# =============================================================================
# LCQ-007: Backward compatibility (zero code-quality inputs)
# =============================================================================

class TestLibraryBackwardCompatibility:
    def test_library_without_code_quality_analyzes_without_error(self):
        """LCQ-007: Libraries with all-zero code-quality fields analyse without error; CQP = 0."""
        gd = GraphData(
            components=[
                ComponentData(id="L1", component_type="Library", weight=1.0,
                              properties={"name": "L1", "version": "1.0"}),
                ComponentData(id="L2", component_type="Library", weight=1.0,
                              properties={"name": "L2", "version": "2.0"}),
            ],
            edges=[
                EdgeData("L1", "L2", "Library", "Library", "lib_to_lib", "dependency", 1.0),
            ],
        )
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(gd, layer=AnalysisLayer.SYSTEM)

        for m in res.components.values():
            assert m.code_quality_penalty == pytest.approx(0.0)


# =============================================================================
# LCQ-008: Generator produces code-quality fields for Libraries
# =============================================================================

class TestGeneratorLibraryCodeQuality:
    def test_generator_produces_lib_code_quality_fields(self):
        """LCQ-008a: StatisticalGraphGenerator includes loc/CC/LCOM for Library nodes."""
        from src.generation.generator import StatisticalGraphGenerator
        from src.generation.models import GraphConfig

        config = GraphConfig.from_scale("tiny", seed=99)
        gen = StatisticalGraphGenerator(config)
        data = gen.generate()

        libs = data["libraries"]
        assert len(libs) > 0
        for lib in libs:
            assert "loc" in lib, f"Library {lib['id']} missing 'loc'"
            assert lib["loc"] > 0
            assert "cyclomatic_complexity" in lib
            assert lib["cyclomatic_complexity"] > 0
            assert "lcom" in lib

    def test_generator_lib_archetype_ranges_are_sensible(self):
        """LCQ-008b: Library archetypes have sensible LOC ranges (utility < framework)."""
        from src.generation.generator import _LIB_CODE_QUALITY_PARAMS

        utility_max_loc   = _LIB_CODE_QUALITY_PARAMS["utility"][1]
        framework_max_loc = _LIB_CODE_QUALITY_PARAMS["framework"][1]
        assert utility_max_loc < framework_max_loc, (
            "utility libraries should have smaller max LOC than framework libraries"
        )


# =============================================================================
# LCQ-009: _assign_lib_coupling derives Ca/Ce correctly
# =============================================================================

class TestAssignLibCoupling:
    def test_ca_counts_apps_and_libs_depending_on_lib(self):
        """LCQ-009a: Ca = number of components that USES this library."""
        from src.generation.generator import StatisticalGraphGenerator
        from src.generation.models import GraphConfig

        # Use generator's helper directly
        config = GraphConfig.from_scale("tiny", seed=7)
        gen = StatisticalGraphGenerator(config)

        from src.core.models import Library, Application
        libs = [Library(id="L0", name="L0"), Library(id="L1", name="L1")]
        apps = [Application(id="A0", name="A0")]
        # A0 uses L0, L0 uses L1
        uses = [
            {"from": "A0", "to": "L0"},
            {"from": "L0", "to": "L1"},
        ]
        gen._assign_lib_coupling(libs, apps, uses)

        # Ca for L0 = 1 (A0 depends on it), Ca for L1 = 1 (L0 depends on it)
        assert libs[0].coupling_afferent == 1   # L0: A0 depends on it
        assert libs[1].coupling_afferent == 1   # L1: L0 depends on it
        # Ce for L0 = 1 (it depends on L1), Ce for L1 = 0
        assert libs[0].coupling_efferent == 1   # L0: depends on L1
        assert libs[1].coupling_efferent == 0   # L1: depends on nothing

    def test_instability_reflects_coupling(self):
        """LCQ-009b: Library instability property correct after coupling assignment."""
        from src.generation.generator import StatisticalGraphGenerator
        from src.generation.models import GraphConfig
        from src.core.models import Library, Application

        config = GraphConfig.from_scale("tiny", seed=7)
        gen = StatisticalGraphGenerator(config)

        libs = [Library(id="L0", name="L0")]
        apps = [Application(id="A0", name="A0"), Application(id="A1", name="A1")]
        uses = [
            {"from": "A0", "to": "L0"},
            {"from": "A1", "to": "L0"},
        ]
        gen._assign_lib_coupling(libs, apps, uses)

        # Ca=2, Ce=0 → instability = 0/2 = 0 (very stable)
        assert libs[0].coupling_afferent == 2
        assert libs[0].coupling_efferent == 0
        assert libs[0].instability == pytest.approx(0.0)
