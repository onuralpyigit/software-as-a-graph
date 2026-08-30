"""
Tests for Step 7: Visualization

Validates the visualization pipeline components:
    - LayerData model (Definition 9: input aggregation)
    - LayerDataCollector (data collection from Steps 1-5)
    - ChartGenerator (visual encoding functions from §6.4)
    - VisualizationService (dashboard assembly from §6.5)

Test categories:
    - Unit tests for LayerData properties and ComponentDetail
    - Unit tests for ChartGenerator output validity
    - Integration tests for data collection pipeline
    - Integration tests for full dashboard generation
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

from saag.visualization import LayerData, ComponentDetail
from saag.visualization.charts import ChartGenerator
from saag.visualization.palette import CRITICALITY_COLORS, RM_COLORS
from saag.visualization.dashboard import DashboardGenerator
from saag.visualization.collector import LayerDataCollector
from saag.visualization import VisualizationService
from saag.simulation.models import SimulationReport, LayerMetrics, ComponentCriticality


# =========================================================================
# LayerData Model Tests (Definition 9)
# =========================================================================

class TestLayerData:
    """Tests for LayerData model and computed properties."""

    def test_basic_creation(self):
        """LayerData initializes with correct defaults."""
        data = LayerData(layer="system", name="Complete System")
        assert data.layer == "system"
        assert data.nodes == 0
        assert data.critical_count == 0
        assert data.spearman == 0.0
        assert data.scatter_data == []
        assert data.component_details == []

    def test_classification_distribution(self):
        """classification_distribution returns correct dict."""
        data = LayerData(layer="app", name="App")
        data.critical_count = 3
        data.high_count = 5
        data.medium_count = 10
        data.low_count = 7
        data.minimal_count = 2

        dist = data.classification_distribution
        assert dist == {
            "CRITICAL": 3, "HIGH": 5, "MEDIUM": 10,
            "LOW": 7, "MINIMAL": 2,
        }

    def test_total_classified(self):
        """total_classified sums all classification counts."""
        data = LayerData(layer="app", name="App")
        data.critical_count = 2
        data.high_count = 3
        data.medium_count = 5
        data.low_count = 4
        data.minimal_count = 1
        assert data.total_classified == 15

    def test_has_validation_true(self):
        """has_validation is True when spearman > 0."""
        data = LayerData(layer="app", name="App")
        data.spearman = 0.85
        assert data.has_validation

    def test_has_validation_false(self):
        """has_validation is False when spearman is 0."""
        data = LayerData(layer="app", name="App")
        assert not data.has_validation

    def test_has_simulation_with_throughput(self):
        """event_throughput alone does NOT enable has_simulation; only max_impact does."""
        data = LayerData(layer="app", name="App")
        data.event_throughput = 100
        assert not data.has_simulation

    def test_has_simulation_with_impact(self):
        """has_simulation is True when max_impact > 0."""
        data = LayerData(layer="app", name="App")
        data.max_impact = 0.5
        assert data.has_simulation

    def test_has_simulation_false(self):
        """has_simulation is False when no simulation data."""
        data = LayerData(layer="app", name="App")
        assert not data.has_simulation


class TestComponentDetail:
    """Tests for ComponentDetail data class."""

    def test_creation(self):
        """ComponentDetail initializes correctly."""
        detail = ComponentDetail(
            id="sensor_fusion",
            name="Sensor Fusion",
            type="Application",
            reliability=0.82,
            maintainability=0.88,
            fault_tolerance=0.76,
            availability=0.90,
            overall=0.84,
            level="CRITICAL",
            impact=0.79,
            cascade_depth=3,
        )
        assert detail.id == "sensor_fusion"
        assert detail.overall == 0.84
        assert detail.impact == 0.79

    def test_to_dict(self):
        """to_dict returns complete dictionary."""
        detail = ComponentDetail(
            id="broker_1", name="Main Broker", type="Broker",
            overall=0.80, level="CRITICAL",
        )
        d = detail.to_dict()
        assert d["id"] == "broker_1"
        assert d["type"] == "Broker"
        assert d["overall"] == 0.80
        assert d["level"] == "CRITICAL"
        assert "reliability" in d
        assert "impact" in d


# =========================================================================
# ChartGenerator Tests (§6.4 Visualization Taxonomy)
# =========================================================================

class TestChartGenerator:
    """Tests for chart generation functions."""

    def setup_method(self):
        self.charts = ChartGenerator()

    def test_criticality_distribution_generates_html(self):
        """§6.4.2: criticality_distribution returns valid HTML with canvas."""
        counts = {"CRITICAL": 5, "HIGH": 8, "MEDIUM": 15, "LOW": 12, "MINIMAL": 8}
        html = self.charts.criticality_distribution(counts)
        assert html is not None
        assert "<canvas" in html
        assert "chart-container" in html

    def test_criticality_distribution_skips_zeros(self):
        """Empty distributions return None."""
        html = self.charts.criticality_distribution(
            {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "MINIMAL": 0}
        )
        assert html is None

    def test_rm_breakdown_generates_html(self):
        """§6.4.3: rm_breakdown returns stacked bar chart."""
        components = [
            ComponentDetail("a", "App A", "Application",
                           0.8, 0.7, 0.9, 0.6, 0.75, "CRITICAL"),
            ComponentDetail("b", "App B", "Application",
                           0.5, 0.6, 0.4, 0.3, 0.45, "MEDIUM"),
        ]
        html = self.charts.rm_breakdown(components)
        assert html is not None
        assert "<canvas" in html
        assert "Fault Tolerance" in html
        assert "Maintainability" in html

    def test_rm_breakdown_empty(self):
        """rm_breakdown returns None for empty components."""
        assert self.charts.rm_breakdown([]) is None

    def test_correlation_scatter_generates_html(self):
        """§6.4.4: correlation_scatter returns scatter plot."""
        scatter_data = [
            ("a", 0.84, 0.79, "CRITICAL"),
            ("b", 0.80, 0.73, "CRITICAL"),
            ("c", 0.64, 0.58, "HIGH"),
            ("d", 0.45, 0.40, "MEDIUM"),
        ]
        html = self.charts.correlation_scatter(scatter_data, spearman=0.876)
        assert html is not None
        assert "<canvas" in html
        assert "0.876" in html  # Spearman value in subtitle

    def test_correlation_scatter_needs_min_points(self):
        """scatter plot requires at least 3 points."""
        data = [("a", 0.5, 0.4, "HIGH"), ("b", 0.3, 0.2, "LOW")]
        assert self.charts.correlation_scatter(data) is None

    def test_correlation_scatter_diagonal_plugin(self):
        """scatter plot includes diagonal reference line plugin."""
        data = [
            ("a", 0.8, 0.7, "HIGH"),
            ("b", 0.6, 0.5, "MEDIUM"),
            ("c", 0.4, 0.3, "LOW"),
        ]
        html = self.charts.correlation_scatter(data)
        assert "diagonalLine" in html

    def test_grouped_bar_chart(self):
        """grouped_bar_chart generates comparison chart."""
        data = {
            "App Layer": {"Critical": 3, "High": 5},
            "Infra Layer": {"Critical": 1, "High": 2},
        }
        html = self.charts.grouped_bar_chart(data, "Test Chart")
        assert html is not None
        assert "<canvas" in html

    def test_cascade_risk_chart_omits_baseline_when_unmeasured(self):
        """When no component carries a measured cascade_risk_topo, the chart
        must render the QoS-enriched series alone rather than fabricating the
        missing baseline as 88% of the enriched score."""
        components = [
            {"id": "a", "name": "A", "cascade_risk": 0.81},
            {"id": "b", "name": "B", "cascade_risk": 0.68},
        ]
        html = self.charts.cascade_risk_chart(components)
        assert html is not None
        assert "Topology-only baseline" not in html
        assert "QoS-enriched" in html

    def test_cascade_risk_chart_shows_measured_baseline(self):
        """A real cascade_risk_topo measurement is rendered as-is, unscaled."""
        components = [
            {"id": "a", "name": "A", "cascade_risk": 0.81, "cascade_risk_topo": 0.71},
        ]
        html = self.charts.cascade_risk_chart(components)
        assert "Topology-only baseline" in html
        assert "0.71" in html

    def test_unique_chart_ids(self):
        """Each chart gets a unique ID."""
        data = {"CRITICAL": 5, "HIGH": 3}
        html1 = self.charts.criticality_distribution(data, "Chart 1")
        html2 = self.charts.criticality_distribution(data, "Chart 2")
        # Extract canvas IDs
        import re
        ids1 = re.findall(r'id="(crit_dist_\d+)"', html1)
        ids2 = re.findall(r'id="(crit_dist_\d+)"', html2)
        assert ids1[0] != ids2[0]


# =========================================================================
# LayerDataCollector Tests
# =========================================================================

@pytest.fixture
def mock_analysis_service():
    """Create a mock analysis service with realistic return data."""
    service = MagicMock()

    # Build mock component
    mock_comp = MagicMock()
    mock_comp.id = "sensor_fusion"
    mock_comp.type = "Application"
    mock_comp.structural.name = "Sensor Fusion"
    mock_comp.scores.overall = 0.84
    mock_comp.scores.reliability = 0.82
    mock_comp.scores.maintainability = 0.88
    mock_comp.scores.availability = 0.90
    mock_comp.scores.fault_tolerance = 0.76
    mock_comp.levels.overall.name = "CRITICAL"

    # Build mock analysis result
    mock_result = MagicMock()
    mock_result.structural.graph_summary.nodes = 48
    mock_result.structural.graph_summary.edges = 127
    mock_result.structural.graph_summary.density = 0.056
    mock_result.structural.graph_summary.num_components = 1
    mock_result.structural.graph_summary.node_types = {"Application": 25, "Broker": 5}
    mock_result.structural.graph_summary.num_articulation_points = 3
    mock_result.quality.components = [mock_comp]
    mock_result.quality.edges = []
    mock_result.problems = []

    service.analyze_layer.return_value = mock_result
    return service


@pytest.fixture
def mock_simulation_service():
    """Mocks generate_report(), not analyze_layer(): the collector calls
    generate_report() directly so it can read the per-component breakdown
    (ComponentCriticality) that analyze_layer() throws away. Real dataclasses
    rather than further MagicMocks so `.get()` / iteration behave correctly.
    """
    service = MagicMock()
    layer_metrics = LayerMetrics(
        layer="system",
        event_throughput=1000,
        event_delivery_rate=98.5,
        avg_reachability_loss=0.15,
        max_impact=0.734,
    )
    component_criticality = [
        ComponentCriticality(
            id="sensor_fusion", type="Application",
            combined_impact=0.734, cascade_depth=2,
        )
    ]
    service.generate_report.return_value = SimulationReport(
        timestamp="2024-01-01T00:00:00",
        graph_summary={},
        layer_metrics={"system": layer_metrics},
        component_criticality=component_criticality,
    )
    return service


@pytest.fixture
def mock_validation_service():
    service = MagicMock()
    mock_val = MagicMock()
    mock_val.spearman = 0.876
    mock_val.f1_score = 0.923
    mock_val.precision = 0.912
    mock_val.recall = 0.857
    mock_val.passed = True
    mock_val.top5_overlap = 0.80
    mock_val.top_5_overlap = 0.80
    mock_val.top10_overlap = 0.70
    mock_val.top_10_overlap = 0.70

    mock_result = MagicMock()
    mock_result.layers = {"system": mock_val}
    service.validate_layers.return_value = mock_result
    return service


@pytest.fixture
def mock_prediction_service():
    """Create a mock prediction service."""
    service = MagicMock()
    
    # Build mock component
    mock_comp = MagicMock()
    mock_comp.id = "sensor_fusion"
    mock_comp.type = "Application"
    mock_comp.structural.name = "Sensor Fusion"
    mock_comp.scores.overall = 0.84
    mock_comp.scores.reliability = 0.82
    mock_comp.scores.maintainability = 0.88
    mock_comp.scores.availability = 0.90
    mock_comp.scores.fault_tolerance = 0.76
    mock_comp.levels.overall.name = "CRITICAL"
    mock_comp.levels.overall.__str__.return_value = "CRITICAL"
    
    mock_result = MagicMock()
    mock_result.components = [mock_comp]
    mock_result.problems = [MagicMock(), MagicMock()]
    mock_result.node_scores = {}
    mock_result.gnn_metrics = {}
    
    service.predict_quality.return_value = mock_result
    service.predict_quality_with_gnn.return_value = mock_result
    return service



class TestLayerDataCollector:
    """Tests for data collection pipeline."""

    def test_collect_basic_data(
        self, mock_analysis_service, mock_prediction_service, mock_simulation_service, mock_validation_service
    ):
        """Collector populates basic layer data from all services."""
        repository = MagicMock()
        collector = LayerDataCollector(
            mock_analysis_service,
            mock_prediction_service,
            mock_simulation_service,
            mock_validation_service,
            repository=repository,
        )

        data = collector.collect_layer_data("system", include_validation=True)

        assert data.layer == "system"
        assert data.nodes == 48
        assert data.edges == 127
        assert data.critical_count == 1
        assert data.spof_count == 3
        assert data.event_throughput == 1000
        assert data.spearman == 0.876

    def test_collect_component_details(
        self, mock_analysis_service, mock_prediction_service, mock_simulation_service, mock_validation_service
    ):
        """Collector builds ComponentDetail list with RM scores."""
        repository = MagicMock()
        collector = LayerDataCollector(
            mock_analysis_service,
            mock_prediction_service,
            mock_simulation_service,
            mock_validation_service,
            repository=repository,
        )

        data = collector.collect_layer_data("system")

        assert len(data.component_details) == 1
        detail = data.component_details[0]
        assert detail.id == "sensor_fusion"
        assert detail.reliability == pytest.approx(0.82)
        assert detail.maintainability == pytest.approx(0.88)
        assert detail.availability == pytest.approx(0.90)
        assert detail.fault_tolerance == pytest.approx(0.76)
        assert detail.overall == pytest.approx(0.84)
        assert detail.level == "CRITICAL"

    def test_collect_scatter_data(
        self, mock_analysis_service, mock_prediction_service, mock_simulation_service, mock_validation_service
    ):
        """Collector builds scatter plot data from Q(v) and I(v)."""
        repository = MagicMock()
        collector = LayerDataCollector(
            mock_analysis_service,
            mock_prediction_service,
            mock_simulation_service,
            mock_validation_service,
            repository=repository,
        )

        data = collector.collect_layer_data("system")

        # Should have scatter data for each component with Q > 0
        assert len(data.scatter_data) >= 1
        comp_id, q_val, i_val, level = data.scatter_data[0]
        assert comp_id == "sensor_fusion"
        assert q_val == pytest.approx(0.84)
        # I(v) sourced from ComponentCriticality.combined_impact, not the
        # 0.0 placeholder scatter_data starts with (see
        # test_collect_simulation_impact_reaches_component_details below —
        # this used to stay 0.0 because analyze_layer()'s LayerMetrics has no
        # per-component breakdown to read it from).
        assert i_val == pytest.approx(0.734)

    def test_collect_simulation_impact_reaches_component_details(
        self, mock_analysis_service, mock_prediction_service, mock_simulation_service, mock_validation_service
    ):
        """ComponentDetail.impact/cascade_depth must be populated from the
        simulation's per-component ComponentCriticality breakdown, not stay at
        their 0.0/0 defaults."""
        repository = MagicMock()
        collector = LayerDataCollector(
            mock_analysis_service,
            mock_prediction_service,
            mock_simulation_service,
            mock_validation_service,
            repository=repository,
        )

        data = collector.collect_layer_data("system", include_validation=False)

        detail = next(d for d in data.component_details if d.id == "sensor_fusion")
        assert detail.impact == pytest.approx(0.734)
        assert detail.cascade_depth == 2

    def test_collect_top_k_overlap(
        self, mock_analysis_service, mock_prediction_service, mock_simulation_service, mock_validation_service
    ):
        """Collector captures Top-K overlap from validation."""
        repository = MagicMock()
        collector = LayerDataCollector(
            mock_analysis_service,
            mock_prediction_service,
            mock_simulation_service,
            mock_validation_service,
            repository=repository,
        )

        data = collector.collect_layer_data("system", include_validation=True)

        assert data.top5_overlap == pytest.approx(0.80)
        assert data.top10_overlap == pytest.approx(0.70)

    def test_collect_unknown_layer_raises(
        self, mock_analysis_service, mock_prediction_service, mock_simulation_service, mock_validation_service
    ):
        """Unknown layer raises ValueError."""
        repository = MagicMock()
        collector = LayerDataCollector(
            mock_analysis_service,
            mock_prediction_service,
            mock_simulation_service,
            mock_validation_service,
            repository=repository,
        )

        with pytest.raises(ValueError, match="Unknown layer"):
            collector.collect_layer_data("nonexistent")

    def test_collect_handles_analysis_failure(
        self, mock_prediction_service, mock_simulation_service, mock_validation_service
    ):
        """Collector gracefully handles analysis service failure."""
        bad_analysis = MagicMock()
        bad_analysis.analyze_layer.side_effect = Exception("Neo4j down")

        repository = MagicMock()
        collector = LayerDataCollector(
            bad_analysis,
            mock_prediction_service,
            mock_simulation_service,
            mock_validation_service,
            repository=repository,
        )

        # Should not raise
        data = collector.collect_layer_data("system")
        assert data.nodes == 0  # Defaults preserved

    def test_collect_frequency_weighted_edges(self, mock_prediction_service, mock_simulation_service, mock_validation_service):
        """Collector should weight edges based on topic and app aggregate frequencies."""
        import networkx as nx
        from saag.analysis.models import LayerAnalysisResult
        
        # Build a NetworkX graph with an App, Topic, and Library
        g = nx.DiGraph()
        g.add_node("A01", type="Application")
        g.add_node("T01", type="Topic", frequency=50.0)
        g.add_node("L01", type="Library")
        g.add_edge("A01", "T01", type="PUBLISHES_TO")
        g.add_edge("A01", "L01", type="USES")

        mock_analysis = MagicMock()
        mock_analysis.graph = g
        mock_analysis.quality.components = []
        
        # Create a mock edge
        mock_edge = MagicMock()
        mock_edge.source = "A01"
        mock_edge.target = "T01"
        mock_edge.weight = 1.0
        mock_edge.dependency_type = "PUBLISHES_TO"
        
        mock_uses_edge = MagicMock()
        mock_uses_edge.source = "A01"
        mock_uses_edge.target = "L01"
        mock_uses_edge.weight = 1.0
        mock_uses_edge.dependency_type = "USES"

        mock_analysis.quality.edges = [mock_edge, mock_uses_edge]

        repository = MagicMock()
        collector = LayerDataCollector(
            MagicMock(),
            mock_prediction_service,
            mock_simulation_service,
            mock_validation_service,
            repository=repository,
        )

        data = LayerData(layer="system", name="System")
        collector._build_network_data(data, mock_analysis)

        # Check weights are populated by dynamic edge frequency
        edge_map = {f"{e['source']}->{e['target']}": e["weight"] for e in data.network_edges}
        
        # A01->T01 should have frequency of T01 (50.0)
        assert edge_map["A01->T01"] == 50.0
        # A01->L01 is a USES edge. A01 connects to T01 (50.0 Hz), so aggregate is 50.0
        assert edge_map["A01->L01"] == 50.0



# =========================================================================
# VisualizationService Integration Tests
# =========================================================================

class TestVisualizationService:
    """Tests for full dashboard generation pipeline."""

    def test_generate_dashboard_creates_file(
        self, mock_analysis_service, mock_prediction_service, mock_simulation_service, mock_validation_service, tmp_path
    ):
        """generate_dashboard produces an HTML file."""
        repository = MagicMock()
        service = VisualizationService(
            mock_analysis_service,
            mock_prediction_service,
            mock_simulation_service,
            mock_validation_service,
            repository=repository,
        )

        output = tmp_path / "test_dashboard.html"

        with patch.object(service, "collector") as mock_collector:
            mock_data = LayerData(layer="system", name="Complete System")
            mock_data.nodes = 48
            mock_data.critical_count = 5
            mock_data.spearman = 0.876
            mock_data.component_details = [
                ComponentDetail("a", "App A", "Application",
                               0.8, 0.7, 0.9, 0.6, 0.75, "CRITICAL"),
            ]
            mock_collector.collect_layer_data.return_value = mock_data

            with patch(
                "saag.visualization.dashboard.DashboardGenerator"
            ) as MockDash:
                dash_instance = MockDash.return_value
                dash_instance.generate.return_value = "<html>Dashboard</html>"

                result = service.generate_dashboard(
                    output_file=str(output),
                    layers=["system"],
                )

                assert str(output) in result
                assert output.exists()

    def test_scalability_auto_disable_network(
        self, tmp_path, mock_analysis_service, mock_prediction_service, mock_simulation_service, mock_validation_service
    ):
        """Large systems auto-disable network graph (> 500 nodes)."""
        repository = MagicMock()
        service = VisualizationService(
            mock_analysis_service,
            mock_prediction_service,
            mock_simulation_service,
            mock_validation_service,
            repository=repository,
        )

        with patch.object(service, "collector") as mock_collector:
            mock_data = LayerData(layer="system", name="Complete System")
            mock_data.nodes = 600  # > 500 threshold in service.py
            mock_collector.collect_layer_data.return_value = mock_data

            with patch.object(service, "_add_network_section") as mock_add:
                with patch("saag.visualization.dashboard.DashboardGenerator") as MockDash:
                    dash_instance = MockDash.return_value
                    dash_instance.generate.return_value = "<html></html>"

                    service.generate_dashboard(
                        output_file=str(tmp_path / "test.html"),
                        layers=["system"],
                        include_network=True,
                    )

                    # _add_network_section should NOT be called
                    assert mock_add.called is False

    def test_unknown_layer_skipped(
        self, tmp_path, mock_analysis_service, mock_prediction_service, mock_simulation_service, mock_validation_service
    ):
        """Empty layers raise ValueError (at least one valid layer required)."""
        repository = MagicMock()
        service = VisualizationService(
            mock_analysis_service,
            mock_prediction_service,
            mock_simulation_service,
            mock_validation_service,
            repository=repository,
        )

        with patch("saag.visualization.dashboard.DashboardGenerator"):
            with pytest.raises(ValueError, match="No layer data collected"):
                service.generate_dashboard(
                    output_file=str(tmp_path / "test.html"),
                    layers=["nonexistent_layer"],
                )


# =========================================================================
# Color Constant Tests (§6.6)
# =========================================================================

class TestColorConstants:
    """Verify color encoding specification from §6.6."""

    def test_all_criticality_levels_have_colors(self):
        """Every criticality level has a defined color."""
        levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]
        for level in levels:
            assert level in CRITICALITY_COLORS
            assert CRITICALITY_COLORS[level].startswith("#")

    def test_all_rm_dimensions_have_colors(self):
        """Every RM dimension has a defined color."""
        dims = ["reliability", "maintainability", "fault_tolerance", "availability"]
        for dim in dims:
            assert dim in RM_COLORS
            assert RM_COLORS[dim].startswith("#")

    def test_palette_is_the_single_source_of_criticality_colors(self):
        """Badge colours are generated from the same palette as chart colours."""
        from saag.visualization.palette import (
            CRITICALITY_BADGE_COLORS, criticality_badge_css,
        )
        assert set(CRITICALITY_BADGE_COLORS) == set(CRITICALITY_COLORS)
        css = criticality_badge_css()
        for level in CRITICALITY_COLORS:
            assert f".badge-{level.lower()}" in css


class TestLayerNaming:
    """Layer names resolve through the canonical saag.core definitions."""

    def test_layer_display_names_come_from_core(self):
        from saag.core.layers import AnalysisLayer, LAYER_DEFINITIONS
        for layer in ("app", "infra", "mw", "system"):
            assert LAYER_DEFINITIONS[AnalysisLayer.from_string(layer)].name

    def test_visualization_does_not_redefine_layers(self):
        """CLAUDE.md invariant: layers are defined once, in saag.core.layers."""
        import saag.visualization.models as viz_models
        assert not hasattr(viz_models, "LAYER_DEFINITIONS")


class TestDashboardGeneration:
    """Tests for DashboardGenerator class."""
    
    def test_dashboard_basic_structure(self):
        """Dashboard should generate valid HTML structure."""
        dash = DashboardGenerator("Test Dash")
        
        dash.start_section("Section 1", "sec1")
        dash.add_kpis({"Metric 1": 100})
        dash.add_table(["Col1", "Col2"], [[1, "A"], [2, "B"]])
        dash.end_section()
        
        html = dash.generate()
        
        assert "<!DOCTYPE html>" in html
        assert "Test Dash" in html
        assert "Metric 1" in html
        assert "100" in html
        assert "<td>A</td>" in html


# =========================================================================
# build_html(): the pure assembly stage shared by the CLI demo and the
# production pipeline.
# =========================================================================

class TestBuildHtml:
    """Tests for VisualizationService.build_html()."""

    @staticmethod
    def _service() -> VisualizationService:
        # build_html() is pure, so unwired backends are sufficient.
        return VisualizationService(None, None, None, None, None)

    def test_demo_fixture_renders_all_six_tabs(self):
        """--demo drives the production assembler, not a parallel one."""
        from cli.visualize_graph import _demo_layer_data

        html = self._service().build_html([_demo_layer_data()])

        for tab_id in ("overview", "components", "validation",
                       "cascade", "topology", "hierarchy"):
            assert f'id="tab-{tab_id}"' in html

    def test_widget_css_classes_are_defined(self):
        """
        Regression guard: the RM segmented bar and the per-dimension ρ bars
        emit these classes, and previously nothing defined them.
        """
        from cli.visualize_graph import _demo_layer_data

        html = self._service().build_html([_demo_layer_data()])

        for css_class in (".rm-bar", ".rm-seg", ".dim-row",
                          ".dim-bar-outer", ".dim-bar-inner", ".dim-val"):
            assert css_class in html, f"{css_class} used but never styled"

    def test_rm_segments_carry_a_single_style_attribute(self):
        """A duplicate style= made the browser drop every segment width."""
        import re
        from cli.visualize_graph import _demo_layer_data

        html = self._service().build_html([_demo_layer_data()])

        segments = re.findall(r'<div class="rm-seg"[^>]*>', html)
        assert segments
        for seg in segments:
            assert seg.count("style=") == 1
            assert "width:" in seg

    def test_build_html_touches_no_filesystem(self, tmp_path):
        """build_html() is pure — generate_dashboard() owns the write."""
        from cli.visualize_graph import _demo_layer_data

        before = set(tmp_path.iterdir())
        self._service().build_html([_demo_layer_data()])
        assert set(tmp_path.iterdir()) == before

    def test_empty_layer_list_rejected(self):
        with pytest.raises(ValueError):
            self._service().build_html([])

    def test_validation_gate_labels_match_validation_targets(self):
        """The gate labels the dashboard renders must match the thresholds
        the gates are actually evaluated against (saag.validation.models.
        ValidationTargets), and G1-G6/G8 must all render. Previously G2/G3's
        labels were hardcoded to different numbers ("F1-score > 0.6",
        "Top-K precision > 0.5") than the real targets (0.75, 0.80), and
        G5/G6/G8 never appeared at all. G7 (CDCC) and G9 (FTR) were retired
        with the Vulnerability/Security dimension — see _GATE_SPECS."""
        from cli.visualize_graph import _demo_layer_data
        from saag.validation.models import ValidationTargets

        data = _demo_layer_data()
        data.gates = {
            "G1_spearman": True, "G2_f1": True, "G3_precision": True,
            "G4_top5": True, "G5_predictive_gain": False,
            "G6_kappa_cta": True,
            "G8_bottleneck_precision": True,
        }

        html = self._service().build_html([data])
        targets = ValidationTargets()

        assert f"G2: F1-score ≥ {targets.f1_score:.2f}" in html
        assert f"G3: Top-K precision ≥ {targets.precision:.2f}" in html
        assert "G7:" not in html
        assert "G9:" not in html

    def test_demo_fixture_renders_triage_panel(self):
        """Demo fixture should render the Triage bridge panel and role badges."""
        from cli.visualize_graph import _demo_layer_data

        html = self._service().build_html([_demo_layer_data()])
        assert "Triage Bridge — Actionable Stakeholder Remediation" in html
        assert "sensor_fusion" in html
        assert "GOD_COMPONENT" in html
        assert "badge-architect" in html
        assert "badge-devops-sre" in html


class TestTriageAndGNNModels:
    """Unit tests for ComponentDetail and LayerData GNN/Triage extensions."""

    def test_component_detail_gnn_and_triage_fields(self):
        detail = ComponentDetail(
            id="app_1",
            name="App 1",
            type="Application",
            overall=0.85,
            level="CRITICAL",
            gnn_score=0.89,
            triage_rank=1,
            triage_priority_action="Split logical topic",
            triage_roles=["Architect", "Developer"],
            triage_pattern="GOD_COMPONENT",
        )
        assert detail.gnn_score == 0.89
        assert detail.triage_rank == 1
        assert detail.triage_roles == ["Architect", "Developer"]

        d = detail.to_dict()
        assert d["gnn_score"] == 0.89
        assert d["triage_rank"] == 1
        assert d["triage_priority_action"] == "Split logical topic"
        assert d["triage_roles"] == ["Architect", "Developer"]
        assert d["triage_pattern"] == "GOD_COMPONENT"

    def test_layer_data_gnn_and_triage_fields(self):
        data = LayerData(layer="system", name="System")
        assert not data.has_gnn
        assert data.triage_entries == []
        assert data.triage_ranking_source == ""

        data.has_gnn = True
        data.gnn_spearman = 0.88
        data.triage_ranking_source = "gnn"
        data.triage_entries = [{"component_id": "c1", "rank": 1}]
        assert data.has_gnn
        assert data.gnn_spearman == 0.88
        assert len(data.triage_entries) == 1


class TestInteractiveNetworkGraph:
    """Unit tests for revised Cytoscape interactive network topology."""

    def test_cytoscape_network_renders_toolbar_and_inspector(self):
        dash = DashboardGenerator("Network Test")
        dash.add_tab("Topology", "topology")
        nodes = [
            {"id": "app_1", "name": "App 1", "type": "Application", "level": "CRITICAL", "value": 35.0, "score": 0.85, "spof": True},
            {"id": "topic_1", "name": "Topic 1", "type": "Topic", "level": "MEDIUM", "value": 20.0, "score": 0.40, "spof": False},
        ]
        edges = [
            {"source": "app_1", "target": "topic_1", "weight": 2.5, "dependency_type": "PUBLISHES"},
        ]
        dash.add_cytoscape_network("test-net", nodes, edges, title="Test Network")
        dash.end_tab()
        html = dash.generate()

        assert "cy-toolbar" in html
        assert "cy-inspector" in html
        assert "window.sagChangeLayout" in html
        assert "window.sagSearchNodes" in html
        assert "round-rectangle" in html
        assert "ellipse" in html
        assert "SPOF (Articulation Point)" in html

    def test_cytoscape_tab_switch_registers_resizing(self):
        dash = DashboardGenerator("Resize Test")
        html = dash.generate()
        assert "window.sagCyInstances" in html
        assert "cy.resize()" in html
        assert "cy.fit(25)" in html


