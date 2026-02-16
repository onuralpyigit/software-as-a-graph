"""
Tests for Step 6: Visualization

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

from src.visualization import (
    LayerData, ComponentDetail, LAYER_DEFINITIONS,
)
from src.visualization.models import ChartOutput, ColorTheme, DEFAULT_THEME
from src.visualization.charts import (
    ChartGenerator, CRITICALITY_COLORS, RMAV_COLORS,
)
from src.visualization.dashboard import DashboardGenerator
from src.visualization.collector import LayerDataCollector
from src.visualization import VisualizationService


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

    def test_scale_category_small(self):
        """Systems with < 50 nodes are 'small'."""
        data = LayerData(layer="app", name="App")
        data.nodes = 30
        assert data.scale_category == "small"
        assert not data.recommend_matrix_only

    def test_scale_category_medium(self):
        """Systems with 50-200 nodes are 'medium'."""
        data = LayerData(layer="app", name="App")
        data.nodes = 100
        assert data.scale_category == "medium"
        assert not data.recommend_matrix_only

    def test_scale_category_large(self):
        """Systems with 200-1000 nodes are 'large' and recommend matrix only."""
        data = LayerData(layer="app", name="App")
        data.nodes = 500
        assert data.scale_category == "large"
        assert data.recommend_matrix_only

    def test_scale_category_xlarge(self):
        """Systems with > 1000 nodes are 'xlarge'."""
        data = LayerData(layer="app", name="App")
        data.nodes = 2000
        assert data.scale_category == "xlarge"
        assert data.recommend_matrix_only

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
        """has_simulation is True when event_throughput > 0."""
        data = LayerData(layer="app", name="App")
        data.event_throughput = 100
        assert data.has_simulation

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
            availability=0.90,
            vulnerability=0.75,
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

    def test_impact_ranking_generates_html(self):
        """§6.4.3: impact_ranking returns bar chart HTML."""
        data = [
            ("sensor_fusion", 0.84, "CRITICAL"),
            ("main_broker", 0.80, "CRITICAL"),
            ("planning", 0.64, "HIGH"),
        ]
        html = self.charts.impact_ranking(data)
        assert html is not None
        assert "<canvas" in html

    def test_impact_ranking_with_names(self):
        """impact_ranking uses name mapping when provided."""
        data = [("sf", 0.84, "CRITICAL")]
        names = {"sf": "Sensor Fusion"}
        html = self.charts.impact_ranking(data, names=names)
        assert html is not None
        assert "Sensor Fusion" in html

    def test_rmav_breakdown_generates_html(self):
        """§6.4.3: rmav_breakdown returns stacked bar chart."""
        components = [
            ComponentDetail("a", "App A", "Application",
                           0.8, 0.7, 0.9, 0.6, 0.75, "CRITICAL"),
            ComponentDetail("b", "App B", "Application",
                           0.5, 0.6, 0.4, 0.3, 0.45, "MEDIUM"),
        ]
        html = self.charts.rmav_breakdown(components)
        assert html is not None
        assert "<canvas" in html
        assert "Reliability" in html
        assert "Maintainability" in html

    def test_rmav_breakdown_empty(self):
        """rmav_breakdown returns None for empty components."""
        assert self.charts.rmav_breakdown([]) is None

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

    def test_pie_chart(self):
        """pie_chart generates generic pie chart."""
        data = {"Application": 10, "Broker": 3, "Node": 5}
        html = self.charts.pie_chart(data)
        assert html is not None
        assert "<canvas" in html

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
    mock_comp.scores.vulnerability = 0.75
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
    service = MagicMock()
    mock_metrics = MagicMock()
    mock_metrics.event_throughput = 1000
    mock_metrics.event_delivery_rate = 98.5
    mock_metrics.avg_reachability_loss = 0.15
    mock_metrics.max_impact = 0.734
    service.analyze_layer.return_value = mock_metrics
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


class TestLayerDataCollector:
    """Tests for data collection pipeline."""

    def test_collect_basic_data(
        self, mock_analysis_service, mock_simulation_service, mock_validation_service
    ):
        """Collector populates basic layer data from all services."""
        repo = MagicMock()
        collector = LayerDataCollector(
            mock_analysis_service,
            mock_simulation_service,
            mock_validation_service,
            repo,
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
        self, mock_analysis_service, mock_simulation_service, mock_validation_service
    ):
        """Collector builds ComponentDetail list with RMAV scores."""
        repo = MagicMock()
        collector = LayerDataCollector(
            mock_analysis_service,
            mock_simulation_service,
            mock_validation_service,
            repo,
        )

        data = collector.collect_layer_data("system")

        assert len(data.component_details) == 1
        detail = data.component_details[0]
        assert detail.id == "sensor_fusion"
        assert detail.reliability == pytest.approx(0.82)
        assert detail.maintainability == pytest.approx(0.88)
        assert detail.availability == pytest.approx(0.90)
        assert detail.vulnerability == pytest.approx(0.75)
        assert detail.overall == pytest.approx(0.84)
        assert detail.level == "CRITICAL"

    def test_collect_scatter_data(
        self, mock_analysis_service, mock_simulation_service, mock_validation_service
    ):
        """Collector builds scatter plot data from Q(v) and I(v)."""
        repo = MagicMock()
        collector = LayerDataCollector(
            mock_analysis_service,
            mock_simulation_service,
            mock_validation_service,
            repo,
        )

        data = collector.collect_layer_data("system")

        # Should have scatter data for each component with Q > 0
        assert len(data.scatter_data) >= 1
        comp_id, q_val, i_val, level = data.scatter_data[0]
        assert comp_id == "sensor_fusion"
        assert q_val == pytest.approx(0.84)

    def test_collect_top_k_overlap(
        self, mock_analysis_service, mock_simulation_service, mock_validation_service
    ):
        """Collector captures Top-K overlap from validation."""
        repo = MagicMock()
        collector = LayerDataCollector(
            mock_analysis_service,
            mock_simulation_service,
            mock_validation_service,
            repo,
        )

        data = collector.collect_layer_data("system", include_validation=True)

        assert data.top5_overlap == pytest.approx(0.80)
        assert data.top10_overlap == pytest.approx(0.70)

    def test_collect_unknown_layer_raises(
        self, mock_analysis_service, mock_simulation_service, mock_validation_service
    ):
        """Unknown layer raises ValueError."""
        repo = MagicMock()
        collector = LayerDataCollector(
            mock_analysis_service,
            mock_simulation_service,
            mock_validation_service,
            repo,
        )

        with pytest.raises(ValueError, match="Unknown layer"):
            collector.collect_layer_data("nonexistent")

    def test_collect_handles_analysis_failure(
        self, mock_simulation_service, mock_validation_service
    ):
        """Collector gracefully handles analysis service failure."""
        bad_analysis = MagicMock()
        bad_analysis.analyze_layer.side_effect = Exception("Neo4j down")

        repo = MagicMock()
        collector = LayerDataCollector(
            bad_analysis,
            mock_simulation_service,
            mock_validation_service,
            repo,
        )

        # Should not raise
        data = collector.collect_layer_data("system")
        assert data.nodes == 0  # Defaults preserved


# =========================================================================
# VisualizationService Integration Tests
# =========================================================================

class TestVisualizationService:
    """Tests for full dashboard generation pipeline."""

    def test_generate_dashboard_creates_file(
        self, mock_analysis_service, mock_simulation_service, mock_validation_service, tmp_path
    ):
        """generate_dashboard produces an HTML file."""
        repo = MagicMock()
        service = VisualizationService(
            mock_analysis_service,
            mock_simulation_service,
            mock_validation_service,
            repo,
        )

        output = tmp_path / "test_dashboard.html"

        with patch.object(service, "collector") as mock_collector:
            mock_data = LayerData(layer="system", name="Complete System")
            mock_data.nodes = 48
            mock_data.critical_count = 5
            mock_data.spearman = 0.876
            mock_data.f1_score = 0.923
            mock_data.precision = 0.912
            mock_data.recall = 0.857
            mock_data.validation_passed = True
            mock_data.component_details = [
                ComponentDetail("a", "App A", "Application",
                               0.8, 0.7, 0.9, 0.6, 0.75, "CRITICAL"),
            ]
            mock_data.scatter_data = [("a", 0.75, 0.70, "CRITICAL")]
            mock_collector.collect_layer_data.return_value = mock_data

            with patch(
                "src.visualization.dashboard.DashboardGenerator"
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
        self, mock_analysis_service, mock_simulation_service, mock_validation_service
    ):
        """Large systems auto-disable network graph (§6.11)."""
        repo = MagicMock()
        service = VisualizationService(
            mock_analysis_service,
            mock_simulation_service,
            mock_validation_service,
            repo,
        )

        with patch.object(service, "collector") as mock_collector:
            mock_data = LayerData(layer="system", name="Complete System")
            mock_data.nodes = 500  # > 200 threshold
            mock_collector.collect_layer_data.return_value = mock_data

            with patch.object(service, "_add_layer_section") as mock_add:
                with patch(
                    "src.visualization.dashboard.DashboardGenerator"
                ) as MockDash:
                    dash_instance = MockDash.return_value
                    dash_instance.generate.return_value = "<html></html>"

                    service.generate_dashboard(
                        output_file="/tmp/test.html",
                        layers=["system"],
                        include_network=True,
                    )

                    # _add_layer_section should be called with include_network=False
                    call_args = mock_add.call_args
                    # effective_network arg (3rd positional) should be False
                    assert call_args[0][2] is False

    def test_unknown_layer_skipped(
        self, mock_analysis_service, mock_simulation_service, mock_validation_service
    ):
        """Unknown layers are skipped with a warning."""
        repo = MagicMock()
        service = VisualizationService(
            mock_analysis_service,
            mock_simulation_service,
            mock_validation_service,
            repo,
        )

        with patch(
            "src.visualization.dashboard.DashboardGenerator"
        ) as MockDash:
            dash_instance = MockDash.return_value
            dash_instance.generate.return_value = "<html></html>"

            # Should not raise
            service.generate_dashboard(
                output_file="/tmp/test.html",
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

    def test_all_rmav_dimensions_have_colors(self):
        """Every RMAV dimension has a defined color."""
        dims = ["reliability", "maintainability", "availability", "vulnerability"]
        for dim in dims:
            assert dim in RMAV_COLORS
            assert RMAV_COLORS[dim].startswith("#")

    def test_layer_definitions_complete(self):
        """All four layers are defined with required fields."""
        expected = {"app", "infra", "mw", "system"}
        assert set(LAYER_DEFINITIONS.keys()) == expected
        for layer, defn in LAYER_DEFINITIONS.items():
            assert "name" in defn
            assert "description" in defn
            assert "icon" in defn
# =========================================================================
# Merged Unit Tests from test_visualization.py
# =========================================================================

class TestColorTheme:
    """Tests for configurable color themes."""
    
    def test_default_theme_has_all_colors(self):
        """Default theme should have all required color attributes."""
        theme = DEFAULT_THEME
        assert theme.primary == "#3498db"
        assert theme.success == "#2ecc71"
        assert theme.danger == "#e74c3c"
        assert theme.critical == "#e74c3c"
        assert theme.layer_app == "#3498db"
    
    def test_custom_theme_override(self):
        """Custom theme should override default colors."""
        theme = ColorTheme(primary="#ff0000", success="#00ff00")
        assert theme.primary == "#ff0000"
        assert theme.success == "#00ff00"
        # Non-overridden should use defaults
        assert theme.danger == "#e74c3c"
    
    def test_theme_to_dict_conversions(self):
        """Theme should convert to backwards-compatible dictionaries."""
        theme = ColorTheme()
        
        colors_dict = theme.to_colors_dict()
        assert "primary" in colors_dict
        assert "danger" in colors_dict
        
        crit_dict = theme.to_criticality_dict()
        assert "CRITICAL" in crit_dict
        assert "LOW" in crit_dict
        
        layer_dict = theme.to_layer_dict()
        assert "app" in layer_dict
        assert "system" in layer_dict


class TestChartOutput:
    """Tests for chart output dataclass."""
    
    def test_chart_output_has_alt_text(self):
        """ChartOutput should support alt_text for accessibility."""
        chart = ChartOutput(
            title="Test Chart",
            png_base64="abc123",
            alt_text="A bar chart showing component distribution"
        )
        assert chart.alt_text == "A bar chart showing component distribution"
    
    def test_chart_output_defaults(self):
        """ChartOutput should have sensible defaults."""
        chart = ChartOutput(title="Test", png_base64="data")
        assert chart.description == ""
        assert chart.alt_text == ""
        assert chart.width == 600
        assert chart.height == 400


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
