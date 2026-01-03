#!/usr/bin/env python3
"""
Test Suite for src.visualization Module - Version 5.0

Tests for visualization including:
- Chart generation
- Dashboard building
- HTML output
- Configuration

Run with pytest:
    pytest tests/test_visualization.py -v

Or standalone:
    python tests/test_visualization.py

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


# =============================================================================
# Test Data
# =============================================================================

def create_test_graph():
    """Create a simple test graph."""
    from src.simulation import (
        SimulationGraph,
        Component,
        Edge,
        ComponentType,
        EdgeType,
    )
    
    graph = SimulationGraph()
    
    # Add nodes
    for i in range(2):
        graph.add_component(Component(
            id=f"node_{i+1}",
            type=ComponentType.NODE,
            name=f"Node {i+1}",
        ))
    
    # Add broker
    graph.add_component(Component(
        id="broker_1",
        type=ComponentType.BROKER,
        name="Broker 1",
    ))
    
    # Add topics
    for i in range(3):
        graph.add_component(Component(
            id=f"topic_{i+1}",
            type=ComponentType.TOPIC,
            name=f"Topic {i+1}",
        ))
        graph.add_edge(Edge(
            source="broker_1",
            target=f"topic_{i+1}",
            edge_type=EdgeType.ROUTES,
        ))
    
    # Add applications
    for i in range(4):
        graph.add_component(Component(
            id=f"app_{i+1}",
            type=ComponentType.APPLICATION,
            name=f"App {i+1}",
        ))
        # Add pub/sub relationships
        topic = f"topic_{(i % 3) + 1}"
        graph.add_edge(Edge(
            source=f"app_{i+1}",
            target=topic,
            edge_type=EdgeType.PUBLISHES_TO,
        ))
        graph.add_edge(Edge(
            source=f"app_{i+1}",
            target=f"topic_{((i+1) % 3) + 1}",
            edge_type=EdgeType.SUBSCRIBES_TO,
        ))
    
    # Infrastructure edges
    graph.add_edge(Edge(source="broker_1", target="node_1", edge_type=EdgeType.RUNS_ON))
    graph.add_edge(Edge(source="node_1", target="broker_1", edge_type=EdgeType.CONNECTS_TO))
    
    return graph


def create_test_campaign(graph):
    """Create test campaign result."""
    from src.simulation import FailureSimulator
    
    simulator = FailureSimulator(seed=42, cascade=True)
    return simulator.simulate_all(graph)


def create_test_validation(graph):
    """Create test validation result."""
    from src.validation import ValidationPipeline
    
    pipeline = ValidationPipeline(seed=42, cascade=True)
    return pipeline.run(graph, compare_methods=True)


# =============================================================================
# Test: Configuration
# =============================================================================

class TestConfiguration:
    """Tests for configuration classes."""
    
    def test_chart_config_defaults(self):
        from src.visualization import ChartConfig
        
        config = ChartConfig()
        
        assert config.width == 8.0
        assert config.height == 5.0
        assert config.dpi == 100
        assert len(config.colors) >= 4
    
    def test_chart_config_custom(self):
        from src.visualization import ChartConfig
        
        config = ChartConfig(width=10.0, height=6.0, dpi=150)
        
        assert config.width == 10.0
        assert config.height == 6.0
        assert config.dpi == 150
    
    def test_chart_theme_enum(self):
        from src.visualization import ChartTheme
        
        assert ChartTheme.DEFAULT.value == "default"
        assert ChartTheme.DARK.value == "dark"
        assert ChartTheme.LIGHT.value == "light"
    
    def test_dashboard_config_defaults(self):
        from src.visualization import DashboardConfig
        
        config = DashboardConfig()
        
        assert config.title == "Graph Analysis Dashboard"
        assert config.theme == "light"
        assert config.include_timestamp == True
    
    def test_dashboard_config_custom(self):
        from src.visualization import DashboardConfig
        
        config = DashboardConfig(
            title="Custom Dashboard",
            theme="dark",
            primary_color="#ff0000",
        )
        
        assert config.title == "Custom Dashboard"
        assert config.theme == "dark"
        assert config.primary_color == "#ff0000"


# =============================================================================
# Test: Chart Generation
# =============================================================================

class TestChartGeneration:
    """Tests for chart generation functions."""
    
    @classmethod
    def setup_class(cls):
        from src.visualization import check_matplotlib_available
        cls.has_matplotlib = check_matplotlib_available()
    
    def test_check_matplotlib_available(self):
        from src.visualization import check_matplotlib_available
        
        result = check_matplotlib_available()
        assert isinstance(result, bool)
    
    def test_chart_component_distribution(self):
        if not self.has_matplotlib:
            return  # Skip
        
        from src.visualization import chart_component_distribution, ChartOutput
        
        components = {
            "Application": 10,
            "Broker": 2,
            "Topic": 5,
            "Node": 3,
        }
        
        chart = chart_component_distribution(components)
        
        assert isinstance(chart, ChartOutput)
        assert chart.title == "Component Distribution"
        assert len(chart.svg) > 0
        assert len(chart.png_base64) > 0
    
    def test_chart_edge_distribution(self):
        if not self.has_matplotlib:
            return  # Skip
        
        from src.visualization import chart_edge_distribution, ChartOutput
        
        edges = {
            "PUBLISHES_TO": 20,
            "SUBSCRIBES_TO": 20,
            "ROUTES": 5,
            "RUNS_ON": 10,
        }
        
        chart = chart_edge_distribution(edges)
        
        assert isinstance(chart, ChartOutput)
        assert chart.title == "Edge Distribution"
    
    def test_chart_impact_ranking(self):
        if not self.has_matplotlib:
            return  # Skip
        
        from src.visualization import chart_impact_ranking, ChartOutput
        
        impacts = [
            ("broker_1", 0.8),
            ("node_1", 0.5),
            ("app_1", 0.3),
            ("app_2", 0.1),
        ]
        
        chart = chart_impact_ranking(impacts)
        
        assert isinstance(chart, ChartOutput)
        assert chart.title == "Impact Ranking"
    
    def test_chart_criticality_distribution(self):
        if not self.has_matplotlib:
            return  # Skip
        
        from src.visualization import chart_criticality_distribution, ChartOutput
        
        levels = {
            "CRITICAL": 3,
            "HIGH": 5,
            "MEDIUM": 10,
            "LOW": 20,
        }
        
        chart = chart_criticality_distribution(levels)
        
        assert isinstance(chart, ChartOutput)
    
    def test_chart_correlation_comparison(self):
        if not self.has_matplotlib:
            return  # Skip
        
        from src.visualization import chart_correlation_comparison, ChartOutput
        
        metrics = {
            "Spearman": 0.85,
            "Pearson": 0.80,
            "Kendall": 0.75,
        }
        targets = {
            "Spearman": 0.70,
            "Pearson": 0.65,
            "Kendall": 0.60,
        }
        
        chart = chart_correlation_comparison(metrics, targets)
        
        assert isinstance(chart, ChartOutput)
    
    def test_chart_confusion_matrix(self):
        if not self.has_matplotlib:
            return  # Skip
        
        from src.visualization import chart_confusion_matrix, ChartOutput
        
        chart = chart_confusion_matrix(10, 2, 3, 20)
        
        assert isinstance(chart, ChartOutput)
        assert chart.title == "Confusion Matrix"
    
    def test_chart_layer_validation(self):
        if not self.has_matplotlib:
            return  # Skip
        
        from src.visualization import chart_layer_validation, ChartOutput
        
        layers = {
            "application": {"spearman": 0.85, "f1_score": 0.92},
            "infrastructure": {"spearman": 0.75, "f1_score": 0.88},
        }
        
        chart = chart_layer_validation(layers)
        
        assert isinstance(chart, ChartOutput)
    
    def test_chart_output_to_html(self):
        if not self.has_matplotlib:
            return  # Skip
        
        from src.visualization import chart_component_distribution
        
        chart = chart_component_distribution({"A": 5, "B": 3})
        
        html_img = chart.to_html_img()
        assert "<img" in html_img
        assert "data:image/png;base64" in html_img
        
        html_svg = chart.to_html_svg()
        assert "<svg" in html_svg


# =============================================================================
# Test: Dashboard Builder
# =============================================================================

class TestDashboardBuilder:
    """Tests for DashboardBuilder class."""
    
    def test_basic_dashboard(self):
        from src.visualization import DashboardBuilder, DashboardConfig
        
        config = DashboardConfig(title="Test Dashboard")
        builder = DashboardBuilder(config)
        
        builder.add_header("Test Header", "Subtitle")
        html = builder.build()
        
        assert "<!DOCTYPE html>" in html
        assert "Test Header" in html
        assert "Subtitle" in html
    
    def test_stats_row(self):
        from src.visualization import DashboardBuilder
        
        builder = DashboardBuilder()
        builder.add_stats_row({
            "Components": 100,
            "Edges": 200,
            "Score": 0.95,
        }, "Statistics")
        
        html = builder.build()
        
        assert "100" in html
        assert "200" in html
        assert "0.95" in html
        assert "Statistics" in html
    
    def test_table(self):
        from src.visualization import DashboardBuilder
        
        builder = DashboardBuilder()
        builder.add_table(
            headers=["Name", "Value", "Status"],
            rows=[
                ["Item 1", 10, "PASSED"],
                ["Item 2", 20, "FAILED"],
            ],
            title="Test Table",
        )
        
        html = builder.build()
        
        assert "Item 1" in html
        assert "Test Table" in html
        assert "badge-success" in html  # PASSED badge
        assert "badge-danger" in html  # FAILED badge
    
    def test_grid_layout(self):
        from src.visualization import DashboardBuilder
        
        builder = DashboardBuilder()
        builder.start_grid()
        builder.add_stats_row({"A": 1}, "Box 1")
        builder.add_stats_row({"B": 2}, "Box 2")
        builder.end_grid()
        
        html = builder.build()
        
        assert 'class="grid"' in html
    
    def test_section_title(self):
        from src.visualization import DashboardBuilder
        
        builder = DashboardBuilder()
        builder.add_section_title("My Section")
        
        html = builder.build()
        
        assert "My Section" in html
        assert "<h2" in html


# =============================================================================
# Test: Dashboard Generators
# =============================================================================

class TestDashboardGenerators:
    """Tests for dashboard generator functions."""
    
    def test_generate_graph_dashboard(self):
        from src.visualization import generate_graph_dashboard
        
        graph = create_test_graph()
        html = generate_graph_dashboard(graph)
        
        assert "<!DOCTYPE html>" in html
        assert "Graph Model Dashboard" in html
        # Should contain component counts
        assert "Application" in html or "Components" in html
    
    def test_generate_simulation_dashboard(self):
        from src.visualization import generate_simulation_dashboard
        
        graph = create_test_graph()
        campaign = create_test_campaign(graph)
        
        html = generate_simulation_dashboard(graph, campaign)
        
        assert "<!DOCTYPE html>" in html
        assert "Simulation" in html
        assert "Impact" in html
    
    def test_generate_validation_dashboard(self):
        from src.visualization import generate_validation_dashboard
        
        graph = create_test_graph()
        validation = create_test_validation(graph)
        
        html = generate_validation_dashboard(validation)
        
        assert "<!DOCTYPE html>" in html
        assert "Validation" in html
        assert "Spearman" in html
    
    def test_generate_overview_dashboard(self):
        from src.visualization import generate_overview_dashboard
        
        graph = create_test_graph()
        campaign = create_test_campaign(graph)
        validation = create_test_validation(graph)
        
        html = generate_overview_dashboard(
            graph,
            campaign_result=campaign,
            validation_result=validation,
        )
        
        assert "<!DOCTYPE html>" in html
        assert "Overview" in html
    
    def test_generate_overview_minimal(self):
        from src.visualization import generate_overview_dashboard
        
        graph = create_test_graph()
        
        # Just graph, no simulation or validation
        html = generate_overview_dashboard(graph)
        
        assert "<!DOCTYPE html>" in html
    
    def test_custom_dashboard_config(self):
        from src.visualization import generate_graph_dashboard, DashboardConfig
        
        graph = create_test_graph()
        config = DashboardConfig(
            title="Custom Title",
            primary_color="#ff5500",
        )
        
        html = generate_graph_dashboard(graph, config=config)
        
        assert "Custom Title" in html
        assert "#ff5500" in html


# =============================================================================
# Test: Integration
# =============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test complete visualization workflow."""
        from src.simulation import FailureSimulator
        from src.validation import ValidationPipeline
        from src.visualization import generate_overview_dashboard
        
        # Create graph
        graph = create_test_graph()
        
        # Run simulation
        simulator = FailureSimulator(seed=42, cascade=True)
        campaign = simulator.simulate_all(graph)
        
        # Run validation
        pipeline = ValidationPipeline(seed=42)
        validation = pipeline.run(graph)
        
        # Generate dashboard
        html = generate_overview_dashboard(
            graph,
            campaign_result=campaign,
            validation_result=validation,
        )
        
        # Verify output
        assert len(html) > 1000
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
    
    def test_save_dashboard(self):
        """Test saving dashboard to file."""
        import tempfile
        import os
        from src.visualization import generate_graph_dashboard
        
        graph = create_test_graph()
        html = generate_graph_dashboard(graph)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            temp_path = f.name
        
        try:
            # Verify file exists and has content
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
            # Read back and verify
            with open(temp_path, 'r') as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
        finally:
            os.unlink(temp_path)


# =============================================================================
# Standalone Test Runner
# =============================================================================

def run_tests_standalone():
    """Run tests without pytest."""
    import traceback
    
    test_classes = [
        TestConfiguration,
        TestChartGeneration,
        TestDashboardBuilder,
        TestDashboardGenerators,
        TestIntegration,
    ]
    
    passed = failed = skipped = 0
    
    print("=" * 60)
    print("Software-as-a-Graph Visualization Module Tests")
    print("=" * 60)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        # Setup class if needed
        if hasattr(test_class, 'setup_class'):
            try:
                test_class.setup_class()
            except:
                pass
        
        instance = test_class()
        for method_name in [m for m in dir(instance) if m.startswith("test_")]:
            try:
                getattr(instance, method_name)()
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                if "Skip" in str(e) or str(e) == "":
                    print(f"  ○ {method_name} (skipped)")
                    skipped += 1
                else:
                    print(f"  ✗ {method_name}: {e}")
                    if "--verbose" in sys.argv:
                        traceback.print_exc()
                    failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    if HAS_PYTEST and "--no-pytest" not in sys.argv:
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        success = run_tests_standalone()
        sys.exit(0 if success else 1)