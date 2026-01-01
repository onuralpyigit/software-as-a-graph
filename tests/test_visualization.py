"""
Tests for src.visualization module
====================================

Tests GraphRenderer and DashboardGenerator.
"""

import pytest


class TestGraphRenderer:
    """Tests for GraphRenderer class"""

    def test_create_renderer(self):
        """Create graph renderer"""
        from src.visualization import GraphRenderer
        
        renderer = GraphRenderer()
        
        assert renderer is not None

    def test_render_basic(self, medium_graph):
        """Render basic graph"""
        from src.visualization import GraphRenderer
        
        renderer = GraphRenderer()
        html = renderer.render(medium_graph)
        
        assert html is not None
        assert "<!DOCTYPE html>" in html
        assert "vis.js" in html or "vis-network" in html

    def test_render_with_criticality(self, medium_graph, criticality_scores):
        """Render graph with criticality coloring"""
        from src.visualization import GraphRenderer, RenderConfig, ColorScheme
        
        config = RenderConfig(color_scheme=ColorScheme.CRITICALITY)
        renderer = GraphRenderer(config)
        html = renderer.render(medium_graph, criticality_scores)
        
        assert html is not None
        assert len(html) > 1000

    def test_render_multi_layer(self, medium_graph):
        """Render multi-layer view"""
        from src.visualization import GraphRenderer
        
        renderer = GraphRenderer()
        html = renderer.render_multi_layer(medium_graph)
        
        assert html is not None
        assert "Multi-Layer" in html or "layer" in html.lower()

    def test_render_multi_layer_with_criticality(self, medium_graph, criticality_scores):
        """Render multi-layer view with criticality"""
        from src.visualization import GraphRenderer
        
        renderer = GraphRenderer()
        html = renderer.render_multi_layer(medium_graph, criticality_scores)
        
        assert html is not None

    def test_render_config(self, medium_graph):
        """Test render configuration"""
        from src.visualization import GraphRenderer, RenderConfig
        
        config = RenderConfig(
            title="Test Graph",
            physics_enabled=False,
            show_labels=False,
        )
        renderer = GraphRenderer(config)
        html = renderer.render(medium_graph)
        
        assert "Test Graph" in html

    def test_render_layer_filter(self, medium_graph):
        """Render specific layer"""
        from src.visualization import GraphRenderer, Layer
        
        renderer = GraphRenderer()
        html = renderer.render(medium_graph, layer=Layer.APPLICATION)
        
        assert html is not None

    def test_color_schemes(self, medium_graph, criticality_scores):
        """Test different color schemes"""
        from src.visualization import GraphRenderer, RenderConfig, ColorScheme
        
        for scheme in [ColorScheme.TYPE, ColorScheme.CRITICALITY, ColorScheme.LAYER]:
            config = RenderConfig(color_scheme=scheme)
            renderer = GraphRenderer(config)
            html = renderer.render(medium_graph, criticality_scores)
            assert html is not None


class TestDashboardGenerator:
    """Tests for DashboardGenerator class"""

    def test_create_generator(self):
        """Create dashboard generator"""
        from src.visualization import DashboardGenerator
        
        generator = DashboardGenerator()
        
        assert generator is not None

    def test_generate_basic(self, medium_graph):
        """Generate basic dashboard"""
        from src.visualization import DashboardGenerator
        
        generator = DashboardGenerator()
        html = generator.generate(medium_graph)
        
        assert html is not None
        assert "<!DOCTYPE html>" in html
        assert "Chart.js" in html or "chart.js" in html

    def test_generate_with_criticality(self, medium_graph, criticality_scores):
        """Generate dashboard with criticality"""
        from src.visualization import DashboardGenerator
        
        generator = DashboardGenerator()
        html = generator.generate(medium_graph, criticality=criticality_scores)
        
        assert html is not None
        assert "Criticality" in html or "critical" in html.lower()

    def test_generate_with_validation(self, medium_graph, criticality_scores):
        """Generate dashboard with validation results"""
        from src.visualization import DashboardGenerator
        from src.validation import ValidationPipeline
        
        # Get validation results
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(medium_graph)
        
        generator = DashboardGenerator()
        html = generator.generate(
            medium_graph,
            criticality=criticality_scores,
            validation=result.validation.to_dict(),
        )
        
        assert html is not None
        assert "Spearman" in html or "spearman" in html.lower()

    def test_generate_with_simulation(self, medium_graph, criticality_scores):
        """Generate dashboard with simulation results"""
        from src.visualization import DashboardGenerator
        from src.simulation import EventSimulator
        
        # Get simulation results
        sim = EventSimulator(seed=42)
        sim_result = sim.simulate(medium_graph, duration_ms=1000, message_rate=50)
        
        generator = DashboardGenerator()
        html = generator.generate(
            medium_graph,
            criticality=criticality_scores,
            simulation=sim_result.metrics.to_dict(),
        )
        
        assert html is not None

    def test_dashboard_config(self, medium_graph):
        """Test dashboard configuration"""
        from src.visualization import DashboardGenerator, DashboardConfig
        
        config = DashboardConfig(
            title="Custom Dashboard",
            theme="dark",
        )
        generator = DashboardGenerator(config)
        html = generator.generate(medium_graph)
        
        assert "Custom Dashboard" in html

    def test_dashboard_light_theme(self, medium_graph):
        """Test light theme dashboard"""
        from src.visualization import DashboardGenerator, DashboardConfig
        
        config = DashboardConfig(theme="light")
        generator = DashboardGenerator(config)
        html = generator.generate(medium_graph)
        
        assert html is not None


class TestVisualizationOutput:
    """Tests for visualization file output"""

    def test_save_html(self, medium_graph, temp_dir):
        """Save visualization to HTML file"""
        from src.visualization import GraphRenderer
        
        renderer = GraphRenderer()
        html = renderer.render(medium_graph)
        
        filepath = temp_dir / "graph.html"
        with open(filepath, 'w') as f:
            f.write(html)
        
        assert filepath.exists()
        assert filepath.stat().st_size > 1000

    def test_save_dashboard(self, medium_graph, temp_dir):
        """Save dashboard to HTML file"""
        from src.visualization import DashboardGenerator
        
        generator = DashboardGenerator()
        html = generator.generate(medium_graph)
        
        filepath = temp_dir / "dashboard.html"
        with open(filepath, 'w') as f:
            f.write(html)
        
        assert filepath.exists()
        assert filepath.stat().st_size > 5000

    def test_multiple_outputs(self, medium_graph, criticality_scores, temp_dir):
        """Generate multiple visualization outputs"""
        from src.visualization import GraphRenderer, DashboardGenerator
        
        # Graph
        renderer = GraphRenderer()
        graph_html = renderer.render(medium_graph, criticality_scores)
        
        # Multi-layer
        multi_html = renderer.render_multi_layer(medium_graph, criticality_scores)
        
        # Dashboard
        generator = DashboardGenerator()
        dash_html = generator.generate(medium_graph, criticality=criticality_scores)
        
        # Save all
        (temp_dir / "graph.html").write_text(graph_html)
        (temp_dir / "multi.html").write_text(multi_html)
        (temp_dir / "dashboard.html").write_text(dash_html)
        
        assert (temp_dir / "graph.html").exists()
        assert (temp_dir / "multi.html").exists()
        assert (temp_dir / "dashboard.html").exists()
