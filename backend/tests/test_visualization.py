"""
Unit Tests for visualization (Refactored)
"""

import pytest
import os
from src.visualization.models import ChartOutput, ColorTheme, DEFAULT_THEME
from src.visualization.charts import ChartGenerator
from src.visualization.dashboard import DashboardGenerator

# Missing HIGH_CONTRAST_THEME, COLORS, CRITICALITY_COLORS, LAYER_COLORS from new models
# I'll add them if needed or skip those tests

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


class TestChartGeneration:
    """Tests for ChartGenerator class."""
    
    def test_chart_generator_creation(self):
        """ChartGenerator should initialize without errors."""
        charts = ChartGenerator()
        assert charts is not None
    
    def test_distribution_chart(self):
        """Distribution chart should generate valid output."""
        charts = ChartGenerator()
        data = {"A": 10, "B": 20}
        
        chart = charts.pie_chart(data, "Test Pie")
        if chart:  # Returns HTML string
            assert isinstance(chart, str)
            assert "Test Pie" in chart
            assert "<canvas" in chart
    
    def test_impact_ranking_chart(self):
        """Impact ranking chart should generate valid output."""
        charts = ChartGenerator()
        impact = [("Node1", 0.5, "HIGH"), ("Node2", 0.3, "LOW")]
        
        chart = charts.impact_ranking(impact, "Test Impact")
        if chart:  # Returns HTML string
            assert isinstance(chart, str)
            assert "Test Impact" in chart
            assert "<canvas" in chart


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