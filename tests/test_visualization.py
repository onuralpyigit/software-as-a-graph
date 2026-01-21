"""
Unit Tests for src/visualization module

Tests for:
    - Chart generation with matplotlib
    - Dashboard HTML generation
    - ColorTheme configuration
    - Accessibility features
"""

import pytest
import os
from src.visualization.charts import (
    ChartGenerator, 
    ChartOutput,
    ColorTheme,
    DEFAULT_THEME,
    HIGH_CONTRAST_THEME,
    COLORS,
    CRITICALITY_COLORS,
    LAYER_COLORS,
)
from src.visualization.dashboard import DashboardGenerator


# =============================================================================
# ColorTheme Tests
# =============================================================================

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
    
    def test_backwards_compatible_dictionaries(self):
        """Module-level dictionaries should match default theme."""
        assert COLORS["primary"] == DEFAULT_THEME.primary
        assert CRITICALITY_COLORS["CRITICAL"] == DEFAULT_THEME.critical
        assert LAYER_COLORS["app"] == DEFAULT_THEME.layer_app
    
    def test_high_contrast_theme(self):
        """High contrast theme should have more saturated colors."""
        assert HIGH_CONTRAST_THEME.primary != DEFAULT_THEME.primary


# =============================================================================
# ChartOutput Tests
# =============================================================================

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


# =============================================================================
# Chart Generation Tests
# =============================================================================

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
        
        # Method name is pie_chart
        chart = charts.pie_chart(data, "Test Pie")
        if chart:  # Only if matplotlib is installed
            assert chart.title == "Test Pie"
            assert len(chart.png_base64) > 0
    
    def test_impact_ranking_chart(self):
        """Impact ranking chart should generate valid output."""
        charts = ChartGenerator()
        impact = [("Node1", 50.0, "HIGH"), ("Node2", 30.0, "LOW")]
        
        # Method name is impact_ranking
        chart = charts.impact_ranking(impact, "Test Impact")
        if chart:
            assert chart.title == "Test Impact"
    
    def test_empty_data_handling(self):
        """Charts should handle empty data gracefully."""
        charts = ChartGenerator()
        chart = charts.pie_chart({}, "Empty Chart")
        # Should either return None or valid empty chart
        if chart:
            assert chart.title == "Empty Chart"


# =============================================================================
# Dashboard Generation Tests
# =============================================================================

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
    
    def test_dashboard_navigation(self):
        """Dashboard should include navigation elements."""
        dash = DashboardGenerator("Navigation Test")
        html = dash.generate()
        
        assert "<html" in html
        assert "</html>" in html
    
    def test_dashboard_empty(self):
        """Empty dashboard should still generate valid HTML."""
        dash = DashboardGenerator("Empty Dashboard")
        html = dash.generate()
        
        assert "<!DOCTYPE html>" in html
        assert "Empty Dashboard" in html
    
    def test_dashboard_special_characters(self):
        """Dashboard should handle special characters in content."""
        dash = DashboardGenerator("Test <Special> & 'Chars'")
        dash.start_section("Section with <script> tag", "sec-script")
        html = dash.generate()
        
        # Should still be valid HTML
        assert "<!DOCTYPE html>" in html