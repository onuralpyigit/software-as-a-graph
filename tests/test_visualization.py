import pytest
import os
from src.visualization.charts import ChartGenerator
from src.visualization.dashboard import DashboardGenerator

def test_chart_generation():
    charts = ChartGenerator()
    data = {"A": 10, "B": 20}
    
    # Test Distribution Chart
    chart = charts.plot_distribution(data, "Test Pie")
    if chart: # Only if matplotlib is installed
        assert chart.title == "Test Pie"
        assert len(chart.png_base64) > 0

    # Test Impact Chart
    impact = {"Node1": 50, "Node2": 30}
    chart = charts.plot_impact_ranking(impact, "Test Impact")
    if chart:
        assert chart.title == "Test Impact"

def test_dashboard_generation():
    dash = DashboardGenerator("Test Dash")
    
    dash.add_section_header("Section 1")
    dash.add_kpis({"Metric 1": 100})
    dash.add_table(["Col1", "Col2"], [[1, "A"], [2, "B"]])
    dash.close_section()
    
    html = dash.generate()
    
    assert "<!DOCTYPE html>" in html
    assert "Test Dash" in html
    assert "Metric 1" in html
    assert "100" in html
    assert "<td>A</td>" in html