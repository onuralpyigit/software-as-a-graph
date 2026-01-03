"""
Visualization Module - Version 5.0

Visualization capabilities for multi-layer graph analysis.

Features:
- Chart generation (bar, pie, heatmap, line charts)
- HTML dashboard generation
- Graph model visualization
- Simulation results visualization
- Validation results visualization
- Combined overview dashboards

Usage:
    from src.simulation import SimulationGraph, FailureSimulator
    from src.validation import ValidationPipeline
    from src.visualization import (
        generate_overview_dashboard,
        generate_simulation_dashboard,
        generate_validation_dashboard,
        chart_impact_ranking,
    )
    
    # Load graph
    graph = SimulationGraph.from_json("system.json")
    
    # Run simulation
    simulator = FailureSimulator(cascade=True)
    campaign = simulator.simulate_all(graph)
    
    # Run validation
    pipeline = ValidationPipeline(seed=42)
    validation = pipeline.run(graph, compare_methods=True)
    
    # Generate dashboards
    html = generate_overview_dashboard(
        graph,
        campaign_result=campaign,
        validation_result=validation
    )
    
    with open("dashboard.html", "w") as f:
        f.write(html)
    
    # Individual charts
    chart = chart_impact_ranking(campaign.ranked_by_impact())
    print(chart.to_html_img())

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

__version__ = "5.0.0"

# Charts
from .charts import (
    # Configuration
    ChartTheme,
    ChartConfig,
    ChartOutput,
    # Graph Statistics Charts
    chart_component_distribution,
    chart_edge_distribution,
    chart_layer_summary,
    # Analysis Charts
    chart_impact_ranking,
    chart_criticality_distribution,
    # Validation Charts
    chart_correlation_comparison,
    chart_confusion_matrix,
    chart_layer_validation,
    chart_method_comparison,
    # Simulation Charts
    chart_delivery_stats,
    chart_layer_performance,
    # Utilities
    check_matplotlib_available,
)

# Dashboard
from .dashboard import (
    # Configuration
    DashboardConfig,
    DashboardBuilder,
    # Dashboard Generators
    generate_graph_dashboard,
    generate_simulation_dashboard,
    generate_validation_dashboard,
    generate_overview_dashboard,
)


__all__ = [
    # Version
    "__version__",
    
    # Charts - Configuration
    "ChartTheme",
    "ChartConfig",
    "ChartOutput",
    
    # Charts - Graph Statistics
    "chart_component_distribution",
    "chart_edge_distribution",
    "chart_layer_summary",
    
    # Charts - Analysis
    "chart_impact_ranking",
    "chart_criticality_distribution",
    
    # Charts - Validation
    "chart_correlation_comparison",
    "chart_confusion_matrix",
    "chart_layer_validation",
    "chart_method_comparison",
    
    # Charts - Simulation
    "chart_delivery_stats",
    "chart_layer_performance",
    
    # Charts - Utilities
    "check_matplotlib_available",
    
    # Dashboard - Configuration
    "DashboardConfig",
    "DashboardBuilder",
    
    # Dashboard - Generators
    "generate_graph_dashboard",
    "generate_simulation_dashboard",
    "generate_validation_dashboard",
    "generate_overview_dashboard",
]
