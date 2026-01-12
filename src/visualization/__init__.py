"""
Visualization Module

Generates interactive dashboards for multi-layer graph analysis.
Visualizes graph statistics, analysis results, simulation outcomes,
and validation metrics.

Features:
    - Multi-layer analysis (app, infra, mw-app, mw-infra, system)
    - Interactive vis.js network graphs
    - Matplotlib-based charts (bar, pie, scatter, heatmap)
    - Responsive HTML dashboards
    - KPI cards, tables, and metrics displays

Layers:
    - app: Application layer (Application components)
    - infra: Infrastructure layer (Node components)
    - mw-app: Middleware-Application (Applications + Brokers)
    - mw-infra: Middleware-Infrastructure (Nodes + Brokers)
    - system: Complete system (all components)

Example:
    >>> from src.visualization import GraphVisualizer
    >>> 
    >>> with GraphVisualizer(uri="bolt://localhost:7687") as viz:
    ...     viz.generate_dashboard(
    ...         output_file="dashboard.html",
    ...         layers=["app", "infra", "system"]
    ...     )
"""

# Charts
from .charts import (
    ChartGenerator,
    ChartOutput,
    COLORS,
    CRITICALITY_COLORS,
    LAYER_COLORS,
)

# Dashboard
from .dashboard import (
    DashboardGenerator,
    NavLink,
)

# Visualizer
from .visualizer import (
    GraphVisualizer,
    LayerData,
    LAYER_DEFINITIONS,
)


__all__ = [
    # Charts
    "ChartGenerator",
    "ChartOutput",
    "COLORS",
    "CRITICALITY_COLORS",
    "LAYER_COLORS",
    
    # Dashboard
    "DashboardGenerator",
    "NavLink",
    
    # Visualizer
    "GraphVisualizer",
    "LayerData",
    "LAYER_DEFINITIONS",
]

__version__ = "2.0.0"