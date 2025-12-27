"""
Visualization Module - Version 4.0

Multi-layer graph visualization and dashboard generation for
pub-sub system analysis results.

Features:
- Interactive graph visualization with vis.js
- Multi-layer architecture views
- Comprehensive dashboards with Chart.js
- Criticality-based coloring
- Statistics and metrics visualization

Usage:
    from src.simulation import SimulationGraph
    from src.visualization import GraphRenderer, DashboardGenerator
    
    # Load graph
    graph = SimulationGraph.from_json("system.json")
    
    # Generate interactive graph
    renderer = GraphRenderer()
    html = renderer.render(graph)
    
    # Generate multi-layer view
    html = renderer.render_multi_layer(graph)
    
    # Generate dashboard
    dashboard = DashboardGenerator()
    html = dashboard.generate(
        graph=graph,
        criticality=criticality_scores,
        validation=validation_results,
        simulation=simulation_results,
    )

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from .graph_renderer import (
    # Classes
    GraphRenderer,
    RenderConfig,
    NodeData,
    EdgeData,
    # Enums
    Layer,
    LayoutAlgorithm,
    ColorScheme,
    # Constants
    COLORS,
    SHAPES,
    SIZES,
)

from .dashboard import (
    DashboardGenerator,
    DashboardConfig,
)

__all__ = [
    # Graph Renderer
    "GraphRenderer",
    "RenderConfig",
    "NodeData",
    "EdgeData",
    # Dashboard
    "DashboardGenerator",
    "DashboardConfig",
    # Enums
    "Layer",
    "LayoutAlgorithm",
    "ColorScheme",
    # Constants
    "COLORS",
    "SHAPES",
    "SIZES",
]

__version__ = "4.0.0"