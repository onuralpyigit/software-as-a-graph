"""
Example: Visualization Modules Usage

Demonstrates:
1. Interactive graph visualization
2. Layer-specific rendering
3. Metrics dashboard creation
4. Complete visualization workflow
"""

import sys
from pathlib import Path
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent / '..'))

from src.visualization.graph_visualizer import (
    GraphVisualizer, VisualizationConfig, LayoutAlgorithm, ColorScheme
)
from src.visualization.layer_renderer import (
    LayerRenderer, Layer, LayerConfig
)
from src.visualization.metrics_dashboard import (
    MetricsDashboard, MetricData
)
from datetime import datetime


def create_example_system():
    """Create example system for visualization"""
    
    G = nx.DiGraph()
    
    # Add components with metadata
    components = {
        # Applications
        'WebApp': {'type': 'Application', 'criticality_score': 0.7, 'criticality_level': 'HIGH'},
        'MobileApp': {'type': 'Application', 'criticality_score': 0.6, 'criticality_level': 'MEDIUM'},
        'AdminPortal': {'type': 'Application', 'criticality_score': 0.5, 'criticality_level': 'MEDIUM'},
        
        # Infrastructure
        'MainBroker': {'type': 'Broker', 'criticality_score': 0.9, 'criticality_level': 'CRITICAL'},
        'BackupBroker': {'type': 'Broker', 'criticality_score': 0.7, 'criticality_level': 'HIGH'},
        'LoadBalancer': {'type': 'Node', 'criticality_score': 0.8, 'criticality_level': 'HIGH'},
        'CacheServer': {'type': 'Node', 'criticality_score': 0.6, 'criticality_level': 'MEDIUM'},
        
        # Topics
        'UserEvents': {'type': 'Topic', 'criticality_score': 0.7, 'criticality_level': 'HIGH'},
        'OrderEvents': {'type': 'Topic', 'criticality_score': 0.8, 'criticality_level': 'HIGH'},
        'NotificationEvents': {'type': 'Topic', 'criticality_score': 0.5, 'criticality_level': 'MEDIUM'},
    }
    
    for name, attrs in components.items():
        G.add_node(name, **attrs)
    
    # Add edges with weights
    edges = [
        ('WebApp', 'LoadBalancer', 10),
        ('MobileApp', 'LoadBalancer', 8),
        ('AdminPortal', 'LoadBalancer', 5),
        ('LoadBalancer', 'MainBroker', 15),
        ('MainBroker', 'BackupBroker', 12),
        ('MainBroker', 'UserEvents', 8),
        ('MainBroker', 'OrderEvents', 10),
        ('MainBroker', 'NotificationEvents', 6),
        ('CacheServer', 'MainBroker', 5),
        ('OrderEvents', 'WebApp', 7),
        ('UserEvents', 'MobileApp', 6),
        ('NotificationEvents', 'AdminPortal', 4),
    ]
    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    return G


def example_1_basic_visualization():
    """Example 1: Basic graph visualization"""
    
    print("\n" + "=" * 70)
    print("Example 1: Basic Graph Visualization")
    print("=" * 70)
    
    # Create graph
    print("\n[Step 1] Creating example system...")
    G = create_example_system()
    print(f"  ✓ System with {len(G)} components")
    
    # Initialize visualizer
    visualizer = GraphVisualizer()
    
    # Create basic visualization
    print("\n[Step 2] Creating interactive visualization...")
    config = VisualizationConfig(
        layout=LayoutAlgorithm.SPRING,
        color_scheme=ColorScheme.TYPE,
        show_labels=True,
        width=1200,
        height=800
    )
    
    html = visualizer.visualize(G, config, output_path='output/graph_basic.html')
    print(f"  ✓ Saved to: graph_basic.html")
    
    # Try different layouts
    print("\n[Step 3] Creating hierarchical layout...")
    config.layout = LayoutAlgorithm.HIERARCHICAL
    visualizer.visualize(G, config, output_path='output/graph_hierarchical.html')
    print(f"  ✓ Saved to: graph_hierarchical.html")
    
    print("\n[Step 4] Creating circular layout...")
    config.layout = LayoutAlgorithm.CIRCULAR
    visualizer.visualize(G, config, output_path='output/graph_circular.html')
    print(f"  ✓ Saved to: graph_circular.html")


def example_2_matplotlib_visualization():
    """Example 2: Static visualization with matplotlib"""
    
    print("\n" + "=" * 70)
    print("Example 2: Matplotlib Visualization")
    print("=" * 70)
    
    G = create_example_system()
    visualizer = GraphVisualizer()
    
    print("\n[Creating static visualization with matplotlib...]")
    
    try:
        config = VisualizationConfig(
            layout=LayoutAlgorithm.SPRING,
            color_scheme=ColorScheme.CRITICALITY,
            show_labels=True,
            interactive=False
        )
        
        visualizer.visualize_with_matplotlib(
            G, 
            config, 
            output_path='output/graph_static.png'
        )
        print(f"  ✓ Saved to: graph_static.png")
        
    except ImportError:
        print("  ⚠️  matplotlib not installed. Skipping...")


def example_3_highlighted_visualization():
    """Example 3: Visualization with highlighting"""
    
    print("\n" + "=" * 70)
    print("Example 3: Highlighted Visualization")
    print("=" * 70)
    
    G = create_example_system()
    visualizer = GraphVisualizer()
    
    print("\n[Step 1] Highlighting critical components...")
    
    # Find critical components
    critical_nodes = [
        node for node in G.nodes()
        if G.nodes[node].get('criticality_level') == 'CRITICAL'
    ]
    
    config = VisualizationConfig(
        layout=LayoutAlgorithm.SPRING,
        color_scheme=ColorScheme.TYPE,
        highlight_nodes=critical_nodes,
        show_labels=True
    )
    
    visualizer.visualize(G, config, output_path='output/graph_highlighted.html')
    print(f"  ✓ Highlighted {len(critical_nodes)} critical components")
    print(f"  ✓ Saved to: graph_highlighted.html")
    
    # Visualize critical path
    print("\n[Step 2] Visualizing critical path...")
    html = visualizer.visualize_critical_path(
        G,
        'WebApp',
        'OrderEvents',
        config
    )
    Path('output/graph_critical_path.html').write_text(html)
    print(f"  ✓ Saved to: graph_critical_path.html")


def example_4_neighborhood_view():
    """Example 4: Neighborhood visualization"""
    
    print("\n" + "=" * 70)
    print("Example 4: Neighborhood Visualization")
    print("=" * 70)
    
    G = create_example_system()
    visualizer = GraphVisualizer()
    
    print("\n[Visualizing neighborhood around MainBroker...]")
    
    html = visualizer.visualize_neighborhood(
        G,
        'MainBroker',
        radius=2
    )
    
    Path('output/graph_neighborhood.html').write_text(html)
    print(f"  ✓ Saved to: graph_neighborhood.html")


def example_5_layer_visualization():
    """Example 5: Layer-specific visualization"""
    
    print("\n" + "=" * 70)
    print("Example 5: Layer-Specific Visualization")
    print("=" * 70)
    
    G = create_example_system()
    renderer = LayerRenderer()
    
    # Render application layer
    print("\n[Step 1] Rendering application layer...")
    renderer.render_layer(
        G,
        Layer.APPLICATION,
        output_path='output/layer_application.html'
    )
    print(f"  ✓ Saved to: layer_application.html")
    
    # Render infrastructure layer
    print("\n[Step 2] Rendering infrastructure layer...")
    renderer.render_layer(
        G,
        Layer.INFRASTRUCTURE,
        output_path='output/layer_infrastructure.html'
    )
    print(f"  ✓ Saved to: layer_infrastructure.html")
    
    # Render topic layer
    print("\n[Step 3] Rendering topic layer...")
    renderer.render_layer(
        G,
        Layer.TOPIC,
        output_path='output/layer_topic.html'
    )
    print(f"  ✓ Saved to: layer_topic.html")
    
    # Get layer statistics
    print("\n[Step 4] Layer statistics:")
    stats = renderer.get_layer_statistics(G)
    for layer, layer_stats in stats.items():
        if layer != 'cross_layer':
            print(f"\n  {layer.title()}:")
            print(f"    Nodes: {layer_stats['node_count']}")
            print(f"    Edges: {layer_stats['edge_count']}")
            print(f"    Avg Degree: {layer_stats['avg_degree']:.2f}")


def example_6_multi_layer_view():
    """Example 6: Multi-layer composite view"""
    
    print("\n" + "=" * 70)
    print("Example 6: Multi-Layer View")
    print("=" * 70)
    
    G = create_example_system()
    renderer = LayerRenderer()
    
    print("\n[Creating multi-layer composite view...]")
    
    renderer.render_all_layers(
        G,
        output_path='output/layers_all.html'
    )
    print(f"  ✓ Saved to: layers_all.html")


def example_7_layer_interactions():
    """Example 7: Layer interaction visualization"""
    
    print("\n" + "=" * 70)
    print("Example 7: Layer Interactions")
    print("=" * 70)
    
    G = create_example_system()
    renderer = LayerRenderer()
    
    print("\n[Visualizing application -> infrastructure interactions...]")
    
    renderer.render_layer_interactions(
        G,
        Layer.APPLICATION,
        Layer.INFRASTRUCTURE,
        output_path='output/layer_interactions.html'
    )
    print(f"  ✓ Saved to: layer_interactions.html")


def example_8_metrics_dashboard():
    """Example 8: Metrics dashboard"""
    
    print("\n" + "=" * 70)
    print("Example 8: Metrics Dashboard")
    print("=" * 70)
    
    G = create_example_system()
    dashboard = MetricsDashboard()
    
    print("\n[Step 1] Creating metrics dashboard...")
    
    # Create dashboard
    html = dashboard.create_dashboard(
        G,
        output_path='output/dashboard.html'
    )
    print(f"  ✓ Saved to: dashboard.html")
    
    # Get health metrics
    print("\n[Step 2] System health metrics:")
    health = dashboard.get_system_health(G)
    print(f"  Overall Health: {health['overall_health']}%")
    print(f"  Status: {health['status']}")
    print(f"  Connectivity: {health['connectivity']}%")
    print(f"  Components: {health['components']}%")
    print(f"  Redundancy: {health['redundancy']}%")


def example_9_complete_workflow():
    """Example 9: Complete visualization workflow"""
    
    print("\n" + "=" * 70)
    print("Example 9: Complete Visualization Workflow")
    print("=" * 70)
    
    # Create system
    print("\n[Step 1] Creating system...")
    G = create_example_system()
    print(f"  ✓ {len(G)} components, {len(G.edges())} connections")
    
    # Create graph visualization
    print("\n[Step 2] Creating graph visualization...")
    visualizer = GraphVisualizer()
    config = VisualizationConfig(
        layout=LayoutAlgorithm.SPRING,
        color_scheme=ColorScheme.CRITICALITY,
        show_labels=True
    )
    visualizer.visualize(G, config, 'output/complete_graph.html')
    print(f"  ✓ Graph visualization ready")
    
    # Create layer views
    print("\n[Step 3] Creating layer views...")
    renderer = LayerRenderer()
    renderer.render_all_layers(G, 'output/complete_layers.html')
    print(f"  ✓ Layer views ready")
    
    # Create dashboard
    print("\n[Step 4] Creating metrics dashboard...")
    dashboard = MetricsDashboard()
    dashboard.create_dashboard(G, output_path='output/complete_dashboard.html')
    print(f"  ✓ Dashboard ready")
    
    print("\n[Step 5] Complete visualization package created!")
    print(f"  Files saved to output/ directory:")
    print(f"    - complete_graph.html (interactive graph)")
    print(f"    - complete_layers.html (layer views)")
    print(f"    - complete_dashboard.html (metrics dashboard)")


def main():
    """Run all examples"""
    
    print("\n" + "=" * 70)
    print("VISUALIZATION MODULES - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    try:
        example_1_basic_visualization()
        example_2_matplotlib_visualization()
        example_3_highlighted_visualization()
        example_4_neighborhood_view()
        example_5_layer_visualization()
        example_6_multi_layer_view()
        example_7_layer_interactions()
        example_8_metrics_dashboard()
        example_9_complete_workflow()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n📚 Summary of Capabilities:")
        print("  ✓ Interactive graph visualization (multiple layouts)")
        print("  ✓ Static image generation (matplotlib)")
        print("  ✓ Node/edge highlighting")
        print("  ✓ Critical path visualization")
        print("  ✓ Neighborhood views")
        print("  ✓ Layer-specific rendering")
        print("  ✓ Multi-layer composite views")
        print("  ✓ Layer interaction analysis")
        print("  ✓ Metrics dashboards")
        print("  ✓ System health monitoring")
        
        print("\n📖 Usage in Your Code:")
        print("""
from refactored.visualization.graph_visualizer import GraphVisualizer, VisualizationConfig
from refactored.visualization.layer_renderer import LayerRenderer, Layer
from refactored.visualization.metrics_dashboard import MetricsDashboard

# Graph visualization
visualizer = GraphVisualizer()
config = VisualizationConfig(layout=LayoutAlgorithm.SPRING)
visualizer.visualize(graph, config, 'output.html')

# Layer rendering
renderer = LayerRenderer()
renderer.render_layer(graph, Layer.APPLICATION, output_path='app_layer.html')

# Metrics dashboard
dashboard = MetricsDashboard()
dashboard.create_dashboard(graph, output_path='dashboard.html')
        """)
        
        print("\n📁 Generated Files:")
        print("  Check the output/ directory for all generated HTML files")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
