#!/usr/bin/env python3
"""
Visualization Demo Script

Demonstrates all features of the multi-layer graph visualization system.
Creates sample graphs and shows various visualization options.

Usage:
    python demo_visualization.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

import networkx as nx
import json
import logging

from src.visualization.graph_visualizer import (
    GraphVisualizer,
    VisualizationConfig,
    LayoutAlgorithm,
    ColorScheme
)

from src.visualization.layer_renderer import (
    LayerRenderer,
    LayerConfig,
    Layer
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_graph() -> nx.DiGraph:
    """
    Create a sample multi-layer pub-sub system graph
    
    Returns:
        Sample graph with Applications, Topics, Brokers, and Nodes
    """
    logger.info("Creating sample graph...")
    
    graph = nx.DiGraph()
    
    # Application Layer
    applications = [
        ('App1', 'TrafficController', 0.85),
        ('App2', 'SensorAggregator', 0.72),
        ('App3', 'DataLogger', 0.45),
        ('App4', 'AlertSystem', 0.68),
        ('App5', 'Dashboard', 0.30)
    ]
    
    for app_id, app_name, criticality in applications:
        graph.add_node(app_id, 
                      name=app_name,
                      type='Application',
                      criticality=criticality,
                      layer='application')
    
    # Topic Layer
    topics = [
        ('T1', '/traffic/sensors', 0.78),
        ('T2', '/traffic/control', 0.82),
        ('T3', '/system/logs', 0.35),
        ('T4', '/alerts/critical', 0.75),
        ('T5', '/dashboard/data', 0.40)
    ]
    
    for topic_id, topic_name, criticality in topics:
        graph.add_node(topic_id,
                      name=topic_name,
                      type='Topic',
                      criticality=criticality,
                      layer='topic',
                      qos={'reliability': 'RELIABLE', 'durability': 'PERSISTENT'})
    
    # Broker Layer
    brokers = [
        ('B1', 'MainBroker', 0.90),
        ('B2', 'BackupBroker', 0.65)
    ]
    
    for broker_id, broker_name, criticality in brokers:
        graph.add_node(broker_id,
                      name=broker_name,
                      type='Broker',
                      criticality=criticality,
                      layer='infrastructure',
                      location='DataCenter1')
    
    # Node Layer
    nodes = [
        ('N1', 'EdgeNode1', 0.55),
        ('N2', 'EdgeNode2', 0.50)
    ]
    
    for node_id, node_name, criticality in nodes:
        graph.add_node(node_id,
                      name=node_name,
                      type='Node',
                      criticality=criticality,
                      layer='infrastructure',
                      location='Edge')
    
    # Application Dependencies
    graph.add_edge('App1', 'App4', type='DEPENDS_ON', weight=1.0)
    graph.add_edge('App2', 'App3', type='DEPENDS_ON', weight=0.7)
    
    # Application -> Topic (Publishing)
    graph.add_edge('App1', 'T1', type='PUBLISHES', weight=1.0)
    graph.add_edge('App1', 'T2', type='PUBLISHES', weight=1.0)
    graph.add_edge('App2', 'T1', type='PUBLISHES', weight=0.8)
    graph.add_edge('App3', 'T3', type='PUBLISHES', weight=0.6)
    graph.add_edge('App4', 'T4', type='PUBLISHES', weight=0.9)
    
    # Topic -> Application (Subscribing)
    graph.add_edge('T1', 'App2', type='SUBSCRIBES', weight=0.8)
    graph.add_edge('T2', 'App1', type='SUBSCRIBES', weight=1.0)
    graph.add_edge('T4', 'App4', type='SUBSCRIBES', weight=0.9)
    graph.add_edge('T1', 'App5', type='SUBSCRIBES', weight=0.5)
    graph.add_edge('T2', 'App5', type='SUBSCRIBES', weight=0.5)
    
    # Topic -> Broker (Routing)
    graph.add_edge('T1', 'B1', type='ROUTES_THROUGH', weight=1.0)
    graph.add_edge('T2', 'B1', type='ROUTES_THROUGH', weight=1.0)
    graph.add_edge('T3', 'B2', type='ROUTES_THROUGH', weight=0.6)
    graph.add_edge('T4', 'B1', type='ROUTES_THROUGH', weight=0.9)
    graph.add_edge('T5', 'B2', type='ROUTES_THROUGH', weight=0.5)
    
    # Broker -> Node (Deployment)
    graph.add_edge('B1', 'N1', type='DEPLOYED_ON', weight=1.0)
    graph.add_edge('B2', 'N2', type='DEPLOYED_ON', weight=1.0)
    
    logger.info(f"✓ Created graph: {len(graph)} nodes, {len(graph.edges())} edges")
    
    return graph


def demo_basic_visualization(graph: nx.DiGraph, output_dir: Path):
    """Demo basic visualization features"""
    
    logger.info("\n" + "="*70)
    logger.info("DEMO 1: Basic Visualization")
    logger.info("="*70)
    
    visualizer = GraphVisualizer()
    
    # Extract criticality scores
    criticality_scores = {
        node: data['criticality'] 
        for node, data in graph.nodes(data=True)
    }
    
    # 1. Spring layout with criticality coloring
    logger.info("Creating spring layout visualization...")
    visualizer.visualize_graph(
        graph,
        str(output_dir / 'demo1_spring_criticality.png'),
        config=VisualizationConfig(
            layout=LayoutAlgorithm.SPRING,
            color_scheme=ColorScheme.CRITICALITY
        ),
        criticality_scores=criticality_scores,
        title='Spring Layout - Criticality Coloring'
    )
    
    # 2. Hierarchical layout with type coloring
    logger.info("Creating hierarchical layout visualization...")
    visualizer.visualize_graph(
        graph,
        str(output_dir / 'demo1_hierarchical_type.png'),
        config=VisualizationConfig(
            layout=LayoutAlgorithm.HIERARCHICAL,
            color_scheme=ColorScheme.TYPE
        ),
        title='Hierarchical Layout - Type Coloring'
    )
    
    # 3. Layered layout with layer coloring
    logger.info("Creating layered layout visualization...")
    visualizer.visualize_graph(
        graph,
        str(output_dir / 'demo1_layered_layer.png'),
        config=VisualizationConfig(
            layout=LayoutAlgorithm.LAYERED,
            color_scheme=ColorScheme.LAYER
        ),
        title='Layered Layout - Layer Coloring'
    )
    
    logger.info("✓ Demo 1 complete")


def demo_multi_layer_rendering(graph: nx.DiGraph, output_dir: Path):
    """Demo multi-layer rendering features"""
    
    logger.info("\n" + "="*70)
    logger.info("DEMO 2: Multi-Layer Rendering")
    logger.info("="*70)
    
    renderer = LayerRenderer()
    
    # 1. Individual layers
    logger.info("Rendering individual layers...")
    for layer in [Layer.APPLICATION, Layer.TOPIC, Layer.INFRASTRUCTURE]:
        renderer.render_layer(
            graph,
            layer,
            output_path=str(output_dir / f'demo2_{layer.value}_layer.html')
        )
    
    # 2. All layers composite
    logger.info("Rendering all layers composite...")
    renderer.render_all_layers(
        graph,
        output_path=str(output_dir / 'demo2_all_layers.html')
    )
    
    # 3. Layer statistics
    logger.info("Computing layer statistics...")
    stats = renderer.get_layer_statistics(graph)
    
    logger.info("\nLayer Statistics:")
    for layer, layer_stats in stats.items():
        if layer == 'cross_layer':
            logger.info(f"  Cross-Layer:")
            logger.info(f"    Total Dependencies: {layer_stats['total_dependencies']}")
        else:
            logger.info(f"  {layer.title()}:")
            logger.info(f"    Nodes: {layer_stats['node_count']}")
            logger.info(f"    Edges: {layer_stats['edge_count']}")
            logger.info(f"    Avg Degree: {layer_stats['avg_degree']:.2f}")
            logger.info(f"    Density: {layer_stats['density']:.3f}")
    
    logger.info("✓ Demo 2 complete")


def demo_interactive_features(graph: nx.DiGraph, output_dir: Path):
    """Demo interactive HTML features"""
    
    logger.info("\n" + "="*70)
    logger.info("DEMO 3: Interactive HTML Visualizations")
    logger.info("="*70)
    
    visualizer = GraphVisualizer()
    
    criticality_scores = {
        node: data['criticality'] 
        for node, data in graph.nodes(data=True)
    }
    
    # Create interactive visualization
    logger.info("Creating interactive HTML...")
    visualizer.create_interactive_html(
        graph,
        str(output_dir / 'demo3_interactive.html'),
        criticality_scores=criticality_scores,
        title='Interactive System Visualization'
    )
    
    logger.info("✓ Open demo3_interactive.html in your browser!")
    logger.info("  Features:")
    logger.info("  - Zoom and pan with mouse")
    logger.info("  - Hover over nodes for details")
    logger.info("  - Toggle physics simulation")
    logger.info("  - Export to PNG")
    logger.info("✓ Demo 3 complete")


def demo_failure_analysis(graph: nx.DiGraph, output_dir: Path):
    """Demo failure impact visualization"""
    
    logger.info("\n" + "="*70)
    logger.info("DEMO 4: Failure Impact Analysis")
    logger.info("="*70)
    
    visualizer = GraphVisualizer()
    
    # Simulate failure of critical broker
    failed_node = 'B1'
    logger.info(f"Simulating failure of {failed_node}...")
    
    failed_graph = graph.copy()
    failed_graph.remove_node(failed_node)
    
    # Visualize impact
    logger.info("Creating failure impact visualization...")
    visualizer.visualize_failure_impact(
        baseline_graph=graph,
        failed_graph=failed_graph,
        failed_node=failed_node,
        output_path=str(output_dir / f'demo4_failure_impact_{failed_node}.png')
    )
    
    logger.info(f"✓ Failure analysis shows impact of losing {failed_node}")
    logger.info(f"  Baseline: {len(graph)} nodes")
    logger.info(f"  After failure: {len(failed_graph)} nodes")
    logger.info(f"  Components lost: {len(graph) - len(failed_graph)}")
    logger.info("✓ Demo 4 complete")


def demo_hidden_dependencies(graph: nx.DiGraph, output_dir: Path):
    """Demo hidden dependency visualization"""
    
    logger.info("\n" + "="*70)
    logger.info("DEMO 5: Hidden Dependencies")
    logger.info("="*70)
    
    visualizer = GraphVisualizer()
    
    # Simulate hidden dependencies
    hidden_deps = [
        ('App1', 'App3', 0.8),  # App1 indirectly depends on App3 through shared topic
        ('App2', 'App4', 0.6),  # App2 indirectly affects App4
    ]
    
    logger.info(f"Identified {len(hidden_deps)} hidden dependencies:")
    for source, target, strength in hidden_deps:
        logger.info(f"  {source} → {target} (strength: {strength:.2f})")
    
    # Visualize
    logger.info("Creating hidden dependencies visualization...")
    visualizer.visualize_hidden_dependencies(
        graph,
        hidden_deps,
        str(output_dir / 'demo5_hidden_dependencies.png')
    )
    
    logger.info("✓ Demo 5 complete")


def export_sample_json(graph: nx.DiGraph, output_path: Path):
    """Export graph to JSON format"""
    
    logger.info("\nExporting sample graph to JSON...")
    
    # Convert to JSON format
    data = {
        'metadata': {
            'name': 'Demo Smart City System',
            'description': 'Sample multi-layer pub-sub system',
            'version': '1.0',
            'layers': ['application', 'topic', 'infrastructure']
        },
        'nodes': [],
        'edges': []
    }
    
    # Add nodes
    for node, node_data in graph.nodes(data=True):
        node_dict = {'id': node}
        node_dict.update(node_data)
        data['nodes'].append(node_dict)
    
    # Add edges
    for source, target, edge_data in graph.edges(data=True):
        edge_dict = {'source': source, 'target': target}
        edge_dict.update(edge_data)
        data['edges'].append(edge_dict)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"✓ Saved sample JSON: {output_path}")


def main():
    """Main demo function"""
    
    print("\n" + "="*70)
    print("Multi-Layer Graph Visualization System - Demo")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = Path('demo_output')
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Create sample graph
    graph = create_sample_graph()
    
    # Export to JSON
    json_path = output_dir / 'sample_system.json'
    export_sample_json(graph, json_path)
    
    # Run demos
    try:
        demo_basic_visualization(graph, output_dir)
        demo_multi_layer_rendering(graph, output_dir)
        demo_interactive_features(graph, output_dir)
        demo_failure_analysis(graph, output_dir)
        demo_hidden_dependencies(graph, output_dir)
        
        # Summary
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        print(f"\nGenerated files in: {output_dir.absolute()}")
        print("\nGenerated visualizations:")
        
        for file in sorted(output_dir.glob('demo*.png')):
            print(f"  • {file.name}")
        
        print("\nInteractive HTML files:")
        for file in sorted(output_dir.glob('demo*.html')):
            print(f"  • {file.name}")
        
        print(f"\nSample JSON: {json_path.name}")
        
        print("\nNext steps:")
        print("  1. Open HTML files in your web browser")
        print("  2. View PNG images in image viewer")
        print("  3. Use sample_system.json with visualize_graph.py:")
        print(f"     python visualize_graph.py --input {json_path} --all")
        
        print("\n" + "="*70 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
