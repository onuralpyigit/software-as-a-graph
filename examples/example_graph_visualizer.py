#!/usr/bin/env python3
"""
Visualization Examples for Pub-Sub System Analysis
====================================================

Demonstrates comprehensive visualization capabilities:
1. Basic HTML visualization
2. Multi-layer architecture view
3. Criticality-based coloring
4. Dashboard generation
5. Export for external tools

Usage:
    python examples/visualization_examples.py
    python examples/visualization_examples.py --example basic
    python examples/visualization_examples.py --output-dir ./output

Author: Software-as-a-Graph Research Project
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from src.visualization import (
    GraphVisualizer,
    DashboardGenerator,
    VisualizationConfig,
    DashboardConfig,
    Layer,
    LayoutAlgorithm,
    ColorScheme,
    MATPLOTLIB_AVAILABLE
)


# ============================================================================
# Terminal Formatting
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(title: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{title.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def print_section(title: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.ENDC}")
    print("-" * 50)


def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.ENDC} {msg}")


# ============================================================================
# Test System Generation
# ============================================================================

def create_iot_smart_city():
    """Create IoT smart city pub-sub system"""
    G = nx.DiGraph()
    
    # Infrastructure layer - Edge nodes and gateways
    infra = [
        ('edge_north', 'Node', 'Edge Node North'),
        ('edge_south', 'Node', 'Edge Node South'),
        ('edge_east', 'Node', 'Edge Node East'),
        ('gateway_main', 'Node', 'Main Gateway'),
        ('cloud_server', 'Node', 'Cloud Server'),
    ]
    
    # Broker layer
    brokers = [
        ('mqtt_broker', 'Broker', 'MQTT Broker'),
        ('kafka_cluster', 'Broker', 'Kafka Cluster'),
        ('redis_pubsub', 'Broker', 'Redis PubSub'),
    ]
    
    # Topic layer
    topics = [
        ('traffic/sensors', 'Topic', 'Traffic Sensors'),
        ('traffic/lights', 'Topic', 'Traffic Lights'),
        ('environment/air', 'Topic', 'Air Quality'),
        ('environment/noise', 'Topic', 'Noise Levels'),
        ('parking/status', 'Topic', 'Parking Status'),
        ('alerts/emergency', 'Topic', 'Emergency Alerts'),
        ('analytics/realtime', 'Topic', 'Real-time Analytics'),
    ]
    
    # Application layer
    apps = [
        ('traffic_sensor_1', 'Application', 'Traffic Sensor 1'),
        ('traffic_sensor_2', 'Application', 'Traffic Sensor 2'),
        ('air_monitor', 'Application', 'Air Quality Monitor'),
        ('noise_monitor', 'Application', 'Noise Monitor'),
        ('parking_sensor', 'Application', 'Parking Sensor'),
        ('traffic_controller', 'Application', 'Traffic Controller'),
        ('env_aggregator', 'Application', 'Environment Aggregator'),
        ('alert_service', 'Application', 'Alert Service'),
        ('analytics_engine', 'Application', 'Analytics Engine'),
        ('city_dashboard', 'Application', 'City Dashboard'),
        ('mobile_app', 'Application', 'Mobile App'),
    ]
    
    # Add all nodes
    for node_id, node_type, name in infra + brokers + topics + apps:
        G.add_node(node_id, type=node_type, name=name)
    
    # Infrastructure connections
    edges = [
        # Brokers on infrastructure
        ('mqtt_broker', 'edge_north', 'RUNS_ON'),
        ('mqtt_broker', 'edge_south', 'RUNS_ON'),
        ('kafka_cluster', 'gateway_main', 'RUNS_ON'),
        ('redis_pubsub', 'cloud_server', 'RUNS_ON'),
        
        # Gateway connections
        ('edge_north', 'gateway_main', 'CONNECTS_TO'),
        ('edge_south', 'gateway_main', 'CONNECTS_TO'),
        ('edge_east', 'gateway_main', 'CONNECTS_TO'),
        ('gateway_main', 'cloud_server', 'CONNECTS_TO'),
        
        # Topics on brokers
        ('traffic/sensors', 'mqtt_broker', 'DEPENDS_ON'),
        ('traffic/lights', 'mqtt_broker', 'DEPENDS_ON'),
        ('environment/air', 'mqtt_broker', 'DEPENDS_ON'),
        ('environment/noise', 'mqtt_broker', 'DEPENDS_ON'),
        ('parking/status', 'mqtt_broker', 'DEPENDS_ON'),
        ('alerts/emergency', 'kafka_cluster', 'DEPENDS_ON'),
        ('analytics/realtime', 'kafka_cluster', 'DEPENDS_ON'),
        
        # Publishers
        ('traffic_sensor_1', 'traffic/sensors', 'PUBLISHES_TO'),
        ('traffic_sensor_2', 'traffic/sensors', 'PUBLISHES_TO'),
        ('air_monitor', 'environment/air', 'PUBLISHES_TO'),
        ('noise_monitor', 'environment/noise', 'PUBLISHES_TO'),
        ('parking_sensor', 'parking/status', 'PUBLISHES_TO'),
        ('alert_service', 'alerts/emergency', 'PUBLISHES_TO'),
        ('analytics_engine', 'analytics/realtime', 'PUBLISHES_TO'),
        
        # Subscribers
        ('traffic/sensors', 'traffic_controller', 'SUBSCRIBES_TO'),
        ('traffic/sensors', 'analytics_engine', 'SUBSCRIBES_TO'),
        ('traffic/lights', 'city_dashboard', 'SUBSCRIBES_TO'),
        ('environment/air', 'env_aggregator', 'SUBSCRIBES_TO'),
        ('environment/noise', 'env_aggregator', 'SUBSCRIBES_TO'),
        ('parking/status', 'city_dashboard', 'SUBSCRIBES_TO'),
        ('parking/status', 'mobile_app', 'SUBSCRIBES_TO'),
        ('alerts/emergency', 'city_dashboard', 'SUBSCRIBES_TO'),
        ('alerts/emergency', 'mobile_app', 'SUBSCRIBES_TO'),
        ('analytics/realtime', 'city_dashboard', 'SUBSCRIBES_TO'),
        
        # Controller outputs
        ('traffic_controller', 'traffic/lights', 'PUBLISHES_TO'),
        ('env_aggregator', 'alerts/emergency', 'PUBLISHES_TO'),
    ]
    
    for source, target, edge_type in edges:
        G.add_edge(source, target, type=edge_type)
    
    return G


def calculate_criticality(graph: nx.DiGraph) -> dict:
    """Calculate criticality scores"""
    criticality = {}
    
    bc = nx.betweenness_centrality(graph)
    max_bc = max(bc.values()) if bc.values() else 1
    
    try:
        aps = set(nx.articulation_points(graph.to_undirected()))
    except:
        aps = set()
    
    degrees = dict(graph.degree())
    max_deg = max(degrees.values()) if degrees.values() else 1
    
    for node in graph.nodes():
        bc_score = bc.get(node, 0) / max_bc if max_bc > 0 else 0
        ap_score = 1.0 if node in aps else 0.0
        
        try:
            descendants = len(nx.descendants(graph, node))
            ancestors = len(nx.ancestors(graph, node))
            impact = (descendants + ancestors) / (2 * graph.number_of_nodes())
        except:
            impact = 0
        
        degree_score = degrees.get(node, 0) / max_deg
        
        score = 0.25 * bc_score + 0.30 * ap_score + 0.25 * impact + 0.20 * degree_score
        score = min(1.0, score)
        
        if score >= 0.7:
            level = 'critical'
        elif score >= 0.5:
            level = 'high'
        elif score >= 0.3:
            level = 'medium'
        elif score >= 0.1:
            level = 'low'
        else:
            level = 'minimal'
        
        criticality[node] = {
            'score': score,
            'level': level,
            'is_articulation_point': node in aps,
            'betweenness': bc_score,
            'degree': degrees.get(node, 0)
        }
    
    return criticality


# ============================================================================
# Example Functions
# ============================================================================

def example_basic_html(output_dir: Path):
    """Generate basic HTML visualization"""
    print_header("EXAMPLE: Basic HTML Visualization")
    
    graph = create_iot_smart_city()
    print(f"System: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    visualizer = GraphVisualizer()
    html = visualizer.render_html(graph, title="IoT Smart City System")
    
    output_path = output_dir / "basic_visualization.html"
    with open(output_path, 'w') as f:
        f.write(html)
    
    print_success(f"Saved to {output_path}")
    return output_path


def example_multi_layer(output_dir: Path):
    """Generate multi-layer visualization"""
    print_header("EXAMPLE: Multi-Layer Visualization")
    
    graph = create_iot_smart_city()
    criticality = calculate_criticality(graph)
    
    print_section("Layer Statistics")
    visualizer = GraphVisualizer()
    visualizer.classify_layers(graph)
    stats = visualizer.get_layer_statistics(graph)
    
    for layer_name, layer_stats in stats.get('layers', {}).items():
        print(f"\n  {layer_name.title()}:")
        print(f"    Nodes: {layer_stats.get('node_count', 0)}")
        print(f"    Internal edges: {layer_stats.get('internal_edges', 0)}")
        print(f"    Cross-layer: {layer_stats.get('cross_layer_edges', 0)}")
    
    html = visualizer.render_multi_layer_html(
        graph, criticality,
        title="IoT Smart City - Multi-Layer Architecture"
    )
    
    output_path = output_dir / "multi_layer.html"
    with open(output_path, 'w') as f:
        f.write(html)
    
    print_success(f"Saved to {output_path}")
    return output_path


def example_criticality_view(output_dir: Path):
    """Generate criticality-colored visualization"""
    print_header("EXAMPLE: Criticality-Based Coloring")
    
    graph = create_iot_smart_city()
    criticality = calculate_criticality(graph)
    
    # Count by level
    level_counts = {}
    for crit in criticality.values():
        level = crit.get('level', 'minimal')
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print_section("Criticality Distribution")
    for level in ['critical', 'high', 'medium', 'low', 'minimal']:
        count = level_counts.get(level, 0)
        print(f"  {level.title():10s}: {count}")
    
    config = VisualizationConfig(
        title="IoT Smart City - Criticality View",
        color_scheme=ColorScheme.CRITICALITY
    )
    visualizer = GraphVisualizer(config)
    
    html = visualizer.render_html(graph, criticality)
    
    output_path = output_dir / "criticality_view.html"
    with open(output_path, 'w') as f:
        f.write(html)
    
    print_success(f"Saved to {output_path}")
    return output_path


def example_layer_views(output_dir: Path):
    """Generate individual layer views"""
    print_header("EXAMPLE: Individual Layer Views")
    
    graph = create_iot_smart_city()
    criticality = calculate_criticality(graph)
    
    visualizer = GraphVisualizer()
    
    layers = [
        (Layer.APPLICATION, "Application Layer"),
        (Layer.TOPIC, "Topic Layer"),
        (Layer.BROKER, "Broker Layer"),
        (Layer.INFRASTRUCTURE, "Infrastructure Layer")
    ]
    
    outputs = []
    for layer, title in layers:
        html = visualizer.render_html(
            graph, criticality,
            title=f"IoT Smart City - {title}",
            layer=layer
        )
        
        output_path = output_dir / f"layer_{layer.value}.html"
        with open(output_path, 'w') as f:
            f.write(html)
        
        print_success(f"{title} saved to {output_path}")
        outputs.append(output_path)
    
    return outputs


def example_layouts(output_dir: Path):
    """Generate visualizations with different layouts"""
    print_header("EXAMPLE: Different Layout Algorithms")
    
    graph = create_iot_smart_city()
    
    layouts = [
        (LayoutAlgorithm.SPRING, "Force-Directed (Spring)"),
        (LayoutAlgorithm.CIRCULAR, "Circular"),
        (LayoutAlgorithm.SHELL, "Shell (by Type)"),
        (LayoutAlgorithm.HIERARCHICAL, "Hierarchical")
    ]
    
    outputs = []
    for layout, name in layouts:
        config = VisualizationConfig(
            title=f"IoT Smart City - {name} Layout",
            layout=layout
        )
        visualizer = GraphVisualizer(config)
        
        html = visualizer.render_html(graph)
        
        output_path = output_dir / f"layout_{layout.value}.html"
        with open(output_path, 'w') as f:
            f.write(html)
        
        print_success(f"{name} layout saved to {output_path}")
        outputs.append(output_path)
    
    return outputs


def example_dashboard(output_dir: Path):
    """Generate comprehensive dashboard"""
    print_header("EXAMPLE: Comprehensive Dashboard")
    
    graph = create_iot_smart_city()
    criticality = calculate_criticality(graph)
    
    # Mock validation results
    validation = {
        'status': 'passed',
        'correlation': {
            'spearman': {'coefficient': 0.85}
        },
        'classification': {
            'overall': {
                'f1_score': 0.92,
                'precision': 0.88,
                'recall': 0.95
            }
        }
    }
    
    # Mock simulation results
    simulation = {
        'total_simulations': 25,
        'results': [
            {'primary_failures': ['kafka_cluster'], 'impact_score': 0.85, 'affected_nodes': 8},
            {'primary_failures': ['mqtt_broker'], 'impact_score': 0.72, 'affected_nodes': 12},
            {'primary_failures': ['gateway_main'], 'impact_score': 0.65, 'affected_nodes': 6},
            {'primary_failures': ['analytics_engine'], 'impact_score': 0.45, 'affected_nodes': 4},
            {'primary_failures': ['traffic_controller'], 'impact_score': 0.38, 'affected_nodes': 3},
        ]
    }
    
    config = DashboardConfig(
        title="IoT Smart City Analysis Dashboard",
        subtitle="Comprehensive System Analysis Report",
        theme="dark"
    )
    generator = DashboardGenerator(config)
    
    html = generator.generate(
        graph=graph,
        criticality=criticality,
        validation=validation,
        simulation=simulation
    )
    
    output_path = output_dir / "dashboard.html"
    with open(output_path, 'w') as f:
        f.write(html)
    
    print_success(f"Dashboard saved to {output_path}")
    return output_path


def example_export(output_dir: Path):
    """Export for external tools"""
    print_header("EXAMPLE: Export for External Tools")
    
    graph = create_iot_smart_city()
    criticality = calculate_criticality(graph)
    
    visualizer = GraphVisualizer()
    visualizer.classify_layers(graph)
    
    # Export for D3.js
    d3_path = output_dir / "graph_d3.json"
    visualizer.export_for_d3(graph, str(d3_path), criticality)
    print_success(f"D3.js format saved to {d3_path}")
    
    # Export for Gephi
    gephi_path = output_dir / "graph_gephi.gexf"
    visualizer.export_for_gephi(graph, str(gephi_path), criticality)
    print_success(f"Gephi format saved to {gephi_path}")
    
    return [d3_path, gephi_path]


def example_static_image(output_dir: Path):
    """Generate static images"""
    print_header("EXAMPLE: Static Image Export")
    
    if not MATPLOTLIB_AVAILABLE:
        print(f"{Colors.WARNING}⚠ Matplotlib not available. Skipping image export.{Colors.ENDC}")
        print("  Install with: pip install matplotlib")
        return None
    
    graph = create_iot_smart_city()
    criticality = calculate_criticality(graph)
    
    config = VisualizationConfig(
        title="IoT Smart City System",
        dpi=150
    )
    visualizer = GraphVisualizer(config)
    
    # PNG export
    png_path = output_dir / "graph.png"
    visualizer.render_image(graph, str(png_path), criticality, format='png')
    print_success(f"PNG image saved to {png_path}")
    
    return png_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualization examples')
    parser.add_argument('--example',
                       choices=['basic', 'multi-layer', 'criticality', 'layers',
                               'layouts', 'dashboard', 'export', 'image', 'all'],
                       default='all',
                       help='Which example to run')
    parser.add_argument('--output-dir', '-o',
                       default='./visualization_output',
                       help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{Colors.BOLD}Graph Visualization Examples{Colors.ENDC}")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if args.example in ['basic', 'all']:
        example_basic_html(output_dir)
    
    if args.example in ['multi-layer', 'all']:
        example_multi_layer(output_dir)
    
    if args.example in ['criticality', 'all']:
        example_criticality_view(output_dir)
    
    if args.example in ['layers', 'all']:
        example_layer_views(output_dir)
    
    if args.example in ['layouts', 'all']:
        example_layouts(output_dir)
    
    if args.example in ['dashboard', 'all']:
        example_dashboard(output_dir)
    
    if args.example in ['export', 'all']:
        example_export(output_dir)
    
    if args.example in ['image', 'all']:
        example_static_image(output_dir)
    
    print(f"\n{Colors.GREEN}All examples completed!{Colors.ENDC}")
    print(f"Open files in {output_dir} to view visualizations.\n")


if __name__ == '__main__':
    main()