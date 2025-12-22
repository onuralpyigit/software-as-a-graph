#!/usr/bin/env python3
"""
Pub-Sub System Visualization Tool
===================================

Comprehensive visualization for multi-layer pub-sub system graphs.
Generates interactive HTML, static images, and dashboards.

Features:
- Multi-layer visualization (Application, Topic, Broker, Infrastructure)
- Interactive HTML with vis.js
- Static image export (PNG, SVG, PDF)
- Comprehensive dashboards
- Multiple layout algorithms
- Criticality-based coloring
- Export for Gephi, D3.js

Usage:
    # Basic HTML visualization
    python visualize_graph.py --input system.json --output graph.html
    
    # Multi-layer view
    python visualize_graph.py --input system.json --output layers.html --multi-layer
    
    # Dashboard with analysis
    python visualize_graph.py --input system.json --output dashboard.html --dashboard
    
    # Static image
    python visualize_graph.py --input system.json --output graph.png --format png
    
    # Layer-specific view
    python visualize_graph.py --input system.json --output apps.html --layer application

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

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
# Terminal Colors
# ============================================================================

class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    _enabled = True
    
    @classmethod
    def disable(cls):
        cls._enabled = False
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'WARNING', 'RED',
                     'ENDC', 'BOLD', 'DIM']:
            setattr(cls, attr, '')


def print_header(text: str):
    width = 70
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(width)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * width}{Colors.ENDC}")


def print_section(title: str):
    print(f"\n{Colors.BOLD}{title}{Colors.ENDC}")
    print("-" * 50)


def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.ENDC} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.ENDC} {msg}")


def print_warning(msg: str):
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {msg}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ{Colors.ENDC} {msg}")


# ============================================================================
# Graph Loading
# ============================================================================

def load_graph_from_json(filepath: str) -> nx.DiGraph:
    """Load graph from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes (infrastructure)
    for node in data.get('nodes', []):
        node_id = node.get('id', node.get('name'))
        G.add_node(node_id, type='Node', **node)
        
    # Add brokers 
    for broker in data.get('brokers', []):
        broker_id = broker.get('id', broker.get('name'))
        G.add_node(broker_id, type='Broker', **broker)
        
    # Add topics
    for topic in data.get('topics', []):
        topic_id = topic.get('id', topic.get('name'))
        G.add_node(topic_id, type='Topic', **topic)
        
    # Add applications
    for app in data.get('applications', []):
        app_id = app.get('id', app.get('name'))
        G.add_node(app_id, type='Application', **app)
        
    # Process relationships
    relationships = data.get('relationships', {})

    # PUBLISHES_TO relationships
    for rel in relationships.get('publishes_to', data.get('publishes', [])):
        app_id = rel.get('from', rel.get('source', rel.get('app')))
        topic_id = rel.get('to', rel.get('target', rel.get('topic')))
        G.add_edge(app_id, topic_id, type='PUBLISHES_TO', **rel)

    # SUBSCRIBES_TO relationships
    for rel in relationships.get('subscribes_to', data.get('subscribes', [])):
        app_id = rel.get('from', rel.get('source', rel.get('app')))
        topic_id = rel.get('to', rel.get('target', rel.get('topic')))
        G.add_edge(topic_id, app_id, type='SUBSCRIBES_TO', **rel)

    # ROUTES relationships
    for rel in relationships.get('routes', data.get('routes', [])):
        topic_id = rel.get('to', rel.get('target', rel.get('topic')))
        broker_id = rel.get('from', rel.get('source', rel.get('broker')))
        G.add_edge(broker_id, topic_id, type='ROUTES', **rel)

    # RUNS_ON relationships
    for rel in relationships.get('runs_on', data.get('runs', [])):
        comp_id = rel.get('from', rel.get('source', rel.get('component')))
        node_id = rel.get('to', rel.get('target', rel.get('node')))
        G.add_edge(comp_id, node_id, type='RUNS_ON', **rel)

    # CONNECTS_TO relationships
    for rel in relationships.get('connects_to', data.get('connects', [])):
        src_id = rel.get('from', rel.get('source', rel.get('source_component')))
        dst_id = rel.get('to', rel.get('target', rel.get('target_component')))
        G.add_edge(src_id, dst_id, type='CONNECTS_TO', **rel)
    
    return G


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


# ============================================================================
# Criticality Calculation
# ============================================================================

def calculate_criticality(graph: nx.DiGraph,
                         alpha: float = 0.25,
                         beta: float = 0.30,
                         gamma: float = 0.25,
                         delta: float = 0.10,
                         epsilon: float = 0.10) -> Dict[str, Dict]:
    """Calculate criticality scores for visualization"""
    
    if graph.number_of_nodes() == 0:
        return {}
    
    criticality = {}
    
    # Betweenness centrality
    bc = nx.betweenness_centrality(graph)
    max_bc = max(bc.values()) if bc.values() else 1
    bc_norm = {k: v / max_bc if max_bc > 0 else 0 for k, v in bc.items()}
    
    # Articulation points
    try:
        aps = set(nx.articulation_points(graph.to_undirected()))
    except:
        aps = set()
    
    # Degree
    degrees = dict(graph.degree())
    max_deg = max(degrees.values()) if degrees.values() else 1
    
    # PageRank
    try:
        pr = nx.pagerank(graph, alpha=0.85)
    except:
        pr = {n: 1.0 / graph.number_of_nodes() for n in graph.nodes()}
    max_pr = max(pr.values()) if pr.values() else 1
    pr_norm = {k: v / max_pr if max_pr > 0 else 0 for k, v in pr.items()}
    
    for node in graph.nodes():
        bc_score = bc_norm.get(node, 0)
        ap_score = 1.0 if node in aps else 0.0
        
        # Impact score
        try:
            descendants = len(nx.descendants(graph, node))
            ancestors = len(nx.ancestors(graph, node))
            impact = (descendants + ancestors) / (2 * graph.number_of_nodes())
        except:
            impact = 0.0
        
        degree_score = degrees.get(node, 0) / max_deg
        pr_score = pr_norm.get(node, 0)
        
        # Composite score
        score = (alpha * bc_score + 
                beta * ap_score + 
                gamma * impact + 
                delta * degree_score + 
                epsilon * pr_score)
        score = min(1.0, score)
        
        # Classify level
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
            'betweenness': bc_norm.get(node, 0),
            'degree': degrees.get(node, 0)
        }
    
    return criticality


# ============================================================================
# Main CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Visualize pub-sub system graphs with multi-layer support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic HTML visualization
    python visualize_graph.py --input system.json --output graph.html
    
    # Multi-layer view
    python visualize_graph.py --input system.json --output layers.html --multi-layer
    
    # Dashboard
    python visualize_graph.py --input system.json --output dashboard.html --dashboard
    
    # Static image (requires matplotlib)
    python visualize_graph.py --input system.json --output graph.png --format png
    
    # Application layer only
    python visualize_graph.py --input system.json --output apps.html --layer application
    
    # Criticality coloring
    python visualize_graph.py --input system.json --output crit.html --color-by criticality
    
    # Export for Gephi
    python visualize_graph.py --input system.json --output graph.gexf --export-gephi

Layouts:
    spring       - Force-directed (default)
    hierarchical - Tree-like by layer
    circular     - Circular arrangement
    shell        - Concentric by type
    kamada_kawai - Energy minimization
    layered      - Layer-based positioning

Color Schemes:
    type         - Color by component type (default)
    criticality  - Color by criticality level
    layer        - Color by system layer
        """
    )
    
    # Input/Output
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input', '-i', required=True,
                         help='Input JSON file with pub-sub system')
    io_group.add_argument('--output', '-o', required=True,
                         help='Output file path')
    io_group.add_argument('--format', '-f',
                         choices=['html', 'png', 'svg', 'pdf'],
                         default='html',
                         help='Output format (default: html)')
    
    # Visualization mode
    mode_group = parser.add_argument_group('Visualization Mode')
    mode_group.add_argument('--multi-layer', action='store_true',
                           help='Generate multi-layer visualization')
    mode_group.add_argument('--dashboard', action='store_true',
                           help='Generate comprehensive dashboard')
    mode_group.add_argument('--layer',
                           choices=['all', 'application', 'topic', 'broker', 'infrastructure'],
                           default='all',
                           help='Layer to visualize (default: all)')
    
    # Layout and styling
    style_group = parser.add_argument_group('Layout and Styling')
    style_group.add_argument('--layout',
                            choices=['spring', 'hierarchical', 'circular', 
                                    'shell', 'kamada_kawai', 'layered', 'spectral'],
                            default='spring',
                            help='Layout algorithm (default: spring)')
    style_group.add_argument('--color-by',
                            choices=['type', 'criticality', 'layer'],
                            default='type',
                            help='Color scheme (default: type)')
    style_group.add_argument('--title',
                            help='Visualization title')
    style_group.add_argument('--theme',
                            choices=['dark', 'light'],
                            default='dark',
                            help='Color theme (default: dark)')
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--analysis-file',
                               help='Load analysis results from JSON file')
    analysis_group.add_argument('--validation-file',
                               help='Load validation results from JSON file')
    analysis_group.add_argument('--simulation-file',
                               help='Load simulation results from JSON file')
    analysis_group.add_argument('--run-analysis', action='store_true',
                               help='Run criticality analysis')
    
    # Scoring weights
    weights_group = parser.add_argument_group('Scoring Weights (for --run-analysis)')
    weights_group.add_argument('--alpha', type=float, default=0.25,
                              help='Betweenness centrality weight (default: 0.25)')
    weights_group.add_argument('--beta', type=float, default=0.30,
                              help='Articulation point weight (default: 0.30)')
    weights_group.add_argument('--gamma', type=float, default=0.25,
                              help='Impact score weight (default: 0.25)')
    weights_group.add_argument('--delta', type=float, default=0.10,
                              help='Degree centrality weight (default: 0.10)')
    weights_group.add_argument('--epsilon', type=float, default=0.10,
                              help='PageRank weight (default: 0.10)')
    
    # Export options
    export_group = parser.add_argument_group('Export Options')
    export_group.add_argument('--export-gephi', action='store_true',
                             help='Export GEXF for Gephi')
    export_group.add_argument('--export-d3', action='store_true',
                             help='Export JSON for D3.js')
    export_group.add_argument('--dpi', type=int, default=150,
                             help='DPI for image export (default: 150)')
    
    # Output control
    output_group = parser.add_argument_group('Output Control')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='Minimal output')
    output_group.add_argument('--no-color', action='store_true',
                             help='Disable colored output')
    output_group.add_argument('--no-physics', action='store_true',
                             help='Disable physics simulation in HTML')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup
    if args.no_color:
        Colors.disable()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)
    
    if not args.quiet:
        print_header("PUB-SUB SYSTEM VISUALIZATION")
        print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")
    
    try:
        # Load graph
        if not args.quiet:
            print_section("Loading Graph")
        
        graph = load_graph_from_json(args.input)
        print_success(f"Loaded graph: {graph.number_of_nodes()} nodes, "
                     f"{graph.number_of_edges()} edges")
        
        # Load or calculate criticality
        criticality = None
        validation = None
        simulation = None
        analysis = None
        
        if args.analysis_file:
            if not args.quiet:
                print_info(f"Loading analysis from {args.analysis_file}")
            analysis = load_json_file(args.analysis_file)
            # Extract criticality from analysis
            if 'criticality' in analysis:
                criticality = analysis['criticality']
            elif 'components' in analysis:
                criticality = {c['id']: c for c in analysis['components']}
        
        if args.validation_file:
            if not args.quiet:
                print_info(f"Loading validation from {args.validation_file}")
            validation = load_json_file(args.validation_file)
        
        if args.simulation_file:
            if not args.quiet:
                print_info(f"Loading simulation from {args.simulation_file}")
            simulation = load_json_file(args.simulation_file)
        
        if args.run_analysis or criticality is None:
            if not args.quiet:
                print_section("Calculating Criticality")
            criticality = calculate_criticality(
                graph,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                delta=args.delta,
                epsilon=args.epsilon
            )
            print_success(f"Calculated criticality for {len(criticality)} components")
        
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Map args to enums
        layer_map = {
            'all': Layer.ALL,
            'application': Layer.APPLICATION,
            'topic': Layer.TOPIC,
            'broker': Layer.BROKER,
            'infrastructure': Layer.INFRASTRUCTURE
        }
        layer = layer_map.get(args.layer, Layer.ALL)
        
        layout_map = {
            'spring': LayoutAlgorithm.SPRING,
            'hierarchical': LayoutAlgorithm.HIERARCHICAL,
            'circular': LayoutAlgorithm.CIRCULAR,
            'shell': LayoutAlgorithm.SHELL,
            'kamada_kawai': LayoutAlgorithm.KAMADA_KAWAI,
            'layered': LayoutAlgorithm.LAYERED,
            'spectral': LayoutAlgorithm.SPECTRAL
        }
        layout = layout_map.get(args.layout, LayoutAlgorithm.SPRING)
        
        color_map = {
            'type': ColorScheme.TYPE,
            'criticality': ColorScheme.CRITICALITY,
            'layer': ColorScheme.LAYER
        }
        color_scheme = color_map.get(args.color_by, ColorScheme.TYPE)
        
        # Generate visualization
        if not args.quiet:
            print_section("Generating Visualization")
        
        if args.dashboard:
            # Generate dashboard
            dashboard_config = DashboardConfig(
                title=args.title or "Pub-Sub System Analysis Dashboard",
                theme=args.theme
            )
            generator = DashboardGenerator(dashboard_config)
            html = generator.generate(
                graph=graph,
                criticality=criticality,
                validation=validation,
                simulation=simulation,
                analysis=analysis
            )
            
            with open(output_path, 'w') as f:
                f.write(html)
            print_success(f"Dashboard saved to {output_path}")
        
        elif args.format == 'html':
            # Generate HTML
            vis_config = VisualizationConfig(
                title=args.title or "Pub-Sub System Visualization",
                layout=layout,
                color_scheme=color_scheme,
                physics_enabled=not args.no_physics,
                dpi=args.dpi
            )
            visualizer = GraphVisualizer(vis_config)
            
            if args.multi_layer:
                html = visualizer.render_multi_layer_html(
                    graph=graph,
                    criticality=criticality,
                    title=args.title or "Multi-Layer System Architecture"
                )
            else:
                html = visualizer.render_html(
                    graph=graph,
                    criticality=criticality,
                    title=args.title,
                    layer=layer
                )
            
            with open(output_path, 'w') as f:
                f.write(html)
            print_success(f"HTML saved to {output_path}")
        
        elif args.format in ['png', 'svg', 'pdf']:
            # Generate static image
            if not MATPLOTLIB_AVAILABLE:
                print_error("Matplotlib is required for image export")
                print_info("Install with: pip install matplotlib")
                return 1
            
            vis_config = VisualizationConfig(
                title=args.title or "Pub-Sub System",
                layout=layout,
                color_scheme=color_scheme,
                dpi=args.dpi
            )
            visualizer = GraphVisualizer(vis_config)
            
            result = visualizer.render_image(
                graph=graph,
                output_path=str(output_path),
                criticality=criticality,
                title=args.title,
                format=args.format
            )
            
            if result:
                print_success(f"Image saved to {output_path}")
            else:
                print_error("Failed to generate image")
                return 1
        
        # Additional exports
        if args.export_gephi:
            gephi_path = output_path.with_suffix('.gexf')
            vis_config = VisualizationConfig()
            visualizer = GraphVisualizer(vis_config)
            visualizer.export_for_gephi(graph, str(gephi_path), criticality)
            print_success(f"Gephi export saved to {gephi_path}")
        
        if args.export_d3:
            d3_path = output_path.with_suffix('.d3.json')
            vis_config = VisualizationConfig()
            visualizer = GraphVisualizer(vis_config)
            visualizer.export_for_d3(graph, str(d3_path), criticality)
            print_success(f"D3.js export saved to {d3_path}")
        
        # Summary
        if not args.quiet and args.verbose:
            print_section("Layer Statistics")
            vis_config = VisualizationConfig()
            visualizer = GraphVisualizer(vis_config)
            visualizer.classify_layers(graph)
            stats = visualizer.get_layer_statistics(graph)
            
            for layer_name, layer_stats in stats.get('layers', {}).items():
                print(f"\n  {layer_name.title()}:")
                print(f"    Nodes: {layer_stats.get('node_count', 0)}")
                print(f"    Internal edges: {layer_stats.get('internal_edges', 0)}")
                print(f"    Cross-layer edges: {layer_stats.get('cross_layer_edges', 0)}")
        
        if not args.quiet:
            print(f"\n{Colors.GREEN}Visualization complete!{Colors.ENDC}\n")
        
        return 0
    
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        return 1
    except Exception as e:
        print_error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())