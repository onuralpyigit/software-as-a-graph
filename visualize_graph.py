#!/usr/bin/env python3
"""
Visualize Graph Script - Enhanced Version

Command-line interface for comprehensive visualization of distributed pub-sub systems
using the modular architecture with enhanced features and error handling.

Features:
- Multi-layer graph visualization (Application, Infrastructure, Topic layers)
- Hidden dependency detection and highlighting
- Failure impact visualization with cascade analysis
- Interactive HTML visualizations with D3.js/Vis.js
- Static image generation (PNG, SVG, PDF)
- Criticality-aware color schemes
- Multiple layout algorithms (force-directed, hierarchical, circular)
- Metrics dashboard generation
- Cross-layer dependency analysis
- Anti-pattern highlighting
- Comparative visualizations (baseline vs. failure)
- Export to multiple formats

Architecture:
  GraphBuilder → GraphModel → GraphExporter → NetworkX
                                                  ↓
                                         AnalysisOrchestrator
                                                  ↓
                                         GraphVisualizer / LayerRenderer
                                                  ↓
                                         Visualization Outputs

Usage:
    # Basic visualization from JSON
    python visualize_graph.py --input system.json
    
    # Visualization from Neo4j
    python visualize_graph.py --neo4j --uri bolt://localhost:7687 \\
        --user neo4j --password password
    
    # Multi-layer visualization with failure simulation
    python visualize_graph.py --input system.json --multi-layer \\
        --simulate-failure B1 --cascade --output-dir visualizations/
    
    # Generate all visualization types
    python visualize_graph.py --input system.json --all \\
        --layout hierarchical --color-scheme criticality
    
    # Hidden dependencies analysis
    python visualize_graph.py --input system.json --hidden-deps \\
        --threshold 0.8
    
    # Dashboard with failure scenarios
    python visualize_graph.py --input system.json --dashboard \\
        --failure-scenarios spof cascade recovery
    
    # Export to multiple formats
    python visualize_graph.py --input system.json --complete \\
        --export-png --export-svg --export-html
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.core.graph_builder import GraphBuilder
    from src.core.graph_exporter import GraphExporter
    from src.core.graph_model import GraphModel
    from src.orchestration.analysis_orchestrator import AnalysisOrchestrator
    import networkx as nx
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Please ensure all required modules are installed:")
    print("  pip install networkx matplotlib")
    sys.exit(1)

# Optional imports
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not installed, image export will be disabled")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Layout algorithms
LAYOUT_ALGORITHMS = {
    'spring': 'Force-directed spring layout (default)',
    'hierarchical': 'Hierarchical tree layout',
    'circular': 'Circular layout',
    'kamada_kawai': 'Kamada-Kawai force-directed layout',
    'shell': 'Concentric shell layout',
    'spectral': 'Spectral layout based on graph Laplacian',
}

# Color schemes
COLOR_SCHEMES = {
    'criticality': 'Color by criticality score (red=critical, green=low)',
    'type': 'Color by component type',
    'layer': 'Color by graph layer',
    'qos': 'Color by QoS priority',
    'health': 'Color by component health status',
}


def setup_logging(verbose: bool = False, log_file: str = 'visualization.log') -> logging.Logger:
    """
    Configure logging with enhanced formatting
    
    Args:
        verbose: Enable DEBUG level logging
        log_file: Path to log file
    
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


def load_graph_from_json(filepath: str, logger: logging.Logger) -> Tuple[GraphModel, nx.DiGraph]:
    """
    Load graph from JSON file with error handling
    
    Args:
        filepath: Path to JSON configuration file
        logger: Logger instance
    
    Returns:
        Tuple of (GraphModel, NetworkX DiGraph)
    """
    logger.info(f"Loading graph from JSON: {filepath}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    try:
        builder = GraphBuilder()
        model = builder.build_from_json(filepath)
        
        logger.info(f"✓ Loaded {len(model.applications)} applications, "
                   f"{len(model.topics)} topics, "
                   f"{len(model.brokers)} brokers, "
                   f"{len(model.nodes)} nodes")
        
        # Convert to NetworkX
        exporter = GraphExporter()
        graph = exporter.export_to_networkx(model)
        
        logger.info(f"✓ Created NetworkX graph: {len(graph)} nodes, {len(graph.edges())} edges")
        
        return model, graph
        
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        raise


def get_node_colors(graph: nx.DiGraph, color_scheme: str,
                    criticality_scores: Optional[Dict] = None) -> List[str]:
    """
    Get node colors based on color scheme
    
    Args:
        graph: NetworkX graph
        color_scheme: Color scheme name
        criticality_scores: Optional criticality scores dictionary
    
    Returns:
        List of color strings for each node
    """
    colors = []
    
    # Type-based colors
    type_colors = {
        'Application': '#3498db',  # Blue
        'Topic': '#2ecc71',        # Green
        'Broker': '#e74c3c',       # Red
        'Node': '#95a5a6',         # Gray
    }
    
    # Layer-based colors
    layer_colors = {
        'Application': '#3498db',
        'Topic': '#2ecc71',
        'Broker': '#e67e22',
        'Infrastructure': '#95a5a6',
    }
    
    for node in graph.nodes():
        node_data = graph.nodes[node]
        
        if color_scheme == 'type':
            node_type = node_data.get('type', 'Unknown')
            colors.append(type_colors.get(node_type, '#34495e'))
            
        elif color_scheme == 'layer':
            node_type = node_data.get('type', 'Unknown')
            if node_type == 'Node':
                colors.append(layer_colors['Infrastructure'])
            else:
                colors.append(layer_colors.get(node_type, '#34495e'))
                
        elif color_scheme == 'criticality':
            if criticality_scores:
                score = criticality_scores.get(node, {}).get('composite_score', 0)
                # Red (high) to Yellow to Green (low)
                if score > 0.7:
                    colors.append('#e74c3c')  # Red
                elif score > 0.5:
                    colors.append('#e67e22')  # Orange
                elif score > 0.3:
                    colors.append('#f39c12')  # Yellow
                else:
                    colors.append('#27ae60')  # Green
            else:
                colors.append('#3498db')  # Default blue
                
        else:
            # Default coloring
            colors.append('#3498db')
    
    return colors


def get_node_sizes(graph: nx.DiGraph, criticality_scores: Optional[Dict] = None) -> List[float]:
    """
    Get node sizes based on criticality scores
    
    Args:
        graph: NetworkX graph
        criticality_scores: Optional criticality scores dictionary
    
    Returns:
        List of sizes for each node
    """
    sizes = []
    
    for node in graph.nodes():
        if criticality_scores:
            score = criticality_scores.get(node, {}).get('composite_score', 0.5)
            # Size range: 300-1500
            size = 300 + (score * 1200)
        else:
            size = 500  # Default size
        
        sizes.append(size)
    
    return sizes


def visualize_complete_system(graph: nx.DiGraph, model: GraphModel,
                              layout: str, color_scheme: str,
                              output_path: Path,
                              criticality_scores: Optional[Dict] = None,
                              logger: logging.Logger = None) -> str:
    """
    Create complete system visualization
    
    Args:
        graph: NetworkX graph
        model: GraphModel instance
        layout: Layout algorithm name
        color_scheme: Color scheme name
        output_path: Output file path
        criticality_scores: Optional criticality scores
        logger: Logger instance
    
    Returns:
        Path to generated visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")
    
    if logger:
        logger.info(f"Creating complete system visualization with {layout} layout...")
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Compute layout
    if layout == 'spring':
        pos = nx.spring_layout(graph, k=2, iterations=50)
    elif layout == 'hierarchical':
        pos = nx.nx_agraph.graphviz_layout(graph, prog='dot') if hasattr(nx, 'nx_agraph') else nx.spring_layout(graph)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(graph)
    elif layout == 'shell':
        # Group nodes by type for shell layout
        node_types = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append(node)
        
        pos = nx.shell_layout(graph, nlist=list(node_types.values()))
    elif layout == 'spectral':
        pos = nx.spectral_layout(graph)
    else:
        pos = nx.spring_layout(graph)
    
    # Get colors and sizes
    colors = get_node_colors(graph, color_scheme, criticality_scores)
    sizes = get_node_sizes(graph, criticality_scores)
    
    # Draw graph
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=sizes, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, arrows=True, arrowsize=20)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold')
    
    # Title
    plt.title(f"Complete System Visualization\n{len(graph)} nodes, {len(graph.edges())} edges",
             fontsize=16, fontweight='bold')
    
    # Legend
    legend_elements = []
    if color_scheme == 'type':
        type_colors = {
            'Application': '#3498db',
            'Topic': '#2ecc71',
            'Broker': '#e74c3c',
            'Node': '#95a5a6',
        }
        from matplotlib.patches import Patch
        for node_type, color in type_colors.items():
            legend_elements.append(Patch(facecolor=color, label=node_type))
        plt.legend(handles=legend_elements, loc='upper left')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"✓ Saved visualization: {output_path}")
    
    return str(output_path)


def generate_html_visualization(graph: nx.DiGraph, model: GraphModel,
                                output_path: Path,
                                criticality_scores: Optional[Dict] = None,
                                logger: logging.Logger = None) -> str:
    """
    Generate interactive HTML visualization using Vis.js
    
    Args:
        graph: NetworkX graph
        model: GraphModel instance
        output_path: Output file path
        criticality_scores: Optional criticality scores
        logger: Logger instance
    
    Returns:
        Path to generated HTML file
    """
    if logger:
        logger.info("Generating interactive HTML visualization...")
    
    # Prepare nodes data
    nodes_data = []
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_type = node_data.get('type', 'Unknown')
        
        # Get criticality
        criticality = 0.5
        if criticality_scores:
            criticality = criticality_scores.get(node, {}).get('composite_score', 0.5)
        
        # Color based on criticality
        if criticality > 0.7:
            color = '#e74c3c'
        elif criticality > 0.5:
            color = '#e67e22'
        elif criticality > 0.3:
            color = '#f39c12'
        else:
            color = '#27ae60'
        
        nodes_data.append({
            'id': node,
            'label': node,
            'title': f"{node}\\nType: {node_type}\\nCriticality: {criticality:.3f}",
            'color': color,
            'size': 10 + (criticality * 20)
        })
    
    # Prepare edges data
    edges_data = []
    for u, v in graph.edges():
        edges_data.append({
            'from': u,
            'to': v,
            'arrows': 'to'
        })
    
    # Generate HTML
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Visualization</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }}
        #header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        #mynetwork {{
            width: 100%;
            height: 800px;
            border: 1px solid #ddd;
        }}
        #info {{
            padding: 20px;
            background: #f5f5f5;
        }}
        .controls {{
            padding: 20px;
            background: white;
            border-bottom: 1px solid #ddd;
        }}
        button {{
            padding: 10px 20px;
            margin: 5px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background: #2980b9;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Pub-Sub System Visualization</h1>
        <p>Generated: {timestamp}</p>
        <p>{len(graph)} nodes, {len(graph.edges())} edges</p>
    </div>
    
    <div class="controls">
        <button onclick="network.fit()">Fit View</button>
        <button onclick="network.stabilize()">Stabilize</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
        <button onclick="exportImage()">Export Image</button>
    </div>
    
    <div id="mynetwork"></div>
    
    <div id="info">
        <h3>Legend</h3>
        <p><span style="color: #e74c3c;">●</span> Critical (>0.7)</p>
        <p><span style="color: #e67e22;">●</span> High (0.5-0.7)</p>
        <p><span style="color: #f39c12;">●</span> Medium (0.3-0.5)</p>
        <p><span style="color: #27ae60;">●</span> Low (<0.3)</p>
    </div>

    <script type="text/javascript">
        // Node and edge data
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});

        // Create a network
        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        
        var options = {{
            nodes: {{
                shape: 'dot',
                font: {{
                    size: 14,
                    face: 'Arial'
                }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                width: 1,
                color: {{
                    color: '#848484',
                    highlight: '#3498db'
                }},
                smooth: {{
                    type: 'continuous'
                }},
                arrows: {{
                    to: {{
                        enabled: true,
                        scaleFactor: 0.5
                    }}
                }}
            }},
            physics: {{
                stabilization: {{
                    iterations: 200
                }},
                barnesHut: {{
                    gravitationalConstant: -8000,
                    springConstant: 0.001,
                    springLength: 200
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        
        // Event handlers
        network.on("click", function (params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var nodeData = nodes.get(nodeId);
                console.log('Clicked node:', nodeData);
            }}
        }});
        
        // Toggle physics
        var physicsEnabled = true;
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{physics: physicsEnabled}});
        }}
        
        // Export image
        function exportImage() {{
            var canvas = document.getElementsByTagName('canvas')[0];
            var link = document.createElement('a');
            link.download = 'network.png';
            link.href = canvas.toDataURL();
            link.click();
        }}
    </script>
</body>
</html>
"""
    
    # Save HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    if logger:
        logger.info(f"✓ Saved HTML visualization: {output_path}")
    
    return str(output_path)


def generate_dashboard(graph: nx.DiGraph, model: GraphModel,
                       analysis_results: Dict[str, Any],
                       output_path: Path,
                       logger: logging.Logger = None) -> str:
    """
    Generate metrics dashboard
    
    Args:
        graph: NetworkX graph
        model: GraphModel instance
        analysis_results: Analysis results dictionary
        output_path: Output file path
        logger: Logger instance
    
    Returns:
        Path to generated dashboard
    """
    if logger:
        logger.info("Generating metrics dashboard...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract metrics
    summary = analysis_results.get('graph_summary', {})
    criticality = analysis_results.get('criticality_scores', {})
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Metrics Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        .metric-value {{
            font-size: 3em;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 1.1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .critical-components {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 30px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .badge {{
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .badge-critical {{ background: #e74c3c; color: white; }}
        .badge-high {{ background: #e67e22; color: white; }}
        .badge-medium {{ background: #f39c12; color: white; }}
        .badge-low {{ background: #27ae60; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 System Metrics Dashboard</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="container">
        <h2>System Overview</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Nodes</div>
                <div class="metric-value">{summary.get('total_nodes', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Edges</div>
                <div class="metric-value">{summary.get('total_edges', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Applications</div>
                <div class="metric-value">{summary.get('applications', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Topics</div>
                <div class="metric-value">{summary.get('topics', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Brokers</div>
                <div class="metric-value">{summary.get('brokers', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Nodes</div>
                <div class="metric-value">{summary.get('nodes', 0)}</div>
            </div>
        </div>
        
        <div class="critical-components">
            <h2>Critical Components</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Component</th>
                        <th>Type</th>
                        <th>Score</th>
                        <th>Level</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    critical_comps = criticality.get('critical_components', [])
    for i, comp in enumerate(critical_comps[:10], 1):
        level = comp.get('criticality_level', 'LOW')
        badge_class = f"badge-{level.lower()}"
        
        html += f"""
                    <tr>
                        <td>{i}</td>
                        <td><strong>{comp.get('component', 'N/A')}</strong></td>
                        <td>{comp.get('type', 'N/A')}</td>
                        <td>{comp.get('composite_score', 0):.3f}</td>
                        <td><span class="badge {badge_class}">{level}</span></td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    
    # Save dashboard
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    if logger:
        logger.info(f"✓ Saved dashboard: {output_path}")
    
    return str(output_path)


def print_summary(output_files: Dict[str, str], elapsed_time: float):
    """
    Print visualization summary
    
    Args:
        output_files: Dictionary of generated files
        elapsed_time: Total execution time
    """
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}VISUALIZATION COMPLETE{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    
    print(f"\n{Colors.OKGREEN}📊 Generated Visualizations:{Colors.ENDC}")
    for vis_type, filepath in output_files.items():
        print(f"   • {vis_type}: {filepath}")
    
    print(f"\n{Colors.OKBLUE}⏱️  Execution Time: {elapsed_time:.2f}s{Colors.ENDC}")
    
    print(f"\n{Colors.OKCYAN}💡 Next Steps:{Colors.ENDC}")
    print("   • Open HTML files in your web browser for interactive visualizations")
    print("   • Use static images (PNG/SVG) for presentations and papers")
    print("   • Explore the metrics dashboard for system health overview")
    print("   • Analyze hidden dependencies and failure scenarios")
    
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def main() -> int:
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Visualize pub-sub system graph - Enhanced Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Layout Algorithms:
  spring              : Force-directed spring layout (default)
  hierarchical        : Hierarchical tree layout
  circular            : Circular layout
  kamada_kawai        : Kamada-Kawai force-directed layout
  shell               : Concentric shell layout
  spectral            : Spectral layout

Color Schemes:
  criticality         : Color by criticality score (default)
  type                : Color by component type
  layer               : Color by graph layer
  qos                 : Color by QoS priority

Examples:
  # Basic visualization
  python visualize_graph.py --input system.json

  # All visualizations with hierarchical layout
  python visualize_graph.py --input system.json --all \\
      --layout hierarchical

  # Interactive HTML with criticality coloring
  python visualize_graph.py --input system.json --complete \\
      --color-scheme criticality --export-html

  # Dashboard with analysis
  python visualize_graph.py --input system.json --dashboard

  # Failure scenario visualization
  python visualize_graph.py --input system.json \\
      --simulate-failure B1 --cascade --export-png

For more information, see docs/VISUALIZE_GRAPH_README.md
        """
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str,
                            help='Input JSON file')
    input_group.add_argument('--neo4j', action='store_true',
                            help='Load from Neo4j database')
    
    # Neo4j connection
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                       help='Neo4j URI')
    parser.add_argument('--user', type=str, default='neo4j',
                       help='Neo4j username')
    parser.add_argument('--password', type=str,
                       help='Neo4j password')
    parser.add_argument('--database', type=str, default='neo4j',
                       help='Neo4j database')
    
    # Visualization options
    parser.add_argument('--layout', type=str, default='spring',
                       choices=list(LAYOUT_ALGORITHMS.keys()),
                       help='Layout algorithm (default: spring)')
    parser.add_argument('--color-scheme', type=str, default='criticality',
                       choices=list(COLOR_SCHEMES.keys()),
                       help='Color scheme (default: criticality)')
    parser.add_argument('--no-labels', action='store_true',
                       help='Hide node labels')
    
    # Visualization types
    parser.add_argument('--all', action='store_true',
                       help='Generate all visualization types')
    parser.add_argument('--complete', action='store_true',
                       help='Complete system visualization')
    parser.add_argument('--dashboard', action='store_true',
                       help='Create metrics dashboard')
    parser.add_argument('--multi-layer', action='store_true',
                       help='Multi-layer visualizations')
    parser.add_argument('--hidden-deps', action='store_true',
                       help='Hidden dependencies analysis')
    
    # Failure analysis
    parser.add_argument('--simulate-failure', type=str,
                       help='Simulate failure of specific component')
    parser.add_argument('--cascade', action='store_true',
                       help='Enable cascading failures')
    parser.add_argument('--failure-scenarios', nargs='+',
                       choices=['spof', 'cascade', 'multi', 'recovery'],
                       help='Generate failure scenario visualizations')
    
    # Analysis options
    parser.add_argument('--run-analysis', action='store_true',
                       help='Run full analysis before visualization')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Threshold for hidden dependency detection')
    
    # Export options
    parser.add_argument('--output-dir', '-o', type=str, default='visualizations',
                       help='Output directory (default: visualizations)')
    parser.add_argument('--export-png', action='store_true',
                       help='Export as PNG image')
    parser.add_argument('--export-svg', action='store_true',
                       help='Export as SVG image')
    parser.add_argument('--export-html', action='store_true', default=True,
                       help='Export as HTML (default: enabled)')
    parser.add_argument('--export-pdf', action='store_true',
                       help='Export as PDF')
    
    # General options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--log-file', type=str, default='visualization.log',
                       help='Log file path')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    # Setup logging
    logger = setup_logging(args.verbose, args.log_file)
    
    try:
        start_time = time.time()
        
        # Print header
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}PUB-SUB SYSTEM VISUALIZATION{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
        
        # Load graph
        logger.info("Loading graph...")
        if args.neo4j:
            if not args.password:
                parser.error("--neo4j requires --password")
            raise NotImplementedError("Neo4j loading not yet implemented for visualization")
        else:
            model, graph = load_graph_from_json(args.input, logger)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis if requested
        criticality_scores = None
        analysis_results = {}
        
        if args.run_analysis or args.dashboard or args.color_scheme == 'criticality':
            logger.info("Running analysis...")
            orchestrator = AnalysisOrchestrator(
                output_dir=str(output_dir / "analysis"),
                enable_qos=True
            )
            analysis_results = orchestrator.analyze_graph(
                graph=graph,
                graph_model=model,
                enable_simulation=False
            )
            
            # Extract criticality scores
            if 'criticality_scores' in analysis_results:
                crit = analysis_results['criticality_scores']
                criticality_scores = {}
                for comp in crit.get('all_scores', []):
                    criticality_scores[comp.get('component')] = comp
        
        # Generate visualizations
        output_files = {}
        
        # Complete visualization
        if args.all or args.complete:
            logger.info("Generating complete system visualization...")
            
            if MATPLOTLIB_AVAILABLE:
                if args.export_png or not args.export_html:
                    png_path = output_dir / "complete_system.png"
                    output_files['Complete System (PNG)'] = visualize_complete_system(
                        graph, model, args.layout, args.color_scheme,
                        png_path, criticality_scores, logger
                    )
                
                if args.export_svg:
                    svg_path = output_dir / "complete_system.svg"
                    output_files['Complete System (SVG)'] = visualize_complete_system(
                        graph, model, args.layout, args.color_scheme,
                        svg_path, criticality_scores, logger
                    )
            
            if args.export_html:
                html_path = output_dir / "complete_system.html"
                output_files['Complete System (HTML)'] = generate_html_visualization(
                    graph, model, html_path, criticality_scores, logger
                )
        
        # Dashboard
        if args.all or args.dashboard:
            if analysis_results:
                dashboard_path = output_dir / "dashboard.html"
                output_files['Dashboard'] = generate_dashboard(
                    graph, model, analysis_results, dashboard_path, logger
                )
            else:
                logger.warning("Dashboard requires analysis results, skipping")
        
        # Print summary
        elapsed_time = time.time() - start_time
        print_summary(output_files, elapsed_time)
        
        print(f"{Colors.OKGREEN}✓ Visualization completed successfully!{Colors.ENDC}\n")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nVisualization interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Visualization error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
