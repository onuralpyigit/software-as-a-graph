#!/usr/bin/env python3
"""
Graph Visualization CLI

Comprehensive visualization system for multi-layer pub-sub systems including:
- Interactive HTML dashboards with D3.js/Vis.js
- Multi-layer graph visualization (Application, Infrastructure, Topic, Broker)
- Criticality-aware color schemes
- Failure impact visualization
- Multiple layout algorithms
- Static exports (PNG, SVG)

Usage Examples:
    # Basic visualization
    python visualize_graph.py --input system.json --output viz.html
    
    # Multi-layer visualization
    python visualize_graph.py --input system.json --output viz.html \\
        --layer all --layout layered
    
    # Application layer only with criticality coloring
    python visualize_graph.py --input system.json --output app.html \\
        --layer application --color-by criticality
    
    # Infrastructure view
    python visualize_graph.py --input system.json --output infra.html \\
        --layer infrastructure --show-load
    
    # Dashboard with all analysis
    python visualize_graph.py --input system.json --output dashboard.html \\
        --dashboard --analysis analysis.json
    
    # Static image export
    python visualize_graph.py --input system.json --output graph.png \\
        --format png --dpi 300

Supported Formats:
    - HTML: Interactive visualization with Vis.js
    - PNG: Static raster image
    - SVG: Vector graphics
    - PDF: Document format
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Error: networkx is required. Install with: pip install networkx")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    @classmethod
    def disable(cls):
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.WARNING = cls.FAIL = cls.ENDC = cls.BOLD = cls.DIM = ''


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")


def print_section(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.DIM}{'-'*50}{Colors.ENDC}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")


def print_error(text: str):
    print(f"{Colors.FAIL}✗{Colors.ENDC} {text}")


def print_warning(text: str):
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {text}")


def print_info(text: str):
    print(f"{Colors.CYAN}ℹ{Colors.ENDC} {text}")


# =============================================================================
# Enums and Constants
# =============================================================================

class Layer(Enum):
    """Graph layers for visualization"""
    ALL = "all"
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    TOPIC = "topic"
    BROKER = "broker"


class Layout(Enum):
    """Layout algorithms"""
    SPRING = "spring"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    LAYERED = "layered"
    KAMADA_KAWAI = "kamada_kawai"
    SHELL = "shell"


class ColorScheme(Enum):
    """Color schemes for nodes"""
    TYPE = "type"
    CRITICALITY = "criticality"
    LAYER = "layer"
    QOS = "qos"


# Color palettes
TYPE_COLORS = {
    'Application': '#3498db',
    'Topic': '#2ecc71',
    'Broker': '#e74c3c',
    'Node': '#9b59b6',
    'Unknown': '#95a5a6'
}

CRITICALITY_COLORS = {
    'CRITICAL': '#e74c3c',
    'HIGH': '#e67e22',
    'MEDIUM': '#f1c40f',
    'LOW': '#27ae60',
    'MINIMAL': '#95a5a6'
}

LAYER_COLORS = {
    Layer.APPLICATION: '#3498db',
    Layer.TOPIC: '#2ecc71',
    Layer.BROKER: '#e74c3c',
    Layer.INFRASTRUCTURE: '#9b59b6'
}

EDGE_COLORS = {
    'PUBLISHES_TO': '#27ae60',
    'SUBSCRIBES_TO': '#3498db',
    'DEPENDS_ON': '#e74c3c',
    'RUNS_ON': '#9b59b6',
    'ROUTES': '#f39c12'
}


# =============================================================================
# Graph Loading
# =============================================================================

def load_graph_from_json(filepath: str) -> Tuple[nx.DiGraph, Dict]:
    """Load graph from JSON file"""
    with open(filepath) as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes
    for node in data.get('nodes', []):
        node_attrs = {k: v for k, v in node.items() if k != 'id'}
        node_attrs['type'] = 'Node'
        node_attrs['layer'] = Layer.INFRASTRUCTURE.value
        if 'name' not in node_attrs:
            node_attrs['name'] = node['id']
        G.add_node(node['id'], **node_attrs)
    
    for broker in data.get('brokers', []):
        broker_attrs = {k: v for k, v in broker.items() if k != 'id'}
        broker_attrs['type'] = 'Broker'
        broker_attrs['layer'] = Layer.BROKER.value
        if 'name' not in broker_attrs:
            broker_attrs['name'] = broker['id']
        G.add_node(broker['id'], **broker_attrs)
    
    for app in data.get('applications', []):
        app_attrs = {k: v for k, v in app.items() if k != 'id'}
        app_attrs['type'] = 'Application'
        app_attrs['layer'] = Layer.APPLICATION.value
        if 'name' not in app_attrs:
            app_attrs['name'] = app['id']
        G.add_node(app['id'], **app_attrs)
    
    for topic in data.get('topics', []):
        topic_attrs = {k: v for k, v in topic.items() if k not in ['id', 'qos']}
        topic_attrs['type'] = 'Topic'
        topic_attrs['layer'] = Layer.TOPIC.value
        if 'name' not in topic_attrs:
            topic_attrs['name'] = topic['id']
        if 'qos' in topic:
            for qk, qv in topic['qos'].items():
                topic_attrs[f'qos_{qk}'] = qv
        G.add_node(topic['id'], **topic_attrs)
    
    # Add edges
    relationships = data.get('relationships', {})
    
    for rel in relationships.get('runs_on', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        if source and target:
            G.add_edge(source, target, type='RUNS_ON')
    
    for rel in relationships.get('publishes_to', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        if source and target:
            G.add_edge(source, target, type='PUBLISHES_TO')
    
    for rel in relationships.get('subscribes_to', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        if source and target:
            G.add_edge(source, target, type='SUBSCRIBES_TO')
    
    for rel in relationships.get('routes', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        if source and target:
            G.add_edge(source, target, type='ROUTES')
    
    # Derive DEPENDS_ON
    G = derive_dependencies(G)
    
    return G, data


def derive_dependencies(G: nx.DiGraph) -> nx.DiGraph:
    """Derive DEPENDS_ON relationships"""
    topics = [n for n, d in G.nodes(data=True) if d.get('type') == 'Topic']
    
    for topic in topics:
        publishers = [s for s, t, d in G.in_edges(topic, data=True) 
                     if d.get('type') == 'PUBLISHES_TO']
        subscribers = [s for s, t, d in G.in_edges(topic, data=True) 
                      if d.get('type') == 'SUBSCRIBES_TO']
        
        for sub in subscribers:
            for pub in publishers:
                if sub != pub and not G.has_edge(sub, pub):
                    G.add_edge(sub, pub, type='DEPENDS_ON', via_topic=topic)
    
    return G


def filter_graph_by_layer(G: nx.DiGraph, layer: Layer) -> nx.DiGraph:
    """Filter graph to show only specific layer"""
    if layer == Layer.ALL:
        return G
    
    type_mapping = {
        Layer.APPLICATION: ['Application'],
        Layer.TOPIC: ['Topic'],
        Layer.BROKER: ['Broker'],
        Layer.INFRASTRUCTURE: ['Node', 'Broker']
    }
    
    allowed_types = type_mapping.get(layer, [])
    nodes_to_keep = [n for n, d in G.nodes(data=True) 
                    if d.get('type') in allowed_types]
    
    return G.subgraph(nodes_to_keep).copy()


# =============================================================================
# Criticality Calculation
# =============================================================================

def calculate_criticality(G: nx.DiGraph) -> Dict[str, Dict]:
    """Calculate basic criticality metrics for visualization"""
    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    max_bc = max(betweenness.values()) if betweenness else 1.0
    if max_bc == 0:
        max_bc = 1.0
    
    # Articulation points
    undirected = G.to_undirected()
    aps = set(nx.articulation_points(undirected))
    
    # Degree
    degree = dict(G.degree())
    max_degree = max(degree.values()) if degree else 1
    
    # Calculate criticality for each node
    criticality = {}
    for node in G.nodes():
        bc_norm = betweenness.get(node, 0) / max_bc
        is_ap = 1.0 if node in aps else 0.0
        deg_norm = degree.get(node, 0) / max_degree
        
        # Simple composite score
        score = 0.4 * bc_norm + 0.3 * is_ap + 0.3 * deg_norm
        
        if score >= 0.8:
            level = 'CRITICAL'
        elif score >= 0.6:
            level = 'HIGH'
        elif score >= 0.4:
            level = 'MEDIUM'
        elif score >= 0.2:
            level = 'LOW'
        else:
            level = 'MINIMAL'
        
        criticality[node] = {
            'score': score,
            'level': level,
            'betweenness': bc_norm,
            'is_articulation_point': node in aps,
            'degree': degree.get(node, 0)
        }
    
    return criticality


# =============================================================================
# Layout Algorithms
# =============================================================================

def compute_layout(G: nx.DiGraph, layout: Layout) -> Dict[str, Tuple[float, float]]:
    """Compute node positions using specified layout algorithm"""
    if len(G) == 0:
        return {}
    
    if layout == Layout.SPRING:
        return nx.spring_layout(G, k=2/np.sqrt(len(G)) if HAS_NUMPY else 1, 
                               iterations=50, seed=42)
    
    elif layout == Layout.HIERARCHICAL:
        # Use spring layout with vertical bias
        pos = nx.spring_layout(G, k=3/np.sqrt(len(G)) if HAS_NUMPY else 1, 
                              iterations=50, seed=42)
        return pos
    
    elif layout == Layout.CIRCULAR:
        return nx.circular_layout(G)
    
    elif layout == Layout.KAMADA_KAWAI:
        try:
            return nx.kamada_kawai_layout(G)
        except:
            return nx.spring_layout(G, seed=42)
    
    elif layout == Layout.SHELL:
        # Group by type for shell layout
        shells = []
        types_order = ['Node', 'Broker', 'Topic', 'Application']
        for node_type in types_order:
            nodes = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
            if nodes:
                shells.append(nodes)
        
        if not shells:
            return nx.spring_layout(G, seed=42)
        return nx.shell_layout(G, shells)
    
    elif layout == Layout.LAYERED:
        return compute_layered_layout(G)
    
    return nx.spring_layout(G, seed=42)


def compute_layered_layout(G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    """Compute layered layout based on node types"""
    pos = {}
    
    # Define layer Y positions
    layer_y = {
        'Node': 0.0,
        'Broker': 0.33,
        'Topic': 0.66,
        'Application': 1.0
    }
    
    # Group nodes by type
    by_type = defaultdict(list)
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'Unknown')
        by_type[node_type].append(node)
    
    # Position nodes
    for node_type, nodes in by_type.items():
        y = layer_y.get(node_type, 0.5)
        n = len(nodes)
        
        for i, node in enumerate(sorted(nodes)):
            x = (i - (n - 1) / 2) / max(1, n - 1) if n > 1 else 0.0
            pos[node] = (x, y)
    
    return pos


# =============================================================================
# HTML Generation
# =============================================================================

def generate_interactive_html(G: nx.DiGraph, 
                             data: Dict,
                             criticality: Dict[str, Dict],
                             color_scheme: ColorScheme,
                             title: str = "Pub-Sub System Visualization",
                             analysis: Optional[Dict] = None) -> str:
    """Generate interactive HTML visualization using Vis.js"""
    
    # Prepare nodes
    nodes_data = []
    for node in G.nodes():
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'Unknown')
        crit = criticality.get(node, {})
        
        # Determine color
        if color_scheme == ColorScheme.TYPE:
            color = TYPE_COLORS.get(node_type, '#95a5a6')
        elif color_scheme == ColorScheme.CRITICALITY:
            color = CRITICALITY_COLORS.get(crit.get('level', 'MINIMAL'), '#95a5a6')
        elif color_scheme == ColorScheme.LAYER:
            layer = node_data.get('layer', 'unknown')
            try:
                color = LAYER_COLORS.get(Layer(layer), '#95a5a6')
            except:
                color = '#95a5a6'
        else:
            color = TYPE_COLORS.get(node_type, '#95a5a6')
        
        # Node size based on criticality
        size = 15 + (crit.get('score', 0.5) * 25)
        
        # Border for articulation points
        border_width = 4 if crit.get('is_articulation_point', False) else 2
        border_color = '#e74c3c' if crit.get('is_articulation_point', False) else '#2c3e50'
        
        # Tooltip
        tooltip = f"""<b>{node}</b><br>
Type: {node_type}<br>
Criticality: {crit.get('level', 'N/A')} ({crit.get('score', 0):.3f})<br>
Degree: {crit.get('degree', 0)}<br>
Articulation Point: {'Yes' if crit.get('is_articulation_point') else 'No'}"""
        
        nodes_data.append({
            'id': node,
            'label': node_data.get('name', node)[:20],
            'color': {
                'background': color,
                'border': border_color,
                'highlight': {'background': color, 'border': '#2c3e50'}
            },
            'size': size,
            'borderWidth': border_width,
            'title': tooltip,
            'font': {'size': 12}
        })
    
    # Prepare edges
    edges_data = []
    for source, target, edge_data in G.edges(data=True):
        edge_type = edge_data.get('type', 'Unknown')
        color = EDGE_COLORS.get(edge_type, '#95a5a6')
        
        width = 2
        dashes = False
        
        if edge_type == 'DEPENDS_ON':
            width = 3
            dashes = True
        elif edge_type == 'RUNS_ON':
            width = 1
        
        edges_data.append({
            'from': source,
            'to': target,
            'arrows': 'to',
            'color': {'color': color, 'opacity': 0.8},
            'width': width,
            'dashes': dashes,
            'title': edge_type
        })
    
    # Statistics
    stats = {
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'applications': sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'Application'),
        'topics': sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'Topic'),
        'brokers': sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'Broker'),
        'infrastructure': sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'Node'),
        'critical': sum(1 for c in criticality.values() if c.get('level') == 'CRITICAL'),
        'high': sum(1 for c in criticality.values() if c.get('level') == 'HIGH'),
        'articulation_points': sum(1 for c in criticality.values() if c.get('is_articulation_point'))
    }
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #ecf0f1;
        }}
        
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        
        #header h1 {{
            font-size: 1.8em;
            font-weight: 600;
        }}
        
        #header .timestamp {{
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        #main-container {{
            display: flex;
            height: calc(100vh - 80px);
        }}
        
        #sidebar {{
            width: 280px;
            background: rgba(255,255,255,0.05);
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid rgba(255,255,255,0.1);
        }}
        
        #network-container {{
            flex: 1;
            position: relative;
        }}
        
        #network {{
            width: 100%;
            height: 100%;
            background: #0f0f23;
        }}
        
        .panel {{
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        
        .panel h3 {{
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            color: #a8a8b3;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 8px;
        }}
        
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            font-size: 0.9em;
        }}
        
        .stat-label {{
            color: #a8a8b3;
        }}
        
        .stat-value {{
            font-weight: 600;
        }}
        
        .stat-value.critical {{
            color: #e74c3c;
        }}
        
        .stat-value.high {{
            color: #e67e22;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            padding: 5px 0;
            font-size: 0.85em;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            margin-right: 10px;
            border: 2px solid rgba(255,255,255,0.3);
        }}
        
        .controls {{
            margin-top: 10px;
        }}
        
        .controls button {{
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: none;
            border-radius: 6px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            cursor: pointer;
            font-size: 0.9em;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .controls button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        
        #info-panel {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 8px;
            font-size: 0.85em;
            max-width: 300px;
            display: none;
        }}
        
        #info-panel.visible {{
            display: block;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🔗 {title}</h1>
        <span class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
    </div>
    
    <div id="main-container">
        <div id="sidebar">
            <div class="panel">
                <h3>📊 Statistics</h3>
                <div class="stat-row">
                    <span class="stat-label">Total Nodes</span>
                    <span class="stat-value">{stats['nodes']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Total Edges</span>
                    <span class="stat-value">{stats['edges']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Applications</span>
                    <span class="stat-value">{stats['applications']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Topics</span>
                    <span class="stat-value">{stats['topics']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Brokers</span>
                    <span class="stat-value">{stats['brokers']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Infrastructure</span>
                    <span class="stat-value">{stats['infrastructure']}</span>
                </div>
            </div>
            
            <div class="panel">
                <h3>⚠️ Criticality</h3>
                <div class="stat-row">
                    <span class="stat-label">Critical</span>
                    <span class="stat-value critical">{stats['critical']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">High</span>
                    <span class="stat-value high">{stats['high']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Articulation Points</span>
                    <span class="stat-value critical">{stats['articulation_points']}</span>
                </div>
            </div>
            
            <div class="panel">
                <h3>🎨 Legend - Nodes</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background: {TYPE_COLORS['Application']}"></div>
                    <span>Application</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: {TYPE_COLORS['Topic']}"></div>
                    <span>Topic</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: {TYPE_COLORS['Broker']}"></div>
                    <span>Broker</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: {TYPE_COLORS['Node']}"></div>
                    <span>Infrastructure</span>
                </div>
            </div>
            
            <div class="panel">
                <h3>🔗 Legend - Edges</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background: {EDGE_COLORS['PUBLISHES_TO']}"></div>
                    <span>Publishes To</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: {EDGE_COLORS['SUBSCRIBES_TO']}"></div>
                    <span>Subscribes To</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: {EDGE_COLORS['DEPENDS_ON']}"></div>
                    <span>Depends On</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: {EDGE_COLORS['RUNS_ON']}"></div>
                    <span>Runs On</span>
                </div>
            </div>
            
            <div class="panel controls">
                <h3>🎛️ Controls</h3>
                <button onclick="network.fit()">Fit to View</button>
                <button onclick="resetPhysics()">Reset Layout</button>
                <button onclick="togglePhysics()">Toggle Physics</button>
            </div>
        </div>
        
        <div id="network-container">
            <div id="network"></div>
            <div id="info-panel"></div>
        </div>
    </div>
    
    <script>
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});
        
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        
        var options = {{
            nodes: {{
                shape: 'dot',
                font: {{
                    size: 12,
                    color: '#ecf0f1'
                }},
                shadow: {{
                    enabled: true,
                    color: 'rgba(0,0,0,0.5)',
                    size: 10
                }}
            }},
            edges: {{
                smooth: {{
                    type: 'continuous',
                    roundness: 0.5
                }},
                font: {{
                    size: 10,
                    color: '#a8a8b3'
                }}
            }},
            physics: {{
                enabled: true,
                barnesHut: {{
                    gravitationalConstant: -3000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.04,
                    damping: 0.09
                }},
                stabilization: {{
                    enabled: true,
                    iterations: 200,
                    updateInterval: 25
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                navigationButtons: true,
                keyboard: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        var physicsEnabled = true;
        
        // Event handlers
        network.on('click', function(params) {{
            var infoPanel = document.getElementById('info-panel');
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                infoPanel.innerHTML = node.title;
                infoPanel.classList.add('visible');
            }} else {{
                infoPanel.classList.remove('visible');
            }}
        }});
        
        function resetPhysics() {{
            network.setOptions({{ physics: {{ enabled: true }} }});
            physicsEnabled = true;
        }}
        
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}
    </script>
</body>
</html>"""
    
    return html


# =============================================================================
# Static Image Generation
# =============================================================================

def generate_static_image(G: nx.DiGraph,
                         criticality: Dict[str, Dict],
                         color_scheme: ColorScheme,
                         layout: Layout,
                         output_path: str,
                         dpi: int = 150,
                         figsize: Tuple[int, int] = (16, 12)):
    """Generate static image visualization"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for static image generation")
    
    # Compute positions
    pos = compute_layout(G, layout)
    
    # Prepare colors
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'Unknown')
        crit = criticality.get(node, {})
        
        if color_scheme == ColorScheme.TYPE:
            color = TYPE_COLORS.get(node_type, '#95a5a6')
        elif color_scheme == ColorScheme.CRITICALITY:
            color = CRITICALITY_COLORS.get(crit.get('level', 'MINIMAL'), '#95a5a6')
        else:
            color = TYPE_COLORS.get(node_type, '#95a5a6')
        
        node_colors.append(color)
        node_sizes.append(300 + crit.get('score', 0.5) * 500)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#0f0f23')
    
    # Draw edges
    edge_colors = [EDGE_COLORS.get(G.edges[e].get('type', 'Unknown'), '#95a5a6') 
                   for e in G.edges()]
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, 
                          alpha=0.6, arrows=True, arrowsize=15)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                          node_size=node_sizes, alpha=0.9)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='white')
    
    # Create legend
    legend_elements = []
    for node_type, color in TYPE_COLORS.items():
        if node_type != 'Unknown':
            legend_elements.append(Patch(facecolor=color, label=node_type))
    
    ax.legend(handles=legend_elements, loc='upper left', facecolor='#2c3e50',
             edgecolor='white', labelcolor='white')
    
    ax.set_title('Pub-Sub System Graph', color='white', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, facecolor='#1a1a2e', edgecolor='none',
               bbox_inches='tight')
    plt.close()


# =============================================================================
# Dashboard Generation
# =============================================================================

def generate_dashboard_html(G: nx.DiGraph,
                           data: Dict,
                           criticality: Dict[str, Dict],
                           analysis: Optional[Dict] = None) -> str:
    """Generate comprehensive dashboard HTML"""
    
    # Statistics
    stats = {
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'by_type': defaultdict(int),
        'by_criticality': defaultdict(int),
        'density': nx.density(G),
        'connected': nx.is_weakly_connected(G)
    }
    
    for _, d in G.nodes(data=True):
        stats['by_type'][d.get('type', 'Unknown')] += 1
    
    for c in criticality.values():
        stats['by_criticality'][c.get('level', 'MINIMAL')] += 1
    
    # Top critical components
    sorted_crit = sorted(criticality.items(), key=lambda x: x[1].get('score', 0), reverse=True)
    top_critical = sorted_crit[:10]
    
    # Articulation points
    undirected = G.to_undirected()
    aps = list(nx.articulation_points(undirected))
    
    # Edge type distribution
    edge_types = defaultdict(int)
    for _, _, d in G.edges(data=True):
        edge_types[d.get('type', 'Unknown')] += 1
    
    # Generate node table rows
    node_table_rows = ""
    for node, crit in top_critical:
        level = crit.get('level', 'MINIMAL')
        level_class = level.lower()
        ap_badge = '<span class="badge badge-danger">AP</span>' if crit.get('is_articulation_point') else ''
        node_table_rows += f"""
        <tr>
            <td>{node}</td>
            <td>{G.nodes[node].get('type', 'Unknown')}</td>
            <td>{crit.get('score', 0):.4f}</td>
            <td><span class="badge badge-{level_class}">{level}</span> {ap_badge}</td>
            <td>{crit.get('degree', 0)}</td>
        </tr>"""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Analysis Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f6fa;
            color: #2c3e50;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .card h3 {{
            font-size: 1.1em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
            color: #667eea;
        }}
        
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
        
        .stat-item {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            font-size: 0.85em;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        
        .badge-critical {{
            background: #e74c3c;
            color: white;
        }}
        
        .badge-high {{
            background: #e67e22;
            color: white;
        }}
        
        .badge-medium {{
            background: #f1c40f;
            color: #2c3e50;
        }}
        
        .badge-low {{
            background: #27ae60;
            color: white;
        }}
        
        .badge-minimal {{
            background: #95a5a6;
            color: white;
        }}
        
        .badge-danger {{
            background: #c0392b;
            color: white;
            margin-left: 5px;
        }}
        
        .chart-container {{
            height: 250px;
            margin-top: 15px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 System Analysis Dashboard</h1>
        <p>Comprehensive Pub-Sub System Analysis</p>
        <p style="opacity: 0.8; margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="container">
        <div class="grid">
            <div class="card">
                <h3>📈 Overview</h3>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-value">{stats['nodes']}</div>
                        <div class="stat-label">Total Nodes</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{stats['edges']}</div>
                        <div class="stat-label">Total Edges</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{stats['density']:.4f}</div>
                        <div class="stat-label">Density</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{'✓' if stats['connected'] else '✗'}</div>
                        <div class="stat-label">Connected</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>🏗️ Component Types</h3>
                <div class="chart-container">
                    <canvas id="typeChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h3>⚠️ Criticality Distribution</h3>
                <div class="chart-container">
                    <canvas id="criticalityChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h3>🔗 Edge Types</h3>
                <div class="chart-container">
                    <canvas id="edgeChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>🎯 Top Critical Components</h3>
            <table>
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Type</th>
                        <th>Score</th>
                        <th>Level</th>
                        <th>Degree</th>
                    </tr>
                </thead>
                <tbody>
                    {node_table_rows}
                </tbody>
            </table>
        </div>
        
        <div class="grid" style="margin-top: 20px;">
            <div class="card">
                <h3>🔴 Articulation Points ({len(aps)})</h3>
                <p style="color: #7f8c8d; margin-bottom: 10px;">Nodes whose removal disconnects the graph</p>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    {''.join(f'<span class="badge badge-danger">{ap}</span>' for ap in aps[:20])}
                    {'<span class="badge badge-minimal">+' + str(len(aps) - 20) + ' more</span>' if len(aps) > 20 else ''}
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Type distribution chart
        new Chart(document.getElementById('typeChart'), {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(list(stats['by_type'].keys()))},
                datasets: [{{
                    data: {json.dumps(list(stats['by_type'].values()))},
                    backgroundColor: ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#95a5a6']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ position: 'bottom' }} }}
            }}
        }});
        
        // Criticality chart
        new Chart(document.getElementById('criticalityChart'), {{
            type: 'bar',
            data: {{
                labels: ['Critical', 'High', 'Medium', 'Low', 'Minimal'],
                datasets: [{{
                    label: 'Components',
                    data: [
                        {stats['by_criticality'].get('CRITICAL', 0)},
                        {stats['by_criticality'].get('HIGH', 0)},
                        {stats['by_criticality'].get('MEDIUM', 0)},
                        {stats['by_criticality'].get('LOW', 0)},
                        {stats['by_criticality'].get('MINIMAL', 0)}
                    ],
                    backgroundColor: ['#e74c3c', '#e67e22', '#f1c40f', '#27ae60', '#95a5a6']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});
        
        // Edge types chart
        new Chart(document.getElementById('edgeChart'), {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(list(edge_types.keys()))},
                datasets: [{{
                    data: {json.dumps(list(edge_types.values()))},
                    backgroundColor: ['#27ae60', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ position: 'bottom' }} }}
            }}
        }});
    </script>
</body>
</html>"""
    
    return html


# =============================================================================
# Main CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Multi-layer pub-sub system graph visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input system.json --output viz.html
  %(prog)s --input system.json --output viz.html --layer application --color-by criticality
  %(prog)s --input system.json --output dashboard.html --dashboard
  %(prog)s --input system.json --output graph.png --format png --dpi 300
        """
    )
    
    # Input/Output
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input', '-i', required=True, help='Input JSON file')
    io_group.add_argument('--output', '-o', required=True, help='Output file')
    io_group.add_argument('--format', '-f', choices=['html', 'png', 'svg', 'pdf'],
                         default='html', help='Output format (default: html)')
    io_group.add_argument('--dpi', type=int, default=150,
                         help='DPI for image output (default: 150)')
    
    # Visualization options
    viz_group = parser.add_argument_group('Visualization')
    viz_group.add_argument('--layer', '-l', 
                          choices=['all', 'application', 'infrastructure', 'topic', 'broker'],
                          default='all', help='Layer to visualize (default: all)')
    viz_group.add_argument('--layout',
                          choices=['spring', 'hierarchical', 'circular', 'layered', 
                                  'kamada_kawai', 'shell'],
                          default='spring', help='Layout algorithm (default: spring)')
    viz_group.add_argument('--color-by',
                          choices=['type', 'criticality', 'layer', 'qos'],
                          default='type', help='Color scheme (default: type)')
    viz_group.add_argument('--title', default='Pub-Sub System Visualization',
                          help='Visualization title')
    
    # Dashboard
    dash_group = parser.add_argument_group('Dashboard')
    dash_group.add_argument('--dashboard', action='store_true',
                           help='Generate comprehensive dashboard')
    dash_group.add_argument('--analysis', metavar='FILE',
                           help='Include analysis results from JSON file')
    
    # Verbosity
    verbosity_group = parser.add_argument_group('Verbosity')
    verbosity_group.add_argument('--verbose', '-v', action='store_true')
    verbosity_group.add_argument('--quiet', '-q', action='store_true')
    verbosity_group.add_argument('--no-color', action='store_true')
    
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
    log_level = logging.DEBUG if args.verbose else (logging.ERROR if args.quiet else logging.INFO)
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    if not args.quiet:
        print_header("GRAPH VISUALIZATION")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load graph
    if not args.quiet:
        print_info(f"Loading graph from {args.input}...")
    
    try:
        G, data = load_graph_from_json(args.input)
        print_success(f"Loaded graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print_error(f"Failed to load graph: {e}")
        return 1
    
    # Filter by layer
    layer = Layer(args.layer)
    if layer != Layer.ALL:
        if not args.quiet:
            print_info(f"Filtering to {layer.value} layer...")
        G = filter_graph_by_layer(G, layer)
        print_success(f"Filtered to {G.number_of_nodes()} nodes")
    
    # Calculate criticality
    if not args.quiet:
        print_info("Calculating criticality metrics...")
    criticality = calculate_criticality(G)
    
    # Load analysis if provided
    analysis = None
    if args.analysis:
        try:
            with open(args.analysis) as f:
                analysis = json.load(f)
            print_success(f"Loaded analysis from {args.analysis}")
        except Exception as e:
            print_warning(f"Could not load analysis: {e}")
    
    # Generate output
    output_path = Path(args.output)
    output_format = args.format
    
    # Auto-detect format from extension
    if output_path.suffix.lower() in ['.html', '.htm']:
        output_format = 'html'
    elif output_path.suffix.lower() == '.png':
        output_format = 'png'
    elif output_path.suffix.lower() == '.svg':
        output_format = 'svg'
    elif output_path.suffix.lower() == '.pdf':
        output_format = 'pdf'
    
    try:
        if args.dashboard:
            if not args.quiet:
                print_info("Generating dashboard...")
            html = generate_dashboard_html(G, data, criticality, analysis)
            with open(output_path, 'w') as f:
                f.write(html)
        
        elif output_format == 'html':
            if not args.quiet:
                print_info("Generating interactive HTML...")
            color_scheme = ColorScheme(args.color_by)
            html = generate_interactive_html(G, data, criticality, color_scheme, args.title, analysis)
            with open(output_path, 'w') as f:
                f.write(html)
        
        else:
            if not HAS_MATPLOTLIB:
                print_error("matplotlib required for image output")
                return 1
            
            if not args.quiet:
                print_info(f"Generating {output_format.upper()} image...")
            
            color_scheme = ColorScheme(args.color_by)
            layout = Layout(args.layout)
            generate_static_image(G, criticality, color_scheme, layout, 
                                str(output_path), args.dpi)
        
        print_success(f"Visualization saved to {output_path}")
        
    except Exception as e:
        logging.exception("Visualization failed")
        print_error(f"Failed to generate visualization: {e}")
        return 1
    
    # Print summary
    if not args.quiet:
        print_section("Summary")
        print(f"  Output: {output_path}")
        print(f"  Format: {output_format.upper()}")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        
        crit_counts = defaultdict(int)
        for c in criticality.values():
            crit_counts[c.get('level', 'MINIMAL')] += 1
        
        if crit_counts.get('CRITICAL', 0) > 0 or crit_counts.get('HIGH', 0) > 0:
            print(f"\n  {Colors.WARNING}Criticality Warnings:{Colors.ENDC}")
            if crit_counts.get('CRITICAL', 0) > 0:
                print(f"    {Colors.FAIL}CRITICAL:{Colors.ENDC} {crit_counts['CRITICAL']}")
            if crit_counts.get('HIGH', 0) > 0:
                print(f"    {Colors.WARNING}HIGH:{Colors.ENDC} {crit_counts['HIGH']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())