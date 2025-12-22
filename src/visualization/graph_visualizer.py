#!/usr/bin/env python3
"""
Multi-Layer Graph Visualizer
==============================

Comprehensive visualization for pub-sub system graphs with multi-layer support.
Renders graphs at both application and infrastructure levels with interactive
HTML output and static image export.

Features:
- Multi-layer visualization (Application, Topic, Broker, Infrastructure)
- Interactive HTML with vis.js
- Static image export (PNG, SVG, PDF)
- Criticality-based coloring
- Cross-layer dependency highlighting
- Dashboard generation
- Multiple layout algorithms

Author: Software-as-a-Graph Research Project
"""

import json
import math
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")

# Optional matplotlib for static images
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================================
# Enums and Constants
# ============================================================================

class Layer(Enum):
    """System layers for multi-layer visualization"""
    INFRASTRUCTURE = "infrastructure"
    BROKER = "broker"
    TOPIC = "topic"
    APPLICATION = "application"
    ALL = "all"


class LayoutAlgorithm(Enum):
    """Available layout algorithms"""
    SPRING = "spring"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    SHELL = "shell"
    KAMADA_KAWAI = "kamada_kawai"
    LAYERED = "layered"
    SPECTRAL = "spectral"


class ColorScheme(Enum):
    """Color schemes for nodes"""
    TYPE = "type"              # Color by component type
    CRITICALITY = "criticality"  # Color by criticality level
    LAYER = "layer"            # Color by system layer
    QOS = "qos"                # Color by QoS policy
    IMPACT = "impact"          # Color by impact score


# Color palettes
class Colors:
    """Color definitions for visualization"""
    
    # Node type colors
    NODE_TYPES = {
        'Application': '#3498db',  # Blue
        'Topic': '#2ecc71',        # Green
        'Broker': '#e74c3c',       # Red
        'Node': '#9b59b6',         # Purple
        'Unknown': '#95a5a6'       # Gray
    }
    
    # Layer colors
    LAYERS = {
        Layer.APPLICATION: '#3498db',
        Layer.TOPIC: '#2ecc71',
        Layer.BROKER: '#e74c3c',
        Layer.INFRASTRUCTURE: '#9b59b6',
        Layer.ALL: '#667eea'
    }
    
    # Edge type colors
    EDGES = {
        'PUBLISHES_TO': '#27ae60',
        'SUBSCRIBES_TO': '#3498db',
        'DEPENDS_ON': '#e74c3c',
        'RUNS_ON': '#9b59b6',
        'CONNECTS_TO': '#f39c12',
        'Unknown': '#95a5a6'
    }
    
    # Criticality colors (gradient from low to critical)
    CRITICALITY = {
        'critical': '#c0392b',
        'high': '#e74c3c',
        'medium': '#f39c12',
        'low': '#27ae60',
        'minimal': '#95a5a6'
    }
    
    # Cross-layer dependency colors
    CROSS_LAYER = {
        'publishes': '#27ae60',
        'subscribes': '#3498db',
        'dependency': '#e74c3c',
        'infrastructure': '#9b59b6'
    }


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class NodeStyle:
    """Style configuration for a node"""
    color: str
    size: int = 25
    shape: str = 'dot'
    border_width: int = 2
    border_color: Optional[str] = None
    font_size: int = 12
    font_color: str = '#333333'
    opacity: float = 1.0
    shadow: bool = True
    
    def to_vis_dict(self) -> Dict[str, Any]:
        """Convert to vis.js node options"""
        return {
            'color': {
                'background': self.color,
                'border': self.border_color or self.color,
                'highlight': {'background': self.color, 'border': '#ffffff'}
            },
            'size': self.size,
            'shape': self.shape,
            'borderWidth': self.border_width,
            'font': {'size': self.font_size, 'color': self.font_color},
            'shadow': self.shadow
        }


@dataclass
class EdgeStyle:
    """Style configuration for an edge"""
    color: str
    width: int = 2
    dashes: bool = False
    arrows: str = 'to'
    smooth_type: str = 'continuous'
    opacity: float = 0.7
    
    def to_vis_dict(self) -> Dict[str, Any]:
        """Convert to vis.js edge options"""
        return {
            'color': {'color': self.color, 'opacity': self.opacity},
            'width': self.width,
            'dashes': self.dashes,
            'arrows': self.arrows,
            'smooth': {'type': self.smooth_type}
        }


@dataclass 
class VisualizationConfig:
    """Configuration for visualization"""
    title: str = "Pub-Sub System Visualization"
    width: int = 1200
    height: int = 800
    background_color: str = '#1a1a2e'
    color_scheme: ColorScheme = ColorScheme.TYPE
    layout: LayoutAlgorithm = LayoutAlgorithm.SPRING
    show_labels: bool = True
    show_legend: bool = True
    physics_enabled: bool = True
    node_spacing: int = 100
    layer_spacing: int = 150
    dpi: int = 150
    font_family: str = "'Segoe UI', Tahoma, sans-serif"


# ============================================================================
# Main Visualizer Class
# ============================================================================

class GraphVisualizer:
    """
    Multi-layer graph visualizer for pub-sub systems.
    
    Supports both interactive HTML (vis.js) and static image (matplotlib) output.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger('GraphVisualizer')
        
        # Cache for processed data
        self._layers: Dict[Layer, Set[str]] = {}
        self._criticality: Dict[str, Dict] = {}
        self._positions: Dict[str, Tuple[float, float]] = {}
    
    def classify_layers(self, graph: nx.DiGraph) -> Dict[Layer, Set[str]]:
        """
        Classify nodes into layers based on type.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dict mapping Layer -> set of node IDs
        """
        layers = {
            Layer.INFRASTRUCTURE: set(),
            Layer.BROKER: set(),
            Layer.TOPIC: set(),
            Layer.APPLICATION: set()
        }
        
        type_to_layer = {
            'Application': Layer.APPLICATION,
            'Topic': Layer.TOPIC,
            'Broker': Layer.BROKER,
            'Node': Layer.INFRASTRUCTURE,
            'Server': Layer.INFRASTRUCTURE,
            'Gateway': Layer.INFRASTRUCTURE
        }
        
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            layer = type_to_layer.get(node_type, Layer.APPLICATION)
            layers[layer].add(node)
        
        self._layers = layers
        return layers
    
    def get_node_style(self, 
                       node: str,
                       node_data: Dict,
                       criticality: Optional[Dict] = None) -> NodeStyle:
        """
        Get style for a node based on configuration.
        
        Args:
            node: Node ID
            node_data: Node attributes
            criticality: Optional criticality data
            
        Returns:
            NodeStyle object
        """
        node_type = node_data.get('type', 'Unknown')
        
        if self.config.color_scheme == ColorScheme.TYPE:
            color = Colors.NODE_TYPES.get(node_type, Colors.NODE_TYPES['Unknown'])
        elif self.config.color_scheme == ColorScheme.CRITICALITY and criticality:
            level = criticality.get('level', 'minimal')
            color = Colors.CRITICALITY.get(level, Colors.CRITICALITY['minimal'])
        elif self.config.color_scheme == ColorScheme.LAYER:
            for layer, nodes in self._layers.items():
                if node in nodes:
                    color = Colors.LAYERS.get(layer, '#95a5a6')
                    break
            else:
                color = '#95a5a6'
        else:
            color = Colors.NODE_TYPES.get(node_type, '#95a5a6')
        
        # Adjust size based on criticality
        size = 25
        if criticality:
            score = criticality.get('score', 0)
            size = int(20 + score * 30)  # 20-50 range
        
        # Add border for articulation points
        border_color = None
        border_width = 2
        if criticality and criticality.get('is_articulation_point'):
            border_color = '#ffffff'
            border_width = 4
        
        return NodeStyle(
            color=color,
            size=size,
            border_width=border_width,
            border_color=border_color
        )
    
    def get_edge_style(self, 
                       source: str,
                       target: str,
                       edge_data: Dict,
                       graph: nx.DiGraph) -> EdgeStyle:
        """
        Get style for an edge.
        
        Args:
            source: Source node
            target: Target node
            edge_data: Edge attributes
            graph: Full graph for context
            
        Returns:
            EdgeStyle object
        """
        edge_type = edge_data.get('type', 'Unknown')
        color = Colors.EDGES.get(edge_type, Colors.EDGES['Unknown'])
        
        # Check if cross-layer
        source_type = graph.nodes[source].get('type')
        target_type = graph.nodes[target].get('type')
        is_cross_layer = source_type != target_type
        
        return EdgeStyle(
            color=color,
            width=3 if is_cross_layer else 2,
            dashes=is_cross_layer
        )
    
    def calculate_layout(self,
                        graph: nx.DiGraph,
                        algorithm: Optional[LayoutAlgorithm] = None) -> Dict[str, Tuple[float, float]]:
        """
        Calculate node positions using specified algorithm.
        
        Args:
            graph: NetworkX graph
            algorithm: Layout algorithm (uses config default if None)
            
        Returns:
            Dict of node -> (x, y) positions
        """
        algorithm = algorithm or self.config.layout
        
        if algorithm == LayoutAlgorithm.SPRING:
            pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        elif algorithm == LayoutAlgorithm.CIRCULAR:
            pos = nx.circular_layout(graph)
        elif algorithm == LayoutAlgorithm.SHELL:
            # Group by type for shell layout
            shells = []
            for layer in [Layer.INFRASTRUCTURE, Layer.BROKER, Layer.TOPIC, Layer.APPLICATION]:
                nodes = list(self._layers.get(layer, []))
                if nodes:
                    shells.append(nodes)
            pos = nx.shell_layout(graph, nlist=shells if shells else None)
        elif algorithm == LayoutAlgorithm.KAMADA_KAWAI:
            pos = nx.kamada_kawai_layout(graph)
        elif algorithm == LayoutAlgorithm.SPECTRAL:
            try:
                pos = nx.spectral_layout(graph)
            except:
                pos = nx.spring_layout(graph, seed=42)
        elif algorithm == LayoutAlgorithm.HIERARCHICAL or algorithm == LayoutAlgorithm.LAYERED:
            pos = self._hierarchical_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42)
        
        # Scale positions
        scale = self.config.width / 4
        self._positions = {
            node: (x * scale, y * scale) 
            for node, (x, y) in pos.items()
        }
        
        return self._positions
    
    def _hierarchical_layout(self, graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Calculate hierarchical layout by layer"""
        positions = {}
        
        layer_y = {
            Layer.INFRASTRUCTURE: 0,
            Layer.BROKER: 1,
            Layer.TOPIC: 2,
            Layer.APPLICATION: 3
        }
        
        for layer, nodes in self._layers.items():
            if not nodes:
                continue
                
            y = layer_y.get(layer, 0)
            sorted_nodes = sorted(nodes)
            n = len(sorted_nodes)
            
            for i, node in enumerate(sorted_nodes):
                x = (i - (n - 1) / 2) * 2 / max(n, 1)
                positions[node] = (x, -y)  # Negative y for top-down
        
        return positions
    
    # =========================================================================
    # HTML Generation
    # =========================================================================
    
    def render_html(self,
                   graph: nx.DiGraph,
                   criticality: Optional[Dict[str, Dict]] = None,
                   title: Optional[str] = None,
                   layer: Layer = Layer.ALL) -> str:
        """
        Render graph as interactive HTML using vis.js.
        
        Args:
            graph: NetworkX graph
            criticality: Optional criticality scores
            title: Optional title
            layer: Layer to render (ALL for complete graph)
            
        Returns:
            HTML string
        """
        title = title or self.config.title
        self._criticality = criticality or {}
        
        # Classify layers
        self.classify_layers(graph)
        
        # Filter graph by layer if needed
        if layer != Layer.ALL:
            nodes = self._layers.get(layer, set())
            graph = graph.subgraph(nodes).copy()
        
        # Prepare node data
        nodes_data = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            crit = self._criticality.get(node, {})
            style = self.get_node_style(node, node_data, crit)
            
            tooltip = self._create_tooltip(node, node_data, crit)
            
            nodes_data.append({
                'id': node,
                'label': self._truncate_label(node_data.get('name', node)),
                'title': tooltip,
                **style.to_vis_dict()
            })
        
        # Prepare edge data
        edges_data = []
        for source, target, data in graph.edges(data=True):
            style = self.get_edge_style(source, target, data, graph)
            
            edges_data.append({
                'from': source,
                'to': target,
                'title': data.get('type', 'Unknown'),
                **style.to_vis_dict()
            })
        
        return self._generate_html(nodes_data, edges_data, title, layer)
    
    def render_multi_layer_html(self,
                               graph: nx.DiGraph,
                               criticality: Optional[Dict[str, Dict]] = None,
                               title: str = "Multi-Layer System Architecture") -> str:
        """
        Render multi-layer visualization with layer separation.
        
        Args:
            graph: NetworkX graph
            criticality: Optional criticality scores
            title: Visualization title
            
        Returns:
            HTML string
        """
        self._criticality = criticality or {}
        self.classify_layers(graph)
        
        # Calculate layered positions
        layer_y = {
            Layer.INFRASTRUCTURE: 0,
            Layer.BROKER: 150,
            Layer.TOPIC: 300,
            Layer.APPLICATION: 450
        }
        
        nodes_data = []
        for layer, nodes in self._layers.items():
            if layer == Layer.ALL:
                continue
                
            base_y = layer_y.get(layer, 0)
            sorted_nodes = sorted(nodes)
            n = len(sorted_nodes)
            
            for i, node in enumerate(sorted_nodes):
                node_data = graph.nodes[node]
                crit = self._criticality.get(node, {})
                style = self.get_node_style(node, node_data, crit)
                
                # Calculate x position
                x = (i - (n - 1) / 2) * 120 if n > 1 else 0
                
                tooltip = self._create_tooltip(node, node_data, crit)
                tooltip += f"<br><b>Layer:</b> {layer.value}"
                
                nodes_data.append({
                    'id': node,
                    'label': self._truncate_label(node_data.get('name', node), 20),
                    'title': tooltip,
                    'x': x,
                    'y': base_y,
                    'fixed': {'y': True},
                    'group': layer.value,
                    **style.to_vis_dict()
                })
        
        # Prepare edges
        edges_data = []
        for source, target, data in graph.edges(data=True):
            style = self.get_edge_style(source, target, data, graph)
            
            edges_data.append({
                'from': source,
                'to': target,
                'title': data.get('type', 'Unknown'),
                **style.to_vis_dict()
            })
        
        return self._generate_multi_layer_html(nodes_data, edges_data, title)
    
    def _create_tooltip(self, node: str, node_data: Dict, crit: Dict) -> str:
        """Create HTML tooltip for node"""
        tooltip = f"<b>{node}</b><br>"
        tooltip += f"Type: {node_data.get('type', 'Unknown')}<br>"
        
        if crit:
            tooltip += f"<br><b>Criticality:</b><br>"
            tooltip += f"Level: {crit.get('level', 'N/A')}<br>"
            tooltip += f"Score: {crit.get('score', 0):.4f}<br>"
            
            if crit.get('is_articulation_point'):
                tooltip += "<span style='color:#e74c3c'><b>⚠ Articulation Point</b></span><br>"
        
        return tooltip
    
    def _truncate_label(self, label: str, max_len: int = 25) -> str:
        """Truncate label for display"""
        if len(label) > max_len:
            return label[:max_len-2] + '..'
        return label
    
    def _generate_html(self,
                      nodes_data: List[Dict],
                      edges_data: List[Dict],
                      title: str,
                      layer: Layer) -> str:
        """Generate complete HTML document"""
        
        layer_color = Colors.LAYERS.get(layer, '#667eea')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: {self.config.font_family};
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #ecf0f1;
        }}
        
        #header {{
            background: linear-gradient(135deg, {layer_color} 0%, {layer_color}cc 100%);
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        
        #header h1 {{
            font-size: 1.5em;
            font-weight: 600;
        }}
        
        #stats {{
            display: flex;
            gap: 30px;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        
        .stat-label {{
            font-size: 0.85em;
            opacity: 0.9;
        }}
        
        #network {{
            width: 100%;
            height: calc(100vh - 80px);
            background: #0f0f23;
        }}
        
        #legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.85);
            padding: 15px 20px;
            border-radius: 10px;
            font-size: 0.85em;
            max-width: 200px;
        }}
        
        #legend h4 {{
            margin-bottom: 10px;
            color: #fff;
            border-bottom: 1px solid #444;
            padding-bottom: 5px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            margin-right: 10px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        #controls {{
            position: fixed;
            top: 100px;
            left: 20px;
            background: rgba(0,0,0,0.85);
            padding: 15px;
            border-radius: 10px;
        }}
        
        #controls button {{
            display: block;
            width: 100%;
            padding: 8px 15px;
            margin: 5px 0;
            background: {layer_color};
            border: none;
            border-radius: 5px;
            color: white;
            cursor: pointer;
            font-size: 0.85em;
        }}
        
        #controls button:hover {{
            opacity: 0.9;
        }}
        
        #info {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.85);
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 0.8em;
            color: #888;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>📊 {title}</h1>
        <div id="stats">
            <div class="stat">
                <div class="stat-value">{len(nodes_data)}</div>
                <div class="stat-label">Nodes</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(edges_data)}</div>
                <div class="stat-label">Edges</div>
            </div>
        </div>
    </div>
    
    <div id="network"></div>
    
    <div id="controls">
        <button onclick="network.fit()">Fit View</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
        <button onclick="resetLayout()">Reset Layout</button>
    </div>
    
    <div id="legend">
        <h4>Node Types</h4>
        <div class="legend-item">
            <div class="legend-color" style="background: {Colors.NODE_TYPES['Application']}"></div>
            <span>Application</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {Colors.NODE_TYPES['Topic']}"></div>
            <span>Topic</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {Colors.NODE_TYPES['Broker']}"></div>
            <span>Broker</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {Colors.NODE_TYPES['Node']}"></div>
            <span>Infrastructure</span>
        </div>
    </div>
    
    <div id="info">
        Generated: {timestamp}
    </div>
    
    <script>
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});
        
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        
        var options = {{
            nodes: {{
                shape: 'dot',
                font: {{ size: 12, color: '#ecf0f1' }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                smooth: {{ type: 'continuous' }},
                shadow: true
            }},
            physics: {{
                enabled: {str(self.config.physics_enabled).lower()},
                barnesHut: {{
                    gravitationalConstant: -4000,
                    springLength: {self.config.node_spacing},
                    springConstant: 0.04,
                    damping: 0.09
                }},
                stabilization: {{
                    iterations: 150,
                    fit: true
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
        var physicsEnabled = {str(self.config.physics_enabled).lower()};
        
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}
        
        function resetLayout() {{
            network.stabilize(100);
        }}
        
        // Fit on load
        network.once('stabilized', function() {{
            network.fit();
        }});
    </script>
</body>
</html>"""
        
        return html
    
    def _generate_multi_layer_html(self,
                                   nodes_data: List[Dict],
                                   edges_data: List[Dict],
                                   title: str) -> str:
        """Generate HTML for multi-layer view"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: {self.config.font_family};
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #ecf0f1;
        }}
        
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px 30px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        
        #header h1 {{
            font-size: 1.6em;
            margin-bottom: 5px;
        }}
        
        #header p {{
            opacity: 0.9;
            font-size: 0.9em;
        }}
        
        #network {{
            width: 100%;
            height: calc(100vh - 100px);
            background: #0f0f23;
        }}
        
        #layer-labels {{
            position: fixed;
            left: 20px;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            flex-direction: column;
            gap: 100px;
        }}
        
        .layer-label {{
            writing-mode: vertical-rl;
            text-orientation: mixed;
            transform: rotate(180deg);
            padding: 10px 5px;
            border-radius: 5px;
            font-weight: 600;
            font-size: 0.85em;
            text-align: center;
        }}
        
        .layer-app {{ background: {Colors.LAYERS[Layer.APPLICATION]}; }}
        .layer-topic {{ background: {Colors.LAYERS[Layer.TOPIC]}; }}
        .layer-broker {{ background: {Colors.LAYERS[Layer.BROKER]}; }}
        .layer-infra {{ background: {Colors.LAYERS[Layer.INFRASTRUCTURE]}; }}
        
        #legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.9);
            padding: 15px 20px;
            border-radius: 10px;
            font-size: 0.85em;
        }}
        
        .legend-section {{
            margin-bottom: 15px;
        }}
        
        .legend-section h4 {{
            margin-bottom: 8px;
            color: #fff;
            font-size: 0.9em;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
            margin-right: 8px;
        }}
        
        #controls {{
            position: fixed;
            top: 120px;
            right: 20px;
            background: rgba(0,0,0,0.85);
            padding: 15px;
            border-radius: 10px;
        }}
        
        #controls button {{
            display: block;
            width: 100%;
            padding: 8px 15px;
            margin: 5px 0;
            background: #667eea;
            border: none;
            border-radius: 5px;
            color: white;
            cursor: pointer;
            font-size: 0.85em;
        }}
        
        #controls button:hover {{
            background: #5a6fd6;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🔄 {title}</h1>
        <p>Multi-layer architecture visualization • {len(nodes_data)} components</p>
    </div>
    
    <div id="network"></div>
    
    <div id="layer-labels">
        <div class="layer-label layer-app">APPLICATION</div>
        <div class="layer-label layer-topic">TOPIC</div>
        <div class="layer-label layer-broker">BROKER</div>
        <div class="layer-label layer-infra">INFRASTRUCTURE</div>
    </div>
    
    <div id="controls">
        <button onclick="network.fit()">Fit View</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
        <button onclick="toggleLabels()">Toggle Labels</button>
    </div>
    
    <div id="legend">
        <div class="legend-section">
            <h4>Layers</h4>
            <div class="legend-item">
                <div class="legend-color" style="background: {Colors.LAYERS[Layer.APPLICATION]}"></div>
                <span>Application</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {Colors.LAYERS[Layer.TOPIC]}"></div>
                <span>Topic</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {Colors.LAYERS[Layer.BROKER]}"></div>
                <span>Broker</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {Colors.LAYERS[Layer.INFRASTRUCTURE]}"></div>
                <span>Infrastructure</span>
            </div>
        </div>
        <div class="legend-section">
            <h4>Edges</h4>
            <div class="legend-item">
                <div class="legend-color" style="background: {Colors.EDGES['PUBLISHES_TO']}"></div>
                <span>Publishes</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {Colors.EDGES['SUBSCRIBES_TO']}"></div>
                <span>Subscribes</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {Colors.EDGES['DEPENDS_ON']}"></div>
                <span>Depends On</span>
            </div>
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
                font: {{ size: 11, color: '#ecf0f1' }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                smooth: {{ type: 'cubicBezier', roundness: 0.5 }},
                shadow: true
            }},
            physics: {{
                enabled: true,
                barnesHut: {{
                    gravitationalConstant: -2000,
                    springLength: 80,
                    springConstant: 0.04
                }},
                stabilization: {{ iterations: 100 }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true
            }},
            groups: {{
                application: {{ color: '{Colors.LAYERS[Layer.APPLICATION]}' }},
                topic: {{ color: '{Colors.LAYERS[Layer.TOPIC]}' }},
                broker: {{ color: '{Colors.LAYERS[Layer.BROKER]}' }},
                infrastructure: {{ color: '{Colors.LAYERS[Layer.INFRASTRUCTURE]}' }}
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        var physicsEnabled = true;
        var labelsVisible = true;
        
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}
        
        function toggleLabels() {{
            labelsVisible = !labelsVisible;
            var fontSize = labelsVisible ? 11 : 0;
            network.setOptions({{ nodes: {{ font: {{ size: fontSize }} }} }});
        }}
    </script>
</body>
</html>"""
        
        return html
    
    # =========================================================================
    # Static Image Generation (Matplotlib)
    # =========================================================================
    
    def render_image(self,
                    graph: nx.DiGraph,
                    output_path: str,
                    criticality: Optional[Dict[str, Dict]] = None,
                    title: Optional[str] = None,
                    format: str = 'png') -> Optional[str]:
        """
        Render graph as static image using matplotlib.
        
        Args:
            graph: NetworkX graph
            output_path: Output file path
            criticality: Optional criticality scores
            title: Optional title
            format: Output format (png, svg, pdf)
            
        Returns:
            Output path or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available for image rendering")
            return None
        
        title = title or self.config.title
        self._criticality = criticality or {}
        self.classify_layers(graph)
        
        # Calculate layout
        pos = self.calculate_layout(graph)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10), facecolor=self.config.background_color)
        ax.set_facecolor(self.config.background_color)
        
        # Draw edges
        for source, target, data in graph.edges(data=True):
            if source in pos and target in pos:
                x1, y1 = pos[source]
                x2, y2 = pos[target]
                
                style = self.get_edge_style(source, target, data, graph)
                linestyle = '--' if style.dashes else '-'
                
                ax.annotate('',
                           xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(
                               arrowstyle='-|>',
                               color=style.color,
                               alpha=style.opacity,
                               linewidth=style.width,
                               linestyle=linestyle
                           ))
        
        # Draw nodes
        for node in graph.nodes():
            if node not in pos:
                continue
                
            x, y = pos[node]
            node_data = graph.nodes[node]
            crit = self._criticality.get(node, {})
            style = self.get_node_style(node, node_data, crit)
            
            circle = plt.Circle(
                (x, y), 
                style.size / 100,
                color=style.color,
                ec=style.border_color or 'white',
                linewidth=style.border_width,
                zorder=2
            )
            ax.add_patch(circle)
            
            # Add label
            if self.config.show_labels:
                ax.annotate(
                    self._truncate_label(node_data.get('name', node), 15),
                    (x, y - style.size/80),
                    ha='center', va='top',
                    fontsize=8, color='white',
                    zorder=3
                )
        
        # Add legend
        if self.config.show_legend:
            legend_elements = [
                mpatches.Patch(color=Colors.NODE_TYPES['Application'], label='Application'),
                mpatches.Patch(color=Colors.NODE_TYPES['Topic'], label='Topic'),
                mpatches.Patch(color=Colors.NODE_TYPES['Broker'], label='Broker'),
                mpatches.Patch(color=Colors.NODE_TYPES['Node'], label='Infrastructure'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', 
                     facecolor='#2c3e50', edgecolor='none',
                     labelcolor='white', fontsize=9)
        
        # Title
        ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=20)
        
        # Remove axes
        ax.set_xlim(ax.get_xlim()[0] - 0.5, ax.get_xlim()[1] + 0.5)
        ax.set_ylim(ax.get_ylim()[0] - 0.5, ax.get_ylim()[1] + 0.5)
        ax.axis('off')
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, format=format, dpi=self.config.dpi,
                   facecolor=self.config.background_color, 
                   edgecolor='none', bbox_inches='tight')
        plt.close()
        
        return output_path
    
    # =========================================================================
    # Export Functions
    # =========================================================================
    
    def export_for_gephi(self,
                        graph: nx.DiGraph,
                        output_path: str,
                        criticality: Optional[Dict[str, Dict]] = None) -> str:
        """
        Export graph in GEXF format for Gephi.
        
        Args:
            graph: NetworkX graph
            output_path: Output file path
            criticality: Optional criticality scores
            
        Returns:
            Output path
        """
        # Add criticality as node attributes
        if criticality:
            for node, crit in criticality.items():
                if node in graph.nodes():
                    graph.nodes[node]['criticality_score'] = crit.get('score', 0)
                    graph.nodes[node]['criticality_level'] = crit.get('level', 'unknown')
        
        nx.write_gexf(graph, output_path)
        return output_path
    
    def export_for_d3(self,
                     graph: nx.DiGraph,
                     output_path: str,
                     criticality: Optional[Dict[str, Dict]] = None) -> str:
        """
        Export graph in JSON format for D3.js.
        
        Args:
            graph: NetworkX graph
            output_path: Output file path
            criticality: Optional criticality scores
            
        Returns:
            Output path
        """
        self.classify_layers(graph)
        
        nodes = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            crit = criticality.get(node, {}) if criticality else {}
            
            # Determine layer
            layer = 'application'
            for l, node_set in self._layers.items():
                if node in node_set:
                    layer = l.value
                    break
            
            nodes.append({
                'id': node,
                'name': node_data.get('name', node),
                'type': node_data.get('type', 'Unknown'),
                'layer': layer,
                'criticality': crit.get('score', 0),
                'level': crit.get('level', 'minimal')
            })
        
        links = []
        for source, target, data in graph.edges(data=True):
            links.append({
                'source': source,
                'target': target,
                'type': data.get('type', 'Unknown')
            })
        
        d3_data = {'nodes': nodes, 'links': links}
        
        with open(output_path, 'w') as f:
            json.dump(d3_data, f, indent=2)
        
        return output_path
    
    def get_layer_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Get statistics for each layer.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Layer statistics
        """
        self.classify_layers(graph)
        
        stats = {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'layers': {}
        }
        
        for layer, nodes in self._layers.items():
            if layer == Layer.ALL:
                continue
                
            subgraph = graph.subgraph(nodes)
            
            # Count cross-layer edges
            cross_edges = 0
            for source, target in graph.edges():
                source_layer = self._get_node_layer(source)
                target_layer = self._get_node_layer(target)
                if source_layer == layer and target_layer != layer:
                    cross_edges += 1
            
            stats['layers'][layer.value] = {
                'node_count': len(nodes),
                'internal_edges': subgraph.number_of_edges(),
                'cross_layer_edges': cross_edges,
                'density': nx.density(subgraph) if len(nodes) > 1 else 0
            }
        
        return stats
    
    def _get_node_layer(self, node: str) -> Optional[Layer]:
        """Get layer for a node"""
        for layer, nodes in self._layers.items():
            if node in nodes:
                return layer
        return None