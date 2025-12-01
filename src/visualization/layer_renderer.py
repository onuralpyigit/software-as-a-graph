"""
Layer Renderer Module

Specialized renderer for multi-layer graph visualization in pub-sub systems.
Handles individual layer rendering, cross-layer interaction visualization,
and grouped layer views.

Supports:
- Application Layer: Shows application dependencies and message flows
- Infrastructure Layer: Displays brokers, nodes, and physical topology
- Topic Layer: Visualizes topic structure and routing
- Cross-Layer: Highlights dependencies between layers
- Grouped Views: Clusters nodes by attributes (QoS, criticality, domain)
"""

import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from datetime import datetime
import logging

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class Layer(Enum):
    """System layers"""
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    TOPIC = "topic"
    BROKER = "broker"
    ALL = "all"


class LayoutStyle(Enum):
    """Layout styles for layer rendering"""
    FORCE_DIRECTED = "force"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    GRID = "grid"
    LAYERED = "layered"


@dataclass
class LayerConfig:
    """Configuration for layer rendering"""
    layer: Layer
    layout: LayoutStyle = LayoutStyle.FORCE_DIRECTED
    show_labels: bool = True
    show_cross_layer_deps: bool = True
    highlight_critical: bool = True
    color_by_criticality: bool = False
    min_criticality_threshold: float = 0.0
    interactive: bool = True
    show_edge_labels: bool = False


class LayerColors:
    """Color schemes for different layers"""
    APPLICATION = {
        'primary': '#3498db',
        'critical': '#e74c3c',
        'high': '#e67e22',
        'medium': '#f1c40f',
        'low': '#27ae60',
        'minimal': '#95a5a6',
        'background': '#ecf0f1'
    }
    
    INFRASTRUCTURE = {
        'primary': '#9b59b6',
        'node': '#8e44ad',
        'broker': '#e74c3c',
        'critical': '#c0392b',
        'background': '#ecf0f1'
    }
    
    TOPIC = {
        'primary': '#2ecc71',
        'high_qos': '#27ae60',
        'low_qos': '#a6e4a6',
        'god_topic': '#e74c3c',
        'orphan': '#95a5a6',
        'background': '#ecf0f1'
    }
    
    BROKER = {
        'primary': '#e74c3c',
        'overloaded': '#c0392b',
        'normal': '#e67e22',
        'background': '#ecf0f1'
    }
    
    CROSS_LAYER = {
        'dependency': '#e67e22',
        'weak': '#f39c12',
        'strong': '#d35400',
        'runs_on': '#9b59b6',
        'routes': '#e74c3c'
    }
    
    EDGES = {
        'PUBLISHES_TO': '#27ae60',
        'SUBSCRIBES_TO': '#3498db',
        'DEPENDS_ON': '#e74c3c',
        'RUNS_ON': '#9b59b6',
        'ROUTES': '#f39c12'
    }


@dataclass
class CrossLayerDependency:
    """Represents a cross-layer dependency"""
    source: str
    target: str
    source_layer: Layer
    target_layer: Layer
    dependency_type: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LayerRenderer:
    """
    Renderer for multi-layer graph visualization.
    
    Provides specialized rendering for each layer with appropriate
    layout algorithms, color schemes, and interaction features.
    """
    
    def __init__(self, graph: 'nx.DiGraph', config: Optional[LayerConfig] = None):
        """
        Initialize the layer renderer.
        
        Args:
            graph: NetworkX directed graph
            config: Layer rendering configuration
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for layer rendering")
        
        self.graph = graph
        self.config = config or LayerConfig(layer=Layer.ALL)
        self.logger = logging.getLogger(__name__)
        
        # Extract layers
        self._extract_layers()
        
        # Calculate cross-layer dependencies
        self._calculate_cross_layer_deps()
    
    def _extract_layers(self):
        """Extract nodes into their respective layers"""
        self.layers: Dict[Layer, List[str]] = {
            Layer.APPLICATION: [],
            Layer.TOPIC: [],
            Layer.BROKER: [],
            Layer.INFRASTRUCTURE: []
        }
        
        type_to_layer = {
            'Application': Layer.APPLICATION,
            'Topic': Layer.TOPIC,
            'Broker': Layer.BROKER,
            'Node': Layer.INFRASTRUCTURE
        }
        
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            layer = type_to_layer.get(node_type)
            if layer:
                self.layers[layer].append(node)
    
    def _calculate_cross_layer_deps(self):
        """Calculate dependencies between layers"""
        self.cross_layer_deps: List[CrossLayerDependency] = []
        
        type_to_layer = {
            'Application': Layer.APPLICATION,
            'Topic': Layer.TOPIC,
            'Broker': Layer.BROKER,
            'Node': Layer.INFRASTRUCTURE
        }
        
        for source, target, data in self.graph.edges(data=True):
            source_type = self.graph.nodes[source].get('type', 'Unknown')
            target_type = self.graph.nodes[target].get('type', 'Unknown')
            
            source_layer = type_to_layer.get(source_type)
            target_layer = type_to_layer.get(target_type)
            
            if source_layer and target_layer and source_layer != target_layer:
                self.cross_layer_deps.append(CrossLayerDependency(
                    source=source,
                    target=target,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    dependency_type=data.get('type', 'Unknown'),
                    weight=data.get('weight', 1.0),
                    metadata=dict(data)
                ))
    
    def get_layer_subgraph(self, layer: Layer) -> 'nx.DiGraph':
        """Get subgraph for a specific layer"""
        if layer == Layer.ALL:
            return self.graph.copy()
        
        nodes = self.layers.get(layer, [])
        return self.graph.subgraph(nodes).copy()
    
    def render_layer_html(self, 
                         layer: Layer,
                         criticality: Optional[Dict[str, Dict]] = None,
                         title: Optional[str] = None) -> str:
        """
        Render a specific layer as interactive HTML.
        
        Args:
            layer: Layer to render
            criticality: Optional criticality scores
            title: Optional title
        
        Returns:
            HTML string
        """
        subgraph = self.get_layer_subgraph(layer)
        
        if title is None:
            title = f"{layer.value.title()} Layer Visualization"
        
        # Prepare node data
        nodes_data = []
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            crit = criticality.get(node, {}) if criticality else {}
            
            color = self._get_node_color(layer, node_data, crit)
            size = self._get_node_size(crit)
            
            tooltip = self._create_tooltip(node, node_data, crit)
            
            nodes_data.append({
                'id': node,
                'label': node_data.get('name', node)[:25],
                'color': color,
                'size': size,
                'title': tooltip,
                'font': {'size': 11}
            })
        
        # Prepare edge data
        edges_data = []
        for source, target, data in subgraph.edges(data=True):
            edge_type = data.get('type', 'Unknown')
            color = LayerColors.EDGES.get(edge_type, '#95a5a6')
            
            edges_data.append({
                'from': source,
                'to': target,
                'arrows': 'to',
                'color': {'color': color, 'opacity': 0.7},
                'width': 2,
                'title': edge_type
            })
        
        # Generate HTML
        return self._generate_layer_html(nodes_data, edges_data, layer, title)
    
    def render_multi_layer_html(self,
                               criticality: Optional[Dict[str, Dict]] = None,
                               title: str = "Multi-Layer System Architecture") -> str:
        """
        Render all layers in a unified multi-layer view.
        
        Args:
            criticality: Optional criticality scores
            title: Visualization title
        
        Returns:
            HTML string
        """
        # Prepare nodes with layer positioning
        nodes_data = []
        
        layer_y_positions = {
            Layer.INFRASTRUCTURE: 0,
            Layer.BROKER: 150,
            Layer.TOPIC: 300,
            Layer.APPLICATION: 450
        }
        
        for layer, nodes in self.layers.items():
            base_y = layer_y_positions.get(layer, 0)
            n_nodes = len(nodes)
            
            for i, node in enumerate(sorted(nodes)):
                node_data = self.graph.nodes[node]
                crit = criticality.get(node, {}) if criticality else {}
                
                color = self._get_node_color(layer, node_data, crit)
                size = self._get_node_size(crit)
                
                # Calculate x position
                x = (i - (n_nodes - 1) / 2) * 120 if n_nodes > 1 else 0
                
                tooltip = self._create_tooltip(node, node_data, crit)
                tooltip += f"<br>Layer: {layer.value}"
                
                nodes_data.append({
                    'id': node,
                    'label': node_data.get('name', node)[:20],
                    'color': color,
                    'size': size,
                    'x': x,
                    'y': base_y,
                    'fixed': {'y': True},
                    'title': tooltip,
                    'font': {'size': 10},
                    'group': layer.value
                })
        
        # Prepare edges
        edges_data = []
        for source, target, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'Unknown')
            color = LayerColors.EDGES.get(edge_type, '#95a5a6')
            
            # Check if cross-layer
            source_type = self.graph.nodes[source].get('type')
            target_type = self.graph.nodes[target].get('type')
            is_cross_layer = source_type != target_type
            
            width = 3 if is_cross_layer else 2
            dashes = is_cross_layer
            
            edges_data.append({
                'from': source,
                'to': target,
                'arrows': 'to',
                'color': {'color': color, 'opacity': 0.7},
                'width': width,
                'dashes': dashes,
                'title': edge_type
            })
        
        return self._generate_multi_layer_html(nodes_data, edges_data, title)
    
    def render_cross_layer_html(self,
                               criticality: Optional[Dict[str, Dict]] = None,
                               title: str = "Cross-Layer Dependencies") -> str:
        """
        Render only cross-layer dependencies.
        
        Args:
            criticality: Optional criticality scores
            title: Visualization title
        
        Returns:
            HTML string
        """
        # Get nodes involved in cross-layer deps
        cross_layer_nodes = set()
        for dep in self.cross_layer_deps:
            cross_layer_nodes.add(dep.source)
            cross_layer_nodes.add(dep.target)
        
        # Prepare nodes
        nodes_data = []
        for node in cross_layer_nodes:
            node_data = self.graph.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            crit = criticality.get(node, {}) if criticality else {}
            
            # Color by type
            type_colors = {
                'Application': '#3498db',
                'Topic': '#2ecc71',
                'Broker': '#e74c3c',
                'Node': '#9b59b6'
            }
            color = type_colors.get(node_type, '#95a5a6')
            
            size = self._get_node_size(crit)
            tooltip = self._create_tooltip(node, node_data, crit)
            
            nodes_data.append({
                'id': node,
                'label': node_data.get('name', node)[:20],
                'color': color,
                'size': size,
                'title': tooltip,
                'font': {'size': 11}
            })
        
        # Prepare edges (only cross-layer)
        edges_data = []
        for dep in self.cross_layer_deps:
            color = LayerColors.CROSS_LAYER.get(dep.dependency_type.lower(), 
                                                LayerColors.CROSS_LAYER['dependency'])
            
            edges_data.append({
                'from': dep.source,
                'to': dep.target,
                'arrows': 'to',
                'color': {'color': color, 'opacity': 0.8},
                'width': 3,
                'dashes': True,
                'title': f"{dep.dependency_type}: {dep.source_layer.value} → {dep.target_layer.value}"
            })
        
        return self._generate_layer_html(nodes_data, edges_data, Layer.ALL, title)
    
    def get_layer_statistics(self) -> Dict[str, Any]:
        """Get statistics for each layer"""
        stats = {}
        
        for layer, nodes in self.layers.items():
            subgraph = self.get_layer_subgraph(layer)
            
            stats[layer.value] = {
                'nodes': len(nodes),
                'edges': subgraph.number_of_edges(),
                'density': nx.density(subgraph) if len(subgraph) > 0 else 0,
                'avg_degree': sum(dict(subgraph.degree()).values()) / max(1, len(nodes))
            }
        
        # Cross-layer statistics
        stats['cross_layer'] = {
            'total_dependencies': len(self.cross_layer_deps),
            'by_type': defaultdict(int)
        }
        
        for dep in self.cross_layer_deps:
            key = f"{dep.source_layer.value}_to_{dep.target_layer.value}"
            stats['cross_layer']['by_type'][key] += 1
        
        stats['cross_layer']['by_type'] = dict(stats['cross_layer']['by_type'])
        
        return stats
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _get_node_color(self, 
                       layer: Layer, 
                       node_data: Dict, 
                       crit: Dict) -> str:
        """Get node color based on layer and criticality"""
        if self.config.color_by_criticality and crit:
            level = crit.get('level', 'MINIMAL')
            colors = LayerColors.APPLICATION
            return colors.get(level.lower(), colors['minimal'])
        
        layer_colors = {
            Layer.APPLICATION: LayerColors.APPLICATION['primary'],
            Layer.TOPIC: LayerColors.TOPIC['primary'],
            Layer.BROKER: LayerColors.BROKER['primary'],
            Layer.INFRASTRUCTURE: LayerColors.INFRASTRUCTURE['primary']
        }
        
        return layer_colors.get(layer, '#95a5a6')
    
    def _get_node_size(self, crit: Dict) -> int:
        """Get node size based on criticality"""
        score = crit.get('score', 0.5) if crit else 0.5
        return int(15 + score * 25)
    
    def _create_tooltip(self, node: str, node_data: Dict, crit: Dict) -> str:
        """Create HTML tooltip for node"""
        tooltip = f"<b>{node}</b><br>"
        tooltip += f"Type: {node_data.get('type', 'Unknown')}<br>"
        
        if crit:
            tooltip += f"Criticality: {crit.get('level', 'N/A')} ({crit.get('score', 0):.3f})<br>"
            tooltip += f"Degree: {crit.get('degree', 0)}<br>"
            if crit.get('is_articulation_point'):
                tooltip += "<b style='color:#e74c3c'>Articulation Point</b><br>"
        
        return tooltip
    
    def _generate_layer_html(self,
                            nodes_data: List[Dict],
                            edges_data: List[Dict],
                            layer: Layer,
                            title: str) -> str:
        """Generate HTML for single layer view"""
        
        layer_color = {
            Layer.APPLICATION: '#3498db',
            Layer.TOPIC: '#2ecc71',
            Layer.BROKER: '#e74c3c',
            Layer.INFRASTRUCTURE: '#9b59b6',
            Layer.ALL: '#667eea'
        }.get(layer, '#667eea')
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
        }}
        
        #header {{
            background: linear-gradient(135deg, {layer_color} 0%, {layer_color}99 100%);
            color: white;
            padding: 25px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        #header h1 {{
            font-size: 1.6em;
            margin: 0;
        }}
        
        #stats {{
            display: flex;
            gap: 30px;
            font-size: 0.9em;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 1.4em;
            font-weight: bold;
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
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 8px;
            color: white;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 0.85em;
        }}
        
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{title}</h1>
        <div id="stats">
            <div class="stat">
                <div class="stat-value">{len(nodes_data)}</div>
                <div>Nodes</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(edges_data)}</div>
                <div>Edges</div>
            </div>
        </div>
    </div>
    
    <div id="network"></div>
    
    <div id="legend">
        <strong>Edge Types</strong>
        <div class="legend-item">
            <div class="legend-color" style="background: {LayerColors.EDGES['PUBLISHES_TO']}"></div>
            <span>Publishes To</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {LayerColors.EDGES['SUBSCRIBES_TO']}"></div>
            <span>Subscribes To</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {LayerColors.EDGES['DEPENDS_ON']}"></div>
            <span>Depends On</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {LayerColors.EDGES['RUNS_ON']}"></div>
            <span>Runs On</span>
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
                smooth: {{ type: 'continuous' }},
                shadow: true
            }},
            physics: {{
                barnesHut: {{
                    gravitationalConstant: -4000,
                    springLength: 120,
                    springConstant: 0.04
                }},
                stabilization: {{ iterations: 150 }}
            }},
            interaction: {{
                hover: true,
                navigationButtons: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""
        
        return html
    
    def _generate_multi_layer_html(self,
                                  nodes_data: List[Dict],
                                  edges_data: List[Dict],
                                  title: str) -> str:
        """Generate HTML for multi-layer view"""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
        }}
        
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            text-align: center;
        }}
        
        #header h1 {{
            margin: 0 0 10px 0;
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
            flex-direction: column-reverse;
            gap: 100px;
        }}
        
        .layer-label {{
            writing-mode: vertical-lr;
            transform: rotate(180deg);
            color: white;
            font-weight: bold;
            padding: 10px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        
        #legend {{
            position: fixed;
            top: 100px;
            right: 20px;
            background: rgba(0,0,0,0.85);
            padding: 15px;
            border-radius: 8px;
            color: white;
            font-size: 0.85em;
        }}
        
        .legend-section {{
            margin-bottom: 15px;
        }}
        
        .legend-section h4 {{
            margin: 0 0 8px 0;
            font-size: 0.9em;
            color: #a8a8b3;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 4px 0;
        }}
        
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🔗 {title}</h1>
        <p>Hierarchical view showing all system layers and cross-layer dependencies</p>
    </div>
    
    <div id="network"></div>
    
    <div id="layer-labels">
        <div class="layer-label" style="background: #9b59b6">Infrastructure</div>
        <div class="layer-label" style="background: #e74c3c">Broker</div>
        <div class="layer-label" style="background: #2ecc71">Topic</div>
        <div class="layer-label" style="background: #3498db">Application</div>
    </div>
    
    <div id="legend">
        <div class="legend-section">
            <h4>Node Types</h4>
            <div class="legend-item">
                <div class="legend-color" style="background: #3498db"></div>
                <span>Application</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #2ecc71"></div>
                <span>Topic</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #e74c3c"></div>
                <span>Broker</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #9b59b6"></div>
                <span>Infrastructure</span>
            </div>
        </div>
        <div class="legend-section">
            <h4>Edge Types</h4>
            <div class="legend-item">
                <div class="legend-color" style="background: #27ae60"></div>
                <span>Publishes</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #3498db"></div>
                <span>Subscribes</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #e74c3c"></div>
                <span>Depends On</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #9b59b6"></div>
                <span>Runs On</span>
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
                font: {{ size: 10, color: '#ecf0f1' }},
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
                    springLength: 100
                }},
                stabilization: {{ iterations: 100 }}
            }},
            interaction: {{
                hover: true,
                navigationButtons: true
            }},
            layout: {{
                improvedLayout: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""
        
        return html