#!/usr/bin/env python3
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

import networkx as nx
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import json


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


@dataclass
class LayerConfig:
    """Configuration for layer rendering"""
    layer: Layer
    layout: LayoutStyle = LayoutStyle.FORCE_DIRECTED
    show_labels: bool = True
    show_cross_layer_deps: bool = True
    highlight_critical: bool = True
    color_by_criticality: bool = True
    min_criticality_threshold: float = 0.0
    interactive: bool = True


class LayerColors:
    """Color schemes for different layers"""
    APPLICATION = {
        'primary': '#3498db',
        'critical': '#e74c3c',
        'background': '#ecf0f1'
    }
    INFRASTRUCTURE = {
        'primary': '#95a5a6',
        'broker': '#e74c3c',
        'node': '#7f8c8d',
        'background': '#ecf0f1'
    }
    TOPIC = {
        'primary': '#2ecc71',
        'high_qos': '#27ae60',
        'low_qos': '#a6e4a6',
        'background': '#ecf0f1'
    }
    CROSS_LAYER = {
        'dependency': '#e67e22',
        'weak': '#f39c12',
        'strong': '#d35400'
    }


class LayerRenderer:
    """
    Renderer for multi-layer graph visualization
    
    Provides specialized rendering for each layer with appropriate
    layout algorithms, color schemes, and interaction features.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize layer renderer
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.layer_colors = {
            Layer.APPLICATION: LayerColors.APPLICATION,
            Layer.INFRASTRUCTURE: LayerColors.INFRASTRUCTURE,
            Layer.TOPIC: LayerColors.TOPIC
        }
    
    def render_layer(self,
                    graph: nx.DiGraph,
                    layer: Layer,
                    config: Optional[LayerConfig] = None,
                    output_path: Optional[str] = None) -> str:
        """
        Render a single layer
        
        Args:
            graph: Full system graph
            layer: Layer to render
            config: Layer configuration
            output_path: Path to save output
        
        Returns:
            HTML string
        """
        if config is None:
            config = LayerConfig(layer=layer)
        
        self.logger.info(f"Rendering {layer.value} layer...")
        
        # Extract layer
        layer_graph = self._extract_layer(graph, layer)
        
        # Get cross-layer dependencies if enabled
        cross_layer_edges = []
        if config.show_cross_layer_deps:
            cross_layer_edges = self._get_cross_layer_deps(graph, layer)
        
        # Generate HTML
        html = self._create_layer_html(
            layer_graph,
            layer,
            cross_layer_edges,
            config
        )
        
        # Save if path provided
        if output_path:
            Path(output_path).write_text(html, encoding='utf-8')
            self.logger.info(f"Saved layer visualization to {output_path}")
        
        return html
    
    def render_all_layers(self,
                         graph: nx.DiGraph,
                         output_path: Optional[str] = None) -> str:
        """
        Render all layers in a composite view
        
        Args:
            graph: NetworkX directed graph
            output_path: Path to save output
        
        Returns:
            HTML string
        """
        self.logger.info("Rendering all layers...")
        
        # Extract each layer
        app_layer = self._extract_layer(graph, Layer.APPLICATION)
        infra_layer = self._extract_layer(graph, Layer.INFRASTRUCTURE)
        topic_layer = self._extract_layer(graph, Layer.TOPIC)
        
        # Get cross-layer dependencies
        cross_layer_deps = self._get_all_cross_layer_deps(graph)
        
        # Generate HTML
        html = self._create_multi_layer_html(
            {
                Layer.APPLICATION: app_layer,
                Layer.INFRASTRUCTURE: infra_layer,
                Layer.TOPIC: topic_layer
            },
            cross_layer_deps
        )
        
        # Save if path provided
        if output_path:
            Path(output_path).write_text(html, encoding='utf-8')
            self.logger.info(f"Saved multi-layer visualization to {output_path}")
        
        return html
    
    def render_layer_interactions(self,
                                  graph: nx.DiGraph,
                                  source_layer: Layer,
                                  target_layer: Layer,
                                  output_path: Optional[str] = None) -> str:
        """
        Visualize interactions between two layers
        
        Args:
            graph: NetworkX directed graph
            source_layer: Source layer
            target_layer: Target layer
            output_path: Path to save output
        
        Returns:
            HTML string
        """
        self.logger.info(f"Rendering interactions: {source_layer.value} -> {target_layer.value}")
        
        # Extract layers
        source_graph = self._extract_layer(graph, source_layer)
        target_graph = self._extract_layer(graph, target_layer)
        
        # Find interactions
        interactions = []
        for u, v in graph.edges():
            u_layer = self._get_node_layer(graph, u)
            v_layer = self._get_node_layer(graph, v)
            
            if u_layer == source_layer and v_layer == target_layer:
                interactions.append((u, v))
        
        # Generate HTML
        html = self._create_interaction_html(
            source_graph,
            target_graph,
            interactions,
            source_layer,
            target_layer
        )
        
        # Save if path provided
        if output_path:
            Path(output_path).write_text(html, encoding='utf-8')
            self.logger.info(f"Saved interaction visualization to {output_path}")
        
        return html
    
    def render_grouped_layer(self,
                            graph: nx.DiGraph,
                            layer: Layer,
                            group_by: str,
                            output_path: Optional[str] = None) -> str:
        """
        Render layer with nodes grouped by attribute
        
        Args:
            graph: NetworkX directed graph
            layer: Layer to render
            group_by: Node attribute to group by (e.g., 'broker', 'qos', 'criticality')
            output_path: Path to save output
        
        Returns:
            HTML string
        """
        self.logger.info(f"Rendering {layer.value} layer grouped by {group_by}...")
        
        # Extract layer
        layer_graph = self._extract_layer(graph, layer)
        
        # Group nodes
        groups = self._group_nodes(layer_graph, group_by)
        
        # Generate HTML
        html = self._create_grouped_layer_html(
            layer_graph,
            groups,
            layer,
            group_by
        )
        
        # Save if path provided
        if output_path:
            Path(output_path).write_text(html, encoding='utf-8')
            self.logger.info(f"Saved grouped layer visualization to {output_path}")
        
        return html
    
    def get_layer_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Get statistics for each layer
        
        Args:
            graph: NetworkX directed graph
        
        Returns:
            Dictionary with layer statistics
        """
        stats = {}
        
        for layer in [Layer.APPLICATION, Layer.INFRASTRUCTURE, Layer.TOPIC]:
            layer_graph = self._extract_layer(graph, layer)
            
            if len(layer_graph) > 0:
                degrees = dict(layer_graph.degree())
                stats[layer.value] = {
                    'node_count': len(layer_graph),
                    'edge_count': len(layer_graph.edges()),
                    'avg_degree': sum(degrees.values()) / len(degrees),
                    'max_degree': max(degrees.values()),
                    'density': nx.density(layer_graph),
                    'components': nx.number_weakly_connected_components(layer_graph)
                }
            else:
                stats[layer.value] = {
                    'node_count': 0,
                    'edge_count': 0,
                    'avg_degree': 0,
                    'max_degree': 0,
                    'density': 0,
                    'components': 0
                }
        
        # Add cross-layer statistics
        cross_deps = self._get_all_cross_layer_deps(graph)
        stats['cross_layer'] = {
            'total_dependencies': len(cross_deps),
            'app_to_infra': len([d for d in cross_deps 
                                if d['from_layer'] == 'application' 
                                and d['to_layer'] == 'infrastructure']),
            'app_to_topic': len([d for d in cross_deps 
                                if d['from_layer'] == 'application' 
                                and d['to_layer'] == 'topic']),
            'topic_to_infra': len([d for d in cross_deps 
                                  if d['from_layer'] == 'topic' 
                                  and d['to_layer'] == 'infrastructure'])
        }
        
        return stats
    
    # ========================================================================
    # Internal Methods
    # ========================================================================
    
    def _extract_layer(self, graph: nx.DiGraph, layer: Layer) -> nx.DiGraph:
        """Extract subgraph for a specific layer"""
        
        if layer == Layer.ALL:
            return graph.copy()
        
        # Filter nodes by type
        layer_nodes = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            
            if layer == Layer.APPLICATION and node_type == 'Application':
                layer_nodes.append(node)
            elif layer == Layer.INFRASTRUCTURE and node_type in ['Broker', 'Node']:
                layer_nodes.append(node)
            elif layer == Layer.TOPIC and node_type == 'Topic':
                layer_nodes.append(node)
            elif layer == Layer.BROKER and node_type == 'Broker':
                layer_nodes.append(node)
        
        return graph.subgraph(layer_nodes).copy()
    
    def _get_node_layer(self, graph: nx.DiGraph, node: str) -> Optional[Layer]:
        """Determine which layer a node belongs to"""
        
        node_type = graph.nodes[node].get('type', 'Unknown')
        
        if node_type == 'Application':
            return Layer.APPLICATION
        elif node_type in ['Broker', 'Node']:
            return Layer.INFRASTRUCTURE
        elif node_type == 'Topic':
            return Layer.TOPIC
        
        return None
    
    def _get_cross_layer_deps(self,
                              graph: nx.DiGraph,
                              layer: Layer) -> List[Tuple[str, str, str, str]]:
        """Get dependencies crossing into/out of a layer"""
        
        cross_deps = []
        
        for u, v in graph.edges():
            u_layer = self._get_node_layer(graph, u)
            v_layer = self._get_node_layer(graph, v)
            
            if u_layer != v_layer:
                if u_layer == layer or v_layer == layer:
                    u_layer_str = u_layer.value if u_layer else 'unknown'
                    v_layer_str = v_layer.value if v_layer else 'unknown'
                    cross_deps.append((u, v, u_layer_str, v_layer_str))
        
        return cross_deps
    
    def _get_all_cross_layer_deps(self, graph: nx.DiGraph) -> List[Dict]:
        """Get all cross-layer dependencies"""
        
        cross_deps = []
        
        for u, v in graph.edges():
            u_layer = self._get_node_layer(graph, u)
            v_layer = self._get_node_layer(graph, v)
            
            if u_layer and v_layer and u_layer != v_layer:
                cross_deps.append({
                    'from': u,
                    'to': v,
                    'from_layer': u_layer.value,
                    'to_layer': v_layer.value
                })
        
        return cross_deps
    
    def _group_nodes(self, graph: nx.DiGraph, attribute: str) -> Dict[str, List[str]]:
        """Group nodes by attribute"""
        
        groups = {}
        
        for node in graph.nodes():
            value = graph.nodes[node].get(attribute, 'Unknown')
            if value not in groups:
                groups[value] = []
            groups[value].append(node)
        
        return groups
    
    def _create_layer_html(self,
                          layer_graph: nx.DiGraph,
                          layer: Layer,
                          cross_layer_edges: List,
                          config: LayerConfig) -> str:
        """Create HTML for single layer view"""
        
        colors = self.layer_colors[layer]
        
        # Calculate statistics
        stats = {
            'nodes': len(layer_graph),
            'edges': len(layer_graph.edges()),
            'cross_layer': len(cross_layer_edges)
        }
        
        # Prepare nodes data
        nodes_data = []
        for node in layer_graph.nodes():
            node_data = layer_graph.nodes[node]
            criticality = node_data.get('criticality', 0.5)
            
            # Color by criticality if enabled
            if config.color_by_criticality:
                if criticality > 0.7:
                    color = '#e74c3c'
                elif criticality > 0.5:
                    color = '#e67e22'
                elif criticality > 0.3:
                    color = '#f39c12'
                else:
                    color = colors['primary']
            else:
                color = colors['primary']
            
            nodes_data.append({
                'id': node,
                'label': node if config.show_labels else '',
                'color': color,
                'title': f"{node}<br>Type: {node_data.get('type', 'Unknown')}<br>Criticality: {criticality:.3f}",
                'size': 10 + (criticality * 25)
            })
        
        # Prepare edges data
        edges_data = []
        for u, v in layer_graph.edges():
            edges_data.append({
                'from': u,
                'to': v,
                'arrows': 'to',
                'color': {'color': '#7f8c8d'}
            })
        
        # Add cross-layer edges if enabled
        if config.show_cross_layer_deps:
            for u, v, u_layer, v_layer in cross_layer_edges:
                edges_data.append({
                    'from': u,
                    'to': v,
                    'arrows': 'to',
                    'color': {'color': LayerColors.CROSS_LAYER['dependency']},
                    'dashes': True,
                    'width': 3,
                    'title': f"Cross-layer: {u_layer} → {v_layer}"
                })
        
        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{layer.value.title()} Layer Visualization</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f6fa;
        }}
        
        #header {{
            background: linear-gradient(135deg, {colors['primary']} 0%, {self._darken_color(colors['primary'])} 100%);
            color: white;
            padding: 30px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        #network {{
            width: 100%;
            height: calc(100vh - 180px);
            background: white;
        }}
        
        #stats {{
            padding: 20px;
            background: white;
            border-top: 1px solid #dfe4ea;
            display: flex;
            justify-content: space-around;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: {colors['primary']};
        }}
        
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>📊 {layer.value.title()} Layer</h1>
        <p>System Architecture Visualization</p>
    </div>
    
    <div id="network"></div>
    
    <div id="stats">
        <div class="stat-item">
            <div class="stat-value">{stats['nodes']}</div>
            <div class="stat-label">Nodes</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{stats['edges']}</div>
            <div class="stat-label">Edges</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{stats['cross_layer']}</div>
            <div class="stat-label">Cross-Layer</div>
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
                font: {{ size: 14 }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                smooth: {{ type: 'continuous' }},
                shadow: true
            }},
            physics: {{
                barnesHut: {{
                    gravitationalConstant: -5000,
                    springLength: 150
                }},
                stabilization: {{ iterations: 200 }}
            }},
            interaction: {{
                hover: true,
                navigationButtons: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""
        
        return html
    
    def _create_multi_layer_html(self,
                                layers: Dict[Layer, nx.DiGraph],
                                cross_deps: List[Dict]) -> str:
        """Create HTML for multi-layer composite view"""
        
        # Prepare all nodes and edges
        all_nodes = []
        all_edges = []
        
        layer_positions = {
            Layer.APPLICATION: 3,
            Layer.TOPIC: 2,
            Layer.INFRASTRUCTURE: 1
        }
        
        # Add nodes from each layer
        for layer, layer_graph in layers.items():
            if layer == Layer.ALL:
                continue
            
            y_pos = layer_positions.get(layer, 0)
            node_count = len(layer_graph)
            
            for i, node in enumerate(layer_graph.nodes()):
                node_data = layer_graph.nodes[node]
                criticality = node_data.get('criticality', 0.5)
                
                color = self.layer_colors[layer]['primary']
                
                all_nodes.append({
                    'id': node,
                    'label': node,
                    'color': color,
                    'x': (i - node_count/2) * 200,
                    'y': y_pos * 300,
                    'size': 10 + (criticality * 25),
                    'title': f"{node}<br>Layer: {layer.value}<br>Criticality: {criticality:.3f}",
                    'font': {'size': 12}
                })
        
        # Add within-layer edges
        for layer, layer_graph in layers.items():
            if layer == Layer.ALL:
                continue
            
            for u, v in layer_graph.edges():
                all_edges.append({
                    'from': u,
                    'to': v,
                    'arrows': 'to',
                    'color': {'color': '#7f8c8d'},
                    'width': 2
                })
        
        # Add cross-layer edges
        for dep in cross_deps:
            all_edges.append({
                'from': dep['from'],
                'to': dep['to'],
                'arrows': 'to',
                'color': {'color': LayerColors.CROSS_LAYER['dependency']},
                'width': 3,
                'dashes': True,
                'title': f"Cross-layer: {dep['from_layer']} → {dep['to_layer']}"
            })
        
        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Layer System Visualization</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f6fa;
        }}
        
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        #network {{
            width: 100%;
            height: calc(100vh - 150px);
            background: white;
        }}
        
        #legend {{
            position: fixed;
            top: 100px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            border: 2px solid #2c3e50;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🔄 Multi-Layer System Architecture</h1>
        <p>Comprehensive View of Distributed Pub-Sub System</p>
    </div>
    
    <div id="network"></div>
    
    <div id="legend">
        <h3>Layers</h3>
        <div class="legend-item">
            <div class="legend-color" style="background: {LayerColors.APPLICATION['primary']}"></div>
            <span>Application</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {LayerColors.TOPIC['primary']}"></div>
            <span>Topic</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {LayerColors.INFRASTRUCTURE['primary']}"></div>
            <span>Infrastructure</span>
        </div>
        <div class="legend-item" style="margin-top: 20px;">
            <div style="width: 30px; height: 2px; background: {LayerColors.CROSS_LAYER['dependency']}; 
                        border: 1px dashed {LayerColors.CROSS_LAYER['dependency']}; margin-right: 10px;"></div>
            <span>Cross-Layer</span>
        </div>
    </div>

    <script>
        var nodes = new vis.DataSet({json.dumps(all_nodes)});
        var edges = new vis.DataSet({json.dumps(all_edges)});
        
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            nodes: {{
                shape: 'dot',
                font: {{ size: 12 }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                smooth: {{ type: 'continuous' }},
                shadow: true
            }},
            physics: {{
                enabled: false  // Use fixed positions
            }},
            interaction: {{
                hover: true,
                navigationButtons: true,
                zoomView: true,
                dragView: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        network.fit();
    </script>
</body>
</html>
"""
        
        return html
    
    def _create_interaction_html(self,
                                source_graph: nx.DiGraph,
                                target_graph: nx.DiGraph,
                                interactions: List[Tuple[str, str]],
                                source_layer: Layer,
                                target_layer: Layer) -> str:
        """Create HTML for layer interaction view"""
        
        # Prepare nodes
        nodes_data = []
        
        # Add source layer nodes
        for i, node in enumerate(source_graph.nodes()):
            node_data = source_graph.nodes[node]
            nodes_data.append({
                'id': node,
                'label': node,
                'color': self.layer_colors[source_layer]['primary'],
                'x': (i - len(source_graph)/2) * 200,
                'y': 200,
                'title': f"{node}<br>Layer: {source_layer.value}"
            })
        
        # Add target layer nodes
        for i, node in enumerate(target_graph.nodes()):
            node_data = target_graph.nodes[node]
            nodes_data.append({
                'id': node,
                'label': node,
                'color': self.layer_colors[target_layer]['primary'],
                'x': (i - len(target_graph)/2) * 200,
                'y': -200,
                'title': f"{node}<br>Layer: {target_layer.value}"
            })
        
        # Prepare interaction edges
        edges_data = []
        for u, v in interactions:
            edges_data.append({
                'from': u,
                'to': v,
                'arrows': 'to',
                'color': {'color': LayerColors.CROSS_LAYER['dependency']},
                'width': 3,
                'title': f"Interaction: {u} → {v}"
            })
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{source_layer.value.title()} ↔ {target_layer.value.title()} Interactions</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f6fa; }}
        #header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 30px; text-align: center;
        }}
        #network {{ width: 100%; height: calc(100vh - 150px); background: white; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{source_layer.value.title()} ↔ {target_layer.value.title()}</h1>
        <p>{len(interactions)} interactions</p>
    </div>
    <div id="network"></div>
    <script>
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});
        var network = new vis.Network(
            document.getElementById('network'),
            {{ nodes: nodes, edges: edges }},
            {{
                physics: {{ enabled: false }},
                nodes: {{ shape: 'dot', size: 20, font: {{ size: 14 }} }},
                edges: {{ smooth: {{ type: 'curvedCW' }} }}
            }}
        );
    </script>
</body>
</html>
"""
        
        return html
    
    def _create_grouped_layer_html(self,
                                  layer_graph: nx.DiGraph,
                                  groups: Dict[str, List[str]],
                                  layer: Layer,
                                  group_by: str) -> str:
        """Create HTML for grouped layer view"""
        
        # Assign positions based on groups
        nodes_data = []
        group_index = 0
        
        for group_name, group_nodes in groups.items():
            x_offset = group_index * 400
            
            for i, node in enumerate(group_nodes):
                node_data = layer_graph.nodes[node]
                criticality = node_data.get('criticality', 0.5)
                
                nodes_data.append({
                    'id': node,
                    'label': node,
                    'color': self.layer_colors[layer]['primary'],
                    'x': x_offset,
                    'y': (i - len(group_nodes)/2) * 100,
                    'size': 10 + (criticality * 25),
                    'title': f"{node}<br>Group: {group_name}<br>Criticality: {criticality:.3f}",
                    'group': group_name
                })
            
            group_index += 1
        
        # Prepare edges
        edges_data = []
        for u, v in layer_graph.edges():
            edges_data.append({
                'from': u,
                'to': v,
                'arrows': 'to',
                'color': {'color': '#7f8c8d'}
            })
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{layer.value.title()} Layer - Grouped by {group_by}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f6fa; }}
        #header {{ 
            background: linear-gradient(135deg, {self.layer_colors[layer]['primary']} 0%, 
                        {self._darken_color(self.layer_colors[layer]['primary'])} 100%);
            color: white; padding: 30px; text-align: center;
        }}
        #network {{ width: 100%; height: calc(100vh - 150px); background: white; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{layer.value.title()} Layer - Grouped by {group_by}</h1>
        <p>{len(groups)} groups | {len(layer_graph)} nodes</p>
    </div>
    <div id="network"></div>
    <script>
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});
        var network = new vis.Network(
            document.getElementById('network'),
            {{ nodes: nodes, edges: edges }},
            {{
                physics: {{ enabled: false }},
                nodes: {{ shape: 'dot', font: {{ size: 12 }} }},
                interaction: {{ hover: true, navigationButtons: true }}
            }}
        );
    </script>
</body>
</html>
"""
        
        return html
    
    def _darken_color(self, color: str, factor: float = 0.8) -> str:
        """Darken a hex color"""
        color = color.lstrip('#')
        r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        r, g, b = int(r * factor), int(g * factor), int(b * factor)
        return f'#{r:02x}{g:02x}{b:02x}'
