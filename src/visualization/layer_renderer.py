"""
Layer Renderer

Creates layer-specific visualizations of the system architecture.
Separates and visualizes different architectural layers (application, infrastructure, topic).

Capabilities:
- Application layer view
- Infrastructure layer view
- Topic/message layer view
- Multi-layer composite view
- Layer interaction visualization
- Cross-layer dependency tracking
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path


class Layer(Enum):
    """System layers"""
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    TOPIC = "topic"
    ALL = "all"


@dataclass
class LayerConfig:
    """Configuration for layer rendering"""
    layer: Layer
    show_cross_layer_deps: bool = True
    layout_direction: str = "TB"  # TB (top-bottom), LR (left-right)
    group_by_attribute: Optional[str] = None
    show_statistics: bool = True
    width: int = 1400
    height: int = 900
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'layer': self.layer.value,
            'show_cross_layer_deps': self.show_cross_layer_deps,
            'layout_direction': self.layout_direction,
            'group_by_attribute': self.group_by_attribute,
            'show_statistics': self.show_statistics,
            'width': self.width,
            'height': self.height
        }


class LayerRenderer:
    """
    Creates layer-specific architectural visualizations
    
    Features:
    - Separate layer views
    - Cross-layer dependencies
    - Hierarchical layouts
    - Grouped visualizations
    - Statistics and metrics
    """
    
    def __init__(self):
        """Initialize layer renderer"""
        self.logger = logging.getLogger(__name__)
        
        # Layer colors
        self.layer_colors = {
            Layer.APPLICATION: {
                'bg': '#e3f2fd',
                'border': '#1976d2',
                'node': '#2196f3'
            },
            Layer.INFRASTRUCTURE: {
                'bg': '#fff3e0',
                'border': '#e65100',
                'node': '#ff9800'
            },
            Layer.TOPIC: {
                'bg': '#e8f5e9',
                'border': '#2e7d32',
                'node': '#4caf50'
            }
        }
    
    def render_layer(self,
                    graph: nx.DiGraph,
                    layer: Layer,
                    config: Optional[LayerConfig] = None,
                    output_path: Optional[str] = None) -> str:
        """
        Render a specific layer
        
        Args:
            graph: NetworkX directed graph
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
            Path(output_path).write_text(html)
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
            Path(output_path).write_text(html)
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
            Path(output_path).write_text(html)
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
            group_by: Node attribute to group by
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
            Path(output_path).write_text(html)
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
            
            stats[layer.value] = {
                'node_count': len(layer_graph),
                'edge_count': len(layer_graph.edges()),
                'avg_degree': sum(dict(layer_graph.degree()).values()) / len(layer_graph) if len(layer_graph) > 0 else 0,
                'density': nx.density(layer_graph),
                'components': nx.number_weakly_connected_components(layer_graph)
            }
        
        # Add cross-layer statistics
        cross_deps = self._get_all_cross_layer_deps(graph)
        stats['cross_layer'] = {
            'total_dependencies': len(cross_deps),
            'app_to_infra': len([d for d in cross_deps if d['from_layer'] == 'application' and d['to_layer'] == 'infrastructure']),
            'app_to_topic': len([d for d in cross_deps if d['from_layer'] == 'application' and d['to_layer'] == 'topic']),
            'topic_to_infra': len([d for d in cross_deps if d['from_layer'] == 'topic' and d['to_layer'] == 'infrastructure'])
        }
        
        return stats
    
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
                    cross_deps.append((u, v, u_layer.value if u_layer else 'unknown', 
                                     v_layer.value if v_layer else 'unknown'))
        
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
            'cross_layer': len(cross_layer_edges),
            'density': nx.density(layer_graph) if len(layer_graph) > 0 else 0
        }
        
        # Build node list
        nodes_html = ""
        for node in layer_graph.nodes():
            node_data = layer_graph.nodes[node]
            nodes_html += f"""
            <div class="node-item">
                <span class="node-name">{node}</span>
                <span class="node-degree">Degree: {layer_graph.degree(node)}</span>
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{layer.value.title()} Layer View</title>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .header {{
                    border-bottom: 3px solid {colors['border']};
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                h1 {{
                    color: {colors['border']};
                    margin: 0 0 10px 0;
                }}
                .subtitle {{
                    color: #666;
                    font-size: 14px;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-card {{
                    background: {colors['bg']};
                    border-left: 4px solid {colors['border']};
                    padding: 20px;
                    border-radius: 5px;
                }}
                .stat-value {{
                    font-size: 32px;
                    font-weight: bold;
                    color: {colors['border']};
                    margin: 10px 0;
                }}
                .stat-label {{
                    color: #666;
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .nodes-section {{
                    margin-top: 30px;
                }}
                .nodes-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }}
                .node-item {{
                    background: {colors['bg']};
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 3px solid {colors['node']};
                }}
                .node-name {{
                    font-weight: 600;
                    color: #2c3e50;
                    display: block;
                    margin-bottom: 5px;
                }}
                .node-degree {{
                    font-size: 12px;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{layer.value.title()} Layer</h1>
                    <div class="subtitle">Architectural layer visualization</div>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-label">Components</div>
                        <div class="stat-value">{stats['nodes']}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Connections</div>
                        <div class="stat-value">{stats['edges']}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Cross-Layer Deps</div>
                        <div class="stat-value">{stats['cross_layer']}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Density</div>
                        <div class="stat-value">{stats['density']:.3f}</div>
                    </div>
                </div>
                
                <div class="nodes-section">
                    <h2>Components in {layer.value.title()} Layer</h2>
                    <div class="nodes-grid">
                        {nodes_html}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_multi_layer_html(self,
                                 layers: Dict[Layer, nx.DiGraph],
                                 cross_deps: List[Dict]) -> str:
        """Create HTML for multi-layer view"""
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Layer Architecture View</title>
            <style>
                body {
                    margin: 0;
                    padding: 20px;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: #f5f5f5;
                }
                .container {
                    max-width: 1600px;
                    margin: 0 auto;
                }
                h1 {
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 40px;
                }
                .layers {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                .layer {
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .layer-header {
                    font-size: 24px;
                    font-weight: 600;
                    margin-bottom: 20px;
                    padding-bottom: 15px;
                    border-bottom: 3px solid;
                }
                .app-layer .layer-header { border-color: #1976d2; color: #1976d2; }
                .infra-layer .layer-header { border-color: #e65100; color: #e65100; }
                .topic-layer .layer-header { border-color: #2e7d32; color: #2e7d32; }
                .layer-content {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                }
                .node-box {
                    padding: 15px 20px;
                    border-radius: 5px;
                    font-weight: 500;
                    color: white;
                }
                .app-layer .node-box { background: #2196f3; }
                .infra-layer .node-box { background: #ff9800; }
                .topic-layer .node-box { background: #4caf50; }
                .cross-deps {
                    margin-top: 30px;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .dep-item {
                    padding: 10px;
                    margin: 5px 0;
                    background: #f5f5f5;
                    border-radius: 5px;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Multi-Layer Architecture View</h1>
                <div class="layers">
        """
        
        # Add each layer
        for layer, graph in layers.items():
            layer_class = layer.value
            nodes_html = ''.join([
                f'<div class="node-box">{node}</div>'
                for node in graph.nodes()
            ])
            
            html += f"""
            <div class="layer {layer_class}-layer">
                <div class="layer-header">{layer.value.title()} Layer ({len(graph)} components)</div>
                <div class="layer-content">
                    {nodes_html}
                </div>
            </div>
            """
        
        # Add cross-layer dependencies
        deps_html = ''.join([
            f'<div class="dep-item">{dep["from"]} ({dep["from_layer"]}) → '
            f'{dep["to"]} ({dep["to_layer"]})</div>'
            for dep in cross_deps[:20]  # Show first 20
        ])
        
        html += f"""
                </div>
                <div class="cross-deps">
                    <h2>Cross-Layer Dependencies ({len(cross_deps)} total)</h2>
                    {deps_html}
                    {f'<p>... and {len(cross_deps) - 20} more</p>' if len(cross_deps) > 20 else ''}
                </div>
            </div>
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
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Layer Interactions</title>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                    background: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                }}
                h1 {{ color: #2c3e50; }}
                .layers {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                    margin: 30px 0;
                }}
                .layer-box {{
                    padding: 20px;
                    border-radius: 5px;
                    border: 2px solid #ddd;
                }}
                .interactions {{
                    margin-top: 30px;
                }}
                .interaction-item {{
                    padding: 10px;
                    margin: 5px 0;
                    background: #e3f2fd;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Layer Interactions: {source_layer.value} → {target_layer.value}</h1>
                <div class="layers">
                    <div class="layer-box">
                        <h2>{source_layer.value.title()}</h2>
                        <p>{len(source_graph)} components</p>
                    </div>
                    <div class="layer-box">
                        <h2>{target_layer.value.title()}</h2>
                        <p>{len(target_graph)} components</p>
                    </div>
                </div>
                <div class="interactions">
                    <h2>{len(interactions)} Interactions</h2>
                    {''.join([f'<div class="interaction-item">{u} → {v}</div>' for u, v in interactions[:50]])}
                </div>
            </div>
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
        
        colors = self.layer_colors[layer]
        
        groups_html = ""
        for group_name, nodes in groups.items():
            nodes_list = ', '.join(nodes[:10])
            if len(nodes) > 10:
                nodes_list += f" ... and {len(nodes) - 10} more"
            
            groups_html += f"""
            <div class="group-card">
                <div class="group-name">{group_name}</div>
                <div class="group-count">{len(nodes)} components</div>
                <div class="group-nodes">{nodes_list}</div>
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{layer.value.title()} Layer (Grouped)</title>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                    background: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                }}
                h1 {{ color: {colors['border']}; }}
                .groups {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 30px;
                }}
                .group-card {{
                    background: {colors['bg']};
                    padding: 20px;
                    border-radius: 5px;
                    border-left: 4px solid {colors['border']};
                }}
                .group-name {{
                    font-size: 18px;
                    font-weight: 600;
                    color: {colors['border']};
                    margin-bottom: 10px;
                }}
                .group-count {{
                    font-size: 24px;
                    font-weight: bold;
                    color: {colors['node']};
                    margin: 10px 0;
                }}
                .group-nodes {{
                    font-size: 12px;
                    color: #666;
                    margin-top: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{layer.value.title()} Layer</h1>
                <p>Grouped by: <strong>{group_by}</strong></p>
                <div class="groups">
                    {groups_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
