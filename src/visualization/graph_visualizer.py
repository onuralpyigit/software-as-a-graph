"""
Graph Visualizer

Creates interactive and static visualizations of the system graph.
Supports multiple layouts, styling options, and export formats.

Capabilities:
- Interactive HTML visualizations
- Static image generation (PNG, SVG, PDF)
- Multiple layout algorithms
- Node/edge styling based on properties
- Highlighting critical paths
- Filtering and focus views
- Animation support
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path


class LayoutAlgorithm(Enum):
    """Graph layout algorithms"""
    SPRING = "spring"           # Force-directed (default)
    HIERARCHICAL = "hierarchical"  # Top-down layers
    CIRCULAR = "circular"       # Circular arrangement
    KAMADA_KAWAI = "kamada_kawai"  # Force-directed variant
    SPECTRAL = "spectral"       # Eigenvalue-based
    SHELL = "shell"             # Concentric shells
    RANDOM = "random"           # Random placement


class ColorScheme(Enum):
    """Color schemes for visualization"""
    DEFAULT = "default"
    CRITICALITY = "criticality"
    QOS = "qos"
    CENTRALITY = "centrality"
    TYPE = "type"
    CUSTOM = "custom"


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization"""
    layout: LayoutAlgorithm = LayoutAlgorithm.SPRING
    color_scheme: ColorScheme = ColorScheme.TYPE
    node_size_attribute: Optional[str] = None
    edge_width_attribute: Optional[str] = "weight"
    show_labels: bool = True
    show_edge_labels: bool = False
    highlight_nodes: Optional[List[str]] = None
    highlight_edges: Optional[List[Tuple[str, str]]] = None
    filter_node_types: Optional[List[str]] = None
    width: int = 1200
    height: int = 800
    interactive: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'layout': self.layout.value,
            'color_scheme': self.color_scheme.value,
            'node_size_attribute': self.node_size_attribute,
            'edge_width_attribute': self.edge_width_attribute,
            'show_labels': self.show_labels,
            'show_edge_labels': self.show_edge_labels,
            'highlight_nodes': self.highlight_nodes or [],
            'highlight_edges': self.highlight_edges or [],
            'filter_node_types': self.filter_node_types or [],
            'width': self.width,
            'height': self.height,
            'interactive': self.interactive
        }


class GraphVisualizer:
    """
    Creates visualizations of system graphs
    
    Features:
    - Multiple layout algorithms
    - Color coding by properties
    - Interactive HTML output
    - Static image export
    - Highlighting and filtering
    - Customizable styling
    """
    
    def __init__(self):
        """Initialize graph visualizer"""
        self.logger = logging.getLogger(__name__)
        
        # Default color palettes
        self.color_palettes = {
            'type': {
                'Application': '#3498db',  # Blue
                'Broker': '#e74c3c',       # Red
                'Topic': '#2ecc71',        # Green
                'Node': '#f39c12',         # Orange
                'Unknown': '#95a5a6'       # Gray
            },
            'criticality': {
                'CRITICAL': '#e74c3c',     # Red
                'HIGH': '#e67e22',         # Orange
                'MEDIUM': '#f39c12',       # Yellow
                'LOW': '#3498db'           # Blue
            },
            'qos': {
                'HIGH': '#e74c3c',
                'MEDIUM': '#f39c12',
                'LOW': '#2ecc71'
            }
        }
    
    def visualize(self,
                  graph: nx.DiGraph,
                  config: Optional[VisualizationConfig] = None,
                  output_path: Optional[str] = None) -> str:
        """
        Create graph visualization
        
        Args:
            graph: NetworkX directed graph
            config: Visualization configuration
            output_path: Optional path to save output
        
        Returns:
            HTML string or path to saved file
        """
        if config is None:
            config = VisualizationConfig()
        
        self.logger.info(f"Creating {config.layout.value} visualization...")
        
        # Filter graph if needed
        viz_graph = self._filter_graph(graph, config)
        
        # Calculate layout
        pos = self._calculate_layout(viz_graph, config.layout)
        
        # Generate visualization
        if config.interactive:
            html = self._create_interactive_html(viz_graph, pos, config)
        else:
            html = self._create_static_html(viz_graph, pos, config)
        
        # Save if path provided
        if output_path:
            Path(output_path).write_text(html)
            self.logger.info(f"Saved visualization to {output_path}")
            return output_path
        
        return html
    
    def visualize_with_matplotlib(self,
                                  graph: nx.DiGraph,
                                  config: Optional[VisualizationConfig] = None,
                                  output_path: Optional[str] = None):
        """
        Create static visualization using matplotlib
        
        Args:
            graph: NetworkX directed graph
            config: Visualization configuration
            output_path: Path to save image
        
        Requires: matplotlib
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            self.logger.error("matplotlib not installed. Use: pip install matplotlib")
            return
        
        if config is None:
            config = VisualizationConfig()
        
        self.logger.info("Creating matplotlib visualization...")
        
        # Filter graph
        viz_graph = self._filter_graph(graph, config)
        
        # Calculate layout
        pos = self._calculate_layout(viz_graph, config.layout)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
        
        # Get colors
        node_colors = self._get_node_colors(viz_graph, config)
        edge_colors = self._get_edge_colors(viz_graph, config)
        
        # Get sizes
        node_sizes = self._get_node_sizes(viz_graph, config)
        edge_widths = self._get_edge_widths(viz_graph, config)
        
        # Draw edges
        nx.draw_networkx_edges(
            viz_graph, pos,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowsize=20,
            ax=ax,
            alpha=0.6
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            viz_graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            ax=ax,
            alpha=0.9
        )
        
        # Draw labels
        if config.show_labels:
            nx.draw_networkx_labels(
                viz_graph, pos,
                font_size=8,
                ax=ax
            )
        
        # Add legend
        self._add_legend(ax, viz_graph, config)
        
        # Style
        ax.set_title("System Graph Visualization", fontsize=16, pad=20)
        ax.axis('off')
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {output_path}")
            plt.close()
        else:
            plt.show()
    
    def visualize_critical_path(self,
                               graph: nx.DiGraph,
                               start_node: str,
                               end_node: str,
                               config: Optional[VisualizationConfig] = None) -> str:
        """
        Visualize critical path between two nodes
        
        Args:
            graph: NetworkX directed graph
            start_node: Starting node
            end_node: Ending node
            config: Visualization configuration
        
        Returns:
            HTML string
        """
        if config is None:
            config = VisualizationConfig()
        
        # Find shortest path
        try:
            path = nx.shortest_path(graph, start_node, end_node)
            path_edges = list(zip(path[:-1], path[1:]))
        except nx.NetworkXNoPath:
            self.logger.warning(f"No path found from {start_node} to {end_node}")
            path = []
            path_edges = []
        
        # Highlight path
        config.highlight_nodes = path
        config.highlight_edges = path_edges
        
        return self.visualize(graph, config)
    
    def visualize_neighborhood(self,
                              graph: nx.DiGraph,
                              center_node: str,
                              radius: int = 2,
                              config: Optional[VisualizationConfig] = None) -> str:
        """
        Visualize neighborhood around a node
        
        Args:
            graph: NetworkX directed graph
            center_node: Central node
            radius: Number of hops to include
            config: Visualization configuration
        
        Returns:
            HTML string
        """
        if config is None:
            config = VisualizationConfig()
        
        # Get neighborhood
        neighborhood = set([center_node])
        current_layer = {center_node}
        
        for _ in range(radius):
            next_layer = set()
            for node in current_layer:
                # Add predecessors and successors
                next_layer.update(graph.predecessors(node))
                next_layer.update(graph.successors(node))
            neighborhood.update(next_layer)
            current_layer = next_layer
        
        # Create subgraph
        subgraph = graph.subgraph(neighborhood).copy()
        
        # Highlight center
        config.highlight_nodes = [center_node]
        
        return self.visualize(subgraph, config)
    
    def visualize_comparison(self,
                           original_graph: nx.DiGraph,
                           modified_graph: nx.DiGraph,
                           output_path: Optional[str] = None) -> str:
        """
        Create side-by-side comparison visualization
        
        Args:
            original_graph: Original graph
            modified_graph: Modified graph
            output_path: Path to save output
        
        Returns:
            HTML string
        """
        self.logger.info("Creating comparison visualization...")
        
        # Find differences
        added_nodes = set(modified_graph.nodes()) - set(original_graph.nodes())
        removed_nodes = set(original_graph.nodes()) - set(modified_graph.nodes())
        
        added_edges = set(modified_graph.edges()) - set(original_graph.edges())
        removed_edges = set(original_graph.edges()) - set(modified_graph.edges())
        
        # Create HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Graph Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ display: flex; gap: 20px; }}
                .graph {{ flex: 1; }}
                .stats {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
                .diff {{ margin: 10px 0; }}
                .added {{ color: green; }}
                .removed {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Graph Comparison</h1>
            <div class="stats">
                <h2>Changes Summary</h2>
                <div class="diff added">Added Nodes: {len(added_nodes)}</div>
                <div class="diff removed">Removed Nodes: {len(removed_nodes)}</div>
                <div class="diff added">Added Edges: {len(added_edges)}</div>
                <div class="diff removed">Removed Edges: {len(removed_edges)}</div>
            </div>
            <div class="container">
                <div class="graph">
                    <h2>Original Graph</h2>
                    <p>Nodes: {len(original_graph)}, Edges: {len(original_graph.edges())}</p>
                </div>
                <div class="graph">
                    <h2>Modified Graph</h2>
                    <p>Nodes: {len(modified_graph)}, Edges: {len(modified_graph.edges())}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        if output_path:
            Path(output_path).write_text(html)
        
        return html
    
    def create_animation(self,
                        graphs: List[Tuple[str, nx.DiGraph]],
                        output_path: str,
                        config: Optional[VisualizationConfig] = None):
        """
        Create animated visualization showing graph evolution
        
        Args:
            graphs: List of (label, graph) tuples
            output_path: Path to save animation
            config: Visualization configuration
        """
        self.logger.info(f"Creating animation with {len(graphs)} frames...")
        
        if config is None:
            config = VisualizationConfig()
        
        # Create HTML with JavaScript animation
        html = self._create_animation_html(graphs, config)
        
        Path(output_path).write_text(html)
        self.logger.info(f"Saved animation to {output_path}")
    
    def _filter_graph(self,
                     graph: nx.DiGraph,
                     config: VisualizationConfig) -> nx.DiGraph:
        """Filter graph based on configuration"""
        
        if config.filter_node_types:
            # Filter by node type
            filtered_nodes = [
                n for n in graph.nodes()
                if graph.nodes[n].get('type') in config.filter_node_types
            ]
            return graph.subgraph(filtered_nodes).copy()
        
        return graph.copy()
    
    def _calculate_layout(self,
                         graph: nx.DiGraph,
                         algorithm: LayoutAlgorithm) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions using specified layout algorithm"""
        
        if algorithm == LayoutAlgorithm.SPRING:
            pos = nx.spring_layout(graph, k=2, iterations=50)
        elif algorithm == LayoutAlgorithm.HIERARCHICAL:
            pos = nx.nx_agraph.graphviz_layout(graph, prog='dot') if self._has_graphviz() else nx.spring_layout(graph)
        elif algorithm == LayoutAlgorithm.CIRCULAR:
            pos = nx.circular_layout(graph)
        elif algorithm == LayoutAlgorithm.KAMADA_KAWAI:
            pos = nx.kamada_kawai_layout(graph)
        elif algorithm == LayoutAlgorithm.SPECTRAL:
            pos = nx.spectral_layout(graph)
        elif algorithm == LayoutAlgorithm.SHELL:
            pos = nx.shell_layout(graph)
        elif algorithm == LayoutAlgorithm.RANDOM:
            pos = nx.random_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        return pos
    
    def _get_node_colors(self,
                        graph: nx.DiGraph,
                        config: VisualizationConfig) -> List[str]:
        """Get node colors based on color scheme"""
        
        colors = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            
            if config.highlight_nodes and node in config.highlight_nodes:
                colors.append('#ff0000')  # Red for highlighted
            elif config.color_scheme == ColorScheme.TYPE:
                node_type = node_data.get('type', 'Unknown')
                colors.append(self.color_palettes['type'].get(node_type, '#95a5a6'))
            elif config.color_scheme == ColorScheme.CRITICALITY:
                crit_level = node_data.get('criticality_level', 'LOW')
                colors.append(self.color_palettes['criticality'].get(crit_level, '#3498db'))
            elif config.color_scheme == ColorScheme.QOS:
                qos_level = node_data.get('qos_level', 'LOW')
                colors.append(self.color_palettes['qos'].get(qos_level, '#2ecc71'))
            else:
                colors.append('#3498db')
        
        return colors
    
    def _get_edge_colors(self,
                        graph: nx.DiGraph,
                        config: VisualizationConfig) -> List[str]:
        """Get edge colors"""
        
        colors = []
        
        for edge in graph.edges():
            if config.highlight_edges and edge in config.highlight_edges:
                colors.append('#ff0000')
            else:
                colors.append('#7f8c8d')
        
        return colors
    
    def _get_node_sizes(self,
                       graph: nx.DiGraph,
                       config: VisualizationConfig) -> List[float]:
        """Get node sizes based on attribute"""
        
        if config.node_size_attribute:
            sizes = []
            for node in graph.nodes():
                value = graph.nodes[node].get(config.node_size_attribute, 0.5)
                sizes.append(300 + value * 500)
            return sizes
        
        return [500] * len(graph)
    
    def _get_edge_widths(self,
                        graph: nx.DiGraph,
                        config: VisualizationConfig) -> List[float]:
        """Get edge widths based on attribute"""
        
        if config.edge_width_attribute:
            widths = []
            for u, v in graph.edges():
                weight = graph[u][v].get(config.edge_width_attribute, 1.0)
                widths.append(weight)
            return widths
        
        return [1.0] * len(graph.edges())
    
    def _add_legend(self, ax, graph: nx.DiGraph, config: VisualizationConfig):
        """Add legend to matplotlib plot"""
        
        try:
            import matplotlib.patches as mpatches
        except ImportError:
            return
        
        if config.color_scheme == ColorScheme.TYPE:
            # Get unique types
            types = set(graph.nodes[n].get('type', 'Unknown') for n in graph.nodes())
            patches = [
                mpatches.Patch(color=self.color_palettes['type'].get(t, '#95a5a6'), label=t)
                for t in sorted(types)
            ]
            ax.legend(handles=patches, loc='upper left', framealpha=0.9)
    
    def _create_interactive_html(self,
                                graph: nx.DiGraph,
                                pos: Dict,
                                config: VisualizationConfig) -> str:
        """Create interactive HTML visualization using vis.js"""
        
        # Convert graph to vis.js format
        nodes_data = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            color = self._get_node_color_for_html(node, node_data, config)
            
            nodes_data.append({
                'id': node,
                'label': node if config.show_labels else '',
                'color': color,
                'title': self._get_node_tooltip(node, node_data),
                'x': pos[node][0] * 500,
                'y': pos[node][1] * 500
            })
        
        edges_data = []
        for u, v in graph.edges():
            edge_data = graph[u][v]
            
            edges_data.append({
                'from': u,
                'to': v,
                'arrows': 'to',
                'color': {'color': '#7f8c8d'},
                'width': edge_data.get('weight', 1.0),
                'title': self._get_edge_tooltip(u, v, edge_data)
            })
        
        # Create HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive Graph Visualization</title>
            <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style>
                body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
                #network {{ width: 100%; height: 100vh; }}
                #info {{ 
                    position: absolute; 
                    top: 10px; 
                    right: 10px; 
                    background: white; 
                    padding: 15px; 
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    max-width: 300px;
                }}
                .stat {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div id="network"></div>
            <div id="info">
                <h3>Graph Statistics</h3>
                <div class="stat">Nodes: {len(graph)}</div>
                <div class="stat">Edges: {len(graph.edges())}</div>
                <div class="stat">Layout: {config.layout.value}</div>
            </div>
            
            <script type="text/javascript">
                var nodes = new vis.DataSet({nodes_data});
                var edges = new vis.DataSet({edges_data});
                
                var container = document.getElementById('network');
                var data = {{ nodes: nodes, edges: edges }};
                var options = {{
                    physics: false,
                    interaction: {{
                        hover: true,
                        tooltipDelay: 200
                    }},
                    nodes: {{
                        shape: 'dot',
                        size: 20,
                        font: {{ size: 14 }}
                    }},
                    edges: {{
                        smooth: {{ type: 'curvedCW', roundness: 0.2 }}
                    }}
                }};
                
                var network = new vis.Network(container, data, options);
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _create_static_html(self,
                           graph: nx.DiGraph,
                           pos: Dict,
                           config: VisualizationConfig) -> str:
        """Create static HTML visualization using SVG"""
        
        # Create SVG
        svg_elements = []
        
        # Draw edges
        for u, v in graph.edges():
            x1, y1 = pos[u][0] * 500 + 600, pos[u][1] * 500 + 400
            x2, y2 = pos[v][0] * 500 + 600, pos[v][1] * 500 + 400
            
            svg_elements.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="#7f8c8d" stroke-width="2" marker-end="url(#arrowhead)"/>'
            )
        
        # Draw nodes
        for node in graph.nodes():
            node_data = graph.nodes[node]
            x, y = pos[node][0] * 500 + 600, pos[node][1] * 500 + 400
            color = self._get_node_color_for_html(node, node_data, config)
            
            svg_elements.append(
                f'<circle cx="{x}" cy="{y}" r="10" fill="{color}" '
                f'stroke="#fff" stroke-width="2"/>'
            )
            
            if config.show_labels:
                svg_elements.append(
                    f'<text x="{x}" y="{y-15}" text-anchor="middle" '
                    f'font-size="12" fill="#2c3e50">{node}</text>'
                )
        
        svg = '\n'.join(svg_elements)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Graph Visualization</title>
            <style>
                body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; }}
            </style>
        </head>
        <body>
            <h1>Graph Visualization</h1>
            <svg width="{config.width}" height="{config.height}">
                <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="10" 
                            refX="9" refY="3" orient="auto">
                        <polygon points="0 0, 10 3, 0 6" fill="#7f8c8d"/>
                    </marker>
                </defs>
                {svg}
            </svg>
        </body>
        </html>
        """
        
        return html
    
    def _create_animation_html(self,
                              graphs: List[Tuple[str, nx.DiGraph]],
                              config: VisualizationConfig) -> str:
        """Create animated HTML visualization"""
        
        # Placeholder for animation
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Graph Animation</title>
            <style>
                body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; }}
                #controls {{ margin: 20px 0; }}
                button {{ padding: 10px 20px; margin: 5px; }}
            </style>
        </head>
        <body>
            <h1>Graph Evolution Animation</h1>
            <div id="controls">
                <button onclick="prevFrame()">Previous</button>
                <button onclick="play()">Play</button>
                <button onclick="pause()">Pause</button>
                <button onclick="nextFrame()">Next</button>
                <span id="frame-info">Frame 1 of {len(graphs)}</span>
            </div>
            <div id="graph-container"></div>
            <script>
                // Animation logic would go here
                alert("Animation feature coming soon!");
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _get_node_color_for_html(self,
                                 node: str,
                                 node_data: Dict,
                                 config: VisualizationConfig) -> str:
        """Get node color for HTML visualization"""
        
        if config.highlight_nodes and node in config.highlight_nodes:
            return '#ff0000'
        
        if config.color_scheme == ColorScheme.TYPE:
            node_type = node_data.get('type', 'Unknown')
            return self.color_palettes['type'].get(node_type, '#95a5a6')
        
        return '#3498db'
    
    def _get_node_tooltip(self, node: str, node_data: Dict) -> str:
        """Generate tooltip text for node"""
        
        lines = [f"<b>{node}</b>"]
        
        if 'type' in node_data:
            lines.append(f"Type: {node_data['type']}")
        
        if 'criticality_score' in node_data:
            lines.append(f"Criticality: {node_data['criticality_score']:.3f}")
        
        return "<br>".join(lines)
    
    def _get_edge_tooltip(self, u: str, v: str, edge_data: Dict) -> str:
        """Generate tooltip text for edge"""
        
        lines = [f"<b>{u} → {v}</b>"]
        
        if 'weight' in edge_data:
            lines.append(f"Weight: {edge_data['weight']}")
        
        if 'type' in edge_data:
            lines.append(f"Type: {edge_data['type']}")
        
        return "<br>".join(lines)
    
    def _has_graphviz(self) -> bool:
        """Check if graphviz is available"""
        try:
            import pygraphviz
            return True
        except ImportError:
            return False
