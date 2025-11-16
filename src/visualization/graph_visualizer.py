#!/usr/bin/env python3
"""
Graph Visualizer Module

General-purpose graph visualization utilities for pub-sub system analysis.
Provides flexible visualization options, layout algorithms, and export formats.

Features:
- Multiple layout algorithms (force-directed, hierarchical, circular, etc.)
- Color schemes (by type, criticality, layer, QoS)
- Static image generation (PNG, SVG, PDF)
- Interactive HTML visualizations
- Failure impact visualization
- Hidden dependency highlighting
- Comparative visualizations (before/after failure)
"""

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import json
import numpy as np


class LayoutAlgorithm(Enum):
    """Available layout algorithms"""
    SPRING = "spring"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    KAMADA_KAWAI = "kamada_kawai"
    SHELL = "shell"
    SPECTRAL = "spectral"
    LAYERED = "layered"


class ColorScheme(Enum):
    """Available color schemes"""
    CRITICALITY = "criticality"
    TYPE = "type"
    LAYER = "layer"
    QOS = "qos"
    DEGREE = "degree"
    FAILURE_IMPACT = "failure_impact"


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization"""
    layout: LayoutAlgorithm = LayoutAlgorithm.SPRING
    color_scheme: ColorScheme = ColorScheme.CRITICALITY
    show_labels: bool = True
    show_legend: bool = True
    highlight_critical: bool = True
    min_criticality_threshold: float = 0.5
    figure_size: Tuple[int, int] = (20, 16)
    dpi: int = 300
    node_size_range: Tuple[int, int] = (300, 1000)
    edge_width_range: Tuple[float, float] = (1.0, 4.0)
    alpha: float = 0.9


class GraphVisualizer:
    """
    Graph visualization engine
    
    Provides comprehensive visualization capabilities for distributed
    pub-sub system graphs with support for multiple layouts, color schemes,
    and export formats.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize graph visualizer
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Color palettes
        self.color_palettes = {
            'criticality': {
                'critical': '#e74c3c',
                'high': '#e67e22',
                'medium': '#f39c12',
                'low': '#27ae60',
                'minimal': '#95a5a6'
            },
            'type': {
                'Application': '#3498db',
                'Topic': '#2ecc71',
                'Broker': '#e74c3c',
                'Node': '#95a5a6',
                'Unknown': '#34495e'
            },
            'layer': {
                'application': '#3498db',
                'topic': '#2ecc71',
                'infrastructure': '#95a5a6'
            }
        }
    
    def visualize_graph(self,
                       graph: nx.DiGraph,
                       output_path: str,
                       config: Optional[VisualizationConfig] = None,
                       criticality_scores: Optional[Dict[str, float]] = None,
                       title: Optional[str] = None) -> str:
        """
        Create graph visualization
        
        Args:
            graph: NetworkX directed graph
            output_path: Path to save visualization
            config: Visualization configuration
            criticality_scores: Optional criticality scores
            title: Optional title
        
        Returns:
            Path to saved visualization
        """
        if config is None:
            config = VisualizationConfig()
        
        self.logger.info(f"Creating visualization with {config.layout.value} layout...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=config.figure_size)
        
        # Compute layout
        pos = self._compute_layout(graph, config.layout)
        
        # Get colors and sizes
        colors = self._get_node_colors(graph, config.color_scheme, criticality_scores)
        sizes = self._get_node_sizes(graph, config, criticality_scores)
        edge_widths = self._get_edge_widths(graph, config)
        
        # Draw graph
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=colors,
            node_size=sizes,
            alpha=config.alpha,
            ax=ax,
            edgecolors='black',
            linewidths=2
        )
        
        nx.draw_networkx_edges(
            graph, pos,
            alpha=0.4,
            arrows=True,
            arrowsize=20,
            edge_color='#7f8c8d',
            width=edge_widths,
            ax=ax
        )
        
        if config.show_labels:
            nx.draw_networkx_labels(
                graph, pos,
                font_size=8,
                font_weight='bold',
                ax=ax
            )
        
        # Add title
        if title:
            ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        else:
            ax.set_title(
                f"System Visualization\n{len(graph)} nodes, {len(graph.edges())} edges",
                fontsize=20, fontweight='bold', pad=20
            )
        
        # Add legend
        if config.show_legend:
            self._add_legend(ax, config.color_scheme, criticality_scores)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"✓ Saved visualization: {output_path}")
        
        return str(output_path)
    
    def visualize_failure_impact(self,
                                 baseline_graph: nx.DiGraph,
                                 failed_graph: nx.DiGraph,
                                 failed_node: str,
                                 output_path: str,
                                 config: Optional[VisualizationConfig] = None) -> str:
        """
        Visualize failure impact on system
        
        Args:
            baseline_graph: Original graph
            failed_graph: Graph after failure
            failed_node: Node that failed
            output_path: Path to save visualization
            config: Visualization configuration
        
        Returns:
            Path to saved visualization
        """
        if config is None:
            config = VisualizationConfig()
        
        self.logger.info(f"Creating failure impact visualization for {failed_node}...")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        
        # Use same layout for both
        pos = self._compute_layout(baseline_graph, config.layout)
        
        # Baseline
        self._draw_subgraph(
            baseline_graph, pos, ax1,
            title="Baseline System",
            config=config,
            highlight_node=failed_node
        )
        
        # Failed
        self._draw_subgraph(
            failed_graph, pos, ax2,
            title=f"After Failure: {failed_node}",
            config=config,
            highlight_node=None
        )
        
        plt.tight_layout()
        
        # Save
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"✓ Saved failure impact visualization: {output_path}")
        
        return str(output_path)
    
    def visualize_hidden_dependencies(self,
                                     graph: nx.DiGraph,
                                     hidden_deps: List[Tuple[str, str, float]],
                                     output_path: str,
                                     config: Optional[VisualizationConfig] = None) -> str:
        """
        Visualize hidden dependencies
        
        Args:
            graph: NetworkX directed graph
            hidden_deps: List of (source, target, strength) tuples
            output_path: Path to save visualization
            config: Visualization configuration
        
        Returns:
            Path to saved visualization
        """
        if config is None:
            config = VisualizationConfig()
        
        self.logger.info("Creating hidden dependencies visualization...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=config.figure_size)
        
        # Compute layout
        pos = self._compute_layout(graph, config.layout)
        
        # Get colors and sizes
        colors = self._get_node_colors(graph, config.color_scheme, None)
        sizes = self._get_node_sizes(graph, config, None)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=colors,
            node_size=sizes,
            alpha=config.alpha,
            ax=ax,
            edgecolors='black',
            linewidths=2
        )
        
        # Draw regular edges (faint)
        nx.draw_networkx_edges(
            graph, pos,
            alpha=0.2,
            arrows=True,
            arrowsize=15,
            edge_color='#bdc3c7',
            width=1,
            ax=ax
        )
        
        # Draw hidden dependencies (emphasized)
        hidden_edge_list = [(u, v) for u, v, _ in hidden_deps]
        hidden_widths = [strength * 5 for _, _, strength in hidden_deps]
        
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=hidden_edge_list,
            alpha=0.8,
            arrows=True,
            arrowsize=25,
            edge_color='#e74c3c',
            width=hidden_widths,
            style='dashed',
            ax=ax
        )
        
        if config.show_labels:
            nx.draw_networkx_labels(
                graph, pos,
                font_size=8,
                font_weight='bold',
                ax=ax
            )
        
        ax.set_title(
            f"Hidden Dependencies\n{len(hidden_deps)} hidden connections identified",
            fontsize=20, fontweight='bold', pad=20
        )
        
        # Add legend
        legend_elements = [
            Patch(facecolor='white', edgecolor='#bdc3c7', 
                  label='Explicit Dependencies', linewidth=2),
            Patch(facecolor='white', edgecolor='#e74c3c', 
                  label='Hidden Dependencies', linestyle='dashed', linewidth=3)
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"✓ Saved hidden dependencies visualization: {output_path}")
        
        return str(output_path)
    
    def create_interactive_html(self,
                              graph: nx.DiGraph,
                              output_path: str,
                              criticality_scores: Optional[Dict[str, float]] = None,
                              title: Optional[str] = None) -> str:
        """
        Create interactive HTML visualization
        
        Args:
            graph: NetworkX directed graph
            output_path: Path to save HTML file
            criticality_scores: Optional criticality scores
            title: Optional title
        
        Returns:
            Path to saved HTML file
        """
        self.logger.info("Creating interactive HTML visualization...")
        
        # Prepare nodes data
        nodes_data = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            
            # Get criticality
            criticality = criticality_scores.get(node, 0.5) if criticality_scores else 0.5
            
            # Color based on criticality
            color = self._get_color_from_criticality(criticality)
            
            # Build tooltip
            tooltip = f"<b>{node}</b><br>"
            tooltip += f"Type: {node_type}<br>"
            tooltip += f"Criticality: {criticality:.3f}<br>"
            tooltip += f"Degree: {graph.degree(node)}"
            
            nodes_data.append({
                'id': node,
                'label': node,
                'title': tooltip,
                'color': color,
                'size': 10 + (criticality * 30),
                'font': {'size': 14}
            })
        
        # Prepare edges data
        edges_data = []
        for u, v in graph.edges():
            edge_data = graph[u][v]
            
            tooltip = f"{u} → {v}"
            if 'type' in edge_data:
                tooltip += f"<br>Type: {edge_data['type']}"
            
            edges_data.append({
                'from': u,
                'to': v,
                'arrows': 'to',
                'title': tooltip,
                'color': {'color': '#848484', 'highlight': '#3498db'}
            })
        
        # Generate HTML
        html = self._generate_interactive_html_template(
            nodes_data, edges_data, title or "System Visualization"
        )
        
        # Save
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            f.write(html)
        
        self.logger.info(f"✓ Saved interactive HTML: {output_path}")
        
        return str(output_path)
    
    # ========================================================================
    # Internal Methods
    # ========================================================================
    
    def _compute_layout(self, graph: nx.DiGraph, layout: LayoutAlgorithm) -> Dict:
        """Compute graph layout"""
        
        if layout == LayoutAlgorithm.SPRING:
            return nx.spring_layout(graph, k=1/np.sqrt(len(graph)), iterations=50)
        elif layout == LayoutAlgorithm.HIERARCHICAL:
            return nx.spring_layout(graph, k=2/np.sqrt(len(graph)))
        elif layout == LayoutAlgorithm.CIRCULAR:
            return nx.circular_layout(graph)
        elif layout == LayoutAlgorithm.KAMADA_KAWAI:
            return nx.kamada_kawai_layout(graph)
        elif layout == LayoutAlgorithm.SHELL:
            # Group by type for shell layout
            node_types = {}
            for node in graph.nodes():
                node_type = graph.nodes[node].get('type', 'Unknown')
                if node_type not in node_types:
                    node_types[node_type] = []
                node_types[node_type].append(node)
            return nx.shell_layout(graph, nlist=list(node_types.values()))
        elif layout == LayoutAlgorithm.SPECTRAL:
            return nx.spectral_layout(graph)
        elif layout == LayoutAlgorithm.LAYERED:
            return self._compute_layered_layout(graph)
        else:
            return nx.spring_layout(graph)
    
    def _compute_layered_layout(self, graph: nx.DiGraph) -> Dict:
        """Compute layered layout based on node types"""
        
        pos = {}
        layers = {
            'Application': [],
            'Topic': [],
            'Broker': [],
            'Node': []
        }
        
        # Group nodes by type
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            if node_type in layers:
                layers[node_type].append(node)
        
        # Assign positions by layer
        y_positions = {'Application': 3, 'Topic': 2, 'Broker': 1, 'Node': 0}
        
        for node_type, nodes in layers.items():
            if not nodes:
                continue
            y = y_positions.get(node_type, 1.5)
            x_spacing = 1.0 / (len(nodes) + 1)
            
            for i, node in enumerate(nodes):
                pos[node] = (x_spacing * (i + 1), y)
        
        return pos
    
    def _get_node_colors(self,
                        graph: nx.DiGraph,
                        color_scheme: ColorScheme,
                        criticality_scores: Optional[Dict[str, float]]) -> List[str]:
        """Get node colors based on color scheme"""
        
        colors = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            
            if color_scheme == ColorScheme.CRITICALITY and criticality_scores:
                score = criticality_scores.get(node, 0.5)
                colors.append(self._get_color_from_criticality(score))
            
            elif color_scheme == ColorScheme.TYPE:
                node_type = node_data.get('type', 'Unknown')
                colors.append(self.color_palettes['type'].get(node_type, '#34495e'))
            
            elif color_scheme == ColorScheme.LAYER:
                node_type = node_data.get('type', 'Unknown')
                if node_type == 'Application':
                    layer = 'application'
                elif node_type == 'Topic':
                    layer = 'topic'
                else:
                    layer = 'infrastructure'
                colors.append(self.color_palettes['layer'].get(layer, '#34495e'))
            
            elif color_scheme == ColorScheme.DEGREE:
                degree = graph.degree(node)
                max_degree = max([d for _, d in graph.degree()]) if len(graph) > 0 else 1
                normalized = degree / max_degree if max_degree > 0 else 0
                colors.append(self._get_color_from_criticality(normalized))
            
            else:
                colors.append('#3498db')  # Default
        
        return colors
    
    def _get_node_sizes(self,
                       graph: nx.DiGraph,
                       config: VisualizationConfig,
                       criticality_scores: Optional[Dict[str, float]]) -> List[float]:
        """Get node sizes"""
        
        sizes = []
        min_size, max_size = config.node_size_range
        
        for node in graph.nodes():
            if criticality_scores and node in criticality_scores:
                score = criticality_scores[node]
                size = min_size + (score * (max_size - min_size))
            else:
                degree = graph.degree(node)
                max_degree = max([d for _, d in graph.degree()]) if len(graph) > 0 else 1
                normalized = degree / max_degree if max_degree > 0 else 0
                size = min_size + (normalized * (max_size - min_size))
            
            sizes.append(size)
        
        return sizes
    
    def _get_edge_widths(self,
                        graph: nx.DiGraph,
                        config: VisualizationConfig) -> List[float]:
        """Get edge widths"""
        
        widths = []
        min_width, max_width = config.edge_width_range
        
        for u, v in graph.edges():
            edge_data = graph[u][v]
            weight = edge_data.get('weight', 1.0)
            width = min_width + ((weight - 1) / 10) * (max_width - min_width)
            widths.append(max(min_width, min(max_width, width)))
        
        return widths
    
    def _get_color_from_criticality(self, score: float) -> str:
        """Map criticality score to color"""
        
        if score > 0.7:
            return self.color_palettes['criticality']['critical']
        elif score > 0.5:
            return self.color_palettes['criticality']['high']
        elif score > 0.3:
            return self.color_palettes['criticality']['medium']
        elif score > 0.1:
            return self.color_palettes['criticality']['low']
        else:
            return self.color_palettes['criticality']['minimal']
    
    def _add_legend(self,
                   ax,
                   color_scheme: ColorScheme,
                   criticality_scores: Optional[Dict[str, float]]):
        """Add legend to plot"""
        
        if color_scheme == ColorScheme.TYPE:
            legend_elements = [
                Patch(facecolor=color, label=node_type, edgecolor='black')
                for node_type, color in self.color_palettes['type'].items()
                if node_type != 'Unknown'
            ]
            ax.legend(handles=legend_elements, loc='upper left', 
                     fontsize=12, framealpha=0.9)
        
        elif color_scheme == ColorScheme.CRITICALITY:
            legend_elements = [
                Patch(facecolor=color, label=f"{level.title()} ({threshold})", 
                      edgecolor='black')
                for level, color, threshold in [
                    ('critical', self.color_palettes['criticality']['critical'], '>0.7'),
                    ('high', self.color_palettes['criticality']['high'], '0.5-0.7'),
                    ('medium', self.color_palettes['criticality']['medium'], '0.3-0.5'),
                    ('low', self.color_palettes['criticality']['low'], '0.1-0.3'),
                    ('minimal', self.color_palettes['criticality']['minimal'], '<0.1')
                ]
            ]
            ax.legend(handles=legend_elements, loc='upper left', 
                     fontsize=12, title='Criticality Score', framealpha=0.9)
    
    def _draw_subgraph(self,
                      graph: nx.DiGraph,
                      pos: Dict,
                      ax,
                      title: str,
                      config: VisualizationConfig,
                      highlight_node: Optional[str] = None):
        """Draw subgraph on axis"""
        
        colors = self._get_node_colors(graph, config.color_scheme, None)
        sizes = self._get_node_sizes(graph, config, None)
        
        # Highlight specific node
        if highlight_node and highlight_node in graph:
            idx = list(graph.nodes()).index(highlight_node)
            colors[idx] = '#e74c3c'
            sizes[idx] = 1500
        
        # Draw
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=colors,
            node_size=sizes,
            alpha=config.alpha,
            ax=ax,
            edgecolors='black',
            linewidths=2
        )
        
        nx.draw_networkx_edges(
            graph, pos,
            alpha=0.4,
            arrows=True,
            arrowsize=20,
            edge_color='#7f8c8d',
            width=2,
            ax=ax
        )
        
        if config.show_labels:
            nx.draw_networkx_labels(
                graph, pos,
                font_size=8,
                font_weight='bold',
                ax=ax
            )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
    
    def _generate_interactive_html_template(self,
                                          nodes_data: List[Dict],
                                          edges_data: List[Dict],
                                          title: str) -> str:
        """Generate interactive HTML template"""
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #f5f6fa; }}
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        #header h1 {{ margin-bottom: 10px; }}
        #network {{ width: 100%; height: calc(100vh - 150px); background: white; }}
        .controls {{
            background: white;
            padding: 15px;
            display: flex;
            gap: 10px;
            border-bottom: 1px solid #dfe4ea;
        }}
        button {{
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
        }}
        button:hover {{ background: #5568d3; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{title}</h1>
        <p>Generated: {timestamp}</p>
    </div>
    <div class="controls">
        <button onclick="network.fit()">Fit View</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
        <button onclick="exportImage()">Export PNG</button>
    </div>
    <div id="network"></div>
    <script>
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});
        var network = new vis.Network(
            document.getElementById('network'),
            {{ nodes: nodes, edges: edges }},
            {{
                nodes: {{ shape: 'dot', font: {{ size: 14 }}, borderWidth: 2, shadow: true }},
                edges: {{ smooth: {{ type: 'continuous' }}, shadow: true }},
                physics: {{ barnesHut: {{ gravitationalConstant: -8000 }} }},
                interaction: {{ hover: true, navigationButtons: true }}
            }}
        );
        var physicsEnabled = true;
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{physics: {{enabled: physicsEnabled}}}});
        }}
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
