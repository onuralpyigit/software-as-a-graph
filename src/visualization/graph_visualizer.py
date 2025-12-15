#!/usr/bin/env python3
"""
Graph Visualizer - Visualization of Pub-Sub Systems
====================================================

Visualizes distributed pub-sub systems including graph topology,
analysis results, simulation impact, and validation comparisons.

Visualization Types:
  - Graph topology with DEPENDS_ON relationships
  - Criticality heatmap showing component importance
  - Impact comparison charts (predicted vs actual)
  - Validation scatter plots with correlation
  - Component ranking bar charts
  - Interactive HTML reports

Usage:
    from src.visualization import GraphVisualizer
    from src.analysis import GraphAnalyzer
    
    analyzer = GraphAnalyzer()
    analyzer.load_from_file('system.json')
    analyzer.analyze()
    
    visualizer = GraphVisualizer(analyzer)
    visualizer.plot_topology('topology.png')
    visualizer.plot_criticality_heatmap('criticality.png')
    visualizer.generate_html_report('report.html')

Author: Software-as-a-Graph Research Project
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import math

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Check for numpy (used by matplotlib)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# Color Schemes
# ============================================================================

class ColorScheme:
    """Color schemes for visualization"""
    
    # Component type colors
    TYPE_COLORS = {
        'Application': '#4CAF50',  # Green
        'Broker': '#2196F3',       # Blue
        'Node': '#FF9800',         # Orange
        'Topic': '#9C27B0',        # Purple
        'Unknown': '#9E9E9E',      # Gray
    }
    
    # Criticality level colors
    CRITICALITY_COLORS = {
        'critical': '#D32F2F',     # Red
        'high': '#FF5722',         # Deep Orange
        'medium': '#FFC107',       # Amber
        'low': '#4CAF50',          # Green
    }
    
    # Dependency type colors
    DEPENDENCY_COLORS = {
        'app_to_app': '#1976D2',       # Blue
        'app_to_broker': '#388E3C',    # Green
        'node_to_node': '#F57C00',     # Orange
        'node_to_broker': '#7B1FA2',   # Purple
    }
    
    # Validation status colors
    STATUS_COLORS = {
        'passed': '#4CAF50',
        'marginal': '#FFC107',
        'failed': '#F44336',
    }
    
    @classmethod
    def get_type_color(cls, node_type: str) -> str:
        return cls.TYPE_COLORS.get(node_type, cls.TYPE_COLORS['Unknown'])
    
    @classmethod
    def get_criticality_color(cls, level: str) -> str:
        return cls.CRITICALITY_COLORS.get(level.lower(), '#9E9E9E')
    
    @classmethod
    def get_impact_color(cls, impact: float) -> str:
        """Get color based on impact score (0-1)"""
        if impact >= 0.7:
            return cls.CRITICALITY_COLORS['critical']
        elif impact >= 0.5:
            return cls.CRITICALITY_COLORS['high']
        elif impact >= 0.3:
            return cls.CRITICALITY_COLORS['medium']
        else:
            return cls.CRITICALITY_COLORS['low']


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 150
    node_size: int = 800
    font_size: int = 10
    edge_width: float = 1.5
    show_labels: bool = True
    show_legend: bool = True
    title_fontsize: int = 14
    colormap: str = 'RdYlGn_r'  # Red-Yellow-Green reversed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'figsize': self.figsize,
            'dpi': self.dpi,
            'node_size': self.node_size,
            'font_size': self.font_size,
            'edge_width': self.edge_width,
            'show_labels': self.show_labels,
            'show_legend': self.show_legend,
        }


# ============================================================================
# Graph Visualizer
# ============================================================================

class GraphVisualizer:
    """
    Visualizes pub-sub system graphs with analysis and simulation results.
    
    Generates static images (PNG/SVG) and interactive HTML reports
    showing graph topology, criticality scores, and validation results.
    """
    
    def __init__(self,
                 analyzer: Optional['GraphAnalyzer'] = None,
                 config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualizer.
        
        Args:
            analyzer: GraphAnalyzer instance with analysis results
            config: Visualization configuration
        """
        self.analyzer = analyzer
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger('graph_visualizer')
        
        # Cache for results
        self._analysis_result = None
        self._simulation_result = None
        self._validation_result = None
        
        # Check matplotlib availability
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available. Image generation disabled.")
    
    def set_analysis_result(self, result: 'AnalysisResult'):
        """Set analysis result for visualization"""
        self._analysis_result = result
    
    def set_simulation_result(self, result: 'BatchSimulationResult'):
        """Set simulation result for visualization"""
        self._simulation_result = result
    
    def set_validation_result(self, result: 'ValidationResult'):
        """Set validation result for visualization"""
        self._validation_result = result
    
    # =========================================================================
    # Graph Topology Visualization
    # =========================================================================
    
    def plot_topology(self,
                     output_path: Optional[str] = None,
                     layout: str = 'spring',
                     color_by: str = 'type',
                     highlight_critical: bool = True) -> Optional[str]:
        """
        Plot graph topology with DEPENDS_ON relationships.
        
        Args:
            output_path: Path to save image (PNG/SVG)
            layout: Layout algorithm ('spring', 'circular', 'shell', 'kamada_kawai')
            color_by: Color nodes by 'type' or 'criticality'
            highlight_critical: Highlight critical components
        
        Returns:
            Path to saved image or None
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available")
            return None
        
        if self.analyzer is None or self.analyzer.G is None:
            self.logger.error("No graph data available")
            return None
        
        G = self.analyzer.G
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Get layout
        pos = self._get_layout(G, layout)
        
        # Get node colors
        node_colors = self._get_node_colors(G, color_by)
        
        # Get node sizes (larger for critical)
        node_sizes = self._get_node_sizes(G, highlight_critical)
        
        # Draw edges by dependency type
        self._draw_edges(G, pos, ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9
        )
        
        # Draw labels
        if self.config.show_labels:
            labels = {n: n for n in G.nodes()}
            nx.draw_networkx_labels(
                G, pos, labels, ax=ax,
                font_size=self.config.font_size,
                font_weight='bold'
            )
        
        # Add legend
        if self.config.show_legend:
            self._add_topology_legend(ax, color_by)
        
        # Title
        ax.set_title(
            f"Pub-Sub System Topology ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
            fontsize=self.config.title_fontsize,
            fontweight='bold'
        )
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save or return
        if output_path:
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close(fig)
            return output_path
        else:
            plt.close(fig)
            return None
    
    def _get_layout(self, G: nx.DiGraph, layout: str) -> Dict:
        """Get node positions using specified layout"""
        if layout == 'spring':
            return nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            return nx.circular_layout(G)
        elif layout == 'shell':
            # Group by type
            shells = self._get_shells_by_type(G)
            return nx.shell_layout(G, shells) if shells else nx.spring_layout(G)
        elif layout == 'kamada_kawai':
            try:
                return nx.kamada_kawai_layout(G)
            except:
                return nx.spring_layout(G, seed=42)
        else:
            return nx.spring_layout(G, seed=42)
    
    def _get_shells_by_type(self, G: nx.DiGraph) -> List[List]:
        """Group nodes by type for shell layout"""
        type_groups = defaultdict(list)
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            type_groups[node_type].append(node)
        
        # Order: Node -> Broker -> Application -> Topic
        order = ['Node', 'Broker', 'Application', 'Topic', 'Unknown']
        shells = []
        for t in order:
            if t in type_groups and type_groups[t]:
                shells.append(type_groups[t])
        
        return shells if shells else None
    
    def _get_node_colors(self, G: nx.DiGraph, color_by: str) -> List[str]:
        """Get node colors based on coloring scheme"""
        colors = []
        
        if color_by == 'criticality' and self._analysis_result:
            # Build criticality lookup
            crit_lookup = {}
            for score in self._analysis_result.criticality_scores:
                crit_lookup[score.node_id] = score.level.value
            
            for node in G.nodes():
                level = crit_lookup.get(node, 'low')
                colors.append(ColorScheme.get_criticality_color(level))
        else:
            # Color by type
            for node, data in G.nodes(data=True):
                node_type = data.get('type', 'Unknown')
                colors.append(ColorScheme.get_type_color(node_type))
        
        return colors
    
    def _get_node_sizes(self, G: nx.DiGraph, highlight_critical: bool) -> List[int]:
        """Get node sizes, optionally larger for critical nodes"""
        base_size = self.config.node_size
        
        if not highlight_critical or not self._analysis_result:
            return [base_size] * G.number_of_nodes()
        
        # Build criticality lookup
        crit_lookup = {}
        for score in self._analysis_result.criticality_scores:
            crit_lookup[score.node_id] = score.level.value
        
        sizes = []
        for node in G.nodes():
            level = crit_lookup.get(node, 'low')
            if level == 'critical':
                sizes.append(base_size * 1.8)
            elif level == 'high':
                sizes.append(base_size * 1.4)
            else:
                sizes.append(base_size)
        
        return sizes
    
    def _draw_edges(self, G: nx.DiGraph, pos: Dict, ax):
        """Draw edges colored by dependency type"""
        edge_colors = []
        for u, v, data in G.edges(data=True):
            dep_type = data.get('dependency_type', 'unknown')
            color = ColorScheme.DEPENDENCY_COLORS.get(dep_type, '#999999')
            edge_colors.append(color)
        
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            width=self.config.edge_width,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            connectionstyle="arc3,rad=0.1"
        )
    
    def _add_topology_legend(self, ax, color_by: str):
        """Add legend to topology plot"""
        patches = []
        
        if color_by == 'criticality':
            for level, color in ColorScheme.CRITICALITY_COLORS.items():
                patches.append(mpatches.Patch(color=color, label=level.capitalize()))
        else:
            for node_type, color in ColorScheme.TYPE_COLORS.items():
                if node_type != 'Unknown':
                    patches.append(mpatches.Patch(color=color, label=node_type))
        
        ax.legend(handles=patches, loc='upper left', fontsize=9)
    
    # =========================================================================
    # Criticality Heatmap
    # =========================================================================
    
    def plot_criticality_heatmap(self,
                                 output_path: Optional[str] = None,
                                 top_n: int = 20) -> Optional[str]:
        """
        Plot criticality scores as a horizontal bar chart.
        
        Args:
            output_path: Path to save image
            top_n: Number of top components to show
        
        Returns:
            Path to saved image or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not self._analysis_result:
            self.logger.error("No analysis result available")
            return None
        
        # Sort by composite score
        scores = sorted(
            self._analysis_result.criticality_scores,
            key=lambda x: x.composite_score,
            reverse=True
        )[:top_n]
        
        if not scores:
            return None
        
        # Prepare data
        components = [s.node_id for s in scores]
        values = [s.composite_score for s in scores]
        colors = [ColorScheme.get_criticality_color(s.level.value) for s in scores]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(components) * 0.4)))
        
        # Create horizontal bar chart
        y_pos = range(len(components))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(components)
        ax.invert_yaxis()
        ax.set_xlabel('Criticality Score')
        ax.set_title(
            f'Component Criticality Ranking (Top {len(components)})',
            fontsize=self.config.title_fontsize,
            fontweight='bold'
        )
        
        # Add legend
        patches = [
            mpatches.Patch(color=ColorScheme.CRITICALITY_COLORS[level], label=level.capitalize())
            for level in ['critical', 'high', 'medium', 'low']
        ]
        ax.legend(handles=patches, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close(fig)
            return output_path
        else:
            plt.close(fig)
            return None
    
    # =========================================================================
    # Impact Comparison Chart
    # =========================================================================
    
    def plot_impact_comparison(self,
                               output_path: Optional[str] = None,
                               top_n: int = 15) -> Optional[str]:
        """
        Plot comparison of predicted criticality vs actual impact.
        
        Args:
            output_path: Path to save image
            top_n: Number of components to show
        
        Returns:
            Path to saved image or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not self._validation_result:
            self.logger.error("No validation result available")
            return None
        
        # Get top components by actual impact
        cv_list = sorted(
            self._validation_result.component_validations,
            key=lambda x: x.actual_impact,
            reverse=True
        )[:top_n]
        
        if not cv_list:
            return None
        
        # Prepare data
        components = [cv.component_id for cv in cv_list]
        predicted = [cv.predicted_score for cv in cv_list]
        actual = [cv.actual_impact for cv in cv_list]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(components))
        width = 0.35
        
        # Create grouped bar chart
        bars1 = ax.bar([i - width/2 for i in x], predicted, width, 
                       label='Predicted', color='#2196F3', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], actual, width,
                       label='Actual Impact', color='#FF5722', alpha=0.8)
        
        # Customize
        ax.set_xlabel('Component')
        ax.set_ylabel('Score / Impact')
        ax.set_title(
            f'Predicted Criticality vs Actual Impact (Top {len(components)})',
            fontsize=self.config.title_fontsize,
            fontweight='bold'
        )
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add correlation annotation
        spearman = self._validation_result.spearman_correlation
        ax.annotate(
            f'Spearman ρ = {spearman:.3f}',
            xy=(0.98, 0.98), xycoords='axes fraction',
            ha='right', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close(fig)
            return output_path
        else:
            plt.close(fig)
            return None
    
    # =========================================================================
    # Validation Scatter Plot
    # =========================================================================
    
    def plot_validation_scatter(self,
                                output_path: Optional[str] = None) -> Optional[str]:
        """
        Plot scatter plot of predicted vs actual with regression line.
        
        Args:
            output_path: Path to save image
        
        Returns:
            Path to saved image or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not self._validation_result:
            self.logger.error("No validation result available")
            return None
        
        cv_list = self._validation_result.component_validations
        if not cv_list:
            return None
        
        # Prepare data
        predicted = [cv.predicted_score for cv in cv_list]
        actual = [cv.actual_impact for cv in cv_list]
        correct = [cv.correctly_classified for cv in cv_list]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot with different colors for correct/incorrect
        colors = ['#4CAF50' if c else '#F44336' for c in correct]
        ax.scatter(predicted, actual, c=colors, s=100, alpha=0.7, edgecolors='white')
        
        # Add diagonal line (perfect prediction)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Prediction')
        
        # Add trend line
        if len(predicted) > 1 and NUMPY_AVAILABLE:
            z = np.polyfit(predicted, actual, 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, max(predicted) * 1.1, 100)
            ax.plot(x_line, p(x_line), 'b-', alpha=0.5, label='Trend Line')
        
        # Customize
        ax.set_xlabel('Predicted Criticality Score', fontsize=12)
        ax.set_ylabel('Actual Impact Score', fontsize=12)
        ax.set_title(
            'Validation: Predicted vs Actual',
            fontsize=self.config.title_fontsize,
            fontweight='bold'
        )
        
        # Set limits
        max_val = max(max(predicted, default=1), max(actual, default=1)) * 1.1
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        
        # Add metrics annotation
        vr = self._validation_result
        metrics_text = (
            f"Spearman ρ = {vr.spearman_correlation:.3f}\n"
            f"Pearson r = {vr.pearson_correlation:.3f}\n"
            f"F1-Score = {vr.confusion_matrix.f1_score:.3f}"
        )
        ax.annotate(
            metrics_text,
            xy=(0.02, 0.98), xycoords='axes fraction',
            ha='left', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='#4CAF50', label='Correctly Classified'),
            mpatches.Patch(color='#F44336', label='Misclassified'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        ax.set_aspect('equal')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close(fig)
            return output_path
        else:
            plt.close(fig)
            return None
    
    # =========================================================================
    # Simulation Impact Distribution
    # =========================================================================
    
    def plot_impact_distribution(self,
                                 output_path: Optional[str] = None) -> Optional[str]:
        """
        Plot distribution of impact scores from simulation.
        
        Args:
            output_path: Path to save image
        
        Returns:
            Path to saved image or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not self._simulation_result:
            self.logger.error("No simulation result available")
            return None
        
        # Get impact scores
        impacts = [r.impact_score for r in self._simulation_result.results
                   if len(r.failed_components) == 1]
        
        if not impacts:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        n, bins, patches = ax.hist(impacts, bins=20, edgecolor='white', alpha=0.7)
        
        # Color bars by severity
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge >= 0.7:
                patch.set_facecolor(ColorScheme.CRITICALITY_COLORS['critical'])
            elif left_edge >= 0.5:
                patch.set_facecolor(ColorScheme.CRITICALITY_COLORS['high'])
            elif left_edge >= 0.3:
                patch.set_facecolor(ColorScheme.CRITICALITY_COLORS['medium'])
            else:
                patch.set_facecolor(ColorScheme.CRITICALITY_COLORS['low'])
        
        # Add statistics
        mean_impact = sum(impacts) / len(impacts)
        max_impact = max(impacts)
        
        ax.axvline(mean_impact, color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_impact:.3f}')
        ax.axvline(max_impact, color='red', linestyle='--', linewidth=2,
                   label=f'Max: {max_impact:.3f}')
        
        # Customize
        ax.set_xlabel('Impact Score', fontsize=12)
        ax.set_ylabel('Number of Components', fontsize=12)
        ax.set_title(
            f'Distribution of Failure Impact Scores ({len(impacts)} simulations)',
            fontsize=self.config.title_fontsize,
            fontweight='bold'
        )
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close(fig)
            return output_path
        else:
            plt.close(fig)
            return None
    
    # =========================================================================
    # HTML Report Generation
    # =========================================================================
    
    def generate_html_report(self,
                            output_path: str,
                            title: str = "Pub-Sub System Analysis Report",
                            include_images: bool = True) -> str:
        """
        Generate comprehensive HTML report with all visualizations.
        
        Args:
            output_path: Path to save HTML file
            title: Report title
            include_images: Include embedded images (base64)
        
        Returns:
            Path to saved HTML file
        """
        output_path = Path(output_path)
        
        # Generate images if matplotlib available
        images = {}
        if include_images and MATPLOTLIB_AVAILABLE:
            import tempfile
            import base64
            
            temp_dir = Path(tempfile.mkdtemp())
            
            # Generate each visualization
            if self.analyzer and self.analyzer.G:
                img_path = temp_dir / 'topology.png'
                if self.plot_topology(str(img_path)):
                    images['topology'] = self._image_to_base64(img_path)
            
            if self._analysis_result:
                img_path = temp_dir / 'criticality.png'
                if self.plot_criticality_heatmap(str(img_path)):
                    images['criticality'] = self._image_to_base64(img_path)
            
            if self._validation_result:
                img_path = temp_dir / 'comparison.png'
                if self.plot_impact_comparison(str(img_path)):
                    images['comparison'] = self._image_to_base64(img_path)
                
                img_path = temp_dir / 'scatter.png'
                if self.plot_validation_scatter(str(img_path)):
                    images['scatter'] = self._image_to_base64(img_path)
            
            if self._simulation_result:
                img_path = temp_dir / 'distribution.png'
                if self.plot_impact_distribution(str(img_path)):
                    images['distribution'] = self._image_to_base64(img_path)
        
        # Build HTML
        html = self._build_html_report(title, images)
        
        # Save
        with open(output_path, 'w') as f:
            f.write(html)
        
        return str(output_path)
    
    def _image_to_base64(self, path: Path) -> str:
        """Convert image file to base64 string"""
        import base64
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _build_html_report(self, title: str, images: Dict[str, str]) -> str:
        """Build HTML report content"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Start HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <style>
        :root {{
            --primary: #2196F3;
            --success: #4CAF50;
            --warning: #FFC107;
            --danger: #F44336;
            --text: #333;
            --bg: #f5f5f5;
            --card-bg: white;
        }}
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1, h2, h3 {{ color: var(--text); margin-top: 0; }}
        .header {{
            background: linear-gradient(135deg, var(--primary), #1976D2);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 2em; }}
        .header p {{ margin: 0; opacity: 0.9; }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{ 
            margin: 0 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--bg);
        }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; }}
        .metric {{
            text-align: center;
            padding: 20px;
            background: var(--bg);
            border-radius: 8px;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .success {{ color: var(--success); }}
        .warning {{ color: var(--warning); }}
        .danger {{ color: var(--danger); }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: var(--bg); font-weight: 600; }}
        tr:hover {{ background: #f9f9f9; }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge-success {{ background: #e8f5e9; color: #2e7d32; }}
        .badge-warning {{ background: #fff8e1; color: #f57f17; }}
        .badge-danger {{ background: #ffebee; color: #c62828; }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>{title}</h1>
        <p>Generated: {timestamp}</p>
    </div>
"""
        
        # Summary section
        html += self._build_summary_section()
        
        # Topology visualization
        if 'topology' in images:
            html += f"""
    <div class="card">
        <h2>📊 System Topology</h2>
        <div class="image-container">
            <img src="data:image/png;base64,{images['topology']}" alt="System Topology">
        </div>
    </div>
"""
        
        # Analysis section
        if self._analysis_result:
            html += self._build_analysis_section(images.get('criticality'))
        
        # Simulation section
        if self._simulation_result:
            html += self._build_simulation_section(images.get('distribution'))
        
        # Validation section
        if self._validation_result:
            html += self._build_validation_section(
                images.get('comparison'),
                images.get('scatter')
            )
        
        # Close HTML
        html += """
</div>
</body>
</html>
"""
        
        return html
    
    def _build_summary_section(self) -> str:
        """Build summary metrics section"""
        html = '<div class="card"><h2>📈 Summary</h2><div class="grid">'
        
        # Graph metrics
        if self.analyzer and self.analyzer.G:
            G = self.analyzer.G
            html += f"""
        <div class="metric">
            <div class="metric-value">{G.number_of_nodes()}</div>
            <div class="metric-label">Components</div>
        </div>
        <div class="metric">
            <div class="metric-value">{G.number_of_edges()}</div>
            <div class="metric-label">Dependencies</div>
        </div>
"""
        
        # Validation metrics
        if self._validation_result:
            vr = self._validation_result
            status_class = {
                'passed': 'success',
                'marginal': 'warning',
                'failed': 'danger'
            }.get(vr.status.value, '')
            
            html += f"""
        <div class="metric">
            <div class="metric-value {status_class}">{vr.spearman_correlation:.3f}</div>
            <div class="metric-label">Spearman Correlation</div>
        </div>
        <div class="metric">
            <div class="metric-value">{vr.confusion_matrix.f1_score:.3f}</div>
            <div class="metric-label">F1-Score</div>
        </div>
"""
        
        html += '</div></div>'
        return html
    
    def _build_analysis_section(self, criticality_img: Optional[str]) -> str:
        """Build analysis results section"""
        ar = self._analysis_result
        
        html = '<div class="card"><h2>🔍 Criticality Analysis</h2>'
        
        # Criticality chart
        if criticality_img:
            html += f"""
        <div class="image-container">
            <img src="data:image/png;base64,{criticality_img}" alt="Criticality Ranking">
        </div>
"""
        
        # Top critical components table
        top_critical = sorted(
            ar.criticality_scores,
            key=lambda x: x.composite_score,
            reverse=True
        )[:10]
        
        html += """
        <h3>Top Critical Components</h3>
        <table>
            <tr>
                <th>Component</th>
                <th>Type</th>
                <th>Score</th>
                <th>Level</th>
                <th>Articulation Point</th>
            </tr>
"""
        
        for score in top_critical:
            level_class = {
                'critical': 'badge-danger',
                'high': 'badge-warning',
                'medium': 'badge-warning',
                'low': 'badge-success'
            }.get(score.level.value, '')
            
            ap = '✓' if score.is_articulation_point else ''
            
            html += f"""
            <tr>
                <td><strong>{score.node_id}</strong></td>
                <td>{score.node_type}</td>
                <td>{score.composite_score:.4f}</td>
                <td><span class="badge {level_class}">{score.level.value}</span></td>
                <td>{ap}</td>
            </tr>
"""
        
        html += '</table></div>'
        return html
    
    def _build_simulation_section(self, distribution_img: Optional[str]) -> str:
        """Build simulation results section"""
        sr = self._simulation_result
        
        html = '<div class="card"><h2>⚡ Failure Simulation Results</h2>'
        
        # Distribution chart
        if distribution_img:
            html += f"""
        <div class="image-container">
            <img src="data:image/png;base64,{distribution_img}" alt="Impact Distribution">
        </div>
"""
        
        # Summary stats
        summary = sr.summary
        html += f"""
        <div class="grid">
            <div class="metric">
                <div class="metric-value">{sr.total_simulations}</div>
                <div class="metric-label">Simulations Run</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['impact_score']['mean']:.1%}</div>
                <div class="metric-label">Mean Impact</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['impact_score']['max']:.1%}</div>
                <div class="metric-label">Max Impact</div>
            </div>
        </div>
"""
        
        # Top impactful components
        ranking = sr.get_impact_ranking()[:10]
        
        html += """
        <h3>Highest Impact Components</h3>
        <table>
            <tr><th>Rank</th><th>Component</th><th>Impact Score</th><th>Severity</th></tr>
"""
        
        for i, (comp, impact) in enumerate(ranking, 1):
            if impact >= 0.7:
                level, level_class = 'Critical', 'badge-danger'
            elif impact >= 0.5:
                level, level_class = 'High', 'badge-warning'
            elif impact >= 0.3:
                level, level_class = 'Medium', 'badge-warning'
            else:
                level, level_class = 'Low', 'badge-success'
            
            html += f"""
            <tr>
                <td>{i}</td>
                <td><strong>{comp}</strong></td>
                <td>{impact:.1%}</td>
                <td><span class="badge {level_class}">{level}</span></td>
            </tr>
"""
        
        html += '</table></div>'
        return html
    
    def _build_validation_section(self,
                                  comparison_img: Optional[str],
                                  scatter_img: Optional[str]) -> str:
        """Build validation results section"""
        vr = self._validation_result
        
        status_class = {
            'passed': 'success',
            'marginal': 'warning',
            'failed': 'danger'
        }.get(vr.status.value, '')
        
        html = f"""
    <div class="card">
        <h2>✅ Validation Results</h2>
        <div class="metric" style="display: inline-block; margin-bottom: 20px;">
            <div class="metric-value {status_class}">{vr.status.value.upper()}</div>
            <div class="metric-label">Validation Status</div>
        </div>
"""
        
        # Charts
        if comparison_img or scatter_img:
            html += '<div class="two-col">'
            if comparison_img:
                html += f"""
            <div class="image-container">
                <img src="data:image/png;base64,{comparison_img}" alt="Impact Comparison">
            </div>
"""
            if scatter_img:
                html += f"""
            <div class="image-container">
                <img src="data:image/png;base64,{scatter_img}" alt="Validation Scatter">
            </div>
"""
            html += '</div>'
        
        # Metrics table
        cm = vr.confusion_matrix
        html += f"""
        <h3>Validation Metrics</h3>
        <div class="grid">
            <div>
                <table>
                    <tr><th>Correlation Metric</th><th>Value</th><th>Target</th></tr>
                    <tr>
                        <td>Spearman Correlation</td>
                        <td><strong>{vr.spearman_correlation:.4f}</strong></td>
                        <td>≥ 0.70</td>
                    </tr>
                    <tr>
                        <td>Pearson Correlation</td>
                        <td>{vr.pearson_correlation:.4f}</td>
                        <td>-</td>
                    </tr>
                </table>
            </div>
            <div>
                <table>
                    <tr><th>Classification Metric</th><th>Value</th><th>Target</th></tr>
                    <tr>
                        <td>Precision</td>
                        <td>{cm.precision:.4f}</td>
                        <td>≥ 0.80</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td>{cm.recall:.4f}</td>
                        <td>≥ 0.80</td>
                    </tr>
                    <tr>
                        <td>F1-Score</td>
                        <td><strong>{cm.f1_score:.4f}</strong></td>
                        <td>≥ 0.90</td>
                    </tr>
                </table>
            </div>
        </div>
"""
        
        # Component comparison table
        html += """
        <h3>Component Comparison</h3>
        <table>
            <tr>
                <th>Component</th>
                <th>Predicted</th>
                <th>Actual</th>
                <th>Rank Δ</th>
                <th>Classification</th>
            </tr>
"""
        
        for cv in vr.component_validations[:15]:
            match_class = 'badge-success' if cv.correctly_classified else 'badge-danger'
            match_text = 'Correct' if cv.correctly_classified else 'Misclassified'
            
            html += f"""
            <tr>
                <td><strong>{cv.component_id}</strong></td>
                <td>{cv.predicted_score:.4f}</td>
                <td>{cv.actual_impact:.2%}</td>
                <td>{cv.rank_difference}</td>
                <td><span class="badge {match_class}">{match_text}</span></td>
            </tr>
"""
        
        html += '</table></div>'
        return html
    
    # =========================================================================
    # Export Functions
    # =========================================================================
    
    def export_graph_data(self, output_path: str, format: str = 'json') -> str:
        """
        Export graph data for external visualization tools.
        
        Args:
            output_path: Path to save file
            format: Export format ('json', 'graphml', 'gexf')
        
        Returns:
            Path to saved file
        """
        if self.analyzer is None or self.analyzer.G is None:
            raise ValueError("No graph data available")
        
        G = self.analyzer.G
        output_path = Path(output_path)
        
        if format == 'json':
            # Export as D3.js compatible JSON
            data = {
                'nodes': [],
                'links': []
            }
            
            # Add criticality scores if available
            crit_lookup = {}
            if self._analysis_result:
                for score in self._analysis_result.criticality_scores:
                    crit_lookup[score.node_id] = {
                        'score': score.composite_score,
                        'level': score.level.value
                    }
            
            for node, attrs in G.nodes(data=True):
                node_data = {
                    'id': node,
                    'type': attrs.get('type', 'Unknown'),
                    'name': attrs.get('name', node),
                }
                if node in crit_lookup:
                    node_data['criticality'] = crit_lookup[node]
                data['nodes'].append(node_data)
            
            for source, target, attrs in G.edges(data=True):
                data['links'].append({
                    'source': source,
                    'target': target,
                    'type': attrs.get('dependency_type', 'unknown'),
                    'weight': attrs.get('weight', 1.0)
                })
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'graphml':
            nx.write_graphml(G, output_path)
        
        elif format == 'gexf':
            nx.write_gexf(G, output_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(output_path)


# ============================================================================
# Convenience Functions
# ============================================================================

def visualize_system(analyzer: 'GraphAnalyzer',
                    output_dir: str,
                    simulation_result: Optional['BatchSimulationResult'] = None,
                    validation_result: Optional['ValidationResult'] = None) -> List[str]:
    """
    Generate all visualizations for a system.
    
    Args:
        analyzer: GraphAnalyzer with analysis complete
        output_dir: Directory to save visualizations
        simulation_result: Optional simulation results
        validation_result: Optional validation results
    
    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = GraphVisualizer(analyzer)
    
    # Get analysis result - run analysis if needed
    if analyzer.G is None:
        analyzer.analyze()
    
    # Try to get existing criticality scores
    if analyzer.depends_on_edges and hasattr(analyzer, 'G'):
        # Re-run analysis to get result object
        from src.analysis import AnalysisResult
        analysis_result = analyzer.analyze()
        visualizer.set_analysis_result(analysis_result)
    
    if simulation_result:
        visualizer.set_simulation_result(simulation_result)
    
    if validation_result:
        visualizer.set_validation_result(validation_result)
    
    files = []
    
    # Generate visualizations
    if MATPLOTLIB_AVAILABLE:
        path = visualizer.plot_topology(str(output_dir / 'topology.png'))
        if path:
            files.append(path)
        
        path = visualizer.plot_criticality_heatmap(str(output_dir / 'criticality.png'))
        if path:
            files.append(path)
        
        if validation_result:
            path = visualizer.plot_impact_comparison(str(output_dir / 'comparison.png'))
            if path:
                files.append(path)
            
            path = visualizer.plot_validation_scatter(str(output_dir / 'scatter.png'))
            if path:
                files.append(path)
        
        if simulation_result:
            path = visualizer.plot_impact_distribution(str(output_dir / 'distribution.png'))
            if path:
                files.append(path)
    
    # Generate HTML report
    html_path = visualizer.generate_html_report(str(output_dir / 'report.html'))
    files.append(html_path)
    
    return files


def quick_visualize(filepath: str, output_dir: str) -> List[str]:
    """
    Quick visualization from a JSON file.
    
    Args:
        filepath: Path to input JSON file
        output_dir: Directory to save visualizations
    
    Returns:
        List of generated file paths
    """
    from src.analysis import GraphAnalyzer
    
    analyzer = GraphAnalyzer()
    analyzer.load_from_file(filepath)
    result = analyzer.analyze()
    
    visualizer = GraphVisualizer(analyzer)
    visualizer.set_analysis_result(result)
    
    return visualize_system(analyzer, output_dir)