"""
Visualization Charts - Version 5.0

Chart generation for graph analysis and simulation results.

Uses matplotlib for static charts and generates embeddable HTML/SVG.

Chart Types:
- Bar charts for component/edge distributions
- Pie charts for type breakdowns
- Heatmaps for layer comparisons
- Line charts for impact rankings
- Scatter plots for correlation analysis

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import io
import base64
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

# Check for matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# =============================================================================
# Configuration
# =============================================================================

class ChartTheme(Enum):
    """Chart color themes."""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"


@dataclass
class ChartConfig:
    """Chart configuration."""
    width: float = 8.0
    height: float = 5.0
    dpi: int = 100
    theme: ChartTheme = ChartTheme.DEFAULT
    title_size: int = 12
    label_size: int = 10
    
    # Color palettes
    colors: List[str] = field(default_factory=lambda: [
        '#2ecc71',  # Green
        '#3498db',  # Blue
        '#9b59b6',  # Purple
        '#e74c3c',  # Red
        '#f39c12',  # Orange
        '#1abc9c',  # Teal
        '#34495e',  # Dark gray
        '#e67e22',  # Dark orange
    ])
    
    # Criticality colors
    critical_color: str = '#e74c3c'
    high_color: str = '#f39c12'
    medium_color: str = '#3498db'
    low_color: str = '#2ecc71'
    
    # Status colors
    passed_color: str = '#2ecc71'
    partial_color: str = '#f39c12'
    failed_color: str = '#e74c3c'


DEFAULT_CONFIG = ChartConfig()


# =============================================================================
# Chart Output
# =============================================================================

@dataclass
class ChartOutput:
    """Output from chart generation."""
    title: str
    svg: str  # SVG string
    png_base64: str  # Base64 encoded PNG
    width: int
    height: int
    
    def to_html_img(self) -> str:
        """Get HTML img tag with embedded PNG."""
        return f'<img src="data:image/png;base64,{self.png_base64}" alt="{self.title}" />'
    
    def to_html_svg(self) -> str:
        """Get embedded SVG."""
        return self.svg


def _fig_to_output(fig, title: str, config: ChartConfig) -> ChartOutput:
    """Convert matplotlib figure to ChartOutput."""
    # Generate PNG
    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format='png', dpi=config.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    png_buffer.seek(0)
    png_base64 = base64.b64encode(png_buffer.read()).decode('utf-8')
    
    # Generate SVG
    svg_buffer = io.BytesIO()
    fig.savefig(svg_buffer, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    svg_buffer.seek(0)
    svg = svg_buffer.read().decode('utf-8')
    
    plt.close(fig)
    
    return ChartOutput(
        title=title,
        svg=svg,
        png_base64=png_base64,
        width=int(config.width * config.dpi),
        height=int(config.height * config.dpi),
    )


# =============================================================================
# Graph Statistics Charts
# =============================================================================

def chart_component_distribution(
    components: Dict[str, int],
    title: str = "Component Distribution",
    config: Optional[ChartConfig] = None,
) -> ChartOutput:
    """
    Bar chart of component counts by type.
    
    Args:
        components: Dict mapping type name to count
        title: Chart title
        config: Chart configuration
    
    Returns:
        ChartOutput with the generated chart
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for charts")
    
    config = config or DEFAULT_CONFIG
    
    fig, ax = plt.subplots(figsize=(config.width, config.height))
    
    types = list(components.keys())
    counts = list(components.values())
    colors = config.colors[:len(types)]
    
    bars = ax.bar(types, counts, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=config.label_size)
    
    ax.set_xlabel('Component Type', fontsize=config.label_size)
    ax.set_ylabel('Count', fontsize=config.label_size)
    ax.set_title(title, fontsize=config.title_size, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_output(fig, title, config)


def chart_edge_distribution(
    edges: Dict[str, int],
    title: str = "Edge Distribution",
    config: Optional[ChartConfig] = None,
) -> ChartOutput:
    """
    Horizontal bar chart of edge counts by type.
    
    Args:
        edges: Dict mapping edge type to count
        title: Chart title
        config: Chart configuration
    
    Returns:
        ChartOutput with the generated chart
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for charts")
    
    config = config or DEFAULT_CONFIG
    
    fig, ax = plt.subplots(figsize=(config.width, config.height))
    
    types = list(edges.keys())
    counts = list(edges.values())
    colors = config.colors[:len(types)]
    
    y_pos = range(len(types))
    bars = ax.barh(y_pos, counts, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(count), ha='left', va='center', fontsize=config.label_size)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types)
    ax.set_xlabel('Count', fontsize=config.label_size)
    ax.set_title(title, fontsize=config.title_size, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_output(fig, title, config)


def chart_layer_summary(
    layers: Dict[str, Dict[str, int]],
    title: str = "Layer Summary",
    config: Optional[ChartConfig] = None,
) -> ChartOutput:
    """
    Grouped bar chart comparing layers.
    
    Args:
        layers: Dict mapping layer name to {components, edges}
        title: Chart title
        config: Chart configuration
    
    Returns:
        ChartOutput with the generated chart
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for charts")
    
    config = config or DEFAULT_CONFIG
    
    fig, ax = plt.subplots(figsize=(config.width, config.height))
    
    layer_names = list(layers.keys())
    components = [layers[l].get('components', 0) for l in layer_names]
    edges = [layers[l].get('edges', 0) for l in layer_names]
    
    x = range(len(layer_names))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], components, width, 
                   label='Components', color=config.colors[0])
    bars2 = ax.bar([i + width/2 for i in x], edges, width,
                   label='Edges', color=config.colors[1])
    
    ax.set_xlabel('Layer', fontsize=config.label_size)
    ax.set_ylabel('Count', fontsize=config.label_size)
    ax.set_title(title, fontsize=config.title_size, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=15, ha='right')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_output(fig, title, config)


# =============================================================================
# Analysis Charts
# =============================================================================

def chart_impact_ranking(
    impacts: List[Tuple[str, float]],
    top_n: int = 15,
    title: str = "Impact Ranking",
    config: Optional[ChartConfig] = None,
) -> ChartOutput:
    """
    Horizontal bar chart of top components by impact.
    
    Args:
        impacts: List of (component_id, impact_score) tuples
        top_n: Number of top components to show
        title: Chart title
        config: Chart configuration
    
    Returns:
        ChartOutput with the generated chart
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for charts")
    
    config = config or DEFAULT_CONFIG
    
    # Take top N and reverse for display
    top_impacts = impacts[:top_n][::-1]
    
    fig, ax = plt.subplots(figsize=(config.width, config.height))
    
    names = [t[0] for t in top_impacts]
    scores = [t[1] for t in top_impacts]
    
    # Color by impact level
    colors = []
    for score in scores:
        if score >= 0.5:
            colors.append(config.critical_color)
        elif score >= 0.3:
            colors.append(config.high_color)
        elif score >= 0.1:
            colors.append(config.medium_color)
        else:
            colors.append(config.low_color)
    
    y_pos = range(len(names))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontsize=config.label_size - 1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=config.label_size - 1)
    ax.set_xlabel('Impact Score', fontsize=config.label_size)
    ax.set_title(title, fontsize=config.title_size, fontweight='bold')
    ax.set_xlim(0, max(scores) * 1.15 if scores else 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=config.critical_color, label='Critical (≥0.5)'),
        mpatches.Patch(color=config.high_color, label='High (≥0.3)'),
        mpatches.Patch(color=config.medium_color, label='Medium (≥0.1)'),
        mpatches.Patch(color=config.low_color, label='Low (<0.1)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=config.label_size - 2)
    
    plt.tight_layout()
    return _fig_to_output(fig, title, config)


def chart_criticality_distribution(
    levels: Dict[str, int],
    title: str = "Criticality Distribution",
    config: Optional[ChartConfig] = None,
) -> ChartOutput:
    """
    Pie chart of criticality level distribution.
    
    Args:
        levels: Dict mapping level name to count
        title: Chart title
        config: Chart configuration
    
    Returns:
        ChartOutput with the generated chart
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for charts")
    
    config = config or DEFAULT_CONFIG
    
    fig, ax = plt.subplots(figsize=(config.width, config.height))
    
    # Filter out zero values
    filtered = {k: v for k, v in levels.items() if v > 0}
    
    if not filtered:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        ax.set_title(title, fontsize=config.title_size, fontweight='bold')
        return _fig_to_output(fig, title, config)
    
    labels = list(filtered.keys())
    sizes = list(filtered.values())
    
    # Map colors
    color_map = {
        'CRITICAL': config.critical_color,
        'HIGH': config.high_color,
        'MEDIUM': config.medium_color,
        'LOW': config.low_color,
    }
    colors = [color_map.get(l.upper(), config.colors[i % len(config.colors)]) 
              for i, l in enumerate(labels)]
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(config.label_size)
    
    ax.set_title(title, fontsize=config.title_size, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_output(fig, title, config)


# =============================================================================
# Validation Charts
# =============================================================================

def chart_correlation_comparison(
    metrics: Dict[str, float],
    targets: Optional[Dict[str, float]] = None,
    title: str = "Correlation Metrics",
    config: Optional[ChartConfig] = None,
) -> ChartOutput:
    """
    Bar chart comparing correlation metrics against targets.
    
    Args:
        metrics: Dict mapping metric name to value
        targets: Dict mapping metric name to target value
        title: Chart title
        config: Chart configuration
    
    Returns:
        ChartOutput with the generated chart
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for charts")
    
    config = config or DEFAULT_CONFIG
    targets = targets or {}
    
    fig, ax = plt.subplots(figsize=(config.width, config.height))
    
    names = list(metrics.keys())
    values = list(metrics.values())
    
    # Color based on meeting target
    colors = []
    for name, value in zip(names, values):
        target = targets.get(name, 0)
        if value >= target:
            colors.append(config.passed_color)
        elif value >= target * 0.8:
            colors.append(config.partial_color)
        else:
            colors.append(config.failed_color)
    
    x = range(len(names))
    bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add target lines
    for i, name in enumerate(names):
        if name in targets:
            ax.hlines(targets[name], i - 0.4, i + 0.4, 
                     colors='black', linestyles='dashed', linewidth=2)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=config.label_size)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Value', fontsize=config.label_size)
    ax.set_title(title, fontsize=config.title_size, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=config.passed_color, label='Meets Target'),
        mpatches.Patch(color=config.partial_color, label='Near Target'),
        mpatches.Patch(color=config.failed_color, label='Below Target'),
        plt.Line2D([0], [0], color='black', linestyle='dashed', label='Target'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=config.label_size - 2)
    
    plt.tight_layout()
    return _fig_to_output(fig, title, config)


def chart_confusion_matrix(
    tp: int, fp: int, fn: int, tn: int,
    title: str = "Confusion Matrix",
    config: Optional[ChartConfig] = None,
) -> ChartOutput:
    """
    Heatmap visualization of confusion matrix.
    
    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        tn: True negatives
        title: Chart title
        config: Chart configuration
    
    Returns:
        ChartOutput with the generated chart
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for charts")
    
    config = config or DEFAULT_CONFIG
    
    fig, ax = plt.subplots(figsize=(config.width * 0.8, config.height * 0.8))
    
    matrix = [[tp, fp], [fn, tn]]
    
    im = ax.imshow(matrix, cmap='Blues')
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Positive', 'Negative'])
    ax.set_yticklabels(['Positive', 'Negative'])
    ax.set_xlabel('Predicted', fontsize=config.label_size)
    ax.set_ylabel('Actual', fontsize=config.label_size)
    
    # Add text annotations
    labels = [['TP', 'FP'], ['FN', 'TN']]
    for i in range(2):
        for j in range(2):
            value = matrix[i][j]
            color = 'white' if value > max(max(matrix)) / 2 else 'black'
            ax.text(j, i, f'{labels[i][j]}\n{value}', 
                   ha='center', va='center', color=color,
                   fontsize=config.label_size + 2, fontweight='bold')
    
    ax.set_title(title, fontsize=config.title_size, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_output(fig, title, config)


def chart_layer_validation(
    layers: Dict[str, Dict[str, float]],
    title: str = "Validation by Layer",
    config: Optional[ChartConfig] = None,
) -> ChartOutput:
    """
    Grouped bar chart of validation metrics by layer.
    
    Args:
        layers: Dict mapping layer to {spearman, f1_score}
        title: Chart title
        config: Chart configuration
    
    Returns:
        ChartOutput with the generated chart
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for charts")
    
    config = config or DEFAULT_CONFIG
    
    fig, ax = plt.subplots(figsize=(config.width, config.height))
    
    layer_names = list(layers.keys())
    spearman = [layers[l].get('spearman', 0) for l in layer_names]
    f1 = [layers[l].get('f1_score', 0) for l in layer_names]
    
    x = range(len(layer_names))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], spearman, width,
                   label='Spearman ρ', color=config.colors[0])
    bars2 = ax.bar([i + width/2 for i in x], f1, width,
                   label='F1-Score', color=config.colors[1])
    
    # Target lines
    ax.axhline(0.7, color='black', linestyle='dashed', alpha=0.5, label='ρ Target (0.7)')
    ax.axhline(0.9, color='gray', linestyle='dotted', alpha=0.5, label='F1 Target (0.9)')
    
    ax.set_xlabel('Layer', fontsize=config.label_size)
    ax.set_ylabel('Score', fontsize=config.label_size)
    ax.set_title(title, fontsize=config.title_size, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=config.label_size - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_output(fig, title, config)


def chart_method_comparison(
    methods: Dict[str, Dict[str, float]],
    title: str = "Method Comparison",
    config: Optional[ChartConfig] = None,
) -> ChartOutput:
    """
    Grouped bar chart comparing analysis methods.
    
    Args:
        methods: Dict mapping method to {spearman, f1_score}
        title: Chart title
        config: Chart configuration
    
    Returns:
        ChartOutput with the generated chart
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for charts")
    
    config = config or DEFAULT_CONFIG
    
    fig, ax = plt.subplots(figsize=(config.width, config.height))
    
    method_names = list(methods.keys())
    spearman = [methods[m].get('spearman', 0) for m in method_names]
    f1 = [methods[m].get('f1_score', 0) for m in method_names]
    
    x = range(len(method_names))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], spearman, width,
                   label='Spearman ρ', color=config.colors[0])
    bars2 = ax.bar([i + width/2 for i in x], f1, width,
                   label='F1-Score', color=config.colors[1])
    
    # Add value labels
    for bar, val in zip(bars1, spearman):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=config.label_size - 2)
    for bar, val in zip(bars2, f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=config.label_size - 2)
    
    ax.set_xlabel('Method', fontsize=config.label_size)
    ax.set_ylabel('Score', fontsize=config.label_size)
    ax.set_title(title, fontsize=config.title_size, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_output(fig, title, config)


# =============================================================================
# Simulation Charts
# =============================================================================

def chart_delivery_stats(
    stats: Dict[str, int],
    title: str = "Message Delivery Statistics",
    config: Optional[ChartConfig] = None,
) -> ChartOutput:
    """
    Pie chart of message delivery statistics.
    
    Args:
        stats: Dict with delivered, failed, timeout counts
        title: Chart title
        config: Chart configuration
    
    Returns:
        ChartOutput with the generated chart
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for charts")
    
    config = config or DEFAULT_CONFIG
    
    fig, ax = plt.subplots(figsize=(config.width, config.height))
    
    # Filter out zero values
    filtered = {k: v for k, v in stats.items() if v > 0}
    
    if not filtered:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        ax.set_title(title, fontsize=config.title_size, fontweight='bold')
        return _fig_to_output(fig, title, config)
    
    labels = list(filtered.keys())
    sizes = list(filtered.values())
    
    # Color mapping
    color_map = {
        'delivered': config.passed_color,
        'failed': config.failed_color,
        'timeout': config.partial_color,
    }
    colors = [color_map.get(l.lower(), config.colors[i % len(config.colors)])
              for i, l in enumerate(labels)]
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    ax.set_title(title, fontsize=config.title_size, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_output(fig, title, config)


def chart_layer_performance(
    layers: Dict[str, Dict[str, Any]],
    title: str = "Layer Performance",
    config: Optional[ChartConfig] = None,
) -> ChartOutput:
    """
    Bar chart of delivery rates by layer.
    
    Args:
        layers: Dict mapping layer to {delivery_rate, avg_latency}
        title: Chart title
        config: Chart configuration
    
    Returns:
        ChartOutput with the generated chart
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for charts")
    
    config = config or DEFAULT_CONFIG
    
    fig, ax1 = plt.subplots(figsize=(config.width, config.height))
    
    layer_names = list(layers.keys())
    delivery_rates = [layers[l].get('delivery_rate', 0) * 100 for l in layer_names]
    
    x = range(len(layer_names))
    
    # Delivery rate bars
    colors = [config.passed_color if r >= 90 else config.partial_color if r >= 70 
              else config.failed_color for r in delivery_rates]
    bars = ax1.bar(x, delivery_rates, color=colors, edgecolor='white', linewidth=1.5)
    
    ax1.set_xlabel('Layer', fontsize=config.label_size)
    ax1.set_ylabel('Delivery Rate (%)', fontsize=config.label_size)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, rotation=15, ha='right')
    ax1.set_ylim(0, 105)
    ax1.axhline(90, color='black', linestyle='dashed', alpha=0.5)
    
    # Add value labels
    for bar, rate in zip(bars, delivery_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=config.label_size - 1)
    
    ax1.set_title(title, fontsize=config.title_size, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_output(fig, title, config)


# =============================================================================
# Utility Functions
# =============================================================================

def check_matplotlib_available() -> bool:
    """Check if matplotlib is available."""
    return HAS_MATPLOTLIB
