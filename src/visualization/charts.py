"""
Chart Generator

Creates matplotlib-based charts for dashboard visualization.
Generates PNG images encoded as base64 for embedding in HTML.

Chart Types:
    - Bar charts (metrics, rankings)
    - Pie charts (distributions)
    - Scatter plots (correlation)
    - Heatmaps (confusion matrices)

Color Configuration:
    Colors can be customized via the ColorTheme dataclass.
    Use predefined themes or create custom themes for accessibility/branding.
"""

from __future__ import annotations
import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

# Check for matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


@dataclass
class ChartOutput:
    """
    Output from chart generation.
    
    Attributes:
        title: Chart title
        png_base64: Base64-encoded PNG image data
        description: Human-readable description of what the chart shows
        alt_text: Accessibility text for screen readers (should describe chart content)
        width: Display width in pixels
        height: Display height in pixels
    """
    title: str
    png_base64: str
    description: str = ""
    alt_text: str = ""  # Added for accessibility
    width: int = 600
    height: int = 400


@dataclass
class ColorTheme:
    """
    Configurable color theme for charts.
    
    Provides semantic color names that can be customized for:
        - Accessibility (high contrast themes)
        - Branding (corporate color schemes)
        - Printing (grayscale-friendly colors)
    
    Example:
        >>> theme = ColorTheme(primary="#1a73e8", success="#34a853")
        >>> charts = ChartGenerator(color_theme=theme)
    """
    # Primary semantic colors
    primary: str = "#3498db"
    secondary: str = "#2c3e50"
    success: str = "#2ecc71"
    warning: str = "#f39c12"
    danger: str = "#e74c3c"
    info: str = "#17a2b8"
    light: str = "#ecf0f1"
    dark: str = "#34495e"
    
    # Criticality level colors (for quality classifications)
    critical: str = "#e74c3c"
    high: str = "#e67e22"
    medium: str = "#f1c40f"
    low: str = "#2ecc71"
    minimal: str = "#95a5a6"
    
    # Layer-specific colors
    layer_app: str = "#3498db"
    layer_infra: str = "#9b59b6"
    layer_mw_app: str = "#1abc9c"
    layer_mw_infra: str = "#e67e22"
    layer_system: str = "#2c3e50"
    
    def to_colors_dict(self) -> Dict[str, str]:
        """Convert to COLORS dictionary format for backwards compatibility."""
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "success": self.success,
            "warning": self.warning,
            "danger": self.danger,
            "info": self.info,
            "light": self.light,
            "dark": self.dark,
        }
    
    def to_criticality_dict(self) -> Dict[str, str]:
        """Convert to CRITICALITY_COLORS dictionary format."""
        return {
            "CRITICAL": self.critical,
            "HIGH": self.high,
            "MEDIUM": self.medium,
            "LOW": self.low,
            "MINIMAL": self.minimal,
        }
    
    def to_layer_dict(self) -> Dict[str, str]:
        """Convert to LAYER_COLORS dictionary format."""
        return {
            "app": self.layer_app,
            "infra": self.layer_infra,
            "mw-app": self.layer_mw_app,
            "mw-infra": self.layer_mw_infra,
            "system": self.layer_system,
        }


# Predefined themes
DEFAULT_THEME = ColorTheme()

HIGH_CONTRAST_THEME = ColorTheme(
    primary="#0066cc",
    secondary="#000000",
    success="#008000",
    warning="#ff8c00",
    danger="#cc0000",
    critical="#cc0000",
    high="#ff8c00",
    medium="#cccc00",
    low="#008000",
)


# Backwards-compatible color dictionaries (use DEFAULT_THEME)
COLORS = DEFAULT_THEME.to_colors_dict()
CRITICALITY_COLORS = DEFAULT_THEME.to_criticality_dict()
LAYER_COLORS = DEFAULT_THEME.to_layer_dict()



class ChartGenerator:
    """
    Generates matplotlib charts as base64-encoded PNG images.
    
    All charts are rendered to in-memory buffers and returned
    as base64 strings for HTML embedding.
    """
    
    def __init__(
        self, 
        style: str = "seaborn-v0_8-whitegrid",
        color_theme: ColorTheme = None
    ):
        self.logger = logging.getLogger(__name__)
        self.theme = color_theme or DEFAULT_THEME
        
        if not HAS_MATPLOTLIB:
            self.logger.warning("matplotlib not available, charts will be disabled")
            return
        
        # Try to set style, fall back to default
        try:
            plt.style.use(style)
        except OSError:
            try:
                plt.style.use("ggplot")
            except OSError:
                pass  # Use default
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64
    
    # =========================================================================
    # Bar Charts
    # =========================================================================
    
    def bar_chart(
        self,
        data: Dict[str, float],
        title: str,
        xlabel: str = "",
        ylabel: str = "",
        color: str = None,
        horizontal: bool = False,
        description: str = ""
    ) -> Optional[ChartOutput]:
        """Create a bar chart."""
        if not HAS_MATPLOTLIB or not data:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        labels = list(data.keys())
        values = list(data.values())
        bar_color = color or self.theme.primary
        
        if horizontal:
            bars = ax.barh(labels, values, color=bar_color, edgecolor='white')
            ax.set_xlabel(ylabel or "Value")
            ax.set_ylabel(xlabel or "")
        else:
            bars = ax.bar(labels, values, color=bar_color, edgecolor='white')
            ax.set_xlabel(xlabel or "")
            ax.set_ylabel(ylabel or "Value")
            plt.xticks(rotation=45, ha='right')
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if horizontal:
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}' if isinstance(val, float) else str(val),
                       va='center', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}' if isinstance(val, float) else str(val),
                       ha='center', fontsize=9)
        
        plt.tight_layout()
        return ChartOutput(
            title=title,
            png_base64=self._fig_to_base64(fig),
            description=description,
            alt_text=f"Bar chart titled '{title}' showing values for {', '.join(labels[:3])}..."
        )
    
    def grouped_bar_chart(
        self,
        data: Dict[str, Dict[str, float]],
        title: str,
        xlabel: str = "",
        ylabel: str = "",
        description: str = ""
    ) -> Optional[ChartOutput]:
        """Create a grouped bar chart for comparing metrics across groups."""
        if not HAS_MATPLOTLIB or not data:
            return None
        
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        groups = list(data.keys())
        metrics = list(data[groups[0]].keys()) if groups else []
        
        x = np.arange(len(groups))
        width = 0.8 / len(metrics) if metrics else 0.8
        
        # Use layer colors for consistency if metrics match layer names, else cycle
        layer_colors = list(self.theme.to_layer_dict().values())
        colors = layer_colors[:len(metrics)]
        
        for i, metric in enumerate(metrics):
            values = [data[g].get(metric, 0) for g in groups]
            offset = (i - len(metrics)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric, color=colors[i % len(colors)])
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        return ChartOutput(
            title=title,
            png_base64=self._fig_to_base64(fig),
            description=description,
            alt_text=f"Grouped bar chart titled '{title}' comparing {', '.join(metrics)} across groups"
        )
    
    # =========================================================================
    # Pie Charts
    # =========================================================================
    
    def pie_chart(
        self,
        data: Dict[str, int],
        title: str,
        colors: Dict[str, str] = None,
        description: str = ""
    ) -> Optional[ChartOutput]:
        """Create a pie chart."""
        if not HAS_MATPLOTLIB or not data:
            return None
        
        # Filter out zero values
        data = {k: v for k, v in data.items() if v > 0}
        if not data:
            return None
        
        fig, ax = plt.subplots(figsize=(7, 7))
        
        labels = list(data.keys())
        values = list(data.values())
        
        if colors:
            pie_colors = [colors.get(l, self.theme.light) for l in labels]
        else:
            pie_colors = plt.cm.Set3.colors[:len(labels)]
        
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90,
            explode=[0.02] * len(values)
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        return ChartOutput(
            title=title,
            png_base64=self._fig_to_base64(fig),
            description=description,
            alt_text=f"Pie chart titled '{title}' showing distribution of {', '.join(labels)}"
        )
    
    def criticality_distribution(
        self,
        counts: Dict[str, int],
        title: str = "Criticality Distribution",
        description: str = ""
    ) -> Optional[ChartOutput]:
        """Create a pie chart for criticality level distribution."""
        return self.pie_chart(counts, title, self.theme.to_criticality_dict(), description)
    
    # =========================================================================
    # Scatter Plots
    # =========================================================================
    
    def scatter_plot(
        self,
        x_values: List[float],
        y_values: List[float],
        labels: List[str] = None,
        title: str = "",
        xlabel: str = "Predicted",
        ylabel: str = "Actual",
        add_diagonal: bool = True,
        description: str = ""
    ) -> Optional[ChartOutput]:
        """Create a scatter plot for correlation visualization."""
        if not HAS_MATPLOTLIB or not x_values or not y_values:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(x_values, y_values, c=self.theme.primary, alpha=0.7, s=60, edgecolors='white')
        
        if add_diagonal:
            min_val = min(min(x_values), min(y_values))
            max_val = max(max(x_values), max(y_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect correlation')
        
        # Add labels for outliers
        if labels:
            for i, (x, y, label) in enumerate(zip(x_values, y_values, labels)):
                error = abs(x - y)
                if error > 0.15:  # Label significant outliers
                    ax.annotate(label, (x, y), fontsize=8, alpha=0.7)
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        return ChartOutput(
            title=title,
            png_base64=self._fig_to_base64(fig),
            description=description,
            alt_text=f"Scatter plot titled '{title}' showing correlation between {xlabel} and {ylabel}"
        )
    
    # =========================================================================
    # Heatmaps
    # =========================================================================
    
    def confusion_matrix(
        self,
        tp: int, fp: int, fn: int, tn: int,
        title: str = "Confusion Matrix",
        description: str = ""
    ) -> Optional[ChartOutput]:
        """Create a confusion matrix heatmap."""
        if not HAS_MATPLOTLIB:
            return None
        
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        matrix = np.array([[tp, fp], [fn, tn]])
        
        im = ax.imshow(matrix, cmap='Blues')
        
        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Critical', 'Non-Critical'])
        ax.set_yticklabels(['Critical', 'Non-Critical'])
        ax.set_xlabel('Actual', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        
        # Add text annotations
        labels = [['TP', 'FP'], ['FN', 'TN']]
        for i in range(2):
            for j in range(2):
                color = 'white' if matrix[i, j] > matrix.max()/2 else 'black'
                ax.text(j, i, f'{labels[i][j]}\n{matrix[i, j]}',
                       ha='center', va='center', color=color, fontsize=12, fontweight='bold')
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        plt.tight_layout()
        return ChartOutput(
            title=title,
            png_base64=self._fig_to_base64(fig),
            description=description,
            alt_text=f"Confusion matrix heat map with TP={tp}, FP={fp}, FN={fn}, TN={tn}"
        )
    
    # =========================================================================
    # Specialized Charts
    # =========================================================================
    
    def impact_ranking(
        self,
        components: List[Tuple[str, float, str]],
        title: str = "Top Components by Impact",
        max_items: int = 10,
        names: Dict[str, str] = None,
        description: str = ""
    ) -> Optional[ChartOutput]:
        """
        Create a horizontal bar chart for component impact ranking.
        
        Args:
            components: List of (id, impact, level) tuples
            names: Optional map of ID to display name
        """
        if not HAS_MATPLOTLIB or not components:
            return None
        
        # Sort and limit
        components = sorted(components, key=lambda x: x[1], reverse=True)[:max_items]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ids = [c[0] for c in components]
        # Use names if available
        labels = [f"{c[0]} ({names.get(c[0], '')})" if names and c[0] in names else c[0] for c in components]
        
        impacts = [c[1] for c in components]
        levels = [c[2] for c in components]
        
        crit_colors = self.theme.to_criticality_dict()
        colors = [crit_colors.get(l.upper(), self.theme.light) for l in levels]
        
        bars = ax.barh(range(len(ids)), impacts, color=colors, edgecolor='white')
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        
        ax.set_xlabel('Impact Score')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # Add value labels
        for bar, impact in zip(bars, impacts):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{impact:.3f}', va='center', fontsize=9)
        
        # Legend
        legend_handles = [mpatches.Patch(color=c, label=l) 
                         for l, c in crit_colors.items()]
        ax.legend(handles=legend_handles, loc='lower right', fontsize=8)
        
        plt.tight_layout()
        return ChartOutput(
            title=title,
            png_base64=self._fig_to_base64(fig),
            description=description,
            alt_text=f"Ranking chart showing top {len(components)} components by impact"
        )
    
    def layer_comparison(
        self,
        layer_data: Dict[str, Dict[str, float]],
        metric: str,
        title: str = "",
        description: str = ""
    ) -> Optional[ChartOutput]:
        """Create a bar chart comparing a metric across layers."""
        if not HAS_MATPLOTLIB or not layer_data:
            return None
        
        data = {layer: metrics.get(metric, 0) for layer, metrics in layer_data.items()}
        colors = [LAYER_COLORS.get(l, COLORS["primary"]) for l in data.keys()]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        bars = ax.bar(data.keys(), data.values(), color=colors, edgecolor='white')
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title or f'{metric.replace("_", " ").title()} by Layer',
                    fontsize=12, fontweight='bold', pad=10)
        
        for bar, val in zip(bars, data.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}' if isinstance(val, float) else str(val),
                   ha='center', fontsize=9)
        
        plt.tight_layout()
        return ChartOutput(
            title=title,
            png_base64=self._fig_to_base64(fig),
            description=description,
            alt_text=f"Bar chart comparing {metric} across {len(layer_data)} layers"
        )
    
    def validation_summary(
        self,
        metrics: Dict[str, Tuple[float, float, bool]],
        title: str = "Validation Metrics",
        description: str = ""
    ) -> Optional[ChartOutput]:
        """
        Create a horizontal bar chart for validation metrics with targets.
        
        Args:
            metrics: Dict mapping metric name to (value, target, passed)
        """
        if not HAS_MATPLOTLIB or not metrics:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        names = list(metrics.keys())
        values = [m[0] for m in metrics.values()]
        targets = [m[1] for m in metrics.values()]
        passed = [m[2] for m in metrics.values()]
        
        y_pos = range(len(names))
        colors = [COLORS["success"] if p else COLORS["danger"] for p in passed]
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.7, label='Actual')
        
        # Add target markers
        for i, (target, name) in enumerate(zip(targets, names)):
            ax.axvline(x=target, ymin=(i)/len(names), ymax=(i+1)/len(names),
                      color='black', linestyle='--', linewidth=2)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Value')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # Add value labels
        for bar, val, p in zip(bars, values, passed):
            icon = "✓" if p else "✗"
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f} {icon}', va='center', fontsize=9)
        
        plt.tight_layout()
        return ChartOutput(
            title=title,
            png_base64=self._fig_to_base64(fig),
            description=description,
            alt_text=f"Validation summary chart showing {len(metrics)} metrics with pass/fail status"
        )