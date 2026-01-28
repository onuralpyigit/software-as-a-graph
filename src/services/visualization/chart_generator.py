"""
Chart Generator Service
"""
import base64
import io
import logging
from typing import Dict, List, Any, Optional, Tuple

from ...models.visualization.chart_data import ChartOutput, ColorTheme, DEFAULT_THEME

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
    mpatches = None


class ChartGenerator:
    """
    Generates matplotlib charts as base64-encoded PNG images.
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
    
    def impact_ranking(
        self,
        components: List[Tuple[str, float, str]],
        title: str = "Top Components by Impact",
        max_items: int = 10,
        names: Dict[str, str] = None,
        description: str = ""
    ) -> Optional[ChartOutput]:
        """Create a horizontal bar chart for component impact ranking."""
        if not HAS_MATPLOTLIB or not components:
            return None
        
        # Sort and limit
        components = sorted(components, key=lambda x: x[1], reverse=True)[:max_items]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ids = [c[0] for c in components]
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
        if mpatches:
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
        layer_colors = self.theme.to_layer_dict()
        colors = [layer_colors.get(l, self.theme.primary) for l in data.keys()]
        
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
        """Create a horizontal bar chart for validation metrics with targets."""
        if not HAS_MATPLOTLIB or not metrics:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        names = list(metrics.keys())
        values = [m[0] for m in metrics.values()]
        targets = [m[1] for m in metrics.values()]
        passed = [m[2] for m in metrics.values()]
        
        y_pos = range(len(names))
        colors = [self.theme.success if p else self.theme.danger for p in passed]
        
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
