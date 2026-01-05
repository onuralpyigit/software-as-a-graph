"""
Visualization Charts

Generates publication-ready static charts (Base64 encoded PNGs) for the dashboard.
"""

import io
import base64
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

try:
    import matplotlib
    matplotlib.use('Agg') # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Color Scheme based on Report Section 4.6
COLORS = {
    "CRITICAL": "#e74c3c", # Red
    "HIGH": "#e67e22",     # Orange
    "MEDIUM": "#f1c40f",   # Yellow
    "LOW": "#2ecc71",      # Green
    "MINIMAL": "#95a5a6"   # Gray
}

@dataclass
class ChartOutput:
    title: str
    png_base64: str
    description: str = ""

class ChartGenerator:
    def __init__(self, style: str = 'ggplot'):
        self.logger = logging.getLogger(__name__)
        if HAS_MATPLOTLIB:
            try:
                plt.style.use(style)
            except Exception:
                pass

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    def plot_criticality_distribution(self, counts: Dict[str, int], title: str) -> Optional[ChartOutput]:
        """Bar chart showing distribution of Criticality Levels."""
        if not HAS_MATPLOTLIB or not counts: return None
        
        levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]
        # Ensure order
        values = [counts.get(l, 0) for l in levels]
        colors = [COLORS.get(l, "#333") for l in levels]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(levels, values, color=colors)
        
        ax.set_ylabel('Component Count')
        ax.set_title(title)
        
        # Add counts on top
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
                
        return ChartOutput(title, self._fig_to_base64(fig), "Distribution of components across criticality levels.")

    def plot_validation_scatter(self, predicted: List[float], actual: List[float], labels: List[str], title: str) -> Optional[ChartOutput]:
        """
        Scatter plot: Predicted Score (Cscore) vs Actual Impact.
        Visualizes the validation correlation (Spearman) mentioned in Report Section 4.5.
        """
        if not HAS_MATPLOTLIB or not predicted: return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(predicted, actual, alpha=0.6, c='#3498db', edgecolors='w', s=80)
        
        # Diagonal line for perfect prediction reference
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        ax.set_xlabel('Predicted Criticality (Cscore)')
        ax.set_ylabel('Actual Failure Impact (Simulation)')
        ax.set_title(title)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return ChartOutput(title, self._fig_to_base64(fig), "Correlation between topological prediction and simulation ground truth.")

    def plot_quality_comparison(self, names: List[str], scores: Dict[str, List[float]], title: str) -> Optional[ChartOutput]:
        """Grouped bar chart comparing R, M, A scores."""
        if not HAS_MATPLOTLIB or not names: return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(names))
        width = 0.25
        multiplier = 0
        
        # Colors for R, M, A
        metric_colors = ["#e74c3c", "#3498db", "#2ecc71"] 
        
        for i, (attribute, measurement) in enumerate(scores.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, color=metric_colors[i % 3])
            multiplier += 1

        ax.set_ylabel('Score (0-1)')
        ax.set_title(title)
        ax.set_xticks(x + width, names)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right', ncols=3)
        plt.xticks(rotation=45, ha='right')
        
        return ChartOutput(title, self._fig_to_base64(fig))

    def plot_impact_ranking(self, impact_map: Dict[str, float], title: str, top_n: int = 10) -> Optional[ChartOutput]:
        """Horizontal bar chart for top failing components."""
        if not HAS_MATPLOTLIB or not impact_map: return None
        
        sorted_items = sorted(impact_map.items(), key=lambda x: x[1], reverse=True)[:top_n]
        sorted_items = sorted_items[::-1] # Reverse for barh
        
        keys = [x[0] for x in sorted_items]
        vals = [x[1] for x in sorted_items]
        
        fig, ax = plt.subplots(figsize=(8, max(4, len(keys) * 0.5)))
        bars = ax.barh(keys, vals, color=COLORS["CRITICAL"])
        
        ax.set_xlabel('Impact Score (0-1)')
        ax.set_title(title)
        ax.set_xlim(0, 1.0)
        
        return ChartOutput(title, self._fig_to_base64(fig))