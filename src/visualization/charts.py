"""
Visualization Charts

Generates publication-ready static charts (Base64 encoded PNGs) for the dashboard.
"""

import io
import base64
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg') # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Consistent Color Scheme
COLORS = {
    "CRITICAL": "#e74c3c", # Red
    "HIGH": "#e67e22",     # Orange
    "MEDIUM": "#f1c40f",   # Yellow
    "LOW": "#2ecc71",      # Green
    "MINIMAL": "#95a5a6",  # Gray
    "APP": "#3498db",      # Blue
    "INFRA": "#9b59b6"     # Purple
}

@dataclass
class ChartOutput:
    title: str
    png_base64: str
    description: str = ""

class ChartGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if HAS_MATPLOTLIB:
            plt.style.use('ggplot')

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    def plot_graph_statistics(self, stats: Dict[str, int], title: str) -> Optional[ChartOutput]:
        """Bar chart for node/edge counts."""
        if not HAS_MATPLOTLIB or not stats: return None
        
        labels = list(stats.keys())
        values = list(stats.values())
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(labels, values, color="#34495e")
        
        ax.set_title(title)
        ax.set_ylabel("Count")
        
        # Add labels on top
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
            
        return ChartOutput(title, self._fig_to_base64(fig), "Overview of graph topology size and complexity.")

    def plot_criticality_distribution(self, counts: Dict[str, int], title: str) -> Optional[ChartOutput]:
        """Bar chart showing distribution of Criticality Levels."""
        if not HAS_MATPLOTLIB or not counts: return None
        
        levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]
        values = [counts.get(l, 0) for l in levels]
        colors = [COLORS.get(l, "#333") for l in levels]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(levels, values, color=colors)
        
        ax.set_ylabel('Component Count')
        ax.set_title(title)
        
        return ChartOutput(title, self._fig_to_base64(fig), "Distribution of components across criticality levels.")

    def plot_validation_scatter(self, predicted: List[float], actual: List[float], ids: List[str], title: str) -> Optional[ChartOutput]:
        """
        Scatter plot: Predicted Score vs Actual Impact.
        """
        if not HAS_MATPLOTLIB or not predicted: return None
        
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(predicted, actual, alpha=0.6, c='#3498db', edgecolors='w', s=80)
        
        # Reference diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="Ideal Prediction")
        
        ax.set_xlabel('Predicted Criticality (Cscore)')
        ax.set_ylabel('Actual Failure Impact (Simulation)')
        ax.set_title(title)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return ChartOutput(title, self._fig_to_base64(fig), "Correlation check: Do predicted critical nodes actually cause high impact?")

    def plot_quality_comparison(self, components: List[any], title: str) -> Optional[ChartOutput]:
        """Grouped bar chart for R/M/A scores of top components."""
        if not HAS_MATPLOTLIB or not components: return None
        
        names = [c.id for c in components]
        r_scores = [c.scores.reliability for c in components]
        m_scores = [c.scores.maintainability for c in components]
        a_scores = [c.scores.availability for c in components]
        
        x = np.arange(len(names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, r_scores, width, label='Reliability', color='#e74c3c')
        ax.bar(x, m_scores, width, label='Maintainability', color='#3498db')
        ax.bar(x + width, a_scores, width, label='Availability', color='#2ecc71')

        ax.set_ylabel('Score (0-1)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        
        return ChartOutput(title, self._fig_to_base64(fig), "Multi-dimensional quality assessment for top critical components.")