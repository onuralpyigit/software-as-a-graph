"""
Visualization Charts

Generates static charts (Base64 encoded PNGs) for embedding in HTML dashboards.
Handles visual comparisons between layers and components.
"""

import io
import base64
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

try:
    import matplotlib
    matplotlib.use('Agg') # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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

    def plot_distribution(self, data: Dict[str, int], title: str) -> Optional[ChartOutput]:
        """Pie chart for component type distribution."""
        if not HAS_MATPLOTLIB or not data: return None
        
        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, _, autotexts = ax.pie(
            data.values(), 
            labels=data.keys(), 
            autopct='%1.1f%%', 
            startangle=90,
            colors=plt.cm.Pastel1.colors
        )
        plt.setp(autotexts, size=10, weight="bold")
        ax.set_title(title)
        return ChartOutput(title, self._fig_to_base64(fig))

    def plot_quality_comparison(self, names: List[str], scores: Dict[str, List[float]], title: str) -> Optional[ChartOutput]:
        """Grouped bar chart comparing R, M, A scores."""
        if not HAS_MATPLOTLIB or not names: return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(names))
        width = 0.25
        multiplier = 0
        
        for attribute, measurement in scores.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            multiplier += 1

        ax.set_ylabel('Score (0-1)')
        ax.set_title(title)
        ax.set_xticks(x + width, names)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right', ncols=3)
        plt.xticks(rotation=45, ha='right')
        
        return ChartOutput(title, self._fig_to_base64(fig))

    def plot_impact_ranking(self, impact_map: Dict[str, float], title: str, top_n: int = 10) -> Optional[ChartOutput]:
        """Horizontal bar chart for simulation impact."""
        if not HAS_MATPLOTLIB or not impact_map: return None
        
        # Sort and slice
        sorted_items = sorted(impact_map.items(), key=lambda x: x[1], reverse=True)[:top_n]
        # Reverse for horizontal bar (top item at top)
        sorted_items = sorted_items[::-1]
        
        keys = [x[0] for x in sorted_items]
        vals = [x[1] for x in sorted_items]
        
        fig, ax = plt.subplots(figsize=(8, max(4, len(keys) * 0.5)))
        bars = ax.barh(keys, vals, color='#e74c3c')
        ax.set_xlabel('Affected Nodes Count')
        ax.set_title(title)
        
        # Add values to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', va='center')
            
        return ChartOutput(title, self._fig_to_base64(fig))