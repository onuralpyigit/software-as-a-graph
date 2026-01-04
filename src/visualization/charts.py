"""
Visualization Charts

Chart generation for graph analysis and simulation results.
Uses matplotlib to generate base64-encoded images for HTML embedding.
"""

import io
import base64
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

try:
    import matplotlib
    matplotlib.use('Agg') # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

@dataclass
class ChartOutput:
    title: str
    png_base64: str

class ChartGenerator:
    """Generates static charts for the dashboard."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        if HAS_MATPLOTLIB:
            try:
                plt.style.use(style)
            except:
                pass # Fallback to default
        
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    def plot_bar(self, data: Dict[str, float], title: str, xlabel: str, ylabel: str) -> Optional[ChartOutput]:
        """Generic vertical bar chart."""
        if not HAS_MATPLOTLIB or not data: return None
        
        fig, ax = plt.subplots(figsize=(8, 5))
        keys = list(data.keys())
        vals = list(data.values())
        
        bars = ax.bar(keys, vals, color='#3498db')
        ax.set_title(title, pad=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
                        
        return ChartOutput(title, self._fig_to_base64(fig))

    def plot_horizontal_bar(self, data: Dict[str, float], title: str, color: str = '#e74c3c') -> Optional[ChartOutput]:
        """Generic horizontal bar chart (good for rankings)."""
        if not HAS_MATPLOTLIB or not data: return None
        
        # Sort data
        sorted_items = sorted(data.items(), key=lambda x: x[1])
        keys = [x[0] for x in sorted_items]
        vals = [x[1] for x in sorted_items]
        
        fig, ax = plt.subplots(figsize=(8, max(4, len(keys) * 0.4)))
        ax.barh(keys, vals, color=color)
        ax.set_title(title)
        return ChartOutput(title, self._fig_to_base64(fig))

    def plot_pie(self, data: Dict[str, int], title: str) -> Optional[ChartOutput]:
        """Pie chart for distributions."""
        if not HAS_MATPLOTLIB or not data: return None
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', startangle=90)
        ax.set_title(title)
        return ChartOutput(title, self._fig_to_base64(fig))

    def plot_grouped_bar(self, categories: List[str], series: Dict[str, List[float]], title: str) -> Optional[ChartOutput]:
        """Grouped bar chart for comparisons (e.g., R/M/A scores)."""
        if not HAS_MATPLOTLIB or not categories: return None
        
        import numpy as np
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(categories))
        width = 0.8 / len(series)
        multiplier = 0
        
        for label, values in series.items():
            offset = width * multiplier
            ax.bar(x + offset, values, width, label=label)
            multiplier += 1
            
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width, categories)
        ax.legend(loc='upper right')
        plt.xticks(rotation=45, ha='right')
        
        return ChartOutput(title, self._fig_to_base64(fig))