"""
Chart Generator for Visualization Dashboard

Generates HTML/JS chart snippets using Chart.js for embedding in the dashboard.
Each chart type maps to a specific visualization in the taxonomy (§6.4):

    criticality_distribution → §6.4.2 Distribution Charts (Pie)
    pie_chart               → §6.4.2 Distribution Charts (Pie)
    impact_ranking          → §6.4.3 Ranking Charts (Bar)
    rmav_breakdown          → §6.4.3 Ranking Charts (Stacked Bar)
    correlation_scatter     → §6.4.4 Correlation Charts (Scatter)
    grouped_bar_chart       → §6.4.3 Ranking Charts (Grouped Bar)
"""
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from .models import ColorTheme, DEFAULT_THEME

logger = logging.getLogger(__name__)

# Use theme for colors
CRITICALITY_COLORS = DEFAULT_THEME.to_criticality_dict()
RMAV_COLORS = {
    "reliability": "#3498DB",
    "maintainability": "#2ECC71",
    "availability": "#E67E22",
    "vulnerability": "#9B59B6",
}
TYPE_COLORS = {
    "Application": "#4A90D9",
    "Broker": "#9B59B6",
    "Node": "#27AE60",
    "Topic": "#F39C12",
    "Library": "#1ABC9C",
}


class ChartGenerator:
    """
    Generates embeddable HTML chart snippets for the dashboard.

    All charts are rendered as self-contained HTML divs with inline
    Chart.js configuration. No external dependencies beyond Chart.js
    (loaded once in the dashboard template).
    """

    _chart_counter = 0

    def _next_id(self, prefix: str = "chart") -> str:
        """Generate a unique chart ID."""
        ChartGenerator._chart_counter += 1
        return f"{prefix}_{ChartGenerator._chart_counter}"

    # ─── §6.4.2: Distribution Charts (Pie) ──────────────────────────────

    def criticality_distribution(
        self,
        counts: Dict[str, int],
        title: str = "Criticality Distribution",
    ) -> Optional[str]:
        """
        Generate a pie chart showing component distribution by criticality level.

        Args:
            counts: Dict mapping level names to counts
                    e.g. {"CRITICAL": 5, "HIGH": 8, "MEDIUM": 15, ...}
            title: Chart title
        """
        # Filter out zero counts
        filtered = {k: v for k, v in counts.items() if v > 0}
        if not filtered:
            return None

        labels = list(filtered.keys())
        values = list(filtered.values())
        colors = [CRITICALITY_COLORS.get(l, "#95A5A6") for l in labels]

        return self._pie_chart_html(
            self._next_id("crit_dist"), labels, values, colors, title
        )

    def pie_chart(
        self,
        data: Dict[str, int],
        title: str = "Distribution",
    ) -> Optional[str]:
        """
        Generate a generic pie chart.

        Args:
            data: Dict mapping category names to counts
            title: Chart title
        """
        if not data:
            return None

        labels = list(data.keys())
        values = list(data.values())
        # Use type colors if keys match, else use a default palette
        default_palette = ["#4A90D9", "#9B59B6", "#27AE60", "#F39C12", "#1ABC9C",
                           "#E74C3C", "#3498DB", "#E67E22"]
        colors = [
            TYPE_COLORS.get(l, default_palette[i % len(default_palette)])
            for i, l in enumerate(labels)
        ]

        return self._pie_chart_html(
            self._next_id("pie"), labels, values, colors, title
        )

    # ─── §6.4.3: Ranking Charts (Bar) ───────────────────────────────────

    def impact_ranking(
        self,
        data: List[Tuple[str, float, str]],
        title: str = "Top Components by Impact",
        names: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Generate a horizontal bar chart ranking components by score.

        Args:
            data: List of (component_id, score, level) tuples
            title: Chart title
            names: Optional ID-to-name mapping for display labels
        """
        if not data:
            return None

        names = names or {}
        chart_id = self._next_id("ranking")

        labels = [names.get(d[0], d[0]) for d in data]
        values = [d[1] for d in data]
        colors = [CRITICALITY_COLORS.get(d[2], "#95A5A6") for d in data]

        config = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "data": values,
                    "backgroundColor": colors,
                    "borderColor": colors,
                    "borderWidth": 1,
                }],
            },
            "options": {
                "indexAxis": "y",
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": True, "text": title, "font": {"size": 14}},
                    "legend": {"display": False},
                },
                "scales": {
                    "x": {"beginAtZero": True, "max": 1.0,
                           "title": {"display": True, "text": "Score"}},
                },
            },
        }

        height = max(250, len(data) * 30 + 80)
        return self._chart_html(chart_id, config, height=height)

    def rmav_breakdown(
        self,
        components: List[Any],
        title: str = "RMAV Quality Breakdown",
        top_n: int = 10,
    ) -> Optional[str]:
        """
        Generate a stacked horizontal bar chart showing RMAV dimension
        contributions for top components.

        This chart answers: "Which quality dimension drives criticality
        for each component?" (§6.7 Workflow 3: Architecture Assessment)

        Args:
            components: List of ComponentDetail objects
            title: Chart title
            top_n: Number of top components to show
        """
        if not components:
            return None

        top = components[:top_n]
        chart_id = self._next_id("rmav")

        labels = [c.name if hasattr(c, "name") else c.id for c in top]

        datasets = []
        for dim_key, dim_label in [
            ("reliability", "Reliability"),
            ("maintainability", "Maintainability"),
            ("availability", "Availability"),
            ("vulnerability", "Vulnerability"),
        ]:
            values = [getattr(c, dim_key, 0.0) for c in top]
            datasets.append({
                "label": dim_label,
                "data": values,
                "backgroundColor": RMAV_COLORS[dim_key],
                "borderWidth": 0,
            })

        config = {
            "type": "bar",
            "data": {"labels": labels, "datasets": datasets},
            "options": {
                "indexAxis": "y",
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": True, "text": title, "font": {"size": 14}},
                    "legend": {"display": True, "position": "bottom"},
                    "tooltip": {
                        "callbacks": {
                            "label": "__FUNC_tooltip__",
                        },
                    },
                },
                "scales": {
                    "x": {
                        "stacked": True,
                        "title": {"display": True, "text": "Score Contribution"},
                    },
                    "y": {"stacked": True},
                },
            },
        }

        height = max(300, len(top) * 35 + 100)
        # Custom tooltip function
        tooltip_fn = "function(context) { return context.dataset.label + ': ' + context.parsed.x.toFixed(3); }"
        return self._chart_html(chart_id, config, height=height, tooltip_fn=tooltip_fn)

    def grouped_bar_chart(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Comparison",
        ylabel: str = "Value",
    ) -> Optional[str]:
        """
        Generate a grouped bar chart for cross-layer comparison.

        Args:
            data: Dict of {group_label: {metric_name: value}}
            title: Chart title
            ylabel: Y-axis label
        """
        if not data:
            return None

        chart_id = self._next_id("grouped")
        groups = list(data.keys())

        # Collect all metric names
        all_metrics = []
        for metrics in data.values():
            for m in metrics:
                if m not in all_metrics:
                    all_metrics.append(m)

        palette = ["#4A90D9", "#E67E22", "#2ECC71", "#9B59B6", "#E74C3C", "#1ABC9C"]
        datasets = []
        for i, metric in enumerate(all_metrics):
            values = [data[g].get(metric, 0) for g in groups]
            datasets.append({
                "label": metric,
                "data": values,
                "backgroundColor": palette[i % len(palette)],
            })

        config = {
            "type": "bar",
            "data": {"labels": groups, "datasets": datasets},
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": True, "text": title, "font": {"size": 14}},
                    "legend": {"display": True, "position": "bottom"},
                },
                "scales": {
                    "y": {"beginAtZero": True,
                           "title": {"display": True, "text": ylabel}},
                },
            },
        }

        return self._chart_html(chart_id, config, height=300)

    # ─── §6.4.4: Correlation Charts (Scatter) ───────────────────────────

    def correlation_scatter(
        self,
        scatter_data: List[Tuple[str, float, float, str]],
        title: str = "Prediction Correlation: Q(v) vs I(v)",
        spearman: float = 0.0,
    ) -> Optional[str]:
        """
        Generate a scatter plot comparing predicted Q(v) against simulated I(v).

        This is the central validation visualization that confirms the
        methodology's core claim: topology predicts impact (§6.4.4).

        Points near the diagonal indicate good prediction. Outliers above
        the diagonal suggest under-prediction; below suggests over-prediction.

        Args:
            scatter_data: List of (component_id, Q(v), I(v), level) tuples
            title: Chart title
            spearman: Spearman correlation to display in subtitle
        """
        if not scatter_data or len(scatter_data) < 3:
            return None

        chart_id = self._next_id("scatter")

        # Group points by criticality level for color-coded datasets
        level_groups: Dict[str, List[Dict]] = {}
        for comp_id, q_val, i_val, level in scatter_data:
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append({"x": q_val, "y": i_val, "label": comp_id})

        datasets = []
        for level, points in level_groups.items():
            color = CRITICALITY_COLORS.get(level, "#95A5A6")
            datasets.append({
                "label": level,
                "data": points,
                "backgroundColor": color,
                "borderColor": color,
                "pointRadius": 6,
                "pointHoverRadius": 9,
            })

        subtitle = f"Spearman ρ = {spearman:.3f}" if spearman else ""

        config = {
            "type": "scatter",
            "data": {"datasets": datasets},
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": [title, subtitle] if subtitle else title,
                        "font": {"size": 14},
                    },
                    "legend": {"display": True, "position": "bottom"},
                    "tooltip": {
                        "callbacks": {
                            "label": "__FUNC_scatter_tooltip__",
                        },
                    },
                },
                "scales": {
                    "x": {
                        "beginAtZero": True,
                        "max": 1.0,
                        "title": {"display": True, "text": "Predicted Q(v)"},
                    },
                    "y": {
                        "beginAtZero": True,
                        "max": 1.0,
                        "title": {"display": True, "text": "Simulated I(v)"},
                    },
                },
            },
        }

        # Diagonal reference line plugin
        diagonal_plugin = """
        {
            id: 'diagonalLine',
            beforeDraw(chart) {
                const ctx = chart.ctx;
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                ctx.save();
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
                ctx.lineWidth = 1;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(xScale.getPixelForValue(0), yScale.getPixelForValue(0));
                ctx.lineTo(xScale.getPixelForValue(1), yScale.getPixelForValue(1));
                ctx.stroke();
                ctx.restore();
            }
        }
        """

        scatter_tooltip = (
            "function(context) {"
            "  var p = context.raw;"
            "  return p.label + ' (Q=' + p.x.toFixed(3) + ', I=' + p.y.toFixed(3) + ')';"
            "}"
        )

        return self._chart_html(
            chart_id,
            config,
            height=400,
            tooltip_fn=scatter_tooltip,
            plugins=[diagonal_plugin],
        )

    # ─── Internal HTML Generation ────────────────────────────────────────

    def _pie_chart_html(
        self,
        chart_id: str,
        labels: List[str],
        values: List[int],
        colors: List[str],
        title: str,
    ) -> str:
        """Generate HTML for a pie chart."""
        config = {
            "type": "pie",
            "data": {
                "labels": labels,
                "datasets": [{
                    "data": values,
                    "backgroundColor": colors,
                    "borderWidth": 2,
                    "borderColor": "#ffffff",
                }],
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": True, "text": title, "font": {"size": 14}},
                    "legend": {"display": True, "position": "bottom"},
                },
            },
        }
        return self._chart_html(chart_id, config, height=280)

    def _chart_html(
        self,
        chart_id: str,
        config: Dict,
        height: int = 300,
        tooltip_fn: Optional[str] = None,
        plugins: Optional[List[str]] = None,
    ) -> str:
        """
        Generate the HTML+JS snippet for a Chart.js chart.

        Handles function injection for tooltips and plugins that
        cannot be expressed in JSON.
        """
        config_json = json.dumps(config, indent=2)

        # Replace tooltip function placeholders
        if tooltip_fn:
            for placeholder in [
                '"__FUNC_tooltip__"',
                '"__FUNC_scatter_tooltip__"',
            ]:
                config_json = config_json.replace(placeholder, tooltip_fn)

        # Add plugins array
        plugins_str = ""
        if plugins:
            plugins_str = f", plugins: [{', '.join(plugins)}]"

        return f"""
        <div class="chart-container" style="height: {height}px; margin-bottom: 20px;">
            <canvas id="{chart_id}"></canvas>
        </div>
        <script>
            (function() {{
                const ctx = document.getElementById('{chart_id}').getContext('2d');
                new Chart(ctx, {config_json}{plugins_str});
            }})();
        </script>
        """