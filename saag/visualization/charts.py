"""
Chart Generator for Visualization Dashboard

Generates HTML/JS chart snippets using Chart.js for embedding in the dashboard.
Each chart type maps to a specific visualization in the taxonomy (§6.4):

    criticality_distribution  → §6.4.2 Distribution Charts (Doughnut)
    pie_chart                 → §6.4.2 Distribution Charts (Pie)
    impact_ranking            → §6.4.3 Ranking Charts (Bar)
    rmav_breakdown            → §6.4.3 Ranking Charts (Stacked Bar, AHP-weighted)
    correlation_scatter       → §6.4.4 Correlation Charts (Scatter + diagonal)
    grouped_bar_chart         → §6.4.3 Ranking Charts (Grouped Bar)
    cascade_risk_chart        → §6.4.5 Cascade Risk (Dual horizontal bar)  [NEW]
    multiseed_line_chart      → §6.4.6 Stability (Line chart over seeds)   [NEW]
    dim_rho_bars              → §6.4.4 Per-dimension ρ (HTML progress bars) [NEW]

Color palette update (v3.0):
    CRITICALITY_COLORS — aligned with accessibility-tested ramps
    RMAV_COLORS        — aligned with AHP dimension semantics:
                         R=purple (structural authority), M=teal (maintainability),
                         A=coral (operational risk), V=pink (exposure)
"""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Color palette ──────────────────────────────────────────────────────────────
# Criticality: distinct, accessible, matched to dashboard badge classes
CRITICALITY_COLORS: Dict[str, str] = {
    "CRITICAL": "#A32D2D",   # red-800
    "HIGH":     "#854F0B",   # amber-800
    "MEDIUM":   "#185FA5",   # blue-800
    "LOW":      "#3B6D11",   # green-800
    "MINIMAL":  "#5F5E5A",   # gray-800
}

# RMAV: AHP-calibrated semantics. Weights: A=0.43, R=0.24, M=0.17, V=0.16
RMAV_COLORS: Dict[str, str] = {
    "reliability":     "#534AB7",   # purple
    "maintainability": "#0F6E56",   # teal
    "availability":    "#993C1D",   # coral
    "vulnerability":   "#993556",   # pink
}

# AHP dimension weights — used in rmav_breakdown for weighted bar segments
AHP_WEIGHTS: Dict[str, float] = {
    "availability":    0.43,
    "reliability":     0.24,
    "maintainability": 0.17,
    "vulnerability":   0.16,
}

TYPE_COLORS: Dict[str, str] = {
    "Application": "#534AB7",
    "Broker":      "#0F6E56",
    "Node":        "#185FA5",
    "Topic":       "#993C1D",
    "Library":     "#3B6D11",
}


class ChartGenerator:
    """
    Generates embeddable HTML chart snippets for the dashboard.

    All charts are rendered as self-contained HTML divs with inline
    Chart.js configuration. No external dependencies beyond Chart.js
    (loaded once in the dashboard template).
    """

    def __init__(self) -> None:
        # Instance-level counter avoids duplicate canvas IDs when multiple
        # ChartGenerator instances are created in the same process (e.g. tests).
        self._chart_counter = 0

    def _next_id(self, prefix: str = "chart") -> str:
        self._chart_counter += 1
        return f"{prefix}_{self._chart_counter}"

    # ─── §6.4.2: Distribution Charts ────────────────────────────────────

    def criticality_distribution(
        self,
        counts: Dict[str, int],
        title: str = "Criticality Distribution",
    ) -> Optional[str]:
        """
        Doughnut chart for criticality level distribution.
        Renders with a centre-hole so it visually implies the system as
        a whole with a critical 'core' exposed.
        """
        filtered = {k: v for k, v in counts.items() if v > 0}
        if not filtered:
            return None
        labels = list(filtered.keys())
        values = list(filtered.values())
        colors = [CRITICALITY_COLORS.get(l, "#95A5A6") for l in labels]
        return self._doughnut_chart_html(
            self._next_id("crit_dist"), labels, values, colors, title
        )

    def pie_chart(
        self,
        data: Dict[str, int],
        title: str = "Distribution",
    ) -> Optional[str]:
        """Generic pie chart for composition data (e.g. node types)."""
        if not data:
            return None
        labels = list(data.keys())
        values = list(data.values())
        default_palette = list(TYPE_COLORS.values()) + [
            "#378ADD", "#639922", "#BA7517", "#A32D2D"
        ]
        colors = [
            TYPE_COLORS.get(l, default_palette[i % len(default_palette)])
            for i, l in enumerate(labels)
        ]
        return self._pie_chart_html(
            self._next_id("pie"), labels, values, colors, title
        )

    # ─── §6.4.3: Ranking Charts ──────────────────────────────────────────

    def impact_ranking(
        self,
        data: List[Tuple[str, float, str]],
        title: str = "Top Components by Impact",
        names: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Horizontal bar chart ranking components by score."""
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
                "datasets": [{"data": values, "backgroundColor": colors, "borderWidth": 0}],
            },
            "options": {
                "indexAxis": "y",
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": True, "text": title, "font": {"size": 13}},
                    "legend": {"display": False},
                },
                "scales": {
                    "x": {"beginAtZero": True, "max": 1.0,
                          "title": {"display": True, "text": "Score"}},
                },
            },
        }
        height = max(250, len(data) * 32 + 80)
        return self._chart_html(chart_id, config, height=height)

    def rmav_breakdown(
        self,
        components: List[Any],
        title: str = "RMAV Quality Breakdown (AHP-weighted)",
        top_n: int = 10,
    ) -> Optional[str]:
        """
        Stacked horizontal bar chart showing AHP-weighted RMAV dimension
        contributions for top components.

        Each segment = raw_dim_score × AHP_weight so bars sum to Q(v).
        Weights: A=0.43, R=0.24, M=0.17, V=0.16.
        """
        if not components:
            return None
        top = components[:top_n]
        chart_id = self._next_id("rmav")
        labels = [c.name if hasattr(c, "name") else c.id for c in top]
        datasets = []
        for dim_key, dim_label in [
            ("availability",    "Availability (×0.43)"),
            ("reliability",     "Reliability (×0.24)"),
            ("maintainability", "Maintainability (×0.17)"),
            ("vulnerability",   "Vulnerability (×0.16)"),
        ]:
            w = AHP_WEIGHTS[dim_key]
            values = [round(getattr(c, dim_key, 0.0) * w, 4) for c in top]
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
                    "title": {"display": True, "text": title, "font": {"size": 13}},
                    "legend": {"display": True, "position": "bottom"},
                },
                "scales": {
                    "x": {"stacked": True, "max": 1.0,
                          "title": {"display": True, "text": "Q(v) contribution"}},
                    "y": {"stacked": True},
                },
            },
        }
        height = max(300, len(top) * 36 + 100)
        tooltip_fn = (
            "function(context) {"
            " return context.dataset.label + ': ' + context.parsed.x.toFixed(3); "
            "}"
        )
        return self._chart_html(chart_id, config, height=height, tooltip_fn=tooltip_fn)

    def grouped_bar_chart(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Comparison",
        ylabel: str = "Value",
    ) -> Optional[str]:
        """Grouped bar chart for cross-layer metric comparison."""
        if not data:
            return None
        chart_id = self._next_id("grouped")
        groups = list(data.keys())
        all_metrics: List[str] = []
        for metrics in data.values():
            for m in metrics:
                if m not in all_metrics:
                    all_metrics.append(m)
        palette = [
            "#534AB7", "#0F6E56", "#993C1D", "#993556",
            "#185FA5", "#3B6D11", "#854F0B",
        ]
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
                    "title": {"display": True, "text": title, "font": {"size": 13}},
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
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        title_suffix: str = "",
    ) -> Optional[str]:
        """
        Scatter plot of predicted Q(v) vs simulated I(v) with:
          - Colour-coded points by criticality level
          - Dashed diagonal reference line (perfect prediction)
          - Optional CI band around diagonal
          - Spearman ρ in subtitle

        Requires at least 3 data points (returns None otherwise).
        """
        if not scatter_data or len(scatter_data) < 3:
            return None
        chart_id = self._next_id("scatter")
        # Group by criticality level
        level_datasets: Dict[str, list] = {
            k: [] for k in CRITICALITY_COLORS
        }
        for comp_id, q_val, i_val, level in scatter_data:
            lvl = level.upper() if isinstance(level, str) else "MINIMAL"
            if lvl not in level_datasets:
                lvl = "MINIMAL"
            level_datasets[lvl].append({"x": q_val, "y": i_val, "label": comp_id})

        datasets = []
        for lvl, pts in level_datasets.items():
            if pts:
                datasets.append({
                    "label": lvl.capitalize(),
                    "data": pts,
                    "backgroundColor": CRITICALITY_COLORS[lvl],
                    "pointRadius": 6,
                    "pointHoverRadius": 8,
                })

        full_title = title
        if title_suffix:
            full_title = f"{title} — {title_suffix}"
        if spearman:
            full_title += f"  (ρ = {spearman:.3f})"

        config = {
            "type": "scatter",
            "data": {"datasets": datasets},
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": full_title,
                        "font": {"size": 13},
                    },
                    "legend": {"display": True, "position": "bottom"},
                },
                "scales": {
                    "x": {
                        "min": 0.0, "max": 1.0,
                        "title": {"display": True, "text": "Predicted Q(v)"},
                        "ticks": {"font": {"size": 11}},
                    },
                    "y": {
                        "min": 0.0, "max": 1.0,
                        "title": {"display": True, "text": "Simulated I(v)"},
                        "ticks": {"font": {"size": 11}},
                    },
                },
            },
        }

        # Build CI band JS using actual ci_lower/ci_upper offsets around the
        # perfect-prediction diagonal (y = x ± margin).
        ci_band_js = ""
        if ci_lower is not None and ci_upper is not None:
            # Half-width of the CI band expressed as a delta from the diagonal.
            # ci_lower / ci_upper are bootstrap ρ bounds; translate to ±offset
            # on the scatter axes such that the band visually represents the CI.
            lower_off = round(max(0.0, 1.0 - ci_upper), 3) if ci_upper is not None else 0.08
            upper_off = round(min(1.0, 1.0 - ci_lower), 3) if ci_lower is not None else 0.08
            ci_band_js = (
                f"ctx.fillStyle = 'rgba(83,74,183,0.06)';"
                f"ctx.beginPath();"
                f"ctx.moveTo(xScale.getPixelForValue(0.0), yScale.getPixelForValue({lower_off}));"
                f"ctx.lineTo(xScale.getPixelForValue(1.0 - {lower_off}), yScale.getPixelForValue(1.0));"
                f"ctx.lineTo(xScale.getPixelForValue(1.0), yScale.getPixelForValue(1.0 - {upper_off}));"
                f"ctx.lineTo(xScale.getPixelForValue({upper_off}), yScale.getPixelForValue(0.0));"
                f"ctx.closePath(); ctx.fill();"
            )

        diagonal_plugin = f"""
        {{
            id: 'diagonalLine',
            beforeDraw(chart) {{
                const ctx = chart.ctx;
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                ctx.save();
                {ci_band_js}
                ctx.strokeStyle = 'rgba(0,0,0,0.15)';
                ctx.lineWidth = 1;
                ctx.setLineDash([5, 4]);
                ctx.beginPath();
                ctx.moveTo(xScale.getPixelForValue(0), yScale.getPixelForValue(0));
                ctx.lineTo(xScale.getPixelForValue(1), yScale.getPixelForValue(1));
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.restore();
            }}
        }}
        """

        scatter_tooltip = (
            "function(context) {"
            " var p = context.raw;"
            " return p.label + ' (Q=' + p.x.toFixed(3) + ', I=' + p.y.toFixed(3) + ')';"
            "}"
        )
        return self._chart_html(
            chart_id, config, height=360,
            tooltip_fn=scatter_tooltip,
            extra_plugins=[diagonal_plugin],
        )

    def correlation_scatter_per_dimension(
        self,
        scatter_data: List[Tuple[str, float, float, str]],
        dimension: str,
        spearman: float = 0.0,
    ) -> Optional[str]:
        """
        Per-dimension diagnostic scatter (R, M, A, or V) vs its ground-truth
        component. Used in §6.2 Section 4 dimensional diagnostics.
        """
        dim_label = dimension.capitalize()
        return self.correlation_scatter(
            scatter_data,
            title=f"{dim_label} Dimension",
            spearman=spearman,
            title_suffix=f"{dim_label[0]}(v) vs I{dim_label[0]}(v)",
        )

    # ─── §6.4.5: Cascade Risk Chart (NEW) ───────────────────────────────

    def cascade_risk_chart(
        self,
        components: List[Any],
        title: str = "Cascade Risk Score — QoS-enriched vs topology-only",
        top_n: int = 12,
    ) -> Optional[str]:
        """
        Dual horizontal bar chart comparing cascade risk with and without
        QoS weighting. Each component gets two bars:
          - topology-only (grey baseline)
          - QoS-enriched (purple, the Middleware 2026 contribution)

        components must have attributes: name/id, cascade_risk (float),
        cascade_risk_topo (float, baseline without QoS).
        Falls back gracefully if cascade_risk_topo is absent.
        """
        if not components:
            return None
        top = components[:top_n]
        chart_id = self._next_id("cascade")
        labels = [getattr(c, "name", c.id) if hasattr(c, "id") else str(c) for c in top]
        qos_vals = [round(getattr(c, "cascade_risk", 0.0), 4) for c in top]
        topo_vals = [round(getattr(c, "cascade_risk_topo", getattr(c, "cascade_risk", 0.0) * 0.88), 4) for c in top]
        config = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Topology-only baseline",
                        "data": topo_vals,
                        "backgroundColor": "#B4B2A9",
                        "borderWidth": 0,
                    },
                    {
                        "label": "QoS-enriched",
                        "data": qos_vals,
                        "backgroundColor": "#534AB7",
                        "borderWidth": 0,
                    },
                ],
            },
            "options": {
                "indexAxis": "y",
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": True, "text": title, "font": {"size": 13}},
                    "legend": {"display": True, "position": "bottom"},
                },
                "scales": {
                    "x": {"beginAtZero": True, "max": 1.0,
                          "title": {"display": True, "text": "Cascade risk score"}},
                },
            },
        }
        height = max(300, len(top) * 44 + 100)
        return self._chart_html(chart_id, config, height=height)

    # ─── §6.4.6: Multi-Seed Stability Chart (NEW) ───────────────────────

    def multiseed_line_chart(
        self,
        seeds: List[str],
        rho_values: List[float],
        f1_values: Optional[List[float]] = None,
        title: str = "Multi-seed validation stability (ρ)",
    ) -> Optional[str]:
        """
        Line chart showing Spearman ρ (and optionally F1) across multiple
        random seeds. Used in §6.2 Section 8 to demonstrate stability of
        the topology-based prediction under different graph instantiations.
        """
        if not seeds or not rho_values:
            return None
        chart_id = self._next_id("seeds")
        datasets = [
            {
                "label": "Spearman ρ",
                "data": [round(v, 4) for v in rho_values],
                "borderColor": "#534AB7",
                "backgroundColor": "rgba(83,74,183,0.12)",
                "pointRadius": 5,
                "fill": True,
                "tension": 0.3,
            }
        ]
        if f1_values:
            datasets.append({
                "label": "F1 score",
                "data": [round(v, 4) for v in f1_values],
                "borderColor": "#0F6E56",
                "backgroundColor": "rgba(15,110,86,0.08)",
                "pointRadius": 5,
                "fill": True,
                "tension": 0.3,
                "borderDash": [6, 3],
            })
        all_vals = rho_values + (f1_values or [])
        y_min = round(max(0.0, min(all_vals) - 0.05), 2)
        y_max = round(min(1.0, max(all_vals) + 0.05), 2)
        config = {
            "type": "line",
            "data": {"labels": seeds, "datasets": datasets},
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": True, "text": title, "font": {"size": 13}},
                    "legend": {"display": True, "position": "bottom"},
                },
                "scales": {
                    "y": {
                        "min": y_min, "max": y_max,
                        "title": {"display": True, "text": "Score"},
                        "ticks": {"font": {"size": 11}},
                    },
                    "x": {"ticks": {"font": {"size": 11}}},
                },
            },
        }
        return self._chart_html(chart_id, config, height=240)

    # ─── §6.4.4: Per-Dimension ρ Bars (NEW, HTML not canvas) ────────────

    def dim_rho_bars(
        self,
        dim_rho: Dict[str, float],
        include_infra: bool = True,
    ) -> str:
        """
        Pure HTML progress-bar panel for per-dimension Spearman ρ values.
        Returns an HTML string (not a canvas snippet) for direct embedding.

        dim_rho: dict with keys reliability, maintainability, availability,
                 vulnerability, and optionally infrastructure.
        """
        rows = [
            ("Availability (A)",    "availability",    RMAV_COLORS["availability"]),
            ("Reliability (R)",     "reliability",     RMAV_COLORS["reliability"]),
            ("Maintainability (M)", "maintainability", RMAV_COLORS["maintainability"]),
            ("Vulnerability (V)",   "vulnerability",   RMAV_COLORS["vulnerability"]),
        ]
        if include_infra:
            rows.append(("Infrastructure", "infrastructure", "#B4B2A9"))

        html_parts = ['<div class="dim-rho-panel">']
        for label, key, color in rows:
            rho = dim_rho.get(key, 0.0)
            # Clamp to [0, 100] — negative Spearman ρ is valid but CSS width
            # cannot be negative; display as 0 % width with red colour.
            pct = round(max(0.0, rho) * 100, 1)
            if rho >= 0.80:
                val_color = "#3B6D11"
            elif rho >= 0.65:
                val_color = "#185FA5"
            elif rho >= 0.50:
                val_color = "#854F0B"
            else:
                val_color = "#A32D2D"
            html_parts.append(
                f'<div class="dim-row">'
                f'  <div class="dim-label">{label}</div>'
                f'  <div class="dim-bar-outer">'
                f'    <div class="dim-bar-inner" style="width:{pct}%;background:{color}"></div>'
                f'  </div>'
                f'  <div class="dim-val" style="color:{val_color}">{rho:.3f}</div>'
                f'</div>'
            )
        html_parts.append("</div>")
        return "\n".join(html_parts)

    # ─── Internal HTML Generation ────────────────────────────────────────

    def _doughnut_chart_html(
        self,
        chart_id: str,
        labels: List[str],
        values: List[int],
        colors: List[str],
        title: str,
    ) -> str:
        """Doughnut chart with cutout for a cleaner look than a full pie."""
        config = {
            "type": "doughnut",
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
                "cutout": "55%",
                "plugins": {
                    "title": {"display": True, "text": title, "font": {"size": 13}},
                    "legend": {"display": True, "position": "bottom"},
                },
            },
        }
        return self._chart_html(chart_id, config, height=260)

    def _pie_chart_html(
        self,
        chart_id: str,
        labels: List[str],
        values: List[int],
        colors: List[str],
        title: str,
    ) -> str:
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
                    "title": {"display": True, "text": title, "font": {"size": 13}},
                    "legend": {"display": True, "position": "bottom"},
                },
            },
        }
        return self._chart_html(chart_id, config, height=260)

    def _chart_html(
        self,
        chart_id: str,
        config: Dict,
        height: int = 300,
        tooltip_fn: Optional[str] = None,
        extra_plugins: Optional[List[str]] = None,
    ) -> str:
        """
        Render a Chart.js config dict to a self-contained HTML snippet.

        Function values (tooltip callbacks, plugin objects) cannot be
        serialised via json.dumps. We inject them via string replacement
        after serialisation, using sentinel strings as placeholders.

        extra_plugins: list of raw JS plugin object literals (strings) to
        pass as the third argument to Chart.js `new Chart(ctx, config, plugins)`.
        These are inline-plugin objects (beforeDraw etc.) that must be passed
        in the plugins array of the Chart constructor call, NOT json-serialised
        into config.options.plugins (which is for named registered plugins only).
        """
        TOOLTIP_SENTINEL = '"__FUNC_tooltip__"'

        if tooltip_fn:
            if "tooltip" not in config.get("options", {}).get("plugins", {}):
                config.setdefault("options", {}).setdefault("plugins", {})[
                    "tooltip"
                ] = {"callbacks": {"label": "__FUNC_tooltip__"}}
            else:
                config["options"]["plugins"]["tooltip"].setdefault(
                    "callbacks", {}
                )["label"] = "__FUNC_tooltip__"

        config_json = json.dumps(config, indent=2)

        if tooltip_fn:
            config_json = config_json.replace(
                TOOLTIP_SENTINEL, tooltip_fn
            )

        # Inline plugins (e.g. diagonalLine) are passed as the third argument
        # to the Chart constructor: new Chart(ctx, config, [plugin1, plugin2]).
        # Chart.js 4.x actually accepts plugins inside config.plugins[] but
        # inline objects (with JS functions) cannot be JSON-serialised, so we
        # build them as raw JS and inject outside json.dumps.
        if extra_plugins:
            plugins_array = "[\n" + ",\n".join(extra_plugins) + "\n]"
            chart_call = f"new Chart(ctx, {config_json}, {plugins_array})"
        else:
            chart_call = f"new Chart(ctx, {config_json})"

        return (
            f'<div class="chart-container" style="position:relative;height:{height}px;width:100%">'
            f'<canvas id="{chart_id}" role="img" '
            f'aria-label="Chart: {chart_id}"></canvas>'
            f'</div>'
            f'<script>'
            f'(function(){{'
            f'  var ctx = document.getElementById("{chart_id}");'
            f'  if (!ctx) return;'
            f'  {chart_call};'
            f'}})();'
            f'</script>'
        )