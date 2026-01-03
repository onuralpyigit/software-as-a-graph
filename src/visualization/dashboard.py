"""
Visualization Dashboard - Version 5.0

HTML dashboard generation for graph analysis visualization.

Features:
- Graph statistics dashboard
- Analysis results dashboard
- Simulation results dashboard
- Validation results dashboard
- Combined overview dashboard

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import html
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

from .charts import (
    ChartConfig,
    ChartOutput,
    chart_component_distribution,
    chart_edge_distribution,
    chart_layer_summary,
    chart_impact_ranking,
    chart_criticality_distribution,
    chart_correlation_comparison,
    chart_confusion_matrix,
    chart_layer_validation,
    chart_method_comparison,
    chart_delivery_stats,
    chart_layer_performance,
    check_matplotlib_available,
)


# =============================================================================
# Dashboard Configuration
# =============================================================================

@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    title: str = "Graph Analysis Dashboard"
    theme: str = "light"  # light, dark
    chart_config: ChartConfig = field(default_factory=ChartConfig)
    include_timestamp: bool = True
    
    # Colors for the dashboard
    primary_color: str = "#3498db"
    success_color: str = "#2ecc71"
    warning_color: str = "#f39c12"
    danger_color: str = "#e74c3c"
    bg_color: str = "#f8f9fa"
    card_bg: str = "#ffffff"
    text_color: str = "#2c3e50"


# =============================================================================
# HTML Templates
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: {bg_color};
            color: {text_color};
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, {primary_color}, #2980b9);
            color: white;
            padding: 30px 20px;
            margin-bottom: 30px;
            border-radius: 8px;
        }}
        header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        header .meta {{ opacity: 0.9; font-size: 0.9em; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: {card_bg};
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .card-header {{
            background: {primary_color};
            color: white;
            padding: 15px 20px;
            font-weight: 600;
        }}
        .card-body {{ padding: 20px; }}
        .card-body img {{ max-width: 100%; height: auto; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .stat-box {{
            background: {bg_color};
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: {primary_color};
        }}
        .stat-label {{ color: #666; font-size: 0.9em; }}
        .status-passed {{ color: {success_color}; }}
        .status-partial {{ color: {warning_color}; }}
        .status-failed {{ color: {danger_color}; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: {bg_color}; font-weight: 600; }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge-success {{ background: {success_color}; color: white; }}
        .badge-warning {{ background: {warning_color}; color: white; }}
        .badge-danger {{ background: {danger_color}; color: white; }}
        .full-width {{ grid-column: 1 / -1; }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
        <footer>
            Generated by Software-as-a-Graph Visualization Module
            {timestamp}
        </footer>
    </div>
</body>
</html>
"""


# =============================================================================
# Dashboard Builder
# =============================================================================

class DashboardBuilder:
    """Builds HTML dashboard content."""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self._sections: List[str] = []
    
    def add_header(self, title: str, subtitle: str = "") -> DashboardBuilder:
        """Add dashboard header."""
        meta = f'<div class="meta">{html.escape(subtitle)}</div>' if subtitle else ""
        self._sections.append(f"""
        <header>
            <h1>{html.escape(title)}</h1>
            {meta}
        </header>
        """)
        return self
    
    def add_stats_row(self, stats: Dict[str, Any], title: str = "") -> DashboardBuilder:
        """Add a row of stat boxes."""
        boxes = []
        for label, value in stats.items():
            if isinstance(value, float):
                formatted = f"{value:.4f}" if value < 10 else f"{value:.2f}"
            else:
                formatted = str(value)
            
            # Check for status colors
            css_class = ""
            if isinstance(value, str):
                if value.upper() == "PASSED":
                    css_class = "status-passed"
                elif value.upper() == "PARTIAL":
                    css_class = "status-partial"
                elif value.upper() == "FAILED":
                    css_class = "status-failed"
            
            boxes.append(f"""
            <div class="stat-box">
                <div class="stat-value {css_class}">{formatted}</div>
                <div class="stat-label">{html.escape(label)}</div>
            </div>
            """)
        
        header = f'<div class="card-header">{html.escape(title)}</div>' if title else ""
        self._sections.append(f"""
        <div class="card full-width">
            {header}
            <div class="card-body">
                <div class="stats-grid">
                    {''.join(boxes)}
                </div>
            </div>
        </div>
        """)
        return self
    
    def add_chart(self, chart: ChartOutput, full_width: bool = False) -> DashboardBuilder:
        """Add a chart card."""
        width_class = "full-width" if full_width else ""
        self._sections.append(f"""
        <div class="card {width_class}">
            <div class="card-header">{html.escape(chart.title)}</div>
            <div class="card-body">
                {chart.to_html_img()}
            </div>
        </div>
        """)
        return self
    
    def add_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        title: str = "",
        full_width: bool = False,
    ) -> DashboardBuilder:
        """Add a data table."""
        header_html = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        
        row_html = []
        for row in rows:
            cells = []
            for cell in row:
                # Format cell
                if isinstance(cell, float):
                    formatted = f"{cell:.4f}"
                elif isinstance(cell, bool):
                    badge_class = "badge-success" if cell else "badge-danger"
                    badge_text = "Yes" if cell else "No"
                    formatted = f'<span class="badge {badge_class}">{badge_text}</span>'
                elif isinstance(cell, str) and cell.upper() in ["PASSED", "PARTIAL", "FAILED"]:
                    badge_map = {
                        "PASSED": "badge-success",
                        "PARTIAL": "badge-warning",
                        "FAILED": "badge-danger",
                    }
                    formatted = f'<span class="badge {badge_map[cell.upper()]}">{cell}</span>'
                else:
                    formatted = html.escape(str(cell))
                cells.append(f"<td>{formatted}</td>")
            row_html.append(f"<tr>{''.join(cells)}</tr>")
        
        width_class = "full-width" if full_width else ""
        card_header = f'<div class="card-header">{html.escape(title)}</div>' if title else ""
        
        self._sections.append(f"""
        <div class="card {width_class}">
            {card_header}
            <div class="card-body">
                <table>
                    <thead><tr>{header_html}</tr></thead>
                    <tbody>{''.join(row_html)}</tbody>
                </table>
            </div>
        </div>
        """)
        return self
    
    def start_grid(self) -> DashboardBuilder:
        """Start a grid layout."""
        self._sections.append('<div class="grid">')
        return self
    
    def end_grid(self) -> DashboardBuilder:
        """End a grid layout."""
        self._sections.append('</div>')
        return self
    
    def add_section_title(self, title: str) -> DashboardBuilder:
        """Add a section title."""
        self._sections.append(f"""
        <div class="full-width" style="margin: 30px 0 15px 0;">
            <h2 style="color: {self.config.primary_color}; border-bottom: 2px solid {self.config.primary_color}; padding-bottom: 10px;">
                {html.escape(title)}
            </h2>
        </div>
        """)
        return self
    
    def build(self) -> str:
        """Build the complete HTML."""
        timestamp = ""
        if self.config.include_timestamp:
            timestamp = f"| {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return HTML_TEMPLATE.format(
            title=self.config.title,
            bg_color=self.config.bg_color,
            card_bg=self.config.card_bg,
            text_color=self.config.text_color,
            primary_color=self.config.primary_color,
            success_color=self.config.success_color,
            warning_color=self.config.warning_color,
            danger_color=self.config.danger_color,
            content="\n".join(self._sections),
            timestamp=timestamp,
        )


# =============================================================================
# Dashboard Generators
# =============================================================================

def generate_graph_dashboard(
    graph,  # SimulationGraph
    title: str = "Graph Model Dashboard",
    config: Optional[DashboardConfig] = None,
) -> str:
    """
    Generate dashboard for graph model statistics.
    
    Args:
        graph: SimulationGraph instance
        title: Dashboard title
        config: Dashboard configuration
    
    Returns:
        HTML string
    """
    config = config or DashboardConfig(title=title)
    builder = DashboardBuilder(config)
    
    summary = graph.summary()
    
    # Header
    builder.add_header(title, f"Graph Model Analysis")
    
    # Stats row
    builder.add_stats_row({
        "Total Components": summary['total_components'],
        "Total Edges": summary['total_edges'],
        "Message Paths": summary['message_paths'],
    }, "Overview")
    
    builder.start_grid()
    
    # Component distribution chart
    if check_matplotlib_available():
        comp_chart = chart_component_distribution(
            summary.get('by_type', {}),
            "Components by Type",
            config.chart_config,
        )
        builder.add_chart(comp_chart)
        
        edge_chart = chart_edge_distribution(
            summary.get('by_edge_type', {}),
            "Edges by Type",
            config.chart_config,
        )
        builder.add_chart(edge_chart)
    
    # Component table
    if summary.get('by_type'):
        builder.add_table(
            headers=["Component Type", "Count"],
            rows=[[t, c] for t, c in summary['by_type'].items()],
            title="Component Breakdown",
        )
    
    # Edge table
    if summary.get('by_edge_type'):
        builder.add_table(
            headers=["Edge Type", "Count"],
            rows=[[t, c] for t, c in summary['by_edge_type'].items()],
            title="Edge Breakdown",
        )
    
    builder.end_grid()
    
    return builder.build()


def generate_simulation_dashboard(
    graph,  # SimulationGraph
    campaign_result,  # CampaignResult
    title: str = "Simulation Results Dashboard",
    config: Optional[DashboardConfig] = None,
) -> str:
    """
    Generate dashboard for simulation results.
    
    Args:
        graph: SimulationGraph instance
        campaign_result: CampaignResult from failure simulation
        title: Dashboard title
        config: Dashboard configuration
    
    Returns:
        HTML string
    """
    config = config or DashboardConfig(title=title)
    builder = DashboardBuilder(config)
    
    # Calculate avg/max impact from results
    impacts = [r.impact_score for r in campaign_result.results]
    avg_impact = sum(impacts) / len(impacts) if impacts else 0
    max_impact = max(impacts) if impacts else 0
    
    # Header
    builder.add_header(title, f"Failure Simulation Analysis")
    
    # Stats row
    builder.add_stats_row({
        "Total Simulations": campaign_result.total_simulations,
        "Critical Components": len(campaign_result.get_critical()),
        "Avg Impact": avg_impact,
        "Max Impact": max_impact,
        "Duration (ms)": f"{campaign_result.duration_ms:.0f}",
    }, "Simulation Summary")
    
    builder.start_grid()
    
    # Impact ranking chart
    if check_matplotlib_available():
        impacts_ranked = campaign_result.ranked_by_impact()
        impact_chart = chart_impact_ranking(
            impacts_ranked, top_n=15,
            title="Top 15 by Impact",
            config=config.chart_config,
        )
        builder.add_chart(impact_chart)
        
        # Criticality distribution
        type_summary = campaign_result.get_type_summary() if hasattr(campaign_result, 'get_type_summary') else {}
        if type_summary:
            crit_counts = {}
            for comp_type, summary in type_summary.items():
                crit_counts[comp_type] = summary.get('critical_count', 0)
            
            if any(crit_counts.values()):
                crit_chart = chart_criticality_distribution(
                    crit_counts,
                    "Critical by Component Type",
                    config.chart_config,
                )
                builder.add_chart(crit_chart)
    
    # Top impacts table
    impacts_ranked = campaign_result.ranked_by_impact()[:15]
    rows = []
    for comp_id, impact in impacts_ranked:
        comp = graph.get_component(comp_id)
        comp_type = comp.type.value if comp else "Unknown"
        critical = impact >= 0.3
        rows.append([comp_id, comp_type, impact, critical])
    
    builder.add_table(
        headers=["Component", "Type", "Impact", "Critical"],
        rows=rows,
        title="Impact Ranking",
        full_width=True,
    )
    
    # Layer results
    if campaign_result.by_layer:
        builder.add_section_title("Results by Layer")
        
        layer_rows = []
        for layer_key, layer_result in sorted(campaign_result.by_layer.items()):
            critical_count = len(layer_result.get_critical())
            layer_rows.append([
                layer_result.layer_name,
                layer_result.count,
                layer_result.avg_impact,
                layer_result.max_impact,
                critical_count,
            ])
        
        builder.add_table(
            headers=["Layer", "Count", "Avg Impact", "Max Impact", "Critical"],
            rows=layer_rows,
            title="Layer Summary",
            full_width=True,
        )
    
    builder.end_grid()
    
    return builder.build()


def generate_validation_dashboard(
    pipeline_result,  # PipelineResult
    title: str = "Validation Results Dashboard",
    config: Optional[DashboardConfig] = None,
) -> str:
    """
    Generate dashboard for validation results.
    
    Args:
        pipeline_result: PipelineResult from validation
        title: Dashboard title
        config: Dashboard configuration
    
    Returns:
        HTML string
    """
    config = config or DashboardConfig(title=title)
    builder = DashboardBuilder(config)
    
    result = pipeline_result.validation
    
    # Header
    builder.add_header(title, f"Prediction vs Simulation Analysis")
    
    # Stats row
    builder.add_stats_row({
        "Status": result.status.value.upper(),
        "Spearman ρ": result.spearman,
        "F1-Score": result.f1_score,
        "Precision": result.classification.precision,
        "Recall": result.classification.recall,
    }, "Validation Summary")
    
    builder.start_grid()
    
    # Charts
    if check_matplotlib_available():
        # Correlation metrics
        corr_metrics = {
            "Spearman": result.spearman,
            "Pearson": result.correlation.pearson,
            "Kendall": result.correlation.kendall,
        }
        targets = {"Spearman": 0.7, "Pearson": 0.65, "Kendall": 0.6}
        corr_chart = chart_correlation_comparison(
            corr_metrics, targets,
            "Correlation Metrics",
            config.chart_config,
        )
        builder.add_chart(corr_chart)
        
        # Confusion matrix
        cm = result.classification.confusion_matrix
        cm_chart = chart_confusion_matrix(
            cm.true_positives, cm.false_positives,
            cm.false_negatives, cm.true_negatives,
            "Confusion Matrix",
            config.chart_config,
        )
        builder.add_chart(cm_chart)
        
        # Layer validation
        if pipeline_result.by_layer:
            layer_data = {}
            for layer, lr in pipeline_result.by_layer.items():
                layer_data[layer] = {
                    'spearman': lr.spearman,
                    'f1_score': lr.f1_score,
                }
            layer_chart = chart_layer_validation(
                layer_data,
                "Validation by Layer",
                config.chart_config,
            )
            builder.add_chart(layer_chart, full_width=True)
        
        # Method comparison
        if pipeline_result.method_comparison:
            method_data = {}
            for method, comp in pipeline_result.method_comparison.items():
                method_data[method] = {
                    'spearman': comp.spearman,
                    'f1_score': comp.f1_score,
                }
            method_chart = chart_method_comparison(
                method_data,
                "Method Comparison",
                config.chart_config,
            )
            builder.add_chart(method_chart, full_width=True)
    
    # Layer results table
    if pipeline_result.by_layer:
        layer_rows = []
        for layer, lr in sorted(pipeline_result.by_layer.items()):
            status = "PASSED" if lr.passed else "FAILED"
            layer_rows.append([layer, lr.spearman, lr.f1_score, lr.count, status])
        
        builder.add_table(
            headers=["Layer", "Spearman ρ", "F1-Score", "Samples", "Status"],
            rows=layer_rows,
            title="Layer Results",
            full_width=True,
        )
    
    # Method comparison table
    if pipeline_result.method_comparison:
        method_rows = []
        for method, comp in sorted(pipeline_result.method_comparison.items()):
            method_rows.append([
                method,
                comp.spearman,
                comp.f1_score,
                comp.status.value.upper(),
            ])
        
        builder.add_table(
            headers=["Method", "Spearman ρ", "F1-Score", "Status"],
            rows=method_rows,
            title="Method Comparison",
        )
    
    # Timing
    builder.add_table(
        headers=["Phase", "Duration (ms)"],
        rows=[
            ["Analysis", pipeline_result.analysis_time_ms],
            ["Simulation", pipeline_result.simulation_time_ms],
            ["Validation", pipeline_result.validation_time_ms],
            ["Total", pipeline_result.analysis_time_ms + pipeline_result.simulation_time_ms + pipeline_result.validation_time_ms],
        ],
        title="Timing",
    )
    
    builder.end_grid()
    
    return builder.build()


def generate_overview_dashboard(
    graph,  # SimulationGraph
    campaign_result=None,  # Optional CampaignResult
    validation_result=None,  # Optional PipelineResult
    title: str = "System Overview Dashboard",
    config: Optional[DashboardConfig] = None,
) -> str:
    """
    Generate combined overview dashboard.
    
    Args:
        graph: SimulationGraph instance
        campaign_result: Optional CampaignResult
        validation_result: Optional PipelineResult
        title: Dashboard title
        config: Dashboard configuration
    
    Returns:
        HTML string
    """
    config = config or DashboardConfig(title=title)
    builder = DashboardBuilder(config)
    
    summary = graph.summary()
    
    # Header
    builder.add_header(title, "Multi-Layer Graph Analysis Overview")
    
    # Overview stats
    stats = {
        "Components": summary['total_components'],
        "Edges": summary['total_edges'],
        "Message Paths": summary['message_paths'],
    }
    
    if campaign_result:
        impacts = [r.impact_score for r in campaign_result.results]
        max_impact = max(impacts) if impacts else 0
        stats["Critical"] = len(campaign_result.get_critical())
        stats["Max Impact"] = max_impact
    
    if validation_result:
        stats["Status"] = validation_result.validation.status.value.upper()
        stats["Spearman ρ"] = validation_result.spearman
    
    builder.add_stats_row(stats, "System Overview")
    
    # Graph section
    builder.add_section_title("Graph Model")
    builder.start_grid()
    
    if check_matplotlib_available():
        comp_chart = chart_component_distribution(
            summary.get('by_type', {}),
            "Components by Type",
            config.chart_config,
        )
        builder.add_chart(comp_chart)
        
        edge_chart = chart_edge_distribution(
            summary.get('by_edge_type', {}),
            "Edges by Type",
            config.chart_config,
        )
        builder.add_chart(edge_chart)
    
    builder.end_grid()
    
    # Simulation section
    if campaign_result:
        builder.add_section_title("Failure Simulation")
        builder.start_grid()
        
        if check_matplotlib_available():
            impacts = campaign_result.ranked_by_impact()
            impact_chart = chart_impact_ranking(
                impacts, top_n=10,
                title="Top 10 by Impact",
                config=config.chart_config,
            )
            builder.add_chart(impact_chart)
        
        # Quick table
        impacts = campaign_result.ranked_by_impact()[:10]
        rows = [[comp_id, f"{impact:.4f}"] for comp_id, impact in impacts]
        builder.add_table(
            headers=["Component", "Impact"],
            rows=rows,
            title="Impact Ranking",
        )
        
        builder.end_grid()
    
    # Validation section
    if validation_result:
        builder.add_section_title("Validation Results")
        builder.start_grid()
        
        result = validation_result.validation
        
        if check_matplotlib_available():
            corr_metrics = {
                "Spearman": result.spearman,
                "F1-Score": result.f1_score,
            }
            targets = {"Spearman": 0.7, "F1-Score": 0.9}
            corr_chart = chart_correlation_comparison(
                corr_metrics, targets,
                "Key Metrics",
                config.chart_config,
            )
            builder.add_chart(corr_chart)
            
            cm = result.classification.confusion_matrix
            cm_chart = chart_confusion_matrix(
                cm.true_positives, cm.false_positives,
                cm.false_negatives, cm.true_negatives,
                "Confusion Matrix",
                config.chart_config,
            )
            builder.add_chart(cm_chart)
        
        builder.end_grid()
    
    return builder.build()
