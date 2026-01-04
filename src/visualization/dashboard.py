"""
Dashboard Generator

Constructs an HTML dashboard to visualize Graph Stats, Quality Analysis, and Simulation Results.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from .charts import ChartOutput

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: #f4f7f6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
        .section {{ background: white; margin-bottom: 20px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
        .card {{ background: #fff; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }}
        .stat-box {{ display: inline-block; background: #3498db; color: white; padding: 15px; border-radius: 4px; margin: 10px; min-width: 150px; text-align: center; }}
        .stat-val {{ font-size: 24px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .critical {{ color: #e74c3c; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; }}
        .footer {{ text-align: center; color: #7f8c8d; margin-top: 20px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>Generated: {timestamp}</p>
        </div>
        
        {content}
        
        <div class="footer">Software-as-a-Graph Visualization Module</div>
    </div>
</body>
</html>
"""

class DashboardGenerator:
    def __init__(self, title: str = "Graph Analysis Dashboard"):
        self.title = title
        self.sections = []

    def add_kpi_section(self, kpis: Dict[str, Any]):
        """Add a row of Key Performance Indicators."""
        html_parts = ['<div class="section"><h2>System Overview</h2><div style="text-align:center">']
        for label, val in kpis.items():
            html_parts.append(f"""
                <div class="stat-box">
                    <div class="stat-val">{val}</div>
                    <div class="stat-label">{label}</div>
                </div>
            """)
        html_parts.append('</div></div>')
        self.sections.append("".join(html_parts))

    def add_charts_section(self, title: str, charts: List[ChartOutput]):
        """Add a grid of charts."""
        if not charts: return
        html_parts = [f'<div class="section"><h2>{title}</h2><div class="grid">']
        for chart in charts:
            html_parts.append(f"""
                <div class="card">
                    <h3>{chart.title}</h3>
                    <img src="data:image/png;base64,{chart.png_base64}" />
                </div>
            """)
        html_parts.append('</div></div>')
        self.sections.append("".join(html_parts))

    def add_table_section(self, title: str, headers: List[str], rows: List[List[Any]]):
        """Add a data table."""
        html_parts = [f'<div class="section"><h2>{title}</h2><table><thead><tr>']
        for h in headers:
            html_parts.append(f'<th>{h}</th>')
        html_parts.append('</tr></thead><tbody>')
        
        for row in rows:
            html_parts.append('<tr>')
            for cell in row:
                # Highlight "CRITICAL" text
                style = 'class="critical"' if str(cell) == "CRITICAL" else ""
                html_parts.append(f'<td {style}>{cell}</td>')
            html_parts.append('</tr>')
            
        html_parts.append('</tbody></table></div>')
        self.sections.append("".join(html_parts))

    def generate(self) -> str:
        """Return full HTML string."""
        return HTML_TEMPLATE.format(
            title=self.title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content="\n".join(self.sections)
        )