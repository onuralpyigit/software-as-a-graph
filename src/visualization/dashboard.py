"""
Dashboard Generator

Constructs a responsive HTML dashboard visualizing multi-layer analysis.
"""

from datetime import datetime
from typing import List, Dict, Any
from .charts import ChartOutput

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{ --primary: #2c3e50; --secondary: #34495e; --accent: #3498db; --danger: #e74c3c; --light: #ecf0f1; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #f4f6f7; color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        
        header {{ background: white; border-bottom: 4px solid var(--primary); padding: 2rem; margin-bottom: 2rem; text-align: center; }}
        h1 {{ margin: 0; font-size: 2.2rem; color: var(--primary); }}
        .subtitle {{ color: #7f8c8d; margin-top: 0.5rem; font-size: 1.1rem; }}
        
        .section {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 2rem; }}
        .section-title {{ border-left: 5px solid var(--accent); padding-left: 15px; color: var(--primary); margin-bottom: 25px; font-size: 1.5rem; }}
        
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .kpi-card {{ background: linear-gradient(135deg, var(--secondary), var(--primary)); color: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .kpi-val {{ font-size: 2.5rem; font-weight: 700; margin-bottom: 5px; }}
        .kpi-label {{ font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; }}
        
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 30px; }}
        .chart-card {{ background: white; padding: 10px; text-align: center; }}
        .chart-desc {{ color: #7f8c8d; font-style: italic; font-size: 0.9rem; margin-top: 10px; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }}
        
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: var(--light); color: var(--primary); font-weight: 600; text-transform: uppercase; font-size: 0.85rem; }}
        tr:hover {{ background: #f8f9fa; }}
        
        .badge {{ padding: 5px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: bold; color: white; }}
        .bg-critical {{ background-color: #e74c3c; }}
        .bg-high {{ background-color: #e67e22; }}
        .bg-medium {{ background-color: #f1c40f; color: #333; }}
        .bg-low {{ background-color: #2ecc71; }}
        .bg-minimal {{ background-color: #95a5a6; }}
        
        footer {{ text-align: center; color: #95a5a6; padding: 30px; border-top: 1px solid #ddd; margin-top: 40px; }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>{title}</h1>
            <div class="subtitle">Multi-Layer Graph Modeling and Analysis Report</div>
            <div style="font-size: 0.9rem; margin-top: 10px; color: #95a5a6;">Generated: {timestamp}</div>
        </div>
    </header>
    
    <div class="container">
        {content}
    </div>
    
    <footer>
        Software-as-a-Graph
    </footer>
</body>
</html>
"""

class DashboardGenerator:
    def __init__(self, title: str = "System Analysis Dashboard"):
        self.title = title
        self.sections = []

    def add_section_header(self, title: str):
        self.sections.append(f'<div class="section"><h2 class="section-title">{title}</h2>')

    def close_section(self):
        self.sections.append('</div>')

    def add_kpis(self, kpis: Dict[str, Any]):
        html = ['<div class="kpi-grid">']
        for label, val in kpis.items():
            html.append(f"""
                <div class="kpi-card">
                    <div class="kpi-val">{val}</div>
                    <div class="kpi-label">{label}</div>
                </div>
            """)
        html.append('</div>')
        self.sections.append("".join(html))

    def add_charts(self, charts: List[ChartOutput]):
        if not charts: return
        html = ['<div class="chart-grid">']
        for chart in charts:
            html.append(f"""
                <div class="chart-card">
                    <h3>{chart.title}</h3>
                    <img src="data:image/png;base64,{chart.png_base64}" alt="{chart.title}"/>
                    <div class="chart-desc">{chart.description}</div>
                </div>
            """)
        html.append('</div>')
        self.sections.append("".join(html))

    def add_table(self, headers: List[str], rows: List[List[Any]]):
        html = ['<div style="overflow-x:auto;"><table><thead><tr>']
        for h in headers:
            html.append(f'<th>{h}</th>')
        html.append('</tr></thead><tbody>')
        
        for row in rows:
            html.append('<tr>')
            for cell in row:
                val_str = str(cell)
                # Badge logic for Criticality (Report Colors)
                badge_class = ""
                upper_val = val_str.upper()
                if upper_val == "CRITICAL": badge_class = "badge bg-critical"
                elif upper_val == "HIGH": badge_class = "badge bg-high"
                elif upper_val == "MEDIUM": badge_class = "badge bg-medium"
                elif upper_val == "LOW": badge_class = "badge bg-low"
                elif upper_val == "MINIMAL": badge_class = "badge bg-minimal"
                
                content = f'<span class="{badge_class}">{val_str}</span>' if badge_class else val_str
                html.append(f'<td>{content}</td>')
            html.append('</tr>')
            
        html.append('</tbody></table></div>')
        self.sections.append("".join(html))

    def generate(self) -> str:
        return HTML_TEMPLATE.format(
            title=self.title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content="\n".join(self.sections)
        )