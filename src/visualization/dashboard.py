"""
Dashboard Generator

Generates responsive HTML dashboards for multi-layer graph analysis.
Includes KPIs, charts, tables, network visualizations, and validation results.

Features:
    - Responsive grid layout
    - Navigation sidebar
    - Interactive vis.js network graphs
    - Collapsible sections
    - Print-friendly styling
"""

from __future__ import annotations
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .charts import ChartOutput


# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/vis-network.min.js"></script>
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #34495e;
            --accent: #3498db;
            --success: #2ecc71;
            --warning: #f39c12;
            --danger: #e74c3c;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --border: #dee2e6;
            --text: #2c3e50;
            --text-muted: #7f8c8d;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        
        /* Navigation */
        .navbar {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 30px;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        }}
        
        .navbar-brand {{
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        
        .navbar-nav {{
            display: flex;
            gap: 25px;
        }}
        
        .navbar-nav a {{
            color: rgba(255,255,255,0.8);
            text-decoration: none;
            font-size: 0.9rem;
            transition: color 0.3s;
        }}
        
        .navbar-nav a:hover {{ color: white; }}
        
        /* Main Content */
        .main-content {{
            margin-top: 60px;
            padding: 30px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }}
        
        /* Header */
        .page-header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.05);
        }}
        
        .page-header h1 {{
            font-size: 2rem;
            color: var(--primary);
            margin-bottom: 10px;
        }}
        
        .page-header .meta {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        
        /* Sections */
        .section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.05);
            scroll-margin-top: 80px;
        }}
        
        .section-header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--accent);
        }}
        
        .section-header h2 {{
            font-size: 1.4rem;
            color: var(--primary);
            margin: 0;
        }}
        
        .section-badge {{
            background: var(--accent);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}
        
        .kpi-card {{
            background: linear-gradient(135deg, var(--secondary), var(--primary));
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.2s;
        }}
        
        .kpi-card:hover {{ transform: translateY(-3px); }}
        
        .kpi-value {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        
        .kpi-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.9;
        }}
        
        .kpi-card.success {{ background: linear-gradient(135deg, #27ae60, #2ecc71); }}
        .kpi-card.warning {{ background: linear-gradient(135deg, #e67e22, #f39c12); }}
        .kpi-card.danger {{ background: linear-gradient(135deg, #c0392b, #e74c3c); }}
        
        /* Chart Grid */
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin-bottom: 25px;
        }}
        
        .chart-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        
        .chart-card h4 {{
            color: var(--secondary);
            margin-bottom: 15px;
            font-size: 1rem;
        }}
        
        .chart-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}
        
        .chart-card .description {{
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 10px;
        }}
        
        /* Tables */
        .table-container {{
            overflow-x: auto;
            margin-bottom: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            background: var(--bg);
            font-weight: 600;
            color: var(--secondary);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }}
        
        tr:hover {{ background: rgba(52, 152, 219, 0.05); }}
        
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
        }}
        
        .badge-critical {{ background: #e74c3c; }}
        .badge-high {{ background: #e67e22; }}
        .badge-medium {{ background: #f1c40f; color: #333; }}
        .badge-low {{ background: #2ecc71; }}
        .badge-minimal {{ background: #95a5a6; }}
        .badge-passed {{ background: #2ecc71; }}
        .badge-failed {{ background: #e74c3c; }}
        
        /* Metrics Box */
        .metrics-box {{
            background: var(--bg);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .metrics-box h4 {{
            margin-bottom: 15px;
            color: var(--secondary);
        }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px dashed var(--border);
        }}
        
        .metric-row:last-child {{ border-bottom: none; }}
        
        .metric-name {{ font-weight: 500; color: var(--text); }}
        .metric-value {{ font-family: 'Consolas', monospace; font-weight: 600; }}
        .metric-value.pass {{ color: var(--success); }}
        .metric-value.fail {{ color: var(--danger); }}
        
        /* Network Graph */
        .network-container {{
            width: 100%;
            height: 500px;
            border: 1px solid var(--border);
            border-radius: 10px;
            margin-bottom: 20px;
            background: white;
        }}
        
        /* Subsections */
        .subsection {{
            margin-bottom: 25px;
        }}
        
        .subsection h3 {{
            font-size: 1.1rem;
            color: var(--secondary);
            margin-bottom: 15px;
            padding-left: 10px;
            border-left: 3px solid var(--accent);
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--border);
            margin-top: 40px;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .navbar {{ padding: 0 15px; }}
            .navbar-nav {{ display: none; }}
            .main-content {{ padding: 15px; }}
            .chart-grid {{ grid-template-columns: 1fr; }}
            .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        
        /* Print */
        @media print {{
            .navbar {{ display: none; }}
            .main-content {{ margin-top: 0; }}
            .section {{ break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-brand">📊 Software-as-a-Graph Dashboard</div>
        <div class="navbar-nav">
            {nav_links}
        </div>
    </nav>
    
    <div class="main-content">
        <div class="page-header">
            <h1>{title}</h1>
            <div class="meta">Generated: {timestamp}</div>
        </div>
        
        {content}
    </div>
    
    <div class="footer">
        Generated by Software-as-a-Graph Visualization Module • {timestamp}
    </div>
    
    {scripts}
</body>
</html>
"""


# =============================================================================
# Dashboard Generator
# =============================================================================

@dataclass
class NavLink:
    """Navigation link."""
    label: str
    anchor: str


class DashboardGenerator:
    """
    Generates responsive HTML dashboards.
    
    Example:
        >>> dash = DashboardGenerator("Analysis Report")
        >>> dash.start_section("Overview", "overview")
        >>> dash.add_kpis({"Nodes": 50, "Edges": 120})
        >>> dash.end_section()
        >>> html = dash.generate()
    """
    
    def __init__(self, title: str):
        self.title = title
        self.sections: List[str] = []
        self.nav_links: List[NavLink] = []
        self.scripts: List[str] = []
        self._current_section_id = ""
    
    def start_section(self, title: str, anchor_id: str = "") -> None:
        """Start a new section."""
        self._current_section_id = anchor_id or title.lower().replace(" ", "-")
        
        self.nav_links.append(NavLink(title, self._current_section_id))
        
        self.sections.append(
            f'<div class="section" id="{self._current_section_id}">'
            f'<div class="section-header"><h2>{title}</h2></div>'
        )
    
    def end_section(self) -> None:
        """End the current section."""
        self.sections.append('</div>')
    
    def add_kpis(self, kpis: Dict[str, Any], styles: Dict[str, str] = None) -> None:
        """
        Add KPI cards.
        
        Args:
            kpis: Dict mapping label to value
            styles: Optional dict mapping label to style class (success, warning, danger)
        """
        styles = styles or {}
        
        html = ['<div class="kpi-grid">']
        for label, value in kpis.items():
            style_class = styles.get(label, "")
            style_attr = f' {style_class}' if style_class else ""
            
            html.append(
                f'<div class="kpi-card{style_attr}">'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'</div>'
            )
        html.append('</div>')
        
        self.sections.append(''.join(html))
    
    def add_charts(self, charts: List[ChartOutput]) -> None:
        """Add chart images."""
        valid_charts = [c for c in charts if c is not None]
        if not valid_charts:
            return
        
        html = ['<div class="chart-grid">']
        for chart in valid_charts:
            html.append(
                f'<div class="chart-card">'
                f'<h4>{chart.title}</h4>'
                f'<img src="data:image/png;base64,{chart.png_base64}" alt="{chart.title}">'
            )
            if chart.description:
                html.append(f'<div class="description">{chart.description}</div>')
            html.append('</div>')
        html.append('</div>')
        
        self.sections.append(''.join(html))
    
    def add_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        title: str = ""
    ) -> None:
        """Add a data table."""
        html = ['<div class="table-container">']
        
        if title:
            html.append(f'<h4 style="margin-bottom: 10px;">{title}</h4>')
        
        html.append('<table><thead><tr>')
        for h in headers:
            html.append(f'<th>{h}</th>')
        html.append('</tr></thead><tbody>')
        
        for row in rows:
            html.append('<tr>')
            for cell in row:
                cell_str = str(cell)
                
                # Apply badge styling for known values
                if cell_str.upper() in ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL'):
                    cell_html = f'<span class="badge badge-{cell_str.lower()}">{cell_str}</span>'
                elif cell_str.upper() in ('PASSED', 'PASS', '✓'):
                    cell_html = f'<span class="badge badge-passed">PASSED</span>'
                elif cell_str.upper() in ('FAILED', 'FAIL', '✗'):
                    cell_html = f'<span class="badge badge-failed">FAILED</span>'
                else:
                    cell_html = cell_str
                
                html.append(f'<td>{cell_html}</td>')
            html.append('</tr>')
        
        html.append('</tbody></table></div>')
        self.sections.append(''.join(html))
    
    def add_metrics_box(
        self,
        metrics: Dict[str, Any],
        title: str = "Metrics",
        highlights: Dict[str, bool] = None
    ) -> None:
        """
        Add a metrics display box.
        
        Args:
            metrics: Dict mapping metric name to value
            title: Box title
            highlights: Dict mapping metric name to pass/fail (True/False)
        """
        highlights = highlights or {}
        
        html = [f'<div class="metrics-box"><h4>{title}</h4>']
        
        for name, value in metrics.items():
            val_class = ""
            if name in highlights:
                val_class = "pass" if highlights[name] else "fail"
            
            val_str = f'{value:.4f}' if isinstance(value, float) else str(value)
            class_attr = f' class="metric-value {val_class}"' if val_class else 'class="metric-value"'
            
            html.append(
                f'<div class="metric-row">'
                f'<span class="metric-name">{name}</span>'
                f'<span {class_attr}>{val_str}</span>'
                f'</div>'
            )
        
        html.append('</div>')
        self.sections.append(''.join(html))
    
    def add_subsection(self, title: str) -> None:
        """Add a subsection header."""
        self.sections.append(f'<div class="subsection"><h3>{title}</h3></div>')
    
    def add_html(self, html: str) -> None:
        """Add raw HTML content."""
        self.sections.append(html)
    
    def add_network_graph(
        self,
        graph_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        title: str = "Network Graph"
    ) -> None:
        """
        Add an interactive vis.js network graph.
        
        Args:
            graph_id: Unique ID for the graph container
            nodes: List of node dicts with id, label, group, etc.
            edges: List of edge dicts with from, to, etc.
            title: Graph title
        """
        import json
        
        html = [
            f'<h4 style="margin-bottom: 10px;">{title}</h4>',
            f'<div id="{graph_id}" class="network-container"></div>'
        ]
        self.sections.append(''.join(html))
        
        # Add vis.js initialization script
        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)
        
        script = f"""
        <script>
        (function() {{
            var nodes = new vis.DataSet({nodes_json});
            var edges = new vis.DataSet({edges_json});
            
            var container = document.getElementById('{graph_id}');
            var data = {{ nodes: nodes, edges: edges }};
            
            var options = {{
                nodes: {{
                    shape: 'dot',
                    font: {{ size: 12, face: 'Segoe UI' }},
                    borderWidth: 2,
                    shadow: true
                }},
                edges: {{
                    arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
                    smooth: {{ type: 'continuous' }},
                    color: {{ opacity: 0.7 }}
                }},
                physics: {{
                    stabilization: {{ iterations: 100 }},
                    barnesHut: {{
                        gravitationalConstant: -2000,
                        springLength: 150
                    }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 100
                }},
                groups: {{
                    Application: {{ color: {{ background: '#3498db', border: '#2980b9' }} }},
                    Broker: {{ color: {{ background: '#9b59b6', border: '#8e44ad' }} }},
                    Node: {{ color: {{ background: '#2ecc71', border: '#27ae60' }} }},
                    Topic: {{ color: {{ background: '#f1c40f', border: '#f39c12' }} }},
                    CRITICAL: {{ color: {{ background: '#e74c3c', border: '#c0392b' }} }},
                    HIGH: {{ color: {{ background: '#e67e22', border: '#d35400' }} }},
                    MEDIUM: {{ color: {{ background: '#f1c40f', border: '#f39c12' }} }},
                    LOW: {{ color: {{ background: '#2ecc71', border: '#27ae60' }} }},
                    MINIMAL: {{ color: {{ background: '#95a5a6', border: '#7f8c8d' }} }}
                }}
            }};
            
            new vis.Network(container, data, options);
        }})();
        </script>
        """
        self.scripts.append(script)
    
    def generate(self) -> str:
        """Generate the complete HTML document."""
        # Build navigation links
        nav_html = ' '.join(
            f'<a href="#{link.anchor}">{link.label}</a>'
            for link in self.nav_links
        )
        
        return HTML_TEMPLATE.format(
            title=self.title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            nav_links=nav_html,
            content='\n'.join(self.sections),
            scripts='\n'.join(self.scripts)
        )