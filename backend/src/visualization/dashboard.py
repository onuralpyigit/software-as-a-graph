"""
Dashboard Generator Service
"""
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .models import ChartOutput


# Component type to layer mapping
COMPONENT_LAYER_MAP = {
    "Application": "layer-app",
    "Library": "layer-app",
    "Topic": "layer-mw",
    "Broker": "layer-mw",
    "Node": "layer-infra",
}

# Layer definitions for compound nodes
LAYER_COMPOUNDS = {
    "layer-app": {"label": "Application Layer", "icon": "📱", "order": 0},
    "layer-mw": {"label": "Middleware Layer", "icon": "🔗", "order": 1},
    "layer-infra": {"label": "Infrastructure Layer", "icon": "🖥️", "order": 2},
}


# HTML Template
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- Fonts & Icons -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lucide@latest"></script>
    
    <!-- Visualization Libraries -->
    <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
    <script src="https://unpkg.com/layout-base@2.0.1/layout-base.js"></script>
    <script src="https://unpkg.com/cose-base@2.2.0/cose-base.js"></script>
    <script src="https://unpkg.com/cytoscape-cose-bilkent@4.1.0/cytoscape-cose-bilkent.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    
    <style>
        :root {{
            --primary: #4f46e5;
            --primary-light: #818cf8;
            --secondary: #ec4899;
            --accent: #f59e0b;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card-bg: rgba(255, 255, 255, 0.75);
            --border: rgba(226, 232, 240, 0.8);
            --text: #0f172a;
            --text-muted: #64748b;
            --glass-border: rgba(255, 255, 255, 0.3);
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
            background-attachment: fixed;
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .navbar {{
            position: fixed;
            top: 0; left: 0; right: 0;
            height: 64px;
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 40px;
            z-index: 1000;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .navbar-brand {{ 
            font-size: 1.1rem; 
            font-weight: 800; 
            letter-spacing: -0.5px; 
            text-transform: uppercase; 
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .navbar-brand span {{ opacity: 0.6; font-weight: 400; font-size: 0.8rem; }}
        
        .navbar-nav {{ display: flex; gap: 8px; }}
        .navbar-nav a {{ 
            color: rgba(255,255,255,0.7); 
            text-decoration: none; 
            font-size: 0.8rem; 
            font-weight: 600;
            transition: all 0.2s; 
            padding: 6px 14px;
            border-radius: 8px;
        }}
        .navbar-nav a:hover {{ color: white; background: rgba(255,255,255,0.1); }}
        
        .main-content {{ margin-top: 100px; padding: 0 40px 40px; max-width: 1600px; margin-left: auto; margin-right: auto; }}
        
        .page-header {{ 
            margin-bottom: 40px; 
            padding: 60px; 
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            position: relative;
            color: white;
            border-radius: 24px; 
            box-shadow: var(--shadow-lg); 
            overflow: hidden;
        }}
        
        .page-header::before {{
            content: '';
            position: absolute;
            top: -50%; left: -20%; width: 140%; height: 200%;
            background: radial-gradient(circle at 20% 30%, rgba(79, 70, 229, 0.15) 0%, transparent 40%),
                        radial-gradient(circle at 80% 70%, rgba(236, 72, 153, 0.1) 0%, transparent 40%);
            pointer-events: none;
        }}
        
        .page-header h1 {{ font-size: 2.75rem; font-weight: 900; margin-bottom: 12px; letter-spacing: -1.5px; line-height: 1.1; }}
        .page-header .meta {{ opacity: 0.7; font-size: 0.9rem; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }}
        
        .section {{ 
            background: var(--card-bg); 
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px; 
            padding: 40px; 
            margin-bottom: 40px; 
            box-shadow: var(--shadow); 
            scroll-margin-top: 100px; 
        }}
        
        .section-header {{ 
            display: flex; 
            align-items: center; 
            justify-content: space-between; 
            margin-bottom: 35px; 
            padding-bottom: 20px; 
            border-bottom: 1px solid var(--border); 
        }}
        
        .section-header h2 {{ font-size: 1.75rem; font-weight: 800; color: var(--text); letter-spacing: -0.7px; }}
        
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 24px; margin-bottom: 40px; }}
        
        .kpi-card {{ 
            background: rgba(255, 255, 255, 0.5); 
            border: 1px solid var(--glass-border);
            padding: 32px; 
            border-radius: 20px; 
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        }}
        
        .kpi-card:hover {{ transform: translateY(-8px) scale(1.02); box-shadow: var(--shadow-lg); background: white; }}
        
        .kpi-value {{ font-size: 3rem; font-weight: 800; margin-bottom: 4px; color: var(--primary); line-height: 1; letter-spacing: -1px; }}
        .kpi-label {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); }}
        
        .kpi-card.danger .kpi-value {{ color: var(--danger); }}
        .kpi-card.warning .kpi-value {{ color: var(--warning); }}
        .kpi-card.success .kpi-value {{ color: var(--success); }}
        
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 32px; margin-bottom: 40px; }}
        
        .chart-card {{ 
            background: rgba(255, 255, 255, 0.8); 
            border: 1px solid var(--glass-border);
            border-radius: 20px; 
            padding: 30px; 
            transition: all 0.3s ease; 
        }}
        .chart-card:hover {{ box-shadow: var(--shadow); background: white; }}
        .chart-card h4 {{ color: var(--text); margin-bottom: 24px; font-size: 1.1rem; font-weight: 700; opacity: 0.9; }}
        
        .table-container {{ overflow-x: auto; margin-bottom: 30px; border-radius: 20px; border: 1px solid var(--border); box-shadow: var(--shadow); background: white; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        th {{ background: #f8fafc; font-weight: 700; color: #475569; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 1px; padding: 18px 24px; border-bottom: 1px solid var(--border); }}
        td {{ padding: 16px 24px; border-bottom: 1px solid var(--border); color: #1e293b; font-weight: 500; }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background: #f1f5f9; }}
        
        .badge {{ display: inline-flex; align-items: center; padding: 6px 12px; border-radius: 8px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }}
        .badge-critical {{ background: #fee2e2; color: #b91c1c; }}
        .badge-high {{ background: #ffedd5; color: #c2410c; }}
        .badge-medium {{ background: #fef9c3; color: #854d0e; }}
        .badge-low {{ background: #dcfce7; color: #15803d; }}
        .badge-minimal {{ background: #f1f5f9; color: #475569; }}
        .badge-passed {{ background: #dcfce7; color: #15803d; }}
        .badge-failed {{ background: #fee2e2; color: #b91c1c; }}
        
        .metrics-box {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 24px; background: rgba(248, 250, 252, 0.5); border-radius: 16px; padding: 30px; border: 1px solid var(--border); }}
        .metric-item {{ display: flex; flex-direction: column; gap: 6px; }}
        .metric-name {{ font-size: 0.7rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }}
        .metric-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: var(--primary); }}
        
        .cytoscape-wrapper {{ border-radius: 20px; border: 1px solid var(--border); background: white; position: relative; overflow: hidden; height: 750px; box-shadow: var(--shadow); }}
        .cytoscape-container {{ height: 100%; width: 100%; }}
        
        .antipattern-card {{ 
            background: white; 
            border: 1px solid var(--border); 
            border-radius: 16px; 
            padding: 30px; 
            margin-bottom: 24px; 
            border-left: 8px solid var(--primary); 
            transition: all 0.3s; 
        }}
        .antipattern-card:hover {{ transform: translateX(8px); box-shadow: var(--shadow); }}
        .antipattern-card.critical {{ border-left-color: var(--danger); }}
        .antipattern-card.high {{ border-left-color: var(--warning); }}
        
        .footer {{ text-align: center; padding: 80px 40px; color: var(--text-muted); font-size: 0.85rem; border-top: 1px solid var(--border); margin-top: 80px; }}
        .footer p {{ margin-bottom: 8px; font-weight: 500; }}
        
        /* Interactive Tabs */
        .stakeholder-tabs {{ display: flex; gap: 8px; margin-bottom: 35px; background: rgba(0,0,0,0.05); padding: 6px; border-radius: 12px; width: fit-content; }}
        .tab-btn {{ padding: 10px 20px; border: none; background: none; font-weight: 700; color: var(--text-muted); cursor: pointer; border-radius: 8px; transition: all 0.2s; font-size: 0.85rem; }}
        .tab-btn:hover {{ background: rgba(255,255,255,0.5); color: var(--text); }}
        .tab-btn.active {{ background: white; color: var(--primary); box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}

        @media (max-width: 1024px) {{ .chart-grid {{ grid-template-columns: 1fr; }} .main-content {{ padding: 0 20px 20px; }} }}
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-brand">Software-as-a-Graph <span>v2.1 Precision Analysis</span></div>
        <div class="navbar-nav">{nav_links}</div>
    </nav>
    <div class="main-content">
        <div class="page-header">
            <div class="meta">Analysis Session • {timestamp} • Visualization Phase</div>
            <h1>{title}</h1>
        </div>
        {content}
    </div>
    <div class="footer">
        <p><strong>Software-as-a-Graph Methodology</strong></p>
        <p>Advanced Architectural Decision Support & Criticality Prediction Pipeline</p>
        <p style="margin-top:12px; opacity: 0.6;">&copy; 2026 Architectural Intelligence Framework</p>
    </div>
    <script>lucide.createIcons();</script>
    {scripts}
</body>
</html>
"""

@dataclass
class NavLink:
    """Navigation link."""
    label: str
    anchor: str


class DashboardGenerator:
    """
    Generates responsive HTML dashboards with Cytoscape.js network visualization.
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
        """Add KPI cards."""
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
    
    def add_charts(self, charts: List[Any]) -> None:
        """Add chart images or Chart.js HTML snippets."""
        valid_charts = [c for c in charts if c is not None]
        if not valid_charts:
            return
        html = ['<div class="chart-grid">']
        for chart in valid_charts:
            # Handle HTML string output (from Chart.js-based generator)
            if isinstance(chart, str):
                html.append(f'<div class="chart-card">{chart}</div>')
            # Handle ChartOutput objects (from PNG-based generator)
            elif hasattr(chart, 'png_base64'):
                html.append(
                    f'<div class="chart-card">'
                    f'<h4>{chart.title}</h4>'
                    f'<img src="data:image/png;base64,{chart.png_base64}" alt="{chart.alt_text or chart.title}">'
                )
                if chart.description:
                    html.append(f'<div class="description">{chart.description}</div>')
                html.append('</div>')
            else:
                # Unknown chart type, skip
                continue
        html.append('</div>')
        self.sections.append(''.join(html))
    
    def add_table(self, headers: List[str], rows: List[List[Any]], title: str = "") -> None:
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
    
    def add_metrics_box(self, metrics: Dict[str, Any], title: str = "Metrics", highlights: Dict[str, bool] = None) -> None:
        """Add a metrics display box."""
        highlights = highlights or {}
        html = [f'<div class="metrics-box"><h4>{title}</h4>']
        for name, value in metrics.items():
            val_class = "pass" if highlights.get(name) else ("fail" if name in highlights else "")
            val_str = f'{value:.4f}' if isinstance(value, float) else str(value)
            class_attr = f' class="metric-value {val_class}"' if val_class else 'class="metric-value"'
            html.append(f'<div class="metric-row"><span class="metric-name">{name}</span><span {class_attr}>{val_str}</span></div>')
        html.append('</div>')
        self.sections.append(''.join(html))
    
    def add_subsection(self, title: str) -> None:
        """Add a subsection header."""
        self.sections.append(f'<div class="subsection"><h3>{title}</h3></div>')
    
    def add_cytoscape_network(
        self, 
        graph_id: str, 
        nodes: List[Dict[str, Any]], 
        edges: List[Dict[str, Any]], 
        title: str = "Network Graph",
        use_compound_nodes: bool = True
    ) -> None:
        """
        Add an interactive Cytoscape.js network graph with compound nodes for layers.
        
        Args:
            graph_id: Unique identifier for the graph container
            nodes: List of node dictionaries with id, label, group, type, level, value
            edges: List of edge dictionaries with source, target, weight, dependency_type
            title: Title displayed above the graph
            use_compound_nodes: If True, group nodes into layer compound nodes
        """
        # Build elements list for Cytoscape
        elements = []
        
        if use_compound_nodes:
            # Add compound parent nodes for each layer
            for layer_id, layer_info in LAYER_COMPOUNDS.items():
                elements.append({
                    "data": {
                        "id": layer_id,
                        "label": f"{layer_info['icon']} {layer_info['label']}",
                        "isCompound": True,
                        "order": layer_info["order"]
                    },
                    "classes": "compound"
                })
        
        # Add regular nodes
        for node in nodes:
            node_type = node.get("type", "Application")
            parent = COMPONENT_LAYER_MAP.get(node_type) if use_compound_nodes else None
            level = node.get("level", "MINIMAL")
            
            node_data = {
                "data": {
                    "id": node["id"],
                    "label": node.get("label", node["id"]),
                    "nodeType": node_type,
                    "level": level,
                    "score": node.get("value", 10),
                    "title": node.get("title", ""),
                },
                "classes": f"node-{node_type.lower()} level-{level.lower()}"
            }
            
            if parent:
                node_data["data"]["parent"] = parent
            
            elements.append(node_data)
        
        # Add edges
        for edge in edges:
            weight = edge.get("weight", 1)
            if weight != weight:  # NaN check
                weight = 1
            edge_data = {
                "data": {
                    "id": f"{edge['source']}-{edge['target']}",
                    "source": edge["source"],
                    "target": edge["target"],
                    "weight": weight,
                    "depType": edge.get("dependency_type", "default"),
                    "title": edge.get("title", "")
                },
                "classes": f"edge-{edge.get('dependency_type', 'default').replace('_', '-')}"
            }
            elements.append(edge_data)
        
        # Build HTML
        html = [
            f'<h4 style="margin-bottom: 10px;">{title}</h4>',
            f'<div class="cytoscape-wrapper">',
            f'  <div class="cytoscape-controls">',
            f'    <button onclick="cy{graph_id.replace("-", "_")}.fit()" title="Fit to view">🔍 Fit</button>',
            f'    <button onclick="cy{graph_id.replace("-", "_")}.reset()" title="Reset view">↺ Reset</button>',
            f'    <button onclick="exportCytoscapeGraph(\'{graph_id}\')" title="Export as PNG">📷 Export</button>',
            f'  </div>',
            f'  <div id="{graph_id}" class="cytoscape-container"></div>',
            f'  <div class="cytoscape-legend" style="font-size:0.75rem;padding:10px;">',
            f'    <div style="display:flex;gap:15px;flex-wrap:wrap;">',
            f'      <span>🔵 Application (Ellipse)</span>',
            f'      <span>💎 Library (Diamond)</span>',
            f'      <span>🟣 Broker (Hexagon)</span>',
            f'      <span>🟩 Node (Box)</span>',
            f'      <span>⭐ Topic (Star)</span>',
            f'    </div>',
            f'    <div style="margin-top:8px;display:flex;gap:10px;flex-wrap:wrap;">',
            f'      <span style="color:#e74c3c">● Critical</span>',
            f'      <span style="color:#e67e22">● High</span>',
            f'      <span style="color:#f1c40f">● Medium</span>',
            f'      <span style="color:#2ecc71">● Low</span>',
            f'    </div>',
            f'  </div>',
            f'</div>',
        ]
        self.sections.append('\n'.join(html))
        
        # Build JavaScript
        elements_json = json.dumps(elements)
        var_name = f"cy{graph_id.replace('-', '_')}"
        
        script = f"""
        <script>
        (function() {{
            const elements = {elements_json};
            const containerId = '{graph_id}';
            
            // Color mappings
            const typeColors = {{
                'Application': '#3498db',
                'Broker': '#9b59b6',
                'Node': '#2ecc71',
                'Topic': '#f1c40f',
                'Library': '#1abc9c'
            }};
            
            const levelColors = {{
                'CRITICAL': '#e74c3c',
                'HIGH': '#e67e22',
                'MEDIUM': '#f1c40f',
                'LOW': '#2ecc71',
                'MINIMAL': '#95a5a6'
            }};
            
            const typeShapes = {{
                'Application': 'ellipse',
                'Library': 'diamond',
                'Broker': 'hexagon',
                'Node': 'rectangle',
                'Topic': 'star'
            }};
            
            // Initialize Cytoscape
            const {var_name} = cytoscape({{
                container: document.getElementById(containerId),
                elements: elements,
                style: [
                    // Compound nodes (layer groups)
                    {{
                        selector: '.compound',
                        style: {{
                            'shape': 'roundrectangle',
                            'background-color': '#f8f9fa',
                            'background-opacity': 0.95,
                            'border-width': 2,
                            'border-color': '#bdc3c7',
                            'border-style': 'dashed',
                            'padding': '30px',
                            'text-valign': 'top',
                            'text-halign': 'center',
                            'font-size': '14px',
                            'font-weight': 'bold',
                            'color': '#2c3e50',
                            'text-margin-y': 10,
                            'label': 'data(label)'
                        }}
                    }},
                    // Regular nodes
                    {{
                        selector: 'node:childless',
                        style: {{
                            'label': function(ele) {{
                                const label = ele.data('label') || ele.data('id');
                                return label.split('\\n')[0];
                            }},
                            'width': function(ele) {{
                                const score = ele.data('score') || 10;
                                return Math.max(20, Math.min(50, score));
                            }},
                            'height': function(ele) {{
                                const score = ele.data('score') || 10;
                                return Math.max(20, Math.min(50, score));
                            }},
                            'background-color': function(ele) {{
                                return typeColors[ele.data('nodeType')] || '#999';
                            }},
                            'shape': function(ele) {{
                                return typeShapes[ele.data('nodeType')] || 'ellipse';
                            }},
                            'border-width': function(ele) {{
                                return (ele.data('spof') || ele.data('is_spof')) ? 5 : 3;
                            }},
                            'border-style': function(ele) {{
                                return (ele.data('spof') || ele.data('is_spof')) ? 'dashed' : 'solid';
                            }},
                            'border-color': function(ele) {{
                                return levelColors[ele.data('level')] || '#95a5a6';
                            }},
                            'font-size': '10px',
                            'text-wrap': 'ellipsis',
                            'text-max-width': '60px',
                            'text-valign': 'bottom',
                            'text-margin-y': 5,
                            'color': '#2c3e50'
                        }}
                    }},
                    // Edges
                    {{
                        selector: 'edge',
                        style: {{
                            'width': function(ele) {{
                                const w = ele.data('weight') || 1;
                                return Math.max(1, Math.min(6, Math.log(w + 1) + 1));
                            }},
                            'line-color': '#7f8c8d',
                            'target-arrow-color': '#7f8c8d',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'opacity': 0.7
                        }}
                    }},
                    // Edge type specific styles
                    {{
                        selector: '.edge-app-to-app',
                        style: {{
                            'line-color': '#3498db',
                            'target-arrow-color': '#3498db'
                        }}
                    }},
                    {{
                        selector: '.edge-node-to-node',
                        style: {{
                            'line-color': '#2ecc71',
                            'target-arrow-color': '#2ecc71'
                        }}
                    }},
                    {{
                        selector: '.edge-app-to-broker',
                        style: {{
                            'line-color': '#9b59b6',
                            'target-arrow-color': '#9b59b6',
                            'line-style': 'dashed'
                        }}
                    }},
                    {{
                        selector: '.edge-node-to-broker',
                        style: {{
                            'line-color': '#9b59b6',
                            'target-arrow-color': '#9b59b6',
                            'line-style': 'dotted'
                        }}
                    }},
                    {{
                        selector: '.edge-app-to-lib',
                        style: {{
                            'line-color': '#e67e22',
                            'target-arrow-color': '#e67e22',
                            'line-style': 'dashed'
                        }}
                    }},
                    {{
                        selector: '.edge-broker-to-broker',
                        style: {{
                            'line-color': '#1abc9c',
                            'target-arrow-color': '#1abc9c',
                            'line-style': 'dotted'
                        }}
                    }},
                    // Raw structural edge styles
                    {{
                        selector: '.edge-publishes-to',
                        style: {{
                            'line-color': '#e74c3c',
                            'target-arrow-color': '#e74c3c',
                            'line-style': 'solid',
                            'opacity': 0.8
                        }}
                    }},
                    {{
                        selector: '.edge-subscribes-to',
                        style: {{
                            'line-color': '#27ae60',
                            'target-arrow-color': '#27ae60',
                            'line-style': 'solid',
                            'opacity': 0.8
                        }}
                    }},
                    {{
                        selector: '.edge-uses',
                        style: {{
                            'line-color': '#8e44ad',
                            'target-arrow-color': '#8e44ad',
                            'line-style': 'dashed',
                            'opacity': 0.8
                        }}
                    }},
                    {{
                        selector: '.edge-runs-on',
                        style: {{
                            'line-color': '#f39c12',
                            'target-arrow-color': '#f39c12',
                            'line-style': 'dotted',
                            'opacity': 0.6
                        }}
                    }},
                    {{
                        selector: '.edge-routes',
                        style: {{
                            'line-color': '#1abc9c',
                            'target-arrow-color': '#1abc9c',
                            'line-style': 'solid',
                            'opacity': 0.8
                        }}
                    }},
                    {{
                        selector: '.edge-connects-to',
                        style: {{
                            'line-color': '#34495e',
                            'target-arrow-color': '#34495e',
                            'line-style': 'solid',
                            'opacity': 0.6
                        }}
                    }},
                    // Hover states
                    {{
                        selector: 'node:childless:selected',
                        style: {{
                            'border-width': 5,
                            'border-color': '#2c3e50',
                            'overlay-opacity': 0.2
                        }}
                    }},
                    {{
                        selector: 'edge:selected',
                        style: {{
                            'width': 4,
                            'opacity': 1
                        }}
                    }}
                ],
                layout: {{
                    name: 'cose-bilkent',
                    animate: false,
                    randomize: false,
                    nodeDimensionsIncludeLabels: true,
                    idealEdgeLength: 100,
                    nodeRepulsion: 4500,
                    nestingFactor: 0.1,
                    gravity: 0.25,
                    numIter: 2500,
                    tile: true,
                    tilingPaddingVertical: 20,
                    tilingPaddingHorizontal: 20,
                    gravityRangeCompound: 1.5,
                    gravityCompound: 1.0,
                    gravityRange: 3.8
                }},
                wheelSensitivity: 0.3,
                minZoom: 0.2,
                maxZoom: 3
            }});
            
            // Expose globally for controls
            window.{var_name} = {var_name};
            
            // Tooltip on hover
            {var_name}.on('mouseover', 'node:childless', function(evt) {{
                const node = evt.target;
                const title = node.data('title');
                if (title) {{
                    node.tippy = tippy(node.popperRef(), {{
                        content: title.replace(/<br>/g, '\\n'),
                        trigger: 'manual',
                        placement: 'top',
                        hideOnClick: false,
                        sticky: true
                    }});
                }}
            }});
            
            // Highlight connected edges on node select
            {var_name}.on('select', 'node:childless', function(evt) {{
                const node = evt.target;
                const connectedEdges = node.connectedEdges();
                connectedEdges.style('opacity', 1);
                connectedEdges.style('width', 4);
            }});
            
            {var_name}.on('unselect', 'node:childless', function(evt) {{
                const node = evt.target;
                const connectedEdges = node.connectedEdges();
                connectedEdges.style('opacity', 0.7);
                connectedEdges.style('width', function(ele) {{
                    const w = ele.data('weight') || 1;
                    return Math.max(1, Math.min(6, Math.log(w + 1) + 1));
                }});
            }});
            
            // Fit to viewport after initial layout
            {var_name}.ready(function() {{
                {var_name}.fit(undefined, 50);
            }});
        }})();
        
        // Toggle layer visibility
        function toggleLayers(graphId) {{
            const cy = window['cy' + graphId.replace(/-/g, '_')];
            const compounds = cy.$('.compound');
            if (compounds.length === 0) return;
            
            const firstCompound = compounds[0];
            const isVisible = firstCompound.style('display') !== 'none';
            
            compounds.forEach(function(node) {{
                node.style('display', isVisible ? 'none' : 'element');
            }});
        }}
        
        // Toggle edge type visibility
        function toggleEdgeType(graphId, edgeType) {{
            const cy = window['cy' + graphId.replace(/-/g, '_')];
            const button = document.getElementById(graphId + '-' + edgeType);
            const isActive = button.classList.contains('active');
            
            let selectors = [];
            if (edgeType === 'depends') {{
                selectors = ['.edge-app-to-app', '.edge-app-to-lib', '.edge-node-to-node', '.edge-default'];
            }} else if (edgeType === 'raw') {{
                selectors = ['.edge-app-to-broker', '.edge-node-to-broker', '.edge-broker-to-broker'];
            }}
            
            selectors.forEach(function(sel) {{
                const edges = cy.$(sel);
                edges.forEach(function(edge) {{
                    edge.style('display', isActive ? 'none' : 'element');
                }});
            }});
            
            if (isActive) {{
                button.classList.remove('active');
            }} else {{
                button.classList.add('active');
            }}
        }}

        // Export Cytoscape graph
        function exportCytoscapeGraph(graphId) {{
            const cy = window['cy' + graphId.replace(/-/g, '_')];
            try {{
                const png = cy.png({{ output: 'blob', bg: 'white', full: true, scale: 2 }});
                const link = document.createElement('a');
                link.href = URL.createObjectURL(png);
                link.download = graphId + '_network.png';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }} catch (err) {{
                console.error('Export failed:', err);
                alert('Failed to export graph: ' + err.message);
            }}
        }}
        </script>
        """
        self.scripts.append(script)
    
    def add_explanation_section(self, explanation: Dict[str, Any]) -> None:
        """Add the human-readable explanation section with stakeholder tabs."""
        if not explanation:
            return

        html = [
            '<div class="explanation-container">',
            '  <div class="stakeholder-tabs">',
            '    <button class="tab-btn active" onclick="filterStakeholder(\'ALL\', this)">All Views</button>',
            '    <button class="tab-btn" onclick="filterStakeholder(\'DevOps\', this)">Availability (DevOps)</button>',
            '    <button class="tab-btn" onclick="filterStakeholder(\'SRE\', this)">Reliability (SRE)</button>',
            '    <button class="tab-btn" onclick="filterStakeholder(\'Architect\', this)">Maintainability (Architect)</button>',
            '    <button class="tab-btn" onclick="filterStakeholder(\'Security\', this)">Vulnerability (Security)</button>',
            '  </div>',
            '  <div class="component-card-grid">'
        ]

        for comp in explanation.get("component_explanations", []):
            severity = comp.get("overall_level", "MINIMAL").lower()
            stakeholders = comp.get("stakeholders", [])
            roles_attr = ",".join(stakeholders)
            
            card_html = [
                f'<div class="component-card {severity}" data-stakeholders="{roles_attr}">',
                '  <div class="card-header">',
                f'    <div class="card-title">{comp["component_id"]}</div>',
                f'    <div class="card-pattern">{comp["pattern"]}</div>',
                '  </div>',
                f'  <div class="card-one-line">{comp["one_line_summary"]}</div>',
                f'  <div class="card-risk">{comp["top_risk_description"]}</div>',
                '  <div class="dimension-grid">'
            ]
            
            for dim in comp.get("dimension_explanations", []):
                dim_severity = dim.get("level", "MINIMAL").lower()
                card_html.extend([
                    '    <div class="dim-box">',
                    '      <div class="dim-header">',
                    f'        <span>{dim["dimension"]}</span>',
                    f'        <span class="badge badge-{dim_severity}">{dim["score"]:.3f}</span>',
                    '      </div>',
                    f'      <div class="dim-driver">Driven by: {dim["driving_metric"]}</div>',
                    f'      <div class="dim-meaning">{dim["plain_meaning"]}</div>',
                    '    </div>'
                ])
            
            card_html.append('  </div>') # end dimension-grid
            
            # Remediation
            card_html.append('  <div class="remediation-box">')
            card_html.append('    <div class="rem-header">Recommended Actions</div>')
            for step in comp.get("remediation_steps", []):
                card_html.append(f'    <div class="rem-step"><div class="rem-dot"></div>{step["action"]}</div>')
            card_html.append('  </div>') # end remediation-box
            
            card_html.append('</div>') # end component-card
            html.append("".join(card_html))

        html.append('  </div>') # end component-card-grid
        html.append('</div>') # end explanation-container
        
        self.sections.append("".join(html))
        
        # Add the filtering script if not already added
        if not any("filterStakeholder" in s for s in self.scripts):
            self.scripts.append("""
            <script>
            function filterStakeholder(role, btn) {
                // Update active button
                const container = btn.closest('.explanation-container');
                container.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // Filter cards
                const grid = container.querySelector('.component-card-grid');
                const cards = grid.querySelectorAll('.component-card');
                cards.forEach(card => {
                    if (role === 'ALL') {
                        card.style.display = 'block';
                    } else {
                        const roles = card.getAttribute('data-stakeholders').split(',');
                        if (roles.includes(role)) {
                            card.style.display = 'block';
                        } else {
                            card.style.display = 'none';
                        }
                    }
                });
            }
            </script>
            """)

    def add_antipattern_catalog(self, smells: List[Dict[str, Any]]) -> None:
        """
        Add an anti-pattern catalog section with detailed analysis and recommendations.
        """
        if not smells:
            self.sections.append('<p style="color: var(--text-muted); font-style: italic;">No anti-patterns detected in this layer.</p>')
            return
            
        html = ['<div class="antipattern-list">']
        for s in smells:
            severity = s.get("severity", "MEDIUM").lower()
            html.append(
                f'<div class="antipattern-card {severity}">'
                f'  <div class="antipattern-header">'
                f'    <span class="badge badge-{severity}">{s.get("severity")}</span>'
                f'    <span class="antipattern-name">{s.get("pattern_name")} [{s.get("pattern_id")}]</span>'
                f'  </div>'
                f'  <div class="antipattern-meta">'
                f'    <span><strong>Dimension:</strong> {s.get("rmav_dimension")}</span> • '
                f'    <span><strong>Components:</strong> {", ".join(s.get("component_ids", []))}</span>'
                f'  </div>'
                f'  <div class="antipattern-body">'
                f'    <div class="antipattern-col">'
                f'      <h5>Risk Analysis</h5>'
                f'      <p>{s.get("risk")}</p>'
                f'    </div>'
                f'    <div class="antipattern-col">'
                f'      <h5>Recommendation</h5>'
                f'      <p>{s.get("recommendation")}</p>'
                f'    </div>'
                f'  </div>'
                f'</div>'
            )
        html.append('</div>')
        self.sections.append(''.join(html))
    
    def add_matrix_view(
        self,
        matrix_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        title: str = "Dependency Matrix",
        rcm_order: Optional[List[str]] = None
    ) -> None:
        """
        Add an interactive adjacency matrix visualization for dense dependency analysis.
        
        Args:
            matrix_id: Unique identifier for the matrix container
            nodes: List of node dictionaries with id, label, type, level
            edges: List of edge dictionaries with source, target, weight
            title: Title displayed above the matrix
        """
        if not nodes:
            return
        
        if rcm_order:
            # Sort nodes by their position in rcm_order
            rcm_index = {nid: i for i, nid in enumerate(rcm_order)}
            sorted_nodes = sorted(nodes, key=lambda n: rcm_index.get(n["id"], 999999))
        else:
            # Fallback: Sort by type then by level (criticality)
            type_order = {"Application": 0, "Library": 1, "Topic": 2, "Broker": 3, "Node": 4}
            level_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "MINIMAL": 4}
            
            sorted_nodes = sorted(nodes, key=lambda n: (
                type_order.get(n.get("type", "Application"), 5),
                level_order.get(n.get("level", "MINIMAL"), 5),
                n.get("id", "")
            ))
        
        node_ids = [n["id"] for n in sorted_nodes]
        node_index = {nid: i for i, nid in enumerate(node_ids)}
        
        # Build matrix data
        matrix_data = []
        edge_lookup = {}
        for edge in edges:
            key = (edge["source"], edge["target"])
            weight = edge.get("weight", 1)
            if weight != weight:  # NaN check
                weight = 1
            edge_lookup[key] = {
                "weight": weight,
                "type": edge.get("dependency_type", "default"),
                "relation": edge.get("relation_type", "DEPENDS_ON")
            }
        
        for i, source_id in enumerate(node_ids):
            for j, target_id in enumerate(node_ids):
                edge_info = edge_lookup.get((source_id, target_id))
                if edge_info:
                    matrix_data.append({
                        "row": i,
                        "col": j,
                        "source": source_id,
                        "target": target_id,
                        "weight": edge_info["weight"],
                        "type": edge_info["type"],
                        "relation": edge_info["relation"]
                    })
        
        node_meta = []
        for i, n in enumerate(sorted_nodes):
            node_meta.append({
                "id": n["id"],
                "label": n.get("label", n["id"]).split("\n")[0],
                "type": n.get("type", "Application"),
                "level": n.get("level", "MINIMAL"),
                "topo": i
            })
        
        # Calculate cell size based on number of nodes
        num_nodes = len(node_ids)
        cell_size = max(12, min(25, 600 // max(num_nodes, 1)))
        matrix_size = cell_size * num_nodes
        label_width = 120
        total_width = matrix_size + label_width + 50
        total_height = matrix_size + label_width + 80
        
        # Build HTML
        html = [
            f'<h4 style="margin-bottom: 10px;">{title}</h4>',
            f'<div class="matrix-wrapper" style="overflow-x: auto;">',
            f'  <div class="matrix-controls" style="margin-bottom: 10px;">',
            f'    <span style="margin-right: 10px; font-size: 0.85rem;">Sort by:</span>',
            f'    <button onclick="sortMatrix(\'{matrix_id}\', \'topological\')" class="active" style="padding: 4px 10px; margin-right: 5px; border-radius: 4px; border: 1px solid #ddd; cursor: pointer; font-size: 0.8rem;">Topological (RCM)</button>',
            f'    <button onclick="sortMatrix(\'{matrix_id}\', \'type\')" style="padding: 4px 10px; margin-right: 5px; border-radius: 4px; border: 1px solid #ddd; cursor: pointer; font-size: 0.8rem;">Type</button>',
            f'    <button onclick="sortMatrix(\'{matrix_id}\', \'level\')" style="padding: 4px 10px; margin-right: 5px; border-radius: 4px; border: 1px solid #ddd; cursor: pointer; font-size: 0.8rem;">Criticality</button>',
            f'    <button onclick="sortMatrix(\'{matrix_id}\', \'name\')" style="padding: 4px 10px; border-radius: 4px; border: 1px solid #ddd; cursor: pointer; font-size: 0.8rem; margin-right: 5px;">Name</button>',
            f'    <button onclick="exportMatrixView(\'{matrix_id}\')" style="padding: 4px 10px; border-radius: 4px; border: 1px solid #ddd; cursor: pointer; font-size: 0.8rem;">📷 Export</button>',
            f'  </div>',
            f'  <div id="{matrix_id}" style="position: relative;"></div>',
            f'  <div class="matrix-legend" style="display: flex; align-items: center; gap: 20px; margin-top: 15px; font-size: 0.8rem;">',
            f'    <span>Dependency Weight:</span>',
            f'    <div style="display: flex; align-items: center;">',
            f'      <span style="margin-right: 5px;">Low</span>',
            f'      <div style="width: 120px; height: 12px; background: linear-gradient(to right, #f0f4ff, #3498db, #1a5276); border-radius: 2px;"></div>',
            f'      <span style="margin-left: 5px;">High</span>',
            f'    </div>',
            f'  </div>',
            f'</div>',
        ]
        self.sections.append('\n'.join(html))
        
        # Build JavaScript
        matrix_json = json.dumps(matrix_data)
        nodes_json = json.dumps(node_meta)
        var_name = f"matrix{matrix_id.replace('-', '_')}"
        
        script = f"""
        <script>
        (function() {{
            const matrixData = {matrix_json};
            const nodes = {nodes_json};
            const containerId = '{matrix_id}';
            const cellSize = {cell_size};
            const labelWidth = {label_width};
            const numNodes = nodes.length;
            
            // Color scales
            const levelColors = {{
                'CRITICAL': '#e74c3c',
                'HIGH': '#e67e22',
                'MEDIUM': '#f1c40f',
                'LOW': '#2ecc71',
                'MINIMAL': '#95a5a6'
            }};
            
            const typeColors = {{
                'Application': '#3498db',
                'Broker': '#9b59b6',
                'Node': '#2ecc71',
                'Topic': '#f1c40f',
                'Library': '#1abc9c'
            }};
            
            // Find max weight for color scale
            const maxWeight = Math.max(...matrixData.map(d => d.weight), 1);
            
            // Create SVG
            const container = document.getElementById(containerId);
            const width = cellSize * numNodes + labelWidth + 20;
            const height = cellSize * numNodes + labelWidth + 20;
            
            const svg = d3.select('#' + containerId)
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            const g = svg.append('g')
                .attr('transform', `translate(${{labelWidth}}, ${{labelWidth}})`);
            
            // Tooltip
            const tooltip = d3.select('body').append('div')
                .attr('class', 'matrix-tooltip')
                .style('position', 'absolute')
                .style('background', 'rgba(0,0,0,0.85)')
                .style('color', 'white')
                .style('padding', '8px 12px')
                .style('border-radius', '4px')
                .style('font-size', '12px')
                .style('pointer-events', 'none')
                .style('opacity', 0)
                .style('z-index', 1000);
            
            // Color scale for cells
            const colorScale = d3.scaleSequential()
                .domain([0, maxWeight])
                .interpolator(d3.interpolateBlues);
            
            // Draw grid background
            g.append('rect')
                .attr('width', cellSize * numNodes)
                .attr('height', cellSize * numNodes)
                .attr('fill', '#f8f9fa')
                .attr('stroke', '#ddd');
            
            // Draw cells
            const cells = g.selectAll('.cell')
                .data(matrixData)
                .enter().append('rect')
                .attr('class', 'cell')
                .attr('x', d => d.col * cellSize)
                .attr('y', d => d.row * cellSize)
                .attr('width', cellSize - 1)
                .attr('height', cellSize - 1)
                .attr('fill', d => colorScale(d.weight))
                .attr('stroke', '#fff')
                .attr('stroke-width', 0.5)
                .style('cursor', 'pointer')
                .on('mouseover', function(event, d) {{
                    d3.select(this).attr('stroke', '#2c3e50').attr('stroke-width', 2);
                    tooltip.transition().duration(100).style('opacity', 1);
                    tooltip.html(`<strong>${{d.source}}</strong> → <strong>${{d.target}}</strong><br/>
                                  Weight: ${{d.weight.toFixed(3)}}<br/>
                                  Type: ${{d.relation}}`)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 10) + 'px');
                }})
                .on('mouseout', function() {{
                    d3.select(this).attr('stroke', '#fff').attr('stroke-width', 0.5);
                    tooltip.transition().duration(100).style('opacity', 0);
                }})
                .on('click', function(event, d) {{
                    // Highlight row and column
                    cells.attr('opacity', function(cell) {{
                        return (cell.row === d.row || cell.col === d.col) ? 1 : 0.3;
                    }});
                    rowLabels.attr('opacity', function(n, i) {{
                        return i === d.row ? 1 : 0.3;
                    }});
                    colLabels.attr('opacity', function(n, i) {{
                        return i === d.col ? 1 : 0.3;
                    }});
                }});
            
            // Row labels (left side)
            const rowLabels = svg.append('g')
                .attr('transform', `translate(${{labelWidth - 5}}, ${{labelWidth}})`)
                .selectAll('.row-label')
                .data(nodes)
                .enter().append('text')
                .attr('class', 'row-label')
                .attr('x', 0)
                .attr('y', (d, i) => i * cellSize + cellSize / 2 + 4)
                .attr('text-anchor', 'end')
                .attr('font-size', Math.min(10, cellSize - 2) + 'px')
                .attr('fill', d => levelColors[d.level] || '#333')
                .text(d => d.label.length > 15 ? d.label.substring(0, 12) + '...' : d.label);
            
            // Column labels (top)
            const colLabels = svg.append('g')
                .attr('transform', `translate(${{labelWidth}}, ${{labelWidth - 5}})`)
                .selectAll('.col-label')
                .data(nodes)
                .enter().append('text')
                .attr('class', 'col-label')
                .attr('x', (d, i) => i * cellSize + cellSize / 2)
                .attr('y', 0)
                .attr('text-anchor', 'start')
                .attr('font-size', Math.min(10, cellSize - 2) + 'px')
                .attr('fill', d => levelColors[d.level] || '#333')
                .attr('transform', (d, i) => `rotate(-45, ${{i * cellSize + cellSize / 2}}, 0)`)
                .text(d => d.label.length > 15 ? d.label.substring(0, 12) + '...' : d.label);
            
            // Type separators
            let currentType = '';
            nodes.forEach((node, i) => {{
                if (node.type !== currentType && i > 0) {{
                    g.append('line')
                        .attr('x1', 0)
                        .attr('x2', cellSize * numNodes)
                        .attr('y1', i * cellSize)
                        .attr('y2', i * cellSize)
                        .attr('stroke', '#7f8c8d')
                        .attr('stroke-width', 1.5);
                    g.append('line')
                        .attr('x1', i * cellSize)
                        .attr('x2', i * cellSize)
                        .attr('y1', 0)
                        .attr('y2', cellSize * numNodes)
                        .attr('stroke', '#7f8c8d')
                        .attr('stroke-width', 1.5);
                }}
                currentType = node.type;
            }});
            
            // Double-click to reset
            svg.on('dblclick', function() {{
                cells.attr('opacity', 1);
                rowLabels.attr('opacity', 1);
                colLabels.attr('opacity', 1);
            }});
            
            // Store for sorting
            window['{var_name}'] = {{ svg, g, cells, rowLabels, colLabels, nodes, matrixData, cellSize, labelWidth }};
        }})();
        
        // Sort function
        function sortMatrix(matrixId, sortBy) {{
            const varName = 'matrix' + matrixId.replace(/-/g, '_');
            const data = window[varName];
            if (!data) return;
            
            const typeOrder = {{'Application': 0, 'Library': 1, 'Topic': 2, 'Broker': 3, 'Node': 4}};
            const levelOrder = {{'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'MINIMAL': 4}};
            
            let sortedNodes;
            if (sortBy === 'topological') {{
                sortedNodes = [...data.nodes].sort((a, b) => a.topo - b.topo);
            }} else if (sortBy === 'type') {{
                sortedNodes = [...data.nodes].sort((a, b) => 
                    (typeOrder[a.type] || 5) - (typeOrder[b.type] || 5) || a.id.localeCompare(b.id));
            }} else if (sortBy === 'level') {{
                sortedNodes = [...data.nodes].sort((a, b) => 
                    (levelOrder[a.level] || 5) - (levelOrder[b.level] || 5) || a.id.localeCompare(b.id));
            }} else {{
                sortedNodes = [...data.nodes].sort((a, b) => a.id.localeCompare(b.id));
            }}
            
            // Create new index mapping
            const newIndex = {{}};
            sortedNodes.forEach((n, i) => newIndex[n.id] = i);
            
            // Update matrix positions with animation
            data.cells.transition().duration(500)
                .attr('x', d => newIndex[d.target] * data.cellSize)
                .attr('y', d => newIndex[d.source] * data.cellSize);
            
            // Update labels
            data.rowLabels.data(sortedNodes)
                .transition().duration(500)
                .attr('y', (d, i) => i * data.cellSize + data.cellSize / 2 + 4)
                .text(d => d.label.length > 15 ? d.label.substring(0, 12) + '...' : d.label);
            
            data.colLabels.data(sortedNodes)
                .transition().duration(500)
                .attr('x', (d, i) => i * data.cellSize + data.cellSize / 2)
                .attr('transform', (d, i) => `rotate(-45, ${{i * data.cellSize + data.cellSize / 2}}, 0)`)
                .text(d => d.label.length > 15 ? d.label.substring(0, 12) + '...' : d.label);
            
            // Update button active state
            const container = document.getElementById(matrixId).parentElement;
            container.querySelectorAll('.matrix-controls button').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }}

        // Export Matrix View
        function exportMatrixView(matrixId) {{
            const svgElement = document.querySelector('#' + matrixId + ' svg');
            if (!svgElement) {{
                alert('Matrix SVG not found');
                return;
            }}
            
            // Get SVG data
            const serializer = new XMLSerializer();
            const svgString = serializer.serializeToString(svgElement);
            const svgData = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgString)));
            
            // Create canvas
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            const width = svgElement.getAttribute('width');
            const height = svgElement.getAttribute('height');
            
            canvas.width = width;
            canvas.height = height;
            
            // Draw to canvas
            const img = new Image();
            img.onload = function() {{
                // Fill white background
                context.fillStyle = '#ffffff';
                context.fillRect(0, 0, width, height);
                
                context.drawImage(img, 0, 0);
                
                // Export
                const link = document.createElement('a');
                link.download = matrixId + '_matrix.png';
                link.href = canvas.toDataURL('image/png');
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }};
            img.src = svgData;
        }}
        </script>
        """
        self.scripts.append(script)
    
    # Keep backward compatibility with the old method name
    def add_network_graph(self, graph_id: str, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], title: str = "Network Graph") -> None:
        """Add an interactive network graph (delegates to add_cytoscape_network)."""
        self.add_cytoscape_network(graph_id, nodes, edges, title, use_compound_nodes=True)
    
    def generate(self) -> str:
        """Generate the complete HTML document."""
        nav_html = ' '.join([f'<a href="#{n.anchor}">{n.label}</a>' for n in self.nav_links])
        return HTML_TEMPLATE.format(
            title=self.title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            nav_links=nav_html,
            content=''.join(self.sections),
            scripts=''.join(self.scripts)
        )
