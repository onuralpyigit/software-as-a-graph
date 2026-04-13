"""
Dashboard Generator Service

v3.1 changes (aligned with interactive prototype):
  - HTML_TEMPLATE redesigned: flat white, clean nav, no glassmorphism
  - add_interactive_table()   — client-side sort + filter (JS inline)
  - add_cascade_risk_panel()  — dual-bar QoS/topo + stat cards
  - add_hierarchy_tree()      — MIL-STD-498 CSS→CSCI→CSC→CSU tree
  - add_dim_rho_panel()       — per-dimension ρ bars + multi-seed chart
  - add_dependency_matrix()   — re-enabled
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .models import ChartOutput


# Component type → layer mapping for Cytoscape compound nodes
COMPONENT_LAYER_MAP = {
    "Application": "layer-app",
    "Library":     "layer-app",
    "Topic":       "layer-mw",
    "Broker":      "layer-mw",
    "Node":        "layer-infra",
}

LAYER_COMPOUNDS = {
    "layer-app":   {"label": "Application Layer",    "icon": "App",   "order": 0},
    "layer-mw":    {"label": "Middleware Layer",      "icon": "MW",    "order": 1},
    "layer-infra": {"label": "Infrastructure Layer",  "icon": "Infra", "order": 2},
}

# ─── HTML Template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{title}}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/layout-base@2.0.1/layout-base.js"></script>
  <script src="https://unpkg.com/cose-base@2.2.0/cose-base.js"></script>
  <script src="https://unpkg.com/cytoscape-cose-bilkent@4.1.0/cytoscape-cose-bilkent.js"></script>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --primary:   #534AB7;
      --success:   #3B6D11;
      --warning:   #854F0B;
      --danger:    #A32D2D;
      --info:      #185FA5;
      --bg:        #f8f9fa;
      --surface:   #ffffff;
      --border:    #e5e7eb;
      --text:      #111827;
      --muted:     #6b7280;
      --nav-bg:    #1e293b;
      --radius-md: 8px;
      --radius-lg: 12px;
      --font-sans: 'Inter', -apple-system, system-ui, sans-serif;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: var(--font-sans);
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
      line-height: 1.6;
    }}
    .db {{ padding: 1.5rem 2rem; max-width: 1400px; margin: 0 auto; }}
    /* ── Tabs ── */
    .tabs {{
      display: flex; gap: 4px; border-bottom: 0.5px solid var(--border);
      margin-bottom: 1.5rem; position: sticky; top: 0;
      background: var(--bg); z-index: 100; padding-top: 10px;
    }}
    .tab {{
      padding: 8px 16px; font-size: 13px; border: none; background: none;
      cursor: pointer; color: var(--muted); border-bottom: 2px solid transparent;
      margin-bottom: -1px; font-weight: 500; transition: all 0.2s;
    }}
    .tab:hover {{ color: var(--text); }}
    .tab.active {{ color: var(--primary); border-bottom-color: var(--primary); }}
    .sag-main {{
      padding-top: 20px;
    }}
    .sag-header {{
      padding: 24px 0; margin-bottom: 0; background: transparent; color: var(--text);
    }}
    .sag-header .meta {{
      font-size: 11px; opacity: 0.5; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;
    }}
    .sag-header h1 {{
      font-size: 24px; font-weight: 600; letter-spacing: -0.5px;
    }}
    /* ── Sections ── */
    .section {{
      background: var(--surface); border: 0.5px solid var(--border);
      border-radius: var(--radius-lg); padding: 28px 32px;
      margin-bottom: 24px; scroll-margin-top: 72px;
    }}
    .section-header {{
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 24px; padding-bottom: 16px;
      border-bottom: 0.5px solid var(--border);
    }}
    .section-header h2 {{
      font-size: 16px; font-weight: 600; color: var(--text);
    }}
    .subsection {{ margin: 20px 0 12px; }}
    .subsection h3 {{
      font-size: 13px; font-weight: 500; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.04em;
    }}
    /* ── KPI cards ── */
    .grid4 {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px; margin-bottom: 1.5rem;
    }}
    .kpi {{ background: var(--surface); border-radius: var(--radius-md); padding: 16px 20px; border: 0.5px solid var(--border); }}
    .kpi-label {{ font-size: 11px; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }}
    .kpi-val {{ font-size: 26px; font-weight: 600; color: var(--primary); }}
    .kpi-val.danger {{ color: var(--danger); }}
    .kpi-val.warning {{ color: var(--warning); }}
    .kpi-val.success {{ color: var(--success); }}
    .kpi-val.info {{ color: var(--info); }}
    /* ── Charts ── */
    .grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; margin-bottom: 1.5rem; }}
    .chart-card {{
      border: 0.5px solid var(--border); border-radius: var(--radius-md);
      padding: 20px;
    }}
    .chart-card h4 {{
      font-size: 12px; font-weight: 500; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 16px;
    }}
    .chart-container {{ position: relative; width: 100%; }}
    /* ── Tables ── */
    .table-container {{
      overflow-x: auto; border: 0.5px solid var(--border);
      border-radius: var(--radius-md); margin-bottom: 20px;
    }}
    .table-filter-row {{
      display: flex; gap: 8px; padding: 12px 16px;
      border-bottom: 0.5px solid var(--border); flex-wrap: wrap;
      align-items: center;
    }}
    .table-filter-row select, .table-filter-row input {{
      font-size: 12px; padding: 4px 10px;
      border: 0.5px solid var(--border); border-radius: var(--radius-md);
      background: var(--surface); color: var(--text); outline: none;
    }}
    .table-filter-row label {{
      font-size: 11px; color: var(--muted); font-weight: 500;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th {{
      text-align: left; padding: 8px 12px;
      font-size: 11px; font-weight: 500; color: var(--muted);
      border-bottom: 0.5px solid var(--border);
      background: #f9fafb; cursor: pointer; user-select: none;
      white-space: nowrap;
    }}
    th:hover {{ color: var(--text); }}
    th .sort-icon {{ margin-left: 4px; opacity: 0.4; }}
    td {{ padding: 7px 12px; border-bottom: 0.5px solid #f3f4f6; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f9fafb; }}
    /* ── Badges ── */
    .badge {{
      display: inline-block; padding: 2px 7px; border-radius: 4px;
      font-size: 11px; font-weight: 500;
    }}
    .badge-critical  {{ background: #FCEBEB; color: #791F1F; }}
    .badge-high      {{ background: #FAEEDA; color: #633806; }}
    .badge-medium    {{ background: #E6F1FB; color: #0C447C; }}
    .badge-low       {{ background: #EAF3DE; color: #27500A; }}
    .badge-minimal   {{ background: #F1EFE8; color: #444441; }}
    .badge-passed    {{ background: #EAF3DE; color: #27500A; }}
    .badge-failed    {{ background: #FCEBEB; color: #791F1F; }}
    .badge-spof      {{ background: #F4C0D1; color: #72243E; }}
    .badge-tag       {{ background: #EEEDFE; color: #3C3489; }}
    /* ── Cascade risk panel ── */
    .cascade-stat {{
      background: #f3f4f6; border-radius: var(--radius-md);
      padding: 14px 16px;
    }}
    .cascade-stat-val {{
      font-size: 22px; font-weight: 600; line-height: 1; margin-bottom: 4px;
    }}
    .cascade-stat-label {{
      font-size: 11px; color: var(--muted); font-weight: 500;
      text-transform: uppercase; letter-spacing: 0.04em;
    }}
    .cascade-note {{
      font-size: 12px; color: var(--muted); padding: 8px 14px;
      border-left: 2px solid var(--border); margin-bottom: 20px;
      line-height: 1.5;
    }}
    /* ── Hierarchy tree ── */
    .hier-node {{
      display: flex; align-items: center; gap: 8px;
      padding: 7px 12px; border-radius: var(--radius-md);
      border: 0.5px solid var(--border); margin-bottom: 4px;
      font-size: 13px;
    }}
    .hier-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
    .hier-badge {{
      display: inline-block; padding: 1px 6px; border-radius: 4px;
      font-size: 11px; font-weight: 500; margin-left: 6px;
    }}
    .hier-q {{ margin-left: auto; font-size: 12px; font-weight: 500; }}
    /* ── Metrics box ── */
    .metrics-box {{
      border: 0.5px solid var(--border); border-radius: var(--radius-md);
      padding: 16px 20px; margin-bottom: 20px;
    }}
    .metrics-box h4 {{
      font-size: 12px; font-weight: 500; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 14px;
    }}
    .metric-row {{
      display: flex; justify-content: space-between; align-items: center;
      padding: 5px 0; border-bottom: 0.5px solid #f3f4f6;
      font-size: 12px;
    }}
    .metric-row:last-child {{ border-bottom: none; }}
    .metric-name {{ color: var(--muted); }}
    .metric-value {{ font-weight: 500; }}
    .metric-value.pass {{ color: var(--success); }}
    .metric-value.fail {{ color: var(--danger); }}
    /* ── Anti-patterns ── */
    .antipattern-card {{
      border-left: 3px solid var(--border); padding: 14px 18px;
      margin-bottom: 12px; border-radius: 0 var(--radius-md) var(--radius-md) 0;
      background: #fafafa;
    }}
    .antipattern-card.critical {{ border-left-color: var(--danger); }}
    .antipattern-card.high     {{ border-left-color: var(--warning); }}
    .antipattern-card h4 {{ font-size: 13px; font-weight: 500; margin-bottom: 4px; }}
    .antipattern-card p  {{ font-size: 12px; color: var(--muted); }}
    /* ── Explanation cards ── */
    .explanation-card {{
      border: 0.5px solid var(--border); border-radius: var(--radius-md);
      padding: 16px 20px; margin-bottom: 12px;
    }}
    .explanation-card h4 {{ font-size: 13px; font-weight: 500; margin-bottom: 6px; }}
    .explanation-card p  {{ font-size: 12px; color: var(--muted); }}
    /* ── Network graph ── */
    .cy-container {{
      width: 100%; border: 0.5px solid var(--border);
      border-radius: var(--radius-md); background: #fafafa;
    }}
    .cy-legend {{
      display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px;
      font-size: 12px; color: var(--muted);
    }}
    .cy-legend-item {{ display: flex; align-items: center; gap: 5px; }}
    .cy-swatch {{
      width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
    }}
    /* ── Footer ── */
    .sag-footer {{
      text-align: center; padding: 40px; color: var(--muted);
      font-size: 12px; border-top: 0.5px solid var(--border); margin-top: 40px;
    }}
    @media (max-width: 768px) {{
      .grid2 {{ grid-template-columns: 1fr; }}
      .sag-main   {{ padding: 16px 16px 40px; }}
    }}
  </style>
</head>
<body>
  <div class="sag-main">
    <div class="sag-header">
      <div class="meta">Step 6 · Visualization · {timestamp}</div>
      <h1>{title}</h1>
    </div>
    <div class="db">
      {tabs_html}
      {content}
    </div>
  </div>
  <div class="sag-footer">
    <p>Software-as-a-Graph Methodology &nbsp;·&nbsp; Architectural Decision Support</p>
  </div>
  <script>
  function switchTab(id, btn) {{
    document.querySelectorAll('[id^=tab-]').forEach(el => el.style.display = 'none');
    var target = document.getElementById('tab-' + id);
    if (target) target.style.display = '';
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    if (btn) btn.classList.add('active');
    // Force chart resize if hidden tab had charts
    window.dispatchEvent(new Event('resize'));
  }}
  </script>
  {scripts}
</body>
</html>
"""


@dataclass
class NavLink:
    label: str
    anchor: str


class DashboardGenerator:
    """
    Assembles responsive HTML dashboards with interactive charts,
    sortable/filterable tables, Cytoscape.js network graphs, D3 dependency
    matrices, cascade risk panels, and MIL-STD-498 hierarchy trees.
    """

    def __init__(self, title: str):
        self.title = title
        self.sections: List[str] = []
        self.nav_links: List[NavLink] = []
        self.scripts: List[str] = []
        self.tabs: List[Dict[str, Any]] = []
        self._current_tab: Optional[Dict[str, Any]] = None
        self._table_counter = 0

    def add_tab(self, name: str, anchor_id: str = "") -> None:
        """Start a new tab group."""
        tid = anchor_id or name.lower().replace(" ", "-")
        new_tab = {
            "id": tid,
            "title": name,
            "sections": []
        }
        self.tabs.append(new_tab)
        self._current_tab = new_tab

    def end_tab(self) -> None:
        """Close the current tab."""
        self._current_tab = None

    def start_section(self, title: str, anchor_id: str = "") -> None:
        sid = anchor_id or title.lower().replace(" ", "-")
        html = (
            f'<div class="section" id="{sid}">'
            f'<div class="section-header"><h2>{title}</h2></div>'
        )
        if self._current_tab is not None:
            self._current_tab["sections"].append(html)
        else:
            # Fallback if no tab is open
            if not self.tabs:
                self.add_tab("Main")
            self.tabs[0]["sections"].append(html)

    def end_section(self) -> None:
        html = "</div>"
        if self._current_tab is not None:
            self._current_tab["sections"].append(html)
        else:
            self.tabs[0]["sections"].append(html)

    def add_subsection(self, title: str) -> None:
        html = f'<div class="subsection"><h3>{title}</h3></div>'
        if self._current_tab is not None:
            self._current_tab["sections"].append(html)
        else:
            self.tabs[0]["sections"].append(html)

    # ── KPI Cards ──────────────────────────────────────────────────────────

    def add_kpis(self, kpis: Dict[str, Any], styles: Dict[str, str] = None) -> None:
        styles = styles or {}
        html = ['<div class="grid4">']
        for label, value in kpis.items():
            style_class = styles.get(label, "info")
            html.append(
                f'<div class="kpi">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-val {style_class}">{value}</div>'
                f'</div>'
            )
        html.append("</div>")
        content = "".join(html)
        if self._current_tab:
            self._current_tab["sections"].append(content)
        else:
            self.tabs[0]["sections"].append(content)

    # ── Chart Grid ─────────────────────────────────────────────────────────

    def add_charts(self, charts: List[Any], grid_class: str = "grid2") -> None:
        valid = [c for c in charts if c is not None]
        if not valid:
            return
        html = [f'<div class="{grid_class}">']
        for chart in valid:
            html.append(f'<div class="chart-card">{chart}</div>')
        html.append("</div>")
        content = "".join(html)
        if self._current_tab:
            self._current_tab["sections"].append(content)
        else:
            self.tabs[0]["sections"].append(content)

    # ── Static Table ───────────────────────────────────────────────────────

    def add_table(
        self, headers: List[str], rows: List[List[Any]], title: str = ""
    ) -> None:
        """Static table — use add_interactive_table for sort/filter."""
        html = ['<div class="table-container">']
        if title:
            html.append(f'<div style="padding:12px 16px;font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase">{title}</div>')
        html.append("<table><thead><tr>")
        for h in headers:
            html.append(f"<th>{h}</th>")
        html.append("</tr></thead><tbody>")
        for row in rows:
            html.append("<tr>")
            for cell in row:
                cell_str = str(cell)
                lu = cell_str.upper()
                if lu in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"):
                    cell_html = f'<span class="badge badge-{lu.lower()}">{cell_str}</span>'
                elif lu in ("PASSED", "PASS", "✓"):
                    cell_html = '<span class="badge badge-passed">PASSED</span>'
                elif lu in ("FAILED", "FAIL", "✗"):
                    cell_html = '<span class="badge badge-failed">FAILED</span>'
                else:
                    cell_html = cell_str
                html.append(f"<td>{cell_html}</td>")
            html.append("</tr>")
        html.append("</tbody></table></div>")
        content = "".join(html)
        if self._current_tab:
            self._current_tab["sections"].append(content)
        else:
            self.tabs[0]["sections"].append(content)

    # ── Interactive Table (sort + filter) ──────────────────────────────────

    def add_interactive_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        title: str = "",
        type_col: Optional[int] = None,
        level_col: Optional[int] = None,
    ) -> None:
        """
        Table with client-side column-header sort and optional filter
        dropdowns for type and criticality level columns.
        """
        self._table_counter += 1
        tid = f"itbl_{self._table_counter}"

        type_vals = sorted({str(rows[i][type_col]) for i in range(len(rows))}) if type_col is not None else []
        level_vals = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]

        html = ['<div class="table-container">']
        html.append('<div class="table-filter-row">')
        if title:
            html.append(f'<span style="font-size:12px;font-weight:500;color:#374151;margin-right:8px">{title}</span>')
        if type_col is not None:
            html.append(
                f'<select id="{tid}_tf" onchange="sagFilterTable(\'{tid}\')">'
                f'<option value="">All types</option>'
                + "".join(f'<option value="{v}">{v}</option>' for v in type_vals)
                + "</select>"
            )
        if level_col is not None:
            html.append(
                f'<select id="{tid}_lf" onchange="sagFilterTable(\'{tid}\')">'
                f'<option value="">All levels</option>'
                + "".join(f'<option value="{v}">{v}</option>' for v in level_vals)
                + "</select>"
            )
        html.append(
            f'<input type="text" id="{tid}_search" placeholder="Search..." '
            f'oninput="sagFilterTable(\'{tid}\')" style="margin-left:auto;width:160px">'
        )
        html.append("</div>")

        html.append(f'<table id="{tid}"><thead><tr>')
        for i, h in enumerate(headers):
            html.append(
                f'<th onclick="sagSortTable(\'{tid}\',{i})">'
                f'{h} <span class="sort-icon">↕</span></th>'
            )
        html.append("</tr></thead><tbody>")
        for row in rows:
            html.append("<tr>")
            for cell in row:
                html.append(f"<td>{cell}</td>")
            html.append("</tr>")
        html.append(f"</tbody></table></div>")

        tf_idx = type_col if type_col is not None else -1
        lf_idx = level_col if level_col is not None else -1

        script = f"""
        <script>
        (function() {{
          window.sagTblMeta = window.sagTblMeta || {{}};
          window.sagTblMeta['{tid}'] = {{ typeCol: {tf_idx}, levelCol: {lf_idx} }};

          window.sagSortTable = window.sagSortTable || function(id, col) {{
            var tbl = document.getElementById(id);
            if (!tbl) return;
            var rows = Array.from(tbl.tBodies[0].rows);
            var asc = tbl.dataset.sortCol == col && tbl.dataset.sortDir == 'asc';
            rows.sort(function(a, b) {{
              var av = a.cells[col].textContent.trim();
              var bv = b.cells[col].textContent.trim();
              var an = parseFloat(av), bn = parseFloat(bv);
              if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
              return asc ? av.localeCompare(bv) : bv.localeCompare(av);
            }});
            rows.forEach(function(r) {{ tbl.tBodies[0].appendChild(r); }});
            tbl.dataset.sortCol = col;
            tbl.dataset.sortDir = asc ? 'desc' : 'asc';
            tbl.querySelectorAll('th .sort-icon').forEach(function(s,i) {{
              s.textContent = (i === col) ? (asc ? ' ↑' : ' ↓') : ' ↕';
            }});
          }};

          function cellText(cell) {{
            return (cell ? cell.textContent : '').trim();
          }}

          window.sagFilterTable = window.sagFilterTable || function(id) {{
            var tbl = document.getElementById(id);
            if (!tbl) return;
            var meta = (window.sagTblMeta || {{}})[id] || {{}};
            var search = (document.getElementById(id + '_search') || {{}}).value || '';
            var tf = (document.getElementById(id + '_tf') || {{}}).value || '';
            var lf = (document.getElementById(id + '_lf') || {{}}).value || '';
            search = search.toLowerCase();
            Array.from(tbl.tBodies[0].rows).forEach(function(row) {{
              var text = row.textContent.toLowerCase();
              var typeOk = !tf || (meta.typeCol >= 0 && cellText(row.cells[meta.typeCol]) === tf);
              var levelOk = !lf || (meta.levelCol >= 0 && cellText(row.cells[meta.levelCol]) === lf);
              var searchOk = !search || text.indexOf(search) >= 0;
              row.style.display = (typeOk && levelOk && searchOk) ? '' : 'none';
            }});
          }};
        }})();
        </script>
        """
        if self._current_tab:
            self._current_tab["sections"].append("".join(html))
        else:
            self.tabs[0]["sections"].append("".join(html))
        self.scripts.append(script)

    # ── Cascade Risk Panel ─────────────────────────────────────────────────

    def add_cascade_risk_panel(
        self,
        cascade_chart_html: Optional[str],
        qos_gini: float = 0.0,
        wilcoxon_p: float = 1.0,
        delta_rho: float = 0.0,
        note: str = "",
    ) -> None:
        """
        Cascade risk section: stat cards (Gini, p-value, Δρ) + dual-bar chart.
        """
        html = []
        if note:
            html.append(f'<div class="cascade-note">{note}</div>')

        html.append('<div class="grid4">')
        stats = [
            ("QoS Gini coefficient",     f"{qos_gini:.3f}",    "info"),
            ("Wilcoxon p-value",          f"{wilcoxon_p:.4f}",  "success" if wilcoxon_p < 0.05 else "warning"),
            ("Δρ (enrichment)", f"+{delta_rho:.3f}",  "success" if delta_rho > 0.03 else "warning"),
        ]
        for label, val, style in stats:
            html.append(
                f'<div class="kpi">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-val {style}">{val}</div>'
                f'</div>'
            )
        html.append("</div>")

        if cascade_chart_html:
            html.append(f'<div class="chart-card"><h4>Cascade risk distribution</h4>{cascade_chart_html}</div>')
        
        content = "".join(html)
        if self._current_tab:
            self._current_tab["sections"].append(content)
        else:
            self.tabs[0]["sections"].append(content)

    def add_top5_bars(self, components: List[Any]) -> None:
        """Add the Top 5 components bar list from prototype."""
        LEVEL_COLORS = {
            'CRITICAL': '#A32D2D', 'HIGH': '#854F0B', 'MEDIUM': '#185FA5',
            'LOW': '#3B6D11', 'MINIMAL': '#5F5E5A'
        }
        top5 = components[:5]
        html = ['<div class="chart-card"><h4>Top 5 components by Q(v)</h4>']
        for c in top5:
            q = getattr(c, 'overall', 0.0)
            level = getattr(c, 'level', 'MINIMAL')
            spof = getattr(c, 'spof', False)
            name = getattr(c, 'name', getattr(c, 'id', 'Unknown'))
            html.append(f"""
                <div style="margin-bottom:12px">
                  <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">
                    <span style="font-weight:600">{name} <span class="badge badge-{level.lower()}">{level}</span>{ '<span class="badge badge-spof" style="margin-left:4px">SPOF</span>' if spof else '' }</span>
                    <span style="font-weight:600">{q:.3f}</span>
                  </div>
                  <div style="height:8px;background:var(--bg);border-radius:4px;overflow:hidden">
                    <div style="height:100%;width:{q*100:.1f}%;background:{LEVEL_COLORS.get(level, '#ccc')};border-radius:4px;transition:width .3s"></div>
                  </div>
                </div>""")
        html.append('</div>')
        content = "".join(html)
        if self._current_tab:
            self._current_tab["sections"].append(content)
        else:
            self.tabs[0]["sections"].append(content)

    # ── Per-Dimension ρ Panel ──────────────────────────────────────────────

    def add_dim_rho_panel(
        self,
        dim_rho_html: str,
        seed_chart_html: Optional[str] = None,
    ) -> None:
        """
        Two-column panel: per-dim ρ progress bars (left) + multi-seed line (right).
        """
        content = (
            f'<div class="grid2">'
            f'<div class="chart-card"><h4>Per-dimension Spearman ρ</h4>{dim_rho_html}</div>'
            + (f'<div class="chart-card"><h4>Multi-seed stability</h4>{seed_chart_html}</div>' if seed_chart_html else "")
            + '</div>'
        )
        if self._current_tab:
            self._current_tab["sections"].append(content)
        else:
            self.tabs[0]["sections"].append(content)

    # ── MIL-STD-498 Hierarchy Tree ──────────────────────────────────────────

    def add_hierarchy_tree(self, tree: Dict[str, Any]) -> None:
        """Render vertical tree."""
        COLORS = {
            "CSS":  ("#EEEDFE", "#534AB7"),
            "CSCI": ("#E1F5EE", "#0F6E56"),
            "CSC":  ("#E6F1FB", "#185FA5"),
            "CSU":  ("#F1EFE8", "#5F5E5A"),
        }

        def _render(node: Dict[str, Any], depth: int) -> List[str]:
            parts = []
            level = node.get("level", "CSU")
            bg, accent = COLORS.get(level, COLORS["CSU"])
            q = node.get("q")
            cbci = node.get("cbci")
            indent = f"margin-left:{depth * 24}px"
            q_html = (
                f'<span class="hier-q" style="color:{accent}">'
                f'Q = {q:.3f}</span>'
            ) if q is not None else ""
            cbci_html = (
                f'<span class="hier-badge" '
                f'style="background:{bg};color:{accent}">CBCI: {cbci:.2f}</span>'
            ) if cbci is not None else ""
            fw = "500" if depth == 0 else "400"
            parts.append(
                f'<div class="hier-node" style="{indent};background:{bg}22">'
                f'  <div class="hier-dot" style="background:{accent}"></div>'
                f'  <span style="font-weight:{fw}">'
                f'    {node.get("label", node["id"])}'
                f'  </span>'
                f'  {cbci_html}'
                f'  {q_html}'
                f'</div>'
            )
            for child in node.get("children", []):
                parts.extend(_render(child, depth + 1))
            return parts

        content = "".join(_render(tree, 0))
        if self._current_tab:
            self._current_tab["sections"].append(content)
        else:
            self.tabs[0]["sections"].append(content)

    # ── Metrics Box ────────────────────────────────────────────────────────

    def add_metrics_box(
        self,
        metrics: Dict[str, Any],
        title: str = "Metrics",
        highlights: Dict[str, bool] = None,
    ) -> None:
        highlights = highlights or {}
        html = [f'<div class="metrics-box"><h4>{title}</h4>']
        for name, value in metrics.items():
            val_class = " pass" if highlights.get(name) else (" fail" if name in highlights else "")
            val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            html.append(
                f'<div class="metric-row">'
                f'<span class="metric-name">{name}</span>'
                f'<span class="metric-value{val_class}">{val_str}</span>'
                f'</div>'
            )
        html.append("</div>")
        content = "".join(html)
        if self._current_tab:
            self._current_tab["sections"].append(content)
        else:
            self.tabs[0]["sections"].append(content)

    # ── Explanation Cards ──────────────────────────────────────────────────

    def add_explanation_section(self, explanation: Dict[str, Any]) -> None:
        if not explanation:
            return
        html = []
        for comp_id, info in explanation.items():
            if isinstance(info, dict):
                title = info.get("title", comp_id)
                text = info.get("explanation", "")
                level = info.get("level", "MINIMAL").lower()
                html.append(
                    f'<div class="explanation-card">'
                    f'<h4>{title} <span class="badge badge-{level}">'
                    f'{level.upper()}</span></h4>'
                    f'<p>{text}</p>'
                    f'</div>'
                )
        if html:
            content = "".join(html)
            if self._current_tab:
                self._current_tab["sections"].append(content)
            else:
                self.tabs[0]["sections"].append(content)

    # ── Anti-Pattern Catalog ───────────────────────────────────────────────

    def add_antipattern_catalog(self, patterns: List[Dict[str, Any]]) -> None:
        if not patterns:
            content = '<p style="color:var(--muted);font-size:13px">No anti-patterns detected.</p>'
        else:
            html = []
            for p in patterns:
                sev = p.get("severity", "medium").lower()
                name = p.get("name", "Unknown pattern")
                desc = p.get("description", "")
                components = p.get("components", [])
                comp_badges = "".join(
                    f'<span class="badge badge-tag">{c}</span> ' for c in components[:5]
                )
                html.append(
                    f'<div class="antipattern-card {sev}">'
                    f'<h4>{name} <span class="badge badge-{sev}">{sev.upper()}</span></h4>'
                    f'<p>{desc}</p>'
                    + (f'<div style="margin-top:6px">{comp_badges}</div>' if comp_badges else "")
                    + f'</div>'
                )
            content = "".join(html)
        
        if self._current_tab:
            self._current_tab["sections"].append(content)
        else:
            self.tabs[0]["sections"].append(content)

    # ── Cytoscape.js Network Graph ─────────────────────────────────────────

    def add_cytoscape_network(
        self,
        graph_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        title: str = "Network Graph",
        use_compound_nodes: bool = True,
    ) -> None:
        elements = []
        if use_compound_nodes:
            for layer_id, layer_info in LAYER_COMPOUNDS.items():
                elements.append({
                    "data": {
                        "id": layer_id,
                        "label": layer_info["label"],
                        "isCompound": True,
                        "order": layer_info["order"],
                    },
                    "classes": "compound",
                })

        CRIT_HEX = {
            "CRITICAL": "#A32D2D", "HIGH": "#854F0B",
            "MEDIUM": "#185FA5",   "LOW": "#3B6D11",
            "MINIMAL": "#5F5E5A",
        }
        for node in nodes:
            node_type = node.get("type", "Application")
            parent = COMPONENT_LAYER_MAP.get(node_type) if use_compound_nodes else None
            level = node.get("level", "MINIMAL")
            nd: Dict[str, Any] = {
                "data": {
                    "id": node["id"],
                    "label": node.get("label", node["id"]),
                    "nodeType": node_type,
                    "level": level,
                    "score": node.get("value", 10),
                    "title": node.get("title", ""),
                    "color": CRIT_HEX.get(level, "#5F5E5A"),
                },
                "classes": f"node-{node_type.lower()} level-{level.lower()}",
            }
            if parent:
                nd["data"]["parent"] = parent
            elements.append(nd)

        for edge in edges:
            weight = edge.get("weight", 1.0)
            if weight != weight:
                weight = 1.0
            elements.append({
                "data": {
                    "id": f"e_{edge['source']}_{edge['target']}",
                    "source": edge["source"],
                    "target": edge["target"],
                    "weight": weight,
                    "depType": edge.get("dependency_type", "DEPENDS_ON"),
                },
            })

        cy_style = [
            {"selector": "node", "style": {
                "label": "data(label)", "font-size": "10px",
                "background-color": "data(color)",
                "width": "data(score)", "height": "data(score)",
                "color": "#fff", "text-valign": "center",
                "text-halign": "center", "text-wrap": "wrap",
                "text-max-width": "80px",
            }},
            {"selector": ".compound", "style": {
                "label": "data(label)", "font-size": "11px",
                "background-opacity": 0.06, "background-color": "#534AB7",
                "border-color": "#534AB7", "border-width": "0.5px",
                "color": "#534AB7", "text-valign": "top",
                "text-halign": "center",
            }},
            {"selector": "edge", "style": {
                "width": 1.5, "line-color": "#d1d5db",
                "target-arrow-color": "#d1d5db",
                "target-arrow-shape": "triangle",
                "curve-style": "bezier",
            }},
        ]

        elem_json = json.dumps(elements)
        style_json = json.dumps(cy_style)
        height = min(640, max(360, len(nodes) * 14))
        content = (
            f'<p style="font-size:13px;font-weight:500;margin-bottom:12px">{title}</p>'
            f'<div id="{graph_id}" class="cy-container" style="height:{height}px"></div>'
        )
        legend_items = "".join(
            f'<div class="cy-legend-item">'
            f'<div class="cy-swatch" style="background:{c}"></div>{lv.capitalize()}'
            f'</div>'
            for lv, c in [("critical", "#A32D2D"), ("high", "#854F0B"),
                          ("medium", "#185FA5"), ("low", "#3B6D11")]
        )
        content += f'<div class="cy-legend">{legend_items}</div>'
        
        if self._current_tab:
            self._current_tab["sections"].append(content)
        else:
            self.tabs[0]["sections"].append(content)

        script = f"""
        <script>
        (function() {{
          if (typeof cytoscape === 'undefined') return;
          var cy = cytoscape({{
            container: document.getElementById('{graph_id}'),
            elements: {elem_json},
            style: {style_json},
            layout: {{
              name: 'cose-bilkent',
              animate: false,
              nodeRepulsion: 6000,
              idealEdgeLength: 80,
              nodeDimensionsIncludeLabels: true,
            }},
          }});
          cy.on('tap', 'node', function(e) {{
            var d = e.target.data();
            if (d.title) alert(d.title.replace(/<[^>]*>/g, ''));
          }});
        }})();
        </script>
        """
        self.scripts.append(script)

    # backward-compat alias
    def add_network_graph(
        self, graph_id: str, nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]], title: str = "Network Graph"
    ) -> None:
        self.add_cytoscape_network(graph_id, nodes, edges, title, use_compound_nodes=True)

    # ── D3 Dependency Matrix ───────────────────────────────────────────────

    def add_dependency_matrix(
        self,
        matrix_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        title: str = "Dependency Matrix",
    ) -> None:
        """Adjacency matrix sorted by Q(v)."""
        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)
        content = (
            f'<p style="font-size:13px;font-weight:500;margin-bottom:12px">{title}</p>'
            f'<div id="{matrix_id}" style="overflow-x:auto"></div>'
        )
        if self._current_tab:
            self._current_tab["sections"].append(content)
        else:
            self.tabs[0]["sections"].append(content)
            
        script = f"""
        <script>
        (function() {{
          if (typeof d3 === 'undefined') return;
          var rawNodes = {nodes_json};
          var rawEdges = {edges_json};
          if (!rawNodes.length) return;

          rawNodes.sort(function(a, b) {{
            return (b.score || 0) - (a.score || 0);
          }});

          var n = rawNodes.length;
          var cellSize = Math.max(8, Math.min(20, Math.floor(560 / n)));
          var margin = {{top: 100, left: 100, right: 10, bottom: 10}};
          var w = n * cellSize + margin.left + margin.right;
          var h = n * cellSize + margin.top + margin.bottom;

          var nodeIndex = {{}};
          rawNodes.forEach(function(nd, i) {{ nodeIndex[nd.id] = i; }});

          var matrix = Array.from({{length: n}}, function() {{
            return new Float32Array(n);
          }});
          rawEdges.forEach(function(e) {{
            var si = nodeIndex[e.source], ti = nodeIndex[e.target];
            if (si != null && ti != null) matrix[si][ti] = e.weight || 1;
          }});

          var crit_colors = {{
            CRITICAL: '#A32D2D', HIGH: '#854F0B',
            MEDIUM: '#185FA5', LOW: '#3B6D11', MINIMAL: '#5F5E5A'
          }};

          var maxVal = 0;
          rawEdges.forEach(function(e) {{ if (e.weight > maxVal) maxVal = e.weight; }});
          maxVal = maxVal || 1;

          var svg = d3.select('#' + '{matrix_id}').append('svg')
            .attr('width', w).attr('height', h);
          var g = svg.append('g')
            .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

          rawNodes.forEach(function(nd, i) {{
            var clr = crit_colors[nd.level] || '#5F5E5A';
            g.append('text')
              .attr('x', -4).attr('y', i * cellSize + cellSize / 2)
              .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
              .attr('font-size', Math.max(8, cellSize - 2))
              .attr('fill', clr)
              .text(nd.label ? nd.label.substring(0, 14) : nd.id);
            g.append('text')
              .attr('x', i * cellSize + cellSize / 2).attr('y', -4)
              .attr('text-anchor', 'start').attr('dominant-baseline', 'middle')
              .attr('font-size', Math.max(8, cellSize - 2))
              .attr('fill', clr)
              .attr('transform', 'rotate(-45,' + (i * cellSize + cellSize/2) + ',-4)')
              .text(nd.label ? nd.label.substring(0, 12) : nd.id);
          }});

          for (var r = 0; r < n; r++) {{
            for (var c = 0; c < n; c++) {{
              var val = matrix[r][c];
              g.append('rect')
                .attr('x', c * cellSize).attr('y', r * cellSize)
                .attr('width', cellSize - 1).attr('height', cellSize - 1)
                .attr('rx', 1)
                .attr('fill', val > 0 ? '#534AB7' : '#f3f4f6')
                .attr('opacity', val > 0 ? 0.2 + 0.8 * (val / maxVal) : 1)
                .attr('stroke', '#e5e7eb').attr('stroke-width', 0.5);
            }}
          }}
        }})();
        </script>
        """
        self.scripts.append(script)

    # ── Generate ───────────────────────────────────────────────────────────

    def generate(self) -> str:
        if not self.tabs:
            self.add_tab("Overview")
        
        tabs_html = ['<div class="tabs">']
        content_html = []
        
        for i, tab in enumerate(self.tabs):
            active = " active" if i == 0 else ""
            display = "" if i == 0 else ' style="display:none"'
            tabs_html.append(
                f'<button class="tab{active}" onclick="switchTab(\'{tab["id"]}\', this)">'
                f'{tab["title"]}</button>'
            )
            content_html.append(
                f'<div id="tab-{tab["id"]}"{display}>'
                + "".join(tab["sections"])
                + '</div>'
            )
        tabs_html.append("</div>")

        return HTML_TEMPLATE.format(
            title=self.title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tabs_html="".join(tabs_html),
            content="".join(content_html),
            scripts="".join(self.scripts),
        )