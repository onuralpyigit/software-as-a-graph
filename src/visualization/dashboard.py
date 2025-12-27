"""
Dashboard Generator - Version 4.0

Comprehensive dashboard generation with Chart.js for visualizing
graph statistics, analysis results, and simulation metrics.

Features:
- Graph statistics overview
- Criticality distribution charts
- Validation metrics visualization
- Simulation results
- Interactive network graph
- Export capabilities

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..simulation import SimulationGraph, ComponentType
from .graph_renderer import COLORS, Layer


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    title: str = "Pub-Sub System Analysis Dashboard"
    theme: str = "dark"  # dark or light
    show_graph: bool = True
    show_statistics: bool = True
    show_criticality: bool = True
    show_validation: bool = True
    show_simulation: bool = True
    graph_height: str = "400px"


# =============================================================================
# Dashboard Generator
# =============================================================================

class DashboardGenerator:
    """
    Generates comprehensive HTML dashboards with Chart.js.
    
    Combines graph visualization, statistics, and analysis results
    into an interactive dashboard.
    """

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()

    def generate(
        self,
        graph: SimulationGraph,
        criticality: Optional[Dict[str, Dict]] = None,
        validation: Optional[Dict] = None,
        simulation: Optional[Dict] = None,
        analysis: Optional[Dict] = None,
    ) -> str:
        """
        Generate complete dashboard HTML.
        
        Args:
            graph: SimulationGraph data
            criticality: Criticality scores {id: {score, level}}
            validation: Validation results
            simulation: Simulation results
            analysis: Analysis results (centrality metrics)
        
        Returns:
            Complete HTML dashboard
        """
        # Compute statistics
        stats = self._compute_statistics(graph, criticality)
        
        # Build sections
        sections = []
        
        if self.config.show_statistics:
            sections.append(self._generate_stats_section(stats))
        
        if self.config.show_graph:
            sections.append(self._generate_graph_section(graph, criticality))
        
        if self.config.show_criticality and criticality:
            sections.append(self._generate_criticality_section(criticality, stats))
        
        if self.config.show_validation and validation:
            sections.append(self._generate_validation_section(validation))
        
        if self.config.show_simulation and simulation:
            sections.append(self._generate_simulation_section(simulation))
        
        if analysis:
            sections.append(self._generate_analysis_section(analysis))
        
        return self._generate_html(sections, stats)

    def _compute_statistics(
        self,
        graph: SimulationGraph,
        criticality: Optional[Dict[str, Dict]],
    ) -> Dict:
        """Compute graph statistics"""
        # Component counts by type
        type_counts = {t.value: 0 for t in ComponentType}
        for comp in graph.components.values():
            type_counts[comp.type.value] += 1
        
        # Edge counts by type
        edge_types = {}
        for conn in graph.connections:
            t = conn.type.value
            edge_types[t] = edge_types.get(t, 0) + 1
        
        # Criticality distribution
        crit_dist = {"critical": 0, "high": 0, "medium": 0, "low": 0, "minimal": 0}
        if criticality:
            for c in criticality.values():
                level = c.get("level", "minimal")
                if level in crit_dist:
                    crit_dist[level] += 1
        
        # Message paths
        n_paths = len(graph.get_all_message_paths())
        
        return {
            "n_components": len(graph.components),
            "n_connections": len(graph.connections),
            "n_paths": n_paths,
            "type_counts": type_counts,
            "edge_types": edge_types,
            "criticality_distribution": crit_dist,
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_stats_section(self, stats: Dict) -> str:
        """Generate statistics overview section"""
        type_counts = stats["type_counts"]
        
        return f"""
        <div class="section">
            <h2>📊 System Overview</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{stats['n_components']}</div>
                    <div class="metric-label">Components</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{stats['n_connections']}</div>
                    <div class="metric-label">Connections</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{stats['n_paths']}</div>
                    <div class="metric-label">Message Paths</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{type_counts.get('Broker', 0)}</div>
                    <div class="metric-label">Brokers</div>
                </div>
            </div>
            
            <div class="charts-row">
                <div class="chart-container">
                    <h3>Component Distribution</h3>
                    <canvas id="typeChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Connection Types</h3>
                    <canvas id="edgeChart"></canvas>
                </div>
            </div>
        </div>
        
        <script>
            new Chart(document.getElementById('typeChart'), {{
                type: 'doughnut',
                data: {{
                    labels: ['Applications', 'Topics', 'Brokers', 'Infrastructure'],
                    datasets: [{{
                        data: [{type_counts.get('Application', 0)}, {type_counts.get('Topic', 0)}, 
                               {type_counts.get('Broker', 0)}, {type_counts.get('Node', 0)}],
                        backgroundColor: ['{COLORS["Application"]}', '{COLORS["Topic"]}', 
                                          '{COLORS["Broker"]}', '{COLORS["Node"]}']
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ color: '#ecf0f1' }} }}
                    }}
                }}
            }});
            
            new Chart(document.getElementById('edgeChart'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(list(stats['edge_types'].keys()))},
                    datasets: [{{
                        label: 'Count',
                        data: {json.dumps(list(stats['edge_types'].values()))},
                        backgroundColor: '#3498db'
                    }}]
                }},
                options: {{
                    responsive: true,
                    indexAxis: 'y',
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        x: {{ ticks: {{ color: '#ecf0f1' }}, grid: {{ color: '#333' }} }},
                        y: {{ ticks: {{ color: '#ecf0f1' }}, grid: {{ color: '#333' }} }}
                    }}
                }}
            }});
        </script>
        """

    def _generate_graph_section(
        self,
        graph: SimulationGraph,
        criticality: Optional[Dict[str, Dict]],
    ) -> str:
        """Generate network graph section"""
        # Prepare nodes
        nodes = []
        for comp_id, comp in graph.components.items():
            color = COLORS.get(comp.type.value, "#95a5a6")
            if criticality and comp_id in criticality:
                level = criticality[comp_id].get("level", "minimal")
                color = COLORS.get(level, color)
            
            nodes.append({
                "id": comp_id,
                "label": comp_id,
                "color": {"background": color, "border": color},
                "shape": "dot",
                "size": 15,
            })
        
        # Prepare edges
        edges = []
        for i, conn in enumerate(graph.connections):
            edges.append({
                "id": f"e{i}",
                "from": conn.source,
                "to": conn.target,
                "color": {"color": COLORS.get(conn.type.value, "#7f8c8d"), "opacity": 0.7},
                "arrows": "to",
            })
        
        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)
        
        return f"""
        <div class="section">
            <h2>🔗 System Graph</h2>
            <div id="network" style="height: {self.config.graph_height}; background: #0f0f23; border-radius: 8px;"></div>
        </div>
        
        <script>
            var nodes = new vis.DataSet({nodes_json});
            var edges = new vis.DataSet({edges_json});
            
            var container = document.getElementById('network');
            var data = {{ nodes: nodes, edges: edges }};
            
            var options = {{
                nodes: {{
                    font: {{ size: 10, color: '#ecf0f1' }},
                    borderWidth: 2,
                    shadow: true
                }},
                edges: {{
                    smooth: {{ type: 'continuous' }},
                    shadow: true
                }},
                physics: {{
                    barnesHut: {{
                        gravitationalConstant: -3000,
                        springLength: 100
                    }},
                    stabilization: {{ iterations: 100 }}
                }},
                interaction: {{
                    hover: true,
                    navigationButtons: true
                }}
            }};
            
            var network = new vis.Network(container, data, options);
        </script>
        """

    def _generate_criticality_section(
        self,
        criticality: Dict[str, Dict],
        stats: Dict,
    ) -> str:
        """Generate criticality analysis section"""
        dist = stats["criticality_distribution"]
        
        # Top critical components
        sorted_crit = sorted(
            criticality.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True
        )[:10]
        
        top_rows = "\n".join([
            f"""<tr>
                <td>{comp}</td>
                <td>{crit.get('score', 0):.4f}</td>
                <td><span class="badge badge-{crit.get('level', 'minimal')}">{crit.get('level', 'minimal')}</span></td>
            </tr>"""
            for comp, crit in sorted_crit
        ])
        
        return f"""
        <div class="section">
            <h2>⚠️ Criticality Analysis</h2>
            
            <div class="charts-row">
                <div class="chart-container">
                    <h3>Criticality Distribution</h3>
                    <canvas id="critChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Top Critical Components</h3>
                    <table class="data-table">
                        <thead>
                            <tr><th>Component</th><th>Score</th><th>Level</th></tr>
                        </thead>
                        <tbody>
                            {top_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <script>
            new Chart(document.getElementById('critChart'), {{
                type: 'bar',
                data: {{
                    labels: ['Critical', 'High', 'Medium', 'Low', 'Minimal'],
                    datasets: [{{
                        label: 'Components',
                        data: [{dist['critical']}, {dist['high']}, {dist['medium']}, 
                               {dist['low']}, {dist['minimal']}],
                        backgroundColor: ['{COLORS["critical"]}', '{COLORS["high"]}', 
                                          '{COLORS["medium"]}', '{COLORS["low"]}', '{COLORS["minimal"]}']
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        x: {{ ticks: {{ color: '#ecf0f1' }}, grid: {{ color: '#333' }} }},
                        y: {{ ticks: {{ color: '#ecf0f1' }}, grid: {{ color: '#333' }} }}
                    }}
                }}
            }});
        </script>
        """

    def _generate_validation_section(self, validation: Dict) -> str:
        """Generate validation results section"""
        # Extract metrics
        corr = validation.get("correlation", {})
        cls = validation.get("classification", {})
        rank = validation.get("ranking", {})
        status = validation.get("status", "unknown")
        
        spearman = corr.get("spearman", {}).get("coefficient", 0)
        f1 = cls.get("f1", 0)
        precision = cls.get("precision", 0)
        recall = cls.get("recall", 0)
        
        # Status color
        status_color = "#27ae60" if status == "passed" else "#e67e22" if status == "partial" else "#e74c3c"
        
        # Top-k overlap
        top_k = rank.get("top_k_overlap", {})
        
        return f"""
        <div class="section">
            <h2>✓ Validation Results</h2>
            
            <div class="status-banner" style="background: {status_color}20; border-left: 4px solid {status_color};">
                <strong>Status:</strong> {status.upper()}
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{spearman:.4f}</div>
                    <div class="metric-label">Spearman ρ</div>
                    <div class="metric-target">Target: ≥0.70</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{f1:.4f}</div>
                    <div class="metric-label">F1-Score</div>
                    <div class="metric-target">Target: ≥0.90</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{precision:.4f}</div>
                    <div class="metric-label">Precision</div>
                    <div class="metric-target">Target: ≥0.80</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{recall:.4f}</div>
                    <div class="metric-label">Recall</div>
                    <div class="metric-target">Target: ≥0.80</div>
                </div>
            </div>
            
            <div class="charts-row">
                <div class="chart-container">
                    <h3>Metrics vs Targets</h3>
                    <canvas id="validationChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Confusion Matrix</h3>
                    <div class="confusion-matrix">
                        <div class="cm-row">
                            <div class="cm-cell cm-header"></div>
                            <div class="cm-cell cm-header">Pred +</div>
                            <div class="cm-cell cm-header">Pred -</div>
                        </div>
                        <div class="cm-row">
                            <div class="cm-cell cm-header">Actual +</div>
                            <div class="cm-cell cm-tp">{cls.get('matrix', {}).get('tp', 0)}</div>
                            <div class="cm-cell cm-fn">{cls.get('matrix', {}).get('fn', 0)}</div>
                        </div>
                        <div class="cm-row">
                            <div class="cm-cell cm-header">Actual -</div>
                            <div class="cm-cell cm-fp">{cls.get('matrix', {}).get('fp', 0)}</div>
                            <div class="cm-cell cm-tn">{cls.get('matrix', {}).get('tn', 0)}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            new Chart(document.getElementById('validationChart'), {{
                type: 'radar',
                data: {{
                    labels: ['Spearman', 'F1-Score', 'Precision', 'Recall', 'Top-5'],
                    datasets: [
                        {{
                            label: 'Achieved',
                            data: [{spearman}, {f1}, {precision}, {recall}, {top_k.get('5', 0)}],
                            backgroundColor: 'rgba(52, 152, 219, 0.2)',
                            borderColor: '#3498db',
                            pointBackgroundColor: '#3498db'
                        }},
                        {{
                            label: 'Target',
                            data: [0.70, 0.90, 0.80, 0.80, 0.60],
                            backgroundColor: 'rgba(46, 204, 113, 0.1)',
                            borderColor: '#2ecc71',
                            borderDash: [5, 5],
                            pointBackgroundColor: '#2ecc71'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        r: {{
                            min: 0,
                            max: 1,
                            ticks: {{ color: '#ecf0f1' }},
                            grid: {{ color: '#444' }},
                            pointLabels: {{ color: '#ecf0f1' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ labels: {{ color: '#ecf0f1' }} }}
                    }}
                }}
            }});
        </script>
        """

    def _generate_simulation_section(self, simulation: Dict) -> str:
        """Generate simulation results section"""
        # Extract metrics
        messages = simulation.get("messages", {})
        latency = simulation.get("latency", {})
        failures = simulation.get("failures", {})
        
        published = messages.get("published", 0)
        delivered = messages.get("delivered", 0)
        failed = messages.get("failed", 0)
        delivery_rate = messages.get("delivery_rate", 0)
        
        avg_latency = latency.get("avg_ms", 0)
        p99_latency = latency.get("p99_ms", 0)
        
        return f"""
        <div class="section">
            <h2>🔄 Simulation Results</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{published}</div>
                    <div class="metric-label">Messages Published</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{delivered}</div>
                    <div class="metric-label">Messages Delivered</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{delivery_rate:.1%}</div>
                    <div class="metric-label">Delivery Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_latency:.2f}ms</div>
                    <div class="metric-label">Avg Latency</div>
                </div>
            </div>
            
            <div class="charts-row">
                <div class="chart-container">
                    <h3>Message Delivery</h3>
                    <canvas id="msgChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Latency Distribution</h3>
                    <canvas id="latencyChart"></canvas>
                </div>
            </div>
        </div>
        
        <script>
            new Chart(document.getElementById('msgChart'), {{
                type: 'pie',
                data: {{
                    labels: ['Delivered', 'Failed', 'Timeout'],
                    datasets: [{{
                        data: [{delivered}, {failed}, {messages.get('timeout', 0)}],
                        backgroundColor: ['#27ae60', '#e74c3c', '#f39c12']
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ color: '#ecf0f1' }} }}
                    }}
                }}
            }});
            
            new Chart(document.getElementById('latencyChart'), {{
                type: 'bar',
                data: {{
                    labels: ['Min', 'Avg', 'P99', 'Max'],
                    datasets: [{{
                        label: 'Latency (ms)',
                        data: [{latency.get('min_ms', 0):.2f}, {avg_latency:.2f}, 
                               {p99_latency:.2f}, {latency.get('max_ms', 0):.2f}],
                        backgroundColor: '#9b59b6'
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        x: {{ ticks: {{ color: '#ecf0f1' }}, grid: {{ color: '#333' }} }},
                        y: {{ ticks: {{ color: '#ecf0f1' }}, grid: {{ color: '#333' }} }}
                    }}
                }}
            }});
        </script>
        """

    def _generate_analysis_section(self, analysis: Dict) -> str:
        """Generate analysis results section"""
        # Extract top components by different metrics
        composite = analysis.get("composite", {})
        betweenness = analysis.get("betweenness", {})
        
        # Sort and take top 10
        top_composite = sorted(composite.items(), key=lambda x: -x[1])[:10]
        
        rows = "\n".join([
            f"<tr><td>{comp}</td><td>{score:.4f}</td></tr>"
            for comp, score in top_composite
        ])
        
        return f"""
        <div class="section">
            <h2>📈 Centrality Analysis</h2>
            
            <div class="charts-row">
                <div class="chart-container">
                    <h3>Top Components by Composite Score</h3>
                    <canvas id="compositeChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Centrality Rankings</h3>
                    <table class="data-table">
                        <thead>
                            <tr><th>Component</th><th>Composite Score</th></tr>
                        </thead>
                        <tbody>
                            {rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <script>
            new Chart(document.getElementById('compositeChart'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps([c for c, _ in top_composite])},
                    datasets: [{{
                        label: 'Composite Score',
                        data: {json.dumps([s for _, s in top_composite])},
                        backgroundColor: '#3498db'
                    }}]
                }},
                options: {{
                    responsive: true,
                    indexAxis: 'y',
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        x: {{ ticks: {{ color: '#ecf0f1' }}, grid: {{ color: '#333' }} }},
                        y: {{ ticks: {{ color: '#ecf0f1' }}, grid: {{ color: '#333' }} }}
                    }}
                }}
            }});
        </script>
        """

    def _generate_html(self, sections: List[str], stats: Dict) -> str:
        """Generate complete HTML document"""
        sections_html = "\n".join(sections)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        bg_color = "#1a1a2e" if self.config.theme == "dark" else "#f5f6fa"
        text_color = "#ecf0f1" if self.config.theme == "dark" else "#2c3e50"
        card_bg = "#16213e" if self.config.theme == "dark" else "#ffffff"
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: {bg_color};
            color: {text_color};
            line-height: 1.6;
        }}
        
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px 30px;
            text-align: center;
        }}
        
        #header h1 {{
            font-size: 1.8em;
            margin-bottom: 5px;
        }}
        
        #header p {{
            opacity: 0.9;
            font-size: 0.95em;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .section {{
            background: {card_bg};
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }}
        
        .section h2 {{
            font-size: 1.3em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .section h3 {{
            font-size: 1.1em;
            margin-bottom: 15px;
            color: #667eea;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        
        .metric-card {{
            background: rgba(102, 126, 234, 0.1);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 5px;
        }}
        
        .metric-target {{
            font-size: 0.75em;
            opacity: 0.6;
            margin-top: 3px;
        }}
        
        .charts-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        
        .chart-container {{
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 20px;
        }}
        
        .chart-container canvas {{
            max-height: 300px;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        
        .data-table th,
        .data-table td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .data-table th {{
            background: rgba(102, 126, 234, 0.2);
            font-weight: 600;
        }}
        
        .data-table tr:hover {{
            background: rgba(102, 126, 234, 0.1);
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .badge-critical {{ background: {COLORS['critical']}; color: white; }}
        .badge-high {{ background: {COLORS['high']}; color: white; }}
        .badge-medium {{ background: {COLORS['medium']}; color: black; }}
        .badge-low {{ background: {COLORS['low']}; color: white; }}
        .badge-minimal {{ background: {COLORS['minimal']}; color: white; }}
        
        .status-banner {{
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 1.1em;
        }}
        
        .confusion-matrix {{
            display: inline-block;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 15px;
        }}
        
        .cm-row {{
            display: flex;
        }}
        
        .cm-cell {{
            width: 80px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .cm-header {{
            background: rgba(102, 126, 234, 0.2);
            font-size: 0.85em;
        }}
        
        .cm-tp {{ background: rgba(39, 174, 96, 0.3); }}
        .cm-tn {{ background: rgba(39, 174, 96, 0.2); }}
        .cm-fp {{ background: rgba(231, 76, 60, 0.3); }}
        .cm-fn {{ background: rgba(231, 76, 60, 0.2); }}
        
        #footer {{
            text-align: center;
            padding: 20px;
            opacity: 0.6;
            font-size: 0.85em;
        }}
        
        @media (max-width: 768px) {{
            .charts-row {{
                grid-template-columns: 1fr;
            }}
            
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{self.config.title}</h1>
        <p>Generated: {timestamp} • {stats['n_components']} components • {stats['n_connections']} connections</p>
    </div>
    
    <div class="container">
        {sections_html}
    </div>
    
    <div id="footer">
        Software-as-a-Graph Research Project • Graph-Based Modeling and Analysis of Distributed Pub-Sub Systems
    </div>
</body>
</html>"""
