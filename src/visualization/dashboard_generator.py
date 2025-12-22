#!/usr/bin/env python3
"""
Dashboard Generator for Pub-Sub System Analysis
=================================================

Generates comprehensive HTML dashboards combining:
- Interactive network visualization
- Analysis metrics and statistics
- Validation results
- Simulation impact visualization
- Layer-by-layer breakdown

Author: Software-as-a-Graph Research Project
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")

from .graph_visualizer import GraphVisualizer, VisualizationConfig, Colors, Layer


@dataclass
class DashboardConfig:
    """Configuration for dashboard generation"""
    title: str = "Pub-Sub System Analysis Dashboard"
    subtitle: str = "Comprehensive Analysis Report"
    theme: str = "dark"  # dark or light
    include_network: bool = True
    include_metrics: bool = True
    include_validation: bool = True
    include_simulation: bool = True
    include_layers: bool = True
    include_components: bool = True
    max_components: int = 50


class DashboardGenerator:
    """
    Generates comprehensive HTML dashboards for pub-sub system analysis.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        Initialize dashboard generator.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self.visualizer = GraphVisualizer()
    
    def generate(self,
                graph: nx.DiGraph,
                criticality: Optional[Dict[str, Dict]] = None,
                validation: Optional[Dict[str, Any]] = None,
                simulation: Optional[Dict[str, Any]] = None,
                analysis: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive dashboard HTML.
        
        Args:
            graph: NetworkX graph
            criticality: Criticality analysis results
            validation: Validation results
            simulation: Simulation results
            analysis: Additional analysis results
            
        Returns:
            Complete HTML string
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Classify layers
        self.visualizer.classify_layers(graph)
        layer_stats = self.visualizer.get_layer_statistics(graph)
        
        # Build sections
        sections = []
        
        # Overview section
        sections.append(self._generate_overview_section(graph, layer_stats, criticality))
        
        # Network visualization
        if self.config.include_network:
            sections.append(self._generate_network_section(graph, criticality))
        
        # Metrics section
        if self.config.include_metrics and criticality:
            sections.append(self._generate_metrics_section(criticality, analysis))
        
        # Validation section
        if self.config.include_validation and validation:
            sections.append(self._generate_validation_section(validation))
        
        # Simulation section
        if self.config.include_simulation and simulation:
            sections.append(self._generate_simulation_section(simulation))
        
        # Layer analysis
        if self.config.include_layers:
            sections.append(self._generate_layers_section(layer_stats))
        
        # Components table
        if self.config.include_components and criticality:
            sections.append(self._generate_components_section(graph, criticality))
        
        return self._wrap_in_html(sections, timestamp)
    
    def _generate_overview_section(self,
                                   graph: nx.DiGraph,
                                   layer_stats: Dict,
                                   criticality: Optional[Dict]) -> str:
        """Generate overview metrics section"""
        
        # Count critical components
        critical_count = 0
        spof_count = 0
        if criticality:
            for node, crit in criticality.items():
                if crit.get('level') in ['critical', 'high']:
                    critical_count += 1
                if crit.get('is_articulation_point'):
                    spof_count += 1
        
        return f"""
        <div class="section" id="overview">
            <h2>📊 System Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{graph.number_of_nodes()}</div>
                    <div class="metric-label">Total Components</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{graph.number_of_edges()}</div>
                    <div class="metric-label">Connections</div>
                </div>
                <div class="metric-card highlight-warning">
                    <div class="metric-value">{critical_count}</div>
                    <div class="metric-label">Critical Components</div>
                </div>
                <div class="metric-card highlight-danger">
                    <div class="metric-value">{spof_count}</div>
                    <div class="metric-label">Single Points of Failure</div>
                </div>
            </div>
            
            <div class="sub-section">
                <h3>Component Distribution</h3>
                <div class="distribution-chart">
                    {self._generate_distribution_bars(layer_stats)}
                </div>
            </div>
        </div>
        """
    
    def _generate_distribution_bars(self, layer_stats: Dict) -> str:
        """Generate distribution bar chart"""
        bars = []
        total = layer_stats.get('total_nodes', 1)
        
        layer_colors = {
            'application': Colors.LAYERS[Layer.APPLICATION],
            'topic': Colors.LAYERS[Layer.TOPIC],
            'broker': Colors.LAYERS[Layer.BROKER],
            'infrastructure': Colors.LAYERS[Layer.INFRASTRUCTURE]
        }
        
        for layer_name, stats in layer_stats.get('layers', {}).items():
            count = stats.get('node_count', 0)
            pct = (count / total * 100) if total > 0 else 0
            color = layer_colors.get(layer_name, '#95a5a6')
            
            bars.append(f"""
                <div class="bar-row">
                    <div class="bar-label">{layer_name.title()}</div>
                    <div class="bar-container">
                        <div class="bar" style="width: {pct}%; background: {color}"></div>
                    </div>
                    <div class="bar-value">{count}</div>
                </div>
            """)
        
        return ''.join(bars)
    
    def _generate_network_section(self,
                                  graph: nx.DiGraph,
                                  criticality: Optional[Dict]) -> str:
        """Generate network visualization section"""
        
        # Prepare nodes
        nodes_data = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            crit = criticality.get(node, {}) if criticality else {}
            
            node_type = node_data.get('type', 'Unknown')
            color = Colors.NODE_TYPES.get(node_type, '#95a5a6')
            
            size = 25
            if crit:
                score = crit.get('score', 0)
                size = int(20 + score * 30)
            
            tooltip = f"<b>{node}</b><br>Type: {node_type}"
            if crit:
                tooltip += f"<br>Criticality: {crit.get('level', 'N/A')} ({crit.get('score', 0):.3f})"
            
            nodes_data.append({
                'id': node,
                'label': node[:20],
                'color': color,
                'size': size,
                'title': tooltip
            })
        
        # Prepare edges
        edges_data = []
        for source, target, data in graph.edges(data=True):
            edge_type = data.get('type', 'Unknown')
            color = Colors.EDGES.get(edge_type, '#95a5a6')
            
            edges_data.append({
                'from': source,
                'to': target,
                'color': {'color': color, 'opacity': 0.7},
                'arrows': 'to',
                'title': edge_type
            })
        
        return f"""
        <div class="section" id="network">
            <h2>🔗 System Architecture</h2>
            <div class="network-container" id="network-graph"></div>
            <div class="network-legend">
                <span class="legend-item"><span class="dot" style="background: {Colors.NODE_TYPES['Application']}"></span> Application</span>
                <span class="legend-item"><span class="dot" style="background: {Colors.NODE_TYPES['Topic']}"></span> Topic</span>
                <span class="legend-item"><span class="dot" style="background: {Colors.NODE_TYPES['Broker']}"></span> Broker</span>
                <span class="legend-item"><span class="dot" style="background: {Colors.NODE_TYPES['Node']}"></span> Infrastructure</span>
            </div>
        </div>
        
        <script>
            var networkNodes = new vis.DataSet({json.dumps(nodes_data)});
            var networkEdges = new vis.DataSet({json.dumps(edges_data)});
            
            var networkContainer = document.getElementById('network-graph');
            var networkData = {{ nodes: networkNodes, edges: networkEdges }};
            
            var networkOptions = {{
                nodes: {{
                    shape: 'dot',
                    font: {{ size: 11, color: '#ecf0f1' }},
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
                    stabilization: {{ iterations: 150 }}
                }},
                interaction: {{
                    hover: true,
                    navigationButtons: true
                }}
            }};
            
            var network = new vis.Network(networkContainer, networkData, networkOptions);
        </script>
        """
    
    def _generate_metrics_section(self,
                                  criticality: Dict,
                                  analysis: Optional[Dict]) -> str:
        """Generate metrics analysis section"""
        
        # Calculate statistics
        scores = [c.get('score', 0) for c in criticality.values()]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        # Count by level
        level_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'minimal': 0}
        for crit in criticality.values():
            level = crit.get('level', 'minimal')
            if level in level_counts:
                level_counts[level] += 1
        
        return f"""
        <div class="section" id="metrics">
            <h2>📈 Criticality Analysis</h2>
            
            <div class="metrics-row">
                <div class="metric-card">
                    <div class="metric-value">{avg_score:.3f}</div>
                    <div class="metric-label">Average Criticality</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{max_score:.3f}</div>
                    <div class="metric-label">Max Criticality</div>
                </div>
            </div>
            
            <div class="sub-section">
                <h3>Criticality Distribution</h3>
                <div class="criticality-chart">
                    <div class="crit-bar">
                        <div class="crit-level critical" style="width: {level_counts['critical'] / max(len(criticality), 1) * 100}%">
                            Critical ({level_counts['critical']})
                        </div>
                    </div>
                    <div class="crit-bar">
                        <div class="crit-level high" style="width: {level_counts['high'] / max(len(criticality), 1) * 100}%">
                            High ({level_counts['high']})
                        </div>
                    </div>
                    <div class="crit-bar">
                        <div class="crit-level medium" style="width: {level_counts['medium'] / max(len(criticality), 1) * 100}%">
                            Medium ({level_counts['medium']})
                        </div>
                    </div>
                    <div class="crit-bar">
                        <div class="crit-level low" style="width: {level_counts['low'] / max(len(criticality), 1) * 100}%">
                            Low ({level_counts['low']})
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_validation_section(self, validation: Dict) -> str:
        """Generate validation results section"""
        
        status = validation.get('status', 'unknown')
        status_class = 'success' if status == 'passed' else 'warning' if status == 'marginal' else 'danger'
        
        corr = validation.get('correlation', {})
        spearman = corr.get('spearman', {}).get('coefficient', 0)
        
        classification = validation.get('classification', {}).get('overall', {})
        f1 = classification.get('f1_score', 0)
        precision = classification.get('precision', 0)
        recall = classification.get('recall', 0)
        
        return f"""
        <div class="section" id="validation">
            <h2>✅ Validation Results</h2>
            
            <div class="status-badge {status_class}">
                {status.upper()}
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{spearman:.3f}</div>
                    <div class="metric-label">Spearman ρ</div>
                    <div class="metric-target">Target: ≥0.70</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{f1:.3f}</div>
                    <div class="metric-label">F1-Score</div>
                    <div class="metric-target">Target: ≥0.90</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{precision:.3f}</div>
                    <div class="metric-label">Precision</div>
                    <div class="metric-target">Target: ≥0.80</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{recall:.3f}</div>
                    <div class="metric-label">Recall</div>
                    <div class="metric-target">Target: ≥0.80</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_simulation_section(self, simulation: Dict) -> str:
        """Generate simulation results section"""
        
        total_simulations = simulation.get('total_simulations', 0)
        
        # Get top failures by impact
        top_failures = []
        results = simulation.get('results', [])
        sorted_results = sorted(results, key=lambda x: x.get('impact_score', 0), reverse=True)[:5]
        
        failure_rows = []
        for result in sorted_results:
            component = result.get('primary_failures', ['Unknown'])[0] if result.get('primary_failures') else 'Unknown'
            impact = result.get('impact_score', 0)
            affected = result.get('affected_nodes', 0)
            
            failure_rows.append(f"""
                <tr>
                    <td>{component}</td>
                    <td>{impact:.3f}</td>
                    <td>{affected}</td>
                </tr>
            """)
        
        return f"""
        <div class="section" id="simulation">
            <h2>🔥 Failure Simulation</h2>
            
            <div class="metric-card single">
                <div class="metric-value">{total_simulations}</div>
                <div class="metric-label">Simulations Run</div>
            </div>
            
            <div class="sub-section">
                <h3>Highest Impact Failures</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Component</th>
                            <th>Impact Score</th>
                            <th>Affected Nodes</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(failure_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        """
    
    def _generate_layers_section(self, layer_stats: Dict) -> str:
        """Generate layer analysis section"""
        
        layer_cards = []
        layer_icons = {
            'application': '📱',
            'topic': '📨',
            'broker': '🔄',
            'infrastructure': '🖥️'
        }
        
        for layer_name, stats in layer_stats.get('layers', {}).items():
            icon = layer_icons.get(layer_name, '📦')
            color = {
                'application': Colors.LAYERS[Layer.APPLICATION],
                'topic': Colors.LAYERS[Layer.TOPIC],
                'broker': Colors.LAYERS[Layer.BROKER],
                'infrastructure': Colors.LAYERS[Layer.INFRASTRUCTURE]
            }.get(layer_name, '#95a5a6')
            
            layer_cards.append(f"""
                <div class="layer-card" style="border-left: 4px solid {color}">
                    <div class="layer-header">
                        <span class="layer-icon">{icon}</span>
                        <span class="layer-name">{layer_name.title()}</span>
                    </div>
                    <div class="layer-stats">
                        <div class="layer-stat">
                            <span class="stat-value">{stats.get('node_count', 0)}</span>
                            <span class="stat-label">Nodes</span>
                        </div>
                        <div class="layer-stat">
                            <span class="stat-value">{stats.get('internal_edges', 0)}</span>
                            <span class="stat-label">Internal Edges</span>
                        </div>
                        <div class="layer-stat">
                            <span class="stat-value">{stats.get('cross_layer_edges', 0)}</span>
                            <span class="stat-label">Cross-Layer</span>
                        </div>
                        <div class="layer-stat">
                            <span class="stat-value">{stats.get('density', 0):.3f}</span>
                            <span class="stat-label">Density</span>
                        </div>
                    </div>
                </div>
            """)
        
        return f"""
        <div class="section" id="layers">
            <h2>🔀 Layer Analysis</h2>
            <div class="layers-grid">
                {''.join(layer_cards)}
            </div>
        </div>
        """
    
    def _generate_components_section(self,
                                     graph: nx.DiGraph,
                                     criticality: Dict) -> str:
        """Generate components table section"""
        
        # Sort by criticality
        sorted_components = sorted(
            criticality.items(),
            key=lambda x: x[1].get('score', 0),
            reverse=True
        )[:self.config.max_components]
        
        rows = []
        for i, (node, crit) in enumerate(sorted_components, 1):
            node_type = graph.nodes[node].get('type', 'Unknown') if node in graph.nodes() else 'Unknown'
            level = crit.get('level', 'minimal')
            score = crit.get('score', 0)
            is_ap = '⚠️' if crit.get('is_articulation_point') else ''
            
            level_class = f"level-{level}"
            
            rows.append(f"""
                <tr>
                    <td>{i}</td>
                    <td>{node}</td>
                    <td>{node_type}</td>
                    <td><span class="level-badge {level_class}">{level}</span></td>
                    <td>{score:.4f}</td>
                    <td>{is_ap}</td>
                </tr>
            """)
        
        return f"""
        <div class="section" id="components">
            <h2>📋 Top Critical Components</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Component</th>
                        <th>Type</th>
                        <th>Level</th>
                        <th>Score</th>
                        <th>SPOF</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """
    
    def _wrap_in_html(self, sections: List[str], timestamp: str) -> str:
        """Wrap sections in complete HTML document"""
        
        is_dark = self.config.theme == 'dark'
        bg_color = '#1a1a2e' if is_dark else '#f5f5f5'
        card_bg = '#16213e' if is_dark else '#ffffff'
        text_color = '#ecf0f1' if is_dark else '#333333'
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: {bg_color};
            color: {text_color};
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.2em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .section {{
            background: {card_bg};
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }}
        
        .sub-section {{
            margin-top: 25px;
        }}
        
        .sub-section h3 {{
            margin-bottom: 15px;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        
        .metrics-row {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .metric-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        
        .metric-card.single {{
            max-width: 200px;
        }}
        
        .metric-value {{
            font-size: 2.5em;
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
            margin-top: 5px;
        }}
        
        .highlight-warning .metric-value {{ color: #f39c12; }}
        .highlight-danger .metric-value {{ color: #e74c3c; }}
        
        .network-container {{
            height: 500px;
            background: #0f0f23;
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        
        .network-legend {{
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
        }}
        
        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        
        .distribution-chart {{
            margin-top: 15px;
        }}
        
        .bar-row {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        
        .bar-label {{
            width: 120px;
            font-size: 0.9em;
        }}
        
        .bar-container {{
            flex: 1;
            height: 24px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin: 0 15px;
        }}
        
        .bar {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        
        .bar-value {{
            width: 40px;
            text-align: right;
            font-weight: bold;
        }}
        
        .criticality-chart {{
            margin-top: 15px;
        }}
        
        .crit-bar {{
            margin: 8px 0;
        }}
        
        .crit-level {{
            padding: 8px 15px;
            border-radius: 4px;
            font-size: 0.85em;
            min-width: 100px;
            display: inline-block;
        }}
        
        .crit-level.critical {{ background: #c0392b; }}
        .crit-level.high {{ background: #e74c3c; }}
        .crit-level.medium {{ background: #f39c12; }}
        .crit-level.low {{ background: #27ae60; }}
        
        .status-badge {{
            display: inline-block;
            padding: 10px 25px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 20px;
        }}
        
        .status-badge.success {{ background: #27ae60; color: white; }}
        .status-badge.warning {{ background: #f39c12; color: white; }}
        .status-badge.danger {{ background: #e74c3c; color: white; }}
        
        .layers-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }}
        
        .layer-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
        }}
        
        .layer-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }}
        
        .layer-icon {{
            font-size: 1.5em;
        }}
        
        .layer-name {{
            font-weight: 600;
            font-size: 1.1em;
        }}
        
        .layer-stats {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }}
        
        .layer-stat {{
            text-align: center;
        }}
        
        .layer-stat .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            display: block;
        }}
        
        .layer-stat .stat-label {{
            font-size: 0.8em;
            opacity: 0.7;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .data-table th,
        .data-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .data-table th {{
            background: rgba(255,255,255,0.05);
            font-weight: 600;
        }}
        
        .data-table tr:hover {{
            background: rgba(255,255,255,0.03);
        }}
        
        .level-badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        
        .level-critical {{ background: #c0392b; color: white; }}
        .level-high {{ background: #e74c3c; color: white; }}
        .level-medium {{ background: #f39c12; color: white; }}
        .level-low {{ background: #27ae60; color: white; }}
        .level-minimal {{ background: #95a5a6; color: white; }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            opacity: 0.6;
            font-size: 0.85em;
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .layers-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.config.title}</h1>
        <p>{self.config.subtitle}</p>
    </div>
    
    <div class="container">
        {''.join(sections)}
    </div>
    
    <div class="footer">
        Generated: {timestamp} | Software-as-a-Graph Analysis Framework
    </div>
</body>
</html>"""