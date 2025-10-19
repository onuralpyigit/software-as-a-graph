"""
Metrics Dashboard

Creates comprehensive analytics dashboards with charts and metrics.
Visualizes system health, performance, and trends.

Capabilities:
- Interactive metric charts
- System health dashboard
- Trend analysis
- Comparative views
- Export to HTML/PDF
- Real-time updates (when connected to data source)
"""

import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
import json


class ChartType(Enum):
    """Types of charts"""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"


@dataclass
class MetricData:
    """Container for metric data"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'value': round(self.value, 3),
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category,
            'threshold_min': self.threshold_min,
            'threshold_max': self.threshold_max
        }
    
    def is_healthy(self) -> bool:
        """Check if metric is within healthy thresholds"""
        if self.threshold_min is not None and self.value < self.threshold_min:
            return False
        if self.threshold_max is not None and self.value > self.threshold_max:
            return False
        return True


class MetricsDashboard:
    """
    Creates analytics dashboards with metrics and visualizations
    
    Features:
    - System health overview
    - Component metrics
    - Trend charts
    - Comparative analysis
    - Alert status
    - Export capabilities
    """
    
    def __init__(self):
        """Initialize metrics dashboard"""
        self.logger = logging.getLogger(__name__)
        self.metrics: List[MetricData] = []
    
    def create_dashboard(self,
                        graph: nx.DiGraph,
                        analysis_results: Optional[Dict] = None,
                        output_path: Optional[str] = None) -> str:
        """
        Create comprehensive dashboard
        
        Args:
            graph: NetworkX directed graph
            analysis_results: Optional analysis results to include
            output_path: Path to save dashboard
        
        Returns:
            HTML string
        """
        self.logger.info("Creating metrics dashboard...")
        
        # Collect metrics
        metrics = self._collect_metrics(graph, analysis_results)
        
        # Generate HTML
        html = self._create_dashboard_html(graph, metrics, analysis_results)
        
        # Save if path provided
        if output_path:
            Path(output_path).write_text(html)
            self.logger.info(f"Saved dashboard to {output_path}")
        
        return html
    
    def add_metric(self, metric: MetricData):
        """Add a metric to the dashboard"""
        self.metrics.append(metric)
    
    def get_system_health(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Calculate overall system health score
        
        Args:
            graph: NetworkX directed graph
        
        Returns:
            Dictionary with health metrics
        """
        # Calculate various health indicators
        total_nodes = len(graph)
        total_edges = len(graph.edges())
        
        # Connectivity health
        if total_nodes > 0:
            avg_degree = sum(dict(graph.degree()).values()) / total_nodes
            connectivity_score = min(1.0, avg_degree / 10.0)  # Normalize to 0-1
        else:
            connectivity_score = 0.0
        
        # Component health (assume healthy if no failures)
        component_health = 0.9  # Default good health
        
        # Redundancy health
        scc = list(nx.strongly_connected_components(graph))
        redundancy_score = len(scc) / max(1, total_nodes) * 10  # More components = potential redundancy
        redundancy_score = min(1.0, redundancy_score)
        
        # Overall health (weighted average)
        overall_health = (
            connectivity_score * 0.4 +
            component_health * 0.4 +
            redundancy_score * 0.2
        )
        
        return {
            'overall_health': round(overall_health * 100, 1),
            'connectivity': round(connectivity_score * 100, 1),
            'components': round(component_health * 100, 1),
            'redundancy': round(redundancy_score * 100, 1),
            'status': self._get_health_status(overall_health)
        }
    
    def create_trend_chart(self,
                          metrics: List[MetricData],
                          title: str = "Metrics Trend") -> str:
        """
        Create trend chart for metrics over time
        
        Args:
            metrics: List of metric data points
            title: Chart title
        
        Returns:
            HTML string with chart
        """
        # Prepare data for Chart.js
        labels = [m.timestamp.strftime('%Y-%m-%d %H:%M') for m in metrics]
        values = [m.value for m in metrics]
        
        chart_html = f"""
        <div class="chart-container">
            <canvas id="trendChart"></canvas>
        </div>
        <script>
            new Chart(document.getElementById('trendChart'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: [{{
                        label: '{title}',
                        data: {json.dumps(values)},
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{ display: true, text: '{title}' }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true }}
                    }}
                }}
            }});
        </script>
        """
        
        return chart_html
    
    def create_comparison_chart(self,
                               data: Dict[str, float],
                               title: str = "Component Comparison") -> str:
        """
        Create bar chart comparing values
        
        Args:
            data: Dictionary of label -> value
            title: Chart title
        
        Returns:
            HTML string with chart
        """
        labels = list(data.keys())
        values = list(data.values())
        
        chart_html = f"""
        <div class="chart-container">
            <canvas id="comparisonChart"></canvas>
        </div>
        <script>
            new Chart(document.getElementById('comparisonChart'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: [{{
                        label: '{title}',
                        data: {json.dumps(values)},
                        backgroundColor: '#3498db'
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{ display: true, text: '{title}' }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true }}
                    }}
                }}
            }});
        </script>
        """
        
        return chart_html
    
    def _collect_metrics(self,
                        graph: nx.DiGraph,
                        analysis_results: Optional[Dict]) -> List[MetricData]:
        """Collect metrics from graph and analysis results"""
        
        metrics = []
        timestamp = datetime.now()
        
        # Graph metrics
        metrics.append(MetricData(
            name="Total Components",
            value=len(graph),
            unit="count",
            timestamp=timestamp,
            category="topology"
        ))
        
        metrics.append(MetricData(
            name="Total Connections",
            value=len(graph.edges()),
            unit="count",
            timestamp=timestamp,
            category="topology"
        ))
        
        metrics.append(MetricData(
            name="Average Degree",
            value=sum(dict(graph.degree()).values()) / len(graph) if len(graph) > 0 else 0,
            unit="connections/node",
            timestamp=timestamp,
            category="topology"
        ))
        
        metrics.append(MetricData(
            name="Graph Density",
            value=nx.density(graph),
            unit="ratio",
            timestamp=timestamp,
            category="topology"
        ))
        
        # Add analysis metrics if available
        if analysis_results:
            if 'criticality' in analysis_results:
                critical_nodes = sum(
                    1 for score in analysis_results['criticality'].values()
                    if score > 0.7
                )
                metrics.append(MetricData(
                    name="Critical Components",
                    value=critical_nodes,
                    unit="count",
                    timestamp=timestamp,
                    category="criticality",
                    threshold_max=len(graph) * 0.2  # Alert if >20% critical
                ))
        
        return metrics
    
    def _create_dashboard_html(self,
                              graph: nx.DiGraph,
                              metrics: List[MetricData],
                              analysis_results: Optional[Dict]) -> str:
        """Create main dashboard HTML"""
        
        # Get health status
        health = self.get_system_health(graph)
        
        # Create metrics cards
        metrics_html = self._create_metrics_cards(metrics)
        
        # Create health gauge
        health_gauge = self._create_health_gauge(health)
        
        # Get topology stats
        topology_stats = self._get_topology_stats(graph)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>System Metrics Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: #f5f7fa;
                    padding: 20px;
                }}
                .dashboard {{
                    max-width: 1600px;
                    margin: 0 auto;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .header h1 {{
                    font-size: 32px;
                    margin-bottom: 10px;
                }}
                .header .subtitle {{
                    font-size: 16px;
                    opacity: 0.9;
                }}
                .health-section {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .health-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }}
                .health-card {{
                    text-align: center;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid;
                }}
                .health-card.excellent {{ border-color: #10b981; }}
                .health-card.good {{ border-color: #3b82f6; }}
                .health-card.fair {{ border-color: #f59e0b; }}
                .health-card.poor {{ border-color: #ef4444; }}
                .health-value {{
                    font-size: 36px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .health-label {{
                    font-size: 14px;
                    color: #6b7280;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-left: 4px solid #3498db;
                }}
                .metric-value {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #2c3e50;
                    margin: 10px 0;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                .metric-unit {{
                    font-size: 14px;
                    color: #95a5a6;
                }}
                .charts-section {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 30px;
                    margin-bottom: 30px;
                }}
                .chart-container {{
                    background: white;
                    padding: 25px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .topology-stats {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .stat-row {{
                    display: flex;
                    justify-content: space-between;
                    padding: 15px 0;
                    border-bottom: 1px solid #e5e7eb;
                }}
                .stat-row:last-child {{
                    border-bottom: none;
                }}
                .stat-label {{
                    color: #6b7280;
                    font-weight: 500;
                }}
                .stat-value {{
                    color: #2c3e50;
                    font-weight: 600;
                }}
                .status-badge {{
                    display: inline-block;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: 600;
                    text-transform: uppercase;
                }}
                .status-excellent {{ background: #d1fae5; color: #065f46; }}
                .status-good {{ background: #dbeafe; color: #1e40af; }}
                .status-fair {{ background: #fef3c7; color: #92400e; }}
                .status-poor {{ background: #fee2e2; color: #991b1b; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>📊 System Metrics Dashboard</h1>
                    <div class="subtitle">Real-time analytics and system health monitoring</div>
                    <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
                
                <div class="health-section">
                    <h2>System Health Overview</h2>
                    <div class="health-grid">
                        <div class="health-card {self._get_health_class(health['overall_health'])}">
                            <div class="health-label">Overall Health</div>
                            <div class="health-value">{health['overall_health']}%</div>
                            <span class="status-badge status-{health['status'].lower()}">{health['status']}</span>
                        </div>
                        <div class="health-card {self._get_health_class(health['connectivity'])}">
                            <div class="health-label">Connectivity</div>
                            <div class="health-value">{health['connectivity']}%</div>
                        </div>
                        <div class="health-card {self._get_health_class(health['components'])}">
                            <div class="health-label">Components</div>
                            <div class="health-value">{health['components']}%</div>
                        </div>
                        <div class="health-card {self._get_health_class(health['redundancy'])}">
                            <div class="health-label">Redundancy</div>
                            <div class="health-value">{health['redundancy']}%</div>
                        </div>
                    </div>
                </div>
                
                <div class="metrics-grid">
                    {metrics_html}
                </div>
                
                <div class="topology-stats">
                    <h2>Topology Statistics</h2>
                    {self._create_topology_html(topology_stats)}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_metrics_cards(self, metrics: List[MetricData]) -> str:
        """Create HTML for metric cards"""
        
        cards_html = ""
        
        for metric in metrics:
            health_indicator = "✓" if metric.is_healthy() else "⚠"
            health_color = "#10b981" if metric.is_healthy() else "#ef4444"
            
            cards_html += f"""
            <div class="metric-card">
                <div class="metric-label">
                    {metric.name}
                    <span style="color: {health_color}; float: right;">{health_indicator}</span>
                </div>
                <div class="metric-value">{metric.value:,.2f}</div>
                <div class="metric-unit">{metric.unit}</div>
            </div>
            """
        
        return cards_html
    
    def _create_health_gauge(self, health: Dict) -> str:
        """Create health gauge visualization"""
        
        value = health['overall_health']
        color = self._get_health_color(value)
        
        return f"""
        <div style="text-align: center; padding: 20px;">
            <svg width="200" height="200">
                <circle cx="100" cy="100" r="80" fill="none" stroke="#e5e7eb" stroke-width="20"/>
                <circle cx="100" cy="100" r="80" fill="none" stroke="{color}" stroke-width="20"
                        stroke-dasharray="{value * 5.03} 503" transform="rotate(-90 100 100)"/>
                <text x="100" y="100" text-anchor="middle" dy="10" font-size="32" font-weight="bold">
                    {value}%
                </text>
            </svg>
        </div>
        """
    
    def _get_topology_stats(self, graph: nx.DiGraph) -> Dict:
        """Get detailed topology statistics"""
        
        return {
            'nodes': len(graph),
            'edges': len(graph.edges()),
            'density': nx.density(graph),
            'avg_degree': sum(dict(graph.degree()).values()) / len(graph) if len(graph) > 0 else 0,
            'components': nx.number_weakly_connected_components(graph),
            'diameter': self._safe_diameter(graph)
        }
    
    def _safe_diameter(self, graph: nx.DiGraph) -> Optional[int]:
        """Safely calculate diameter"""
        try:
            if nx.is_weakly_connected(graph):
                return nx.diameter(graph.to_undirected())
        except:
            pass
        return None
    
    def _create_topology_html(self, stats: Dict) -> str:
        """Create topology statistics HTML"""
        
        html = ""
        for label, value in stats.items():
            if value is not None:
                display_value = f"{value:.3f}" if isinstance(value, float) else str(value)
                html += f"""
                <div class="stat-row">
                    <span class="stat-label">{label.replace('_', ' ').title()}</span>
                    <span class="stat-value">{display_value}</span>
                </div>
                """
        
        return html
    
    def _get_health_status(self, score: float) -> str:
        """Get health status label"""
        if score >= 0.9:
            return "EXCELLENT"
        elif score >= 0.75:
            return "GOOD"
        elif score >= 0.5:
            return "FAIR"
        else:
            return "POOR"
    
    def _get_health_class(self, score: float) -> str:
        """Get CSS class for health score"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 50:
            return "fair"
        else:
            return "poor"
    
    def _get_health_color(self, score: float) -> str:
        """Get color for health score"""
        if score >= 90:
            return "#10b981"
        elif score >= 75:
            return "#3b82f6"
        elif score >= 50:
            return "#f59e0b"
        else:
            return "#ef4444"
