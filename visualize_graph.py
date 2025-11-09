#!/usr/bin/env python3
"""
Visualize Graph Script - Refactored Architecture

Command-line interface for comprehensive visualization of distributed pub-sub systems
using the refactored modular architecture.

Features:
- Multi-layer graph visualization (Application, Infrastructure, Topic layers)
- Hidden dependency detection and highlighting
- Failure impact visualization
- Interactive HTML visualizations
- Static image generation (PNG, SVG)
- Criticality-aware color schemes
- Custom layout algorithms
- Metrics dashboard generation
- Cross-layer dependency analysis

Architecture:
  GraphBuilder → GraphModel → GraphExporter → NetworkX
                                                  ↓
                                         AnalysisOrchestrator
                                                  ↓
                                         GraphVisualizer / LayerRenderer
                                                  ↓
                                         Visualization Outputs

Usage:
    # Basic visualization from JSON
    python visualize_graph.py --input system.json
    
    # Visualization from Neo4j
    python visualize_graph.py --neo4j --uri bolt://localhost:7687 \\
        --user neo4j --password password
    
    # Multi-layer visualization with failure simulation
    python visualize_graph.py --input system.json --multi-layer \\
        --simulate-failure B1 --output-dir visualizations/
    
    # Generate all visualization types
    python visualize_graph.py --input system.json --all \\
        --layout force-directed --color-scheme criticality
    
    # Hidden dependencies analysis
    python visualize_graph.py --input system.json --hidden-deps \\
        --threshold 0.8
    
    # Dashboard with failure scenarios
    python visualize_graph.py --input system.json --dashboard \\
        --failure-scenarios spof cascade
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.core.graph_builder import GraphBuilder
    from src.core.graph_exporter import GraphExporter
    from src.core.graph_model import GraphModel
    from src.orchestration.analysis_orchestrator import AnalysisOrchestrator
    from src.visualization.graph_visualizer import (
        GraphVisualizer, VisualizationConfig, LayoutAlgorithm, ColorScheme
    )
    from src.visualization.layer_renderer import LayerRenderer, Layer, LayerConfig
    from src.visualization.metrics_dashboard import MetricsDashboard
    from src.simulation.failure_simulator import FailureSimulator
    import networkx as nx
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are in the src/ directory")
    sys.exit(1)


class VisualizationOrchestrator:
    """
    Orchestrates comprehensive visualization workflows for pub-sub systems
    """
    
    def __init__(self, output_dir: Path = Path("visualizations")):
        """
        Initialize visualization orchestrator
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize visualization components
        self.graph_visualizer = GraphVisualizer()
        self.layer_renderer = LayerRenderer()
        self.dashboard = MetricsDashboard()
        self.failure_simulator = FailureSimulator()
    
    def visualize_complete_system(self,
                                  graph: nx.DiGraph,
                                  graph_model: GraphModel,
                                  layout: str = "spring",
                                  color_scheme: str = "criticality",
                                  show_labels: bool = True) -> str:
        """
        Create complete system visualization
        
        Args:
            graph: NetworkX graph
            graph_model: GraphModel instance
            layout: Layout algorithm
            color_scheme: Color scheme
            show_labels: Show node labels
            
        Returns:
            Path to output file
        """
        self.logger.info("Creating complete system visualization...")
        
        # Map string parameters to enums
        layout_algo = self._get_layout_algorithm(layout)
        color_enum = self._get_color_scheme(color_scheme)
        
        # Create configuration
        config = VisualizationConfig(
            layout=layout_algo,
            color_scheme=color_enum,
            show_labels=show_labels
        )
        
        # Generate visualization
        output_path = self.output_dir / "complete_system.html"
        self.graph_visualizer.visualize(graph, config, str(output_path))
        
        self.logger.info(f"✓ Complete system visualization saved to: {output_path}")
        return str(output_path)
    
    def visualize_multi_layer(self,
                             graph: nx.DiGraph,
                             include_interactions: bool = True) -> Dict[str, str]:
        """
        Create multi-layer visualizations
        
        Args:
            graph: NetworkX graph
            include_interactions: Include cross-layer interactions
            
        Returns:
            Dictionary mapping layer names to output paths
        """
        self.logger.info("Creating multi-layer visualizations...")
        
        outputs = {}
        
        # Individual layer visualizations
        layers = [Layer.APPLICATION, Layer.INFRASTRUCTURE, Layer.TOPIC]
        
        for layer in layers:
            layer_name = layer.value.lower()
            output_path = self.output_dir / f"layer_{layer_name}.html"
            
            self.logger.info(f"  Rendering {layer_name} layer...")
            self.layer_renderer.render_layer(
                graph,
                layer,
                output_path=str(output_path)
            )
            
            outputs[layer_name] = str(output_path)
        
        # Composite view
        composite_path = self.output_dir / "layers_composite.html"
        self.logger.info("  Creating composite view...")
        self.layer_renderer.render_all_layers(graph, str(composite_path))
        outputs['composite'] = str(composite_path)
        
        # Cross-layer interactions
        if include_interactions:
            interaction_pairs = [
                (Layer.APPLICATION, Layer.INFRASTRUCTURE, "app_infra"),
                (Layer.APPLICATION, Layer.TOPIC, "app_topic"),
                (Layer.INFRASTRUCTURE, Layer.TOPIC, "infra_topic")
            ]
            
            for layer1, layer2, name in interaction_pairs:
                interaction_path = self.output_dir / f"interaction_{name}.html"
                self.logger.info(f"  Analyzing {name} interactions...")
                
                self.layer_renderer.render_layer_interactions(
                    graph,
                    layer1,
                    layer2,
                    output_path=str(interaction_path)
                )
                
                outputs[f'interaction_{name}'] = str(interaction_path)
        
        self.logger.info(f"✓ Multi-layer visualizations complete: {len(outputs)} files")
        return outputs
    
    def visualize_hidden_dependencies(self,
                                     graph: nx.DiGraph,
                                     threshold: float = 0.7) -> str:
        """
        Detect and visualize hidden dependencies
        
        Args:
            graph: NetworkX graph
            threshold: Threshold for considering dependencies critical
            
        Returns:
            Path to output file
        """
        self.logger.info(f"Analyzing hidden dependencies (threshold: {threshold})...")
        
        # Detect hidden dependencies through transitive closure
        hidden_deps = self._detect_hidden_dependencies(graph, threshold)
        
        # Create highlighted visualization
        config = VisualizationConfig(
            layout=LayoutAlgorithm.HIERARCHICAL,
            color_scheme=ColorScheme.CRITICALITY,
            show_labels=True,
            highlight_nodes=hidden_deps['critical_nodes'],
            highlight_edges=hidden_deps['hidden_edges']
        )
        
        output_path = self.output_dir / "hidden_dependencies.html"
        self.graph_visualizer.visualize(graph, config, str(output_path))
        
        # Generate report
        self._generate_dependency_report(hidden_deps, output_path)
        
        self.logger.info(f"✓ Hidden dependencies visualization saved to: {output_path}")
        self.logger.info(f"  Found {len(hidden_deps['hidden_edges'])} hidden dependencies")
        return str(output_path)
    
    def visualize_failure_impact(self,
                                graph: nx.DiGraph,
                                component_id: str,
                                enable_cascade: bool = True) -> Dict[str, Any]:
        """
        Visualize failure impact for a specific component
        
        Args:
            graph: NetworkX graph
            component_id: Component to fail
            enable_cascade: Enable cascading failures
            
        Returns:
            Dictionary with visualization paths and impact metrics
        """
        self.logger.info(f"Simulating failure of component: {component_id}")
        
        # Simulate failure
        result = self.failure_simulator.simulate_single_failure(
            graph,
            component_id,
            enable_cascade=enable_cascade
        )
        
        # Create before/after visualizations
        output_before = self.output_dir / f"failure_{component_id}_before.html"
        output_after = self.output_dir / f"failure_{component_id}_after.html"
        output_impact = self.output_dir / f"failure_{component_id}_impact.html"
        
        # Before visualization (highlight component to fail)
        config_before = VisualizationConfig(
            layout=LayoutAlgorithm.SPRING,
            color_scheme=ColorScheme.CRITICALITY,
            show_labels=True,
            highlight_nodes=[component_id]
        )
        self.graph_visualizer.visualize(graph, config_before, str(output_before))
        
        # After visualization (show failed and affected components)
        failed_components = [component_id] + result.cascade_failures
        config_after = VisualizationConfig(
            layout=LayoutAlgorithm.SPRING,
            color_scheme=ColorScheme.TYPE,
            show_labels=True,
            highlight_nodes=result.affected_components,
            highlight_edges=[],
            node_colors={node: '#ff0000' for node in failed_components}
        )
        self.graph_visualizer.visualize(graph, config_after, str(output_after))
        
        # Impact visualization with metrics
        self._create_impact_visualization(
            graph, result, component_id, str(output_impact)
        )
        
        self.logger.info(f"✓ Failure impact visualization complete")
        self.logger.info(f"  Impact Score: {result.impact_score:.3f}")
        self.logger.info(f"  Affected Components: {len(result.affected_components)}")
        self.logger.info(f"  Cascade Failures: {len(result.cascade_failures)}")
        
        return {
            'before': str(output_before),
            'after': str(output_after),
            'impact': str(output_impact),
            'metrics': {
                'impact_score': result.impact_score,
                'affected_count': len(result.affected_components),
                'cascade_count': len(result.cascade_failures),
                'resilience_score': result.resilience_score
            }
        }
    
    def visualize_failure_scenarios(self,
                                   graph: nx.DiGraph,
                                   scenarios: List[str] = None) -> Dict[str, str]:
        """
        Visualize multiple failure scenarios
        
        Args:
            graph: NetworkX graph
            scenarios: List of scenario types (spof, cascade, multi, etc.)
            
        Returns:
            Dictionary mapping scenario names to output paths
        """
        if scenarios is None:
            scenarios = ['spof', 'cascade']
        
        self.logger.info(f"Generating {len(scenarios)} failure scenarios...")
        
        outputs = {}
        
        for scenario in scenarios:
            if scenario == 'spof':
                outputs['spof'] = self._visualize_spof(graph)
            elif scenario == 'cascade':
                outputs['cascade'] = self._visualize_cascade_risk(graph)
            elif scenario == 'multi':
                outputs['multi'] = self._visualize_multi_failure(graph)
            elif scenario == 'recovery':
                outputs['recovery'] = self._visualize_recovery_paths(graph)
        
        return outputs
    
    def create_metrics_dashboard(self,
                                graph: nx.DiGraph,
                                include_health: bool = True,
                                include_trends: bool = False) -> str:
        """
        Create comprehensive metrics dashboard
        
        Args:
            graph: NetworkX graph
            include_health: Include system health metrics
            include_trends: Include trend analysis (requires historical data)
            
        Returns:
            Path to dashboard file
        """
        self.logger.info("Creating metrics dashboard...")
        
        output_path = self.output_dir / "metrics_dashboard.html"
        
        # Generate dashboard
        self.dashboard.create_dashboard(
            graph,
            output_path=str(output_path),
            include_health=include_health,
            include_trends=include_trends
        )
        
        # Get system health
        if include_health:
            health = self.dashboard.get_system_health(graph)
            self.logger.info(f"  System Health: {health['overall_health']}% ({health['status']})")
        
        self.logger.info(f"✓ Metrics dashboard saved to: {output_path}")
        return str(output_path)
    
    def generate_visualization_report(self,
                                    graph: nx.DiGraph,
                                    graph_model: GraphModel,
                                    outputs: Dict[str, Any]) -> str:
        """
        Generate comprehensive visualization report
        
        Args:
            graph: NetworkX graph
            graph_model: GraphModel instance
            outputs: Dictionary of output files
            
        Returns:
            Path to report file
        """
        self.logger.info("Generating visualization report...")
        
        report_path = self.output_dir / "visualization_report.html"
        
        # Collect system statistics
        stats = self._collect_system_stats(graph, graph_model)
        
        # Generate HTML report
        html_content = self._generate_html_report(stats, outputs)
        
        report_path.write_text(html_content)
        
        self.logger.info(f"✓ Visualization report saved to: {report_path}")
        return str(report_path)
    
    # Helper methods
    
    def _get_layout_algorithm(self, layout: str) -> LayoutAlgorithm:
        """Convert string to LayoutAlgorithm enum"""
        mapping = {
            'spring': LayoutAlgorithm.SPRING,
            'force-directed': LayoutAlgorithm.SPRING,
            'hierarchical': LayoutAlgorithm.HIERARCHICAL,
            'circular': LayoutAlgorithm.CIRCULAR,
            'random': LayoutAlgorithm.RANDOM
        }
        return mapping.get(layout.lower(), LayoutAlgorithm.SPRING)
    
    def _get_color_scheme(self, scheme: str) -> ColorScheme:
        """Convert string to ColorScheme enum"""
        mapping = {
            'type': ColorScheme.TYPE,
            'criticality': ColorScheme.CRITICALITY,
            'layer': ColorScheme.LAYER,
            'custom': ColorScheme.CUSTOM
        }
        return mapping.get(scheme.lower(), ColorScheme.CRITICALITY)
    
    def _detect_hidden_dependencies(self,
                                   graph: nx.DiGraph,
                                   threshold: float) -> Dict[str, List]:
        """
        Detect hidden dependencies through indirect paths
        
        Returns dict with:
        - hidden_edges: List of (source, target) tuples
        - critical_nodes: Nodes involved in hidden dependencies
        """
        hidden_edges = []
        critical_nodes = set()
        
        # Find all nodes with high criticality
        critical_threshold = threshold
        
        for node in graph.nodes():
            score = graph.nodes[node].get('criticality_score', 0)
            if score >= critical_threshold:
                critical_nodes.add(node)
        
        # Detect indirect dependencies (paths of length > 1)
        for source in graph.nodes():
            for target in graph.nodes():
                if source != target and not graph.has_edge(source, target):
                    # Check if there's an indirect path
                    try:
                        if nx.has_path(graph, source, target):
                            paths = list(nx.all_simple_paths(
                                graph, source, target, cutoff=3
                            ))
                            
                            # If shortest path is > 1, it's a hidden dependency
                            if paths and len(min(paths, key=len)) > 2:
                                hidden_edges.append((source, target))
                                critical_nodes.update([source, target])
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
        
        return {
            'hidden_edges': hidden_edges,
            'critical_nodes': list(critical_nodes)
        }
    
    def _generate_dependency_report(self,
                                   hidden_deps: Dict,
                                   output_path: Path) -> None:
        """Generate text report for hidden dependencies"""
        report_path = output_path.with_suffix('.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("HIDDEN DEPENDENCIES ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            f.write(f"Critical Nodes: {len(hidden_deps['critical_nodes'])}\n")
            for node in hidden_deps['critical_nodes']:
                f.write(f"  - {node}\n")
            
            f.write(f"\nHidden Dependencies: {len(hidden_deps['hidden_edges'])}\n")
            for source, target in hidden_deps['hidden_edges']:
                f.write(f"  - {source} ⇢ {target} (indirect)\n")
    
    def _create_impact_visualization(self,
                                    graph: nx.DiGraph,
                                    result: Any,
                                    component_id: str,
                                    output_path: str) -> None:
        """Create detailed impact visualization with metrics overlay"""
        
        # Create HTML with embedded metrics
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Failure Impact Analysis: {component}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #e74c3c;
                    padding-bottom: 10px;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: #ecf0f1;
                    padding: 15px;
                    border-radius: 6px;
                    border-left: 4px solid #3498db;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-top: 5px;
                }}
                .critical {{
                    border-left-color: #e74c3c;
                }}
                .warning {{
                    border-left-color: #f39c12;
                }}
                .success {{
                    border-left-color: #27ae60;
                }}
                .affected-list {{
                    background: #fff3cd;
                    padding: 15px;
                    border-radius: 6px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>⚠️ Failure Impact Analysis: {component}</h1>
                
                <div class="metrics">
                    <div class="metric-card critical">
                        <div class="metric-value">{impact_score:.1%}</div>
                        <div class="metric-label">Impact Score</div>
                    </div>
                    <div class="metric-card warning">
                        <div class="metric-value">{affected_count}</div>
                        <div class="metric-label">Affected Components</div>
                    </div>
                    <div class="metric-card warning">
                        <div class="metric-value">{cascade_count}</div>
                        <div class="metric-label">Cascade Failures</div>
                    </div>
                    <div class="metric-card success">
                        <div class="metric-value">{resilience:.1%}</div>
                        <div class="metric-label">Resilience Score</div>
                    </div>
                </div>
                
                <div class="affected-list">
                    <h3>Affected Components</h3>
                    <ul>
                        {affected_items}
                    </ul>
                </div>
                
                <p><strong>Analysis Time:</strong> {timestamp}</p>
            </div>
        </body>
        </html>
        """
        
        affected_items = '\n'.join([
            f'<li>{comp}</li>' for comp in result.affected_components
        ])
        
        html_content = html_template.format(
            component=component_id,
            impact_score=result.impact_score,
            affected_count=len(result.affected_components),
            cascade_count=len(result.cascade_failures),
            resilience=result.resilience_score,
            affected_items=affected_items,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        Path(output_path).write_text(html_content)
    
    def _visualize_spof(self, graph: nx.DiGraph) -> str:
        """Visualize single points of failure"""
        self.logger.info("  Identifying single points of failure...")
        
        # Find articulation points
        if graph.is_directed():
            undirected = graph.to_undirected()
        else:
            undirected = graph
        
        spofs = list(nx.articulation_points(undirected))
        
        # Create visualization highlighting SPOFs
        config = VisualizationConfig(
            layout=LayoutAlgorithm.SPRING,
            color_scheme=ColorScheme.CRITICALITY,
            show_labels=True,
            highlight_nodes=spofs,
            node_colors={node: '#e74c3c' for node in spofs}
        )
        
        output_path = self.output_dir / "scenario_spof.html"
        self.graph_visualizer.visualize(graph, config, str(output_path))
        
        self.logger.info(f"    Found {len(spofs)} single points of failure")
        return str(output_path)
    
    def _visualize_cascade_risk(self, graph: nx.DiGraph) -> str:
        """Visualize cascade failure risk"""
        self.logger.info("  Analyzing cascade failure risk...")
        
        # Calculate betweenness centrality (high values = cascade risk)
        betweenness = nx.betweenness_centrality(graph)
        high_risk = [node for node, score in betweenness.items() 
                    if score > 0.1]
        
        config = VisualizationConfig(
            layout=LayoutAlgorithm.SPRING,
            color_scheme=ColorScheme.CRITICALITY,
            show_labels=True,
            highlight_nodes=high_risk,
            node_colors={node: '#f39c12' for node in high_risk}
        )
        
        output_path = self.output_dir / "scenario_cascade.html"
        self.graph_visualizer.visualize(graph, config, str(output_path))
        
        self.logger.info(f"    Identified {len(high_risk)} high-risk nodes")
        return str(output_path)
    
    def _visualize_multi_failure(self, graph: nx.DiGraph) -> str:
        """Visualize multiple concurrent failures"""
        self.logger.info("  Simulating multiple failures...")
        
        # Select multiple high-criticality components
        nodes_by_criticality = sorted(
            graph.nodes(),
            key=lambda n: graph.nodes[n].get('criticality_score', 0),
            reverse=True
        )
        
        failed_nodes = nodes_by_criticality[:min(3, len(nodes_by_criticality))]
        
        config = VisualizationConfig(
            layout=LayoutAlgorithm.SPRING,
            color_scheme=ColorScheme.TYPE,
            show_labels=True,
            highlight_nodes=failed_nodes,
            node_colors={node: '#c0392b' for node in failed_nodes}
        )
        
        output_path = self.output_dir / "scenario_multi_failure.html"
        self.graph_visualizer.visualize(graph, config, str(output_path))
        
        return str(output_path)
    
    def _visualize_recovery_paths(self, graph: nx.DiGraph) -> str:
        """Visualize recovery paths and redundancy"""
        self.logger.info("  Analyzing recovery paths...")
        
        # Find nodes with multiple paths (redundancy)
        redundant_nodes = []
        
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            if in_degree > 1:
                redundant_nodes.append(node)
        
        config = VisualizationConfig(
            layout=LayoutAlgorithm.HIERARCHICAL,
            color_scheme=ColorScheme.TYPE,
            show_labels=True,
            highlight_nodes=redundant_nodes,
            node_colors={node: '#27ae60' for node in redundant_nodes}
        )
        
        output_path = self.output_dir / "scenario_recovery.html"
        self.graph_visualizer.visualize(graph, config, str(output_path))
        
        self.logger.info(f"    Found {len(redundant_nodes)} redundant nodes")
        return str(output_path)
    
    def _collect_system_stats(self,
                             graph: nx.DiGraph,
                             graph_model: GraphModel) -> Dict[str, Any]:
        """Collect comprehensive system statistics"""
        
        # Node statistics by type
        node_types = {}
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Edge statistics
        edge_count = graph.number_of_edges()
        
        # Criticality statistics
        criticality_scores = [
            graph.nodes[node].get('criticality_score', 0)
            for node in graph.nodes()
        ]
        
        avg_criticality = sum(criticality_scores) / len(criticality_scores) if criticality_scores else 0
        
        # Connectivity
        if graph.is_directed():
            undirected = graph.to_undirected()
        else:
            undirected = graph
        
        is_connected = nx.is_connected(undirected)
        num_components = nx.number_connected_components(undirected)
        
        return {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': edge_count,
            'node_types': node_types,
            'avg_criticality': avg_criticality,
            'is_connected': is_connected,
            'num_components': num_components,
            'density': nx.density(graph),
            'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        }
    
    def _generate_html_report(self,
                             stats: Dict[str, Any],
                             outputs: Dict[str, Any]) -> str:
        """Generate HTML report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Visualization Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background: #ecf0f1;
                    padding: 15px;
                    border-radius: 6px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .stat-label {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-top: 5px;
                }}
                .file-list {{
                    list-style: none;
                    padding: 0;
                }}
                .file-list li {{
                    padding: 10px;
                    margin: 5px 0;
                    background: #f8f9fa;
                    border-left: 4px solid #3498db;
                    border-radius: 4px;
                }}
                .file-list a {{
                    color: #2980b9;
                    text-decoration: none;
                    font-weight: 500;
                }}
                .file-list a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Visualization Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>System Statistics</h2>
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value">{stats['total_nodes']}</div>
                        <div class="stat-label">Total Nodes</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats['total_edges']}</div>
                        <div class="stat-label">Total Edges</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats['avg_criticality']:.2f}</div>
                        <div class="stat-label">Avg Criticality</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats['density']:.3f}</div>
                        <div class="stat-label">Graph Density</div>
                    </div>
                </div>
                
                <h2>Node Distribution</h2>
                <ul class="file-list">
        """
        
        for node_type, count in stats['node_types'].items():
            html += f"<li>{node_type}: {count} nodes</li>\n"
        
        html += """
                </ul>
                
                <h2>Generated Visualizations</h2>
                <ul class="file-list">
        """
        
        for name, path in outputs.items():
            if isinstance(path, str):
                filename = Path(path).name
                html += f'<li><a href="{filename}">{name}</a> - {filename}</li>\n'
            elif isinstance(path, dict):
                html += f'<li><strong>{name}</strong><ul>\n'
                for subname, subpath in path.items():
                    if isinstance(subpath, str):
                        filename = Path(subpath).name
                        html += f'<li><a href="{filename}">{subname}</a> - {filename}</li>\n'
                html += '</ul></li>\n'
        
        html += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html


def setup_logging(verbose: bool = False) -> None:
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('visualization.log', mode='w')
        ]
    )


def load_graph(args) -> Tuple[GraphModel, nx.DiGraph]:
    """Load graph from JSON or Neo4j"""
    logger = logging.getLogger(__name__)
    
    builder = GraphBuilder()
    
    if args.neo4j:
        logger.info(f"Loading graph from Neo4j: {args.uri}")
        auth = (args.user, args.password)
        model = builder.build_from_neo4j(
            uri=args.uri,
            auth=auth,
            database=args.database
        )
    else:
        logger.info(f"Loading graph from file: {args.input}")
        model = builder.build_from_json(args.input)
    
    # Export to NetworkX
    exporter = GraphExporter()
    graph = exporter.export_to_networkx(model)
    
    # Run analysis to get criticality scores
    orchestrator = AnalysisOrchestrator(
        output_dir=str(Path(args.output_dir) / "analysis"),
        enable_qos=True
    )
    
    logger.info("Running analysis to compute metrics...")
    analysis_results = orchestrator.analyze_graph(
        graph=graph,
        graph_model=model,
        enable_simulation=args.simulate_failure is not None
    )
    
    # Update graph with analysis results
    for node_id, score_data in analysis_results['criticality_scores'].items():
        if node_id in graph.nodes:
            graph.nodes[node_id]['criticality_score'] = score_data['composite_score']
            graph.nodes[node_id]['criticality_level'] = score_data['criticality_level']
    
    logger.info(f"✓ Graph loaded: {len(graph)} nodes, {len(graph.edges())} edges")
    
    return model, graph


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Visualize pub-sub system graphs with comprehensive analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                            help='Input JSON file')
    input_group.add_argument('--neo4j', action='store_true',
                            help='Load from Neo4j database')
    
    # Neo4j options
    parser.add_argument('--uri', type=str,
                       help='Neo4j URI (e.g., bolt://localhost:7687)')
    parser.add_argument('--user', type=str, default='neo4j',
                       help='Neo4j username')
    parser.add_argument('--password', type=str,
                       help='Neo4j password')
    parser.add_argument('--database', type=str, default='neo4j',
                       help='Neo4j database name')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    
    # Visualization options
    parser.add_argument('--layout', type=str, default='spring',
                       choices=['spring', 'force-directed', 'hierarchical', 'circular'],
                       help='Layout algorithm')
    parser.add_argument('--color-scheme', type=str, default='criticality',
                       choices=['type', 'criticality', 'layer'],
                       help='Color scheme')
    parser.add_argument('--no-labels', action='store_true',
                       help='Hide node labels')
    
    # Visualization types
    parser.add_argument('--all', action='store_true',
                       help='Generate all visualization types')
    parser.add_argument('--complete', action='store_true',
                       help='Complete system visualization')
    parser.add_argument('--multi-layer', action='store_true',
                       help='Multi-layer visualizations')
    parser.add_argument('--hidden-deps', action='store_true',
                       help='Hidden dependencies analysis')
    parser.add_argument('--dashboard', action='store_true',
                       help='Create metrics dashboard')
    
    # Failure analysis
    parser.add_argument('--simulate-failure', type=str,
                       help='Simulate failure of specific component')
    parser.add_argument('--cascade', action='store_true',
                       help='Enable cascading failures')
    parser.add_argument('--failure-scenarios', nargs='+',
                       choices=['spof', 'cascade', 'multi', 'recovery'],
                       help='Generate failure scenario visualizations')
    
    # Analysis options
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Threshold for hidden dependency detection')
    
    # General options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate Neo4j arguments
    if args.neo4j and not all([args.uri, args.password]):
        parser.error("--neo4j requires --uri and --password")
    
    try:
        start_time = time.time()
        
        # Load graph
        logger.info("=" * 70)
        logger.info("GRAPH VISUALIZATION - STARTING")
        logger.info("=" * 70)
        
        model, graph = load_graph(args)
        
        # Initialize orchestrator
        output_dir = Path(args.output_dir)
        orchestrator = VisualizationOrchestrator(output_dir)
        
        outputs = {}
        
        # Generate visualizations based on arguments
        if args.all or args.complete:
            outputs['complete'] = orchestrator.visualize_complete_system(
                graph, model,
                layout=args.layout,
                color_scheme=args.color_scheme,
                show_labels=not args.no_labels
            )
        
        if args.all or args.multi_layer:
            outputs['layers'] = orchestrator.visualize_multi_layer(
                graph, include_interactions=True
            )
        
        if args.all or args.hidden_deps:
            outputs['hidden_deps'] = orchestrator.visualize_hidden_dependencies(
                graph, threshold=args.threshold
            )
        
        if args.all or args.dashboard:
            outputs['dashboard'] = orchestrator.create_metrics_dashboard(
                graph, include_health=True
            )
        
        if args.simulate_failure:
            outputs['failure_impact'] = orchestrator.visualize_failure_impact(
                graph, args.simulate_failure, enable_cascade=args.cascade
            )
        
        if args.failure_scenarios:
            outputs['scenarios'] = orchestrator.visualize_failure_scenarios(
                graph, args.failure_scenarios
            )
        
        # Generate report
        if outputs:
            report_path = orchestrator.generate_visualization_report(
                graph, model, outputs
            )
            outputs['report'] = report_path
        
        # Summary
        elapsed = time.time() - start_time
        
        logger.info("=" * 70)
        logger.info("VISUALIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"⏱️  Execution Time: {elapsed:.2f}s")
        logger.info(f"📁 Output Directory: {output_dir}")
        logger.info(f"📊 Generated Files: {len([v for v in outputs.values() if isinstance(v, str)])}")
        
        logger.info("\n📈 Visualizations:")
        for name, path in outputs.items():
            if isinstance(path, str):
                logger.info(f"  ✓ {name}: {Path(path).name}")
            elif isinstance(path, dict):
                logger.info(f"  ✓ {name}:")
                for subname in path.keys():
                    logger.info(f"      - {subname}")
        
        logger.info("\n💡 Next Steps:")
        logger.info("  - Open HTML files in web browser")
        logger.info("  - Review metrics dashboard for system health")
        logger.info("  - Analyze hidden dependencies report")
        logger.info("  - Use failure scenarios for resilience planning")
        
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
