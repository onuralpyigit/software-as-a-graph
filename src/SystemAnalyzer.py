import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
import json
from pathlib import Path
import networkx as nx

from src.GraphExporter import GraphExporter
from src.QosAwareComponentAnalyzer import QosAwareComponentAnalyzer
from src.GraphAnalyzer import GraphAnalyzer
from src.ReachabilityAnalyzer import ReachabilityAnalyzer
from src.GraphVisualizer import GraphVisualizer

class SystemAnalyzer:
    """
    System analyzer that orchestrates all analysis components
    for comprehensive pub-sub system evaluation
    """
    
    def __init__(self, output_dir: str = "output/"):
        """Initialize all analysis components"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.graph_exporter = GraphExporter()
        self.qos_analyzer = QosAwareComponentAnalyzer(self.graph_exporter)
        self.graph_analyzer = GraphAnalyzer(self.qos_analyzer, str(self.output_dir))
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        
        # Analysis results storage
        self.results = {
            'timestamp': None,
            'graph_summary': {},
            'criticality_analysis': {},
            'reachability_analysis': {},
            'layer_analysis': {},
            'recommendations': [],
            'execution_time': {}
        }
    
    def run_comprehensive_analysis(self, 
                                  enable_visualization: bool = False,
                                  enable_simulation: bool = True,
                                  export_results: bool = True) -> Dict[str, Any]:
        """
        Run complete system analysis pipeline
        
        Args:
            enable_visualization: Generate visualizations
            enable_simulation: Run failure simulations
            export_results: Export results to files
            
        Returns:
            Comprehensive analysis results
        """
        self.logger.info("Starting comprehensive system analysis...")
        self.results['timestamp'] = pd.Timestamp.now().isoformat()
        
        total_start = time.time()
        
        # Step 1: Export and prepare graph
        step_start = time.time()
        graph = self._prepare_graph()
        self.results['execution_time']['graph_preparation'] = time.time() - step_start
        
        # Step 2: Analyze graph structure and criticality
        step_start = time.time()
        self._analyze_structure_and_criticality(graph)
        self.results['execution_time']['criticality_analysis'] = time.time() - step_start
        
        # Step 3: Analyze layered dependencies
        step_start = time.time()
        self._analyze_layers(graph)
        self.results['execution_time']['layer_analysis'] = time.time() - step_start
        
        # Step 4: Run reachability and failure analysis
        if enable_simulation:
            step_start = time.time()
            self._analyze_reachability_and_failures(graph)
            self.results['execution_time']['reachability_analysis'] = time.time() - step_start
        
        # Step 5: Generate recommendations
        step_start = time.time()
        self._generate_system_recommendations()
        self.results['execution_time']['recommendations'] = time.time() - step_start
        
        # Step 6: Create visualizations
        if enable_visualization:
            step_start = time.time()
            self._create_visualizations(graph)
            self.results['execution_time']['visualization'] = time.time() - step_start
        
        # Step 7: Export results
        if export_results:
            self._export_results()
        
        self.results['execution_time'] = time.time() - total_start
        
        self.logger.info(f"Analysis completed in {self.results['execution_time']:.2f} seconds")
        
        return self.results
    
    def _prepare_graph(self) -> Any:
        """Export and prepare the graph for analysis"""
        self.logger.info("Exporting graph from Neo4j...")
        
        # Export main graph
        graph = self.graph_exporter.export_graph()
        
        # Store graph summary
        self.results['graph_summary'] = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'node_types': self._count_node_types(graph),
            'edge_types': self._count_edge_types(graph),
            'density': nx.density(graph),
            'is_connected': nx.is_weakly_connected(graph),
            'components': nx.number_weakly_connected_components(graph)
        }
        
        self.logger.info(f"Graph loaded: {self.results['graph_summary']['nodes']} nodes, "
                        f"{self.results['graph_summary']['edges']} edges")
        
        return graph
    
    def _count_node_types(self, graph) -> Dict[str, int]:
        """Count nodes by type"""
        type_counts = {}
        for _, data in graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
    
    def _count_edge_types(self, graph) -> Dict[str, int]:
        """Count edges by type"""
        type_counts = {}
        for _, _, data in graph.edges(data=True):
            edge_type = data.get('type', 'Unknown')
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        return type_counts
    
    def _analyze_structure_and_criticality(self, graph):
        """Analyze graph structure and component criticality"""
        self.logger.info("Analyzing graph structure and criticality...")
        
        # Run comprehensive criticality analysis
        criticality_df = self.graph_analyzer.analyze_comprehensive_criticality(graph)
        
        # Generate criticality report
        report = self.graph_analyzer.generate_criticality_report(criticality_df)
        
        # Store results
        self.results['criticality_analysis'] = {
            'summary': report['summary'],
            'critical_components': report['critical_components'][:10],  # Top 10
            'vulnerabilities': report['vulnerabilities'],
            'metrics_distribution': self._calculate_metrics_distribution(criticality_df)
        }
        
        # Store full DataFrame for later use
        self.criticality_df = criticality_df
    
    def _calculate_metrics_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate distribution statistics for metrics"""
        metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 
                  'pagerank', 'qos_score', 'composite_score']
        
        distribution = {}
        for metric in metrics:
            if metric in df.columns:
                distribution[metric] = {
                    'mean': float(df[metric].mean()),
                    'std': float(df[metric].std()),
                    'min': float(df[metric].min()),
                    'max': float(df[metric].max()),
                    'q25': float(df[metric].quantile(0.25)),
                    'q50': float(df[metric].quantile(0.50)),
                    'q75': float(df[metric].quantile(0.75))
                }
        
        return distribution
    
    def _analyze_layers(self, graph):
        """Analyze system layers separately"""
        self.logger.info("Analyzing system layers...")
        
        # Application layer
        app_graph = self.graph_exporter.export_graph_for_application_level_analysis()
        if app_graph.number_of_nodes() > 0:
            app_critical = self.graph_analyzer.analyze_comprehensive_criticality(app_graph)
            app_report = self.graph_analyzer.generate_criticality_report(app_critical)
            self.results['layer_analysis']['application'] = {
                'nodes': app_graph.number_of_nodes(),
                'edges': app_graph.number_of_edges(),
                'critical_count': len(app_report['critical_components'][:10]),
                'top_critical': [app_report['critical_components'][:10]]
            }
        
        # Infrastructure layer
        infra_graph = self.graph_exporter.export_graph_for_infrastructure_level_analysis()
        if infra_graph.number_of_nodes() > 0:
            infra_critical = self.graph_analyzer.analyze_comprehensive_criticality(infra_graph)
            infra_report = self.graph_analyzer.generate_criticality_report(infra_critical)
            self.results['layer_analysis']['infrastructure'] = {
                'nodes': infra_graph.number_of_nodes(),
                'edges': infra_graph.number_of_edges(),
                'critical_count': len(infra_report['critical_components'][:10]),
                'top_critical': [infra_report['critical_components'][:10]]
            }
        
        # Cross-layer dependencies
        self._analyze_cross_layer_dependencies(graph)
    
    def _analyze_cross_layer_dependencies(self, graph):
        """Analyze dependencies between layers"""
        cross_layer = {
            'node_to_app_mapping': {},
            'node_to_broker_mapping': {},
            'broker_to_topic_mapping': {},
            'critical_cross_layer_paths': []
        }
        
        # Map physical nodes to applications/brokers
        for node, node_data in graph.nodes(data=True):
            if node_data.get('type') == 'Node':
                apps = []
                brokers = []
                
                for source, _, edge_data in graph.in_edges(node, data=True):
                    if edge_data.get('type') == 'RUNS_ON':
                        source_type = graph.nodes[source].get('type')
                        if source_type == 'Application':
                            apps.append(source)
                        elif source_type == 'Broker':
                            brokers.append(source)
                
                if apps:
                    cross_layer['node_to_app_mapping'][node] = apps
                if brokers:
                    cross_layer['node_to_broker_mapping'][node] = brokers
        
        # Map brokers to topics
        for node, node_data in graph.nodes(data=True):
            if node_data.get('type') == 'Broker':
                topics = []
                for _, target, edge_data in graph.out_edges(node, data=True):
                    if edge_data.get('type') == 'ROUTES':
                        topics.append(target)
                
                if topics:
                    cross_layer['broker_to_topic_mapping'][node] = topics
        
        self.results['layer_analysis']['cross_layer'] = cross_layer
    
    def _analyze_reachability_and_failures(self, graph):
        """Analyze reachability and simulate failures"""
        self.logger.info("Analyzing reachability and simulating failures...")
        
        # Initialize reachability analyzer
        reachability_analyzer = ReachabilityAnalyzer(graph)
        
        # Compute baseline reachability
        baseline_matrix = reachability_analyzer.compute_reachability_matrix()
        
        # Run failure impact analysis for critical components
        critical_nodes = self.criticality_df.head(10)['node'].tolist()
        
        failure_impacts = []
        for node in critical_nodes:
            if node in graph:
                # Complete failure
                complete_matrix = reachability_analyzer.simulate_component_failure(
                    node, 'complete'
                )
                complete_impact = reachability_analyzer.calculate_reachability_impact(
                    baseline_matrix, complete_matrix
                )
                
                # Partial failure
                partial_matrix = reachability_analyzer.simulate_component_failure(
                    node, 'partial'
                )
                partial_impact = reachability_analyzer.calculate_reachability_impact(
                    baseline_matrix, partial_matrix
                )
                
                failure_impacts.append({
                    'component': node,
                    'type': graph.nodes[node].get('type'),
                    'complete_failure_impact': complete_impact['reachability_loss'],
                    'partial_failure_impact': partial_impact['reachability_loss'],
                    'isolated_apps_complete': complete_impact['isolated_apps'],
                    'isolated_apps_partial': partial_impact['isolated_apps']
                })
        
        # Identify critical paths
        critical_paths = reachability_analyzer.identify_critical_paths()
        
        self.results['reachability_analysis'] = {
            'baseline_connectivity': float(baseline_matrix.sum().sum() / 
                                          (len(baseline_matrix) * (len(baseline_matrix) - 1))),
            'failure_impacts': failure_impacts[:5],  # Top 5
            'critical_paths': critical_paths[:3],  # Top 3
            'resilience_score': self._calculate_resilience_score(failure_impacts)
        }
    
    def _calculate_resilience_score(self, failure_impacts: List[Dict]) -> float:
        """Calculate overall system resilience score"""
        if not failure_impacts:
            return 1.0
        
        # Average impact of failures (inverted for resilience)
        avg_complete_impact = np.mean([f['complete_failure_impact'] for f in failure_impacts])
        avg_partial_impact = np.mean([f['partial_failure_impact'] for f in failure_impacts])
        
        # Weighted resilience score (0-1, higher is better)
        resilience = 1.0 - (0.7 * avg_complete_impact/100 + 0.3 * avg_partial_impact/100)
        
        return float(max(0, min(1, resilience)))
    
    def _generate_system_recommendations(self):
        """Generate comprehensive system recommendations"""
        self.logger.info("Generating system recommendations...")
        
        recommendations = []
        
        # Based on criticality analysis
        if 'critical_components' in self.results['criticality_analysis']:
            critical = self.results['criticality_analysis']['critical_components']
            
            # Recommend redundancy for top critical components
            for comp in critical[:3]:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Redundancy',
                    'component': comp['node'],
                    'action': f"Add redundancy for {comp['type']} component",
                    'reason': f"Critical component with score {comp['composite_score']:.3f}",
                    'impact': 'Reduce single point of failure risk'
                })
        
        # Based on reachability analysis
        if 'failure_impacts' in self.results['reachability_analysis']:
            impacts = self.results['reachability_analysis']['failure_impacts']
            
            for impact in impacts:
                if impact['complete_failure_impact'] > 30:  # >30% impact
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': 'Failover',
                        'component': impact['component'],
                        'action': 'Implement automatic failover mechanism',
                        'reason': f"{impact['complete_failure_impact']:.1f}% reachability loss on failure",
                        'impact': f"Prevent isolation of {impact['isolated_apps_complete']} applications"
                    })
        
        # Based on layer analysis
        if 'cross_layer' in self.results['layer_analysis']:
            cross_layer = self.results['layer_analysis']['cross_layer']
            
            # Check for overloaded nodes
            for node, apps in cross_layer['node_to_app_mapping'].items():
                if len(apps) > 5:  # More than 5 apps on single node
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'category': 'Load Distribution',
                        'component': node,
                        'action': f"Redistribute {len(apps)} applications across multiple nodes",
                        'reason': 'High concentration of applications on single node',
                        'impact': 'Improve fault tolerance and load balancing'
                    })
        
        # Based on resilience score
        resilience = self.results['reachability_analysis'].get('resilience_score', 1.0)
        if resilience < 0.7:  # Low resilience
            recommendations.append({
                'priority': 'HIGH',
                'category': 'System Architecture',
                'component': 'System-wide',
                'action': 'Review and improve overall system architecture',
                'reason': f'Low system resilience score: {resilience:.2f}',
                'impact': 'Improve overall system fault tolerance'
            })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        self.results['recommendations'] = recommendations[:10]  # Top 10
    
    def _create_visualizations(self, graph):
        """Create comprehensive visualizations"""
        self.logger.info("Creating visualizations...")
        
        visualizer = GraphVisualizer()
        
        # Visualize main graph
        visualizer.visualize_graph(graph)
        
        # Visualize layers
        visualizer.visualize_layers(graph)
        
        # Create criticality heatmap
        if hasattr(self, 'criticality_df'):
            self._create_criticality_heatmap()
        
        # Create failure impact visualization
        if 'failure_impacts' in self.results['reachability_analysis']:
            self._create_failure_impact_chart()
    
    def _create_criticality_heatmap(self):
        """Create heatmap of criticality metrics"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Select top 20 components
        top_components = self.criticality_df.head(20)
        
        # Select metrics for heatmap
        metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 
                  'pagerank', 'qos_score']
        
        # Create heatmap data
        heatmap_data = top_components[metrics].T
        heatmap_data.columns = top_components['node'].values
        
        # Create figure
        plt.figure(figsize=(15, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Score'})
        plt.title('Component Criticality Metrics Heatmap')
        plt.xlabel('Components')
        plt.ylabel('Metrics')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'criticality_heatmap.png', dpi=300)
        plt.close()
    
    def _create_failure_impact_chart(self):
        """Create failure impact comparison chart"""
        import matplotlib.pyplot as plt
        
        impacts = self.results['reachability_analysis']['failure_impacts']
        
        if not impacts:
            return
        
        # Prepare data
        components = [i['component'] for i in impacts]
        complete_impacts = [i['complete_failure_impact'] for i in impacts]
        partial_impacts = [i['partial_failure_impact'] for i in impacts]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(components))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, complete_impacts, width, label='Complete Failure', 
                      color='#e74c3c')
        bars2 = ax.bar(x + width/2, partial_impacts, width, label='Partial Failure', 
                      color='#f39c12')
        
        ax.set_xlabel('Component')
        ax.set_ylabel('Reachability Loss (%)')
        ax.set_title('Failure Impact Analysis: Complete vs Partial Failures')
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'failure_impact_comparison.png', dpi=300)
        plt.close()
    
    def _export_results(self):
        """Export analysis results to files"""
        self.logger.info("Exporting results...")
        
        # Export JSON report
        json_path = self.output_dir / 'analysis_report.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Export criticality DataFrame to CSV
        if hasattr(self, 'criticality_df'):
            csv_path = self.output_dir / 'criticality_analysis.csv'
            self.criticality_df.to_csv(csv_path, index=False)
        
        # Generate markdown report
        self._generate_markdown_report()
        
        self.logger.info(f"Results exported to {self.output_dir}")
    
    def _generate_markdown_report(self):
        """Generate a human-readable markdown report"""
        report_path = self.output_dir / 'analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Publish-Subscribe System Analysis Report\n\n")
            f.write(f"Generated: {self.results['timestamp']}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Components**: {self.results['graph_summary']['nodes']}\n")
            f.write(f"- **Total Connections**: {self.results['graph_summary']['edges']}\n")
            f.write(f"- **System Resilience Score**: "
                   f"{self.results['reachability_analysis'].get('resilience_score', 'N/A'):.2f}\n")
            f.write(f"- **Critical Components**: "
                   f"{len(self.results['criticality_analysis']['critical_components'])}\n")
            #f.write(f"- **Analysis Time**: {self.results['execution_time']:.2f} seconds\n\n")
            
            # Critical Components
            f.write("## Critical Components\n\n")
            for comp in self.results['criticality_analysis']['critical_components'][:5]:
                f.write(f"### {comp['node']} ({comp['type']})\n")
                f.write(f"- **Criticality Level**: {comp['criticality_level']}\n")
                f.write(f"- **Score**: {comp['composite_score']:.3f}\n")
                f.write(f"- **Key Factors**: {', '.join(comp['key_factors'])}\n\n")
            
            # Recommendations
            f.write("## Priority Recommendations\n\n")
            for i, rec in enumerate(self.results['recommendations'][:5], 1):
                f.write(f"### {i}. {rec['category']} - {rec['component']}\n")
                f.write(f"- **Priority**: {rec['priority']}\n")
                f.write(f"- **Action**: {rec['action']}\n")
                f.write(f"- **Reason**: {rec['reason']}\n")
                f.write(f"- **Expected Impact**: {rec['impact']}\n\n")
            
            # System Vulnerabilities
            f.write("## Identified Vulnerabilities\n\n")
            for vuln in self.results['criticality_analysis'].get('vulnerabilities', []):
                f.write(f"- **{vuln['type']}** ({vuln['severity']}): {vuln['description']}\n")