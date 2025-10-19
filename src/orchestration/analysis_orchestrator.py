"""
Analysis Orchestrator

Unified pipeline for comprehensive pub-sub system analysis combining:
- Graph modeling and construction
- Multi-metric centrality analysis
- QoS-aware criticality scoring
- Failure simulation and impact assessment
- Multi-layer dependency analysis
- Visualization and reporting
"""

import networkx as nx
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import json
import pandas as pd

from ..core.graph_model import GraphModel
from ..analysis.criticality_scorer import CompositeCriticalityScorer, CompositeCriticalityScore
from ..analysis.qos_analyzer import QoSAnalyzer
from ..analysis.reachability_analyzer import ReachabilityAnalyzer
from ..analysis.structural_analyzer import StructuralAnalyzer


class AnalysisOrchestrator:
    """
    Orchestrates comprehensive analysis of pub-sub systems
    
    This class provides a unified interface for:
    1. Graph construction and validation
    2. Multi-metric analysis
    3. Criticality assessment
    4. Failure simulation
    5. Report generation
    """
    
    def __init__(self, 
                 output_dir: str = "output/",
                 enable_qos: bool = True,
                 criticality_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the orchestrator
        
        Args:
            output_dir: Directory for output files
            enable_qos: Enable QoS-aware analysis
            criticality_weights: Custom weights for criticality scoring
                               (alpha, beta, gamma for betweenness, AP, impact)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_qos = enable_qos
        
        # Initialize analyzers
        if criticality_weights:
            self.criticality_scorer = CompositeCriticalityScorer(
                alpha=criticality_weights.get('alpha', 0.4),
                beta=criticality_weights.get('beta', 0.3),
                gamma=criticality_weights.get('gamma', 0.3),
                qos_enabled=enable_qos
            )
        else:
            self.criticality_scorer = CompositeCriticalityScorer(qos_enabled=enable_qos)
        
        self.qos_analyzer = QoSAnalyzer() if enable_qos else None
        self.structural_analyzer = StructuralAnalyzer()
        
        # Results storage
        self.results = {
            'timestamp': None,
            'graph_summary': {},
            'criticality_scores': {},
            'structural_analysis': {},
            'qos_analysis': {},
            'layer_analysis': {},
            'failure_simulation': {},
            'recommendations': [],
            'execution_time': {}
        }
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_graph(self, 
                     graph: nx.DiGraph,
                     graph_model: Optional[GraphModel] = None,
                     enable_simulation: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive analysis on the graph
        
        Args:
            graph: NetworkX directed graph to analyze
            graph_model: Optional GraphModel with enhanced metadata
            enable_simulation: Whether to run failure simulations
        
        Returns:
            Dictionary with complete analysis results
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Comprehensive Graph Analysis")
        self.logger.info("=" * 60)
        
        total_start = time.time()
        self.results['timestamp'] = pd.Timestamp.now().isoformat()
        
        # Step 1: Graph Summary
        self.logger.info("\n[1/7] Analyzing graph structure...")
        step_start = time.time()
        self.results['graph_summary'] = self._analyze_graph_structure(graph)
        self.results['execution_time']['graph_summary'] = time.time() - step_start
        self.logger.info(f"  ✓ Completed in {self.results['execution_time']['graph_summary']:.2f}s")
        
        # Step 2: QoS Analysis (if enabled)
        if self.enable_qos and self.qos_analyzer:
            self.logger.info("\n[2/7] Analyzing QoS policies...")
            step_start = time.time()
            self.results['qos_analysis'] = self.qos_analyzer.analyze_graph(graph, graph_model)
            self.results['execution_time']['qos_analysis'] = time.time() - step_start
            self.logger.info(f"  ✓ Completed in {self.results['execution_time']['qos_analysis']:.2f}s")
        else:
            self.logger.info("\n[2/7] Skipping QoS analysis (disabled)")
            self.results['qos_analysis'] = {}
        
        # Step 3: Structural Analysis
        self.logger.info("\n[3/7] Performing structural analysis...")
        step_start = time.time()
        self.results['structural_analysis'] = self.structural_analyzer.analyze(graph)
        self.results['execution_time']['structural_analysis'] = time.time() - step_start
        self.logger.info(f"  ✓ Completed in {self.results['execution_time']['structural_analysis']:.2f}s")
        
        # Step 4: Criticality Scoring
        self.logger.info("\n[4/7] Calculating criticality scores...")
        step_start = time.time()
        qos_scores = self.results['qos_analysis'].get('component_scores', {}) if self.enable_qos else None
        criticality_scores = self.criticality_scorer.calculate_all_scores(graph, qos_scores)
        self.results['criticality_scores'] = self._format_criticality_results(criticality_scores)
        self.results['execution_time']['criticality_scoring'] = time.time() - step_start
        self.logger.info(f"  ✓ Completed in {self.results['execution_time']['criticality_scoring']:.2f}s")
        
        # Step 5: Layer Analysis
        self.logger.info("\n[5/7] Analyzing system layers...")
        step_start = time.time()
        self.results['layer_analysis'] = self._analyze_layers(graph, criticality_scores)
        self.results['execution_time']['layer_analysis'] = time.time() - step_start
        self.logger.info(f"  ✓ Completed in {self.results['execution_time']['layer_analysis']:.2f}s")
        
        # Step 6: Failure Simulation (if enabled)
        if enable_simulation:
            self.logger.info("\n[6/7] Running failure simulations...")
            step_start = time.time()
            self.results['failure_simulation'] = self._simulate_failures(graph, criticality_scores)
            self.results['execution_time']['failure_simulation'] = time.time() - step_start
            self.logger.info(f"  ✓ Completed in {self.results['execution_time']['failure_simulation']:.2f}s")
        else:
            self.logger.info("\n[6/7] Skipping failure simulation (disabled)")
            self.results['failure_simulation'] = {}
        
        # Step 7: Generate Recommendations
        self.logger.info("\n[7/7] Generating recommendations...")
        step_start = time.time()
        self.results['recommendations'] = self._generate_recommendations(
            graph, 
            criticality_scores,
            self.results['structural_analysis'],
            self.results['qos_analysis']
        )
        self.results['execution_time']['recommendations'] = time.time() - step_start
        self.logger.info(f"  ✓ Completed in {self.results['execution_time']['recommendations']:.2f}s")
        
        # Total time
        total_time = time.time() - total_start
        self.results['execution_time']['total'] = total_time
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"Analysis Complete in {total_time:.2f}s")
        self.logger.info("=" * 60)
        
        return self.results
    
    def _analyze_graph_structure(self, graph: nx.DiGraph) -> Dict:
        """Analyze basic graph structure"""
        summary = {
            'total_nodes': len(graph),
            'total_edges': len(graph.edges()),
            'density': nx.density(graph),
            'is_connected': nx.is_weakly_connected(graph),
            'num_connected_components': nx.number_weakly_connected_components(graph)
        }
        
        # Count by type
        node_types = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        summary['node_types'] = node_types
        
        # Count edge types
        edge_types = {}
        for _, _, data in graph.edges(data=True):
            edge_type = data.get('type', 'Unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        summary['edge_types'] = edge_types
        
        # Degree statistics
        degrees = [d for n, d in graph.degree()]
        summary['avg_degree'] = sum(degrees) / len(degrees) if degrees else 0
        summary['max_degree'] = max(degrees) if degrees else 0
        summary['min_degree'] = min(degrees) if degrees else 0
        
        return summary
    
    def _format_criticality_results(self, 
                                    scores: Dict[str, CompositeCriticalityScore]) -> Dict:
        """Format criticality scores for output"""
        # Summary statistics
        summary = self.criticality_scorer.summarize_criticality(scores)
        
        # Top critical components
        top_critical = self.criticality_scorer.get_top_critical(scores, n=20)
        
        # Format results
        return {
            'summary': summary,
            'top_critical_components': [
                {
                    'component': s.component,
                    'type': s.component_type,
                    'composite_score': round(s.composite_score, 3),
                    'level': s.criticality_level.value,
                    'betweenness': round(s.betweenness_centrality_norm, 3),
                    'is_articulation_point': bool(s.is_articulation_point),
                    'impact_score': round(s.impact_score, 3),
                    'components_affected': s.components_affected,
                    'reachability_loss_pct': round(s.reachability_loss_percentage, 1)
                }
                for s in top_critical
            ],
            'all_scores': {
                node: {
                    'composite_score': round(score.composite_score, 3),
                    'level': score.criticality_level.value,
                    'betweenness': round(score.betweenness_centrality_norm, 3),
                    'is_articulation_point': bool(score.is_articulation_point),
                    'impact_score': round(score.impact_score, 3)
                }
                for node, score in scores.items()
            }
        }
    
    def _analyze_layers(self, 
                       graph: nx.DiGraph,
                       criticality_scores: Dict[str, CompositeCriticalityScore]) -> Dict:
        """Analyze different system layers"""
        layers = {
            'application_layer': self._analyze_application_layer(graph, criticality_scores),
            'infrastructure_layer': self._analyze_infrastructure_layer(graph, criticality_scores),
            'cross_layer_dependencies': self._analyze_cross_layer(graph)
        }
        return layers
    
    def _analyze_application_layer(self, 
                                   graph: nx.DiGraph,
                                   criticality_scores: Dict[str, CompositeCriticalityScore]) -> Dict:
        """Analyze application-level dependencies"""
        # Extract application nodes
        apps = [n for n, d in graph.nodes(data=True) if d.get('type') == 'Application']
        
        # Create application dependency subgraph
        app_graph = nx.DiGraph()
        for app in apps:
            app_graph.add_node(app, **graph.nodes[app])
        
        # Add DEPENDS_ON edges
        for source, target, data in graph.edges(data=True):
            if data.get('type') == 'DEPENDS_ON':
                if source in apps and target in apps:
                    app_graph.add_edge(source, target, **data)
        
        # Analyze
        critical_apps = [
            s for s in criticality_scores.values()
            if s.component_type == 'Application' and s.composite_score >= 0.6
        ]
        
        return {
            'total_applications': len(apps),
            'total_dependencies': len(app_graph.edges()),
            'critical_applications': len(critical_apps),
            'top_critical': [
                {'name': s.component, 'score': round(s.composite_score, 3)}
                for s in sorted(critical_apps, key=lambda x: x.composite_score, reverse=True)[:5]
            ]
        }
    
    def _analyze_infrastructure_layer(self, 
                                      graph: nx.DiGraph,
                                      criticality_scores: Dict[str, CompositeCriticalityScore]) -> Dict:
        """Analyze infrastructure-level connectivity"""
        # Extract nodes and brokers
        nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'Node']
        brokers = [n for n, d in graph.nodes(data=True) if d.get('type') == 'Broker']
        
        # Critical infrastructure
        critical_nodes = [
            s for s in criticality_scores.values()
            if s.component_type in ['Node', 'Broker'] and s.composite_score >= 0.6
        ]
        
        return {
            'total_nodes': len(nodes),
            'total_brokers': len(brokers),
            'critical_infrastructure': len(critical_nodes),
            'top_critical': [
                {'name': s.component, 'type': s.component_type, 'score': round(s.composite_score, 3)}
                for s in sorted(critical_nodes, key=lambda x: x.composite_score, reverse=True)[:5]
            ]
        }
    
    def _analyze_cross_layer(self, graph: nx.DiGraph) -> Dict:
        """Analyze cross-layer dependencies"""
        # Count mappings between layers
        app_to_node = {}
        broker_to_node = {}
        
        for source, target, data in graph.edges(data=True):
            if data.get('type') == 'RUNS_ON':
                source_type = graph.nodes[source].get('type')
                if source_type == 'Application':
                    app_to_node[source] = target
                elif source_type == 'Broker':
                    broker_to_node[source] = target
        
        # Identify nodes hosting multiple critical components
        node_component_count = {}
        for node in set(list(app_to_node.values()) + list(broker_to_node.values())):
            apps = [a for a, n in app_to_node.items() if n == node]
            brokers = [b for b, n in broker_to_node.items() if n == node]
            node_component_count[node] = len(apps) + len(brokers)
        
        # Find nodes with high concentration
        high_concentration = [
            {'node': node, 'component_count': count}
            for node, count in node_component_count.items()
            if count > 3
        ]
        
        return {
            'applications_mapped': len(app_to_node),
            'brokers_mapped': len(broker_to_node),
            'high_concentration_nodes': high_concentration
        }
    
    def _simulate_failures(self, 
                          graph: nx.DiGraph,
                          criticality_scores: Dict[str, CompositeCriticalityScore]) -> Dict:
        """Simulate failures of critical components"""
        # Get top 10 critical components
        top_critical = self.criticality_scorer.get_top_critical(criticality_scores, n=10)
        
        simulations = []
        for score in top_critical:
            component = score.component
            
            # Simulate removal
            test_graph = graph.copy()
            test_graph.remove_node(component)
            
            # Measure impact
            original_components = nx.number_weakly_connected_components(graph)
            new_components = nx.number_weakly_connected_components(test_graph)
            
            simulation = {
                'component': component,
                'type': score.component_type,
                'criticality_score': round(score.composite_score, 3),
                'nodes_affected': score.components_affected,
                'reachability_loss_pct': round(score.reachability_loss_percentage, 1),
                'connectivity_impact': {
                    'before_components': original_components,
                    'after_components': new_components,
                    'fragments_created': new_components - original_components
                }
            }
            
            simulations.append(simulation)
        
        return {
            'simulations_run': len(simulations),
            'results': simulations
        }
    
    def _generate_recommendations(self, 
                                 graph: nx.DiGraph,
                                 criticality_scores: Dict[str, CompositeCriticalityScore],
                                 structural_analysis: Dict,
                                 qos_analysis: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # 1. Critical component recommendations
        critical_components = self.criticality_scorer.get_critical_components(
            criticality_scores, threshold=0.7
        )
        
        for score in critical_components[:5]:  # Top 5
            if score.is_articulation_point:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'type': 'Redundancy',
                    'component': score.component,
                    'issue': f'{score.component_type} is a single point of failure (articulation point)',
                    'recommendation': f'Add redundancy for {score.component} or redistribute its responsibilities',
                    'impact': f'Affects {score.components_affected} components',
                    'risk_reduction': 'High'
                })
            
            if score.impact_score > 0.5:
                recommendations.append({
                    'priority': 'HIGH',
                    'type': 'High Impact',
                    'component': score.component,
                    'issue': f'Failure would cause {score.reachability_loss_percentage:.0f}% reachability loss',
                    'recommendation': f'Implement fallback mechanisms for {score.component}',
                    'impact': f'{score.reachability_loss_percentage:.0f}% of system affected',
                    'risk_reduction': 'Medium-High'
                })
        
        # 2. Structural vulnerability recommendations
        if 'bridges' in structural_analysis and structural_analysis['bridges']:
            for bridge in structural_analysis['bridges'][:3]:
                recommendations.append({
                    'priority': 'HIGH',
                    'type': 'Network Vulnerability',
                    'component': f'{bridge[0]} ↔ {bridge[1]}',
                    'issue': 'Critical network link (bridge)',
                    'recommendation': 'Add alternative network path',
                    'impact': 'Network fragmentation risk',
                    'risk_reduction': 'High'
                })
        
        # 3. QoS recommendations (if available)
        if qos_analysis and 'high_priority_topics' in qos_analysis:
            for topic in qos_analysis['high_priority_topics'][:3]:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'type': 'QoS Optimization',
                    'component': topic['name'],
                    'issue': f'High-priority topic with QoS score {topic["score"]:.2f}',
                    'recommendation': 'Ensure dedicated resources and monitoring',
                    'impact': 'Service quality degradation risk',
                    'risk_reduction': 'Medium'
                })
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations
    
    def export_results(self, filename: Optional[str] = None) -> Path:
        """Export results to JSON file"""
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results exported to {filepath}")
        return filepath
    
    def print_summary(self):
        """Print analysis summary to console"""
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        
        # Graph summary
        gs = self.results['graph_summary']
        print(f"\nGraph Structure:")
        print(f"  Nodes: {gs['total_nodes']}, Edges: {gs['total_edges']}")
        print(f"  Node Types: {gs['node_types']}")
        print(f"  Connected Components: {gs['num_connected_components']}")
        
        # Criticality summary
        cs = self.results['criticality_scores']['summary']
        print(f"\nCriticality Distribution:")
        print(f"  Critical: {cs['critical_count']}")
        print(f"  High: {cs['high_count']}")
        print(f"  Medium: {cs['medium_count']}")
        print(f"  Low: {cs['low_count']}")
        print(f"  Minimal: {cs['minimal_count']}")
        print(f"  Articulation Points: {cs['articulation_points']}")
        
        # Top critical components
        print(f"\nTop 5 Critical Components:")
        for i, comp in enumerate(self.results['criticality_scores']['top_critical_components'][:5], 1):
            print(f"  {i}. {comp['component']} ({comp['type']})")
            print(f"     Score: {comp['composite_score']}, Level: {comp['level']}")
            print(f"     Affects {comp['components_affected']} components")
        
        # Recommendations
        print(f"\nTop 3 Recommendations:")
        for i, rec in enumerate(self.results['recommendations'][:3], 1):
            print(f"  {i}. [{rec['priority']}] {rec['type']}")
            print(f"     Component: {rec['component']}")
            print(f"     Issue: {rec['issue']}")
            print(f"     Action: {rec['recommendation']}")
        
        # Performance
        print(f"\nExecution Time:")
        print(f"  Total: {self.results['execution_time']['total']:.2f}s")
        
        print("=" * 70)
