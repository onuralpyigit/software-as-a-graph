import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
import logging

from src.ThresholdCalculator import ThresholdCalculator, CriticalityLevel
from src.QosAwareComponentAnalyzer import QosAwareComponentAnalyzer

@dataclass
class CriticalityMetrics:
    """Comprehensive metrics for component criticality assessment"""
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    pagerank: float
    eigenvector_centrality: float
    is_articulation_point: bool
    is_bridge_endpoint: bool
    qos_score: float
    dependency_count: int
    reachability_impact: float
    composite_score: float
    criticality_level: CriticalityLevel

class GraphAnalyzer:
    """Graph Analyzer for Distributed Publish-Subscribe Systems"""
    
    def __init__(self, component_analyzer: QosAwareComponentAnalyzer, output_dir: str = "output/"):
        self.component_analyzer = component_analyzer
        self.output_file_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Configurable weights for composite criticality score
        self.criticality_weights = {
            'degree_centrality': 0.10,
            'betweenness_centrality': 0.20,
            'closeness_centrality': 0.10,
            'pagerank': 0.15,
            'eigenvector_centrality': 0.10,
            'articulation_point': 0.15,
            'qos_score': 0.20
        }
        
        # Cache for expensive computations
        self._centrality_cache = {}
        self._reachability_cache = {}
    
    def analyze_comprehensive_criticality(self, graph: nx.DiGraph) -> pd.DataFrame:
        """
        Perform comprehensive criticality analysis combining all metrics
        
        Args:
            graph: NetworkX directed graph of the pub-sub system
            
        Returns:
            DataFrame with comprehensive criticality metrics for all components
        """
        self.logger.info("Starting comprehensive criticality analysis...")

        if len(graph) == 0:
            self.logger.warning("Empty graph provided for analysis")
            return pd.DataFrame(columns=['criticality_level'])
        
        # Clear caches for new analysis
        self._centrality_cache.clear()
        self._reachability_cache.clear()
        
        # Compute all centrality metrics
        centrality_metrics = self._compute_all_centralities(graph)
        
        # Identify structural vulnerabilities
        articulation_points = self._find_articulation_points(graph)
        bridge_endpoints = self._find_bridge_endpoints(graph)
        
        # Compute QoS scores based on component type
        qos_scores = self._compute_qos_scores_for_graph(graph)
        
        # Compute dependency metrics
        dependency_metrics = self._compute_dependency_metrics(graph)
        
        # Compute reachability impact (sampled for performance)
        reachability_impacts = self._compute_reachability_impacts(graph, sample_size=min(20, len(graph)))
        
        # Combine all metrics into DataFrame
        results = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            
            # Create comprehensive metrics
            metrics = CriticalityMetrics(
                degree_centrality=centrality_metrics['degree'][node],
                betweenness_centrality=centrality_metrics['betweenness'][node],
                closeness_centrality=centrality_metrics['closeness'][node],
                pagerank=centrality_metrics['pagerank'][node],
                eigenvector_centrality=centrality_metrics.get('eigenvector', {}).get(node, 0),
                is_articulation_point=node in articulation_points,
                is_bridge_endpoint=node in bridge_endpoints,
                qos_score=qos_scores.get(node, 0),
                dependency_count=dependency_metrics.get(node, 0),
                reachability_impact=reachability_impacts.get(node, 0),
                composite_score=0,  # Will be calculated
                criticality_level=CriticalityLevel.LOW  # Will be updated
            )
            
            # Calculate composite criticality score
            metrics.composite_score = self._calculate_composite_score(metrics)
            
            # Add to results
            results.append({
                'node': node,
                'type': node_data.get('type', 'Unknown'),
                'name': node_data.get('name', node),
                **metrics.__dict__
            })
        
        # Convert to DataFrame and classify criticality levels
        df = pd.DataFrame(results)
        df = self._classify_criticality_levels(df)
        
        # Sort by composite score
        df = df.sort_values('composite_score', ascending=False)
        
        self.logger.info(f"Analyzed {len(df)} components")
        return df
    
    def _compute_all_centralities(self, graph: nx.DiGraph) -> Dict[str, Dict]:
        """Compute all centrality metrics efficiently with caching"""
        
        if 'centralities' in self._centrality_cache:
            return self._centrality_cache['centralities']
        
        self.logger.debug("Computing centrality metrics...")
        
        metrics = {
            'degree': nx.degree_centrality(graph),
            'betweenness': nx.betweenness_centrality(graph, normalized=True),
            'closeness': nx.closeness_centrality(graph),
            'pagerank': nx.pagerank(graph, max_iter=100)
        }
        
        # Eigenvector centrality may fail for some graphs
        try:
            metrics['eigenvector'] = nx.eigenvector_centrality(graph, max_iter=1000)
        except:
            self.logger.warning("Eigenvector centrality computation failed, using PageRank as fallback")
            metrics['eigenvector'] = metrics['pagerank']
        
        self._centrality_cache['centralities'] = metrics
        return metrics
    
    def _find_articulation_points(self, graph: nx.DiGraph) -> set:
        """Find articulation points in the graph"""
        # Convert to undirected for articulation point detection
        undirected = graph.to_undirected()
        return set(nx.articulation_points(undirected))
    
    def _find_bridge_endpoints(self, graph: nx.DiGraph) -> set:
        """Find endpoints of bridge edges"""
        undirected = graph.to_undirected()
        bridges = nx.bridges(undirected)
        endpoints = set()
        for u, v in bridges:
            endpoints.add(u)
            endpoints.add(v)
        return endpoints
    
    def _compute_qos_scores_for_graph(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Compute QoS scores for all components in the graph"""
        qos_scores = {}
        
        for node, data in graph.nodes(data=True):
            node_type = data.get('type')
            
            if node_type == 'Application':
                score = self._compute_application_qos(graph, node)
            elif node_type == 'Topic':
                score = self._compute_topic_qos(data)
            elif node_type == 'Broker':
                score = self._compute_broker_qos(graph, node)
            elif node_type == 'Node':
                score = self._compute_node_qos(graph, node)
            else:
                score = 0.0
            
            qos_scores[node] = score
        
        return qos_scores
    
    def _compute_application_qos(self, graph: nx.DiGraph, app_node: str) -> float:
        """Compute QoS score for an application"""
        # Get published and subscribed topics
        published_topics = []
        subscribed_topics = []
        
        for _, target, data in graph.out_edges(app_node, data=True):
            if data.get('type') == 'PUBLISHES_TO':
                topic_data = graph.nodes[target]
                published_topics.append(topic_data)
            elif data.get('type') == 'SUBSCRIBES_TO':
                topic_data = graph.nodes[target]
                subscribed_topics.append(topic_data)
        
        return self.component_analyzer.calculate_application_qos_score(
            published_topics, subscribed_topics
        )
    
    def _compute_topic_qos(self, topic_data: Dict) -> float:
        """Compute QoS score for a topic"""
        return self.component_analyzer.calculate_topic_qos_score(topic_data)
    
    def _compute_broker_qos(self, graph: nx.DiGraph, broker_node: str) -> float:
        """Compute QoS score for a broker"""
        routed_topics = []
        
        for _, target, data in graph.out_edges(broker_node, data=True):
            if data.get('type') == 'ROUTES':
                topic_data = graph.nodes[target]
                routed_topics.append(topic_data)
        
        return self.component_analyzer.calculate_broker_qos_score(routed_topics)
    
    def _compute_node_qos(self, graph: nx.DiGraph, node_node: str) -> float:
        """Compute QoS score for a physical node"""
        running_apps_scores = []
        
        for source, _, data in graph.in_edges(node_node, data=True):
            if data.get('type') == 'RUNS_ON':
                source_type = graph.nodes[source].get('type')
                if source_type == 'Application':
                    app_score = self._compute_application_qos(graph, source)
                    running_apps_scores.append(app_score)
        
        return self.component_analyzer.calculate_node_qos_score(running_apps_scores)
    
    def _compute_dependency_metrics(self, graph: nx.DiGraph) -> Dict[str, int]:
        """Compute dependency counts for all nodes"""
        dependency_counts = {}
        
        for node in graph.nodes():
            # Count both direct and transitive dependencies
            descendants = nx.descendants(graph, node)
            dependency_counts[node] = len(descendants)
        
        return dependency_counts
    
    def _compute_reachability_impacts(self, graph: nx.DiGraph, sample_size: int = 20) -> Dict[str, float]:
        """
        Compute reachability impact for a sample of nodes (expensive operation)
        
        Args:
            graph: The graph to analyze
            sample_size: Number of nodes to sample for reachability analysis
        """
        impacts = {}
        
        # Sample nodes for reachability analysis
        nodes_to_analyze = list(graph.nodes())
        if len(nodes_to_analyze) > sample_size:
            import random
            nodes_to_analyze = random.sample(nodes_to_analyze, sample_size)
        
        for node in nodes_to_analyze:
            impact = self._calculate_single_reachability_impact(graph, node)
            impacts[node] = impact
        
        # Estimate impacts for non-sampled nodes based on centrality
        centralities = self._compute_all_centralities(graph)
        avg_impact = np.mean(list(impacts.values())) if impacts else 0
        
        for node in graph.nodes():
            if node not in impacts:
                # Estimate based on betweenness centrality
                impacts[node] = centralities['betweenness'][node] * avg_impact * 2
        
        return impacts
    
    def _calculate_single_reachability_impact(self, graph: nx.DiGraph, node: str) -> float:
        """Calculate reachability impact of removing a single node"""
        # Get applications only
        apps = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'Application']
        
        if not apps or len(apps) < 2:
            return 0.0
        
        # Calculate original reachability
        original_paths = 0
        for source in apps:
            for target in apps:
                if source != target and nx.has_path(graph, source, target):
                    original_paths += 1
        
        # Remove node and recalculate
        test_graph = graph.copy()
        test_graph.remove_node(node)
        
        new_paths = 0
        remaining_apps = [a for a in apps if a != node]
        for source in remaining_apps:
            for target in remaining_apps:
                if source != target and nx.has_path(test_graph, source, target):
                    new_paths += 1
        
        # Calculate impact
        if original_paths == 0:
            return 0.0
        
        return (original_paths - new_paths) / original_paths
    
    def _calculate_composite_score(self, metrics: CriticalityMetrics) -> float:
        """Calculate composite criticality score from individual metrics"""
        score = 0.0
        
        # Normalize and weight each metric
        score += self.criticality_weights['degree_centrality'] * metrics.degree_centrality
        score += self.criticality_weights['betweenness_centrality'] * metrics.betweenness_centrality
        score += self.criticality_weights['closeness_centrality'] * metrics.closeness_centrality
        score += self.criticality_weights['pagerank'] * metrics.pagerank * 10  # Scale PageRank
        score += self.criticality_weights['eigenvector_centrality'] * metrics.eigenvector_centrality
        score += self.criticality_weights['articulation_point'] * (1.0 if metrics.is_articulation_point else 0.0)
        score += self.criticality_weights['qos_score'] * metrics.qos_score
        
        # Add reachability impact as a multiplier
        score *= (1 + metrics.reachability_impact)
        
        # Ensure score is in [0, 1] range
        return min(1.0, max(0.0, score))
    
    def _classify_criticality_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify components into criticality levels"""
        calculator = ThresholdCalculator()
        
        # Calculate thresholds based on composite scores
        scores = df['composite_score'].tolist()
        thresholds = calculator.calculate_percentile_thresholds(scores)
        
        # Classify each component
        df['criticality_level'] = df['composite_score'].apply(
            lambda score: thresholds.get_criticality(score).value
        )
        
        return df
    
    def analyze_layered_dependencies(self, graph: nx.DiGraph) -> Dict[str, pd.DataFrame]:
        """
        Analyze dependencies at different layers of the system
        
        Returns:
            Dictionary with DataFrames for each layer analysis
        """
        results = {}
        
        # Application layer analysis
        app_graph = self._extract_application_layer(graph)
        if len(app_graph) > 0:
            results['application'] = self.analyze_comprehensive_criticality(app_graph)
        
        # Infrastructure layer analysis
        infra_graph = self._extract_infrastructure_layer(graph)
        if len(infra_graph) > 0:
            results['infrastructure'] = self.analyze_comprehensive_criticality(infra_graph)
        
        # Broker layer analysis
        broker_graph = self._extract_broker_layer(graph)
        if len(broker_graph) > 0:
            results['broker'] = self.analyze_comprehensive_criticality(broker_graph)
        
        # Cross-layer impact analysis
        results['cross_layer'] = self._analyze_cross_layer_impacts(graph)
        
        return results
    
    def _extract_application_layer(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Extract application-level subgraph"""
        app_graph = nx.DiGraph()
        
        # Add application nodes and their relationships
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'Application':
                app_graph.add_node(node, **data)
        
        # Add DEPENDS_ON edges between applications
        for source, target, edge_data in graph.edges(data=True):
            if (edge_data.get('type') == 'DEPENDS_ON' and 
                source in app_graph and target in app_graph):
                app_graph.add_edge(source, target, **edge_data)
        
        return app_graph
    
    def _extract_infrastructure_layer(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Extract infrastructure-level subgraph"""
        infra_graph = nx.DiGraph()
        
        # Add physical nodes
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'Node':
                infra_graph.add_node(node, **data)
        
        # Add CONNECTS_TO edges
        for source, target, edge_data in graph.edges(data=True):
            if (edge_data.get('type') == 'CONNECTS_TO' and 
                source in infra_graph and target in infra_graph):
                infra_graph.add_edge(source, target, **edge_data)
        
        return infra_graph
    
    def _extract_broker_layer(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Extract broker-level subgraph"""
        broker_graph = nx.DiGraph()
        
        # Add brokers and topics
        for node, data in graph.nodes(data=True):
            if data.get('type') in ['Broker', 'Topic']:
                broker_graph.add_node(node, **data)
        
        # Add ROUTES edges
        for source, target, edge_data in graph.edges(data=True):
            if edge_data.get('type') == 'ROUTES':
                broker_graph.add_edge(source, target, **edge_data)
        
        return broker_graph
    
    def _analyze_cross_layer_impacts(self, graph: nx.DiGraph) -> pd.DataFrame:
        """Analyze impacts across system layers"""
        cross_layer_impacts = []
        
        # Analyze impact of physical nodes on applications
        for node, node_data in graph.nodes(data=True):
            if node_data.get('type') == 'Node':
                affected_apps = self._get_apps_on_node(graph, node)
                affected_brokers = self._get_brokers_on_node(graph, node)
                
                impact = {
                    'node': node,
                    'type': 'Node',
                    'affected_applications': len(affected_apps),
                    'affected_brokers': len(affected_brokers),
                    'total_impact': len(affected_apps) + len(affected_brokers) * 2  # Brokers weighted higher
                }
                cross_layer_impacts.append(impact)
        
        # Analyze impact of brokers on communication
        for node, node_data in graph.nodes(data=True):
            if node_data.get('type') == 'Broker':
                routed_topics = self._get_routed_topics(graph, node)
                dependent_apps = self._get_dependent_apps_for_broker(graph, node)
                
                impact = {
                    'node': node,
                    'type': 'Broker',
                    'routed_topics': len(routed_topics),
                    'dependent_applications': len(dependent_apps),
                    'total_impact': len(routed_topics) * len(dependent_apps)
                }
                cross_layer_impacts.append(impact)
        
        return pd.DataFrame(cross_layer_impacts)
    
    def _get_apps_on_node(self, graph: nx.DiGraph, node: str) -> List[str]:
        """Get applications running on a physical node"""
        apps = []
        for source, _, edge_data in graph.in_edges(node, data=True):
            if (edge_data.get('type') == 'RUNS_ON' and 
                graph.nodes[source].get('type') == 'Application'):
                apps.append(source)
        return apps
    
    def _get_brokers_on_node(self, graph: nx.DiGraph, node: str) -> List[str]:
        """Get brokers running on a physical node"""
        brokers = []
        for source, _, edge_data in graph.in_edges(node, data=True):
            if (edge_data.get('type') == 'RUNS_ON' and 
                graph.nodes[source].get('type') == 'Broker'):
                brokers.append(source)
        return brokers
    
    def _get_routed_topics(self, graph: nx.DiGraph, broker: str) -> List[str]:
        """Get topics routed by a broker"""
        topics = []
        for _, target, edge_data in graph.out_edges(broker, data=True):
            if edge_data.get('type') == 'ROUTES':
                topics.append(target)
        return topics
    
    def _get_dependent_apps_for_broker(self, graph: nx.DiGraph, broker: str) -> List[str]:
        """Get applications dependent on a broker"""
        dependent_apps = set()
        
        # Get topics routed by this broker
        routed_topics = self._get_routed_topics(graph, broker)
        
        # Find apps publishing or subscribing to these topics
        for topic in routed_topics:
            for source, _, edge_data in graph.in_edges(topic, data=True):
                if (edge_data.get('type') in ['PUBLISHES_TO', 'SUBSCRIBES_TO'] and
                    graph.nodes[source].get('type') == 'Application'):
                    dependent_apps.add(source)
        
        return list(dependent_apps)
    
    def generate_criticality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive criticality report
        
        Args:
            df: DataFrame with criticality analysis results
            
        Returns:
            Dictionary containing report sections
        """
        report = {
            'summary': self._generate_summary_statistics(df),
            'critical_components': self._identify_critical_components(df),
            'vulnerabilities': self._identify_vulnerabilities(df),
            'recommendations': self._generate_recommendations(df)
        }
        
        return report
    
    def _generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics from analysis"""
        return {
            'total_components': len(df),
            'critical_components': len(df[df['criticality_level'].isin(['VERY_HIGH', 'HIGH'])]),
            'articulation_points': len(df[df['is_articulation_point'] == True]),
            'avg_qos_score': df['qos_score'].mean(),
            'component_distribution': df['type'].value_counts().to_dict(),
            'criticality_distribution': df['criticality_level'].value_counts().to_dict()
        }
    
    def _identify_critical_components(self, df: pd.DataFrame) -> List[Dict]:
        """Identify most critical components"""
        critical = df[df['criticality_level'].isin(['VERY_HIGH', 'HIGH'])]
        
        components = []
        for _, row in critical.iterrows():
            components.append({
                'node': row['node'],
                'type': row['type'],
                'criticality_level': row['criticality_level'],
                'composite_score': row['composite_score'],
                'key_factors': self._identify_key_factors(row)
            })
        
        return components
    
    def _identify_key_factors(self, row: pd.Series) -> List[str]:
        """Identify key factors contributing to criticality"""
        factors = []
        
        if row['is_articulation_point']:
            factors.append('Articulation Point')
        if row['betweenness_centrality'] > 0.5:
            factors.append('High Betweenness Centrality')
        if row['qos_score'] > 0.7:
            factors.append('High QoS Requirements')
        if row['reachability_impact'] > 0.3:
            factors.append('High Reachability Impact')
        if row['dependency_count'] > 10:
            factors.append(f"High Dependencies ({row['dependency_count']})")
        
        return factors
    
    def _identify_vulnerabilities(self, df: pd.DataFrame) -> List[Dict]:
        """Identify system vulnerabilities"""
        vulnerabilities = []
        
        # Single points of failure
        spof = df[df['is_articulation_point'] == True]
        if len(spof) > 0:
            vulnerabilities.append({
                'type': 'Single Points of Failure',
                'severity': 'HIGH',
                'affected_components': spof['node'].tolist(),
                'description': f"{len(spof)} components are single points of failure"
            })
        
        # High concentration of critical components
        critical_types = df[df['criticality_level'].isin(['VERY_HIGH', 'HIGH'])]['type'].value_counts()
        for comp_type, count in critical_types.items():
            if count > len(df[df['type'] == comp_type]) * 0.3:
                vulnerabilities.append({
                    'type': 'High Criticality Concentration',
                    'severity': 'MEDIUM',
                    'affected_components': [comp_type],
                    'description': f"{count} {comp_type} components are critical"
                })
        
        return vulnerabilities
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Articulation points need redundancy
        ap_components = df[df['is_articulation_point'] == True]
        if len(ap_components) > 0:
            recommendations.append({
                'priority': 'HIGH',
                'type': 'Add Redundancy',
                'components': ap_components['node'].tolist()[:5],
                'action': 'Add redundant paths or backup components for articulation points'
            })
        
        # High QoS components need monitoring
        high_qos = df[df['qos_score'] > 0.7]
        if len(high_qos) > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'type': 'Enhanced Monitoring',
                'components': high_qos['node'].tolist()[:5],
                'action': 'Implement enhanced monitoring for high QoS components'
            })
        
        # Components with high reachability impact
        high_impact = df[df['reachability_impact'] > 0.3]
        if len(high_impact) > 0:
            recommendations.append({
                'priority': 'HIGH',
                'type': 'Failover Planning',
                'components': high_impact['node'].tolist()[:5],
                'action': 'Develop failover strategies for high-impact components'
            })
        
        return recommendations