# Enhanced Graph Analyzer with Advanced Features
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ================== ENHANCED DATA STRUCTURES ==================

@dataclass
class EnhancedQoSPolicy:
    """Enhanced QoS Policy with additional real-world parameters"""
    durability: float  # 0-1: volatile to persistent
    reliability: float  # 0-1: best_effort to reliable  
    transport_priority: float  # 0-10: priority level
    deadline: float  # milliseconds
    lifespan: float  # seconds
    history: float  # 0-1: keep_last to keep_all
    throughput: float  # messages per second
    latency_budget: float  # milliseconds
    partition_tolerance: float  # 0-1: tolerance to network splits
    ordering_guarantee: float  # 0-1: unordered to strictly ordered
    
    def get_normalized_vector(self) -> np.ndarray:
        """Return normalized feature vector for ML models"""
        return np.array([
            self.durability,
            self.reliability,
            self.transport_priority / 10,
            min(self.deadline / 1000, 1.0),  # Normalize to 0-1
            min(self.lifespan / 3600, 1.0),  # Normalize to 0-1 (max 1 hour)
            self.history,
            min(self.throughput / 10000, 1.0),  # Normalize (max 10k msg/s)
            min(self.latency_budget / 1000, 1.0),  # Normalize to 0-1
            self.partition_tolerance,
            self.ordering_guarantee
        ])

@dataclass
class ComponentMetrics:
    """Runtime metrics for components"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_io: float = 0.0
    message_rate: float = 0.0
    error_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    queue_depth: int = 0
    connection_count: int = 0
    last_failure_time: Optional[float] = None
    mtbf: float = float('inf')  # Mean time between failures
    
class CriticalityLevel(Enum):
    """Enhanced criticality levels with numeric values"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5
    CRITICAL = 6  # New level for extreme cases

# ================== ADVANCED GRAPH ANALYZER ==================

class EnhancedGraphAnalyzer:
    """
    Advanced Graph Analyzer with ML-ready features and comprehensive metrics
    """
    
    def __init__(self, enable_caching: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_caching = enable_caching
        
        # Enhanced weight configurations
        self.metric_weights = {
            'structural': {
                'degree_centrality': 0.10,
                'betweenness_centrality': 0.15,
                'closeness_centrality': 0.10,
                'eigenvector_centrality': 0.10,
                'pagerank': 0.10,
                'katz_centrality': 0.05,
                'harmonic_centrality': 0.05,
                'load_centrality': 0.05
            },
            'vulnerability': {
                'articulation_point': 0.15,
                'bridge_endpoint': 0.10,
                'k_connectivity': 0.05
            },
            'qos': {
                'qos_composite': 0.20,
                'sla_compliance': 0.10
            },
            'runtime': {
                'performance_score': 0.10,
                'reliability_score': 0.10
            }
        }
        
        # Caching
        self._cache = {} if enable_caching else None
        
        # ML feature scaler
        self.scaler = MinMaxScaler()
        
    def compute_advanced_centralities(self, G: nx.Graph) -> Dict[str, Dict]:
        """
        Compute comprehensive set of centrality metrics including advanced ones
        """
        cache_key = "advanced_centralities"
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]
            
        metrics = {}
        
        # Basic centralities
        metrics['degree'] = nx.degree_centrality(G)
        metrics['betweenness'] = nx.betweenness_centrality(G, normalized=True)
        metrics['closeness'] = nx.closeness_centrality(G)
        metrics['pagerank'] = nx.pagerank(G, max_iter=200)
        
        # Advanced centralities
        try:
            metrics['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            metrics['eigenvector'] = metrics['degree']  # Fallback
            
        # Katz centrality (considers indirect influence)
        try:
            metrics['katz'] = nx.katz_centrality(G, alpha=0.1, beta=1.0)
        except:
            metrics['katz'] = metrics['degree']
            
        # Harmonic centrality (handles disconnected components better)
        metrics['harmonic'] = nx.harmonic_centrality(G)
        
        # Load centrality (fraction of shortest paths)
        metrics['load'] = nx.load_centrality(G)
        
        # Subgraph centrality (participation in closed walks)
        metrics['subgraph'] = nx.subgraph_centrality(G.to_undirected())
        
        if self._cache is not None:
            self._cache[cache_key] = metrics
            
        return metrics
    
    def analyze_structural_vulnerabilities(self, G: nx.Graph) -> Dict:
        """
        Comprehensive structural vulnerability analysis
        """
        vulnerabilities = {
            'articulation_points': [],
            'bridges': [],
            'k_components': {},
            'vertex_connectivity': 0,
            'edge_connectivity': 0,
            'clustering_coefficient': {},
            'core_numbers': {},
            'vulnerability_score': {}
        }
        
        # Convert to undirected for some analyses
        G_undirected = G.to_undirected()
        
        # Articulation points
        vulnerabilities['articulation_points'] = list(
            nx.articulation_points(G_undirected)
        )
        
        # Bridges (critical edges)
        vulnerabilities['bridges'] = list(nx.bridges(G_undirected))
        
        # K-components (groups with k vertex-disjoint paths)
        for k in [2, 3]:
            try:
                k_comps = list(nx.k_components(G_undirected, k))
                vulnerabilities['k_components'][k] = k_comps
            except:
                vulnerabilities['k_components'][k] = []
        
        # Connectivity metrics
        if nx.is_connected(G_undirected):
            vulnerabilities['vertex_connectivity'] = nx.node_connectivity(G_undirected)
            vulnerabilities['edge_connectivity'] = nx.edge_connectivity(G_undirected)
        
        # Clustering coefficient (local density)
        vulnerabilities['clustering_coefficient'] = nx.clustering(G_undirected)
        
        # K-core decomposition
        vulnerabilities['core_numbers'] = nx.core_number(G_undirected)
        
        # Calculate vulnerability score for each node
        for node in G.nodes():
            score = 0.0
            
            # Is articulation point?
            if node in vulnerabilities['articulation_points']:
                score += 0.3
            
            # Is bridge endpoint?
            for u, v in vulnerabilities['bridges']:
                if node in [u, v]:
                    score += 0.2
                    break
            
            # Low k-core number indicates peripheral position
            core_num = vulnerabilities['core_numbers'].get(node, 0)
            max_core = max(vulnerabilities['core_numbers'].values()) if vulnerabilities['core_numbers'] else 1
            score += 0.2 * (1 - core_num / max_core)
            
            # Low clustering indicates bridge position
            clustering = vulnerabilities['clustering_coefficient'].get(node, 0)
            score += 0.3 * (1 - clustering)
            
            vulnerabilities['vulnerability_score'][node] = score
        
        return vulnerabilities
    
    def calculate_qos_aware_criticality(self, 
                                       G: nx.Graph,
                                       qos_policies: Dict[str, EnhancedQoSPolicy]) -> Dict:
        """
        Calculate QoS-aware criticality with SLA considerations
        """
        qos_scores = {}
        
        for node in G.nodes():
            if node not in qos_policies:
                qos_scores[node] = {'composite': 0.0, 'sla_risk': 0.0}
                continue
                
            policy = qos_policies[node]
            
            # Weighted QoS score
            weights = {
                'reliability': 0.25,
                'durability': 0.20,
                'latency': 0.20,
                'throughput': 0.15,
                'ordering': 0.10,
                'partition_tolerance': 0.10
            }
            
            qos_score = (
                weights['reliability'] * policy.reliability +
                weights['durability'] * policy.durability +
                weights['latency'] * (1 - min(policy.latency_budget / 1000, 1.0)) +
                weights['throughput'] * min(policy.throughput / 10000, 1.0) +
                weights['ordering'] * policy.ordering_guarantee +
                weights['partition_tolerance'] * policy.partition_tolerance
            )
            
            # SLA risk score (inverse of meeting SLA requirements)
            sla_risk = 0.0
            if policy.deadline < 100:  # Strict deadline
                sla_risk += 0.3
            if policy.reliability > 0.99:  # High reliability requirement
                sla_risk += 0.3
            if policy.throughput > 5000:  # High throughput requirement
                sla_risk += 0.2
            if policy.ordering_guarantee > 0.9:  # Strict ordering
                sla_risk += 0.2
                
            qos_scores[node] = {
                'composite': qos_score,
                'sla_risk': sla_risk,
                'priority_normalized': policy.transport_priority / 10
            }
        
        return qos_scores
    
    def analyze_cascade_risk(self, G: nx.Graph, 
                            failure_node: str,
                            cascade_threshold: float = 0.7) -> Dict:
        """
        Analyze cascading failure risk when a node fails
        """
        cascade_analysis = {
            'directly_affected': [],
            'cascade_risk_nodes': [],
            'affected_paths': [],
            'service_impact': 0.0,
            'recovery_complexity': 0.0
        }
        
        # Find directly connected nodes
        cascade_analysis['directly_affected'] = list(G.neighbors(failure_node))
        
        # Calculate load redistribution impact
        initial_betweenness = nx.betweenness_centrality(G)
        
        # Create graph without failed node
        G_failed = G.copy()
        G_failed.remove_node(failure_node)
        
        if len(G_failed) > 0:
            new_betweenness = nx.betweenness_centrality(G_failed)
            
            # Find nodes with significant load increase
            for node in G_failed.nodes():
                initial_load = initial_betweenness.get(node, 0)
                new_load = new_betweenness.get(node, 0)
                
                if initial_load > 0:
                    load_increase = (new_load - initial_load) / initial_load
                    if load_increase > cascade_threshold:
                        cascade_analysis['cascade_risk_nodes'].append({
                            'node': node,
                            'load_increase': load_increase,
                            'risk_level': 'HIGH' if load_increase > 1.0 else 'MEDIUM'
                        })
        
        # Analyze affected paths
        for source in G.nodes():
            if source == failure_node:
                continue
            for target in G.nodes():
                if target == failure_node or target == source:
                    continue
                    
                try:
                    # Check if path exists through failure node
                    paths_before = list(nx.all_shortest_paths(G, source, target))
                    paths_through_failed = [p for p in paths_before if failure_node in p]
                    
                    if paths_through_failed:
                        cascade_analysis['affected_paths'].append({
                            'source': source,
                            'target': target,
                            'paths_lost': len(paths_through_failed),
                            'total_paths': len(paths_before)
                        })
                except nx.NetworkXNoPath:
                    continue
        
        # Calculate overall service impact
        total_nodes = len(G.nodes())
        cascade_analysis['service_impact'] = (
            len(cascade_analysis['directly_affected']) + 
            len(cascade_analysis['cascade_risk_nodes'])
        ) / total_nodes
        
        # Recovery complexity based on dependencies
        cascade_analysis['recovery_complexity'] = (
            len(cascade_analysis['directly_affected']) * 0.3 +
            len(cascade_analysis['cascade_risk_nodes']) * 0.5 +
            len(cascade_analysis['affected_paths']) * 0.2
        ) / total_nodes
        
        return cascade_analysis
    
    def calculate_composite_criticality(self,
                                       G: nx.Graph,
                                       centralities: Dict,
                                       vulnerabilities: Dict,
                                       qos_scores: Dict,
                                       runtime_metrics: Optional[Dict[str, ComponentMetrics]] = None) -> pd.DataFrame:
        """
        Calculate comprehensive composite criticality scores
        """
        results = []
        
        for node in G.nodes():
            scores = {}
            
            # Structural scores (centralities)
            structural_score = 0.0
            for metric, weight in self.metric_weights['structural'].items():
                if metric in centralities:
                    structural_score += weight * centralities[metric].get(node, 0)
            scores['structural'] = structural_score
            
            # Vulnerability scores
            vulnerability_score = (
                self.metric_weights['vulnerability']['articulation_point'] * 
                (1.0 if node in vulnerabilities['articulation_points'] else 0.0) +
                self.metric_weights['vulnerability']['bridge_endpoint'] * 
                vulnerabilities['vulnerability_score'].get(node, 0)
            )
            scores['vulnerability'] = vulnerability_score
            
            # QoS scores
            node_qos = qos_scores.get(node, {'composite': 0, 'sla_risk': 0})
            qos_score = (
                self.metric_weights['qos']['qos_composite'] * node_qos['composite'] +
                self.metric_weights['qos']['sla_compliance'] * node_qos['sla_risk']
            )
            scores['qos'] = qos_score
            
            # Runtime performance scores
            if runtime_metrics and node in runtime_metrics:
                metrics = runtime_metrics[node]
                performance_score = 1.0 - (
                    (metrics.cpu_usage + metrics.memory_usage) / 2 * 0.3 +
                    min(metrics.latency_p99 / 1000, 1.0) * 0.4 +
                    metrics.error_rate * 0.3
                )
                reliability_score = 1.0 / (1 + np.exp(-5 * (metrics.mtbf / 86400 - 1)))  # Sigmoid
                
                runtime_score = (
                    self.metric_weights['runtime']['performance_score'] * performance_score +
                    self.metric_weights['runtime']['reliability_score'] * reliability_score
                )
            else:
                runtime_score = 0.0
            scores['runtime'] = runtime_score
            
            # Calculate final composite score
            composite = (
                scores['structural'] * 0.3 +
                scores['vulnerability'] * 0.25 +
                scores['qos'] * 0.25 +
                scores['runtime'] * 0.2
            )
            
            results.append({
                'node': node,
                'node_type': G.nodes[node].get('type', 'Unknown'),
                'structural_score': scores['structural'],
                'vulnerability_score': scores['vulnerability'],
                'qos_score': scores['qos'],
                'runtime_score': scores['runtime'],
                'composite_score': composite,
                'is_articulation_point': node in vulnerabilities['articulation_points'],
                'core_number': vulnerabilities['core_numbers'].get(node, 0),
                'clustering_coefficient': vulnerabilities['clustering_coefficient'].get(node, 0)
            })
        
        df = pd.DataFrame(results)
        
        # Add criticality classifications
        df['criticality_level'] = self.classify_criticality_advanced(
            df['composite_score'].values
        )
        
        return df.sort_values('composite_score', ascending=False)
    
    def classify_criticality_advanced(self, scores: np.ndarray) -> List[str]:
        """
        Advanced classification using multiple statistical methods
        """
        classifications = []
        
        # Use both IQR and percentile methods
        q1, q2, q3 = np.percentile(scores, [25, 50, 75])
        iqr = q3 - q1
        
        # Also calculate percentiles for finer granularity
        p10, p90, p95, p99 = np.percentile(scores, [10, 90, 95, 99])
        
        for score in scores:
            # Extreme outliers (top 1%)
            if score >= p99:
                level = CriticalityLevel.CRITICAL
            # Very high (top 5%)
            elif score >= p95:
                level = CriticalityLevel.VERY_HIGH
            # High (top 10% or above Q3 + 1.5*IQR)
            elif score >= p90 or score > q3 + 1.5 * iqr:
                level = CriticalityLevel.HIGH
            # Medium (above median)
            elif score >= q2:
                level = CriticalityLevel.MEDIUM
            # Low (above Q1)
            elif score >= q1:
                level = CriticalityLevel.LOW
            # Very low (below Q1)
            else:
                level = CriticalityLevel.VERY_LOW
            
            classifications.append(level.name)
        
        return classifications
    
    def generate_ml_features(self, 
                            G: nx.Graph,
                            node: str,
                            centralities: Dict,
                            vulnerabilities: Dict,
                            qos_policy: Optional[EnhancedQoSPolicy] = None,
                            runtime_metrics: Optional[ComponentMetrics] = None) -> np.ndarray:
        """
        Generate feature vector for ML models (GNN input)
        """
        features = []
        
        # Structural features (8 dimensions)
        for metric in ['degree', 'betweenness', 'closeness', 'pagerank', 
                      'eigenvector', 'katz', 'harmonic', 'load']:
            features.append(centralities.get(metric, {}).get(node, 0))
        
        # Vulnerability features (4 dimensions)
        features.append(1.0 if node in vulnerabilities['articulation_points'] else 0.0)
        features.append(vulnerabilities['vulnerability_score'].get(node, 0))
        features.append(vulnerabilities['core_numbers'].get(node, 0) / 
                       max(vulnerabilities['core_numbers'].values(), default=1))
        features.append(vulnerabilities['clustering_coefficient'].get(node, 0))
        
        # Graph position features (3 dimensions)
        features.append(G.in_degree(node) / G.number_of_nodes())
        features.append(G.out_degree(node) / G.number_of_nodes())
        features.append(len(list(G.neighbors(node))) / G.number_of_nodes())
        
        # QoS features (10 dimensions)
        if qos_policy:
            features.extend(qos_policy.get_normalized_vector())
        else:
            features.extend([0.0] * 10)
        
        # Runtime features (8 dimensions)
        if runtime_metrics:
            features.extend([
                runtime_metrics.cpu_usage,
                runtime_metrics.memory_usage,
                min(runtime_metrics.network_io / 1000, 1.0),
                min(runtime_metrics.message_rate / 10000, 1.0),
                runtime_metrics.error_rate,
                min(runtime_metrics.latency_p99 / 1000, 1.0),
                min(runtime_metrics.queue_depth / 1000, 1.0),
                min(runtime_metrics.connection_count / 100, 1.0)
            ])
        else:
            features.extend([0.0] * 8)
        
        return np.array(features)

# ================== SPECIALIZED ANALYZERS ==================

class MessageFlowAnalyzer:
    """
    Specialized analyzer for message flow criticality in pub-sub systems
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.flow_paths = {}
        self.critical_flows = []
        
    def analyze_topic_flows(self) -> Dict:
        """
        Analyze message flows through topics
        """
        topic_flows = {}
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'Topic':
                publishers = []
                subscribers = []
                
                for src, dst, data in self.graph.edges(data=True):
                    if dst == node and data.get('type') == 'PUBLISHES_TO':
                        publishers.append(src)
                    elif src == node and data.get('type') == 'SUBSCRIBES_TO':
                        subscribers.append(dst)
                
                flow_score = len(publishers) * len(subscribers)
                
                topic_flows[node] = {
                    'publishers': publishers,
                    'subscribers': subscribers,
                    'flow_volume': flow_score,
                    'fan_out_ratio': len(subscribers) / max(len(publishers), 1),
                    'is_bottleneck': len(publishers) > 5 and len(subscribers) > 10
                }
        
        return topic_flows
    
    def identify_critical_paths(self, threshold: float = 0.7) -> List[Dict]:
        """
        Identify critical message paths in the system
        """
        critical_paths = []
        
        # Find all publisher-subscriber pairs
        publishers = [n for n in self.graph.nodes() 
                     if any(e[2].get('type') == 'PUBLISHES_TO' 
                           for e in self.graph.out_edges(n, data=True))]
        
        subscribers = [n for n in self.graph.nodes()
                      if any(e[2].get('type') == 'SUBSCRIBES_TO'
                            for e in self.graph.in_edges(n, data=True))]
        
        for pub in publishers:
            for sub in subscribers:
                if pub == sub:
                    continue
                    
                try:
                    # Find all paths through topics
                    paths = list(nx.all_simple_paths(self.graph, pub, sub, cutoff=5))
                    
                    for path in paths:
                        # Check if path includes topic
                        topics_in_path = [n for n in path 
                                        if self.graph.nodes[n].get('type') == 'Topic']
                        
                        if topics_in_path:
                            # Calculate path criticality
                            path_length = len(path)
                            topic_importance = len(topics_in_path) / path_length
                            
                            criticality = (1.0 / path_length) * (1 + topic_importance)
                            
                            if criticality > threshold:
                                critical_paths.append({
                                    'publisher': pub,
                                    'subscriber': sub,
                                    'path': path,
                                    'topics': topics_in_path,
                                    'criticality': criticality,
                                    'length': path_length
                                })
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by criticality
        critical_paths.sort(key=lambda x: x['criticality'], reverse=True)
        self.critical_flows = critical_paths[:20]  # Keep top 20
        
        return self.critical_flows

# ================== EXAMPLE USAGE ==================

def demonstration_example():
    """
    Demonstration of enhanced graph analysis capabilities
    """
    # Create sample graph
    G = nx.DiGraph()
    
    # Add nodes with types
    nodes = {
        'app1': 'Application',
        'app2': 'Application',
        'app3': 'Application',
        'broker1': 'Broker',
        'broker2': 'Broker',
        'topic1': 'Topic',
        'topic2': 'Topic',
        'topic3': 'Topic',
        'node1': 'Node',
        'node2': 'Node'
    }
    
    for node_id, node_type in nodes.items():
        G.add_node(node_id, type=node_type)
    
    # Add edges
    edges = [
        ('app1', 'topic1', 'PUBLISHES_TO'),
        ('app2', 'topic2', 'PUBLISHES_TO'),
        ('topic1', 'app3', 'SUBSCRIBES_TO'),
        ('topic2', 'app3', 'SUBSCRIBES_TO'),
        ('broker1', 'topic1', 'ROUTES'),
        ('broker2', 'topic2', 'ROUTES'),
        ('broker2', 'topic3', 'ROUTES'),
        ('app1', 'node1', 'RUNS_ON'),
        ('app2', 'node1', 'RUNS_ON'),
        ('app3', 'node2', 'RUNS_ON'),
        ('broker1', 'node1', 'RUNS_ON'),
        ('broker2', 'node2', 'RUNS_ON')
    ]
    
    for src, dst, edge_type in edges:
        G.add_edge(src, dst, type=edge_type)
    
    # Create QoS policies
    qos_policies = {
        'topic1': EnhancedQoSPolicy(
            durability=1.0, reliability=0.99, transport_priority=9,
            deadline=100, lifespan=3600, history=1.0,
            throughput=5000, latency_budget=50,
            partition_tolerance=0.8, ordering_guarantee=1.0
        ),
        'topic2': EnhancedQoSPolicy(
            durability=0.5, reliability=0.95, transport_priority=5,
            deadline=500, lifespan=1800, history=0.5,
            throughput=1000, latency_budget=200,
            partition_tolerance=0.5, ordering_guarantee=0.5
        )
    }
    
    # Create runtime metrics
    runtime_metrics = {
        'app1': ComponentMetrics(
            cpu_usage=0.75, memory_usage=0.60,
            message_rate=1000, error_rate=0.01,
            latency_p99=150, mtbf=86400
        ),
        'broker1': ComponentMetrics(
            cpu_usage=0.85, memory_usage=0.70,
            message_rate=5000, error_rate=0.001,
            latency_p99=50, mtbf=172800
        )
    }
    
    # Run analysis
    print("="*60)
    print("ENHANCED GRAPH ANALYSIS DEMONSTRATION")
    print("="*60)
    
    analyzer = EnhancedGraphAnalyzer()
    
    # Compute centralities
    print("\n1. Computing advanced centrality metrics...")
    centralities = analyzer.compute_advanced_centralities(G)
    
    # Analyze vulnerabilities
    print("2. Analyzing structural vulnerabilities...")
    vulnerabilities = analyzer.analyze_structural_vulnerabilities(G)
    
    # Calculate QoS scores
    print("3. Calculating QoS-aware criticality...")
    qos_scores = analyzer.calculate_qos_aware_criticality(G, qos_policies)
    
    # Composite criticality
    print("4. Computing composite criticality scores...")
    results_df = analyzer.calculate_composite_criticality(
        G, centralities, vulnerabilities, qos_scores, runtime_metrics
    )
    
    # Display results
    print("\n" + "="*60)
    print("CRITICALITY ANALYSIS RESULTS")
    print("="*60)
    print("\nTop 5 Critical Components:")
    print(results_df[['node', 'node_type', 'composite_score', 'criticality_level']].head())
    
    print(f"\nArticulation Points: {vulnerabilities['articulation_points']}")
    print(f"Bridges: {vulnerabilities['bridges']}")
    
    # Cascade risk analysis
    print("\n5. Analyzing cascade risk for critical node...")
    critical_node = results_df.iloc[0]['node']
    cascade_risk = analyzer.analyze_cascade_risk(G, critical_node)
    print(f"\nCascade Risk Analysis for '{critical_node}':")
    print(f"  Directly affected nodes: {cascade_risk['directly_affected']}")
    print(f"  Service impact: {cascade_risk['service_impact']:.2%}")
    print(f"  Recovery complexity: {cascade_risk['recovery_complexity']:.2f}")
    
    # Message flow analysis
    print("\n6. Analyzing message flows...")
    flow_analyzer = MessageFlowAnalyzer(G)
    topic_flows = flow_analyzer.analyze_topic_flows()
    critical_paths = flow_analyzer.identify_critical_paths()
    
    print("\nTopic Flow Analysis:")
    for topic, flow_info in topic_flows.items():
        print(f"  {topic}:")
        print(f"    Publishers: {flow_info['publishers']}")
        print(f"    Subscribers: {flow_info['subscribers']}")
        print(f"    Fan-out ratio: {flow_info['fan_out_ratio']:.2f}")
        print(f"    Is bottleneck: {flow_info['is_bottleneck']}")
    
    if critical_paths:
        print(f"\nTop Critical Message Paths:")
        for i, path_info in enumerate(critical_paths[:3], 1):
            print(f"  {i}. {path_info['publisher']} → {path_info['subscriber']}")
            print(f"     Path: {' → '.join(path_info['path'])}")
            print(f"     Criticality: {path_info['criticality']:.3f}")
    
    # Generate ML features for a node
    print("\n7. Generating ML features for GNN...")
    ml_features = analyzer.generate_ml_features(
        G, 'app1', centralities, vulnerabilities,
        qos_policies.get('topic1'), runtime_metrics.get('app1')
    )
    print(f"Feature vector shape: {ml_features.shape}")
    print(f"Feature vector (first 10): {ml_features[:10]}")
    
    return analyzer, results_df

if __name__ == "__main__":
    analyzer, results = demonstration_example()