"""
Comprehensive Analysis of Critical Components and Paths in Distributed Publish-Subscribe Systems

This notebook demonstrates:
1. Multi-layered graph modeling of pub-sub architecture
2. Topological metrics (centrality, articulation points)
3. QoS-based topic importance
4. Composite criticality scoring
5. Impact score through reachability analysis
6. Critical path identification
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class QoSProfile:
    """Quality of Service profile for topics"""
    latency_requirement: float  # ms
    throughput_requirement: float  # msgs/sec
    reliability: float  # 0-1
    priority: int  # 1-5
    
    def importance_score(self) -> float:
        """Calculate topic importance from QoS parameters"""
        # Normalize and weight QoS factors
        norm_latency = 1.0 / (1.0 + self.latency_requirement / 100)
        norm_throughput = min(self.throughput_requirement / 10000, 1.0)
        norm_priority = self.priority / 5.0
        
        return (0.3 * norm_latency + 
                0.2 * norm_throughput + 
                0.3 * self.reliability + 
                0.2 * norm_priority)


@dataclass
class Component:
    """Base component in pub-sub system"""
    id: str
    type: str  # 'topic', 'application', 'broker', 'node'
    metadata: Dict = None


# ============================================================================
# MULTI-LAYER GRAPH MODEL
# ============================================================================

class PubSubGraphModel:
    """Multi-layered graph model for pub-sub systems"""
    
    def __init__(self):
        # Separate layers
        self.physical_layer = nx.DiGraph()  # Nodes and network
        self.broker_layer = nx.DiGraph()    # Broker topology
        self.logical_layer = nx.DiGraph()   # Topics and subscriptions
        self.application_layer = nx.DiGraph()  # Applications
        
        # Composite graph
        self.composite_graph = nx.DiGraph()
        
        # Metadata
        self.components = {}
        self.topic_qos = {}
        
    def add_physical_node(self, node_id: str, metadata: Dict = None):
        """Add physical infrastructure node"""
        self.physical_layer.add_node(node_id, layer='physical', **metadata or {})
        self.composite_graph.add_node(node_id, layer='physical', type='node')
        self.components[node_id] = Component(node_id, 'node', metadata)
        
    def add_broker(self, broker_id: str, node_id: str, metadata: Dict = None):
        """Add broker on a physical node"""
        self.broker_layer.add_node(broker_id, layer='broker', **metadata or {})
        self.composite_graph.add_node(broker_id, layer='broker', type='broker')
        self.composite_graph.add_edge(broker_id, node_id, relation='hosted_on')
        self.components[broker_id] = Component(broker_id, 'broker', metadata)
        
    def add_topic(self, topic_id: str, broker_id: str, qos: QoSProfile):
        """Add topic managed by broker"""
        self.logical_layer.add_node(topic_id, layer='logical', 
                                    importance=qos.importance_score())
        self.composite_graph.add_node(topic_id, layer='logical', type='topic',
                                     importance=qos.importance_score())
        self.composite_graph.add_edge(topic_id, broker_id, relation='managed_by')
        self.topic_qos[topic_id] = qos
        self.components[topic_id] = Component(topic_id, 'topic', {'qos': qos})
        
    def add_application(self, app_id: str, app_type: str, metadata: Dict = None):
        """Add publisher or subscriber application"""
        self.application_layer.add_node(app_id, layer='application', 
                                       app_type=app_type, **metadata or {})
        self.composite_graph.add_node(app_id, layer='application', 
                                     type='application', app_type=app_type)
        self.components[app_id] = Component(app_id, 'application', metadata)
        
    def add_publish_relation(self, publisher_id: str, topic_id: str, rate: float = 1.0):
        """Publisher publishes to topic"""
        self.application_layer.add_edge(publisher_id, topic_id, 
                                       relation='publishes', rate=rate)
        self.composite_graph.add_edge(publisher_id, topic_id, 
                                     relation='publishes', rate=rate)
        
    def add_subscribe_relation(self, topic_id: str, subscriber_id: str):
        """Subscriber subscribes to topic"""
        self.application_layer.add_edge(topic_id, subscriber_id, 
                                       relation='subscribes')
        self.composite_graph.add_edge(topic_id, subscriber_id, 
                                     relation='subscribes')
        
    def add_broker_link(self, broker1: str, broker2: str, bidirectional: bool = True):
        """Add connection between brokers (replication/federation)"""
        self.broker_layer.add_edge(broker1, broker2, relation='replicates')
        self.composite_graph.add_edge(broker1, broker2, relation='replicates')
        if bidirectional:
            self.broker_layer.add_edge(broker2, broker1, relation='replicates')
            self.composite_graph.add_edge(broker2, broker1, relation='replicates')


# ============================================================================
# TOPOLOGICAL METRICS
# ============================================================================

class TopologicalAnalyzer:
    """Compute centrality metrics and structural properties"""
    
    @staticmethod
    def compute_centrality_metrics(G: nx.Graph) -> pd.DataFrame:
        """Compute multiple centrality metrics"""
        metrics = {}
        
        # Degree centrality
        metrics['degree_centrality'] = nx.degree_centrality(G)
        
        # Betweenness centrality
        metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
        
        # Closeness centrality
        try:
            metrics['closeness_centrality'] = nx.closeness_centrality(G)
        except:
            metrics['closeness_centrality'] = {n: 0 for n in G.nodes()}
        
        # Eigenvector centrality (if graph is strongly connected)
        try:
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            metrics['eigenvector_centrality'] = {n: 0 for n in G.nodes()}
        
        # PageRank (works for directed graphs)
        metrics['pagerank'] = nx.pagerank(G)
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics).fillna(0)
        
        # Normalize each metric to 0-1
        for col in df.columns:
            max_val = df[col].max()
            if max_val > 0:
                df[col] = df[col] / max_val
                
        return df
    
    @staticmethod
    def find_articulation_points(G: nx.Graph) -> Set[str]:
        """Find articulation points (cut vertices)"""
        # Convert to undirected for articulation point analysis
        G_undirected = G.to_undirected()
        return set(nx.articulation_points(G_undirected))
    
    @staticmethod
    def find_bridges(G: nx.Graph) -> Set[Tuple[str, str]]:
        """Find bridges (cut edges)"""
        G_undirected = G.to_undirected()
        return set(nx.bridges(G_undirected))


# ============================================================================
# CRITICALITY SCORING
# ============================================================================

class CriticalityScorer:
    """Compute composite criticality scores"""
    
    def __init__(self, model: PubSubGraphModel):
        self.model = model
        self.topology_metrics = None
        self.articulation_points = None
        
    def compute_topology_score(self) -> Dict[str, float]:
        """Compute topology-based criticality"""
        analyzer = TopologicalAnalyzer()
        
        # Get centrality metrics
        self.topology_metrics = analyzer.compute_centrality_metrics(
            self.model.composite_graph
        )
        
        # Find articulation points
        self.articulation_points = analyzer.find_articulation_points(
            self.model.composite_graph
        )
        
        # Weighted combination of centrality metrics
        weights = {
            'betweenness_centrality': 0.35,
            'degree_centrality': 0.25,
            'pagerank': 0.20,
            'closeness_centrality': 0.15,
            'eigenvector_centrality': 0.05
        }
        
        topology_scores = {}
        for node in self.model.composite_graph.nodes():
            score = sum(
                self.topology_metrics.loc[node, metric] * weight
                for metric, weight in weights.items()
            )
            
            # Bonus for articulation points
            if node in self.articulation_points:
                score *= 1.5
                
            topology_scores[node] = min(score, 1.0)
            
        return topology_scores
    
    def compute_qos_importance(self) -> Dict[str, float]:
        """Get QoS-based importance for topics"""
        importance = {}
        for node in self.model.composite_graph.nodes():
            if node in self.model.topic_qos:
                importance[node] = self.model.topic_qos[node].importance_score()
            else:
                importance[node] = 0.0
        return importance
    
    def compute_composite_score(self) -> pd.DataFrame:
        """Compute final composite criticality score"""
        # Get individual scores
        topology_scores = self.compute_topology_score()
        qos_scores = self.compute_qos_importance()
        
        # Component-specific weights
        component_weights = {
            'topic': {'topology': 0.4, 'qos': 0.6},
            'broker': {'topology': 0.8, 'qos': 0.2},
            'node': {'topology': 0.9, 'qos': 0.1},
            'application': {'topology': 0.7, 'qos': 0.3}
        }
        
        results = []
        for node in self.model.composite_graph.nodes():
            node_data = self.model.composite_graph.nodes[node]
            component_type = node_data.get('type', 'unknown')
            
            weights = component_weights.get(component_type, {'topology': 0.5, 'qos': 0.5})
            
            composite_score = (
                weights['topology'] * topology_scores.get(node, 0) +
                weights['qos'] * qos_scores.get(node, 0)
            )
            
            results.append({
                'component': node,
                'type': component_type,
                'topology_score': topology_scores.get(node, 0),
                'qos_score': qos_scores.get(node, 0),
                'composite_score': composite_score,
                'is_articulation_point': node in self.articulation_points
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('composite_score', ascending=False)


# ============================================================================
# IMPACT ANALYSIS (REACHABILITY LOSS)
# ============================================================================

class ImpactAnalyzer:
    """Analyze impact through reachability loss"""
    
    def __init__(self, model: PubSubGraphModel):
        self.model = model
        
    def compute_reachability_matrix(self) -> Dict[Tuple[str, str], bool]:
        """Compute which nodes can reach which other nodes"""
        G = self.model.composite_graph
        reachability = {}
        
        for source in G.nodes():
            reachable = nx.descendants(G, source)
            reachable.add(source)
            for target in G.nodes():
                reachability[(source, target)] = target in reachable
                
        return reachability
    
    def compute_impact_score(self, node: str) -> float:
        """Compute impact of removing a node (reachability loss)"""
        G = self.model.composite_graph
        
        # Original reachability
        original_reachability = self.compute_reachability_matrix()
        total_pairs = len(G.nodes()) * (len(G.nodes()) - 1)
        original_reachable = sum(original_reachability.values())
        
        # Create graph without the node
        G_reduced = G.copy()
        G_reduced.remove_node(node)
        
        # New reachability
        new_reachability = {}
        for source in G_reduced.nodes():
            reachable = nx.descendants(G_reduced, source)
            reachable.add(source)
            for target in G_reduced.nodes():
                new_reachability[(source, target)] = target in reachable
        
        new_reachable = sum(new_reachability.values())
        
        # Impact = fraction of reachability lost
        if original_reachable > 0:
            impact = (original_reachable - new_reachable) / original_reachable
        else:
            impact = 0.0
            
        return impact
    
    def compute_all_impacts(self) -> pd.DataFrame:
        """Compute impact scores for all components"""
        results = []
        
        for node in self.model.composite_graph.nodes():
            node_data = self.model.composite_graph.nodes[node]
            impact = self.compute_impact_score(node)
            
            results.append({
                'component': node,
                'type': node_data.get('type', 'unknown'),
                'impact_score': impact
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('impact_score', ascending=False)


# ============================================================================
# CRITICAL PATH ANALYSIS
# ============================================================================

class CriticalPathAnalyzer:
    """Identify critical paths in the system"""
    
    def __init__(self, model: PubSubGraphModel):
        self.model = model
        
    def find_publisher_to_subscriber_paths(self) -> List[Dict]:
        """Find all paths from publishers to subscribers"""
        G = self.model.composite_graph
        paths = []
        
        # Identify publishers and subscribers
        publishers = [n for n in G.nodes() 
                     if G.nodes[n].get('app_type') == 'publisher']
        subscribers = [n for n in G.nodes() 
                      if G.nodes[n].get('app_type') == 'subscriber']
        
        # Find all simple paths
        for pub in publishers:
            for sub in subscribers:
                try:
                    all_paths = list(nx.all_simple_paths(G, pub, sub, cutoff=10))
                    for path in all_paths:
                        paths.append({
                            'publisher': pub,
                            'subscriber': sub,
                            'path': path,
                            'length': len(path)
                        })
                except nx.NetworkXNoPath:
                    continue
                    
        return paths
    
    def score_path_criticality(self, path: List[str], 
                               component_scores: Dict[str, float]) -> float:
        """Score path criticality based on component scores"""
        # Use minimum score along path (weakest link)
        min_score = min(component_scores.get(node, 0) for node in path)
        # Use average score
        avg_score = np.mean([component_scores.get(node, 0) for node in path])
        
        # Combined metric
        return 0.6 * min_score + 0.4 * avg_score
    
    def identify_critical_paths(self, component_scores: Dict[str, float], 
                                top_k: int = 10) -> pd.DataFrame:
        """Identify most critical paths"""
        paths = self.find_publisher_to_subscriber_paths()
        
        results = []
        for path_info in paths:
            path = path_info['path']
            criticality = self.score_path_criticality(path, component_scores)
            
            results.append({
                'publisher': path_info['publisher'],
                'subscriber': path_info['subscriber'],
                'path': ' -> '.join(path),
                'path_length': len(path),
                'criticality_score': criticality,
                'components': path
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('criticality_score', ascending=False).head(top_k)


# ============================================================================
# INTEGRATED ANALYSIS
# ============================================================================

class IntegratedAnalysis:
    """Combine all analysis methods"""
    
    def __init__(self, model: PubSubGraphModel):
        self.model = model
        self.criticality_scorer = CriticalityScorer(model)
        self.impact_analyzer = ImpactAnalyzer(model)
        self.path_analyzer = CriticalPathAnalyzer(model)
        
    def run_complete_analysis(self) -> Dict[str, pd.DataFrame]:
        """Run all analyses and combine results"""
        print("Computing composite criticality scores...")
        criticality_df = self.criticality_scorer.compute_composite_score()
        
        print("Computing impact scores (reachability loss)...")
        impact_df = self.impact_analyzer.compute_all_impacts()
        
        print("Analyzing critical paths...")
        component_scores = dict(zip(criticality_df['component'], 
                                   criticality_df['composite_score']))
        critical_paths_df = self.path_analyzer.identify_critical_paths(component_scores)
        
        # Merge criticality and impact
        combined_df = criticality_df.merge(
            impact_df[['component', 'impact_score']], 
            on='component',
            how='left'
        )
        
        # Calculate final integrated score
        combined_df['integrated_score'] = (
            0.5 * combined_df['composite_score'] + 
            0.5 * combined_df['impact_score']
        )
        
        combined_df = combined_df.sort_values('integrated_score', ascending=False)
        
        return {
            'component_analysis': combined_df,
            'critical_paths': critical_paths_df,
            'topology_metrics': self.criticality_scorer.topology_metrics
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def plot_component_scores(analysis_results: pd.DataFrame, top_n: int = 15):
        """Plot component criticality scores"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        top_components = analysis_results.head(top_n)
        
        # Topology score
        axes[0].barh(top_components['component'], 
                    top_components['topology_score'],
                    color='steelblue')
        axes[0].set_xlabel('Topology Score')
        axes[0].set_title('Topological Criticality')
        axes[0].invert_yaxis()
        
        # QoS importance
        axes[1].barh(top_components['component'], 
                    top_components['qos_score'],
                    color='coral')
        axes[1].set_xlabel('QoS Score')
        axes[1].set_title('QoS-based Importance')
        axes[1].invert_yaxis()
        
        # Impact score
        axes[2].barh(top_components['component'], 
                    top_components['impact_score'],
                    color='darkred')
        axes[2].set_xlabel('Impact Score')
        axes[2].set_title('Reachability Impact')
        axes[2].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_integrated_scores(analysis_results: pd.DataFrame, top_n: int = 15):
        """Plot integrated criticality scores"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_components = analysis_results.head(top_n)
        
        x = np.arange(len(top_components))
        width = 0.35
        
        ax.barh(x - width/2, top_components['composite_score'], 
               width, label='Composite Score', color='steelblue', alpha=0.8)
        ax.barh(x + width/2, top_components['integrated_score'], 
               width, label='Integrated Score', color='darkgreen', alpha=0.8)
        
        ax.set_ylabel('Component')
        ax.set_xlabel('Score')
        ax.set_title(f'Top {top_n} Critical Components')
        ax.set_yticks(x)
        ax.set_yticklabels(top_components['component'])
        ax.legend()
        ax.invert_yaxis()
        
        # Highlight articulation points
        for i, (idx, row) in enumerate(top_components.iterrows()):
            if row['is_articulation_point']:
                ax.plot(1.05, i, 'r*', markersize=15)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_graph_with_scores(model: PubSubGraphModel, scores: Dict[str, float]):
        """Visualize graph with node sizes based on criticality"""
        G = model.composite_graph
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Node colors by type
        color_map = {
            'topic': 'lightblue',
            'broker': 'lightgreen',
            'node': 'lightcoral',
            'application': 'lightyellow'
        }
        
        node_colors = [color_map.get(G.nodes[n].get('type', 'unknown'), 'gray') 
                      for n in G.nodes()]
        
        # Node sizes based on scores
        node_sizes = [scores.get(n, 0.1) * 3000 + 500 for n in G.nodes()]
        
        plt.figure(figsize=(16, 12))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, 
                              arrowsize=10, width=1.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9,
                              edgecolors='black', linewidths=2)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title('Pub-Sub System Graph (Node size = Criticality)', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def create_example_pubsub_system() -> PubSubGraphModel:
    """Create example distributed pub-sub system"""
    model = PubSubGraphModel()
    
    # Physical nodes
    model.add_physical_node('node1', {'datacenter': 'DC1'})
    model.add_physical_node('node2', {'datacenter': 'DC1'})
    model.add_physical_node('node3', {'datacenter': 'DC2'})
    
    # Brokers
    model.add_broker('broker1', 'node1', {'capacity': 1000})
    model.add_broker('broker2', 'node2', {'capacity': 1000})
    model.add_broker('broker3', 'node3', {'capacity': 800})
    
    # Broker replication
    model.add_broker_link('broker1', 'broker2')
    model.add_broker_link('broker2', 'broker3')
    
    # Topics with different QoS profiles
    model.add_topic('payment_events', 'broker1', 
                   QoSProfile(latency_requirement=50, throughput_requirement=5000,
                            reliability=0.99, priority=5))
    
    model.add_topic('user_analytics', 'broker1',
                   QoSProfile(latency_requirement=500, throughput_requirement=1000,
                            reliability=0.95, priority=3))
    
    model.add_topic('inventory_updates', 'broker2',
                   QoSProfile(latency_requirement=100, throughput_requirement=3000,
                            reliability=0.98, priority=4))
    
    model.add_topic('logs', 'broker3',
                   QoSProfile(latency_requirement=1000, throughput_requirement=500,
                            reliability=0.90, priority=2))
    
    model.add_topic('notifications', 'broker2',
                   QoSProfile(latency_requirement=200, throughput_requirement=2000,
                            reliability=0.96, priority=3))
    
    # Publishers
    model.add_application('payment_service', 'publisher', {'criticality': 'high'})
    model.add_application('analytics_service', 'publisher', {'criticality': 'medium'})
    model.add_application('inventory_service', 'publisher', {'criticality': 'high'})
    model.add_application('app_servers', 'publisher', {'criticality': 'medium'})
    
    # Subscribers
    model.add_application('fraud_detection', 'subscriber', {'criticality': 'high'})
    model.add_application('dashboard', 'subscriber', {'criticality': 'medium'})
    model.add_application('warehouse_system', 'subscriber', {'criticality': 'high'})
    model.add_application('notification_service', 'subscriber', {'criticality': 'medium'})
    model.add_application('audit_system', 'subscriber', {'criticality': 'low'})
    
    # Publish relationships
    model.add_publish_relation('payment_service', 'payment_events', rate=1000)
    model.add_publish_relation('analytics_service', 'user_analytics', rate=500)
    model.add_publish_relation('inventory_service', 'inventory_updates', rate=800)
    model.add_publish_relation('app_servers', 'logs', rate=2000)
    model.add_publish_relation('payment_service', 'notifications', rate=300)
    
    # Subscribe relationships
    model.add_subscribe_relation('payment_events', 'fraud_detection')
    model.add_subscribe_relation('user_analytics', 'dashboard')
    model.add_subscribe_relation('inventory_updates', 'warehouse_system')
    model.add_subscribe_relation('logs', 'audit_system')
    model.add_subscribe_relation('notifications', 'notification_service')
    model.add_subscribe_relation('payment_events', 'audit_system')
    model.add_subscribe_relation('inventory_updates', 'dashboard')
    
    return model


def main():
    """Run complete analysis"""
    print("="*80)
    print("CRITICAL COMPONENT ANALYSIS FOR DISTRIBUTED PUB-SUB SYSTEMS")
    print("="*80)
    print()
    
    # Create example system
    print("Creating example pub-sub system...")
    model = create_example_pubsub_system()
    print(f"System created with {len(model.composite_graph.nodes())} components")
    print(f"- Physical nodes: {len(model.physical_layer.nodes())}")
    print(f"- Brokers: {len(model.broker_layer.nodes())}")
    print(f"- Topics: {len(model.logical_layer.nodes())}")
    print(f"- Applications: {len(model.application_layer.nodes())}")
    print()
    
    # Run integrated analysis
    print("Running integrated analysis...")
    analyzer = IntegratedAnalysis(model)
    results = analyzer.run_complete_analysis()
    print()
    
    # Display results
    print("="*80)
    print("TOP CRITICAL COMPONENTS")
    print("="*80)
    print(results['component_analysis'][
        ['component', 'type', 'topology_score', 'qos_score', 
         'impact_score', 'integrated_score', 'is_articulation_point']
    ].head(10).to_string(index=False))
    print()
    
    print("="*80)
    print("CRITICAL PATHS (Publisher -> Subscriber)")
    print("="*80)
    print(results['critical_paths'][
        ['publisher', 'subscriber', 'path_length', 'criticality_score']
    ].head(10).to_string(index=False))
    print()
    
    print("="*80)
    print("TOPOLOGY METRICS FOR TOP COMPONENTS")
    print("="*80)
    top_components = results['component_analysis'].head(5)['component'].tolist()
    print(results['topology_metrics'].loc[top_components].to_string())
    print()
    
    # Visualizations
    print("Generating visualizations...")
    viz = Visualizer()
    
    viz.plot_component_scores(results['component_analysis'])
    viz.plot_integrated_scores(results['component_analysis'])
    
    # Graph visualization
    scores = dict(zip(results['component_analysis']['component'],
                     results['component_analysis']['integrated_score']))
    viz.plot_graph_with_scores(model, scores)
    
    print("\nAnalysis complete!")
    
    return model, results


# Run the analysis
if __name__ == "__main__":
    model, results = main()
