import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class NodeType(Enum):
    """Node types in the publish-subscribe system"""
    APPLICATION = "application"
    BROKER = "broker"
    TOPIC = "topic"
    MACHINE = "machine"


class EdgeType(Enum):
    """Edge types representing relationships"""
    PUBLISHES_TO = "publishes_to"
    SUBSCRIBES_TO = "subscribes_to"
    ROUTES = "routes"
    RUNS_ON = "runs_on"
    DEPENDS_ON = "depends_on"
    CONNECTS_TO = "connects_to"


class CriticalityLevel(Enum):
    """Criticality classification levels"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


@dataclass
class QoSPolicy:
    """QoS Policy configuration for topics"""
    durability: float  # 0-1: 0=volatile, 1=persistent
    reliability: float  # 0-1: 0=best_effort, 1=reliable
    transport_priority: float  # 0-10: higher is more critical
    deadline: float  # 0-1: normalized deadline constraint
    lifespan: float  # 0-1: normalized message lifespan
    history: float  # 0-1: history depth importance
    
    def to_dict(self):
        return {
            'durability': self.durability,
            'reliability': self.reliability,
            'transport_priority': self.transport_priority / 10,  # Normalize to 0-1
            'deadline': self.deadline,
            'lifespan': self.lifespan,
            'history': self.history
        }


class PubSubGraphAnalyzer:
    """
    Comprehensive analyzer for distributed publish-subscribe systems using graph-based metrics
    and QoS-aware analysis to identify critical components and message flows.
    """
    
    def __init__(self, graph: nx.Graph = None):
        """
        Initialize the analyzer with an optional graph.
        
        Args:
            graph: NetworkX graph representing the pub-sub system
        """
        self.graph = graph if graph else nx.Graph()
        self.node_criticality_scores = {}
        self.topic_criticality_scores = {}
        self.edge_criticality_scores = {}
        self.criticality_classifications = {}
        
        # Weights for different metrics in criticality calculation
        self.metric_weights = {
            'degree_centrality': 0.15,
            'betweenness_centrality': 0.20,
            'closeness_centrality': 0.15,
            'eigenvector_centrality': 0.15,
            'pagerank': 0.15,
            'is_articulation_point': 0.20
        }
        
        # QoS policy weights for topic criticality
        self.qos_weights = {
            'durability': 0.20,
            'reliability': 0.25,
            'transport_priority': 0.15,
            'deadline': 0.20,
            'lifespan': 0.10,
            'history': 0.10
        }
    
    def create_sample_pubsub_graph(self) -> nx.Graph:
        """
        Create a sample publish-subscribe system graph for demonstration.
        
        Returns:
            NetworkX graph with nodes and edges representing a pub-sub system
        """
        G = nx.Graph()
        
        # Add nodes with types
        # Applications
        apps = ['app1', 'app2', 'app3', 'app4', 'app5']
        for app in apps:
            G.add_node(app, node_type=NodeType.APPLICATION.value, label=f"App_{app}")
        
        # Brokers
        brokers = ['broker1', 'broker2', 'broker3']
        for broker in brokers:
            G.add_node(broker, node_type=NodeType.BROKER.value, label=f"Broker_{broker}")
        
        # Topics with QoS policies
        topics_qos = {
            'topic1': QoSPolicy(1.0, 1.0, 10, 0.9, 0.8, 0.7),  # Critical topic
            'topic2': QoSPolicy(0.5, 0.7, 5, 0.5, 0.5, 0.5),   # Medium priority
            'topic3': QoSPolicy(0.0, 0.3, 2, 0.2, 0.3, 0.2),   # Low priority
            'topic4': QoSPolicy(0.8, 0.9, 8, 0.8, 0.7, 0.6),   # High priority
            'topic5': QoSPolicy(0.3, 0.4, 3, 0.4, 0.4, 0.4)    # Low-medium priority
        }
        
        for topic, qos in topics_qos.items():
            G.add_node(topic, node_type=NodeType.TOPIC.value, 
                      label=f"Topic_{topic}", qos_policy=qos)
        
        # Machines
        machines = ['machine1', 'machine2', 'machine3', 'machine4']
        for machine in machines:
            G.add_node(machine, node_type=NodeType.MACHINE.value, 
                      label=f"Machine_{machine}")
        
        # Add edges
        # Publishers (App -> Topic)
        pub_edges = [
            ('app1', 'topic1'), ('app1', 'topic2'),
            ('app2', 'topic2'), ('app2', 'topic4'),
            ('app3', 'topic3'), ('app3', 'topic5')
        ]
        for src, dst in pub_edges:
            G.add_edge(src, dst, edge_type=EdgeType.PUBLISHES_TO.value)
        
        # Subscribers (App -> Topic)
        sub_edges = [
            ('app3', 'topic1'), ('app4', 'topic1'),
            ('app4', 'topic2'), ('app5', 'topic3'),
            ('app5', 'topic4'), ('app5', 'topic5')
        ]
        for src, dst in sub_edges:
            G.add_edge(src, dst, edge_type=EdgeType.SUBSCRIBES_TO.value)
        
        # Broker routing (Broker -> Topic)
        route_edges = [
            ('broker1', 'topic1'), ('broker1', 'topic2'),
            ('broker2', 'topic2'), ('broker2', 'topic3'), ('broker2', 'topic4'),
            ('broker3', 'topic4'), ('broker3', 'topic5')
        ]
        for src, dst in route_edges:
            G.add_edge(src, dst, edge_type=EdgeType.ROUTES.value)
        
        # Deployment (App/Broker -> Machine)
        deploy_edges = [
            ('app1', 'machine1'), ('app2', 'machine1'),
            ('app3', 'machine2'), ('app4', 'machine3'),
            ('app5', 'machine3'), ('broker1', 'machine1'),
            ('broker2', 'machine2'), ('broker3', 'machine4')
        ]
        for src, dst in deploy_edges:
            G.add_edge(src, dst, edge_type=EdgeType.RUNS_ON.value)
        
        # Machine connections
        machine_edges = [
            ('machine1', 'machine2'), ('machine2', 'machine3'),
            ('machine3', 'machine4'), ('machine1', 'machine4')
        ]
        for src, dst in machine_edges:
            G.add_edge(src, dst, edge_type=EdgeType.CONNECTS_TO.value)
        
        self.graph = G
        return G
    
    def calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate various centrality metrics for all nodes.
        
        Returns:
            Dictionary mapping metric names to node scores
        """
        metrics = {}
        
        # Degree Centrality
        metrics['degree_centrality'] = nx.degree_centrality(self.graph)
        
        # Betweenness Centrality
        metrics['betweenness_centrality'] = nx.betweenness_centrality(self.graph)
        
        # Closeness Centrality
        metrics['closeness_centrality'] = nx.closeness_centrality(self.graph)
        
        # Eigenvector Centrality
        try:
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(
                self.graph, max_iter=1000, tol=1e-06
            )
        except:
            # Fallback to degree centrality if eigenvector fails to converge
            metrics['eigenvector_centrality'] = metrics['degree_centrality']
        
        # PageRank
        metrics['pagerank'] = nx.pagerank(self.graph)
        
        return metrics
    
    def find_articulation_points(self) -> List[str]:
        """
        Find articulation points (nodes whose removal disconnects the graph).
        
        Returns:
            List of articulation point node IDs
        """
        return list(nx.articulation_points(self.graph.to_undirected()))
    
    def calculate_node_criticality(self) -> Dict[str, float]:
        """
        Calculate criticality score for each node based on centrality metrics
        and articulation point status.
        
        Returns:
            Dictionary mapping node IDs to criticality scores
        """
        # Get centrality metrics
        centrality_metrics = self.calculate_centrality_metrics()
        
        # Get articulation points
        articulation_points = self.find_articulation_points()
        
        # Initialize scores
        criticality_scores = {}
        
        for node in self.graph.nodes():
            score = 0.0
            
            # Add weighted centrality scores
            for metric, weight in self.metric_weights.items():
                if metric == 'is_articulation_point':
                    # Binary feature for articulation point
                    score += weight * (1.0 if node in articulation_points else 0.0)
                else:
                    # Centrality metric
                    if metric in centrality_metrics:
                        score += weight * centrality_metrics[metric].get(node, 0)
            
            criticality_scores[node] = score
        
        self.node_criticality_scores = criticality_scores
        return criticality_scores
    
    def calculate_topic_criticality(self) -> Dict[str, float]:
        """
        Calculate criticality score for topics based on QoS policies and
        structural importance.
        
        Returns:
            Dictionary mapping topic node IDs to criticality scores
        """
        topic_scores = {}
        
        # Get node criticality scores first if not already calculated
        if not self.node_criticality_scores:
            self.calculate_node_criticality()
        
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == NodeType.TOPIC.value:
                # Base structural score
                structural_score = self.node_criticality_scores.get(node, 0)
                
                # QoS-based score
                qos_score = 0.0
                if 'qos_policy' in attrs:
                    qos_policy = attrs['qos_policy']
                    qos_dict = qos_policy.to_dict()
                    
                    for policy, weight in self.qos_weights.items():
                        qos_score += weight * qos_dict.get(policy, 0)
                
                # Combine structural and QoS scores
                # 60% QoS, 40% structural for topics
                combined_score = 0.6 * qos_score + 0.4 * structural_score
                topic_scores[node] = combined_score
        
        self.topic_criticality_scores = topic_scores
        return topic_scores
    
    def calculate_path_criticality(self, source: str, target: str) -> float:
        """
        Calculate criticality of a path between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Path criticality score
        """
        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=5))
            
            if not paths:
                return 0.0
            
            # Calculate criticality for each path
            path_scores = []
            for path in paths:
                # Average node criticality along the path
                node_scores = [self.node_criticality_scores.get(n, 0) for n in path]
                
                # Check for critical topics in path
                topic_boost = 0
                for node in path:
                    if node in self.topic_criticality_scores:
                        topic_boost = max(topic_boost, 
                                        self.topic_criticality_scores[node])
                
                # Path score is average node score with topic boost
                path_score = np.mean(node_scores) + 0.2 * topic_boost
                path_scores.append(path_score)
            
            # Return maximum criticality among all paths
            return max(path_scores)
            
        except nx.NetworkXNoPath:
            return 0.0
    
    def calculate_edge_criticality(self) -> Dict[Tuple[str, str], float]:
        """
        Calculate criticality scores for edges based on connected node criticality
        and edge type importance.
        
        Returns:
            Dictionary mapping edge tuples to criticality scores
        """
        edge_scores = {}
        
        # Edge type weights (how critical each relationship type is)
        edge_type_weights = {
            EdgeType.PUBLISHES_TO.value: 0.9,
            EdgeType.SUBSCRIBES_TO.value: 0.9,
            EdgeType.ROUTES.value: 1.0,
            EdgeType.RUNS_ON.value: 0.7,
            EdgeType.DEPENDS_ON.value: 0.8,
            EdgeType.CONNECTS_TO.value: 0.6
        }
        
        for u, v, attrs in self.graph.edges(data=True):
            # Get node criticalities
            u_score = self.node_criticality_scores.get(u, 0)
            v_score = self.node_criticality_scores.get(v, 0)
            
            # Get edge type weight
            edge_type = attrs.get('edge_type', EdgeType.CONNECTS_TO.value)
            type_weight = edge_type_weights.get(edge_type, 0.5)
            
            # Edge criticality is weighted combination
            edge_score = type_weight * (u_score + v_score) / 2
            
            # Check if edge connects to critical topic
            if u in self.topic_criticality_scores:
                edge_score *= (1 + self.topic_criticality_scores[u])
            if v in self.topic_criticality_scores:
                edge_score *= (1 + self.topic_criticality_scores[v])
            
            edge_scores[(u, v)] = edge_score
        
        self.edge_criticality_scores = edge_scores
        return edge_scores
    
    def classify_criticality_boxplot(self, scores: Dict[str, float]) -> Dict[str, str]:
        """
        Classify components into criticality levels using box plot statistical method.
        
        Args:
            scores: Dictionary of component IDs to criticality scores
            
        Returns:
            Dictionary mapping component IDs to criticality levels
        """
        if not scores:
            return {}
        
        # Convert to array for statistical analysis
        values = np.array(list(scores.values()))
        
        # Calculate quartiles and IQR
        q1 = np.percentile(values, 25)
        q2 = np.percentile(values, 50)  # median
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        # Define thresholds based on box plot method
        # Outliers on lower end: below Q1 - 1.5*IQR
        very_low_threshold = q1 - 1.5 * iqr
        # Between Q1 - 1.5*IQR and Q1
        low_threshold = q1
        # Between Q1 and Q2 (median)
        medium_low_threshold = q2
        # Between Q2 and Q3
        medium_high_threshold = q3
        # Above Q3 but below Q3 + 1.5*IQR
        high_threshold = q3 + 1.5 * iqr
        # Outliers on upper end: above Q3 + 1.5*IQR
        
        classifications = {}
        for component, score in scores.items():
            if score <= very_low_threshold:
                level = CriticalityLevel.VERY_LOW
            elif score <= low_threshold:
                level = CriticalityLevel.LOW
            elif score <= medium_low_threshold:
                level = CriticalityLevel.MEDIUM
            elif score <= medium_high_threshold:
                level = CriticalityLevel.HIGH
            elif score > high_threshold:
                level = CriticalityLevel.VERY_HIGH
            else:
                level = CriticalityLevel.HIGH
            
            classifications[component] = level.value
        
        return classifications
    
    def analyze_system(self) -> Dict[str, Any]:
        """
        Perform complete system analysis including all criticality calculations
        and classifications.
        
        Returns:
            Comprehensive analysis results
        """
        print("Starting comprehensive system analysis...")
        
        # Calculate all criticality scores
        print("1. Calculating node criticality scores...")
        node_scores = self.calculate_node_criticality()
        
        print("2. Calculating topic criticality scores...")
        topic_scores = self.calculate_topic_criticality()
        
        print("3. Calculating edge criticality scores...")
        edge_scores = self.calculate_edge_criticality()
        
        # Classify components
        print("4. Classifying node criticality levels...")
        node_classifications = self.classify_criticality_boxplot(node_scores)
        
        print("5. Classifying topic criticality levels...")
        topic_classifications = self.classify_criticality_boxplot(topic_scores)
        
        # Find critical components
        articulation_points = self.find_articulation_points()
        
        # Identify most critical paths (example between apps)
        critical_paths = []
        app_nodes = [n for n, d in self.graph.nodes(data=True) 
                     if d.get('node_type') == NodeType.APPLICATION.value]
        
        for i, src in enumerate(app_nodes):
            for dst in app_nodes[i+1:]:
                path_score = self.calculate_path_criticality(src, dst)
                if path_score > 0:
                    critical_paths.append({
                        'source': src,
                        'target': dst,
                        'criticality': path_score
                    })
        
        # Sort critical paths by score
        critical_paths.sort(key=lambda x: x['criticality'], reverse=True)
        
        # Prepare results
        results = {
            'node_criticality_scores': node_scores,
            'topic_criticality_scores': topic_scores,
            'edge_criticality_scores': {f"{u}-{v}": score 
                                       for (u, v), score in edge_scores.items()},
            'node_classifications': node_classifications,
            'topic_classifications': topic_classifications,
            'articulation_points': articulation_points,
            'critical_paths': critical_paths[:10],  # Top 10 critical paths
            'statistics': {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'num_articulation_points': len(articulation_points),
                'avg_node_criticality': np.mean(list(node_scores.values())),
                'max_node_criticality': max(node_scores.values()) if node_scores else 0,
                'avg_topic_criticality': np.mean(list(topic_scores.values())) if topic_scores else 0,
            }
        }
        
        self.criticality_classifications = node_classifications
        
        print("\nAnalysis complete!")
        return results
    
    def visualize_criticality_distribution(self):
        """
        Visualize the distribution of criticality scores using box plots and histograms.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Node criticality distribution
        if self.node_criticality_scores:
            node_values = list(self.node_criticality_scores.values())
            node_labels = list(self.node_criticality_scores.keys())
            
            # Box plot
            axes[0, 0].boxplot(node_values, vert=True)
            axes[0, 0].set_title('Node Criticality Distribution (Box Plot)')
            axes[0, 0].set_ylabel('Criticality Score')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Histogram
            axes[1, 0].hist(node_values, bins=20, edgecolor='black', alpha=0.7)
            axes[1, 0].set_title('Node Criticality Distribution (Histogram)')
            axes[1, 0].set_xlabel('Criticality Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Topic criticality distribution
        if self.topic_criticality_scores:
            topic_values = list(self.topic_criticality_scores.values())
            
            # Box plot
            axes[0, 1].boxplot(topic_values, vert=True)
            axes[0, 1].set_title('Topic Criticality Distribution (Box Plot)')
            axes[0, 1].set_ylabel('Criticality Score')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Histogram
            axes[1, 1].hist(topic_values, bins=15, edgecolor='black', 
                          alpha=0.7, color='orange')
            axes[1, 1].set_title('Topic Criticality Distribution (Histogram)')
            axes[1, 1].set_xlabel('Criticality Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Classification distribution
        if self.criticality_classifications:
            class_counts = pd.Series(self.criticality_classifications.values()).value_counts()
            colors = {'VERY_LOW': 'green', 'LOW': 'lightgreen', 
                     'MEDIUM': 'yellow', 'HIGH': 'orange', 'VERY_HIGH': 'red'}
            
            # Bar chart for classifications
            bars = axes[0, 2].bar(range(len(class_counts)), class_counts.values)
            axes[0, 2].set_xticks(range(len(class_counts)))
            axes[0, 2].set_xticklabels(class_counts.index, rotation=45)
            axes[0, 2].set_title('Criticality Level Distribution')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Color bars based on criticality level
            for i, (bar, label) in enumerate(zip(bars, class_counts.index)):
                bar.set_color(colors.get(label, 'gray'))
            
            # Pie chart for classifications
            axes[1, 2].pie(class_counts.values, labels=class_counts.index,
                          autopct='%1.1f%%', startangle=90,
                          colors=[colors.get(l, 'gray') for l in class_counts.index])
            axes[1, 2].set_title('Criticality Level Proportions')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable report of the analysis results.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*60)
        report.append("PUBLISH-SUBSCRIBE SYSTEM CRITICALITY ANALYSIS REPORT")
        report.append("="*60)
        
        # System Overview
        report.append("\n## SYSTEM OVERVIEW")
        report.append(f"Total Nodes: {results['statistics']['total_nodes']}")
        report.append(f"Total Edges: {results['statistics']['total_edges']}")
        report.append(f"Articulation Points: {results['statistics']['num_articulation_points']}")
        
        # Critical Components
        report.append("\n## CRITICAL COMPONENTS")
        
        # Top critical nodes
        report.append("\n### Most Critical Nodes (Top 5):")
        sorted_nodes = sorted(results['node_criticality_scores'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        for i, (node, score) in enumerate(sorted_nodes, 1):
            classification = results['node_classifications'].get(node, 'UNKNOWN')
            node_type = self.graph.nodes[node].get('node_type', 'unknown')
            report.append(f"  {i}. {node} (Type: {node_type})")
            report.append(f"     Score: {score:.4f} | Level: {classification}")
        
        # Critical topics
        if results['topic_criticality_scores']:
            report.append("\n### Critical Topics:")
            sorted_topics = sorted(results['topic_criticality_scores'].items(), 
                                 key=lambda x: x[1], reverse=True)
            for topic, score in sorted_topics:
                classification = results['topic_classifications'].get(topic, 'UNKNOWN')
                report.append(f"  - {topic}: Score={score:.4f}, Level={classification}")
        
        # Articulation points
        report.append("\n### Articulation Points (Single Points of Failure):")
        for ap in results['articulation_points']:
            node_type = self.graph.nodes[ap].get('node_type', 'unknown')
            report.append(f"  - {ap} (Type: {node_type})")
        
        # Critical Paths
        if results['critical_paths']:
            report.append("\n### Most Critical Message Paths (Top 5):")
            for i, path in enumerate(results['critical_paths'][:5], 1):
                report.append(f"  {i}. {path['source']} -> {path['target']}")
                report.append(f"     Criticality: {path['criticality']:.4f}")
        
        # Classification Summary
        report.append("\n## CRITICALITY CLASSIFICATION SUMMARY")
        class_counts = pd.Series(results['node_classifications'].values()).value_counts()
        for level in ['VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW']:
            count = class_counts.get(level, 0)
            percentage = (count / len(results['node_classifications'])) * 100
            report.append(f"  {level:10s}: {count:3d} components ({percentage:5.1f}%)")
        
        # Recommendations
        report.append("\n## RECOMMENDATIONS")
        
        if results['articulation_points']:
            report.append("\n### High Priority:")
            report.append("  - Add redundancy for articulation points to eliminate")
            report.append("    single points of failure")
        
        very_high_critical = [n for n, c in results['node_classifications'].items() 
                            if c == 'VERY_HIGH']
        if very_high_critical:
            report.append(f"  - Monitor and add failover for {len(very_high_critical)} ")
            report.append(f"    VERY_HIGH criticality components")
        
        high_critical_topics = [t for t, c in results['topic_classifications'].items() 
                               if c in ['HIGH', 'VERY_HIGH']]
        if high_critical_topics:
            report.append("\n### Medium Priority:")
            report.append(f"  - Ensure proper replication for {len(high_critical_topics)}")
            report.append(f"    critical topics")
            report.append("  - Consider load balancing for high-traffic topics")
        
        report.append("\n### Monitoring Focus:")
        report.append("  - Implement enhanced monitoring for all HIGH and VERY_HIGH")
        report.append("    criticality components")
        report.append("  - Set up alerting for critical path disruptions")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


# Example usage and demonstration
def main():
    """
    Demonstrate the publish-subscribe system criticality analysis.
    """
    # Create analyzer instance
    analyzer = PubSubGraphAnalyzer()
    
    # Create sample graph
    print("Creating sample publish-subscribe system graph...")
    graph = analyzer.create_sample_pubsub_graph()
    print(f"Created graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Perform analysis
    print("\n" + "="*60)
    results = analyzer.analyze_system()
    
    # Generate and print report
    report = analyzer.generate_report(results)
    print(report)
    
    # Visualize distributions
    print("\nGenerating criticality distribution visualizations...")
    analyzer.visualize_criticality_distribution()
    
    # Example: Analyze specific path criticality
    print("\n" + "="*60)
    print("EXAMPLE: Path Criticality Analysis")
    print("="*60)
    
    source_app = 'app1'
    target_app = 'app5'
    path_criticality = analyzer.calculate_path_criticality(source_app, target_app)
    print(f"Path criticality from {source_app} to {target_app}: {path_criticality:.4f}")
    
    # Display detailed metrics for a specific critical node
    critical_node = max(results['node_criticality_scores'].items(), 
                       key=lambda x: x[1])[0]
    
    print(f"\nDetailed Analysis for Most Critical Node: {critical_node}")
    print("-"*40)
    
    # Get all metrics for this node
    centrality_metrics = analyzer.calculate_centrality_metrics()
    for metric_name, node_scores in centrality_metrics.items():
        if critical_node in node_scores:
            print(f"  {metric_name:25s}: {node_scores[critical_node]:.4f}")
    
    is_articulation = critical_node in analyzer.find_articulation_points()
    print(f"  {'is_articulation_point':25s}: {is_articulation}")
    print(f"  {'final_criticality_score':25s}: {results['node_criticality_scores'][critical_node]:.4f}")
    print(f"  {'criticality_level':25s}: {results['node_classifications'][critical_node]}")
    
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()