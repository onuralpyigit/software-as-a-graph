import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict

class FailureSimulator:
    def __init__(self, graph):
        """
        Initialize the failure simulator with a NetworkX graph representation of the pub-sub system.
        
        Args:
            graph: NetworkX DiGraph with nodes representing system components and edges representing relationships
        """
        self.graph = graph.copy()
        self.original_graph = graph.copy()
        self.node_types = nx.get_node_attributes(self.graph, 'type')
        self.output_file_dir = "output/"
        
    def reset_graph(self):
        """Reset the graph to its original state"""
        self.graph = self.original_graph.copy()
        
    def simulate_node_failure(self, node_id):
        """
        Simulate the failure of a specific node and its impact
        
        Args:
            node_id: ID of the node to simulate failure for
            
        Returns:
            dict: Impact metrics of the failure
        """
        # Create a copy of the original graph
        self.reset_graph()
        
        # Store pre-failure metrics
        pre_failure_metrics = self.calculate_system_metrics()
        
        # Identify the type of the failing component
        node_type = self.node_types.get(node_id)
        
        # Remove the node and its relationships
        if node_type == 'Node':
            # If it's a physical node, also remove all applications and brokers running on it
            components_to_remove = [node_id]
            for u, v, rel in self.graph.edges(data=True):
                if rel.get('type') == 'RUNS_ON' and v == node_id:
                    components_to_remove.append(u)
            
            for component in components_to_remove:
                if self.graph.has_node(component):
                    self.graph.remove_node(component)
        else:
            # For other component types, just remove the node
            self.graph.remove_node(node_id)
        
        # Calculate post-failure metrics
        post_failure_metrics = self.calculate_system_metrics()
        
        # Calculate impact metrics (degradation percentages)
        impact = self._calculate_impact(pre_failure_metrics, post_failure_metrics)
        impact['failed_component'] = node_id
        impact['component_type'] = node_type
        
        return impact
    
    def simulate_all_failures(self):
        """
        Simulate failures for each component in the system
        
        Returns:
            dict: Components sorted by their impact
        """
        impact_results = []
        
        # For each node in the original graph
        for node_id in self.original_graph.nodes():
            # Skip topic nodes for simplicity (or include them if needed)
            if self.node_types.get(node_id) == 'Topic':
                continue
                
            # Simulate failure and collect impact metrics
            impact = self.simulate_node_failure(node_id)
            impact_results.append(impact)
        
        # Convert to DataFrame for easier analysis
        impact_df = pd.DataFrame(impact_results)
        
        # Sort by overall impact
        impact_df = impact_df.sort_values('overall_impact', ascending=False)
        
        return impact_df
    
    def calculate_system_metrics(self):
        """
        Calculate various system health metrics
        
        Returns:
            dict: Various system health metrics
        """
        metrics = {}
        
        # Get node lists by type
        apps = [n for n, t in self.node_types.items() if t == 'Application' and n in self.graph]
        topics = [n for n, t in self.node_types.items() if t == 'Topic' and n in self.graph]
        brokers = [n for n, t in self.node_types.items() if t == 'Broker' and n in self.graph]
        nodes = [n for n, t in self.node_types.items() if t == 'Node' and n in self.graph]
        
        # Count active components
        metrics['active_applications'] = len(apps)
        metrics['active_topics'] = len(topics)
        metrics['active_brokers'] = len(brokers)
        metrics['active_nodes'] = len(nodes)
        
        # Calculate reachability metrics
        pub_sub_paths = 0
        possible_paths = 0
        
        # Message delivery ratio
        for app1 in apps:
            for app2 in apps:
                if app1 != app2:
                    # Check if app1 can send messages to app2 through any topic
                    app_can_communicate = False
                    
                    # Get topics app1 publishes to
                    pub_topics = []
                    for _, t, rel in self.graph.out_edges(app1, data=True):
                        if rel.get('type') == 'PUBLISHES_TO' and t in self.graph:
                            pub_topics.append(t)
                    
                    # Get topics app2 subscribes to
                    sub_topics = []
                    for _, t, rel in self.graph.out_edges(app2, data=True):
                        if rel.get('type') == 'SUBSCRIBES_TO' and t in self.graph:
                            sub_topics.append(t)
                    
                    # Check if there's a common topic with broker routing
                    for topic in set(pub_topics).intersection(sub_topics):
                        # Check if any broker routes this topic
                        for broker in brokers:
                            for _, t, rel in self.graph.out_edges(broker, data=True):
                                if rel.get('type') == 'ROUTES' and t == topic:
                                    app_can_communicate = True
                                    break
                    
                    if app_can_communicate:
                        pub_sub_paths += 1
                    
                    possible_paths += 1
        
        # Message delivery ratio (reachability)
        metrics['message_delivery_ratio'] = pub_sub_paths / possible_paths if possible_paths > 0 else 0
        
        # Topic availability ratio
        routed_topics = 0
        for topic in topics:
            is_routed = False
            for broker in brokers:
                for _, t, rel in self.graph.out_edges(broker, data=True):
                    if rel.get('type') == 'ROUTES' and t == topic:
                        is_routed = True
                        break
            if is_routed:
                routed_topics += 1
        
        metrics['topic_availability_ratio'] = routed_topics / len(topics) if topics else 0
        
        # Broker redundancy (average number of brokers per topic)
        total_routes = 0
        for topic in topics:
            broker_count = 0
            for broker in brokers:
                for _, t, rel in self.graph.out_edges(broker, data=True):
                    if rel.get('type') == 'ROUTES' and t == topic:
                        broker_count += 1
            total_routes += broker_count
        
        metrics['broker_redundancy'] = total_routes / len(topics) if topics else 0
        
        # Network connectivity (for physical nodes)
        if len(nodes) > 1:
            node_connectivity = 0
            for n1 in nodes:
                for n2 in nodes:
                    if n1 != n2:
                        for _, dest, rel in self.graph.out_edges(n1, data=True):
                            if rel.get('type') == 'CONNECTS_TO' and dest == n2:
                                node_connectivity += 1
            
            max_possible_connections = len(nodes) * (len(nodes) - 1)
            metrics['network_connectivity'] = node_connectivity / max_possible_connections
        else:
            metrics['network_connectivity'] = 0
        
        return metrics
    
    def _calculate_impact(self, pre_metrics, post_metrics):
        """
        Calculate the impact of a failure by comparing pre and post metrics
        
        Args:
            pre_metrics: Metrics before failure
            post_metrics: Metrics after failure
            
        Returns:
            dict: Impact measurements as degradation percentages
        """
        impact = {}
        
        # Calculate percentage degradation for each metric
        for metric in pre_metrics:
            if pre_metrics[metric] > 0:  # Avoid division by zero
                degradation = (pre_metrics[metric] - post_metrics[metric]) / pre_metrics[metric]
                impact[f'{metric}_degradation'] = degradation * 100  # as percentage
            else:
                impact[f'{metric}_degradation'] = 0
        
        # Calculate overall impact as weighted average of key metrics
        weights = {
            'message_delivery_ratio_degradation': 0.4,
            'topic_availability_ratio_degradation': 0.3,
            'broker_redundancy_degradation': 0.2,
            'network_connectivity_degradation': 0.1
        }
        
        overall_impact = 0
        for metric, weight in weights.items():
            if metric in impact:
                overall_impact += impact[metric] * weight
        
        impact['overall_impact'] = overall_impact
        
        return impact
    
    def visualize_impact_results(self, impact_df, top_n=10):
        """
        Visualize the impact results
        
        Args:
            impact_df: DataFrame with impact results
            top_n: Number of top components to display
        """
        # Select top components
        top_components = impact_df.head(top_n)
        
        # Format node names for better display
        node_names = []
        for _, row in top_components.iterrows():
            name = str(row['failed_component'])
            if len(name) > 15:  # Truncate long names
                name = name[:12] + '...'
            node_names.append(f"{name}\n({row['component_type']})")
        
        # Key metrics to display
        metrics = [
            'message_delivery_ratio_degradation', 
            'topic_availability_ratio_degradation',
            'broker_redundancy_degradation', 
            'network_connectivity_degradation'
        ]
        
        metric_labels = {
            'message_delivery_ratio_degradation': 'Message Delivery',
            'topic_availability_ratio_degradation': 'Topic Availability', 
            'broker_redundancy_degradation': 'Broker Redundancy',
            'network_connectivity_degradation': 'Network Connectivity'
        }
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Create heatmap data
        heatmap_data = []
        for _, row in top_components.iterrows():
            heatmap_data.append([row[m] for m in metrics])
        
        # Create heatmap
        ax = sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                         xticklabels=[metric_labels[m] for m in metrics],
                         yticklabels=node_names)
        
        plt.title('Impact of Component Failures on System Quality Metrics (%)', fontsize=16)
        plt.tight_layout()
        output_file = self.output_file_dir + 'Impact of Component Failures on System Quality Metrics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        # Create bar chart for overall impact
        plt.figure(figsize=(14, 8))
        bars = plt.barh(node_names, top_components['overall_impact'], color='darkred')
        
        # Add value labels to the bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                     f'{width:.1f}%', ha='left', va='center')
        
        plt.xlabel('Overall Impact (Weighted Degradation %)', fontsize=12)
        plt.title('Critical Components by Failure Impact', fontsize=16)
        plt.xlim(0, max(top_components['overall_impact']) * 1.1)  # Add some padding
        plt.tight_layout()
        output_file = self.output_file_dir + 'Critical Components by Failure Impact.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    def identify_critical_components(self, impact_df, threshold_method='percentile', threshold_value=90):
        """
        Identify critical components based on their impact
        
        Args:
            impact_df: DataFrame with impact results
            threshold_method: Method to use for thresholding ('percentile', 'absolute', 'elbow')
            threshold_value: Value to use for thresholding
            
        Returns:
            DataFrame: Critical components
        """
        if threshold_method == 'percentile':
            # Use percentile threshold
            threshold = np.percentile(impact_df['overall_impact'], threshold_value)
            critical = impact_df[impact_df['overall_impact'] >= threshold]
            
        elif threshold_method == 'absolute':
            # Use absolute threshold (e.g., components causing >30% degradation)
            critical = impact_df[impact_df['overall_impact'] >= threshold_value]
            
        elif threshold_method == 'elbow':
            # Use elbow method to find natural threshold
            from kneed import KneeLocator
            
            # Sort values in descending order
            sorted_impact = sorted(impact_df['overall_impact'].values, reverse=True)
            indices = list(range(1, len(sorted_impact) + 1))
            
            # Find the elbow point
            kneedle = KneeLocator(indices, sorted_impact, curve='convex', direction='decreasing')
            elbow_index = kneedle.knee
            
            if elbow_index:
                threshold = sorted_impact[elbow_index - 1]
                critical = impact_df[impact_df['overall_impact'] >= threshold]
            else:
                # Fallback to percentile method if elbow not found
                threshold = np.percentile(impact_df['overall_impact'], 80)
                critical = impact_df[impact_df['overall_impact'] >= threshold]
        
        return critical
    
    def generate_resilience_recommendations(self, critical_components):
        """
        Generate recommendations to improve system resilience based on critical components
        
        Args:
            critical_components: DataFrame with critical components
            
        Returns:
            dict: Recommendations for system improvement
        """
        recommendations = {
            'broker_redundancy': [],
            'node_distribution': [],
            'topic_replication': [],
            'network_connectivity': []
        }
        
        # Analyze critical components by type
        for _, component in critical_components.iterrows():
            comp_id = component['failed_component']
            comp_type = component['component_type']
            
            if comp_type == 'Broker':
                # Check which topics are routed by this broker
                routed_topics = []
                for _, topic, rel in self.original_graph.out_edges(comp_id, data=True):
                    if rel.get('type') == 'ROUTES':
                        routed_topics.append(topic)
                
                recommendations['broker_redundancy'].append({
                    'broker': comp_id,
                    'impact': component['overall_impact'],
                    'routed_topics': len(routed_topics),
                    'recommendation': 'Add redundant broker for topics: ' + 
                                    ', '.join([str(t) for t in routed_topics[:5]]) +
                                    (f" and {len(routed_topics)-5} more" if len(routed_topics) > 5 else "")
                })
                
            elif comp_type == 'Node':
                # Check which components run on this node
                running_components = []
                broker_count = 0
                app_count = 0
                
                for src, dest, rel in self.original_graph.edges(data=True):
                    if rel.get('type') == 'RUNS_ON' and dest == comp_id:
                        running_components.append(src)
                        if self.node_types.get(src) == 'Broker':
                            broker_count += 1
                        elif self.node_types.get(src) == 'Application':
                            app_count += 1
                
                if broker_count > 0:
                    recommendations['node_distribution'].append({
                        'node': comp_id,
                        'impact': component['overall_impact'],
                        'brokers': broker_count,
                        'applications': app_count,
                        'recommendation': f'Distribute {broker_count} brokers and {app_count} ' +
                                        f'applications across multiple nodes to reduce single point of failure'
                    })
            
            elif comp_type == 'Application':
                # Check which topics this app publishes to
                pub_topics = []
                for _, topic, rel in self.original_graph.out_edges(comp_id, data=True):
                    if rel.get('type') == 'PUBLISHES_TO':
                        pub_topics.append(topic)
                
                if len(pub_topics) > 0:
                    recommendations['topic_replication'].append({
                        'application': comp_id,
                        'impact': component['overall_impact'],
                        'publishes_to': len(pub_topics),
                        'recommendation': 'Implement redundant publishers for critical topics'
                    })
        
        # Network connectivity recommendations based on overall analysis
        nodes = [n for n, t in self.node_types.items() if t == 'Node']
        connection_count = defaultdict(int)
        
        for n1 in nodes:
            for _, n2, rel in self.original_graph.out_edges(n1, data=True):
                if rel.get('type') == 'CONNECTS_TO':
                    connection_count[n1] += 1
        
        poorly_connected = []
        for node, count in connection_count.items():
            if count < len(nodes) / 2:  # Less than half of possible connections
                poorly_connected.append(node)
        
        if poorly_connected:
            recommendations['network_connectivity'].append({
                'nodes': poorly_connected,
                'recommendation': 'Increase network connectivity for poorly connected nodes: ' + 
                                  ', '.join([str(n) for n in poorly_connected])
            })
        
        return recommendations

def generate_example_graph():
    # Create a simple example graph
    graph = nx.DiGraph()
    
    # Add nodes
    # Physical nodes
    for i in range(1, 5):  # 4 nodes
        graph.add_node(f"node{i}", type="Node", name=f"Node {i}")
    
    # Brokers
    for i in range(1, 3):  # 2 brokers
        graph.add_node(f"broker{i}", type="Broker", name=f"Broker {i}")
        graph.add_edge(f"broker{i}", f"node{i}", type="RUNS_ON")
    
    # Applications
    for i in range(1, 11):  # 10 applications
        graph.add_node(f"app{i}", type="Application", name=f"App {i}")
        # Assign to a random node
        node_id = f"node{np.random.randint(1, 5)}"
        graph.add_edge(f"app{i}", node_id, type="RUNS_ON")
    
    # Topics
    for i in range(1, 26):  # 25 topics
        graph.add_node(f"topic{i}", type="Topic", name=f"Topic {i}")
        # Assign to a broker
        broker_id = f"broker{np.random.randint(1, 3)}"
        graph.add_edge(broker_id, f"topic{i}", type="ROUTES")
    
    # Create publish relationships
    for i in range(1, 11):  # For each app
        # Publish to 1-5 random topics
        num_topics = np.random.randint(1, 6)
        topic_ids = np.random.choice(range(1, 26), num_topics, replace=False)
        for topic_num in topic_ids:
            graph.add_edge(f"app{i}", f"topic{topic_num}", type="PUBLISHES_TO")
    
    # Create subscribe relationships
    for i in range(1, 11):  # For each app
        # Subscribe to 1-8 random topics
        num_topics = np.random.randint(1, 9)
        topic_ids = np.random.choice(range(1, 26), num_topics, replace=False)
        for topic_num in topic_ids:
            graph.add_edge(f"app{i}", f"topic{topic_num}", type="SUBSCRIBES_TO")
    
    # Create node connections (CONNECTS_TO)
    for i in range(1, 5):
        for j in range(i + 1, 5):
            graph.add_edge(f"node{i}", f"node{j}", type="CONNECTS_TO")
    
    # Create DEPENDS_ON relationships
    # Application-level dependencies
    for app1 in range(1, 11):
        for app2 in range(1, 11):
            if app1 != app2:
                for topic in range(1, 26):
                    # If app1 publishes to topic and app2 subscribes to it
                    if graph.has_edge(f"app{app1}", f"topic{topic}") and graph.has_edge(f"app{app2}", f"topic{topic}"):
                        if graph.get_edge_data(f"app{app1}", f"topic{topic}")["type"] == "PUBLISHES_TO" and \
                           graph.get_edge_data(f"app{app2}", f"topic{topic}")["type"] == "SUBSCRIBES_TO":
                            graph.add_edge(f"app{app2}", f"app{app1}", type="DEPENDS_ON")
    
    # Application-broker dependencies
    for app in range(1, 11):
        for topic in range(1, 26):
            if graph.has_edge(f"app{app}", f"topic{topic}"):
                for broker in range(1, 3):
                    if graph.has_edge(f"broker{broker}", f"topic{topic}"):
                        graph.add_edge(f"app{app}", f"broker{broker}", type="DEPENDS_ON")

    return graph

if __name__ == "__main__":
    # Generate example graph
    graph = generate_example_graph()

    # Initialize simulator
    simulator = FailureSimulator(graph)
    
    # Simulate failures for all components
    impact_results = simulator.simulate_all_failures()
    
    # Print top 5 critical components
    print("Top 5 Critical Components by Impact:")
    top_components = impact_results.head(5)
    for i, (_, component) in enumerate(top_components.iterrows(), 1):
        print(f"{i}. {component['failed_component']} ({component['component_type']}): " +
              f"{component['overall_impact']:.2f}% overall impact")
        print(f"   - Message Delivery Degradation: {component['message_delivery_ratio_degradation']:.2f}%")
        print(f"   - Topic Availability Degradation: {component['topic_availability_ratio_degradation']:.2f}%")
    
    # Visualize results
    simulator.visualize_impact_results(impact_results)
    
    # Identify critical components using different thresholding methods
    critical_percentile = simulator.identify_critical_components(
        impact_results, threshold_method='percentile', threshold_value=80)
    
    critical_absolute = simulator.identify_critical_components(
        impact_results, threshold_method='absolute', threshold_value=30)
    
    critical_elbow = simulator.identify_critical_components(
        impact_results, threshold_method='elbow')
    
    print(f"\nCritical components identified (percentile method): {len(critical_percentile)}")
    print(f"Critical components identified (absolute threshold method): {len(critical_absolute)}")
    print(f"Critical components identified (elbow method): {len(critical_elbow)}")
    
    # Generate resilience recommendations
    recommendations = simulator.generate_resilience_recommendations(critical_percentile)
    
    print("\nKey Recommendations for System Resilience:")
    
    if recommendations['broker_redundancy']:
        print("\nBroker Redundancy:")
        for i, rec in enumerate(recommendations['broker_redundancy'][:3], 1):
            print(f"{i}. {rec['recommendation']}")
    
    if recommendations['node_distribution']:
        print("\nNode Distribution:")
        for i, rec in enumerate(recommendations['node_distribution'][:3], 1):
            print(f"{i}. {rec['recommendation']}")
    
    if recommendations['topic_replication']:
        print("\nTopic Replication:")
        for i, rec in enumerate(recommendations['topic_replication'][:3], 1):
            print(f"{i}. {rec['recommendation']}")
    
    if recommendations['network_connectivity']:
        print("\nNetwork Connectivity:")
        for i, rec in enumerate(recommendations['network_connectivity'], 1):
            print(f"{i}. {rec['recommendation']}")