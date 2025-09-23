import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
from tqdm import tqdm

class ReachabilityAnalyzer:
    def __init__(self, graph):
        """
        Initialize the reachability analyzer for pub-sub systems
        
        Args:
            graph: NetworkX DiGraph representing the pub-sub system
        """
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.node_types = nx.get_node_attributes(self.graph, 'type')
        self.node_names = nx.get_node_attributes(self.graph, 'name')
        
        # Extract components by type
        self.applications = [n for n, t in self.node_types.items() if t == 'Application']
        self.topics = [n for n, t in self.node_types.items() if t == 'Topic']
        self.brokers = [n for n, t in self.node_types.items() if t == 'Broker']
        self.nodes = [n for n, t in self.node_types.items() if t == 'Node']
        
        # Store the reachability matrix for the original system
        self.original_reachability = self.compute_reachability_matrix()

        self.output_file_dir = "output/"
    
    def reset_graph(self):
        """Reset the graph to its original state"""
        self.graph = self.original_graph.copy()
    
    def compute_reachability_matrix(self):
        """
        Compute the reachability matrix for applications in the pub-sub system
        
        Returns:
            pandas.DataFrame: Reachability matrix where cell[i,j] = 1 if app i can reach app j, 0 otherwise
        """
        applications = [n for n, t in self.node_types.items() if t == 'Application' and n in self.graph]
        reachability = pd.DataFrame(0, index=applications, columns=applications)
        
        # For each pair of applications
        for source_app in applications:
            for target_app in applications:
                if source_app != target_app:
                    # Check if source can reach target through the pub-sub system
                    if self.can_app_reach_app(source_app, target_app):
                        reachability.loc[source_app, target_app] = 1
        
        return reachability
    
    def can_app_reach_app(self, source_app, target_app):
        """
        Check if a source application can reach a target application through the pub-sub system
        
        Args:
            source_app: Source application node ID
            target_app: Target application node ID
            
        Returns:
            bool: True if source can reach target, False otherwise
        """
        # Get topics that source publishes to
        pub_topics = []
        for _, topic, rel in self.graph.out_edges(source_app, data=True):
            if rel.get('type') == 'PUBLISHES_TO':
                pub_topics.append(topic)
        
        # Get topics that target subscribes to
        sub_topics = []
        for _, topic, rel in self.graph.out_edges(target_app, data=True):
            if rel.get('type') == 'SUBSCRIBES_TO':
                sub_topics.append(topic)
        
        # Find common topics
        common_topics = set(pub_topics).intersection(set(sub_topics))
        
        # Check if there's a broker routing any of the common topics
        for topic in common_topics:
            for broker in self.brokers:
                if broker in self.graph.nodes():
                    for _, t, rel in self.graph.out_edges(broker, data=True):
                        if rel.get('type') == 'ROUTES' and t == topic:
                            # Verify the broker is running on an active node
                            for _, node, rel in self.graph.out_edges(broker, data=True):
                                if rel.get('type') == 'RUNS_ON' and node in self.graph.nodes():
                                    return True
        
        return False
    
    def simulate_component_failure(self, component_id, failure_type='complete'):
        """
        Simulate the failure of a component and compute the new reachability
        
        Args:
            component_id: ID of the component to fail
            failure_type: Type of failure ('complete' or 'partial')
            
        Returns:
            pandas.DataFrame: New reachability matrix after failure
        """
        # Reset graph to original state
        self.reset_graph()
        
        # Get component type
        component_type = self.node_types.get(component_id)
        
        if failure_type == 'complete':
            self.simulate_complete_failure(component_id, component_type)
        else:  # partial failure
            self.simulate_partial_failure(component_id, component_type)
        
        # Compute the new reachability matrix
        new_reachability = self.compute_reachability_matrix()
        
        return new_reachability
    
    def simulate_complete_failure(self, component_id, component_type):
        """
        Simulate complete failure of a component
        
        Args:
            component_id: ID of the component to fail
            component_type: Type of the component
        """
        if component_type == 'Node':
            # If it's a physical node, also remove all applications and brokers running on it
            components_to_remove = [component_id]
            for component, node, rel in self.graph.edges(data=True):
                if rel.get('type') == 'RUNS_ON' and node == component_id:
                    components_to_remove.append(component)
            
            for component in components_to_remove:
                if self.graph.has_node(component):
                    self.graph.remove_node(component)
        else:
            # For other component types, just remove the node
            self.graph.remove_node(component_id)
    
    def simulate_partial_failure(self, component_id, component_type):
        """
        Simulate partial failure of a component (degraded performance)
        
        Args:
            component_id: ID of the component to partially fail
            component_type: Type of the component
        """
        if component_type == 'Broker':
            # For broker, randomly remove some of its topic routes (50%)
            routes_to_remove = []
            for _, topic, rel in self.graph.out_edges(component_id, data=True):
                if rel.get('type') == 'ROUTES' and random.random() < 0.5:
                    routes_to_remove.append((component_id, topic))
            
            for source, target in routes_to_remove:
                self.graph.remove_edge(source, target)
                
        elif component_type == 'Node':
            # For node, randomly remove some connections to other nodes (50%)
            connections_to_remove = []
            for source, target, rel in self.graph.edges(data=True):
                if source == component_id and rel.get('type') == 'CONNECTS_TO' and random.random() < 0.5:
                    connections_to_remove.append((source, target))
            
            for source, target in connections_to_remove:
                self.graph.remove_edge(source, target)
                
        elif component_type == 'Application':
            # For application, randomly remove some publish/subscribe relationships (50%)
            relationships_to_remove = []
            for _, topic, rel in self.graph.out_edges(component_id, data=True):
                if (rel.get('type') == 'PUBLISHES_TO' or rel.get('type') == 'SUBSCRIBES_TO') and random.random() < 0.5:
                    relationships_to_remove.append((component_id, topic))
            
            for source, target in relationships_to_remove:
                self.graph.remove_edge(source, target)
    
    def calculate_reachability_impact(self, original_reachability, new_reachability):
        """
        Calculate the impact on reachability after a component failure
        
        Args:
            original_reachability: Reachability matrix before failure
            new_reachability: Reachability matrix after failure
            
        Returns:
            dict: Impact metrics
        """
        # Get only common indices (some nodes might have been removed)
        common_apps = list(set(original_reachability.index).intersection(set(new_reachability.index)))
        
        if not common_apps:
            return {
                'reachability_percentage': 0,
                'reachability_loss': 100,
                'isolated_apps': len(original_reachability),
                'total_paths': 0,
                'remaining_paths': 0
            }
        
        original_subset = original_reachability.loc[common_apps, common_apps]
        new_subset = new_reachability.loc[common_apps, common_apps]
        
        # Calculate metrics
        total_possible_paths = len(common_apps) * (len(common_apps) - 1)
        original_paths = original_subset.sum().sum()
        remaining_paths = new_subset.sum().sum()
        
        if original_paths == 0:
            reachability_percentage = 100  # No paths to begin with
            reachability_loss = 0
        else:
            reachability_percentage = (remaining_paths / original_paths) * 100
            reachability_loss = 100 - reachability_percentage
        
        # Count isolated applications
        connected_before = set()
        connected_after = set()
        
        for app in common_apps:
            # An app is connected if it can reach any other app or be reached by any other app
            if original_subset.loc[app].sum() > 0 or original_subset[app].sum() > 0:
                connected_before.add(app)
                
            if new_subset.loc[app].sum() > 0 or new_subset[app].sum() > 0:
                connected_after.add(app)
        
        isolated_apps = len(connected_before - connected_after)
        
        return {
            'reachability_percentage': reachability_percentage,
            'reachability_loss': reachability_loss,
            'isolated_apps': isolated_apps,
            'total_paths': original_paths,
            'remaining_paths': remaining_paths
        }
    
    def run_comprehensive_analysis(self, failure_types=['complete']):
        """
        Run a comprehensive analysis of the impact of failures on all components
        
        Args:
            failure_types: List of failure types to simulate ('complete', 'partial')
            
        Returns:
            pandas.DataFrame: Results of the analysis
        """
        results = []
        
        # Test all components except topics
        components_to_test = self.applications + self.brokers + self.nodes
        
        for component_id in tqdm(components_to_test, desc="Analyzing components"):
            component_type = self.node_types.get(component_id)
            component_name = self.node_names.get(component_id, str(component_id))
            
            for failure_type in failure_types:
                # Simulate failure and get new reachability
                new_reachability = self.simulate_component_failure(component_id, failure_type)
                
                # Calculate impact
                impact = self.calculate_reachability_impact(self.original_reachability, new_reachability)
                
                # Add component info
                impact['component_id'] = component_id
                impact['component_name'] = component_name
                impact['component_type'] = component_type
                impact['failure_type'] = failure_type
                
                results.append(impact)
        
        # Convert to DataFrame and sort by impact
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('reachability_loss', ascending=False)
        
        return results_df
    
    def visualize_reachability_matrix(self, reachability_matrix, title):
        """
        Visualize a reachability matrix
        
        Args:
            reachability_matrix: Reachability matrix to visualize
            title: Title for the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(reachability_matrix, cmap='Blues', cbar=False, 
                   xticklabels=True, yticklabels=True)
        plt.title(title, fontsize=14)
        plt.xlabel('Target Application', fontsize=12)
        plt.ylabel('Source Application', fontsize=12)
        plt.tight_layout()
        output_file = self.output_file_dir + title + ".png"
        plt.savefig(output_file)
    
    def visualize_impact_results(self, results_df, top_n=10):
        """
        Visualize the results of the impact analysis
        
        Args:
            results_df: DataFrame with impact results
            top_n: Number of top components to display
        """
        # Select top components
        top_components = results_df.head(top_n)
        
        # Create readable labels for components
        labels = [f"{row['component_name']}\n({row['component_type']})" 
                 for _, row in top_components.iterrows()]
        
        # Create bar chart for reachability loss
        plt.figure(figsize=(12, 8))
        bars = plt.barh(labels, top_components['reachability_loss'], color='firebrick')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center')
        
        plt.xlabel('Reachability Loss (%)', fontsize=12)
        plt.title('Critical Components by Reachability Impact', fontsize=14)
        plt.xlim(0, 100)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_file = self.output_file_dir + "Critical Components by Reachability Impact.png"
        plt.savefig(output_file)
        
        # Create scatter plot showing isolated apps vs reachability loss
        plt.figure(figsize=(12, 8))
        
        # Group by component type for coloring
        component_types = top_components['component_type'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(component_types)))
        
        for i, comp_type in enumerate(component_types):
            subset = top_components[top_components['component_type'] == comp_type]
            plt.scatter(subset['reachability_loss'], subset['isolated_apps'], 
                       label=comp_type, color=colors[i], s=100, alpha=0.7)
        
        # Add labels
        for _, row in top_components.iterrows():
            plt.annotate(row['component_name'], 
                        (row['reachability_loss'], row['isolated_apps']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Reachability Loss (%)', fontsize=12)
        plt.ylabel('Number of Isolated Applications', fontsize=12)
        plt.title('Reachability Impact: Path Loss vs Application Isolation', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_file = self.output_file_dir + "Reachability Impact - Path Loss vs Application Isolation.png"
        plt.savefig(output_file)
    
    def analyze_partial_vs_complete_failure(self, component_id):
        """
        Compare the impact of partial vs complete failure for a specific component
        
        Args:
            component_id: ID of the component to analyze
            
        Returns:
            dict: Comparison of impacts
        """
        # Complete failure
        complete_reachability = self.simulate_component_failure(component_id, 'complete')
        complete_impact = self.calculate_reachability_impact(
            self.original_reachability, complete_reachability)
        
        # Partial failure
        partial_reachability = self.simulate_component_failure(component_id, 'partial')
        partial_impact = self.calculate_reachability_impact(
            self.original_reachability, partial_reachability)
        
        # Compare reachability matrices
        self.visualize_reachability_matrix(self.original_reachability, 
                                          'Original Reachability Matrix')
        self.visualize_reachability_matrix(partial_reachability, 
                                          f'Reachability After Partial Failure of {component_id}')
        self.visualize_reachability_matrix(complete_reachability, 
                                          f'Reachability After Complete Failure of {component_id}')
        
        return {
            'component_id': component_id,
            'component_type': self.node_types.get(component_id),
            'complete_failure': complete_impact,
            'partial_failure': partial_impact,
            'difference': {
                'reachability_loss': complete_impact['reachability_loss'] - partial_impact['reachability_loss'],
                'isolated_apps': complete_impact['isolated_apps'] - partial_impact['isolated_apps']
            }
        }
    
    def identify_critical_paths(self):
        """
        Identify critical communication paths in the system
        
        Returns:
            list: Critical paths (app-topic-broker-topic-app)
        """
        critical_paths = []
        
        # Analyze the impact of each topic's failure
        for topic in self.topics:
            self.reset_graph()
            
            # Get publishers and subscribers for this topic
            publishers = []
            subscribers = []
            
            for app in self.applications:
                for _, t, rel in self.original_graph.out_edges(app, data=True):
                    if t == topic:
                        if rel.get('type') == 'PUBLISHES_TO':
                            publishers.append(app)
                        elif rel.get('type') == 'SUBSCRIBES_TO':
                            subscribers.append(app)
            
            # Get brokers routing this topic
            routing_brokers = []
            for broker in self.brokers:
                for _, t, rel in self.original_graph.out_edges(broker, data=True):
                    if t == topic and rel.get('type') == 'ROUTES':
                        routing_brokers.append(broker)
            
            # Calculate potential paths through this topic
            num_paths = len(publishers) * len(subscribers) * len(routing_brokers)
            
            if num_paths > 0:
                # Simulate failure of this topic
                self.graph.remove_node(topic)
                new_reachability = self.compute_reachability_matrix()
                impact = self.calculate_reachability_impact(self.original_reachability, new_reachability)
                
                # If impact is significant, consider it a critical topic
                if impact['reachability_loss'] > 5:  # More than 5% loss
                    critical_paths.append({
                        'topic': topic,
                        'publishers': publishers,
                        'subscribers': subscribers,
                        'brokers': routing_brokers,
                        'potential_paths': num_paths,
                        'reachability_impact': impact['reachability_loss']
                    })
        
        # Sort by impact
        critical_paths.sort(key=lambda x: x['reachability_impact'], reverse=True)
        return critical_paths

    def generate_resilience_recommendations(self, results_df):
        """
        Generate targeted recommendations to improve system resilience
        based on reachability analysis
        
        Args:
            results_df: DataFrame with impact analysis results
            
        Returns:
            dict: Recommendations for improving system resilience
        """
        top_critical = results_df.head(5)  # Top 5 critical components
        
        recommendations = {
            'broker_redundancy': [],
            'application_deployment': [],
            'node_connectivity': [],
            'topic_distribution': []
        }
        
        # Analyze each critical component
        for _, component in top_critical.iterrows():
            comp_id = component['component_id']
            comp_type = component['component_type']
            
            if comp_type == 'Broker':
                # Identify topics routed by this broker
                routed_topics = []
                for _, topic, rel in self.original_graph.out_edges(comp_id, data=True):
                    if rel.get('type') == 'ROUTES':
                        routed_topics.append(topic)
                
                # Check if there are redundant brokers for these topics
                topics_without_redundancy = []
                for topic in routed_topics:
                    routing_brokers = 0
                    for broker in self.brokers:
                        if broker != comp_id:
                            for _, t, rel in self.original_graph.out_edges(broker, data=True):
                                if t == topic and rel.get('type') == 'ROUTES':
                                    routing_brokers += 1
                    
                    if routing_brokers == 0:
                        topics_without_redundancy.append(topic)
                
                if topics_without_redundancy:
                    recommendations['broker_redundancy'].append({
                        'broker': comp_id,
                        'topics_without_redundancy': topics_without_redundancy,
                        'recommendation': f"Add redundant broker for {len(topics_without_redundancy)} topics"
                    })
            
            elif comp_type == 'Node':
                # Check which brokers and applications run on this node
                brokers_on_node = []
                apps_on_node = []
                
                for comp, node, rel in self.original_graph.edges(data=True):
                    if node == comp_id and rel.get('type') == 'RUNS_ON':
                        comp_type = self.node_types.get(comp)
                        if comp_type == 'Broker':
                            brokers_on_node.append(comp)
                        elif comp_type == 'Application':
                            apps_on_node.append(comp)
                
                # If multiple critical components run on this node, suggest redistribution
                if brokers_on_node and apps_on_node:
                    recommendations['application_deployment'].append({
                        'node': comp_id,
                        'brokers': brokers_on_node,
                        'applications': apps_on_node,
                        'recommendation': f"Redistribute {len(brokers_on_node)} brokers and {len(apps_on_node)} applications across multiple nodes"
                    })
                
                # Check node connectivity
                connections = 0
                for _, target, rel in self.original_graph.out_edges(comp_id, data=True):
                    if rel.get('type') == 'CONNECTS_TO':
                        connections += 1
                
                if connections < len(self.nodes) - 1:
                    recommendations['node_connectivity'].append({
                        'node': comp_id,
                        'current_connections': connections,
                        'max_possible': len(self.nodes) - 1,
                        'recommendation': f"Increase network connectivity for this node"
                    })
            
            elif comp_type == 'Application':
                # Identify topics this app publishes to
                published_topics = []
                for _, topic, rel in self.original_graph.out_edges(comp_id, data=True):
                    if rel.get('type') == 'PUBLISHES_TO':
                        published_topics.append(topic)
                
                # Check if these topics have alternate publishers
                critical_topics = []
                for topic in published_topics:
                    publishers = 0
                    for app in self.applications:
                        if app != comp_id:
                            for _, t, rel in self.original_graph.out_edges(app, data=True):
                                if t == topic and rel.get('type') == 'PUBLISHES_TO':
                                    publishers += 1
                    
                    if publishers == 0:
                        critical_topics.append(topic)
                
                if critical_topics:
                    recommendations['topic_distribution'].append({
                        'application': comp_id,
                        'sole_publisher_topics': critical_topics,
                        'recommendation': f"Implement redundant publishers for {len(critical_topics)} topics"
                    })
        
        return recommendations

def generate_example_graph():
    # Create a sample pub-sub system graph
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
        node_id = f"node{random.randint(1, 4)}"
        graph.add_edge(f"app{i}", node_id, type="RUNS_ON")
    
    # Topics
    for i in range(1, 26):  # 25 topics
        graph.add_node(f"topic{i}", type="Topic", name=f"Topic {i}")
        # Assign to a broker
        broker_id = f"broker{random.randint(1, 2)}"
        graph.add_edge(broker_id, f"topic{i}", type="ROUTES")
    
    # Create publish relationships
    for i in range(1, 11):  # For each app
        # Publish to 1-5 random topics
        num_topics = random.randint(1, 6)
        topic_ids = random.sample(range(1, 26), num_topics)
        for topic_num in topic_ids:
            graph.add_edge(f"app{i}", f"topic{topic_num}", type="PUBLISHES_TO")
    
    # Create subscribe relationships
    for i in range(1, 11):  # For each app
        # Subscribe to 1-8 random topics
        num_topics = random.randint(1, 9)
        topic_ids = random.sample(range(1, 26), num_topics)
        for topic_num in topic_ids:
            graph.add_edge(f"app{i}", f"topic{topic_num}", type="SUBSCRIBES_TO")
    
    # Create node connections (CONNECTS_TO)
    for i in range(1, 5):
        for j in range(i + 1, 5):
            graph.add_edge(f"node{i}", f"node{j}", type="CONNECTS_TO")
    
    return graph

if __name__ == "__main__":
    # Generate an example pub-sub system graph
    graph = generate_example_graph()
    
    # Initialize the analyzer
    analyzer = ReachabilityAnalyzer(graph)
    
    # Display the original reachability matrix
    print("Computing original reachability matrix...")
    analyzer.visualize_reachability_matrix(analyzer.original_reachability, 
                                         "Original System Reachability Matrix")
    
    # Run comprehensive analysis for both complete and partial failures
    print("Running comprehensive failure analysis...")
    results = analyzer.run_comprehensive_analysis(failure_types=['complete', 'partial'])
    
    # Display top critical components
    print("\nTop 5 Critical Components by Reachability Impact:")
    top5 = results.head(5)
    for i, (_, component) in enumerate(top5.iterrows(), 1):
        print(f"{i}. {component['component_name']} ({component['component_type']}, {component['failure_type']} failure)")
        print(f"   - Reachability Loss: {component['reachability_loss']:.2f}%")
        print(f"   - Isolated Applications: {component['isolated_apps']}")
        print(f"   - Remaining Communication Paths: {component['remaining_paths']} out of {component['total_paths']}")
    
    # Visualize the results
    print("\nVisualizing impact results...")
    analyzer.visualize_impact_results(results)
    
    # Analyze partial vs. complete failure for the most critical component
    most_critical = results.iloc[0]['component_id']
    print(f"\nAnalyzing partial vs. complete failure for most critical component: {most_critical}")
    comparison = analyzer.analyze_partial_vs_complete_failure(most_critical)
    
    print(f"Complete Failure Impact: {comparison['complete_failure']['reachability_loss']:.2f}% reachability loss")
    print(f"Partial Failure Impact: {comparison['partial_failure']['reachability_loss']:.2f}% reachability loss")
    print(f"Difference: {comparison['difference']['reachability_loss']:.2f}% additional loss in complete failure")
    
    # Identify critical communication paths
    print("\nIdentifying critical communication paths...")
    critical_paths = analyzer.identify_critical_paths()
    
    print("Top 3 Critical Communication Paths:")
    for i, path in enumerate(critical_paths[:3], 1):
        print(f"{i}. Topic: {path['topic']}")
        print(f"   - Publishers: {', '.join([str(p) for p in path['publishers']])}")
        print(f"   - Subscribers: {', '.join([str(s) for s in path['subscribers']])}")
        print(f"   - Brokers: {', '.join([str(b) for b in path['brokers']])}")
        print(f"   - Reachability Impact: {path['reachability_impact']:.2f}%")
    
    # Generate resilience recommendations
    print("\nGenerating resilience recommendations...")
    recommendations = analyzer.generate_resilience_recommendations(results)
    
    # Display recommendations
    if recommendations['broker_redundancy']:
        print("\nBroker Redundancy Recommendations:")
        for i, rec in enumerate(recommendations['broker_redundancy'], 1):
            print(f"{i}. {rec['recommendation']} for broker {rec['broker']}")
    
    if recommendations['application_deployment']:
        print("\nApplication Deployment Recommendations:")
        for i, rec in enumerate(recommendations['application_deployment'], 1):
            print(f"{i}. {rec['recommendation']} on node {rec['node']}")
    
    if recommendations['node_connectivity']:
        print("\nNode Connectivity Recommendations:")
        for i, rec in enumerate(recommendations['node_connectivity'], 1):
            print(f"{i}. {rec['recommendation']} ({rec['current_connections']}/{rec['max_possible']} connections)")
    
    if recommendations['topic_distribution']:
        print("\nTopic Distribution Recommendations:")
        for i, rec in enumerate(recommendations['topic_distribution'], 1):
            print(f"{i}. {rec['recommendation']} from application {rec['application']}")