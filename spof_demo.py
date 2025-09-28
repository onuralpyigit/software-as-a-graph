import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def create_ecommerce_pubsub_graph():
    """
    Create a realistic e-commerce pub-sub system graph
    """
    G = nx.DiGraph()
    
    # Add Applications/Services (Publishers and Subscribers)
    services = [
        # Core services
        {'id': 'order-service', 'type': 'application', 'criticality': 'high'},
        {'id': 'payment-service', 'type': 'application', 'criticality': 'critical'},
        {'id': 'inventory-service', 'type': 'application', 'criticality': 'high'},
        {'id': 'shipping-service', 'type': 'application', 'criticality': 'medium'},
        {'id': 'notification-service', 'type': 'application', 'criticality': 'low'},
        
        # Analytics services
        {'id': 'analytics-service', 'type': 'application', 'criticality': 'low'},
        {'id': 'fraud-detection', 'type': 'application', 'criticality': 'high'},
        
        # Customer-facing services
        {'id': 'recommendation-engine', 'type': 'application', 'criticality': 'low'},
        {'id': 'customer-service', 'type': 'application', 'criticality': 'medium'},
        
        # External integrations
        {'id': 'warehouse-integration', 'type': 'application', 'criticality': 'high'},
        {'id': 'accounting-integration', 'type': 'application', 'criticality': 'medium'}
    ]
    
    for service in services:
        G.add_node(service['id'], **service)
    
    # Add Topics
    topics = [
        {'id': 'orders.created', 'type': 'topic', 'durability': 'persistent', 'partitions': 3},
        {'id': 'orders.validated', 'type': 'topic', 'durability': 'persistent', 'partitions': 3},
        {'id': 'payments.processed', 'type': 'topic', 'durability': 'persistent', 'partitions': 1},
        {'id': 'payments.failed', 'type': 'topic', 'durability': 'persistent', 'partitions': 1},
        {'id': 'inventory.reserved', 'type': 'topic', 'durability': 'persistent', 'partitions': 5},
        {'id': 'inventory.updated', 'type': 'topic', 'durability': 'volatile', 'partitions': 5},
        {'id': 'shipments.created', 'type': 'topic', 'durability': 'persistent', 'partitions': 2},
        {'id': 'notifications.email', 'type': 'topic', 'durability': 'volatile', 'partitions': 1},
        {'id': 'analytics.events', 'type': 'topic', 'durability': 'volatile', 'partitions': 10},
        {'id': 'fraud.alerts', 'type': 'topic', 'durability': 'persistent', 'partitions': 1}
    ]
    
    for topic in topics:
        G.add_node(topic['id'], **topic)
    
    # Add Brokers
    brokers = [
        {'id': 'broker-1', 'type': 'broker', 'region': 'us-east'},
        {'id': 'broker-2', 'type': 'broker', 'region': 'us-west'},
        {'id': 'broker-3', 'type': 'broker', 'region': 'eu-central'}
    ]
    
    for broker in brokers:
        G.add_node(broker['id'], **broker)
    
    # Define Publishing Relationships (Application -> Topic)
    publish_edges = [
        # Order service publishes
        ('order-service', 'orders.created', {'msg_rate_per_sec': 100, 'avg_msg_size_kb': 2}),
        ('order-service', 'orders.validated', {'msg_rate_per_sec': 95, 'avg_msg_size_kb': 2}),
        
        # Payment service publishes - CRITICAL: SINGLE PUBLISHER!
        ('payment-service', 'payments.processed', {'msg_rate_per_sec': 90, 'avg_msg_size_kb': 1}),
        ('payment-service', 'payments.failed', {'msg_rate_per_sec': 5, 'avg_msg_size_kb': 1}),
        
        # Inventory service publishes
        ('inventory-service', 'inventory.reserved', {'msg_rate_per_sec': 95, 'avg_msg_size_kb': 1}),
        ('inventory-service', 'inventory.updated', {'msg_rate_per_sec': 200, 'avg_msg_size_kb': 0.5}),
        ('warehouse-integration', 'inventory.updated', {'msg_rate_per_sec': 50, 'avg_msg_size_kb': 0.5}),
        
        # Shipping service publishes
        ('shipping-service', 'shipments.created', {'msg_rate_per_sec': 85, 'avg_msg_size_kb': 3}),
        
        # Notification service publishes
        ('notification-service', 'notifications.email', {'msg_rate_per_sec': 150, 'avg_msg_size_kb': 5}),
        
        # Fraud detection publishes
        ('fraud-detection', 'fraud.alerts', {'msg_rate_per_sec': 2, 'avg_msg_size_kb': 10}),
        
        # Multiple services publish to analytics
        ('order-service', 'analytics.events', {'msg_rate_per_sec': 100, 'avg_msg_size_kb': 1}),
        ('payment-service', 'analytics.events', {'msg_rate_per_sec': 95, 'avg_msg_size_kb': 1}),
        ('shipping-service', 'analytics.events', {'msg_rate_per_sec': 85, 'avg_msg_size_kb': 1})
    ]
    
    for source, target, attrs in publish_edges:
        G.add_edge(source, target, relationship='PUBLISHES_TO', **attrs)
    
    # Define Subscription Relationships (Topic -> Application)
    subscribe_edges = [
        # Orders created subscriptions
        ('orders.created', 'payment-service', {'processing_time_ms': 50}),
        ('orders.created', 'inventory-service', {'processing_time_ms': 30}),
        ('orders.created', 'fraud-detection', {'processing_time_ms': 100}),
        ('orders.created', 'analytics-service', {'processing_time_ms': 10}),
        
        # Orders validated subscriptions
        ('orders.validated', 'shipping-service', {'processing_time_ms': 40}),
        ('orders.validated', 'notification-service', {'processing_time_ms': 20}),
        
        # Payment processed subscriptions - MANY DEPENDENTS!
        ('payments.processed', 'order-service', {'processing_time_ms': 20}),
        ('payments.processed', 'inventory-service', {'processing_time_ms': 30}),
        ('payments.processed', 'shipping-service', {'processing_time_ms': 40}),
        ('payments.processed', 'notification-service', {'processing_time_ms': 20}),
        ('payments.processed', 'accounting-integration', {'processing_time_ms': 60}),
        ('payments.processed', 'analytics-service', {'processing_time_ms': 10}),
        
        # Payment failed subscriptions
        ('payments.failed', 'order-service', {'processing_time_ms': 30}),
        ('payments.failed', 'inventory-service', {'processing_time_ms': 20}),
        ('payments.failed', 'notification-service', {'processing_time_ms': 20}),
        
        # Other subscriptions
        ('inventory.reserved', 'shipping-service', {'processing_time_ms': 30}),
        ('inventory.updated', 'recommendation-engine', {'processing_time_ms': 200}),
        ('shipments.created', 'notification-service', {'processing_time_ms': 20}),
        ('shipments.created', 'customer-service', {'processing_time_ms': 10}),
        ('fraud.alerts', 'order-service', {'processing_time_ms': 10}),
        ('fraud.alerts', 'customer-service', {'processing_time_ms': 10})
    ]
    
    for source, target, attrs in subscribe_edges:
        G.add_edge(source, target, relationship='SUBSCRIBES_TO', **attrs)
    
    # Add Broker-Topic relationships
    broker_topics = [
        ('broker-1', 'orders.created'),
        ('broker-1', 'orders.validated'),
        ('broker-1', 'payments.processed'),  # Single broker for payments!
        ('broker-1', 'payments.failed'),
        ('broker-2', 'inventory.reserved'),
        ('broker-2', 'inventory.updated'),
        ('broker-2', 'shipments.created'),
        ('broker-3', 'notifications.email'),
        ('broker-3', 'analytics.events'),
        ('broker-3', 'fraud.alerts')
    ]
    
    for broker, topic in broker_topics:
        G.add_edge(broker, topic, relationship='HOSTS')
    
    return G

def apply_single_point_of_failure_rule(graph):
    """
    Apply the Single Point of Failure rule to identify critical components
    """
    results = []
    
    # Analyze each node in the graph
    for node in graph.nodes():
        node_attrs = graph.nodes[node]
        node_type = node_attrs.get('type')
        
        # Check different SPOF scenarios based on node type
        if node_type == 'application':
            # Check if this application is the sole publisher to any topic
            published_topics = [
                target for target in graph.successors(node)
                if graph.nodes[target].get('type') == 'topic'
            ]
            
            for topic in published_topics:
                # Get all publishers for this topic
                all_publishers = [
                    source for source in graph.predecessors(topic)
                    if graph.nodes[source].get('type') == 'application'
                ]
                
                if len(all_publishers) == 1:  # This is the only publisher!
                    # Find all subscribers dependent on this topic
                    subscribers = [
                        target for target in graph.successors(topic)
                        if graph.nodes[target].get('type') == 'application'
                    ]
                    
                    # Calculate the impact
                    total_msg_rate = graph.edges[(node, topic)].get('msg_rate_per_sec', 0)
                    
                    # Calculate criticality score
                    criticality_score = calculate_spof_criticality(
                        subscriber_count=len(subscribers),
                        msg_rate=total_msg_rate,
                        topic_durability=graph.nodes[topic].get('durability'),
                        publisher_criticality=node_attrs.get('criticality')
                    )
                    
                    result = {
                        'component': node,
                        'component_type': node_type,
                        'spof_type': 'sole_publisher',
                        'affected_topic': topic,
                        'dependent_subscribers': subscribers,
                        'subscriber_count': len(subscribers),
                        'message_rate': total_msg_rate,
                        'criticality_score': criticality_score,
                        'impact_analysis': analyze_failure_impact(graph, node, topic, subscribers)
                    }
                    results.append(result)
        
        elif node_type == 'broker':
            # Check if this broker is the only one hosting certain topics
            hosted_topics = [
                target for target in graph.successors(node)
                if graph.nodes[target].get('type') == 'topic'
            ]
            
            single_hosted_topics = []
            for topic in hosted_topics:
                all_brokers = [
                    source for source in graph.predecessors(topic)
                    if graph.nodes[source].get('type') == 'broker'
                ]
                
                if len(all_brokers) == 1:  # Single broker hosting this topic
                    single_hosted_topics.append(topic)
            
            if single_hosted_topics:
                impact = calculate_broker_spof_impact(graph, node, single_hosted_topics)
                
                result = {
                    'component': node,
                    'component_type': node_type,
                    'spof_type': 'sole_broker',
                    'affected_topics': single_hosted_topics,
                    'criticality_score': impact['criticality_score'],
                    'total_affected_flows': impact['affected_flows'],
                    'impact_analysis': impact
                }
                results.append(result)
    
    return sorted(results, key=lambda x: x['criticality_score'], reverse=True)

def calculate_broker_spof_impact(graph, broker, topics):
    """
    Calculate impact of broker SPOF
    """
    affected_flows = 0
    total_msg_rate = 0
    for topic in topics:
        subscribers = [
            target for target in graph.successors(topic)
            if graph.nodes[target].get('type') == 'application'
        ]
        affected_flows += len(subscribers)
        
        # Sum message rates from all publishers to this topic
        publishers = [
            source for source in graph.predecessors(topic)
            if graph.nodes[source].get('type') == 'application'
        ]
        for pub in publishers:
            total_msg_rate += graph.edges[(pub, topic)].get('msg_rate_per_sec', 0)
    
    # Simple criticality score based on number of affected flows and message rate
    criticality_score = min(1.0, (affected_flows / 20) + (total_msg_rate / 1000))
    
    return {
        'criticality_score': criticality_score,
        'affected_flows': affected_flows,
        'total_msg_rate': total_msg_rate,
        'immediate_effects': [
            f"Topics {topics} become unavailable",
            f"{affected_flows} subscriber flows interrupted",
            f"Total message rate of {total_msg_rate} msg/sec interrupted"
        ],
        'cascading_effects': [],
        'business_impact': [],
        'recovery_time_estimate': 600  # Assume longer recovery for broker failures
    }

def calculate_spof_criticality(subscriber_count, msg_rate, topic_durability, publisher_criticality):
    """
    Calculate criticality score for a SPOF
    """
    # Base score from subscriber count (0-0.4)
    subscriber_score = min(0.4, subscriber_count * 0.05)
    
    # Message rate score (0-0.2)
    rate_score = min(0.2, msg_rate / 500)
    
    # Durability score (0-0.2)
    durability_score = 0.2 if topic_durability == 'persistent' else 0.1
    
    # Publisher criticality score (0-0.2)
    criticality_map = {'critical': 0.2, 'high': 0.15, 'medium': 0.1, 'low': 0.05}
    publisher_score = criticality_map.get(publisher_criticality, 0.05)
    
    return subscriber_score + rate_score + durability_score + publisher_score

def analyze_failure_impact(graph, failed_node, topic, affected_subscribers):
    """
    Detailed impact analysis of SPOF failure
    """
    impact = {
        'immediate_effects': [],
        'cascading_effects': [],
        'business_impact': [],
        'recovery_time_estimate': 0
    }
    
    # Immediate effects
    impact['immediate_effects'] = [
        f"Topic '{topic}' stops receiving new messages",
        f"{len(affected_subscribers)} subscribers stop receiving updates",
        f"Message flow rate of {graph.edges[(failed_node, topic)].get('msg_rate_per_sec', 0)} msg/sec interrupted"
    ]
    
    # Analyze cascading effects
    for subscriber in affected_subscribers:
        subscriber_attrs = graph.nodes[subscriber]
        
        # Check what this subscriber publishes
        subscriber_publishes = [
            target for target in graph.successors(subscriber)
            if graph.nodes[target].get('type') == 'topic'
        ]
        
        if subscriber_publishes:
            impact['cascading_effects'].append(
                f"{subscriber} cannot publish to {subscriber_publishes} due to missing input"
            )
        
        # Business impact based on criticality
        if subscriber_attrs.get('criticality') in ['critical', 'high']:
            impact['business_impact'].append(
                f"Critical service '{subscriber}' disrupted"
            )
    
    # Estimate recovery time
    if graph.nodes[topic].get('durability') == 'persistent':
        impact['recovery_time_estimate'] = 300  # 5 minutes for persistent topics
    else:
        impact['recovery_time_estimate'] = 60   # 1 minute for volatile topics
    
    return impact

def demonstrate_spof_detection():
    """
    Complete demonstration of SPOF detection
    """
    # Create the graph
    print("Creating E-Commerce Pub-Sub System Graph...")
    graph = create_ecommerce_pubsub_graph()
    
    print(f"Graph Statistics:")
    print(f"  - Total Nodes: {graph.number_of_nodes()}")
    print(f"  - Total Edges: {graph.number_of_edges()}")
    print(f"  - Applications: {len([n for n in graph.nodes() if graph.nodes[n]['type'] == 'application'])}")
    print(f"  - Topics: {len([n for n in graph.nodes() if graph.nodes[n]['type'] == 'topic'])}")
    print(f"  - Brokers: {len([n for n in graph.nodes() if graph.nodes[n]['type'] == 'broker'])}")
    
    # Apply SPOF rule
    print("\n" + "="*80)
    print("APPLYING SINGLE POINT OF FAILURE DETECTION RULE")
    print("="*80)
    
    spof_results = apply_single_point_of_failure_rule(graph)
    
    # Display results
    for i, result in enumerate(spof_results, 1):
        print(f"\n{i}. CRITICAL COMPONENT DETECTED: {result['component']}")
        print("-" * 60)
        print(f"   Component Type: {result['component_type']}")
        print(f"   SPOF Type: {result['spof_type']}")
        print(f"   Criticality Score: {result['criticality_score']:.3f}")
        
        if result['spof_type'] == 'sole_publisher':
            print(f"   Affected Topic: {result['affected_topic']}")
            print(f"   Dependent Subscribers ({result['subscriber_count']}): {', '.join(result['dependent_subscribers'][:5])}")
            if result['subscriber_count'] > 5:
                print(f"      ... and {result['subscriber_count'] - 5} more")
            print(f"   Message Rate: {result['message_rate']} msg/sec")
            
            print("\n   IMPACT ANALYSIS:")
            impact = result['impact_analysis']
            
            print("   Immediate Effects:")
            for effect in impact['immediate_effects']:
                print(f"      • {effect}")
            
            if impact['cascading_effects']:
                print("   Cascading Effects:")
                for effect in impact['cascading_effects']:
                    print(f"      • {effect}")
            
            if impact['business_impact']:
                print("   Business Impact:")
                for effect in impact['business_impact']:
                    print(f"      • {effect}")
            
            print(f"   Estimated Recovery Time: {impact['recovery_time_estimate']} seconds")
    
    # Generate recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR ADDRESSING SPOF RISKS")
    print("="*80)
    
    for result in spof_results[:3]:  # Top 3 critical SPOFs
        recommendations = generate_spof_recommendations(graph, result)
        
        print(f"\nFor {result['component']}:")
        for rec in recommendations:
            print(f"  {rec['priority']}. {rec['recommendation']}")
            print(f"     Implementation: {rec['implementation']}")
            print(f"     Effort: {rec['effort']} | Risk Reduction: {rec['risk_reduction']}")

def generate_spof_recommendations(graph, spof_result):
    """
    Generate specific recommendations for addressing SPOF
    """
    recommendations = []
    
    if spof_result['spof_type'] == 'sole_publisher':
        component = spof_result['component']
        topic = spof_result['affected_topic']
        
        # Recommendation 1: Add redundant publisher
        recommendations.append({
            'priority': 1,
            'recommendation': f"Deploy redundant instance of {component}",
            'implementation': f"Create {component}-replica with active-passive or active-active configuration",
            'effort': 'Medium',
            'risk_reduction': 'High'
        })
        
        # Recommendation 2: Implement circuit breaker
        recommendations.append({
            'priority': 2,
            'recommendation': f"Implement circuit breaker pattern for {topic} subscribers",
            'implementation': "Add fallback mechanisms in subscribers to handle missing messages",
            'effort': 'Low',
            'risk_reduction': 'Medium'
        })
        
        # Recommendation 3: Add monitoring
        recommendations.append({
            'priority': 3,
            'recommendation': f"Enhanced monitoring for {component}",
            'implementation': "Set up health checks, alerts for service degradation, and automated failover",
            'effort': 'Low',
            'risk_reduction': 'Medium'
        })
    
    return recommendations

def visualize_spof_analysis(graph, spof_results):
    """
    Visualize the graph with SPOF components highlighted
    """
    plt.figure(figsize=(16, 10))
    
    # Prepare node colors based on criticality
    node_colors = []
    node_sizes = []
    
    # Get SPOF components for highlighting
    spof_components = {r['component']: r['criticality_score'] 
                       for r in spof_results}
    
    for node in graph.nodes():
        if node in spof_components:
            # Red for critical SPOF
            if spof_components[node] > 0.7:
                node_colors.append('#FF0000')
                node_sizes.append(1000)
            # Orange for moderate SPOF
            elif spof_components[node] > 0.5:
                node_colors.append('#FFA500')
                node_sizes.append(800)
            else:
                node_colors.append('#FFFF00')
                node_sizes.append(600)
        else:
            # Color by node type
            node_type = graph.nodes[node].get('type')
            if node_type == 'application':
                node_colors.append('#90EE90')
                node_sizes.append(500)
            elif node_type == 'topic':
                node_colors.append('#87CEEB')
                node_sizes.append(400)
            else:  # broker
                node_colors.append('#DDA0DD')
                node_sizes.append(600)
    
    # Layout
    pos = nx.spring_layout(graph, k=2, iterations=50)
    
    # Draw the graph
    nx.draw(graph, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            font_size=8,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            alpha=0.7)
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='#FF0000', s=100, label='Critical SPOF (>0.7)'),
        plt.scatter([], [], c='#FFA500', s=100, label='Moderate SPOF (0.5-0.7)'),
        plt.scatter([], [], c='#FFFF00', s=100, label='Low SPOF (<0.5)'),
        plt.scatter([], [], c='#90EE90', s=100, label='Application'),
        plt.scatter([], [], c='#87CEEB', s=100, label='Topic'),
        plt.scatter([], [], c='#DDA0DD', s=100, label='Broker')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title("Single Point of Failure Analysis - E-Commerce Pub-Sub System", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Execute the complete demonstration
if __name__ == "__main__":
    demonstrate_spof_detection()
    
    # Create visualization
    graph = create_ecommerce_pubsub_graph()
    spof_results = apply_single_point_of_failure_rule(graph)
    visualize_spof_analysis(graph, spof_results)