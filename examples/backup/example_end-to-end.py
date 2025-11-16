"""
Complete Example: End-to-end Software-as-a-Graph Analysis

This example demonstrates the full capabilities of the implementation
"""

import networkx as nx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / '..'))

from src.core.graph_model import (
    GraphModel, ApplicationNode, TopicNode, BrokerNode, InfrastructureNode,
    QoSPolicy, QoSDurability, QoSReliability,
    ApplicationType, MessagePattern,
    PublishesEdge, SubscribesEdge, RoutesEdge, RunsOnEdge, ConnectsToEdge, DependsOnEdge
)
from src.analysis.criticality_scorer import CompositeCriticalityScorer
from src.orchestration.analysis_orchestrator import AnalysisOrchestrator


def create_sample_system() -> GraphModel:
    """Create a sample pub-sub system with complete properties"""
    
    model = GraphModel()
    
    # Infrastructure nodes
    node1 = InfrastructureNode(
        name="node1",
        datacenter="DC1",
        zone="us-east-1a",
        total_cpu_cores=8.0,
        total_memory_mb=16384.0,
        cpu_utilization=0.6,
        memory_utilization=0.5
    )
    node2 = InfrastructureNode(
        name="node2",
        datacenter="DC1",
        zone="us-east-1b",
        total_cpu_cores=8.0,
        total_memory_mb=16384.0,
        cpu_utilization=0.7,
        memory_utilization=0.6
    )
    node3 = InfrastructureNode(
        name="node3",
        datacenter="DC2",
        zone="us-west-1a",
        total_cpu_cores=4.0,
        total_memory_mb=8192.0,
        cpu_utilization=0.8,
        memory_utilization=0.7
    )
    
    model.add_node(node1)
    model.add_node(node2)
    model.add_node(node3)
    
    # Brokers
    broker1 = BrokerNode(
        name="broker1",
        node_host="node1",
        max_throughput_msgs_per_sec=10000.0,
        current_throughput=6000.0,
        cpu_utilization=0.6,
        broker_type="Kafka"
    )
    broker2 = BrokerNode(
        name="broker2",
        node_host="node2",
        max_throughput_msgs_per_sec=10000.0,
        current_throughput=7000.0,
        cpu_utilization=0.7,
        broker_type="Kafka"
    )
    
    model.add_broker(broker1)
    model.add_broker(broker2)
    
    # Topics with QoS policies
    topics_config = [
        {
            'name': 'orders.created',
            'broker': 'broker1',
            'qos': QoSPolicy(
                durability=QoSDurability.PERSISTENT,
                reliability=QoSReliability.RELIABLE,
                deadline_ms=1000,
                transport_priority=90
            ),
            'rate': 100.0
        },
        {
            'name': 'orders.updated',
            'broker': 'broker1',
            'qos': QoSPolicy(
                durability=QoSDurability.TRANSIENT_LOCAL,
                reliability=QoSReliability.RELIABLE,
                deadline_ms=2000,
                transport_priority=70
            ),
            'rate': 50.0
        },
        {
            'name': 'inventory.changed',
            'broker': 'broker2',
            'qos': QoSPolicy(
                durability=QoSDurability.PERSISTENT,
                reliability=QoSReliability.RELIABLE,
                deadline_ms=500,
                transport_priority=95
            ),
            'rate': 200.0
        },
        {
            'name': 'notifications.email',
            'broker': 'broker2',
            'qos': QoSPolicy(
                durability=QoSDurability.VOLATILE,
                reliability=QoSReliability.BEST_EFFORT,
                deadline_ms=5000,
                transport_priority=30
            ),
            'rate': 20.0
        },
        {
            'name': 'analytics.events',
            'broker': 'broker1',
            'qos': QoSPolicy(
                durability=QoSDurability.VOLATILE,
                reliability=QoSReliability.BEST_EFFORT,
                transport_priority=20
            ),
            'rate': 500.0
        }
    ]
    
    for config in topics_config:
        topic = TopicNode(
            name=config['name'],
            broker=config['broker'],
            qos_policy=config['qos'],
            message_rate_per_sec=config['rate']
        )
        model.add_topic(topic)
    
    # Applications
    apps_config = [
        {
            'name': 'OrderService',
            'type': ApplicationType.PROSUMER,
            'node': 'node1',
            'publishes': ['orders.created', 'orders.updated'],
            'subscribes': ['inventory.changed'],
            'latency': 100,
            'cpu': 2.0,
            'mem': 2048.0,
            'domain': 'Commerce'
        },
        {
            'name': 'InventoryService',
            'type': ApplicationType.PROSUMER,
            'node': 'node2',
            'publishes': ['inventory.changed'],
            'subscribes': ['orders.created'],
            'latency': 50,
            'cpu': 1.5,
            'mem': 1024.0,
            'domain': 'Inventory'
        },
        {
            'name': 'NotificationService',
            'type': ApplicationType.CONSUMER,
            'node': 'node2',
            'publishes': ['notifications.email'],
            'subscribes': ['orders.created', 'orders.updated'],
            'latency': 200,
            'cpu': 1.0,
            'mem': 512.0,
            'domain': 'Notifications'
        },
        {
            'name': 'AnalyticsService',
            'type': ApplicationType.CONSUMER,
            'node': 'node3',
            'publishes': [],
            'subscribes': ['orders.created', 'orders.updated', 'inventory.changed', 'analytics.events'],
            'latency': 1000,
            'cpu': 2.0,
            'mem': 4096.0,
            'domain': 'Analytics'
        },
        {
            'name': 'ReportingService',
            'type': ApplicationType.PRODUCER,
            'node': 'node3',
            'publishes': ['analytics.events'],
            'subscribes': [],
            'latency': 500,
            'cpu': 1.0,
            'mem': 1024.0,
            'domain': 'Analytics'
        }
    ]
    
    for config in apps_config:
        app = ApplicationNode(
            name=config['name'],
            app_type=config['type'],
            node_host=config['node'],
            required_latency_ms=config['latency'],
            cpu_cores=config['cpu'],
            memory_mb=config['mem'],
            business_domain=config['domain']
        )
        model.add_application(app)
        
        # Add publish edges
        for topic in config['publishes']:
            model.publishes_edges.append(PublishesEdge(
                source=config['name'],
                target=topic,
                message_pattern=MessagePattern.EVENT_DRIVEN,
                message_rate_per_sec=50.0
            ))
        
        # Add subscribe edges
        for topic in config['subscribes']:
            model.subscribes_edges.append(SubscribesEdge(
                source=config['name'],
                target=topic,
                qos_compatible=True
            ))
    
    # Add broker routes edges
    for topic in topics_config:
        model.routes_edges.append(RoutesEdge(
            source=topic['broker'],
            target=topic['name'],
            routing_weight=1.0
        ))
    
    # Add RUNS_ON edges
    for config in apps_config:
        model.runs_on_edges.append(RunsOnEdge(
            source=config['name'],
            target=config['node']
        ))
    
    model.runs_on_edges.append(RunsOnEdge(source='broker1', target='node1'))
    model.runs_on_edges.append(RunsOnEdge(source='broker2', target='node2'))
    
    # Add network connectivity
    model.connects_to_edges.append(ConnectsToEdge(
        source='node1',
        target='node2',
        bandwidth_mbps=10000.0,
        latency_ms=1.0
    ))
    model.connects_to_edges.append(ConnectsToEdge(
        source='node2',
        target='node3',
        bandwidth_mbps=1000.0,
        latency_ms=50.0
    ))
    model.connects_to_edges.append(ConnectsToEdge(
        source='node1',
        target='node3',
        bandwidth_mbps=1000.0,
        latency_ms=45.0
    ))
    
    # Derive DEPENDS_ON edges
    # OrderService depends on InventoryService via topics
    model.depends_on_edges.append(DependsOnEdge(
        source='OrderService',
        target='InventoryService',
        dependency_type='FUNCTIONAL',
        strength=0.9
    ))
    model.depends_on_edges.append(DependsOnEdge(
        source='InventoryService',
        target='OrderService',
        dependency_type='FUNCTIONAL',
        strength=0.8
    ))
    model.depends_on_edges.append(DependsOnEdge(
        source='NotificationService',
        target='OrderService',
        dependency_type='FUNCTIONAL',
        strength=0.7
    ))
    
    return model


def convert_model_to_networkx(model: GraphModel) -> nx.DiGraph:
    """Convert GraphModel to NetworkX DiGraph"""
    
    graph = nx.DiGraph()
    
    # Add all nodes
    for name, node_dict in model.get_all_nodes().items():
        graph.add_node(name, **node_dict)
    
    # Add all edges
    for edge_dict in model.get_all_edges():
        source = edge_dict.pop('source')
        target = edge_dict.pop('target')
        graph.add_edge(source, target, **edge_dict)
    
    return graph


def main():
    """Run complete analysis example"""
    
    print("=" * 70)
    print("Software-as-a-Graph: Implementation Example")
    print("=" * 70)
    
    # Step 1: Create sample system
    print("\n[1] Creating sample pub-sub system...")
    model = create_sample_system()
    summary = model.summary()
    print(f"  ✓ Created system with:")
    print(f"    - {summary['applications']} applications")
    print(f"    - {summary['topics']} topics")
    print(f"    - {summary['brokers']} brokers")
    print(f"    - {summary['nodes']} infrastructure nodes")
    print(f"    - {summary['total_edges']} edges")
    
    # Step 2: Convert to NetworkX
    print("\n[2] Converting to NetworkX graph...")
    graph = convert_model_to_networkx(model)
    print(f"  ✓ Graph ready: {len(graph)} nodes, {len(graph.edges())} edges")
    
    # Step 3: Basic criticality analysis
    print("\n[3] Running basic criticality analysis...")
    scorer = CompositeCriticalityScorer(
        alpha=0.4,
        beta=0.3,
        gamma=0.3,
        qos_enabled=True
    )
    
    scores = scorer.calculate_all_scores(graph)
    top_10 = scorer.get_top_critical(scores, n=10)
    
    print(f"  ✓ Analyzed {len(scores)} components")
    print(f"\n  Top 5 Critical Components:")
    for i, score in enumerate(top_10[:5], 1):
        print(f"    {i}. {score.component} ({score.component_type})")
        print(f"       Score: {score.composite_score:.3f} [{score.criticality_level.value}]")
        print(f"       Betweenness: {score.betweenness_centrality_norm:.3f}, "
              f"AP: {bool(score.is_articulation_point)}, "
              f"Impact: {score.impact_score:.3f}")
    
    # Step 4: Comprehensive orchestrated analysis
    print("\n[4] Running comprehensive analysis pipeline...")
    orchestrator = AnalysisOrchestrator(
        output_dir="output/",
        enable_qos=True
    )
    
    results = orchestrator.analyze_graph(
        graph=graph,
        graph_model=model,
        enable_simulation=True
    )
    
    print(f"  ✓ Analysis complete in {results['execution_time']['total']:.2f}s")
    
    # Step 5: Print detailed summary
    print("\n[5] Analysis Summary:")
    orchestrator.print_summary()
    
    # Step 6: Export results
    print("\n[6] Exporting results...")
    output_file = orchestrator.export_results("example_analysis.json")
    print(f"  ✓ Results saved to {output_file}")
    
    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    
    # Return for further use
    return model, graph, scores, results


if __name__ == "__main__":
    model, graph, scores, results = main()
    
    # Additional analysis examples
    print("\n" + "=" * 70)
    print("Additional Analysis Examples")
    print("=" * 70)
    
    # Get summary statistics
    from src.analysis.criticality_scorer import CompositeCriticalityScorer
    scorer = CompositeCriticalityScorer(qos_enabled=True)
    summary = scorer.summarize_criticality(scores)
    
    print(f"\nCriticality Distribution:")
    print(f"  Total Components: {summary['total_components']}")
    print(f"  Critical (≥0.8): {summary['critical_count']}")
    print(f"  High (0.6-0.8): {summary['high_count']}")
    print(f"  Medium (0.4-0.6): {summary['medium_count']}")
    print(f"  Low (0.2-0.4): {summary['low_count']}")
    print(f"  Minimal (<0.2): {summary['minimal_count']}")
    print(f"\n  Articulation Points: {summary['articulation_points']}")
    print(f"  High Impact Components: {summary['high_impact_count']}")
    print(f"  Average Score: {summary['avg_score']:.3f}")
    print(f"  Score Std Dev: {summary['std_score']:.3f}")
    
    # Show specific component details
    print(f"\nDetailed Component Analysis:")
    for component_name in ['broker1', 'orders.created', 'OrderService']:
        if component_name in scores:
            score = scores[component_name]
            print(f"\n  {component_name}:")
            print(f"    Type: {score.component_type}")
            print(f"    Composite Score: {score.composite_score:.3f}")
            print(f"    Level: {score.criticality_level.value}")
            print(f"    Betweenness Centrality: {score.betweenness_centrality_norm:.3f}")
            print(f"    Degree Centrality: {score.degree_centrality:.3f}")
            print(f"    Closeness Centrality: {score.closeness_centrality:.3f}")
            print(f"    PageRank: {score.pagerank:.3f}")
            print(f"    QoS Score: {score.qos_score:.3f}")
            print(f"    Is Articulation Point: {bool(score.is_articulation_point)}")
            print(f"    Impact Score: {score.impact_score:.3f}")
            print(f"    Components Affected: {score.components_affected}")
            print(f"    Reachability Loss: {score.reachability_loss_percentage:.1f}%")
