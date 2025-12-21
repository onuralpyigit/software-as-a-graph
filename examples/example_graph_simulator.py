#!/usr/bin/env python3
"""
Simulation Examples for Pub-Sub Systems
=========================================

Quick examples demonstrating simulation capabilities:
1. Failure Simulation Examples
2. Event-Driven Simulation Examples
3. Combined Analysis Examples

Run with:
    python examples/simulation_examples.py
    
Or individual examples:
    python examples/simulation_examples.py --example failure
    python examples/simulation_examples.py --example event
    python examples/simulation_examples.py --example all

Author: Software-as-a-Graph Research Project
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from src.simulation import (
    FailureSimulator,
    EventDrivenSimulator,
    AttackStrategy,
    FailureType,
    FailureMode
)


# ============================================================================
# Terminal Formatting
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(title: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{title.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def print_section(title: str):
    """Print subsection header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.ENDC}")
    print(f"{Colors.DIM}{'-'*50}{Colors.ENDC}" if hasattr(Colors, 'DIM') else '-'*50)


def print_result(label: str, value, color=None):
    """Print a result line"""
    color = color or ''
    end_color = Colors.ENDC if color else ''
    print(f"  {label}: {color}{value}{end_color}")


# ============================================================================
# Graph Generation
# ============================================================================

def create_financial_trading_system() -> nx.DiGraph:
    """
    Create a realistic financial trading pub-sub system.
    
    Components:
    - Market data feed handlers
    - Order management systems
    - Risk management services
    - Trade execution engines
    - Reporting services
    """
    G = nx.DiGraph()
    
    # Infrastructure nodes
    infra = [
        ('datacenter_east', 'Node'),
        ('datacenter_west', 'Node'),
    ]
    
    # Brokers (message brokers)
    brokers = [
        ('broker_primary', 'Broker'),
        ('broker_backup', 'Broker'),
        ('broker_regional', 'Broker'),
    ]
    
    # Topics (message channels)
    topics = [
        ('market_data', 'Topic'),
        ('orders', 'Topic'),
        ('executions', 'Topic'),
        ('risk_alerts', 'Topic'),
        ('audit_log', 'Topic'),
    ]
    
    # Applications
    apps = [
        ('market_feed_nyse', 'Application'),
        ('market_feed_nasdaq', 'Application'),
        ('order_gateway', 'Application'),
        ('matching_engine', 'Application'),
        ('risk_manager', 'Application'),
        ('position_keeper', 'Application'),
        ('trade_reporter', 'Application'),
        ('compliance_monitor', 'Application'),
    ]
    
    # Add all nodes
    for node_id, node_type in infra + brokers + topics + apps:
        G.add_node(node_id, type=node_type)
    
    # Infrastructure connections
    G.add_edge('datacenter_east', 'broker_primary')
    G.add_edge('datacenter_west', 'broker_backup')
    G.add_edge('datacenter_east', 'broker_regional')
    G.add_edge('broker_primary', 'broker_backup')  # Replication
    
    # Broker to topic routing
    G.add_edge('broker_primary', 'market_data')
    G.add_edge('broker_primary', 'orders')
    G.add_edge('broker_primary', 'executions')
    G.add_edge('broker_backup', 'risk_alerts')
    G.add_edge('broker_regional', 'audit_log')
    
    # Publishers
    G.add_edge('market_feed_nyse', 'market_data')
    G.add_edge('market_feed_nasdaq', 'market_data')
    G.add_edge('order_gateway', 'orders')
    G.add_edge('matching_engine', 'executions')
    G.add_edge('risk_manager', 'risk_alerts')
    
    # Subscribers
    G.add_edge('market_data', 'matching_engine')
    G.add_edge('market_data', 'risk_manager')
    G.add_edge('orders', 'matching_engine')
    G.add_edge('orders', 'risk_manager')
    G.add_edge('executions', 'position_keeper')
    G.add_edge('executions', 'trade_reporter')
    G.add_edge('risk_alerts', 'order_gateway')
    G.add_edge('risk_alerts', 'compliance_monitor')
    
    # Audit trail
    G.add_edge('order_gateway', 'audit_log')
    G.add_edge('matching_engine', 'audit_log')
    G.add_edge('audit_log', 'compliance_monitor')
    
    return G


def create_iot_smart_city() -> nx.DiGraph:
    """
    Create an IoT smart city pub-sub system.
    
    Components:
    - Sensor gateways
    - Data aggregators
    - Analytics services
    - Alert systems
    """
    G = nx.DiGraph()
    
    # Edge nodes
    nodes = [
        ('edge_north', 'Node'),
        ('edge_south', 'Node'),
        ('edge_central', 'Node'),
    ]
    
    # Message brokers
    brokers = [
        ('mqtt_broker_1', 'Broker'),
        ('mqtt_broker_2', 'Broker'),
        ('kafka_cluster', 'Broker'),
    ]
    
    # Topics
    topics = [
        ('traffic_sensors', 'Topic'),
        ('weather_data', 'Topic'),
        ('air_quality', 'Topic'),
        ('emergency_alerts', 'Topic'),
        ('aggregated_metrics', 'Topic'),
    ]
    
    # Applications
    apps = [
        ('traffic_gateway_1', 'Application'),
        ('traffic_gateway_2', 'Application'),
        ('weather_station', 'Application'),
        ('air_monitor_1', 'Application'),
        ('air_monitor_2', 'Application'),
        ('data_aggregator', 'Application'),
        ('ml_predictor', 'Application'),
        ('alert_service', 'Application'),
        ('dashboard_api', 'Application'),
        ('mobile_notifier', 'Application'),
    ]
    
    for node_id, node_type in nodes + brokers + topics + apps:
        G.add_node(node_id, type=node_type)
    
    # Infrastructure
    G.add_edge('edge_north', 'mqtt_broker_1')
    G.add_edge('edge_south', 'mqtt_broker_2')
    G.add_edge('edge_central', 'kafka_cluster')
    G.add_edge('mqtt_broker_1', 'kafka_cluster')
    G.add_edge('mqtt_broker_2', 'kafka_cluster')
    
    # Broker routing
    G.add_edge('mqtt_broker_1', 'traffic_sensors')
    G.add_edge('mqtt_broker_1', 'weather_data')
    G.add_edge('mqtt_broker_2', 'air_quality')
    G.add_edge('kafka_cluster', 'aggregated_metrics')
    G.add_edge('kafka_cluster', 'emergency_alerts')
    
    # Publishers
    G.add_edge('traffic_gateway_1', 'traffic_sensors')
    G.add_edge('traffic_gateway_2', 'traffic_sensors')
    G.add_edge('weather_station', 'weather_data')
    G.add_edge('air_monitor_1', 'air_quality')
    G.add_edge('air_monitor_2', 'air_quality')
    G.add_edge('data_aggregator', 'aggregated_metrics')
    G.add_edge('ml_predictor', 'emergency_alerts')
    
    # Subscribers
    G.add_edge('traffic_sensors', 'data_aggregator')
    G.add_edge('traffic_sensors', 'ml_predictor')
    G.add_edge('weather_data', 'data_aggregator')
    G.add_edge('weather_data', 'ml_predictor')
    G.add_edge('air_quality', 'data_aggregator')
    G.add_edge('air_quality', 'alert_service')
    G.add_edge('aggregated_metrics', 'dashboard_api')
    G.add_edge('emergency_alerts', 'alert_service')
    G.add_edge('emergency_alerts', 'mobile_notifier')
    
    return G


def create_microservices_system() -> nx.DiGraph:
    """
    Create a microservices e-commerce pub-sub system.
    """
    G = nx.DiGraph()
    
    # Infrastructure
    G.add_node('k8s_cluster', type='Node')
    G.add_node('rabbitmq', type='Broker')
    G.add_node('redis_pubsub', type='Broker')
    
    # Topics/Queues
    topics = ['user_events', 'order_events', 'inventory_events', 
              'payment_events', 'notification_queue']
    for t in topics:
        G.add_node(t, type='Topic')
    
    # Services
    services = ['api_gateway', 'user_service', 'order_service', 
                'inventory_service', 'payment_service', 'notification_service',
                'analytics_service', 'recommendation_engine']
    for s in services:
        G.add_node(s, type='Application')
    
    # Infrastructure connections
    G.add_edge('k8s_cluster', 'rabbitmq')
    G.add_edge('k8s_cluster', 'redis_pubsub')
    
    # Broker routing
    G.add_edge('rabbitmq', 'user_events')
    G.add_edge('rabbitmq', 'order_events')
    G.add_edge('rabbitmq', 'inventory_events')
    G.add_edge('rabbitmq', 'payment_events')
    G.add_edge('redis_pubsub', 'notification_queue')
    
    # Publishers
    G.add_edge('api_gateway', 'user_events')
    G.add_edge('user_service', 'user_events')
    G.add_edge('order_service', 'order_events')
    G.add_edge('inventory_service', 'inventory_events')
    G.add_edge('payment_service', 'payment_events')
    
    # Subscribers
    G.add_edge('user_events', 'analytics_service')
    G.add_edge('user_events', 'recommendation_engine')
    G.add_edge('order_events', 'inventory_service')
    G.add_edge('order_events', 'payment_service')
    G.add_edge('order_events', 'notification_service')
    G.add_edge('inventory_events', 'order_service')
    G.add_edge('payment_events', 'order_service')
    G.add_edge('payment_events', 'notification_service')
    G.add_edge('notification_queue', 'notification_service')
    
    return G


# ============================================================================
# Example Functions
# ============================================================================

def example_single_failure():
    """Example: Single component failure simulation"""
    print_header("EXAMPLE: Single Component Failure")
    
    graph = create_financial_trading_system()
    print(f"System: Financial Trading ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
    
    simulator = FailureSimulator(seed=42)
    
    # Fail the primary broker
    result = simulator.simulate_single_failure(
        graph, 
        'broker_primary',
        failure_type=FailureType.COMPLETE,
        enable_cascade=True
    )
    
    print_section("Results")
    print_result("Simulation ID", result.simulation_id)
    print_result("Component Failed", 'broker_primary')
    print_result("Cascade Failures", len(result.cascade_failures))
    print_result("Reachability Loss", f"{result.impact.reachability_loss*100:.1f}%", Colors.WARNING)
    print_result("Impact Score", f"{result.impact_score:.4f}", Colors.RED if result.impact_score > 0.5 else Colors.GREEN)
    print_result("Resilience Score", f"{result.resilience_score:.4f}")
    
    if result.cascade_failures:
        print(f"\n  Cascade chain: broker_primary → {' → '.join(result.cascade_failures[:5])}")
    
    return result


def example_multiple_failures():
    """Example: Multiple simultaneous failures"""
    print_header("EXAMPLE: Multiple Component Failures")
    
    graph = create_iot_smart_city()
    print(f"System: IoT Smart City ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
    
    simulator = FailureSimulator(seed=42)
    
    # Fail multiple edge nodes (simulating regional outage)
    components = ['edge_north', 'mqtt_broker_1']
    
    result = simulator.simulate_multiple_failures(
        graph,
        components,
        enable_cascade=True
    )
    
    print_section("Results")
    print_result("Components Failed", ', '.join(components))
    print_result("Total Failures", len(result.all_failures))
    print_result("Cascade Failures", len(result.cascade_failures))
    print_result("Fragmentation", f"{result.impact.fragmentation} new components")
    print_result("Impact Score", f"{result.impact_score:.4f}")
    
    if result.impact.isolated_nodes:
        print(f"\n  Isolated nodes: {', '.join(result.impact.isolated_nodes[:5])}")
    
    return result


def example_targeted_attack():
    """Example: Targeted attack on critical components"""
    print_header("EXAMPLE: Targeted Attack Simulation")
    
    graph = create_microservices_system()
    print(f"System: E-commerce Microservices ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
    
    simulator = FailureSimulator(seed=42)
    
    # Attack using different strategies
    strategies = [
        (AttackStrategy.CRITICALITY, "Criticality-based"),
        (AttackStrategy.BETWEENNESS, "Betweenness-based"),
        (AttackStrategy.DEGREE, "Degree-based"),
    ]
    
    print_section("Attack Comparison (3 targets each)")
    
    for strategy, name in strategies:
        result = simulator.simulate_targeted_attack(
            graph,
            strategy=strategy,
            target_count=3,
            enable_cascade=False
        )
        
        targets = ', '.join(result.primary_failures)
        print(f"\n  {Colors.BOLD}{name}:{Colors.ENDC}")
        print(f"    Targets: {targets}")
        print(f"    Impact: {result.impact_score:.4f}")
    
    return result


def example_exhaustive_campaign():
    """Example: Exhaustive failure campaign"""
    print_header("EXAMPLE: Exhaustive Failure Campaign")
    
    graph = create_financial_trading_system()
    print(f"System: Financial Trading ({graph.number_of_nodes()} nodes)")
    
    simulator = FailureSimulator(seed=42)
    
    # Test all components
    result = simulator.simulate_exhaustive(
        graph,
        enable_cascade=False
    )
    
    print_section("Campaign Results")
    print_result("Total Tests", result.total_simulations)
    print_result("Completed", result.completed_simulations)
    print_result("Avg Impact", f"{result.avg_impact_score:.4f}")
    print_result("Max Impact", f"{result.max_impact_score:.4f}", Colors.RED)
    print_result("Min Impact", f"{result.min_impact_score:.4f}", Colors.GREEN)
    print_result("Duration", f"{result.total_duration_ms:.0f}ms")
    
    print_section("Most Critical Components")
    for i, (comp, score) in enumerate(result.most_critical[:5], 1):
        color = Colors.RED if score > 0.5 else Colors.WARNING if score > 0.3 else Colors.GREEN
        print(f"  {i}. {comp:25s} {color}{score:.4f}{Colors.ENDC}")
    
    print_section("Least Critical Components")
    for i, (comp, score) in enumerate(result.least_critical[:3], 1):
        print(f"  {i}. {comp:25s} {score:.4f}")
    
    return result


def example_event_simulation():
    """Example: Event-driven message simulation"""
    print_header("EXAMPLE: Event-Driven Simulation")
    
    graph = create_iot_smart_city()
    print(f"System: IoT Smart City ({graph.number_of_nodes()} nodes)")
    
    simulator = EventDrivenSimulator(seed=42)
    
    # Run simulation
    result = simulator.simulate(
        graph,
        duration_ms=5000,  # 5 seconds simulated
        message_rate=100   # 100 messages/second
    )
    
    print_section("Simulation Results")
    print_result("Duration (simulated)", f"{result.duration_ms:.0f}ms")
    print_result("Duration (real)", f"{result.real_time_ms:.0f}ms")
    print_result("Speedup", f"{result.speedup:.0f}x real-time", Colors.CYAN)
    
    print_section("Message Statistics")
    m = result.metrics
    delivery_color = Colors.GREEN if m.delivery_rate > 0.95 else Colors.WARNING
    print_result("Total Messages", m.total_messages)
    print_result("Delivered", f"{m.delivered_messages} ({m.delivery_rate*100:.1f}%)", delivery_color)
    print_result("Dropped", m.dropped_messages)
    print_result("Throughput", f"{m.messages_per_second:.0f} msg/sec")
    
    print_section("Latency Metrics")
    print_result("Average", f"{m.avg_latency:.2f}ms")
    print_result("P50", f"{m.p50_latency:.2f}ms")
    print_result("P95", f"{m.p95_latency:.2f}ms")
    print_result("P99", f"{m.p99_latency:.2f}ms")
    
    return result


def example_failure_injection():
    """Example: Event simulation with failure injection"""
    print_header("EXAMPLE: Event Simulation with Failure Injection")
    
    graph = create_financial_trading_system()
    print(f"System: Financial Trading ({graph.number_of_nodes()} nodes)")
    
    simulator = EventDrivenSimulator(seed=42)
    
    # Schedule failures
    failure_schedule = [
        {'time_ms': 2000, 'component': 'broker_primary', 'duration_ms': 1000},
        {'time_ms': 4000, 'component': 'matching_engine', 'duration_ms': 500},
    ]
    
    print(f"Scheduled failures:")
    for f in failure_schedule:
        print(f"  - {f['component']} at t={f['time_ms']}ms for {f['duration_ms']}ms")
    
    # Run simulation
    result = simulator.simulate(
        graph,
        duration_ms=6000,
        message_rate=200,
        failure_schedule=failure_schedule
    )
    
    print_section("Results with Failures")
    m = result.metrics
    print_result("Total Messages", m.total_messages)
    print_result("Delivery Rate", f"{m.delivery_rate*100:.1f}%", 
                Colors.GREEN if m.delivery_rate > 0.9 else Colors.WARNING)
    print_result("Component Failures", m.component_failures, Colors.RED)
    print_result("Avg Latency", f"{m.avg_latency:.2f}ms")
    print_result("P99 Latency", f"{m.p99_latency:.2f}ms")
    
    return result


def example_load_test():
    """Example: Load testing with ramping rate"""
    print_header("EXAMPLE: Load Testing")
    
    graph = create_microservices_system()
    print(f"System: E-commerce ({graph.number_of_nodes()} nodes)")
    
    simulator = EventDrivenSimulator(seed=42)
    
    print("Load profile: 10 → 500 msg/sec over 3 seconds, sustain for 2 seconds")
    
    result = simulator.simulate_with_load_test(
        graph,
        duration_ms=5000,
        initial_rate=10,
        peak_rate=500,
        ramp_time_ms=3000
    )
    
    print_section("Load Test Results")
    m = result.metrics
    print_result("Peak Throughput", f"{m.messages_per_second:.0f} msg/sec")
    print_result("Total Messages", m.total_messages)
    print_result("Delivery Rate", f"{m.delivery_rate*100:.1f}%")
    print_result("P99 Latency", f"{m.p99_latency:.2f}ms")
    print_result("Speedup", f"{result.speedup:.0f}x")
    
    return result


def example_chaos_engineering():
    """Example: Chaos engineering simulation"""
    print_header("EXAMPLE: Chaos Engineering")
    
    graph = create_iot_smart_city()
    print(f"System: IoT Smart City ({graph.number_of_nodes()} nodes)")
    
    simulator = EventDrivenSimulator(seed=42)
    
    print("Chaos parameters:")
    print("  - Failure probability: 5% per check")
    print("  - Recovery probability: 50% per check")
    print("  - Check interval: 500ms")
    
    result = simulator.simulate_chaos(
        graph,
        duration_ms=5000,
        message_rate=100,
        failure_probability=0.05,
        recovery_probability=0.5,
        check_interval_ms=500
    )
    
    print_section("Chaos Test Results")
    m = result.metrics
    print_result("Total Messages", m.total_messages)
    print_result("Delivery Rate", f"{m.delivery_rate*100:.1f}%")
    print_result("Component Failures", m.component_failures, Colors.RED if m.component_failures > 5 else Colors.WARNING)
    print_result("P99 Latency", f"{m.p99_latency:.2f}ms")
    
    # Show component availability
    print_section("Component Availability (sample)")
    sorted_stats = sorted(
        result.component_stats.values(),
        key=lambda s: s.availability
    )[:5]  # Show lowest availability
    
    for stats in sorted_stats:
        avail_color = Colors.RED if stats.availability < 0.9 else Colors.WARNING if stats.availability < 0.95 else Colors.GREEN
        print(f"  {stats.component_id:25s} {avail_color}{stats.availability*100:.1f}%{Colors.ENDC}")
    
    return result


def example_combined_analysis():
    """Example: Combined failure analysis with impact validation"""
    print_header("EXAMPLE: Combined Analysis")
    
    graph = create_financial_trading_system()
    print(f"System: Financial Trading")
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    
    # Step 1: Find critical components via exhaustive simulation
    print_section("Step 1: Identify Critical Components")
    
    failure_sim = FailureSimulator(seed=42)
    campaign = failure_sim.simulate_exhaustive(graph, enable_cascade=False)
    
    top_critical = campaign.most_critical[:3]
    print("Top 3 critical components:")
    for comp, score in top_critical:
        print(f"  - {comp}: impact={score:.4f}")
    
    # Step 2: Validate with event-driven simulation
    print_section("Step 2: Validate with Event Simulation")
    
    event_sim = EventDrivenSimulator(seed=42)
    
    # Baseline (no failures)
    baseline = event_sim.simulate(graph, duration_ms=3000, message_rate=200)
    print(f"\nBaseline: {baseline.metrics.delivery_rate*100:.1f}% delivery, "
          f"P99={baseline.metrics.p99_latency:.2f}ms")
    
    # With critical component failure
    most_critical = top_critical[0][0]
    failure_schedule = [{'time_ms': 1000, 'component': most_critical, 'duration_ms': 1000}]
    
    with_failure = event_sim.simulate(
        graph, duration_ms=3000, message_rate=200,
        failure_schedule=failure_schedule
    )
    
    print(f"\nWith {most_critical} failure:")
    print(f"  Delivery: {with_failure.metrics.delivery_rate*100:.1f}% "
          f"(Δ{(with_failure.metrics.delivery_rate - baseline.metrics.delivery_rate)*100:+.1f}%)")
    print(f"  P99 Latency: {with_failure.metrics.p99_latency:.2f}ms "
          f"(Δ{with_failure.metrics.p99_latency - baseline.metrics.p99_latency:+.2f}ms)")
    
    # Step 3: Summary
    print_section("Summary")
    print(f"Most critical component: {Colors.RED}{most_critical}{Colors.ENDC}")
    print(f"Static impact score: {top_critical[0][1]:.4f}")
    print(f"Actual delivery impact: {(baseline.metrics.delivery_rate - with_failure.metrics.delivery_rate)*100:.1f}%")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Simulation examples')
    parser.add_argument('--example', choices=['failure', 'event', 'all'], 
                       default='all', help='Which examples to run')
    parser.add_argument('--export', help='Export results to JSON file')
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}Pub-Sub System Simulation Examples{Colors.ENDC}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    if args.example in ['failure', 'all']:
        results['single_failure'] = example_single_failure()
        results['multiple_failures'] = example_multiple_failures()
        results['targeted_attack'] = example_targeted_attack()
        results['exhaustive'] = example_exhaustive_campaign()
    
    if args.example in ['event', 'all']:
        results['event_sim'] = example_event_simulation()
        results['failure_injection'] = example_failure_injection()
        results['load_test'] = example_load_test()
        results['chaos'] = example_chaos_engineering()
    
    if args.example == 'all':
        example_combined_analysis()
    
    # Export if requested
    if args.export:
        export_data = {}
        for name, result in results.items():
            if hasattr(result, 'to_dict'):
                export_data[name] = result.to_dict()
        
        with open(args.export, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"\n{Colors.GREEN}✓{Colors.ENDC} Results exported to {args.export}")
    
    print(f"\n{Colors.GREEN}All examples completed successfully!{Colors.ENDC}\n")


if __name__ == '__main__':
    main()