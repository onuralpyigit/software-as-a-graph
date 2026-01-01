#!/usr/bin/env python3
"""
Example Simulation Script - Version 5.0

Demonstrates the simulation capabilities of the software-as-a-graph project.

Features Demonstrated:
1. Failure Simulation - Component failure impact analysis
2. Event-Driven Simulation - Message flow modeling
3. Component-Type Specific Simulation
4. Campaign Simulation - Test all components
5. Neo4j Integration (optional)

Usage:
    # Run all demos (no Neo4j required)
    python examples/example_simulation.py --demo

    # Run with Neo4j
    python examples/example_simulation.py --uri bolt://localhost:7687 --password secret

    # Export results to JSON
    python examples/example_simulation.py --demo --output results.json

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import (
    # Graph Model
    SimulationGraph,
    Component,
    Edge,
    ComponentType,
    EdgeType,
    ComponentStatus,
    QoSPolicy,
    create_simulation_graph,
    # Failure Simulator
    FailureSimulator,
    FailureMode,
    FailureResult,
    CampaignResult,
    simulate_single_failure,
    simulate_all_components,
    # Event Simulator
    EventSimulator,
    EventType,
    MessageStatus,
    SimulationResult,
    run_event_simulation,
    run_stress_test,
)


# =============================================================================
# Output Formatting
# =============================================================================

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GRAY = "\033[90m"


def print_header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}{title.center(60)}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def print_subheader(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print(f"{'-' * 40}")


def print_success(msg: str) -> None:
    print(f"{GREEN}✓ {msg}{RESET}")


def print_info(msg: str) -> None:
    print(f"{BLUE}ℹ {msg}{RESET}")


def impact_color(score: float) -> str:
    if score >= 0.5:
        return RED
    elif score >= 0.3:
        return YELLOW
    elif score >= 0.1:
        return BLUE
    return GREEN


# =============================================================================
# Demo Graph Creation
# =============================================================================

def create_demo_graph() -> SimulationGraph:
    """
    Create a realistic demo pub-sub system graph.
    
    System Structure:
    - 3 Nodes (compute infrastructure)
    - 2 Brokers (message routing)
    - 5 Topics (message channels)
    - 8 Applications (publishers/subscribers)
    
    This represents a small IoT system with sensors, processors, and displays.
    """
    graph = SimulationGraph()
    
    # === Nodes (Infrastructure Layer) ===
    nodes = [
        ("node_edge", "Edge Server"),
        ("node_cloud1", "Cloud Server 1"),
        ("node_cloud2", "Cloud Server 2"),
    ]
    for node_id, name in nodes:
        graph.add_component(Component(
            id=node_id,
            type=ComponentType.NODE,
            name=name,
            properties={"cpu_cores": 4, "memory_gb": 16}
        ))
    
    # === Brokers (Messaging Layer) ===
    brokers = [
        ("broker_edge", "Edge Broker", "node_edge"),
        ("broker_cloud", "Cloud Broker", "node_cloud1"),
    ]
    for broker_id, name, node in brokers:
        graph.add_component(Component(
            id=broker_id,
            type=ComponentType.BROKER,
            name=name,
            properties={"max_connections": 1000}
        ))
        graph.add_edge(Edge(
            source=broker_id,
            target=node,
            edge_type=EdgeType.RUNS_ON
        ))
    
    # === Topics (Communication Channels) ===
    topics = [
        ("topic_sensor_data", "Sensor Data", "broker_edge"),
        ("topic_alerts", "Alerts", "broker_edge"),
        ("topic_commands", "Commands", "broker_cloud"),
        ("topic_analytics", "Analytics", "broker_cloud"),
        ("topic_status", "Status Updates", "broker_cloud"),
    ]
    for topic_id, name, broker in topics:
        graph.add_component(Component(
            id=topic_id,
            type=ComponentType.TOPIC,
            name=name,
            properties={"retention_hours": 24}
        ))
        graph.add_edge(Edge(
            source=broker,
            target=topic_id,
            edge_type=EdgeType.ROUTES
        ))
    
    # === Applications ===
    applications = [
        # Publishers
        ("app_sensor1", "Temperature Sensor", "node_edge", 
         ["topic_sensor_data"], []),
        ("app_sensor2", "Humidity Sensor", "node_edge",
         ["topic_sensor_data"], []),
        ("app_controller", "Controller", "node_cloud1",
         ["topic_commands", "topic_alerts"], ["topic_sensor_data"]),
        
        # Processors
        ("app_processor", "Data Processor", "node_cloud1",
         ["topic_analytics"], ["topic_sensor_data", "topic_commands"]),
        ("app_alerter", "Alert Generator", "node_cloud1",
         ["topic_alerts"], ["topic_sensor_data"]),
        
        # Subscribers
        ("app_dashboard", "Dashboard", "node_cloud2",
         [], ["topic_alerts", "topic_analytics", "topic_status"]),
        ("app_logger", "Data Logger", "node_cloud2",
         ["topic_status"], ["topic_sensor_data", "topic_analytics"]),
        ("app_notifier", "Notification Service", "node_cloud2",
         [], ["topic_alerts"]),
    ]
    
    for app_id, name, node, pub_topics, sub_topics in applications:
        graph.add_component(Component(
            id=app_id,
            type=ComponentType.APPLICATION,
            name=name,
            properties={"version": "1.0"}
        ))
        graph.add_edge(Edge(
            source=app_id,
            target=node,
            edge_type=EdgeType.RUNS_ON
        ))
        
        # Publish relationships
        for topic in pub_topics:
            graph.add_edge(Edge(
                source=app_id,
                target=topic,
                edge_type=EdgeType.PUBLISHES_TO,
                qos=QoSPolicy(reliability="reliable", priority=5)
            ))
        
        # Subscribe relationships
        for topic in sub_topics:
            graph.add_edge(Edge(
                source=app_id,
                target=topic,
                edge_type=EdgeType.SUBSCRIBES_TO,
                qos=QoSPolicy(reliability="reliable")
            ))
    
    # Add node connectivity
    graph.add_edge(Edge(
        source="node_edge",
        target="node_cloud1",
        edge_type=EdgeType.CONNECTS_TO
    ))
    graph.add_edge(Edge(
        source="node_cloud1",
        target="node_cloud2",
        edge_type=EdgeType.CONNECTS_TO
    ))
    
    return graph


# =============================================================================
# Demo Functions
# =============================================================================

def demo_graph_model() -> SimulationGraph:
    """Demo 1: Graph Model Creation and Inspection"""
    print_header("DEMO 1: SIMULATION GRAPH MODEL")
    
    graph = create_demo_graph()
    
    # Display statistics
    stats = graph.get_statistics()
    print(f"Graph Statistics:")
    print(f"  Total Components: {stats['total_components']}")
    print(f"  Total Edges: {stats['total_edges']}")
    
    print(f"\n  By Component Type:")
    for comp_type, count in stats["by_component_type"].items():
        print(f"    {comp_type}: {count}")
    
    print(f"\n  By Edge Type:")
    for edge_type, count in stats["by_edge_type"].items():
        print(f"    {edge_type}: {count}")
    
    # Query examples
    print_subheader("Query Examples")
    
    # Publishers for a topic
    topic = "topic_sensor_data"
    publishers = graph.get_publishers(topic)
    print(f"\n  Publishers of '{topic}':")
    for pub in publishers:
        print(f"    - {pub}")
    
    # Subscribers
    subscribers = graph.get_subscribers(topic)
    print(f"\n  Subscribers of '{topic}':")
    for sub in subscribers:
        print(f"    - {sub}")
    
    # Components on a node
    node = "node_cloud1"
    on_node = graph.get_components_on_node(node)
    print(f"\n  Components on '{node}':")
    for comp in on_node:
        print(f"    - {comp}")
    
    # Downstream from a component
    app = "app_sensor1"
    downstream = graph.get_downstream_components(app)
    print(f"\n  Downstream from '{app}':")
    for comp in sorted(downstream):
        print(f"    - {comp}")
    
    print_success("Graph model demo complete")
    return graph


def demo_single_failure(graph: SimulationGraph) -> FailureResult:
    """Demo 2: Single Component Failure Simulation"""
    print_header("DEMO 2: SINGLE FAILURE SIMULATION")
    
    simulator = FailureSimulator(seed=42, cascade=True)
    
    # Simulate broker failure (high impact expected)
    component = "broker_edge"
    print(f"Simulating CRASH failure of: {component}")
    
    result = simulator.simulate_failure(graph, component, FailureMode.CRASH)
    
    # Display results
    color = impact_color(result.impact_score)
    print(f"\n  Component: {result.failed_component}")
    print(f"  Type: {result.component_type.value}")
    print(f"  Failure Mode: {result.failure_mode.value}")
    print(f"  {BOLD}Impact Score: {color}{result.impact_score:.4f}{RESET}")
    print(f"  Total Affected: {result.total_affected} / {result.total_components}")
    print(f"  Message Paths Broken: {result.message_paths_broken}")
    
    print(f"\n  Directly Affected ({len(result.directly_affected)}):")
    for comp in sorted(result.directly_affected)[:10]:
        print(f"    - {comp}")
    
    if result.cascade_affected:
        print(f"\n  Cascade Affected ({len(result.cascade_affected)}):")
        for comp in sorted(result.cascade_affected)[:5]:
            print(f"    - {comp}")
    
    print(f"\n  Affected by Type:")
    for comp_type, count in result.affected_by_type.items():
        print(f"    {comp_type}: {count}")
    
    print_success("Single failure demo complete")
    return result


def demo_component_type_simulation(graph: SimulationGraph) -> Dict[str, CampaignResult]:
    """Demo 3: Component-Type Specific Simulation"""
    print_header("DEMO 3: COMPONENT-TYPE SPECIFIC SIMULATION")
    
    simulator = FailureSimulator(seed=42, cascade=True, critical_threshold=0.2)
    results = {}
    
    for comp_type in ComponentType:
        components = graph.get_components_by_type(comp_type)
        if not components:
            continue
        
        print_subheader(f"{comp_type.value} Components ({len(components)})")
        
        campaign = simulator.simulate_by_component_type(graph, comp_type)
        results[comp_type.value] = campaign
        
        # Summary
        impacts = [r.impact_score for r in campaign.results]
        avg_impact = sum(impacts) / len(impacts) if impacts else 0
        max_impact = max(impacts) if impacts else 0
        critical_count = len(campaign.critical_components)
        
        print(f"  Average Impact: {avg_impact:.4f}")
        print(f"  Max Impact: {max_impact:.4f}")
        print(f"  Critical Count: {RED}{critical_count}{RESET}")
        
        # Top 3 most impactful
        if campaign.ranked_by_impact:
            print(f"\n  Top 3 by Impact:")
            for comp_id, impact in campaign.ranked_by_impact[:3]:
                color = impact_color(impact)
                print(f"    {color}{comp_id}{RESET}: {impact:.4f}")
    
    print_success("Component-type simulation demo complete")
    return results


def demo_failure_campaign(graph: SimulationGraph) -> CampaignResult:
    """Demo 4: Full Failure Campaign"""
    print_header("DEMO 4: FULL FAILURE CAMPAIGN")
    
    simulator = FailureSimulator(
        seed=42,
        cascade=True,
        cascade_threshold=0.5,
        critical_threshold=0.25
    )
    
    print(f"Testing all {len(graph.components)} components...")
    campaign = simulator.simulate_all_failures(graph)
    
    # Overall statistics
    print(f"\n  Campaign Results:")
    print(f"    Total Tested: {len(campaign.results)}")
    print(f"    Critical Threshold: {campaign.critical_threshold:.0%}")
    print(f"    Critical Components: {RED}{len(campaign.critical_components)}{RESET}")
    
    # Top 10 most critical
    print(f"\n  {BOLD}Top 10 Most Critical Components:{RESET}")
    for i, (comp_id, impact) in enumerate(campaign.ranked_by_impact[:10], 1):
        comp = graph.get_component(comp_id)
        comp_type = comp.type.value if comp else "?"
        color = impact_color(impact)
        print(f"    {i:2}. {color}{comp_id}{RESET} ({comp_type}): {impact:.4f}")
    
    # Distribution analysis
    print(f"\n  Impact Distribution:")
    ranges = [
        (0.0, 0.1, "Minimal"),
        (0.1, 0.25, "Low"),
        (0.25, 0.5, "Medium"),
        (0.5, 0.75, "High"),
        (0.75, 1.0, "Critical"),
    ]
    for low, high, label in ranges:
        count = sum(1 for r in campaign.results if low <= r.impact_score < high)
        if count > 0:
            print(f"    {label} ({low:.0%}-{high:.0%}): {count}")
    
    print_success("Failure campaign demo complete")
    return campaign


def demo_event_simulation(graph: SimulationGraph) -> SimulationResult:
    """Demo 5: Event-Driven Simulation"""
    print_header("DEMO 5: EVENT-DRIVEN SIMULATION")
    
    simulator = EventSimulator(
        graph,
        seed=42,
        default_latency=10.0,
        default_processing_time=1.0,
        queue_capacity=500,
        timeout=5000.0
    )
    
    # Run simulation
    duration = 2000.0  # 2 seconds simulated time
    message_rate = 20.0  # 20 messages per ms
    
    print(f"Running event simulation:")
    print(f"  Duration: {duration}ms")
    print(f"  Message Rate: {message_rate} msg/ms")
    
    result = simulator.run(duration=duration, message_rate=message_rate)
    stats = result.statistics
    
    # Message statistics
    print(f"\n  {BOLD}Message Statistics:{RESET}")
    print(f"    Total Generated: {stats.total_messages}")
    print(f"    Delivered: {GREEN}{stats.delivered_messages}{RESET}")
    print(f"    Failed: {RED}{stats.failed_messages}{RESET}")
    print(f"    Dropped: {YELLOW}{stats.dropped_messages}{RESET}")
    print(f"    Timeout: {YELLOW}{stats.timeout_messages}{RESET}")
    print(f"    Delivery Rate: {stats.delivery_rate:.2%}")
    
    # Performance
    print(f"\n  {BOLD}Performance Metrics:{RESET}")
    print(f"    Average Latency: {stats.average_latency:.2f} ms")
    print(f"    P99 Latency: {stats.p99_latency:.2f} ms")
    print(f"    Throughput: {stats.average_throughput:.2f} msg/s")
    print(f"    Total Events Processed: {stats.total_events}")
    print(f"    Wall Clock Time: {stats.wall_clock_time:.4f}s")
    print(f"    Speedup: {duration / 1000 / stats.wall_clock_time:.1f}x real-time")
    
    # Component load analysis
    print(f"\n  {BOLD}Component Load (Top 5 by Messages):{RESET}")
    sorted_loads = sorted(
        result.component_loads.values(),
        key=lambda x: x.messages_processed,
        reverse=True
    )[:5]
    
    for load in sorted_loads:
        if load.messages_processed > 0:
            print(f"    {load.component_id}:")
            print(f"      Processed: {load.messages_processed}, "
                  f"Dropped: {load.messages_dropped}, "
                  f"Avg Latency: {load.average_latency:.2f}ms")
    
    print_success("Event simulation demo complete")
    return result


def demo_stress_test(graph: SimulationGraph) -> SimulationResult:
    """Demo 6: Stress Test with Failures"""
    print_header("DEMO 6: STRESS TEST WITH FAILURES")
    
    simulator = EventSimulator(
        graph,
        seed=42,
        default_latency=5.0,
        queue_capacity=200,  # Smaller queue to induce drops
    )
    
    # Inject failures
    print(f"Injecting failures during simulation:")
    failures = [
        ("broker_edge", 500, 200),    # Fail at 500ms, recover after 200ms
        ("node_cloud1", 1000, None),  # Fail at 1000ms, permanent
    ]
    
    for comp_id, fail_time, duration in failures:
        simulator.inject_failure(comp_id, fail_time, duration)
        dur_str = f"{duration}ms" if duration else "permanent"
        print(f"  - {comp_id} at {fail_time}ms ({dur_str})")
    
    # Run stress test
    result = simulator.run(duration=3000, message_rate=50.0)
    stats = result.statistics
    
    # Results
    print(f"\n  {BOLD}Stress Test Results:{RESET}")
    print(f"    Total Messages: {stats.total_messages}")
    print(f"    Delivery Rate: {stats.delivery_rate:.2%}")
    print(f"    Failed: {RED}{stats.failed_messages}{RESET}")
    print(f"    Dropped: {YELLOW}{stats.dropped_messages}{RESET}")
    
    # Bottlenecks
    bottlenecks = result.get_bottlenecks(threshold=50)
    if bottlenecks:
        print(f"\n  {YELLOW}Bottlenecks Detected:{RESET}")
        for comp_id in bottlenecks:
            load = result.component_loads[comp_id]
            print(f"    - {comp_id}: peak queue = {load.peak_queue_size}")
    
    # High drop rate components
    critical = result.get_critical_components(threshold=0.05)
    if critical:
        print(f"\n  {RED}High Drop Rate Components:{RESET}")
        for comp_id in critical:
            load = result.component_loads[comp_id]
            print(f"    - {comp_id}: drop rate = {load.drop_rate:.2%}")
    
    # Failed components
    if result.failed_components:
        print(f"\n  {RED}Failed Components at End:{RESET}")
        for comp_id in result.failed_components:
            print(f"    - {comp_id}")
    
    print_success("Stress test demo complete")
    return result


def demo_json_export(graph: SimulationGraph) -> Dict[str, Any]:
    """Demo 7: JSON Export"""
    print_header("DEMO 7: JSON EXPORT")
    
    # Run simulations
    failure_sim = FailureSimulator(seed=42)
    campaign = failure_sim.simulate_all_failures(graph)
    
    event_sim = EventSimulator(graph, seed=42)
    event_result = event_sim.run(duration=1000, message_rate=10.0)
    
    # Build export data
    export_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "graph_stats": graph.get_statistics(),
        },
        "failure_campaign": {
            "total_tested": len(campaign.results),
            "critical_count": len(campaign.critical_components),
            "critical_threshold": campaign.critical_threshold,
            "ranked_by_impact": campaign.ranked_by_impact[:10],
            "by_type_summary": {},
        },
        "event_simulation": event_result.statistics.to_dict(),
    }
    
    # Add by-type summary
    for comp_type in ComponentType:
        type_results = campaign.get_by_type(comp_type)
        if type_results:
            impacts = [r.impact_score for r in type_results]
            export_data["failure_campaign"]["by_type_summary"][comp_type.value] = {
                "count": len(type_results),
                "avg_impact": round(sum(impacts) / len(impacts), 4),
                "max_impact": round(max(impacts), 4),
            }
    
    # Display preview
    print(f"Export Data Preview:")
    print(f"  Timestamp: {export_data['metadata']['timestamp']}")
    print(f"  Graph Components: {export_data['metadata']['graph_stats']['total_components']}")
    print(f"  Critical Components: {export_data['failure_campaign']['critical_count']}")
    print(f"  Event Delivery Rate: {export_data['event_simulation']['delivery_rate']:.2%}")
    
    print_success("JSON export demo complete")
    return export_data


# =============================================================================
# Live Mode (Neo4j)
# =============================================================================

def run_live_mode(args) -> Optional[Dict[str, Any]]:
    """Run with Neo4j database"""
    print_header("LIVE MODE - NEO4J CONNECTION")
    
    try:
        from src.simulation import (
            Neo4jSimulationClient,
            load_graph_from_neo4j,
        )
    except ImportError as e:
        print(f"{RED}Error: Neo4j driver not installed.{RESET}")
        print(f"Install with: pip install neo4j")
        return None
    
    print(f"Connecting to Neo4j at {args.uri}...")
    
    try:
        with Neo4jSimulationClient(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database
        ) as client:
            # Verify connection
            if not client.verify_connection():
                print(f"{RED}Failed to connect to Neo4j{RESET}")
                return None
            
            print_success("Connected to Neo4j")
            
            # Get stats
            stats = client.get_graph_stats()
            print(f"\n  Database Statistics:")
            print(f"    Components: {stats['total_components']}")
            print(f"    Edges: {stats['total_edges']}")
            
            if stats['total_components'] == 0:
                print(f"\n{YELLOW}Database is empty. Running demo mode instead.{RESET}")
                return run_demo_mode(args)
            
            # Load graph
            print(f"\n  Loading graph...")
            graph = client.load_full_graph()
            
            # Run simulations
            results = {}
            
            # Failure campaign
            print_subheader("Failure Campaign")
            simulator = FailureSimulator(seed=args.seed)
            campaign = simulator.simulate_all_failures(graph)
            
            print(f"  Tested: {len(campaign.results)} components")
            print(f"  Critical: {len(campaign.critical_components)}")
            
            results["campaign"] = campaign.to_dict()
            
            # Event simulation
            print_subheader("Event Simulation")
            event_sim = EventSimulator(graph, seed=args.seed)
            event_result = event_sim.run(duration=1000, message_rate=20.0)
            
            print(f"  Delivery Rate: {event_result.statistics.delivery_rate:.2%}")
            print(f"  Avg Latency: {event_result.statistics.average_latency:.2f}ms")
            
            results["event_simulation"] = event_result.to_dict()
            
            return results
            
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Demo Mode
# =============================================================================

def run_demo_mode(args) -> Dict[str, Any]:
    """Run all demos without Neo4j"""
    results = {}
    
    # Demo 1: Graph Model
    graph = demo_graph_model()
    results["graph_stats"] = graph.get_statistics()
    
    # Demo 2: Single Failure
    single_result = demo_single_failure(graph)
    results["single_failure"] = single_result.to_dict()
    
    # Demo 3: Component-Type Simulation
    type_results = demo_component_type_simulation(graph)
    results["by_component_type"] = {
        k: {
            "count": len(v.results),
            "critical_count": len(v.critical_components),
            "top_3": v.ranked_by_impact[:3]
        }
        for k, v in type_results.items()
    }
    
    # Demo 4: Full Campaign
    campaign = demo_failure_campaign(graph)
    results["campaign"] = {
        "total": len(campaign.results),
        "critical": len(campaign.critical_components),
        "top_10": campaign.ranked_by_impact[:10]
    }
    
    # Demo 5: Event Simulation
    event_result = demo_event_simulation(graph)
    results["event_simulation"] = event_result.statistics.to_dict()
    
    # Demo 6: Stress Test
    stress_result = demo_stress_test(graph)
    results["stress_test"] = stress_result.statistics.to_dict()
    
    # Demo 7: JSON Export
    export_data = demo_json_export(graph)
    results["export_preview"] = export_data
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulation module examples and demonstrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all demos
  python examples/example_simulation.py --demo
  
  # Run with Neo4j
  python examples/example_simulation.py --uri bolt://localhost:7687 --password secret
  
  # Export results
  python examples/example_simulation.py --demo --output results.json
        """
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode (no Neo4j required)"
    )
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI"
    )
    parser.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--password",
        default="password",
        help="Neo4j password"
    )
    parser.add_argument(
        "--database",
        default="neo4j",
        help="Neo4j database name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    print_header("SOFTWARE-AS-A-GRAPH SIMULATION EXAMPLES")
    print(f"Version: 5.0")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run appropriate mode
    if args.demo:
        results = run_demo_mode(args)
    else:
        results = run_live_mode(args)
        if results is None:
            print(f"\n{YELLOW}Falling back to demo mode...{RESET}")
            results = run_demo_mode(args)
    
    # Export if requested
    if args.output and results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{GREEN}✓ Results saved to: {output_path}{RESET}")
    
    # Summary
    print_header("EXAMPLES COMPLETE")
    print(f"{GREEN}All simulation examples completed successfully!{RESET}")
    print(f"\nKey Capabilities Demonstrated:")
    print(f"  ✓ Multi-layer graph model with original edge types")
    print(f"  ✓ Component-type specific simulation")
    print(f"  ✓ Failure simulation with cascade propagation")
    print(f"  ✓ Event-driven discrete event simulation")
    print(f"  ✓ Stress testing with injected failures")
    print(f"  ✓ JSON export for integration")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
