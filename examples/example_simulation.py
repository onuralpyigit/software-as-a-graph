#!/usr/bin/env python3
"""
Simulation Example - Version 5.0

Demonstrates the simulation module capabilities:
1. SimulationGraph creation and manipulation
2. Failure simulation with cascade
3. Event-driven simulation
4. Layer-specific reporting

This example runs in demo mode (no external dependencies).

Usage:
    python examples/example_simulation.py
    python examples/example_simulation.py --verbose

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Terminal Output
# =============================================================================

BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"
RESET = "\033[0m"


def print_header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}{title:^60}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def print_section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print(f"{'-' * 40}")


def print_success(msg: str) -> None:
    print(f"{GREEN}✓{RESET} {msg}")


def print_info(msg: str) -> None:
    print(f"{BLUE}ℹ{RESET} {msg}")


def print_code(code: str) -> None:
    for line in code.strip().split("\n"):
        print(f"  {GRAY}{line}{RESET}")


def impact_color(impact: float) -> str:
    if impact >= 0.5:
        return RED
    elif impact >= 0.3:
        return YELLOW
    elif impact >= 0.1:
        return BLUE
    return GRAY


# =============================================================================
# Demo Functions
# =============================================================================

def demo_simulation_graph():
    """Demo 1: SimulationGraph creation and queries."""
    print_header("DEMO 1: SIMULATION GRAPH")
    
    from src.simulation import (
        SimulationGraph,
        create_simulation_graph,
        ComponentType,
        EdgeType,
    )
    
    print("Creating a simulation graph with original edge types...")
    print()
    print("Key Design: Uses ORIGINAL edges (PUBLISHES_TO, SUBSCRIBES_TO,")
    print("            ROUTES, RUNS_ON, CONNECTS_TO) NOT DEPENDS_ON")
    print()
    
    # Create graph
    graph = create_simulation_graph(
        applications=6,
        brokers=2,
        topics=8,
        nodes=3,
        seed=42,
    )
    
    summary = graph.summary()
    print(f"  {BOLD}Graph Summary:{RESET}")
    print(f"    Components: {summary['total_components']}")
    print(f"    Edges: {summary['total_edges']}")
    print(f"    Message Paths: {summary['message_paths']}")
    
    print(f"\n  {BOLD}By Component Type:{RESET}")
    for comp_type, count in summary['by_component_type'].items():
        print(f"    {comp_type}: {count}")
    
    print(f"\n  {BOLD}By Edge Type:{RESET}")
    for edge_type, count in summary['by_edge_type'].items():
        print(f"    {edge_type}: {count}")
    
    # Query examples
    print_section("Query Examples")
    
    # Get publishers for a topic
    topics = list(graph._by_type[ComponentType.TOPIC])
    if topics:
        topic = topics[0]
        publishers = graph.get_publishers(topic)
        subscribers = graph.get_subscribers(topic)
        broker = graph.get_broker_for_topic(topic)
        
        print(f"\n  Topic: {topic}")
        print(f"    Publishers: {list(publishers)[:3]}")
        print(f"    Subscribers: {list(subscribers)[:3]}")
        print(f"    Routed by: {broker}")
    
    # Get message paths
    paths = graph.get_message_paths()
    print(f"\n  Sample Message Paths (publisher -> topic -> subscriber):")
    for pub, topic, sub in paths[:5]:
        print(f"    {pub} -> {topic} -> {sub}")
    
    print_success("Graph created and queried successfully")
    return graph


def demo_layer_extraction(graph):
    """Demo 2: Layer extraction."""
    print_header("DEMO 2: LAYER EXTRACTION")
    
    print("The system can be analyzed by layer:")
    print()
    
    for layer_key, layer_def in graph.LAYER_DEFINITIONS.items():
        print(f"  {BOLD}{layer_key:15}{RESET}: {layer_def['name']}")
        print(f"    Components: {', '.join(t.value for t in layer_def['component_types'])}")
        print(f"    Edge Types: {', '.join(t.value for t in layer_def['edge_types'])}")
        print()
    
    # Extract application layer
    print_section("Application Layer Subgraph")
    
    app_layer = graph.get_layer_subgraph("application")
    stats = app_layer.summary()
    
    print(f"  Components: {stats['total_components']}")
    print(f"  Edges: {stats['total_edges']}")
    print(f"  Message Paths: {stats['message_paths']}")
    
    print_success("Layer extraction demonstrated")


def demo_failure_simulation(graph):
    """Demo 3: Failure simulation."""
    print_header("DEMO 3: FAILURE SIMULATION")
    
    from src.simulation import FailureSimulator, FailureMode, ComponentType
    
    print("Failure simulation measures impact by:")
    print("  - Reachability Loss: Message paths broken")
    print("  - Cascade Extent: Components failing due to dependencies")
    print()
    
    simulator = FailureSimulator(
        seed=42,
        cascade=True,
        cascade_threshold=0.5,
    )
    
    # Single failure - broker
    print_section("Single Failure: Broker")
    
    brokers = list(graph._by_type[ComponentType.BROKER])
    if brokers:
        broker_id = brokers[0]
        result = simulator.simulate_failure(graph, broker_id)
        
        color = impact_color(result.impact_score)
        print(f"\n  Component: {broker_id}")
        print(f"  Impact Score: {color}{result.impact_score:.4f}{RESET}")
        print(f"  Reachability Loss: {result.reachability_loss:.2%}")
        print(f"  Paths Broken: {result.paths_broken}/{result.total_paths}")
        print(f"  Directly Affected: {len(result.directly_affected)}")
        print(f"  Cascade Affected: {len(result.cascade_affected)}")
    
    # Single failure - application
    print_section("Single Failure: Application")
    
    apps = list(graph._by_type[ComponentType.APPLICATION])
    if apps:
        app_id = apps[0]
        result = simulator.simulate_failure(graph, app_id)
        
        color = impact_color(result.impact_score)
        print(f"\n  Component: {app_id}")
        print(f"  Impact Score: {color}{result.impact_score:.4f}{RESET}")
        print(f"  Reachability Loss: {result.reachability_loss:.2%}")
        print(f"  Paths Broken: {result.paths_broken}/{result.total_paths}")
    
    # Compare failure modes
    print_section("Failure Mode Comparison")
    
    if brokers:
        broker_id = brokers[0]
        
        for mode in [FailureMode.CRASH, FailureMode.DEGRADED, FailureMode.PARTITION]:
            result = simulator.simulate_failure(graph, broker_id, mode)
            color = impact_color(result.impact_score)
            print(f"  {mode.value:12}: {color}{result.impact_score:.4f}{RESET} "
                  f"(factor: {mode.impact_factor})")
    
    print_success("Failure simulation demonstrated")
    return simulator


def demo_failure_campaign(graph, simulator):
    """Demo 4: Failure campaign."""
    print_header("DEMO 4: FAILURE CAMPAIGN")
    
    print("Simulating failure of EVERY component...")
    print()
    
    campaign = simulator.simulate_all(graph)
    
    print(f"  Total Simulations: {campaign.total_simulations}")
    print(f"  Duration: {campaign.duration_ms:.0f}ms")
    print(f"  Critical (>0.3): {len(campaign.get_critical())}")
    
    # Top impacts
    print_section("Top 10 Impact")
    
    for i, (comp_id, impact) in enumerate(campaign.ranked_by_impact()[:10], 1):
        comp = graph.get_component(comp_id)
        comp_type = comp.type.value if comp else "?"
        color = impact_color(impact)
        print(f"  {i:2}. {color}{comp_id:20}{RESET} ({comp_type:11}): {impact:.4f}")
    
    # By layer
    print_section("Results by Layer")
    
    for layer_key, layer_result in campaign.by_layer.items():
        if layer_result.count > 0:
            print(f"\n  {BOLD}{layer_result.layer_name}{RESET}")
            print(f"    Components: {layer_result.count}")
            print(f"    Avg Impact: {layer_result.avg_impact:.4f}")
            print(f"    Max Impact: {layer_result.max_impact:.4f}")
            print(f"    Critical: {len(layer_result.get_critical())}")
    
    # By type
    print_section("Results by Component Type")
    
    for comp_type, type_results in campaign.by_type.items():
        if type_results:
            impacts = [r.impact_score for r in type_results]
            print(f"\n  {BOLD}{comp_type.value}{RESET}")
            print(f"    Count: {len(type_results)}")
            print(f"    Avg Impact: {sum(impacts)/len(impacts):.4f}")
            print(f"    Max Impact: {max(impacts):.4f}")
    
    print_success("Failure campaign complete")
    return campaign


def demo_event_simulation(graph):
    """Demo 5: Event-driven simulation."""
    print_header("DEMO 5: EVENT-DRIVEN SIMULATION")
    
    from src.simulation import EventSimulator
    
    print("Event simulation models message flow through the system:")
    print("  - Message generation with configurable rate")
    print("  - Routing through brokers")
    print("  - Delivery to subscribers")
    print("  - Latency and queue modeling")
    print()
    
    simulator = EventSimulator(
        seed=42,
        default_latency=10.0,   # 10ms network latency
        queue_capacity=500,     # Queue capacity per component
        timeout=5000.0,         # 5s timeout
    )
    
    # Run simulation
    duration = 1000.0   # 1 second
    message_rate = 50.0  # 50 msg/sec
    
    print(f"Running simulation:")
    print(f"  Duration: {duration}ms")
    print(f"  Message Rate: {message_rate}/sec")
    
    result = simulator.run(graph, duration=duration, message_rate=message_rate)
    stats = result.stats
    
    # Message statistics
    print_section("Message Statistics")
    
    print(f"  Total Generated: {stats.total_messages}")
    print(f"  Delivered: {GREEN}{stats.delivered_messages}{RESET}")
    print(f"  Failed: {RED}{stats.failed_messages}{RESET}")
    print(f"  Dropped: {YELLOW}{stats.dropped_messages}{RESET}")
    print(f"  Timeout: {YELLOW}{stats.timeout_messages}{RESET}")
    print(f"  Delivery Rate: {stats.delivery_rate:.2%}")
    
    # Performance
    print_section("Performance Metrics")
    
    print(f"  Avg Latency: {stats.avg_latency:.2f}ms")
    print(f"  P99 Latency: {stats.p99_latency:.2f}ms")
    print(f"  Events Processed: {stats.total_events}")
    print(f"  Wall Clock Time: {stats.wall_clock_time:.3f}s")
    
    # Layer statistics
    print_section("Layer Statistics")
    
    for layer_key, layer_stats in result.layer_stats.items():
        if layer_stats.messages_sent > 0:
            print(f"\n  {BOLD}{layer_stats.layer_name}{RESET}")
            print(f"    Messages Sent: {layer_stats.messages_sent}")
            print(f"    Delivered: {layer_stats.messages_delivered}")
            print(f"    Delivery Rate: {layer_stats.delivery_rate:.2%}")
            if layer_stats.avg_latency > 0:
                print(f"    Avg Latency: {layer_stats.avg_latency:.2f}ms")
    
    # Component loads
    print_section("Component Loads")
    
    # Show top loaded components
    loads = sorted(
        result.component_loads.values(),
        key=lambda l: l.messages_processed,
        reverse=True
    )[:5]
    
    for load in loads:
        if load.messages_processed > 0:
            print(f"  {load.component_id} ({load.component_type.value}):")
            print(f"    Processed: {load.messages_processed}")
            print(f"    Peak Queue: {load.peak_queue_size}")
            if load.drop_rate > 0:
                print(f"    Drop Rate: {RED}{load.drop_rate:.2%}{RESET}")
    
    print_success("Event simulation complete")
    return result


def demo_failure_injection(graph):
    """Demo 6: Event simulation with failure injection."""
    print_header("DEMO 6: FAILURE INJECTION")
    
    from src.simulation import EventSimulator, ComponentType
    
    print("Inject failures during event simulation to test resilience...")
    print()
    
    simulator = EventSimulator(seed=42)
    
    # Find a broker to fail
    brokers = list(graph._by_type[ComponentType.BROKER])
    if not brokers:
        print_info("No brokers in graph to fail")
        return
    
    broker_id = brokers[0]
    
    # Run with failure injection
    failure_schedule = [
        {
            "component_id": broker_id,
            "time": 300.0,      # Fail at 300ms
            "duration": 200.0,  # Recover after 200ms
        }
    ]
    
    print(f"Failure Schedule:")
    print(f"  Component: {broker_id}")
    print(f"  Fail at: 300ms")
    print(f"  Recovery at: 500ms")
    print()
    
    # Run without failure (baseline)
    result_normal = simulator.run(graph, duration=1000.0, message_rate=50.0)
    
    # Reset and run with failure
    result_failure = simulator.run(
        graph,
        duration=1000.0,
        message_rate=50.0,
        failure_schedule=failure_schedule,
    )
    
    # Compare
    print_section("Comparison")
    
    print(f"  {'Metric':<25} {'Normal':>12} {'With Failure':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Delivery Rate':<25} {result_normal.stats.delivery_rate:>11.2%} "
          f"{result_failure.stats.delivery_rate:>11.2%}")
    print(f"  {'Failed Messages':<25} {result_normal.stats.failed_messages:>12} "
          f"{result_failure.stats.failed_messages:>12}")
    print(f"  {'Avg Latency':<25} {result_normal.stats.avg_latency:>10.2f}ms "
          f"{result_failure.stats.avg_latency:>10.2f}ms")
    
    print_success("Failure injection demonstrated")


def demo_cli_usage():
    """Demo 7: CLI usage examples."""
    print_header("DEMO 7: CLI USAGE")
    
    print("The simulate_graph.py CLI provides these commands:")
    print()
    
    examples = [
        ("Single component failure",
         "python simulate_graph.py --input graph.json --failure --component broker_1"),
        ("Full failure campaign",
         "python simulate_graph.py --input graph.json --failure --all"),
        ("Type-specific failure",
         "python simulate_graph.py --input graph.json --failure --type Broker"),
        ("Event simulation",
         "python simulate_graph.py --input graph.json --event --duration 5000"),
        ("With failure injection",
         "python simulate_graph.py --input graph.json --event --inject-failure broker_1"),
        ("Export results",
         "python simulate_graph.py --input graph.json --failure --all --output results.json"),
    ]
    
    for desc, cmd in examples:
        print(f"  {desc}:")
        print_code(cmd)
        print()
    
    print_success("See --help for all options")


def demo_neo4j_integration():
    """Demo 8: Neo4j Integration"""
    print_header("DEMO 8: NEO4J INTEGRATION")
    
    from src.simulation import check_neo4j_available
    
    print("The simulation module can load graphs directly from Neo4j.")
    print()
    
    # Check if Neo4j is available
    if not check_neo4j_available():
        print(f"  {YELLOW}Neo4j driver not installed{RESET}")
        print(f"  Install with: pip install neo4j")
        print()
        print_neo4j_cli_examples()
        print_success("Neo4j integration available when driver is installed")
        return
    
    print(f"  {GREEN}✓{RESET} Neo4j driver installed")
    
    # Try to connect
    try:
        from src.simulation import Neo4jSimulationClient
        
        print(f"\n  Testing connection to localhost:7687...")
        
        with Neo4jSimulationClient() as client:
            if client.verify_connection():
                print(f"  {GREEN}✓{RESET} Connected to Neo4j")
                
                # Get statistics
                stats = client.get_statistics()
                print(f"\n  {BOLD}Database Statistics{RESET}")
                print(f"    Components: {stats['total_components']}")
                print(f"    Edges: {stats['total_edges']}")
                
                if stats['total_components'] > 0:
                    # Load and display graph
                    graph = client.load_full_graph()
                    summary = graph.summary()
                    
                    print(f"\n  {BOLD}Loaded Graph{RESET}")
                    print(f"    Components: {summary['total_components']}")
                    print(f"    Edges: {summary['total_edges']}")
                    print(f"    Message Paths: {summary['message_paths']}")
                    
                    # Show layer statistics
                    print(f"\n  {BOLD}Layer Statistics{RESET}")
                    layer_stats = client.get_layer_statistics()
                    for layer, lstats in layer_stats.items():
                        print(f"    {layer}: {lstats['components']} components, "
                              f"{lstats['edges']} edges")
                else:
                    print(f"\n  {YELLOW}Database is empty{RESET}")
            else:
                print(f"  {RED}✗{RESET} Could not connect to Neo4j")
                print(f"    Make sure Neo4j is running on localhost:7687")
    
    except Exception as e:
        print(f"  {YELLOW}⚠{RESET} Connection failed: {e}")
        print(f"    Make sure Neo4j is running")
    
    print()
    print_neo4j_cli_examples()
    print_success("Neo4j integration demonstrated")


def print_neo4j_cli_examples():
    """Print Neo4j CLI examples."""
    print(f"\n  {BOLD}Neo4j CLI Examples{RESET}")
    print()
    
    neo4j_examples = [
        ("Load from Neo4j",
         "python simulate_graph.py --neo4j --failure --all"),
        ("Custom connection",
         "python simulate_graph.py --neo4j --uri bolt://host:7687 --user neo4j --password pass --failure --all"),
        ("Load specific layer",
         "python simulate_graph.py --neo4j --layer application --failure --all"),
        ("Event simulation from Neo4j",
         "python simulate_graph.py --neo4j --event --duration 5000"),
    ]
    
    for desc, cmd in neo4j_examples:
        print(f"  {desc}:")
        print_code(cmd)
        print()


def demo_neo4j_python_api():
    """Demo 9: Neo4j Python API"""
    print_header("DEMO 9: NEO4J PYTHON API")
    
    print("Using Neo4j in Python code:")
    print()
    
    code = '''
from src.simulation import (
    Neo4jSimulationClient,
    load_graph_from_neo4j,
    FailureSimulator,
)

# Method 1: Using context manager
with Neo4jSimulationClient(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
) as client:
    # Load full graph
    graph = client.load_full_graph()
    
    # Or load specific layer
    app_graph = client.load_layer("application")
    
    # Get statistics
    stats = client.get_statistics()
    print(f"Components: {stats['total_components']}")

# Method 2: Using factory function
graph = load_graph_from_neo4j(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    layer="app_broker",  # Optional: load specific layer
)

# Run simulation on loaded graph
simulator = FailureSimulator(cascade=True)
campaign = simulator.simulate_all(graph)

for comp_id, impact in campaign.ranked_by_impact()[:5]:
    print(f"{comp_id}: {impact:.4f}")
'''
    
    print_code(code)
    print_success("Neo4j Python API demonstrated")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulation Module Demo"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print_header("Simulation Module Demo")
    
    # Run demos
    graph = demo_simulation_graph()
    demo_layer_extraction(graph)
    simulator = demo_failure_simulation(graph)
    demo_failure_campaign(graph, simulator)
    demo_event_simulation(graph)
    demo_failure_injection(graph)
    demo_cli_usage()
    demo_neo4j_integration()
    demo_neo4j_python_api()
    
    print_header("Demo Complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())