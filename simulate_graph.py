#!/usr/bin/env python3
"""
Simulate Graph CLI - Version 5.0

Command-line interface for graph-based simulation of pub-sub systems.

Input Sources:
- JSON file (--input)
- Neo4j database (--neo4j)

Simulation Types:
1. Failure Simulation: Test system resilience by failing components
2. Event Simulation: Simulate message flow with discrete events

Features:
- Component-type specific simulation
- Cascade failure propagation
- Layer-specific reporting
- QoS-aware message simulation

Usage:
    # Load from file
    python simulate_graph.py --input graph.json --failure --all
    
    # Load from Neo4j
    python simulate_graph.py --neo4j --failure --all
    python simulate_graph.py --neo4j --uri bolt://localhost:7687 --failure --all
    
    # Load specific layer from Neo4j
    python simulate_graph.py --neo4j --layer application --failure --all
    
    # Event simulation
    python simulate_graph.py --input graph.json --event --duration 5000

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    
    @classmethod
    def disable(cls):
        for attr in ["BOLD", "GREEN", "BLUE", "CYAN", "YELLOW", "RED", "GRAY", "RESET"]:
            setattr(cls, attr, "")


def use_colors() -> bool:
    import os
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.name != "nt"


# =============================================================================
# Output Helpers
# =============================================================================

def print_header(title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def print_section(title: str) -> None:
    print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{'-' * 40}")


def print_success(msg: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_error(msg: str) -> None:
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_info(msg: str) -> None:
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {msg}")


def print_kv(key: str, value: Any, indent: int = 2) -> None:
    print(f"{' ' * indent}{key}: {value}")


def impact_color(impact: float) -> str:
    if impact >= 0.5:
        return Colors.RED
    elif impact >= 0.3:
        return Colors.YELLOW
    elif impact >= 0.1:
        return Colors.BLUE
    return Colors.GRAY


# =============================================================================
# Graph Loading
# =============================================================================

def load_graph_from_file(args):
    """Load simulation graph from JSON file."""
    from src.simulation import SimulationGraph
    
    if not args.input:
        return None
    
    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"File not found: {input_path}")
        return None
    
    print_info(f"Loading graph from {input_path}")
    
    try:
        graph = SimulationGraph.from_json(str(input_path))
        summary = graph.summary()
        
        print_success(f"Loaded {summary['total_components']} components, {summary['total_edges']} edges")
        print_kv("Message paths", summary['message_paths'])
        
        return graph
    except Exception as e:
        print_error(f"Failed to load graph: {e}")
        return None


def load_graph_from_neo4j(args):
    """Load simulation graph from Neo4j database."""
    from src.simulation import Neo4jSimulationClient, check_neo4j_available
    
    if not check_neo4j_available():
        print_error("Neo4j driver not installed. Install with: pip install neo4j")
        return None
    
    print_info(f"Connecting to Neo4j at {args.uri}")
    
    try:
        with Neo4jSimulationClient(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        ) as client:
            # Verify connection
            if not client.verify_connection():
                print_error("Failed to connect to Neo4j")
                return None
            
            print_success("Connected to Neo4j")
            
            # Get statistics
            stats = client.get_statistics()
            print_kv("Components in DB", stats['total_components'])
            print_kv("Edges in DB", stats['total_edges'])
            
            # Load graph
            if args.layer:
                print_info(f"Loading layer: {args.layer}")
                graph = client.load_layer(args.layer)
            else:
                print_info("Loading full graph")
                graph = client.load_full_graph()
            
            summary = graph.summary()
            print_success(f"Loaded {summary['total_components']} components, {summary['total_edges']} edges")
            print_kv("Message paths", summary['message_paths'])
            
            return graph
            
    except Exception as e:
        print_error(f"Failed to load from Neo4j: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


def load_graph(args):
    """Load graph from file or Neo4j."""
    if args.neo4j:
        return load_graph_from_neo4j(args)
    elif args.input:
        return load_graph_from_file(args)
    else:
        print_error("Specify --input <file> or --neo4j")
        return None


# =============================================================================
# Failure Simulation
# =============================================================================

def run_failure_simulation(args, graph) -> Optional[Dict[str, Any]]:
    """Run failure simulation."""
    from src.simulation import FailureSimulator, FailureMode, ComponentType
    
    print_header("FAILURE SIMULATION")
    
    # Configure simulator
    simulator = FailureSimulator(
        seed=args.seed,
        cascade=args.cascade,
        cascade_threshold=args.cascade_threshold,
        critical_threshold=args.critical_threshold,
    )
    
    print_kv("Cascade", "enabled" if args.cascade else "disabled")
    print_kv("Cascade Threshold", args.cascade_threshold)
    
    # Get failure mode
    failure_mode = FailureMode(args.failure_mode)
    
    results = {}
    
    # Single component failure
    if args.component:
        print_section(f"Single Failure: {args.component}")
        
        try:
            result = simulator.simulate_failure(graph, args.component, failure_mode)
            
            color = impact_color(result.impact_score)
            print(f"\n  {Colors.BOLD}Impact Score:{Colors.RESET} {color}{result.impact_score:.4f}{Colors.RESET}")
            print_kv("Reachability Loss", f"{result.reachability_loss:.2%}")
            print_kv("Paths Broken", f"{result.paths_broken}/{result.total_paths}")
            print_kv("Directly Affected", len(result.directly_affected))
            
            if args.cascade:
                print_kv("Cascade Affected", len(result.cascade_affected))
            
            results = result.to_dict()
            
        except ValueError as e:
            print_error(str(e))
            return None
    
    # Type-specific failures
    elif args.type:
        comp_type = ComponentType(args.type)
        print_section(f"Failures by Type: {args.type}")
        
        campaign = simulator.simulate_type(graph, comp_type, failure_mode)
        
        print_kv("Simulations", campaign.total_simulations)
        print_kv("Duration", f"{campaign.duration_ms:.0f}ms")
        
        print_section("Results by Impact")
        
        for comp_id, impact in campaign.ranked_by_impact()[:15]:
            color = impact_color(impact)
            print(f"  {color}{comp_id:20s}{Colors.RESET}: {impact:.4f}")
        
        results = campaign.to_dict()
    
    # All component failures
    elif args.all:
        print_section("Full Failure Campaign")
        
        campaign = simulator.simulate_all(graph, None, failure_mode)
        
        print_kv("Total Simulations", campaign.total_simulations)
        print_kv("Duration", f"{campaign.duration_ms:.0f}ms")
        print_kv("Critical Components", len(campaign.get_critical()))
        
        # Top 15 by impact
        print_section("Top 15 by Impact")
        
        for i, (comp_id, impact) in enumerate(campaign.ranked_by_impact()[:15], 1):
            comp = graph.get_component(comp_id)
            comp_type = comp.type.value if comp else "?"
            color = impact_color(impact)
            print(f"  {i:2d}. {color}{comp_id:20s}{Colors.RESET} ({comp_type:11s}): {impact:.4f}")
        
        # Layer results
        if campaign.by_layer:
            print_section("Results by Layer")
            
            print(f"\n  {'Layer':<25} {'Count':<8} {'Avg':<10} {'Max':<10} {'Critical'}")
            print(f"  {'-' * 65}")
            
            for layer_key, layer_result in sorted(campaign.by_layer.items()):
                print(f"  {layer_result.layer_name:<25} "
                      f"{layer_result.count:<8} "
                      f"{layer_result.avg_impact:<10.4f} "
                      f"{layer_result.max_impact:<10.4f} "
                      f"{len(layer_result.get_critical())}")
        
        # Type results
        print_section("Results by Component Type")
        print(f"\n  {'Component Type':<15} {'Count':<8} {'Avg':<10} {'Max':<10} {'Critical'}")
        print(f"  {'-' * 65}")
        
        for comp_type, results in campaign.by_type.items():
            if not results:
                continue
            
            avg_impact = sum(r.impact_score for r in results) / len(results)
            max_impact = max(r.impact_score for r in results)
            
            print(f"  {comp_type.value:<15} "
                  f"{len(results):<8} "
                  f"{avg_impact:<10.4f} "
                  f"{max_impact:<10.4f} "
                  f"{len(campaign.get_critical(max_impact))}")
        
        results = campaign.to_dict()
    else:
        print_info("Specify --component, --type, or --all")
        return None
    
    return results


# =============================================================================
# Event Simulation
# =============================================================================

def run_event_simulation(args, graph) -> Optional[Dict[str, Any]]:
    """Run event-driven simulation."""
    from src.simulation import EventSimulator
    
    print_header("EVENT SIMULATION")
    
    # Configure simulator
    simulator = EventSimulator(
        seed=args.seed,
        default_latency=args.latency,
        queue_capacity=args.queue_capacity,
        timeout=args.timeout,
    )
    
    print_kv("Duration", f"{args.duration}ms")
    print_kv("Message Rate", f"{args.message_rate}/sec")
    print_kv("Default Latency", f"{args.latency}ms")
    print_kv("Queue Capacity", args.queue_capacity)
    
    # Build failure schedule
    failure_schedule = None
    if args.inject_failure:
        failure_time = args.failure_time or args.duration / 4
        failure_schedule = {}
        
        for comp_id in args.inject_failure:
            if args.failure_duration:
                failure_schedule[comp_id] = (failure_time, failure_time + args.failure_duration)
            else:
                failure_schedule[comp_id] = (failure_time, None)
        
        print_section("Failure Schedule")
        for comp_id, (start, end) in failure_schedule.items():
            if end:
                print_kv(comp_id, f"fails at {start}ms, recovers at {end}ms")
            else:
                print_kv(comp_id, f"fails at {start}ms (permanent)")
    
    # Run simulation
    print_section("Running Simulation")
    
    result = simulator.run(
        graph,
        duration=args.duration,
        message_rate=args.message_rate,
        failure_schedule=failure_schedule,
    )
    
    # Results
    print_section("Simulation Results")
    
    stats = result.stats
    print(f"\n  {Colors.BOLD}Message Statistics{Colors.RESET}")
    print_kv("Messages Generated", stats.total_messages)
    print_kv("Messages Delivered", stats.delivered_messages)
    print_kv("Messages Failed", stats.failed_messages)
    print_kv("Messages Timeout", stats.timeout_messages)
    
    delivery_color = Colors.GREEN if stats.delivery_rate > 0.9 else Colors.YELLOW if stats.delivery_rate > 0.7 else Colors.RED
    print(f"\n  {Colors.BOLD}Delivery Rate:{Colors.RESET} {delivery_color}{stats.delivery_rate:.2%}{Colors.RESET}")
    
    print(f"\n  {Colors.BOLD}Latency (ms){Colors.RESET}")
    print_kv("Average", f"{stats.avg_latency:.2f}")
    print_kv("P99", f"{stats.p99_latency:.2f}")
    
    # Layer stats
    if result.layer_stats:
        print_section("Results by Layer")
        
        print(f"\n  {'Layer':<25} {'Sent':<10} {'Delivered':<12} {'Rate':<10} {'Latency'}")
        print(f"  {'-' * 70}")
        
        for layer, layer_stats in sorted(result.layer_stats.items()):
            rate_color = Colors.GREEN if layer_stats.delivery_rate > 0.9 else Colors.YELLOW
            print(f"  {layer:<25} "
                  f"{layer_stats.messages_sent:<10} "
                  f"{layer_stats.messages_delivered:<12} "
                  f"{rate_color}{layer_stats.delivery_rate:.2%}{Colors.RESET}     "
                  f"{layer_stats.avg_latency:.2f}ms")
    
    # Component loads
    if args.verbose and result.component_loads:
        print_section("Component Loads (Top 10)")
        
        sorted_loads = sorted(
            result.component_loads.items(),
            key=lambda x: -x[1].messages_processed
        )[:10]
        
        for comp_id, load in sorted_loads:
            print(f"  {comp_id}: processed={load.messages_processed}, "
                  f"dropped={load.messages_dropped}, "
                  f"avg_latency={load.avg_latency:.2f}ms")
    
    return result.to_dict()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate pub-sub system failures and message flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load from file
    python simulate_graph.py --input graph.json --failure --all
    python simulate_graph.py --input graph.json --event --duration 5000
    
    # Load from Neo4j
    python simulate_graph.py --neo4j --failure --all
    python simulate_graph.py --neo4j --uri bolt://localhost:7687 --failure --all
    
    # Load specific layer from Neo4j
    python simulate_graph.py --neo4j --layer application --failure --all
    
    # Export results
    python simulate_graph.py --neo4j --failure --all --output results.json

Layers:
    application   - App-to-app via topics
    infrastructure- Node-to-node connections
    app_broker    - App-to-broker messaging
    node_broker   - Deployment topology
        """
    )
    
    # Input source group
    input_group = parser.add_argument_group("Input Source")
    input_group.add_argument(
        "--input", "-i",
        help="Input graph JSON file"
    )
    input_group.add_argument(
        "--neo4j", "-n",
        action="store_true",
        help="Load graph from Neo4j database"
    )
    
    # Neo4j connection options
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI (default: bolt://localhost:7687)"
    )
    neo4j_group.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--password",
        default="password",
        help="Neo4j password (default: password)"
    )
    neo4j_group.add_argument(
        "--database",
        default="neo4j",
        help="Neo4j database name (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--layer",
        choices=["application", "infrastructure", "app_broker", "node_broker"],
        help="Load specific layer from Neo4j"
    )
    
    # Simulation type
    sim_type = parser.add_mutually_exclusive_group()
    sim_type.add_argument(
        "--failure", "-f",
        action="store_true",
        help="Run failure simulation"
    )
    sim_type.add_argument(
        "--event", "-e",
        action="store_true",
        help="Run event-driven simulation"
    )
    
    # Failure options
    failure_group = parser.add_argument_group("Failure Simulation")
    failure_group.add_argument(
        "--component", "-c",
        help="Simulate failure of specific component"
    )
    failure_group.add_argument(
        "--type", "-t",
        choices=["Application", "Broker", "Topic", "Node"],
        help="Simulate all components of type"
    )
    failure_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Simulate all component failures"
    )
    failure_group.add_argument(
        "--cascade",
        action="store_true",
        default=True,
        help="Enable cascade propagation (default)"
    )
    failure_group.add_argument(
        "--no-cascade",
        action="store_false",
        dest="cascade",
        help="Disable cascade propagation"
    )
    failure_group.add_argument(
        "--cascade-threshold",
        type=float,
        default=0.5,
        help="Cascade threshold (default: 0.5)"
    )
    failure_group.add_argument(
        "--critical-threshold",
        type=float,
        default=0.3,
        help="Critical impact threshold (default: 0.3)"
    )
    failure_group.add_argument(
        "--failure-mode",
        choices=["crash", "degraded", "partition"],
        default="crash",
        help="Failure mode (default: crash)"
    )
    
    # Event options
    event_group = parser.add_argument_group("Event Simulation")
    event_group.add_argument(
        "--duration", "-d",
        type=float,
        default=1000.0,
        help="Simulation duration in ms (default: 1000)"
    )
    event_group.add_argument(
        "--message-rate", "-r",
        type=float,
        default=100.0,
        help="Messages per second (default: 100)"
    )
    event_group.add_argument(
        "--latency",
        type=float,
        default=10.0,
        help="Default network latency in ms (default: 10)"
    )
    event_group.add_argument(
        "--queue-capacity",
        type=int,
        default=1000,
        help="Queue capacity per component (default: 1000)"
    )
    event_group.add_argument(
        "--timeout",
        type=float,
        default=5000.0,
        help="Message timeout in ms (default: 5000)"
    )
    event_group.add_argument(
        "--inject-failure",
        nargs="+",
        help="Components to fail during simulation"
    )
    event_group.add_argument(
        "--failure-time",
        type=float,
        help="When to inject failure (ms)"
    )
    event_group.add_argument(
        "--failure-duration",
        type=float,
        help="Failure duration (ms, omit for permanent)"
    )
    
    # Common options
    parser.add_argument(
        "--seed", "-s",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output", "-o",
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    args = parser.parse_args()
    
    # Handle colors
    if args.no_color or not use_colors():
        Colors.disable()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level)
    
    # Import check
    try:
        from src.simulation import SimulationGraph
    except ImportError as e:
        print_error(f"Import failed: {e}")
        return 1
    
    # Load graph
    graph = load_graph(args)
    if graph is None:
        return 1
    
    # Run simulation
    results = None
    
    if args.failure:
        results = run_failure_simulation(args, graph)
    elif args.event:
        results = run_event_simulation(args, graph)
    else:
        print_info("Specify --failure or --event simulation type")
        
        # Show graph summary
        print_section("Graph Summary")
        summary = graph.summary()
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"\n  {key}:")
                for k, v in value.items():
                    print_kv(k, v, indent=4)
            else:
                print_kv(key, value)
        
        return 0
    
    # Export results
    if results and args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print_success(f"Results exported to {output_path}")
    
    print_header("Simulation Complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())