#!/usr/bin/env python3
"""
Simulate Graph CLI - Version 5.0

Command-line interface for graph-based simulation of distributed pub-sub systems.

Simulation Types:
1. Failure Simulation: Test system resilience by failing components
2. Event Simulation: Simulate message flow with discrete events

Features:
- Component-type specific simulation
- Load graph from JSON file or Neo4j database
- Cascade failure propagation
- QoS-aware message simulation
- Impact scoring and statistics

Usage:
    # Failure simulation from JSON file
    python simulate_graph.py --input graph.json --failure

    # Failure simulation from Neo4j
    python simulate_graph.py --neo4j --password secret --failure

    # Event-driven simulation
    python simulate_graph.py --input graph.json --event --duration 5000

    # Simulate specific component type
    python simulate_graph.py --input graph.json --failure --component-type Application

    # Full campaign with cascade
    python simulate_graph.py --input graph.json --failure --cascade --all

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

try:
    from src.simulation import (
        SimulationGraph,
        ComponentType,
        FailureSimulator,
        FailureMode,
        EventSimulator,
        load_graph_from_neo4j,
        create_simulation_graph,
    )
    HAS_SIMULATION = True
except ImportError as e:
    HAS_SIMULATION = False
    IMPORT_ERROR = str(e)


# =============================================================================
# Output Formatting
# =============================================================================

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
GRAY = "\033[90m"
CYAN = "\033[96m"


def print_header(title: str) -> None:
    """Print a section header"""
    print(f"\n{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}{CYAN}{title.center(70)}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}\n")


def print_subheader(title: str) -> None:
    """Print a subsection header"""
    print(f"\n{BOLD}{title}{RESET}")
    print(f"{'-' * 50}")


def impact_color(score: float) -> str:
    """Get color based on impact score"""
    if score >= 0.5:
        return RED
    elif score >= 0.3:
        return YELLOW
    elif score >= 0.1:
        return BLUE
    return GREEN


# =============================================================================
# Graph Loading
# =============================================================================

def load_graph(args) -> Optional[SimulationGraph]:
    """Load graph from file or Neo4j"""
    try:
        if args.neo4j:
            print(f"{CYAN}Loading graph from Neo4j at {args.uri}...{RESET}")
            
            # Determine what to load
            component_type = None
            if args.component_type:
                component_type = ComponentType.from_string(args.component_type)
            
            graph = load_graph_from_neo4j(
                uri=args.uri,
                user=args.user,
                password=args.password,
                database=args.database,
                component_type=component_type,
            )
        elif args.input:
            print(f"{CYAN}Loading graph from {args.input}...{RESET}")
            graph = create_simulation_graph(args.input)
        else:
            print(f"{RED}Error: Must specify --input or --neo4j{RESET}")
            return None
        
        stats = graph.get_statistics()
        print(f"  Components: {stats['total_components']}")
        print(f"  Edges: {stats['total_edges']}")
        
        return graph
        
    except Exception as e:
        print(f"{RED}Error loading graph: {e}{RESET}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


# =============================================================================
# Failure Simulation
# =============================================================================

def run_failure_simulation(args, graph: SimulationGraph) -> Optional[Dict[str, Any]]:
    """Run failure simulation"""
    print_header("FAILURE SIMULATION")
    
    # Configure simulator
    simulator = FailureSimulator(
        seed=args.seed,
        cascade=args.cascade,
        cascade_threshold=args.cascade_threshold,
        critical_threshold=args.critical_threshold,
    )
    
    # Determine failure mode
    failure_mode = FailureMode.CRASH
    if args.failure_mode:
        try:
            failure_mode = FailureMode(args.failure_mode)
        except ValueError:
            print(f"{YELLOW}Unknown failure mode '{args.failure_mode}', using CRASH{RESET}")
    
    # Single component simulation
    if args.component:
        print_subheader(f"Single Failure: {args.component}")
        
        result = simulator.simulate_failure(graph, args.component, failure_mode)
        
        print(f"  Component: {result.failed_component}")
        print(f"  Type: {result.component_type.value}")
        print(f"  Mode: {result.failure_mode.value}")
        color = impact_color(result.impact_score)
        print(f"  {BOLD}Impact: {color}{result.impact_score:.4f}{RESET}")
        print(f"  Directly Affected: {len(result.directly_affected)}")
        print(f"  Cascade Affected: {len(result.cascade_affected)}")
        print(f"  Total Affected: {result.total_affected} / {result.total_components}")
        print(f"  Message Paths Broken: {result.message_paths_broken}")
        
        if args.verbose and result.directly_affected:
            print(f"\n  Directly Affected Components:")
            for comp_id in sorted(result.directly_affected)[:20]:
                print(f"    - {comp_id}")
        
        return {"single_failure": result.to_dict()}
    
    # Campaign simulation
    component_types = None
    if args.component_type:
        component_types = [ComponentType.from_string(args.component_type)]
        print_subheader(f"Campaign: {args.component_type} Components")
    elif args.all:
        print_subheader("Campaign: All Components")
    else:
        print_subheader("Campaign: All Components")
    
    campaign = simulator.simulate_all_failures(
        graph, 
        failure_mode, 
        component_types
    )
    
    # Summary
    print(f"  Total Tested: {len(campaign.results)}")
    print(f"  Critical (>{campaign.critical_threshold:.0%} impact): "
          f"{RED}{len(campaign.critical_components)}{RESET}")
    
    # Top 10 by impact
    print(f"\n  {BOLD}Top 10 by Impact:{RESET}")
    for i, (comp_id, impact) in enumerate(campaign.ranked_by_impact[:10], 1):
        comp = graph.get_component(comp_id)
        comp_type = comp.type.value if comp else "Unknown"
        color = impact_color(impact)
        print(f"    {i:2}. {color}{comp_id}{RESET} ({comp_type}): {impact:.4f}")
    
    # By type summary
    if not component_types:
        print(f"\n  {BOLD}By Component Type:{RESET}")
        for comp_type in ComponentType:
            type_results = campaign.get_by_type(comp_type)
            if type_results:
                avg_impact = sum(r.impact_score for r in type_results) / len(type_results)
                max_impact = max(r.impact_score for r in type_results)
                print(f"    {comp_type.value}:")
                print(f"      Count: {len(type_results)}, Avg Impact: {avg_impact:.4f}, "
                      f"Max: {max_impact:.4f}")
    
    return {"campaign": campaign.to_dict()}


# =============================================================================
# Event Simulation
# =============================================================================

def run_event_simulation(args, graph: SimulationGraph) -> Optional[Dict[str, Any]]:
    """Run event-driven simulation"""
    print_header("EVENT-DRIVEN SIMULATION")
    
    # Configure simulator
    simulator = EventSimulator(
        graph,
        seed=args.seed,
        default_latency=args.latency,
        queue_capacity=args.queue_capacity,
        timeout=args.timeout,
    )
    
    # Inject failures if specified
    if args.inject_failure:
        for comp_id in args.inject_failure:
            fail_time = args.failure_time or 0.0
            duration = args.failure_duration
            simulator.inject_failure(comp_id, fail_time, duration)
            print(f"  Scheduled failure: {comp_id} at {fail_time}ms" +
                  (f" (duration: {duration}ms)" if duration else " (permanent)"))
    
    # Run simulation
    print_subheader(f"Running for {args.duration}ms at {args.message_rate} msg/ms")
    
    result = simulator.run(
        duration=args.duration,
        message_rate=args.message_rate,
        log_events=args.verbose,
    )
    
    # Statistics
    stats = result.statistics
    print(f"\n  {BOLD}Message Statistics:{RESET}")
    print(f"    Total Messages: {stats.total_messages}")
    print(f"    Delivered: {GREEN}{stats.delivered_messages}{RESET}")
    print(f"    Failed: {RED}{stats.failed_messages}{RESET}")
    print(f"    Dropped: {YELLOW}{stats.dropped_messages}{RESET}")
    print(f"    Timeout: {YELLOW}{stats.timeout_messages}{RESET}")
    print(f"    Delivery Rate: {stats.delivery_rate:.2%}")
    
    print(f"\n  {BOLD}Performance:{RESET}")
    print(f"    Avg Latency: {stats.average_latency:.2f} ms")
    print(f"    P99 Latency: {stats.p99_latency:.2f} ms")
    print(f"    Throughput: {stats.average_throughput:.2f} msg/s")
    print(f"    Total Events: {stats.total_events}")
    print(f"    Wall Clock Time: {stats.wall_clock_time:.3f}s")
    
    # Component loads
    if args.verbose:
        print(f"\n  {BOLD}Component Loads (Top 10 by Processed):{RESET}")
        sorted_loads = sorted(
            result.component_loads.values(),
            key=lambda x: x.messages_processed,
            reverse=True
        )[:10]
        
        for load in sorted_loads:
            if load.messages_processed > 0:
                print(f"    {load.component_id}: "
                      f"processed={load.messages_processed}, "
                      f"dropped={load.messages_dropped}, "
                      f"avg_lat={load.average_latency:.2f}ms")
    
    # Bottlenecks and issues
    bottlenecks = result.get_bottlenecks(threshold=args.queue_capacity // 2)
    critical = result.get_critical_components(threshold=0.1)
    
    if bottlenecks:
        print(f"\n  {YELLOW}Bottlenecks Detected:{RESET}")
        for comp_id in bottlenecks:
            load = result.component_loads[comp_id]
            print(f"    - {comp_id}: peak queue = {load.peak_queue_size}")
    
    if critical:
        print(f"\n  {RED}High Drop Rate Components:{RESET}")
        for comp_id in critical:
            load = result.component_loads[comp_id]
            print(f"    - {comp_id}: drop rate = {load.drop_rate:.2%}")
    
    if result.failed_components:
        print(f"\n  {RED}Failed Components:{RESET}")
        for comp_id in result.failed_components:
            print(f"    - {comp_id}")
    
    return {"event_simulation": result.to_dict()}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Graph-based simulation of distributed pub-sub systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Failure simulation from JSON file
  python simulate_graph.py --input graph.json --failure --cascade
  
  # Single component failure
  python simulate_graph.py --input graph.json --failure --component broker1
  
  # Event simulation
  python simulate_graph.py --input graph.json --event --duration 5000 --message-rate 50
  
  # Load from Neo4j
  python simulate_graph.py --neo4j --password secret --failure --all
  
  # Simulate specific component type
  python simulate_graph.py --input graph.json --failure --component-type Application
  
  # Event simulation with injected failure
  python simulate_graph.py --input graph.json --event --inject-failure broker1 --failure-time 2000
  
  # Export results to JSON
  python simulate_graph.py --input graph.json --failure --output results.json
        """
    )
    
    # Input source
    input_group = parser.add_argument_group("Input Source")
    input_group.add_argument(
        "--input", "-i",
        help="Input graph file (JSON format)"
    )
    input_group.add_argument(
        "--neo4j",
        action="store_true",
        help="Load graph from Neo4j database"
    )
    
    # Neo4j connection
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument(
        "--uri", "-u",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI (default: bolt://localhost:7687)"
    )
    neo4j_group.add_argument(
        "--user", "-U",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password"
    )
    neo4j_group.add_argument(
        "--database", "-d",
        default="neo4j",
        help="Neo4j database name (default: neo4j)"
    )
    
    # Simulation type
    sim_type = parser.add_argument_group("Simulation Type")
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
    
    # Failure simulation options
    failure_group = parser.add_argument_group("Failure Simulation Options")
    failure_group.add_argument(
        "--component", "-c",
        help="Simulate failure of specific component"
    )
    failure_group.add_argument(
        "--component-type", "-t",
        choices=["Application", "Broker", "Topic", "Node"],
        help="Simulate failures for specific component type"
    )
    failure_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Simulate all component failures (campaign)"
    )
    failure_group.add_argument(
        "--cascade",
        action="store_true",
        default=True,
        help="Enable cascade failure propagation (default: True)"
    )
    failure_group.add_argument(
        "--no-cascade",
        action="store_false",
        dest="cascade",
        help="Disable cascade failure propagation"
    )
    failure_group.add_argument(
        "--cascade-threshold",
        type=float,
        default=0.5,
        help="Fraction of dependencies that must fail to cascade (default: 0.5)"
    )
    failure_group.add_argument(
        "--critical-threshold",
        type=float,
        default=0.3,
        help="Impact threshold for critical classification (default: 0.3)"
    )
    failure_group.add_argument(
        "--failure-mode",
        choices=["crash", "degraded", "partition", "overload"],
        default="crash",
        help="Type of failure to simulate (default: crash)"
    )
    
    # Event simulation options
    event_group = parser.add_argument_group("Event Simulation Options")
    event_group.add_argument(
        "--duration",
        type=float,
        default=1000.0,
        help="Simulation duration in ms (default: 1000)"
    )
    event_group.add_argument(
        "--message-rate",
        type=float,
        default=10.0,
        help="Messages per ms (default: 10)"
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
        help="Max queue size per component (default: 1000)"
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
        help="Component IDs to fail during simulation"
    )
    event_group.add_argument(
        "--failure-time",
        type=float,
        help="When to trigger injected failure (ms)"
    )
    event_group.add_argument(
        "--failure-duration",
        type=float,
        help="Duration of injected failure (ms, omit for permanent)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        help="Output file path (JSON format)"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    output_group.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Check module availability
    if not HAS_SIMULATION:
        print(f"{RED}Error: Simulation module not available.{RESET}")
        print(f"Import error: {IMPORT_ERROR}")
        print("Make sure you're running from the project root.")
        sys.exit(1)
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else (
        logging.DEBUG if args.verbose else logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Validate arguments
    if not args.input and not args.neo4j:
        print(f"{RED}Error: Must specify --input or --neo4j{RESET}")
        parser.print_help()
        sys.exit(1)
    
    if not args.failure and not args.event:
        print(f"{YELLOW}No simulation type specified, defaulting to --failure{RESET}")
        args.failure = True
    
    # Load graph
    graph = load_graph(args)
    if not graph:
        sys.exit(1)
    
    # Run simulation
    result = None
    try:
        if args.failure:
            result = run_failure_simulation(args, graph)
        elif args.event:
            result = run_event_simulation(args, graph)
        
    except Exception as e:
        print(f"{RED}Error during simulation: {e}{RESET}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Output results
    if result:
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n{GREEN}✓ Results saved to: {args.output}{RESET}")
        
        if args.json:
            print(json.dumps(result, indent=2, default=str))
    
    print(f"\n{GREEN}✓ Simulation complete{RESET}")


if __name__ == "__main__":
    main()