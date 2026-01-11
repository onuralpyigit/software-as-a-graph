#!/usr/bin/env python3
"""
Graph Simulation CLI

Comprehensive simulation for distributed pub-sub systems.
Supports event-driven and failure simulation on the raw graph model.

Simulation Types:
    - Event: Message flow simulation (throughput, latency, drops)
    - Failure: Component failure with cascade propagation
    - Report: Full analysis with criticality classification

Layers:
    app      : Application layer
    infra    : Infrastructure layer
    mw-app   : Middleware-Application layer
    mw-infra : Middleware-Infrastructure layer
    system   : Complete system

Usage:
    python simulate_graph.py --event App1
    python simulate_graph.py --failure Broker1
    python simulate_graph.py --report --layers app,infra,system
    python simulate_graph.py --exhaustive --layer system

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation import (
    Simulator,
    EventResult,
    FailureResult,
    SimulationReport,
    LayerMetrics,
    ComponentCriticality,
)


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def colored(text: str, color: str, bold: bool = False) -> str:
    """Apply color to text."""
    style = Colors.BOLD if bold else ""
    return f"{style}{color}{text}{Colors.RESET}"


def level_color(level: str) -> str:
    """Get color for criticality level."""
    return {
        "critical": Colors.RED,
        "high": Colors.YELLOW,
        "medium": Colors.BLUE,
        "low": Colors.WHITE,
        "minimal": Colors.GRAY,
    }.get(level.lower(), Colors.RESET)


# =============================================================================
# Display Functions
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 78) -> None:
    """Print a formatted header."""
    print(f"\n{colored(char * width, Colors.CYAN)}")
    print(f"{colored(f' {title} '.center(width), Colors.CYAN, bold=True)}")
    print(f"{colored(char * width, Colors.CYAN)}")


def print_subheader(title: str) -> None:
    """Print a subheader."""
    print(f"\n{colored(f'>> {title}', Colors.WHITE, bold=True)}")


def display_event_result(result: EventResult) -> None:
    """Display event simulation results."""
    print_header(f"Event Simulation: {result.source_app}")
    
    print(f"\n  {colored('Scenario:', Colors.CYAN)}     {result.scenario}")
    print(f"  {colored('Duration:', Colors.CYAN)}     {result.duration * 1000:.2f} ms")
    
    # Throughput metrics
    print_subheader("Throughput Metrics")
    m = result.metrics
    print(f"\n  Messages Published:  {m.messages_published}")
    print(f"  Messages Delivered:  {colored(str(m.messages_delivered), Colors.GREEN)}")
    
    drop_color = Colors.RED if m.messages_dropped > 0 else Colors.GRAY
    print(f"  Messages Dropped:    {colored(str(m.messages_dropped), drop_color)}")
    print(f"  Delivery Rate:       {m.delivery_rate:.1f}%")
    print(f"  Drop Rate:           {m.drop_rate:.1f}%")
    
    # Latency metrics
    print_subheader("Latency Metrics")
    print(f"\n  Average Latency:     {m.avg_latency * 1000:.3f} ms")
    
    if m.min_latency != float('inf'):
        print(f"  Min Latency:         {m.min_latency * 1000:.3f} ms")
    else:
        print(f"  Min Latency:         N/A")
    
    print(f"  Max Latency:         {m.max_latency * 1000:.3f} ms")
    print(f"  P50 Latency:         {m.p50_latency * 1000:.3f} ms")
    print(f"  P99 Latency:         {m.p99_latency * 1000:.3f} ms")
    print(f"  Throughput:          {m.throughput:.1f} msg/sec")
    
    # Path analysis
    if result.affected_topics:
        print_subheader("Path Analysis")
        print(f"\n  Affected Topics:     {', '.join(result.affected_topics)}")
        print(f"  Brokers Used:        {', '.join(result.brokers_used) or 'None'}")
        print(f"  Reached Subscribers: {len(result.reached_subscribers)}")
        if result.reached_subscribers:
            subs = result.reached_subscribers[:5]
            print(f"                       {', '.join(subs)}")
            if len(result.reached_subscribers) > 5:
                print(f"                       ... and {len(result.reached_subscribers) - 5} more")
    
    # Drop reasons
    if result.drop_reasons:
        print_subheader("Drop Analysis")
        for reason, count in result.drop_reasons.items():
            print(f"  {reason}: {count}")
    
    # Top impacted components
    if result.component_impacts:
        print_subheader("Component Impacts (Top 5)")
        sorted_impacts = sorted(
            result.component_impacts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        print(f"\n  {'Component':<25} {'Impact Score':<15}")
        print(f"  {'-' * 40}")
        for comp_id, impact in sorted_impacts:
            print(f"  {comp_id:<25} {impact:.4f}")


def display_failure_result(result: FailureResult) -> None:
    """Display failure simulation results."""
    print_header(f"Failure Simulation: {result.target_id}")
    
    print(f"\n  {colored('Target:', Colors.CYAN)}       {result.target_id} ({result.target_type})")
    print(f"  {colored('Scenario:', Colors.CYAN)}     {result.scenario}")
    
    # Impact metrics
    print_subheader("Impact Metrics")
    impact = result.impact
    
    # Composite impact with color coding
    imp_color = Colors.RED if impact.composite_impact > 0.5 else (
        Colors.YELLOW if impact.composite_impact > 0.2 else Colors.GREEN
    )
    print(f"\n  {colored('Composite Impact:', Colors.WHITE, bold=True)} {colored(f'{impact.composite_impact:.4f}', imp_color)}")
    
    print(f"\n  Reachability Loss:   {impact.reachability_loss * 100:.1f}%")
    print(f"    Initial Paths:     {impact.initial_paths}")
    print(f"    Remaining Paths:   {impact.remaining_paths}")
    
    print(f"\n  Fragmentation:       {impact.fragmentation * 100:.1f}%")
    print(f"    Initial Components: {impact.initial_components}")
    print(f"    Failed Components:  {impact.failed_components}")
    
    print(f"\n  Throughput Loss:     {impact.throughput_loss * 100:.1f}%")
    print(f"    Affected Topics:   {impact.affected_topics}")
    
    # Affected entities
    print_subheader("Affected Entities")
    print(f"\n  Publishers Affected:  {impact.affected_publishers}")
    print(f"  Subscribers Affected: {impact.affected_subscribers}")
    
    # Cascade analysis
    print_subheader("Cascade Analysis")
    print(f"\n  Cascade Count:       {impact.cascade_count}")
    print(f"  Cascade Depth:       {impact.cascade_depth}")
    
    if impact.cascade_by_type:
        print(f"  Cascade by Type:")
        for comp_type, count in impact.cascade_by_type.items():
            print(f"    {comp_type}: {count}")
    
    # Cascaded failures
    if result.cascaded_failures:
        print_subheader("Cascaded Failures")
        for i, comp_id in enumerate(result.cascaded_failures[:10]):
            print(f"  {i+1}. {comp_id}")
        if len(result.cascaded_failures) > 10:
            print(f"  ... and {len(result.cascaded_failures) - 10} more")
    
    # Layer impacts
    if result.layer_impacts:
        print_subheader("Per-Layer Impact")
        print(f"\n  {'Layer':<15} {'Impact':<10}")
        print(f"  {'-' * 25}")
        for layer, layer_impact in result.layer_impacts.items():
            color = Colors.RED if layer_impact > 0.3 else (
                Colors.YELLOW if layer_impact > 0.1 else Colors.GREEN
            )
            print(f"  {layer:<15} {colored(f'{layer_impact:.4f}', color)}")


def display_exhaustive_results(results: List[FailureResult], limit: int = 15) -> None:
    """Display exhaustive failure simulation results."""
    print_header("Exhaustive Failure Analysis")
    
    print(f"\n  Total Components Analyzed: {len(results)}")
    
    if not results:
        print(f"  {colored('No results.', Colors.GRAY)}")
        return
    
    # Summary statistics
    impacts = [r.impact.composite_impact for r in results]
    avg_impact = sum(impacts) / len(impacts)
    max_impact = max(impacts)
    
    print(f"  Average Impact:           {avg_impact:.4f}")
    print(f"  Maximum Impact:           {max_impact:.4f}")
    
    # Count by impact level
    critical = sum(1 for r in results if r.impact.composite_impact > 0.5)
    high = sum(1 for r in results if 0.3 < r.impact.composite_impact <= 0.5)
    medium = sum(1 for r in results if 0.1 < r.impact.composite_impact <= 0.3)
    low = sum(1 for r in results if r.impact.composite_impact <= 0.1)
    
    print(f"\n  Impact Distribution:")
    print(f"    {colored('CRITICAL (>0.5):', Colors.RED)} {critical}")
    print(f"    {colored('HIGH (0.3-0.5):', Colors.YELLOW)} {high}")
    print(f"    {colored('MEDIUM (0.1-0.3):', Colors.BLUE)} {medium}")
    print(f"    {colored('LOW (≤0.1):', Colors.GREEN)} {low}")
    
    # Top components
    print_subheader(f"Top {limit} Components by Impact")
    
    print(f"\n  {'Component':<20} {'Type':<12} {'Impact':<10} {'Cascade':<10} {'Reach Loss':<10}")
    print(f"  {'-' * 65}")
    
    for r in results[:limit]:
        imp = r.impact.composite_impact
        color = Colors.RED if imp > 0.5 else (
            Colors.YELLOW if imp > 0.2 else Colors.GREEN
        )
        
        print(
            f"  {r.target_id:<20} {r.target_type:<12} "
            f"{colored(f'{imp:.4f}', color):<10} "
            f"{r.impact.cascade_count:<10} "
            f"{r.impact.reachability_loss*100:.1f}%"
        )
    
    if len(results) > limit:
        print(f"\n  {colored(f'... and {len(results) - limit} more', Colors.GRAY)}")


def display_report(report: SimulationReport) -> None:
    """Display comprehensive simulation report."""
    print_header("Simulation Report", "═")
    
    print(f"\n  {colored('Timestamp:', Colors.CYAN)}  {report.timestamp}")
    
    # Graph summary
    print_subheader("Graph Summary")
    gs = report.graph_summary
    print(f"\n  Total Nodes:      {gs.get('total_nodes', 0)}")
    print(f"  Total Edges:      {gs.get('total_edges', 0)}")
    print(f"  Topics:           {gs.get('topics', 0)}")
    print(f"  Pub-Sub Paths:    {gs.get('pub_sub_paths', 0)}")
    
    if gs.get('component_types'):
        types_str = ", ".join(f"{k}: {v}" for k, v in gs['component_types'].items())
        print(f"  Component Types:  {types_str}")
    
    # Layer metrics
    for layer_name, metrics in report.layer_metrics.items():
        print_subheader(f"Layer: {layer_name.upper()}")
        
        print(f"\n  {colored('Event Simulation:', Colors.CYAN)}")
        print(f"    Throughput:        {metrics.event_throughput} messages")
        print(f"    Delivery Rate:     {metrics.event_delivery_rate:.1f}%")
        print(f"    Drop Rate:         {metrics.event_drop_rate:.1f}%")
        print(f"    Avg Latency:       {metrics.event_avg_latency_ms:.3f} ms")
        
        print(f"\n  {colored('Failure Simulation:', Colors.CYAN)}")
        print(f"    Avg Reach Loss:    {metrics.avg_reachability_loss * 100:.1f}%")
        print(f"    Avg Fragmentation: {metrics.avg_fragmentation * 100:.1f}%")
        print(f"    Avg Throughput Loss: {metrics.avg_throughput_loss * 100:.1f}%")
        print(f"    Max Impact:        {metrics.max_impact:.4f}")
        
        print(f"\n  {colored('Criticality:', Colors.CYAN)}")
        print(f"    Total Components:  {metrics.total_components}")
        print(f"    Critical:          {colored(str(metrics.critical_count), Colors.RED)}")
        print(f"    High:              {colored(str(metrics.high_count), Colors.YELLOW)}")
        print(f"    Medium:            {metrics.medium_count}")
        print(f"    SPOFs:             {metrics.spof_count}")
    
    # Top critical components
    if report.top_critical:
        print_subheader("Top Critical Components")
        
        print(f"\n  {'ID':<20} {'Type':<12} {'Level':<10} {'Impact':<10} {'Cascade':<10}")
        print(f"  {'-' * 65}")
        
        for c in report.top_critical:
            color = level_color(c['level'])
            print(
                f"  {c['id']:<20} {c['type']:<12} "
                f"{colored(c['level'], color):<10} "
                f"{c['combined_impact']:.4f}     "
                f"{c['cascade_count']:>3}"
            )
    
    # Recommendations
    if report.recommendations:
        print_subheader("Recommendations")
        for i, rec in enumerate(report.recommendations, 1):
            # Color code by severity keyword
            if "CRITICAL" in rec:
                color = Colors.RED
            elif "HIGH" in rec or "SPOF" in rec:
                color = Colors.YELLOW
            elif "HEALTHY" in rec:
                color = Colors.GREEN
            else:
                color = Colors.WHITE
            
            print(f"\n  {i}. {colored(rec, color)}")
    
    # Final summary
    print_subheader("Summary")
    
    total_critical = sum(m.critical_count for m in report.layer_metrics.values())
    total_high = sum(m.high_count for m in report.layer_metrics.values())
    
    if total_critical > 0:
        print(f"\n  {colored(f'⚠ {total_critical} CRITICAL components require immediate attention', Colors.RED, bold=True)}")
    if total_high > 0:
        print(f"  {colored(f'⚠ {total_high} HIGH priority components should be reviewed', Colors.YELLOW)}")
    if total_critical == 0 and total_high == 0:
        print(f"\n  {colored('✓ No critical issues detected. System looks healthy!', Colors.GREEN, bold=True)}")
    
    print()


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive simulation for distributed pub-sub systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simulation Types:
  --event APP_ID    Event simulation from source application
  --failure COMP_ID Failure simulation for target component
  --exhaustive      Failure analysis for all components
  --report          Full simulation report with analysis

Layers:
  app      Application layer (Applications only)
  infra    Infrastructure layer (Nodes only)
  mw-app   Middleware-Application (Applications + Brokers)
  mw-infra Middleware-Infrastructure (Nodes + Brokers)
  system   Complete system (all components)

Examples:
  %(prog)s --event App1 --messages 100
  %(prog)s --failure Broker1
  %(prog)s --exhaustive --layer system
  %(prog)s --report --layers app,infra,system
  %(prog)s --report --output results/report.json
        """
    )
    
    # Action (required)
    action_group = parser.add_argument_group("Action (one required)")
    action_mutex = action_group.add_mutually_exclusive_group(required=True)
    action_mutex.add_argument(
        "--event", "-e",
        metavar="APP_ID",
        help="Run event simulation from source application"
    )
    action_mutex.add_argument(
        "--failure", "-f",
        metavar="COMP_ID",
        help="Run failure simulation for target component"
    )
    action_mutex.add_argument(
        "--exhaustive",
        action="store_true",
        help="Run exhaustive failure analysis for all components"
    )
    action_mutex.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate full simulation report"
    )
    
    # Simulation parameters
    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument(
        "--layer", "-l",
        choices=["app", "infra", "mw-app", "mw-infra", "system"],
        default="system",
        help="Analysis layer (default: system)"
    )
    sim_group.add_argument(
        "--layers",
        help="Comma-separated layers for report (default: app,infra,system)"
    )
    sim_group.add_argument(
        "--messages", "-m",
        type=int,
        default=100,
        help="Number of messages for event simulation (default: 100)"
    )
    sim_group.add_argument(
        "--duration", "-d",
        type=float,
        default=10.0,
        help="Event simulation duration in seconds (default: 10.0)"
    )
    sim_group.add_argument(
        "--cascade-prob",
        type=float,
        default=1.0,
        help="Cascade propagation probability (default: 1.0)"
    )
    
    # Neo4j connection
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )
    neo4j_group.add_argument(
        "--user", "-u",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password (default: password)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Export results to JSON file"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with debug information"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    try:
        with Simulator(
            uri=args.uri,
            user=args.user,
            password=args.password
        ) as sim:
            
            # === Event Simulation ===
            if args.event:
                result = sim.run_event_simulation(
                    source_app=args.event,
                    num_messages=args.messages,
                    duration=args.duration,
                )
                
                if args.json:
                    print(json.dumps(result.to_dict(), indent=2))
                elif not args.quiet:
                    display_event_result(result)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result.to_dict(), f, indent=2)
                    if not args.quiet:
                        print(f"\n{colored(f'Results saved to: {args.output}', Colors.GREEN)}")
            
            # === Failure Simulation ===
            elif args.failure:
                result = sim.run_failure_simulation(
                    target_id=args.failure,
                    layer=args.layer,
                    cascade_probability=args.cascade_prob,
                )
                
                if args.json:
                    print(json.dumps(result.to_dict(), indent=2))
                elif not args.quiet:
                    display_failure_result(result)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result.to_dict(), f, indent=2)
                    if not args.quiet:
                        print(f"\n{colored(f'Results saved to: {args.output}', Colors.GREEN)}")
            
            # === Exhaustive Analysis ===
            elif args.exhaustive:
                results = sim.run_failure_simulation_exhaustive(
                    layer=args.layer,
                    cascade_probability=args.cascade_prob,
                )
                
                if args.json:
                    print(json.dumps([r.to_dict() for r in results], indent=2))
                elif not args.quiet:
                    display_exhaustive_results(results)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump([r.to_dict() for r in results], f, indent=2)
                    if not args.quiet:
                        print(f"\n{colored(f'Results saved to: {args.output}', Colors.GREEN)}")
            
            # === Report ===
            elif args.report:
                layers = args.layers.split(",") if args.layers else ["app", "infra", "system"]
                
                report = sim.generate_report(layers=layers)
                
                if args.json:
                    print(json.dumps(report.to_dict(), indent=2))
                elif not args.quiet:
                    display_report(report)
                
                if args.output:
                    sim.export_report(report, args.output)
                    if not args.quiet:
                        print(f"\n{colored(f'Report saved to: {args.output}', Colors.GREEN)}")
        
        return 0
    
    except KeyboardInterrupt:
        print(f"\n{colored('Simulation interrupted.', Colors.YELLOW)}")
        return 130
    
    except Exception as e:
        logging.exception("Simulation failed")
        print(f"{colored(f'Error: {e}', Colors.RED)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())