#!/usr/bin/env python3
"""
Software-as-a-Graph Simulation CLI

Executes Event and Failure simulations on the raw graph model.
Supports both JSON files and Neo4j database as data sources.

Usage:
    # Event simulation
    python simulate_graph.py --event A1
    
    # Failure simulation for a specific target
    python simulate_graph.py --failure N1 --layer infrastructure
    
    # Full simulation report
    python simulate_graph.py --report
    
    # Exhaustive failure analysis with export
    python simulate_graph.py --exhaustive --output results/impacts.csv

Author: Software-as-a-Graph Research Project
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation import (
    Simulator,
    SimulationReport,
    EventScenario,
    EventResult,
    FailureScenario,
    FailureResult,
    RuntimeMetrics,
    ImpactMetrics,
)

# ============================================================================
# ANSI Color Codes
# ============================================================================
COLORS = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "GRAY": "\033[90m",
    "BOLD": "\033[1m",
    "RESET": "\033[0m",
}

# Disable colors if not a TTY
if not sys.stdout.isatty():
    COLORS = {k: "" for k in COLORS}

RED = COLORS["RED"]
GREEN = COLORS["GREEN"]
YELLOW = COLORS["YELLOW"]
BLUE = COLORS["BLUE"]
MAGENTA = COLORS["MAGENTA"]
CYAN = COLORS["CYAN"]
WHITE = COLORS["WHITE"]
GRAY = COLORS["GRAY"]
BOLD = COLORS["BOLD"]
RESET = COLORS["RESET"]


# ============================================================================
# Display Functions
# ============================================================================

def print_header(title: str, char: str = "=") -> None:
    """Print a section header."""
    width = 70
    print(f"\n{BOLD}{char * width}{RESET}")
    print(f"{BOLD} {title}{RESET}")
    print(f"{BOLD}{char * width}{RESET}")


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n{BOLD}>> {title}{RESET}")


def level_color(level: str) -> str:
    """Get color for criticality level."""
    return {
        "critical": RED,
        "high": YELLOW,
        "medium": BLUE,
        "low": WHITE,
        "minimal": GRAY,
    }.get(level.lower(), RESET)


def display_event_result(result: EventResult) -> None:
    """Display event simulation results."""
    print_header(f"Event Simulation: {result.source_app}")
    
    print(f"\n  {CYAN}Scenario:{RESET}     {result.scenario}")
    print(f"  {CYAN}Duration:{RESET}     {result.duration * 1000:.2f} ms")
    
    # Throughput metrics
    print_subheader("Throughput Metrics")
    m = result.metrics
    print(f"\n  Messages Published:  {m.messages_published}")
    print(f"  Messages Delivered:  {GREEN}{m.messages_delivered}{RESET}")
    print(f"  Messages Dropped:    {RED if m.messages_dropped > 0 else GRAY}{m.messages_dropped}{RESET}")
    print(f"  Delivery Rate:       {m.delivery_rate:.1f}%")
    print(f"  Drop Rate:           {m.drop_rate:.1f}%")
    
    # Latency metrics
    print_subheader("Latency Metrics")
    print(f"\n  Average Latency:     {m.avg_latency * 1000:.3f} ms")
    print(f"  Min Latency:         {m.min_latency * 1000:.3f} ms" if m.min_latency != float('inf') else f"  Min Latency:         N/A")
    print(f"  Max Latency:         {m.max_latency * 1000:.3f} ms")
    print(f"  P50 Latency:         {m.p50_latency * 1000:.3f} ms")
    print(f"  P99 Latency:         {m.p99_latency * 1000:.3f} ms")
    
    # Path analysis
    if result.affected_topics:
        print_subheader("Path Analysis")
        print(f"\n  Affected Topics:     {', '.join(result.affected_topics)}")
        print(f"  Reached Subscribers: {len(result.reached_subscribers)}")
        if result.reached_subscribers:
            print(f"                       {', '.join(result.reached_subscribers[:5])}")
    
    # Top impacted components
    if result.component_impacts:
        print_subheader("Component Impacts (Top 5)")
        sorted_impacts = sorted(
            result.component_impacts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        print(f"\n  {'Component':<20} {'Impact Score':<15}")
        print(f"  {'-' * 35}")
        for comp_id, impact in sorted_impacts:
            print(f"  {comp_id:<20} {impact:.4f}")


def display_failure_result(result: FailureResult) -> None:
    """Display failure simulation results."""
    print_header(f"Failure Simulation: {result.target_id}")
    
    print(f"\n  {CYAN}Scenario:{RESET}     {result.scenario}")
    print(f"  {CYAN}Target:{RESET}       {result.target_id} ({result.target_type})")
    print(f"  {CYAN}Duration:{RESET}     {result.duration_ms:.2f} ms")
    
    # Cascade information
    print_subheader("Cascade Analysis")
    print(f"\n  Cascaded Failures:   {RED if result.cascaded_failures else GREEN}{len(result.cascaded_failures)}{RESET}")
    
    if result.cascaded_failures:
        print(f"  Failed Components:   {', '.join(result.cascaded_failures[:10])}")
        if len(result.cascaded_failures) > 10:
            print(f"                       ... and {len(result.cascaded_failures) - 10} more")
    
    # Impact metrics
    impact = result.impact
    print_subheader("Impact Metrics")
    
    # Reachability
    reach_color = RED if impact.reachability_loss > 0.5 else (YELLOW if impact.reachability_loss > 0.2 else GREEN)
    print(f"\n  {CYAN}Reachability Loss:{RESET}")
    print(f"    Initial Paths:     {impact.initial_paths}")
    print(f"    Remaining Paths:   {impact.remaining_paths}")
    print(f"    Loss:              {reach_color}{impact.reachability_loss * 100:.1f}%{RESET}")
    
    # Fragmentation
    frag_color = RED if impact.fragmentation > 0.3 else (YELLOW if impact.fragmentation > 0.1 else GREEN)
    print(f"\n  {CYAN}Infrastructure:{RESET}")
    print(f"    Initial Active:    {impact.initial_components}")
    print(f"    Remaining Active:  {impact.remaining_components}")
    print(f"    Fragmentation:     {frag_color}{impact.fragmentation * 100:.1f}%{RESET}")
    
    # Throughput
    tput_color = RED if impact.throughput_loss > 0.5 else (YELLOW if impact.throughput_loss > 0.2 else GREEN)
    print(f"\n  {CYAN}Throughput:{RESET}")
    print(f"    Affected Topics:   {impact.affected_topics}")
    print(f"    Throughput Loss:   {tput_color}{impact.throughput_loss * 100:.1f}%{RESET}")
    
    # Composite impact
    composite_color = RED if impact.composite_impact > 0.5 else (YELLOW if impact.composite_impact > 0.3 else GREEN)
    print(f"\n  {BOLD}Composite Impact:    {composite_color}{impact.composite_impact:.4f}{RESET}")
    
    # Layer breakdown
    print_subheader("Layer Breakdown")
    print(f"\n  Application Layer:   {result.app_layer_impact * 100:.1f}%")
    print(f"  Infrastructure:      {result.infra_layer_impact * 100:.1f}%")


def display_exhaustive_results(results: List[FailureResult], limit: int = 20) -> None:
    """Display exhaustive failure simulation results."""
    print_header("Exhaustive Failure Analysis")
    
    print(f"\n  {CYAN}Total Components:{RESET}  {len(results)}")
    
    # Summary statistics
    impacts = [r.impact.composite_impact for r in results]
    avg_impact = sum(impacts) / len(impacts) if impacts else 0
    max_impact = max(impacts) if impacts else 0
    
    print(f"  {CYAN}Average Impact:{RESET}    {avg_impact:.4f}")
    print(f"  {CYAN}Maximum Impact:{RESET}    {max_impact:.4f}")
    
    # Critical components (high impact)
    critical = [r for r in results if r.impact.composite_impact > 0.5]
    high = [r for r in results if 0.3 < r.impact.composite_impact <= 0.5]
    
    print(f"\n  {RED}Critical (>50%):{RESET}   {len(critical)}")
    print(f"  {YELLOW}High (30-50%):{RESET}     {len(high)}")
    
    # Top components table
    print_subheader(f"Top {min(limit, len(results))} Critical Components")
    
    print(f"\n  {'-' * 75}")
    print(f"  {'ID':<15} {'Type':<12} {'Impact':<10} {'Reach Loss':<12} {'Cascade':<10}")
    print(f"  {'-' * 75}")
    
    for r in results[:limit]:
        impact_color = RED if r.impact.composite_impact > 0.5 else (YELLOW if r.impact.composite_impact > 0.3 else GRAY)
        print(
            f"  {r.target_id:<15} {r.target_type:<12} "
            f"{impact_color}{r.impact.composite_impact:.4f}{RESET}     "
            f"{r.impact.reachability_loss * 100:>5.1f}%       "
            f"{len(r.cascaded_failures):>3}"
        )
    
    if len(results) > limit:
        print(f"\n  {GRAY}... and {len(results) - limit} more{RESET}")


def display_simulation_report(report: SimulationReport) -> None:
    """Display comprehensive simulation report."""
    print_header("Simulation Report", "═")
    
    print(f"\n  {CYAN}Timestamp:{RESET}  {report.timestamp}")
    
    # Graph summary
    print_subheader("Graph Summary")
    gs = report.graph_summary
    print(f"\n  Total Nodes:      {gs.get('total_nodes', 0)}")
    print(f"  Total Edges:      {gs.get('total_edges', 0)}")
    print(f"  Pub-Sub Paths:    {gs.get('pub_sub_paths', 0)}")
    
    if gs.get('component_types'):
        types_str = ", ".join(f"{k}: {v}" for k, v in gs['component_types'].items())
        print(f"  Component Types:  {types_str}")
    
    # Layer metrics
    for layer_name, metrics in report.layer_metrics.items():
        print_subheader(f"Layer: {layer_name.title()}")
        
        # Event metrics
        print(f"\n  {CYAN}Event Simulation:{RESET}")
        print(f"    Throughput:        {metrics.event_throughput:.0f} messages")
        print(f"    Delivery Rate:     {metrics.event_delivery_rate:.1f}%")
        print(f"    Drop Rate:         {metrics.event_drop_rate:.1f}%")
        print(f"    Avg Latency:       {metrics.event_avg_latency:.3f} ms")
        
        # Failure metrics
        print(f"\n  {CYAN}Failure Simulation:{RESET}")
        print(f"    Avg Reach Loss:    {metrics.avg_reachability_loss * 100:.1f}%")
        print(f"    Avg Fragmentation: {metrics.avg_fragmentation * 100:.1f}%")
        print(f"    Max Impact:        {metrics.max_impact:.4f}")
        
        # Criticality
        print(f"\n  {CYAN}Criticality:{RESET}")
        print(f"    Critical:          {RED}{metrics.critical_count}{RESET}")
        print(f"    High:              {YELLOW}{metrics.high_count}{RESET}")
        print(f"    SPOFs:             {metrics.spof_count}")
    
    # Top critical components
    if report.top_critical:
        print_subheader("Top Critical Components")
        
        print(f"\n  {'-' * 70}")
        print(f"  {'ID':<15} {'Type':<12} {'Level':<10} {'Impact':<10} {'Cascade':<10}")
        print(f"  {'-' * 70}")
        
        for c in report.top_critical:
            lcolor = level_color(c['level'])
            print(
                f"  {c['id']:<15} {c['type']:<12} "
                f"{lcolor}{c['level']:<10}{RESET} "
                f"{c['combined_impact']:.4f}     "
                f"{c['cascade_count']:>3}"
            )
    
    # Recommendations
    if report.recommendations:
        print_subheader("Recommendations")
        for i, rec in enumerate(report.recommendations, 1):
            # Color code by severity
            if "CRITICAL" in rec:
                color = RED
            elif "HIGH" in rec or "SPOF" in rec:
                color = YELLOW
            else:
                color = GREEN
            
            print(f"\n  {i}. {color}{rec}{RESET}")
    
    # Summary
    print_header("Summary", "-")
    
    total_critical = sum(m.critical_count for m in report.layer_metrics.values())
    total_high = sum(m.high_count for m in report.layer_metrics.values())
    
    if total_critical > 0:
        print(f"\n  {RED}⚠ {total_critical} CRITICAL components require immediate attention{RESET}")
    if total_high > 0:
        print(f"  {YELLOW}⚠ {total_high} HIGH priority components should be reviewed{RESET}")
    if total_critical == 0 and total_high == 0:
        print(f"\n  {GREEN}✓ No critical issues detected{RESET}")


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Software-as-a-Graph Simulation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run event simulation
  python simulate_graph.py --event A1
  
  # Run failure simulation for node N1
  python simulate_graph.py --failure N1
  
  # Generate full simulation report
  python simulate_graph.py --report
  
  # Exhaustive failure analysis for infrastructure layer
  python simulate_graph.py --exhaustive --layer infrastructure
  
  # Export results to JSON
  python simulate_graph.py --report --output results/report.json
  
  # Neo4j with custom credentials
  python simulate_graph.py --user admin --password secret --report
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
        help="Generate comprehensive simulation report"
    )
    
    # Options
    options_group = parser.add_argument_group("Options")
    options_group.add_argument(
        "--layer", "-l",
        choices=["application", "infrastructure", "complete"],
        default="complete",
        help="Analysis layer (default: complete)"
    )
    options_group.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Output file path for JSON/CSV results"
    )
    options_group.add_argument(
        "--messages", "-m",
        type=int,
        default=100,
        help="Number of messages for event simulation (default: 100)"
    )
    options_group.add_argument(
        "--duration", "-d",
        type=float,
        default=10.0,
        help="Simulation duration in seconds (default: 10.0)"
    )
    options_group.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Cascade threshold for failure simulation (default: 0.5)"
    )
    options_group.add_argument(
        "--probability",
        type=float,
        default=0.7,
        help="Cascade probability for failure simulation (default: 0.7)"
    )
    
    # Neo4j credentials
    neo4j_group = parser.add_argument_group("Neo4j Credentials")
    neo4j_group.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI (default: bolt://localhost:7687)")
    neo4j_group.add_argument("--user", default="neo4j", help="Neo4j username (default: neo4j)")
    neo4j_group.add_argument("--password", default="password", help="Neo4j password")
    
    # Output control
    output_group = parser.add_argument_group("Output Control")
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Initialize simulator
    try:
        sim_kwargs = {}
        sim_kwargs["uri"] = args.uri
        sim_kwargs["user"] = args.user
        sim_kwargs["password"] = args.password
        
        with Simulator(**sim_kwargs) as sim:
            
            # === Event Simulation ===
            if args.event:
                result = sim.run_event_simulation(
                    source_app=args.event,
                    num_messages=args.messages,
                    duration=args.duration,
                )
                
                if args.json:
                    print(json.dumps(result.to_dict(), indent=2))
                else:
                    display_event_result(result)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result.to_dict(), f, indent=2)
                    if not args.quiet:
                        print(f"\nResults saved to: {args.output}")
            
            # === Failure Simulation ===
            elif args.failure:
                result = sim.run_failure_simulation(
                    target_id=args.failure,
                    layer=args.layer,
                    cascade_threshold=args.threshold,
                    cascade_probability=args.probability,
                )
                
                if args.json:
                    print(json.dumps(result.to_dict(), indent=2))
                else:
                    display_failure_result(result)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result.to_dict(), f, indent=2)
                    if not args.quiet:
                        print(f"\nResults saved to: {args.output}")
            
            # === Exhaustive Analysis ===
            elif args.exhaustive:
                results = sim.run_failure_simulation_exhaustive(
                    layer=args.layer,
                    cascade_threshold=args.threshold,
                    cascade_probability=args.probability,
                )
                
                if args.json:
                    data = [r.to_flat_dict() for r in results]
                    print(json.dumps(data, indent=2))
                else:
                    display_exhaustive_results(results)
                
                if args.output:
                    # Export as CSV if .csv extension, else JSON
                    if args.output.endswith('.csv'):
                        lines = ["component_id,component_type,impact_score,reachability_loss,cascade_count"]
                        for r in results:
                            lines.append(
                                f"{r.target_id},{r.target_type},"
                                f"{r.impact.composite_impact:.4f},"
                                f"{r.impact.reachability_loss:.4f},"
                                f"{r.impact.cascade_count}"
                            )
                        with open(args.output, 'w') as f:
                            f.write("\n".join(lines))
                    else:
                        with open(args.output, 'w') as f:
                            json.dump([r.to_dict() for r in results], f, indent=2)
                    
                    if not args.quiet:
                        print(f"\nResults saved to: {args.output}")
            
            # === Full Report ===
            elif args.report:
                report = sim.generate_report()
                
                if args.json:
                    print(json.dumps(report.to_dict(), indent=2))
                else:
                    display_simulation_report(report)
                
                if args.output:
                    sim.export_report(report, args.output)
                    if not args.quiet:
                        print(f"\nReport saved to: {args.output}")
    
    except FileNotFoundError as e:
        print(f"{RED}Error: {e}{RESET}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        if "neo4j" in str(e).lower():
            print(f"{RED}Error: Neo4j driver not installed.{RESET}", file=sys.stderr)
            print(f"Install with: pip install neo4j", file=sys.stderr)
        else:
            print(f"{RED}Error: Missing dependency: {e}{RESET}", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as e:
        print(f"{RED}Error: Failed to connect to Neo4j: {e}{RESET}", file=sys.stderr)
        print(f"Check that Neo4j is running and credentials are correct.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if "ServiceUnavailable" in str(type(e).__name__) or "Unable to retrieve routing" in str(e):
            print(f"{RED}Error: Cannot connect to Neo4j at {args.neo4j}{RESET}", file=sys.stderr)
            print(f"Check that Neo4j is running and the URI is correct.", file=sys.stderr)
        elif "AuthError" in str(type(e).__name__) or "authentication" in str(e).lower():
            print(f"{RED}Error: Neo4j authentication failed{RESET}", file=sys.stderr)
            print(f"Check username and password.", file=sys.stderr)
        else:
            logging.exception("Simulation failed")
            print(f"{RED}Error: {e}{RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()