#!/usr/bin/env python3
"""
Graph Simulation CLI

Comprehensive event-driven and failure simulation for distributed
pub-sub systems on the multi-layer graph model.

Operates directly on raw structural relationships retrieved from Neo4j:
    - PUBLISHES_TO, SUBSCRIBES_TO, USES     (app layer)
    - RUNS_ON, CONNECTS_TO                   (infra layer)
    - ROUTES, PUBLISHES_TO, SUBSCRIBES_TO    (mw layer)
    - All of the above                       (system layer)

Usage Examples:
    # Event simulation from a specific publisher
    python simulate_graph.py event --source App1 --messages 200 --layer app

    # Event simulation from all publishers
    python simulate_graph.py event --all --layer mw

    # Failure simulation for a single component
    python simulate_graph.py failure --target Broker1 --layer mw

    # Exhaustive failure analysis for a layer
    python simulate_graph.py failure --exhaustive --layer infra

    # Full multi-layer report
    python simulate_graph.py report --layers app,infra,mw,system -o report.json

    # Classify components by criticality
    python simulate_graph.py classify --layer system --top 20

    # Classify edges by criticality
    python simulate_graph.py classify --edges --layer app
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging

from src.config import Container


# =============================================================================
# Layer Descriptions (for help text)
# =============================================================================

LAYER_HELP = {
    "app": "Application layer: Apps, Libraries via PUBLISHES_TO/SUBSCRIBES_TO/USES",
    "infra": "Infrastructure layer: Nodes via RUNS_ON/CONNECTS_TO",
    "mw": "Middleware layer: Brokers via ROUTES/PUBLISHES_TO/SUBSCRIBES_TO",
    "system": "Complete system: all components and relationships",
}


# =============================================================================
# Argument Parsing
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    # Parent parser for common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    
    # --- Global options ---
    global_group = common_parser.add_argument_group("Neo4j Connection")
    global_group.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    global_group.add_argument("--user", "-u", default="neo4j", help="Neo4j username")
    global_group.add_argument("--password", "-p", default="password", help="Neo4j password")

    output_group = common_parser.add_argument_group("Output")
    output_group.add_argument("--output", "-o", metavar="FILE", help="Export results to JSON")
    output_group.add_argument("--json", action="store_true", help="Print JSON to stdout")
    output_group.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    output_group.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    # Main parser
    parser = argparse.ArgumentParser(
        prog="simulate_graph.py",
        description="Multi-layer simulation for distributed pub-sub systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<8} {v}" for k, v in LAYER_HELP.items()),
    )

    # --- Subcommands ---
    subs = parser.add_subparsers(dest="command", help="Simulation command")

    # event
    ev = subs.add_parser("event", help="Event-driven message flow simulation", parents=[common_parser])
    ev_target = ev.add_mutually_exclusive_group(required=True)
    ev_target.add_argument("--source", "-s", metavar="APP_ID", help="Source publisher app")
    ev_target.add_argument("--all", "-a", action="store_true", help="Simulate all publishers")
    ev.add_argument("--layer", "-l", choices=LAYER_HELP, default="system", help="Simulation layer")
    ev.add_argument("--messages", "-m", type=int, default=100, help="Messages per publisher")
    ev.add_argument("--duration", "-d", type=float, default=10.0, help="Duration (seconds)")

    # failure
    fl = subs.add_parser("failure", help="Component failure and cascade simulation", parents=[common_parser])
    fl_target = fl.add_mutually_exclusive_group(required=True)
    fl_target.add_argument("--target", "-t", metavar="COMP_ID", help="Target component to fail")
    fl_target.add_argument("--exhaustive", "-x", action="store_true", help="Fail every component in layer")
    fl.add_argument("--layer", "-l", choices=LAYER_HELP, default="system", help="Simulation layer")
    fl.add_argument("--cascade-prob", type=float, default=1.0, help="Cascade probability (0-1)")

    # report
    rp = subs.add_parser("report", help="Generate comprehensive multi-layer report", parents=[common_parser])
    rp.add_argument(
        "--layers", default="app,infra,mw,system",
        help="Comma-separated layers to include",
    )
    rp.add_argument("--edges", action="store_true", help="Include edge criticality")

    # classify
    cl = subs.add_parser("classify", help="Classify components or edges by criticality", parents=[common_parser])
    cl.add_argument("--layer", "-l", choices=LAYER_HELP, default="system", help="Layer to classify")
    cl.add_argument("--edges", action="store_true", help="Classify edges instead of components")
    cl.add_argument("--top", type=int, default=15, help="Show top N results")
    cl.add_argument("--k-factor", type=float, default=1.5, help="BoxPlot k-factor")

    return parser


# =============================================================================
# Command Handlers
# =============================================================================

def handle_event(args, sim, display) -> dict:
    """Handle the 'event' subcommand."""
    if args.all:
        results = sim.run_event_simulation_all(
            num_messages=args.messages,
            duration=args.duration,
            layer=args.layer,
        )
        if not args.quiet:
            display.print_header(f"Event Simulation: All Publishers ({args.layer} layer)")
            _display_event_summary(display, results, args.layer)
        return {app: r.to_dict() for app, r in results.items()}
    else:
        result = sim.run_event_simulation(
            source_app=args.source,
            num_messages=args.messages,
            duration=args.duration,
        )
        if not args.quiet:
            display.display_event_result(result)
        return result.to_dict()


def handle_failure(args, sim, display) -> dict:
    """Handle the 'failure' subcommand."""
    if args.exhaustive:
        results = sim.run_failure_simulation_exhaustive(
            layer=args.layer,
            cascade_probability=args.cascade_prob,
        )
        if not args.quiet:
            display.display_exhaustive_results(results)
        return [r.to_dict() for r in results]
    else:
        result = sim.run_failure_simulation(
            target_id=args.target,
            layer=args.layer,
            cascade_probability=args.cascade_prob,
        )
        if not args.quiet:
            display.display_failure_result(result)
        return result.to_dict()


def handle_report(args, sim, display) -> dict:
    """Handle the 'report' subcommand."""
    layers = [l.strip() for l in args.layers.split(",")]
    report = sim.generate_report(
        layers=layers,
        classify_edges=args.edges,
    )
    if not args.quiet:
        display.display_simulation_report(report)
    return report.to_dict()


def handle_classify(args, sim, display) -> dict:
    """Handle the 'classify' subcommand."""
    if args.edges:
        results = sim.classify_edges(layer=args.layer, k_factor=args.k_factor)
        if not args.quiet:
            _display_edge_classification(display, results, args.layer, args.top)
        return [e.to_dict() for e in results]
    else:
        results = sim.classify_components(layer=args.layer, k_factor=args.k_factor)
        if not args.quiet:
            _display_component_classification(display, results, args.layer, args.top)
        return [c.to_dict() for c in results]


# =============================================================================
# Display Helpers
# =============================================================================

def _display_event_summary(display, results, layer):
    """Display aggregated event simulation summary for all publishers."""
    total_pub = sum(r.metrics.messages_published for r in results.values())
    total_del = sum(r.metrics.messages_delivered for r in results.values())
    total_drop = sum(r.metrics.messages_dropped for r in results.values())

    print(f"\n  Layer:               {layer}")
    print(f"  Publishers Tested:   {len(results)}")
    print(f"  Total Published:     {total_pub}")
    print(f"  Total Delivered:     {display.colored(str(total_del), display.Colors.GREEN)}")
    drop_color = display.Colors.RED if total_drop > 0 else display.Colors.GRAY
    print(f"  Total Dropped:       {display.colored(str(total_drop), drop_color)}")

    if total_pub > 0:
        rate = total_del / total_pub * 100
        print(f"  Delivery Rate:       {rate:.1f}%")

    # Per-publisher breakdown (top 10 by throughput)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].metrics.messages_published,
        reverse=True,
    )
    display.print_subheader(f"Top Publishers (showing {min(10, len(sorted_results))})")
    print(f"\n  {'Publisher':<25} {'Published':<12} {'Delivered':<12} {'Dropped':<10} {'Rate':<8}")
    print(f"  {'-' * 67}")
    for app_id, r in sorted_results[:10]:
        m = r.metrics
        rate = m.delivery_rate
        color = display.Colors.GREEN if rate > 90 else (display.Colors.YELLOW if rate > 50 else display.Colors.RED)
        print(
            f"  {app_id:<25} {m.messages_published:<12} "
            f"{m.messages_delivered:<12} {m.messages_dropped:<10} "
            f"{display.colored(f'{rate:.1f}%', color)}"
        )


def _display_component_classification(display, results, layer, top_n):
    """Display component criticality classification."""
    display.print_header(f"Component Criticality: {layer.upper()} layer")

    # Summary counts
    counts = {}
    for c in results:
        counts[c.level] = counts.get(c.level, 0) + 1

    print(f"\n  Total Components:    {len(results)}")
    for level in ("critical", "high", "medium", "low", "minimal"):
        count = counts.get(level, 0)
        color = display.level_color(level)
        print(f"  {level.capitalize():<20} {display.colored(str(count), color)}")

    # Top N table
    display.print_subheader(f"Top {min(top_n, len(results))} Components")
    print(f"\n  {'#':<4} {'Component':<25} {'Type':<14} {'Combined':<10} {'Event':<10} {'Failure':<10} {'Level':<10}")
    print(f"  {'-' * 83}")
    for i, c in enumerate(results[:top_n], 1):
        color = display.level_color(c.level)
        print(
            f"  {i:<4} {c.id:<25} {c.type:<14} "
            f"{c.combined_impact:<10.4f} {c.event_impact:<10.4f} "
            f"{c.failure_impact:<10.4f} {display.colored(c.level, color)}"
        )


def _display_edge_classification(display, results, layer, top_n):
    """Display edge criticality classification."""
    display.print_header(f"Edge Criticality: {layer.upper()} layer")

    counts = {}
    for e in results:
        counts[e.level] = counts.get(e.level, 0) + 1

    print(f"\n  Total Edges Analyzed: {len(results)}")
    for level in ("critical", "high", "medium", "low", "minimal"):
        count = counts.get(level, 0)
        color = display.level_color(level)
        print(f"  {level.capitalize():<20} {display.colored(str(count), color)}")

    display.print_subheader(f"Top {min(top_n, len(results))} Edges")
    print(f"\n  {'#':<4} {'Source':<20} {'Target':<20} {'Relationship':<18} {'Impact':<10} {'Level':<10}")
    print(f"  {'-' * 82}")
    for i, e in enumerate(results[:top_n], 1):
        color = display.level_color(e.level)
        print(
            f"  {i:<4} {e.source:<20} {e.target:<20} "
            f"{e.relationship:<18} {e.combined_impact:<10.4f} "
            f"{display.colored(e.level, color)}"
        )


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Logging
    log_level = (
        logging.WARNING if args.quiet
        else logging.DEBUG if args.verbose
        else logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Initialize container and services
    container = Container(uri=args.uri, user=args.user, password=args.password)
    display = container.display_service()

    try:
        sim = container.simulation_service()

        # Dispatch to handler
        handlers = {
            "event": handle_event,
            "failure": handle_failure,
            "report": handle_report,
            "classify": handle_classify,
        }
        handler = handlers[args.command]
        result_data = handler(args, sim, display)

        # JSON stdout
        if args.json:
            print(json.dumps(result_data, indent=2))

        # File export
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result_data, f, indent=2)
            if not args.quiet:
                print(f"\n{display.colored(f'Results saved to: {args.output}', display.Colors.GREEN)}")

        return 0

    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
        return 130
    except Exception as e:
        print(display.colored(f"Error: {e}", display.Colors.RED), file=sys.stderr)
        if args.verbose:
            logging.exception("Simulation failed")
        return 1
    finally:
        container.close()


if __name__ == "__main__":
    sys.exit(main())