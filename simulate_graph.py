#!/usr/bin/env python3
"""
Graph Simulator CLI - Version 4.0

Simulates failures and message flow in pub-sub systems.
Works directly on graph model JSON files without Neo4j.

Simulation Modes:
1. Failure Simulation - Single/multiple component failures with cascade
2. Event-Driven Simulation - Message flow and delivery metrics
3. Load Testing - Ramp up message rate to find limits
4. Chaos Engineering - Random failures during message flow
5. Attack Simulation - Targeted component removal
6. Exhaustive Campaign - Test all components

Usage:
    # Single failure
    python simulate_graph.py --input system.json --fail broker_0
    
    # Multiple failures with cascade
    python simulate_graph.py --input system.json --fail app_1 app_2 --cascade
    
    # Event-driven simulation
    python simulate_graph.py --input system.json --event --duration 10000 --rate 100
    
    # Load testing
    python simulate_graph.py --input system.json --load-test --peak-rate 500
    
    # Chaos engineering
    python simulate_graph.py --input system.json --chaos --failure-prob 0.01
    
    # Exhaustive failure campaign
    python simulate_graph.py --input system.json --campaign

Requirements:
    - Graph JSON file (from generate_graph.py)
    
Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation import (
    SimulationGraph,
    FailureSimulator,
    EventSimulator,
    FailureType,
    AttackStrategy,
    QoSLevel,
    ComponentType,
)


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    @classmethod
    def disable(cls):
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED', 'END', 'BOLD', 'DIM']:
            setattr(cls, attr, '')


def use_colors() -> bool:
    import os
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and not os.getenv('NO_COLOR')


# =============================================================================
# Output Helpers
# =============================================================================

def print_header(text: str) -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")


def print_section(title: str) -> None:
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.DIM}{'-'*50}{Colors.END}")


def print_subsection(title: str) -> None:
    print(f"\n  {Colors.BLUE}{title}{Colors.END}")


def print_kv(key: str, value, indent: int = 2) -> None:
    print(f"{' '*indent}{Colors.DIM}{key}:{Colors.END} {value}")


def print_success(text: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_error(text: str) -> None:
    print(f"{Colors.RED}✗{Colors.END} {text}", file=sys.stderr)


def print_warning(text: str) -> None:
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def impact_color(score: float) -> str:
    """Get color based on impact score"""
    if score >= 0.5:
        return Colors.RED
    elif score >= 0.2:
        return Colors.YELLOW
    return Colors.GREEN


def rate_color(rate: float) -> str:
    """Get color based on delivery rate"""
    if rate >= 0.95:
        return Colors.GREEN
    elif rate >= 0.8:
        return Colors.YELLOW
    return Colors.RED


# =============================================================================
# Display Functions
# =============================================================================

def print_graph_summary(graph: SimulationGraph) -> None:
    """Print graph summary"""
    stats = graph.get_stats()
    
    print_section("Graph Summary")
    
    print_subsection("Components")
    for comp_type, count in stats["components"]["by_type"].items():
        print_kv(comp_type, count, indent=4)
    print_kv("Total", stats["components"]["total"], indent=4)
    
    print_subsection("Connections")
    for conn_type, count in stats["connections"]["by_type"].items():
        print_kv(conn_type, count, indent=4)
    print_kv("Total", stats["connections"]["total"], indent=4)
    
    print_subsection("Message Paths")
    print_kv("Total paths", stats["paths"]["total"], indent=4)


def print_failure_result(result, verbose: bool = False) -> None:
    """Print failure simulation result"""
    print_section("Failure Simulation Result")
    
    print_kv("Simulation ID", result.simulation_id)
    print_kv("Duration", f"{result.duration_ms:.0f} ms")
    
    # Failures
    print_subsection("Failures")
    print_kv("Primary", len(result.primary_failures), indent=4)
    print_kv("Cascade", len(result.cascade_failures), indent=4)
    print_kv("Total", len(result.primary_failures) + len(result.cascade_failures), indent=4)
    
    # Impact
    impact = result.impact
    ic = impact_color(impact.impact_score)
    
    print_subsection("Impact")
    print_kv("Impact Score", f"{ic}{impact.impact_score:.4f}{Colors.END}", indent=4)
    print_kv("Reachability Loss", f"{impact.reachability_loss_pct:.1f}%", indent=4)
    print_kv("Paths Lost", f"{impact.paths_lost} ({impact.paths_lost_pct:.1f}%)", indent=4)
    print_kv("Fragmentation", f"+{impact.fragmentation} components", indent=4)
    
    # Isolated components
    if result.isolated_components:
        print_kv("Isolated Components", len(result.isolated_components), indent=4)
    
    # Failure events (verbose)
    if verbose and result.failure_events:
        print_subsection("Failure Events")
        for event in result.failure_events[:15]:
            cascade = " [CASCADE]" if event.is_cascade else ""
            print(f"    • {event.component} ({event.component_type}){cascade}")
            print(f"      {event.cause}")


def print_batch_result(result, verbose: bool = False) -> None:
    """Print batch simulation result"""
    print_section("Batch Simulation Result")
    
    print_kv("Simulations", result.simulation_count)
    print_kv("Duration", f"{result.total_duration_ms:.0f} ms")
    
    print_subsection("Impact Statistics")
    print_kv("Average Impact", f"{result.avg_impact_score:.4f}", indent=4)
    print_kv("Maximum Impact", f"{result.max_impact_score:.4f}", indent=4)
    
    # Critical components
    if result.critical_components:
        print_subsection("Critical Components (Top 10)")
        for comp, score in result.critical_components[:10]:
            ic = impact_color(score)
            bar_len = int(score * 30)
            bar = '█' * bar_len
            print(f"    {comp:20} {ic}{bar}{Colors.END} {score:.4f}")


def print_event_result(result, verbose: bool = False) -> None:
    """Print event simulation result"""
    print_section("Event Simulation Result")
    
    print_kv("Simulation ID", result.simulation_id)
    print_kv("Duration", f"{result.duration_ms:.0f} ms (simulated)")
    print_kv("Real Time", f"{result.real_time_ms:.0f} ms")
    print_kv("Speedup", f"{result.speedup:.0f}x")
    print_kv("Events", result.events_processed)
    
    # Messages
    m = result.metrics
    print_subsection("Messages")
    print_kv("Published", m.messages_published, indent=4)
    print_kv("Delivered", m.messages_delivered, indent=4)
    print_kv("Failed", m.messages_failed, indent=4)
    print_kv("Timeout", m.messages_timeout, indent=4)
    
    rate = m.delivery_rate()
    rc = rate_color(rate)
    print_kv("Delivery Rate", f"{rc}{rate:.1%}{Colors.END}", indent=4)
    
    # Latency
    print_subsection("Latency")
    print_kv("Average", f"{m.avg_latency():.2f} ms", indent=4)
    print_kv("P99", f"{m.p99_latency():.2f} ms", indent=4)
    if m.latencies:
        print_kv("Min", f"{min(m.latencies):.2f} ms", indent=4)
        print_kv("Max", f"{max(m.latencies):.2f} ms", indent=4)
    
    # Throughput
    if result.duration_ms > 0:
        throughput = m.messages_published / (result.duration_ms / 1000)
        print_subsection("Throughput")
        print_kv("Messages/sec", f"{throughput:.1f}", indent=4)
        print_kv("Bytes Total", f"{m.bytes_total:,}", indent=4)
    
    # Failures
    if m.component_failures > 0:
        print_subsection("Failures During Simulation")
        print_kv("Component Failures", m.component_failures, indent=4)
    
    # Component stats (verbose)
    if verbose and result.component_stats:
        print_subsection("Component Statistics (Top 10 by messages)")
        sorted_stats = sorted(
            result.component_stats.items(),
            key=lambda x: x[1].messages_received + x[1].messages_sent,
            reverse=True,
        )
        for comp_id, stats in sorted_stats[:10]:
            print(f"    {comp_id}: sent={stats.messages_sent}, recv={stats.messages_received}")


# =============================================================================
# Export
# =============================================================================

def export_results(result, output_dir: Path, formats: List[str]) -> None:
    """Export results to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if "json" in formats:
        json_file = output_dir / f"simulation_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print_success(f"JSON exported: {json_file}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate failures and message flow in pub-sub systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single failure
    python simulate_graph.py --input system.json --fail broker_0
    
    # Multiple failures with cascade
    python simulate_graph.py --input system.json --fail app_1 app_2 --cascade
    
    # Event-driven simulation
    python simulate_graph.py --input system.json --event --duration 10000
    
    # Load testing
    python simulate_graph.py --input system.json --load-test --peak-rate 500
    
    # Chaos engineering
    python simulate_graph.py --input system.json --chaos
    
    # Targeted attack
    python simulate_graph.py --input system.json --attack --strategy highest_degree --count 3
    
    # Exhaustive campaign
    python simulate_graph.py --input system.json --campaign
        """,
    )
    
    # Input
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Input graph JSON file",
    )
    
    # Simulation mode
    mode = parser.add_argument_group("Simulation Mode")
    mode.add_argument(
        "--fail", "-f", nargs="+",
        help="Component(s) to fail",
    )
    mode.add_argument(
        "--event", "-e", action="store_true",
        help="Run event-driven simulation",
    )
    mode.add_argument(
        "--load-test", action="store_true",
        help="Run load testing simulation",
    )
    mode.add_argument(
        "--chaos", action="store_true",
        help="Run chaos engineering simulation",
    )
    mode.add_argument(
        "--attack", action="store_true",
        help="Run targeted attack simulation",
    )
    mode.add_argument(
        "--campaign", action="store_true",
        help="Run exhaustive failure campaign",
    )
    
    # Failure options
    fail_opts = parser.add_argument_group("Failure Options")
    fail_opts.add_argument(
        "--cascade", action="store_true", default=True,
        help="Enable cascade propagation (default: True)",
    )
    fail_opts.add_argument(
        "--no-cascade", dest="cascade", action="store_false",
        help="Disable cascade propagation",
    )
    fail_opts.add_argument(
        "--cascade-threshold", type=float, default=0.5,
        help="Cascade threshold (default: 0.5)",
    )
    fail_opts.add_argument(
        "--cascade-prob", type=float, default=0.7,
        help="Cascade probability (default: 0.7)",
    )
    fail_opts.add_argument(
        "--max-cascade", type=int, default=5,
        help="Maximum cascade depth (default: 5)",
    )
    
    # Attack options
    attack_opts = parser.add_argument_group("Attack Options")
    attack_opts.add_argument(
        "--strategy", type=str, default="highest_degree",
        choices=["random", "highest_degree", "brokers_first", "nodes_first"],
        help="Attack strategy (default: highest_degree)",
    )
    attack_opts.add_argument(
        "--count", type=int, default=1,
        help="Number of components to attack (default: 1)",
    )
    
    # Campaign options
    campaign_opts = parser.add_argument_group("Campaign Options")
    campaign_opts.add_argument(
        "--component-types", nargs="+",
        choices=["Application", "Broker", "Topic", "Node"],
        help="Component types to include in campaign",
    )
    
    # Event simulation options
    event_opts = parser.add_argument_group("Event Simulation Options")
    event_opts.add_argument(
        "--duration", type=float, default=10000,
        help="Simulation duration in ms (default: 10000)",
    )
    event_opts.add_argument(
        "--rate", type=float, default=100,
        help="Message rate per second (default: 100)",
    )
    event_opts.add_argument(
        "--qos", type=int, default=1, choices=[0, 1, 2],
        help="QoS level: 0=at-most-once, 1=at-least-once, 2=exactly-once",
    )
    
    # Load test options
    load_opts = parser.add_argument_group("Load Test Options")
    load_opts.add_argument(
        "--initial-rate", type=float, default=10,
        help="Initial message rate (default: 10)",
    )
    load_opts.add_argument(
        "--peak-rate", type=float, default=500,
        help="Peak message rate (default: 500)",
    )
    load_opts.add_argument(
        "--ramp-time", type=float, default=10000,
        help="Ramp up time in ms (default: 10000)",
    )
    
    # Chaos options
    chaos_opts = parser.add_argument_group("Chaos Options")
    chaos_opts.add_argument(
        "--failure-prob", type=float, default=0.01,
        help="Failure probability per check (default: 0.01)",
    )
    chaos_opts.add_argument(
        "--recovery-prob", type=float, default=0.1,
        help="Recovery probability per check (default: 0.1)",
    )
    
    # Output options
    output_opts = parser.add_argument_group("Output Options")
    output_opts.add_argument(
        "--output", "-o", type=Path,
        help="Output directory for results",
    )
    output_opts.add_argument(
        "--format", nargs="+", default=["json"],
        help="Output formats (default: json)",
    )
    output_opts.add_argument(
        "--json", action="store_true",
        help="Output JSON to stdout",
    )
    output_opts.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    output_opts.add_argument(
        "--quiet", "-q", action="store_true",
        help="Minimal output",
    )
    output_opts.add_argument(
        "--no-color", action="store_true",
        help="Disable colors",
    )
    output_opts.add_argument(
        "--seed", type=int,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Handle colors
    if args.no_color or not use_colors():
        Colors.disable()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    
    result = None
    
    try:
        if not args.quiet:
            print_header("Pub-Sub System Simulator")
        
        # Load graph
        if not args.quiet:
            print_section("Loading Graph")
            print_kv("File", args.input)
        
        if not args.input.exists():
            print_error(f"File not found: {args.input}")
            return 1
        
        graph = SimulationGraph.from_json(args.input)
        
        if not args.quiet:
            print_success(f"Loaded: {len(graph.components)} components, {len(graph.connections)} connections")
            print_graph_summary(graph)
        
        # Determine mode and run simulation
        if args.fail:
            # Failure simulation
            if not args.quiet:
                print_section("Failure Simulation")
                print_kv("Components", ", ".join(args.fail))
                print_kv("Cascade", "Enabled" if args.cascade else "Disabled")
            
            simulator = FailureSimulator(
                cascade_threshold=args.cascade_threshold,
                cascade_probability=args.cascade_prob,
                max_cascade_depth=args.max_cascade,
                seed=args.seed,
            )
            
            if len(args.fail) == 1:
                result = simulator.simulate_failure(
                    graph, args.fail[0], enable_cascade=args.cascade
                )
            else:
                result = simulator.simulate_multiple_failures(
                    graph, args.fail, enable_cascade=args.cascade
                )
            
            if not args.json and not args.quiet:
                print_failure_result(result, verbose=args.verbose)
        
        elif args.attack:
            # Attack simulation
            if not args.quiet:
                print_section("Attack Simulation")
                print_kv("Strategy", args.strategy)
                print_kv("Count", args.count)
            
            simulator = FailureSimulator(
                cascade_threshold=args.cascade_threshold,
                cascade_probability=args.cascade_prob,
                max_cascade_depth=args.max_cascade,
                seed=args.seed,
            )
            
            strategy = AttackStrategy(args.strategy)
            result = simulator.simulate_attack(
                graph, strategy, count=args.count, enable_cascade=args.cascade
            )
            
            if not args.json and not args.quiet:
                print_failure_result(result, verbose=args.verbose)
        
        elif args.campaign:
            # Exhaustive campaign
            if not args.quiet:
                print_section("Exhaustive Failure Campaign")
            
            simulator = FailureSimulator(
                cascade_threshold=args.cascade_threshold,
                cascade_probability=args.cascade_prob,
                max_cascade_depth=args.max_cascade,
                seed=args.seed,
            )
            
            comp_types = None
            if args.component_types:
                comp_types = [ComponentType(t) for t in args.component_types]
            
            result = simulator.simulate_all_failures(
                graph, component_types=comp_types, enable_cascade=args.cascade
            )
            
            if not args.json and not args.quiet:
                print_batch_result(result, verbose=args.verbose)
        
        elif args.load_test:
            # Load testing
            if not args.quiet:
                print_section("Load Testing")
                print_kv("Initial Rate", f"{args.initial_rate}/sec")
                print_kv("Peak Rate", f"{args.peak_rate}/sec")
                print_kv("Ramp Time", f"{args.ramp_time} ms")
            
            simulator = EventSimulator(seed=args.seed)
            result = simulator.simulate_load_test(
                graph,
                duration_ms=args.duration,
                initial_rate=args.initial_rate,
                peak_rate=args.peak_rate,
                ramp_time_ms=args.ramp_time,
                qos=QoSLevel(args.qos),
            )
            
            if not args.json and not args.quiet:
                print_event_result(result, verbose=args.verbose)
        
        elif args.chaos:
            # Chaos engineering
            if not args.quiet:
                print_section("Chaos Engineering")
                print_kv("Failure Prob", args.failure_prob)
                print_kv("Recovery Prob", args.recovery_prob)
            
            simulator = EventSimulator(seed=args.seed)
            result = simulator.simulate_chaos(
                graph,
                duration_ms=args.duration,
                message_rate=args.rate,
                failure_probability=args.failure_prob,
                recovery_probability=args.recovery_prob,
                qos=QoSLevel(args.qos),
            )
            
            if not args.json and not args.quiet:
                print_event_result(result, verbose=args.verbose)
        
        elif args.event:
            # Event-driven simulation
            if not args.quiet:
                print_section("Event-Driven Simulation")
                print_kv("Duration", f"{args.duration} ms")
                print_kv("Rate", f"{args.rate}/sec")
                print_kv("QoS", args.qos)
            
            simulator = EventSimulator(seed=args.seed)
            result = simulator.simulate(
                graph,
                duration_ms=args.duration,
                message_rate=args.rate,
                qos=QoSLevel(args.qos),
            )
            
            if not args.json and not args.quiet:
                print_event_result(result, verbose=args.verbose)
        
        else:
            # Default: run basic event simulation
            if not args.quiet:
                print_section("Default Event Simulation")
                print_kv("Duration", "5000 ms")
                print_kv("Rate", "50/sec")
            
            simulator = EventSimulator(seed=args.seed)
            result = simulator.simulate(
                graph,
                duration_ms=5000,
                message_rate=50,
            )
            
            if not args.json and not args.quiet:
                print_event_result(result, verbose=args.verbose)
        
        # JSON output
        if args.json and result:
            print(json.dumps(result.to_dict(), indent=2))
        
        # Export
        if args.output and result:
            if not args.quiet:
                print_section("Exporting Results")
            export_results(result, args.output, args.format)
        
        if not args.quiet:
            print_success("\nSimulation complete!")
        
        return 0
    
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    
    except ValueError as e:
        print_error(f"Invalid input: {e}")
        return 1
    
    except KeyboardInterrupt:
        print_warning("\nSimulation interrupted")
        return 130
    
    except Exception as e:
        print_error(f"Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())