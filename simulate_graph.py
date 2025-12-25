#!/usr/bin/env python3
"""
Graph Simulator CLI
====================

Simulates failures and message flow in distributed pub-sub systems.
Loads graph data directly from Neo4j.

Usage:
    # Single component failure
    python simulate_graph.py --component broker1
    
    # Multiple failures with cascade
    python simulate_graph.py --components app1 app2 --cascade
    
    # Event-driven simulation
    python simulate_graph.py --event-sim --duration 60000
    
    # Load testing
    python simulate_graph.py --load-test --peak-rate 1000
    
    # Chaos engineering
    python simulate_graph.py --chaos --failure-prob 0.01
    
    # Exhaustive campaign
    python simulate_graph.py --campaign
    
    # Targeted attack
    python simulate_graph.py --attack --strategy highest_degree --count 3

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from src.simulation import (
    Neo4jGraphLoader,
    SimulationGraph,
    FailureSimulator,
    EventDrivenSimulator,
    FailureType,
    FailureMode,
    AttackStrategy,
    QoSLevel,
    SimulationResult,
    BatchSimulationResult,
    EventSimulationResult,
    ComponentType
)


# ============================================================================
# Terminal Colors
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    @classmethod
    def disable(cls):
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED', 
                     'ENDC', 'BOLD', 'DIM']:
            setattr(cls, attr, '')


def print_header(text: str):
    """Print formatted header"""
    width = 70
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*width}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^{width}}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*width}{Colors.ENDC}")


def print_section(text: str):
    """Print section header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.DIM}{'-'*50}{Colors.ENDC}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠{Colors.ENDC} {text}")


def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.ENDC} {text}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ{Colors.ENDC} {text}")


def impact_color(score: float) -> str:
    """Get color based on impact score"""
    if score >= 0.7:
        return Colors.RED
    elif score >= 0.4:
        return Colors.YELLOW
    elif score >= 0.2:
        return Colors.CYAN
    return Colors.GREEN


# ============================================================================
# Output Functions
# ============================================================================

def print_graph_summary(graph: SimulationGraph):
    """Print graph summary"""
    print_section("GRAPH SUMMARY")
    
    stats = graph.to_dict()['stats']
    print(f"  Components:   {stats['component_count']}")
    print(f"  Dependencies: {stats['dependency_count']}")
    
    print("\n  By Type:")
    for ctype, count in stats['components_by_type'].items():
        if count > 0:
            print(f"    {ctype}: {count}")


def print_failure_result(result: SimulationResult, verbose: bool = False):
    """Print failure simulation result"""
    print_section("FAILURE SIMULATION RESULT")
    
    print(f"  Simulation ID: {result.simulation_id}")
    print(f"  Type: {result.simulation_type}")
    print(f"  Duration: {result.duration_ms:.2f} ms")
    
    # Failures
    print(f"\n  {Colors.RED}Failures:{Colors.ENDC}")
    print(f"    Primary:  {len(result.primary_failures)}")
    print(f"    Cascade:  {len(result.cascade_failures)}")
    print(f"    Total:    {len(result.primary_failures) + len(result.cascade_failures)}")
    
    if verbose and result.primary_failures:
        print(f"\n    Primary: {', '.join(result.primary_failures[:10])}")
    if verbose and result.cascade_failures:
        print(f"    Cascade: {', '.join(result.cascade_failures[:10])}")
    
    # Impact
    impact = result.impact
    ic = impact_color(impact.impact_score)
    print(f"\n  {Colors.CYAN}Impact:{Colors.ENDC}")
    print(f"    Reachability Loss: {impact.reachability_loss_pct:.1f}%")
    print(f"    Fragmentation:     +{impact.components_added} components")
    print(f"    Affected Deps:     {result.affected_dependencies}")
    print(f"    Impact Score:      {ic}{impact.impact_score:.4f}{Colors.ENDC}")
    
    if result.isolated_components:
        print(f"\n  Isolated Components: {len(result.isolated_components)}")
        if verbose:
            print(f"    {', '.join(result.isolated_components[:10])}")
    
    # Failure events
    if verbose and result.failure_events:
        print(f"\n  Failure Events:")
        for event in result.failure_events[:10]:
            cascade_mark = " [CASCADE]" if event.is_cascade else ""
            print(f"    • {event.component} ({event.component_type}){cascade_mark}")
            print(f"      {event.cause}")


def print_batch_result(result: BatchSimulationResult, verbose: bool = False):
    """Print batch simulation result"""
    print_section("BATCH SIMULATION RESULT")
    
    print(f"  Simulations: {result.simulation_count}")
    print(f"  Duration:    {result.total_duration_ms:.0f} ms")
    print(f"  Avg Impact:  {result.avg_impact_score:.4f}")
    print(f"  Max Impact:  {result.max_impact_score:.4f}")
    
    if result.critical_components:
        print(f"\n  {Colors.RED}Critical Components (Top 10):{Colors.ENDC}")
        for comp, score in result.critical_components[:10]:
            ic = impact_color(score)
            bar_len = int(score * 30)
            bar = '█' * bar_len
            print(f"    {comp:20} {ic}{bar}{Colors.ENDC} {score:.4f}")


def print_event_result(result: EventSimulationResult, verbose: bool = False):
    """Print event simulation result"""
    print_section("EVENT SIMULATION RESULT")
    
    print(f"  Simulation ID: {result.simulation_id}")
    print(f"  Duration:      {result.duration_ms:.0f} ms (simulated)")
    print(f"  Real Time:     {result.real_time_ms:.0f} ms")
    print(f"  Speedup:       {result.speedup:.0f}x")
    print(f"  Events:        {result.events_processed}")
    
    # Message metrics
    m = result.metrics
    print(f"\n  {Colors.CYAN}Messages:{Colors.ENDC}")
    print(f"    Published:     {m.messages_published}")
    print(f"    Delivered:     {m.messages_delivered}")
    print(f"    Failed:        {m.messages_failed}")
    print(f"    Timeout:       {m.messages_timeout}")
    
    rate = m.delivery_rate()
    rate_color = Colors.GREEN if rate >= 0.95 else Colors.YELLOW if rate >= 0.8 else Colors.RED
    print(f"    Delivery Rate: {rate_color}{rate:.1%}{Colors.ENDC}")
    
    # Latency
    print(f"\n  {Colors.CYAN}Latency:{Colors.ENDC}")
    print(f"    Average: {m.avg_latency():.2f} ms")
    print(f"    P99:     {m.p99_latency():.2f} ms")
    if m.latencies:
        print(f"    Min:     {min(m.latencies):.2f} ms")
        print(f"    Max:     {max(m.latencies):.2f} ms")
    
    # Throughput
    if result.duration_ms > 0:
        throughput = m.messages_published / (result.duration_ms / 1000)
        print(f"\n  {Colors.CYAN}Throughput:{Colors.ENDC}")
        print(f"    Messages/sec: {throughput:.1f}")
        print(f"    Bytes total:  {m.bytes_total:,}")
    
    # Failures
    if m.component_failures > 0:
        print(f"\n  {Colors.RED}Failures Injected:{Colors.ENDC}")
        print(f"    Component failures: {m.component_failures}")
    
    # Component stats
    if verbose and result.component_stats:
        print(f"\n  {Colors.CYAN}Component Statistics (Top 10):{Colors.ENDC}")
        sorted_stats = sorted(
            result.component_stats.values(),
            key=lambda s: s.messages_received,
            reverse=True
        )
        for stats in sorted_stats[:10]:
            print(f"    {stats.component_id}:")
            print(f"      Received: {stats.messages_received}, "
                  f"Sent: {stats.messages_sent}, "
                  f"Dropped: {stats.messages_dropped}")


def export_results(result: Any, output_dir: str, formats: List[str]):
    """Export results to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if 'json' in formats:
        json_file = output_path / f"simulation_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print_success(f"JSON exported: {json_file}")
    
    if 'summary' in formats:
        summary_file = output_path / f"simulation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Simulation Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            if isinstance(result, SimulationResult):
                f.write(f"Type: {result.simulation_type}\n")
                f.write(f"Primary Failures: {len(result.primary_failures)}\n")
                f.write(f"Cascade Failures: {len(result.cascade_failures)}\n")
                f.write(f"Impact Score: {result.impact.impact_score:.4f}\n")
            elif isinstance(result, BatchSimulationResult):
                f.write(f"Simulations: {result.simulation_count}\n")
                f.write(f"Avg Impact: {result.avg_impact_score:.4f}\n")
                f.write(f"Max Impact: {result.max_impact_score:.4f}\n")
            elif isinstance(result, EventSimulationResult):
                f.write(f"Duration: {result.duration_ms:.0f} ms\n")
                f.write(f"Messages: {result.metrics.messages_published}\n")
                f.write(f"Delivery Rate: {result.metrics.delivery_rate():.1%}\n")
        
        print_success(f"Summary exported: {summary_file}")


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Simulate failures and message flow in pub-sub systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single failure
    python simulate_graph.py --component broker_0
    
    # Multiple failures with cascade
    python simulate_graph.py --components app_1 app_2 --cascade
    
    # Event-driven simulation
    python simulate_graph.py --event-sim --duration 30000 --rate 100
    
    # Load testing
    python simulate_graph.py --load-test --peak-rate 500
    
    # Chaos engineering
    python simulate_graph.py --chaos --failure-prob 0.01
    
    # Targeted attack
    python simulate_graph.py --attack --strategy highest_degree --count 3
    
    # Exhaustive campaign
    python simulate_graph.py --campaign --component-types Application Broker
        """
    )
    
    # Neo4j connection
    conn_group = parser.add_argument_group('Neo4j Connection')
    conn_group.add_argument('--uri', default='bolt://localhost:7687',
                           help='Neo4j URI (default: bolt://localhost:7687)')
    conn_group.add_argument('--user', '-u', default='neo4j',
                           help='Neo4j username (default: neo4j)')
    conn_group.add_argument('--password', '-p', default='password',
                           help='Neo4j password (default: password)')
    conn_group.add_argument('--database', '-d', default='neo4j',
                           help='Neo4j database (default: neo4j)')
    
    # Simulation mode
    mode_group = parser.add_argument_group('Simulation Mode')
    mode_group.add_argument('--component', '-c',
                           help='Single component to fail')
    mode_group.add_argument('--components', nargs='+',
                           help='Multiple components to fail')
    mode_group.add_argument('--event-sim', action='store_true',
                           help='Run event-driven simulation')
    mode_group.add_argument('--load-test', action='store_true',
                           help='Run load testing simulation')
    mode_group.add_argument('--chaos', action='store_true',
                           help='Run chaos engineering simulation')
    mode_group.add_argument('--campaign', action='store_true',
                           help='Run exhaustive failure campaign')
    mode_group.add_argument('--attack', action='store_true',
                           help='Run targeted attack simulation')
    
    # Failure options
    fail_group = parser.add_argument_group('Failure Options')
    fail_group.add_argument('--cascade', action='store_true', default=True,
                           help='Enable cascading failures (default: True)')
    fail_group.add_argument('--no-cascade', action='store_true',
                           help='Disable cascading failures')
    fail_group.add_argument('--cascade-threshold', type=float, default=0.5,
                           help='Dependency loss ratio to trigger cascade (default: 0.5)')
    fail_group.add_argument('--cascade-prob', type=float, default=0.7,
                           help='Cascade probability (default: 0.7)')
    fail_group.add_argument('--max-cascade-depth', type=int, default=5,
                           help='Max cascade depth (default: 5)')
    
    # Attack options
    attack_group = parser.add_argument_group('Attack Options')
    attack_group.add_argument('--strategy', 
                             choices=['random', 'highest_degree', 'highest_weight',
                                     'highest_betweenness', 'brokers_first'],
                             default='highest_degree',
                             help='Attack strategy (default: highest_degree)')
    attack_group.add_argument('--count', type=int, default=3,
                             help='Number of components to attack (default: 3)')
    
    # Event simulation options
    event_group = parser.add_argument_group('Event Simulation Options')
    event_group.add_argument('--duration', type=float, default=10000,
                            help='Simulation duration in ms (default: 10000)')
    event_group.add_argument('--rate', type=float, default=100,
                            help='Message rate per second (default: 100)')
    event_group.add_argument('--initial-rate', type=float, default=10,
                            help='Initial rate for load test (default: 10)')
    event_group.add_argument('--peak-rate', type=float, default=500,
                            help='Peak rate for load test (default: 500)')
    event_group.add_argument('--ramp-time', type=float, default=5000,
                            help='Ramp time for load test (default: 5000)')
    event_group.add_argument('--failure-prob', type=float, default=0.01,
                            help='Failure probability for chaos (default: 0.01)')
    event_group.add_argument('--recovery-prob', type=float, default=0.1,
                            help='Recovery probability for chaos (default: 0.1)')
    
    # Campaign options
    campaign_group = parser.add_argument_group('Campaign Options')
    campaign_group.add_argument('--component-types', nargs='+',
                               choices=['Application', 'Broker', 'Node', 'Topic'],
                               help='Component types to test in campaign')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', '-o',
                             help='Output directory for results')
    output_group.add_argument('--format', nargs='+',
                             choices=['json', 'summary'],
                             default=['json'],
                             help='Output formats')
    
    # Display options
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument('--verbose', '-v', action='store_true',
                              help='Verbose output')
    display_group.add_argument('--quiet', '-q', action='store_true',
                              help='Minimal output')
    display_group.add_argument('--no-color', action='store_true',
                              help='Disable colored output')
    display_group.add_argument('--json-stdout', action='store_true',
                              help='Output JSON to stdout')
    display_group.add_argument('--seed', type=int,
                              help='Random seed for reproducibility')
    display_group.add_argument('--debug', action='store_true',
                              help='Enable debug logging')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)
    
    logger = logging.getLogger('simulate_graph')
    
    # Handle colors
    if not sys.stdout.isatty() or args.no_color:
        Colors.disable()
    
    # Print header
    if not args.quiet:
        print_header("PUB-SUB SYSTEM SIMULATOR")
        print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Neo4j: {args.uri}")
    
    try:
        # Load graph from Neo4j
        if not args.quiet:
            print_section("LOADING GRAPH FROM NEO4J")
        
        loader = Neo4jGraphLoader(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database
        )
        
        graph = loader.load_graph()
        loader.close()
        
        print_success(f"Loaded graph: {len(graph.components)} components, "
                     f"{len(graph.dependencies)} dependencies")
        
        if not args.quiet:
            print_graph_summary(graph)
        
        # Determine cascade setting
        enable_cascade = args.cascade and not args.no_cascade
        
        # Initialize result
        result = None
        
        # Run appropriate simulation
        if args.event_sim:
            # Event-driven simulation
            if not args.quiet:
                print_section("EVENT-DRIVEN SIMULATION")
                print_info(f"Duration: {args.duration}ms, Rate: {args.rate}/sec")
            
            simulator = EventDrivenSimulator(seed=args.seed)
            result = simulator.simulate(
                graph,
                duration_ms=args.duration,
                message_rate=args.rate
            )
            
            if not args.json_stdout and not args.quiet:
                print_event_result(result, verbose=args.verbose)
        
        elif args.load_test:
            # Load testing
            if not args.quiet:
                print_section("LOAD TESTING")
                print_info(f"Initial: {args.initial_rate}/sec, Peak: {args.peak_rate}/sec")
            
            simulator = EventDrivenSimulator(seed=args.seed)
            result = simulator.simulate_with_load_test(
                graph,
                duration_ms=args.duration,
                initial_rate=args.initial_rate,
                peak_rate=args.peak_rate,
                ramp_time_ms=args.ramp_time
            )
            
            if not args.json_stdout and not args.quiet:
                print_event_result(result, verbose=args.verbose)
        
        elif args.chaos:
            # Chaos engineering
            if not args.quiet:
                print_section("CHAOS ENGINEERING")
                print_info(f"Failure prob: {args.failure_prob}, Recovery: {args.recovery_prob}")
            
            simulator = EventDrivenSimulator(seed=args.seed)
            result = simulator.simulate_chaos(
                graph,
                duration_ms=args.duration,
                message_rate=args.rate,
                failure_probability=args.failure_prob,
                recovery_probability=args.recovery_prob
            )
            
            if not args.json_stdout and not args.quiet:
                print_event_result(result, verbose=args.verbose)
        
        elif args.campaign:
            # Exhaustive campaign
            if not args.quiet:
                print_section("EXHAUSTIVE FAILURE CAMPAIGN")
            
            simulator = FailureSimulator(
                cascade_threshold=args.cascade_threshold,
                cascade_probability=args.cascade_prob,
                max_cascade_depth=args.max_cascade_depth,
                seed=args.seed
            )
            
            component_types = None
            if args.component_types:
                component_types = [ComponentType(ct) for ct in args.component_types]
            
            result = simulator.simulate_all_single_failures(
                graph,
                component_types=component_types,
                enable_cascade=enable_cascade
            )
            
            if not args.json_stdout and not args.quiet:
                print_batch_result(result, verbose=args.verbose)
        
        elif args.attack:
            # Targeted attack
            if not args.quiet:
                print_section("TARGETED ATTACK SIMULATION")
                print_info(f"Strategy: {args.strategy}, Targets: {args.count}")
            
            simulator = FailureSimulator(
                cascade_threshold=args.cascade_threshold,
                cascade_probability=args.cascade_prob,
                max_cascade_depth=args.max_cascade_depth,
                seed=args.seed
            )
            
            strategy_map = {
                'random': AttackStrategy.RANDOM,
                'highest_degree': AttackStrategy.HIGHEST_DEGREE,
                'highest_weight': AttackStrategy.HIGHEST_WEIGHT,
                'highest_betweenness': AttackStrategy.HIGHEST_BETWEENNESS,
                'brokers_first': AttackStrategy.BROKERS_FIRST
            }
            
            result = simulator.simulate_targeted_attack(
                graph,
                strategy=strategy_map[args.strategy],
                count=args.count,
                enable_cascade=enable_cascade
            )
            
            if not args.json_stdout and not args.quiet:
                print_failure_result(result, verbose=args.verbose)
        
        elif args.components:
            # Multiple component failure
            if not args.quiet:
                print_section("MULTIPLE COMPONENT FAILURE")
                print_info(f"Failing: {', '.join(args.components)}")
            
            simulator = FailureSimulator(
                cascade_threshold=args.cascade_threshold,
                cascade_probability=args.cascade_prob,
                max_cascade_depth=args.max_cascade_depth,
                seed=args.seed
            )
            
            result = simulator.simulate_multiple_failures(
                graph,
                components=args.components,
                enable_cascade=enable_cascade
            )
            
            if not args.json_stdout and not args.quiet:
                print_failure_result(result, verbose=args.verbose)
        
        elif args.component:
            # Single component failure
            if not args.quiet:
                print_section("SINGLE COMPONENT FAILURE")
                print_info(f"Failing: {args.component}")
            
            simulator = FailureSimulator(
                cascade_threshold=args.cascade_threshold,
                cascade_probability=args.cascade_prob,
                max_cascade_depth=args.max_cascade_depth,
                seed=args.seed
            )
            
            result = simulator.simulate_single_failure(
                graph,
                component=args.component,
                enable_cascade=enable_cascade
            )
            
            if not args.json_stdout and not args.quiet:
                print_failure_result(result, verbose=args.verbose)
        
        else:
            # Default: run a basic event simulation
            if not args.quiet:
                print_section("DEFAULT SIMULATION")
                print_info("Running basic event simulation (use --help for options)")
            
            simulator = EventDrivenSimulator(seed=args.seed)
            result = simulator.simulate(
                graph,
                duration_ms=5000,
                message_rate=50
            )
            
            if not args.json_stdout and not args.quiet:
                print_event_result(result, verbose=args.verbose)
        
        # JSON stdout
        if args.json_stdout and result:
            print(json.dumps(result.to_dict(), indent=2))
        
        # Export if requested
        if args.output and result:
            if not args.quiet:
                print_section("EXPORTING RESULTS")
            export_results(result, args.output, args.format)
        
        if not args.quiet:
            print_success("\nSimulation complete!")
        
        sys.exit(0)
        
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_warning("Install with: pip install neo4j")
        sys.exit(1)
        
    except ValueError as e:
        print_error(f"Invalid input: {e}")
        sys.exit(1)
        
    except Exception as e:
        print_error(f"Simulation failed: {e}")
        if args.debug or args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()