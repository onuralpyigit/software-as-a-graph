#!/usr/bin/env python3
"""
Pub-Sub System Simulator
=========================

Comprehensive simulation tool for distributed pub-sub systems.
Supports both failure simulation and event-driven message flow simulation.

Usage:
    # Single component failure
    python simulate_graph.py --input system.json --component broker1
    
    # Multiple failures with cascade
    python simulate_graph.py --input system.json --components app1 app2 --cascade
    
    # Event-driven simulation
    python simulate_graph.py --input system.json --event-sim --duration 60000
    
    # Load testing
    python simulate_graph.py --input system.json --load-test --peak-rate 1000
    
    # Chaos engineering
    python simulate_graph.py --input system.json --chaos --failure-prob 0.1
    
    # Exhaustive campaign
    python simulate_graph.py --input system.json --campaign

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

import networkx as nx

from src.simulation import (
    FailureSimulator,
    EventDrivenSimulator,
    FailureType,
    FailureMode,
    AttackStrategy,
    SimulationResult,
    BatchSimulationResult,
    EventSimulationResult
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
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    _enabled = True
    
    @classmethod
    def disable(cls):
        cls._enabled = False
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'WARNING', 'RED', 
                     'ENDC', 'BOLD', 'DIM']:
            setattr(cls, attr, '')


def print_header(text: str):
    """Print header"""
    width = 70
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(width)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * width}{Colors.ENDC}")


def print_section(title: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{title}{Colors.ENDC}")
    print("-" * 50)


def print_success(msg: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {msg}")


def print_error(msg: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.ENDC} {msg}")


def print_warning(msg: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {msg}")


def print_info(msg: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.ENDC} {msg}")


# ============================================================================
# Graph Loading
# ============================================================================

def load_graph_from_json(filepath: str) -> nx.DiGraph:
    """Load graph from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes (infrastructure)
    for node in data.get('nodes', []):
        node_id = node.get('id', node.get('name'))
        G.add_node(node_id, type='Node', **node)
        
    # Add brokers 
    for broker in data.get('brokers', []):
        broker_id = broker.get('id', broker.get('name'))
        G.add_node(broker_id, type='Broker', **broker)
        
    # Add topics
    for topic in data.get('topics', []):
        topic_id = topic.get('id', topic.get('name'))
        G.add_node(topic_id, type='Topic', **topic)
        
    # Add applications
    for app in data.get('applications', []):
        app_id = app.get('id', app.get('name'))
        G.add_node(app_id, type='Application', **app)
        
    # Process relationships
    relationships = data.get('relationships', {})

    # PUBLISHES_TO relationships
    for rel in relationships.get('publishes_to', data.get('publishes', [])):
        app_id = rel.get('from', rel.get('source', rel.get('app')))
        topic_id = rel.get('to', rel.get('target', rel.get('topic')))
        G.add_edge(app_id, topic_id, type='PUBLISHES_TO', **rel)

    # SUBSCRIBES_TO relationships
    for rel in relationships.get('subscribes_to', data.get('subscribes', [])):
        app_id = rel.get('from', rel.get('source', rel.get('app')))
        topic_id = rel.get('to', rel.get('target', rel.get('topic')))
        G.add_edge(topic_id, app_id, type='SUBSCRIBES_TO', **rel)

    # ROUTES relationships
    for rel in relationships.get('routes', data.get('routes', [])):
        topic_id = rel.get('to', rel.get('target', rel.get('topic')))
        broker_id = rel.get('from', rel.get('source', rel.get('broker')))
        G.add_edge(broker_id, topic_id, type='ROUTES', **rel)

    # RUNS_ON relationships
    for rel in relationships.get('runs_on', data.get('runs', [])):
        comp_id = rel.get('from', rel.get('source', rel.get('component')))
        node_id = rel.get('to', rel.get('target', rel.get('node')))
        G.add_edge(comp_id, node_id, type='RUNS_ON', **rel)

    # CONNECTS_TO relationships
    for rel in relationships.get('connects_to', data.get('connects', [])):
        src_id = rel.get('from', rel.get('source', rel.get('source_component')))
        dst_id = rel.get('to', rel.get('target', rel.get('target_component')))
        G.add_edge(src_id, dst_id, type='CONNECTS_TO', **rel)
    
    return G


# ============================================================================
# Result Printing
# ============================================================================

def print_failure_result(result: SimulationResult, verbose: bool = False):
    """Print failure simulation result"""
    print_section("FAILURE SIMULATION RESULTS")
    
    print(f"\n  {Colors.BOLD}Simulation:{Colors.ENDC} {result.simulation_id}")
    print(f"  {Colors.BOLD}Type:{Colors.ENDC} {result.simulation_type}")
    print(f"  {Colors.BOLD}Duration:{Colors.ENDC} {result.duration_ms:.2f}ms")
    
    # Failures
    print(f"\n  {Colors.BOLD}Failures:{Colors.ENDC}")
    print(f"    Primary: {len(result.primary_failures)}")
    if result.primary_failures:
        for comp in result.primary_failures[:5]:
            print(f"      - {comp}")
        if len(result.primary_failures) > 5:
            print(f"      ... and {len(result.primary_failures) - 5} more")
    
    if result.cascade_failures:
        print(f"    Cascade: {len(result.cascade_failures)}")
        for comp in result.cascade_failures[:5]:
            print(f"      - {comp}")
        if len(result.cascade_failures) > 5:
            print(f"      ... and {len(result.cascade_failures) - 5} more")
    
    # Impact
    impact = result.impact
    print(f"\n  {Colors.BOLD}Impact:{Colors.ENDC}")
    print(f"    Reachability Loss: {Colors.WARNING}{impact.reachability_loss*100:.1f}%{Colors.ENDC}")
    print(f"    Nodes Affected: {len(impact.affected_nodes)}")
    print(f"    Fragmentation: {impact.fragmentation} new components")
    
    if impact.affected_applications:
        print(f"    Applications: {len(impact.affected_applications)}")
    if impact.affected_brokers:
        print(f"    Brokers: {len(impact.affected_brokers)}")
    if impact.affected_topics:
        print(f"    Topics: {len(impact.affected_topics)}")
    
    # Scores
    print(f"\n  {Colors.BOLD}Scores:{Colors.ENDC}")
    
    impact_color = Colors.GREEN if result.impact_score < 0.3 else \
                   Colors.WARNING if result.impact_score < 0.6 else Colors.RED
    resilience_color = Colors.RED if result.resilience_score < 0.4 else \
                       Colors.WARNING if result.resilience_score < 0.7 else Colors.GREEN
    
    print(f"    Impact Score: {impact_color}{result.impact_score:.4f}{Colors.ENDC}")
    print(f"    Resilience Score: {resilience_color}{result.resilience_score:.4f}{Colors.ENDC}")
    
    if result.spof_triggered:
        print(f"\n  {Colors.RED}⚠ SPOF TRIGGERED{Colors.ENDC}")
    
    if verbose:
        print(f"\n  {Colors.BOLD}Graph State:{Colors.ENDC}")
        print(f"    Original: {result.original_nodes} nodes, {result.original_edges} edges")
        print(f"    Remaining: {result.remaining_nodes} nodes, {result.remaining_edges} edges")
        print(f"    Lost: {result.original_nodes - result.remaining_nodes} nodes, "
              f"{result.original_edges - result.remaining_edges} edges")


def print_batch_result(result: BatchSimulationResult, verbose: bool = False):
    """Print batch simulation result"""
    print_section("BATCH SIMULATION RESULTS")
    
    print(f"\n  {Colors.BOLD}Summary:{Colors.ENDC}")
    print(f"    Total Simulations: {result.total_simulations}")
    print(f"    Completed: {result.completed_simulations}")
    print(f"    Failed: {result.failed_simulations}")
    print(f"    Duration: {result.total_duration_ms:.0f}ms ({result.avg_duration_ms:.2f}ms avg)")
    
    print(f"\n  {Colors.BOLD}Impact Statistics:{Colors.ENDC}")
    print(f"    Average: {result.avg_impact_score:.4f}")
    print(f"    Maximum: {result.max_impact_score:.4f}")
    print(f"    Minimum: {result.min_impact_score:.4f}")
    
    if result.most_critical:
        print(f"\n  {Colors.BOLD}Most Critical Components:{Colors.ENDC}")
        for i, (comp, score) in enumerate(result.most_critical[:10], 1):
            color = Colors.RED if score > 0.5 else Colors.WARNING if score > 0.3 else Colors.GREEN
            print(f"    {i:2d}. {comp:30s} {color}{score:.4f}{Colors.ENDC}")
    
    if verbose and result.least_critical:
        print(f"\n  {Colors.BOLD}Least Critical Components:{Colors.ENDC}")
        for i, (comp, score) in enumerate(result.least_critical[:5], 1):
            print(f"    {i:2d}. {comp:30s} {score:.4f}")


def print_event_result(result: EventSimulationResult, verbose: bool = False):
    """Print event simulation result"""
    print_section("EVENT SIMULATION RESULTS")
    
    print(f"\n  {Colors.BOLD}Simulation:{Colors.ENDC} {result.simulation_id}")
    print(f"  {Colors.BOLD}Duration:{Colors.ENDC} {result.duration_ms:.0f}ms (simulated)")
    print(f"  {Colors.BOLD}Real Time:{Colors.ENDC} {result.real_time_ms:.0f}ms")
    print(f"  {Colors.BOLD}Speedup:{Colors.ENDC} {Colors.CYAN}{result.speedup:.0f}x{Colors.ENDC} real-time")
    
    m = result.metrics
    
    # Messages
    print(f"\n  {Colors.BOLD}Messages:{Colors.ENDC}")
    print(f"    Total: {m.total_messages}")
    
    delivery_color = Colors.GREEN if m.delivery_rate > 0.95 else \
                    Colors.WARNING if m.delivery_rate > 0.8 else Colors.RED
    print(f"    Delivered: {m.delivered_messages} ({delivery_color}{m.delivery_rate*100:.1f}%{Colors.ENDC})")
    
    if m.dropped_messages > 0:
        print(f"    Dropped: {Colors.RED}{m.dropped_messages}{Colors.ENDC}")
    if m.expired_messages > 0:
        print(f"    Expired: {Colors.WARNING}{m.expired_messages}{Colors.ENDC}")
    
    # Latency
    print(f"\n  {Colors.BOLD}Latency:{Colors.ENDC}")
    print(f"    Average: {m.avg_latency:.2f}ms")
    print(f"    P50: {m.p50_latency:.2f}ms")
    print(f"    P95: {m.p95_latency:.2f}ms")
    print(f"    P99: {m.p99_latency:.2f}ms")
    if m.max_latency > 0:
        print(f"    Max: {m.max_latency:.2f}ms")
    
    # Throughput
    print(f"\n  {Colors.BOLD}Throughput:{Colors.ENDC}")
    print(f"    {m.messages_per_second:.0f} messages/second")
    
    # Failures
    if m.component_failures > 0:
        print(f"\n  {Colors.BOLD}Failures:{Colors.ENDC}")
        print(f"    Component Failures: {Colors.RED}{m.component_failures}{Colors.ENDC}")
    
    if verbose:
        print(f"\n  {Colors.BOLD}Events:{Colors.ENDC}")
        print(f"    Total: {result.total_events}")
        for event_type, count in sorted(result.events_by_type.items()):
            print(f"    {event_type}: {count}")
        
        if result.component_stats:
            print(f"\n  {Colors.BOLD}Component Stats (top 5 by throughput):{Colors.ENDC}")
            sorted_stats = sorted(
                result.component_stats.values(),
                key=lambda s: s.throughput,
                reverse=True
            )[:5]
            for stats in sorted_stats:
                print(f"    {stats.component_id}: {stats.throughput:.0f} msg/s, "
                      f"recv={stats.messages_received}, sent={stats.messages_sent}")


# ============================================================================
# Main CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Simulate pub-sub systems with failure and event-driven simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single component failure
    python simulate_graph.py --input system.json --component broker1
    
    # Multiple failures with cascade
    python simulate_graph.py --input system.json --components app1 app2 --cascade
    
    # Event-driven simulation
    python simulate_graph.py --input system.json --event-sim --duration 60000
    
    # Load testing
    python simulate_graph.py --input system.json --load-test --peak-rate 1000
    
    # Chaos engineering
    python simulate_graph.py --input system.json --chaos --failure-prob 0.1
    
    # Exhaustive failure campaign
    python simulate_graph.py --input system.json --campaign
    
    # Targeted attack simulation
    python simulate_graph.py --input system.json --attack --strategy criticality --count 5
        """
    )
    
    # Input
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--input', '-i', required=True,
                            help='Input JSON file with pub-sub system data')
    
    # Simulation mode
    mode_group = parser.add_argument_group('Simulation Mode')
    mode_group.add_argument('--component', '-c',
                           help='Single component to fail')
    mode_group.add_argument('--components', '-C', nargs='+',
                           help='Multiple components to fail')
    mode_group.add_argument('--campaign', action='store_true',
                           help='Run exhaustive simulation (test all components)')
    mode_group.add_argument('--attack', action='store_true',
                           help='Targeted attack simulation')
    mode_group.add_argument('--random', action='store_true',
                           help='Random failure simulation')
    mode_group.add_argument('--event-sim', action='store_true',
                           help='Event-driven message simulation')
    mode_group.add_argument('--load-test', action='store_true',
                           help='Load testing with ramping rate')
    mode_group.add_argument('--chaos', action='store_true',
                           help='Chaos engineering mode')
    
    # Failure options
    failure_group = parser.add_argument_group('Failure Options')
    failure_group.add_argument('--cascade', action='store_true', default=True,
                              help='Enable cascading failures (default: True)')
    failure_group.add_argument('--no-cascade', action='store_true',
                              help='Disable cascading failures')
    failure_group.add_argument('--cascade-threshold', type=float, default=0.7,
                              help='Cascade trigger threshold (default: 0.7)')
    failure_group.add_argument('--cascade-prob', type=float, default=0.5,
                              help='Cascade probability (default: 0.5)')
    failure_group.add_argument('--max-cascade-depth', type=int, default=5,
                              help='Maximum cascade depth (default: 5)')
    
    # Attack options
    attack_group = parser.add_argument_group('Attack Options')
    attack_group.add_argument('--strategy', choices=['random', 'criticality', 'betweenness', 
                                                     'degree', 'articulation'],
                             default='criticality',
                             help='Attack targeting strategy (default: criticality)')
    attack_group.add_argument('--count', type=int, default=5,
                             help='Number of components to target (default: 5)')
    
    # Random failure options
    random_group = parser.add_argument_group('Random Failure Options')
    random_group.add_argument('--failure-prob', type=float, default=0.1,
                             help='Failure probability per component (default: 0.1)')
    random_group.add_argument('--component-types', nargs='+',
                             help='Limit to specific component types')
    
    # Event simulation options
    event_group = parser.add_argument_group('Event Simulation Options')
    event_group.add_argument('--duration', type=float, default=10000,
                            help='Simulation duration in ms (default: 10000)')
    event_group.add_argument('--message-rate', type=float, default=100,
                            help='Message rate per second (default: 100)')
    event_group.add_argument('--failure-at', type=float, nargs='+',
                            help='Inject failures at specified times (ms)')
    event_group.add_argument('--failure-component', nargs='+',
                            help='Components to fail during event simulation')
    
    # Load test options
    load_group = parser.add_argument_group('Load Test Options')
    load_group.add_argument('--initial-rate', type=float, default=10,
                           help='Initial message rate (default: 10)')
    load_group.add_argument('--peak-rate', type=float, default=1000,
                           help='Peak message rate (default: 1000)')
    load_group.add_argument('--ramp-time', type=float, default=10000,
                           help='Time to ramp to peak (default: 10000ms)')
    
    # Chaos options
    chaos_group = parser.add_argument_group('Chaos Options')
    chaos_group.add_argument('--recovery-prob', type=float, default=0.3,
                            help='Recovery probability (default: 0.3)')
    chaos_group.add_argument('--check-interval', type=float, default=1000,
                            help='Failure check interval (default: 1000ms)')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', '-o',
                             help='Output directory for results')
    output_group.add_argument('--export-json',
                             help='Export results to JSON file')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='Minimal output')
    output_group.add_argument('--no-color', action='store_true',
                             help='Disable colored output')
    
    # Other
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup
    if args.no_color:
        Colors.disable()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)
    
    if not args.quiet:
        print_header("PUB-SUB SYSTEM SIMULATOR")
        print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Input: {args.input}")
    
    try:
        # Load graph
        if not args.quiet:
            print_section("LOADING GRAPH")
        
        graph = load_graph_from_json(args.input)
        print_success(f"Loaded graph: {graph.number_of_nodes()} nodes, "
                     f"{graph.number_of_edges()} edges")
        
        # Determine cascade setting
        enable_cascade = args.cascade and not args.no_cascade
        
        # Run appropriate simulation
        result = None
        
        if args.event_sim:
            # Event-driven simulation
            if not args.quiet:
                print_section("EVENT-DRIVEN SIMULATION")
                print_info(f"Duration: {args.duration}ms, Rate: {args.message_rate}/sec")
            
            simulator = EventDrivenSimulator(seed=args.seed)
            
            # Build failure schedule if specified
            failure_schedule = []
            if args.failure_at and args.failure_component:
                for time_ms in args.failure_at:
                    for comp in args.failure_component:
                        failure_schedule.append({
                            'time_ms': time_ms,
                            'component': comp,
                            'duration_ms': 5000
                        })
            
            result = simulator.simulate(
                graph,
                duration_ms=args.duration,
                message_rate=args.message_rate,
                failure_schedule=failure_schedule
            )
            
            if not args.quiet:
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
            
            if not args.quiet:
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
                message_rate=args.message_rate,
                failure_probability=args.failure_prob,
                recovery_probability=args.recovery_prob,
                check_interval_ms=args.check_interval
            )
            
            if not args.quiet:
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
            
            result = simulator.simulate_exhaustive(
                graph,
                component_types=args.component_types,
                enable_cascade=enable_cascade
            )
            
            if not args.quiet:
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
                'criticality': AttackStrategy.CRITICALITY,
                'betweenness': AttackStrategy.BETWEENNESS,
                'degree': AttackStrategy.DEGREE,
                'articulation': AttackStrategy.ARTICULATION
            }
            
            result = simulator.simulate_targeted_attack(
                graph,
                strategy=strategy_map[args.strategy],
                target_count=args.count,
                enable_cascade=enable_cascade
            )
            
            if not args.quiet:
                print_failure_result(result, verbose=args.verbose)
        
        elif args.random:
            # Random failures
            if not args.quiet:
                print_section("RANDOM FAILURE SIMULATION")
                print_info(f"Failure probability: {args.failure_prob}")
            
            simulator = FailureSimulator(
                cascade_threshold=args.cascade_threshold,
                cascade_probability=args.cascade_prob,
                max_cascade_depth=args.max_cascade_depth,
                seed=args.seed
            )
            
            result = simulator.simulate_random_failures(
                graph,
                failure_probability=args.failure_prob,
                component_types=args.component_types,
                enable_cascade=enable_cascade
            )
            
            if not args.quiet:
                print_failure_result(result, verbose=args.verbose)
        
        elif args.components:
            # Multiple component failure
            if not args.quiet:
                print_section("MULTIPLE FAILURE SIMULATION")
                print_info(f"Components: {', '.join(args.components)}")
            
            simulator = FailureSimulator(
                cascade_threshold=args.cascade_threshold,
                cascade_probability=args.cascade_prob,
                max_cascade_depth=args.max_cascade_depth,
                seed=args.seed
            )
            
            result = simulator.simulate_multiple_failures(
                graph,
                args.components,
                enable_cascade=enable_cascade
            )
            
            if not args.quiet:
                print_failure_result(result, verbose=args.verbose)
        
        elif args.component:
            # Single component failure
            if not args.quiet:
                print_section("SINGLE FAILURE SIMULATION")
                print_info(f"Component: {args.component}")
            
            simulator = FailureSimulator(
                cascade_threshold=args.cascade_threshold,
                cascade_probability=args.cascade_prob,
                max_cascade_depth=args.max_cascade_depth,
                seed=args.seed
            )
            
            result = simulator.simulate_single_failure(
                graph,
                args.component,
                enable_cascade=enable_cascade
            )
            
            if not args.quiet:
                print_failure_result(result, verbose=args.verbose)
        
        else:
            # Default: show help
            parser.print_help()
            return 0
        
        # Export results
        if result and args.export_json:
            output_path = Path(args.export_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            print_success(f"Exported results to {output_path}")
        
        if result and args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export JSON
            json_path = output_dir / 'simulation_results.json'
            with open(json_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            print_success(f"Exported JSON: {json_path}")
            
            # Export summary
            if hasattr(result, 'summary'):
                summary_path = output_dir / 'simulation_summary.txt'
                with open(summary_path, 'w') as f:
                    f.write(result.summary())
                print_success(f"Exported summary: {summary_path}")
        
        return 0
    
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid value: {e}")
        return 1
    except Exception as e:
        print_error(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())