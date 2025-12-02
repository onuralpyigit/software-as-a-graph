#!/usr/bin/env python3
"""
Graph Simulation CLI

Comprehensive simulation capabilities for pub-sub system graphs including:
- Component failure simulation
- Cascading failure analysis
- Event-driven message flow simulation
- Load testing
- Chaos engineering scenarios
- Impact assessment and validation

Usage Examples:
    # Simulate single component failure
    python simulate_graph.py --input system.json --component app_1
    
    # Simulate multiple failures
    python simulate_graph.py --input system.json \\
        --components app_1 app_2 app_3 --cascade
    
    # Targeted attack simulation
    python simulate_graph.py --input system.json \\
        --attack --strategy criticality --count 5
    
    # Run failure campaign (test all components)
    python simulate_graph.py --input system.json --campaign
    
    # Event-driven simulation with failures
    python simulate_graph.py --input system.json \\
        --event-sim --duration 60000 --failure-at 30000
    
    # Load test
    python simulate_graph.py --input system.json \\
        --load-test --initial-rate 10 --peak-rate 1000
    
    # Chaos simulation
    python simulate_graph.py --input system.json \\
        --chaos --failure-prob 0.05 --recovery-prob 0.3

Research Validation:
    - Compare predicted vs actual impact scores
    - Calculate Spearman correlation with criticality scores
    - Validate SPOF identification accuracy
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Error: networkx is required. Install with: pip install networkx")
    sys.exit(1)

try:
    from src.simulation.failure_simulator import (
        FailureSimulator, FailureType, FailureMode, SimulationResult
    )
    HAS_FAILURE_SIM = True
except ImportError as e:
    HAS_FAILURE_SIM = False
    FAILURE_SIM_ERROR = str(e)

try:
    from src.simulation.event_simulator import (
        EventDrivenSimulator, SimulationMetrics
    )
    HAS_EVENT_SIM = True
except ImportError as e:
    HAS_EVENT_SIM = False
    EVENT_SIM_ERROR = str(e)


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    @classmethod
    def disable(cls):
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.WARNING = cls.FAIL = cls.ENDC = cls.BOLD = cls.DIM = ''


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")


def print_section(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.DIM}{'-'*50}{Colors.ENDC}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")


def print_error(text: str):
    print(f"{Colors.FAIL}✗{Colors.ENDC} {text}")


def print_warning(text: str):
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {text}")


def print_info(text: str):
    print(f"{Colors.CYAN}ℹ{Colors.ENDC} {text}")


def print_kv(key: str, value: Any, indent: int = 2):
    spaces = ' ' * indent
    print(f"{spaces}{Colors.DIM}{key}:{Colors.ENDC} {value}")


# =============================================================================
# Graph Loading
# =============================================================================

def load_graph_from_json(filepath: str) -> Tuple[nx.DiGraph, Dict]:
    """Load graph from JSON file"""
    with open(filepath) as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes
    for node in data.get('nodes', []):
        node_attrs = {k: v for k, v in node.items() if k != 'id'}
        node_attrs['type'] = 'Node'
        if 'name' not in node_attrs:
            node_attrs['name'] = node['id']
        G.add_node(node['id'], **node_attrs)
    
    for broker in data.get('brokers', []):
        broker_attrs = {k: v for k, v in broker.items() if k != 'id'}
        broker_attrs['type'] = 'Broker'
        if 'name' not in broker_attrs:
            broker_attrs['name'] = broker['id']
        G.add_node(broker['id'], **broker_attrs)
    
    for app in data.get('applications', []):
        app_attrs = {k: v for k, v in app.items() if k != 'id'}
        app_attrs['type'] = 'Application'
        if 'name' not in app_attrs:
            app_attrs['name'] = app['id']
        G.add_node(app['id'], **app_attrs)
    
    for topic in data.get('topics', []):
        topic_attrs = {k: v for k, v in topic.items() if k not in ['id', 'qos']}
        topic_attrs['type'] = 'Topic'
        if 'name' not in topic_attrs:
            topic_attrs['name'] = topic['id']
        if 'qos' in topic:
            for qk, qv in topic['qos'].items():
                topic_attrs[f'qos_{qk}'] = qv
        G.add_node(topic['id'], **topic_attrs)
    
    # Add edges
    relationships = data.get('relationships', {})
    
    for rel in relationships.get('runs_on', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        if source and target:
            G.add_edge(source, target, type='RUNS_ON')
    
    for rel in relationships.get('publishes_to', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        if source and target:
            G.add_edge(source, target, type='PUBLISHES_TO',
                      period_ms=rel.get('period_ms'), msg_size=rel.get('msg_size_bytes'))
    
    for rel in relationships.get('subscribes_to', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        if source and target:
            G.add_edge(source, target, type='SUBSCRIBES_TO')
    
    for rel in relationships.get('routes', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        if source and target:
            G.add_edge(source, target, type='ROUTES')
    
    # Derive DEPENDS_ON
    G = derive_dependencies(G)
    
    return G, data


def derive_dependencies(G: nx.DiGraph) -> nx.DiGraph:
    """Derive DEPENDS_ON relationships from pub/sub patterns"""
    topics = [n for n, d in G.nodes(data=True) if d.get('type') == 'Topic']
    
    for topic in topics:
        publishers = [s for s, t, d in G.in_edges(topic, data=True) 
                     if d.get('type') == 'PUBLISHES_TO']
        subscribers = [s for s, t, d in G.in_edges(topic, data=True) 
                      if d.get('type') == 'SUBSCRIBES_TO']
        
        for sub in subscribers:
            for pub in publishers:
                if sub != pub and not G.has_edge(sub, pub):
                    G.add_edge(sub, pub, type='DEPENDS_ON', 
                              dependency_type='app_to_app', via_topic=topic)
    
    return G


# =============================================================================
# Result Printing
# =============================================================================

def print_simulation_result(result: SimulationResult, verbose: bool = False):
    """Print formatted simulation result"""
    
    # Severity coloring
    impact = result.impact_score
    if impact >= 0.7:
        severity_color = Colors.FAIL
        severity = "CRITICAL"
    elif impact >= 0.4:
        severity_color = Colors.WARNING
        severity = "HIGH"
    elif impact >= 0.2:
        severity_color = Colors.CYAN
        severity = "MEDIUM"
    else:
        severity_color = Colors.GREEN
        severity = "LOW"
    
    print_section(f"Simulation Result: {result.simulation_id}")
    print(f"  {Colors.BOLD}Type:{Colors.ENDC} {result.simulation_type}")
    print(f"  {Colors.BOLD}Severity:{Colors.ENDC} {severity_color}{severity}{Colors.ENDC}")
    
    print(f"\n  {Colors.BOLD}Impact Metrics:{Colors.ENDC}")
    print_kv("Impact Score", f"{result.impact_score:.4f}", indent=4)
    print_kv("Resilience Score", f"{result.resilience_score:.4f}", indent=4)
    print_kv("Service Continuity", f"{result.service_continuity:.4f}", indent=4)
    
    print(f"\n  {Colors.BOLD}Failures:{Colors.ENDC}")
    print_kv("Primary Failures", len(result.failed_components) - len(result.cascade_failures), indent=4)
    print_kv("Cascade Failures", len(result.cascade_failures), indent=4)
    print_kv("Total Failed", len(result.failed_components), indent=4)
    
    if result.failed_components and verbose:
        print(f"    {Colors.DIM}Failed:{Colors.ENDC} {', '.join(result.failed_components[:10])}")
        if len(result.failed_components) > 10:
            print(f"    {Colors.DIM}... and {len(result.failed_components) - 10} more{Colors.ENDC}")
    
    print(f"\n  {Colors.BOLD}Connectivity:{Colors.ENDC}")
    print_kv("Reachability Loss", f"{result.reachability_loss_pct:.2f}%", indent=4)
    print_kv("Fragmentation", f"+{result.fragmentation} components", indent=4)
    print_kv("Affected Components", len(result.affected_components), indent=4)
    print_kv("Isolated Components", len(result.isolated_components), indent=4)


def print_event_simulation_result(result: Dict[str, Any]):
    """Print event simulation results"""
    
    print_section("Event Simulation Results")
    
    if 'pre_failure_metrics' in result:
        print(f"\n  {Colors.BOLD}Pre-Failure Phase:{Colors.ENDC}")
        pre = result['pre_failure_metrics']
        print_kv("Messages Published", pre['messages']['published'], indent=4)
        print_kv("Delivery Rate", f"{pre['messages']['delivery_rate']:.2%}", indent=4)
        print_kv("Avg Latency", f"{pre['latency_ms']['avg']:.2f}ms", indent=4)
    
    if 'post_failure_metrics' in result:
        print(f"\n  {Colors.BOLD}Post-Failure Phase:{Colors.ENDC}")
        post = result['post_failure_metrics']
        print_kv("Messages Published", post['messages']['published'], indent=4)
        print_kv("Delivery Rate", f"{post['messages']['delivery_rate']:.2%}", indent=4)
        print_kv("Avg Latency", f"{post['latency_ms']['avg']:.2f}ms", indent=4)
        print_kv("Messages Dropped", post['messages']['dropped'], indent=4)
    
    if 'impact_analysis' in result:
        print(f"\n  {Colors.BOLD}Impact Analysis:{Colors.ENDC}")
        impact = result['impact_analysis']
        severity_color = Colors.FAIL if impact['severity'] == 'CRITICAL' else Colors.WARNING
        print(f"    {Colors.DIM}Severity:{Colors.ENDC} {severity_color}{impact['severity']}{Colors.ENDC}")
        print_kv("Delivery Rate Impact", f"{impact['delivery_rate_impact']:+.2%}", indent=4)
        print_kv("Latency Impact", f"+{impact['latency_impact_ms']:.2f}ms", indent=4)
    
    if 'failed_components' in result:
        print(f"\n  {Colors.BOLD}Failed Components:{Colors.ENDC}")
        for comp in result['failed_components'][:5]:
            print(f"    {Colors.FAIL}•{Colors.ENDC} {comp}")
        if len(result['failed_components']) > 5:
            print(f"    {Colors.DIM}... and {len(result['failed_components']) - 5} more{Colors.ENDC}")


def print_campaign_summary(results: List[SimulationResult]):
    """Print summary of failure campaign"""
    
    print_section(f"Failure Campaign Summary ({len(results)} simulations)")
    
    # Aggregate statistics
    impact_scores = [r.impact_score for r in results]
    reachability_losses = [r.reachability_loss_pct for r in results]
    cascade_counts = [len(r.cascade_failures) for r in results]
    
    print(f"\n  {Colors.BOLD}Impact Distribution:{Colors.ENDC}")
    print_kv("Avg Impact Score", f"{sum(impact_scores)/len(impact_scores):.4f}", indent=4)
    print_kv("Max Impact Score", f"{max(impact_scores):.4f}", indent=4)
    print_kv("Min Impact Score", f"{min(impact_scores):.4f}", indent=4)
    
    # Categorize by severity
    critical = sum(1 for s in impact_scores if s >= 0.7)
    high = sum(1 for s in impact_scores if 0.4 <= s < 0.7)
    medium = sum(1 for s in impact_scores if 0.2 <= s < 0.4)
    low = sum(1 for s in impact_scores if s < 0.2)
    
    print(f"\n  {Colors.BOLD}By Severity:{Colors.ENDC}")
    print(f"    {Colors.FAIL}CRITICAL:{Colors.ENDC} {critical}")
    print(f"    {Colors.WARNING}HIGH:{Colors.ENDC} {high}")
    print(f"    {Colors.CYAN}MEDIUM:{Colors.ENDC} {medium}")
    print(f"    {Colors.GREEN}LOW:{Colors.ENDC} {low}")
    
    print(f"\n  {Colors.BOLD}Reachability:{Colors.ENDC}")
    print_kv("Avg Loss", f"{sum(reachability_losses)/len(reachability_losses):.2f}%", indent=4)
    print_kv("Max Loss", f"{max(reachability_losses):.2f}%", indent=4)
    
    print(f"\n  {Colors.BOLD}Cascade Behavior:{Colors.ENDC}")
    print_kv("Simulations with Cascades", sum(1 for c in cascade_counts if c > 0), indent=4)
    print_kv("Avg Cascade Size", f"{sum(cascade_counts)/len(cascade_counts):.1f}", indent=4)
    print_kv("Max Cascade Size", max(cascade_counts), indent=4)
    
    # Top impactful components
    sorted_results = sorted(results, key=lambda r: r.impact_score, reverse=True)
    print(f"\n  {Colors.BOLD}Most Impactful Failures:{Colors.ENDC}")
    for result in sorted_results[:5]:
        comp = result.failed_components[0] if result.failed_components else "unknown"
        print(f"    {Colors.FAIL}•{Colors.ENDC} {comp}: impact={result.impact_score:.4f}, "
              f"cascades={len(result.cascade_failures)}")


# =============================================================================
# Export Functions
# =============================================================================

def export_results_json(results: Any, filepath: str):
    """Export results to JSON file"""
    if isinstance(results, SimulationResult):
        data = results.to_dict()
    elif isinstance(results, list):
        data = {
            'simulations': [r.to_dict() if isinstance(r, SimulationResult) else r for r in results],
            'count': len(results),
            'exported_at': datetime.now().isoformat()
        }
    else:
        data = results
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def export_results_csv(results: List[SimulationResult], filepath: str):
    """Export campaign results to CSV"""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'component', 'impact_score', 'resilience_score', 'service_continuity',
            'reachability_loss_pct', 'cascade_failures', 'affected_components',
            'fragmentation', 'simulation_type'
        ])
        
        for result in results:
            comp = result.failed_components[0] if result.failed_components else "multiple"
            writer.writerow([
                comp,
                round(result.impact_score, 4),
                round(result.resilience_score, 4),
                round(result.service_continuity, 4),
                round(result.reachability_loss_pct, 2),
                len(result.cascade_failures),
                len(result.affected_components),
                result.fragmentation,
                result.simulation_type
            ])


# =============================================================================
# Main CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Comprehensive pub-sub system simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input system.json --component app_1
  %(prog)s --input system.json --attack --strategy criticality --count 5
  %(prog)s --input system.json --campaign
  %(prog)s --input system.json --event-sim --duration 60000
  %(prog)s --input system.json --load-test --peak-rate 1000
  %(prog)s --input system.json --chaos --failure-prob 0.05
        """
    )
    
    # Input
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--input', '-i', required=True, help='Input JSON file')
    input_group.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    # Failure simulation options
    failure_group = parser.add_argument_group('Failure Simulation')
    failure_group.add_argument('--component', '-c', help='Single component to fail')
    failure_group.add_argument('--components', nargs='+', help='Multiple components to fail')
    failure_group.add_argument('--cascade', action='store_true', default=True,
                              help='Enable cascading failures (default: True)')
    failure_group.add_argument('--no-cascade', action='store_false', dest='cascade',
                              help='Disable cascading failures')
    
    # Attack simulation
    attack_group = parser.add_argument_group('Attack Simulation')
    attack_group.add_argument('--attack', action='store_true', help='Run targeted attack simulation')
    attack_group.add_argument('--strategy', choices=['criticality', 'degree', 'betweenness'],
                             default='criticality', help='Attack target selection strategy')
    attack_group.add_argument('--count', type=int, default=5, help='Number of targets')
    
    # Random failures
    random_group = parser.add_argument_group('Random Failures')
    random_group.add_argument('--random', action='store_true', help='Simulate random failures')
    random_group.add_argument('--failure-prob', type=float, default=0.1,
                             help='Failure probability (default: 0.1)')
    
    # Campaign
    campaign_group = parser.add_argument_group('Failure Campaign')
    campaign_group.add_argument('--campaign', action='store_true',
                               help='Run failure simulation for each component')
    campaign_group.add_argument('--component-types', nargs='+',
                               default=['Application'],
                               help='Component types to test (default: Application)')
    
    # Event-driven simulation
    event_group = parser.add_argument_group('Event-Driven Simulation')
    event_group.add_argument('--event-sim', action='store_true',
                            help='Run event-driven simulation')
    event_group.add_argument('--duration', type=float, default=60000,
                            help='Simulation duration in ms (default: 60000)')
    event_group.add_argument('--failure-at', type=float,
                            help='Inject failure at this time (ms)')
    event_group.add_argument('--message-rate', type=float, default=100,
                            help='Messages per second (default: 100)')
    
    # Load test
    load_group = parser.add_argument_group('Load Testing')
    load_group.add_argument('--load-test', action='store_true', help='Run load test')
    load_group.add_argument('--initial-rate', type=float, default=10,
                           help='Initial message rate (default: 10)')
    load_group.add_argument('--peak-rate', type=float, default=1000,
                           help='Peak message rate (default: 1000)')
    load_group.add_argument('--ramp-time', type=float, default=10000,
                           help='Ramp up time in ms (default: 10000)')
    
    # Chaos simulation
    chaos_group = parser.add_argument_group('Chaos Engineering')
    chaos_group.add_argument('--chaos', action='store_true', help='Run chaos simulation')
    chaos_group.add_argument('--recovery-prob', type=float, default=0.3,
                            help='Recovery probability (default: 0.3)')
    
    # Export
    export_group = parser.add_argument_group('Export')
    export_group.add_argument('--export-json', metavar='FILE', help='Export results to JSON')
    export_group.add_argument('--export-csv', metavar='FILE', help='Export campaign to CSV')
    
    # Verbosity
    verbosity_group = parser.add_argument_group('Verbosity')
    verbosity_group.add_argument('--verbose', '-v', action='store_true')
    verbosity_group.add_argument('--quiet', '-q', action='store_true')
    verbosity_group.add_argument('--no-color', action='store_true')
    
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
    log_level = logging.DEBUG if args.verbose else (logging.ERROR if args.quiet else logging.INFO)
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    if not args.quiet:
        print_header("GRAPH SIMULATION")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load graph
    if not args.quiet:
        print_info(f"Loading graph from {args.input}...")
    
    try:
        G, data = load_graph_from_json(args.input)
        print_success(f"Loaded graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print_error(f"Failed to load graph: {e}")
        return 1
    
    results = None
    
    # Run appropriate simulation
    try:
        # Event-driven simulations
        if args.event_sim or args.load_test or args.chaos:
            if not HAS_EVENT_SIM:
                print_error(f"Event simulator not available: {EVENT_SIM_ERROR}")
                return 1
            
            simulator = EventDrivenSimulator(G, data, seed=args.seed)
            
            if args.event_sim:
                if not args.quiet:
                    print_info("Running event-driven simulation...")
                
                if args.failure_at:
                    results = simulator.run_failure_simulation(
                        duration_ms=args.duration,
                        failure_time_ms=args.failure_at,
                        message_rate_per_sec=args.message_rate,
                        enable_cascade=args.cascade
                    )
                else:
                    metrics = simulator.run_baseline_simulation(
                        duration_ms=args.duration,
                        message_rate_per_sec=args.message_rate
                    )
                    results = {'baseline_metrics': metrics.to_dict()}
                
                if not args.quiet:
                    print_event_simulation_result(results)
            
            elif args.load_test:
                if not args.quiet:
                    print_info("Running load test...")
                
                results = simulator.run_load_test(
                    duration_ms=args.duration,
                    initial_rate=args.initial_rate,
                    peak_rate=args.peak_rate,
                    ramp_time_ms=args.ramp_time
                )
                
                if not args.quiet:
                    print_section("Load Test Results")
                    m = results['metrics']
                    print_kv("Peak Rate Achieved", f"{m['throughput']['messages_per_sec']:.1f} msg/s")
                    print_kv("Delivery Rate", f"{m['messages']['delivery_rate']:.2%}")
                    print_kv("P99 Latency", f"{m['latency_ms']['p99']:.2f}ms")
                    print_kv("Saturated", results['saturation_analysis']['saturated'])
            
            elif args.chaos:
                if not args.quiet:
                    print_info("Running chaos simulation...")
                
                results = simulator.run_chaos_simulation(
                    duration_ms=args.duration,
                    failure_probability=args.failure_prob,
                    recovery_probability=args.recovery_prob,
                    message_rate_per_sec=args.message_rate
                )
                
                if not args.quiet:
                    print_section("Chaos Simulation Results")
                    m = results['metrics']
                    print_kv("Delivery Rate", f"{m['messages']['delivery_rate']:.2%}")
                    print_kv("Component Failures", m['failures']['component_failures'])
                    print_kv("Cascade Failures", m['failures']['cascade_failures'])
                    print_kv("Resilience Score", f"{results['resilience_score']:.4f}")
        
        # Failure simulations
        else:
            if not HAS_FAILURE_SIM:
                print_error(f"Failure simulator not available: {FAILURE_SIM_ERROR}")
                return 1
            
            simulator = FailureSimulator(seed=args.seed)
            
            if args.campaign:
                if not args.quiet:
                    print_info(f"Running failure campaign for {args.component_types}...")
                
                # Get components to test
                components = [
                    n for n, d in G.nodes(data=True)
                    if d.get('type') in args.component_types
                ]
                
                if not args.quiet:
                    print_info(f"Testing {len(components)} components...")
                
                results = simulator.run_failure_campaign(G, components)
                
                if not args.quiet:
                    print_campaign_summary(results)
            
            elif args.attack:
                if not args.quiet:
                    print_info(f"Running targeted attack ({args.strategy}, n={args.count})...")
                
                result = simulator.simulate_targeted_attack(
                    G,
                    target_count=args.count,
                    strategy=args.strategy
                )
                results = result
                
                if not args.quiet:
                    print_simulation_result(result, args.verbose)
            
            elif args.random:
                if not args.quiet:
                    print_info(f"Running random failure simulation (p={args.failure_prob})...")
                
                result = simulator.simulate_random_failures(
                    G,
                    failure_probability=args.failure_prob,
                    enable_cascade=args.cascade
                )
                results = result
                
                if not args.quiet:
                    print_simulation_result(result, args.verbose)
            
            elif args.components:
                if not args.quiet:
                    print_info(f"Simulating failure of {len(args.components)} components...")
                
                result = simulator.simulate_multiple_failures(
                    G,
                    args.components,
                    enable_cascade=args.cascade
                )
                results = result
                
                if not args.quiet:
                    print_simulation_result(result, args.verbose)
            
            elif args.component:
                if not args.quiet:
                    print_info(f"Simulating failure of {args.component}...")
                
                result = simulator.simulate_single_failure(
                    G,
                    args.component,
                    enable_cascade=args.cascade
                )
                results = result
                
                if not args.quiet:
                    print_simulation_result(result, args.verbose)
            
            else:
                print_warning("No simulation type specified. Use --help for options.")
                return 0
    
    except Exception as e:
        logger.exception("Simulation failed")
        print_error(f"Simulation failed: {e}")
        return 1
    
    # Export results
    if results:
        if args.export_json:
            export_results_json(results, args.export_json)
            print_success(f"Results exported to {args.export_json}")
        
        if args.export_csv and isinstance(results, list):
            export_results_csv(results, args.export_csv)
            print_success(f"Campaign results exported to {args.export_csv}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())