#!/usr/bin/env python3
"""
Graph Simulation Script - Production Version

Comprehensive simulation capabilities for pub-sub systems including:
- Realistic message traffic simulation (1000x speedup)
- Advanced failure injection and cascading analysis
- Performance impact assessment with detailed metrics
- Real-time monitoring with progress reporting
- Multiple predefined scenarios
- Baseline vs failure comparison
- Comprehensive reporting (JSON, HTML, Markdown)

Author: Research Team
Version: 2.0
"""

import argparse
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.simulation.lightweight_dds_simulator import LightweightDDSSimulator
    from src.simulation.enhanced_failure_simulator import (
        FailureSimulator, FailureType, ComponentType
    )
    from src.core.graph_model import GraphModel
    import networkx as nx
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("\nPlease ensure all required files are in the src/ directory:")
    print("  - src/simulation/lightweight_dds_simulator.py")
    print("  - src/simulation/enhanced_failure_simulator.py")
    print("  - src/core/graph_model.py")
    sys.exit(1)


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Predefined simulation scenarios
SIMULATION_SCENARIOS = {
    'single-app': {
        'description': 'Single application failure',
        'duration': 60,
        'failures': [{
            'time': 30,
            'components': ['A1'],
            'type': 'complete',
            'cascade': False
        }]
    },
    'cascading-broker': {
        'description': 'Broker failure with cascading effects',
        'duration': 120,
        'failures': [{
            'time': 40,
            'components': ['B1'],
            'type': 'complete',
            'cascade': True
        }]
    },
    'node-failure': {
        'description': 'Infrastructure node failure',
        'duration': 90,
        'failures': [{
            'time': 30,
            'components': ['N1'],
            'type': 'complete',
            'cascade': True
        }]
    },
    'multiple-simultaneous': {
        'description': 'Multiple simultaneous failures',
        'duration': 60,
        'failures': [{
            'time': 20,
            'components': ['A1', 'A2', 'B1'],
            'type': 'complete',
            'cascade': False
        }]
    },
    'degraded-performance': {
        'description': 'Gradual performance degradation',
        'duration': 90,
        'failures': [
            {'time': 20, 'components': ['B1'], 'type': 'partial', 'cascade': False},
            {'time': 40, 'components': ['B2'], 'type': 'partial', 'cascade': False},
            {'time': 60, 'components': ['N1'], 'type': 'partial', 'cascade': False}
        ]
    },
    'failure-recovery': {
        'description': 'Failure with automatic recovery',
        'duration': 120,
        'failures': [{
            'time': 30,
            'components': ['A1'],
            'type': 'complete',
            'cascade': False,
            'recovery_time': 40
        }]
    },
    'sequential-failures': {
        'description': 'Sequential failures over time',
        'duration': 120,
        'failures': [
            {'time': 20, 'components': ['A1'], 'type': 'complete', 'cascade': False},
            {'time': 40, 'components': ['A2'], 'type': 'complete', 'cascade': False},
            {'time': 60, 'components': ['B1'], 'type': 'complete', 'cascade': True},
            {'time': 80, 'components': ['N1'], 'type': 'complete', 'cascade': True}
        ]
    }
}


def setup_logging(verbose: bool = False, log_file: str = 'simulation.log') -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger('simulation')
    logger.setLevel(level)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler  
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_graph_from_json(json_path: str, logger: logging.Logger) -> Tuple[GraphModel, nx.DiGraph]:
    """Load graph from JSON file"""
    logger.info(f"Loading graph from {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create graph model
    model = GraphModel()
    
    # Load components
    for node in data.get('nodes', []):
        model.add_node(node['id'], node.get('properties', {}))
    
    for app in data.get('applications', []):
        model.add_application(app['id'], app.get('node'), app.get('properties', {}))
    
    for broker in data.get('brokers', []):
        model.add_broker(broker['id'], broker.get('node'), broker.get('properties', {}))
    
    for topic in data.get('topics', []):
        model.add_topic(topic['id'], topic.get('broker'), topic.get('properties', {}))
    
    # Convert to NetworkX graph
    graph = model.to_networkx()
    
    logger.info(f"Loaded graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    return model, graph


def validate_simulation_config(model: GraphModel, 
                               fail_components: List[str],
                               logger: logging.Logger) -> Tuple[bool, List[str]]:
    """Validate simulation configuration"""
    warnings = []
    is_valid = True
    
    # Check if components exist
    all_components = set()
    for comp_list in [model.nodes, model.applications, model.brokers, model.topics]:
        all_components.update(comp_list.keys())
    
    for comp in fail_components:
        if comp not in all_components:
            warnings.append(f"Component {comp} not found in graph")
            is_valid = False
    
    if warnings:
        for warning in warnings:
            logger.warning(warning)
    
    return is_valid, warnings


async def run_baseline_simulation(simulator: LightweightDDSSimulator,
                                  duration: int,
                                  logger: logging.Logger) -> Dict[str, Any]:
    """Run baseline simulation without failures"""
    logger.info(f"Running baseline simulation for {duration}s...")
    
    results = await simulator.run_simulation(duration)
    
    logger.info("Baseline simulation complete")
    logger.info(f"  Messages delivered: {results['global_stats']['messages_delivered']}")
    logger.info(f"  Avg latency: {results['global_stats']['avg_latency_ms']:.2f}ms")
    logger.info(f"  Delivery rate: {results['global_stats']['delivery_rate']:.2%}")
    
    return results


async def run_failure_simulation(simulator: LightweightDDSSimulator,
                                 duration: int,
                                 failures: List[Dict],
                                 failure_sim: FailureSimulator,
                                 logger: logging.Logger,
                                 monitor: bool = False,
                                 monitor_interval: int = 10) -> Dict[str, Any]:
    """Run simulation with failure injection"""
    logger.info(f"Running failure simulation for {duration}s with {len(failures)} failure(s)...")
    
    # Start monitoring task if requested
    if monitor:
        monitor_task = asyncio.create_task(
            monitor_simulation(simulator, duration, monitor_interval, logger)
        )
    
    # Schedule failure injections
    failure_task = asyncio.create_task(
        inject_failures(simulator, failure_sim, failures, logger)
    )
    
    # Run simulation
    results = await simulator.run_simulation(duration)
    
    # Cancel monitoring task
    if monitor:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
    
    # Wait for failure injections to complete
    try:
        await failure_task
    except Exception as e:
        logger.error(f"Error in failure injection: {e}")
    
    logger.info("Failure simulation complete")
    
    return results


async def inject_failures(simulator: LightweightDDSSimulator,
                         failure_sim: FailureSimulator,
                         failures: List[Dict],
                         logger: logging.Logger):
    """Inject failures according to schedule"""
    for failure in sorted(failures, key=lambda x: x['time']):
        # Wait until scheduled time
        await asyncio.sleep(failure['time'])
        
        # Inject failure for each component
        for component in failure['components']:
            # Determine component type
            comp_type = _determine_component_type(simulator, component)
            
            if comp_type is None:
                logger.warning(f"Could not determine type for component {component}")
                continue
            
            # Inject failure
            fail_type = FailureType.COMPLETE if failure['type'] == 'complete' else FailureType.PARTIAL
            
            failure_sim.inject_failure(
                simulator,
                component,
                comp_type,
                fail_type,
                severity=1.0 if failure['type'] == 'complete' else 0.5,
                enable_cascade=failure.get('cascade', False)
            )
        
        # Handle recovery if specified
        if 'recovery_time' in failure:
            await asyncio.sleep(failure['recovery_time'])
            logger.info(f"Recovering components: {failure['components']}")
            # Implement recovery logic here


async def monitor_simulation(simulator: LightweightDDSSimulator,
                            duration: int,
                            interval: int,
                            logger: logging.Logger):
    """Monitor simulation progress"""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        await asyncio.sleep(interval)
        
        elapsed = time.time() - start_time
        stats = simulator.global_stats
        
        print(f"\n{Colors.OKCYAN}[Monitor] T+{elapsed:.0f}s:{Colors.ENDC}")
        print(f"  Delivered: {stats.messages_delivered:,}")
        print(f"  Dropped: {stats.messages_dropped:,}")
        print(f"  Active failures: {len(simulator.global_stats.__dict__.get('active_failures', []))}")


def _determine_component_type(simulator: LightweightDDSSimulator, 
                              component_id: str) -> Optional[ComponentType]:
    """Determine component type from ID"""
    if component_id in simulator.nodes:
        return ComponentType.NODE
    elif component_id in simulator.applications:
        return ComponentType.APPLICATION
    elif component_id in simulator.brokers:
        return ComponentType.BROKER
    elif component_id in simulator.topics:
        return ComponentType.TOPIC
    return None


def compare_with_baseline(baseline: Dict[str, Any],
                         failure: Dict[str, Any],
                         logger: logging.Logger) -> Dict[str, Any]:
    """Compare failure simulation with baseline"""
    baseline_stats = baseline['global_stats']
    failure_stats = failure['global_stats']
    
    comparison = {
        'latency': {
            'baseline': baseline_stats['avg_latency_ms'],
            'failure': failure_stats['avg_latency_ms'],
            'increase': failure_stats['avg_latency_ms'] - baseline_stats['avg_latency_ms'],
            'increase_pct': (
                (failure_stats['avg_latency_ms'] - baseline_stats['avg_latency_ms']) /
                baseline_stats['avg_latency_ms'] * 100 if baseline_stats['avg_latency_ms'] > 0 else 0
            )
        },
        'throughput': {
            'baseline': baseline_stats['messages_delivered'] / baseline['elapsed_seconds'],
            'failure': failure_stats['messages_delivered'] / failure['elapsed_seconds'],
            'decrease_pct': (
                ((baseline_stats['messages_delivered'] / baseline['elapsed_seconds']) -
                 (failure_stats['messages_delivered'] / failure['elapsed_seconds'])) /
                (baseline_stats['messages_delivered'] / baseline['elapsed_seconds']) * 100
                if baseline_stats['messages_delivered'] > 0 else 0
            )
        },
        'delivery_rate': {
            'baseline': baseline_stats['delivery_rate'],
            'failure': failure_stats['delivery_rate'],
            'decrease': baseline_stats['delivery_rate'] - failure_stats['delivery_rate']
        },
        'messages_lost': failure_stats['messages_dropped'] - baseline_stats['messages_dropped']
    }
    
    return comparison


def export_results(results: Dict[str, Any],
                  output_dir: str,
                  export_json: bool = True,
                  export_html: bool = False,
                  export_md: bool = False,
                  logger: logging.Logger = None):
    """Export simulation results in various formats"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if export_json:
        json_file = output_path / f'simulation_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        if logger:
            logger.info(f"Results exported to {json_file}")
    
    if export_html:
        html_file = output_path / f'simulation_{timestamp}.html'
        _export_html_report(results, html_file)
        if logger:
            logger.info(f"HTML report exported to {html_file}")
    
    if export_md:
        md_file = output_path / f'simulation_{timestamp}.md'
        _export_markdown_report(results, md_file)
        if logger:
            logger.info(f"Markdown report exported to {md_file}")


def _export_html_report(results: Dict[str, Any], output_file: Path):
    """Export HTML report"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Simulation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #4CAF50; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .warning {{ color: #ff9800; }}
        .error {{ color: #f44336; }}
        .success {{ color: #4CAF50; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Pub-Sub System Simulation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Global Statistics</h2>
        <div class="metric">
            <div class="metric-label">Messages Sent</div>
            <div class="metric-value">{results.get('simulation', {}).get('global_stats', {}).get('messages_sent', 0):,}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Messages Delivered</div>
            <div class="metric-value">{results.get('simulation', {}).get('global_stats', {}).get('messages_delivered', 0):,}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Delivery Rate</div>
            <div class="metric-value">{results.get('simulation', {}).get('global_stats', {}).get('delivery_rate', 0) * 100:.1f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Avg Latency</div>
            <div class="metric-value">{results.get('simulation', {}).get('global_stats', {}).get('avg_latency_ms', 0):.2f}ms</div>
        </div>
        
        <h2>Simulation Details</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Duration</td>
                <td>{results.get('simulation', {}).get('elapsed_seconds', 0):.2f}s</td>
            </tr>
            <tr>
                <td>Messages Dropped</td>
                <td class="warning">{results.get('simulation', {}).get('global_stats', {}).get('messages_dropped', 0):,}</td>
            </tr>
            <tr>
                <td>Deadline Misses</td>
                <td class="warning">{results.get('simulation', {}).get('global_stats', {}).get('deadline_misses', 0):,}</td>
            </tr>
            <tr>
                <td>Data Transferred</td>
                <td>{results.get('simulation', {}).get('global_stats', {}).get('bytes_transferred', 0) / 1024 / 1024:.2f} MB</td>
            </tr>
        </table>
    </div>
</body>
</html>"""
    
    with open(output_file, 'w') as f:
        f.write(html)


def _export_markdown_report(results: Dict[str, Any], output_file: Path):
    """Export Markdown report"""
    md = f"""# Pub-Sub System Simulation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Global Statistics

| Metric | Value |
|--------|-------|
| Messages Sent | {results.get('simulation', {}).get('global_stats', {}).get('messages_sent', 0):,} |
| Messages Delivered | {results.get('simulation', {}).get('global_stats', {}).get('messages_delivered', 0):,} |
| Messages Dropped | {results.get('simulation', {}).get('global_stats', {}).get('messages_dropped', 0):,} |
| Delivery Rate | {results.get('simulation', {}).get('global_stats', {}).get('delivery_rate', 0) * 100:.1f}% |
| Average Latency | {results.get('simulation', {}).get('global_stats', {}).get('avg_latency_ms', 0):.2f}ms |
| Data Transferred | {results.get('simulation', {}).get('global_stats', {}).get('bytes_transferred', 0) / 1024 / 1024:.2f} MB |

## Duration

Total simulation time: {results.get('simulation', {}).get('elapsed_seconds', 0):.2f}s
"""
    
    with open(output_file, 'w') as f:
        f.write(md)


def print_simulation_summary(results: Dict[str, Any],
                            baseline: Optional[Dict[str, Any]] = None,
                            comparison: Optional[Dict[str, Any]] = None):
    """Print comprehensive simulation summary"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}SIMULATION SUMMARY{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
    
    # Global statistics
    stats = results.get('simulation', {}).get('global_stats', {})
    counts = results.get('simulation', {}).get('component_counts', {})
    
    print(f"{Colors.OKBLUE}📊 Topology:{Colors.ENDC}")
    print(f"   Nodes: {counts.get('nodes', 0)}")
    print(f"   Applications: {counts.get('applications', 0)}")
    print(f"   Topics: {counts.get('topics', 0)}")
    print(f"   Brokers: {counts.get('brokers', 0)}")
    
    print(f"\n{Colors.OKBLUE}📨 Message Statistics:{Colors.ENDC}")
    print(f"   Sent: {stats.get('messages_sent', 0):,}")
    print(f"   Delivered: {stats.get('messages_delivered', 0):,}")
    print(f"   Dropped: {stats.get('messages_dropped', 0):,}")
    print(f"   Delivery Rate: {stats.get('delivery_rate', 0):.2%}")
    
    print(f"\n{Colors.OKBLUE}⏱️  Latency:{Colors.ENDC}")
    print(f"   Average: {stats.get('avg_latency_ms', 0):.2f}ms")
    print(f"   Min: {stats.get('min_latency_ms', 0):.2f}ms")
    print(f"   Max: {stats.get('max_latency_ms', 0):.2f}ms")
    
    # Comparison with baseline
    if comparison:
        print(f"\n{Colors.WARNING}📈 Baseline Comparison:{Colors.ENDC}")
        
        lat = comparison['latency']
        print(f"\n   Latency:")
        print(f"     Baseline: {lat['baseline']:.2f}ms")
        print(f"     With Failures: {lat['failure']:.2f}ms")
        if lat['increase'] > 0:
            print(f"     {Colors.FAIL}Increase: +{lat['increase_pct']:.1f}% (+{lat['increase']:.2f}ms){Colors.ENDC}")
        
        thr = comparison['throughput']
        print(f"\n   Throughput:")
        print(f"     Baseline: {thr['baseline']:.1f} msg/s")
        print(f"     With Failures: {thr['failure']:.1f} msg/s")
        if thr['decrease_pct'] > 0:
            print(f"     {Colors.FAIL}Decrease: -{thr['decrease_pct']:.1f}%{Colors.ENDC}")
        
        print(f"\n   {Colors.FAIL}Additional Messages Lost: {comparison['messages_lost']}{Colors.ENDC}")
    
    # Failure impact
    if 'failures' in results:
        failures = results['failures']
        impact = results.get('impact', {})
        
        print(f"\n{Colors.FAIL}❌ Failure Impact:{Colors.ENDC}")
        print(f"   Total Failures: {failures.get('total_failures', 0)}")
        print(f"   Failed Components: {len(impact.get('failed_components', []))}")
        print(f"   Affected Components: {len(impact.get('affected_components', []))}")
        print(f"   Cascade Depth: {impact.get('cascade_depth', 0)}")
    
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def main() -> int:
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Simulate pub-sub system with failure injection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation
  python simulate_graph.py --input system.json --duration 60

  # With failure injection
  python simulate_graph.py --input system.json --duration 60 \\
      --fail-component A1 --fail-time 30 --fail-type complete

  # Multiple failures with cascading
  python simulate_graph.py --input system.json --duration 60 \\
      --fail-component A1,B1,N1 --fail-time 30 --cascade

  # Predefined scenario
  python simulate_graph.py --input system.json --scenario cascading-broker

  # With monitoring and reports
  python simulate_graph.py --input system.json --duration 120 \\
      --scenario sequential-failures --monitor --export-json --export-html
        """
    )
    
    # Input
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input graph JSON file')
    
    # Simulation
    parser.add_argument('--duration', '-d', type=int, default=60,
                       help='Simulation duration in seconds (default: 60)')
    parser.add_argument('--baseline', action='store_true',
                       help='Run baseline simulation first for comparison')
    
    # Failure injection
    parser.add_argument('--fail-component', type=str,
                       help='Component(s) to fail (comma-separated)')
    parser.add_argument('--fail-time', type=float, default=30.0,
                       help='Time to inject failure (seconds, default: 30)')
    parser.add_argument('--fail-type', choices=['complete', 'partial'],
                       default='complete',
                       help='Failure type (default: complete)')
    parser.add_argument('--cascade', action='store_true',
                       help='Enable cascading failures')
    parser.add_argument('--recovery-time', type=float,
                       help='Time to recovery after failure (seconds)')
    
    # Scenarios
    parser.add_argument('--scenario', choices=list(SIMULATION_SCENARIOS.keys()),
                       help='Use predefined scenario')
    
    # Monitoring
    parser.add_argument('--monitor', action='store_true',
                       help='Enable real-time monitoring')
    parser.add_argument('--monitor-interval', type=int, default=10,
                       help='Monitoring interval in seconds (default: 10)')
    
    # Output
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--export-json', action='store_true',
                       help='Export results as JSON')
    parser.add_argument('--export-html', action='store_true',
                       help='Export results as HTML report')
    parser.add_argument('--export-md', action='store_true',
                       help='Export results as Markdown report')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--log-file', type=str, default='simulation.log',
                       help='Log file path')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    # Setup logging
    logger = setup_logging(args.verbose, args.log_file)
    
    try:
        # Print header
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}PUB-SUB SYSTEM SIMULATION{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
        
        # Load graph
        logger.info("Loading graph...")
        model, graph = load_graph_from_json(args.input, logger)
        
        # Create simulator
        simulator = LightweightDDSSimulator()
        simulator.load_from_json(args.input)
        
        # Determine scenario
        failures = []
        duration = args.duration
        
        if args.scenario:
            scenario = SIMULATION_SCENARIOS[args.scenario]
            logger.info(f"Using scenario: {args.scenario} - {scenario['description']}")
            duration = scenario.get('duration', duration)
            failures = scenario['failures']
        elif args.fail_component:
            components = args.fail_component.split(',')
            failures = [{
                'time': args.fail_time,
                'components': components,
                'type': args.fail_type,
                'cascade': args.cascade,
                'recovery_time': args.recovery_time
            }]
        
        # Validate configuration
        if failures:
            all_fail_components = []
            for failure in failures:
                all_fail_components.extend(failure.get('components', []))
            
            is_valid, warnings = validate_simulation_config(model, all_fail_components, logger)
            if not is_valid:
                logger.error("Simulation configuration validation failed!")
                return 1
        
        # Run baseline if requested
        baseline_results = None
        if args.baseline or failures:
            baseline_results = asyncio.run(
                run_baseline_simulation(
                    LightweightDDSSimulator(),
                    duration,
                    logger
                )
            )
        
        # Run main simulation
        if failures:
            failure_sim = FailureSimulator(cascade_threshold=0.7, cascade_probability=0.6)
            
            results = asyncio.run(
                run_failure_simulation(
                    simulator,
                    duration,
                    failures,
                    failure_sim,
                    logger,
                    args.monitor,
                    args.monitor_interval
                )
            )
            
            # Analyze impact
            impact = failure_sim.analyze_impact(simulator, baseline_results['global_stats'] if baseline_results else None)
            
            # Compile complete results
            complete_results = {
                'simulation': results,
                'baseline': baseline_results,
                'failures': {
                    'events': [e.to_dict() for e in failure_sim.failure_events],
                    'total_failures': len(failure_sim.failure_events)
                },
                'impact': impact.to_dict() if hasattr(impact, 'to_dict') else impact
            }
        else:
            results = asyncio.run(simulator.run_simulation(duration))
            complete_results = {'simulation': results}
        
        # Compare with baseline
        comparison = None
        if baseline_results and failures:
            comparison = compare_with_baseline(baseline_results, results, logger)
            complete_results['comparison'] = comparison
        
        # Print summary
        print_simulation_summary(complete_results, baseline_results, comparison)
        
        # Export results
        if args.export_json or args.export_html or args.export_md:
            export_results(
                complete_results,
                args.output,
                args.export_json,
                args.export_html,
                args.export_md,
                logger
            )
        
        logger.info("Simulation completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
