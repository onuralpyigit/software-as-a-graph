#!/usr/bin/env python3
"""
Graph Simulation Script - Refactored Architecture

Command-line interface for comprehensive pub-sub system simulation including:
- Message traffic simulation
- Failure injection and cascading analysis
- Performance impact assessment
- Real-time monitoring and reporting
- Multiple simulation scenarios

Architecture Integration:
  GraphBuilder → GraphModel → LightweightDDSSimulator
                                        ↓
                               FailureSimulator
                                        ↓
                                  Impact Analysis
                                        ↓
                                Results & Reports

Features:
- Load graphs from JSON files or Neo4j database
- Realistic message traffic simulation (1000x speedup)
- Single and multiple component failures
- Cascading failure propagation
- Performance degradation analysis
- Baseline vs. failure comparison
- Scheduled failure scenarios
- Real-time monitoring
- Comprehensive reports (JSON, CSV, HTML)

Usage:
    # Basic traffic simulation
    python simulate_graph.py --input system.json --duration 60
    
    # Simulate with single failure
    python simulate_graph.py --input system.json --duration 60 \\
        --fail-component A1 --fail-type complete
    
    # Multiple simultaneous failures
    python simulate_graph.py --input system.json --duration 60 \\
        --fail-component A1,B1,N1 --fail-time 30
    
    # Scheduled failure scenario
    python simulate_graph.py --input system.json --duration 120 \\
        --scenario cascading-broker
    
    # With real-time monitoring
    python simulate_graph.py --input system.json --duration 60 \\
        --monitor --monitor-interval 5
    
    # Export comprehensive report
    python simulate_graph.py --input system.json --duration 60 \\
        --export-json --export-html --output results/
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

from src.simulation.lightweight_dds_simulator import LightweightDDSSimulator
from src.simulation.enhanced_failure_simulator import (
    FailureSimulator,
    FailureType,
    ComponentType,
    FailureImpact
)
from src.core.graph_builder import GraphBuilder
from src.core.graph_model import GraphModel


# Predefined simulation scenarios
SIMULATION_SCENARIOS = {
    'single-app': {
        'description': 'Single application failure',
        'failures': [
            {'time': 30, 'component': 'A1', 'type': 'application', 
             'failure_type': 'complete', 'severity': 1.0, 'cascade': True}
        ]
    },
    'cascading-broker': {
        'description': 'Broker failure with cascading effects',
        'failures': [
            {'time': 30, 'component': 'B1', 'type': 'broker',
             'failure_type': 'complete', 'severity': 1.0, 'cascade': True}
        ]
    },
    'node-failure': {
        'description': 'Infrastructure node failure affecting multiple apps',
        'failures': [
            {'time': 25, 'component': 'N1', 'type': 'node',
             'failure_type': 'complete', 'severity': 1.0, 'cascade': True}
        ]
    },
    'multiple-simultaneous': {
        'description': 'Multiple simultaneous component failures',
        'failures': [
            {'time': 30, 'component': 'A1', 'type': 'application',
             'failure_type': 'complete', 'severity': 1.0, 'cascade': False},
            {'time': 30, 'component': 'A5', 'type': 'application',
             'failure_type': 'complete', 'severity': 1.0, 'cascade': False},
            {'time': 30, 'component': 'B2', 'type': 'broker',
             'failure_type': 'partial', 'severity': 0.5, 'cascade': False}
        ]
    },
    'gradual-degradation': {
        'description': 'Gradual system degradation over time',
        'failures': [
            {'time': 20, 'component': 'A1', 'type': 'application',
             'failure_type': 'partial', 'severity': 0.3, 'cascade': False},
            {'time': 40, 'component': 'A2', 'type': 'application',
             'failure_type': 'partial', 'severity': 0.5, 'cascade': False},
            {'time': 60, 'component': 'B1', 'type': 'broker',
             'failure_type': 'partial', 'severity': 0.4, 'cascade': False}
        ]
    },
    'recovery-test': {
        'description': 'Failure followed by recovery',
        'failures': [
            {'time': 30, 'component': 'A1', 'type': 'application',
             'failure_type': 'complete', 'severity': 1.0, 'cascade': False},
            {'time': 60, 'component': 'A1', 'type': 'recovery',
             'failure_type': 'complete', 'severity': 0.0, 'cascade': False}
        ]
    },
    'stress-test': {
        'description': 'Multiple cascading failures to test resilience',
        'failures': [
            {'time': 30, 'component': 'B1', 'type': 'broker',
             'failure_type': 'complete', 'severity': 1.0, 'cascade': True},
            {'time': 45, 'component': 'B2', 'type': 'broker',
             'failure_type': 'complete', 'severity': 1.0, 'cascade': True}
        ]
    }
}


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging
    
    Args:
        verbose: Enable DEBUG level logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('simulation.log', mode='w')
        ]
    )


def load_graph_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load graph configuration from JSON file
    
    Args:
        filepath: Path to JSON configuration file
    
    Returns:
        Graph configuration dictionary
    """
    logger = logging.getLogger(__name__)
    
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        logger.info(f"✓ Loaded graph from: {filepath}")
        
        # Basic validation
        if 'nodes' not in config and 'applications' not in config:
            logger.warning("Graph appears to be empty or invalid")
        
        return config
        
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        raise


async def run_baseline_simulation(simulator: LightweightDDSSimulator,
                                  duration: int) -> Dict[str, Any]:
    """
    Run baseline simulation without failures
    
    Args:
        simulator: Configured simulator instance
        duration: Simulation duration in seconds
    
    Returns:
        Baseline performance metrics
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running baseline simulation ({duration}s)...")
    
    start_time = time.time()
    results = await simulator.run_simulation(duration)
    elapsed = time.time() - start_time
    
    # Calculate key metrics
    global_stats = results['global_stats']
    
    baseline = {
        'duration_seconds': duration,
        'execution_time': elapsed,
        'messages_sent': global_stats['messages_sent'],
        'messages_delivered': global_stats['messages_delivered'],
        'messages_dropped': global_stats['messages_dropped'],
        'delivery_rate': global_stats['delivery_rate'],
        'avg_latency_ms': global_stats['avg_latency_ms'],
        'throughput_msg_per_sec': global_stats['messages_delivered'] / duration if duration > 0 else 0,
        'deadline_misses': global_stats['deadline_misses']
    }
    
    logger.info(f"✓ Baseline complete:")
    logger.info(f"  Throughput: {baseline['throughput_msg_per_sec']:.1f} msg/s")
    logger.info(f"  Avg Latency: {baseline['avg_latency_ms']:.2f} ms")
    logger.info(f"  Delivery Rate: {baseline['delivery_rate']:.2%}")
    
    return baseline


async def run_failure_simulation(simulator: LightweightDDSSimulator,
                                failure_sim: FailureSimulator,
                                duration: int,
                                failures: List[Dict],
                                baseline: Optional[Dict] = None,
                                monitor: bool = False,
                                monitor_interval: int = 5) -> Dict[str, Any]:
    """
    Run simulation with scheduled failures
    
    Args:
        simulator: Configured simulator instance
        failure_sim: Failure simulator instance
        duration: Simulation duration in seconds
        failures: List of failure specifications
        baseline: Baseline metrics for comparison
        monitor: Enable real-time monitoring
        monitor_interval: Monitoring interval in seconds
    
    Returns:
        Complete simulation results with impact analysis
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running failure simulation ({duration}s)...")
    logger.info(f"Scheduled failures: {len(failures)}")
    
    for i, failure in enumerate(failures, 1):
        logger.info(f"  {i}. T+{failure['time']}s: {failure['type'].upper()} "
                   f"{failure['component']} - {failure['failure_type']}")
    
    # Create monitoring task if enabled
    monitor_task = None
    monitor_data = []
    
    async def monitor_loop():
        """Monitor simulation progress"""
        while True:
            await asyncio.sleep(monitor_interval)
            
            snapshot = {
                'timestamp': time.time(),
                'messages_delivered': simulator.global_stats.messages_delivered,
                'messages_dropped': simulator.global_stats.messages_dropped,
                'active_failures': len(failure_sim.active_failures)
            }
            monitor_data.append(snapshot)
            
            logger.info(f"[Monitor] Delivered: {snapshot['messages_delivered']}, "
                       f"Dropped: {snapshot['messages_dropped']}, "
                       f"Failures: {snapshot['active_failures']}")
    
    if monitor:
        monitor_task = asyncio.create_task(monitor_loop())
    
    # Create failure injection task
    failure_task = asyncio.create_task(
        failure_sim.inject_scheduled_failures(simulator, failures)
    )
    
    # Run simulation
    start_time = time.time()
    results = await simulator.run_simulation(duration)
    elapsed = time.time() - start_time
    
    # Cancel monitoring and failure tasks
    if monitor_task:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
    
    failure_task.cancel()
    try:
        await failure_task
    except asyncio.CancelledError:
        pass
    
    # Analyze impact
    impact = failure_sim.analyze_impact(simulator, baseline)
    
    # Generate recommendations
    recommendations = failure_sim.generate_recommendations(simulator, impact)
    
    logger.info(f"✓ Simulation complete:")
    logger.info(f"  Failed Components: {len(impact.failed_components)}")
    logger.info(f"  Affected Components: {len(impact.affected_components)}")
    logger.info(f"  Messages Lost: {impact.messages_lost}")
    
    return {
        'duration_seconds': duration,
        'execution_time': elapsed,
        'results': results,
        'failures': {
            'scheduled': failures,
            'events': [e.to_dict() for e in failure_sim.failure_events],
            'active_count': len(failure_sim.active_failures)
        },
        'impact': impact.to_dict(),
        'recommendations': recommendations,
        'monitoring': monitor_data if monitor else None,
        'baseline_comparison': compare_to_baseline(results, baseline) if baseline else None
    }


def compare_to_baseline(failure_results: Dict, baseline: Dict) -> Dict[str, Any]:
    """
    Compare failure simulation results to baseline
    
    Args:
        failure_results: Results from failure simulation
        baseline: Baseline metrics
    
    Returns:
        Comparison metrics
    """
    global_stats = failure_results['global_stats']
    
    comparison = {
        'latency_increase_ms': global_stats['avg_latency_ms'] - baseline['avg_latency_ms'],
        'latency_increase_pct': (
            ((global_stats['avg_latency_ms'] - baseline['avg_latency_ms']) / 
             baseline['avg_latency_ms'] * 100)
            if baseline['avg_latency_ms'] > 0 else 0
        ),
        'throughput_decrease_pct': (
            ((baseline['throughput_msg_per_sec'] - 
              global_stats['messages_delivered'] / baseline['duration_seconds']) /
             baseline['throughput_msg_per_sec'] * 100)
            if baseline['throughput_msg_per_sec'] > 0 else 0
        ),
        'delivery_rate_decrease': baseline['delivery_rate'] - global_stats['delivery_rate'],
        'additional_messages_dropped': (
            global_stats['messages_dropped'] - baseline['messages_dropped']
        ),
        'additional_deadline_misses': (
            global_stats['deadline_misses'] - baseline['deadline_misses']
        )
    }
    
    return comparison


def print_simulation_summary(results: Dict[str, Any],
                            baseline: Optional[Dict] = None) -> None:
    """
    Print formatted simulation summary
    
    Args:
        results: Simulation results
        baseline: Optional baseline for comparison
    """
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    
    # Timing
    print(f"\n⏱️  Execution:")
    print(f"   Simulation Duration: {results['duration_seconds']}s")
    print(f"   Real Time: {results['execution_time']:.2f}s")
    print(f"   Speedup: {results['duration_seconds'] / results['execution_time']:.1f}x")
    
    # Traffic stats
    global_stats = results['results']['global_stats']
    print(f"\n📊 Message Traffic:")
    print(f"   Messages Sent: {global_stats['messages_sent']:,}")
    print(f"   Messages Delivered: {global_stats['messages_delivered']:,}")
    print(f"   Messages Dropped: {global_stats['messages_dropped']:,}")
    print(f"   Delivery Rate: {global_stats['delivery_rate']:.2%}")
    print(f"   Throughput: {global_stats['messages_delivered'] / results['duration_seconds']:.1f} msg/s")
    print(f"   Avg Latency: {global_stats['avg_latency_ms']:.2f} ms")
    print(f"   Deadline Misses: {global_stats['deadline_misses']:,}")
    
    # Failure impact
    if 'impact' in results:
        impact = results['impact']
        print(f"\n💥 Failure Impact:")
        print(f"   Failed Components: {len(impact['failed_components'])}")
        if impact['failed_components']:
            print(f"      {', '.join(list(impact['failed_components'])[:5])}")
        
        print(f"   Affected Components: {len(impact['affected_components'])}")
        print(f"   Isolated Applications: {len(impact['isolated_applications'])}")
        print(f"   Unavailable Topics: {len(impact['unavailable_topics'])}")
        
        if impact['cascade_depth'] > 0:
            print(f"   Cascade Depth: {impact['cascade_depth']}")
            print(f"   Cascade Width: {impact['cascade_width']}")
    
    # Baseline comparison
    if 'baseline_comparison' in results and results['baseline_comparison']:
        comp = results['baseline_comparison']
        print(f"\n📈 Baseline Comparison:")
        print(f"   Latency Increase: {comp['latency_increase_ms']:.2f} ms "
              f"({comp['latency_increase_pct']:+.1f}%)")
        print(f"   Throughput Decrease: {comp['throughput_decrease_pct']:+.1f}%")
        print(f"   Delivery Rate Decrease: {comp['delivery_rate_decrease']:+.2%}")
        print(f"   Additional Dropped: {comp['additional_messages_dropped']:,}")
    
    # Recommendations
    if 'recommendations' in results and any(results['recommendations'].values()):
        print(f"\n💡 Recommendations:")
        for category, recs in results['recommendations'].items():
            if recs:
                print(f"\n   {category.upper()}:")
                for rec in recs[:3]:  # Top 3 per category
                    print(f"   • {rec}")
    
    print("\n" + "=" * 70)


def export_results_json(results: Dict[str, Any],
                       output_dir: Path,
                       prefix: str = "simulation") -> None:
    """
    Export results to JSON file
    
    Args:
        results: Simulation results
        output_dir: Output directory
        prefix: Filename prefix
    """
    logger = logging.getLogger(__name__)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{prefix}_{timestamp}.json"
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✓ Results exported to: {filepath}")


def export_results_html(results: Dict[str, Any],
                       output_dir: Path,
                       prefix: str = "simulation") -> None:
    """
    Export results as HTML report
    
    Args:
        results: Simulation results
        output_dir: Output directory
        prefix: Filename prefix
    """
    logger = logging.getLogger(__name__)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{prefix}_{timestamp}.html"
    
    global_stats = results['results']['global_stats']
    impact = results.get('impact', {})
    baseline_comp = results.get('baseline_comparison', {})
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Simulation Report - {timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .metric-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-card .unit {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        .impact-critical {{ border-left-color: #e74c3c; }}
        .impact-warning {{ border-left-color: #f39c12; }}
        .impact-info {{ border-left-color: #3498db; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .recommendation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Simulation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Duration:</strong> {results['duration_seconds']}s 
           (Real time: {results['execution_time']:.2f}s, 
           Speedup: {results['duration_seconds'] / results['execution_time']:.1f}x)</p>
        
        <h2>Message Traffic Statistics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Messages Sent</h3>
                <div class="value">{global_stats['messages_sent']:,}</div>
            </div>
            <div class="metric-card">
                <h3>Messages Delivered</h3>
                <div class="value">{global_stats['messages_delivered']:,}</div>
            </div>
            <div class="metric-card">
                <h3>Delivery Rate</h3>
                <div class="value">{global_stats['delivery_rate']:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Throughput</h3>
                <div class="value">{global_stats['messages_delivered'] / results['duration_seconds']:.1f}</div>
                <div class="unit">msg/s</div>
            </div>
            <div class="metric-card">
                <h3>Avg Latency</h3>
                <div class="value">{global_stats['avg_latency_ms']:.2f}</div>
                <div class="unit">ms</div>
            </div>
            <div class="metric-card">
                <h3>Messages Dropped</h3>
                <div class="value">{global_stats['messages_dropped']:,}</div>
            </div>
        </div>
"""
    
    # Add failure impact if present
    if impact:
        html_content += f"""
        <h2>Failure Impact Analysis</h2>
        <div class="metric-grid">
            <div class="metric-card impact-critical">
                <h3>Failed Components</h3>
                <div class="value">{len(impact.get('failed_components', []))}</div>
            </div>
            <div class="metric-card impact-warning">
                <h3>Affected Components</h3>
                <div class="value">{len(impact.get('affected_components', []))}</div>
            </div>
            <div class="metric-card impact-warning">
                <h3>Isolated Applications</h3>
                <div class="value">{len(impact.get('isolated_applications', []))}</div>
            </div>
            <div class="metric-card impact-info">
                <h3>Cascade Depth</h3>
                <div class="value">{impact.get('cascade_depth', 0)}</div>
            </div>
        </div>
"""
    
    # Add baseline comparison if present
    if baseline_comp:
        html_content += f"""
        <h2>Performance Degradation</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Change</th>
            </tr>
            <tr>
                <td>Latency Increase</td>
                <td>{baseline_comp['latency_increase_pct']:+.1f}% 
                    ({baseline_comp['latency_increase_ms']:+.2f} ms)</td>
            </tr>
            <tr>
                <td>Throughput Decrease</td>
                <td>{baseline_comp['throughput_decrease_pct']:+.1f}%</td>
            </tr>
            <tr>
                <td>Delivery Rate Decrease</td>
                <td>{baseline_comp['delivery_rate_decrease']:+.2%}</td>
            </tr>
            <tr>
                <td>Additional Messages Dropped</td>
                <td>{baseline_comp['additional_messages_dropped']:,}</td>
            </tr>
        </table>
"""
    
    # Add recommendations
    if 'recommendations' in results and any(results['recommendations'].values()):
        html_content += "<h2>Recommendations</h2>"
        for category, recs in results['recommendations'].items():
            if recs:
                html_content += f"<h3>{category.upper()}</h3>"
                for rec in recs[:5]:
                    html_content += f'<div class="recommendation">• {rec}</div>'
    
    html_content += """
        <div class="footer">
            <p>Generated by Software-as-a-Graph Simulation Framework</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(filepath, 'w') as f:
        f.write(html_content)
    
    logger.info(f"✓ HTML report exported to: {filepath}")


async def main_async(args: argparse.Namespace) -> int:
    """
    Async main execution
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load graph
        logger.info("=" * 70)
        logger.info("PUB-SUB SYSTEM SIMULATION")
        logger.info("Refactored Architecture")
        logger.info("=" * 70)
        
        if args.input:
            logger.info(f"\nLoading graph from JSON: {args.input}")
            graph_config = load_graph_from_json(args.input)
        else:
            logger.error("Input source required (--input)")
            return 1
        
        # Initialize simulator
        simulator = LightweightDDSSimulator()
        simulator.load_from_dict(graph_config)
        
        logger.info(f"✓ Simulator initialized:")
        logger.info(f"  Applications: {len(simulator.applications)}")
        logger.info(f"  Topics: {len(simulator.topics)}")
        logger.info(f"  Brokers: {len(simulator.brokers)}")
        
        # Run baseline if requested
        baseline = None
        if args.baseline or args.fail_component or args.scenario:
            # Create fresh simulator for baseline
            baseline_sim = LightweightDDSSimulator()
            baseline_sim.load_from_dict(graph_config)
            
            baseline_duration = args.baseline_duration if args.baseline else min(args.duration, 30)
            baseline = await run_baseline_simulation(baseline_sim, baseline_duration)
        
        # Determine failure scenario
        failures = []
        
        if args.scenario:
            # Use predefined scenario
            if args.scenario not in SIMULATION_SCENARIOS:
                logger.error(f"Unknown scenario: {args.scenario}")
                logger.info(f"Available scenarios: {', '.join(SIMULATION_SCENARIOS.keys())}")
                return 1
            
            scenario = SIMULATION_SCENARIOS[args.scenario]
            logger.info(f"\nUsing scenario: {args.scenario}")
            logger.info(f"Description: {scenario['description']}")
            
            failures = [
                {
                    'time': f['time'],
                    'component': f['component'],
                    'component_type': f['type'],
                    'failure_type': f['failure_type'],
                    'severity': f['severity'],
                    'enable_cascade': f.get('cascade', False)
                }
                for f in scenario['failures']
            ]
            
        elif args.fail_component:
            # Custom failure specification
            components = args.fail_component.split(',')
            fail_time = args.fail_time if args.fail_time else args.duration // 2
            fail_type = args.fail_type if args.fail_type else 'complete'
            
            # Try to infer component types from IDs
            for comp_id in components:
                comp_type = 'application'  # default
                if comp_id.startswith('B'):
                    comp_type = 'broker'
                elif comp_id.startswith('N'):
                    comp_type = 'node'
                elif comp_id.startswith('T'):
                    comp_type = 'topic'
                
                failures.append({
                    'time': fail_time,
                    'component': comp_id.strip(),
                    'component_type': comp_type,
                    'failure_type': fail_type,
                    'severity': 1.0 if fail_type == 'complete' else 0.5,
                    'enable_cascade': args.enable_cascade
                })
        
        # Run simulation
        if failures:
            # Run with failures
            failure_sim = FailureSimulator(
                cascade_threshold=args.cascade_threshold,
                cascade_probability=args.cascade_probability,
                recovery_enabled=args.recovery
            )
            
            results = await run_failure_simulation(
                simulator,
                failure_sim,
                args.duration,
                failures,
                baseline,
                args.monitor,
                args.monitor_interval
            )
        else:
            # Run without failures (traffic simulation only)
            logger.info("\nRunning traffic simulation (no failures)...")
            
            start_time = time.time()
            sim_results = await simulator.run_simulation(args.duration)
            elapsed = time.time() - start_time
            
            results = {
                'duration_seconds': args.duration,
                'execution_time': elapsed,
                'results': sim_results,
                'failures': None,
                'impact': None,
                'recommendations': None
            }
        
        # Print summary
        print_simulation_summary(results, baseline)
        
        # Export results
        if args.export_json or args.export_html:
            output_dir = Path(args.output) if args.output else Path("output")
            
            if args.export_json:
                export_results_json(results, output_dir)
            
            if args.export_html:
                export_results_html(results, output_dir)
        
        logger.info("\n✓ Simulation complete!")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Simulate pub-sub system with message traffic and failures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic traffic simulation
  python simulate_graph.py --input system.json --duration 60

  # Single component failure
  python simulate_graph.py --input system.json --duration 60 \\
      --fail-component A1 --fail-type complete

  # Multiple failures
  python simulate_graph.py --input system.json --duration 60 \\
      --fail-component "A1,B1,N1" --fail-time 30

  # Use predefined scenario
  python simulate_graph.py --input system.json --duration 120 \\
      --scenario cascading-broker

  # With baseline comparison
  python simulate_graph.py --input system.json --duration 60 \\
      --baseline --fail-component B1

  # Real-time monitoring
  python simulate_graph.py --input system.json --duration 60 \\
      --monitor --monitor-interval 5

  # Export comprehensive report
  python simulate_graph.py --input system.json --duration 60 \\
      --scenario stress-test --export-json --export-html

Available scenarios: {', '.join(SIMULATION_SCENARIOS.keys())}
        """
    )
    
    # Input source
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input JSON graph file'
    )
    
    # Simulation parameters
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='Simulation duration in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--baseline', '-b',
        action='store_true',
        help='Run baseline simulation first for comparison'
    )
    
    parser.add_argument(
        '--baseline-duration',
        type=int,
        default=30,
        help='Baseline duration in seconds (default: 30)'
    )
    
    # Failure configuration
    parser.add_argument(
        '--fail-component',
        type=str,
        help='Component(s) to fail (comma-separated, e.g., "A1,B1,N1")'
    )
    
    parser.add_argument(
        '--fail-type',
        type=str,
        choices=['complete', 'partial'],
        default='complete',
        help='Failure type (default: complete)'
    )
    
    parser.add_argument(
        '--fail-time',
        type=int,
        help='Time to inject failure (default: duration/2)'
    )
    
    parser.add_argument(
        '--enable-cascade',
        action='store_true',
        help='Enable cascading failures'
    )
    
    parser.add_argument(
        '--cascade-threshold',
        type=float,
        default=0.7,
        help='Cascade threshold (0-1, default: 0.7)'
    )
    
    parser.add_argument(
        '--cascade-probability',
        type=float,
        default=0.6,
        help='Cascade probability (0-1, default: 0.6)'
    )
    
    parser.add_argument(
        '--recovery',
        action='store_true',
        help='Enable automatic component recovery'
    )
    
    # Scenario selection
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        choices=list(SIMULATION_SCENARIOS.keys()),
        help='Use predefined failure scenario'
    )
    
    # Monitoring
    parser.add_argument(
        '--monitor', '-m',
        action='store_true',
        help='Enable real-time monitoring'
    )
    
    parser.add_argument(
        '--monitor-interval',
        type=int,
        default=5,
        help='Monitoring interval in seconds (default: 5)'
    )
    
    # Export options
    parser.add_argument(
        '--export-json',
        action='store_true',
        help='Export results to JSON'
    )
    
    parser.add_argument(
        '--export-html',
        action='store_true',
        help='Export results to HTML report'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory (default: output/)'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run async main
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
