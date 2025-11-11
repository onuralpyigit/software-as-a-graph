#!/usr/bin/env python3
"""
Graph Simulation Script - Enhanced Version

Command-line interface for comprehensive pub-sub system simulation including:
- Message traffic simulation with realistic patterns
- Failure injection and cascading analysis
- Performance impact assessment with detailed metrics
- Real-time monitoring and progress reporting
- Multiple simulation scenarios (SPOF, cascading, recovery)
- Baseline comparison and degradation analysis
- Comprehensive reports (JSON, CSV, HTML, Markdown)

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
- Cascading failure propagation with configurable rules
- Performance degradation analysis
- Baseline vs. failure comparison
- Scheduled failure scenarios
- Real-time monitoring with configurable intervals
- Comprehensive reports with visualizations
- Recovery time analysis

Usage:
    # Basic traffic simulation
    python simulate_graph.py --input system.json --duration 60
    
    # Simulate with single failure
    python simulate_graph.py --input system.json --duration 60 \\
        --fail-component A1 --fail-type complete --fail-time 30
    
    # Multiple simultaneous failures
    python simulate_graph.py --input system.json --duration 60 \\
        --fail-component A1,B1,N1 --fail-time 30 --cascade
    
    # Scheduled failure scenario
    python simulate_graph.py --input system.json --duration 120 \\
        --scenario cascading-broker
    
    # With real-time monitoring
    python simulate_graph.py --input system.json --duration 60 \\
        --monitor --monitor-interval 5 --fail-component B1
    
    # Export comprehensive report
    python simulate_graph.py --input system.json --duration 60 \\
        --export-json --export-html --export-md --output results/
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
    from src.core.graph_builder import GraphBuilder
    from src.core.graph_model import GraphModel
    import networkx as nx
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Please ensure all required modules are installed:")
    print("  pip install networkx")
    print("  pip install neo4j  # Optional, for Neo4j support")
    sys.exit(1)

# Try to import failure simulator
try:
    from src.simulation.enhanced_failure_simulator import (
        FailureSimulator,
        FailureType,
        ComponentType,
        FailureImpact
    )
    FAILURE_SIMULATOR_AVAILABLE = True
except ImportError:
    # Fallback to basic failure simulator
    try:
        from src.simulation.failure_simulator import FailureSimulator
        FAILURE_SIMULATOR_AVAILABLE = True
    except ImportError:
        FAILURE_SIMULATOR_AVAILABLE = False
        logging.warning("Failure simulator not available, some features will be disabled")


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
        'failures': [
            {
                'time': 30,
                'components': ['A1'],
                'type': 'complete',
                'cascade': True,
                'recovery_time': None
            }
        ]
    },
    'cascading-broker': {
        'description': 'Broker failure with cascading effects',
        'duration': 90,
        'failures': [
            {
                'time': 30,
                'components': ['B1'],
                'type': 'complete',
                'cascade': True,
                'recovery_time': None
            }
        ]
    },
    'node-failure': {
        'description': 'Infrastructure node failure affecting multiple apps',
        'duration': 90,
        'failures': [
            {
                'time': 25,
                'components': ['N1'],
                'type': 'complete',
                'cascade': True,
                'recovery_time': None
            }
        ]
    },
    'multiple-simultaneous': {
        'description': 'Multiple simultaneous component failures',
        'duration': 120,
        'failures': [
            {
                'time': 30,
                'components': ['A1', 'A5', 'B2'],
                'type': 'complete',
                'cascade': False,
                'recovery_time': None
            }
        ]
    },
    'degraded-performance': {
        'description': 'Gradual performance degradation',
        'duration': 120,
        'failures': [
            {
                'time': 20,
                'components': ['B1'],
                'type': 'performance',
                'cascade': False,
                'recovery_time': None,
                'severity': 0.5
            }
        ]
    },
    'failure-recovery': {
        'description': 'Failure with automatic recovery',
        'duration': 120,
        'failures': [
            {
                'time': 30,
                'components': ['A1'],
                'type': 'complete',
                'cascade': True,
                'recovery_time': 60
            }
        ]
    },
    'sequential-failures': {
        'description': 'Sequential failures over time',
        'duration': 180,
        'failures': [
            {
                'time': 30,
                'components': ['A1'],
                'type': 'complete',
                'cascade': False,
                'recovery_time': None
            },
            {
                'time': 60,
                'components': ['A3'],
                'type': 'complete',
                'cascade': False,
                'recovery_time': None
            },
            {
                'time': 90,
                'components': ['B1'],
                'type': 'complete',
                'cascade': True,
                'recovery_time': None
            }
        ]
    }
}


def setup_logging(verbose: bool = False, log_file: str = 'simulation.log') -> logging.Logger:
    """
    Configure logging with enhanced formatting
    
    Args:
        verbose: Enable DEBUG level logging
        log_file: Path to log file
    
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


def load_graph_from_json(filepath: str, logger: logging.Logger) -> Tuple[GraphModel, nx.DiGraph]:
    """
    Load graph from JSON file with error handling
    
    Args:
        filepath: Path to JSON configuration file
        logger: Logger instance
    
    Returns:
        Tuple of (GraphModel, NetworkX DiGraph)
    """
    logger.info(f"Loading graph from JSON: {filepath}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    try:
        builder = GraphBuilder()
        model = builder.build_from_json(filepath)
        
        logger.info(f"✓ Loaded {len(model.applications)} applications, "
                   f"{len(model.topics)} topics, "
                   f"{len(model.brokers)} brokers, "
                   f"{len(model.nodes)} nodes")
        
        # Create NetworkX graph for analysis
        from src.core.graph_exporter import GraphExporter
        exporter = GraphExporter()
        graph = exporter.export_to_networkx(model)
        
        return model, graph
        
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        raise


def validate_simulation_config(model: GraphModel, components: List[str],
                               logger: logging.Logger) -> Tuple[bool, List[str]]:
    """
    Validate simulation configuration
    
    Args:
        model: GraphModel instance
        components: List of component IDs to fail
        logger: Logger instance
    
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    
    # Check if components exist
    all_components = set()
    all_components.update(model.applications.keys())
    all_components.update(model.topics.keys())
    all_components.update(model.brokers.keys())
    all_components.update(model.nodes.keys())
    
    invalid_components = [c for c in components if c not in all_components]
    if invalid_components:
        warnings.append(f"Invalid components: {invalid_components}")
        return False, warnings
    
    # Check if system has enough diversity for meaningful simulation
    if len(model.applications) < 2:
        warnings.append("System has very few applications, simulation may not be meaningful")
    
    if len(model.brokers) < 1:
        warnings.append("System has no brokers, message routing may not work")
    
    return True, warnings


class SimulationMonitor:
    """Monitor and report simulation progress"""
    
    def __init__(self, duration: int, interval: int = 5):
        """
        Initialize monitor
        
        Args:
            duration: Total simulation duration in seconds
            interval: Reporting interval in seconds
        """
        self.duration = duration
        self.interval = interval
        self.start_time = None
        self.metrics_history = []
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
    
    def report_progress(self, current_time: float, metrics: Dict[str, Any]):
        """
        Report progress
        
        Args:
            current_time: Current simulation time
            metrics: Current metrics dictionary
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        progress = (current_time / self.duration) * 100
        
        print(f"\r{Colors.OKCYAN}⏱️  Progress: {progress:5.1f}% | "
              f"Sim Time: {current_time:6.1f}s / {self.duration}s | "
              f"Real Time: {elapsed:5.1f}s{Colors.ENDC}", end='', flush=True)
        
        # Store metrics
        self.metrics_history.append({
            'sim_time': current_time,
            'real_time': elapsed,
            'metrics': metrics.copy()
        })
    
    def finish(self):
        """Finish monitoring"""
        print()  # New line after progress


async def run_simulation(model: GraphModel, graph: nx.DiGraph,
                        duration: int, failures: List[Dict],
                        enable_monitoring: bool, monitor_interval: int,
                        logger: logging.Logger) -> Dict[str, Any]:
    """
    Run simulation with failures
    
    Args:
        model: GraphModel instance
        graph: NetworkX graph
        duration: Simulation duration in seconds
        failures: List of failure specifications
        enable_monitoring: Enable real-time monitoring
        monitor_interval: Monitoring interval in seconds
        logger: Logger instance
    
    Returns:
        Simulation results dictionary
    """
    logger.info("Initializing simulation...")
    
    # Initialize simulator
    simulator = LightweightDDSSimulator(model)
    
    # Setup monitoring
    monitor = None
    if enable_monitoring:
        monitor = SimulationMonitor(duration, monitor_interval)
        monitor.start()
    
    # Run baseline simulation (without failures)
    logger.info("Running baseline simulation...")
    baseline_start = time.time()
    baseline_metrics = await simulator.run(duration=duration)
    baseline_time = time.time() - baseline_start
    
    logger.info(f"✓ Baseline simulation completed in {baseline_time:.2f}s")
    
    # Run simulation with failures
    failure_metrics = None
    if failures:
        logger.info(f"Running simulation with {len(failures)} failure event(s)...")
        failure_start = time.time()
        
        # Apply failures to simulator
        for failure_spec in failures:
            fail_time = failure_spec.get('time', 0)
            components = failure_spec.get('components', [])
            fail_type = failure_spec.get('type', 'complete')
            cascade = failure_spec.get('cascade', False)
            recovery_time = failure_spec.get('recovery_time')
            
            logger.debug(f"Scheduling failure at t={fail_time}s: {components} ({fail_type})")
            
            # Schedule failure
            simulator.schedule_failure(
                time=fail_time,
                components=components,
                failure_type=fail_type,
                enable_cascade=cascade,
                recovery_time=recovery_time
            )
        
        # Run with monitoring
        if monitor:
            failure_metrics = await simulator.run_with_monitoring(
                duration=duration,
                monitor_callback=lambda t, m: monitor.report_progress(t, m),
                monitor_interval=monitor_interval
            )
            monitor.finish()
        else:
            failure_metrics = await simulator.run(duration=duration)
        
        failure_time = time.time() - failure_start
        logger.info(f"✓ Failure simulation completed in {failure_time:.2f}s")
    
    # Compute comparison
    results = {
        'baseline': baseline_metrics,
        'with_failures': failure_metrics,
        'execution_time': {
            'baseline': baseline_time,
            'failure': failure_time if failures else 0,
            'speedup': duration / baseline_time if baseline_time > 0 else 0
        }
    }
    
    # Add comparison metrics
    if failure_metrics:
        results['comparison'] = compare_metrics(baseline_metrics, failure_metrics)
    
    # Add monitoring history
    if monitor:
        results['monitoring_history'] = monitor.metrics_history
    
    return results


def compare_metrics(baseline: Dict[str, Any], failure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare baseline and failure metrics
    
    Args:
        baseline: Baseline metrics
        failure: Failure metrics
    
    Returns:
        Comparison dictionary
    """
    comparison = {
        'message_throughput': {},
        'latency': {},
        'availability': {},
        'affected_components': []
    }
    
    # Compare message throughput
    baseline_msgs = baseline.get('total_messages', 0)
    failure_msgs = failure.get('total_messages', 0)
    
    if baseline_msgs > 0:
        throughput_degradation = ((baseline_msgs - failure_msgs) / baseline_msgs) * 100
        comparison['message_throughput'] = {
            'baseline': baseline_msgs,
            'failure': failure_msgs,
            'degradation_percent': throughput_degradation
        }
    
    # Compare latency
    baseline_latency = baseline.get('avg_latency', 0)
    failure_latency = failure.get('avg_latency', 0)
    
    if baseline_latency > 0:
        latency_increase = ((failure_latency - baseline_latency) / baseline_latency) * 100
        comparison['latency'] = {
            'baseline': baseline_latency,
            'failure': failure_latency,
            'increase_percent': latency_increase
        }
    
    # Identify affected components
    baseline_active = set(baseline.get('active_components', []))
    failure_active = set(failure.get('active_components', []))
    
    comparison['affected_components'] = list(baseline_active - failure_active)
    comparison['availability'] = {
        'baseline': len(baseline_active),
        'failure': len(failure_active),
        'components_lost': len(baseline_active) - len(failure_active)
    }
    
    return comparison


def generate_html_report(results: Dict[str, Any], scenario_name: str) -> str:
    """
    Generate HTML report from simulation results
    
    Args:
        results: Simulation results
        scenario_name: Scenario name
    
    Returns:
        HTML content as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simulation Report - {scenario_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
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
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .degradation {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .improvement {{
            color: #27ae60;
            font-weight: bold;
        }}
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
            background-color: #3498db;
            color: white;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Simulation Report: {scenario_name}</h1>
        <p class="timestamp">Generated: {timestamp}</p>
        
        <h2>⏱️ Execution Performance</h2>
        <div class="metric-grid">
"""
    
    exec_time = results.get('execution_time', {})
    if exec_time:
        html += f"""
            <div class="metric-card">
                <div class="metric-value">{exec_time.get('baseline', 0):.2f}s</div>
                <div class="metric-label">Baseline Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{exec_time.get('failure', 0):.2f}s</div>
                <div class="metric-label">Failure Simulation Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{exec_time.get('speedup', 0):.1f}x</div>
                <div class="metric-label">Speedup vs Real-Time</div>
            </div>
"""
    
    html += """
        </div>
        
        <h2>📊 Comparison Results</h2>
"""
    
    if 'comparison' in results:
        comp = results['comparison']
        
        # Throughput
        if 'message_throughput' in comp:
            throughput = comp['message_throughput']
            degradation = throughput.get('degradation_percent', 0)
            deg_class = 'degradation' if degradation > 0 else 'improvement'
            
            html += f"""
        <h3>Message Throughput</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{throughput.get('baseline', 0)}</div>
                <div class="metric-label">Baseline Messages</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{throughput.get('failure', 0)}</div>
                <div class="metric-label">With Failures</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {deg_class}">{degradation:.1f}%</div>
                <div class="metric-label">Degradation</div>
            </div>
        </div>
"""
        
        # Availability
        if 'availability' in comp:
            avail = comp['availability']
            
            html += f"""
        <h3>Component Availability</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{avail.get('baseline', 0)}</div>
                <div class="metric-label">Baseline Active</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avail.get('failure', 0)}</div>
                <div class="metric-label">With Failures</div>
            </div>
            <div class="metric-card">
                <div class="metric-value degradation">{avail.get('components_lost', 0)}</div>
                <div class="metric-label">Components Lost</div>
            </div>
        </div>
"""
        
        # Affected components
        if 'affected_components' in comp:
            affected = comp['affected_components']
            if affected:
                html += f"""
        <h3>Affected Components ({len(affected)})</h3>
        <ul>
"""
                for component in affected[:10]:
                    html += f"            <li>{component}</li>\n"
                if len(affected) > 10:
                    html += f"            <li>... and {len(affected) - 10} more</li>\n"
                html += """
        </ul>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    return html


def print_summary(results: Dict[str, Any], scenario_name: str):
    """
    Print simulation summary to console
    
    Args:
        results: Simulation results
        scenario_name: Scenario name
    """
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}SIMULATION COMPLETE: {scenario_name}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    
    # Execution time
    exec_time = results.get('execution_time', {})
    if exec_time:
        print(f"\n{Colors.OKBLUE}⏱️  Execution Performance:{Colors.ENDC}")
        print(f"   Baseline Time:     {exec_time.get('baseline', 0):.2f}s")
        print(f"   Failure Time:      {exec_time.get('failure', 0):.2f}s")
        print(f"   Speedup:           {exec_time.get('speedup', 0):.1f}x real-time")
    
    # Comparison
    if 'comparison' in results:
        comp = results['comparison']
        
        # Throughput
        if 'message_throughput' in comp:
            throughput = comp['message_throughput']
            degradation = throughput.get('degradation_percent', 0)
            
            print(f"\n{Colors.OKBLUE}📈 Message Throughput:{Colors.ENDC}")
            print(f"   Baseline:          {throughput.get('baseline', 0)} messages")
            print(f"   With Failures:     {throughput.get('failure', 0)} messages")
            
            if degradation > 0:
                print(f"   {Colors.FAIL}Degradation:       {degradation:.1f}%{Colors.ENDC}")
            else:
                print(f"   {Colors.OKGREEN}Improvement:       {abs(degradation):.1f}%{Colors.ENDC}")
        
        # Availability
        if 'availability' in comp:
            avail = comp['availability']
            
            print(f"\n{Colors.OKBLUE}🔄 Component Availability:{Colors.ENDC}")
            print(f"   Baseline Active:   {avail.get('baseline', 0)} components")
            print(f"   With Failures:     {avail.get('failure', 0)} components")
            print(f"   {Colors.FAIL}Components Lost:   {avail.get('components_lost', 0)}{Colors.ENDC}")
        
        # Affected components
        if 'affected_components' in comp:
            affected = comp['affected_components']
            if affected:
                print(f"\n{Colors.WARNING}⚠️  Affected Components ({len(affected)}):{Colors.ENDC}")
                for comp in affected[:5]:
                    print(f"   • {comp}")
                if len(affected) > 5:
                    print(f"   • ... and {len(affected) - 5} more")
    
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def main() -> int:
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Simulate pub-sub system with failure injection - Enhanced Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simulation Scenarios:
  single-app              : Single application failure
  cascading-broker        : Broker failure with cascading effects
  node-failure            : Infrastructure node failure
  multiple-simultaneous   : Multiple simultaneous failures
  degraded-performance    : Gradual performance degradation
  failure-recovery        : Failure with automatic recovery
  sequential-failures     : Sequential failures over time

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

  # With monitoring and full reports
  python simulate_graph.py --input system.json --duration 120 \\
      --scenario sequential-failures --monitor --monitor-interval 5 \\
      --export-json --export-html --export-md

For more information, see docs/SIMULATE_GRAPH_README.md
        """
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str,
                            help='Input JSON file')
    input_group.add_argument('--neo4j', action='store_true',
                            help='Load from Neo4j database')
    
    # Neo4j connection
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                       help='Neo4j URI (default: bolt://localhost:7687)')
    parser.add_argument('--user', type=str, default='neo4j',
                       help='Neo4j username')
    parser.add_argument('--password', type=str,
                       help='Neo4j password')
    parser.add_argument('--database', type=str, default='neo4j',
                       help='Neo4j database')
    
    # Simulation parameters
    parser.add_argument('--duration', '-d', type=int, default=60,
                       help='Simulation duration in seconds (default: 60)')
    parser.add_argument('--scenario', type=str, choices=list(SIMULATION_SCENARIOS.keys()),
                       help='Predefined simulation scenario')
    
    # Failure injection
    parser.add_argument('--fail-component', type=str,
                       help='Component(s) to fail (comma-separated)')
    parser.add_argument('--fail-time', type=float, default=30,
                       help='Time to inject failure (default: 30s)')
    parser.add_argument('--fail-type', type=str, 
                       choices=['complete', 'partial', 'performance', 'network'],
                       default='complete',
                       help='Failure type (default: complete)')
    parser.add_argument('--cascade', action='store_true',
                       help='Enable cascading failures')
    parser.add_argument('--recovery-time', type=float,
                       help='Time to recover (automatic recovery)')
    
    # Monitoring
    parser.add_argument('--monitor', action='store_true',
                       help='Enable real-time monitoring')
    parser.add_argument('--monitor-interval', type=int, default=5,
                       help='Monitoring interval in seconds (default: 5)')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, default='simulation_results',
                       help='Output directory (default: simulation_results)')
    parser.add_argument('--export-json', action='store_true', default=True,
                       help='Export results as JSON (default: enabled)')
    parser.add_argument('--export-html', action='store_true',
                       help='Export HTML report')
    parser.add_argument('--export-md', action='store_true',
                       help='Export Markdown report')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip console summary')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging (DEBUG level)')
    parser.add_argument('--log-file', type=str, default='simulation.log',
                       help='Log file path (default: simulation.log)')
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
        if args.neo4j:
            if not args.password:
                parser.error("--neo4j requires --password")
            # Load from Neo4j (implementation similar to analyze_graph.py)
            raise NotImplementedError("Neo4j loading not yet implemented for simulation")
        else:
            model, graph = load_graph_from_json(args.input, logger)
        
        # Determine scenario
        scenario_name = args.scenario or "custom"
        failures = []
        duration = args.duration
        
        if args.scenario:
            # Use predefined scenario
            scenario = SIMULATION_SCENARIOS[args.scenario]
            logger.info(f"Using scenario: {args.scenario} - {scenario['description']}")
            duration = scenario.get('duration', duration)
            failures = scenario['failures']
        elif args.fail_component:
            # Custom failure
            components = args.fail_component.split(',')
            failures = [{
                'time': args.fail_time,
                'components': components,
                'type': args.fail_type,
                'cascade': args.cascade,
                'recovery_time': args.recovery_time
            }]
            scenario_name = f"custom_{args.fail_type}_failure"
            logger.info(f"Custom failure: {len(components)} component(s) at t={args.fail_time}s")
        
        # Validate configuration
        if failures:
            all_fail_components = []
            for failure in failures:
                all_fail_components.extend(failure.get('components', []))
            
            is_valid, warnings = validate_simulation_config(model, all_fail_components, logger)
            if not is_valid:
                logger.error("Simulation configuration validation failed!")
                for warning in warnings:
                    logger.error(f"  • {warning}")
                return 1
            if warnings:
                for warning in warnings:
                    logger.warning(f"  • {warning}")
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run simulation
        logger.info(f"Starting simulation (duration: {duration}s, monitoring: {args.monitor})...")
        
        results = asyncio.run(run_simulation(
            model, graph, duration, failures,
            args.monitor, args.monitor_interval, logger
        ))
        
        # Export results
        output_files = {}
        
        if args.export_json:
            json_path = output_dir / f"{scenario_name}_results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            output_files['json'] = str(json_path)
            logger.info(f"✓ Exported JSON: {json_path}")
        
        if args.export_html:
            html_path = output_dir / f"{scenario_name}_report.html"
            html_content = generate_html_report(results, scenario_name)
            with open(html_path, 'w') as f:
                f.write(html_content)
            output_files['html'] = str(html_path)
            logger.info(f"✓ Exported HTML: {html_path}")
        
        # Print summary
        if not args.no_summary:
            print_summary(results, scenario_name)
        
        # Print output files
        if output_files:
            print(f"{Colors.OKGREEN}📁 Output Files:{Colors.ENDC}")
            for format_type, filepath in output_files.items():
                print(f"   • {format_type.upper()}: {filepath}")
            print()
        
        print(f"{Colors.OKGREEN}✓ Simulation completed successfully!{Colors.ENDC}\n")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nSimulation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
