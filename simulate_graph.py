#!/usr/bin/env python3
"""
Pub-Sub System Failure Simulator
=================================

Simulates component failures in distributed pub-sub systems and measures
impact using DEPENDS_ON relationships.

Usage:
    # Simulate single component failure
    python simulate_graph.py --component app_1
    
    # Simulate with cascade propagation
    python simulate_graph.py --component app_1 --cascade
    
    # Simulate multiple failures
    python simulate_graph.py --components app_1 app_2 app_3
    
    # Run exhaustive simulation (all components)
    python simulate_graph.py --exhaustive
    
    # Load from Neo4j
    python simulate_graph.py --neo4j-uri bolt://localhost:7687 --component broker_1

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from src.analysis import GraphAnalyzer, NEO4J_AVAILABLE
from src.simulation import (
    GraphSimulator,
    SimulationResult,
    BatchSimulationResult,
    FailureMode,
    SimulationMode
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
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.WARNING = ''
        cls.RED = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.DIM = ''
        cls._enabled = False


def print_header(text: str):
    """Print a header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")


def print_section(text: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}── {text} ──{Colors.ENDC}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {text}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.ENDC} {text}")


def severity_color(severity: str) -> str:
    """Get color for severity level"""
    if severity == "CRITICAL":
        return Colors.RED
    elif severity == "HIGH":
        return Colors.WARNING
    elif severity == "MEDIUM":
        return Colors.CYAN
    else:
        return Colors.GREEN


# ============================================================================
# Output Formatters
# ============================================================================

def print_single_result(result: SimulationResult, verbose: bool = False):
    """Print results for a single simulation"""
    
    severity = classify_severity(result.impact_score)
    sev_color = severity_color(severity)
    
    print_section("Simulation Result")
    
    print(f"\n  ID: {result.simulation_id}")
    print(f"  Mode: {result.simulation_mode.value}")
    print(f"  Duration: {(result.end_time - result.start_time).total_seconds() * 1000:.2f}ms")
    
    print_section("Impact Assessment")
    
    print(f"\n  Severity: {sev_color}{Colors.BOLD}{severity}{Colors.ENDC}")
    print(f"  Impact Score: {sev_color}{result.impact_score:.1%}{Colors.ENDC}")
    print(f"  Resilience Score: {result.resilience_score:.1%}")
    
    print_section("Reachability")
    
    print(f"\n  Original Pairs: {result.original_reachability}")
    print(f"  Remaining Pairs: {result.remaining_reachability}")
    print(f"  Lost Pairs: {result.reachability_loss}")
    print(f"  Loss Percentage: {result.reachability_loss_pct:.1f}%")
    
    print_section("Failures")
    
    primary = len(result.failed_components) - len(result.cascade_failures)
    print(f"\n  Primary Failures: {primary}")
    print(f"  Cascade Failures: {len(result.cascade_failures)}")
    print(f"  Total Failed: {len(result.failed_components)}")
    
    if verbose:
        print(f"  Components: {', '.join(result.failed_components)}")
        if result.cascade_failures:
            print(f"  Cascade Chain: {' → '.join(result.cascade_failures)}")
    
    print_section("System State")
    
    print(f"\n  Affected Components: {len(result.affected_components)}")
    print(f"  Isolated Components: {len(result.isolated_components)}")
    print(f"  Fragmentation: {result.fragmentation}")
    
    if verbose and result.affected_components:
        print(f"  Affected: {', '.join(result.affected_components[:10])}")
        if len(result.affected_components) > 10:
            print(f"           ... and {len(result.affected_components) - 10} more")


def print_batch_result(batch: BatchSimulationResult, top_n: int = 10, verbose: bool = False):
    """Print results for batch simulation"""
    
    print_section("Batch Simulation Summary")
    
    print(f"\n  Total Simulations: {batch.total_simulations}")
    
    summary = batch.summary
    
    print_section("Impact Statistics")
    
    print(f"\n  Impact Score:")
    print(f"    Min: {summary['impact_score']['min']:.1%}")
    print(f"    Max: {summary['impact_score']['max']:.1%}")
    print(f"    Mean: {summary['impact_score']['mean']:.1%}")
    
    print(f"\n  Reachability Loss:")
    print(f"    Min: {summary['reachability_loss']['min']:.1f}%")
    print(f"    Max: {summary['reachability_loss']['max']:.1f}%")
    print(f"    Mean: {summary['reachability_loss']['mean']:.1f}%")
    
    print_section(f"Top {top_n} Most Critical Components")
    
    ranking = batch.get_impact_ranking()
    for i, (comp, impact) in enumerate(ranking[:top_n], 1):
        severity = classify_severity(impact)
        sev_color = severity_color(severity)
        print(f"  {i:2d}. {comp:30s} {sev_color}{impact:.1%}{Colors.ENDC} [{severity}]")
    
    if summary['cascade_failures']['total'] > 0:
        print_section("Cascade Statistics")
        print(f"\n  Total Cascade Failures: {summary['cascade_failures']['total']}")
        print(f"  Max Cascade per Sim: {summary['cascade_failures']['max']}")
        print(f"  Simulations with Cascade: {summary['cascade_failures']['simulations_with_cascade']}")


def classify_severity(impact_score: float) -> str:
    """Classify impact severity"""
    if impact_score >= 0.7:
        return "CRITICAL"
    elif impact_score >= 0.5:
        return "HIGH"
    elif impact_score >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"


# ============================================================================
# Export Functions
# ============================================================================

def export_json(result, filepath: Path):
    """Export results to JSON"""
    data = result.to_dict()
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print_success(f"JSON exported to {filepath}")


def export_csv(batch: BatchSimulationResult, filepath: Path):
    """Export batch results to CSV"""
    with open(filepath, 'w') as f:
        # Header
        f.write("component,impact_score,reachability_loss_pct,cascade_count,severity\n")
        
        for result in batch.results:
            if len(result.failed_components) == 1:
                comp = result.failed_components[0]
                severity = classify_severity(result.impact_score)
                f.write(f"{comp},{result.impact_score:.4f},{result.reachability_loss_pct:.2f},"
                       f"{len(result.cascade_failures)},{severity}\n")
    
    print_success(f"CSV exported to {filepath}")


def export_html(result, filepath: Path, title: str = "Simulation Report"):
    """Export results to HTML report"""
    
    is_batch = isinstance(result, BatchSimulationResult)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #333; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .critical {{ color: #dc3545; }}
        .high {{ color: #fd7e14; }}
        .medium {{ color: #17a2b8; }}
        .low {{ color: #28a745; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; }}
        .progress {{ background: #e9ecef; border-radius: 4px; height: 20px; overflow: hidden; }}
        .progress-bar {{ height: 100%; transition: width 0.3s; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
    
    if is_batch:
        summary = result.summary
        html += f"""
    <div class="card">
        <h2>Summary</h2>
        <div class="metric">
            <div class="metric-value">{result.total_simulations}</div>
            <div class="metric-label">Simulations</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['impact_score']['mean']:.1%}</div>
            <div class="metric-label">Mean Impact</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['reachability_loss']['mean']:.1f}%</div>
            <div class="metric-label">Mean Reach Loss</div>
        </div>
    </div>
    
    <div class="card">
        <h2>Component Impact Ranking</h2>
        <table>
            <tr><th>Rank</th><th>Component</th><th>Impact Score</th><th>Severity</th></tr>
"""
        ranking = result.get_impact_ranking()
        for i, (comp, impact) in enumerate(ranking[:20], 1):
            severity = classify_severity(impact)
            html += f"""
            <tr>
                <td>{i}</td>
                <td>{comp}</td>
                <td>
                    <div class="progress">
                        <div class="progress-bar {severity.lower()}" style="width: {impact*100}%"></div>
                    </div>
                    {impact:.1%}
                </td>
                <td class="{severity.lower()}">{severity}</td>
            </tr>
"""
        html += """
        </table>
    </div>
"""
    else:
        severity = classify_severity(result.impact_score)
        html += f"""
    <div class="card">
        <h2>Impact Assessment</h2>
        <div class="metric">
            <div class="metric-value {severity.lower()}">{severity}</div>
            <div class="metric-label">Severity</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.impact_score:.1%}</div>
            <div class="metric-label">Impact Score</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.reachability_loss_pct:.1f}%</div>
            <div class="metric-label">Reachability Loss</div>
        </div>
    </div>
    
    <div class="card">
        <h2>Failed Components</h2>
        <p>Primary: {len(result.failed_components) - len(result.cascade_failures)}, 
           Cascade: {len(result.cascade_failures)}</p>
        <p>{', '.join(result.failed_components)}</p>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(filepath, 'w') as f:
        f.write(html)
    
    print_success(f"HTML report exported to {filepath}")


# ============================================================================
# Main Functions
# ============================================================================

def run_simulation(args) -> int:
    """Run the simulation"""
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger('simulate_graph')
    
    # Handle colors
    if not sys.stdout.isatty() or args.no_color:
        Colors.disable()
    
    # Print header
    if not args.quiet:
        print_header("PUB-SUB FAILURE SIMULATOR")
    
    try:
        # Load data
        if not args.quiet:
            print_section("Loading Data")
        
        analyzer = GraphAnalyzer()
        
        if not NEO4J_AVAILABLE:
            print_error("Neo4j driver not installed. Install with: pip install neo4j")
            return 1
        
        analyzer.load_from_neo4j(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database
        )
        if not args.quiet:
            print_success(f"Loaded from Neo4j: {args.neo4j_uri}")
        
        # Build dependency graph
        if not args.quiet:
            print_section("Building Dependency Graph")
        
        analyzer.derive_depends_on()
        graph = analyzer.build_dependency_graph()
        
        if not args.quiet:
            print_success(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Create simulator
        simulator = GraphSimulator(
            cascade_threshold=args.cascade_threshold,
            cascade_probability=args.cascade_probability,
            max_cascade_depth=args.max_cascade_depth,
            seed=args.seed
        )
        
        # Run simulation
        if not args.quiet:
            print_section("Running Simulation")
        
        result = None
        
        if args.exhaustive:
            # Run exhaustive simulation
            result = simulator.simulate_all_single_failures(
                graph,
                component_types=args.component_types,
                enable_cascade=args.cascade
            )
            if not args.quiet:
                print_success(f"Completed {result.total_simulations} simulations")
                print_batch_result(result, top_n=args.top, verbose=args.verbose)
        
        elif args.components:
            # Multiple component failure
            result = simulator.simulate_multiple_failures(
                graph,
                components=args.components,
                enable_cascade=args.cascade
            )
            if not args.quiet:
                print_success("Simulation completed")
                print_single_result(result, verbose=args.verbose)
        
        elif args.component:
            # Single component failure
            result = simulator.simulate_failure(
                graph,
                component=args.component,
                enable_cascade=args.cascade
            )
            if not args.quiet:
                print_success("Simulation completed")
                print_single_result(result, verbose=args.verbose)
        
        else:
            print_error("Specify --component, --components, or --exhaustive")
            return 1
        
        # Export results
        if args.output_dir and result:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not args.quiet:
                print_section("Exporting Results")
            
            for fmt in args.format:
                if fmt == 'json':
                    export_json(result, output_dir / 'simulation_results.json')
                elif fmt == 'html':
                    export_html(result, output_dir / 'simulation_report.html')
                elif fmt == 'csv' and isinstance(result, BatchSimulationResult):
                    export_csv(result, output_dir / 'impact_scores.csv')
        
        # JSON to stdout
        if args.json_output and result:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        
        return 0
        
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid input: {e}")
        return 1
    except Exception as e:
        print_error(f"Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Simulate failures in pub-sub systems using DEPENDS_ON relationships',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single component failure
    python simulate_graph.py --component app_1
    
    # With cascade propagation
    python simulate_graph.py --component broker_1 --cascade
    
    # Multiple simultaneous failures
    python simulate_graph.py --components app_1 app_2 broker_1
    
    # Exhaustive simulation (all components)
    python simulate_graph.py --exhaustive
    
    # Filter by component type
    python simulate_graph.py --exhaustive --component-types Application Broker
    
    # Export results
    python simulate_graph.py --exhaustive --output-dir results/ --format json html csv
    
    # Load from Neo4j
    python simulate_graph.py --neo4j-uri bolt://localhost:7687 --component app_1
        """
    )
    
    # Neo4j options
    neo4j_group = parser.add_argument_group('Neo4j Connection')
    neo4j_group.add_argument('--neo4j-uri', default='bolt://localhost:7687',
                             help='Neo4j URI (default: bolt://localhost:7687)')
    neo4j_group.add_argument('--neo4j-user', default='neo4j',
                             help='Neo4j username (default: neo4j)')
    neo4j_group.add_argument('--neo4j-password', default='password',
                             help='Neo4j password (default: password)')
    neo4j_group.add_argument('--neo4j-database', default='neo4j',
                             help='Neo4j database name (default: neo4j)')
    
    # Simulation target
    target_group = parser.add_argument_group('Simulation Target')
    target_group.add_argument('--component', '-c',
                              help='Single component to fail')
    target_group.add_argument('--components', '-C', nargs='+',
                              help='Multiple components to fail simultaneously')
    target_group.add_argument('--exhaustive', '-e', action='store_true',
                              help='Test all components individually')
    target_group.add_argument('--component-types', nargs='+',
                              help='Filter exhaustive by types (e.g., Application Broker)')
    
    # Simulation options
    sim_group = parser.add_argument_group('Simulation Options')
    sim_group.add_argument('--cascade', action='store_true',
                           help='Enable cascade failure propagation')
    sim_group.add_argument('--cascade-threshold', type=float, default=0.7,
                           help='Dependency loss ratio to trigger cascade (default: 0.7)')
    sim_group.add_argument('--cascade-probability', type=float, default=0.5,
                           help='Base probability of cascade (default: 0.5)')
    sim_group.add_argument('--max-cascade-depth', type=int, default=5,
                           help='Maximum cascade propagation depth (default: 5)')
    sim_group.add_argument('--seed', type=int,
                           help='Random seed for reproducibility')
    
    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', '-o',
                              help='Output directory for exports')
    output_group.add_argument('--format', '-f', nargs='+',
                              choices=['json', 'html', 'csv'],
                              default=['json', 'html'],
                              help='Export formats (default: json html)')
    output_group.add_argument('--top', type=int, default=10,
                              help='Number of top components to show (default: 10)')
    
    # General options
    parser.add_argument('--json-output', action='store_true',
                        help='Print JSON results to stdout')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    
    args = parser.parse_args()
    
    return run_simulation(args)


if __name__ == '__main__':
    sys.exit(main())