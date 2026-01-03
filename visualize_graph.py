#!/usr/bin/env python3
"""
Visualize Graph CLI - Version 5.0

Generate visualizations and dashboards for graph analysis.

Input Sources:
- JSON file (--input)
- Neo4j database (--neo4j)

Dashboard Types:
- graph: Graph model statistics
- simulation: Failure simulation results
- validation: Validation results
- overview: Combined dashboard

Usage:
    # Graph model dashboard
    python visualize_graph.py --input graph.json --type graph
    
    # Simulation dashboard
    python visualize_graph.py --input graph.json --type simulation
    
    # Validation dashboard
    python visualize_graph.py --input graph.json --type validation
    
    # Combined overview
    python visualize_graph.py --input graph.json --type overview
    
    # From Neo4j
    python visualize_graph.py --neo4j --type overview
    
    # Custom output
    python visualize_graph.py --input graph.json --output dashboard.html

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import json
import logging
import sys
import webbrowser
from pathlib import Path
from typing import Optional


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    
    @classmethod
    def disable(cls):
        for attr in ["BOLD", "GREEN", "BLUE", "CYAN", "YELLOW", "RED", "GRAY", "RESET"]:
            setattr(cls, attr, "")


def use_colors() -> bool:
    import os
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.name != "nt"


# =============================================================================
# Output Helpers
# =============================================================================

def print_header(title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def print_section(title: str) -> None:
    print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{'-' * 40}")


def print_success(msg: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_error(msg: str) -> None:
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_info(msg: str) -> None:
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {msg}")


# =============================================================================
# Graph Loading
# =============================================================================

def load_graph_from_file(args):
    """Load simulation graph from JSON file."""
    from src.simulation import SimulationGraph
    
    if not args.input:
        return None
    
    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"File not found: {input_path}")
        return None
    
    print_info(f"Loading graph from {input_path}")
    
    try:
        graph = SimulationGraph.from_json(str(input_path))
        summary = graph.summary()
        print_success(f"Loaded {summary['total_components']} components, {summary['total_edges']} edges")
        return graph
    except Exception as e:
        print_error(f"Failed to load graph: {e}")
        return None


def load_graph_from_neo4j(args):
    """Load simulation graph from Neo4j database."""
    from src.simulation import Neo4jSimulationClient, check_neo4j_available
    
    if not check_neo4j_available():
        print_error("Neo4j driver not installed. Install with: pip install neo4j")
        return None
    
    print_info(f"Connecting to Neo4j at {args.uri}")
    
    try:
        with Neo4jSimulationClient(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        ) as client:
            if not client.verify_connection():
                print_error("Failed to connect to Neo4j")
                return None
            
            print_success("Connected to Neo4j")
            
            stats = client.get_statistics()
            print_info(f"Components in DB: {stats['total_components']}")
            
            if args.layer:
                print_info(f"Loading layer: {args.layer}")
                graph = client.load_layer(args.layer)
            else:
                print_info("Loading full graph")
                graph = client.load_full_graph()
            
            summary = graph.summary()
            print_success(f"Loaded {summary['total_components']} components, {summary['total_edges']} edges")
            return graph
            
    except Exception as e:
        print_error(f"Failed to load from Neo4j: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


def load_graph(args):
    """Load graph from file or Neo4j."""
    if args.neo4j:
        return load_graph_from_neo4j(args)
    elif args.input:
        return load_graph_from_file(args)
    else:
        print_error("Specify --input <file> or --neo4j")
        return None


# =============================================================================
# Dashboard Generation
# =============================================================================

def generate_graph_dashboard_cli(args, graph) -> Optional[str]:
    """Generate graph model dashboard."""
    from src.visualization import generate_graph_dashboard, DashboardConfig
    
    print_section("Generating Graph Dashboard")
    
    config = DashboardConfig(title=args.title or "Graph Model Dashboard")
    html = generate_graph_dashboard(graph, config=config)
    
    print_success("Graph dashboard generated")
    return html


def generate_simulation_dashboard_cli(args, graph) -> Optional[str]:
    """Generate simulation dashboard."""
    from src.simulation import FailureSimulator
    from src.visualization import generate_simulation_dashboard, DashboardConfig
    
    print_section("Running Failure Simulation")
    
    simulator = FailureSimulator(
        seed=args.seed,
        cascade=args.cascade,
    )
    
    campaign = simulator.simulate_all(graph)
    
    print_success(f"Simulated {campaign.total_simulations} failures")
    print_info(f"Critical components: {len(campaign.get_critical())}")
    print_info(f"Max impact: {campaign.max_impact:.4f}")
    
    print_section("Generating Simulation Dashboard")
    
    config = DashboardConfig(title=args.title or "Simulation Results Dashboard")
    html = generate_simulation_dashboard(graph, campaign, config=config)
    
    print_success("Simulation dashboard generated")
    return html


def generate_validation_dashboard_cli(args, graph) -> Optional[str]:
    """Generate validation dashboard."""
    from src.validation import ValidationPipeline, AnalysisMethod
    from src.visualization import generate_validation_dashboard, DashboardConfig
    
    print_section("Running Validation Pipeline")
    
    pipeline = ValidationPipeline(seed=args.seed, cascade=args.cascade)
    
    method = AnalysisMethod(args.method)
    result = pipeline.run(graph, analysis_method=method, compare_methods=args.compare_methods)
    
    status_color = Colors.GREEN if result.validation.status.value == "passed" else Colors.YELLOW
    print_success(f"Status: {status_color}{result.validation.status.value.upper()}{Colors.RESET}")
    print_info(f"Spearman ρ: {result.spearman:.4f}")
    print_info(f"F1-Score: {result.f1_score:.4f}")
    
    print_section("Generating Validation Dashboard")
    
    config = DashboardConfig(title=args.title or "Validation Results Dashboard")
    html = generate_validation_dashboard(result, config=config)
    
    print_success("Validation dashboard generated")
    return html


def generate_overview_dashboard_cli(args, graph) -> Optional[str]:
    """Generate combined overview dashboard."""
    from src.simulation import FailureSimulator
    from src.validation import ValidationPipeline, AnalysisMethod
    from src.visualization import generate_overview_dashboard, DashboardConfig
    
    campaign = None
    validation = None
    
    # Run simulation
    print_section("Running Failure Simulation")
    
    simulator = FailureSimulator(seed=args.seed, cascade=args.cascade)
    campaign = simulator.simulate_all(graph)
    
    print_success(f"Simulated {campaign.total_simulations} failures")
    
    # Run validation
    print_section("Running Validation Pipeline")
    
    pipeline = ValidationPipeline(seed=args.seed, cascade=args.cascade)
    method = AnalysisMethod(args.method)
    validation = pipeline.run(graph, analysis_method=method, compare_methods=args.compare_methods)
    
    status_color = Colors.GREEN if validation.validation.status.value == "passed" else Colors.YELLOW
    print_success(f"Status: {status_color}{validation.validation.status.value.upper()}{Colors.RESET}")
    
    # Generate dashboard
    print_section("Generating Overview Dashboard")
    
    config = DashboardConfig(title=args.title or "System Overview Dashboard")
    html = generate_overview_dashboard(
        graph,
        campaign_result=campaign,
        validation_result=validation,
        config=config,
    )
    
    print_success("Overview dashboard generated")
    return html


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate visualizations for graph analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Graph model dashboard
    python visualize_graph.py --input graph.json --type graph
    
    # Simulation dashboard
    python visualize_graph.py --input graph.json --type simulation
    
    # Validation dashboard
    python visualize_graph.py --input graph.json --type validation
    
    # Combined overview
    python visualize_graph.py --input graph.json --type overview
    
    # From Neo4j
    python visualize_graph.py --neo4j --type overview
    
    # Custom output and open in browser
    python visualize_graph.py --input graph.json --output dash.html --open

Dashboard Types:
    graph      - Graph model statistics (components, edges, layers)
    simulation - Failure simulation results (impact ranking, critical)
    validation - Validation results (correlation, classification)
    overview   - Combined dashboard with all sections
        """
    )
    
    # Input source
    input_group = parser.add_argument_group("Input Source")
    input_group.add_argument(
        "--input", "-i",
        help="Input graph JSON file"
    )
    input_group.add_argument(
        "--neo4j", "-n",
        action="store_true",
        help="Load graph from Neo4j database"
    )
    
    # Neo4j options
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI (default: bolt://localhost:7687)"
    )
    neo4j_group.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--password",
        default="password",
        help="Neo4j password (default: password)"
    )
    neo4j_group.add_argument(
        "--database",
        default="neo4j",
        help="Neo4j database name (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--layer",
        choices=["application", "infrastructure", "app_broker", "node_broker"],
        help="Load specific layer from Neo4j"
    )
    
    # Dashboard options
    dash_group = parser.add_argument_group("Dashboard Options")
    dash_group.add_argument(
        "--type", "-t",
        choices=["graph", "simulation", "validation", "overview"],
        default="overview",
        help="Dashboard type (default: overview)"
    )
    dash_group.add_argument(
        "--title",
        help="Custom dashboard title"
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--method", "-m",
        choices=["betweenness", "pagerank", "degree", "composite"],
        default="composite",
        help="Analysis method (default: composite)"
    )
    analysis_group.add_argument(
        "--compare-methods", "-c",
        action="store_true",
        help="Compare all analysis methods"
    )
    analysis_group.add_argument(
        "--cascade",
        action="store_true",
        default=True,
        help="Enable cascade propagation (default)"
    )
    analysis_group.add_argument(
        "--no-cascade",
        action="store_false",
        dest="cascade",
        help="Disable cascade propagation"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        help="Output HTML file (default: dashboard.html)"
    )
    output_group.add_argument(
        "--open",
        action="store_true",
        help="Open dashboard in browser after generation"
    )
    
    # Common options
    parser.add_argument(
        "--seed", "-s",
        type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colors"
    )
    
    args = parser.parse_args()
    
    # Handle colors
    if args.no_color or not use_colors():
        Colors.disable()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level)
    
    # Check matplotlib
    from src.visualization import check_matplotlib_available
    if not check_matplotlib_available():
        print_error("matplotlib not installed. Install with: pip install matplotlib")
        print_info("Dashboards will be generated without charts")
    
    # Load graph
    graph = load_graph(args)
    if graph is None:
        return 1
    
    # Generate dashboard
    print_header("VISUALIZATION DASHBOARD")
    
    html = None
    
    if args.type == "graph":
        html = generate_graph_dashboard_cli(args, graph)
    elif args.type == "simulation":
        html = generate_simulation_dashboard_cli(args, graph)
    elif args.type == "validation":
        html = generate_validation_dashboard_cli(args, graph)
    elif args.type == "overview":
        html = generate_overview_dashboard_cli(args, graph)
    
    if html is None:
        print_error("Failed to generate dashboard")
        return 1
    
    # Save output
    output_path = Path(args.output or "dashboard.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(html)
    
    print_success(f"Dashboard saved to {output_path}")
    
    # Open in browser
    if args.open:
        print_info("Opening in browser...")
        webbrowser.open(f"file://{output_path.absolute()}")
    
    print_header("Visualization Complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())