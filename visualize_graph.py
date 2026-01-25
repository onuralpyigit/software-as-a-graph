#!/usr/bin/env python3
"""
Graph Visualization CLI

Generates multi-layer analysis dashboards for the Software-as-a-Graph system.
Visualizes graph statistics, analysis results, simulation outcomes,
and validation metrics.

Layers:
    app      : Application layer
    infra    : Infrastructure layer
    mw-app   : Middleware-Application layer
    mw-infra : Middleware-Infrastructure layer
    system   : Complete system

Usage:
    python visualize_graph.py --layers app,infra,system
    python visualize_graph.py --all
    python visualize_graph.py --layer system --output report.html

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.visualization import (
    GraphVisualizer,
    DashboardGenerator,
    ChartGenerator,
    LAYER_DEFINITIONS,
)
from src.visualization.display import Colors, colored, print_header


# =============================================================================
# CLI Output Utilities
# =============================================================================

def print_step(step: int, total: int, message: str) -> None:
    """Print a step indicator."""
    print(f"\n  {colored(f'[{step}/{total}]', Colors.BLUE, bold=True)} {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"  {colored('✓', Colors.GREEN, bold=True)} {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"  {colored('⚠', Colors.YELLOW)} {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"  {colored('✗', Colors.RED, bold=True)} {message}")


# =============================================================================
# Layer Display
# =============================================================================

def display_available_layers() -> None:
    """Display available layers."""
    print(f"\n{colored('Available Layers:', Colors.WHITE, bold=True)}")
    print(f"  {'-' * 50}")
    
    for layer, definition in LAYER_DEFINITIONS.items():
        print(f"  {colored(f'{layer:<10}', Colors.CYAN)} {definition['icon']} {definition['name']}")
        print(f"  {' ' * 10} {colored(definition['description'], Colors.DIM)}")
    
    print()


def display_layer_summary(layers: List[str]) -> None:
    """Display summary of layers to be processed."""
    print(f"\n{colored('Layers to Process:', Colors.WHITE, bold=True)}")
    
    for layer in layers:
        if layer in LAYER_DEFINITIONS:
            definition = LAYER_DEFINITIONS[layer]
            print(f"  {definition['icon']} {definition['name']} ({layer})")
    
    print()


# =============================================================================
# Dashboard Generation
# =============================================================================

def generate_dashboard(
    uri: str,
    user: str,
    password: str,
    layers: List[str],
    output_file: str,
    include_network: bool = True,
    include_validation: bool = True,
    verbose: bool = False
) -> str:
    """
    Generate the visualization dashboard.
    
    Args:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        layers: Layers to analyze
        output_file: Output HTML file path
        include_network: Include interactive network visualization
        include_validation: Include validation metrics
        verbose: Verbose output
        
    Returns:
        Path to generated dashboard
    """
    total_steps = len(layers) + 2
    
    print_step(1, total_steps, "Initializing visualization pipeline...")
    
    with GraphVisualizer(uri=uri, user=user, password=password) as viz:
        # Check connections
        if viz.analyzer is None:
            print_warning("Analysis module not available")
        else:
            print_success("Analysis module connected")
        
        if viz.simulator is None:
            print_warning("Simulation module not available")
        else:
            print_success("Simulation module connected")
        
        if include_validation and viz.validator is None:
            print_warning("Validation module not available")
            include_validation = False
        elif include_validation:
            print_success("Validation module connected")
        
        # Process layers
        for i, layer in enumerate(layers, start=2):
            definition = LAYER_DEFINITIONS.get(layer, {"name": layer, "icon": "📊"})
            print_step(i, total_steps, f"Processing {definition['icon']} {definition['name']}...")
            
            if verbose:
                print(f"      Analyzing graph structure...")
                print(f"      Running failure simulation...")
                if include_validation:
                    print(f"      Computing validation metrics...")
        
        # Generate dashboard
        print_step(total_steps, total_steps, "Generating HTML dashboard...")
        
        output_path = viz.generate_dashboard(
            output_file=output_file,
            layers=layers,
            include_network=include_network,
            include_validation=include_validation
        )
        
        print_success(f"Dashboard generated: {output_path}")
        
        return output_path


def generate_quick_dashboard(
    output_file: str,
    title: str = "Quick Analysis Dashboard"
) -> str:
    """
    Generate a quick dashboard without Neo4j connection.
    Uses sample data for demonstration.
    """
    dash = DashboardGenerator(title)
    charts = ChartGenerator()
    
    # Overview section
    dash.start_section("📊 Overview", "overview")
    
    dash.add_kpis({
        "Total Nodes": 48,
        "Total Edges": 127,
        "Critical Components": 5,
        "SPOFs Detected": 3,
        "Problems Found": 2,
    }, {"Critical Components": "danger", "SPOFs Detected": "warning"})
    
    # Sample charts
    crit_chart = charts.pie_chart(
        {"CRITICAL": 5, "HIGH": 8, "MEDIUM": 15, "LOW": 20},
        "Criticality Distribution",
        {
            "CRITICAL": "#e74c3c",
            "HIGH": "#e67e22",
            "MEDIUM": "#f1c40f",
            "LOW": "#2ecc71",
        }
    )
    
    type_chart = charts.pie_chart(
        {"Application": 25, "Broker": 3, "Node": 8, "Topic": 12},
        "Component Types"
    )
    
    dash.add_charts([crit_chart, type_chart])
    
    dash.end_section()
    
    # Sample layer section
    dash.start_section("🌐 System Layer", "system")
    
    dash.add_kpis({
        "Nodes": 48,
        "Edges": 127,
        "Density": "0.056",
        "Critical": 5,
        "SPOFs": 3,
    })
    
    # Sample table
    headers = ["Component", "Type", "Score", "Level"]
    rows = [
        ["sensor_fusion", "Application", "0.892", "CRITICAL"],
        ["main_broker", "Broker", "0.856", "CRITICAL"],
        ["planning_node", "Application", "0.789", "HIGH"],
        ["gateway_node", "Node", "0.734", "HIGH"],
        ["control_app", "Application", "0.678", "MEDIUM"],
    ]
    dash.add_table(headers, rows, "Top Components by Quality Score")
    
    # Validation metrics
    dash.add_metrics_box(
        {
            "Spearman ρ": 0.876,
            "F1 Score": 0.923,
            "Precision": 0.912,
            "Recall": 0.857,
            "Status": "PASSED",
        },
        "Validation Metrics",
        {
            "Spearman ρ": True,
            "F1 Score": True,
            "Precision": True,
            "Recall": True,
        }
    )
    
    # Sample network
    sample_nodes = [
        {"id": "A1", "label": "sensor_fusion", "group": "CRITICAL", "value": 35},
        {"id": "B1", "label": "main_broker", "group": "CRITICAL", "value": 32},
        {"id": "A2", "label": "planning", "group": "HIGH", "value": 28},
        {"id": "A3", "label": "control", "group": "MEDIUM", "value": 22},
        {"id": "N1", "label": "gateway", "group": "HIGH", "value": 25},
    ]
    sample_edges = [
        {"from": "A1", "to": "B1"},
        {"from": "A2", "to": "B1"},
        {"from": "B1", "to": "A3"},
        {"from": "A3", "to": "N1"},
    ]
    
    dash.add_network_graph("network-sample", sample_nodes, sample_edges, "Network Topology")
    
    dash.end_section()
    
    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(dash.generate())
    
    return str(output_path)


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multi-layer graph analysis dashboards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Layers:
  app      Application layer (Applications only)
  infra    Infrastructure layer (Nodes only)
  mw-app   Middleware-Application (Applications + Brokers)
  mw-infra Middleware-Infrastructure (Nodes + Brokers)
  system   Complete system (all components)

Examples:
  %(prog)s --layers app,infra,system
  %(prog)s --all --output report.html
  %(prog)s --layer system --no-network
  %(prog)s --demo  # Generate demo dashboard without Neo4j
        """
    )
    
    # Layer selection
    layer_group = parser.add_argument_group("Layer Selection")
    layer_mutex = layer_group.add_mutually_exclusive_group(required=True)
    layer_mutex.add_argument(
        "--layers", "-l",
        help="Comma-separated layers to visualize (e.g., app,infra,system)"
    )
    layer_mutex.add_argument(
        "--layer",
        choices=list(LAYER_DEFINITIONS.keys()),
        help="Single layer to visualize"
    )
    layer_mutex.add_argument(
        "--all", "-a",
        action="store_true",
        help="Visualize all layers"
    )
    layer_mutex.add_argument(
        "--demo",
        action="store_true",
        help="Generate demo dashboard without Neo4j connection"
    )
    layer_mutex.add_argument(
        "--list-layers",
        action="store_true",
        help="List available layers and exit"
    )
    
    # Neo4j connection
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )
    neo4j_group.add_argument(
        "--user", "-u",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password (default: password)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        default="dashboard.html",
        help="Output HTML file (default: dashboard.html)"
    )
    output_group.add_argument(
        "--no-network",
        action="store_true",
        help="Exclude interactive network visualization"
    )
    output_group.add_argument(
        "--no-validation",
        action="store_true",
        help="Exclude validation metrics"
    )
    output_group.add_argument(
        "--open", "-O",
        action="store_true",
        help="Open dashboard in browser after generation"
    )
    
    # General options
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    general_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # List layers
    if args.list_layers:
        display_available_layers()
        return 0
    
    # Demo mode
    if args.demo:
        if not args.quiet:
            print_header("SOFTWARE-AS-A-GRAPH VISUALIZATION", "═")
            print(f"\n  {colored('Mode:', Colors.CYAN)} Demo (sample data)")
            print(f"  {colored('Output:', Colors.CYAN)} {args.output}")
        
        try:
            output_path = generate_quick_dashboard(args.output)
            
            if not args.quiet:
                print_success(f"Demo dashboard generated: {output_path}")
            
            if args.open:
                abs_path = os.path.abspath(output_path)
                webbrowser.open(f"file://{abs_path}")
            
            return 0
        
        except Exception as e:
            print_error(f"Failed to generate demo dashboard: {e}")
            return 1
    
    # Determine layers
    if args.all:
        layers = list(LAYER_DEFINITIONS.keys())
    elif args.layer:
        layers = [args.layer]
    else:
        layers = [l.strip() for l in args.layers.split(",")]
    
    # Validate layers
    valid_layers = [l for l in layers if l in LAYER_DEFINITIONS]
    invalid_layers = [l for l in layers if l not in LAYER_DEFINITIONS]
    
    if invalid_layers:
        print_warning(f"Invalid layers ignored: {', '.join(invalid_layers)}")
    
    if not valid_layers:
        print_error("No valid layers specified")
        display_available_layers()
        return 1
    
    # Print header
    if not args.quiet:
        print_header("SOFTWARE-AS-A-GRAPH VISUALIZATION", "═")
        print(f"\n  {colored('Neo4j:', Colors.CYAN)} {args.uri}")
        print(f"  {colored('Output:', Colors.CYAN)} {args.output}")
        display_layer_summary(valid_layers)
    
    try:
        output_path = generate_dashboard(
            uri=args.uri,
            user=args.user,
            password=args.password,
            layers=valid_layers,
            output_file=args.output,
            include_network=not args.no_network,
            include_validation=not args.no_validation,
            verbose=args.verbose
        )
        
        if not args.quiet:
            print_header("DASHBOARD GENERATED", "-")
            abs_path = os.path.abspath(output_path)
            print(f"\n  {colored('File:', Colors.GREEN)} {abs_path}")
            print(f"  {colored('Size:', Colors.GREEN)} {os.path.getsize(output_path) / 1024:.1f} KB")
            
            if args.open:
                print(f"\n  Opening in browser...")
        
        if args.open:
            abs_path = os.path.abspath(output_path)
            webbrowser.open(f"file://{abs_path}")
        
        return 0
    
    except KeyboardInterrupt:
        print(f"\n{colored('Visualization interrupted.', Colors.YELLOW)}")
        return 130
    
    except Exception as e:
        logging.exception("Visualization failed")
        print_error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())