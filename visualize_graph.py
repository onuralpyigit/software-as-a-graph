#!/usr/bin/env python3
"""
Graph Visualization CLI (Refactored)

Generates multi-layer analysis dashboards using the VisualizationService.
"""

import argparse
import logging
import os
import sys
import webbrowser
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.infrastructure import Container
from src.models.visualization.layer_data import LAYER_DEFINITIONS


def display_available_layers(display) -> None:
    """Display available layers."""
    display.print_subheader("Available Layers")
    for layer, definition in LAYER_DEFINITIONS.items():
        print(f"  {display.colored(f'{layer:<10}', display.Colors.CYAN)} {definition['icon']} {definition['name']}")
        print(f"  {' ' * 10} {display.colored(definition['description'], display.Colors.GRAY)}")
    print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate multi-layer graph analysis dashboards.")
    
    layer_group = parser.add_argument_group("Layer Selection")
    layer_mutex = layer_group.add_mutually_exclusive_group(required=True)
    layer_mutex.add_argument("--layers", "-l", help="Comma-separated layers (e.g., app,infra,system)")
    layer_mutex.add_argument("--layer", choices=list(LAYER_DEFINITIONS.keys()), help="Single layer")
    layer_mutex.add_argument("--all", "-a", action="store_true", help="All layers")
    layer_mutex.add_argument("--list-layers", action="store_true", help="List layers and exit")
    
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    neo4j_group.add_argument("--user", "-u", default="neo4j", help="Neo4j username")
    neo4j_group.add_argument("--password", "-p", default="password", help="Neo4j password")
    
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output", "-o", default="dashboard.html", help="Output file")
    output_group.add_argument("--no-network", action="store_true", help="Exclude network graph")
    output_group.add_argument("--no-validation", action="store_true", help="Exclude validation")
    output_group.add_argument("--open", "-O", action="store_true", help="Open in browser")
    
    args = parser.parse_args()
    
    container = Container(uri=args.uri, user=args.user, password=args.password)
    display = container.display_service()
    
    if args.list_layers:
        display_available_layers(display)
        return 0
    
    # Determine layers
    if args.all:
        layers = list(LAYER_DEFINITIONS.keys())
    elif args.layer:
        layers = [args.layer]
    else:
        layers = [l.strip() for l in args.layers.split(",")]
    
    valid_layers = [l for l in layers if l in LAYER_DEFINITIONS]
    if not valid_layers:
        print(display.colored("✗ No valid layers specified", display.Colors.RED))
        return 1
    
    display.print_header("SOFTWARE-AS-A-GRAPH VISUALIZATION", "═")
    print(f"\n  {display.colored('Neo4j:', display.Colors.CYAN)}  {args.uri}")
    print(f"  {display.colored('Output:', display.Colors.CYAN)} {args.output}")
    print(f"  {display.colored('Layers:', display.Colors.CYAN)} {', '.join(valid_layers)}")
    
    try:
        viz_service = container.visualization_service()
        output_path = viz_service.generate_dashboard(
            output_file=args.output,
            layers=valid_layers,
            include_network=not args.no_network,
            include_validation=not args.no_validation
        )
        
        display.print_header("DASHBOARD GENERATED", "-")
        abs_path = os.path.abspath(output_path)
        print(f"\n  {display.colored('File:', display.Colors.GREEN)} {abs_path}")
        
        if args.open:
            webbrowser.open(f"file://{abs_path}")
        
        return 0
        
    except Exception as e:
        print(display.colored(f"\n✗ Error: {e}", display.Colors.RED))
        logging.exception("Visualization failed")
        return 1
    finally:
        container.close()


if __name__ == "__main__":
    sys.exit(main())