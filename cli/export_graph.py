#!/usr/bin/env python3
"""
CLI script to export graph data from Neo4j to JSON file.

Example usage:
    python cli/export_graph.py --output output/exported_graph.json
    python cli/export_graph.py --output graph.json --uri bolt://localhost:7687
    python cli/export_graph.py --output graph.json --format analysis  # Export flat analysis format
    python cli/export_graph.py --output graph.json --format analysis --include-structural # Include raw edges
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path to support direct execution (python cli/export_graph.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, Any
from saag import Client
from cli.common.arguments import add_neo4j_arguments, add_common_arguments, setup_logging
from cli.common.console import ConsoleDisplay


def main() -> None:
    """Main entry point for graph export CLI."""
    parser = argparse.ArgumentParser(
        description="Export Graph Data from Neo4j to JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_neo4j_arguments(parser)
    add_common_arguments(parser)
    
    parser.add_argument(
        "--format",
        choices=["persistence", "analysis"],
        default="persistence",
        help="Export format: 'persistence' (nested JSON) or 'analysis' (flat components/edges lists)"
    )
    parser.add_argument(
        "--include-structural",
        action="store_true",
        help="Include structural edges (RUNS_ON, ROUTES, etc.) in 'analysis' format"
    )
    
    args = parser.parse_args()

    if not args.output:
        parser.error("--output / -o is required for export_graph.py")

    setup_logging(args)
    console = ConsoleDisplay()
    
    console.print_header("Graph Export")
    console.print_step(f"Connecting to Neo4j at {args.uri}...")
    
    try:
        client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
        
        if args.format == "analysis":
            layer = args.layer or "system"
            console.print_step(f"Exporting graph in analysis format (layer={layer}, structural={args.include_structural})...")
            from saag.infrastructure.neo4j_repo import LAYER_DEFINITIONS, _resolve_layer
            canonical = _resolve_layer(layer)
            defn = LAYER_DEFINITIONS[canonical]
            graph_data = client.get_graph_data(
                component_types=defn["component_types"],
                dependency_types=defn["dependency_types"],
                include_raw=args.include_structural,
            )
            data = graph_data.to_dict()
        else:
            console.print_step("Exporting graph in nested persistence format...")
            data = client.export_topology()

        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        console.print_success(f"Success! Saved to {args.output}")
        
        if args.format == "analysis":
            console.print_step(f"Nodes: {len(data['components'])}, Edges: {len(data['edges'])}")
        else:
            console.display_graph_data_summary(data, title="Graph Export Summary")

    except Exception as e:
        console.print_error(f"Export failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
