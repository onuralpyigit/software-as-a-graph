#!/usr/bin/env python3
"""
CLI script to export graph data from Neo4j to JSON file.

Example usage:
    python export_graph.py --output output/exported_graph.json
    python export_graph.py --output graph.json --uri bolt://localhost:7687
    python export_graph.py --output graph.json --raw  # Export raw format with weights
"""
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saag import Client
from bin._shared import add_neo4j_args, add_common_args, setup_logging
from bin.common.console import ConsoleDisplay


def main() -> None:
    """Main entry point for graph export CLI."""
    parser = argparse.ArgumentParser(
        description="Export Graph Data from Neo4j to JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()

    if not args.output:
        parser.error("--output / -o is required for export_graph.py")

    setup_logging(args)
    console = ConsoleDisplay()
    
    console.print_header("Graph Export")
    console.print_step(f"Connecting to Neo4j at {args.uri}...")
    
    try:
        client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
        console.print_step("Exporting graph data from database...")
        data = client.repo.export_json()

        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        console.print_success(f"Success! Saved to {args.output}")
        console.display_graph_data_summary(data, title="Graph Export Summary")

    except Exception as e:
        console.print_error(f"Export failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
