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
    
    print(f"Connecting to Neo4j at {args.uri}...")
    
    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
 
    try:
        print("Exporting graph data...")
        data = client.repo.export_json()

        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Success! Saved to {args.output}")
        print_export_stats(data)

    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        sys.exit(1)


def print_export_stats(data: Dict[str, Any]) -> None:
    """Print formatted export statistics."""
    print("-" * 30)
    print("Exported Entities:")
    print(f"  Nodes:         {len(data.get('nodes', []))}")
    print(f"  Brokers:       {len(data.get('brokers', []))}")
    print(f"  Applications:  {len(data.get('applications', []))}")
    print(f"  Topics:        {len(data.get('topics', []))}")
    print(f"  Libraries:     {len(data.get('libraries', []))}")
    print("-" * 30)

    print("Exported Relationships:")
    rels = data.get("relationships", {})
    print(f"  Publishes To:  {len(rels.get('publishes_to', []))}")
    print(f"  Subscribes To: {len(rels.get('subscribes_to', []))}")
    print(f"  Routes:        {len(rels.get('routes', []))}")
    print(f"  Runs On:       {len(rels.get('runs_on', []))}")
    print(f"  Connects To:   {len(rels.get('connects_to', []))}")
    print(f"  Uses:          {len(rels.get('uses', []))}")
    print("-" * 30)


if __name__ == "__main__":
    main()
