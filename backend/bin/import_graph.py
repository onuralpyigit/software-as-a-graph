#!/usr/bin/env python3
"""
CLI script to import graph data into Neo4j.

Example usage:
    python import_graph.py --input output/graph.json --clear
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from src.core import create_repository


def print_import_stats(stats: Dict[str, int]) -> None:
    """Print formatted import statistics."""
    print("\nImport & Derivation Complete!")
    print("-" * 30)
    print("Components Imported:")
    print(f"  Nodes:       {stats.get('node_count', 0)}")
    print(f"  Brokers:     {stats.get('broker_count', 0)}")
    print(f"  Topics:      {stats.get('topic_count', 0)}")
    print(f"  Apps:        {stats.get('application_count', 0)}")
    print(f"  Libraries:   {stats.get('library_count', 0)}")
    print("-" * 30)
    print("Relationships Imported:")
    print(f"  RUNS_ON:     {stats.get('runs_on_count', 0)}")
    print(f"  ROUTES:      {stats.get('routes_count', 0)}")
    print(f"  PUBLISHES_TO: {stats.get('publishes_to_count', 0)}")
    print(f"  SUBSCRIBES_TO: {stats.get('subscribes_to_count', 0)}")
    print(f"  CONNECTS_TO: {stats.get('connects_to_count', 0)}")
    print(f"  USES:        {stats.get('uses_count', 0)}")
    print("-" * 30)
    print("Dependencies Derived:")
    print(f"  App->App:    {stats.get('app_to_app_count', 0)}")
    print(f"  App->Broker: {stats.get('app_to_broker_count', 0)}")
    print(f"  Node->Node:  {stats.get('node_to_node_count', 0)}")
    print(f"  Node->Broker:{stats.get('node_to_broker_count', 0)}")
    print("-" * 30)
    print("Weight Calculation:")
    print("  - Intrinsic weights (QoS/Size) applied to Topics/Edges.")
    print("  - Aggregate weights applied to Apps/Brokers/Nodes.")
    print("  - Final criticality scores (Intrinsic + Centrality) updated.")
    print("-" * 30)


def main() -> None:
    """Main entry point for graph import CLI."""
    parser = argparse.ArgumentParser(
        description="Import Graph to Neo4j & Derive Dependencies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file",
    )
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j Bolt URI",
    )
    parser.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j Username",
    )
    parser.add_argument(
        "--password",
        default="password",
        help="Neo4j Password",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing DB before import",
    )
    parser.add_argument(
        "--db",
        default="neo4j",
        help="Database name",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {args.input}...")
    try:
        with open(input_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to Neo4j at {args.uri}...")
    
    # Initialize Repository directly
    repo = create_repository(
        uri=args.uri,
        user=args.user,
        password=args.password
    )
    
    try:
        repo.save_graph(data, clear=args.clear)
        stats = repo.get_statistics()
        print_import_stats(stats)
            
    except Exception as e:
        print(f"Import failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        repo.close()


if __name__ == "__main__":
    main()
