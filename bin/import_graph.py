#!/usr/bin/env python3
"""
CLI script to import graph data into Neo4j.
"""

import sys
from pathlib import Path

# Provide resolving so `saag` and `bin._shared` can be accessed natively
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from saag import Client
from bin._shared import add_neo4j_args, setup_logging

def main():
    parser = argparse.ArgumentParser(
        description="Import Graph to Neo4j & Derive Dependencies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--clear", action="store_true", help="Clear existing DB before import")
    
    add_neo4j_args(parser)
    args = parser.parse_args()
    setup_logging(args)

    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to Neo4j at {args.uri}...")
    
    stats = client.import_topology(filepath=args.input, clear=args.clear)
    
    print("\nImport & Derivation Complete!")
    print("-" * 30)
    print("Components Imported:")
    print(f"  Nodes:       {stats.get('nodes_imported', 0)}")
    print(f"  Edges:       {stats.get('edges_imported', 0)}")
    print(f"  Duration:    {stats.get('duration_ms', 0):.2f} ms")
    print("-" * 30)

if __name__ == "__main__":
    main()
