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
from bin.common.console import ConsoleDisplay

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
    console = ConsoleDisplay()

    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    input_path = Path(args.input)
    if not input_path.exists():
        console.print_error(f"Input file '{args.input}' not found.")
        sys.exit(1)

    console.print_header("Graph Import")
    if args.clear:
        console.print_step("Clearing existing database...")
    console.print_step(f"Importing {input_path.name} into Neo4j at {args.uri}...")
    
    try:
        stats = client.import_topology(filepath=args.input, clear=args.clear)
        console.print_success("Import & Derivation Complete!")
        console.display_import_summary(stats)
    except Exception as e:
        console.print_error(f"Import failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
