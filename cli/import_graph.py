#!/usr/bin/env python3
"""
CLI script to import graph data into Neo4j.
"""

import argparse
from pathlib import Path
from saag import Client
from cli._shared import add_neo4j_args, add_common_args, setup_logging
from cli.common.console import ConsoleDisplay

def main():
    parser = argparse.ArgumentParser(
        description="Import Graph to Neo4j & Derive Dependencies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--clear", action="store_true", help="Clear existing DB before import")
    parser.add_argument("--dry-run", action="store_true", help="Validate input without importing")
    
    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)
    console = ConsoleDisplay()

    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    input_path = Path(args.input)
    if not input_path.exists():
        console.print_error(f"Input file '{args.input}' not found.")
        sys.exit(1)

    if args.dry_run:
        console.print_header("Graph Import (Dry Run)")
    else:
        console.print_header("Graph Import")

    if args.clear and not args.dry_run:
        console.print_step("Clearing existing database...")
    
    if args.dry_run:
        console.print_step(f"Validating {input_path.name} (simulated import)...")
    else:
        console.print_step(f"Importing {input_path.name} into Neo4j at {args.uri}...")
    
    try:
        stats = client.import_topology(filepath=args.input, clear=args.clear, dry_run=args.dry_run)
        
        if args.dry_run:
            console.print_success("Dry Run Validation Complete!")
        else:
            console.print_success("Import & Derivation Complete!")
            
        console.display_import_summary(stats)
        
        if args.output:
            import json
            with open(args.output, "w") as f:
                json.dump(stats.to_dict(), f, indent=2)
            console.print_step(f"Import stats saved to: {args.output}")


    except Exception as e:

        console.print_error(f"Import failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
