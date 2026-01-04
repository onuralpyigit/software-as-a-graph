#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))
from src.core import GraphImporter

def main():
    parser = argparse.ArgumentParser(description="Import Graph to Neo4j & Derive Dependencies")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j Username")
    parser.add_argument("--password", default="password", help="Neo4j Password")
    parser.add_argument("--clear", action="store_true", help="Clear existing DB before import")
    parser.add_argument("--db", default="neo4j", help="Database name")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"Reading {args.input}...")
    try:
        with open(args.input) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)

    print(f"Connecting to Neo4j at {args.uri}...")
    try:
        with GraphImporter(uri=args.uri, user=args.user, password=args.password, database=args.db) as importer:
            stats = importer.import_graph(data, clear=args.clear)
            
            print("\nImport & Derivation Complete!")
            print("-" * 30)
            print(f"Entities Imported:")
            print(f"  Nodes:       {stats.get('nodes', 0)}")
            print(f"  Brokers:     {stats.get('brokers', 0)}")
            print(f"  Topics:      {stats.get('topics', 0)}")
            print(f"  Apps:        {stats.get('apps', 0)}")
            print("-" * 30)
            print(f"Dependencies Derived:")
            print(f"  App->App:    {stats.get('deps_app_app', 0)}")
            print(f"  App->Broker: {stats.get('deps_app_broker', 0)}")
            print(f"  Node->Node:  {stats.get('deps_node_node', 0)}")
            print(f"  Node->Broker:{stats.get('deps_node_broker', 0)}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()