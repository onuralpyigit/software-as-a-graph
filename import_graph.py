#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.core import GraphImporter

def main():
    parser = argparse.ArgumentParser(description="Import Graph to Neo4j")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--clear", action="store_true", help="Clear DB first")
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    with open(args.input) as f:
        data = json.load(f)

    print(f"Importing to Neo4j at {args.uri}...")
    try:
        with GraphImporter(uri=args.uri, user=args.user, password=args.password) as importer:
            stats = importer.import_graph(data, clear=args.clear)
            
            print("\nImport Success!")
            print(f"  Nodes:       {stats['nodes']}")
            print(f"  Brokers:     {stats['brokers']}")
            print(f"  Topics:      {stats['topics']}")
            print(f"  Apps:        {stats['apps']}")
            print(f"  Derived Dependencies:")
            print(f"    App->App:       {stats['deps_app_app']}")
            print(f"    Node->Node:     {stats['deps_node_node']}")
            print(f"    App->Broker:    {stats['deps_app_broker']}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()