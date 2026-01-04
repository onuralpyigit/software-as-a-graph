#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.core import generate_graph

def main():
    parser = argparse.ArgumentParser(description="Generate Pub-Sub Graph")
    parser.add_argument("--scale", default="medium", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating '{args.scale}' graph...")
    data = generate_graph(scale=args.scale, seed=args.seed)
    
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved to {args.output} ({len(data['nodes'])} nodes, {len(data['applications'])} apps)")

if __name__ == "__main__":
    main()