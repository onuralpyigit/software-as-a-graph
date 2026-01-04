#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))
from src.core import generate_graph

def main():
    parser = argparse.ArgumentParser(description="Generate Pub-Sub Graph Data")
    parser.add_argument("--scale", default="medium", choices=["tiny", "small", "medium", "large", "xlarge"])
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Generating '{args.scale}' graph (Seed: {args.seed})...")
    
    try:
        data = generate_graph(scale=args.scale, seed=args.seed)
        
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Success! Saved to {args.output}")
        print(f"Stats: {len(data['nodes'])} Nodes, {len(data['applications'])} Apps, {len(data['topics'])} Topics, {len(data['brokers'])} Brokers")
        
    except Exception as e:
        print(f"Error generating graph: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()