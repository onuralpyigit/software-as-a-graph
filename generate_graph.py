#!/usr/bin/env python3
"""
CLI script to generate pub-sub graph data.

Example usage:
    python generate_graph.py --scale medium --output output/graph.json --seed 42
"""

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

from src.services.graph_generator import generate_graph


def main() -> None:
    """Main entry point for graph generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate Pub-Sub Graph Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scale",
        default="medium",
        choices=["tiny", "small", "medium", "large", "xlarge"],
        help="Scale of the graph to generate",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
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
        print(
            f"Stats: "
            f"{len(data['nodes'])} Nodes, "
            f"{len(data['applications'])} Apps, "
            f"{len(data.get('libraries', []))} Libs, "
            f"{len(data['topics'])} Topics, "
            f"{len(data['brokers'])} Brokers"
        )

    except Exception as e:
        print(f"Error generating graph: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()