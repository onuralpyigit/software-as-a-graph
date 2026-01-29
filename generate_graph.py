#!/usr/bin/env python3
"""
CLI script to generate pub-sub graph data.

Example usage:
    python generate_graph.py --scale medium --output output/graph.json --seed 42
    python generate_graph.py --config input/graph_config.yaml --output output/graph.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

from src.services.graph_generator import GraphGenerator, GraphConfig, load_config


def main() -> None:
    """Main entry point for graph generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate Pub-Sub Graph Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Create mutually exclusive group for scale vs config
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--scale",
        default=None,
        choices=["tiny", "small", "medium", "large", "xlarge"],
        help="Scale of the graph to generate (preset)",
    )
    config_group.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
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
        help="Random seed for reproducibility (ignored if --config is used)",
    )
    args = parser.parse_args()

    # Determine configuration source
    if args.config:
        if not args.config.exists():
            print(f"Error: Config file '{args.config}' not found.", file=sys.stderr)
            sys.exit(1)
        try:
            config = load_config(args.config)
            print(f"Loading configuration from '{args.config}'...")
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
        generator = GraphGenerator(config=config)
        config_desc = f"config={args.config}"
    else:
        # Use scale preset (default to medium if neither provided)
        scale = args.scale or "medium"
        generator = GraphGenerator(scale=scale, seed=args.seed)
        config_desc = f"scale={scale}, seed={args.seed}"
    
    print(f"Generating graph ({config_desc})...")

    try:
        data = generator.generate()

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