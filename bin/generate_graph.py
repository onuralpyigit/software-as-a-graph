#!/usr/bin/env python3
"""
CLI script to generate pub-sub graph data.
Adapts CLI arguments to the Application Service.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add backend to path if running from bin/
project_root = Path(__file__).resolve().parent.parent
backend_path = project_root / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from src.generation import GenerationService, load_config, generate_graph


def main() -> None:
    """Main entry point for graph generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate Pub-Sub Graph Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
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
        help="Path to graph configuration YAML file",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output JSON file",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    try:
        graph_data = {}
        
        if args.config:
            print(f"Loading configuration from {args.config}...")
            config = load_config(args.config)
            # When using config file, seed and scale are usually inside it, 
            # but we can allow override or just pass the config object.
            # The Service handles the config object priority.
            service = GenerationService(config=config)
            graph_data = service.generate()
        else:
            scale = args.scale or "medium"
            print(f"Generating {scale} graph with seed {args.seed}...")
            graph_data = generate_graph(scale=scale, seed=args.seed)
            
        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.output, "w") as f:
            json.dump(graph_data, f, indent=2)
            
        print(f"Graph generated successfully: {args.output}")
        component_counts = {
            "nodes": len(graph_data.get("nodes", [])),
            "brokers": len(graph_data.get("brokers", [])),
            "topics": len(graph_data.get("topics", [])),
            "applications": len(graph_data.get("applications", [])),
            "libraries": len(graph_data.get("libraries", [])),
        }
        print(f"Components: {component_counts}")
        
    except Exception as e:
        print(f"Error generating graph: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()