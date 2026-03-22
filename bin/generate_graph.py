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

# Add project root and backend to path if running from bin/
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
backend_path = project_root / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from bin.common.dispatcher import dispatch_generate
from bin.common.arguments import add_common_arguments


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
    
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain for realistic naming (e.g. e-commerce, robotics)",
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario mapping for QoS generation (e.g. sensor_telemetry, events)",
    )
    
    add_common_arguments(parser)
    
    args = parser.parse_args()
    
    try:
        if args.config:
            print(f"Loading configuration from {args.config}...")
        else:
            scale = args.scale or "medium"
            print(f"Generating {scale} graph with seed {args.seed}...")
            if args.domain:
                print(f"Using domain dataset: {args.domain}" + (f" (scenario: {args.scenario})" if args.scenario else ""))
        
        graph_data = dispatch_generate(args)
            
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