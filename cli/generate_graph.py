#!/usr/bin/env python3
"""
CLI script to generate pub-sub graph data.
Adapts CLI arguments to the Application Service.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path to support direct execution (python cli/generate_graph.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional
from cli.common.dispatcher import dispatch_generate
from cli.common.arguments import add_runtime_arguments
from cli.common.console import ConsoleDisplay
from cli.common.batch_generation import run_batch_generation, add_batch_arguments
from cli.common.dataset_validation import run_dataset_validation, add_validation_arguments


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
        choices=["tiny", "small", "medium", "large", "jumbo", "xlarge"],
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
        default=Path("output/graph.json"),
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
        choices=["av", "iot", "finance", "healthcare", "hub-and-spoke", "microservices", "enterprise"],
        help="Scenario mapping for QoS generation",
    )
    
    add_runtime_arguments(parser)
    
    subparsers = parser.add_subparsers(dest="command", help="Optional command mode")
    
    batch_parser = subparsers.add_parser("batch", help="Batch generate datasets for scenarios", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_batch_arguments(batch_parser)
    
    validate_parser = subparsers.add_parser("validate", help="Topology-class validation for scenarios", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_validation_arguments(validate_parser)
    
    # Handle positional output override (hack for user convenience: python cli/generate_graph.py dataset.json)
    output_override = None
    if len(sys.argv) > 1:
        last_arg = sys.argv[-1]
        # If the last arg is not a flag, not a subcommand, and not a help flag
        if not last_arg.startswith('-') and last_arg not in ['batch', 'validate'] and last_arg != 'generate_graph.py':
            output_override = sys.argv.pop()
    
    args = parser.parse_args()
    
    if output_override:
        args.output = Path(output_override)
    
    if getattr(args, "command", None) == "batch":
        sys.exit(run_batch_generation(args))
    elif getattr(args, "command", None) == "validate":
        sys.exit(run_dataset_validation(args))

    console = ConsoleDisplay()
    
    try:
        console.print_header("Graph Generation")
        if args.config:
            console.print_step(f"Loading configuration from {args.config}...")
        else:
            scale = args.scale or "medium"
            console.print_step(f"Generating {scale} graph with seed {args.seed}...")
            if args.domain:
                console.print_step(f"Using domain dataset: {args.domain}" + (f" (scenario: {args.scenario})" if args.scenario else ""))
        
        graph_data = dispatch_generate(args)
            
        console.print_success(f"Graph generated successfully: {args.output}")
        console.display_graph_data_summary(graph_data, title="Graph Generation Summary")
        
    except Exception as e:
        console.print_error(f"Error generating graph: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()