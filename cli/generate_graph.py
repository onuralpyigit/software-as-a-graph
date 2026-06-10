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
        choices=["av", "iot", "finance", "healthcare", "hub-and-spoke", "microservices", "enterprise", "atm"],
        help="Domain for realistic naming (one of av, iot, finance, healthcare, hub-and-spoke, microservices, enterprise, atm)",
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=["av", "iot", "finance", "healthcare", "hub-and-spoke", "microservices", "enterprise", "atm"],
        help="Scenario mapping for QoS generation",
    )
    
    parser.add_argument(
        "--connection-density",
        type=float,
        default=0.3,
        help="Connection density (probability of connects_to edges between nodes)",
    )
    
    add_runtime_arguments(parser)
    
    subparsers = parser.add_subparsers(dest="command", help="Optional command mode")
    
    batch_parser = subparsers.add_parser("batch", help="Batch generate datasets for scenarios", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_batch_arguments(batch_parser)
    
    validate_parser = subparsers.add_parser("validate", help="Topology-class validation for scenarios", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_validation_arguments(validate_parser)

    # Support positional output path so users can run either:
    #   python cli/generate_graph.py --output result.json --scale medium
    #   python cli/generate_graph.py result.json --scale medium
    # We synchronously rewrite sys.argv before argparse sees it, mirroring
    # the original convenience behaviour with corrected flag awareness.
    _value_flags = {
        "--output", "--domain", "--scenario", "--config", "--seed",
        "--connection-density", "--scale", "--input-dir", "--output-dir",
        "--seeds", "--manifest", "--report", "--layer", "--neo4j-uri",
        "--neo4j-user", "--neo4j-password", "--from-results",
        "-o", "-u", "-p",
    }
    _skip_next_flag = False
    _output_path_idx = None
    for _i, _arg in enumerate(sys.argv):
        if _skip_next_flag:
            _skip_next_flag = False
            continue
        if _output_path_idx is not None:
            break
        if _arg.startswith("-"):
            if _arg in _value_flags:
                _skip_next_flag = True
            continue
        if _arg in ("batch", "validate") or _arg == sys.argv[0]:
            continue
        _prev = sys.argv[_i - 1] if _i > 0 else None
        if _prev is not None and _prev in _value_flags:
            continue
        _output_path_idx = _i

    if _output_path_idx is not None:
        sys.argv.insert(_output_path_idx, "--output")

    args = parser.parse_args()
    
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