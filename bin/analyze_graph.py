#!/usr/bin/env python3
"""
Graph Analysis CLI

Multi-layer graph analysis for distributed pub-sub systems.
Applies graph topology analysis to predict critical components using
DEPENDS_ON relationships derived from the system model.

Layers:
    app     - Application layer (app_to_app dependencies)
    infra   - Infrastructure layer (node_to_node dependencies)
    mw      - Middleware layer (app_to_broker + node_to_broker dependencies)
    system  - Complete system (all layers combined)

Pipeline per layer:
    1. Structural Analysis  → Centrality metrics (PageRank, Betweenness, …)
    2. Quality Analysis     → RMAV scores with Box-Plot classification
    3. Problem Detection    → Architectural smells and risks

Usage:
    python bin/analyze_graph.py --layer app
    python bin/analyze_graph.py --all --output output/analysis.json
    python bin/analyze_graph.py --list-layers
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import argparse
import json
import logging
from datetime import datetime

from src.infrastructure import create_repository
from src.core import AnalysisLayer
from src.analysis import AnalysisService, MultiLayerAnalysisResult
from common.console import ConsoleDisplay
from common.arguments import (
    add_neo4j_arguments, 
    add_common_arguments, 
    add_layer_argument
)


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with clear grouping."""
    parser = argparse.ArgumentParser(
        prog="analyze_graph",
        description="Multi-layer graph analysis for distributed pub-sub systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --layer app               Analyze application layer only
  %(prog)s --layer infra             Analyze infrastructure layer only
  %(prog)s --layer mw                Analyze middleware layer only
  %(prog)s --layer system            Analyze complete system
  %(prog)s --all                     Analyze all layers
  %(prog)s --all -o results.json     Analyze all and export to JSON
  %(prog)s --list-layers             Show available layers
""",
    )

    # --- Layer selection (mutually exclusive) ---
    layer_group = parser.add_mutually_exclusive_group()
    add_layer_argument(layer_group, default="system")
    layer_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Analyze all four layers",
    )
    layer_group.add_argument(
        "--list-layers",
        action="store_true",
        help="List available layers and exit",
    )

    # --- Neo4j connection ---
    add_neo4j_arguments(parser)

    # --- Analysis options ---
    analysis = parser.add_argument_group("Analysis options")
    analysis.add_argument(
        "--use-ahp",
        action="store_true",
        help="Use AHP-derived weights instead of default fixed weights",
    )
    analysis.add_argument(
        "--norm",
        choices=["max", "robust"],
        default="robust",
        help="Normalization method for quality scores (default: robust)",
    )
    analysis.add_argument(
        "--winsorize",
        action="store_true",
        default=True,
        help="Enable outlier mitigation (winsorization) (default: True)",
    )
    analysis.add_argument(
        "--no-winsorize",
        dest="winsorize",
        action="store_false",
        help="Disable outlier mitigation",
    )
    analysis.add_argument(
        "--winsorize-limit",
        type=float,
        default=0.05,
        help="Percentile limit for winsorization (default: 0.05)",
    )
    analysis.add_argument(
        "--sensitivity", "-s",
        action="store_true",
        help="Run ranking sensitivity analysis via weight perturbations",
    )
    analysis.add_argument(
        "--perturbations",
        type=int,
        default=200,
        help="Number of perturbations for sensitivity analysis (default: 200)",
    )
    analysis.add_argument(
        "--noise",
        type=float,
        default=0.05,
        help="Standard deviation of noise for perturbations (default: 0.05)",
    )
    analysis.add_argument(
        "--gnn-model",
        metavar="PATH",
        help="Path to pre-trained GNN model/checkpoint",
    )
    analysis.add_argument(
        "--ensemble",
        action="store_true",
        help="Run ensemble prediction blending GNN and RMAV",
    )

    # --- Output ---
    output = parser.add_argument_group("Output")
    output.add_argument("--output", "-o", metavar="FILE", help="Export results to JSON file")
    output.add_argument("--json", action="store_true", help="Print results as JSON to stdout")
    add_common_arguments(parser)

    return parser


# ---------------------------------------------------------------------------
# Analysis Logic
# ---------------------------------------------------------------------------

from common.dispatcher import dispatch_analyze


# ---------------------------------------------------------------------------
# Analysis Logic
# ---------------------------------------------------------------------------

# Logic moved to src.cli.dispatcher.dispatch_analyze


# ---------------------------------------------------------------------------
# Output Helpers
# ---------------------------------------------------------------------------

def export_json(results: MultiLayerAnalysisResult, path: str) -> None:
    """Write results to a JSON file, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # --list-layers: informational, then exit
    if args.list_layers:
        print(", ".join([l.value for l in AnalysisLayer]))
        return 0

    # Logging setup
    log_level = (
        logging.DEBUG if args.verbose
        else logging.WARNING if args.quiet
        else logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Initialize display
    display = ConsoleDisplay()

    try:
        # Initialize repository
        repo = create_repository(uri=args.uri, user=args.user, password=args.password)
        
        try:
            # Run the analysis pipeline via dispatcher
            results = dispatch_analyze(repo, args)

            # Export to file if requested
            if args.output:
                export_json(results, args.output)
                if not args.quiet:
                    print(f"\n✓ Results exported to: {args.output}")

            # Display to stdout
            if args.json:
                print(json.dumps(results.to_dict(), indent=2, default=str))
            elif not args.quiet:
                display.display_multi_layer_analysis_result(results)

            return 0
        finally:
            repo.close()

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            logging.exception("Analysis failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())