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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from datetime import datetime

from src.application.container import Container
from src.domain.config.layers import AnalysisLayer, list_layers
from src.domain.models.analysis.results import MultiLayerAnalysisResult


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
    layer_group.add_argument(
        "--layer", "-l",
        choices=[la.value for la in AnalysisLayer],
        default="system",
        help="Analysis layer (default: system)",
    )
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
    neo4j = parser.add_argument_group("Neo4j connection")
    neo4j.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    neo4j.add_argument("--user", "-u", default="neo4j", help="Neo4j username")
    neo4j.add_argument("--password", "-p", default="password", help="Neo4j password")

    # --- Output ---
    output = parser.add_argument_group("Output")
    output.add_argument("--output", "-o", metavar="FILE", help="Export results to JSON file")
    output.add_argument("--json", action="store_true", help="Print results as JSON to stdout")
    output.add_argument("--quiet", "-q", action="store_true", help="Suppress console display")
    output.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    return parser


# ---------------------------------------------------------------------------
# Analysis Logic
# ---------------------------------------------------------------------------

def run_analysis(args: argparse.Namespace) -> MultiLayerAnalysisResult:
    """
    Execute analysis based on parsed CLI arguments.

    Returns a MultiLayerAnalysisResult regardless of whether a single layer
    or all layers were requested, providing a uniform interface for display
    and export.
    """
    container = Container(uri=args.uri, user=args.user, password=args.password)

    try:
        analyzer = container.analysis_service()

        if args.all:
            return analyzer.analyze_all_layers()

        # Single-layer analysis — wrap in MultiLayerAnalysisResult for consistency
        layer_result = analyzer.analyze_layer(args.layer)
        return MultiLayerAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layers={layer_result.layer: layer_result},
            cross_layer_insights=[],
        )
    finally:
        container.close()


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
        print(list_layers())
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

    container = Container(uri=args.uri, user=args.user, password=args.password)
    display = container.display_service()

    try:
        # Run the analysis pipeline
        results = run_analysis(args)

        # Export to file if requested
        if args.output:
            export_json(results, args.output)
            if not args.quiet:
                print(display.colored(
                    f"\n✓ Results exported to: {args.output}",
                    display.Colors.GREEN,
                ))

        # Display to stdout
        if args.json:
            print(json.dumps(results.to_dict(), indent=2, default=str))
        elif not args.quiet:
            display.display_multi_layer_analysis_result(results)

        return 0

    except Exception as exc:
        print(display.colored(f"Error: {exc}", display.Colors.RED), file=sys.stderr)
        if args.verbose:
            logging.exception("Analysis failed")
        return 1

    finally:
        container.close()


if __name__ == "__main__":
    sys.exit(main())