#!/usr/bin/env python3
"""
Graph Analysis CLI

Multi-layer graph analysis for distributed pub-sub systems.

Layers:
    app     - Application layer (app_to_app dependencies)
    infra   - Infrastructure layer (node_to_node dependencies)
    mw      - Middleware layer (app_to_broker + node_to_broker dependencies)
    system  - Complete system (all layers combined)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from datetime import datetime

from src.infrastructure import Container
from src.models.analysis.layers import AnalysisLayer, list_layers
from src.models.analysis.results import MultiLayerAnalysisResult


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-layer graph analysis for distributed pub-sub systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --layer app           # Analyze application layer only
  %(prog)s --layer infra         # Analyze infrastructure layer only
  %(prog)s --layer mw            # Analyze middleware layer only
  %(prog)s --layer system        # Analyze complete system
  %(prog)s --all                 # Analyze all layers
  %(prog)s --list-layers         # Show available layers
"""
    )
    
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--layer", "-l",
        choices=["app", "infra", "mw", "system"],
        default="system",
        help="Analysis layer (default: system)"
    )
    layer_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Analyze all layers"
    )
    
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="List available layers with descriptions"
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    
    # Neo4j connection
    parser.add_argument("--uri", "-n", default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--user", "-u", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", "-p", default="password", help="Neo4j password")
    
    # Display options
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Handle --list-layers
    if args.list_layers:
        print(list_layers())
        return 0
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    
    container = Container(uri=args.uri, user=args.user, password=args.password)
    display = container.display_service()
    
    try:
        analyzer = container.analysis_service()
        
        # Run analysis
        if args.all:
            results = analyzer.analyze_all_layers()
        else:
            layer_result = analyzer.analyze_layer(args.layer)
            results = MultiLayerAnalysisResult(
                timestamp=datetime.now().isoformat(),
                layers={layer_result.layer: layer_result},
                cross_layer_insights=[],
            )
        
        # Export if requested
        if args.output:
            analyzer.export_results(results, args.output)
            if not args.quiet:
                print(f"\n{display.colored(f'Results exported to: {args.output}', display.Colors.GREEN)}")
        
        # Output
        if args.json:
            print(json.dumps(results.to_dict(), indent=2, default=str))
        elif not args.quiet:
            display.display_multi_layer_analysis_result(results)
        
        return 0
    
    except Exception as e:
        print(display.colored(f"Error: {e}", display.Colors.RED), file=sys.stderr)
        if args.verbose:
            logging.exception("Analysis failed")
        return 1
    finally:
        container.close()


if __name__ == "__main__":
    sys.exit(main())
