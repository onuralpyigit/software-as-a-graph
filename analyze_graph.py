#!/usr/bin/env python3
"""
Graph Analysis CLI (Refactored)

Multi-layer graph analysis for distributed pub-sub systems.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.infrastructure import Container
from src.domain.models.analysis.results import MultiLayerAnalysisResult


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-layer graph analysis for distributed pub-sub systems.")
    
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layer", "-l", choices=["app", "infra", "mw-app", "mw-infra", "system"], default="system", help="Analysis layer (default: system)")
    layer_group.add_argument("--all", "-a", action="store_true", help="Analyze all primary layers")
    
    parser.add_argument("--include-middleware", action="store_true", help="Include middleware layers when using --all")
    parser.add_argument("--output", "-o", metavar="FILE", help="Export results to JSON file")
    parser.add_argument("--json", action="store_true", help="Output results as JSON to stdout")
    
    # Neo4j connection
    parser.add_argument("--uri", "-n", default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--user", "-u", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", "-p", default="password", help="Neo4j password")
    
    # Display options
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    
    container = Container(uri=args.uri, user=args.user, password=args.password)
    display = container.display_service()
    
    try:
        analyzer = container.analysis_service()
        
        # Run analysis
        if args.all:
            results = analyzer.analyze_all_layers(include_middleware=args.include_middleware)
        else:
            from datetime import datetime
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