#!/usr/bin/env python3
"""
Graph Analysis CLI

Multi-layer graph analysis for distributed pub-sub systems.
Identifies critical components, detects architectural problems,
and assesses reliability, maintainability, and availability.

Layers:
    app      : Application layer (app_to_app dependencies)
    infra    : Infrastructure layer (node_to_node dependencies)
    mw-app   : Middleware-Application (app_to_broker dependencies)
    mw-infra : Middleware-Infrastructure (node_to_broker dependencies)
    system   : Complete system (all layers)

Usage:
    python analyze_graph.py --layer app
    python analyze_graph.py --all
    python analyze_graph.py --layer system --output results.json

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis import (
    GraphAnalyzer,
    AnalysisLayer,
    MultiLayerAnalysisResult,
)
from src.analysis.display import (
    Colors,
    colored,
    display_layer_result,
    display_multi_layer_result,
    display_final_summary,
)


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-layer graph analysis for distributed pub-sub systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Layers:
  app        Application layer (app_to_app dependencies)
  infra      Infrastructure layer (node_to_node dependencies)
  mw-app     Middleware-Application (app_to_broker dependencies)
  mw-infra   Middleware-Infrastructure (node_to_broker dependencies)
  system     Complete system (all dependencies)

Examples:
  %(prog)s --layer app
  %(prog)s --layer system --output results.json
  %(prog)s --all --include-middleware
  %(prog)s --all --json
        """
    )
    
    # Layer selection
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--layer", "-l",
        choices=["app", "infra", "mw-app", "mw-infra", "system"],
        default="system",
        help="Analysis layer (default: system)"
    )
    layer_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Analyze all primary layers (app, infra, system)"
    )
    
    # Additional options
    parser.add_argument(
        "--include-middleware",
        action="store_true",
        help="Include middleware layers when using --all"
    )
    
    # Output options
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
    parser.add_argument(
        "--uri", "-n",
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--user", "-u",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password (default: password)"
    )
    
    # Analysis parameters
    parser.add_argument(
        "--k-factor", "-k",
        type=float,
        default=1.5,
        help="Box-plot IQR multiplier for classification (default: 1.5)"
    )
    parser.add_argument(
        "--damping", "-d",
        type=float,
        default=0.85,
        help="PageRank damping factor (default: 0.85)"
    )
    parser.add_argument(
        "--ahp",
        action="store_true",
        help="Use Analytic Hierarchy Process (AHP) for calculating quality weights"
    )

    # Display options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (useful with --output)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with debug information"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    try:
        # Create analyzer
        with GraphAnalyzer(
            uri=args.uri,
            user=args.user,
            password=args.password,
            damping_factor=args.damping,
            k_factor=args.k_factor,
            use_ahp=args.ahp
        ) as analyzer:
            
            # Run analysis
            if args.all:
                results = analyzer.analyze_all_layers(include_middleware=args.include_middleware)
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
                    print(f"\n{colored(f'Results exported to: {args.output}', Colors.GREEN)}")
            
            # Output
            if args.json:
                print(json.dumps(results.to_dict(), indent=2, default=str))
            elif not args.quiet:
                if args.all or len(results.layers) > 1:
                    display_multi_layer_result(results)
                else:
                    layer_result = list(results.layers.values())[0]
                    display_layer_result(layer_result)
                
                # Final action items
                display_final_summary(results)
        
        return 0
    
    except KeyboardInterrupt:
        print(f"\n{colored('Analysis interrupted.', Colors.YELLOW)}")
        return 130
    
    except FileNotFoundError as e:
        print(f"{colored(f'Error: {e}', Colors.RED)}", file=sys.stderr)
        return 1
    
    except Exception as e:
        logging.exception("Analysis failed")
        print(f"{colored(f'Error: {e}', Colors.RED)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())