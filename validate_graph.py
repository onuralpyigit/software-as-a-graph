#!/usr/bin/env python3
"""
Graph Validation CLI

Validates the graph modeling and analysis approach by comparing
predicted criticality scores against actual failure impact.

Validation Pipeline:
    1. Graph Analysis → Predicted criticality scores Q(v)
    2. Failure Simulation → Actual impact scores I(v)
    3. Statistical Comparison → Validation metrics

Layers:
    app      : Application layer
    infra    : Infrastructure layer
    mw-app   : Middleware-Application layer
    mw-infra : Middleware-Infrastructure layer
    system   : Complete system

Usage:
    python validate_graph.py --layers app,infra,system
    python validate_graph.py --all
    python validate_graph.py --quick predicted.json actual.json

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.validation import (
    ValidationPipeline,
    PipelineResult,
    LayerValidationResult,
    QuickValidator,
    ValidationTargets,
    LAYER_DEFINITIONS,
)
from src.visualization.display import (
    display_pipeline_validation_result as display_pipeline_result,
    display_layer_validation_result as display_layer_result,
    status_icon,
    status_text,
)
from src.visualization.display import Colors, colored, print_header


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate graph modeling and analysis approach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validation Pipeline:
  1. Graph Analysis → Predicted criticality scores Q(v)
  2. Failure Simulation → Actual impact scores I(v)
  3. Statistical Comparison → Validation metrics

Layers:
  app      Application layer (Applications only)
  infra    Infrastructure layer (Nodes only)
  mw-app   Middleware-Application (Applications + Brokers)
  mw-infra Middleware-Infrastructure (Nodes + Brokers)
  system   Complete system (all components)

Validation Targets:
  Spearman ρ ≥ 0.70  (rank correlation)
  F1 Score ≥ 0.80    (classification accuracy)
  Precision ≥ 0.80   (positive predictive value)
  Recall ≥ 0.80      (sensitivity)
  Top-5 Overlap ≥ 0.60 (ranking agreement)

Examples:
  %(prog)s --layer app,infra,system
  %(prog)s --all
  %(prog)s --quick predicted.json actual.json
  %(prog)s --layer system --output results/validation.json
        """
    )
    
    # Action
    action_group = parser.add_argument_group("Action")
    action_mutex = action_group.add_mutually_exclusive_group(required=True)
    action_mutex.add_argument(
        "--layer", "-l",
        help="Comma-separated layers to validate (e.g., app,infra,system)"
    )
    action_mutex.add_argument(
        "--all", "-a",
        action="store_true",
        help="Validate all layers"
    )
    action_mutex.add_argument(
        "--quick", "-q",
        nargs=2,
        metavar=("PREDICTED", "ACTUAL"),
        help="Quick validation from JSON files"
    )
    
    # Neo4j connection
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )
    neo4j_group.add_argument(
        "--user", "-u",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password (default: password)"
    )
    
    # Validation targets
    targets_group = parser.add_argument_group("Validation Targets")
    targets_group.add_argument(
        "--spearman",
        type=float,
        default=0.70,
        help="Target Spearman correlation (default: 0.70)"
    )
    targets_group.add_argument(
        "--f1",
        type=float,
        default=0.80,
        help="Target F1 score (default: 0.80)"
    )
    targets_group.add_argument(
        "--precision",
        type=float,
        default=0.80,
        help="Target precision (default: 0.80)"
    )
    targets_group.add_argument(
        "--recall",
        type=float,
        default=0.80,
        help="Target recall (default: 0.80)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Export results to JSON file"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Create targets
    targets = ValidationTargets(
        spearman=args.spearman,
        f1_score=args.f1,
        precision=args.precision,
        recall=args.recall,
    )
    
    try:
        # Quick validation
        if args.quick:
            predicted_file, actual_file = args.quick
            
            validator = QuickValidator(targets=targets)
            result = validator.validate_from_files(
                predicted_file=predicted_file,
                actual_file=actual_file,
            )
            
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            elif not args.quiet:
                # Display simplified result
                print_header("Quick Validation Result")
                print(f"\n  Files: {predicted_file} vs {actual_file}")
                print(f"  Matched: {result.matched_count} components")
                print(f"\n  Status: {status_text(result.passed)}")
                
                overall = result.overall
                print(f"\n  Spearman:  {overall.correlation.spearman:.4f}  {status_icon(overall.correlation.spearman >= targets.spearman)}")
                print(f"  F1 Score:  {overall.classification.f1_score:.4f}  {status_icon(overall.classification.f1_score >= targets.f1_score)}")
                print(f"  Precision: {overall.classification.precision:.4f}  {status_icon(overall.classification.precision >= targets.precision)}")
                print(f"  Recall:    {overall.classification.recall:.4f}  {status_icon(overall.classification.recall >= targets.recall)}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                if not args.quiet:
                    print(f"\n{colored(f'Results saved to: {args.output}', Colors.GREEN)}")
            
            return 0 if result.passed else 1
        
        # Full pipeline validation
        if args.all:
            layers = list(LAYER_DEFINITIONS.keys())
        else:
            layers = [l.strip() for l in args.layer.split(",")]
        
        # Create container and repository
        from src.infrastructure import Container
        container = Container(uri=args.uri, user=args.user, password=args.password)
        repository = container.graph_repository()
        
        # Create and run pipeline
        pipeline = ValidationPipeline(
            uri=args.uri,
            user=args.user,
            password=args.password,
            targets=targets,
            repository=repository
        )
        
        result = pipeline.run(layers=layers)
        
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        elif not args.quiet:
            display_pipeline_result(result)
        
        if args.output:
            pipeline.export_result(result, args.output)
            if not args.quiet:
                print(f"\n{colored(f'Results saved to: {args.output}', Colors.GREEN)}")
            
            return 0 if result.all_passed else 1


    
    except KeyboardInterrupt:
        print(f"\n{colored('Validation interrupted.', Colors.YELLOW)}")
        return 130
    
    except Exception as e:
        logging.exception("Validation failed")
        print(f"{colored(f'Error: {e}', Colors.RED)}", file=sys.stderr)
        return 1

    finally:
        container.close()


if __name__ == "__main__":
    sys.exit(main())