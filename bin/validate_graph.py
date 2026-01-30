#!/usr/bin/env python3
"""
Graph Validation CLI (Refactored)

Validates the graph modeling and analysis approach by comparing
predicted criticality scores against actual failure impact.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.infrastructure import Container
from src.models.validation.metrics import ValidationTargets
from src.models.visualization.layer_data import LAYER_DEFINITIONS


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate graph modeling and analysis approach.")
    
    action_group = parser.add_argument_group("Action")
    action_mutex = action_group.add_mutually_exclusive_group(required=True)
    action_mutex.add_argument("--layer", "-l", help="Comma-separated layers (e.g., app,infra,system)")
    action_mutex.add_argument("--all", "-a", action="store_true", help="Validate all layers")
    action_mutex.add_argument("--quick", "-q", nargs=2, metavar=("PREDICTED", "ACTUAL"), help="Quick validation from JSON files")
    
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    neo4j_group.add_argument("--user", "-u", default="neo4j", help="Neo4j username")
    neo4j_group.add_argument("--password", "-p", default="password", help="Neo4j password")
    
    # Validation targets
    targets_group = parser.add_argument_group("Validation Targets")
    targets_group.add_argument("--spearman", type=float, default=0.70, help="Target Spearman ρ")
    targets_group.add_argument("--f1", type=float, default=0.80, help="Target F1 score")
    targets_group.add_argument("--precision", type=float, default=0.80, help="Target precision")
    targets_group.add_argument("--recall", type=float, default=0.80, help="Target recall")
    
    parser.add_argument("--output", "-o", metavar="FILE", help="Export results to JSON")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    
    targets = ValidationTargets(spearman=args.spearman, f1_score=args.f1, precision=args.precision, recall=args.recall)
    
    container = Container(uri=args.uri, user=args.user, password=args.password)
    display = container.display_service()
    
    try:
        val_service = container.validation_service(targets=targets)
        
        if args.quick:
            predicted_file, actual_file = args.quick
            with open(predicted_file, 'r') as f: predicted = json.load(f)
            with open(actual_file, 'r') as f: actual = json.load(f)
                
            result = val_service.validate_from_data(predicted, actual)
            
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            elif not args.quiet:
                display.print_header("Quick Validation Result")
                print(f"\n  Files: {predicted_file} vs {actual_file}")
                print(f"  Matched: {result.matched_count} components")
                print(f"\n  Status: {display.status_text(result.passed)}")
                print(f"\n  Spearman:  {result.overall.correlation.spearman:.4f}")
                print(f"  F1 Score:  {result.overall.classification.f1_score:.4f}")
            
            if args.output:
                with open(args.output, 'w') as f: json.dump(result.to_dict(), f, indent=2)
            
            return 0 if result.passed else 1
        
        # Full pipeline validation
        layers = list(LAYER_DEFINITIONS.keys()) if args.all else [l.strip() for l in args.layer.split(",")]
        result = val_service.validate_layers(layers=layers)
        
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        elif not args.quiet:
            display.display_pipeline_validation_result(result)
        
        if args.output:
            with open(args.output, 'w') as f: json.dump(result.to_dict(), f, indent=2)
            if not args.quiet:
                print(f"\n{display.colored(f'Results saved to: {args.output}', display.Colors.GREEN)}")
            
        return 0 if result.all_passed else 1

    except Exception as e:
        print(display.colored(f"Error: {e}", display.Colors.RED), file=sys.stderr)
        if args.verbose: logging.exception("Validation failed")
        return 1
    finally:
        container.close()


if __name__ == "__main__":
    sys.exit(main())
