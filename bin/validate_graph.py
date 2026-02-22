#!/usr/bin/env python3
"""
Graph Validation CLI (Refactored)

Validates the graph modeling and analysis approach by comparing
predicted criticality scores against actual failure impact.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))

import argparse
import json
import logging
import webbrowser

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import create_repository, SimulationLayer
from src.validation import ValidationService, ValidationTargets
from src.cli.console import ConsoleDisplay


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate graph modeling and analysis approach.")
    
    action_group = parser.add_argument_group("Action")
    # Optional arguments for targeted validation
    action_group.add_argument("--layer", "-l", help="Comma-separated layers (e.g., app,infra,system). Defaults to ALL.")
    action_group.add_argument("--quick", "-q", nargs=2, metavar=("PREDICTED", "ACTUAL"), help="Quick validation from JSON files")
    
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
    targets_group.add_argument("--top5", type=float, default=0.40, help="Target top-5 overlap")
    targets_group.add_argument("--ndcg-k", type=int, default=10, help="K for NDCG@K calculation")
    
    parser.add_argument("--output", "-o", metavar="FILE", help="Export results to JSON")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization dashboard")
    parser.add_argument("--viz-output", default="validation_dashboard.html", help="Visualization output file")
    parser.add_argument("--open", "-O", action="store_true", help="Open visualization in browser")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    
    targets = ValidationTargets(spearman=args.spearman, f1_score=args.f1, precision=args.precision, recall=args.recall, top_5_overlap=args.top5)
    
    # Initialize repository and display
    repo = create_repository(uri=args.uri, user=args.user, password=args.password)
    display = ConsoleDisplay()

    try:
        from src.analysis import AnalysisService
        from src.simulation import SimulationService
        
        analysis_service = AnalysisService(repo)
        simulation_service = SimulationService(repo)
        val_service = ValidationService(analysis_service, simulation_service, targets=targets, ndcg_k=args.ndcg_k)

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
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, 'w') as f: json.dump(result.to_dict(), f, indent=2)

            return 0 if result.passed else 1

        if sorted_layers := args.layer:
            layers_to_validate = [l.strip() for l in sorted_layers.split(",")]
        else:
            # Default to all primary layers
            layers_to_validate = [layer.value for layer in SimulationLayer]

        result = val_service.validate_layers(layers=layers_to_validate)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        elif not args.quiet:
            display.display_pipeline_validation_result(result)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f: json.dump(result.to_dict(), f, indent=2)
            if not args.quiet:
                print(f"\n{display.colored(f'Results saved to: {args.output}', display.Colors.GREEN)}")

        print(f"\n{display.colored(f'Validation passed: {result.all_passed}', display.Colors.GREEN)}")

        # Generate visualization if requested
        if args.visualize:
            try:
                from src.visualization import VisualizationService
                from src.analysis import AnalysisService
                from src.simulation import SimulationService
                
                # VisualizationService needs several services in this version
                analysis_service = AnalysisService(repo)
                simulation_service = SimulationService(repo)
                viz_service = VisualizationService(
                    analysis_service=analysis_service,
                    simulation_service=simulation_service,
                    validation_service=val_service,
                    repository=repo
                )
                viz_path = viz_service.generate_dashboard(
                    output_file=args.viz_output,
                    layers=layers_to_validate,
                    include_validation=True
                )
                import os
                abs_path = os.path.abspath(viz_path)
                print(f"\n{display.colored(f'Dashboard generated: {abs_path}', display.Colors.GREEN)}")
                if args.open:
                    webbrowser.open(f"file://{abs_path}")
            except Exception as viz_err:
                print(display.colored(f"\nVisualization error: {viz_err}", display.Colors.YELLOW), file=sys.stderr)

        return 0

    except Exception as e:
        print(display.colored(f"Error: {e}", display.Colors.RED), file=sys.stderr)
        if args.verbose: logging.exception("Validation failed")
        return 1
    finally:
        repo.close()


if __name__ == "__main__":
    sys.exit(main())
