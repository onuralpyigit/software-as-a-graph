#!/usr/bin/env python3
"""
Graph Validation CLI (Refactored)

Validates the graph modeling and analysis approach by comparing
predicted criticality scores against actual failure impact.
"""
import sys
from typing import List, Optional, Any
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
from types import SimpleNamespace


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate graph modeling and analysis approach.")
    
    action_group = parser.add_argument_group("Action")
    # Optional arguments for targeted validation
    action_group.add_argument("--layer", "-l", help="Comma-separated layers (e.g., app,infra,system). Defaults to ALL.")
    action_group.add_argument("--quick", "-q", action="store_true", help="Quick validation from JSON files (uses positional args)")
    
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
    parser.add_argument("--dimensional", action="store_true", help="Display dimension-specific metrics (RMAV)")
    
    # Positional arguments for quick validation (optional)
    parser.add_argument("predicted", nargs="?", help="Predicted criticality JSON file (optional if --quick is used)")
    parser.add_argument("actual", nargs="?", help="Actual impact JSON file (optional if --quick is used)")
    
    args = parser.parse_args()
    
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    
    targets = ValidationTargets(spearman=args.spearman, f1_score=args.f1, precision=args.precision, recall=args.recall, top_5_overlap=args.top5)
    
    # Initialize repository and display
    repo = create_repository(uri=args.uri, user=args.user, password=args.password)
    display = ConsoleDisplay()

    try:
        from src.usecases import ValidateGraphUseCase
        
        use_case = ValidateGraphUseCase(repo)

        # Determine inputs for quick validation
        predicted_file = args.predicted
        actual_file = args.actual

        if predicted_file and actual_file:
            from src.analysis import AnalysisService
            from src.prediction import PredictionService
            from src.simulation import SimulationService
            from src.validation import ValidationService
            
            analysis_service = AnalysisService(repo)
            prediction_service = PredictionService()
            simulation_service = SimulationService(repo)
            val_service = ValidationService(analysis_service, prediction_service, simulation_service, targets=targets, ndcg_k=args.ndcg_k)
            
            with open(predicted_file, 'r') as f: predicted_data = json.load(f)
            with open(actual_file, 'r') as f: actual_data = json.load(f)

            predicted_data = _extract_predicted_rich(predicted_data, args.layer)
            
            # Use the selected layer or 'app' as default
            target_layer = args.layer.split(",")[0] if args.layer else "app"
            
            # Simple mocks for quick validation from files
            analysis_mock = MagicMock()
            analysis_mock.quality.components = [] # Simulating data extraction logic
            sim_results_mock = []

            result = val_service.validate_single_layer_from_results(
                analysis_mock, sim_results_mock, target_layer
            )

            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            elif not args.quiet:
                # result here is a LayerValidationResult
                display.print_header("Quick Validation Result")
                print(f"\n  Files: {predicted_file} vs {actual_file}")
                print(f"  Matched: {result.matched_components} components")
                print(f"\n  Status: {display.status_text(result.passed)}")
                print(f"  Spearman:  {result.spearman:.4f}")
                print(f"  F1 Score:  {result.f1_score:.4f}")
                
                display.display_gate_verdicts(result.gates)
                
                if args.dimensional:
                    display.display_dimensional_results(result.dimensional_validation)

            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, 'w') as f: json.dump(result.to_dict(), f, indent=2)

            return 0 if result.passed else 1

        # Standard Pipeline Validation (using Use Case)
        else:
            if sorted_layers := args.layer:
                layers_to_validate = [l.strip() for l in sorted_layers.split(",")]
            else:
                layers_to_validate = ["app", "infra", "mw", "system"]

            result = use_case.execute(layers=layers_to_validate)

            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            elif not args.quiet:
                display.display_pipeline_validation_result(result)
                
                for layer_name, layer_res in result.layers.items():
                    if getattr(layer_res, 'gates', None):
                        display.display_gate_verdicts(layer_res.gates)
                    if args.dimensional and getattr(layer_res, 'dimensional_validation', None):
                        display.display_dimensional_results(layer_res.dimensional_validation)

            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, 'w') as f: json.dump(result.to_dict(), f, indent=2)

            return 0 if result.all_passed else 1
            
            result = val_service.validate_single_layer_from_results(
                analysis_mock, sim_results_mock, target_layer
            )

            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            elif not args.quiet:
                # result here is a LayerValidationResult
                display.print_header("Quick Validation Result")
                print(f"\n  Files: {predicted_file} vs {actual_file}")
                print(f"  Matched: {result.matched_components} components")
                print(f"\n  Status: {display.status_text(result.passed)}")
                print(f"  Spearman:  {result.spearman:.4f}")
                print(f"  F1 Score:  {result.f1_score:.4f}")
                
                display.display_gate_verdicts(result.gates)
                
                if args.dimensional:
                    display.display_dimensional_results(result.dimensional_validation)

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
            
            # Additional display for Gates, Stratification and Dimensions
            for layer_name, layer_res in result.layers.items():
                if getattr(layer_res, 'gates', None):
                    display.display_gate_verdicts(layer_res.gates)
                
                if args.dimensional and getattr(layer_res, 'dimensional_validation', None):
                    display.display_dimensional_results(layer_res.dimensional_validation)

                if getattr(layer_res, 'node_type_stratified', None):
                    print(f"\n    {display.colored('Node-Type Stratified ρ:', display.Colors.WHITE, bold=True)}")
                    for ntype, data in layer_res.node_type_stratified.items():
                        print(f"      - {ntype:11}: ρ={data['spearman']:.4f} (n={data['n']})")

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
                prediction_service = PredictionService()
                simulation_service = SimulationService(repo)
                viz_service = VisualizationService(
                    analysis_service=analysis_service,
                    prediction_service=prediction_service,
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

def _extract_predicted_rich(data: dict, filter_layer: Optional[str] = None) -> dict:
    """Extract {id: {scores: {}, type: ""}} from AnalysisService output."""
    if not isinstance(data, dict):
        return {}
    
    extracted = {}
    
    # Mapping of layer to component types
    LAYER_MAP = {
        "app": {"Application"},
        "infra": {"Node"},
        "mw": {"Broker"},
        "system": None
    }
    
    req_types = set()
    if filter_layer:
        for l in filter_layer.split(","):
            l = l.strip().lower()
            if l in LAYER_MAP and LAYER_MAP[l]:
                req_types.update(LAYER_MAP[l])
            elif l == "system":
                req_types = None
                break

    # Try nested structure: layers -> [layer] -> quality_analysis -> components
    if "layers" in data:
        for layer_info in data["layers"].values():
            components = layer_info.get("quality_analysis", {}).get("components")
            if not components:
                components = layer_info.get("components")
                
            if components:
                for comp in components:
                    ctype = comp.get("type", "Unknown")
                    if req_types is not None and ctype not in req_types:
                        continue
                        
                    extracted[comp["id"]] = {
                        "scores": comp["scores"],
                        "type": ctype,
                        "name": comp.get("name", comp["id"]),
                        "metrics": comp.get("metrics", {}),
                        "is_articulation_point": comp.get("is_articulation_point", False)
                    }
        return extracted
    
    # Try flat dict (already scores)
    if isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
        return {k: {"scores": {"overall": v}, "type": "Unknown"} for k, v in data.items()}
        
    return data

def _extract_actual(data: Any) -> dict:
    """Extract {id: score} from FailureSimulator exhaustive output."""
    if isinstance(data, list):
        scores = {}
        for res in data:
            target_id = res.get("target_id")
            impact = res.get("impact", {})
            score = impact.get("composite_impact", 0.0)
            if target_id:
                scores[target_id] = score
        return scores
    
    # Try flat dict
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    sys.exit(main())
