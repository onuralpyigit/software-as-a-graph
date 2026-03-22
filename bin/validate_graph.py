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

from src.infrastructure import create_repository
from src.core import SimulationLayer
from src.validation import ValidationService, ValidationTargets
from common.console import ConsoleDisplay
from types import SimpleNamespace


from common.dispatcher import dispatch_validate
from common.arguments import add_neo4j_arguments, add_common_arguments


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate graph modeling and analysis approach.")
    
    action_group = parser.add_argument_group("Action")
    # Optional arguments for targeted validation
    action_group.add_argument("--layer", "-l", help="Comma-separated layers (e.g., app,infra,system). Defaults to ALL.")
    action_group.add_argument("--quick", "-Q", action="store_true", help="Quick validation from JSON files (uses positional args)")
    
    add_neo4j_arguments(parser)
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
    add_common_arguments(parser)
    parser.add_argument("--dimensional", action="store_true", help="Display dimension-specific metrics (RMAV)")
    
    # Positional arguments for quick validation (optional)
    parser.add_argument("predicted", nargs="?", help="Predicted criticality JSON file (optional if --quick is used)")
    parser.add_argument("actual", nargs="?", help="Actual impact JSON file (optional if --quick is used)")
    
    args = parser.parse_args()
    
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    
    # Initialize repository and display
    repo = create_repository(uri=args.uri, user=args.user, password=args.password)
    display = ConsoleDisplay()

    try:
        # Determine if quick validation or standard pipeline
        if args.predicted and args.actual:
            # Quick validation logic remains here as it's a CLI-specific shortcut
            from src.validation import Validator
            with open(args.predicted, 'r') as f: predicted_raw = json.load(f)
            with open(args.actual, 'r') as f: actual_raw = json.load(f)

            predicted_processed = _extract_predicted_rich(predicted_raw, args.layer)
            from src.prediction.data_preparation import extract_simulation_dict
            actual_processed = extract_simulation_dict(actual_raw)
            impact_data = actual_raw.get("results", []) if isinstance(actual_raw, dict) else actual_raw

            pred_scores = {cid: data["scores"]["overall"] for cid, data in predicted_processed.items()}
            actual_scores = {cid: data["composite"] for cid, data in actual_processed.items() if "composite" in data}
            comp_types = {cid: data["type"] for cid, data in predicted_processed.items()}
            
            validator = Validator()
            target_layer = args.layer.split(",")[0] if args.layer else "app"
            result = validator.validate(
                predicted_scores=pred_scores,
                actual_scores=actual_scores,
                impact_data=impact_data,
                component_types=comp_types,
                layer=target_layer,
                context=f"Quick Validation: {target_layer}"
            )
        else:
            # Standard Pipeline Validation via dispatcher
            result = dispatch_validate(repo, args)

        if args.json:
            if hasattr(result, 'to_dict'):
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(json.dumps(result, indent=2))
        elif not args.quiet:
            if hasattr(result, 'layers'):
                 display.display_pipeline_validation_result(result)
            elif hasattr(result, 'passed'):
                 display.print_header("Quick Validation Result")
                 print(f"\n  Status: {display.status_text(result.passed)}")
                 if hasattr(result, 'overall') and hasattr(result.overall, 'correlation'):
                     print(f"  Spearman:  {result.overall.correlation.spearman:.4f}")
                 if hasattr(result, 'overall') and hasattr(result.overall, 'classification'):
                     print(f"  F1 Score:  {result.overall.classification.f1_score:.4f}")
                 if hasattr(result, 'gates'):
                     display.display_gate_verdicts(result.gates)
            elif isinstance(result, dict):
                 display.print_header("Validation Result (Summary)")
                 print(f"\n  Status: {'PASSED' if result.get('passed') else 'FAILED'}")

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                if hasattr(result, 'to_dict'):
                    json.dump(result.to_dict(), f, indent=2)
                else:
                    json.dump(result, f, indent=2)

        # Final return code
        if hasattr(result, 'passed'):
            all_passed = getattr(result, 'all_passed', True)
            return 0 if (all_passed and result.passed) else 1
        elif isinstance(result, dict):
            return 0 if result.get('passed', False) else 1
        return 0

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
    
    # NEW: Try GNN output structure (node_scores or ensemble_scores)
    gnn_src = data.get("ensemble_scores") or data.get("node_scores")
    if gnn_src:
        for cid, score_data in gnn_src.items():
            # If score_data is already a dict (from to_dict())
            scores = {
                "overall": score_data.get("composite_score"),
                "reliability": score_data.get("reliability_score"),
                "maintainability": score_data.get("maintainability_score"),
                "availability": score_data.get("availability_score"),
                "vulnerability": score_data.get("vulnerability_score"),
            }
            extracted[cid] = {
                "scores": scores,
                "type": "Unknown", # GNN output currently doesn't carry type
                "name": score_data.get("component") or cid,
                "is_articulation_point": False
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
