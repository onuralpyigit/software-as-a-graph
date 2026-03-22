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

from src.adapters import create_repository
from src.core import AnalysisLayer
from src.analysis import AnalysisService, MultiLayerAnalysisResult
from src.cli.console import ConsoleDisplay


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
    output.add_argument("--quiet", "-q", action="store_true", help="Suppress console display")
    output.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    return parser


# ---------------------------------------------------------------------------
# Analysis Logic
# ---------------------------------------------------------------------------

def run_analysis(args: argparse.Namespace) -> MultiLayerAnalysisResult:
    """
    Execute analysis based on parsed CLI arguments.
    """
    repo = create_repository(uri=args.uri, user=args.user, password=args.password)
    analyzer = AnalysisService(repo)
    # The following options are now handled by PredictionService/QualityAnalyzer
    # and passed via PredictGraphUseCase if needed.

    from src.usecases import AnalyzeGraphUseCase, PredictGraphUseCase
    from src.analysis.models import LayerAnalysisResult
    from src.prediction import PredictionService, ProblemDetector
    
    # Initialize services with CLI arguments
    pred_svc = PredictionService(
        use_ahp=args.use_ahp,
        normalization_method=args.norm,
        winsorize=args.winsorize,
        winsorize_limit=args.winsorize_limit
    )
    
    analyze_uc = AnalyzeGraphUseCase(repo)
    predict_uc = PredictGraphUseCase(repo, prediction_service=pred_svc)
    detector = ProblemDetector()
    
    def run_gnn(layer_res, model_path):
        from src.prediction import GNNService, extract_structural_metrics_dict, extract_rmav_scores_dict
        try:
            logging.info(f"Loading GNN model for layer {layer_res.layer} from {model_path}...")
            gnn_svc = GNNService.from_checkpoint(model_path, graph=layer_res.structural.graph)
            
            s_dict = extract_structural_metrics_dict(layer_res.structural)
            r_dict = extract_rmav_scores_dict(layer_res.quality)
            
            prediction_result = gnn_svc.predict(
                graph=layer_res.structural.graph,
                structural_metrics=s_dict,
                rmav_scores=r_dict
            )
            layer_res.prediction = prediction_result.to_dict()
            logging.info(f"GNN prediction for {layer_res.layer} complete.")
        except Exception as e:
            logging.error(f"GNN prediction for {layer_res.layer} failed: {e}")

    try:
        if args.all:
            layers = ["app", "infra", "mw", "system"]
            results_map = {}
            for l in layers:
                s_res = analyze_uc.execute(l)
                q_res, problems = predict_uc.execute(l, s_res, detect_problems=True)
                problem_summary = detector.summarize(problems)
                
                layer_res = LayerAnalysisResult(
                    layer=l,
                    layer_name=l.capitalize(),
                    description=f"{l.capitalize()} layer analysis",
                    structural=s_res,
                    quality=q_res,
                    problems=problems,
                    problem_summary=problem_summary
                )
                if args.gnn_model:
                    run_gnn(layer_res, args.gnn_model)
                results_map[l] = layer_res
                
            return MultiLayerAnalysisResult(
                timestamp=datetime.now().isoformat(),
                layers=results_map,
                cross_layer_insights=[]
            )

        # Single-layer analysis
        s_res = analyze_uc.execute(args.layer)
        q_res, problems = predict_uc.execute(args.layer, s_res, detect_problems=True)
        problem_summary = detector.summarize(problems)
        
        layer_result = LayerAnalysisResult(
            layer=args.layer,
            layer_name=args.layer.capitalize(),
            description=f"{args.layer.capitalize()} layer analysis",
            structural=s_res,
            quality=q_res,
            problems=problems,
            problem_summary=problem_summary
        )

        # Optional GNN Inference
        if args.gnn_model:
            run_gnn(layer_result, args.gnn_model)

        return MultiLayerAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layers={layer_result.layer: layer_result},
            cross_layer_insights=[],
        )
    finally:
        repo.close()


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
        from src.core import list_layers
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

    # Initialize display
    display = ConsoleDisplay()

    try:
        # Run the analysis pipeline
        results = run_analysis(args)

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

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            logging.exception("Analysis failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())