#!/usr/bin/env python3
"""
bin/predict_graph.py — Run GNN inference on a new graph
======================================================
Loads a pre-trained GNN and predicts component and relationship criticality 
for a new system topology.

Usage
-----
  python bin/predict_graph.py --layer app --checkpoint output/gnn_checkpoints/
  python bin/predict_graph.py --layer app --checkpoint output/gnn_checkpoints/ --compare-rmav
  python bin/predict_graph.py --layer app --checkpoint output/gnn_checkpoints/ --show-edges
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.cli.console import ConsoleDisplay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict_graph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GNN inference — predict criticality on a new system graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layer", default="app",
        help="Layer(s) to process (comma-separated: app,infra,mw,system)"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint directory from train_graph.py")

    neo4j = parser.add_argument_group("Neo4j (live pipeline)")
    neo4j.add_argument("--uri", default=None)
    neo4j.add_argument("--user", default=None)
    neo4j.add_argument("--password", default=None)

    inputs = parser.add_argument_group("Pre-computed inputs")
    inputs.add_argument("--structural", type=str, default=None)
    inputs.add_argument("--rmav", type=str, default=None)
    inputs.add_argument("--simulated", type=str, default=None)

    display_opt = parser.add_argument_group("Display options")
    display_opt.add_argument("--top-n", type=int, default=15, help="Number of nodes")
    display_opt.add_argument("--compare-rmav", action="store_true", help="Side-by-side vs RMAV")
    display_opt.add_argument("--show-edges", action="store_true", help="Display edge scores")
    display_opt.add_argument("--output", type=str, default=None, help="Save to JSON")

    return parser.parse_args()


def load_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    display = ConsoleDisplay()
    display.print_header(f"GNN Prediction: {args.layer.upper()} Layer")

    from src.prediction import GNNService, extract_structural_metrics_dict, \
        extract_rmav_scores_dict

    # ── Data Loading ────────────────────────────────────────────────────────
    structural_raw = load_json(args.structural)
    rmav_raw = load_json(args.rmav)
    simulation_raw = load_json(args.simulated)
    nx_graph = None

    # Flatten multi-layer structures if present
    def _flatten_results(raw_data, sub_key=None):
        if not isinstance(raw_data, dict):
            return raw_data
        if "layers" in raw_data:
            flattened = {}
            for layer_name, layer_res in raw_data["layers"].items():
                # Extract the relevant sub-part (structural or quality)
                part = layer_res.get(sub_key) if sub_key else layer_res
                if part and isinstance(part, dict):
                    # extract_structural_metrics_dict can handle the inner structural part
                    # but here we just want to ensure we have a collection of components
                    # The extract_*_dict functions expect the return of .to_dict()
                    flattened.update(extract_structural_metrics_dict(part) if sub_key == "structural_analysis" 
                                    else extract_rmav_scores_dict(part))
            return flattened
        return raw_data

    # Use the normalisation helpers to get flat {id: metrics} dicts
    structural_dict = _flatten_results(structural_raw, "structural_analysis")
    if not isinstance(structural_dict, dict) or not structural_dict:
        structural_dict = extract_structural_metrics_dict(structural_raw)

    rmav_dict = _flatten_results(rmav_raw, "quality_analysis")
    if not isinstance(rmav_dict, dict) or not rmav_dict:
         rmav_dict = extract_rmav_scores_dict(rmav_raw)
         rmav_dict = extract_rmav_scores_dict(rmav_raw)

    simulation_dict = simulation_raw # extract_simulation_dict is already robust
    
    # Rest of the loading logic...
    if not structural_dict:
        logger.info("No --structural provided; running pipeline steps...")
        try:
            from src.adapters import create_repository
            from src.usecases import AnalyzeGraphUseCase, PredictGraphUseCase
        except ImportError as e:
            logger.error(f"Pipeline modules not available: {e}")
            sys.exit(1)

        conn_kwargs = {k: v for k, v in [("uri", args.uri), ("username", args.user), 
                                         ("password", args.password)] if v}
        repo = create_repository(**conn_kwargs)

        try:
            analyze_uc = AnalyzeGraphUseCase(repo)
            predict_uc = PredictGraphUseCase(repo)
            
            # Step 2: Structural Analysis
            layer_result = analyze_uc.execute(args.layer)
            nx_graph = layer_result.graph
            structural_dict = extract_structural_metrics_dict(layer_result.structural)
            
            if rmav_dict is None:
                # Step 3: Quality Prediction
                quality_res, _ = predict_uc.execute(args.layer)
                rmav_dict = extract_rmav_scores_dict(quality_res)
        finally:
            repo.close()

    if nx_graph is None:
        import networkx as nx
        nx_graph = nx.DiGraph()
        for name in (structural_dict or {}).keys():
            nx_graph.add_node(name, type="Application")

    # ── Inference ───────────────────────────────────────────────────────────
    logger.info(f"Loading GNN models from {args.checkpoint}...")
    try:
        service = GNNService.from_checkpoint(args.checkpoint, graph=nx_graph)
    except Exception as e:
        logger.error(f"Checkpoint failure: {e}")
        sys.exit(1)

    logger.info("Running GNN inference...")
    result = service.predict(
        graph=nx_graph,
        structural_metrics=structural_dict,
        rmav_scores=rmav_dict,
        simulation_results=simulation_dict,
    )

    # ── Results ─────────────────────────────────────────────────────────────
    summary = result.summary()
    display.print_subheader(f"System Summary ({summary['total_components']} components)")
    
    for lvl in ["critical", "high", "medium", "low", "minimal"]:
        count = summary[lvl]
        bar = "█" * min(count * 2, 40)
        color = display.level_color(lvl)
        print(f"  {lvl.upper():<9} {count:>3}  {display.colored(bar, color)}")

    top_nodes = result.top_critical_nodes(args.top_n)
    source_label = "Ensemble" if result.ensemble_scores else "GNN"
    display.print_subheader(f"Top {args.top_n} Critical Components ({source_label})")
    
    header = f"  {'#':<4} {'Component':<30} {'Score':>7} {'Level':<10} {'R':>6} {'M':>6} {'A':>6} {'V':>6}"
    print(display.colored(header, display.Colors.WHITE, bold=True))
    print("  " + "-" * 78)
    for i, s in enumerate(top_nodes, 1):
        color = display.level_color(s.criticality_level)
        print(
            f"  {i:<4} {s.component[:29]:<30} "
            f"{display.colored(f'{s.composite_score:>7.4f}', color)} {display.colored(s.criticality_level[:10], color):<10} "
            f"{s.reliability_score:>6.3f} {s.maintainability_score:>6.3f} "
            f"{s.availability_score:>6.3f} {s.vulnerability_score:>6.3f}"
        )

    # Side-by-side comparison
    if args.compare_rmav and rmav_dict:
        display.print_subheader(f"GNN vs RMAV Comparison (Top {args.top_n} by GNN)")
        print(f"  {'Component':<28} {'GNN':>7} {'RMAV':>7} {'Δ':>7} {'GNN Level':<10} {'RMAV Level':<10}")
        print("  " + "-" * 78)
        for s in top_nodes:
            rmav_score = rmav_dict.get(s.component, {}).get("overall", 0.0)
            diff = s.composite_score - rmav_score
            sign = "+" if diff >= 0 else ""
            
            # Derive RMAV level
            if rmav_score >= 0.75: rl = "CRITICAL"
            elif rmav_score >= 0.55: rl = "HIGH"
            elif rmav_score >= 0.35: rl = "MEDIUM"
            elif rmav_score >= 0.15: rl = "LOW"
            else: rl = "MINIMAL"
            
            d_color = display.Colors.GREEN if abs(diff) < 0.1 else display.Colors.YELLOW
            print(
                f"  {s.component[:27]:<28} "
                f"{s.composite_score:>7.4f} {rmav_score:>7.4f} "
                f"{display.colored(f'{sign}{diff:>6.3f}', d_color)} "
                f"{display.level_color(s.criticality_level)}{s.criticality_level[:10]:<10}{display.Colors.RESET} "
                f"{display.level_color(rl)}{rl:<10}{display.Colors.RESET}"
            )

    if args.show_edges and result.edge_scores:
        top_edges = result.top_critical_edges(args.top_n)
        display.print_subheader(f"Top {args.top_n} Critical Relationships")
        print(f"  {'#':<4} {'Source':<20} {'→':<3} {'Target':<20} {'Type':<15} {'Score':>7}")
        print("  " + "-" * 74)
        for i, e in enumerate(top_edges, 1):
            color = display.level_color(e.criticality_level)
            print(
                f"  {i:<4} {e.source_node[:19]:<20} → {e.target_node[:19]:<20} "
                f"{display.colored(e.edge_type[:14], display.Colors.CYAN):<15} "
                f"{display.colored(f'{e.composite_score:>7.4f}', color)}"
            )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n  {display.colored('Predictions exported:', display.Colors.GREEN)} {out_path}")

    print(f"\n  {display.colored('Done.', display.Colors.GREEN)}")


if __name__ == "__main__":
    main()
