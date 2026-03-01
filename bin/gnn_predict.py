#!/usr/bin/env python3
"""
bin/gnn_predict.py — Run GNN inference on a new graph
======================================================
Loads a pre-trained GNN (from ``bin/gnn_train.py``) and predicts component
and relationship criticality for a new system topology — without requiring
re-training or failure simulation.

This is the key capability of the GNN extension: **pre-deployment prediction
on unseen system instances using learned structural representations**.

Usage
-----
  # Predict using live Neo4j graph
  python bin/gnn_predict.py --layer app --checkpoint output/gnn_checkpoints/

  # Predict from pre-exported structural metrics only (no pipeline required)
  python bin/gnn_predict.py \\
      --structural results/metrics.json \\
      --rmav       results/quality.json \\
      --checkpoint output/gnn_checkpoints/

  # Compare GNN vs RMAV side-by-side with ground truth
  python bin/gnn_predict.py \\
      --layer app \\
      --checkpoint output/gnn_checkpoints/ \\
      --compare-rmav \\
      --simulated results/impact.json

  # Export predictions to JSON
  python bin/gnn_predict.py --layer app --checkpoint output/gnn_checkpoints/ \\
      --output results/gnn_predictions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gnn_predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GNN inference — predict criticality on a new system graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layer", default="app",
        choices=["app", "infra", "mw", "system"],
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint directory from gnn_train.py")

    neo4j = parser.add_argument_group("Neo4j (live pipeline)")
    neo4j.add_argument("--uri", default=None)
    neo4j.add_argument("--user", default=None)
    neo4j.add_argument("--password", default=None)

    inputs = parser.add_argument_group("Pre-computed inputs")
    inputs.add_argument("--structural", type=str, default=None,
                        help="Path to structural metrics JSON")
    inputs.add_argument("--rmav", type=str, default=None,
                        help="Path to RMAV quality scores JSON (for ensemble)")
    inputs.add_argument("--simulated", type=str, default=None,
                        help="Optional: simulation results for validation metrics")

    display = parser.add_argument_group("Display options")
    display.add_argument("--top-n", type=int, default=15,
                         help="Number of top components to display (default: 15)")
    display.add_argument("--compare-rmav", action="store_true",
                         help="Show side-by-side GNN vs RMAV score comparison")
    display.add_argument("--show-edges", action="store_true",
                         help="Display top critical edges/relationships")
    display.add_argument("--output", type=str, default=None,
                         help="Export predictions to JSON file")

    return parser.parse_args()


def load_json(path: str | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    logger.info("=" * 65)
    logger.info("GNN Inference | layer=%s | checkpoint=%s", args.layer, args.checkpoint)
    logger.info("=" * 65)

    from src.gnn import GNNService, extract_structural_metrics_dict, extract_rmav_scores_dict

    # ── Load pre-computed data ────────────────────────────────────────────────
    structural_dict = load_json(args.structural)
    rmav_dict = load_json(args.rmav)
    simulation_dict = load_json(args.simulated)
    nx_graph = None

    if structural_dict is None:
        logger.info("No --structural provided; running pipeline steps…")
        try:
            from src.core import create_repository
            from src.analysis import AnalysisService
        except ImportError as e:
            logger.error("Pipeline not available: %s", e)
            sys.exit(1)

        conn_kwargs = {k: v for k, v in
                       [("uri", args.uri), ("username", args.user), ("password", args.password)]
                       if v is not None}
        try:
            repo = create_repository(**conn_kwargs)
        except Exception as e:
            logger.error("Neo4j connection failed: %s", e)
            sys.exit(1)

        try:
            analysis_svc = AnalysisService(repo)
            layer_result = analysis_svc.analyze_layer(args.layer)
            nx_graph = layer_result.graph
            structural_dict = extract_structural_metrics_dict(layer_result.structural)
            if rmav_dict is None:
                rmav_dict = extract_rmav_scores_dict(layer_result.quality)
        finally:
            repo.close()

    # Build a skeleton graph if needed for HeteroData metadata
    if nx_graph is None:
        import networkx as nx
        nx_graph = nx.DiGraph()
        for name in (structural_dict or {}).keys():
            nx_graph.add_node(name, type="Application")

    # ── Load GNN service ──────────────────────────────────────────────────────
    logger.info("Loading GNN models from checkpoint…")
    try:
        service = GNNService.from_checkpoint(
            checkpoint_dir=args.checkpoint,
            graph=nx_graph,
        )
    except Exception as e:
        logger.error("Failed to load checkpoint: %s", e)
        sys.exit(1)

    # ── Run prediction ────────────────────────────────────────────────────────
    logger.info("Running GNN inference…")
    result = service.predict(
        graph=nx_graph,
        structural_metrics=structural_dict,
        rmav_scores=rmav_dict,
        simulation_results=simulation_dict,
    )

    # ── Display results ───────────────────────────────────────────────────────
    n = args.top_n

    print("\n" + "=" * 70)
    print("GNN CRITICALITY PREDICTIONS")
    print("=" * 70)

    summary = result.summary()
    print(f"\nSystem summary ({summary['total_components']} components):")
    for lvl in ["critical", "high", "medium", "low", "minimal"]:
        bar = "█" * summary[lvl]
        print(f"  {lvl.upper():<9} {summary[lvl]:>4}  {bar}")
    print(f"  Critical edges: {summary['critical_edges']}")

    top_nodes = result.top_critical_nodes(n)
    header = f"\nTop {n} Critical Components"
    header += " (Ensemble)" if result.ensemble_scores else " (GNN)"
    print(header)
    print(f"  {'#':<4} {'Component':<30} {'Score':>7} {'Level':<10} "
          f"{'R':>6} {'M':>6} {'A':>6} {'V':>6}")
    print("  " + "-" * 78)
    for i, s in enumerate(top_nodes, 1):
        print(
            f"  {i:<4} {s.component[:29]:<30} "
            f"{s.composite_score:>7.4f} {s.criticality_level:<10} "
            f"{s.reliability_score:>6.3f} {s.maintainability_score:>6.3f} "
            f"{s.availability_score:>6.3f} {s.vulnerability_score:>6.3f}"
        )

    # ── Side-by-side RMAV comparison ──────────────────────────────────────────
    if args.compare_rmav and rmav_dict:
        print(f"\nSide-by-side: GNN vs RMAV (top {n} by GNN score)")
        print(f"  {'Component':<28} {'GNN':>7} {'RMAV':>7} {'Δ':>7} {'GNN Level':<10} {'RMAV Level':<10}")
        print("  " + "-" * 78)
        for s in top_nodes:
            rmav_composite = rmav_dict.get(s.component, {}).get("overall", 0.0)
            delta = s.composite_score - rmav_composite
            sign = "+" if delta >= 0 else ""
            # Derive RMAV level from score
            if rmav_composite >= 0.75:
                rmav_level = "CRITICAL"
            elif rmav_composite >= 0.55:
                rmav_level = "HIGH"
            elif rmav_composite >= 0.35:
                rmav_level = "MEDIUM"
            elif rmav_composite >= 0.15:
                rmav_level = "LOW"
            else:
                rmav_level = "MINIMAL"
            print(
                f"  {s.component[:27]:<28} "
                f"{s.composite_score:>7.4f} {rmav_composite:>7.4f} "
                f"{sign}{delta:>6.4f} "
                f"{s.criticality_level:<10} {rmav_level:<10}"
            )

    # ── Edge criticality ──────────────────────────────────────────────────────
    if args.show_edges and result.edge_scores:
        top_edges = result.top_critical_edges(n)
        print(f"\nTop {n} Critical Relationships:")
        print(f"  {'#':<4} {'Source':<22} {'Target':<22} {'Type':<16} {'Score':>7} {'Level':<10}")
        print("  " + "-" * 86)
        for i, e in enumerate(top_edges, 1):
            print(
                f"  {i:<4} {e.source_node[:21]:<22} {e.target_node[:21]:<22} "
                f"{e.edge_type[:15]:<16} {e.composite_score:>7.4f} {e.criticality_level:<10}"
            )

    # ── Validation metrics ────────────────────────────────────────────────────
    if result.gnn_metrics:
        print("\nGNN Validation Metrics (vs simulation ground truth):")
        print(result.gnn_metrics)
        if rmav_dict and simulation_dict:
            print("\n  Comparison with RMAV baseline:")
            print("  (Run 'python bin/validate_graph.py' for full RMAV metrics)")

    # ── Ensemble weights ──────────────────────────────────────────────────────
    if result.ensemble_alpha:
        print("\nLearned ensemble weights (α = GNN contribution):")
        dims = ["composite", "reliability", "maintainability", "availability", "vulnerability"]
        for dim, a in zip(dims, result.ensemble_alpha):
            gnn_pct = a * 100
            rmav_pct = (1 - a) * 100
            bar_gnn = "▓" * int(gnn_pct / 5)
            bar_rmav = "░" * int(rmav_pct / 5)
            print(f"  {dim:<20}: GNN {gnn_pct:5.1f}% {bar_gnn}{bar_rmav} RMAV {rmav_pct:5.1f}%")

    # ── Export ────────────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nPredictions exported to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
