#!/usr/bin/env python3
"""
bin/gnn_train.py — Train GNN criticality models
=================================================
Trains a Heterogeneous Graph Attention Network (HeteroGAT) to predict
component and relationship criticality using simulation ground-truth labels.

Workflow
--------
1. Connects to Neo4j (same as the existing pipeline)
2. Exports the structural graph for the requested layer
3. Runs structural analysis (Step 2) to obtain node feature vectors
4. Runs quality scoring (Step 3) to obtain RMAV scores
5. Runs failure simulation (Step 4) to obtain ground-truth labels I(v)
6. Converts to HeteroData and trains the GNN
7. Evaluates on held-out nodes and prints metric comparison vs RMAV

Usage
-----
  python bin/gnn_train.py --layer app
  python bin/gnn_train.py --layer system --epochs 500 --hidden 128 --heads 8
  python bin/gnn_train.py --layer app --multi-scenario --checkpoint output/gnn_ckpt/

  # Load existing structural/simulation results instead of re-running
  python bin/gnn_train.py --layer app \\
      --structural results/metrics.json \\
      --simulated  results/impact.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gnn_train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GNN criticality models on pub-sub system graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layer", default="app",
        choices=["app", "infra", "mw", "system"],
        help="System layer to analyse (default: app)",
    )

    # Neo4j connection
    neo4j = parser.add_argument_group("Neo4j connection")
    neo4j.add_argument("--uri", default=None,
                       help="Neo4j URI (overrides NEO4J_URI env var)")
    neo4j.add_argument("--user", default=None,
                       help="Neo4j username (overrides NEO4J_USERNAME)")
    neo4j.add_argument("--password", default=None,
                       help="Neo4j password (overrides NEO4J_PASSWORD)")

    # Pre-computed inputs (skip pipeline steps)
    inputs = parser.add_argument_group("Pre-computed inputs (skip pipeline steps)")
    inputs.add_argument("--structural", type=str, default=None,
                        help="Path to structural metrics JSON (skips Step 2)")
    inputs.add_argument("--simulated", type=str, default=None,
                        help="Path to simulation results JSON (skips Step 4)")
    inputs.add_argument("--rmav", type=str, default=None,
                        help="Path to RMAV scores JSON (skips Step 3)")

    # GNN hyperparameters
    gnn = parser.add_argument_group("GNN hyperparameters")
    gnn.add_argument("--hidden", type=int, default=64,
                     help="Hidden embedding dimension (default: 64)")
    gnn.add_argument("--heads", type=int, default=4,
                     help="GAT attention heads (default: 4)")
    gnn.add_argument("--layers", type=int, default=3,
                     help="Message-passing depth (default: 3)")
    gnn.add_argument("--dropout", type=float, default=0.2,
                     help="Dropout probability (default: 0.2)")
    gnn.add_argument("--epochs", type=int, default=300,
                     help="Maximum training epochs (default: 300)")
    gnn.add_argument("--lr", type=float, default=3e-4,
                     help="Learning rate (default: 3e-4)")
    gnn.add_argument("--patience", type=int, default=30,
                     help="Early-stopping patience epochs (default: 30)")
    gnn.add_argument("--train-ratio", type=float, default=0.6,
                     help="Fraction of nodes for training (default: 0.6)")
    gnn.add_argument("--val-ratio", type=float, default=0.2,
                     help="Fraction of nodes for validation (default: 0.2)")
    gnn.add_argument("--no-edge-model", action="store_true",
                     help="Skip EdgeCriticalityGNN (node-only prediction)")

    # Output
    output = parser.add_argument_group("Output")
    output.add_argument("--checkpoint", default="output/gnn_checkpoints",
                        help="Directory for model checkpoints (default: output/gnn_checkpoints)")
    output.add_argument("--output", default=None,
                        help="Save GNN analysis results to JSON file")
    output.add_argument("--use-ahp", action="store_true",
                        help="Use AHP-derived weights for RMAV quality scoring")

    return parser.parse_args()


def load_json_if_path(path: str | None) -> dict | None:
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
    logger.info("GNN Training Pipeline | layer=%s | epochs=%d", args.layer, args.epochs)
    logger.info("=" * 65)

    # ── Load GNN module ───────────────────────────────────────────────────────
    try:
        from src.gnn import GNNService
    except ImportError as e:
        logger.error("Cannot import GNN module: %s", e)
        sys.exit(1)

    # ── Step 1-4: Connect to pipeline or load pre-computed results ────────────
    structural_dict = load_json_if_path(args.structural)
    simulation_dict = load_json_if_path(args.simulated)
    rmav_dict = load_json_if_path(args.rmav)
    nx_graph = None

    if any(x is None for x in [structural_dict, simulation_dict]):
        logger.info("Connecting to Neo4j and running pipeline steps…")
        try:
            from src.core import create_repository
            from src.analysis import AnalysisService
            from src.simulation import SimulationService
        except ImportError as e:
            logger.error(
                "Pipeline modules not available: %s\n"
                "Provide --structural and --simulated JSON files to skip pipeline steps.",
                e,
            )
            sys.exit(1)

        # Build connection kwargs
        conn_kwargs = {}
        if args.uri:
            conn_kwargs["uri"] = args.uri
        if args.user:
            conn_kwargs["username"] = args.user
        if args.password:
            conn_kwargs["password"] = args.password

        try:
            repo = create_repository(**conn_kwargs)
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", e)
            sys.exit(1)

        try:
            # Step 2+3: Structural analysis + RMAV scoring
            if structural_dict is None or rmav_dict is None:
                logger.info("[Step 2+3] Running structural analysis and RMAV scoring…")
                analysis_svc = AnalysisService(repo, use_ahp=args.use_ahp)
                layer_result = analysis_svc.analyze_layer(args.layer)
                nx_graph = layer_result.graph

                from src.gnn import extract_structural_metrics_dict, extract_rmav_scores_dict
                if structural_dict is None:
                    structural_dict = extract_structural_metrics_dict(layer_result.structural)
                if rmav_dict is None:
                    rmav_dict = extract_rmav_scores_dict(layer_result.quality)

            # Step 4: Failure simulation
            if simulation_dict is None:
                logger.info("[Step 4] Running exhaustive failure simulation…")
                sim_svc = SimulationService(repo)
                sim_results = sim_svc.run_failure_simulation_exhaustive(layer=args.layer)
                from src.gnn import extract_simulation_dict
                simulation_dict = extract_simulation_dict(sim_results)
                logger.info("  Simulated %d components.", len(simulation_dict))

        finally:
            repo.close()

    # If no nx_graph (all pre-computed), reconstruct a minimal one from keys
    if nx_graph is None:
        logger.info(
            "No live graph available — building skeleton NetworkX graph from "
            "structural metrics keys for HeteroData conversion."
        )
        import networkx as nx
        nx_graph = nx.DiGraph()
        for name, metrics in (structural_dict or {}).items():
            nx_graph.add_node(name, type="Application")  # type will be overridden by metrics if present

    # ── Train GNN ─────────────────────────────────────────────────────────────
    logger.info("\n[GNN] Initialising GNNService…")
    service = GNNService(
        hidden_channels=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
        predict_edges=not args.no_edge_model,
        checkpoint_dir=args.checkpoint,
    )

    logger.info("[GNN] Starting training…")
    result = service.train(
        graph=nx_graph,
        structural_metrics=structural_dict,
        simulation_results=simulation_dict,
        rmav_scores=rmav_dict,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    # ── Save models ───────────────────────────────────────────────────────────
    saved_dir = service.save()
    logger.info("\nModels saved to: %s", saved_dir)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("GNN CRITICALITY ANALYSIS RESULTS")
    print("=" * 65)

    summary = result.summary()
    print(f"\nSummary ({summary['total_components']} components):")
    print(f"  CRITICAL : {summary['critical']}")
    print(f"  HIGH     : {summary['high']}")
    print(f"  MEDIUM   : {summary['medium']}")
    print(f"  LOW      : {summary['low']}")
    print(f"  MINIMAL  : {summary['minimal']}")
    print(f"  Critical edges: {summary['critical_edges']}")

    print("\nTop 10 Critical Components (Ensemble):")
    print(f"  {'Rank':<5} {'Component':<30} {'Score':>7} {'Level':<10} {'R':>6} {'M':>6} {'A':>6} {'V':>6}")
    print("  " + "-" * 79)
    for i, score in enumerate(result.top_critical_nodes(10), 1):
        print(
            f"  {i:<5} {score.component[:29]:<30} "
            f"{score.composite_score:>7.4f} {score.criticality_level:<10} "
            f"{score.reliability_score:>6.3f} {score.maintainability_score:>6.3f} "
            f"{score.availability_score:>6.3f} {score.vulnerability_score:>6.3f}"
        )

    if result.edge_scores:
        print("\nTop 10 Critical Relationships (Edges):")
        print(f"  {'Rank':<5} {'Source':<20} {'→':<3} {'Target':<20} {'Type':<15} {'Score':>7}")
        print("  " + "-" * 74)
        for i, e in enumerate(result.top_critical_edges(10), 1):
            print(
                f"  {i:<5} {e.source_node[:19]:<20} → {e.target_node[:19]:<20} "
                f"{e.edge_type[:14]:<15} {e.composite_score:>7.4f}"
            )

    if result.gnn_metrics:
        print("\nGNN Validation Metrics (test set):")
        print(result.gnn_metrics)

    if result.ensemble_alpha:
        print(f"\nLearned ensemble α (composite/R/M/A/V):")
        cols = ["composite", "reliability", "maintainability", "availability", "vulnerability"]
        for col, a in zip(cols, result.ensemble_alpha):
            print(f"  {col:<20}: {a:.4f} (GNN weight)")

    # ── Export results JSON ───────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info("Results exported to: %s", out_path)

    print("\nDone. Next step: python bin/gnn_predict.py --layer", args.layer)


if __name__ == "__main__":
    main()
