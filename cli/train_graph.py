#!/usr/bin/env python3
"""
cli/train_graph.py — Train GNN criticality models
=================================================
Trains a Heterogeneous Graph Attention Network (HeteroGAT) to predict
component and relationship criticality using simulation ground-truth labels.

Usage
-----
  python cli/train_graph.py --layer app
  python cli/train_graph.py --layer system --epochs 500 --hidden 128 --heads 8
  python cli/train_graph.py --layer app --checkpoint output/gnn_checkpoints/

  # Load existing structural/simulation results instead of re-running
  python cli/train_graph.py --layer app \
      --structural results/metrics.json \
      --simulated  results/impact.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from cli.common.console import ConsoleDisplay
from cli.common.arguments import add_neo4j_arguments, add_runtime_arguments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_graph")


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

    # --- Neo4j Connection ---
    add_neo4j_arguments(parser)

    # Pre-computed inputs
    inputs = parser.add_argument_group("Pre-computed inputs (skip pipeline steps)")
    inputs.add_argument("--structural", type=str, default=None,
                        help="Path to structural metrics JSON (skips Step 2)")
    inputs.add_argument("--simulated", type=str, default=None,
                        help="Path to simulation results JSON (skips Step 4)")
    inputs.add_argument("--rmav", type=str, default=None,
                        help="Path to RMAV scores JSON (skips Step 3)")

    # GNN hyperparameters
    gnn = parser.add_argument_group("GNN hyperparameters")
    gnn.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    gnn.add_argument("--heads", type=int, default=4, help="Attention heads")
    gnn.add_argument("--layers", type=int, default=3, help="GNN layers")
    gnn.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    gnn.add_argument("--epochs", type=int, default=300, help="Max epochs")
    gnn.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    gnn.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    gnn.add_argument("--train-ratio", type=float, default=0.6, help="Train split")
    gnn.add_argument("--val-ratio", type=float, default=0.2, help="Val split")
    gnn.add_argument("--no-edge-model", action="store_true", help="Skip edge model")
    gnn.add_argument("--seeds", type=int, nargs="+", help="Seed list for stability validation")
    gnn.add_argument("--multi-scenario", action="store_true", help="Inductive training on all domain scenarios")
    gnn.add_argument("--mode", choices=["rmav", "gnn", "ensemble"], default="ensemble", help="Evaluation path for final summary (default: ensemble)")

    # Output
    output = parser.add_argument_group("Output")
    output.add_argument("--checkpoint", default="output/gnn_checkpoints",
                        help="Checkpoint directory")
    output.add_argument("--output", default=None, help="Save result JSON")
    output.add_argument("--use-ahp", action="store_true", help="Use AHP weights for RMAV")
    add_runtime_arguments(parser)

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
    display.print_header(f"GNN Training: {args.layer.upper()} Layer")

    # ── Imports ─────────────────────────────────────────────────────────────
    try:
        from saag.prediction import GNNService, extract_structural_metrics_dict, \
            extract_rmav_scores_dict, extract_simulation_dict
    except ImportError as e:
        logger.error(f"GNN module not available: {e}")
        sys.exit(1)

    # ── Data Loading ────────────────────────────────────────────────────────
    structural_dict = load_json(args.structural)
    simulation_dict = load_json(args.simulated)
    rmav_dict = load_json(args.rmav)
    nx_graph = None

    if any(x is None for x in [structural_dict, simulation_dict]):
        display.print_step("Connecting to Neo4j to retrieve graph data...")
        try:
            from saag.analysis import AnalysisService
            from saag.simulation import SimulationService
        except ImportError as e:
            display.print_error(f"Pipeline modules not available: {e}")
            sys.exit(1)

        from saag import Client
        try:
            client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
            repo = client.repo
            if not repo:
                raise ValueError("No repository connection established.")

            # We need the inner repository here to get the raw graph
            from saag.usecases import AnalyzeGraphUseCase
            analyze_uc = AnalyzeGraphUseCase(repo)

            if structural_dict is None or rmav_dict is None:
                display.print_step("[Step 2+3] Running analysis and quality scoring...")
                analysis_svc = AnalysisService(repo)
                layer_result = analysis_svc.analyze_layer(args.layer)
                nx_graph = layer_result.graph
                if structural_dict is None:
                    structural_dict = extract_structural_metrics_dict(layer_result.structural)
                if rmav_dict is None:
                    rmav_dict = extract_rmav_scores_dict(layer_result.quality)

            if simulation_dict is None:
                display.print_step("[Step 4] Running failure simulation ground truth (exhaustive)...")
                sim_svc = SimulationService(repo)
                sim_results = sim_svc.run_failure_simulation_exhaustive(layer=args.layer)
                simulation_dict = extract_simulation_dict(sim_results)
        finally:
            if 'repo' in locals() and repo:
                repo.close()

    if nx_graph is None:
        import networkx as nx
        nx_graph = nx.DiGraph()
        for name in (structural_dict or {}).keys():
            nx_graph.add_node(name, type="Application")

    # ── Inductive Data Discovery ────────────────────────────────────────────
    inductive_graphs = []
    if args.multi_scenario:
        logger.info("Discovering additional scenarios for inductive training...")
        output_root = Path("output")
        for scenario_dir in output_root.glob("*_results"):
            if scenario_dir.is_dir() and args.layer in scenario_dir.name:
                # This is a heuristic; better would be to check config inside
                pass
            
            # Look for pre-computed files in the output directory
            # For simplicity, we search for standard results files
            s_path = scenario_dir / "structural_metrics.json"
            q_path = scenario_dir / "quality_scores.json" 
            i_path = scenario_dir / "failure_impact.json"
            
            if s_path.exists() and i_path.exists():
                logger.info(f"  Found scenario: {scenario_dir.name}")
                s_dict = load_json(str(s_path))
                i_dict = load_json(str(i_path))
                r_dict = load_json(str(q_path)) if q_path.exists() else None
                
                # Create a minimal graph for conversion
                tmp_graph = nx.DiGraph()
                for n_id in s_dict.keys():
                    tmp_graph.add_node(n_id, type="Application")
                
                # Convert to HeteroData
                from saag.prediction.data_preparation import networkx_to_hetero_data
                h_conv = networkx_to_hetero_data(tmp_graph, s_dict, i_dict, r_dict)
                inductive_graphs.append(h_conv.hetero_data)

    # ── Training ────────────────────────────────────────────────────────────
    service = GNNService(
        hidden_channels=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
        predict_edges=not args.no_edge_model,
        checkpoint_dir=args.checkpoint,
    )

    display.print_step("Starting GNN training session...")
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
        inductive_graphs=inductive_graphs if inductive_graphs else None,
        seeds=args.seeds,
        mode=args.mode,
        layer=args.layer,
    )

    # ── Results ─────────────────────────────────────────────────────────────
    service.save()
    display.print_success(f"Models saved to {args.checkpoint}")
    
    display.display_training_summary(result.summary())
    
    if result.gnn_metrics:
        display.display_training_metrics(result.gnn_metrics.to_dict())

    # Display Top 10 using specialized method
    display.display_top_critical_components(result.top_critical_nodes(10), n=10)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n  {display.colored('Results exported to:', display.Colors.GREEN)} {out_path}")

    print(f"\n  {display.colored('Done.', display.Colors.GREEN)} Models saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
