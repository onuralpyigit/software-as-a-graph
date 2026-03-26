"""
Step 6 — GNN Criticality Prediction
=====================================
Demonstrates how to train and use the Heterogeneous Graph Neural Network (GNN)
to predict component criticality.

The GNN uses:
  • Structural features (from AnalysisService)
  • RMAV baseline scores (from AnalysisService)
  • Simulation ground truth (from SimulationService)

It trains a model to predict the I(v) ground truth, and outputs both pure GNN
predictions and an Ensemble (GNN + RMAV) score.

Prerequisites:
  • Neo4j running with imported data (run examples/example_import.py first)

Run from the project root:
    python examples/example_prediction.py
"""
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.adapters import create_repository
from src.analysis import AnalysisService
from src.simulation import SimulationService
from src.prediction import (
    GNNService,
    extract_structural_metrics_dict,
    extract_rmav_scores_dict,
    extract_simulation_dict,
)


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    try:
        repo = create_repository()
    except Exception as e:
        print(f"[ERROR] Could not connect to Neo4j: {e}")
        print("  Ensure Neo4j is running and import has been done.")
        return

    # Temporary directory for GNN checkpoints
    checkpoint_dir = ROOT / "output" / "example_gnn_checkpoint"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1. Preparation ─────────────────────────────────────────────
        print_section("Preparation: Fetching features and ground truth")
        analysis = AnalysisService(repo)
        simulation = SimulationService(repo)

        layer = "app"
        print(f"  Running analysis for layer: {layer}")
        analysis_result = analysis.analyze_layer(layer)
        nx_graph = analysis_result.graph

        print("  Running exhaustive failure simulation for ground truth...")
        sim_results = simulation.run_failure_simulation_exhaustive(layer=layer)

        # Convert to dictionary formats expected by the GNN service
        structural_dict = extract_structural_metrics_dict(analysis_result.structural)
        rmav_dict = extract_rmav_scores_dict(analysis_result.quality)
        simulation_dict = extract_simulation_dict(sim_results)

        print(f"  Extracted features for {len(structural_dict)} components.")

        # ── 2. Training the GNN ────────────────────────────────────────
        print_section("Training the Heterogeneous GNN")
        gnn_service = GNNService(
            hidden_channels=32,
            num_heads=2,
            num_layers=2,
            dropout=0.1,
            predict_edges=True,
            checkpoint_dir=str(checkpoint_dir),
        )

        epochs = 15
        print(f"  Training for {epochs} epochs...")
        train_result = gnn_service.train(
            graph=nx_graph,
            structural_metrics=structural_dict,
            simulation_results=simulation_dict,
            rmav_scores=rmav_dict,
            num_epochs=epochs,
            lr=1e-3,
        )

        print("\n  Training complete. Validation metrics:")
        if train_result.gnn_metrics:
            metrics = train_result.gnn_metrics
            print(f"    RMSE           : {metrics.rmse:.4f}")
            print(f"    Spearman ρ     : {metrics.spearman_rho:.4f}")
            print(f"    F1 Score       : {metrics.f1_score:.4f}")
            print(f"    Top-5 Overlap  : {metrics.top_5_overlap:.4f}")

        # Saving model
        gnn_service.save()
        print(f"  Model saved to: {checkpoint_dir}")

        # ── 3. Predictions ─────────────────────────────────────────────
        print_section("Ensemble Predictions (Top Components)")
        top_nodes = train_result.top_critical_nodes(10)

        header = f"  {'Rank':<5} {'ID':<30} {'Score':>7} {'Level':<10}"
        print(header)
        print(f"  {'-'*60}")
        for i, node in enumerate(top_nodes, 1):
            print(
                f"  {i:<5} {node.component[:29]:<30}"
                f" {node.composite_score:>7.4f}  {node.criticality_level}"
            )

        print_section("Ensemble Predictions (Top Edges)")
        if train_result.edge_scores:
            top_edges = train_result.top_critical_edges(5)
            header_e = f"  {'Source':<20} → {'Target':<20} {'Score':>7} {'Level'}"
            print(header_e)
            print(f"  {'-'*65}")
            for edge in top_edges:
                print(
                    f"  {edge.source_node[:19]:<20} "
                    f"→ {edge.target_node[:19]:<20}"
                    f" {edge.composite_score:>7.4f}  {edge.criticality_level}"
                )
        else:
            print("  (Edge criticality predictions disabled or unavailable)")

    finally:
        repo.close()

    print()
    print_section("Done")
    print("  Next step: run examples/example_visualization.py")


if __name__ == "__main__":
    main()
