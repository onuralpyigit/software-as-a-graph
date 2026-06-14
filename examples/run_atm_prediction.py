#!/usr/bin/env python3
"""
Worked Example: Inductive GNN and Ensemble Prediction on the Air Traffic Management (ATM) Dataset.
Adheres strictly to the structural and functional specifications of Step 3 Predict.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag import Client
from saag.infrastructure.memory_repo import MemoryRepository
from saag.infrastructure.neo4j_repo import Neo4jRepository


def print_table(title, headers, rows):
    """Utility to print a clean ASCII table."""
    print(f"\n=== {title} ===")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
            
    header_str = " | ".join(f"{str(h).ljust(col_widths[i])}" for i, h in enumerate(headers))
    print(header_str)
    print("-" * (sum(col_widths) + len(headers) * 3 - 1))
    
    for row in rows:
        row_str = " | ".join(f"{str(val).ljust(col_widths[i])}" for i, val in enumerate(row))
        print(row_str)


def run_atm_prediction(args):
    # 1. Resolve path to atm_system.json and checkpoints
    script_dir = Path(__file__).resolve().parent
    json_path = script_dir.parent / "data" / "scenarios" / "atm_system.json"
    
    if not json_path.exists():
        print(f"Error: ATM System JSON not found at {json_path}")
        sys.exit(1)
        
    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        print(f"Error: GNN model checkpoints directory not found at {checkpoint_dir}")
        sys.exit(1)
        
    print(f"Loading safety-critical ATM topology JSON from: {json_path}")
    with open(json_path, "r") as f:
        topology_data = json.load(f)

    # 2. Select and initialize repository
    if args.neo4j:
        print(f"Initializing Neo4jRepository (URI: {args.uri})...")
        repo = Neo4jRepository(uri=args.uri, user=args.user, password=args.password)
    else:
        print("Initializing MemoryRepository...")
        repo = MemoryRepository()

    try:
        # 3. Save graph and derive logical dependencies
        print("Importing ATM topology and deriving dependencies...")
        repo.save_graph(topology_data, clear=True)
        repo.derive_dependencies()
        
        # 4. Create SDK client
        client = Client(repo=repo)
        
        # 5. Run structural analysis on the 'app' layer
        # Since the checkpoint was trained on the 'app' layer, we must target 'app' to avoid layer mismatch.
        print("Executing structural analysis on the 'app' layer to build input features...")
        analysis_result = client.analyze(layer="app")

        # 6. Execute GNN and ensemble inference
        print(f"Running GNN inference using checkpoint directory: {checkpoint_dir}")
        print(f"Prediction mode: {args.mode}")
        prediction = client.predict(
            analysis_result,
            mode=args.mode,
            gnn_checkpoint=str(checkpoint_dir)
        )

        # 7. Extract and display GNN/ensemble node scores
        components = prediction.all_components
        print(f"Successfully generated criticality predictions for {len(components)} components.")
        
        # Print ensemble alpha if available
        raw_inner = prediction.raw
        if hasattr(raw_inner, "ensemble_alpha") and raw_inner.ensemble_alpha:
            alpha_str = " | ".join(f"{val:.3f}" for val in raw_inner.ensemble_alpha)
            print(f"Ensemble blend coefficients (\u03b1) [Q | R | M | A | S]: {alpha_str}")

        comp_rows = []
        # Sort components by composite overall score descending, show top 10
        top_comps = components[:10]
        for comp in top_comps:
            scores = comp.scores
            comp_rows.append([
                comp.id,
                comp.name,
                comp.type,
                f"{comp.rmav_score:.4f}",  # In PredictionResult, .rmav_score is the predicted composite/overall score
                f"{scores.get('reliability', 0.0):.4f}",
                f"{scores.get('maintainability', 0.0):.4f}",
                f"{scores.get('availability', 0.0):.4f}",
                f"{scores.get('security', 0.0):.4f}",
                comp.criticality_level
            ])
            
        print_table(
            f"ATM Component Criticality Ranks (Top 10 - Mode: {args.mode})",
            ["ID", "Name", "Type", "Composite (Q)", "Reliability (R)", "Maintainability (M)", "Availability (A)", "Security (S)", "Level"],
            comp_rows
        )

        # 8. Extract and display GNN edge scores
        edges = prediction.edges
        print(f"\nSuccessfully generated criticality predictions for {len(edges)} directed dependencies.")
        
        edge_rows = []
        # Sort edges by overall score descending, show top 10
        sorted_edges = sorted(edges, key=lambda e: e.scores.overall, reverse=True)[:10]
        for edge in sorted_edges:
            edge_rows.append([
                edge.source,
                edge.target,
                edge.dependency_type,
                f"{edge.scores.overall:.4f}",
                f"{edge.scores.reliability:.4f}",
                f"{edge.scores.maintainability:.4f}",
                f"{edge.scores.availability:.4f}",
                f"{edge.scores.security:.4f}",
                edge.level.name if hasattr(edge.level, 'name') else str(edge.level)
            ])
            
        print_table(
            "ATM Edge Criticality Ranks (Top 10)",
            ["Source", "Target", "Relation Type", "Composite (Q)", "Reliability (R)", "Maintainability (M)", "Availability (A)", "Security (S)", "Level"],
            edge_rows
        )

        # 9. Verification assertions
        print("\nVerifying GNN Prediction results:")
        assert len(components) > 0, "No components returned in prediction result"
        assert len(edges) > 0, "No edges returned in prediction result"
        
        # Verify that all scores are normalized in [0, 1]
        for comp in components:
            scores = comp.scores
            assert 0.0 <= comp.rmav_score <= 1.0, f"Component score out of bounds: {comp.rmav_score}"
            for dim, score in scores.items():
                assert 0.0 <= score <= 1.0, f"Component dimension {dim} score out of bounds: {score}"
                
        for edge in edges:
            assert 0.0 <= edge.scores.overall <= 1.0, f"Edge overall score out of bounds: {edge.scores.overall}"
            assert 0.0 <= edge.scores.reliability <= 1.0, f"Edge reliability score out of bounds: {edge.scores.reliability}"
            assert 0.0 <= edge.scores.maintainability <= 1.0, f"Edge maintainability score out of bounds: {edge.scores.maintainability}"
            assert 0.0 <= edge.scores.availability <= 1.0, f"Edge availability score out of bounds: {edge.scores.availability}"
            assert 0.0 <= edge.scores.security <= 1.0, f"Edge security score out of bounds: {edge.scores.security}"

        print("  [PASS] Component and edge counts, and all score boundary checks validated successfully.")

        # 10. Save results to output JSON
        output_path = Path(args.output)
        prediction.save(str(output_path))
        print(f"\nSaved predictions report JSON to: {output_path}")
        print("\nGNN prediction pipeline executed and verified successfully on ATM dataset!")

    finally:
        repo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ATM GNN Prediction worked example")
    parser.add_argument("--neo4j", action="store_true", help="Run against a live Neo4j instance instead of in-memory")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--checkpoint", default="models/gnn_checkpoints", help="Path to GNN checkpoint directory")
    parser.add_argument("--mode", default="ensemble", choices=["ensemble", "gnn"], help="Prediction mode: 'ensemble' (GNN + RMAV) or 'gnn' (raw GNN)")
    parser.add_argument("--output", default="output/atm_system_predictions.json", help="Path to save prediction results JSON")
    
    args = parser.parse_args()
    run_atm_prediction(args)
