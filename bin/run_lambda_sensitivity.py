#!/usr/bin/env python3
"""
bin/run_lambda_sensitivity.py — λ Sensitivity Sweep
===================================================
Empirically validates the AHP shrinkage factor λ by measuring 
Spearman ρ and F1-score across λ ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}.
"""

import sys
from pathlib import Path

# Provide resolving so `saag` can be accessed natively
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
from saag import Client

def run_sweep(uri, user, password, layer="system", scale="small"):
    client = Client(neo4j_uri=uri, user=user, password=password)
    
    print(f"--- λ Sensitivity Study (Layer: {layer}, Scale: {scale}) ---")
    
    # 1. Ensure we have a graph and run simulation ground truth once
    # We use a fixed seed for reproducibility of the study
    seed = 42
    print(f"Generating and importing '{scale}' graph (seed={seed})...")
    from tools.generation import GenerationService
    gen = GenerationService(scale=scale, seed=seed)
    data = gen.generate()
    client.import_topology(graph_data=data, clear=True)
    
    print("Running exhaustive simulation to establish ground truth I(v)...")
    sim_results = client.simulate(layer=layer, mode="exhaustive")
    
    # Extract actual impact from simulation results
    actual_scores = {}
    for r in sim_results:
        # sim_results are SimulationResult objects
        actual_scores[r.target_id] = r.impact.composite_impact
    
    # Get component types for classification
    analysis = client.analyze(layer=layer)
    component_types = {cid: c.type for cid, c in analysis.raw.components.items()}
    
    # 2. Sweep λ
    lambdas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    
    print(f"\n{'λ':>5} | {'Spearman ρ':>12} | {'F1-Score':>10} | {'Top-5 Overlap':>14}")
    print("-" * 55)
    
    for l in lambdas:
        # Run prediction with this λ
        # client.predict currently returns PredictionResult (facade)
        pred_result = client.predict(analysis, ahp_shrinkage=l)
        
        # Extract predicted scores
        pred_scores = {c.id: c.rmav_score for c in pred_result.all_components}
        
        # Validate using internal validator
        from src.validation.validator import Validator
        validator = Validator()
        val_out = validator.validate(
            predicted_scores=pred_scores,
            actual_scores=actual_scores,
            component_types=component_types,
            layer=layer
        )
        
        rho = val_out.overall.correlation.spearman
        f1 = val_out.overall.classification.f1_score
        top5 = val_out.overall.ranking.top_5_overlap
        
        results.append((l, rho, f1, top5))
        print(f"{l:5.2f} | {rho:12.4f} | {f1:10.4f} | {top5:14.4f}")
    
    print("-" * 55)
    
    # Find optimal λ
    best_rho = max(results, key=lambda x: x[1])
    print(f"\nOptimal λ by Spearman ρ: {best_rho[0]:.2f} (ρ = {best_rho[1]:.4f})")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run λ sensitivity sweep.")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--layer", default="system")
    parser.add_argument("--scale", default="small")
    
    args = parser.parse_args()
    
    # Disable most logging to keep output clean
    logging.getLogger().setLevel(logging.ERROR)
    
    run_sweep(args.uri, args.user, args.password, layer=args.layer, scale=args.scale)

if __name__ == "__main__":
    main()
