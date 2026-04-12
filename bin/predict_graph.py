#!/usr/bin/env python3
"""
bin/predict_graph.py — Run GNN inference on a new graph
======================================================
Predicts component and relationship criticality for a system topology.
"""

import sys
from pathlib import Path

# Provide resolving so `saag` and `bin._shared` can be accessed natively
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from saag import Client
from bin._shared import add_neo4j_args, add_common_args, setup_logging
from bin.common.console import ConsoleDisplay

def main():
    display = ConsoleDisplay()
    parser = argparse.ArgumentParser(
        description="GNN inference — predict criticality on a new system graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--equal-weights", action="store_true", help="Use equal 0.25 weights for all Q(v) dimensions (baseline)")
    parser.add_argument("--ahp-shrinkage", type=float, default=0.7, help="Shrinkage factor λ for AHP weights [0, 1] (default: 0.7)")
    parser.add_argument("--detect-problems", action="store_true", default=True, help="Run anti-pattern detector on prediction results (default: True)")
    parser.add_argument("--no-detect-problems", action="store_false", dest="detect_problems", help="Disable anti-pattern detection")
    
    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)
    
    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    layers = [args.layer]
    if args.layer.lower() == "all":
        layers = ["app", "infra", "mw", "system"]
    elif "," in args.layer:
        layers = [l.strip() for l in args.layer.split(",") if l.strip()]

    for layer in layers:
        display.print_header(f"GNN Prediction: {layer.upper()} Layer")
        
        # Predict relies on structural analysis
        display.print_step(f"Running structural analysis for {layer}...")
        client.analyze(layer=layer, equal_weights=args.equal_weights)
        
        display.print_step(f"Running GNN inference for {layer}...")
        result = client.predict(layer=layer, equal_weights=args.equal_weights, ahp_shrinkage=args.ahp_shrinkage)
        
        display.display_prediction_summary(result)
        
        if args.detect_problems:
            display.print_step(f"Scanning {layer} for architectural anti-patterns...")
            problems = client.detect_antipatterns(result)
            total_components = len(result.all_components)
            display.display_antipatterns(problems, [layer], total_components)

        if args.output:
            out_path = args.output
            if len(layers) > 1:
                base, ext = out_path.rsplit('.', 1) if '.' in out_path else (out_path, 'json')
                out_path = f"{base}_{layer}.{ext}"
            result.save(out_path)
            display.print_success(f"Prediction saved to {out_path}")
        else:
            display.print_success(f"Prediction for {layer} completed successfully.")

if __name__ == "__main__":
    main()
