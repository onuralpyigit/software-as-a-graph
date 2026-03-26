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
    
    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    display.print_header(f"GNN Prediction: {args.layer.upper()} Layer")
    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    # Predict relies on structural analysis
    display.print_step("Running structural analysis...")
    analysis = client.analyze(layer=args.layer, equal_weights=args.equal_weights)
    
    display.print_step("Running GNN inference...")
    result = client.predict(analysis, equal_weights=args.equal_weights, ahp_shrinkage=args.ahp_shrinkage)
    
    display.display_prediction_summary(result)
    
    if args.output:
        result.save(args.output)
        display.print_success(f"Prediction saved to {args.output}")
    else:
        display.print_success("Prediction completed successfully.")

if __name__ == "__main__":
    main()
