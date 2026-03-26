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

def main():
    parser = argparse.ArgumentParser(
        description="GNN inference — predict criticality on a new system graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--equal-weights", action="store_true", help="Use equal 0.25 weights for all Q(v) dimensions (baseline)")
    
    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)

    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    # Predict relies on structural analysis
    analysis = client.analyze(layer=args.layer, equal_weights=args.equal_weights)
    result = client.predict(analysis, equal_weights=args.equal_weights)
    
    if args.output:
        result.save(args.output)
        if not getattr(args, "quiet", False):
            print(f"Prediction saved to {args.output}")
    elif not getattr(args, "quiet", False):
        print("Prediction completed successfully. (No output file specified)")

if __name__ == "__main__":
    main()
