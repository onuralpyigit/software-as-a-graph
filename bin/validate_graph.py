#!/usr/bin/env python3
"""
bin/validate_graph.py — Validation Pipeline CLI
===============================================
Validates actual simulation results against predicted metrics.
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
        description="Validate predictions vs ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)

    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    # Validation uses a list of layers. 
    # 'layer' from arguments might be comma-separated strings if multiple layers requested
    layers = [l.strip() for l in args.layer.split(",") if l.strip()]
    
    result = client.validate(layers=layers)
    
    if args.output:
        result.save(args.output)
        if not getattr(args, "quiet", False):
            print(f"Validation report saved to {args.output}")
    elif not getattr(args, "quiet", False):
        print("Validation pipeline executed successfully.")

if __name__ == "__main__":
    main()
