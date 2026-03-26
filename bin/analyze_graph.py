#!/usr/bin/env python3
"""
Graph Analysis CLI

Multi-layer graph analysis for distributed pub-sub systems.
Applies graph topology analysis to predict critical components using
DEPENDS_ON relationships derived from the system model.
"""
import sys
from pathlib import Path

# Provide resolving so `saag` and `bin._shared` can be accessed natively if script run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from saag import Client
from bin._shared import add_neo4j_args, add_common_args, setup_logging

def main():
    parser = argparse.ArgumentParser(description="Multi-layer graph analysis for distributed pub-sub systems.")
    
    # Specific args
    parser.add_argument("--use-ahp", action="store_true", help="Use AHP-derived weights instead of default fixed weights")
    parser.add_argument("--equal-weights", action="store_true", help="Use equal 0.25 weights for all Q(v) dimensions (baseline)")
    
    add_neo4j_args(parser)
    add_common_args(parser)
    
    args = parser.parse_args()
    setup_logging(args)

    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    result = client.analyze(layer=args.layer, use_ahp=args.use_ahp, equal_weights=args.equal_weights)
    
    if args.output:
        result.save(args.output)
        if not args.quiet:
            print(f"Analysis saved to {args.output}")
    elif not args.quiet:
        print("Analysis completed successfully. (No output file specified)")

if __name__ == "__main__":
    main()