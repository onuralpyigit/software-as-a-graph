#!/usr/bin/env python3
"""
bin/detect_antipatterns.py — Pub-Sub Architectural Anti-Pattern Detector
========================================================================
Detects bad smells from GNN predictions and structural metrics.
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
        description="Pub-Sub Anti-Pattern & Bad Smell Detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)

    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    analysis = client.analyze(layer=args.layer)
    prediction = client.predict(analysis)
    problems = client.detect_antipatterns(prediction)
    
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump([p.to_dict() for p in problems], f, indent=2)
        if not getattr(args, "quiet", False):
            print(f"Detected problems saved to {args.output}")
    elif not getattr(args, "quiet", False):
        print(f"Detected {len(problems)} architectural problems/smells.")

if __name__ == "__main__":
    main()
