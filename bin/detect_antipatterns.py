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
from bin.common.console import ConsoleDisplay

def main():
    parser = argparse.ArgumentParser(
        description="Pub-Sub Anti-Pattern & Bad Smell Detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)

    display = ConsoleDisplay()
    display.print_header("Architectural Anti-Pattern Detection")
    
    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    display.print_step(f"Analyzing layer '{args.layer}' for bad smells...")
    analysis = client.analyze(layer=args.layer)
    
    display.print_step("Generating criticality predictions...")
    prediction = client.predict(analysis)
    
    display.print_step("Scanning for structural and probabilistic anti-patterns...")
    problems = client.detect_antipatterns(prediction)
    
    # Report results
    total_components = len(analysis.raw.components)
    display.display_antipatterns(problems, [args.layer], total_components)
    
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([p.to_dict() for p in problems], f, indent=2)
        display.print_success(f"Detailed anti-pattern report saved to {args.output}")
    else:
        display.print_success("Anti-pattern detection complete.")

if __name__ == "__main__":
    main()
