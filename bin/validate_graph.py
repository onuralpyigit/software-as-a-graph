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
from bin.common.console import ConsoleDisplay

def main():
    display = ConsoleDisplay()
    parser = argparse.ArgumentParser(
        description="Validate predictions vs ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    display.print_header("Validation Pipeline")
    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    # Validation uses a list of layers. 
    layers = [l.strip() for l in args.layer.split(",") if l.strip()]
    
    display.print_step(f"Validating layers: {', '.join(layers)}")
    result = client.validate(layers=layers)
    
    display.display_validation_summary(result)
    
    if args.output:
        result.save(args.output)
        display.print_success(f"Validation report saved to {args.output}")
    else:
        display.print_success("Validation pipeline executed successfully.")

if __name__ == "__main__":
    main()
