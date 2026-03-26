#!/usr/bin/env python3
"""
bin/simulate_graph.py — Failure Simulation CLI
==============================================
Simulates failure cascades.
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
        description="Graph Failure Simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--mode", default="exhaustive", 
        choices=["exhaustive", "monte_carlo", "single", "pairwise"],
        help="Simulation execution mode"
    )
    
    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    display.print_header(f"Failure Simulation: {args.layer.upper()} Layer")
    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    # Simulate
    display.print_step(f"Running {args.mode} simulation...")
    report = client.simulate(layer=args.layer, mode=args.mode)
    
    display.display_simulation_summary(report)
    
    if args.output:
        import json
        with open(args.output, "w") as f:
            if hasattr(report, "to_dict"):
                json.dump(report.to_dict(), f, indent=2, default=str)
            else:
                from dataclasses import asdict
                json.dump(asdict(report), f, indent=2, default=str)
        display.print_success(f"Simulation report saved to {args.output}")
    else:
        display.print_success("Simulation executed successfully.")

if __name__ == "__main__":
    main()