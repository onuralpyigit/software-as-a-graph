#!/usr/bin/env python3
"""
Graph Validation CLI

Compares static Analysis predictions against dynamic Simulation results.
Supports validating Application, Infrastructure, or Complete layers.
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.validation.pipeline import ValidationPipeline
from src.validation.metrics import ValidationTargets

# ANSI Colors
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"; YELLOW = "\033[93m"; RESET = "\033[0m"
BOLD = "\033[1m"

def print_result_table(result):
    res_dict = result.to_dict()
    
    print(f"\n{BOLD}Context: {res_dict['context']} {RESET} ({res_dict['timestamp']})")
    print(f"{BOLD}{'Group':<15} | {'Size':<6} | {'Stat':<4} | {'Metrics'}{RESET}")
    print("-" * 115)

    def _print_row(name, data):
        if data['n'] == 0:
            print(f"{name:<15} | N=0    | N/A")
            return

        color = GREEN if data['passed'] else RED
        status = "PASS" if data['passed'] else "FAIL"
        m = data['metrics']
        
        row = (
            f"{name:<15} | N={data['n']:<4} | {color}{status:<4}{RESET} | "
            f"Rho: {m['rho']:>5.3f} | "
            f"F1: {m['f1']:>5.3f} | "
            f"RMSE: {m['rmse']:>5.3f} | "
            f"Top5 Overlap: {m['top5_overlap']:>5.3f} | "
            f"Top10 Overlap: {m['top10_overlap']:>5.3f}"
        )
        print(row)
    
    # Overall
    _print_row("Overall", res_dict['overall'])
    
    # By Type
    for dtype, data in res_dict['by_type'].items():
        _print_row(f"  .{dtype}", data)
    print("-" * 115)

def main():
    parser = argparse.ArgumentParser(description="Software-as-a-Graph Validation")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    
    # Scope
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--layer", choices=["application", "infrastructure", "complete"], 
                        help="Validate specific layer")
    group.add_argument("--all", action="store_true", help="Validate ALL layers sequentially")
    
    parser.add_argument("--output", help="JSON output file (e.g. results/validation.json)")
    
    # Targets (Optional override)
    parser.add_argument("--target-spearman", type=float, default=0.70)
    parser.add_argument("--target-f1", type=float, default=0.80)

    args = parser.parse_args()
    
    # Default to 'complete' if nothing specified
    if not args.all and not args.layer:
        args.layer = "complete"

    targets = ValidationTargets(
        spearman=args.target_spearman,
        f1_score=args.target_f1
    )

    print(f"{CYAN}Initializing Validation Pipeline...{RESET}")
    print(f"Targets: Spearman >= {targets.spearman}, F1 >= {targets.f1_score}")
    
    full_report = {}
    
    try:
        with ValidationPipeline(args.uri, args.user, args.password) as pipeline:
            
            layers_to_run = []
            if args.all:
                layers_to_run = ["application", "infrastructure", "complete"]
            else:
                layers_to_run = [args.layer]
            
            for layer in layers_to_run:
                print(f"\n{YELLOW}>> Running Validation for Layer: {layer.upper()}{RESET}")
                
                result = pipeline.run(layer=layer, targets=targets)
                print_result_table(result)
                full_report[layer] = result.to_dict()
            
            # Export
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(full_report, f, indent=2)
                print(f"\n{GREEN}Full validation report saved to {args.output}{RESET}")
                
    except Exception as e:
        print(f"\n{RED}Validation Error: {e}{RESET}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())