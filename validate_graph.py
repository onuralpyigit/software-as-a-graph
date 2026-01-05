#!/usr/bin/env python3
"""
Graph Validation CLI

Compares the static Analysis predictions against dynamic Simulation results
to validate the accuracy of the software graph model.

Updates:
- Output formatted to show Spearman, F1, Precision, Recall.
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.validation.pipeline import ValidationPipeline
from src.validation.metrics import ValidationTargets

# Colors
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"; RESET = "\033[0m"
BOLD = "\033[1m"

def print_result(result):
    res_dict = result.to_dict()
    
    print(f"\n{BOLD}=== Validation Report ==={RESET}")
    print(f"Timestamp: {res_dict['timestamp']}")
    
    print(f"\n{BOLD}Targets:{RESET}")
    for k, v in res_dict['targets'].items():
        print(f"  {k:<15}: {v}")

    def _print_row(name, data):
        color = GREEN if data['passed'] else RED
        status = "PASS" if data['passed'] else "FAIL"
        m = data['metrics']
        
        # Format string for metrics row
        row = (
            f"{name:<15} | N={data['n']:<4} | {color}{status:<4}{RESET} | "
            f"Rho: {m['rho']:>5.3f} | F1: {m['f1']:>5.3f} | "
            f"Prec: {m['precision']:>5.3f} | Rec: {m['recall']:>5.3f} | "
            f"Top5: {m['top5_overlap']:>5.3f} | Top10: {m['top10_overlap']:>5.3f}"
        )
        print(row)

    print(f"\n{BOLD}{'Group':<15} | {'Size':<6} | {'Stat':<4} | {'Metrics'}{RESET}")
    print("-" * 100)
    
    # Overall
    _print_row("Overall", res_dict['overall'])
    print("-" * 100)
    
    # By Type
    for dtype, data in res_dict['by_type'].items():
        _print_row(dtype, data)

def main():
    parser = argparse.ArgumentParser(description="Software-as-a-Graph Validation")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--output", help="JSON output file")
    
    # Custom Targets (Defaults from Table 5)
    parser.add_argument("--target-spearman", type=float, default=0.70)
    parser.add_argument("--target-f1", type=float, default=0.80)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--target-overlap", type=float, default=0.60)

    args = parser.parse_args()

    targets = ValidationTargets(
        spearman=args.target_spearman,
        f1_score=args.target_f1,
        precision=args.target_precision,
        recall=args.target_recall,
        top_5_overlap=args.target_overlap,
        top_10_overlap=args.target_overlap
    )

    try:
        with ValidationPipeline(args.uri, args.user, args.password) as pipeline:
            result = pipeline.run(targets)
            print_result(result)
            
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                print(f"\nResults saved to {args.output}")
                
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())