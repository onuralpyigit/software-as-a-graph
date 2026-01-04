#!/usr/bin/env python3
"""
Validate Graph CLI

Runs the validation pipeline comparing Analysis (Predicted) vs Simulation (Actual).
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.validation.pipeline import ValidationPipeline
from src.validation.metrics import ValidationTargets

# Colors
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; CYAN = "\033[96m"; RESET = "\033[0m"

def print_group_result(name, res, indent=2):
    sp = " " * indent
    status_color = GREEN if res["passed"] else RED
    print(f"{sp}{name:<15} | n={res['n']:<4} | Rho={res['spearman']:.4f} | F1={res['f1']:.4f} | {status_color}{'PASS' if res['passed'] else 'FAIL'}{RESET}")

def main():
    parser = argparse.ArgumentParser(description="Graph Validation CLI")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--output", help="Save JSON results")
    args = parser.parse_args()
    
    targets = ValidationTargets(spearman=0.7, f1_score=0.7)
    
    try:
        with ValidationPipeline(uri=args.uri, user=args.user, password=args.password) as pipeline:
            print(f"{CYAN}Running Validation Pipeline...{RESET}")
            result = pipeline.run(targets)
            
            print(f"\n{CYAN}=== Validation Results ==={RESET}")
            print(f"Timestamp: {result.timestamp}")
            
            print(f"\n{CYAN}Overall:{RESET}")
            print_group_result("All Nodes", result.to_dict()["overall"])
            
            print(f"\n{CYAN}By Component Type:{RESET}")
            for k, v in result.to_dict()["by_type"].items():
                print_group_result(k, v)
                
            print(f"\n{CYAN}By Layer:{RESET}")
            for k, v in result.to_dict()["by_layer"].items():
                print_group_result(k, v)
            
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                print(f"\nResults saved to {args.output}")

    except Exception as e:
        print(f"{RED}Error:{RESET} {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())