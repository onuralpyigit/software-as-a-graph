#!/usr/bin/env python3
"""
Multi-Seed Stability Summary

Aggregates multiple validation JSON results to compute mean and standard
deviation of Spearman correlation, verifying statistical stability.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

def compute_stats(values: List[float]):
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = math.sqrt(variance)
    return mean, std_dev

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed validation results and compute Spearman stability statistics.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="RESULT_JSON",
        help="One or more validation JSON result files to aggregate.",
    )
    args = parser.parse_args()
    files = args.files
    spearmans = []
    
    print(f"Aggregating {len(files)} result files...")
    
    for f_path in files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                # Handle both PipelineResult and LayerValidationResult
                if "layers" in data:
                    # Use system layer if available, otherwise first layer
                    layer = data["layers"].get("system") or next(iter(data["layers"].values()))
                    spearmans.append(layer["spearman"])
                else:
                    spearmans.append(data["spearman"])
        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    if not spearmans:
        print("No valid results found.")
        sys.exit(1)

    mean_rho, std_rho = compute_stats(spearmans)
    
    print("\n" + "="*50)
    print("           STABILITY PROTOCOL SUMMARY")
    print("="*50)
    print(f"  Seeds analyzed:   {len(spearmans)}")
    print(f"  Mean Spearman \u03bc:  {mean_rho:.4f}")
    print(f"  Std Dev \u03c3:       {std_rho:.4f}")
    print("-"*50)
    
    # Stability Criterion: \u03bc_\u03c1 \u2265 0.80 and \u03c3_\u03c1 \u2264 0.05
    pass_mean = mean_rho >= 0.80
    pass_std = std_rho <= 0.05
    
    status = "PASSED" if pass_mean and pass_std else "FAILED"
    print(f"  Stability Verdict: {status}")
    
    if not pass_mean:
        print(f"  [!] Mean \u03bc below 0.80 target.")
    if not pass_std:
        print(f"  [!] Std Dev \u03c3 above 0.05 target (high variance).")
    
    print("="*50)

if __name__ == "__main__":
    main()
