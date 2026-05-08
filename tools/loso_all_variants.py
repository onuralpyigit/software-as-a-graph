#!/usr/bin/env python3
"""
tools/loso_all_variants.py — Block E: LOSO × 4 variants
=========================================================

Runs cli/loso_evaluate.py for each of the 4 model variants and
aggregates results into a unified JSON for Table 4 (paper §6.5).

Usage
-----
  # Full sweep: 4 variants × 5 seeds × all cached scenarios
  python tools/loso_all_variants.py

  # Smoke test: 1 variant, 2 seeds
  python tools/loso_all_variants.py --variants hetero_qos homo_unweighted --seeds 42 123

  # Resume (skips variants whose output dir already has results.json)
  python tools/loso_all_variants.py --resume
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALL_VARIANTS = ["topology_rmav", "homo_unweighted", "homo_scalar", "hetero_qos"]
DEFAULT_SEEDS = "42,123,456,789,2024"
OUTPUT_BASE   = Path("output/loso")
RESULTS_DIR   = Path("results")


# ── Subprocess runner ─────────────────────────────────────────────────────────

def _run_variant(
    variant: str,
    seeds: str,
    cache_dir: Path,
    epochs: int,
    extra_args: List[str],
    verbose: bool,
) -> Optional[Dict]:
    """Invoke loso_evaluate.py for one variant and return its results dict."""
    out_dir = OUTPUT_BASE / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    cmd = [
        sys.executable, "-m", "cli.loso_evaluate",
        "--cache-dir", str(cache_dir),
        "--output-dir", str(out_dir),
        "--variant", variant,
        "--seeds", seeds,
        "--epochs", str(epochs),
        *extra_args,
    ]
    if verbose:
        print(f"    CMD: {' '.join(cmd)}")

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),  # project root
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  ✗ {variant} FAILED ({elapsed:.1f}s)")
        if not verbose and proc.stderr:
            # Print last 10 lines of stderr for quick triage
            lines = proc.stderr.strip().splitlines()
            for line in lines[-10:]:
                print(f"    {line}")
        return None

    if not results_path.exists():
        print(f"  ✗ {variant}: results.json not found after run")
        return None

    print(f"  ✓ {variant} ({elapsed:.1f}s)")
    return json.loads(results_path.read_text())


# ── Comparison table ──────────────────────────────────────────────────────────

_VARIANT_LABELS = {
    "hetero_qos":      "Q-HGL (ours)",
    "homo_scalar":     "Homo-Scalar",
    "homo_unweighted": "Homo-Unweighted",
    "topology_rmav":   "RMAV baseline",
}


def _build_comparison_table(
    results_by_variant: Dict[str, Dict]
) -> Dict:
    """Merge per-variant LOSO results into a unified comparison table."""
    table = {}
    for variant, data in results_by_variant.items():
        if data is None:
            continue
        label = _VARIANT_LABELS.get(variant, variant)
        # Extract per-fold metrics
        folds = data.get("folds", [])
        rho_vals = [f.get("mean_metrics", {}).get("spearman_rho", 0.0) for f in folds]
        f1_vals  = [f.get("mean_metrics", {}).get("f1_at_k", 0.0) for f in folds]

        import numpy as np
        n = len(rho_vals)
        table[variant] = {
            "label": label,
            "n_folds": n,
            "mean_rho": round(float(np.mean(rho_vals)), 4) if rho_vals else None,
            "std_rho":  round(float(np.std(rho_vals)), 4) if rho_vals else None,
            "mean_f1":  round(float(np.mean(f1_vals)), 4) if f1_vals else None,
            "per_fold": [
                {
                    "holdout":  f.get("holdout_id"),
                    "mean_rho": f.get("mean_metrics", {}).get("spearman_rho"),
                    "std_rho":  f.get("std_metrics", {}).get("spearman_rho"),
                }
                for f in folds
            ],
        }

    # Compute Δρ (hetero_qos vs best baseline)
    if "hetero_qos" in table:
        hq_rho = table["hetero_qos"].get("mean_rho", 0.0)
        baseline_rhos = [
            v.get("mean_rho", 0.0)
            for k, v in table.items()
            if k != "hetero_qos" and v.get("mean_rho") is not None
        ]
        best_baseline = max(baseline_rhos, default=0.0)
        table["hetero_qos"]["delta_vs_best_baseline"] = round(hq_rho - best_baseline, 4)

    return table


def _print_comparison_table(table: Dict):
    print("\n  ═══════════════════════════════════════════════════════════════")
    print(f"  Table 4: LOSO Inductive Evaluation — 4 Variants")
    print(f"  {'Variant':<25} {'Mean ρ':<10} {'Std ρ':<10} {'F1@K':<8} {'Δρ vs best BL'}")
    print("  " + "─" * 65)
    order = ["hetero_qos", "homo_scalar", "homo_unweighted", "topology_rmav"]
    for variant in order:
        if variant not in table:
            continue
        r = table[variant]
        rho  = f"{r.get('mean_rho', 0):.4f}" if r.get('mean_rho') is not None else "—"
        std  = f"{r.get('std_rho', 0):.4f}" if r.get('std_rho') is not None else "—"
        f1   = f"{r.get('mean_f1', 0):.4f}" if r.get('mean_f1') is not None else "—"
        delta = r.get("delta_vs_best_baseline")
        delta_s = f"+{delta:.4f}" if delta is not None else "—"
        label = r.get("label", variant)
        print(f"  {label:<25} {rho:<10} {std:<10} {f1:<8} {delta_s}")
    print("  ═══════════════════════════════════════════════════════════════")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Block E: LOSO × all variants.")
    p.add_argument("--variants", nargs="+", default=None, choices=ALL_VARIANTS,
                   help="Variants to run (default: all 4)")
    p.add_argument("--seeds", default=DEFAULT_SEEDS,
                   help="Comma-separated seeds (default: 5 seeds)")
    p.add_argument("--cache-dir", type=Path, default=Path("output/loso_cache"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "loso_all_variants.json")
    p.add_argument("--resume", action="store_true",
                   help="Skip variants with existing results.json")
    p.add_argument("--table-only", action="store_true",
                   help="Load existing results and print table only (no training)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    variants = args.variants or ALL_VARIANTS

    print(f"\n  LOSO All-Variants Sweep (Block E)")
    print(f"  Variants  : {variants}")
    print(f"  Seeds     : {args.seeds}")
    print(f"  Cache dir : {args.cache_dir}")
    print()

    results_by_variant: Dict[str, Optional[Dict]] = {}

    if args.table_only:
        # Load existing per-variant results
        for var in variants:
            rp = OUTPUT_BASE / var / "results.json"
            if rp.exists():
                results_by_variant[var] = json.loads(rp.read_text())
                print(f"  Loaded: {var} ({rp})")
            else:
                print(f"  Missing: {var} ({rp})")
                results_by_variant[var] = None
    else:
        if not args.cache_dir.exists():
            print(f"Error: --cache-dir {args.cache_dir} does not exist.", file=sys.stderr)
            sys.exit(1)

        for var in variants:
            rp = OUTPUT_BASE / var / "results.json"
            if args.resume and rp.exists():
                print(f"  SKIP (resume): {var}")
                results_by_variant[var] = json.loads(rp.read_text())
                continue

            print(f"  Running {var} ...")
            data = _run_variant(
                variant=var, seeds=args.seeds,
                cache_dir=args.cache_dir, epochs=args.epochs,
                extra_args=[], verbose=args.verbose,
            )
            results_by_variant[var] = data

    # Build + save comparison table
    table = _build_comparison_table(results_by_variant)
    output = {
        "comparison_table": table,
        "per_variant_results": {k: v for k, v in results_by_variant.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {args.output}")

    _print_comparison_table(table)


if __name__ == "__main__":
    main()
