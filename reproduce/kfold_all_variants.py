#!/usr/bin/env python3
"""
reproduce/kfold_all_variants.py — Per-domain k-fold × all variants
====================================================================

Runs cli/kfold_evaluate.py for each of the 5 model variants (topology_rmav, gl,
gl_qos, hgl, hgl_qos; see ALL_VARIANTS below) and aggregates results into a
unified JSON comparison table — the k-fold counterpart of
reproduce/loso_all_variants.py.

Unlike LOSO (cross-scenario, zero-shot transfer), k-fold validates in-domain
fit: for each scenario independently, repeated stratified k-fold over that
scenario's own labelled nodes. See cli/kfold_evaluate.py's module docstring
for the full protocol and rationale.

Usage
-----
  # Full sweep: 5 variants × 5 seeds × k=5 folds × all cached scenarios
  python reproduce/kfold_all_variants.py

  # Smoke test: 1 variant, 2 seeds, k=3
  python reproduce/kfold_all_variants.py --variants hgl_qos --seeds 42,123 --k 3

  # Resume (skips variants whose output dir already has results.json)
  python reproduce/kfold_all_variants.py --resume
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

ALL_VARIANTS = ["topology_rmav", "gl", "gl_qos", "hgl", "hgl_qos"]
DEFAULT_SEEDS = "42,123,456,789,2024"
DEFAULT_K     = 5
OUTPUT_BASE   = Path("output/kfold")
RESULTS_DIR   = Path("results")


# ── Subprocess runner ─────────────────────────────────────────────────────────

def _run_variant(
    variant: str,
    seeds: str,
    k: int,
    cache_dir: Path,
    epochs: int,
    extra_args: List[str],
    verbose: bool,
) -> Optional[Dict]:
    """Invoke kfold_evaluate.py for one variant and return its results dict."""
    out_dir = OUTPUT_BASE / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    cmd = [
        sys.executable, "-m", "cli.kfold_evaluate",
        "--cache-dir", str(cache_dir),
        "--output-dir", str(out_dir),
        "--variant", variant,
        "--seeds", seeds,
        "--k", str(k),
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
    "hgl_qos":         "HGL-QoS",
    "hgl":             "HGL",
    "gl_qos":          "GL-QoS",
    "gl":              "GL",
    "topology_rmav":   "RMAV baseline",
}


def _build_comparison_table(
    results_by_variant: Dict[str, Dict]
) -> Dict:
    """Merge per-variant k-fold results into a unified comparison table."""
    table = {}
    for variant, data in results_by_variant.items():
        if data is None:
            continue
        label = _VARIANT_LABELS.get(variant, variant)
        # Extract per-scenario metrics (each scenario's own in-domain k-fold fit)
        scenarios = data.get("scenarios", [])
        rho_vals = [s.get("mean_metrics", {}).get("spearman_rho", 0.0) for s in scenarios]
        f1_vals  = [s.get("mean_metrics", {}).get("f1_at_k", 0.0) for s in scenarios]

        import numpy as np
        n = len(rho_vals)
        table[variant] = {
            "label": label,
            "n_scenarios": n,
            "mean_rho": round(float(np.mean(rho_vals)), 4) if rho_vals else None,
            "std_rho":  round(float(np.std(rho_vals)), 4) if rho_vals else None,
            "mean_f1":  round(float(np.mean(f1_vals)), 4) if f1_vals else None,
            "per_scenario": [
                {
                    "scenario_id": s.get("scenario_id"),
                    "mean_rho": s.get("mean_metrics", {}).get("spearman_rho"),
                    "std_rho":  s.get("std_metrics", {}).get("spearman_rho"),
                }
                for s in scenarios
            ],
        }

    # Compute Δρ (hgl_qos vs best baseline)
    if "hgl_qos" in table:
        hq_rho = table["hgl_qos"].get("mean_rho", 0.0)
        baseline_rhos = [
            v.get("mean_rho", 0.0)
            for k, v in table.items()
            if k != "hgl_qos" and v.get("mean_rho") is not None
        ]
        best_baseline = max(baseline_rhos, default=0.0)
        table["hgl_qos"]["delta_vs_best_baseline"] = round(hq_rho - best_baseline, 4)

    return table


def _print_comparison_table(table: Dict):
    print("\n  ═══════════════════════════════════════════════════════════════")
    print(f"  Table: Per-Domain K-Fold In-Domain Evaluation — 5 Variants")
    print(f"  {'Variant':<25} {'Mean ρ':<10} {'Std ρ':<10} {'F1@K':<8} {'Δρ vs best BL'}")
    print("  " + "─" * 65)
    order = ["hgl_qos", "hgl", "gl_qos", "gl", "topology_rmav"]
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
    p = argparse.ArgumentParser(description="Per-domain k-fold × all variants.")
    p.add_argument("--variants", nargs="+", default=None, choices=ALL_VARIANTS,
                   help="Variants to run (default: all 5)")
    p.add_argument("--seeds", default=DEFAULT_SEEDS,
                   help="Comma-separated seeds (default: 5 seeds)")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help="Number of folds per scenario (default: 5)")
    p.add_argument("--cache-dir", type=Path, default=Path("output/loso_cache"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "kfold_all_variants.json")
    p.add_argument("--resume", action="store_true",
                   help="Skip variants with existing results.json")
    p.add_argument("--table-only", action="store_true",
                   help="Load existing results and print table only (no training)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    variants = args.variants or ALL_VARIANTS

    print(f"\n  K-Fold All-Variants Sweep")
    print(f"  Variants  : {variants}")
    print(f"  Seeds     : {args.seeds}")
    print(f"  k         : {args.k}")
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
                variant=var, seeds=args.seeds, k=args.k,
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
