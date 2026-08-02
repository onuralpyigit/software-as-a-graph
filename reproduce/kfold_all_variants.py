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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Structural baselines first: training-free, so their score is the bar a
# learned variant must clear to justify being trained at all.
ALL_VARIANTS = ["topo_baseline", "topo_qos", "topology_rmav", "gl", "gl_qos", "hgl", "hgl_qos"]
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

    # Clear stale state before running. GNNService restores from any checkpoint
    # it finds, so a leftover workspace makes the run skip training entirely and
    # score a model fitted for a different configuration (see the same guard in
    # loso_all_variants.py, where it flipped hgl from -0.576 to +0.594).
    workspace = out_dir / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    if results_path.exists():
        results_path.unlink()

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


def _confidence_interval(values: List[float], confidence: float = 0.95):
    """t-distribution CI, appropriate for the small (scenario-count-sized)
    samples this table works with — a normal-approximation CI understates
    width at n < ~10. Degenerates gracefully: n=0 -> (None, None), n=1 ->
    (value, value) since a single observation has no dispersion to bound.
    """
    n = len(values)
    if n == 0:
        return (None, None)
    if n == 1:
        return (round(float(values[0]), 4), round(float(values[0]), 4))

    import numpy as np
    from scipy.stats import t as t_dist

    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1)) / (n ** 0.5)
    if sem < 1e-12:
        return (round(mean, 4), round(mean, 4))
    half_width = t_dist.ppf((1 + confidence) / 2, df=n - 1) * sem
    return (round(mean - half_width, 4), round(mean + half_width, 4))


def _paired_deltas(
    a_by_scenario: Dict[str, float], b_by_scenario: Dict[str, float]
) -> Optional[Dict]:
    """Per-scenario (a - b) deltas, matched by scenario id, plus their mean and CI.

    Comparing two variants' pooled means treats each variant's scenario
    spread as independent, when in practice a scenario that is hard for one
    variant tends to be hard for all of them — the pooled comparison can
    show overlapping spreads even when the SAME-scenario delta is
    consistently one-directional. Pairing by scenario removes that
    confound. Only scenarios present in both are compared.
    """
    common = sorted(set(a_by_scenario) & set(b_by_scenario))
    if not common:
        return None
    deltas = [a_by_scenario[s] - b_by_scenario[s] for s in common]
    lo, hi = _confidence_interval(deltas)
    return {
        "n_paired_scenarios": len(common),
        "mean_delta": round(sum(deltas) / len(deltas), 4),
        "ci_95": [lo, hi],
        "per_scenario_delta": {s: round(d, 4) for s, d in zip(common, deltas)},
    }


def _build_comparison_table(
    results_by_variant: Dict[str, Dict]
) -> Dict:
    """Merge per-variant k-fold results into a unified comparison table."""
    table = {}
    rho_by_scenario_by_variant: Dict[str, Dict[str, float]] = {}
    for variant, data in results_by_variant.items():
        if data is None:
            continue
        label = _VARIANT_LABELS.get(variant, variant)
        # Extract per-scenario metrics (each scenario's own in-domain k-fold fit)
        scenarios = data.get("scenarios", [])
        rho_vals = [s.get("mean_metrics", {}).get("spearman_rho", 0.0) for s in scenarios]
        f1_vals  = [s.get("mean_metrics", {}).get("f1_at_k", 0.0) for s in scenarios]
        rho_by_scenario_by_variant[variant] = {
            s.get("scenario_id"): s.get("mean_metrics", {}).get("spearman_rho", 0.0)
            for s in scenarios
        }

        import numpy as np
        n = len(rho_vals)
        ci_lo, ci_hi = _confidence_interval(rho_vals)
        table[variant] = {
            "label": label,
            "n_scenarios": n,
            "mean_rho": round(float(np.mean(rho_vals)), 4) if rho_vals else None,
            "std_rho":  round(float(np.std(rho_vals)), 4) if rho_vals else None,
            "ci95_rho": [ci_lo, ci_hi],
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

    # Paired per-scenario delta (hgl_qos vs each baseline), replacing the
    # single difference-of-pooled-means figure: pairing by scenario controls
    # for scenario difficulty instead of comparing two independent spreads.
    if "hgl_qos" in table:
        hq_by_scenario = rho_by_scenario_by_variant.get("hgl_qos", {})
        deltas_vs_baseline = {}
        for variant in table:
            if variant == "hgl_qos":
                continue
            paired = _paired_deltas(hq_by_scenario, rho_by_scenario_by_variant.get(variant, {}))
            if paired is not None:
                deltas_vs_baseline[variant] = paired
        table["hgl_qos"]["paired_delta_vs_baseline"] = deltas_vs_baseline

        best_baseline_variant = max(
            deltas_vs_baseline,
            key=lambda v: table[v].get("mean_rho") or float("-inf"),
            default=None,
        )
        if best_baseline_variant is not None:
            table["hgl_qos"]["delta_vs_best_baseline"] = (
                deltas_vs_baseline[best_baseline_variant]["mean_delta"]
            )

    return table


def _print_comparison_table(table: Dict):
    print("\n  ═══════════════════════════════════════════════════════════════")
    print(f"  Table: Per-Domain K-Fold In-Domain Evaluation — 5 Variants")
    print(f"  {'Variant':<25} {'Mean ρ':<10} {'Std ρ':<10} {'F1@K':<8} {'Δρ vs best BL'}")
    print("  " + "─" * 65)
    order = [v for v in ALL_VARIANTS if v in table]
    for variant in order:
        if variant not in table:
            continue
        r = table[variant]
        rho  = f"{r.get('mean_rho', 0):.4f}" if r.get('mean_rho') is not None else "—"
        std  = f"{r.get('std_rho', 0):.4f}" if r.get('std_rho') is not None else "—"
        f1   = f"{r.get('mean_f1', 0):.4f}" if r.get('mean_f1') is not None else "—"
        delta = r.get("delta_vs_best_baseline")
        delta_s = f"{delta:+.4f}" if delta is not None else "—"
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
