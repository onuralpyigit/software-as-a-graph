#!/usr/bin/env python3
"""
reproduce/loso_all_variants.py — Block E: LOSO × 4 variants
=========================================================

Runs cli/loso_evaluate.py for each of the 4 model variants (gl, gl_qos, hgl, hgl_qos; see
ALL_VARIANTS below) and aggregates results into a unified JSON for the paper's LOSO table
(Leave-One-Scenario-Out Cross-Validation Results, docs/research/jss/draft.md Section 8.1).

Usage
-----
  # Full sweep: 4 variants × 5 seeds × all cached scenarios
  python reproduce/loso_all_variants.py

  # Smoke test: 1 variant, 2 seeds
  python reproduce/loso_all_variants.py --variants gl hgl --seeds 42,123

  # Resume (reuses a variant's results.json only when it is newer than the
  # cache and less than _RESUME_MAX_AGE_DAYS old; otherwise re-runs it)
  python reproduce/loso_all_variants.py --resume
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

from reproduce._provenance import stamp
from saag.evaluation import variant_registry as _registry

# Structural baselines first: they are training-free, so their LOSO score is
# their only score, and a learned variant has to beat them to be worth training.
ALL_VARIANTS = ["topo_baseline", "topo_qos", "topology_rm", "gl", "gl_qos", "hgl", "hgl_qos"]
DEFAULT_SEEDS = "42,123,456,789,2024"
OUTPUT_BASE   = Path("output/loso")
#: Beyond this, a --resume result is re-run rather than trusted. Model code
#: changes far more often than the cache does, and a code change leaves no
#: mtime on any input this can check.
_RESUME_MAX_AGE_DAYS = 2
RESULTS_DIR   = Path("results")


# ── Subprocess runner ─────────────────────────────────────────────────────────

def _run_variant(
    variant: str,
    seeds: str,
    cache_dir: Path,
    epochs: int,
    extra_args: List[str],
    verbose: bool,
    output_base: Path = OUTPUT_BASE,
) -> Optional[Dict]:
    """Invoke loso_evaluate.py for one variant and return its results dict."""
    out_dir = output_base / variant

    # Clear the per-variant workspace first. GNNService restores from any
    # checkpoint it finds under the fold's checkpoint dir, so a leftover
    # workspace from an earlier run makes the next run *skip training* and score
    # a model fitted for a different configuration. Measured on hgl: a dirty
    # workspace finishes in 3.2 s and reports LOSO rho = -0.576, while the same
    # command on a clean one trains for 322 s and reports +0.594. Stale state
    # was silently producing the published number.
    workspace = out_dir / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    if results_path.exists():
        results_path.unlink()

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


def _staleness(results_path: Path, cache_dir: Path) -> Optional[str]:
    """Why ``results_path`` cannot be reused, or None when it is safe to reuse.

    ``--resume`` used to skip any variant with a ``results.json`` on disk,
    regardless of age. On this repository that is not a convenience, it is a
    correctness hazard: `make table4` passes ``--resume``, so a table could be
    assembled from variants trained weeks apart against different caches, and
    nothing would say so. The one artifact that was checked this way —
    results/loso_all_variants.json — turned out not to reproduce from any commit.

    A result is reusable only when it is newer than every input that feeds it.
    """
    if not results_path.exists():
        return "no results.json"
    result_mtime = results_path.stat().st_mtime

    newest_input, newest_name = 0.0, ""
    for artefact in cache_dir.rglob("*.json"):
        m = artefact.stat().st_mtime
        if m > newest_input:
            newest_input, newest_name = m, str(artefact)
    if newest_input > result_mtime:
        return (f"cache artefact {newest_name} is newer "
                f"({_ago(newest_input)} vs result {_ago(result_mtime)})")

    age_days = (time.time() - result_mtime) / 86400
    if age_days > _RESUME_MAX_AGE_DAYS:
        return f"result is {age_days:.1f} days old (limit {_RESUME_MAX_AGE_DAYS})"
    return None


def _ago(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _extra_args(args) -> List[str]:
    """Flags forwarded verbatim to every cli/loso_evaluate.py invocation.

    Built once so every variant in the sweep runs under the same configuration —
    a flag applied to some variants and not others silently turns the comparison
    into an ablation of the flag.
    """
    extra = ["--eval-population", args.eval_population]
    if not args.auto_layers:
        extra.append("--no-auto-layers")
    extra += ["--inner-val-scenario", args.inner_val_scenario]
    if args.skip:
        extra += ["--skip", args.skip]
    if args.rank_normalize_features:
        extra.append("--rank-normalize-features")
    if args.rank_normalize_labels:
        extra.append("--rank-normalize-labels")
    return extra


# ── Comparison table ──────────────────────────────────────────────────────────

# Display names come from saag/evaluation/variant_registry.py. Under this
# harness gl/gl_qos run on the native graph, so they report as GAT-N / GAT-N-QoS.
_VARIANT_LABELS = {
    v: _registry.label(v, harness="loso")
    for v in ("topology_rm", "gl", "gl_qos", "hgl", "hgl_qos")
}


def _build_comparison_table(
    results_by_variant: Dict[str, Dict],
    eval_population: Optional[str] = None,
) -> Dict:
    """Merge per-variant LOSO results into a unified comparison table.

    ``eval_population`` is the value this run passed to every subprocess; it is
    used only as a fallback for payloads written before ``cli/loso_evaluate.py``
    started recording the field, so a resumed run does not report the population
    as unknown for some variants and known for others.
    """
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
            # Carried so a rendered table can name the population it was scored
            # on: the same variant on a pooled population is a different number,
            # not a noisier version of this one.
            "eval_population": (
                data.get("summary", {}).get("eval_population") or eval_population),
            "per_type_summary": data.get("per_type_summary", {}),
            "per_fold": [
                {
                    "holdout":  f.get("holdout_id"),
                    "mean_rho": f.get("mean_metrics", {}).get("spearman_rho"),
                    "std_rho":  f.get("std_metrics", {}).get("spearman_rho"),
                }
                for f in folds
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
    print("  Table 4: LOSO Inductive Evaluation")
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
    p.add_argument("--output-base", type=Path, default=OUTPUT_BASE,
                   help="Per-variant workspace root (default: output/loso). Point a "
                        "run that changes the corpus (e.g. --skip) at its own base so "
                        "it does not overwrite the workspace backing a published table.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "loso_all_variants.json")
    p.add_argument("--resume", action="store_true",
                   help="Reuse a variant's existing results.json when it is newer "
                        "than every cache artefact and less than "
                        f"{_RESUME_MAX_AGE_DAYS} days old. Anything older is re-run: "
                        "a table assembled from variants trained against different "
                        "caches is wrong in a way nothing else detects.")
    p.add_argument("--table-only", action="store_true",
                   help="Load existing results and print table only (no training)")
    p.add_argument(
        "--eval-population", default="application",
        choices=["application", "app_lib", "labeled"],
        help="Node population every variant is scored on; forwarded to "
             "cli/loso_evaluate.py. Defaults to 'application' so this table and "
             "reproduce/main_table.py compare like with like.",
    )
    p.add_argument(
        "--auto-layers", dest="auto_layers", action="store_true", default=False,
        help="Let cli/loso_evaluate.py derive the layer count from the primary "
             "scenario's size. OFF by default in this sweep: the primary changes "
             "with the holdout, so the enterprise fold trains 2 layers while the "
             "other seven train 3, and a capacity difference that tracks the fold "
             "is a confound in the very comparison this table reports.",
    )
    p.add_argument(
        "--inner-val-scenario", default="none", choices=["none", "auto"],
        help="Forwarded to cli/loso_evaluate.py. 'auto' selects checkpoints on a "
             "held-out training scenario instead of a within-primary split.",
    )
    p.add_argument("--skip", default="",
                   help="Comma-separated scenario id substrings dropped from the "
                        "corpus entirely, forwarded to cli/loso_evaluate.py. A "
                        "skipped scenario is neither a fold nor a training graph, "
                        "so this measures the corpus without it — not a holdout.")
    p.add_argument("--rank-normalize-features", action="store_true",
                   help="Forwarded to cli/loso_evaluate.py.")
    p.add_argument("--rank-normalize-labels", action="store_true",
                   help="Forwarded to cli/loso_evaluate.py.")
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

    output_base = args.output_base

    if args.table_only:
        # Load existing per-variant results
        for var in variants:
            rp = output_base / var / "results.json"
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
            rp = output_base / var / "results.json"
            if args.resume and rp.exists():
                stale = _staleness(rp, args.cache_dir)
                if stale is None:
                    print(f"  SKIP (resume): {var}")
                    results_by_variant[var] = json.loads(rp.read_text())
                    continue
                print(f"  STALE, re-running {var}: {stale}")

            print(f"  Running {var} ...")
            data = _run_variant(
                variant=var, seeds=args.seeds,
                cache_dir=args.cache_dir, epochs=args.epochs,
                extra_args=_extra_args(args),
                verbose=args.verbose,
                output_base=output_base,
            )
            results_by_variant[var] = data

    # Build + save comparison table
    table = _build_comparison_table(results_by_variant, args.eval_population)
    output = {
        "comparison_table": table,
        "per_variant_results": {k: v for k, v in results_by_variant.items()},
        "provenance": stamp(
            variants=list(results_by_variant), seeds=args.seeds, epochs=args.epochs,
            cache_dir=str(args.cache_dir), eval_population=args.eval_population,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {args.output}")

    _print_comparison_table(table)


if __name__ == "__main__":
    main()
