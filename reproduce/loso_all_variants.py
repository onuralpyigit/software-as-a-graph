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

  # Eight concurrent (fold, seed) fits. Changes wall-clock only — every fit is
  # independent and seeded from its own seed, so the table is bit-identical.
  python reproduce/loso_all_variants.py --jobs 8

  # Resume. A variant is skipped outright when its results.json is still fresh
  # (see _staleness); otherwise it re-runs, and cli/loso_evaluate.py reuses each
  # individual fit whose fingerprint — configuration, cache contents, model code
  # — still matches, so an interrupted sweep does not start over.
  python reproduce/loso_all_variants.py --resume --jobs 8
"""

from __future__ import annotations

import argparse
import json
import os
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
#: RQ2 confound controls (Section 7.2). Held apart from ALL_VARIANTS so a plain
#: sweep still produces exactly the manuscript's variant set; pass them
#: explicitly via --variants to run the controls. They must be run in the SAME
#: invocation as the arms they are compared against, because the Wilcoxon tests
#: downstream are paired over folds.
CONTROL_VARIANTS = [
    "gl_full_cap", "gl_full_qos_cap", "gl_full_qos16_cap", "hgl_qos_uni",
    # Not a confound control but the same kind of arm: learned, scored on the
    # same folds, and only meaningful paired against the GNNs it is compared to.
    "tab_gbm",
]

#: Dispatch order, measured rather than assumed (one 12-fold x 5-seed sweep on a
#: Tesla T4: hgl 12635 s, gl_qos 5096 s, gl 3432 s, topology_rm 214 s, topo_qos
#: 32 s, topo_baseline 24 s; hgl_qos did not finish). ALL_VARIANTS stays in
#: reporting order — this only decides what runs first. The expensive arms go
#: first deliberately: a sweep that is interrupted should lose the cheap rows,
#: which cost minutes to redo, not the headline arm that costs hours. The run
#: this was written after lost exactly the headline arm.
_DISPATCH_COST = {
    "hgl_qos": 0, "hgl": 1, "gl_qos": 2, "gl": 3,
    "topology_rm": 4, "topo_qos": 5, "topo_baseline": 6,
    # No epochs and no forward pass: seconds, not hours.
    "tab_gbm": 7,
}


def _dispatch_order(variants: List[str]) -> List[str]:
    return sorted(variants, key=lambda v: (_DISPATCH_COST.get(v, 3), v))


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
    resume: bool = False,
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
    #
    # Under --resume the workspace is kept instead, because the hazard is now
    # handled where it belongs: cli/loso_evaluate.py stores a fingerprint of the
    # configuration, the cache contents and the model code beside every
    # completed fit, reuses a fit only on an exact match, and deletes the
    # checkpoint of any fit it cannot match before re-running it. Wiping here
    # too would mean a sweep interrupted at hour three restarts at hour zero.
    workspace = out_dir / "workspace"
    if workspace.exists() and not resume:
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
    if resume:
        cmd.append("--resume")
    if verbose:
        print(f"    CMD: {' '.join(cmd)}")

    # BLAS/OpenMP thread pools are worse than useless on graphs this size, and
    # actively harmful once --jobs puts several fits on one machine: each worker
    # would claim every core. cli/loso_evaluate.py pins torch's own intra-op
    # threads; these cover the numpy/scipy side.
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=True,
        env=env,
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
    if getattr(args, "device", None):
        extra += ["--device", args.device]
    extra += ["--jobs", str(args.jobs), "--torch-threads", str(args.torch_threads)]
    if not args.preflight:
        extra.append("--no-preflight")
    return extra


# ── Comparison table ──────────────────────────────────────────────────────────

# Display names come from saag/evaluation/variant_registry.py. Under this
# harness gl/gl_qos run on the native graph, so they report as GAT-N / GAT-N-QoS.
_VARIANT_LABELS = {
    v: _registry.label(v, harness="loso")
    for v in ("topology_rm", "gl", "gl_qos", "hgl", "hgl_qos", *CONTROL_VARIANTS)
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
    p.add_argument("--variants", nargs="+", default=None, choices=ALL_VARIANTS + CONTROL_VARIANTS,
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
        choices=["application", "app_lib", "labeled",
                             "topic", "node", "broker", "library"],
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
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Device for model training/inference (default: auto -> cuda if available else cpu)",
    )
    p.add_argument(
        "--jobs", type=int, default=1,
        help="Concurrent (fold, seed) fits within each variant (default: 1). "
             "The 240 fits behind this table are independent — each seeds torch "
             "and numpy from its own seed and writes to its own directory — so "
             "this changes wall-clock only; a fold reproduces bit-identically at "
             "any --jobs. Fits are latency-bound rather than compute-bound on a "
             "corpus this size, so several workers share one device well.",
    )
    p.add_argument(
        "--torch-threads", type=int, default=1,
        help="Intra-op threads per fit (default: 1). Forwarded to "
             "cli/loso_evaluate.py; see its help for the measurement.",
    )
    p.add_argument(
        "--no-preflight", dest="preflight", action="store_false", default=True,
        help="Skip each variant's one-epoch probe fit. The probe costs seconds "
             "and is the difference between learning about an environment fault "
             "now and learning about it after the sweep.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    import torch
    args = parse_args()
    variants = args.variants or ALL_VARIANTS

    dev_desc = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    if dev_desc == "cuda":
        dev_desc += f" ({torch.cuda.get_device_name(0)})"

    print(f"\n  LOSO All-Variants Sweep (Block E)")
    print(f"  Variants  : {variants}")
    print(f"  Seeds     : {args.seeds}")
    print(f"  Device    : {dev_desc}")
    print(f"  Cache dir : {args.cache_dir}")
    print(f"  Jobs      : {args.jobs} concurrent fit(s), {args.torch_threads} thread(s) each")
    print(f"  Resume    : {'on (fingerprinted, per fit)' if args.resume else 'off'}")
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

        for var in _dispatch_order(variants):
            rp = output_base / var / "results.json"
            if args.resume and rp.exists():
                stale = _staleness(rp, args.cache_dir)
                if stale is None:
                    print(f"  SKIP (resume): {var}")
                    results_by_variant[var] = json.loads(rp.read_text())
                    continue
                # Re-running no longer means starting from zero: the child
                # reuses every individual fit whose fingerprint still matches,
                # so a variant that is stale for one reason (an age bound, a
                # touched cache file) does not pay for all 60 fits again.
                print(f"  STALE, re-running {var}: {stale}")

            print(f"  Running {var} ...")
            data = _run_variant(
                variant=var, seeds=args.seeds,
                cache_dir=args.cache_dir, epochs=args.epochs,
                extra_args=_extra_args(args),
                verbose=args.verbose,
                output_base=output_base,
                resume=args.resume,
            )
            results_by_variant[var] = data

    # Build + save comparison table
    # Reporting order is ALL_VARIANTS', not the order they happened to run in.
    results_by_variant = {
        v: results_by_variant[v]
        for v in [*ALL_VARIANTS, *CONTROL_VARIANTS] if v in results_by_variant
    }
    table = _build_comparison_table(results_by_variant, args.eval_population)
    output = {
        "comparison_table": table,
        "per_variant_results": {k: v for k, v in results_by_variant.items()},
        "provenance": stamp(
            variants=list(results_by_variant), seeds=args.seeds, epochs=args.epochs,
            cache_dir=str(args.cache_dir), eval_population=args.eval_population,
            # Recorded because the CPU->GPU re-baseline makes rows from
            # different devices non-comparable, and nothing else on disk
            # distinguishes them.
            device=getattr(args, "device", None),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {args.output}")

    _print_comparison_table(table)


if __name__ == "__main__":
    main()
