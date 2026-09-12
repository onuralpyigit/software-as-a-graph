#!/usr/bin/env python3
"""
reproduce/nested_loso_search.py — Nested-CV configuration search for LOSO
=========================================================================

Selects the predictor configuration by **inner** leave-one-scenario-out over the
N-1 training scenarios of each outer fold, then scores that configuration once
on the outer holdout. The outer holdout takes no part in selection: not in
early stopping, not in checkpoint choice, not in configuration choice.

Why this exists
---------------
`reproduce/loso_all_variants.py` scores one fixed configuration. Tuning that
configuration by looking at its LOSO table and re-running is test-set fishing —
with 8 folds and a discrete signed-rank statistic it is also very easy to do by
accident. This script makes the selection loop explicit and keeps it inside the
training set, which is the protocol `docs/research/jss/PREREGISTRATION.md`
commits to.

Protocol
--------
    for each outer fold k:
        inner_bundles := scenarios \\ {k}
        for each configuration c:
            inner_rho(c) := mean over inner folds j of LOSO(inner_bundles, j, c)
        c*(k) := argmax_c inner_rho(c)
        report LOSO(all_bundles, k, c*(k))

Cost is multiplicative: |outer| x |configs| x |inner| fold-trainings for the
search, plus |outer| for the final scores. Use --inner-seeds 42 (the default)
and a staged grid; the full grid is offered but is hours, not minutes.

Usage
-----
    PYTHONPATH=. python reproduce/nested_loso_search.py --grid stage1
    PYTHONPATH=. python reproduce/nested_loso_search.py --grid full --resume
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.loso_evaluate import ScenarioBundle, discover_scenarios, run_one_fold

RESULTS_DIR = Path("results")


# ── Search space ──────────────────────────────────────────────────────────────
# Every knob here is already exposed by cli/loso_evaluate.py. Nothing in this
# module changes the model; it only chooses among configurations the harness
# can already produce.

_AXES: Dict[str, Dict[str, List[Any]]] = {
    # Representation only: the two switches motivated by
    # results/feature_shift_diagnostic.md, crossed with depth.
    "stage1": {
        "rank_normalize_features": [False, True],
        "rank_normalize_labels": [False, True],
        "layers": [2, 3],
    },
    # Loss shape only: the metric is Spearman, so the ranking terms are the
    # ones with a prior reason to matter.
    "stage2": {
        "ranking_weight": [0.3, 1.0],
        "pairwise_ranking_weight": [0.1, 0.5],
        "dropout": [0.2, 0.4],
    },
    "full": {
        "rank_normalize_features": [False, True],
        "rank_normalize_labels": [False, True],
        "layers": [2, 3],
        "ranking_weight": [0.3, 1.0],
        "pairwise_ranking_weight": [0.1, 0.5],
        "dropout": [0.2, 0.4],
    },
}


def build_grid(grid: str, base: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Cartesian product of the named axes, each merged onto ``base``."""
    axes = _AXES[grid]
    keys = sorted(axes)
    configs = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        cfg = dict(base or {})
        cfg.update(dict(zip(keys, combo)))
        configs.append(cfg)
    return configs


def config_id(cfg: Dict[str, Any]) -> str:
    return ",".join(f"{k}={cfg[k]}" for k in sorted(cfg))


def _slug(cid: str) -> str:
    """Stable directory name for a config id.

    Not ``hash()``: PYTHONHASHSEED randomises that per process, so a resumed
    run would write its checkpoints somewhere else than the run it resumes.
    """
    return hashlib.md5(cid.encode()).hexdigest()[:12]


# ── One fold under one configuration ──────────────────────────────────────────

def _score_fold(
    bundles: List[ScenarioBundle],
    holdout_idx: int,
    cfg: Dict[str, Any],
    seeds: List[int],
    workdir: Path,
    variant: str,
    epochs: int,
    eval_population: str,
    device: Optional[str] = "auto",
) -> float:
    """Mean Spearman rho for one fold under one configuration.

    The workdir is cleared first. ``run_one_fold`` restores from any checkpoint
    it finds under the fold's directory, so a leftover tree from a different
    configuration makes the run skip training and score the wrong model — the
    same stale-state failure documented in reproduce/loso_all_variants.py.
    """
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    fold = run_one_fold(
        bundles=bundles,
        holdout_idx=holdout_idx,
        seeds=seeds,
        layer="app",
        epochs=epochs,
        lr=3e-4,
        hidden=64,
        heads=4,
        layers=cfg.get("layers", 3),
        dropout=cfg.get("dropout", 0.2),
        workdir=workdir,
        mode="gnn",
        variant=variant,
        eval_population=eval_population,
        # Depth comes from the configuration, never from the fold's primary
        # size: an auto-downgrade would make capacity a function of which
        # scenario is held out, which is the confound this whole pass removes.
        auto_layers=False,
        ranking_weight=cfg.get("ranking_weight", 0.3),
        pairwise_ranking_weight=cfg.get("pairwise_ranking_weight", 0.1),
        inner_val=cfg.get("inner_val", "auto"),
        rank_normalize_features=cfg.get("rank_normalize_features", False),
        rank_normalize_labels=cfg.get("rank_normalize_labels", False),
        device=device,
    )
    return float(fold.mean_metrics["spearman_rho"])


# ── Nested search ─────────────────────────────────────────────────────────────

def _inner_indices(n_inner: int, mode: str, k: int = 2) -> List[int]:
    """Which inner folds to score a configuration on.

    ``loso`` scores every inner fold — the pre-registered rule, and the most
    stable, but it costs |outer| x |configs| x (N-1) trainings.
    ``holdout`` scores ``k`` evenly-spaced inner folds instead. The inner loop's
    job is to *rank configurations*, not to estimate performance, so a smaller
    inner sample is a defensible trade; it is leakage-free either way because
    the outer holdout is not in the inner set at all. Deterministic, so a
    resumed run reuses the same cached fold scores.
    """
    if mode == "loso" or n_inner <= k:
        return list(range(n_inner))
    step = n_inner / k
    return sorted({int(i * step) for i in range(k)})


def run_search(
    bundles: List[ScenarioBundle],
    configs: List[Dict[str, Any]],
    variant: str,
    outer_seeds: List[int],
    inner_seeds: List[int],
    epochs: int,
    eval_population: str,
    workroot: Path,
    cache_path: Optional[Path],
    inner_mode: str = "loso",
    inner_epochs: Optional[int] = None,
    inner_k: int = 2,
    device: Optional[str] = "auto",
) -> Dict[str, Any]:
    cache: Dict[str, float] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"  Resuming from {cache_path} ({len(cache)} cached fold scores)")

    def _cached(key: str, compute) -> float:
        if key in cache:
            return cache[key]
        value = compute()
        cache[key] = value
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
        return value

    outer_records: List[Dict[str, Any]] = []
    t0 = time.time()

    for outer_idx, outer in enumerate(bundles):
        inner_bundles = [b for b in bundles if b.scenario_id != outer.scenario_id]
        # The guarantee the whole claim rests on.
        assert outer.scenario_id not in {b.scenario_id for b in inner_bundles}, (
            f"nested-CV leakage: outer holdout {outer.scenario_id} is in the "
            "inner search set"
        )

        print(f"\n  ── outer fold {outer_idx + 1}/{len(bundles)}: "
              f"holdout={outer.scenario_id} ──")

        inner_means: Dict[str, float] = {}
        for cfg in configs:
            cid = config_id(cfg)
            scores = []
            for inner_idx in _inner_indices(len(inner_bundles), inner_mode, inner_k):
                key = (f"{outer.scenario_id}|{cid}|"
                       f"{inner_bundles[inner_idx].scenario_id}|e{inner_epochs or epochs}")
                wd = workroot / "inner" / outer.scenario_id / _slug(cid)
                scores.append(_cached(key, lambda: _score_fold(
                    inner_bundles, inner_idx, cfg, inner_seeds, wd,
                    variant, inner_epochs or epochs, eval_population,
                    device=device,
                )))
            inner_means[cid] = float(np.mean(scores))
            print(f"     inner rho={inner_means[cid]:+.4f}  {cid}")

        best_cid = max(inner_means, key=inner_means.get)
        best_cfg = next(c for c in configs if config_id(c) == best_cid)

        outer_key = f"OUTER|{outer.scenario_id}|{best_cid}"
        outer_rho = _cached(outer_key, lambda: _score_fold(
            bundles, outer_idx, best_cfg, outer_seeds,
            workroot / "outer" / outer.scenario_id, variant, epochs,
            eval_population,
            device=device,
        ))

        print(f"     selected: {best_cid}  (inner {inner_means[best_cid]:+.4f})")
        print(f"     OUTER rho on {outer.scenario_id}: {outer_rho:+.4f}")

        outer_records.append({
            "holdout": outer.scenario_id,
            "selected_config": best_cfg,
            "selected_config_id": best_cid,
            "inner_mean_rho": inner_means[best_cid],
            "inner_mean_rho_by_config": inner_means,
            "outer_rho": outer_rho,
        })

    rhos = [r["outer_rho"] for r in outer_records]
    return {
        "variant": variant,
        "n_outer_folds": len(outer_records),
        "outer_seeds": outer_seeds,
        "inner_seeds": inner_seeds,
        "epochs": epochs,
        "eval_population": eval_population,
        "n_configs": len(configs),
        "inner_mode": inner_mode,
        "inner_k": inner_k,
        "inner_epochs": inner_epochs or epochs,
        "mean_outer_rho": float(np.mean(rhos)),
        "std_outer_rho": float(np.std(rhos)),
        "elapsed_s": round(time.time() - t0, 1),
        "folds": outer_records,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Nested-CV configuration search for LOSO.")
    p.add_argument("--cache-dir", type=Path, default=Path("output/loso_cache"))
    p.add_argument("--variant", default="hgl_qos",
                   choices=["hgl_qos", "hgl", "gl_qos", "gl"])
    p.add_argument("--grid", default="stage1", choices=sorted(_AXES))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seeds", default="42,123,456,789,2024",
                   help="Seeds for the outer (reported) folds")
    p.add_argument("--inner-seeds", default="42",
                   help="Seeds for inner selection folds. One is usually enough: "
                        "the inner score picks a configuration, it is not reported.")
    p.add_argument("--eval-population", default="application",
                   choices=["application", "app_lib", "labeled"])
    p.add_argument("--workroot", type=Path, default=Path("output/nested_search"))
    p.add_argument("--output", type=Path,
                   default=RESULTS_DIR / "nested_loso_search.json")
    p.add_argument("--resume", action="store_true",
                   help="Reuse fold scores cached from an earlier run")
    p.add_argument(
        "--inner-mode", default="loso", choices=["loso", "holdout"],
        help="How each configuration is scored inside a fold's training set. "
             "'loso' (default, pre-registered) scores every inner fold. "
             "'holdout' scores --inner-k of them, cutting cost by roughly "
             "(N-1)/k at some cost in selection stability. Neither can leak: "
             "the outer holdout is not in the inner set.",
    )
    p.add_argument("--inner-k", type=int, default=2,
                   help="Inner folds scored per configuration when "
                        "--inner-mode holdout (default: 2)")
    p.add_argument("--inner-epochs", type=int, default=None,
                   help="Epochs for inner selection runs (default: same as "
                        "--epochs). Selection only ranks configurations, so a "
                        "shorter budget is often enough; the reported outer "
                        "folds always use --epochs.")
    p.add_argument("--skip", default="", help="Comma-separated scenario id substrings")
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Device to use for training/evaluation (default: auto).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.cache_dir.exists():
        print(f"Error: --cache-dir {args.cache_dir} does not exist.", file=sys.stderr)
        return 2

    bundles = discover_scenarios(
        args.cache_dir, skip=[s.strip() for s in args.skip.split(",") if s.strip()]
    )
    configs = build_grid(args.grid)
    outer_seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    inner_seeds = [int(s) for s in args.inner_seeds.split(",") if s.strip()]

    per_config = len(_inner_indices(len(bundles) - 1, args.inner_mode, args.inner_k))
    n_inner = len(bundles) * len(configs) * per_config
    print(f"\n  Nested LOSO search — variant={args.variant} grid={args.grid}")
    print(f"  Scenarios : {len(bundles)}")
    print(f"  Configs   : {len(configs)}")
    print(f"  Device    : {args.device}")
    print(f"  Inner mode: {args.inner_mode} ({per_config} inner fold(s) per config, "
          f"{args.inner_epochs or args.epochs} epochs)")
    print(f"  Inner fold-trainings: {n_inner} x {len(inner_seeds)} seed(s)")

    cache_path = (args.workroot / f"{args.variant}_{args.grid}_cache.json"
                  if args.resume else None)
    report = run_search(
        bundles=bundles, configs=configs, variant=args.variant,
        outer_seeds=outer_seeds, inner_seeds=inner_seeds, epochs=args.epochs,
        eval_population=args.eval_population, workroot=args.workroot,
        cache_path=cache_path, inner_mode=args.inner_mode,
        inner_epochs=args.inner_epochs, inner_k=args.inner_k,
        device=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\n  Nested LOSO rho = {report['mean_outer_rho']:.4f} "
          f"± {report['std_outer_rho']:.4f}  (n={report['n_outer_folds']})")
    print(f"  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
