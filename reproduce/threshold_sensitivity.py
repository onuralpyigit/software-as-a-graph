#!/usr/bin/env python3
"""
reproduce/threshold_sensitivity.py — ground-truth robustness sweeps
==================================================================

Produces ``results/threshold_sensitivity.json``, the artifact behind the claim
in the JSS draft §8.3 that "the conclusions do not hinge on the canonical
default of 0.2". That claim previously had no committed evidence.

Two knobs are swept, because both are free parameters of the *ground truth*
rather than of the model, and a result that only holds at one setting of them is
a result about the setting:

``propagation_threshold``
    The average feed loss above which a subscriber is treated as starved and
    propagates the cascade. Canonical default 0.2.

``--norm``
    How Tier-1 structural metrics are scaled before the RM weighted sum. The
    default is rank-based, which makes the composite close to a Borda count and
    discards magnitude; ``minmax`` keeps magnitude and is outlier-sensitive.
    Sweeping it answers whether the reported ordering is an artifact of ranking.

Both sweeps report rho of the deterministic RM score against FaultInjector
labels, per scenario and in aggregate.

Usage
-----
    PYTHONPATH=. python reproduce/threshold_sensitivity.py
    PYTHONPATH=. python reproduce/threshold_sensitivity.py --thresholds 0.0 0.2 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger("threshold_sensitivity")

DEFAULT_THRESHOLDS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
DEFAULT_NORMS = ["robust", "minmax", "zscore"]
RESULTS_DIR = Path("results")


def _rho(
    pred: Dict[str, float], truth: Dict[str, float], scenario: str | None = None
) -> float | None:
    """Spearman rho on the Application population when the scenario is known.

    Pooling node types mixes populations with different impact scales and base
    rates; on this corpus that pushes the RM composite's pooled rho outside the
    range spanned by its own per-type values. Every headline table is scored on
    Applications, so this sweep is too.
    """
    if scenario is not None:
        from cli.loso_evaluate import _build_graph_from_json
        from saag.evaluation.metrics import resolve_eval_keys

        graph = _build_graph_from_json(_topology(scenario))
        common = resolve_eval_keys(pred, truth, graph, population="application")
    else:
        common = sorted(set(pred) & set(truth))
    if len(common) < 3:
        return None
    y_pred = np.array([pred[k] for k in common])
    y_true = np.array([truth[k] for k in common])
    if np.ptp(y_pred) == 0 or np.ptp(y_true) == 0:
        return None
    rho, _ = spearmanr(y_pred, y_true)
    return None if np.isnan(rho) else float(rho)


def _labels_at_threshold(scenario: str, threshold: float, seeds: List[int]) -> Dict[str, float]:
    """Re-label the scenario with FaultInjector at a given propagation threshold."""
    from cli.loso_evaluate import _build_graph_from_json
    from saag.simulation.fault_injector import FaultInjector
    from reproduce.ahp_sensitivity import _load_topology

    graph = _build_graph_from_json(_load_topology(scenario))
    injector = FaultInjector(
        graph=graph, seeds=seeds, cascade_depth_limit=0,
        propagation_threshold=threshold,
    )
    result = injector.run(node_types=["Application", "Broker", "Library"])
    return {nid: float(rec.impact_score) for nid, rec in result.records.items()}


def _rm_scores(scenario: str, norm: str) -> Dict[str, float]:
    """Deterministic Q(v) under a given normalisation method.

    Scored over the full typed graph rather than the Application+Library
    DEPENDS_ON projection; see ``reproduce.ahp_sensitivity.score_with_analyzer``
    for why the projection variant's Availability channel is degenerate. The
    structural analysis behind it is memoised, so sweeping a parameter that
    cannot move it costs one analysis per scenario rather than one per grid point.
    """
    from reproduce.ahp_sensitivity import _load_topology, score_with_analyzer
    from saag.analysis.analyzer import QualityAnalyzer

    return score_with_analyzer(
        _topology(scenario),
        QualityAnalyzer(use_ahp=True, normalization_method=norm),
    )


@lru_cache(maxsize=None)
def _topology(scenario: str) -> Dict[str, Any]:
    """Cached topology, so the memoised structural analysis keys stay stable."""
    from reproduce.ahp_sensitivity import _load_topology

    return _load_topology(scenario)


def sweep_thresholds(scenarios: List[str], thresholds: List[float], seeds: List[int]) -> List[Dict]:
    rows = []
    for threshold in thresholds:
        per_scenario: Dict[str, float] = {}
        label_scale: Dict[str, float] = {}
        for scenario in scenarios:
            try:
                truth = _labels_at_threshold(scenario, threshold, seeds)
                pred = _rm_scores(scenario, "robust")
            except Exception as exc:      # noqa: BLE001
                logger.warning("%s @ threshold=%.2f failed: %s", scenario, threshold, exc)
                continue
            rho = _rho(pred, truth, scenario)
            if rho is not None:
                per_scenario[scenario] = rho
            if truth:
                label_scale[scenario] = round(max(truth.values()), 4)

        rows.append({
            "propagation_threshold": threshold,
            "mean_rho": float(np.mean(list(per_scenario.values()))) if per_scenario else None,
            "per_scenario_rho": {k: round(v, 4) for k, v in sorted(per_scenario.items())},
            "label_scale_max": label_scale,
            "n_scenarios": len(per_scenario),
        })
        mean = rows[-1]["mean_rho"]
        print(f"  threshold={threshold:.2f}  mean_rho={mean if mean is None else round(mean, 4)}"
              f"  ({rows[-1]['n_scenarios']} scenarios)")
    return rows


def sweep_norms(scenarios: List[str], norms: List[str]) -> List[Dict]:
    from reproduce.main_table import _load_scenario_data

    truths: Dict[str, Dict[str, float]] = {}
    for scenario in scenarios:
        try:
            _g, _s, simulation, _r, _gt = _load_scenario_data(scenario, substrate="projection")
            truths[scenario] = {k: float(v.get("composite", 0.0)) for k, v in simulation.items()}
        except Exception as exc:      # noqa: BLE001
            logger.warning("%s failed to load: %s", scenario, exc)

    rows = []
    for norm in norms:
        per_scenario: Dict[str, float] = {}
        for scenario, truth in truths.items():
            try:
                pred = _rm_scores(scenario, norm)
            except Exception as exc:      # noqa: BLE001
                logger.warning("%s @ norm=%s failed: %s", scenario, norm, exc)
                continue
            rho = _rho(pred, truth, scenario)
            if rho is not None:
                per_scenario[scenario] = rho

        rows.append({
            "normalization": norm,
            "mean_rho": float(np.mean(list(per_scenario.values()))) if per_scenario else None,
            "per_scenario_rho": {k: round(v, 4) for k, v in sorted(per_scenario.items())},
            "n_scenarios": len(per_scenario),
        })
        mean = rows[-1]["mean_rho"]
        print(f"  norm={norm:<8} mean_rho={mean if mean is None else round(mean, 4)}"
              f"  ({rows[-1]['n_scenarios']} scenarios)")
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Ground-truth robustness sweeps")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    p.add_argument("--norms", nargs="+", default=DEFAULT_NORMS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--skip-thresholds", action="store_true")
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "threshold_sensitivity.json")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR)
    from reproduce.main_table import ALL_SCENARIOS

    scenarios = args.scenarios or ALL_SCENARIOS

    threshold_rows: List[Dict[str, Any]] = []
    if not args.skip_thresholds:
        print(f"Propagation-threshold sweep: {len(scenarios)} scenarios x {len(args.thresholds)} thresholds")
        threshold_rows = sweep_thresholds(scenarios, args.thresholds, args.seeds)

    print(f"\nNormalisation sweep: {len(scenarios)} scenarios x {len(args.norms)} methods")
    norm_rows = sweep_norms(scenarios, args.norms)

    defined_t = [r for r in threshold_rows if r["mean_rho"] is not None]
    defined_n = [r for r in norm_rows if r["mean_rho"] is not None]

    report = {
        "seeds": args.seeds,
        "scenarios": scenarios,
        "threshold_sweep": threshold_rows,
        "normalization_sweep": norm_rows,
        "interpretation": {
            "threshold_rho_spread": (
                round(max(r["mean_rho"] for r in defined_t)
                      - min(r["mean_rho"] for r in defined_t), 4)
                if defined_t else None
            ),
            "normalization_rho_spread": (
                round(max(r["mean_rho"] for r in defined_n)
                      - min(r["mean_rho"] for r in defined_n), 4)
                if defined_n else None
            ),
            "note": (
                "A small threshold spread means the reported ordering does not depend "
                "on the cascade-eligibility cutoff. A small normalisation spread means "
                "it does not depend on rank- vs magnitude-scaling. Neither spread says "
                "the ordering is *correct* — only that it is stable under the free "
                "parameters of the ground truth and the scorer."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    for k, v in report["interpretation"].items():
        if k != "note":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
