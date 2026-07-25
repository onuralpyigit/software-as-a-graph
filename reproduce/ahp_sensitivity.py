#!/usr/bin/env python3
"""
reproduce/ahp_sensitivity.py — AHP shrinkage (lambda) sensitivity sweep
======================================================================

Produces ``results/ahp_shrinkage_sweep.json``, the artifact that
``docs/structural-analysis.md`` §11.2 and §11.6 cite for the claim that Spearman
rho plateaus over lambda in [0.65, 0.75] and that the lambda = 0.70 default is
therefore not a tuned coincidence. That artifact did not previously exist —
``docs/internal/`` was empty — so the sensitivity claim rested on nothing
committed.

What the sweep varies
---------------------
lambda blends the stated AHP judgement toward a uniform prior:

    w_final(d) = lambda * w_AHP(d) + (1 - lambda) * (1 / n)

lambda = 1.0 is the raw judgement; lambda = 0.0 is equal weights, i.e. the
``--equal-weights`` baseline. The sweep therefore also answers the sharper
question a reviewer will ask: *does the AHP judgement buy anything at all over
weighting the four dimensions equally?*

Two things worth knowing when reading the output
------------------------------------------------
1. Spearman rho is rank-based, so it responds only to how the weights reorder
   components — not to the scale of the composite. A flat curve means the
   ranking is insensitive to the weighting, which is a robustness result and
   simultaneously an argument that the exact AHP numbers are not load-bearing.
2. ``AHPProcessor`` applies shrinkage to the **intra-dimension** vectors
   as well as the composite, so lambda moves every RMAV formula at once. The
   sweep reports the resulting weights alongside rho so the two can be read
   together.

Usage
-----
    PYTHONPATH=. python reproduce/ahp_sensitivity.py
    PYTHONPATH=. python reproduce/ahp_sensitivity.py --lambdas 0.0 0.5 0.7 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger("ahp_sensitivity")

DEFAULT_LAMBDAS = [0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
RESULTS_DIR = Path("results")


def _score_components(topology: Dict[str, Any], structural: Dict[str, Any], lam: float) -> Dict[str, float]:
    """Composite Q(v) for every component under shrinkage factor *lam*.

    Delegates to the same scorer the main table uses, so the sweep cannot drift
    from the pipeline it is claiming to characterise.
    """
    from reproduce.main_table import _compute_rmav_from_structural
    from saag.analysis.analyzer import QualityAnalyzer

    scored = _compute_rmav_from_structural(
        topology, structural,
        analyzer=QualityAnalyzer(use_ahp=True, ahp_shrinkage=lam),
    )
    return {nid: float(v.get("composite", 0.0)) for nid, v in scored.items()}


def _weights_snapshot(lam: float) -> Dict[str, Any]:
    """The weights actually shipped at this lambda, for both dimension levels.

    Recorded because the docs print pre-shrinkage AHP weights while the code
    ships post-shrinkage ones — shrinkage is applied to the intra-dimension
    vectors too, not only the composite.
    """
    from saag.analysis.weight_calculator import AHPProcessor

    w = AHPProcessor(shrinkage_factor=lam).compute_weights()
    return {
        "composite": {
            "availability": round(w.q_availability, 4),
            "reliability": round(w.q_reliability, 4),
            "maintainability": round(w.q_maintainability, 4),
            "security": round(w.q_security, 4),
        },
        "reliability_terms": {
            "reverse_pagerank": round(w.r_reverse_pagerank, 4),
            "in_degree": round(w.r_in_degree, 4),
            "cdpot": round(w.r_cdpot, 4),
        },
    }


def _load_topology(scenario: str) -> Dict[str, Any]:
    """Cache topology if present, else the scenario definition — same order as main_table."""
    from reproduce.main_table import _find_cache_dir, SCENARIOS_DIR

    cache_topo = _find_cache_dir(scenario) / "topology.json"
    path = cache_topo if cache_topo.exists() else SCENARIOS_DIR / f"{scenario}.json"
    return json.loads(path.read_text())


def run_sweep(scenarios: List[str], lambdas: List[float]) -> Dict[str, Any]:
    from reproduce.main_table import _load_scenario_data

    # Load features once; only the weighting changes across lambdas.
    loaded: Dict[str, Any] = {}
    for scenario in scenarios:
        try:
            topology = _load_topology(scenario)
            _g, structural, simulation, _r, _gt = _load_scenario_data(scenario, substrate="projection")
            loaded[scenario] = (topology, structural, simulation)
        except Exception as exc:      # noqa: BLE001 - one bad scenario must not kill the sweep
            logger.warning("%s failed to load: %s", scenario, exc)

    rows: List[Dict[str, Any]] = []
    for lam in lambdas:
        per_scenario: Dict[str, float] = {}
        for scenario, (topology, structural, simulation) in loaded.items():
            try:
                pred = _score_components(topology, structural, lam)
            except Exception as exc:      # noqa: BLE001
                logger.warning("%s @ lambda=%.2f failed: %s", scenario, lam, exc)
                continue

            truth = {k: float(v.get("composite", 0.0)) for k, v in simulation.items()}
            common = sorted(set(pred) & set(truth))
            if len(common) < 3:
                continue
            y_pred = np.array([pred[k] for k in common])
            y_true = np.array([truth[k] for k in common])
            if np.ptp(y_pred) == 0 or np.ptp(y_true) == 0:
                continue
            rho, _ = spearmanr(y_pred, y_true)
            if not np.isnan(rho):
                per_scenario[scenario] = float(rho)

        rows.append({
            "lambda": lam,
            "mean_rho": float(np.mean(list(per_scenario.values()))) if per_scenario else None,
            "std_rho": float(np.std(list(per_scenario.values()))) if per_scenario else None,
            "per_scenario_rho": {k: round(v, 4) for k, v in sorted(per_scenario.items())},
            "n_scenarios": len(per_scenario),
            "weights": _weights_snapshot(lam),
        })
        mean = rows[-1]["mean_rho"]
        print(f"  lambda={lam:.2f}  mean_rho={mean if mean is None else round(mean, 4)}  "
              f"({rows[-1]['n_scenarios']} scenarios)")

    defined = [r for r in rows if r["mean_rho"] is not None]
    best = max(defined, key=lambda r: r["mean_rho"]) if defined else None
    equal = next((r for r in defined if abs(r["lambda"]) < 1e-9), None)
    default = next((r for r in defined if abs(r["lambda"] - 0.7) < 1e-9), None)

    interpretation = {}
    if best and equal and default:
        rhos = [r["mean_rho"] for r in defined]
        spread = max(rhos) - min(rhos)
        # Is the curve monotone in lambda? If it is, there is no plateau to
        # appeal to, and the direction of that monotonicity decides whether the
        # AHP judgement helps or hurts.
        ordered = [r["mean_rho"] for r in sorted(defined, key=lambda r: r["lambda"])]
        monotone_decreasing = all(b <= a + 1e-9 for a, b in zip(ordered, ordered[1:]))
        monotone_increasing = all(b >= a - 1e-9 for a, b in zip(ordered, ordered[1:]))

        interpretation = {
            "best_lambda": best["lambda"],
            "best_mean_rho": round(best["mean_rho"], 4),
            "default_lambda_mean_rho": round(default["mean_rho"], 4),
            "equal_weights_mean_rho": round(equal["mean_rho"], 4),
            "ahp_lift_over_equal_weights": round(default["mean_rho"] - equal["mean_rho"], 4),
            "rho_spread_across_lambda": round(spread, 4),
            "monotone_decreasing_in_lambda": monotone_decreasing,
            "monotone_increasing_in_lambda": monotone_increasing,
            "plateau_in_065_075": (
                max(r["mean_rho"] for r in defined if 0.65 <= r["lambda"] <= 0.75)
                - min(r["mean_rho"] for r in defined if 0.65 <= r["lambda"] <= 0.75)
            ) < 0.01 if any(0.65 <= r["lambda"] <= 0.75 for r in defined) else None,
            "note": (
                "Spearman rho is rank-based, so this measures only how the weighting "
                "reorders components. Read `monotone_decreasing_in_lambda` before "
                "quoting any plateau: if the curve is monotone there is no plateau to "
                "appeal to, and a negative `ahp_lift_over_equal_weights` means the "
                "stated AHP judgement ranks components *worse* than weighting the four "
                "dimensions equally on this cohort."
            ),
        }

    return {
        "lambdas": lambdas,
        "scenarios": scenarios,
        "rows": rows,
        "interpretation": interpretation,
    }


def parse_args():
    p = argparse.ArgumentParser(description="AHP shrinkage sensitivity sweep")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "ahp_shrinkage_sweep.json")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR)
    from reproduce.main_table import ALL_SCENARIOS

    scenarios = args.scenarios or ALL_SCENARIOS
    print(f"AHP shrinkage sweep: {len(scenarios)} scenarios x {len(args.lambdas)} lambdas")
    report = run_sweep(scenarios, args.lambdas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    if report["interpretation"]:
        for k, v in report["interpretation"].items():
            if k != "note":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
