#!/usr/bin/env python3
"""
reproduce/weight_global_sensitivity.py — joint (Morris + Dirichlet) sensitivity
================================================================================

Produces ``results/weight_global_sensitivity.json``.

Why this exists
----------------
Every other sweep in ``reproduce/`` (``topic_weight_sensitivity.py``,
``ahp_sensitivity.py``, ``domain_weight_comparison.py``, ``threshold_sensitivity.py``)
varies exactly one declared constant, or one convex combination on its own
simplex, at a time. That is the textbook one-at-a-time (OAT) design criticised
in the global sensitivity analysis literature (Saltelli et al. 2008): it cannot
detect interactions between factors, and — because each sweep fixes every other
constant at its shipped value — it cannot say whether a factor found "flat" in
isolation would still be flat if a second factor moved with it.

This script answers the joint question directly, over every hand-set weight
constant this revision touched or left in place:

    beta, alpha, psi                       (models.py: topic weight split)
    W_RELIABILITY, W_DURABILITY, W_PRIORITY (models.py: QoS sub-weight split)
    COMPONENT_POWER_MEAN_P                 (models.py: vertex aggregation exponent)
    LIB_FANOUT_GAMMA                       (models.py: library fan-out multiplier)
    AHP_SHRINKAGE_LAMBDA                   (models.py: intra-dimension shrinkage)
    r_alpha                                (QualityWeights: FT-vs-A blend)

k = 10 factors. Two designs, both cheap enough to run on a multi-scenario
corpus, reported side by side because they answer different questions:

1. **Morris elementary-effects screening** (Morris 1991, trajectory design per
   Campolongo et al. 2007) over the full hyper-rectangle each factor's plausible
   range spans. This deliberately does **not** hold beta+alpha+psi (or
   W_REL+W_DUR+W_PRI) to a unit sum during the one-at-a-time steps — Morris is
   specifically the tool for finding interactions and non-linearities an
   on-simplex sweep cannot see, and the pipeline's actual formulas
   (``compute_topic_weight``, ``QoSPolicy.calculate_weight``) do not require
   the three terms to sum to 1 to produce a well-defined, clamped weight; they
   just stop being read as "shares of a declared budget" off the simplex.
   Reports mu* (mean absolute elementary effect, ranks influence) and sigma
   (effect variability — large sigma relative to mu* signals interaction or
   non-linearity, which no OAT sweep in this repo can distinguish from noise).

2. **Dirichlet simplex sampling** of the two convex-combination factor groups
   (holding the four remaining independent factors at shipped values), which *does*
   respect the unit-sum constraint and therefore answers the sharper,
   practically relevant question: how much does rho actually vary over the
   realistic space of "declared budgets" the paper's framing assumes.

Both propagate through the same full-pipeline scorer every other RM sweep in
this directory uses (structural analysis + ``QualityAnalyzer``), scored
against I*(v) on the Application population by Spearman rho; the Dirichlet
stage additionally reports Kendall tau and top-20%-of-population Jaccard
overlap against the shipped ranking (rank-stability of the deliverable,
a ranked shortlist, rather than of the score itself) — never a parallel
reimplementation of any formula.

Cost note
---------
Evaluation cost is dominated by per-scenario structural analysis, which does
not scale uniformly: on this corpus ``enterprise_system`` (520 components)
costs roughly 15x any other scenario (~80s vs 1-6s per evaluation). It is
therefore excluded from the default scenario set so a default run finishes in
minutes rather than hours; pass ``--scenarios`` explicitly (optionally
including it) for a full-corpus run, budgeting accordingly.

Usage
-----
    PYTHONPATH=. python reproduce/weight_global_sensitivity.py
    PYTHONPATH=. python reproduce/weight_global_sensitivity.py --morris-r 20 --dirichlet-n 512
    PYTHONPATH=. python reproduce/weight_global_sensitivity.py --scenarios av_system enterprise_system
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import contextmanager
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import kendalltau, spearmanr

from saag.evaluation.metrics import resolve_eval_keys

logger = logging.getLogger("weight_global_sensitivity")

RESULTS_DIR = Path("results")

#: enterprise_system costs ~15x any other scenario per evaluation (large
#: component count drives structural analysis cost); excluded from the
#: default set so a default run finishes in minutes. Pass --scenarios to
#: include it for a full-corpus run.
DEFAULT_SCENARIOS: List[str] = [
    "av_system", "financial_trading_system", "healthcare_system",
    "hub_and_spoke_system", "microservices_system", "iot_smart_city_system",
]

#: (name, lo, hi, shipped) for every factor. Order fixes the Morris
#: trajectory's dimension order and the report's column order.
FACTORS: List[Tuple[str, float, float, float]] = [
    ("beta",           0.0,  1.0,  0.75),
    ("alpha",          0.0,  1.0,  0.15),
    ("psi",            0.0,  1.0,  0.10),
    ("w_reliability",  0.0,  1.0,  0.24),
    ("w_durability",   0.0,  1.0,  0.62),
    ("w_priority",     0.0,  1.0,  0.14),
    ("power_mean_p",   1.0,  10.0, 3.0),
    ("lib_fanout_gamma", 0.0, 0.5, 0.15),
    ("ahp_shrinkage_lambda", 0.0, 1.0, 0.70),
    ("r_alpha",        0.1,  0.9,  0.36),
]
FACTOR_NAMES = [f[0] for f in FACTORS]
K = len(FACTORS)

#: The two convex-combination groups the Dirichlet stage samples on-simplex.
SIMPLEX_GROUPS: List[Tuple[str, ...]] = [
    ("beta", "alpha", "psi"),
    ("w_reliability", "w_durability", "w_priority"),
]


# ── Point application: one dict of factor values -> the live pipeline ─────────

@contextmanager
def _apply_point(point: Dict[str, float]):
    """Install one factor assignment process-wide, restoring on exit.

    Patches the same module attributes ``reproduce/topic_weight_sensitivity.py``
    and ``reproduce/ahp_sensitivity.py`` already rely on being read at call
    time, plus ``QoSPolicy.W_*`` (class attributes) and the two vertex-
    aggregation constants (``COMPONENT_POWER_MEAN_P``, ``LIB_FANOUT_GAMMA``),
    which no existing sweep varies. ``ahp_shrinkage_lambda`` and ``r_alpha``
    are not module patches -- they are threaded into the ``QualityAnalyzer``
    constructed per evaluation instead, since shrinkage is a constructor
    argument and r_alpha lives on the ``QualityWeights`` the analyzer holds.

    ``LIB_FANOUT_GAMMA`` is also patched directly on
    ``saag.infrastructure.memory_repo``: that module imports it via
    ``from saag.core.models import LIB_FANOUT_GAMMA``, which binds a private
    copy of the float in its own namespace at import time, so patching only
    ``models.LIB_FANOUT_GAMMA`` would silently do nothing to the code path
    this sweep actually exercises (``MemoryRepository``).
    """
    from saag.core import models
    from saag.infrastructure import memory_repo

    module_attrs = {
        "TOPIC_QOS_WEIGHT_BETA": point["beta"],
        "TOPIC_SIZE_WEIGHT_ALPHA": point["alpha"],
        "TOPIC_FREQ_WEIGHT_PSI": point["psi"],
        "COMPONENT_POWER_MEAN_P": point["power_mean_p"],
        "LIB_FANOUT_GAMMA": point["lib_fanout_gamma"],
    }
    class_attrs = {
        "W_RELIABILITY": point["w_reliability"],
        "W_DURABILITY": point["w_durability"],
        "W_PRIORITY": point["w_priority"],
    }

    saved_module = {k: getattr(models, k) for k in module_attrs}
    saved_memory_repo_gamma = memory_repo.LIB_FANOUT_GAMMA
    saved_class = {k: getattr(models.QoSPolicy, k) for k in class_attrs}
    try:
        for k, v in module_attrs.items():
            setattr(models, k, float(v))
        memory_repo.LIB_FANOUT_GAMMA = float(point["lib_fanout_gamma"])
        for k, v in class_attrs.items():
            setattr(models.QoSPolicy, k, float(v))
        yield
    finally:
        for k, v in saved_module.items():
            setattr(models, k, v)
        memory_repo.LIB_FANOUT_GAMMA = saved_memory_repo_gamma
        for k, v in saved_class.items():
            setattr(models.QoSPolicy, k, v)


def _analyzer_for_point(point: Dict[str, float]):
    """QualityAnalyzer at this point's (lambda, r_alpha), full pipeline weights."""
    from saag.analysis.analyzer import QualityAnalyzer
    from saag.analysis.weight_calculator import AHPProcessor

    weights = AHPProcessor(shrinkage_factor=point["ahp_shrinkage_lambda"]).compute_weights()
    weights = dataclass_replace(weights, r_alpha=point["r_alpha"])
    return QualityAnalyzer(weights=weights, use_ahp=False)


# ── One evaluation: a point -> mean rho (and rank-stability) over scenarios ──

def _evaluate_point(
    point: Dict[str, float],
    topologies: Dict[str, Dict],
    truths: Dict[str, Dict[str, float]],
    graphs: Dict[str, Any],
) -> Dict[str, Any]:
    """Mean Spearman rho against I*(v) over scenarios, plus raw per-scenario
    predictions (for the Dirichlet stage's rank-stability metrics below).

    Structural analysis is recomputed fresh per point (no cache): almost every
    factor here can move the structural substrate (w(t), w(v), and therefore
    every derived edge weight), so a cache keyed on fewer than all nine
    factors would silently serve one point's structure to another's scoring —
    exactly the bug class ``full_pipeline_structural``'s cache key comment in
    ``ahp_sensitivity.py`` warns about for the two factors it already covers.
    """
    from saag.analysis.service import AnalysisService
    from saag.infrastructure.memory_repo import MemoryRepository

    rhos: List[float] = []
    pred_by_scenario: Dict[str, Dict[str, float]] = {}
    for scenario, topology in topologies.items():
        truth = truths.get(scenario)
        if not truth:
            continue
        try:
            with _apply_point(point):
                repo = MemoryRepository()
                repo.save_graph(topology, clear=True)
                structural = AnalysisService(repo).analyze_layer("system").structural
                analyzer = _analyzer_for_point(point)
                result = analyzer.analyze(structural)
            pred = {
                str(getattr(c, "id", getattr(c, "component_id", ""))): float(c.scores.overall)
                for c in result.components
            }
            keys = resolve_eval_keys(pred, truth, graphs[scenario], "application")
            if len(keys) < 3:
                continue
            pred_by_scenario[scenario] = {k: pred[k] for k in keys}
            a = np.array([pred[k] for k in keys])
            b = np.array([truth[k] for k in keys])
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                continue
            r, _ = spearmanr(a, b)
            if not np.isnan(r):
                rhos.append(float(r))
        except Exception as exc:      # noqa: BLE001 - one bad scenario must not kill a sweep point
            logger.warning("%s failed at point %s: %s", scenario, point, exc)

    return {
        "mean_rho": round(float(np.mean(rhos)), 4) if rhos else None,
        "n_scenarios": len(rhos),
        "pred_by_scenario": pred_by_scenario,
    }


def _rank_stability(
    pred_by_scenario: Dict[str, Dict[str, float]],
    reference_by_scenario: Dict[str, Dict[str, float]],
    top_frac: float = 0.2,
) -> Dict[str, Optional[float]]:
    """Kendall tau and top-K Jaccard overlap against a reference ranking.

    K = ceil(top_frac * n) per scenario (matching the tie-robust convention
    used elsewhere in this repo's oracle-agreement analyses), averaged over
    scenarios present in both dicts.
    """
    taus, jaccards = [], []
    for scenario, pred in pred_by_scenario.items():
        ref = reference_by_scenario.get(scenario)
        if not ref:
            continue
        keys = sorted(set(pred) & set(ref))
        if len(keys) < 3:
            continue
        a = np.array([pred[k] for k in keys])
        b = np.array([ref[k] for k in keys])
        if np.ptp(a) > 0 and np.ptp(b) > 0:
            tau, _ = kendalltau(a, b)
            if not np.isnan(tau):
                taus.append(float(tau))
        k = max(1, int(np.ceil(len(keys) * top_frac)))
        top_a = set(np.array(keys)[np.argsort(-a)[:k]])
        top_b = set(np.array(keys)[np.argsort(-b)[:k]])
        jaccards.append(len(top_a & top_b) / len(top_a | top_b) if (top_a | top_b) else 1.0)
    return {
        "mean_tau_vs_shipped": round(float(np.mean(taus)), 4) if taus else None,
        "mean_jaccard_top20pct_vs_shipped": round(float(np.mean(jaccards)), 4) if jaccards else None,
    }


def _shipped_point() -> Dict[str, float]:
    return {name: shipped for name, _lo, _hi, shipped in FACTORS}


# ── Morris elementary-effects design ───────────────────────────────────────

def _morris_trajectories(r: int, levels: int, rng: np.random.Generator) -> List[np.ndarray]:
    """r trajectories of (k+1) points each in the unit hypercube [0,1]^k.

    Standard Morris (1991) construction: a random base point on a p-level
    grid, then k steps, each moving one randomly-chosen not-yet-moved
    dimension by a fixed signed step of size ``delta = levels/(2*(levels-1))``.
    Order of dimensions and step signs are independently randomised per
    trajectory (Campolongo et al. 2007).
    """
    grid = np.linspace(0.0, 1.0 - 1.0 / (levels - 1), levels - 1)
    delta = levels / (2.0 * (levels - 1))
    trajectories = []
    for _ in range(r):
        base = rng.choice(grid, size=K)
        order = rng.permutation(K)
        signs = rng.choice([-1.0, 1.0], size=K)
        points = [base.copy()]
        current = base.copy()
        for dim in order:
            step = signs[dim] * delta
            new = current.copy()
            candidate = new[dim] + step
            # Reflect at the boundary rather than clip, so the step size (and
            # therefore the elementary effect's denominator) stays exact.
            new[dim] = candidate if 0.0 <= candidate <= 1.0 else new[dim] - step
            points.append(new)
            current = new
        trajectories.append(np.clip(np.array(points), 0.0, 1.0))
    return trajectories


def _to_point(unit_vec: np.ndarray) -> Dict[str, float]:
    return {
        name: lo + u * (hi - lo)
        for (name, lo, hi, _shipped), u in zip(FACTORS, unit_vec)
    }


def run_morris(
    r: int, levels: int, topologies, truths, graphs, seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    trajectories = _morris_trajectories(r, levels, rng)
    delta = levels / (2.0 * (levels - 1))

    effects: Dict[str, List[float]] = {name: [] for name in FACTOR_NAMES}
    n_evals = 0
    for traj in trajectories:
        prev_point = _to_point(traj[0])
        prev_rho = _evaluate_point(prev_point, topologies, truths, graphs)["mean_rho"]
        n_evals += 1
        for i in range(1, len(traj)):
            point = _to_point(traj[i])
            rho = _evaluate_point(point, topologies, truths, graphs)["mean_rho"]
            n_evals += 1
            moved = np.nonzero(traj[i] - traj[i - 1])[0]
            if len(moved) == 1 and rho is not None and prev_rho is not None:
                dim = moved[0]
                signed_step = (traj[i][dim] - traj[i - 1][dim])
                ee = (rho - prev_rho) / signed_step if signed_step != 0 else 0.0
                effects[FACTOR_NAMES[dim]].append(ee)
            prev_point, prev_rho = point, rho
        print(f"    trajectory done ({n_evals} evaluations so far)")

    summary = {}
    for name in FACTOR_NAMES:
        vals = np.array(effects[name])
        summary[name] = {
            "n": int(vals.size),
            "mu_star": round(float(np.mean(np.abs(vals))), 5) if vals.size else None,
            "sigma": round(float(np.std(vals)), 5) if vals.size else None,
        }
    ranked = sorted(
        (n for n in FACTOR_NAMES if summary[n]["mu_star"] is not None),
        key=lambda n: -summary[n]["mu_star"],
    )
    return {
        "design": {"r_trajectories": r, "levels": levels, "n_evaluations": n_evals, "seed": seed},
        "per_factor": summary,
        "ranked_by_influence": ranked,
        "note": (
            "mu_star ranks influence on mean rho; a large sigma relative to "
            "mu_star for a factor means its effect depends on where the other "
            "eight factors sit (interaction or non-linearity) rather than "
            "being a fixed marginal slope -- the specific limitation an "
            "on-simplex, one-at-a-time sweep cannot diagnose."
        ),
    }


# ── Dirichlet simplex sampling ─────────────────────────────────────────────

def run_dirichlet(n: int, topologies, truths, graphs, seed: int = 43) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    shipped = _shipped_point()

    shipped_result = _evaluate_point(shipped, topologies, truths, graphs)
    shipped_rho = shipped_result["mean_rho"]
    shipped_pred = shipped_result["pred_by_scenario"]

    rhos: List[float] = []
    taus: List[float] = []
    jaccards: List[float] = []
    for i in range(n):
        point = dict(shipped)
        for group in SIMPLEX_GROUPS:
            draw = rng.dirichlet(np.ones(len(group)))
            for name, value in zip(group, draw):
                point[name] = float(value)
        result = _evaluate_point(point, topologies, truths, graphs)
        if result["mean_rho"] is not None:
            rhos.append(result["mean_rho"])
        stability = _rank_stability(result["pred_by_scenario"], shipped_pred)
        if stability["mean_tau_vs_shipped"] is not None:
            taus.append(stability["mean_tau_vs_shipped"])
        if stability["mean_jaccard_top20pct_vs_shipped"] is not None:
            jaccards.append(stability["mean_jaccard_top20pct_vs_shipped"])
        if (i + 1) % max(1, n // 10) == 0:
            print(f"    Dirichlet draw {i + 1}/{n}")

    arr = np.array(rhos)
    tau_arr = np.array(taus)
    jac_arr = np.array(jaccards)
    return {
        "n_draws": n,
        "shipped_rho": shipped_rho,
        "mean_rho": round(float(arr.mean()), 4) if arr.size else None,
        "sd_rho": round(float(arr.std()), 4) if arr.size else None,
        "min_rho": round(float(arr.min()), 4) if arr.size else None,
        "max_rho": round(float(arr.max()), 4) if arr.size else None,
        "ci_90": [round(float(np.percentile(arr, 5)), 4), round(float(np.percentile(arr, 95)), 4)] if arr.size else None,
        "rank_stability_vs_shipped": {
            "mean_kendall_tau": round(float(tau_arr.mean()), 4) if tau_arr.size else None,
            "min_kendall_tau": round(float(tau_arr.min()), 4) if tau_arr.size else None,
            "mean_jaccard_top20pct": round(float(jac_arr.mean()), 4) if jac_arr.size else None,
            "min_jaccard_top20pct": round(float(jac_arr.min()), 4) if jac_arr.size else None,
        },
        "note": (
            "Draws respect the unit-sum constraint on both (beta,alpha,psi) "
            "and (w_reliability,w_durability,w_priority) simultaneously, "
            "holding the other six factors at shipped values -- the "
            "realistic 'declared budget' space, as opposed to Morris's "
            "unconstrained hyper-rectangle above."
        ),
    }


# ── Data loading (shared with the other RM sweeps) ─────────────────────────

def _load_data(scenarios: List[str]):
    from cli.loso_evaluate import _build_graph_from_json
    from reproduce.ahp_sensitivity import _load_topology
    from reproduce.main_table import _load_scenario_data

    topologies, truths, graphs = {}, {}, {}
    for scenario in scenarios:
        try:
            topology = _load_topology(scenario)
            _g, _s, simulation, _r, _gt = _load_scenario_data(scenario, substrate="projection")
            topologies[scenario] = topology
            truths[scenario] = {k: float(v.get("composite", 0.0)) for k, v in simulation.items()}
            graphs[scenario] = _build_graph_from_json(topology)
        except Exception as exc:      # noqa: BLE001
            logger.warning("%s failed to load: %s", scenario, exc)
    return topologies, truths, graphs


def parse_args():
    p = argparse.ArgumentParser(description="Joint Morris + Dirichlet sensitivity of every weight constant")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help=f"default: {DEFAULT_SCENARIOS} (excludes enterprise_system; see cost note)")
    p.add_argument("--morris-r", type=int, default=10, help="Morris trajectories (cost: r*(k+1) evaluations)")
    p.add_argument("--morris-levels", type=int, default=4)
    p.add_argument("--dirichlet-n", type=int, default=100)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "weight_global_sensitivity.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    scenarios = args.scenarios or DEFAULT_SCENARIOS
    n_evals = args.morris_r * (K + 1) + args.dirichlet_n
    print(f"Weight global sensitivity: k={K} factors, {len(scenarios)} scenarios, "
          f"~{n_evals} evaluation points (Morris r={args.morris_r}, Dirichlet n={args.dirichlet_n})")

    topologies, truths, graphs = _load_data(scenarios)
    if not topologies:
        print("No scenarios loaded; aborting.")
        return

    print("Running Morris elementary-effects screening...")
    morris = run_morris(args.morris_r, args.morris_levels, topologies, truths, graphs, seed=args.seed)
    for name in morris["ranked_by_influence"]:
        s = morris["per_factor"][name]
        print(f"  {name:<22s} mu*={s['mu_star']:.5f}  sigma={s['sigma']:.5f}  n={s['n']}")

    print("Running Dirichlet simplex sampling...")
    dirichlet = run_dirichlet(args.dirichlet_n, topologies, truths, graphs, seed=args.seed + 1)
    print(f"  shipped rho={dirichlet['shipped_rho']}  mean={dirichlet['mean_rho']}  "
          f"sd={dirichlet['sd_rho']}  range=[{dirichlet['min_rho']}, {dirichlet['max_rho']}]")

    report = {
        "factors": [
            {"name": n, "lo": lo, "hi": hi, "shipped": s} for n, lo, hi, s in FACTORS
        ],
        "scenarios": sorted(topologies),
        "morris": morris,
        "dirichlet": dirichlet,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
