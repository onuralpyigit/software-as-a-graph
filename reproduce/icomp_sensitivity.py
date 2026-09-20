#!/usr/bin/env python3
"""
reproduce/icomp_sensitivity.py — sensitivity sweep of I_comp(v) severity weights
================================================================================

Addresses JSS Round 3 Reviewer Comment V3:
"Sweep the four severity coefficients (0.35, 0.25, 0.25, 0.15) under the
Morris/OFAT protocol already implemented, and report whether the
pooled-versus-stratified gap (rho = 0.098 vs. 0.119–0.566) is stable."

Evaluates the multi-metric composite oracle:
    I_comp(v) = w_1 * RL(v) + w_2 * FR(v) + w_3 * TL(v) + w_4 * FD(v)
where:
    w_1 = reachability loss weight
    w_2 = fragmentation weight
    w_3 = throughput loss weight
    w_4 = flow disruption weight
    sum(w) = 1, w_i >= 0.

Measures:
1. Baseline at shipped AHP weights (0.35, 0.25, 0.25, 0.15)
2. OFAT (One-Factor-At-A-Time) sweep across w_i in [0.05, 0.70]
3. Uniform random sampling on the 4-simplex: w ~ Dirichlet(1, 1, 1, 1) (N=1,000)
4. Morris elementary-effects screening over the 4 simplex factors

Outputs:
    results/icomp_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from reproduce._provenance import stamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("icomp_sensitivity")

RESULTS_DIR = Path("results")
CACHE_FILE = RESULTS_DIR / "icomp_scenario_cache.json"
OUTPUT_FILE = RESULTS_DIR / "icomp_sensitivity.json"

#: The scenario suite, imported rather than restated. This file used to carry
#: its own copy of the eight-scenario AuSE list, and the duplicate is what let
#: the two drift: Supplementary S1.2 described a corpus that Section 7.3 no
#: longer used, including a 17-component regression fixture absent from every
#: other table in the manuscript. One definition, selected by ``--scenarios``.
from reproduce.detection_validation import DETECTION_SCENARIOS  # noqa: E402

SHIPPED_WEIGHTS = {
    "reachability": 0.35,
    "fragmentation": 0.25,
    "throughput": 0.25,
    "flow_disruption": 0.15,
}

WEIGHT_KEYS = ["reachability", "fragmentation", "throughput", "flow_disruption"]


def _spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    res = spearmanr(x, y)
    val = float(res.correlation)
    return 0.0 if np.isnan(val) else val


def extract_scenario_data(scenario: str) -> Dict[str, Any]:
    """Extract Q(v) scores and raw simulation impact components for one scenario."""
    from reproduce.ahp_sensitivity import _load_topology
    from saag.analysis.service import AnalysisService
    from saag.infrastructure.memory_repo import MemoryRepository
    from saag.prediction.service import PredictionService
    from saag.simulation.service import SimulationService

    logger.info(f"Extracting scenario data for {scenario}...")
    topology = _load_topology(scenario)
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)

    # 1. Structural analysis and RM prediction
    layer_res = AnalysisService(repo).analyze_layer("system")
    quality = PredictionService().predict_quality(layer_res.structural)

    # 2. Exhaustive failure simulation
    sim_results = SimulationService(repo).run_failure_simulation_exhaustive(
        layer="system", propagation_threshold=0.2, seed=42
    )

    components = {c.id: c for c in quality.components}
    sim_map = {r.target_id: r.impact for r in sim_results}
    shared_ids = sorted(set(components) & set(sim_map))

    node_data = []
    for cid in shared_ids:
        c = components[cid]
        imp = sim_map[cid]
        node_data.append({
            "id": cid,
            "type": c.type,
            "q_score": float(c.scores.overall),
            "reachability": float(imp.reachability_loss),
            "fragmentation": float(imp.fragmentation),
            "throughput": float(imp.throughput_loss),
            "flow_disruption": float(imp.flow_disruption),
        })

    return {
        "scenario": scenario,
        "n_components": len(components),
        "n_scored": len(node_data),
        "nodes": node_data,
    }


def build_or_load_cache(
    force_refresh: bool = False,
    scenarios: Optional[List[str]] = None,
    cache_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build or load the per-scenario severity-component cache.

    The cache is keyed to the scenario set it was built from, and a cache built
    for one set silently answers for another: that is how a sweep could report
    ranges over eight topologies while its caption named a different corpus.
    A cache whose scenarios do not match the request is rebuilt rather than
    reused, and the set it describes is stored alongside it.
    """
    scenarios = list(scenarios or DETECTION_SCENARIOS)
    cache_file = Path(cache_file or CACHE_FILE)

    if cache_file.exists() and not force_refresh:
        with open(cache_file, "r", encoding="utf8") as f:
            cached = json.load(f)
        cached_ids = [c.get("scenario") for c in cached]
        if cached_ids == scenarios:
            logger.info(f"Loading cached scenario data from {cache_file}")
            return cached
        logger.warning(
            "Cache %s describes %d scenario(s) %s, not the %d requested — rebuilding.",
            cache_file, len(cached_ids), cached_ids[:3], len(scenarios))

    logger.info(f"Building scenario cache across {len(scenarios)} topologies...")
    cache = []
    for sc in scenarios:
        t0 = time.time()
        data = extract_scenario_data(sc)
        logger.info(f"Scenario {sc} completed in {time.time() - t0:.2f}s ({data['n_scored']} components)")
        cache.append(data)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf8") as f:
        json.dump(cache, f, indent=2)
    logger.info(f"Saved scenario cache to {cache_file}")
    return cache


def evaluate_weights(
    weights: Dict[str, float], cache: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Evaluate Spearman correlation across strata for one weight assignment."""
    w_rl = weights["reachability"]
    w_fr = weights["fragmentation"]
    w_tp = weights["throughput"]
    w_fd = weights["flow_disruption"]

    rhos_app: List[float] = []
    rhos_broker: List[float] = []
    rhos_node: List[float] = []
    rhos_pooled: List[float] = []

    for sc_data in cache:
        nodes = sc_data["nodes"]
        if not nodes:
            continue

        q_all = [n["q_score"] for n in nodes]
        act_all = [
            w_rl * n["reachability"]
            + w_fr * n["fragmentation"]
            + w_tp * n["throughput"]
            + w_fd * n["flow_disruption"]
            for n in nodes
        ]
        rhos_pooled.append(_spearman_rho(q_all, act_all))

        # By type
        by_type: Dict[str, List[Tuple[float, float]]] = {}
        for n, act in zip(nodes, act_all):
            by_type.setdefault(n["type"], []).append((n["q_score"], act))

        for ctype, pairs in by_type.items():
            if len(pairs) < 3:
                continue
            qs, acts = zip(*pairs)
            rho_t = _spearman_rho(qs, acts)
            if ctype == "Application":
                rhos_app.append(rho_t)
            elif ctype == "Broker":
                rhos_broker.append(rho_t)
            elif ctype in ("Node", "InfraNode"):
                rhos_node.append(rho_t)

    mean_app = float(np.mean(rhos_app)) if rhos_app else 0.0
    mean_broker = float(np.mean(rhos_broker)) if rhos_broker else 0.0
    mean_node = float(np.mean(rhos_node)) if rhos_node else 0.0
    mean_pooled = float(np.mean(rhos_pooled)) if rhos_pooled else 0.0

    min_stratified = min(mean_app, mean_broker, mean_node)
    simpson_gap = min_stratified - mean_pooled

    return {
        "mean_app": round(mean_app, 4),
        "mean_broker": round(mean_broker, 4),
        "mean_node": round(mean_node, 4),
        "mean_pooled": round(mean_pooled, 4),
        "min_stratified": round(min_stratified, 4),
        "simpson_gap": round(simpson_gap, 4),
        "simpson_holds": bool(simpson_gap > 0.0),
    }


def run_ofat(cache: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One-Factor-At-A-Time sweep across w_i in [0.05, 0.70]."""
    logger.info("Running OFAT sensitivity sweep...")
    results = {}
    sweep_values = np.linspace(0.05, 0.70, 14)

    for factor in WEIGHT_KEYS:
        factor_runs = []
        other_factors = [k for k in WEIGHT_KEYS if k != factor]
        other_sum_base = sum(SHIPPED_WEIGHTS[k] for k in other_factors)

        for val in sweep_values:
            val = float(val)
            rem = 1.0 - val
            point = {factor: val}
            for k in other_factors:
                point[k] = rem * (SHIPPED_WEIGHTS[k] / other_sum_base)

            eval_res = evaluate_weights(point, cache)
            factor_runs.append({
                "weight_value": round(val, 4),
                "weights": {k: round(v, 4) for k, v in point.items()},
                **eval_res,
            })
        results[factor] = factor_runs

    return results


def run_dirichlet_sampling(
    cache: List[Dict[str, Any]], n_samples: int = 1000, seed: int = 42
) -> Dict[str, Any]:
    """Sample uniformly on the 4-simplex via Dirichlet(1, 1, 1, 1)."""
    logger.info(f"Running Dirichlet uniform simplex sampling (N={n_samples})...")
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(np.ones(len(WEIGHT_KEYS)), size=n_samples)

    pooled_vals: List[float] = []
    app_vals: List[float] = []
    broker_vals: List[float] = []
    node_vals: List[float] = []
    gaps: List[float] = []
    simpson_holds_count = 0

    all_runs = []
    for s in samples:
        point = dict(zip(WEIGHT_KEYS, [float(x) for x in s]))
        res = evaluate_weights(point, cache)
        pooled_vals.append(res["mean_pooled"])
        app_vals.append(res["mean_app"])
        broker_vals.append(res["mean_broker"])
        node_vals.append(res["mean_node"])
        gaps.append(res["simpson_gap"])
        if res["simpson_holds"]:
            simpson_holds_count += 1
        all_runs.append({"weights": {k: round(v, 4) for k, v in point.items()}, **res})

    return {
        "n_samples": n_samples,
        "simpson_holds_fraction": round(simpson_holds_count / n_samples, 4),
        "pooled_range": [round(float(np.min(pooled_vals)), 4), round(float(np.max(pooled_vals)), 4)],
        "app_range": [round(float(np.min(app_vals)), 4), round(float(np.max(app_vals)), 4)],
        "broker_range": [round(float(np.min(broker_vals)), 4), round(float(np.max(broker_vals)), 4)],
        "node_range": [round(float(np.min(node_vals)), 4), round(float(np.max(node_vals)), 4)],
        "gap_range": [round(float(np.min(gaps)), 4), round(float(np.max(gaps)), 4)],
        "mean_gap": round(float(np.mean(gaps)), 4),
    }


def run_morris(
    cache: List[Dict[str, Any]], r: int = 15, levels: int = 4, seed: int = 42
) -> Dict[str, Any]:
    """Morris elementary-effects screening over the 4 simplex factors."""
    logger.info(f"Running Morris screening (r={r}, levels={levels})...")
    rng = np.random.default_rng(seed)
    delta = levels / (2 * (levels - 1))
    k = len(WEIGHT_KEYS)

    effects_pooled = {name: [] for name in WEIGHT_KEYS}
    effects_gap = {name: [] for name in WEIGHT_KEYS}

    for _ in range(r):
        x0 = rng.choice(np.linspace(0.1, 0.6, levels), size=k)
        x0 = x0 / np.sum(x0)
        curr = x0.copy()
        curr_dict = dict(zip(WEIGHT_KEYS, curr))
        curr_res = evaluate_weights(curr_dict, cache)

        order = rng.permutation(k)
        for idx in order:
            step = curr.copy()
            step[idx] += delta if step[idx] + delta <= 0.8 else -delta
            step = np.clip(step, 0.05, 0.8)
            step = step / np.sum(step)

            step_dict = dict(zip(WEIGHT_KEYS, step))
            step_res = evaluate_weights(step_dict, cache)

            dx = step[idx] - curr[idx]
            if abs(dx) > 1e-6:
                ee_p = (step_res["mean_pooled"] - curr_res["mean_pooled"]) / dx
                ee_g = (step_res["simpson_gap"] - curr_res["simpson_gap"]) / dx
                effects_pooled[WEIGHT_KEYS[idx]].append(abs(ee_p))
                effects_gap[WEIGHT_KEYS[idx]].append(abs(ee_g))

            curr = step
            curr_res = step_res

    per_factor = {}
    for name in WEIGHT_KEYS:
        ees_p = effects_pooled[name]
        ees_g = effects_gap[name]
        per_factor[name] = {
            "mu_star_pooled": round(float(np.mean(ees_p)), 4) if ees_p else 0.0,
            "sigma_pooled": round(float(np.std(ees_p)), 4) if ees_p else 0.0,
            "mu_star_gap": round(float(np.mean(ees_g)), 4) if ees_g else 0.0,
            "sigma_gap": round(float(np.std(ees_g)), 4) if ees_g else 0.0,
        }

    return per_factor


def main():
    parser = argparse.ArgumentParser(description="Sensitivity sweep of I_comp severity weights")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Scenarios to sweep. Default: the AuSE detection suite "
                             f"{DETECTION_SCENARIOS}. Pass the manuscript's own corpus "
                             "explicitly when the sweep is to back a JSS table.")
    parser.add_argument("--cache-file", default=str(CACHE_FILE),
                        help="Per-scenario severity-component cache. Keyed to the "
                             "scenario set, so a different --scenarios needs a "
                             "different cache file or --force-refresh.")
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--force-refresh", action="store_true", help="Force rebuild scenario cache")
    parser.add_argument("--dirichlet-n", type=int, default=1000, help="Number of Dirichlet simplex samples")
    parser.add_argument("--morris-r", type=int, default=15, help="Number of Morris trajectories")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenarios = args.scenarios or DETECTION_SCENARIOS
    cache = build_or_load_cache(force_refresh=args.force_refresh,
                                scenarios=scenarios,
                                cache_file=Path(args.cache_file))

    baseline = evaluate_weights(SHIPPED_WEIGHTS, cache)
    logger.info(f"Baseline (shipped AHP weights {SHIPPED_WEIGHTS}): {baseline}")

    ofat = run_ofat(cache)
    dirichlet = run_dirichlet_sampling(cache, n_samples=args.dirichlet_n, seed=args.seed)
    morris = run_morris(cache, r=args.morris_r, seed=args.seed)

    output = {
        "description": "Sensitivity analysis of Multi-Metric Composite Oracle I_comp(v) severity weights",
        "shipped_weights": SHIPPED_WEIGHTS,
        "baseline": baseline,
        "ofat": ofat,
        "dirichlet_simplex_sweep": dirichlet,
        "morris_screening": morris,
    }
    output["provenance"] = stamp()

    out_path = Path(args.output)
    output["scenarios"] = scenarios
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results written to {out_path}")

    print("\n" + "=" * 60)
    print("I_comp(v) SENSITIVITY SWEEP SUMMARY")
    print("=" * 60)
    print(f"Baseline: App={baseline['mean_app']}, Broker={baseline['mean_broker']}, Node={baseline['mean_node']}, Pooled={baseline['mean_pooled']}")
    print(f"Baseline Simpson Gap: {baseline['simpson_gap']} (Simpson's paradox holds: {baseline['simpson_holds']})")
    print(f"Dirichlet Simplex Sweep (N={dirichlet['n_samples']}):")
    print(f"  Simpson holds fraction: {dirichlet['simpson_holds_fraction'] * 100:.1f}%")
    print(f"  Pooled rho range: {dirichlet['pooled_range']}")
    print(f"  App rho range: {dirichlet['app_range']}")
    print(f"  Broker rho range: {dirichlet['broker_range']}")
    print(f"  Node rho range: {dirichlet['node_range']}")
    print(f"  Gap range: {dirichlet['gap_range']}")
    print("Morris Screening (Influence on Gap):")
    for k, v in morris.items():
        print(f"  {k}: mu*={v['mu_star_gap']}, sigma={v['sigma_gap']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
