#!/usr/bin/env python3
"""
reproduce/domain_weight_comparison.py — does domain-derived weighting rank better?
====================================================================================

Measures whether deriving the RM composite weight w_R (= q_reliability, with
q_maintainability = 1 - w_R) from a scenario's declared deployment domain
(saag.core.quality_model.derive_rm_weights — Layer 3 of the layered SQuaRE
quality model, docs/criticality.md §3.5) improves ranking correlation against
I*(v), compared to the static default and equal weights.

**Stated prior, so the result is read honestly either way.** Equal weights
already beat the calibrated AHP vector by 0.111 rho on the shrinkage sweep
(reproduce/ahp_sensitivity.py, docs/structural-analysis.md §11.6). The domain
derivation is a different construction — it comes from ISO/IEC 25019:2023
stakeholder priorities, not from a pairwise-comparison judgement — but nothing
guarantees it ranks any better, and it most likely does not. Both outcomes are
reportable: an improvement would be the first evidence that a *non-arbitrary*
weighting beats equal weights; a null result documents that Layer 3, as
currently specified, does not help ranking either, which still leaves its
value as an *attribution* device (explaining criticality in stakeholder terms)
intact per the existing scoping of the RM composite (docs/criticality.md §4.3).

**Why a 1-D sweep, not four discrete arms.** With the RM composite down to a
single free parameter (w_R, since w_M = 1 - w_R), the four arms this script
used to compare — static (0.80), domain_derived (in [0.70, 0.76] across the
six domains), ahp_070 (retired — AHP no longer touches the composite, see
reproduce/ahp_sensitivity.py), equal (0.50) — collapse to three points on one
axis, with three of them within 0.10 of each other. That is not enough spread
to be testable against seed/scenario noise, so instead this script sweeps
w_R across its full range and reports where each named point falls on the
resulting curve. The curve is the complete characterisation of composite
weighting in the 2-D model; the discrete comparison is a readout of it, not a
separate experiment.

Ten scenarios: the seven synthetic scenarios of reproduce/main_table.py
ALL_SCENARIOS (LOSO-cached, scored via the same _compute_rm_from_structural /
_load_scenario_data path as reproduce/ahp_sensitivity.py) plus the three
real-world graphs used for RQ4 (uncached; scored via a direct structural-feature
build and a fresh FaultInjector run, since cli.loso_evaluate._build_graph_from_json
crashes on real-world Broker/Node entries that already carry a raw `type` field
— a separate, pre-existing defect, not touched here).

Usage
-----
    PYTHONPATH=. python reproduce/domain_weight_comparison.py
    PYTHONPATH=. python reproduce/domain_weight_comparison.py --step 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger("domain_weight_comparison")

RESULTS_DIR = Path("results")
SCENARIOS_DIR = Path("data/scenarios")

#: w_R sweep grid — w_M = 1 - w_R at every point.
DEFAULT_STEP = 0.05

#: The seven synthetic scenarios with a populated LOSO cache.
SYNTHETIC_SCENARIOS = [
    "av_system",
    "iot_smart_city_system",
    "financial_trading_system",
    "healthcare_system",
    "hub_and_spoke_system",
    "microservices_system",
    "enterprise_system",
]

#: The three real-world graphs backing RQ4. No LOSO cache exists for these.
REALWORLD_SCENARIOS = [
    "realworld_autoware_ros2",
    "realworld_trainticket",
    "realworld_cloud_microservices",
]

#: Named points marked on the w_R sweep curve. "domain_derived" is resolved
#: per-scenario (its w_R depends on the scenario's domain), not fixed here.
STATIC_W_R = 0.80
EQUAL_W_R = 0.50


# ---------------------------------------------------------------------------
# Loading: cached path for synthetic scenarios, direct build for real-world
# ---------------------------------------------------------------------------

def _domain_of(topology: Dict[str, Any]) -> Optional[str]:
    return (topology.get("metadata") or {}).get("domain")


def _load_synthetic(scenario: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    """Topology, structural features, and I*(v)-backed truth via the LOSO
    cache — identical path to reproduce/ahp_sensitivity.py, so this cannot
    drift from the pipeline it is compared against."""
    from reproduce.main_table import _load_scenario_data, _find_cache_dir

    cache_topo = _find_cache_dir(scenario) / "topology.json"
    path = cache_topo if cache_topo.exists() else SCENARIOS_DIR / f"{scenario}.json"
    topology = json.loads(path.read_text())

    _g, structural, simulation, _r, _gt = _load_scenario_data(scenario, substrate="projection")
    truth = {k: float(v.get("composite", 0.0)) for k, v in simulation.items()}
    return topology, structural, truth


def _build_graph_safe(topology: Dict[str, Any]):
    """Local re-implementation of cli.loso_evaluate._build_graph_from_json that
    tolerates entities whose JSON already carries a 'type' field (real-world
    Broker/Node entries do) — the shared helper raises
    'got multiple values for keyword argument type' on those. Scratch-scoped
    workaround for this script only; not a fix to the shared loader."""
    import networkx as nx

    g = nx.DiGraph()
    buckets = [
        ("applications", "Application"), ("brokers", "Broker"),
        ("topics", "Topic"), ("nodes", "Node"), ("libraries", "Library"),
    ]
    for key, type_label in buckets:
        for entity in topology.get(key, []):
            attrs = {k: v for k, v in entity.items() if k not in ("id", "name", "type")}
            g.add_node(entity["id"], type=type_label, name=entity.get("name", entity["id"]), **attrs)

    rels = topology.get("relationships", {}) or {}
    edge_buckets = [
        (rels.get("publishes_to", []) + topology.get("publishes", []), "PUBLISHES_TO"),
        (rels.get("subscribes_to", []) + topology.get("subscribes", []), "SUBSCRIBES_TO"),
        (rels.get("routes", []) + topology.get("routes", []), "ROUTES"),
        (rels.get("runs_on", []) + topology.get("runs_on", []), "RUNS_ON"),
        (rels.get("connects_to", []) + topology.get("connects_to", []), "CONNECTS_TO"),
        (rels.get("uses", []) + topology.get("uses", []), "USES"),
    ]
    for items, type_label in edge_buckets:
        for r in items:
            src = (r.get("source") or r.get("from") or r.get("application_id")
                   or r.get("topic_id") or r.get("node_id") or r.get("broker_id"))
            dst = (r.get("target") or r.get("to") or r.get("topic_id")
                   or r.get("broker_id") or r.get("application_id") or r.get("node_id"))
            if src and dst and src != dst:
                g.add_edge(src, dst, type=r.get("type", type_label), qos_profile=r.get("qos_profile", {}))
    return g


def _load_realworld(scenario: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    """Topology, structural features (via the cache-independent
    _saag_structural_features), and fresh I*(v) truth from FaultInjector.

    Restricted to Application, the one population with real ground truth on
    these graphs — Broker I(v) is degenerate here (the cascade cannot express
    RUNS_ON/CONNECTS_TO failure) and Library is never fault-injected at all.
    """
    from reproduce.main_table import _saag_structural_features
    from saag.simulation.fault_injector import FaultInjector

    topology = json.loads((SCENARIOS_DIR / f"{scenario}.json").read_text())
    structural = _saag_structural_features(topology)

    nx_graph = _build_graph_safe(topology)
    fi = FaultInjector(nx_graph, seeds=[42])
    fi_result = fi.run()  # default node_types=["Application", "Broker"]
    truth = {
        nid: float(r.impact_score)
        for nid, r in fi_result.records.items()
        if r.node_type == "Application"
    }
    return topology, structural, truth


def _load(scenario: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    return (
        _load_realworld(scenario) if scenario in REALWORLD_SCENARIOS
        else _load_synthetic(scenario)
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _weights_at(w_r: float):
    """QualityWeights with composite (w_r, 1 - w_r); every other weight at
    its default (intra-dimension weights and r_alpha are not swept here)."""
    from saag.analysis.weight_calculator import QualityWeights

    return QualityWeights(q_reliability=w_r, q_maintainability=1.0 - w_r)


def _domain_w_r(domain: Optional[str]) -> Tuple[float, bool]:
    """The w_R a scenario's domain derives to, and whether derivation applied
    (False falls back to the static default, per derive_rm_weights)."""
    from saag.core.quality_model import derive_rm_weights

    weights, derived = derive_rm_weights(domain)
    return weights.q_reliability, derived


def _score(topology: Dict[str, Any], structural: Dict[str, Any], weights) -> Dict[str, float]:
    """Q(v) under *weights*, scored over the full typed graph.

    ``structural`` is accepted for call-site compatibility and deliberately not
    used: it holds features restricted to the Application+Library ``DEPENDS_ON``
    projection, on which Availability collapses to a constant across Applications
    (see ``reproduce.ahp_sensitivity.score_with_analyzer``). Sweeping w_R against
    a composite whose Reliability half is largely constant would measure the
    degeneracy rather than the reweighting.
    """
    from reproduce.ahp_sensitivity import score_with_analyzer
    from saag.analysis.analyzer import QualityAnalyzer

    return score_with_analyzer(topology, QualityAnalyzer(weights=weights))


_GRAPH_CACHE: Dict[int, Any] = {}


def _graph(topology: Dict[str, Any]):
    """NetworkX view of *topology*, built once per sweep.

    The w_R grid re-scores the same graph at every point, so rebuilding it per
    grid point was 21x the necessary work per scenario.
    """
    key = id(topology)
    if key not in _GRAPH_CACHE:
        from cli.loso_evaluate import _build_graph_from_json

        _GRAPH_CACHE[key] = _build_graph_from_json(topology)
    return _GRAPH_CACHE[key]


def _rho(
    pred: Dict[str, float], truth: Dict[str, float],
    topology: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Spearman rho on the Application population when the topology is known.

    Pooling node types repeats the aggregation hazard the headline tables guard
    against, and every table this sweep is read next to is Application-scored.
    """
    if topology is not None:
        from saag.evaluation.metrics import resolve_eval_keys

        common = resolve_eval_keys(
            pred, truth, _graph(topology), population="application")
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(scenarios: List[str], step: float) -> Dict[str, Any]:
    w_r_grid = [round(i * step, 4) for i in range(int(round(1.0 / step)) + 1)]
    rows: List[Dict[str, Any]] = []

    for scenario in scenarios:
        try:
            topology, structural, truth = _load(scenario)
        except Exception as exc:  # noqa: BLE001 - one bad scenario must not kill the run
            logger.warning("%s failed to load: %s", scenario, exc)
            continue
        if not structural or not truth:
            logger.warning("%s: empty structural features or truth, skipping", scenario)
            continue

        domain = _domain_of(topology)
        domain_w_r, derived = _domain_w_r(domain)

        curve: Dict[str, Optional[float]] = {}
        for w_r in w_r_grid:
            pred = _score(topology, structural, _weights_at(w_r))
            rho = _rho(pred, truth, topology)
            curve[f"{w_r:.2f}"] = None if rho is None else round(rho, 4)

        static_rho = _rho(_score(topology, structural, _weights_at(STATIC_W_R)), truth, topology)
        equal_rho = _rho(_score(topology, structural, _weights_at(EQUAL_W_R)), truth, topology)
        domain_rho = _rho(_score(topology, structural, _weights_at(domain_w_r)), truth, topology)

        # Kendall tau between domain-derived and static component rankings —
        # the headline number. w_R moves by at most 0.04 from the static 0.80
        # default across all six declared domains, so this is expected to sit
        # near 1.0: the composite weight barely perturbs the ranking even
        # though it visibly perturbs the score, and rho (rank-based) mostly
        # can't see the difference either.
        static_pred = _score(topology, structural, _weights_at(STATIC_W_R))
        domain_pred = _score(topology, structural, _weights_at(domain_w_r))
        common = sorted(set(static_pred) & set(domain_pred))
        tau = None
        if len(common) >= 3:
            t, _ = kendalltau(
                [static_pred[k] for k in common], [domain_pred[k] for k in common]
            )
            tau = None if np.isnan(t) else round(float(t), 4)

        row: Dict[str, Any] = {
            "scenario": scenario,
            "domain": domain,
            "n_labels": len(truth),
            "domain_w_r": round(domain_w_r, 4),
            "domain_derived_from_priorities": derived,
            "static_rho": None if static_rho is None else round(static_rho, 4),
            "equal_rho": None if equal_rho is None else round(equal_rho, 4),
            "domain_rho": None if domain_rho is None else round(domain_rho, 4),
            "kendall_tau_domain_vs_static_ranking": tau,
            "w_r_sweep": curve,
        }
        rows.append(row)
        print(f"  {scenario:34s} domain={domain!s:26s} w_r_domain={row['domain_w_r']:.4f}  "
              f"static={row['static_rho']}  equal={row['equal_rho']}  "
              f"domain={row['domain_rho']}  tau={tau}")

    return _summarise(rows, w_r_grid)


def _summarise(rows: List[Dict[str, Any]], w_r_grid: List[float]) -> Dict[str, Any]:
    def _mean(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    mean_static = _mean("static_rho")
    mean_equal = _mean("equal_rho")
    mean_domain = _mean("domain_rho")
    taus = [r["kendall_tau_domain_vs_static_ranking"] for r in rows
            if r.get("kendall_tau_domain_vs_static_ranking") is not None]

    # Mean sweep curve across scenarios, for plotting rho(Q(w_R), I*) vs w_R.
    mean_curve: Dict[str, Optional[float]] = {}
    for w_r in w_r_grid:
        key = f"{w_r:.2f}"
        vals = [r["w_r_sweep"][key] for r in rows if r["w_r_sweep"].get(key) is not None]
        mean_curve[key] = round(float(np.mean(vals)), 4) if vals else None

    interpretation: Dict[str, Any] = {
        "mean_rho": {
            "static": None if mean_static is None else round(mean_static, 4),
            "equal": None if mean_equal is None else round(mean_equal, 4),
            "domain_derived": None if mean_domain is None else round(mean_domain, 4),
        },
        "mean_kendall_tau_domain_vs_static_ranking": (
            round(float(np.mean(taus)), 4) if taus else None
        ),
        "mean_w_r_sweep_curve": mean_curve,
    }

    if mean_domain is not None and mean_equal is not None:
        interpretation["domain_beats_equal"] = mean_domain > mean_equal
        interpretation["domain_vs_equal_delta"] = round(mean_domain - mean_equal, 4)
    if mean_domain is not None and mean_static is not None:
        interpretation["domain_beats_static"] = mean_domain > mean_static
        interpretation["domain_vs_static_delta"] = round(mean_domain - mean_static, 4)

    interpretation["note"] = (
        "The headline number is mean_kendall_tau_domain_vs_static_ranking: across "
        "all six declared domains, domain-derived w_R sits within 0.04 of the "
        "static 0.80 default (range [0.70, 0.76]), so the two weightings are "
        "expected to rank components almost identically (tau close to 1.0) even "
        "before looking at rho against I*(v). mean_w_r_sweep_curve is the "
        "complete characterisation this script produces: rho(Q(w_R), I*) for "
        "w_R swept over its full range, with static/equal/domain_derived being "
        "three readouts of that one curve rather than independent arms. A small "
        "or negative domain_vs_static_delta is not evidence against Layer 3 "
        "domain derivation — the free parameter it moves is small enough that "
        "no weighting scheme confined to it can move rho much; the value of the "
        "domain derivation remains attributional (explaining criticality in "
        "stakeholder terms), not a ranking-improvement device, consistent with "
        "the scoping in docs/criticality.md §4.3."
    )

    return {"rows": rows, "w_r_grid": w_r_grid, "interpretation": interpretation}


def parse_args():
    p = argparse.ArgumentParser(description="Domain-derived vs. static/equal RM composite weighting")
    p.add_argument("--step", type=float, default=DEFAULT_STEP, help="w_R sweep step size")
    p.add_argument("--scenarios", nargs="+", default=None)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    scenarios = args.scenarios or (SYNTHETIC_SCENARIOS + REALWORLD_SCENARIOS)
    n_points = int(round(1.0 / args.step)) + 1
    print(f"Domain-weight comparison: {len(scenarios)} scenarios x {n_points}-point w_R sweep")
    report = run(scenarios, args.step)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "domain_weight_comparison.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}\n")
    for k, v in report["interpretation"].items():
        if k not in ("note", "mean_w_r_sweep_curve"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
