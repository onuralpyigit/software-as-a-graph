#!/usr/bin/env python3
"""
reproduce/domain_weight_comparison.py — does domain-derived weighting rank better?
====================================================================================

Measures whether deriving the RMAV composite weights from a scenario's declared
deployment domain (saag.core.quality_model.derive_rmav_weights — Layer 3 of the
layered SQuaRE quality model, docs/criticality.md §3.5) improves ranking
correlation against I*(v), compared to the static default, equal weights, and
the AHP lambda=0.7 vector.

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
intact per the existing scoping of the RMAV composite (docs/criticality.md §4.3).

Ten scenarios: the seven synthetic scenarios of reproduce/main_table.py
ALL_SCENARIOS (LOSO-cached, scored via the same _compute_rmav_from_structural /
_load_scenario_data path as reproduce/ahp_sensitivity.py) plus the three
real-world graphs used for RQ4 (uncached; scored via a direct structural-feature
build and a fresh FaultInjector run, since cli.loso_evaluate._build_graph_from_json
crashes on real-world Broker/Node entries that already carry a raw `type` field
— a separate, pre-existing defect, not touched here).

Usage
-----
    PYTHONPATH=. python reproduce/domain_weight_comparison.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import spearmanr, wilcoxon

logger = logging.getLogger("domain_weight_comparison")

RESULTS_DIR = Path("results")
SCENARIOS_DIR = Path("data/scenarios")

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

WEIGHTINGS = ["static", "equal", "domain_derived", "ahp_070"]


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

def _weights_for(kind: str, domain: Optional[str]):
    from saag.analysis.weight_calculator import QualityWeights, AHPProcessor
    from saag.core.quality_model import derive_rmav_weights

    if kind == "static":
        return QualityWeights(), None
    if kind == "equal":
        return dataclass_replace(
            QualityWeights(),
            q_reliability=0.25, q_maintainability=0.25,
            q_availability=0.25, q_security=0.25,
        ), None
    if kind == "ahp_070":
        return AHPProcessor(shrinkage_factor=0.7).compute_weights(), None
    if kind == "domain_derived":
        weights, derived = derive_rmav_weights(domain)
        return weights, derived
    raise ValueError(kind)


def _score(topology: Dict[str, Any], structural: Dict[str, Any], weights) -> Dict[str, float]:
    from reproduce.main_table import _compute_rmav_from_structural
    from saag.analysis.analyzer import QualityAnalyzer

    scored = _compute_rmav_from_structural(
        topology, structural, analyzer=QualityAnalyzer(weights=weights)
    )
    return {nid: float(v.get("composite", 0.0)) for nid, v in scored.items()}


def _rho(pred: Dict[str, float], truth: Dict[str, float]) -> Optional[float]:
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

def run(scenarios: List[str]) -> Dict[str, Any]:
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
        row: Dict[str, Any] = {"scenario": scenario, "domain": domain, "n_labels": len(truth)}

        for kind in WEIGHTINGS:
            weights, derived = _weights_for(kind, domain)
            pred = _score(topology, structural, weights)
            rho = _rho(pred, truth)
            row[kind] = None if rho is None else round(rho, 4)
            if kind == "domain_derived":
                row["domain_derived_from_priorities"] = derived

        rows.append(row)
        print(f"  {scenario:34s} domain={domain!s:26s} "
              f"static={row['static']}  equal={row['equal']}  "
              f"domain={row['domain_derived']}  ahp070={row['ahp_070']}")

    return _summarise(rows)


def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    means: Dict[str, float] = {}
    for kind in WEIGHTINGS:
        vals = [r[kind] for r in rows if r.get(kind) is not None]
        if vals:
            means[kind] = float(np.mean(vals))

    interpretation: Dict[str, Any] = {"mean_rho": {k: round(v, 4) for k, v in means.items()}}

    def _paired_test(a_key: str, b_key: str) -> Dict[str, Any]:
        paired = [(r[a_key], r[b_key]) for r in rows
                  if r.get(a_key) is not None and r.get(b_key) is not None]
        if len(paired) < 3:
            return {}
        a_vals, b_vals = zip(*paired)
        diffs = np.array(b_vals) - np.array(a_vals)
        try:
            _, p = wilcoxon(diffs)
            p = round(float(p), 4)
        except ValueError:
            p = None  # all-zero diffs
        wins = int((diffs > 0).sum())
        losses = int((diffs < 0).sum())
        ties = int((diffs == 0).sum())
        return {
            "mean_delta": round(float(np.mean(diffs)), 4),
            "wilcoxon_p": p,
            "wins": wins, "losses": losses, "ties": ties, "n": len(diffs),
        }

    interpretation["domain_vs_static"] = _paired_test("static", "domain_derived")
    interpretation["domain_vs_equal"] = _paired_test("equal", "domain_derived")
    # Back-compat flat keys for anything reading the old shape.
    if interpretation["domain_vs_static"]:
        interpretation["domain_vs_static_wilcoxon_p"] = interpretation["domain_vs_static"]["wilcoxon_p"]
        interpretation["domain_vs_static_mean_delta"] = interpretation["domain_vs_static"]["mean_delta"]

    if "domain_derived" in means and "equal" in means:
        interpretation["domain_beats_equal"] = means["domain_derived"] > means["equal"]
        interpretation["domain_vs_equal_delta"] = round(means["domain_derived"] - means["equal"], 4)
    if "domain_derived" in means and "static" in means:
        interpretation["domain_beats_static"] = means["domain_derived"] > means["static"]
        interpretation["domain_vs_static_delta"] = round(means["domain_derived"] - means["static"], 4)

    interpretation["note"] = (
        "Read wilcoxon_p before quoting either delta: with 10 heterogeneous "
        "scenarios both paired tests are under-powered, and a non-significant "
        "p means the sign of that delta should not be over-interpreted. The "
        "two comparisons answer different questions: domain_vs_static asks "
        "whether the derivation beats the current shipped default; "
        "domain_vs_equal asks the sharper question of whether it beats the "
        "hardest baseline to beat (equal weights already outperform the AHP "
        "judgement by 0.111 rho on the shrinkage sweep). A win over static "
        "that is a coin flip against equal weights means the derivation "
        "recovers most, but not all, of what equal weighting already gets for "
        "free — real, but not yet evidence that this specific domain-priority "
        "encoding is the better construction."
    )

    return {"rows": rows, "interpretation": interpretation}


def main():
    logging.basicConfig(level=logging.WARNING)
    scenarios = SYNTHETIC_SCENARIOS + REALWORLD_SCENARIOS
    print(f"Domain-weight comparison: {len(scenarios)} scenarios x {len(WEIGHTINGS)} weightings")
    report = run(scenarios)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "domain_weight_comparison.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}\n")
    for k, v in report["interpretation"].items():
        if k != "note":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
