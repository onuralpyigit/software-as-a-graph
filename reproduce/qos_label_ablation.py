#!/usr/bin/env python3
"""
reproduce/qos_label_ablation.py — is the QoS lift real, or is it construct overlap?
==================================================================================

Produces ``results/qos_label_ablation.json``.

Why this exists
---------------
QoS already propagates into the *predictor*: w(t) inherits onto every edge, and
from there into QSPOF, the QoS-weighted in/out degrees and QoS-weighted
betweenness that RMAV and ``Topo-QoS`` are built from. Adding QoS to the *label*
therefore raises those predictors' correlation whether or not the model learned
anything — the label and the predictor now share a term.

A prediction-side ablation (``cli/validate_graph.py compare``) cannot detect
this: it varies the predictor while holding the label fixed. This script varies
the **label** instead and reports the delta per predictor.

Reading the result
------------------
For each scenario we compute I(v) under three QoS-factor modes and score the same
predictors against each:

``none``     QoS scaling disabled — the topology-only label.
``ladder``   The published x1.2 / x1.15 / x1.05 constants.
``wt``       The shared w(t) weight, so durability participates too.

The diagnostic is the *spread of the deltas across predictors*, not any single
delta. If Δρ is roughly uniform, the QoS-enriched label is simply a better target
and every method benefits. If Δρ is much larger for the QoS-weighted structural
baseline than for the QoS-masked ones, the lift is the label and the predictor
sharing w(t) — report it as construct overlap, not as model quality.

Usage
-----
    PYTHONPATH=. python reproduce/qos_label_ablation.py
    PYTHONPATH=. python reproduce/qos_label_ablation.py --scenarios atm_system
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

logger = logging.getLogger("qos_label_ablation")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = REPO_ROOT / "data" / "scenarios"
DEFAULT_OUT = REPO_ROOT / "results" / "qos_label_ablation.json"

#: Label arms, by FaultInjector.qos_factor_mode.
LABEL_ARMS = ("none", "ladder", "wt")

#: Seeds matching FaultInjector.RECOMMENDED_SEEDS.
SEEDS = [42, 123, 456, 789, 2024]

#: Node types the labeler covers (cli/simulate_graph.py's documented default).
NODE_TYPES = ["Application", "Broker", "Library"]


def _load_topology(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _labels(topology: Dict[str, Any], mode: str) -> Dict[str, float]:
    """I(v) for every candidate node under one QoS-factor mode."""
    from cli.loso_evaluate import _build_graph_from_json
    from saag.simulation.fault_injector import FaultInjector

    graph = _build_graph_from_json(topology)
    injector = FaultInjector(graph=graph, seeds=SEEDS, qos_factor_mode=mode)
    result = injector.run(node_types=NODE_TYPES)
    return {nid: rec.impact_score for nid, rec in result.records.items()}


def _predictors(topology: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Structural predictors spanning the QoS-aware / QoS-blind contrast.

    ``topo_qos`` is the one that shares w(t) with a QoS-enriched label, so it is
    the predictor whose delta exposes construct overlap. ``topo_bl`` and
    ``degree`` are QoS-blind controls.
    """
    import networkx as nx

    from cli.loso_evaluate import _build_graph_from_json

    graph = _build_graph_from_json(topology)
    undirected = nx.Graph()
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0) or 1.0)
        undirected.add_edge(u, v, weight=weight, distance=1.0 / (weight + 1e-9))

    if undirected.number_of_edges() == 0:
        return {}

    return {
        # QoS-blind: plain betweenness on the unweighted projection.
        "topo_bl": nx.betweenness_centrality(undirected),
        # QoS-aware: betweenness on the w(t)-derived distance graph. Shares its
        # QoS term with the label under the 'ladder'/'wt' arms.
        "topo_qos": nx.betweenness_centrality(undirected, weight="distance"),
        # QoS-blind control with no path structure at all.
        "degree": dict(nx.degree_centrality(undirected)),
    }


def _rho(pred: Dict[str, float], labels: Dict[str, float]) -> float:
    keys = sorted(set(pred) & set(labels))
    if len(keys) < 3:
        return float("nan")
    y_pred = np.array([pred[k] for k in keys], dtype=np.float64)
    y_true = np.array([labels[k] for k in keys], dtype=np.float64)
    if np.ptp(y_pred) == 0.0 or np.ptp(y_true) == 0.0:
        return float("nan")
    rho, _ = spearmanr(y_pred, y_true)
    return float(rho)


def run_scenario(path: Path) -> Dict[str, Any]:
    topology = _load_topology(path)
    predictors = _predictors(topology)
    if not predictors:
        return {"scenario": path.stem, "skipped": "no_edges"}

    labels = {arm: _labels(topology, arm) for arm in LABEL_ARMS}

    rho: Dict[str, Dict[str, float]] = {
        name: {arm: _rho(scores, labels[arm]) for arm in LABEL_ARMS}
        for name, scores in predictors.items()
    }

    # Delta each predictor gains when QoS enters the label.
    deltas = {
        name: {
            arm: rho[name][arm] - rho[name]["none"]
            for arm in LABEL_ARMS if arm != "none"
        }
        for name in rho
    }

    # The circularity signal: how unevenly the QoS-enriched label helps.
    spread = {}
    for arm in LABEL_ARMS:
        if arm == "none":
            continue
        vals = [d[arm] for d in deltas.values() if not np.isnan(d[arm])]
        spread[arm] = {
            "max_minus_min": float(np.ptp(vals)) if vals else float("nan"),
            "topo_qos_minus_topo_bl": float(
                deltas.get("topo_qos", {}).get(arm, float("nan"))
                - deltas.get("topo_bl", {}).get(arm, float("nan"))
            ),
        }

    label_shift = {
        arm: float(
            np.mean([
                abs(labels[arm][k] - labels["none"][k])
                for k in labels["none"]
            ])
        )
        for arm in LABEL_ARMS
    }

    return {
        "scenario": path.stem,
        "n_labeled": len(labels["none"]),
        "spearman_by_predictor_and_label_arm": rho,
        "delta_vs_topology_only_label": deltas,
        "delta_spread_across_predictors": spread,
        "mean_absolute_label_shift": label_shift,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--scenarios", nargs="*", default=None,
        help="Scenario stems to run (default: every *_system.json under data/scenarios).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = (
        [SCENARIO_DIR / f"{s}.json" for s in args.scenarios]
        if args.scenarios
        else sorted(SCENARIO_DIR.glob("*_system.json"))
    )
    missing = [p for p in paths if not p.exists()]
    if missing:
        logger.error("Missing scenario files: %s", ", ".join(str(p) for p in missing))
        return 2

    results = []
    for path in paths:
        logger.info("→ %s", path.stem)
        results.append(run_scenario(path))

    payload = {
        "label_arms": list(LABEL_ARMS),
        "seeds": SEEDS,
        "node_types": NODE_TYPES,
        "scenarios": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", args.out)

    _print_summary(results)
    return 0


def _print_summary(results: List[Dict[str, Any]]) -> None:
    logger.info("")
    logger.info("%-28s %10s %10s %10s", "scenario", "Δρ topo_qos", "Δρ topo_bl", "spread")
    logger.info("-" * 62)
    for res in results:
        if "skipped" in res:
            logger.info("%-28s %10s", res["scenario"], res["skipped"])
            continue
        deltas = res["delta_vs_topology_only_label"]
        spread = res["delta_spread_across_predictors"]["ladder"]["max_minus_min"]
        logger.info(
            "%-28s %10.4f %10.4f %10.4f",
            res["scenario"],
            deltas.get("topo_qos", {}).get("ladder", float("nan")),
            deltas.get("topo_bl", {}).get("ladder", float("nan")),
            spread,
        )
    logger.info("")
    logger.info(
        "A large positive 'Δρ topo_qos' alongside a near-zero 'Δρ topo_bl' means the "
        "QoS-enriched label is lifting the predictor that already shares w(t) with it "
        "— construct overlap, not model quality."
    )


if __name__ == "__main__":
    raise SystemExit(main())
