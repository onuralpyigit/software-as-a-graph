#!/usr/bin/env python3
"""
reproduce/passive_stratum_labels.py — ground truth for Topics and host Nodes
============================================================================

Produces ``results/passive_stratum_labels.json``, the artifact behind the JSS
claim that label coverage now spans all five component types.

Why this exists
---------------
``FaultInjector`` could not express the failure of a Topic or a physical Node:
``compute_topic_loss`` derived a topic's feed loss from its *publishers and
routers* and never asked whether the topic itself was down, and RUNS_ON was not
indexed at all. Both types therefore scored a constant 0.0, which is why 30-47%
of each system carried no ground truth (JSS Section 8.3, Limitation L1). Two
additive branches in the cascade fixed that; existing Application, Broker and
Library labels are bit-identical either way, which
``tests/test_passive_strata.py`` pins.

What this reports, and why the baselines are not optional
---------------------------------------------------------
Coverage is not informativeness. A host outage kills everything deployed on the
host, so its impact could be little more than a restatement of how many
components sit there; a topic outage severs its subscribers, so its impact could
be a restatement of fan-out. Either would be full coverage carrying no signal.
This script therefore reports each stratum against the trivial count baseline
that would explain it away — colocation count for Node, subscriber fan-out for
Topic — so the strata are read against the null that makes them uninteresting
rather than in isolation.

Usage
-----
    PYTHONPATH=. python reproduce/passive_stratum_labels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import spearmanr

from reproduce._provenance import stamp
from saag.simulation.fault_injector import (
    FaultInjector,
    RECOMMENDED_SEEDS,
    _PubSubIndex,
)

#: The passive types this script labels. Application/Broker/Library are the
#: pre-existing strata and are untouched here.
NODE_TYPES = ["Topic", "Node"]

OUTPUT_PATH = Path("results/passive_stratum_labels.json")

#: Fewer than this many nodes in a stratum and a rank correlation is noise.
_MIN_STRATUM = 3


def _trivial_baseline(node_type: str, node_id: str, index: _PubSubIndex) -> float:
    """The count that would explain a stratum's labels away if it correlated."""
    if node_type == "Topic":
        return float(len(index.subscribers_of(node_id)))
    if node_type == "Node":
        return float(len(index.residents_of(node_id)))
    raise ValueError(f"no trivial baseline defined for {node_type!r}")


def _stratum(records, node_type: str, index: _PubSubIndex) -> Dict[str, Any]:
    rows = [(nid, rec.impact_score) for nid, rec in records.items()
            if rec.node_type == node_type]
    block: Dict[str, Any] = {"n": len(rows)}
    if len(rows) < _MIN_STRATUM:
        block["note"] = "stratum too small"
        return block

    impacts = [v for _, v in rows]
    baseline = [_trivial_baseline(node_type, nid, index) for nid, _ in rows]

    block.update({
        "n_nonzero": sum(1 for v in impacts if v > 1e-9),
        "min": round(min(impacts), 6),
        "max": round(max(impacts), 6),
        "mean": round(sum(impacts) / len(impacts), 6),
        "baseline": "subscriber_fanout" if node_type == "Topic" else "colocation_count",
    })
    # A degenerate stratum is a constant, and a constant has no rank order; say
    # so rather than reporting a correlation scipy would return as NaN.
    if len(set(impacts)) == 1 or len(set(baseline)) == 1:
        block["rho_vs_trivial_baseline"] = None
        block["note"] = "constant impact or constant baseline"
        return block

    block["rho_vs_trivial_baseline"] = round(
        float(spearmanr(baseline, impacts).correlation), 4)
    return block


def run(scenarios: List[str], seeds: List[int]) -> Dict[str, Any]:
    from saag.core.graph_io import build_graph_from_json as _build_graph_from_json
    from reproduce.ahp_sensitivity import _load_topology

    per_scenario: Dict[str, Any] = {}
    for scenario in scenarios:
        try:
            graph = _build_graph_from_json(_load_topology(scenario))
        except FileNotFoundError:
            print(f"  [skip] {scenario}: topology not found")
            continue

        index = _PubSubIndex(graph)
        # strict: a degenerate Topic or Node stratum here means the injection
        # modes regressed, not that the system has no passive infrastructure.
        result = FaultInjector(
            graph=graph, seeds=seeds, strict_labels=True
        ).run(node_types=NODE_TYPES)

        strata = {t: _stratum(result.records, t, index) for t in NODE_TYPES}
        n_total = graph.number_of_nodes()
        per_scenario[scenario] = {
            "n_graph_nodes": n_total,
            "n_passive_labelled": len(result.records),
            "passive_share_of_graph": round(len(result.records) / n_total, 4) if n_total else None,
            "strata": strata,
        }

        line = "  ".join(
            f"{t}: n={strata[t].get('n', 0)} "
            f"nonzero={strata[t].get('n_nonzero', 0)} "
            f"rho_trivial={strata[t].get('rho_vs_trivial_baseline')}"
            for t in NODE_TYPES
        )
        print(f"  {scenario:28s} {line}")

    return per_scenario


def main() -> None:
    from reproduce.main_table import ALL_SCENARIOS

    seeds = list(RECOMMENDED_SEEDS)
    print(f"Passive-stratum labels ({', '.join(NODE_TYPES)}) over "
          f"{len(ALL_SCENARIOS)} scenarios\n")
    per_scenario = run(ALL_SCENARIOS, seeds)

    rhos = {
        t: [s["strata"][t]["rho_vs_trivial_baseline"] for s in per_scenario.values()
            if s["strata"][t].get("rho_vs_trivial_baseline") is not None]
        for t in NODE_TYPES
    }
    summary = {
        "n_scenarios": len(per_scenario),
        "rho_vs_trivial_baseline_range": {
            t: [round(min(v), 4), round(max(v), 4)] if v else None
            for t, v in rhos.items()
        },
        "note": (
            "Coverage is not informativeness. These strata are reported against "
            "the count baseline that would explain them away (subscriber fan-out "
            "for Topic, colocation count for Node). No learned model is trained "
            "on them; the typing result of Section 7.2 does not extend here."
        ),
        "provenance": stamp(
            seeds=seeds,
            node_types=NODE_TYPES,
            qos_factor_mode="ladder",
            cascade_depth_limit=0,
            propagation_threshold=0.2,
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"summary": summary, "scenarios": per_scenario}, indent=2))
    print(f"\nrho vs trivial baseline: {summary['rho_vs_trivial_baseline_range']}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
