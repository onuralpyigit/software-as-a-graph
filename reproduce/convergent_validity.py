#!/usr/bin/env python3
"""
reproduce/convergent_validity.py — how much do the two oracles agree?
====================================================================

Produces ``results/convergent_validity.json``.

The project has **two** simulation oracles and they measure different things:

``I*(v)``      ``FaultInjector`` — mean subscriber feed-loss fraction. Supplies
               GNN training labels and backs Tables 3/4.
``I_comp(v)``  ``FailureSimulator`` — 0.35*reachability + 0.25*fragmentation
               + 0.25*throughput + 0.15*flow_disruption. Backs the Validate-stage
               gates and the four-dimensional IR/IM/IA/IV decomposition.

``docs/validation.md`` states they "correlate only loosely" but never quantifies
it, while the paper draws on both — the main table from I*, the stratified and
library analyses from I_comp. If a claim rests on one oracle and its supporting
argument on the other, the strength of that link is a number the paper owes the
reader. This script measures it.

Interpretation
--------------
High agreement is a convergent-validity argument: two differently-constructed
simulators rank components alike, so neither is likely to be an artifact of its
own construction. Low agreement bounds what either can support on its own, and
means results must name their oracle.

Usage
-----
    PYTHONPATH=. python reproduce/convergent_validity.py
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
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger("convergent_validity")

RESULTS_DIR = Path("results")


def _fault_injector_labels(scenario: str, seeds: List[int]) -> Dict[str, float]:
    """I*(v) — the labels that back the published tables."""
    from cli.loso_evaluate import _build_graph_from_json
    from saag.simulation.fault_injector import FaultInjector
    from reproduce.ahp_sensitivity import _load_topology

    graph = _build_graph_from_json(_load_topology(scenario))
    injector = FaultInjector(graph=graph, seeds=seeds, cascade_depth_limit=0)
    result = injector.run(node_types=["Application", "Broker", "Library"])
    return {nid: float(rec.impact_score) for nid, rec in result.records.items()}


def _failure_simulator_labels(scenario: str, layer: str = "system") -> Dict[str, float]:
    """I_comp(v) — the composite that backs the validation gates."""
    from saag.infrastructure.memory_repo import MemoryRepository
    from saag.simulation.failure_simulator import FailureSimulator
    from saag.simulation.graph import SimulationGraph
    from reproduce.ahp_sensitivity import _load_topology

    repo = MemoryRepository()
    repo.save_graph(_load_topology(scenario), clear=True)
    sim = FailureSimulator(SimulationGraph(repo.get_graph_data(include_raw=True)))
    return {
        r.target_id: float(r.impact.composite_impact)
        for r in sim.simulate_exhaustive(layer=layer, seed=42)
    }


def compare(scenario: str, seeds: List[int]) -> Dict[str, Any]:
    star = _fault_injector_labels(scenario, seeds)
    comp = _failure_simulator_labels(scenario)

    common = sorted(set(star) & set(comp))
    row: Dict[str, Any] = {
        "scenario": scenario,
        "n_fault_injector": len(star),
        "n_failure_simulator": len(comp),
        "n_common": len(common),
    }
    if len(common) < 3:
        row["note"] = "insufficient overlap"
        return row

    a = np.array([star[k] for k in common])
    b = np.array([comp[k] for k in common])
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        row["note"] = "constant oracle on the shared node set"
        return row

    rho, p = spearmanr(a, b)
    tau, _ = kendalltau(a, b)

    # Do they agree on *who is critical*, not just on the ordering? Top-K set
    # overlap is the property a practitioner actually consumes.
    k = max(1, int(round(len(common) * 0.20)))
    top_star = set(np.argsort(-a)[:k].tolist())
    top_comp = set(np.argsort(-b)[:k].tolist())
    jaccard = len(top_star & top_comp) / len(top_star | top_comp)

    row.update({
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p), 6),
        "kendall_tau": round(float(tau), 4),
        "topk_jaccard": round(float(jaccard), 4),
        "k": k,
        "scale_max_i_star": round(float(a.max()), 4),
        "scale_max_i_comp": round(float(b.max()), 4),
    })
    return row


def parse_args():
    p = argparse.ArgumentParser(description="Inter-oracle convergent validity")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "convergent_validity.json")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR)
    from reproduce.main_table import ALL_SCENARIOS

    scenarios = args.scenarios or ALL_SCENARIOS
    print(f"Convergent validity: I*(v) [FaultInjector] vs I_comp(v) [FailureSimulator]")
    print(f"  {len(scenarios)} scenarios\n")

    rows = []
    for scenario in scenarios:
        try:
            row = compare(scenario, args.seeds)
        except Exception as exc:      # noqa: BLE001
            logger.warning("%s failed: %s", scenario, exc)
            rows.append({"scenario": scenario, "error": str(exc)})
            continue
        rows.append(row)
        rho = row.get("spearman_rho")
        print(f"  {scenario:28} rho={rho if rho is not None else '—':>8}  "
              f"tau={row.get('kendall_tau', '—'):>8}  "
              f"topK-Jaccard={row.get('topk_jaccard', '—'):>7}  n={row.get('n_common', 0)}")

    rhos = [r["spearman_rho"] for r in rows if isinstance(r.get("spearman_rho"), float)]
    jac = [r["topk_jaccard"] for r in rows if isinstance(r.get("topk_jaccard"), float)]

    report = {
        "seeds": args.seeds,
        "per_scenario": rows,
        "summary": {
            "mean_spearman_rho": round(float(np.mean(rhos)), 4) if rhos else None,
            "min_spearman_rho": round(float(np.min(rhos)), 4) if rhos else None,
            "mean_topk_jaccard": round(float(np.mean(jac)), 4) if jac else None,
            "n_scenarios_measured": len(rhos),
            "note": (
                "I*(v) and I_comp(v) are different quantities on different scales; only "
                "their rank agreement is meaningful. Read this before attributing a "
                "result measured against one oracle to evidence gathered against the "
                "other — the two are used in different sections of the paper."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    for k, v in report["summary"].items():
        if k != "note":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
