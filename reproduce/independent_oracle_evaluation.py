#!/usr/bin/env python3
"""
reproduce/independent_oracle_evaluation.py
===========================================

Evaluates rankers against independent simulation oracles:
  - I* (FaultInjector reachability cascade)
  - I_dyn (MessageFlowSimulator dynamic queue-flow traffic)
  - I_comp (FailureSimulator multi-criteria composite)

Also computes:
  - Active-stratum Spearman rho (rho_active, restricting to components with >0 oracle impact)
  - Analytic first-order I* approximation (publisher-share-weighted subscriber count)
  - Evaluates InDeg, Reach, Topo-QoS, GAT-P-QoS, and Analytic-I*

Usage:
  PYTHONPATH=. python3 reproduce/independent_oracle_evaluation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = ROOT / "data" / "scenarios"
DATA_BENCHMARKS = ROOT / "data" / "benchmarks"
RESULTS_DIR = ROOT / "results"
IDYN_CACHE = RESULTS_DIR / "idyn_scenario_cache_jss12.json"
ICOMP_CACHE = DATA_BENCHMARKS / "icomp_failure_simulator_cache_jss12.json"
OUTPUT_FILE = DATA_BENCHMARKS / "independent_oracle_evaluation.json"
OUTPUT_FILE_LEGACY = RESULTS_DIR / "independent_oracle_evaluation.json"

from reproduce.training_free_suite import (
    FOLDS,
    SYSTEMS,
    SEEDS,
    _flow,
    indeg,
    reach,
    topo_qos,
    labels_for,
    mean_ci,
    _mean,
    paired,
)
from reproduce.convergent_validity import _message_flow_labels, _failure_simulator_labels


def _compute_analytic_first_order(topo: Dict[str, Any]) -> Dict[str, float]:
    """Publisher-share-weighted subscriber count:
    For an Application v, sum_{t in pub(v)} |sub(t)| / |pub(t)|
    """
    topic_pubs: Dict[str, set] = {}
    topic_subs: Dict[str, set] = {}
    for r in topo.get("relationships", {}).get("publishes_to", []):
        p, t = str(r.get("from") or r.get("source")), str(r.get("to") or r.get("target"))
        topic_pubs.setdefault(t, set()).add(p)
    for r in topo.get("relationships", {}).get("subscribes_to", []):
        s, t = str(r.get("from") or r.get("source")), str(r.get("to") or r.get("target"))
        topic_subs.setdefault(t, set()).add(s)

    apps = [str(a["id"]) for a in topo.get("applications", [])]
    pred: Dict[str, float] = {}
    for a in apps:
        val = 0.0
        for t, pubs in topic_pubs.items():
            if a in pubs:
                subs = topic_subs.get(t, set())
                val += len(subs) / len(pubs)
        pred[a] = val
    return pred


def _get_idyn_for_scenario(scenario: str) -> Tuple[str, Dict[str, float]]:
    print(f"  [I_dyn] Simulating {scenario}...")
    t0 = time.time()
    labels = _message_flow_labels(
        scenario,
        duration=60.0,
        seed=42,
        max_candidates=30,
        qos_mode="full",
        target_utilization=0.65,
    )
    print(f"  [I_dyn] {scenario} finished in {time.time()-t0:.1f}s ({len(labels)} candidates)")
    return scenario, labels


def get_all_idyn_labels(scenarios: List[str], max_workers: int = 12) -> Dict[str, Dict[str, float]]:
    if IDYN_CACHE.exists():
        print(f"Loading cached I_dyn from {IDYN_CACHE}")
        return json.loads(IDYN_CACHE.read_text())

    print(f"Computing I_dyn for {len(scenarios)} scenarios with {max_workers} workers...")
    out: Dict[str, Dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_get_idyn_for_scenario, scenarios))
    for s, labels in results:
        out[s] = labels

    IDYN_CACHE.write_text(json.dumps(out, indent=2))
    print(f"Saved I_dyn cache to {IDYN_CACHE}")
    return out


def get_all_icomp_labels(scenarios: List[str]) -> Dict[str, Dict[str, float]]:
    """Genuine FailureSimulator multi-criteria failure simulation labels.
    Computes exhaustive component failure impacts across reachability,
    fragmentation, throughput, and flow disruption with default weights.
    """
    if ICOMP_CACHE.exists():
        print(f"Loading cached genuine I_comp from {ICOMP_CACHE}")
        return json.loads(ICOMP_CACHE.read_text())

    print(f"Computing genuine FailureSimulator I_comp for {len(scenarios)} scenarios...")
    out: Dict[str, Dict[str, float]] = {}
    for s in scenarios:
        t0 = time.time()
        labels = _failure_simulator_labels(s, qos=True, layer="Application")
        out[s] = labels
        print(f"  [I_comp] {s} finished in {time.time()-t0:.1f}s ({len(labels)} apps)")

    DATA_BENCHMARKS.mkdir(parents=True, exist_ok=True)
    ICOMP_CACHE.write_text(json.dumps(out, indent=2))
    print(f"Saved genuine I_comp cache to {ICOMP_CACHE}")
    return out


def score_ranker(
    pred: Dict[str, float],
    impact: Dict[str, float],
    apps: List[str],
) -> Dict[str, Optional[float]]:
    common = [a for a in apps if a in pred and a in impact]
    if len(common) < 3:
        return {"rho": None, "rho_active": None, "n": len(common), "n_active": 0}

    x = [pred[a] for a in common]
    y = [impact[a] for a in common]
    rho = None
    if len(set(x)) > 1 and len(set(y)) > 1:
        rho = float(spearmanr(x, y).correlation)

    active = [a for a in common if impact[a] > 0.0]
    rho_active = None
    if len(active) >= 3:
        x_act = [pred[a] for a in active]
        y_act = [impact[a] for a in active]
        if len(set(x_act)) > 1 and len(set(y_act)) > 1:
            rho_active = float(spearmanr(x_act, y_act).correlation)

    return {
        "rho": round(rho, 4) if rho is not None else None,
        "rho_active": round(rho_active, 4) if rho_active is not None else None,
        "n": len(common),
        "n_active": len(active),
    }


def main():
    scenario_keys = list(FOLDS.keys())
    print("Evaluating independent oracles across 12 folds...")

    idyn_labels = get_all_idyn_labels(scenario_keys, max_workers=min(12, os.cpu_count() or 4))
    icomp_labels = get_all_icomp_labels(scenario_keys)

    results_by_oracle = {
        "i_star": {},
        "i_dyn": {},
        "i_comp": {},
    }

    rankers = ["Analytic-I*", "InDeg", "Reach", "Topo-QoS"]

    per_fold_data = {s: {} for s in scenario_keys}

    for scenario in scenario_keys:
        topo_path = SCENARIOS_DIR / f"{scenario}.json"
        topo = json.loads(topo_path.read_text())
        apps = [str(a["id"]) for a in topo.get("applications", [])]

        flow = _flow(topo)
        preds = {
            "InDeg": indeg(flow),
            "Reach": reach(flow),
            "Topo-QoS": topo_qos(flow),
            "Analytic-I*": _compute_analytic_first_order(topo),
        }

        # Oracles
        istar = labels_for(topo_path)["impact"]
        idyn = idyn_labels.get(scenario, {})
        icomp = icomp_labels.get(scenario, {})

        oracles = {
            "i_star": istar,
            "i_dyn": idyn,
            "i_comp": icomp,
        }

        for o_name, o_labels in oracles.items():
            per_fold_data[scenario][o_name] = {}
            for r_name in rankers:
                sc = score_ranker(preds[r_name], o_labels, apps)
                per_fold_data[scenario][o_name][r_name] = sc

    # Aggregate summaries
    summary = {}
    for o_name in ["i_star", "i_dyn", "i_comp"]:
        summary[o_name] = {}
        for r_name in rankers:
            rhos = [per_fold_data[s][o_name][r_name]["rho"] for s in scenario_keys
                    if per_fold_data[s][o_name][r_name]["rho"] is not None]
            rhos_act = [per_fold_data[s][o_name][r_name]["rho_active"] for s in scenario_keys
                        if per_fold_data[s][o_name][r_name]["rho_active"] is not None]

            # Contrasts vs Topo-QoS
            deltas = []
            wins = 0
            for s in scenario_keys:
                r_val = per_fold_data[s][o_name][r_name]["rho"]
                t_val = per_fold_data[s][o_name]["Topo-QoS"]["rho"]
                if r_val is not None and t_val is not None:
                    deltas.append(r_val - t_val)
                    if r_val > t_val:
                        wins += 1
            mean_delta = _mean(deltas) if deltas else 0.0
            p_val = None
            if len(deltas) >= 5 and any(abs(d) > 1e-7 for d in deltas) and r_name != "Topo-QoS":
                try:
                    p_val = float(wilcoxon(deltas).pvalue)
                except Exception:
                    p_val = None

            summary[o_name][r_name] = {
                "mean_rho": round(_mean(rhos), 4) if rhos else None,
                "ci95_rho": mean_ci(rhos) if rhos else None,
                "mean_rho_active": round(_mean(rhos_act), 4) if rhos_act else None,
                "ci95_rho_active": mean_ci(rhos_act) if rhos_act else None,
                "delta_vs_topoqos": round(mean_delta, 4),
                "wins_vs_topoqos": f"{wins}/{len(deltas)}",
                "p_wilcoxon": round(p_val, 4) if p_val is not None else None,
                "n_folds": len(rhos),
            }

    print("\n" + "=" * 90)
    print("INDEPENDENT ORACLE EVALUATION SUMMARY (12 LOSO Folds)")
    print("=" * 90)
    for o_name in ["i_star", "i_dyn", "i_comp"]:
        print(f"\n--- Oracle: {o_name} ---")
        print(f"{'Ranker':15s} | {'Mean rho':10s} | {'95% CI':20s} | {'Active rho':12s} | {'Delta vs Topo':14s} | {'Wins':8s} | {'p (Wilcoxon)':12s}")
        print("-" * 105)
        for r_name in rankers:
            st = summary[o_name][r_name]
            m_rho = f"{st['mean_rho']:.3f}" if st['mean_rho'] is not None else "N/A"
            ci = f"[{st['ci95_rho'][0]:.3f}, {st['ci95_rho'][1]:.3f}]" if st['ci95_rho'] else "N/A"
            m_act = f"{st['mean_rho_active']:.3f}" if st['mean_rho_active'] is not None else "N/A"
            delta = f"{st['delta_vs_topoqos']:+.3f}"
            wins = st['wins_vs_topoqos']
            pval = f"{st['p_wilcoxon']:.4f}" if st['p_wilcoxon'] is not None else "---"
            print(f"{r_name:15s} | {m_rho:10s} | {ci:20s} | {m_act:12s} | {delta:14s} | {wins:8s} | {pval:12s}")

    output_data = {
        "summary": summary,
        "per_fold": per_fold_data,
        "provenance": {
            "folds": list(FOLDS.values()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    OUTPUT_FILE.write_text(json.dumps(output_data, indent=2))
    print(f"\nWrote full evaluation to {OUTPUT_FILE}")
    try:
        OUTPUT_FILE_LEGACY.write_text(json.dumps(output_data, indent=2))
        print(f"Also synced legacy output to {OUTPUT_FILE_LEGACY}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

