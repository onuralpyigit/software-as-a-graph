#!/usr/bin/env python3
"""
reproduce/training_free_suite.py — Amendment 7: training-free baselines and QoS controls
=======================================================================================

Everything registered in PREREGISTRATION.md Amendment 7, computed without PyTorch
and without Neo4j:

* ``gate``        — rebuild I*(v) and Topo-QoS from the committed scenarios and check
                    them against the published per-fold values (Supplementary S22).
* ``baselines``   — Reach, Reach-QoS, CDI, InDeg against Topo-QoS (LOSO folds and the
                    five system models), full and active strata, plus the
                    inert-vs-active rule.
* ``controls``    — Topo-Mult (constant topic weight) and Topo-QoS-Perm (QoS profiles
                    permuted across topics; labels untouched).
* ``oracle``      — I*(v) relabelled over propagation threshold x depth-damping step.
* ``descriptives``— size, zero share, projection density and tie fraction per graph.
* ``qos-indep``   — Topo / Topo-QoS on the corpus regenerated with
                    ``qos_affinity: false`` (see tools/generation/generator.py).

Labels are cached under output/tf_labels/ keyed by scenario and oracle setting.
Every artifact is written to results/ with a provenance stamp.

Usage:
    PYTHONPATH=. python reproduce/training_free_suite.py gate
    PYTHONPATH=. python reproduce/training_free_suite.py all
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.stats import spearmanr, wilcoxon

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import stamp  # noqa: E402
from saag.analysis.structural_analyzer import StructuralAnalyzer  # noqa: E402
from saag.core.graph_io import build_graph_from_json, load_graph  # noqa: E402
from saag.evaluation.metrics import compute_inductive_metrics  # noqa: E402
from saag.prediction.structural_predictor import (  # noqa: E402
    TopoPredictor,
    TopoQoSPredictor,
    derive_flow_projection,
)
from saag.simulation.fault_injector import FaultInjector  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = ROOT / "data" / "scenarios"
LABEL_CACHE = ROOT / "output" / "tf_labels"
RESULTS = ROOT / "results"
SEEDS = [42, 123, 456, 789, 2024]
NODE_TYPES = ["Application", "Broker", "Library"]

# The twelve LOSO folds (scripts/populate_loso_cache.sh) with the display names the
# manuscript uses, and the five hand-authored system models.
FOLDS: Dict[str, str] = {
    "atm_system": "ATM",
    "av_system": "AV System",
    "enterprise_system": "Enterprise",
    "financial_trading_system": "Financial Trading",
    "healthcare_system": "Healthcare",
    "hub_and_spoke_system": "Enterprise Integration (ESB)",
    "industrial_scada_system": "Industrial SCADA",
    "iot_smart_city_system": "IoT Smart City",
    "logistics_fleet_system": "Logistics Fleet",
    "microservices_system": "Microservices",
    "realtime_gaming_system": "Real-Time Gaming",
    "telecom_ran_system": "Telecom RAN",
}
SYSTEMS: Dict[str, str] = {
    "realworld_autoware_ros2": "Autoware.universe (ROS 2)",
    "realworld_edgex": "EdgeX Foundry",
    "realworld_homeassistant": "Home Assistant",
    "realworld_cloud_microservices": "Online Boutique (pub-sub model)",
    "realworld_trainticket": "Train-Ticket",
}

# Published per-fold values the gate and the decision rules read. Supplementary S22
# ("as run" columns) and S23 (CPU sweeps), keyed by display name.
PUBLISHED_TOPO = {
    "ATM": 0.256, "AV System": 0.468, "Enterprise": 0.431, "Financial Trading": 0.237,
    "Healthcare": 0.077, "Enterprise Integration (ESB)": 0.112, "Industrial SCADA": 0.608,
    "IoT Smart City": 0.251, "Logistics Fleet": 0.576, "Microservices": 0.229,
    "Real-Time Gaming": 0.377, "Telecom RAN": 0.562,
}
PUBLISHED_TOPO_QOS = {
    "ATM": 0.311, "AV System": 0.753, "Enterprise": 0.795, "Financial Trading": 0.586,
    "Healthcare": 0.369, "Enterprise Integration (ESB)": 0.430, "Industrial SCADA": 0.650,
    "IoT Smart City": 0.351, "Logistics Fleet": 0.741, "Microservices": 0.265,
    "Real-Time Gaming": 0.810, "Telecom RAN": 0.576,
}
PUBLISHED_HGT_QOS_CPU = {
    "ATM": 0.523, "AV System": 0.704, "Enterprise": 0.426, "Financial Trading": 0.695,
    "Healthcare": 0.730, "Enterprise Integration (ESB)": 0.548, "Industrial SCADA": 0.684,
    "IoT Smart City": 0.688, "Logistics Fleet": 0.771, "Microservices": 0.475,
    "Real-Time Gaming": 0.789, "Telecom RAN": 0.427,
}
PUBLISHED_GAT_QOS_CPU = {
    "ATM": 0.506, "AV System": 0.732, "Enterprise": 0.407, "Financial Trading": 0.713,
    "Healthcare": 0.798, "Enterprise Integration (ESB)": 0.630, "Industrial SCADA": 0.721,
    "IoT Smart City": 0.720, "Logistics Fleet": 0.654, "Microservices": 0.479,
    "Real-Time Gaming": 0.685, "Telecom RAN": 0.574,
}


# ── labels ────────────────────────────────────────────────────────────────────

def _topology(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def labels_for(path: Path, theta: float = 0.2, damp_step: float = 0.15,
               cache: bool = True) -> Dict[str, Any]:
    """I*(v) with the published settings unless overridden; cached per setting."""
    key = f"{path.stem}__th{theta:.2f}_ds{damp_step:.2f}.json"
    cpath = LABEL_CACHE / key
    if cache and cpath.exists():
        return json.loads(cpath.read_text())
    injector = FaultInjector(
        graph=load_graph(path), seeds=SEEDS, cascade_depth_limit=0,
        propagation_threshold=theta, qos_factor_mode="ladder",
        depth_damp_step=damp_step,
    )
    result = injector.run(node_types=NODE_TYPES)
    out = {
        "impact": {nid: float(r.impact_score) for nid, r in result.records.items()},
        "per_seed": {nid: {str(s): float(v) for s, v in r.seed_impact_scores.items()}
                     for nid, r in result.records.items()},
    }
    if cache:
        LABEL_CACHE.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(out))
    return out


# ── predictors ────────────────────────────────────────────────────────────────

def _flow(topology: Dict[str, Any]) -> nx.DiGraph:
    return derive_flow_projection(topology)


def topo_qos(flow: nx.DiGraph) -> Dict[str, float]:
    # Articulation term zero, exactly as the evaluated implementation ran (S22).
    return TopoQoSPredictor().predict(flow, {})


def topo_unweighted(flow: nx.DiGraph) -> Dict[str, float]:
    # Unweighted betweenness on the projection. The published Topo read Neo4j-cached
    # betweenness instead, so this arm is an approximation of the published one.
    return TopoPredictor().predict(flow, {})


def reach(flow: nx.DiGraph) -> Dict[str, float]:
    n = max(1, flow.number_of_nodes() - 1)
    return {str(v): len(nx.ancestors(flow, v)) / n for v in flow.nodes}


def reach_qos(flow: nx.DiGraph) -> Dict[str, float]:
    """Sum over transitive dependents u of the best-path product of qos_weight u→v."""
    rev = flow.reverse(copy=False)
    out: Dict[str, float] = {}
    for v in flow.nodes:
        best = {v: 1.0}
        heap = [(-1.0, v)]
        while heap:
            negp, x = heapq.heappop(heap)
            p = -negp
            if p < best.get(x, 0.0):
                continue
            for y in rev.successors(x):
                w = float(flow[y][x].get("qos_weight", 1.0))
                q = p * max(min(w, 1.0), 1e-6)
                if q > best.get(y, 0.0):
                    best[y] = q
                    heapq.heappush(heap, (-q, y))
        out[str(v)] = float(sum(p for u, p in best.items() if u != v))
    return out


def cdi(flow: nx.DiGraph) -> Dict[str, float]:
    scores = StructuralAnalyzer._compute_continuous_ap_scores(flow)
    return {str(n): float(s.get("cdi", 0.0)) for n, s in scores.items()}


def indeg(flow: nx.DiGraph) -> Dict[str, float]:
    return {str(v): float(d) for v, d in flow.in_degree()}


def _with_topic_weight(topology: Dict[str, Any], c: float) -> Dict[str, Any]:
    t = copy.deepcopy(topology)
    for topic in t.get("topics", []):
        topic["weight"] = c
    return t


def _with_permuted_qos(topology: Dict[str, Any], seed: int) -> Dict[str, Any]:
    t = copy.deepcopy(topology)
    topics = t.get("topics", [])
    qos = [tp.get("qos") for tp in topics]
    random.Random(seed).shuffle(qos)
    for tp, q in zip(topics, qos):
        tp["qos"] = q
    return t


# ── scoring ───────────────────────────────────────────────────────────────────

def score(pred: Dict[str, float], impact: Dict[str, float], graph: nx.DiGraph) -> Dict[str, Any]:
    full = compute_inductive_metrics(pred, impact, graph, population="application")
    apps = [n for n, d in graph.nodes(data=True)
            if d.get("type") == "Application" and n in impact and n in pred]
    active = [n for n in apps if impact[n] > 0]
    rho_active = None
    if len(active) >= 3:
        a = [pred[n] for n in active]
        b = [impact[n] for n in active]
        if len(set(a)) > 1 and len(set(b)) > 1:
            rho_active = float(spearmanr(a, b).correlation)
    return {
        "rho": _f(full.get("spearman_rho")),
        "rho_active": rho_active,
        "overlap_at_k": _f(full.get("overlap_at_k")),
        "pr_auc": _f(full.get("pr_auc")),
        "n_apps": len(apps),
        "n_active": len(active),
    }


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def paired(a: List[float], b: List[float], seed: int = 0, B: int = 2000) -> Dict[str, Any]:
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(d, size=len(d), replace=True))) for _ in range(B)]
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return {
        "delta": float(np.mean(d)),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "won": int(np.sum(d > 0)),
        "n": int(len(d)),
        "p": p,
    }


def holm(ps: Dict[str, float]) -> Dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def _write(name: str, payload: Dict[str, Any], **config: Any) -> Path:
    RESULTS.mkdir(exist_ok=True)
    payload["provenance"] = stamp(script="reproduce/training_free_suite.py", **config)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"wrote {path}")
    return path


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


# ── subcommands ───────────────────────────────────────────────────────────────

def cmd_gate(_: argparse.Namespace) -> int:
    rows, worst = {}, 0.0
    for sid, name in FOLDS.items():
        path = SCENARIOS_DIR / f"{sid}.json"
        topo = _topology(path)
        lab = labels_for(path)
        graph = build_graph_from_json(topo)
        flow = _flow(topo)
        r = score(topo_qos(flow), lab["impact"], graph)["rho"]
        diff = abs(r - PUBLISHED_TOPO_QOS[name])
        worst = max(worst, diff)
        # Label test-retest: pairwise Spearman across seeds on Applications.
        apps = sorted(n for n, d in graph.nodes(data=True)
                      if d.get("type") == "Application" and n in lab["per_seed"])
        per = [[lab["per_seed"][n][str(s)] for n in apps] for s in SEEDS]
        rr = [spearmanr(per[i], per[j]).correlation
              for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]
        rr = [x for x in rr if not math.isnan(x)]
        rows[name] = {"rebuilt": r, "published": PUBLISHED_TOPO_QOS[name], "abs_diff": diff,
                      "test_retest_min": float(min(rr)) if rr else None,
                      "test_retest_median": float(np.median(rr)) if rr else None}
        print(f"{name:30s} rebuilt={r:.3f} published={PUBLISHED_TOPO_QOS[name]:.3f} "
              f"retest_min={rows[name]['test_retest_min']:.3f}")
    mean = _mean([v["rebuilt"] for v in rows.values()])
    passed = worst < 0.0005
    print(f"mean rebuilt Topo-QoS = {mean:.3f}; max |diff| = {worst:.4f}; gate "
          f"{'PASSED' if passed else 'FAILED'}")
    _write("tf_reproduction_gate.json",
           {"per_fold": rows, "mean_rebuilt": mean, "max_abs_diff": worst, "passed": passed},
           experiment="gate")
    return 0 if passed else 1


PREDICTORS = {
    "Topo-QoS": topo_qos,
    "Topo (projection)": topo_unweighted,
    "Reach": reach,
    "Reach-QoS": reach_qos,
    "CDI": cdi,
    "InDeg": indeg,
}
NEW_RANKERS = ["Reach", "Reach-QoS", "CDI", "InDeg"]


def _eval_set(ids: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sid, name in ids.items():
        path = SCENARIOS_DIR / f"{sid}.json"
        topo = _topology(path)
        lab = labels_for(path)["impact"]
        graph = build_graph_from_json(topo)
        flow = _flow(topo)
        row: Dict[str, Any] = {}
        for pname, fn in PREDICTORS.items():
            row[pname] = score(fn(flow), lab, graph)
        # Inert-vs-active rule: predict active iff the component has a dependent.
        rch = reach(flow)
        apps = [n for n, d in graph.nodes(data=True)
                if d.get("type") == "Application" and n in lab]
        y = np.array([lab[n] > 0 for n in apps])
        yhat = np.array([rch.get(n, 0.0) > 0 for n in apps])
        tp = int(np.sum(y & yhat)); fp = int(np.sum(~y & yhat))
        fn_ = int(np.sum(y & ~yhat)); tn = int(np.sum(~y & ~yhat))
        tpr = tp / (tp + fn_) if tp + fn_ else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        row["inert_rule"] = {
            "accuracy": (tp + tn) / max(1, len(apps)),
            "balanced_accuracy": (tpr + tnr) / 2,
            "f1_active": (2 * tp / (2 * tp + fp + fn_)) if tp else 0.0,
            "tp": tp, "fp": fp, "fn": fn_, "tn": tn,
        }
        out[name] = row
        print(f"{name:32s} " + " ".join(f"{p}={row[p]['rho']:.3f}" for p in PREDICTORS
                                          if row[p]['rho'] is not None))
    return out


def cmd_baselines(_: argparse.Namespace) -> int:
    loso = _eval_set(FOLDS)
    systems = _eval_set(SYSTEMS)
    names = list(FOLDS.values())

    def col(p: str, key: str = "rho", src=loso) -> List[Optional[float]]:
        return [src[n][p][key] for n in src]

    summary = {
        p: {
            "loso_mean_rho": _mean(col(p)),
            "loso_mean_rho_active": _mean(col(p, "rho_active")),
            "loso_mean_overlap": _mean(col(p, "overlap_at_k")),
            "systems_mean_rho": _mean(col(p, src=systems)),
            "systems_mean_rho_active": _mean(col(p, "rho_active", systems)),
            "systems_mean_overlap": _mean(col(p, "overlap_at_k", systems)),
            "systems_mean_pr_auc": _mean(col(p, "pr_auc", systems)),
        }
        for p in PREDICTORS
    }
    tq = [loso[n]["Topo-QoS"]["rho"] for n in names]
    contrasts = {p: paired([loso[n][p]["rho"] for n in names], tq) for p in NEW_RANKERS}
    ph = holm({p: c["p"] for p, c in contrasts.items()})
    for p in contrasts:
        contrasts[p]["p_holm"] = ph[p]
    vs_learned = {
        p: {
            "vs_HGT-QoS_cpu": paired([loso[n][p]["rho"] for n in names],
                                     [PUBLISHED_HGT_QOS_CPU[n] for n in names]),
            "vs_GAT-QoS_cpu": paired([loso[n][p]["rho"] for n in names],
                                     [PUBLISHED_GAT_QOS_CPU[n] for n in names]),
        }
        for p in NEW_RANKERS
    }
    best = max(NEW_RANKERS, key=lambda p: summary[p]["loso_mean_rho"] or -1)
    r1 = (summary[best]["loso_mean_rho"] or -1) >= 0.622
    inert = {
        "loso_mean_accuracy": _mean([loso[n]["inert_rule"]["accuracy"] for n in names]),
        "loso_mean_balanced_accuracy": _mean(
            [loso[n]["inert_rule"]["balanced_accuracy"] for n in names]),
        "loso_mean_f1_active": _mean([loso[n]["inert_rule"]["f1_active"] for n in names]),
        "systems_mean_balanced_accuracy": _mean(
            [systems[n]["inert_rule"]["balanced_accuracy"] for n in systems]),
    }
    for p, s in summary.items():
        print(f"{p:18s} LOSO {s['loso_mean_rho']:.3f} (active {s['loso_mean_rho_active']:.3f}) "
              f"| systems {s['systems_mean_rho']:.3f}")
    for p, c in contrasts.items():
        print(f"{p:10s} vs Topo-QoS {c['delta']:+.3f} {c['ci95']} {c['won']}/12 "
              f"p={c['p']:.4f} holm={c['p_holm']:.4f}")
    print(f"best new ranker {best}; R1 {'TRIGGERED' if r1 else 'not triggered'}")
    print("inert rule:", inert)
    _write("tf_baselines.json", {
        "per_fold": loso, "per_system": systems, "summary": summary,
        "contrasts_vs_topo_qos": contrasts, "vs_published_learned": vs_learned,
        "inert_rule": inert, "decision_R1": {"best": best, "triggered": r1},
    }, experiment="baselines")
    return 0


def cmd_controls(_: argparse.Namespace) -> int:
    per_fold: Dict[str, Any] = {}
    for sid, name in FOLDS.items():
        path = SCENARIOS_DIR / f"{sid}.json"
        topo = _topology(path)
        lab = labels_for(path)["impact"]
        graph = build_graph_from_json(topo)
        mult = score(topo_qos(_flow(_with_topic_weight(topo, 0.5))), lab, graph)["rho"]
        perms = [score(topo_qos(_flow(_with_permuted_qos(topo, s))), lab, graph)["rho"]
                 for s in range(20)]
        per_fold[name] = {
            "Topo-QoS": score(topo_qos(_flow(topo)), lab, graph)["rho"],
            "Topo-Mult": mult,
            "Topo-QoS-Perm_mean": float(np.mean(perms)),
            "Topo-QoS-Perm_sd": float(np.std(perms)),
            "Topo (published)": PUBLISHED_TOPO[name],
        }
        print(f"{name:30s} " + " ".join(f"{k}={v:.3f}" for k, v in per_fold[name].items()))
    names = list(FOLDS.values())
    gain = 0.553 - 0.349
    summ = {}
    for arm in ("Topo-QoS", "Topo-Mult", "Topo-QoS-Perm_mean"):
        m = _mean([per_fold[n][arm] for n in names])
        summ[arm] = {"mean_rho": m, "retained_gain": (m - 0.349) / gain}
    tq = [per_fold[n]["Topo-QoS"] for n in names]
    contr = {
        "Topo-QoS vs Topo-Mult": paired(tq, [per_fold[n]["Topo-Mult"] for n in names]),
        "Topo-QoS vs Topo-QoS-Perm": paired(tq, [per_fold[n]["Topo-QoS-Perm_mean"] for n in names]),
        "Topo-Mult vs Topo (published)": paired([per_fold[n]["Topo-Mult"] for n in names],
                                                [PUBLISHED_TOPO[n] for n in names]),
    }
    r2 = any(summ[a]["retained_gain"] >= 0.5 for a in ("Topo-Mult", "Topo-QoS-Perm_mean"))
    print(json.dumps(summ, indent=1)); print(json.dumps(contr, indent=1))
    print(f"R2 {'TRIGGERED' if r2 else 'not triggered'}")
    _write("qos_attribution_controls.json", {
        "per_fold": per_fold, "summary": summ, "contrasts": contr,
        "decision_R2": {"triggered": r2},
    }, experiment="controls", constant_topic_weight=0.5, permutations=20)
    return 0


def cmd_oracle(_: argparse.Namespace) -> int:
    grid = [(th, ds) for th in (0.1, 0.2, 0.3) for ds in (0.10, 0.15, 0.20)]
    per_setting: Dict[str, Any] = {}
    for th, ds in grid:
        key = f"theta={th:.1f},step={ds:.2f}"
        agree, rhos = [], []
        for sid, name in FOLDS.items():
            path = SCENARIOS_DIR / f"{sid}.json"
            topo = _topology(path)
            graph = build_graph_from_json(topo)
            base = labels_for(path)["impact"]
            lab = labels_for(path, theta=th, damp_step=ds)["impact"]
            apps = sorted(n for n, d in graph.nodes(data=True)
                          if d.get("type") == "Application" and n in lab)
            a = spearmanr([base[n] for n in apps], [lab[n] for n in apps]).correlation
            agree.append(float(a))
            rhos.append(score(topo_qos(_flow(topo)), lab, graph)["rho"])
        per_setting[key] = {"label_agreement_mean": float(np.nanmean(agree)),
                            "label_agreement_min": float(np.nanmin(agree)),
                            "topo_qos_mean_rho": _mean(rhos)}
        print(key, per_setting[key])
    _write("oracle_param_sensitivity.json", {"per_setting": per_setting},
           experiment="oracle", grid=[list(g) for g in grid])
    return 0


def cmd_descriptives(_: argparse.Namespace) -> int:
    out: Dict[str, Any] = {}
    for group, ids in (("loso", FOLDS), ("systems", SYSTEMS)):
        rows = {}
        for sid, name in ids.items():
            path = SCENARIOS_DIR / f"{sid}.json"
            topo = _topology(path)
            lab = labels_for(path)["impact"]
            graph = build_graph_from_json(topo)
            flow = _flow(topo)
            apps = [n for n, d in graph.nodes(data=True)
                    if d.get("type") == "Application" and n in lab]
            vals = [lab[n] for n in apps]
            n = len(apps)
            fa = flow.subgraph(apps)
            rows[name] = {
                "n_apps": n,
                "zero_share": sum(v == 0 for v in vals) / max(1, n),
                "tie_fraction": 1 - len(set(vals)) / max(1, n),
                "projection_density": nx.density(fa) if n > 1 else 0.0,
            }
        out[group] = rows
        out[group + "_mean"] = {k: _mean([r[k] for r in rows.values()])
                                for k in ("n_apps", "zero_share", "tie_fraction",
                                          "projection_density")}
        print(group, out[group + "_mean"])
    _write("system_model_descriptives.json", out, experiment="descriptives")
    return 0


def cmd_qos_indep(args: argparse.Namespace) -> int:
    vdir = Path(args.variant_dir)
    rows: Dict[str, Any] = {}
    for sid, name in FOLDS.items():
        path = vdir / f"{sid}.json"
        if not path.exists():
            print(f"missing variant {path}")
            continue
        topo = _topology(path)
        lab = labels_for(path, cache=False)["impact"]
        graph = build_graph_from_json(topo)
        flow = _flow(topo)
        rows[name] = {"Topo (projection)": score(topo_unweighted(flow), lab, graph)["rho"],
                      "Topo-QoS": score(topo_qos(flow), lab, graph)["rho"]}
        # Same arms on the committed corpus, same unweighted-projection Topo, so the
        # gain is compared like for like.
        otopo = _topology(SCENARIOS_DIR / f"{sid}.json")
        olab = labels_for(SCENARIOS_DIR / f"{sid}.json")["impact"]
        og = build_graph_from_json(otopo)
        of = _flow(otopo)
        rows[name]["orig Topo (projection)"] = score(topo_unweighted(of), olab, og)["rho"]
        rows[name]["orig Topo-QoS"] = score(topo_qos(of), olab, og)["rho"]
        print(name, rows[name])
    names = list(rows)
    summ = {k: _mean([rows[n][k] for n in names]) for k in
            ("Topo (projection)", "Topo-QoS", "orig Topo (projection)", "orig Topo-QoS")}
    summ["gain_variant"] = summ["Topo-QoS"] - summ["Topo (projection)"]
    summ["gain_original"] = summ["orig Topo-QoS"] - summ["orig Topo (projection)"]
    summ["gain_variant_vs_published"] = summ["gain_variant"] / 0.204
    contr = {"variant gain": paired([rows[n]["Topo-QoS"] for n in names],
                                    [rows[n]["Topo (projection)"] for n in names])}
    r2p = summ["gain_variant"] < 0.5 * 0.204
    print(json.dumps(summ, indent=1), contr, f"R2' {'TRIGGERED' if r2p else 'not triggered'}")
    _write("qos_indep_corpus.json", {"per_fold": rows, "summary": summ, "contrasts": contr,
                                      "decision_R2prime": {"triggered": r2p}},
           experiment="qos-indep", variant_dir=str(vdir))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("command", choices=["gate", "baselines", "controls", "oracle",
                                        "descriptives", "qos-indep", "all"])
    ap.add_argument("--variant-dir", default="output/variants/qos_indep")
    args = ap.parse_args()
    cmds = {"gate": cmd_gate, "baselines": cmd_baselines, "controls": cmd_controls,
            "oracle": cmd_oracle, "descriptives": cmd_descriptives, "qos-indep": cmd_qos_indep}
    if args.command == "all":
        rc = cmd_gate(args)
        if rc:
            return rc
        for c in ("baselines", "controls", "oracle", "descriptives"):
            cmds[c](args)
        return 0
    return cmds[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
