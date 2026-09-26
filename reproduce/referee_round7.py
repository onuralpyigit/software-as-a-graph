#!/usr/bin/env python3
"""
reproduce/referee_round7.py — Amendment 12: round-7 referee analyses
=====================================================================

Everything registered in PREREGISTRATION.md Amendment 12:

* ``raw``      — R1: training-free rankers computed on the raw multigraph (no
                 DEPENDS_ON derivation) against I*, I_dyn-full and I_comp.
* ``partial``  — R2: partial Spearman rho(r, I_dyn-full | I*) and
                 rho(r, I_dyn-full | Analytic-I*), plus Table 7 on the full population.
* ``learned``  — R3: the saved LOSO predictions of the learned engines re-scored on
                 I_dyn-full and I_comp (gate G3: the I* re-score must reproduce the
                 published mean within 1e-3).
* ``recall``   — R4: recall of the true top-20% set by the predicted top-k%, ties
                 resolved in expectation.
* ``zeroshot`` — R5: InDeg / Reach on the zero-shot harness's own labels, and
                 per-scenario structural descriptors.
* ``latency``  — R6: projection + count vs one I* labelling pass, by graph size.

I_dyn is the published n=30 lexical sample (Amendment 12 deviation; see IDYN_N30).

Usage:
    PYTHONPATH=. python reproduce/referee_round7.py all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import networkx as nx
import numpy as np
from scipy.stats import rankdata, spearmanr

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import stamp  # noqa: E402
from reproduce.independent_oracle_evaluation import (  # noqa: E402
    ICOMP_CACHE,
    _compute_analytic_first_order,
)
from reproduce.training_free_suite import (  # noqa: E402
    FOLDS,
    SCENARIOS_DIR,
    SEEDS,
    SYSTEMS,
    _flow,
    _mean,
    degree_raw,
    holm,
    indeg,
    labels_for,
    mean_ci,
    paired,
    pubs_raw,
    reach,
    reach_r1,
    score,
    subscriber_count_raw,
    topo_qos,
)
from saag.core.graph_io import build_graph_from_json  # noqa: E402
from saag.evaluation.metrics import compute_inductive_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_BENCHMARKS = ROOT / "data" / "benchmarks"
RESULTS = ROOT / "results"
#: Deviation (Amendment 12, logged): Amendment 11 was stopped before its full-population
#: labels existed, so I_dyn is the published seed-42 sample -- the first 30 Applications
#: of each fold in lexicographic order (all 26 on ATM), LOSO folds only.
IDYN_N30 = RESULTS / "idyn_scenario_cache_jss12.json"
IDYN_SOURCE = "published n=30 lexical sample, seed 42 (results/idyn_scenario_cache_jss12.json)"
ICOMP_SYSTEMS_CACHE = RESULTS / "icomp_systems_labels_cache.json"
ORACLES = ("i_star", "i_dyn", "i_comp")

#: Published LOSO mean rho on I* (Table 6) and the saved predictions behind each
#: learned engine. Gate G3 re-scores the predictions against I* and must land
#: within 1e-3 of the published mean before any other oracle is reported.
LEARNED = {
    "HGT-QoS": ("output/loso_cpu_hybrid/hgl_qos", 0.622),
    "GAT-QoS": ("output/loso_cpu_hybrid_gat/gl_full_qos16_cap", 0.635),
    "Hybrid-HGT": ("output/loso_cpu_hybrid/hgl_qos_prior", 0.657),
    "Hybrid-GAT": ("output/loso_cpu_hybrid_gat/gl_qos16_prior", 0.683),
    "GAT-P-QoS": ("output/loso_cpu_dependency_graph/gl_proj_qos16_cap", 0.748),
    "HGT-P-QoS": ("output/loso_cpu_dependency_graph/hgl_proj_qos", 0.514),
    "GAT-P+InDeg": ("output/loso_cpu_dependency_graph/gl_proj_qos16_indeg_prior", 0.758),
}
G3_TOL = 1e-3
K_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
TRUE_TOP = 0.20


# ── shared ────────────────────────────────────────────────────────────────────

def _topology(sid: str) -> Dict[str, Any]:
    return json.loads((SCENARIOS_DIR / f"{sid}.json").read_text())


def _apps(graph: nx.Graph) -> List[str]:
    return [str(n) for n, d in graph.nodes(data=True) if d.get("type") == "Application"]


def load_oracles(names: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """I*, I_dyn (published n=30 lexical sample; see IDYN_N30) and I_comp, by scenario id."""
    idyn = json.loads(IDYN_N30.read_text())
    icomp = dict(json.loads(ICOMP_CACHE.read_text()))
    if ICOMP_SYSTEMS_CACHE.exists():
        icomp.update(json.loads(ICOMP_SYSTEMS_CACHE.read_text())["labels"])
    return {
        "i_star": {n: labels_for(SCENARIOS_DIR / f"{n}.json")["impact"] for n in names},
        "i_dyn": {n: idyn[n] for n in names if n in idyn},
        "i_comp": {n: icomp[n] for n in names if n in icomp},
    }


def pagerank_raw(topology: Dict[str, Any], reverse: bool = False) -> Dict[str, float]:
    """Unweighted PageRank (alpha 0.85) on the raw structural multigraph."""
    g = nx.DiGraph(build_graph_from_json(topology))
    if reverse:
        g = g.reverse(copy=False)
    return {str(k): float(v) for k, v in nx.pagerank(g, alpha=0.85, weight=None).items()}


def raw_rankers(topology: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Every ranker scores every Application. subscriber_count_raw and pubs_raw only
    emit components that publish, and the shared metric code scores only the nodes a
    ranker emits, so without the zero fill a non-publisher (count 0) would silently
    drop out of that ranker's population."""
    flow = _flow(topology)
    apps = [str(a["id"]) for a in topology.get("applications", [])]
    return {name: {**{a: 0.0 for a in apps}, **scores}
            for name, scores in _raw_rankers(topology, flow).items()}


def _raw_rankers(topology: Dict[str, Any], flow: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    return {
        "InDeg": indeg(flow),
        "Raw2Hop": subscriber_count_raw(topology),
        "Degree-raw": degree_raw(topology),
        "Pubs-raw": pubs_raw(topology),
        "Reach-R1": reach_r1(flow),
        "PR-raw": pagerank_raw(topology),
        "RevPR-raw": pagerank_raw(topology, reverse=True),
        "Topo-QoS": topo_qos(flow),
        "Reach": reach(flow),
        "Analytic-I*": _compute_analytic_first_order(topology),
    }


#: Taken once, at process start: the artifacts go to the tracked data/benchmarks/,
#: so stamping each as it is written would mark all but the first as dirty.
_PROV = stamp(script="reproduce/referee_round7.py", amendment=12)


def _write(name: str, payload: Dict[str, Any], **config: Any) -> Path:
    payload["provenance"] = {**_PROV, "config": {**_PROV["config"], **config}}
    DATA_BENCHMARKS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    text = json.dumps(payload, indent=2)
    for d in (DATA_BENCHMARKS, RESULTS):
        (d / name).write_text(text)
    print(f"wrote {DATA_BENCHMARKS / name}")
    return DATA_BENCHMARKS / name


def _summ(per: Dict[str, Dict[str, Any]], names: List[str], key: str = "rho") -> Dict[str, Any]:
    xs = [per[n][key] for n in names if per[n] is not None]
    xs = [x for x in xs if x is not None]
    return {"mean": _mean(xs), "ci95": mean_ci(xs) if len(xs) > 1 else None, "n": len(xs)}


# ── R1: raw-graph baselines ───────────────────────────────────────────────────

RAW_ARMS = ("InDeg", "Raw2Hop", "Degree-raw", "Pubs-raw", "Reach-R1", "PR-raw", "RevPR-raw",
            "Topo-QoS", "Reach", "Analytic-I*")
R1_CONTRASTS = ("PR-raw", "RevPR-raw", "Degree-raw", "Pubs-raw")


def cmd_raw(_: argparse.Namespace) -> int:
    names = list(FOLDS) + list(SYSTEMS)
    oracles = load_oracles(names)
    per: Dict[str, Dict[str, Dict[str, Any]]] = {}
    identity = 0.0
    for sid in names:
        topo = _topology(sid)
        graph = build_graph_from_json(topo)
        rk = raw_rankers(topo)
        apps = _apps(graph)
        identity = max(identity, max(abs(rk["InDeg"].get(a, 0.0) - rk["Raw2Hop"].get(a, 0.0))
                                     for a in apps))
        per[sid] = {o: {r: (score(rk[r], oracles[o][sid], graph) if sid in oracles[o] else None)
                        for r in RAW_ARMS} for o in ORACLES}
        print(f"{sid:32s} " + " ".join(
            f"{o}:{r}={per[sid][o][r]['rho']:.3f}" for o in ORACLES for r in ("InDeg", "RevPR-raw")
            if per[sid][o][r] is not None and per[sid][o][r]["rho"] is not None))
    folds, systems = list(FOLDS), list(SYSTEMS)
    summary = {o: {r: {"loso": _summ({n: per[n][o][r] for n in folds}, folds),
                       "loso_active": _summ({n: per[n][o][r] for n in folds}, folds, "rho_active"),
                       "systems": _summ({n: per[n][o][r] for n in systems
                                         if per[n][o][r] is not None},
                                        [n for n in systems if per[n][o][r] is not None])}
                   for r in RAW_ARMS} for o in ORACLES}
    contrasts: Dict[str, Any] = {}
    for o in ORACLES:
        for r in R1_CONTRASTS:
            a = [per[n][o]["InDeg"]["rho"] for n in folds]
            b = [per[n][o][r]["rho"] or 0.0 for n in folds]
            contrasts[f"{o}: InDeg vs {r}"] = paired(a, b)
    for k, p in holm({k: c["p"] for k, c in contrasts.items()}).items():
        contrasts[k]["p_holm"] = p
    decisions = {"D1": {"identity_max_abs_diff": identity, "applies": identity == 0.0},
                 "D2": {o: all(contrasts[f"{o}: InDeg vs {r}"]["delta"] > 0
                               and contrasts[f"{o}: InDeg vs {r}"]["p_holm"] < 0.05
                               for r in R1_CONTRASTS) for o in ORACLES}}
    for o in ORACLES:
        print(o, {r: round(summary[o][r]["loso"]["mean"], 3) for r in RAW_ARMS
                  if summary[o][r]["loso"]["mean"] is not None})
    print("decisions:", decisions)
    _write("referee_round7_raw_baselines.json",
           {"per_scenario": per, "summary": summary, "contrasts": contrasts,
            "decisions": decisions}, experiment="R1")
    return 0


# ── R2: beyond first order ────────────────────────────────────────────────────

def partial_spearman(x: List[float], y: List[float], z: List[float]) -> Optional[float]:
    """Pearson correlation of the residuals of rank(x) and rank(y) regressed on rank(z)."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    design = np.column_stack([np.ones_like(rz), rz])
    res = []
    for r in (rx, ry):
        beta, *_ = np.linalg.lstsq(design, r, rcond=None)
        res.append(r - design @ beta)
    if np.std(res[0]) < 1e-9 or np.std(res[1]) < 1e-9:
        return None
    return float(np.corrcoef(res[0], res[1])[0, 1])


def cmd_partial(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    oracles = load_oracles(folds)
    rankers = ("InDeg", "Reach", "Topo-QoS", "Analytic-I*", "RevPR-raw", "PR-raw")
    per: Dict[str, Any] = {}
    for sid in folds:
        topo = _topology(sid)
        graph = build_graph_from_json(topo)
        rk = raw_rankers(topo)
        dyn, star = oracles["i_dyn"][sid], oracles["i_star"][sid]
        apps = [a for a in _apps(graph) if a in dyn and a in star]
        ana = rk["Analytic-I*"]
        row: Dict[str, Any] = {"n": len(apps),
                               "rho_istar_idyn": float(spearmanr([star[a] for a in apps],
                                                                 [dyn[a] for a in apps]).correlation)}
        for r in rankers:
            x = [rk[r].get(a, 0.0) for a in apps]
            y = [dyn[a] for a in apps]
            row[r] = {
                "rho": score(rk[r], dyn, graph)["rho"],
                "partial_given_istar": partial_spearman(x, y, [star[a] for a in apps]),
                "partial_given_analytic": (None if r == "Analytic-I*" else
                                           partial_spearman(x, y, [ana.get(a, 0.0) for a in apps])),
            }
        per[sid] = row
        print(f"{sid:28s} n={row['n']:3d} " + " ".join(
            f"{r}={row[r]['rho']}|{row[r]['partial_given_istar']}"
            for r in rankers))
    summary: Dict[str, Any] = {}
    for r in rankers:
        s = {}
        for key in ("rho", "partial_given_istar", "partial_given_analytic"):
            xs = [per[f][r][key] for f in folds if per[f][r][key] is not None]
            s[key] = {"mean": _mean(xs), "ci95": mean_ci(xs) if len(xs) > 1 else None,
                      "n_folds": len(xs)}
        both = [f for f in folds
                if per[f][r]["rho"] is not None and per[f]["Topo-QoS"]["rho"] is not None]
        s["vs_topo_qos"] = (paired([per[f][r]["rho"] for f in both],
                                   [per[f]["Topo-QoS"]["rho"] for f in both])
                            if r != "Topo-QoS" else None)
        ci = s["partial_given_istar"]["ci95"]
        s["D3"] = None if ci is None else ("D3" if ci[0] <= 0 <= ci[1] else
                                           ("D3'" if s["partial_given_istar"]["mean"] > 0 else "negative"))
        summary[r] = s
    summary["_oracle"] = {
        "idyn_source": IDYN_SOURCE,
        "rho_istar_idyn": _summ({f: {"rho": per[f]["rho_istar_idyn"]} for f in folds}, folds),
        "n_per_fold": {f: per[f]["n"] for f in folds},
    }
    for r in rankers:
        print(f"{r:12s} rho={summary[r]['rho']['mean']} "
              f"partial|I*={summary[r]['partial_given_istar']['mean']} "
              f"CI={summary[r]['partial_given_istar']['ci95']} {summary[r]['D3']}")
    print("oracle:", summary["_oracle"]["rho_istar_idyn"])
    _write("referee_round7_partial.json", {"per_fold": per, "summary": summary,
                                           "idyn_source": IDYN_SOURCE},
           experiment="R2")
    return 0


# ── R3: learned engines on every oracle ───────────────────────────────────────

def _learned_preds(path: str) -> Dict[str, Dict[str, float]]:
    raw = json.loads((ROOT / path / "inductive_predictions.json").read_text())
    return {sid: {str(n): float(v["overall"] if isinstance(v, dict) else v)
                  for n, v in nodes.items()} for sid, nodes in raw.items()}


def _seed_mean_published(path: str) -> Optional[float]:
    """Mean over folds of the mean over seeds of each seed's own logged rho."""
    folds = []
    for fold in sorted((ROOT / path / "workspace").glob("fold_*")):
        rs = [json.loads(p.read_text())["metrics"]["spearman_rho"]
              for p in fold.glob("seed_*/seed_result.json")]
        if rs:
            folds.append(float(np.mean(rs)))
    return _mean(folds)


def cmd_learned(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    oracles = load_oracles(folds)
    graphs = {sid: build_graph_from_json(_topology(sid)) for sid in folds}
    out: Dict[str, Any] = {}
    for engine, (path, published) in LEARNED.items():
        preds = _learned_preds(path)
        per = {sid: {o: score(preds[sid], oracles[o][sid], graphs[sid]) for o in ORACLES}
               for sid in folds}
        istar = _mean([per[s]["i_star"]["rho"] for s in folds])
        g3 = abs(istar - published) <= G3_TOL + 5e-4  # published values are 3-d.p. rounded
        seed_mean = _seed_mean_published(path)
        # Deviation (logged in Amendment 12): the saved predictions are the seed
        # ensemble, so G3 as registered compares the ensemble's rho with the mean of
        # per-seed rho. Every engine is reported as a seed ensemble, beside its own
        # I* value and the check that the per-seed logs reproduce the published mean.
        block: Dict[str, Any] = {
            "path": path, "statistic": "seed-ensemble (mean of 5 seeds' predictions)",
            "published_i_star": published, "ensemble_i_star": istar,
            "G3_passed_as_registered": bool(g3),
            "per_seed_logs_mean_i_star": seed_mean,
            "per_seed_logs_reproduce_published": (seed_mean is not None
                                                  and abs(seed_mean - published) <= 5e-4 + 1e-9),
            "per_fold": per,
            "summary": {o: {"rho": _summ({s: per[s][o] for s in folds}, folds),
                            "rho_active": _summ({s: per[s][o] for s in folds}, folds, "rho_active"),
                            "overlap_at_k": _summ({s: per[s][o] for s in folds}, folds,
                                                  "overlap_at_k")}
                        for o in ORACLES},
        }
        out[engine] = block
        msg = " ".join(f"{o}={block['summary'][o]['rho']['mean']:.3f}" for o in ORACLES)
        print(f"{engine:13s} G3 {'pass' if g3 else 'FAIL'} (ensemble I* {istar:.4f}, "
              f"seed logs {seed_mean:.4f}, published {published}) {msg}")
    # The InDeg reference on the same oracles, for the D4 comparison.
    ref = {}
    for sid in folds:
        flow = _flow(_topology(sid))
        ref[sid] = {o: score(indeg(flow), oracles[o][sid], graphs[sid]) for o in ORACLES}
    out["_InDeg"] = {o: _summ({s: ref[s][o] for s in folds}, folds) for o in ORACLES}
    _write("referee_round7_learned_oracles.json", out, experiment="R3", g3_tolerance=G3_TOL)
    return 0


# ── R4: recall@k ──────────────────────────────────────────────────────────────

def expected_recall(pred: Dict[str, float], true: Dict[str, float], apps: List[str],
                    k_frac: float) -> Dict[str, float]:
    """Recall of the tie-inclusive true top-20% by the predicted top-k%, exact under ties."""
    n = len(apps)
    y = np.array([true[a] for a in apps])
    s = np.array([pred.get(a, 0.0) for a in apps])
    kt = max(1, int(round(TRUE_TOP * n)))
    thr = np.sort(y)[::-1][kt - 1]
    # Tie-inclusive at the boundary; if the boundary falls among zero-impact
    # components, the critical set is the active components only.
    crit = y > 0 if thr <= 0 else y >= thr
    n_crit = int(crit.sum())
    if n_crit == 0:
        return {"expected": float("nan"), "optimistic": float("nan"), "pessimistic": float("nan")}
    m = max(1, int(round(k_frac * n)))
    cut = np.sort(s)[::-1][m - 1]
    above = s > cut
    group = s == cut
    slots = m - int(above.sum())
    hit_above = int((above & crit).sum())
    g_crit = int((group & crit).sum())
    g_size = int(group.sum())
    exp = hit_above + slots * g_crit / g_size
    opt = hit_above + min(slots, g_crit)
    pes = hit_above + max(0, slots - (g_size - g_crit))
    return {"expected": exp / n_crit, "optimistic": opt / n_crit, "pessimistic": pes / n_crit}


def cmd_recall(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    oracles = load_oracles(folds)
    gatp = _learned_preds(LEARNED["GAT-P-QoS"][0])
    per: Dict[str, Any] = {}
    for sid in folds:
        topo = _topology(sid)
        graph = build_graph_from_json(topo)
        rk = raw_rankers(topo)
        rk["GAT-P-QoS"] = gatp[sid]
        per[sid] = {}
        for o in ("i_star", "i_dyn"):
            lab = oracles[o][sid]
            apps = [a for a in _apps(graph) if a in lab]
            per[sid][o] = {r: {f"{k:.2f}": expected_recall(rk[r], lab, apps, k) for k in K_GRID}
                           for r in ("InDeg", "Reach", "Topo-QoS", "Analytic-I*", "GAT-P-QoS")}
    curves: Dict[str, Any] = {}
    for o in ("i_star", "i_dyn"):
        curves[o] = {}
        for r in ("InDeg", "Reach", "Topo-QoS", "Analytic-I*", "GAT-P-QoS"):
            c = {}
            for k in K_GRID:
                kk = f"{k:.2f}"
                c[kk] = {b: _mean([per[s][o][r][kk][b] for s in folds
                                   if not np.isnan(per[s][o][r][kk][b])])
                         for b in ("expected", "optimistic", "pessimistic")}
            margin = {}
            for target in (0.80, 0.90):
                hit = [k for k in K_GRID if c[f"{k:.2f}"]["expected"] >= target]
                margin[f"{target:.2f}"] = min(hit) if hit else None
            curves[o][r] = {"curve": c, "safety_margin": margin}
            print(o, r, " ".join(f"{k}:{v['expected']:.2f}" for k, v in c.items()), margin)
    _write("referee_round7_recall.json", {"per_fold": per, "curves": curves,
                                          "true_top": TRUE_TOP, "k_grid": K_GRID},
           experiment="R4")
    return 0


# ── R5: zero-shot on one harness + descriptors ────────────────────────────────

def _depth(flow: nx.DiGraph, apps: List[str]) -> int:
    """Longest shortest-path distance from any dependent to an Application."""
    rev = flow.reverse(copy=False)
    best = 0
    for a in apps:
        if a in rev:
            d = nx.single_source_shortest_path_length(rev, a)
            best = max(best, max(d.values(), default=0))
    return best


def _gini(x: List[float]) -> float:
    v = np.sort(np.asarray(x, dtype=float))
    if v.sum() == 0:
        return 0.0
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))


def descriptors(topo: Dict[str, Any], impact: Dict[str, float]) -> Dict[str, Any]:
    graph = build_graph_from_json(topo)
    flow = _flow(topo)
    apps = [a for a in _apps(graph) if a in impact]
    ind = indeg(flow)
    return {"n_apps": len(apps),
            "zero_impact_share": float(np.mean([impact[a] <= 0 for a in apps])) if apps else None,
            "indeg_gini": _gini([ind.get(a, 0.0) for a in apps]),
            "max_dependency_depth": _depth(flow, apps)}


def cmd_zeroshot(_: argparse.Namespace) -> int:
    from reproduce.main_table import _parse_failure_impact, _remap_node_ids
    cache = ROOT / "output" / "realworld_cache"
    per: Dict[str, Any] = {}
    for sid in SYSTEMS:
        topo = json.loads((cache / sid / "topology.json").read_text())
        graph = build_graph_from_json(topo)
        sim = _remap_node_ids(_parse_failure_impact(
            json.loads((cache / sid / "failure_impact.json").read_text())),
            {str(n) for n in graph.nodes()})
        true = {n: float(d.get("composite", 0.0)) for n, d in sim.items()}
        flow = _flow(topo)
        row: Dict[str, Any] = {}
        for r, pred in (("InDeg", indeg(flow)), ("Reach", reach(flow))):
            m = compute_inductive_metrics(pred, true, graph, population="application")
            row[r] = {**score(pred, true, graph), "pr_auc": m.get("pr_auc")}
        row["descriptors"] = descriptors(topo, true)
        per[sid] = row
        print(f"{sid:32s} InDeg={row['InDeg']['rho']:.3f} Reach={row['Reach']['rho']:.3f} "
              f"{row['descriptors']}")
    systems = list(SYSTEMS)
    summary = {r: {"rho": _summ({s: per[s][r] for s in systems}, systems),
                   "rho_active": _summ({s: per[s][r] for s in systems}, systems, "rho_active"),
                   "pr_auc": _summ({s: per[s][r] for s in systems}, systems, "pr_auc")}
               for r in ("InDeg", "Reach")}
    folds = {}
    for sid in FOLDS:
        folds[sid] = descriptors(_topology(sid), labels_for(SCENARIOS_DIR / f"{sid}.json")["impact"])
    for k in ("rho", "rho_active", "pr_auc"):
        print(k, {r: round(summary[r][k]["mean"], 3) for r in summary})
    _write("referee_round7_zeroshot.json",
           {"per_system": per, "summary": summary, "fold_descriptors": folds,
            "label_source": "output/realworld_cache/<system>/failure_impact.json (composite)"},
           experiment="R5")
    return 0


# ── R6: latency by size ───────────────────────────────────────────────────────

def _time(fn: Callable[[], Any], repeats: int) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def cmd_latency(args: argparse.Namespace) -> int:
    from reproduce.inference_latency import _counts_for
    from saag.simulation.fault_injector import FaultInjector
    from tools.generation.models import GraphConfig
    from tools.generation.service import generate_graph

    rows = []
    for n in args.sizes:
        cfg = GraphConfig.from_yaml({"graph": {"seed": 42, "counts": _counts_for(n)}})
        topo = generate_graph(config=cfg, seed=42)
        graph = build_graph_from_json(topo)
        flow = _flow(topo)
        row = {
            "n_target": n, "n_actual": graph.number_of_nodes(), "n_edges": graph.number_of_edges(),
            "projection_s": _time(lambda: _flow(topo), args.repeats),
            "indeg_s": _time(lambda: indeg(flow), args.repeats),
            "reach_s": _time(lambda: reach(flow), args.repeats),
        }
        # I* grows roughly quadratically with size (about 5 s at 250 components and
        # 20 s at 500 on this machine), so above --istar-max it is not timed.
        row["istar_s"] = None if n > args.istar_max else _time(lambda: FaultInjector(
            graph=build_graph_from_json(topo), seeds=SEEDS, cascade_depth_limit=0,
            propagation_threshold=0.2, qos_factor_mode="ladder",
        ).run(node_types=["Application", "Broker", "Library"]),
            args.repeats if n <= args.istar_full_repeats_max else 1)
        row["istar_repeats"] = (0 if n > args.istar_max else
                                args.repeats if n <= args.istar_full_repeats_max else 1)
        row["count_path_s"] = row["projection_s"] + row["indeg_s"]
        rows.append(row)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    _write("referee_round7_latency.json", {"sizes": rows}, experiment="R6",
           repeats=args.repeats, statistic="median", istar_max=args.istar_max,
           istar_full_repeats_max=args.istar_full_repeats_max)
    return 0


# ── summary-statistic sensitivity (descriptive; logged beside Amendment 12) ─────

def cmd_averaging(_: argparse.Namespace) -> int:
    """Table 6 means under arithmetic, Fisher-z and |V_app|-weighted averaging."""
    from reproduce.training_free_suite import (
        PUBLISHED_GAT_QOS_CPU, PUBLISHED_HGT_QOS_CPU, PUBLISHED_HYBRID_GAT_CPU,
        PUBLISHED_HYBRID_HGT_CPU, PUBLISHED_TOPO)
    folds = list(FOLDS)
    ioe = json.loads((RESULTS / "independent_oracle_evaluation.json").read_text())["per_fold"]
    dg = json.loads((RESULTS / "dependency_graph_contrasts.json").read_text())["per_fold"]
    n = {f: ioe[f]["i_star"]["InDeg"]["n"] for f in folds}
    by_name = {FOLDS[f]: f for f in folds}
    per: Dict[str, Dict[str, float]] = {
        "Analytic-I*": {f: ioe[f]["i_star"]["Analytic-I*"]["rho"] for f in folds},
        "InDeg": {f: ioe[f]["i_star"]["InDeg"]["rho"] for f in folds},
        "Reach": {f: ioe[f]["i_star"]["Reach"]["rho"] for f in folds},
        "Topo-QoS": {f: ioe[f]["i_star"]["Topo-QoS"]["rho"] for f in folds},
        "GAT-P-QoS": {f: dg["gl_proj_qos16_cap"][f]["rho"] for f in folds},
        "GAT-P+InDeg": {f: dg["gl_proj_qos16_indeg_prior"][f]["rho"] for f in folds},
        "HGT-P-QoS": {f: dg["hgl_proj_qos"][f]["rho"] for f in folds},
    }
    for lab, table in (("Topo", PUBLISHED_TOPO), ("HGT-QoS", PUBLISHED_HGT_QOS_CPU),
                       ("GAT-QoS", PUBLISHED_GAT_QOS_CPU), ("Hybrid-HGT", PUBLISHED_HYBRID_HGT_CPU),
                       ("Hybrid-GAT", PUBLISHED_HYBRID_GAT_CPU)):
        per[lab] = {by_name[k]: v for k, v in table.items()}
    w = np.array([n[f] for f in folds], dtype=float)
    out: Dict[str, Any] = {}
    for lab, vals in per.items():
        x = np.array([vals[f] for f in folds])
        z = np.arctanh(np.clip(x, -0.999999, 0.999999))
        out[lab] = {"arithmetic": float(x.mean()), "fisher_z": float(np.tanh(z.mean())),
                    "size_weighted": float((w * x).sum() / w.sum())}
    # Seed spread of each learned engine: per fold, max - min of the five seeds'
    # logged rho, then the mean and max over folds (M11).
    spread: Dict[str, Any] = {}
    for lab, (path, _) in LEARNED.items():
        rng, sds = [], []
        for fold in sorted((ROOT / path / "workspace").glob("fold_*")):
            rs = [json.loads(q.read_text())["metrics"]["spearman_rho"]
                  for q in fold.glob("seed_*/seed_result.json")]
            if len(rs) > 1:
                rng.append(max(rs) - min(rs))
                sds.append(float(np.std(rs, ddof=1)))
        spread[lab] = {"mean_range": _mean(rng), "max_range": float(max(rng)) if rng else None,
                       "mean_sd": _mean(sds), "n_folds": len(rng)}
        print(f"seed spread {lab:13s} range mean {spread[lab]['mean_range']:.3f} "
              f"max {spread[lab]['max_range']:.3f} sd {spread[lab]['mean_sd']:.3f}")
    orders = {k: [lab for lab, _ in sorted(out.items(), key=lambda kv: -kv[1][k])]
              for k in ("arithmetic", "fisher_z", "size_weighted")}
    for lab, v in sorted(out.items(), key=lambda kv: -kv[1]["arithmetic"]):
        print(f"{lab:13s} " + " ".join(f"{k}={v[k]:.3f}" for k in v))
    print("orders equal:", orders["arithmetic"] == orders["fisher_z"] == orders["size_weighted"])
    _write("referee_round7_averaging.json",
           {"per_ranker": out, "orders": orders, "n_apps": n, "seed_spread": spread,
            "orders_identical": orders["arithmetic"] == orders["fisher_z"] == orders["size_weighted"]},
           experiment="averaging (descriptive, not registered)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stage", choices=["raw", "partial", "learned", "recall", "zeroshot",
                                      "latency", "averaging", "all"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000, 2000, 5000, 10000])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--istar-max", type=int, default=10000,
                    help="largest size at which the I* labelling pass is timed")
    ap.add_argument("--istar-full-repeats-max", type=int, default=10000,
                    help="above this size I* is timed once rather than --repeats times")
    args = ap.parse_args()
    stages = {"raw": cmd_raw, "partial": cmd_partial, "learned": cmd_learned,
              "recall": cmd_recall, "zeroshot": cmd_zeroshot, "latency": cmd_latency,
              "averaging": cmd_averaging}
    if args.stage == "all":
        return max(fn(args) for fn in stages.values())
    return stages[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
