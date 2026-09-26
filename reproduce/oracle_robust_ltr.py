#!/usr/bin/env python3
"""
reproduce/oracle_robust_ltr.py — Amendment 11: learned combination across oracles
==================================================================================

Everything registered in PREREGISTRATION.md Amendment 11:

* ``labels`` — I_dyn-full (every Application, seeds 42/123/456/789/2024, mean over
               seeds) on the twelve LOSO folds and the five system models, and
               I_comp on the system models. Cached under results/ (keyed by corpus
               digest), so a rerun does not repeat ~3 CPU-hours of simulation.
* ``all``    — labels, then gates G1/G2, then the four GBM arms under LOSO and
               zero-shot, the three Holm families and the decision rules.

The four arms are a 2 x 2: feature set (S, or S + Q) x training oracle (I*, or
I_dyn-full). I_comp is never a training label: FailureSimulator is the
Validate-stage oracle (CLAUDE.md invariants), and ``fit_predict`` refuses it.

The provenance stamp is taken once, at process start. The artifacts go to the
tracked data/benchmarks/, so stamping each one as it is written would mark every
artifact after the first as built from a dirty tree.

Usage:
    PYTHONPATH=. python reproduce/oracle_robust_ltr.py all --workers 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import corpus_digest, stamp  # noqa: E402
from reproduce.independent_oracle_evaluation import (  # noqa: E402
    ICOMP_CACHE,
    _compute_analytic_first_order,
)
from reproduce.training_free_suite import (  # noqa: E402
    FOLDS,
    SEEDS,
    SYSTEMS,
    _flow,
    _mean,
    holm,
    indeg,
    labels_for,
    mean_ci,
    paired,
    reach,
    reach_qos,
    score,
    topo_qos,
    topo_unweighted,
)
from saag.prediction.models.tabular import _build_regressor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = ROOT / "data" / "scenarios"
DATA_BENCHMARKS = ROOT / "data" / "benchmarks"
RESULTS = ROOT / "results"
IDYN_N30_CACHE = RESULTS / "idyn_scenario_cache_jss12.json"
IDYN_FULL_CACHE = RESULTS / "idyn_full_labels_cache.json"
ICOMP_SYSTEMS_CACHE = RESULTS / "icomp_systems_labels_cache.json"
PUBLISHED_ORACLE_EVAL = DATA_BENCHMARKS / "independent_oracle_evaluation.json"

#: Published I_dyn settings (reproduce/independent_oracle_evaluation.py).
IDYN_SETTINGS = {"duration": 60.0, "qos_mode": "full", "target_utilization": 0.65}
ORACLES = ("i_star", "i_dyn", "i_comp")
TRAIN_ORACLES = ("i_star", "i_dyn")
COMPARATORS = ("Analytic-I*", "InDeg", "Reach", "Topo-QoS")
G1_TOL = 1e-9
G2_TOL = 1e-3

S_NAMES = ["InDeg", "Reach", "Analytic-I*", "OutDeg", "Pubs", "Subs", "Libs",
           "CoHosted", "Topo"]
Q_NAMES = ["Topo-QoS", "Reach-QoS", "QoS-InDeg", "PubRate", "PubBytes",
           "ReliableShare", "DurableShare", "DeadlineShare", "MaxPriority"]

#: The four registered arms: (feature sets, training oracle).
ARMS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "gbm_dep": (("S",), "i_star"),
    "gbm_dep_qos": (("S", "Q"), "i_star"),
    "gbm_dep_dyn": (("S",), "i_dyn"),
    "gbm_dep_qos_dyn": (("S", "Q"), "i_dyn"),
}

_PRIORITY = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4, "URGENT": 4, "HIGHEST": 4}


# ── topology helpers ──────────────────────────────────────────────────────────

def _path(name: str) -> Path:
    return SCENARIOS_DIR / f"{name}.json"


def _topology(name: str) -> Dict[str, Any]:
    return json.loads(_path(name).read_text())


def _edges(topo: Dict[str, Any], kind: str) -> List[Tuple[str, str]]:
    return [(str(r.get("from") or r.get("source")), str(r.get("to") or r.get("target")))
            for r in topo.get("relationships", {}).get(kind, [])]


def app_ids(topo: Dict[str, Any]) -> List[str]:
    return [str(a["id"]) for a in topo.get("applications", [])]


def pct(values: np.ndarray) -> np.ndarray:
    """Within-scenario percentile rank in (0, 1], average ties."""
    values = np.asarray(values, dtype=float)
    return rankdata(values, method="average") / len(values) if len(values) else values


# ── features ──────────────────────────────────────────────────────────────────

def raw_features(topo: Dict[str, Any]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Per-Application S and Q feature matrices, raw (not yet rank-normalised)."""
    apps = app_ids(topo)
    flow = _flow(topo)
    topics = {str(t["id"]): t for t in topo.get("topics", [])}

    pub = _edges(topo, "publishes_to")
    sub = _edges(topo, "subscribes_to")
    pubs_of: Dict[str, List[str]] = {}
    for a, t in pub:
        pubs_of.setdefault(a, []).append(t)
    subs_of: Dict[str, set] = {}
    for a, t in sub:
        subs_of.setdefault(a, set()).add(t)
    libs_of: Dict[str, set] = {}
    for a, lib in _edges(topo, "uses"):
        libs_of.setdefault(a, set()).add(lib)
    hosts_of: Dict[str, set] = {}
    apps_on: Dict[str, set] = {}
    for a, n in _edges(topo, "runs_on"):
        hosts_of.setdefault(a, set()).add(n)
        apps_on.setdefault(n, set()).add(a)

    s = {
        "InDeg": indeg(flow),
        "Reach": reach(flow),
        "Analytic-I*": _compute_analytic_first_order(topo),
        "OutDeg": {str(v): float(d) for v, d in flow.out_degree()},
        "Topo": topo_unweighted(flow),
        "Topo-QoS": topo_qos(flow),
        "Reach-QoS": reach_qos(flow),
    }
    qos_in = {str(v): float(sum(d.get("qos_weight", 1.0)
                                for _, _, d in flow.in_edges(v, data=True)))
              for v in flow.nodes}

    def topic_stats(a: str) -> List[float]:
        ts = [topics[t] for t in dict.fromkeys(pubs_of.get(a, [])) if t in topics]
        if not ts:
            return [0.0] * 6
        q = [t.get("qos", {}) or {} for t in ts]
        freq = [float(t.get("frequency", 0.0) or 0.0) for t in ts]
        size = [float(t.get("size", 0.0) or 0.0) for t in ts]
        deadline = [t.get("deadline_ms", qq.get("deadline_ms")) for t, qq in zip(ts, q)]
        return [
            float(sum(freq)),
            float(sum(f * z for f, z in zip(freq, size))),
            float(np.mean([qq.get("reliability") == "RELIABLE" for qq in q])),
            float(np.mean([qq.get("durability", "VOLATILE") != "VOLATILE" for qq in q])),
            float(np.mean([d is not None and float(d) > 0 for d in deadline])),
            float(max(_PRIORITY.get(str(qq.get("transport_priority", "")).upper(), 0)
                      for qq in q)),
        ]

    S = np.array([[
        s["InDeg"].get(a, 0.0), s["Reach"].get(a, 0.0), s["Analytic-I*"].get(a, 0.0),
        s["OutDeg"].get(a, 0.0), float(len(pubs_of.get(a, []))),
        float(len(subs_of.get(a, ()))), float(len(libs_of.get(a, ()))),
        float(len(set().union(*(apps_on[n] for n in hosts_of.get(a, ()))) - {a})
              if hosts_of.get(a) else 0.0),
        s["Topo"].get(a, 0.0),
    ] for a in apps], dtype=float)
    Q = np.array([[
        s["Topo-QoS"].get(a, 0.0), s["Reach-QoS"].get(a, 0.0), qos_in.get(a, 0.0),
        *topic_stats(a),
    ] for a in apps], dtype=float)
    return apps, {"S": S, "Q": Q}


def features(topo: Dict[str, Any]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Every column converted to a within-scenario percentile rank (registered)."""
    apps, raw = raw_features(topo)
    return apps, {k: np.column_stack([pct(m[:, j]) for j in range(m.shape[1])])
                  for k, m in raw.items()}


def comparator_scores(topo: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    flow = _flow(topo)
    return {
        "Analytic-I*": _compute_analytic_first_order(topo),
        "InDeg": indeg(flow),
        "Reach": reach(flow),
        "Topo-QoS": topo_qos(flow),
    }


# ── labels ────────────────────────────────────────────────────────────────────

def _idyn_job(job: Tuple[str, int]) -> Tuple[str, int, Dict[str, float], float]:
    from reproduce.convergent_validity import _message_flow_labels
    name, seed = job
    t0 = time.perf_counter()
    labels = _message_flow_labels(name, seed=seed, only=app_ids(_topology(name)),
                                  **IDYN_SETTINGS)
    return name, seed, labels, time.perf_counter() - t0


def _icomp_job(name: str) -> Tuple[str, Dict[str, float]]:
    from reproduce.convergent_validity import _failure_simulator_labels
    return name, _failure_simulator_labels(name, qos=True, layer="Application")


def _cache_key() -> Dict[str, Any]:
    return {"corpus_digest": corpus_digest(), "seeds": SEEDS, **IDYN_SETTINGS}


def idyn_full_labels(names: List[str], workers: int) -> Dict[str, Any]:
    """I_dyn-full per scenario: per-seed labels, the seed mean and the wall-clock."""
    key = _cache_key()
    if IDYN_FULL_CACHE.exists():
        cached = json.loads(IDYN_FULL_CACHE.read_text())
        if cached.get("key") == key and set(names) <= set(cached["labels"]):
            print(f"loaded I_dyn-full from {IDYN_FULL_CACHE}")
            return cached["labels"]
    jobs = [(n, s) for n in names for s in SEEDS]
    print(f"I_dyn-full: {len(jobs)} (scenario, seed) jobs on {workers} workers")
    out: Dict[str, Any] = {n: {"per_seed": {}, "seconds": {}} for n in names}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, seed, labels, secs in ex.map(_idyn_job, jobs):
            out[name]["per_seed"][str(seed)] = labels
            out[name]["seconds"][str(seed)] = round(secs, 2)
            print(f"  {name} seed {seed}: {len(labels)} apps in {secs:.0f}s", flush=True)
    for name, block in out.items():
        # A component the engine cannot observe is omitted, not scored 0.0; the
        # mean is therefore over the seeds in which it was observed.
        nodes = sorted(set().union(*(set(v) for v in block["per_seed"].values())))
        block["mean"] = {v: float(np.mean([block["per_seed"][s][v]
                                           for s in block["per_seed"]
                                           if v in block["per_seed"][s]]))
                         for v in nodes}
    RESULTS.mkdir(exist_ok=True)
    IDYN_FULL_CACHE.write_text(json.dumps({"key": key, "labels": out}))
    return out


def icomp_labels(workers: int) -> Dict[str, Dict[str, float]]:
    """Published I_comp cache for the folds; computed here for the system models."""
    out = dict(json.loads(ICOMP_CACHE.read_text()))
    key = {"corpus_digest": corpus_digest()}
    systems: Dict[str, Dict[str, float]] = {}
    if ICOMP_SYSTEMS_CACHE.exists():
        cached = json.loads(ICOMP_SYSTEMS_CACHE.read_text())
        if cached.get("key") == key:
            systems = cached["labels"]
    if not systems:
        with ProcessPoolExecutor(max_workers=min(workers, len(SYSTEMS))) as ex:
            for name, labels in ex.map(_icomp_job, list(SYSTEMS)):
                systems[name] = labels
        RESULTS.mkdir(exist_ok=True)
        ICOMP_SYSTEMS_CACHE.write_text(json.dumps({"key": key, "labels": systems}))
    out.update(systems)
    return out


# ── gates ─────────────────────────────────────────────────────────────────────

def gate_g1(idyn: Dict[str, Any]) -> Dict[str, Any]:
    """Seed-42 I_dyn-full on the published 30 lexical candidates must match exactly."""
    published = json.loads(IDYN_N30_CACHE.read_text())
    per: Dict[str, Any] = {}
    worst = 0.0
    for name in FOLDS:
        full = idyn[name]["per_seed"]["42"]
        cand = published[name]
        shared = [v for v in cand if v in full]
        diff = max((abs(full[v] - cand[v]) for v in shared), default=0.0)
        worst = max(worst, diff)
        per[name] = {"n_published": len(cand), "n_compared": len(shared),
                     "not_application": sorted(set(cand) - set(full)),
                     "max_abs_diff": diff}
    missing = sum(p["n_published"] - p["n_compared"] for p in per.values())
    return {"passed": bool(worst <= G1_TOL and missing == 0), "tolerance": G1_TOL,
            "max_abs_diff": worst, "n_uncompared": missing, "per_fold": per}


def gate_g2(oracles: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    """Recomputed comparators must match the published I* and I_comp rho per fold."""
    published = json.loads(PUBLISHED_ORACLE_EVAL.read_text())["per_fold"]
    rows, worst = [], 0.0
    for name in FOLDS:
        topo = _topology(name)
        flow = _flow(topo)
        comps = comparator_scores(topo)
        for oracle in ("i_star", "i_comp"):
            for ranker in COMPARATORS:
                ours = score(comps[ranker], oracles[oracle][name], flow)["rho"]
                theirs = published[name][oracle][ranker]["rho"]
                d = abs(ours - theirs)
                worst = max(worst, d)
                rows.append({"fold": name, "oracle": oracle, "ranker": ranker,
                             "rho": ours, "published": theirs, "abs_diff": d})
    return {"passed": bool(worst <= G2_TOL), "tolerance": G2_TOL,
            "max_abs_diff": worst, "checks": rows}


# ── learning ──────────────────────────────────────────────────────────────────

def _design(name: str, sets: Tuple[str, ...], cache: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    if name not in cache:
        cache[name] = features(_topology(name))
    apps, mats = cache[name]
    return apps, np.column_stack([mats[s] for s in sets])


def fit_predict(train: List[str], test: str, sets: Tuple[str, ...], train_oracle: str,
                oracles: Dict[str, Dict[str, Dict[str, float]]], seed: int,
                feat_cache: Dict[str, Any]) -> Dict[str, float]:
    """Fit on the labelled Applications of *train*, predict every Application of *test*."""
    if train_oracle not in TRAIN_ORACLES:
        raise ValueError(f"{train_oracle!r} is not a training oracle; I_comp is the "
                         "Validate-stage oracle and may not label a predictor")
    xs, ys = [], []
    for name in train:
        apps, X = _design(name, sets, feat_cache)
        lab = oracles[train_oracle][name]
        rows = [i for i, a in enumerate(apps) if a in lab]
        if len(rows) < 2:
            continue
        xs.append(X[rows])
        ys.append(pct(np.array([lab[apps[i]] for i in rows])))
    model = _build_regressor(seed)
    model.fit(np.vstack(xs), np.concatenate(ys))
    apps, X = _design(test, sets, feat_cache)
    return {a: float(p) for a, p in zip(apps, model.predict(X))}


def _cell(pred: Dict[str, float], oracles: Dict[str, Dict[str, Dict[str, float]]],
          name: str, flow) -> Dict[str, Any]:
    return {o: score(pred, oracles[o][name], flow)
            for o in ORACLES if oracles[o].get(name)}


def _seed_mean(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for o in cells[0]:
        out[o] = {m: _mean([c[o][m] for c in cells])
                  for m in ("rho", "rho_active", "overlap_at_k")}
        out[o]["rho_seed_sd"] = float(np.std([c[o]["rho"] for c in cells
                                              if c[o]["rho"] is not None]))
        out[o]["n_apps"] = cells[0][o]["n_apps"]
        out[o]["n_active"] = cells[0][o]["n_active"]
    return out


def evaluate(names: List[str], train_pool: List[str], oracles, loso: bool) -> Dict[str, Any]:
    feat_cache: Dict[str, Any] = {}
    out: Dict[str, Any] = {}
    for name in names:
        topo = _topology(name)
        flow = _flow(topo)
        train = [n for n in train_pool if n != name] if loso else list(train_pool)
        row: Dict[str, Any] = {"comparators": {r: _cell(p, oracles, name, flow)
                                               for r, p in comparator_scores(topo).items()},
                               "arms": {}}
        t0 = time.perf_counter()
        _design(name, ("S", "Q"), feat_cache)
        row["feature_seconds"] = round(time.perf_counter() - t0, 4)
        for arm, (sets, train_oracle) in ARMS.items():
            cells = [_cell(fit_predict(train, name, sets, train_oracle, oracles, seed,
                                       feat_cache), oracles, name, flow)
                     for seed in SEEDS]
            row["arms"][arm] = _seed_mean(cells)
        out[name] = row
        print(f"  {name}: " + ", ".join(
            f"{a} {row['arms'][a]['i_star']['rho']:.3f}" for a in ARMS), flush=True)
    return out


# ── contrasts and decisions ───────────────────────────────────────────────────

def _rho(row: Dict[str, Any], who: str, oracle: str) -> Optional[float]:
    block = row["arms"].get(who) or row["comparators"].get(who)
    return block[oracle]["rho"]


def worst_case(row: Dict[str, Any], who: str) -> float:
    return min(_rho(row, who, o) for o in ORACLES)


def _contrast(per_fold: Dict[str, Any], a: str, b: str, stat) -> Dict[str, Any]:
    xa = [stat(per_fold[f], a) for f in FOLDS]
    xb = [stat(per_fold[f], b) for f in FOLDS]
    return {"a": a, "b": b, "mean_a": float(np.mean(xa)), "mean_b": float(np.mean(xb)),
            "per_fold_a": xa, "per_fold_b": xb, **paired(xa, xb)}


def contrasts(per_fold: Dict[str, Any]) -> Dict[str, Any]:
    on = lambda o: (lambda row, who: _rho(row, who, o))  # noqa: E731
    families = {
        "A": {"A1": _contrast(per_fold, "gbm_dep_qos", "Analytic-I*", worst_case),
              "A2": _contrast(per_fold, "gbm_dep_qos", "InDeg", worst_case)},
        "B": {"B1": _contrast(per_fold, "gbm_dep_qos_dyn", "Analytic-I*", on("i_dyn")),
              "B2": _contrast(per_fold, "gbm_dep_qos_dyn", "InDeg", on("i_dyn"))},
        "C": {"C1": _contrast(per_fold, "gbm_dep_qos", "gbm_dep", on("i_star")),
              "C2": _contrast(per_fold, "gbm_dep_qos_dyn", "gbm_dep_dyn", on("i_dyn"))},
    }
    for fam in families.values():
        for k, p in holm({k: c["p"] for k, c in fam.items()}).items():
            fam[k]["p_holm"] = p
    return families


def _sig(c: Dict[str, Any]) -> bool:
    return c["p_holm"] < 0.05 and c["delta"] > 0


def decisions(fam: Dict[str, Any], zeroshot: Dict[str, Any]) -> Dict[str, Any]:
    A, B, C = fam["A"], fam["B"], fam["C"]
    if _sig(A["A1"]):
        a = "A"
    elif _sig(A["A2"]):
        a = "A′"
    else:
        a = "A″" + (" (unhedged: mean Δ vs Analytic-I* < 0)" if A["A1"]["delta"] < 0 else "")
    c = []
    if _sig(C["C2"]) and not _sig(C["C1"]):
        c.append("C")
    if _sig(C["C1"]):
        c.append("C′")
    if not _sig(C["C1"]) and not _sig(C["C2"]):
        c.append("C″")
    best_zs = max(zeroshot["arm_means"][a_]["i_star"] for a_ in ARMS)
    return {"A": a, "B": "B" if _sig(B["B1"]) else "B′", "C": c,
            "Z": {"best_arm_i_star": best_zs, "reach_i_star": zeroshot["reach_i_star"],
                  "applies": best_zs < zeroshot["reach_i_star"]},
            "R": "always"}


def _means(per: Dict[str, Any], names) -> Dict[str, Any]:
    return {
        "arm_means": {a: {o: _mean([per[n]["arms"][a].get(o, {}).get("rho") for n in names])
                          for o in ORACLES} for a in ARMS},
        "comparator_means": {r: {o: _mean([per[n]["comparators"][r].get(o, {}).get("rho")
                                           for n in names]) for o in ORACLES}
                             for r in COMPARATORS},
    }


def m1_sensitivity(idyn: Dict[str, Any]) -> Dict[str, Any]:
    """Comparators on I_dyn: published n=30, seed-42 full population, 5-seed full."""
    n30 = json.loads(IDYN_N30_CACHE.read_text())
    views = {"n30_seed42": lambda n: n30[n],
             "full_seed42": lambda n: idyn[n]["per_seed"]["42"],
             "full_5seed": lambda n: idyn[n]["mean"]}
    out: Dict[str, Any] = {}
    for view, get in views.items():
        per = {}
        for name in FOLDS:
            topo = _topology(name)
            flow = _flow(topo)
            per[name] = {r: score(p, get(name), flow)["rho"]
                         for r, p in comparator_scores(topo).items()}
        out[view] = {r: {"mean_rho": _mean([per[n][r] for n in FOLDS]),
                         "ci95": mean_ci([per[n][r] for n in FOLDS]),
                         "per_fold": {n: per[n][r] for n in FOLDS}} for r in COMPARATORS}
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def _write(path: Path, payload: Dict[str, Any], prov: Dict[str, Any]) -> None:
    payload["provenance"] = prov
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stage", choices=["labels", "all"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    prov = stamp(script="reproduce/oracle_robust_ltr.py", amendment=11, seeds=SEEDS,
                 idyn=IDYN_SETTINGS, learner="GradientBoostingRegressor (tab_gbm defaults)",
                 features={"S": S_NAMES, "Q": Q_NAMES}, arms={k: {"features": list(v[0]),
                                                                  "train": v[1]}
                                                              for k, v in ARMS.items()})
    names = list(FOLDS) + list(SYSTEMS)
    idyn = idyn_full_labels(names, args.workers)
    icomp = icomp_labels(args.workers)
    DATA_BENCHMARKS.mkdir(parents=True, exist_ok=True)
    _write(DATA_BENCHMARKS / "idyn_full_labels_jss12.json",
           {"settings": IDYN_SETTINGS, "seeds": SEEDS, "labels": idyn,
            "icomp_systems": {n: icomp[n] for n in SYSTEMS}}, prov)
    if args.stage == "labels":
        return 0

    oracles = {
        "i_star": {n: labels_for(_path(n))["impact"] for n in names},
        "i_dyn": {n: idyn[n]["mean"] for n in names},
        "i_comp": {n: icomp.get(n, {}) for n in names},
    }
    gates = {"G1": gate_g1(idyn), "G2": gate_g2(oracles)}
    print(f"G1 passed={gates['G1']['passed']} max|Δ|={gates['G1']['max_abs_diff']:.2e}; "
          f"G2 passed={gates['G2']['passed']} max|Δ|={gates['G2']['max_abs_diff']:.2e}")
    out_main = DATA_BENCHMARKS / "oracle_robust_ltr.json"
    if not (gates["G1"]["passed"] and gates["G2"]["passed"]):
        # Registered: nothing but the failure is reported.
        _write(out_main, {"gates": gates, "status": "gate failed; nothing else reported"}, prov)
        return 1

    print("LOSO:")
    loso = evaluate(list(FOLDS), list(FOLDS), oracles, loso=True)
    print("zero-shot:")
    zs = evaluate(list(SYSTEMS), list(FOLDS), oracles, loso=False)
    zs_means = _means(zs, list(SYSTEMS))
    zs_means["reach_i_star"] = zs_means["comparator_means"]["Reach"]["i_star"]

    label_cost = {n: float(np.mean(list(idyn[n]["seconds"].values()))) for n in names}
    _write(out_main, {
        "gates": gates,
        "loso": {"per_fold": loso, **_means(loso, list(FOLDS)),
                 "worst_case": {w: _mean([worst_case(loso[f], w) for f in FOLDS])
                                for w in list(ARMS) + list(COMPARATORS)}},
        "zeroshot": {"per_system": zs, **zs_means},
        "m1_sensitivity": m1_sensitivity(idyn),
        "cost": {"idyn_label_seconds_per_seed": label_cost,
                 "feature_seconds": {n: (loso.get(n) or zs.get(n))["feature_seconds"]
                                     for n in names}},
    }, prov)

    fam = contrasts(loso)
    _write(DATA_BENCHMARKS / "oracle_robust_significance.json",
           {"families": fam, "decisions": decisions(fam, zs_means),
            "note": "Holm within each family; not part of the 13-contrast omnibus."}, prov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
