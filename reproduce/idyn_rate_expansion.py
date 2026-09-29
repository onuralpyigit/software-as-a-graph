#!/usr/bin/env python3
"""
reproduce/idyn_rate_expansion.py — the rate-weighted first-order expansion of I_dyn (Amendment 15)
===================================================================================================

Amendment 15 (post hoc, exploratory; docs/research/jss/PREREGISTRATION.md). The advisor's v5
review asked whether the gradient-boosted I_dyn approximation of Amendment 11
(``gbm_dep_qos_dyn``) beats the closed forms because it *learns*, or because it reads declared
publication rates and payload sizes that the unweighted first-order expansion of I* (Eq. 6,
``Analytic-I*``) ignores. Two analyses answer it:

1. **Closed forms that read the same inputs.** Eq. 6 weighted by the declared publication rate
   ``r_t`` of each topic (Eq. 7, the I_dyn reference)::

       I_dyn,1^rate(v) = sum_{t in pub(v)} (r_t / |pub(t)|) * |sub(t)|

   plus a payload-weighted variant (``r_t * B_t``) and the bare publication rate of ``v``. Each is
   scored against I*, I_dyn and I_comp on the twelve LOSO folds and zero-shot on the five system
   models, exactly as Amendment 11 scores its comparators.
2. **Input attribution for the learned approximation.** The Amendment 11 learner and seeds,
   trained on I_dyn with the structural set S plus (a) nothing, (b) only the rate and payload
   columns of Q, (c) only the QoS-policy columns of Q, (d) all of Q. Arm (d) is the published
   ``gbm_dep_qos_dyn`` and must reproduce it (gate G_A15).

No simulator runs here: I_dyn comes from the Amendment 11 label cache (refused unless its key
matches), I* from the published label cache and I_comp from the published I_comp cache.

Usage:
    PYTHONPATH=. python reproduce/idyn_rate_expansion.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import stamp  # noqa: E402
from reproduce.oracle_robust_ltr import (  # noqa: E402
    DATA_BENCHMARKS,
    IDYN_FULL_CACHE,
    Q_NAMES,
    S_NAMES,
    _build_regressor,
    _cache_key,
    _edges,
    _path,
    _topology,
    app_ids,
    features,
    icomp_labels,
    pct,
)
from reproduce.referee_round8 import _partial  # noqa: E402
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
    score,
)

OUT = DATA_BENCHMARKS / "idyn_rate_expansion.json"
ORACLE_ROBUST = DATA_BENCHMARKS / "oracle_robust_ltr.json"
ORACLES = ("i_star", "i_dyn", "i_comp")
CLOSED_FORMS = ("Analytic-I*", "Rate-I_dyn", "RatePayload-I_dyn", "PubRate")
GATE_TOL = 1e-3

RATE_COLS = ["PubRate", "PubBytes"]
POLICY_COLS = [c for c in Q_NAMES if c not in RATE_COLS]
#: Attribution arms: columns of Q appended to S (None = S alone).
ATTRIBUTION: Dict[str, Any] = {
    "S": None,
    "S+rate,payload": RATE_COLS,
    "S+QoS-policy": POLICY_COLS,
    "S+Q": list(Q_NAMES),
}


# ── closed forms ──────────────────────────────────────────────────────────────

def closed_forms(topo: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Eq. 6, Eq. 7, the payload-weighted variant and the bare publication rate."""
    pubs: Dict[str, set] = {}
    subs: Dict[str, set] = {}
    for a, t in _edges(topo, "publishes_to"):
        pubs.setdefault(t, set()).add(a)
    for a, t in _edges(topo, "subscribes_to"):
        subs.setdefault(t, set()).add(a)
    topics = {str(t["id"]): t for t in topo.get("topics", [])}
    out: Dict[str, Dict[str, float]] = {k: {} for k in CLOSED_FORMS}
    for a in app_ids(topo):
        v = dict.fromkeys(CLOSED_FORMS, 0.0)
        for t, p in pubs.items():
            if a not in p:
                continue
            r = float(topics.get(t, {}).get("frequency", 0.0) or 0.0)
            b = float(topics.get(t, {}).get("size", 0.0) or 0.0)
            base = len(subs.get(t, ())) / len(p)
            v["Analytic-I*"] += base
            v["Rate-I_dyn"] += r * base
            v["RatePayload-I_dyn"] += r * b * base
            v["PubRate"] += r
        for k, x in v.items():
            out[k][a] = x
    return out


def _timed_rate(topo: Dict[str, Any], repeats: int = 5) -> float:
    """Best-of-*repeats* wall-clock (s) to compute every closed form from the parsed manifest."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        closed_forms(topo)
        best = min(best, time.perf_counter() - t0)
    return best


# ── attribution ───────────────────────────────────────────────────────────────

def _design(name: str, cols, cache: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    if name not in cache:
        cache[name] = features(_topology(name))
    apps, m = cache[name]
    if cols is None:
        return apps, m["S"]
    idx = [Q_NAMES.index(c) for c in cols]
    return apps, np.column_stack([m["S"], m["Q"][:, idx]])


def attribution(idyn: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """LOSO rho on I_dyn per arm and fold, mean over the Amendment 11 seeds."""
    cache: Dict[str, Any] = {}
    out: Dict[str, Dict[str, float]] = {arm: {} for arm in ATTRIBUTION}
    for test in FOLDS:
        flow = _flow(_topology(test))
        for arm, cols in ATTRIBUTION.items():
            xs, ys = [], []
            for tr in FOLDS:
                if tr == test:
                    continue
                apps, X = _design(tr, cols, cache)
                rows = [i for i, a in enumerate(apps) if a in idyn[tr]]
                xs.append(X[rows])
                ys.append(pct(np.array([idyn[tr][apps[i]] for i in rows])))
            apps, X = _design(test, cols, cache)
            rhos = []
            for seed in SEEDS:
                model = _build_regressor(seed)
                model.fit(np.vstack(xs), np.concatenate(ys))
                rhos.append(score(dict(zip(apps, model.predict(X))), idyn[test], flow)["rho"])
            out[arm][test] = float(np.mean(rhos))
        print(f"  {test}: " + ", ".join(f"{a} {out[a][test]:.3f}" for a in ATTRIBUTION), flush=True)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def _summ(xs: List[float]) -> Dict[str, Any]:
    return {"mean": _mean(xs), "ci95": mean_ci(xs)}


def main() -> int:
    cached = json.loads(IDYN_FULL_CACHE.read_text())
    if cached.get("key") != _cache_key():
        print(f"refusing: {IDYN_FULL_CACHE} was not built from this corpus and I_dyn setting; "
              "run `make -f reproduce/Makefile rq-oracle-robust` first")
        return 1
    names = list(FOLDS) + list(SYSTEMS)
    icomp = icomp_labels(workers=1)
    oracles = {
        "i_star": {n: labels_for(_path(n))["impact"] for n in names},
        "i_dyn": {n: cached["labels"][n]["mean"] for n in names},
        "i_comp": {n: icomp.get(n, {}) for n in names},
    }
    a11 = json.loads(ORACLE_ROBUST.read_text())

    per: Dict[str, Any] = {}
    for name in names:
        topo = _topology(name)
        flow = _flow(topo)
        apps = app_ids(topo)
        forms = closed_forms(topo)
        row: Dict[str, Any] = {"seconds": _timed_rate(topo), "n_apps": len(apps)}
        for k, pred in forms.items():
            row[k] = {o: score(pred, oracles[o][name], flow)
                      for o in ORACLES if oracles[o].get(name)}
            row[k]["partial_idyn_given_istar"] = _partial(
                pred, oracles["i_dyn"][name], oracles["i_star"][name], apps)
        row["InDeg"] = {"i_dyn": score(indeg(flow), oracles["i_dyn"][name], flow)}
        per[name] = row

    def rho(names_: List[str], k: str, o: str) -> List[float]:
        return [per[n][k][o]["rho"] for n in names_ if o in per[n][k]]

    summary: Dict[str, Any] = {}
    for scope, ns in (("loso", list(FOLDS)), ("zeroshot", list(SYSTEMS))):
        summary[scope] = {}
        for k in CLOSED_FORMS:
            s = {o: _summ(rho(ns, k, o)) for o in ORACLES if rho(ns, k, o)}
            s["i_dyn_active"] = _mean([per[n][k]["i_dyn"]["rho_active"] for n in ns])
            if scope == "loso":
                s["partial_idyn_given_istar"] = _summ(
                    [per[n][k]["partial_idyn_given_istar"] for n in ns])
            summary[scope][k] = s

    # Gate: Analytic-I* reproduces the published Amendment 11 comparator on every fold and oracle.
    worst = 0.0
    for n in names:
        blk = a11["loso"]["per_fold"].get(n) or a11["zeroshot"]["per_system"][n]
        for o in ORACLES:
            if o in per[n]["Analytic-I*"] and o in blk["comparators"]["Analytic-I*"]:
                worst = max(worst, abs(per[n]["Analytic-I*"][o]["rho"]
                                       - blk["comparators"]["Analytic-I*"][o]["rho"]))

    # Contrasts on I_dyn: Eq. 7 against the learned approximation, Eq. 6 and InDeg.
    gbm = {n: (a11["loso"]["per_fold"].get(n) or a11["zeroshot"]["per_system"][n])
           ["arms"]["gbm_dep_qos_dyn"]["i_dyn"]["rho"] for n in names}
    contrasts: Dict[str, Any] = {}
    for scope, ns in (("loso", list(FOLDS)), ("zeroshot", list(SYSTEMS))):
        rate = rho(ns, "Rate-I_dyn", "i_dyn")
        fam = {
            "Rate-I_dyn vs gbm_dep_qos_dyn": paired(rate, [gbm[n] for n in ns]),
            "Rate-I_dyn vs Analytic-I*": paired(rate, rho(ns, "Analytic-I*", "i_dyn")),
            "Rate-I_dyn vs InDeg": paired(rate, [per[n]["InDeg"]["i_dyn"]["rho"] for n in ns]),
        }
        ph = holm({k: v["p"] for k, v in fam.items()})
        for k in fam:
            fam[k]["p_holm"] = ph[k]
        contrasts[scope] = fam

    print("attribution (LOSO, I_dyn):")
    att = attribution(oracles["i_dyn"])
    att_means = {a: _summ([att[a][f] for f in FOLDS]) for a in ATTRIBUTION}
    published = a11["loso"]["arm_means"]["gbm_dep_qos_dyn"]["i_dyn"]
    gate_a15 = abs(att_means["S+Q"]["mean"] - published)
    att_fam = {
        "S+rate,payload vs S": paired([att["S+rate,payload"][f] for f in FOLDS], [att["S"][f] for f in FOLDS]),
        "S+QoS-policy vs S": paired([att["S+QoS-policy"][f] for f in FOLDS], [att["S"][f] for f in FOLDS]),
        "S+Q vs S+rate,payload": paired([att["S+Q"][f] for f in FOLDS], [att["S+rate,payload"][f] for f in FOLDS]),
    }
    ph = holm({k: v["p"] for k, v in att_fam.items()})
    for k in att_fam:
        att_fam[k]["p_holm"] = ph[k]

    gates = {
        "G_analytic": {"passed": bool(worst <= GATE_TOL), "tolerance": GATE_TOL, "max_abs_diff": worst},
        "G_A15": {"passed": bool(gate_a15 <= GATE_TOL), "tolerance": GATE_TOL,
                  "published_gbm_dep_qos_dyn": published, "recomputed": att_means["S+Q"]["mean"],
                  "abs_diff": gate_a15},
    }
    secs = [per[n]["seconds"] for n in names]
    payload = {
        "gates": gates,
        "summary": summary,
        "contrasts": contrasts,
        "attribution": {"columns": {a: c for a, c in ATTRIBUTION.items()},
                        "means": att_means, "per_fold": att, "contrasts": att_fam},
        "timing": {"closed_form_seconds_max": max(secs), "closed_form_seconds_median": float(np.median(secs))},
        "per_fold": per,
        "provenance": stamp(script="reproduce/idyn_rate_expansion.py", amendment=15, seeds=SEEDS,
                            idyn_cache_key=_cache_key(), features={"S": S_NAMES, "Q": Q_NAMES}),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"gates: analytic max|Δ|={worst:.2e} passed={gates['G_analytic']['passed']}; "
          f"A15 S+Q {att_means['S+Q']['mean']:.4f} vs {published:.4f} passed={gates['G_A15']['passed']}")
    for scope in ("loso", "zeroshot"):
        print(scope, {k: round(v["i_dyn"]["mean"], 4) for k, v in summary[scope].items()})
    print("attribution", {a: round(v["mean"], 4) for a, v in att_means.items()})
    print(f"wrote {OUT}")
    return 0 if all(g["passed"] for g in gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
