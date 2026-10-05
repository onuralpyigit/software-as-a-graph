#!/usr/bin/env python3
"""
reproduce/referee_round12.py — round-12 referee analyses (PREREGISTRATION.md Amendment 17)
=========================================================================================
Reads the Amendment 17 sweep (``results/loso_amendment17_cpu.json``, its per-seed
predictions under ``output/loso_cpu_amendment17``), the zero-shot runs and the cached
I_dyn labels, and writes ``data/benchmarks/referee_round12_amendment17.json``:

  f12          tabular learners on top of Eq. 7 (stacking, residual), trained on the
               cached I_dyn labels -- no simulator runs
  amendment17  gate G0; F11 (oracle-aligned features zeroed); F12 (every learner on top
               of Eq. 7, Holm across the three); F13 (node-order permutation gate);
               every arm on I*, I_dyn-full and I_comp from per-seed predictions
  descriptive  creation-index leakage check; partial rho(., I* | Eq. 6) for learned and
               reference rows; InDeg reference vs the in-degree feature
  perm         Amendment 17b: GAT-P-QoS under three node-order permutations (seeds 17-19)
               against its published value; rule P1/P2 (descriptive)

``f12`` must run before ``amendment17`` (its per-fold rhos are read back from disk).

Usage:
    PYTHONPATH=. python reproduce/referee_round12.py f12 amendment17 descriptive perm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import spearmanr

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import stamp  # noqa: E402
from reproduce.idyn_rate_expansion import _design, closed_forms  # noqa: E402
from reproduce.oracle_robust_ltr import (  # noqa: E402
    IDYN_FULL_CACHE,
    _build_regressor,
    _cache_key,
    _topology,
    app_ids,
    pct,
)
from reproduce.referee_round7 import load_oracles  # noqa: E402
from reproduce.referee_round8 import (  # noqa: E402
    _partial,
    _per_seed_preds,
    load_idyn_full,
    per_seed_rho,
    seed_mean,
)
from reproduce.training_free_suite import (  # noqa: E402
    FOLDS,
    SEEDS,
    _flow,
    _mean,
    holm,
    indeg,
    mean_ci,
    paired,
    score,
    topo_qos,
)
from saag.core.graph_io import build_graph_from_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_BENCHMARKS = ROOT / "data" / "benchmarks"
RESULTS = ROOT / "results"

A17_ARTIFACT = "loso_amendment17_cpu.json"
A17_OUTPUT = "output/loso_cpu_amendment17"
A16_OUTPUT = "output/loso_cpu_amendment16"
F12_ARTIFACT = "referee_round12_f12.json"
LABELS = {
    "gl_proj_qos16_cap": "GAT-P-QoS", "gl_full_qos16_cap": "GAT-QoS",
    "gl_full_qos16_cap_rev": "GAT-QoS-R", "gin_proj_qos16": "GIN-P-QoS",
    "gl_proj_qos16_cap_idyn": "GAT-P-QoS-dyn",
    "gl_proj_qos16_cap_min": "GAT-P-QoS-min", "gl_full_qos16_cap_rev_min": "GAT-QoS-R-min",
    "gl_full_qos16_cap_min": "GAT-QoS-min", "gin_proj_qos16_min": "GIN-P-QoS-min",
    "gin_proj_qos16_const": "GIN-P-QoS-const", "gl_proj_qos16_cap_perm": "GAT-P-QoS-perm",
    "gl_proj_qos16_cap_idyn_rate": "GAT-P-QoS-dyn+Eq7",
}
#: Gate G0: comparators re-run in this invocation vs their published artifacts.
G0 = {"gl_proj_qos16_cap": "loso_dependency_graph_cpu.json",
      "gl_full_qos16_cap": "loso_hybrid_gat_cpu.json",
      "gl_full_qos16_cap_rev": "loso_amendment16_cpu.json",
      "gin_proj_qos16": "loso_amendment14_cpu.json",
      "gl_proj_qos16_cap_idyn": "loso_amendment14_cpu.json"}
G0_TOL = 1e-6
ZERO_SHOT = ("gl_proj_qos16_cap_min", "gl_full_qos16_cap_rev_min", "gl_full_qos16_cap_min",
             "gin_proj_qos16_min", "gin_proj_qos16_const")
#: Learned rows of Tables 5-6 whose saved per-seed predictions feed the partial rho.
PARTIAL_ROWS = {"GAT-P-QoS": (A17_OUTPUT, "gl_proj_qos16_cap"),
                "GAT-QoS": (A17_OUTPUT, "gl_full_qos16_cap"),
                "GAT-QoS-R": (A17_OUTPUT, "gl_full_qos16_cap_rev"),
                "GAT-P-QoS-min": (A17_OUTPUT, "gl_proj_qos16_cap_min"),
                "GAT-QoS-R-min": (A17_OUTPUT, "gl_full_qos16_cap_rev_min"),
                "HGT-QoS": (A16_OUTPUT, "hgl_qos"),
                "Hybrid-GAT": (A16_OUTPUT, "gl_qos16_prior"),
                "Hybrid-HGT": (A16_OUTPUT, "hgl_qos_prior")}

_PROV = stamp(script="reproduce/referee_round12.py", amendment=17)


def _write(name: str, payload: Dict[str, Any], **config: Any) -> Path:
    payload["provenance"] = {**_PROV, "config": {**_PROV["config"], **config}}
    text = json.dumps(payload, indent=2)
    for d in (DATA_BENCHMARKS, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)
    print(f"wrote {DATA_BENCHMARKS / name}")
    return DATA_BENCHMARKS / name


def _family(pairs, rho: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    folds = list(FOLDS)
    fam = {f"{a} vs {b}": paired([rho[a][f] for f in folds], [rho[b][f] for f in folds])
           for a, b in pairs}
    for k, p in holm({k: c["p"] for k, c in fam.items()}).items():
        fam[k]["p_holm"] = p
    return fam


def _sig_pos(c: Dict[str, Any]) -> bool:
    return c["p_holm"] < 0.05 and c["delta"] > 0


def _idyn_labels() -> Dict[str, Dict[str, float]]:
    cached = json.loads(IDYN_FULL_CACHE.read_text())
    if cached.get("key") != _cache_key():
        raise SystemExit(f"refusing: {IDYN_FULL_CACHE} was not built from this corpus")
    return {n: cached["labels"][n]["mean"] for n in FOLDS}


# ── F12 (tabular): learners on top of Eq. 7 ──────────────────────────────────

def cmd_f12(_: argparse.Namespace) -> int:
    """Amendment 11's learner and seeds on the S+Q design, (a) plus pct(Eq. 7) as a column,
    (b) trained on the residual pct(I_dyn) - pct(Eq. 7). S+Q itself is recomputed as gate."""
    idyn = _idyn_labels()
    cache: Dict[str, Any] = {}
    eq7 = {n: closed_forms(_topology(n))["Rate-I_dyn"] for n in FOLDS}
    arms = ("S+Q", "S+Q+Eq7", "resid-Eq7")
    out: Dict[str, Dict[str, float]] = {a: {} for a in arms}
    eq7_rho: Dict[str, float] = {}

    def design(n: str):
        apps, X = _design(n, list_q(), cache)
        e = pct(np.array([eq7[n].get(a, 0.0) for a in apps]))
        return apps, X, e

    for test in FOLDS:
        flow = _flow(_topology(test))
        xs, xs_e, ys, rs = [], [], [], []
        for tr in FOLDS:
            if tr == test:
                continue
            apps, X, e = design(tr)
            rows = [i for i, a in enumerate(apps) if a in idyn[tr]]
            y = pct(np.array([idyn[tr][apps[i]] for i in rows]))
            e_rows = pct(e[rows])
            xs.append(X[rows])
            xs_e.append(np.column_stack([X[rows], e_rows]))
            ys.append(y)
            rs.append(y - e_rows)
        apps, X, e = design(test)
        eq7_rho[test] = score(dict(zip(apps, e)), idyn[test], flow)["rho"]
        rhos: Dict[str, List[float]] = {a: [] for a in arms}
        for seed in SEEDS:
            m = _build_regressor(seed).fit(np.vstack(xs), np.concatenate(ys))
            rhos["S+Q"].append(score(dict(zip(apps, m.predict(X))), idyn[test], flow)["rho"])
            m = _build_regressor(seed).fit(np.vstack(xs_e), np.concatenate(ys))
            rhos["S+Q+Eq7"].append(score(dict(zip(apps, m.predict(np.column_stack([X, e])))),
                                         idyn[test], flow)["rho"])
            m = _build_regressor(seed).fit(np.vstack(xs), np.concatenate(rs))
            rhos["resid-Eq7"].append(score(dict(zip(apps, e + m.predict(X))), idyn[test], flow)["rho"])
        for a in arms:
            out[a][test] = float(np.mean(rhos[a]))
        print(f"  {test}: Eq7 {eq7_rho[test]:.3f} " + " ".join(f"{a} {out[a][test]:.3f}" for a in arms),
              flush=True)

    a11 = json.loads((DATA_BENCHMARKS / "oracle_robust_ltr.json").read_text())
    published = float(a11["loso"]["arm_means"]["gbm_dep_qos_dyn"]["i_dyn"])
    recomputed = _mean(list(out["S+Q"].values()))
    pub_eq7 = json.loads((DATA_BENCHMARKS / "idyn_rate_expansion.json").read_text())["per_fold"]
    eq7_gap = max(abs(eq7_rho[f] - pub_eq7[f]["Rate-I_dyn"]["i_dyn"]["rho"]) for f in FOLDS)
    gate = {"S+Q": {"published": published, "recomputed": recomputed,
                    "abs_diff": abs(recomputed - published),
                    "passed": abs(recomputed - published) < G0_TOL},
            "Eq7": {"max_abs_diff": eq7_gap, "passed": eq7_gap < G0_TOL}}
    print("G0 (f12):", gate)
    _write(F12_ARTIFACT, {"gate_G0": gate, "per_fold": {**out, "Eq7": eq7_rho},
                          "means": {a: _mean(list(v.values())) for a, v in {**out, "Eq7": eq7_rho}.items()}},
           experiment="Amendment 17 F12 tabular arms")
    return 0 if all(g["passed"] for g in gate.values()) else 1


def list_q():
    from reproduce.oracle_robust_ltr import Q_NAMES

    return list(Q_NAMES)


# ── Amendment 17: gate, families, cells ──────────────────────────────────────

def cmd_amendment17(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    ps = {v: per_seed_rho(A17_ARTIFACT, v) for v in LABELS}
    gate = {}
    for v, art in G0.items():
        pub = per_seed_rho(art, v)
        diff = max(abs(a - b) for f in folds for a, b in zip(ps[v][f], pub[f]))
        gate[v] = {"max_abs_diff": diff, "passed": diff < G0_TOL}
    print("G0:", {v: round(g["max_abs_diff"], 9) for v, g in gate.items()})
    rho = {LABELS[v]: seed_mean(p) for v, p in ps.items()}

    # Every arm on every oracle from per-seed predictions.
    graphs = {f: build_graph_from_json(_topology(f)) for f in folds}
    oracles = load_oracles(folds)
    oracles["i_dyn"] = {f: v for f, v in load_idyn_full().items() if f in FOLDS}
    cells: Dict[str, Any] = {}
    for v, lab in LABELS.items():
        preds = _per_seed_preds(f"{A17_OUTPUT}/{v}")
        cells[lab] = {}
        for o in ("i_star", "i_dyn", "i_comp"):
            per = {f: [score(p, oracles[o][f], graphs[f]) for p in preds.get(f, {}).values()]
                   for f in folds}
            cells[lab][o] = {f: _mean([s["rho"] for s in per[f]]) for f in folds}
            if o == "i_star":
                cells[lab]["i_star_active"] = {
                    f: _mean([s["rho_active"] for s in per[f] if s["rho_active"] is not None])
                    for f in folds}

    f12 = json.loads((DATA_BENCHMARKS / F12_ARTIFACT).read_text())
    dyn = {"Eq7": f12["per_fold"]["Eq7"], "GBM-P-QoS-dyn+Eq7": f12["per_fold"]["S+Q+Eq7"],
           "GBM-dyn-resid": f12["per_fold"]["resid-Eq7"], "GBM-P-QoS-dyn": f12["per_fold"]["S+Q"],
           "GAT-P-QoS-dyn+Eq7": cells["GAT-P-QoS-dyn+Eq7"]["i_dyn"],
           "GAT-P-QoS-dyn": cells["GAT-P-QoS-dyn"]["i_dyn"]}
    fam = {
        "F11": _family([("GAT-P-QoS-min", "GAT-QoS-R-min"), ("GAT-P-QoS-min", "GAT-P-QoS"),
                        ("GIN-P-QoS-min", "GAT-P-QoS-min")], rho),
        "F12": _family([("GBM-P-QoS-dyn+Eq7", "Eq7"), ("GBM-dyn-resid", "Eq7"),
                        ("GAT-P-QoS-dyn+Eq7", "Eq7")], dyn),
    }
    f13 = paired([rho["GAT-P-QoS-perm"][f] for f in folds], [rho["GAT-P-QoS"][f] for f in folds])
    ioe = json.loads((RESULTS / "independent_oracle_evaluation.json").read_text())["per_fold"]
    indeg_rho = {f: float(ioe[f]["i_star"]["InDeg"]["rho"]) for f in folds}
    descriptive = {
        "GIN-P-QoS-const vs InDeg": paired([rho["GIN-P-QoS-const"][f] for f in folds],
                                           [indeg_rho[f] for f in folds]),
        "GAT-QoS-min vs GAT-QoS": paired([rho["GAT-QoS-min"][f] for f in folds],
                                         [rho["GAT-QoS"][f] for f in folds]),
        "GAT-QoS-R-min vs GAT-QoS-R": paired([rho["GAT-QoS-R-min"][f] for f in folds],
                                             [rho["GAT-QoS-R"][f] for f in folds]),
        "GAT-P-QoS-dyn+Eq7 vs GAT-P-QoS-dyn (I_dyn)": paired(
            [dyn["GAT-P-QoS-dyn+Eq7"][f] for f in folds], [dyn["GAT-P-QoS-dyn"][f] for f in folds]),
    }
    f11a = fam["F11"]["GAT-P-QoS-min vs GAT-QoS-R-min"]
    rules = {
        "F11": "F11a" if _sig_pos(f11a) else "F11b",
        "F12": "F12a" if any(_sig_pos(c) for c in fam["F12"].values()) else "F12b",
        "F13": "F13a" if (f13["p"] < 0.05 or abs(f13["delta"]) > 0.02) else "F13b",
    }
    summary = {}
    for lab in LABELS.values():
        row = {"loso_i_star": _mean(list(rho[lab].values()))}
        for o in ("i_star", "i_dyn", "i_comp", "i_star_active"):
            row[o] = _mean(list(cells[lab][o].values()))
        summary[lab] = row
    summary["dyn_means"] = {k: _mean(list(v.values())) for k, v in dyn.items()}
    zs = {}
    for v in ZERO_SHOT:
        p = RESULTS / f"realworld_zeroshot_{v}_amendment17.json"
        if p.exists():
            z = json.loads(p.read_text())
            active = [s.get("mean_rho_positive") for s in z["per_system"].values()
                      if s.get("mean_rho_positive") is not None]
            zs[LABELS[v]] = {"mean_rho": z.get("mean_rho_across_systems"),
                             "mean_rho_positive": _mean(active),
                             "per_system": {k: s.get("mean_rho") for k, s in z["per_system"].items()}}

    for k, v in summary.items():
        print(f"{k:20s} " + " ".join(f"{o}={x:.3f}" for o, x in v.items() if x is not None))
    for name, fm in {**fam, "F13": {"GAT-P-QoS-perm vs GAT-P-QoS": f13}, "desc": descriptive}.items():
        for k, c in fm.items():
            ph = c.get("p_holm")
            print(f"{name:4s} {k:44s} d={c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
                  f"won={c['won']}/12 p={c['p']:.4f}" + (f" p_holm={ph:.4f}" if ph is not None else ""))
    print("zero-shot:", {k: round(v["mean_rho"], 3) for k, v in zs.items() if v["mean_rho"] is not None})
    print("rules:", rules)
    _write("referee_round12_amendment17.json",
           {"gate_G0": {**gate, "f12": f12["gate_G0"]}, **fam, "F13": f13,
            "descriptive": descriptive, "decision_rules": rules, "summary": summary,
            "zero_shot": zs,
            "per_fold": {"loso_i_star": rho, "cells": cells, "i_dyn": dyn, "indeg": indeg_rho}},
           experiment="Amendment 17 arms")
    return 0 if all(g["passed"] for g in gate.values()) and all(
        g["passed"] for g in f12["gate_G0"].values()) else 1


# ── descriptive: leakage, symmetric circularity, feature vs reference ────────

def _creation_index(node_id: str):
    m = re.fullmatch(r"A(\d+)", node_id)
    return int(m.group(1)) if m else None


def cmd_descriptive(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    oracles = load_oracles(folds)
    oracles["i_dyn"] = {f: v for f, v in load_idyn_full().items() if f in FOLDS}

    # (1) Creation index against the labels.
    leak = {}
    for f in folds:
        apps = app_ids(_topology(f))
        idx = {a: _creation_index(a) for a in apps}
        row = {}
        for o in ("i_star", "i_dyn"):
            keep = [a for a in apps if idx[a] is not None and a in oracles[o][f]]
            row[o] = float(spearmanr([idx[a] for a in keep], [oracles[o][f][a] for a in keep]).correlation)
        leak[f] = row

    # (2) Partial rho(., I* | Eq. 6): agreement with I* beyond its first-order expansion.
    eq6 = {f: closed_forms(_topology(f))["Analytic-I*"] for f in folds}
    partial: Dict[str, Dict[str, Any]] = {}
    for lab, (out_dir, v) in PARTIAL_ROWS.items():
        preds = _per_seed_preds(f"{out_dir}/{v}")
        if not preds:
            print(f"  skip {lab}: no predictions under {out_dir}/{v}")
            continue
        per = {}
        for f in folds:
            apps = app_ids(_topology(f))
            vals = [_partial(p, oracles["i_star"][f], eq6[f], apps) for p in preds.get(f, {}).values()]
            vals = [x for x in vals if x is not None]
            per[f] = _mean(vals) if vals else None
        partial[lab] = per
    for lab, fn in (("InDeg", indeg), ("Topo-QoS", topo_qos)):
        per = {}
        for f in folds:
            topo = _topology(f)
            per[f] = _partial(fn(_flow(topo)), oracles["i_star"][f], eq6[f], app_ids(topo))
        partial[lab] = per
    partial_means = {k: _mean([x for x in v.values() if x is not None]) for k, v in partial.items()}

    # (3) The InDeg reference (direct subscriptions) against the in-degree feature (which
    # also follows up to three USES links).
    feat_vs_ref = {}
    for f in folds:
        comps = json.loads((ROOT / "output" / "loso_cache" / f / "structural_metrics.json")
                           .read_text())["structural_analysis"]["components"]
        feat = {str(c["id"]): float(c["in_degree_raw"]) for c in comps if c.get("type") == "Application"}
        ref = indeg(_flow(_topology(f)))
        apps = [a for a in feat if a in ref]
        differ = sum(1 for a in apps if feat[a] != ref[a])
        feat_vs_ref[f] = {"n_apps": len(apps), "n_differ": differ,
                          "spearman": float(spearmanr([feat[a] for a in apps],
                                                      [ref[a] for a in apps]).correlation)}
    tot = sum(v["n_apps"] for v in feat_vs_ref.values())
    dif = sum(v["n_differ"] for v in feat_vs_ref.values())

    # (4) The corrected baseline (articulation term restored) for Table 5's extra row.
    tap = json.loads((DATA_BENCHMARKS / "topo_ap_sensitivity.json").read_text())["per_scenario"]
    ap_vals = [float(tap[f]["topo_qos_ap_restored"]) for f in folds]
    topo_ap = {"mean": _mean(ap_vals), "ci95": mean_ci(ap_vals)}

    print("creation index vs labels:", {f: {o: round(x, 2) for o, x in r.items()} for f, r in leak.items()})
    print("partial rho(., I* | Eq.6):", {k: round(v, 3) for k, v in partial_means.items()})
    print(f"InDeg ref vs feature: {dif}/{tot} Applications differ; "
          f"min fold spearman {min(v['spearman'] for v in feat_vs_ref.values()):.3f}")
    _write("referee_round12_descriptive.json",
           {"creation_index": leak, "partial_istar_given_eq6": {"per_fold": partial, "means": partial_means},
            "indeg_feature_vs_reference": {"per_fold": feat_vs_ref, "n_apps": tot, "n_differ": dif},
            "topo_qos_ap": topo_ap},
           experiment="Amendment 17 descriptive analyses")
    return 0


# ── Amendment 17b: three node-order permutations ─────────────────────────────

def cmd_perm(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    pub = per_seed_rho("loso_dependency_graph_cpu.json", "gl_proj_qos16_cap")
    rerun = per_seed_rho("loso_amendment17b_cpu.json", "gl_proj_qos16_cap")
    diff = max(abs(a - b) for f in folds for a, b in zip(rerun[f], pub[f]))
    gate = {"gl_proj_qos16_cap": {"max_abs_diff": diff, "passed": diff < G0_TOL}}
    published = seed_mean(pub)
    perms = {17: seed_mean(per_seed_rho(A17_ARTIFACT, "gl_proj_qos16_cap_perm")),
             18: seed_mean(per_seed_rho("loso_amendment17b_cpu.json", "gl_proj_qos16_cap_perm18")),
             19: seed_mean(per_seed_rho("loso_amendment17b_cpu.json", "gl_proj_qos16_cap_perm19"))}
    per_fold = {}
    for f in folds:
        vals = [perms[k][f] for k in sorted(perms)]
        per_fold[f] = {"published": published[f], "permuted": vals, "mean": float(np.mean(vals)),
                       "spread": float(max(vals) - min(vals)),
                       "published_above_all": bool(published[f] > max(vals))}
    perm_means = {k: _mean(list(v.values())) for k, v in perms.items()}
    perm_mean = _mean([per_fold[f]["mean"] for f in folds])
    spread = _mean([per_fold[f]["spread"] for f in folds])
    gap = _mean(list(published.values())) - perm_mean
    rule = "P1" if gap > spread else "P2"
    vs_mean = paired([published[f] for f in folds], [per_fold[f]["mean"] for f in folds])
    above = sum(per_fold[f]["published_above_all"] for f in folds)
    print("G0:", gate)
    print(f"published {_mean(list(published.values())):.3f}; permutations "
          + ", ".join(f"{k}: {v:.3f}" for k, v in perm_means.items())
          + f"; mean {perm_mean:.3f}; gap {gap:+.3f}; mean per-fold spread {spread:.3f}; "
          f"published above all three on {above}/12 folds; rule {rule}")
    _write("referee_round12_perm.json",
           {"gate_G0": gate, "published_mean": _mean(list(published.values())),
            "permutation_means": {str(k): v for k, v in perm_means.items()},
            "permuted_mean": perm_mean, "gap": gap, "mean_spread": spread,
            "folds_published_above_all": above, "published_vs_permuted_mean": vs_mean,
            "decision_rule": rule, "per_fold": per_fold},
           experiment="Amendment 17b node-order permutations")
    return 0 if gate["gl_proj_qos16_cap"]["passed"] else 1


def main() -> int:
    stages = {"f12": cmd_f12, "amendment17": cmd_amendment17, "descriptive": cmd_descriptive,
              "perm": cmd_perm}
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stages", nargs="+", choices=list(stages))
    args = ap.parse_args()
    return max(stages[s](args) for s in args.stages)


if __name__ == "__main__":
    sys.exit(main())
