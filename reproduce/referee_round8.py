#!/usr/bin/env python3
"""
reproduce/referee_round8.py — round-8 referee analyses (PREREGISTRATION.md Amendment 14)
======================================================================================

Everything here scores artifacts that already exist, except ``amendment14`` (which
reads the new sweep) and ``cost`` (which times). Stages:

  hybrid        F7: each hybrid vs its own learner (Amendments 5/6, re-read as a
                gate), vs unweighted betweenness and vs constant topic weights;
                |V_app|-weighted sensitivity with an exact sign-flip p, and a
                Nadeau-Bengio corrected t
  tost          F3 on the published GAT-P-QoS: distance to InDeg, 90% CI, TOST at
                +-0.05 (t and Wilcoxon), per-seed mean and seed ensemble; plus the
                Nadeau-Bengio corrected t for the plan's two contrasts and the hybrids
  table7        Table 7 on Amendment 11's full-population I_dyn: every ranker,
                learned ones included, with partial rho given I* and given
                Analytic-I*; the published n=30 sample kept as the sensitivity check
  hierarchical  fold -> seed bootstrap CIs for the learned rows of Table 6
  amendment14   F1-F6 and descriptive cells for the Amendment 14 arms, plus the
                degree-leak table; gate G0 on the re-run comparators
  cost          like-for-like timing: analysis gate / app layer vs one I* pass vs
                the five-seed sweep vs the counting path, same graphs, one session

Usage:
    PYTHONPATH=. python reproduce/referee_round8.py hybrid tost table7 hierarchical
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr, t as t_dist, wilcoxon

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import stamp  # noqa: E402
from reproduce.referee_round7 import (  # noqa: E402
    IDYN_N30,
    LEARNED,
    _apps,
    _learned_preds,
    _topology,
    load_oracles,
    partial_spearman,
    raw_rankers,
)
from reproduce.training_free_suite import (  # noqa: E402
    FOLDS,
    SYSTEMS,
    _mean,
    holm,
    mean_ci,
    paired,
    score,
)
from saag.core.graph_io import build_graph_from_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_BENCHMARKS = ROOT / "data" / "benchmarks"
RESULTS = ROOT / "results"
IDYN_FULL = DATA_BENCHMARKS / "idyn_full_labels_jss12.json"

#: Registered in Amendment 14 before any Amendment 14 result existed; see its note on
#: the TOST preview, which makes this margin not blind.
MARGIN = 0.05
B_HIER = 10_000

#: Per-fold, per-seed rho of every published learned row: (artifact, variant id).
PER_SEED = {
    "HGT-QoS": ("loso_hybrid_cpu.json", "hgl_qos"),
    "Hybrid-HGT": ("loso_hybrid_cpu.json", "hgl_qos_prior"),
    "GAT-QoS": ("loso_hybrid_gat_cpu.json", "gl_full_qos16_cap"),
    "Hybrid-GAT": ("loso_hybrid_gat_cpu.json", "gl_qos16_prior"),
    "GAT-P-QoS": ("loso_dependency_graph_cpu.json", "gl_proj_qos16_cap"),
    "HGT-P-QoS": ("loso_dependency_graph_cpu.json", "hgl_proj_qos"),
    "GAT-P+InDeg": ("loso_dependency_graph_cpu.json", "gl_proj_qos16_indeg_prior"),
    "Topo-QoS": ("loso_hybrid_cpu.json", "topo_qos"),
    "HGT": ("loso_rq2_matched.json", "hgl"),
    "GAT": ("loso_rq2_matched.json", "gl_full_cap"),
}
PUBLISHED_I_STAR = {"HGT-QoS": 0.622, "GAT-QoS": 0.635, "Hybrid-HGT": 0.657, "Hybrid-GAT": 0.683,
                    "GAT-P-QoS": 0.748, "HGT-P-QoS": 0.514, "GAT-P+InDeg": 0.758}

#: Taken once, at process start (see referee_round7._PROV).
_PROV = stamp(script="reproduce/referee_round8.py", amendment=14)


# ── shared ────────────────────────────────────────────────────────────────────

def _write(name: str, payload: Dict[str, Any], **config: Any) -> Path:
    payload["provenance"] = {**_PROV, "config": {**_PROV["config"], **config}}
    text = json.dumps(payload, indent=2)
    for d in (DATA_BENCHMARKS, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)
    print(f"wrote {DATA_BENCHMARKS / name}")
    return DATA_BENCHMARKS / name


def per_seed_rho(artifact: str, variant: str) -> Dict[str, List[float]]:
    """``{fold: [rho per seed, in seed order]}`` from a LOSO sweep artifact."""
    d = json.loads((RESULTS / artifact).read_text())["per_variant_results"][variant]
    return {f["holdout_id"]: [float(s["spearman_rho"]) for s in
                              sorted(f["seed_metrics"], key=lambda s: s["seed"])]
            for f in d["folds"]}


def seed_mean(per_seed: Dict[str, List[float]]) -> Dict[str, float]:
    return {f: float(np.mean(v)) for f, v in per_seed.items()}


def n_apps() -> Dict[str, int]:
    ioe = json.loads((RESULTS / "independent_oracle_evaluation.json").read_text())["per_fold"]
    return {f: int(ioe[f]["i_star"]["InDeg"]["n"]) for f in FOLDS}


def load_idyn_full() -> Dict[str, Dict[str, float]]:
    data = json.loads(IDYN_FULL.read_text())
    return {sid: {str(k): float(v) for k, v in lab["mean"].items()}
            for sid, lab in data["labels"].items()}


def tost_t(d: Sequence[float], margin: float, var_factor: Optional[float] = None) -> Dict[str, Any]:
    """Two one-sided t tests of |mean(d)| < margin; ``var_factor`` defaults to 1/n.

    With ``var_factor = 1/n + n_test/n_train`` this is the Nadeau-Bengio corrected
    version for overlapping training sets.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    vf = 1.0 / n if var_factor is None else var_factor
    se = float(np.sqrt(vf * np.var(d, ddof=1)))
    m = float(d.mean())
    df = n - 1
    p_lower = float(1 - t_dist.cdf((m + margin) / se, df))
    p_upper = float(t_dist.cdf((m - margin) / se, df))
    half = float(t_dist.ppf(0.95, df) * se)
    ci90 = [m - half, m + half]
    return {"mean": m, "se": se, "df": df, "margin": margin, "p": max(p_lower, p_upper),
            "ci90": ci90, "equivalence_bound": max(abs(ci90[0]), abs(ci90[1]))}


def tost_wilcoxon(d: Sequence[float], margin: float) -> Dict[str, Any]:
    d = np.asarray(d, dtype=float)
    p1 = float(wilcoxon(d + margin, alternative="greater").pvalue)
    p2 = float(wilcoxon(d - margin, alternative="less").pvalue)
    return {"margin": margin, "p": max(p1, p2)}


def nb_corrected_t(d: Sequence[float], test_train_ratio: float) -> Dict[str, Any]:
    """Nadeau-Bengio corrected resampled t for a mean paired difference."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    se = float(np.sqrt((1.0 / n + test_train_ratio) * np.var(d, ddof=1)))
    tt = float(d.mean() / se) if se > 0 else float("inf")
    return {"mean": float(d.mean()), "t": tt, "df": n - 1, "ratio": test_train_ratio,
            "p": float(2 * (1 - t_dist.cdf(abs(tt), n - 1)))}


def weighted_signflip(d: Sequence[float], w: Sequence[float]) -> Dict[str, Any]:
    """|V_app|-weighted mean difference with an exact two-sided sign-flip p."""
    d, w = np.asarray(d, dtype=float), np.asarray(w, dtype=float)
    obs = float((w * d).sum() / w.sum())
    flips = np.array(list(itertools.product((-1.0, 1.0), repeat=len(d))))
    null = (flips * (w * d)).sum(axis=1) / w.sum()
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(d), len(d))
        boots.append(float((w[i] * d[i]).sum() / w[i].sum()))
    return {"delta_weighted": obs, "p_signflip": float(np.mean(np.abs(null) >= abs(obs) - 1e-12)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def _ratios() -> Dict[str, float]:
    """Nadeau-Bengio test/train ratios: scenario as the unit, and Applications."""
    n = n_apps()
    tot = sum(n.values())
    return {"scenario": 1.0 / 11.0,
            "applications": float(np.mean([n[f] / (tot - n[f]) for f in FOLDS]))}


def _contrast(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, Any]:
    folds = list(FOLDS)
    return paired([a[f] for f in folds], [b[f] for f in folds])


# ── F7: hybrid attribution ────────────────────────────────────────────────────

def cmd_hybrid(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    rho = {k: seed_mean(per_seed_rho(*PER_SEED[k]))
           for k in ("HGT-QoS", "Hybrid-HGT", "GAT-QoS", "Hybrid-GAT", "Topo-QoS")}
    tf = json.loads((RESULTS / "tf_baselines.json").read_text())["per_fold"]
    qa = json.loads((RESULTS / "qos_attribution_controls.json").read_text())["per_fold"]

    def _r(v):
        return float(v["rho"] if isinstance(v, dict) else v)

    rho["Topo (projection)"] = {f: _r(tf[FOLDS[f]]["Topo (projection)"]) for f in folds}
    rho["Topo-Mult"] = {f: _r(qa[FOLDS[f]]["Topo-Mult"]) for f in folds}

    # Gate G1: the registered own-learner rows of Amendments 5 and 6 reproduce.
    gate = {}
    for art, hyb, base in (("loso_significance_hybrid_cpu.json", "Hybrid-HGT", "HGT-QoS"),
                           ("loso_significance_hybrid_gat_cpu.json", "Hybrid-GAT", "GAT-QoS")):
        sig = json.loads((RESULTS / art).read_text())
        rows = sig.get("hybrid_gat") or sig["hybrid"]
        row = next(r for r in rows if r["baseline"] == PER_SEED[base][1])
        mine = _contrast(rho[hyb], rho[base])
        gate[f"{hyb} vs {base}"] = {"artifact_delta": row["mean_delta"], "delta": mine["delta"],
                                    "artifact_p": row["p"], "p": mine["p"],
                                    "passed": abs(row["mean_delta"] - mine["delta"]) < 1e-9}
    if not all(g["passed"] for g in gate.values()):
        print("G1 FAILED", gate)
        return 1

    registered = {f"{h} vs {b}": _contrast(rho[h], rho[b])
                  for h, b in (("Hybrid-HGT", "HGT-QoS"), ("Hybrid-GAT", "GAT-QoS"),
                               ("Hybrid-HGT", "Topo-QoS"), ("Hybrid-GAT", "Topo-QoS"))}
    f7 = {f"{h} vs {b}": _contrast(rho[h], rho[b])
          for h in ("Hybrid-HGT", "Hybrid-GAT") for b in ("Topo (projection)", "Topo-Mult")}
    for k, p in holm({k: c["p"] for k, c in f7.items()}).items():
        f7[k]["p_holm"] = p

    n = n_apps()
    w = [n[f] for f in folds]
    ratios = _ratios()
    sensitivity = {}
    for k in list(registered) + list(f7):
        h, b = k.split(" vs ")
        d = [rho[h][f] - rho[b][f] for f in folds]
        sensitivity[k] = {"weighted": weighted_signflip(d, w),
                          "nadeau_bengio": {u: nb_corrected_t(d, r) for u, r in ratios.items()}}
    means = {k: {"arithmetic": _mean(list(v.values())),
                 "size_weighted": float(sum(n[f] * v[f] for f in folds) / sum(w))}
             for k, v in rho.items()}
    for k, c in {**registered, **f7}.items():
        print(f"{k:34s} d={c['delta']:+.3f} p={c['p']:.4f} holm={c.get('p_holm', '-')} "
              f"w={sensitivity[k]['weighted']['delta_weighted']:+.3f} "
              f"p_w={sensitivity[k]['weighted']['p_signflip']:.4f}")
    _write("referee_round8_hybrid.json",
           {"gate_G1": gate, "registered_amendments_5_6": registered, "F7": f7,
            "sensitivity": sensitivity, "means": means, "per_fold": rho},
           experiment="F7 hybrid attribution")
    return 0


# ── F3 on the published GAT-P-QoS, and Nadeau-Bengio for the registered contrasts ──

def cmd_tost(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    ioe = json.loads((RESULTS / "independent_oracle_evaluation.json").read_text())["per_fold"]
    indeg = {f: float(ioe[f]["i_star"]["InDeg"]["rho"]) for f in folds}
    gatp = seed_mean(per_seed_rho(*PER_SEED["GAT-P-QoS"]))
    graphs = {f: build_graph_from_json(_topology(f)) for f in folds}
    istar = load_oracles(folds)["i_star"]
    preds = _learned_preds(LEARNED["GAT-P-QoS"][0])
    ens = {f: score(preds[f], istar[f], graphs[f])["rho"] for f in folds}
    ratios = _ratios()
    out: Dict[str, Any] = {}
    for label, series in (("per_seed_mean", gatp), ("seed_ensemble", ens)):
        d = [series[f] - indeg[f] for f in folds]
        out[label] = {"mean_delta": float(np.mean(d)), "won": int(np.sum(np.asarray(d) > 0)),
                      "tost_t": tost_t(d, MARGIN), "tost_wilcoxon": tost_wilcoxon(d, MARGIN),
                      "tost_nadeau_bengio": tost_t(d, MARGIN, 1 / len(d) + ratios["scenario"]),
                      "per_fold_delta": dict(zip(folds, d))}
        print(f"GAT-P-QoS ({label}) - InDeg: {out[label]['mean_delta']:+.4f} "
              f"TOST p={out[label]['tost_t']['p']:.3f} bound={out[label]['tost_t']['equivalence_bound']:.3f}")
    rho = {k: seed_mean(per_seed_rho(*PER_SEED[k]))
           for k in ("HGT-QoS", "HGT", "Hybrid-HGT", "Hybrid-GAT", "Topo-QoS")}
    # The plan's two contrasts come from the v5 artifact's per-fold deltas.
    v5 = json.loads((RESULTS / "loso_significance_v5.json").read_text())["preregistered"]
    nb = {}
    for row in v5:
        d = list(row["per_fold_delta"].values())
        nb[f"{row['label']} vs {row['baseline_label']} (plan, v5)"] = {
            "mean_delta": float(np.mean(d)),
            **{u: nb_corrected_t(d, r) for u, r in ratios.items()}}
    for h in ("HGT-QoS", "Hybrid-HGT", "Hybrid-GAT"):
        d = [rho[h][f] - rho["Topo-QoS"][f] for f in folds]
        nb[f"{h} vs Topo-QoS (CPU)"] = {"mean_delta": float(np.mean(d)),
                                        **{u: nb_corrected_t(d, r) for u, r in ratios.items()}}
    d = [gatp[f] - indeg[f] for f in folds]
    nb["GAT-P-QoS vs InDeg"] = {"mean_delta": float(np.mean(d)),
                                **{u: nb_corrected_t(d, r) for u, r in ratios.items()}}
    for k, v in nb.items():
        print(f"NB {k:40s} d={v['mean_delta']:+.3f} p(scen)={v['scenario']['p']:.4f}")
    _write("referee_round8_tost.json",
           {"margin": MARGIN, "GAT-P-QoS_vs_InDeg": out, "nadeau_bengio": nb, "ratios": ratios},
           experiment="F3 on published GAT-P-QoS; Nadeau-Bengio sensitivity", margin=MARGIN)
    return 0


# ── Table 7 on full-population I_dyn ──────────────────────────────────────────

TABLE7_TF = ("Analytic-I*", "InDeg", "Reach", "Pubs-raw", "Topo-QoS", "Degree-raw")
TABLE7_LEARNED = ("GAT-P-QoS", "Hybrid-GAT", "Hybrid-HGT", "HGT-QoS", "GAT-QoS", "HGT-P-QoS")


def _per_seed_preds(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """``{fold: {seed: {node: overall}}}`` from a sweep's per-seed logs."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for fold in sorted((ROOT / path / "workspace").glob("fold_*")):
        sid = fold.name[len("fold_"):]
        for p in sorted(fold.glob("seed_*/seed_result.json")):
            fs = json.loads(p.read_text())["metrics"]["_full_scores"]
            out.setdefault(sid, {})[p.parent.name] = {str(k): float(v["overall"]) for k, v in fs.items()}
    return out


def _partial(pred: Dict[str, float], y: Dict[str, float], z: Dict[str, float],
             apps: List[str]) -> Optional[float]:
    keep = [a for a in apps if a in y and a in z and a in pred]
    if len(keep) < 5:
        return None
    return partial_spearman([pred[a] for a in keep], [y[a] for a in keep], [z[a] for a in keep])


def cmd_table7(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    oracles = load_oracles(folds)
    oracles["i_dyn_n30"] = oracles.pop("i_dyn")
    oracles["i_dyn"] = {f: v for f, v in load_idyn_full().items() if f in FOLDS}
    names = ("i_star", "i_dyn", "i_comp", "i_dyn_n30")
    graphs = {f: build_graph_from_json(_topology(f)) for f in folds}
    rankers: Dict[str, Dict[str, Dict[str, float]]] = {}
    for f in folds:
        rk = raw_rankers(_topology(f))
        for r in TABLE7_TF:
            rankers.setdefault(r, {})[f] = rk[r]
    for eng in TABLE7_LEARNED:
        rankers[eng] = _learned_preds(LEARNED[eng][0])
    per_seed = {eng: _per_seed_preds(LEARNED[eng][0]) for eng in TABLE7_LEARNED}

    # Gate G3': the ensemble rho on I* reproduces the round-7 artifact.
    r7 = json.loads((RESULTS / "referee_round7_learned_oracles.json").read_text())
    gate = {}
    for eng in TABLE7_LEARNED:
        mine = _mean([score(rankers[eng][f], oracles["i_star"][f], graphs[f])["rho"] for f in folds])
        gate[eng] = {"round7": r7[eng]["ensemble_i_star"], "here": mine,
                     "passed": abs(mine - r7[eng]["ensemble_i_star"]) < 1e-9}
    if not all(g["passed"] for g in gate.values()):
        print("G3' FAILED", gate)
        return 1

    per: Dict[str, Dict[str, Any]] = {}
    for r, preds in rankers.items():
        per[r] = {}
        for f in folds:
            apps = _apps(graphs[f])
            row = {o: (score(preds[f], oracles[o][f], graphs[f]) if f in oracles[o] else None)
                   for o in names}
            row["partial_idyn_given_istar"] = _partial(preds[f], oracles["i_dyn"][f],
                                                       oracles["i_star"][f], apps)
            row["partial_idyn_given_analytic"] = (
                None if r == "Analytic-I*" else
                _partial(preds[f], oracles["i_dyn"][f], rankers["Analytic-I*"][f], apps))
            if r in per_seed:
                seeds = per_seed[r].get(f, {})
                row["per_seed_mean"] = {o: _mean([score(p, oracles[o][f], graphs[f])["rho"]
                                                  for p in seeds.values()])
                                        for o in ("i_star", "i_dyn", "i_comp")}
            per[r][f] = row
    summary: Dict[str, Any] = {}
    for r in per:
        s: Dict[str, Any] = {}
        for o in names:
            xs = [per[r][f][o]["rho"] for f in folds if per[r][f][o] is not None]
            s[o] = {"mean": _mean(xs), "ci95": mean_ci(xs) if len(xs) > 1 else None}
        s["i_star_active"] = _mean([per[r][f]["i_star"]["rho_active"] for f in folds])
        for key in ("partial_idyn_given_istar", "partial_idyn_given_analytic"):
            xs = [per[r][f][key] for f in folds if per[r][f][key] is not None]
            s[key] = {"mean": _mean(xs), "ci95": mean_ci(xs) if len(xs) > 1 else None,
                      "n_folds": len(xs)}
        if r in per_seed:
            s["per_seed_mean"] = {o: _mean([per[r][f]["per_seed_mean"][o] for f in folds])
                                  for o in ("i_star", "i_dyn", "i_comp")}
        summary[r] = s
        print(f"{r:12s} I*={s['i_star']['mean']:.3f} Idyn={s['i_dyn']['mean']:.3f} "
              f"Icomp={s['i_comp']['mean']:.3f} n30={s['i_dyn_n30']['mean']:.3f} "
              f"partial|I*={s['partial_idyn_given_istar']['mean']}")
    agree = [float(spearmanr([oracles['i_star'][f][a] for a in oracles['i_dyn'][f]
                               if a in oracles['i_star'][f]],
                              [oracles['i_dyn'][f][a] for a in oracles['i_dyn'][f]
                               if a in oracles['i_star'][f]]).correlation) for f in folds]
    n_lab = {f: len(oracles["i_dyn"][f]) for f in folds}
    summary["_oracle"] = {"rho_istar_idyn": {"mean": _mean(agree), "ci95": mean_ci(agree)},
                          "idyn_labelled_per_fold": n_lab,
                          "idyn_source": str(IDYN_FULL.relative_to(ROOT)),
                          "idyn_n30_source": str(IDYN_N30.relative_to(ROOT))}
    print("I*~I_dyn(full):", summary["_oracle"]["rho_istar_idyn"])
    _write("referee_round8_table7.json", {"gate_G3": gate, "summary": summary, "per_fold": per},
           experiment="Table 7 on full-population I_dyn (Amendment 11 rule R)")
    return 0


# ── hierarchical fold -> seed bootstrap ───────────────────────────────────────

def cmd_hierarchical(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    rng = np.random.default_rng(0)
    topo = seed_mean(per_seed_rho(*PER_SEED["Topo-QoS"]))
    out: Dict[str, Any] = {}
    for lab in PUBLISHED_I_STAR:
        ps = per_seed_rho(*PER_SEED[lab])
        arr = [np.asarray(ps[f]) for f in folds]
        point = float(np.mean([a.mean() for a in arr]))
        if abs(point - PUBLISHED_I_STAR[lab]) > 5e-4:
            print(f"gate failed for {lab}: {point} vs {PUBLISHED_I_STAR[lab]}")
            return 1
        boots, dboots = [], []
        for _ in range(B_HIER):
            fi = rng.integers(0, len(folds), len(folds))
            vals = [rng.choice(arr[i], size=len(arr[i]), replace=True).mean() for i in fi]
            boots.append(float(np.mean(vals)))
            dboots.append(float(np.mean([v - topo[folds[i]] for v, i in zip(vals, fi)])))
        out[lab] = {"point": point,
                    "ci95_hier": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    "ci95_fold_only": mean_ci([a.mean() for a in arr]),
                    "delta_vs_topo_qos_ci95_hier": [float(np.percentile(dboots, 2.5)),
                                                    float(np.percentile(dboots, 97.5))],
                    "mean_within_fold_seed_sd": float(np.mean([a.std(ddof=1) for a in arr]))}
        print(f"{lab:12s} {point:.3f} hier {out[lab]['ci95_hier']} fold {out[lab]['ci95_fold_only']}")
    _write("referee_round8_hierarchical.json", {"rows": out, "B": B_HIER},
           experiment="hierarchical bootstrap (descriptive)")
    return 0


# ── Amendment 14 arms ─────────────────────────────────────────────────────────

A14_ARTIFACT = "loso_amendment14_cpu.json"
A14_OUTPUT = "output/loso_cpu_amendment14"
A14_LABELS = {
    "gl_proj_qos16_cap": "GAT-P-QoS", "gl_proj_qos16_cap_nodeg": "GAT-P-QoS-deg",
    "gl_proj_qos16_cap_nodeg_strict": "GAT-P-QoS-deg*", "gl_full_qos16_cap": "GAT-QoS",
    "gl_full_qos16_cap_nodeg": "GAT-QoS-deg", "gin_proj_qos16": "GIN-P-QoS",
    "gin_proj_qos16_nodeg": "GIN-P-QoS-deg", "gin_proj_qos16_nodeg_strict": "GIN-P-QoS-deg*",
    "gl_full_cap": "GAT", "hgl": "HGT", "hgl_qos": "HGT-QoS", "gl_full_cap_win": "GAT+w_in",
    "hgl_win": "HGT+w_in", "gl_proj_qos16_cap_idyn": "GAT-P-QoS-dyn",
    "gl_proj_qos16_cap_istar_app": "GAT-P-QoS[I*-App]", "topo_qos": "Topo-QoS",
    "topo_baseline": "Topo",
}
#: Gate G0: comparators re-run in the Amendment 14 invocation vs their published artifacts.
G0 = {"gl_proj_qos16_cap": "loso_dependency_graph_cpu.json", "hgl_qos": "loso_hybrid_cpu.json",
      "gl_full_qos16_cap": "loso_hybrid_gat_cpu.json", "gl_full_cap": "loso_rq2_matched.json",
      "hgl": "loso_rq2_matched.json", "topo_qos": "loso_hybrid_cpu.json"}
G0_TOL = 1e-6
#: Recorded, not gated: the sweep ran from a git worktree without output/loso_cache,
#: and main_table._find_cache_dir resolves that relative path, so the training-free
#: topo_qos row fell back to data/scenarios (a different substrate). No Amendment 14
#: contrast reads it; the published Topo-QoS is used wherever one is needed.
G0_RECORD_ONLY = {"topo_qos"}


def _family(pairs, rho) -> Dict[str, Any]:
    fam = {f"{A14_LABELS[a]} vs {A14_LABELS[b]}": _contrast(rho[a], rho[b]) for a, b in pairs}
    for k, p in holm({k: c["p"] for k, c in fam.items()}).items():
        fam[k]["p_holm"] = p
    return fam


def cmd_amendment14(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    ps = {v: per_seed_rho(A14_ARTIFACT, v) for v in A14_LABELS}
    gate = {}
    for v, art in G0.items():
        pub = per_seed_rho(art, v)
        diff = max(abs(a - b) for f in folds for a, b in zip(ps[v][f], pub[f]))
        gate[v] = {"max_abs_diff": diff, "passed": diff < G0_TOL or v in G0_RECORD_ONLY,
                   "record_only": v in G0_RECORD_ONLY}
    print("G0:", {v: round(g["max_abs_diff"], 9) for v, g in gate.items()})
    rho = {v: seed_mean(p) for v, p in ps.items()}
    ioe = json.loads((RESULTS / "independent_oracle_evaluation.json").read_text())["per_fold"]
    indeg = {f: float(ioe[f]["i_star"]["InDeg"]["rho"]) for f in folds}
    fam = {
        "F1": _family([("gl_proj_qos16_cap_nodeg", "gl_proj_qos16_cap"),
                       ("gl_proj_qos16_cap_nodeg_strict", "gl_proj_qos16_cap"),
                       ("gl_full_qos16_cap_nodeg", "gl_full_qos16_cap")], rho),
        "F2": _family([("gin_proj_qos16", "gl_proj_qos16_cap"),
                       ("gin_proj_qos16_nodeg", "gl_proj_qos16_cap_nodeg"),
                       ("gin_proj_qos16_nodeg_strict", "gl_proj_qos16_cap_nodeg_strict")], rho),
    }
    # F3: distance to the InDeg reference (equivalence only).
    graphs = {f: build_graph_from_json(_topology(f)) for f in folds}
    oracles = load_oracles(folds)
    oracles["i_dyn"] = {f: v for f, v in load_idyn_full().items() if f in FOLDS}
    f3 = {}
    for v in ("gl_proj_qos16_cap", "gl_proj_qos16_cap_nodeg", "gl_proj_qos16_cap_nodeg_strict",
              "gin_proj_qos16", "gin_proj_qos16_nodeg", "gin_proj_qos16_nodeg_strict"):
        d = [rho[v][f] - indeg[f] for f in folds]
        f3[A14_LABELS[v]] = {"mean": float(np.mean(d)), "won": int(np.sum(np.asarray(d) > 0)),
                             "tost_t": tost_t(d, MARGIN), "tost_wilcoxon": tost_wilcoxon(d, MARGIN)}
    # F4: the w_in-held 2x2, as loso_significance.py computed it.
    sig = json.loads((RESULTS / "loso_significance_amendment14_cpu.json").read_text())
    f4 = sig.get("factorial", [])
    # F5: the surrogate on I_dyn, and every arm on every oracle from per-seed predictions.
    cells: Dict[str, Any] = {}
    for v in A14_LABELS:
        if v.startswith("topo"):
            continue
        preds = _per_seed_preds(f"{A14_OUTPUT}/{v}")
        cells[v] = {o: {f: _mean([score(p, oracles[o][f], graphs[f])["rho"]
                                  for p in preds.get(f, {}).values()]) for f in folds}
                    for o in ("i_star", "i_dyn", "i_comp")}
    ana = {f: raw_rankers(_topology(f))["Analytic-I*"] for f in folds}
    ana_dyn = {f: score(ana[f], oracles["i_dyn"][f], graphs[f])["rho"] for f in folds}
    ltr = json.loads((DATA_BENCHMARKS / "oracle_robust_ltr.json").read_text())
    gbm_dyn = _gbm_idyn(ltr)
    y = cells["gl_proj_qos16_cap_idyn"]["i_dyn"]
    f5 = {"GAT-P-QoS-dyn vs Analytic-I*": _contrast(y, ana_dyn),
          "GAT-P-QoS-dyn vs GBM-Dep-QoS-dyn": _contrast(y, gbm_dyn) if gbm_dyn else None,
          "GAT-P-QoS-dyn vs GAT-P-QoS[I*-App]": _contrast(
              y, cells["gl_proj_qos16_cap_istar_app"]["i_dyn"])}
    live = {k: c for k, c in f5.items() if c is not None}
    for k, p in holm({k: c["p"] for k, c in live.items()}).items():
        f5[k]["p_holm"] = p
    summary = {A14_LABELS[v]: {"loso_i_star": _mean(list(rho[v].values())),
                               **({o: _mean(list(cells[v][o].values())) for o in cells[v]}
                                  if v in cells else {})}
               for v in A14_LABELS}
    zs = {}
    for v in A14_LABELS:
        p = RESULTS / f"realworld_zeroshot_{v}_amendment14.json"
        if p.exists():
            zs[A14_LABELS[v]] = json.loads(p.read_text()).get("mean_rho_across_systems")
    for k, v in summary.items():
        print(f"{k:20s} " + " ".join(f"{o}={x:.3f}" for o, x in v.items() if x is not None))
    for name, fm in fam.items():
        for k, c in fm.items():
            print(f"{name} {k:40s} d={c['delta']:+.3f} p_holm={c['p_holm']:.4f}")
    for row in f4:
        print(f"F4 {row.get('quantity', row.get('label'))}: d={row.get('mean_delta', 0):+.3f} "
              f"p={row.get('p')} p_holm={row.get('p_holm')}")
    for k, c in f5.items():
        if c:
            print(f"F5 {k:40s} d={c['delta']:+.3f} p_holm={c['p_holm']:.4f}")
    for k, c in f3.items():
        print(f"F3 {k:18s} d={c['mean']:+.3f} TOST p={c['tost_t']['p']:.3f} "
              f"bound={c['tost_t']['equivalence_bound']:.3f}")
    _write("referee_round8_amendment14.json",
           {"gate_G0": gate, **fam, "F3": f3, "F4": f4, "F5": f5, "summary": summary,
            "zero_shot": zs, "per_fold": {"loso_i_star": rho, "cells": cells,
                                          "analytic_i_dyn": ana_dyn, "gbm_dep_qos_dyn": gbm_dyn}},
           experiment="Amendment 14 arms", margin=MARGIN)
    return 0 if all(g["passed"] for g in gate.values()) else 1


def _gbm_idyn(ltr: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Amendment 11's GBM-Dep-QoS->dyn per-fold rho on I_dyn-full, if recorded."""
    try:
        per = ltr["loso"]["per_fold"]
        return {f: float(per[f]["arms"]["gbm_dep_qos_dyn"]["i_dyn"]["rho"]) for f in FOLDS}
    except (KeyError, TypeError):
        return None


def cmd_degree_leak(_: argparse.Namespace) -> int:
    """Spearman rho of every Application feature column with InDeg, per fold."""
    from cli.loso_evaluate import discover_scenarios
    from reproduce.training_free_suite import _flow, indeg
    from saag.prediction.data_preparation import KEYS_BY_TYPE, networkx_to_hetero_data

    bundles = {b.scenario_id: b for b in discover_scenarios(Path("output/loso_cache"), [])}
    keys = KEYS_BY_TYPE["Application"]
    per: Dict[str, Dict[str, Optional[float]]] = {}
    for f in FOLDS:
        b = bundles[f]
        conv = networkx_to_hetero_data(b.graph, b.structural, b.simulation, b.rm)
        x = conv.hetero_data["Application"].x.numpy()
        ind = indeg(_flow(_topology(f)))
        y = [ind.get(a, 0.0) for a in conv.node_id_map["Application"]]
        per[f] = {k: (None if np.std(x[:, c]) < 1e-12 else float(spearmanr(x[:, c], y).correlation))
                  for c, k in enumerate(keys)}
    summary = {k: _mean([per[f][k] for f in FOLDS if per[f][k] is not None]) for k in keys}
    for k, v in sorted(summary.items(), key=lambda kv: -(kv[1] or 0)):
        print(f"{k:28s} {v}")
    _write("referee_round8_degree_leak.json", {"summary": summary, "per_fold": per},
           experiment="degree leak (descriptive)")
    return 0



# ── F6: the registered selection rule (arm N) ────────────────────────────────

NESTED_DIR = RESULTS / "nested_a14_shards"
FIXED_CONFIG = {"layers": 3, "rank_normalize_features": False, "rank_normalize_labels": False}
NESTED = {"hgl_qos": ("HGT-QoS", "loso_hybrid_cpu.json"),
          "gl_proj_qos16_cap": ("GAT-P-QoS", "loso_dependency_graph_cpu.json")}


def cmd_nested(_: argparse.Namespace) -> int:
    """Merge the per-outer-fold shards of arm N and run family F6."""
    folds = list(FOLDS)
    per: Dict[str, Dict[str, Any]] = {}
    gate: Dict[str, Any] = {}
    for v, (lab, art) in NESTED.items():
        fixed = seed_mean(per_seed_rho(art, v))
        recs = {}
        for f in folds:
            rep = json.loads((NESTED_DIR / f"{v}__{f}.json").read_text())
            prov = rep["provenance"]
            assert not prov["dirty"] and prov["config"]["grid"] == "stage1", (v, f)
            rec = rep["folds"][0]
            assert rec["holdout"] == f
            recs[f] = rec
        # Gate: where the inner search picked the published configuration, the outer
        # score must equal the published per-fold value.
        same = {f: abs(recs[f]["outer_rho"] - fixed[f]) for f in folds
                if recs[f]["selected_config"] == FIXED_CONFIG}
        gate[lab] = {"folds_selecting_fixed": sorted(same), "max_abs_diff": max(same.values(), default=None),
                     "passed": all(d < 1e-6 for d in same.values())}
        per[lab] = {"nested": {f: float(recs[f]["outer_rho"]) for f in folds}, "fixed": fixed,
                    "selected": {f: recs[f]["selected_config"] for f in folds},
                    "commits": sorted({json.loads((NESTED_DIR / f"{v}__{f}.json").read_text())
                                       ["provenance"]["commit"][:8] for f in folds})}
    topo = seed_mean(per_seed_rho(*PER_SEED["Topo-QoS"]))
    fam = {"nested HGT-QoS vs fixed HGT-QoS": _contrast(per["HGT-QoS"]["nested"], per["HGT-QoS"]["fixed"]),
           "nested GAT-P-QoS vs fixed GAT-P-QoS": _contrast(per["GAT-P-QoS"]["nested"], per["GAT-P-QoS"]["fixed"]),
           "nested HGT-QoS vs Topo-QoS": _contrast(per["HGT-QoS"]["nested"], topo)}
    for k, q in holm({k: c["p"] for k, c in fam.items()}).items():
        fam[k]["p_holm"] = q
    ratios = _ratios()
    d = [per["HGT-QoS"]["nested"][f] - topo[f] for f in folds]
    fam["nested HGT-QoS vs Topo-QoS"]["nadeau_bengio"] = nb_corrected_t(d, ratios["scenario"])
    summary = {lab: {"nested": _mean(list(x["nested"].values())), "fixed": _mean(list(x["fixed"].values())),
                     "n_folds_selecting_fixed": len(gate[lab]["folds_selecting_fixed"])}
               for lab, x in per.items()}
    rule = "F6a" if (fam["nested HGT-QoS vs Topo-QoS"]["p_holm"] < 0.05
                     and fam["nested HGT-QoS vs Topo-QoS"]["delta"] > 0) else "F6b"
    print("gate:", gate)
    print("summary:", summary)
    for k, c in fam.items():
        print(f"F6 {k:40s} d={c['delta']:+.3f} won={c['won']} p={c['p']:.4f} holm={c['p_holm']:.4f}")
    print("decision:", rule)
    _write("referee_round8_nested.json", {"gate": gate, "F6": fam, "summary": summary,
                                          "decision": rule, "per_fold": per},
           experiment="F6 selection rule (arm N)")
    return 0 if all(g["passed"] for g in gate.values()) else 1

# ── cost ──────────────────────────────────────────────────────────────────────

def _median_time(fn, repeats: int) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def cmd_cost(args: argparse.Namespace) -> int:
    """Like-for-like wall-clock: every region timed on the same graph in one session."""
    import platform

    from reproduce.training_free_suite import _flow, indeg
    from saag.analysis.structural_analyzer import record_phases

    graphs: Dict[str, Dict[str, Any]] = {f: _topology(f) for f in FOLDS}
    for size in args.sizes:
        graphs[f"generated_{size}"] = _generated(size)
    rows: Dict[str, Any] = {}
    for name, topo in graphs.items():
        big = name.startswith("generated_") and int(name.split("_")[1]) >= 5000
        reps = 1 if big else args.repeats
        with record_phases() as sink:
            t_app = _median_time(lambda: _analysis(topo, "app"), reps)
            phases = {k: v / reps for k, v in sink.items()}
        t_gate = _median_time(lambda: _gate(topo), reps)
        t_one = _median_time(lambda: _istar(topo, [42], ["Application"]), reps)
        t_five = _median_time(lambda: _istar(topo, SEEDS5, ["Application"]), reps)
        t_sweep = (None if big else
                   _median_time(lambda: _istar(topo, SEEDS5, ["Application", "Broker", "Library"]), reps))
        t_count = _median_time(lambda: indeg(_flow(topo)), max(reps, 3))
        n_v = sum(len(topo.get(k, [])) for k in ("applications", "brokers", "topics", "nodes", "libraries"))
        rows[name] = {"n_components": n_v, "repeats": reps, "analyze_app_s": t_app,
                      "analyze_app_phases_s": phases, "gate_system_s": t_gate,
                      "istar_one_pass_app_s": t_one, "istar_five_seed_app_s": t_five,
                      "istar_sweep_s": t_sweep, "count_s": t_count,
                      "ratio_app_analysis_to_one_pass": t_app / t_one,
                      "ratio_gate_to_sweep": (t_gate / t_sweep) if t_sweep else None,
                      "ratio_one_pass_to_count": t_one / t_count}
        print(f"{name:28s} |V|={n_v:5d} app={t_app:8.2f}s gate={t_gate:8.2f}s 1pass={t_one:8.2f}s "
              f"5app={t_five:8.2f}s sweep={t_sweep if t_sweep is None else round(t_sweep, 2)}s "
              f"count={t_count * 1000:7.1f}ms", flush=True)
    corpus = [rows[f] for f in FOLDS]
    keys = ("analyze_app_s", "gate_system_s", "istar_one_pass_app_s", "istar_sweep_s", "count_s",
            "ratio_app_analysis_to_one_pass", "ratio_gate_to_sweep", "ratio_one_pass_to_count")
    summary = {k: {"min": float(min(r[k] for r in corpus)),
                   "median": float(np.median([r[k] for r in corpus])),
                   "max": float(max(r[k] for r in corpus))} for k in keys}
    _write("referee_round8_cost.json",
           {"rows": rows, "corpus_summary": summary,
            "definitions": {
                "analyze_app_s": "AnalysisService.analyze_layer('app'), the stage behind the learners' node features",
                "gate_system_s": "detection gate: analyze_layer('system') + predict_quality + 18 detectors (DEEP_PIPELINE excluded)",
                "istar_one_pass_app_s": "FaultInjector, seed 42, Applications only: one I* labelling pass",
                "istar_five_seed_app_s": "FaultInjector, five seeds, Applications only",
                "istar_sweep_s": "FaultInjector, five seeds, Application+Broker+Library (the published sweep)",
                "count_s": "Application-Library DEPENDS_ON projection + InDeg"},
            "machine": {"processor": platform.processor(), "python": platform.python_version()}},
           experiment="cost reconciliation", repeats=args.repeats, sizes=args.sizes)
    return 0


SEEDS5 = [42, 123, 456, 789, 2024]


def _generated(size: int) -> Dict[str, Any]:
    from reproduce.inference_latency import _counts_for
    from tools.generation.models import GraphConfig
    from tools.generation.service import generate_graph

    cfg = GraphConfig.from_yaml({"graph": {"seed": 42, "counts": _counts_for(size)}})
    return generate_graph(config=cfg, seed=42)


def _analysis(topo: Dict[str, Any], layer: str):
    from saag.analysis.service import AnalysisService
    from saag.infrastructure.memory_repo import MemoryRepository

    repo = MemoryRepository()
    repo.save_graph(topo, clear=True)
    return AnalysisService(repo).analyze_layer(layer)


def _gate(topo: Dict[str, Any]) -> None:
    """reproduce/detection_validation.py's gate on an in-memory topology."""
    from reproduce.detection_validation import DEFAULT_EXCLUDED_PATTERNS
    from saag.analysis.antipattern_detector import CATALOG, AntiPatternDetector
    from saag.prediction.service import PredictionService

    quality = PredictionService().predict_quality(_analysis(topo, "system").structural)
    for pid in CATALOG:
        if pid not in DEFAULT_EXCLUDED_PATTERNS:
            AntiPatternDetector(active_patterns=[pid]).detect(quality, layer="system")


def _istar(topo: Dict[str, Any], seeds: List[int], types: List[str]) -> None:
    """reproduce/oracle_timing.py's FaultInjector settings, graph built in the timed region."""
    import tempfile

    from reproduce.oracle_timing import CASCADE_DEPTH, PROPAGATION_THRESHOLD, QOS_FACTOR
    from saag.core.graph_io import load_graph
    from saag.simulation.fault_injector import FaultInjector

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(topo, fh)
    graph = load_graph(Path(fh.name))
    Path(fh.name).unlink()
    FaultInjector(graph=graph, seeds=seeds,
                  cascade_depth_limit=CASCADE_DEPTH, propagation_threshold=PROPAGATION_THRESHOLD,
                  qos_factor_mode=QOS_FACTOR).run(node_types=types, node_ids=None)


def main() -> int:
    stages = {"hybrid": cmd_hybrid, "tost": cmd_tost, "table7": cmd_table7,
              "hierarchical": cmd_hierarchical, "amendment14": cmd_amendment14,
              "degree_leak": cmd_degree_leak, "nested": cmd_nested, "cost": cmd_cost}
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stages", nargs="+", choices=list(stages))
    ap.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000, 2000, 5000])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    return max(stages[s](args) for s in args.stages)


if __name__ == "__main__":
    sys.exit(main())
