#!/usr/bin/env python3
"""
reproduce/referee_round14.py — round-14 referee analyses (PREREGISTRATION.md Amendment 19)
=========================================================================================
Reads the Amendment 19 sweeps and writes ``data/benchmarks/referee_round14_*.json``:

  amendment19  gate G0 (comparators re-run in the same invocation against their
               published artifacts); F14 (sum aggregation with reverse edges on the raw
               multigraph); F15 (rate-fed learned approximations of I_dyn against Eq. 7);
               F16 (tie-aware listwise loss) and the tie-permutation spread rule; the
               small-capacity arm; every arm on I*, I_dyn and I_comp from per-seed
               predictions; zero-shot for the GIN-QoS-R arms
  lc           the learning curve over K training scenarios (K = 11 from the main sweep)
  descriptive  the nested-protocol GAT-P-QoS against afferent coupling (InDeg); a
               mixed-effects sensitivity (seed nested in fold); per-fold I_dyn headroom

``amendment19`` must run before ``lc`` (K = 11 is read from its artifact).

Usage:
    PYTHONPATH=. python reproduce/referee_round14.py amendment19 lc descriptive
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import stamp  # noqa: E402
from reproduce.idyn_rate_expansion import closed_forms  # noqa: E402
from reproduce.oracle_robust_ltr import _topology, app_ids  # noqa: E402
from reproduce.referee_round7 import load_oracles  # noqa: E402
from reproduce.referee_round8 import (  # noqa: E402
    _partial,
    _per_seed_preds,
    load_idyn_full,
    per_seed_rho,
    seed_mean,
)
from reproduce.training_free_suite import FOLDS, _mean, holm, mean_ci, paired, score  # noqa: E402
from saag.core.graph_io import build_graph_from_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_BENCHMARKS = ROOT / "data" / "benchmarks"
RESULTS = ROOT / "results"

A19_ARTIFACT = "loso_amendment19_cpu.json"
A19_OUTPUT = "output/loso_cpu_amendment19"
A19_JSON = "referee_round14_amendment19.json"
LABELS = {
    # comparators
    "gl_proj_qos16_cap": "GAT-P-QoS", "gl_full_qos16_cap": "GAT-QoS",
    "gl_full_qos16_cap_rev": "GAT-QoS-R", "gin_proj_qos16": "GIN-P-QoS",
    "gl_proj_qos16_cap_idyn": "GAT-P-QoS-dyn", "gl_proj_qos16_cap_min": "GAT-P-QoS-min",
    "gl_full_qos16_cap_rev_min": "GAT-QoS-R-min", "gin_proj_qos16_min": "GIN-P-QoS-min",
    "gin_proj_qos16_const": "GIN-P-QoS-const",
    # Amendment 19 arms
    "gin_full_qos16_rev": "GIN-QoS-R", "gin_full_qos16_rev_min": "GIN-QoS-R-min",
    "gin_full_qos16_rev_const": "GIN-QoS-R-const",
    "gl_proj_qos16_cap_idyn_r": "GAT-P-QoS-dyn+rate",
    "gl_proj_qos16_cap_idyn_re": "GAT-P-QoS-dyn+rate-e",
    "gin_proj_qos16_idyn_re": "GIN-P-QoS-dyn+rate-e",
    "gl_proj_qos16_cap_tie": "GAT-P-QoS-tie", "gl_full_qos16_cap_rev_tie": "GAT-QoS-R-tie",
    "gl_proj_qos16_cap_tie_perm": "GAT-P-QoS-tie-perm17",
    "gl_proj_qos16_cap_tie_perm18": "GAT-P-QoS-tie-perm18",
    "gl_proj_qos16_cap_tie_perm19": "GAT-P-QoS-tie-perm19",
    "gl_proj_qos16_s": "GAT-S-P-QoS",
}
#: Gate G0: comparators re-run in this invocation vs their published artifacts.
G0 = {"gl_proj_qos16_cap": "loso_dependency_graph_cpu.json",
      "gl_full_qos16_cap": "loso_hybrid_gat_cpu.json",
      "gl_full_qos16_cap_rev": "loso_amendment16_cpu.json",
      "gin_proj_qos16": "loso_amendment14_cpu.json",
      "gl_proj_qos16_cap_idyn": "loso_amendment14_cpu.json",
      "gl_proj_qos16_cap_min": "loso_amendment17_cpu.json",
      "gl_full_qos16_cap_rev_min": "loso_amendment17_cpu.json",
      "gin_proj_qos16_min": "loso_amendment17_cpu.json",
      "gin_proj_qos16_const": "loso_amendment17_cpu.json"}
G0_TOL = 1e-6
ZERO_SHOT = ("gin_full_qos16_rev", "gin_full_qos16_rev_min", "gin_full_qos16_rev_const")
RATE_ARMS = ("GAT-P-QoS-dyn+rate", "GAT-P-QoS-dyn+rate-e", "GIN-P-QoS-dyn+rate-e")
TIE_PERMS = ("GAT-P-QoS-tie-perm17", "GAT-P-QoS-tie-perm18", "GAT-P-QoS-tie-perm19")
#: Amendment 17b's mean per-fold spread across three ListMLE permutations.
LISTMLE_SPREAD = 0.044
LC_LEARNERS = {"gl_proj_qos16_cap": "GAT-P-QoS", "gin_proj_qos16": "GIN-P-QoS",
               "gl_full_qos16_cap": "GAT-QoS"}
LC_K = (1, 2, 4, 8)
LC_DRAWS = (1, 2, 3)

_PROV = stamp(script="reproduce/referee_round14.py", amendment=19)


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


def _pair(rho, a: str, b: str) -> Dict[str, Any]:
    return paired([rho[a][f] for f in FOLDS], [rho[b][f] for f in FOLDS])


def _sig_pos(c: Dict[str, Any]) -> bool:
    return c["p_holm"] < 0.05 and c["delta"] > 0


def _indeg_rho() -> Dict[str, float]:
    ioe = json.loads((RESULTS / "independent_oracle_evaluation.json").read_text())["per_fold"]
    return {f: float(ioe[f]["i_star"]["InDeg"]["rho"]) for f in FOLDS}


def _print_family(name: str, fam: Dict[str, Any]) -> None:
    for k, c in fam.items():
        ph = c.get("p_holm")
        print(f"{name:5s} {k:46s} d={c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
              f"won={c['won']}/12 p={c['p']:.4f}" + (f" p_holm={ph:.4f}" if ph is not None else ""))


# ── Amendment 19: gate, families, cells ──────────────────────────────────────

def cmd_amendment19(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    ps = {v: per_seed_rho(A19_ARTIFACT, v) for v in LABELS}
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
        preds = _per_seed_preds(f"{A19_OUTPUT}/{v}")
        cells[lab] = {}
        for o in ("i_star", "i_dyn", "i_comp"):
            per = {f: [score(p, oracles[o][f], graphs[f]) for p in preds.get(f, {}).values()]
                   for f in folds}
            cells[lab][o] = {f: _mean([s["rho"] for s in per[f]]) for f in folds}
            if o == "i_star":
                cells[lab]["i_star_active"] = {
                    f: _mean([s["rho_active"] for s in per[f] if s["rho_active"] is not None])
                    for f in folds}

    # F14: does the derived graph still win under matched sum aggregation?
    f14 = _family([("GIN-P-QoS-min", "GIN-QoS-R-min"), ("GIN-P-QoS", "GIN-QoS-R"),
                   ("GIN-P-QoS-const", "GIN-QoS-R-const")], rho)
    indeg_rho = _indeg_rho()
    m = {lab: _mean(list(r.values())) for lab, r in rho.items()}
    gap = m["GAT-P-QoS-min"] - m["GAT-QoS-R-min"]
    share_closed = (m["GIN-QoS-R-min"] - m["GAT-QoS-R-min"]) / gap if gap else None
    f14_desc = {
        "GIN-QoS-R vs GAT-QoS-R": _pair(rho, "GIN-QoS-R", "GAT-QoS-R"),
        "GIN-QoS-R-min vs GAT-QoS-R-min": _pair(rho, "GIN-QoS-R-min", "GAT-QoS-R-min"),
        "GAT-P-QoS vs GIN-QoS-R": _pair(rho, "GAT-P-QoS", "GIN-QoS-R"),
        "GAT-P-QoS-min vs GIN-QoS-R-min": _pair(rho, "GAT-P-QoS-min", "GIN-QoS-R-min"),
        "GIN-QoS-R-const vs InDeg": paired([rho["GIN-QoS-R-const"][f] for f in folds],
                                           [indeg_rho[f] for f in folds]),
        "share_of_min_gap_closed": share_closed,
    }
    f14a = f14["GIN-P-QoS-min vs GIN-QoS-R-min"]
    if _sig_pos(f14a):
        f14_rule = "F14a"
    elif share_closed is not None and share_closed >= 0.5:
        f14_rule = "F14b"
    else:
        f14_rule = "F14c"

    # F15: rate-fed learned approximations of I_dyn against Eq. 7.
    f12 = json.loads((DATA_BENCHMARKS / "referee_round12_f12.json").read_text())
    dyn = {"Eq7": f12["per_fold"]["Eq7"], "GBM-P-QoS-dyn": f12["per_fold"]["S+Q"],
           **{lab: cells[lab]["i_dyn"] for lab in ("GAT-P-QoS-dyn", *RATE_ARMS)}}
    f15 = _family([(lab, "Eq7") for lab in RATE_ARMS], dyn)
    f15_desc = {}
    for lab in RATE_ARMS:
        f15_desc[f"{lab} vs GAT-P-QoS-dyn"] = _pair(dyn, lab, "GAT-P-QoS-dyn")
        f15_desc[f"{lab} vs GBM-P-QoS-dyn"] = _pair(dyn, lab, "GBM-P-QoS-dyn")
    eq7 = {f: closed_forms(_topology(f))["Rate-I_dyn"] for f in folds}
    partial_eq7 = {}
    for v, lab in LABELS.items():
        if lab not in ("GAT-P-QoS-dyn", *RATE_ARMS):
            continue
        preds = _per_seed_preds(f"{A19_OUTPUT}/{v}")
        per = {}
        for f in folds:
            apps = app_ids(_topology(f))
            vals = [_partial(p, oracles["i_dyn"][f], eq7[f], apps) for p in preds.get(f, {}).values()]
            vals = [x for x in vals if x is not None]
            per[f] = _mean(vals) if vals else None
        partial_eq7[lab] = {"per_fold": per, "mean": _mean(list(per.values()))}
    dyn_m = {k: _mean(list(v.values())) for k, v in dyn.items()}
    best = max(RATE_ARMS, key=lambda lab: dyn_m[lab])
    if any(_sig_pos(c) for c in f15.values()):
        f15_rule = "F15a"
    elif (abs(dyn_m[best] - dyn_m["GBM-P-QoS-dyn"]) <= 0.02
          or (f15_desc[f"{best} vs GAT-P-QoS-dyn"]["p"] < 0.05
              and f15_desc[f"{best} vs GAT-P-QoS-dyn"]["delta"] > 0)):
        f15_rule = "F15b"
    else:
        f15_rule = "F15c"

    # F16: the tie-aware listwise loss.
    f16 = _family([("GAT-P-QoS-tie", "GAT-P-QoS"), ("GAT-P-QoS-tie", "GAT-QoS-R-tie")], rho)
    spreads = {f: float(max(rho[p][f] for p in TIE_PERMS) - min(rho[p][f] for p in TIE_PERMS))
               for f in folds}
    tie_spread = _mean(list(spreads.values()))
    spread_rule = "S1" if tie_spread <= LISTMLE_SPREAD / 2 else "S2"
    f16b = f16["GAT-P-QoS-tie vs GAT-QoS-R-tie"]
    f16_rule = "F16b-holds" if _sig_pos(f16b) else "F16b-conditional"
    f16_desc = {
        "tie_perm_mean_spread": tie_spread, "listmle_perm_mean_spread": LISTMLE_SPREAD,
        "tie_perm_means": {p: m[p] for p in TIE_PERMS},
        "GAT-P-QoS-tie vs GAT-P-QoS-tie-perm-mean": paired(
            [rho["GAT-P-QoS-tie"][f] for f in folds],
            [float(np.mean([rho[p][f] for p in TIE_PERMS])) for f in folds]),
        "per_fold_spread": spreads,
    }

    small = {"GAT-S-P-QoS vs GAT-P-QoS": _pair(rho, "GAT-S-P-QoS", "GAT-P-QoS"),
             "GAT-S-P-QoS vs InDeg": paired([rho["GAT-S-P-QoS"][f] for f in folds],
                                            [indeg_rho[f] for f in folds])}

    summary = {}
    for lab in LABELS.values():
        row = {"loso_i_star": m[lab], "ci95": mean_ci(list(rho[lab].values()))}
        for o in ("i_star", "i_dyn", "i_comp", "i_star_active"):
            row[o] = _mean(list(cells[lab][o].values()))
        summary[lab] = row
    summary["dyn_means"] = dyn_m
    zs = {}
    for v in ZERO_SHOT:
        p = RESULTS / f"realworld_zeroshot_{v}_amendment19.json"
        if p.exists():
            z = json.loads(p.read_text())
            active = [s.get("mean_rho_positive") for s in z["per_system"].values()
                      if s.get("mean_rho_positive") is not None]
            zs[LABELS[v]] = {"mean_rho": z.get("mean_rho_across_systems"),
                             "mean_rho_positive": _mean(active),
                             "per_system": {k: s.get("mean_rho") for k, s in z["per_system"].items()}}
    rules = {"F14": f14_rule, "F15": f15_rule, "F16": f16_rule, "S": spread_rule}

    for k, v in summary.items():
        if k != "dyn_means":
            print(f"{k:22s} " + " ".join(f"{o}={x:.3f}" for o, x in v.items()
                                         if isinstance(x, float)))
    print("I_dyn means:", {k: round(v, 3) for k, v in dyn_m.items()})
    for name, fm in (("F14", f14), ("F15", f15), ("F16", f16)):
        _print_family(name, fm)
    _print_family("desc", {k: c for k, c in {**f14_desc, **f15_desc, **small}.items()
                           if isinstance(c, dict)})
    print(f"share of the -min gap closed by GIN-QoS-R-min: {share_closed}")
    print(f"tie-loss permutation spread {tie_spread:.3f} (ListMLE {LISTMLE_SPREAD})")
    print("zero-shot:", {k: v["mean_rho"] for k, v in zs.items()})
    print("rules:", rules)
    _write(A19_JSON,
           {"gate_G0": gate, "F14": f14, "F15": f15, "F16": f16,
            "descriptive": {"F14": f14_desc, "F15": f15_desc, "F16": f16_desc, "small": small,
                            "partial_idyn_given_eq7": partial_eq7},
            "decision_rules": rules, "summary": summary, "zero_shot": zs,
            "per_fold": {"loso_i_star": rho, "cells": cells, "i_dyn": dyn, "indeg": indeg_rho}},
           experiment="Amendment 19 arms")
    return 0 if all(g["passed"] for g in gate.values()) else 1


# ── learning curve ───────────────────────────────────────────────────────────

def cmd_lc(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    a19 = json.loads((DATA_BENCHMARKS / A19_JSON).read_text())
    indeg_rho = _indeg_rho()
    curve: Dict[str, Dict[str, Dict[str, float]]] = {}
    for v, lab in LC_LEARNERS.items():
        curve[lab] = {"11": a19["per_fold"]["loso_i_star"][lab]}
        for k in LC_K:
            draws = [seed_mean(per_seed_rho(f"loso_amendment19_lc_K{k}_d{d}_cpu.json", v))
                     for d in LC_DRAWS]
            curve[lab][str(k)] = {f: float(np.mean([dr[f] for dr in draws])) for f in folds}
    fam = {}
    for lab, c in curve.items():
        fam[f"{lab} K11 vs K4"] = paired([c["11"][f] for f in folds], [c["4"][f] for f in folds])
    for k, p in holm({k: c["p"] for k, c in fam.items()}).items():
        fam[k]["p_holm"] = p
    desc, rules, means = {}, {}, {}
    for lab, c in curve.items():
        means[lab] = {k: {"mean": _mean(list(v.values())), "ci95": mean_ci(list(v.values()))}
                      for k, v in c.items()}
        d118 = paired([c["11"][f] for f in folds], [c["8"][f] for f in folds])
        gaps = {k: paired([v[f] for f in folds], [indeg_rho[f] for f in folds])
                for k, v in c.items()}
        slope = (means[lab]["11"]["mean"] - means[lab]["4"]["mean"]) / np.log2(11 / 4)
        within = [int(k) for k in sorted(c, key=int)
                  if means[lab]["11"]["mean"] - means[lab][k]["mean"] <= 0.02]
        desc[lab] = {"K11 vs K8": d118, "gap_to_indeg": gaps,
                     "slope_per_doubling_4_to_11": float(slope),
                     "smallest_K_within_0.02_of_K11": min(within)}
        c114 = fam[f"{lab} K11 vs K4"]
        if d118["ci95"][1] < 0.02 and not c114["p_holm"] < 0.05:
            rules[lab] = "LC-a"
        elif _sig_pos(c114) and d118["delta"] > 0.02:
            rules[lab] = "LC-b"
        else:
            rules[lab] = "LC-c"
    overall = ("LC-b" if "LC-b" in rules.values()
               else "LC-a" if all(r == "LC-a" for r in rules.values()) else "LC-c")
    for lab, mm in means.items():
        print(f"{lab:10s} " + " ".join(f"K{k}={mm[k]['mean']:.3f}" for k in ("1", "2", "4", "8", "11"))
              + f"  rule {rules[lab]}  slope/doubling {desc[lab]['slope_per_doubling_4_to_11']:+.3f}")
    _print_family("LC", fam)
    print("overall:", overall)
    _write("referee_round14_lc.json",
           {"LC": fam, "descriptive": desc, "decision_rules": {**rules, "overall": overall},
            "means": means, "indeg_mean": _mean(list(indeg_rho.values())),
            "per_fold": curve},
           experiment="Amendment 19 learning curve", K=list(LC_K) + [11], draws=list(LC_DRAWS))
    return 0


# ── descriptive: nested protocol, mixed effects, headroom ────────────────────

def _mixed(a: Dict[str, List[float]], b: Dict[str, List[float]]) -> Dict[str, Any]:
    """MixedLM of per-seed rho on arm, fold as group with a random arm slope."""
    import pandas as pd
    import statsmodels.formula.api as smf

    rows = [{"fold": f, "arm": arm, "rho": r}
            for arm, src in ((1, a), (0, b)) for f in FOLDS for r in src[f]]
    df = pd.DataFrame(rows)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = smf.mixedlm("rho ~ arm", df, groups=df["fold"], re_formula="~arm").fit(reml=True)
    return {"beta": float(fit.params["arm"]), "se": float(fit.bse["arm"]),
            "p": float(fit.pvalues["arm"]),
            "ci95": [float(x) for x in fit.conf_int().loc["arm"]],
            "converged": bool(fit.converged), "warnings": sorted({str(w.message)[:120] for w in caught})}


def cmd_descriptive(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    indeg_rho = _indeg_rho()
    nested = json.loads((DATA_BENCHMARKS / "referee_round8_nested.json").read_text())
    nested_gat = nested["per_fold"]["GAT-P-QoS"]["nested"]
    nested_vs_indeg = paired([nested_gat[f] for f in folds], [indeg_rho[f] for f in folds])

    a19 = json.loads((DATA_BENCHMARKS / A19_JSON).read_text())
    ps = {LABELS[v]: per_seed_rho(A19_ARTIFACT, v) for v in LABELS}
    indeg_seed = {f: [indeg_rho[f]] * len(ps["GAT-P-QoS"][f]) for f in folds}
    mixed = {
        "F8b GAT-P-QoS vs GAT-QoS-R": _mixed(ps["GAT-P-QoS"], ps["GAT-QoS-R"]),
        "F11a GAT-P-QoS-min vs GAT-QoS-R-min": _mixed(ps["GAT-P-QoS-min"], ps["GAT-QoS-R-min"]),
        "F14a GIN-P-QoS-min vs GIN-QoS-R-min": _mixed(ps["GIN-P-QoS-min"], ps["GIN-QoS-R-min"]),
        "F16b GAT-P-QoS-tie vs GAT-QoS-R-tie": _mixed(ps["GAT-P-QoS-tie"], ps["GAT-QoS-R-tie"]),
        "GAT-P-QoS vs InDeg": _mixed(ps["GAT-P-QoS"], indeg_seed),
    }

    rel = json.loads((DATA_BENCHMARKS / "idyn_rate_expansion.json").read_text())
    r = {f: float(rel["label_reliability"]["per_fold"][f]["seed_mean"]) for f in folds}
    dyn = a19["per_fold"]["i_dyn"]
    best_learned = max(("GBM-P-QoS-dyn", *RATE_ARMS),
                       key=lambda k: _mean(list(dyn[k].values())))
    headroom = {}
    for lab in ("Eq7", best_learned):
        h = {f: float(np.sqrt(r[f]) - dyn[lab][f]) for f in folds}
        headroom[lab] = {"per_fold": h, "mean": _mean(list(h.values())),
                         "min": min(h.values()), "max": max(h.values())}
    print(f"nested GAT-P-QoS vs InDeg: d={nested_vs_indeg['delta']:+.3f} "
          f"[{nested_vs_indeg['ci95'][0]:+.3f}, {nested_vs_indeg['ci95'][1]:+.3f}] "
          f"won={nested_vs_indeg['won']}/12 p={nested_vs_indeg['p']:.4f}")
    for k, v in mixed.items():
        print(f"mixed {k:40s} beta={v['beta']:+.3f} se={v['se']:.3f} p={v['p']:.4f} "
              f"converged={v['converged']}")
    for k, v in headroom.items():
        print(f"headroom {k}: mean {v['mean']:.3f} range [{v['min']:.3f}, {v['max']:.3f}]")
    _write("referee_round14_descriptive.json",
           {"nested_gat_p_qos_vs_indeg": nested_vs_indeg,
            "nested_gat_p_qos_mean": _mean(list(nested_gat.values())),
            "mixed_effects": mixed, "idyn_headroom": headroom,
            "label_reliability_seed_mean": r},
           experiment="Amendment 19 descriptive analyses")
    return 0


def main() -> int:
    stages = {"amendment19": cmd_amendment19, "lc": cmd_lc, "descriptive": cmd_descriptive}
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stages", nargs="+", choices=list(stages))
    args = ap.parse_args()
    return max(stages[s](args) for s in args.stages)


if __name__ == "__main__":
    sys.exit(main())
