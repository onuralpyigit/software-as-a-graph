#!/usr/bin/env python3
"""
reproduce/referee_round11.py — round-11 referee analyses (PREREGISTRATION.md Amendment 16)
=========================================================================================
Reads the Amendment 16 sweep (``results/loso_amendment16_cpu.json``, its per-seed
predictions under ``output/loso_cpu_amendment16``) and the zero-shot runs, and writes
``data/benchmarks/referee_round11_amendment16.json``:

  gate G0   the re-run comparators reproduce their published per-seed rho
  F8        direction: GAT-QoS-R vs GAT-QoS; GAT-P-QoS vs GAT-QoS-R
  F9        corrected prior: AP hybrids vs Topo-QoS-AP and vs their base learners
  F10       InDeg prior: GAT-QoS+InDeg / HGT-QoS+InDeg vs their base learners
  cells     every arm on I*, I_dyn-full and I_comp from per-seed predictions (descriptive)

Usage:
    PYTHONPATH=. python reproduce/referee_round11.py amendment16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import stamp  # noqa: E402
from reproduce.referee_round7 import _topology, load_oracles  # noqa: E402
from reproduce.referee_round8 import (  # noqa: E402
    MARGIN,
    _per_seed_preds,
    load_idyn_full,
    per_seed_rho,
    seed_mean,
    tost_t,
    tost_wilcoxon,
)
from reproduce.training_free_suite import FOLDS, _mean, holm, paired, score  # noqa: E402
from saag.core.graph_io import build_graph_from_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_BENCHMARKS = ROOT / "data" / "benchmarks"
RESULTS = ROOT / "results"

A16_ARTIFACT = "loso_amendment16_cpu.json"
A16_OUTPUT = "output/loso_cpu_amendment16"
LABELS = {
    "topo_qos": "Topo-QoS", "gl_full_qos16_cap": "GAT-QoS", "gl_proj_qos16_cap": "GAT-P-QoS",
    "hgl_qos": "HGT-QoS", "gl_qos16_prior": "Hybrid-GAT", "hgl_qos_prior": "Hybrid-HGT",
    "gl_full_qos16_cap_rev": "GAT-QoS-R", "gl_qos16_prior_ap": "Hybrid-GAT-AP",
    "hgl_qos_prior_ap": "Hybrid-HGT-AP", "gl_qos16_indeg_prior": "GAT-QoS+InDeg",
    "hgl_qos_indeg_prior": "HGT-QoS+InDeg",
}
ARMS = ("gl_full_qos16_cap_rev", "gl_qos16_prior_ap", "hgl_qos_prior_ap",
        "gl_qos16_indeg_prior", "hgl_qos_indeg_prior")
#: Gate G0: comparators re-run in this invocation vs their published artifacts.
G0 = {"gl_full_qos16_cap": "loso_hybrid_gat_cpu.json", "gl_qos16_prior": "loso_hybrid_gat_cpu.json",
      "gl_proj_qos16_cap": "loso_dependency_graph_cpu.json", "hgl_qos": "loso_hybrid_cpu.json",
      "hgl_qos_prior": "loso_hybrid_cpu.json", "topo_qos": "loso_hybrid_cpu.json"}
G0_TOL = 1e-6
ZERO_SHOT = ("gl_full_qos16_cap_rev", "gl_qos16_prior_ap", "hgl_qos_prior_ap")

_PROV = stamp(script="reproduce/referee_round11.py", amendment=16)


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


def cmd_amendment16(_: argparse.Namespace) -> int:
    folds = list(FOLDS)
    ps = {v: per_seed_rho(A16_ARTIFACT, v) for v in LABELS}
    gate = {}
    for v, art in G0.items():
        pub = per_seed_rho(art, v)
        diff = max(abs(a - b) for f in folds for a, b in zip(ps[v][f], pub[f]))
        gate[v] = {"max_abs_diff": diff, "passed": diff < G0_TOL}
    print("G0:", {v: round(g["max_abs_diff"], 9) for v, g in gate.items()})

    rho = {LABELS[v]: seed_mean(p) for v, p in ps.items()}
    tap = json.loads((DATA_BENCHMARKS / "topo_ap_sensitivity.json").read_text())["per_scenario"]
    rho["Topo-QoS-AP"] = {f: float(tap[f]["topo_qos_ap_restored"]) for f in folds}
    ioe = json.loads((RESULTS / "independent_oracle_evaluation.json").read_text())["per_fold"]
    indeg = {f: float(ioe[f]["i_star"]["InDeg"]["rho"]) for f in folds}

    fam = {
        "F8": _family([("GAT-QoS-R", "GAT-QoS"), ("GAT-P-QoS", "GAT-QoS-R")], rho),
        "F9": _family([("Hybrid-GAT-AP", "Topo-QoS-AP"), ("Hybrid-HGT-AP", "Topo-QoS-AP"),
                       ("Hybrid-GAT-AP", "GAT-QoS"), ("Hybrid-HGT-AP", "HGT-QoS")], rho),
        "F10": _family([("GAT-QoS+InDeg", "GAT-QoS"), ("HGT-QoS+InDeg", "HGT-QoS")], rho),
    }
    # Descriptive: each AP hybrid against its published defective-prior counterpart,
    # and each InDeg-prior arm's distance to the InDeg reference.
    descriptive = {
        "Hybrid-GAT-AP vs Hybrid-GAT": paired([rho["Hybrid-GAT-AP"][f] for f in folds],
                                              [rho["Hybrid-GAT"][f] for f in folds]),
        "Hybrid-HGT-AP vs Hybrid-HGT": paired([rho["Hybrid-HGT-AP"][f] for f in folds],
                                              [rho["Hybrid-HGT"][f] for f in folds]),
    }
    to_indeg = {}
    for lab in ("GAT-QoS+InDeg", "HGT-QoS+InDeg"):
        d = [rho[lab][f] - indeg[f] for f in folds]
        to_indeg[lab] = {"mean": float(np.mean(d)), "won": int(np.sum(np.asarray(d) > 0)),
                         "tost_t": tost_t(d, MARGIN), "tost_wilcoxon": tost_wilcoxon(d, MARGIN)}

    f8a, f8b = fam["F8"]["GAT-QoS-R vs GAT-QoS"], fam["F8"]["GAT-P-QoS vs GAT-QoS-R"]
    rules = {
        "F8": "F8a" if _sig_pos(f8b) else ("F8b" if _sig_pos(f8a) else "F8c"),
        "F9": "F9a" if all(_sig_pos(fam["F9"][f"{h} vs Topo-QoS-AP"])
                           for h in ("Hybrid-GAT-AP", "Hybrid-HGT-AP")) else "F9b",
        "F9c": {h: _sig_pos(fam["F9"][f"{h} vs {b}"])
                for h, b in (("Hybrid-GAT-AP", "GAT-QoS"), ("Hybrid-HGT-AP", "HGT-QoS"))},
    }

    # Every arm on every oracle from per-seed predictions (descriptive).
    graphs = {f: build_graph_from_json(_topology(f)) for f in folds}
    oracles = load_oracles(folds)
    oracles["i_dyn"] = {f: v for f, v in load_idyn_full().items() if f in FOLDS}
    cells: Dict[str, Any] = {}
    for v in LABELS:
        if v == "topo_qos":
            continue
        preds = _per_seed_preds(f"{A16_OUTPUT}/{v}")
        cells[LABELS[v]] = {}
        for o in ("i_star", "i_dyn", "i_comp"):
            per = {f: [score(p, oracles[o][f], graphs[f]) for p in preds.get(f, {}).values()]
                   for f in folds}
            cells[LABELS[v]][o] = {f: _mean([s["rho"] for s in per[f]]) for f in folds}
            if o == "i_star":
                cells[LABELS[v]]["i_star_active"] = {
                    f: _mean([s["rho_active"] for s in per[f] if s["rho_active"] is not None])
                    for f in folds}
    summary = {}
    for lab in list(LABELS.values()) + ["Topo-QoS-AP"]:
        row = {"loso_i_star": _mean(list(rho[lab].values()))}
        for o in ("i_star", "i_dyn", "i_comp", "i_star_active"):
            if lab in cells:
                row[o] = _mean(list(cells[lab][o].values()))
        summary[lab] = row
    zs = {}
    for v in ZERO_SHOT:
        p = RESULTS / f"realworld_zeroshot_{v}_amendment16.json"
        if p.exists():
            z = json.loads(p.read_text())
            active = [s.get("mean_rho_positive") for s in z["per_system"].values()
                      if s.get("mean_rho_positive") is not None]
            zs[LABELS[v]] = {"mean_rho": z.get("mean_rho_across_systems"),
                             "mean_rho_positive": _mean(active),
                             "per_system": {k: s.get("mean_rho") for k, s in z["per_system"].items()}}

    for k, v in summary.items():
        print(f"{k:16s} " + " ".join(f"{o}={x:.3f}" for o, x in v.items() if x is not None))
    for name, fm in fam.items():
        for k, c in fm.items():
            print(f"{name:3s} {k:32s} d={c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
                  f"won={c['won']}/12 p={c['p']:.4f} p_holm={c['p_holm']:.4f}")
    for k, c in descriptive.items():
        print(f"desc {k:32s} d={c['delta']:+.3f} won={c['won']}/12 p={c['p']:.4f}")
    for k, c in to_indeg.items():
        print(f"toInDeg {k:16s} d={c['mean']:+.3f} TOST p={c['tost_t']['p']:.3f}")
    print("zero-shot:", {k: round(v["mean_rho"], 3) for k, v in zs.items() if v["mean_rho"] is not None})
    print("rules:", rules)
    _write("referee_round11_amendment16.json",
           {"gate_G0": gate, **fam, "descriptive": descriptive, "to_indeg": to_indeg,
            "decision_rules": rules, "summary": summary, "zero_shot": zs,
            "per_fold": {"loso_i_star": rho, "cells": cells, "indeg": indeg}},
           experiment="Amendment 16 arms", margin=MARGIN)
    return 0 if all(g["passed"] for g in gate.values()) else 1


def main() -> int:
    stages = {"amendment16": cmd_amendment16}
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stages", nargs="+", choices=list(stages))
    args = ap.parse_args()
    return max(stages[s](args) for s in args.stages)


if __name__ == "__main__":
    sys.exit(main())
