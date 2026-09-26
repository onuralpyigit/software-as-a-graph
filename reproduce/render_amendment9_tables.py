#!/usr/bin/env python3
"""
reproduce/render_amendment9_tables.py — supplementary tables for Amendments 9 and 10
===================================================================================

Renders ``docs/research/jss/latex/supp_amendment9.tex`` from the artifacts of
``make -f reproduce/Makefile rq-dependency-graph`` (Amendment 9) and
``reproduce/training_free_suite.py derivation`` (Amendment 10), so the
supplementary tables cannot drift from the runs:

    results/dependency_graph_contrasts.json        per-fold rows, zero-shot, probe
    results/derivation_ablation.json               InDeg / Reach vs raw-graph counts

Usage:
    PYTHONPATH=. python reproduce/render_amendment9_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from reproduce.render_amendment7_tables import _esc, _f
from reproduce.training_free_suite import SYSTEMS
from saag.evaluation.variant_registry import label

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "research" / "jss" / "latex" / "supp_amendment9.tex"

FOLDS = {
    "atm_system": "ATM", "av_system": "AV System", "enterprise_system": "Enterprise",
    "financial_trading_system": "Financial Trading", "healthcare_system": "Healthcare",
    "hub_and_spoke_system": "Enterprise Integration (ESB)",
    "industrial_scada_system": "Industrial SCADA", "iot_smart_city_system": "IoT Smart City",
    "logistics_fleet_system": "Logistics Fleet", "microservices_system": "Microservices",
    "realtime_gaming_system": "Real-Time Gaming", "telecom_ran_system": "Telecom RAN",
}
#: (per_fold key, printed label), raw-multigraph counterpart beside each projection arm.
A9_COLUMNS = [
    ("InDeg", "InDeg"), ("Reach", "Reach"),
    ("gl_full_cap@published", "GAT"), ("gl_proj_cap", "GAT-P"),
    ("gl_full_qos16_cap@published", "GAT-QoS"), ("gl_proj_qos16_cap", "GAT-P-QoS"),
    ("gl_qos16_prior@published", "Hybrid-GAT"), ("gl_proj_qos16_indeg_prior", "Hybrid-GAT-P"),
    ("hgl_qos@published", "HGT-QoS"), ("hgl_proj_qos", "HGT-P-QoS"),
]
A10_ARMS = ["InDeg", "Degree-raw", "Pubs-raw", "Reach", "Reach-R1"]


def _load(name: str) -> Dict[str, Any]:
    p = RESULTS / name
    if not p.exists():
        p = ROOT / "data" / "benchmarks" / name
    return json.loads(p.read_text())


def _table(caption: str, label: str, cols: str, head: List[str], rows: List[str]) -> str:
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        " & ".join(rf"\textbf{{{h}}}" for h in head) + r" \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}",
    ])


def a9_folds(d: Dict[str, Any]) -> str:
    pf, means = d["per_fold"], d["means"]
    rows = [rf"{_esc(name)} & " + " & ".join(_f(pf[k][sid]["rho"]) for k, _ in A9_COLUMNS) + r" \\"
            for sid, name in FOLDS.items()]
    rows.append(r"\midrule")
    rows.append(r"\textbf{Mean} & " + " & ".join(_f(means[k]["loso_mean_rho"]) for k, _ in A9_COLUMNS) + r" \\")
    rows.append(r"Mean $\rho_{>0}$ & " + " & ".join(_f(means[k]["loso_mean_rho_active"]) for k, _ in A9_COLUMNS)
                + r" \\")
    return _table(
        r"Learners on the dependency graph (Amendment~9) beside their raw-multigraph counterparts "
        r"and the dependency counts, per LOSO holdout: Spearman $\rho$ against $I^*(v)$, Application "
        r"population, mean over five seeds. Raw-multigraph values are read from their own clean "
        r"artifacts; \texttt{InDeg} and \texttt{Reach} are recomputed on the same labels and match "
        r"Table~\ref{tab:a7-folds} exactly. Rendered from \texttt{results/dependency\_graph\_contrasts.json}.",
        "tab:a9-folds", "l" + "c" * len(A9_COLUMNS), ["Holdout", *[lab for _, lab in A9_COLUMNS]], rows)


def a9_contrasts(d: Dict[str, Any]) -> str:
    rows = []
    for name, c in d["contrasts"].items():
        lo, hi = c["ci95"]
        rows.append(rf"{_esc(name)} & {_f(c['delta'], sign=True)} & [{_f(lo, sign=True)}, {_f(hi, sign=True)}]"
                    rf" & {c['won']}/{c['n']} & {c['p']:.4f} & {c['p_holm']:.3f} \\")
    dec = d["decisions"]
    rule = ", ".join(f"{k} {'triggered' if v['triggered'] else 'not triggered'}" for k, v in dec.items())
    return _table(
        r"The twelve registered contrasts of Amendment~9: two-sided Wilcoxon over the twelve folds, "
        r"fold-bootstrap 95\% CI ($B = 2{,}000$), Holm within this exploratory family. Decision "
        rf"rules: {rule}. Rendered from \texttt{{results/dependency\_graph\_contrasts.json}}.",
        "tab:a9-contrasts", "lccccc", ["Contrast", r"$\Delta\rho$", "95\\% CI", "Won", "$p$",
                                       r"$p_{\text{Holm}}$"], rows)


def a9_zeroshot(d: Dict[str, Any]) -> str:
    z = d["zeroshot"]
    cols = [c for c in A9_COLUMNS if z.get(c[0])]
    systems = list(z["InDeg"]["per_system"])
    rows = [rf"{_esc(SYSTEMS.get(s, s))} & "
            + " & ".join(_f(z[k]["per_system"][s]["rho"]) for k, _ in cols) + r" \\" for s in systems]
    rows.append(r"\midrule")
    rows.append(r"\textbf{Mean} & " + " & ".join(_f(z[k]["mean_rho"]) for k, _ in cols) + r" \\")
    return _table(
        r"Zero-shot transfer to the five system models (descriptive): learners on the dependency "
        r"graph trained on all twelve synthetic scenarios (3 layers, 300 epochs, five seeds) beside "
        r"their raw-multigraph counterparts and the dependency counts. Spearman $\rho$ against "
        r"$I^*(v)$, Application population. Rendered from \texttt{results/dependency\_graph\_contrasts.json}.",
        "tab:a9-zeroshot", "l" + "c" * len(cols), ["System model", *[lab for _, lab in cols]], rows)


def a9_probe(d: Dict[str, Any]) -> str:
    probe = d["receptive_field_probe"]
    g = probe["gradient_probe"]
    rows = [rf"{_esc(FOLDS[sid])} & {v['n_app']} & {v['gl_proj_qos16_cap']['mean_rf_nodes']:.1f}"
            rf" & {_f(v['gl_proj_qos16_cap']['share_rf_equals_3hop_dependents'])}"
            rf" & {_f(v['hgl_proj_qos']['mean_rf_share'])} \\" for sid, v in g.items()]
    dele = probe["edge_deletion_max_abs_delta"]
    note = "; ".join(f"{label(k, 'loso')} {max(v.values()):.2f}" for k, v in dele.items()
                     if isinstance(v, dict) and "skipped" not in v)
    return _table(
        r"Receptive field of the dependency-graph learners at an Application (random weights, "
        r"3 layers): mean receptive-field size of the GAT, the share of Applications whose receptive "
        r"field is exactly the Application and its dependents within three hops, and the share of the "
        r"graph the bidirectional HGT reaches. Trained zero-shot checkpoints (seed 42), largest change in "
        rf"any Application prediction when every edge is deleted: {note}. Rendered from "
        r"\texttt{results/dependency\_graph\_contrasts.json}.",
        "tab:a9-probe", "lcccc", ["Holdout", r"$|V_{\text{app}}|$", "GAT RF (nodes)",
                                  "RF = 3-hop dependents", "HGT RF share"], rows)


def a10_derivation(d: Dict[str, Any]) -> str:
    pf, s = d["per_fold"], d["summary"]
    rows = [rf"{_esc(name)} & " + " & ".join(_f(pf[name][a]["rho"]) for a in A10_ARMS) + r" \\"
            for name in FOLDS.values()]
    rows.append(r"\midrule")
    rows.append(r"\textbf{Mean} & " + " & ".join(_f(s[a]["loso_mean_rho"]) for a in A10_ARMS) + r" \\")
    rows.append(r"Five system models & " + " & ".join(_f(s[a]["systems_mean_rho"]) for a in A10_ARMS) + r" \\")
    con = "; ".join(
        f"{k} {_f(c['delta'], sign=True)} ({c['won']}/{c['n']}, Holm $p = {c['p_holm']:.4f}$)"
        for k, c in d["contrasts"].items())
    dec = d["decisions"]
    rule = ", ".join(f"{k} {'triggered' if v['triggered'] else 'not triggered'}" for k, v in dec.items())
    return _table(
        r"The value of the dependency derivation (Amendment~10): dependency counts on the derived graph "
        r"against counts available on the raw multigraph without derivation (Degree-raw: total degree; "
        r"Pubs-raw: topics published) and against Rule-1-only reach (Reach-R1). Spearman $\rho$, "
        rf"Application population. Registered contrasts: {con}. Decision rules: {rule}. For every "
        r"Application in all seventeen graphs, \texttt{InDeg} equals the raw 2-hop subscriber count "
        r"(max $|\Delta| = 0$). Rendered from \texttt{results/derivation\_ablation.json}.",
        "tab:a10-derivation", "l" + "c" * len(A10_ARMS), ["Holdout", *A10_ARMS], rows)


def main() -> int:
    dg = _load("dependency_graph_contrasts.json")
    dv = _load("derivation_ablation.json")
    parts = ["% Generated by reproduce/render_amendment9_tables.py -- do not edit by hand.",
             a9_folds(dg), a9_contrasts(dg), a9_zeroshot(dg), a9_probe(dg), a10_derivation(dv)]
    OUT.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
