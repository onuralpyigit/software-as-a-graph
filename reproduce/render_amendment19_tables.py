#!/usr/bin/env python3
"""
reproduce/render_amendment19_tables.py — Supplementary Tables tab:a19 and tab:a19lc
==================================================================================
Renders the Amendment 19 control arms and learning curve from
data/benchmarks/referee_round14_{amendment19,lc}.json into
docs/research/jss/latex/supp_amendment19.tex (generated; do not edit by hand).
``reconcile_manuscript.check_amendment19`` re-renders and compares.

Usage:
    PYTHONPATH=. python reproduce/render_amendment19_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "benchmarks"
OUT = ROOT / "docs" / "research" / "jss" / "latex" / "supp_amendment19.tex"
EQ7 = r"Eq.~\eqref{M-eq:rate-expansion}"

#: (section title, [(arm, comparator, family key or None, oracle)]); oracle "i_dyn" rows
#: report rho on I_dyn, all others on I*.
ROWS = [
    ("F14: sum aggregation with reverse edges on the raw multigraph (registered)", [
        ("GIN-P-QoS-min", "GIN-QoS-R-min", "F14", "i_star"),
        ("GIN-P-QoS", "GIN-QoS-R", "F14", "i_star"),
        ("GIN-P-QoS-const", "GIN-QoS-R-const", "F14", "i_star")]),
    ("Descriptive (no test)", [
        ("GIN-QoS-R", "GAT-QoS-R", None, "i_star"),
        ("GIN-QoS-R-min", "GAT-QoS-R-min", None, "i_star"),
        ("GIN-QoS-R-const", "InDeg", None, "i_star")]),
    ("F15: rate-fed learned approximations, scored on $I_{\\text{dyn}}$ (registered)", [
        ("GAT-P-QoS-dyn+rate", "Eq7", "F15", "i_dyn"),
        ("GAT-P-QoS-dyn+rate-e", "Eq7", "F15", "i_dyn"),
        ("GIN-P-QoS-dyn+rate-e", "Eq7", "F15", "i_dyn"),
        ("GAT-P-QoS-dyn+rate", "GAT-P-QoS-dyn", None, "i_dyn"),
        ("GAT-P-QoS-dyn+rate-e", "GAT-P-QoS-dyn", None, "i_dyn"),
        ("GIN-P-QoS-dyn+rate-e", "GAT-P-QoS-dyn", None, "i_dyn")]),
    ("F16: tie-aware listwise loss (registered)", [
        ("GAT-P-QoS-tie", "GAT-P-QoS", "F16", "i_star"),
        ("GAT-P-QoS-tie", "GAT-QoS-R-tie", "F16", "i_star")]),
    ("Small-capacity arm (descriptive)", [
        ("GAT-S-P-QoS", "GAT-P-QoS", None, "i_star"),
        ("GAT-S-P-QoS", "InDeg", None, "i_star")]),
]
LC_LEARNERS = ("GAT-P-QoS", "GIN-P-QoS", "GAT-QoS")
LC_K = ("1", "2", "4", "8", "11")


def _label(name: str) -> str:
    if name == "Eq7":
        return EQ7
    if name == "InDeg":
        return r"\texttt{InDeg}"
    return name.replace("-dyn", r"$\to$dyn")


def _f(x: float) -> str:
    return f"{x:.3f}" if x >= 0 else f"$-${abs(x):.3f}"


def _p(p: float) -> str:
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def _contrast(a19: Dict[str, Any], arm: str, comp: str, fam: Optional[str]) -> Dict[str, Any]:
    key = f"{arm} vs {comp}"
    if fam is not None:
        return a19[fam][key]
    d = a19["descriptive"]
    for block in (d["F14"], d["F15"], d["small"]):
        if key in block:
            return block[key]
    raise KeyError(key)


def _rho(a19: Dict[str, Any], name: str, oracle: str) -> Optional[float]:
    if name == "InDeg":
        per = a19["per_fold"]["indeg"]
        return sum(per.values()) / len(per)
    if oracle == "i_dyn":
        return a19["summary"]["dyn_means"][name]
    return a19["summary"][name]["loso_i_star"]


def render_a19(a19: Dict[str, Any]) -> str:
    lines: List[str] = []
    for title, rows in ROWS:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{title}}}}} \\")
        for arm, comp, fam, oracle in rows:
            c = _contrast(a19, arm, comp, fam)
            lo, hi = c["ci95"]
            cr = _rho(a19, comp, oracle)
            cells = [
                _label(arm), f"{_label(comp)} (${cr:.3f}$)", f"{_rho(a19, arm, oracle):.3f}",
                f"${c['delta']:+.3f}$ $[{lo:+.3f}, {hi:+.3f}]$", f"{c['won']}/12",
                _p(c["p_holm"]) if fam is not None else "---",
            ]
            if oracle == "i_star" and arm in a19["summary"]:
                s = a19["summary"][arm]
                cells.append(f"{_f(s['i_dyn'])} / {_f(s['i_comp'])}")
            else:
                cells.append("---")
            z = a19["zero_shot"].get(arm)
            cells.append(f"{z['mean_rho']:.3f}" if z and z.get("mean_rho") is not None else "---")
            lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines[1:])  # the first \midrule follows the header's own
    spread = a19["descriptive"]["F16"]["tie_perm_mean_spread"]
    rules = a19["decision_rules"]
    return (
        r"""\begin{table}[htbp]
\centering
\small
\caption{Amendment~19 control arms: twelve LOSO folds, five seeds, one CPU invocation with every comparator re-run to an exact match, Application population. $\Delta\rho$ is paired by fold against the comparator named, with a bootstrap 95\% CI; Holm within each registered family (F14: aggregator; F15: rate-fed approximations against """
        + EQ7
        + r"""; F16: tie-aware loss). \texttt{GIN}: sum aggregation (GINE layers); \texttt{-R}: every raw-graph edge also passed in reverse; \texttt{-min}: oracle-aligned features zeroed; \texttt{-const}: every node feature zeroed; \texttt{+rate}: each node's summed declared publication rate as an input column; \texttt{+rate-e}: also each Rule-1 edge's share of """
        + EQ7
        + r""" as an edge column; \texttt{-tie}: tie-aware listwise loss; \texttt{GAT-S-P-QoS}: width 64. F15 rows report $\rho$ on $I_{\text{dyn}}$; all others on $I^*$. $I_{\text{dyn}}$ / $I_{\text{comp}}$: per-seed means from saved predictions. Zero-shot: mean $\rho$ on the five system models (--- where not run). Mean per-fold spread across three node-order permutations of \texttt{GAT-P-QoS-tie}: """
        + f"${spread:.3f}$"
        + r""" (ListMLE: $0.044$). Decision rules: """
        + ", ".join(f"{k} {v}" for k, v in rules.items())
        + r""". Registered secondary.}
\label{tab:a19}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llcccccc}
\toprule
Arm & Comparator & $\rho$ & $\Delta\rho$ [95\% CI] & Won & $p_{\text{Holm}}$ & $I_{\text{dyn}}$ / $I_{\text{comp}}$ & Zero-shot \\
\midrule
"""
        + body
        + "\n" + r"""\bottomrule
\end{tabular}%
}
\end{table}
"""
    )


def render_lc(lc: Dict[str, Any]) -> str:
    rows = []
    for name in LC_LEARNERS:
        m = lc["means"][name]
        c = lc["LC"][f"{name} K11 vs K4"]
        d8 = lc["descriptive"][name]["K11 vs K8"]
        cells = [name] + [f"{m[k]['mean']:.3f}" for k in LC_K] + [
            f"${c['delta']:+.3f}$ $[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]$", _p(c["p_holm"]),
            f"${d8['delta']:+.3f}$ $[{d8['ci95'][0]:+.3f}, {d8['ci95'][1]:+.3f}]$",
            lc["decision_rules"][name]]
        rows.append(" & ".join(cells) + r" \\")
    return (
        r"""\begin{table}[htbp]
\centering
\small
\caption{Learning curve (Amendment~19): LOSO Spearman $\rho$ against $I^*$ with each fold's learner trained on $K$ of its eleven training scenarios (mean over three nested subset draws and five seeds; $K = 11$ is the full corpus). $\Delta(11{-}4)$ is paired by fold, Holm across the three learners; $\Delta(11{-}8)$ is descriptive. Afferent coupling (\texttt{InDeg}) reaches """
        + f"${lc['indeg_mean']:.3f}$"
        + r""" without training. Overall rule: """
        + lc["decision_rules"]["overall"]
        + r""". Registered secondary.}
\label{tab:a19lc}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
Learner & $K=1$ & $K=2$ & $K=4$ & $K=8$ & $K=11$ & $\Delta(11{-}4)$ [95\% CI] & $p_{\text{Holm}}$ & $\Delta(11{-}8)$ [95\% CI] & Rule \\
\midrule
"""
        + "\n".join(rows)
        + "\n" + r"""\bottomrule
\end{tabular}%
}
\end{table}
"""
    )


def render() -> str:
    a19 = json.loads((BENCH / "referee_round14_amendment19.json").read_text())
    lc = json.loads((BENCH / "referee_round14_lc.json").read_text())
    return ("% Generated by reproduce/render_amendment19_tables.py -- do not edit by hand.\n"
            + render_a19(a19) + "\n" + render_lc(lc))


def main() -> None:
    OUT.write_text(render())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
