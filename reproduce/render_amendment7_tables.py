#!/usr/bin/env python3
"""
reproduce/render_amendment7_tables.py — supplementary tables for Amendment 7
===========================================================================

Renders ``docs/research/jss/latex/supp_amendment7.tex`` from the artifacts that
``reproduce/training_free_suite.py`` writes, so the supplementary tables cannot
drift from the runs:

    results/tf_baselines.json              per-fold and per-system training-free rankers
    results/qos_attribution_controls.json  Topo-Mult / Topo-QoS-Perm
    results/qos_indep_corpus.json          the QoS-independent corpus
    results/topo_substrate_check.json      what the published Topo measured
    results/oracle_param_sensitivity.json  I* over theta x damping
    results/system_model_descriptives.json label structure of folds vs. systems

Usage:
    PYTHONPATH=. python reproduce/render_amendment7_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "research" / "jss" / "latex" / "supp_amendment7.tex"

RANKERS = ["Topo-QoS", "Topo (projection)", "InDeg", "Reach", "Reach-QoS", "CDI"]
LABEL = {
    "Topo-QoS": r"\texttt{Topo-QoS}",
    "Topo (projection)": "Betweenness (proj.)",
    "InDeg": "InDeg",
    "Reach": "Reach",
    "Reach-QoS": "Reach-QoS",
    "CDI": "CDI",
}


def _load(name: str) -> Dict[str, Any]:
    return json.loads((RESULTS / name).read_text())


def _f(x: Optional[float], nd: int = 3, sign: bool = False) -> str:
    if x is None:
        return "---"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def _esc(s: str) -> str:
    return s.replace("&", r"\&")


def per_fold_table(tf: Dict[str, Any]) -> str:
    folds = tf["per_fold"]
    rows = []
    for name, row in folds.items():
        cells = [_f(row[p]["rho"]) for p in RANKERS]
        rows.append(rf"{_esc(name)} & " + " & ".join(cells) + r" \\")
    s = tf["summary"]
    mean = " & ".join(_f(s[p]["loso_mean_rho"]) for p in RANKERS)
    act = " & ".join(_f(s[p]["loso_mean_rho_active"]) for p in RANKERS)
    ovl = " & ".join(_f(s[p]["loso_mean_overlap"]) for p in RANKERS)
    head = " & ".join(rf"\textbf{{{LABEL[p]}}}" for p in RANKERS)
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Training-free rankers of Amendment~7, per LOSO holdout: Spearman $\rho$ "
        r"against $I^*(v)$, Application population, labels regenerated with the published "
        r"oracle settings. \texttt{Topo-QoS} reproduces the published per-fold values to three "
        r"decimals (\texttt{results/tf\_reproduction\_gate.json}). Betweenness (proj.) is unweighted "
        r"betweenness on the same Application--Library projection. Rendered from "
        r"\texttt{results/tf\_baselines.json}.}",
        r"\label{tab:a7-folds}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(RANKERS) + "}",
        r"\toprule",
        r"\textbf{Holdout} & " + head + r" \\",
        r"\midrule",
        *rows,
        r"\midrule",
        r"\textbf{Mean $\rho$} & " + mean + r" \\",
        r"\textbf{Mean $\rho_{>0}$ (active)} & " + act + r" \\",
        r"\textbf{Mean Overlap@$K$} & " + ovl + r" \\",
        r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}",
    ])


def systems_table(tf: Dict[str, Any]) -> str:
    sysd = tf["per_system"]
    rows = []
    for name, row in sysd.items():
        cells = [f"{_f(row[p]['rho'])} / {_f(row[p]['overlap_at_k'])}" for p in RANKERS]
        rows.append(rf"{_esc(name)} & {row['Topo-QoS']['n_apps']} & " + " & ".join(cells) + r" \\")
    s = tf["summary"]
    mean = " & ".join(f"{_f(s[p]['systems_mean_rho'])} / {_f(s[p]['systems_mean_overlap'])}"
                      for p in RANKERS)
    pr = " & ".join(_f(s[p]["systems_mean_pr_auc"]) for p in RANKERS)
    act = " & ".join(_f(s[p]["systems_mean_rho_active"]) for p in RANKERS)
    head = " & ".join(rf"\textbf{{{LABEL[p]}}}" for p in RANKERS)
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Training-free rankers on the five hand-authored system models: Spearman "
        r"$\rho$ / Overlap@$K$, Application population. Same oracle settings as "
        r"Table~\ref{M-tab:9b} of the main manuscript, but projection and closed-form scores are computed by the "
        r"Amendment~7 harness from the committed topology files; \texttt{Topo-QoS} scores "
        r"$0.582$ here against $0.526$ in the main manuscript's zero-shot table, whose "
        r"reference scores use the native-graph projection path and the cached articulation "
        r"term. Rendered from \texttt{results/tf\_baselines.json}.}",
        r"\label{tab:a7-systems}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lr" + "c" * len(RANKERS) + "}",
        r"\toprule",
        r"\textbf{System model} & \textbf{$|V_{\text{app}}|$} & " + head + r" \\",
        r"\midrule",
        *rows,
        r"\midrule",
        r"\textbf{Mean} & --- & " + mean + r" \\",
        r"\textbf{Mean PR-AUC} & --- & " + pr + r" \\",
        r"\textbf{Mean $\rho_{>0}$ (active)} & --- & " + act + r" \\",
        r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}",
    ])


def contrasts_table(tf: Dict[str, Any]) -> str:
    c = tf["contrasts_vs_topo_qos"]
    v = tf["vs_published_learned"]
    rows = []
    for p in ["InDeg", "Reach", "Reach-QoS", "CDI", "Topo (projection)"]:
        cc = c[p]
        cells = [
            f"{_f(cc['delta'], sign=True)} [{_f(cc['ci95'][0], sign=True)}, {_f(cc['ci95'][1], sign=True)}]",
            f"{cc['won']}/12",
            f"{cc['p']:.4f}" + (f" ({cc['p_holm']:.4f})" if "p_holm" in cc else ""),
        ]
        for k in ("vs_HGT-QoS_cpu", "vs_GAT-QoS_cpu", "vs_Hybrid-HGT_cpu", "vs_Hybrid-GAT_cpu"):
            d = v[p][k]
            cells.append(f"{_f(d['delta'], sign=True)} ({d['won']}/12, {d['p']:.4f})")
        rows.append(rf"{LABEL[p]} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Paired contrasts of the training-free rankers over the twelve LOSO folds "
        r"(two-sided Wilcoxon, bootstrap 95\% CI of the mean difference, $B = 2{,}000$). The "
        r"registered family (Amendment~7) is the four new rankers against \texttt{Topo-QoS}, "
        r"Holm-corrected in parentheses; the unweighted projection betweenness is descriptive. "
        r"The last four columns pair each ranker with the published CPU per-fold values of the "
        r"learned and hybrid engines (Section~\ref{supp:hybrid-folds}): $\Delta\rho$ "
        r"(folds won, $p$).}",
        r"\label{tab:a7-contrasts}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llllllll}",
        r"\toprule",
        r"\textbf{Ranker} & \textbf{$\Delta\rho$ vs \texttt{Topo-QoS} [95\% CI]} & \textbf{Won} & "
        r"\textbf{$p$ ($p_{\text{Holm}}$)} & \textbf{vs \texttt{HGT-QoS}} & \textbf{vs \texttt{GAT-QoS}} & "
        r"\textbf{vs Hybrid-HGT} & \textbf{vs Hybrid-GAT} \\",
        r"\midrule",
        *rows,
        r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}",
    ])


def controls_table(ctl: Dict[str, Any], ind: Dict[str, Any], sub: Dict[str, Any]) -> str:
    rows = []
    for name, r in ctl["per_fold"].items():
        s = sub["per_fold"][name]
        q = ind["per_fold"].get(name, {})
        rows.append(
            rf"{_esc(name)} & {_f(r['Topo (published)'])} & {_f(s['app-layer BT (memory rebuild)'])} & "
            rf"{_f(s['projection BT, unweighted'])} & {_f(r['Topo-QoS'])} & {_f(r['Topo-Mult'])} & "
            rf"{_f(r['Topo-QoS-Perm_mean'])} & {_f(q.get('Topo (projection)'))} & {_f(q.get('Topo-QoS'))} \\"
        )
    m = ctl["summary"]
    ss = sub["summary"]
    iq = ind["summary"]
    mean = (rf"\textbf{{Mean}} & {_f(ss['published Topo'])} & {_f(ss['app-layer BT (memory rebuild)'])} & "
            rf"{_f(ss['projection BT, unweighted'])} & {_f(m['Topo-QoS']['mean_rho'])} & "
            rf"{_f(m['Topo-Mult']['mean_rho'])} & {_f(m['Topo-QoS-Perm_mean']['mean_rho'])} & "
            rf"{_f(iq['Topo (projection)'])} & {_f(iq['Topo-QoS'])} \\")
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Where the closed-form gain comes from, per LOSO holdout (Spearman $\rho$, "
        r"Application population). \emph{Published Topo}: the registered comparator, which read "
        r"betweenness from the analysis stage's application-layer dependency graph; "
        r"\emph{app-layer BT}: that betweenness rebuilt in memory (\texttt{MemoryRepository}, "
        r"which differs slightly from Neo4j on Rule-1 weights). \emph{Proj.\ BT}: unweighted "
        r"betweenness on the Application--Library projection \texttt{Topo-QoS} uses. "
        r"\emph{Topo-Mult}: \texttt{Topo-QoS} with every topic weight fixed at $0.5$. "
        r"\emph{Perm}: \texttt{Topo-QoS} with QoS profiles permuted across topics (mean of 20). "
        r"\emph{Indep.}: the corpus regenerated with QoS no longer steering topology "
        r"(\texttt{qos\_affinity: false}), relabelled. Rendered from "
        r"\texttt{results/qos\_attribution\_controls.json}, \texttt{results/topo\_substrate\_check.json} "
        r"and \texttt{results/qos\_indep\_corpus.json}.}",
        r"\label{tab:a7-controls}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Substrate}} & \multicolumn{3}{c}{\textbf{QoS content}} & "
        r"\multicolumn{2}{c}{\textbf{Indep.\ corpus}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
        r"\textbf{Holdout} & \textbf{Published Topo} & \textbf{App-layer BT} & \textbf{Proj.\ BT} & "
        r"\textbf{\texttt{Topo-QoS}} & \textbf{Topo-Mult} & \textbf{Perm} & \textbf{Proj.\ BT} & "
        r"\textbf{\texttt{Topo-QoS}} \\",
        r"\midrule",
        *rows,
        r"\midrule",
        mean,
        r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}",
    ])


def oracle_table(orc: Dict[str, Any]) -> str:
    rows = []
    for key, r in orc["per_setting"].items():
        th, st = key.split(",")
        rows.append(
            rf"{th.split('=')[1]} & {st.split('=')[1]} & {_f(r['label_agreement_mean'])} & "
            rf"{_f(r['label_agreement_min'])} & {_f(r['topo_qos_mean_rho'])} \\"
        )
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Sensitivity of $I^*(v)$ to its propagation threshold $\theta$ and per-wave "
        r"damping step (floor $0.25$). Label agreement is Spearman $\rho$ against the shipped "
        r"setting ($\theta = 0.2$, step $0.15$) on Applications, mean and minimum over the twelve "
        r"folds; the last column is \texttt{Topo-QoS} scored against each label set. Rendered "
        r"from \texttt{results/oracle\_param\_sensitivity.json}.}",
        r"\label{tab:a7-oracle}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"\textbf{$\theta$} & \textbf{Damping step} & \textbf{Agreement (mean)} & "
        r"\textbf{Agreement (min)} & \textbf{\texttt{Topo-QoS} $\rho$} \\",
        r"\midrule",
        *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])


def descriptives_table(desc: Dict[str, Any], tf: Dict[str, Any]) -> str:
    def row(label: str, d: Dict[str, Any], rule: Optional[float]) -> str:
        return (rf"{label} & {d['n_apps']:.0f} & {_f(d['zero_share'])} & "
                rf"{_f(d['tie_fraction'])} & {_f(d['projection_density'])} & {_f(rule)} \\")
    ir = tf["inert_rule"]
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Label structure of the LOSO folds and the system models (means), and the "
        r"inert-vs-active rule of Amendment~7: a component is predicted to propagate failure "
        r"($I^* > 0$) iff it has at least one transitive dependent on the projection. Rendered "
        r"from \texttt{results/system\_model\_descriptives.json} and \texttt{results/tf\_baselines.json}.}",
        r"\label{tab:a7-descriptives}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Set} & \textbf{$|V_{\text{app}}|$} & \textbf{Zero share} & "
        r"\textbf{Tie fraction} & \textbf{Proj.\ density} & \textbf{Rule bal.\ acc.} \\",
        r"\midrule",
        row("Twelve LOSO folds", desc["loso_mean"], ir["loso_mean_balanced_accuracy"]),
        row("Five system models", desc["systems_mean"], ir["systems_mean_balanced_accuracy"]),
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])


def main() -> int:
    tf = _load("tf_baselines.json")
    parts: List[str] = [
        "% Generated by reproduce/render_amendment7_tables.py -- do not edit by hand.",
        per_fold_table(tf),
        contrasts_table(tf),
        systems_table(tf),
        controls_table(_load("qos_attribution_controls.json"), _load("qos_indep_corpus.json"),
                       _load("topo_substrate_check.json")),
        oracle_table(_load("oracle_param_sensitivity.json")),
        descriptives_table(_load("system_model_descriptives.json"), tf),
    ]
    OUT.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
