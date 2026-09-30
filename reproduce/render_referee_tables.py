#!/usr/bin/env python3
"""
reproduce/render_referee_tables.py — supplementary tables for Amendment 12
===========================================================================

Renders docs/research/jss/latex/supp_referee.tex from the Amendment 12 artifacts
(data/benchmarks/referee_round7_*.json). Trains and simulates nothing; the
reconciler re-renders this file and compares it byte for byte.

Usage:
    PYTHONPATH=. python reproduce/render_referee_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from reproduce.render_amendment7_tables import _esc, _f
from reproduce.render_amendment9_tables import _table
from reproduce.training_free_suite import FOLDS, SYSTEMS

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "benchmarks"
OUT = ROOT / "docs" / "research" / "jss" / "latex" / "supp_referee.tex"
RAW_ARMS = ("InDeg", "Pubs-raw", "Degree-raw", "RevPR-raw", "Reach-R1", "Reach")
ORACLE_TEX = {"i_star": "$I^*$", "i_dyn": "$I_{\\text{dyn}}$ ($n = 30$/fold)",
              "i_comp": "$I_{\\text{comp}}$"}


def _load(name: str) -> Dict[str, Any]:
    return json.loads((BENCH / name).read_text())


def raw_folds(d: Dict[str, Any], oracle: str) -> str:
    per, summ = d["per_scenario"], d["summary"][oracle]
    rows = [rf"{_esc(FOLDS[f])} & " + " & ".join(
        _f((per[f][oracle][a] or {}).get("rho")) for a in RAW_ARMS) + r" \\" for f in FOLDS]
    rows += [r"\midrule", r"\textbf{Mean} & " + " & ".join(
        _f(summ[a]["loso"]["mean"]) for a in RAW_ARMS) + r" \\"]
    return _table(
        rf"Raw-multigraph rankers against {ORACLE_TEX[oracle]}, per LOSO fold (Amendment~12, R1; "
        r"Spearman $\rho$, Application population). PR-raw is constant for every Application and is "
        r"omitted. Rendered from \texttt{data/benchmarks/referee\_round7\_raw\_baselines.json}.",
        f"tab:ref-raw-{oracle.replace('_', '')}", "l" + "c" * len(RAW_ARMS),
        ["Holdout", *RAW_ARMS], rows)


def raw_contrasts(d: Dict[str, Any]) -> str:
    rows = []
    for k, c in d["contrasts"].items():
        lo, hi = c["ci95"]
        name = k
        for o, t in ORACLE_TEX.items():
            name = name.replace(o, t.split(" (")[0])
        rows.append(rf"{_esc(name)} & {_f(c['delta'], sign=True)} & [{_f(lo, sign=True)}, {_f(hi, sign=True)}]"
                    rf" & {c['won']}/{c['n']} & {c['p']:.4f} & {c['p_holm']:.4f} \\")
    dec = d["decisions"]
    return _table(
        r"The twelve registered R1 contrasts (Amendment~12): \texttt{InDeg} against each untyped or "
        r"trivially typed raw-graph ranker on each oracle, two-sided Wilcoxon over twelve folds, Holm "
        r"within the family. PR-raw has no defined $\rho$, so its contrasts compare \texttt{InDeg} with "
        rf"zero. Identity check: max $|\texttt{{InDeg}} - \text{{Raw2Hop}}| = {dec['D1']['identity_max_abs_diff']:.0f}$ "
        r"on all seventeen graphs. Rendered from \texttt{data/benchmarks/referee\_round7\_raw\_baselines.json}.",
        "tab:ref-raw-contrasts", "lccccc",
        ["Contrast", r"$\Delta\rho$", "95\\% CI", "Won", "$p$", r"$p_{\text{Holm}}$"], rows)


def partial_folds(d: Dict[str, Any]) -> str:
    per, s = d["per_fold"], d["summary"]
    arms = ("InDeg", "Reach", "Topo-QoS", "Analytic-I*")
    rows = [rf"{_esc(FOLDS[f])} & {per[f]['n']} & {_f(per[f]['rho_istar_idyn'])} & " + " & ".join(
        _f(per[f][a]["partial_given_istar"]) for a in arms) + r" \\" for f in FOLDS]
    rows += [r"\midrule", r"\textbf{Mean} & & " + _f(s["_oracle"]["rho_istar_idyn"]["mean"]) + " & "
             + " & ".join(_f(s[a]["partial_given_istar"]["mean"]) for a in arms) + r" \\",
             r"\textbf{95\% CI} & & & " + " & ".join(
                 "[" + ", ".join(_f(x) for x in s[a]["partial_given_istar"]["ci95"]) + "]"
                 for a in arms) + r" \\",
             r"\textbf{Given first order} & & & " + " & ".join(
                 _f(s[a]["partial_given_analytic"]["mean"]) for a in arms) + r" \\"]
    return _table(
        r"Partial Spearman correlation with $I_{\text{dyn}}$ after the rank of $I^*$ is regressed out "
        r"(Amendment~12, R2), per fold, on the published $n = 30$ lexical sample; $\rho(I^*, I_{\text{dyn}})$ "
        r"is the agreement of the two oracles on the same sample. The last row conditions on the "
        r"first-order expansion instead of $I^*$. Rendered from "
        r"\texttt{data/benchmarks/referee\_round7\_partial.json}.",
        "tab:ref-partial", "lcccccc",
        ["Holdout", "$n$", r"$\rho(I^*, I_{\text{dyn}})$", "InDeg", "Reach", "Topo-QoS", "First-order"],
        rows)


def learned(d: Dict[str, Any]) -> str:
    rows = []
    for e, b in d.items():
        if e.startswith("_") or "summary" not in b:
            continue
        sm = b["summary"]
        rows.append(rf"{_esc(e)} & {_f(b['published_i_star'])} & {_f(b['per_seed_logs_mean_i_star'])} & "
                    rf"{_f(sm['i_star']['rho']['mean'])} & {_f(sm['i_dyn']['rho']['mean'])} & "
                    rf"{_f(sm['i_comp']['rho']['mean'])} \\")
    ref = d["_InDeg"]
    rows += [r"\midrule", rf"InDeg & {_f(ref['i_star']['mean'])} & --- & {_f(ref['i_star']['mean'])} & "
             rf"{_f(ref['i_dyn']['mean'])} & {_f(ref['i_comp']['mean'])} \\"]
    return _table(
        r"Learned engines re-scored on every oracle (Amendment~12, R3). The saved predictions are the "
        r"mean of five seeds' predictions, so registered gate G3 (reproduce the published per-seed "
        r"mean) fails by construction; the per-seed logs reproduce every published value. Columns: "
        r"published LOSO $\rho$; mean of the per-seed logs; seed-ensemble $\rho$ on each oracle. "
        r"Rendered from \texttt{data/benchmarks/referee\_round7\_learned\_oracles.json}.",
        "tab:ref-learned", "lccccc",
        ["Engine", "Published", "Seed logs", "Ensemble $I^*$", r"Ensemble $I_{\text{dyn}}$",
         r"Ensemble $I_{\text{comp}}$"], rows)


def averaging(d: Dict[str, Any]) -> str:
    per, spread = d["per_ranker"], d["seed_spread"]
    order = sorted(per, key=lambda k: -per[k]["arithmetic"])
    rows = []
    for k in order:
        sp = spread.get(k)
        rows.append(rf"{_esc(k)} & {_f(per[k]['arithmetic'])} & {_f(per[k]['fisher_z'])} & "
                    rf"{_f(per[k]['size_weighted'])} & "
                    + (rf"{_f(sp['mean_sd'])} & {_f(sp['max_range'])}" if sp else "--- & ---") + r" \\")
    return _table(
        r"Table~\ref{M-tab:hybrid}'s means under three averaging rules, and the seed spread of each "
        r"learned engine: the mean over folds of the SD of the five seeds' $\rho$, and the largest "
        r"within-fold range. Descriptive, computed from existing artifacts. Rendered from "
        r"\texttt{data/benchmarks/referee\_round7\_averaging.json}.",
        "tab:ref-averaging", "lccccc",
        ["Ranker", "Arithmetic", "Fisher-$z$", r"$|V_{\text{app}}|$-weighted", "Seed SD", "Max range"],
        rows)


def zeroshot(d: Dict[str, Any]) -> str:
    rows = []
    for sid, name in SYSTEMS.items():
        r = d["per_system"][sid]
        ds = r["descriptors"]
        rows.append(rf"{_esc(name)} & {ds['n_apps']} & {_f(ds['zero_impact_share'], 2)} & "
                    rf"{_f(ds['indeg_gini'], 2)} & {ds['max_dependency_depth']} & "
                    rf"{_f(r['InDeg']['rho'])} & {_f(r['Reach']['rho'])} \\")
    fd = d["fold_descriptors"].values()
    mean = lambda k: sum(v[k] for v in fd) / len(fd)  # noqa: E731
    rows += [r"\midrule", rf"\textit{{Synthetic folds (mean)}} & {mean('n_apps'):.0f} & "
             rf"{_f(mean('zero_impact_share'), 2)} & {_f(mean('indeg_gini'), 2)} & "
             rf"{mean('max_dependency_depth'):.1f} & --- & --- \\"]
    return _table(
        r"The five system models beside the synthetic folds (Amendment~12, R5): Applications, share with "
        r"zero $I^*$, Gini coefficient of \texttt{InDeg}, longest dependency chain, and \texttt{InDeg} and "
        r"\texttt{Reach} scored on the learned engines' own zero-shot labels. Rendered from "
        r"\texttt{data/benchmarks/referee\_round7\_zeroshot.json}.",
        "tab:ref-zeroshot", "lcccccc",
        ["System model", r"$|V_{\text{app}}|$", "Zero share", "Fan-in Gini", "Depth", "InDeg", "Reach"],
        rows)


def latency(d: Dict[str, Any]) -> str:
    rows = []
    for r in d["sizes"]:
        ist = r["istar_s"]
        rows.append(rf"{r['n_actual']:,} & {r['n_edges']:,} & {1000 * r['projection_s']:.1f} & "
                    rf"{1000 * r['indeg_s']:.2f} & {1000 * r['reach_s']:.1f} & "
                    + (rf"{ist:.1f} ({r['istar_repeats']})" if ist is not None else "not timed")
                    + r" \\")
    return _table(
        r"Counting path against one $I^*$ labeling pass on generated graphs (Amendment~12, R6; median of "
        r"repeats, count in brackets for $I^*$). Projection, \texttt{InDeg} and \texttt{Reach} in ms; "
        r"$I^*$ in s. Rendered from \texttt{data/benchmarks/referee\_round7\_latency.json}.",
        "tab:ref-latency", "rrrrrr",
        ["$|V|$", "$|E|$", "Projection (ms)", "InDeg (ms)", "Reach (ms)", "$I^*$ (s)"],
        [row.replace(",", "{,}") for row in rows])


def main() -> int:
    raw = _load("referee_round7_raw_baselines.json")
    parts = ["% Generated by reproduce/render_referee_tables.py -- do not edit by hand.",
             raw_folds(raw, "i_star"), raw_folds(raw, "i_dyn"), raw_folds(raw, "i_comp"),
             raw_contrasts(raw), partial_folds(_load("referee_round7_partial.json")),
             learned(_load("referee_round7_learned_oracles.json")),
             averaging(_load("referee_round7_averaging.json")),
             zeroshot(_load("referee_round7_zeroshot.json"))]
    if (BENCH / "referee_round7_latency.json").exists():
        parts.append(latency(_load("referee_round7_latency.json")))
    OUT.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
