#!/usr/bin/env python3
"""
tools/render_table.py — Block C: Table 3 + Table 4 LaTeX/CSV/Markdown renderer
===============================================================================

Reads results/main_table.json (Block C) and/or results/loso_all_variants.json
(Block E) and produces:

  - results/table3_main_results.tex   (LaTeX booktabs table for paper §6.2)
  - results/table4_loso_results.tex   (LaTeX booktabs table for paper §6.5)
  - results/table3_main_results.csv   (CSV for Excel/R)
  - results/table3_main_results.md    (Markdown for README / GitHub)

Usage
-----
  python tools/render_table.py
  python tools/render_table.py --table3 results/main_table.json
  python tools/render_table.py --table4 results/loso_all_variants.json --tex-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Display config ─────────────────────────────────────────────────────────────

_SCENARIO_LABELS = {
    "atm_system":             "ATM System",
    "av_system":              "AV System",
    "iot_smart_city_system":  "IoT Smart City",
    "financial_trading_system": "Financial Trading",
    "healthcare_system":      "Healthcare",
    "hub_and_spoke_system":   "Hub-and-Spoke",
    "microservices_system":   "Microservices",
    "enterprise_system":      "Enterprise",
}

_VARIANT_LABELS = {
    "topo_baseline":     r"\textsc{Topo-BL}",
    "q_topo_baseline":   r"\textsc{Q-Topo-BL}",
    "homo_unweighted":   r"\textsc{Homo-U}",
    "homo_scalar":       r"\textsc{Homo-S}",
    "hgl":               r"\textsc{HGL}",
    "hetero_qos":        r"\textbf{Q-HGL}",
}
_VARIANT_LABELS_PLAIN = {
    "topo_baseline":     "Topo-BL",
    "q_topo_baseline":   "Q-Topo-BL",
    "homo_unweighted":   "Homo-U",
    "homo_scalar":       "Homo-S",
    "hgl":               "HGL",
    "hetero_qos":        "Q-HGL (ours)",
}
_VARIANT_ORDER = [
    "topo_baseline", "q_topo_baseline",
    "homo_unweighted", "homo_scalar",
    "hgl", "hetero_qos",
]

_RESULTS_DIR = Path("results")


# ── Table 3: Main results (8 scenarios × 4 variants) ─────────────────────────

def _load_table3_data(path: Path) -> Dict:
    data = json.loads(path.read_text())
    cells = data.get("cells", [])
    agg = data.get("aggregate", {})
    return {"cells": cells, "aggregate": agg}


def _get_cell_stats(agg: Dict, scenario: str, variant: str) -> Dict:
    key = f"{scenario}|{variant}"
    return agg.get(key, {})


def _format_rho(mean: Optional[float], ci_lo: Optional[float], ci_hi: Optional[float],
                bold: bool = False) -> str:
    """Format 'mean [lo, hi]' for LaTeX cell, optionally bold."""
    if mean is None:
        return r"\textemdash"
    s = f"{mean:.3f}"
    if ci_lo is not None and ci_hi is not None:
        s += rf" $[{ci_lo:.3f},{ci_hi:.3f}]$"
    if bold:
        s = rf"\textbf{{{s}}}"
    return s


def _pval_star(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 0.001: return "$^{***}$"
    if p < 0.01:  return "$^{**}$"
    if p < 0.05:  return "$^{*}$"
    return ""


def _calibration_marker(stats: Dict) -> str:
    """LaTeX footnote marker for the calibration policy of an aggregate cell."""
    cal = stats.get("calibration", "rank_matched")
    if cal == "rank_matched":
        return ""
    if "degenerate" in cal:
        return r"$^{\ddagger}$"
    if cal == "fixed":
        return r"$^{\dagger}$"
    return r"$^{?}$"


def _calibration_marker_md(stats: Dict) -> str:
    """Markdown footnote marker for the calibration policy of an aggregate cell."""
    cal = stats.get("calibration", "rank_matched")
    if cal == "rank_matched":
        return ""
    if "degenerate" in cal:
        return "‡"
    if cal == "fixed":
        return "†"
    return "?"


def _fmt_f1_md(stats: Dict) -> str:
    """Format mean F1 for markdown: shows NaN‡ for degenerate, value†/‡ for others."""
    if stats.get("n_needs_recalibration", 0) > 0:
        return "— (re-train)"
    f1 = stats.get("mean_f1")
    marker = _calibration_marker_md(stats)
    if f1 is None:
        return f"NaN{marker}"
    return f"{f1:.3f}{marker}"


def render_table3_tex(data: Dict, output: Path):
    """LaTeX booktabs Table 3: Spearman ρ per scenario × variant."""
    agg = data["aggregate"]
    scenarios = sorted({
        k.split("|")[0] for k in agg.keys()
        if not k.startswith("_")  # skip meta entries like _factorial_contrasts
    })
    if not scenarios:
        print("  No aggregate data found in table3 input.")
        return

    n_vars = len(_VARIANT_ORDER)
    col_spec = "ll" + "c" * n_vars + "c"
    header_row = " & ".join(_VARIANT_LABELS.get(v, v) for v in _VARIANT_ORDER)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Spearman $\rho$ (composite score) across 8 scenarios $\times$ {n_vars} variants,",
        r"         5 seeds, Bootstrap 95\% CI. $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ vs Q-HGL (Wilcoxon).}",
        r"\label{tab:main_results}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Scenario & GT & {header_row} & $\Delta\rho$ (QoS) \\",
        r"\midrule",
    ]

    for sc in scenarios:
        label = _SCENARIO_LABELS.get(sc, sc)
        cells = []
        # Find best mean_rho across variants to bold it
        best_rho = max(
            (agg.get(f"{sc}|{v}", {}).get("mean_rho", 0.0) or 0.0)
            for v in _VARIANT_ORDER
        )
        for var in _VARIANT_ORDER:
            stats = _get_cell_stats(agg, sc, var)
            mean_r = stats.get("mean_rho")
            ci_lo  = stats.get("ci_lo")
            ci_hi  = stats.get("ci_hi")
            p_val  = stats.get("wilcoxon_p_vs_hetero")
            is_best = mean_r is not None and abs(mean_r - best_rho) < 0.001
            cell = _format_rho(mean_r, ci_lo, ci_hi, bold=(is_best and var == "hetero_qos"))
            if var != "hetero_qos" and p_val is not None:
                cell += _pval_star(p_val)
            cells.append(cell)

        # Delta column: hetero_qos - hetero_no_qos
        stats_qos = _get_cell_stats(agg, sc, "hetero_qos")
        stats_none = _get_cell_stats(agg, sc, "hetero_no_qos")
        r_qos = stats_qos.get("mean_rho")
        r_none = stats_none.get("mean_rho")
        p_delta = stats_none.get("wilcoxon_p_vs_hetero") # p-value for the pair
        
        delta_str = r"\textemdash"
        if r_qos is not None and r_none is not None:
            diff = r_qos - r_none
            delta_str = rf"{'+' if diff >= 0 else ''}{diff:.3f}"
            delta_str += _pval_star(p_delta)

        gt_source = agg.get(f"{sc}|topo_baseline", {}).get("gt_source", "Sim")
        lines.append(rf"{label} & {gt_source} & {' & '.join(cells)} & {delta_str} \\")

    # Summary row: cross-scenario mean
    lines.append(r"\midrule")
    avg_cells = []
    import numpy as np
    all_means = []
    for var in _VARIANT_ORDER:
        rhos = [
            agg.get(f"{sc}|{var}", {}).get("mean_rho")
            for sc in scenarios
            if agg.get(f"{sc}|{var}", {}).get("mean_rho") is not None
        ]
        all_means.append(np.mean(rhos) if rhos else 0.0)
    
    best_avg = max(all_means)
    avg_cells = []
    for i, var in enumerate(_VARIANT_ORDER):
        m = all_means[i]
        cell = f"{m:.3f}"
        if abs(m - best_avg) < 0.001 and var == "hetero_qos":
            cell = rf"\textbf{{{cell}}}"
        avg_cells.append(cell)

    avg_delta = all_means[-1] - all_means[-2] # hetero_qos - hetero_no_qos
    avg_cells.append(rf"{'+' if avg_delta >= 0 else ''}{avg_delta:.3f}")

    lines.append(rf"\textbf{{Mean}} & & {' & '.join(avg_cells)} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"  Saved LaTeX Table 3: {output}")


def render_table3_csv(data: Dict, output: Path):
    agg = data["aggregate"]
    scenarios = sorted({k.split("|")[0] for k in agg if not k.startswith("_")})
    rows = []
    header = ["scenario", "gt_source"] + [f"{v}_rho" for v in _VARIANT_ORDER] + \
             [f"{v}_ci_lo" for v in _VARIANT_ORDER] + [f"{v}_ci_hi" for v in _VARIANT_ORDER] + \
             [f"{v}_pval" for v in _VARIANT_ORDER if v != "hetero_qos"]
    for sc in scenarios:
        gt_source = agg.get(f"{sc}|topo_baseline", {}).get("gt_source", "Sim")
        row = {"scenario": sc, "gt_source": gt_source}
        for v in _VARIANT_ORDER:
            st = agg.get(f"{sc}|{v}", {})
            row[f"{v}_rho"]   = st.get("mean_rho", "")
            row[f"{v}_ci_lo"] = st.get("ci_lo", "")
            row[f"{v}_ci_hi"] = st.get("ci_hi", "")
            if v != "hetero_qos":
                row[f"{v}_pval"] = st.get("wilcoxon_p_vs_hetero", "")
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV Table 3:    {output}")



def render_table3_md(data: Dict, output: Path):
    agg = data["aggregate"]
    scenarios = sorted({k.split("|")[0] for k in agg if not k.startswith("_")})
    
    headers = ["Scenario", "GT"] + [_VARIANT_LABELS_PLAIN.get(v, v) for v in _VARIANT_ORDER] + ["Δρ (QoS)"]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    for sc in scenarios:
        label = _SCENARIO_LABELS.get(sc, sc)
        best_rho = max((agg.get(f"{sc}|{v}", {}).get("mean_rho", 0.0) or 0.0) for v in _VARIANT_ORDER)
        gt_source = agg.get(f"{sc}|topo_baseline", {}).get("gt_source", "Sim")
        
        cells = [f"**{label}**", gt_source]
        for var in _VARIANT_ORDER:
            st = agg.get(f"{sc}|{var}", {})
            r = st.get("mean_rho")
            p = st.get("wilcoxon_p_vs_hetero")
            if r is None:
                cell = "—"
            else:
                cell = f"{r:.3f}"
                if abs(r - best_rho) < 0.001 and var == "hetero_qos":
                    cell = f"**{cell}**"
                if var != "hetero_qos" and p is not None:
                    if p < 0.001: cell += "***"
                    elif p < 0.01: cell += "**"
                    elif p < 0.05: cell += "*"
            cells.append(cell)
            
        # Delta
        r_qos = agg.get(f"{sc}|hetero_qos", {}).get("mean_rho")
        r_none = agg.get(f"{sc}|hetero_no_qos", {}).get("mean_rho")
        p_delta = agg.get(f"{sc}|hetero_no_qos", {}).get("wilcoxon_p_vs_hetero")
        if r_qos is not None and r_none is not None:
            diff = r_qos - r_none
            d_str = f"{'+' if diff >= 0 else ''}{diff:.3f}"
            if p_delta is not None:
                if p_delta < 0.001: d_str += "***"
                elif p_delta < 0.01: d_str += "**"
                elif p_delta < 0.05: d_str += "*"
            cells.append(d_str)
        else:
            cells.append("—")
            
        rows.append("| " + " | ".join(cells) + " |")

    # Mean row
    import numpy as np
    all_means = []
    for var in _VARIANT_ORDER:
        rhos = [agg.get(f"{sc}|{var}", {}).get("mean_rho") for sc in scenarios if agg.get(f"{sc}|{var}", {}).get("mean_rho") is not None]
        all_means.append(np.mean(rhos) if rhos else 0.0)
    
    avg_cells = ["**Mean**", ""]
    best_avg = max(all_means)
    for i, var in enumerate(_VARIANT_ORDER):
        m = all_means[i]
        cell = f"{m:.3f}"
        if abs(m - best_avg) < 0.001 and var == "hetero_qos":
            cell = f"**{cell}**"
        avg_cells.append(cell)
    
    avg_delta = all_means[-1] - all_means[-2]
    avg_cells.append(f"{'+' if avg_delta >= 0 else ''}{avg_delta:.3f}")
    rows.append("| " + " | ".join(avg_cells) + " |")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved Markdown Table 3: {output}")


def print_table3_console(data: Dict):
    agg = data["aggregate"]
    scenarios = sorted({k.split("|")[0] for k in agg if not k.startswith("_")})
    var_labels = [_VARIANT_LABELS_PLAIN.get(v, v) for v in _VARIANT_ORDER]
    col_w = 26

    print("\n  Table 3: Spearman ρ — Main Results")
    header = f"  {'Scenario':<30} {'GT':<12}" + "".join(f"{lbl:<{col_w}}" for lbl in var_labels) + "Δρ (QoS)"
    print(header)
    print("  " + "─" * (42 + col_w * len(_VARIANT_ORDER) + 12))

    for sc in scenarios:
        label = _SCENARIO_LABELS.get(sc, sc)
        best_rho = max((agg.get(f"{sc}|{v}", {}).get("mean_rho", 0.0) or 0.0) for v in _VARIANT_ORDER)
        gt_source = agg.get(f"{sc}|topo_baseline", {}).get("gt_source", "Sim")
        row = f"  {label:<30} {gt_source:<12}"
        for v in _VARIANT_ORDER:
            st = agg.get(f"{sc}|{v}", {})
            r = st.get("mean_rho")
            ci_lo = st.get("ci_lo")
            ci_hi = st.get("ci_hi")
            p_val = st.get("wilcoxon_p_vs_hetero")
            
            if r is None:
                cell = "—"
            else:
                cell = f"{r:.3f}"
                if ci_lo is not None and ci_hi is not None:
                    cell += f" [{ci_lo:.3f}, {ci_hi:.3f}]"
                if abs(r - best_rho) < 0.001 and v == "hetero_qos":
                    cell = f"*{cell}*"
            
            if v != "hetero_qos" and p_val is not None:
                if p_val < 0.001: cell += "***"
                elif p_val < 0.01: cell += "**"
                elif p_val < 0.05: cell += "*"
                
            row += cell.ljust(col_w)
        
        # Delta
        r_qos = agg.get(f"{sc}|hetero_qos", {}).get("mean_rho")
        r_none = agg.get(f"{sc}|hetero_no_qos", {}).get("mean_rho")
        p_delta = agg.get(f"{sc}|hetero_no_qos", {}).get("wilcoxon_p_vs_hetero")
        if r_qos is not None and r_none is not None:
            diff = r_qos - r_none
            d_str = f"{'+' if diff >= 0 else ''}{diff:.3f}"
            if p_delta is not None:
                if p_delta < 0.001: d_str += "***"
                elif p_delta < 0.01: d_str += "**"
                elif p_delta < 0.05: d_str += "*"
            row += d_str
            
        print(row)

        # Per-node-type breakdown (Block F story)
        node_types = sorted({nt for v in _VARIANT_ORDER for nt in agg.get(f"{sc}|{v}", {}).get("per_node_type", {})})
        for nt in node_types:
            subrow = f"    └─ {nt:<27} {'':<12}"
            for v in _VARIANT_ORDER:
                nt_rho = agg.get(f"{sc}|{v}", {}).get("per_node_type", {}).get(nt)
                cell = f"{nt_rho:.3f}" if nt_rho is not None else "—"
                subrow += cell.ljust(col_w)
            print(subrow)

    # Summary row: cross-scenario mean
    print("  " + "─" * (42 + col_w * len(_VARIANT_ORDER) + 12))
    import numpy as np
    all_means = []
    for var in _VARIANT_ORDER:
        rhos = [
            agg.get(f"{sc}|{var}", {}).get("mean_rho")
            for sc in scenarios
            if agg.get(f"{sc}|{var}", {}).get("mean_rho") is not None
        ]
        all_means.append(np.mean(rhos) if rhos else 0.0)
    
    avg_row = f"  {'Mean':<30} {'':<12}"
    for m in all_means:
        avg_row += f"{m:.3f}".ljust(col_w)
    
    avg_delta = all_means[-1] - all_means[-2]
    avg_row += f"{'+' if avg_delta >= 0 else ''}{avg_delta:.3f}"
    print(avg_row)


# ── Identification Metrics (F1, Prec, Rec, Top-5) ───────────────────────────

def render_id_metrics_md(data: Dict, output: Path):
    agg = data["aggregate"]
    scenarios = sorted({k.split("|")[0] for k in agg if not k.startswith("_")})

    header  = "| Scenario | GT | Variant | Spearman ρ | F1 | Precision | Recall | Accuracy | RMSE | MAE | NDCG@10 |"
    divider = "|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, divider]

    for sc in scenarios:
        label     = _SCENARIO_LABELS.get(sc, sc)
        gt_source = agg.get(f"{sc}|topo_baseline", {}).get("gt_source", "Sim")
        for v in _VARIANT_ORDER:
            st = agg.get(f"{sc}|{v}", {})
            f1_str = _fmt_f1_md(st)

            def _fmt(x):
                return "—" if x is None else f"{x:.3f}"

            rows.append(
                f"| {label} | {gt_source} | {_VARIANT_LABELS_PLAIN.get(v, v)} "
                f"| {_fmt(st.get('mean_rho'))} "
                f"| {f1_str} "
                f"| {_fmt(st.get('mean_precision'))} "
                f"| {_fmt(st.get('mean_recall'))} "
                f"| {_fmt(st.get('mean_accuracy'))} "
                f"| {_fmt(st.get('mean_rmse'))} "
                f"| {_fmt(st.get('mean_mae'))} "
                f"| {_fmt(st.get('mean_ndcg_10'))} |"
            )
            label     = ""  # Only show scenario once
            gt_source = ""
        rows.append("| | | | | | | | | | | |")

    rows += [
        "",
        "**Calibration:** All identification metrics use rank-matched binarization "
        "(top-K predicted = critical, K = #ground-truth criticals).",
        "",
        "- † = legacy fixed-threshold (0.5) binarization; not yet recalibrated",
        "- ‡ = degenerate label distribution (F1 undefined)",
        "- '— (re-train)' = checkpoint missing or recalibration failed; re-run this cell",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved ID Metrics MD: {output}")


def print_id_metrics_console(data: Dict):
    agg = data["aggregate"]
    scenarios = sorted({k.split("|")[0] for k in agg if not k.startswith("_")})

    print("\n  Identification Metrics (Critical Component Detection)")
    header = f"  {'Scenario':<25} {'Variant':<15} {'Rho':<8} {'F1':<10} {'Prec':<8} {'Rec':<8} {'Acc':<8} {'RMSE':<8} {'MAE':<8} {'NDCG':<8} Cal"
    print(header)
    print("  " + "─" * 125)

    for sc in scenarios:
        label = _SCENARIO_LABELS.get(sc, sc)
        for v in _VARIANT_ORDER:
            st   = agg.get(f"{sc}|{v}", {})
            rho  = st.get("mean_rho")
            f1   = st.get("mean_f1")
            prec = st.get("mean_precision")
            rec  = st.get("mean_recall")
            acc  = st.get("mean_accuracy")
            rmse = st.get("mean_rmse")
            mae  = st.get("mean_mae")
            ndcg = st.get("mean_ndcg_10", 0.0)
            cal  = st.get("calibration",  "rank_matched")
            marker = "" if cal == "rank_matched" else ("‡" if "degenerate" in cal else ("†" if cal == "fixed" else "?"))
            f1_s = f"{f1:.3f}" if f1 is not None else "NaN"

            print(f"  {label:<25} {_VARIANT_LABELS_PLAIN.get(v, v):<15} "
                  f"{(rho or 0.0):<8.3f} {f1_s+marker:<10} {(prec or 0.0):<8.3f} {(rec or 0.0):<8.3f} "
                  f"{(acc or 0.0):<8.3f} {(rmse or 0.0):<8.3f} {(mae or 0.0):<8.3f} {ndcg:<8.3f} {cal}")
            label = ""
        print("")


# ── Table 4: LOSO results (4 variants) ───────────────────────────────────────

def render_table4_tex(loso_data: Dict, output: Path):
    """LaTeX booktabs Table 4: LOSO per-fold ρ × variant."""
    table = loso_data.get("comparison_table", {})
    if not table:
        print("  No LOSO comparison table found.")
        return

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LOSO inductive evaluation (Leave-One-Scenario-Out), mean Spearman $\rho \pm \sigma$",
        r"         across folds and seeds. \textbf{Bold} = best per row.}",
        r"\label{tab:loso_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variant & Mean $\rho$ & Std $\rho$ & Mean F1@K & $\Delta\rho$ vs BL \\",
        r"\midrule",
    ]

    for var in _VARIANT_ORDER:
        if var not in table:
            continue
        r = table[var]
        label = _VARIANT_LABELS.get(var, var)
        mean_r = r.get("mean_rho")
        std_r  = r.get("std_rho")
        f1     = r.get("mean_f1")
        delta  = r.get("delta_vs_best_baseline")

        mean_s  = f"{mean_r:.4f}" if mean_r is not None else "—"
        std_s   = f"{std_r:.4f}" if std_r is not None else "—"
        f1_s    = f"{f1:.4f}" if f1 is not None else "—"
        delta_s = (f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}") if delta is not None else "—"

        if var == "hetero_qos":
            mean_s = rf"\textbf{{{mean_s}}}"
        lines.append(rf"{label} & {mean_s} & {std_s} & {f1_s} & {delta_s} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"  Saved LaTeX Table 4: {output}")


def render_table4_md(loso_data: Dict, output: Path):
    table = loso_data.get("comparison_table", {})
    rows = [
        "| Variant | Mean ρ | Std ρ | F1@K | Δρ vs BL |",
        "|---|---|---|---|---|",
    ]
    for var in _VARIANT_ORDER:
        if var not in table:
            continue
        r = table[var]
        label = _VARIANT_LABELS_PLAIN.get(var, var)
        mean_r = r.get("mean_rho")
        std_r  = r.get("std_rho")
        f1     = r.get("mean_f1")
        delta  = r.get("delta_vs_best_baseline")

        mean_s  = f"{mean_r:.4f}" if mean_r is not None else "—"
        std_s   = f"{std_r:.4f}" if std_r is not None else "—"
        f1_s    = f"{f1:.4f}" if f1 is not None else "—"
        delta_s = (f"+{delta:.4f}" if delta and delta > 0 else f"{delta:.4f}") if delta is not None else "—"

        if var == "hetero_qos":
            mean_s = f"**{mean_s}**"
        rows.append(f"| {label} | {mean_s} | {std_s} | {f1_s} | {delta_s} |")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved Markdown Table 4: {output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Block C/E: Render LaTeX/CSV/MD tables.")
    p.add_argument("--table3", type=Path, default=_RESULTS_DIR / "main_table.json",
                   help="Path to main_table.json (Block C output)")
    p.add_argument("--table4", type=Path, default=_RESULTS_DIR / "loso_all_variants.json",
                   help="Path to loso_all_variants.json (Block E output)")
    p.add_argument("--output-dir", type=Path, default=_RESULTS_DIR)
    p.add_argument("--tex-only", action="store_true", help="Only generate .tex files")
    p.add_argument("--no-tex", action="store_true", help="Skip .tex files")
    p.add_argument("--console", action="store_true", help="Print tables to console only")
    return p.parse_args()


def render_per_type_table_md(data: Dict, output: Path):
    agg = data["aggregate"]
    scenarios = sorted({k.split("|")[0] for k in agg if not k.startswith("_")})
    
    header = "| Scenario | Node Type | " + " | ".join([_VARIANT_LABELS_PLAIN.get(v, v) for v in _VARIANT_ORDER]) + " |"
    divider = "|---|---| " + " | ".join(["---"] * len(_VARIANT_ORDER)) + " |"
    rows = [header, divider]

    for sc in scenarios:
        label = _SCENARIO_LABELS.get(sc, sc)
        node_types = sorted({nt for v in _VARIANT_ORDER for nt in agg.get(f"{sc}|{v}", {}).get("per_node_type", {})})
        for nt in node_types:
            row = f"| {label} | {nt} |"
            for v in _VARIANT_ORDER:
                rho = agg.get(f"{sc}|{v}", {}).get("per_node_type", {}).get(nt)
                row += f" {rho:.3f} |" if rho is not None else " — |"
            rows.append(row)
            label = ""
        rows.append("| | | " + " | ".join([""] * len(_VARIANT_ORDER)) + " |")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved Markdown Table 5: {output}")


def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  Table Renderer (Blocks C + E)")

    # ── Table 3 ───────────────────────────────────────────────────────────────
    if args.table3.exists():
        print(f"\n  [Table 3] {args.table3}")
        data3 = _load_table3_data(args.table3)
        print_table3_console(data3)
        if not args.console:
            if not args.no_tex:
                render_table3_tex(data3, out / "table3_main_results.tex")
            if not args.tex_only:
                render_table3_csv(data3, out / "table3_main_results.csv")
                render_table3_md(data3,  out / "table3_main_results.md")
                render_id_metrics_md(data3, out / "table3_id_metrics.md")
                render_per_type_table_md(data3, out / "table5_per_type_metrics.md")
        
        print_id_metrics_console(data3)
    else:
        print(f"\n  [Table 3] Not found: {args.table3}")
        print("  Run: python tools/middleware26_main_table.py")

    # ── Table 4 ───────────────────────────────────────────────────────────────
    if args.table4.exists():
        print(f"\n  [Table 4] {args.table4}")
        loso_data = json.loads(args.table4.read_text())
        if not args.console:
            if not args.no_tex:
                render_table4_tex(loso_data, out / "table4_loso_results.tex")
            if not args.tex_only:
                render_table4_md(loso_data, out / "table4_loso_results.md")
        table = loso_data.get("comparison_table", {})
        print("\n  Table 4: LOSO Results")
        print(f"  {'Variant':<25} {'Mean ρ':<10} {'Std ρ':<10} {'Δρ vs BL'}")
        print("  " + "─" * 55)
        for var in _VARIANT_ORDER:
            if var not in table:
                continue
            r = table[var]
            label = _VARIANT_LABELS_PLAIN.get(var, var)
            mean_r = r.get("mean_rho")
            std_r  = r.get("std_rho")
            delta  = r.get("delta_vs_best_baseline")
            mean_s  = f"{mean_r:.4f}" if mean_r is not None else "—"
            std_s   = f"{std_r:.4f}" if std_r is not None else "—"
            delta_s = (f"+{delta:.4f}" if delta and delta > 0 else f"{delta:.4f}") if delta is not None else "—"
            print(f"  {label:<25} {mean_s:<10} {std_s:<10} {delta_s}")
    else:
        print(f"\n  [Table 4] Not found: {args.table4}")
        print("  Run: python tools/loso_all_variants.py")

    print("\n  Done. Files written to:", out)


if __name__ == "__main__":
    main()
