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
    "topology_rmav":   r"\textsc{RMAV}",
    "homo_unweighted": r"\textsc{Homo-U}",
    "homo_scalar":     r"\textsc{Homo-S}",
    "hetero_qos":      r"\textbf{Q-HGL}",
}
_VARIANT_LABELS_PLAIN = {
    "topology_rmav":   "RMAV",
    "homo_unweighted": "Homo-U",
    "homo_scalar":     "Homo-S",
    "hetero_qos":      "Q-HGL (ours)",
}
_VARIANT_ORDER = ["topology_rmav", "homo_unweighted", "homo_scalar", "hetero_qos"]

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


def render_table3_tex(data: Dict, output: Path):
    """LaTeX booktabs Table 3: Spearman ρ per scenario × variant."""
    agg = data["aggregate"]
    scenarios = sorted({k.split("|")[0] for k in agg.keys()})
    if not scenarios:
        print("  No aggregate data found in table3 input.")
        return

    n_vars = len(_VARIANT_ORDER)
    col_spec = "l" + "c" * n_vars
    header_row = " & ".join(_VARIANT_LABELS.get(v, v) for v in _VARIANT_ORDER)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Spearman $\rho$ (composite score) across 8 scenarios $\times$ 4 variants,",
        r"         5 seeds, Bootstrap 95\% CI. $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ vs Q-HGL (Wilcoxon).}",
        r"\label{tab:main_results}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Scenario & {header_row} \\",
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
            is_hq   = var == "hetero_qos"
            cell = _format_rho(mean_r, ci_lo, ci_hi, bold=(is_best and is_hq))
            if var != "hetero_qos" and p_val is not None:
                cell += _pval_star(p_val)
            cells.append(cell)

        lines.append(rf"{label} & {' & '.join(cells)} \\")

    # Summary row: cross-scenario mean
    lines.append(r"\midrule")
    avg_cells = []
    for var in _VARIANT_ORDER:
        rhos = [
            agg.get(f"{sc}|{var}", {}).get("mean_rho")
            for sc in scenarios
            if agg.get(f"{sc}|{var}", {}).get("mean_rho") is not None
        ]
        if rhos:
            import numpy as np
            m = np.mean(rhos)
            avg_cells.append(rf"\textbf{{{m:.3f}}}" if var == "hetero_qos" else f"{m:.3f}")
        else:
            avg_cells.append("—")

    lines.append(rf"\textbf{{Mean}} & {' & '.join(avg_cells)} \\")
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
    scenarios = sorted({k.split("|")[0] for k in agg})
    rows = []
    header = ["scenario"] + [f"{v}_rho" for v in _VARIANT_ORDER] + \
             [f"{v}_ci_lo" for v in _VARIANT_ORDER] + [f"{v}_ci_hi" for v in _VARIANT_ORDER]
    for sc in scenarios:
        row = {"scenario": sc}
        for v in _VARIANT_ORDER:
            st = agg.get(f"{sc}|{v}", {})
            row[f"{v}_rho"]   = st.get("mean_rho", "")
            row[f"{v}_ci_lo"] = st.get("ci_lo", "")
            row[f"{v}_ci_hi"] = st.get("ci_hi", "")
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV Table 3:    {output}")


def render_table3_md(data: Dict, output: Path):
    agg = data["aggregate"]
    scenarios = sorted({k.split("|")[0] for k in agg})
    var_labels = [_VARIANT_LABELS_PLAIN.get(v, v) for v in _VARIANT_ORDER]

    header  = "| Scenario | " + " | ".join(var_labels) + " |"
    divider = "|" + "---|" * (len(_VARIANT_ORDER) + 1)
    rows = [header, divider]

    for sc in scenarios:
        label = _SCENARIO_LABELS.get(sc, sc)
        best_rho = max(
            (agg.get(f"{sc}|{v}", {}).get("mean_rho", 0.0) or 0.0)
            for v in _VARIANT_ORDER
        )
        cells = []
        for v in _VARIANT_ORDER:
            st = agg.get(f"{sc}|{v}", {})
            r = st.get("mean_rho")
            if r is None:
                cells.append("—")
            elif abs(r - best_rho) < 0.001:
                cells.append(f"**{r:.3f}**")
            else:
                cells.append(f"{r:.3f}")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved Markdown Table 3: {output}")


def print_table3_console(data: Dict):
    agg = data["aggregate"]
    scenarios = sorted({k.split("|")[0] for k in agg})
    var_labels = [_VARIANT_LABELS_PLAIN.get(v, v) for v in _VARIANT_ORDER]
    col_w = 13

    print("\n  Table 3: Spearman ρ — Main Results")
    header = f"  {'Scenario':<30}" + "".join(f"{lbl:<{col_w}}" for lbl in var_labels)
    print(header)
    print("  " + "─" * (30 + col_w * len(_VARIANT_ORDER)))

    for sc in scenarios:
        label = _SCENARIO_LABELS.get(sc, sc)
        best_rho = max(
            (agg.get(f"{sc}|{v}", {}).get("mean_rho", 0.0) or 0.0)
            for v in _VARIANT_ORDER
        )
        row = f"  {label:<30}"
        for v in _VARIANT_ORDER:
            st = agg.get(f"{sc}|{v}", {})
            r  = st.get("mean_rho")
            if r is None:
                row += f"{'—':<{col_w}}"
            elif abs(r - best_rho) < 0.001:
                row += f"*{r:.3f}*     "[:col_w].ljust(col_w)
            else:
                row += f"{r:.3f}".ljust(col_w)
        print(row)


# ── Identification Metrics (F1, Prec, Rec, Top-5) ───────────────────────────

def render_id_metrics_md(data: Dict, output: Path):
    agg = data.get("aggregate", {})
    scenarios = sorted({k.split("|")[0] for k in agg})
    
    header = "| Scenario | Variant | F1 | Precision | Recall | Top-5 | Top-10 |"
    divider = "|---|---|---|---|---|---|---|"
    rows = [header, divider]
    
    for sc in scenarios:
        label = _SCENARIO_LABELS.get(sc, sc)
        for v in _VARIANT_ORDER:
            st = agg.get(f"{sc}|{v}", {})
            f1   = st.get("mean_f1", "—")
            prec = st.get("mean_precision", "—")
            rec  = st.get("mean_recall", "—")
            t5   = st.get("mean_top5", "—")
            t10  = st.get("mean_top10", "—")
            
            f1_s   = f"{f1:.3f}" if isinstance(f1, float) else f1
            prec_s = f"{prec:.3f}" if isinstance(prec, float) else prec
            rec_s  = f"{rec:.3f}" if isinstance(rec, float) else rec
            t5_s   = f"{t5:.3f}" if isinstance(t5, float) else t5
            t10_s  = f"{t10:.3f}" if isinstance(t10, float) else t10
            
            v_label = _VARIANT_LABELS_PLAIN.get(v, v)
            rows.append(f"| {label} | {v_label} | {f1_s} | {prec_s} | {rec_s} | {t5_s} | {t10_s} |")
        rows.append("| | | | | | | |") # spacer
        
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved ID Metrics MD: {output}")


def print_id_metrics_console(data: Dict):
    agg = data.get("aggregate", {})
    scenarios = sorted({k.split("|")[0] for k in agg})
    
    print("\n  Identification Metrics (Critical Component Detection)")
    header = f"  {'Scenario':<25} {'Variant':<15} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Top-5':<8} {'Top-10'}"
    print(header)
    print("  " + "─" * 85)
    
    for sc in scenarios:
        label = _SCENARIO_LABELS.get(sc, sc)
        for v in _VARIANT_ORDER:
            st = agg.get(f"{sc}|{v}", {})
            f1   = st.get("mean_f1", 0.0)
            prec = st.get("mean_precision", 0.0)
            rec  = st.get("mean_recall", 0.0)
            t5   = st.get("mean_top5", 0.0)
            t10  = st.get("mean_top10", 0.0)
            
            v_label = _VARIANT_LABELS_PLAIN.get(v, v)
            print(f"  {label[:25]:<25} {v_label:<15} {f1:<8.3f} {prec:<8.3f} {rec:<8.3f} {t5:<8.3f} {t10:<8.3f}")
        print()


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
