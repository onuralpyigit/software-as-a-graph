#!/usr/bin/env python3
"""
reproduce/render_table.py — Block C: Table 3 + Table 4 LaTeX/CSV/Markdown renderer
===============================================================================

Reads results/main_table.json (Block C) and/or results/loso_all_variants.json
(Block E) and produces:

  - results/table3_main_results.tex   (LaTeX booktabs table; feeds JSS Table 6 & 7 --
                                        In-Distribution Held-Out Ranking Performance & Paired Wilcoxon)
  - results/table4_loso_results.tex   (LaTeX booktabs table; feeds JSS Table 8 --
                                        Leave-One-Scenario-Out Cross-Validation Results)
  - results/table3_main_results.csv   (CSV for Excel/R)
  - results/table3_main_results.md    (Markdown for README / GitHub)

Usage
-----
  python reproduce/render_table.py
  python reproduce/render_table.py --table3 results/main_table.json
  python reproduce/render_table.py --table4 results/loso_all_variants.json --tex-only
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

from saag.evaluation import variant_registry as _registry

# ── Display config ─────────────────────────────────────────────────────────────

_SCENARIO_LABELS = {
    "atm_system":             "ATM System",
    "av_system":              "AV System",
    "iot_smart_city_system":  "IoT Smart City",
    "financial_trading_system": "Financial Trading",
    "healthcare_system":      "Healthcare",
    "hub_and_spoke_system":   "Enterprise Integration (ESB)",
    "microservices_system":   "Microservices",
    "enterprise_system":      "Enterprise",
    "industrial_scada_system": "Industrial SCADA",
    "logistics_fleet_system": "Logistics Fleet",
    "realtime_gaming_system": "Real-Time Gaming",
    "telecom_ran_system":     "Telecom RAN",
}

# Display labels and family-grouped ordering come from the registry, so this
# renderer, the figure scripts and the LOSO/k-fold harnesses cannot print two
# different names for one variant. See saag/evaluation/variant_registry.py.
#
# Two label sets, because gl/gl_qos do not denote one substrate: this file's
# main table is in-distribution (projection), while the LOSO tables below report
# the same ids run on the native graph (GAT-N / GAT-N-QoS).
_IN_DIST_VARIANTS = _registry.order(
    include=["topo_baseline", "topo_qos", "gl", "gl_qos", "hgl", "hgl_qos"]
)

#: The same table with the native-substrate homogeneous controls in the GAT
#: slots. The manuscript's Table 5 is this one: the RQ2 confound work replaced
#: the projection-substrate gl/gl_qos with gl_full/gl_full_qos so that every
#: learned variant consumes the identical native multigraph, and the registry
#: reports that pair as GAT-N / GAT-N-QoS. results/main_table_v3.json -- the
#: artifact the manuscript is reconciled against -- carries this set.
_IN_DIST_VARIANTS_NATIVE = _registry.order(
    include=["topo_baseline", "topo_qos", "gl_full", "gl_full_qos", "hgl", "hgl_qos"]
)

_VARIANT_LABELS = {v: _registry.label(v, latex=True) for v in _IN_DIST_VARIANTS}
_VARIANT_LABELS_PLAIN = {v: _registry.label(v) for v in _IN_DIST_VARIANTS}
_VARIANT_ORDER = list(_IN_DIST_VARIANTS)


def _bind_in_dist_variants(data: Dict) -> None:
    """Point the module's in-distribution column set at what the artifact ran.

    Rendering a fixed variant list against an artifact that ran a different one
    silently produces a table of em-dashes and a column mean of 0.000 -- which
    is what results/table3_main_results.* held: every GAT cell blank because the
    renderer looked for gl/gl_qos while main_table_v3.json carries
    gl_full/gl_full_qos. A blank column reads as "this variant scored nothing",
    not as "this renderer was asking the wrong question".
    """
    global _VARIANT_ORDER, _VARIANT_LABELS, _VARIANT_LABELS_PLAIN
    present = {k.split("|")[1] for k in (data.get("aggregate") or {})
               if not k.startswith("_") and "|" in k}
    order = list(_IN_DIST_VARIANTS)
    if present and not ({"gl", "gl_qos"} & present) and ({"gl_full", "gl_full_qos"} & present):
        order = list(_IN_DIST_VARIANTS_NATIVE)
    _VARIANT_ORDER = order
    _VARIANT_LABELS = {v: _registry.label(v, latex=True) for v in order}
    _VARIANT_LABELS_PLAIN = {v: _registry.label(v) for v in order}
# Table 3 has no `topology_rm` cells (it is only computed by the LOSO/k-fold
# harnesses), so it stays out of _VARIANT_ORDER — adding it there would render
# an empty column. LOSO's comparison_table does carry it (see
# reproduce/loso_all_variants.py's ALL_VARIANTS), and previously it was
# silently dropped by every LOSO renderer because they all iterated
# _VARIANT_ORDER instead.
_LOSO_VARIANT_ORDER = _registry.order(
    include=["topo_baseline", "topo_qos", "topology_rm", "gl", "gl_qos", "hgl", "hgl_qos"]
)
# LOSO and k-fold both run gl/gl_qos on the native graph, so both report them as
# GAT-N / GAT-N-QoS. One map serves the renderers of either harness.
_NATIVE_VARIANT_LABELS = {
    v: _registry.label(v, harness="loso", latex=True) for v in _LOSO_VARIANT_ORDER
}
_NATIVE_VARIANT_LABELS_PLAIN = {
    v: _registry.label(v, harness="loso") for v in _LOSO_VARIANT_ORDER
}

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
    _bind_in_dist_variants(data)
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
        r"         5 seeds, Bootstrap 95\% CI. $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ vs HGT-QoS (Wilcoxon).}",
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
            cell = _format_rho(mean_r, ci_lo, ci_hi, bold=(is_best and var == "hgl_qos"))
            if var != "hgl_qos" and p_val is not None:
                cell += _pval_star(p_val)
            cells.append(cell)

        # Delta column: hgl_qos - hgl
        stats_qos = _get_cell_stats(agg, sc, "hgl_qos")
        stats_none = _get_cell_stats(agg, sc, "hgl")
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
        if abs(m - best_avg) < 0.001 and var == "hgl_qos":
            cell = rf"\textbf{{{cell}}}"
        avg_cells.append(cell)

    avg_delta = all_means[-1] - all_means[-2] # hgl_qos - hgl
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
    _bind_in_dist_variants(data)
    agg = data["aggregate"]
    scenarios = sorted({k.split("|")[0] for k in agg if not k.startswith("_")})
    rows = []
    header = ["scenario", "gt_source"] + [f"{v}_rho" for v in _VARIANT_ORDER] + \
             [f"{v}_ci_lo" for v in _VARIANT_ORDER] + [f"{v}_ci_hi" for v in _VARIANT_ORDER] + \
             [f"{v}_pval" for v in _VARIANT_ORDER if v != "hgl_qos"]
    for sc in scenarios:
        gt_source = agg.get(f"{sc}|topo_baseline", {}).get("gt_source", "Sim")
        row = {"scenario": sc, "gt_source": gt_source}
        for v in _VARIANT_ORDER:
            st = agg.get(f"{sc}|{v}", {})
            row[f"{v}_rho"]   = st.get("mean_rho", "")
            row[f"{v}_ci_lo"] = st.get("ci_lo", "")
            row[f"{v}_ci_hi"] = st.get("ci_hi", "")
            if v != "hgl_qos":
                row[f"{v}_pval"] = st.get("wilcoxon_p_vs_hetero", "")
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV Table 3:    {output}")



def render_table3_md(data: Dict, output: Path):
    _bind_in_dist_variants(data)
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
                if abs(r - best_rho) < 0.001 and var == "hgl_qos":
                    cell = f"**{cell}**"
                if var != "hgl_qos" and p is not None:
                    if p < 0.001: cell += "***"
                    elif p < 0.01: cell += "**"
                    elif p < 0.05: cell += "*"
            cells.append(cell)
            
        # Delta
        r_qos = agg.get(f"{sc}|hgl_qos", {}).get("mean_rho")
        r_none = agg.get(f"{sc}|hgl", {}).get("mean_rho")
        p_delta = agg.get(f"{sc}|hgl", {}).get("wilcoxon_p_vs_hetero")
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
        if abs(m - best_avg) < 0.001 and var == "hgl_qos":
            cell = f"**{cell}**"
        avg_cells.append(cell)
    
    avg_delta = all_means[-1] - all_means[-2]
    avg_cells.append(f"{'+' if avg_delta >= 0 else ''}{avg_delta:.3f}")
    rows.append("| " + " | ".join(avg_cells) + " |")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved Markdown Table 3: {output}")


def _per_type_cell(entry, ndigits: int = 3) -> str:
    """Render one per-node-type cell.

    The aggregator now emits ``{"rho": float | "undefined", "n_nodes": int, ...}``
    rather than a bare float, because a stratum whose labels are constant (Topic
    and Node carry no ground truth) has an undefined correlation, not one of
    0.0. Printing "und." keeps that distinction visible in the table instead of
    letting a coverage gap read as a measured failure.
    """
    if entry is None:
        return "—"
    value = entry.get("rho", entry.get("mean")) if isinstance(entry, dict) else entry
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{ndigits}f}"
    return "und."


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
                if abs(r - best_rho) < 0.001 and v == "hgl_qos":
                    cell = f"*{cell}*"
            
            if v != "hgl_qos" and p_val is not None:
                if p_val < 0.001: cell += "***"
                elif p_val < 0.01: cell += "**"
                elif p_val < 0.05: cell += "*"
                
            row += cell.ljust(col_w)
        
        # Delta
        r_qos = agg.get(f"{sc}|hgl_qos", {}).get("mean_rho")
        r_none = agg.get(f"{sc}|hgl", {}).get("mean_rho")
        p_delta = agg.get(f"{sc}|hgl", {}).get("wilcoxon_p_vs_hetero")
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
                cell = _per_type_cell(
                    agg.get(f"{sc}|{v}", {}).get("per_node_type", {}).get(nt)
                )
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

    header  = "| Scenario | GT | Variant | Spearman ρ | F1 | Accuracy | RMSE | MAE | NDCG@10 |"
    divider = "|---|---|---|---|---|---|---|---|---|"
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
                f"| {_fmt(st.get('mean_accuracy'))} "
                f"| {_fmt(st.get('mean_rmse'))} "
                f"| {_fmt(st.get('mean_mae'))} "
                f"| {_fmt(st.get('mean_ndcg_10'))} |"
            )
            label     = ""  # Only show scenario once
            gt_source = ""
        rows.append("| | | | | | | | |")

    rows += [
        "",
        "**Calibration:** The `F1` column is `f1_at_k` — the top-K predicted set "
        "against the top-K true set, K = round(0.20 n). Both sets have exactly K "
        "members, so precision, recall and F1 are identically equal here and the "
        "column measures rank overlap, not identification. K is *not* the number "
        "of ground-truth criticals. For F1 proper see "
        "results/table3_identification_metrics.md.",
        "",
        "- † = legacy fixed-threshold (0.5) binarization; not yet recalibrated",
        "- ‡ = degenerate label distribution (F1 undefined)",
        "- '— (re-train)' = checkpoint missing or recalibration failed; re-run this cell",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved ID Metrics MD: {output}")


def render_identification_md(data: Dict, output: Path):
    """Per-scenario identification quality, with a non-degenerate F1.

    Separate from :func:`render_id_metrics_md` because the ``F1`` column there
    is ``f1_at_k`` — top-K predicted against top-K true, both sets of size
    K = round(0.20 n). Those cardinalities are equal by construction, so
    precision, recall and F1 are the same number and none of them is an F1
    score; ``overlap@K`` is what it measures. This table reports the three
    operating points where identification is actually testable:

    ``F1@τ``       top-K prediction vs. the labels' own critical set
                   (I*(v) >= 0.5 max I*). Precision and recall diverge, but
                   both are capped by the mismatch between K and the size of
                   that set, so a perfect ranking does not score 1.0.
    ``F1@τ̂``      the same relative cut applied to *both* vectors. The
                   predicted set size floats, the caps disappear, and a perfect
                   ranking scores 1.0. This is the operating point a user of
                   the tool would actually see.
    ``F1max``      the best F1 any cut of the ranking could reach, next to
                   ``F1₊``, the F1 of calling everything critical — the floor
                   every ranking clears for free.
    """
    agg = data["aggregate"]
    scenarios = sorted({k.split("|")[0] for k in agg if not k.startswith("_")})

    header = ("| Scenario | Variant | ρ | overlap@K | F1@τ | P@τ̂ | R@τ̂ | F1@τ̂ "
              "| F1max | F1₊ | PR-AUC | n | #crit |")
    rows = [header, "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def _f(x):
        return "—" if not isinstance(x, (int, float)) else f"{x:.3f}"

    for sc in scenarios:
        label = _SCENARIO_LABELS.get(sc, sc)
        for v in _VARIANT_ORDER:
            st = agg.get(f"{sc}|{v}", {})
            if not st:
                continue
            rows.append(
                f"| {label} | {_VARIANT_LABELS_PLAIN.get(v, v)} "
                f"| {_f(st.get('mean_rho'))} | {_f(st.get('mean_f1'))} "
                f"| {_f(st.get('mean_f1_at_tau'))} "
                f"| {_f(st.get('mean_precision_at_threshold'))} "
                f"| {_f(st.get('mean_recall_at_threshold'))} "
                f"| {_f(st.get('mean_f1_at_threshold'))} "
                f"| {_f(st.get('mean_f1_max'))} | {_f(st.get('mean_f1_all_positive'))} "
                f"| {_f(st.get('mean_pr_auc'))} "
                f"| {st.get('n_evaluated', '—')} | {st.get('n_true_critical', '—')} |"
            )
            label = ""
        rows.append("| | | | | | | | | | | | | |")

    # Cohort means, per variant, over the scenarios where each figure is defined.
    rows += ["", "**Mean across scenarios**", "",
             "| Variant | ρ | overlap@K | F1@τ | F1@τ̂ | F1max | F1₊ | PR-AUC |",
             "|---|---|---|---|---|---|---|---|"]
    for v in _VARIANT_ORDER:
        cells = [agg[f"{sc}|{v}"] for sc in scenarios if f"{sc}|{v}" in agg]

        def _mean(key):
            vals = [c.get(key) for c in cells]
            vals = [x for x in vals if isinstance(x, (int, float))]
            return sum(vals) / len(vals) if vals else None

        rows.append(
            f"| {_VARIANT_LABELS_PLAIN.get(v, v)} | {_f(_mean('mean_rho'))} "
            f"| {_f(_mean('mean_f1'))} | {_f(_mean('mean_f1_at_tau'))} "
            f"| {_f(_mean('mean_f1_at_threshold'))} | {_f(_mean('mean_f1_max'))} "
            f"| {_f(_mean('mean_f1_all_positive'))} | {_f(_mean('mean_pr_auc'))} |"
        )

    rows += [
        "",
        "Five seeds per cell; every figure is the mean over the seeds on which it "
        "was defined. `n` is the held-out Application population, `#crit` the "
        "number of components the oracle marks critical in it.",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved identification metrics MD: {output}")


def render_realworld_identification_md(data: Dict, output: Path):
    """The same identification family for the five transcribed real systems.

    Zero-shot: the learned variant is trained on the synthetic corpus only and
    never sees these graphs. The training-free references (RM, Topo, Topo-QoS)
    are scored against the same oracle, population and node set, so the columns
    are comparable down the page.
    """
    per_system = data.get("per_system", {})
    references = data.get("references", {})
    learned = data.get("label", data.get("variant", "learned"))

    header = ("| System | Predictor | ρ | overlap@K | F1@τ | P@τ̂ | R@τ̂ | F1@τ̂ "
              "| F1max | F1₊ | PR-AUC | n | #crit |")
    rows = [header, "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def _f(x):
        return "—" if not isinstance(x, (int, float)) else f"{x:.3f}"

    def _i(x):
        return "—" if not isinstance(x, (int, float)) else f"{int(x)}"

    for sid in sorted(per_system):
        st = per_system[sid]
        if not st.get("n_seeds"):
            continue
        label = sid.replace("realworld_", "")
        rows.append(
            f"| {label} | {learned} | {_f(st.get('mean_rho'))} "
            f"| {_f(st.get('mean_f1_at_k'))} | {_f(st.get('mean_f1_at_tau'))} "
            f"| {_f(st.get('mean_precision_at_threshold'))} "
            f"| {_f(st.get('mean_recall_at_threshold'))} "
            f"| {_f(st.get('mean_f1_at_threshold'))} "
            f"| {_f(st.get('mean_f1_max'))} | {_f(st.get('mean_f1_all_positive'))} "
            f"| {_f(st.get('mean_pr_auc'))} "
            f"| {st.get('n_evaluated', '—')} | {st.get('n_true_critical', '—')} |"
        )
        for name in sorted(references):
            ref = references[name].get(sid, {})
            if "rho" not in ref:
                # Reference recorded why it could not be computed; say so rather
                # than print a dash that reads like a missing run.
                rows.append(f"| | {name} | _{ref.get('unavailable', 'unavailable')}_ "
                            + "| " * 10 + "|")
                continue
            rows.append(
                f"| | {name} | {_f(ref.get('rho'))} | {_f(ref.get('f1_at_k'))} "
                f"| {_f(ref.get('f1_at_tau'))} | {_f(ref.get('precision_at_threshold'))} "
                f"| {_f(ref.get('recall_at_threshold'))} | {_f(ref.get('f1_at_threshold'))} "
                f"| {_f(ref.get('f1_max'))} | {_f(ref.get('f1_all_positive'))} "
                f"| {_f(ref.get('pr_auc'))} | {st.get('n_evaluated', '—')} "
                f"| {_i(ref.get('n_true_critical'))} |"
            )
        rows.append("| | | | | | | | | | | | | |")

    rows += [
        "",
        f"Oracle: {data.get('oracle', 'I*(v)')}. Population: "
        f"{data.get('eval_population', 'application')}. "
        f"Seeds: {data.get('seeds', [])} (references are training-free and "
        "deterministic, so they carry no seed variance).",
        "",
        "Column definitions are the same as results/table3_identification_metrics.md: "
        "`overlap@K` is not an F1 (precision == recall == F1 by construction); "
        "`F1@τ̂` is the non-degenerate one.",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved real-world identification metrics MD: {output}")


def print_id_metrics_console(data: Dict):
    agg = data["aggregate"]
    scenarios = sorted({k.split("|")[0] for k in agg if not k.startswith("_")})

    print("\n  Identification Metrics (Critical Component Detection)")
    header = f"  {'Scenario':<25} {'Variant':<15} {'Rho':<8} {'F1':<10} {'Acc':<8} {'RMSE':<8} {'MAE':<8} {'NDCG':<8} Cal"
    print(header)
    print("  " + "─" * 105)

    for sc in scenarios:
        label = _SCENARIO_LABELS.get(sc, sc)
        for v in _VARIANT_ORDER:
            st   = agg.get(f"{sc}|{v}", {})
            rho  = st.get("mean_rho")
            f1   = st.get("mean_f1")
            acc  = st.get("mean_accuracy")
            rmse = st.get("mean_rmse")
            mae  = st.get("mean_mae")
            ndcg = st.get("mean_ndcg_10")
            cal  = st.get("calibration",  "rank_matched")
            marker = "" if cal == "rank_matched" else ("‡" if "degenerate" in cal else ("†" if cal == "fixed" else "?"))
            f1_s = f"{f1:.3f}" if f1 is not None else "NaN"

            print(f"  {label:<25} {_VARIANT_LABELS_PLAIN.get(v, v):<15} "
                  f"{(rho or 0.0):<8.3f} {f1_s+marker:<10} "
                  f"{(acc or 0.0):<8.3f} {(rmse or 0.0):<8.3f} {(mae or 0.0):<8.3f} {(ndcg or 0.0):<8.3f} {cal}")
            label = ""
        print("")


# ── Table 4: LOSO results (4 variants) ───────────────────────────────────────

#: The comparator every Δρ is measured against, from the registry that owns it.
_LOSO_DELTA_BASELINE = _registry.PREREGISTERED_BASELINE


def _loso_contrasts(sig_data: Optional[Dict]) -> Dict[str, Dict]:
    """Per-variant Δρ against the pre-registered baseline, with its interval.

    The interval is the point of this. Table 4 used to print a bare "+0.0346"
    under the heading "Δρ vs best baseline", where "best baseline" was `max`
    over every other row and had selected GAT-N-QoS — a *learned* variant. The
    pre-registered contrast is against Topo-QoS, is +0.0851, and its 95% CI
    includes zero. The bare number under the wrong comparator read as a positive
    result that the abstract explicitly declines to claim.

    The statistics stay in loso_significance.py; this only reads them, and
    returns {} when that artifact is absent so the column degrades to the
    artifact's own paired Δ rather than disappearing.
    """
    if not sig_data:
        return {}
    out: Dict[str, Dict] = {}
    for section in ("preregistered", "exploratory"):
        for row in sig_data.get(section) or []:
            if row.get("baseline") != _LOSO_DELTA_BASELINE:
                continue
            out[row["variant"]] = {
                "mean_delta": row.get("mean_delta"),
                "ci95": row.get("delta_ci95"),
                "role": row.get("role", section),
            }
    return out


def _delta_cell(var: str, row: Dict, contrasts: Dict[str, Dict], latex: bool) -> str:
    """One Δρ cell: the paired mean, and its CI when the significance artifact has one."""
    if var == _LOSO_DELTA_BASELINE:
        return "—"
    contrast = contrasts.get(var, {})
    delta = contrast.get("mean_delta")
    if delta is None:
        delta = row.get("delta_vs_baseline")
    if delta is None:
        return "—"
    cell = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
    ci = contrast.get("ci95")
    if ci and len(ci) == 2:
        lo, hi = ci
        cell += (rf" $[{lo:+.3f}, {hi:+.3f}]$" if latex else f" [{lo:+.3f}, {hi:+.3f}]")
    return cell


def _kfold_best_variant(table: Dict) -> Optional[str]:
    """The variant that actually wins on mean rho.

    The k-fold captions promise "best per row" while both renderers bolded
    ``hgl_qos`` unconditionally, so a table in which a baseline outscored the
    proposed model would have said the opposite of what it showed. Table 4's
    renderer already picks the winner rather than assuming it.
    """
    candidates = [v for v in _VARIANT_ORDER
                  if v in table and table[v].get("mean_rho") is not None]
    return max(candidates, key=lambda v: table[v]["mean_rho"], default=None)


def _kfold_delta_cell(var: str, row: Dict, latex: bool) -> str:
    """One k-fold Δρ cell: the paired mean against the pre-registered baseline.

    The k-fold harness pairs by scenario and stores the interval alongside the
    mean (``_paired_deltas``), so unlike Table 4 this cell needs no separate
    significance artifact to show one.
    """
    if var == _LOSO_DELTA_BASELINE:
        return "—"
    delta = row.get("delta_vs_baseline")
    if delta is None:
        return "—"
    cell = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
    ci = row.get("delta_vs_baseline_ci95")
    if ci and len(ci) == 2 and all(c is not None for c in ci):
        lo, hi = ci
        cell += (rf" $[{lo:+.3f}, {hi:+.3f}]$" if latex else f" [{lo:+.3f}, {hi:+.3f}]")
    return cell


def render_table4_tex(loso_data: Dict, output: Path, sig_data: Optional[Dict] = None):
    """LaTeX booktabs Table 4: LOSO per-fold ρ × variant."""
    table = loso_data.get("comparison_table", {})
    if not table:
        print("  No LOSO comparison table found.")
        return

    # Bold the variant that actually wins on mean_rho, not whichever variant
    # happens to be the proposed method — the table caption promises "best per
    # row", and a hardcoded winner silently lies when a baseline outperforms it.
    best_var = max(
        (v for v in _LOSO_VARIANT_ORDER if v in table and table[v].get("mean_rho") is not None),
        key=lambda v: table[v]["mean_rho"],
        default=None,
    )

    # Name the population in the caption: the same variant scored on a pooled
    # node set is a different measurement, and a caption that omits which one
    # was used makes two incomparable tables look comparable.
    population = next(
        (table[v].get("eval_population") for v in _LOSO_VARIANT_ORDER
         if v in table and table[v].get("eval_population")), None)
    pop_note = (rf" Scored on the \texttt{{{population}}} node population."
                if population else "")

    contrasts = _loso_contrasts(sig_data)
    baseline_label = _NATIVE_VARIANT_LABELS.get(_LOSO_DELTA_BASELINE, _LOSO_DELTA_BASELINE)
    ci_note = (" 95\\% CIs are bootstrap intervals on the paired per-fold difference"
               " (reproduce/loso\\_significance.py)." if contrasts else "")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LOSO inductive evaluation (Leave-One-Scenario-Out), mean Spearman $\rho \pm \sigma$",
        rf"         across folds and seeds.{pop_note} \textbf{{Bold}} = best per row.",
        rf"         $\Delta\rho$ is paired by fold against {baseline_label},",
        rf"         the pre-registered comparator.{ci_note}}}",
        r"\label{tab:loso_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        rf"Variant & Mean $\rho$ & Std $\rho$ & Mean F1@K & $\Delta\rho$ vs {baseline_label} (95\% CI) \\",
        r"\midrule",
    ]

    for var in _LOSO_VARIANT_ORDER:
        if var not in table:
            continue
        r = table[var]
        label = _NATIVE_VARIANT_LABELS.get(var, var)
        mean_r = r.get("mean_rho")
        std_r  = r.get("std_rho")
        f1     = r.get("mean_f1")

        mean_s  = f"{mean_r:.4f}" if mean_r is not None else "—"
        std_s   = f"{std_r:.4f}" if std_r is not None else "—"
        f1_s    = f"{f1:.4f}" if f1 is not None else "—"
        delta_s = _delta_cell(var, r, contrasts, latex=True)

        if var == best_var:
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


def render_table4_md(loso_data: Dict, output: Path, sig_data: Optional[Dict] = None):
    table = loso_data.get("comparison_table", {})
    best_var = max(
        (v for v in _LOSO_VARIANT_ORDER if v in table and table[v].get("mean_rho") is not None),
        key=lambda v: table[v]["mean_rho"],
        default=None,
    )
    contrasts = _loso_contrasts(sig_data)
    baseline_label = _NATIVE_VARIANT_LABELS_PLAIN.get(_LOSO_DELTA_BASELINE, _LOSO_DELTA_BASELINE)
    rows = [
        f"| Variant | Mean ρ | Std ρ | F1@K | Δρ vs {baseline_label} (95% CI) |",
        "|---|---|---|---|---|",
    ]
    for var in _LOSO_VARIANT_ORDER:
        if var not in table:
            continue
        r = table[var]
        label = _NATIVE_VARIANT_LABELS_PLAIN.get(var, var)
        mean_r = r.get("mean_rho")
        std_r  = r.get("std_rho")
        f1     = r.get("mean_f1")

        mean_s  = f"{mean_r:.4f}" if mean_r is not None else "—"
        std_s   = f"{std_r:.4f}" if std_r is not None else "—"
        f1_s    = f"{f1:.4f}" if f1 is not None else "—"
        delta_s = _delta_cell(var, r, contrasts, latex=False)

        if var == best_var:
            mean_s = f"**{mean_s}**"
        rows.append(f"| {label} | {mean_s} | {std_s} | {f1_s} | {delta_s} |")

    rows.append("")
    rows.append(f"Δρ is paired by fold against {baseline_label}, the pre-registered comparator "
                "(PREREGISTRATION.md). Intervals are bootstrap 95% CIs on the per-fold "
                "difference, from reproduce/loso_significance.py; one that spans zero means "
                "the contrast is not resolved at this fold count."
                if contrasts else
                f"Δρ is paired by fold against {baseline_label}, the pre-registered comparator. "
                "No significance artifact was supplied, so no intervals are shown — pass "
                "--significance to include them.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved Markdown Table 4: {output}")



# ── Table: RQ2 confound controls ─────────────────────────────────────────────
# A separate table rather than extra columns on Table 4, because these arms are
# not manuscript columns and because the column that matters here -- parameter
# count -- has no place in Table 4. The absence of that column is precisely
# what let the capacity confound survive internal review, so it is mandatory
# here: a controls table without it does not answer the question it was built
# to answer.

#: The RQ2 comparison order: the proposed model first, then the arm it was
#: originally compared against, then one control per confound.
_CONTROL_TABLE_ORDER = [
    "hgl_qos", "gl_full_qos", "gl_full_qos_cap", "gl_full_qos16_cap",
    "hgl_qos_uni", "hgl", "gl_full_cap",
]


def _variant_param_count(variant_id: str) -> Optional[int]:
    """Parameters this variant trains with, or None if it cannot be built here.

    Counted against the LOSO primary graph's relation set, which is what
    tests/test_baselines.py::TestControlArmCapacityParity pins.
    """
    try:
        from saag.prediction.models.baselines import build_baseline
        from saag.prediction.models.core import NodeCriticalityGNN
        from saag.prediction.data_preparation import NODE_TYPE_TO_DIM
    except Exception:
        return None

    node_types = ["Application", "Library", "Broker", "Topic", "Node"]
    relations = [
        ("Application", "SUBSCRIBES_TO", "Topic"), ("Application", "RUNS_ON", "Node"),
        ("Application", "USES", "Library"), ("Application", "PUBLISHES_TO", "Topic"),
        ("Broker", "ROUTES", "Topic"), ("Broker", "RUNS_ON", "Node"),
        ("Node", "CONNECTS_TO", "Node"), ("Library", "PUBLISHES_TO", "Topic"),
        ("Library", "SUBSCRIBES_TO", "Topic"), ("Library", "USES", "Library"),
    ]
    spec = _registry.VARIANTS.get(variant_id)
    if spec is None or spec.substrate == "none":
        return None
    try:
        if spec.family == "heterogeneous" or variant_id.startswith("hgl"):
            model = NodeCriticalityGNN(
                (node_types, relations),
                use_bidirectional=_registry.bidirectional_for(variant_id),
            )
        else:
            edge_dim = _registry.edge_dim(variant_id, "loso")
            model = build_baseline(
                "homo_unweighted" if edge_dim is None else "homo_scalar",
                node_type_dims=NODE_TYPE_TO_DIM,
                hidden_channels=_registry.hidden_for(variant_id, 64, "loso"),
                edge_dim=edge_dim,
            )
        return sum(q.numel() for q in model.parameters())
    except Exception:
        return None


def render_rq2_controls_md(loso_data: Dict, output: Path):
    """Section 7.2.1 controls table: one row per arm, with its parameter count."""
    table = loso_data.get("comparison_table", {})
    present = [v for v in _CONTROL_TABLE_ORDER if v in table]
    if not present:
        print("  [RQ2 controls] no control arms in this artifact; skipping")
        return

    rows = [
        "| Arm | Controls for | Params | Mean ρ | Fold σ | F1@K |",
        "|---|---|---|---|---|---|",
    ]
    def _fmt(value, spec_fmt):
        return format(value, spec_fmt) if value is not None else "—"

    for var in present:
        r = table[var]
        cells = [
            _registry.label(var, harness="loso"),
            _registry.VARIANTS[var].control_for or "—",
            _fmt(_variant_param_count(var), ","),
            _fmt(r.get("mean_rho"), ".4f"),
            _fmt(r.get("std_rho"), ".4f"),
            _fmt(r.get("mean_f1"), ".4f"),
        ]
        rows.append("| " + " | ".join(cells) + " |")

    missing = [v for v in _CONTROL_TABLE_ORDER if v not in table]
    if missing:
        rows.append("")
        rows.append(
            "Arms not present in this artifact: "
            + ", ".join(_registry.label(v, harness="loso") for v in missing)
            + ". Absent arms were not run; none was dropped for its result "
            "(PREREGISTRATION.md, Amendment 2)."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved RQ2 controls table: {output}")


# ── Table: per-domain k-fold results (5 variants) ────────────────────────────

def render_table4kfold_tex(kfold_data: Dict, output: Path):
    """LaTeX booktabs table: per-domain k-fold in-domain ρ × variant."""
    table = kfold_data.get("comparison_table", {})
    if not table:
        print("  No k-fold comparison table found.")
        return

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-domain k-fold in-domain evaluation, mean Spearman $\rho \pm \sigma$",
        r"         across scenarios and seeds. \textbf{Bold} = best per row.}",
        r"\label{tab:kfold_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variant & Mean $\rho$ & Std $\rho$ & Mean F1@K & $\Delta\rho$ vs \texttt{Topo-QoS} (95\% CI) \\",
        r"\midrule",
    ]

    for var in _VARIANT_ORDER:
        if var not in table:
            continue
        r = table[var]
        label = _NATIVE_VARIANT_LABELS.get(var, var)
        mean_r = r.get("mean_rho")
        std_r  = r.get("std_rho")
        f1     = r.get("mean_f1")

        mean_s  = f"{mean_r:.4f}" if mean_r is not None else "—"
        std_s   = f"{std_r:.4f}" if std_r is not None else "—"
        f1_s    = f"{f1:.4f}" if f1 is not None else "—"
        delta_s = _kfold_delta_cell(var, r, latex=True)

        if var == _kfold_best_variant(table):
            mean_s = rf"\textbf{{{mean_s}}}"
        lines.append(rf"{label} & {mean_s} & {std_s} & {f1_s} & {delta_s} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"  Saved LaTeX k-fold table: {output}")


def render_table4kfold_md(kfold_data: Dict, output: Path):
    table = kfold_data.get("comparison_table", {})
    rows = [
        f"| Variant | Mean ρ | Std ρ | F1@K | Δρ vs {_NATIVE_VARIANT_LABELS_PLAIN.get(_LOSO_DELTA_BASELINE, _LOSO_DELTA_BASELINE)} (95% CI) |",
        "|---|---|---|---|---|",
    ]
    for var in _VARIANT_ORDER:
        if var not in table:
            continue
        r = table[var]
        label = _NATIVE_VARIANT_LABELS_PLAIN.get(var, var)
        mean_r = r.get("mean_rho")
        std_r  = r.get("std_rho")
        f1     = r.get("mean_f1")

        mean_s  = f"{mean_r:.4f}" if mean_r is not None else "—"
        std_s   = f"{std_r:.4f}" if std_r is not None else "—"
        f1_s    = f"{f1:.4f}" if f1 is not None else "—"
        delta_s = _kfold_delta_cell(var, r, latex=False)

        if var == _kfold_best_variant(table):
            mean_s = f"**{mean_s}**"
        rows.append(f"| {label} | {mean_s} | {std_s} | {f1_s} | {delta_s} |")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved Markdown k-fold table: {output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Block C/E: Render LaTeX/CSV/MD tables.")
    p.add_argument("--table3", type=Path, default=_RESULTS_DIR / "main_table.json",
                   help="Path to main_table.json (Block C output)")
    p.add_argument("--table-controls", type=Path, default=None,
                   help="LOSO artifact to render the Section 7.2.1 RQ2 controls "
                        "table from (results/table_rq2_controls.md). Skipped "
                        "when the artifact carries no control arms.")
    p.add_argument("--table4", type=Path, default=_RESULTS_DIR / "loso_all_variants_v4.json",
                   help="Path to loso_all_variants.json (Block E output)")
    p.add_argument("--table-kfold", type=Path, default=_RESULTS_DIR / "kfold_all_variants.json",
                   help="Path to kfold_all_variants.json (per-domain k-fold output)")
    p.add_argument("--significance", type=Path, default=None,
                   help="Path to loso_significance*.json. Supplies the 95%% CI on "
                        "Table 4's paired Δρ column; without it the column shows "
                        "the point estimate alone.")
    p.add_argument("--realworld", type=Path,
                   default=_RESULTS_DIR / "realworld_zeroshot.json",
                   help="Path to realworld_zeroshot.json; renders the zero-shot "
                        "identification table. Skipped when absent.")
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
                row += " " + _per_type_cell(
                    agg.get(f"{sc}|{v}", {}).get("per_node_type", {}).get(nt)
                ) + " |"
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
                render_identification_md(data3, out / "table3_identification_metrics.md")
                render_per_type_table_md(data3, out / "table5_per_type_metrics.md")
        
        print_id_metrics_console(data3)
    else:
        print(f"\n  [Table 3] Not found: {args.table3}")
        print("  Run: python reproduce/main_table.py")

    # ── Real-world zero-shot identification ──────────────────────────────────
    if args.realworld is not None and args.realworld.exists() and not args.console:
        print(f"\n  [Real-world zero-shot] {args.realworld}")
        render_realworld_identification_md(
            json.loads(args.realworld.read_text()),
            out / "realworld_identification_metrics.md",
        )

    # ── Table 4 ───────────────────────────────────────────────────────────────
    if args.table_controls is not None and args.table_controls.exists():
        print(f"\n  [RQ2 controls] {args.table_controls}")
        render_rq2_controls_md(
            json.loads(args.table_controls.read_text()),
            args.output_dir / "table_rq2_controls.md",
        )

    if args.table4.exists():
        print(f"\n  [Table 4] {args.table4}")
        loso_data = json.loads(args.table4.read_text())
        sig_data = None
        if args.significance is not None:
            if args.significance.exists():
                sig_data = json.loads(args.significance.read_text())
                print(f"  [Table 4 CIs] {args.significance}")
            else:
                print(f"  [Table 4 CIs] Not found: {args.significance} — Δρ without intervals")
        if not args.console:
            if not args.no_tex:
                render_table4_tex(loso_data, out / "table4_loso_results.tex", sig_data)
            if not args.tex_only:
                render_table4_md(loso_data, out / "table4_loso_results.md", sig_data)
        table = loso_data.get("comparison_table", {})
        contrasts = _loso_contrasts(sig_data)
        baseline_label = _NATIVE_VARIANT_LABELS_PLAIN.get(
            _LOSO_DELTA_BASELINE, _LOSO_DELTA_BASELINE)
        print("\n  Table 4: LOSO Results")
        print(f"  {'Variant':<25} {'Mean ρ':<10} {'Std ρ':<10} {'Δρ vs ' + baseline_label}")
        print("  " + "─" * 70)
        for var in _LOSO_VARIANT_ORDER:
            if var not in table:
                continue
            r = table[var]
            label = _NATIVE_VARIANT_LABELS_PLAIN.get(var, var)
            mean_r = r.get("mean_rho")
            std_r  = r.get("std_rho")
            mean_s  = f"{mean_r:.4f}" if mean_r is not None else "—"
            std_s   = f"{std_r:.4f}" if std_r is not None else "—"
            delta_s = _delta_cell(var, r, contrasts, latex=False)
            print(f"  {label:<25} {mean_s:<10} {std_s:<10} {delta_s}")
    else:
        print(f"\n  [Table 4] Not found: {args.table4}")
        print("  Run: python reproduce/loso_all_variants.py")

    # ── K-Fold ────────────────────────────────────────────────────────────────
    if args.table_kfold.exists():
        print(f"\n  [K-Fold] {args.table_kfold}")
        kfold_data = json.loads(args.table_kfold.read_text())
        if not args.console:
            if not args.no_tex:
                render_table4kfold_tex(kfold_data, out / "table4_kfold_results.tex")
            if not args.tex_only:
                render_table4kfold_md(kfold_data, out / "table4_kfold_results.md")
        table = kfold_data.get("comparison_table", {})
        print("\n  Per-Domain K-Fold Results")
        baseline_label = _NATIVE_VARIANT_LABELS_PLAIN.get(
            _LOSO_DELTA_BASELINE, _LOSO_DELTA_BASELINE)
        print(f"  {'Variant':<25} {'Mean ρ':<10} {'Std ρ':<10} {'Δρ vs ' + baseline_label}")
        print("  " + "─" * 70)
        for var in _VARIANT_ORDER:
            if var not in table:
                continue
            r = table[var]
            label = _NATIVE_VARIANT_LABELS_PLAIN.get(var, var)
            mean_r = r.get("mean_rho")
            std_r  = r.get("std_rho")
            mean_s  = f"{mean_r:.4f}" if mean_r is not None else "—"
            std_s   = f"{std_r:.4f}" if std_r is not None else "—"
            delta_s = _kfold_delta_cell(var, r, latex=False)
            print(f"  {label:<25} {mean_s:<10} {std_s:<10} {delta_s}")
    else:
        print(f"\n  [K-Fold] Not found: {args.table_kfold}")
        print("  Run: python reproduce/kfold_all_variants.py")

    print("\n  Done. Files written to:", out)


if __name__ == "__main__":
    main()
