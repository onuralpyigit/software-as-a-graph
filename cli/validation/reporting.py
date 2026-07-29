"""Console, LaTeX and pre-flight reporting for the validation CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import networkx as nx

from .runners import AblationReport
from .statistics import GATE_THRESHOLDS, SweepReport, ValidationResult


_COLOR = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}


def _c(text: str, color: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{_COLOR[color]}{text}{_COLOR['reset']}"


def _tick(ok: bool, use_color: bool) -> str:
    return _c("✓", "green", use_color) if ok else _c("✗", "red", use_color)


def print_single_report(vr: ValidationResult, topo_class: str, use_color: bool = True):
    bold = _COLOR["bold"] if use_color else ""
    reset = _COLOR["reset"] if use_color else ""

    print(f"\n{bold}{'═'*64}{reset}")
    print(f"{bold}  VALIDATION REPORT  seed={vr.seed}  QoS={'ON' if vr.qos_enabled else 'OFF'}{reset}")
    print(f"{bold}{'═'*64}{reset}")
    print(f"  Nodes: {vr.n_nodes}  (Applications: {vr.n_app_nodes})  "
          f"Topology class: {_c(topo_class, 'cyan', use_color)}")

    print(f"\n{bold}  Rank Correlation{reset}")
    print(f"    Spearman ρ  = {_c(f'{vr.spearman_rho:.4f}', 'green' if vr.spearman_rho>=0.80 else 'red', use_color)}"
          f"  (p={vr.spearman_p:.4f})"
          f"  95% CI [{vr.bootstrap_ci_lo:.4f}, {vr.bootstrap_ci_hi:.4f}]")
    print(f"    Kendall τ   = {vr.kendall_tau:.4f}  (p={vr.kendall_p:.4f})")

    print(f"\n{bold}  Classification @ K={vr.top_k}{reset}")
    print(f"    Precision   = {vr.precision_at_k:.4f}")
    print(f"    Recall      = {vr.recall_at_k:.4f}")
    print(f"    F1          = {_c(f'{vr.f1_at_k:.4f}', 'green' if vr.f1_at_k>=0.70 else 'red', use_color)}")
    print(f"    SPOF-F1     = {vr.spof_f1:.4f}")
    print(f"    FTR         = {vr.ftr:.4f}")

    print(f"\n{bold}  Specialist Metrics{reset}")
    print(f"    ICR@K       = {vr.icr_at_k:.4f}")
    print(f"    BCE         = {vr.bce:.4f}")
    print(f"    PG (vs DC)  = {_c(f'{vr.pg:.4f}', 'green' if vr.pg>=0.03 else 'yellow', use_color)}")

    print(f"\n{bold}  Wilcoxon (Q > DC){reset}")
    print(f"    stat={vr.wilcoxon_stat:.2f}  p={vr.wilcoxon_p:.4f}  "
          f"{'significant' if vr.wilcoxon_significant else 'not significant'}")

    print(f"\n{bold}  Gate Evaluation  ({topo_class}){reset}")
    for gate, passed in vr.gates_passed.items():
        print(f"    {_tick(passed, use_color)} {gate}")

    overall = _c("PASS", "green", use_color) if vr.overall_pass else _c("FAIL", "red", use_color)
    print(f"\n  Overall: {bold}{overall}{reset}\n")

    if vr.strata:
        print(f"{bold}  Node-type Strata{reset}")
        for ntype, s in vr.strata.items():
            n_str = s.get("n", 0)
            if "note" in s:
                print(f"    {ntype:16s} n={n_str}  {s['note']}")
            else:
                rho_str = _c(f'{s["spearman_rho"]:.4f}', 'green' if s["spearman_rho"] >= 0.70 else 'yellow', use_color)
                print(f"    {ntype:16s} n={n_str:4d}  ρ={rho_str}  F1={s['f1_at_k']:.4f}")


def print_sweep_report(sr: SweepReport, use_color: bool = True):
    bold = _COLOR["bold"] if use_color else ""
    reset = _COLOR["reset"] if use_color else ""

    print(f"\n{bold}{'═'*64}{reset}")
    print(f"{bold}  SWEEP REPORT  QoS={'ON' if sr.qos_enabled else 'OFF'}  "
          f"seeds={sr.seeds}{reset}")
    print(f"{bold}{'═'*64}{reset}")
    print(f"  ρ  mean={_c(f'{sr.rho_mean:.4f}','green' if sr.rho_mean>=0.80 else 'red',use_color)}"
          f"  std={sr.rho_std:.4f}  "
          f"[{sr.rho_min:.4f}, {sr.rho_max:.4f}]")
    print(f"  F1 mean={sr.f1_mean:.4f}")
    print(f"  PG mean={sr.pg_mean:.4f}")
    print(f"  RCR     = {_c(f'{sr.rcr:.4f}','green' if sr.rcr>=0.90 else 'yellow',use_color)}")
    print(f"  All-gates pass rate = {sr.all_gates_pass_rate:.2%}\n")

    print(f"{bold}  Per-seed ρ{reset}")
    for r in sr.per_seed:
        ok = _tick(r.overall_pass, use_color)
        print(f"    seed={r.seed}  ρ={r.spearman_rho:.4f}  F1={r.f1_at_k:.4f}  PG={r.pg:.4f}  {ok}")


def print_ablation_report(ar: AblationReport, use_color: bool = True):
    bold = _COLOR["bold"] if use_color else ""
    reset = _COLOR["reset"] if use_color else ""

    def _delta(v: float) -> str:
        sign = "+" if v >= 0 else ""
        col = "green" if v > 0.005 else ("yellow" if v > -0.005 else "red")
        return _c(f"{sign}{v:.4f}", col, use_color)

    print(f"\n{bold}{'═'*64}{reset}")
    print(f"{bold}  ABLATION REPORT  (topology-only vs QoS-enriched){reset}")
    print(f"{bold}{'═'*64}{reset}")
    print(f"  Topology class: {ar.topology_class}   "
          f"Nodes: {ar.n_nodes}  Apps: {ar.n_app_nodes}   "
          f"Seeds: {ar.seeds}")

    header = f"\n  {'Metric':<20} {'Topo-only':>12} {'QoS-enr':>12} {'Δ':>10}"
    sep    = f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}"
    print(header)
    print(sep)

    def row(label, b, e):
        d = e - b
        print(f"  {label:<20} {b:>12.4f} {e:>12.4f} {_delta(d):>10}")

    row("ρ  (mean)",   ar.base_rho_mean,  ar.enr_rho_mean)
    row("ρ  (std)",    ar.base_rho_std,   ar.enr_rho_std)
    row("F1 (mean)",   ar.base_f1_mean,   ar.enr_f1_mean)
    row("PG (mean)",   ar.base_pg_mean,   ar.enr_pg_mean)
    row("RCR",         ar.base_rcr,       ar.enr_rcr)

    sig_str = _c("significant (p<α)", "green", use_color) if ar.rho_lift_significant \
              else _c("not significant", "yellow", use_color)
    print(f"\n  QoS-enriched ρ lift: {sig_str}")
    print(f"  Δρ = {ar.delta_rho:+.4f}  ΔF1 = {ar.delta_f1:+.4f}  ΔPG = {ar.delta_pg:+.4f}\n")


def write_latex_table(ar: AblationReport, path: str):
    """
    Write a ready-to-paste IEEE two-column LaTeX table (booktabs style)
    suitable for Middleware 2026 / VISSOFT 2026 / UYMS 2026.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation Study: Topology-Only vs.\ QoS-Enriched Prediction}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{@{}lSSS@{}}",
        r"\toprule",
        r"Metric & {Topo-Only} & {QoS-Enr.} & {$\Delta$} \\",
        r"\midrule",
        rf"Spearman $\rho$ (mean) & {ar.base_rho_mean:.4f} & {ar.enr_rho_mean:.4f} & {ar.delta_rho:+.4f} \\",
        rf"Spearman $\rho$ (std)  & {ar.base_rho_std:.4f}  & {ar.enr_rho_std:.4f}  & {ar.enr_rho_std - ar.base_rho_std:+.4f} \\",
        rf"F1 @ $K$               & {ar.base_f1_mean:.4f} & {ar.enr_f1_mean:.4f} & {ar.delta_f1:+.4f} \\",
        rf"Predictive Gain (PG)   & {ar.base_pg_mean:.4f} & {ar.enr_pg_mean:.4f} & {ar.delta_pg:+.4f} \\",
        rf"RCR                    & {ar.base_rcr:.4f}      & {ar.enr_rcr:.4f}     & {ar.enr_rcr - ar.base_rcr:+.4f} \\",
        r"\midrule",
    ]
    sig_note = r"$p < \alpha$, significant" if ar.rho_lift_significant \
               else r"not significant"
    lines += [
        rf"\multicolumn{{4}}{{l}}{{\small QoS $\rho$-lift: {sig_note}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    Path(path).write_text("\n".join(lines))


_MIN_APPS_FOR_RELIABLE_RHO = 10


def _check_min_apps(G: nx.DiGraph, use_color: bool = True):
    """Warn if fewer than MIN_APPS Application nodes are present."""
    n_apps = sum(1 for _, d in G.nodes(data=True) if d.get("ntype") == "Application")
    if n_apps < _MIN_APPS_FOR_RELIABLE_RHO:
        msg = (f"WARNING: only {n_apps} Application nodes found "
               f"(minimum recommended: {_MIN_APPS_FOR_RELIABLE_RHO}). "
               f"Spearman ρ will have high variance on small n.")
        print(_c(msg, "yellow", use_color))
    return n_apps
