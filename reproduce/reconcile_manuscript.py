#!/usr/bin/env python3
"""Check every table figure in the manuscript against the artifact that produced it.

This exists because the failure it detects has already happened. The scenario
corpus was regenerated on 2026-09-07; most of Section 7 was not re-run, and the
manuscript reported two different corpora as one for a day without anything
noticing. The test suite could not catch it: ``pytest`` verifies code invariants,
not that a number typeset in LaTeX still matches the JSON it came from.

What it checks
--------------
Each entry in :data:`CHECKS` names a manuscript table, the artifact that backs
it, and how to pull the same quantity from both. A row is reported when the two
disagree by more than its declared tolerance, and — separately — when the
artifact is *older than the corpus it claims to describe*, which is the specific
staleness that caused the original incident.

What it does not check
----------------------
Numbers appearing only in prose. Extracting those reliably would need the
manuscript to carry machine-readable markers; the tables are where the
concentration of reported values is, and where a stale figure does most damage.
Prose figures derived from the same artifacts are listed in ``PROSE_NOTES`` as a
reminder that they move together.

Usage
-----
    python reproduce/reconcile_manuscript.py            # check, exit 1 on drift
    python reproduce/reconcile_manuscript.py --verbose  # show every comparison
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "docs/research/jss/latex/sections"
RESULTS = ROOT / "results"
CORPUS = ROOT / "data/scenarios"

#: Variant id -> display label, mirroring saag/evaluation/variant_registry.py.
LOSO_LABELS = {
    "topo_baseline": "Topo",
    "topo_qos": "Topo-QoS",
    "topology_rm": "RM / $Q(v)$",
    "gl": "GAT-N",
    "gl_qos": "GAT-N-QoS",
    "hgl": "HGT",
    "hgl_qos": "HGT-QoS",
}


@dataclass
class Finding:
    table: str
    row: str
    field: str
    manuscript: Any
    artifact: Any
    detail: str = ""


@dataclass
class Report:
    findings: List[Finding] = field(default_factory=list)
    stale: List[str] = field(default_factory=list)
    checked: int = 0
    skipped: List[str] = field(default_factory=list)


def _load(name: str) -> Optional[dict]:
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _tex(name: str) -> str:
    return (SECTIONS / name).read_text()


def _rows(tex: str, start_marker: str, end_marker: str = r"\bottomrule",
          after_label: Optional[str] = None) -> List[str]:
    """Return the LaTeX row strings of one table body.

    ``after_label`` anchors the search past a given ``\\label{...}``, which is
    required whenever two tables share first-column text -- Table 9 and Table 9b
    both list the same five system names but carry different columns.
    """
    base = 0
    if after_label is not None:
        base = tex.index(after_label)
    try:
        i = tex.index(start_marker, base)
    except ValueError:
        return []
    j = tex.index(end_marker, i)
    out = []
    for line in tex[i:j].split("\n"):
        line = line.strip()
        if line.startswith(r"\textbf{") and line.endswith(r"\\"):
            out.append(line)
    return out


def _cells(row: str) -> List[str]:
    return [c.strip() for c in row.rstrip("\\").split("&")]


_NUM = re.compile(r"-?\d+\.\d+|\$-\$\d+\.\d+|-?\d+")


def _num(cell: str) -> Optional[float]:
    """First numeric literal in a LaTeX cell, honouring $-$ as a minus sign."""
    c = cell.replace(r"$-$", "-").replace("{,}", "").replace("{", "").replace("}", "")
    c = re.sub(r"\\textbf|\\scriptsize|\\pm.*", "", c)
    m = _NUM.search(c)
    return float(m.group()) if m else None


def _label(cell: str) -> str:
    return re.sub(r"\\textbf\{|\}|\\", "", cell).strip()


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------

def check_table4_corpus(rep: Report) -> None:
    """Table 4 entity and edge counts against the committed topologies.

    This is the check that would have caught the stale edge counts: six of seven
    scenarios disagreed with the corpus by up to +144 edges after the rebuild.
    """
    tex = _tex("sec6_experimental_setup.tex")
    rows = _rows(tex, r"\textbf{Autonomous Vehicle (AV)}")
    name_to_file = {
        "Autonomous Vehicle (AV)": "av_system", "Enterprise Pub-Sub": "enterprise_system",
        "Financial Trading": "financial_trading_system", "Healthcare Integration": "healthcare_system",
        "Hub-and-Spoke Enterprise": "hub_and_spoke_system", "IoT Smart City": "iot_smart_city_system",
        "Microservices Mesh": "microservices_system", "Telecom RAN": "telecom_ran_system",
        "Industrial SCADA": "industrial_scada_system", "Real-Time Gaming": "realtime_gaming_system",
        "Logistics Fleet": "logistics_fleet_system", "Air Traffic Management (ATM)": "atm_system",
        "Autoware.universe": "realworld_autoware_ros2", "Cloud Microservices": "realworld_cloud_microservices",
        "Train-Ticket": "realworld_trainticket", "Home Assistant": "realworld_homeassistant",
        "EdgeX Foundry": "realworld_edgex",
    }
    for row in rows:
        cells = _cells(row)
        if len(cells) < 9:
            continue
        label = _label(cells[0]).split(" \\cite")[0].split("\\cite")[0].strip()
        fname = next((v for k, v in name_to_file.items() if label.startswith(k)), None)
        if not fname:
            continue
        path = CORPUS / f"{fname}.json"
        if not path.exists():
            rep.skipped.append(f"tab:4 {label}: no committed topology")
            continue
        d = json.loads(path.read_text())
        counts = {k: len(d.get(k, [])) for k in
                  ("applications", "topics", "brokers", "nodes", "libraries")}
        true_V = sum(counts.values())
        true_E = sum(len(v) for v in d["relationships"].values())
        for idx, (fieldname, truth) in enumerate(
            [("|V|", true_V), ("|V_app|", counts["applications"]), ("Topics", counts["topics"]),
             ("Brokers", counts["brokers"]), ("Hosts", counts["nodes"]),
             ("Libs", counts["libraries"]), ("|E|", true_E)], start=2
        ):
            got = _num(cells[idx])
            rep.checked += 1
            if got is None or int(got) != truth:
                rep.findings.append(Finding("tab:4", label, fieldname, got, truth))


def check_table7_loso(rep: Report, artifact: str) -> None:
    """Table 7 LOSO means and F1 against the variants artifact."""
    d = _load(artifact)
    if d is None:
        rep.skipped.append(f"tab:7: {artifact} absent")
        return
    ct = d["comparison_table"]
    tex = _tex("sec7_results.tex")
    rows = _rows(tex, r"\multicolumn{7}{l}{\textit{Training-free structural baselines}}")
    by_label = {LOSO_LABELS[k]: v for k, v in ct.items() if k in LOSO_LABELS}
    for row in rows:
        cells = _cells(row)
        if len(cells) < 7:
            continue
        label = _label(cells[0])
        blk = by_label.get(label) or by_label.get(label.replace("RM / Q(v)", "RM / $Q(v)$"))
        if blk is None:
            continue
        for idx, key, tol in ((1, "mean_rho", 0.001), (5, "mean_f1", 0.001)):
            got, truth = _num(cells[idx]), blk.get(key)
            rep.checked += 1
            if truth is not None and (got is None or abs(got - truth) > tol):
                rep.findings.append(Finding("tab:7", label, key, got, round(truth, 4)))


def check_table7c_active(rep: Report, artifact: str) -> None:
    """Table 7c full-population and active-stratum LOSO means.

    ``rho_>0`` is the mean over folds of ``spearman_rho_positive``; folds where
    the metric is undefined (fewer than three positive components, or no spread)
    record ``None`` and are excluded from the mean rather than counted as zero.
    """
    d = _load(artifact)
    if d is None:
        rep.skipped.append(f"tab:7c: {artifact} absent")
        return
    ct = d["comparison_table"]
    pv = d["per_variant_results"]
    tex = _tex("sec7_results.tex")
    rows = _rows(tex, r"\textbf{RM / $Q(v)$}", after_label=r"\label{tab:7c}")
    by_label = {LOSO_LABELS[k]: k for k in ct if k in LOSO_LABELS}
    for row in rows:
        cells = _cells(row)
        if len(cells) < 4:
            continue
        label = _label(cells[0])
        key = by_label.get(label)
        if key is None:
            continue
        vals = [f["mean_metrics"].get("spearman_rho_positive")
                for f in pv[key]["folds"]]
        vals = [v for v in vals if v is not None]
        truths = (ct[key].get("mean_rho"),
                  sum(vals) / len(vals) if vals else None)
        for idx, truth, name in ((1, truths[0], "mean_rho"),
                                 (2, truths[1], "mean_rho_positive")):
            got = _num(cells[idx])
            rep.checked += 1
            if truth is not None and (got is None or abs(got - truth) > 0.001):
                rep.findings.append(
                    Finding("tab:7c", label, name, got, round(truth, 4)))


def check_scale_table(rep: Report) -> None:
    """Table tab:scale per-stage latency against inference_latency artifact."""
    d = _load("inference_latency_v3.json") or _load("inference_latency.json")
    if d is None:
        rep.skipped.append("tab:scale: no inference_latency artifact")
        return
    sizes = {s["n_actual"]: s for s in d["sizes"]}
    tex = _tex("sec7_results.tex")
    i = tex.index(r"\textbf{$|V|$} & \textbf{$|E|$}")
    j = tex.index(r"\bottomrule", i)
    for line in tex[i:j].split("\n"):
        line = line.strip()
        if not line.endswith(r"\\") or "&" not in line or "textbf{$|V|$}" in line:
            continue
        cells = _cells(line)
        n = _num(cells[0])
        if n is None or int(n) not in sizes:
            continue
        s = sizes[int(n)]
        an = s["analyze_s"]; an = an["median"] if isinstance(an, dict) else an
        fw = s["forward_ms"]; fw = fw["median"] if isinstance(fw, dict) else fw
        for idx, truth, tol, nm in ((2, an, 0.05, "analyse_s"), (4, fw, 0.5, "forward_ms")):
            got = _num(cells[idx])
            rep.checked += 1
            if got is None or abs(got - truth) > tol:
                rep.findings.append(Finding("tab:scale", f"n={int(n)}", nm, got, round(truth, 2)))


def check_realworld(rep: Report) -> None:
    """Table 9b full-population and active-stratum correlations."""
    d = _load("realworld_zeroshot.json")
    if d is None:
        rep.skipped.append("tab:9b: realworld_zeroshot.json absent")
        return
    per = d["per_system"]
    name_to_key = {
        "Cloud Microservices Mesh": "realworld_cloud_microservices",
        "Train-Ticket Booking Mesh": "realworld_trainticket",
        "Autoware.universe (ROS~2)": "realworld_autoware_ros2",
        "EdgeX Foundry (Industrial IoT)": "realworld_edgex",
        "Home Assistant (Smart Home)": "realworld_homeassistant",
    }
    tex = _tex("sec7_results.tex")
    rows = _rows(tex, r"\textbf{Cloud Microservices Mesh}", after_label=r"\label{tab:9b}")
    for row in rows:
        cells = _cells(row)
        if len(cells) < 8:
            continue
        label = _label(cells[0])
        key = next((v for k, v in name_to_key.items() if label.startswith(k.split(" (")[0])), None)
        if key is None or key not in per:
            continue
        s = per[key]
        for idx, k, tol, nm in ((4, "mean_rho", 0.002, "rho"),
                                (5, "mean_rho_positive", 0.002, "rho_positive"),
                                (6, "n_positive", 0.5, "n_positive")):
            got, truth = _num(cells[idx]), s.get(k)
            rep.checked += 1
            if truth is not None and (got is None or abs(got - truth) > tol):
                rep.findings.append(Finding("tab:9b", label, nm, got, round(truth, 4)))


def check_freshness(rep: Report) -> None:
    """Flag any backing artifact older than the corpus it describes.

    This is the staleness check proper. An artifact that predates the newest
    committed topology was computed against a corpus that no longer exists.
    """
    newest_corpus = max(p.stat().st_mtime for p in CORPUS.glob("*_system.json"))
    for name in ("loso_all_variants_v4.json", "loso_all_variants_v3.json",
                 "main_table_v3.json", "inference_latency_v3.json",
                 "realworld_zeroshot.json", "detection_validation_v3.json",
                 "convergent_validity.json", "topic_weight_sensitivity_v3.json",
                 "weight_global_sensitivity_v3.json", "ahp_shrinkage_sweep_v3.json",
                 "threshold_sensitivity_v3.json", "atm_scale_sweep_v3.json"):
        p = RESULTS / name
        if not p.exists():
            continue
        if p.stat().st_mtime < newest_corpus:
            rep.stale.append(f"{name} predates the committed corpus")


PROSE_NOTES = [
    "Wilcoxon contrasts in 7.1/7.2 <- results/loso_significance_v*.json",
    "QoS ablation deltas in 7.3.1 <- loso_all_variants_v*.json",
    "sigma-hat diagnostic in 7.2.3 <- output/loso_v*/<variant>/inductive_predictions.json",
    "label-noise ceiling in 7.1 <- output/loso_cache/*/failure_impact.json label_stability",
    "gate range in 7.5 <- results/detection_validation_timed_v3.json gate_seconds",
    "oracle sweep cost in 7.5.1 <- scratchpad/oracle_timing.csv",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loso", default="loso_all_variants_v4.json",
                    help="LOSO artifact backing Table 7 (default: v3)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rep = Report()
    check_freshness(rep)
    check_table4_corpus(rep)
    check_table7_loso(rep, args.loso)
    check_table7c_active(rep, args.loso)
    check_scale_table(rep)
    check_realworld(rep)

    print(f"\n  Reconciled {rep.checked} table figures against committed artifacts.\n")

    if rep.stale:
        print("  STALE ARTIFACTS (older than the corpus they describe):")
        for s in rep.stale:
            print(f"    ! {s}")
        print()

    if rep.findings:
        print("  MISMATCHES:")
        w = max(len(f.table) for f in rep.findings)
        for f in rep.findings:
            print(f"    ! {f.table:{w}}  {f.row:34}  {f.field:14} "
                  f"manuscript={f.manuscript}  artifact={f.artifact}")
        print()

    if rep.skipped and args.verbose:
        print("  SKIPPED:")
        for s in rep.skipped:
            print(f"    - {s}")
        print()

    if args.verbose:
        print("  Prose figures not machine-checked (move with the same artifacts):")
        for n in PROSE_NOTES:
            print(f"    - {n}")
        print()

    ok = not rep.findings and not rep.stale
    print("  OK — every checked figure matches its artifact.\n" if ok
          else f"  {len(rep.findings)} mismatch(es), {len(rep.stale)} stale artifact(s).\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
