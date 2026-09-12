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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduce._provenance import corpus_digest  # noqa: E402
from saag.evaluation import variant_registry as _registry  # noqa: E402

SECTIONS = ROOT / "docs/research/jss/latex/sections"
LATEX = ROOT / "docs/research/jss/latex"
RESULTS = ROOT / "results"
CORPUS = ROOT / "data/scenarios"

#: Variant id -> display label under the LOSO harness.
#:
#: Derived from saag/evaluation/variant_registry.py rather than hand-copied.
#: A row whose label is missing here is *skipped* by check_table7_loso, so a
#: stale hand-written mirror made the reconciler silently report a clean run
#: over rows it never checked -- the one failure mode this script exists to
#: prevent. RM keeps its manuscript spelling, which carries maths the registry
#: label does not.
#: ``gl_full``/``gl_full_qos`` are excluded: they are in-distribution-only ids,
#: and under LOSO the registry resolves ``gl``/``gl_qos`` to the same two
#: labels, so including both sides would put two ids under one label and let
#: whichever came last win a lookup silently.
LOSO_LABELS = {
    v: _registry.label(v, harness="loso")
    for v in _registry.VARIANTS
    if v not in ("gl_full", "gl_full_qos")
}
LOSO_LABELS["topology_rm"] = "RM / $Q(v)$"

if len(set(LOSO_LABELS.values())) != len(LOSO_LABELS):
    raise RuntimeError(
        "LOSO_LABELS maps two variant ids to one label; a table row would be "
        "reconciled against the wrong artifact column."
    )


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
    dirty: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    checked: int = 0
    skipped: List[str] = field(default_factory=list)


def _load(name: str) -> Optional[dict]:
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _tex(name: str) -> str:
    return (SECTIONS / name).read_text()


def _supp() -> str:
    """The supplement, which is a separate document beside ``sections/``.

    It carries its own copies of body figures (S6 restates 7.3.6's
    stratification numbers, S7 restates the degree comparison). Those copies
    went stale once while the body was corrected, because nothing checked them.
    """
    return (LATEX / "supplementary.tex").read_text()


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


#: Table 5 row label -> scenario id. Only the display spelling differs; the
#: artifact keys its aggregate as "<scenario>|<variant>".
TABLE5_SCENARIOS = {
    "ATM System": "atm_system",
    "AV System": "av_system",
    "Enterprise": "enterprise_system",
    "Financial Trading": "financial_trading_system",
    "Healthcare": "healthcare_system",
    "Hub-and-Spoke": "hub_and_spoke_system",
    "Industrial SCADA": "industrial_scada_system",
    "IoT Smart City": "iot_smart_city_system",
    "Logistics Fleet": "logistics_fleet_system",
    "Microservices": "microservices_system",
    "Real-Time Gaming": "realtime_gaming_system",
    "Telecom RAN": "telecom_ran_system",
}

#: Table 5's column order, as variant ids. Labels come from the registry under
#: the in-distribution harness rather than being hand-copied, for the same
#: reason LOSO_LABELS does: a hand-written mirror going stale is how a table
#: gets reconciled against the wrong column while still reporting clean.
TABLE5_VARIANTS = ["topo_baseline", "topo_qos", "gl", "gl_qos", "hgl", "hgl_qos"]


def check_table5_indist(rep: Report, artifact: str = "main_table.json") -> None:
    """Table 5's in-distribution cells and column means."""
    d = _load(artifact)
    if d is None:
        rep.skipped.append(f"tab:5: {artifact} absent")
        return
    agg = d.get("aggregate") or {}
    cfg = d.get("config") or {}
    # Guard against being pointed at a smoke run.
    if len(d.get("cells") or []) < 42:
        rep.skipped.append(
            f"tab:5: {artifact} holds {len(d.get('cells') or [])} cells — too few to "
            "back a 12x6 table; refusing to reconcile against a smoke run")
        return
    present = {k.split("|")[1] for k in agg if not k.startswith("_") and "|" in k}
    variants = list(TABLE5_VARIANTS)
    if present and not ({"gl", "gl_qos"} & present) and ({"gl_full", "gl_full_qos"} & present):
        variants = ["topo_baseline", "topo_qos", "gl_full", "gl_full_qos", "hgl", "hgl_qos"]
    labels = {v: _registry.label(v, harness="in_distribution") for v in variants}

    tex = _tex("sec7_results.tex")
    rows = _rows(tex, r"\textbf{Scenario} & \textbf{$n$}")
    seen = {}
    for row in rows:
        cells = _cells(row)
        if len(cells) < 8:
            continue
        label = _label(cells[0])
        scen = TABLE5_SCENARIOS.get(label)
        if scen is None:
            continue
        seen[scen] = True
        for idx, vid in enumerate(variants, start=2):
            blk = agg.get(f"{scen}|{vid}")
            if blk is None:
                rep.skipped.append(f"tab:5: no aggregate for {scen}|{vid}")
                continue
            got, truth = _num(cells[idx]), blk.get("mean_rho")
            rep.checked += 1
            if truth is not None and (got is None or abs(got - truth) > 0.001):
                rep.findings.append(
                    Finding("tab:5", f"{label} / {labels[vid]}", "mean_rho",
                            got, round(truth, 4)))

    missing = set(TABLE5_SCENARIOS.values()) - set(seen)
    if missing:
        rep.skipped.append(f"tab:5: rows not found in the tex for {sorted(missing)}")

    # Column means, recomputed over exactly the scenarios the artifact ran.
    scenarios = cfg.get("scenarios") or sorted(TABLE5_SCENARIOS.values())
    mean_rows = _rows(tex, r"\textbf{Mean} & ---")
    if not mean_rows:
        rep.skipped.append("tab:5: Mean row not found")
        return
    cells = _cells(mean_rows[0])
    for idx, vid in enumerate(variants, start=2):
        vals = [agg[f"{sc}|{vid}"]["mean_rho"] for sc in scenarios
                if f"{sc}|{vid}" in agg and agg[f"{sc}|{vid}"].get("mean_rho") is not None]
        if not vals:
            continue
        truth = sum(vals) / len(vals)
        got = _num(cells[idx]) if idx < len(cells) else None
        rep.checked += 1
        if got is None or abs(got - truth) > 0.001:
            rep.findings.append(
                Finding("tab:5", f"Mean / {labels[vid]}", "mean_rho", got, round(truth, 4)))


def check_supplement_stratification(rep: Report) -> None:
    """Supplement S6/S7's copies of the 7.3.6 stratification figures.

    The supplement restates body numbers as literal text -- it cannot \ref into
    the body, since the two documents do not share an .aux. Nothing checked it,
    and when detection_validation was refreshed the body was corrected while
    both supplement copies kept the superseded pooled rho and degree comparison.
    """
    art = _load("detection_validation_v3.json")
    if art is None:
        rep.skipped.append("detection_validation_v3.json absent; S6/S7 unchecked")
        return
    summ = art.get("summary") or {}
    pooled = (summ.get("pooling_check") or {}).get("pooled_mean_rho")
    by_type = summ.get("q_by_type") or {}
    degree = summ.get("degree") or {}
    composite = summ.get("q_composite") or {}
    tex = _supp()

    expected = [("Application", (by_type.get("Application") or {}).get("mean_spearman_rho")),
                ("Broker", (by_type.get("Broker") or {}).get("mean_spearman_rho")),
                ("Node", (by_type.get("Node") or {}).get("mean_spearman_rho")),
                ("pooled", pooled),
                ("degree rho", degree.get("mean_spearman_rho")),
                ("degree F1", degree.get("mean_f1")),
                ("RM rho", composite.get("mean_spearman_rho")),
                ("RM F1", composite.get("mean_f1"))]
    for name, truth in expected:
        if truth is None:
            continue
        rep.checked += 1
        # Presence of the 3-decimal literal as a standalone number. The
        # supplement writes these variously as "$0.566$" and "$\rho = 0.566$",
        # so match the token, not a fixed wrapper. This is a weaker test than
        # the table checks above -- it catches a figure that was updated in the
        # body and not here, which is the defect that actually occurred.
        if not re.search(rf"(?<![\d.]){re.escape(f'{truth:.3f}')}(?![\d])", tex):
            rep.findings.append(
                Finding("supp:S6", name, "value", "not present in supplementary.tex",
                        f"{truth:.3f}"))


def check_supplement_shrinkage(rep: Report) -> None:
    """Supplement S1's AHP shrinkage endpoints, in the table AND in the prose.

    S1 states the same sweep twice, once in the sensitivity table and once in
    the paragraph below it. They disagreed: the table said 0.262 -> 0.166 with
    spread 0.096 while the prose two paragraphs later said 0.319 -> 0.200 with
    spread 0.119, which is what the artifact says. A figure restated in two
    places in one document is exactly what goes half-stale.
    """
    art = _load("ahp_shrinkage_sweep_v3.json")
    if art is None:
        rep.skipped.append("ahp_shrinkage_sweep_v3.json absent; S1 shrinkage unchecked")
        return
    rows = {r.get("lambda"): r.get("mean_rho") for r in art.get("rows", [])}
    lo, hi = rows.get(0.0), rows.get(1.0)
    if lo is None or hi is None:
        rep.skipped.append("ahp_shrinkage_sweep_v3.json has no lambda 0.0/1.0 row")
        return
    tex = _supp()
    for name, truth in (("lambda=0 rho", lo), ("lambda=1 rho", hi), ("spread", lo - hi)):
        rep.checked += 1
        # Both the table row and the prose must carry the same value, so a
        # count of at least two is the real expectation; one means the other
        # copy has drifted.
        hits = len(re.findall(rf"(?<![\d.]){re.escape(f'{truth:.3f}')}(?![\d])", tex))
        if hits < 2:
            rep.findings.append(
                Finding("supp:S1", name, "occurrences",
                        f"{hits} (table and prose must agree)", f"{truth:.3f} x2"))


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
    d = _load("realworld_zeroshot_v4.json") or _load("realworld_zeroshot.json")
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
                                (6, "n_positive", 0.5, "n_positive"),
                                (7, "mean_f1_at_k", 0.002, "f1_at_k")):
            got, truth = _num(cells[idx]), s.get(k)
            rep.checked += 1
            if truth is not None and (got is None or abs(got - truth) > tol):
                rep.findings.append(Finding("tab:9b", label, nm, got, round(truth, 4)))


#: Artifacts whose freshness is checked, and what each one backs. Only artifacts
#: that are actually consumed belong here: a superseded version left on disk
#: (``*_v3`` once ``*_v4`` is in use) describes a corpus nobody reads it against,
#: so flagging it trains the reader to ignore the warning list.
#: The ``*_v5`` family was previously declared here before it existed. Absent
#: targets are ``continue``d, so declaring an artifact nobody produces bought
#: nothing and cost something: ``main_table_v5.json`` was the only declared
#: check on Table 5's forty-two cells, and because it is absent those cells went
#: unchecked while the summary line still read "every checked figure matches".
#: Declare an artifact here only once it is consumed; Table 5 is now checked
#: against the artifact that actually backs it (``check_table5_indist``).
FRESHNESS_TARGETS = {
    "loso_all_variants_v4.json": "Tables 7/7c",
    "realworld_zeroshot_v4.json": "Table 9b",
    "detection_validation_v3.json": "7.3 stratification",
    "convergent_validity.json": "Table 8c",
    "topic_weight_sensitivity_v3.json": "Supplementary S1",
    "weight_global_sensitivity_v3.json": "Supplementary S1",
    "ahp_shrinkage_sweep_v3.json": "Supplementary S1",
    "threshold_sensitivity_v3.json": "Supplementary S3",
    "atm_scale_sweep_v3.json": "Supplementary S6",
    "qos_label_ablation.json": "Section 4.3",
}

#: Artifacts that never read the corpus, so the corpus-freshness rule cannot
#: apply to them and applying it produces a permanent false positive.
#:
#: ``inference_latency_v3.json`` times generate/analyse/forward over topologies
#: it synthesises itself (``tools.generation.service.generate_graph`` at fixed
#: sizes, seed 42); it contains no reference to ``data/scenarios`` at all. It
#: was flagged stale on every run because its mtime predates a corpus it does
#: not read. Re-running to clear that flag would silently re-time Table 12, 7.5,
#: 1.5 and the abstract on whatever machine and load happened to be current,
#: which is a real cost paid for no epistemic gain. What its numbers depend on
#: is the machine, so re-measure it deliberately, not to quiet a warning.
CORPUS_INDEPENDENT_ARTIFACTS = {
    "inference_latency_v3.json": "scale/latency table (self-generated topologies)",
}

#: Wall-clock artifacts, deliberately outside FRESHNESS_TARGETS.
#:
#: The corpus-freshness rule does not apply to them and applying it is actively
#: harmful. ``oracle_timing_v4.json`` measures how long the labeler takes over a
#: node population the corpus change left identical (39/104/360/... before and
#: after), so it is not corpus-stale in any meaningful sense; what its numbers
#: depend on is the machine and its load. Worse, its ratio pairs an oracle time
#: it measures itself against a gate time it reads from
#: ``detection_validation_timed_v3.json``. Refreshing one half alone silently
#: produces a ratio from two different measurement sessions --- doing exactly
#: that moved it from 11.45x to 15.35x with no change to the work being timed.
#: Re-measure the pair together, on an idle machine, or leave both.
PAIRED_TIMING_ARTIFACTS = {
    "oracle_timing_v4.json": "detection_validation_timed_v3.json",
}


def check_freshness(rep: Report) -> None:
    """Flag any backing artifact that does not describe the corpus on disk.

    Content first, mtime only as a fallback. mtime was the original test and it
    is wrong in both directions: the corpus regenerates byte-identically (CI
    asserts it), so a routine regeneration bumps every mtime and ages artifacts
    that are still perfectly valid, while a corpus restored from an older
    checkout carries mtimes newer than its content. An artifact stamped with
    ``provenance.corpus_digest`` is judged on that digest and its timestamp is
    irrelevant; one without a stamp falls back to the timestamp test and says
    so, because an unstamped artifact genuinely cannot prove what it described.
    """
    current = corpus_digest()
    newest_corpus = max(p.stat().st_mtime for p in CORPUS.glob("*_system.json"))
    for name, backs in FRESHNESS_TARGETS.items():
        p = RESULTS / name
        if not p.exists():
            # An artifact this script declares as backing a table, that is not
            # on disk, is a hole in the verification -- not a pass. results/ is
            # gitignored, so on a fresh clone this is EVERY artifact, and the
            # old `continue` let that report as a clean run.
            rep.missing.append(f"{name} ({backs}) is not in results/")
            continue
        prov = {}
        try:
            prov = json.loads(p.read_text()).get("provenance") or {}
        except (OSError, ValueError, AttributeError):
            prov = {}

        # A figure produced from a modified working tree cannot be regenerated
        # from the commit it names. _provenance.py records this as the failure
        # that already cost this project a published table, and stamps `dirty`
        # for exactly this check -- which nothing was making.
        if prov.get("dirty"):
            commit = str(prov.get("commit") or "unknown")[:8]
            rep.dirty.append(
                f"{name} ({backs}) was produced from a dirty tree at {commit}; "
                "it does not reproduce from any commit — re-run from a clean tree")

        stamped = prov.get("corpus_digest")
        if current and stamped:
            if stamped != current:
                rep.stale.append(
                    f"{name} ({backs}) describes a different corpus "
                    f"[{stamped[:12]} != {current[:12]}]")
            continue

        if p.stat().st_mtime < newest_corpus:
            rep.stale.append(
                f"{name} ({backs}) predates the corpus files; unstamped, so "
                "this is the weaker timestamp test — re-run to stamp it")


def check_oracle_timing(rep: Report) -> None:
    """Check Section 7.5.1's oracle-cost comparison against its artifact.

    These two figures are prose, not a table, and they are checked anyway
    because they carry a headline claim --- that the static gate is more
    expensive than the simulation it was meant to displace --- and because they
    previously had no committed artifact at all.
    """
    art = _load("oracle_timing_v4.json")
    if art is None:
        rep.skipped.append("oracle_timing_v4.json absent; 7.5.1 unchecked")
        return
    tex = _tex("sec7_results.tex")
    summary = art["summary"]
    lo, hi = summary["oracle_seconds"]["min"], summary["oracle_seconds"]["max"]

    m = re.search(r"gives \$([\d.]+)\$--\$([\d.]+)\\,\\text\{s\}\$ per scenario", tex)
    rep.checked += 1
    if not m:
        rep.findings.append(Finding("sec:7.5.oracle", "oracle sweep", "range",
                                    "not found", f"{lo}-{hi}"))
    else:
        got_lo, got_hi = float(m.group(1)), float(m.group(2))
        if abs(got_lo - lo) > 0.05 or abs(got_hi - hi) > 0.05:
            rep.findings.append(Finding("sec:7.5.oracle", "oracle sweep", "range",
                                        f"{got_lo}-{got_hi}", f"{lo}-{hi}"))

    ratio = summary.get("gate_over_oracle_at_max")
    if ratio is not None:
        words = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen"}
        expected = words.get(round(ratio))
        rep.checked += 1
        if expected is None or f"roughly {expected} times" not in tex:
            rep.findings.append(Finding("sec:7.5.oracle", "gate/oracle", "ratio",
                                        "see text", f"{ratio}x -> 'roughly {expected} times'"))


def check_qos_label_ablation(rep: Report) -> None:
    """Check Section 4.3's claim about how much QoS the primary label carries.

    Prose, and checked for the same reason as the oracle timing above: it is
    load-bearing. It states that I*(v)'s ordering is almost entirely recoverable
    without any QoS term, which is what bounds the QoS-encoding gain reported in
    7.3.1. An unbacked figure here would let a corpus change quietly invalidate
    the bound while the ablation number it qualifies stayed put.
    """
    art = _load("qos_label_ablation.json")
    if art is None:
        rep.skipped.append("qos_label_ablation.json absent; 4.3 QoS bound unchecked")
        return

    blocks = [
        s["label_rank_agreement_application"]
        for s in art.get("scenarios", [])
        if "label_rank_agreement_application" in s
    ]
    if not blocks:
        rep.skipped.append(
            "qos_label_ablation.json predates label_rank_agreement_application; "
            "re-run reproduce/qos_label_ablation.py"
        )
        return

    def _mean(arm: str, key: str) -> Optional[float]:
        vals = [b[arm][key] for b in blocks if key in b.get(arm, {})]
        return sum(vals) / len(vals) if vals else None

    tex = _tex("sec4_failure_impact_prediction.tex")
    for arm, key, pattern, tol in (
        ("ladder", "spearman_rho_vs_none",
         r"mean Spearman \$\\rho = ([\d.]+)\$ against the ladder", 0.002),
        ("wt", "spearman_rho_vs_none",
         r"durability-aware \$w\(t\)\$ scaling moves it less still \(\$\\rho = ([\d.]+)\$\)", 0.002),
        ("ladder", "topk_jaccard_vs_none",
         r"agree at mean Jaccard \$([\d.]+)\$", 0.002),
    ):
        expected = _mean(arm, key)
        if expected is None:
            continue
        rep.checked += 1
        m = re.search(pattern, tex)
        if not m:
            rep.findings.append(Finding("sec:4.3", f"{arm} {key}", "value",
                                        "not found", round(expected, 4)))
        elif abs(float(m.group(1)) - expected) > tol:
            rep.findings.append(Finding("sec:4.3", f"{arm} {key}", "value",
                                        float(m.group(1)), round(expected, 4)))

    lo = min(b["ladder"]["spearman_rho_vs_none"] for b in blocks)
    hi = max(b["ladder"]["spearman_rho_vs_none"] for b in blocks)
    rep.checked += 1
    m = re.search(r"\(range \$([\d.]+)\$--\$([\d.]+)\$\)", tex)
    if not m:
        rep.findings.append(Finding("sec:4.3", "ladder rho", "range",
                                    "not found", f"{lo:.3f}-{hi:.3f}"))
    elif abs(float(m.group(1)) - lo) > 0.002 or abs(float(m.group(2)) - hi) > 0.002:
        rep.findings.append(Finding("sec:4.3", "ladder rho", "range",
                                    f"{m.group(1)}-{m.group(2)}", f"{lo:.3f}-{hi:.3f}"))


PROSE_NOTES = [
    "Wilcoxon contrasts in 7.1/7.2 <- results/loso_significance_v*.json",
    "QoS ablation deltas in 7.3.1 <- loso_all_variants_v*.json",
    "sigma-hat diagnostic in 7.2.3 <- output/loso_v*/<variant>/inductive_predictions.json",
    "label-noise ceiling in 7.1 <- output/loso_cache/*/failure_impact.json label_stability",
    "gate range in 7.5 <- results/detection_validation_timed_v3.json gate_seconds",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loso", default="loso_all_variants_v4.json",
                    help="LOSO artifact backing Tables 7/7c")
    ap.add_argument("--main-table", default="main_table.json",
                    help="in-distribution artifact backing Table 5")
    ap.add_argument("--allow-missing", action="store_true",
                    help="do not fail when a declared artifact is absent from "
                         "results/ (results/ is gitignored, so a fresh clone has "
                         "none of them and would otherwise report a clean run "
                         "having verified almost nothing)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rep = Report()
    check_freshness(rep)
    check_table4_corpus(rep)
    check_table5_indist(rep, args.main_table)
    check_supplement_stratification(rep)
    check_supplement_shrinkage(rep)
    check_table7_loso(rep, args.loso)
    check_table7c_active(rep, args.loso)
    check_scale_table(rep)
    check_realworld(rep)
    check_oracle_timing(rep)
    check_qos_label_ablation(rep)

    print(f"\n  Reconciled {rep.checked} table figures against committed artifacts "
          f"({len(rep.skipped)} check(s) skipped).\n")

    if rep.missing:
        print("  MISSING ARTIFACTS (declared as backing a table, not on disk):")
        for m in rep.missing:
            print(f"    ? {m}")
        print("    results/ is gitignored — regenerate with reproduce/Makefile, "
              "or pass --allow-missing to treat this as non-fatal.\n")

    if rep.dirty:
        print("  DIRTY PROVENANCE (produced from a modified working tree):")
        for d in rep.dirty:
            print(f"    ! {d}")
        print()

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

    missing_fatal = rep.missing and not args.allow_missing
    ok = not rep.findings and not rep.stale and not rep.dirty and not missing_fatal
    if ok:
        print(f"  OK — {rep.checked} figures match their artifacts.\n")
    else:
        print(f"  {len(rep.findings)} mismatch(es), {len(rep.stale)} stale, "
              f"{len(rep.dirty)} dirty, {len(rep.missing)} missing artifact(s).\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
