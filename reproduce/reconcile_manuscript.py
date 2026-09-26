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
from reproduce.omnibus_holm import collect, omnibus  # noqa: E402
from reproduce import engine_regimes as _regimes  # noqa: E402
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
        alt = (ROOT / "data" / "benchmarks") / name
        if alt.exists():
            return json.loads(alt.read_text())
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
    # The per-scenario table moved to the supplement when the body was cut to
    # length; the body keeps only regime subtotals. Check wherever it lives.
    # Anchored on the table's own label: S5's generative-parameter table opens
    # with the same row name, and an unanchored search silently matched it --
    # nine-column rows became five-column rows and every check fell through.
    tex = _supp()
    rows = _rows(tex, r"\textbf{Autonomous Vehicle (AV)}",
                 after_label=r"\label{tab:supp-corpus}")
    if not rows:
        tex = _tex("sec6_experimental_setup.tex")
        rows = _rows(tex, r"\textbf{Autonomous Vehicle (AV)}")
    name_to_file = {
        "Autonomous Vehicle (AV)": "av_system", "Enterprise Pub-Sub": "enterprise_system",
        "Financial Trading": "financial_trading_system", "Healthcare Integration": "healthcare_system",
        "Enterprise Integration (ESB)": "hub_and_spoke_system",
        "Hub-and-Spoke Enterprise": "hub_and_spoke_system", "IoT Smart City": "iot_smart_city_system",
        "Microservices Mesh": "microservices_system", "Telecom RAN": "telecom_ran_system",
        "Industrial SCADA": "industrial_scada_system", "Real-Time Gaming": "realtime_gaming_system",
        "Logistics Fleet": "logistics_fleet_system", "Air Traffic Management (ATM)": "atm_system",
        "Autoware.universe": "realworld_autoware_ros2", "Cloud Microservices": "realworld_cloud_microservices",
        "Online Boutique": "realworld_cloud_microservices",
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
    "Enterprise Integration (ESB)": "hub_and_spoke_system",
    "Hub-and-Spoke": "hub_and_spoke_system",
    "Industrial SCADA": "industrial_scada_system",
    "IoT Smart City": "iot_smart_city_system",
    "Logistics Fleet": "logistics_fleet_system",
    "Microservices": "microservices_system",
    "Microservices (synthetic)": "microservices_system",
    "Real-Time Gaming": "realtime_gaming_system",
    "Telecom RAN": "telecom_ran_system",
}

#: Table 5's column order, as variant ids. Labels come from the registry under
#: the in-distribution harness rather than being hand-copied, for the same
#: reason LOSO_LABELS does: a hand-written mirror going stale is how a table
#: gets reconciled against the wrong column while still reporting clean.
TABLE5_VARIANTS = ["topo_baseline", "topo_qos", "gl", "gl_qos", "hgl", "hgl_qos"]


def check_table5_indist(rep: Report, artifact: str = "main_table.json") -> None:
    """The in-distribution per-scenario cells and column means.

    This table was moved from the body to the supplement during the revision
    that added the protocol-matched real-world arm: its columns are not
    comparable to each other, so it documents per-scenario fitting rather than
    supporting a contrast. The check follows it rather than lapsing into a
    silent skip, which is the failure mode ``tests/test_reconcile_guards.py``
    exists to catch.
    """
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

    tex = _supp()
    rows = _rows(tex, r"\textbf{Scenario} & \textbf{$n$}",
                 after_label=r"\label{tab:supp-indist-cells}")
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
        is_dual = len(cells) >= 14
        for idx, vid in enumerate(variants):
            blk = agg.get(f"{scen}|{vid}")
            if blk is None:
                rep.skipped.append(f"tab:5: no aggregate for {scen}|{vid}")
                continue
            rho_col = (2 + 2 * idx) if is_dual else (2 + idx)
            got, truth = _num(cells[rho_col]), blk.get("mean_rho")
            rep.checked += 1
            if truth is not None and (got is None or abs(got - truth) > 0.001):
                rep.findings.append(
                    Finding("tab:5", f"{label} / {labels[vid]}", "mean_rho",
                            got, round(truth, 4)))
            if is_dual:
                f1_col = rho_col + 1
                got_f1, truth_f1 = _num(cells[f1_col]), blk.get("mean_f1")
                rep.checked += 1
                if truth_f1 is not None and (got_f1 is None or abs(got_f1 - truth_f1) > 0.001):
                    rep.findings.append(
                        Finding("tab:5", f"{label} / {labels[vid]}", "mean_f1",
                                got_f1, round(truth_f1, 4)))

    missing = set(TABLE5_SCENARIOS.values()) - set(seen)
    if missing:
        rep.skipped.append(f"tab:5: rows not found in the tex for {sorted(missing)}")

    # Column means, recomputed over exactly the scenarios the artifact ran.
    scenarios = cfg.get("scenarios") or sorted(TABLE5_SCENARIOS.values())
    try:
        t5_start = tex.index(r"\label{tab:5}")
        t5_end = tex.index(r"\end{table}", t5_start)
        tab5_body = tex[t5_start:t5_end]
    except ValueError:
        tab5_body = tex
    mean_rows = [line.strip() for line in tab5_body.split("\n")
                 if line.strip().startswith(r"\textbf{Mean}") and line.strip().endswith(r"\\")]
    if not mean_rows:
        rep.skipped.append("tab:5: Mean row not found")
        return
    cells = _cells(mean_rows[0])
    is_dual = len(cells) >= 14
    for idx, vid in enumerate(variants):
        vals = [agg[f"{sc}|{vid}"]["mean_rho"] for sc in scenarios
                if f"{sc}|{vid}" in agg and agg[f"{sc}|{vid}"].get("mean_rho") is not None]
        if not vals:
            continue
        truth = sum(vals) / len(vals)
        rho_col = (2 + 2 * idx) if is_dual else (2 + idx)
        got = _num(cells[rho_col]) if rho_col < len(cells) else None
        rep.checked += 1
        if got is None or abs(got - truth) > 0.001:
            rep.findings.append(
                Finding("tab:5", f"Mean / {labels[vid]}", "mean_rho", got, round(truth, 4)))
        if is_dual:
            vals_f1 = [agg[f"{sc}|{vid}"]["mean_f1"] for sc in scenarios
                       if f"{sc}|{vid}" in agg and agg[f"{sc}|{vid}"].get("mean_f1") is not None]
            if vals_f1:
                truth_f1 = sum(vals_f1) / len(vals_f1)
                f1_col = rho_col + 1
                got_f1 = _num(cells[f1_col]) if f1_col < len(cells) else None
                rep.checked += 1
                if got_f1 is None or abs(got_f1 - truth_f1) > 0.001:
                    rep.findings.append(
                        Finding("tab:5", f"Mean / {labels[vid]}", "mean_f1", got_f1, round(truth_f1, 4)))


def check_table5_columns(rep: Report, artifact: str = "main_table.json") -> None:
    """Table 5's printed column headers against the variants the artifact ran.

    Position-wise cell checks cannot catch a mislabelled column: every value
    matches, because the checker reads column 3 and the artifact's third variant
    and they agree -- on the wrong variant. That is not hypothetical. Table 5 was
    once regenerated from a run of ``gl``/``gl_qos`` (which
    ``reproduce/main_table.py`` puts on the DEPENDS_ON projection) while the
    headers still read GAT-N / GAT-N-QoS, the native-substrate pair. The table
    reconciled cleanly and its caption asserted substrate parity that did not
    hold, which silently reopened the RQ2 in-distribution confound.

    So: read the variant ids out of the artifact's own ``config``, ask the
    registry what those are called under the in-distribution harness, and require
    the header row to say exactly that.
    """
    d = _load(artifact)
    if d is None:
        rep.skipped.append(f"tab:5 columns: {artifact} absent")
        return
    ran = (d.get("config") or {}).get("variants") or []
    if not ran:
        rep.skipped.append(f"tab:5 columns: {artifact} records no variant list")
        return

    tex = _supp()
    header = _rows(tex, r"\toprule", after_label=r"\label{tab:supp-indist-cells}")
    if not header:
        m = re.search(r"\\textbf\{Scenario\} & \\textbf\{\$n\$\}([^\\]*(?:\\(?!\\)[^\\]*)*)",
                      tex)
        header = [m.group(0)] if m else []
    if not header:
        rep.skipped.append("tab:5 columns: header row not found")
        return
    raw_cells = _cells(header[0])
    printed = []
    for c in raw_cells:
        clean = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", c)
        clean = _label(clean)
        if clean and clean not in ("Scenario", "$n$", "n"):
            printed.append(clean)


    # Order comes from TABLE5_VARIANTS -- the same constant the cell check reads
    # columns by -- restricted to the variants this artifact actually ran. Taking
    # it from `config.variants` instead made this check fail whenever the sweep
    # happened to dispatch in a different order than the table prints, which is a
    # presentation choice and not a mislabelling. Membership is still enforced:
    # a variant that ran but is not printed, or vice versa, still fails below.
    if set(ran) != set(v for v in TABLE5_VARIANTS if v in ran):
        rep.findings.append(
            Finding("tab:5", "column set", "variants run vs printable",
                    " | ".join(sorted(ran)), " | ".join(TABLE5_VARIANTS)))
    expected = []
    for v in TABLE5_VARIANTS:
        if v not in ran:
            continue
        try:
            expected.append(_registry.label(v, harness="in_distribution"))
        except Exception:
            expected.append(v)

    rep.checked += 1
    if printed[:len(expected)] != expected:
        rep.findings.append(
            Finding("tab:5", "column headers", "variant labels",
                    " | ".join(printed[:len(expected)]) or "(none)",
                    " | ".join(expected)))


def check_supplement_stratification(rep: Report) -> None:
    """Supplement S6/S7's copies of the 7.3.6 stratification figures.

    The supplement restates body numbers as literal text -- it cannot \ref into
    the body, since the two documents do not share an .aux. Nothing checked it,
    and when detection_validation was refreshed the body was corrected while
    both supplement copies kept the superseded pooled rho and degree comparison.
    """
    art = (_load("detection_validation_jss12.json")
           or _load("detection_validation_v4.json"))
    if art is None:
        rep.skipped.append("detection_validation_v*.json absent; S6/S7 unchecked")
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
    tex = _supp()  # the registered GPU sweep, Supplementary S30
    rows = _rows(tex, r"\multicolumn{8}{l}{\textit{Training-free structural baselines}}")
    if not rows:
        # _rows returns [] for a marker it cannot find, which would otherwise
        # report as a clean run over rows nobody checked -- the exact silent
        # pass this script exists to prevent. The marker carries the column
        # count, so it moves whenever a column is added to Table 7.
        rep.skipped.append("tab:7: group marker not found in supplementary.tex "
                           "(did the column count change?)")
        return
    by_label = {LOSO_LABELS[k]: v for k, v in ct.items() if k in LOSO_LABELS}
    for row in rows:
        cells = _cells(row)
        if len(cells) < 8:
            continue
        label = _label(cells[0])
        blk = by_label.get(label) or by_label.get(label.replace("RM / Q(v)", "RM / $Q(v)$"))
        if blk is None:
            continue
        # Column order: label | mean rho | CI | delta vs baseline | fold sd |
        # seed sd | F1@K | requires training.
        for idx, key, tol in ((1, "mean_rho", 0.001), (6, "mean_f1", 0.001)):
            got, truth = _num(cells[idx]), blk.get(key)
            rep.checked += 1
            if truth is not None and (got is None or abs(got - truth) > tol):
                rep.findings.append(Finding("tab:7", label, key, got, round(truth, 4)))


def check_table7_delta(rep: Report, artifact: str = "loso_significance_v5.json") -> None:
    """Table 7's Δρ column against the pre-registered contrasts that license it.

    The column previously rendered in the artifact tables was "Δρ vs best
    baseline", computed as ``max`` over every other row -- which selected a
    *learned* variant and reported +0.0346 where the pre-registered contrast
    against Topo-QoS is +0.0851 with an interval that includes zero. Checking the
    manuscript's column against the significance artifact, rather than against
    the variants artifact it sits next to, is what pins the comparator.
    """
    d = _load(artifact)
    if d is None:
        rep.skipped.append(f"tab:7 Δρ: {artifact} absent")
        return
    truth = {}
    for section in ("preregistered", "exploratory"):
        for r in d.get(section) or []:
            if r.get("baseline") == "topo_qos" and r.get("variant") in LOSO_LABELS:
                truth[LOSO_LABELS[r["variant"]]] = r.get("mean_delta")
    tex = _supp()  # the registered GPU sweep, Supplementary S30
    rows = _rows(tex, r"\multicolumn{8}{l}{\textit{Training-free structural baselines}}")
    for row in rows:
        cells = _cells(row)
        if len(cells) < 8:
            continue
        label = _label(cells[0])
        want = truth.get(label) or truth.get(label.replace("RM / Q(v)", "RM / $Q(v)$"))
        if want is None:
            # The baseline row itself has no delta against itself; its cell reads
            # "(reference)" and there is nothing to check.
            continue
        got = _num(cells[3])
        rep.checked += 1
        if got is None or abs(got - want) > 0.001:
            rep.findings.append(Finding("tab:7", label, "delta_vs_topo_qos", got, round(want, 4)))


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
    tex = _supp()  # moved to the supplement (S25-S28)
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


def _norm(cell: str) -> str:
    """Row label reduced to comparable words: LaTeX stripped, case folded.

    Table 8's labels carry math and bold markup, so a literal comparison against
    the rowmap silently matches nothing -- and a check that matches nothing
    reports success. Normalising both sides makes a relabelled row fail loudly.
    """
    t = re.sub(r"\\[a-zA-Z]+", " ", _label(cell))
    t = re.sub(r"[^a-z0-9]+", " ", t.lower())
    return " ".join(t.split())


def check_contrasts(rep: Report, artifact: str = "loso_significance_v5.json") -> None:
    """Table 8's factorial and simple-effect rows against the significance artifact.

    These were hand-copied prose figures until now -- PROSE_NOTES listed the
    Wilcoxon contrasts as unchecked, which is precisely the class of drift that
    once left three tables reporting a superseded run. Table 8 carries the
    paper's headline claim, so it is the last table that should go unverified.
    """
    d = _load(artifact)
    if d is None:
        rep.skipped.append(f"tab:contrasts: {artifact} absent")
        return

    truth = {}
    for r in d.get("factorial") or []:
        truth[r["quantity"]] = r
    for r in d.get("architecture") or []:
        truth[f'{r["variant"]}|{r["baseline"]}'] = r

    # Row label (first cell) -> key in `truth`. Simple effects are keyed by the
    # variant pair so a relabelled row cannot silently match the wrong contrast.
    rowmap = {
        "Typing (main effect)":            "main_typing",
        "QoS channel (main effect)":       "main_qos",
        "Typing $\\times$ QoS interaction": "interaction",
        "Typing, QoS absent":              "hgl|gl",
        "Typing, QoS present":             "hgl_qos|gl_qos",
        "QoS channel, typing absent":      "gl_qos|gl",
        "QoS channel, typing present":     "hgl_qos|hgl",
    }

    tex = _supp()  # moved to the supplement (S25-S28)
    rows = _rows(tex, r"\multicolumn{7}{l}{\textit{The $2\times2$",
                 after_label=r"\label{tab:contrasts}")
    seen = 0
    for row in rows:
        cells = _cells(row)
        if len(cells) < 6:
            continue
        label = _norm(cells[0])
        key = next((v for k, v in rowmap.items() if _norm(k) == label), None)
        blk = truth.get(key) if key else None
        if blk is None:
            continue
        seen += 1
        for idx, field, tol, nm in ((2, "mean_delta", 0.001, "delta_rho"),
                                    (4, "W", 0.05, "W"),
                                    (5, "p", 0.0001, "p")):
            got, t = _num(cells[idx]), blk.get(field)
            rep.checked += 1
            if t is not None and (got is None or abs(got - t) > tol):
                rep.findings.append(Finding("tab:contrasts", label, nm, got, round(t, 4)))
        if "p_holm" in blk and len(cells) > 6:
            got = _num(cells[6])
            rep.checked += 1
            if got is not None and abs(got - blk["p_holm"]) > 0.0001:
                rep.findings.append(
                    Finding("tab:contrasts", label, "p_holm", got, round(blk["p_holm"], 4)))
    if seen == 0:
        rep.skipped.append("tab:contrasts: no rows matched the significance artifact")


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
    """Table tab:9b: per-system zero-shot correlations of the training-free and learned engines.

    Column order: system | |V_app| | n_>0 | Topo | Topo-QoS | HGT-QoS rho (+/- seeds) |
    GAT-QoS rho (+/- seeds) | HGT-QoS rho_>0. The table previously had eight columns
    while this check skipped any row shorter than nine, so none of its rows was ever
    checked; a check that matches nothing now reports a skip instead.
    """
    d = (_load("realworld_zeroshot_v7.json") or _load("realworld_zeroshot_v6.json")
         or _load("realworld_zeroshot_v5.json") or _load("realworld_zeroshot.json"))
    gat = _load("realworld_zeroshot_gl_full_qos16_cap_cpu.json")
    if d is None or gat is None:
        rep.skipped.append("tab:9b: realworld zero-shot artifacts absent")
        return
    per, per_gat, refs = d["per_system"], gat["per_system"], d.get("references", {})
    name_to_key = {
        "Online Boutique": "realworld_cloud_microservices",
        "Train-Ticket": "realworld_trainticket",
        "Autoware.universe": "realworld_autoware_ros2",
        "EdgeX Foundry": "realworld_edgex",
        "Home Assistant": "realworld_homeassistant",
    }
    tex = _tex("sec7_results.tex")
    seen = 0
    for row in _rows(tex, r"\midrule", after_label=r"\label{tab:9b}"):
        cells = _cells(row)
        if len(cells) != 8:
            continue
        label = _label(cells[0])
        key = next((v for k, v in name_to_key.items() if label.startswith(k)), None)
        if key is None or key not in per:
            continue
        seen += 1
        checks = (
            (2, per[key].get("n_positive"), 0.5, "n_positive"),
            (3, refs.get("Topo", {}).get(key, {}).get("rho"), 0.002, "topo_rho"),
            (4, refs.get("Topo-QoS", {}).get(key, {}).get("rho"), 0.002, "topoqos_rho"),
            (5, per[key].get("mean_rho"), 0.002, "hgt_qos_rho"),
            (6, per_gat.get(key, {}).get("mean_rho"), 0.002, "gat_qos_rho"),
            (7, per[key].get("mean_rho_positive"), 0.002, "hgt_qos_rho_positive"),
        )
        for idx, truth, tol, nm in checks:
            got = _num(cells[idx])
            rep.checked += 1
            if truth is None or got is None or abs(got - truth) > tol:
                rep.findings.append(Finding("tab:9b", label, nm, got,
                                            None if truth is None else round(truth, 4)))
    if seen == 0:
        rep.skipped.append("tab:9b: no row matched; the table moved or its columns changed")


def check_table9c_active(rep: Report) -> None:
    """Table 9c: the real-world active stratum, for every predictor.

    Table 9b reports ``rho_>0`` for the learned model alone, which leaves its
    headline decline with nothing to be a decline *relative to*. The training-free
    references are scored on the same labels, population and node set, so their
    active-stratum figures are computable from the same artifact -- they were
    simply absent from the artifact version that shipped. This table carries the
    comparison and is checked against the same file Table 9b is, so the two
    cannot come from different runs.
    """
    d = (_load("realworld_zeroshot_v7.json") or _load("realworld_zeroshot_v6.json")
         or _load("realworld_zeroshot_v5.json") or _load("realworld_zeroshot.json"))
    if d is None:
        rep.skipped.append("tab:9c: realworld_zeroshot.json absent")
        return
    tex = _supp()  # moved to the supplement (S25-S28)
    if r"\label{tab:9c}" not in tex:
        rep.skipped.append("tab:9c: table not present in sec7_results.tex")
        return
    per, refs = d["per_system"], d.get("references", {})
    systems = sorted(per)

    def _mean(vals: List[Optional[float]]) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if len(vals) == len(systems) else None

    learned = _registry.label(d.get("variant", "hgl_qos"), harness="loso")
    truth = {
        learned: (_mean([per[s].get("mean_rho") for s in systems]),
                  _mean([per[s].get("mean_rho_positive") for s in systems])),
    }
    for name, block in refs.items():
        truth[name] = (_mean([(block.get(s) or {}).get("rho") for s in systems]),
                       _mean([(block.get(s) or {}).get("rho_positive") for s in systems]))

    rows = _rows(tex, r"\textbf{RM / $Q(v)$}", after_label=r"\label{tab:9c}")
    if not rows:
        rep.skipped.append("tab:9c: no rows matched in sec7_results.tex")
        return
    # The manuscript spells the diagnostic row with its maths ("RM / $Q(v)$");
    # the artifact keys the same predictor "RM". Resolving that by hand rather
    # than by a `.replace` that quietly no-ops: an unresolved row is reported,
    # never skipped, because a row nobody checks reads as a row that passed.
    row_to_key = {"RM / $Q(v)$": "RM"}
    for row in rows:
        cells = _cells(row)
        if len(cells) < 3:
            continue
        label = _label(cells[0])
        pair = truth.get(row_to_key.get(label, label))
        if pair is None:
            rep.skipped.append(f"tab:9c: row {label!r} matches no predictor in the artifact")
            continue
        for idx, want, nm in ((1, pair[0], "rho"), (2, pair[1], "rho_positive")):
            got = _num(cells[idx])
            rep.checked += 1
            if want is not None and (got is None or abs(got - want) > 0.002):
                rep.findings.append(Finding("tab:9c", label, nm, got, round(want, 4)))


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
    "loso_all_variants_v5.json": "Tables 7/7c",
    "realworld_zeroshot_v7.json": "Tables 9b/9c (protocol-matched)",
    "detection_validation_jss12.json": "7.3 stratification",
    # S1.2 cited this by filename while nothing checked it and the bundle never
    # shipped it; it also ran on a different scenario suite than the section it
    # backs, which is exactly what an undeclared backing artifact hides.
    "icomp_sensitivity_jss12.json": "Supplementary S1.2 sensitivity sweep",
    "convergent_validity.json": "Supplementary S9",
    "label_stability.json": "7.1 label-noise ceiling",
    "topic_weight_sensitivity_v3.json": "Supplementary S1",
    "weight_global_sensitivity_v3.json": "Supplementary S1",
    "ahp_shrinkage_sweep_v3.json": "Supplementary S1",
    "threshold_sensitivity_v3.json": "Supplementary S3",
    "atm_scale_sweep_v3.json": "Supplementary S6",
    "qos_label_ablation.json": "Section 4.3",
    "loso_significance_v5.json": "Table 8 contrasts",
    # Hybrid-HGT (PREREGISTRATION.md Amendment 5): Table 13 and Supplementary S21-S23.
    "loso_hybrid_cpu.json": "Table 13 LOSO (hybrid CPU sweep)",
    "loso_significance_hybrid_cpu.json": "Table 13 contrasts",
    "realworld_zeroshot_hgl_qos_cpu.json": "Table 13 system models (HGT-QoS, CPU)",
    "realworld_zeroshot_hgl_qos_prior_cpu.json": "Table 13 system models (Hybrid-HGT)",
    "topo_ap_sensitivity.json": "Section 6.2.1 / Supplementary S22",
    "factorial_seed_robustness_v5.json": "Section 7.2 / Supplementary S21",
    # Amendment 2's capacity- and channel-matched 2x2 and its zero-shot arm.
    "loso_rq2_matched.json": "Table 11 matched 2x2 (CPU sweep)",
    "loso_significance_rq2_matched.json": "Table 11 matched contrasts",
    "realworld_zeroshot_gl_full_qos16_cap_cpu.json": "Section 7.4 (GAT-QoS transfer)",
    # Amendment 6: Hybrid-GAT.
    "loso_hybrid_gat_cpu.json": "Table 14 Hybrid-GAT LOSO (CPU sweep)",
    "loso_significance_hybrid_gat_cpu.json": "Table 14 Hybrid-GAT contrasts",
    "realworld_zeroshot_gl_qos16_prior_cpu.json": "Table 14 Hybrid-GAT system models",
    # Holm over every registered contrast of the plan and its amendments.
    "omnibus_registered_holm.json": "Section 6.3 / Supplementary S24 omnibus correction",
    # Where graph learning helps (post hoc): Section 8.2 and Supplementary Engine Regimes.
    "engine_regimes.json": "Section 8.2 / Supplementary Engine Regimes",
    # Amendments 7, 9 and 10: dependency counts, dependency-graph learners, derivation.
    "tf_baselines.json": "Table 7 InDeg / Reach rows; Supplementary Amendment 7",
    "loso_dependency_graph_cpu.json": "Table 7 / tab:dg-learners (Amendment 9 LOSO)",
    "dependency_graph_contrasts.json": "tab:dg-learners, Section 8.2, Supplementary Amendment 9",
    "derivation_ablation.json": "Section 7.1 / Supplementary Amendment 10",
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
#: harmful. ``oracle_timing_v*.json`` measures how long the labeler takes over a
#: node population the corpus change left identical (39/104/360/... before and
#: after), so it is not corpus-stale in any meaningful sense; what its numbers
#: depend on is the machine and its load. Worse, its ratio pairs an oracle time
#: it measures itself against a gate time it reads from
#: the timed detection run. Refreshing one half alone silently produces a ratio
#: from two different measurement sessions --- doing exactly that moved it from
#: 11.45x to 15.35x with no change to the work being timed. Re-measure the pair
#: together, on one machine, or leave both; ``oracle_timing.py --gate-file``
#: names the half it was paired with, and the artifact records it.
PAIRED_TIMING_ARTIFACTS = {
    "gate_oracle_ratio.json": "Table gate_ratio (Section 7.5)",
    "oracle_timing_jss12.json": "detection_validation_timed_jss12.json",
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
            alt = (ROOT / "data" / "benchmarks") / name
            if alt.exists():
                p = alt
            else:
                rep.missing.append(f"{name} ({backs}) is not in results/ or data/benchmarks/")
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
    art = _load("oracle_timing_jss12.json") or _load("oracle_timing_v5.json")
    if art is None:
        rep.skipped.append("oracle_timing_v*.json absent; 7.5.1 unchecked")
        return
    tex = _supp()  # moved to the supplement (S25-S28)
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
        words = {8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
                 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
                 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}
        expected = words.get(round(ratio))
        rep.checked += 1
        if expected is None or f"roughly {expected} times" not in tex:
            rep.findings.append(Finding("sec:7.5.oracle", "gate/oracle", "ratio",
                                        "see text", f"{ratio}x -> 'roughly {expected} times'"))


def check_gate_ratio_table(rep: Report) -> None:
    """Table `tab:gate_ratio`: the per-scenario gate/oracle cost distribution.

    Section 7.5 previously reported this comparison as one number taken at the
    joint maximum. The table states the distribution behind it, so each row has
    to agree with the pairing artifact rather than with a remembered figure.
    """
    d = _load("gate_oracle_ratio.json")
    if d is None:
        rep.skipped.append("tab:gate_ratio: gate_oracle_ratio.json absent")
        return
    by_scenario = {r["scenario"]: r for r in d.get("per_scenario", [])}
    name_to_key = {
        "Enterprise": "enterprise_system",
        "Enterprise Integration (ESB)": "hub_and_spoke_system",
        "AV System": "av_system",
        "Financial Trading": "financial_trading_system",
        "Real-Time Gaming": "realtime_gaming_system",
        "Healthcare": "healthcare_system",
        "IoT Smart City": "iot_smart_city_system",
        "Telecom RAN": "telecom_ran_system",
        "Logistics Fleet": "logistics_fleet_system",
        "Microservices (synthetic)": "microservices_system",
        "Industrial SCADA": "industrial_scada_system",
        "ATM System": "atm_system",
    }
    tex = _supp()  # moved to the supplement (S25-S28)
    rows = _rows(tex, r"\midrule", after_label=r"\label{tab:gate_ratio}")
    seen = 0
    for row in rows:
        cells = _cells(row)
        if len(cells) < 5:
            continue
        key = name_to_key.get(_label(cells[0]))
        if key is None or key not in by_scenario:
            continue
        seen += 1
        truth = by_scenario[key]
        for col, field, tol in ((1, "projection_edges", 0.5), (2, "gate_s", 0.02),
                                (3, "oracle_s", 0.002), (4, "ratio", 0.05)):
            got, want = _num(cells[col]), truth.get(field)
            rep.checked += 1
            if want is not None and (got is None or abs(got - want) > tol):
                rep.findings.append(
                    Finding("tab:gate_ratio", _label(cells[0]), field, got, want))
    if seen == 0:
        rep.skipped.append("tab:gate_ratio: no row matched; the table moved or was renamed")


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
         r"(?:mean Spearman \$\\rho = ([\d.]+)\$ against the ladder|leaves the Application ordering largely intact \(\$\\rho = ([\d.]+)\$ across the twelve folds\))", 0.002),
        ("wt", "spearman_rho_vs_none",
         r"(?:durability-aware \$w\(t\)\$ scaling moves it less still \(\$\\rho = ([\d.]+)\$\)|substituting durability-aware rescaling moves it less still \(\$\\rho = ([\d.]+)[\$;])", 0.002),
    ):
        expected = _mean(arm, key)
        if expected is None:
            continue
        rep.checked += 1
        m = re.search(pattern, tex)
        if not m:
            rep.findings.append(Finding("sec:4.3", f"{arm} {key}", "value",
                                        "not found", round(expected, 4)))
        else:
            val = next(float(g) for g in m.groups() if g is not None)
            if abs(val - expected) > tol:
                rep.findings.append(Finding("sec:4.3", f"{arm} {key}", "value",
                                            val, round(expected, 4)))


PROSE_NOTES = [
    "QoS-channel seed spreads in sec:rq2 <- loso_all_variants_v*.json",
    "label-noise ceiling in sec:rq1 <- output/loso_cache/*/failure_impact.json label_stability",
    "gate range in sec:rq4 <- results/detection_validation_timed_jss12.json gate_seconds",
]


def check_hybrid_table(rep: Report) -> None:
    """Table tab:hybrid: SaG-Hybrid and its same-invocation CPU comparators.

    Every LOSO cell comes from one CPU sweep (``loso_hybrid_cpu.json``) and its
    significance artifact; the system-model columns come from the two CPU
    zero-shot artifacts. Checked here so the hybrid table cannot drift from
    the runs registered under PREREGISTRATION.md Amendment 5.
    """
    loso = _load("loso_hybrid_cpu.json")
    sig = _load("loso_significance_hybrid_cpu.json")
    # Amendment 6's sweep supplies the untyped rows; its topo_qos and
    # gl_full_qos16_cap cells are bit-identical to the other CPU sweeps.
    loso_gat = _load("loso_hybrid_gat_cpu.json") or {}
    sig_gat = _load("loso_significance_hybrid_gat_cpu.json") or {}
    rw = {v: _load(f"realworld_zeroshot_{v}_cpu.json")
          for v in ("hgl_qos", "hgl_qos_prior", "gl_full_qos16_cap", "gl_qos16_prior")}
    tex = _tex("sec7_results.tex")
    if loso is None or sig is None or r"\label{tab:hybrid}" not in tex:
        rep.skipped.append("tab:hybrid: hybrid artifacts or table absent")
        return
    table = {**loso_gat.get("comparison_table", {}), **loso["comparison_table"]}
    deltas = {r["variant"]: r for r in sig_gat.get("exploratory", [])
              + sig.get("exploratory", []) + sig.get("preregistered", [])}
    ref = next(iter(r for r in rw.values() if r), None) or {}
    boot = ref.get("bootstrap_ci", {})
    rw_rho = {
        "topo_baseline": boot.get("Topo", {}).get("rho", {}).get("mean"),
        "topo_qos": boot.get("Topo-QoS", {}).get("rho", {}).get("mean"),
        "hgl_qos": (rw["hgl_qos"] or {}).get("mean_rho_across_systems"),
        "hgl_qos_prior": (rw["hgl_qos_prior"] or {}).get("mean_rho_across_systems"),
        "gl_full_qos16_cap": (rw["gl_full_qos16_cap"] or {}).get("mean_rho_across_systems"),
        "gl_qos16_prior": (rw["gl_qos16_prior"] or {}).get("mean_rho_across_systems"),
    }
    labels = {_registry.label(v, "loso"): v
              for v in ("topo_baseline", "topo_qos", "hgl_qos", "hgl_qos_prior",
                        "gl_full_qos16_cap", "gl_qos16_prior")}
    for row in _rows(tex, r"\midrule", after_label=r"\label{tab:hybrid}"):
        cells = _cells(row)
        v = labels.get(_label(cells[0]))
        if v is None or v not in table:
            continue
        checks = [(1, table[v]["mean_rho"], "mean_rho"), (6, table[v]["mean_f1"], "overlap_at_k")]
        if v in deltas:
            checks.append((3, deltas[v]["mean_delta"], "delta_vs_topo_qos"))
        for idx, truth, nm in checks:
            if truth is None:
                continue
            got = _num(cells[idx]) if idx < len(cells) else None
            rep.checked += 1
            if got is None or abs(got - truth) > 0.001:
                rep.findings.append(Finding("tab:hybrid", _label(cells[0]), nm, got, round(truth, 4)))


def check_system_models_transfer(rep: Report) -> None:
    """Table tab:system_models_transfer / tab:9b: Zero-shot transfer to system models."""
    tex = _tex("sec7_results.tex")
    if r"\label{tab:system_models_transfer}" not in tex and r"\label{tab:9b}" not in tex:
        rep.skipped.append("tab:system_models_transfer absent")
        return
    import numpy as np
    tf = _load("tf_baselines.json")
    rw = {v: _load(f"realworld_zeroshot_{v}_cpu.json")
          for v in ("hgl_qos", "hgl_qos_prior", "gl_full_qos16_cap", "gl_qos16_prior")}
    z_proj = _load("realworld_zeroshot_gl_proj_qos16_cap_dependency_graph.json")
    if None in (tf, z_proj) or any(v is None for v in rw.values()):
        rep.skipped.append("system models artifacts absent")
        return
    ref = rw["hgl_qos"]
    boot = ref.get("bootstrap_ci", {})
    expected = {
        "Topo": (boot.get("Topo", {}).get("rho", {}).get("mean"), boot.get("Topo", {}).get("pr_auc", {}).get("mean")),
        "Topo-QoS": (boot.get("Topo-QoS", {}).get("rho", {}).get("mean"), boot.get("Topo-QoS", {}).get("pr_auc", {}).get("mean")),
        "Reach": (tf["summary"]["Reach"]["systems_mean_rho"], tf["summary"]["Reach"]["systems_mean_pr_auc"]),
        "InDeg": (tf["summary"]["InDeg"]["systems_mean_rho"], tf["summary"]["InDeg"]["systems_mean_pr_auc"]),
        "HGT-QoS": (rw["hgl_qos"]["mean_rho_across_systems"], float(np.mean([x["mean_pr_auc"] for x in rw["hgl_qos"]["per_system"].values()]))),
        "GAT-QoS": (rw["gl_full_qos16_cap"]["mean_rho_across_systems"], float(np.mean([x["mean_pr_auc"] for x in rw["gl_full_qos16_cap"]["per_system"].values()]))),
        "Hybrid-HGT": (rw["hgl_qos_prior"]["mean_rho_across_systems"], float(np.mean([x["mean_pr_auc"] for x in rw["hgl_qos_prior"]["per_system"].values()]))),
        "Hybrid-GAT": (rw["gl_qos16_prior"]["mean_rho_across_systems"], float(np.mean([x["mean_pr_auc"] for x in rw["gl_qos16_prior"]["per_system"].values()]))),
        "GAT-P-QoS": (z_proj["mean_rho_across_systems"], float(np.mean([x["mean_pr_auc"] for x in z_proj["per_system"].values()]))),
    }
    for row in _rows(tex, r"\midrule", after_label=r"\label{tab:system_models_transfer}"):
        cells = _cells(row)
        name = _label(cells[0]).replace("$^\\dagger$", "").replace("^\\dagger", "").strip()
        if name in expected:
            rho_truth, prauc_truth = expected[name]
            got_rho = _num(cells[2])
            got_prauc = _num(cells[4])
            rep.checked += 1
            if got_rho is None or abs(got_rho - rho_truth) > 0.001:
                rep.findings.append(Finding("tab:system_models_transfer", name, "mean_rho", got_rho, round(rho_truth, 4)))
            rep.checked += 1
            if got_prauc is None or abs(got_prauc - prauc_truth) > 0.001:
                rep.findings.append(Finding("tab:system_models_transfer", name, "pr_auc", got_prauc, round(prauc_truth, 4)))


def check_independent_oracles(rep: Report) -> None:
    """Table tab:independent_oracles: Agreement across independent simulation paradigms."""
    art = _load("independent_oracle_evaluation.json")
    if art is None:
        rep.skipped.append("independent_oracle_evaluation.json absent")
        return
    tex = _tex("sec7_results.tex")
    if r"\label{tab:independent_oracles}" not in tex:
        rep.skipped.append("tab:independent_oracles absent")
        return
    label_map = {
        "Analytic $I^*$": "Analytic-I*",
        "InDeg": "InDeg",
        "Reach": "Reach",
        "Topo-QoS": "Topo-QoS",
    }
    summ = art["summary"]
    for row in _rows(tex, r"\midrule", after_label=r"\label{tab:independent_oracles}"):
        cells = _cells(row)
        name = _label(cells[0])
        key = label_map.get(name)
        if key is None:
            continue
        truths = [
            (1, summ["i_star"][key]["mean_rho"], "i_star_mean_rho"),
            (2, summ["i_star"][key]["mean_rho_active"], "i_star_rho_active"),
            (3, summ["i_dyn"][key]["mean_rho"], "i_dyn_mean_rho"),
            (4, summ["i_dyn"][key]["mean_rho_active"], "i_dyn_rho_active"),
            (5, summ["i_comp"][key]["mean_rho"], "i_comp_mean_rho"),
            (6, summ["i_comp"][key]["mean_rho_active"], "i_comp_rho_active"),
        ]
        for idx, truth, nm in truths:
            got = _num(cells[idx]) if idx < len(cells) else None
            rep.checked += 1
            if got is None or abs(got - truth) > 0.001:
                rep.findings.append(Finding("tab:independent_oracles", name, nm, got, round(truth, 4)))


def check_contrasts_matched(rep: Report) -> None:
    """Table tab:contrasts_matched: the capacity- and channel-matched 2x2.

    The three orthogonal quantities come from the significance artifact's
    ``factorial`` block; the simple effects are recomputed from the same sweep
    with the same paired test, so the table cannot drift from the run
    registered under PREREGISTRATION.md Amendment 2.
    """
    from reproduce.loso_significance import compare

    loso = _load("loso_rq2_matched.json")
    sig = _load("loso_significance_rq2_matched.json")
    tex = _tex("sec7_results.tex")
    if loso is None or sig is None or r"\label{tab:contrasts_matched}" not in tex:
        rep.skipped.append("tab:contrasts_matched: artifacts or table absent")
        return
    table = loso["comparison_table"]
    factorial = {r["quantity"]: r for r in sig.get("factorial", [])}
    simple = {
        "Typing, QoS absent": ("hgl", "gl_full_cap"),
        "Typing, QoS present": ("hgl_qos", "gl_full_qos16_cap"),
        "QoS channel, typing absent": ("gl_full_qos16_cap", "gl_full_cap"),
        "QoS channel, typing present": ("hgl_qos", "hgl"),
    }
    rowmap = {"Typing (main effect)": "main_typing",
              "QoS channel (main effect)": "main_qos",
              "Typing $times$ QoS interaction": "interaction"}
    for row in _rows(tex, r"\midrule", after_label=r"\label{tab:contrasts_matched}"):
        cells = _cells(row)
        label = _label(cells[0])
        if label in rowmap and rowmap[label] in factorial:
            truth = factorial[rowmap[label]]
        elif label in simple:
            truth = compare(table, *simple[label])
        else:
            continue
        for idx, key, tol in ((2, "mean_delta", 0.001), (6, "p", 0.001)):
            got = _num(cells[idx]) if idx < len(cells) else None
            rep.checked += 1
            if got is None or abs(got - truth[key]) > tol:
                rep.findings.append(Finding("tab:contrasts_matched", label, key, got,
                                            round(truth[key], 4)))


#: Prose sites quoting the omnibus-adjusted p of the two hybrid primaries, as
#: (file, pattern). Each pattern captures (SaG-Hybrid, SaG-Hybrid-GAT) in that
#: order; the sites that name one engine first say so in the pattern.
OMNIBUS_PROSE = [
    ("sec7_results.tex", r"thirteen (?:registered|confirmatory) contrasts(?: of the study)? \(\$p_\{\\text\{omni\}\} = ([\d.]+)\$ and \$([\d.]+)\$"),
]


def check_omnibus_holm(rep: Report) -> None:
    """Omnibus Holm over every registered contrast (Section 6.3, Supplementary S24).

    Three things are checked. The artifact must still equal a fresh pooling of
    the significance artifacts it reads, so a re-run hybrid sweep cannot leave
    the omnibus figures behind. The supplement's table must match it row by row.
    The prose quotes in Sections 1, 6.3, 7.5 and 9 must match it at their
    printed precision.
    """
    art = _load("omnibus_registered_holm.json")
    if art is None:
        rep.skipped.append("omnibus_registered_holm.json absent; omnibus figures unchecked")
        return
    res_dir = RESULTS if (RESULTS / "loso_significance_v5.json").exists() else (ROOT / "data" / "benchmarks")
    try:
        fresh = {r["contrast"]: r for r in omnibus(collect(res_dir))}
    except (FileNotFoundError, KeyError) as exc:
        rep.skipped.append(f"omnibus: source significance artifact unreadable ({exc})")
        fresh = {}
    # The artifacts predate the 2026-09-24 relabelling; compare in current labels.
    fresh = {_registry.relabel(k): v for k, v in fresh.items()}
    by_contrast = {_registry.relabel(r["contrast"]): r for r in art["contrasts"]}
    for name, r in fresh.items():
        rep.checked += 1
        old = by_contrast.get(name)
        if old is None or abs(old["p_holm_omnibus"] - r["p_holm_omnibus"]) > 1e-9:
            rep.findings.append(Finding("omnibus", name, "p_holm_omnibus",
                                        None if old is None else old["p_holm_omnibus"],
                                        r["p_holm_omnibus"], "artifact stale"))

    def plain(tex: str) -> str:
        tex = re.sub(r"\\texttt\{([^}]*)\}", r"\1", tex)
        return tex.replace(r" $\times$ ", " x ").strip()

    supp = _supp()
    if r"\label{tab:supp-omnibus}" not in supp:
        rep.skipped.append("tab:supp-omnibus: table absent from supplementary.tex")
    else:
        i = supp.index(r"\midrule", supp.index(r"\label{tab:supp-omnibus}"))
        body = supp[i:supp.index(r"\bottomrule", i)]
        for line in body.split("\n")[1:]:
            cells = _cells(line.strip())
            if len(cells) < 6:
                continue
            truth = by_contrast.get(plain(cells[1]))
            if truth is None:
                rep.findings.append(Finding("tab:supp-omnibus", cells[1], "contrast",
                                            cells[1], None, "no such registered contrast"))
                continue
            for idx, key, tol in ((2, "mean_delta", 0.0006), (3, "p", 0.00006),
                                  (4, "p_holm_family", 0.00006), (5, "p_holm_omnibus", 0.00006)):
                got, want = _num(cells[idx]), truth[key]
                rep.checked += 1
                if got is None or abs(got - want) > tol:
                    rep.findings.append(Finding("tab:supp-omnibus", cells[1], key, got, round(want, 4)))

    hyb, gat = _registry.label("hgl_qos_prior", "loso"), _registry.label("gl_qos16_prior", "loso")
    want = {v: by_contrast[f"{v} vs Topo-QoS"]["p_holm_omnibus"] for v in (hyb, gat)}
    sec6 = _tex("sec6_experimental_setup.tex")
    m6 = re.search(re.escape(gat) + r" \$p_\{\\text\{omni\}\} = ([\d.]+)\$, "
                   + re.escape(hyb) + r" \$p_\{\\text\{omni\}\} = ([\d.]+)\$", sec6)
    found = [("sec6_experimental_setup.tex", m6.group(2), m6.group(1))] if m6 else []
    if not m6:
        rep.findings.append(Finding("omnibus prose", "sec6_experimental_setup.tex", "p_omni",
                                    "not found", None, "quote moved or reworded"))
    for f, pat in OMNIBUS_PROSE:
        m = re.search(pat, _tex(f))
        if m is None:
            rep.findings.append(Finding("omnibus prose", f, "p_omni", "not found", None,
                                        "quote moved or reworded"))
            continue
        found.append((f, m.group(1), m.group(2)))
    for f, p_hyb, p_gat in found:
        for got, v in ((p_hyb, hyb), (p_gat, gat)):
            rep.checked += 1
            decimals = len(got.split(".")[1])
            if round(want[v], decimals) != float(got):
                rep.findings.append(Finding("omnibus prose", f, v, float(got), round(want[v], 4)))


def _quote(rep: Report, where: str, text: str, pattern: str,
           truths: List[Tuple[int, float]]) -> None:
    """Check figures quoted in prose: capture group ``i`` must equal ``truth``
    at the precision it is printed with."""
    m = re.search(pattern, text, re.S)
    if m is None:
        rep.findings.append(Finding(where, pattern[:34], "quote", "not found", None,
                                    "quote moved or reworded"))
        return
    for i, truth in truths:
        raw = m.group(i).replace("+", "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        rep.checked += 1
        if abs(float(raw) - round(truth, decimals)) > 1e-9:
            rep.findings.append(Finding(where, pattern[:34], f"group {i}", float(raw), round(truth, 4)))


#: Supplement Table tab:supp-regimes-desc, column order.
REGIME_DESC_COLUMNS = ("n_apps", "n_topics", "n_brokers", "n_hosts", "n_libraries", "edges_per_node",
                       "uses_per_app", "subscriber_gini", "zero_share", "tie_fraction",
                       "projection_density")


def check_engine_regimes(rep: Report) -> None:
    """Section 8.2 and the supplement's Engine Regimes section.

    The artifact must equal a fresh ``engine_regimes.build()`` over the sweep
    artifacts it joins, so a re-run sweep cannot leave the regime figures
    behind. Then every figure quoted in Sections 7.2, 8.1 and 8.2, the regime
    table, and the four supplement tables are checked against it.
    """
    art = _load("engine_regimes.json")
    if art is None:
        rep.skipped.append("engine_regimes.json absent; Section 8.2 unchecked")
        return
    try:
        fresh = _regimes.build()
    except (OSError, KeyError, ValueError) as exc:
        rep.skipped.append(f"engine_regimes: a source artifact is unreadable ({exc})")
        fresh = None
    if fresh is not None:
        rep.checked += 1
        if json.loads(json.dumps(fresh)) != {k: v for k, v in art.items() if k != "provenance"}:
            rep.findings.append(Finding("engine_regimes", "artifact", "content", "differs",
                                        "fresh build", "artifact stale"))

    L, Z = art["loso"], art["zero_shot"]
    pf, reg, zs = L["per_fold"], L["regimes_by_topo_qos_tercile"], Z["per_system"]
    corr = {(c["quantity"], c["descriptor"]): c for c in L["correlations"]}
    cov = L["receptive_field_coverage"]
    rf = cov["hgl_qos_rf_share"].values()
    folds = L["folds"]

    def zs_range(systems, key, arms=_regimes.PURE_LEARNED):
        vals = [zs[a][s_][key] for a in arms for s_ in systems]
        return min(vals), max(vals)

    pubsub = _regimes.PUBSUB_ORIGINALS
    rpc = [s_ for s_ in Z["systems"] if s_ not in pubsub]
    ps_lo, ps_hi = zs_range(pubsub, "rho")
    ps_cf_lo, ps_cf_hi = zs_range(pubsub, "rho", ("Topo", "Topo-QoS"))
    ps_pos_lo, ps_pos_hi = zs_range(pubsub, "rho_pos")
    rpc_lo, rpc_hi = zs_range(rpc, "rho")
    rpc_pos_lo, rpc_pos_hi = zs_range(rpc, "rho_pos")
    cf_pos_max = zs_range(pubsub, "rho_pos", ("Topo", "Topo-QoS"))[1]
    rep.checked += 1
    if cf_pos_max >= 0:
        rep.findings.append(Finding("tab:regimes", "originally pub-sub", "closed-form rho>0",
                                    "negative", round(cf_pos_max, 4)))
    w = {t: reg[t]["wins_vs_topo_qos"] for t in reg}
    for t, arms in (("weak", ("HGT-QoS", "GAT-QoS")), ("middle", ("Hybrid-HGT", "Hybrid-GAT"))):
        rep.checked += 1
        if len({w[t][a] for a in arms}) != 1:
            rep.findings.append(Finding("tab:regimes", t, "'both' wins", "equal",
                                        str({a: w[t][a] for a in arms})))
    m = lambda t, a: reg[t]["mean_rho"][a]
    n = lambda t: len(reg[t]["folds"])
    num = r"\$([-+]?[\d.]+)\$"

    sec8 = _tex("sec8_discussion.tex")
    table = sec8[sec8.index(r"\label{tab:regimes}"):sec8.index(r"\end{table}", sec8.index(r"\label{tab:regimes}"))]
    _quote(rep, "tab:regimes", table,
           r"Closed-form ranks poorly.*?\\texttt\{Topo-QoS\} " + num + r"; \\texttt\{HGT-QoS\} " + num
           + r", \\texttt\{GAT-QoS\} " + num + r", both (\d+)/(\d+) folds.*?hybrids " + num + " / " + num,
           [(1, m("weak", "Topo-QoS")), (2, m("weak", "HGT-QoS")), (3, m("weak", "GAT-QoS")),
            (4, w["weak"]["HGT-QoS"]), (5, n("weak")), (6, m("weak", "Hybrid-HGT")), (7, m("weak", "Hybrid-GAT"))])
    _quote(rep, "tab:regimes", table,
           r"Intermediate.*?\\texttt\{Topo-QoS\} " + num + r"; \\texttt\{HGT-QoS\} " + num
           + r", \\texttt\{GAT-QoS\} " + num + r"; Hybrid-HGT " + num + r", Hybrid-GAT " + num
           + r", both (\d+)/(\d+) folds",
           [(1, m("middle", "Topo-QoS")), (2, m("middle", "HGT-QoS")), (3, m("middle", "GAT-QoS")),
            (4, m("middle", "Hybrid-HGT")), (5, m("middle", "Hybrid-GAT")),
            (6, w["middle"]["Hybrid-HGT"]), (7, n("middle"))])
    _quote(rep, "tab:regimes", table,
           r"Closed-form ranks well.*?\\texttt\{Topo-QoS\} " + num + r"; \\texttt\{HGT-QoS\} " + num
           + r" \((\d+)/(\d+) folds\), \\texttt\{GAT-QoS\} " + num + r" \((\d+)/(\d+)\); hybrids "
           + num + " / " + num,
           [(1, m("strong", "Topo-QoS")), (2, m("strong", "HGT-QoS")), (3, w["strong"]["HGT-QoS"]),
            (4, n("strong")), (5, m("strong", "GAT-QoS")), (6, w["strong"]["GAT-QoS"]), (7, n("strong")),
            (8, m("strong", "Hybrid-HGT")), (9, m("strong", "Hybrid-GAT"))])
    _quote(rep, "tab:regimes", table,
           r"originally pub-sub.*?Learned " + num + "--" + num + r" vs.\\ closed-form " + num + "--" + num
           + r"; learned \$\\rho_\{>0\}\$ " + num + "--" + num,
           [(1, ps_lo), (2, ps_hi), (3, ps_cf_lo), (4, ps_cf_hi), (5, ps_pos_lo), (6, ps_pos_hi)])
    _quote(rep, "tab:regimes", table,
           r"originally RPC.*?Learned " + num + "--" + num + r", but Topo " + num
           + r" on Online Boutique; learned \$\\rho_\{>0\}\$ " + num + " to " + num,
           [(1, rpc_lo), (2, rpc_hi), (3, zs["Topo"]["realworld_cloud_microservices"]["rho"]),
            (4, rpc_pos_lo), (5, rpc_pos_hi)])
    for pattern, truths in (
        (r"\\texttt\{GBM-Feat\}\) achieve \$\\rho = ([\d.]+)\$ under LOSO, level with \\texttt\{GAT-QoS\} \(\$([\d.]+)\$\)",
         [(1, L["mean_rho"]["GBM-Feat"]), (2, L["mean_rho"]["GAT-QoS"])]),
        (r"typing main effect " + num + r", interaction " + num,
         [(1, -0.014), (2, +0.001)]),
        (r"within-fold seed spread " + num + r", mean \$\\rho = ([\d.]+)\$",
         [(1, 0.208), (2, 0.514)]),
        (r"\\texttt\{GAT-P-QoS\}\) converged reliably \(\$\\rho = ([\d.]+)\$",
         [(1, 0.748)]),
    ):
        _quote(rep, "sec:8.2", sec8, pattern, truths)

    # The two registered control arms: HGT-QoS-U quoted in Section 7.2, both in the supplement.
    dirn = _load("loso_significance_directionality_cpu.json") or {}
    capa = _load("loso_significance_capacity_cpu.json") or {}
    ctl = {r["variant"] + "|" + r["baseline"]: r for a in (dirn, capa) for r in a.get("rq2_controls", [])}
    uni, capr = ctl.get("hgl_qos|hgl_qos_uni"), ctl.get("hgl_qos|gl_full_qos_cap")
    if uni is None or capr is None:
        rep.skipped.append("registered control contrasts absent; Section 7.2 control quotes unchecked")
        return
    sec7 = _tex("sec7_results.tex")
    if "The directionality control registered in Amendment~2" in sec7:
        _quote(rep, "sec:rq2", sec7,
               r"The directionality control registered in Amendment~2 agrees:.*?reaches " + num
               + r" against \\texttt\{HGT-QoS\}'s "
               + num + r" \(registered contrast \\texttt\{HGT-QoS\} vs.\\ \\texttt\{HGT-QoS-U\} " + num
               + r", \\texttt\{HGT-QoS\} ahead on (\d+)/12 folds, \$p = ([\d.]+)\$",
               [(1, L["mean_rho"]["HGT-QoS-U"]), (2, L["mean_rho"]["HGT-QoS"]),
                (3, uni["mean_delta"]), (4, uni["wins"]), (5, uni["p"])])
    _quote(rep, "supp:amendments", _supp(),
           r"reaches \$\\rho = ([\d.]+)\$ \(registered contrast \\texttt\{HGT-QoS\} vs.\\ \\texttt\{GAT-w\} " + num
           + r", \\texttt\{HGT-QoS\} ahead on (\d+)/12 folds, \$p = ([\d.]+)\$\).*?reaches " + num
           + r" \(\\texttt\{HGT-QoS\} vs.\\ \\texttt\{HGT-QoS-U\} " + num + r", (\d+)/12, \$p = ([\d.]+)\$\), "
           r"and transfers zero-shot at " + num + " against " + num,
           [(1, L["mean_rho"]["GAT-w"]), (2, capr["mean_delta"]), (3, capr["wins"]), (4, capr["p"]),
            (5, L["mean_rho"]["HGT-QoS-U"]), (6, uni["mean_delta"]), (7, uni["wins"]), (8, uni["p"]),
            (9, Z["mean_rho"]["HGT-QoS-U"]), (10, Z["mean_rho"]["HGT-QoS"])])

    # Supplement tables, cell by cell.
    supp = _supp()
    _quote(rep, "supp:regimes", supp,
           r"over the twelve folds, (\d+) correlations in all.*?the smallest is " + num
           + r".*?its Spearman correlation is " + num + r" with \\texttt\{HGT-QoS\}'s per-fold \$\\rho\$, " + num
           + r" with its gain over \\texttt\{Topo-QoS\}, and " + num,
           [(1, L["n_tests"]), (2, L["min_q_bh"]), (3, cov["HGT-QoS rho"]["rho"]),
            (4, cov["HGT-QoS - Topo-QoS"]["rho"]), (5, cov["Hybrid-HGT - HGT-QoS"]["rho"])])
    fold_of = {v: k for k, v in _regimes.FOLD_LABELS.items()}
    sys_of = {v: k for k, v in _regimes.SYSTEM_LABELS.items()}

    def body(label):
        if label not in supp:
            rep.skipped.append(f"{label} absent from supplementary.tex")
            return None, []
        i = supp.index(r"\toprule", supp.index(label))
        head = [_label(c) for c in _cells(supp[i:supp.index(r"\midrule", i)].split("\n")[1])]
        j = supp.index(r"\midrule", i)
        return head, [ln.strip() for ln in supp[j:supp.index(r"\bottomrule", j)].split("\n")[1:]]

    def cmp(where, row, col, got, want, tol=0.0006):
        rep.checked += 1
        if got is None or abs(got - want) > tol:
            rep.findings.append(Finding(where, row, col, got, round(want, 4)))

    head, rows = body(r"\label{tab:supp-regimes-folds}")
    for ln in rows:
        cells = _cells(ln)
        if len(cells) < 3:
            continue
        name = _label(cells[0])
        for col, cell in zip(head[2:], cells[2:]):
            if name in fold_of:
                cmp("tab:supp-regimes-folds", name, col, _num(cell), pf[col][fold_of[name]]["rho"])
            elif name.startswith("Mean"):
                key = "mean_rho_pos" if "rho" in name else "mean_rho"
                cmp("tab:supp-regimes-folds", name, col, _num(cell), L[key][col])
        if name in fold_of:
            rep.checked += 1
            want_t = {"weak": "W", "middle": "I", "strong": "S"}[
                next(t for t in reg if fold_of[name] in reg[t]["folds"])]
            if cells[1].strip() != want_t:
                rep.findings.append(Finding("tab:supp-regimes-folds", name, "tercile", cells[1].strip(), want_t))

    _, rows = body(r"\label{tab:supp-regimes-desc}")
    for ln in rows:
        cells = _cells(ln)
        name = _label(cells[0]) if cells else ""
        if name not in fold_of:
            continue
        f = fold_of[name]
        for key, cell in zip(REGIME_DESC_COLUMNS, cells[1:]):
            cmp("tab:supp-regimes-desc", name, key, _num(cell), L["descriptors"][f][key], tol=0.006)
        cmp("tab:supp-regimes-desc", name, "rf_share", _num(cells[-1]), cov["hgl_qos_rf_share"][f], tol=0.006)

    head, rows = body(r"\label{tab:supp-regimes-corr}")
    gains = list(dict.fromkeys(c["quantity"] for c in L["correlations"]))
    descs = list(dict.fromkeys(c["descriptor"] for c in L["correlations"]))
    data_rows = [ln for ln in rows if len(_cells(ln)) == len(gains) + 1]
    rep.checked += 1
    if len(data_rows) != len(descs):
        rep.findings.append(Finding("tab:supp-regimes-corr", "rows", "count", len(data_rows), len(descs)))
    for desc, ln in zip(descs, data_rows):
        for g, cell in zip(gains, _cells(ln)[1:]):
            c = corr[(g, desc)]
            cmp("tab:supp-regimes-corr", desc, g, _num(cell.replace("+", "")), c["rho"], tol=0.006)
            rep.checked += 1
            if ("*" in cell) != (c["p"] < 0.05):
                rep.findings.append(Finding("tab:supp-regimes-corr", desc, g + " star", "*" in cell, c["p"] < 0.05))

    head, rows = body(r"\label{tab:supp-regimes-zs}")
    key = None
    for ln in rows:
        if r"\multicolumn" in ln:
            key = "rho_pos" if "rho_{>0}" in ln else "rho"
            continue
        cells = _cells(ln)
        if len(cells) < 2 or key is None:
            continue
        name = _label(cells[0])
        for col, cell in zip(head[1:], cells[1:]):
            got = _num(cell.replace("$", ""))
            if name in sys_of:
                cmp("tab:supp-regimes-zs", name, f"{col} {key}", got, zs[col][sys_of[name]][key])
            elif name == "Mean":
                cmp("tab:supp-regimes-zs", name, f"{col} {key}", got,
                    Z["mean_rho" if key == "rho" else "mean_rho_pos"][col])


def check_dependency_graph(rep: Report) -> None:
    """The dependency-graph results (PREREGISTRATION.md Amendments 7, 9 and 10).

    Table tab:hybrid's InDeg / Reach rows (``tf_baselines.json``) and its
    dependency-graph learner rows (``dependency_graph_contrasts.json`` and the
    Amendment 9 LOSO / zero-shot artifacts); Table tab:dg-learners; the
    Amendment 10 and regime figures quoted in prose; and both supplementary
    table files, re-rendered from their artifacts and compared byte for byte.
    """
    import numpy as np
    from scipy.stats import wilcoxon

    tf, dg = _load("tf_baselines.json"), _load("dependency_graph_contrasts.json")
    dv, dl = _load("derivation_ablation.json"), _load("loso_dependency_graph_cpu.json")
    if None in (tf, dg, dv, dl):
        rep.skipped.append("dependency-graph results: an Amendment 7/9/10 artifact is absent")
        return
    tex = _tex("sec7_results.tex")
    folds = list(dg["per_fold"]["topo_qos"])
    topo = np.array([dg["per_fold"]["topo_qos"][f]["rho"] for f in folds])

    def _cmp(where: str, row: str, name: str, got, truth, tol: float = 0.0006) -> None:
        rep.checked += 1
        if got is None or abs(got - truth) > tol:
            rep.findings.append(Finding(where, row, name, got, round(truth, 4)))

    # Table tab:hybrid.
    counts = {"InDeg": "InDeg", "Reach": "Reach"}
    learners = {_registry.label(v, "loso"): v for v in ("gl_proj_qos16_cap",)}
    seen = 0
    for row in _rows(tex, r"\midrule", after_label=r"\label{tab:hybrid}"):
        cells = _cells(row)
        raw_name = _label(cells[0])
        name = raw_name.split()[0] if raw_name.startswith(("InDeg", "Reach")) else raw_name
        if name in counts:
            s_, c = tf["summary"][counts[name]], tf["contrasts_vs_topo_qos"][counts[name]]
            truths = ((1, s_["loso_mean_rho"], "mean_rho"), (3, c["delta"], "delta"),
                      (4, c["won"], "won"), (6, s_["loso_mean_overlap"], "overlap_at_k"))
        elif name in learners:
            v = learners[name]
            f = np.array([dg["per_fold"][v][k]["rho"] for k in folds])
            truths = ((1, f.mean(), "mean_rho"), (3, (f - topo).mean(), "delta"),
                      (4, int(((f - topo) > 0).sum()), "won"),
                      (5, float(wilcoxon(f, topo).pvalue), "p"),
                      (6, dl["comparison_table"][v]["mean_f1"], "overlap_at_k"))
        else:
            continue
        seen += 1
        for idx, truth, nm in truths:
            _cmp("tab:hybrid", name, nm, _num(cells[idx]) if idx < len(cells) else None, truth,
                 0.00006 if nm == "p" else 0.0006)
    if seen != 3:
        rep.findings.append(Finding("tab:hybrid", "dependency-graph rows", "rows", seen, 3))

    means, con, zs = dg["means"], dg["contrasts"], dg["zeroshot"]
    # Table tab:dg-learners (if present in main text).
    if r"\label{tab:dg-learners}" in tex:
        arms = {_registry.label(v, "loso"): v for v in
                ("gl_proj_cap", "gl_proj_qos16_cap", "gl_proj_qos16_indeg_prior", "hgl_proj_qos")}
        triple = re.compile(r"([-+]?\d\.\d+)\$?(?: vs [^(]*)? \((\d+)/12, ([\d.]+)\)")
        seen = 0
        for row in _rows(tex, r"\midrule", after_label=r"\label{tab:dg-learners}"):
            cells = _cells(row)
            v = arms.get(_label(cells[0]))
            if v is None:
                continue
            seen += 1
            lab = _registry.label(v, "loso")
            _cmp("tab:dg-learners", lab, "loso_rho", _num(cells[1]), means[v]["loso_mean_rho"])
            _cmp("tab:dg-learners", lab, "zero_shot", _num(cells[5]), zs[v]["mean_rho"])
            refs = [f"{lab} vs InDeg", f"{lab} vs Reach"] + [k for k in con if k.startswith(f"{lab} vs ")
                                                              and k.split(" vs ")[1] not in ("InDeg", "Reach")]
            for idx, key in zip((2, 3, 4), refs):
                m = triple.search(cells[idx].replace("$", "").replace(r"\mathbf{", "").replace("}", ""))
                if m is None:
                    rep.findings.append(Finding("tab:dg-learners", lab, key, "unparsed", None))
                    continue
                for got, truth, nm in ((float(m.group(1)), con[key]["delta"], "delta"),
                                       (float(m.group(2)), con[key]["won"], "won"),
                                       (float(m.group(3)), con[key]["p_holm"], "p_holm")):
                    _cmp("tab:dg-learners", key, nm, got, truth, 0.0006)
        if seen != 4:
            rep.findings.append(Finding("tab:dg-learners", "rows", "rows", seen, 4))


    # Amendment 10 and regime figures quoted in prose.
    c10, s10 = dv["contrasts"], dv["summary"]
    num = r"\$([-+]?[\d.]+)\$"
    _quote(rep, "sec:rq1", tex,
           r"without Rule~5, \\texttt\{Reach\} falls by " + num + r" \((\d+)/12 folds, Holm \$p = ([\d.]+)[\$;]",
           [(1, c10["Reach vs Reach-R1"]["delta"]), (2, c10["Reach vs Reach-R1"]["won"]),
            (3, c10["Reach vs Reach-R1"]["p_holm"])])
    _quote(rep, "abstract", _tex("abstract.tex"),
           r"Spearman \$\\rho = ([\d.]+)\$, beating closed-form centrality on all twelve held-out architectures \(\$\+([\d.]+)\$\).*?"
           r"reaches \$\\rho = ([\d.]+)\$ without training.*?"
           r"match the count \(\$\\rho = ([\d.]+)\$\)",
           [(1, tf["summary"]["InDeg"]["loso_mean_rho"]), (2, tf["contrasts_vs_topo_qos"]["InDeg"]["delta"]),
            (3, tf["summary"]["Reach"]["systems_mean_rho"]),
            (4, means["gl_proj_qos16_cap"]["loso_mean_rho"])])
    regimes = _load("engine_regimes.json")
    if regimes is not None:
        by = regimes["loso"]["regimes_by_topo_qos_tercile"]
        sec8 = _tex("sec8_discussion.tex")
        table = sec8[sec8.index(r"\label{tab:regimes}"):sec8.index(r"\end{table}", sec8.index(r"\label{tab:regimes}"))]

        def tm(t: str, arm: str) -> float:
            ids = by[t]["folds"]
            if arm in ("InDeg", "Reach"):
                return float(np.mean([tf["per_fold"][FOLD_NAMES_A7[i]][arm]["rho"] for i in ids]))
            return float(np.mean([dg["per_fold"][arm][i]["rho"] for i in ids]))
        for label_, t, pairs in (
            ("Closed-form ranks poorly", "weak", (("InDeg", r"\\texttt\{InDeg\}"), ("Reach", r"\\texttt\{Reach\}"),
                                                  ("gl_proj_qos16_cap", r"\\texttt\{GAT-P-QoS\}"))),
            ("Intermediate", "middle", (("InDeg", r"\\texttt\{InDeg\}"), ("gl_proj_qos16_cap", r"\\texttt\{GAT-P-QoS\}"))),
            ("Closed-form ranks well", "strong", (("InDeg", r"\\texttt\{InDeg\}"),
                                                  ("gl_proj_qos16_cap", r"\\texttt\{GAT-P-QoS\}"))),
        ):
            pat = label_ + r".*?" + r".*?".join(tex_ + " " + num for _, tex_ in pairs)
            _quote(rep, "tab:regimes", table, pat, [(i + 1, tm(t, a)) for i, (a, _) in enumerate(pairs)])

    # Both supplementary table files, re-rendered and compared byte for byte.
    try:
        from reproduce import render_amendment7_tables as r7
        from reproduce import render_amendment9_tables as r9
        tf_all = r7._load("tf_baselines.json")
        a7 = "\n\n".join([
            "% Generated by reproduce/render_amendment7_tables.py -- do not edit by hand.",
            r7.per_fold_table(tf_all), r7.contrasts_table(tf_all), r7.systems_table(tf_all),
            r7.controls_table(r7._load("qos_attribution_controls.json"), r7._load("qos_indep_corpus.json"),
                              r7._load("topo_substrate_check.json")),
            r7.oracle_table(r7._load("oracle_param_sensitivity.json")),
            r7.descriptives_table(r7._load("system_model_descriptives.json"), tf_all),
        ]) + "\n"
        a9 = "\n\n".join(["% Generated by reproduce/render_amendment9_tables.py -- do not edit by hand.",
                          r9.a9_folds(dg), r9.a9_contrasts(dg), r9.a9_zeroshot(dg), r9.a9_probe(dg),
                          r9.a10_derivation(dv)]) + "\n"
    except FileNotFoundError:
        rep.skipped.append("supp_amendment7/9.tex: an artifact is absent")
        return
    for fname, expected, script in (("supp_amendment7.tex", a7, "render_amendment7_tables.py"),
                                     ("supp_amendment9.tex", a9, "render_amendment9_tables.py")):
        rep.checked += 1
        if (ROOT / "docs/research/jss/latex" / fname).read_text() != expected:
            rep.findings.append(Finding(fname, "rendered tables", "content", "stale", "re-render",
                                        f"run reproduce/{script}"))


#: Amendment 7's artifacts key folds by display name.
FOLD_NAMES_A7 = {
    "atm_system": "ATM", "av_system": "AV System", "enterprise_system": "Enterprise",
    "financial_trading_system": "Financial Trading", "healthcare_system": "Healthcare",
    "hub_and_spoke_system": "Enterprise Integration (ESB)",
    "industrial_scada_system": "Industrial SCADA", "iot_smart_city_system": "IoT Smart City",
    "logistics_fleet_system": "Logistics Fleet", "microservices_system": "Microservices",
    "realtime_gaming_system": "Real-Time Gaming", "telecom_ran_system": "Telecom RAN",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loso", default="loso_all_variants_v5.json",
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
    check_table5_columns(rep, args.main_table)
    check_supplement_stratification(rep)
    check_supplement_shrinkage(rep)
    check_table7_loso(rep, args.loso)
    check_table7_delta(rep)
    check_table7c_active(rep, args.loso)
    check_contrasts(rep)
    check_scale_table(rep)
    check_realworld(rep)
    check_table9c_active(rep)
    check_oracle_timing(rep)
    check_gate_ratio_table(rep)
    check_qos_label_ablation(rep)
    check_hybrid_table(rep)
    check_system_models_transfer(rep)
    check_independent_oracles(rep)
    check_contrasts_matched(rep)
    check_omnibus_holm(rep)
    check_engine_regimes(rep)
    check_dependency_graph(rep)

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
