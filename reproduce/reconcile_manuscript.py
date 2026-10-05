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


def _table_source(label: str) -> str:
    """Section 7, or the supplement once the table carrying ``label`` moved there.

    The round-13 condensation moved Tables tab:a14/a16/a17 to the supplement
    (Supplementary Section supp:controls) and left a digest, tab:controls, in
    Section 7; the table checks follow the table, the prose quotes stay on Section 7.
    """
    sec7 = _tex("sec7_results.tex")
    return sec7 if label in sec7 else _supp()


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
    supp_text = _supp()
    if r"\label{tab:supp-scale}" in supp_text or r"\label{tab:scale}" in supp_text:
        tex = supp_text
    else:
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
    tex = _tex("sec7_results.tex")
    if r"\label{tab:system_models_transfer}" in tex:
        return
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
    "gate_oracle_ratio.json": "Table gate_ratio (Section 7.5)",
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
    sources = [
        ("tab:hybrid", _rows(tex, r"\midrule", after_label=r"\label{tab:hybrid}")),
        ("tab:supp-moved-baselines", _rows(_supp(), r"\midrule", after_label=r"\label{tab:supp-moved-baselines}")),
    ]
    for tab_name, rows_list in sources:
        for row in rows_list:
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
                    rep.findings.append(Finding(tab_name, _label(cells[0]), nm, got, round(truth, 4)))


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
    sources = [
        ("tab:system_models_transfer", _rows(tex, r"\midrule", after_label=r"\label{tab:system_models_transfer}")),
        ("tab:supp-moved-systems", _rows(_supp(), r"\midrule", after_label=r"\label{tab:supp-moved-systems}")),
    ]
    for tab_name, rows_list in sources:
        for row in rows_list:
            cells = _cells(row)
            name = _label(cells[0]).replace("$^\\dagger$", "").replace("^\\dagger", "").strip()
            if name in expected:
                rho_truth, prauc_truth = expected[name]
                got_rho = _num(cells[2])
                got_prauc = _num(cells[4])
                rep.checked += 1
                if got_rho is None or abs(got_rho - rho_truth) > 0.001:
                    rep.findings.append(Finding(tab_name, name, "mean_rho", got_rho, round(rho_truth, 4)))
                rep.checked += 1
                if got_prauc is None or abs(got_prauc - prauc_truth) > 0.001:
                    rep.findings.append(Finding(tab_name, name, "pr_auc", got_prauc, round(prauc_truth, 4)))


#: Rankings that restate I*'s propagation rule (Proposition 1). Amendment 13 reports
#: them as references beside Analytic-I*, never as predictors: no contrast against
#: Topo-QoS in the body, no row in the predictor taxonomy or the guidance table.
REFERENCE_RANKERS = ("InDeg", "Reach", "Pubs-raw", "Reach-R1")

#: Engines whose prior is a reference ranking; Amendment 13 keeps them in the
#: supplement only.
SUPPLEMENT_ONLY_ENGINES = ("GAT-P+InDeg",)


def check_independent_oracles(rep: Report) -> None:
    """Table tab:independent_oracles: rankers against I*, full-population I_dyn, I_comp.

    Round 8 (Amendment 11 rule R): every ranker's I*, I_dyn (five-seed full population),
    partial correlations given I* and given the first-order expansion, and I_comp come
    from referee_round8_table7.json; Amendment 11's GBM rows from oracle_robust_ltr.json;
    the GNN surrogate from referee_round8_amendment14.json (per-seed means).
    """
    t7 = _load("referee_round8_table7.json")
    ltr = _load("oracle_robust_ltr.json")
    a14 = _load("referee_round8_amendment14.json")
    tex = _tex("sec7_results.tex")
    if None in (t7, ltr, a14) or r"\label{tab:independent_oracles}" not in tex:
        rep.skipped.append("tab:independent_oracles: table or a round-8 artifact absent")
        return
    sm = t7["summary"]

    def ranker(key: str):
        r = sm[key]
        return [(1, r["i_star"]["mean"], "i_star"), (2, r["i_dyn"]["mean"], "i_dyn"),
                (3, r["partial_idyn_given_istar"]["mean"], "partial"),
                (4, None if key == "Analytic-I*" else r["partial_idyn_given_analytic"]["mean"],
                 "partial_analytic"),
                (5, r["i_comp"]["mean"], "i_comp")]

    arms = ltr["loso"]["arm_means"]
    dyn = a14["summary"]["GAT-P-QoS-dyn"]
    rows = {"Analytic $I^*$": ranker("Analytic-I*")}
    rows.update({k: ranker(k) for k in sm if not k.startswith("_") and k != "Analytic-I*"})
    rows["GBM-P-QoS"] = [(1, arms["gbm_dep_qos"]["i_star"], "i_star"),
                         (2, arms["gbm_dep_qos"]["i_dyn"], "i_dyn"),
                         (5, arms["gbm_dep_qos"]["i_comp"], "i_comp")]
    rows["GBM-P-QoS$to$dyn$^star$"] = [(1, arms["gbm_dep_qos_dyn"]["i_star"], "i_star"),
                                       (2, arms["gbm_dep_qos_dyn"]["i_dyn"], "i_dyn"),
                                       (5, arms["gbm_dep_qos_dyn"]["i_comp"], "i_comp")]
    rows["GAT-P-QoS$to$dyn$^star$"] = [(1, dyn["i_star"], "i_star"),
                                                     (2, dyn["i_dyn"], "i_dyn"),
                                                     (5, dyn["i_comp"], "i_comp")]
    seen = 0
    sources = [
        ("tab:independent_oracles", _rows(tex, r"\midrule", after_label=r"\label{tab:independent_oracles}")),
        ("tab:supp-moved-oracles", _rows(_supp(), r"\midrule", after_label=r"\label{tab:supp-moved-oracles}")),
    ]
    for tab_name, rows_list in sources:
        for row in rows_list:
            cells = _cells(row)
            name = _label(cells[0]).replace("underline", "")
            if name not in rows:
                continue
            seen += 1
            for idx, truth, nm in rows[name]:
                if truth is None:
                    continue
                got = _num(cells[idx].replace("\\underline", "")) if idx < len(cells) else None
                rep.checked += 1
                if got is None or abs(got - truth) > 0.0006:
                    rep.findings.append(Finding(tab_name, name, nm, got, round(truth, 4)))
    rep.checked += 1
    if seen != len(rows):
        rep.findings.append(Finding("tab:independent_oracles", "rows", "count", seen, len(rows)))


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
    # Round 12 moved the table to the supplement; read whichever document holds it.
    tex = _tex("sec7_results.tex")
    if r"\label{tab:contrasts_matched}" not in tex:
        tex = _supp()
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
    ("sec7_results.tex", r"omnibus Holm correction \(\$p_\{\\text\{omni\}\} = ([\d.]+)\$ and \$([\d.]+)\$"),
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
    supp = _supp()
    lbl = r"\label{tab:supp-regimes}" if r"\label{tab:supp-regimes}" in supp else r"\label{tab:regimes}"
    table_source = supp if lbl in supp else sec8
    table = table_source[table_source.index(lbl):table_source.index(r"\end{table}", table_source.index(lbl))]
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
           r"originally pub-sub.*?Learned " + num + "--" + num + r" vs.\\ baseline " + num + "--" + num
           + r"; learned \$\\rho_\{>0\}\$ " + num + "--" + num,
           [(1, ps_lo), (2, ps_hi), (3, ps_cf_lo), (4, ps_cf_hi), (5, ps_pos_lo), (6, ps_pos_hi)])
    _quote(rep, "tab:regimes", table,
           r"originally RPC.*?Learned " + num + "--" + num + r", but application-layer baseline " + num
           + r" on Online Boutique.*?learned \$\\rho_\{>0\}\$ " + num + " to " + num,
           [(1, rpc_lo), (2, rpc_hi), (3, zs["Topo"]["realworld_cloud_microservices"]["rho"]),
            (4, rpc_pos_lo), (5, rpc_pos_hi)])
    for pattern, truths in (
        (r"\\texttt\{GBM-Feat\}\) (?:achieves?|reach(?:es)?) \$\\rho = ([\d.]+)\$ under LOSO, (?:level with|similar to) \\texttt\{GAT-QoS\} \(\$([\d.]+)\$\)",
         [(1, L["mean_rho"]["GBM-Feat"]), (2, L["mean_rho"]["GAT-QoS"])]),
        # Round 12: Section 7.2 no longer quotes the raw-graph typing effects (they live in the
        # supplement's matched 2x2, checked by check_contrasts_matched).
    ):
        # Round 13: the GBM-Feat result is stated once, in Section 7.2; 8.2 refers to it.
        _quote(rep, "sec:rq2", _tex("sec7_results.tex"), pattern, truths)
    avg = _load("referee_round7_averaging.json")
    if avg is not None:
        sp = avg["seed_spread"]
        _quote(rep, "sec:8.2", sec8,
               r"mean within-fold seed standard deviation " + num + r", mean \$\\rho = ([\d.]+)\$\)"
               r".*?\\texttt\{GAT-P-QoS\} was (?:not \(|stable \()" + num + r", (?:mean )?\$\\rho = ([\d.]+)\$\)",
               [(1, sp["HGT-P-QoS"]["mean_sd"]), (2, avg["per_ranker"]["HGT-P-QoS"]["arithmetic"]),
                (3, sp["GAT-P-QoS"]["mean_sd"]), (4, avg["per_ranker"]["GAT-P-QoS"]["arithmetic"])])
    else:
        rep.skipped.append("referee_round7_averaging.json absent; seed-spread quotes unchecked")

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
            s_ = tf["summary"][counts[name]]
            truths = ((1, s_["loso_mean_rho"], "mean_rho"), (6, s_["loso_mean_overlap"], "overlap_at_k"))
            for idx in (3, 4, 5):  # Amendment 13: a reference carries no contrast
                rep.checked += 1
                if idx >= len(cells) or cells[idx] != "---":
                    rep.findings.append(Finding("tab:hybrid", name, f"cell {idx}",
                                                cells[idx] if idx < len(cells) else None, "---"))
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

    means, zs = dg["means"], dg["zeroshot"]
    con = {_registry.relabel(k): v for k, v in dg["contrasts"].items()}
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
           r"without Rule~5, \\texttt\{Reach\} (?:falls|drops) by " + num + r" \((\d+)/12 folds, Holm \$p = ([\d.]+)[\$;]",
           [(1, c10["Reach vs Reach-R1"]["delta"]), (2, c10["Reach vs Reach-R1"]["won"]),
            (3, c10["Reach vs Reach-R1"]["p_holm"])])
    _quote(rep, "sec:3.3 Rule 5", _tex("sec3_sag_model.tex"),
           r"including Rule~5 raises the agreement of transitive reach with \$I\^\*\$ by " + num
           + r" \((\d+) of 12 folds, Holm \$p = ([\d.]+)\$",
           [(1, c10["Reach vs Reach-R1"]["delta"]), (2, c10["Reach vs Reach-R1"]["won"]),
            (3, c10["Reach vs Reach-R1"]["p_holm"])])
    ioe = _load("independent_oracle_evaluation.json")
    raw12, lrn = _load("referee_round7_raw_baselines.json"), _load("referee_round7_learned_oracles.json")
    if None not in (ioe, raw12, lrn):
        learned_comp = max(v["summary"]["i_comp"]["rho"]["mean"] for k, v in lrn.items()
                           if isinstance(v, dict) and "summary" in v and k not in SUPPLEMENT_ONLY_ENGINES)
        t7 = _load("referee_round8_table7.json")
        arms = (_load("oracle_robust_ltr.json") or {}).get("loso", {}).get("arm_means", {})
        a15 = _load("idyn_rate_expansion.json") or {}
        _quote(rep, "abstract", _tex("abstract.tex"),
               r"reaches Spearman's \$\\rho = ([\d.]+)\$.*?"
               r"counting direct dependents \(\$\\rho = ([\d.]+)\$[;)].*?"
               r"rate-weighted first-order approximation reaches \$\\rho = ([\d.]+)\$.*?"
               r"trained on simulator labels \(\$\\rho = ([\d.]+)\$[;)]",
               [(1, means["gl_proj_qos16_cap"]["loso_mean_rho"]), (2, tf["summary"]["InDeg"]["loso_mean_rho"]),
                (3, a15["summary"]["loso"]["Rate-I_dyn"]["i_dyn"]["mean"]),
                (4, arms["gbm_dep_qos_dyn"]["i_dyn"])])
    regimes = _load("engine_regimes.json")
    if regimes is not None:
        by = regimes["loso"]["regimes_by_topo_qos_tercile"]
        supp = _supp()
        lbl = r"\label{tab:supp-regimes}" if r"\label{tab:supp-regimes}" in supp else r"\label{tab:regimes}"
        table_source = supp if lbl in supp else _tex("sec8_discussion.tex")
        table = table_source[table_source.index(lbl):table_source.index(r"\end{table}", table_source.index(lbl))]

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


def check_referee_round7(rep: Report) -> None:
    """Amendment 12 (round-7 referee analyses): Table 6's raw-graph rows, the prose
    figures of Section 7.1, the rendered supplement tables, and the Guide's length
    limits on the abstract (250 words) and the highlights (85 characters)."""
    raw = _load("referee_round7_raw_baselines.json")
    part = _load("referee_round7_partial.json")
    rec = _load("referee_round7_recall.json")
    if None in (raw, part, rec):
        rep.skipped.append("Amendment 12 artifacts absent; round-7 checks skipped")
        return
    from reproduce.training_free_suite import FOLDS as _F, mean_ci as _ci
    tex = _tex("sec7_results.tex")
    per = raw["per_scenario"]
    sources = [
        ("tab:hybrid", _rows(tex, r"\midrule", after_label=r"\label{tab:hybrid}")),
        ("tab:supp-moved-baselines", _rows(_supp(), r"\midrule", after_label=r"\label{tab:supp-moved-baselines}")),
    ]
    for tab_name, rows_list in sources:
        for row in rows_list:
            cells = _cells(row)
            name = _label(cells[0]).replace("$^\\S$", "").strip()
            if name not in ("Degree-raw", "RevPR-raw", "Pubs-raw", "Reach-R1"):
                continue
            xs = [per[f]["i_star"][name]["rho"] for f in _F]
            for idx, truth, nm in ((1, sum(xs) / len(xs), "mean_rho"),
                                   (2, sum(per[f]["i_star"][name]["rho_active"] for f in _F) / len(_F), "rho_active"),
                                   (6, sum(per[f]["i_star"][name]["overlap_at_k"] for f in _F) / len(_F), "overlap")):
                got = _num(cells[idx])
                rep.checked += 1
                if got is None or abs(got - truth) > 0.0006:
                    rep.findings.append(Finding(tab_name, name, nm, got, round(truth, 4)))
            lo, hi = _ci(xs)
            rep.checked += 1
            if f"[{lo:.3f}, {hi:.3f}]".replace("-", "") not in cells[1].replace("$", "").replace("-", ""):
                rep.findings.append(Finding(tab_name, name, "ci95", cells[1], f"[{lo:.3f}, {hi:.3f}]"))

    ps, cur = part["summary"], rec["curves"]["i_star"]["InDeg"]["curve"]
    num = r"\$([-+]?[\d.]+)\$"
    t7 = _load("referee_round8_table7.json")
    if t7 is not None:
        q = t7["summary"]
        _quote(rep, "sec:rq1", tex,
               r"After (?:the rank of \$I\^\*\$ is removed|removing the rank of \$I\^\*\$), \\texttt\{InDeg\} keeps " + num
               + r" \$\[([\d.]+), ([\d.]+)\]\$, \\texttt\{Reach\} " + num
               + r", \\texttt\{Topo-QoS\} " + num,
               [(1, q["InDeg"]["partial_idyn_given_istar"]["mean"]),
                (2, q["InDeg"]["partial_idyn_given_istar"]["ci95"][0]),
                (3, q["InDeg"]["partial_idyn_given_istar"]["ci95"][1]),
                (4, q["Reach"]["partial_idyn_given_istar"]["mean"]),
                (5, q["Topo-QoS"]["partial_idyn_given_istar"]["mean"])])
    gat, topo_c = rec["curves"]["i_star"]["GAT-P-QoS"]["curve"], rec["curves"]["i_star"]["Topo-QoS"]["curve"]
    ana, gat_dyn = rec["curves"]["i_star"]["Analytic-I*"]["curve"], rec["curves"]["i_dyn"]["GAT-P-QoS"]["curve"]
    indeg_dyn = rec["curves"]["i_dyn"]["InDeg"]["curve"]
    full = _load("referee_round10_recall_idyn_full.json")  # Figure 5B: exhaustive I_dyn
    if full is None:
        rep.skipped.append("referee_round10_recall_idyn_full.json absent; Figure 5B prose unchecked")
    else:
        gat_dyn = full["curves"]["i_dyn"]["GAT-P-QoS"]["curve"]
        indeg_dyn = full["curves"]["i_dyn"]["InDeg"]["curve"]
        rate_dyn = full["curves"]["i_dyn"]["Rate-weighted"]["curve"]
        _quote(rep, "sec:rq1 Figure 5B Eq. 7", tex,
               r"first-order rule, recovers " + num + r" of that set at 20\\% and " + num + r" at 30\\%",
               [(1, rate_dyn["0.20"]["expected"]), (2, rate_dyn["0.30"]["expected"])])
    _quote(rep, "sec:rq1", tex,
           r"At \$k = 20\\%\$, \\texttt\{GAT-P-QoS\} recovers " + num + r" of the critical set and \\texttt\{Topo-QoS\} (?:recovers )?"
           + num + r".*?(?:top \$?40\\%\$? by \\texttt\{GAT-P-QoS\}|\\texttt\{GAT-P-QoS\} must flag the top \$?40\\%\$?) \(" + num,
           [(1, gat["0.20"]["expected"]), (2, topo_c["0.20"]["expected"]), (3, gat["0.40"]["expected"])])
    _quote(rep, "sec:rq1", tex,
           r"at \$k = 20\\%\$, \\texttt\{InDeg\} recovers " + num + r" \(tie-breaking bounds " + num + "--" + num
           + r"\), and \$?80\\%\$? (?:needs|recall requires) the top \$?45\\%\$? by \\texttt\{InDeg\} \(" + num
           + r"\) or by the first-order expansion \(" + num,
           [(1, cur["0.20"]["expected"]), (2, cur["0.20"]["pessimistic"]), (3, cur["0.20"]["optimistic"]),
            (4, cur["0.45"]["expected"]), (5, ana["0.45"]["expected"])])
    _quote(rep, "sec:rq1", tex,
           r"\\texttt\{GAT-P-QoS\} reaches " + num + r" at \$?40\\%\$? and " + num + r" at \$?50\\%\$?, (?:against|compared to) "
           + num + r" at \$?40\\%\$? for the \\texttt\{InDeg\} reference",
           [(1, gat_dyn["0.40"]["expected"]), (2, gat_dyn["0.50"]["expected"]), (3, indeg_dyn["0.40"]["expected"])])
    for key, k80 in (("InDeg", 0.45), ("GAT-P-QoS", 0.40), ("Analytic-I*", 0.45)):
        rep.checked += 1
        if rec["curves"]["i_star"][key]["safety_margin"]["0.80"] != k80:
            rep.findings.append(Finding("sec:rq1", f"{key} 80% recall margin", "k", k80,
                                        rec["curves"]["i_star"][key]["safety_margin"]["0.80"]))
    for key in ("Topo-QoS", "Reach"):
        rep.checked += 1
        if rec["curves"]["i_star"][key]["safety_margin"]["0.80"] is not None:
            rep.findings.append(Finding("sec:rq1", f"{key} 80% recall margin", "k", "none within 50%",
                                        rec["curves"]["i_star"][key]["safety_margin"]["0.80"]))

    try:
        from reproduce import render_referee_tables as rr
        import io
        import contextlib
        target = ROOT / "docs/research/jss/latex/supp_referee.tex"
        before = target.read_text()
        with contextlib.redirect_stdout(io.StringIO()):
            rr.main()
        after = target.read_text()
        rep.checked += 1
        if before != after:
            target.write_text(before)
            rep.findings.append(Finding("supp_referee.tex", "rendered tables", "content", "stale",
                                        "re-render", "run reproduce/render_referee_tables.py"))
    except FileNotFoundError:
        rep.skipped.append("supp_referee.tex: an artifact is absent")

    lat = _load("referee_round7_latency.json")
    if lat is not None:
        by_n = {r["n_actual"]: r for r in lat["sizes"]}
        supp_text = _supp()
        lbl_cs = r"\label{tab:supp-count-scale}" if r"\label{tab:supp-count-scale}" in supp_text else r"\label{tab:count-scale}"
        tex_cs = supp_text if lbl_cs in supp_text else tex
        k0 = tex_cs.index(r"\midrule", tex_cs.index(lbl_cs))
        body = tex_cs[k0:tex_cs.index(r"\bottomrule", k0)]
        rows_cs = [ln.strip() for ln in body.split("\n") if ln.strip().endswith(r"\\")]
        rep.checked += 1
        if len(rows_cs) != len(by_n):
            rep.findings.append(Finding("tab:count-scale", "rows", "count", len(rows_cs), len(by_n)))
        for row in rows_cs:
            cells = _cells(row)
            n = _num(cells[0])
            if n is None or int(n) not in by_n:
                continue
            r = by_n[int(n)]
            truths = [(2, 1000 * r["count_path_s"], 0.051), (3, 1000 * r["reach_s"], 0.051)]
            if r["istar_s"] is not None:
                truths += [(4, r["istar_s"], 0.051), (5, r["istar_s"] / r["count_path_s"], 0.51)]
            for idx, truth, tol in truths:
                got = _num(cells[idx])
                rep.checked += 1
                if got is None or abs(got - truth) > tol:
                    rep.findings.append(Finding("tab:count-scale", f"n={int(n)}", str(idx), got, round(truth, 2)))
    else:
        rep.skipped.append("referee_round7_latency.json absent; tab:count-scale unchecked")

    words = len(re.sub(r"\$[^$]*\$", "X", _tex("abstract.tex")).split())
    rep.checked += 1
    if words > 250:
        rep.findings.append(Finding("abstract", "length", "words", words, 250))
    for line in (ROOT / "docs/research/jss/latex/highlights.tex").read_text().splitlines():
        if line.strip() and not line.startswith(("%", "\\")):
            rep.checked += 1
            if len(line) > 85:
                rep.findings.append(Finding("highlights", line[:30], "characters", len(line), 85))


def check_reference_demotion(rep: Report) -> None:
    """Amendment 13: dependency counts are references, not predictors.

    ``InDeg``, ``Reach``, Pubs-raw and Reach-R1 restate I*'s propagation rule
    (Proposition 1). They may appear only in the reference blocks of the results
    tables and in the circularity discussion. They must not appear in the title,
    abstract, highlights, conclusion, predictor taxonomy or guidance table, and
    an engine whose prior is a reference must not have a row in the main text.
    The numbers in the reference rows are checked by the table checks.
    """
    ref_pat = re.compile(r"InDeg|\bReach\b|Pubs-raw|Reach-R1")

    def _absent(where: str, text: str) -> None:
        rep.checked += 1
        m = ref_pat.search(text)
        if m:
            rep.findings.append(Finding(where, "reference ranker", "mention", m.group(), "absent",
                                        "Amendment 13: references are not predictors"))

    manuscript = (LATEX / "manuscript.tex").read_text()
    title = manuscript[manuscript.index(r"\title{"):manuscript.index(r"\author", manuscript.index(r"\title{"))]
    _absent("title", title)
    _absent("abstract", _tex("abstract.tex"))
    _absent("highlights", (LATEX / "highlights.tex").read_text())
    _absent("sec:9", _tex("sec9_conclusion.tex"))
    tex = _tex("sec8_discussion.tex")
    k = tex.index(r"\label{tab:guidance}")
    _absent("tab:guidance", tex[tex.index(r"\begin{tabular}", k):tex.index(r"\end{tabular}", k)])

    # The ranker taxonomy lists the references for completeness (advisor v7_2),
    # but only inside its own "Analytical: references" block.
    tex = _tex("sec6_experimental_setup.tex")
    k = tex.index(r"\label{tab:predictor_taxonomy}")
    tab = tex[tex.index(r"\begin{tabular}", k):tex.index(r"\end{tabular}", k)]
    in_ref, got_ref = False, set()
    for line in tab.splitlines():
        s_ = line.strip()
        if s_.startswith(r"\multicolumn"):
            in_ref = "references" in s_
            continue
        if s_ == r"\midrule" or "&" not in s_:
            in_ref = False if s_ == r"\midrule" else in_ref
            continue
        first = _cells(s_)[0]
        rep.checked += 1
        if in_ref:
            got_ref.update(n for n in ("Analytic", "Rate-weighted", "InDeg", "Reach") if re.search(r"\b%s\b" % n, first))
        elif ref_pat.search(first):
            rep.findings.append(Finding("tab:predictor_taxonomy", first[:30], "row", "outside reference block",
                                        "reference block only", "Amendment 13: references are not predictors"))
    rep.checked += 1
    if got_ref != {"Analytic", "Rate-weighted", "InDeg", "Reach"}:
        rep.findings.append(Finding("tab:predictor_taxonomy", "reference block", "rows", sorted(got_ref),
                                    ["Analytic", "InDeg", "Rate-weighted", "Reach"]))

    for fname in sorted(p.name for p in SECTIONS.glob("*.tex")):
        for line in _tex(fname).splitlines():
            if any(line.strip().startswith(r"\textbf{%s}" % e) for e in SUPPLEMENT_ONLY_ENGINES):
                rep.findings.append(Finding(fname, line.strip()[:30], "row", "present", "absent",
                                            "Amendment 13: supplement only"))
    rep.checked += 1

    # Each results table: references sit in the reference block and nowhere else.
    sec7 = _tex("sec7_results.tex")
    blocks = {"tab:hybrid": {"Analytic $I^*$", "InDeg", "Reach"},
              # Eq. 7 (Amendment 15) restates I_dyn's first-order rule.
              "tab:independent_oracles": {"Analytic $I^*$", "InDeg", "Reach", "Rate-weighted"},
              "tab:system_models_transfer": {"Reach", "InDeg"}}
    for label, expected in blocks.items():
        k = sec7.index(r"\label{%s}" % label)
        body = sec7[sec7.index(r"\toprule", k):sec7.index(r"\bottomrule", k)]
        in_ref, got_ref, stray = False, set(), set()
        for line in body.splitlines():
            s = line.strip()
            if s.startswith(r"\multicolumn"):
                in_ref = "Reference" in s
                continue
            if s == r"\midrule":
                in_ref = False
                continue
            if not s.startswith(r"\textbf{"):
                continue
            name = _label(_cells(s)[0]).replace("$^S$", "").replace("$^\\S$", "").split(" (")[0].strip()
            name = name.replace("$^S$", "").strip()
            if in_ref:
                got_ref.add(name)
            elif name in REFERENCE_RANKERS or name == "Analytic $I^*$":
                stray.add(name)
        rep.checked += 1
        if got_ref != expected or stray:
            rep.findings.append(Finding(label, "reference block", "rows",
                                        f"ref={sorted(got_ref)} stray={sorted(stray)}", sorted(expected)))



def check_round8(rep: Report) -> None:
    """Round 8 (Amendment 14): Table tab:a14 and the quoted F3/F6/F7 figures.

    Rows of tab:a14 are matched by their first cell; every numeric cell is checked
    against referee_round8_amendment14.json (and the w_in-held factorial against
    loso_significance_amendment14_cpu.json). Prose quotes are checked at printed
    precision.
    """
    a14 = _load("referee_round8_amendment14.json")
    sig = _load("loso_significance_amendment14_cpu.json")
    hyb = _load("referee_round8_hybrid.json")
    tost = _load("referee_round8_tost.json")
    nest = _load("referee_round8_nested.json")
    tex = _tex("sec7_results.tex")
    if None in (a14, sig, hyb, tost, nest) or r"\label{tab:a14}" not in _table_source(r"\label{tab:a14}"):
        rep.skipped.append("round 8: an Amendment 14 artifact or tab:a14 absent")
        return
    fam = {**a14["F1"], **a14["F2"]}
    summ, zs = a14["summary"], a14["zero_shot"]
    fact = {r["quantity"]: r for r in sig["factorial"]}
    ttex = _table_source(r"\label{tab:a14}")
    i = ttex.index(r"\label{tab:a14}")
    body = ttex[ttex.index(r"\toprule", i):ttex.index(r"\bottomrule", i)]
    seen = 0
    for line in body.splitlines():
        t = line.strip()
        if "&" not in t or t.startswith((r"\multicolumn", "Arm &")):
            continue
        cells = _cells(t)
        name = cells[0].replace("$-$", "-").replace("$^*$", "*")
        comp = cells[1].split(" (")[0].replace("$-$", "-").replace("$^*$", "*")
        checks = []
        if name in summ:
            key = f"{name} vs {comp}"
            c = fam[key]
            dyn, comp_ = cells[6].split("/")
            checks = [(cells[2], summ[name]["loso_i_star"]), (cells[3], c["delta"]),
                      (cells[5], c["p_holm"]), (dyn, summ[name]["i_dyn"]),
                      (comp_, summ[name]["i_comp"]), (cells[7], zs[name])]
        else:
            q = {"Typing (main effect)": "main_typing", "``QoS'' inputs (main effect)": "main_qos",
                 "Interaction": "interaction"}.get(name)
            if q is None:
                continue
            checks = [(cells[3], fact[q]["mean_delta"]), (cells[5], fact[q]["p_holm"])]
        seen += 1
        for cell, truth in checks:
            got = _num(cell)
            rep.checked += 1
            tol = 0.0006 if abs(truth) >= 0.01 else 0.00006
            if got is None or abs(got - truth) > max(tol, 0.0006):
                rep.findings.append(Finding("tab:a14", name, "cell", got, round(truth, 4)))
    rep.checked += 1
    if seen != 9:
        rep.findings.append(Finding("tab:a14", "rows", "count", seen, 9))
    num = r"\$?([-+]?[\d.]+)\$?"
    _quote(rep, "sec:rq1", tex,
           r"a two one-sided test at the registered margin of \$\\pm 0\.05\$ (?:fails|does not pass) \(\$p = ([\d.]+)\$ per seed, \$?([\d.]+)\$? for the ensemble",
           [(1, tost["GAT-P-QoS_vs_InDeg"]["per_seed_mean"]["tost_t"]["p"]),
            (2, tost["GAT-P-QoS_vs_InDeg"]["seed_ensemble"]["tost_t"]["p"])])
    reg, f7 = hyb["registered_amendments_5_6"], hyb["F7"]
    _quote(rep, "sec:rq1", tex,
           r"Hybrid-HGT vs\.(?:\\ )?\s*\\texttt\{HGT-QoS\} \$\+([\d.]+)\$, \$p = ([\d.]+)\$; Hybrid-GAT vs\.(?:\\ )?\s*\\texttt\{GAT-QoS\} \$\+([\d.]+)\$, \$p = ([\d.]+)\$",
           [(1, reg["Hybrid-HGT vs HGT-QoS"]["delta"]), (2, reg["Hybrid-HGT vs HGT-QoS"]["p"]),
            (3, reg["Hybrid-GAT vs GAT-QoS"]["delta"]), (4, reg["Hybrid-GAT vs GAT-QoS"]["p"])])
    _quote(rep, "supp:baselines", _supp(),
           r"unweighted betweenness on the same graph \$\+([\d.]+)\$, constant topic weights \$\+([\d.]+)\$, both Holm \$p = ([\d.]+)\$",
           [(1, f7["Hybrid-GAT vs Topo (projection)"]["delta"]), (2, f7["Hybrid-GAT vs Topo-Mult"]["delta"]),
            (3, f7["Hybrid-GAT vs Topo-Mult"]["p_holm"])])
    F = nest["F6"]
    _quote(rep, "sec:rq2", tex,
           r"(?:it )?(?:moves|changes) \\texttt\{HGT-QoS\} from \$([\d.]+)\$ to \$([\d.]+)\$ \(\$\+([\d.]+)\$\) and \\texttt\{GAT-P-QoS\} from \$([\d.]+)\$ to \$([\d.]+)\$ \(\$-([\d.]+)\$\), neither(?: of which is)? significant \(Holm \$p = ([\d.]+)\$\)[;.]\s*[Nn]ested \\texttt\{HGT-QoS\} (?:against|versus) \\texttt\{Topo-QoS\} is \$\+([\d.]+)\$ on 9 of 12 folds, still not significant \(Holm \$p = ([\d.]+)\$",
           [(1, nest["summary"]["HGT-QoS"]["fixed"]), (2, nest["summary"]["HGT-QoS"]["nested"]),
            (3, F["nested HGT-QoS vs fixed HGT-QoS"]["delta"]), (4, nest["summary"]["GAT-P-QoS"]["fixed"]),
            (5, nest["summary"]["GAT-P-QoS"]["nested"]), (6, -F["nested GAT-P-QoS vs fixed GAT-P-QoS"]["delta"]),
            (7, F["nested HGT-QoS vs fixed HGT-QoS"]["p_holm"]), (8, F["nested HGT-QoS vs Topo-QoS"]["delta"]),
            (9, F["nested HGT-QoS vs Topo-QoS"]["p_holm"])])
    for f, pat in (("abstract.tex", r"without exceeding|never exceed"),
                   ("sec9_conclusion.tex", r"never exceed"),
                   ("sec4_failure_impact_prediction.tex", r"registered \(Amendment~11\) but not completed")):
        rep.checked += 1
        if re.search(pat, _tex(f)):
            rep.findings.append(Finding("round 8 wording", f, "withdrawn phrase", pat, None))


def check_amendment16(rep: Report) -> None:
    """Round 11 (Amendment 16): Table tab:a16 and the quoted F8/F9/F10 figures.

    Rows are matched by (arm, comparator); every numeric cell is checked against
    referee_round11_amendment16.json, zero-shot cells against its zero_shot block.
    """
    a16 = _load("referee_round11_amendment16.json")
    tex = _tex("sec7_results.tex")
    if a16 is None or r"\label{tab:a16}" not in _table_source(r"\label{tab:a16}"):
        rep.skipped.append("round 11: referee_round11_amendment16.json or tab:a16 absent")
        return
    fam = {**a16["F8"], **a16["F9"], **a16["F10"]}
    summ, zs = a16["summary"], a16["zero_shot"]
    ttex = _table_source(r"\label{tab:a16}")
    i = ttex.index(r"\label{tab:a16}")
    body = ttex[ttex.index(r"\toprule", i):ttex.index(r"\bottomrule", i)]
    seen = 0
    for line in body.splitlines():
        t = line.strip()
        if "&" not in t or t.startswith((r"\multicolumn", "Arm &")):
            continue
        cells = _cells(t)
        name, comp = cells[0], cells[1].split(" (")[0]
        c = fam[f"{name} vs {comp}"]
        checks = [(cells[2], summ[name]["loso_i_star"]), (cells[3], c["delta"]), (cells[5], c["p_holm"])]
        lo, hi = c["ci95"]
        rep.checked += 1
        if f"[{lo:+.3f}, {hi:+.3f}]".replace("+0.000", "+0.000") not in cells[3].replace("$", ""):
            rep.findings.append(Finding("tab:a16", name, "ci95", cells[3], f"[{lo:+.3f}, {hi:+.3f}]"))
        rep.checked += 1
        if cells[4] != f"{c['won']}/12":
            rep.findings.append(Finding("tab:a16", name, "won", cells[4], c["won"]))
        if "---" not in cells[6]:
            dyn, comp_ = cells[6].split("/")
            checks += [(dyn, summ[name]["i_dyn"]), (comp_, summ[name]["i_comp"])]
        if "---" not in cells[7]:
            checks.append((cells[7], zs[name]["mean_rho"]))
        seen += 1
        for cell, truth in checks:
            got = _num(cell)
            rep.checked += 1
            if got is None or abs(got - truth) > 0.0006:
                rep.findings.append(Finding("tab:a16", f"{name} vs {comp}", "cell", got, round(truth, 4)))
    rep.checked += 1
    if seen != 8:
        rep.findings.append(Finding("tab:a16", "rows", "count", seen, 8))
    num = r"\$?([-+]?[\d.]+)\$?"
    f8a, f8b = a16["F8"]["GAT-QoS-R vs GAT-QoS"], a16["F8"]["GAT-P-QoS vs GAT-QoS-R"]
    _quote(rep, "abstract direction control", _tex("abstract.tex"),
           r"\$\+([\d.]+)\$ above the same model with reverse edges on the raw multigraph",
           [(1, f8b["delta"])])
    _quote(rep, "sec:rq2 direction", tex,
           r"It reaches " + num + r": direction alone recovers \$\+([\d.]+)\$ of the gain, which is not significant \(Holm \$p = ([\d.]+)\$\), and \\texttt\{GAT-P-QoS\} still exceeds it by \$\+([\d.]+)\$ \$\[\+([\d.]+), \+([\d.]+)\]\$ on 10 of 12 folds \(Holm \$p = ([\d.]+)\$\)",
           [(1, summ["GAT-QoS-R"]["loso_i_star"]), (2, f8a["delta"]), (3, f8a["p_holm"]), (4, f8b["delta"]),
            (5, f8b["ci95"][0]), (6, f8b["ci95"][1]), (7, f8b["p_holm"])])
    g, h = a16["F9"]["Hybrid-GAT-AP vs Topo-QoS-AP"], a16["F9"]["Hybrid-HGT-AP vs Topo-QoS-AP"]
    gb, hb = a16["F9"]["Hybrid-GAT-AP vs GAT-QoS"], a16["F9"]["Hybrid-HGT-AP vs HGT-QoS"]
    _quote(rep, "supp:controls corrected-prior hybrids", _supp(),
           r"Hybrid-GAT-AP reaches " + num + r" \(\$\+([\d.]+)\$, Holm \$p = ([\d.]+)\$\) and Hybrid-HGT-AP " + num
           + r" \(\$\+([\d.]+)\$, Holm \$p = ([\d.]+)\$\), each on 11 of 12 folds, and again neither differs from its base learner \(\$\+([\d.]+)\$ and \$\+([\d.]+)\$, Holm \$p = ([\d.]+)\$\)",
           [(1, summ["Hybrid-GAT-AP"]["loso_i_star"]), (2, g["delta"]), (3, g["p_holm"]),
            (4, summ["Hybrid-HGT-AP"]["loso_i_star"]), (5, h["delta"]), (6, h["p_holm"]),
            (7, gb["delta"]), (8, hb["delta"]), (9, gb["p_holm"])])
    ti = a16["to_indeg"]
    _quote(rep, "supp:controls InDeg prior", _supp(),
           r"\\texttt\{GAT-QoS\+InDeg\} \(" + num + r"\) and \\texttt\{HGT-QoS\+InDeg\} \(" + num + r"\) gain \$\+([\d.]+)\$ and \$\+([\d.]+)\$ over their base learners \(Holm \$p = ([\d.]+)\$\) and land within \$\\pm ([\d.]+)\$",
           [(1, summ["GAT-QoS+InDeg"]["loso_i_star"]), (2, summ["HGT-QoS+InDeg"]["loso_i_star"]),
            (3, a16["F10"]["GAT-QoS+InDeg vs GAT-QoS"]["delta"]), (4, a16["F10"]["HGT-QoS+InDeg vs HGT-QoS"]["delta"]),
            (5, a16["F10"]["GAT-QoS+InDeg vs GAT-QoS"]["p_holm"]),
            (6, max(ti[k]["tost_t"]["equivalence_bound"] for k in ti))])


def check_amendment17(rep: Report) -> None:
    """Round 12 (Amendment 17/17b): Table tab:a17, the corrected-baseline row of tab:hybrid, and the
    F11/F12/F13/17b and descriptive figures quoted in the text.

    Rows of tab:a17 are matched by (arm, comparator) after normalising display labels; every
    numeric cell is checked against referee_round12_amendment17.json.
    """
    a17 = _load("referee_round12_amendment17.json")
    desc = _load("referee_round12_descriptive.json")
    perm = _load("referee_round12_perm.json")
    tex = _tex("sec7_results.tex")
    if None in (a17, desc, perm) or r"\label{tab:a17}" not in _table_source(r"\label{tab:a17}"):
        rep.skipped.append("round 12: a referee_round12 artifact or tab:a17 absent")
        return
    summ, zs, dyn = a17["summary"], a17["zero_shot"], a17["summary"]["dyn_means"]
    contrasts = {**a17["F11"], **a17["F12"], **a17["descriptive"],
                 "GAT-P-QoS-perm vs GAT-P-QoS": a17["F13"]}

    def _norm(cell: str) -> str:
        cell = cell.split(" (")[0].replace(r"$\to$", "-").replace(r"\texttt{", "").replace("}", "")
        # The supplement's copy cross-references the body equation as M-eq:... (xr-hyper).
        return "Eq7" if cell.startswith((r"Eq.~\eqref{eq:rate-expansion", r"Eq.~\eqref{M-eq:rate-expansion")) else cell

    ttex = _table_source(r"\label{tab:a17}")
    i = ttex.index(r"\label{tab:a17}")
    body = ttex[ttex.index(r"\toprule", i):ttex.index(r"\bottomrule", i)]
    seen = 0
    for line in body.splitlines():
        t = line.strip()
        if "&" not in t or t.startswith((r"\multicolumn", "Arm &")):
            continue
        cells = _cells(t)
        name, comp = _norm(cells[0]), _norm(cells[1])
        c = contrasts[f"{name} vs {comp}"]
        rho = dyn[name] if name in dyn else summ[name]["loso_i_star"]
        checks = [(cells[2], rho), (cells[3], c["delta"])]
        lo, hi = c["ci95"]
        rep.checked += 1
        if f"[{lo:+.3f}, {hi:+.3f}]" not in cells[3].replace("$", ""):
            rep.findings.append(Finding("tab:a17", name, "ci95", cells[3], f"[{lo:+.3f}, {hi:+.3f}]"))
        rep.checked += 1
        if cells[4] != f"{c['won']}/12":
            rep.findings.append(Finding("tab:a17", name, "won", cells[4], c["won"]))
        if "---" not in cells[5]:
            checks.append((cells[5], c["p_holm"] if "p_holm" in c else c["p"]))
        if "---" not in cells[6]:
            d_, c_ = cells[6].split("/")
            checks += [(d_, summ[name]["i_dyn"]), (c_, summ[name]["i_comp"])]
        if "---" not in cells[7]:
            checks.append((cells[7], zs[name]["mean_rho"]))
        seen += 1
        for cell, truth in checks:
            got = _num(cell)
            rep.checked += 1
            if got is None or abs(got - truth) > 0.0006:
                rep.findings.append(Finding("tab:a17", f"{name} vs {comp}", "cell", got, round(truth, 4)))
    rep.checked += 1
    if seen != 10:
        rep.findings.append(Finding("tab:a17", "rows", "count", seen, 10))

    num = r"\$?([-+]?[\d.]+)\$?"
    f11a = a17["F11"]["GAT-P-QoS-min vs GAT-QoS-R-min"]
    for fname in ("sec1_introduction.tex", "sec7_results.tex"):
        _quote(rep, f"{fname} F11", _tex(fname), r"reverse-edge control by \$\+([\d.]+)\$ \((?:10 of 12 folds, )?Holm \$p = ([\d.]+)\$",
               [(1, f11a["delta"]), (2, f11a["p_holm"])])
    _quote(rep, "sec:rq2 F11", tex,
           r"\\texttt\{GAT-QoS\} to " + num + r", the reverse-edge control to " + num + r"\), whereas the dependency-graph GAT keeps " + num
           + r".*?costs \\texttt\{GAT-P-QoS\} " + num + r".*?a sum-aggregation GNN on the dependency graph reaches " + num + r", " + num
           + r" below \\texttt\{InDeg\} \((\d+) of 12 folds\)",
           [(1, summ["GAT-QoS-min"]["loso_i_star"]), (2, summ["GAT-QoS-R-min"]["loso_i_star"]),
            (3, summ["GAT-P-QoS-min"]["loso_i_star"]), (4, -a17["F11"]["GAT-P-QoS-min vs GAT-P-QoS"]["delta"]),
            (5, summ["GIN-P-QoS-const"]["loso_i_star"]), (6, -a17["descriptive"]["GIN-P-QoS-const vs InDeg"]["delta"]),
            (7, a17["descriptive"]["GIN-P-QoS-const vs InDeg"]["won"])])
    f12 = a17["F12"]
    gat_gain = a17["descriptive"]["GAT-P-QoS-dyn+Eq7 vs GAT-P-QoS-dyn (I_dyn)"]
    _quote(rep, "sec:rq1 F12", tex,
           r"the gradient-boosted approximation reaches " + num + r", the formula's value \(\$\+([\d.]+)\$, (\d+) of 12 folds, Holm \$p = ([\d.]+)\$\).*?"
           r"it reaches " + num + r" \(" + num + r", Holm \$p = ([\d.]+)\$\).*?gains \$\+([\d.]+)\$ over the same GAT without it \((\d+) of 12 folds\)"
           r" but still falls below the formula \(" + num + r", " + num + r", (\d+) of 12 folds, Holm \$p = ([\d.]+)\$\)",
           [(1, dyn["GBM-P-QoS-dyn+Eq7"]), (2, abs(f12["GBM-P-QoS-dyn+Eq7 vs Eq7"]["delta"])),
            (3, f12["GBM-P-QoS-dyn+Eq7 vs Eq7"]["won"]), (4, f12["GBM-P-QoS-dyn+Eq7 vs Eq7"]["p_holm"]),
            (5, dyn["GBM-dyn-resid"]), (6, f12["GBM-dyn-resid vs Eq7"]["delta"]), (7, f12["GBM-dyn-resid vs Eq7"]["p_holm"]),
            (8, gat_gain["delta"]), (9, gat_gain["won"]), (10, dyn["GAT-P-QoS-dyn+Eq7"]),
            (11, f12["GAT-P-QoS-dyn+Eq7 vs Eq7"]["delta"]), (12, f12["GAT-P-QoS-dyn+Eq7 vs Eq7"]["won"]),
            (13, f12["GAT-P-QoS-dyn+Eq7 vs Eq7"]["p_holm"])])
    agg = a17["F11"]["GIN-P-QoS-min vs GAT-P-QoS-min"]
    _quote(rep, "sec:rq2 F11c", tex,
           r"sum aggregation beats attention on the dependency graph \(\$\+([\d.]+)\$, (\d+) of 12 folds, Holm \$p = ([\d.]+)\$\)",
           [(1, agg["delta"]), (2, agg["won"]), (3, agg["p_holm"])])
    _quote(rep, "sec1 F12", _tex("sec1_introduction.tex"),
           r"leaves a graph attention network below it \(" + num + r"\)",
           [(1, f12["GAT-P-QoS-dyn+Eq7 vs Eq7"]["delta"])])
    # Node order (F13) and Amendment 17b.
    f13 = a17["F13"]
    _quote(rep, "sec:rq2 F13", tex,
           r"\\texttt\{GAT-P-QoS-perm\}\) scores " + num + r" against " + num + r" \(" + num + r", (\d+) of 12 folds, \$p = ([\d.]+)\$",
           [(1, summ["GAT-P-QoS-perm"]["loso_i_star"]), (2, summ["GAT-P-QoS"]["loso_i_star"]), (3, f13["delta"]),
            (4, f13["won"]), (5, f13["p"])])
    pm = perm["permutation_means"]
    _quote(rep, "sec:rq2 17b", tex,
           r"seeds 17, 18 and 19 give " + num + r", " + num + r" and " + num + r" \(mean " + num + r"\).*?per-fold spread across the three permutations averages " + num,
           [(1, pm["17"]), (2, pm["18"]), (3, pm["19"]), (4, perm["permuted_mean"]), (5, perm["mean_spread"])])
    # Descriptive: corrected baseline row and the feature-vs-reference count.
    ap = desc["topo_qos_ap"]
    _quote(rep, "tab:hybrid corrected row", tex,
           r"corrected \(articulation term restored\) & " + num + r" \$\[([\d.]+), ([\d.]+)\]\$",
           [(1, ap["mean"]), (2, ap["ci95"][0]), (3, ap["ci95"][1])])
    fv = desc["indeg_feature_vs_reference"]
    sp = [v["spearman"] for v in fv["per_fold"].values()]
    rep.checked += 1
    sec3 = _tex("sec3_sag_model.tex")
    want = f"differ for {fv['n_differ']} of the {fv['n_apps']:,}".replace(",", "{,}")
    if want not in sec3 or f"$\\rho = {min(sp):.2f}$--${max(sp):.2f}$" not in sec3:
        rep.findings.append(Finding("sec:3.3", "InDeg feature vs reference", "quote", "see text",
                                    f"{want}; rho {min(sp):.2f}-{max(sp):.2f}"))


def check_controls_digest(rep: Report) -> None:
    """Round 13: Table tab:controls, the Section 7.2 digest of tab:a14/a16/a17.

    The full tables (checked cell by cell against their artifacts by check_round8,
    check_amendment16 and check_amendment17) live in the supplement. Each digest row
    must repeat the first six cells (arm, comparator, rho, delta [CI], won, p_Holm)
    of exactly one full-table row, so the digest cannot drift from what was verified.
    """
    sec7 = _tex("sec7_results.tex")
    if r"\label{tab:controls}" not in sec7:
        rep.skipped.append("tab:controls absent from sec7_results.tex")
        return

    def body(tex: str, label: str) -> List[List[str]]:
        i = tex.index(label)
        rows = []
        for line in tex[tex.index(r"\toprule", i):tex.index(r"\bottomrule", i)].splitlines():
            t = line.strip()
            if "&" in t and not t.startswith((r"\multicolumn", "Arm &")):
                rows.append(_cells(t.replace("{M-eq:", "{eq:")))
        return rows

    full = [r[:6] for lab in ("tab:a14", "tab:a16", "tab:a17")
            for r in body(_table_source(rf"\label{{{lab}}}"), rf"\label{{{lab}}}")]
    for cells in body(sec7, r"\label{tab:controls}"):
        rep.checked += 1
        n = sum(1 for r in full if r == cells[:6])
        if n != 1:
            rep.findings.append(Finding("tab:controls", " vs ".join(cells[:2])[:40], "row",
                                        f"{n} matching full-table rows", 1))


def check_cost_ll(rep: Report) -> None:
    """Table tab:cost-ll (round 8): like-for-like timings against referee_round8_cost.json."""
    d = _load("referee_round8_cost.json")
    tex = _tex("sec7_results.tex")
    if d is None or r"\label{tab:cost-ll}" not in tex:
        rep.skipped.append("tab:cost-ll: artifact or table absent")
        return
    by_n = {r["n_components"]: r for k, r in d["rows"].items() if k.startswith("generated_")}
    i = tex.index(r"\label{tab:cost-ll}")
    body = tex[tex.index(r"\midrule", i):tex.index(r"\bottomrule", i)]
    seen = 0
    for line in body.splitlines():
        t = line.strip()
        if "&" not in t or t.startswith("Corpus"):
            continue
        cells = _cells(t)
        n = _num(cells[0])
        r = by_n.get(int(n)) if n is not None else None
        if r is None:
            continue
        seen += 1
        truths = [(1, 1000 * r["count_s"], 0.051), (2, r["istar_one_pass_app_s"], 0.0051),
                  (3, r["istar_sweep_s"], 0.0051), (4, r["analyze_app_s"], 0.0051),
                  (5, r["gate_system_s"], 0.0051), (6, r["ratio_one_pass_to_count"], 0.51),
                  (7, r["ratio_app_analysis_to_one_pass"], 0.051)]
        for idx, truth, tol in truths:
            if truth is None:
                continue
            got = _num(cells[idx])
            rep.checked += 1
            if got is None or abs(got - truth) > tol:
                rep.findings.append(Finding("tab:cost-ll", f"n={int(n)}", str(idx), got, round(truth, 3)))
    rep.checked += 1
    if seen != len(by_n):
        rep.findings.append(Finding("tab:cost-ll", "rows", "count", seen, len(by_n)))
    sm = d["corpus_summary"]["ratio_app_analysis_to_one_pass"]
    _quote(rep, "sec:rq4", tex, r"feature extraction every learned ranker needs \$([\d.]+)\$--\$([\d.]+)\\times\$ more \(median \$([\d.]+)\\times\$\)",
           [(1, sm["min"]), (2, sm["max"]), (3, sm["median"])])
    # Round 10: corpus totals quoted in the RQ4 prose. These were once mislabelled
    # (the I* sweep total was called the count, the detection gate the features).
    folds = [r for k, r in d["rows"].items() if not k.startswith("generated_")]
    _quote(rep, "sec:rq4 corpus totals", tex,
           r"feature extraction that the learned rankers read took \$([\d.]+)\$~s.*?"
           r"and the dependency count \$([\d.]+)\$~s",
           [(1, sum(r["analyze_app_s"] for r in folds)), (2, sum(r["count_s"] for r in folds))])
    en = _load("energy_estimate.json")
    if en is None:
        rep.skipped.append("energy_estimate.json absent; RQ4 energy totals unchecked")
    else:
        tot = en["corpus_totals"]
        _quote(rep, "sec:rq4 energy", tex,
               r"five-seed \$I\^\*\$ labeling sweep \$([\d.]+)\$~s \(\$([\d.]+)\$~Wh.*?"
               r"full detection gate \$([\d.]+)\$~s \(\$([\d.]+)\$~Wh",
               [(1, tot["oracle_s"]), (2, tot["oracle_Wh_upper"]), (3, tot["gate_s"]), (4, tot["gate_Wh_upper"])])


def check_rate_expansion(rep: Report) -> None:
    """Amendment 15: the rate-weighted I_dyn reference (Eq. 7) and the input attribution.

    Table tab:independent_oracles' Eq. 7 row, the supplement's three Amendment 15 tables
    (every cell), and the figures quoted in Sections 6.1 and 7 and the Conclusion, all
    from data/benchmarks/idyn_rate_expansion.json. Also checks that the committed
    artifact reproduced Amendment 11 (its two gates).
    """
    a15 = _load("idyn_rate_expansion.json")
    ltr = _load("oracle_robust_ltr.json")
    if a15 is None or ltr is None:
        rep.skipped.append("check_rate_expansion: idyn_rate_expansion.json or oracle_robust_ltr.json absent")
        return
    for g, v in a15["gates"].items():
        rep.checked += 1
        if not v["passed"]:
            rep.findings.append(Finding("idyn_rate_expansion.json", g, "gate", "failed", "passed"))
    L, Z = a15["summary"]["loso"], a15["summary"]["zeroshot"]
    rate, C = L["Rate-I_dyn"], a15["contrasts"]["loso"]
    att, AC = a15["attribution"]["means"], a15["attribution"]["contrasts"]
    gbm_l = ltr["loso"]["arm_means"]["gbm_dep_qos_dyn"]["i_dyn"]
    gbm_z = ltr["zeroshot"]["arm_means"]["gbm_dep_qos_dyn"]["i_dyn"]

    # Table 6 row: I*, I_dyn [CI], partial|I* [CI], partial|Eq. 6, I_comp.
    sec7 = _tex("sec7_results.tex")
    row = [r for r in _rows(sec7, r"\midrule", after_label=r"\label{tab:independent_oracles}")
           if r.startswith(r"\textbf{Rate-weighted}")]
    rep.checked += 1
    if len(row) != 1:
        rep.findings.append(Finding("tab:independent_oracles", "Rate-weighted", "row", len(row), 1))
    else:
        nums = [float(x) for x in re.findall(r"-?\d+\.\d+", row[0].split("&", 1)[1])]
        truth = [rate["i_star"]["mean"], rate["i_dyn"]["mean"], *rate["i_dyn"]["ci95"],
                 rate["partial_idyn_given_istar"]["mean"], *rate["partial_idyn_given_istar"]["ci95"],
                 rate["partial_idyn_given_analytic"]["mean"], rate["i_comp"]["mean"]]
        rep.checked += 1
        if len(nums) != len(truth) or any(abs(a - round(b, 3)) > 1e-9 for a, b in zip(nums, truth)):
            rep.findings.append(Finding("tab:independent_oracles", "Rate-weighted", "cells",
                                        nums, [round(t, 3) for t in truth]))

    # Supplement tables: every numeric cell of every row.
    supp = (LATEX / "supp_advisor_v6.tex").read_text()
    names = {r.split(" & ")[0]: r for r in supp.splitlines() if r.endswith(r"\\") and " & " in r}
    per, afold = a15["per_fold"], a15["attribution"]["per_fold"]

    def gbm(n: str) -> float:
        b = ltr["loso"]["per_fold"].get(n) or ltr["zeroshot"]["per_system"][n]
        return b["arms"]["gbm_dep_qos_dyn"]["i_dyn"]["rho"]

    forms = ("Analytic-I*", "Rate-I_dyn", "RatePayload-I_dyn", "PubRate")
    expect: Dict[str, List[float]] = {}
    for n in per:
        label = _A15_NAMES.get(n)
        vals = [per[n][k]["i_dyn"]["rho"] for k in forms] + [gbm(n)]
        if n in afold["S"]:
            vals += [afold[a][n] for a in ("S", "S+rate,payload", "S+QoS-policy")]
        expect[label] = vals
    # The two Mean rows: LOSO table first, then the zero-shot table.
    means = [[L[k]["i_dyn"]["mean"] for k in forms] + [gbm_l]
             + [att[a]["mean"] for a in ("S", "S+rate,payload", "S+QoS-policy")],
             [Z[k]["i_dyn"]["mean"] for k in forms] + [gbm_z]]
    mean_rows = [r for r in supp.splitlines() if r.startswith(r"\textbf{Mean}")]
    checks = [(label, names.get(label), vals) for label, vals in expect.items()]
    checks += [(f"Mean ({k})", mean_rows[i] if i < len(mean_rows) else None, v)
               for i, (k, v) in enumerate(zip(("LOSO", "zero-shot"), means))]
    for label, r, vals in checks:
        rep.checked += 1
        if r is None:
            rep.findings.append(Finding("supp_advisor_v6.tex", label, "row", "absent", "present"))
            continue
        got = [float(x) for x in re.findall(r"-?\d+\.\d+", r.split("&", 1)[1])]
        if len(got) != len(vals) or any(abs(a - round(b, 3)) > 1e-9 for a, b in zip(got, vals)):
            rep.findings.append(Finding("supp_advisor_v6.tex", label, "cells", got, [round(v, 3) for v in vals]))

    num = r"\$([-+]?[\d.]+)\$"
    # Section 6.1: attribution and the Eq. 7 contrast.
    _quote(rep, "sec:rq1 attribution", sec7,
           r"rate and payload columns alone add " + num + r" \((\d+)/12 folds, Holm \$p = ([\d.]+)\$\), "
           r"and the seven QoS-derived columns \(\$w\(t\)\$-weighted scores and policy shares\) add "
           + num + r" \(\$p = ([\d.]+)\$\)",
           [(1, AC["S+rate,payload vs S"]["delta"]), (2, AC["S+rate,payload vs S"]["won"]),
            (3, AC["S+rate,payload vs S"]["p_holm"]), (4, AC["S+QoS-policy vs S"]["delta"]),
            (5, AC["S+QoS-policy vs S"]["p"])])
    c = C["Rate-I_dyn vs gbm_dep_qos_dyn"]
    _quote(rep, "sec:rq1 Eq. 7", sec7,
           r"raises it to " + num + r" \$\[([\d.]+), ([\d.]+)\]\$ without training.*?"
           r"falls below the rate-weighted reference, which exceeds it by " + num
           + r" \$\[([-+\d.]+), ([-+\d.]+)\]\$ on (\d+) of 12 folds \(nominal Holm \$p = ([\d.]+)\$",
           [(1, rate["i_dyn"]["mean"]), (2, rate["i_dyn"]["ci95"][0]), (3, rate["i_dyn"]["ci95"][1]),
            (4, c["delta"]), (5, c["ci95"][0]), (6, c["ci95"][1]), (7, c["won"]), (8, c["p_holm"])])
    # Section 4.3: I_dyn label reliability, its sqrt(r) ceiling, and the claim that no Table 6
    # ranker exceeds even r.
    rel = a15.get("label_reliability")
    if rel is None:
        rep.findings.append(Finding("idyn_rate_expansion.json", "label_reliability", "block", "absent", "present"))
    else:
        (s1, s2), (m1, m2) = rel["single_seed_range"], rel["seed_mean_range"]
        _quote(rep, "sec:4.3 reliability", _tex("sec4_failure_impact_prediction.tex"),
               r"agrees with another at \$\\rho = ([\d.]+)\$--\$([\d.]+)\$ per fold; the five-seed mean"
               r" used as the label has an estimated reliability of \$r = ([\d.]+)\$--\$([\d.]+)\$"
               r".*?more than \$\\sqrt\{r\} \\approx ([\d.]+)\$--\$([\d.]+)\$",
               [(1, s1), (2, s2), (3, m1), (4, m2), (5, m1 ** 0.5), (6, m2 ** 0.5)])
        rep.checked += 1
        above = {r: f for r, f in rel["folds_above_seed_mean_bound"].items() if f}
        if above:
            rep.findings.append(Finding("sec4_failure_impact_prediction.tex", "every ranker stays below",
                                        "claim", str(above), "no fold above its bound"))

    # Every other main-text mention of Eq. 7's value and of the learned approximation it is set against.
    for fname in ("abstract.tex", "sec1_introduction.tex", "sec7_results.tex", "sec8_discussion.tex",
                  "sec9_conclusion.tex"):
        text = _tex(fname)
        for m in re.finditer(r"rate-weighted[^.]*?\$(?:\\rho = )?(0\.8\d\d)\$", text):
            rep.checked += 1
            if abs(float(m.group(1)) - round(rate["i_dyn"]["mean"], 3)) > 1e-9:
                rep.findings.append(Finding(fname, "Eq. 7 rho", "quote", float(m.group(1)),
                                            round(rate["i_dyn"]["mean"], 3)))
        rep.checked += 1
        if "0.830" in text and f"{gbm_l:.3f}" not in text and fname != "sec7_results.tex":
            rep.findings.append(Finding(fname, "learned approximation", "quote", "absent", round(gbm_l, 3)))


#: Supplement row labels for Amendment 15's per-fold and per-system tables.
_A15_NAMES = {
    "atm_system": "ATM", "av_system": "AV", "enterprise_system": "Enterprise",
    "financial_trading_system": "Financial Trading", "healthcare_system": "Healthcare",
    "hub_and_spoke_system": "Hub-and-Spoke", "industrial_scada_system": "Industrial SCADA",
    "iot_smart_city_system": "IoT Smart City", "logistics_fleet_system": "Logistics Fleet",
    "microservices_system": "Microservices", "realtime_gaming_system": "Real-Time Gaming",
    "telecom_ran_system": "Telecom RAN", "realworld_autoware_ros2": "Autoware",
    "realworld_edgex": "EdgeX", "realworld_homeassistant": "Home Assistant",
    "realworld_cloud_microservices": "Online Boutique", "realworld_trainticket": "Train-Ticket",
}


def check_learning_focus(rep: Report, profile: str = "learning-focus") -> None:
    """Enforce the learning-based and dependency-graph focus of the JSS paper:
    1. Forbidden terms (centralit) must be absent from title, abstract,
       highlights, and Section 9 (Conclusion). "Benchmark" was forbidden too
       until the advisor's v6 revision, whose Conclusion names the released
       benchmark as a contribution.
    2. Non-registered training-free baselines (Degree-raw, RevPR-raw, PR-raw,
       unweighted Topo) must not appear as rows in main body tables (sec*.tex).
    3. Topo-QoS must be the sole training-free baseline in Table tab:predictor_taxonomy.
    """
    if profile == "advisor-revision":
        rep.skipped.append("check_learning_focus: bypassed in advisor-revision profile to accommodate review baseline structure")
        return

    import re
    forbidden_pattern = re.compile(r"centralit")

    # 1. Title in manuscript.tex and title_page.tex
    for fname in ("manuscript.tex", "title_page.tex"):
        content = (LATEX / fname).read_text()
        m = re.search(r"\\title(?:\[.*?\])?\{([^}]+)\}", content, re.DOTALL)
        if m:
            title_text = m.group(1)
            rep.checked += 1
            if forbidden_pattern.search(title_text):
                rep.findings.append(Finding(fname, "title", "forbidden_terms", "found", "absent"))

    # Abstract
    abs_tex = _tex("abstract.tex")
    rep.checked += 1
    if forbidden_pattern.search(abs_tex):
        rep.findings.append(Finding("abstract.tex", "abstract", "forbidden_terms", "found", "absent"))

    # Highlights
    hl_tex = (LATEX / "highlights.tex").read_text()
    rep.checked += 1
    if forbidden_pattern.search(hl_tex):
        rep.findings.append(Finding("highlights.tex", "highlights", "forbidden_terms", "found", "absent"))

    # Section 9 Conclusion
    sec9_tex = _tex("sec9_conclusion.tex")
    rep.checked += 1
    if forbidden_pattern.search(sec9_tex):
        rep.findings.append(Finding("sec9_conclusion.tex", "conclusion", "forbidden_terms", "found", "absent"))

    # 2. No moved baselines as rows in body tables (sec*.tex)
    moved_baselines = ("Degree-raw", "RevPR-raw", "PR-raw", "Topo")
    sec_dir = Path("docs/research/jss/latex/sections")
    for p in sorted(sec_dir.glob("sec*.tex")):
        content = p.read_text(encoding="utf-8")
        for table_chunk in re.findall(r"\\begin\{table\}.*?\\end\{table\}", content, re.DOTALL):
            for bl in moved_baselines:
                rep.checked += 1
                if re.search(r"\\textbf\{" + re.escape(bl) + r"(?:\$[^}]*\$)?\}\s*&", table_chunk):
                    rep.findings.append(Finding(p.name, bl, "body_table_row", "found", "absent"))

    # 3. Topo-QoS is the only training-free predictor in tab:predictor_taxonomy
    sec6 = _tex("sec6_experimental_setup.tex")
    if r"\label{tab:predictor_taxonomy}" in sec6:
        tax_table = sec6[sec6.index(r"\label{tab:predictor_taxonomy}"):sec6.index(r"\end{table}", sec6.index(r"\label{tab:predictor_taxonomy}"))]
        key = "training-free baseline (registered comparator)"
        rep.checked += 1
        if key not in tax_table:
            rep.findings.append(Finding("tab:predictor_taxonomy", "baseline block", "heading", "absent", key))
        else:
            start = tax_table.index(key)
            end = tax_table.index(r"\midrule", start)
            tf_block = tax_table[start:end]
            tf_predictors = []
            for line in tf_block.splitlines():
                if "&" in line and not line.strip().startswith(r"\multicolumn"):
                    cells = _cells(line)
                    if cells:
                        m = re.search(r"\\texttt\{([^}]+)\}", cells[0])
                        if m:
                            tf_predictors.append(m.group(1))
            rep.checked += 1
            if tf_predictors != ["Topo-QoS"]:
                rep.findings.append(Finding("tab:predictor_taxonomy", "tf_predictors", "only_Topo-QoS", tf_predictors, ["Topo-QoS"]))


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
    ap.add_argument("--profile", default="learning-focus",
                    choices=["learning-focus", "advisor-revision"],
                    help="validation profile: 'learning-focus' (enforces strict learning-focus framing) "
                         "or 'advisor-revision' (supports advisor review baseline structure)")
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
    check_referee_round7(rep)
    check_contrasts_matched(rep)
    check_omnibus_holm(rep)
    check_engine_regimes(rep)
    check_dependency_graph(rep)
    check_reference_demotion(rep)
    check_round8(rep)
    check_cost_ll(rep)
    check_amendment16(rep)
    check_amendment17(rep)
    check_controls_digest(rep)
    check_rate_expansion(rep)
    check_learning_focus(rep, profile=args.profile)

    # Data Availability quotes this script's own count; it must not go stale.
    m = re.search(r"mechanically verifies \$?([\d,{}]+)\$? reported figures", _tex("declarations.tex"))
    quoted = int(re.sub(r"\D", "", m.group(1))) if m else None
    if quoted != rep.checked:
        rep.findings.append(Finding("declarations.tex", "Data Availability figure count", "quote",
                                    quoted, rep.checked))

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
