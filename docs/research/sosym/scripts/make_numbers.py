#!/usr/bin/env python3
"""Generate docs/research/sosym/latex/numbers.tex from the artifacts in results/sosym/.

Every empirical number in the SoSyM tool paper is a macro defined here, so no
result is typed into a section file by hand. A missing artifact yields "??"
(and a warning) rather than a stale value.

Usage:
    PYTHONPATH=. python docs/research/sosym/scripts/make_numbers.py
"""
import json
import re
import sys
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "results/sosym"
OUT = ROOT / "docs/research/sosym/latex/numbers.tex"

NAMES = {
    "av_system": "Autonomous vehicle", "iot_smart_city_system": "IoT smart city",
    "financial_trading_system": "Financial trading", "healthcare_system": "Healthcare",
    "hub_and_spoke_system": "Hub-and-spoke", "microservices_system": "Microservices",
    "enterprise_system": "Enterprise", "tiny_system": "Tiny",
    "realworld_autoware_ros2": "Autoware.Universe", "realworld_trainticket": "Train-Ticket",
    "realworld_cloud_microservices": "Cloud e-commerce mesh",
    "realworld_homeassistant": "Home Assistant", "realworld_edgex": "EdgeX Foundry",
}
CORPUS = {"generated": "synthetic", "transcribed": "open-source"}
OPERATORS = {"node_reallocation": "Anti-affinity reallocation", "qos_upgrade": "QoS hardening",
             "topic_split": "Topic split"}
CATALOG_IDS = ["SPOF", "FAILURE_HUB", "GOD_COMPONENT", "SYSTEMIC_RISK", "COMPOUND_RISK", "BRIDGE_EDGE",
               "BOTTLENECK_EDGE", "BROKER_OVERLOAD", "DEEP_PIPELINE", "CYCLE", "CONCENTRATION_RISK",
               "TOPIC_FANOUT", "QOS_MISMATCH", "HUB_AND_SPOKE", "CHATTY_PAIR", "ORPHANED_TOPIC",
               "UNSTABLE_INTERFACE", "CHAIN", "ISOLATED"]

macros: dict = {}
missing: list = []


def load(name):
    p = RES / name
    if not p.exists():
        missing.append(name)
        return None
    return json.loads(p.read_text())


def m(name, value):
    macros[name] = str(value)


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}\\%"


def f2(x):
    return f"{x:.2f}"


def topo(scenario):
    return json.loads((ROOT / "data/scenarios" / f"{scenario}.json").read_text())


def n_relations(t):
    return sum(len(v) for v in t["relationships"].values())


def n_components(t):
    return sum(len(t.get(k, [])) for k in ("applications", "libraries", "topics", "brokers", "nodes",
                                            "network_devices"))


def fmt_secs(s):
    h, r = divmod(int(round(s)), 3600)
    mi = r // 60
    if not h and not mi:
        return f"{int(round(s))}~s"
    return f"{h}~h {mi}~min" if h else f"{mi}~min"


def signed(x, digits):
    """Signed number with a typographic minus for LaTeX text mode."""
    return f"{x:+.{digits}f}".replace("-", "$-$")


# ── provenance ────────────────────────────────────────────────────────────────
prov = load("provenance.json") or {}
m("ProvCommit", (prov.get("commit") or "??")[:8])
m("ProvCorpus", (prov.get("corpus_digest") or "??")[:12])

# ── agreement (detection_validation) ─────────────────────────────────────────
app = load("detection_validation_app.json")
if app:
    s = app["summary"]
    m("RhoQApp", f2(s["q_composite"]["mean_spearman_rho"]))
    m("RhoBetwApp", f2(s["betweenness"]["mean_spearman_rho"]))
    m("RhoDegApp", f2(s["degree"]["mean_spearman_rho"]))
    q = [r["predictors"]["q_composite"]["spearman_rho"] for r in app["per_scenario"]]
    d = [r["predictors"]["degree"]["spearman_rho"] for r in app["per_scenario"]]
    m("QBeatsDegApp", sum(a > b for a, b in zip(q, d)))
    m("QvsDegP", f"{wilcoxon(q, d).pvalue:.3f}")
    m("LessonSpofZeroApp", sum(1 for r in app["per_scenario"] if r["spof"]["n_predicted_ap"] == 0))
    m("LessonSpofFOneApp", f2(s["spof"]["mean_f1"]))
sysl = load("detection_validation_system.json")
if sysl:
    bt = sysl["summary"]["q_by_type"]
    m("RhoQSysApp", f2(bt["Application"]["mean_spearman_rho"]))
    m("RhoQSysNode", f2(bt["Node"]["mean_spearman_rho"]))
    m("RhoQSysBroker", f2(bt["Broker"]["mean_spearman_rho"]))

# ── seed sweep: catalog, cost, open-source models, which patterns fire ───────
sweep = load("detection_seed_sweep.json")
if sweep:
    t3 = sweep["table3_catalog_vs_oracle"]
    for tag, key in (("Generated", "generated"), ("Transcribed", "transcribed")):
        b = t3[key]
        m(f"Kappa{tag}", f"{b['cohens_kappa']['mean']:+.2f}")
        m(f"Prec{tag}", f2(b["precision"]["mean"]))
        m(f"Rec{tag}", f2(b["recall"]["mean"]))
        m(f"Impl{tag}", f"{b['pct_scored_implicated']['mean']:.0f}\\%")
    rows_ok = [r for r in sweep["rows"] if "error" not in r]
    id_of_name = {}
    from saag.analysis.antipattern_detector import CATALOG  # noqa: E402
    id_of_name = {spec.name: pid for pid, spec in CATALOG.items()}
    fired = set()
    for r in rows_ok:
        fired |= {id_of_name[n] for n, c in r["findings"]["by_pattern"].items() if c}
    for pid in CATALOG_IDS:
        mark = "excl." if pid == "DEEP_PIPELINE" else ("\\checkmark" if pid in fired else "")
        m(f"trig@{pid}", mark)
    m("NPatternsFired", len(fired))

    scale_rows, gate_means, det_max = [], [], 0.0
    for r in sweep["table2_gate_cost"]:
        a, dt, g = r["analyze_seconds"], r["detect_seconds"], r["gate_seconds"]
        gate_means.append(g["mean"])
        det_max = max(det_max, dt["mean"])
        scale_rows.append(f"{NAMES[r['scenario']]} & {CORPUS[r['corpus']]} & {r['n_components']} & "
                          f"${a['mean']:.2f} \\pm {a['std']:.2f}$ & ${dt['mean']:.2f}$ & "
                          f"${g['mean']:.2f} \\pm {g['std']:.2f}$ \\\\")
    m("ScaleRows", "\n".join(scale_rows))
    m("DetectorSecondsMax", f2(det_max) if det_max >= 0.01 else "0.01")
    m("GateSecondsMax", f"{max(gate_means):.0f}")
    m("AbsGateRange", f"{min(gate_means):.2f}--{max(gate_means):.0f}")

    by_s = {}
    for r in rows_ok:
        by_s.setdefault(r["scenario"], []).append(r)
    rw_rows, flagged = [], []
    for scen in ("realworld_autoware_ros2", "realworld_trainticket", "realworld_cloud_microservices",
                 "realworld_edgex", "realworld_homeassistant"):
        grp = by_s.get(scen, [])
        if not grp:
            continue
        t = topo(scen)
        nf = sum(r["findings"]["n_findings"] for r in grp) / len(grp)
        npat = max(sum(1 for c in r["findings"]["by_pattern"].values() if c) for r in grp)
        fl = sum(r["findings"]["pct_components_flagged"] for r in grp) / len(grp)
        flagged.append(fl)
        gs = sum(r["gate_seconds"] for r in grp) / len(grp)
        codes = sorted({r["gate_exit_code"] for r in grp})
        rw_rows.append(f"{NAMES[scen]} & {n_components(t)} & {n_relations(t)} & {nf:.0f} & {npat} & "
                       f"{fl:.1f}\\% & {gs:.2f} & {'/'.join(map(str, codes))} \\\\")
    m("RealWorldRows", "\n".join(rw_rows))
    if flagged:
        m("RealWorldFlaggedRange", f"{min(flagged):.0f}--{max(flagged):.0f}\\%")
    gd = sweep.get("gate_decisions") or {}
    m("NAlwaysExitTwo", f"{gd.get('n_scenarios_always_exit_2', '??')}")
    m("NGateModels", f"{gd.get('n_scenarios_total', '??')}")

# ── refactoring ──────────────────────────────────────────────────────────────
pres = load("prescribe_all.json")
timing = {}
tp = RES / "prescribe_timing.txt"
if tp.exists():
    for line in tp.read_text().splitlines():
        mt = re.match(r"WALL (\S+) ([\d.]+) s", line)
        if mt:
            timing[mt.group(1)] = float(mt.group(2))
else:
    missing.append("prescribe_timing.txt")
if pres:
    recs = pres["scenarios"] if isinstance(pres, dict) and "scenarios" in pres else pres
    rows, cand, acc, deltas, sri_up, sims = [], 0, 0, [], 0, 0
    ops = {}
    for r in recs:
        c, a = r["n_candidate_edits"], r["n_accepted_edits"]
        cand += c
        acc += a
        sims += 9 * (c + 1)
        di = r["mean_cascade_impact_reduction"]
        if di is not None:
            deltas.append(di)
        sri_up += r["sri_improvement"] > 0
        for e in r["edit_verdicts"]:
            o = ops.setdefault(e["kind"], [0, 0])
            o[0] += 1
            o[1] += bool(e["accepted"])
        name = NAMES[r["file"].replace(".json", "")]
        tsec = timing.get(r["file"])
        rows.append(f"{name} & {c} & {a} & {pct(a / c) if c else '--'} & "
                    f"{'--' if di is None else signed(di, 3)} & {signed(r['sri_improvement'], 4)} & "
                    f"{fmt_secs(tsec) if tsec else '--'} \\\\")
    m("PrescribeRows", "\n".join(rows))
    m("PrescribeCandidates", f"{cand:,}".replace(",", "{,}"))
    m("PrescribeAccepted", f"{acc:,}".replace(",", "{,}"))
    m("PrescribeAcceptedPct", pct(acc / cand))
    m("PrescribeRejectedPct", pct(1 - acc / cand, 0))
    m("PrescribeImprovedScenarios", sum(1 for x in deltas if x > 0))
    m("PrescribeRegressedScenarios", sum(1 for x in deltas if x < 0))
    m("PrescribeMeanDeltaI", f"{sum(deltas) / len(deltas):+.3f}" if deltas else "??")
    m("PrescribeSRIImproved", sri_up)
    m("PrescribeSimulations", f"{sims:,}".replace(",", "{,}"))
    m("PrescribeWallClock", fmt_secs(sum(timing.values())) if timing else "??")
    m("OperatorRows", "\n".join(
        f"{OPERATORS.get(k, k)} & {v[0]} & {v[1]} & {pct(v[1] / v[0])} \\\\"
        for k, v in sorted(ops.items(), key=lambda kv: -kv[1][1] / kv[1][0])))

ex = load("prescribe_example.json")
if ex:
    m("ExPrescribeCandidates", ex["n_candidates"])
    m("ExPrescribeAccepted", ex["n_accepted"])
    m("ExPrescribeWorse", ex["n_rejected_worse"])

# ── CI scenario ──────────────────────────────────────────────────────────────
ci = load("ci_scenario.json")
if ci:
    mini = ci["mini_atm"]
    m("CIMiniBase", mini["n_base"])
    m("CIMiniCand", mini["n_candidate"])
    m("CIMiniCarried", mini["delta"]["carried"])
    m("CIMiniBlocking", len(mini["delta"]["blocking"]))
    m("CIMiniAdvisory", len(mini["delta"]["new_advisory"]) + len(mini["delta"]["near_fence"]))
    rows, tot, ab2, de = [], 0, 0, {0: 0, 1: 0, 2: 0}
    for r in ci["corpus"]:
        a = {int(k): v for k, v in r["absolute"].items()}
        dl = {int(k): v for k, v in r["delta"].items()}
        tot += r["edits"]
        ab2 += a.get(2, 0)
        for k in de:
            de[k] += dl.get(k, 0)
        trip = lambda c: f"{c.get(2, 0)} / {c.get(1, 0)} / {c.get(0, 0)}"  # noqa: E731
        rows.append(f"{NAMES[r['scenario']]} & {CORPUS[r['corpus']]} & {r['n_components']} & {r['edits']} & "
                    f"{trip(a)} & {trip(dl)} \\\\")
    m("CIRows", "\n".join(rows))
    m("CIEdits", ci["corpus"][0]["edits"] if ci["corpus"] else "??")
    if tot:
        m("CIAbsBlockPct", pct(ab2 / tot, 0))
        m("CIDeltaBlockPct", pct(de[2] / tot, 0))
        m("CIDeltaWarnPct", pct(de[1] / tot, 0))
        m("CIDeltaPassPct", pct(de[0] / tot, 0))

# ── write ────────────────────────────────────────────────────────────────────
REQUIRED = ["ProvCommit", "ProvCorpus", "RhoQApp", "RhoBetwApp", "RhoDegApp", "QBeatsDegApp", "QvsDegP",
            "LessonSpofZeroApp", "LessonSpofFOneApp", "RhoQSysApp", "RhoQSysNode", "RhoQSysBroker",
            "KappaGenerated", "KappaTranscribed", "PrecGenerated", "PrecTranscribed", "RecGenerated",
            "RecTranscribed", "ImplGenerated", "ImplTranscribed", "ScaleRows", "DetectorSecondsMax",
            "GateSecondsMax", "AbsGateRange", "RealWorldRows", "RealWorldFlaggedRange", "PrescribeRows",
            "PrescribeCandidates", "PrescribeAccepted", "PrescribeAcceptedPct", "PrescribeRejectedPct",
            "PrescribeImprovedScenarios", "PrescribeRegressedScenarios", "PrescribeMeanDeltaI",
            "PrescribeSRIImproved", "PrescribeSimulations", "PrescribeWallClock", "OperatorRows",
            "ExPrescribeCandidates", "ExPrescribeAccepted", "ExPrescribeWorse", "CIMiniBase", "CIMiniCand",
            "CIMiniCarried", "CIMiniBlocking", "CIMiniAdvisory", "CIRows", "CIEdits", "CIAbsBlockPct",
            "CIDeltaBlockPct", "CIDeltaWarnPct", "CIDeltaPassPct"]
TABLE_MACROS = {"ScaleRows", "RealWorldRows", "PrescribeRows", "OperatorRows", "CIRows"}

lines = ["% Generated by docs/research/sosym/scripts/make_numbers.py -- do not edit by hand.",
         f"% Source artifacts: results/sosym/ (commit {macros.get('ProvCommit', '??')}).", ""]
for name in REQUIRED:
    val = macros.get(name)
    if val is None:
        val = "\\multicolumn{2}{l}{?? (artifact missing)} \\\\" if name in TABLE_MACROS else "??"
    lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
lines.append("")
lines.append("% Which catalog patterns fire on the 13-model corpus (system layer, seed sweep).")
lines.append("\\newcommand{\\trig}[1]{\\csname trig@#1\\endcsname}")
for pid in CATALOG_IDS:
    lines.append(f"\\expandafter\\def\\csname trig@{pid}\\endcsname{{{macros.get('trig@' + pid, '??')}}}")
OUT.write_text("\n".join(lines) + "\n")
if missing:
    print("WARNING: missing artifacts: " + ", ".join(missing), file=sys.stderr)
print(f"wrote {OUT} ({len(macros)} values)")
