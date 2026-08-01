#!/usr/bin/env python3
"""
reproduce/detection_validation.py — does the anti-pattern catalog find real risk?
================================================================================

Produces ``results/detection_validation.json``.

The AuSE manuscript's RQ1 asks two things of the 21-pattern catalog
(``saag/analysis/antipattern_detector.py``):

1. **Efficacy** — do its findings correspond to components that actually matter
   when the system fails?
2. **Precision** — does it stay quiet on a well-structured topology, rather than
   flagging everything and letting the reader find the signal?

Neither had a reproduction script. The figures the draft carried
(:math:`\\rho = 0.876`, :math:`F_1 = 0.923`, precision 0.912, recall 0.857,
Top-5 0.80) trace to a prior conference publication and appear in this repository
only as prose and as hard-coded fixtures in ``tests/test_visualization_service.py``.
This script measures them here, against a committed oracle, so the manuscript can
cite an artifact instead of a memory.

What is measured
----------------
Ground truth is ``I_comp(v)`` — the ``FailureSimulator`` composite (0.35
reachability + 0.25 fragmentation + 0.25 throughput + 0.15 flow disruption),
obtained by exhaustive single-component removal. This is the oracle the
Validate-stage gates and the prescriptive acceptance criterion already run on, so
detection and remediation are scored against the same yardstick.

``I_comp`` is *not* ``I*`` (``FaultInjector``), and the two agree only moderately
— mean Spearman rho 0.4046 in ``results/convergent_validity.json``. A number
produced here is therefore not evidence for a claim measured against ``I*``.

Three predictors are scored against that oracle on the same node set:

``Q(v)``       the RMAV composite criticality score — the ranking the framework
               actually surfaces.
``catalog``    the binary "was this component flagged by at least one
               CRITICAL/HIGH anti-pattern" verdict — the *named* finding, which
               is what distinguishes this work from an unnamed criticality score.
``betweenness`` / ``degree``
               single-metric structural baselines, so the composite's value is a
               measured margin rather than an assertion.

Critical-set thresholding follows ``saag/validation/validator.py`` exactly:
box-plot top tier (>= Q3) when n >= 20, else a top-20% percentile mask, applied
identically to prediction and to truth. Because both sides are thresholded to
near-equal sizes, precision and recall converge by construction; that is a
property of the rule, not a finding, and the paper says so.

Every reported quantity is rank-based (Spearman, top-K overlap, quantile masks),
so results are invariant to the monotone ``robust_sigmoid_scale_dict`` rescaling
``ValidationService`` applies internally, and no rescaling is done here.

Wall-clock timing
-----------------
``detect_seconds`` times load -> analyze -> RMAV -> all 21 detectors, i.e. exactly
the work a CI/CD detection gate performs. It excludes the oracle sweep, which is
an evaluation cost and never runs in a gate.

Usage
-----
    PYTHONPATH=. python reproduce/detection_validation.py
    PYTHONPATH=. python reproduce/detection_validation.py --scenarios tiny_system
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

logger = logging.getLogger("detection_validation")

RESULTS_DIR = Path("results")

#: The eight-scenario detection suite of the AuSE manuscript. Wider than
#: ``reproduce.main_table.ALL_SCENARIOS`` (seven), which excludes the
#: deterministic tiny regression fixture because it carries no
#: domain-representative signal for the learned predictors.
DETECTION_SCENARIOS = [
    "av_system",
    "iot_smart_city_system",
    "financial_trading_system",
    "healthcare_system",
    "hub_and_spoke_system",
    "microservices_system",
    "enterprise_system",
    "tiny_system",
]

#: Severities that constitute a "flagged" component for the catalog predictor.
FLAGGING_SEVERITIES = ("CRITICAL", "HIGH")

#: Patterns held out of the default run, with the reason recorded in the artifact.
#:
#: ``DEEP_PIPELINE`` enumerates every simple source-to-sink path via
#: ``nx.all_simple_paths`` (antipattern_detector.py:_enumerate_deep_pipelines) and
#: emits one finding per path, keyed on the path string rather than a component.
#: On the 29-component ``tiny_system`` fixture that is 247,761 findings; on the
#: 50-application healthcare topology it does not terminate within ten minutes.
#: It is excluded so the remaining twenty detectors can be measured at all. The
#: exclusion is a measured defect, not a methodological choice, and belongs in the
#: manuscript as such.
DEFAULT_EXCLUDED_PATTERNS = ["DEEP_PIPELINE"]


# =============================================================================
# Predictors and oracle
# =============================================================================

def _analyze(
    scenario: str, layer: str, excluded: List[str]
) -> Tuple[Any, List[Any], Dict[str, Any]]:
    """Run the detection gate: load -> analyze -> RMAV -> catalog detectors.

    Detectors are invoked one at a time rather than through
    ``MultiLayerAnalysisUseCase`` so each one's wall-clock cost and finding count
    is attributable. A CI/CD gate budget is spent on exactly this work, and
    knowing *which* detector spends it is the difference between a runtime number
    and an actionable one.
    """
    from saag.analysis.antipattern_detector import CATALOG, AntiPatternDetector
    from saag.analysis.service import AnalysisService
    from saag.infrastructure.memory_repo import MemoryRepository
    from saag.prediction.service import PredictionService
    from reproduce.ahp_sensitivity import _load_topology

    topology = _load_topology(scenario)
    active = [pid for pid in CATALOG if pid not in excluded]

    started = time.perf_counter()
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)
    layer_res = AnalysisService(repo).analyze_layer(layer)
    quality = PredictionService().predict_quality(layer_res.structural)
    analyze_seconds = time.perf_counter() - started

    problems: List[Any] = []
    per_detector: Dict[str, Dict[str, Any]] = {}
    for pid in active:
        t0 = time.perf_counter()
        found = AntiPatternDetector(active_patterns=[pid]).detect(quality, layer=layer)
        per_detector[pid] = {
            "seconds": round(time.perf_counter() - t0, 3),
            "n_findings": len(found),
        }
        problems.extend(found)

    timing = {
        "analyze_seconds": round(analyze_seconds, 2),
        "detect_seconds": round(sum(d["seconds"] for d in per_detector.values()), 2),
        "per_detector": dict(
            sorted(per_detector.items(), key=lambda kv: -kv[1]["seconds"])
        ),
    }
    timing["gate_seconds"] = round(
        timing["analyze_seconds"] + timing["detect_seconds"], 2
    )
    return quality, problems, timing


def _oracle(scenario: str, layer: str, seed: int, threshold: float) -> Dict[str, Any]:
    """I_comp(v) by exhaustive component removal, plus its availability channel.

    Goes through ``SimulationService`` rather than ``FailureSimulator`` directly
    so the discrete-event baseline flows are primed exactly as they are for the
    Validate-stage gates; the raw path measures flow disruption differently.
    """
    from saag.infrastructure.memory_repo import MemoryRepository
    from saag.simulation.service import SimulationService
    from reproduce.ahp_sensitivity import _load_topology

    repo = MemoryRepository()
    repo.save_graph(_load_topology(scenario), clear=True)
    results = SimulationService(repo).run_failure_simulation_exhaustive(
        layer=layer, propagation_threshold=threshold, seed=seed,
    )
    return {
        "composite": {r.target_id: float(r.impact.composite_impact) for r in results},
        "availability": {
            r.target_id: float(getattr(r.impact, "availability_impact", 0.0))
            for r in results
        },
    }


# =============================================================================
# Scoring
# =============================================================================

def _critical_mask(values: Sequence[float]) -> List[bool]:
    """The critical-set rule of ``Validator._validate_group``, verbatim.

    Box-plot top tier (>= Q3) once there are enough components to estimate
    quartiles; a top-20% percentile mask below that. Applied identically to
    predictions and to ground truth.
    """
    from saag.analysis.classifier import BoxPlotClassifier

    vals = list(values)
    if not vals:
        return []
    if len(vals) >= 20:
        q3 = BoxPlotClassifier().compute_stats(vals).q3
        return [v >= q3 for v in vals]
    cutoff = float(np.percentile(vals, 80.0))
    return [v >= cutoff for v in vals]


def _rank_block(
    predicted: Dict[str, float], actual: Dict[str, float], ids: List[str]
) -> Dict[str, Any]:
    """Correlation + critical-set classification + top-K overlap for one predictor."""
    from saag.validation.metric_calculator import (
        calculate_classification,
        calculate_correlation,
        calculate_ranking,
    )

    pred_vals = [predicted[k] for k in ids]
    actual_vals = [actual[k] for k in ids]

    if np.ptp(pred_vals) == 0 or np.ptp(actual_vals) == 0:
        return {"note": "constant predictor or oracle on the shared node set"}

    corr = calculate_correlation(pred_vals, actual_vals)
    cls = calculate_classification(_critical_mask(pred_vals), _critical_mask(actual_vals))
    rank = calculate_ranking(
        {k: predicted[k] for k in ids}, {k: actual[k] for k in ids}, k_values=[5, 10]
    )

    return {
        "spearman_rho": round(corr.spearman, 4),
        "spearman_p": round(corr.spearman_p, 6),
        "spearman_ci": [round(corr.spearman_ci_lower, 4), round(corr.spearman_ci_upper, 4)],
        "precision": round(cls.precision, 4),
        "recall": round(cls.recall, 4),
        "f1": round(cls.f1_score, 4),
        "f1_ci": [round(cls.f1_ci_lower, 4), round(cls.f1_ci_upper, 4)],
        "cohens_kappa": round(cls.cohens_kappa, 4),
        "top_5_overlap": round(rank.top_5_overlap, 4),
        "top_10_overlap": round(rank.top_10_overlap, 4),
        "ndcg_10": round(rank.ndcg_10, 4),
    }


def _implicated(entity_id: str, component_ids: set) -> set:
    """Components a finding implicates, whether it is keyed on one or not.

    Only some detectors key on a component. ``BRIDGE_EDGE``/``BOTTLENECK_EDGE``
    key on ``"A0->A10"``, ``CHATTY_PAIR`` on ``"A0<->A10"``, ``CYCLE`` on a
    ``" -> "``-joined member list. A practitioner reading "Bottleneck Dependency
    on A0->A10" treats both endpoints as implicated, so scoring the catalog on
    component-keyed findings alone would understate it to near zero. Tokens that
    do not resolve to a component (e.g. the ``CHAIN-`` synthetic key) are dropped.
    """
    if entity_id in component_ids:
        return {entity_id}
    tokens = entity_id.replace("<->", "->").split("->")
    return {t.strip() for t in tokens if t.strip() in component_ids}


def _catalog_block(
    problems: List[Any], actual: Dict[str, float], ids: List[str], component_ids: set
) -> Dict[str, Any]:
    """How the *named* findings score against the oracle's critical set.

    This is the pattern-level claim, and it differs from the Q(v) block above in
    one way that matters: the flagged set is whatever the detectors chose to
    flag, so its size is not pinned to the truth set's size. Precision and recall
    can therefore diverge, and a catalog that flags nothing scores zero rather
    than being silently excused.
    """
    from saag.validation.metric_calculator import calculate_classification

    scored = set(ids)
    flagged: set = set()
    direct: set = set()
    for p in problems:
        if p.severity not in FLAGGING_SEVERITIES:
            continue
        hits = _implicated(p.entity_id, component_ids) & scored
        flagged |= hits
        if p.entity_id in scored:
            direct.add(p.entity_id)

    actual_crit = dict(zip(ids, _critical_mask([actual[k] for k in ids])))
    cls = calculate_classification(
        [k in flagged for k in ids], [actual_crit[k] for k in ids]
    )
    return {
        "n_flagged": len(flagged),
        "n_flagged_directly": len(direct),
        "n_actual_critical": sum(actual_crit.values()),
        "precision": round(cls.precision, 4),
        "recall": round(cls.recall, 4),
        "f1": round(cls.f1_score, 4),
        "cohens_kappa": round(cls.cohens_kappa, 4),
        "true_positives": cls.true_positives,
        "false_positives": cls.false_positives,
        "false_negatives": cls.false_negatives,
    }


def _findings_block(
    problems: List[Any], component_ids: set, n_components: int
) -> Dict[str, Any]:
    """Raw finding counts — the over-flagging question, answered by volume.

    A catalog that flags a third of the system is not usable regardless of how
    well its scores correlate, so this is reported next to the correlations
    rather than buried.

    Findings are split by whether their ``entity_id`` names a component. Several
    detectors key on an edge (``BRIDGE_EDGE``, ``BOTTLENECK_EDGE``, ``CHATTY_PAIR``)
    or on a path (``DEEP_PIPELINE``, ``CHAIN``), and counting those against a
    component population would inflate the flagged fraction past 100%.
    """
    by_severity: Dict[str, int] = {}
    by_pattern: Dict[str, int] = {}
    for p in problems:
        by_severity[p.severity] = by_severity.get(p.severity, 0) + 1
        by_pattern[p.name] = by_pattern.get(p.name, 0) + 1

    component_findings = [p for p in problems if p.entity_id in component_ids]
    flagged = {
        p.entity_id for p in component_findings if p.severity in FLAGGING_SEVERITIES
    }

    return {
        "n_findings": len(problems),
        "n_component_findings": len(component_findings),
        "n_non_component_findings": len(problems) - len(component_findings),
        "n_distinct_patterns": len(by_pattern),
        "by_severity": dict(sorted(by_severity.items())),
        "by_pattern": dict(sorted(by_pattern.items(), key=lambda kv: -kv[1])),
        "n_components_flagged": len(flagged),
        "pct_components_flagged": (
            round(100.0 * len(flagged) / n_components, 2) if n_components else 0.0
        ),
    }


def evaluate(
    scenario: str, layer: str, seed: int, threshold: float, excluded: List[str]
) -> Dict[str, Any]:
    """Score one scenario's catalog and criticality ranking against the oracle."""
    from saag.validation.metric_calculator import calculate_spof_f1

    quality, problems, timing = _analyze(scenario, layer, excluded)
    oracle = _oracle(scenario, layer, seed=seed, threshold=threshold)

    actual = oracle["composite"]
    components = {c.id: c for c in quality.components}
    ids = sorted(set(components) & set(actual))

    row: Dict[str, Any] = {
        "scenario": scenario,
        "layer": layer,
        "n_components": len(components),
        "n_scored": len(ids),
        "n_unscored": len(components) - len(ids),
        "timing": timing,
        "gate_seconds": timing["gate_seconds"],
        "findings": _findings_block(problems, set(components), len(components)),
    }
    if len(ids) < 3:
        row["note"] = "insufficient overlap between analysis and oracle"
        return row

    predictors = {
        "q_composite": {k: float(components[k].scores.overall) for k in ids},
        "betweenness": {k: float(components[k].structural.betweenness) for k in ids},
        "degree": {k: float(components[k].structural.degree) for k in ids},
    }
    row["predictors"] = {
        name: _rank_block(scores, actual, ids) for name, scores in predictors.items()
    }
    row["catalog"] = _catalog_block(problems, actual, ids, set(components))

    # Pooling node types across a heterogeneous graph is exactly the Simpson
    # pattern this project already guards against elsewhere, and it bites here:
    # the pooled figure can sit outside the per-type range. Report both.
    by_type: Dict[str, Any] = {}
    groups: Dict[str, List[str]] = {}
    for k in ids:
        groups.setdefault(components[k].type, []).append(k)
    for ctype, members in sorted(groups.items()):
        if len(members) < 3:
            continue
        by_type[ctype] = {
            "n": len(members),
            **_rank_block(predictors["q_composite"], actual, members),
        }
    row["q_by_type"] = by_type

    # The over-flagging measure that matters: of the components the oracle
    # scores, what fraction does the catalog implicate at CRITICAL/HIGH?
    row["findings"]["pct_scored_implicated"] = round(
        100.0 * row["catalog"]["n_flagged"] / len(ids), 2
    )

    # SPOF is the one pattern with a purpose-built metric already in the repo,
    # and the catalog's most-cited critical finding — so it gets scored directly
    # rather than only through the aggregate.
    #
    # Two thresholds, because the repo default is not on this oracle's scale.
    # ``calculate_spof_f1`` defaults to ia_threshold=0.50, but availability impact
    # on these topologies tops out around 0.36, so the fixed threshold admits no
    # true SPOF at all and returns a structural zero. The adaptive variant uses
    # the oracle's own upper quartile, which is what the rest of this script does.
    predicted_ap = {k: float(components[k].structural.ap_c_directed) for k in ids}
    actual_ia = {k: oracle["availability"].get(k, 0.0) for k in ids}
    ia_values = list(actual_ia.values())
    adaptive_ia = float(np.percentile(ia_values, 75.0)) if ia_values else 0.5

    row["spof"] = {
        **{k: round(v, 4) for k, v in calculate_spof_f1(
            predicted_ap, actual_ia, ia_threshold=adaptive_ia).items()},
        "ia_threshold": round(adaptive_ia, 4),
        "ia_max": round(max(ia_values), 4) if ia_values else 0.0,
        "fixed_050": {
            k: round(v, 4) for k, v in calculate_spof_f1(
                predicted_ap, actual_ia, ia_threshold=0.50).items()
        },
        "n_predicted_ap": sum(1 for v in predicted_ap.values() if v > 0.0),
    }
    return row


# =============================================================================
# Aggregation and CLI
# =============================================================================

def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-scenario means, plus the size split the 'improves at scale' claim needs."""
    scored = [r for r in rows if "predictors" in r]

    def _mean(vals: List[float]) -> Any:
        return round(float(np.mean(vals)), 4) if vals else None

    def _collect(getter) -> List[float]:
        out = []
        for r in scored:
            v = getter(r)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    summary: Dict[str, Any] = {"n_scenarios_measured": len(scored)}

    for name in ("q_composite", "betweenness", "degree"):
        summary[name] = {
            "mean_spearman_rho": _mean(_collect(lambda r, n=name: r["predictors"][n].get("spearman_rho"))),
            "mean_f1": _mean(_collect(lambda r, n=name: r["predictors"][n].get("f1"))),
            "mean_precision": _mean(_collect(lambda r, n=name: r["predictors"][n].get("precision"))),
            "mean_recall": _mean(_collect(lambda r, n=name: r["predictors"][n].get("recall"))),
            "mean_top_5_overlap": _mean(_collect(lambda r, n=name: r["predictors"][n].get("top_5_overlap"))),
        }

    summary["catalog"] = {
        "mean_precision": _mean(_collect(lambda r: r["catalog"].get("precision"))),
        "mean_recall": _mean(_collect(lambda r: r["catalog"].get("recall"))),
        "mean_f1": _mean(_collect(lambda r: r["catalog"].get("f1"))),
        "mean_pct_scored_implicated": _mean(
            _collect(lambda r: r["findings"].get("pct_scored_implicated"))),
    }
    summary["spof"] = {"mean_f1": _mean(_collect(lambda r: r["spof"].get("f1")))}

    # Per-type means, and whether the pooled figure sits inside their range —
    # the check that tells a reader whether the pooled number means anything.
    per_type: Dict[str, List[float]] = {}
    for r in scored:
        for ctype, blk in r.get("q_by_type", {}).items():
            if isinstance(blk.get("spearman_rho"), float):
                per_type.setdefault(ctype, []).append(blk["spearman_rho"])
    summary["q_by_type"] = {
        ctype: {"mean_spearman_rho": _mean(vals), "n_scenarios": len(vals)}
        for ctype, vals in sorted(per_type.items())
    }
    type_means = [v["mean_spearman_rho"] for v in summary["q_by_type"].values()
                  if isinstance(v["mean_spearman_rho"], float)]
    pooled = summary["q_composite"]["mean_spearman_rho"]
    summary["pooling_check"] = {
        "pooled_mean_rho": pooled,
        "per_type_mean_range": (
            [min(type_means), max(type_means)] if type_means else None),
        "pooled_inside_per_type_range": (
            bool(type_means) and isinstance(pooled, float)
            and min(type_means) <= pooled <= max(type_means)),
        "note": (
            "False means the pooled figure is not a summary of the per-type "
            "figures — a Simpson's-paradox effect — and only the stratified "
            "numbers should be quoted."
        ),
    }

    # Does accuracy actually rise with scale? Split rather than asserted.
    large = [r for r in scored if r["n_components"] >= 150]
    small = [r for r in scored if r["n_components"] < 150]
    summary["by_scale"] = {
        "large_ge_150_components": {
            "scenarios": [r["scenario"] for r in large],
            "mean_spearman_rho": _mean(
                [r["predictors"]["q_composite"]["spearman_rho"] for r in large
                 if isinstance(r["predictors"]["q_composite"].get("spearman_rho"), float)]),
        },
        "small_lt_150_components": {
            "scenarios": [r["scenario"] for r in small],
            "mean_spearman_rho": _mean(
                [r["predictors"]["q_composite"]["spearman_rho"] for r in small
                 if isinstance(r["predictors"]["q_composite"].get("spearman_rho"), float)]),
        },
    }
    gate = _collect(lambda r: r["gate_seconds"])
    summary["gate_seconds"] = {
        "min": round(min(gate), 2) if gate else None,
        "max": round(max(gate), 2) if gate else None,
    }

    # Which detector actually spends the gate's budget, summed across scenarios.
    detector_cost: Dict[str, float] = {}
    for r in scored:
        for pid, blk in r.get("timing", {}).get("per_detector", {}).items():
            detector_cost[pid] = detector_cost.get(pid, 0.0) + blk["seconds"]
    summary["detector_seconds_total"] = dict(
        sorted(((k, round(v, 2)) for k, v in detector_cost.items()), key=lambda kv: -kv[1])
    )
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Anti-pattern detection validation (RQ1)")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help=f"Default: the eight-scenario suite {DETECTION_SCENARIOS}")
    p.add_argument("--layer", default="system", choices=["app", "infra", "mw", "system"])
    p.add_argument("--seed", type=int, default=42, help="Oracle seed (default: 42).")
    p.add_argument("--propagation-threshold", type=float, default=0.2,
                   help="Cascade propagation threshold for the oracle (canonical: 0.2).")
    p.add_argument("--exclude-patterns", nargs="*", default=DEFAULT_EXCLUDED_PATTERNS,
                   help="Catalog IDs to hold out. Default excludes DEEP_PIPELINE, whose "
                        "all-simple-paths enumeration does not terminate on the larger "
                        "scenarios. Pass an empty list to attempt the full 21.")
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "detection_validation.json")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR)

    from saag.analysis.antipattern_detector import CATALOG

    scenarios = args.scenarios or DETECTION_SCENARIOS
    excluded = list(args.exclude_patterns or [])
    n_active = len(CATALOG) - len(excluded)

    print("Detection validation — anti-pattern catalog vs I_comp(v) [FailureSimulator]")
    print(f"  layer={args.layer}  seed={args.seed}  "
          f"propagation_threshold={args.propagation_threshold}")
    print(f"  {n_active} of {len(CATALOG)} detectors active"
          + (f" (excluded: {', '.join(excluded)})" if excluded else ""))
    print(f"  {len(scenarios)} scenarios\n")

    header = ("| Scenario | Comps | rho(Q,I) | F1 | Prec | Rec | Top-5 | "
              "Catalog P/R/F1 | Flagged % | SPOF F1 | Gate s |")
    print(header)
    print("|" + "|".join([":---"] + [":---:"] * 10) + "|")

    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        try:
            row = evaluate(
                scenario, args.layer, args.seed, args.propagation_threshold, excluded)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed: %s", scenario, exc)
            rows.append({"scenario": scenario, "error": str(exc)})
            print(f"| {scenario} | ERROR: {exc} |")
            continue

        rows.append(row)
        q = row.get("predictors", {}).get("q_composite", {})
        cat = row.get("catalog", {})
        print(
            f"| {scenario} | {row['n_scored']} | {q.get('spearman_rho', '—')} | "
            f"{q.get('f1', '—')} | {q.get('precision', '—')} | {q.get('recall', '—')} | "
            f"{q.get('top_5_overlap', '—')} | "
            f"{cat.get('precision', '—')}/{cat.get('recall', '—')}/{cat.get('f1', '—')} | "
            f"{row['findings'].get('pct_scored_implicated', '—')} | "
            f"{row.get('spof', {}).get('f1', '—')} | {row['gate_seconds']} |"
        )
        sys.stdout.flush()

    report = {
        "oracle": "I_comp(v) — FailureSimulator composite, exhaustive removal",
        "oracle_config": {
            "layer": args.layer,
            "seed": args.seed,
            "propagation_threshold": args.propagation_threshold,
        },
        "catalog": {
            "n_total": len(CATALOG),
            "n_active": n_active,
            "excluded_patterns": excluded,
            "exclusion_reason": (
                "DEEP_PIPELINE enumerates every simple source-to-sink path "
                "(nx.all_simple_paths) and emits one finding per path: 247,761 findings "
                "on the 29-component tiny_system fixture, and no termination within ten "
                "minutes on the 50-application healthcare topology. Excluding it is the "
                "only way to measure the remaining detectors; the blowup is itself a "
                "reportable result."
            ) if "DEEP_PIPELINE" in excluded else None,
        },
        "critical_set_rule": (
            "Box-plot top tier (>= Q3) when n >= 20, else top-20% percentile; applied "
            "identically to prediction and truth, per saag/validation/validator.py. "
            "For the Q(v)/betweenness/degree predictors this pins the two sets to "
            "near-equal size, so precision and recall converge by construction. The "
            "catalog block is not thresholded this way — its flagged set is whatever "
            "the detectors produced — so its precision and recall are independent."
        ),
        "per_scenario": rows,
        "summary": _summarize(rows),
        "note": (
            "All metrics are rank-based and therefore invariant to the monotone "
            "robust_sigmoid rescaling ValidationService applies internally. Scored "
            "against I_comp only; I_comp and I* agree at mean rho 0.4046 "
            "(results/convergent_validity.json), so these figures do not transfer to "
            "claims measured against I*."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")

    s = report["summary"]
    print(f"  Q(v):        mean rho={s['q_composite']['mean_spearman_rho']}  "
          f"F1={s['q_composite']['mean_f1']}  Top-5={s['q_composite']['mean_top_5_overlap']}")
    print(f"  betweenness: mean rho={s['betweenness']['mean_spearman_rho']}")
    print(f"  degree:      mean rho={s['degree']['mean_spearman_rho']}")
    print(f"  catalog:     mean P={s['catalog']['mean_precision']} "
          f"R={s['catalog']['mean_recall']} F1={s['catalog']['mean_f1']}  "
          f"implicated={s['catalog']['mean_pct_scored_implicated']}%")
    print(f"  by scale:    large={s['by_scale']['large_ge_150_components']['mean_spearman_rho']}  "
          f"small={s['by_scale']['small_lt_150_components']['mean_spearman_rho']}")
    print(f"  gate:        {s['gate_seconds']['min']}s – {s['gate_seconds']['max']}s")
    costly = list(s["detector_seconds_total"].items())[:3]
    print("  costliest detectors: "
          + ", ".join(f"{pid} {sec}s" for pid, sec in costly))


if __name__ == "__main__":
    main()
