#!/usr/bin/env python3
"""
reproduce/atm_scale_sweep.py — detection_validation, swept across ATM system scale.

``reproduce/detection_seed_sweep.py`` swept the anti-pattern catalog's gate cost and its
agreement with the ``I_comp`` cascade oracle across eight *domains* (plus three transcribed
architectures) at a fixed scale each. That answers "does this generalize across domains?" but
confounds domain and scale together — it can't say whether verification quality changes as
*one* system grows, because no two scenarios in that corpus share a domain.

This script holds domain fixed (air_traffic_management, matching the industrial deployment that
motivates this paper — see draft.md §2.2) and varies only scale: five datasets
(``atm_system_tiny``, ``atm_system``, ``atm_system_medium``, ``atm_system_large``,
``atm_system_xlarge`` — 29/74/148/296/444 components) generated from configs that share
identical QoS-mix, fan-out-shape, and criticality-proportion statistics blocks and differ only in
``graph.counts`` (see data/scenarios/scenario_1{0,4,5,6,7}_atm_*.yaml). Scale is therefore the
sole independent variable in this sweep, unlike the cross-domain corpus.

Reuses ``evaluate()`` from ``reproduce/detection_validation.py`` unchanged — same measurement
code as detection_seed_sweep.py, different scenario axis and different aggregation. As before,
five oracle seeds — 42, 123, 456, 789, 2024 — serve double duty: independent draws of the
I_comp(v) oracle, and independent repetitions of the gate-cost timing (the analysis path does
not depend on the seed at all).

**Effective-n note carried forward from the cross-domain sweep**: last session's sweep found
catalog/oracle output is deterministic given the graph, and the oracle's critical-set membership
was seed-invariant for 10 of 11 scenarios. If that holds here too, the honest sample size for the
catalog-vs-oracle trend is 5 (scale points, each seed-confirmed), not 25 (scale x seed pairs) —
this script reports both the per-seed values (to make seed-invariance checkable) and the
per-scale summary (mean +/- std over seeds, which collapses to the single value when
seed-invariant).

Usage
-----
    PYTHONPATH=. python reproduce/atm_scale_sweep.py
    PYTHONPATH=. python reproduce/atm_scale_sweep.py --seeds 42 123
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from reproduce.detection_validation import DEFAULT_EXCLUDED_PATTERNS, evaluate

logger = logging.getLogger("atm_scale_sweep")

RESULTS_DIR = Path("results")
SEEDS = [42, 123, 456, 789, 2024]

#: The five ATM scale variants, ordered small to large. See module docstring.
ATM_SCALE_SCENARIOS: List[str] = [
    "atm_system_tiny",
    "atm_system",
    "atm_system_medium",
    "atm_system_large",
    "atm_system_xlarge",
]

#: Severities that make cli/detect_antipatterns.py exit non-zero, and at what code.
_GATE_EXIT_CRITICAL_HIGH = 2
_GATE_EXIT_ANY_FINDING = 1
_GATE_EXIT_CLEAN = 0


def _gate_exit_code(by_severity: Dict[str, int]) -> int:
    """Mirror cli/detect_antipatterns.py's exit-code rule exactly (lines 90-98)."""
    if by_severity.get("CRITICAL", 0) > 0 or by_severity.get("HIGH", 0) > 0:
        return _GATE_EXIT_CRITICAL_HIGH
    if sum(by_severity.values()) > 0:
        return _GATE_EXIT_ANY_FINDING
    return _GATE_EXIT_CLEAN


def run_sweep(
    scenarios: List[str], seeds: List[int], layer: str, threshold: float, excluded: List[str]
) -> List[Dict[str, Any]]:
    """One row per (scenario, seed). A failure is recorded, not raised."""
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for seed in seeds:
            try:
                row = evaluate(scenario, layer, seed, threshold, excluded)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s (seed=%s) failed: %s", scenario, seed, exc)
                rows.append({"scenario": scenario, "seed": seed, "error": str(exc)})
                continue
            row["seed"] = seed
            if "findings" in row:
                row["gate_exit_code"] = _gate_exit_code(row["findings"].get("by_severity", {}))
            rows.append(row)
            logger.info("%s seed=%s: gate=%.2fs", scenario, seed, row.get("gate_seconds", float("nan")))
    return rows


# =============================================================================
# Aggregation into paper tables
# =============================================================================

def _mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    return {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std(ddof=0)), 4)}


def _by_scenario(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["scenario"], []).append(r)
    return grouped


def build_cost_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gate cost vs. ATM scale — components + analysis/detect/gate seconds, mean +/- std over seeds."""
    out = []
    for scenario, group in _by_scenario(rows).items():
        ok = [r for r in group if "error" not in r]
        if not ok:
            continue
        n_components = ok[0]["n_components"]
        analyze = _mean_std([r["timing"]["analyze_seconds"] for r in ok])
        detect = _mean_std([r["timing"]["detect_seconds"] for r in ok])
        gate = _mean_std([r["gate_seconds"] for r in ok])
        out.append({
            "scenario": scenario, "n_components": n_components, "n_seeds": len(ok),
            "analyze_seconds": analyze, "detect_seconds": detect, "gate_seconds": gate,
        })
    out.sort(key=lambda r: r["n_components"])
    return out


def build_catalog_trend_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Catalog vs. cascade oracle, one row per scale (not pooled across scales — the scale
    trend is the result; averaging it away would hide the thing this sweep measures)."""
    out = []
    for scenario, group in _by_scenario(rows).items():
        scored = [r for r in group if "catalog" in r]
        if not scored:
            continue
        kappas = [r["catalog"]["cohens_kappa"] for r in scored]
        seed_invariant = len({round(k, 6) for k in kappas}) == 1
        out.append({
            "scenario": scenario,
            "n_components": scored[0]["n_components"],
            "n_seeds": len(scored),
            "seed_invariant": seed_invariant,
            "precision": _mean_std([r["catalog"]["precision"] for r in scored]),
            "recall": _mean_std([r["catalog"]["recall"] for r in scored]),
            "f1": _mean_std([r["catalog"]["f1"] for r in scored]),
            "cohens_kappa": _mean_std(kappas),
            "n_flagged_over_n_scored_pct": _mean_std(
                [100.0 * r["catalog"]["n_flagged"] / r["n_scored"] for r in scored if r.get("n_scored")]
            ),
        })
    out.sort(key=lambda r: r["n_components"])
    return out


def build_gate_decisions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per scenario, is the exit code stable across seeds, and what is it?"""
    out = {}
    for scenario, group in _by_scenario(rows).items():
        ok = [r for r in group if "gate_exit_code" in r]
        if not ok:
            continue
        codes = sorted({r["gate_exit_code"] for r in ok})
        out[scenario] = {"exit_codes_observed": codes, "stable_across_seeds": len(codes) == 1}
    n_would_block = sum(1 for v in out.values() if v["exit_codes_observed"] == [2])
    return {"per_scenario": out, "n_scenarios_always_exit_2": n_would_block, "n_scenarios_total": len(out)}


def render_markdown(cost_table, catalog_table, gate_decisions, seeds: List[int]) -> str:
    lines = [
        "# ATM scale sweep — gate cost and detection quality vs. system scale",
        "",
        f"Seeds: {seeds}. Domain held fixed (air_traffic_management); scale is the sole "
        "independent variable. Source: `results/atm_scale_sweep.json`. "
        "Generated by `reproduce/atm_scale_sweep.py`.",
        "",
        "## Gate cost by ATM scale",
        "",
        "| Scale | Components | Analysis (s) | Detection (s) | Gate total (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in cost_table:
        lines.append(
            f"| {r['scenario']} | {r['n_components']} | "
            f"{r['analyze_seconds']['mean']:.2f} ± {r['analyze_seconds']['std']:.2f} | "
            f"{r['detect_seconds']['mean']:.2f} ± {r['detect_seconds']['std']:.2f} | "
            f"{r['gate_seconds']['mean']:.2f} ± {r['gate_seconds']['std']:.2f} |"
        )
    lines += [
        "", "## Detector catalog vs. cascade oracle, by ATM scale", "",
        "One row per scale (not pooled — the trend across scale is the result). "
        "`seed_invariant=True` means all 5 seeds gave bit-identical catalog metrics at that "
        "scale, i.e. the honest n for that row is 1 confirmed measurement, not 5 independent ones.",
        "",
        "| Scale | Components | Precision | Recall | F1 | Cohen's kappa | % scored flagged | Seed-invariant |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in catalog_table:
        lines.append(
            f"| {r['scenario']} | {r['n_components']} | "
            f"{r['precision']['mean']:.3f} ± {r['precision']['std']:.3f} | "
            f"{r['recall']['mean']:.3f} ± {r['recall']['std']:.3f} | "
            f"{r['f1']['mean']:.3f} ± {r['f1']['std']:.3f} | "
            f"{r['cohens_kappa']['mean']:.3f} ± {r['cohens_kappa']['std']:.3f} | "
            f"{r['n_flagged_over_n_scored_pct']['mean']:.1f}% | "
            f"{r['seed_invariant']} |"
        )
    lines += [
        "", "## Gate decision distribution", "",
        f"{gate_decisions['n_scenarios_always_exit_2']} of "
        f"{gate_decisions['n_scenarios_total']} ATM scales return exit code 2 "
        "(CRITICAL/HIGH present) at every seed.",
        "",
        "| Scale | Exit codes observed | Stable across seeds |",
        "|---|---|---|",
    ]
    for scenario, v in gate_decisions["per_scenario"].items():
        lines.append(f"| {scenario} | {v['exit_codes_observed']} | {v['stable_across_seeds']} |")
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(description="ATM scale sweep (detection cost + catalog quality vs. scale)")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--layer", default="system", choices=["app", "infra", "mw", "system"])
    p.add_argument("--propagation-threshold", type=float, default=0.2)
    p.add_argument("--exclude-patterns", nargs="*", default=DEFAULT_EXCLUDED_PATTERNS)
    p.add_argument("--output-json", type=Path, default=RESULTS_DIR / "atm_scale_sweep.json")
    p.add_argument("--output-md", type=Path, default=RESULTS_DIR / "atm_scale_tables.md")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    excluded = list(args.exclude_patterns or [])
    print(f"Sweeping {len(ATM_SCALE_SCENARIOS)} ATM scales x {len(args.seeds)} seeds "
          f"({len(ATM_SCALE_SCENARIOS) * len(args.seeds)} runs), layer={args.layer}")

    rows = run_sweep(ATM_SCALE_SCENARIOS, args.seeds, args.layer, args.propagation_threshold, excluded)
    n_errors = sum(1 for r in rows if "error" in r)
    if n_errors:
        print(f"WARNING: {n_errors} of {len(rows)} runs errored — see rows with an 'error' key.")

    cost_table = build_cost_table(rows)
    catalog_table = build_catalog_trend_table(rows)
    gate_decisions = build_gate_decisions(rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "config": {
                "seeds": args.seeds, "layer": args.layer,
                "propagation_threshold": args.propagation_threshold,
                "excluded_patterns": excluded, "scenarios": ATM_SCALE_SCENARIOS,
            },
            "rows": rows,
            "cost_by_scale": cost_table,
            "catalog_trend_by_scale": catalog_table,
            "gate_decisions": gate_decisions,
        }, f, indent=2)
    print(f"Wrote {args.output_json}")

    md = render_markdown(cost_table, catalog_table, gate_decisions, args.seeds)
    with open(args.output_md, "w") as f:
        f.write(md)
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
