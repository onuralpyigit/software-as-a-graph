#!/usr/bin/env python3
"""
reproduce/detection_seed_sweep.py — multi-seed repetition of detection_validation, plus the
three transcribed real-world architectures.

``reproduce/detection_validation.py`` measures the anti-pattern catalog's gate cost and its
agreement with the ``I_comp`` cascade oracle, but does so with a single run at a single seed
(``--seed 42``). Two consequences follow directly from that, and are named as threats to
validity in the Middleware 2026 Industrial Track draft (docs/research/middleware2026/industry/
draft.md §5.3): the reported timings carry no variance estimate, and the catalog-vs-oracle
scores are measured only on the eight generator-produced scenarios, so they may reflect the
generator's structural priors rather than a field distribution.

This script re-runs ``evaluate()`` — unchanged, imported directly — across:

* the same eight-scenario suite (tag ``generated``), and
* the three architectures transcribed from open-source systems, ``realworld_autoware_ros2``,
  ``realworld_trainticket``, ``realworld_cloud_microservices`` (tag ``transcribed``) — loadable
  as of this session's fix to ``saag/adapters/realworld_adapter.py``, which now emits the
  canonical relation-keyed ``relationships`` dict instead of a flat list only one example
  script knew how to read.

at five seeds — 42, 123, 456, 789, 2024, the set already standard elsewhere in this repository
(``reproduce/main_table.py``, ``reproduce/convergent_validity.py``). The oracle is the only thing
the seed touches (``_oracle(..., seed=seed)``); the analysis path (topology load -> analyze ->
RMAV -> catalog detectors) does not depend on it. So the same five passes serve double duty:
they are independent draws of the oracle *and* independent repetitions of the gate-cost timing.

Usage
-----
    PYTHONPATH=. python reproduce/detection_seed_sweep.py
    PYTHONPATH=. python reproduce/detection_seed_sweep.py --seeds 42 123
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

from reproduce.detection_validation import (
    DEFAULT_EXCLUDED_PATTERNS,
    DETECTION_SCENARIOS,
    evaluate,
)

logger = logging.getLogger("detection_seed_sweep")

RESULTS_DIR = Path("results")
SEEDS = [42, 123, 456, 789, 2024]

#: (scenario, corpus tag). Order preserved for table rendering.
SCENARIOS: List[tuple] = (
    [(s, "generated") for s in DETECTION_SCENARIOS]
    + [
        ("realworld_autoware_ros2", "transcribed"),
        ("realworld_trainticket", "transcribed"),
        ("realworld_cloud_microservices", "transcribed"),
    ]
)

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
    scenarios: List[tuple], seeds: List[int], layer: str, threshold: float, excluded: List[str]
) -> List[Dict[str, Any]]:
    """One row per (scenario, seed). A failure is recorded, not raised — a partial sweep is
    still reportable, and detection_validation.py already treats scenario failure this way."""
    rows: List[Dict[str, Any]] = []
    for scenario, corpus in scenarios:
        for seed in seeds:
            try:
                row = evaluate(scenario, layer, seed, threshold, excluded)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s (seed=%s) failed: %s", scenario, seed, exc)
                rows.append({"scenario": scenario, "corpus": corpus, "seed": seed, "error": str(exc)})
                continue
            row["corpus"] = corpus
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


def build_table2(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gate cost by scale — components + analysis/detect/gate seconds, mean ± std over seeds."""
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
            "scenario": scenario, "corpus": ok[0]["corpus"], "n_components": n_components,
            "n_seeds": len(ok), "analyze_seconds": analyze, "detect_seconds": detect,
            "gate_seconds": gate,
        })
    out.sort(key=lambda r: r["n_components"])
    return out


def build_table3(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Catalog vs. cascade oracle — mean ± std across (scenario, seed), split by corpus tag."""
    def _block(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        scored = [r for r in subset if "catalog" in r]
        if not scored:
            return {"n_scenario_seed_pairs": 0}
        return {
            "n_scenario_seed_pairs": len(scored),
            "precision": _mean_std([r["catalog"]["precision"] for r in scored]),
            "recall": _mean_std([r["catalog"]["recall"] for r in scored]),
            "f1": _mean_std([r["catalog"]["f1"] for r in scored]),
            "cohens_kappa": _mean_std([r["catalog"]["cohens_kappa"] for r in scored]),
            "pct_scored_implicated": _mean_std(
                [r["findings"]["pct_scored_implicated"] for r in scored
                 if "pct_scored_implicated" in r["findings"]]),
        }

    generated = [r for r in rows if r.get("corpus") == "generated"]
    transcribed = [r for r in rows if r.get("corpus") == "transcribed"]
    return {
        "generated": _block(generated),
        "transcribed": _block(transcribed),
        "pooled": _block(rows),
    }


def build_table4(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transcribed architectures — scale plus measured findings/flagged/gate outcome."""
    out = []
    for scenario, group in _by_scenario(rows).items():
        ok = [r for r in group if "error" not in r]
        if not ok or ok[0]["corpus"] != "transcribed":
            continue
        n_findings = _mean_std([r["findings"]["n_findings"] for r in ok])
        pct_flagged = _mean_std([r["findings"]["pct_components_flagged"] for r in ok])
        exit_codes = sorted({r["gate_exit_code"] for r in ok if "gate_exit_code" in r})
        out.append({
            "scenario": scenario, "n_components": ok[0]["n_components"], "n_seeds": len(ok),
            "n_findings": n_findings, "pct_components_flagged": pct_flagged,
            "gate_exit_codes_observed": exit_codes,
        })
    return out


def build_gate_decisions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per scenario, is the exit code stable across seeds, and what is it?"""
    out = {}
    for scenario, group in _by_scenario(rows).items():
        ok = [r for r in group if "gate_exit_code" in r]
        if not ok:
            continue
        codes = sorted({r["gate_exit_code"] for r in ok})
        out[scenario] = {
            "corpus": ok[0]["corpus"], "exit_codes_observed": codes,
            "stable_across_seeds": len(codes) == 1,
        }
    n_would_block = sum(1 for v in out.values() if v["exit_codes_observed"] == [2])
    return {
        "per_scenario": out,
        "n_scenarios_always_exit_2": n_would_block,
        "n_scenarios_total": len(out),
    }


def render_markdown(table2, table3, table4, gate_decisions, seeds: List[int]) -> str:
    lines = [
        "# Detection validation — seed sweep tables",
        "",
        f"Seeds: {seeds}. Source: `results/detection_seed_sweep.json`. "
        "Generated by `reproduce/detection_seed_sweep.py`.",
        "",
        "## Table 2 — gate cost by scale",
        "",
        "| Scenario | Corpus | Components | Analysis (s) | Detection (s) | Gate total (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in table2:
        lines.append(
            f"| {r['scenario']} | {r['corpus']} | {r['n_components']} | "
            f"{r['analyze_seconds']['mean']:.2f} ± {r['analyze_seconds']['std']:.2f} | "
            f"{r['detect_seconds']['mean']:.2f} ± {r['detect_seconds']['std']:.2f} | "
            f"{r['gate_seconds']['mean']:.2f} ± {r['gate_seconds']['std']:.2f} |"
        )
    lines += ["", "## Table 3 — catalog vs. cascade oracle", ""]
    for label, blk in table3.items():
        if blk.get("n_scenario_seed_pairs", 0) == 0:
            lines.append(f"**{label}**: no scored rows.")
            continue
        lines.append(
            f"**{label}** (n={blk['n_scenario_seed_pairs']} scenario x seed pairs): "
            f"precision {blk['precision']['mean']:.3f} ± {blk['precision']['std']:.3f}, "
            f"recall {blk['recall']['mean']:.3f} ± {blk['recall']['std']:.3f}, "
            f"F1 {blk['f1']['mean']:.3f} ± {blk['f1']['std']:.3f}, "
            f"kappa {blk['cohens_kappa']['mean']:.3f} ± {blk['cohens_kappa']['std']:.3f}"
        )
    lines += ["", "## Table 4 — transcribed architectures", "",
               "| Scenario | Components | Findings (mean ± std) | % Components Flagged | Gate exit codes |",
               "|---|---:|---:|---:|---|"]
    for r in table4:
        lines.append(
            f"| {r['scenario']} | {r['n_components']} | "
            f"{r['n_findings']['mean']:.1f} ± {r['n_findings']['std']:.1f} | "
            f"{r['pct_components_flagged']['mean']:.1f}% ± {r['pct_components_flagged']['std']:.1f} | "
            f"{r['gate_exit_codes_observed']} |"
        )
    lines += [
        "", "## Gate decision distribution",
        "",
        f"{gate_decisions['n_scenarios_always_exit_2']} of "
        f"{gate_decisions['n_scenarios_total']} scenarios return exit code 2 "
        "(CRITICAL/HIGH present) at every seed.",
        "",
        "| Scenario | Corpus | Exit codes observed | Stable across seeds |",
        "|---|---|---|---|",
    ]
    for scenario, v in gate_decisions["per_scenario"].items():
        lines.append(
            f"| {scenario} | {v['corpus']} | {v['exit_codes_observed']} | "
            f"{v['stable_across_seeds']} |"
        )
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed detection validation sweep")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--layer", default="system", choices=["app", "infra", "mw", "system"])
    p.add_argument("--propagation-threshold", type=float, default=0.2)
    p.add_argument("--exclude-patterns", nargs="*", default=DEFAULT_EXCLUDED_PATTERNS)
    p.add_argument("--output-json", type=Path, default=RESULTS_DIR / "detection_seed_sweep.json")
    p.add_argument("--output-md", type=Path, default=RESULTS_DIR / "detection_tables.md")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    excluded = list(args.exclude_patterns or [])
    print(f"Sweeping {len(SCENARIOS)} scenarios x {len(args.seeds)} seeds "
          f"({len(SCENARIOS) * len(args.seeds)} runs), layer={args.layer}")

    rows = run_sweep(SCENARIOS, args.seeds, args.layer, args.propagation_threshold, excluded)
    n_errors = sum(1 for r in rows if "error" in r)
    if n_errors:
        print(f"WARNING: {n_errors} of {len(rows)} runs errored — see rows with an 'error' key.")

    table2 = build_table2(rows)
    table3 = build_table3(rows)
    table4 = build_table4(rows)
    gate_decisions = build_gate_decisions(rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "config": {
                "seeds": args.seeds, "layer": args.layer,
                "propagation_threshold": args.propagation_threshold,
                "excluded_patterns": excluded, "scenarios": SCENARIOS,
            },
            "rows": rows,
            "table2_gate_cost": table2,
            "table3_catalog_vs_oracle": table3,
            "table4_transcribed": table4,
            "gate_decisions": gate_decisions,
        }, f, indent=2)
    print(f"Wrote {args.output_json}")

    md = render_markdown(table2, table3, table4, gate_decisions, args.seeds)
    with open(args.output_md, "w") as f:
        f.write(md)
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
