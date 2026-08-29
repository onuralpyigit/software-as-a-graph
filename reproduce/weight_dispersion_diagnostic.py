#!/usr/bin/env python3
"""
reproduce/weight_dispersion_diagnostic.py — realized dynamic range of the weight model
=======================================================================================

Produces ``results/weight_dispersion.json`` and a companion markdown table.

What this answers
-----------------
Every sensitivity sweep in ``reproduce/`` reports the same shape of result: moving
a declared constant barely moves downstream rho. ``topic_weight_sensitivity.py``
reads that as robustness. It is not — or not only. A coefficient cannot reorder
anything if the term it multiplies does not vary, and three of the terms in this
model barely do.

This script measures that directly. For every committed scenario it reports, per
quantity and per entity type:

* ``n``, ``min``, ``median``, ``max``
* ``sd`` and ``cv`` — the dispersion the downstream scorers actually see
* ``n_distinct`` — rank ties cap what any Spearman-based evaluation can resolve
* ``floor_frac`` / ``ceil_frac`` — clipping at ``MIN_TOPIC_WEIGHT`` and at 1.0,
  where rank information is destroyed outright

and, for ``w(t)`` only, the per-term weighted standard deviations, which is where
the declared (beta, alpha, psi) budget is either honoured or is not.

Read the ``interpretation`` block alongside the tables: a term whose weighted sd
is an order of magnitude below another's cannot reorder topics whatever
coefficient it carries, and a quantity whose CV is ~0.005 across 300 components
is a constant offset wearing the costume of a feature.

Usage
-----
    PYTHONPATH=. python reproduce/weight_dispersion_diagnostic.py
    PYTHONPATH=. python reproduce/weight_dispersion_diagnostic.py --scenarios av_system
    PYTHONPATH=. python reproduce/weight_dispersion_diagnostic.py --label pre_fix
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

logger = logging.getLogger("weight_dispersion")

RESULTS_DIR = Path("results")
SCENARIOS_DIR = Path("data/scenarios")

#: Entity collections carrying an aggregate vertex weight, in propagation order.
VERTEX_TYPES = ("applications", "libraries", "brokers", "nodes")


# ── Summary statistics ────────────────────────────────────────────────────────

def _summarise(values: Sequence[float], floor: float) -> Optional[Dict[str, Any]]:
    """Dispersion of one weight population, or None when there is nothing to say."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean())
    return {
        "n": int(arr.size),
        "min": round(float(arr.min()), 6),
        "median": round(float(np.median(arr)), 6),
        "max": round(float(arr.max()), 6),
        "sd": round(float(arr.std()), 6),
        "cv": round(float(arr.std() / mean), 6) if mean else None,
        "n_distinct": int(len(set(np.round(arr, 9)))),
        "floor_frac": round(float(np.mean(arr <= floor + 1e-12)), 4),
        "ceil_frac": round(float(np.mean(arr >= 1.0 - 1e-12)), 4),
    }


def _term_contributions(topics: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Weighted spread of each w(t) term, at the shipped coefficients.

    Mirrors ``compute_topic_weight`` rather than reimplementing it: the three
    normalised terms are read back through the same helpers the formula uses, so
    this cannot drift from the shipped weight.
    """
    from saag.core import models
    from saag.core.models import QoSPolicy

    qos, size, freq = [], [], []
    for t in topics:
        policy = QoSPolicy.from_node_attrs(t)
        raw_size = t.get("size", t.get("message_size", 1024)) or 1024
        # topics here come from repo.data (post save_graph flattening), where
        # flatten_component renames "frequency" to "topic_frequency" -- the
        # same rename that made MemoryRepository._calculate_intrinsic_weights
        # silently ignore it (see saag/infrastructure/memory_repo.py).
        raw_freq = t.get("frequency", t.get("topic_frequency"))
        qos.append(policy.calculate_weight())
        size.append(models.compute_size_norm(raw_size))
        freq.append(models.compute_freq_norm(raw_freq))

    coeffs = (
        ("qos", models.TOPIC_QOS_WEIGHT_BETA, qos),
        ("size_norm", models.TOPIC_SIZE_WEIGHT_ALPHA, size),
        ("freq_norm", models.TOPIC_FREQ_WEIGHT_PSI, freq),
    )
    out: Dict[str, Dict[str, float]] = {}
    for name, coeff, series in coeffs:
        arr = np.array(series, dtype=float)
        out[name] = {
            "coefficient": round(float(coeff), 4),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "sd_raw": round(float(arr.std()), 6),
            "sd_weighted": round(float((coeff * arr).std()), 6),
        }
    total = sum(v["sd_weighted"] for v in out.values()) or 1.0
    for v in out.values():
        # The share of realized spread the term supplies, against the share its
        # coefficient nominally claims. A large gap is the defect being measured.
        v["realized_share"] = round(v["sd_weighted"] / total, 4)
    return out


# ── Per-scenario measurement ──────────────────────────────────────────────────

def measure_scenario(topology: Dict[str, Any]) -> Dict[str, Any]:
    """Every weight population the construction phases produce for one topology."""
    from saag.core.models import MIN_TOPIC_WEIGHT
    from saag.infrastructure.memory_repo import MemoryRepository

    repo = MemoryRepository()
    repo.save_graph(topology)
    repo.derive_dependencies()

    topics = repo.data.get("topics", [])
    report: Dict[str, Any] = {
        "topic_weight": _summarise(
            [t.get("weight") for t in topics], MIN_TOPIC_WEIGHT
        ),
        "topic_weight_terms": _term_contributions(topics) if topics else None,
        "vertex_weight": {},
        "edge_weight": {},
    }

    for key in VERTEX_TYPES:
        summary = _summarise(
            [c.get("weight") for c in repo.data.get(key, [])], MIN_TOPIC_WEIGHT
        )
        if summary:
            report["vertex_weight"][key] = summary

    by_type: Dict[str, List[float]] = {}
    for edge in repo.data["relationships"].get("depends_on", []):
        by_type.setdefault(edge["dependency_type"], []).append(float(edge["weight"]))
    for dep_type, weights in sorted(by_type.items()):
        report["edge_weight"][dep_type] = _summarise(weights, MIN_TOPIC_WEIGHT)

    return report


# ── Rendering ─────────────────────────────────────────────────────────────────

_COLUMNS = ("n", "min", "median", "max", "sd", "cv", "n_distinct", "floor_frac", "ceil_frac")


def _render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Realized dynamic range of the weight model",
        "",
        "Generated by `reproduce/weight_dispersion_diagnostic.py`.",
        "",
        "`cv` is the coefficient of variation the downstream scorers see; `n_distinct`",
        "caps what any rank correlation can resolve; `ceil_frac` is outright rank",
        "information loss at the top of the distribution.",
        "",
    ]

    header = "| scenario | quantity | " + " | ".join(_COLUMNS) + " |"
    divider = "|:---|:---|" + "---:|" * len(_COLUMNS)

    def rows(section: str, label: str) -> List[str]:
        out = []
        for scenario, data in sorted(report["scenarios"].items()):
            block = data.get(section) or {}
            items = block.items() if isinstance(block, dict) and section != "topic_weight" else []
            if section == "topic_weight":
                items = [("w(t)", block)] if block else []
            for name, stats in items:
                if not stats:
                    continue
                cells = [
                    "" if stats.get(c) is None else
                    (str(stats[c]) if c in ("n", "n_distinct") else f"{stats[c]:.4f}")
                    for c in _COLUMNS
                ]
                out.append(f"| {scenario} | {label}`{name}` | " + " | ".join(cells) + " |")
        return out

    for section, label, title in (
        ("topic_weight", "", "## Intrinsic topic weight"),
        ("vertex_weight", "w(v) ", "## Aggregate vertex weight (Phase 5a)"),
        ("edge_weight", "w_E ", "## Derived DEPENDS_ON edge weight (Phases 4, 5b)"),
    ):
        body = rows(section, label)
        if body:
            lines += [title, "", header, divider, *body, ""]

    lines += [
        "## w(t) term contributions at the shipped coefficients",
        "",
        "| scenario | term | coefficient | min | max | sd_raw | sd_weighted | realized_share |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, data in sorted(report["scenarios"].items()):
        for term, stats in (data.get("topic_weight_terms") or {}).items():
            lines.append(
                f"| {scenario} | {term} | {stats['coefficient']:.2f} | {stats['min']:.4f} | "
                f"{stats['max']:.4f} | {stats['sd_raw']:.4f} | {stats['sd_weighted']:.5f} | "
                f"{stats['realized_share']:.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def _discover_scenarios() -> List[str]:
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*_system.json"))


def run(scenarios: List[str]) -> Dict[str, Any]:
    from saag.core import models

    per_scenario: Dict[str, Any] = {}
    for name in scenarios:
        path = SCENARIOS_DIR / f"{name}.json"
        try:
            topology = json.loads(path.read_text())
            per_scenario[name] = measure_scenario(topology)
        except Exception as exc:      # noqa: BLE001 - one bad scenario must not kill the run
            logger.warning("%s failed: %s", name, exc)
            continue
        tw = per_scenario[name]["topic_weight"]
        print(f"  {name:<38s} w(t) cv={tw['cv']:.4f} n_distinct={tw['n_distinct']:<4d} "
              f"floor={tw['floor_frac']:.2f} ceil={tw['ceil_frac']:.2f}")

    return {
        "constants": {
            "beta": models.TOPIC_QOS_WEIGHT_BETA,
            "alpha": models.TOPIC_SIZE_WEIGHT_ALPHA,
            "psi": models.TOPIC_FREQ_WEIGHT_PSI,
            "w_reliability": models.QoSPolicy.W_RELIABILITY,
            "w_durability": models.QoSPolicy.W_DURABILITY,
            "w_priority": models.QoSPolicy.W_PRIORITY,
            "power_mean_p": models.COMPONENT_POWER_MEAN_P,
            "lib_fanout_gamma": models.LIB_FANOUT_GAMMA,
            "min_topic_weight": models.MIN_TOPIC_WEIGHT,
        },
        "scenarios": per_scenario,
        "interpretation": {
            "note": (
                "A coefficient cannot reorder anything if the term it multiplies "
                "does not vary. Compare each term's `realized_share` against its "
                "`coefficient`: a large gap means the declared convex combination "
                "is not the combination the pipeline actually applies. Likewise a "
                "vertex-weight `cv` near zero means w(v) enters A(v) as a constant "
                "offset rather than a discriminating feature, and a `ceil_frac` "
                "above zero is rank information destroyed by clipping. Flat "
                "sensitivity curves elsewhere in `reproduce/` should be read "
                "against this table before being called robustness."
            ),
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="Realized dynamic range of the weight model")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help="Scenario stems (default: every data/scenarios/*_system.json)")
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "weight_dispersion.json")
    p.add_argument("--label", default=None,
                   help="Suffix the output stem, e.g. --label pre_fix")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    scenarios = args.scenarios or _discover_scenarios()
    print(f"Weight dispersion over {len(scenarios)} scenarios")
    report = run(scenarios)

    output = args.output
    if args.label:
        output = output.with_name(f"{output.stem}.{args.label}{output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    output.with_suffix(".md").write_text(_render_markdown(report))
    print(f"\nWrote {output}")
    print(f"Wrote {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
