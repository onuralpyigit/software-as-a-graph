#!/usr/bin/env python3
"""
reproduce/label_stability_check.py — post-fix I*(v) label-stability re-measurement
===================================================================================

Produces ``results/label_stability.json``, the artifact behind the JSS draft's
§7.5/§8.1/§9.2 claim about how far the `FaultInjector` labels agree with
themselves across seeds ("test–retest ρ", "top-20% Jaccard").

Why this script exists rather than reusing ``output/loso_cache/*/failure_impact.json``
directly: those cached files were generated before a determinism fix to
``saag/simulation/fault_injector.py`` (the per-wave subscriber loop iterated an
unordered ``Set[str]`` while consuming seeded RNG draws, so *which* subscriber got
*which* draw depended on ``PYTHONHASHSEED`` — the run was seed-reproducible only
within one interpreter process, not across processes). This script re-runs the
label generation against the same cached topologies with the same parameters
`scripts/populate_loso_cache.sh` used, on the fixed code, and writes a *separate*
artifact rather than overwriting the cache — see docs/research/jss/draft.md §9.2
for why the tables computed from the original cache remain valid as reported.

This does not touch ``output/loso_cache/``. It reads only ``topology.json`` from
each scenario's cache directory.

Usage
-----
    PYTHONPATH=. python reproduce/label_stability_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.simulate_graph import _load_graph
from saag.simulation.fault_injector import FaultInjector, RECOMMENDED_SEEDS

#: The seven evaluation scenarios (Tables 3/18/20 of the JSS draft). Matches
#: reproduce/main_table.py's ALL_SCENARIOS.
SCENARIOS = [
    "av_system",
    "iot_smart_city_system",
    "financial_trading_system",
    "healthcare_system",
    "hub_and_spoke_system",
    "microservices_system",
    "enterprise_system",
]

#: Matches scripts/populate_loso_cache.sh's fault-inject invocation exactly:
#: Application/Broker/Library only (Topic and Node cannot express failure in
#: the cascade and would contribute spurious zeros), the five canonical seeds,
#: and FaultInjector's own defaults for cascade_depth_limit (0, unlimited),
#: propagation_threshold (0.2) and qos_factor_mode ("ladder").
NODE_TYPES = ["Application", "Broker", "Library"]

CACHE_DIR = Path("output/loso_cache")
OUTPUT_PATH = Path("results/label_stability.json")


def main() -> None:
    results: dict = {}
    for scenario in SCENARIOS:
        topology_path = CACHE_DIR / scenario / "topology.json"
        if not topology_path.exists():
            print(f"  [skip] {scenario}: no cached topology at {topology_path}")
            continue

        graph = _load_graph(topology_path)
        result = FaultInjector(
            graph,
            seeds=list(RECOMMENDED_SEEDS),
            cascade_depth_limit=0,
            propagation_threshold=0.2,
            qos_factor_mode="ladder",
        ).run(node_types=NODE_TYPES)

        stability = result.label_stability
        results[scenario] = stability
        rho = stability.get("test_retest_spearman")
        jac = stability.get("topk_jaccard")
        print(f"  {scenario:28s} test_retest_rho={rho:.4f}  topk_jaccard={jac:.4f}"
              if rho is not None else f"  {scenario:28s} (insufficient seeds)")

    rhos = [v["test_retest_spearman"] for v in results.values()
            if v.get("test_retest_spearman") is not None]
    jacs = [v["topk_jaccard"] for v in results.values()
            if v.get("topk_jaccard") is not None]
    summary = {
        "test_retest_spearman_range": [round(min(rhos), 4), round(max(rhos), 4)] if rhos else None,
        "topk_jaccard_range": [round(min(jacs), 4), round(max(jacs), 4)] if jacs else None,
        "n_scenarios": len(results),
        "seeds": list(RECOMMENDED_SEEDS),
        "node_types": NODE_TYPES,
        "note": (
            "Measured post-determinism-fix (sorted subscriber iteration in "
            "fault_injector.py); reproducible across PYTHONHASHSEED values, "
            "unlike the labels retained in output/loso_cache/."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"summary": summary, "scenarios": results}, indent=2))
    print(f"\nRange across {len(rhos)} scenarios: "
          f"test_retest_rho {summary['test_retest_spearman_range']}, "
          f"topk_jaccard {summary['topk_jaccard_range']}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
