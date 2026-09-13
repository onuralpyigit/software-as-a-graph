#!/usr/bin/env python3
"""Time the FaultInjector ground-truth labelling sweep, per scenario.

Section 7.5.1 compares the cost of the static analysis gate against the cost of
the simulation oracle it was meant to displace. The gate side of that comparison
comes from ``results/detection_validation_timed_v3.json`` (``summary.gate_seconds``);
this script produces the oracle side, which previously had no committed artifact.

What is timed is exactly what ``cli/simulate_graph.py fault-inject`` times: graph
load plus ``FaultInjector.run`` over the CLI's own defaults --- the five
recommended seeds, node types Application/Broker/Library, unlimited cascade
depth, propagation threshold 0.2, ladder QoS factor. These are the settings that
produced ``output/loso_cache/<scenario>/failure_impact.json``, so the measurement
times the labelling run the manuscript's ground truth actually came from.

Reported per scenario as the median of ``--repeats`` runs, matching the
median-of-3 convention of Table `tab:scale`.

Usage
-----
    python reproduce/oracle_timing.py
    python reproduce/oracle_timing.py --repeats 3 --output results/oracle_timing_v4.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saag.core.graph_io import load_graph as _load_graph  # noqa: E402
from reproduce._provenance import stamp  # noqa: E402
from saag.simulation.fault_injector import FaultInjector  # noqa: E402

#: CLI defaults of ``simulate_graph.py fault-inject``; see its argument parser.
SEEDS = [42, 123, 456, 789, 2024]
NODE_TYPES = ["Application", "Broker", "Library"]
CASCADE_DEPTH = 0
PROPAGATION_THRESHOLD = 0.2
QOS_FACTOR = "ladder"

CACHE = ROOT / "output" / "loso_cache"


def time_one(topology: Path) -> tuple[float, int]:
    t0 = time.perf_counter()
    g = _load_graph(topology)
    injector = FaultInjector(
        graph=g,
        seeds=SEEDS,
        cascade_depth_limit=CASCADE_DEPTH,
        propagation_threshold=PROPAGATION_THRESHOLD,
        qos_factor_mode=QOS_FACTOR,
    )
    result = injector.run(node_types=NODE_TYPES, node_ids=None)
    return time.perf_counter() - t0, result.total_nodes_injected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", default="results/oracle_timing_v4.json")
    args = ap.parse_args()

    scenarios = sorted(d.name for d in CACHE.iterdir()
                       if d.is_dir() and (d / "topology.json").exists())
    if not scenarios:
        print(f"no cached topologies under {CACHE}", file=sys.stderr)
        return 1

    per_scenario = {}
    for sid in scenarios:
        topology = CACHE / sid / "topology.json"
        runs = []
        injected = 0
        for _ in range(args.repeats):
            elapsed, injected = time_one(topology)
            runs.append(elapsed)
        per_scenario[sid] = {
            "median_s": round(statistics.median(runs), 4),
            "min_s": round(min(runs), 4),
            "max_s": round(max(runs), 4),
            "runs_s": [round(r, 4) for r in runs],
            "nodes_injected": injected,
        }
        print(f"{sid:<28} median {per_scenario[sid]['median_s']:>8.3f} s "
              f"({injected} nodes injected)")

    medians = [v["median_s"] for v in per_scenario.values()]
    payload = {
        "measurement": "FaultInjector ground-truth labelling sweep, per scenario",
        "timed_region": "graph load + FaultInjector.run (as cli/simulate_graph.py fault-inject)",
        "settings": {
            "seeds": SEEDS,
            "node_types": NODE_TYPES,
            "cascade_depth_limit": CASCADE_DEPTH,
            "propagation_threshold": PROPAGATION_THRESHOLD,
            "qos_factor_mode": QOS_FACTOR,
            "repeats": args.repeats,
            "statistic": "median of repeats",
        },
        "platform": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
            "processor": platform.processor(),
        },
        "per_scenario": per_scenario,
        "summary": {
            "n_scenarios": len(medians),
            "oracle_seconds": {"min": round(min(medians), 4), "max": round(max(medians), 4)},
        },
    }

    gate_path = ROOT / "results" / "detection_validation_timed_v3.json"
    if gate_path.exists():
        gate = json.load(open(gate_path))["summary"]["gate_seconds"]
        payload["summary"]["gate_seconds"] = gate
        payload["summary"]["gate_over_oracle_at_max"] = round(gate["max"] / max(medians), 2)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    # Stamp the corpus this artifact describes, by content. Without it
    # reconcile_manuscript.py can only compare timestamps, which the
    # byte-identical corpus regeneration routinely invalidates in both
    # directions.
    payload["provenance"] = stamp()
    json.dump(payload, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    print(f"oracle sweep: {min(medians):.2f}-{max(medians):.2f} s per scenario")
    if "gate_over_oracle_at_max" in payload["summary"]:
        print(f"gate/oracle at the largest scenario: "
              f"{payload['summary']['gate_over_oracle_at_max']}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
