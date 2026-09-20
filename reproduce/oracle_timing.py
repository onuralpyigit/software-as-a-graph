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


def emit_ratio_table(
    oracle_path: Path, gate_path: Path, cache: Path, out: Path
) -> int:
    """Pair an existing oracle sweep with a gate run, scenario by scenario.

    The manuscript reported this comparison as a single number taken at the
    joint maximum -- the gate costs "roughly eighteen times" the oracle -- which
    is the ratio on one scenario of twelve and the largest of the set. A single
    scenario cannot say whether the premium is typical or exceptional, and here
    it is exceptional: the median is a third of it.

    Nothing is re-measured. Both halves must come from the same measurement
    session or the ratio pairs unrelated machines; the caller is responsible for
    that pairing and the source of each half is recorded in the artifact.

    Each row also carries the size of the derived ``DEPENDS_ON`` projection the
    analysis stage actually traverses, read from the committed cache. That is
    the column that explains the spread: the gate's cost premium tracks the
    derived edge count, not the component count.
    """
    oracle = json.loads(oracle_path.read_text())["per_scenario"]
    gate_doc = json.loads(gate_path.read_text())
    gate = {
        s["scenario"]: s["timing"]["analyze_seconds"] + s["timing"]["detect_seconds"]
        for s in gate_doc["per_scenario"]
    }

    rows = []
    for name in sorted(set(oracle) & set(gate)):
        summary = {}
        metrics = cache / name / "structural_metrics.json"
        if metrics.exists():
            summary = json.loads(metrics.read_text()).get("graph_summary", {})
        oracle_s = oracle[name]["median_s"]
        rows.append({
            "scenario": name,
            "oracle_s": round(oracle_s, 3),
            "gate_s": round(gate[name], 2),
            "ratio": round(gate[name] / oracle_s, 1),
            "projection_nodes": summary.get("nodes"),
            "projection_edges": summary.get("edges"),
            "projection_density": (
                round(summary["density"], 3) if "density" in summary else None
            ),
        })
    if not rows:
        print("Error: no scenario is present in both artifacts.", file=sys.stderr)
        return 2

    ratios = sorted(r["ratio"] for r in rows)
    payload = {
        "measurement": "static analysis gate vs FaultInjector oracle, per scenario",
        "note": (
            "Derived from two existing artifacts; nothing is re-timed. The "
            "projection columns describe the app-layer DEPENDS_ON projection "
            "held in the committed cache, while the gate itself analyses the "
            "system layer -- the column is reported as the density diagnostic "
            "that orders the scenarios, not as the gate's own edge count."
        ),
        "oracle_source": str(oracle_path.relative_to(ROOT)),
        "gate_source": str(gate_path.relative_to(ROOT)),
        "n_scenarios": len(rows),
        "ratio": {
            "min": ratios[0],
            "max": ratios[-1],
            "median": round(statistics.median(ratios), 1),
            "mean": round(statistics.fmean(ratios), 1),
        },
        "per_scenario": sorted(rows, key=lambda r: -r["ratio"]),
    }
    payload["provenance"] = stamp(
        oracle_source=payload["oracle_source"], gate_source=payload["gate_source"]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"  {'scenario':<26}{'|E_proj|':>10}{'gate s':>9}"
          f"{'oracle s':>10}{'ratio':>8}")
    print("  " + "─" * 63)
    for r in payload["per_scenario"]:
        edges = r["projection_edges"] if r["projection_edges"] is not None else 0
        print(f"  {r['scenario']:<26}{edges:>10}{r['gate_s']:>9.2f}"
              f"{r['oracle_s']:>10.3f}{r['ratio']:>7.1f}x")
    rt = payload["ratio"]
    print(f"\n  n = {len(rows)}   min {rt['min']}x   median {rt['median']}x   "
          f"mean {rt['mean']}x   max {rt['max']}x")
    print(f"  wrote {out}")
    return 0


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
    ap.add_argument(
        "--emit-table", action="store_true",
        help="Post-hoc mode: re-time nothing, and instead pair an existing "
             "oracle sweep with a gate run into a per-scenario ratio table. "
             "Section 7.5.1 reports the ratio at the joint maximum; this is "
             "the distribution behind that single number.",
    )
    ap.add_argument(
        "--oracle-file", default="results/oracle_timing_jss12.json",
        help="Oracle sweep to read in --emit-table mode. Must come from the "
             "same measurement session as --gate-file.",
    )
    ap.add_argument(
        "--table-output", default="results/gate_oracle_ratio.json",
        help="Where --emit-table writes the per-scenario pairing.",
    )
    ap.add_argument(
        "--gate-file", default="results/detection_validation_timed_v3.json",
        help="Detection run supplying the gate side of the ratio. The two halves "
             "must come from one measurement session on one machine: refreshing "
             "only one produced a ratio spanning two sessions and moved it from "
             "11.45x to 15.35x with no change to the work being timed.")
    args = ap.parse_args()

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    if args.emit_table:
        return emit_ratio_table(
            _resolve(args.oracle_file),
            _resolve(args.gate_file),
            CACHE,
            _resolve(args.table_output),
        )

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

    # The gate half of the ratio. Its path was hardcoded to a ``_v3`` spelling,
    # so re-running the detection benchmark under any later version left this
    # silently reading the superseded file — and when the file was absent the
    # ratio was dropped without comment, taking a headline claim's only
    # mechanical check with it (reconcile_manuscript skips a null ratio).
    gate_path = Path(args.gate_file)
    if not gate_path.is_absolute():
        gate_path = ROOT / gate_path
    if gate_path.exists():
        gate = json.load(open(gate_path))["summary"]["gate_seconds"]
        payload["summary"]["gate_seconds"] = gate
        payload["summary"]["gate_over_oracle_at_max"] = round(gate["max"] / max(medians), 2)
        # Name the paired artifact in the artifact, so a reader can tell whether
        # the two halves came from one session without reconstructing it.
        payload["summary"]["gate_source"] = str(gate_path.relative_to(ROOT))
    else:
        print(f"WARNING: gate file {gate_path} absent — the gate/oracle ratio, "
              f"which Section 7.5.1 reports, is NOT in this artifact.", file=sys.stderr)

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
