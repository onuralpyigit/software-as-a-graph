#!/usr/bin/env python3
"""Bound the energy of one pre-deployment gate run, from measured CPU time.

Section 8.2 argues a sustainability case it never measures. This script does not
fix that -- nothing short of RAPL or a wall meter does -- but it replaces an
unquantified claim with a quantified upper bound, which is a different and more
honest kind of statement.

The arithmetic is deliberately crude: measured wall-clock seconds times a
declared package power. Three things make the result an upper bound rather than
an estimate, and all three are reported alongside it:

  * The gate and the oracle are single-threaded Python. The package power below
    is for the whole package, so attributing it entirely to one busy core
    overstates the draw by roughly the core count under load.
  * Wall-clock is not CPU-busy time; interpreter startup and I/O are included.
  * Package power excludes everything outside the SoC and includes no idle
    baseline, so it is neither a floor nor a ceiling on system draw.

What survives those caveats is the order of magnitude, and the order of
magnitude is the point: the quantity a pre-deployment gate saves by not
provisioning a staging cluster is not in the same units as the quantity it
spends computing, and this makes the spent side concrete enough to compare.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(ROOT))

from reproduce._provenance import stamp  # noqa: E402

#: Intel processor base power for the measurement machine's SoC, from the
#: vendor's published specification. Named, not inferred, so a reader on other
#: hardware can substitute their own figure and redo the arithmetic.
DEFAULT_CPU = "13th Gen Intel(R) Core(TM) i7-1370P"
DEFAULT_PACKAGE_W = 28.0
DEFAULT_POWER_SOURCE = "Intel processor base power (PBP) for the i7-1370P"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-file", default="results/detection_validation_timed_jss12.json")
    ap.add_argument("--oracle-file", default="results/oracle_timing_jss12.json")
    ap.add_argument("--package-watts", type=float, default=DEFAULT_PACKAGE_W)
    ap.add_argument("--cpu", default=DEFAULT_CPU)
    ap.add_argument("--output", default="results/energy_estimate.json")
    args = ap.parse_args()

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    gate_doc = json.loads(_resolve(args.gate_file).read_text())
    oracle_doc = json.loads(_resolve(args.oracle_file).read_text())
    gate = {
        s["scenario"]: s["timing"]["analyze_seconds"] + s["timing"]["detect_seconds"]
        for s in gate_doc["per_scenario"]
    }
    oracle = {k: v["median_s"] for k, v in oracle_doc["per_scenario"].items()}
    shared = sorted(set(gate) & set(oracle))
    if not shared:
        print("Error: the two artifacts share no scenario.", file=sys.stderr)
        return 2

    w = args.package_watts

    def joules(seconds: float) -> float:
        return seconds * w

    rows = [{
        "scenario": s,
        "gate_s": round(gate[s], 2),
        "oracle_s": round(oracle[s], 3),
        "gate_J_upper": round(joules(gate[s]), 1),
        "oracle_J_upper": round(joules(oracle[s]), 1),
    } for s in shared]

    gate_total = sum(gate[s] for s in shared)
    oracle_total = sum(oracle[s] for s in shared)
    payload = {
        "measurement": (
            "Upper bound on the energy of one pre-deployment gate run and of "
            "the simulation oracle it was meant to displace, from measured "
            "wall-clock seconds times a declared package power."
        ),
        "method": "wall_clock_seconds * package_watts",
        "not_a_measurement": (
            "This is arithmetic over a nameplate figure, not an instrumented "
            "reading. RAPL or NVML counters would measure the draw; neither is "
            "used here. The workload is single-threaded, so attributing whole-"
            "package power to it overstates the result by roughly the loaded "
            "core count, which is why every figure is labelled an upper bound."
        ),
        "cpu": args.cpu,
        "package_watts": w,
        "power_source": DEFAULT_POWER_SOURCE,
        "gate_source": args.gate_file,
        "oracle_source": args.oracle_file,
        "n_scenarios": len(shared),
        "corpus_totals": {
            "gate_s": round(gate_total, 1),
            "oracle_s": round(oracle_total, 1),
            "gate_J_upper": round(joules(gate_total), 1),
            "oracle_J_upper": round(joules(oracle_total), 1),
            "gate_Wh_upper": round(joules(gate_total) / 3600.0, 4),
            "oracle_Wh_upper": round(joules(oracle_total) / 3600.0, 4),
        },
        "largest_scenario": max(rows, key=lambda r: r["gate_s"]),
        "per_scenario": rows,
    }
    payload["provenance"] = stamp(
        package_watts=w, cpu=args.cpu,
        gate_source=args.gate_file, oracle_source=args.oracle_file,
    )

    out = _resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    t = payload["corpus_totals"]
    big = payload["largest_scenario"]
    print(f"  {args.cpu}  @ {w:.0f} W package (upper bound, single-threaded load)")
    print("  " + "─" * 66)
    print(f"  whole corpus ({len(shared)} scenarios)")
    print(f"    gate    {t['gate_s']:>8.1f} s   <= {t['gate_J_upper']:>8.1f} J "
          f"({t['gate_Wh_upper']:.4f} Wh)")
    print(f"    oracle  {t['oracle_s']:>8.1f} s   <= {t['oracle_J_upper']:>8.1f} J "
          f"({t['oracle_Wh_upper']:.4f} Wh)")
    print(f"  largest scenario ({big['scenario']})")
    print(f"    gate    {big['gate_s']:>8.2f} s   <= {big['gate_J_upper']:>8.1f} J")
    print(f"    oracle  {big['oracle_s']:>8.3f} s   <= {big['oracle_J_upper']:>8.1f} J")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
