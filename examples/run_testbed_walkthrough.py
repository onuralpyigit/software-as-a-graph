#!/usr/bin/env python3
"""
Example script that runs the container testbed (docs/simulation-extensions.md,
section 3) on the 15-component mini-ATM topology and builds an interactive
HTML walkthrough.

It performs two real runs, each with one fault: a broker relay that stops for
twelve seconds, and a host that is killed. For each run it calibrates from the
telemetry, runs the fidelity comparison with declared and with calibrated
parameters, and sets the observed trace beside the simulator's.

Needs Docker with the compose plugin; the first run builds an image that
downloads ``python:3.11-slim`` and ``cyclonedds``. About three minutes in all.
"""

import argparse
import collections
import json
import logging
import shutil
import statistics
import sys
import warnings
from pathlib import Path

# Add project root to sys.path to support direct execution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saag.simulation.runtime.faults import FaultSpec
from saag.testbed import orchestrate
from saag.testbed.collect import load_run
from saag.validation.fidelity import stalled_in_trace

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = SCRIPT_DIR / "mini_atm.json"
TEMPLATE = SCRIPT_DIR / "mini_atm_testbed_run.template.html"
PAGE_HEAD = ('<!doctype html>\n<meta charset="utf-8">\n'
             '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
             '<style>body{margin:0}[hidden]{display:none!important}</style>\n')

DURATION_S = 40.0
WARMUP_S = 2.0
RUNS = {
    "run_a": [{"target": "B1", "mode": "transient_crash", "start_s": 15, "duration_s": 12}],
    "run_b": [{"target": "N1", "mode": "crash", "start_s": 20}],
}
FIDELITY_FIELDS = ("median_rel_error", "max_rel_error", "n_compared", "share_within_tolerance")


def r4(x):
    return None if x is None else round(float(x), 4)


def execute(run_dir: Path, faults: list) -> None:
    """build -> run (docker) -> calibrate -> fidelity, declared and calibrated."""
    orchestrate.build(TOPOLOGY, run_dir, DURATION_S, qos_semantics="dds")
    meta = orchestrate.run(run_dir, DURATION_S, [FaultSpec.from_dict(f) for f in faults], mode="docker")
    if meta["unsupported"]:
        sys.exit(f"Faults not emulated on the testbed: {meta['unsupported']}")
    orchestrate.calibrate(run_dir, warmup_s=WARMUP_S)
    orchestrate.fidelity(run_dir, warmup_s=WARMUP_S)
    orchestrate.fidelity(run_dir, calibrated=True, warmup_s=WARMUP_S)


def trace_view(trace, n: int, first_fault: float) -> dict:
    """Per-second publication and delivery counts, the stall set and pre-fault median latency."""
    pub = collections.defaultdict(lambda: [0] * n)
    dlv = collections.defaultdict(lambda: [0] * n)
    latency = collections.defaultdict(list)
    for e in trace.events:
        second = int(e["t"])
        if not 0 <= second < n:
            continue
        if e["kind"] == "publish":
            pub[e["app"]][second] += 1
        elif e["kind"] == "deliver":
            dlv[f"{e['app']}|{e['topic']}"][second] += 1
            if WARMUP_S <= e["t"] < first_fault and e.get("latency_ms") is not None:
                latency[e["topic"]].append(e["latency_ms"])
    return {"pub": dict(pub), "dlv": dict(dlv), "stalled": sorted(stalled_in_trace(trace, first_fault)),
            "lat_p50": {t: r4(statistics.median(v)) for t, v in latency.items()}}


def collect(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text())
    plan = json.loads((run_dir / "deployment.json").read_text())
    first_fault = min(f["start_s"] for f in meta["faults"])
    entry = {
        "meta": {k: meta[k] for k in ("duration_s", "faults", "mode")},
        "chaos": [{"t": r4(c["t"]), "what": " ".join(c["argv"]) if "argv" in c
                   else f"control file {c['control']}.json = {json.dumps(c['state'])}"} for c in meta["chaos_log"]],
        "side": {"observed": trace_view(load_run(run_dir), int(meta["duration_s"]), first_fault),
                 "simulated": trace_view(orchestrate.simulate_like(run_dir), int(meta["duration_s"]), first_fault)},
    }
    for name in ("fidelity", "fidelity_calibrated"):
        report = json.loads((run_dir / f"{name}.json").read_text())
        entry[name] = {
            "stall_jaccard": report["fault_response"]["stall_jaccard"],
            "metrics": {k: {f: r4(v[f]) for f in FIDELITY_FIELDS}
                        for k, v in report["steady_state"]["metrics"].items() if "median_rel_error" in v},
        }
        if name == "fidelity":
            entry["dds"] = {k: {"observed": v["run"]["fired_observed"], "simulated": v["run"]["fired_simulated"],
                                "jaccard": r4(v["run"]["fired_jaccard"])}
                            for k, v in report["dds_status"]["statuses"].items()}
            entry["dds_excluded"] = report["dds_status"]["excluded_observed_entities"]
    profile = json.loads((run_dir / "profile.json").read_text())
    entry["profile"] = {
        "rates": {t: [plan["topics"][t]["rate_hz"], r4(v)] for t, v in profile["msg_rate_per_sec"].items()},
        "service": {a: [r4(plan["apps"][a]["service_mean_s"] * 1000), r4(v["mean_s"] * 1000), v["dist"], v["n"]]
                    for a, v in profile["service_time_dist"].items()},
        "policy": {a: [plan["apps"][a]["policy"], profile["trigger_policy"].get(a)] for a in plan["apps"]},
    }
    return entry


def plan_view(run_dir: Path) -> dict:
    plan = json.loads((run_dir / "deployment.json").read_text())
    return {
        "hosts": plan["hosts"], "brokers": plan["brokers"],
        "apps": {a: {k: v[k] for k in ("policy", "publishes", "subscribes", "service_mean_s", "staleness_s")}
                 for a, v in plan["apps"].items()},
        "topics": {t: {k: v[k] for k in ("rate_hz", "reliability", "durability", "routers", "publishers", "deadline_ms")}
                   for t, v in plan["topics"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "mini_atm_testbed_run.html",
                        help="Where to write the HTML walkthrough")
    parser.add_argument("--workdir", type=Path, default=ROOT / "output/testbed_walkthrough",
                        help="Where the testbed directories and telemetry are kept")
    parser.add_argument("--reuse", action="store_true",
                        help="Rebuild the page from runs already in --workdir, without Docker")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    if not args.reuse and shutil.which("docker") is None:
        sys.exit("Docker was not found. Install it, or pass --reuse to rebuild the page from an earlier run.")

    data = {}
    for name, faults in RUNS.items():
        run_dir = args.workdir / name
        if not args.reuse:
            print(f"{name}: {faults[0]['mode']} on {faults[0]['target']} at {faults[0]['start_s']} s ...")
            execute(run_dir, faults)
        data[name] = collect(run_dir)
        delivery = data[name]["fidelity"]["metrics"]["delivery_rate"]["median_rel_error"]
        p95 = data[name]["fidelity"]["metrics"]["latency_p95"]["median_rel_error"]
        print(f"  stalled observed {data[name]['side']['observed']['stalled']} | "
              f"simulated {data[name]['side']['simulated']['stalled']} | "
              f"delivery error {delivery:.3f} | latency p95 error {p95:.3f}")
    data["plan"] = plan_view(args.workdir / "run_a")

    page = TEMPLATE.read_text().replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    args.output.write_text(PAGE_HEAD + page)
    print(f"\nWalkthrough written to {args.output}")


if __name__ == "__main__":
    main()
