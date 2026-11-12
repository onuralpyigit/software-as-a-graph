#!/usr/bin/env python3
"""
Example script that walks through the two ends of deployment on the mini-ATM
topology and builds an interactive HTML page.

Before deployment (docs/deployment.md): the system is given four hosts in two
racks and a deployment spec; the optimizer proposes placements, every candidate
and baseline is re-scored on the replica-aware failure simulator, and a
placement is recommended with or without a reliability claim.

After deployment (docs/simulation-extensions.md, section 3): the recommended
placement is run as real processes on the in-process testbed while one host is
crashed; the telemetry is calibrated and compared with the simulator's version
of the same run. No Docker is needed; the run takes about half a minute.
"""

import argparse
import collections
import copy
import json
import logging
import sys
import warnings
from pathlib import Path

# Add project root to sys.path to support direct execution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saag import Client
from saag.deployment import apply_placement
from saag.infrastructure.memory_repo import MemoryRepository
from saag.simulation.runtime.faults import FaultSpec
from saag.testbed import orchestrate
from saag.testbed.collect import load_run
from saag.validation.fidelity import stalled_in_trace

from walkthrough_page import write_page

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = SCRIPT_DIR / "mini_atm.json"
TEMPLATE = SCRIPT_DIR / "deployment_walkthrough.template.html"

EXTRA_HOSTS = [{"id": "N3", "name": "sensor-standby"}, {"id": "N4", "name": "ops-standby"}]
EXTRA_LINKS = [("N1", "N3"), ("N2", "N4"), ("N3", "N4")]
#: Two racks of two hosts. Each host fits about four components, tracker and
#: conflict-detector run as two replicas, and replicas must sit in different racks.
SPEC = {
    "version": 1,
    "defaults": {"node": {"cpu": 4, "mem_mb": 8192, "p": 0.02}, "domain": {"p": 0.005},
                 "app": {"cpu": 1.0, "mem_mb": 512}, "broker": {"cpu": 1.0, "mem_mb": 1024}},
    "nodes": {"N1": {"domain": "rack-a"}, "N3": {"domain": "rack-a"},
              "N2": {"domain": "rack-b"}, "N4": {"domain": "rack-b"}},
    "domains": {"rack-a": {"p": 0.004}, "rack-b": {"p": 0.004}},
    "latency": {"intra_domain_ms": 0.5, "inter_domain_ms": 5.0},
    "apps": {"A2": {"replicas": 2}, "A3": {"replicas": 2}},
    "constraints": {"anti_affinity": "domain", "use_all_hosts": False, "max_utilization": 0.9},
}
DURATION_S, FAULT_S, WARMUP_S = 24.0, 10.0, 2.0
FIDELITY_FIELDS = ("median_rel_error", "max_rel_error", "n_compared", "share_within_tolerance")


def r5(x):
    return None if x is None else round(float(x), 5)


def four_host_topology() -> dict:
    topology = copy.deepcopy(json.loads(TOPOLOGY.read_text()))
    topology["nodes"] += EXTRA_HOSTS
    topology["relationships"]["connects_to"] += [{"from": a, "to": b} for a, b in EXTRA_LINKS]
    return topology


def candidate_view(c: dict) -> dict:
    held = c["held_out"] or {}
    return {
        "name": c["name"], "kind": c["kind"], "placement": c["placement"], "traffic": r5(c["traffic"]),
        "cross_host": r5(c["cross_host_share"]), "blast": r5(c["blast_exp"]),
        "impact": r5(held.get("expected_impact")), "worst": r5(held.get("worst_impact")),
        "sigma": r5(held.get("sigma_seed")), "per_event": {k: r5(v) for k, v in (held.get("per_event") or {}).items()},
        "verdict": c["verdict"], "margin": r5(c["margin"]), "noise_bar": r5(c["noise_bar"]),
        "violations": {k: v for k, v in c["violations"].items() if v},
    }


def per_second(trace, n: int) -> dict:
    pub = collections.defaultdict(lambda: [0] * n)
    for e in trace.events:
        if e["kind"] == "publish" and 0 <= int(e["t"]) < n:
            pub[e["app"]][int(e["t"])] += 1
    return dict(pub)


def post_deployment(placed: dict, placement: dict, workdir: Path) -> dict:
    """Run the placed system on the in-process testbed with one host crash, then calibrate and compare."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "placed.json").write_text(json.dumps(placed))
    testbed = workdir / "testbed"
    target = placement["A2"][0]                       # the host carrying tracker's primary copy
    orchestrate.build(workdir / "placed.json", testbed, DURATION_S, qos_semantics="dds")
    meta = orchestrate.run(testbed, DURATION_S, [FaultSpec.from_dict({"target": target, "mode": "crash",
                                                                      "start_s": FAULT_S})], mode="local")
    profile = orchestrate.calibrate(testbed, warmup_s=WARMUP_S)
    fidelity = orchestrate.fidelity(testbed, warmup_s=WARMUP_S)
    plan = json.loads((testbed / "deployment.json").read_text())
    observed, simulated, n = load_run(testbed), orchestrate.simulate_like(testbed), int(DURATION_S)
    return {
        "target": target, "duration": DURATION_S, "fault_s": FAULT_S,
        "chaos": [{"t": r5(c["t"]), "what": c.get("control") or " ".join(c.get("argv", []))} for c in meta["chaos_log"]],
        "states": [{"t": r5(e["t"]), "app": e["app"], "state": e["state"]} for e in observed.events if e["kind"] == "state"],
        "pub": {"observed": per_second(observed, n), "simulated": per_second(simulated, n)},
        "stalled": {"observed": sorted(stalled_in_trace(observed, FAULT_S)),
                    "simulated": sorted(stalled_in_trace(simulated, FAULT_S))},
        "stall_jaccard": fidelity["fault_response"]["stall_jaccard"],
        "fidelity": {k: {f: r5(v[f]) for f in FIDELITY_FIELDS}
                     for k, v in fidelity["steady_state"]["metrics"].items() if "median_rel_error" in v},
        "rates": {t: [plan["topics"][t]["rate_hz"], r5(v)] for t, v in profile["msg_rate_per_sec"].items()},
        "service": {a: [r5(plan["apps"][a]["service_mean_s"] * 1000), r5(v["mean_s"] * 1000), v["n"]]
                    for a, v in profile["service_time_dist"].items()},
        "policy": {a: [plan["apps"][a]["policy"], profile["trigger_policy"].get(a)] for a in plan["apps"]},
        "routers": {t: plan["topics"][t]["routers"] for t in plan["topics"]},
        "rates_hz": {a: plan["topics"][plan["apps"][a]["publishes"][0]]["rate_hz"]
                     for a in plan["apps"] if plan["apps"][a]["publishes"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "deployment_walkthrough.html",
                        help="Where to write the HTML walkthrough")
    parser.add_argument("--workdir", type=Path, default=ROOT / "output/deployment_walkthrough",
                        help="Where the placed topology and the testbed run are kept")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    topology = four_host_topology()
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)
    repo.derive_dependencies()
    result = Client(repo=repo).deploy(SPEC).to_dict()
    if not result["recommended"]:
        sys.exit(f"No placement was recommended (status {result['status']}).")
    candidates = [candidate_view(c) for c in result["candidates"]]
    recommended = next(c for c in candidates if c["name"] == result["recommended"])
    print(f"Pre-deployment: {result['status']}, recommended {result['recommended']}, "
          f"best baseline {result['best_baseline']}, reliability claim: {result['reliability_claim']}")
    print(f"  {result['rationale']}")

    placed = apply_placement(repo.export_json(), recommended["placement"])
    print(f"Post-deployment: running the placement for {DURATION_S:.0f} s on the in-process testbed ...")
    post = post_deployment(placed, recommended["placement"], args.workdir)
    print(f"  crashed {post['target']} at {FAULT_S:.0f} s; stalled observed {post['stalled']['observed']}, "
          f"simulated {post['stalled']['simulated']}")

    data = {
        "hosts": {h["id"]: h["name"] for h in topology["nodes"]},
        "spec": {"nodes": SPEC["nodes"], "node_default": SPEC["defaults"]["node"], "domain_p": SPEC["domains"],
                 "replicas": {a: v["replicas"] for a, v in SPEC["apps"].items()},
                 "app_default": SPEC["defaults"]["app"], "broker_default": SPEC["defaults"]["broker"],
                 "constraints": SPEC["constraints"]},
        "pre": {
            "status": result["status"], "solver": result["solver"], "candidates": candidates,
            "recommended": result["recommended"], "best_baseline": result["best_baseline"],
            "claim": result["reliability_claim"], "rationale": result["rationale"],
            "noise_floor": r5(result["noise_floor"]), "proxy_oracle_rho": r5(result["proxy_oracle_rho"]),
            "events": result["oracle"]["events"], "p": result["oracle"]["p"], "seeds": result["oracle"]["seeds"],
        },
        "post": post,
    }
    write_page(TEMPLATE, args.output, data)
    print(f"\nWalkthrough written to {args.output}")


if __name__ == "__main__":
    main()
