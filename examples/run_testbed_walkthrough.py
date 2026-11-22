#!/usr/bin/env python3
"""
Example script that runs the container testbed (docs/simulation.md,
section 3) on the 15-component mini-ATM topology and builds an interactive
HTML walkthrough.

It fails, one run each, every component the testbed can really fail -- the six
applications, both broker relays and both hosts -- with a permanent crash at
20 s, plus a twelve-second outage of the sensor-broker relay to show recovery.
For each run it calibrates from the telemetry, runs the fidelity comparison
with declared and with calibrated parameters, and sets the observed trace
beside the simulator's. It also runs FaultInjector and FailureSimulator, the
engines of the Failure Lab (run_simulator_walkthrough.py), so the page can set
their predictions beside what the containers did.

Needs Docker with the compose plugin; the first run builds an image that
downloads ``python:3.11-slim`` and ``cyclonedds``. About twelve minutes in all.
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

from mini_atm_scenarios import SCENARIOS
from run_simulator_walkthrough import run_failure_simulator, run_fault_injector, run_scenarios

from saag.core.graph_io import load_graph
from saag.infrastructure.memory_repo import MemoryRepository
from saag.simulation.runtime.faults import FaultSpec
from saag.simulation.service import SimulationService
from saag.testbed import orchestrate
from saag.testbed.collect import load_run
from saag.validation.fidelity import stalled_in_trace
from saag.validation.health import (
    compare_health_summaries,
    observed_operational_criticality,
    summarize_system_health,
)

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = SCRIPT_DIR / "mini_atm.json"
TEMPLATE = SCRIPT_DIR / "mini_atm_testbed_run.template.html"
PAGE_HEAD = ('<!doctype html>\n<meta charset="utf-8">\n'
             '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
             '<style>body{margin:0}[hidden]{display:none!important}</style>\n')

DURATION_S = 40.0
WARMUP_S = 2.0
FAULT_S = 20.0
# Topics and the library are not processes on the testbed, so they cannot be failed there.
RUNS = {c: [{"target": c, "mode": "crash", "start_s": FAULT_S}]
        for c in ("A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "N1", "N2")}
RUNS["B1_transient"] = [{"target": "B1", "mode": "transient_crash", "start_s": 15, "duration_s": 12}]
# The proof-of-concept incidents, shared with the Failure Lab.
RUNS.update({sc["id"]: sc["faults"] for sc in SCENARIOS})
GATED = {sc["id"] for sc in SCENARIOS if sc.get("network_gating")}
FIDELITY_FIELDS = ("median_rel_error", "max_rel_error", "n_compared", "share_within_tolerance")


def r4(x):
    return None if x is None else round(float(x), 4)


def execute(run_dir: Path, faults: list, network_gating: bool = False,
            mode: str = "docker", duration_s: float = DURATION_S) -> bool:
    """build -> run -> calibrate -> fidelity (declared & calibrated) -> health."""
    orchestrate.build(TOPOLOGY, run_dir, duration_s, qos_semantics="dds", network_gating=network_gating)
    meta = orchestrate.run(run_dir, duration_s, [FaultSpec.from_dict(f) for f in faults], mode=mode)
    if meta["unsupported"]:
        if mode == "local":
            return False
        sys.exit(f"Faults not emulated on the testbed: {meta['unsupported']}")
    orchestrate.calibrate(run_dir, warmup_s=WARMUP_S)
    orchestrate.fidelity(run_dir, warmup_s=WARMUP_S)
    orchestrate.fidelity(run_dir, calibrated=True, warmup_s=WARMUP_S)
    orchestrate.health(run_dir, warmup_s=WARMUP_S)
    return True


def _compact_health(hs: dict) -> dict:
    sh = hs["system_health"]
    piu = hs["product_in_use"]
    return {
        "counts": sh["counts"],
        "by_type": {t: b["counts"] for t, b in sh["by_type"].items()},
        "op_crit": sh["operational_critical"]["counts"],
        "qiu": r4(piu["overall_qiu_health_index"]),
        "ben": r4(piu["beneficialness"]["score"]),
        "risk": r4(piu["freedom_from_risk"]["score"]),
        "acc": r4(piu["acceptability"]["score"]),
        "zero_tol_loss": bool(piu["freedom_from_risk"]["zero_tolerance_critical_loss"]),
        "n_minus_0": piu["freedom_from_risk"]["n_minus_0_critical_ids"],
        "components": [
            {
                "id": cid,
                "type": c["type"],
                "state": c["state"],
                "score": r4(c["health_score"]),
                "crit": c["criticality"]["level"],
                "prov": c["criticality"]["provenance"],
                "reasons": c["reasons"],
            }
            for cid, c in sorted(hs["components"].items(), key=lambda kv: (kv[1]["type"], kv[0]))
        ],
    }


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


def chaos_text(c: dict) -> str:
    """One orchestrator action: a docker command, a control file, or a tc partition rule."""
    if "argv" in c:
        return " ".join(c["argv"])
    if "partition" in c:
        return (f"tc on {c['partition']}: drop traffic from {', '.join(c['blocked'])}" if c["blocked"]
                else f"tc on {c['partition']}: accept all traffic")
    return f"control file {c['control']}.json = {json.dumps(c['state'])}"


def collect(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text())
    plan = json.loads((run_dir / "deployment.json").read_text())
    first_fault = min(f["start_s"] for f in meta["faults"])
    n = int(meta["duration_s"])
    obs_tr = load_run(run_dir)
    sim_tr = orchestrate.simulate_like(run_dir)
    entry = {
        "meta": {k: meta[k] for k in ("duration_s", "faults", "mode")},
        "chaos": [{"t": r4(c["t"]), "what": chaos_text(c)} for c in meta["chaos_log"]],
        "side": {"observed": trace_view(obs_tr, n, first_fault),
                 "simulated": trace_view(sim_tr, n, first_fault)},
        "health_events_count": sum(1 for e in obs_tr.events if e["kind"] == "health"),
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
    h_path = run_dir / "health_summary.json"
    if h_path.exists():
        obs_hs = json.loads(h_path.read_text())
    else:
        obs_hs = summarize_system_health(
            obs_tr, plan, fault_time=first_fault, warmup_s=WARMUP_S,
            faults=meta["faults"], docker_cpu_p95=meta.get("docker_cpu_p95"),
        )
    sim_hs = summarize_system_health(
        sim_tr, plan, fault_time=first_fault, warmup_s=WARMUP_S, faults=meta["faults"],
    )
    h_cmp = compare_health_summaries(sim_hs, obs_hs)
    primary_f = meta["faults"][0]
    oc = observed_operational_criticality(
        obs_tr, plan, str(primary_f["target"]), float(primary_f["start_s"]),
        warmup_s=WARMUP_S, fault_duration_s=primary_f.get("duration_s"), health_summary=obs_hs,
    )
    entry["health"] = _compact_health(obs_hs)
    entry["health_agreement"] = {
        "exact_state_agreement": r4(h_cmp["exact_state_agreement"]),
        "op_crit_lost_jaccard": r4(h_cmp["operational_critical_lost_jaccard"]),
        "qiu_abs_error": r4(h_cmp["qiu_health_index_abs_error"]),
    }
    entry["op_crit"] = {
        "target": oc["target"],
        "target_type": oc["target_type"],
        "static_crit": (oc.get("static_criticality") or {}).get("level") or "LOW",
        "static_prov": (oc.get("static_criticality") or {}).get("provenance") or "default",
        "i_flow": r4(oc["i_flow"]),
        "i_delivery": r4(oc["i_delivery"]),
        "i_delivery_consumer": r4(oc["i_delivery_consumer"]),
        "i_timeliness": r4(oc["i_timeliness"]),
        "i_recovery": r4(oc["i_recovery"]),
        "qiu_impact": r4(oc["qiu_impact"]),
        "op_crit_lost": int(oc["lost_operational_critical_count"]),
    }
    profile = json.loads((run_dir / "profile.json").read_text())
    entry["profile"] = {
        "rates": {t: [plan["topics"][t]["rate_hz"], r4(v)] for t, v in profile["msg_rate_per_sec"].items()},
        "service": {a: [r4(plan["apps"][a]["service_mean_s"] * 1000), r4(v["mean_s"] * 1000), v["dist"], v["n"]]
                    for a, v in profile["service_time_dist"].items()},
        "policy": {a: [plan["apps"][a]["policy"], profile["trigger_policy"].get(a)] for a in plan["apps"]},
    }
    return entry


def predictions() -> dict:
    """What the Failure Lab's FaultInjector (I*) and FailureSimulator (I_comp) say each failure and incident does."""
    graph = load_graph(TOPOLOGY)
    fi = run_fault_injector(load_graph(TOPOLOGY))
    repo = MemoryRepository()
    repo.save_graph(json.loads(TOPOLOGY.read_text()), clear=True)
    service = SimulationService(repo)
    fs, _ = run_failure_simulator(service)
    return {
        "scenarios": SCENARIOS,
        # The lab's I_comp for each incident, where FailureSimulator can express it.
        "sc_fs": {k: v.get("fs") for k, v in run_scenarios(graph, service).items()},
        "fi": {c: {"score": r["score"], "loss": r["loss"],
                   "failed": sorted({a for w in r["waves"] for a in w["failed"]})} for c, r in fi.items()},
        "fs": {c: {"comp": r["comp"], "failed": r["failed"],
                   "hit": sorted({e["id"] for e in r["seq"] if e["depth"] > 0})} for c, r in fs.items()},
    }


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
    parser.add_argument("--mode", choices=("docker", "local"), default="docker",
                        help="Testbed execution mode (default: docker; local runs in-process without Docker)")
    parser.add_argument("--duration", type=float, default=DURATION_S,
                        help="Per-run duration in seconds (default: 40.0)")
    parser.add_argument("--reuse", action="store_true",
                        help="Rebuild the page from runs already in --workdir, without running")
    parser.add_argument("--only", default="",
                        help="Comma-separated run ids to (re)run, e.g. S1,S4; the others are reused")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    only = {r for r in args.only.split(",") if r}
    if only - set(RUNS):
        sys.exit(f"Unknown run ids: {sorted(only - set(RUNS))}")
    if not args.reuse and args.mode == "docker" and shutil.which("docker") is None:
        sys.exit("Docker was not found. Install it, pass --mode local, or pass --reuse.")

    data = {"runs": {}, "pred": predictions(), "mode": args.mode}
    for name, faults in RUNS.items():
        run_dir = args.workdir / name
        if not args.reuse and (not only or name in only):
            print(f"{name}: " + ", ".join(f"{f['mode']} on {f['target']} at {f['start_s']} s" for f in faults) + " ...")
            ok = execute(run_dir, faults, network_gating=name in GATED,
                         mode=args.mode, duration_s=args.duration)
            if not ok:
                print(f"  {name}: skipped (unsupported in {args.mode} mode)")
                continue
        if not (run_dir / "run.json").exists():
            continue
        run = data["runs"][name] = collect(run_dir)
        obs, sim = run["side"]["observed"], run["side"]["simulated"]
        delivery = run["fidelity"]["metrics"]["delivery_rate"]["median_rel_error"]
        print(f"  {name}: silent observed {obs['stalled']} simulated {sim['stalled']} | "
              f"delivery err {delivery:.3f} | QiU {run['health']['qiu']:.3f} | i_flow {run['op_crit']['i_flow']:.3f}")
    data["plan"] = plan_view(args.workdir / "A1")
    data["op_crit_ranking"] = sorted(
        [
            data["runs"][c]["op_crit"]
            for c in ("A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "N1", "N2")
            if c in data["runs"] and "op_crit" in data["runs"][c]
        ],
        key=lambda r: (-(r["i_flow"] if r["i_flow"] is not None else -1e9), r["target"]),
    )

    page = TEMPLATE.read_text().replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    args.output.write_text(PAGE_HEAD + page)
    print(f"\nWalkthrough written to {args.output}")


if __name__ == "__main__":
    main()
