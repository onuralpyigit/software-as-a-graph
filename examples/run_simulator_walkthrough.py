#!/usr/bin/env python3
"""
Example script that runs the four Step 5 simulators (docs/failure-simulation.md)
on the 15-component mini-ATM topology and builds an interactive HTML walkthrough
of what each engine does when a component fails.
"""

import argparse
import json
import logging
import sys
from collections import deque
from pathlib import Path

# Add project root to sys.path to support direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag.core.graph_io import load_graph
from saag.infrastructure.memory_repo import MemoryRepository
from mini_atm_scenarios import DURATION_S as SCENARIO_S
from mini_atm_scenarios import SCENARIOS

from saag.simulation.change_propagation import ChangePropagationSimulator
from saag.simulation.failure_simulator import FailureSimulator
from saag.simulation.fault_injector import RECOMMENDED_SEEDS, FaultInjector
from saag.simulation.message_flow_simulator import MessageFlowSimulator
from saag.simulation.models import FailureMode
from saag.simulation.runtime.faults import FaultSpec
from saag.simulation.service import SimulationService
from saag.testbed.collect import TelemetryTrace
from saag.validation.fidelity import stalled_in_trace

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = SCRIPT_DIR / "mini_atm.json"
TEMPLATE = SCRIPT_DIR / "mini_atm_failure_lab.template.html"
PAGE_HEAD = ('<!doctype html>\n<meta charset="utf-8">\n'
             '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
             '<style>body{margin:0}[hidden]{display:none!important}</style>\n')

DURATION_S, FAULT_TIME_S = 60.0, 30.0
THETA_LOOSE = THETA_STABLE = 0.20
# The settings the testbed's fidelity command gives MessageFlowSimulator
# (saag.testbed.orchestrate.simulate_like): reactive applications whose stale
# inputs stop them, and broker relays in the data path, so broker, host and
# link faults act on traffic.
TESTBED_LIKE = {"behavior_model": "reactive", "deployment_model": "stations", "rng_streams": "per_process",
                "replica_aware": True, "target_utilization": 0.65, "staleness_periods": 3.0,
                "qos_semantics": "dds", "dds_status": True, "liveliness_lease_s": 1.0}


def r4(x) -> float:
    return round(float(x), 4)


def run_fault_injector(graph) -> dict:
    """I*(v): cascade waves and final feed loss per subscriber, for every node type."""
    result = FaultInjector(graph, seeds=RECOMMENDED_SEEDS).run(
        node_types=["Application", "Broker", "Library", "Node", "Topic"])
    return {
        node: {
            "score": r4(rec.impact_score), "std": r4(rec.impact_score_std), "depth": rec.cascade_depth,
            "loss": {k: r4(v) for k, v in rec.per_subscriber_feed_loss.items()},
            "waves": [{"topics": w.newly_orphaned_topics, "subs": w.newly_impacted_subscribers,
                       "failed": w.newly_failed_publishers} for w in rec.cascade_waves],
        }
        for node, rec in result.records.items()
    }


def run_failure_simulator(service: SimulationService) -> tuple:
    """I_comp(v): system-layer sweep, plus one single run for the library."""
    results = list(service.run_failure_simulation_exhaustive(layer="system"))
    results.append(service.run_failure_simulation(["L1"]))
    out = {}
    for r in results:
        i = r.impact
        out[r.target_id] = {
            "comp": r4(i.composite_impact), "RL": r4(i.reachability_loss), "FR": r4(i.fragmentation),
            "TL": r4(i.throughput_loss), "FD": r4(i.flow_disruption),
            "paths": [i.initial_paths, i.remaining_paths],
            "failed": sorted(r.cascaded_failures),
            "seq": [{"id": e.component_id, "cause": e.cause, "depth": e.depth} for e in r.cascade_sequence],
        }
    weights = {k: r4(v) for k, v in results[0].impact.impact_weights.items()}
    return out, weights


def run_change_propagation(sim_graph) -> tuple:
    """I_M(v) sub-metrics, with the walk that produced each reached set."""
    dep = FailureSimulator(sim_graph)._build_dependency_view()
    results = ChangePropagationSimulator(theta_loose=THETA_LOOSE, theta_stable=THETA_STABLE).simulate_all(
        component_ids=dep.component_ids, dependency_edges=dep.edges, component_weights=dep.weights,
        component_in_degrees=dep.in_degrees, component_out_degrees=dep.out_degrees)

    def instability(c: str) -> float:
        return dep.out_degrees.get(c, 0) / (dep.in_degrees.get(c, 0) + dep.out_degrees.get(c, 0) + 1e-9)

    dependents = {c: [] for c in dep.component_ids}
    for src, tgt, w in dep.edges:
        if tgt in dependents:
            dependents[tgt].append((src, w))

    out = {}
    for source, res in results.items():
        # The simulator returns only the reached set, so the walk is retraced
        # here for the diagram and checked against it.
        reached, depth, queue, trace = set(), {source: 0}, deque([source]), []
        while queue:
            current = queue.popleft()
            for neighbor, w in dependents.get(current, []):
                if neighbor in reached or neighbor == source:
                    continue
                reached.add(neighbor)
                depth[neighbor] = depth[current] + 1
                loose, stable = w < THETA_LOOSE, instability(neighbor) < THETA_STABLE
                trace.append({"id": neighbor, "from": current, "d": depth[neighbor], "w": r4(w),
                              "stop": "loose" if loose else ("stable" if stable else "")})
                if not (loose or stable):
                    queue.append(neighbor)
        assert sorted(reached) == sorted(res.reached_components), source
        out[source] = {
            "reached": sorted(res.reached_components), "reach": r4(res.change_reach),
            "wimp": r4(res.weighted_change_impact), "depth": res.max_bfs_depth,
            "ndepth": r4(res.normalized_change_depth), "trace": trace,
        }
    return out, {c: r4(instability(c)) for c in dep.component_ids}


def run_message_flow(graph) -> dict:
    """I_dyn(v): one 60 s run per application, with deliveries binned per second."""
    apps = sorted(n for n, d in graph.nodes(data=True) if d.get("type") == "Application")
    topics = sorted(n for n, d in graph.nodes(data=True) if d.get("type") == "Topic")
    out = {}
    for app in apps:
        run = MessageFlowSimulator(graph, duration=DURATION_S, fault_node=app, fault_time=FAULT_TIME_S,
                                   seed=42, qos_mode="full", target_utilization=0.65,
                                   record_events=True).run()
        fault = run.fault_event
        bins = {t: [0] * int(DURATION_S) for t in topics}
        for event in run.events:
            if event["name"] == "saag.deliver":
                second = min(int(DURATION_S) - 1, int(event["time_unix_s"]))
                bins[event["attributes"]["messaging.destination.name"]][second] += 1
        out[app] = {
            "idyn": r4(fault.i_dyn), "pre": r4(fault.delivery_rate_before), "post": r4(fault.delivery_rate_after),
            "per_topic": {k: r4(v) for k, v in fault.per_topic_i_dyn.items()}, "bins": bins,
            "deadline": run.total_deadline_violations, "overflow": run.total_queue_overflows,
            "published": run.total_messages_published, "delivered": run.total_messages_delivered,
        }
    return out


def fault_window(faults: list) -> tuple:
    """From the first fault's start to the last fault's end (the run's end if one never ends)."""
    ends = [f["start_s"] + f["duration_s"] if f.get("duration_s") else SCENARIO_S for f in faults]
    return min(f["start_s"] for f in faults), max(ends)


def run_scenarios(graph, service: SimulationService) -> dict:
    """The proof-of-concept incidents: MessageFlowSimulator for every one, FailureSimulator where it can say it."""
    n, out = int(SCENARIO_S), {}
    for sc in SCENARIOS:
        res = MessageFlowSimulator(graph, duration=SCENARIO_S, seed=0, record_events=True,
                                   faults=[FaultSpec.from_dict(f) for f in sc["faults"]],
                                   network_gating=sc.get("network_gating", False), **TESTBED_LIKE).run()
        trace = TelemetryTrace.from_records(res.events, start_s=0.0, end_s=SCENARIO_S)
        f0, f1 = fault_window(sc["faults"])
        pub, dlv = {}, {}
        for e in trace.events:
            second = int(e["t"])
            if e["kind"] in ("publish", "deliver") and 0 <= second < n:
                book, key = (pub, e["app"]) if e["kind"] == "publish" else (dlv, e["topic"])
                book.setdefault(key, [0] * n)[second] += 1
        total = [sum(v[s] for v in dlv.values()) for s in range(n)]

        def rate(a, b):
            a, b = int(a), int(b)
            return r4(sum(total[a:b]) / (b - a)) if b > a else None

        entry = {"pub": pub, "dlv": dlv, "stalled": sorted(stalled_in_trace(trace, f0)),
                 "rate": [rate(2, f0), rate(f0, f1), rate(f1, n)],
                 "deadline": res.total_deadline_violations, "overflow": res.total_queue_overflows}
        if sc["fs"]:
            r = service.run_failure_simulation(sc["fs"]["targets"], failure_mode=FailureMode(sc["fs"]["mode"]))
            i = r.impact
            entry["fs"] = {"comp": r4(i.composite_impact), "RL": r4(i.reachability_loss), "FR": r4(i.fragmentation),
                           "TL": r4(i.throughput_loss), "FD": r4(i.flow_disruption),
                           "failed": sorted(r.cascaded_failures),
                           "seq": [{"id": e.component_id, "cause": e.cause, "depth": e.depth}
                                   for e in r.cascade_sequence]}
        out[sc["id"]] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "mini_atm_failure_lab.html",
                        help="Where to write the HTML walkthrough")
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    graph = load_graph(TOPOLOGY)
    repo = MemoryRepository()
    repo.save_graph(json.loads(TOPOLOGY.read_text()), clear=True)
    service = SimulationService(repo)
    sim_graph = service._get_graph()

    data = {"fi": run_fault_injector(graph)}
    data["fs"], data["fs_weights"] = run_failure_simulator(service)
    data["cp"], data["instab"] = run_change_propagation(sim_graph)
    data["mf"] = run_message_flow(graph)
    data["topic_w"] = {t: r4(info.weight) for t, info in sim_graph.topics.items()}
    data["sc"] = run_scenarios(graph, service)
    data["scenarios"] = SCENARIOS

    print(f"{'ID':<4}{'I*':>8}{'I_comp':>8}{'I_dyn':>8}{'I_M':>8}")
    for node in data["fi"]:
        cp = data["cp"][node]
        cells = [data["fi"][node]["score"],
                 data["fs"].get(node, {}).get("comp"),
                 data["mf"].get(node, {}).get("idyn"),
                 0.45 * cp["reach"] + 0.35 * cp["wimp"] + 0.20 * cp["ndepth"]]
        print(f"{node:<4}" + "".join(f"{'n/a' if c is None else format(c, '.3f'):>8}" for c in cells))
    print(f"\n{'Scenario':<10}{'before':>8}{'during':>8}{'after':>8}{'I_comp':>8}  stalled")
    for sid, sc in data["sc"].items():
        cells = sc["rate"] + [sc.get("fs", {}).get("comp")]
        print(f"{sid:<10}" + "".join(f"{'–' if c is None else format(c, '.2f'):>8}" for c in cells)
              + f"  {sc['stalled']}")

    page = TEMPLATE.read_text().replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    args.output.write_text(PAGE_HEAD + page)
    print(f"\nWalkthrough written to {args.output}")


if __name__ == "__main__":
    main()
