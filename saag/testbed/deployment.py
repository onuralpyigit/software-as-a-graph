"""Scenario -> deployment plan for the emulated testbed."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from saag.core.graph_io import load_graph
from saag.core.models import QoSPolicy
from saag.core.network import NETWORK_TYPES
from saag.simulation.runtime.behavior import STALENESS_PERIODS, policy_for, staleness_s

#: Host given to components that declare no RUNS_ON.
DEFAULT_HOST = "h_default"
#: How a stub's service time is spent: asleep (no CPU) or burning its own CPU.
SERVICE_WORK = ("sleep", "cpu")


def _rate(attrs: Dict[str, Any]) -> float:
    try:
        return float(attrs.get("frequency") or attrs.get("topic_frequency") or 10.0)
    except (TypeError, ValueError):
        return 10.0


def build_deployment(
    scenario: Path,
    target_utilization: float = 0.65,
    staleness_periods: float = STALENESS_PERIODS,
    qos_semantics: str = "engine",
    liveliness_lease_s: float = 1.0,
    open_loop: bool = False,
    service_work: str = "sleep",
    carry_payload: bool = False,
    network_gating: bool = False,
) -> Dict[str, Any]:
    """Resolve everything an agent needs, once, so all agents agree on it.

    ``qos_semantics`` as in `MessageFlowSimulator`: ``"dds"`` makes the agents
    count deadlines as period misses instead of dropping late samples, and
    replace the oldest sample of a full BEST_EFFORT queue.
    ``liveliness_lease_s`` becomes every data writer's and reader's
    AUTOMATIC liveliness lease.
    ``open_loop`` makes every application emit on schedule whatever its inputs
    do, the published `MessageFlowSimulator`'s assumption.
    ``service_work="cpu"`` makes each service burn its time on the CPU instead
    of sleeping, and caps every host container at the simulator's host
    capacity (``cpu_cores``, else one core per resident station), so
    co-located applications really compete for a host. Broker relays keep
    sleeping: the simulator's broker stations do not share a host CPU.
    ``carry_payload`` puts each topic's ``size`` bytes on the wire.
    ``network_gating`` partitions hosts by CONNECTS_TO reachability when links
    flap or network vertices crash, as `MessageFlowSimulator(network_gating=True)`.
    """
    if service_work not in SERVICE_WORK:
        raise ValueError(f"service_work must be one of {SERVICE_WORK}, got {service_work!r}")
    from saag.simulation.runtime.dds_status import QOS_SEMANTICS
    if qos_semantics not in QOS_SEMANTICS:
        raise ValueError(f"qos_semantics must be one of {QOS_SEMANTICS}, got {qos_semantics!r}")
    g = load_graph(Path(scenario))
    ntype = {n: d.get("type") for n, d in g.nodes(data=True)}
    pubs, subs, routers, hosts_of = (defaultdict(list) for _ in range(4))
    for u, v, d in g.edges(data=True):
        t = d.get("type")
        if t == "PUBLISHES_TO":
            pubs[v].append(u)
        elif t == "SUBSCRIBES_TO":
            subs[v].append(u)
        elif t == "ROUTES":
            routers[v].append(u)
        elif t == "RUNS_ON":
            hosts_of[u].append(v)

    topics: Dict[str, Any] = {}
    for t, d in sorted(g.nodes(data=True)):
        if d.get("type") != "Topic":
            continue
        qos = QoSPolicy.from_node_attrs(d)
        topics[t] = {
            "rate_hz": _rate(d),
            "liveliness_lease_s": liveliness_lease_s,
            "reliability": qos.reliability,
            "durability": qos.durability,
            "deadline_ms": d.get("deadline_ms"),
            "history_depth": d.get("history_depth"),
            "priority": qos.transport_priority,
            "size": int(d.get("size") or 64),
            "publishers": sorted(pubs.get(t, [])),
            "routers": sorted(routers.get(t, [])),
        }

    published = {t for t, ps in pubs.items() if ps}
    apps: Dict[str, Any] = {}
    for a in sorted(n for n, t in ntype.items() if t in ("Application", "Library")):
        out = sorted(t for t, ps in pubs.items() if a in ps and t in topics)
        ins = sorted(t for t, ss in subs.items() if a in ss and t in topics)
        if not out and not ins:
            continue
        required = sorted((set(ins) & published) - set(out))
        load = sum(topics[t]["rate_hz"] for t in ins)
        apps[a] = {
            "publishes": out,
            "subscribes": ins,
            "inputs": required,
            "policy": "periodic" if open_loop or not required else (
                g.nodes[a].get("trigger") or policy_for(g.nodes[a]).kind),
            "staleness_s": {t: staleness_s(g.nodes[t], topics[t]["rate_hz"], staleness_periods)
                            for t in required},
            "service_mean_s": (target_utilization / load) if load > 0 else 0.0,
            "per_publisher_rate": {t: topics[t]["rate_hz"] / len(topics[t]["publishers"])
                                   for t in out},
            "phase_index": {t: topics[t]["publishers"].index(a) for t in out},
            "hotstandby": bool(g.nodes[a].get("hotstandby")),
            # Every host it RUNS_ON: one instance each, lowest-id live one active.
            "hosts": sorted(hosts_of.get(a, [])) or [DEFAULT_HOST],
        }

    brokers: Dict[str, Any] = {}
    for b in sorted(n for n, t in ntype.items() if t == "Broker"):
        routed = sorted(t for t, rs in routers.items() if b in rs and t in topics)
        load = sum(topics[t]["rate_hz"] for t in routed)
        brokers[b] = {"routes": routed,
                      "service_mean_s": (target_utilization / load) if load > 0 else 0.0}

    hosts: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"apps": [], "brokers": []})
    # A multi-hosted application runs an instance on every host (active /
    # standby, see runtime.AppRuntime), so a host outage takes it down only when
    # all of its hosts are down -- `replica_aware` semantics. Brokers stay on
    # their first host: their redundancy is the router set of each topic.
    for a in apps:
        for h in apps[a]["hosts"]:
            hosts[h]["apps"].append(a)
    for b in brokers:
        for h in (sorted(hosts_of.get(b, [])) or [DEFAULT_HOST])[:1]:
            hosts[h]["brokers"].append(b)

    cpus: Dict[str, int] = {}
    if service_work == "cpu":
        for h in hosts:
            # As `_share_host_cpu`: the active instance (first host) holds a station.
            stations = sum(1 for a in hosts[h]["apps"]
                           if apps[a]["subscribes"] and apps[a]["hosts"][0] == h)
            declared = g.nodes[h].get("cpu_cores") if h in g else None
            cpus[h] = int(declared) if declared else max(1, stations)

    return {
        "schema_version": "1.0",
        "scenario": g.graph.get("id", Path(scenario).stem),
        "target_utilization": target_utilization,
        "staleness_periods": staleness_periods,
        "qos_semantics": qos_semantics,
        "open_loop": open_loop,
        "liveliness_lease_s": liveliness_lease_s,
        "service_work": service_work,
        "carry_payload": carry_payload,
        "cpus": cpus,
        "topics": topics,
        "apps": apps,
        "brokers": brokers,
        "hosts": {h: hosts[h] for h in sorted(hosts)},
        "links": sorted({tuple(sorted((u, v))) for u, v, d in g.edges(data=True)
                         if d.get("type") == "CONNECTS_TO"}),
        "network": {"gating": network_gating,
                    "vertices": sorted(n for n, t in ntype.items() if t in NETWORK_TYPES)},
    }
