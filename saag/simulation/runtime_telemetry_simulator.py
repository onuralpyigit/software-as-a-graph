"""
runtime_telemetry_simulator.py
───────────────────────────────
Unified all-in-one discrete-event runtime simulator for the SaG pipeline.
Simulates concurrent traffic across physical Nodes, logical Brokers & Topics,
and software Applications & Libraries, emitting complete operational telemetry.
"""

from __future__ import annotations

import heapq
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from saag.core.models import QoSPolicy, GraphData, ComponentData, EdgeData
from saag.simulation.graph import SimulationGraph
from saag.simulation.models import ComponentState
from saag.simulation.simulation_results import FaultInjectionResult, FaultInjectionRecord
from saag.simulation._stats import percentile
from .telemetry.models import (
    ComponentTelemetry,
    NodeTelemetry,
    QoSViolationEvent,
    StarvationEvent,
    SystemTelemetry,
    TelemetryScenario,
    TopicTelemetry,
)
from .telemetry.impact_calculator import TelemetryImpactCalculator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal Event & Message Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class _SimEvent:
    """Internal event representation sorted by time, then sequence tie-breaker."""
    time: float
    sequence: int
    event_type: str = field(compare=False)
    data: Dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class _SimMessage:
    """In-flight message carrying timing and payload attributes."""
    msg_id: str
    topic_id: str
    publisher_id: str
    created_at: float
    payload_size_bytes: int = 64
    qos_reliability: str = "RELIABLE"
    qos_durability: str = "VOLATILE"
    deadline_ms: Optional[float] = None
    lifespan_ms: Optional[float] = None
    hops: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Queue Modeling
# ─────────────────────────────────────────────────────────────────────────────

class _BoundedQueue:
    """Receive queue with DDS KEEP_LAST overflow policy (head-drop for RELIABLE, tail-drop for BEST_EFFORT)."""

    def __init__(self, capacity: int, reliability: str = "RELIABLE") -> None:
        self.capacity = max(1, capacity)
        self.reliability = reliability.upper()
        self.items: List[_SimMessage] = []
        self.total_dropped_overflow: int = 0

    @property
    def depth(self) -> int:
        return len(self.items)

    def put(self, msg: _SimMessage) -> bool:
        """Enqueue message; returns True if accepted, False if dropped."""
        if len(self.items) >= self.capacity:
            self.total_dropped_overflow += 1
            if self.reliability == "BEST_EFFORT":
                return False
            # RELIABLE head-drop (pop oldest to keep fresh)
            if self.items:
                self.items.pop(0)
        self.items.append(msg)
        return True

    def pop(self) -> Optional[_SimMessage]:
        return self.items.pop(0) if self.items else None


# ─────────────────────────────────────────────────────────────────────────────
# Runtime Telemetry Simulator
# ─────────────────────────────────────────────────────────────────────────────

class RuntimeTelemetrySimulator:
    """
    All-in-one discrete-event simulator executing concurrent cross-layer traffic.

    Simulates:
      - Multi-publisher concurrent workloads (periodic or Poisson).
      - Physical host Node CPU and network interface bandwidth.
      - Broker queues with head-drop/tail-drop overflow enforcement.
      - Topic DDS QoS contracts (reliability, deadlines, lifespans, history).
      - Multi-target fault injection cascading across RUNS_ON, ROUTES, and USES.
      - System runtime telemetry collection and delta-based impact calculation.
    """

    def __init__(
        self,
        graph: Union[SimulationGraph, nx.DiGraph, Any],
        scenario: Optional[TelemetryScenario] = None,
    ) -> None:
        self.graph = self._ensure_simulation_graph(graph)
        self.scenario = scenario or TelemetryScenario()
        self.impact_calculator = TelemetryImpactCalculator(
            propagation_threshold=self.scenario.propagation_threshold
        )

    @staticmethod
    def _ensure_simulation_graph(graph: Any) -> SimulationGraph:
        """Normalize graph input to a SimulationGraph instance."""
        if isinstance(graph, SimulationGraph):
            return graph
        if isinstance(graph, nx.DiGraph):
            components: List[ComponentData] = []
            edges: List[EdgeData] = []
            for node, data in graph.nodes(data=True):
                ntype = data.get("type") or data.get("ntype") or "Application"
                components.append(
                    ComponentData(
                        id=str(node),
                        component_type=str(ntype),
                        weight=float(data.get("weight", 1.0)),
                        properties=dict(data),
                    )
                )
            for src, tgt, data in graph.edges(data=True):
                etype = (data.get("type") or data.get("etype") or data.get("relation") or "UNKNOWN").upper()
                dep_type = data.get("dependency_type") or "structural"
                edges.append(
                    EdgeData(
                        source_id=str(src),
                        target_id=str(tgt),
                        source_type="Unknown",
                        target_type="Unknown",
                        dependency_type=str(dep_type),
                        relation_type=str(etype),
                        weight=float(data.get("weight", 1.0)),
                    )
                )
            gd = GraphData(components=components, edges=edges)
            return SimulationGraph(graph_data=gd)
        raise TypeError(f"Expected SimulationGraph or nx.DiGraph, got {type(graph).__name__}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public Execution APIs
    # ─────────────────────────────────────────────────────────────────────────

    def simulate(self, scenario: Optional[TelemetryScenario] = None) -> SystemTelemetry:
        """Execute simulation for the given scenario and return collected telemetry."""
        sc = scenario or self.scenario
        self.graph.reset()

        rng = random.Random(sc.seed)
        event_queue: List[_SimEvent] = []
        seq = 0

        def push_event(t: float, ev_type: str, data: Dict[str, Any]) -> None:
            nonlocal seq
            heapq.heappush(event_queue, _SimEvent(time=t, sequence=seq, event_type=ev_type, data=data))
            seq += 1

        # State tracking
        sim_time = 0.0
        failed_nodes: Set[str] = set()
        active_components: Dict[str, bool] = {cid: True for cid in self.graph.components}

        # Initialize telemetry containers
        comp_telemetry: Dict[str, ComponentTelemetry] = {}
        for cid, comp in self.graph.components.items():
            name = comp.properties.get("name", cid) if comp.properties else cid
            comp_telemetry[cid] = ComponentTelemetry(
                component_id=cid,
                component_type=comp.type,
                component_name=name,
            )

        topic_telemetry: Dict[str, TopicTelemetry] = {}
        for tid, tinfo in self.graph.topics.items():
            comp_node = self.graph.components.get(tid)
            props = comp_node.properties if comp_node else {}
            dl = getattr(tinfo, "deadline_ms", None) or props.get("qos_deadline_ms") or props.get("deadline_ms")
            topic_telemetry[tid] = TopicTelemetry(
                topic_id=tid,
                topic_name=tinfo.name or tid,
                qos_reliability=tinfo.qos_reliability,
                qos_durability=tinfo.qos_durability,
                deadline_ms=dl,
            )

        node_telemetry: Dict[str, NodeTelemetry] = {}
        for cid, comp in self.graph.components.items():
            if comp.type == "Node":
                hosted = self.graph.get_hosted_components(cid)
                node_telemetry[cid] = NodeTelemetry(
                    node_id=cid,
                    node_name=comp.properties.get("name", cid) if comp.properties else cid,
                    hosted_components=hosted,
                )

        # Queues
        broker_queues: Dict[str, _BoundedQueue] = {}
        for cid, comp in self.graph.components.items():
            if comp.type == "Broker":
                broker_queues[cid] = _BoundedQueue(capacity=sc.default_queue_capacity)

        subscriber_queues: Dict[Tuple[str, str], _BoundedQueue] = {}  # (topic_id, sub_id)
        for tid in self.graph.topics:
            tinfo = self.graph.topics[tid]
            rel = tinfo.qos_reliability or "RELIABLE"
            for sub_id in self.graph.get_subscribers(tid):
                subscriber_queues[(tid, sub_id)] = _BoundedQueue(
                    capacity=sc.default_queue_capacity,
                    reliability=rel,
                )

        # Telemetry samples
        latency_samples_all: List[float] = []
        latency_samples_pre: List[float] = []
        latency_samples_post: List[float] = []
        qos_violations: List[QoSViolationEvent] = []
        starvation_events: List[StarvationEvent] = []
        delivered_subscribers: Dict[str, Set[str]] = defaultdict(set)

        total_msgs_generated = 0
        total_msgs_delivered = 0
        total_msgs_dropped = 0

        pre_delivered = 0
        pre_expected = 0
        post_delivered = 0
        post_expected = 0

        fault_time = sc.fault_time if sc.fault_node else None

        # ── Schedule Initial Fault Injection ─────────────────────────────────
        if sc.fault_node and sc.fault_node in self.graph.components and fault_time is not None:
            push_event(fault_time, "INJECT_FAULT", {"target_id": sc.fault_node})

        # ── Schedule Publisher Workloads ─────────────────────────────────────
        msg_id_counter = 0
        sorted_topics = sorted(self.graph.topics.keys())

        for tid in sorted_topics:
            tinfo = self.graph.topics[tid]
            publishers = sorted(self.graph.get_publishers(tid))
            if not publishers:
                continue

            comp_node = self.graph.components.get(tid)
            props = comp_node.properties if comp_node else {}
            rate_hz = sc.default_publish_rate_hz
            if "frequency" in props:
                try:
                    rate_hz = float(props["frequency"])
                except (TypeError, ValueError):
                    pass
            elif "topic_frequency" in props:
                try:
                    rate_hz = float(props["topic_frequency"])
                except (TypeError, ValueError):
                    pass

            per_pub_rate = rate_hz / len(publishers) if publishers else rate_hz
            deadline_ms = getattr(tinfo, "deadline_ms", None) or props.get("qos_deadline_ms") or props.get("deadline_ms")

            for pub_id in publishers:
                t = (1.0 / per_pub_rate) * rng.random() if sc.poisson_arrivals else 0.0
                while t < sc.duration:
                    msg_id = f"m_{msg_id_counter}"
                    msg_id_counter += 1
                    push_event(
                        t,
                        "PUBLISH",
                        {
                            "msg_id": msg_id,
                            "topic_id": tid,
                            "publisher_id": pub_id,
                            "created_at": t,
                            "size_bytes": getattr(tinfo, "message_size", 64) or 64,
                            "qos_reliability": tinfo.qos_reliability,
                            "qos_durability": tinfo.qos_durability,
                            "deadline_ms": deadline_ms,
                        },
                    )
                    interval = rng.expovariate(per_pub_rate) if sc.poisson_arrivals else (1.0 / per_pub_rate)
                    t += interval

        # ── Discrete Event Loop ──────────────────────────────────────────────
        while event_queue:
            event = heapq.heappop(event_queue)
            if event.time > sc.duration:
                break
            sim_time = event.time
            ev_type = event.event_type
            d = event.data

            # 1. INJECT FAULT
            if ev_type == "INJECT_FAULT":
                tgt = d["target_id"]
                if tgt in self.graph.components:
                    failed_nodes.add(tgt)
                    active_components[tgt] = False
                    comp_telemetry[tgt].is_failed = True
                    self.graph.fail_component(tgt)

                    comp_type = self.graph.components[tgt].type
                    # Physical cascade: Node down -> all hosted apps and brokers down
                    if comp_type == "Node":
                        if tgt in node_telemetry:
                            node_telemetry[tgt].is_failed = True
                        for resident in self.graph.get_hosted_components(tgt):
                            failed_nodes.add(resident)
                            active_components[resident] = False
                            if resident in comp_telemetry:
                                comp_telemetry[resident].is_failed = True
                            self.graph.fail_component(resident)

                    # Library cascade: Library down -> all apps using it fail
                    elif comp_type == "Library":
                        for consumer in self.graph.get_uses_consumers(tgt):
                            failed_nodes.add(consumer)
                            active_components[consumer] = False
                            if consumer in comp_telemetry:
                                comp_telemetry[consumer].is_failed = True
                            self.graph.fail_component(consumer)

            # 2. PUBLISH
            elif ev_type == "PUBLISH":
                pub_id = d["publisher_id"]
                tid = d["topic_id"]
                configured_subs = sorted([e[0] for e in self.graph._subscribers.get(tid, [])])
                n_subs = len(configured_subs)

                total_msgs_generated += 1
                if fault_time is not None:
                    if sim_time < fault_time:
                        pre_expected += n_subs
                    else:
                        post_expected += n_subs
                else:
                    pre_expected += n_subs

                if pub_id in comp_telemetry:
                    comp_telemetry[pub_id].messages_sent += 1
                if tid in topic_telemetry:
                    topic_telemetry[tid].published_count += 1

                # If publisher is dead, message never leaves
                if not active_components.get(pub_id, True):
                    total_msgs_dropped += n_subs
                    if pub_id in comp_telemetry:
                        comp_telemetry[pub_id].messages_dropped += n_subs
                    continue

                # Record Node telemetry (NIC outbound)
                host_node = self.graph._hosted_on.get(pub_id)
                if host_node and host_node in node_telemetry:
                    node_telemetry[host_node].messages_out += 1
                    node_telemetry[host_node].bandwidth_bps_out += d["size_bytes"]

                # Route message to subscribers
                has_brokers = self.graph.has_configured_brokers(tid)
                live_brokers = sorted(self.graph.get_routing_brokers(tid))

                if has_brokers:
                    if not live_brokers:
                        # All brokers configured for this topic are failed!
                        topic_telemetry[tid].dropped_no_route += n_subs
                        total_msgs_dropped += n_subs
                        continue

                    for b in live_brokers:
                        push_event(
                            sim_time + sc.default_network_latency_s,
                            "ROUTE",
                            {
                                "broker_id": b,
                                "topic_id": tid,
                                "msg_id": d["msg_id"],
                                "created_at": d["created_at"],
                                "size_bytes": d["size_bytes"],
                                "qos_reliability": d["qos_reliability"],
                                "deadline_ms": d["deadline_ms"],
                            },
                        )
                else:
                    # Genuinely brokerless DDS direct delivery
                    if not configured_subs:
                        topic_telemetry[tid].dropped_no_route += 1
                        total_msgs_dropped += 1
                        continue

                    for s in configured_subs:
                        transit_time = sim_time + sc.default_network_latency_s
                        push_event(
                            transit_time,
                            "DELIVER",
                            {
                                "msg_id": d["msg_id"],
                                "topic_id": tid,
                                "subscriber_id": s,
                                "created_at": d["created_at"],
                                "size_bytes": d["size_bytes"],
                                "qos_reliability": d["qos_reliability"],
                                "deadline_ms": d["deadline_ms"],
                            },
                        )

            # 3. ROUTE
            elif ev_type == "ROUTE":
                bid = d["broker_id"]
                tid = d["topic_id"]
                configured_subs = sorted([e[0] for e in self.graph._subscribers.get(tid, [])])
                n_subs = len(configured_subs)

                if not active_components.get(bid, True):
                    # Broker is dead
                    topic_telemetry[tid].dropped_no_route += n_subs
                    total_msgs_dropped += n_subs
                    continue

                if bid in comp_telemetry:
                    comp_telemetry[bid].messages_routed += 1

                bq = broker_queues.get(bid)
                msg_obj = _SimMessage(
                    msg_id=d["msg_id"],
                    topic_id=tid,
                    publisher_id="",
                    created_at=d["created_at"],
                    payload_size_bytes=d["size_bytes"],
                    qos_reliability=d["qos_reliability"],
                )
                if bq and not bq.put(msg_obj):
                    # Broker queue overflow drop
                    topic_telemetry[tid].dropped_queue_full += n_subs
                    total_msgs_dropped += n_subs
                    continue

                # Forward from broker to subscribers
                for s in configured_subs:
                    push_event(
                        sim_time + sc.default_network_latency_s,
                        "DELIVER",
                        {
                            "msg_id": d["msg_id"],
                            "topic_id": tid,
                            "subscriber_id": s,
                            "created_at": d["created_at"],
                            "size_bytes": d["size_bytes"],
                            "qos_reliability": d["qos_reliability"],
                            "deadline_ms": d["deadline_ms"],
                        },
                    )

            # 4. DELIVER
            elif ev_type == "DELIVER":
                sid = d["subscriber_id"]
                tid = d["topic_id"]

                if not active_components.get(sid, True):
                    # Subscriber is dead
                    if tid in topic_telemetry:
                        topic_telemetry[tid].dropped_no_route += 1
                    total_msgs_dropped += 1
                    continue

                sq = subscriber_queues.get((tid, sid))
                msg_obj = _SimMessage(
                    msg_id=d["msg_id"],
                    topic_id=tid,
                    publisher_id="",
                    created_at=d["created_at"],
                    payload_size_bytes=d["size_bytes"],
                    qos_reliability=d["qos_reliability"],
                    deadline_ms=d["deadline_ms"],
                )

                if sq and not sq.put(msg_obj):
                    if tid in topic_telemetry:
                        topic_telemetry[tid].dropped_queue_full += 1
                    if sid in comp_telemetry:
                        comp_telemetry[sid].messages_dropped += 1
                    total_msgs_dropped += 1
                    qos_violations.append(
                        QoSViolationEvent(
                            time=sim_time,
                            message_id=d["msg_id"],
                            topic_id=tid,
                            subscriber_id=sid,
                            violation_type="queue_overflow",
                            latency_ms=(sim_time - d["created_at"]) * 1000.0,
                        )
                    )
                    continue

                # Schedule subscriber processing
                push_event(
                    sim_time + sc.default_processing_time_s,
                    "PROCESS",
                    {
                        "msg_id": d["msg_id"],
                        "topic_id": tid,
                        "subscriber_id": sid,
                        "created_at": d["created_at"],
                        "size_bytes": d["size_bytes"],
                        "deadline_ms": d["deadline_ms"],
                    },
                )

            # 5. PROCESS
            elif ev_type == "PROCESS":
                sid = d["subscriber_id"]
                tid = d["topic_id"]

                if not active_components.get(sid, True):
                    total_msgs_dropped += 1
                    continue

                # Ignore duplicate delivery if multiple redundant brokers delivered this message
                if sid in delivered_subscribers[d["msg_id"]]:
                    continue
                delivered_subscribers[d["msg_id"]].add(sid)

                e2e_latency_ms = (sim_time - d["created_at"]) * 1000.0

                # Deadline enforcement
                deadline = d.get("deadline_ms")
                if deadline is not None and e2e_latency_ms > deadline:
                    if tid in topic_telemetry:
                        topic_telemetry[tid].dropped_deadline += 1
                    total_msgs_dropped += 1
                    qos_violations.append(
                        QoSViolationEvent(
                            time=sim_time,
                            message_id=d["msg_id"],
                            topic_id=tid,
                            subscriber_id=sid,
                            violation_type="deadline",
                            latency_ms=e2e_latency_ms,
                            threshold_ms=deadline,
                        )
                    )
                    continue

                # Successful delivery!
                total_msgs_delivered += 1
                if sid in comp_telemetry:
                    comp_telemetry[sid].messages_received += 1
                    comp_telemetry[sid].processing_time_total += sc.default_processing_time_s

                if tid in topic_telemetry:
                    topic_telemetry[tid].delivered_count += 1

                # Record Node inbound telemetry
                host_node = self.graph._hosted_on.get(sid)
                if host_node and host_node in node_telemetry:
                    node_telemetry[host_node].messages_in += 1
                    node_telemetry[host_node].bandwidth_bps_in += d["size_bytes"]

                latency_samples_all.append(e2e_latency_ms)
                if fault_time is not None:
                    if sim_time < fault_time:
                        pre_delivered += 1
                        latency_samples_pre.append(e2e_latency_ms)
                    else:
                        post_delivered += 1
                        latency_samples_post.append(e2e_latency_ms)

        # ── Finalize Telemetry Aggregates ─────────────────────────────────────
        total_expected = pre_expected + post_expected
        denom = max(1, total_expected) if total_expected > 0 else max(1, total_msgs_generated)
        sys_delivery = min(1.0, total_msgs_delivered / denom)
        sys_drop = min(1.0, total_msgs_dropped / denom)

        pre_rate = min(1.0, pre_delivered / max(1, pre_expected)) if pre_expected else None
        post_rate = min(1.0, post_delivered / max(1, post_expected)) if post_expected else None

        pre_p95 = percentile(latency_samples_pre, 95) if latency_samples_pre else None
        post_p95 = percentile(latency_samples_post, 95) if latency_samples_post else None

        for tid, t in topic_telemetry.items():
            t.delivery_rate = min(1.0, t.delivered_count / max(1, t.published_count))

        return SystemTelemetry(
            graph_id=getattr(self.graph, "id", "") or "system_graph",
            simulation_duration=sc.duration,
            seed=sc.seed,
            total_messages_generated=total_msgs_generated,
            total_messages_delivered=total_msgs_delivered,
            total_messages_dropped=total_msgs_dropped,
            system_delivery_rate=sys_delivery,
            system_drop_rate=sys_drop,
            system_latency_p50_ms=percentile(latency_samples_all, 50),
            system_latency_p95_ms=percentile(latency_samples_all, 95),
            system_latency_p99_ms=percentile(latency_samples_all, 99),
            system_latency_mean_ms=sum(latency_samples_all) / len(latency_samples_all) if latency_samples_all else 0.0,
            component_metrics=comp_telemetry,
            topic_metrics=topic_telemetry,
            node_metrics=node_telemetry,
            qos_violations=qos_violations,
            starvation_events=starvation_events,
            faulted_nodes=failed_nodes,
            fault_time=fault_time,
            pre_fault_delivery_rate=pre_rate,
            post_fault_delivery_rate=post_rate,
            pre_fault_p95_latency_ms=pre_p95,
            post_fault_p95_latency_ms=post_p95,
        )

    def run_baseline(self, duration: float = 60.0) -> SystemTelemetry:
        """Run pristine baseline without any fault injection."""
        sc = TelemetryScenario(
            duration=duration,
            fault_node=None,
            fault_time=None,
            seed=self.scenario.seed,
        )
        return self.simulate(sc)

    def run_fault(
        self,
        target_id: str,
        duration: float = 100.0,
        fault_time: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> SystemTelemetry:
        """Run simulation with fault injected into target_id at fault_time."""
        sc = TelemetryScenario(
            duration=duration,
            fault_node=target_id,
            fault_time=fault_time if fault_time is not None else duration / 2.0,
            seed=seed if seed is not None else self.scenario.seed,
            propagation_threshold=self.scenario.propagation_threshold,
        )
        return self.simulate(sc)

    def sweep_all_components(
        self,
        node_types: Optional[List[str]] = None,
        candidate_ids: Optional[List[str]] = None,
        duration: float = 60.0,
        seeds: Optional[List[int]] = None,
    ) -> FaultInjectionResult:
        """
        Execute systematic single-node fault injection sweep across candidate nodes,
        evaluating telemetry deltas and producing a training/validation FaultInjectionResult.
        """
        seeds = seeds or [42]
        if candidate_ids:
            candidates = [cid for cid in candidate_ids if cid in self.graph.components]
        else:
            types = node_types or ["Application", "Broker", "Node", "Library"]
            candidates = [
                cid for cid, c in self.graph.components.items()
                if c.type in types
            ]

        # 1. Run baseline
        baseline = self.run_baseline(duration=duration / 2.0)

        records: Dict[str, FaultInjectionRecord] = {}
        fault_time = duration / 2.0

        for target_id in candidates:
            target_comp = self.graph.components[target_id]
            seed_scores: Dict[int, float] = {}
            primary_impact_data = None

            for s in seeds:
                degraded = self.run_fault(
                    target_id=target_id,
                    duration=duration,
                    fault_time=fault_time,
                    seed=s,
                )
                imp_data = self.impact_calculator.compute_node_impact(
                    pre_telemetry=baseline,
                    post_telemetry=degraded,
                    target_id=target_id,
                    target_type=target_comp.type,
                    target_name=target_comp.properties.get("name", target_id),
                )
                seed_scores[s] = imp_data["impact_score"]
                if primary_impact_data is None:
                    primary_impact_data = imp_data

            rec = self.impact_calculator.to_fault_injection_record(
                impact_data=primary_impact_data,
                seed_scores=seed_scores,
            )
            records[target_id] = rec

        unlabeled = sorted(set(self.graph.components.keys()) - set(candidates))

        return self.impact_calculator.to_fault_injection_result(
            graph_id=getattr(self.graph, "id", "") or "system_graph",
            records=records,
            seeds=seeds,
            unlabeled_node_ids=unlabeled,
            labeled_node_types=sorted(node_types or ["Application", "Broker", "Node", "Library"]),
        )
