"""
message_flow_simulator.py
─────────────────────────
Discrete-event pub-sub message-flow simulator for the SaG pipeline.

Built on SimPy (https://simpy.readthedocs.io/).

PURPOSE
───────
Where the FaultInjector works on pure topology, this simulator runs the
system forward in time, modelling:

  • Publisher processes   – emit at publish_rate Hz per topic
  • Fan-out queues        – correct pub-sub semantics: each subscriber gets
                           its own receive queue; the publisher fans out to
                           all of them  [FIX: BUG-MFS-1]
  • Subscriber processes  – pull from their own queue; failure check happens
                           BEFORE get() to avoid put-back  [FIX: BUG-MFS-4]
  • QoS enforcement       – RELIABLE / BEST_EFFORT; deadline_ms checked on
                           end-to-end latency (after subscriber processing)
                           [FIX: BUG-MFS-5]
  • Fault injection       – node added to failed_nodes at fault_time; cascade
                           info annotated using graph topology
  • Pre/post-fault rates  – per-topic published counts tracked in publisher to
                           give accurate before/after delivery rates
                           [FIX: BUG-MFS-2]

FIXES IN THIS VERSION
─────────────────────
  BUG-MFS-1  Fan-out: one receive queue per (topic, subscriber) pair; publisher
             fans out to all live subscriber queues.
  BUG-MFS-2  Before/after delivery rates use per-window published counts
             tracked by publisher processes; rate is always in [0, 1].
  BUG-MFS-3  Orphaned topic list checks other live publishers before marking
             a topic as orphaned.
  BUG-MFS-4  Subscriber checks failed_nodes before issuing get(); no message
             put-back needed.
  BUG-MFS-5  Latency is measured end-to-end (after subscriber processing
             delay); deadline check uses end-to-end latency.
  BUG-MFS-6  Instance-level message counter instead of module global.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

try:
    import simpy  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SimPy is required for message-flow simulation.  "
        "Install it with:  pip install simpy"
    ) from exc

import networkx as nx

from ._stats import percentile

from .simulation_results import (
    FaultEventRecord,
    MessageFlowResult,
    SubscriberFlowStats,
    TopicFlowStats,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Message
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    msg_id: int
    topic_id: str
    publisher_id: str
    created_at: float            # simulated seconds
    payload_size_bytes: int = 64


# ─────────────────────────────────────────────────────────────────────────────
# QoS profile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QoSProfile:
    reliability: str = "RELIABLE"       # RELIABLE | BEST_EFFORT
    durability: str = "VOLATILE"        # VOLATILE | TRANSIENT_LOCAL
    deadline_ms: Optional[float] = None
    lifespan_ms: Optional[float] = None
    queue_size: int = 100
    history_depth: int = 10


def _extract_qos(data: Dict[str, Any], default_queue: int = 100) -> QoSProfile:
    qos_raw = data.get("qos_profile") or data.get("qos_policy") or {}
    if not isinstance(qos_raw, dict):
        qos_raw = {}
    return QoSProfile(
        reliability=str(qos_raw.get("reliability", "RELIABLE")).upper(),
        durability=str(qos_raw.get("durability", "VOLATILE")).upper(),
        deadline_ms=qos_raw.get("deadline_ms") or qos_raw.get("deadline"),
        lifespan_ms=qos_raw.get("lifespan_ms"),
        queue_size=int(qos_raw.get("queue_size", default_queue)),
        history_depth=int(qos_raw.get("history_depth", 10)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fan-out queue model
#
# BUG-MFS-1 FIX: correct pub-sub fan-out semantics.
#
# One SubscriberQueue per (topic_id, subscriber_id) pair.
# TopicFanout holds all subscriber queues for one topic and provides a
# single publish() call that fans the message out to every live subscriber.
# ─────────────────────────────────────────────────────────────────────────────

class SubscriberQueue:
    """Receive queue owned by one (topic, subscriber) pair."""

    def __init__(
        self,
        env: simpy.Environment,
        topic_id: str,
        subscriber_id: str,
        qos: QoSProfile,
    ) -> None:
        self.env = env
        self.topic_id = topic_id
        self.subscriber_id = subscriber_id
        self.qos = qos
        self._store: simpy.Store = simpy.Store(env, capacity=qos.queue_size)

    def get(self) -> "simpy.resources.store.StoreGet":
        return self._store.get()

    @property
    def depth(self) -> int:
        return len(self._store.items)

    def _try_put(self, msg: Message, stats: TopicFlowStats) -> bool:
        """Enqueue msg; apply overflow policy; return True if enqueued."""
        if self.depth >= self.qos.queue_size:
            stats.total_dropped_queue_full += 1
            if self.qos.reliability == "BEST_EFFORT":
                stats.total_dropped_best_effort += 1
                return False
            # RELIABLE: head-drop oldest to make room
            if self._store.items:
                self._store.items.pop(0)
        self._store.put(msg)
        return True


class TopicFanout:
    """
    Manages fan-out from one publisher topic to all registered subscriber
    queues.

    The publisher calls publish(msg, failed_nodes) once per message.
    Each live subscriber's queue receives a copy.
    """

    def __init__(self, topic_id: str, qos: QoSProfile, stats: TopicFlowStats) -> None:
        self.topic_id = topic_id
        self.qos = qos
        self.stats = stats
        # subscriber_id → SubscriberQueue
        self._queues: Dict[str, SubscriberQueue] = {}

    def register(self, env: simpy.Environment, subscriber_id: str) -> SubscriberQueue:
        """Create and register a per-subscriber receive queue."""
        sq = SubscriberQueue(env, self.topic_id, subscriber_id, self.qos)
        self._queues[subscriber_id] = sq
        return sq

    def queue_for(self, subscriber_id: str) -> Optional[SubscriberQueue]:
        return self._queues.get(subscriber_id)

    def publish(self, msg: Message, failed_nodes: Set[str]) -> int:
        """
        Fan out *msg* to all live subscriber queues.

        Returns the number of queues the message was placed into.
        Increments stats.total_published once regardless of fan-out width.
        """
        n_queued = 0
        for sub_id, sq in self._queues.items():
            if sub_id in failed_nodes:
                continue
            if sq._try_put(msg, self.stats):
                n_queued += 1
        if n_queued > 0:
            self.stats.total_published += 1
        return n_queued

    @property
    def subscriber_ids(self) -> List[str]:
        return list(self._queues.keys())


# ─────────────────────────────────────────────────────────────────────────────
# SimPy process functions
# ─────────────────────────────────────────────────────────────────────────────

def _publisher_process(
    env: simpy.Environment,
    app_id: str,
    topic_id: str,
    rate_hz: float,
    fanout: TopicFanout,
    failed_nodes: Set[str],
    fault_time: Optional[float],
    # BUG-MFS-2 FIX: track published counts per time window
    window_counts: Dict[str, int],   # "pre" / "post" keys, mutated in-place
    msg_counter: List[int],          # single-element list used as a mutable int
    rng: random.Random,
    processing_time_s: float = 0.0,
    use_poisson: bool = False,
) -> Generator:
    """
    Publisher SimPy process.

    Emits one message periodically or stochastically (Poisson process) based on rate_hz.
    Stops silently when app_id is in failed_nodes.
    """
    while True:
        if use_poisson:
            interval = rng.expovariate(rate_hz) if rate_hz > 0 else 1.0
        else:
            interval = 1.0 / rate_hz if rate_hz > 0 else 1.0

        yield env.timeout(interval)


        if app_id in failed_nodes:
            return

        # Optional processing delay before publish
        if processing_time_s > 0:
            yield env.timeout(processing_time_s * (0.8 + 0.4 * rng.random()))

        msg_counter[0] += 1
        msg = Message(
            msg_id=msg_counter[0],
            topic_id=topic_id,
            publisher_id=app_id,
            created_at=env.now,
        )
        fanout.publish(msg, failed_nodes)

        # BUG-MFS-2 FIX: count publishes in the correct time window
        if fault_time is not None and env.now < fault_time:
            window_counts["pre"] += 1
        else:
            window_counts["post"] += 1




def _subscriber_process(
    env: simpy.Environment,
    app_id: str,
    topic_id: str,
    sq: SubscriberQueue,                # BUG-MFS-1 FIX: per-subscriber queue
    qos: QoSProfile,
    failed_nodes: Set[str],
    fault_time: Optional[float],
    sub_stats: SubscriberFlowStats,
    topic_stats: TopicFlowStats,
    # BUG-MFS-2 FIX: window-level delivery counters
    delivery_window_counts: Dict[str, int],  # "pre" / "post" keys
    rng: random.Random,
    max_latency_samples: int = 10_000,
    processing_time_s: float = 0.0,
    latency_windows: Optional[Dict[str, list]] = None,
) -> Generator:
    """
    Subscriber SimPy process.

    Pulls messages from its private receive queue, applies deadline /
    lifespan checks on end-to-end latency, and records statistics.

    BUG-MFS-4 FIX: failed_nodes check happens BEFORE get(), so no
    message needs to be put back into the queue.

    BUG-MFS-5 FIX: latency is measured AFTER subscriber processing delay
    to reflect true end-to-end delivery time.
    """
    received_key = topic_id
    sub_stats.received_per_topic.setdefault(received_key, 0)
    sub_stats.missed_per_topic.setdefault(received_key, 0)
    sub_stats.deadline_violations_per_topic.setdefault(received_key, 0)

    while True:
        # BUG-MFS-4 FIX: bail out BEFORE get() if subscriber has failed.
        # This avoids ever dequeuing a message only to discard it.
        if app_id in failed_nodes:
            return

        msg_event = sq.get()
        msg: Message = yield msg_event

        # Double-check: failure could have been injected while waiting in get()
        if app_id in failed_nodes:
            # Message is already dequeued and lost — count as missed
            sub_stats.missed_per_topic[received_key] += 1
            return

        enqueue_time = msg.created_at
        arrival_time = env.now   # time message left the queue

        # Optional subscriber-side processing delay (models application compute)
        if processing_time_s > 0:
            yield env.timeout(processing_time_s * (0.8 + 0.4 * rng.random()))

        # BUG-MFS-5 FIX: end-to-end latency includes subscriber processing time
        delivery_time = env.now
        e2e_latency_ms = (delivery_time - enqueue_time) * 1000.0

        # Lifespan check (message may have expired while queued)
        if qos.lifespan_ms is not None and e2e_latency_ms > qos.lifespan_ms:
            sub_stats.missed_per_topic[received_key] += 1
            continue

        # Deadline check (DDS deadline = end-to-end)
        if qos.deadline_ms is not None and e2e_latency_ms > qos.deadline_ms:
            sub_stats.deadline_violations_per_topic[received_key] += 1
            topic_stats.total_dropped_deadline += 1
            sub_stats.missed_per_topic[received_key] += 1
            continue

        # Delivered
        sub_stats.received_per_topic[received_key] += 1
        topic_stats.total_delivered += 1

        # BUG-MFS-2 FIX: count deliveries in the correct time window
        if fault_time is not None and arrival_time < fault_time:
            delivery_window_counts["pre"] += 1
        else:
            delivery_window_counts["post"] += 1

        # Post-fault tracking on sub_stats
        if fault_time is not None and arrival_time >= fault_time:
            sub_stats.received_post_fault += 1

        # Latency sample
        if len(topic_stats.latency_samples) < max_latency_samples:
            topic_stats.latency_samples.append(e2e_latency_ms)
        if latency_windows is not None and fault_time is not None:
            bucket = "pre" if arrival_time < fault_time else "post"
            latency_windows[bucket].append(e2e_latency_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Main simulator class
# ─────────────────────────────────────────────────────────────────────────────

class MessageFlowSimulator:
    """
    Discrete-event pub-sub message flow simulator.

    Parameters
    ----------
    graph : nx.DiGraph
        SaG graph (exported by GraphExporter).
    duration : float
        Simulation duration in simulated seconds.
    fault_node : str, optional
        Node ID to fail at fault_time.
    fault_time : float, optional
        When to inject the fault.  Default: duration / 2.
    seed : int
        Random seed.
    default_queue_size : int
        Fallback per-(topic,subscriber) queue capacity.
    default_publish_rate_hz : float
        Fallback publish rate when not in graph metadata.
    default_processing_time_s : float
        Fallback per-component processing latency in seconds.
    max_latency_samples : int
        Max latency samples stored per topic (memory guard).
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        duration: float = 100.0,
        fault_node: Optional[str] = None,
        fault_time: Optional[float] = None,
        seed: int = 42,
        default_queue_size: int = 100,
        default_publish_rate_hz: float = 10.0,
        default_processing_time_s: float = 0.001,
        max_latency_samples: int = 10_000,
    ) -> None:
        self.graph = graph
        self.duration = duration
        self.fault_node = fault_node
        self.fault_time = fault_time if fault_time is not None else duration / 2.0
        self.seed = seed
        self.default_queue_size = default_queue_size
        self.default_publish_rate_hz = default_publish_rate_hz
        self.default_processing_time_s = default_processing_time_s
        self.max_latency_samples = max_latency_samples

    # ── Public API ──────────────────────────────────────────────────────────

    def generate_workload(self, topic_id: str) -> float:
        """Resolve the publish rate (frequency) for a given topic ID from the graph node's attributes.
        Honors topic.frequency / topic_frequency as the Poisson/periodic rate.
        If there are multiple publishers for this topic, the rate is divided equally
        among them to ensure the aggregate topic traffic matches the frequency.
        Falls back to default_publish_rate_hz.
        """
        base_rate = self.default_publish_rate_hz
        if topic_id in self.graph.nodes:
            topic_node = self.graph.nodes[topic_id]
            freq = topic_node.get("frequency", topic_node.get("topic_frequency"))
            if freq is not None:
                try:
                    base_rate = float(freq)
                except (TypeError, ValueError):
                    pass

        # Count the number of active publishers publishing to this topic
        num_pubs = 0
        for src, tgt, data in self.graph.edges(data=True):
            if data.get("type") == "PUBLISHES_TO" and tgt == topic_id:
                num_pubs += 1

        if num_pubs > 0:
            return base_rate / num_pubs
        return base_rate

    # ── Setup helpers ───────────────────────────────────────────────────────

    def _edges_by_type(self) -> Dict[str, List[Tuple[str, str, dict]]]:
        """
        Bucket the graph's edges by relationship type in a single pass.

        Ordering within each bucket is the graph's own edge order, which the
        callers rely on: SimPy resolves same-timestamp events by process
        creation order, so the sequence in which publisher and subscriber
        processes are spawned is part of the simulation's determinism.
        """
        buckets: Dict[str, List[Tuple[str, str, dict]]] = defaultdict(list)
        for src, tgt, data in self.graph.edges(data=True):
            buckets[data.get("type")].append((src, tgt, data))
        return buckets

    def _build_topics(self) -> Tuple[Dict[str, QoSProfile], Dict[str, TopicFlowStats]]:
        """Resolve every Topic node's QoS profile and seed its stats record."""
        topic_qos: Dict[str, QoSProfile] = {}
        topic_stats: Dict[str, TopicFlowStats] = {}

        for node, data in self.graph.nodes(data=True):
            if data.get("type") != "Topic":
                continue
            qos = _extract_qos(data, self.default_queue_size)
            topic_qos[node] = qos
            topic_stats[node] = TopicFlowStats(
                topic_id=node,
                topic_name=data.get("name", node),
                reliability_policy=qos.reliability,
                deadline_ms=qos.deadline_ms,
                durability_policy=qos.durability,
            )
        return topic_qos, topic_stats

    def _build_subscribers(
        self,
        env: simpy.Environment,
        sub_edges: List[Tuple[str, str, dict]],
        fanouts: Dict[str, TopicFanout],
    ) -> Tuple[Dict[str, SubscriberFlowStats], Dict[Tuple[str, str], SubscriberQueue]]:
        """Register one queue per (topic, subscriber) pair and seed subscriber stats."""
        sub_topics: Dict[str, List[str]] = defaultdict(list)
        for src, tgt, _ in sub_edges:
            sub_topics[src].append(tgt)

        sub_stats = {
            sub_id: SubscriberFlowStats(subscriber_id=sub_id, subscribed_topics=topics)
            for sub_id, topics in sub_topics.items()
        }

        sub_queues: Dict[Tuple[str, str], SubscriberQueue] = {}
        for src, tgt, _ in sub_edges:
            sub_queues[(tgt, src)] = fanouts[tgt].register(env, src)

        return sub_stats, sub_queues

    def _subscriber_qos(self, edge_data: dict, topic_qos: QoSProfile) -> QoSProfile:
        """Subscriber-side QoS, with the topic-level deadline taking precedence."""
        qos = _extract_qos(edge_data, self.default_queue_size)
        if topic_qos.deadline_ms:
            qos.deadline_ms = topic_qos.deadline_ms
        return qos

    def _node_processing_times(self) -> Dict[str, float]:
        """Per-node processing time, falling back to the configured default."""
        proc_time: Dict[str, float] = {}
        for node, data in self.graph.nodes(data=True):
            try:
                proc_time[node] = float(
                    data.get("processing_time", self.default_processing_time_s))
            except (TypeError, ValueError):
                proc_time[node] = self.default_processing_time_s
        return proc_time

    def _annotate_fault_cascade(
        self,
        record: FaultEventRecord,
        edges: Dict[str, List[Tuple[str, str, dict]]],
        fanouts: Dict[str, TopicFanout],
        pub_window: Dict[str, Dict[str, int]],
        del_window: Dict[str, Dict[str, int]],
        latency_windows: Dict[str, list],
    ) -> None:
        """
        Fill in what the fault actually cost: which topics it orphaned, which
        subscribers lost a feed, and the delivery/latency shift across the
        fault boundary.
        """
        # A topic is only orphaned if the faulted node was its *last* publisher.
        publishers_of: Dict[str, Set[str]] = defaultdict(set)
        for src, tgt, _ in edges["PUBLISHES_TO"]:
            publishers_of[tgt].add(src)

        orphaned = sorted({
            tgt for src, tgt, _ in edges["PUBLISHES_TO"]
            if src == self.fault_node and not (publishers_of[tgt] - {self.fault_node})
        })
        impacted = sorted({
            src for src, tgt, _ in edges["SUBSCRIBES_TO"] if tgt in orphaned
        })

        # Delivery rates come from the windowed counters, normalised against
        # fan-out (one publish becomes N expected deliveries).
        def _rate(window: str) -> float:
            delivered = sum(dw[window] for dw in del_window.values())
            expected = sum(
                pub_window[tid][window] * max(1, len(fanouts[tid].subscriber_ids))
                for tid in fanouts
            )
            return min(1.0, delivered / expected if expected else 0.0)

        record.cascade_silenced_publishers = [self.fault_node]
        record.cascade_orphaned_topics = orphaned
        record.cascade_impacted_subscribers = impacted
        record.delivery_rate_before = _rate("pre")
        record.delivery_rate_after = _rate("post")
        record.latency_p50_before = percentile(latency_windows["pre"], 50)
        record.latency_p50_after = percentile(latency_windows["post"], 50)
        record.latency_p95_before = percentile(latency_windows["pre"], 95)
        record.latency_p95_after = percentile(latency_windows["post"], 95)

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> MessageFlowResult:
        """Execute the simulation and return a MessageFlowResult."""
        rng = random.Random(self.seed)
        env = simpy.Environment()
        failed_nodes: Set[str] = set()
        msg_counter: List[int] = [0]   # shared, mutable message-id counter

        edges = self._edges_by_type()
        topic_qos, topic_stats = self._build_topics()
        fanouts = {
            tid: TopicFanout(tid, topic_qos[tid], topic_stats[tid]) for tid in topic_qos
        }

        pub_edges = [e for e in edges["PUBLISHES_TO"] if e[1] in fanouts]
        sub_edges = [e for e in edges["SUBSCRIBES_TO"] if e[1] in fanouts]

        sub_stats, sub_queues = self._build_subscribers(env, sub_edges, fanouts)
        proc_time = self._node_processing_times()

        # Per-topic publish/delivery counters, split on the fault boundary.
        pub_window = {tid: {"pre": 0, "post": 0} for tid in fanouts}
        del_window = {tid: {"pre": 0, "post": 0} for tid in fanouts}
        latency_windows: Dict[str, list] = {"pre": [], "post": []}

        fault_time = self.fault_time if self.fault_node else None

        for src, tgt, _ in pub_edges:
            topic_node = self.graph.nodes[tgt] if tgt in self.graph.nodes else {}
            env.process(_publisher_process(
                env=env,
                app_id=src,
                topic_id=tgt,
                rate_hz=self.generate_workload(tgt),
                fanout=fanouts[tgt],
                failed_nodes=failed_nodes,
                fault_time=fault_time,
                window_counts=pub_window[tgt],
                msg_counter=msg_counter,
                rng=rng,
                processing_time_s=proc_time.get(src, self.default_processing_time_s),
                use_poisson=str(topic_node.get("workload_type", "")).lower() == "poisson",
            ))

        for src, tgt, data in sub_edges:
            sq = sub_queues.get((tgt, src))
            if sq is None:
                continue
            env.process(_subscriber_process(
                env=env,
                app_id=src,
                topic_id=tgt,
                sq=sq,
                qos=self._subscriber_qos(data, topic_qos[tgt]),
                failed_nodes=failed_nodes,
                fault_time=fault_time,
                sub_stats=sub_stats[src],
                topic_stats=topic_stats[tgt],
                delivery_window_counts=del_window[tgt],
                rng=rng,
                max_latency_samples=self.max_latency_samples,
                processing_time_s=proc_time.get(src, self.default_processing_time_s),
                latency_windows=latency_windows,
            ))

        fault_event_record: Optional[FaultEventRecord] = None

        if self.fault_node is not None:
            def _fault_process(env: simpy.Environment) -> Generator:
                nonlocal fault_event_record
                yield env.timeout(self.fault_time)
                node_type = self.graph.nodes.get(self.fault_node, {}).get("type", "Unknown")
                logger.info(
                    "  [t=%.1f] Injecting fault: %s (%s)",
                    env.now, self.fault_node, node_type,
                )
                failed_nodes.add(self.fault_node)
                fault_event_record = FaultEventRecord(
                    fault_time=env.now,
                    faulted_node_id=self.fault_node,
                    faulted_node_type=node_type,
                    cascade_silenced_publishers=[],
                    cascade_orphaned_topics=[],
                    cascade_impacted_subscribers=[],
                    delivery_rate_before=0.0,
                    delivery_rate_after=0.0,
                )

            env.process(_fault_process(env))

        logger.info(
            "Message-flow sim: duration=%.1fs | fault=%s | seed=%d",
            self.duration, self.fault_node or "none", self.seed,
        )
        env.run(until=self.duration)
        logger.info("Simulation complete.")

        total_delivered = sum(ts.total_delivered for ts in topic_stats.values())
        # A message is "fully delivered" once every subscriber receives it, so
        # normalise against published × fan-out to get a per-copy rate.
        total_expected = sum(
            ts.total_published * max(1, len(fanouts[tid].subscriber_ids))
            for tid, ts in topic_stats.items()
        )
        system_delivery = total_delivered / total_expected if total_expected else 0.0

        if fault_event_record is not None:
            self._annotate_fault_cascade(
                fault_event_record, edges, fanouts,
                pub_window, del_window, latency_windows,
            )

        return MessageFlowResult(
            graph_id=self.graph.graph.get("id", ""),
            simulation_duration=self.duration,
            seed=self.seed,
            fault_event=fault_event_record,
            system_delivery_rate=round(min(1.0, system_delivery), 4),
            system_drop_rate=round(max(0.0, 1.0 - system_delivery), 4),
            total_messages_published=sum(ts.total_published for ts in topic_stats.values()),
            total_messages_delivered=total_delivered,
            total_deadline_violations=sum(ts.total_dropped_deadline for ts in topic_stats.values()),
            total_queue_overflows=sum(ts.total_dropped_queue_full for ts in topic_stats.values()),
            topic_stats=topic_stats,
            subscriber_stats=sub_stats,
        )
