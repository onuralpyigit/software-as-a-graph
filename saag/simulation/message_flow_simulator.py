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
from typing import Any, Dict, Generator, List, Optional, Set

try:
    import simpy  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SimPy is required for message-flow simulation.  "
        "Install it with:  pip install simpy"
    ) from exc

import networkx as nx

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
) -> Generator:
    """
    Publisher SimPy process.

    Emits one message every 1/rate_hz simulated seconds.  Stops silently
    when app_id is in failed_nodes.  Tracks pre/post-fault publish counts
    for accurate delivery-rate estimation.
    """
    interval = 1.0 / rate_hz if rate_hz > 0 else 1.0

    while True:
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

    def run(self) -> MessageFlowResult:
        """Execute the simulation and return a MessageFlowResult."""
        rng = random.Random(self.seed)
        env = simpy.Environment()
        failed_nodes: Set[str] = set()

        # BUG-MFS-6 FIX: instance-level counter via single-element list
        msg_counter: List[int] = [0]

        # ── Topic QoS and stats ───────────────────────────────────────────

        topic_qos: Dict[str, QoSProfile] = {}
        topic_stats: Dict[str, TopicFlowStats] = {}

        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "Topic":
                qos = _extract_qos(data, self.default_queue_size)
                topic_qos[node] = qos
                topic_stats[node] = TopicFlowStats(
                    topic_id=node,
                    topic_name=data.get("name", node),
                    reliability_policy=qos.reliability,
                    deadline_ms=qos.deadline_ms,
                    durability_policy=qos.durability,
                )

        # ── Fan-out objects (one per topic) ───────────────────────────────

        fanouts: Dict[str, TopicFanout] = {
            tid: TopicFanout(tid, topic_qos[tid], topic_stats[tid])
            for tid in topic_qos
        }

        # ── Subscriber stats and receive queues ───────────────────────────

        sub_stats: Dict[str, SubscriberFlowStats] = {}
        sub_topics: Dict[str, List[str]] = defaultdict(list)
        # (topic_id, sub_id) → SubscriberQueue
        sub_queues: Dict[tuple, SubscriberQueue] = {}

        for src, tgt, data in self.graph.edges(data=True):
            if data.get("type") == "SUBSCRIBES_TO" and tgt in fanouts:
                sub_topics[src].append(tgt)

        for sub_id, topics in sub_topics.items():
            sub_stats[sub_id] = SubscriberFlowStats(
                subscriber_id=sub_id, subscribed_topics=topics
            )

        for src, tgt, data in self.graph.edges(data=True):
            if data.get("type") == "SUBSCRIBES_TO" and tgt in fanouts:
                # Merge subscriber-side and topic-level QoS
                sub_qos = _extract_qos(data, self.default_queue_size)
                topic_level_qos = topic_qos[tgt]
                # Topic-level deadline takes precedence if set
                if topic_level_qos.deadline_ms:
                    sub_qos.deadline_ms = topic_level_qos.deadline_ms

                sq = fanouts[tgt].register(env, src)
                sub_queues[(tgt, src)] = sq

                if src not in sub_stats:
                    sub_stats[src] = SubscriberFlowStats(
                        subscriber_id=src, subscribed_topics=[tgt]
                    )

        # ── Node processing times ─────────────────────────────────────────

        proc_time: Dict[str, float] = {}
        for node, data in self.graph.nodes(data=True):
            pt = data.get("processing_time", self.default_processing_time_s)
            try:
                proc_time[node] = float(pt)
            except (TypeError, ValueError):
                proc_time[node] = self.default_processing_time_s

        # ── BUG-MFS-2 FIX: per-topic window publish/delivery counters ─────

        pub_window: Dict[str, Dict[str, int]] = {
            tid: {"pre": 0, "post": 0} for tid in fanouts
        }
        del_window: Dict[str, Dict[str, int]] = {
            tid: {"pre": 0, "post": 0} for tid in fanouts
        }

        # ── Spawn publisher processes ─────────────────────────────────────

        for src, tgt, data in self.graph.edges(data=True):
            if data.get("type") == "PUBLISHES_TO" and tgt in fanouts:
                rate_raw = data.get("rate_hz") or data.get("publish_rate_hz")
                try:
                    rate = float(rate_raw) if rate_raw else self.default_publish_rate_hz
                except (TypeError, ValueError):
                    rate = self.default_publish_rate_hz

                env.process(
                    _publisher_process(
                        env=env,
                        app_id=src,
                        topic_id=tgt,
                        rate_hz=rate,
                        fanout=fanouts[tgt],
                        failed_nodes=failed_nodes,
                        fault_time=self.fault_time if self.fault_node else None,
                        window_counts=pub_window[tgt],
                        msg_counter=msg_counter,
                        rng=rng,
                        processing_time_s=proc_time.get(src, self.default_processing_time_s),
                    )
                )

        # ── Spawn subscriber processes ────────────────────────────────────

        for src, tgt, data in self.graph.edges(data=True):
            if data.get("type") == "SUBSCRIBES_TO" and tgt in fanouts:
                sq = sub_queues.get((tgt, src))
                if sq is None:
                    continue

                sub_qos = _extract_qos(data, self.default_queue_size)
                if topic_qos[tgt].deadline_ms:
                    sub_qos.deadline_ms = topic_qos[tgt].deadline_ms

                env.process(
                    _subscriber_process(
                        env=env,
                        app_id=src,
                        topic_id=tgt,
                        sq=sq,
                        qos=sub_qos,
                        failed_nodes=failed_nodes,
                        fault_time=self.fault_time if self.fault_node else None,
                        sub_stats=sub_stats[src],
                        topic_stats=topic_stats[tgt],
                        delivery_window_counts=del_window[tgt],
                        rng=rng,
                        max_latency_samples=self.max_latency_samples,
                        processing_time_s=proc_time.get(src, self.default_processing_time_s),
                    )
                )

        # ── Fault injection process ───────────────────────────────────────

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

        # ── Run simulation ────────────────────────────────────────────────

        logger.info(
            "Message-flow sim: duration=%.1fs | fault=%s | seed=%d",
            self.duration, self.fault_node or "none", self.seed,
        )
        env.run(until=self.duration)
        logger.info("Simulation complete.")

        # ── Aggregate results ─────────────────────────────────────────────

        total_published = sum(ts.total_published for ts in topic_stats.values())
        total_delivered = sum(ts.total_delivered for ts in topic_stats.values())
        total_deadline_viol = sum(ts.total_dropped_deadline for ts in topic_stats.values())
        total_overflow = sum(ts.total_dropped_queue_full for ts in topic_stats.values())

        # Delivery rate: total_delivered / (total_published * avg_subscriber_count)
        # A message is "fully delivered" when every subscriber receives it.
        # We normalise against published×fan-out to get a per-copy rate.
        total_expected = sum(
            ts.total_published * max(1, len(fanouts[tid].subscriber_ids))
            for tid, ts in topic_stats.items()
        )
        system_delivery = total_delivered / total_expected if total_expected else 0.0

        # ── Fault cascade annotation ──────────────────────────────────────

        if fault_event_record is not None:
            # BUG-MFS-3 FIX: check for other live publishers before marking
            # a topic as orphaned.
            other_pubs: Dict[str, Set[str]] = defaultdict(set)
            for s2, t2, d2 in self.graph.edges(data=True):
                if d2.get("type") == "PUBLISHES_TO":
                    other_pubs[t2].add(s2)

            orphaned: List[str] = []
            for s2, t2, d2 in self.graph.edges(data=True):
                if d2.get("type") == "PUBLISHES_TO" and s2 == self.fault_node:
                    remaining = other_pubs[t2] - {self.fault_node}
                    if not remaining:
                        orphaned.append(t2)

            impacted: List[str] = []
            for t2 in orphaned:
                for s2, tgt2, d2 in self.graph.edges(data=True):
                    if d2.get("type") == "SUBSCRIBES_TO" and tgt2 == t2:
                        impacted.append(s2)

            # BUG-MFS-2 FIX: accurate before/after rates from window counters
            total_pre_pub = sum(pw["pre"] for pw in pub_window.values())
            total_post_pub = sum(pw["post"] for pw in pub_window.values())
            total_pre_del = sum(dw["pre"] for dw in del_window.values())
            total_post_del = sum(dw["post"] for dw in del_window.values())

            # Normalise against fan-out (each pub fans to N subscribers)
            pre_expected = sum(
                pub_window[tid]["pre"] * max(1, len(fanouts[tid].subscriber_ids))
                for tid in fanouts
            )
            post_expected = sum(
                pub_window[tid]["post"] * max(1, len(fanouts[tid].subscriber_ids))
                for tid in fanouts
            )

            fault_event_record.cascade_silenced_publishers = [self.fault_node]
            fault_event_record.cascade_orphaned_topics = sorted(set(orphaned))
            fault_event_record.cascade_impacted_subscribers = sorted(set(impacted))
            fault_event_record.delivery_rate_before = min(
                1.0, total_pre_del / pre_expected if pre_expected else 0.0
            )
            fault_event_record.delivery_rate_after = min(
                1.0, total_post_del / post_expected if post_expected else 0.0
            )

        result = MessageFlowResult(
            graph_id=self.graph.graph.get("id", ""),
            simulation_duration=self.duration,
            seed=self.seed,
            fault_event=fault_event_record,
            system_delivery_rate=round(min(1.0, system_delivery), 4),
            system_drop_rate=round(max(0.0, 1.0 - system_delivery), 4),
            total_messages_published=total_published,
            total_messages_delivered=total_delivered,
            total_deadline_violations=total_deadline_viol,
            total_queue_overflows=total_overflow,
            topic_stats=topic_stats,
            subscriber_stats=sub_stats,
        )
        return result
