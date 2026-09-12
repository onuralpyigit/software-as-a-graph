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
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

try:
    import simpy  # type: ignore
except ImportError:  # pragma: no cover
    simpy = None  # type: ignore


def _require_simpy() -> None:
    if simpy is None:
        raise ImportError(
            "SimPy is required for message-flow simulation.  "
            "Install it with:  pip install simpy"
        )


import networkx as nx

from ._stats import percentile

from saag.core.models import QoSPolicy

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
    #: Which fault window this delivery belongs to, when that differs from
    #: `created_at`. Only durability replay sets it: a replayed sample carries a
    #: pre-fault `created_at` (its staleness is real and its deadline is judged
    #: against it) but the *recovery* it represents happened post-fault, filling
    #: a gap whose demand was counted post-fault. Bucketing it by `created_at`
    #: credited the recovery to the pre window on top of the original delivery,
    #: which double-counted there and drove I_dyn above 1.0.
    window_at: Optional[float] = None

    @property
    def window_time(self) -> float:
        return self.window_at if self.window_at is not None else self.created_at


# ─────────────────────────────────────────────────────────────────────────────
# QoS profile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QoSProfile:
    reliability: str = "RELIABLE"       # RELIABLE | BEST_EFFORT
    durability: str = "VOLATILE"        # VOLATILE | TRANSIENT_LOCAL | TRANSIENT | PERSISTENT
    deadline_ms: Optional[float] = None
    lifespan_ms: Optional[float] = None
    queue_size: int = 100
    history_depth: int = 10
    transport_priority: str = "MEDIUM"  # LOW | MEDIUM | HIGH | CRITICAL/URGENT/HIGHEST

    # ── Which fields the source actually declared ────────────────────────────
    # `QoSPolicy.from_node_attrs` substitutes DEFAULT_HISTORY_DEPTH (10) and
    # MEDIUM for absent keys, so a resolved profile cannot otherwise say whether
    # a value was written down or filled in. Two things need that distinction:
    # the `history_depth` -> queue-capacity rule (imposing KEEP_LAST(10) on a
    # topic that declared no history would silently apply the corpus's
    # second-tightest contract to every real-world topic that omits the field),
    # and the per-(topic, subscriber) profile merge, which must let an edge
    # override the topic only where the edge genuinely said something.
    reliability_declared: bool = False
    durability_declared: bool = False
    deadline_declared: bool = False
    history_depth_declared: bool = False
    priority_declared: bool = False
    queue_size_declared: bool = False


#: What each `qos_mode` switches on. Mirrors `FaultInjector.QOS_FACTOR_MODES`
#: so the two oracles' ablation switches are shaped alike.
#:
#: ``none``       calibrated load, every QoS policy neutralised. The QoS-off arm
#:                of the label ablation. Note that load stays ON: the arm must
#:                differ from the others in *policy*, not in how busy the system
#:                is, or the ablation confounds "QoS does nothing" with "nothing
#:                was ever contended".
#: ``contracts``  queue depth from history_depth, deadlines enforced.
#: ``recovery``   durability replay only (Stage 4).
#: ``full``       everything, plus priority-ordered service (Stage 3).
#: ``legacy``     the engine's historical QoS *policy*: flat queue size,
#:                deadlines on, FIFO, no replay, no calibration. Note it does
#:                not restore historical *defects* -- the Stage 0 accounting
#:                fixes and the co-publisher phase stagger apply in every mode,
#:                so `legacy` is the pre-QoS policy on a correct engine, not a
#:                bit-for-bit replay of old numbers.
QOS_MODES: Tuple[str, ...] = ("none", "contracts", "recovery", "full", "legacy")

#: Modes in which a declared history_depth becomes the reader queue capacity.
_CAPACITY_FROM_HISTORY: frozenset = frozenset({"contracts", "full"})
#: Modes in which deadline_ms / lifespan_ms are enforced on delivery.
_DEADLINES_ENFORCED: frozenset = frozenset({"contracts", "full", "legacy"})
#: Modes in which transport_priority orders service at the contended resource.
_PRIORITY_ORDERED: frozenset = frozenset({"full"})
#: Modes in which durability replay fires after a fault.
_REPLAY_ENABLED: frozenset = frozenset({"recovery", "full"})
#: Modes that honour `target_utilization`. `legacy` ignores it by definition.
_CALIBRATED_MODES: frozenset = frozenset({"none", "contracts", "recovery", "full"})

#: How service rates are derived from a target utilization.
#:
#: ``per_subscriber``  each subscriber gets its own service rate, so every one
#:                     sits at the same baseline rho whatever its native load.
#:                     This is what makes rho dimensionless and comparable
#:                     across scenarios whose arrival rates span 3-4 orders of
#:                     magnitude (financial_trading medians 700 Hz per
#:                     subscriber, iot_smart_city 2 Hz).
#: ``global``          one service rate per scenario, taken from the p90
#:                     subscriber. The sensitivity arm: it keeps the *level* of
#:                     load heterogeneous, at the cost of stressing some
#:                     scenarios and leaving others idle.
UTILIZATION_MODES: Tuple[str, ...] = ("per_subscriber", "global")

#: What a replayed sample's deadline is measured from.
#:
#: ``original``  the sample keeps its original `created_at`, so any declared
#:               deadline drops it on arrival. Physically correct — *durability
#:               recovers state, not timeliness* — and it means durability is
#:               inert on the 60-80% of corpus topics that declare a deadline
#:               and effective on the rest. That is a real finding about the
#:               limits of durability under a real-time contract, not a bug.
#: ``reset``     the sample is treated as a fresh state snapshot and its
#:               deadline runs from the replay instant. The sensitivity arm:
#:               the gap between the two is how much the conclusion depends on
#:               this modelling choice.
REPLAY_DEADLINE_POLICIES: Tuple[str, ...] = ("original", "reset")

#: Service-time distributions. See `ServiceStation.sample_service_time`.
SERVICE_DISTRIBUTIONS: Tuple[str, ...] = ("exponential", "uniform", "deterministic")


def _extract_qos(data: Dict[str, Any], default_queue: int = 100) -> QoSProfile:
    """Resolve a QoSProfile from Topic-node attributes or relationship metadata.

    Topic QoS reaches this engine in three shapes and none of them is wrong:
    ``qos_profile`` (written onto relationships by the repositories),
    ``qos_policy``, and a nested ``qos`` sub-dict — which is how raw topology
    JSON and the ``cli`` research loaders shape it, and which this function did
    not read.  Every generated scenario therefore resolved to the fallbacks
    below, so reliability and durability were never enforced on the research
    path.  ``QoSPolicy.from_node_attrs`` is the canonical resolver for the node
    shapes; it also accepts the flat ``qos_reliability`` / ``qos_durability``
    keys and canonicalises mixed-case values (the corpus mixes ``reliable`` and
    ``RELIABLE``).  See tests/test_qos_resolution.py.

    Its *defaults* are deliberately not adopted: ``QoSPolicy`` falls back to
    BEST_EFFORT/VOLATILE, whereas an undeclared queue has always behaved as
    RELIABLE here.  Changing that is a semantic decision about queue overflow
    policy, separate from reading the key, so the engine's own fallbacks stand.
    """
    qos_raw = data.get("qos_profile") or data.get("qos_policy") or data.get("qos") or {}
    if not isinstance(qos_raw, dict):
        qos_raw = {}

    declared = QoSPolicy.from_node_attrs({**data, "qos": qos_raw})
    has_reliability = bool(data.get("qos_reliability") or qos_raw.get("reliability"))
    has_durability = bool(data.get("qos_durability") or qos_raw.get("durability"))
    has_priority = bool(
        data.get("qos_transport_priority")
        or data.get("qos_priority")
        or qos_raw.get("transport_priority")
        or qos_raw.get("priority")
    )
    # `_first_present` rather than truthiness: a declared deadline or depth of 0
    # is a real (if pathological) declaration and must not read as absent.
    has_deadline = QoSPolicy._first_present(
        data.get("deadline_ms"), data.get("qos_deadline_ms"),
        qos_raw.get("deadline_ms"), qos_raw.get("deadline"),
    ) is not None
    has_history = QoSPolicy._first_present(
        data.get("history_depth"), data.get("qos_history_depth"),
        qos_raw.get("history_depth"),
    ) is not None

    # deadline_ms / history_depth need no separate lookup: from_node_attrs()
    # above already resolves both across the flat and nested shapes.
    raw_q_size = QoSPolicy._first_present(
        data.get("queue_size"), qos_raw.get("queue_size")
    )
    q_size = raw_q_size if raw_q_size is not None else default_queue

    return QoSProfile(
        reliability=declared.reliability if has_reliability else "RELIABLE",
        durability=declared.durability if has_durability else "VOLATILE",
        deadline_ms=declared.deadline_ms,
        lifespan_ms=qos_raw.get("lifespan_ms"),
        queue_size=int(q_size),
        history_depth=int(declared.history_depth),
        transport_priority=declared.transport_priority,
        reliability_declared=has_reliability,
        durability_declared=has_durability,
        deadline_declared=has_deadline,
        history_depth_declared=has_history,
        priority_declared=has_priority,
        queue_size_declared=raw_q_size is not None,
    )


class ServiceStation:
    """One subscriber's compute, shared across every topic it reads.

    This is the engine's only contended resource, and adding it is the whole
    point of the QoS work. `run()` spawns one `_subscriber_process` per
    SUBSCRIBES_TO edge, each with its own service delay, so a subscriber reading
    five topics had five independent servers and nothing in the simulation ever
    queued behind anything else. That is why utilization never exceeded ~0.2,
    why queue depth was not the binding lever, why `transport_priority` had
    nowhere to apply, and ultimately why I_dyn reduced to "which feeds
    disappeared" -- a re-measurement of the topological oracle.

    The private per-(topic, subscriber) queues stay: BUG-MFS-1 was head-of-line
    blocking *across topics in a shared queue*, and that does not come back. A
    blocked low-priority topic accumulates in its own bounded queue rather than
    stalling another topic's. What is now shared is the *server*, not the queue.
    """

    def __init__(
        self,
        env: simpy.Environment,
        subscriber_id: str,
        service_time_s: float,
        concurrency: int = 1,
        distribution: str = "exponential",
        calibrated: bool = True,
        measure_from: float = 0.0,
    ) -> None:
        _require_simpy()
        self.env = env
        self.subscriber_id = subscriber_id
        self.service_time_s = max(0.0, service_time_s)
        self.distribution = distribution
        self.calibrated = calibrated
        #: Busy time before this instant is not counted. The queue starts empty,
        #: so the transient is idle time that does not belong to the steady state
        #: being measured -- including it read ~0.08 low at rho = 0.9, where the
        #: transient is longest, and would have failed the V0 gate for a
        #: calibration that was in fact correct.
        self.measure_from = measure_from
        self.concurrency = max(1, concurrency)
        self.resource = simpy.PriorityResource(env, capacity=self.concurrency)
        #: Accumulated service time, for the realised-utilization check that
        #: tells you whether the calibration actually landed.
        self.busy_time = 0.0

    def sample_service_time(self, rng: random.Random) -> float:
        """Draw one service time.

        Exponential by default. The engine's historical `U(0.8, 1.2)` jitter is
        near-deterministic, and M/D/1 wait is half M/M/1, which pushes the
        region where deadlines start to bind past rho > 0.9 -- where steady
        state is slow to reach and the measurement is hypersensitive to warm-up.
        Exponential is also the more honest model of application compute.
        """
        if self.service_time_s <= 0.0:
            return 0.0
        if self.distribution == "exponential":
            return rng.expovariate(1.0 / self.service_time_s)
        if self.distribution == "uniform":
            return self.service_time_s * (0.8 + 0.4 * rng.random())
        return self.service_time_s

    def record_service(self, service_time: float) -> None:
        """Accumulate busy time, ignoring the pre-steady-state transient."""
        if self.env.now >= self.measure_from:
            self.busy_time += service_time

    def utilization(self, until: float) -> float:
        """Realised busy fraction of this station's capacity, post warm-up."""
        span = (until - self.measure_from) * self.concurrency
        return self.busy_time / span if span > 0 else 0.0


def _merge_qos(topic: QoSProfile, edge: QoSProfile) -> QoSProfile:
    """Effective contract for one (topic, subscriber) pair.

    The topic is the offered policy and the SUBSCRIBES_TO edge the requested
    one; the edge wins only on fields it actually declared. That condition is
    the whole point of the `*_declared` flags: `_extract_qos` on an edge with no
    QoS keys returns the engine's own fallbacks (RELIABLE / VOLATILE /
    `default_queue_size`), so merging on values alone would overwrite every
    topic's declared policy with defaults on the majority of edges, which is a
    louder bug than the one it fixes.

    Timing terms take the *stricter* of the two, which is the conservative
    reading of a contract pair: a reader asking for a tighter deadline than the
    writer offers does not get the writer's slack.

    On the research path `cli.loso_evaluate._project_topic_qos_onto_edges`
    copies the topic's profile onto its edges, so this merge is an identity
    there. It is not an identity on the repository path, which is why the
    edge-side value can no longer be parsed and dropped.
    """
    merged = QoSProfile(**{**topic.__dict__})

    if edge.reliability_declared:
        merged.reliability = edge.reliability
        merged.reliability_declared = True
    if edge.durability_declared:
        merged.durability = edge.durability
        merged.durability_declared = True
    if edge.priority_declared:
        merged.transport_priority = edge.transport_priority
        merged.priority_declared = True
    if edge.queue_size_declared:
        merged.queue_size = edge.queue_size
        merged.queue_size_declared = True

    if edge.deadline_declared and edge.deadline_ms is not None:
        merged.deadline_ms = (
            edge.deadline_ms if merged.deadline_ms is None
            else min(merged.deadline_ms, edge.deadline_ms)
        )
        merged.deadline_declared = True
    if edge.history_depth_declared:
        merged.history_depth = (
            edge.history_depth if not merged.history_depth_declared
            else min(merged.history_depth, edge.history_depth)
        )
        merged.history_depth_declared = True
    if edge.lifespan_ms is not None:
        merged.lifespan_ms = (
            edge.lifespan_ms if merged.lifespan_ms is None
            else min(merged.lifespan_ms, edge.lifespan_ms)
        )
    return merged


def _apply_qos_mode(qos: QoSProfile, qos_mode: str, default_queue: int) -> QoSProfile:
    """Project a declared contract onto what `qos_mode` actually enforces.

    Done once, here, rather than by scattering `if qos_mode == ...` through the
    processes: every consumer then reads a profile that already says what this
    run enforces, and adding a mode cannot miss a site.
    """
    effective = QoSProfile(**{**qos.__dict__})

    # Reader queue capacity. DDS KEEP_LAST(history_depth) -- which is what
    # docs/failure-simulation.md has claimed all along while the code used a flat
    # default_queue_size. An explicitly declared queue_size still wins; an
    # undeclared history stays an unconstrained reader cache rather than being
    # forced to DEFAULT_HISTORY_DEPTH, because the resolver cannot tell a topic
    # that asked for 10 from one that asked for nothing.
    if (qos_mode in _CAPACITY_FROM_HISTORY
            and qos.history_depth_declared
            and not qos.queue_size_declared):
        effective.queue_size = max(1, qos.history_depth)

    if qos_mode not in _DEADLINES_ENFORCED:
        effective.deadline_ms = None
        effective.lifespan_ms = None
        effective.deadline_declared = False

    if qos_mode == "none":
        # Policy neutralised, load left alone (see QOS_MODES).
        effective.reliability = "RELIABLE"
        effective.durability = "VOLATILE"
        effective.transport_priority = "MEDIUM"
        effective.queue_size = (
            qos.queue_size if qos.queue_size_declared else default_queue
        )
    return effective


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
        sub_stats: Optional[SubscriberFlowStats] = None,
        bucket_of: Any = None,
    ) -> None:
        _require_simpy()
        self.env = env
        self.topic_id = topic_id
        self.subscriber_id = subscriber_id
        self.qos = qos
        #: Owner's stats record, so an overflow drop is attributable to the
        #: subscriber that lost the sample and not only to the topic.
        self.sub_stats = sub_stats
        #: Shared window bucketer. `_try_put` runs inside `publish()`, i.e. at the
        #: message's creation instant, so bucketing on `env.now` here is the same
        #: clock the publisher and subscriber use.
        self.bucket_of = bucket_of
        self._store: simpy.Store = simpy.Store(env, capacity=qos.queue_size)

    def get(self) -> "simpy.resources.store.StoreGet":
        return self._store.get()

    @property
    def depth(self) -> int:
        return len(self._store.items)

    def _record_drop(self, stats: TopicFlowStats) -> None:
        """Attribute one lost sample to this queue's subscriber.

        Overflow drops used to be counted on TopicFlowStats only, so a message
        discarded here appeared in neither `received_per_topic` nor
        `missed_per_topic` — it simply vanished. That made a RELIABLE topic's
        loss unmeasurable, because the head-drop below is its *only* loss mode.
        """
        window = self.bucket_of(self.env.now) if self.bucket_of else None
        if window == "pre":
            stats.queue_overflows_pre += 1
        elif window == "post":
            stats.queue_overflows_post += 1

        if self.sub_stats is None:
            return
        self.sub_stats.missed_per_topic[self.topic_id] = (
            self.sub_stats.missed_per_topic.get(self.topic_id, 0) + 1
        )
        if window == "post":
            self.sub_stats.missed_post_fault += 1

    def _try_put(self, msg: Message, stats: TopicFlowStats) -> bool:
        """Enqueue msg; apply overflow policy; return True if enqueued."""
        if self.depth >= self.qos.queue_size:
            stats.total_dropped_queue_full += 1
            if self.qos.reliability == "BEST_EFFORT":
                stats.total_dropped_best_effort += 1
                self._record_drop(stats)
                return False
            # RELIABLE: head-drop oldest to make room
            if self._store.items:
                self._store.items.pop(0)
                self._record_drop(stats)
        self._store.put(msg)
        return True


class TopicFanout:
    """
    Manages fan-out from one publisher topic to all registered subscriber
    queues.

    The publisher calls publish(msg, failed_nodes) once per message.
    Each live subscriber's queue receives a copy.
    """

    #: Durability levels whose retained history survives the writer that held it.
    #: TRANSIENT_LOCAL keeps history only in the writer's own process, so it is
    #: recoverable from a *surviving* co-publisher but dies with an orphaned
    #: topic; TRANSIENT and PERSISTENT are backed by a durability service that
    #: outlives the writer. VOLATILE retains nothing. Ordered exactly as
    #: QoSPolicy.DURABILITY_SCORES, so replay's effect is monotone in the score.
    REPLAY_IF_CO_PUBLISHER: frozenset = frozenset(
        {"TRANSIENT_LOCAL", "TRANSIENT", "PERSISTENT"})
    REPLAY_IF_ORPHANED: frozenset = frozenset({"TRANSIENT", "PERSISTENT"})

    def __init__(self, topic_id: str, qos: QoSProfile, stats: TopicFlowStats) -> None:
        self.topic_id = topic_id
        self.qos = qos
        self.stats = stats
        # subscriber_id → SubscriberQueue
        self._queues: Dict[str, SubscriberQueue] = {}
        #: Writer-side retained samples, KEEP_LAST(history_depth). Always kept:
        #: durability governs who may *read* it back after a loss, not whether
        #: the writer caches.
        self._history: "deque[Message]" = deque(maxlen=max(1, qos.history_depth))
        #: Messages actually emitted post-fault, as against the demand counted in
        #: `stats.published_post` (which includes what a silenced publisher would
        #: have sent). Their difference is what the outage cost.
        self._emitted_post = 0

    def register(
        self,
        env: simpy.Environment,
        subscriber_id: str,
        sub_stats: Optional[SubscriberFlowStats] = None,
        bucket_of: Any = None,
        qos: Optional[QoSProfile] = None,
    ) -> SubscriberQueue:
        """Create and register a per-subscriber receive queue.

        `qos` is the *effective* profile for this (topic, subscriber) pair. It
        defaults to the topic's own offered policy, which is what every caller
        got before the reader side was allowed to matter: the queue was built
        from `self.qos` while the edge's parsed profile was used for nothing but
        the deadline, so a reader-declared depth or overflow policy was read and
        thrown away.
        """
        sq = SubscriberQueue(
            env, self.topic_id, subscriber_id, qos or self.qos, sub_stats, bucket_of
        )
        self._queues[subscriber_id] = sq
        return sq

    def queue_for(self, subscriber_id: str) -> Optional[SubscriberQueue]:
        return self._queues.get(subscriber_id)

    def publish(self, msg: Message, failed_nodes: Set[str]) -> int:
        """
        Fan out *msg* to all live subscriber queues.

        Returns the number of queues the message was placed into.
        Increments stats.total_published once regardless of fan-out width —
        including when no queue accepted it, so that this denominator agrees
        with the windowed publish counters used for the fault-impact rates.
        """
        n_queued = 0
        for sub_id, sq in self._queues.items():
            if sub_id in failed_nodes:
                continue
            if sq._try_put(msg, self.stats):
                n_queued += 1
        self.stats.total_published += 1
        self._history.append(msg)
        return n_queued

    def replay(
        self,
        env: simpy.Environment,
        failed_nodes: Set[str],
        orphaned: bool,
        deadline_policy: str = "original",
    ) -> int:
        """Re-deliver retained samples to surviving readers after an outage.

        Bounded by `min(history_depth, messages lost during the outage)`: a
        durability service can return what the reader missed, never more. Without
        that cap a 1 Hz topic with history_depth=100 would inject 100 samples
        into a window that only ever expected ~55.

        Replayed samples go through the ordinary `_try_put`, so the burst
        genuinely competes for queue space, costs service time at the shared
        station, and delays the live stream queued behind it.
        """
        if not self._history:
            return 0
        eligible = (
            self.REPLAY_IF_ORPHANED if orphaned else self.REPLAY_IF_CO_PUBLISHER
        )
        if self.qos.durability not in eligible:
            return 0

        lost = max(0, self.stats.published_post - self._emitted_post)
        n = min(len(self._history), lost)
        if n <= 0:
            return 0

        replayed = list(self._history)[-n:]
        for sub_id, sq in self._queues.items():
            if sub_id in failed_nodes:
                continue
            for original in replayed:
                # `window_at` always moves to the replay instant: the recovery is
                # a post-fault event whatever the sample's own age. Only `reset`
                # additionally moves `created_at`, which is what the deadline and
                # the latency sample are measured from.
                msg = Message(
                    msg_id=original.msg_id,
                    topic_id=original.topic_id,
                    publisher_id=original.publisher_id,
                    created_at=env.now if deadline_policy == "reset" else original.created_at,
                    payload_size_bytes=original.payload_size_bytes,
                    window_at=env.now,
                )
                self.stats.replayed_total += 1
                if sq._try_put(msg, self.stats):
                    self.stats.replayed_enqueued += 1
        return n

    @property
    def subscriber_ids(self) -> List[str]:
        return list(self._queues.keys())


# ─────────────────────────────────────────────────────────────────────────────
# SimPy process functions
# ─────────────────────────────────────────────────────────────────────────────

def _make_bucketer(
    fault_time: Optional[float],
    warmup_s: float = 0.0,
    guard_band_s: float = 0.0,
):
    """Map a message's *creation* time to "pre", "post", or None (excluded).

    One bucketer serves both the publisher (demand) and the subscriber
    (delivery), which is the point: the two used to bucket on different clocks
    -- demand on emission time, delivery on queue-*exit* time -- so a message
    published just before the fault but dequeued just after landed in
    pub_window["pre"] and del_window["post"]. The windows were not
    mass-conserving across the boundary, which is why the rate had to be clamped
    into [0, 1] afterwards. At the 1 ms service times that skew was invisible;
    under contention the sojourn is many periods and it dominates.

    ``warmup_s`` drops the transient while queues fill from empty.
    ``guard_band_s`` drops the messages straddling the fault, so "pre" is pure
    steady-state-before and "post" pure steady-state-after.
    """
    def bucket(created_at: float) -> Optional[str]:
        if created_at < warmup_s:
            return None
        if fault_time is None:
            return "post"          # no fault: preserve the single-window behaviour
        if abs(created_at - fault_time) < guard_band_s:
            return None
        return "pre" if created_at < fault_time else "post"

    return bucket


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
    bucket_of: Any,                  # created_at -> "pre" | "post" | None
    processing_time_s: float = 0.0,
    use_poisson: bool = False,
    phase_offset_s: float = 0.0,
) -> Generator:
    """
    Publisher SimPy process.

    Emits one message periodically or stochastically (Poisson process) based on rate_hz.
    Stops silently when app_id is in failed_nodes.
    """
    # BUG-MFS-7 FIX: absolute emission schedule. Previously the loop was
    # timeout(interval) -> ... -> timeout(processing_time), so the publisher's own
    # delay was *added* to the inter-publication period and the effective rate was
    # 1/(1/f + proc) rather than f. At the 1 ms default that is a 4.8% shortfall at
    # 50 Hz and 16.7% at 200 Hz, so every high-rate topic in the corpus silently
    # under-published. Tracking the next emission as an absolute time lets a
    # publisher catch up after its own bookkeeping instead of drifting behind it,
    # while still never exceeding the declared rate.
    # BUG-MFS-8 FIX: stagger co-publishers of one topic.
    #
    # Every publisher of a topic used to start at t=0 with the same interval
    # (frequency / n_publishers each), so all N fired at the *same instants*: the
    # topic's aggregate stream was a burst of N every N/f seconds rather than an
    # even stream at f. With 100-deep queues all N fitted and nothing showed, but
    # once history_depth binds the capacity a depth-1 reader keeps exactly one of
    # each burst -- delivery collapses to 1/N, which is a phase artifact and not
    # a QoS effect. Offsetting publisher i of N by i*interval/N makes the
    # aggregate evenly spaced, which is what "topic frequency f" should mean.
    if phase_offset_s > 0:
        yield env.timeout(phase_offset_s)

    next_emit = env.now
    while True:
        if use_poisson:
            interval = rng.expovariate(rate_hz) if rate_hz > 0 else 1.0
        else:
            interval = 1.0 / rate_hz if rate_hz > 0 else 1.0

        next_emit += interval

        # Publisher compute for the upcoming sample, taken *before* the emission
        # instant so the absolute schedule absorbs it. Placing it after the wait
        # would put a gap between the instant this loop counts demand at and the
        # `created_at` it stamps, and those two have to be the same timestamp for
        # the pre/post windows to conserve mass.
        if processing_time_s > 0:
            yield env.timeout(processing_time_s * (0.8 + 0.4 * rng.random()))

        yield env.timeout(max(0.0, next_emit - env.now))
        emit_time = env.now

        # Count the demand this workload places on the topic in the correct time
        # window, whether or not the publisher is still alive to serve it. A
        # silenced publisher's subscribers still expect the feed, so its lost
        # messages have to stay in the denominator — otherwise they leave the
        # numerator and denominator together and the fault cancels itself out.
        window = bucket_of(emit_time)
        if window is not None:
            window_counts[window] += 1
            if window == "pre":
                fanout.stats.published_pre += 1
            else:
                fanout.stats.published_post += 1

        # Failed publishers keep generating demand but emit nothing.
        if app_id in failed_nodes:
            continue

        if window == "post":
            fanout._emitted_post += 1

        msg_counter[0] += 1
        msg = Message(
            msg_id=msg_counter[0],
            topic_id=topic_id,
            publisher_id=app_id,
            created_at=emit_time,
        )
        fanout.publish(msg, failed_nodes)




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
    bucket_of: Any,                          # created_at -> "pre" | "post" | None
    max_latency_samples: int = 10_000,
    processing_time_s: float = 0.0,
    latency_windows: Optional[Dict[str, list]] = None,
    station: Optional[ServiceStation] = None,
    service_priority: int = 0,
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
            if fault_time is not None and env.now >= fault_time:
                sub_stats.missed_post_fault += 1
            return

        enqueue_time = msg.created_at

        # Subscriber-side compute. When a station is present this is the one
        # contended resource in the engine: every topic this subscriber reads
        # queues for the same server, so a fault that silences one publisher
        # relieves the others -- an effect no topological oracle can express.
        if station is not None:
            with station.resource.request(priority=service_priority) as req:
                yield req
                service_time = station.sample_service_time(rng)
                if service_time > 0:
                    station.record_service(service_time)
                    yield env.timeout(service_time)
        elif processing_time_s > 0:
            yield env.timeout(processing_time_s * (0.8 + 0.4 * rng.random()))

        # BUG-MFS-5 FIX: end-to-end latency includes subscriber processing time
        delivery_time = env.now
        e2e_latency_ms = (delivery_time - enqueue_time) * 1000.0

        # Window membership follows the message's *creation* time, the same
        # timestamp the publisher counted its demand at, so numerator and
        # denominator always describe the same population of messages.
        window = bucket_of(msg.window_time)
        post_fault = window == "post"

        # Lifespan check (message may have expired while queued)
        if qos.lifespan_ms is not None and e2e_latency_ms > qos.lifespan_ms:
            sub_stats.missed_per_topic[received_key] += 1
            if post_fault:
                sub_stats.missed_post_fault += 1
            continue

        # Deadline check (DDS deadline = end-to-end)
        if qos.deadline_ms is not None and e2e_latency_ms > qos.deadline_ms:
            sub_stats.deadline_violations_per_topic[received_key] += 1
            topic_stats.total_dropped_deadline += 1
            if window == "pre":
                topic_stats.deadline_violations_pre += 1
            elif window == "post":
                topic_stats.deadline_violations_post += 1
            sub_stats.missed_per_topic[received_key] += 1
            if post_fault:
                sub_stats.missed_post_fault += 1
            continue

        # Delivered
        sub_stats.received_per_topic[received_key] += 1
        if msg.window_at is not None:
            # A replayed sample is kept out of the lifetime `total_delivered`,
            # whose denominator counts messages the publishers actually emitted.
            # Replay recovers demand that was *never* emitted (the dead writer's
            # share), so adding it to that numerator pushed `delivery_rate` above
            # 1.0 on any topic whose publisher was faulted. The windowed counters
            # below are safe to credit: their denominator is demand, and replay
            # is capped at `min(history_depth, demand_post - emitted_post)`, so
            # delivered_post can never exceed it.
            topic_stats.replayed_delivered += 1
        else:
            topic_stats.total_delivered += 1

        # BUG-MFS-2 FIX: count deliveries in the correct time window.
        # Mirrored onto sub_stats so _annotate_fault_cascade can subtract the
        # faulted node's own receipts out of both windows.
        if window is not None:
            delivery_window_counts[window] += 1
            if window == "pre":
                topic_stats.delivered_pre += 1
                sub_stats.received_pre_fault += 1
            else:
                topic_stats.delivered_post += 1
                if fault_time is not None:
                    sub_stats.received_post_fault += 1

        # Latency sample
        if len(topic_stats.latency_samples) < max_latency_samples:
            topic_stats.latency_samples.append(e2e_latency_ms)
        if latency_windows is not None and fault_time is not None and window is not None:
            latency_windows[window].append(e2e_latency_ms)


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
    warmup_s : float
        Messages created before this time are counted in neither window. Lets
        queues reach steady state before measurement starts. Default 0.0, which
        preserves the historical behaviour of measuring from an empty system.
    guard_band_s : float
        Messages created within this distance of ``fault_time`` are counted in
        neither window, so "pre" is pure steady-state-before and "post" pure
        steady-state-after. Default 0.0.
    qos_mode : str
        Which QoS policies this run enforces; one of ``QOS_MODES``. Defaults to
        ``"full"``: declared QoS is enforced unless a caller opts out. Pass
        ``"legacy"`` for the historical policy (flat queue size, FIFO, no
        replay, no calibration).
    target_utilization : float, optional
        Baseline load each subscriber is calibrated to, or None to leave service
        times uncalibrated. Defaults to 0.65 -- the operating point where QoS
        contracts bind while the measurement stays reproducible. Above it the
        run-to-run variance grows faster than the signal (I_dyn's own
        test-retest falls from 0.93 at rho=0.65 to 0.89 at rho=0.8), and below
        it nothing is contended and no contract is reachable.
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
        warmup_s: float = 0.0,
        guard_band_s: float = 0.0,
        qos_mode: str = "full",
        target_utilization: Optional[float] = 0.65,
        utilization_mode: str = "per_subscriber",
        service_distribution: str = "exponential",
        service_concurrency: int = 1,
        durability_replay_deadline: str = "original",
        durability_replay_delay_s: Optional[float] = None,
    ) -> None:
        _require_simpy()
        if qos_mode not in QOS_MODES:
            raise ValueError(
                f"qos_mode must be one of {QOS_MODES}, got {qos_mode!r}"
            )
        if utilization_mode not in UTILIZATION_MODES:
            raise ValueError(
                f"utilization_mode must be one of {UTILIZATION_MODES}, "
                f"got {utilization_mode!r}"
            )
        if service_distribution not in SERVICE_DISTRIBUTIONS:
            raise ValueError(
                f"service_distribution must be one of {SERVICE_DISTRIBUTIONS}, "
                f"got {service_distribution!r}"
            )
        if target_utilization is not None and not 0.0 < target_utilization < 1.0:
            raise ValueError(
                "target_utilization must lie strictly in (0, 1) -- at rho >= 1 "
                f"the queue is unstable and has no steady state; got {target_utilization!r}"
            )
        self.qos_mode = qos_mode
        self.target_utilization = target_utilization
        self.utilization_mode = utilization_mode
        self.service_distribution = service_distribution
        self.service_concurrency = service_concurrency
        if durability_replay_deadline not in REPLAY_DEADLINE_POLICIES:
            raise ValueError(
                f"durability_replay_deadline must be one of "
                f"{REPLAY_DEADLINE_POLICIES}, got {durability_replay_deadline!r}"
            )
        self.durability_replay_deadline = durability_replay_deadline
        self.durability_replay_delay_s = durability_replay_delay_s
        self.graph = graph
        self.duration = duration
        self.fault_node = fault_node
        self.fault_time = fault_time if fault_time is not None else duration / 2.0
        self.seed = seed
        self.default_queue_size = default_queue_size
        self.default_publish_rate_hz = default_publish_rate_hz
        self.default_processing_time_s = default_processing_time_s
        self.max_latency_samples = max_latency_samples
        self.warmup_s = warmup_s
        self.guard_band_s = guard_band_s

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
            qos = _apply_qos_mode(
                _extract_qos(data, self.default_queue_size),
                self.qos_mode,
                self.default_queue_size,
            )
            topic_qos[node] = qos
            topic_stats[node] = TopicFlowStats(
                topic_id=node,
                topic_name=data.get("name", node),
                reliability_policy=qos.reliability,
                deadline_ms=qos.deadline_ms,
                durability_policy=qos.durability,
                history_depth=qos.history_depth,
            )
        return topic_qos, topic_stats

    def _build_subscribers(
        self,
        env: simpy.Environment,
        sub_edges: List[Tuple[str, str, dict]],
        fanouts: Dict[str, TopicFanout],
        topic_qos: Dict[str, QoSProfile],
        bucket_of: Any = None,
    ) -> Tuple[
        Dict[str, SubscriberFlowStats],
        Dict[Tuple[str, str], SubscriberQueue],
        Dict[Tuple[str, str], QoSProfile],
    ]:
        """Register one queue per (topic, subscriber) pair and seed subscriber stats."""
        sub_topics: Dict[str, List[str]] = defaultdict(list)
        for src, tgt, _ in sub_edges:
            sub_topics[src].append(tgt)

        sub_stats = {
            sub_id: SubscriberFlowStats(subscriber_id=sub_id, subscribed_topics=topics)
            for sub_id, topics in sub_topics.items()
        }

        # The effective profile is resolved *before* the queue is constructed,
        # because it is what sets the queue's capacity and overflow policy.
        sub_queues: Dict[Tuple[str, str], SubscriberQueue] = {}
        effective_qos: Dict[Tuple[str, str], QoSProfile] = {}
        for src, tgt, data in sub_edges:
            qos = self._subscriber_qos(data, topic_qos[tgt])
            effective_qos[(tgt, src)] = qos
            sub_queues[(tgt, src)] = fanouts[tgt].register(
                env, src, sub_stats[src], bucket_of, qos
            )

        return sub_stats, sub_queues, effective_qos

    def _subscriber_qos(self, edge_data: dict, topic_qos: QoSProfile) -> QoSProfile:
        """Effective QoS for one (topic, subscriber) pair.

        `topic_qos` has already been through `_apply_qos_mode`, so the merge runs
        against what this run enforces and the result is re-projected to pick up
        any capacity rule the edge's own declarations changed.
        """
        edge_qos = _extract_qos(edge_data, self.default_queue_size)
        return _apply_qos_mode(
            _merge_qos(topic_qos, edge_qos), self.qos_mode, self.default_queue_size
        )

    @property
    def _calibration_active(self) -> bool:
        return (
            self.target_utilization is not None
            and self.qos_mode in _CALIBRATED_MODES
        )

    def topic_aggregate_rate(self, topic_id: str) -> float:
        """The topic's declared publication rate, before it is split per publisher.

        `generate_workload` returns the *per-publisher* share. Calibration needs
        the aggregate: sizing a subscriber's service rate off the per-publisher
        share would make its operating point depend on how many publishers the
        topic happens to have — i.e. on exactly the topology the fault perturbs,
        which would confound the thing being measured with the thing perturbing it.
        """
        if topic_id in self.graph.nodes:
            node = self.graph.nodes[topic_id]
            freq = node.get("frequency", node.get("topic_frequency"))
            if freq is not None:
                try:
                    return float(freq)
                except (TypeError, ValueError):
                    pass
        return self.default_publish_rate_hz

    def _build_service_stations(
        self,
        env: simpy.Environment,
        sub_edges: List[Tuple[str, str, dict]],
        proc_time: Dict[str, float],
    ) -> Dict[str, ServiceStation]:
        """One station per subscriber, sized to hit `target_utilization`.

        For subscriber *s* reading topics *T(s)*, the offered load is
        ``Lambda_s = sum of aggregate rates over T(s)`` and the mean service time
        that puts it at ``rho`` is ``E[S_s] = rho / Lambda_s``.

        An explicit per-node ``processing_time`` attribute still wins and is
        recorded as uncalibrated, so a scenario that deliberately models a slow
        component keeps doing so.
        """
        offered: Dict[str, float] = defaultdict(float)
        for src, tgt, _ in sub_edges:
            offered[src] += self.topic_aggregate_rate(tgt)

        rho = self.target_utilization
        global_load = None
        if self._calibration_active and self.utilization_mode == "global" and offered:
            ordered = sorted(offered.values())
            global_load = ordered[max(0, int(0.9 * len(ordered)) - 1)]

        stations: Dict[str, ServiceStation] = {}
        for sub_id, load in offered.items():
            declared = self.graph.nodes.get(sub_id, {}).get("processing_time")
            calibrated = False

            if declared is not None:
                service_time = proc_time.get(sub_id, self.default_processing_time_s)
            elif not self._calibration_active:
                service_time = proc_time.get(sub_id, self.default_processing_time_s)
            else:
                effective_load = global_load if global_load is not None else load
                if effective_load > 0:
                    service_time = rho / effective_load
                    calibrated = True
                else:
                    # No inbound traffic: nothing to calibrate against. Falls back
                    # rather than dividing by zero, and is excluded from the
                    # realised-utilization report so it cannot dilute the check.
                    service_time = self.default_processing_time_s

            stations[sub_id] = ServiceStation(
                env=env,
                subscriber_id=sub_id,
                service_time_s=service_time,
                concurrency=self.service_concurrency,
                distribution=(
                    self.service_distribution if self._calibration_active else "uniform"
                ),
                calibrated=calibrated,
                measure_from=self.warmup_s,
            )
        return stations

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
        sub_stats: Dict[str, SubscriberFlowStats],
    ) -> None:
        """
        Fill in what the fault actually cost: which topics it orphaned, which
        subscribers lost a feed, and the delivery/latency shift across the
        fault boundary.

        The delivery rates measure harm to the components that *survived*. The
        faulted node's own undelivered messages are excluded from both windows:
        counting them makes the impact score track how much the node consumed
        rather than how much the rest of the system depended on it.
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
        # fan-out (one publish becomes N expected deliveries). Both the faulted
        # node's receipts and its share of the fan-out are removed, so the two
        # windows describe the same surviving population.
        faulted = sub_stats.get(self.fault_node)
        own = {
            "pre": faulted.received_pre_fault if faulted else 0,
            "post": faulted.received_post_fault if faulted else 0,
        }
        surviving_subs = {
            tid: len([s for s in fo.subscriber_ids if s != self.fault_node])
            for tid, fo in fanouts.items()
        }

        def _rate(window: str) -> float:
            delivered = sum(dw[window] for dw in del_window.values()) - own[window]
            expected = sum(
                pub_window[tid][window] * surviving_subs[tid] for tid in fanouts
            )
            if not expected:
                return 0.0
            # Deliberately unclamped. The old min/max existed because demand and
            # delivery were bucketed on different clocks, so the windows did not
            # conserve mass and the ratio could stray outside [0, 1] for purely
            # accounting reasons. Both now bucket on the message's creation time,
            # so a value outside [0, 1] is a real measurement -- and once the
            # subscriber's compute is a contended resource, a negative I_dyn is a
            # genuine result: removing a chatty publisher can relieve more
            # contention than it removes feeds. Clamping would hide exactly that.
            return delivered / expected

        record.cascade_silenced_publishers = [self.fault_node]
        record.cascade_orphaned_topics = orphaned
        record.cascade_impacted_subscribers = impacted
        record.delivery_rate_before = _rate("pre")
        record.delivery_rate_after = _rate("post")

        # Per-topic decomposition, so a result can say *which* topics carried the
        # loss rather than only how large the system-wide drop was. Topics whose
        # windows carried no demand are omitted: unmeasured is not zero.
        topic_stats = {tid: fo.stats for tid, fo in fanouts.items()}
        record.per_topic_i_dyn = {
            tid: round(stats.i_dyn_topic, 6)
            for tid, stats in topic_stats.items()
            if stats.i_dyn_topic is not None
        }
        record.deadline_violations_before = sum(
            s.deadline_violations_pre for s in topic_stats.values())
        record.deadline_violations_after = sum(
            s.deadline_violations_post for s in topic_stats.values())
        record.queue_overflows_before = sum(
            s.queue_overflows_pre for s in topic_stats.values())
        record.queue_overflows_after = sum(
            s.queue_overflows_post for s in topic_stats.values())

        record.warmup_s = self.warmup_s
        record.guard_band_s = self.guard_band_s
        record.pre_window_s = max(
            0.0, self.fault_time - self.guard_band_s - self.warmup_s)
        record.post_window_s = max(
            0.0, self.duration - self.fault_time - self.guard_band_s)
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

        # Resolved before the queues are built: SubscriberQueue needs it to bucket
        # overflow drops into the pre/post windows.
        fault_time = self.fault_time if self.fault_node else None

        bucket_of = _make_bucketer(fault_time, self.warmup_s, self.guard_band_s)

        sub_stats, sub_queues, effective_qos = self._build_subscribers(
            env, sub_edges, fanouts, topic_qos, bucket_of
        )
        proc_time = self._node_processing_times()
        stations = self._build_service_stations(env, sub_edges, proc_time)

        # Fan-out width, recorded once the subscribers are registered. Without it
        # TopicFlowStats.delivery_rate divides per-(subscriber, message) deliveries
        # by per-message publications and exceeds 1.0 on any multi-subscriber topic.
        for tid, fanout in fanouts.items():
            topic_stats[tid].n_subscribers = len(fanout.subscriber_ids)

        # Per-topic publish/delivery counters, split on the fault boundary.
        pub_window = {tid: {"pre": 0, "post": 0} for tid in fanouts}
        del_window = {tid: {"pre": 0, "post": 0} for tid in fanouts}
        latency_windows: Dict[str, list] = {"pre": [], "post": []}

        # Index each publisher within its own topic, so co-publishers can be
        # phase-staggered rather than all firing on the same instants.
        pubs_of: Dict[str, List[str]] = defaultdict(list)
        for src, tgt, _ in pub_edges:
            pubs_of[tgt].append(src)

        for src, tgt, _ in pub_edges:
            topic_node = self.graph.nodes[tgt] if tgt in self.graph.nodes else {}
            rate = self.generate_workload(tgt)
            siblings = pubs_of[tgt]
            phase = (
                (siblings.index(src) / len(siblings)) / rate
                if rate > 0 and len(siblings) > 1 else 0.0
            )
            env.process(_publisher_process(
                env=env,
                app_id=src,
                topic_id=tgt,
                rate_hz=rate,
                fanout=fanouts[tgt],
                failed_nodes=failed_nodes,
                fault_time=fault_time,
                window_counts=pub_window[tgt],
                msg_counter=msg_counter,
                rng=rng,
                bucket_of=bucket_of,
                processing_time_s=(
                    0.0 if self._calibration_active
                    else proc_time.get(src, self.default_processing_time_s)
                ),
                use_poisson=str(topic_node.get("workload_type", "")).lower() == "poisson",
                phase_offset_s=phase,
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
                qos=effective_qos[(tgt, src)],
                failed_nodes=failed_nodes,
                fault_time=fault_time,
                sub_stats=sub_stats[src],
                topic_stats=topic_stats[tgt],
                delivery_window_counts=del_window[tgt],
                rng=rng,
                bucket_of=bucket_of,
                max_latency_samples=self.max_latency_samples,
                processing_time_s=proc_time.get(src, self.default_processing_time_s),
                latency_windows=latency_windows,
                station=stations.get(src),
                # SimPy PriorityResource is lowest-value-first and stable within
                # a class, so every mode but `full` degenerates cleanly to FIFO.
                service_priority=(
                    -int(1000 * QoSPolicy.PRIORITY_SCORES.get(
                        effective_qos[(tgt, src)].transport_priority, 0.33))
                    if self.qos_mode in _PRIORITY_ORDERED else 0
                ),
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

            if self.qos_mode in _REPLAY_ENABLED:
                # Which topics lose their last publisher, computed once from the
                # topology rather than from a counter, so an orphaned topic is
                # identified the same way `_annotate_fault_cascade` identifies it.
                publishers_of: Dict[str, Set[str]] = defaultdict(set)
                for src, tgt, _ in pub_edges:
                    publishers_of[tgt].add(src)

                def _durability_replay_process(env: simpy.Environment) -> Generator:
                    """Hand retained samples back once the outage is detectable.

                    The fault is permanent — `failed_nodes` is never cleared — so
                    replay cannot be triggered by the publisher recovering. It
                    fires on takeover instead: surviving co-publishers (or the
                    durability service, for TRANSIENT/PERSISTENT) serve the gap
                    the dead writer left.
                    """
                    for tid, fanout in fanouts.items():
                        if self.fault_node not in publishers_of.get(tid, set()):
                            continue
                        # Liveliness/discovery lease: roughly three missed
                        # deadlines before the loss is acted on.
                        delay = self.durability_replay_delay_s
                        if delay is None:
                            rate = self.topic_aggregate_rate(tid)
                            delay = max(1.0, 3.0 / rate) if rate > 0 else 1.0
                        yield env.timeout(max(0.0, self.fault_time + delay - env.now))

                        orphaned = not (publishers_of[tid] - failed_nodes)
                        fanout.replay(
                            env, failed_nodes, orphaned,
                            self.durability_replay_deadline,
                        )

                env.process(_durability_replay_process(env))

        logger.info(
            "Message-flow sim: duration=%.1fs | fault=%s | seed=%d",
            self.duration, self.fault_node or "none", self.seed,
        )
        env.run(until=self.duration)
        logger.info("Simulation complete.")

        total_delivered = sum(ts.total_delivered for ts in topic_stats.values())
        # A message is "fully delivered" once every subscriber receives it, so
        # normalise against published × fan-out to get a per-copy rate. Topics
        # with no subscribers drop out rather than being credited an expectation
        # that nobody actually holds.
        total_expected = sum(
            ts.total_published * len(fanouts[tid].subscriber_ids)
            for tid, ts in topic_stats.items()
        )
        system_delivery = total_delivered / total_expected if total_expected else 0.0

        if fault_event_record is not None:
            self._annotate_fault_cascade(
                fault_event_record, edges, fanouts,
                pub_window, del_window, latency_windows, sub_stats,
            )

        # This engine can only observe components that carry pub/sub traffic.
        # Brokers (ROUTES) and Nodes (RUNS_ON) are invisible to it, so faulting
        # one is a no-op — they are reported unlabelled rather than scored 0.0,
        # because an omitted component is unmeasured, not measured as harmless.
        labeled = {s for s, _, _ in pub_edges} | {s for s, _, _ in sub_edges}

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
            qos_mode=self.qos_mode,
            target_utilization=self.target_utilization,
            utilization_mode=self.utilization_mode,
            service_distribution=self.service_distribution,
            measured_utilization={
                sub_id: round(st.utilization(self.duration), 4)
                for sub_id, st in stations.items() if st.calibrated
            },
            service_time_s={
                sub_id: st.service_time_s for sub_id, st in stations.items()
            },
            topic_stats=topic_stats,
            subscriber_stats=sub_stats,
            labeled_node_ids=sorted(labeled),
            unlabeled_node_ids=sorted(set(self.graph.nodes) - labeled),
        )
