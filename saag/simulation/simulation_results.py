"""
simulation_results.py
─────────────────────
Dataclasses for the two simulation modes in the SaG pipeline:

  1. FaultInjectionResult  – per-node proxy ground-truth I(v) produced by
                             the BFS cascade fault injector.  This is the
                             I(v) vector that Q(v) is validated against
                             (Spearman ρ, F1, etc.).

  2. MessageFlowResult     – aggregate statistics from a discrete-event
                             pub-sub message-flow simulation run (SimPy).

Both are JSON-serialisable via `asdict()` / `to_dict()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Fault Injection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CascadeWave:
    """One propagation wave in a fault-injection cascade."""
    wave_index: int                      # 0-based
    newly_orphaned_topics: List[str]     # Topics that lost all publishers in this wave
    newly_impacted_subscribers: List[str]  # Subscribers that lost ≥1 feed in this wave
    newly_failed_publishers: List[str]   # Publishers silenced by this wave (cascade spread)


@dataclass
class FaultInjectionRecord:
    """
    Full result of injecting a single-node fault.

    I(v) – the proxy ground-truth impact score – is the primary output.
    It is defined as the *weighted* fraction of subscriber data-feed capacity
    destroyed:

        I(v) = Σ_{a ∈ Subscribers}  |lost_feeds(a)| / |total_feeds(a)|
               ─────────────────────────────────────────────────────────
                              |Subscribers|

    This is a richer measure than binary "is subscriber impacted?" because it
    captures partial feed loss (e.g. an ATCWorkstation losing 1 of 3 feeds
    scores less than losing all 3).
    """
    node_id: str
    node_type: str                   # Application | Broker | Node | Library
    node_name: str

    # Core impact score (the I(v) used for Spearman correlation with Q(v))
    impact_score: float              # ∈ [0, 1]

    # Cascade statistics
    total_orphaned_topics: int       # Topics that lost all publishers
    total_impacted_subscribers: int  # Subscribers that lost ≥1 feed
    total_subscribers: int           # Denominator
    cascade_depth: int               # Number of cascade waves fired

    # Derived breakdown
    directly_orphaned_topics: List[str]     # Topics orphaned by removing *this* node
    all_orphaned_topics: List[str]          # Including cascaded orphaning
    impacted_subscriber_ids: List[str]      # Unique subscriber IDs impacted
    per_subscriber_feed_loss: Dict[str, float]  # subscriber_id → fraction of feeds lost

    # Cascade trace (for visualisation / debugging)
    cascade_waves: List[CascadeWave] = field(default_factory=list)

    # Multi-seed stability (populated when --seeds used)
    seed_impact_scores: Dict[int, float] = field(default_factory=dict)  # seed → I(v)
    impact_score_std: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cascade_waves"] = [asdict(w) for w in self.cascade_waves]
        return d


@dataclass
class FaultInjectionResult:
    """
    Aggregated result of a full fault-injection experiment across all nodes.
    This is the canonical output written to ``impact_scores.json``.
    """
    schema_version: str = "2.0"
    graph_id: str = ""
    total_nodes_injected: int = 0
    total_application_nodes: int = 0
    total_broker_nodes: int = 0
    total_subscribers: int = 0          # Denominator used for all I(v)
    seeds_used: List[int] = field(default_factory=list)

    # Per-node records, keyed by node_id
    records: Dict[str, FaultInjectionRecord] = field(default_factory=dict)

    # Ranked summary (top-k by I(v)), populated after all records are added
    top_k_by_impact: List[Dict[str, Any]] = field(default_factory=list)

    def add_record(self, rec: FaultInjectionRecord) -> None:
        self.records[rec.node_id] = rec
        self.total_nodes_injected = len(self.records)

    def finalise(self, top_k: int = 20) -> None:
        """Sort records and build top-k summary list."""
        ranked = sorted(
            self.records.values(),
            key=lambda r: r.impact_score,
            reverse=True,
        )
        self.top_k_by_impact = [
            {
                "rank": i + 1,
                "node_id": r.node_id,
                "node_type": r.node_type,
                "node_name": r.node_name,
                "impact_score": round(r.impact_score, 4),
                "cascade_depth": r.cascade_depth,
                "orphaned_topics": r.total_orphaned_topics,
                "impacted_subscribers": r.total_impacted_subscribers,
                "impact_score_std": round(r.impact_score_std, 4),
            }
            for i, r in enumerate(ranked[:top_k])
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "total_nodes_injected": self.total_nodes_injected,
            "total_application_nodes": self.total_application_nodes,
            "total_broker_nodes": self.total_broker_nodes,
            "total_subscribers": self.total_subscribers,
            "seeds_used": self.seeds_used,
            "top_k_by_impact": self.top_k_by_impact,
            "records": {
                nid: rec.to_dict() for nid, rec in self.records.items()
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Message Flow Simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TopicFlowStats:
    """Per-topic statistics from a message-flow simulation run."""
    topic_id: str
    topic_name: str
    reliability_policy: str             # RELIABLE | BEST_EFFORT
    deadline_ms: Optional[float]        # None means no deadline enforced
    durability_policy: str              # VOLATILE | TRANSIENT_LOCAL

    total_published: int = 0            # Messages injected by all publishers
    total_delivered: int = 0            # Messages received by ≥1 subscriber
    total_dropped_queue_full: int = 0   # Dropped due to queue overflow
    total_dropped_deadline: int = 0     # Dropped due to deadline violation
    total_dropped_best_effort: int = 0  # Dropped because policy = BEST_EFFORT under load

    latency_samples: List[float] = field(default_factory=list)  # ms, sampled

    @property
    def delivery_rate(self) -> float:
        return self.total_delivered / self.total_published if self.total_published else 0.0

    @property
    def drop_rate(self) -> float:
        return 1.0 - self.delivery_rate

    @property
    def latency_p50(self) -> Optional[float]:
        return _percentile(self.latency_samples, 50)

    @property
    def latency_p95(self) -> Optional[float]:
        return _percentile(self.latency_samples, 95)

    @property
    def latency_p99(self) -> Optional[float]:
        return _percentile(self.latency_samples, 99)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "reliability_policy": self.reliability_policy,
            "deadline_ms": self.deadline_ms,
            "durability_policy": self.durability_policy,
            "total_published": self.total_published,
            "total_delivered": self.total_delivered,
            "total_dropped_queue_full": self.total_dropped_queue_full,
            "total_dropped_deadline": self.total_dropped_deadline,
            "total_dropped_best_effort": self.total_dropped_best_effort,
            "delivery_rate": round(self.delivery_rate, 4),
            "drop_rate": round(self.drop_rate, 4),
            "latency_p50_ms": round(self.latency_p50, 3) if self.latency_p50 is not None else None,
            "latency_p95_ms": round(self.latency_p95, 3) if self.latency_p95 is not None else None,
            "latency_p99_ms": round(self.latency_p99, 3) if self.latency_p99 is not None else None,
        }


@dataclass
class SubscriberFlowStats:
    """Per-subscriber statistics from a message-flow simulation run."""
    subscriber_id: str
    subscribed_topics: List[str]

    received_per_topic: Dict[str, int] = field(default_factory=dict)     # topic_id → count
    missed_per_topic: Dict[str, int] = field(default_factory=dict)       # topic_id → count
    deadline_violations_per_topic: Dict[str, int] = field(default_factory=dict)

    # Post-fault statistics (populated only when fault injection is enabled)
    received_post_fault: int = 0
    missed_post_fault: int = 0

    @property
    def total_received(self) -> int:
        return sum(self.received_per_topic.values())

    @property
    def total_missed(self) -> int:
        return sum(self.missed_per_topic.values())

    @property
    def overall_delivery_rate(self) -> float:
        total = self.total_received + self.total_missed
        return self.total_received / total if total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subscriber_id": self.subscriber_id,
            "subscribed_topics": self.subscribed_topics,
            "received_per_topic": self.received_per_topic,
            "missed_per_topic": self.missed_per_topic,
            "deadline_violations_per_topic": self.deadline_violations_per_topic,
            "total_received": self.total_received,
            "total_missed": self.total_missed,
            "overall_delivery_rate": round(self.overall_delivery_rate, 4),
            "received_post_fault": self.received_post_fault,
            "missed_post_fault": self.missed_post_fault,
        }


@dataclass
class FaultEventRecord:
    """Records the timing and cascade of a fault injected during simulation."""
    fault_time: float
    faulted_node_id: str
    faulted_node_type: str
    cascade_silenced_publishers: List[str]  # Publishers that went silent after fault
    cascade_orphaned_topics: List[str]      # Topics that died after fault
    cascade_impacted_subscribers: List[str]  # Subscribers that lost feeds after fault
    delivery_rate_before: float            # System-wide rate in [0, fault_time]
    delivery_rate_after: float             # System-wide rate in (fault_time, end]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MessageFlowResult:
    """
    Full result of one discrete-event message-flow simulation run.
    """
    schema_version: str = "2.0"
    graph_id: str = ""
    simulation_duration: float = 0.0     # simulated seconds
    seed: int = 42
    fault_event: Optional[FaultEventRecord] = None

    # Aggregate system-wide metrics
    system_delivery_rate: float = 0.0    # fraction of all published messages delivered
    system_drop_rate: float = 0.0
    total_messages_published: int = 0
    total_messages_delivered: int = 0
    total_deadline_violations: int = 0
    total_queue_overflows: int = 0

    # Per-topic and per-subscriber breakdowns
    topic_stats: Dict[str, TopicFlowStats] = field(default_factory=dict)
    subscriber_stats: Dict[str, SubscriberFlowStats] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "simulation_duration": self.simulation_duration,
            "seed": self.seed,
            "fault_event": self.fault_event.to_dict() if self.fault_event else None,
            "system_delivery_rate": round(self.system_delivery_rate, 4),
            "system_drop_rate": round(self.system_drop_rate, 4),
            "total_messages_published": self.total_messages_published,
            "total_messages_delivered": self.total_messages_delivered,
            "total_deadline_violations": self.total_deadline_violations,
            "total_queue_overflows": self.total_queue_overflows,
            "topic_stats": {tid: ts.to_dict() for tid, ts in self.topic_stats.items()},
            "subscriber_stats": {sid: ss.to_dict() for sid, ss in self.subscriber_stats.items()},
        }
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _percentile(data: List[float], p: int) -> Optional[float]:
    if not data:
        return None
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)
