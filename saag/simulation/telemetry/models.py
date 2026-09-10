"""
models.py
─────────
Telemetry data structures for the Unified RuntimeTelemetrySimulator.
Models cross-layer performance, QoS contracts, and operational metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


@dataclass
class QoSViolationEvent:
    """Record of a runtime QoS violation (deadline, lifespan, or queue drop)."""
    time: float
    message_id: str
    topic_id: str
    subscriber_id: str
    violation_type: str  # "deadline", "lifespan", "queue_overflow"
    latency_ms: float
    threshold_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": round(self.time, 6),
            "message_id": self.message_id,
            "topic_id": self.topic_id,
            "subscriber_id": self.subscriber_id,
            "violation_type": self.violation_type,
            "latency_ms": round(self.latency_ms, 4),
            "threshold_ms": self.threshold_ms,
        }


@dataclass
class StarvationEvent:
    """Record of a subscriber suffering significant feed loss."""
    time: float
    subscriber_id: str
    topic_id: str
    feed_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": round(self.time, 6),
            "subscriber_id": self.subscriber_id,
            "topic_id": self.topic_id,
            "feed_loss": round(self.feed_loss, 4),
        }


@dataclass
class ComponentTelemetry:
    """Operational metrics for an individual Application, Broker, or Library."""
    component_id: str
    component_type: str
    component_name: str = ""
    messages_sent: int = 0
    messages_received: int = 0
    messages_routed: int = 0
    messages_dropped: int = 0
    feed_starvation_ratio: float = 0.0  # Fraction of feeds starved (0.0 to 1.0)
    processing_time_total: float = 0.0
    active_duration: float = 0.0        # Time spent active in the run
    is_failed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "component_name": self.component_name or self.component_id,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_routed": self.messages_routed,
            "messages_dropped": self.messages_dropped,
            "feed_starvation_ratio": round(self.feed_starvation_ratio, 4),
            "processing_time_total": round(self.processing_time_total, 6),
            "active_duration": round(self.active_duration, 4),
            "is_failed": self.is_failed,
        }


@dataclass
class TopicTelemetry:
    """Operational metrics and QoS enforcement results for a Topic."""
    topic_id: str
    topic_name: str
    published_count: int = 0
    delivered_count: int = 0
    dropped_queue_full: int = 0
    dropped_deadline: int = 0
    dropped_lifespan: int = 0
    dropped_no_route: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0
    delivery_rate: float = 0.0
    qos_reliability: str = "RELIABLE"
    qos_durability: str = "VOLATILE"
    deadline_ms: Optional[float] = None
    lifespan_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "published_count": self.published_count,
            "delivered_count": self.delivered_count,
            "dropped_queue_full": self.dropped_queue_full,
            "dropped_deadline": self.dropped_deadline,
            "dropped_lifespan": self.dropped_lifespan,
            "dropped_no_route": self.dropped_no_route,
            "latency_p50_ms": round(self.latency_p50_ms, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 3),
            "latency_p99_ms": round(self.latency_p99_ms, 3),
            "latency_mean_ms": round(self.latency_mean_ms, 3),
            "delivery_rate": round(self.delivery_rate, 4),
            "qos_reliability": self.qos_reliability,
            "qos_durability": self.qos_durability,
            "deadline_ms": self.deadline_ms,
            "lifespan_ms": self.lifespan_ms,
        }


@dataclass
class NodeTelemetry:
    """Infrastructure metrics for physical host machines (Node)."""
    node_id: str
    node_name: str
    messages_in: int = 0
    messages_out: int = 0
    bandwidth_bps_in: float = 0.0
    bandwidth_bps_out: float = 0.0
    estimated_cpu_load: float = 0.0
    hosted_components: List[str] = field(default_factory=list)
    is_failed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "messages_in": self.messages_in,
            "messages_out": self.messages_out,
            "bandwidth_bps_in": round(self.bandwidth_bps_in, 2),
            "bandwidth_bps_out": round(self.bandwidth_bps_out, 2),
            "estimated_cpu_load": round(self.estimated_cpu_load, 4),
            "hosted_components": list(self.hosted_components),
            "is_failed": self.is_failed,
        }


@dataclass
class SystemTelemetry:
    """Aggregate runtime telemetry collected across the entire graph."""
    graph_id: str = ""
    simulation_duration: float = 0.0
    seed: int = 42
    total_messages_generated: int = 0
    total_messages_delivered: int = 0
    total_messages_dropped: int = 0
    system_delivery_rate: float = 0.0
    system_drop_rate: float = 0.0
    system_latency_p50_ms: float = 0.0
    system_latency_p95_ms: float = 0.0
    system_latency_p99_ms: float = 0.0
    system_latency_mean_ms: float = 0.0

    component_metrics: Dict[str, ComponentTelemetry] = field(default_factory=dict)
    topic_metrics: Dict[str, TopicTelemetry] = field(default_factory=dict)
    node_metrics: Dict[str, NodeTelemetry] = field(default_factory=dict)

    qos_violations: List[QoSViolationEvent] = field(default_factory=list)
    starvation_events: List[StarvationEvent] = field(default_factory=list)

    faulted_nodes: Set[str] = field(default_factory=set)
    fault_time: Optional[float] = None

    # Pre-fault / post-fault split metrics when a fault was injected
    pre_fault_delivery_rate: Optional[float] = None
    post_fault_delivery_rate: Optional[float] = None
    pre_fault_p95_latency_ms: Optional[float] = None
    post_fault_p95_latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "2.0",
            "graph_id": self.graph_id,
            "simulation_duration": round(self.simulation_duration, 4),
            "seed": self.seed,
            "total_messages_generated": self.total_messages_generated,
            "total_messages_delivered": self.total_messages_delivered,
            "total_messages_dropped": self.total_messages_dropped,
            "system_delivery_rate": round(self.system_delivery_rate, 4),
            "system_drop_rate": round(self.system_drop_rate, 4),
            "system_latency_p50_ms": round(self.system_latency_p50_ms, 3),
            "system_latency_p95_ms": round(self.system_latency_p95_ms, 3),
            "system_latency_p99_ms": round(self.system_latency_p99_ms, 3),
            "system_latency_mean_ms": round(self.system_latency_mean_ms, 3),
            "faulted_nodes": sorted(self.faulted_nodes),
            "fault_time": self.fault_time,
            "pre_fault_delivery_rate": round(self.pre_fault_delivery_rate, 4) if self.pre_fault_delivery_rate is not None else None,
            "post_fault_delivery_rate": round(self.post_fault_delivery_rate, 4) if self.post_fault_delivery_rate is not None else None,
            "pre_fault_p95_latency_ms": round(self.pre_fault_p95_latency_ms, 3) if self.pre_fault_p95_latency_ms is not None else None,
            "post_fault_p95_latency_ms": round(self.post_fault_p95_latency_ms, 3) if self.post_fault_p95_latency_ms is not None else None,
            "components": {k: v.to_dict() for k, v in self.component_metrics.items()},
            "topics": {k: v.to_dict() for k, v in self.topic_metrics.items()},
            "nodes": {k: v.to_dict() for k, v in self.node_metrics.items()},
            "qos_violations_count": len(self.qos_violations),
            "starvation_events_count": len(self.starvation_events),
        }

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))


@dataclass
class TelemetryScenario:
    """Scenario configuration for a RuntimeTelemetrySimulator run."""
    duration: float = 100.0
    fault_node: Optional[str] = None
    fault_time: Optional[float] = None
    seed: int = 42
    default_publish_rate_hz: float = 10.0
    default_processing_time_s: float = 0.001
    default_network_latency_s: float = 0.0005
    default_queue_capacity: int = 100
    propagation_threshold: float = 0.20
    poisson_arrivals: bool = False
    max_latency_samples: int = 10_000
