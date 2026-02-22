from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Any, Optional
from enum import Enum


# =============================================================================
# Core Simulation Enums
# =============================================================================

class ComponentState(Enum):
    """State of a component during simulation."""
    ACTIVE = "active"
    FAILED = "failed"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"


class RelationType(Enum):
    """RAW structural relationship types in the pub-sub graph."""
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    ROUTES = "ROUTES"
    RUNS_ON = "RUNS_ON"
    CONNECTS_TO = "CONNECTS_TO"
    USES = "USES"


class EventType(Enum):
    """Types of discrete events in the simulation."""
    PUBLISH = "publish"       # Message published by app
    ROUTE = "route"           # Message routed by broker
    DELIVER = "deliver"       # Message delivered to subscriber
    ACK = "ack"               # Acknowledgment (for reliable delivery)
    TIMEOUT = "timeout"       # Delivery timeout
    DROP = "drop"             # Message dropped


class FailureMode(Enum):
    """Types of component failure modes."""
    CRASH = "crash"           # Complete failure - component stops
    DEGRADED = "degraded"     # Partial failure (50% capacity). Starvation threshold: SL < 0.3
    PARTITION = "partition"   # Network partition - unreachable
    OVERLOAD = "overload"     # Resource exhaustion (Future work)


class CascadeRule(Enum):
    """Rules governing failure cascade propagation."""
    PHYSICAL = "physical"     # Node failure cascades to hosted components
    LOGICAL = "logical"       # Broker failure affects topic routing
    NETWORK = "network"       # Network partition propagation
    ALL = "all"               # All cascade rules applied


# =============================================================================
# Core Entities
# =============================================================================

@dataclass
class ComponentInfo:
    """Information about a component in the simulation."""
    id: str
    type: str  # Application, Topic, Broker, Node
    state: ComponentState = ComponentState.ACTIVE
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime metrics (accumulated during simulation)
    messages_sent: int = 0
    messages_received: int = 0
    messages_dropped: int = 0
    messages_routed: int = 0
    total_latency: float = 0.0
    
    def reset_metrics(self) -> None:
        """Reset runtime metrics for a new simulation run."""
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_dropped = 0
        self.messages_routed = 0
        self.total_latency = 0.0
    
    @property
    def avg_latency(self) -> float:
        """Average latency per message processed."""
        if self.messages_sent + self.messages_received == 0:
            return 0.0
        return self.total_latency / (self.messages_sent + self.messages_received)

    @property
    def performance(self) -> float:
        """Performance level (1.0 = healthy, 0.5 = degraded, 0.0 = failed)."""
        if self.state == ComponentState.FAILED:
            return 0.0
        elif self.state == ComponentState.DEGRADED:
            return 0.5
        return 1.0
        total = self.messages_received + self.messages_routed
        return self.total_latency / total if total > 0 else 0.0
    
    @property
    def throughput(self) -> int:
        """Total messages processed (sent + routed)."""
        return self.messages_sent + self.messages_routed


@dataclass
class TopicInfo:
    """Information about a topic including QoS settings."""
    id: str
    name: str
    message_size: int = 1024
    qos_reliability: str = "BEST_EFFORT"  # BEST_EFFORT, RELIABLE
    qos_durability: str = "VOLATILE"      # VOLATILE, TRANSIENT, PERSISTENT
    qos_priority: str = "LOW"             # LOW, MEDIUM, HIGH, URGENT
    weight: float = 1.0
    
    @property
    def requires_ack(self) -> bool:
        """Check if topic requires acknowledgment (reliable delivery)."""
        return self.qos_reliability == "RELIABLE"
    
    @property
    def priority_value(self) -> int:
        """Numeric priority for scheduling."""
        return {"URGENT": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(self.qos_priority, 1)
    
    @property
    def persistence_factor(self) -> float:
        """Factor for persistence overhead."""
        return {"PERSISTENT": 1.5, "TRANSIENT": 1.2, "VOLATILE": 1.0}.get(self.qos_durability, 1.0)


# =============================================================================
# Event Simulation Models
# =============================================================================

@dataclass(order=True)
class Event:
    """A discrete event in the simulation."""
    time: float                                      # Event time (for priority queue)
    event_type: EventType = field(compare=False)
    source: str = field(compare=False)               # Source component
    target: str = field(compare=False)               # Target component
    message_id: str = field(compare=False)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class Message:
    """A message being transmitted through the pub-sub system."""
    id: str
    source_app: str
    topic_id: str
    size: int
    priority: int
    requires_ack: bool
    created_at: float
    hops: int = 0
    delivered_to: Set[str] = field(default_factory=set)
    dropped: bool = False
    drop_reason: Optional[str] = None
    delivered_at: Optional[float] = None
    
    @property
    def latency(self) -> float:
        """End-to-end latency (only valid after delivery)."""
        if self.delivered_at is not None:
            return self.delivered_at - self.created_at
        return 0.0


@dataclass
class EventScenario:
    """Configuration for an event simulation run."""
    source_app: str
    description: str = ""
    num_messages: int = 100
    message_interval: float = 0.01
    message_size: int = 1024
    duration: float = 10.0
    seed: Optional[int] = None
    publish_latency: float = 0.001
    broker_latency: float = 0.002
    network_latency: float = 0.005
    subscribe_latency: float = 0.001
    drop_probability: float = 0.0
    broker_failure_prob: float = 0.0
    delivery_timeout: float = 1.0


@dataclass
class RuntimeMetrics:
    """Runtime metrics collected during event simulation."""
    messages_published: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    messages_in_flight: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    latencies: List[float] = field(default_factory=list)
    simulation_duration: float = 0.0
    
    @property
    def delivery_rate(self) -> float:
        """Percentage of messages delivered."""
        total = self.messages_published
        return (self.messages_delivered / total * 100) if total > 0 else 0.0
    
    @property
    def drop_rate(self) -> float:
        """Percentage of messages dropped."""
        total = self.messages_published
        return (self.messages_dropped / total * 100) if total > 0 else 0.0
    
    @property
    def avg_latency(self) -> float:
        """Average latency in seconds."""
        return self.total_latency / self.messages_delivered if self.messages_delivered > 0 else 0.0
    
    @property
    def p99_latency(self) -> float:
        """99th percentile latency."""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]
    
    @property
    def throughput(self) -> float:
        """Messages per second."""
        return self.messages_delivered / self.simulation_duration if self.simulation_duration > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages_published": self.messages_published,
            "messages_delivered": self.messages_delivered,
            "messages_dropped": self.messages_dropped,
            "delivery_rate_percent": round(self.delivery_rate, 2),
            "drop_rate_percent": round(self.drop_rate, 2),
            "avg_latency_ms": round(self.avg_latency * 1000, 3),
            "p99_latency_ms": round(self.p99_latency * 1000, 3),
            "throughput_per_sec": round(self.throughput, 2),
        }


@dataclass
class EventResult:
    """Result of an event simulation run."""
    source_app: str
    scenario: str
    duration: float
    metrics: RuntimeMetrics
    affected_topics: List[str] = field(default_factory=list)
    reached_subscribers: List[str] = field(default_factory=list)
    brokers_used: List[str] = field(default_factory=list)
    successful_flows: List[Tuple[str, str, str]] = field(default_factory=list) # (Pub, Topic, Sub)
    component_impacts: Dict[str, float] = field(default_factory=dict)
    failed_components: List[str] = field(default_factory=list)
    drop_reasons: Dict[str, int] = field(default_factory=dict)
    component_names: Dict[str, str] = field(default_factory=dict)
    related_components: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_app": self.source_app,
            "scenario": self.scenario,
            "duration_sec": round(self.duration, 3),
            "metrics": self.metrics.to_dict(),
            "affected_topics": self.affected_topics,
            "reached_subscribers": self.reached_subscribers,
            "brokers_used": self.brokers_used,
            "component_impacts": {k: round(v, 4) for k, v in self.component_impacts.items()},
            "failed_components": self.failed_components,
            "drop_reasons": self.drop_reasons,
            "related_components": self.related_components,
        }


# =============================================================================
# Failure Simulation Models
# =============================================================================

@dataclass
class FailureScenario:
    """Configuration for a failure simulation."""
    target_ids: List[str]                            # Simultaneous initial targets
    description: str = ""
    failure_mode: FailureMode = FailureMode.CRASH
    cascade_rule: CascadeRule = CascadeRule.ALL
    cascade_probability: float = 1.0
    max_cascade_depth: int = 10
    layer: str = "system"
    seed: Optional[int] = None

    @property
    def target_id(self) -> str:
        """Backward compatibility for single-target scenarios."""
        return self.target_ids[0] if self.target_ids else ""


@dataclass
class ImpactMetrics:
    """Impact metrics from a failure simulation."""
    initial_paths: int = 0
    remaining_paths: int = 0
    reachability_loss: float = 0.0
    initial_components: int = 0
    failed_components: int = 0
    initial_connected_components: int = 1
    final_connected_components: int = 1
    fragmentation: float = 0.0
    initial_throughput: float = 1.0
    remaining_throughput: float = 1.0
    throughput_loss: float = 0.0
    affected_topics: int = 0
    affected_subscribers: int = 0
    affected_publishers: int = 0
    cascade_count: int = 0
    cascade_depth: int = 0
    flow_disruption: float = 0.0 # Fraction of event-sim flows broken
    cascade_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Impact weights I(v) - Formally derived via AHP in weight_calculator.py
    impact_weights: Dict[str, float] = field(default_factory=lambda: {
        "reachability": 0.35,
        "fragmentation": 0.25, 
        "throughput": 0.25,
        "flow_disruption": 0.15
    })

    # -----------------------------------------------------------------------
    # IR(v): Reliability-specific ground truth (fault propagation dynamics)
    # -----------------------------------------------------------------------
    # Populated by failure_simulator.simulate_exhaustive() in a post-pass.
    # cascade_reach = cascade_count / (|V| - 1)
    # weighted_cascade_impact = Σw(cascaded) / Σw(all)
    # normalized_cascade_depth = cascade_depth / max_observed_depth_in_run
    cascade_reach: float = 0.0
    weighted_cascade_impact: float = 0.0
    normalized_cascade_depth: float = 0.0

    reliability_weights: Dict[str, float] = field(default_factory=lambda: {
        "cascade_reach": 0.45,
        "weighted_cascade_impact": 0.35,
        "normalized_depth": 0.20,
    })
    
    @property
    def composite_impact(self) -> float:
        w = self.impact_weights
        return (
            w.get("reachability", 0.35) * self.reachability_loss +
            w.get("fragmentation", 0.25) * self.fragmentation +
            w.get("throughput", 0.25) * self.throughput_loss +
            w.get("flow_disruption", 0.15) * self.flow_disruption
        )

    @property
    def reliability_impact(self) -> float:
        """IR(v) — Reliability-specific ground truth from fault propagation dynamics.

        Measures cascade propagation directly rather than Availability-biased
        structural connectivity loss.  Only meaningful after the exhaustive
        simulation post-pass has populated cascade_reach, weighted_cascade_impact,
        and normalized_cascade_depth.  Defaults to 0.0 until then.
        """
        w = self.reliability_weights
        return (
            w.get("cascade_reach", 0.45) * self.cascade_reach +
            w.get("weighted_cascade_impact", 0.35) * self.weighted_cascade_impact +
            w.get("normalized_depth", 0.20) * self.normalized_cascade_depth
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reachability": {
                "initial_paths": self.initial_paths,
                "remaining_paths": self.remaining_paths,
                "loss_percent": round(self.reachability_loss * 100, 2),
            },
            "fragmentation": {
                "fragmentation_percent": round(self.fragmentation * 100, 2),
            },
            "throughput": {
                "loss_percent": round(self.throughput_loss * 100, 2),
            },
            "flow_disruption": {
                "loss_percent": round(self.flow_disruption * 100, 2),
            },
            "cascade": {
                "count": self.cascade_count,
                "depth": self.cascade_depth,
            },
            "composite_impact": round(self.composite_impact, 4),
            "reliability": {
                "cascade_reach": round(self.cascade_reach, 4),
                "weighted_cascade_impact": round(self.weighted_cascade_impact, 4),
                "normalized_cascade_depth": round(self.normalized_cascade_depth, 4),
                "reliability_impact": round(self.reliability_impact, 4),
            },
        }


@dataclass
class CascadeEvent:
    """Record of a cascade propagation event."""
    component_id: str
    component_type: str
    cause: str
    depth: int


@dataclass
class FailureResult:
    """Result of a failure simulation."""
    target_id: str
    target_type: str
    scenario: str
    impact: ImpactMetrics
    cascaded_failures: List[str] = field(default_factory=list)
    cascade_sequence: List[CascadeEvent] = field(default_factory=list)
    layer_impacts: Dict[str, float] = field(default_factory=dict)
    related_components: List[str] = field(default_factory=list)
    component_names: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_type": self.target_type,
            "scenario": self.scenario,
            "impact": self.impact.to_dict(),
            "cascaded_failures": self.cascaded_failures,
            "cascade_sequence": [
                {"id": e.component_id, "type": e.component_type, "cause": e.cause, "depth": e.depth}
                for e in self.cascade_sequence
            ],
            "layer_impacts": {k: round(v, 4) for k, v in self.layer_impacts.items()},
        }

    def cascade_to_graph(self) -> Dict[str, Any]:
        """
        Convert the cascade sequence into a graph structure (nodes and edges)
        suitable for visualization.
        """
        nodes = []
        edges = []
        
        # Add root target node
        nodes.append({
            "id": self.target_id,
            "type": self.target_type,
            "name": self.component_names.get(self.target_id, self.target_id),
            "depth": 0,
            "is_target": True
        })
        
        # Add cascaded nodes and edges
        for event in self.cascade_sequence:
            nodes.append({
                "id": event.component_id,
                "type": event.component_type,
                "name": self.component_names.get(event.component_id, event.component_id),
                "depth": event.depth,
                "cause": event.cause,
                "is_target": False
            })
            
            # Find the cause in the existing nodes to create an edge
            # This is a simplification: we assume the cause is the parent ID
            edges.append({
                "source": event.cause if event.cause in [n["id"] for n in nodes] else self.target_id,
                "target": event.component_id,
                "type": "CASCADE"
            })
            
        return {"nodes": nodes, "edges": edges}


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo stochastic simulation."""
    target_id: str
    n_trials: int
    mean_impact: float
    std_impact: float
    ci_95: Tuple[float, float]
    trial_impacts: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "n_trials": self.n_trials,
            "mean_impact": round(self.mean_impact, 4),
            "std_impact": round(self.std_impact, 4),
        }


# =============================================================================
# Aggregated Results
# =============================================================================

@dataclass
class LayerMetrics:
    """Aggregated metrics for a single simulation layer."""
    layer: str
    event_throughput: int = 0
    event_delivered: int = 0
    event_dropped: int = 0
    event_delivery_rate: float = 0.0
    event_drop_rate: float = 0.0
    event_avg_latency_ms: float = 0.0
    event_p99_latency_ms: float = 0.0
    event_throughput_per_sec: float = 0.0
    event_drop_reasons: Dict[str, int] = field(default_factory=dict)
    avg_reachability_loss: float = 0.0
    avg_fragmentation: float = 0.0
    avg_throughput_loss: float = 0.0
    max_impact: float = 0.0
    max_impact_component: str = ""
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    minimal_count: int = 0
    spof_count: int = 0
    total_components: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "event_metrics": {
                "delivery_rate_percent": round(self.event_delivery_rate, 2),
                "avg_latency_ms": round(self.event_avg_latency_ms, 3),
            },
            "failure_metrics": {
                "avg_reachability_loss_percent": round(self.avg_reachability_loss * 100, 2),
                "max_impact": round(self.max_impact, 4),
            },
            "criticality": {
                "critical": self.critical_count,
                "high": self.high_count,
            },
        }


@dataclass
class ComponentCriticality:
    """Criticality assessment for a component."""
    id: str
    type: str
    event_impact: float = 0.0
    failure_impact: float = 0.0
    combined_impact: float = 0.0
    level: str = "minimal"
    cascade_count: int = 0
    cascade_depth: int = 0
    message_throughput: int = 0
    reachability_loss: float = 0.0
    throughput_loss: float = 0.0
    affected_topics: int = 0
    affected_subscribers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "level": self.level,
            "combined_impact": round(self.combined_impact, 4),
        }


@dataclass
class EdgeCriticality:
    """Criticality assessment for an edge."""
    source: str
    target: str
    relationship: str
    flow_impact: float = 0.0
    connectivity_impact: float = 0.0
    combined_impact: float = 0.0
    level: str = "minimal"
    messages_traversed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "level": self.level,
            "combined_impact": round(self.combined_impact, 4),
        }


@dataclass
class SimulationReport:
    """Comprehensive simulation report."""
    timestamp: str
    graph_summary: Dict[str, Any]
    layer_metrics: Dict[str, LayerMetrics] = field(default_factory=dict)
    component_criticality: List[ComponentCriticality] = field(default_factory=list)
    edge_criticality: List[EdgeCriticality] = field(default_factory=list)
    top_critical: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    component_names: Dict[str, str] = field(default_factory=dict)
    library_usage: Dict[str, List[str]] = field(default_factory=dict)
    node_allocations: Dict[str, List[str]] = field(default_factory=dict)
    broker_routing: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "layer_metrics": {k: v.to_dict() for k, v in self.layer_metrics.items()},
            "top_critical": self.top_critical,
        }
