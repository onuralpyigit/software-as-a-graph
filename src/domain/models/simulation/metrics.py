"""
Simulation Metrics

Data classes for simulation results, layer metrics, component/edge criticality,
and comprehensive simulation reports.

Metrics are organized by simulation type:
    - Event Metrics: throughput, latency, delivery rate, message drops
    - Failure Metrics: reachability loss, fragmentation, cascade depth
    - Combined: weighted composition for criticality classification
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


# =============================================================================
# Layer Metrics
# =============================================================================

@dataclass
class LayerMetrics:
    """
    Aggregated metrics for a single simulation layer.

    Combines event simulation metrics (throughput, latency, drops)
    with failure simulation metrics (impact, cascade, fragmentation).
    Each layer (app, infra, mw, system) produces independent metrics.
    """
    layer: str

    # --- Event Simulation Metrics ---
    event_throughput: int = 0            # Total messages published
    event_delivered: int = 0             # Total messages delivered
    event_dropped: int = 0              # Total messages dropped
    event_delivery_rate: float = 0.0     # % messages delivered
    event_drop_rate: float = 0.0         # % messages dropped
    event_avg_latency_ms: float = 0.0    # Average end-to-end latency (ms)
    event_p99_latency_ms: float = 0.0    # P99 latency (ms)
    event_throughput_per_sec: float = 0.0 # Messages per second

    # Per-component drop reasons
    event_drop_reasons: Dict[str, int] = field(default_factory=dict)

    # --- Failure Simulation Metrics ---
    avg_reachability_loss: float = 0.0   # Average path loss across failures
    avg_fragmentation: float = 0.0       # Average graph fragmentation
    avg_throughput_loss: float = 0.0     # Average capacity loss
    max_impact: float = 0.0              # Highest single-component impact
    max_impact_component: str = ""       # Component with highest impact

    # --- Criticality Counts ---
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    minimal_count: int = 0
    spof_count: int = 0

    # --- Component Count ---
    total_components: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "total_components": self.total_components,
            "event_metrics": {
                "throughput": self.event_throughput,
                "delivered": self.event_delivered,
                "dropped": self.event_dropped,
                "delivery_rate_percent": round(self.event_delivery_rate, 2),
                "drop_rate_percent": round(self.event_drop_rate, 2),
                "avg_latency_ms": round(self.event_avg_latency_ms, 3),
                "p99_latency_ms": round(self.event_p99_latency_ms, 3),
                "throughput_per_sec": round(self.event_throughput_per_sec, 2),
                "drop_reasons": self.event_drop_reasons,
            },
            "failure_metrics": {
                "avg_reachability_loss_percent": round(self.avg_reachability_loss * 100, 2),
                "avg_fragmentation_percent": round(self.avg_fragmentation * 100, 2),
                "avg_throughput_loss_percent": round(self.avg_throughput_loss * 100, 2),
                "max_impact": round(self.max_impact, 4),
                "max_impact_component": self.max_impact_component,
            },
            "criticality": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
                "minimal": self.minimal_count,
                "spof_count": self.spof_count,
            },
        }


# =============================================================================
# Component Criticality
# =============================================================================

@dataclass
class ComponentCriticality:
    """
    Criticality assessment for a component based on simulation results.

    Combines event impact (message handling importance) with
    failure impact (cascade and reachability effects) into a single
    combined_impact score used for box-plot classification.
    """
    id: str
    type: str

    # Simulation-based scores
    event_impact: float = 0.0         # Normalized event handling importance
    failure_impact: float = 0.0       # Normalized failure cascade impact
    combined_impact: float = 0.0      # Weighted combination

    # Classification level (set by BoxPlotClassifier or threshold)
    level: str = "minimal"

    # Supporting detail metrics
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
            "scores": {
                "event_impact": round(self.event_impact, 4),
                "failure_impact": round(self.failure_impact, 4),
                "combined_impact": round(self.combined_impact, 4),
            },
            "level": self.level,
            "metrics": {
                "cascade_count": self.cascade_count,
                "cascade_depth": self.cascade_depth,
                "message_throughput": self.message_throughput,
                "reachability_loss_percent": round(self.reachability_loss * 100, 2),
                "throughput_loss_percent": round(self.throughput_loss * 100, 2),
                "affected_topics": self.affected_topics,
                "affected_subscribers": self.affected_subscribers,
            },
        }


# =============================================================================
# Edge Criticality
# =============================================================================

@dataclass
class EdgeCriticality:
    """
    Criticality assessment for an edge (relationship) based on simulation.

    Edges are classified by how much removing them would disrupt
    message flow or connectivity.
    """
    source: str
    target: str
    relationship: str

    # Scores
    flow_impact: float = 0.0          # Message flow disruption when removed
    connectivity_impact: float = 0.0  # Graph connectivity loss
    combined_impact: float = 0.0      # Weighted combination

    # Classification
    level: str = "minimal"

    # Detail
    messages_traversed: int = 0       # Messages that used this edge

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
            "scores": {
                "flow_impact": round(self.flow_impact, 4),
                "connectivity_impact": round(self.connectivity_impact, 4),
                "combined_impact": round(self.combined_impact, 4),
            },
            "level": self.level,
            "messages_traversed": self.messages_traversed,
        }


# =============================================================================
# Simulation Report
# =============================================================================

@dataclass
class SimulationReport:
    """
    Comprehensive simulation report combining all layer analyses.

    Generated by SimulationService.generate_report() and consumed
    by display/visualization services.
    """
    timestamp: str
    graph_summary: Dict[str, Any]

    # Per-layer metrics
    layer_metrics: Dict[str, LayerMetrics] = field(default_factory=dict)

    # Component criticality (classified by simulation)
    component_criticality: List[ComponentCriticality] = field(default_factory=list)

    # Edge criticality (classified by simulation)
    edge_criticality: List[EdgeCriticality] = field(default_factory=list)

    # Top critical components (pre-sorted summary)
    top_critical: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Auxiliary mappings for display
    component_names: Dict[str, str] = field(default_factory=dict)
    library_usage: Dict[str, List[str]] = field(default_factory=dict)
    node_allocations: Dict[str, List[str]] = field(default_factory=dict)
    broker_routing: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "graph_summary": self.graph_summary,
            "layer_metrics": {k: v.to_dict() for k, v in self.layer_metrics.items()},
            "component_criticality": [c.to_dict() for c in self.component_criticality],
            "edge_criticality": [e.to_dict() for e in self.edge_criticality],
            "top_critical": self.top_critical,
            "recommendations": self.recommendations,
            "library_usage": self.library_usage,
            "node_allocations": self.node_allocations,
            "broker_routing": self.broker_routing,
        }

    def get_critical_components(self) -> List[ComponentCriticality]:
        """Get all CRITICAL level components."""
        return [c for c in self.component_criticality if c.level == "critical"]

    def get_high_priority(self) -> List[ComponentCriticality]:
        """Get CRITICAL and HIGH level components."""
        return [c for c in self.component_criticality if c.level in ("critical", "high")]

    def get_layer_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary dict per layer for quick comparison."""
        return {
            layer: {
                "components": m.total_components,
                "event_throughput": m.event_throughput,
                "delivery_rate": m.event_delivery_rate,
                "max_impact": m.max_impact,
                "spofs": m.spof_count,
                "critical": m.critical_count,
            }
            for layer, m in self.layer_metrics.items()
        }