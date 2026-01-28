from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class LayerMetrics:
    """
    Aggregated metrics for a single analysis layer.
    
    Combines event simulation metrics (throughput, latency, drops)
    with failure simulation metrics (impact, cascade, fragmentation).
    """
    layer: str
    
    # Event simulation metrics
    event_throughput: int = 0            # Total messages processed
    event_delivery_rate: float = 0.0     # % messages delivered
    event_drop_rate: float = 0.0         # % messages dropped
    event_avg_latency_ms: float = 0.0    # Average latency in ms
    
    # Failure simulation metrics
    avg_reachability_loss: float = 0.0   # Average path loss
    avg_fragmentation: float = 0.0       # Average component loss
    avg_throughput_loss: float = 0.0     # Average capacity loss
    max_impact: float = 0.0              # Highest single-point impact
    
    # Criticality counts
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    spof_count: int = 0
    
    # Component count
    total_components: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "event_metrics": {
                "throughput": self.event_throughput,
                "delivery_rate_percent": round(self.event_delivery_rate, 2),
                "drop_rate_percent": round(self.event_drop_rate, 2),
                "avg_latency_ms": round(self.event_avg_latency_ms, 3),
            },
            "failure_metrics": {
                "avg_reachability_loss_percent": round(self.avg_reachability_loss * 100, 2),
                "avg_fragmentation_percent": round(self.avg_fragmentation * 100, 2),
                "avg_throughput_loss_percent": round(self.avg_throughput_loss * 100, 2),
                "max_impact": round(self.max_impact, 4),
            },
            "criticality": {
                "total_components": self.total_components,
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "spof_count": self.spof_count,
            },
        }


@dataclass
class ComponentCriticality:
    """
    Criticality assessment for a component based on simulation results.
    
    Combines event impact (message handling importance) with
    failure impact (cascade and reachability effects).
    """
    id: str
    type: str
    
    # Simulation-based scores
    event_impact: float = 0.0         # Impact from event simulation
    failure_impact: float = 0.0       # Impact from failure simulation
    combined_impact: float = 0.0      # Weighted combination
    
    # Classification level
    level: str = "minimal"
    
    # Supporting metrics
    cascade_count: int = 0
    message_throughput: int = 0
    reachability_loss: float = 0.0
    
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
                "message_throughput": self.message_throughput,
                "reachability_loss_percent": round(self.reachability_loss * 100, 2),
            },
        }


@dataclass
class SimulationReport:
    """
    Comprehensive simulation report combining all analyses.
    """
    timestamp: str
    graph_summary: Dict[str, Any]
    
    # Per-layer metrics
    layer_metrics: Dict[str, LayerMetrics]
    
    # Component criticality (classified by simulation)
    component_criticality: List[ComponentCriticality]
    
    # Top critical components
    top_critical: List[Dict[str, Any]]
    
    # Recommendations
    recommendations: List[str]
    
    # Name mapping for display
    component_names: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "graph_summary": self.graph_summary,
            "layer_metrics": {k: v.to_dict() for k, v in self.layer_metrics.items()},
            "component_criticality": [c.to_dict() for c in self.component_criticality],
            "top_critical": self.top_critical,
            "recommendations": self.recommendations,
        }
    
    def get_critical_components(self) -> List[ComponentCriticality]:
        """Get all CRITICAL level components."""
        return [c for c in self.component_criticality if c.level == "critical"]
    
    def get_high_priority(self) -> List[ComponentCriticality]:
        """Get CRITICAL and HIGH level components."""
        return [c for c in self.component_criticality if c.level in ("critical", "high")]
