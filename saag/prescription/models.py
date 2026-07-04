"""
saag/prescription/models.py
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class PrescriptionPolicy:
    """Represents the compiled optimization policy Delta(G) to be applied to the graph."""
    topic_splits: List[Dict[str, Any]] = field(default_factory=list)
    node_reallocations: List[Dict[str, Any]] = field(default_factory=list)
    qos_upgrades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_splits": self.topic_splits,
            "node_reallocations": self.node_reallocations,
            "qos_upgrades": self.qos_upgrades,
        }

@dataclass
class PrescribeResult:
    """Result of the prescriptive Stage 6 optimization and closed-loop validation."""
    original_sri: float
    mutated_sri: float
    sri_improvement: float
    original_metrics: Dict[str, Any]
    mutated_metrics: Dict[str, Any]
    policy: PrescriptionPolicy
    applied_changes: List[str] = field(default_factory=list)
    # Per-component simulated cascade impact I(v) (composite_impact from the canonical
    # FailureSimulator), before and after mutation, restricted to remediated components whose
    # identity is stable across the mutation (node reallocations, QoS upgrades — topic splits
    # replace the original topic id and so have no stable before/after counterpart).
    remediated_component_impact_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Mean fractional cascade-impact reduction (I_before - I_after) / I_before, averaged over
    # remediated_component_impact_deltas entries with I_before > 0. None if no such component exists.
    mean_cascade_impact_reduction: Optional[float] = None
    # True when sri_improvement > 0 -- the mutated policy reduced overall system risk.
    # Reported, not enforced: a rejected policy is still returned in full for inspection.
    accepted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_sri": self.original_sri,
            "mutated_sri": self.mutated_sri,
            "sri_improvement": self.sri_improvement,
            "original_metrics": self.original_metrics,
            "mutated_metrics": self.mutated_metrics,
            "policy": self.policy.to_dict(),
            "applied_changes": self.applied_changes,
            "remediated_component_impact_deltas": self.remediated_component_impact_deltas,
            "mean_cascade_impact_reduction": self.mean_cascade_impact_reduction,
            "accepted": self.accepted,
        }
