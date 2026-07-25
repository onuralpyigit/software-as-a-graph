"""
saag/prescription/models.py
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class EditVerdict:
    """Outcome of verifying one candidate edit on its own.

    A policy used to be applied wholesale and judged by a single end-state SRI
    check, so an edit that made the system worse could ride along with edits
    that made it better — the mechanism behind the reported +4.61% mean that
    concealed regressions of up to -31.67%. Each edit now carries its own
    verdict from its own counterfactual simulation.
    """
    kind: str                       # "topic_split" | "node_reallocation" | "qos_upgrade"
    target: str                     # topic or component id the edit acts on
    delta_impact: float = 0.0       # mean I(v) reduction; positive is an improvement
    sigma_seed: float = 0.0         # across-seed std of that reduction
    kappa: float = 1.0              # acceptance multiple required of sigma
    accepted: bool = False
    reason: str = ""
    #: Per-threshold delta across the propagation-threshold sweep. An edit is
    #: accepted only if it clears the bar at *every* threshold, so acceptance
    #: cannot be an artifact of the canonical 0.2 default.
    per_threshold: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "delta_impact": round(self.delta_impact, 6),
            "sigma_seed": round(self.sigma_seed, 6),
            "kappa": self.kappa,
            "accepted": self.accepted,
            "reason": self.reason,
            "per_threshold": {k: round(v, 6) for k, v in self.per_threshold.items()},
        }


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

    def is_empty(self) -> bool:
        return not (self.topic_splits or self.node_reallocations or self.qos_upgrades)

    def edits(self) -> List[tuple]:
        """Flatten to ``(kind, target, payload)`` triples for per-edit verification."""
        out = []
        for split in self.topic_splits:
            out.append(("topic_split", split.get("topic", ""), split))
        for realloc in self.node_reallocations:
            out.append(("node_reallocation", realloc.get("component", ""), realloc))
        for upgrade in self.qos_upgrades:
            out.append(("qos_upgrade", upgrade.get("topic", ""), upgrade))
        return out

    @classmethod
    def from_edits(cls, edits: List[tuple]) -> "PrescriptionPolicy":
        """Rebuild a policy from the subset of ``edits()`` triples that passed."""
        policy = cls()
        for kind, _target, payload in edits:
            if kind == "topic_split":
                policy.topic_splits.append(payload)
            elif kind == "node_reallocation":
                policy.node_reallocations.append(payload)
            elif kind == "qos_upgrade":
                policy.qos_upgrades.append(payload)
        return policy

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
    #: Per-edit verdicts from the counterfactual acceptance filter. Only edits
    #: with ``accepted=True`` are present in ``policy``; the rejected ones are
    #: retained here so a run reports what it declined and why.
    edit_verdicts: List[EditVerdict] = field(default_factory=list)
    #: The full candidate set before filtering, for accept-rate reporting.
    candidate_policy: Optional[PrescriptionPolicy] = None

    @property
    def n_accepted(self) -> int:
        return sum(1 for v in self.edit_verdicts if v.accepted)

    @property
    def n_rejected(self) -> int:
        return sum(1 for v in self.edit_verdicts if not v.accepted)

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
            "edit_verdicts": [v.to_dict() for v in self.edit_verdicts],
            "n_candidate_edits": len(self.edit_verdicts),
            "n_accepted_edits": self.n_accepted,
            "n_rejected_edits": self.n_rejected,
            "candidate_policy": self.candidate_policy.to_dict() if self.candidate_policy else None,
        }
