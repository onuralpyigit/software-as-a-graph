"""
saag/prescription/models.py
"""
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Union


@dataclass(frozen=True)
class TopicSplit:
    """Split a congested topic into one dedicated sub-topic per publisher."""
    KIND: ClassVar[str] = "topic_split"

    topic: str
    publishers: List[str]
    subscribers: List[str]

    @property
    def target(self) -> str:
        return self.topic

    def describe(self) -> str:
        return (
            f"Split topic '{self.topic}' into sub-topics per publisher: "
            f"{', '.join(self.publishers)}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "publishers": list(self.publishers),
            "subscribers": list(self.subscribers),
        }


@dataclass(frozen=True)
class NodeReallocation:
    """Move a co-located process off a SPOF/critical host onto its own node."""
    KIND: ClassVar[str] = "node_reallocation"

    component: str
    from_node: str
    to_node: str

    @property
    def target(self) -> str:
        return self.component

    def describe(self) -> str:
        return (
            f"Moved process '{self.component}' from SPOF node '{self.from_node}' "
            f"to isolated node '{self.to_node}'"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "from_node": self.from_node,
            "to_node": self.to_node,
        }


@dataclass(frozen=True)
class QosUpgrade:
    """Harden a critical channel's transport contract."""
    KIND: ClassVar[str] = "qos_upgrade"

    topic: str
    original_reliability: str
    original_durability: str
    target_reliability: str = "RELIABLE"
    target_durability: str = "TRANSIENT"

    @property
    def target(self) -> str:
        return self.topic

    def describe(self) -> str:
        return (
            f"Hardened QoS on topic '{self.topic}': "
            f"Reliability -> {self.target_reliability}, "
            f"Durability -> {self.target_durability}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "original_reliability": self.original_reliability,
            "original_durability": self.original_durability,
            "target_reliability": self.target_reliability,
            "target_durability": self.target_durability,
        }


#: Any of the three transformation operators that make up a policy.
PolicyEdit = Union[TopicSplit, NodeReallocation, QosUpgrade]

#: Which PrescriptionPolicy bucket each operator belongs in.
_EDIT_BUCKETS = {
    TopicSplit: "topic_splits",
    NodeReallocation: "node_reallocations",
    QosUpgrade: "qos_upgrades",
}


@dataclass(frozen=True)
class ThresholdStat:
    """One propagation threshold's slice of an edit's verification sweep.

    ``sigma_seed`` is the spread across *seeds at this threshold only*. Pooling
    seeds and thresholds into one sigma would measure threshold sensitivity, not
    simulator noise, and inflate the acceptance bar accordingly.
    """
    mean_delta: float
    sigma_seed: float

    def margin(self, kappa: float) -> float:
        """How far the mean reduction clears the noise bar. Positive accepts."""
        return self.mean_delta - kappa * self.sigma_seed

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_delta": round(self.mean_delta, 6),
            "sigma_seed": round(self.sigma_seed, 6),
        }


@dataclass
class EditVerdict:
    """Outcome of verifying one candidate edit on its own.

    A policy used to be applied wholesale and judged by a single end-state SRI
    check, so an edit that made the system worse could ride along with edits
    that made it better. Each edit now carries its own verdict from its own
    counterfactual simulation.
    """
    #: Bumped when the ``to_dict()`` shape changes. 2 replaced the scalar
    #: ``delta_impact``/``sigma_seed`` pair with per-threshold statistics.
    SCHEMA: ClassVar[int] = 2

    kind: str                       # TopicSplit.KIND | NodeReallocation.KIND | QosUpgrade.KIND
    target: str                     # topic or component id the edit acts on
    kappa: float = 1.0              # acceptance multiple required of sigma
    accepted: bool = False
    reason: str = ""
    #: Per-threshold statistics across the propagation-threshold sweep. An edit
    #: is accepted only if it clears the bar at *every* threshold, so acceptance
    #: cannot be an artifact of the canonical 0.2 default.
    per_threshold: Dict[str, ThresholdStat] = field(default_factory=dict)

    @property
    def worst_delta(self) -> float:
        """Smallest mean impact reduction over the threshold sweep."""
        return min((s.mean_delta for s in self.per_threshold.values()), default=0.0)

    @property
    def worst_margin(self) -> float:
        """The quantity acceptance is decided on: min over thresholds of
        ``mean_delta - kappa * sigma_seed``. Accepted iff strictly positive."""
        return min(
            (s.margin(self.kappa) for s in self.per_threshold.values()),
            default=0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "kind": self.kind,
            "target": self.target,
            "kappa": self.kappa,
            "accepted": self.accepted,
            "reason": self.reason,
            "worst_delta": round(self.worst_delta, 6),
            "per_threshold": {k: s.to_dict() for k, s in self.per_threshold.items()},
        }


@dataclass
class PrescriptionPolicy:
    """The compiled optimization policy Delta(G) to be applied to the graph."""
    topic_splits: List[TopicSplit] = field(default_factory=list)
    node_reallocations: List[NodeReallocation] = field(default_factory=list)
    qos_upgrades: List[QosUpgrade] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            bucket: [edit.to_dict() for edit in getattr(self, bucket)]
            for bucket in _EDIT_BUCKETS.values()
        }

    def is_empty(self) -> bool:
        return not (self.topic_splits or self.node_reallocations or self.qos_upgrades)

    def edits(self) -> List[PolicyEdit]:
        """Flatten to a single ordered list for per-edit verification.

        The order — splits, then reallocations, then upgrades — is part of the
        contract: ``PrescribeService.prescribe`` zips verdicts against this list
        positionally, and ``applied_changes`` is rendered in the same order.
        """
        return [*self.topic_splits, *self.node_reallocations, *self.qos_upgrades]

    @classmethod
    def from_edits(cls, edits: List[PolicyEdit]) -> "PrescriptionPolicy":
        """Rebuild a policy from the subset of ``edits()`` that passed."""
        policy = cls()
        for edit in edits:
            getattr(policy, _EDIT_BUCKETS[type(edit)]).append(edit)
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
