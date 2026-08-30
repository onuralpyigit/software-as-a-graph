"""
saag/analysis/triage.py

The Triage bridge: scopes Pathway A's (RM) root-cause diagnosis to the
components Pathway B (GNN blast-radius ranking, or RM itself in cold start)
flagged as the Top-K most critical. Pure and read-only — no repository, no
I/O, no saag.simulation import (Simulation is an independent oracle, never a
triage input; see ARCHITECTURE.md's Independence Guarantee).

Joins by component id rather than reading a root cause off the ranked
object: a GNNAnalysisResult's ``components`` shim leaves fault_tolerance/
availability at 0.0 and ``profile`` at None (those are RM-only outputs), so
the diagnosis always comes from the RM substrate — either the standalone RM
result in cold start, or the RM result the GNN ranking was produced
alongside (``GNNAnalysisResult.rm_result``).
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from saag.explanation.engine import ExplanationEngine, resolve_roles

logger = logging.getLogger(__name__)


@dataclass
class TriageEntry:
    """One shortlisted component: Pathway B's ranking joined to Pathway A's
    root-cause diagnosis for that same id."""
    component_id: str
    rank: int                                          # 1-based
    ranking_score: float                                # B's score, or RM Q*(v) in cold start
    component_type: str
    pattern: str                                        # CriticalityProfile.pattern (root cause)
    level: str                                          # RM overall CriticalityLevel, e.g. "CRITICAL"
    elevated_dimensions: List[Dict[str, Any]] = field(default_factory=list)
    priority_action: str = ""
    roles: List[str] = field(default_factory=list)      # SRE / DevOps / Architect / Security

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "rank": self.rank,
            "ranking_score": round(self.ranking_score, 4),
            "component_type": self.component_type,
            "pattern": self.pattern,
            "level": self.level,
            "elevated_dimensions": self.elevated_dimensions,
            "priority_action": self.priority_action,
            "roles": self.roles,
        }


@dataclass
class TriageResult:
    """Pathway B's Top-K shortlist, each entry annotated with Pathway A's
    root-cause diagnosis. See ``triage()``."""
    layer: str
    k: int
    ranking_source: str      # "gnn" | "rm"
    population: int          # size of the ranked population before Top-K selection
    entries: List[TriageEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "k": self.k,
            "ranking_source": self.ranking_source,
            "population": self.population,
            "entries": [e.to_dict() for e in self.entries],
        }


def select_top_k(
    prediction_result: Any,
    k: int,
    node_types: Optional[Sequence[str]] = None,
) -> List[Tuple[str, float]]:
    """Return the Top-K (component_id, score) pairs from a prediction
    result's ``.components``, sorted by ``scores.overall`` descending.

    Works for both ``QualityAnalysisResult`` (RM) and ``GNNAnalysisResult``
    (GNN) — both expose ``.components -> List[ComponentQuality]``. Ties break
    on id ascending so the shortlist is deterministic. ``k`` is clamped to
    the (optionally type-filtered) population; ``k <= 0`` is a caller error.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    components = prediction_result.components
    if node_types is not None:
        types = set(node_types)
        components = [c for c in components if c.type in types]

    ranked = sorted(components, key=lambda c: (-c.scores.overall, c.id))
    return [(c.id, c.scores.overall) for c in ranked[:k]]


def triage(
    prediction_result: Any,
    k: int = 10,
    layer: str = "system",
    node_types: Optional[Sequence[str]] = None,
) -> TriageResult:
    """Shortlist the Top-K critical components from *prediction_result* and
    attach each one's RM root-cause diagnosis (pattern, elevated dimensions,
    priority action, stakeholder roles).

    ``prediction_result`` may be a standalone RM ``QualityAnalysisResult``
    (cold start — no trained GNN checkpoint) or a ``GNNAnalysisResult``. In
    the latter case the diagnosis is read from ``prediction_result.rm_result``
    (the RM pass the GNN ranking was produced alongside), never from the
    GNN result's own ``.components`` shim.
    """
    selected = select_top_k(prediction_result, k, node_types=node_types)

    rm_substrate = getattr(prediction_result, "rm_result", None) or prediction_result

    prediction_mode = getattr(prediction_result, "prediction_mode", "rm") or "rm"
    ranking_source = "gnn" if prediction_mode.startswith("gnn") else "rm"

    components_by_id = {c.id: c for c in rm_substrate.components}
    problems_by_id: Dict[str, list] = defaultdict(list)
    for problem in getattr(rm_substrate, "problems", None) or []:
        problems_by_id[problem.entity_id].append(problem)

    engine = ExplanationEngine()
    entries: List[TriageEntry] = []
    for rank, (component_id, score) in enumerate(selected, start=1):
        cq = components_by_id.get(component_id)
        if cq is None:
            logger.warning(
                "Triage: component '%s' ranked by Pathway B but absent from the "
                "RM substrate; skipping (the two pathways were run on different "
                "populations).", component_id,
            )
            continue

        exp = engine.explain_component(cq, problems_by_id.get(component_id, []))
        entries.append(TriageEntry(
            component_id=component_id,
            rank=rank,
            ranking_score=score,
            component_type=cq.type,
            pattern=exp.pattern,
            level=exp.level,
            elevated_dimensions=[d.to_dict() for d in exp.dimensions],
            priority_action=exp.priority_action,
            roles=resolve_roles(exp),
        ))

    return TriageResult(
        layer=layer,
        k=k,
        ranking_source=ranking_source,
        population=len(prediction_result.components),
        entries=entries,
    )
