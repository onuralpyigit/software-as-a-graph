"""
Public SDK Data Models

These models provide a stable, discoverable facade over the internal
SoftwareAsAGraph result structures.
"""
from typing import List, Optional, Any
from dataclasses import dataclass

from saag.analysis.models import LayerAnalysisResult as _LayerAnalysisResult, StructuralAnalysisResult as _StructuralAnalysisResult
from saag.prediction.models import QualityAnalysisResult as _QualityAnalysisResult, ComponentQuality as _ComponentQuality, DetectedProblem as _DetectedProblem
from saag.validation.models import LayerValidationResult as _LayerValidationResult
from saag.usecases.models import ImportStats as _ImportStats
from saag.core.criticality import CriticalityRanking, CompatNamespace


class ComponentFacade:
    """A developer-friendly wrapper around a component's quality metrics."""
    def __init__(self, inner: CriticalityRanking):
        self._inner = inner

    @property
    def id(self) -> str:
        """The component identifier."""
        return self._inner.id

    @property
    def rmav_score(self) -> float:
        """The overall RMAV criticality score."""
        return self._inner.overall
        
    @property
    def type(self) -> str:
        """The component type (Application, Broker, etc)."""
        return self._inner.type
        
    @property
    def is_critical(self) -> bool:
        """Is this component deemed critical?"""
        return self._inner.level.lower() == "critical"

    @property
    def name(self) -> str:
        """The logical name of the component."""
        return self._inner.name or self._inner.id

    @property
    def blast_radius(self) -> int:
        """How many nodes become unreachable if this node is removed."""
        return self._inner.blast_radius

    @property
    def cascade_depth(self) -> int:
        """Length of the longest reachable propagation path from this node."""
        return self._inner.cascade_depth

    @property
    def criticality_level(self) -> str:
        """The overall risk classification level as a string."""
        return self._inner.level

    @property
    def criticality_levels(self) -> dict:
        """The breakdown of criticality levels by dimension."""
        return dict(self._inner.levels)

    @property
    def scores(self) -> dict:
        """The breakdown of quality predicted scores."""
        return dict(self._inner.scores)

    @property
    def levels(self) -> dict:
        """Alias for criticality_levels to support backward compatibility."""
        return self.criticality_levels

    @property
    def structural(self) -> CompatNamespace:
        """Structural metrics mock/facade for backward compatibility."""
        return CompatNamespace(
            name=self.name,
            is_articulation_point=self._inner.is_articulation_point,
            blast_radius=self.blast_radius,
            cascade_depth=self.cascade_depth
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "is_critical": self.is_critical,
            "rmav_score": self.rmav_score,
            "criticality_level": self.criticality_level,
            "criticality_levels": self.criticality_levels,
            "scores": self.scores,
            "blast_radius": self.blast_radius,
            "cascade_depth": self.cascade_depth
        }


class AnalysisResult:
    """Result of the deterministic Analyze stage: structural metrics, RMAV dimension scores, Q(v), and anti-pattern problems."""
    def __init__(self, inner: _LayerAnalysisResult):
        self._inner = inner

    @property
    def raw(self) -> _LayerAnalysisResult:
        """Access the underlying internal model."""
        return self._inner

    @property
    def quality(self) -> _QualityAnalysisResult:
        """RMAV quality scores and criticality levels for all components (closed-form, deterministic)."""
        return self._inner.quality

    @property
    def critical_components(self) -> List["ComponentFacade"]:
        """Components classified as CRITICAL by the RMAV/Q scoring."""
        return [c for c in self.all_components if c.criticality_level.lower() == "critical"]

    @property
    def all_components(self) -> List["ComponentFacade"]:
        """All assessed components with their RMAV scores and criticality levels."""
        from saag.core.criticality import CriticalityRanking
        rankings = []
        for cq in self._inner.quality.components:
            scores = {
                "reliability": cq.scores.reliability,
                "maintainability": cq.scores.maintainability,
                "availability": cq.scores.availability,
                "security": cq.scores.security,
                "overall": cq.scores.overall,
            }
            levels = {
                "reliability": cq.levels.reliability.name if hasattr(cq.levels.reliability, "name") else str(cq.levels.reliability).upper(),
                "maintainability": cq.levels.maintainability.name if hasattr(cq.levels.maintainability, "name") else str(cq.levels.maintainability).upper(),
                "availability": cq.levels.availability.name if hasattr(cq.levels.availability, "name") else str(cq.levels.availability).upper(),
                "security": cq.levels.security.name if hasattr(cq.levels.security, "name") else str(cq.levels.security).upper(),
                "overall": cq.levels.overall.name if hasattr(cq.levels.overall, "name") else str(cq.levels.overall).upper(),
            }
            name = getattr(cq.structural, "name", cq.id) if cq.structural else cq.id
            blast_radius = getattr(cq.structural, "blast_radius", 0) if cq.structural else 0
            cascade_depth = getattr(cq.structural, "cascade_depth", 0) if cq.structural else 0
            is_articulation_point = getattr(cq.structural, "is_articulation_point", False) if cq.structural else False
            rankings.append(CriticalityRanking(
                id=cq.id,
                type=cq.type,
                scores=scores,
                levels=levels,
                overall=cq.scores.overall,
                level=levels["overall"],
                provenance="rmav",
                name=name,
                blast_radius=blast_radius,
                cascade_depth=cascade_depth,
                is_articulation_point=is_articulation_point
            ))
        return [ComponentFacade(r) for r in rankings]

    @property
    def problems(self) -> List[_DetectedProblem]:
        """Anti-patterns and architectural smells detected from Q(v) thresholds."""
        return self._inner.problems or []

    def save(self, filepath: str) -> None:
        """Export the analysis result to a JSON file."""
        import json
        from pathlib import Path
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self._inner.to_dict()
        with out.open("w") as f:
            json.dump(data, f, indent=2, default=str)


from types import SimpleNamespace


class PredictionResult:
    """Result of the inductive Predict stage: GNN-derived criticality ranks, attention weights, and ensemble-blended scores.

    This stage generalises beyond the closed-form RMAV composite by learning nonlinear
    interactions and multi-hop motifs that AHP-weighted scoring cannot encode.
    RMAV/Q scores (deterministic) are available on AnalysisResult, not here.
    """
    def __init__(self, inner: Any):
        if isinstance(inner, dict):
            # Parse from dictionary
            self._inner = SimpleNamespace(
                prediction_mode=inner.get("prediction_mode", "gnn_only"),
                node_scores={
                    k: SimpleNamespace(
                        component=v.get("component", k),
                        composite_score=v.get("composite_score", 0.0),
                        reliability_score=v.get("reliability_score", 0.0),
                        maintainability_score=v.get("maintainability_score", 0.0),
                        availability_score=v.get("availability_score", 0.0),
                        security_score=v.get("security_score", 0.0),
                        criticality_level=v.get("criticality_level", "MINIMAL"),
                    )
                    for k, v in inner.get("node_scores", {}).items()
                },
                ensemble_scores={
                    k: SimpleNamespace(
                        component=v.get("component", k),
                        composite_score=v.get("composite_score", 0.0),
                        reliability_score=v.get("reliability_score", 0.0),
                        maintainability_score=v.get("maintainability_score", 0.0),
                        availability_score=v.get("availability_score", 0.0),
                        security_score=v.get("security_score", 0.0),
                        criticality_level=v.get("criticality_level", "MINIMAL"),
                    )
                    for k, v in inner.get("ensemble_scores", {}).items()
                },
                edges=[
                    SimpleNamespace(
                        source=e.get("source"),
                        target=e.get("target"),
                        dependency_type=e.get("edge_type", "DEPENDS_ON"),
                        level=SimpleNamespace(value=e.get("criticality_level", "minimal")),
                        scores=SimpleNamespace(
                            reliability=e.get("reliability_score", 0.0),
                            maintainability=e.get("maintainability_score", 0.0),
                            availability=e.get("availability_score", 0.0),
                            security=e.get("security_score", 0.0),
                            overall=e.get("composite_score", 0.0),
                        )
                    )
                    for e in inner.get("edge_scores", [])
                ],
                _structural_cache=inner.get("_structural_cache", {})
            )
        else:
            self._inner = inner

    @property
    def critical_components(self) -> List[ComponentFacade]:
        """Components identified as CRITICAL by the prediction model."""
        return [c for c in self.all_components if c.criticality_level.lower() == "critical"]

    @property
    def all_components(self) -> List[ComponentFacade]:
        """All components ranked by criticality."""
        from saag.core.criticality import CriticalityRanking
        
        # Helper to map scores to level strings
        def _score_to_level_str(val: float) -> str:
            if val >= 0.75: return "critical"
            if val >= 0.55: return "high"
            if val >= 0.35: return "medium"
            return "low" if val >= 0.15 else "minimal"

        if hasattr(self._inner, "prediction_mode") or hasattr(self._inner, "node_scores"):
            # GNN path (GNNAnalysisResult or SimpleNamespace)
            source_dict = getattr(self._inner, "ensemble_scores", getattr(self._inner, "node_scores", {}))
            if not source_dict:
                source_dict = getattr(self._inner, "node_scores", {})
            provenance = getattr(self._inner, "prediction_mode", "gnn_only")
            if provenance == "gnn_only":
                provenance = "gnn"
            elif provenance == "rmav_only":
                provenance = "rmav"
                
            struct_cache = getattr(self._inner, "_structural_cache", {})
            rankings = []
            for score in source_dict.values():
                scores = {
                    "reliability": score.reliability_score,
                    "maintainability": score.maintainability_score,
                    "availability": score.availability_score,
                    "security": score.security_score,
                    "overall": score.composite_score,
                }
                levels = {
                    "reliability": _score_to_level_str(score.reliability_score).upper(),
                    "maintainability": _score_to_level_str(score.maintainability_score).upper(),
                    "availability": _score_to_level_str(score.availability_score).upper(),
                    "security": _score_to_level_str(score.security_score).upper(),
                    "overall": score.criticality_level.upper(),
                }
                s_dict = struct_cache.get(score.component, {}) if struct_cache else {}
                rankings.append(CriticalityRanking(
                    id=score.component,
                    type=s_dict.get("type", "Application"),
                    scores=scores,
                    levels=levels,
                    overall=score.composite_score,
                    level=levels["overall"],
                    provenance=provenance,
                    name=s_dict.get("name", score.component),
                    blast_radius=s_dict.get("blast_radius", 0),
                    cascade_depth=s_dict.get("cascade_depth", 0),
                    is_articulation_point=s_dict.get("is_articulation_point", False),
                ))
            # Sort by overall score descending
            rankings.sort(key=lambda x: x.overall, reverse=True)
            return [ComponentFacade(r) for r in rankings]
        else:
            # RMAV path (QualityAnalysisResult)
            rankings = []
            for cq in getattr(self._inner, "components", []):
                scores = {
                    "reliability": cq.scores.reliability,
                    "maintainability": cq.scores.maintainability,
                    "availability": cq.scores.availability,
                    "security": cq.scores.security,
                    "overall": cq.scores.overall,
                }
                levels = {
                    "reliability": cq.levels.reliability.name if hasattr(cq.levels.reliability, "name") else str(cq.levels.reliability).upper(),
                    "maintainability": cq.levels.maintainability.name if hasattr(cq.levels.maintainability, "name") else str(cq.levels.maintainability).upper(),
                    "availability": cq.levels.availability.name if hasattr(cq.levels.availability, "name") else str(cq.levels.availability).upper(),
                    "security": cq.levels.security.name if hasattr(cq.levels.security, "name") else str(cq.levels.security).upper(),
                    "overall": cq.levels.overall.name if hasattr(cq.levels.overall, "name") else str(cq.levels.overall).upper(),
                }
                name = getattr(cq.structural, "name", cq.id) if cq.structural else cq.id
                blast_radius = getattr(cq.structural, "blast_radius", 0) if cq.structural else 0
                cascade_depth = getattr(cq.structural, "cascade_depth", 0) if cq.structural else 0
                is_articulation_point = getattr(cq.structural, "is_articulation_point", False) if cq.structural else False
                rankings.append(CriticalityRanking(
                    id=cq.id,
                    type=cq.type,
                    scores=scores,
                    levels=levels,
                    overall=cq.scores.overall,
                    level=levels["overall"],
                    provenance="rmav",
                    name=name,
                    blast_radius=blast_radius,
                    cascade_depth=cascade_depth,
                    is_articulation_point=is_articulation_point
                ))
            # Sort by overall score descending
            rankings.sort(key=lambda x: x.overall, reverse=True)
            return [ComponentFacade(r) for r in rankings]

    @property
    def components(self) -> List[ComponentFacade]:
        """Alias for all_components to support backward compatibility."""
        return self.all_components

    @property
    def edges(self) -> List[Any]:
        """All prediction edges."""
        if hasattr(self._inner, "edges"):
            return list(self._inner.edges)
        return []

    @property
    def raw(self) -> Any:
        """Access the underlying result."""
        return self._inner

    def save(self, filepath: str) -> None:
        """Export the prediction result to a JSON file."""
        import json
        from pathlib import Path
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self._inner, "to_dict"):
            data = self._inner.to_dict()
        else:
            data = getattr(self._inner, "__dict__", self._inner)
        with out.open("w") as f:
            json.dump(data, f, indent=2, default=str)


class ValidationResult:
    """Result of comparing predictions against simulated ground truth."""
    def __init__(self, inner: _LayerValidationResult):
        self._inner = inner

    @property
    def spearman_rho(self) -> float:
        """The primary structural correlation (Spearman Rank)."""
        return self._inner.spearman

    @property
    def f1_score(self) -> float:
        """The primary classification F1 score."""
        return self._inner.f1_score
        
    @property
    def raw(self) -> _LayerValidationResult:
        """Access the underlying internal model."""
        return self._inner

    def save(self, filepath: str) -> None:
        """Export the validation result to a JSON file."""
        import json
        from pathlib import Path
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(self._inner.to_dict(), f, indent=2, default=str)


class ValidationPipelineFacade:
    """Result of comparing predictions against simulated ground truth across multiple layers."""
    def __init__(self, inner):
        self._inner = inner

    @property
    def layers(self) -> dict:
        return {k: ValidationResult(v) for k, v in self._inner.layers.items()}
        
    @property
    def raw(self):
        return self._inner
        
    def to_dict(self) -> dict:
        return self._inner.to_dict()

    def save(self, filepath: str) -> None:
        """Export the validation result to a JSON file."""
        import json
        from pathlib import Path
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class PipelineExecutionResult:
    """Aggregate result from running the full Pipeline sequentially.

    Stage mapping:
      analysis   — deterministic Analyze stage (structural metrics + RMAV/Q scores + anti-patterns)
      prediction — inductive Predict stage (GNN criticality ranks, ensemble blend); optional
      simulation — Simulate stage (counterfactual cascade ground truth)
      validation — Validate stage (Predict/Analyze vs Simulate ground truth)
    """
    analysis: Optional[AnalysisResult] = None
    prediction: Optional[PredictionResult] = None
    simulation: Optional[Any] = None
    validation: Optional[ValidationPipelineFacade] = None

    def save(self, filepath: str) -> None:
        """Export the full pipeline result to a JSON file."""
        import json
        from pathlib import Path
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        data = {}
        if self.analysis:
            # Reusing the structure defined in AnalysisResult.save()
            data["analysis"] = {
                "layer": getattr(self.analysis.raw.layer, "value", str(self.analysis.raw.layer)),
                "graph_summary": self.analysis.raw.graph_summary.to_dict() if hasattr(self.analysis.raw.graph_summary, "to_dict") else {},
                "components": {k: c.to_dict() for k, c in self.analysis.raw.components.items()},
            }
        if self.prediction:
            data["prediction"] = self.prediction.raw.to_dict()
        if self.simulation:
            if hasattr(self.simulation, "to_dict"):
                data["simulation"] = self.simulation.to_dict()
            else:
                data["simulation"] = self.simulation
        if self.validation:
            data["validation"] = self.validation.to_dict()
        with out.open("w") as f:
            json.dump(data, f, indent=2, default=str)
class ImportResult:
    """Result of a graph import operation."""
    def __init__(self, inner: _ImportStats):
        self._inner = inner

    @property
    def nodes_imported(self) -> int:
        """Number of nodes created or updated."""
        return self._inner.nodes_imported

    @property
    def edges_imported(self) -> int:
        """Number of relationships created or updated."""
        return self._inner.edges_imported

    @property
    def duration_ms(self) -> float:
        """Time taken for the import in milliseconds."""
        return self._inner.duration_ms

    @property
    def success(self) -> bool:
        """Whether the import was successful."""
        return self._inner.success

    @property
    def message(self) -> str:
        """Status message from the import process."""
        return self._inner.message

    @property
    def details(self) -> dict:
        """Additional raw statistics from the repository."""
        return self._inner.details

    def to_dict(self) -> dict:
        """Convert to a standardized dictionary representation."""
        return self._inner.to_dict()

    @property
    def raw(self) -> _ImportStats:
        """Access the underlying internal model."""
        return self._inner
