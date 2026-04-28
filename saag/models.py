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


class ComponentFacade:
    """A developer-friendly wrapper around a component's quality metrics."""
    def __init__(self, inner: _ComponentQuality):
        self._inner = inner

    @property
    def id(self) -> str:
        """The component identifier."""
        return self._inner.id

    @property
    def rmav_score(self) -> float:
        """The overall RMAV criticality score."""
        return self._inner.scores.overall
        
    @property
    def type(self) -> str:
        """The component type (Application, Broker, etc)."""
        return self._inner.type
        
    @property
    def is_critical(self) -> bool:
        """Is this component deemed critical?"""
        return self._inner.is_critical

    @property
    def name(self) -> str:
        """The logical name of the component."""
        return getattr(self._inner.structural, 'name', self.id) if hasattr(self._inner, 'structural') and self._inner.structural else self.id

    @property
    def blast_radius(self) -> int:
        """How many nodes become unreachable if this node is removed."""
        return getattr(self._inner.structural, 'blast_radius', 0) if hasattr(self._inner, 'structural') and self._inner.structural else 0

    @property
    def cascade_depth(self) -> int:
        """Length of the longest reachable propagation path from this node."""
        return getattr(self._inner.structural, 'cascade_depth', 0) if hasattr(self._inner, 'structural') and self._inner.structural else 0

    @property
    def criticality_level(self) -> str:
        """The overall risk classification level as a string."""
        return self._inner.levels.overall.value if hasattr(self._inner.levels.overall, 'value') else str(self._inner.levels.overall)

    @property
    def criticality_levels(self) -> dict:
        """The breakdown of criticality levels by dimension."""
        return {
            "reliability": self._inner.levels.reliability.value if hasattr(self._inner.levels.reliability, 'value') else "unknown",
            "maintainability": self._inner.levels.maintainability.value if hasattr(self._inner.levels.maintainability, 'value') else "unknown",
            "availability": self._inner.levels.availability.value if hasattr(self._inner.levels.availability, 'value') else "unknown",
            "vulnerability": self._inner.levels.vulnerability.value if hasattr(self._inner.levels.vulnerability, 'value') else "unknown",
            "overall": self.criticality_level,
        }

    @property
    def scores(self) -> dict:
        """The breakdown of quality predicted scores."""
        return {
            "reliability": self._inner.scores.reliability,
            "maintainability": self._inner.scores.maintainability,
            "availability": self._inner.scores.availability,
            "vulnerability": self._inner.scores.vulnerability,
            "overall": self._inner.scores.overall,
        }

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
        return [ComponentFacade(c) for c in self._inner.quality.get_critical_components()]

    @property
    def all_components(self) -> List["ComponentFacade"]:
        """All assessed components with their RMAV scores and criticality levels."""
        return [ComponentFacade(c) for c in self._inner.quality.components]

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


class PredictionResult:
    """Result of the inductive Predict stage: GNN-derived criticality ranks, attention weights, and ensemble-blended scores.

    This stage generalises beyond the closed-form RMAV composite by learning nonlinear
    interactions and multi-hop motifs that AHP-weighted scoring cannot encode.
    RMAV/Q scores (deterministic) are available on AnalysisResult, not here.
    """
    def __init__(self, inner: Any):
        self._inner = inner

    @property
    def critical_components(self) -> List[ComponentFacade]:
        """Components identified as CRITICAL by the GNN/ensemble model."""
        source_dict = getattr(self._inner, "ensemble_scores", getattr(self._inner, "node_scores", {}))
        return [
            ComponentFacade(score)
            for score in source_dict.values()
            if getattr(score, "criticality_level", "") == "CRITICAL"
        ]

    @property
    def all_components(self) -> List[ComponentFacade]:
        """All components ranked by GNN/ensemble criticality."""
        source_dict = getattr(self._inner, "ensemble_scores", getattr(self._inner, "node_scores", {}))
        return [ComponentFacade(score) for score in source_dict.values()]

    @property
    def raw(self) -> Any:
        """Access the underlying GNNAnalysisResult."""
        return self._inner

    def save(self, filepath: str) -> None:
        """Export the prediction result to a JSON file."""
        import json
        from pathlib import Path
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(self._inner.to_dict(), f, indent=2, default=str)


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
