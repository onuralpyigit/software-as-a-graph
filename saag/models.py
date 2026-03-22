"""
Public SDK Data Models

These models provide a stable, discoverable facade over the internal
SoftwareAsAGraph result structures.
"""
from typing import List, Optional, Any
from dataclasses import dataclass

# We use TYPE_CHECKING or direct imports since saag/__init__.py 
# will have injected the backend into sys.path before this is imported.
from src.analysis.models import StructuralAnalysisResult as _StructuralAnalysisResult
from src.prediction.models import QualityAnalysisResult as _QualityAnalysisResult, ComponentQuality as _ComponentQuality, DetectedProblem as _DetectedProblem
from src.validation.models import LayerValidationResult as _LayerValidationResult


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


class AnalysisResult:
    """Result of the structural graph analysis step."""
    def __init__(self, inner: _StructuralAnalysisResult):
        self._inner = inner
    
    @property
    def raw(self) -> _StructuralAnalysisResult:
        """Access the underlying internal model."""
        return self._inner

    def save(self, filepath: str) -> None:
        """Export the analysis result to a JSON file."""
        import json
        from pathlib import Path
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "layer": getattr(self._inner.layer, "value", str(self._inner.layer)),
            "graph_summary": self._inner.graph_summary.to_dict() if hasattr(self._inner.graph_summary, "to_dict") else {},
            "components": {k: c.to_dict() for k, c in self._inner.components.items()},
            "edges": {f"{k[0]}->{k[1]}": e.to_dict() for k, e in self._inner.edges.items()},
            "qos_profile": self._inner.qos_profile,
            "rcm_order": self._inner.rcm_order
        }
        with out.open("w") as f:
            json.dump(data, f, indent=2, default=str)


class PredictionResult:
    """Result of the GNN quality prediction step."""
    def __init__(self, inner: _QualityAnalysisResult):
        self._inner = inner

    @property
    def critical_components(self) -> List[ComponentFacade]:
        """Components identified as CRITICAL by the prediction model."""
        return [ComponentFacade(c) for c in self._inner.get_critical_components()]
        
    @property
    def all_components(self) -> List[ComponentFacade]:
        """All assessed components in this layer."""
        return [ComponentFacade(c) for c in self._inner.components]

    @property
    def raw(self) -> _QualityAnalysisResult:
        """Access the underlying internal model."""
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


@dataclass
class PipelineExecutionResult:
    """Aggregate result from running the full Pipeline sequentially."""
    analysis: Optional[AnalysisResult] = None
    prediction: Optional[PredictionResult] = None
    validation: Optional[ValidationResult] = None
    problems: Optional[List[_DetectedProblem]] = None

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
        if self.validation:
            data["validation"] = self.validation.raw.to_dict()
        if self.problems:
            data["problems"] = [p.to_dict() for p in self.problems]
            
        with out.open("w") as f:
            json.dump(data, f, indent=2, default=str)
