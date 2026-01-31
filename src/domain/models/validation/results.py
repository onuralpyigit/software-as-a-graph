"""
Validation Results Domain Models
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from .metrics import ValidationTargets, CorrelationMetrics, ErrorMetrics, ClassificationMetrics, RankingMetrics

@dataclass
class ComponentComparison:
    """Comparison result for a single component."""
    id: str
    type: str
    predicted: float
    actual: float
    error: float
    predicted_critical: bool
    actual_critical: bool
    classification: str  # TP, FP, TN, FN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "predicted": round(self.predicted, 4),
            "actual": round(self.actual, 4),
            "error": round(self.error, 4),
            "predicted_critical": self.predicted_critical,
            "actual_critical": self.actual_critical,
            "classification": self.classification,
        }

@dataclass
class ValidationGroupResult:
    """Validation result for a specific group (overall, by type, by layer)."""
    group_name: str
    sample_size: int
    
    correlation: CorrelationMetrics
    error: ErrorMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    
    passed: bool
    targets: ValidationTargets = field(default_factory=ValidationTargets)
    
    # Component details
    components: List[ComponentComparison] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_name": self.group_name,
            "sample_size": self.sample_size,
            "passed": self.passed,
            "metrics": {
                "correlation": self.correlation.to_dict(),
                "error": self.error.to_dict(),
                "classification": self.classification.to_dict(),
                "ranking": self.ranking.to_dict(),
            },
            "summary": {
                "spearman": round(self.correlation.spearman, 3),
                "f1": round(self.classification.f1_score, 3),
                "precision": round(self.classification.precision, 3),
                "recall": round(self.classification.recall, 3),
                "rmse": round(self.error.rmse, 3),
                "top5_overlap": round(self.ranking.top_5_overlap, 3),
            },
        }

@dataclass
class ValidationResult:
    """Result for a validation run."""
    timestamp: str
    layer: str
    context: str
    targets: ValidationTargets
    
    # Overall result
    overall: ValidationGroupResult
    
    # Breakdown by component type
    by_type: Dict[str, ValidationGroupResult] = field(default_factory=dict)
    
    # Data alignment info
    predicted_count: int = 0
    actual_count: int = 0
    matched_count: int = 0
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """Check if overall validation passed."""
        return self.overall.passed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "layer": self.layer,
            "context": self.context,
            "passed": self.passed,
            "data_alignment": {
                "predicted_count": self.predicted_count,
                "actual_count": self.actual_count,
                "matched_count": self.matched_count,
            },
            "targets": self.targets.to_dict(),
            "overall": self.overall.to_dict(),
            "by_type": {k: v.to_dict() for k, v in self.by_type.items()},
            "warnings": self.warnings,
        }

@dataclass
class LayerValidationResult:
    """Higher-level result for CLI/Service consumption."""
    layer: str
    layer_name: str
    
    predicted_components: int = 0
    simulated_components: int = 0
    matched_components: int = 0
    
    validation_result: Optional[ValidationResult] = None
    
    # Summary metrics shortcuts
    spearman: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    top_5_overlap: float = 0.0
    rmse: float = 0.0
    
    passed: bool = False
    
    comparisons: List[ComponentComparison] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    component_names: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "layer_name": self.layer_name,
            "data": {
                "predicted_components": self.predicted_components,
                "simulated_components": self.simulated_components,
                "matched_components": self.matched_components,
            },
            "summary": {
                "passed": self.passed,
                "spearman": round(self.spearman, 4),
                "f1_score": round(self.f1_score, 4),
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "top_5_overlap": round(self.top_5_overlap, 4),
                "rmse": round(self.rmse, 4),
            },
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "warnings": self.warnings,
        }

@dataclass
class PipelineResult:
    """Complete validation pipeline result."""
    timestamp: str
    layers: Dict[str, LayerValidationResult] = field(default_factory=dict)
    total_components: int = 0
    layers_passed: int = 0
    all_passed: bool = False
    targets: ValidationTargets = field(default_factory=ValidationTargets)
    cross_layer_insights: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_components": self.total_components,
                "layers_validated": len(self.layers),
                "layers_passed": self.layers_passed,
                "all_passed": self.all_passed,
            },
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "targets": self.targets.to_dict(),
            "cross_layer_insights": self.cross_layer_insights,
        }
