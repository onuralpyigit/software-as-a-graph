"""
Validation Metrics and Result Models
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class ValidationTargets:
    """Target thresholds for validation success as per unified gate table."""
    # Tier 1 — Primary Gates (all must pass for HIGH-BAR)
    spearman: float = 0.70             # G1: ρ ≥ 0.70
    f1_score: float = 0.75             # G2: F1 ≥ 0.75
    precision: float = 0.80            # G3: Precision ≥ 0.80
    top_5_overlap: float = 0.60        # G4: Top-5 ≥ 0.60

    # Tier 2 — Secondary Gates
    predictive_gain: float = 0.0       # G5: PG > 0
    weighted_kappa_cta: float = 0.70   # G6: κ_CTA ≥ 0.70
    cdcc_max: float = 0.30             # G7: CDCC < 0.30

    # Tier 3 — Dimension-Specific Specialist Gates
    bottleneck_precision_target: float = 0.70  # G8: BP ≥ 0.70
    ftr_max: float = 0.20                      # G9: FTR ≤ 0.20
    ahcr_5: float = 0.70                       # AHCR@5 ≥ 0.70
    spof_f1: float = 0.90                      # SPOF_F1 ≥ 0.90
    ccr_5: float = 0.80                        # CCR@5 ≥ 0.80

    # Reported only / Legacy compatibility / Additional targets
    spearman_p_max: float = 0.05
    rmse_max: float = 0.25
    mae_max: float = 0.20
    pearson: float = 0.65
    kendall: float = 0.50
    recall: float = 0.80
    accuracy: float = 0.75
    cohens_kappa: float = 0.60
    top_10_overlap: float = 0.50
    ndcg_10: float = 0.70
    
    # Reliability-specific
    cme: float = 0.10
    reliability_spearman: float = 0.75

    # Maintainability-specific
    maintainability_spearman: float = 0.70
    cocr_5: float = 0.75

    # Availability-specific
    availability_spearman: float = 0.80
    hsrr: float = 0.65
    dasa: float = 0.70
    rri: float = 0.80

    # Vulnerability-specific
    vulnerability_spearman: float = 0.70
    apar: float = 0.60
    cdcc: float = 0.40  # default label
    cdcc: float = 0.40  # default / general

    # Composite / Orthogonality
    composite_spearman: float = 0.85         # ρ(Q*(v), I*(v)) ≥ 0.85
    max_interdim_correlation: float = 0.40  # CDCC target

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in asdict(self).items() if isinstance(v, (float, int))}


@dataclass
class CorrelationMetrics:
    """Correlation coefficients with confidence intervals."""
    spearman: float = 0.0
    spearman_p: float = 1.0
    spearman_ci_lower: float = 0.0
    spearman_ci_upper: float = 0.0
    pearson: float = 0.0
    pearson_p: float = 1.0
    kendall: float = 0.0
    spearman_kendall_gap: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spearman": round(self.spearman, 4),
            "spearman_p_value": round(self.spearman_p, 6),
            "spearman_ci": [round(self.spearman_ci_lower, 4), round(self.spearman_ci_upper, 4)],
            "pearson": round(self.pearson, 4),
            "kendall": round(self.kendall, 4),
            "spearman_kendall_gap": round(self.spearman_kendall_gap, 4),
        }


@dataclass
class ErrorMetrics:
    """Error measurements including normalised RMSE."""
    rmse: float = 0.0
    nrmse: float = 0.0
    mae: float = 0.0
    mse: float = 0.0
    max_error: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


@dataclass
class ClassificationMetrics:
    """Binary classification metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    f1_ci_lower: float = 0.0
    f1_ci_upper: float = 0.0
    accuracy: float = 0.0
    cohens_kappa: float = 0.0
    auc_pr: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def confusion_matrix(self) -> Dict[str, int]:
        return {
            "tp": self.true_positives,
            "fp": self.false_positives,
            "tn": self.true_negatives,
            "fn": self.false_negatives,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "f1_ci": [round(self.f1_ci_lower, 4), round(self.f1_ci_upper, 4)],
            "accuracy": round(self.accuracy, 4),
            "cohens_kappa": round(self.cohens_kappa, 4),
            "auc_pr": round(self.auc_pr, 4),
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass
class RankingMetrics:
    """Ranking agreement metrics."""
    top_5_overlap: float = 0.0
    top_10_overlap: float = 0.0
    ndcg_5: float = 0.0
    ndcg_10: float = 0.0
    top_5_predicted: List[str] = field(default_factory=list)
    top_5_actual: List[str] = field(default_factory=list)
    top_5_common: List[str] = field(default_factory=list)
    top_5_ci_lower: float = 0.0
    top_5_ci_upper: float = 0.0
    # Reliability-specific ranking metrics
    ccr_5: float = 0.0   # Cascade Capture Rate @ 5
    cme: float = 0.0     # Cascade Magnitude Error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_5_overlap": round(self.top_5_overlap, 4),
            "top_5_ci": [round(self.top_5_ci_lower, 4), round(self.top_5_ci_upper, 4)],
            "top_10_overlap": round(self.top_10_overlap, 4),
            "ndcg_5": round(self.ndcg_5, 4),
            "ndcg_10": round(self.ndcg_10, 4),
            "top_5_agreement": {
                "predicted": self.top_5_predicted,
                "actual": self.top_5_actual,
                "common": self.top_5_common,
            },
            "ccr_5": round(self.ccr_5, 4),
            "cme": round(self.cme, 4),
        }


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
    """Validation result for a specific group."""
    group_name: str
    sample_size: int
    correlation: CorrelationMetrics
    error: ErrorMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    passed: bool
    gates: Dict[str, bool] = field(default_factory=dict)
    targets: ValidationTargets = field(default_factory=ValidationTargets)
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
        }


@dataclass
class ValidationResult:
    """Result for a validation run."""
    timestamp: str
    layer: str
    context: str
    targets: ValidationTargets
    overall: ValidationGroupResult
    by_type: Dict[str, ValidationGroupResult] = field(default_factory=dict)
    predicted_count: int = 0
    actual_count: int = 0
    matched_count: int = 0
    gates: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
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
            "gates": self.gates,
            "warnings": self.warnings,
        }


@dataclass
class LayerValidationResult:
    """Higher-level result for a layer."""
    layer: str
    layer_name: str
    predicted_components: int = 0
    simulated_components: int = 0
    matched_components: int = 0
    validation_result: Optional[ValidationResult] = None
    spearman: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    top_5_overlap: float = 0.0
    top_10_overlap: float = 0.0
    rmse: float = 0.0
    reliability_spearman: float = 0.0  # ρ(R(v), IR(v)) — reliability-specific correlation
    maintainability_spearman: float = 0.0  # ρ(M(v), IM(v)) — maintainability-specific correlation
    availability_spearman: float = 0.0  # ρ(A(v), IA(v)) — availability-specific correlation
    vulnerability_spearman: float = 0.0  # ρ(V(v), IV(v)) — vulnerability-specific correlation
    # Composite Q*(v) vs I*(v)
    composite_spearman: float = 0.0     # ρ(Q*(v), I*(v)) — the primary composite validation gate
    predictive_gain: float = 0.0        # PG = ρ_composite − max(dim ρ) > 0.03
    system_health: Dict[str, float] = field(default_factory=dict)
    # system_health keys: H_R, H_M, H_A, H_V, SRI, RCI
    passed: bool = False
    comparisons: List[ComponentComparison] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    component_names: Dict[str, str] = field(default_factory=dict)
    dimensional_validation: Dict[str, Any] = field(default_factory=dict)
    gates: Dict[str, bool] = field(default_factory=dict)
    node_type_stratified: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "layer_name": self.layer_name,
            "passed": self.passed,
            "summary": {
                "spearman": round(self.spearman, 4),
                "f1_score": round(self.f1_score, 4),
                "top_5_overlap": round(self.top_5_overlap, 4),
                "rmse": round(self.rmse, 4),
                "reliability_spearman": round(self.reliability_spearman, 4),
                "maintainability_spearman": round(self.maintainability_spearman, 4),
                "availability_spearman": round(self.availability_spearman, 4),
                "vulnerability_spearman": round(self.vulnerability_spearman, 4),
                "composite_spearman": round(self.composite_spearman, 4),
                "predictive_gain": round(self.predictive_gain, 4),
                "system_health": {k: round(v, 4) for k, v in self.system_health.items()},
            },
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "gates": self.gates,
            "node_type_stratified": self.node_type_stratified,
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
    targets: Optional[ValidationTargets] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def overall_passed(self) -> bool:
        return self.all_passed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "all_passed": self.all_passed,
            "total_components": self.total_components,
            "layers_passed": self.layers_passed,
            "targets": self.targets.to_dict() if self.targets else None,
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "warnings": self.warnings,
        }