"""
Validation Metrics and Result Models
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class ValidationTargets:
    """Thresholds the validation gates are evaluated against.

    Every field below is read by something. The gate that consumes each one is
    named in its comment; fields with no gate are consumed by the frontend
    (``/api/v1/validation/targets``) or by the reported-only metrics.
    """

    # Tier 1 — primary gates. `LayerValidationResult.passed` is G1 ∧ G2 ∧ G3 ∧ G4.
    spearman: float = 0.70              # G1: ρ(Q, I) ≥ 0.70
    f1_score: float = 0.75              # G2: F1@K ≥ 0.75
    precision: float = 0.80             # G3: Precision@K ≥ 0.80
    top_5_overlap: float = 0.60         # G4: Top-5 overlap ≥ 0.60

    # Tier 2 — secondary gates (reported, not part of `passed`)
    predictive_gain: float = 0.03       # G5: PG > 0.03
    weighted_kappa_cta: float = 0.70    # G6: κ_CTA ≥ 0.70
    cdcc_max: float = 0.30              # G7: CDCC < 0.30

    # Tier 3 — dimension-specific specialist gates
    bottleneck_precision_target: float = 0.70   # G8: BP ≥ 0.70
    ftr_max: float = 0.20                       # G9: FTR ≤ 0.20

    # Per-group gates evaluated inside Validator._validate_group
    spearman_p_max: float = 0.05        # p_value_pass: p ≤ 0.05
    rmse_max: float = 0.25              # G5_rmse: RMSE ≤ 0.25 (also read by the web UI)

    # Rule-based baseline and GNN forecasting acceptance thresholds
    baseline_spearman: float = 0.85
    baseline_macro_f1: float = 0.88
    baseline_ndcg_10: float = 0.90

    # Security specialist metric parameters (not gates — inputs to FTR/APAR)
    security_v_threshold: float = 0.60
    security_reach_threshold: float = 0.10

    # Composite Q*(v) vs I*(v) and dimension orthogonality
    composite_spearman: float = 0.85        # ρ(Q*, I*) ≥ 0.85
    composite_f1: float = 0.90              # F1(Q*, I*) ≥ 0.90
    composite_top5_overlap: float = 0.80    # Top-5(Q*, I*) ≥ 0.80
    max_interdim_correlation: float = 0.40  # orthogonality warning threshold

    # Reported only — surfaced by the web UI, no gate reads them
    recall: float = 0.80
    pearson: float = 0.65
    kendall: float = 0.50
    top_10_overlap: float = 0.50

    # Non-scalar configuration. `to_dict` filters these out, so they never reach
    # the API payload — keep any new entry here non-numeric for that reason.
    dimension_weights: Dict[str, float] = field(
        default_factory=lambda: {"r": 0.25, "m": 0.25, "a": 0.25, "s": 0.25}
    )
    node_type_rho: Dict[str, float] = field(
        default_factory=lambda: {"Application": 0.75, "Broker": 0.70, "Node": 0.65, "Library": 0.60}
    )
    node_type_rho_default: float = 0.70

    def to_dict(self) -> Dict[str, float]:
        """Scalar targets only — the shape the API and web UI consume."""
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
    macro_f1: float = 0.0

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
            "macro_f1": round(self.macro_f1, 4),
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
    security_spearman: float = 0.0  # ρ(S(v), IS(v)) — security-specific correlation
    # Composite Q*(v) vs I*(v)
    composite_spearman: float = 0.0     # ρ(Q*(v), I*(v)) — the primary composite validation gate
    predictive_gain: float = 0.0        # PG = ρ_composite − max(dim ρ) > 0.03
    system_health: Dict[str, float] = field(default_factory=dict)
    # system_health keys: H_R, H_M, H_A, H_S, SRI, RCI
    passed: bool = False
    comparisons: List[ComponentComparison] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    csc_names: Dict[str, str] = field(default_factory=dict)
    dimensional_validation: Dict[str, Any] = field(default_factory=dict)
    gates: Dict[str, bool] = field(default_factory=dict)
    node_type_stratified: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    frequency_decile_stratified: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # New: Full scatter data per dimension for visualization
    # Dict mapping dimension name to List of (id, predicted_score, actual_impact, level)
    dimensional_scatter: Dict[str, List[Tuple[str, float, float, str]]] = field(default_factory=dict)
    # New: Confidence intervals per dimension
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    gnn_forecasting_metrics: Optional[Dict[str, Any]] = None
    rule_based_baseline_metrics: Optional[Dict[str, Any]] = None

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
                "security_spearman": round(self.security_spearman, 4),
                "composite_spearman": round(self.composite_spearman, 4),
                "predictive_gain": round(self.predictive_gain, 4),
                "system_health": {k: round(v, 4) for k, v in self.system_health.items()},
            },
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "gates": self.gates,
            "node_type_stratified": self.node_type_stratified,
            "frequency_decile_stratified": self.frequency_decile_stratified,
            "warnings": self.warnings,
            "gnn_forecasting_metrics": self.gnn_forecasting_metrics,
            "rule_based_baseline_metrics": self.rule_based_baseline_metrics,
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