"""
Validation Metrics and Result Models
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple


#: Gates whose conjunction defines `LayerValidationResult.passed`.
#: These three are the independent signals: a global rank correlation and two
#: set-overlap statistics at different K. Precision@K was retired because the
#: predicted and actual critical sets are both the top quartile, so precision,
#: recall and F1 are identically the same number -- the manuscript states this
#: too (sec6_experimental_setup.tex).
RELEASE_GATES: Tuple[str, ...] = ("spearman", "overlap_at_q3", "top5_overlap")

#: Reported alongside the release gates, never part of `passed`. A value of
#: None means the underlying dimension was never validated, which is distinct
#: from a measured failure.
REPORTED_GATES: Tuple[str, ...] = ("predictive_gain", "kappa_cta", "bottleneck_precision")


def evaluate_gate(
    value: Optional[float], threshold: float, strict: bool = False
) -> Optional[bool]:
    """Compare a metric against its threshold, preserving "never measured".

    Returns None when `value` is None. Coercing that to 0.0 would report an
    unmeasured dimension as a failed one -- which is what G6/G8 did while the
    maintainability ground truth was degenerate.
    """
    if value is None:
        return None
    return float(value) > threshold if strict else float(value) >= threshold


@dataclass
class ValidationTargets:
    """Thresholds the validation gates are evaluated against.

    Every field is read by a gate named in its comment, or by the web UI. Fields
    that no longer had a reader were removed rather than left to rot; see
    RELEASE_GATES / REPORTED_GATES for the gate set they serve.
    """

    # Release gates. `LayerValidationResult.passed` is their conjunction.
    spearman: float = 0.70              # spearman: rho(Q, I) >= 0.70
    f1_score: float = 0.75              # overlap_at_q3: top-quartile overlap >= 0.75
    top_5_overlap: float = 0.60         # top5_overlap: Top-5 overlap >= 0.60

    # Reported gates (not part of `passed`).
    #
    # PG = rho(Q*, I*) - max over dimensions of rho(dim, I*). It is negative on
    # every scenario measured, but the reason changed once the maintainability
    # oracle was repaired: M(v) correlates with its own ground truth IM(v) at
    # rho ~= -0.01, so blending 20% of it into Q* dilutes the reliability signal
    # that I* is mostly made of. 0.03 remains a specification of what a composite
    # would have to add to earn its place, not a value fitted to the data.
    predictive_gain: float = 0.03       # predictive_gain: PG > 0.03
    weighted_kappa_cta: float = 0.70    # kappa_cta: weighted Cohen's kappa >= 0.70
    bottleneck_precision_target: float = 0.70   # bottleneck_precision: BP >= 0.70

    # Rule-based and GNN forecasting acceptance thresholds.
    gnn_spearman: float = 0.85
    gnn_macro_f1: float = 0.88
    gnn_ndcg_10: float = 0.90

    # Reported only -- surfaced by the web UI, no gate reads them.
    precision: float = 0.80

    # Dimension orthogonality warning (logged, not gated).
    max_interdim_correlation: float = 0.40

    # Non-scalar configuration. `to_dict` filters these out, so they never reach
    # the API payload -- keep any new entry here non-numeric for that reason.
    # Deliberately EQUAL (0.5/0.5), not the scoring weights (0.80/0.20): the
    # ground-truth composite I*(v) must not be weighted by the same judgement
    # it validates, or rho(Q*, I*) would be partly circular.
    dimension_weights: Dict[str, float] = field(
        default_factory=lambda: {"r": 0.5, "m": 0.5}
    )
    node_type_rho: Dict[str, float] = field(
        default_factory=lambda: {"Application": 0.75, "Broker": 0.70, "Node": 0.65, "Library": 0.60}
    )
    node_type_rho_default: float = 0.70

    def to_dict(self) -> Dict[str, float]:
        """Scalar targets only -- the shape the API and web UI consume."""
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
    gates: Dict[str, Optional[bool]] = field(default_factory=dict)
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
    availability_spearman: float = 0.0  # ρ(A(v), IA(v)) — sub-characteristic diagnostic, not a gate input
    # Composite Q*(v) vs I*(v)
    #: None when I*(v) could not be built (every dimension degenerate).
    composite_spearman: Optional[float] = None   # ρ(Q*(v), I*(v))
    predictive_gain: Optional[float] = None      # PG = ρ_composite − max(dim ρ)
    system_health: Dict[str, float] = field(default_factory=dict)
    # system_health keys: H_R, H_M, H_FT, H_A, SRI, RCI (SRI sums only H_R, H_M)
    passed: bool = False
    #: True iff this layer never got a real validation run — analysis,
    #: prediction, or simulation raised before comparison could happen.
    #: `passed=False` alone doesn't distinguish "validated and failed" from
    #: "crashed before validating"; ValidationService.validate_layers sets
    #: this in its except branch.
    errored: bool = False
    comparisons: List[ComponentComparison] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    csc_names: Dict[str, str] = field(default_factory=dict)
    dimensional_validation: Dict[str, Any] = field(default_factory=dict)
    gates: Dict[str, Optional[bool]] = field(default_factory=dict)
    node_type_stratified: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    frequency_decile_stratified: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # New: Full scatter data per dimension for visualization
    # Dict mapping dimension name to List of (id, predicted_score, actual_impact, level)
    dimensional_scatter: Dict[str, List[Tuple[str, float, float, str]]] = field(default_factory=dict)
    # New: Confidence intervals per dimension
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    gnn_forecasting_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "layer_name": self.layer_name,
            "passed": self.passed,
            "errored": self.errored,
            "summary": {
                "spearman": round(self.spearman, 4),
                "f1_score": round(self.f1_score, 4),
                "top_5_overlap": round(self.top_5_overlap, 4),
                "rmse": round(self.rmse, 4),
                "reliability_spearman": round(self.reliability_spearman, 4),
                "maintainability_spearman": round(self.maintainability_spearman, 4),
                "availability_spearman": round(self.availability_spearman, 4),
                "composite_spearman": round(self.composite_spearman, 4) if self.composite_spearman is not None else None,
                "predictive_gain": round(self.predictive_gain, 4) if self.predictive_gain is not None else None,
                "system_health": {k: round(v, 4) for k, v in self.system_health.items()},
            },
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "gates": self.gates,
            "node_type_stratified": self.node_type_stratified,
            "frequency_decile_stratified": self.frequency_decile_stratified,
            "warnings": self.warnings,
            "gnn_forecasting_metrics": self.gnn_forecasting_metrics,
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