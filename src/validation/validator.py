"""
Validator

Compares Predicted Scores (from Graph Analysis) against Actual Impact
(from Failure Simulation) using statistical validation metrics.

Validation Framework:
    1. Correlation Analysis - Do rankings correlate?
    2. Error Analysis - How accurate are the scores?
    3. Classification Analysis - Can we detect critical components?
    4. Ranking Analysis - Do top-K lists agree?

Validation Levels:
    - Overall: All components together
    - By Type: Separate validation per component type
    - By Layer: Separate validation per graph layer
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .metrics import (
    ValidationTargets,
    CorrelationMetrics,
    ErrorMetrics,
    ClassificationMetrics,
    RankingMetrics,
    spearman_correlation,
    pearson_correlation,
    kendall_correlation,
    calculate_error_metrics,
    calculate_classification_metrics,
    calculate_ranking_metrics,
)


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
    
    def get_pass_fail_details(self) -> Dict[str, Tuple[float, float, bool]]:
        """Get (actual, target, passed) for each metric."""
        return {
            "spearman": (self.correlation.spearman, self.targets.spearman,
                        self.correlation.spearman >= self.targets.spearman),
            "f1_score": (self.classification.f1_score, self.targets.f1_score,
                        self.classification.f1_score >= self.targets.f1_score),
            "precision": (self.classification.precision, self.targets.precision,
                         self.classification.precision >= self.targets.precision),
            "recall": (self.classification.recall, self.targets.recall,
                      self.classification.recall >= self.targets.recall),
            "top_5_overlap": (self.ranking.top_5_overlap, self.targets.top_5_overlap,
                             self.ranking.top_5_overlap >= self.targets.top_5_overlap),
            "rmse": (self.error.rmse, self.targets.rmse_max,
                    self.error.rmse <= self.targets.rmse_max),
        }


@dataclass
class ValidationResult:
    """Complete validation result for a layer."""
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


class Validator:
    """
    Validates graph analysis predictions against simulation results.
    
    Uses statistical metrics to compare:
        - Predicted criticality scores (from graph analysis)
        - Actual impact scores (from failure simulation)
    
    Validation criteria (defaults):
        - Correlation: Spearman >= 0.70
        - Classification: F1 >= 0.80
        - Ranking: Top-5 overlap >= 0.60
    
    Example:
        >>> validator = Validator()
        >>> result = validator.validate(
        ...     predicted_scores={"A": 0.8, "B": 0.6},
        ...     actual_scores={"A": 0.85, "B": 0.55},
        ... )
        >>> print(f"Passed: {result.passed}")
    """
    
    def __init__(
        self,
        targets: Optional[ValidationTargets] = None,
        critical_percentile: float = 75.0
    ):
        """
        Initialize validator.
        
        Args:
            targets: Validation success criteria
            critical_percentile: Percentile threshold for critical classification
        """
        self.targets = targets or ValidationTargets()
        self.critical_percentile = critical_percentile
        self.logger = logging.getLogger(__name__)
    
    def validate(
        self,
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        component_types: Optional[Dict[str, str]] = None,
        layer: str = "system",
        context: str = "Validation"
    ) -> ValidationResult:
        """
        Main validation method.
        
        Args:
            predicted_scores: Dict mapping component ID to predicted score
            actual_scores: Dict mapping component ID to actual impact score
            component_types: Dict mapping component ID to type (for breakdown)
            layer: Analysis layer name
            context: Context description
            
        Returns:
            ValidationResult with overall and per-type metrics
        """
        timestamp = datetime.now().isoformat()
        warnings = []
        
        # Data alignment
        pred_ids = set(predicted_scores.keys())
        actual_ids = set(actual_scores.keys())
        common_ids = pred_ids & actual_ids
        
        if len(common_ids) < len(pred_ids):
            missing = len(pred_ids) - len(common_ids)
            warnings.append(f"{missing} predicted components not in actual scores")
        
        if len(common_ids) < len(actual_ids):
            extra = len(actual_ids) - len(common_ids)
            warnings.append(f"{extra} actual components not in predictions")
        
        if len(common_ids) < 3:
            warnings.append("Insufficient data (n < 3)")
            return ValidationResult(
                timestamp=timestamp,
                layer=layer,
                context=context,
                targets=self.targets,
                overall=self._empty_group_result("Overall"),
                predicted_count=len(pred_ids),
                actual_count=len(actual_ids),
                matched_count=len(common_ids),
                warnings=warnings,
            )
        
        # Filter to common IDs
        pred_filtered = {k: predicted_scores[k] for k in common_ids}
        actual_filtered = {k: actual_scores[k] for k in common_ids}
        types_filtered = {k: component_types.get(k, "Unknown") for k in common_ids} if component_types else {}
        
        self.logger.info(f"Validating {len(common_ids)} components for layer '{layer}'")
        
        # 1. Overall validation
        overall = self._validate_group(
            "Overall",
            pred_filtered,
            actual_filtered,
            types_filtered,
        )
        
        # 2. Per-type validation
        by_type: Dict[str, ValidationGroupResult] = {}
        
        if types_filtered:
            type_groups: Dict[str, List[str]] = {}
            for comp_id, comp_type in types_filtered.items():
                if comp_type not in type_groups:
                    type_groups[comp_type] = []
                type_groups[comp_type].append(comp_id)
            
            for comp_type, comp_ids in type_groups.items():
                if len(comp_ids) >= 3:
                    pred_type = {k: pred_filtered[k] for k in comp_ids}
                    actual_type = {k: actual_filtered[k] for k in comp_ids}
                    types_type = {k: types_filtered[k] for k in comp_ids}
                    
                    by_type[comp_type] = self._validate_group(
                        comp_type,
                        pred_type,
                        actual_type,
                        types_type,
                    )
        
        return ValidationResult(
            timestamp=timestamp,
            layer=layer,
            context=context,
            targets=self.targets,
            overall=overall,
            by_type=by_type,
            predicted_count=len(pred_ids),
            actual_count=len(actual_ids),
            matched_count=len(common_ids),
            warnings=warnings,
        )
    
    def _validate_group(
        self,
        group_name: str,
        predicted: Dict[str, float],
        actual: Dict[str, float],
        types: Dict[str, str],
    ) -> ValidationGroupResult:
        """Validate a group of components."""
        n = len(predicted)
        
        # Convert to lists
        ids = list(predicted.keys())
        pred_values = [predicted[k] for k in ids]
        actual_values = [actual[k] for k in ids]
        
        # 1. Correlation metrics
        spearman_rho, spearman_p = spearman_correlation(pred_values, actual_values)
        pearson_r, pearson_p = pearson_correlation(pred_values, actual_values)
        kendall_tau = kendall_correlation(pred_values, actual_values)
        
        correlation = CorrelationMetrics(
            spearman=spearman_rho,
            spearman_p=spearman_p,
            pearson=pearson_r,
            pearson_p=pearson_p,
            kendall=kendall_tau,
        )
        
        # 2. Error metrics
        error = calculate_error_metrics(pred_values, actual_values)
        
        # 3. Classification metrics
        pred_threshold = self._percentile(pred_values, self.critical_percentile)
        actual_threshold = self._percentile(actual_values, self.critical_percentile)
        
        pred_critical = [v >= pred_threshold for v in pred_values]
        actual_critical = [v >= actual_threshold for v in actual_values]
        
        classification = calculate_classification_metrics(pred_critical, actual_critical)
        
        # 4. Ranking metrics
        ranking = calculate_ranking_metrics(predicted, actual)
        
        # 5. Component comparisons
        components = []
        for i, comp_id in enumerate(ids):
            p_crit = pred_critical[i]
            a_crit = actual_critical[i]
            
            if p_crit and a_crit:
                cls = "TP"
            elif p_crit and not a_crit:
                cls = "FP"
            elif not p_crit and a_crit:
                cls = "FN"
            else:
                cls = "TN"
            
            components.append(ComponentComparison(
                id=comp_id,
                type=types.get(comp_id, "Unknown"),
                predicted=pred_values[i],
                actual=actual_values[i],
                error=abs(pred_values[i] - actual_values[i]),
                predicted_critical=p_crit,
                actual_critical=a_crit,
                classification=cls,
            ))
        
        # Sort by error (descending)
        components.sort(key=lambda x: x.error, reverse=True)
        
        # Determine pass/fail
        passed = (
            correlation.spearman >= self.targets.spearman and
            classification.f1_score >= self.targets.f1_score and
            ranking.top_5_overlap >= self.targets.top_5_overlap
        )
        
        return ValidationGroupResult(
            group_name=group_name,
            sample_size=n,
            correlation=correlation,
            error=error,
            classification=classification,
            ranking=ranking,
            passed=passed,
            targets=self.targets,
            components=components,
        )
    
    def _empty_group_result(self, group_name: str) -> ValidationGroupResult:
        """Create an empty group result."""
        return ValidationGroupResult(
            group_name=group_name,
            sample_size=0,
            correlation=CorrelationMetrics(),
            error=ErrorMetrics(),
            classification=ClassificationMetrics(),
            ranking=RankingMetrics(),
            passed=False,
            targets=self.targets,
        )
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100.0
        f = int(k)
        c = min(f + 1, len(sorted_values) - 1)
        
        if f == c:
            return sorted_values[f]
        
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)