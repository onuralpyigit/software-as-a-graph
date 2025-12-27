"""
Graph Validator - Version 4.0

Validates graph-based criticality predictions by comparing against
actual impact scores from failure simulation.

Validation Approach:
1. Correlation Analysis: Spearman rank correlation (target ≥ 0.70)
2. Classification Metrics: F1-Score (≥ 0.90), Precision/Recall (≥ 0.80)
3. Ranking Analysis: Top-k overlap for critical component identification
4. Statistical Confidence: Bootstrap confidence intervals, p-values

This module answers the key research question: Do topological metrics
accurately predict actual system impact when components fail?

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .metrics import (
    ValidationStatus,
    MetricStatus,
    CorrelationMetrics,
    ConfusionMatrix,
    RankingMetrics,
    BootstrapCI,
    ValidationTargets,
    calculate_correlation,
    calculate_confusion,
    calculate_ranking,
    bootstrap_confidence_interval,
    spearman,
    percentile,
    std_dev,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComponentValidation:
    """Validation details for a single component"""
    component_id: str
    component_type: str
    predicted_score: float
    actual_impact: float
    predicted_rank: int
    actual_rank: int
    rank_difference: int
    predicted_critical: bool
    actual_critical: bool
    correct: bool

    def to_dict(self) -> Dict:
        return {
            "id": self.component_id,
            "type": self.component_type,
            "predicted_score": round(self.predicted_score, 4),
            "actual_impact": round(self.actual_impact, 4),
            "predicted_rank": self.predicted_rank,
            "actual_rank": self.actual_rank,
            "rank_diff": self.rank_difference,
            "predicted_critical": self.predicted_critical,
            "actual_critical": self.actual_critical,
            "correct": self.correct,
        }


@dataclass
class ValidationResult:
    """Complete validation result"""
    timestamp: datetime
    status: ValidationStatus
    n_components: int
    
    # Core metrics
    correlation: CorrelationMetrics
    classification: ConfusionMatrix
    ranking: RankingMetrics
    
    # Targets and achievements
    targets: ValidationTargets
    achieved: Dict[str, Tuple[float, MetricStatus]]
    
    # Component details
    components: List[ComponentValidation]
    
    # Misclassified
    false_positives: List[str]
    false_negatives: List[str]
    
    # Optional bootstrap
    bootstrap: Optional[List[BootstrapCI]] = None

    def to_dict(self) -> Dict:
        result = {
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "n_components": self.n_components,
            "correlation": self.correlation.to_dict(),
            "classification": self.classification.to_dict(),
            "ranking": self.ranking.to_dict(),
            "targets": self.targets.to_dict(),
            "achieved": {
                k: {"value": round(v[0], 4), "status": v[1].value}
                for k, v in self.achieved.items()
            },
            "misclassified": {
                "false_positives": self.false_positives[:10],
                "false_negatives": self.false_negatives[:10],
            },
            "summary": self.summary(),
        }
        
        if self.bootstrap:
            result["bootstrap"] = [b.to_dict() for b in self.bootstrap]
        
        return result

    def summary(self) -> Dict:
        """Generate summary statistics"""
        met = sum(1 for _, (_, s) in self.achieved.items() if s == MetricStatus.MET)
        total = len(self.achieved)
        
        return {
            "metrics_met": f"{met}/{total}",
            "pass_rate": round(met / total, 2) if total > 0 else 0,
            "spearman": round(self.correlation.spearman, 4),
            "f1": round(self.classification.f1, 4),
            "precision": round(self.classification.precision, 4),
            "recall": round(self.classification.recall, 4),
            "top_5": round(self.ranking.top_k_overlap.get(5, 0), 4),
        }


# =============================================================================
# Validator
# =============================================================================

class Validator:
    """
    Validates graph-based predictions against simulation results.
    
    Compares:
    - Predicted criticality scores from graph analysis
    - Actual impact scores from failure simulation
    
    Uses statistical methods to determine if predictions are reliable.
    """

    def __init__(
        self,
        targets: Optional[ValidationTargets] = None,
        critical_percentile: float = 75,
        seed: Optional[int] = None,
    ):
        """
        Initialize the validator.
        
        Args:
            targets: Validation target metrics
            critical_percentile: Percentile for critical threshold (default: top 25%)
            seed: Random seed for reproducibility
        """
        self.targets = targets or ValidationTargets()
        self.critical_percentile = critical_percentile
        self.seed = seed
        self.logger = logging.getLogger(__name__)

    def validate(
        self,
        predicted: Dict[str, float],
        actual: Dict[str, float],
        component_types: Optional[Dict[str, str]] = None,
    ) -> ValidationResult:
        """
        Validate predicted scores against actual impacts.
        
        Args:
            predicted: Component ID -> predicted criticality score
            actual: Component ID -> actual impact score (from simulation)
            component_types: Component ID -> type (optional)
        
        Returns:
            ValidationResult with all metrics
        """
        # Find common components
        common = list(set(predicted.keys()) & set(actual.keys()))
        n = len(common)
        
        self.logger.info(f"Validating {n} components")
        
        if n < 5:
            return self._insufficient_data_result(n)
        
        # Prepare data
        pred_list = [predicted[c] for c in common]
        actual_list = [actual[c] for c in common]
        
        # Calculate thresholds (top 25% are critical)
        pred_threshold = percentile(pred_list, self.critical_percentile)
        actual_threshold = percentile(actual_list, self.critical_percentile)
        
        # Calculate metrics
        correlation = calculate_correlation(pred_list, actual_list)
        
        confusion, fp_list, fn_list = calculate_confusion(
            predicted, actual, pred_threshold, actual_threshold
        )
        
        ranking = calculate_ranking(predicted, actual)
        
        # Build component validations
        components = self._build_component_validations(
            common, predicted, actual,
            pred_threshold, actual_threshold,
            component_types or {},
        )
        
        # Evaluate against targets
        achieved = self._evaluate_targets(correlation, confusion, ranking)
        
        # Determine status
        status = self._determine_status(achieved)
        
        return ValidationResult(
            timestamp=datetime.now(),
            status=status,
            n_components=n,
            correlation=correlation,
            classification=confusion,
            ranking=ranking,
            targets=self.targets,
            achieved=achieved,
            components=components,
            false_positives=fp_list,
            false_negatives=fn_list,
        )

    def validate_with_bootstrap(
        self,
        predicted: Dict[str, float],
        actual: Dict[str, float],
        component_types: Optional[Dict[str, str]] = None,
        n_iterations: int = 1000,
        confidence: float = 0.95,
    ) -> ValidationResult:
        """
        Validate with bootstrap confidence intervals.
        
        Args:
            predicted: Predicted scores
            actual: Actual impacts
            component_types: Component types
            n_iterations: Bootstrap iterations
            confidence: Confidence level
        
        Returns:
            ValidationResult with bootstrap confidence intervals
        """
        # First run standard validation
        result = self.validate(predicted, actual, component_types)
        
        if result.status == ValidationStatus.INSUFFICIENT:
            return result
        
        # Prepare data for bootstrap
        common = list(set(predicted.keys()) & set(actual.keys()))
        pred_list = [predicted[c] for c in common]
        actual_list = [actual[c] for c in common]
        
        # Bootstrap for Spearman
        def spearman_fn(x, y):
            r, _ = spearman(x, y)
            return r
        
        spearman_ci = bootstrap_confidence_interval(
            pred_list, actual_list, spearman_fn,
            n_iterations=n_iterations, confidence=confidence, seed=self.seed
        )
        spearman_ci.metric = "spearman"
        
        # Bootstrap for F1 (simplified)
        pred_thresh = percentile(pred_list, self.critical_percentile)
        actual_thresh = percentile(actual_list, self.critical_percentile)
        
        def f1_fn(x, y):
            n = len(x)
            pt = percentile(x, self.critical_percentile)
            at = percentile(y, self.critical_percentile)
            
            tp = sum(1 for px, py in zip(x, y) if px >= pt and py >= at)
            fp = sum(1 for px, py in zip(x, y) if px >= pt and py < at)
            fn = sum(1 for px, py in zip(x, y) if px < pt and py >= at)
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        f1_ci = bootstrap_confidence_interval(
            pred_list, actual_list, f1_fn,
            n_iterations=n_iterations, confidence=confidence, seed=self.seed
        )
        f1_ci.metric = "f1"
        
        result.bootstrap = [spearman_ci, f1_ci]
        return result

    def _insufficient_data_result(self, n: int) -> ValidationResult:
        """Return result for insufficient data"""
        return ValidationResult(
            timestamp=datetime.now(),
            status=ValidationStatus.INSUFFICIENT,
            n_components=n,
            correlation=CorrelationMetrics(0, 1, 0, 1, 0, n),
            classification=ConfusionMatrix(0, 0, 0, 0, 0),
            ranking=RankingMetrics({}, 0, 0, 0),
            targets=self.targets,
            achieved={},
            components=[],
            false_positives=[],
            false_negatives=[],
        )

    def _build_component_validations(
        self,
        components: List[str],
        predicted: Dict[str, float],
        actual: Dict[str, float],
        pred_threshold: float,
        actual_threshold: float,
        types: Dict[str, str],
    ) -> List[ComponentValidation]:
        """Build per-component validation details"""
        # Rank components
        pred_ranked = sorted(components, key=lambda c: -predicted[c])
        actual_ranked = sorted(components, key=lambda c: -actual[c])
        
        pred_rank = {c: i + 1 for i, c in enumerate(pred_ranked)}
        actual_rank = {c: i + 1 for i, c in enumerate(actual_ranked)}
        
        validations = []
        for comp in components:
            pred_crit = predicted[comp] >= pred_threshold
            actual_crit = actual[comp] >= actual_threshold
            
            validations.append(ComponentValidation(
                component_id=comp,
                component_type=types.get(comp, "Unknown"),
                predicted_score=predicted[comp],
                actual_impact=actual[comp],
                predicted_rank=pred_rank[comp],
                actual_rank=actual_rank[comp],
                rank_difference=abs(pred_rank[comp] - actual_rank[comp]),
                predicted_critical=pred_crit,
                actual_critical=actual_crit,
                correct=(pred_crit == actual_crit),
            ))
        
        # Sort by actual impact (most critical first)
        validations.sort(key=lambda v: -v.actual_impact)
        return validations

    def _evaluate_targets(
        self,
        correlation: CorrelationMetrics,
        confusion: ConfusionMatrix,
        ranking: RankingMetrics,
    ) -> Dict[str, Tuple[float, MetricStatus]]:
        """Evaluate achieved metrics against targets"""
        achieved = {}
        
        # Spearman
        achieved["spearman"] = (
            correlation.spearman,
            self._metric_status(correlation.spearman, self.targets.spearman),
        )
        
        # F1
        achieved["f1"] = (
            confusion.f1,
            self._metric_status(confusion.f1, self.targets.f1),
        )
        
        # Precision
        achieved["precision"] = (
            confusion.precision,
            self._metric_status(confusion.precision, self.targets.precision),
        )
        
        # Recall
        achieved["recall"] = (
            confusion.recall,
            self._metric_status(confusion.recall, self.targets.recall),
        )
        
        # Top-5 overlap
        top5 = ranking.top_k_overlap.get(5, 0)
        achieved["top_5"] = (top5, self._metric_status(top5, self.targets.top_5))
        
        # Top-10 overlap
        top10 = ranking.top_k_overlap.get(10, 0)
        achieved["top_10"] = (top10, self._metric_status(top10, self.targets.top_10))
        
        return achieved

    def _metric_status(self, value: float, target: float) -> MetricStatus:
        """Determine metric status"""
        if value >= target:
            return MetricStatus.MET
        elif value >= target * 0.95:
            return MetricStatus.BORDERLINE
        return MetricStatus.NOT_MET

    def _determine_status(
        self,
        achieved: Dict[str, Tuple[float, MetricStatus]],
    ) -> ValidationStatus:
        """Determine overall validation status"""
        met = sum(1 for _, (_, s) in achieved.items() if s == MetricStatus.MET)
        total = len(achieved)
        
        # Check critical metrics
        spearman_met = achieved.get("spearman", (0, MetricStatus.NOT_MET))[1] == MetricStatus.MET
        f1_met = achieved.get("f1", (0, MetricStatus.NOT_MET))[1] == MetricStatus.MET
        
        if spearman_met and f1_met and met >= total * 0.8:
            return ValidationStatus.PASSED
        elif met >= total * 0.5 or spearman_met or f1_met:
            return ValidationStatus.PARTIAL
        return ValidationStatus.FAILED


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_predictions(
    predicted: Dict[str, float],
    actual: Dict[str, float],
    component_types: Optional[Dict[str, str]] = None,
    targets: Optional[ValidationTargets] = None,
) -> ValidationResult:
    """
    Convenience function for quick validation.
    
    Args:
        predicted: Predicted scores
        actual: Actual impacts
        component_types: Component types
        targets: Validation targets
    
    Returns:
        ValidationResult
    """
    validator = Validator(targets=targets)
    return validator.validate(predicted, actual, component_types)


def quick_validate(
    predicted: Dict[str, float],
    actual: Dict[str, float],
) -> Dict[str, Any]:
    """
    Quick validation returning key metrics.
    
    Args:
        predicted: Predicted scores
        actual: Actual impacts
    
    Returns:
        Dictionary with key metrics
    """
    result = validate_predictions(predicted, actual)
    return result.summary()
