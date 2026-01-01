"""
Validator - Version 5.0

Core validation logic for comparing predicted criticality scores
(from graph analysis) against actual impact scores (from simulation).

Supports:
- Component-type specific validation
- Overall system validation
- Multiple analysis method comparison
- Detailed component-level results

Key Research Question:
Do graph-based topological metrics accurately predict which components
will have the highest impact when they fail?

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Sequence
from enum import Enum

from .metrics import (
    ValidationStatus,
    MetricStatus,
    ValidationTargets,
    CorrelationMetrics,
    ClassificationMetrics,
    RankingMetrics,
    BootstrapCI,
    calculate_correlation,
    calculate_classification,
    calculate_ranking,
    bootstrap_confidence_interval,
    spearman_correlation,
    mean,
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
    actual_score: float
    predicted_rank: int
    actual_rank: int
    rank_difference: int
    predicted_critical: bool
    actual_critical: bool
    correctly_classified: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "predicted_score": round(self.predicted_score, 6),
            "actual_score": round(self.actual_score, 6),
            "predicted_rank": self.predicted_rank,
            "actual_rank": self.actual_rank,
            "rank_difference": self.rank_difference,
            "predicted_critical": self.predicted_critical,
            "actual_critical": self.actual_critical,
            "correctly_classified": self.correctly_classified,
        }


@dataclass
class TypeValidationResult:
    """Validation result for a specific component type"""
    component_type: str
    count: int
    correlation: CorrelationMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    spearman_ci: Optional[BootstrapCI] = None
    status: ValidationStatus = ValidationStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_type": self.component_type,
            "count": self.count,
            "correlation": self.correlation.to_dict(),
            "classification": self.classification.to_dict(),
            "ranking": self.ranking.to_dict(),
            "spearman_ci": self.spearman_ci.to_dict() if self.spearman_ci else None,
            "status": self.status.value,
        }


@dataclass
class ValidationResult:
    """Complete validation result"""
    correlation: CorrelationMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    components: List[ComponentValidation]
    by_type: Dict[str, TypeValidationResult]
    status: ValidationStatus
    targets: ValidationTargets
    spearman_ci: Optional[BootstrapCI] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        """Check if validation passed"""
        return self.status == ValidationStatus.PASSED
    
    @property
    def spearman(self) -> float:
        """Shortcut for Spearman correlation"""
        return self.correlation.spearman
    
    @property
    def f1_score(self) -> float:
        """Shortcut for F1-score"""
        return self.classification.f1_score
    
    def get_misclassified(self) -> List[ComponentValidation]:
        """Get incorrectly classified components"""
        return [c for c in self.components if not c.correctly_classified]
    
    def get_top_rank_errors(self, n: int = 10) -> List[ComponentValidation]:
        """Get components with largest rank differences"""
        return sorted(self.components, key=lambda c: -abs(c.rank_difference))[:n]
    
    def get_false_positives(self) -> List[ComponentValidation]:
        """Get components predicted critical but not actually critical"""
        return [c for c in self.components 
                if c.predicted_critical and not c.actual_critical]
    
    def get_false_negatives(self) -> List[ComponentValidation]:
        """Get components actually critical but not predicted"""
        return [c for c in self.components 
                if c.actual_critical and not c.predicted_critical]
    
    def summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            f"Validation Result: {self.status.value.upper()}",
            f"  Samples: {self.correlation.n_samples}",
            f"  Spearman ρ: {self.spearman:.4f} (target: ≥{self.targets.spearman})",
            f"  F1-Score: {self.f1_score:.4f} (target: ≥{self.targets.f1_score})",
            f"  Precision: {self.classification.precision:.4f}",
            f"  Recall: {self.classification.recall:.4f}",
            f"  Top-5 Overlap: {self.ranking.top_5_overlap:.2%}",
            f"  Top-10 Overlap: {self.ranking.top_10_overlap:.2%}",
        ]
        
        if self.by_type:
            lines.append("\n  By Component Type:")
            for comp_type, result in self.by_type.items():
                lines.append(f"    {comp_type}: ρ={result.correlation.spearman:.4f}, "
                           f"F1={result.classification.f1_score:.4f}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "correlation": self.correlation.to_dict(),
            "classification": self.classification.to_dict(),
            "ranking": self.ranking.to_dict(),
            "spearman_ci": self.spearman_ci.to_dict() if self.spearman_ci else None,
            "by_type": {k: v.to_dict() for k, v in self.by_type.items()},
            "targets": self.targets.to_dict(),
            "component_count": len(self.components),
            "misclassified_count": len(self.get_misclassified()),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# Validator
# =============================================================================

class Validator:
    """
    Validates predicted criticality against actual impact.
    
    Compares scores from graph analysis (predicted) with
    impact scores from failure simulation (actual) to determine
    if topological metrics accurately predict system behavior.
    """
    
    def __init__(
        self,
        targets: Optional[ValidationTargets] = None,
        critical_threshold: Optional[float] = None,
        bootstrap_samples: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Initialize validator.
        
        Args:
            targets: Validation target thresholds
            critical_threshold: Threshold for critical classification
                              (default: 75th percentile of actual scores)
            bootstrap_samples: Number of bootstrap samples for CI
            seed: Random seed for reproducibility
        """
        self.targets = targets or ValidationTargets()
        self.critical_threshold = critical_threshold
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed
        self._logger = logging.getLogger(__name__)
    
    def validate(
        self,
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        component_types: Optional[Dict[str, str]] = None,
        compute_ci: bool = True,
    ) -> ValidationResult:
        """
        Validate predicted scores against actual scores.
        
        Args:
            predicted_scores: Dict mapping component_id -> predicted score
            actual_scores: Dict mapping component_id -> actual impact score
            component_types: Dict mapping component_id -> type (e.g., "Application")
            compute_ci: Whether to compute confidence intervals
        
        Returns:
            ValidationResult with all metrics
        """
        # Get common components
        common_ids = set(predicted_scores.keys()) & set(actual_scores.keys())
        
        if not common_ids:
            self._logger.warning("No common components between predicted and actual")
            return self._empty_result()
        
        # Extract aligned values
        ids = sorted(common_ids)
        predicted = [predicted_scores[cid] for cid in ids]
        actual = [actual_scores[cid] for cid in ids]
        
        # Determine threshold
        threshold = self.critical_threshold
        if threshold is None:
            sorted_actual = sorted(actual)
            idx = int(len(sorted_actual) * 0.75)
            threshold = sorted_actual[min(idx, len(sorted_actual) - 1)]
        
        # Calculate metrics
        correlation = calculate_correlation(predicted, actual)
        classification = calculate_classification(predicted, actual, threshold)
        
        # Build rankings
        pred_ranked = sorted(ids, key=lambda x: -predicted_scores[x])
        actual_ranked = sorted(ids, key=lambda x: -actual_scores[x])
        ranking = calculate_ranking(pred_ranked, actual_ranked)
        
        # Component-level validation
        pred_ranks = {cid: i for i, cid in enumerate(pred_ranked)}
        actual_ranks = {cid: i for i, cid in enumerate(actual_ranked)}
        
        components = []
        for cid in ids:
            pred_score = predicted_scores[cid]
            actual_score = actual_scores[cid]
            pred_rank = pred_ranks[cid]
            actual_rank = actual_ranks[cid]
            pred_critical = pred_score >= threshold
            actual_critical = actual_score >= threshold
            
            components.append(ComponentValidation(
                component_id=cid,
                component_type=component_types.get(cid, "Unknown") if component_types else "Unknown",
                predicted_score=pred_score,
                actual_score=actual_score,
                predicted_rank=pred_rank,
                actual_rank=actual_rank,
                rank_difference=pred_rank - actual_rank,
                predicted_critical=pred_critical,
                actual_critical=actual_critical,
                correctly_classified=pred_critical == actual_critical,
            ))
        
        # Validate by component type
        by_type = {}
        if component_types:
            by_type = self._validate_by_type(
                ids, predicted_scores, actual_scores, 
                component_types, threshold, compute_ci
            )
        
        # Bootstrap CI for Spearman
        spearman_ci = None
        if compute_ci and len(predicted) >= 10:
            spearman_ci = bootstrap_confidence_interval(
                predicted, actual, spearman_correlation,
                n_bootstrap=self.bootstrap_samples,
                seed=self.seed
            )
        
        # Determine overall status
        status = self._determine_status(correlation, classification, ranking)
        
        return ValidationResult(
            correlation=correlation,
            classification=classification,
            ranking=ranking,
            components=components,
            by_type=by_type,
            status=status,
            targets=self.targets,
            spearman_ci=spearman_ci,
            metadata={
                "total_components": len(ids),
                "threshold": threshold,
            },
        )
    
    def _validate_by_type(
        self,
        ids: List[str],
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        component_types: Dict[str, str],
        threshold: float,
        compute_ci: bool,
    ) -> Dict[str, TypeValidationResult]:
        """Validate each component type separately"""
        # Group by type
        by_type_ids: Dict[str, List[str]] = {}
        for cid in ids:
            comp_type = component_types.get(cid, "Unknown")
            if comp_type not in by_type_ids:
                by_type_ids[comp_type] = []
            by_type_ids[comp_type].append(cid)
        
        results = {}
        for comp_type, type_ids in by_type_ids.items():
            if len(type_ids) < 3:
                continue
            
            predicted = [predicted_scores[cid] for cid in type_ids]
            actual = [actual_scores[cid] for cid in type_ids]
            
            correlation = calculate_correlation(predicted, actual)
            classification = calculate_classification(predicted, actual, threshold)
            
            pred_ranked = sorted(type_ids, key=lambda x: -predicted_scores[x])
            actual_ranked = sorted(type_ids, key=lambda x: -actual_scores[x])
            ranking = calculate_ranking(pred_ranked, actual_ranked)
            
            spearman_ci = None
            if compute_ci and len(predicted) >= 10:
                spearman_ci = bootstrap_confidence_interval(
                    predicted, actual, spearman_correlation,
                    n_bootstrap=self.bootstrap_samples,
                    seed=self.seed
                )
            
            status = self._determine_status(correlation, classification, ranking)
            
            results[comp_type] = TypeValidationResult(
                component_type=comp_type,
                count=len(type_ids),
                correlation=correlation,
                classification=classification,
                ranking=ranking,
                spearman_ci=spearman_ci,
                status=status,
            )
        
        return results
    
    def _determine_status(
        self,
        correlation: CorrelationMetrics,
        classification: ClassificationMetrics,
        ranking: RankingMetrics,
    ) -> ValidationStatus:
        """Determine validation status based on metrics"""
        # Primary criteria
        spearman_passed = correlation.spearman >= self.targets.spearman
        f1_passed = classification.f1_score >= self.targets.f1_score
        
        # Secondary criteria
        precision_passed = classification.precision >= self.targets.precision
        recall_passed = classification.recall >= self.targets.recall
        
        if spearman_passed and f1_passed:
            return ValidationStatus.PASSED
        elif spearman_passed or f1_passed:
            return ValidationStatus.PARTIAL
        else:
            return ValidationStatus.FAILED
    
    def _empty_result(self) -> ValidationResult:
        """Return empty validation result"""
        return ValidationResult(
            correlation=CorrelationMetrics(),
            classification=ClassificationMetrics(
                confusion_matrix=ConfusionMatrix(),
                threshold=0.0
            ),
            ranking=RankingMetrics(),
            components=[],
            by_type={},
            status=ValidationStatus.ERROR,
            targets=self.targets,
        )
    
    def compare_methods(
        self,
        methods: Dict[str, Dict[str, float]],
        actual_scores: Dict[str, float],
        component_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, ValidationResult]:
        """
        Compare multiple analysis methods.
        
        Args:
            methods: Dict mapping method_name -> {component_id: score}
            actual_scores: Actual impact scores
            component_types: Component type mapping
        
        Returns:
            Dict mapping method_name -> ValidationResult
        """
        results = {}
        for method_name, predicted_scores in methods.items():
            results[method_name] = self.validate(
                predicted_scores, actual_scores, component_types
            )
        return results


# =============================================================================
# Factory Functions
# =============================================================================

def validate_predictions(
    predicted_scores: Dict[str, float],
    actual_scores: Dict[str, float],
    component_types: Optional[Dict[str, str]] = None,
    targets: Optional[ValidationTargets] = None,
) -> ValidationResult:
    """
    Quick validation function.
    
    Args:
        predicted_scores: Predicted criticality scores
        actual_scores: Actual impact scores
        component_types: Optional component type mapping
        targets: Validation targets
    
    Returns:
        ValidationResult
    """
    validator = Validator(targets=targets)
    return validator.validate(predicted_scores, actual_scores, component_types)


def quick_validate(
    predicted_scores: Dict[str, float],
    actual_scores: Dict[str, float],
) -> Tuple[float, float, bool]:
    """
    Quick validation returning just key metrics.
    
    Returns:
        Tuple of (spearman, f1_score, passed)
    """
    result = validate_predictions(predicted_scores, actual_scores)
    return (result.spearman, result.f1_score, result.passed)


# Import ConfusionMatrix for _empty_result
from .metrics import ConfusionMatrix
