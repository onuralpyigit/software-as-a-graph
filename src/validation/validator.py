"""
Validator - Version 5.0

Core validator for comparing predicted scores (from analysis)
against actual scores (from simulation).

Supports:
- Overall validation
- Layer-specific validation
- Component-type validation
- Bootstrap confidence intervals

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

from .metrics import (
    ValidationStatus,
    ValidationTargets,
    CorrelationMetrics,
    ClassificationMetrics,
    ConfusionMatrix,
    RankingMetrics,
    BootstrapCI,
    calculate_correlation,
    calculate_classification,
    calculate_ranking,
    bootstrap_confidence_interval,
    spearman_correlation,
    percentile,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComponentValidation:
    """Validation details for a single component."""
    component_id: str
    component_type: str
    layer: str
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
            "layer": self.layer,
            "predicted_score": round(self.predicted_score, 4),
            "actual_score": round(self.actual_score, 4),
            "predicted_rank": self.predicted_rank,
            "actual_rank": self.actual_rank,
            "rank_difference": self.rank_difference,
            "predicted_critical": self.predicted_critical,
            "actual_critical": self.actual_critical,
            "correctly_classified": self.correctly_classified,
        }


@dataclass
class LayerValidationResult:
    """Validation result for a specific layer."""
    layer: str
    layer_name: str
    count: int
    correlation: CorrelationMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    status: ValidationStatus
    spearman_ci: Optional[BootstrapCI] = None
    
    @property
    def spearman(self) -> float:
        return self.correlation.spearman
    
    @property
    def f1_score(self) -> float:
        return self.classification.f1_score
    
    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "layer_name": self.layer_name,
            "count": self.count,
            "spearman": round(self.spearman, 4),
            "f1_score": round(self.f1_score, 4),
            "status": self.status.value,
            "correlation": self.correlation.to_dict(),
            "classification": self.classification.to_dict(),
            "ranking": self.ranking.to_dict(),
            "spearman_ci": self.spearman_ci.to_dict() if self.spearman_ci else None,
        }


@dataclass
class TypeValidationResult:
    """Validation result for a specific component type."""
    component_type: str
    count: int
    correlation: CorrelationMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    status: ValidationStatus
    spearman_ci: Optional[BootstrapCI] = None
    
    @property
    def spearman(self) -> float:
        return self.correlation.spearman
    
    @property
    def f1_score(self) -> float:
        return self.classification.f1_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_type": self.component_type,
            "count": self.count,
            "spearman": round(self.spearman, 4),
            "f1_score": round(self.f1_score, 4),
            "status": self.status.value,
            "correlation": self.correlation.to_dict(),
            "classification": self.classification.to_dict(),
            "ranking": self.ranking.to_dict(),
            "spearman_ci": self.spearman_ci.to_dict() if self.spearman_ci else None,
        }


@dataclass
class ValidationResult:
    """Complete validation result."""
    correlation: CorrelationMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    status: ValidationStatus
    targets: ValidationTargets
    components: List[ComponentValidation] = field(default_factory=list)
    by_layer: Dict[str, LayerValidationResult] = field(default_factory=dict)
    by_type: Dict[str, TypeValidationResult] = field(default_factory=dict)
    spearman_ci: Optional[BootstrapCI] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def spearman(self) -> float:
        return self.correlation.spearman
    
    @property
    def f1_score(self) -> float:
        return self.classification.f1_score
    
    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED
    
    def get_false_positives(self) -> List[ComponentValidation]:
        """Get components predicted critical but not actually."""
        return [c for c in self.components 
                if c.predicted_critical and not c.actual_critical]
    
    def get_false_negatives(self) -> List[ComponentValidation]:
        """Get components actually critical but not predicted."""
        return [c for c in self.components 
                if not c.predicted_critical and c.actual_critical]
    
    def get_misclassified(self) -> List[ComponentValidation]:
        """Get all misclassified components."""
        return [c for c in self.components if not c.correctly_classified]
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Validation: {self.status.value.upper()}",
            f"  Spearman ρ: {self.spearman:.4f} (target: ≥{self.targets.spearman})",
            f"  F1-Score: {self.f1_score:.4f} (target: ≥{self.targets.f1_score})",
            f"  Precision: {self.classification.precision:.4f}",
            f"  Recall: {self.classification.recall:.4f}",
            f"  Top-5 Overlap: {self.ranking.top_5_overlap:.2%}",
        ]
        
        if self.by_layer:
            lines.append("\n  By Layer:")
            for layer, result in self.by_layer.items():
                lines.append(f"    {layer}: ρ={result.spearman:.4f}, "
                           f"F1={result.f1_score:.4f}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "spearman": round(self.spearman, 4),
            "f1_score": round(self.f1_score, 4),
            "correlation": self.correlation.to_dict(),
            "classification": self.classification.to_dict(),
            "ranking": self.ranking.to_dict(),
            "targets": self.targets.to_dict(),
            "by_layer": {k: v.to_dict() for k, v in self.by_layer.items()},
            "by_type": {k: v.to_dict() for k, v in self.by_type.items()},
            "spearman_ci": self.spearman_ci.to_dict() if self.spearman_ci else None,
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
    Validates predicted scores against actual scores.
    
    Compares analysis predictions with simulation results to
    determine if topological metrics accurately predict impact.
    
    Example:
        validator = Validator(targets=ValidationTargets())
        
        result = validator.validate(
            predicted_scores={"comp1": 0.8, "comp2": 0.5},
            actual_scores={"comp1": 0.75, "comp2": 0.6},
            component_info={
                "comp1": {"type": "Broker", "layer": "app_broker"},
                "comp2": {"type": "Application", "layer": "application"},
            }
        )
        
        print(f"Status: {result.status.value}")
        print(f"Spearman: {result.spearman:.4f}")
    """
    
    # Layer definitions
    LAYER_NAMES = {
        "application": "Application Layer (app_to_app)",
        "infrastructure": "Infrastructure Layer (node_to_node)",
        "app_broker": "Application-Broker Layer (app_to_broker)",
        "node_broker": "Node-Broker Layer (node_to_broker)",
    }
    
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
            bootstrap_samples: Number of bootstrap samples
            seed: Random seed
        """
        self.targets = targets or ValidationTargets()
        self.critical_threshold = critical_threshold
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed
        self.logger = logging.getLogger(__name__)
    
    def validate(
        self,
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        component_info: Optional[Dict[str, Dict[str, str]]] = None,
        compute_ci: bool = True,
    ) -> ValidationResult:
        """
        Validate predicted scores against actual scores.
        
        Args:
            predicted_scores: {component_id: predicted_score}
            actual_scores: {component_id: actual_score}
            component_info: {component_id: {"type": str, "layer": str}}
            compute_ci: Compute bootstrap confidence intervals
        
        Returns:
            ValidationResult with all metrics
        """
        # Get common components
        common_ids = set(predicted_scores.keys()) & set(actual_scores.keys())
        
        if len(common_ids) < 3:
            self.logger.warning(f"Too few common components: {len(common_ids)}")
            return self._empty_result()
        
        ids = sorted(common_ids)
        predicted = [predicted_scores[cid] for cid in ids]
        actual = [actual_scores[cid] for cid in ids]
        
        # Determine threshold
        threshold = self.critical_threshold
        if threshold is None:
            threshold = percentile(actual, 75)
        
        # Calculate overall metrics
        correlation = calculate_correlation(predicted, actual)
        classification = calculate_classification(predicted, actual, threshold)
        ranking = calculate_ranking(
            {cid: predicted_scores[cid] for cid in ids},
            {cid: actual_scores[cid] for cid in ids},
        )
        
        # Build component validations
        pred_ranked = sorted(ids, key=lambda x: -predicted_scores[x])
        actual_ranked = sorted(ids, key=lambda x: -actual_scores[x])
        pred_ranks = {cid: i + 1 for i, cid in enumerate(pred_ranked)}
        actual_ranks = {cid: i + 1 for i, cid in enumerate(actual_ranked)}
        
        components = []
        for cid in ids:
            pred_score = predicted_scores[cid]
            act_score = actual_scores[cid]
            pred_critical = pred_score >= threshold
            act_critical = act_score >= threshold
            
            # Get component info
            info = component_info.get(cid, {}) if component_info else {}
            comp_type = info.get("type", "Unknown")
            layer = info.get("layer", "unknown")
            
            components.append(ComponentValidation(
                component_id=cid,
                component_type=comp_type,
                layer=layer,
                predicted_score=pred_score,
                actual_score=act_score,
                predicted_rank=pred_ranks[cid],
                actual_rank=actual_ranks[cid],
                rank_difference=pred_ranks[cid] - actual_ranks[cid],
                predicted_critical=pred_critical,
                actual_critical=act_critical,
                correctly_classified=pred_critical == act_critical,
            ))
        
        # Validate by layer
        by_layer = {}
        if component_info:
            by_layer = self._validate_by_layer(
                ids, predicted_scores, actual_scores, 
                component_info, threshold, compute_ci
            )
        
        # Validate by type
        by_type = {}
        if component_info:
            by_type = self._validate_by_type(
                ids, predicted_scores, actual_scores,
                component_info, threshold, compute_ci
            )
        
        # Bootstrap CI
        spearman_ci = None
        if compute_ci and len(predicted) >= 10:
            spearman_ci = bootstrap_confidence_interval(
                predicted, actual, spearman_correlation,
                n_bootstrap=self.bootstrap_samples,
                seed=self.seed,
            )
        
        # Determine status
        status = self._determine_status(correlation, classification, ranking)
        
        return ValidationResult(
            correlation=correlation,
            classification=classification,
            ranking=ranking,
            status=status,
            targets=self.targets,
            components=components,
            by_layer=by_layer,
            by_type=by_type,
            spearman_ci=spearman_ci,
            metadata={
                "total_components": len(ids),
                "threshold": threshold,
            },
        )
    
    def _validate_by_layer(
        self,
        ids: List[str],
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        component_info: Dict[str, Dict[str, str]],
        threshold: float,
        compute_ci: bool,
    ) -> Dict[str, LayerValidationResult]:
        """Validate for each layer."""
        results = {}
        
        # Group by layer
        layer_ids: Dict[str, List[str]] = {}
        for cid in ids:
            info = component_info.get(cid, {})
            layer = info.get("layer", "unknown")
            if layer not in layer_ids:
                layer_ids[layer] = []
            layer_ids[layer].append(cid)
        
        for layer, layer_cids in layer_ids.items():
            if len(layer_cids) < 3:
                continue
            
            pred = [predicted_scores[cid] for cid in layer_cids]
            act = [actual_scores[cid] for cid in layer_cids]
            
            correlation = calculate_correlation(pred, act)
            classification = calculate_classification(pred, act, threshold)
            ranking = calculate_ranking(
                {cid: predicted_scores[cid] for cid in layer_cids},
                {cid: actual_scores[cid] for cid in layer_cids},
            )
            
            spearman_ci = None
            if compute_ci and len(pred) >= 10:
                spearman_ci = bootstrap_confidence_interval(
                    pred, act, spearman_correlation,
                    n_bootstrap=self.bootstrap_samples,
                    seed=self.seed,
                )
            
            status = self._determine_status(correlation, classification, ranking)
            
            results[layer] = LayerValidationResult(
                layer=layer,
                layer_name=self.LAYER_NAMES.get(layer, layer),
                count=len(layer_cids),
                correlation=correlation,
                classification=classification,
                ranking=ranking,
                status=status,
                spearman_ci=spearman_ci,
            )
        
        return results
    
    def _validate_by_type(
        self,
        ids: List[str],
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        component_info: Dict[str, Dict[str, str]],
        threshold: float,
        compute_ci: bool,
    ) -> Dict[str, TypeValidationResult]:
        """Validate for each component type."""
        results = {}
        
        # Group by type
        type_ids: Dict[str, List[str]] = {}
        for cid in ids:
            info = component_info.get(cid, {})
            comp_type = info.get("type", "Unknown")
            if comp_type not in type_ids:
                type_ids[comp_type] = []
            type_ids[comp_type].append(cid)
        
        for comp_type, type_cids in type_ids.items():
            if len(type_cids) < 3:
                continue
            
            pred = [predicted_scores[cid] for cid in type_cids]
            act = [actual_scores[cid] for cid in type_cids]
            
            correlation = calculate_correlation(pred, act)
            classification = calculate_classification(pred, act, threshold)
            ranking = calculate_ranking(
                {cid: predicted_scores[cid] for cid in type_cids},
                {cid: actual_scores[cid] for cid in type_cids},
            )
            
            spearman_ci = None
            if compute_ci and len(pred) >= 10:
                spearman_ci = bootstrap_confidence_interval(
                    pred, act, spearman_correlation,
                    n_bootstrap=self.bootstrap_samples,
                    seed=self.seed,
                )
            
            status = self._determine_status(correlation, classification, ranking)
            
            results[comp_type] = TypeValidationResult(
                component_type=comp_type,
                count=len(type_cids),
                correlation=correlation,
                classification=classification,
                ranking=ranking,
                status=status,
                spearman_ci=spearman_ci,
            )
        
        return results
    
    def _determine_status(
        self,
        correlation: CorrelationMetrics,
        classification: ClassificationMetrics,
        ranking: RankingMetrics,
    ) -> ValidationStatus:
        """Determine validation status from metrics."""
        spearman_passed = correlation.spearman >= self.targets.spearman
        f1_passed = classification.f1_score >= self.targets.f1_score
        
        if spearman_passed and f1_passed:
            return ValidationStatus.PASSED
        elif spearman_passed or f1_passed:
            return ValidationStatus.PARTIAL
        else:
            return ValidationStatus.FAILED
    
    def _empty_result(self) -> ValidationResult:
        """Return empty result for error cases."""
        return ValidationResult(
            correlation=CorrelationMetrics(),
            classification=ClassificationMetrics(
                confusion_matrix=ConfusionMatrix(),
                threshold=0.0,
            ),
            ranking=RankingMetrics(),
            status=ValidationStatus.ERROR,
            targets=self.targets,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def validate_predictions(
    predicted_scores: Dict[str, float],
    actual_scores: Dict[str, float],
    component_info: Optional[Dict[str, Dict[str, str]]] = None,
    targets: Optional[ValidationTargets] = None,
) -> ValidationResult:
    """
    Quick validation function.
    
    Args:
        predicted_scores: Predicted criticality scores
        actual_scores: Actual impact scores
        component_info: Component metadata
        targets: Validation targets
    
    Returns:
        ValidationResult
    """
    validator = Validator(targets=targets)
    return validator.validate(predicted_scores, actual_scores, component_info)


def quick_validate(
    predicted_scores: Dict[str, float],
    actual_scores: Dict[str, float],
) -> ValidationStatus:
    """
    Quick status check.
    
    Returns:
        ValidationStatus
    """
    result = validate_predictions(predicted_scores, actual_scores)
    return result.status
