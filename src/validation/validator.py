"""
Validator

Compares Predicted Scores (from Analysis) against Actual Impact (from Simulation).
Implements the Statistical Validation Framework.

Features:
- Overall validation across all components
- Per-type validation (Application, Broker, Node)
- Per-layer validation (application, infrastructure, complete)
- Box-plot based criticality classification
- Comprehensive metrics reporting

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .metrics import (
    ValidationTargets,
    ValidationSummary,
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

# Import classifier from analysis module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from analysis.classifier import BoxPlotClassifier, CriticalityLevel
    HAS_CLASSIFIER = True
except ImportError:
    try:
        from src.analysis.classifier import BoxPlotClassifier, CriticalityLevel
        HAS_CLASSIFIER = True
    except ImportError:
        HAS_CLASSIFIER = False
        BoxPlotClassifier = None
        CriticalityLevel = None


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
    components: List[Dict[str, Any]] = field(default_factory=list)
    
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
    
    def to_summary(self) -> ValidationSummary:
        """Convert to ValidationSummary."""
        return ValidationSummary(
            correlation=self.correlation,
            error=self.error,
            classification=self.classification,
            ranking=self.ranking,
            sample_size=self.sample_size,
            targets=self.targets,
        )


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
    
    # Warnings and notes
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
    
    Validation criteria:
    - Correlation: Spearman >= 0.70
    - Classification: F1 >= 0.80
    - Ranking: Top-5 overlap >= 0.60
    """
    
    def __init__(
        self,
        targets: Optional[ValidationTargets] = None,
        k_factor: float = 1.5
    ):
        """
        Initialize validator.
        
        Args:
            targets: Validation success criteria (uses defaults if None)
            k_factor: Box-plot IQR multiplier for outlier detection
        """
        self.targets = targets or ValidationTargets()
        self.k_factor = k_factor
        self.logger = logging.getLogger(__name__)
        
        # Initialize classifier for criticality determination
        if HAS_CLASSIFIER:
            self.classifier = BoxPlotClassifier(k_factor=k_factor)
        else:
            self.classifier = None
            self.logger.warning("BoxPlotClassifier not available, using percentile-based classification")
    
    def validate(
        self,
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        component_types: Optional[Dict[str, str]] = None,
        layer: str = "complete",
        context: str = "Validation"
    ) -> ValidationResult:
        """
        Main validation method.
        
        Compares predicted criticality scores against actual impact scores.
        
        Args:
            predicted_scores: Dict mapping component ID to predicted score
            actual_scores: Dict mapping component ID to actual impact score
            component_types: Dict mapping component ID to type (for breakdown)
            layer: Analysis layer name
            context: Context description for reporting
            
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
            warnings.append(f"{missing} predicted components not found in actual scores")
        
        if len(common_ids) < len(actual_ids):
            extra = len(actual_ids) - len(common_ids)
            warnings.append(f"{extra} actual components not found in predictions")
        
        if len(common_ids) < 3:
            warnings.append("Insufficient data for meaningful validation (n < 3)")
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
        
        # 1. Overall validation
        self.logger.info(f"Validating {len(common_ids)} components...")
        overall = self._validate_group(
            "Overall",
            pred_filtered,
            actual_filtered,
            types_filtered,
        )
        
        # 2. Per-type validation
        by_type = {}
        if component_types:
            unique_types = set(types_filtered.values())
            
            for comp_type in unique_types:
                type_ids = [k for k, t in types_filtered.items() if t == comp_type]
                
                if len(type_ids) >= 3:  # Minimum sample size
                    pred_type = {k: pred_filtered[k] for k in type_ids}
                    actual_type = {k: actual_filtered[k] for k in type_ids}
                    types_type = {k: comp_type for k in type_ids}
                    
                    by_type[comp_type] = self._validate_group(
                        comp_type,
                        pred_type,
                        actual_type,
                        types_type,
                    )
                else:
                    warnings.append(f"Insufficient {comp_type} components for per-type validation (n={len(type_ids)})")
        
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
        component_types: Dict[str, str],
    ) -> ValidationGroupResult:
        """Validate a specific group of components."""
        
        if len(predicted) < 2:
            return self._empty_group_result(group_name)
        
        # Align data
        common_ids = sorted(predicted.keys())
        pred_values = [predicted[i] for i in common_ids]
        actual_values = [actual[i] for i in common_ids]
        
        # 1. Correlation analysis
        correlation = CorrelationMetrics(
            spearman=spearman_correlation(pred_values, actual_values),
            pearson=pearson_correlation(pred_values, actual_values),
            kendall=kendall_correlation(pred_values, actual_values),
        )
        
        # 2. Error analysis
        error = calculate_error_metrics(pred_values, actual_values)
        
        # 3. Classification analysis
        pred_critical, actual_critical = self._classify_critical(
            common_ids, predicted, actual
        )
        classification = calculate_classification_metrics(pred_critical, actual_critical)
        
        # 4. Ranking analysis
        ranking = calculate_ranking_metrics(predicted, actual)
        
        # 5. Pass/fail decision
        passed = self._check_passed(correlation, classification, ranking)
        
        # 6. Component details
        components = []
        for i, comp_id in enumerate(common_ids):
            components.append({
                "id": comp_id,
                "type": component_types.get(comp_id, "Unknown"),
                "predicted": round(predicted[comp_id], 4),
                "actual": round(actual[comp_id], 4),
                "error": round(abs(predicted[comp_id] - actual[comp_id]), 4),
                "pred_critical": pred_critical[i],
                "actual_critical": actual_critical[i],
            })
        
        return ValidationGroupResult(
            group_name=group_name,
            sample_size=len(common_ids),
            correlation=correlation,
            error=error,
            classification=classification,
            ranking=ranking,
            passed=passed,
            targets=self.targets,
            components=components,
        )
    
    def _classify_critical(
        self,
        ids: List[str],
        predicted: Dict[str, float],
        actual: Dict[str, float]
    ) -> Tuple[List[bool], List[bool]]:
        """
        Classify components as critical or not critical.
        
        Uses box-plot classification if available, otherwise percentile-based.
        """
        if self.classifier is not None and HAS_CLASSIFIER:
            # Use box-plot classifier
            pred_items = [{"id": i, "score": predicted[i]} for i in ids]
            actual_items = [{"id": i, "score": actual[i]} for i in ids]
            
            pred_result = self.classifier.classify(pred_items, metric_name="predicted")
            actual_result = self.classifier.classify(actual_items, metric_name="actual")
            
            pred_levels = {item.id: item.level for item in pred_result.items}
            actual_levels = {item.id: item.level for item in actual_result.items}
            
            # Critical = HIGH or CRITICAL level
            pred_critical = [pred_levels[i] >= CriticalityLevel.HIGH for i in ids]
            actual_critical = [actual_levels[i] >= CriticalityLevel.HIGH for i in ids]
        else:
            # Fallback: percentile-based classification
            pred_values = [predicted[i] for i in ids]
            actual_values = [actual[i] for i in ids]
            
            pred_threshold = self._percentile(pred_values, 75)
            actual_threshold = self._percentile(actual_values, 75)
            
            pred_critical = [predicted[i] >= pred_threshold for i in ids]
            actual_critical = [actual[i] >= actual_threshold for i in ids]
        
        return pred_critical, actual_critical
    
    def _check_passed(
        self,
        correlation: CorrelationMetrics,
        classification: ClassificationMetrics,
        ranking: RankingMetrics
    ) -> bool:
        """Check if validation passes based on primary targets."""
        return (
            correlation.spearman >= self.targets.spearman and
            classification.f1_score >= self.targets.f1_score
        )
    
    def _empty_group_result(self, group_name: str) -> ValidationGroupResult:
        """Create empty result for insufficient data."""
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
    
    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        import math
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_values[int(k)]
        
        return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


class MultiLayerValidator:
    """
    Validates across multiple layers and aggregates results.
    """
    
    def __init__(self, targets: Optional[ValidationTargets] = None):
        self.targets = targets or ValidationTargets()
        self.validator = Validator(targets=self.targets)
        self.logger = logging.getLogger(__name__)
    
    def validate_all_layers(
        self,
        layer_data: Dict[str, Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]]
    ) -> Dict[str, ValidationResult]:
        """
        Validate all layers.
        
        Args:
            layer_data: Dict mapping layer name to (predicted, actual, types) tuple
            
        Returns:
            Dict mapping layer name to ValidationResult
        """
        results = {}
        
        for layer, (predicted, actual, types) in layer_data.items():
            self.logger.info(f"Validating layer: {layer}")
            results[layer] = self.validator.validate(
                predicted_scores=predicted,
                actual_scores=actual,
                component_types=types,
                layer=layer,
                context=f"{layer.title()} Layer Validation",
            )
        
        return results
    
    def generate_summary(
        self,
        results: Dict[str, ValidationResult]
    ) -> Dict[str, Any]:
        """Generate summary across all layers."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "layers_validated": len(results),
            "layers_passed": sum(1 for r in results.values() if r.passed),
            "overall_passed": all(r.passed for r in results.values()),
            "layer_results": {},
        }
        
        for layer, result in results.items():
            summary["layer_results"][layer] = {
                "passed": result.passed,
                "sample_size": result.overall.sample_size,
                "spearman": round(result.overall.correlation.spearman, 3),
                "f1_score": round(result.overall.classification.f1_score, 3),
                "top5_overlap": round(result.overall.ranking.top_5_overlap, 3),
            }
        
        return summary