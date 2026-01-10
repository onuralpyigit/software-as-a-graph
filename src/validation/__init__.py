"""
Validation Module

Validates graph-based criticality predictions against simulation results.

Components:
- ValidationTargets: Target thresholds for validation success
- Validator: Compares predicted vs actual scores
- ValidationPipeline: Orchestrates analysis + simulation + validation
- QuickValidator: Validates pre-computed scores

Metrics:
- Correlation: Spearman, Pearson, Kendall
- Classification: Precision, Recall, F1, Accuracy
- Ranking: Top-K overlap, NDCG
- Error: RMSE, MAE

Author: Software-as-a-Graph Research Project
"""

from .metrics import (
    # Target thresholds
    ValidationTargets,
    
    # Metric result classes
    CorrelationMetrics,
    ErrorMetrics,
    ClassificationMetrics,
    RankingMetrics,
    ValidationSummary,
    
    # Metric calculation functions
    spearman_correlation,
    pearson_correlation,
    kendall_correlation,
    calculate_error_metrics,
    calculate_classification_metrics,
    calculate_ranking_metrics,
    calculate_all_metrics,
)

from .validator import (
    Validator,
    MultiLayerValidator,
    ValidationResult,
    ValidationGroupResult,
)

from .pipeline import (
    ValidationPipeline,
    QuickValidator,
    PipelineResult,
    LayerValidationResult,
    ComponentComparison,
    LAYER_DEFINITIONS,
)


__all__ = [
    # Targets
    "ValidationTargets",
    
    # Metric classes
    "CorrelationMetrics",
    "ErrorMetrics",
    "ClassificationMetrics",
    "RankingMetrics",
    "ValidationSummary",
    
    # Metric functions
    "spearman_correlation",
    "pearson_correlation",
    "kendall_correlation",
    "calculate_error_metrics",
    "calculate_classification_metrics",
    "calculate_ranking_metrics",
    "calculate_all_metrics",
    
    # Validator
    "Validator",
    "MultiLayerValidator",
    "ValidationResult",
    "ValidationGroupResult",
    
    # Pipeline
    "ValidationPipeline",
    "QuickValidator",
    "PipelineResult",
    "LayerValidationResult",
    "ComponentComparison",
    "LAYER_DEFINITIONS",
]