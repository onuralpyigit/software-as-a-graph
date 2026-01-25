"""
Validation Module

Statistical validation for comparing graph analysis predictions
against failure simulation results.

This module validates that topological metrics can reliably predict
critical components by comparing:
    - Predicted criticality scores (from graph analysis)
    - Actual impact scores (from failure simulation)

Validation Metrics:
    - Correlation: Spearman ρ, Pearson r, Kendall τ
    - Classification: Precision, Recall, F1, Accuracy
    - Ranking: Top-K overlap, NDCG
    - Error: RMSE, MAE

Validation Targets (defaults):
    - Spearman ρ ≥ 0.70 (strong rank correlation)
    - F1 Score ≥ 0.80 (balanced classification)
    - Top-5 Overlap ≥ 0.60 (ranking agreement)

Layers:
    - app: Application layer
    - infra: Infrastructure layer
    - mw-app: Middleware-Application layer
    - mw-infra: Middleware-Infrastructure layer
    - system: Complete system

Example:
    >>> from src.validation import ValidationPipeline
    >>> 
    >>> pipeline = ValidationPipeline(uri="bolt://localhost:7687")
    >>> result = pipeline.run(layers=["app", "infra", "system"])
    >>> 
    >>> print(f"All passed: {result.all_passed}")
    >>> for layer, layer_result in result.layers.items():
    ...     print(f"  {layer}: ρ={layer_result.spearman:.3f}, F1={layer_result.f1_score:.3f}")
"""

# Metrics
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

# Validator
from .validator import (
    Validator,
    ValidationResult,
    ValidationGroupResult,
    ComponentComparison,
)

# Pipeline
from .pipeline import (
    ValidationPipeline,
    PipelineResult,
    LayerValidationResult,
    QuickValidator,
    LAYER_DEFINITIONS,
)

# Display Functions
from ..visualization.display import (
    display_pipeline_validation_result as display_pipeline_result,
    display_layer_validation_result as display_layer_result,
    status_icon,
    status_text,
    metric_color,
)


__all__ = [
    # Metrics
    "ValidationTargets",
    "CorrelationMetrics",
    "ErrorMetrics",
    "ClassificationMetrics",
    "RankingMetrics",
    "spearman_correlation",
    "pearson_correlation",
    "kendall_correlation",
    "calculate_error_metrics",
    "calculate_classification_metrics",
    "calculate_ranking_metrics",
    
    # Validator
    "Validator",
    "ValidationResult",
    "ValidationGroupResult",
    "ComponentComparison",
    
    # Pipeline
    "ValidationPipeline",
    "PipelineResult",
    "LayerValidationResult",
    "QuickValidator",
    "LAYER_DEFINITIONS",
    
    # Display Functions
    "display_pipeline_result",
    "display_layer_result",
    "status_icon",
    "status_text",
    "metric_color",
]

__version__ = "2.0.0"