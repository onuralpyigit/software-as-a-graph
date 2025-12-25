"""
Validation Module for Graph-Based Analysis
============================================

Validates graph-based criticality analysis by comparing predicted scores
from topological analysis against actual impact scores from failure simulation.

Target Validation Metrics:
- Spearman Correlation: ≥ 0.70
- F1-Score: ≥ 0.90
- Precision/Recall: ≥ 0.80
- Top-5 Overlap: ≥ 60%
- Top-10 Overlap: ≥ 70%

Usage:
    from src.validation import IntegratedValidator
    
    # Run complete validation pipeline
    validator = IntegratedValidator(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    result = validator.run_validation()
    result.print_summary()
    
    # Quick validation
    from src.validation import run_quick_validation
    metrics = run_quick_validation()
    print(f"Spearman: {metrics['spearman']}")
    
    # Manual validation
    from src.validation import GraphValidator
    validator = GraphValidator()
    result = validator.validate(predicted_scores, actual_impacts)
"""

from .graph_validator import (
    # Main class
    GraphValidator,
    
    # Enums
    ValidationStatus,
    MetricStatus,
    
    # Data classes
    CorrelationMetrics,
    ConfusionMatrix,
    RankingMetrics,
    ComponentValidation,
    BootstrapResult,
    ValidationTargets,
    ValidationResult,
    
    # Statistical functions
    spearman_correlation,
    pearson_correlation,
    kendall_tau,
    percentile,
    
    # Convenience functions
    validate_predictions,
    quick_validate
)

from .integrated_validator import (
    IntegratedValidator,
    IntegratedValidationResult,
    run_quick_validation
)

__all__ = [
    # Main classes
    'GraphValidator',
    'IntegratedValidator',
    
    # Enums
    'ValidationStatus',
    'MetricStatus',
    
    # Data classes
    'CorrelationMetrics',
    'ConfusionMatrix',
    'RankingMetrics',
    'ComponentValidation',
    'BootstrapResult',
    'ValidationTargets',
    'ValidationResult',
    'IntegratedValidationResult',
    
    # Statistical functions
    'spearman_correlation',
    'pearson_correlation',
    'kendall_tau',
    'percentile',
    
    # Convenience functions
    'validate_predictions',
    'quick_validate',
    'run_quick_validation'
]

__version__ = '2.0.0'