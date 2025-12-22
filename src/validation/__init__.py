"""
Validation Module for Graph-Based Analysis
============================================

Comprehensive validation of graph-based criticality analysis by comparing
predicted scores from topological analysis against actual impact scores
from failure simulation.

Key Validation Metrics:
- Spearman Correlation: Target ≥ 0.70
- F1-Score: Target ≥ 0.90
- Precision/Recall: Target ≥ 0.80
- Top-k Overlap: Top-5 ≥ 60%, Top-10 ≥ 70%

Usage:
    from src.validation import GraphValidator, validate_analysis
    
    # Quick validation
    result = validate_analysis(graph, predicted_scores, actual_impacts)
    print(f"Status: {result.status}")
    print(f"Spearman: {result.correlation.spearman_coefficient:.3f}")
    
    # Full validation with advanced analysis
    validator = GraphValidator()
    result = validator.validate(graph, predicted_scores, actual_impacts)
    validator.run_sensitivity_analysis(graph, predicted_scores, actual_impacts)
    validator.run_bootstrap_analysis(graph, predicted_scores, actual_impacts)
"""

from .graph_validator import (
    # Main class
    GraphValidator,
    
    # Enums
    ValidationStatus,
    CriticalityLevel,
    
    # Data classes
    ConfusionMatrix,
    ComponentValidation,
    CorrelationResult,
    RankingMetrics,
    SensitivityResult,
    BootstrapResult,
    CrossValidationResult,
    ValidationResult,
    
    # Statistical functions
    spearman_correlation,
    pearson_correlation,
    kendall_tau,
    calculate_percentile,
    
    # Convenience functions
    validate_analysis,
    quick_validate,
    compare_methods
)

__all__ = [
    # Main class
    'GraphValidator',
    
    # Enums
    'ValidationStatus',
    'CriticalityLevel',
    
    # Data classes
    'ConfusionMatrix',
    'ComponentValidation',
    'CorrelationResult',
    'RankingMetrics',
    'SensitivityResult',
    'BootstrapResult',
    'CrossValidationResult',
    'ValidationResult',
    
    # Statistical functions
    'spearman_correlation',
    'pearson_correlation',
    'kendall_tau',
    'calculate_percentile',
    
    # Convenience functions
    'validate_analysis',
    'quick_validate',
    'compare_methods'
]

__version__ = '1.0.0'