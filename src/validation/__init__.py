"""
Validation Module - Version 4.0

Validates graph-based criticality analysis by comparing predicted scores
from topological analysis against actual impact scores from failure simulation.

Target Validation Metrics:
- Spearman Correlation: ≥ 0.70 (rank correlation)
- F1-Score: ≥ 0.90 (classification accuracy)
- Precision/Recall: ≥ 0.80
- Top-5 Overlap: ≥ 60%
- Top-10 Overlap: ≥ 70%

Key Question: Do graph-based topological metrics accurately predict
which components will have the highest impact when they fail?

Usage:
    from src.validation import ValidationPipeline, SimulationGraph

    # Load graph
    graph = SimulationGraph.from_json("system.json")
    
    # Run complete validation pipeline
    pipeline = ValidationPipeline()
    result = pipeline.run(graph, analysis_method="composite")
    
    print(f"Status: {result.validation.status.value}")
    print(f"Spearman: {result.validation.correlation.spearman:.4f}")
    print(f"F1-Score: {result.validation.classification.f1:.4f}")
    
    # Compare analysis methods
    results = pipeline.compare_methods(graph)
    for method, r in results.items():
        print(f"{method}: ρ={r.validation.correlation.spearman:.4f}")
    
    # Quick validation
    from src.validation import quick_pipeline
    metrics = quick_pipeline(graph)
    print(metrics)

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

# Metrics
from .metrics import (
    # Enums
    ValidationStatus,
    MetricStatus,
    # Data classes
    CorrelationMetrics,
    ConfusionMatrix,
    RankingMetrics,
    BootstrapCI,
    ValidationTargets,
    # Statistical functions
    spearman,
    pearson,
    kendall,
    percentile,
    std_dev,
    calculate_correlation,
    calculate_confusion,
    calculate_ranking,
    bootstrap_confidence_interval,
)

# Validator
from .validator import (
    Validator,
    ComponentValidation,
    ValidationResult,
    validate_predictions,
    quick_validate,
)

# Pipeline
from .pipeline import (
    GraphAnalyzer,
    ValidationPipeline,
    PipelineResult,
    run_validation,
    quick_pipeline,
)

__all__ = [
    # Enums
    "ValidationStatus",
    "MetricStatus",
    # Metrics
    "CorrelationMetrics",
    "ConfusionMatrix",
    "RankingMetrics",
    "BootstrapCI",
    "ValidationTargets",
    # Statistical functions
    "spearman",
    "pearson",
    "kendall",
    "percentile",
    "std_dev",
    "calculate_correlation",
    "calculate_confusion",
    "calculate_ranking",
    "bootstrap_confidence_interval",
    # Validator
    "Validator",
    "ComponentValidation",
    "ValidationResult",
    "validate_predictions",
    "quick_validate",
    # Pipeline
    "GraphAnalyzer",
    "ValidationPipeline",
    "PipelineResult",
    "run_validation",
    "quick_pipeline",
]

__version__ = "4.0.0"