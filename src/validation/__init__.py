"""
Validation Module - Version 5.0

Validates graph-based criticality predictions by comparing against
actual impact scores from failure simulation.

Key Research Question:
Do graph-based topological metrics accurately predict which components
will have the highest impact when they fail?

Validation Targets:
- Spearman ρ ≥ 0.70 (rank correlation)
- F1-Score ≥ 0.90 (classification accuracy)
- Precision ≥ 0.80
- Recall ≥ 0.80
- Top-5 Overlap ≥ 60%
- Top-10 Overlap ≥ 70%

Features:
- Component-type specific validation (Application, Broker, Topic, Node)
- Multiple analysis method comparison
- Neo4j integration for live validation
- Bootstrap confidence intervals
- Detailed component-level results

Usage:
    from src.simulation import SimulationGraph
    from src.validation import ValidationPipeline, run_validation
    
    # Load graph
    graph = SimulationGraph.from_json("system.json")
    
    # Run validation pipeline
    pipeline = ValidationPipeline(seed=42)
    result = pipeline.run(graph, compare_methods=True)
    
    print(f"Status: {result.validation.status.value}")
    print(f"Spearman: {result.spearman:.4f}")
    print(f"F1-Score: {result.f1_score:.4f}")
    
    # Compare analysis methods
    if result.method_comparison:
        for method, comp in result.method_comparison.items():
            print(f"{method}: ρ={comp.spearman:.4f}")
    
    # Validate specific component type
    for comp_type, type_result in result.by_component_type.items():
        print(f"{comp_type}: ρ={type_result.spearman:.4f}")
    
    # Neo4j validation
    from src.validation import Neo4jValidationClient
    
    with Neo4jValidationClient(uri, user, password) as client:
        result = client.run_full_validation()

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

# Metrics
from .metrics import (
    # Enums
    ValidationStatus,
    MetricStatus,
    # Data Classes
    ValidationTargets,
    CorrelationMetrics,
    ClassificationMetrics,
    ConfusionMatrix,
    RankingMetrics,
    BootstrapCI,
    # Correlation Functions
    spearman_correlation,
    pearson_correlation,
    kendall_correlation,
    calculate_correlation,
    # Classification Functions
    calculate_confusion_matrix,
    calculate_classification,
    # Ranking Functions
    calculate_ranking,
    # Bootstrap
    bootstrap_confidence_interval,
    # Statistical Utilities
    percentile,
    mean,
    std_dev,
    median,
    iqr,
)

# Validator
from .validator import (
    # Data Classes
    ComponentValidation,
    TypeValidationResult,
    ValidationResult,
    # Main Class
    Validator,
    # Factory Functions
    validate_predictions,
    quick_validate,
)

# Pipeline
from .pipeline import (
    # Enums
    AnalysisMethod,
    # Classes
    GraphAnalyzer,
    ValidationPipeline,
    # Data Classes
    MethodComparison,
    PipelineResult,
    # Factory Functions
    run_validation,
    quick_pipeline,
)

# Neo4j Client
from .neo4j_validator import (
    Neo4jValidationConfig,
    Neo4jValidationClient,
    validate_from_neo4j,
    run_neo4j_validation_pipeline,
)

__all__ = [
    # === Metrics ===
    # Enums
    "ValidationStatus",
    "MetricStatus",
    # Data Classes
    "ValidationTargets",
    "CorrelationMetrics",
    "ClassificationMetrics",
    "ConfusionMatrix",
    "RankingMetrics",
    "BootstrapCI",
    # Correlation Functions
    "spearman_correlation",
    "pearson_correlation",
    "kendall_correlation",
    "calculate_correlation",
    # Classification Functions
    "calculate_confusion_matrix",
    "calculate_classification",
    # Ranking Functions
    "calculate_ranking",
    # Bootstrap
    "bootstrap_confidence_interval",
    # Utilities
    "percentile",
    "mean",
    "std_dev",
    "median",
    "iqr",
    
    # === Validator ===
    "ComponentValidation",
    "TypeValidationResult",
    "ValidationResult",
    "Validator",
    "validate_predictions",
    "quick_validate",
    
    # === Pipeline ===
    "AnalysisMethod",
    "GraphAnalyzer",
    "ValidationPipeline",
    "MethodComparison",
    "PipelineResult",
    "run_validation",
    "quick_pipeline",
    
    # === Neo4j ===
    "Neo4jValidationConfig",
    "Neo4jValidationClient",
    "validate_from_neo4j",
    "run_neo4j_validation_pipeline",
]

__version__ = "5.0.0"
