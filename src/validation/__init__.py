"""
Validation Module - Version 5.0

Validates graph analysis predictions against simulation results.

Key Question:
    Do topological metrics accurately predict which components
    will have the highest impact when they fail?

Validation Targets:
    - Spearman ρ ≥ 0.70 (rank correlation)
    - F1-Score ≥ 0.90 (classification)
    - Precision ≥ 0.80
    - Recall ≥ 0.80
    - Top-5 Overlap ≥ 60%

Layers Validated:
    - application: app_to_app dependencies
    - infrastructure: node_to_node dependencies  
    - app_broker: app_to_broker dependencies
    - node_broker: node_to_broker dependencies

Usage:
    from src.simulation import SimulationGraph
    from src.validation import ValidationPipeline, run_validation
    
    # Load from file
    graph = SimulationGraph.from_json("system.json")
    
    # Run validation
    pipeline = ValidationPipeline(seed=42)
    result = pipeline.run(graph, compare_methods=True)
    
    print(f"Status: {result.validation.status.value}")
    print(f"Spearman: {result.spearman:.4f}")
    print(f"F1-Score: {result.f1_score:.4f}")
    
    # Layer-specific results
    for layer, layer_result in result.by_layer.items():
        print(f"{layer}: ρ={layer_result.spearman:.4f}")
    
    # Load from Neo4j
    from src.validation import validate_from_neo4j
    
    result = validate_from_neo4j(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        compare_methods=True
    )

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

__version__ = "5.0.0"

# Metrics
from .metrics import (
    # Enums
    ValidationStatus,
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
    # Utilities
    mean,
    std_dev,
    median,
    percentile,
    iqr,
)

# Validator
from .validator import (
    # Data Classes
    ComponentValidation,
    LayerValidationResult,
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
    # Utilities
    get_component_layer,
    build_component_info,
    # Factory Functions
    run_validation,
    quick_pipeline,
)

# Neo4j Client
from .neo4j_client import (
    # Config
    Neo4jConfig,
    # Main Class
    Neo4jValidationClient,
    # Factory Functions
    validate_from_neo4j,
    check_neo4j_available,
)


__all__ = [
    # Version
    "__version__",
    
    # Metrics - Enums
    "ValidationStatus",
    # Metrics - Data Classes
    "ValidationTargets",
    "CorrelationMetrics",
    "ClassificationMetrics",
    "ConfusionMatrix",
    "RankingMetrics",
    "BootstrapCI",
    # Metrics - Functions
    "spearman_correlation",
    "pearson_correlation",
    "kendall_correlation",
    "calculate_correlation",
    "calculate_confusion_matrix",
    "calculate_classification",
    "calculate_ranking",
    "bootstrap_confidence_interval",
    "mean",
    "std_dev",
    "median",
    "percentile",
    "iqr",
    
    # Validator - Data Classes
    "ComponentValidation",
    "LayerValidationResult",
    "TypeValidationResult",
    "ValidationResult",
    # Validator - Main
    "Validator",
    "validate_predictions",
    "quick_validate",
    
    # Pipeline - Enums
    "AnalysisMethod",
    # Pipeline - Classes
    "GraphAnalyzer",
    "ValidationPipeline",
    # Pipeline - Data Classes
    "MethodComparison",
    "PipelineResult",
    # Pipeline - Functions
    "get_component_layer",
    "build_component_info",
    "run_validation",
    "quick_pipeline",
    
    # Neo4j Client
    "Neo4jConfig",
    "Neo4jValidationClient",
    "validate_from_neo4j",
    "check_neo4j_available",
]
