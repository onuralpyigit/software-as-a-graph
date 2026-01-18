"""
Analysis Module

Multi-layer graph analysis for distributed pub-sub systems.

This module provides comprehensive analysis capabilities for identifying
critical components, detecting architectural problems, and assessing
reliability, maintainability, and availability of distributed systems.

Layers:
    - app: Application layer (app_to_app dependencies)
    - infra: Infrastructure layer (node_to_node dependencies)
    - mw-app: Middleware-Application layer (app_to_broker dependencies)
    - mw-infra: Middleware-Infrastructure layer (node_to_broker dependencies)
    - system: Complete system (all layers combined)

Components:
    - StructuralAnalyzer: Computes graph metrics (PageRank, Betweenness, etc.)
    - QualityAnalyzer: Computes R, M, A quality scores
    - BoxPlotClassifier: Adaptive classification using box-plot statistics
    - ProblemDetector: Identifies architectural issues and risks
    - GraphAnalyzer: Main orchestrator for analysis pipeline

Example:
    >>> from src.analysis import GraphAnalyzer, AnalysisLayer
    >>> 
    >>> with GraphAnalyzer(uri="bolt://localhost:7687") as analyzer:
    ...     # Analyze application layer
    ...     result = analyzer.analyze_layer(AnalysisLayer.APP)
    ...     
    ...     # Get critical components
    ...     critical = result.quality.get_critical_components()
    ...     
    ...     # Get detected problems
    ...     problems = result.problems
"""

# Layer definitions
from .layers import (
    AnalysisLayer,
    LayerDefinition,
    LAYER_DEFINITIONS,
    get_layer_definition,
    get_all_layers,
    get_primary_layers,
    DEPENDENCY_TO_LAYER,
)

# Box-plot classifier
from .classifier import (
    CriticalityLevel,
    BoxPlotClassifier,
    BoxPlotStats,
    ClassifiedItem,
    ClassificationResult,
    combine_levels,
    weighted_combine,
)

# Weight calculator
from .weight_calculator import (
    QualityWeights,
    AHPProcessor,
    AHPMatrices,
)

# Metrics data classes
from .metrics import (
    StructuralMetrics,
    EdgeMetrics,
    GraphSummary,
    QualityScores,
    QualityLevels,
    ComponentQuality,
    EdgeQuality,
    ClassificationSummary,
)

# Structural analyzer
from .structural_analyzer import (
    StructuralAnalyzer,
    StructuralAnalysisResult,
    extract_layer_subgraph,
)

# Quality analyzer
from .quality_analyzer import (
    QualityAnalyzer,
    QualityAnalysisResult,
)

# Problem detector
from .problem_detector import (
    ProblemDetector,
    DetectedProblem,
    ProblemSummary,
    ProblemCategory,
    ProblemSeverity,
)

# Main analyzer
from .analyzer import (
    GraphAnalyzer,
    LayerAnalysisResult,
    MultiLayerAnalysisResult,
    analyze_graph,
)


__all__ = [
    # Layers
    "AnalysisLayer",
    "LayerDefinition",
    "LAYER_DEFINITIONS",
    "get_layer_definition",
    "get_all_layers",
    "get_primary_layers",
    "DEPENDENCY_TO_LAYER",
    
    # Classifier
    "CriticalityLevel",
    "BoxPlotClassifier",
    "BoxPlotStats",
    "ClassifiedItem",
    "ClassificationResult",
    "combine_levels",
    "weighted_combine",

    # Weight Calculator
    "QualityWeights",
    "AHPProcessor",
    "AHPMatrices",
    
    # Metrics
    "StructuralMetrics",
    "EdgeMetrics",
    "GraphSummary",
    "QualityScores",
    "QualityLevels",
    "ComponentQuality",
    "EdgeQuality",
    "ClassificationSummary",
    
    # Structural Analyzer
    "StructuralAnalyzer",
    "StructuralAnalysisResult",
    "extract_layer_subgraph",
    
    # Quality Analyzer
    "QualityAnalyzer",
    "QualityAnalysisResult",
    
    # Problem Detector
    "ProblemDetector",
    "DetectedProblem",
    "ProblemSummary",
    "ProblemCategory",
    "ProblemSeverity",
    
    # Main Analyzer
    "GraphAnalyzer",
    "LayerAnalysisResult",
    "MultiLayerAnalysisResult",
    "analyze_graph",
]