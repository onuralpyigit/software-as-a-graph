"""
Analysis Module

Multi-layer graph analysis for distributed pub-sub systems.

Components:
- StructuralAnalyzer: Computes graph metrics (PageRank, Betweenness, etc.)
- QualityAnalyzer: Computes R, M, A quality scores
- BoxPlotClassifier: Adaptive classification using box-plot statistics
- ProblemDetector: Identifies architectural issues and risks
- GraphAnalyzer: Main orchestrator for analysis pipeline

Author: Software-as-a-Graph Research Project
"""

from .classifier import (
    CriticalityLevel,
    BoxPlotClassifier,
    BoxPlotStats,
    ClassifiedItem,
    ClassificationResult,
    combine_levels,
)

from .structural_analyzer import (
    StructuralAnalyzer,
    StructuralMetrics,
    EdgeMetrics,
    StructuralAnalysisResult,
    GraphSummary,
    AnalysisLayer,
    LAYER_DEFINITIONS,
)

from .quality_analyzer import (
    QualityAnalyzer,
    QualityScores,
    QualityLevels,
    ComponentQuality,
    EdgeQuality,
    QualityAnalysisResult,
)

from .problem_detector import (
    ProblemDetector,
    DetectedProblem,
    ProblemSummary,
    ProblemCategory,
    ProblemSeverity,
)

from .analyzer import (
    GraphAnalyzer,
    LayerAnalysisResult,
    MultiLayerAnalysisResult,
    analyze_graph,
)


__all__ = [
    # Classifier
    "CriticalityLevel",
    "BoxPlotClassifier",
    "BoxPlotStats",
    "ClassifiedItem",
    "ClassificationResult",
    "combine_levels",
    
    # Structural Analyzer
    "StructuralAnalyzer",
    "StructuralMetrics",
    "EdgeMetrics",
    "StructuralAnalysisResult",
    "GraphSummary",
    "AnalysisLayer",
    "LAYER_DEFINITIONS",
    
    # Quality Analyzer
    "QualityAnalyzer",
    "QualityScores",
    "QualityLevels",
    "ComponentQuality",
    "EdgeQuality",
    "QualityAnalysisResult",
    
    # Problem Detector
    "ProblemDetector",
    "DetectedProblem",
    "ProblemSummary",
    "ProblemCategory",
    "ProblemSeverity",
    
    # Analyzer
    "GraphAnalyzer",
    "LayerAnalysisResult",
    "MultiLayerAnalysisResult",
    "analyze_graph",
]