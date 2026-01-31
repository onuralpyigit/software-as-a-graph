"""
Domain Services Package

Pure domain logic services for graph analysis and quality assessment.
These services contain the core business logic without infrastructure dependencies.
"""

from .classifier import BoxPlotClassifier, combine_levels, weighted_combine
from .weight_calculator import QualityWeights, AHPProcessor, AHPMatrices
from .structural_analyzer import StructuralAnalyzer, StructuralAnalysisResult, extract_layer_subgraph
from .quality_analyzer import QualityAnalyzer, QualityAnalysisResult
from .problem_detector import ProblemDetector, DetectedProblem, ProblemSummary, ProblemCategory, ProblemSeverity

__all__ = [
    # Classifier
    "BoxPlotClassifier",
    "combine_levels",
    "weighted_combine",
    # Weights
    "QualityWeights",
    "AHPProcessor",
    "AHPMatrices",
    # Structural Analysis
    "StructuralAnalyzer",
    "StructuralAnalysisResult",
    "extract_layer_subgraph",
    # Quality Analysis
    "QualityAnalyzer",
    "QualityAnalysisResult",
    # Problem Detection
    "ProblemDetector",
    "DetectedProblem",
    "ProblemSummary",
    "ProblemCategory",
    "ProblemSeverity",
]
