"""
Domain Services Package

Pure domain logic services for graph analysis and quality assessment.
These services contain the core business logic without infrastructure dependencies.

Analysis pipeline:
    StructuralAnalyzer → QualityAnalyzer → ProblemDetector
         (metrics)        (RMAV + classify)   (risks)
"""

from .classifier import BoxPlotClassifier, combine_levels, weighted_combine
from .weight_calculator import QualityWeights, AHPProcessor, AHPMatrices
from .structural_analyzer import StructuralAnalyzer, StructuralAnalysisResult, extract_layer_subgraph
from .quality_analyzer import QualityAnalyzer, QualityAnalysisResult
from .problem_detector import ProblemDetector, DetectedProblem, ProblemSummary, ProblemCategory, ProblemSeverity

# Simulation services
from .event_simulator import EventSimulator, EventScenario, EventResult, RuntimeMetrics
from .failure_simulator import FailureSimulator, FailureScenario, FailureResult, ImpactMetrics

# Validation services
from .validator import Validator
from .metric_calculator import (
    calculate_correlation, calculate_error, calculate_classification, calculate_ranking,
)

__all__ = [
    # Classification
    "BoxPlotClassifier", "combine_levels", "weighted_combine",
    # Weights
    "QualityWeights", "AHPProcessor", "AHPMatrices",
    # Analysis pipeline
    "StructuralAnalyzer", "StructuralAnalysisResult", "extract_layer_subgraph",
    "QualityAnalyzer", "QualityAnalysisResult",
    "ProblemDetector", "DetectedProblem", "ProblemSummary",
    "ProblemCategory", "ProblemSeverity",
    # Simulation
    "EventSimulator", "EventScenario", "EventResult", "RuntimeMetrics",
    "FailureSimulator", "FailureScenario", "FailureResult", "ImpactMetrics",
    # Validation
    "Validator",
    "calculate_correlation", "calculate_error",
    "calculate_classification", "calculate_ranking",
]