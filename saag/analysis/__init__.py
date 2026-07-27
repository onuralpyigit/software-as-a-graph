"""
Analysis Package
"""
from .service import AnalysisService
from .cross_layer import compute_cross_layer_insights
from .structural_analyzer import StructuralAnalyzer
from .statistics import (
    extract_cross_cutting_data,
    compute_all_extras_statistics,
    analyze_for_bottleneck,
    compute_bottleneck_stats_from_structural,
)
from .models import (
    StructuralAnalysisResult,
    LayerAnalysisResult,
    MultiLayerAnalysisResult,
    QualityAnalysisResult,
    DetectedProblem,
    ProblemSummary,
)
from .antipattern_detector import AntiPatternDetector, CATALOG, PatternSpec
from .analyzer import QualityAnalyzer, CriticalityProfile
from .classifier import BoxPlotClassifier
from .weight_calculator import AHPProcessor, QualityWeights
from .problem_detector import ProblemDetector

__all__ = [
    "AnalysisService",
    "compute_cross_layer_insights",
    "StructuralAnalyzer",
    "StructuralAnalysisResult",
    "LayerAnalysisResult",
    "MultiLayerAnalysisResult",
    "AntiPatternDetector",
    "CATALOG",
    "PatternSpec",
    "QualityAnalyzer",
    "CriticalityProfile",
    "BoxPlotClassifier",
    "AHPProcessor",
    "QualityWeights",
    "ProblemDetector",
    "QualityAnalysisResult",
    "DetectedProblem",
    "ProblemSummary",
]
