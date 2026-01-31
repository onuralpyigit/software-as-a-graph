"""
Analysis Sub-Module Exports

Re-exports from domain services layer for backward compatibility.
The actual implementation now lives in src/domain/services/.
"""

# Re-export from domain services (source of truth)
from src.domain.services.structural_analyzer import StructuralAnalyzer, StructuralAnalysisResult
from src.domain.services.quality_analyzer import QualityAnalyzer, QualityAnalysisResult
from src.domain.services.problem_detector import ProblemDetector, DetectedProblem, ProblemSummary
from src.domain.services.classifier import BoxPlotClassifier
from src.domain.services.weight_calculator import QualityWeights, AHPProcessor

__all__ = [
    "StructuralAnalyzer",
    "StructuralAnalysisResult",
    "QualityAnalyzer",
    "QualityAnalysisResult",
    "ProblemDetector",
    "DetectedProblem",
    "ProblemSummary",
    "BoxPlotClassifier",
    "QualityWeights",
    "AHPProcessor",
]
