"""
Compatibility shim: src.analysis

Provides backward-compatible imports for the old analysis module paths.
"""
from src.analysis.analyzer import GraphAnalyzer
from src.analysis.classifier import BoxPlotClassifier, CriticalityLevel
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.analysis.quality_analyzer import QualityAnalyzer
from src.analysis.problem_detector import ProblemDetector

__all__ = [
    "GraphAnalyzer", "BoxPlotClassifier", "CriticalityLevel",
    "StructuralAnalyzer", "QualityAnalyzer", "ProblemDetector",
]
