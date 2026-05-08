"""
Analysis Package
"""
from .service import AnalysisService
from .structural_analyzer import StructuralAnalyzer
from .statistics import (
    extract_cross_cutting_data,
    compute_all_extras_statistics,
    analyze_for_bottleneck,
    compute_bottleneck_stats_from_structural,
)
from .models import StructuralAnalysisResult, LayerAnalysisResult, MultiLayerAnalysisResult
from .antipattern_detector import AntiPatternDetector, CATALOG, PatternSpec

__all__ = [
    "AnalysisService",
    "StructuralAnalyzer",
    "StructuralAnalysisResult",
    "LayerAnalysisResult",
    "MultiLayerAnalysisResult",
    "AntiPatternDetector",
    "CATALOG",
    "PatternSpec",
]
