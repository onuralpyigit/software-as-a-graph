"""
backend/src/prediction/problem_detector.py — Unified Problem Detection Wrapper
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from types import SimpleNamespace

from src.analysis.antipattern_detector import AntiPatternDetector
from .models import QualityAnalysisResult, DetectedProblem, ProblemSummary

logger = logging.getLogger("problem_detector_compat")

class ProblemDetector:
    """
    Backward-compatible wrapper for the unified AntiPatternDetector.
    
    This class satisfies the original ProblemDetector API used by the prediction
    and CLI layers, but delegates all heavy lifting to the richer 
    AntiPatternDetector implementation.
    """

    def __init__(self, active_patterns: Optional[List[str]] = None) -> None:
        self._engine = AntiPatternDetector(active_patterns=active_patterns)

    def detect(self, quality: QualityAnalysisResult) -> List[DetectedProblem]:
        """
        Detect all problems from quality analysis results.
        
        Delegates to AntiPatternDetector, wrapping the quality object to match
        the expected interface.
        """
        # Create a shim for LayerAnalysisResult expected by engine
        shim = SimpleNamespace(quality=quality)
        layer_name = getattr(quality, "layer", "system")
        
        problems = self._engine.detect(shim, layer_name)
        
        # Original ProblemDetector sorted by (-priority, entity_id)
        # AntiPatternDetector already sorts by severity, but we can re-sort if needed
        return problems

    def summarize(self, problems: List[DetectedProblem]) -> ProblemSummary:
        """Create aggregated summary of detected problems."""
        # Use the engine's summary logic
        raw_summary = self._engine.summarize(problems)
        
        # Map to ProblemSummary dataclass
        comp_ids = {p.entity_id for p in problems if p.entity_type == "Component"}
        edge_ids = {p.entity_id for p in problems if p.entity_type == "Edge"}
        
        return ProblemSummary(
            total_problems=raw_summary["total"],
            by_severity=raw_summary["by_severity"],
            by_category=raw_summary["by_category"],
            affected_components=len(comp_ids),
            affected_edges=len(edge_ids)
        )