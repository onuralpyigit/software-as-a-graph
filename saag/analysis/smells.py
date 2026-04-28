"""
saag/analysis/smells.py — High-level Architectural Anti-Pattern API
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from .antipattern_detector import AntiPatternDetector
from saag.prediction.models import DetectedProblem

if TYPE_CHECKING:
    from .models import LayerAnalysisResult

@dataclass
class AntiPatternReport:
    """Consolidated report for architectural anti-patterns."""
    problems: List[DetectedProblem]
    summary: Dict[str, Any]
    
    @property
    def total(self) -> int:
        return self.summary.get("total", 0)
        
    @property
    def by_severity(self) -> Dict[str, int]:
        return self.summary.get("by_severity", {})

    @property
    def by_category(self) -> Dict[str, int]:
        return self.summary.get("by_category", {})


class SmellDetector:
    """
    High-level detector for architectural smells and anti-patterns.
    Provides a consolidated API for analyzing multiple layers.
    """

    def __init__(self, active_patterns: Optional[List[str]] = None) -> None:
        self.engine = AntiPatternDetector(active_patterns=active_patterns)

    def detect_all(self, layer_results: List[LayerAnalysisResult]) -> AntiPatternReport:
        """
        Run anti-pattern detection across multiple layer results.
        
        Args:
            layer_results: List of LayerAnalysisResult objects to analyze.
            
        Returns:
            An AntiPatternReport containing all findings and summaries.
        """
        all_problems = []
        for result in layer_results:
            layer_id = getattr(result, "layer", "system")
            problems = self.engine.detect(result, layer_id)
            all_problems.extend(problems)
            
        summary = self.engine.summarize(all_problems)
        return AntiPatternReport(problems=all_problems, summary=summary)

    def print_findings(self, report: AntiPatternReport) -> None:
        """Print findings to stdout in a readable format."""
        if not report.problems:
            print("✨ No architectural anti-patterns detected.")
            return

        print(f"🔍 Detected {report.total} architectural anti-patterns:")
        
        # Group by severity
        for severity in ["CRITICAL", "HIGH", "MEDIUM"]:
            severity_problems = [p for p in report.problems if p.severity == severity]
            if not severity_problems:
                continue
                
            print(f"\n[{severity}] {len(severity_problems)} findings:")
            for p in severity_problems:
                entity = f"({p.entity_type}: {p.entity_id})"
                print(f"  • {p.name:<30} {entity}")
                print(f"    {p.description}")
                print(f"    Fix: {p.recommendation}")
