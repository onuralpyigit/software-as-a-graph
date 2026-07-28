"""
saag/analysis/smells.py — High-level Architectural Anti-Pattern API
"""
from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass

from saag.analysis.models import DetectedProblem


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
