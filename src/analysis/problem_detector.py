"""
Problem Detector

Identifies problems using dynamic Criticality Levels and Structural flags.
"""

from dataclasses import dataclass
from typing import List
from .quality_analyzer import QualityAnalysisResult, CriticalityLevel

@dataclass
class DetectedProblem:
    entity_id: str
    type: str
    category: str
    severity: str
    description: str
    recommendation: str

class ProblemDetector:
    def detect(self, quality_result: QualityAnalysisResult) -> List[DetectedProblem]:
        problems = []
        
        # 1. Component Problems
        for c in quality_result.components:
            # We use >= HIGH because these are statistical outliers in the top quartiles
            is_critical = c.level >= CriticalityLevel.HIGH
            
            # --- Availability Problems ---
            # SPOF: A critical node that is also an articulation point
            # (Note: Requires structural data, but we infer from High Avail Score + Level)
            if is_critical and c.scores.availability > 0.7: 
                 # High Avail Score is heavily weighted by Articulation Point status
                problems.append(DetectedProblem(
                    c.id, c.type, "Availability", c.level.value.upper(),
                    "Single Point of Failure (SPOF)",
                    "Introduce redundancy; add parallel paths or backup nodes."
                ))

            # --- Maintainability Problems ---
            # God Object / Bottleneck: High Maintainability Score (High Complexity/Centrality)
            if is_critical and c.scores.maintainability > 0.7:
                problems.append(DetectedProblem(
                    c.id, c.type, "Maintainability", c.level.value.upper(),
                    "Structural Bottleneck / God Component",
                    "Refactor to decouple responsibilities; split component."
                ))

            # --- Reliability Problems ---
            # Propagation Node: High Reliability Score (High In-Degree/Usage)
            if is_critical and c.scores.reliability > 0.7:
                problems.append(DetectedProblem(
                    c.id, c.type, "Reliability", c.level.value.upper(),
                    "Failure Propagation Hub",
                    "Implement Circuit Breakers and Bulkheads."
                ))

        # 2. Edge Problems
        for e in quality_result.edges:
            # Critical Dependency
            if e.level == CriticalityLevel.CRITICAL:
                problems.append(DetectedProblem(
                    e.id, "Dependency", "Architecture", "CRITICAL",
                    f"Critical Path Dependency ({e.type})",
                    "Optimize payload size; verify QoS requirements; ensure connection stability."
                ))

        return problems