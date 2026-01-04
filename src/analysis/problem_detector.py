"""
Problem Detector

Identifies problems using dynamic classification levels.
"""

from dataclasses import dataclass
from typing import List
from enum import Enum

from .quality_analyzer import QualityAnalysisResult, CriticalityLevel

@dataclass
class DetectedProblem:
    entity_id: str
    type: str # Component or Edge
    category: str
    severity: str
    description: str
    recommendation: str

class ProblemDetector:
    def detect(self, quality_result: QualityAnalysisResult) -> List[DetectedProblem]:
        problems = []
        
        # 1. Component Problems
        for c in quality_result.components:
            # Availability: SPOF
            if c.scores.availability > 0.8 and c.level >= CriticalityLevel.HIGH:
                problems.append(DetectedProblem(
                    c.id, "Component", "Availability", "CRITICAL",
                    f"High Availability Risk (Score: {c.scores.availability:.2f})",
                    "Add redundancy or bypass paths."
                ))
            
            # Maintainability: Bottleneck
            # High Maintainability Score means POOR maintainability (high cost)
            if c.scores.maintainability > 0.8 and c.level >= CriticalityLevel.HIGH:
                problems.append(DetectedProblem(
                    c.id, "Component", "Maintainability", "HIGH",
                    "Bottleneck / High Coupling",
                    "Refactor to decouple responsibilities."
                ))

            # Reliability: Propagation
            if c.scores.reliability > 0.8 and c.level >= CriticalityLevel.HIGH:
                problems.append(DetectedProblem(
                    c.id, "Component", "Reliability", "HIGH",
                    "High Failure Propagation Risk",
                    "Implement circuit breakers."
                ))

        # 2. Edge Problems
        for e in quality_result.edges:
            # Critical Dependency
            if e.level == CriticalityLevel.CRITICAL:
                problems.append(DetectedProblem(
                    e.id, "Edge", "Architecture", "CRITICAL",
                    f"Critical Dependency ({e.type})",
                    "Verify this dependency is necessary; optimize payload/QoS."
                ))

        return problems