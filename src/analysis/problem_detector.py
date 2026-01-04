"""
Problem Detector

Identifies specific issues in Reliability, Maintainability, and Availability
by analyzing quality scores and structural flags.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

from .quality_analyzer import QualityAnalysisResult, ComponentQuality
from .classifier import CriticalityLevel

class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class DetectedProblem:
    component_id: str
    category: str # Reliability, Maintainability, Availability
    severity: Severity
    description: str
    symptoms: List[str]
    recommendation: str

class ProblemDetector:
    def detect(self, quality_result: QualityAnalysisResult) -> List[DetectedProblem]:
        problems = []
        
        for comp in quality_result.components:
            # 1. Availability Problems
            if comp.metrics.is_articulation_point:
                problems.append(DetectedProblem(
                    component_id=comp.id,
                    category="Availability",
                    severity=Severity.CRITICAL,
                    description="Single Point of Failure (SPOF)",
                    symptoms=["Component is an Articulation Point", "Removal disconnects graph"],
                    recommendation="Add redundancy: Replicate this component or add bypass paths."
                ))
            
            if comp.scores.availability > 0.7: # Heuristic threshold or based on classification
                problems.append(DetectedProblem(
                    component_id=comp.id,
                    category="Availability",
                    severity=Severity.HIGH,
                    description="High Availability Risk",
                    symptoms=[f"Availability Score: {comp.scores.availability:.2f}"],
                    recommendation="Ensure high uptime guarantees and rapid recovery mechanisms."
                ))

            # 2. Maintainability Problems
            # High Coupling (High Betweenness)
            if comp.metrics.betweenness > 0.5: # Threshold example
                problems.append(DetectedProblem(
                    component_id=comp.id,
                    category="Maintainability",
                    severity=Severity.HIGH,
                    description="Bottleneck Component (High Coupling)",
                    symptoms=[f"Betweenness Centrality: {comp.metrics.betweenness:.2f}"],
                    recommendation="Refactor to decouple responsibilities or split component."
                ))
            
            # Poor Modularity (Low Clustering)
            if comp.metrics.clustering_coefficient < 0.1 and comp.metrics.degree > 2:
                problems.append(DetectedProblem(
                    component_id=comp.id,
                    category="Maintainability",
                    severity=Severity.MEDIUM,
                    description="Poor Modularity",
                    symptoms=["Low Clustering Coefficient", "Neighbors not connected"],
                    recommendation="Review cohesion; components may be loosely related utilities."
                ))

            # 3. Reliability Problems
            if comp.metrics.failure_propagation > 0.5:
                problems.append(DetectedProblem(
                    component_id=comp.id,
                    category="Reliability",
                    severity=Severity.CRITICAL if comp.level == CriticalityLevel.CRITICAL else Severity.HIGH,
                    description="High Cascade Failure Potential",
                    symptoms=[f"Failure Propagation Score: {comp.metrics.failure_propagation:.2f}"],
                    recommendation="Implement circuit breakers and strict isolation."
                ))

        return problems