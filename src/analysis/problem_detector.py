"""
Problem Detector

Identifies problems using granular Criticality Levels (R, M, A)
and structural context. Uses NO static thresholds.
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
            # We focus on items that are statistical outliers (HIGH or CRITICAL)
            # in specific dimensions.
            
            # --- Availability: Single Point of Failure (SPOF) ---
            # Criteria: High Availability Criticality + Structural Articulation Point
            if c.levels.availability >= CriticalityLevel.HIGH:
                if c.structural.is_articulation_point:
                    problems.append(DetectedProblem(
                        c.id, c.type, "Availability", "CRITICAL",
                        "Single Point of Failure (SPOF)",
                        "Node is a cut-vertex. Add redundant paths or backup instances to bypass this node."
                    ))
                elif c.structural.bridge_ratio > 0:
                     problems.append(DetectedProblem(
                        c.id, c.type, "Availability", "HIGH",
                        "Structural Bridge",
                        "Node connects disparate clusters. Failure splits the network."
                    ))

            # --- Maintainability: God Object / Bottleneck ---
            # Criteria: High Maintainability Criticality (High Complexity + Centrality)
            if c.levels.maintainability >= CriticalityLevel.HIGH:
                 problems.append(DetectedProblem(
                    c.id, c.type, "Maintainability", c.levels.maintainability.value.upper(),
                    "God Component / Structural Bottleneck",
                    f"High Betweenness & Coupling. Refactor to split responsibilities. (Score: {c.scores.maintainability:.2f})"
                ))

            # --- Reliability: Failure Propagation Hub ---
            # Criteria: High Reliability Criticality (High Fan-In / Importance)
            if c.levels.reliability >= CriticalityLevel.HIGH:
                problems.append(DetectedProblem(
                    c.id, c.type, "Reliability", c.levels.reliability.value.upper(),
                    "Failure Propagation Hub",
                    "High downstream dependency count. Implement Circuit Breakers and Bulkheads here."
                ))

        # 2. Edge Problems
        for e in quality_result.edges:
            # Critical Dependencies
            if e.level >= CriticalityLevel.HIGH:
                issue_type = "Critical Bridge" if "bridge" in e.type.lower() or "structural" in e.type.lower() else "Critical Dependency"
                problems.append(DetectedProblem(
                    e.id, "Dependency", "Architecture", e.level.value.upper(),
                    f"{issue_type} ({e.type})",
                    "Optimize payload, ensure high bandwidth, and verify asynchronous handling."
                ))

        return problems