"""
Problem Detector

Identifies architectural smells and risks using Granular Criticality Levels (R, M, A)
and structural context. Relies on Box-Plot outliers rather than static thresholds.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from .quality_analyzer import QualityAnalysisResult, CriticalityLevel

@dataclass
class DetectedProblem:
    entity_id: str
    entity_type: str
    aspect: str          # Reliability, Maintainability, Availability, Architecture
    severity: str        # CRITICAL, HIGH
    name: str
    description: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ProblemDetector:
    def detect(self, quality_result: QualityAnalysisResult) -> List[DetectedProblem]:
        problems = []
        
        # 1. Component Analysis
        for c in quality_result.components:
            # We focus on items classified as HIGH or CRITICAL by the Box-Plot Classifier
            
            # --- Availability Risks ---
            if c.levels.availability >= CriticalityLevel.HIGH:
                if c.structural.is_articulation_point:
                    problems.append(DetectedProblem(
                        c.id, c.type, "Availability", "CRITICAL",
                        "Single Point of Failure (SPOF)",
                        "Node is a structural cut-vertex. Removal disconnects the graph.",
                        "Introduce redundant paths or backup instances to bypass this node."
                    ))
                elif c.structural.bridge_ratio > 0.5:
                     problems.append(DetectedProblem(
                        c.id, c.type, "Availability", "HIGH",
                        "Structural Bridge",
                        f"Node controls {c.structural.bridge_ratio:.0%} of connected bridges.",
                        "Review network topology; failure here partitions the system clusters."
                    ))

            # --- Maintainability Risks ---
            if c.levels.maintainability >= CriticalityLevel.HIGH:
                 problems.append(DetectedProblem(
                    c.id, c.type, "Maintainability", c.levels.maintainability.value.upper(),
                    "God Component (High Coupling)",
                    f"High Betweenness & Degree. Tightly coupled to many parts of the system.",
                    "Refactor to split responsibilities or decouple using an event bus / broker."
                ))

            # --- Reliability Risks ---
            if c.levels.reliability >= CriticalityLevel.HIGH:
                problems.append(DetectedProblem(
                    c.id, c.type, "Reliability", c.levels.reliability.value.upper(),
                    "Failure Propagation Hub",
                    "High influence (Reverse PageRank) on the rest of the system.",
                    "Implement Circuit Breakers, Bulkheads, and robust retry logic here."
                ))

        # 2. Edge Analysis
        for e in quality_result.edges:
            if e.level >= CriticalityLevel.HIGH:
                is_structural = "structural" in e.type.lower() or "runs_on" in e.type.lower()
                name = "Critical Infrastructure Link" if is_structural else "Critical Logical Dependency"
                
                problems.append(DetectedProblem(
                    e.id, "Dependency", "Architecture", e.level.value.upper(),
                    f"{name} ({e.type})",
                    f"High combined criticality of endpoints (Score: {e.scores.overall:.2f}).",
                    "Ensure high bandwidth, monitoring, and asynchronous handling if possible."
                ))

        # Sort problems by severity (CRITICAL first)
        problems.sort(key=lambda p: 0 if p.severity == "CRITICAL" else 1)
        return problems