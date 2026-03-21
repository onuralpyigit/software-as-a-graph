"""
Prediction Domain Models

Data structures for quality analysis results, classifications, and problem detection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from src.core.metrics import ComponentQuality, EdgeQuality
from src.core.criticality import CriticalityLevel, BoxPlotStats


@dataclass
class QualityAnalysisResult:
    """Complete quality analysis result for a single layer."""
    timestamp: str
    layer: str
    context: str
    components: List[ComponentQuality]
    edges: List[EdgeQuality]
    classification_summary: Any  # Avoid circular or complex imports for now
    weights: Any = None
    stats: Dict[str, BoxPlotStats] = field(default_factory=dict)
    sensitivity: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "timestamp": self.timestamp,
            "layer": self.layer,
            "context": self.context,
            "components": [c.to_dict() for c in self.components],
            "edges": [e.to_dict() for e in self.edges],
            "classification_summary": self.classification_summary.to_dict() if hasattr(self.classification_summary, "to_dict") else self.classification_summary,
        }
        if self.sensitivity:
            result["sensitivity"] = self.sensitivity
        return result

    def get_critical_components(self) -> List[ComponentQuality]:
        return [c for c in self.components if c.levels.overall == CriticalityLevel.CRITICAL]

    def get_high_priority(self) -> List[ComponentQuality]:
        return [c for c in self.components if c.levels.overall >= CriticalityLevel.HIGH]

    def get_by_type(self, comp_type: str) -> List[ComponentQuality]:
        return [c for c in self.components if c.type == comp_type]

    def get_critical_edges(self) -> List[EdgeQuality]:
        return [e for e in self.edges if e.level == CriticalityLevel.CRITICAL]

    def get_requiring_attention(self) -> tuple[List[ComponentQuality], List[EdgeQuality]]:
        comps = [c for c in self.components if c.requires_attention]
        edges = [e for e in self.edges if e.level >= CriticalityLevel.HIGH]
        return comps, edges


@dataclass
class DetectedProblem:
    """A detected architectural problem or risk."""
    entity_id: str
    entity_type: str           # Component | Edge | System
    category: str              # ProblemCategory value
    severity: str              # CRITICAL | HIGH | MEDIUM | LOW
    name: str
    description: str
    recommendation: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @property
    def priority(self) -> int:
        return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(self.severity, 0)


@dataclass
class ProblemSummary:
    """Aggregated problem counts."""
    total_problems: int
    by_severity: Dict[str, int]
    by_category: Dict[str, int]
    affected_components: int
    affected_edges: int

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @property
    def has_critical(self) -> bool:
        return self.by_severity.get("CRITICAL", 0) > 0

    @property
    def requires_attention(self) -> int:
        return self.by_severity.get("CRITICAL", 0) + self.by_severity.get("HIGH", 0)
