"""
Analysis Domain Models

Consolidated data structures for graph analysis results, including structural,
quality, and problem detection models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

from src.core.layers import AnalysisLayer
from src.core.metrics import StructuralMetrics, EdgeMetrics, GraphSummary, QualityScores, QualityLevels, ComponentQuality, EdgeQuality
from src.core.criticality import CriticalityLevel, BoxPlotStats

# ---------------------------------------------------------------------------
# Structural Analysis Models
# ---------------------------------------------------------------------------

@dataclass
class StructuralAnalysisResult:
    """Container for all raw structural analysis results for one layer."""
    layer: AnalysisLayer
    components: Dict[str, StructuralMetrics]
    edges: Dict[Tuple[str, str], EdgeMetrics]
    graph_summary: GraphSummary

    def get_components_by_type(self, comp_type: str) -> List[StructuralMetrics]:
        return [c for c in self.components.values() if c.type == comp_type]

    def get_articulation_points(self) -> List[StructuralMetrics]:
        return [c for c in self.components.values() if c.is_articulation_point]

    def get_bridges(self) -> List[EdgeMetrics]:
        return [e for e in self.edges.values() if e.is_bridge]

    def get_top_by_metric(
        self, metric: str, n: int = 10, reverse: bool = True,
    ) -> List[StructuralMetrics]:
        return sorted(
            self.components.values(),
            key=lambda c: getattr(c, metric, 0),
            reverse=reverse,
        )[:n]

# ---------------------------------------------------------------------------
# Quality Analysis Models
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Problem Detection Models
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Aggregate Results
# ---------------------------------------------------------------------------

@dataclass
class LayerAnalysisResult:
    """
    Complete analysis result for a single layer.
    """
    layer: str
    layer_name: str
    description: str
    structural: StructuralAnalysisResult
    quality: QualityAnalysisResult
    problems: List[DetectedProblem]
    problem_summary: ProblemSummary
    library_usage: Dict[str, List[str]] = field(default_factory=dict)
    node_allocations: Dict[str, List[str]] = field(default_factory=dict)
    broker_routing: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "layer_name": self.layer_name,
            "description": self.description,
            "graph_summary": self.structural.graph_summary.to_dict(),
            "quality_analysis": self.quality.to_dict(),
            "problems": [p.to_dict() for p in self.problems],
            "problem_summary": self.problem_summary.to_dict(),
            "library_usage": self.library_usage,
            "node_allocations": self.node_allocations,
            "broker_routing": self.broker_routing,
        }

@dataclass
class MultiLayerAnalysisResult:
    """Complete analysis across multiple layers."""
    timestamp: str
    layers: Dict[str, LayerAnalysisResult]
    cross_layer_insights: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "cross_layer_insights": self.cross_layer_insights,
        }
