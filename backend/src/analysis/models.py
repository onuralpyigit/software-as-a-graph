"""
Analysis Domain Models

Consolidated data structures for graph analysis results, including structural,
quality, and problem detection models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx
    from src.explanation.engine import SystemReport

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
    graph: Optional[nx.DiGraph] = None
    qos_profile: Dict[str, Any] = field(default_factory=dict)
    rcm_order: List[str] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer.value,
            "graph_summary": self.graph_summary.to_dict(),
            "components": [v.to_dict() for v in self.components.values()],
            "edges": [v.to_dict() for v in self.edges.values()],
        }

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

from src.prediction.models import QualityAnalysisResult, DetectedProblem, ProblemSummary

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
    prediction: Optional[Dict[str, Any]] = None
    explanation: Optional[SystemReport] = None
    
    @property
    def graph(self) -> Optional[nx.DiGraph]:
        """Proxy to the underlying NetworkX graph stored in structural results."""
        return self.structural.graph

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "layer_name": self.layer_name,
            "description": self.description,
            "graph_summary": self.structural.graph_summary.to_dict(),
            "structural_analysis": {
                "components": [
                    {
                        "id": c.id,
                        "metrics": {
                            "betweenness": c.betweenness,
                            "degree": c.degree,
                            "in_degree": c.in_degree,
                            "out_degree": c.out_degree,
                        }
                    }
                    for c in self.structural.components.values()
                ]
            },
            "quality_analysis": self.quality.to_dict(),
            "problems": [p.to_dict() for p in self.problems],
            "problem_summary": self.problem_summary.to_dict(),
            "library_usage": self.library_usage,
            "node_allocations": self.node_allocations,
            "broker_routing": self.broker_routing,
            "prediction": self.prediction,
            "explanation": self.explanation.to_dict() if self.explanation else None,
        }

@dataclass
class CrossLayerInsight:
    """
    An insight derived from correlating results across two or more layers.

    insight_type values:
      - "compound_critical"  : component is CRITICAL or HIGH in ≥2 layers
      - "systemic_spof"      : component is an articulation point in ≥2 layers
      - "layer_concentration": one layer has a disproportionate CRITICAL fraction
    """
    component_id: str
    component_name: str
    insight_type: str
    layers_affected: List[str]
    severity: str      # "CRITICAL" | "HIGH" | "MEDIUM"
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "insight_type": self.insight_type,
            "layers_affected": self.layers_affected,
            "severity": self.severity,
            "description": self.description,
        }


@dataclass
class MultiLayerAnalysisResult:
    """Complete analysis across multiple layers."""
    timestamp: str
    layers: Dict[str, LayerAnalysisResult]
    cross_layer_insights: List[CrossLayerInsight]

    @property
    def summary(self) -> Dict[str, Any]:
        """Aggregate summary statistics across all layers."""
        total_components = sum(len(l.quality.components) for l in self.layers.values())
        total_problems = sum(l.problem_summary.total_problems for l in self.layers.values())
        critical_problems = sum(
            l.problem_summary.by_severity.get("CRITICAL", 0) for l in self.layers.values()
        )
        return {
            "layers_analyzed": len(self.layers),
            "total_components": total_components,
            "total_problems": total_problems,
            "critical_problems": critical_problems,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "cross_layer_insights": [i.to_dict() for i in self.cross_layer_insights],
            "summary": self.summary,
        }
