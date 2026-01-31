"""
Analysis Result Domain Models

Aggregates for complete analysis results.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.services.structural_analyzer import StructuralAnalysisResult
    from src.domain.services.quality_analyzer import QualityAnalysisResult
    from src.domain.services.problem_detector import DetectedProblem, ProblemSummary


@dataclass
class LayerAnalysisResult:
    """
    Complete analysis result for a single layer.
    
    Contains structural metrics, quality scores, and detected problems.
    """
    layer: str
    layer_name: str
    description: str
    structural: "StructuralAnalysisResult"
    quality: "QualityAnalysisResult"
    problems: List["DetectedProblem"]
    problem_summary: "ProblemSummary"
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
    
    @property
    def is_healthy(self) -> bool:
        """Check if layer has no critical issues."""
        return self.problem_summary.by_severity.get("CRITICAL", 0) == 0
    
    @property
    def requires_attention(self) -> bool:
        """Check if layer has critical or high severity issues."""
        return self.problem_summary.requires_attention > 0


@dataclass
class MultiLayerAnalysisResult:
    """Complete analysis result across multiple layers."""
    timestamp: str
    layers: Dict[str, LayerAnalysisResult]
    cross_layer_insights: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "cross_layer_insights": self.cross_layer_insights,
        }
    
    def get_all_problems(self) -> List["DetectedProblem"]:
        """Collect all problems from all layers."""
        all_problems = []
        for layer_res in self.layers.values():
            all_problems.extend(layer_res.problems)
        return all_problems

    @property
    def summary(self) -> Dict[str, Any]:
        """Get high-level summary of analysis."""
        total_problems = sum(l.problem_summary.total_problems for l in self.layers.values())
        critical_probs = sum(l.problem_summary.by_severity.get("CRITICAL", 0) for l in self.layers.values())
        
        return {
            "layers_analyzed": len(self.layers),
            "total_components": sum(l.structural.graph_summary.nodes for l in self.layers.values()),
            "total_problems": total_problems,
            "critical_problems": critical_probs,
        }
