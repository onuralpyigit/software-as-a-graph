"""
Analysis Result Domain Models

These are the **canonical** result aggregates for graph analysis.
Both single-layer and multi-layer results are defined here.

Note: The legacy ``src/domain/models/results.py`` should re-export from
this module to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, TYPE_CHECKING

from src.domain.config.layers import AnalysisLayer

if TYPE_CHECKING:
    from src.domain.services.structural_analyzer import StructuralAnalysisResult
    from src.domain.services.quality_analyzer import QualityAnalysisResult
    from src.domain.services.problem_detector import DetectedProblem, ProblemSummary


# ---------------------------------------------------------------------------
# Single-layer result
# ---------------------------------------------------------------------------

@dataclass
class LayerAnalysisResult:
    """
    Complete analysis result for a single layer.

    Contains structural metrics, RMAV quality scores (with box-plot
    classification), detected problems, and contextual data.
    """
    layer: str                                  # e.g. "app"
    layer_name: str                             # e.g. "Application Layer"
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
        """No CRITICAL issues in this layer."""
        return self.problem_summary.by_severity.get("CRITICAL", 0) == 0

    @property
    def requires_attention(self) -> bool:
        """Has CRITICAL or HIGH severity issues."""
        return self.problem_summary.requires_attention > 0


# ---------------------------------------------------------------------------
# Multi-layer result
# ---------------------------------------------------------------------------

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

    def get_all_problems(self) -> List["DetectedProblem"]:
        """Collect problems from every layer."""
        out: List["DetectedProblem"] = []
        for res in self.layers.values():
            out.extend(res.problems)
        return out

    @property
    def summary(self) -> Dict[str, Any]:
        """High-level summary across all layers."""
        total_problems = sum(r.problem_summary.total_problems for r in self.layers.values())
        total_critical = sum(
            r.problem_summary.by_severity.get("CRITICAL", 0) for r in self.layers.values()
        )
        all_ids = set()
        for res in self.layers.values():
            # structural is forward reference, but at runtime it is the object
            all_ids.update(res.structural.components.keys())
            
        return {
            "layers_analyzed": list(self.layers.keys()),
            "total_components": len(all_ids),
            "total_problems": total_problems,
            "total_critical": total_critical,
            "all_healthy": all(r.is_healthy for r in self.layers.values()),
        }