"""
Presenter for analysis results, decoupling domain models from API response formats.
"""

from typing import Dict, Any, List, Optional
from saag import AnalysisResult, PredictionResult

def serialize_component(c) -> Dict[str, Any]:
    """Convert a classified component to API response format."""
    return c.to_dict()


def serialize_edge(e, component_names: Dict[str, str]) -> Dict[str, Any]:
    """Convert a classified edge to API response format."""
    level = e.level.value if hasattr(e, 'level') and hasattr(e.level, 'value') else (str(e.level) if hasattr(e, 'level') else "minimal")
    scores = {}
    if hasattr(e, 'scores') and e.scores is not None:
        scores = {
            "reliability": e.scores.reliability,
            "maintainability": e.scores.maintainability,
            "availability": e.scores.availability,
            "vulnerability": e.scores.vulnerability,
            "overall": e.scores.overall,
        }
    return {
        "source": e.source,
        "target": e.target,
        "source_name": component_names.get(e.source, e.source),
        "target_name": component_names.get(e.target, e.target),
        "type": e.dependency_type,
        "criticality_level": level,
        "scores": scores,
    }


def serialize_problem(p) -> Dict[str, Any]:
    """Convert a detected problem to API response format."""
    return {
        "entity_id": p.entity_id,
        "type": p.entity_type,
        "category": p.category.value if hasattr(p.category, 'value') else str(p.category),
        "severity": p.severity.value if hasattr(p.severity, 'value') else str(p.severity),
        "name": p.name,
        "description": p.description,
        "recommendation": p.recommendation,
    }


def build_analysis_response(
    analysis: AnalysisResult,
    prediction: PredictionResult,
    problems: List[Any],
    context: str = "",
    description: str = "",
    component_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a standardised analysis response envelope.

    `components`, `edges`, and `problems` are the (possibly filtered) lists
    to serialise.  Summary statistics are computed from these lists.
    """
    # Extract values from SDK
    all_components = prediction.all_components
    layer_name = analysis.raw.layer.value if hasattr(analysis.raw.layer, 'value') else str(analysis.raw.layer)
    # EdgeQuality objects (have .level) come from the prediction result
    quality_edges = list(prediction.raw.edges)

    if component_type:
        all_components = [c for c in all_components if c.type == component_type]
        filtered_ids = {c.id for c in all_components}
        quality_edges = [e for e in quality_edges if e.source in filtered_ids or e.target in filtered_ids]
        problems = [p for p in problems if p.entity_id in filtered_ids]

    component_names = {c.id: c.name for c in all_components}

    summary = {
        "total_components": len(all_components),
        "critical_count": sum(1 for c in all_components if c.criticality_level == "critical"),
        "high_count": sum(1 for c in all_components if c.criticality_level == "high"),
        "total_problems": len(problems),
        "critical_problems": sum(
            1 for p in problems
            if (p.severity == "CRITICAL" or (hasattr(p.severity, 'value') and p.severity.value == "CRITICAL"))
        ),
        "components": {
            level: sum(1 for c in all_components if c.criticality_level == level)
            for level in ["critical", "high", "medium", "low", "minimal"]
        },
        "edges": {
            level: sum(
                1 for e in quality_edges
                if (e.level.value if hasattr(e.level, 'value') else str(e.level)) == level
            )
            for level in ["critical", "high", "medium", "low", "minimal"]
        },
    }

    envelope: Dict[str, Any] = {
        "success": True,
        "layer": layer_name,
    }
    if component_type:
        envelope["component_type"] = component_type

    envelope["analysis"] = {
        "context": context or layer_name.capitalize(),
        "description": description or f"Analysis for layer {layer_name}",
        "summary": summary,
        "stats": {
            "nodes": analysis.raw.graph_summary.nodes,
            "edges": analysis.raw.graph_summary.edges,
            "density": analysis.raw.graph_summary.density,
            "avg_degree": analysis.raw.graph_summary.avg_degree,
        },
        "components": [serialize_component(c) for c in all_components],
        "edges": [serialize_edge(e, component_names) for e in quality_edges],
        "problems": [serialize_problem(p) for p in problems],
    }
    return envelope
