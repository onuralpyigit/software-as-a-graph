"""
Presenter for analysis results, decoupling domain models from API response formats.
"""

from typing import Dict, Any, List, Optional
from src.analysis.models import LayerAnalysisResult

def serialize_component(c) -> Dict[str, Any]:
    """Convert a classified component to API response format."""
    return {
        "id": c.id,
        "name": c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id,
        "type": c.type,
        "criticality_level": c.levels.overall.value,
        "criticality_levels": {
            "reliability": c.levels.reliability.value,
            "maintainability": c.levels.maintainability.value,
            "availability": c.levels.availability.value,
            "vulnerability": c.levels.vulnerability.value,
            "overall": c.levels.overall.value,
        },
        "scores": {
            "reliability": c.scores.reliability,
            "maintainability": c.scores.maintainability,
            "availability": c.scores.availability,
            "vulnerability": c.scores.vulnerability,
            "overall": c.scores.overall,
        },
    }


def serialize_edge(e, component_names: Dict[str, str]) -> Dict[str, Any]:
    """Convert a classified edge to API response format."""
    return {
        "source": e.source,
        "target": e.target,
        "source_name": component_names.get(e.source, e.source),
        "target_name": component_names.get(e.target, e.target),
        "type": e.dependency_type,
        "criticality_level": e.level.value,
        "scores": {
            "reliability": e.scores.reliability,
            "maintainability": e.scores.maintainability,
            "availability": e.scores.availability,
            "vulnerability": e.scores.vulnerability,
            "overall": e.scores.overall,
        },
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
    result: LayerAnalysisResult,
    components,
    edges,
    problems,
    context: str = "",
    description: str = "",
    component_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a standardised analysis response envelope.

    `components`, `edges`, and `problems` are the (possibly filtered) lists
    to serialise.  Summary statistics are computed from these lists.
    """
    component_names = {
        c.id: (c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id)
        for c in components
    }

    summary = {
        "total_components": len(components),
        "critical_count": sum(1 for c in components if c.levels.overall.value == "critical"),
        "high_count": sum(1 for c in components if c.levels.overall.value == "high"),
        "total_problems": len(problems),
        "critical_problems": sum(
            1 for p in problems
            if (p.severity == "CRITICAL" or (hasattr(p.severity, 'value') and p.severity.value == "CRITICAL"))
        ),
        "components": {
            level: sum(1 for c in components if c.levels.overall.value == level)
            for level in ["critical", "high", "medium", "low", "minimal"]
        },
        "edges": {
            level: sum(1 for e in edges if e.level.value == level)
            for level in ["critical", "high", "medium", "low", "minimal"]
        },
    }

    envelope: Dict[str, Any] = {
        "success": True,
        "layer": result.layer,
    }
    if component_type:
        envelope["component_type"] = component_type

    envelope["analysis"] = {
        "context": context or result.layer_name,
        "description": description or result.description,
        "summary": summary,
        "stats": {
            "nodes": result.structural.graph_summary.nodes if hasattr(result, 'structural') and result.structural else len(components),
            "edges": result.structural.graph_summary.edges if hasattr(result, 'structural') and result.structural else len(edges),
            "density": result.structural.graph_summary.density if hasattr(result, 'structural') and result.structural else 0,
            "avg_degree": result.structural.graph_summary.avg_degree if hasattr(result, 'structural') and result.structural else 0,
        },
        "components": [serialize_component(c) for c in components],
        "edges": [serialize_edge(e, component_names) for e in edges],
        "problems": [serialize_problem(p) for p in problems],
    }
    return envelope
