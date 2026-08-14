"""
Presenter for component/edge query endpoints, decoupling domain models from
API response formats.
"""

from typing import Any, Dict


def serialize_critical_edge(e: Any) -> Dict[str, Any]:
    """Convert a classified edge (EdgeQuality) to API response format."""
    return {
        "source": e.source,
        "target": e.target,
        "criticality_level": e.level.value if hasattr(e.level, "value") else str(e.level),
        "scores": {
            "reliability": e.scores.reliability,
            "maintainability": e.scores.maintainability,
            "fault_tolerance": e.scores.fault_tolerance,
            "availability": e.scores.availability,
            "overall": e.scores.overall,
        },
    }
