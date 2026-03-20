"""
Presenter for simulation results, formatting event and failure simulation data for API responses.
"""

from typing import Dict, Any, List

def format_event_simulation_response(result: Any) -> Dict[str, Any]:
    """Format event simulation result for API response."""
    return {
        "success": True,
        "simulation_type": "event",
        "result": result.to_dict() if hasattr(result, "to_dict") else result
    }


def format_failure_simulation_response(result: Any) -> Dict[str, Any]:
    """Format failure simulation result for API response."""
    return {
        "success": True,
        "simulation_type": "failure",
        "result": result.to_dict() if hasattr(result, "to_dict") else result
    }


def format_exhaustive_simulation_response(results: List[Any], layer: str) -> Dict[str, Any]:
    """Format exhaustive simulation results with summary stats."""
    # Create summary from results
    summary = {
        "total_components": len(results),
        "avg_impact": sum(r.impact.composite_impact for r in results) / len(results) if results else 0,
        "max_impact": max((r.impact.composite_impact for r in results), default=0),
        "critical_count": sum(1 for r in results if r.impact.composite_impact > 0.7),
        "high_count": sum(1 for r in results if 0.4 < r.impact.composite_impact <= 0.7),
        "medium_count": sum(1 for r in results if 0.2 < r.impact.composite_impact <= 0.4),
        "spof_count": sum(1 for r in results if r.impact.fragmentation > 0.01),
    }
    
    return {
        "success": True,
        "simulation_type": "exhaustive",
        "layer": layer,
        "summary": summary,
        "results": [r.to_dict() if hasattr(r, "to_dict") else r for r in results]
    }


def format_simulation_report_response(report: Any) -> Dict[str, Any]:
    """Format comprehensive simulation report for API response."""
    report_dict = report.to_dict() if hasattr(report, "to_dict") else report
    
    # Transform top_critical to match frontend expectations (nested structure)
    if "top_critical" in report_dict:
        report_dict["top_critical"] = [
            {
                "id": comp["id"],
                "type": comp["type"],
                "level": comp["level"],
                "scores": {
                    "event_impact": 0.0,
                    "failure_impact": 0.0,
                    "combined_impact": comp.get("combined_impact", 0.0),
                },
                "metrics": {
                    "cascade_count": comp.get("cascade_count", 0),
                    "message_throughput": 0,
                    "reachability_loss_percent": 0.0,
                },
            }
            for comp in report_dict["top_critical"]
        ]
    
    return {
        "success": True,
        "report": report_dict
    }
