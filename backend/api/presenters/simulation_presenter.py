"""
Presenter for simulation results, formatting event and failure simulation data for API responses.
"""

from typing import Dict, Any, List

def _latency_stats(metrics: Any) -> Dict[str, float]:
    """Compute min/p50/max from raw latency samples on the RuntimeMetrics object."""
    latencies = getattr(metrics, "latencies", [])
    if not latencies:
        return {"min_latency_ms": 0.0, "p50_latency_ms": 0.0, "max_latency_ms": 0.0}
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    raw_min = getattr(metrics, "min_latency", sorted_lat[0])
    safe_min = raw_min if raw_min != float("inf") else sorted_lat[0]
    p50 = sorted_lat[min(int(n * 0.50), n - 1)]
    raw_max = getattr(metrics, "max_latency", sorted_lat[-1])
    return {
        "min_latency_ms": round(safe_min * 1000, 3),
        "p50_latency_ms": round(p50 * 1000, 3),
        "max_latency_ms": round(raw_max * 1000, 3),
    }


def format_event_simulation_response(result: Any) -> Dict[str, Any]:
    """Format event simulation result for API response."""
    result_dict = result.to_dict() if hasattr(result, "to_dict") else result
    if isinstance(result_dict, dict) and "metrics" in result_dict:
        metrics_obj = getattr(result, "metrics", None)
        if metrics_obj is not None:
            result_dict["metrics"].update(_latency_stats(metrics_obj))
    return {
        "success": True,
        "simulation_type": "event",
        "result": result_dict,
    }


def _inject_impact_extras(impact_dict: Dict[str, Any], impact_obj: Any) -> None:
    """Inject fields that ImpactMetrics.to_dict() omits into an already-serialized impact dict."""
    impact_dict["affected"] = {
        "topics": getattr(impact_obj, "affected_topics", 0),
        "publishers": getattr(impact_obj, "affected_publishers", 0),
        "subscribers": getattr(impact_obj, "affected_subscribers", 0),
    }
    if "cascade" in impact_dict:
        impact_dict["cascade"]["by_type"] = dict(getattr(impact_obj, "cascade_by_type", {}))


def _augment_layer_metrics(layer_dict: Dict[str, Any], layer_obj: Any) -> None:
    """Add fields that LayerMetrics.to_dict() omits."""
    layer_dict["event_metrics"]["throughput"] = round(
        getattr(layer_obj, "event_throughput_per_sec", 0.0), 2
    )
    layer_dict["criticality"].update({
        "total_components": getattr(layer_obj, "total_components", 0),
        "medium": getattr(layer_obj, "medium_count", 0),
        "spof_count": getattr(layer_obj, "spof_count", 0),
    })


def format_failure_simulation_response(result: Any) -> Dict[str, Any]:
    """Format failure simulation result for API response."""
    result_dict = result.to_dict() if hasattr(result, "to_dict") else result
    if isinstance(result_dict, dict) and "impact" in result_dict:
        impact_obj = getattr(result, "impact", None)
        if impact_obj is not None:
            _inject_impact_extras(result_dict["impact"], impact_obj)
    return {
        "success": True,
        "simulation_type": "failure",
        "result": result_dict,
    }


def format_exhaustive_simulation_response(results: List[Any], layer: str) -> Dict[str, Any]:
    """Format exhaustive simulation results with summary stats."""
    summary = {
        "total_components": len(results),
        "avg_impact": sum(r.impact.composite_impact for r in results) / len(results) if results else 0,
        "max_impact": max((r.impact.composite_impact for r in results), default=0),
        "critical_count": sum(1 for r in results if r.impact.composite_impact > 0.7),
        "high_count": sum(1 for r in results if 0.4 < r.impact.composite_impact <= 0.7),
        "medium_count": sum(1 for r in results if 0.2 < r.impact.composite_impact <= 0.4),
        "low_count": sum(1 for r in results if 0.1 < r.impact.composite_impact <= 0.2),
        "spof_count": sum(1 for r in results if r.impact.fragmentation > 0.01),
    }

    serialized_results = []
    for r in results:
        r_dict = r.to_dict() if hasattr(r, "to_dict") else r
        if isinstance(r_dict, dict) and "impact" in r_dict:
            impact_obj = getattr(r, "impact", None)
            if impact_obj is not None:
                _inject_impact_extras(r_dict["impact"], impact_obj)
        serialized_results.append(r_dict)

    return {
        "success": True,
        "simulation_type": "exhaustive",
        "layer": layer,
        "summary": summary,
        "results": serialized_results,
    }


def format_simulation_report_response(report: Any) -> Dict[str, Any]:
    """Format comprehensive simulation report for API response."""
    report_dict = report.to_dict() if hasattr(report, "to_dict") else report

    # Inject fields that SimulationReport.to_dict() omits
    report_dict["graph_summary"] = dict(getattr(report, "graph_summary", {}) or {})
    report_dict["recommendations"] = list(getattr(report, "recommendations", []) or [])

    # Augment each layer's metrics with fields LayerMetrics.to_dict() omits
    raw_layer_metrics = getattr(report, "layer_metrics", {})
    for layer_name, layer_obj in (raw_layer_metrics or {}).items():
        if layer_name in report_dict.get("layer_metrics", {}):
            _augment_layer_metrics(report_dict["layer_metrics"][layer_name], layer_obj)

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
        "report": report_dict,
    }
