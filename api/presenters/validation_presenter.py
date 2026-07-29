"""
Presenter for validation results, decoupling domain models from API response formats.

The key names emitted here are consumed directly by the web UI
(``smart/app/validation/page.tsx``), so treat them as a published contract.
"""

from typing import Any, Dict


def serialize_layer(layer) -> Dict[str, Any]:
    """Convert a LayerValidationResult to the shape the frontend expects."""
    payload = layer.to_dict() if hasattr(layer, "to_dict") else {}

    payload["data"] = {
        "predicted_components": layer.predicted_components,
        "simulated_components": layer.simulated_components,
        "matched_components": layer.matched_components,
    }

    summary = payload.setdefault("summary", {})
    summary.update({
        "passed": layer.passed,
        "spearman": round(layer.spearman, 4),
        "f1_score": round(layer.f1_score, 4),
        "precision": round(layer.precision, 4),
        "recall": round(layer.recall, 4),
        "top_5_overlap": round(layer.top_5_overlap, 4),
        "rmse": round(layer.rmse, 4),
    })
    return payload


def build_pipeline_response(result) -> Dict[str, Any]:
    """Full validation pipeline result."""
    return {
        "success": True,
        "result": {
            "timestamp": result.timestamp,
            "summary": {
                "total_components": result.total_components,
                "layers_validated": len(result.layers),
                "layers_passed": result.layers_passed,
                "all_passed": result.all_passed,
            },
            "layers": {k: serialize_layer(v) for k, v in result.layers.items()},
            "cross_layer_insights": result.warnings,
            "targets": result.targets.to_dict() if result.targets else None,
        },
    }


def build_quick_response(result) -> Dict[str, Any]:
    """Standalone predicted-vs-actual comparison."""
    return {"success": True, "result": result.to_dict()}


def build_targets_response(targets) -> Dict[str, Any]:
    """Default validation targets (success criteria)."""
    return {"success": True, "targets": targets.to_dict()}


def build_layers_response(layer_definitions) -> Dict[str, Any]:
    """Available validation layers and their definitions."""
    return {"success": True, "layers": layer_definitions}
