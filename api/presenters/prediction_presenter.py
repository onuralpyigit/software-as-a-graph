"""
Presenter for GNN prediction results, formatting checkpoint metadata, node and
edge criticality scores, and evaluation metrics for API responses.

Returns plain dicts; the routers declare the Pydantic response models that
validate them.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def format_checkpoint_info(directory: Path) -> Optional[Dict[str, Any]]:
    """Return checkpoint metadata if *directory* looks like a valid GNN checkpoint."""
    cfg_path = directory / "service_config.json"
    node_path = directory / "node_model.pt"
    # Accept either node_model.pt or best_model.pt as the model file marker
    if not cfg_path.exists() or not (node_path.exists() or (directory / "best_model.pt").exists()):
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "path": str(directory),
        "name": directory.name,
        "layer": cfg.get("layer", ""),
        "hidden_channels": cfg.get("hidden_channels", 64),
        "num_heads": cfg.get("num_heads", 4),
        "num_layers": cfg.get("num_layers", 3),
        "dropout": cfg.get("dropout", 0.2),
        "predict_edges": cfg.get("predict_edges", True),
        "has_node_model": node_path.exists(),
        "has_edge_model": (directory / "edge_model.pt").exists(),
    }


def format_metrics(metrics: Any) -> Optional[Dict[str, Any]]:
    """Format an EvalMetrics object down to the subset the API exposes."""
    if metrics is None:
        return None
    return {
        field: getattr(metrics, field, None)
        for field in ("spearman_rho", "f1_score", "rmse", "mae", "ndcg_10")
    }


def format_node_score(score: Any, name_lookup: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Format a single GNNCriticalityScore, resolving the display name."""
    d = score.to_dict()
    node_id = d["component"]
    return {
        **d,
        "node_name": (name_lookup or {}).get(node_id, node_id),
        "criticality_level": d["criticality_level"].upper(),
        "source": d.get("source", "GNN"),
    }


def format_edge_score(edge: Any, name_lookup: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Format a single GNNEdgeCriticalityScore, resolving endpoint display names."""
    d = edge.to_dict()
    lookup = name_lookup or {}
    return {
        **d,
        "source_name": lookup.get(d["source"], d["source"]),
        "target_name": lookup.get(d["target"], d["target"]),
        "criticality_level": d["criticality_level"].upper(),
    }


def format_node_scores(
    scores: List[Any], name_lookup: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    return [format_node_score(s, name_lookup) for s in scores]


def format_edge_scores(
    edges: List[Any], name_lookup: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    return [format_edge_score(e, name_lookup) for e in edges]


def format_summary(result: Any) -> Dict[str, int]:
    """Level histogram for a GNNAnalysisResult."""
    return result.summary()
