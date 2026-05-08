from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from api.models import Neo4jCredentials, GraphStatsResponse
from saag.core.ports.graph_repository import IGraphRepository
from saag.core.layers import AnalysisLayer
from api.dependencies import get_repository, get_client
from saag import Client
from saag.analysis.statistics import (
    analyze_for_bottleneck,
    extract_cross_cutting_data,
    compute_all_extras_statistics,
    compute_topic_bandwidth_stats,
    compute_app_balance_stats,
    compute_topic_fanout_stats,
    compute_cross_node_heatmap_stats,
    compute_node_comm_load_stats,
    compute_segment_comm_stats,
    compute_criticality_io_stats,
    compute_lib_dependency_stats,
    compute_node_critical_density_stats,
    compute_segment_diversity_stats,
    compute_bottleneck_stats_from_structural,
    compute_network_usage_stats,
    to_serializable,
)

router = APIRouter(prefix="/api/v1/stats", tags=["statistics"])
logger = logging.getLogger(__name__)


# ── Summary endpoint (used by connection store) ──────────────────────────

@router.post("/summary", response_model=GraphStatsResponse)
async def get_graph_stats(
    credentials: Neo4jCredentials,
    repo: IGraphRepository = Depends(get_repository),
):
    """Get overall graph statistics including structural relationships."""
    try:
        logger.info("Getting graph statistics")
        return {"success": True, "stats": repo.get_statistics()}
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@router.post("")
@router.post("/")
async def get_statistics(
    credentials: Neo4jCredentials,
    repo: IGraphRepository = Depends(get_repository),
):
    """Cross-cutting statistics computed from full graph export."""
    try:
        logger.info("Computing statistics")

        def default_risk_weight_fn(_, value: str) -> float:
            mapping = {"High": 3.0, "Medium": 2.0, "Low": 1.0, "NOT_FOUND": 1.0}
            return mapping.get(value, 1.0)

        raw_data = repo.export_json()
        cc = extract_cross_cutting_data(raw_data)
        stats = compute_all_extras_statistics(cc, risk_weight_fn=default_risk_weight_fn)

        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Statistics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")


# ── Per-chart lazy endpoints ─────────────────────────────────────────────

_CHART_FN_MAP = {
    "topic_bandwidth": compute_topic_bandwidth_stats,
    "app_balance": compute_app_balance_stats,
    "topic_fanout": compute_topic_fanout_stats,
    "cross_node_heatmap": compute_cross_node_heatmap_stats,
    "node_comm_load": compute_node_comm_load_stats,
    "domain_comm": compute_segment_comm_stats,
    "criticality_io": compute_criticality_io_stats,
    "lib_dependency": compute_lib_dependency_stats,
    "node_critical_density": compute_node_critical_density_stats,
    "domain_diversity": compute_segment_diversity_stats,
    "network_usage": compute_network_usage_stats,
}


@router.post("/chart/{chart_id}")
async def get_chart_statistics(
    chart_id: str,
    credentials: Neo4jCredentials,
    repo: IGraphRepository = Depends(get_repository),
):
    """Compute statistics for a single chart tab (lazy loading)."""
    if chart_id not in _CHART_FN_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown chart '{chart_id}'")
    try:
        logger.info(f"Computing statistics for chart: {chart_id}")
        raw_data = repo.export_json()
        cc = extract_cross_cutting_data(raw_data)
        result = _CHART_FN_MAP[chart_id](cc)
        return {
            "success": True,
            "chart_id": chart_id,
            "data": to_serializable(result),
        }
    except Exception as e:
        logger.error(f"Statistics failed for chart {chart_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")


@router.post("/bottleneck")
async def get_bottleneck_stats(
    credentials: Neo4jCredentials,
    client: Client = Depends(get_client),
):
    """Identify structural bottlenecks using real topological metrics.

    Ranks components by a composite bottleneck score derived from:
    - betweenness centrality (lies on many shortest paths, weight 0.40)
    - ap_c_directed (directed articulation-point severity, weight 0.25)
    - blast_radius_norm (fraction of graph made unreachable, weight 0.20)
    - bridge_ratio (fraction of incident edges that are bridges, weight 0.15)

    Articulation points and directed APs are also flagged explicitly.
    """
    try:
        logger.info("Computing structural bottleneck statistics")
        graph_data = client.repo.get_graph_data(include_raw=True)
        components_dict = analyze_for_bottleneck(graph_data, layer=AnalysisLayer.SYSTEM, use_structural=True)
        bottleneck_data = compute_bottleneck_stats_from_structural(components_dict)
        return {
            "success": True,
            "data": to_serializable(bottleneck_data),
        }
    except Exception as e:
        logger.error(f"Bottleneck statistics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")

