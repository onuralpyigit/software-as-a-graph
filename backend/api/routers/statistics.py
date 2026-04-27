from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging

from api.models import (
    Neo4jCredentials,
    GraphStatsResponse,
    DegreeDistributionResponse,
    ConnectivityDensityResponse,
    ClusteringCoefficientResponse,
    DependencyDepthResponse,
    ComponentIsolationResponse,
    MessageFlowPatternsResponse,
    ComponentRedundancyResponse,
    NodeWeightDistributionResponse,
    EdgeWeightDistributionResponse
)
from src.analysis.statistics_service import StatisticsService
from src.core.ports.graph_repository import IGraphRepository
from src.core.layers import AnalysisLayer
from api.bottleneck_fast import analyze_for_bottleneck
from api.dependencies import get_statistics_service, get_repository, get_client
from saag import Client
from api.presenters import statistics_presenter
from api.statistics import (
    extract_cross_cutting_data,
    compute_all_extras_statistics,
    compute_topic_bandwidth_stats,
    compute_app_balance_stats,
    compute_topic_fanout_stats,
    compute_cross_node_heatmap_stats,
    compute_node_comm_load_stats,
    compute_domain_comm_stats,
    compute_criticality_io_stats,
    compute_lib_dependency_stats,
    compute_node_critical_density_stats,
    compute_domain_diversity_stats,
    compute_bottleneck_stats_from_structural,
    to_serializable,
)

router = APIRouter(prefix="/api/v1/stats", tags=["statistics"])
logger = logging.getLogger(__name__)


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/summary", response_model=GraphStatsResponse)
async def get_graph_stats(
    credentials: Neo4jCredentials, 
    service: StatisticsService = Depends(get_statistics_service)
):
    """Get overall graph statistics including structural relationships."""
    try:
        logger.info("Getting graph statistics")
        return {"success": True, "stats": service.get_graph_stats()}
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@router.post("/degree-distribution", response_model=DegreeDistributionResponse)
async def get_degree_distribution_stats(
    credentials: Neo4jCredentials,
    service: StatisticsService = Depends(get_statistics_service)
):
    """Degree distribution statistics (in/out/total, hubs, isolated nodes)."""
    try:
        logger.info("Computing degree distribution statistics")
        result = service.get_degree_distribution(node_type=credentials.node_type)
        computation_time_ms = result.pop("computation_time_ms", 0.0)
        return {"success": True, "stats": result, "computation_time_ms": computation_time_ms}
    except AttributeError as e:
        logger.warning(f"Degree distribution unavailable: {e}")
        return statistics_presenter.format_empty_statistics_response(
            statistics_presenter.get_degree_distribution_defaults()
        )
    except Exception as e:
        logger.error(f"Degree distribution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")


@router.post("/connectivity-density", response_model=ConnectivityDensityResponse)
async def get_connectivity_density_stats(
    credentials: Neo4jCredentials,
    service: StatisticsService = Depends(get_statistics_service)
):
    """Connectivity density statistics."""
    try:
        logger.info("Computing connectivity density statistics")
        return statistics_presenter.format_statistics_response(
            service.get_connectivity_density(node_type=credentials.node_type)
        )
    except AttributeError as e:
        logger.warning(f"Connectivity density unavailable: {e}")
        return statistics_presenter.format_empty_statistics_response(
            statistics_presenter.get_connectivity_density_defaults()
        )
    except Exception as e:
        logger.error(f"Connectivity density failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")


@router.post("/clustering-coefficient", response_model=ClusteringCoefficientResponse)
async def get_clustering_coefficient_stats(
    credentials: Neo4jCredentials,
    service: StatisticsService = Depends(get_statistics_service)
):
    """Clustering coefficient statistics with per-node breakdown."""
    try:
        logger.info("Computing clustering coefficient statistics")
        return statistics_presenter.format_statistics_response(
            service.get_clustering_coefficient(node_type=credentials.node_type)
        )
    except AttributeError as e:
        logger.warning(f"Clustering coefficient unavailable: {e}")
        return statistics_presenter.format_empty_statistics_response(
            statistics_presenter.get_clustering_coefficient_defaults()
        )
    except Exception as e:
        logger.error(f"Clustering coefficient failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")


@router.post("/dependency-depth", response_model=DependencyDepthResponse)
async def get_dependency_depth_stats(
    credentials: Neo4jCredentials,
    service: StatisticsService = Depends(get_statistics_service)
):
    """Dependency depth statistics with categorisation and root/leaf lists."""
    try:
        logger.info("Computing dependency depth statistics")
        return statistics_presenter.format_statistics_response(service.get_dependency_depth())
    except AttributeError as e:
        logger.warning(f"Dependency depth unavailable: {e}")
        return statistics_presenter.format_empty_statistics_response(
            statistics_presenter.get_dependency_depth_defaults()
        )
    except Exception as e:
        logger.error(f"Dependency depth failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")


@router.post("/component-isolation", response_model=ComponentIsolationResponse)
async def get_component_isolation_stats(
    credentials: Neo4jCredentials,
    service: StatisticsService = Depends(get_statistics_service)
):
    """Component isolation (isolated, source, sink, bidirectional)."""
    try:
        logger.info("Computing component isolation statistics")
        return statistics_presenter.format_statistics_response(service.get_component_isolation())
    except AttributeError as e:
        logger.warning(f"Component isolation unavailable: {e}")
        return statistics_presenter.format_empty_statistics_response(
            statistics_presenter.get_component_isolation_defaults()
        )
    except Exception as e:
        logger.error(f"Component isolation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")


@router.post("/message-flow-patterns", response_model=MessageFlowPatternsResponse)
async def get_message_flow_patterns(
    credentials: Neo4jCredentials,
    service: StatisticsService = Depends(get_statistics_service)
):
    """Message flow pattern statistics (pub/sub analysis)."""
    try:
        logger.info("Computing message flow pattern statistics")
        return statistics_presenter.format_statistics_response(service.get_message_flow_patterns())
    except Exception as e:
        logger.error(f"Message flow pattern computation failed: {e}")
        return statistics_presenter.format_empty_statistics_response(
            statistics_presenter.get_message_flow_patterns_defaults(),
            error=str(e)
        )


@router.post("/component-redundancy", response_model=ComponentRedundancyResponse)
async def get_component_redundancy_stats(
    credentials: Neo4jCredentials,
    service: StatisticsService = Depends(get_statistics_service)
):
    """Component redundancy / SPOF / resilience statistics."""
    try:
        logger.info("Computing component redundancy statistics")
        return statistics_presenter.format_statistics_response(service.get_component_redundancy())
    except Exception as e:
        logger.error(f"Component redundancy computation failed: {e}")
        return statistics_presenter.format_empty_statistics_response(
            statistics_presenter.get_component_redundancy_defaults(),
            error=str(e)
        )


@router.post("/node-weight-distribution", response_model=NodeWeightDistributionResponse)
async def get_node_weight_distribution_stats(
    credentials: Neo4jCredentials,
    service: StatisticsService = Depends(get_statistics_service)
):
    """Node weight distribution (degree-based importance)."""
    try:
        logger.info("Computing node weight distribution")
        return statistics_presenter.format_statistics_response(service.get_node_weight_distribution())
    except Exception as e:
        logger.error(f"Node weight distribution failed: {e}")
        return statistics_presenter.format_empty_statistics_response(
            statistics_presenter.get_weight_distribution_defaults(),
            error=str(e)
        )


@router.post("/edge-weight-distribution", response_model=EdgeWeightDistributionResponse)
async def get_edge_weight_distribution_stats(
    credentials: Neo4jCredentials,
    service: StatisticsService = Depends(get_statistics_service)
):
    """Edge weight distribution with type-based breakdown."""
    try:
        logger.info("Computing edge weight distribution")
        return statistics_presenter.format_statistics_response(service.get_edge_weight_distribution())
    except Exception as e:
        logger.error(f"Edge weight distribution failed: {e}")
        return statistics_presenter.format_empty_statistics_response(
            statistics_presenter.get_weight_distribution_defaults(),
            error=str(e)
        )



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
            # Default weights for QoS risk scoring
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
    "domain_comm": compute_domain_comm_stats,
    "criticality_io": compute_criticality_io_stats,
    "lib_dependency": compute_lib_dependency_stats,
    "node_critical_density": compute_node_critical_density_stats,
    "domain_diversity": compute_domain_diversity_stats,
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
        graph_data = client.repo.get_graph_data()
        components_dict = analyze_for_bottleneck(graph_data, layer=AnalysisLayer.SYSTEM)
        bottleneck_data = compute_bottleneck_stats_from_structural(components_dict)
        return {
            "success": True,
            "data": to_serializable(bottleneck_data),
        }
    except Exception as e:
        logger.error(f"Bottleneck statistics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")

