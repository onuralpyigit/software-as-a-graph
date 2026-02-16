"""
Statistics endpoints for graph metrics and distributions.

This router is a thin adapter that delegates all computation to
``StatisticsService``.  Each endpoint:
  1. Creates (or receives) a service instance
  2. Calls the corresponding service method
  3. Wraps the result in a ``{success, stats, ...}`` envelope
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from api.models import Neo4jCredentials
from src.analysis.statistics_service import StatisticsService

router = APIRouter(prefix="/api/v1/stats", tags=["statistics"])
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────

def _ok(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a stats dict in a success envelope."""
    return {
        "success": True,
        "stats": stats,
        "computation_time_ms": stats.get("computation_time_ms", 0),
    }


def _empty(defaults: Dict[str, Any], error: str = "Feature temporarily unavailable") -> Dict[str, Any]:
    """Return a graceful-degradation response with zero-value defaults."""
    return {"success": False, "error": error, "stats": defaults, "computation_time_ms": 0}


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("", response_model=Dict[str, Any])
@router.post("/", response_model=Dict[str, Any])
async def get_graph_stats(credentials: Neo4jCredentials):
    """Get overall graph statistics including structural relationships."""
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Getting graph statistics")
        return {"success": True, "stats": service.get_graph_stats()}
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    finally:
        service.close()


@router.post("/degree-distribution", response_model=Dict[str, Any])
async def get_degree_distribution_stats(credentials: Neo4jCredentials):
    """Degree distribution statistics (in/out/total, hubs, isolated nodes)."""
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing degree distribution statistics")
        return service.get_degree_distribution(node_type=credentials.node_type)
    except AttributeError as e:
        logger.warning(f"Degree distribution unavailable: {e}")
        return _empty({
            "in_degree": {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0},
            "out_degree": {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0},
            "total_degree": {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0},
            "hub_nodes": [], "isolated_nodes": 0, "total_nodes": 0,
            "hub_threshold": 0,
        })
    except Exception as e:
        logger.error(f"Degree distribution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")
    finally:
        service.close()


@router.post("/connectivity-density", response_model=Dict[str, Any])
async def get_connectivity_density_stats(credentials: Neo4jCredentials):
    """Connectivity density statistics."""
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing connectivity density statistics")
        return _ok(service.get_connectivity_density(node_type=credentials.node_type))
    except AttributeError as e:
        logger.warning(f"Connectivity density unavailable: {e}")
        return _empty({
            "density": 0, "total_nodes": 0, "total_edges": 0,
            "max_possible_edges": 0, "interpretation": "Data unavailable",
            "category": "unavailable", "most_dense_components": [],
        })
    except Exception as e:
        logger.error(f"Connectivity density failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")
    finally:
        service.close()


@router.post("/clustering-coefficient", response_model=Dict[str, Any])
async def get_clustering_coefficient_stats(credentials: Neo4jCredentials):
    """Clustering coefficient statistics with per-node breakdown."""
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing clustering coefficient statistics")
        return _ok(service.get_clustering_coefficient(node_type=credentials.node_type))
    except AttributeError as e:
        logger.warning(f"Clustering coefficient unavailable: {e}")
        return _empty({
            "avg_clustering_coefficient": 0, "global_clustering": 0,
            "average_clustering": 0, "max_coefficient": 0, "median_coefficient": 0,
            "min_coefficient": 0, "std_coefficient": 0,
            "high_clustering_count": 0, "medium_clustering_count": 0,
            "low_clustering_count": 0, "zero_clustering_count": 0,
            "total_nodes": 0, "high_clustering_nodes": [], "zero_clustering_nodes": [],
        })
    except Exception as e:
        logger.error(f"Clustering coefficient failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")
    finally:
        service.close()


@router.post("/dependency-depth", response_model=Dict[str, Any])
async def get_dependency_depth_stats(credentials: Neo4jCredentials):
    """Dependency depth statistics with categorisation and root/leaf lists."""
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing dependency depth statistics")
        return _ok(service.get_dependency_depth())
    except AttributeError as e:
        logger.warning(f"Dependency depth unavailable: {e}")
        return _empty({
            "max_depth": 0, "avg_depth": 0, "median_depth": 0, "min_depth": 0,
            "std_depth": 0, "interpretation": "Data unavailable", "category": "unavailable",
            "depth_distribution": {}, "shallow_count": 0, "low_depth_count": 0,
            "medium_depth_count": 0, "high_depth_count": 0, "total_nodes": 0,
            "root_nodes": [], "leaf_nodes": [], "deepest_components": [],
        })
    except Exception as e:
        logger.error(f"Dependency depth failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")
    finally:
        service.close()


@router.post("/component-isolation", response_model=Dict[str, Any])
async def get_component_isolation_stats(credentials: Neo4jCredentials):
    """Component isolation (isolated, source, sink, bidirectional)."""
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing component isolation statistics")
        return _ok(service.get_component_isolation())
    except AttributeError as e:
        logger.warning(f"Component isolation unavailable: {e}")
        return _empty({
            "isolated_count": 0, "isolated_percentage": 0,
            "source_count": 0, "source_percentage": 0,
            "sink_count": 0, "sink_percentage": 0,
            "bidirectional_count": 0, "bidirectional_percentage": 0,
            "category": "unavailable", "interpretation": "Data unavailable",
            "health": "unknown", "top_sources": [], "top_sinks": [],
            "isolated_components": [], "total_nodes": 0,
        })
    except Exception as e:
        logger.error(f"Component isolation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")
    finally:
        service.close()


@router.post("/message-flow-patterns", response_model=Dict[str, Any])
async def get_message_flow_patterns(credentials: Neo4jCredentials):
    """Message flow pattern statistics (pub/sub analysis)."""
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing message flow pattern statistics")
        return _ok(service.get_message_flow_patterns())
    except Exception as e:
        logger.error(f"Message flow pattern computation failed: {e}")
        return _empty({
            "total_topics": 0, "total_brokers": 0, "total_applications": 0,
            "active_applications": 0, "avg_publishers_per_topic": 0,
            "avg_subscribers_per_topic": 0, "avg_topics_per_broker": 0,
            "interpretation": "Data unavailable", "category": "error",
            "health": "unknown", "hot_topics": [], "broker_utilization": [],
            "isolated_applications": [], "top_publishers": [], "top_subscribers": [],
        }, error=str(e))
    finally:
        service.close()


@router.post("/component-redundancy", response_model=Dict[str, Any])
async def get_component_redundancy_stats(credentials: Neo4jCredentials):
    """Component redundancy / SPOF / resilience statistics."""
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing component redundancy statistics")
        return _ok(service.get_component_redundancy())
    except Exception as e:
        logger.error(f"Component redundancy computation failed: {e}")
        return _empty({
            "total_components": 0, "spof_count": 0, "spof_percentage": 0,
            "redundant_count": 0, "redundancy_percentage": 0, "resilience_score": 0,
            "interpretation": "Data unavailable", "category": "error",
            "health": "unknown", "single_points_of_failure": [], "bridge_components": [],
        }, error=str(e))
    finally:
        service.close()


@router.post("/node-weight-distribution", response_model=Dict[str, Any])
async def get_node_weight_distribution_stats(credentials: Neo4jCredentials):
    """Node weight distribution (degree-based importance)."""
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing node weight distribution")
        return _ok(service.get_node_weight_distribution())
    except Exception as e:
        logger.error(f"Node weight distribution failed: {e}")
        return _empty({
            "total_components": 0, "total_weight": 0, "avg_weight": 0,
            "median_weight": 0, "min_weight": 0, "max_weight": 0,
            "std_weight": 0, "weight_concentration": 0,
            "interpretation": "Data unavailable", "category": "error",
            "health": "unknown", "very_high_count": 0, "high_count": 0,
            "medium_count": 0, "low_count": 0, "very_low_count": 0,
            "top_components": [], "type_stats": {},
        }, error=str(e))
    finally:
        service.close()


@router.post("/edge-weight-distribution", response_model=Dict[str, Any])
async def get_edge_weight_distribution_stats(credentials: Neo4jCredentials):
    """Edge weight distribution with type-based breakdown."""
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing edge weight distribution")
        return _ok(service.get_edge_weight_distribution())
    except Exception as e:
        logger.error(f"Edge weight distribution failed: {e}")
        return _empty({
            "total_edges": 0, "total_weight": 0, "avg_weight": 0,
            "median_weight": 0, "min_weight": 0, "max_weight": 0,
            "std_weight": 0, "weight_concentration": 0,
            "interpretation": "Data unavailable", "category": "error",
            "health": "unknown", "very_high_count": 0, "high_count": 0,
            "medium_count": 0, "low_count": 0, "very_low_count": 0,
            "top_edges": [], "type_stats": {},
        }, error=str(e))
    finally:
        service.close()
