"""
Statistics endpoints for graph metrics and distributions.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from api.models import Neo4jCredentials
from src.analysis.statistics_service import StatisticsService

router = APIRouter(prefix="/api/v1/stats", tags=["statistics"])
logger = logging.getLogger(__name__)


@router.post("", response_model=Dict[str, Any])
@router.post("/", response_model=Dict[str, Any])
async def get_graph_stats(credentials: Neo4jCredentials):
    """
    Get overall graph statistics including structural relationships.
    """
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Getting graph statistics")
        stats = service.get_graph_stats()
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Stats query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    finally:
        service.close()


@router.post("/degree-distribution", response_model=Dict[str, Any])
async def get_degree_distribution_stats(credentials: Neo4jCredentials):
    """
    Get fast degree distribution statistics.
    
    Computes:
    - In-degree, out-degree, and total degree statistics (mean, median, max, min, std)
    - Hub nodes (degree > mean + 2*std)
    - Isolated nodes count
    
    Optionally filter by node_type to analyze specific component types.
    Runs in O(V+E) time - very fast even for large graphs.
    """
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        filter_msg = f" (filtered by type: {credentials.node_type})" if credentials.node_type else ""
        logger.info(f"Computing degree distribution statistics{filter_msg}")
        
        stats = service.get_degree_distribution(node_type=credentials.node_type)
        return stats
    except Exception as e:
        logger.error(f"Degree distribution computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
    finally:
        service.close()


@router.post("/connectivity-density", response_model=Dict[str, Any])
async def get_connectivity_density_stats(credentials: Neo4jCredentials):
    """
    Get connectivity density statistics - measures how interconnected the system is.
    """
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        filter_msg = f" (filtered by type: {credentials.node_type})" if credentials.node_type else ""
        logger.info(f"Computing connectivity density statistics{filter_msg}")
        
        stats = service.get_connectivity_density(node_type=credentials.node_type)
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except Exception as e:
        logger.error(f"Connectivity density computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
    finally:
        service.close()


@router.post("/clustering-coefficient", response_model=Dict[str, Any])
async def get_clustering_coefficient_stats(credentials: Neo4jCredentials):
    """
    Get clustering coefficient statistics - measures how nodes tend to cluster together.
    """
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        filter_msg = f" (filtered by type: {credentials.node_type})" if credentials.node_type else ""
        logger.info(f"Computing clustering coefficient statistics{filter_msg}")
        
        stats = service.get_clustering_coefficient(node_type=credentials.node_type)
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except Exception as e:
        logger.error(f"Clustering coefficient computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
    finally:
        service.close()


@router.post("/dependency-depth", response_model=Dict[str, Any])
async def get_dependency_depth_stats(credentials: Neo4jCredentials):
    """
    Get dependency depth statistics - measures the depth of dependency chains.
    """
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing dependency depth statistics")
        stats = service.get_dependency_depth()
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except Exception as e:
        logger.error(f"Dependency depth computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
    finally:
        service.close()


@router.post("/component-isolation", response_model=Dict[str, Any])
async def get_component_isolation_stats(credentials: Neo4jCredentials):
    """
    Get component isolation statistics - identifies isolated, source, and sink components.
    """
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing component isolation statistics")
        stats = service.get_component_isolation()
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except Exception as e:
        logger.error(f"Component isolation computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
    finally:
        service.close()


@router.post("/message-flow-patterns", response_model=Dict[str, Any])
async def get_message_flow_patterns(credentials: Neo4jCredentials):
    """
    Get message flow pattern statistics - analyzes communication patterns in pub-sub system.
    """
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing message flow pattern statistics")
        stats = service.get_message_flow_patterns()
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except Exception as e:
        logger.error(f"Message flow pattern computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
    finally:
        service.close()


@router.post("/component-redundancy", response_model=Dict[str, Any])
async def get_component_redundancy_stats(credentials: Neo4jCredentials):
    """
    Get component redundancy statistics - identifies SPOFs and bridge components.
    """
    try:
        logger.info("Computing component redundancy statistics")
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        result = service.get_component_redundancy()
        
        return {
            "success": result.get("success", True),
            "stats": result.get("stats", {}),
            "computation_time_ms": result.get("computation_time_ms", 0)
        }
    except Exception as e:
        logger.error(f"Component redundancy computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@router.post("/node-weight-distribution", response_model=Dict[str, Any])
async def get_node_weight_distribution_stats(credentials: Neo4jCredentials):
    """
    Get node weight distribution statistics - analyzes how component importance is distributed.
    
    Node weight distribution reveals:
    - Distribution of component weights (importance scores)
    - High-value vs low-value components
    - Weight concentration patterns
    - Critical component identification by weight
    
    Runs in O(V) - extremely fast.
    Provides insights into component importance hierarchy and architectural focus areas.
    """
    try:
        logger.info("Computing node weight distribution statistics")
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        stats = service.get_node_weight_distribution()
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Node weight distribution computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@router.post("/edge-weight-distribution", response_model=Dict[str, Any])
async def get_edge_weight_distribution_stats(credentials: Neo4jCredentials):
    """
    Get edge weight distribution statistics - analyzes how dependency importance is distributed.
    
    Edge weight distribution reveals:
    - Distribution of dependency weights (connection strength)
    - Critical vs weak dependencies
    - Weight concentration patterns
    - Dependency type importance patterns
    
    Runs in O(E) - extremely fast.
    Provides insights into dependency criticality and architectural coupling patterns.
    """
    try:
        logger.info("Computing edge weight distribution statistics")
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        stats = service.get_edge_weight_distribution()
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Edge weight distribution computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
