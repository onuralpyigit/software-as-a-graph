"""
Component and edge query endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from api.models import Neo4jCredentials
from src.core import create_repository
from src.analysis import AnalysisService

router = APIRouter(prefix="/api/v1", tags=["components", "edges"])
logger = logging.getLogger(__name__)


@router.post("/components", response_model=Dict[str, Any])
async def get_components(
    credentials: Neo4jCredentials,
    component_type: Optional[str] = Query(None, description="Filter by component type"),
    min_weight: Optional[float] = Query(None, description="Minimum weight threshold"),
    limit: int = Query(100, description="Maximum number of components to return")
):
    """
    Get components from the graph with optional filtering.
    """
    try:
        logger.info(f"Querying components: type={component_type}, min_weight={min_weight}")
        
        repo = create_repository(credentials.uri, credentials.user, credentials.password)
        try:
            result = repo.get_components_with_filter(component_type, min_weight, limit)
            
            return {
                "success": True,
                "count": result["count"],
                "components": result["components"]
            }
        finally:
            repo.close()
    except Exception as e:
        logger.error(f"Component query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/components/critical", response_model=Dict[str, Any])
async def get_critical_components(
    credentials: Neo4jCredentials,
    limit: int = Query(20, description="Maximum number of components to return")
):
    """
    Get the most critical components based on analysis.
    """
    try:
        logger.info("Querying critical components")
        
        repo = create_repository(credentials.uri, credentials.user, credentials.password)
        try:
            service = AnalysisService(repo)
            result = service.analyze_layer("system")
            
            # Sort components by overall score and take top N
            components = sorted(
                result.quality.components,
                key=lambda c: c.scores.overall,
                reverse=True
            )[:limit]
            
            # Format components for response
            formatted_components = [
                {
                    "id": c.id,
                    "type": c.type,
                    "criticality_level": c.levels.overall.value,
                    "criticality_levels": {
                        "reliability": c.levels.reliability.value,
                        "maintainability": c.levels.maintainability.value,
                        "availability": c.levels.availability.value,
                        "vulnerability": c.levels.vulnerability.value,
                        "overall": c.levels.overall.value
                    },
                    "scores": {
                        "reliability": c.scores.reliability,
                        "maintainability": c.scores.maintainability,
                        "availability": c.scores.availability,
                        "vulnerability": c.scores.vulnerability,
                        "overall": c.scores.overall
                    }
                }
                for c in components
            ]
            
            return {
                "success": True,
                "count": len(formatted_components),
                "components": formatted_components
            }
        finally:
            repo.close()
    except Exception as e:
        logger.error(f"Critical components query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/edges", response_model=Dict[str, Any])
async def get_edges(
    credentials: Neo4jCredentials,
    dependency_type: Optional[str] = Query(None, description="Filter by dependency type"),
    min_weight: Optional[float] = Query(None, description="Minimum weight threshold"),
    limit: int = Query(100, description="Maximum number of edges to return")
):
    """
    Get edges from the graph with optional filtering.
    
    Valid dependency types: app_to_app, node_to_node, app_to_broker, node_to_broker
    """
    try:
        logger.info(f"Querying edges: type={dependency_type}, min_weight={min_weight}")
        
        repo = create_repository(credentials.uri, credentials.user, credentials.password)
        try:
            result = repo.get_edges_with_filter(dependency_type, min_weight, limit)
            
            return {
                "success": True,
                "count": result["count"],
                "edges": result["edges"]
            }
        finally:
            repo.close()
    except Exception as e:
        logger.error(f"Edge query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/edges/critical", response_model=Dict[str, Any])
async def get_critical_edges(
    credentials: Neo4jCredentials,
    limit: int = Query(20, description="Maximum number of edges to return")
):
    """
    Get the most critical edges based on analysis.
    """
    try:
        logger.info("Querying critical edges")
        
        repo = create_repository(credentials.uri, credentials.user, credentials.password)
        try:
            service = AnalysisService(repo)
            result = service.analyze_layer("system")
            
            # Sort edges by overall score and take top N
            edges = sorted(
                result.quality.edges,
                key=lambda e: e.scores.overall,
                reverse=True
            )[:limit]
            
            # Format edges for response
            formatted_edges = [
                {
                    "source": e.source,
                    "target": e.target,
                    "criticality_level": e.level.value,
                    "scores": {
                        "reliability": e.scores.reliability,
                        "maintainability": e.scores.maintainability,
                        "availability": e.scores.availability,
                        "vulnerability": e.scores.vulnerability,
                        "overall": e.scores.overall
                    }
                }
                for e in edges
            ]
            
            return {
                "success": True,
                "count": len(formatted_edges),
                "edges": formatted_edges
            }
        finally:
            repo.close()
    except Exception as e:
        logger.error(f"Critical edges query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
