"""
Component and edge query endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from api.models import Neo4jCredentials
from api.dependencies import get_repository
from saag.adapters import create_repository
from saag.core.ports.graph_repository import IGraphRepository
from saag import Client

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
            client = Client(repo=repo)
            analysis = client.analyze(layer="system")
            prediction = client.predict(analysis)
            
            # Sort components by overall score and take top N
            components = sorted(
                prediction.all_components,
                key=lambda c: c.rmav_score,
                reverse=True
            )[:limit]
            
            # Format components for response
            formatted_components = [c.to_dict() for c in components]
            
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
            client = Client(repo=repo)
            analysis = client.analyze(layer="system")
            prediction = client.predict(analysis)
            
            # Prediction doesn't wrap edges, access internal quality via .raw
            # Sort edges by overall score and take top N
            edges = sorted(
                prediction.raw.edges,
                key=lambda e: e.scores.overall,
                reverse=True
            )[:limit]
            
            # Format edges for response
            formatted_edges = [
                {
                    "source": e.source,
                    "target": e.target,
                    "criticality_level": e.level.value if hasattr(e.level, 'value') else str(e.level),
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


@router.get("/apps/csms-names", response_model=Dict[str, Any])
async def get_unique_csms_names(
    repo: IGraphRepository = Depends(get_repository),
):
    """
    Return the unique, non-empty values of the ``csms_name`` field
    that exist on Application nodes in Neo4j.
    """
    try:
        driver = repo.driver  # type: ignore[attr-defined]
        database = repo.database  # type: ignore[attr-defined]
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (a:Application)
                WHERE a.csms_name IS NOT NULL AND trim(a.csms_name) <> ''
                RETURN DISTINCT a.csms_name AS csms_name
                ORDER BY a.csms_name
                """
            )
            csms_names = [record["csms_name"] for record in result]
        return {
            "success": True,
            "count": len(csms_names),
            "csms_names": csms_names,
        }
    except Exception as e:
        logger.error(f"CSMS names query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
