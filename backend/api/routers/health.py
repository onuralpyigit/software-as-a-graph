"""
Health check and connection endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime

from api.models import Neo4jCredentials, HealthResponse
from src.core import create_repository

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Distributed System Graph API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "stats": "/api/v1/stats",
            "generate": "/api/v1/graph/generate",
            "import": "/api/v1/graph/import",
            "export": "/api/v1/graph/export",
            "export_limited": "/api/v1/graph/export-limited",
            "export_neo4j_data": "/api/v1/graph/export-neo4j-data",
            "generate_and_import": "/api/v1/graph/generate-and-import",
            "analyze_full": "/api/v1/analysis/full",
            "analyze_type": "/api/v1/analysis/type/{component_type}",
            "analyze_layer": "/api/v1/analysis/layer/{layer}",
            "classify": "/api/v1/classify",
            "components": "/api/v1/components",
            "edges": "/api/v1/edges",
            "critical_components": "/api/v1/components/critical",
            "critical_edges": "/api/v1/edges/critical",
            "simulate_event": "/api/v1/simulation/event",
            "simulate_failure": "/api/v1/simulation/failure",
            "simulate_exhaustive": "/api/v1/simulation/exhaustive",
            "simulation_report": "/api/v1/simulation/report"
        }
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Verifies API is running.
    Note: Neo4j connection is validated via the /api/v1/connect endpoint.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        neo4j_connected=False,
        message="API is running. Configure Neo4j connection in settings."
    )


@router.post("/api/v1/connect", response_model=Dict[str, Any])
async def test_connection(credentials: Neo4jCredentials):
    """
    Test Neo4j database connection with provided credentials.
    This endpoint is used by the frontend to validate credentials.
    """
    repo = None
    try:
        logger.info(f"Testing Neo4j connection to {credentials.uri}")
        repo = create_repository(credentials.uri, credentials.user, credentials.password)
        repo.check_connection()
        
        return {
            "success": True,
            "message": "Successfully connected to Neo4j",
            "neo4j_connected": True
        }
    except Exception as e:
        logger.error(f"Neo4j connection test failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Connection failed: {str(e)}")
    finally:
        if repo:
            repo.close()
