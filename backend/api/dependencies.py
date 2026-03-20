"""
FastAPI dependency injection for API routes.

Provides:
  - Default Neo4j connection settings from environment
  - ``get_repository`` dependency: creates a request-scoped repository
    from the ``Neo4jCredentials`` body and ensures it is closed after use
"""

import os
import logging
from typing import AsyncGenerator

from fastapi import Depends, Request, HTTPException
from src.core import create_repository
from src.core.interfaces import IGraphRepository
from src.analysis import AnalysisService, StatisticsService
from src.simulation import SimulationService
from api.models import Neo4jCredentials

# ── Configuration ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DEFAULT_NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")


# ── Dependencies ─────────────────────────────────────────────────────────

async def get_repository(request: Request) -> AsyncGenerator[IGraphRepository, None]:
    """
    Request-scoped repository dependency.

    Extracts Neo4j credentials from the request body (either at the top level
    or nested under a 'credentials' key) and provides a repository instance.
    """
    body = await request.json()
    
    # Try to extract credentials from the 'credentials' key first (standard pattern)
    # Then fall back to the top-level body
    creds_data = body.get("credentials") or body
    
    try:
        credentials = Neo4jCredentials(**creds_data)
    except Exception as e:
        logging.error(f"Failed to parse Neo4j credentials from body: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid Neo4j credentials: {str(e)}"
        )

    repo = create_repository(
        uri=credentials.uri,
        user=credentials.user,
        password=credentials.password,
    )
    try:
        yield repo
    finally:
        repo.close()


def get_analysis_service(repo: IGraphRepository = Depends(get_repository)) -> AnalysisService:
    """
    Dependency to provide a configured AnalysisService instance.
    Automatically uses the request-scoped repository.
    """
    return AnalysisService(repo)


def get_simulation_service(repo: IGraphRepository = Depends(get_repository)) -> SimulationService:
    """
    Dependency to provide a configured SimulationService instance.
    Automatically uses the request-scoped repository.
    """
    return SimulationService(repo)


def get_statistics_service(repo: IGraphRepository = Depends(get_repository)) -> StatisticsService:
    """
    Dependency to provide a configured StatisticsService instance.
    Automatically uses the request-scoped repository.
    """
    return StatisticsService(repo)
