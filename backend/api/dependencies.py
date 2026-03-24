"""
FastAPI dependency injection for API routes.

Provides:
  - Default Neo4j connection settings from environment
  - ``get_repository`` dependency: creates a request-scoped repository
    from the ``Neo4jCredentials`` body and ensures it is closed after use
"""

import logging
from typing import AsyncGenerator

from fastapi import Depends, Request, HTTPException
from src.infrastructure import create_repository
from src.core.ports.graph_repository import IGraphRepository
from src.analysis import AnalysisService, StatisticsService
from src.prediction import PredictionService
from tools.generation import GenerationService
from api.models import (
    Neo4jCredentials,
    GenerateGraphRequest,
    GenerateGraphFileRequest
)
from src.infrastructure import config
from saag import Client, Pipeline

# ── Configuration ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ── Dependencies ─────────────────────────────────────────────────────────

async def get_repository(request: Request) -> AsyncGenerator[IGraphRepository, None]:
    """
    Request-scoped repository dependency.

    For POST/PUT requests: extracts Neo4j credentials from the JSON body
    (either at the top level or nested under a 'credentials' key).
    For GET requests (no body): reads credentials from query parameters
    (uri, user, password, database).
    """
    creds_data: dict = {}

    # Only attempt to read a JSON body for methods that carry one
    if request.method not in ("GET", "HEAD", "DELETE"):
        try:
            body = await request.json()
            creds_data = body.get("credentials") or body
        except Exception:
            pass  # Fall through to query-param extraction

    # Fall back to (or exclusively use) query parameters
    if not creds_data:
        params = dict(request.query_params)
        creds_data = {k: v for k, v in params.items() if k in ("uri", "user", "password", "database")}

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



def get_prediction_service() -> PredictionService:
    """
    Dependency to provide a PredictionService instance.
    """
    return PredictionService()


def get_statistics_service(repo: IGraphRepository = Depends(get_repository)) -> StatisticsService:
    """
    Dependency to provide a configured StatisticsService instance.
    Automatically uses the request-scoped repository.
    """
    return StatisticsService(repo)


def get_generation_service(request: GenerateGraphRequest | GenerateGraphFileRequest) -> GenerationService:
    """
    Dependency to provide a configured GenerationService instance.
    """
    return GenerationService(
        scale=request.scale, 
        seed=request.seed, 
        domain=request.domain, 
        scenario=request.scenario
    )

def get_client(repo: IGraphRepository = Depends(get_repository)) -> Client:
    """Dependency to provide an SDK Client using the request-scoped Neo4j repository."""
    return Client(repo=repo)

def get_pipeline(repo: IGraphRepository = Depends(get_repository)) -> Pipeline:
    """Dependency to provide an SDK Pipeline builder using the request-scoped Neo4j repository."""
    return Pipeline(repo=repo)
