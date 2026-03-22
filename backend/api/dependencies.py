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
from src.usecases import (
    ModelGraphUseCase,
    AnalyzeGraphUseCase,
    PredictGraphUseCase,
    SimulateGraphUseCase,
    ValidateGraphUseCase,
    VisualizeGraphUseCase
)

# ── Configuration ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


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

def get_model_graph_use_case(repo: IGraphRepository = Depends(get_repository)) -> ModelGraphUseCase:
    return ModelGraphUseCase(repo)

def get_analyze_graph_use_case(repo: IGraphRepository = Depends(get_repository)) -> AnalyzeGraphUseCase:
    return AnalyzeGraphUseCase(repo)

def get_predict_graph_use_case(repo: IGraphRepository = Depends(get_repository)) -> PredictGraphUseCase:
    return PredictGraphUseCase(repo)

def get_simulate_graph_use_case(repo: IGraphRepository = Depends(get_repository)) -> SimulateGraphUseCase:
    return SimulateGraphUseCase(repo)

def get_validate_graph_use_case(repo: IGraphRepository = Depends(get_repository)) -> ValidateGraphUseCase:
    return ValidateGraphUseCase(repo)

def get_visualize_graph_use_case(repo: IGraphRepository = Depends(get_repository)) -> VisualizeGraphUseCase:
    return VisualizeGraphUseCase(repo)
