"""
Validation endpoints for criticality score validation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import Field
from typing import Dict, Any, List, Optional
import logging
import json

from api.models import GraphRequestWithCredentials
from api.presenters.validation_presenter import (
    build_layers_response, build_pipeline_response,
    build_quick_response, build_targets_response,
)
from saag.adapters import create_repository
from saag.core import LAYER_DEFINITIONS
from saag.analysis import AnalysisService
from saag.prediction import PredictionService
from saag.simulation import SimulationService
from saag.validation import ValidationService, ValidationTargets, Validator

router = APIRouter(prefix="/api/v1/validation", tags=["validation"])
logger = logging.getLogger(__name__)


class ValidationRequest(GraphRequestWithCredentials):
    layers: List[str] = Field(default=["app", "infra", "system"], description="Layers to validate")
    include_comparisons: bool = Field(default=True, description="Include detailed component comparisons")


class QuickValidationRequest(GraphRequestWithCredentials):
    predicted_file: Optional[str] = Field(None, description="Path to predicted scores JSON file")
    actual_file: Optional[str] = Field(None, description="Path to actual scores JSON file")
    predicted_data: Optional[Dict[str, float]] = Field(None, description="Predicted scores dictionary")
    actual_data: Optional[Dict[str, float]] = Field(None, description="Actual scores dictionary")


def _validation_service(repo) -> ValidationService:
    """Wire a ValidationService against a repository."""
    return ValidationService(
        analysis_service=AnalysisService(repo),
        prediction_service=PredictionService(),
        simulation_service=SimulationService(repo),
        targets=ValidationTargets(),
    )


@router.post("/run-pipeline", response_model=Dict[str, Any])
async def run_validation_pipeline(request: ValidationRequest):
    """
    Run the full validation pipeline.

    This endpoint orchestrates:
    1. Graph analysis to get predicted criticality scores
    2. Failure simulation to get actual impact scores
    3. Statistical validation comparing predictions vs reality

    Args:
        request: Validation configuration with credentials and layers

    Returns:
        Complete validation results with metrics for each layer
    """
    repo = create_repository(
        uri=request.credentials.uri,
        user=request.credentials.user,
        password=request.credentials.password
    )
    try:
        logger.info(f"Starting validation pipeline for layers: {request.layers}")
        result = _validation_service(repo).validate_layers(layers=request.layers)
        return build_pipeline_response(result)
    except Exception as e:
        logger.error(f"Validation pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )
    finally:
        repo.close()


@router.post("/quick", response_model=Dict[str, Any])
async def quick_validation(request: QuickValidationRequest):
    """
    Quick validation from provided or file-based data.

    Compare predicted scores against actual scores using
    statistical validation metrics without running the full pipeline.

    Args:
        request: Predicted and actual scores (as files or data)

    Returns:
        Validation metrics and results
    """
    try:
        logger.info("Starting quick validation")

        predicted_scores = _load_scores(request.predicted_data, request.predicted_file)
        actual_scores = _load_scores(request.actual_data, request.actual_file)

        if not predicted_scores or not actual_scores:
            raise HTTPException(
                status_code=400,
                detail="Must provide either files or data for both predicted and actual scores"
            )

        # Purely statistical: no graph access, so no repository is opened.
        result = Validator(targets=ValidationTargets()).validate(
            predicted_scores, actual_scores, context="Quick Validation"
        )
        return build_quick_response(result)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Quick validation failed: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )


def _load_scores(data: Optional[Dict[str, float]], path: Optional[str]) -> Dict[str, float]:
    """Take inline scores if given, otherwise read them from a JSON file."""
    if data:
        return data
    if not path:
        return {}
    with open(path, 'r') as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


@router.get("/layers", response_model=Dict[str, Any])
async def get_validation_layers():
    """
    Get available validation layers and their definitions.

    Returns:
        Dictionary of layer definitions with descriptions
    """
    return build_layers_response(LAYER_DEFINITIONS)


@router.get("/targets", response_model=Dict[str, Any])
async def get_validation_targets():
    """
    Get default validation targets (success criteria).

    Returns:
        Dictionary of validation metrics and their target thresholds
    """
    return build_targets_response(ValidationTargets())
