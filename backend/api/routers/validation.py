"""
Validation endpoints for criticality score validation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import Field
from typing import Dict, Any, List, Optional
import logging
import json

from api.models import GraphRequestWithCredentials
from src.core import create_repository, LAYER_DEFINITIONS
from src.analysis import AnalysisService
from src.simulation import SimulationService
from src.validation import ValidationService, ValidationTargets

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
        
        analysis_service = AnalysisService(repo)
        simulation_service = SimulationService(repo)
        validation_service = ValidationService(analysis_service, simulation_service, targets=ValidationTargets())
        
        # Run validation
        result = validation_service.validate_layers(layers=request.layers)
        
        # Transform the result to match frontend expectations
        result_dict = result.to_dict()
        
        # Enhance layer results with missing 'data' field expected by frontend
        enhanced_layers = {}
        for layer_key, layer_result in result_dict["layers"].items():
            # Get the original LayerValidationResult object to access all fields
            original_layer = result.layers[layer_key]
            
            # Add the missing 'data' field
            enhanced_layer = dict(layer_result)
            enhanced_layer["data"] = {
                "predicted_components": original_layer.predicted_components,
                "simulated_components": original_layer.simulated_components,
                "matched_components": original_layer.matched_components,
            }
            
            # Ensure summary has all required fields
            if "summary" not in enhanced_layer:
                enhanced_layer["summary"] = {}
            enhanced_layer["summary"].update({
                "passed": original_layer.passed,
                "spearman": round(original_layer.spearman, 4),
                "f1_score": round(original_layer.f1_score, 4),
                "precision": round(original_layer.precision, 4),
                "recall": round(original_layer.recall, 4),
                "top_5_overlap": round(original_layer.top_5_overlap, 4),
                "rmse": round(original_layer.rmse, 4),
            })
            
            enhanced_layers[layer_key] = enhanced_layer
        
        # Restructure response for frontend compatibility
        transformed_result = {
            "timestamp": result_dict["timestamp"],
            "summary": {
                "total_components": result_dict["total_components"],
                "layers_validated": len(result_dict["layers"]),
                "layers_passed": result_dict["layers_passed"],
                "all_passed": result_dict["all_passed"],
            },
            "layers": enhanced_layers,
            "cross_layer_insights": result_dict.get("warnings", []),
            "targets": result_dict["targets"],
        }
        
        return {
            "success": True,
            "result": transformed_result
        }
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
    # Note: quick_validation is mostly standalone statistics.
    # We still need a repo for create_repository if we want to follow the same pattern,
    # but validate_from_data might not actually use it.
    repo = create_repository(
        uri=request.credentials.uri,
        user=request.credentials.user,
        password=request.credentials.password
    )
    try:
        logger.info("Starting quick validation")
        
        # Load data
        predicted_scores = {}
        actual_scores = {}
        
        if request.predicted_data:
            predicted_scores = request.predicted_data
        elif request.predicted_file:
            with open(request.predicted_file, 'r') as f:
                data = json.load(f)
                predicted_scores = data if isinstance(data, dict) else {}
        
        if request.actual_data:
            actual_scores = request.actual_data
        elif request.actual_file:
            with open(request.actual_file, 'r') as f:
                data = json.load(f)
                actual_scores = data if isinstance(data, dict) else {}
        
        if not predicted_scores or not actual_scores:
            raise HTTPException(
                status_code=400,
                detail="Must provide either files or data for both predicted and actual scores"
            )
        
        analysis_service = AnalysisService(repo)
        simulation_service = SimulationService(repo)
        validation_service = ValidationService(analysis_service, simulation_service, targets=ValidationTargets())
        
        # Run quick validation
        result = validation_service.validate_from_data(
            predicted=predicted_scores,
            actual=actual_scores
        )
        
        return {
            "success": True,
            "result": result.to_dict()
        }
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
    finally:
        repo.close()


@router.get("/layers", response_model=Dict[str, Any])
async def get_validation_layers():
    """
    Get available validation layers and their definitions.
    
    Returns:
        Dictionary of layer definitions with descriptions
    """
    return {
        "success": True,
        "layers": LAYER_DEFINITIONS
    }


@router.get("/targets", response_model=Dict[str, Any])
async def get_validation_targets():
    """
    Get default validation targets (success criteria).
    
    Returns:
        Dictionary of validation metrics and their target thresholds
    """
    targets = ValidationTargets()
    return {
        "success": True,
        "targets": targets.to_dict()
    }
