"""
Validation endpoints for criticality score validation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import Field
from typing import Dict, Any, List, Optional
import logging
import json

from api.models import GraphRequestWithCredentials
from src.adapters import create_repository
from src.core import LAYER_DEFINITIONS
from src.analysis import AnalysisService
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.analysis.models import LayerAnalysisResult
from src.core.layers import AnalysisLayer, get_layer_definition
from src.prediction import PredictionService
from src.prediction.models import ProblemSummary
from src.simulation import SimulationService
from src.validation import ValidationService, ValidationTargets
from saag import Client

router = APIRouter(prefix="/api/v1/validation", tags=["validation"])
logger = logging.getLogger(__name__)


class _SafeAnalysisService:
    """
    Wraps structural analysis without calling AntiPatternDetector, which
    incorrectly receives a StructuralAnalysisResult instead of a LayerAnalysisResult
    and crashes accessing .quality. Only the fields used by ValidationService are populated.
    """

    def __init__(self, repo):
        self._repo = repo

    def analyze_layer(self, layer: str) -> LayerAnalysisResult:
        graph_data = self._repo.get_graph_data()
        analyzer = StructuralAnalyzer()
        layer_enum = AnalysisLayer.from_string(layer)
        layer_def = get_layer_definition(layer_enum)
        struct_result = analyzer.analyze(graph_data, layer=layer_enum)
        empty_summary = ProblemSummary(
            total_problems=0,
            by_severity={},
            by_category={},
            affected_components=0,
            affected_edges=0,
        )
        return LayerAnalysisResult(
            layer=layer_enum.value,
            layer_name=layer_def.name,
            description=layer_def.description,
            structural=struct_result,
            quality=None,       # enriched by ValidationService after this call
            problems=[],
            problem_summary=empty_summary,
        )


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
        
        safe_analysis = _SafeAnalysisService(repo)
        prediction_svc = PredictionService()
        simulation_svc = SimulationService(repo)
        validation_svc = ValidationService(
            analysis_service=safe_analysis,
            prediction_service=prediction_svc,
            simulation_service=simulation_svc,
        )
        result = validation_svc.validate_layers(layers=request.layers)
        
        # Wrap in a minimal facade-compatible dict
        
        # Enhance layer results with missing 'data' field expected by frontend
        enhanced_layers = {}
        for layer_key, original_layer in result.layers.items():
            # Add the missing 'data' field
            enhanced_layer = original_layer.to_dict() if hasattr(original_layer, 'to_dict') else {}
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
            "timestamp": result.timestamp,
            "summary": {
                "total_components": result.total_components,
                "layers_validated": len(result.layers),
                "layers_passed": result.layers_passed,
                "all_passed": result.all_passed,
            },
            "layers": enhanced_layers,
            "cross_layer_insights": result.warnings,
            "targets": result.targets.to_dict() if result.targets else None,
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
        prediction_service = PredictionService()
        simulation_service = SimulationService(repo)
        validation_service = ValidationService(analysis_service, prediction_service, simulation_service, targets=ValidationTargets())
        
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
