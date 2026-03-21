"""
Analysis endpoints for system, type, and layer analysis.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging

from api.dependencies import (
    get_analysis_service, get_repository,
    get_analyze_graph_use_case, get_predict_graph_use_case
)
from src.core.ports.graph_repository import IGraphRepository
from src.analysis import AnalysisService
from src.prediction import ProblemDetector, PredictionService, DetectedProblem
from api.presenters import analysis_presenter
from api.models import AnalysisEnvelope
from src.usecases import AnalyzeGraphUseCase, PredictGraphUseCase, SimulateGraphUseCase, ValidateGraphUseCase
from src.prediction.models import QualityAnalysisResult

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/full", response_model=AnalysisEnvelope)
async def analyze_full_system(
    analyze_uc: AnalyzeGraphUseCase = Depends(get_analyze_graph_use_case),
    predict_uc: PredictGraphUseCase = Depends(get_predict_graph_use_case)
):
    """
    Run complete system analysis including:
    - Structural metrics (centrality, clustering, etc.)
    - Quality scores (reliability, maintainability, availability)
    - Problem detection
    """
    try:
        logger.info("Running full system analysis via injected use cases")
        
        # Structural Analysis
        s_res = analyze_uc.execute("system")
        
        # Quality Analysis (Prediction)
        q_res, detected_problems = predict_uc.execute("system", s_res, detect_problems=True)
        summary = ProblemDetector().summarize(detected_problems)
        
        # Layer Result Construction
        from src.analysis.models import LayerAnalysisResult
        layer_result = LayerAnalysisResult(
            layer="system",
            layer_name="System",
            description="Full system analysis",
            structural=s_res,
            quality=q_res,
            problems=detected_problems,
            problem_summary=summary
        )
        
        return analysis_presenter.build_analysis_response(
            layer_result,
            q_res.components,
            q_res.edges,
            detected_problems,
        )
    except Exception as e:
        logger.error(f"Full analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/type/{component_type}", response_model=AnalysisEnvelope)
async def analyze_by_type(component_type: str, service: AnalysisService = Depends(get_analysis_service)):
    """
    Run analysis filtered by component type.
    Accepts: node, app, broker, Application, Node, Broker
    """
    # Normalize component type (handle variations from frontend)
    type_mapping = {
        "application": "Application",
        "app": "Application",
        "node": "Node",
        "broker": "Broker",
    }

    normalized_type = type_mapping.get(component_type.lower())
    if not normalized_type:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid component type: {component_type}. Valid types: node, app, broker, Application, Node, Broker"
        )

    try:
        logger.info(f"Analyzing component type: {component_type} (normalized to {normalized_type})")
        # Step 2: Analysis
        result = service.analyze_layer("system")
        
        # Step 3: Prediction (using PredictionService directly for filtered view)
        from src.prediction.service import PredictionService
        pred_service = PredictionService()
        quality_res = pred_service.predict_quality(result.structural)

        # Filter components and edges
        filtered_components = [c for c in quality_res.components if c.type == normalized_type]
        filtered_ids = {c.id for c in filtered_components}
        filtered_edges = [e for e in quality_res.edges if e.source in filtered_ids or e.target in filtered_ids]
        filtered_problems = [p for p in quality_res.problems if p.entity_id in filtered_ids]

        return analysis_presenter.build_analysis_response(
            result,
            filtered_components,
            filtered_edges,
            filtered_problems,
            context=f"{normalized_type} Components Analysis",
            description=f"Analysis filtered by component type: {normalized_type}",
            component_type=normalized_type,
        )
    except Exception as e:
        logger.error(f"Type analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")




@router.post("/layer/{layer}", response_model=AnalysisEnvelope)
async def analyze_layer(
    layer: str, 
    analyze_uc: AnalyzeGraphUseCase = Depends(get_analyze_graph_use_case),
    predict_uc: PredictGraphUseCase = Depends(get_predict_graph_use_case)
):
    """
    Analyze a specific architectural layer.

    Valid layers: app, infra, application, infrastructure, system
    """
    valid_layers = ["app", "infra", "application", "infrastructure", "system", "mw-app", "mw-infra"]
    if layer not in valid_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer. Must be one of: {', '.join(valid_layers)}"
        )

    try:
        logger.info(f"Analyzing layer: {layer}")
        
        struct_res = analyze_uc.execute(layer)
        qual_res, problems = predict_uc.execute(layer, struct_res, detect_problems=True)
        summary = ProblemDetector().summarize(problems)
        
        from src.analysis.models import LayerAnalysisResult
        mock_result = LayerAnalysisResult(
            layer=layer,
            layer_name=layer.capitalize(),
            description=f"{layer.capitalize()} layer analysis",
            structural=struct_res,
            quality=qual_res,
            problems=problems,
            problem_summary=summary
        )
        
        return analysis_presenter.build_analysis_response(
            mock_result,
            qual_res.components,
            qual_res.edges,
            problems,
        )
    except Exception as e:
        logger.error(f"Layer analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
