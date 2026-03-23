"""
Analysis endpoints for system, type, and layer analysis.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging

from api.dependencies import get_client
from saag import Client
from src.prediction import ProblemDetector, PredictionService, DetectedProblem
from api.presenters import analysis_presenter
from api.models import AnalysisEnvelope

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/full", response_model=AnalysisEnvelope)
async def analyze_full_system(
    client: Client = Depends(get_client)
):
    """
    Run complete system analysis including:
    - Structural metrics (centrality, clustering, etc.)
    - Quality scores (reliability, maintainability, availability)
    - Problem detection
    """
    try:
        logger.info("Running full system analysis via SDK Client")
        
        # SDK calls
        analysis = client.analyze(layer="system")
        prediction = client.predict(analysis)
        problems = client.detect_antipatterns(prediction)
        
        return analysis_presenter.build_analysis_response(
            analysis,
            prediction,
            problems,
        )
    except Exception as e:
        logger.error(f"Full analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/type/{component_type}", response_model=AnalysisEnvelope)
async def analyze_by_type(
    component_type: str,
    client: Client = Depends(get_client),
):
    """
    Run analysis filtered by component type.
    Accepts: node, app, broker, Application, Node, Broker
    """
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

        # SDK calls
        analysis = client.analyze(layer="system")
        prediction = client.predict(analysis)
        problems = client.detect_antipatterns(prediction)

        return analysis_presenter.build_analysis_response(
            analysis,
            prediction,
            problems,
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
    client: Client = Depends(get_client),
):
    """
    Analyze a specific architectural layer.
    """
    valid_layers = ["app", "infra", "application", "infrastructure", "system", "mw-app", "mw-infra", "middleware"]
    if layer not in valid_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer. Must be one of: {', '.join(valid_layers)}"
        )

    # Normalize frontend-friendly alias
    if layer == "middleware":
        layer = "mw-app"

    try:
        logger.info(f"Analyzing layer: {layer}")
        
        # SDK calls
        analysis = client.analyze(layer=layer)
        prediction = client.predict(analysis)
        problems = client.detect_antipatterns(prediction)
        
        return analysis_presenter.build_analysis_response(
            analysis,
            prediction,
            problems,
        )
    except Exception as e:
        logger.error(f"Layer analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
