"""
Analysis endpoints for system, type, and layer analysis.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging

from api.dependencies import get_analysis_service
from src.analysis import AnalysisService
from api.presenters import analysis_presenter
from api.models import AnalysisEnvelope

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/full", response_model=AnalysisEnvelope)
async def analyze_full_system(service: AnalysisService = Depends(get_analysis_service)):
    """
    Run complete system analysis including:
    - Structural metrics (centrality, clustering, etc.)
    - Quality scores (reliability, maintainability, availability)
    - Problem detection
    """
    try:
        logger.info("Running full system analysis")
        result = service.analyze_layer("system")
        return analysis_presenter.build_analysis_response(
            result,
            result.quality.components,
            result.quality.edges,
            result.problems,
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
        result = service.analyze_layer("system")

        # Filter components and edges
        filtered_components = [c for c in result.quality.components if c.type == normalized_type]
        filtered_ids = {c.id for c in filtered_components}
        filtered_edges = [e for e in result.quality.edges if e.source in filtered_ids or e.target in filtered_ids]
        filtered_problems = [p for p in result.problems if p.entity_id in filtered_ids]

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
async def analyze_layer(layer: str, service: AnalysisService = Depends(get_analysis_service)):
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
        result = service.analyze_layer(layer)
        return analysis_presenter.build_analysis_response(
            result,
            result.quality.components,
            result.quality.edges,
            result.problems,
        )
    except Exception as e:
        logger.error(f"Layer analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
