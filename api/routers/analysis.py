"""
Analysis endpoints for system, type, and layer analysis.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Any
import logging

from api.dependencies import get_client
from saag import Client
from saag.models import AnalysisResult as SaagAnalysisResult, PredictionResult as SaagPredictionResult
from saag.core.layers import AnalysisLayer
from api.presenters import analysis_presenter
from api.models import AnalysisEnvelope

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)


def _analyze(client: Client, layer: str, **presenter_kwargs: Any) -> AnalysisEnvelope:
    """
    Run the multi-layer analysis use case for one layer and present the result.

    Shared by all three endpoints; ``presenter_kwargs`` carries the endpoint-
    specific context/description fields through to the presenter.
    """
    from saag.usecases.multi_layer_analysis import MultiLayerAnalysisUseCase

    use_case = MultiLayerAnalysisUseCase(client.repo)
    res = use_case.execute(layers=[layer])
    layer_res = res.layers[layer]

    return analysis_presenter.build_analysis_response(
        SaagAnalysisResult(layer_res),
        SaagPredictionResult(layer_res.prediction or layer_res.quality),
        layer_res.problems,
        **presenter_kwargs,
    )


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
        logger.info("Running full system analysis via MultiLayerAnalysisUseCase")
        return _analyze(client, "system")
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
        return _analyze(
            client, "system",
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
    try:
        layer_canonical = AnalysisLayer.from_string(layer).value
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    try:
        logger.info(f"Analyzing layer: {layer_canonical} (input: {layer})")
        return _analyze(client, layer_canonical)
    except Exception as e:
        logger.error(f"Layer analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
