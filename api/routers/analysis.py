"""
Analysis endpoints for system, type, and layer analysis.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging

from types import SimpleNamespace

from api.dependencies import get_client
from saag import Client
from saag.models import AnalysisResult as SaagAnalysisResult, PredictionResult as SaagPredictionResult
from saag.prediction import ProblemDetector, PredictionService, DetectedProblem
from saag.usecases.predict_graph import PredictGraphUseCase as _PredictGraphUseCase
from saag.analysis.structural_analyzer import StructuralAnalyzer
from saag.analysis.antipattern_detector import AntiPatternDetector
from saag.core.layers import AnalysisLayer
from api.presenters import analysis_presenter
from api.models import AnalysisEnvelope

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)


def _structural_analyze(client: Client, layer: str) -> SaagAnalysisResult:
    """
    Run structural analysis directly via StructuralAnalyzer, bypassing AnalysisService
    which incorrectly passes StructuralAnalysisResult to AntiPatternDetector.detect().
    """
    graph_data = client.repo.get_graph_data()
    analyzer = StructuralAnalyzer()
    layer_enum = AnalysisLayer.from_string(layer)
    raw = analyzer.analyze(graph_data, layer=layer_enum)
    return SaagAnalysisResult(raw)


def _predict(analysis: SaagAnalysisResult) -> SaagPredictionResult:
    """
    Run quality prediction using an already-computed structural analysis result,
    bypassing client.predict() which incorrectly expects a layer string.
    """
    prediction_service = PredictionService()
    predict_uc = _PredictGraphUseCase(prediction_service)
    quality, _ = predict_uc.execute(
        layer="system",
        structural_result=analysis.raw,
        detect_problems=False,
    )
    return SaagPredictionResult(quality)


def _detect_antipatterns(prediction) -> list:
    """
    Detect antipatterns by calling AntiPatternDetector directly with a shim that
    includes the required 'components' attribute, bypassing the broken SimpleNamespace
    shim in ProblemDetector which omits 'components'.
    """
    quality = prediction.raw
    layer_name = getattr(quality, "layer", "system")
    if hasattr(layer_name, "value"):
        layer_name = layer_name.value
    shim = SimpleNamespace(quality=quality, components=quality.components)
    detector = AntiPatternDetector()
    return detector.detect(shim, layer_name)


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
        
        # SDK calls (bypassing AnalysisService to avoid SmellDetector type mismatch)
        analysis = _structural_analyze(client, "system")
        prediction = _predict(analysis)
        problems = _detect_antipatterns(prediction)
        
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

        # SDK calls (bypassing AnalysisService to avoid SmellDetector type mismatch)
        analysis = _structural_analyze(client, "system")
        prediction = _predict(analysis)
        problems = _detect_antipatterns(prediction)

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
    try:
        layer_enum = AnalysisLayer.from_string(layer)
        layer_canonical = layer_enum.value
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    try:
        logger.info(f"Analyzing layer: {layer_canonical} (input: {layer})")
        
        # SDK calls (bypassing AnalysisService to avoid SmellDetector type mismatch)
        analysis = _structural_analyze(client, layer_canonical)
        prediction = _predict(analysis)
        problems = _detect_antipatterns(prediction)
        
        return analysis_presenter.build_analysis_response(
            analysis,
            prediction,
            problems,
        )
    except Exception as e:
        logger.error(f"Layer analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
