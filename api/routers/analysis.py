"""
Analysis endpoints for system, type, and layer analysis.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging
import time

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

# Loggers whose records should be captured for the analysis log output.
_CAPTURE_LOGGERS = [
    "api.routers.analysis",
    "saag.analysis.structural_analyzer",
    "saag.analysis.service",
    "saag.analysis.antipattern_detector",
    "saag.prediction",
]


class _ListHandler(logging.Handler):
    """Logging handler that appends formatted records to a list."""

    def __init__(self, records: List[str]) -> None:
        super().__init__(level=logging.DEBUG)
        self._records = records
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._records.append(self.format(record))
        except Exception:
            pass


class _AnalysisLogCapture:
    """Context-manager that attaches a list handler to the analysis loggers."""

    def __init__(self) -> None:
        self.records: List[str] = []
        self._handler = _ListHandler(self.records)
        self._loggers: List[logging.Logger] = []

    def __enter__(self) -> "_AnalysisLogCapture":
        for name in _CAPTURE_LOGGERS:
            lg = logging.getLogger(name)
            lg.addHandler(self._handler)
            self._loggers.append(lg)
        return self

    def __exit__(self, *args: object) -> None:
        for lg in self._loggers:
            lg.removeHandler(self._handler)


def _structural_analyze(client: Client, layer: str) -> SaagAnalysisResult:
    """
    Run structural analysis directly via StructuralAnalyzer, bypassing AnalysisService
    which incorrectly passes StructuralAnalysisResult to AntiPatternDetector.detect().
    """
    # Pre-analysis stage: derive DEPENDS_ON edges before reading graph data.
    logger.info("Step 1/4 — Deriving dependency edges from graph relationships…")
    t0 = time.perf_counter()
    client.repo.derive_dependencies()
    graph_data = client.repo.get_graph_data()
    logger.info(
        "Step 1/4 — Dependency derivation complete (%.2fs)",
        time.perf_counter() - t0,
    )

    logger.info("Step 2/4 — Running structural analysis for layer '%s'…", layer)
    t1 = time.perf_counter()
    analyzer = StructuralAnalyzer()
    layer_enum = AnalysisLayer.from_string(layer)
    raw = analyzer.analyze(graph_data, layer=layer_enum)
    logger.info(
        "Step 2/4 — Structural analysis complete: %d nodes, %d edges (%.2fs)",
        raw.graph_summary.nodes,
        raw.graph_summary.edges,
        time.perf_counter() - t1,
    )
    return SaagAnalysisResult(raw)


def _predict(analysis: SaagAnalysisResult) -> SaagPredictionResult:
    """
    Run quality prediction using an already-computed structural analysis result,
    bypassing client.predict() which incorrectly expects a layer string.
    """
    logger.info("Step 3/4 — Scoring RMAV quality dimensions (reliability, maintainability, availability, vulnerability)…")
    t0 = time.perf_counter()
    prediction_service = PredictionService()
    predict_uc = _PredictGraphUseCase(prediction_service)
    quality, _ = predict_uc.execute(
        layer="system",
        structural_result=analysis.raw,
        detect_problems=False,
    )
    logger.info(
        "Step 3/4 — Quality scoring complete (%.2fs)",
        time.perf_counter() - t0,
    )
    return SaagPredictionResult(quality)


def _detect_antipatterns(prediction) -> list:
    """
    Detect antipatterns by calling AntiPatternDetector directly with a shim that
    includes the required 'components' attribute, bypassing the broken SimpleNamespace
    shim in ProblemDetector which omits 'components'.
    """
    logger.info("Step 4/4 — Detecting architectural anti-patterns…")
    t0 = time.perf_counter()
    quality = prediction.raw
    layer_name = getattr(quality, "layer", "system")
    if hasattr(layer_name, "value"):
        layer_name = layer_name.value
    shim = SimpleNamespace(quality=quality, components=quality.components, edges=quality.edges)
    detector = AntiPatternDetector()
    problems = detector.detect(shim, layer_name)
    logger.info(
        "Step 4/4 — Anti-pattern detection complete: %d problem(s) found (%.2fs)",
        len(problems),
        time.perf_counter() - t0,
    )
    return problems


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
        
        with _AnalysisLogCapture() as cap:
            cap.records.append("INFO api.routers.analysis: Starting full system analysis…")
            # SDK calls (bypassing AnalysisService to avoid SmellDetector type mismatch)
            analysis = _structural_analyze(client, "system")
            prediction = _predict(analysis)
            problems = _detect_antipatterns(prediction)
            cap.records.append("INFO api.routers.analysis: Analysis complete.")
        
        return analysis_presenter.build_analysis_response(
            analysis,
            prediction,
            problems,
            logs=cap.records,
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

        with _AnalysisLogCapture() as cap:
            cap.records.append(f"INFO api.routers.analysis: Starting analysis for component type '{normalized_type}'…")
            # SDK calls (bypassing AnalysisService to avoid SmellDetector type mismatch)
            analysis = _structural_analyze(client, "system")
            prediction = _predict(analysis)
            problems = _detect_antipatterns(prediction)
            cap.records.append("INFO api.routers.analysis: Analysis complete.")

        return analysis_presenter.build_analysis_response(
            analysis,
            prediction,
            problems,
            context=f"{normalized_type} Components Analysis",
            description=f"Analysis filtered by component type: {normalized_type}",
            component_type=normalized_type,
            logs=cap.records,
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

        with _AnalysisLogCapture() as cap:
            cap.records.append(f"INFO api.routers.analysis: Starting analysis for layer '{layer_canonical}'…")
            # SDK calls (bypassing AnalysisService to avoid SmellDetector type mismatch)
            analysis = _structural_analyze(client, layer_canonical)
            prediction = _predict(analysis)
            problems = _detect_antipatterns(prediction)
            cap.records.append("INFO api.routers.analysis: Analysis complete.")

        return analysis_presenter.build_analysis_response(
            analysis,
            prediction,
            problems,
            logs=cap.records,
        )
    except Exception as e:
        logger.error(f"Layer analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
