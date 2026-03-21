"""
Simulation endpoints for event and failure analysis.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging

from api.models import (
    EventSimulationRequest,
    FailureSimulationRequest,
    ExhaustiveSimulationRequest,
    ReportRequest,
    EventSimulationResponse,
    FailureSimulationResponse,
    ExhaustiveSimulationResponse,
    SimulationReportResponse
)
from api.dependencies import (
    get_simulation_service, get_repository, get_simulate_graph_use_case
)
from src.core import IGraphRepository
from src.simulation import SimulationService
from src.usecases import SimulateGraphUseCase, SimulationMode
from api.presenters import simulation_presenter

router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])
logger = logging.getLogger(__name__)


@router.post("/event", response_model=EventSimulationResponse)
async def simulate_event(
    request: EventSimulationRequest, 
    service: SimulationService = Depends(get_simulation_service)
):
    """
    Run event simulation from a source application.
    
    Simulates message flow through the pub-sub system, measuring:
    - Throughput (messages published, delivered, dropped)
    - Latency metrics (avg, min, max, p50, p99)
    - Path analysis (topics, brokers, subscribers)
    - Component impacts
    """
    try:
        logger.info(f"Running event simulation: source={request.source_app}, messages={request.num_messages}")
        
        result = service.run_event_simulation(
            source_app=request.source_app,
            num_messages=request.num_messages,
            duration=request.duration
        )
        
        return simulation_presenter.format_event_simulation_response(result)
    except Exception as e:
        logger.error(f"Event simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Event simulation failed: {str(e)}")


@router.post("/failure", response_model=FailureSimulationResponse)
async def simulate_failure(
    request: FailureSimulationRequest,
    use_case: SimulateGraphUseCase = Depends(get_simulate_graph_use_case)
):
    """
    Run failure simulation for a target component.
    """
    valid_layers = ["app", "infra", "mw-app", "mw-infra", "system"]
    if request.layer not in valid_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer. Must be one of: {', '.join(valid_layers)}"
        )
    
    try:
        logger.info(f"Running failure simulation: target={request.target_id}, layer={request.layer}")
        
        result = use_case.execute(
            target_id=request.target_id,
            layer=request.layer,
            mode=SimulationMode.SINGLE
        )
        
        return simulation_presenter.format_failure_simulation_response(result)
    except Exception as e:
        logger.error(f"Failure simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failure simulation failed: {str(e)}")


@router.post("/exhaustive", response_model=ExhaustiveSimulationResponse)
async def simulate_exhaustive(
    request: ExhaustiveSimulationRequest,
    use_case: SimulateGraphUseCase = Depends(get_simulate_graph_use_case)
):
    """
    Run exhaustive failure analysis for all components in a layer.
    """
    valid_layers = ["app", "infra", "mw-app", "mw-infra", "system"]
    if request.layer not in valid_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer. Must be one of: {', '.join(valid_layers)}"
        )
    
    try:
        logger.info(f"Running exhaustive failure analysis: layer={request.layer}")
        
        results = use_case.execute(layer=request.layer, mode=SimulationMode.EXHAUSTIVE)
        
        return simulation_presenter.format_exhaustive_simulation_response(results, request.layer)
    except Exception as e:
        logger.error(f"Exhaustive simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Exhaustive simulation failed: {str(e)}")


@router.post("/report", response_model=SimulationReportResponse)
async def generate_simulation_report(
    request: ReportRequest,
    service: SimulationService = Depends(get_simulation_service)
):
    """
    Generate comprehensive simulation report with analysis across multiple layers.
    
    Includes:
    - Graph summary statistics
    - Per-layer event and failure simulation metrics
    - Criticality classification (critical, high, medium, low, minimal)
    - SPOF (Single Point of Failure) detection
    - Top critical components
    - System health recommendations
    
    Valid layers: app, infra, mw, system (or legacy: application, infrastructure, complete)
    """
    # Map legacy layer names to canonical names
    layer_aliases = {
        "application": "app",
        "infrastructure": "infra",
        "app_broker": "mw",
        "complete": "system",
    }
    
    valid_layers = ["app", "infra", "mw", "system"]
    mapped_layers = []
    
    for layer in request.layers:
        # Map legacy name to canonical name
        canonical_layer = layer_aliases.get(layer, layer)
        
        if canonical_layer not in valid_layers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid layer '{layer}'. Must be one of: {', '.join(valid_layers + list(layer_aliases.keys()))}"
            )
        mapped_layers.append(canonical_layer)
    
    try:
        logger.info(f"Generating simulation report: layers={mapped_layers}")
        
        report = service.generate_report(layers=mapped_layers)
        
        return simulation_presenter.format_simulation_report_response(report)
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
