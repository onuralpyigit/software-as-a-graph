"""
Simulation endpoints for event and failure analysis.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from api.models import (
    EventSimulationRequest,
    FailureSimulationRequest,
    ExhaustiveSimulationRequest,
    ReportRequest
)
from src.core import create_repository
from src.simulation import SimulationService

router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])
logger = logging.getLogger(__name__)


@router.post("/event", response_model=Dict[str, Any])
async def simulate_event(request: EventSimulationRequest):
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
        
        creds = request.credentials
        repo = create_repository(uri=creds.uri, user=creds.user, password=creds.password)
        
        try:
            service = SimulationService(repo)
            
            result = service.run_event_simulation(
                source_app=request.source_app,
                num_messages=request.num_messages,
                duration=request.duration
            )
            
            return {
                "success": True,
                "simulation_type": "event",
                "result": result.to_dict()
            }
        finally:
            repo.close()
    except Exception as e:
        logger.error(f"Event simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Event simulation failed: {str(e)}")


@router.post("/failure", response_model=Dict[str, Any])
async def simulate_failure(request: FailureSimulationRequest):
    """
    Run failure simulation for a target component.
    
    Simulates component failure and analyzes:
    - Composite impact score
    - Reachability loss (connectivity degradation)
    - Fragmentation (component isolation)
    - Throughput loss (capacity degradation)
    - Cascade propagation (dependent failures)
    - Per-layer impacts
    
    Valid layers: app, infra, mw-app, mw-infra, system
    """
    valid_layers = ["app", "infra", "mw-app", "mw-infra", "system"]
    if request.layer not in valid_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer. Must be one of: {', '.join(valid_layers)}"
        )
    
    try:
        logger.info(f"Running failure simulation: target={request.target_id}, layer={request.layer}")
        
        creds = request.credentials
        repo = create_repository(uri=creds.uri, user=creds.user, password=creds.password)
        
        try:
            service = SimulationService(repo)
            
            result = service.run_failure_simulation(
                target_id=request.target_id,
                layer=request.layer,
                cascade_probability=request.cascade_probability
            )
            
            return {
                "success": True,
                "simulation_type": "failure",
                "result": result.to_dict()
            }
        finally:
            repo.close()
    except Exception as e:
        logger.error(f"Failure simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failure simulation failed: {str(e)}")


@router.post("/exhaustive", response_model=Dict[str, Any])
async def simulate_exhaustive(request: ExhaustiveSimulationRequest):
    """
    Run exhaustive failure analysis for all components in a layer.
    
    Analyzes failure impact for every component, sorted by impact score.
    Useful for identifying the most critical components in the system.
    
    Valid layers: app, infra, mw-app, mw-infra, system
    
    Warning: This can take significant time for large graphs.
    """
    valid_layers = ["app", "infra", "mw-app", "mw-infra", "system"]
    if request.layer not in valid_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer. Must be one of: {', '.join(valid_layers)}"
        )
    
    try:
        logger.info(f"Running exhaustive failure analysis: layer={request.layer}")
        
        creds = request.credentials
        repo = create_repository(uri=creds.uri, user=creds.user, password=creds.password)
        
        try:
            service = SimulationService(repo)
            
            results = service.run_failure_simulation_exhaustive(
                layer=request.layer,
                cascade_probability=request.cascade_probability
            )
            
            # Create summary from results
            summary = {
                "total_components": len(results),
                "avg_impact": sum(r.impact.composite_impact for r in results) / len(results) if results else 0,
                "max_impact": max((r.impact.composite_impact for r in results), default=0),
                "critical_count": sum(1 for r in results if r.impact.composite_impact > 0.7),
                "high_count": sum(1 for r in results if 0.4 < r.impact.composite_impact <= 0.7),
                "medium_count": sum(1 for r in results if 0.2 < r.impact.composite_impact <= 0.4),
                "spof_count": sum(1 for r in results if r.impact.fragmentation > 0.01),
            }
            
            return {
                "success": True,
                "simulation_type": "exhaustive",
                "layer": request.layer,
                "summary": summary,
                "results": [r.to_dict() for r in results]
            }
        finally:
            repo.close()
    except Exception as e:
        logger.error(f"Exhaustive simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Exhaustive simulation failed: {str(e)}")


@router.post("/report", response_model=Dict[str, Any])
async def generate_simulation_report(request: ReportRequest):
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
        
        creds = request.credentials
        repo = create_repository(uri=creds.uri, user=creds.user, password=creds.password)
        
        try:
            service = SimulationService(repo)
            
            report = service.generate_report(layers=mapped_layers)
            
            # Transform top_critical to match frontend expectations (nested structure)
            report_dict = report.to_dict()
            if "top_critical" in report_dict:
                report_dict["top_critical"] = [
                    {
                        "id": comp["id"],
                        "type": comp["type"],
                        "level": comp["level"],
                        "scores": {
                            "event_impact": 0.0,
                            "failure_impact": 0.0,
                            "combined_impact": comp.get("combined_impact", 0.0),
                        },
                        "metrics": {
                            "cascade_count": comp.get("cascade_count", 0),
                            "message_throughput": 0,
                            "reachability_loss_percent": 0.0,
                        },
                    }
                    for comp in report_dict["top_critical"]
                ]
            
            return {
                "success": True,
                "report": report_dict
            }
        finally:
            repo.close()
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
