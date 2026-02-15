"""
Analysis endpoints for system, type, and layer analysis.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from api.models import Neo4jCredentials
from src.core import create_repository
from src.analysis import AnalysisService

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)


@router.post("/full", response_model=Dict[str, Any])
async def analyze_full_system(credentials: Neo4jCredentials):
    """
    Run complete system analysis including:
    - Structural metrics (centrality, clustering, etc.)
    - Quality scores (reliability, maintainability, availability)
    - Problem detection
    """
    repo = create_repository(uri=credentials.uri, user=credentials.user, password=credentials.password)
    try:
        logger.info("Running full system analysis")
        service = AnalysisService(repo)
        result = service.analyze_layer("system")
        
        # Create a map of component IDs to names from structural data
        component_names = {c.id: c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id 
                          for c in result.quality.components}
        
        return {
            "success": True,
            "layer": result.layer,
            "analysis": {
                "context": result.layer_name,
                "description": result.description,
                "summary": {
                    "total_components": result.quality.classification_summary.total_components,
                    "critical_count": result.quality.classification_summary.component_distribution.get("critical", 0),
                    "high_count": result.quality.classification_summary.component_distribution.get("high", 0),
                    "total_problems": result.problem_summary.total_problems,
                    "critical_problems": result.problem_summary.by_severity.get("CRITICAL", 0),
                    "components": dict(result.quality.classification_summary.component_distribution),
                    "edges": dict(result.quality.classification_summary.edge_distribution)
                },
                "stats": {
                    "nodes": result.structural.graph_summary.nodes,
                    "edges": result.structural.graph_summary.edges,
                    "density": result.structural.graph_summary.density,
                    "avg_degree": result.structural.graph_summary.avg_degree
                },
                "components": [
                    {
                        "id": c.id,
                        "name": c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id,
                        "type": c.type,
                        "criticality_level": c.levels.overall.value,
                        "criticality_levels": {
                            "reliability": c.levels.reliability.value,
                            "maintainability": c.levels.maintainability.value,
                            "availability": c.levels.availability.value,
                            "vulnerability": c.levels.vulnerability.value,
                            "overall": c.levels.overall.value
                        },
                        "scores": {
                            "reliability": c.scores.reliability,
                            "maintainability": c.scores.maintainability,
                            "availability": c.scores.availability,
                            "vulnerability": c.scores.vulnerability,
                            "overall": c.scores.overall
                        }
                    }
                    for c in result.quality.components
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "source_name": component_names.get(e.source, e.source),
                        "target_name": component_names.get(e.target, e.target),
                        "type": e.dependency_type,
                        "criticality_level": e.level.value,
                        "scores": {
                            "reliability": e.scores.reliability,
                            "maintainability": e.scores.maintainability,
                            "availability": e.scores.availability,
                            "vulnerability": e.scores.vulnerability,
                            "overall": e.scores.overall
                        }
                    }
                    for e in result.quality.edges
                ],
                "problems": [
                    {
                        "entity_id": p.entity_id,
                        "type": p.entity_type,
                        "category": p.category.value if hasattr(p.category, 'value') else str(p.category),
                        "severity": p.severity.value if hasattr(p.severity, 'value') else str(p.severity),
                        "name": p.name,
                        "description": p.description,
                        "recommendation": p.recommendation
                    }
                    for p in result.problems
                ]
            }
        }
    except Exception as e:
        logger.error(f"Full analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        repo.close()


@router.post("/type/{component_type}", response_model=Dict[str, Any])
async def analyze_by_type(component_type: str, credentials: Neo4jCredentials):
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
        
    repo = create_repository(uri=credentials.uri, user=credentials.user, password=credentials.password)
    try:
        logger.info(f"Analyzing component type: {component_type} (normalized to {normalized_type})")
        service = AnalysisService(repo)
        result = service.analyze_layer("system")
        
        # Filter components by type
        filtered_components = [c for c in result.quality.components if c.type == normalized_type]
        
        # Create a map of component IDs to names from structural data
        component_names = {c.id: c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id 
                          for c in filtered_components}
        
        # Filter edges to only include those between filtered components
        filtered_component_ids = {c.id for c in filtered_components}
        filtered_edges = [e for e in result.quality.edges 
                         if e.source in filtered_component_ids or e.target in filtered_component_ids]
        
        return {
            "success": True,
            "layer": result.layer,
            "component_type": normalized_type,
            "analysis": {
                "context": f"{normalized_type} Components Analysis",
                "description": f"Analysis filtered by component type: {normalized_type}",
                "summary": {
                    "total_components": len(filtered_components),
                    "critical_count": sum(1 for c in filtered_components if c.levels.overall.value == "critical"),
                    "high_count": sum(1 for c in filtered_components if c.levels.overall.value == "high"),
                    "total_problems": sum(1 for p in result.problems if p.entity_id in filtered_component_ids),
                    "critical_problems": sum(1 for p in result.problems 
                                           if p.entity_id in filtered_component_ids and 
                                           (p.severity == "CRITICAL" or (hasattr(p.severity, 'value') and p.severity.value == "CRITICAL"))),
                    "components": {
                        level: sum(1 for c in filtered_components if c.levels.overall.value == level)
                        for level in ["critical", "high", "medium", "low", "minimal"]
                    },
                    "edges": {
                        level: sum(1 for e in filtered_edges if e.level.value == level)
                        for level in ["critical", "high", "medium", "low", "minimal"]
                    }
                },
                "stats": {
                    "nodes": len(filtered_components),
                    "edges": len(filtered_edges),
                    "density": result.structural.graph_summary.density,
                    "avg_degree": result.structural.graph_summary.avg_degree
                },
                "components": [
                    {
                        "id": c.id,
                        "name": c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id,
                        "type": c.type,
                        "criticality_level": c.levels.overall.value,
                        "criticality_levels": {
                            "reliability": c.levels.reliability.value,
                            "maintainability": c.levels.maintainability.value,
                            "availability": c.levels.availability.value,
                            "vulnerability": c.levels.vulnerability.value,
                            "overall": c.levels.overall.value
                        },
                        "scores": {
                            "reliability": c.scores.reliability,
                            "maintainability": c.scores.maintainability,
                            "availability": c.scores.availability,
                            "vulnerability": c.scores.vulnerability,
                            "overall": c.scores.overall
                        }
                    }
                    for c in filtered_components
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "source_name": component_names.get(e.source, e.source),
                        "target_name": component_names.get(e.target, e.target),
                        "type": e.dependency_type,
                        "criticality_level": e.level.value,
                        "scores": {
                            "reliability": e.scores.reliability,
                            "maintainability": e.scores.maintainability,
                            "availability": e.scores.availability,
                            "vulnerability": e.scores.vulnerability,
                            "overall": e.scores.overall
                        }
                    }
                    for e in filtered_edges
                ],
                "problems": [
                    {
                        "entity_id": p.entity_id,
                        "type": p.entity_type,
                        "category": p.category.value if hasattr(p.category, 'value') else str(p.category),
                        "severity": p.severity.value if hasattr(p.severity, 'value') else str(p.severity),
                        "name": p.name,
                        "description": p.description,
                        "recommendation": p.recommendation
                    }
                    for p in result.problems if p.entity_id in filtered_component_ids
                ]
            }
        }
    except Exception as e:
        logger.error(f"Type analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        repo.close()


@router.post("/layer/{layer}", response_model=Dict[str, Any])
async def analyze_layer(layer: str, credentials: Neo4jCredentials):
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
    
    repo = create_repository(uri=credentials.uri, user=credentials.user, password=credentials.password)
    try:
        logger.info(f"Analyzing layer: {layer}")
        service = AnalysisService(repo)
        result = service.analyze_layer(layer)
        
        # Create a map of component IDs to names from structural data
        component_names = {c.id: c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id 
                          for c in result.quality.components}
        
        return {
            "success": True,
            "layer": result.layer,
            "analysis": {
                "context": result.layer_name,
                "description": result.description,
                "summary": {
                    "total_components": result.quality.classification_summary.total_components,
                    "critical_count": result.quality.classification_summary.component_distribution.get("critical", 0),
                    "high_count": result.quality.classification_summary.component_distribution.get("high", 0),
                    "total_problems": result.problem_summary.total_problems,
                    "critical_problems": result.problem_summary.by_severity.get("CRITICAL", 0),
                    "components": dict(result.quality.classification_summary.component_distribution),
                    "edges": dict(result.quality.classification_summary.edge_distribution)
                },
                "stats": {
                    "nodes": result.structural.graph_summary.nodes,
                    "edges": result.structural.graph_summary.edges,
                    "density": result.structural.graph_summary.density,
                    "avg_degree": result.structural.graph_summary.avg_degree
                },
                "components": [
                    {
                        "id": c.id,
                        "name": c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id,
                        "type": c.type,
                        "criticality_level": c.levels.overall.value,
                        "criticality_levels": {
                            "reliability": c.levels.reliability.value,
                            "maintainability": c.levels.maintainability.value,
                            "availability": c.levels.availability.value,
                            "vulnerability": c.levels.vulnerability.value,
                            "overall": c.levels.overall.value
                        },
                        "scores": {
                            "reliability": c.scores.reliability,
                            "maintainability": c.scores.maintainability,
                            "availability": c.scores.availability,
                            "vulnerability": c.scores.vulnerability,
                            "overall": c.scores.overall
                        }
                    }
                    for c in result.quality.components
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "source_name": component_names.get(e.source, e.source),
                        "target_name": component_names.get(e.target, e.target),
                        "type": e.dependency_type,
                        "criticality_level": e.level.value,
                        "scores": {
                            "reliability": e.scores.reliability,
                            "maintainability": e.scores.maintainability,
                            "availability": e.scores.availability,
                            "vulnerability": e.scores.vulnerability,
                            "overall": e.scores.overall
                        }
                    }
                    for e in result.quality.edges
                ],
                "problems": [
                    {
                        "entity_id": p.entity_id,
                        "type": p.entity_type,
                        "category": p.category.value if hasattr(p.category, 'value') else str(p.category),
                        "severity": p.severity.value if hasattr(p.severity, 'value') else str(p.severity),
                        "name": p.name,
                        "description": p.description,
                        "recommendation": p.recommendation
                    }
                    for p in result.problems
                ]
            }
        }
    except Exception as e:
        logger.error(f"Layer analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        repo.close()
