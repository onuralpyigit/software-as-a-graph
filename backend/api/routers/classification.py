"""
Classification endpoint for component criticality analysis.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging

from api.models import Neo4jCredentials
from src.core import create_repository
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.analysis.classifier import BoxPlotClassifier

router = APIRouter(prefix="/api/v1", tags=["classification"])
logger = logging.getLogger(__name__)


@router.post("/classify", response_model=Dict[str, Any])
async def classify_components(
    credentials: Neo4jCredentials,
    metrics: Optional[List[str]] = Query(None, description="Metrics to use for classification"),
    use_fuzzy: bool = Query(False, description="Use fuzzy logic classifier"),
    use_weights: bool = Query(True, description="Consider edge weights"),
    dependency_types: Optional[List[str]] = Query(None, description="Dependency types to analyze")
):
    """
    Classify components using BoxPlot or Fuzzy logic based on centrality metrics.
    
    Available metrics: betweenness, pagerank, degree
    """
    try:
        logger.info(f"Running classification: metrics={metrics}, fuzzy={use_fuzzy}, weights={use_weights}")
        
        repo = create_repository(credentials.uri, credentials.user, credentials.password)
        try:
            # Get graph data first
            graph_data = repo.get_graph_data()
            
            # Filter by dependency types if specified
            if dependency_types:
                filtered_edges = [e for e in graph_data.edges if e.dependency_type in dependency_types]
                # Create filtered GraphData
                from src.core.models import GraphData
                graph_data = GraphData(components=graph_data.components, edges=filtered_edges)
            
            # Run structural analysis
            analyzer = StructuralAnalyzer()
            structural = analyzer.analyze(graph_data)
            
            # Prepare metrics for classification
            metrics_to_use = metrics or ['betweenness', 'pagerank', 'degree']
            metric_data = {}
            
            for metric_name in metrics_to_use:
                values = {}
                for comp_id, comp_metrics in structural.components.items():
                    if metric_name == 'betweenness':
                        values[comp_id] = comp_metrics.betweenness
                    elif metric_name == 'pagerank':
                        values[comp_id] = comp_metrics.pagerank
                    elif metric_name == 'degree':
                        values[comp_id] = comp_metrics.degree
                
                if values:
                    metric_data[metric_name] = values
            
            # Run classification for each metric
            classifications = {}
            for metric_name, values in metric_data.items():
                if use_fuzzy:
                    # Use fuzzy classifier (if implemented)
                    # For now, fall back to boxplot
                    classifier = BoxPlotClassifier()
                else:
                    classifier = BoxPlotClassifier()
                
                result = classifier.classify_scores(values, item_type="component", metric_name=metric_name)
                
                classifications[metric_name] = {
                    "statistics": {
                        "min_val": result.stats.min_val,
                        "max_val": result.stats.max_val,
                        "median": result.stats.median,
                        "q1": result.stats.q1,
                        "q3": result.stats.q3,
                        "iqr": result.stats.iqr,
                        "upper_fence": result.stats.upper_fence
                    },
                    "distribution": result.summary(),
                    "components": [
                        {
                            "id": item.id,
                            "level": item.level.value,
                            "score": item.score
                        }
                        for item in result.items
                    ]
                }
            
            # Create merged ranking combining all metrics
            component_scores = {}
            for metric_name, classification in classifications.items():
                for comp in classification["components"]:
                    if comp["id"] not in component_scores:
                        component_scores[comp["id"]] = {
                            "scores": {},
                            "levels": []
                        }
                    component_scores[comp["id"]]["scores"][metric_name] = comp["score"]
                    component_scores[comp["id"]]["levels"].append(comp["level"])
            
            # Calculate merged scores and determine dominant level
            merged_ranking = []
            level_priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            
            for comp_id, data in component_scores.items():
                merged_score = sum(data["scores"].values()) / len(data["scores"])
                
                # Dominant level is the highest priority level across all metrics
                dominant = max(data["levels"], key=lambda l: level_priority.get(l, 0))
                
                merged_ranking.append({
                    "id": comp_id,
                    "merged_score": merged_score,
                    "dominant_level": dominant,
                    "scores_by_metric": data["scores"]
                })
            
            # Sort by merged score descending
            merged_ranking.sort(key=lambda x: x["merged_score"], reverse=True)
            
            return {
                "success": True,
                "classifications": classifications,
                "merged_ranking": merged_ranking,
                "metadata": {
                    "metrics_used": metrics_to_use,
                    "use_fuzzy": use_fuzzy,
                    "use_weights": use_weights,
                    "dependency_types": dependency_types
                }
            }
        finally:
            repo.close()
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
