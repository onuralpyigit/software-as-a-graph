"""
Use case for running multi-layer system analysis.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from saag.analysis.models import LayerAnalysisResult, MultiLayerAnalysisResult
from saag.analysis.structural_analyzer import StructuralAnalyzer
from saag.prediction.quality_scoring_service import QualityScoringService
from saag.analysis.antipattern_detector import AntiPatternDetector
from saag.core.layers import AnalysisLayer, get_layer_definition

logger = logging.getLogger(__name__)

class MultiLayerAnalysisUseCase:
    """
    Use case for orchestrating multi-layer structural analysis, deterministic RMAV quality scoring,
    anti-pattern detection, and optional GNN prediction.
    """
    
    def __init__(self, repository: Any):
        self.repository = repository

    def execute(
        self,
        layers: List[str],
        use_ahp: bool = False,
        normalization_method: str = "robust",
        winsorize: bool = True,
        winsorize_limit: float = 0.05,
        gnn_model: Optional[str] = None,
        equal_weights: bool = False,
        ahp_shrinkage: float = 0.7,
        **kwargs
    ) -> MultiLayerAnalysisResult:
        # Pre-analysis stage: derive DEPENDS_ON edges
        self.repository.derive_dependencies()
        graph_data = self.repository.get_graph_data()
        
        # Initialize analyzers
        structural_analyzer = StructuralAnalyzer()
        
        pred_svc = QualityScoringService(
            use_ahp=use_ahp,
            normalization_method=normalization_method,
            winsorize=winsorize,
            winsorize_limit=winsorize_limit,
            equal_weights=equal_weights,
            ahp_shrinkage=ahp_shrinkage
        )
        smell_detector = AntiPatternDetector()
        
        results_map = {}
        for layer in layers:
            try:
                layer_enum = AnalysisLayer.from_string(layer)
            except ValueError:
                layer_enum = AnalysisLayer.SYSTEM
                
            layer_def = get_layer_definition(layer_enum)
            
            # 1. Structural Analysis
            struct_result = structural_analyzer.analyze(graph_data, layer=layer_enum)
            
            # 2. Quality Analysis (RMAV)
            pred_quality_kwargs = {k: v for k, v in kwargs.items() if k in ["run_sensitivity", "sensitivity_perturbations", "sensitivity_noise"]}
            quality_result = pred_svc.predict_quality(struct_result, **pred_quality_kwargs)
            
            # 3. Anti-Pattern Detection
            problems = smell_detector.detect(quality_result, layer=layer)
            problem_summary = pred_svc.summarize_problems(problems)
            
            layer_res = LayerAnalysisResult(
                layer=layer_enum.value,
                layer_name=layer_def.name,
                description=layer_def.description,
                structural=struct_result,
                quality=quality_result,
                problems=problems,
                problem_summary=problem_summary
            )
            
            # 4. Optional GNN Prediction
            if gnn_model:
                try:
                    from saag.prediction.gnn_service import GNNService, extract_structural_metrics_dict, extract_rmav_scores_dict
                    gnn_svc = GNNService.from_checkpoint(gnn_model, graph=layer_res.structural.graph)
                    prediction_result = gnn_svc.predict(
                        graph=layer_res.structural.graph,
                        structural_metrics=extract_structural_metrics_dict(layer_res.structural),
                        rmav_scores=extract_rmav_scores_dict(layer_res.quality)
                    )
                    layer_res.prediction = prediction_result.to_dict()
                except Exception as e:
                    logger.error(f"GNN prediction for layer {layer} failed: {e}")
                    
            results_map[layer] = layer_res
            
        # 5. Cross-layer insights
        from saag.analysis.service import AnalysisService
        svc = AnalysisService(self.repository)
        insights = svc._compute_cross_layer_insights(results_map)
        
        return MultiLayerAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layers=results_map,
            cross_layer_insights=insights
        )
