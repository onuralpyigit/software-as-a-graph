"""
saag/usecases/predictive.py

Pathway B (Predictive / Heterogeneous Graph Learning) Use Case.
Executes non-linear relational message passing (Heterogeneous Graph Transformer)
with 16-D QoS edge encodings to forecast quantitative failure blast radii (I*(v))
and rank critical components.
"""
from typing import Any, List, Optional, Tuple, Union

from saag.analysis.models import (
    QualityAnalysisResult,
    StructuralAnalysisResult,
)


class PredictiveUseCase:
    """
    Use Case for Pathway B: Inductive Criticality & Blast Radius Forecasting (HGL).
    
    INDEPENDENCE GUARANTEE:
    Pathway B consumes structural graph topology and edge QoS attributes.
    Simulation-derived labels are used strictly during offline model training,
    never as an online inference dependency.
    """

    def __init__(
        self,
        prediction_service: Optional[Any] = None,
        gnn_checkpoint_dir: Optional[str] = None,
        prefer_gnn: bool = True,
    ):
        if prediction_service is not None:
            self.service = prediction_service
        else:
            from saag.prediction.service import PredictionService
            self.service = PredictionService(
                gnn_checkpoint_dir=gnn_checkpoint_dir,
                prefer_gnn=prefer_gnn,
            )

    def execute(
        self,
        layer: str = "system",
        structural_result: Optional[StructuralAnalysisResult] = None,
        graph: Optional[Any] = None,
        k: int = 10,
        active_patterns: Optional[List[str]] = None,
        run_sensitivity: bool = False,
        **kwargs,
    ) -> Any:
        """
        Execute Pathway B predictive inference to obtain quantitative blast radius predictions.
        
        When a GNN checkpoint is present and prefer_gnn=True, executes HGT inference.
        Otherwise falls back cleanly to deterministic RM scores (Zero-GNN cold-start mode).
        
        Returns:
            GNNAnalysisResult or QualityAnalysisResult with ranked components.
        """
        if structural_result is None:
            raise ValueError("structural_result must be provided to execute PredictiveUseCase.")

        # Execute prediction through PredictionService
        result = self.service.predict_quality_with_gnn(
            structural_result=structural_result,
            graph=graph,
            layer=layer,
            active_patterns=active_patterns,
            run_sensitivity=run_sensitivity,
        )
        return result

    def get_top_k_critical(
        self,
        prediction_result: Any,
        k: int = 10,
        node_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Extract the Top-K shortlisted critical components based on predicted blast radius.
        """
        from saag.analysis.triage import select_top_k
        return select_top_k(prediction_result, k=k, node_types=node_types)


# Backward-compatible alias
PredictiveGraphUseCase = PredictiveUseCase
