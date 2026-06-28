"""
saag/usecases/prescribe_graph.py
"""
from typing import Any, Optional
from saag.prescription.service import PrescribeService
from saag.prescription.models import PrescribeResult

class PrescribeGraphUseCase:
    """
    Use case for generating prescriptive optimizations and evaluating them in closed-loop.
    """
    
    def __init__(self, service: PrescribeService):
        self.service = service
        
    def execute(
        self, 
        analysis_result: Any, 
        prediction_result: Optional[Any] = None, 
        layer: str = "system",
        gnn_checkpoint: Optional[str] = None
    ) -> PrescribeResult:
        return self.service.prescribe(
            analysis_result=analysis_result,
            prediction_result=prediction_result,
            layer=layer,
            gnn_checkpoint=gnn_checkpoint
        )
