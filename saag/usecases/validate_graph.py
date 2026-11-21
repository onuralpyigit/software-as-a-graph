from typing import Any, Dict, List, Optional

from saag.validation.service import ValidationService
from saag.validation.models import PipelineResult


class ValidateGraphUseCase:
    """Use case for running the full validation pipeline across layers."""

    def __init__(self, service: ValidationService):
        self.service = service

    def execute(
        self,
        layers: List[str],
        precomputed_analyses: Optional[Dict[str, Any]] = None,
        precomputed_predictions: Optional[Dict[str, Any]] = None,
        precomputed_simulations: Optional[Dict[str, List[Any]]] = None,
    ) -> PipelineResult:
        return self.service.validate_layers(
            layers=layers,
            precomputed_analyses=precomputed_analyses,
            precomputed_predictions=precomputed_predictions,
            precomputed_simulations=precomputed_simulations,
        )
