from typing import List
from saag.core.ports.graph_repository import IGraphRepository
from saag.analysis.service import AnalysisService
from saag.simulation.service import SimulationService
from saag.validation.service import ValidationService
from saag.validation.models import PipelineResult

class ValidateGraphUseCase:
    """Use case for running the full validation pipeline across layers."""
    
    def __init__(self, service: ValidationService):
        self.service = service

        
    def execute(self, layers: List[str]) -> PipelineResult:
        return self.service.validate_layers(layers=layers)
