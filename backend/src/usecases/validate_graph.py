from typing import List
from src.core.ports.graph_repository import IGraphRepository
from src.analysis.service import AnalysisService
from src.simulation.service import SimulationService
from src.validation.service import ValidationService
from src.validation.models import PipelineResult

class ValidateGraphUseCase:
    """Use case for running the full validation pipeline across layers."""
    
    def __init__(self, service: ValidationService):
        self.service = service

        
    def execute(self, layers: List[str]) -> PipelineResult:
        return self.service.validate_layers(layers=layers)
