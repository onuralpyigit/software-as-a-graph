from typing import List
from src.core.ports.graph_repository import IGraphRepository
from src.analysis.service import AnalysisService
from src.simulation.service import SimulationService
from src.validation.service import ValidationService
from src.validation.models import PipelineResult

class ValidateGraphUseCase:
    """Use case for running the full validation pipeline across layers."""
    
    def __init__(self, repository: IGraphRepository):
        self.repository = repository
        
        # Instantiate services needed by ValidationService
        # ValidationService orchestrates Analysis and Simulation
        analysis_service = AnalysisService(repository)
        simulation_service = SimulationService(repository)
        
        self.service = ValidationService(
            analysis_service=analysis_service,
            simulation_service=simulation_service
        )
        
    def execute(self, layers: List[str]) -> PipelineResult:
        return self.service.validate_layers(layers=layers)
