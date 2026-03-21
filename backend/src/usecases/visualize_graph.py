from typing import List
from src.core.ports.graph_repository import IGraphRepository
from src.analysis.service import AnalysisService
from src.simulation.service import SimulationService
from src.validation.service import ValidationService
from src.visualization.service import VisualizationService
from .models import VisOptions

class VisualizeGraphUseCase:
    """Use case for generating the visualization dashboard."""
    
    def __init__(self, repository: IGraphRepository):
        self.repository = repository
        
        # Instantiate services needed by VisualizationService
        analysis_service = AnalysisService(repository)
        simulation_service = SimulationService(repository)
        validation_service = ValidationService(analysis_service, simulation_service)
        
        self.service = VisualizationService(
            analysis_service=analysis_service,
            simulation_service=simulation_service,
            validation_service=validation_service,
            repository=repository
        )
        
    def execute(self, layers: List[str], output_file: str, options: VisOptions) -> str:
        return self.service.generate_dashboard(
            output_file=output_file,
            layers=layers,
            include_network=options.include_network,
            include_matrix=options.include_matrix,
            include_validation=options.include_validation,
            include_per_dim_scatter=options.include_per_dim_scatter,
            antipatterns_file=options.antipatterns_file,
            multi_seed=options.multi_seed
        )
