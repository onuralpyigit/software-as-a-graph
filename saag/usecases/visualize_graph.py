from typing import List
from saag.core.ports.graph_repository import IGraphRepository
from saag.analysis.service import AnalysisService
from saag.simulation.service import SimulationService
from saag.validation.service import ValidationService
from saag.visualization.service import VisualizationService
from .models import VisOptions

class VisualizeGraphUseCase:
    """Use case for generating the visualization dashboard."""
    
    def __init__(self, service: VisualizationService):
        self.service = service

        
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
