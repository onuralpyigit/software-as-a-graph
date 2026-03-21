from typing import List, Any
from src.core.ports.graph_repository import IGraphRepository
from src.simulation.service import SimulationService
from src.simulation.models import FailureResult
from .models import SimulationMode

class SimulateGraphUseCase:
    """Use case for running graph simulations (primarily failure analysis)."""
    
    def __init__(self, repository: IGraphRepository):
        self.repository = repository
        self.service = SimulationService(repository)
        
    def execute(self, layer: str, mode: SimulationMode = SimulationMode.EXHAUSTIVE, target_id: str = None) -> Any:
        if mode == SimulationMode.EXHAUSTIVE:
            return self.service.run_failure_simulation_exhaustive(layer=layer)
        elif mode == SimulationMode.SINGLE:
            if not target_id:
                raise ValueError("target_id is required for SINGLE simulation mode")
            return self.service.run_failure_simulation(target_id=target_id, layer=layer)
            
        # Default to exhaustive
        return self.service.run_failure_simulation_exhaustive(layer=layer)
