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
        
    def execute(self, layer: str = "system", mode: SimulationMode = SimulationMode.EXHAUSTIVE, target_id: str = None, **kwargs) -> Any:
        if mode == SimulationMode.EXHAUSTIVE:
            return self.service.run_failure_simulation_exhaustive(layer=layer, **kwargs)
        elif mode == SimulationMode.SINGLE:
            if not target_id:
                raise ValueError("target_id is required for SINGLE simulation mode")
            return self.service.run_failure_simulation(target_id=target_id, layer=layer, **kwargs)
        elif mode == SimulationMode.EVENT:
            source_app = kwargs.pop("source_app", "all")
            if source_app == "all":
                return self.service.run_event_simulation_all(layer=layer, **kwargs)
            return self.service.run_event_simulation(source_app=source_app, layer=layer, **kwargs)
        elif mode == SimulationMode.MONTE_CARLO:
            if not target_id:
                raise ValueError("target_id is required for MONTE_CARLO simulation mode")
            return self.service.run_failure_simulation_monte_carlo(target_id=target_id, layer=layer, **kwargs)
        elif mode == SimulationMode.PAIRWISE:
            return self.service.run_failure_simulation_pairwise(layer=layer, **kwargs)
        elif mode == SimulationMode.REPORT:
            # Avoid multiple values for 'layers' by popping it from kwargs
            layers = kwargs.pop("layers", [layer])
            return self.service.generate_report(layers=layers, **kwargs)
        elif mode == SimulationMode.CLASSIFY:
            edges = kwargs.get("edges", False)
            if edges:
                return self.service.classify_edges(layer=layer, **kwargs)
            return self.service.classify_components(layer=layer, **kwargs)
            
        # Default to exhaustive
        return self.service.run_failure_simulation_exhaustive(layer=layer, **kwargs)
