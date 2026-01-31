"""
Simulation Service

Application service implementing ISimulationUseCase.
Orchestrates domain logic for simulations.
"""

from typing import List, Optional, Any

from src.application.ports.inbound.simulation_port import ISimulationUseCase
from src.application.ports.outbound.graph_repository import IGraphRepository

# Import legacy service for compatibility during migration
from src.services.simulation_service import SimulationService as LegacySimulationService


class SimulationService(ISimulationUseCase):
    """
    Application service for simulations.
    
    Implements ISimulationUseCase port and delegates to domain services.
    Uses dependency injection for repository access.
    """
    
    def __init__(self, repository: IGraphRepository):
        """
        Initialize simulation service.
        
        Args:
            repository: Graph repository for data access
        """
        self._repository = repository
        # Use legacy service internally during migration
        self._legacy = LegacySimulationService(repository=repository)
    
    def run_event_simulation(
        self, 
        source_app: str, 
        num_messages: int = 100,
        duration: float = 10.0,
        **kwargs
    ) -> Any:
        """Run event simulation from a specific source application."""
        return self._legacy.run_event_simulation(
            source_app, num_messages, duration, **kwargs
        )
    
    def run_failure_simulation(
        self, 
        target_id: str, 
        layer: str = "system",
        cascade_probability: float = 1.0,
        **kwargs
    ) -> Any:
        """Run failure simulation for a specific component."""
        return self._legacy.run_failure_simulation(
            target_id, layer, cascade_probability, **kwargs
        )
    
    def run_failure_simulation_exhaustive(
        self,
        layer: str = "system",
        cascade_probability: float = 1.0
    ) -> List[Any]:
        """Run failure simulation for all components in a layer."""
        return self._legacy.run_failure_simulation_exhaustive(
            layer, cascade_probability
        )
    
    def generate_report(self, layers: Optional[List[str]] = None) -> Any:
        """Generate comprehensive simulation report."""
        return self._legacy.generate_report(layers)
    
    def classify_components(
        self, 
        layer: str = "system",
        k_factor: float = 1.5
    ) -> List[Any]:
        """Classify components by criticality based on simulation results."""
        return self._legacy.classify_components(layer, k_factor)
    
    def export_report(self, report: Any, output_file: str) -> None:
        """Export simulation report to JSON file."""
        return self._legacy.export_report(report, output_file)
