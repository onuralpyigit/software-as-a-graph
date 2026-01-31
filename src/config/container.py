"""
Dependency Injection Container

Wires ports to adapters and manages service lifecycle.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from .settings import Settings

# Import ports
from src.application.ports.outbound.graph_repository import IGraphRepository
from src.application.ports.inbound.analysis_port import IAnalysisUseCase
from src.application.ports.inbound.simulation_port import ISimulationUseCase
from src.application.ports.inbound.validation_port import IValidationUseCase

# Import adapters
from src.adapters.outbound.persistence.neo4j_repository import Neo4jGraphRepository
from src.adapters.inbound.cli.display import ConsoleDisplay

# Import application services
from src.application.services.analysis_service import AnalysisService
from src.application.services.simulation_service import SimulationService
from src.application.services.validation_service import ValidationService

from src.application.services.visualization_service import VisualizationService


@dataclass
class Container:
    """
    Dependency injection container.
    
    Wires hexagonal architecture components:
    - Ports define contracts
    - Adapters implement ports
    - Services orchestrate domain logic
    """
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    
    _repository: Optional[IGraphRepository] = field(default=None, repr=False)
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "Container":
        """Create container from settings."""
        return cls(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
    
    def graph_repository(self) -> IGraphRepository:
        """Get the graph repository singleton."""
        if not self._repository:
            self._repository = Neo4jGraphRepository(
                self.uri, self.user, self.password
            )
        return self._repository
    
    def analysis_service(self) -> IAnalysisUseCase:
        """Get analysis use case implementation."""
        return AnalysisService(repository=self.graph_repository())
    
    def simulation_service(self) -> ISimulationUseCase:
        """Get simulation use case implementation."""
        return SimulationService(repository=self.graph_repository())
    
    def validation_service(self, targets: Optional[List[str]] = None) -> IValidationUseCase:
        """Get validation use case implementation."""
        return ValidationService(
            analysis_service=self.analysis_service(),
            simulation_service=self.simulation_service(),
            targets=targets
        )
    
    def display_service(self) -> ConsoleDisplay:
        """Get console display adapter."""
        return ConsoleDisplay()
    
    def visualization_service(self) -> VisualizationService:
        """Get visualization service."""
        return VisualizationService(
            analysis_service=self.analysis_service(),
            simulation_service=self.simulation_service(),
            validation_service=self.validation_service()
        )
    
    def close(self) -> None:
        """Close all resources."""
        if self._repository:
            self._repository.close()
            self._repository = None
