from dataclasses import dataclass
from typing import Optional

from src.repositories.graph_repository import GraphRepository
from src.repositories.memory_repository import InMemoryGraphRepository

@dataclass
class Container:
    """Dependency injection container."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    
    _repository: Optional[GraphRepository] = None
    
    def graph_repository(self) -> GraphRepository:
        """Get the graph repository singleton."""
        if not self._repository:
            # Check availability or just instantiate
            self._repository = GraphRepository(self.uri, self.user, self.password)
        return self._repository
    
    def analysis_service(self) -> 'AnalysisService':
        from src.services.analysis_service import AnalysisService
        return AnalysisService(repository=self.graph_repository())

    def simulation_service(self) -> 'SimulationService':
        from src.services.simulation_service import SimulationService
        return SimulationService(repository=self.graph_repository())

    def validation_service(self, targets=None) -> 'ValidationService':
        from src.services.validation_service import ValidationService
        return ValidationService(
            analysis_service=self.analysis_service(),
            simulation_service=self.simulation_service(),
            targets=targets
        )

    def display_service(self) -> 'ConsoleDisplay':
        from src.cli.display import ConsoleDisplay
        return ConsoleDisplay()

    def visualization_service(self) -> 'VisualizationService':
        from src.services.visualization_service import VisualizationService
        return VisualizationService(
            analysis_service=self.analysis_service(),
            simulation_service=self.simulation_service(),
            validation_service=self.validation_service()
        )

    def close(self) -> None:
        if self._repository:
            self._repository.close()
            self._repository = None
