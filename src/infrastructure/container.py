from dataclasses import dataclass
from typing import Optional

from ..application.ports import GraphRepository
from ..adapters.persistence import Neo4jGraphRepository, InMemoryGraphRepository
from ..application.use_cases import ImportGraphUseCase, AnalyzeGraphUseCase

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
            self._repository = Neo4jGraphRepository(self.uri, self.user, self.password)
        return self._repository
    
    def import_use_case(self) -> ImportGraphUseCase:
        return ImportGraphUseCase(repository=self.graph_repository())
    
    def analyze_use_case(self) -> AnalyzeGraphUseCase:
        return AnalyzeGraphUseCase(repository=self.graph_repository())
    
    def analysis_service(self) -> 'AnalysisService':
        from ..application.services.analysis_service import AnalysisService
        return AnalysisService(repository=self.graph_repository())

    def simulation_service(self) -> 'SimulationService':
        from ..application.services.simulation_service import SimulationService
        return SimulationService(repository=self.graph_repository())

    def validation_service(self, targets=None) -> 'ValidationService':
        from ..application.services.validation_service import ValidationService
        return ValidationService(
            analysis_service=self.analysis_service(),
            simulation_service=self.simulation_service(),
            targets=targets
        )

    def close(self) -> None:
        if self._repository:
            self._repository.close()
            self._repository = None
