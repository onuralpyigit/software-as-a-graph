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
    
    def close(self) -> None:
        if self._repository:
            self._repository.close()
            self._repository = None
