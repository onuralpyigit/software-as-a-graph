"""
Application Container

Dependency injection container that wires ports to adapters and manages service lifecycle.
Includes settings configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Settings
# =============================================================================

@dataclass
class Settings:
    """Application settings from environment."""
    
    # Neo4j connection
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


# =============================================================================
# Container (Lazy import to avoid circular dependencies)
# =============================================================================

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
    
    _repository: Optional[object] = field(default=None, repr=False)
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "Container":
        """Create container from settings."""
        return cls(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
    
    def graph_repository(self):
        """Get the graph repository singleton."""
        if not self._repository:
            # Lazy import to avoid circular dependencies
            from src.adapters.outbound.neo4j_repo import Neo4jGraphRepository
            self._repository = Neo4jGraphRepository(
                self.uri, self.user, self.password
            )
        return self._repository
    
    def analysis_service(self):
        """Get analysis use case implementation."""
        from src.application.services.analysis_service import AnalysisService
        return AnalysisService(repository=self.graph_repository())
    
    def simulation_service(self):
        """Get simulation use case implementation."""
        from src.application.services.simulation_service import SimulationService
        return SimulationService(repository=self.graph_repository())
    
    def validation_service(self, targets=None):
        """Get validation use case implementation."""
        from src.application.services.validation_service import ValidationService
        return ValidationService(
            analysis_service=self.analysis_service(),
            simulation_service=self.simulation_service(),
            targets=targets
        )
    
    def display_service(self):
        """Get console display adapter."""
        from src.adapters.inbound.cli.display import ConsoleDisplay
        return ConsoleDisplay()
    
    def visualization_service(self):
        """Get visualization service."""
        from src.application.services.visualization_service import VisualizationService
        return VisualizationService(
            analysis_service=self.analysis_service(),
            simulation_service=self.simulation_service(),
            validation_service=self.validation_service(),
            repository=self.graph_repository()
        )
    
    def close(self) -> None:
        """Close all resources."""
        if self._repository:
            self._repository.close()
            self._repository = None
