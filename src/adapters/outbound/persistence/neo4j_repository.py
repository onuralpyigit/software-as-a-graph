"""
Neo4j Graph Repository Adapter

Implements IGraphRepository using Neo4j as the backend.
"""

from typing import Dict, List, Optional, Any

from src.application.ports.outbound.graph_repository import IGraphRepository
from src.domain.models import GraphData

# Re-use legacy implementation during migration
from src.repositories.graph_repository import GraphRepository as LegacyGraphRepository


class Neo4jGraphRepository(IGraphRepository):
    """
    Neo4j adapter implementing IGraphRepository.
    
    Wraps the legacy GraphRepository for incremental migration.
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        """Initialize Neo4j repository."""
        self._legacy = LegacyGraphRepository(uri, user, password, database)
    
    def get_graph_data(
        self,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None,
        include_raw: bool = False
    ) -> GraphData:
        """Retrieve graph data with optional type filtering."""
        return self._legacy.get_graph_data(component_types, dependency_types, include_raw)
    
    def get_layer_data(self, layer: str) -> GraphData:
        """Retrieve graph data for a specific layer."""
        return self._legacy.get_layer_data(layer)
    
    def save_graph(self, data: Dict[str, Any], clear: bool = False) -> None:
        """Save graph data to the repository."""
        self._legacy.save_graph(data, clear)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retrieve graph statistics."""
        return self._legacy.get_statistics()
    
    def close(self) -> None:
        """Close Neo4j driver connection."""
        self._legacy.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
