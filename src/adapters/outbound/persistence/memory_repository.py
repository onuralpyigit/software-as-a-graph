"""
In-Memory Graph Repository Adapter

Implements IGraphRepository using in-memory storage for testing.
"""

from typing import Dict, List, Optional, Any

from src.application.ports.outbound.graph_repository import IGraphRepository
from src.domain.models import GraphData

# Re-use legacy implementation during migration
from src.repositories.memory_repository import InMemoryGraphRepository as LegacyMemoryRepository


class InMemoryGraphRepository(IGraphRepository):
    """
    In-memory adapter implementing IGraphRepository.
    
    Wraps the legacy InMemoryGraphRepository for incremental migration.
    Useful for testing without Neo4j dependency.
    """
    
    def __init__(self):
        """Initialize in-memory repository."""
        self._legacy = LegacyMemoryRepository()
    
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
        """No-op for in-memory repository."""
        pass
