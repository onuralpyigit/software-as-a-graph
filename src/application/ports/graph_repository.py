from typing import Protocol, Dict, Any, List, Optional, Set
from ...domain.models import GraphEntity
from ..dto import GraphData

class GraphRepository(Protocol):
    """Port for graph persistence operations."""
    
    def save_graph(self, data: Dict[str, Any], clear: bool = False) -> None:
        """Import graph data into the repository."""
        ...
    
    def get_graph_data(
        self,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None,
        include_raw: bool = False,
    ) -> GraphData:
        """Retrieve graph data with optional filtering."""
        ...
    
    def get_layer_data(self, layer: str) -> GraphData:
        """Retrieve graph data for a specific layer."""
        ...
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        ...
    
    def export_json(self) -> Dict[str, Any]:
        """Export graph as JSON."""
        ...
    
    def close(self) -> None:
        """Close connection to repository."""
        ...
