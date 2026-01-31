"""
Graph Repository Port

Interface defining the contract for graph data persistence.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from src.domain.models import GraphData


class IGraphRepository(ABC):
    """
    Outbound port for graph data persistence.
    
    Defines the contract for loading and saving graph data
    regardless of the underlying storage mechanism (Neo4j, in-memory, etc.).
    """
    
    @abstractmethod
    def get_graph_data(
        self,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None,
        include_raw: bool = False
    ) -> GraphData:
        """
        Retrieve graph data with optional type filtering.
        
        Args:
            component_types: Types of components to include
            dependency_types: Types of dependencies to include
            include_raw: Include raw structural relationships
            
        Returns:
            GraphData containing components and edges
        """
        pass
    
    @abstractmethod
    def get_layer_data(self, layer: str) -> GraphData:
        """
        Retrieve graph data for a specific layer.
        
        Args:
            layer: Layer name (app, infra, mw, system)
            
        Returns:
            GraphData filtered for the layer
        """
        pass
    
    @abstractmethod
    def save_graph(self, data: Dict[str, Any], clear: bool = False) -> None:
        """
        Save graph data to the repository.
        
        Args:
            data: Graph data dictionary
            clear: Clear existing data before import
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve graph statistics.
        
        Returns:
            Dictionary with component and dependency counts
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close any open connections."""
        pass
