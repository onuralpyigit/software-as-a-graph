"""
Repository Interface

Defines the IGraphRepository Protocol — the contract that all repository
implementations (Neo4j, in-memory, etc.) must satisfy.

Services depend on this Protocol rather than concrete implementations,
enabling dependency inversion and clean testability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.core.models import GraphData


@runtime_checkable
class IGraphRepository(Protocol):
    """
    Port for graph data persistence.

    Any class implementing these methods satisfies this protocol
    via structural subtyping — no explicit inheritance required.
    """

    def close(self) -> None:
        """Release resources associated with this repository."""
        ...

    def save_graph(self, data: Dict[str, Any], clear: bool = False) -> None:
        """Persist graph data. Optionally clear existing data first."""
        ...

    def get_graph_data(
        self,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None,
        include_raw: bool = False,
    ) -> GraphData:
        """Retrieve graph data with optional filtering by component/dependency type."""
        ...

    def get_layer_data(self, layer: str) -> GraphData:
        """Retrieve graph data projected onto a specific architectural layer."""
        ...

    def get_statistics(self) -> Dict[str, int]:
        """Retrieve aggregate counts of components and dependencies by type."""
        ...

    def export_json(self) -> Dict[str, Any]:
        """Export the complete graph as a JSON-compatible dictionary."""
        ...
