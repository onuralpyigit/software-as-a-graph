"""
Compatibility shim: src.core.graph_exporter

Maps the old GraphExporter API to Neo4jGraphRepository,
and re-exports GraphData, ComponentData, EdgeData from domain models.
"""
from src.adapters.outbound.neo4j_repo import Neo4jGraphRepository
from src.domain.models.graph import GraphData, ComponentData, EdgeData

# Structural relationship types used by the API
STRUCTURAL_REL_TYPES = [
    "RUNS_ON", "PUBLISHES_TO", "SUBSCRIBES_TO", "ROUTES", "CONNECTS_TO", "USES"
]


class GraphExporter:
    """Backward-compatible wrapper around Neo4jGraphRepository for export/query."""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password", database="neo4j"):
        self._repo = Neo4jGraphRepository(uri=uri, user=user, password=password, database=database)
        # Expose driver for raw Cypher queries used in api.py
        self.driver = self._repo.driver

    def __enter__(self):
        self._repo.__enter__()
        return self

    def __exit__(self, *args):
        self._repo.__exit__(*args)

    def close(self):
        self._repo.close()

    def get_graph_data(self, component_types=None, dependency_types=None):
        """Retrieve graph data — delegates to repository."""
        return self._repo.get_graph_data(
            component_types=component_types,
            dependency_types=dependency_types,
        )

    def export_graph_data(self):
        """Export graph in input-compatible format — delegates to repository."""
        return self._repo.export_json()


__all__ = [
    "GraphExporter", "GraphData", "ComponentData", "EdgeData",
    "STRUCTURAL_REL_TYPES",
]
