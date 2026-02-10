"""
Compatibility shim: src.core.graph_importer

Maps the old GraphImporter API to Neo4jGraphRepository.
"""
from src.adapters.outbound.neo4j_repo import Neo4jGraphRepository


class GraphImporter:
    """Backward-compatible wrapper around Neo4jGraphRepository for import."""

    def __init__(self, uri, user, password, database="neo4j"):
        self._repo = Neo4jGraphRepository(uri=uri, user=user, password=password, database=database)

    def __enter__(self):
        self._repo.__enter__()
        return self

    def __exit__(self, *args):
        self._repo.__exit__(*args)

    def close(self):
        self._repo.close()

    def import_graph(self, data, clear=False):
        """Import graph data — delegates to Neo4jGraphRepository.save_graph."""
        return self._repo.save_graph(data, clear=clear)


__all__ = ["GraphImporter"]
