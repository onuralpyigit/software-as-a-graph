"""
Compatibility shim: src.analysis.analyzer

Maps the old GraphAnalyzer(uri, user, password) API to the new
AnalysisService + Neo4jGraphRepository combo.
"""
from src.core.neo4j_repo import Neo4jGraphRepository
from src.analysis import AnalysisService


class GraphAnalyzer:
    """Backward-compatible facade wrapping AnalysisService."""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self._repo = Neo4jGraphRepository(uri=uri, user=user, password=password)
        self._service = AnalysisService(repository=self._repo)

    def __enter__(self):
        self._repo.__enter__()
        return self

    def __exit__(self, *args):
        self._repo.__exit__(*args)

    def close(self):
        self._repo.close()

    def analyze_layer(self, layer="system"):
        """Run the full analysis pipeline for a single layer."""
        return self._service.analyze_layer(layer)

    def analyze_all_layers(self, include_cross_layer=True):
        """Analyse every primary layer."""
        return self._service.analyze_all_layers(include_cross_layer=include_cross_layer)


__all__ = ["GraphAnalyzer"]
