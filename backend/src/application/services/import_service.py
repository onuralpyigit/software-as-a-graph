"""
Graph Import Application Service
"""
from typing import Dict, Any
from src.adapters.outbound.neo4j_repo import Neo4jGraphRepository

class ImportService:
    """Application service for importing graph data into Neo4j."""
    
    def __init__(self, uri, user, password, database="neo4j"):
        self.repo = Neo4jGraphRepository(uri, user, password, database)

    def __enter__(self):
        self.repo.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.repo.__exit__(exc_type, exc_val, exc_tb)
    
    def import_graph(self, data: Dict[str, Any], clear: bool = False) -> Dict[str, Any]:
        """Import graph data into the database."""
        return self.repo.save_graph(data, clear=clear)
