from typing import Dict, Any
from src.core.ports.graph_repository import IGraphRepository
from .models import ImportStats

class ModelGraphUseCase:
    """Use case for importing/modeling a graph into the repository."""
    
    def __init__(self, repository: IGraphRepository):
        self.repository = repository
        
    def execute(self, graph_data: Dict[str, Any], clear: bool = False) -> ImportStats:
        import time
        start_time = time.time()
        
        #repository.save_graph(graph_data, clear=clear)
        # Assuming save_graph returns nothing or we need to fetch stats
        # Let's check save_graph implementation in neo4j_repo.py
        
        self.repository.save_graph(graph_data, clear=clear)
        
        # Simple stats for now, as save_graph doesn't return much
        duration = (time.time() - start_time) * 1000
        
        # Get full statistics for details
        full_stats = self.repository.get_statistics()
        
        return ImportStats(
            nodes_imported=full_stats.get("node_count", 0),
            edges_imported=full_stats.get("connects_to_count", 0) + full_stats.get("uses_count", 0), # Simplified
            duration_ms=duration,
            details=full_stats,
            success=True,
            message="Graph modeled successfully"
        )
