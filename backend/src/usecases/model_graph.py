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
        
        self.repository.save_graph(graph_data, clear=clear)
        
        duration = (time.time() - start_time) * 1000
        
        # Get full statistics for details
        full_stats = self.repository.get_statistics()
        
        return ImportStats(
            nodes_imported=full_stats.get("total_nodes", 0),
            edges_imported=full_stats.get("total_relationships", 0),
            duration_ms=duration,
            details=full_stats,
            success=True,
            message="Graph modeled successfully"
        )
