from typing import Dict, Any
from src.core.ports.graph_repository import IGraphRepository
from .models import ImportStats

class ModelGraphUseCase:
    """Use case for importing/modeling a graph into the repository."""
    
    def __init__(self, repository: IGraphRepository):
        self.repository = repository
        
    def execute(self, graph_data: Dict[str, Any], clear: bool = False, dry_run: bool = False) -> ImportStats:
        import time
        start_time = time.time()
        
        if dry_run:
            # Simulate counts from input data
            nodes_count = (
                len(graph_data.get("nodes", [])) +
                len(graph_data.get("brokers", [])) +
                len(graph_data.get("topics", [])) +
                len(graph_data.get("applications", [])) +
                len(graph_data.get("libraries", []))
            )
            rel_dict = graph_data.get("relationships", {})
            edges_count = sum(len(items) for items in rel_dict.values())
            
            duration = (time.time() - start_time) * 1000
            
            return ImportStats(
                nodes_imported=nodes_count,
                edges_imported=edges_count,
                duration_ms=duration,
                details={"dry_run": True, "source": "input_file_parsing"},
                success=True,
                message="Dry run complete (no changes made to database)"
            )

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

