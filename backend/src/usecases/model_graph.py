from typing import Dict, Any
from src.core.ports.graph_repository import IGraphRepository
from .models import ImportStats

class ModelGraphUseCase:
    """Use case for importing/modeling a graph into the repository."""
    
    def __init__(self, repository: IGraphRepository):
        self.repository = repository
        
    def _validate_graph_data(self, data: Dict[str, Any]) -> None:
        """Lightweight structural validation of input graph data."""
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary.")

        # Validate nodes, brokers etc. have IDs
        for key in ["nodes", "brokers", "topics", "applications", "libraries"]:
            items = data.get(key, [])
            if not isinstance(items, list):
                 raise ValueError(f"Section '{key}' must be a list of dictionaries.")
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                     raise ValueError(f"Item {i} in '{key}' must be a dictionary.")
                if "id" not in item or not item["id"]:
                     raise ValueError(f"Item {i} in '{key}' is missing a required non-empty 'id'.")

        # Validate relationships
        rels = data.get("relationships", {})
        if not isinstance(rels, dict):
             raise ValueError("Section 'relationships' must be a dictionary of lists.")
        for rel_type, items in rels.items():
            if not isinstance(items, list):
                 raise ValueError(f"Relationship type '{rel_type}' must be a list of dictionaries.")
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                     raise ValueError(f"Item {i} in relationship type '{rel_type}' must be a dictionary.")
                src = item.get("source") or item.get("from")
                tgt = item.get("target") or item.get("to")
                if not src:
                     raise ValueError(f"Item {i} in relationship type '{rel_type}' is missing 'source' (or 'from').")
                if not tgt:
                     raise ValueError(f"Item {i} in relationship type '{rel_type}' is missing 'target' (or 'to').")

    def execute(self, graph_data: Dict[str, Any], clear: bool = False, dry_run: bool = False) -> ImportStats:
        import time
        start_time = time.time()
        
        self._validate_graph_data(graph_data)
        
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

