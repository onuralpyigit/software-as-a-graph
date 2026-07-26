import time
from typing import Any, Dict

from saag.core.ports.graph_repository import IGraphRepository
from .models import ImportStats

#: Topology sections holding vertices, in the order they are reported.
COMPONENT_SECTIONS = ("nodes", "brokers", "topics", "applications", "libraries")

#: Relationship sections that drive the derivation rules producing most
#: DEPENDS_ON edges (Rules 1, 2 and 5); used for the dry-run estimate.
DERIVATION_DRIVING_SECTIONS = ("publishes_to", "subscribes_to", "uses")


class ModelGraphUseCase:
    """Use case for importing/modeling a graph into the repository."""

    def __init__(self, repository: IGraphRepository):
        self.repository = repository

    def _validate_graph_data(self, data: Dict[str, Any]) -> None:
        """Lightweight structural validation of input graph data."""
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary.")

        # Every vertex needs a non-empty id — edges are resolved by id
        for section in COMPONENT_SECTIONS:
            for index, item in self._iter_entries(data.get(section, []), f"section '{section}'"):
                if not item.get("id"):
                    raise ValueError(
                        f"Item {index} in {section} is missing a required non-empty 'id'."
                    )

        # Every edge needs both endpoints, under either accepted key spelling
        rels = data.get("relationships", {})
        if not isinstance(rels, dict):
            raise ValueError("Section 'relationships' must be a dictionary of lists.")
        for rel_type, items in rels.items():
            label = f"relationship type '{rel_type}'"
            for index, item in self._iter_entries(items, label):
                if not (item.get("source") or item.get("from")):
                    raise ValueError(f"Item {index} in {label} is missing 'source' (or 'from').")
                if not (item.get("target") or item.get("to")):
                    raise ValueError(f"Item {index} in {label} is missing 'target' (or 'to').")

    @staticmethod
    def _iter_entries(items: Any, label: str):
        """Yield ``(index, entry)`` after checking the section is a list of dicts."""
        if not isinstance(items, list):
            raise ValueError(f"{label.capitalize()} must be a list of dictionaries.")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Item {index} in {label} must be a dictionary.")
            yield index, item

    def execute(
        self, graph_data: Dict[str, Any], clear: bool = False, dry_run: bool = False
    ) -> ImportStats:
        start_time = time.time()
        self._validate_graph_data(graph_data)

        if dry_run:
            return self._dry_run_stats(graph_data, start_time)

        self.repository.save_graph(graph_data, clear=clear)
        full_stats = self.repository.get_statistics()

        return ImportStats(
            nodes_imported=full_stats.get("total_nodes", 0),
            edges_imported=full_stats.get("total_relationships", 0),
            duration_ms=(time.time() - start_time) * 1000,
            details=full_stats,
            success=True,
            message="Graph modeled successfully"
        )

    @staticmethod
    def _dry_run_stats(graph_data: Dict[str, Any], start_time: float) -> ImportStats:
        """Report the counts an import would produce, without touching the database."""
        nodes_count = sum(len(graph_data.get(section, [])) for section in COMPONENT_SECTIONS)
        rels = graph_data.get("relationships", {})
        structural_edges = sum(len(items) for items in rels.values())

        # Lower bound only: the real count is decided by Cypher MERGE semantics
        # when the pre-analysis stage derives the edges.
        estimated_depends_on = sum(
            len(rels.get(section, [])) for section in DERIVATION_DRIVING_SECTIONS
        )

        return ImportStats(
            nodes_imported=nodes_count,
            edges_imported=structural_edges,
            duration_ms=(time.time() - start_time) * 1000,
            details={
                "dry_run": True,
                "source": "input_file_parsing",
                "structural_edges": structural_edges,
                "estimated_depends_on": estimated_depends_on,
                "note": "edges_imported counts structural relationships only; "
                        "DEPENDS_ON edges are derived in the pre-analysis stage",
            },
            success=True,
            message="Dry run complete (no changes made to database)"
        )
