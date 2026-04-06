from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

@dataclass
class ImportStats:
    """Statistics from the graph import process."""
    nodes_imported: int
    edges_imported: int
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a flat dictionary for display/JSON serialization."""
        d = {
            "nodes_imported": self.nodes_imported,
            "edges_imported": self.edges_imported,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "message": self.message,
        }
        if self.details:
            d.update(self.details)
        return d


class SimulationMode(Enum):
    EXHAUSTIVE = "exhaustive"
    MONTE_CARLO = "monte_carlo"
    SINGLE = "single"
    PAIRWISE = "pairwise"
    EVENT = "event"
    REPORT = "report"
    CLASSIFY = "classify"

@dataclass
class VisOptions:
    """Options for dashboard visualization."""
    include_network: bool = True
    include_matrix: bool = True
    include_validation: bool = True
    include_per_dim_scatter: bool = True
    antipatterns_file: Optional[str] = None
    multi_seed: int = 0
