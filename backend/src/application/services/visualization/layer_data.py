"""
Layer Data Domain Models

Aggregated data structures for the visualization pipeline (Step 6).
Each LayerData instance captures the complete output set from Steps 1-5
for a single layer, as specified in Definition 9 of docs/visualization.md.

Fields are organized by source step:
    - Step 1: Graph statistics, component counts, network topology
    - Step 2: Structural metrics (via top_components)
    - Step 3: Quality scores, RMAV breakdown, classifications
    - Step 4: Simulation impact scores, cascade data
    - Step 5: Validation metrics (Spearman, F1, Top-K overlap)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

# Layer definitions
LAYER_DEFINITIONS = {
    "app": {
        "name": "Application Layer",
        "description": "Application-to-application dependencies",
        "icon": "📱",
    },
    "infra": {
        "name": "Infrastructure Layer",
        "description": "Node-to-node connections",
        "icon": "🖥️",
    },
    "mw": {
        "name": "Middleware Layer",
        "description": "Broker dependencies",
        "icon": "🔗",
    },
    "system": {
        "name": "Complete System",
        "description": "All components and dependencies",
        "icon": "🌐",
    },
}


@dataclass
class ComponentDetail:
    """
    Detailed component data for visualization tables and charts.
    
    Combines structural metrics (Step 2), quality scores (Step 3),
    and impact scores (Step 4) for a single component.
    """
    id: str
    name: str
    type: str
    # RMAV scores (Step 3)
    reliability: float = 0.0
    maintainability: float = 0.0
    availability: float = 0.0
    vulnerability: float = 0.0
    overall: float = 0.0
    level: str = "MINIMAL"
    # Impact score (Step 4)
    impact: float = 0.0
    cascade_depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "reliability": self.reliability,
            "maintainability": self.maintainability,
            "availability": self.availability,
            "vulnerability": self.vulnerability,
            "overall": self.overall,
            "level": self.level,
            "impact": self.impact,
            "cascade_depth": self.cascade_depth,
        }


@dataclass
class LayerData:
    """
    Aggregated data for a single analysis layer.
    
    This is the primary data structure consumed by the visualization pipeline.
    Each field traces back to a specific methodology step, ensuring analytical
    traceability (§6.3.3 of visualization.md).
    """
    layer: str
    name: str

    # ─── Step 1: Graph Statistics ────────────────────────────────────────
    nodes: int = 0
    edges: int = 0
    density: float = 0.0
    connected_components: int = 0
    component_counts: Dict[str, int] = field(default_factory=dict)

    # ─── Step 3: Classification Counts ───────────────────────────────────
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    minimal_count: int = 0
    spof_count: int = 0
    problems_count: int = 0

    # ─── Steps 2+3+4: Component Details (for tables, charts) ────────────
    top_components: List[Dict[str, Any]] = field(default_factory=list)
    component_details: List[ComponentDetail] = field(default_factory=list)

    # ─── Step 3+4: Scatter Plot Data (Q(v) vs I(v)) ─────────────────────
    scatter_data: List[Tuple[str, float, float, str]] = field(default_factory=list)
    # Each tuple: (component_id, Q(v), I(v), level)

    # ─── Step 4: Simulation Results ──────────────────────────────────────
    avg_impact: float = 0.0
    max_impact: float = 0.0
    event_throughput: int = 0
    event_delivery_rate: float = 0.0

    # ─── Step 5: Validation Results ──────────────────────────────────────
    spearman: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    top5_overlap: float = 0.0
    top10_overlap: float = 0.0
    validation_passed: bool = False

    # ─── Step 1: Network Graph Data (for interactive visualization) ──────
    network_nodes: List[Dict[str, Any]] = field(default_factory=list)
    network_edges: List[Dict[str, Any]] = field(default_factory=list)

    # ─── Auxiliary Mappings ──────────────────────────────────────────────
    component_names: Dict[str, str] = field(default_factory=dict)

    # ─── Scalability Metadata ────────────────────────────────────────────
    @property
    def scale_category(self) -> str:
        """Determine system scale for visualization strategy (§6.11)."""
        if self.nodes > 1000:
            return "xlarge"
        elif self.nodes > 200:
            return "large"
        elif self.nodes > 50:
            return "medium"
        return "small"

    @property
    def recommend_matrix_only(self) -> bool:
        """Whether to recommend matrix over network graph for this scale."""
        return self.nodes > 200

    @property
    def classification_distribution(self) -> Dict[str, int]:
        """Criticality distribution as a dict (for chart generation)."""
        return {
            "CRITICAL": self.critical_count,
            "HIGH": self.high_count,
            "MEDIUM": self.medium_count,
            "LOW": self.low_count,
            "MINIMAL": self.minimal_count,
        }

    @property
    def total_classified(self) -> int:
        """Total number of classified components."""
        return (
            self.critical_count
            + self.high_count
            + self.medium_count
            + self.low_count
            + self.minimal_count
        )

    @property
    def has_validation(self) -> bool:
        """Whether validation data is available."""
        return self.spearman > 0

    @property
    def has_simulation(self) -> bool:
        """Whether simulation data is available."""
        return self.event_throughput > 0 or self.max_impact > 0