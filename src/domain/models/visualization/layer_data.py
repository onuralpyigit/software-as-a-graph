"""
Layer Data Domain Models
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

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
    """Detailed component info with RMAV quality breakdown and impact."""
    id: str
    name: str
    type: str
    reliability: float = 0.0
    maintainability: float = 0.0
    availability: float = 0.0
    vulnerability: float = 0.0
    overall: float = 0.0
    level: str = "MINIMAL"
    impact: float = 0.0

@dataclass
class LayerData:
    """Aggregated data for a single layer."""
    layer: str
    name: str
    
    # Graph statistics
    nodes: int = 0
    edges: int = 0
    density: float = 0.0
    connected_components: int = 0
    
    # Component breakdown
    component_counts: Dict[str, int] = field(default_factory=dict)
    
    # Analysis results
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    minimal_count: int = 0
    spof_count: int = 0
    problems_count: int = 0
    
    # Top components
    top_components: List[Dict[str, Any]] = field(default_factory=list)
    
    # Simulation results
    avg_impact: float = 0.0
    max_impact: float = 0.0
    event_throughput: int = 0
    event_delivery_rate: float = 0.0
    
    # Validation results
    spearman: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    validation_passed: bool = False
    
    # Network graph data
    network_nodes: List[Dict[str, Any]] = field(default_factory=list)
    network_edges: List[Dict[str, Any]] = field(default_factory=list)

    # Name mapping
    component_names: Dict[str, str] = field(default_factory=dict)

    # Component details with RMAV breakdown
    component_details: List[ComponentDetail] = field(default_factory=list)

    # Scatter plot data: (id, Q(v), I(v), level)
    scatter_data: List[Tuple[str, float, float, str]] = field(default_factory=list)

    # Top-K overlap metrics
    top5_overlap: float = 0.0
    top10_overlap: float = 0.0

    # Flag indicating matrix-only visualization is recommended (for large graphs)
    recommend_matrix_only: bool = False

    @property
    def has_validation(self) -> bool:
        """Check if validation data is present."""
        return self.spearman != 0.0 or self.f1_score != 0.0

    @property
    def has_simulation(self) -> bool:
        """Check if simulation data is present."""
        return self.event_throughput > 0 or self.avg_impact > 0.0

    @property
    def classification_distribution(self) -> Dict[str, int]:
        """Get distribution of classification levels."""
        return {
            "CRITICAL": self.critical_count,
            "HIGH": self.high_count,
            "MEDIUM": self.medium_count,
            "LOW": self.low_count,
            "MINIMAL": self.minimal_count,
        }
