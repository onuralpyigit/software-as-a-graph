"""
Visualization Data Models
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional

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
class ChartOutput:
    """Output from chart generation."""
    title: str
    png_base64: str
    description: str = ""
    alt_text: str = ""
    width: int = 600
    height: int = 400

@dataclass
class ColorTheme:
    """Configurable color theme for charts."""
    # Primary semantic colors
    primary: str = "#3498db"
    secondary: str = "#2c3e50"
    success: str = "#2ecc71"
    warning: str = "#f39c12"
    danger: str = "#e74c3c"
    info: str = "#17a2b8"
    light: str = "#ecf0f1"
    dark: str = "#34495e"
    
    # Criticality level colors
    critical: str = "#e74c3c"
    high: str = "#e67e22"
    medium: str = "#f1c40f"
    low: str = "#2ecc71"
    minimal: str = "#95a5a6"
    
    # Layer-specific colors
    layer_app: str = "#3498db"
    layer_infra: str = "#9b59b6"
    layer_mw_app: str = "#1abc9c"
    layer_mw_infra: str = "#e67e22"
    layer_system: str = "#2c3e50"
    
    def to_colors_dict(self) -> Dict[str, str]:
        """Convert to COLORS dictionary format."""
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "success": self.success,
            "warning": self.warning,
            "danger": self.danger,
            "info": self.info,
            "light": self.light,
            "dark": self.dark,
        }
    
    def to_criticality_dict(self) -> Dict[str, str]:
        """Convert to CRITICALITY_COLORS dictionary format."""
        return {
            "CRITICAL": self.critical,
            "HIGH": self.high,
            "MEDIUM": self.medium,
            "LOW": self.low,
            "MINIMAL": self.minimal,
        }
    
    def to_layer_dict(self) -> Dict[str, str]:
        """Convert to LAYER_COLORS dictionary format."""
        return {
            "app": self.layer_app,
            "infra": self.layer_infra,
            "mw-app": self.layer_mw_app,
            "mw-infra": self.layer_mw_infra,
            "system": self.layer_system,
        }

DEFAULT_THEME = ColorTheme()

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
            "cascade_depth": self.cascade_depth
        }

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
    rcm_order: List[str] = field(default_factory=list)

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

    @property
    def recommend_matrix_only(self) -> bool:
        """Recommend matrix visualization for large graphs."""
        return self.nodes > 200

    @property
    def has_validation(self) -> bool:
        """Check if validation data is present."""
        return self.spearman != 0.0 or self.f1_score != 0.0

    @property
    def has_simulation(self) -> bool:
        """Check if simulation data is present."""
        return self.event_throughput > 0 or self.avg_impact > 0.0 or self.max_impact > 0.0

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

    @property
    def total_classified(self) -> int:
        """Sum of all classified components."""
        return (
            self.critical_count +
            self.high_count +
            self.medium_count +
            self.low_count +
            self.minimal_count
        )

    @property
    def scale_category(self) -> str:
        """Categorize layer size based on node count."""
        if self.nodes > 1000:
            return "xlarge"
        if self.nodes > 200:
            return "large"
        if self.nodes > 50:
            return "medium"
        return "small"
