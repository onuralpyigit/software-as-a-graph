"""
Visualization Data Models

v3.1 additions to LayerData:
    cascade_results     — list of per-component cascade risk dicts (for §6.4.5)
    cascade_risk_topo   — topology-only cascade baseline dict
    qos_gini            — QoS heterogeneity coefficient (Middleware 2026)
    cascade_wilcoxon_p  — Wilcoxon p-value for QoS ablation test
    cascade_delta_rho   — Δρ (QoS-enriched − topology-only)
    hierarchy_data      — MIL-STD-498 tree dict for §6.2 Section 10
    multiseed_rho       — list of ρ values per seed (for stability panel)
    multiseed_f1        — list of F1 values per seed
    multiseed_seeds     — list of seed labels (str) matching above lists
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
    primary: str = "#3498db"
    secondary: str = "#2c3e50"
    success: str = "#2ecc71"
    warning: str = "#f39c12"
    danger: str = "#e74c3c"
    info: str = "#17a2b8"
    light: str = "#ecf0f1"
    dark: str = "#34495e"

    critical: str = "#e74c3c"
    high: str = "#e67e22"
    medium: str = "#f1c40f"
    low: str = "#2ecc71"
    minimal: str = "#95a5a6"

    layer_app: str = "#3498db"
    layer_infra: str = "#9b59b6"
    layer_mw_app: str = "#1abc9c"
    layer_mw_infra: str = "#e67e22"
    layer_system: str = "#2c3e50"

    def to_colors_dict(self) -> Dict[str, str]:
        return {
            "primary": self.primary, "secondary": self.secondary,
            "success": self.success, "warning": self.warning,
            "danger": self.danger, "info": self.info,
            "light": self.light, "dark": self.dark,
        }

    def to_criticality_dict(self) -> Dict[str, str]:
        return {
            "CRITICAL": self.critical, "HIGH": self.high,
            "MEDIUM": self.medium, "LOW": self.low,
            "MINIMAL": self.minimal,
        }

    def to_layer_dict(self) -> Dict[str, str]:
        return {
            "app": self.layer_app, "infra": self.layer_infra,
            "mw-app": self.layer_mw_app, "mw-infra": self.layer_mw_infra,
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
    anti_patterns: List[str] = field(default_factory=list)
    mpci: float = 0.0
    foc: float = 0.0
    spof: bool = False
    explanation: Optional[Dict[str, Any]] = None
    # Cascade risk (§6.4.5) — populated by cascade_risk_scorer
    cascade_risk: float = 0.0
    cascade_risk_topo: float = 0.0  # topology-only baseline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "type": self.type,
            "reliability": self.reliability,
            "maintainability": self.maintainability,
            "availability": self.availability,
            "vulnerability": self.vulnerability,
            "overall": self.overall, "level": self.level,
            "impact": self.impact, "cascade_depth": self.cascade_depth,
            "anti_patterns": self.anti_patterns,
            "mpci": self.mpci, "foc": self.foc, "spof": self.spof,
            "explanation": self.explanation,
            "cascade_risk": self.cascade_risk,
            "cascade_risk_topo": self.cascade_risk_topo,
        }


@dataclass
class LayerData:
    """Aggregated data for a single analysis layer (Definition 9)."""
    layer: str
    name: str

    # Graph topology
    nodes: int = 0
    edges: int = 0
    density: float = 0.0
    connected_components: int = 0

    # Component type counts
    component_counts: Dict[str, int] = field(default_factory=dict)

    # Criticality classification
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    minimal_count: int = 0
    spof_count: int = 0
    problems_count: int = 0

    # Top component ranking
    top_components: List[Dict[str, Any]] = field(default_factory=list)

    # Simulation outputs
    avg_impact: float = 0.0
    max_impact: float = 0.0
    event_throughput: int = 0
    event_delivery_rate: float = 0.0

    # Primary validation metrics
    spearman: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    validation_passed: bool = False

    # Network graph data for interactive visualisation
    network_nodes: List[Dict[str, Any]] = field(default_factory=list)
    network_edges: List[Dict[str, Any]] = field(default_factory=list)
    rcm_order: List[str] = field(default_factory=list)

    # Per-dimension Spearman ρ
    reliability_spearman: float = 0.0
    maintainability_spearman: float = 0.0
    availability_spearman: float = 0.0
    vulnerability_spearman: float = 0.0
    composite_spearman: float = 0.0
    predictive_gain: float = 0.0

    # Validation gates G1-G4
    gates: Dict[str, bool] = field(default_factory=dict)

    # Anti-pattern findings
    anti_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # Component ID → display name mapping
    csc_names: Dict[str, str] = field(default_factory=dict)

    # Full component RMAV details
    component_details: List[ComponentDetail] = field(default_factory=list)

    # Scatter plot data: (id, Q(v), I(v), level)
    scatter_data: List[Tuple[str, float, float, str]] = field(default_factory=list)
    reliability_scatter: List[Tuple[str, float, float, str]] = field(default_factory=list)
    maintainability_scatter: List[Tuple[str, float, float, str]] = field(default_factory=list)
    availability_scatter: List[Tuple[str, float, float, str]] = field(default_factory=list)
    vulnerability_scatter: List[Tuple[str, float, float, str]] = field(default_factory=list)

    # Bootstrap confidence intervals per dimension
    reliability_ci: Optional[Tuple[float, float]] = None
    maintainability_ci: Optional[Tuple[float, float]] = None
    availability_ci: Optional[Tuple[float, float]] = None
    vulnerability_ci: Optional[Tuple[float, float]] = None
    composite_ci: Optional[Tuple[float, float]] = None

    # Top-K overlap
    top5_overlap: float = 0.0
    top10_overlap: float = 0.0

    # Human-readable architectural explanation
    explanation: Optional[Dict[str, Any]] = None

    # ── v3.1 additions ─────────────────────────────────────────────────

    # §6.4.5 Cascade risk (QoS ablation, Middleware 2026)
    # Each entry: {"id", "name", "type", "cascade_risk", "cascade_risk_topo",
    #              "cascade_depth", "level"}
    cascade_results: List[Dict[str, Any]] = field(default_factory=list)
    qos_gini: float = 0.0           # QoS heterogeneity coefficient
    cascade_wilcoxon_p: float = 1.0  # Wilcoxon p for QoS vs topo-only
    cascade_delta_rho: float = 0.0   # Δρ (QoS-enriched − baseline)

    # §6.2 Section 10 — MIL-STD-498 hierarchy tree
    # Schema: {"id", "label", "level" (CSS/CSCI/CSC/CSU), "q", "cbci",
    #          "children": [...recursive...]}
    hierarchy_data: Optional[Dict[str, Any]] = None

    # §6.2 Section 8 — Multi-seed stability
    multiseed_rho: List[float] = field(default_factory=list)
    multiseed_f1: List[float] = field(default_factory=list)
    multiseed_seeds: List[str] = field(default_factory=list)

    # ── Computed properties ─────────────────────────────────────────────

    @property
    def classification_distribution(self) -> Dict[str, int]:
        return {
            "CRITICAL": self.critical_count,
            "HIGH":     self.high_count,
            "MEDIUM":   self.medium_count,
            "LOW":      self.low_count,
            "MINIMAL":  self.minimal_count,
        }

    @property
    def total_classified(self) -> int:
        return (
            self.critical_count + self.high_count + self.medium_count
            + self.low_count + self.minimal_count
        )

    @property
    def scale_category(self) -> str:
        if self.nodes < 50:
            return "small"
        elif self.nodes < 200:
            return "medium"
        elif self.nodes < 500:
            return "large"
        return "xlarge"

    @property
    def recommend_matrix_only(self) -> bool:
        """Recommend matrix-only view for graphs > 80 nodes."""
        return self.nodes > 80

    @property
    def has_simulation(self) -> bool:
        return self.max_impact > 0

    @property
    def has_validation(self) -> bool:
        return self.spearman > 0

    @property
    def has_cascade(self) -> bool:
        return len(self.cascade_results) > 0

    @property
    def has_hierarchy(self) -> bool:
        return self.hierarchy_data is not None

    @property
    def dim_rho(self) -> Dict[str, float]:
        """Convenience dict of per-dimension ρ for dim_rho_bars()."""
        return {
            "availability":    self.availability_spearman,
            "reliability":     self.reliability_spearman,
            "maintainability": self.maintainability_spearman,
            "vulnerability":   self.vulnerability_spearman,
            "infrastructure":  getattr(self, "infrastructure_spearman", 0.0),
        }