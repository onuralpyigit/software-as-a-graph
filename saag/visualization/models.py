"""
Visualization Data Models

`LayerData` is the single aggregate the dashboard renders from: one instance
per analysis layer, populated by `LayerDataCollector` from the analysis,
prediction, simulation and validation services, then consumed by
`VisualizationService.build_html()`.

Layer names and display labels come from `saag.core.layers` — this module
does not define its own.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional


@dataclass
class ComponentDetail:
    """Detailed component info with RMAV quality breakdown and impact."""
    id: str
    name: str
    type: str
    reliability: float = 0.0
    maintainability: float = 0.0
    availability: float = 0.0
    security: float = 0.0
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
            "security": self.security,
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

    # Per-dimension Spearman ρ
    reliability_spearman: float = 0.0
    maintainability_spearman: float = 0.0
    availability_spearman: float = 0.0
    security_spearman: float = 0.0
    composite_spearman: float = 0.0
    predictive_gain: float = 0.0

    # Validation gates G1-G4
    gates: Dict[str, bool] = field(default_factory=dict)

    # Anti-pattern findings
    anti_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # Full component RMAV details
    component_details: List[ComponentDetail] = field(default_factory=list)

    # Scatter plot data: (id, Q(v), I(v), level)
    scatter_data: List[Tuple[str, float, float, str]] = field(default_factory=list)
    reliability_scatter: List[Tuple[str, float, float, str]] = field(default_factory=list)
    maintainability_scatter: List[Tuple[str, float, float, str]] = field(default_factory=list)
    availability_scatter: List[Tuple[str, float, float, str]] = field(default_factory=list)
    security_scatter: List[Tuple[str, float, float, str]] = field(default_factory=list)

    # Bootstrap confidence intervals per dimension
    reliability_ci: Optional[Tuple[float, float]] = None
    maintainability_ci: Optional[Tuple[float, float]] = None
    availability_ci: Optional[Tuple[float, float]] = None
    security_ci: Optional[Tuple[float, float]] = None
    composite_ci: Optional[Tuple[float, float]] = None

    # Top-K overlap
    top5_overlap: float = 0.0
    top10_overlap: float = 0.0

    # Human-readable architectural explanation
    explanation: Optional[Dict[str, Any]] = None

    # §6.4.5 Cascade risk (QoS ablation, Middleware 2026)
    # Each entry: {"id", "name", "type", "cascade_risk", "cascade_risk_topo",
    #              "cascade_depth", "level"}
    cascade_results: List[Dict[str, Any]] = field(default_factory=list)
    qos_gini: float = 0.0           # QoS heterogeneity coefficient
    cascade_wilcoxon_p: float = 1.0  # Wilcoxon p for QoS vs topo-only
    cascade_delta_rho: float = 0.0   # Δρ (QoS-enriched − baseline)

    # MIL-STD-498 tab — hierarchy tree
    # Schema: {"id", "label", "level" (CSS/CSCI/CSC/CSU), "q", "cbci",
    #          "children": [...recursive...]}
    hierarchy_data: Optional[Dict[str, Any]] = None

    # Validation tab — multi-seed stability
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
            "security":        self.security_spearman,
        }