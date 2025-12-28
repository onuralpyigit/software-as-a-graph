# API Reference

This document provides a comprehensive reference for the Python API.

---

## Table of Contents

1. [Core Module](#core-module)
2. [Simulation Module](#simulation-module)
3. [Validation Module](#validation-module)
4. [Analysis Module](#analysis-module)
5. [Visualization Module](#visualization-module)

---

## Core Module

Graph model and generation.

```python
from src.core import (
    # Graph Generation
    generate_graph,
    GraphGenerator,
    GraphConfig,
    
    # Graph Model
    GraphModel,
    Application,
    Topic,
    Broker,
    Node,
    Edge,
    
    # Enums
    VertexType,
    EdgeType,
    DependencyType,
    
    # QoS
    QoSPolicy,
    Durability,
    Reliability,
    Priority,
    
    # Neo4j
    GraphImporter,
)
```

### generate_graph()

Generate a pub-sub system graph.

```python
def generate_graph(
    scale: str = "medium",           # tiny, small, medium, large, xlarge
    scenario: str = "generic",       # generic, iot, financial, healthcare, etc.
    seed: Optional[int] = None,      # Random seed for reproducibility
    antipatterns: List[str] = None,  # god_topic, spof, chatty, bottleneck
) -> Dict:
    """Generate a pub-sub system graph."""
```

**Returns**: Dictionary with vertices and edges.

**Example**:
```python
graph = generate_graph(
    scale="medium",
    scenario="iot",
    seed=42,
    antipatterns=["god_topic", "spof"]
)
```

### GraphModel

Type-safe graph representation.

```python
class GraphModel:
    """Immutable graph model with query methods."""
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GraphModel":
        """Create from dictionary."""
    
    @property
    def applications(self) -> List[Application]:
        """All applications."""
    
    @property
    def topics(self) -> List[Topic]:
        """All topics."""
    
    @property
    def brokers(self) -> List[Broker]:
        """All brokers."""
    
    @property
    def nodes(self) -> List[Node]:
        """All infrastructure nodes."""
    
    def get_publishers(self, topic_id: str) -> List[Application]:
        """Get applications publishing to topic."""
    
    def get_subscribers(self, topic_id: str) -> List[Application]:
        """Get applications subscribing to topic."""
    
    def get_routed_topics(self, broker_id: str) -> List[Topic]:
        """Get topics routed by broker."""
    
    def summary(self) -> str:
        """Human-readable summary."""
```

### QoSPolicy

Quality of Service settings.

```python
@dataclass(frozen=True)
class QoSPolicy:
    durability: Durability = Durability.VOLATILE
    reliability: Reliability = Reliability.BEST_EFFORT
    priority: Priority = Priority.MEDIUM
    
    def weight(self) -> float:
        """Calculate QoS weight for dependency scoring."""
```

### GraphImporter

Neo4j import functionality.

```python
class GraphImporter:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        """Initialize importer with Neo4j credentials."""
    
    def import_graph(
        self,
        graph_data: Dict,
        batch_size: int = 100,
        clear_first: bool = True,
        derive_dependencies: bool = True,
    ) -> Dict:
        """Import graph to Neo4j."""
    
    def get_statistics(self) -> Dict:
        """Get import statistics."""
    
    def close(self):
        """Close connection."""
```

---

## Simulation Module

Failure and event simulation.

```python
from src.simulation import (
    # Graph Model
    SimulationGraph,
    Component,
    Connection,
    ComponentType,
    
    # Failure Simulation
    FailureSimulator,
    FailureResult,
    BatchSimulationResult,
    ImpactMetrics,
    AttackStrategy,
    
    # Event Simulation
    EventSimulator,
    SimulationResult,
    QoSLevel,
)
```

### SimulationGraph

Graph representation for simulation.

```python
class SimulationGraph:
    """Mutable graph for simulation operations."""
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SimulationGraph":
        """Create from dictionary."""
    
    @property
    def components(self) -> Dict[str, Component]:
        """All components by ID."""
    
    @property
    def connections(self) -> List[Connection]:
        """All connections."""
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """Get component by ID."""
    
    def get_components_by_type(self, comp_type: ComponentType) -> List[Component]:
        """Get all components of type."""
    
    def get_message_paths(self) -> List[Tuple[str, str, str]]:
        """Get all publisher→topic→subscriber paths."""
    
    def get_reachable_from(self, component_id: str) -> Set[str]:
        """Get all components reachable from source."""
    
    def copy(self) -> "SimulationGraph":
        """Create deep copy."""
    
    def remove_component(self, component_id: str):
        """Remove component and its connections."""
```

### FailureSimulator

Failure impact simulation.

```python
class FailureSimulator:
    def __init__(
        self,
        cascade_threshold: float = 0.5,
        cascade_probability: float = 0.7,
        max_cascade_depth: int = 5,
        seed: Optional[int] = None,
    ):
        """Initialize simulator with cascade settings."""
    
    def simulate_failure(
        self,
        graph: SimulationGraph,
        component_id: str,
        enable_cascade: bool = False,
    ) -> FailureResult:
        """Simulate single component failure."""
    
    def simulate_batch_failure(
        self,
        graph: SimulationGraph,
        component_ids: List[str],
        enable_cascade: bool = False,
    ) -> FailureResult:
        """Simulate multiple simultaneous failures."""
    
    def simulate_all_failures(
        self,
        graph: SimulationGraph,
        component_types: Optional[List[ComponentType]] = None,
        enable_cascade: bool = False,
    ) -> BatchSimulationResult:
        """Simulate failure of every component."""
    
    def simulate_attack(
        self,
        graph: SimulationGraph,
        strategy: AttackStrategy,
        count: int = 1,
        enable_cascade: bool = False,
        incremental: bool = True,
    ) -> FailureResult:
        """Simulate targeted attack."""
```

### FailureResult

Result of failure simulation.

```python
@dataclass
class FailureResult:
    simulation_id: str
    primary_failures: List[str]
    cascade_failures: List[Tuple[str, int]]  # (component, depth)
    impact: ImpactMetrics
    affected_paths: int
    affected_components: List[str]
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""

@dataclass
class ImpactMetrics:
    impact_score: float
    reachability_loss: float
    fragmentation: float
    cascade_extent: float
    paths_lost: int
    total_affected: int
```

### EventSimulator

Discrete event simulation.

```python
class EventSimulator:
    def __init__(self, seed: Optional[int] = None):
        """Initialize event simulator."""
    
    def simulate(
        self,
        graph: SimulationGraph,
        duration_ms: float = 10000,
        message_rate: float = 100,
        qos: QoSLevel = QoSLevel.AT_LEAST_ONCE,
        failure_schedule: Optional[List[Dict]] = None,
    ) -> SimulationResult:
        """Run event-driven simulation."""
    
    def simulate_load_test(
        self,
        graph: SimulationGraph,
        duration_ms: float = 30000,
        initial_rate: float = 10,
        peak_rate: float = 500,
        ramp_time_ms: float = 10000,
        qos: QoSLevel = QoSLevel.AT_LEAST_ONCE,
    ) -> SimulationResult:
        """Run load testing simulation."""
    
    def simulate_chaos(
        self,
        graph: SimulationGraph,
        duration_ms: float = 30000,
        message_rate: float = 100,
        failure_probability: float = 0.01,
        recovery_probability: float = 0.1,
        qos: QoSLevel = QoSLevel.AT_LEAST_ONCE,
    ) -> SimulationResult:
        """Run chaos engineering simulation."""
```

---

## Validation Module

Statistical validation.

```python
from src.validation import (
    # Pipeline
    ValidationPipeline,
    PipelineResult,
    
    # Analyzer
    GraphAnalyzer,
    
    # Validator
    Validator,
    ValidationResult,
    ValidationStatus,
    ValidationTargets,
    
    # Metrics
    CorrelationMetrics,
    ConfusionMatrix,
    RankingMetrics,
)
```

### ValidationPipeline

Integrated validation pipeline.

```python
class ValidationPipeline:
    def __init__(
        self,
        spearman_target: float = 0.70,
        f1_target: float = 0.90,
        cascade_threshold: float = 0.5,
        cascade_probability: float = 0.7,
        seed: Optional[int] = None,
    ):
        """Initialize pipeline with targets."""
    
    def run(
        self,
        graph: SimulationGraph,
        analysis_method: str = "composite",
        component_types: Optional[List[ComponentType]] = None,
        enable_cascade: bool = True,
    ) -> PipelineResult:
        """Run complete validation pipeline."""
    
    def compare_methods(
        self,
        graph: SimulationGraph,
        methods: Optional[List[str]] = None,
        enable_cascade: bool = True,
    ) -> Dict[str, PipelineResult]:
        """Compare multiple analysis methods."""
```

### PipelineResult

Complete pipeline result.

```python
@dataclass
class PipelineResult:
    timestamp: datetime
    n_components: int
    n_connections: int
    n_paths: int
    predicted_scores: Dict[str, float]
    analysis_method: str
    analysis_time_ms: float
    actual_impacts: Dict[str, float]
    simulation_time_ms: float
    validation: ValidationResult
    validation_time_ms: float
    total_time_ms: float
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
```

### GraphAnalyzer

Graph analysis and metrics.

```python
class GraphAnalyzer:
    def __init__(self, graph: SimulationGraph):
        """Initialize with graph."""
    
    def betweenness_centrality(self) -> Dict[str, float]:
        """Calculate betweenness centrality."""
    
    def degree_centrality(self) -> Dict[str, float]:
        """Calculate degree centrality."""
    
    def pagerank(self, damping: float = 0.85) -> Dict[str, float]:
        """Calculate PageRank."""
    
    def message_path_centrality(self) -> Dict[str, float]:
        """Calculate message path centrality."""
    
    def articulation_points(self) -> Set[str]:
        """Find articulation points."""
    
    def composite_score(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Calculate composite criticality score."""
    
    def analyze_all(self) -> Dict[str, Dict[str, float]]:
        """Compute all metrics."""
```

### Validator

Statistical validation.

```python
class Validator:
    def __init__(
        self,
        targets: Optional[ValidationTargets] = None,
        critical_threshold_percentile: float = 80,
        seed: Optional[int] = None,
    ):
        """Initialize with validation targets."""
    
    def validate(
        self,
        predicted: Dict[str, float],
        actual: Dict[str, float],
        component_types: Optional[Dict[str, str]] = None,
    ) -> ValidationResult:
        """Validate predictions against actuals."""
    
    def validate_with_bootstrap(
        self,
        predicted: Dict[str, float],
        actual: Dict[str, float],
        n_iterations: int = 1000,
        confidence: float = 0.95,
    ) -> ValidationResult:
        """Validate with bootstrap confidence intervals."""
```

### ValidationResult

Validation results.

```python
@dataclass
class ValidationResult:
    status: ValidationStatus
    total_components: int
    correlation: CorrelationMetrics
    classification: ConfusionMatrix
    ranking: RankingMetrics
    targets: ValidationTargets
    achieved: Dict[str, Tuple[float, MetricStatus]]
    component_validations: List[ComponentValidation]
    false_positives: List[str]
    false_negatives: List[str]
    bootstrap_results: Optional[List[BootstrapResult]]
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
    
    def summary(self) -> Dict:
        """Quick summary statistics."""
```

---

## Analysis Module

Advanced analysis capabilities.

```python
from src.analysis import (
    # Classification
    BoxPlotClassifier,
    ClassificationResult,
    CriticalityLevel,
    
    # Neo4j GDS
    GDSClient,
    
    # Anti-patterns
    AntiPatternDetector,
    AntiPattern,
)
```

### BoxPlotClassifier

Box-plot statistical classification.

```python
class BoxPlotClassifier:
    def __init__(self, k_factor: float = 1.5):
        """Initialize with outlier factor."""
    
    def classify(
        self,
        items: List[Dict],
        metric_name: str = "score",
    ) -> ClassificationResult:
        """Classify items by score."""
```

### ClassificationResult

Classification results.

```python
@dataclass
class ClassificationResult:
    items: List[ClassifiedItem]
    stats: BoxPlotStats
    by_level: Dict[str, List[ClassifiedItem]]
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""

@dataclass
class ClassifiedItem:
    id: str
    type: str
    score: float
    level: CriticalityLevel

@dataclass
class BoxPlotStats:
    q1: float
    median: float
    q3: float
    iqr: float
    lower_fence: float
    upper_fence: float
```

### GDSClient

Neo4j Graph Data Science integration.

```python
class GDSClient:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        """Initialize GDS client."""
    
    def create_depends_on_projection(self, name: str) -> Dict:
        """Create graph projection for analysis."""
    
    def betweenness_centrality(self, projection: str) -> Dict[str, float]:
        """Run betweenness centrality algorithm."""
    
    def pagerank(self, projection: str) -> Dict[str, float]:
        """Run PageRank algorithm."""
    
    def louvain_community(self, projection: str) -> Dict[str, int]:
        """Run Louvain community detection."""
    
    def drop_projection(self, name: str):
        """Drop graph projection."""
    
    def close(self):
        """Close connection."""
```

---

## Visualization Module

Graph visualization and dashboards.

```python
from src.visualization import (
    GraphRenderer,
    DashboardGenerator,
)
```

### GraphRenderer

Graph visualization.

```python
class GraphRenderer:
    def render(
        self,
        graph: SimulationGraph,
        criticality: Dict[str, Any],
        layout: str = "physics",
        show_labels: bool = True,
        edge_arrows: bool = True,
        physics_enabled: bool = True,
        width: str = "100%",
        height: str = "600px",
    ) -> str:
        """Render interactive network graph."""
    
    def render_multi_layer(
        self,
        graph: SimulationGraph,
        criticality: Dict[str, Any],
        layer_spacing: int = 150,
        show_dependencies: bool = True,
        show_labels: bool = True,
    ) -> str:
        """Render multi-layer architecture view."""
```

### DashboardGenerator

Comprehensive dashboard.

```python
class DashboardGenerator:
    def generate(
        self,
        graph: SimulationGraph,
        criticality: Optional[Dict[str, Any]] = None,
        validation: Optional[Dict] = None,
        simulation: Optional[Dict] = None,
        title: str = "Criticality Analysis Dashboard",
        theme: str = "light",
        show_network: bool = True,
        show_table: bool = True,
        show_charts: bool = True,
    ) -> str:
        """Generate comprehensive HTML dashboard."""
```

---

## Navigation

- **Previous:** [← Visualization](visualization.md)
- **Next:** [CLI Reference →](cli-reference.md)
