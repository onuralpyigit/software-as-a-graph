# Software Design Description

## Software-as-a-Graph
### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 1.0**  
**January 2026**

Istanbul Technical University  
Computer Engineering Department

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Overview](#2-system-overview)
3. [System Architecture](#3-system-architecture)
4. [Data Design](#4-data-design)
5. [Component Design](#5-component-design)
6. [Interface Design](#6-interface-design)
7. [Algorithmic Design](#7-algorithmic-design)
8. [Database Design](#8-database-design)
9. [User Interface Design](#9-user-interface-design)
10. [Appendices](#appendices)

---

## 1. Introduction

### 1.1 Purpose

This Software Design Description (SDD) document provides a detailed technical design for the Software-as-a-Graph framework. It describes the system architecture, component designs, data structures, algorithms, and interfaces that implement the requirements specified in the Software Requirements Specification (SRS).

### 1.2 Scope

This document covers the complete design of the Software-as-a-Graph framework, including:
- System architecture and module decomposition
- Data structures and database schema
- Component interfaces and interactions
- Algorithm specifications
- User interface design

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| CLI | Command Line Interface |
| DAO | Data Access Object |
| DI | Dependency Injection |
| DTO | Data Transfer Object |
| GDS | Graph Data Science (Neo4j library) |
| IQR | Interquartile Range |
| OOP | Object-Oriented Programming |
| SOLID | Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion |

### 1.4 References

- IEEE 1016-2009: IEEE Standard for Information Technology—Systems Design—Software Design Descriptions
- Software Requirements Specification (SRS) for Software-as-a-Graph v1.0
- Neo4j Developer Documentation
- NetworkX Documentation

### 1.5 Design Overview

The Software-as-a-Graph framework follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│              (CLI Tools, HTML Dashboards)                    │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                         │
│        (Pipeline Orchestration, Report Generation)           │
├─────────────────────────────────────────────────────────────┤
│                     Service Layer                            │
│   (Analysis, Simulation, Validation, Visualization)          │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer                              │
│        (Graph Model, Import/Export, Neo4j Client)            │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                       │
│              (Neo4j Database, File System)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. System Overview

### 2.1 System Context

```
                              ┌──────────────────┐
                              │   System Admin   │
                              └────────┬─────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │                                      │
   ┌────────────┐   │     Software-as-a-Graph Framework   │   ┌────────────┐
   │   JSON     │──▶│                                      │──▶│   HTML     │
   │  Topology  │   │  ┌────────┐ ┌──────┐ ┌───────────┐  │   │ Dashboard  │
   │   Input    │   │  │ Import │→│Analyze│→│ Visualize │  │   │  Output    │
   └────────────┘   │  └────────┘ └──────┘ └───────────┘  │   └────────────┘
                    │                                      │
                    └──────────────────┬──────────────────┘
                                       │
                              ┌────────▼─────────┐
                              │    Neo4j DB      │
                              └──────────────────┘
```

### 2.2 Design Constraints

| Constraint | Description |
|------------|-------------|
| Database | Neo4j 5.x required for graph storage |
| Runtime | Python 3.9+ for analysis modules |
| Memory | Graph size limited by available RAM |
| Algorithms | NetworkX for graph computations |

### 2.3 Design Principles

The system adheres to these design principles:

1. **Separation of Concerns**: Each module handles a specific responsibility
2. **Single Responsibility**: Classes have one reason to change
3. **Open/Closed**: Open for extension, closed for modification
4. **Dependency Inversion**: High-level modules don't depend on low-level modules
5. **Interface Segregation**: Small, focused interfaces
6. **Composition over Inheritance**: Prefer composition for flexibility

---

## 3. System Architecture

### 3.1 Architectural Style

The system uses a **Layered Architecture** with **Facade Pattern** for module access:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │ run.py      │ │analyze_     │ │simulate_    │ │visualize_   │   │
│  │ (Pipeline)  │ │graph.py     │ │graph.py     │ │graph.py     │   │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘   │
└─────────┼───────────────┼───────────────┼───────────────┼──────────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         SERVICE LAYER                                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │
│  │ GraphAnalyzer   │ │ Simulator       │ │ GraphVisualizer │        │
│  │ (Facade)        │ │ (Facade)        │ │ (Facade)        │        │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘        │
│           │                   │                   │                  │
│  ┌────────▼────────┐ ┌────────▼────────┐ ┌───────▼─────────┐        │
│  │StructuralAnalyzer│ │FailureSimulator │ │DashboardGenerator│       │
│  │ QualityAnalyzer  │ │ EventSimulator  │ │ ChartGenerator   │       │
│  │ ProblemDetector  │ │SimulationGraph  │ │ NetworkRenderer  │       │
│  └────────┬────────┘ └────────┬────────┘ └───────┬─────────┘        │
└───────────┼───────────────────┼──────────────────┼──────────────────┘
            │                   │                  │
            ▼                   ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │
│  │ GraphImporter   │ │ GraphExporter   │ │ GraphData       │        │
│  │                 │ │                 │ │ ComponentData   │        │
│  │                 │ │                 │ │ EdgeData        │        │
│  └────────┬────────┘ └────────┬────────┘ └─────────────────┘        │
└───────────┼───────────────────┼─────────────────────────────────────┘
            │                   │
            ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INFRASTRUCTURE LAYER                            │
│  ┌─────────────────────────────┐ ┌─────────────────────────────┐    │
│  │        Neo4j Database       │ │       File System           │    │
│  │   (Bolt Protocol Driver)    │ │   (JSON, HTML, CSV)         │    │
│  └─────────────────────────────┘ └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Decomposition

```
software-as-a-graph/
├── src/
│   ├── core/                      # Data Layer
│   │   ├── __init__.py
│   │   ├── graph_model.py         # Domain entities
│   │   ├── graph_generator.py     # Synthetic data generation
│   │   ├── graph_importer.py      # Neo4j import
│   │   └── graph_exporter.py      # Neo4j export & data access
│   │
│   ├── analysis/                  # Analysis Service
│   │   ├── __init__.py
│   │   ├── layers.py              # Layer definitions
│   │   ├── classifier.py          # Box-plot classification
│   │   ├── weight_calculator.py   # AHP weight computation
│   │   ├── metrics.py             # Metric data structures
│   │   ├── structural_analyzer.py # Graph metrics
│   │   ├── quality_analyzer.py    # RMAV scoring
│   │   ├── problem_detector.py    # Issue detection
│   │   └── analyzer.py            # Facade orchestrator
│   │
│   ├── simulation/                # Simulation Service
│   │   ├── __init__.py
│   │   ├── simulation_graph.py    # In-memory graph model
│   │   ├── event_simulator.py     # Message flow simulation
│   │   ├── failure_simulator.py   # Failure injection
│   │   └── simulator.py           # Facade orchestrator
│   │
│   ├── validation/                # Validation Service
│   │   ├── __init__.py
│   │   ├── metrics.py             # Statistical metrics
│   │   └── validator.py           # Comparison logic
│   │
│   └── visualization/             # Visualization Service
│       ├── __init__.py
│       ├── charts.py              # Chart generation
│       ├── dashboard.py           # HTML dashboard
│       ├── display.py             # Terminal output
│       └── visualizer.py          # Facade orchestrator
│
├── generate_graph.py              # CLI: Generate data
├── import_graph.py                # CLI: Import to Neo4j
├── analyze_graph.py               # CLI: Run analysis
├── simulate_graph.py              # CLI: Run simulation
├── validate_graph.py              # CLI: Run validation
├── visualize_graph.py             # CLI: Generate dashboard
├── run.py                         # CLI: Full pipeline
└── benchmark.py                   # CLI: Benchmark suite
```

### 3.3 Module Dependencies

```
┌──────────────────────────────────────────────────────────────┐
│                    CLI Scripts (run.py, etc.)                 │
└───────────────────────────┬──────────────────────────────────┘
                            │ uses
                            ▼
┌──────────────┬──────────────┬──────────────┬─────────────────┐
│   analysis   │  simulation  │  validation  │  visualization  │
│   (Facade)   │   (Facade)   │   (Facade)   │    (Facade)     │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬────────┘
       │              │              │                │
       │              │              │                │
       └──────────────┴──────────────┴────────────────┘
                            │ uses
                            ▼
               ┌────────────────────────────┐
               │         src/core           │
               │  (GraphImporter/Exporter)  │
               └────────────┬───────────────┘
                            │ uses
                            ▼
               ┌────────────────────────────┐
               │      Neo4j / NetworkX      │
               └────────────────────────────┘
```

### 3.4 Design Patterns Used

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Facade** | `GraphAnalyzer`, `Simulator`, `GraphVisualizer` | Simplified interface to subsystems |
| **Strategy** | `BoxPlotClassifier`, `AHPProcessor` | Interchangeable algorithms |
| **Data Transfer Object** | `GraphData`, `ComponentData`, `EdgeData` | Data transfer between layers |
| **Builder** | `DashboardGenerator` | Step-by-step object construction |
| **Factory** | `generate_graph()` | Object creation abstraction |
| **Context Manager** | All database clients | Resource management |
| **Observer** | Event simulation callbacks | Event notification |

---

## 4. Data Design

### 4.1 Data Structures Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Domain Model Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Application │  │   Broker    │  │    Topic    │              │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤              │
│  │ id: str     │  │ id: str     │  │ id: str     │              │
│  │ name: str   │  │ name: str   │  │ name: str   │              │
│  │ role: str   │  │ weight: f   │  │ qos: QoS    │              │
│  │ app_type: s │  └─────────────┘  │ size: int   │              │
│  │ version: s  │                   │ weight: f   │              │
│  │ weight: f   │  ┌─────────────┐  └─────────────┘              │
│  └─────────────┘  │    Node     │                               │
│                   ├─────────────┤  ┌─────────────┐              │
│  ┌─────────────┐  │ id: str     │  │   Library   │              │
│  │  QoSPolicy  │  │ name: str   │  ├─────────────┤              │
│  ├─────────────┤  │ weight: f   │  │ id: str     │              │
│  │reliability  │  └─────────────┘  │ name: str   │              │
│  │durability   │                   │ version: str│              │
│  │priority     │                   │ weight: f   │              │
│  │message_size │                   └─────────────┘              │
│  └─────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Data Transfer Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐  ┌───────────────────┐                   │
│  │   ComponentData   │  │     EdgeData      │                   │
│  ├───────────────────┤  ├───────────────────┤                   │
│  │ id: str           │  │ source_id: str    │                   │
│  │ component_type: s │  │ target_id: str    │                   │
│  │ weight: float     │  │ dependency_type: s│                   │
│  │ properties: dict  │  │ weight: float     │                   │
│  └───────────────────┘  │ properties: dict  │                   │
│                         └───────────────────┘                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      GraphData                           │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ components: List[ComponentData]                          │    │
│  │ edges: List[EdgeData]                                    │    │
│  │ + to_dict() → Dict                                       │    │
│  │ + get_components_by_type(type) → List[ComponentData]     │    │
│  │ + get_edges_by_type(type) → List[EdgeData]               │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Metrics Data Structures

```python
@dataclass
class StructuralMetrics:
    """Raw topological metrics for a graph component."""
    id: str
    name: str
    type: str  # Application, Node, Broker, Topic
    
    # Centrality Metrics
    pagerank: float = 0.0
    reverse_pagerank: float = 0.0
    betweenness: float = 0.0
    closeness: float = 0.0
    eigenvector: float = 0.0
    
    # Degree Metrics
    degree: float = 0.0
    in_degree: float = 0.0
    out_degree: float = 0.0
    in_degree_raw: int = 0
    out_degree_raw: int = 0
    
    # Resilience Metrics
    clustering_coefficient: float = 0.0
    is_articulation_point: bool = False
    is_isolated: bool = False
    bridge_ratio: float = 0.0
    
    # Weights
    weight: float = 1.0
    dependency_weight_in: float = 0.0
    dependency_weight_out: float = 0.0
```

```python
@dataclass
class QualityScores:
    """RMAV quality scores for a component."""
    reliability: float = 0.0      # R(v)
    maintainability: float = 0.0  # M(v)
    availability: float = 0.0     # A(v)
    vulnerability: float = 0.0    # V(v)
    overall: float = 0.0          # Q(v)

@dataclass
class QualityLevels:
    """Criticality levels for each dimension."""
    reliability: CriticalityLevel = CriticalityLevel.MINIMAL
    maintainability: CriticalityLevel = CriticalityLevel.MINIMAL
    availability: CriticalityLevel = CriticalityLevel.MINIMAL
    vulnerability: CriticalityLevel = CriticalityLevel.MINIMAL
    overall: CriticalityLevel = CriticalityLevel.MINIMAL
```

### 4.3 Analysis Result Structures

```python
@dataclass
class StructuralAnalysisResult:
    """Container for structural analysis results."""
    layer: AnalysisLayer
    components: Dict[str, StructuralMetrics]
    edges: Dict[Tuple[str, str], EdgeMetrics]
    graph_summary: GraphSummary

@dataclass
class QualityAnalysisResult:
    """Container for quality analysis results."""
    timestamp: str
    layer: str
    context: str
    components: List[ComponentQuality]
    edges: List[EdgeQuality]
    classification_summary: ClassificationSummary
    weights: QualityWeights

@dataclass
class LayerAnalysisResult:
    """Complete analysis result for a single layer."""
    layer: AnalysisLayer
    structural: StructuralAnalysisResult
    quality: QualityAnalysisResult
    problems: List[DetectedProblem]
    timestamp: str
```

### 4.4 Simulation Data Structures

```python
@dataclass
class ImpactMetrics:
    """Impact metrics from failure simulation."""
    # Reachability
    initial_paths: int = 0
    remaining_paths: int = 0
    reachability_loss: float = 0.0
    
    # Infrastructure
    initial_components: int = 0
    failed_components: int = 0
    fragmentation: float = 0.0
    
    # Throughput
    throughput_loss: float = 0.0
    
    # Cascade
    cascade_count: int = 0
    cascade_depth: int = 0
    cascade_by_type: Dict[str, int] = field(default_factory=dict)
    
    @property
    def composite_impact(self) -> float:
        """I(v) = w_r×reachability + w_f×fragmentation + w_t×throughput"""
        w = self.impact_weights
        return (w["reachability"] * self.reachability_loss +
                w["fragmentation"] * self.fragmentation +
                w["throughput"] * self.throughput_loss)

@dataclass
class FailureResult:
    """Result of a failure simulation."""
    target_id: str
    target_type: str
    scenario: str
    impact: ImpactMetrics
    cascaded_failures: List[str]
    cascade_sequence: List[CascadeEvent]
    layer_impacts: Dict[str, float]
```

### 4.5 Validation Data Structures

```python
@dataclass
class CorrelationMetrics:
    """Correlation coefficients between predicted and actual."""
    spearman: float = 0.0
    spearman_p: float = 1.0
    pearson: float = 0.0
    kendall: float = 0.0

@dataclass
class ClassificationMetrics:
    """Binary classification metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

@dataclass
class RankingMetrics:
    """Ranking quality metrics."""
    top_5_overlap: float = 0.0
    top_10_overlap: float = 0.0
    ndcg_10: float = 0.0

@dataclass
class ValidationGroupResult:
    """Validation result for a group (layer)."""
    group_name: str
    sample_size: int
    correlation: CorrelationMetrics
    error: ErrorMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    passed: bool
    targets: ValidationTargets
    components: List[ComponentComparison]
```

---

## 5. Component Design

### 5.1 Core Module (`src/core`)

#### 5.1.1 GraphImporter

**Purpose**: Import system topology into Neo4j with dependency derivation.

```python
class GraphImporter:
    """Imports graph data into Neo4j with post-processing."""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
    
    def import_graph(self, data: Dict, clear: bool = False) -> Dict[str, int]:
        """
        Orchestrates full import and derivation process.
        
        Steps:
        1. Clear database (optional)
        2. Import entities (Nodes, Brokers, Apps, Topics, Libraries)
        3. Import structural relationships
        4. Calculate intrinsic weights from QoS
        5. Derive DEPENDS_ON relationships
        6. Calculate component criticality weights
        """
        
    def _calculate_intrinsic_weights(self) -> None:
        """Calculate weights from Topic QoS settings."""
        
    def _derive_dependencies(self) -> Dict[str, int]:
        """
        Derive DEPENDS_ON relationships:
        - app_to_app: App subscribes to Topic, other App publishes
        - node_to_node: Apps on nodes have dependencies
        - app_to_broker: App uses Topic routed by Broker
        - node_to_broker: Node hosts App that uses Broker
        """
```

**Sequence Diagram: Import Process**

```
┌────────┐     ┌────────────┐     ┌────────┐
│  CLI   │     │GraphImporter│     │ Neo4j  │
└───┬────┘     └─────┬──────┘     └───┬────┘
    │                │                 │
    │ import_graph() │                 │
    │───────────────▶│                 │
    │                │  MATCH DELETE   │
    │                │────────────────▶│
    │                │                 │
    │                │  MERGE entities │
    │                │────────────────▶│
    │                │                 │
    │                │  CREATE rels    │
    │                │────────────────▶│
    │                │                 │
    │                │ calc weights    │
    │                │────────────────▶│
    │                │                 │
    │                │ derive DEPENDS  │
    │                │────────────────▶│
    │                │                 │
    │   stats        │                 │
    │◀───────────────│                 │
```

#### 5.1.2 GraphExporter

**Purpose**: Export graph data from Neo4j for analysis.

```python
class GraphExporter:
    """Neo4j client for graph data retrieval and export."""
    
    def get_graph_data(
        self,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None
    ) -> GraphData:
        """Retrieve graph data with optional type filtering."""
        
    def get_layer_data(self, layer: str) -> GraphData:
        """Retrieve graph data for a specific architectural layer."""
        
    def export_graph_json(self) -> Dict[str, Any]:
        """Export graph in format compatible with GraphImporter."""
```

#### 5.1.3 Graph Model

**Purpose**: Domain entity definitions.

```python
class VertexType(str, Enum):
    APPLICATION = "Application"
    BROKER = "Broker"
    TOPIC = "Topic"
    NODE = "Node"
    LIBRARY = "Library"

class EdgeType(str, Enum):
    RUNS_ON = "RUNS_ON"
    ROUTES = "ROUTES"
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    CONNECTS_TO = "CONNECTS_TO"
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"

class DependencyType(str, Enum):
    APP_TO_APP = "app_to_app"
    NODE_TO_NODE = "node_to_node"
    APP_TO_BROKER = "app_to_broker"
    NODE_TO_BROKER = "node_to_broker"

@dataclass
class QoSPolicy:
    """Quality of Service attributes for Topics."""
    reliability: str = "BEST_EFFORT"  # RELIABLE, BEST_EFFORT
    durability: str = "VOLATILE"       # PERSISTENT, TRANSIENT, VOLATILE
    transport_priority: str = "MEDIUM" # URGENT, HIGH, MEDIUM, LOW
    message_size: int = 256            # bytes
    
    def calculate_weight(self) -> float:
        """
        Calculate Topic weight from QoS settings.
        W = S_reliability + S_durability + S_priority + S_size
        """
```

### 5.2 Analysis Module (`src/analysis`)

#### 5.2.1 StructuralAnalyzer

**Purpose**: Compute topological metrics using NetworkX.

```python
class StructuralAnalyzer:
    """Analyzes graph structure to compute topological metrics."""
    
    def __init__(self, damping_factor: float = 0.85):
        self.damping_factor = damping_factor
    
    def analyze(
        self,
        graph_data: GraphData,
        layer: AnalysisLayer = AnalysisLayer.SYSTEM
    ) -> StructuralAnalysisResult:
        """
        Compute structural metrics for the graph.
        
        Pipeline:
        1. Build NetworkX DiGraph from GraphData
        2. Filter to layer-specific components/edges
        3. Compute centrality metrics (PageRank, Betweenness, etc.)
        4. Compute degree metrics
        5. Identify articulation points and bridges
        6. Normalize all metrics to [0, 1]
        """
        
    def _compute_centrality_metrics(self, G: nx.DiGraph) -> Dict[str, Dict]:
        """Compute PageRank, Betweenness, Eigenvector, Closeness."""
        
    def _identify_critical_structures(self, G: nx.DiGraph) -> Tuple[Set, Set]:
        """Identify articulation points and bridge edges."""
```

**Class Diagram: Analysis Components**

```
┌─────────────────────────────────────────────────────────────────────┐
│                          GraphAnalyzer                               │
│                            (Facade)                                  │
├─────────────────────────────────────────────────────────────────────┤
│ - structural: StructuralAnalyzer                                     │
│ - quality: QualityAnalyzer                                          │
│ - detector: ProblemDetector                                         │
│ - client: GraphExporter                                             │
├─────────────────────────────────────────────────────────────────────┤
│ + analyze_layer(layer) → LayerAnalysisResult                        │
│ + analyze_all_layers() → MultiLayerAnalysisResult                   │
│ + export_results(results, path)                                     │
└───────────────────┬─────────────────────────────────────────────────┘
                    │ uses
        ┌───────────┼───────────┬───────────────┐
        ▼           ▼           ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌──────────────┐
│Structural     │ │Quality        │ │Problem        │ │BoxPlot       │
│Analyzer       │ │Analyzer       │ │Detector       │ │Classifier    │
├───────────────┤ ├───────────────┤ ├───────────────┤ ├──────────────┤
│+analyze()     │ │+analyze()     │ │+detect()      │ │+classify()   │
│-compute_      │ │-compute_      │ │-detect_spof() │ │-compute_     │
│ centrality()  │ │ scores()      │ │-detect_hub()  │ │ quartiles()  │
│-identify_     │ │-classify_     │ │-detect_       │ │-assign_      │
│ bridges()     │ │ components()  │ │ bottleneck()  │ │ levels()     │
└───────────────┘ └───────────────┘ └───────────────┘ └──────────────┘
```

#### 5.2.2 QualityAnalyzer

**Purpose**: Compute RMAV quality scores.

```python
class QualityAnalyzer:
    """Computes quality scores and classifications."""
    
    def __init__(
        self,
        k_factor: float = 1.5,
        weights: Optional[QualityWeights] = None,
        use_ahp: bool = False
    ):
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        if use_ahp:
            self.weights = AHPProcessor().compute_weights()
        else:
            self.weights = weights or QualityWeights()
    
    def analyze(
        self,
        structural_result: StructuralAnalysisResult,
        context: Optional[str] = None
    ) -> QualityAnalysisResult:
        """
        Compute quality scores and classifications.
        
        Steps:
        1. Normalize metrics across components
        2. Compute R, M, A, V scores using weighted formulas
        3. Classify using box-plot method
        4. Analyze edges
        5. Build summary statistics
        """
        
    def _compute_scores(self, m: StructuralMetrics, norm: Dict) -> QualityScores:
        """
        Compute RMAV scores for a component.
        
        R(v) = w₁×PR + w₂×RPR + w₃×ID
        M(v) = w₁×BT + w₂×DG + w₃×(1-CC)
        A(v) = w₁×AP + w₂×BR + w₃×Importance
        V(v) = w₁×EV + w₂×CL + w₃×ID
        Q(v) = w_R×R + w_M×M + w_A×A + w_V×V
        """
```

#### 5.2.3 BoxPlotClassifier

**Purpose**: Adaptive classification using statistical quartiles.

```python
class CriticalityLevel(Enum):
    """Component criticality levels."""
    CRITICAL = 5  # > Q3 + 1.5×IQR
    HIGH = 4      # > Q3
    MEDIUM = 3    # > Median
    LOW = 2       # > Q1
    MINIMAL = 1   # ≤ Q1

class BoxPlotClassifier:
    """Adaptive classification using box-plot statistics."""
    
    def __init__(self, k_factor: float = 1.5):
        self.k_factor = k_factor
    
    def classify(
        self,
        data: List[Dict],
        metric_name: str = "score"
    ) -> ClassificationResult:
        """
        Classify items using box-plot thresholds.
        
        Thresholds:
        - CRITICAL: score > Q3 + k×IQR
        - HIGH: score > Q3
        - MEDIUM: score > Median
        - LOW: score > Q1
        - MINIMAL: score ≤ Q1
        """
        
    def _compute_quartiles(self, values: List[float]) -> BoxPlotStats:
        """Compute Q1, Median, Q3, IQR."""
```

#### 5.2.4 AHPProcessor

**Purpose**: Compute weights using Analytic Hierarchy Process.

```python
@dataclass
class AHPMatrices:
    """Pairwise comparison matrices for AHP."""
    criteria_reliability: List[List[float]]     # PR, RPR, ID
    criteria_maintainability: List[List[float]] # BT, DG, CC
    criteria_availability: List[List[float]]    # AP, BR, IM
    criteria_vulnerability: List[List[float]]   # EV, CL, ID
    criteria_overall: List[List[float]]         # R, M, A, V

class AHPProcessor:
    """Calculates weights from pairwise comparison matrices."""
    
    def __init__(self, matrices: AHPMatrices = None):
        self.matrices = matrices or AHPMatrices()
    
    def compute_weights(self) -> QualityWeights:
        """
        Compute all weights using geometric mean method.
        
        For each matrix:
        1. Compute geometric mean of each row
        2. Normalize to get priority vector
        3. Validate consistency (CR < 0.10)
        """
        
    def _calculate_priority_vector(self, matrix: List[List[float]]) -> List[float]:
        """
        Geometric mean method:
        GM_i = (∏ a_ij)^(1/n)
        w_i = GM_i / Σ GM_j
        """
        
    def _check_consistency(self, matrix: List[List[float]], weights: List[float]) -> float:
        """
        Compute Consistency Ratio:
        CR = CI / RI
        CI = (λ_max - n) / (n - 1)
        """
```

### 5.3 Simulation Module (`src/simulation`)

#### 5.3.1 SimulationGraph

**Purpose**: In-memory graph model for simulation.

```python
class ComponentState(Enum):
    ACTIVE = "active"
    FAILED = "failed"
    DEGRADED = "degraded"

class SimulationGraph:
    """In-memory graph for simulation operations."""
    
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.topics: Dict[str, TopicInfo] = {}
        self._routing: Dict[str, List[str]] = {}  # topic -> brokers
        self._hosting: Dict[str, List[str]] = {}  # node -> components
        self._pub_sub: Dict[str, Tuple[Set, Set]] = {}  # topic -> (pubs, subs)
    
    def load_from_neo4j(self, uri: str, user: str, password: str) -> None:
        """Load graph data from Neo4j."""
        
    def fail_component(self, component_id: str) -> None:
        """Mark component as failed."""
        
    def reset(self) -> None:
        """Reset all components to ACTIVE state."""
        
    def get_pub_sub_paths(self, active_only: bool = True) -> List[Tuple]:
        """Get all publisher-subscriber paths."""
        
    def get_components_by_layer(self, layer: str) -> List[str]:
        """Get component IDs for a specific layer."""
```

#### 5.3.2 FailureSimulator

**Purpose**: Simulate failures with cascade propagation.

```python
class FailureMode(Enum):
    CRASH = "crash"         # Complete failure
    DEGRADED = "degraded"   # Reduced capacity
    PARTITION = "partition" # Network isolation
    OVERLOAD = "overload"   # Resource exhaustion

class CascadeRule(Enum):
    PHYSICAL = "physical"   # Node → hosted components
    LOGICAL = "logical"     # Broker → topics
    NETWORK = "network"     # Partition propagation
    ALL = "all"

class FailureSimulator:
    """Simulates component failures and cascade propagation."""
    
    def __init__(self, graph: SimulationGraph):
        self.graph = graph
    
    def simulate(self, scenario: FailureScenario) -> FailureResult:
        """
        Run failure simulation.
        
        Steps:
        1. Capture baseline state
        2. Fail target component
        3. Propagate cascade according to rules
        4. Calculate impact metrics
        5. Return results
        """
        
    def simulate_exhaustive(
        self,
        layer: str = "system"
    ) -> List[FailureResult]:
        """Simulate failure for all components in layer."""
        
    def _propagate_cascade(
        self,
        scenario: FailureScenario,
        initial_target: str,
        failed_set: Set[str],
        cascade_sequence: List[CascadeEvent]
    ) -> int:
        """
        Propagate failure cascade.
        
        Physical Cascade: Node fails → hosted Apps/Brokers fail
        Logical Cascade: Broker fails → Topics unreachable
        Network Cascade: Partition propagation
        """
        
    def _calculate_impact(
        self,
        target_id: str,
        failed_set: Set[str]
    ) -> ImpactMetrics:
        """Calculate impact metrics after cascade."""
```

**Sequence Diagram: Failure Simulation**

```
┌────────┐   ┌───────────────┐   ┌────────────────┐
│  CLI   │   │FailureSimulator│   │SimulationGraph │
└───┬────┘   └───────┬───────┘   └───────┬────────┘
    │                │                    │
    │ simulate()     │                    │
    │───────────────▶│                    │
    │                │ reset()            │
    │                │───────────────────▶│
    │                │                    │
    │                │ fail_component()   │
    │                │───────────────────▶│
    │                │                    │
    │                │ propagate_cascade()│
    │                │───────────────────▶│
    │                │ ◀───────── (loop)  │
    │                │                    │
    │                │ get_pub_sub_paths()│
    │                │───────────────────▶│
    │                │                    │
    │ FailureResult  │                    │
    │◀───────────────│                    │
```

### 5.4 Validation Module (`src/validation`)

#### 5.4.1 Validator

**Purpose**: Compare predictions against ground truth.

```python
class Validator:
    """Statistical validation of predictions vs actual impact."""
    
    def __init__(
        self,
        targets: Optional[ValidationTargets] = None,
        critical_percentile: float = 75.0
    ):
        self.targets = targets or ValidationTargets()
        self.critical_percentile = critical_percentile
    
    def validate(
        self,
        predicted: Dict[str, float],  # Q(v) scores
        actual: Dict[str, float],     # I(v) impacts
        types: Optional[Dict[str, str]] = None
    ) -> ValidationGroupResult:
        """
        Compare predicted vs actual values.
        
        Steps:
        1. Align data by component ID
        2. Compute correlation metrics
        3. Compute error metrics
        4. Compute classification metrics
        5. Compute ranking metrics
        6. Determine pass/fail
        """
        
    def validate_layer(
        self,
        analysis_result: QualityAnalysisResult,
        simulation_results: List[FailureResult]
    ) -> ValidationGroupResult:
        """Validate a complete layer analysis."""
```

### 5.5 Visualization Module (`src/visualization`)

#### 5.5.1 DashboardGenerator

**Purpose**: Build HTML dashboards incrementally.

```python
class DashboardGenerator:
    """Generates responsive HTML dashboards."""
    
    def __init__(self, title: str):
        self.title = title
        self.sections: List[str] = []
        self.nav_links: List[NavLink] = []
        self.scripts: List[str] = []
    
    def start_section(self, title: str, anchor_id: str) -> None:
        """Begin a new dashboard section."""
        
    def add_kpis(self, kpis: Dict[str, Any], highlights: Dict = None) -> None:
        """Add KPI cards to current section."""
        
    def add_charts(self, charts: List[str]) -> None:
        """Add charts (pie, bar) to current section."""
        
    def add_table(self, headers: List[str], rows: List[List[str]], title: str) -> None:
        """Add data table to current section."""
        
    def add_network(self, nodes: List[Dict], edges: List[Dict]) -> None:
        """Add interactive network visualization."""
        
    def generate(self) -> str:
        """Generate complete HTML document."""
```

#### 5.5.2 GraphVisualizer

**Purpose**: Facade for visualization operations.

```python
class GraphVisualizer:
    """Orchestrates visualization generation."""
    
    def __init__(self, uri: str, user: str, password: str):
        self._analyzer = GraphAnalyzer(uri, user, password)
        self._simulator = Simulator(uri, user, password)
        self._validator = Validator()
    
    def generate_dashboard(
        self,
        output_file: str,
        layers: List[str] = None,
        include_network: bool = True,
        include_validation: bool = True
    ) -> str:
        """
        Generate comprehensive dashboard.
        
        Steps:
        1. Collect data for all layers
        2. Add overview section
        3. Add layer comparison section
        4. Add individual layer sections
        5. Write HTML output
        """
```

---

## 6. Interface Design

### 6.1 Module Interfaces

#### 6.1.1 Analysis Module Interface

```python
# Public API
def analyze_graph(
    layer: str = "system",
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    output: Optional[str] = None,
    use_ahp: bool = False
) -> Union[LayerAnalysisResult, MultiLayerAnalysisResult]:
    """
    Convenience function for quick graph analysis.
    
    Args:
        layer: Layer to analyze ("app", "infra", "system", "all")
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        output: Optional path to export results
        use_ahp: Use AHP-derived weights
        
    Returns:
        Analysis results for specified layer(s)
    """
```

#### 6.1.2 Simulation Module Interface

```python
# Public API
class Simulator:
    """Main simulation facade."""
    
    def run_failure_simulation(
        self,
        target_id: str,
        layer: str = "system",
        cascade_probability: float = 1.0
    ) -> FailureResult:
        """Simulate single component failure."""
        
    def run_failure_simulation_exhaustive(
        self,
        layer: str = "system",
        cascade_probability: float = 1.0
    ) -> List[FailureResult]:
        """Simulate failure for all components."""
        
    def generate_report(
        self,
        layers: List[str] = None,
        output_path: Optional[str] = None
    ) -> SimulationReport:
        """Generate comprehensive simulation report."""
```

### 6.2 CLI Interface

#### 6.2.1 Pipeline CLI (`run.py`)

```
Usage: python run.py [OPTIONS]

Options:
  --generate              Generate synthetic data
  --import               Import data to Neo4j
  --analyze              Run structural analysis
  --simulate             Run failure simulation
  --validate             Run validation
  --visualize            Generate dashboard
  --all                  Run complete pipeline
  
  --layer LAYER          Analysis layer (app, infra, mw-app, mw-infra, system)
  --layers LAYERS        Multiple layers (comma-separated)
  --scale SCALE          Data scale (tiny, small, medium, large, xlarge)
  
  --uri URI              Neo4j URI (default: bolt://localhost:7687)
  --user USER            Neo4j username (default: neo4j)
  --password PASS        Neo4j password (default: password)
  
  --input FILE           Input JSON file
  --output DIR           Output directory
  --open                 Open dashboard in browser
  
  -v, --verbose          Verbose output
  -q, --quiet            Quiet mode
```

#### 6.2.2 Analysis CLI (`analyze_graph.py`)

```
Usage: python analyze_graph.py [OPTIONS]

Options:
  --layer LAYER          Single layer to analyze
  --layers LAYERS        Multiple layers (comma-separated)
  --all                  Analyze all layers
  --use-ahp              Use AHP-derived weights
  --output FILE          Export results to JSON
  --show-problems        Display detected problems
```

### 6.3 Data Exchange Formats

#### 6.3.1 Input JSON Format

```json
{
  "nodes": [
    {"id": "node1", "name": "Server 1"}
  ],
  "brokers": [
    {"id": "broker1", "name": "Main Broker"}
  ],
  "applications": [
    {
      "id": "app1",
      "name": "Sensor Fusion",
      "role": "subscriber",
      "app_type": "processor",
      "version": "1.0.0",
      "criticality": "high"
    }
  ],
  "topics": [
    {
      "id": "topic1",
      "name": "/sensors/lidar",
      "size": 1024,
      "qos": {
        "reliability": "RELIABLE",
        "durability": "VOLATILE",
        "transport_priority": "HIGH"
      }
    }
  ],
  "libraries": [
    {"id": "lib1", "name": "Navigation Lib", "version": "2.0.0"}
  ],
  "relationships": {
    "runs_on": [{"source": "app1", "target": "node1"}],
    "routes": [{"source": "broker1", "target": "topic1"}],
    "publishes_to": [{"source": "app2", "target": "topic1"}],
    "subscribes_to": [{"source": "app1", "target": "topic1"}],
    "connects_to": [{"source": "node1", "target": "node2"}],
    "uses": [{"source": "app1", "target": "lib1"}]
  }
}
```

#### 6.3.2 Analysis Output JSON Format

```json
{
  "timestamp": "2026-01-27T10:30:00",
  "layer": "system",
  "components": [
    {
      "id": "app1",
      "name": "Sensor Fusion",
      "type": "Application",
      "scores": {
        "reliability": 0.82,
        "maintainability": 0.75,
        "availability": 0.90,
        "vulnerability": 0.68,
        "overall": 0.79
      },
      "levels": {
        "reliability": "HIGH",
        "maintainability": "MEDIUM",
        "availability": "CRITICAL",
        "vulnerability": "MEDIUM",
        "overall": "HIGH"
      }
    }
  ],
  "classification_summary": {
    "total_components": 48,
    "component_distribution": {
      "CRITICAL": 5,
      "HIGH": 8,
      "MEDIUM": 15,
      "LOW": 12,
      "MINIMAL": 8
    }
  }
}
```

---

## 7. Algorithmic Design

### 7.1 PageRank Algorithm

**Purpose**: Measure transitive importance in dependency graph.

```
Algorithm: PageRank
Input: Graph G = (V, E), damping factor d = 0.85, iterations = 100
Output: PageRank scores PR[v] for all v ∈ V

1. Initialize: PR[v] = 1/|V| for all v
2. Repeat until convergence:
   For each v ∈ V:
     PR[v] = (1-d)/|V| + d × Σ (PR[u]/out_degree[u])
                          u ∈ in_neighbors(v)
3. Return PR
```

**Complexity**: O(|V| + |E|) per iteration

### 7.2 Betweenness Centrality (Brandes Algorithm)

**Purpose**: Measure bottleneck position in graph.

```
Algorithm: Brandes Betweenness Centrality
Input: Graph G = (V, E)
Output: Betweenness scores BT[v] for all v ∈ V

1. Initialize: BT[v] = 0 for all v
2. For each source s ∈ V:
   a. Run BFS/Dijkstra from s to compute:
      - σ[t] = number of shortest paths from s to t
      - d[t] = distance from s to t
      - P[t] = predecessors of t on shortest paths
   b. Backpropagate dependencies:
      δ[v] = Σ (σ[v]/σ[w]) × (1 + δ[w])
             w: v ∈ P[w]
   c. Accumulate: BT[v] += δ[v] for v ≠ s
3. Normalize: BT[v] = BT[v] / ((|V|-1)(|V|-2)/2)
4. Return BT
```

**Complexity**: O(|V| × |E|) for unweighted, O(|V| × |E| + |V|² log |V|) for weighted

### 7.3 Articulation Point Detection

**Purpose**: Identify single points of failure.

```
Algorithm: Tarjan's Articulation Points
Input: Graph G = (V, E)
Output: Set of articulation points AP

1. Initialize:
   visited[v] = false, disc[v] = 0, low[v] = 0, parent[v] = nil
   AP = {}, time = 0
   
2. For each unvisited vertex v:
   DFS_AP(v)
   
3. DFS_AP(u):
   visited[u] = true
   disc[u] = low[u] = ++time
   children = 0
   
   For each neighbor v of u:
     If not visited[v]:
       children++
       parent[v] = u
       DFS_AP(v)
       low[u] = min(low[u], low[v])
       
       // u is AP if:
       // (1) u is root with 2+ children, or
       // (2) u is not root and low[v] >= disc[u]
       If (parent[u] = nil and children >= 2) or
          (parent[u] ≠ nil and low[v] >= disc[u]):
         AP = AP ∪ {u}
     Else if v ≠ parent[u]:
       low[u] = min(low[u], disc[v])

4. Return AP
```

**Complexity**: O(|V| + |E|)

### 7.4 AHP Weight Calculation

**Purpose**: Derive weights from pairwise comparison matrices.

```
Algorithm: AHP Geometric Mean Method
Input: Pairwise comparison matrix A (n × n)
Output: Priority vector w (weights)

1. For each row i:
   GM[i] = (∏ A[i][j])^(1/n)
           j=1..n
           
2. total = Σ GM[i]
           i=1..n
           
3. For each i:
   w[i] = GM[i] / total

4. Validate consistency:
   λ_max = Σ (A × w)[i] / w[i] / n
           i=1..n
   CI = (λ_max - n) / (n - 1)
   CR = CI / RI[n]
   
   If CR > 0.10: WARN "Inconsistent matrix"

5. Return w
```

**Random Index (RI) values**:
| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| RI | 0 | 0 | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 |

### 7.5 Box-Plot Classification

**Purpose**: Adaptive threshold classification.

```
Algorithm: Box-Plot Classification
Input: Values V = [v₁, ..., vₙ], k_factor = 1.5
Output: Level[i] for each value

1. Sort V
2. Compute quartiles:
   Q1 = percentile(V, 25)
   Median = percentile(V, 50)
   Q3 = percentile(V, 75)
   IQR = Q3 - Q1
   
3. Compute thresholds:
   upper_fence = Q3 + k × IQR
   
4. For each value v[i]:
   If v[i] > upper_fence: Level[i] = CRITICAL
   Else if v[i] > Q3:     Level[i] = HIGH
   Else if v[i] > Median: Level[i] = MEDIUM
   Else if v[i] > Q1:     Level[i] = LOW
   Else:                  Level[i] = MINIMAL

5. Return Level
```

### 7.6 Failure Cascade Propagation

**Purpose**: Simulate cascade effects.

```
Algorithm: Cascade Propagation
Input: Graph G, target t, cascade_rules, max_depth
Output: Failed set F, cascade sequence S

1. Initialize:
   F = {t}, queue = [(t, 0)], S = [(t, "initial", 0)]
   Mark t as FAILED
   
2. While queue not empty:
   (current, depth) = queue.pop()
   If depth >= max_depth: continue
   
   // Physical cascade (Node → hosted)
   If current.type = "Node" and PHYSICAL in rules:
     For each component c hosted on current:
       If c ∉ F and random() < cascade_prob:
         F = F ∪ {c}
         Mark c as FAILED
         S.append((c, "hosted_on:" + current, depth+1))
         queue.push((c, depth+1))
   
   // Logical cascade (Broker → topics)
   If current.type = "Broker" and LOGICAL in rules:
     For each topic routed by current:
       If no other active broker routes topic:
         Mark topic as UNREACHABLE
         // Don't fail subscribers, but track impact
   
   // Network cascade (partitions)
   If current.type = "Node" and NETWORK in rules:
     // Check connectivity impact
     
3. Return F, S
```

### 7.7 Spearman Rank Correlation

**Purpose**: Measure ranking correlation.

```
Algorithm: Spearman Correlation
Input: Paired values (X, Y) of size n
Output: Correlation coefficient ρ

1. Compute ranks:
   R_X[i] = rank of X[i] in X
   R_Y[i] = rank of Y[i] in Y
   
2. Compute differences:
   d[i] = R_X[i] - R_Y[i]
   
3. Compute coefficient:
   ρ = 1 - (6 × Σ d[i]²) / (n × (n² - 1))
   
4. Return ρ
```

**Interpretation**:
| ρ | Strength |
|---|----------|
| 0.9 - 1.0 | Very strong |
| 0.7 - 0.9 | Strong |
| 0.5 - 0.7 | Moderate |
| 0.3 - 0.5 | Weak |
| 0.0 - 0.3 | Negligible |

---

## 8. Database Design

### 8.1 Neo4j Graph Schema

#### 8.1.1 Node Labels (Vertices)

```cypher
// Component nodes
(:Application {
  id: String!,        // Unique identifier
  name: String,       // Human-readable name
  role: String,       // publisher, subscriber, both
  app_type: String,   // service, driver, controller, etc.
  version: String,    // Semantic version
  criticality: String,// low, medium, high, critical
  weight: Float       // Computed importance weight
})

(:Broker {
  id: String!,
  name: String,
  weight: Float
})

(:Topic {
  id: String!,
  name: String,
  size: Integer,          // Message size in bytes
  qos_reliability: String,
  qos_durability: String,
  qos_transport_priority: String,
  weight: Float           // Computed from QoS
})

(:Node {
  id: String!,
  name: String,
  weight: Float
})

(:Library {
  id: String!,
  name: String,
  version: String,
  weight: Float
})
```

#### 8.1.2 Relationship Types (Edges)

```cypher
// Structural relationships
(app:Application)-[:RUNS_ON]->(node:Node)
(broker:Broker)-[:RUNS_ON]->(node:Node)
(broker:Broker)-[:ROUTES {weight: Float}]->(topic:Topic)
(app:Application)-[:PUBLISHES_TO {weight: Float}]->(topic:Topic)
(app:Application)-[:SUBSCRIBES_TO {weight: Float}]->(topic:Topic)
(node1:Node)-[:CONNECTS_TO]->(node2:Node)
(app:Application)-[:USES {weight: Float}]->(lib:Library)
(lib1:Library)-[:USES {weight: Float}]->(lib2:Library)

// Derived dependencies
(source)-[:DEPENDS_ON {
  dependency_type: String,  // app_to_app, node_to_node, etc.
  weight: Float,
  shared_topics: Integer,
  shared_brokers: Integer
}]->(target)
```

### 8.2 Cypher Queries

#### 8.2.1 Dependency Derivation

```cypher
// Derive app_to_app dependencies
MATCH (pub:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(sub:Application)
WHERE pub <> sub
WITH sub, pub, collect(t) as shared_topics
MERGE (sub)-[d:DEPENDS_ON]->(pub)
SET d.dependency_type = 'app_to_app',
    d.weight = reduce(w = 0.0, t IN shared_topics | w + t.weight),
    d.shared_topics = size(shared_topics)

// Derive app_to_broker dependencies
MATCH (app:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(b:Broker)
WITH app, b, collect(DISTINCT t) as topics
MERGE (app)-[d:DEPENDS_ON]->(b)
SET d.dependency_type = 'app_to_broker',
    d.weight = reduce(w = 0.0, t IN topics | w + t.weight)
```

#### 8.2.2 Layer Extraction

```cypher
// Get application layer subgraph
MATCH (a:Application)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(b:Application)
RETURN a.id, a.name, a.weight, b.id, b.name, d.weight

// Get infrastructure layer subgraph
MATCH (n1:Node)-[d:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2:Node)
RETURN n1.id, n1.name, n2.id, n2.name, d.weight
```

### 8.3 Index Design

```cypher
// Uniqueness constraints (implicit indexes)
CREATE CONSTRAINT FOR (a:Application) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT FOR (b:Broker) REQUIRE b.id IS UNIQUE;
CREATE CONSTRAINT FOR (t:Topic) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT FOR (n:Node) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT FOR (l:Library) REQUIRE l.id IS UNIQUE;

// Performance indexes
CREATE INDEX FOR (d:DEPENDS_ON) ON (d.dependency_type);
CREATE INDEX FOR (a:Application) ON (a.weight);
```

---

## 9. User Interface Design

### 9.1 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 📊 Software-as-a-Graph Dashboard          [Overview] [App] [Infra] [Sys]│
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │   48    │ │   127   │ │    5    │ │    3    │ │    2    │          │
│  │  Nodes  │ │  Edges  │ │Critical │ │  SPOFs  │ │Problems │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
│                                                                         │
│  ┌────────────────────────────┐ ┌────────────────────────────┐         │
│  │   Criticality Distribution │ │     Component Types        │         │
│  │        [PIE CHART]         │ │       [PIE CHART]          │         │
│  └────────────────────────────┘ └────────────────────────────┘         │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Interactive Network Graph                      │  │
│  │                         [vis.js]                                  │  │
│  │                                                                   │  │
│  │     ●───────●                 ●                                  │  │
│  │    / \     / \               /|\                                  │  │
│  │   ●   ●   ●   ●             ● ● ●                                 │  │
│  │                                                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Top Components by Quality Score                                   │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ Component      │ Type       │   R   │   M   │   A   │   Q  │Level│  │
│  │────────────────┼────────────┼───────┼───────┼───────┼──────┼─────│  │
│  │ sensor_fusion  │Application │ 0.82  │ 0.75  │ 0.90  │ 0.84 │CRIT │  │
│  │ main_broker    │Broker      │ 0.78  │ 0.65  │ 0.95  │ 0.80 │CRIT │  │
│  │ planning_node  │Application │ 0.71  │ 0.73  │ 0.45  │ 0.64 │HIGH │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────────┐                                   │
│  │ Validation Metrics              │                                   │
│  ├─────────────────────────────────┤                                   │
│  │ Spearman ρ:  0.85  ✓           │                                   │
│  │ F1-Score:    0.83  ✓           │                                   │
│  │ Precision:   0.86  ✓           │                                   │
│  │ Recall:      0.80  ✓           │                                   │
│  │ Status:      PASSED            │                                   │
│  └─────────────────────────────────┘                                   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ Generated by Software-as-a-Graph • January 2026                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Color Scheme

| Element | Color | Hex | Usage |
|---------|-------|-----|-------|
| CRITICAL | Red | #E74C3C | Highest severity |
| HIGH | Orange | #E67E22 | High severity |
| MEDIUM | Yellow | #F1C40F | Medium severity |
| LOW | Green | #2ECC71 | Low severity |
| MINIMAL | Blue | #3498DB | Minimal severity |
| Primary | Dark Blue | #2C3E50 | Headers, emphasis |
| Secondary | Gray | #7F8C8D | Muted text |
| Background | Light Gray | #ECF0F1 | Page background |
| Surface | White | #FFFFFF | Cards, tables |

### 9.3 Network Visualization

**Node Styling**:
- Size: Proportional to quality score
- Color: Based on criticality level
- Label: Component name
- Shape: By component type
  - Application: Circle
  - Broker: Diamond
  - Node: Square
  - Topic: Triangle

**Edge Styling**:
- Width: Proportional to weight
- Color: By dependency type
- Arrow: Direction of dependency

---

## Appendices

### Appendix A: Layer Definitions

```python
LAYER_DEFINITIONS = {
    "app": LayerDefinition(
        name="Application Layer",
        description="Application-level dependencies",
        component_types=frozenset(["Application"]),
        dependency_types=frozenset(["app_to_app"]),
        icon="📱"
    ),
    "infra": LayerDefinition(
        name="Infrastructure Layer",
        description="Infrastructure-level dependencies",
        component_types=frozenset(["Node"]),
        dependency_types=frozenset(["node_to_node"]),
        icon="🖥️"
    ),
    "mw-app": LayerDefinition(
        name="Middleware-Application Layer",
        description="Application to middleware dependencies",
        component_types=frozenset(["Application", "Broker"]),
        dependency_types=frozenset(["app_to_app", "app_to_broker"]),
        icon="🔗"
    ),
    "mw-infra": LayerDefinition(
        name="Middleware-Infrastructure Layer",
        description="Infrastructure to middleware dependencies",
        component_types=frozenset(["Node", "Broker"]),
        dependency_types=frozenset(["node_to_node", "node_to_broker"]),
        icon="⚙️"
    ),
    "system": LayerDefinition(
        name="Complete System",
        description="All components and dependencies",
        component_types=frozenset(["Application", "Broker", "Node", "Topic", "Library"]),
        dependency_types=frozenset(["app_to_app", "node_to_node", "app_to_broker", "node_to_broker"]),
        icon="🌐"
    ),
}
```

### Appendix B: Default AHP Matrices

```python
# Reliability: PageRank, Reverse PageRank, In-Degree
criteria_reliability = [
    [1.0, 2.0, 2.0],  # PR moderately more important
    [0.5, 1.0, 1.0],  # RPR
    [0.5, 1.0, 1.0],  # ID
]
# Resulting weights: [0.50, 0.25, 0.25]

# Maintainability: Betweenness, Degree, Clustering
criteria_maintainability = [
    [1.0, 2.0, 3.0],  # BT more important
    [0.5, 1.0, 2.0],  # DG
    [0.33, 0.5, 1.0], # CC
]
# Resulting weights: [0.54, 0.30, 0.16]

# Availability: Articulation, Bridge, Importance
criteria_availability = [
    [1.0, 3.0, 5.0],  # AP strongly dominant
    [0.33, 1.0, 2.0], # BR
    [0.2, 0.5, 1.0],  # IM
]
# Resulting weights: [0.65, 0.23, 0.12]

# Vulnerability: Eigenvector, Closeness, In-Degree
criteria_vulnerability = [
    [1.0, 2.0, 2.0],
    [0.5, 1.0, 1.0],
    [0.5, 1.0, 1.0],
]
# Resulting weights: [0.50, 0.25, 0.25]

# Overall: R, M, A, V (equal by default)
criteria_overall = [
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
]
# Resulting weights: [0.25, 0.25, 0.25, 0.25]
```

### Appendix C: Error Handling Strategy

| Error Type | Handling Strategy |
|------------|-------------------|
| Neo4j Connection | Retry with exponential backoff, then fail gracefully |
| Invalid Input | Validate early, return descriptive error |
| Empty Graph | Return empty result with warning |
| Algorithm Convergence | Use fallback values, log warning |
| Memory Exhaustion | Implement streaming/batching for large graphs |
| File I/O | Use context managers, ensure cleanup |

### Appendix D: Testing Strategy

| Test Type | Coverage Target | Tools |
|-----------|-----------------|-------|
| Unit Tests | 80% | pytest |
| Integration Tests | Key workflows | pytest + Neo4j testcontainer |
| Performance Tests | Scale benchmarks | custom benchmark.py |
| Validation Tests | Statistical accuracy | scipy.stats |

---

*Document Version: 1.0*  
*Last Updated: January 2026*  
*Software-as-a-Graph Framework*
