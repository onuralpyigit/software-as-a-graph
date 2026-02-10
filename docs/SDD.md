# Software Design Description

## Software-as-a-Graph

### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 2.0** · **February 2026**

Istanbul Technical University, Computer Engineering Department

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Overview](#2-system-overview)
3. [System Architecture](#3-system-architecture)
4. [Data Design](#4-data-design)
5. [Component Design](#5-component-design)
6. [Algorithmic Design](#6-algorithmic-design)
7. [Database Design](#7-database-design)
8. [Interface Design](#8-interface-design)
9. [User Interface Design](#9-user-interface-design)
10. [Appendices](#10-appendices)

---

## 1. Introduction

### 1.1 Purpose

This document describes the technical design of the Software-as-a-Graph framework — how the system is structured, how its components interact, what algorithms it uses, and how data flows through the pipeline. It serves as the primary reference for developers maintaining or extending the codebase.

### 1.2 Scope

The design covers the full six-step methodology pipeline: graph model construction, structural analysis, quality scoring, failure simulation, statistical validation, and interactive visualization. The system follows a layered architecture with hexagonal (ports-and-adapters) principles in the domain core.

### 1.3 References

| Reference | Description |
|-----------|-------------|
| IEEE 1016-2009 | IEEE Standard for Software Design Descriptions |
| SRS v2.0 | Software Requirements Specification for this project |
| Neo4j Documentation | https://neo4j.com/docs/ |
| NetworkX Documentation | https://networkx.org/documentation/ |

### 1.4 Glossary

| Term | Definition |
|------|------------|
| CLI | Command-Line Interface |
| DTO | Data Transfer Object — plain data carrier between layers |
| GDS | Graph Data Science (Neo4j plugin) |
| IQR | Interquartile Range (Q3 − Q1) |
| RMAV | Reliability, Maintainability, Availability, Vulnerability |
| SOLID | Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion |

---

## 2. System Overview

### 2.1 System Context

```
                              ┌──────────────────┐
                              │      User        │
                              └────────┬─────────┘
                                       │ CLI commands
                    ┌──────────────────▼──────────────────┐
                    │                                      │
   ┌────────────┐   │     Software-as-a-Graph Framework    │   ┌────────────┐
   │   JSON     │──▶│                                      │──▶│   HTML     │
   │  Topology  │   │  Generate → Import → Analyze →       │   │ Dashboard  │
   │   Input    │   │  Simulate → Validate → Visualize     │   │  + JSON    │
   └────────────┘   │                                      │   └────────────┘
                    └──────────────────┬──────────────────┘
                                       │
                              ┌────────▼─────────┐
                              │   Neo4j Graph DB  │
                              └──────────────────┘
```

### 2.2 Design Constraints

| Constraint | Implication |
|------------|-------------|
| Neo4j 5.x required | Graph storage and GDS algorithm execution |
| Python 3.9+ | Type hints, dataclasses, walrus operator usage |
| Memory-bound | Graph size limited by available RAM; target ≤ 1,000 components |
| NetworkX dependency | All centrality algorithms delegated to NetworkX |

### 2.3 Design Principles

The system follows SOLID principles with emphasis on three key decisions:

**Separation of prediction from validation.** Steps 2–3 produce predicted scores Q(v) using only topology. Step 4 produces ground-truth scores I(v) using simulation. Step 5 compares the two. This separation prevents circular reasoning and ensures methodological rigor.

**Layered architecture with dependency inversion.** Domain logic (models, services, algorithms) has zero dependencies on infrastructure (Neo4j, file system). Infrastructure adapters implement domain-defined interfaces, making the core testable without a database.

**Composition over inheritance.** Services are composed via constructor injection. Classification strategies (box-plot, percentile), weight calculation strategies (default, AHP), and normalization strategies are interchangeable without modifying calling code.

---

## 3. System Architecture

### 3.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                           │
│                                                                     │
│  bin/run.py          bin/analyze_graph.py     bin/simulate_graph.py │
│  bin/generate_graph.py  bin/validate_graph.py  bin/visualize_graph.py│
│                                                                     │
│  Each CLI tool parses arguments, creates a Container, and           │
│  delegates to application services.                                 │
├─────────────────────────────────────────────────────────────────────┤
│                        APPLICATION LAYER                            │
│                                                                     │
│  Container            AnalysisService      SimulationService        │
│  (DI wiring)          ValidationService    VisualizationService     │
│                                                                     │
│  Orchestrates service calls and manages cross-cutting concerns      │
│  (logging, configuration, error handling).                          │
├─────────────────────────────────────────────────────────────────────┤
│                         DOMAIN LAYER                                │
│                                                                     │
│  Models:  GraphData, StructuralMetrics, QualityScores,              │
│           ImpactMetrics, FailureResult, ValidationGroupResult       │
│                                                                     │
│  Services: StructuralAnalyzer, QualityAnalyzer, BoxPlotClassifier,  │
│            AHPProcessor, ProblemDetector, SimulationGraph            │
│                                                                     │
│  Ports:   IGraphRepository (interface for data access)              │
│                                                                     │
│  Pure Python. No infrastructure dependencies.                       │
├─────────────────────────────────────────────────────────────────────┤
│                      INFRASTRUCTURE LAYER                           │
│                                                                     │
│  Neo4jGraphRepository    FileSystemExporter    DashboardGenerator   │
│  (implements IGraphRepository)                                      │
│                                                                     │
│  Neo4j Bolt driver, file I/O, HTML generation.                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Decomposition

```
software-as-a-graph/
├── bin/                              # Presentation Layer
│   ├── run.py                        #   Pipeline orchestrator
│   ├── generate_graph.py             #   Synthetic data generation
│   ├── import_graph.py               #   Neo4j import
│   ├── analyze_graph.py              #   Analysis + quality scoring
│   ├── simulate_graph.py             #   Failure simulation
│   ├── validate_graph.py             #   Statistical validation
│   └── visualize_graph.py            #   Dashboard generation
│
├── src/
│   ├── domain/                       # Domain Layer (pure logic)
│   │   ├── models/                   #   Data classes and value objects
│   │   │   ├── value_objects.py      #     QoSPolicy, Component entities
│   │   │   ├── metrics.py            #     StructuralMetrics, QualityScores
│   │   │   ├── simulation/           #     ImpactMetrics, FailureResult
│   │   │   └── validation/           #     CorrelationMetrics, ValidationTargets
│   │   ├── services/                 #   Domain services (algorithms)
│   │   │   ├── structural_analyzer.py
│   │   │   ├── quality_analyzer.py
│   │   │   ├── weight_calculator.py  #     AHP processor + QualityWeights
│   │   │   ├── classifier.py         #     BoxPlotClassifier
│   │   │   └── problem_detector.py
│   │   └── config/                   #   Layer definitions, constants
│   │
│   ├── application/                  # Application Layer (orchestration)
│   │   ├── container.py              #   Dependency injection container
│   │   └── services/                 #   AnalysisService, SimulationService,
│   │                                 #   ValidationService, VisualizationService
│   │
│   └── infrastructure/               # Infrastructure Layer (adapters)
│       ├── neo4j_repo.py             #   Neo4j graph repository
│       ├── file_exporter.py          #   JSON/CSV/GraphML export
│       └── visualization/            #   DashboardGenerator, ChartGenerator
│
├── config/                           # YAML scale presets
├── docs/                             # Methodology documentation
└── tests/                            # Unit and integration tests
```

### 3.3 Design Patterns

| Pattern | Where Used | Purpose |
|---------|-----------|---------|
| **Facade** | `AnalysisService`, `SimulationService` | Simplified entry point to multi-step workflows |
| **Strategy** | `BoxPlotClassifier` / percentile fallback, `AHPProcessor` / default weights | Interchangeable classification and weight algorithms |
| **Builder** | `DashboardGenerator` | Incremental construction of HTML dashboards |
| **Repository** | `IGraphRepository` → `Neo4jGraphRepository` | Abstract data access behind a domain interface |
| **DTO** | `GraphData`, `ComponentData`, `EdgeData` | Clean data transfer between layers |
| **Factory** | `generate_graph()` | Synthetic topology generation with configurable scale |
| **Context Manager** | All database clients | Safe resource acquisition and release |

### 3.4 Data Flow Through the Pipeline

```
JSON Topology
     │
     ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Import to   │    │ Structural  │    │  Quality    │
│  Neo4j       │───▶│  Analyzer   │───▶│  Analyzer   │
│  (Step 1)    │    │  (Step 2)   │    │  (Step 3)   │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                    G_structural             Q(v) predicted
                         │                   │
                         ▼                   │
                   ┌─────────────┐           │
                   │  Failure    │           │
                   │  Simulator  │           │
                   │  (Step 4)   │           │
                   └──────┬──────┘           │
                          │                  │
                     I(v) actual             │
                          │                  │
                          ▼                  ▼
                   ┌─────────────────────────────┐
                   │        Validator             │
                   │        (Step 5)              │
                   │  Compare Q(v) vs I(v)        │
                   └──────────────┬──────────────┘
                                  │
                          All results
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │    Dashboard Generator       │
                   │        (Step 6)              │
                   └─────────────────────────────┘
```

Key: Steps 2–3 operate on **G_analysis** (derived DEPENDS_ON edges only). Step 4 operates on **G_structural** (all raw relationships) for realistic cascade propagation. This separation is deliberate — analysis needs abstracted dependencies for centrality, simulation needs physical topology for cascades.

---

## 4. Data Design

### 4.1 Domain Model Entities

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Application │  │   Broker    │  │    Topic    │  │    Node     │  │   Library   │
├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤
│ id: str     │  │ id: str     │  │ id: str     │  │ id: str     │  │ id: str     │
│ name: str   │  │ name: str   │  │ name: str   │  │ name: str   │  │ name: str   │
│ role: str   │  │ weight: f   │  │ qos: QoS    │  │ weight: f   │  │ version: str│
│ app_type: s │  └─────────────┘  │ size: int   │  └─────────────┘  │ weight: f   │
│ weight: f   │                   │ weight: f   │                   └─────────────┘
└─────────────┘  ┌─────────────┐  └─────────────┘
                 │  QoSPolicy  │
                 ├─────────────┤
                 │ reliability │  → RELIABLE | BEST_EFFORT
                 │ durability  │  → PERSISTENT | TRANSIENT | VOLATILE
                 │ priority    │  → URGENT | HIGH | MEDIUM | LOW
                 │ msg_size    │  → bytes
                 └─────────────┘
```

### 4.2 Metrics Data Structures

These are Python dataclasses that flow between services. All continuous metrics are normalized to [0, 1].

**StructuralMetrics** — output of Step 2, one per component:

| Field | Type | Description |
|-------|------|-------------|
| `pagerank` | float | Transitive importance |
| `reverse_pagerank` | float | Reverse transitive importance |
| `betweenness` | float | Shortest-path bottleneck position |
| `closeness` | float | Average distance to all others |
| `eigenvector` | float | Connection to important hubs |
| `in_degree_raw` | int | Count of incoming DEPENDS_ON edges |
| `out_degree_raw` | int | Count of outgoing DEPENDS_ON edges |
| `clustering_coefficient` | float | Neighbor interconnectedness |
| `is_articulation_point` | bool | Removal disconnects graph |
| `bridge_ratio` | float | Fraction of edges that are bridges |
| `weight` | float | Component's own QoS-derived weight |

**QualityScores** — output of Step 3, one per component:

| Field | Type | Description |
|-------|------|-------------|
| `reliability` | float | R(v) — fault propagation risk |
| `maintainability` | float | M(v) — coupling complexity |
| `availability` | float | A(v) — SPOF risk |
| `vulnerability` | float | V(v) — security exposure |
| `overall` | float | Q(v) — composite quality score |

**ImpactMetrics** — output of Step 4, one per simulated failure:

| Field | Type | Description |
|-------|------|-------------|
| `reachability_loss` | float | Fraction of broken pub-sub paths |
| `fragmentation` | float | Graph disconnection severity |
| `throughput_loss` | float | Weighted message capacity reduction |
| `cascade_count` | int | Number of cascaded failures |
| `cascade_depth` | int | Maximum cascade propagation depth |
| `composite_impact` | float | I(v) = weighted combination |

**ValidationGroupResult** — output of Step 5, one per layer:

| Field | Type | Description |
|-------|------|-------------|
| `correlation` | CorrelationMetrics | Spearman ρ, Kendall τ, Pearson r |
| `classification` | ClassificationMetrics | Precision, Recall, F1, Cohen's κ |
| `ranking` | RankingMetrics | Top-5 overlap, Top-10 overlap, NDCG |
| `error` | ErrorMetrics | RMSE, MAE |
| `passed` | bool | Whether all primary gates passed |

### 4.3 Data Transfer Objects

`GraphData`, `ComponentData`, and `EdgeData` are lightweight DTOs that carry graph data between layers without exposing domain internals:

```python
@dataclass
class ComponentData:
    id: str
    component_type: str
    weight: float
    properties: Dict[str, Any]

@dataclass
class EdgeData:
    source_id: str
    target_id: str
    dependency_type: str
    weight: float
    properties: Dict[str, Any]

@dataclass
class GraphData:
    components: List[ComponentData]
    edges: List[EdgeData]
```

---

## 5. Component Design

### 5.1 Import Pipeline (`Neo4jGraphRepository`)

Imports system topology into Neo4j and derives dependencies in four phases:

```
CLI: import_graph.py
         │
         ▼
  Neo4jGraphRepository.import_graph(data, clear=True)
         │
    Phase 1: Create vertices (Node, Broker, Topic, Application, Library)
    Phase 2: Create structural edges (RUNS_ON, ROUTES, PUBLISHES_TO, ...)
    Phase 3: Compute QoS-based weights on Topics, propagate to edges
    Phase 4: Derive DEPENDS_ON edges using four rules:
             - app_to_app:    App → Topic → App
             - app_to_broker: App → Topic ← Broker
             - node_to_node:  Node hosts Broker → Topic ← Broker on Node
             - node_to_broker: Node hosts App → Topic ← Broker
         │
         ▼
    Return import statistics (counts per entity type)
```

### 5.2 Analysis Pipeline (`StructuralAnalyzer` + `QualityAnalyzer`)

```
AnalysisService.analyze_layer(layer)
         │
    1. Export G_analysis(layer) from Neo4j → NetworkX DiGraph
    2. StructuralAnalyzer.analyze(graph)
         │  → Compute 13 metrics per component
         │  → Compute edge metrics (betweenness, bridge detection)
         │  → Compute continuous AP_c scores via iterated removal
         │  → Return StructuralAnalysisResult
    3. QualityAnalyzer.analyze(structural_result)
         │  → Normalize metrics to [0, 1] via min-max
         │  → Compute RMAV scores using QualityWeights
         │  → Classify via BoxPlotClassifier (or percentile fallback)
         │  → Score edges using endpoint-aware RMAV formulas
         │  → Return QualityAnalysisResult
    4. ProblemDetector.detect(quality_result)
         │  → Identify architectural anti-patterns (SPOFs, god components,
         │     bottleneck edges, systemic risk patterns)
         │  → Return List[DetectedProblem]
         │
         ▼
    Return LayerAnalysisResult
```

### 5.3 Simulation Pipeline (`SimulationGraph` + `FailureSimulator`)

```
SimulationService.run_exhaustive(layer)
         │
    1. Build SimulationGraph from G_structural (all raw relationships)
    2. Compute baseline state (paths, components, topic weights)
    3. For each component v in layer:
         │
         a. Remove v from graph
         b. Propagate cascades:
              Physical: Node fails → hosted Apps/Brokers fail
              Logical:  Broker fails → exclusively-routed Topics die
              Application: Publisher fails → starved Subscribers fail
              (repeat until no new failures)
         c. Measure impact vs. baseline:
              ReachabilityLoss, Fragmentation, ThroughputLoss
         d. Record FailureResult with cascade sequence
         e. Restore v
         │
         ▼
    Return List[FailureResult] with I(v) for every component
```

### 5.4 Validation Pipeline (`Validator`)

```
ValidationService.validate_layer(analysis_result, simulation_results)
         │
    1. Extract Q(v) from QualityAnalysisResult
    2. Extract I(v) from List[FailureResult]
    3. Align by component ID (warn on mismatches)
    4. Compute:
         - Correlation: Spearman ρ, Kendall τ, Pearson r
         - Classification: Precision, Recall, F1, Cohen's κ
         - Ranking: Top-5 overlap, Top-10 overlap, NDCG@K
         - Error: RMSE, MAE
    5. Evaluate against ValidationTargets → pass/fail
         │
         ▼
    Return ValidationGroupResult
```

### 5.5 Visualization Pipeline (`DashboardGenerator`)

```
VisualizationService.generate_dashboard(layers, output_file)
         │
    1. For each layer: collect analysis + simulation + validation data
    2. DashboardGenerator (Builder pattern):
         │
         .start_section("Overview")
         .add_kpis({nodes, edges, critical, spofs, problems})
         .add_charts([criticality_pie, rmav_bars, scatter_plot])
         .end_section()
         │
         .start_section("Layer: Application")
         .add_table(component_details)
         .add_network(nodes, edges)       ← vis.js interactive graph
         .add_matrix(dependency_heatmap)
         .end_section()
         │
         ... repeat for each layer ...
         │
         .start_section("Validation")
         .add_validation_metrics(pass/fail indicators)
         .end_section()
         │
         .generate() → complete HTML string
         │
         ▼
    Write HTML file, optionally open in browser
```

---

## 6. Algorithmic Design

### 6.1 PageRank

Measures transitive importance in the dependency graph.

```
Input:  G = (V, E), damping d = 0.85, max iterations = 100
Output: PR[v] ∈ [0, 1] for all v

1. Initialize PR[v] = 1/|V| for all v
2. Repeat until convergence:
     PR[v] = (1−d)/|V| + d × Σ PR[u]/out_degree(u)  for u ∈ in_neighbors(v)
3. Normalize to [0, 1]
```

Complexity: O(|V| + |E|) per iteration. Delegated to `networkx.pagerank()`.

### 6.2 Betweenness Centrality (Brandes)

Measures bottleneck position — how often a component sits on shortest paths.

```
Input:  G = (V, E)
Output: BT[v] ∈ [0, 1] for all v

1. For each source s ∈ V:
     a. BFS from s → shortest path counts σ[t], predecessors P[t]
     b. Backpropagate: δ[v] = Σ (σ[v]/σ[w]) × (1 + δ[w])  for w where v ∈ P[w]
     c. Accumulate: BT[v] += δ[v]
2. Normalize: BT[v] /= (|V|−1)(|V|−2)/2
```

Complexity: O(|V| × |E|). Delegated to `networkx.betweenness_centrality()`.

### 6.3 Articulation Point Detection (Tarjan)

Identifies single points of failure — vertices whose removal disconnects the graph.

```
Input:  G = (V, E)
Output: Set of articulation points AP

1. DFS traversal, tracking discovery time disc[v] and lowest reachable ancestor low[v]
2. Vertex u is an articulation point if:
     (a) u is root of DFS tree and has ≥ 2 children, OR
     (b) u is not root and has a child v where low[v] ≥ disc[u]
```

Complexity: O(|V| + |E|). The binary result is extended to a continuous score:

```
AP_c(v) = 1 − |largest_CC(G \ {v})| / (|V| − 1)
```

This requires O(|V| × (|V| + |E|)) for all vertices.

### 6.4 AHP Weight Calculation (Geometric Mean)

Derives quality weights from expert pairwise comparison matrices.

```
Input:  n × n comparison matrix A (Saaty's scale: 1–9)
Output: Priority vector w (weights summing to 1.0)

1. Geometric mean per row: GM[i] = (∏ A[i][j])^(1/n)
2. Normalize: w[i] = GM[i] / Σ GM
3. Consistency check:
     λ_max = average of (A × w)[i] / w[i]
     CI = (λ_max − n) / (n − 1)
     CR = CI / RI[n]
     If CR > 0.10: warn "Inconsistent matrix"
```

Random Index (RI) values for matrix sizes 3–10:

| n | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| RI | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 |

### 6.5 Box-Plot Classification

Adaptive threshold classification that adjusts to actual data distributions.

```
Input:  Scores V = [v₁, ..., vₙ], k = 1.5
Output: Criticality level per component

1. Compute: Q1, Median, Q3, IQR = Q3 − Q1
2. Upper fence = Q3 + k × IQR
3. Classify:
     v > upper_fence  → CRITICAL
     v > Q3           → HIGH
     v > Median        → MEDIUM
     v > Q1           → LOW
     v ≤ Q1           → MINIMAL
```

Fallback for small samples (n < 12): fixed percentile thresholds (top 10% = CRITICAL, top 25% = HIGH, top 50% = MEDIUM, top 75% = LOW, rest = MINIMAL).

### 6.6 Cascade Propagation

Simulates failure spread through the system graph.

```
Input:  G_structural, target component t, cascade rules, max_depth
Output: Failed set F, cascade sequence S

1. F = {t}, queue = [t], S = [(t, "initial", depth=0)]
2. While queue not empty:
     current = queue.pop()
     
     Physical cascade (if current is Node):
       For each component hosted on current → add to F, enqueue
     
     Logical cascade (if current is Broker):
       For each topic exclusively routed by current → mark unreachable
       For each subscriber with no remaining data source → add to F, enqueue
     
     Application cascade (if current is publisher):
       For each subscriber starved of all publishers → add to F, enqueue
     
3. Repeat step 2 until no new failures (fixed-point)
4. Return F, S
```

In Monte Carlo mode, each cascade step propagates with probability p < 1.0, and N trials produce a distribution of I(v) values.

### 6.7 Spearman Rank Correlation

Measures whether predicted and actual rankings agree.

```
Input:  Paired values (X, Y) of size n
Output: Correlation coefficient ρ ∈ [−1, 1]

1. R_X = ranks of X, R_Y = ranks of Y
2. d[i] = R_X[i] − R_Y[i]
3. ρ = 1 − (6 × Σ d[i]²) / (n × (n² − 1))
```

| ρ Range | Interpretation |
|---------|---------------|
| 0.9 – 1.0 | Very strong |
| 0.7 – 0.9 | Strong |
| 0.5 – 0.7 | Moderate |
| 0.3 – 0.5 | Weak |
| 0.0 – 0.3 | Negligible |

---

## 7. Database Design

### 7.1 Neo4j Schema

#### Node Labels (Vertices)

```cypher
(:Application {id: String!, name: String, role: String,
               app_type: String, version: String, weight: Float})

(:Broker      {id: String!, name: String, weight: Float})

(:Topic       {id: String!, name: String, size: Integer,
               qos_reliability: String, qos_durability: String,
               qos_transport_priority: String, weight: Float})

(:Node        {id: String!, name: String, weight: Float})

(:Library     {id: String!, name: String, version: String, weight: Float})
```

#### Relationship Types (Edges)

```cypher
// Structural (from input topology)
(:Application)-[:RUNS_ON]->(:Node)
(:Broker)-[:RUNS_ON]->(:Node)
(:Broker)-[:ROUTES {weight: Float}]->(:Topic)
(:Application)-[:PUBLISHES_TO {weight: Float}]->(:Topic)
(:Application)-[:SUBSCRIBES_TO {weight: Float}]->(:Topic)
(:Node)-[:CONNECTS_TO]->(:Node)
(:Application)-[:USES {weight: Float}]->(:Library)

// Derived (computed during import)
(source)-[:DEPENDS_ON {
    dependency_type: String,  // app_to_app | node_to_node | app_to_broker | node_to_broker
    weight: Float,
    shared_topics: Integer
}]->(target)
```

### 7.2 Key Cypher Queries

**Dependency derivation (app_to_app)**:

```cypher
MATCH (pub:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(sub:Application)
WHERE pub <> sub
WITH sub, pub, collect(t) AS shared_topics
MERGE (sub)-[d:DEPENDS_ON]->(pub)
SET d.dependency_type = 'app_to_app',
    d.weight = reduce(w = 0.0, t IN shared_topics | w + t.weight),
    d.shared_topics = size(shared_topics)
```

**Layer extraction (application layer)**:

```cypher
MATCH (a:Application)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(b:Application)
RETURN a.id, a.name, a.weight, b.id, b.name, d.weight
```

### 7.3 Indexes and Constraints

```cypher
// Uniqueness constraints (also serve as indexes)
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Application) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (b:Broker) REQUIRE b.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (l:Library) REQUIRE l.id IS UNIQUE;

// Performance index for layer filtering
CREATE INDEX IF NOT EXISTS FOR ()-[d:DEPENDS_ON]-() ON (d.dependency_type);
```

---

## 8. Interface Design

### 8.1 CLI Interface

**Pipeline orchestrator** (`bin/run.py`):

```
python bin/run.py --all --layer system [--scale medium] [--open]
python bin/run.py --generate --import --analyze --layer app
```

**Individual step CLIs** — all share common flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--layer LAYER` | Target layer: app, infra, mw, system | system |
| `--uri URI` | Neo4j connection | bolt://localhost:7687 |
| `--user` / `--password` | Neo4j credentials | neo4j / password |
| `--output FILE` | Export results to JSON | — |
| `--verbose` / `--quiet` | Log level control | INFO |

### 8.2 Data Exchange Formats

**Input topology** (JSON):

```json
{
  "nodes": [{"id": "node1", "name": "Server 1"}],
  "brokers": [{"id": "broker1", "name": "Main Broker"}],
  "applications": [{
    "id": "app1", "name": "Sensor Fusion",
    "role": "subscriber", "app_type": "processor"
  }],
  "topics": [{
    "id": "topic1", "name": "/sensors/lidar", "size": 1024,
    "qos": {"reliability": "RELIABLE", "durability": "VOLATILE",
            "transport_priority": "HIGH"}
  }],
  "libraries": [{"id": "lib1", "name": "Nav Lib", "version": "2.0"}],
  "relationships": {
    "runs_on": [{"source": "app1", "target": "node1"}],
    "routes": [{"source": "broker1", "target": "topic1"}],
    "publishes_to": [{"source": "app2", "target": "topic1"}],
    "subscribes_to": [{"source": "app1", "target": "topic1"}]
  }
}
```

**Analysis output** (JSON):

```json
{
  "timestamp": "2026-02-10T10:30:00",
  "layer": "system",
  "components": [{
    "id": "app1", "name": "Sensor Fusion", "type": "Application",
    "scores": {"reliability": 0.82, "maintainability": 0.75,
               "availability": 0.90, "vulnerability": 0.68, "overall": 0.79},
    "levels": {"overall": "HIGH"}
  }],
  "classification_summary": {
    "total_components": 48,
    "distribution": {"CRITICAL": 5, "HIGH": 8, "MEDIUM": 15, "LOW": 12, "MINIMAL": 8}
  }
}
```

---

## 9. User Interface Design

### 9.1 Dashboard Layout

The HTML dashboard uses a responsive single-page layout with tabbed layer navigation:

```
┌──────────────────────────────────────────────────────────────────┐
│  Software-as-a-Graph Dashboard         [Overview][App][Infra][Sys]│
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │  48    │ │  127   │ │   5    │ │   3    │ │   2    │        │
│  │ Nodes  │ │ Edges  │ │Critical│ │ SPOFs  │ │Problems│        │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐             │
│  │ Criticality Dist.    │  │ RMAV Breakdown       │             │
│  │     [Pie Chart]      │  │   [Bar Chart]        │             │
│  └──────────────────────┘  └──────────────────────┘             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            Interactive Network Graph (vis.js)             │   │
│  │    Hover: details · Click: neighbors · Drag: reposition  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Component   │ Type   │  R   │  M   │  A   │  Q  │Level │   │
│  │──────────────┼────────┼──────┼──────┼──────┼─────┼──────│   │
│  │ sensor_fusion│ App    │ 0.82 │ 0.75 │ 0.90 │ 0.84│ CRIT │   │
│  │ main_broker  │ Broker │ 0.78 │ 0.65 │ 0.95 │ 0.80│ CRIT │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────┐                                   │
│  │ Validation: PASSED       │                                   │
│  │ Spearman ρ: 0.85 ✓      │                                   │
│  │ F1-Score:   0.83 ✓      │                                   │
│  └──────────────────────────┘                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 9.2 Visual Encoding

**Criticality colors:**

| Level | Color | Hex |
|-------|-------|-----|
| CRITICAL | Red | #E74C3C |
| HIGH | Orange | #E67E22 |
| MEDIUM | Yellow | #F1C40F |
| LOW | Green | #2ECC71 |
| MINIMAL | Blue | #3498DB |

**Network graph encoding:**

| Visual Property | Maps To |
|----------------|---------|
| Node size | Q(v) quality score |
| Node color | Criticality level (colors above) |
| Node shape | Component type (circle = App, diamond = Broker, square = Node) |
| Edge thickness | Dependency weight w(e) |
| Edge arrow | Dependency direction |
| Spatial position | Force-directed layout (central = high centrality) |

---

## 10. Appendices

### Appendix A: Layer Definitions

| Layer ID | Name | Component Types | Dependency Types |
|----------|------|----------------|-----------------|
| `app` | Application Layer | Application | app_to_app |
| `infra` | Infrastructure Layer | Node | node_to_node |
| `mw-app` | Middleware-Application | Application, Broker | app_to_app, app_to_broker |
| `mw-infra` | Middleware-Infrastructure | Node, Broker | node_to_node, node_to_broker |
| `system` | Complete System | All types | All dependency types |

### Appendix B: Default AHP Matrices

```python
# Reliability: [PageRank, ReversePageRank, InDegree]
criteria_reliability = [
    [1.0, 2.0, 2.0],   # PR moderately more important
    [0.5, 1.0, 1.0],   # RPR
    [0.5, 1.0, 1.0],   # InDegree
]  # → weights ≈ [0.50, 0.25, 0.25], defaults: [0.40, 0.35, 0.25]

# Maintainability: [Betweenness, OutDegree, (1−Clustering)]
criteria_maintainability = [
    [1.0, 2.0, 3.0],
    [0.5, 1.0, 2.0],
    [0.33, 0.5, 1.0],
]  # → weights ≈ [0.54, 0.30, 0.16], defaults: [0.40, 0.35, 0.25]

# Availability: [AP_c, BridgeRatio, Importance]
criteria_availability = [
    [1.0, 3.0, 5.0],   # AP_c strongly dominant
    [0.33, 1.0, 2.0],
    [0.2, 0.5, 1.0],
]  # → weights ≈ [0.65, 0.23, 0.12], defaults: [0.50, 0.30, 0.20]

# Vulnerability: [Eigenvector, Closeness, OutDegree]
criteria_vulnerability = [
    [1.0, 2.0, 2.0],
    [0.5, 1.0, 1.0],
    [0.5, 1.0, 1.0],
]  # → weights ≈ [0.50, 0.25, 0.25], defaults: [0.40, 0.30, 0.30]

# Overall: [R, M, A, V] — equal by default
criteria_overall = [
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
]  # → weights = [0.25, 0.25, 0.25, 0.25]
```

### Appendix C: Error Handling Strategy

| Error Type | Strategy |
|------------|----------|
| Neo4j connection failure | Retry with exponential backoff, then fail with descriptive message |
| Invalid input topology | Validate early at import; return specific error (missing ID, dangling edge, etc.) |
| Empty graph / layer | Return empty result with warning; do not raise exception |
| Algorithm non-convergence | Use fallback values (e.g., uniform PageRank), log warning |
| AHP inconsistency | Warn if CR > 0.10 but proceed; do not block analysis |
| Memory exhaustion | Stream large graphs in batches; log memory usage at large scale |

---

*Software-as-a-Graph Framework v2.0 · February 2026*