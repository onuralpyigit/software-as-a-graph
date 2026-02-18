# Software Design Description

## Software-as-a-Graph

### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 2.1** · **February 2026**

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

The design covers the full six-step methodology pipeline: graph model construction, structural analysis, quality scoring, failure simulation, statistical validation, and interactive visualization. The system follows a **three-layer pipeline architecture** (Presentation, Pipeline Components, Core) with dependency inversion at the repository boundary: the Core layer defines the `IGraphRepository` interface, and the Neo4j adapter implements it, keeping domain logic free of infrastructure dependencies.

### 1.3 References

| Reference | Description |
|-----------|-------------|
| IEEE 1016-2009 | IEEE Standard for Software Design Descriptions |
| SRS v2.1 | Software Requirements Specification for this project |
| STD v2.1 | Software and System Test Document for this project |
| IEEE RASSE 2025 | Published methodology paper (doi: 10.1109/RASSE64831.2025.11315354) |
| Neo4j Documentation | https://neo4j.com/docs/ |
| NetworkX Documentation | https://networkx.org/documentation/ |
| Saaty, T.L. (1980) | *The Analytic Hierarchy Process*, McGraw-Hill |

### 1.4 Document Conventions

- Design elements are identified by their module path (e.g., `src.analysis.service.AnalysisService`) for unambiguous cross-reference with source code.
- Pseudocode uses indented block notation; `→` means "produces" or "returns."
- Complexity annotations use standard Big-O notation.
- All mathematical symbols are defined at first use or in the Glossary (§1.5).
- Requirement cross-references use IDs from SRS v2.1 (e.g., REQ-GM-01).

### 1.5 Glossary

| Term | Definition |
|------|------------|
| AP | Articulation Point — a vertex whose removal disconnects the graph |
| AP\_c | Continuous articulation point score: fraction of graph fragmented upon vertex removal |
| BR | Bridge Ratio — fraction of a vertex's incident edges that are bridges |
| BT | Betweenness Centrality |
| CC | Clustering Coefficient |
| CI | Consistency Index in AHP: (λ\_max − n) / (n − 1) |
| CL | Closeness Centrality |
| CLI | Command-Line Interface |
| CR | Consistency Ratio in AHP: CI / RI |
| DG\_in / DG\_out | In-Degree / Out-Degree centrality |
| DTO | Data Transfer Object — plain data carrier between layers |
| EV | Eigenvector Centrality |
| GDS | Graph Data Science (Neo4j plugin) |
| IQR | Interquartile Range (Q3 − Q1) |
| NDCG | Normalized Discounted Cumulative Gain — ranking quality metric |
| PR / RPR | PageRank / Reverse PageRank |
| RI | Random Index for AHP consistency check |
| RMAV | Reliability, Maintainability, Availability, Vulnerability |
| SOLID | Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion |

### 1.6 Document Overview

Section 2 describes the system context, design constraints, and guiding principles. Section 3 covers the layered pipeline architecture, module decomposition, design patterns, data flow, and deployment. Sections 4–5 address data structures and component design (service pipelines). Section 6 provides algorithmic pseudocode and complexity analysis for all non-trivial algorithms. Section 7 covers the Neo4j database schema and key Cypher queries. Section 8 covers CLI and REST API interfaces with data exchange formats. Section 9 describes both visualization surfaces: the static HTML dashboard and the Genieus web application. Appendix A gives layer definitions, Appendix B the default AHP matrices, Appendix C the error handling strategy, and Appendix D provides SRS-to-design traceability.

---

## 2. System Overview

### 2.1 System Context

The framework offers two usage modes: a **CLI pipeline** for batch analysis and scripting, and a **Genieus web application** for interactive exploration. Both share the same Python domain logic and Neo4j storage.

```
                              ┌──────────────────┐
                              │      User        │
                              └─────┬──────┬─────┘
                                    │      │
                         CLI commands│      │Browser (HTTP)
                                    │      │
             ┌──────────────────────▼──┐  ┌▼───────────────────────┐
             │   CLI Pipeline (bin/)   │  │  Genieus Web App        │
             │                        │  │                         │
             │  Generate → Import →   │  │  Next.js Frontend       │
             │  Analyze → Simulate →  │  │  (port 7000)            │
             │  Validate → Visualize  │  │       │ HTTP             │
             │                        │  │  FastAPI Backend         │
             └───────────┬────────────┘  │  (port 8000)            │
                         │               └──────────┬──────────────┘
                         │                          │
                         │    ┌─────────────────────┘
                         │    │  Bolt / Python driver
                         ▼    ▼
               ┌────────────────────────┐          ┌──────────────────┐
               │   Neo4j Graph DB        │    ─────▶│  HTML Dashboard  │
               │   (port 7687 / 7474)   │          │   (file output)  │
               └────────────────────────┘          └──────────────────┘

Input formats: JSON topology (REQ-GM-01), GraphML topology (REQ-GM-02)
```

### 2.2 Design Constraints

| Constraint | Implication |
|------------|-------------|
| Neo4j 5.x required | Graph storage and GDS algorithm execution |
| Python 3.9+ | Type hints, dataclasses, walrus operator usage |
| Node.js 20+ | Next.js 15 frontend build and runtime |
| Memory-bound | Graph size limited by available RAM; target ≤ 1,000 components |
| NetworkX dependency | All centrality algorithms delegated to NetworkX |
| Static analysis only | No runtime instrumentation; input topology must be complete |

### 2.3 Design Principles

The system follows SOLID principles with emphasis on three key decisions:

**Separation of prediction from validation.** Steps 2–3 produce predicted scores Q(v) using only topology. Step 4 produces ground-truth scores I(v) using simulation. Step 5 compares the two. This separation prevents circular reasoning and ensures methodological rigor.

**Layered architecture with dependency inversion.** Domain logic (models, services, algorithms) has zero dependencies on infrastructure (Neo4j, file system). Infrastructure adapters implement domain-defined interfaces, making the core testable without a database.

**Composition over inheritance.** Services are composed via constructor injection. Classification strategies (box-plot, percentile), weight calculation strategies (default, AHP), and normalization strategies are interchangeable without modifying calling code.

---

## 3. System Architecture

### 3.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                           │
│                                                                     │
│  bin/run.py          bin/analyze_graph.py     bin/simulate_graph.py │
│  bin/generate_graph.py  bin/validate_graph.py  bin/visualize_graph.py│
│                                                                     │
│  External CLI entry points that parse arguments and invoke          │
│  pipeline components.                                               │
├─────────────────────────────────────────────────────────────────────┤
│                    WEB APPLICATION LAYER (Genieus)                  │
│                                                                     │
│  frontend/        (Next.js 15, port 7000)                           │
│  backend/api/     (FastAPI, port 8000)                              │
│                                                                     │
│  REST API exposes the same pipeline operations as the CLI.          │
│  Frontend calls API; API calls the same domain services as CLI.     │
├─────────────────────────────────────────────────────────────────────┤
│                        PIPELINE COMPONENTS                          │
│                                                                     │
│  src.analysis         src.simulation           src.validation       │
│  (Structural/Quality) (Event/Failure)          (Statistical)        │
│                                                                     │
│  src.visualization    src.generation           src.cli              │
│  (Dashboard/Charts)   (Synthetic Graphs)       (Shared Utils)       │
│                                                                     │
│  Feature-based packages implementing specific pipeline steps.       │
├─────────────────────────────────────────────────────────────────────┤
│                           CORE LAYER                                │
│                                                                     │
│  src.core                                                           │
│                                                                     │
│  - Domain Models (GraphData, ComponentData)                         │
│  - Interface (IGraphRepository)                                     │
│  - Implementation (Neo4jGraphRepository)                            │
│                                                                     │
│  Shared foundation for all pipeline components.                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Decomposition

```
software-as-a-graph/
├── bin/                              # Presentation Layer (CLI)
│   ├── run.py                        #   Pipeline orchestrator
│   ├── generate_graph.py             #   Synthetic data generation
│   ├── import_graph.py               #   Neo4j import
│   ├── analyze_graph.py              #   Analysis + quality scoring
│   ├── simulate_graph.py             #   Failure simulation
│   ├── validate_graph.py             #   Statistical validation
│   └── visualize_graph.py            #   Dashboard generation
│
├── frontend/                         # Web Application Layer — Next.js 15
│   ├── app/                          #   Next.js App Router pages
│   ├── components/                   #   React UI components
│   └── package.json                  #   Node.js dependencies
│
├── backend/
│   ├── api/                          # Web Application Layer — FastAPI
│   │   ├── main.py                   #   FastAPI app, CORS, health endpoint
│   │   ├── routers/                  #   Route handlers per domain area
│   │   └── schemas/                  #   Pydantic request/response models
│   └── src/                          # Pipeline Components + Core (shared with CLI)
│       ├── core/                     #   Core Layer
│       │   ├── models.py             #     Domain entities (GraphData, etc.)
│       │   ├── interfaces.py         #     IGraphRepository interface
│       │   ├── neo4j_repo.py         #     Graph database adapter
│       │   └── layers.py             #     Layer definitions
│       │
│       ├── analysis/                 #   Analysis Package
│       │   ├── analyzer.py           #     Backward-compatible AnalysisService wrapper
│       │   ├── service.py            #     AnalysisService pipeline
│       │   ├── structural_analyzer.py#     StructuralAnalyzer
│       │   └── quality_analyzer.py   #     QualityAnalyzer
│       │
│       ├── simulation/               #   Simulation Package
│       │   ├── service.py            #     SimulationService
│       │   ├── failure_simulator.py  #     FailureSimulator
│       │   └── event_simulator.py    #     EventSimulator
│       │
│       ├── validation/               #   Validation Package
│       │   ├── service.py            #     ValidationService
│       │   └── validator.py          #     Statistical validator
│       │
│       ├── visualization/            #   Visualization Package
│       │   ├── service.py            #     VisualizationService
│       │   └── dashboard.py          #     DashboardGenerator
│       │
│       ├── generation/               #   Generation Package
│       │   └── service.py            #     GenerationService
│       │
│       └── cli/                      #   CLI Utilities
│           └── console.py            #     ConsoleDisplay
│
├── config/                           # YAML scale presets
├── docs/                             # Methodology documentation
└── tests/                            # Unit and integration tests
```

> **Relationship between CLI and Web API:** The CLI scripts in `bin/` and the FastAPI routers in `backend/api/` both import from `backend/src/`. The domain logic is written once and exposed through two delivery mechanisms. Adding the FastAPI layer required no changes to the domain packages.

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
| **Adapter** | `Neo4jGraphRepository` | Translates domain calls to Cypher queries |

### 3.4 Data Flow Through the Pipeline

```
JSON / GraphML Topology
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

**Key:** Steps 2–3 operate on **G\_analysis** (derived DEPENDS\_ON edges only). Step 4 operates on **G\_structural** (all raw relationships) for realistic cascade propagation. This separation is deliberate — analysis needs abstracted dependencies for centrality, simulation needs physical topology for cascades.

### 3.5 Deployment Architecture

The full stack is containerized in a single multi-stage Docker image that starts all four services via a shell orchestration script.

```
┌───────────────────────────────────────────────────────────────┐
│                   Docker Container (genieus:x.y.z)            │
│                                                               │
│   ┌─────────────────┐   ┌─────────────────────────────────┐  │
│   │   Neo4j 5.x     │   │      Python Environment         │  │
│   │   port 7474 (HTTP)│  │                                 │  │
│   │   port 7687 (Bolt)│  │  FastAPI (uvicorn, 2 workers)   │  │
│   └─────────────────┘   │  port 8000                      │  │
│                         │                                 │  │
│   ┌─────────────────┐   │  CLI scripts (bin/)             │  │
│   │   Next.js 15    │   └─────────────────────────────────┘  │
│   │   port 7000     │                                        │
│   └─────────────────┘                                        │
│                                                               │
│   Startup order: Neo4j → (wait for readiness) → FastAPI → Next.js │
└───────────────────────────────────────────────────────────────┘

Port mapping (host → container):
  7474 → 7474  (Neo4j Browser)
  7687 → 7687  (Neo4j Bolt)
  8000 → 8000  (FastAPI REST API)
  7000 → 7000  (Next.js frontend)
```

**Multi-stage build** (3 stages):
1. **python-builder** — installs Python dependencies into a virtual environment.
2. **frontend-deps / frontend-builder** — runs `npm ci` and `next build` (produces standalone Next.js output).
3. **runtime** — copies the Python venv, Next.js standalone build, and CLI scripts into a single production image.

**Health check:** The container health check polls `http://localhost:8000/health` every 30 seconds; the container is marked unhealthy if FastAPI does not respond after 3 retries.

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

These are Python dataclasses that flow between services. All continuous metrics are normalized to [0, 1] via min-max scaling (see §6.8).

**StructuralMetrics** — output of Step 2, one per component. All 13 metrics of the output vector M(v) are listed:

| Field | Type | Metric Symbol | Description |
|-------|------|---------------|-------------|
| `pagerank` | float | PR(v) | Transitive importance (random-walk) |
| `reverse_pagerank` | float | RPR(v) | Reverse transitive importance |
| `betweenness` | float | BT(v) | Shortest-path bottleneck position |
| `closeness` | float | CL(v) | Average distance to all reachable vertices |
| `eigenvector` | float | EV(v) | Connection to important hubs |
| `in_degree` | float | DG\_in(v) | Normalized count of incoming DEPENDS\_ON edges |
| `in_degree_raw` | int | — | Raw (unnormalized) in-degree count |
| `out_degree` | float | DG\_out(v) | Normalized count of outgoing DEPENDS\_ON edges |
| `out_degree_raw` | int | — | Raw (unnormalized) out-degree count |
| `clustering_coefficient` | float | CC(v) | Neighbor interconnectedness |
| `ap_score` | float | AP\_c(v) | Continuous articulation point fragmentation score |
| `is_articulation_point` | bool | — | Binary flag: removal disconnects the graph |
| `bridge_ratio` | float | BR(v) | Fraction of incident edges that are bridges |
| `weight` | float | w(v) | Component's own QoS-derived weight |
| `weighted_in_degree` | float | w\_in(v) | Sum of weights of incoming DEPENDS\_ON edges (normalized) |
| `weighted_out_degree` | float | w\_out(v) | Sum of weights of outgoing DEPENDS\_ON edges (normalized) |

> **Note:** `ap_score`, `weighted_in_degree`, and `weighted_out_degree` are the three fields that complete the 13-metric vector M(v). The continuous `ap_score` is derived from the binary `is_articulation_point` detection via the reachability-loss formula in §6.4.

**QualityScores** — output of Step 3, one per component:

| Field | Type | Description |
|-------|------|-------------|
| `reliability` | float | R(v) — fault propagation risk |
| `maintainability` | float | M(v) — coupling complexity |
| `availability` | float | A(v) — SPOF risk |
| `vulnerability` | float | V(v) — security exposure |
| `overall` | float | Q(v) — composite quality score |
| `level` | str | Criticality level: CRITICAL / HIGH / MEDIUM / LOW / MINIMAL |

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
| `ranking` | RankingMetrics | Top-5 overlap, Top-10 overlap, NDCG@K |
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

### 5.1 Import Pipeline

```
CLI: import_graph.py  (or FastAPI POST /api/graph/import)
         │
         ▼
  Neo4jGraphRepository.save_graph(data, clear=True)
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

### 5.2 Analysis Pipeline

```
src.analysis.service.AnalysisService.analyze_layer(layer)
  (also exposed as FastAPI GET /api/analysis/{layer})
         │
    1. Export G_analysis(layer) from Neo4j → NetworkX DiGraph
    2. StructuralAnalyzer.analyze(graph)
         │  → Compute all 13 metrics per component (§6.1–§6.7, §6.8)
         │  → Compute continuous AP_c scores via iterated removal (§6.4)
         │  → Detect bridge edges (§6.6)
         │  → Return StructuralAnalysisResult
    3. QualityAnalyzer.analyze(structural_result)
         │  → Normalize metrics to [0, 1] via min-max (§6.8)
         │  → Compute RMAV scores using QualityWeights
         │  → Classify via BoxPlotClassifier or percentile fallback (§6.5)
         │  → Score edges using endpoint-aware RMAV formulas
         │  → Return QualityAnalysisResult
    4. ProblemDetector.detect(quality_result)        ← see §5.6
         │  → Identify architectural anti-patterns
         │  → Return List[DetectedProblem]
         │
         ▼
    Return LayerAnalysisResult
```

### 5.3 Simulation Pipeline

```
src.simulation.service.SimulationService.run_failure_simulation(layer)
  (also exposed as FastAPI POST /api/simulation/failure)
         │
    1. Build SimulationGraph from G_structural (all raw relationships)
    2. Compute baseline state (paths, components, topic weights)
    3. For each component v in layer:
         │
         a. Remove v from graph
         b. Propagate cascades (FailureSimulator, §6.7):
              Physical: Node fails → hosted Apps/Brokers fail
              Logical:  Broker fails → exclusively-routed Topics die
              Application: Publisher fails → starved Subscribers fail
              (repeat until no new failures — fixed-point iteration)
         c. Measure impact vs. baseline:
              ReachabilityLoss, Fragmentation, ThroughputLoss
         d. Record FailureResult with cascade sequence
         e. Restore v
         │
         ▼
    Return List[FailureResult] with I(v) for every component
```

### 5.4 Validation Pipeline

```
src.validation.service.ValidationService.validate_layer(analysis_result, simulation_results)
  (also exposed as FastAPI POST /api/validation)
         │
    1. Extract Q(v) from QualityAnalysisResult
    2. Extract I(v) from List[FailureResult]
    3. Align by component ID (warn on mismatches)
    4. Compute (Validator):
         - Correlation: Spearman ρ, Kendall τ, Pearson r
         - Classification: Precision, Recall, F1, Cohen's κ
         - Ranking: Top-5 overlap, Top-10 overlap, NDCG@K (§6.9)
         - Error: RMSE, MAE
    5. Evaluate against ValidationTargets → pass/fail per metric
         │
         ▼
    Return ValidationGroupResult
```

### 5.5 Visualization Pipeline

```
src.visualization.service.VisualizationService.generate_dashboard(layers, output_file)
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
         ... repeat for each requested layer ...
         │
         .start_section("Validation")
         .add_validation_metrics(pass/fail indicators)
         .end_section()
         │
         .generate() → complete HTML string
         │
         ▼
    Write HTML file; optionally open in browser
```

### 5.6 Anti-Pattern Detection Pipeline

`ProblemDetector` is invoked at the end of the Analysis Pipeline (§5.2, Step 4) and identifies four categories of architectural anti-patterns from `QualityAnalysisResult`:

| Anti-Pattern | Detection Rule | Severity |
|-------------|---------------|----------|
| **SPOF (Single Point of Failure)** | `ap_score > 0` — component's removal fragments the graph | CRITICAL |
| **God Component** | Q(v) > Q3 + 1.5×IQR **and** DG\_in + DG\_out > 75th percentile of degree distribution | HIGH |
| **Bottleneck Edge** | Edge betweenness > Q3 + 1.5×IQR of edge betweenness distribution | HIGH |
| **Systemic Risk Cluster** | ≥ 3 CRITICAL components with mutual DEPENDS\_ON edges forming a clique | CRITICAL |

Each detected problem is returned as a `DetectedProblem` dataclass:

```python
@dataclass
class DetectedProblem:
    pattern_type: str           # "SPOF" | "GOD_COMPONENT" | "BOTTLENECK_EDGE" | "SYSTEMIC_RISK"
    severity: str               # "CRITICAL" | "HIGH"
    component_ids: List[str]    # components involved
    description: str            # human-readable explanation
    recommendation: str         # suggested remediation
```

---

## 6. Algorithmic Design

### 6.1 PageRank

Measures transitive importance in the dependency graph.

```
Input:  G = (V, E), damping d = 0.85, max iterations = 100, tol = 1e-6
Output: PR[v] ∈ [0, 1] for all v

1. Initialize PR[v] = 1/|V| for all v
2. Repeat until ‖PR_new − PR‖₁ < tol × |V|:
     PR[v] = (1−d)/|V| + d × Σ PR[u]/out_degree(u)  for u ∈ in_neighbors(v)
3. Normalize to [0, 1] via min-max (§6.8)
```

**Reverse PageRank (RPR):** Computed identically on the transposed graph G^T where every edge A→B becomes B→A. High RPR(v) identifies components from which failure propagates outward — sources in dependency chains.

Complexity: O(|V| + |E|) per iteration. Delegated to `networkx.pagerank()`.

### 6.2 Betweenness Centrality (Brandes)

Measures bottleneck position — how often a component sits on the shortest path between other pairs.

```
Input:  G = (V, E)
Output: BT[v] ∈ [0, 1] for all v

1. For each source s ∈ V:
     a. BFS from s → shortest path counts σ[t], predecessors P[t]
     b. Backpropagate: δ[v] = Σ (σ[v]/σ[w]) × (1 + δ[w])  for w where v ∈ P[w]
     c. Accumulate: BT[v] += δ[v]
2. Normalize: BT[v] /= (|V|−1)(|V|−2)   [directed graph normalization]
```

Complexity: O(|V| × |E|). Delegated to `networkx.betweenness_centrality()`.

### 6.3 Closeness Centrality (Wasserman-Faust)

Measures how quickly a component can reach all others. The Wasserman-Faust variant handles directed graphs where some vertex pairs may be unreachable.

```
Input:  G = (V, E)
Output: CL[v] ∈ [0, 1] for all v

1. For each vertex v, BFS → reachable set R(v), distances dist(v, u) for u ∈ R(v)
2. CL(v) = (|R(v)| / (|V|−1)) × (|R(v)| / Σ dist(v,u))
                                              u ∈ R(v)
   The leading factor penalizes vertices that can reach only a small fraction of the graph.
   If R(v) = ∅, then CL(v) = 0.
```

Complexity: O(|V| × (|V| + |E|)). Delegated to `networkx.closeness_centrality()`.

### 6.4 Eigenvector Centrality

Measures connection quality — being connected to high-EV vertices contributes more than many connections to low-EV vertices.

```
Input:  G = (V, E), max_iter = 1000, tol = 1e-6
Output: EV[v] ∈ [0, 1] for all v

1. Initialize EV[v] = 1 for all v
2. Repeat until convergence:
     EV_new[v] = Σ EV[u]  for u ∈ in_neighbors(v)
     Normalize: EV = EV_new / ‖EV_new‖
3. If algorithm does not converge within max_iter iterations:
     WARN "Eigenvector centrality did not converge; falling back to in-degree centrality"
     EV[v] = DG_in[v]
```

Complexity: O(|V| + |E|) per iteration. Delegated to `networkx.eigenvector_centrality()`. The fallback to in-degree is safe because in-degree is the zeroth-order approximation of eigenvector centrality.

### 6.5 Articulation Point Detection and AP\_c Score

The binary Tarjan DFS identifies articulation points. The continuous AP\_c score quantifies the severity of fragmentation:

**Binary detection (Tarjan's algorithm):**

```
Input:  G = (V, E)
Output: Set of articulation points AP

1. DFS traversal, tracking:
     disc[v] = discovery time
     low[v]  = lowest discovery time reachable via DFS subtree
2. Vertex u is an articulation point if:
     (a) u is root of DFS tree and has ≥ 2 DFS children, OR
     (b) u is not root and ∃ child v where low[v] ≥ disc[u]
```

Complexity: O(|V| + |E|). The binary result populates `is_articulation_point`.

**Continuous score AP\_c:**

```
For each vertex v:
  G' = G with v and all incident edges removed
  AP_c(v) = 1 − |largest weakly connected component of G'| / (|V| − 1)

Interpretation:
  v not an AP → AP_c(v) = 0   (graph stays connected)
  v splits graph roughly in half → AP_c(v) ≈ 0.5
  v removal isolates all others → AP_c(v) → 1.0
```

Complexity: O(|V| × (|V| + |E|)) for all vertices.

### 6.6 Bridge Detection

A bridge is an edge whose removal disconnects the graph. For each vertex, Bridge Ratio BR(v) measures what fraction of its incident edges are bridges.

```
Input:  G = (V, E) treated as undirected for bridge computation
Output: Set of bridge edges B; BR[v] for all v

1. DFS-based identification: edge (u,v) is a bridge if low[v] > disc[u]
   (no back-edge from subtree of v reaches u or above)
2. BR(v) = |{e ∈ B : v ∈ e}| / degree_undirected(v)
   If degree(v) = 0, then BR(v) = 0
```

Complexity: O(|V| + |E|). Delegated to `networkx.bridges()`.

### 6.7 AHP Weight Calculation (Geometric Mean Method)

Derives quality dimension weights from expert pairwise comparison matrices.

```
Input:  n × n comparison matrix A (Saaty's 1–9 scale, A[j][i] = 1/A[i][j])
Output: Priority vector w (weights summing to 1.0), Consistency Ratio CR

1. Geometric mean per row: GM[i] = (∏ A[i][j])^(1/n)
2. Normalize: w[i] = GM[i] / Σ GM[k]
3. Consistency check:
     λ_max = mean of (A × w)[i] / w[i]
     CI    = (λ_max − n) / (n − 1)
     CR    = CI / RI[n]
     If CR > 0.10: ABORT with diagnostic — matrix is inconsistent (REQ-QS-08)
```

Random Index (RI) values (Saaty, 1980):

| n | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| RI | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 |

### 6.8 Min-Max Normalization

All raw metric values are normalized to [0, 1] before entering RMAV formulas.

```
Input:  Raw values x[v] for all v ∈ V
Output: Normalized values x_norm[v] ∈ [0, 1]

x_norm[v] = (x[v] − min(x)) / (max(x) − min(x))

Edge case: If max(x) = min(x) (all values identical):
  x_norm[v] = 0 for all v   (no discriminating power)
```

This normalization is applied independently per metric and per layer, so scores are relative within each analysis context.

### 6.9 Box-Plot Classification

Adaptive threshold classification that adjusts to actual Q(v) distributions.

```
Input:  Scores S = [Q(v₁), ..., Q(vₙ)], k = 1.5
Output: Criticality level per component

Normal path (n ≥ 12):
  Compute Q1, Median, Q3 from S
  IQR = Q3 − Q1
  upper_fence = Q3 + k × IQR
  Classify each Q(v):
    Q(v) > upper_fence → CRITICAL
    Q(v) > Q3          → HIGH
    Q(v) > Median      → MEDIUM
    Q(v) > Q1          → LOW
    Q(v) ≤ Q1          → MINIMAL

Fallback (n < 12): Fixed percentile thresholds
    top 10%  → CRITICAL
    top 25%  → HIGH
    top 50%  → MEDIUM
    top 75%  → LOW
    rest     → MINIMAL
```

### 6.10 Cascade Propagation

Simulates failure spread through the physical system graph G\_structural.

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

     Application cascade (if current is Publisher):
       For each subscriber starved of all publishers → add to F, enqueue

3. Repeat step 2 until fixed-point (no new failures)
4. Return F, S
```

In Monte Carlo mode, each cascade propagation step fires with probability p ∈ (0, 1], and N trials produce a distribution of I(v) values from which mean and variance are reported.

### 6.11 Spearman Rank Correlation

Measures whether predicted and actual rankings agree monotonically.

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

Delegated to `scipy.stats.spearmanr()`.

### 6.12 NDCG@K (Normalized Discounted Cumulative Gain)

Measures ranking quality with position-sensitive weighting — errors in top positions penalize more than errors at lower ranks.

```
Input:  Predicted ranking P, ground-truth relevance scores R, cutoff K
Output: NDCG@K ∈ [0, 1]

1. DCG@K = Σ  R[P[i]] / log₂(i + 2)   for i = 0..K-1
2. IDCG@K = DCG of ideal ranking (R sorted descending)
3. NDCG@K = DCG@K / IDCG@K

If IDCG@K = 0 (all relevance scores are 0): NDCG@K = 1.0 by convention.
```

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

// Derived (computed during import, Phase 4)
(source)-[:DEPENDS_ON {
    dependency_type: String,  // app_to_app | app_to_broker | node_to_node | node_to_broker
    weight: Float,
    shared_topics: Integer
}]->(target)
```

### 7.2 Key Cypher Queries

**Dependency derivation (app\_to\_app)**:

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

**Layer extraction (system — all dependency types)**:

```cypher
MATCH (a)-[d:DEPENDS_ON]->(b)
RETURN a.id, labels(a)[0] AS a_type, a.name, a.weight,
       b.id, labels(b)[0] AS b_type, b.name, d.weight, d.dependency_type
```

### 7.3 Indexes and Constraints

```cypher
// Uniqueness constraints (also serve as lookup indexes)
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Application) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (b:Broker)      REQUIRE b.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic)       REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node)        REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (l:Library)     REQUIRE l.id IS UNIQUE;

// Performance index for layer-based filtering
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

### 8.2 REST API Interface (FastAPI)

The FastAPI backend exposes the pipeline as a RESTful API, consumed by the Genieus frontend. The full interactive documentation is available at `http://localhost:8000/docs` (Swagger UI) when the container is running.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — returns `{"status": "ok"}` |
| GET | `/api/graph/summary` | Counts of all vertex and edge types in Neo4j |
| POST | `/api/graph/import` | Import JSON or GraphML topology into Neo4j |
| GET | `/api/analysis/{layer}` | Run structural analysis + quality scoring for a layer |
| POST | `/api/simulation/failure` | Run failure simulation for a layer |
| POST | `/api/validation` | Run statistical validation for a layer |
| GET | `/api/layers` | List available analysis layers |
| GET | `/api/components/{layer}` | Retrieve all component records for a layer |

All endpoints return JSON. Error responses follow RFC 7807 (Problem Details for HTTP APIs) with `status`, `title`, and `detail` fields.

### 8.3 Data Exchange Formats

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
    "levels": {"overall": "HIGH"},
    "metrics": {"pagerank": 0.82, "betweenness": 0.67,
                "ap_score": 0.50, "is_articulation_point": true}
  }],
  "classification_summary": {
    "total_components": 48,
    "distribution": {"CRITICAL": 5, "HIGH": 8, "MEDIUM": 15, "LOW": 12, "MINIMAL": 8}
  }
}
```

---

## 9. User Interface Design

The framework provides two distinct visualization surfaces. They share the same underlying data and visual encoding; the choice depends on whether interactive exploration or shareable reporting is needed.

### 9.1 Static HTML Dashboard

Generated by `bin/visualize_graph.py`. A single self-contained file (~1–3 MB) that requires no server — open directly in any browser. Uses vis.js for the network graph and Chart.js for charts.

**Dashboard layout:**

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
│  │ Criticality Dist.    │  │ RMAV Dimension Bars  │             │
│  │ [Pie Chart]          │  │ [Bar Chart]          │             │
│  └──────────────────────┘  └──────────────────────┘             │
│                                                                  │
│  [Component Detail Table — sortable, filterable]                 │
│                                                                  │
│  [Correlation Scatter Plot: Q(v) vs I(v)]                        │
│                                                                  │
│  [Interactive Network Graph — vis.js]                            │
│  (hover=tooltip, click=detail, drag=layout, zoom=magnify)        │
│                                                                  │
│  [Dependency Matrix Heatmap — sorted by criticality]             │
│                                                                  │
│  [Validation Report — Spearman ρ, F1, pass/fail badges]         │
└──────────────────────────────────────────────────────────────────┘
```

### 9.2 Genieus Web Application

Launched via `docker compose up`. A Next.js 15 frontend (port 7000) communicating with the FastAPI backend (port 8000). All pipeline operations can be triggered and viewed from the browser.

**Application structure — five tabs:**

| Tab | Primary View | Function |
|-----|-------------|----------|
| Dashboard | KPI cards, criticality pie, top-10 list, validation badges | High-level system health at a glance |
| Graph Explorer | Interactive 2D/3D force-directed dependency graph | Explore topology, filter by layer, inspect components |
| Analysis | Layer selector, weight mode toggle, real-time results | Trigger and view structural analysis |
| Simulation | Component selector, failure mode picker, cascade animation | Run and visualize failure simulations |
| Settings | Neo4j URI, credentials, database name | Connection configuration |

**Graph Explorer interaction model:**

- Filter by layer: app / infra / mw / system
- Search to highlight and center on a specific component
- Criticality overlay: nodes colored and sized by CRITICAL / HIGH / MEDIUM / LOW / MINIMAL
- Component type overlay: nodes shaped by Application / Broker / Node / Topic / Library
- Click a node: opens a side panel with full RMAV scores, I(v), criticality level, cascade count, and direct dependency list
- 2D / 3D toggle: three-dimensional layout for dense graphs (> 100 components)

**Visual encoding reference (shared across both surfaces):**

| Property | Encoding |
|----------|----------|
| CRITICAL | Red (dark) |
| HIGH | Orange |
| MEDIUM | Yellow |
| LOW | Light blue |
| MINIMAL | Grey |
| Node size | Proportional to Q(v) |
| Edge thickness | Proportional to DEPENDS\_ON weight |
| SPOF marker | Skull icon or dashed border |

---

## 10. Appendices

### Appendix A: Layer Definitions

| Layer ID | Name | Component Types | Dependency Types |
|----------|------|----------------|-----------------|
| `app` | Application Layer | Application | app\_to\_app |
| `infra` | Infrastructure Layer | Node | node\_to\_node |
| `mw-app` | Middleware-Application | Application, Broker | app\_to\_app, app\_to\_broker |
| `mw-infra` | Middleware-Infrastructure | Node, Broker | node\_to\_node, node\_to\_broker |
| `system` | Complete System | All types | All dependency types |

### Appendix B: Default AHP Matrices

```python
# Reliability: [PageRank, ReversePageRank, InDegree]
criteria_reliability = [
    [1.0, 2.0, 2.0],   # PR moderately more important than RPR and DGin
    [0.5, 1.0, 1.0],   # RPR and DGin equally important to each other
    [0.5, 1.0, 1.0],
]  # → GM: [1.587, 0.794, 0.794] → Normalized: [0.50, 0.25, 0.25]
   # Default used: [0.40, 0.35, 0.25]  (empirically smoothed; CR ≈ 0.00)

# Maintainability: [Betweenness, OutDegree, (1−Clustering)]
criteria_maintainability = [
    [1.0, 2.0, 3.0],
    [0.5, 1.0, 2.0],
    [0.33, 0.5, 1.0],
]  # → Normalized: [0.54, 0.30, 0.16]
   # Default used: [0.40, 0.35, 0.25]  (smoothed; CR ≈ 0.003)

# Availability: [AP_c, BridgeRatio, ComponentWeight w(v)]
criteria_availability = [
    [1.0, 3.0, 5.0],   # AP_c strongly dominant (structural SPOF is primary)
    [0.33, 1.0, 2.0],
    [0.2, 0.5, 1.0],
]  # → Normalized: [0.65, 0.23, 0.12]
   # Default used: [0.50, 0.30, 0.20]  (conservative — reduces AP_c dominance; CR ≈ 0.02)

# Vulnerability: [Eigenvector, Closeness, OutDegree]
criteria_vulnerability = [
    [1.0, 2.0, 2.0],
    [0.5, 1.0, 1.0],
    [0.5, 1.0, 1.0],
]  # → Normalized: [0.50, 0.25, 0.25]
   # Default used: [0.40, 0.30, 0.30]  (DGout elevated slightly; CR ≈ 0.00)

# Overall Q(v): [R, M, A, V] — equal by default
criteria_overall = [
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
]  # → weights = [0.25, 0.25, 0.25, 0.25]; CR = 0.00
```

> **Note on "Default used" vs. "Normalized" weights:** The normalized values come directly from the geometric mean of the pairwise matrix. The "default used" values are the production defaults embedded in the framework — they are empirically smoothed from the AHP result toward more balanced weighting to reduce sensitivity to outlier matrix entries. All defaults satisfy CR < 0.10 when checked against their respective matrices.

### Appendix C: Error Handling Strategy

| Error Type | Strategy |
|------------|----------|
| Neo4j connection failure | Retry with exponential backoff (3 attempts, 1/2/4 s delay), then raise with descriptive message including URI and credentials hint |
| Invalid input topology | Validate early at import; return specific error identifying missing ID, dangling edge, or duplicate component |
| Empty graph / layer | Return empty result with WARNING log; do not raise exception |
| Algorithm non-convergence (EV) | Fall back to in-degree centrality; log WARNING with component count |
| AHP matrix inconsistency (CR > 0.10) | **ABORT** with diagnostic message showing CR value and which dimension's matrix failed (REQ-QS-08). Do not proceed with inconsistent weights. |
| Memory exhaustion | Stream large graphs in batches; log memory usage at DEBUG level at large scale |
| GraphML parse error | Return specific line/element in error message; do not partially import |

> **AHP handling rationale:** Earlier versions warned but continued on AHP inconsistency. SRS REQ-QS-08 (v2.1) requires aborting, because proceeding with an inconsistent matrix would silently produce unreliable weights, invalidating the entire RMAV score. The analyst must revise pairwise judgments before proceeding.

### Appendix D: SRS-to-Design Traceability

| SRS Requirement | Design Element |
|-----------------|---------------|
| REQ-GM-01 | §5.1 Import Pipeline (JSON parser), §8.3 input topology JSON schema |
| REQ-GM-02 | §5.1 Import Pipeline (GraphML parser), §2.1 system context diagram |
| REQ-GM-03–05 | §5.1 Phase 1–3; §7.1 relationship types; §7.2 dependency derivation Cypher |
| REQ-GM-07, REQ-ML-01–04 | §5.2 Step 1 (layer export); Appendix A layer definitions |
| REQ-SA-01–02 | §6.1 PageRank / Reverse PageRank |
| REQ-SA-03 | §6.2 Betweenness Centrality (Brandes) |
| REQ-SA-04 | §6.3 Closeness Centrality (Wasserman-Faust) |
| REQ-SA-05 | §6.4 Eigenvector Centrality (power iteration + fallback) |
| REQ-SA-06 | §4.2 StructuralMetrics: in\_degree / out\_degree fields |
| REQ-SA-08 | §6.5 Articulation Point Detection (binary + continuous AP\_c) |
| REQ-SA-09 | §6.6 Bridge Detection and Bridge Ratio |
| REQ-SA-10, REQ-SA-11 | §4.2 StructuralMetrics weight fields; §6.8 min-max normalization |
| REQ-QS-01–05 | §5.2 Step 3 (QualityAnalyzer), Appendix B default AHP matrices |
| REQ-QS-07–08 | §6.7 AHP Weight Calculation; Appendix C AHP error handling |
| REQ-QS-09–10 | §6.9 Box-Plot Classification (normal path and fallback) |
| REQ-FS-01–07 | §5.3 Simulation Pipeline; §6.10 Cascade Propagation |
| REQ-VA-01 | §6.11 Spearman Rank Correlation |
| REQ-VA-02–06 | §5.4 Validation Pipeline (classification, ranking, error metrics) |
| REQ-VA-04, NDCG | §6.12 NDCG@K |
| REQ-VZ-01–08 | §5.5 Visualization Pipeline; §9.1 Static Dashboard |
| REQ-CLI-01–04 | §8.1 CLI Interface |
| REQ-SEC-01–03 | §2.2 Design Constraints; Appendix C (Neo4j connection error handling) |
| REQ-LOG-01–03 | Appendix C (logging strategy per error type) |
| REQ-PERF-01–04 | §3.5 Deployment Architecture (uvicorn 2 workers); §6.1–§6.11 complexity annotations |
| REQ-SCAL-01–02 | §2.2 Design Constraints (memory-bound) |
| REQ-PORT-01–02 | §3.5 Docker deployment (platform-agnostic container) |
| REQ-MAINT-01–03 | §3.3 Design Patterns (composition, strategy, repository) |

---

*Software-as-a-Graph Framework v2.1 · February 2026*
*Istanbul Technical University, Computer Engineering Department*