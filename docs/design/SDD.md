# Software Design Description

## Software-as-a-Graph

### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 2.3** · **March 2026**

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

The design covers the full pipeline: Import (graph model construction), Analyze (structural analysis + deterministic RMAV/Q scoring), Predict (optional inductive GNN forecasting), Simulate (failure simulation), Validate (statistical validation), and Visualize (interactive visualization). The system follows a **four-layer architecture** (Presentation, Web Application, Pipeline Components, Core) with dependency inversion at the repository boundary: the Core layer defines the `IGraphRepository` interface, and the Neo4j adapter implements it, keeping domain logic free of infrastructure dependencies.

The system is delivered through two mechanisms — a **CLI pipeline** (`bin/`) and a **Genieus web application** (FastAPI backend + Next.js frontend) — both of which invoke the same underlying domain packages.

### 1.3 References

| Reference | Description |
|-----------|-------------|
| IEEE 1016-2009 | IEEE Standard for Software Design Descriptions |
| SRS v2.2 | Software Requirements Specification for this project |
| STD v2.2 | Software and System Test Document for this project |
| IEEE RASSE 2025 | Published methodology paper (doi: 10.1109/RASSE64831.2025.11315354) |
| Neo4j Documentation | https://neo4j.com/docs/ |
| NetworkX Documentation | https://networkx.org/documentation/ |
| Saaty, T.L. (1980) | *The Analytic Hierarchy Process*, McGraw-Hill |

### 1.4 Document Conventions

- Design elements are identified by their module path (e.g., `src.analysis.service.AnalysisService`) for unambiguous cross-reference with source code.
- Pseudocode uses indented block notation; `→` means "produces" or "returns."
- Complexity annotations use standard Big-O notation.
- All mathematical symbols are defined at first use or in the Glossary (§1.5).
- Requirement cross-references use IDs from SRS v2.2 (e.g., REQ-GM-01).

### 1.5 Glossary

| Term | Definition |
|------|------------|
| AP | Articulation Point — a vertex whose removal disconnects the graph |
| AP\_c | Continuous articulation point score: fraction of graph fragmented upon vertex removal |
| BR | Bridge Ratio — fraction of a vertex's incident edges that are bridges |
| BT | Betweenness Centrality |
| CC | Clustering Coefficient |
| CDI | Connectivity Degradation Index — normalised increase in average path length when v is removed |
| CDPot | Cascade Depth Potential — `((RPR + DG_in) / 2) × (1 − min(DG_out / DG_in, 1))` |
| CI | Consistency Index in AHP: (λ\_max − n) / (n − 1) |
| CL | Closeness Centrality |
| CLI | Command-Line Interface |
| CouplingRisk | `1 − |2·Instability − 1|` where `Instability = DG_out / (DG_in + DG_out + ε)` |
| CR | Consistency Ratio in AHP: CI / RI |
| DG\_in / DG\_out | In-Degree / Out-Degree centrality |
| DTO | Data Transfer Object — plain data carrier between layers |
| EV | Eigenvector Centrality |
| GDS | Graph Data Science (Neo4j plugin) |
| IQR | Interquartile Range (Q3 − Q1) |
| NDCG | Normalized Discounted Cumulative Gain — ranking quality metric |
| PR / RPR | PageRank / Reverse PageRank |
| QADS | QoS-weighted Attack Dependent Surface — synonym for w\_in(v) when used in V(v) |
| QSPOF | QoS-weighted SPOF Severity — `AP_c_directed(v) × w(v)` |
| RCL | Reverse Closeness Centrality — closeness computed on G^T |
| REV | Reverse Eigenvector Centrality — eigenvector centrality computed on G^T |
| RI | Random Index for AHP consistency check |
| RMAV | Reliability, Maintainability, Availability, Vulnerability |
| SOLID | Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion |

### 1.6 Document Overview

Section 2 describes the system context, design constraints, and guiding principles. Section 3 covers the four-layer architecture, module decomposition, design patterns, data flow, and deployment. Sections 4–5 address data structures and component design (service pipelines). Section 6 provides algorithmic pseudocode and complexity analysis for all non-trivial algorithms, including the full RMAV formula derivations. Section 7 covers the Neo4j database schema and key Cypher queries. Section 8 covers CLI and REST API interfaces with data exchange formats. Section 9 describes both visualization surfaces: the static HTML dashboard and the Genieus web application. Appendix A gives layer definitions, Appendix B the default AHP matrices, Appendix C the error handling strategy, and Appendix D provides SRS-to-design traceability.

### 1.7 Change History

| 2.2 | February 2026 | Updated RMAV formulas to match implementation (§4.2, §5.2, §6, Appendix B); corrected architecture description to four-layer (§1.2, §3.1); fixed REST API endpoint paths to `/api/v1/` with correct HTTP methods (§8.2); added missing endpoints; added `benchmark/` to module decomposition (§3.2); added CDPot, CouplingRisk, QSPOF, AP_c_directed, CDI, REV, RCL algorithmic descriptions (§6.13–§6.18); corrected metric-to-dimension orthogonality table (§4.2); updated Appendix A layer IDs; updated Appendix B AHP matrices; extended Appendix D traceability for SRS v2.2 requirements |
| 2.3 | March 2026 | Refactored backend API to use **Presenters** for decoupled response formatting (§3.1, §3.2); updated module decomposition to include `backend/api/presenters/`; enhanced dependency injection in `backend/api/dependencies.py`; updated quality formulas to v2.3 (5-term Maintainability, QoS-weighted SPOF, CDPot_enh) |
| 2.4 | April 2026 | Clarified pipeline stage semantics: **Analyze** (deterministic, closed-form RMAV/Q scoring + anti-patterns) and **Predict** (inductive GNN forecasting, optional) are now named distinct stages. Updated §1.2, §2.1, §2.3, §3.4, §4.2 to reflect Import → Analyze → Predict → Simulate → Validate → Visualize naming. Updated `saag.Pipeline`, `saag.Client`, `saag.AnalysisResult`, and `saag.PredictionResult` SDK contracts accordingly. |

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
             │  Import → Analyze →    │  │  Next.js Frontend       │
             │  Predict → Simulate →  │  │  (port 7000)            │
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
| Node.js 20+ | Next.js 16 frontend build and runtime |
| Memory-bound | Graph size limited by available RAM; target ≤ 1,000 components |
| NetworkX dependency | All centrality algorithms delegated to NetworkX |
| Static analysis only | No runtime instrumentation; input topology must be complete |
| Docker required | Full-stack deployment and integration testing depend on Docker Compose |

### 2.3 Design Principles

The system follows SOLID principles with emphasis on three key decisions:

**Separation of scoring from validation.** The Analyze stage (Step 2) produces deterministic Q(v) from topology alone; the optional Predict stage (Step 3) refines to Q_ens(v) via GNN. The Simulate stage (Step 4) produces ground-truth I(v) independently. The Validate stage (Step 5) compares the two. This separation prevents circular reasoning and ensures methodological rigor.

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
│  frontend/        (Next.js 16, port 7000)                           │
│  backend/api/     (FastAPI, port 8000, /api/v1/ prefix)             │
│                                                                     │
│  REST API exposes the same pipeline operations as the CLI.          │
│  Frontend calls API; API calls the same domain services as CLI.     │
│  API uses **Presenters** to decouple domain logic from API response│
│  formatting, following the Hexagonal Architecture pattern.         │
├─────────────────────────────────────────────────────────────────────┤
│                        PIPELINE COMPONENTS                          │
│                                                                     │
│  src.analysis         src.simulation           src.validation       │
│  (Structural/Quality) (Event/Failure)          (Statistical)        │
│                                                                     │
│  src.visualization    tools.generation         tools.benchmark      │
│  (Dashboard/Charts)   (Synthetic Graphs)       (Benchmarking)       │
│                                                                     │
│  bin/common                                                         │
│  (Shared CLI Utilities)                                             │
│                                                                     │
│  Feature-based packages implementing specific pipeline steps.       │
├─────────────────────────────────────────────────────────────────────┤
│                           CORE LAYER                                │
│                                                                     │
│  src.core                                                           │
│                                                                     │
│  - Domain Models (GraphData, ComponentData)                         │
│  - Interface (IGraphRepository)                                     │
│  - Implementations (Neo4jGraphRepository, InMemoryGraphRepository)  │
│  - Layer Definitions                                                │
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
│   ├── visualize_graph.py            #   Dashboard generation
│   ├── export_graph.py               #   Export graph data from Neo4j
│   ├── benchmark.py                  #   Benchmarking across scales
│   ├── common/                       #   Shared CLI Utilities
│   │   ├── console.py                #     ConsoleDisplay (shared output formatting)
│   │   ├── dispatcher.py             #     Command dispatch logic
│   │   └── arguments.py              #     Shared argparse logic
│   └── run_scenarios.sh              #   Batch-run all scenario configs
│
├── frontend/                         # Web Application Layer — Next.js 16
│   ├── app/                          #   Next.js App Router pages
│   ├── components/                   #   React UI components
│   └── package.json                  #   Node.js dependencies
│
├── backend/
│   ├── api/                          # Web Application Layer — FastAPI
│   │   ├── main.py                   #   FastAPI app, CORS, health endpoint
│   │   ├── routers/                  #   REST endpoints (thin layer)
│   │   ├── presenters/               #   Response formatting & API translation
│   │   ├── dependencies.py           #   Service & Repository injection
│   │   └── models.py                 #   Pydantic request/response models
│   └── src/                          # Pipeline Components + Core (shared with CLI)
│       ├── core/                     #   Core Layer
│       │   ├── models.py             #     Domain entities (GraphData, ComponentData, EdgeData)
│       │   ├── interfaces.py         #     IGraphRepository interface
│       │   ├── neo4j_repo.py         #     Graph database adapter (Neo4j)
│       │   ├── memory_repo.py        #     In-memory adapter (testing)
│       │   └── layers.py             #     Layer definitions and projection rules
│       │
│       ├── analysis/                 #   Analysis Package
│       │   ├── analyzer.py           #     Backward-compatible AnalysisService wrapper
│       │   ├── service.py            #     AnalysisService pipeline orchestrator
│       │   ├── structural_analyzer.py#     StructuralAnalyzer (16-field metric computation)
│       │   └── quality_analyzer.py   #     QualityAnalyzer (RMAV scoring + classification)
│       │
│       ├── simulation/               #   Simulation Package
│       │   ├── service.py            #     SimulationService orchestrator
│       │   ├── failure_simulator.py  #     FailureSimulator (cascade propagation)
│       │   └── event_simulator.py    #     EventSimulator (message flow simulation)
│       │
│       ├── validation/               #   Validation Package
│       │   ├── service.py            #     ValidationService orchestrator
│       │   └── validator.py          #     Statistical validator (Spearman, F1, NDCG, …)
│       │
│       ├── visualization/            #   Visualization Package
│       │   ├── service.py            #     VisualizationService orchestrator
│       │   └── dashboard.py          #     DashboardGenerator (HTML builder)
│       │
│       ├── generation/               #   Generation Package
│       │   └── service.py            #     GenerationService (synthetic topology)
│       │
│       ├── benchmark/                #   Benchmark Package
│       │   └── service.py            #     BenchmarkService (scale performance testing)
│       │
│
├── config/                           # YAML scale presets and scenario configs
├── input/                            # Topology JSON & YAML scenario configs (8 scenarios)
├── output/                           # Pipeline output artefacts (dashboards, reports)
├── results/                          # Validation results from previous runs
├── benchmarks/                       # Benchmark data
├── examples/                         # Annotated Python usage examples
├── docs/                             # Methodology documentation
└── tests/                            # Pytest unit & integration tests (24 files)
```

> **Relationship between CLI and Web API:** The CLI scripts in `bin/` and the FastAPI routers in `backend/api/` both import from `backend/src/`. The domain logic is written once and exposed through two delivery mechanisms. Adding the FastAPI layer required no changes to the domain packages.

### 3.3 Design Patterns

| Pattern | Where Used | Purpose |
|---------|-----------|---------|
| **Facade** | `AnalysisService`, `SimulationService` | Simplified entry point to multi-step workflows |
| **Strategy** | `BoxPlotClassifier` / percentile fallback, `AHPProcessor` / default weights | Interchangeable classification and weight algorithms |
| **Builder** | `DashboardGenerator` | Incremental construction of HTML dashboards |
| **Repository** | `IGraphRepository` → `Neo4jGraphRepository`, `InMemoryGraphRepository` | Abstract data access behind a domain interface |
| **DTO** | `GraphData`, `ComponentData`, `EdgeData` | Clean data transfer between layers |
| **Factory** | `generate_graph()` | Synthetic topology generation with configurable scale |
| **Context Manager** | All database clients | Safe resource acquisition and release |
| **Adapter** | `Neo4jGraphRepository` | Translates domain calls to Cypher queries |

### 3.4 Data Flow Through the Pipeline

```
JSON / GraphML Topology
     │
     ▼
┌─────────────┐    ┌──────────────────────────┐    ┌─────────────────────┐
│  Import      │    │  Analyze                 │    │  Predict (optional) │
│  (Step 1)    │───▶│  Structural Analyzer     │───▶│  GNN Service        │
│              │    │  + Quality Analyzer      │    │  (Step 3)           │
│              │    │  + Anti-Pattern Detector │    │                     │
└─────────────┘    │  (Step 2)                │    └──────────┬──────────┘
                   └────────────┬─────────────┘               │
                                │                              │
                    G_analysis  Q(v) + RMAV                   Q_ens(v)
                                │        └────────────────────┘
                    G_structural │                Q(v) or Q_ens(v) predicted
                         │      │                        │
                         ▼      │                        │
                   ┌─────────────┐                      │
                   │  Simulate   │                      │
                   │  (Step 4)   │                      │
                   └──────┬──────┘                      │
                          │                             │
                     I(v) ground truth                  │
                          │                             │
                          ▼                             ▼
                   ┌─────────────────────────────────────┐
                   │        Validate (Step 5)             │
                   │   Compare Q(v)/Q_ens(v) vs I(v)      │
                   └──────────────┬──────────────────────┘
                                  │
                          All results
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │    Visualize (Step 6)        │
                   └─────────────────────────────┘
```

**Key:** Steps 2–3 (Analyze and Predict) operate on **G\_analysis** (derived DEPENDS\_ON edges only). Step 4 (Simulate) operates on **G\_structural** (all raw relationships) for realistic cascade propagation. This separation is deliberate — analysis needs abstracted dependencies for centrality, simulation needs physical topology for cascades.

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
│   │   Next.js 16    │   └─────────────────────────────────┘  │
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
                 │ durability  │  → PERSISTENT | TRANSIENT | TRANSIENT_LOCAL | VOLATILE
                 │ priority    │  → URGENT | HIGH | MEDIUM | LOW
                 │ msg_size    │  → bytes
                 └─────────────┘
```

### 4.2 Metrics Data Structures

These are Python dataclasses that flow between services. All continuous metrics are normalized to [0, 1] via min-max scaling (see §6.8).

**StructuralMetrics** — output of the Analyze stage structural sub-phase (Step 2), one per component. The output vector M(v) has up to 20 fields:

| Field | Type | Symbol | Description | RMAV Usage |
|-------|------|--------|-------------|------------|
| `pagerank` | float | PR(v) | Transitive importance (random-walk) | Diagnostic |
| `reverse_pagerank` | float | RPR(v) | Transitive reachability | R(v) |
| `betweenness` | float | BT(v) | Shortest-path bottleneck position | M(v) |
| `closeness` | float | CL(v) | Average distance (forward) | Diagnostic |
| `reverse_closeness`| float | RCL(v) | Average distance (reverse) | V(v) |
| `eigenvector` | float | EV(v) | Connection to important hubs | Diagnostic |
| `reverse_eigenvector`| float | REV(v) | Connection from important dependents | V(v) |
| `in_degree_raw` | int | — | Count of incoming edges | R(v) (norm) |
| `out_degree_raw` | int | — | Count of outgoing edges | M/V (norm) |
| `clustering_coeff` | float | CC(v) | Local neighbor interconnectedness | M(v) |
| `ap_score` | float | AP_c(v) | Continuous articulation score (undirected) | A(v) |
| `bridge_ratio` | float | BR(v) | Fraction of incident edges that are bridges | A(v) |
| `weight` | float | w(v) | Component's own QoS weight | A(v) |
| `w_in` | float | w_in(v) | QoS-weighted in-degree | V(v) (QADS) |
| `w_out` | float | w_out(v) | QoS-weighted out-degree | M(v) |
| `cdi` | float | CDI | Connectivity Degradation Index | A(v) |
| `mpci` | float | MPCI | Multi-Path Coupling Intensity | R(v) (CDPot) |
| `foc` | float | FOC | Fan-Out Criticality (Topics) | R(v) (Topics) |
| `cqp` | float | CQP | Code Quality Penalty (Applications) | M(v) |
| `ap_c_dir` | float | AP_c_d | Directed articulation point score | A(v) |

> **On metric count:** The output vector M(v) has expanded from 16 to 20 fields in version 2.3 to include refined signals like MPCI, FOC, CQP, and AP_c_dir directly in the analysis step. All 20 fields are present in the `StructuralMetrics` dataclass and are available for both RMAV and GNN prediction paths.

**Metric-to-Dimension Orthogonality:** Each raw metric from the 20-field vector M(v) feeds **exactly one** RMAV dimension. No metric appears in more than one formula.

| Metric | Symbol | R | M | A | V | Notes |
|--------|--------|:-:|:-:|:-:|:-:|-------|
| Reverse PageRank | RPR | ✓ | | | | Global cascade reach |
| In-Degree | DG_in | ✓ | | | | Immediate blast radius |
| MPCI | MPCI | ✓ | | | | Multi-path coupling (via CDPot_enh) |
| Fan-Out Criticality | FOC | ✓ | | | | Topics subscriber fan-out |
| Betweenness | BT | | ✓ | | | Structural bottleneck position |
| QoS-Weighted Out-Degree | w_out | | ✓ | | | SLA-weighted efferent coupling |
| Code Quality Penalty | CQP | | ✓ | | | Complexity + instability + LCOM |
| Coupling Risk | CR | | ✓ | | | Afferent/efferent imbalance |
| Clustering Coefficient | CC | | ✓ | | | Inverse redundancy (1-CC) |
| Directed AP Score | AP_c_d | | | ✓ | | Directed articulation score |
| Bridge Ratio | BR | | | ✓ | | Network-level SPOF fraction |
| CDI | CDI | | | ✓ | | Path elongation on removal |
| QoS-Weighted SPOF | QSPOF | | | ✓ | | Case-weighted SPOF (AP_c_d * weight) |
| Reverse Eigenvector | REV | | | | ✓ | Strategic exposure (downstream) |
| Reverse Closeness | RCL | | | | ✓ | Propagation speed (upstream) |
| QoS-Weighted In-Degree | w_in | | | | ✓ | Attack surface (QADS) |
| PageRank | PR | — | — | — | — | Diagnostic only |
| Closeness | CL | — | — | — | — | Diagnostic only |
| Eigenvector | EV | — | — | — | — | Diagnostic only |
| Out-Degree | DG_out | — | — | — | — | Diagnostic only |
| PageRank | PR | — | — | — | — | Reported only |
| Closeness | CL | — | — | — | — | Reported only |
| Eigenvector | EV | — | — | — | — | Reported only |

**QualityScores** — output of the Analyze stage (Step 2), one per component:

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
CLI: import_graph.py  (or FastAPI POST /api/v1/graph/import)
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
  (also exposed as FastAPI POST /api/v1/analysis/layer/{layer})
         │
    1. Export G_analysis(layer) from Neo4j → NetworkX DiGraph
    2. StructuralAnalyzer.analyze(graph)
         │  → Compute all 16 metric fields per component (§6.1–§6.6, §6.8)
         │  → Compute reverse-graph metrics (RPR, REV, RCL) on G^T (§6.1, §6.18)
         │  → Compute continuous AP_c scores via iterated removal (§6.5)
         │  → Detect bridge edges (§6.6)
         │  → Return StructuralAnalysisResult
    3. PredictionEngine.analyze(structural_result)       ← RMAV + optional GNN
         │  → Normalize metrics to [0, 1] via min-max (§6.8)
         │  → Compute derived terms: CDPot (§6.13), CouplingRisk (§6.14),
         │    QSPOF (§6.15), AP_c_directed (§6.16), CDI (§6.17)
         │  → Compute RMAV scores using QualityWeights (§6.19–§6.22)
         │  → Compute composite Q(v) (§6.23)
         │  → Classify via BoxPlotClassifier or percentile fallback (§6.9)
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
  (also exposed as FastAPI POST /api/v1/simulation/failure)
         │
    1. Build SimulationGraph from G_structural (all raw relationships)
    2. Compute baseline state (paths, components, topic weights)
    3. For each component v in layer:
         │
         a. Remove v from graph
         b. Propagate cascades (FailureSimulator, §6.10):
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
  (also exposed as FastAPI POST /api/v1/validation/run-pipeline)
         │
    1. Extract Q(v) from QualityAnalysisResult
    2. Extract I(v) from List[FailureResult]
    3. Align by component ID (warn on mismatches; require n ≥ 5)
    4. Compute (Validator):
         - Correlation: Spearman ρ, Kendall τ, Pearson r
         - Classification: Precision, Recall, F1, Cohen's κ
         - Ranking: Top-5 overlap, Top-10 overlap, NDCG@K (§6.12)
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

The `AntiPatternDetector` (consolidated from the legacy `SmellDetector` and `ProblemDetector`) is invoked at the end of the Analysis Pipeline (§5.2, Step 4). It identifies 12 categories of architectural anti-patterns from `QualityAnalysisResult` using box-plot classification levels (never static thresholds).

| Category | Anti-Pattern | Detection Rule (Predicate) | Severity |
|:---------|:-------------|:---------------------------|:---------|
| **Avail.** | **SPOF** | `is_articulation_point == True` | CRITICAL |
| **Avail.** | **Bridge Edge** | `is_bridge == True` | HIGH |
| **Reliab.** | **Failure Hub** | `reliability_level >= CRITICAL` | CRITICAL |
| **Reliab.** | **Concentration Risk** | Top-3 PageRank components hold > 50% importance | MEDIUM |
| **Maint.** | **God Component** | `maintainability_level >= CRITICAL` and `betweenness > 0.3` | CRITICAL |
| **Maint.** | **Hub-and-Spoke** | `clustering < 0.1` and `degree > 3` | MEDIUM |
| **Maint.** | **Bottleneck Edge** | `edge_betweenness > 0.2` and `edge_level >= HIGH` | MEDIUM |
| **Security** | **Target** | `vulnerability_level >= CRITICAL` | CRITICAL |
| **Security** | **Exposure** | `vulnerability_level == HIGH` and `closeness > 0.6` | HIGH |
| **Arch.** | **Cycle** | Circular dependency path of length ≥ 2 | HIGH |
| **Arch.** | **Chain** | Linear sequence of ≥ 4 nodes with in/out degree ≤ 1 | MEDIUM |
| **Arch.** | **Isolated** | Node with zero in-layer dependencies | MEDIUM |
| **Maint.** | **Unstable Intf.** | `coupling_risk_enh_level >= HIGH` | MEDIUM |

Each detected problem is returned as a `DetectedProblem` dataclass (defined in `src.prediction.models`):

```python
@dataclass
class DetectedProblem:
    entity_id: str             # ID of component, edge, or "SYSTEM"
    entity_type: str           # "Component" | "Edge" | "Architecture" | "System"
    category: str              # Category (Availability, Reliability, etc.)
    severity: str              # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    name: str                  # Short name (e.g., "God Component")
    description: str           # Contextual explanation
    recommendation: str        # Suggested remediation steps
    evidence: Dict[str, Any]   # Underlying metrics and state
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

**Reverse PageRank (RPR):** Computed identically on the transposed graph G^T where every edge A→B becomes B→A. High RPR(v) identifies components from which failure propagates outward — sources in dependency chains. RPR feeds into R(v) (§6.19).

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

> **Note:** CL is reported in the output but does not directly enter any RMAV formula. The Vulnerability dimension uses **Reverse Closeness (RCL)** — closeness computed on G^T — to capture how rapidly dependents can reach v.

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

> **Note:** EV is reported in output but does not directly enter any RMAV formula. The Vulnerability dimension uses **Reverse Eigenvector (REV)** — EV computed on G^T — to capture strategic exposure through downstream dependencies.

### 6.5 Articulation Point Detection and AP\_c Score

The binary Tarjan DFS identifies articulation points. The continuous AP\_c score quantifies the severity of fragmentation:

**Binary detection (Tarjan's algorithm):**

```
Input:  G = (V, E) treated as undirected for standard AP detection
Output: Set of articulation points AP

1. DFS traversal, tracking:
     disc[v] = discovery time
     low[v]  = lowest discovery time reachable via DFS subtree
2. Vertex u is an articulation point if:
     (a) u is root of DFS tree and has ≥ 2 DFS children, OR
     (b) u is not root and ∃ child v where low[v] ≥ disc[u]
```

Complexity: O(|V| + |E|). The binary result populates `is_articulation_point`.

**Continuous score AP\_c (undirected):**

```
For each vertex v:
  G' = G with v and all incident edges removed
  AP_c(v) = 1 − |largest weakly connected component of G'| / (|V| − 1)

Interpretation:
  v not an AP → AP_c(v) = 0   (graph stays connected)
  v splits graph roughly in half → AP_c(v) ≈ 0.5
  v removal isolates all others → AP_c(v) → 1.0
```

Complexity: O(|V| × (|V| + |E|)) for all vertices. This undirected AP_c is stored in `ap_score`. The directed variant `AP_c_directed` is derived in §6.16.

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
Input:  Paired values (X, Y) of size n ≥ 5
Output: Correlation coefficient ρ ∈ [−1, 1], p-value

1. Rank X → rank_X;  Rank Y → rank_Y  (ties → average rank)
2. d[i] = rank_X[i] − rank_Y[i]
3. ρ = 1 − (6 × Σ d[i]²) / (n × (n² − 1))
4. t = ρ × √(n − 2) / √(1 − ρ²)
5. p-value from t-distribution with (n−2) degrees of freedom

Requires n ≥ 5; raises ValueError and logs warning for n < 5.
```

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

### 6.13 Cascade Depth Potential (CDPot_enh)

CDPot_enh measures the self-reinforcement depth of a cascade, amplified by multi-path coupling intensity.

```
Input:  RPR(v), DG_in(v), DG_out(v), MPCI(v) [all normalized]
Output: CDPot_enh(v) ∈ [0, 1]

CDPot_base(v) = ((RPR(v) + DG_in(v)) / 2) × (1 − min(DG_out(v) / (DG_in(v) + ε), 1.0))
CDPot_enh(v)  = min(CDPot_base(v) × (1 + MPCI(v)), 1.0)
```

Interpretation:
  - Fan-out hubs (DG_out >> DG_in): CDPot → 0 (wide, shallow cascade; quickly absorbed)
  - Absorber nodes (DG_in >> DG_out) with high MPCI: CDPot_enh is maximized (deep, multi-channel cascade)

CDPot_enh is a Tier 1 derived signal that feeds exclusively into R(v) (§6.19).

### 6.14 Coupling Risk (Enhanced)

Measures the structural coupling imbalance enriched by topological path complexity. Components that are deeply embedded on both afferent and efferent sides and have complex communication paths are hardest to modify safely.

```
Input:  DG_in_raw(v), DG_out_raw(v)  [raw integer counts]
Output: CouplingRisk_enh(v) ∈ [0, 1]

Instability(v) = DG_out_raw(v) / (DG_in_raw(v) + DG_out_raw(v) + ε)

CouplingRisk_base(v) = 1 − |2 × Instability(v) − 1|

CouplingRisk_enh(v) = min(1.0, CouplingRisk_base(v) × (1 + Δ × path_complexity(v)))
```

`CouplingRisk_enh` is computed inline in `QualityAnalyzer` and feeds exclusively into $M(v)$ (§6.20).

### 6.15 QoS-Weighted SPOF Severity (QSPOF)

Scales the directed articulation point score by the component's QoS weight, so high-criticality SPOF components dominate availability risk.

```
Input:  AP_c_directed(v) (§6.16), w(v)  [normalized QoS weight]
Output: QSPOF(v) ∈ [0, 1]

QSPOF(v) = AP_c_directed(v) × w(v)
```

QSPOF feeds exclusively into A(v) (§6.21).

### 6.16 Directed Articulation Point Score (AP\_c\_directed)

Extends the undirected AP\_c to capture directional path disruption in a directed graph. Uses the worst-case direction (out-reachability vs. in-reachability) upon vertex removal.

```
Input:  G = (V, E) [directed], vertex v
Output: AP_c_directed(v) ∈ [0, 1]

1. G' = G with v removed
2. AP_c_out(v) = 1 − |R_out largest| / (|V| − 1)
   where R_out largest = largest strongly reachable component from any remaining vertex
3. AP_c_in(v) = 1 − |R_in largest| / (|V| − 1)
   where R_in largest = largest set reaching any single vertex
4. AP_c_directed(v) = max(AP_c_out(v), AP_c_in(v))
```

AP_c_directed feeds into QSPOF (§6.15) and directly into A(v) (§6.21).

### 6.17 Connectivity Degradation Index (CDI)

Measures average path elongation when v is removed — catches non-articulation vulnerability where no partition occurs but paths become significantly longer.

```
Input:  G = (V, E), vertex v
Output: CDI(v) ∈ [0, 1]

1. Compute APSP (All-Pairs Shortest Paths) on G → avg_path_G
2. Compute APSP on G' (G with v removed) → avg_path_G'
   (pairs involving v are excluded from both computations)
3. CDI(v) = min((avg_path_G' − avg_path_G) / avg_path_G, 1.0)
   If G' is disconnected: avg_path_G' = ∞ → CDI(v) = 1.0

Edge case: If |V| ≤ 2 after removal: CDI(v) = 0
```

Complexity: O(|V| × (|V| + |E|)) via BFS for each APSP. For enterprise-scale systems (|V| > 300), sampled APSP (random subset of source nodes) is used with a fixed seed for reproducibility.

CDI feeds exclusively into A(v) (§6.21).

### 6.18 Reverse Centrality Metrics (REV, RCL)

Reverse Eigenvector (REV) and Reverse Closeness (RCL) are computed by running the standard eigenvector and closeness algorithms on the **transposed graph G^T** (all edges reversed).

```
G^T = transpose(G)   [edge A→B in G becomes B→A in G^T]

REV(v) = eigenvector_centrality(G^T, max_iter=1000)[v]
         (with in-degree fallback on non-convergence, as per §6.4)

RCL(v) = closeness_centrality(G^T, Wasserman-Faust)[v]
```

**Semantic interpretation:**
- **High REV(v):** v receives influence from other high-REV components in the reversed graph — meaning v is connected to downstream critical hubs in the original graph. This signals strategic exposure to dependent failures.
- **High RCL(v):** v is "close" to many components in G^T — meaning many components can reach v quickly in the original graph. Adversarial paths from dependents are short.

Both REV and RCL feed exclusively into V(v) (§6.22).

### 6.19 Reliability Score R(v)

```
R(v) = 0.45 × RPR(v) + 0.30 × DG_in(v) + 0.25 × CDPot_enh(v)
```

| Term | Weight | Rationale |
|------|--------|-----------|
| RPR(v) | 0.45 | Reverse PageRank — global cascade reach; how broadly v's failure propagates in the reverse-dependency direction |
| DG_in(v) | 0.30 | In-degree — count of direct dependents; immediate structural blast radius |
| CDPot_enh(v) | 0.25 | Enhanced Cascade Depth Potential — amplified by MPCI; captures depth and multi-channel coupling risk |

A component with high R(v) is one whose failure would propagate both broadly and deeply through the dependency graph.

### 6.20 Maintainability Score M(v)

```
M(v) = 0.35 × BT(v) + 0.30 × w_out(v) + 0.15 × CQP(v) + 0.12 × CouplingRisk_enh(v) + 0.08 × (1 − CC(v))
```

| Term | Weight | Rationale |
|------|--------|-----------|
| BT(v) | 0.35 | Betweenness — structural bottleneck position; v lies on many dependency paths |
| w_out(v) | 0.30 | QoS-weighted efferent coupling — counts outgoing dependencies weighted by SLA priority |
| CQP(v) | 0.15 | Code Quality Penalty — composite of complexity, instability, and LCOM |
| CouplingRisk_enh(v) | 0.12 | Afferent/efferent imbalance enriched by path complexity — components embedded on both sides are hardest to change safely |
| (1 − CC(v)) | 0.08 | Inverse clustering — low local redundancy means connection is a unique structural link |

A component with high M(v) is a structural bottleneck that has many tightly-contracted outgoing dependencies and sits at an unstable coupling boundary.

### 6.21 Availability Score A(v)

```
A(v) = 0.35 × AP_c_directed(v) + 0.25 × QSPOF(v) + 0.25 × BR(v) + 0.10 × CDI(v) + 0.05 × w(v)
```

| Term | Weight | Rationale |
|------|--------|-----------|
| AP_c_directed(v) | 0.35 | Directed articulation score — primary structural SPOF baseline; identifies bottlenecks regardless of operational priority |
| QSPOF(v) | 0.25 | QoS-weighted SPOF severity — `AP_c_directed × w(v)`; highlights high-priority structural SPOFs |
| BR(v) | 0.25 | Bridge ratio — fraction of incident edges that are bridges; identifies irreplaceable structural foundations |
| CDI(v) | 0.10 | Connectivity degradation — path elongation upon removal catches non-SPOF availability risk |
| w(v) | 0.05 | Component QoS weight — direct operational priority focus for availability focus |

### 6.22 Vulnerability Score V(v)

```
V(v) = 0.40 × REV(v) + 0.35 × RCL(v) + 0.25 × w_in(v)
```

| Term | Weight | Rationale |
|------|--------|-----------|
| REV(v) | 0.40 | Reverse Eigenvector — connection to downstream critical components signals strategic exposure |
| RCL(v) | 0.35 | Reverse Closeness — short paths from dependents means vulnerabilities propagate rapidly to v |
| w\_in(v) (QADS) | 0.25 | QoS-weighted in-degree — immediate attack surface exposed to dependents, weighted by their SLA priority |

### 6.23 Composite Quality Score Q(v)

```
Q(v) = w_R × R(v) + w_M × M(v) + w_A × A(v) + w_V × V(v)
```

**Default weights:** w\_R = w\_M = w\_A = w\_V = 0.25 (equal weighting). The equal default reflects deliberate expert judgment — an AHP pairwise comparison matrix where all dimensions are rated equally important produces exactly this 0.25/0.25/0.25/0.25 split, with CR = 0.00. For domain-specific priorities (security-critical, high-availability, actively-developed systems), custom AHP matrices are supported via `--use-ahp` (see Appendix B for domain examples).

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
python bin/run.py --all --layer system --verbose
```

**Individual step CLIs** — all share common flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--layer LAYER` | Target layer: app, infra, mw, system | system |
| `--uri URI` | Neo4j connection URI | bolt://localhost:7687 |
| `--user` / `--password` | Neo4j credentials | neo4j / password |
| `--output FILE` | Export results to JSON | — |
| `--verbose` / `--quiet` | Log level control (DEBUG / WARNING) | INFO |
| `--scale SCALE` | Scale preset for generation: tiny, small, medium, large, xlarge | medium |
| `--open` | Open generated HTML dashboard in browser after completion | false |

### 8.2 REST API Interface (FastAPI)

The FastAPI backend exposes the pipeline as a versioned RESTful API (`/api/v1/` prefix), consumed by the Genieus frontend. The full interactive documentation is available at `http://localhost:8000/docs` (Swagger UI) when the container is running.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — returns `{"status": "ok"}` |
| GET | `/api/v1/graph/summary` | Counts of all vertex and edge types in Neo4j |
| POST | `/api/v1/graph/import` | Import JSON or GraphML topology into Neo4j |
| GET | `/api/v1/graph/search-nodes` | Search for nodes by name/type query parameter |
| POST | `/api/v1/analysis/layer/{layer}` | Run structural analysis + quality scoring for a layer |
| POST | `/api/v1/simulation/failure` | Run failure simulation for a layer |
| POST | `/api/v1/validation/run-pipeline` | Run statistical validation for a layer |
| GET | `/api/v1/validation/layers` | List available analysis layers with metadata |
| GET | `/api/v1/components` | Retrieve all component records for a layer (query param) |

All endpoints return JSON. Error responses follow RFC 7807 (Problem Details for HTTP APIs) with `status`, `title`, and `detail` fields. HTTP 422 is returned for invalid request payloads.

> **API versioning note:** All domain endpoints use the `/api/v1/` prefix. The health endpoint `/health` is unversioned (infrastructure concern). The Swagger documentation at `/docs` covers all versioned endpoints.

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
    "metrics": {"reverse_pagerank": 0.76, "betweenness": 0.67,
                "ap_score": 0.50, "is_articulation_point": true,
                "bridge_ratio": 0.33, "weighted_in_degree": 0.61}
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

Launched via `docker compose up`. A Next.js 16 frontend (port 7000) communicating with the FastAPI backend (port 8000). All pipeline operations can be triggered and viewed from the browser.

**Application structure — five tabs:**

| Tab | Primary View | Function |
|-----|-------------|----------|
| Dashboard | KPI cards, criticality pie, top-10 list, validation badges | High-level system health at a glance |
| Graph Explorer | Interactive 2D/3D force-directed dependency graph | Explore topology, filter by layer, inspect components |
| Analysis | Layer selector, weight mode toggle, real-time results | Trigger and view structural analysis |
| Simulation | Component selector, failure mode picker, cascade animation | Run and visualize failure simulations |
| Settings | Neo4j URI, credentials, database name | Connection configuration (persisted in browser storage) |

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
| `mw` | Middleware Layer | Application, Broker | app\_to\_app, app\_to\_broker |
| `system` | Complete System | All types | All dependency types |

> **Note:** Earlier versions of this document used `mw-app` and `mw-infra` as separate layer IDs. These have been consolidated into a single `mw` layer that covers all middleware dependencies. The `mw` layer is what the SRS, STD, and CLI `--layer mw` flag all refer to.

### Appendix B: Default AHP Matrices

The production defaults are empirically smoothed from the AHP geometric-mean result toward more balanced weighting to reduce sensitivity to outlier matrix entries. All defaults satisfy CR < 0.10.

```python
# Reliability R(v): inputs = [RPR, DG_in, CDPot_enh]
# AHP judgment: RPR (reach) > DG_in (immediate) > CDPot (depth)
criteria_reliability = [
    [1.0,  1.5,   2.0],  # RPR: reach is primary
    [0.67, 1.0,   1.5],  # DG_in: immediate dependents
    [0.5,  0.67,  1.0],  # CDPot: cascade depth (secondary)
]
# → Normalized: [0.45, 0.30, 0.25]; CR ≈ 0.001

# Maintainability M(v): inputs = [BT, w_out, CQP, CouplingRisk, (1-CC)]
# AHP judgment: structural bottleneck (BT) and contracts (w_out) are dominant
criteria_maintainability = [
    # BT    w_out   CQP    CR    (1-CC)
    [1.0,  1.17,  2.33,  2.92,  4.38],
    [0.86, 1.0,   2.0,   2.5,   3.75],
    [0.43, 0.5,   1.0,   1.25,  1.88],
    [0.34, 0.4,   0.8,   1.0,   1.5],
    [0.23, 0.27,  0.53,  0.67,  1.0],
]
# → Normalized: [0.35, 0.30, 0.15, 0.12, 0.08]; CR ≈ 0.000 (perfectly consistent)

# Availability A(v): inputs = [QSPOF, BR, AP_c_dir, CDI]
# AHP judgment: QSPOF strongly dominant; BR significant; AP_c_dir and CDI supporting
criteria_availability = [
    [1.0,  3.0,  5.0,  9.0],
    [0.33, 1.0,  2.0,  4.0],
    [0.2,  0.5,  1.0,  3.0],
    [0.11, 0.25, 0.33, 1.0],
]
# → Normalized (after λ=0.7 shrinkage): [0.45, 0.30, 0.15, 0.10]

# Vulnerability V(v): inputs = [REV, RCL, QADS]
# AHP judgment: REV reach > RCL speed > QADS surface
criteria_vulnerability = [
    [1.0,  1.14,  1.6],
    [0.88, 1.0,   1.4],
    [0.62, 0.71,  1.0],
]
# → Normalized: [0.40, 0.35, 0.25]; CR ≈ 0.000

# Overall Q(v): [R, M, A, V] — equal by default (general-purpose analysis)
criteria_overall = [
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
]
# → weights = [0.25, 0.25, 0.25, 0.25]; CR = 0.00
```

**Domain-specific weight examples** (custom `--use-ahp` configurations):

| System Type | Priority | w\_R | w\_M | w\_A | w\_V |
|-------------|----------|:----:|:----:|:----:|:----:|
| High-availability (medical, aerospace) | A > R > M > V | 0.30 | 0.20 | 0.40 | 0.10 |
| Security-critical (financial, government) | V > A > R > M | 0.20 | 0.10 | 0.30 | 0.40 |
| Actively developed (fast iteration) | M > R > A > V | 0.30 | 0.40 | 0.20 | 0.10 |
| General-purpose | Equal | 0.25 | 0.25 | 0.25 | 0.25 |

### Appendix C: Error Handling Strategy

| Error Condition | Handling Strategy |
|----------------|------------------|
| Neo4j connection failure | Retry up to 3 times with exponential backoff (1s, 2s, 4s); raise `ConnectionError` with URI in message after all retries exhausted |
| Invalid topology (missing required field) | Raise `TopologyValidationError` with field name and constraint; do not partially import |
| Dangling edge reference (endpoint ID not found) | Raise `TopologyValidationError` with edge ID and missing component ID |
| Duplicate component ID | Raise `TopologyValidationError` with duplicate ID |
| Disconnected graph (weakly) | Log WARNING; proceed with analysis per weakly connected component |
| Eigenvector centrality non-convergence | Fall back to in-degree; log WARNING with component count |
| AHP consistency failure (CR > 0.10) | Abort with `AHPConsistencyError` including CR value and matrix. Do not proceed with inconsistent weights |
| Memory exhaustion | Stream large graphs in batches; log memory usage at DEBUG level at large scale |
| GraphML parse error | Return specific line/element in error message; do not partially import |

> **AHP handling rationale:** Earlier versions warned but continued on AHP inconsistency. SRS REQ-QS-08 (v2.2) requires aborting, because proceeding with an inconsistent matrix would silently produce unreliable weights, invalidating the entire RMAV score. The analyst must revise pairwise judgments before proceeding.

### Appendix D: SRS-to-Design Traceability

| SRS Requirement | Design Element |
|-----------------|---------------|
| REQ-GM-01 | §5.1 Import Pipeline (JSON parser), §8.3 input topology JSON schema |
| REQ-GM-02 | §5.1 Import Pipeline (GraphML parser), §2.1 system context diagram |
| REQ-GM-03–05 | §5.1 Phase 1–3; §7.1 relationship types; §7.2 dependency derivation Cypher |
| REQ-GM-06 | §5.1 Phase 3 (QoS weight computation); §8.3 QoS field in input schema |
| REQ-GM-07 | §5.1 Phase 4 (DEPENDS\_ON weight propagation from Topics) |
| REQ-GM-08, REQ-ML-01–04 | §5.2 Step 1 (layer export); Appendix A layer definitions |
| REQ-SA-01–02 | §6.1 PageRank / Reverse PageRank |
| REQ-SA-03 | §6.2 Betweenness Centrality (Brandes) |
| REQ-SA-04 | §6.3 Closeness Centrality (Wasserman-Faust) |
| REQ-SA-05 | §6.4 Eigenvector Centrality (power iteration + fallback) |
| REQ-SA-06 | §4.2 StructuralMetrics: in\_degree / out\_degree fields |
| REQ-SA-07 | §4.2 StructuralMetrics: clustering\_coefficient field |
| REQ-SA-08 | §6.5 Articulation Point Detection (binary + AP\_c); §6.16 AP\_c\_directed |
| REQ-SA-09 | §6.6 Bridge Detection and Bridge Ratio |
| REQ-SA-10 | §4.2 StructuralMetrics weight fields (w, w\_in, w\_out) |
| REQ-SA-11 | §6.8 Min-Max Normalization |
| REQ-SA-12 | §5.2 StructuralAnalyzer: graph-level summary statistics |
| REQ-QS-01 | §6.19 Reliability Score R(v)  [Prediction — RMAV] |
| REQ-QS-02 | §6.20 Maintainability Score M(v) |
| REQ-QS-03 | §6.21 Availability Score A(v) |
| REQ-QS-04 | §6.22 Vulnerability Score V(v) |
| REQ-QS-05 | §6.23 Composite Quality Score Q(v) |
| REQ-QS-06 | §6.9 Box-Plot Classification |
| REQ-QS-07–08 | §6.7 AHP Weight Calculation; Appendix C AHP error handling |
| REQ-QS-09 | §6.9 Box-Plot Classification (normal path and fallback) |
| REQ-QS-10 | §5.6 Anti-Pattern Detection Pipeline |
| REQ-FS-01–05 | §5.3 Simulation Pipeline; §6.10 Cascade Propagation |
| REQ-VL-01 | §6.11 Spearman Rank Correlation |
| REQ-VL-02–10 | §5.4 Validation Pipeline (classification, ranking, error metrics, min-n check) |
| REQ-VL-07 | §6.12 NDCG@K |
| REQ-VZ-01–08 | §5.5 Visualization Pipeline; §9.1 Static Dashboard |
| REQ-CLI-01–04 | §8.1 CLI Interface |
| REQ-CLI-05 | §8.1 CLI Interface (`--generate` flag via `bin/run.py`) |
| REQ-CLI-06 | §8.1 CLI Interface (`--scale` flag) |
| REQ-CLI-07 | §8.1 CLI Interface (`--verbose` / `--quiet` flags) |
| REQ-CLI-08 | §8.1 CLI Interface (`--open` flag) |
| REQ-API-01 | §8.2 REST API: `GET /health` |
| REQ-API-02 | §8.2 REST API: `GET /api/v1/graph/summary` |
| REQ-API-03 | §8.2 REST API: `POST /api/v1/graph/import` |
| REQ-API-04 | §8.2 REST API: `GET /api/v1/graph/search-nodes` |
| REQ-API-05 | §8.2 REST API: `POST /api/v1/analysis/layer/{layer}` |
| REQ-API-06 | §8.2 REST API: `POST /api/v1/simulation/failure` |
| REQ-API-07 | §8.2 REST API: `POST /api/v1/validation/run-pipeline` |
| REQ-API-08 | §8.2 REST API: `GET /api/v1/validation/layers` |
| REQ-API-09 | §8.2 REST API: `GET /api/v1/components` |
| REQ-API-10 | §8.2 HTTP 422 error handling; Appendix C error strategy |
| REQ-API-11 | §8.2 Swagger UI at `/docs` |
| REQ-WEB-01–10 | §9.2 Genieus Web Application design |
| REQ-SEC-01–03 | §2.2 Design Constraints; Appendix C (Neo4j connection error handling) |
| REQ-LOG-01–03 | Appendix C (logging strategy per error type) |
| REQ-PERF-01–04 | §3.5 Deployment Architecture (uvicorn 2 workers); §6.1–§6.12 complexity annotations |
| REQ-PERF-05 | §3.5 Deployment Architecture (uvicorn 2 workers, async endpoints) |
| REQ-SCAL-01–02 | §2.2 Design Constraints (memory-bound); §6.17 CDI sampled APSP for enterprise scale |
| REQ-PORT-01–02 | §3.5 Docker deployment (platform-agnostic container) |
| REQ-PORT-03 | §3.5 Deployment Architecture (multi-stage Docker build) |
| REQ-MAINT-01–03 | §3.3 Design Patterns (composition, strategy, repository) |

---

*Software-as-a-Graph Framework v2.3 · March 2026*
*Istanbul Technical University, Computer Engineering Department*