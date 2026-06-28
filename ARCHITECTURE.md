# Architecture

**System architecture design, component boundaries, and data flow of the Software-as-a-Graph (SaG) framework.**

[README](README.md) | → [Step 1: Model (Import)](docs/graph-model.md)

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [System Pipeline & Data Flow](#system-pipeline--data-flow)
3. [Core SDK (`saag/`)](#core-sdk-saag)
4. [REST API (`api/`)](#rest-api-api)
5. [Command Line Interface (`cli/`)](#command-line-interface-cli)
6. [Web Dashboard (`smart/`)](#web-dashboard-smart)
7. [Tools (`tools/`)](#tools-tools)
8. [Graph Schema & Model](#graph-schema--model)
9. [Deployment & Verification Architecture](#deployment--verification-architecture)

---

## Repository Layout

The repository is structured into distinct top-level directories partitioning the core SDK, REST endpoints, CLI utilities, and the dashboard frontend:

```
saag/          # Core SDK — domain models, services, use cases, infrastructure
api/           # FastAPI REST layer — routers, presenters, dependency injection
cli/           # Pipeline CLI scripts (one per stage) + shared utilities
smart/         # Next.js web application (SMART dashboard)
tools/         # Synthetic graph generation and benchmarking
data/          # Topology JSONs, scenario YAMLs, and configuration datasets
models/        # Trained GNN checkpoints
tests/         # Pytest test suite
docs/          # Detailed step-by-step methodology documentation
```

---

## System Pipeline & Data Flow

The analytical pipeline is structured as a Directed Acyclic Graph (DAG) rather than a linear chain. Step 2 (Analyze) computes structural metrics and feeds them to both Step 3 (Predict) and Step 4 (Simulate), which run independently. Step 5 (Validate) then compares prediction outcomes against the simulation ground-truth labels.

```
                  ┌──────────────┐
                  │ Topology JSON│
                  └──────┬───────┘
                         │
                         ▼ [Step 1: Model]
                  ┌──────────────┐
                  │ Neo4j Graph  │
                  └──────┬───────┘
                         │
                         ▼ [Step 2: Analyze]
              ┌─────────────────────────┐
              │ StructuralAnalysisResult│
              └───────────┬─────────────┘
                          │
         ┌────────────────┴────────────────┐
         ▼ [Step 3: Predict]               ▼ [Step 4: Simulate]
   ┌───────────┐                     ┌───────────┐
   │  Quality  │                     │Simulation │
   │  Analysis │                     │  Result   │
   │  Result   │                     │  (Labels) │
   └─────┬─────┘                     └─────┬─────┘
         │                                 │
         │      - - - (trains) - - - >     │
         │     [Simulate ground-truth]     │
         │                                 │
         └────────────────┬────────────────┘
                          │
                           ▼ [Step 5: Validate]
                    ┌──────────────┐
                    │  Validation  │
                    │    Result    │
                    └──────┬───────┘
                           │
                           ▼ [Step 6: Prescribe]
                    ┌──────────────┐
                    │  Prescribe   │
                    │    Result    │
                    └──────┬───────┘
                           │
                           ▼ [Step 7: Visualize]
                    ┌──────────────┐
                    │  Dashboard   │
                    └──────┬───────┘
```

> [!NOTE]
> **First-run sequencing:** Step 3 (Predict) depends on simulation-derived training labels for GNN training. On the first run, execute Steps 1 $\rightarrow$ 2 $\rightarrow$ 4 to generate those labels, then train the GNN model, and finally run Step 3 inference. The Analyze stage (Step 2) is fully self-contained and produces valid RMAV $Q^*(v)$ scores without requiring a GNN checkpoint.

---

## Core SDK (`saag/`)

The SDK follows a **hexagonal (ports & adapters) architecture**. Domain logic is isolated from database infrastructure (Neo4j) and presentation layers (HTTP API, CLI).

```
┌──────────────────────────────────────────────────────────┐
│  Entry Points                                            │
│   Pipeline (fluent builder)   Client (step-by-step API) │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│  Use Cases (usecases/)                                   │
│   ModelGraphUseCase    AnalyzeGraphUseCase               │
│   PredictGraphUseCase  SimulateGraphUseCase              │
│   ValidateGraphUseCase PrescribeGraphUseCase             │
│   VisualizeGraphUseCase                                  │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│  Services                                                │
│   AnalysisService   PredictionService  SimulationService │
│   ValidationService PrescribeService   VisualizationServ │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│  Core Domain (core/)                                     │
│   models.py  metrics.py  layers.py  criticality.py       │
│   Ports: IGraphRepository                                │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│  Infrastructure (infrastructure/)                        │
│   Neo4jRepository          MemoryRepository              │
│   (production)             (testing)                     │
└──────────────────────────────────────────────────────────┘
```

### `core/` — Domain Models & Ports
Contains the core domain models and persistence ports. Modules in this package are pure python; they do not depend on Neo4j, NetworkX, or presentation frameworks.
- `models.py` — Represents physical pub-sub entities: `ComponentData`, `EdgeData`, `GraphData`, `Application`, `Broker`, `Node`, `Library`, `Topic`, and `QoSPolicy`.
- `metrics.py` — Defines analytical models: `StructuralMetrics`, `ComponentQuality`, `EdgeMetrics`, and `GraphSummary`.
- `layers.py` — Configures layer projections (`AnalysisLayer` enum: `app`, `infra`, `mw`, `system`) and their associated member mappings (`LAYER_DEFINITIONS`).
- `criticality.py` — Represents thresholding structures: `CriticalityLevel` and `BoxPlotStats`.
- `ports/graph_repository.py` — Defines the `IGraphRepository` interface port outlining required lifecycle adapters (`save_graph()`, `get_graph_data()`, and `export_json()`).

### `analysis/` — Step 2 Analytical Engine
Computes structural metrics, AHP-weighted RMAV quality scores, and architectural anti-patterns on the layer subgraph.
- `StructuralAnalyzer` — Implements NetworkX-based algorithms for PageRank, Betweenness, Harmonic Closeness, Eigenvector, and Reverse PageRank, alongside custom pub-sub metrics (MPCI, FOC, CDI, and PC).
- `AnalysisService` — Orchestrates layer projections and calculations. It pulls graph projections from `IGraphRepository`, runs the `StructuralAnalyzer`, triggers `QualityScoringService` to apply AHP-weighted formulas, executes `AntiPatternDetector`, and invokes `ExplanationEngine` for natural language text descriptions.
- `AntiPatternDetector` — Audits scores to flag architectural smells (SPOF, FAILURE_HUB, GOD_COMPONENT, etc.).

### `prediction/` — Step 3 Predictive Engine
Houses the AHP-weighted scoring implementation and the inductive Graph Attention Network.
- `QualityAnalyzer` — Applies AHP-weighted composite quality formulas with shrinkage ($\lambda=0.70$) to output Reliability ($R$), Maintainability ($M$), Availability ($A$), and Vulnerability ($V$) scores.
- `PredictionService` — Orchestrates the prediction stage, utilizing raw GNN outputs for inductive node and edge criticality predictions.
- `GNNService` — Loads a checkpoint containing the `NodeCriticalityGNN` (built using three stacked `EdgeAwareHGTConv` layers with edge feature injection) to run inductive prediction.
- `BoxPlotClassifier` — Performs adaptive outlier-fence classification.

### `simulation/` — Step 4 Simulation Engine
A discrete-event and BFS cascade failure simulator evaluating propagation boundaries on raw structural edges.
- `SimulationGraph` — Wraps the structural topology projection for traversal operations.
- `FailureSimulator` — Runs the main BFS cascade simulation under different scenarios (CRASH, DEGRADED, etc.) across physical, logical, network, and library pathways.
- `EventSimulator` — Models transient message flow to estimate throughput degradation and queue delays.
- `ChangePropagationSimulator` — Propagates code-level modifications against G^T to evaluate change-reach bounds.
- `CompromisePropagationSimulator` — Propagates cyber-breach scenarios along trust-weighted dependency paths.

### `validation/` — Step 5 Validation Engine
Correlates predictions against simulation ground-truth metrics to verify thesis validation gates.
- `Validator` — Evaluates prediction output arrays against ground truth using Spearman $\rho$, Kendall $\tau$, F1, Precision, and Recall.
- `ValidationService` — Evaluates validation targets across the 9-gate tier system and computes system health indices (SRI, RCI).

### `prescription/` — Step 6 Prescriptive Engine
Generates rule-based architectural optimization policies (logical splitting, host anti-affinity container reallocations, and transport contract QoS upgrades) and validates resilience improvements in-memory.

### `visualization/` — Step 7 Visualization Engine
Compiles the metrics, classifications, problems, and simulations into visual dashboard formats.
- `VisualizationService` — Assembles the multi-stage dataset into serializable models.
- `DashboardGenerator` — Renders self-contained static HTML pages including Cytoscape network views and interactive charts.

### `explanation/` — Natural Language Explanations
Exposes translation features that turn numeric metrics and dependency traces into readable reports.
- `ExplanationEngine` — Formulates narrative structures by binding metric values to text templates.

### `usecases/` — Application Layer orchestrators
Exposes thin interactor patterns representing the application boundaries. Each pipeline step is mapped to a single class (e.g. `ModelGraphUseCase`, `AnalyzeGraphUseCase`) delegating directly to services.

### `infrastructure/` — Persistence Adapters
Implements concrete adapters matching the persistence port.
- `Neo4jRepository` — The production adapter. Handles database connection sessions, executes Cypher queries to load/export topologies, and drives the Cypher-based `DEPENDS_ON` relationship derivation logic.
- `MemoryRepository` — An in-memory, thread-safe mock adapter utilized during testing to run the pipeline without Neo4j database instances.

---

## REST API (`api/`)

The REST API exposes the analytical pipeline as a JSON-based web service utilizing the FastAPI framework:
- **Routers (`api/routers/`)** — Thin presentation entry points. They validate request schemas and pass parameters directly to SDK Use Case interactor boundaries.
- **Presenters (`api/presenters/`)** — Decouple domain response schemas from HTTP endpoints. They transform complex SDK use case results into API-ready dictionaries.
- **Dependency Injection (`api/dependencies.py`)** — Resolves request-scoped database connections and service lifecycles. It dynamically binds `IGraphRepository` adapters based on credentials provided in HTTP request headers.

---

## Command Line Interface (`cli/`)

The CLI directory contains executable scripts mirroring the stages of the analytical pipeline:
- `run.py` — Main entry point executing multiple stages in sequence.
- `generate_graph.py` — Generates synthetic topologies using statistical presets.
- `import_graph.py` & `export_graph.py` — Import topology JSON files into Neo4j or export database representations.
- `analyze_graph.py`, `train_graph.py`, `predict_graph.py` — Step 2 and Step 3 analytical and prediction controllers.
- `simulate_graph.py` & `validate_graph.py` — Step 4 simulation and Step 5 statistical validation controllers.
- `prescribe_graph.py` — Step 6 prescriptive optimization and closed-loop validation controller.
- `visualize_graph.py` — Step 7 dashboard rendering controller.

---

## Web Dashboard (`smart/`)

The frontend component (**SMART**) is a single-page Next.js dashboard application interacting with the FastAPI backend:
- **App Router (`app/`)** — Defines frontend routes (e.g., `/dashboard`, `/explorer`, `/simulation`, `/validation`) organizing visualization concerns.
- **React Force Graph** — Renders interactive 2D and 3D network visualizations in `/explorer` to inspect derived dependency links.
- **Connection Context Store** — Manages active connection parameters to Neo4j and FastAPI endpoints.

---

## Tools (`tools/`)

Auxiliary libraries supporting experimental generation and performance metrics:
- `tools/generation/` — Exposes the `StatisticalGraphGenerator` which generates pub-sub topologies matching specific scale parameters and QoS probability distributions.
- `tools/benchmark/` — Exposes the `BenchmarkRunner` that sequentially runs the generation, import, and scoring pipeline to measure processing latency and memory utilization.

---

## Graph Schema & Model

### Node Schema
Topological nodes are categorized into five entity types within the graph database:

| Entity Type | Represents | Core Schema Attributes |
|:---|:---|:---|
| `Application` | Executable process | `id`, `name`, `role`, `app_type`, `version`, static code metrics (`cm_*`) |
| `Library` | Shared package | `id`, `name`, `version`, static code coupling metrics |
| `Broker` | Message broker instance | `id`, `name`, operational weight |
| `Node` | Physical or virtual host | `id`, `name`, IP address, hardware capacity details |
| `Topic` | Message queue channel | `id`, `name`, QoS policy (Reliability, Durability, Priority), payload size |

### Analysis Layer Projections
Analytic metrics are calculated on specific subgraphs matching the active layer:

| Layer | Node Types Included | Derived Edges Evaluated | Primary RMAV Dimension |
|:---|:---|:---|:---|
| `app` | `Application`, `Library` | `app_to_app`, `app_to_lib` | Reliability ($R$) |
| `infra` | `Node` | `node_to_node` | Availability ($A$) |
| `mw` | `Broker` | `app_to_broker`, `node_to_broker`, `broker_to_broker` | Maintainability ($M$) |
| `system` | All types | All derived dependency edges | Overall Quality ($Q^*$) |

### Dependency Derivation Rules
Structural connections (e.g. pub/sub topics and broker routing) are transformed into logical `DEPENDS_ON` edges pointing from the **dependent component to its dependency**:

| Rule | Dependency Type | Derived Pathway | Semantics |
|:---|:---|:---|:---|
| 1 | `app_to_app` | Subscriber $\rightarrow$ Topic $\leftarrow$ Publisher | subscriber depends on data produced by publisher |
| 2 | `app_to_broker` | App $\rightarrow$ Topic $\leftarrow$ Router Broker | component depends on broker handling its message routing |
| 3 | `node_to_node` | Host $\rightarrow$ App $\rightarrow$ App $\rightarrow$ Host | host node depends on remote host running dependent publisher |
| 4 | `node_to_broker` | Host $\rightarrow$ App $\rightarrow$ Router Broker | host node inherits broker dependencies of its hosted applications |
| 5 | `app_to_lib` | App $\rightarrow$ USES $\rightarrow$ Library | application depends on library package logic (shared blast risk) |
| 6 | `broker_to_broker` | Broker $\leftrightarrow$ Host $\leftrightarrow$ Broker | co-located brokers share hardware fate (bidirectional) |

---

## Deployment & Verification Architecture

### Multi-Service Topology
The application is designed to deploy as three decoupled containerized services coordinated via Docker Compose:

```
                  ┌───────────────────┐
                  │    User Browser   │
                  └─────────┬─────────┘
                            │
               HTTP (7000)  │  HTTP (8000)
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
┌─────────────────┐  HTTP (8000)  ┌─────────────────┐
│ Next.js Web App │ ─────────────>│ FastAPI Backend │
│     (SMART)     │               │     (saag)      │
└─────────────────┘               └────────┬────────┘
                                           │
                                           │ Bolt (7687)
                                           ▼
                                  ┌─────────────────┐
                                  │ Neo4j Database  │
                                  │   (GDS + APOC)  │
                                  └─────────────────┘
```

- **Database Container** — Serves Bolt connections on port `7687` for transaction execution, and HTTP on `7474` for browser access.
- **FastAPI API Container** — Exposes REST endpoints on port `8000` to process pipeline orchestrations.
- **SMART Web Container** — Serves the React web app on port `7000`.

### Verification & Testing Architecture
The test suite utilizes a decoupled testing design:
- **Unit Verification** — Runs unit checks on services, use cases, and mathematical scoring components using the `MemoryRepository`. This mock repository performs in-memory graph operations, allowing tests to run quickly in CI/CD without spinning up a live Neo4j database instance.
- **Integration Verification** — Validates end-to-end cypher execution and import/export roundtrips against a running Neo4j instance. These integration tests are tagged with the `integration` mark and run during full staging builds.
