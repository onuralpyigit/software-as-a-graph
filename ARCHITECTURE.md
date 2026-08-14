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

```
saag/          # Core SDK — domain models, services, use cases, infrastructure
api/           # FastAPI REST layer — routers, presenters, dependency injection
cli/           # Pipeline CLI scripts (one per stage) + shared utilities
smart/         # Next.js web application (SMART dashboard)
tools/         # Synthetic graph generation, benchmarking, static analysis, Neo4j plugin
reproduce/     # Paper reproduction package — its own Makefile/Dockerfile/README
scripts/       # Shell orchestrators for longer experiment sweeps
examples/      # Annotated, runnable SDK usage examples
data/          # Topology JSONs, scenario YAMLs, and configuration datasets
models/        # Trained GNN checkpoints
output/        # Pipeline run artifacts (dashboards, predictions, checkpoints)
results/       # Rendered paper tables/figures from reproduce/
evaluation/    # Ad-hoc evaluation artifacts
tests/         # Pytest test suite
docs/          # Per-stage methodology documentation + formal specs
```

---

## System Pipeline & Data Flow

The analytical pipeline is a Directed Acyclic Graph, not a linear chain. Step 2 (Analyze) computes structural metrics only and feeds them to both Step 3 (Predict) and Step 4 (Simulate), which run independently. Step 5 (Validate) then compares prediction outcomes against the simulation ground-truth labels.

Step 3 (Predict) is a unified **Prediction Step**: the legacy "Quality Scoring" mechanism that used to live inside Analyze has been removed and replaced by a single step that always computes rule-based (RM) scores, blends in ML (GNN) inference when a trained checkpoint is available, and runs anti-pattern detection and explanation generation on the result.

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
   │Prediction │                     │Simulation │
   │  Result   │                     │  Result   │
   │(RM+GNN+   │                     │  (Labels) │
   │AntiPattern│                     │           │
   │+Explain)  │                     │           │
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
> **First-run sequencing:** Step 3 (Predict) depends on simulation-derived training labels for GNN training. On the first run, execute Steps 1 $\rightarrow$ 2 $\rightarrow$ 4 to generate those labels, then train the GNN model, and finally run Step 3 inference. The Predict stage (Step 3) is fully self-contained for its rule-based path and produces valid RM $Q^*(v)$ scores without requiring a GNN checkpoint (the Analyze stage, Step 2, is structural-only and does not compute RM/Q scores).

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
│   PredictGraphUseCase  SimulateGraphUseCase               │
│   ValidateGraphUseCase PrescribeGraphUseCase             │
│   VisualizeGraphUseCase MultiLayerAnalysisUseCase         │
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
│   Ports: IGraphRepository, IFileStore                    │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│  Infrastructure (infrastructure/)                        │
│   Neo4jRepository          MemoryRepository              │
│   (production)             (testing)                     │
└──────────────────────────────────────────────────────────┘
```

### `core/` — Domain Models & Ports
Pure Python; no dependency on Neo4j, NetworkX, or presentation frameworks.
- `models.py` — Physical pub-sub entities: `ComponentData`, `EdgeData`, `GraphData`, `Application`, `Broker`, `Node`, `Library`, `Topic`, `QoSPolicy`.
- `metrics.py` — Analytical models: `StructuralMetrics`, `ComponentQuality`, `EdgeMetrics`, `GraphSummary`.
- `layers.py` — Layer projections (`AnalysisLayer` enum: `app`, `infra`, `mw`, `system`) and their member mappings (`LAYER_DEFINITIONS`).
- `criticality.py` — Thresholding structures: `CriticalityLevel`, `BoxPlotStats`.
- `ports/graph_repository.py` — `IGraphRepository`: `save_graph()`, `derive_dependencies()`, `get_graph_data()`, `get_layer_data()`, `export_json()`.
- `ports/file_store.py` — `IFileStore`, implemented by `file_exporter.py`'s `LocalFileStore` for filesystem I/O.
- `utils/serialization.py` — Flatten/reconstruct helpers between nested JSON and flat graph properties.

### `analysis/` — Step 2 Analytical Engine
Computes structural metrics only on the layer subgraph. No RM/Q scores or anti-patterns — those are produced by the Predict stage (Step 3).
- `StructuralAnalyzer` — NetworkX-based PageRank, Betweenness, Harmonic Closeness, Eigenvector, and Reverse PageRank, plus custom pub-sub metrics (MPCI, FOC, CDI, PC).
- `AnalysisService` — Orchestrates layer projections and calculations against `IGraphRepository`.
- `AntiPatternDetector` — Audits RM scores to flag architectural smells (SPOF, FAILURE_HUB, GOD_COMPONENT, etc.). It lives here but is invoked by `prediction/`, since it operates on Predict-stage output.
- `QualityAnalyzer`, `BoxPlotClassifier`, `AHPProcessor` — the RM scoring, classification, and AHP-weighting implementations used by the Predict stage. Import them from here; `saag/prediction/` no longer re-exports them.

### `prediction/` — Step 3 Predictive Engine (unified Prediction Step)
A single step that always computes rule-based RM scores, blends in ML/GNN inference when available, and runs anti-pattern detection and explanation generation.
- `PredictionService` — The single entry point for the unified Predict stage: RM scoring and problem detection (delegated to `analysis/`), then GNN inference when a checkpoint is available (else falling back to RM), then anti-pattern detection and explanation generation.
- `GNNService` — Loads a checkpoint containing `NodeCriticalityGNN`: `N` stacked stock `torch_geometric.nn.HGTConv` layers, with an `EdgeFeatureEncoder` injecting edge features before each layer ([core.py:146-290](saag/prediction/models/core.py#L146-L290)). Runs inductive prediction.
- `ExplanationEngine` (from `explanation/`) — Generates the natural-language narrative attached to each Predict-stage result.

> **Back-compat shims — not architectural components.** `saag/adapters/` and `saag/core/graph_generator.py` are thin re-export stubs kept for import compatibility; their real implementations live in `tools/generation/`. Do not extend the shims directly.

### `simulation/` — Step 4 Simulation Engine
A discrete-event and BFS cascade failure simulation suite evaluating propagation boundaries on raw structural edges.
- `SimulationGraph` — Wraps the structural topology projection for traversal operations.
- `FaultInjector` — **Canonical Predict-stage labeler.** Pub-sub BFS cascade producing the scalar $I^*(v)$ written to `impact_scores.json`, which supplies the supervised training labels for the GNN. Deterministic and multi-seed, and emits its own provenance (`labeler`, `labeled_node_types`, `labeled_dimensions`, `unlabeled_node_ids`) plus a `label_stability` block giving the ceiling on any correlation reported against it.
- `FailureSimulator` — **Canonical Validate-stage oracle.** Runs the main BFS cascade simulation under different scenarios (CRASH, DEGRADED, etc.) across physical, logical, network, and library pathways, producing the composite and IR/IM/IA/IS decomposition the validation gates are written against.
- `EventSimulator` — Models transient message flow to estimate throughput degradation and queue delays.
- `MessageFlowSimulator` — SimPy-based discrete-event flow simulation with QoS-aware message queues.
- `TrafficSimulator` — Analytical (non-discrete-event) load estimator.
- `ChangePropagationSimulator` — Propagates code-level modifications against $G^T$ to evaluate change-reach bounds.
- `CompromisePropagationSimulator` — Propagates cyber-breach scenarios along trust-weighted dependency paths.
- `ComplexityProcessor` — Converts component complexity into processing-latency estimates for flow simulation.
- `SimulationService` — Orchestrates all of the above for use-case consumption.

> **`FaultInjector` and `FailureSimulator` both emit a quantity called "impact", and the two are not interchangeable.** Each owns exactly one pipeline stage — labels vs. validation oracle — and mixing them within a stage is a correctness error, enforced by `tests/test_groundtruth_contract.py`. See [docs/failure-simulation.md §2.1](docs/failure-simulation.md#21-which-engine-is-canonical-for-what).

### `validation/` — Step 5 Validation Engine
Correlates predictions against simulation ground-truth metrics to verify thesis validation gates.
- `Validator` — Evaluates prediction output arrays against ground truth using Spearman $\rho$, Kendall $\tau$, F1, Precision, and Recall.
- `ValidationService` — Evaluates the nine-gate tier system (see [README §Validation Gates](README.md#validation-gates)) and computes system health indices (SRI, RCI).

### `prescription/` — Step 6 Prescriptive Engine
Generates rule-based architectural optimization policies (logical splitting, host anti-affinity container reallocations, and transport contract QoS upgrades) and validates resilience improvements in-memory, accepting only edits whose counterfactual impact reduction beats the simulator's seed noise at every propagation threshold. See [docs/prescription.md](docs/prescription.md).
- `rules.compile_policy` — Compiles the candidate policy Δ(G) from analysis + prediction. Pure: no repository, no simulation.
- `mutator.apply_policy` — Rewrites the flat JSON topology export. Pure.
- `GraphEvaluator` — Runs Analyze → Predict → Simulate → Validate over a repository; one simulation sweep feeds SRI, metrics and the per-component impact map.
- `EditVerifier` — The per-edit acceptance filter and its arithmetic.
- `PrescribeService` — Orchestrates the above and assembles the `PrescribeResult`.

### `visualization/` — Step 7 Visualization Engine
Compiles the metrics, classifications, problems, and simulations into visual dashboard formats.
- `VisualizationService` — Assembles the multi-stage dataset into serializable models; composes analysis + prediction + simulation + validation services.
- `LayerDataCollector` — Aggregates per-layer data across services.
- `DashboardGenerator` — Renders self-contained static HTML pages including Cytoscape network views and interactive charts.
- `ChartGenerator` — Produces embeddable chart snippets.

### `explanation/` — Natural Language Explanations
- `ExplanationEngine` — Binds metric values to text templates to produce component- and system-level narrative reports.
- `CLIFormatter` — Renders the same explanations as human-readable CLI cards.

### `evaluation/` — Cross-Variant Evaluation Contract
Supports the paper's result tables independently of the runtime pipeline.
- `metrics.py` — `compute_inductive_metrics()`, `resolve_eval_keys()`, critical-topic-coverage-at-K, and per-QoS-tier Spearman $\rho$. Consumed by `reproduce/main_table.py` and `reproduce/loso_all_variants.py`.

### `usecases/` — Application Layer orchestrators
Thin interactor classes representing the application boundaries; each pipeline step maps to one class (`ModelGraphUseCase`, `AnalyzeGraphUseCase`, `PredictGraphUseCase`, `SimulateGraphUseCase`, `ValidateGraphUseCase`, `PrescribeGraphUseCase`, `VisualizeGraphUseCase`, `MultiLayerAnalysisUseCase`), each delegating to a single service's `execute()`.

### `infrastructure/` — Persistence Adapters
- `Neo4jRepository` — Production adapter. Handles connection sessions, executes Cypher to load/export topologies, and drives the Cypher-based `DEPENDS_ON` derivation logic.
- `MemoryRepository` — Pure-Python, in-process adapter satisfying the same `IGraphRepository` protocol, used by tests and by `cli/prescribe_graph.py` to run without a live database.

---

## REST API (`api/`)

The REST API exposes the analytical pipeline as a JSON web service via FastAPI (`api/main.py`), CORS-open, mounted with 10 routers:

| Router | Prefix | Endpoints |
|:---|:---|:---:|
| `graph.py` | `/api/v1/graph` | 13 |
| `components.py` | `/api/v1` | 5 |
| `statistics.py` | `/api/v1/stats` | 5 |
| `prediction.py` | `/api/v1/graph/prediction` | 4 |
| `simulation.py` | `/api/v1/simulation` | 4 |
| `validation.py` | `/api/v1/validation` | 4 |
| `analysis.py` | `/api/v1/analysis` | 3 |
| `traffic.py` | `/api/v1/traffic` | 3 |
| `health.py` | *(none)* | 3 |
| `classification.py` | `/api/v1` | 1 |

- **Routers (`api/routers/`)** — Validate request schemas (`api/models.py`) and call SDK services or use cases.
- **Presenters (`api/presenters/`)** — Decouple domain response schemas from HTTP endpoints. Used by `graph.py`, `simulation.py`, and `analysis.py`; the other routers currently serialize inline — a gap to close, not a pattern to follow.
- **Dependency Injection (`api/dependencies.py`)** — `get_repository()` builds a request-scoped `Neo4jRepository`. Credentials are read from the **JSON request body** (top-level or nested under `"credentials"`) on POST/PUT, or from **query parameters** (`uri`, `user`, `password`, `database`) on GET/HEAD/DELETE — not from HTTP headers.

---

## Command Line Interface (`cli/`)

Scripts mirror the pipeline stages. Eight have console-script entry points installed by `pyproject.toml`; the rest are run as `python cli/<script>.py`.

| Script | Stage | Entry point |
|:---|:---|:---|
| `run.py` | Orchestrates all stages in dependency order | `saag` |
| `generate_graph.py` | Offline prep — synthetic topology | `saag-generate` |
| `import_graph.py` | Step 1 — import + derive dependencies | `saag-import` |
| `export_graph.py` | Step 1 — export Neo4j → JSON | *(none)* |
| `analyze_graph.py` | Step 2 — structural metrics | `saag-analyze` |
| `train_graph.py` | Step 3 (training) — GNN training | *(none)* |
| `predict_graph.py` | Step 3 (inference) — RM + GNN + anti-patterns | `saag-predict` |
| `detect_antipatterns.py` | Standalone anti-pattern / CI gate | *(none)* |
| `simulate_graph.py` | Step 4 — `fault-inject` \| `message-flow` \| `combined` | `saag-simulate` |
| `validate_graph.py` | Step 5 — `single` \| `sweep` \| `report` \| `compare` \| `harness` | `saag-validate` |
| `prescribe_graph.py` | Step 6 — optimize + closed-loop validate | *(none)* |
| `visualize_graph.py` | Step 7 — HTML dashboard | `saag-visualize` |
| `statistics_graph.py` | Cross-cutting topology/communication stats | *(none)* |
| `benchmark.py` | Scale-preset performance benchmark | *(none)* |
| `kfold_evaluate.py` | Per-domain repeated k-fold GNN evaluation (primary) | *(none)* |
| `loso_evaluate.py` | Leave-one-scenario-out GNN evaluation (domain-gap) | *(none)* |
| `multi_seed_summary.py` | Aggregate results across seeds | *(none)* |

Details and flags for each script: [docs/cli-pipeline-guide.md](docs/cli-pipeline-guide.md).

---

## Web Dashboard (`smart/`)

Next.js 14 (App Router) + React 18 application talking to the FastAPI backend over the hand-written axios clients in `smart/lib/api/`.

Routes under `smart/app/`: `dashboard`, `explorer` (2D/3D force-directed graph), `analysis`, `simulator` (failure-injection animation), `predict`, `train`, `traffic`, `statistics`, `validation`, `data`, `glossary`, `settings`. A connection-store React context manages active Neo4j and API connection parameters.

---

## Tools (`tools/`)

Auxiliary packages, installed alongside `saag` but not part of the core SDK:
- `tools/generation/` — `GenerationService` / `StatisticalGraphGenerator` generate pub-sub topologies matching scale presets and QoS distributions.
- `tools/benchmark/` — `BenchmarkRunner` sequentially runs generation, import, and scoring to measure processing latency and memory use.
- `tools/neo4j-plugin/graph-relationship-manager/` — A Java/Maven Neo4j plugin providing `custom.*` Cypher procedures used during import; built and installed by the root `Dockerfile`.
- `tools/static-system-analyzer/` — A separate static-analysis sub-project (own `src/`, `tests/`, `config/`, `docs/`) for extracting code-quality metrics feeding `Application`/`Library` nodes.

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

| Layer | Node Types Included | Derived Edges Evaluated | Primary RM Dimension |
|:---|:---|:---|:---|
| `app` | `Application`, `Library` | `app_to_app`, `app_to_lib` | Reliability ($R$) |
| `infra` | `Node` | `node_to_node` | Availability ($A$, Reliability sub-characteristic) |
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

### Container Topology
`docker-compose.yml` runs a **single** all-in-one container (built from the root `Dockerfile`), bundling Neo4j, the FastAPI backend, and the SMART frontend, and publishing all four ports:

```
                  ┌───────────────────┐
                  │    User Browser   │
                  └─────────┬─────────┘
                            │
               HTTP (7000)  │  HTTP (8000)
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
┌─────────────────────────────────────────────────────┐
│                  Single Container                    │
│  ┌─────────────────┐  HTTP   ┌─────────────────┐    │
│  │ Next.js Web App │ ───────▶│ FastAPI Backend │    │
│  │     (SMART)     │         │     (saag)      │    │
│  └─────────────────┘         └────────┬────────┘    │
│                                        │ Bolt (7687) │
│                                        ▼             │
│                              ┌─────────────────┐     │
│                              │ Neo4j Database  │     │
│                              └─────────────────┘     │
└───────────────────────────────────────────────────────┘
```

- `Dockerfile` (root) — multi-stage build: Python + PyG wheels, Next.js build, the Java Neo4j plugin, then Neo4j + API + frontend combined into one runtime image.
- `api/Dockerfile` and `smart/Dockerfile` — separable single-service images, useful if you want the API and frontend as independent deployments instead of the all-in-one image.
- `reproduce/Dockerfile` — a separate, pinned environment for exact experiment reproduction (its own torch/PyG version, isolated from the app image).

No Neo4j GDS or APOC plugins are used anywhere in the codebase; graph algorithms run in NetworkX (`saag/analysis/structural_analyzer.py`). The only non-standard database dependency is the custom Java plugin under `tools/neo4j-plugin/`.

### Verification & Testing Architecture
- **Unit verification** — 71 test files under `tests/` exercise services, use cases, and scoring math via `MemoryRepository`, an in-memory `IGraphRepository` implementation — no live Neo4j needed, and this is what CI (`.github/workflows/tests.yml`, Python 3.11) runs.
- **Integration verification** — tests tagged `@pytest.mark.integration` validate end-to-end Cypher execution and import/export roundtrips against a running Neo4j instance; run separately (`pytest -m integration`). `pyproject.toml` also defines a `slow` marker and a 120 s per-test timeout.
- **Architectural guard tests** — a few tests enforce package boundaries by static inspection rather than behavior: `test_independence_guarantee.py` (simulation must not import prediction), `test_predict_simulate_separation.py` (the Predict use case must not import simulation), and `test_groundtruth_contract.py` (enforces the `FaultInjector`/`FailureSimulator` role split above).
