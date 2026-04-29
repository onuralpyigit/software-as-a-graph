# Architecture

**saag** (Software-as-a-Graph) is a Python framework that predicts which components in a distributed publish-subscribe system will cause the most damage when they fail, using only the system's architecture. It models the topology as a weighted directed graph, applies topological analysis (RMAV quality scoring, centrality metrics), and validates predictions against cascade failure simulations.

Published at IEEE RASSE 2025.

The framework is accessible in four ways: a Python SDK, a REST API, a set of CLI scripts, and the **SMART** web application (Genieus).

---

## Repository Layout

```
saag/          # Core SDK — domain models, services, use cases, infrastructure
api/           # FastAPI REST layer — routers, presenters, dependency injection
cli/           # Pipeline CLI scripts (one per stage) + shared utilities
smart/         # Next.js web application (Genieus) — port 7000
tools/         # Synthetic graph generation and benchmarking
data/          # Topology JSONs, scenario YAMLs, simulation results
models/        # Trained GNN checkpoints
tests/         # Pytest test suite
docs/          # Methodology and research documentation
```

---

## The Pipeline

Seven stages run in sequence. Each stage consumes the output of the previous one.

| Step | Name | CLI Script | Description |
|------|------|-----------|-------------|
| 0 | **Generate** | `generate_graph.py` | Produce a synthetic pub-sub topology (StatisticalGraphGenerator) |
| 1 | **Model** | `import_graph.py` / `export_graph.py` | Load topology JSON → Neo4j; derive DEPENDS_ON edges |
| 2 | **Analyze** | `analyze_graph.py` | Compute structural metrics → RMAV dimension scores via AHP |
| 3 | **Predict** | `train_graph.py` / `predict_graph.py` | Train or run GNN; ensemble RMAV + GNN for criticality rankings |
| 4 | **Simulate** | `simulate_graph.py` | Inject cascade failures to generate per-component ground-truth labels |
| 5 | **Validate** | `validate_graph.py` | Compare prediction rankings to simulation ground truth (Spearman, F1) |
| 6 | **Visualize** | `visualize_graph.py` | Render interactive HTML dashboards |

**Data flow:**

```
Topology JSON
    └─[Step 1]→ Neo4j Graph
                  └─[Step 2]→ StructuralAnalysisResult
                               └─[Step 3]→ QualityAnalysisResult
                                            └─[Step 4]→ SimulationResult
                                                          └─[Step 5]→ ValidationResult
                                                                        └─[Step 6]→ Dashboard
```

**Orchestration:**

```bash
# Full pipeline via CLI
python cli/run.py --all --layer system

# Full pipeline via SDK
result = Pipeline.from_json("topology.json").analyze().predict().simulate().validate().visualize().run()
```

---

## SDK (`saag/`)

The SDK follows a **hexagonal (ports & adapters) architecture**. Domain logic is isolated from infrastructure (Neo4j) and presentation (HTTP, CLI).

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
│   ValidateGraphUseCase VisualizeGraphUseCase             │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│  Services                                                │
│   AnalysisService   PredictionService  SimulationService │
│   ValidationService VisualizationService                 │
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

### `core/`

Domain models and the persistence port. Nothing here depends on Neo4j, NetworkX, or any framework.

| Module | Key Types |
|--------|-----------|
| `models.py` | `ComponentData`, `EdgeData`, `GraphData`, `Application`, `Broker`, `Node`, `Library`, `Topic`, `QoSPolicy` |
| `metrics.py` | `StructuralMetrics`, `ComponentQuality`, `EdgeMetrics`, `GraphSummary` |
| `layers.py` | `AnalysisLayer` (APP, INFRA, MW, SYSTEM), `LayerDefinition`, `LAYER_DEFINITIONS` |
| `criticality.py` | `CriticalityLevel`, `ClassificationResult`, `BoxPlotStats` |
| `ports/graph_repository.py` | `IGraphRepository` protocol — `save_graph`, `get_graph_data`, `get_layer_data`, `export_json` |

### `analysis/`

Converts a raw graph into per-component structural metrics and detects anti-patterns.

- `StructuralAnalyzer` — NetworkX-based computation: PageRank, Betweenness, Harmonic Closeness, Eigenvector, Reverse PageRank, clustering, articulation points, bridges, pub-sub–specific metrics.
- `AnalysisService` — Orchestrates single-layer and multi-layer analysis; calls `AntiPatternDetector`.
- `AntiPatternDetector` — Identifies SPOF, FAILURE_HUB, GOD_COMPONENT, TARGET, BRIDGE_EDGE, EXPOSURE, CYCLE, HUB_AND_SPOKE, CHAIN, SYSTEMIC_RISK.
- `StatisticsService` — Aggregate distribution statistics over components.

**Output:** `StructuralAnalysisResult` (metrics per component and edge, per layer).

### `prediction/`

Maps structural metrics to RMAV quality scores and optionally refines them with a GNN.

- `QualityAnalyzer` — Applies closed-form RMAV formulas with AHP-derived weights (shrinkage λ=0.70). Produces R(v), M(v), A(v), V(v) and overall Q(v) per component.
- `PredictionService` — Orchestrates quality scoring + optional GNN inference + anti-pattern detection.
- `GNNService` — Loads a pre-trained HeteroGAT checkpoint; runs inductive inference; returns per-component criticality ranks and attention weights.
- `BoxPlotClassifier` — Assigns `CriticalityLevel` using box-plot thresholds (Q3 + k·IQR).
- `ProblemDetector` — Converts quality scores into `DetectedProblem` entries for reporting.
- `WeightCalculator` — AHP weight derivation with shrinkage toward the uniform prior.

**Ensemble blending:**
```
Q_ensemble(v) = α · Q_GNN + (1 − α) · Q_RMAV      (α typically 0.6–0.8)
```

**Output:** `QualityAnalysisResult` (RMAV scores, criticality levels, detected problems per component and edge).

### `simulation/`

Discrete-event cascade failure engine. Operates on raw structural edges (PUBLISHES_TO, ROUTES, RUNS_ON, USES, CONNECTS_TO, SUBSCRIBES_TO) — not DEPENDS_ON.

- `SimulationGraph` — Wraps graph data; provides topology queries for cascade propagation.
- `FailureSimulator` — Injects a `FailureScenario` (target + mode) and propagates cascades via four rules: PHYSICAL (RUNS_ON), LOGICAL (broker pub-sub), NETWORK (CONNECTS_TO), LIBRARY (USES). Computes IR(v), IA(v), and I(v).
- `EventSimulator` — Simulates message delivery across the pub-sub topology; produces flow-disruption metrics.
- `ChangePropagationSimulator` — BFS on G^T for development-time change reach (IM(v)).
- `CompromisePropagationSimulator` — BFS on G^T with trust threshold (IV(v)).
- `SimulationService` — Exposes modes: EXHAUSTIVE, SINGLE, MONTE_CARLO, PAIRWISE, EVENT.

**Component states:** ACTIVE, FAILED, DEGRADED, OVERLOADED, COMPROMISED  
**Failure modes:** CRASH, DEGRADED, PARTITION, OVERLOAD

**Output:** `SimulationResult` (per-component impact labels IR/IM/IA/IV/I, cascade traces).

### `validation/`

Compares prediction rankings to simulation ground truth and evaluates against statistical gates.

- `Validator` — Runs Spearman correlation, F1, Precision, Recall, NDCG@K, dimension-specific metrics (CCR@K, SPOF_F1, AHCR@K, etc.).
- `ValidationService` — Orchestrates multi-layer validation; evaluates against `ValidationTargets` tier gates.
- `MetricCalculator` — Pure-function metric implementations.

**Tier-1 gates:** Spearman ≥ 0.70, F1 ≥ 0.75, Precision ≥ 0.80, Top-5 Overlap ≥ 0.60.

**Output:** `ValidationResult` (per-layer metrics, gate pass/fail).

### `visualization/`

Generates interactive HTML dashboards from pipeline results.

- `VisualizationService` — Collects results from all prior stages; assembles dashboard data.
- `DashboardGenerator` — Renders the full HTML report (network graph, dependency matrix, per-dimension scatter plots, cascade risk, MIL-STD-498 hierarchy).
- `ChartGenerator` — Produces Plotly chart data.

### `explanation/`

Produces natural-language narratives explaining why components are flagged as critical.

- `ExplanationEngine` — Fills templates with metric values and cascade traces.
- `templates.py` — Text templates per anti-pattern and RMAV dimension.

### `usecases/`

One thin orchestrator class per pipeline stage. These are the clean-architecture boundary between the API/CLI and the service layer — they hold no business logic themselves.

| Use Case | Delegates To |
|----------|-------------|
| `ModelGraphUseCase` | Neo4j import (5-phase) |
| `AnalyzeGraphUseCase` | `AnalysisService.analyze_layer()` |
| `PredictGraphUseCase` | `PredictionService` + `ProblemDetector` |
| `SimulateGraphUseCase` | `SimulationService` |
| `ValidateGraphUseCase` | `ValidationService` |
| `VisualizeGraphUseCase` | `VisualizationService` |

### `infrastructure/`

Repository adapters. Swap them without touching any service.

- `Neo4jRepository` — Production adapter. Runs a 5-phase import (entities → structural edges → QoS weights → DEPENDS_ON derivation → aggregate weights), then exposes Cypher-based layer projections.
- `MemoryRepository` — In-memory adapter used by tests. No external dependencies.
- `create_repository(uri, ...)` — Factory that returns the appropriate implementation.

---

## REST API (`api/`)

FastAPI application on port **8000**. Each router delegates directly to a use case or service via dependency injection.

### Routers

| Router | Prefix | Purpose |
|--------|--------|---------|
| `health` | `/health` | Liveness probe |
| `graph` | `/api/v1/graph` | Import and export topology |
| `analysis` | `/api/v1/analysis` | Structural analysis and RMAV scoring |
| `prediction` | `/api/v1/prediction` | GNN + ensemble criticality prediction |
| `components` | `/api/v1` | Component detail queries |
| `statistics` | `/api/v1` | Aggregate graph statistics |
| `simulation` | `/api/v1/simulation` | Cascade failure simulation |
| `classification` | `/api/v1` | Criticality classification |
| `validation` | `/api/v1/validation` | Prediction-vs-simulation validation gates |
| `traffic` | `/api/v1/traffic` | Message flow and traffic analysis |

### Presenters (`api/presenters/`)

Decouple response shaping from business logic. Each presenter takes a domain result and returns a serialisable dict.

- `analysis_presenter` — Formats `StructuralAnalysisResult` / `QualityAnalysisResult`
- `graph_presenter` — Formats `GraphData`, component and edge lists
- `simulation_presenter` — Formats `SimulationResult`
- `statistics_presenter` — Formats aggregate statistics

### Dependency Injection (`api/dependencies.py`)

FastAPI `Depends()` providers:

| Provider | Returns |
|----------|---------|
| `get_repository()` | Request-scoped `IGraphRepository` (Neo4j credentials extracted from request body) |
| `get_client()` | `Client` wrapping the request-scoped repo |
| `get_pipeline()` | `Pipeline` builder wrapping the request-scoped repo |
| `get_prediction_service()` | `PredictionService` singleton |
| `get_statistics_service()` | `StatisticsService` with request-scoped repo |
| `get_generation_service()` | `GenerationService` with scale/domain/seed from request body |

**OpenAPI schema:** `/docs` (Swagger UI) and `api/openapi.json`.

---

## CLI (`cli/`)

```
cli/
├── run.py                  # Orchestrator — runs any combination of stages
├── generate_graph.py       # Step 0: generate synthetic topology
├── import_graph.py         # Step 1a: import JSON → Neo4j
├── export_graph.py         # Step 1b: export Neo4j → JSON
├── analyze_graph.py        # Step 2: structural analysis
├── train_graph.py          # Step 3a: train HeteroGAT
├── predict_graph.py        # Step 3b: RMAV + GNN prediction
├── simulate_graph.py       # Step 4: cascade failure simulation
├── validate_graph.py       # Step 5: validation
├── visualize_graph.py      # Step 6: dashboard generation
├── detect_antipatterns.py  # Standalone anti-pattern scan
├── statistics_graph.py     # Standalone statistics
├── benchmark.py            # Benchmarking across scales
├── multi_seed_summary.py   # Aggregate results across seeds
├── run_scenarios.sh        # Run all 8 domain scenarios
└── common/                 # Shared utilities
    ├── arguments.py        # Reusable argparse helpers
    ├── dispatcher.py       # Command dispatch routing
    ├── console.py          # Formatted console output
    ├── batch_generation.py # Batch generation helpers
    └── dataset_validation.py
```

**Entry points** (registered in `pyproject.toml`):

```
saag              → cli.run:main
saag-analyze      → cli.analyze_graph:main
saag-predict      → cli.predict_graph:main
saag-simulate     → cli.simulate_graph:main
saag-validate     → cli.validate_graph:main
saag-visualize    → cli.visualize_graph:main
saag-import       → cli.import_graph:main
saag-generate     → cli.generate_graph:main
```

**`predict_graph.py` exit codes:**  
`0` — clean (no patterns), `1` — MEDIUM patterns, `2` — HIGH/CRITICAL (blocks deployment).

---

## Web Application (`smart/`)

**Genieus** — Next.js 16 + React 19 + TypeScript frontend served on port **7000**.

### Pages (App Router)

| Route | Purpose |
|-------|---------|
| `/dashboard` | Main visualization hub |
| `/analysis` | Structural metrics and RMAV scores |
| `/data` | Import topology JSON, manage stored graphs |
| `/dictionary` | Anti-pattern catalog and metrics glossary |
| `/explorer` | Interactive graph viewer (`react-force-graph-2d/3d`) |
| `/predict` | GNN criticality prediction results |
| `/train` | GNN model training interface |
| `/simulation` | Run and visualize failure cascades |
| `/statistics` | Aggregate system statistics |
| `/traffic` | Message flow visualization |
| `/validation` | Prediction vs simulation comparison |
| `/settings` | Neo4j connection configuration |
| `/tutorial` | Guided onboarding walkthrough |

### Key Modules

```
smart/
├── app/                    # Next.js App Router pages
├── components/
│   ├── layout/             # App shell: sidebar, header, connection banner
│   ├── settings/           # Connection form
│   └── ui/                 # 24+ Radix/shadcn primitives
├── lib/
│   ├── api/                # Axios-based REST clients (client.ts, simulation-client.ts, ...)
│   ├── config/api.ts       # NEXT_PUBLIC_API_URL
│   ├── stores/             # React context: connection-store, analysis-store
│   └── types/api.ts        # TypeScript response types
└── public/                 # Static assets
```

---

## Tools (`tools/`)

### `tools/generation/`

Synthetic pub-sub topology generator — no Neo4j dependency.

- `StatisticalGraphGenerator` — Produces Application, Broker, Node, Library, Topic entities with statistically parameterized code metrics, QoS policies, and structural edges. Scale presets: TINY, SMALL, MEDIUM, LARGE, HUGE, ENTERPRISE.
- `GenerationService` — Thin wrapper; accepts scale, seed, domain, scenario.
- `datasets.py` — Domain-specific naming and QoS lookup tables (8 domains: autonomous vehicle, IoT, financial trading, healthcare, hub-and-spoke, microservices, enterprise XL, tiny regression).

### `tools/benchmark/`

- `BenchmarkRunner` — Executes the full gen → import → analyze → simulate → validate pipeline across scales; measures wall-clock time and memory.
- `reporting.py` — Produces tabular benchmark reports.

---

## Graph Model

### Node Types

| Type | Represents |
|------|-----------|
| `Application` | Microservice or process |
| `Library` | Shared dependency |
| `Broker` | Message broker |
| `Node` | Physical/virtual host |
| `Topic` | Pub-sub topic |

### Analysis Layers

Each layer is a projection of the full graph onto a relevant architectural concern.

| Layer | Components Analyzed | Primary RMAV Focus |
|-------|--------------------|--------------------|
| `app` | Application, Library | Reliability |
| `infra` | Node | Availability |
| `mw` | Broker | Maintainability |
| `system` | All types | Overall Q(v) |

### DEPENDS_ON Derivation

Raw structural edges (PUBLISHES_TO, ROUTES, RUNS_ON, USES, CONNECTS_TO) are transformed into six typed DEPENDS_ON subtypes during import. Direction: **dependent → dependency**. All carry `weight ∈ [0,1]` (max QoS severity) and `path_count` (coupling intensity).

| Rule | Type | Semantic |
|------|------|---------|
| 1 | `app_to_app` | App subscriber depends on App publisher via shared Topic |
| 2 | `app_to_broker` | App depends on the Broker routing its topics |
| 3 | `node_to_node` | Host Node lifted from Rule 1 |
| 4 | `node_to_broker` | Host Node lifted from Rule 2 |
| 5 | `app_to_lib` | App depends on Library (blast-radius risk) |
| 6 | `broker_to_broker` | Co-located Brokers sharing a physical Node |

---

## Deployment

### All-in-One Container

```bash
docker compose up   # Builds and starts everything
```

| Port | Service |
|------|---------|
| 7474 | Neo4j Browser UI |
| 7687 | Neo4j Bolt |
| 8000 | FastAPI REST API |
| 7000 | Next.js Web App |

**Environment variables:**

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Local Development

```bash
# Python (SDK + API + CLI)
pip install -e ".[all]"
uvicorn api.main:app --reload --port 8000

# Frontend
cd smart && npm install && npm run dev   # http://localhost:7000

# Full pipeline
python cli/run.py --all --layer system
```

### Testing

```bash
cd tests
pytest               # All tests
pytest -x            # Stop on first failure
pytest -k "test_name"
```

Test markers: `slow` (skip with `--quick`), `integration`.  
The test suite uses `MemoryRepository` — no Neo4j required for unit tests.
