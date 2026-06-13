# CLAUDE.md

## 1. LLM Coding Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Simplicity First
**Minimum code that solves the problem. Nothing speculative.**
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Surgical Changes
**Touch only what you must. Clean up only your own mess.**
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.
- When your changes create orphans:
  - Remove imports/variables/functions that YOUR changes made unused.
  - Don't remove pre-existing dead code unless asked.
- The test: Every changed line should trace directly to the user's request.

### Goal-Driven Execution
**Define success criteria. Loop until verified.**
- Transform tasks into verifiable goals:
  - "Add validation" → "Write tests for invalid inputs, then make them pass"
  - "Fix the bug" → "Write a test that reproduces it, then make it pass"
  - "Refactor X" → "Ensure tests pass before and after"
- For multi-step tasks, state a brief plan:
  ```
  1. [Step] → verify: [check]
  2. [Step] → verify: [check]
  3. [Step] → verify: [check]
  ```
- Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
- Guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 2. Project Quick Start & Commands

### Prerequisites
- Python 3.11 (project requires `>=3.9`; dev baseline pinned to 3.11) with a virtual environment (`.venv/` recommended)
- Node.js 18+ (for the frontend)
- Neo4j 5.x (local or via Docker)
- Docker & Docker Compose (for full-stack deployment)

### Environment Variables
Root `.env` (used by Docker Compose and local dev):
```ini
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Running the Full Stack (Docker)
```bash
docker compose up --build
```
Single all-in-one container exposes:
- **Web Dashboard:** http://localhost:7000
- **API (Swagger docs):** http://localhost:8000/docs
- **Neo4j Browser:** http://localhost:7474 (neo4j/password)

### Running Backend Locally
```bash
# Install all extras (neo4j driver, GNN, API server, dev tools)
pip install -e ".[all]"

# Start the API server
uvicorn api.main:app --reload --port 8000

# Run CLI pipeline (single scenario)
python cli/run.py --all --layer system

# Run all 8 scenarios
bash cli/run_scenarios.sh
```

### Running Frontend Locally
```bash
cd smart
npm install
npm run dev          # http://localhost:7000
npm run generate-client  # Regenerate API client from OpenAPI spec
```

### Running Tests
```bash
pytest               # All tests (verbose, short traceback by default)
pytest -x            # Stop on first failure
pytest tests/test_analysis_service.py  # Single test file
pytest -k "test_name"  # Run by test name pattern
```
- **Test configuration:** `pyproject.toml` (pytest section)  
- **Test markers:** `slow` (skip with `--quick`), `integration`  
- **Test timeout:** 120 seconds per test  
- **Test coverage includes:**
  - `test_analysis_service.py` — structural & quality analysis
  - `test_simulation_service.py` — failure and event simulation
  - `test_validation_service.py` — validation pipeline
  - `test_visualization_service.py` — dashboard generation
  - `test_benchmark_service.py` — benchmark service
  - `test_domain_model.py` — core models
  - `test_cli.py` — all CLI scripts
  - `test_api_statistics.py` — API statistics endpoints
  - `test_api_graph.py` — API graph endpoints
  - `test_generation_service.py` — graph generation
  - **Dimension-specific tests:**
    - `test_reliability_dimension.py` — R(v) v4, IR(v), CCR@K, CME
    - `test_maintainability_dimension.py` — M(v) v5, IM(v), COCR@K, weighted-κ CTA, Bottleneck Precision
    - `test_availability_dimension.py` — A(v) v2, IA(v), SPOF_F1, RRI
    - `test_vulnerability_dimension.py` — V(v) v2, IV(v), AHCR@K, FTR, APAR
  - **Orthogonality & sensitivity tests:**
    - `test_availability_orthogonality.py`, `test_vulnerability_orthogonality.py`
    - `test_weight_sensitivity.py`, `test_ahp_shrinkage.py`
    - `test_impact_sensitivity.py`, `test_weighted_reachability.py`
    - `test_pairwise_failure.py`, `test_failure_modes.py`, `test_flow_disruption.py`

---

## 3. Architecture & Components

### Project Overview
**Software-as-a-Graph** predicts which components in a distributed publish-subscribe system will cause the most damage when they fail, using only the system's architecture. It models system topology as a weighted directed graph and applies topological analysis (centrality metrics, quality scoring) to identify critical components — validated against cascade failure simulations (Spearman > 0.87, F1 > 0.90).

### Core Components

#### 1. Core SDK (`saag/`)
- **Purpose:** All domain logic, services, use cases, and infrastructure adapters. Entry points for programmatic use.
- **Key classes:** `Pipeline` (fluent builder), `Client` (step-by-step service façade), `AnalysisResult`, `PredictionResult`, `ValidationResult`.
- **Internal packages:** `core/` (domain models, ports), `analysis/`, `prediction/`, `simulation/`, `validation/`, `visualization/`, `explanation/`, `usecases/`, `infrastructure/`.
- **Repository pattern:** `saag/infrastructure/neo4j_repo.py` (production) and `saag/infrastructure/memory_repo.py` (testing) both implement `IGraphRepository`.

#### 2. REST API (`api/`)
- **Language:** Python 3.11 FastAPI on port 8000.
- **Routers:** `api/routers/` — health, graph, analysis, prediction, components, statistics, simulation, classification, validation, traffic.
- **Presenters:** `api/presenters/` — decoupled response formatting.
- **Dependency injection:** `api/dependencies.py` — `get_repository`, `get_client`, `get_pipeline`, etc.
- **OpenAPI schema:** served at `/docs`; static copy at `api/openapi.json`.

#### 3. Web Application (`smart/`)
- **Name:** SMART
- **Framework:** Next.js 16 with React 19, TypeScript
- **Styling:** Tailwind CSS 4
- **UI components:** Radix UI primitives, shadcn/ui pattern (`components.json`)
- **Key libraries:** `recharts` (charts), `react-force-graph-2d`/`3d` (graph visualization), `axios` (HTTP), `zod` (validation)
- **API clients:** `smart/lib/api/` — `client.ts`, `simulation-client.ts`, `traffic-client.ts`, `validation-client.ts`
- **State:** React context stores in `smart/lib/stores/` — `connection-store`, `analysis-store`
- **Dev server port:** 7000 (`next dev -p 7000`)

#### 4. CLI (`cli/`)
Pipeline scripts that can run independently or via the orchestrator. All run from the repo root:
- `run.py` — End-to-end pipeline orchestrator (`--all` flag, or any combination of stage flags).
- `generate_graph.py` — Offline Prep: Generate — synthetic topology generation.
- `import_graph.py` / `export_graph.py` — Step 1: Model — import JSON into Neo4j (import); export back to JSON (export).
- `analyze_graph.py` — Step 2: structural metrics + RMAV/Q scoring + anti-patterns.
- `train_graph.py` / `predict_graph.py` — Step 3: GNN training and inference.
- `simulate_graph.py` — Step 4: cascade failure simulation (`--mode exhaustive|monte_carlo|single|pairwise`).
- `validate_graph.py` — Step 5: statistical validation.
- `visualize_graph.py` — Step 6: interactive HTML dashboard.
- `loso_evaluate.py` — Leave-One-Scenario-Out (LOSO) inductive validation protocol for the GNN stage.
- `multi_seed_summary.py` — Aggregates multiple validation JSON results to compute statistical stability (mean and standard deviation of Spearman).
- `statistics_graph.py` — Computes cross-cutting statistics and outlier analysis (topic bandwidth, app balance, topic fanout, node load, etc.) without requiring a live Neo4j instance when reading pre-exported JSON.
- `detect_antipatterns.py` — Standalone anti-pattern scan (CI/CD gate; exit code 0/1/2).
- `benchmark.py` — Benchmarking across scale presets.
- `run_scenarios.sh` — Batch pipeline across all 8 domain scenarios.
- `common/` — Shared argument parsers, dispatcher, console output.

#### 5. Tools (`tools/`)
- `tools/generation/` — `StatisticalGraphGenerator`; produces synthetic pub-sub topologies. No Neo4j dependency.
- `tools/benchmark/` — `BenchmarkRunner`; runs the full pipeline across scale presets and reports timing.
- `tools/neo4j-plugin/` — Custom Neo4j database extension containing stored procedures or relationship management helper utilities.
- `tools/static-system-analyzer/` — Out-of-tree standalone pipeline for repository cloning, topic metadata parsing (manual XML/CodeQL mode), CK metrics scanning, relations aggregation, and statistical anomaly reporting.

#### 6. Reproduce (`reproduce/`)
- **Purpose:** Reproducibility scripts for paper experiments (Middleware 2026, RASSE 2025, UYMS 2026), including attention extraction, main table recalibration, GNN variant evaluation, and plot rendering.

#### 7. Documentation (`docs/`)
- Detailed documentation for each pipeline step: `graph-model.md`, `structural-analysis.md`, `prediction.md`, `failure-simulation.md`, `validation.md`, `visualization.md`, `SDD.md` (Design), `SRS.md` (Requirements), `STD.md` (Test Description).

---

### Project Structure Diagram
```text
.
├── saag/                       # Core SDK — all domain logic, services, use cases
│   ├── pipeline.py             #   saag.Pipeline — fluent builder (entry point)
│   ├── client.py               #   saag.Client — step-by-step service façade
│   ├── models.py               #   Public result & data model types
│   ├── core/                   #   Domain models, ports, layer definitions
│   │   ├── models.py           #     ComponentData, EdgeData, GraphData, domain entities
│   │   ├── metrics.py          #     StructuralMetrics, ComponentQuality
│   │   ├── layers.py           #     Canonical LAYER_DEFINITIONS & DEPENDENCY_TO_LAYER
│   │   ├── criticality.py      #     CriticalityLevel, BoxPlotStats
│   │   └── ports/              #     IGraphRepository protocol
│   ├── analysis/               #   StructuralAnalyzer, AntiPatternDetector, AnalysisService
│   ├── prediction/             #   QualityAnalyzer (RMAV), GNNService, PredictionService
│   ├── simulation/             #   FailureSimulator, EventSimulator, SimulationService
│   ├── validation/             #   Validator, MetricCalculator, ValidationService
│   ├── visualization/          #   DashboardGenerator, VisualizationService
│   ├── explanation/            #   ExplanationEngine (natural-language failure narratives)
│   ├── usecases/               #   One UseCase per pipeline stage (clean architecture boundary)
│   └── infrastructure/         #   Neo4jRepository (production), MemoryRepository (testing)
├── api/                        # FastAPI REST API (port 8000)
│   ├── main.py                 #   App init, CORS, router registration
│   ├── dependencies.py         #   Dependency injection (get_repository, get_client, ...)
│   ├── models.py               #   Pydantic request/response schemas
│   ├── routers/                #   health, graph, analysis, prediction, components, etc.
│   └── presenters/             #   Decoupled response formatters
├── cli/                        # CLI pipeline scripts (run from repo root)
│   ├── run.py                  #   Orchestrator — --all or individual stage flags
│   ├── generate_graph.py       #   Offline Prep: Generate
│   ├── import_graph.py         #   Step 1a: Model (import)
│   ├── export_graph.py         #   Step 1b: Model (export)
│   ├── analyze_graph.py        #   Step 2: Analyze
│   ├── train_graph.py          #   Step 3a: Predict (GNN training)
│   ├── predict_graph.py        #   Step 3b: Predict (inference)
│   ├── simulate_graph.py       #   Step 4: Simulate
│   ├── validate_graph.py       #   Step 5: Validate
│   ├── visualize_graph.py      #   Step 6: Visualize
│   └── common/                 #   Shared argument parsers, dispatcher, console output
├── tools/                      # Standalone tooling (no Neo4j dependency)
│   ├── generation/             #   StatisticalGraphGenerator, GenerationService
│   ├── benchmark/              #   BenchmarkRunner, reporting
│   ├── neo4j-plugin/           #   Neo4j plugins and Makefile scripts
│   └── static-system-analyzer/ #   Repository cloner, XML/CodeQL analyzer, and reporter
├── reproduce/                  # Research reproducibility scripts (Makefile, scripts)
├── smart/                      # Next.js web application (smart) — port 7000
├── tests/                      # Pytest test suite (uses MemoryRepository — no Neo4j needed)
├── data/                       # Topology JSONs and scenario YAMLs (8 domain scenarios)
├── models/                     # Trained GNN checkpoints
├── output/                     # Generated dashboards and reports
├── results/                    # Validation results from previous runs
├── docs/                       # Per-step methodology documentation
├── pyproject.toml              # Python package config, dependencies, entry points
└── docker-compose.yml          # Full-stack orchestration (single all-in-one container)
```

---

## 4. Key Patterns & Conventions

### Python Conventions
- **Repository pattern:** `saag/infrastructure/neo4j_repo.py` (production) and `saag/infrastructure/memory_repo.py` (testing) both implement `IGraphRepository` from `saag/core/ports/graph_repository.py`. Tests never require a live Neo4j instance.
- **SDK entry point:** Prefer `saag.Pipeline` for programmatic use. It handles repository lifecycle and stage orchestration. Use `saag.Client` for step-by-step control.
- **CLI scripts** in `cli/` import from `saag/` — all must be run from the repo root (e.g., `python cli/analyze_graph.py`).
- **API routers** in `api/routers/` follow a consistent pattern with dependency injection via `api/dependencies.py`. Each router delegates to a use case or service; it holds no business logic.
- **Graph layers:** Four layers: `app`, `infra`, `mw`, `system`. Canonical definitions are in `saag/core/layers.py` (`LAYER_DEFINITIONS`, `DEPENDENCY_TO_LAYER`). Key: the `app` layer includes both Application **and Library** nodes — library blast-radius risk is visible at this layer.
- **Dependency types:** Six `DEPENDS_ON` subtypes derived by `Neo4jRepository._derive_dependencies()`: `app_to_app`, `app_to_lib`, `app_to_broker`, `node_to_node`, `node_to_broker`, `broker_to_broker`. All carry `weight ∈ [0,1]` (max QoS severity) and `path_count` (coupling intensity).
- **Use cases:** Each pipeline stage has a dedicated `UseCase` class in `saag/usecases/`. These are the boundary between the API/CLI layer and the service layer.
- **Input data:** Topology JSON files in `data/` (e.g., `system.json`) and YAML scenario configs (`data/scenario_0N_*.yaml`).
- **Scenarios:** 8 domain scenarios (autonomous vehicle, IoT, financial trading, healthcare, hub-and-spoke, microservices, enterprise XL, tiny regression).

### Frontend Conventions
- **App Router** (Next.js `app/` directory).
- **Components** in `smart/components/` — follows shadcn/ui conventions.
- **API utilities** in `smart/lib/`.
- **OpenAPI-generated client** in `smart/lib/api/generated/`.

### Docker Conventions
- Single all-in-one container (`docker-compose.yml`): Neo4j + FastAPI + Next.js bundled in one image.
- Ports: 7474 (Neo4j Browser), 7687 (Bolt), 8000 (FastAPI), 7000 (Next.js).
- `Dockerfile` at root builds the all-in-one image.
- Neo4j plugins: APOC and Graph Data Science.

---

## 5. The Pipeline & Analytical Methodology

```
Generate → Model  → Analyze → Predict → Simulate → Validate → Visualize
Offline    Step 1   Step 2    Step 3    Step 4      Step 5      Step 6
Prep
```

Offline Prep: Generate — Produces a synthetic pub-sub topology JSON using `StatisticalGraphGenerator`.
1. **Model** — Converts topology JSON to a weighted directed graph in Neo4j; derives `DEPENDS_ON` edges.
2. **Analyze** — Deterministic structural scoring (RPR, Betweenness, Bridge Ratio, etc.), RMAV dimension mapping, and anti-pattern scans.
3. **Predict** — Inductive forecasting using a 3-layer **EdgeAwareHGTConv (HGT)** GNN: training and inference, blending GNN predictions and rule-based RMAV scores.
4. **Simulate** — Counterfactual cascade simulation engine injecting faults to calculate ground truth impact metrics.
5. **Validate** — Evaluates quality scores against simulation-derived ground truths using Spearman correlation, F1, and NDCG@K.
6. **Visualize** — Generates dashboards (web dashboard and static HTML).

### DEPENDS_ON Derivation Rules
`Neo4jRepository._derive_dependencies()` reads structural edges and emits `DEPENDS_ON` edges (direction: **dependent → dependency**).

| Rule | `dependency_type` | Source pattern | Weight |
|------|-------------------|----------------|--------|
| 1 | `app_to_app` | App_sub → App_pub via shared Topic; also transitive via `USES*1..3` chain | `max(t.weight)` |
| 2 | `app_to_broker` | App → Broker routing its topics; also transitive via `USES*1..3` chain | `max(t.weight)` |
| 3 | `node_to_node` | Lifted from Rule 1: Node_B → Node_A when hosted apps share an app_to_app edge | lifted `max(d.weight)` |
| 4 | `node_to_broker` | Lifted from Rule 2: Node → Broker when a hosted app has an app_to_broker edge | lifted `max(dep.weight)` |
| 5 | `app_to_lib` | App → Library (USES). Simultaneous multi-consumer blast, not sequential cascade. | `app.weight` (set in aggregate phase) |
| 6 | `broker_to_broker` | Bidirectional colocation edge between brokers sharing a physical Node. Symmetric shared-fate risk. | `node.weight` |

*Note: Simulation operates on **G_structural** (raw edges), not on DEPENDS_ON.*

### RMAV Prediction Formulas
Weights are derived via AHP with shrinkage factor λ=0.7 (blends with uniform prior).

#### Reliability — R(v) v6
```
R(v) = 0.45·RPR + 0.30·DG_in + 0.25·CDPot_enh
```
- **RPR**: Reverse PageRank (fault propagation reach).
- **DG_in**: Normalized in-degree (direct dependent count).
- **CDPot_enh**: Enhanced Cascade Depth Potential = `CDPot_base * (1 + MPCI)`.

#### Maintainability — M(v) v6
```
M(v) = 0.35·BT + 0.30·w_out + 0.15·CQP + 0.12·CouplingRisk + 0.08·(1 − CC)
```
- **BT**: Betweenness centrality (structural bottleneck position).
- **w_out**: QoS-weighted efferent coupling (outgoing dependency weight).
- **CQP**: Code Quality Penalty = `0.40·complexity_norm + 0.35·instability_code + 0.25·lcom_norm`.
  - `complexity_norm` and `lcom_norm` are normalized independently for Applications and Libraries (min-max).
- **CouplingRisk**: `1 − |2·Instability − 1|` where `Instability = DG_out / (DG_in + DG_out)`.
- **(1−CC)**: Inverse clustering coefficient.

#### Availability — A(v) v3
```
A(v) = 0.35·AP_c_directed + 0.25·QSPOF + 0.25·BR + 0.10·CDI + 0.05·w(v)
```
- **AP_c_directed**: `max(AP_c_out, AP_c_in)` — directional articulation point score.
- **QSPOF**: `AP_c_directed × w(v)` — QoS-scaled SPOF severity.
- **BR**: Bridge ratio (fraction of incident edges that are bridges).
- **CDI**: Connectivity Degradation Index.
- **w(v)**: Pure operational priority weight.

#### Vulnerability — V(v) v2
```
V(v) = 0.40·REV + 0.35·RCL + 0.25·QADS
```
- **REV**: Reverse Eigenvector centrality on G^T.
- **RCL**: Reverse Closeness centrality on G^T.
- **QADS**: QoS-weighted attack-dependent surface (`w_in`).

#### Overall Quality
```
Q(v) = 0.24·R(v) + 0.17·M(v) + 0.43·A(v) + 0.16·V(v)
```

#### Anti-Pattern Detection
The `AntiPatternDetector` audits quality results and flags architectural smells:

| Anti-Pattern | Trigger / Heuristic | Severity |
|---|---|---|
| **SPOF** | `is_articulation_point == True` | CRITICAL |
| **FAILURE_HUB** | `R(v) >= CRITICAL` | CRITICAL |
| **GOD_COMPONENT** | `M(v) >= CRITICAL` and `betweenness > 0.3` | CRITICAL |
| **TARGET** | `V(v) >= CRITICAL` | CRITICAL |
| **BRIDGE_EDGE** | `is_bridge == True` (Edge) | HIGH |
| **EXPOSURE** | `V(v) == HIGH` and `closeness > 0.6` | HIGH |
| **CYCLE** | Strongly Connected Component size >= 2 | HIGH |
| **HUB_AND_SPOKE** | `clustering < 0.1` and `degree > 3` | MEDIUM |
| **CHAIN** | Weakly connected sequence length >= 4 | MEDIUM |
| **SYSTEMIC_RISK** | `CRITICAL` nodes count > 20% of system | CRITICAL |

#### Predict Stage — GNN Ensemble
Integrates GNN predictions via an ensemble approach:
```
Q_ensemble(v) = α · Q_GNN + (1 - α) · Q_RMAV
```
- **Q_GNN**: Criticality learned via **EdgeAwareHGTConv (HGT)** message passing across heterogeneous node and edge types.
- **α**: Blending coefficient (learned per dimension during training, typically 0.6-0.8).

#### Classification (Box-Plot)
- `CRITICAL`: score > Q3 + k×IQR (k=0.75 by default).
- `HIGH`: score > Q3.
- `MEDIUM`: score > Median.
- `LOW`: score > Q1.
- `MINIMAL`: score ≤ Q1.
- Fallback for samples < 12: fixed percentile fallback (top 10% → CRITICAL, etc.)

### Simulation Ground Truths
Calculated via exhaustive fault injection using four concurrent post-passes.

#### Overall Impact — I(v)
```
I(v) = 0.35·reachability_loss + 0.25·fragmentation + 0.25·throughput_loss + 0.15·flow_disruption
```

#### Reliability Ground Truth — IR(v)
```
IR(v) = 0.45·CascadeReach + 0.35·WeightedCascadeImpact + 0.20·NormalizedCascadeDepth
```
Measures fault-propagation dynamics (cascade spread and depth); orthogonal to connectivity-loss.

#### Maintainability Ground Truth — IM(v)
```
IM(v) = 0.45·ChangeReach + 0.35·WeightedChangeImpact + 0.20·NormalizedChangeDepth
```
Computed by `ChangePropagationSimulator` via BFS on the transposed DEPENDS_ON graph G^T (stops at loose-coupling or stable interfaces).

#### Availability Ground Truth — IA(v)
```
IA(v) = 0.50·WeightedReachabilityLoss + 0.35·WeightedFragmentation + 0.15·PathBreakingThroughputLoss
```
QoS-weighted connectivity disruption from removing v.

#### Vulnerability Ground Truth — IV(v)
```
IV(v) = 0.40·AttackReach + 0.35·WeightedAttackImpact + 0.25·HighValueContamination
```
Computed by `CompromisePropagationSimulator` via BFS on G^T with a trust threshold θ_trust=0.30.

### Validation Metrics (per dimension)
Validation calculates Spearman targets and other performance metrics:

| Dimension | Spearman Target | Additional Metrics |
|---|---|---|
| **Overall** | ρ(Q, I) | F1, Precision, Recall, NDCG@K, Top-5/10 overlap, RMSE |
| **Reliability** | ρ(R, IR) | CCR@5 (Cascade Capture Rate), CME (Cascade Magnitude Error) |
| **Maintainability** | ρ(M, IM) | COCR@5 (Change Overlap Capture Rate), weighted-κ CTA, Bottleneck Precision |
| **Availability** | ρ(A, IA) | SPOF_F1 (SPOF classification F1), RRI (Robustness Rank Improvement) |
| **Vulnerability** | ρ(V, IV) | AHCR@5 (Attack Hit Capture Rate), FTR (False Trust Rate), APAR (Attack Path Agreement Rate), CDCC (Cross-Dimensional Contamination Check) |
