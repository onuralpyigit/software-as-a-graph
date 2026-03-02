# CLAUDE.md

## Project Overview

**Software-as-a-Graph** predicts which components in a distributed publish-subscribe system will cause the most damage when they fail, using only the system's architecture. It models system topology as a weighted directed graph and applies topological analysis (centrality metrics, quality scoring) to identify critical components — validated against cascade failure simulations (Spearman > 0.87, F1 > 0.90).

Published at IEEE RASSE 2025.

## Architecture

The project is a full-stack application with three main components:

### Backend (`backend/`)
- **Language:** Python 3.9+
- **API:** FastAPI (`backend/api/main.py`) with routers for health, graph, analysis, components, statistics, simulation, classification, and validation
- **Source code:** `backend/src/` — modular packages:
  - `core/` — domain models (`models.py`), metrics, layers, Neo4j repository (`neo4j_repo.py`), memory repository, criticality, exporters/importers, interfaces
  - `analysis/` — structural analyzer, quality analyzer, weight calculator (AHP), classifier, statistics service, problem detector
  - `simulation/` — event simulator, failure simulator, change propagation simulator, compromise propagation simulator, simulation graph, models, service
  - `validation/` — validation logic, validator, metric calculator, models
  - `visualization/` — dashboard and visualization services
  - `generation/` — graph generation logic
  - `benchmark/` — benchmarking services
  - `cli/` — CLI utilities
- **Database:** Neo4j 5.x (accessed via `neo4j` Python driver)
- **Dependencies:** `backend/requirements.txt` — `neo4j`, `networkx`, `fastapi`, `uvicorn`, `pydantic`, `matplotlib`, `numpy`, `scipy`

### Frontend (`frontend/`)
- **Name:** Genieus
- **Framework:** Next.js 16 with React 19, TypeScript
- **Styling:** Tailwind CSS 4
- **UI Components:** Radix UI primitives, shadcn/ui pattern (`components.json`)
- **Key libraries:** `recharts` (charts), `react-force-graph-2d`/`3d` (graph visualization), `axios` (HTTP), `zod` (validation), `react-hook-form`
- **API client:** Auto-generated from OpenAPI spec via `npm run generate-client`
- **Dev server port:** 7000 (`next dev -p 7000`)

### CLI Tools (`bin/`)
Pipeline scripts that can run independently or via the orchestrator:
- `run.py` — End-to-end pipeline orchestrator
- `run_scenarios.sh` — Batch-run the pipeline across all 8 scenario datasets
- `generate_graph.py` — Synthetic graph data generation
- `import_graph.py` — Import graph data into Neo4j
- `analyze_graph.py` — Structural and quality analysis
- `simulate_graph.py` — Cascade failure simulation (subcommands: `failure`, `event`, `report`)
- `validate_graph.py` — Statistical validation
- `visualize_graph.py` — Static HTML dashboard generation
- `benchmark.py` — Benchmarking across scales
- `export_graph.py` — Export graph data
- `ground_threshold.py` — SPOF threshold grounding: sweeps all 8 scenarios, computes F1/AUC to justify the I(v) > 0.5 SPOF threshold empirically

## Development

### Prerequisites
- Python 3.9+ with a virtual environment (`software_system_env/`)
- Node.js (for frontend)
- Neo4j 5.x (local or via Docker)
- Docker & Docker Compose (for full-stack deployment)

### Running the Full Stack (Docker)
```bash
docker compose up --build
```
- **Web Dashboard:** http://localhost:7000
- **API (Swagger docs):** http://localhost:8000/docs
- **Neo4j Browser:** http://localhost:7474 (neo4j/password)

### Environment Variables
Root `.env` (used by Docker Compose):
```
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_URI=bolt://neo4j:7687
NEXT_PUBLIC_API_URL=http://localhost:8000
```

`backend/.env` (used for local dev):
```
NEO4J_HOST=localhost
NEO4J_BOLT_PORT=7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### Running Backend Locally
```bash
# Install dependencies
pip install -r backend/requirements.txt

# Start the API server
cd backend && uvicorn api.main:app --reload --port 8000

# Run CLI pipeline (single scenario)
python bin/run.py --all --layer system

# Run all 8 scenarios
bash bin/run_scenarios.sh
```

### Running Frontend Locally
```bash
cd frontend
npm install
npm run dev          # http://localhost:7000
npm run generate-client  # Regenerate API client from OpenAPI spec
```

### Running Tests
```bash
cd backend
pytest               # All tests (verbose, short traceback by default)
pytest -x            # Stop on first failure
pytest tests/test_analysis_service.py  # Single test file
pytest -k "test_name"  # Run by test name pattern
```

**Test configuration:** `backend/pytest.ini`  
**Test markers:** `slow` (skip with `--quick`), `integration`  
**Test timeout:** 120 seconds per test  
**Test files:** `backend/tests/test_*.py` — coverage includes:
  - `test_analysis_service.py` — structural & quality analysis
  - `test_simulation_service.py` — failure and event simulation
  - `test_validation_service.py` — validation pipeline
  - `test_visualization_service.py` — dashboard generation
  - `test_benchmark_service.py` — benchmark service
  - `test_domain_model.py` — core models
  - `test_cli.py` — all CLI scripts
  - `test_api_statistics.py` — API statistics endpoints
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

## Key Patterns & Conventions

### Python Backend
- **No `pyproject.toml` or `setup.py`** — dependencies managed via `requirements.txt`; modules imported with relative/absolute paths from the project root
- **Repository pattern:** `core/neo4j_repo.py` (production) and `core/memory_repo.py` (testing) implement the same interface
- **CLI scripts** in `bin/` import from `backend/src/` — they must be run from the repo root (e.g., `python bin/analyze_graph.py`)
- **API routers** in `backend/api/routers/` follow a consistent pattern with dependency injection via `backend/api/dependencies.py`
- **Graph layers:** The system supports four graph layers: `app`, `infra`, `mw` (middleware), `system`
- **Input data:** Topology JSON files in `input/` (e.g., `system.json`, `dataset.json`) and YAML configs (`graph_config.yaml`)
- **Scenarios:** 8 domain scenarios under `input/scenario_0N_*.yaml` (autonomous vehicle, IoT, financial trading, healthcare, hub-and-spoke, microservices, enterprise XL, tiny regression)

### Frontend
- **App Router** (Next.js `app/` directory)
- **Components** in `frontend/components/` — follows shadcn/ui conventions
- **API utilities** in `frontend/lib/`
- **OpenAPI-generated client** in `frontend/lib/api/generated/`

### Docker
- Three services: `neo4j`, `api` (builds from `backend/Dockerfile`), `frontend` (builds from `frontend/Dockerfile`)
- `Dockerfile.all-in-one` and `Dockerfile.api` exist at root for alternative deployment
- Neo4j plugins: APOC and Graph Data Science

## The 6-Step Pipeline

```
Architecture → Graph → Metrics → Scores → Simulation → Validation → Dashboard
   (input)     Step 1   Step 2    Step 3    Step 4       Step 5       Step 6
```

1. **Graph Model** — Converts topology JSON to a weighted directed graph in Neo4j
2. **Structural Analysis** — Computes centrality metrics (Reverse PageRank, Betweenness, Closeness, Eigenvector, Reverse variants, Bridge Ratio, Clustering, etc.)
3. **Prediction** — Maps metrics to RMAV dimensions using AHP-derived weights (see below)
4. **Failure Simulation** — Injects faults; runs four parallel ground-truth simulators; exhaustive or Monte Carlo modes
5. **Validation** — Per-dimension statistical comparison: overall Q(v) vs I(v) plus per-RMAV-dimension comparators
6. **Visualization** — Generates interactive dashboards (web or static HTML)

## RMAV Prediction Formulas

Quality scores are computed per component v. Weights are derived via AHP with shrinkage factor λ=0.7 (blends with uniform prior).

### Reliability — R(v) v4
```
R(v) = 0.40·RPR + 0.35·w_in + 0.25·CDPot
```
- **RPR**: Reverse PageRank (fault propagation reach)
- **w_in**: QoS-weighted in-degree (dependent count weighted by topic priority/QoS)
- **CDPot**: Cascade Depth Potential = `((RPR + w_in) / 2) * (1 - min(w_out / w_in, 1))`

### Maintainability — M(v) v5
```
M(v) = 0.40·BT + 0.35·w_out + 0.15·CouplingRisk + 0.10·(1 − CC)
```
- **BT**: Betweenness centrality (structural bottleneck position)
- **w_out**: QoS-weighted efferent coupling (outgoing dependency weight)
- **CouplingRisk**: `1 − |2·Instability − 1|` where `Instability = DG_out / (DG_in + DG_out)` — maximised at 0.5 (deeply embedded on both sides)
- **(1−CC)**: Inverse clustering coefficient (direction-agnostic proxy, reduced weight)

### Availability — A(v) v2
```
A(v) = 0.45·QSPOF + 0.30·BR + 0.15·AP_c_directed + 0.10·CDI
```
- **QSPOF**: `AP_c_directed × w(v)` — QoS-scaled SPOF severity
- **BR**: Bridge ratio (fraction of incident edges that are bridges)
- **AP_c_directed**: `max(AP_c_out, AP_c_in)` — worst-case directional articulation point score; `AP_c_out = 1 - |largest_CC(G \ {v})| / (|V|−1)`
- **CDI**: Connectivity Degradation Index — normalised increase in average path length after removing v

### Vulnerability — V(v) v2
```
V(v) = 0.40·REV + 0.35·RCL + 0.25·QADS
```
- **REV**: Reverse Eigenvector centrality on G^T (strategic attack reach)
- **RCL**: Reverse Closeness centrality on G^T (adversarial propagation speed)
- **QADS**: QoS-weighted attack-dependent surface (w_in — inbound dependency weight)

### Overall Quality
```
Q(v) = 0.25·R(v) + 0.25·M(v) + 0.25·A(v) + 0.25·V(v)   [default equal weights]
```
Dimension weights are configurable; defaults assume balanced system priorities.

### Classification (Box-Plot)
- `CRITICAL`: score > Q3 + k×IQR (k=0.75 by default)
- `HIGH`: score > Q3
- `MEDIUM`: score > Median
- `LOW`: score > Q1
- `MINIMAL`: score ≤ Q1
- For samples < 12: fixed percentile fallback (top 10% → CRITICAL, etc.)

## Simulation Ground Truths

The failure simulator runs four concurrent post-passes after exhaustive simulation, producing per-RMAV ground truth values for each component.

### Overall Impact — I(v)
```
I(v) = 0.35·reachability_loss + 0.25·fragmentation + 0.25·throughput_loss + 0.15·flow_disruption
```
- `flow_disruption`: fraction of event-simulation flows interrupted by v's failure

### Reliability Ground Truth — IR(v)
```
IR(v) = 0.45·CascadeReach + 0.35·WeightedCascadeImpact + 0.20·NormalizedCascadeDepth
```
Measures fault-propagation dynamics (cascade spread and depth); orthogonal to connectivity-loss (Availability).

### Maintainability Ground Truth — IM(v)
```
IM(v) = 0.45·ChangeReach + 0.35·WeightedChangeImpact + 0.20·NormalizedChangeDepth
```
Computed by `ChangePropagationSimulator` via BFS on the transposed DEPENDS_ON graph G^T.
- Stop conditions: loose-coupling (edge weight < θ_loose=0.20) and stable-interface (Instability(u) < θ_stable=0.20)
- Models development-time change propagation, not runtime failure.

### Availability Ground Truth — IA(v)
```
IA(v) = 0.50·WeightedReachabilityLoss + 0.35·WeightedFragmentation + 0.15·PathBreakingThroughputLoss
```
QoS-weighted connectivity disruption from removing v; orthogonal to cascade-propagation (IR(v)).

### Vulnerability Ground Truth — IV(v)
```
IV(v) = 0.40·AttackReach + 0.35·WeightedAttackImpact + 0.25·HighValueContamination
```
Computed by `CompromisePropagationSimulator` via BFS on G^T with a trust threshold θ_trust=0.30.
- Models adversarial compromise propagation over trusted dependency graph.

## Validation Metrics (per dimension)

Each RMAV dimension has its own specialist validator and set of metrics in `validation/metric_calculator.py`.

| Dimension | Spearman Target | Additional Metrics |
|---|---|---|
| **Overall** | ρ(Q, I) | F1, Precision, Recall, NDCG@K, Top-5/10 overlap, RMSE |
| **Reliability** | ρ(R, IR) | CCR@5 (Cascade Capture Rate), CME (Cascade Magnitude Error) |
| **Maintainability** | ρ(M, IM) | COCR@5 (Change Overlap Capture Rate), weighted-κ CTA, Bottleneck Precision |
| **Availability** | ρ(A, IA) | SPOF_F1 (SPOF classification F1), RRI (Robustness Rank Improvement) |
| **Vulnerability** | ρ(V, IV) | AHCR@5 (Attack Hit Capture Rate), FTR (False Trust Rate), APAR (Attack Path Agreement Rate), CDCC (Cross-Dimensional Contamination Check) |

Validation also reports statistical power tables and Spearman–Kendall gap diagnostics.

## Documentation

- `docs/` contains detailed documentation for each pipeline step:
  - `graph-model.md`, `structural-analysis.md`, `prediction.md`, `failure-simulation.md`, `validation.md`, `visualization.md`
  - `SDD.md` (Software Design Description), `SRS.md` (Requirements), `STD.md` (Test Description)
- `examples/` contains runnable example scripts for programmatic API usage
- `output/` — pipeline output artefacts (dashboards, reports, exported graphs)
- `results/` — validation results from previous runs
- `benchmarks/` — benchmark data
