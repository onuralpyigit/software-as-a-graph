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
  - `core/` — domain models (`models.py`), metrics, layers, Neo4j repository (`neo4j_repo.py`), memory repository, criticality, exporters/importers
  - `analysis/` — structural and quality analysis services
  - `simulation/` — event and failure simulation services
  - `validation/` — validation logic
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
- `generate_graph.py` — Synthetic graph data generation
- `import_graph.py` — Import graph data into Neo4j
- `analyze_graph.py` — Structural and quality analysis
- `simulate_graph.py` — Cascade failure simulation (subcommands: `failure`, `event`, `report`)
- `validate_graph.py` — Statistical validation
- `visualize_graph.py` — Static HTML dashboard generation
- `benchmark.py` — Benchmarking across scales
- `export_graph.py` — Export graph data

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

# Run CLI pipeline
python bin/run.py --all --layer system
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
**Test files:** `backend/tests/test_*.py` — organized by service (analysis, simulation, validation, visualization, benchmark, generation, domain model, CLI, API statistics)

## Key Patterns & Conventions

### Python Backend
- **No `pyproject.toml` or `setup.py`** — dependencies managed via `requirements.txt`; modules imported with relative/absolute paths from the project root
- **Repository pattern:** `core/neo4j_repo.py` (production) and `core/memory_repo.py` (testing) implement the same interface
- **CLI scripts** in `bin/` import from `backend/src/` — they must be run from the repo root (e.g., `python bin/analyze_graph.py`)
- **API routers** in `backend/api/routers/` follow a consistent pattern with dependency injection via `backend/api/dependencies.py`
- **Graph layers:** The system supports four graph layers: `app`, `infra`, `mw` (middleware), `system`
- **Input data:** Topology JSON files in `input/` (e.g., `system.json`, `dataset.json`) and YAML configs (`graph_config.yaml`)

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
2. **Structural Analysis** — Computes centrality metrics (PageRank, Betweenness, Closeness, etc.)
3. **Quality Scoring** — Maps metrics to quality dimensions (RMAV) using AHP weights
4. **Failure Simulation** — Injects faults, measures cascade impact (exhaustive or Monte Carlo)
5. **Validation** — Statistically compares predictions against simulation ground truth
6. **Visualization** — Generates interactive dashboards (web or static HTML)

## Documentation

- `docs/` contains detailed documentation for each pipeline step as well as SDD, SRS, and STD documents
- `examples/` contains runnable example scripts for programmatic API usage
