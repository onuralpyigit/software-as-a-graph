# Software-as-a-Graph

**Predict which components in a distributed system will cause the most damage when they fail — using only the system's architecture.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![Docker](https://img.shields.io/badge/docker-compose-blue)](https://www.docker.com/)
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-green.svg)](https://neo4j.com/)
[![IEEE RASSE 2025](https://img.shields.io/badge/IEEE-RASSE%202025-orange.svg)](#citation)

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Our Approach](#our-approach)
3. [Empirical Validation](#empirical-validation)
4. [Supported Platforms](#supported-platforms)
5. [Quick Start (Docker)](#quick-start-docker)
6. [Web Interface (Genieus)](#web-interface-genieus)
7. [Development Setup (CLI)](#development-setup-cli)
8. [How It Works — The 6-Step Pipeline](#how-it-works--the-6-step-pipeline)
9. [RMAV Prediction](#rmav-prediction)
10. [Project Structure](#project-structure)
11. [Research Context](#research-context)
12. [Citation](#citation)
13. [License](#license)

---

## The Problem

In distributed publish-subscribe systems (ROS 2, Kafka, MQTT, etc.), some components are far more critical than others. When they fail, failures cascade through the system. Identifying these weak points traditionally requires either expensive runtime monitoring or waiting for production incidents.

This reactive approach has two fundamental problems: runtime monitoring adds overhead to production systems, and by the time a critical failure is discovered, the damage is already done.

## Our Approach

We treat your system architecture as a graph and apply topological analysis to **predict** critical components **before deployment**. The key insight is:

> **A component's position in the dependency graph reliably predicts its real-world failure impact.**

Applications, brokers, topics, and infrastructure nodes are modelled as vertices. Their publish-subscribe relationships become weighted directed edges, with weights derived from QoS settings (reliability, durability, priority, message size). Graph-theoretic metrics — Reverse PageRank, betweenness centrality, articulation point detection, bridge ratio, and more — are combined into an **RMAV composite quality score** (Reliability, Maintainability, Availability, Vulnerability) using AHP-derived weights.

Ground-truth impact scores are produced by four parallel failure simulators (cascade, change-propagation, connectivity-loss, and compromise-propagation), each aligned to one RMAV dimension.

This shifts the analysis paradigm from reactive runtime monitoring to **proactive pre-deployment prediction**.

## Empirical Validation

The methodology has been empirically validated across 8 scenario datasets spanning multiple system scales and domains:

| Metric | Target | Achieved |
|--------|--------|----------|
| Spearman correlation ρ(Q*, I*) | ≥ 0.85 | **> 0.85** |
| Overall F1-score | ≥ 0.90 | **> 0.90** |
| Predictive Gain (PG) vs. degree baseline | > 0.03 | **> 0.03** |
| Analysis layer | — | Application layer outperforms infrastructure layer |
| Scale effect | — | Prediction accuracy improves with system size |

Validation domains include ROS 2 autonomous vehicles, IoT smart cities, financial trading platforms, and healthcare systems.

## Supported Platforms

The graph model maps naturally to any pub-sub middleware:

| Graph Concept | ROS 2 / DDS | Apache Kafka | MQTT |
|---------------|-------------|--------------|------|
| Application | ROS Node | Producer / Consumer | MQTT Client |
| Topic | ROS Topic | Kafka Topic | MQTT Topic |
| Broker | DDS Participant | Kafka Broker | MQTT Broker |
| Infrastructure Node | Host / Container | Broker Host | Broker Server |

---

## Quick Start (Docker)

The fastest way to run the full system — web dashboard, REST API, and graph database — is via Docker Compose.

**Prerequisites:** Docker & Docker Compose, 4 GB+ RAM available.

```bash
git clone https://github.com/<your-org>/software-as-a-graph.git
cd software-as-a-graph

# Build and start all services
docker compose up --build
```

Once running, open:

| Service | URL | Credentials |
|---------|-----|-------------|
| Web Dashboard (Genieus) | http://localhost:7000 | — |
| REST API (FastAPI docs) | http://localhost:8000/docs | — |
| Neo4j Browser | http://localhost:7474 | `neo4j` / `password` |

> **Note:** Default Neo4j credentials (`neo4j` / `password`) are for local development only. Change them in the root `.env` file before any shared or production deployment.

### Running a Pre-Built Image

```bash
docker run --name genieus --network host genieus:1.0.0
```

`--network host` allows the container to connect to Neo4j running on the host (port 7687). Useful for deploying a specific image alongside an existing Neo4j instance.

---

## Web Interface (Genieus)

The **Genieus** dashboard (Next.js 16, React 19, TypeScript) provides an interactive frontend for the full analysis pipeline:

1. **Dashboard** — High-level KPIs, criticality distribution heatmap, and top critical component list.
2. **Graph Explorer** — Interactive 2D/3D force-directed graph. Filter by layer (app / infra / middleware / system), search components, and inspect dependency details.
3. **Analysis** — Trigger structural analysis and RMAV quality scoring for a selected layer.
4. **Simulation** — Simulate component failures and visualise cascade propagation paths.
5. **Settings** — Configure the Neo4j connection (URI, credentials, database name).

---

## Development Setup (CLI)

Use the CLI when you want to run individual pipeline stages, integrate with scripts, or work without the frontend.

### Prerequisites

- Python 3.9+ (virtual environment recommended — `software_system_env/` is `.gitignore`d)
- Neo4j 5.x (via Docker or a local installation with the APOC and GDS plugins)
- Node.js 18+ (frontend only)

### Backend & CLI

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Copy and configure environment variables
cp backend/.env.example backend/.env
# Edit backend/.env: set NEO4J_HOST, NEO4J_BOLT_PORT, NEO4J_USERNAME, NEO4J_PASSWORD

# Run the full pipeline in one command
python bin/run.py --all --layer system
```

### Frontend (optional)

```bash
cd frontend
npm install
npm run dev           # http://localhost:7000

# Regenerate the API client from the OpenAPI spec (after backend changes)
npm run generate-client
```

### Individual Pipeline Scripts

Each step has its own CLI script in `bin/`. All scripts must be run from the repo root:

```bash
# Step 1 — Modeling: generate synthetic topology & import into Neo4j
python bin/generate_graph.py --scale medium --output input/system.json
python bin/import_graph.py --input input/system.json --clear

# Step 2 — Analysis: structural metrics (13 topological indicators per component)
python bin/analyze_graph.py --layer system

# Step 3 — Prediction: RMAV quality scoring (AHP-weighted criticality classification)
python bin/analyze_graph.py --layer system --use-ahp

# Step 4 — Simulation: failure simulation (produces per-dimension ground-truth scores)
python bin/simulate_graph.py failure --layer system --exhaustive

# Step 5 — Validation: statistical validation (Spearman ρ, F1-score, per-RMAV metrics)
python bin/validate_graph.py --layer system

# Step 6 — Visualization: interactive HTML dashboard (self-contained, no server needed)
python bin/visualize_graph.py --layer system --output output/dashboard.html --open
```

The `--layer` flag accepts `app`, `infra`, `mw`, or `system` (all layers combined).

Additional utility scripts:

```bash
# Export graph data from Neo4j
python bin/export_graph.py --layer system --output output/graph_export.json

# Benchmark across all scale presets
python bin/benchmark.py

# Run the full pipeline across all 8 domain scenarios
bash bin/run_scenarios.sh

# Ground the SPOF threshold empirically across all 8 scenarios
python bin/ground_threshold.py
```

### Running Tests

```bash
cd backend
pytest                # All tests
pytest -x             # Stop on first failure
pytest tests/test_analysis_service.py   # Single file
pytest -k "reliability"                 # Filter by name
```

### Programmatic API

For Python-based integration, see the annotated examples in `examples/`:

| File | What it demonstrates |
|------|----------------------|
| `examples/example_end_to_end.py` | Full pipeline from modeling to validation |
| `examples/example_generation.py` | Generating topology data programmatically |
| `examples/example_import.py` | Importing a graph into Neo4j via the Python API |
| `examples/example_analysis.py` | Running structural analysis |
| `examples/example_simulation.py` | Running failure simulations |
| `examples/example_validation.py` | Validating predictions against ground truth |
| `examples/example_visualization.py` | Generating HTML dashboards |

---

## How It Works — The 6-Step Pipeline

```
Architecture JSON
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Step 1     │    │  Step 2     │    │  Step 3     │
│  Modeling   │───▶│  Analysis   │───▶│  Prediction │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
      ┌──────────────────────────────────────┘
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Step 4     │    │  Step 5     │    │  Step 6     │
│  Simulation │───▶│  Validation │───▶│  Visuali-   │
│             │    │             │    │  zation     │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
                                      HTML Dashboard
```

| Step | What It Does | Key Output |
|------|-------------|------------|
| **1. [Modeling](docs/graph-model.md)** | Converts topology JSON to a weighted directed graph G(V, E, w); derives DEPENDS_ON edges from pub-sub relationships | G_structural and G_analysis(l) |
| **2. [Analysis](docs/structural-analysis.md)** | Computes Reverse PageRank, betweenness, closeness, eigenvector centralities, bridge ratio, articulation points, clustering | Metric vector M(v) per component |
| **3. [Prediction](docs/prediction.md)** | Maps M(v) to RMAV dimensions using AHP-derived weights; classifies criticality via box-plot adaptive thresholds | Q*(v) composite score and Q(v) ∈ {MINIMAL, LOW, MEDIUM, HIGH, CRITICAL} |
| **4. [Simulation](docs/failure-simulation.md)** | Runs four parallel simulators (cascade, change-propagation, connectivity-loss, compromise-propagation) | Per-dimension ground-truth scores IR(v), IM(v), IA(v), IV(v) and composite I*(v) |
| **5. [Validation](docs/validation.md)** | Computes Spearman ρ and Kendall τ between Q*(v) and I*(v) and per-dimension pairs; evaluates F1, NDCG@K, Top-K overlap, and specialist metrics | Statistical evidence of predictive validity |
| **6. [Visualization](docs/visualization.md)** | Renders interactive dashboards with network graphs, dependency matrices, and layer comparison views | `dashboard.html` (fully self-contained) |

### Scale Presets

The synthetic generator supports five scale presets for rapid experimentation:

| Preset | Apps | Topics | Brokers | Nodes | Typical Use |
|--------|------|--------|---------|-------|-------------|
| `tiny` | 5–8 | 3–5 | 1 | 2–3 | Unit tests |
| `small` | 10–15 | 8–12 | 2 | 3–4 | Quick checks |
| `medium` | 20–35 | 15–25 | 3–5 | 5–8 | Development |
| `large` | 50–80 | 30–50 | 5–8 | 8–12 | Integration tests |
| `xlarge` | 100–200 | 60–100 | 8–15 | 15–25 | Performance benchmarks |

---

## RMAV Prediction

Quality scores are computed per component v. AHP weights use a shrinkage factor λ=0.7 (blending learned weights with a uniform prior for robustness on small graphs).

| Dimension | Formula | Captures |
|-----------|---------|----------|
| **R** — Reliability | `0.45·RPR + 0.30·DG_in + 0.25·CDPot` | Fault propagation blast radius |
| **M** — Maintainability | `0.40·BT + 0.35·CouplingRisk + 0.25·CC_inv` | Coupling complexity and change fragility |
| **A** — Availability | `0.45·AP_c + 0.35·BR + 0.20·QSPOF` | Single-point-of-failure risk |
| **V** — Vulnerability | `0.40·PR + 0.35·w_in + 0.25·EV` | Attack surface and compromise exposure |

Criticality is classified into five levels — MINIMAL, LOW, MEDIUM, HIGH, CRITICAL — using adaptive box-plot thresholds derived from the actual score distribution of the system under analysis.

---

## Project Structure

```
.
├── bin/                        # CLI pipeline scripts
│   ├── run.py                  #   Master pipeline runner (--all flag)
│   ├── generate_graph.py       #   Synthetic topology generator
│   ├── import_graph.py         #   Step 1: Modeling — Neo4j import
│   ├── analyze_graph.py        #   Steps 2 & 3: Analysis + Prediction
│   ├── simulate_graph.py       #   Step 4: Simulation
│   ├── validate_graph.py       #   Step 5: Validation
│   ├── visualize_graph.py      #   Step 6: Visualization
│   ├── export_graph.py         #   Export graph data from Neo4j
│   ├── benchmark.py            #   Benchmark across scale presets
│   ├── run_scenarios.sh        #   Full pipeline across 8 domain scenarios
│   └── ground_threshold.py     #   Empirical SPOF threshold grounding
│
├── backend/                    # Python backend (hexagonal architecture)
│   ├── api/                    #   FastAPI application & routers
│   ├── src/                    #   Domain source code
│   │   ├── core/               #     Domain models, ports, Neo4j & memory repos
│   │   ├── analysis/           #     Structural metrics + RMAV quality scoring
│   │   ├── simulation/         #     Four parallel failure/event simulators
│   │   ├── validation/         #     Per-dimension statistical validation
│   │   ├── visualization/      #     Dashboard & chart generation
│   │   ├── generation/         #     Synthetic graph generation
│   │   ├── benchmark/          #     Benchmarking services
│   │   └── cli/                #     Shared CLI utilities
│   ├── tests/                  #   Pytest unit & integration tests (24 files)
│   └── requirements.txt        #   Python dependencies
│
├── examples/                   # Annotated Python usage examples
├── input/                      # Topology JSON & YAML scenario configs (8 scenarios)
├── output/                     # Pipeline output artefacts (dashboards, reports)
├── results/                    # Validation results from previous runs
├── benchmarks/                 # Benchmark data
└── docs/                       # Per-step methodology documentation
    ├── graph-model.md          #   Step 1: Modeling
    ├── structural-analysis.md  #   Step 2: Analysis
    ├── prediction.md      #   Step 3: Prediction
    ├── failure-simulation.md   #   Step 4: Simulation
    ├── validation.md           #   Step 5: Validation
    ├── visualization.md        #   Step 6: Visualization
    ├── SRS.md                  #   Software Requirements Specification
    ├── SDD.md                  #   Software Design Description
    └── STD.md                  #   Software Test Description
```

---

## Research Context

This framework is the software artifact for the PhD dissertation **"Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems"** at Istanbul Technical University, Department of Computer Engineering.

The underlying methodology was peer-reviewed and published at:

> **IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE 2025)**
> *A Graph-Based Dependency Analysis Method for Identifying Critical Components in Distributed Publish-Subscribe Systems*

The primary research contribution is the demonstration that **topological graph metrics can reliably predict real-world failure impact without runtime instrumentation**, validated empirically across four application domains (autonomous vehicles, IoT, financial trading, healthcare) and multiple system scales.

---

## Citation

If you use this framework or methodology in your research, please cite:

```bibtex
@inproceedings{yigit2025graphbased,
  title     = {A Graph-Based Dependency Analysis Method for Identifying
               Critical Components in Distributed Publish-Subscribe Systems},
  author    = {Yigit, Ibrahim Onuralp and Buzluca, Feza},
  booktitle = {2025 IEEE International Conference on Recent Advances in
               Systems Science and Engineering (RASSE)},
  year      = {2025},
  doi       = {10.1109/RASSE64831.2025.11315354}
}
```

---

## License

See [LICENSE](LICENSE) for terms of use.