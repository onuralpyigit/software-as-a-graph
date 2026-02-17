# Software-as-a-Graph

**Predict which components in a distributed system will cause the most damage when they fail — using only the system's architecture.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black)](https://nextjs.org/)
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
9. [Project Structure](#project-structure)
10. [Research Context](#research-context)
11. [Citation](#citation)
12. [License](#license)

---

## The Problem

In distributed publish-subscribe systems (ROS 2, Kafka, MQTT, etc.), some components are far more critical than others. When they fail, failures cascade through the system. Identifying these weak points traditionally requires either expensive runtime monitoring or waiting for production incidents.

This reactive approach has two fundamental problems: runtime monitoring adds overhead to production systems, and by the time a critical failure is discovered, the damage is already done.

## Our Approach

We treat your system architecture as a graph and apply topological analysis to **predict** critical components **before deployment**. The key insight is:

> **A component's position in the dependency graph reliably predicts its real-world failure impact.**

Applications, brokers, topics, and infrastructure nodes are modeled as vertices. Their publish-subscribe relationships become weighted directed edges, with weights derived from QoS settings (reliability, durability, priority, message size). Graph-theoretic metrics — PageRank, betweenness centrality, articulation point detection — are then combined into an **RMAV quality score** (Reliability, Maintainability, Availability, Vulnerability) using AHP-derived weights.

This shifts the analysis paradigm from reactive runtime monitoring to **proactive pre-deployment prediction**.

## Empirical Validation

The methodology has been empirically validated across multiple system scales and domains:

| Metric | Target | Achieved |
|--------|--------|----------|
| Spearman correlation (ρ) | ≥ 0.70 | **0.876** |
| F1-score | ≥ 0.90 | **> 0.90** |
| Analysis layer | — | Application layer outperforms infrastructure layer |
| Scale effect | — | Prediction accuracy improves with system size |

Validation domains include ROS 2 autonomous vehicles, IoT smart city deployments, financial trading platforms, and healthcare systems.

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

# Start the full stack
docker compose up --build
```

Once running, open:

| Service | URL | Credentials |
|---------|-----|-------------|
| Web Dashboard (Genieus) | http://localhost:7000 | — |
| REST API (FastAPI docs) | http://localhost:8000/docs | — |
| Neo4j Browser | http://localhost:7474 | `neo4j` / `password` |

> **Note:** Default Neo4j credentials (`neo4j` / `password`) are for local development only. Change them via the `NEO4J_AUTH` environment variable in `docker-compose.yml` before any shared or production deployment.

---

## Web Interface (Genieus)

The **Genieus** dashboard provides an interactive frontend for the full analysis pipeline:

1. **Dashboard** — High-level KPIs, criticality distribution heatmap, and top critical component list.
2. **Graph Explorer** — Interactive 2D/3D force-directed graph. Filter by layer (app / infra / middleware / system), search components, and inspect dependency details.
3. **Analysis** — Trigger structural analysis and quality scoring for a selected layer.
4. **Simulation** — Simulate component failures and visualise cascade propagation paths.
5. **Settings** — Configure the Neo4j connection (URI, credentials, database name).

---

## Development Setup (CLI)

Use the CLI when you want to run individual pipeline stages, integrate with scripts, or work without the frontend.

### Prerequisites

- Python 3.9+
- Neo4j 5.x (either via Docker or a local installation with the APOC and GDS plugins)
- Node.js 18+ (frontend only)

### Backend & CLI

```bash
# Install Python dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env: set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD as needed

# Run the full pipeline in one command
python bin/run.py --all --layer system
```

### Frontend (optional)

```bash
cd genieus
npm install
npm run dev
# Runs on http://localhost:7000
```

### Individual Pipeline Scripts

Each step of the methodology has its own CLI script in `bin/`:

```bash
# Step 1 — Generate synthetic topology & import into Neo4j
python bin/generate_graph.py --scale medium --output data/system.json
python bin/import_graph.py --input data/system.json --clear

# Step 2 & 3 — Structural analysis + RMAV quality scoring
python bin/analyze_graph.py --layer system

# Step 4 — Failure simulation (produces ground-truth impact scores)
python bin/simulate_graph.py failure --layer system --exhaustive

# Step 5 — Statistical validation (Spearman ρ, F1-score, etc.)
python bin/validate_graph.py --layer system

# Step 6 — Interactive HTML dashboard
python bin/visualize_graph.py --layer system --output dashboard.html --open
```

The `--layer` flag accepts `app`, `infra`, `mw`, or `system` (all layers combined).

### Programmatic API

For Python-based integration, see the annotated examples in `examples/`:

| File | What it demonstrates |
|------|----------------------|
| `examples/example_generation.py` | Generating topology data programmatically |
| `examples/example_import.py` | Importing a graph into Neo4j via the Python API |
| `examples/example_analysis.py` | Running structural and quality analysis |
| `examples/example_simulation.py` | Running failure and event simulations |
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
│  Graph      │───▶│  Structural │───▶│  Quality    │
│  Model      │    │  Analysis   │    │  Scoring    │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
      ┌──────────────────────────────────────┘
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Step 4     │    │  Step 5     │    │  Step 6     │
│  Failure    │───▶│  Validation │───▶│  Visuali-   │
│  Simulation │    │             │    │  zation     │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
                                      HTML Dashboard
```

| Step | What It Does | Key Output |
|------|-------------|------------|
| **1. [Graph Model](docs/graph-model.md)** | Converts topology JSON to a weighted directed graph G(V, E, w); derives DEPENDS_ON edges from pub-sub relationships | G_structural and G_analysis(l) |
| **2. [Structural Analysis](docs/structural-analysis.md)** | Computes PageRank, Reverse PageRank, betweenness centrality, articulation points, and clustering coefficients | Metric vector M(v) per component |
| **3. [Quality Scoring](docs/quality-scoring.md)** | Maps M(v) to RMAV dimensions using AHP-derived weights; classifies criticality via box-plot adaptive thresholds | Quality score Q(v) ∈ {LOW, MEDIUM, HIGH, CRITICAL} |
| **4. [Failure Simulation](docs/failure-simulation.md)** | Injects faults exhaustively and propagates cascades through G_structural | Ground-truth impact score I(v) per component |
| **5. [Validation](docs/validation.md)** | Computes Spearman ρ between Q(v) and I(v); evaluates classification accuracy (F1, precision, recall) | Statistical evidence of predictive validity |
| **6. [Visualization](docs/visualization.md)** | Renders interactive dashboards with network graphs, dependency matrices, and layer comparison views | `dashboard.html` (self-contained) |

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

## Project Structure

```
software-as-a-graph/
├── genieus/                    # Next.js 15 web dashboard
├── api.py                      # FastAPI backend entry point
├── docker-compose.yml          # Full-stack orchestration
│
├── bin/                        # CLI entry points (one per pipeline step)
│   ├── run.py                  #   Pipeline orchestrator (runs all steps)
│   ├── generate_graph.py       #   Synthetic topology generation
│   ├── import_graph.py         #   Neo4j import & dependency derivation
│   ├── analyze_graph.py        #   Structural analysis + quality scoring
│   ├── simulate_graph.py       #   Failure simulation
│   ├── validate_graph.py       #   Statistical validation
│   └── visualize_graph.py      #   Dashboard generation
│
├── src/                        # Python source (hexagonal architecture)
│   ├── core/                   #   Domain models (GraphData, ComponentData)
│   │                           #   Ports (IGraphRepository) & adapters (Neo4j)
│   ├── analysis/               #   Structural metrics + RMAV quality scoring
│   ├── simulation/             #   Event-based failure propagation
│   ├── validation/             #   Spearman, F1, classification metrics
│   ├── visualization/          #   Dashboard & chart generation
│   ├── generation/             #   Synthetic graph generation
│   └── cli/                    #   Shared CLI utilities
│
├── examples/                   # Annotated Python usage examples
├── config/                     # Scale preset YAML configs
├── docs/                       # Per-step methodology documentation
│   ├── graph-model.md
│   ├── structural-analysis.md
│   ├── quality-scoring.md
│   ├── failure-simulation.md
│   ├── validation.md
│   └── visualization.md
└── tests/                      # Pytest unit & integration tests
```

---

## Research Context

This framework is the software artifact for the PhD dissertation **"Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems"** at Istanbul Technical University, Department of Computer Engineering.

The underlying methodology was peer-reviewed and published at:

> **IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE 2025)**
> *A Graph-Based Dependency Analysis Method for Identifying Critical Components in Distributed Publish-Subscribe Systems*

The primary research contribution is the demonstration that **topological graph metrics can reliably predict real-world failure impact without runtime instrumentation**, validated empirically across four application domains and multiple system scales.

---

## Citation

If you use this framework or build on the methodology in academic work, please cite:

```bibtex
@INPROCEEDINGS{11315354,
  author    = {Yigit, Ibrahim Onuralp and Buzluca, Feza},
  booktitle = {2025 IEEE International Conference on Recent Advances in
               Systems Science and Engineering (RASSE)},
  title     = {A Graph-Based Dependency Analysis Method for Identifying
               Critical Components in Distributed Publish-Subscribe Systems},
  year      = {2025},
  pages     = {1--9},
  doi       = {10.1109/RASSE64831.2025.11315354}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.