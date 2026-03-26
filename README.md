# Software-as-a-Graph

**Predict which components in a distributed system will cause the most damage when they fail — using only the system's architecture.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB)](https://react.dev/)
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
8. [Local Development](#local-development)
9. [How It Works — The 6-Step Pipeline](#how-it-works--the-6-step-pipeline)
10. [RMAV Prediction](#rmav-prediction)
11. [Anti-Pattern Detection](#anti-pattern-detection)
12. [Python SDK (`saag`)](#python-sdk-saag)
13. [Project Structure](#project-structure)
14. [Research Context](#research-context)
15. [Citation](#citation)
16. [License](#license)

---

## The Problem

In distributed publish-subscribe systems (ROS 2, Kafka, MQTT, etc.), some components are far more critical than others. When they fail, failures cascade through the system. Identifying these weak points traditionally requires either expensive runtime monitoring or waiting for production incidents.

This reactive approach has two fundamental problems: runtime monitoring adds overhead to production systems, and by the time a critical failure is discovered, the damage is already done.

## Our Approach

We treat your system architecture as a graph and apply topological analysis to **predict** critical components **before deployment**. The key insight is:

> **A component's position in the dependency graph reliably predicts its real-world failure impact.**

Applications, brokers, topics, and infrastructure nodes are modelled as vertices. Their publish-subscribe relationships become weighted directed edges, with weights derived from QoS settings (reliability, durability, priority, message size). Graph-theoretic metrics — Reverse PageRank, betweenness centrality, articulation point detection, bridge ratio, and more — are combined into an **RMAV composite quality score** (Reliability, Maintainability, Availability, Vulnerability) using AHP-derived weights.

An optional **Graph Neural Network (GNN)** layer (Graph Attention Network) refines predictions beyond rule-based scoring by learning from simulation results and blending them with the RMAV scores via a per-dimension ensemble. An **anti-pattern detector** then audits the quality results to flag 10 categories of architectural smells — enabling automated CI/CD policy gates.

Ground-truth impact scores are produced by four parallel failure simulators (cascade, change-propagation, connectivity-loss, and compromise-propagation), each aligned to one RMAV dimension.

This shifts the analysis paradigm from reactive runtime monitoring to **proactive pre-deployment prediction**.

## Empirical Validation

The methodology has been empirically validated across 8 scenario datasets spanning multiple system scales and domains:

| Metric | Target | Achieved |
|--------|--------|----------|
| Spearman correlation ρ(Q*, I*) | ≥ 0.85 | **> 0.87** |
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

The **Genieus** dashboard (Next.js 16, React 19, TypeScript, Tailwind CSS 4) provides an interactive frontend for the full analysis pipeline:

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

# Step 3 — Prediction: RMAV quality scoring + anti-pattern detection
python bin/analyze_graph.py --layer system --predict

# Step 3.5 — (Optional) GNN training: train the Graph Attention Network
python bin/train_graph.py --layer system

# Step 3.5 — (Optional) GNN inference: predict criticality using a trained model
python bin/predict_graph.py --layer system

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
# Standalone anti-pattern detection (CI/CD gating)
python bin/detect_antipatterns.py --layer system --output output/antipatterns.json

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

### Programmatic API (Examples)

For Python-based integration, see the annotated examples in `examples/`. Run them from the project root:

| Order | File | What it demonstrates | Needs Neo4j |
|:---:|------|----------------------|:-----------:|
| 0 | `examples/example_introduction.py` | Core concepts: why topology predicts risk (no database) | No |
| 1 | `examples/example_generation.py` | Generating a synthetic topology programmatically | No |
| 2 | `examples/example_import.py` | Importing a graph into Neo4j via the Python API | Yes |
| 3 | `examples/example_analysis.py` | Structural metrics, RMAV scoring & anti-pattern detection | Yes |
| 4 | `examples/example_simulation.py` | Exhaustive failure simulations (ground-truth I(v)) | Yes |
| 5 | `examples/example_validation.py` | Validating predictions against ground truth | Yes |
| 6 | `examples/example_prediction.py` | GNN-based criticality refinement | Yes |
| 7 | `examples/example_visualization.py` | Generating self-contained HTML dashboards | Yes |
| 8 | `examples/example_end_to_end.py` | Full 6-step pipeline in a single script | Yes |
| 9 | `examples/example_antipatterns.py` | Anti-pattern gating for CI/CD pipelines | Yes |
| 10 | `examples/example_compare.py` | Comparing two architectural designs side-by-side | Yes |

See [`examples/README.md`](examples/README.md) for prerequisites and guidance on interpreting outputs.

---

## Local Development

Use this guide when you want to run the backend API and frontend separately with hot-reload, without Docker.

### Backend (FastAPI)

Start Neo4j:
```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:2026.02.2
```

```bash
python3.11 -m venv backend/env
source backend/env/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn api.main:app --reload --port 8000
```

The API will be available at http://localhost:8000 and the interactive docs at http://localhost:8000/docs.

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

---

## How It Works — The 6-Step Pipeline

```
Architecture JSON
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
│  Step 1     │    │  Step 2     │    │  Step 3             │
│  Modeling   │───▶│  Analysis   │───▶│  Prediction         │
│  (Import)   │    │  (Metrics)  │    │  (RMAV + GNN +      │
│             │    │             │    │   Anti-Patterns)     │
└─────────────┘    └─────────────┘    └─────────────────────┘
                                                    │
       ┌──────────────────────────────────────┘
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Step 4     │    │  Step 5     │    │  Step 6     │
│  Simulation │───▶│  Validation │───▶│  Visuali-   │
│  (I(v) GT)  │    │  (ρ, F1)    │    │  zation     │
└─────────────┘    └─────────────┘    └─────────────┘
                                                │
                                                ▼
                                         HTML Dashboard
```

| Step | What It Does | Key Output |
|------|-------------|------------|
| **1. [Modeling](docs/graph-model.md)** | Converts topology JSON to a weighted directed graph G(V, E, w); derives DEPENDS_ON edges from pub-sub relationships | G_structural and G_analysis(l) |
| **2. [Analysis](docs/structural-analysis.md)** | Computes Reverse PageRank, betweenness, closeness, eigenvector centralities, bridge ratio, articulation points, clustering | Metric vector M(v) per component |
| **3. [Prediction](docs/prediction.md)** | Maps M(v) to RMAV dimensions using AHP-derived weights; optional GNN refinement; classifies criticality via box-plot adaptive thresholds; detects anti-patterns | Q*(v) composite score, Q(v) ∈ {MINIMAL, LOW, MEDIUM, HIGH, CRITICAL}, structural smell report |
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

### Reliability — R(v)

```
R(v) = 0.45·RPR + 0.30·DG_in + 0.25·CDPot_enh
```

| Term | Description |
|------|-------------|
| **RPR** | Reverse PageRank — fault propagation reach |
| **DG_in** | Normalised in-degree — direct dependent count |
| **CDPot_enh** | Enhanced cascade depth potential = CDPot_base × (1 + MPCI) |

### Maintainability — M(v)

```
M(v) = 0.35·BT + 0.30·w_out + 0.15·CQP + 0.12·CouplingRisk + 0.08·(1 − CC)
```

| Term | Description |
|------|-------------|
| **BT** | Betweenness centrality — structural bottleneck position |
| **w_out** | QoS-weighted efferent coupling (outgoing dependency weight) |
| **CQP** | Code Quality Penalty = 0.40·complexity_norm + 0.35·instability_code + 0.25·lcom_norm (zero when absent — backward-compatible) |
| **CouplingRisk** | `1 − |2·Instability − 1|` — maximised at 0.5 for deeply embedded components |
| **(1−CC)** | Inverse clustering coefficient (reduced weight) |

### Availability — A(v)

```
A(v) = 0.35·AP_c_directed + 0.25·QSPOF + 0.25·BR + 0.10·CDI + 0.05·w(v)
```

| Term | Description |
|------|-------------|
| **AP_c_directed** | `max(AP_c_out, AP_c_in)` — directional articulation point score |
| **QSPOF** | `AP_c_directed × w(v)` — QoS-scaled SPOF severity |
| **BR** | Bridge ratio — fraction of incident edges that are bridges |
| **CDI** | Connectivity Degradation Index — normalised increase in path length on removal |
| **w(v)** | Pure operational priority weight |

### Vulnerability — V(v)

```
V(v) = 0.40·REV + 0.35·RCL + 0.25·QADS
```

| Term | Description |
|------|-------------|
| **REV** | Reverse eigenvector centrality on G^T — strategic attack reach |
| **RCL** | Reverse closeness centrality on G^T — adversarial propagation speed |
| **QADS** | QoS-weighted attack-dependent surface (inbound dependency weight) |

### Overall Quality Score — Q*(v)

```
Q*(v) = 0.24·R(v) + 0.17·M(v) + 0.43·A(v) + 0.16·V(v)
```

Dimension weights are derived via AHP across all four RMAV axes. Availability is dominant because connectivity disruption has the highest structural alignment with composite failure impact across all validated domains.

### Criticality Classification (Adaptive Box-Plot)

Criticality is classified into five levels using adaptive box-plot thresholds derived from the actual score distribution of the system under analysis:

| Level | Threshold |
|-------|-----------|
| **CRITICAL** | score > Q3 + 0.75 × IQR |
| **HIGH** | score > Q3 |
| **MEDIUM** | score > Median |
| **LOW** | score > Q1 |
| **MINIMAL** | score ≤ Q1 |

> For graphs with fewer than 12 components, a fixed-percentile fallback is used (top 10% → CRITICAL, etc.).

### GNN Ensemble Refinement (Step 3.5)

An optional Graph Attention Network (GAT) layer refines predictions beyond rule-based scoring. After training (or loading a pre-trained model), predictions are blended:

```
Q_ensemble(v) = α · Q_GNN(v) + (1 − α) · Q_RMAV(v)
```

where α is a per-dimension blending coefficient learned during training (typically 0.6–0.8). Training uses simulation ground truth (I(v)) as supervision.

```bash
# Train or retrain the GNN on the current dataset
python bin/train_graph.py --layer system
```

---

## Anti-Pattern Detection

After RMAV scoring, the `AntiPatternDetector` audits results and flags architectural smells across 10 categories. It can be run standalone via `bin/detect_antipatterns.py` or consumed programmatically within CI/CD pipelines (see `examples/example_antipatterns.py`).

| Anti-Pattern | Trigger Condition | Severity |
|---|---|---|
| **SPOF** | Component is an articulation point | CRITICAL |
| **FAILURE_HUB** | R(v) ≥ CRITICAL threshold | CRITICAL |
| **GOD_COMPONENT** | M(v) ≥ CRITICAL and betweenness > 0.3 | CRITICAL |
| **TARGET** | V(v) ≥ CRITICAL threshold | CRITICAL |
| **SYSTEMIC_RISK** | CRITICAL components > 20% of system | CRITICAL |
| **BRIDGE_EDGE** | Edge is a graph bridge | HIGH |
| **EXPOSURE** | V(v) == HIGH and closeness > 0.6 | HIGH |
| **CYCLE** | Strongly Connected Component ≥ 2 nodes | HIGH |
| **HUB_AND_SPOKE** | Clustering < 0.1 and degree > 3 | MEDIUM |
| **CHAIN** | Weakly connected sequence ≥ 4 nodes | MEDIUM |

---

## Python SDK (`saag`)

The `saag` package provides a fluent Python API for building analysis pipelines without touching the CLI. It is the recommended interface for notebook-based and programmatic workflows.

```python
import saag

# Full pipeline in 5 lines
result = (
    saag.Pipeline.from_json("input/system.json", clear=True)
        .analyze(layer="app")
        .simulate(layer="app", mode="exhaustive")
        .validate()
        .visualize(output="output/report.html")
        .run()
)

print(f"Spearman ρ = {result.validation.overall.spearman:.3f}")
print(f"F1-Score   = {result.validation.overall.f1:.3f}")
```

**Key classes:**

| Class | Purpose |
|-------|---------|
| `saag.Pipeline` | Fluent builder that chains and executes pipeline stages |
| `saag.Client` | Low-level service wrapper (analysis, simulation, validation, visualization) |
| `saag.AnalysisResult` | Holds structural metrics and RMAV quality scores |
| `saag.PredictionResult` | Holds criticality levels and anti-pattern report |
| `saag.ValidationResult` | Holds Spearman ρ, F1-score, and per-RMAV correlations |

**Independence guarantee:** `Q(v)` (RMAV prediction, Step 3) is computed using only graph topology. `I(v)` (simulation impact, Step 4) is computed using only cascade propagation rules and never reads `Q(v)`. Measuring their agreement in Step 5 is therefore a genuine empirical test — not a consistency check.

---

## Project Structure

```
.
├── bin/                        # CLI pipeline scripts
│   ├── run.py                  #   Master pipeline orchestrator (--all flag)
│   ├── generate_graph.py       #   Synthetic topology generator
│   ├── import_graph.py         #   Step 1: Modeling — Neo4j import
│   ├── analyze_graph.py        #   Step 2: Analysis
│   ├── train_graph.py          #   Step 3: GNN training (optional)
│   ├── predict_graph.py        #   Step 3: GNN inference on a new graph
│   ├── detect_antipatterns.py  #   Standalone anti-pattern / CI gate
│   ├── simulate_graph.py       #   Step 4: Simulation
│   ├── validate_graph.py       #   Step 5: Validation
│   ├── visualize_graph.py      #   Step 6: Visualization
│   ├── export_graph.py         #   Export graph data from Neo4j
│   ├── benchmark.py            #   Benchmark across scale presets
│   ├── run_scenarios.sh        #   Full pipeline across 8 domain scenarios
│   └── ground_threshold.py     #   Empirical SPOF threshold grounding
│
├── saag/                       # Public Python SDK (fluent pipeline API)
│   ├── pipeline.py             #   saag.Pipeline — fluent builder
│   ├── client.py               #   saag.Client — service façade
│   └── models.py               #   Result & data model types
│
├── backend/                    # Python backend (hexagonal architecture)
│   ├── api/                    #   FastAPI application & routers
│   │   ├── presenters/         #     Response formatting & API translation
│   │   ├── routers/            #     REST endpoints (thin layer)
│   │   └── dependencies.py     #     Service & Repository injection
│   ├── src/                    #   Domain source code
│   │   ├── core/               #     Domain models, ports, Neo4j & memory repos
│   │   ├── analysis/           #     Structural metrics + anti-pattern detection
│   │   ├── prediction/         #     RMAV quality scoring + GNN predictor
│   │   ├── simulation/         #     Four parallel failure/event simulators
│   │   ├── validation/         #     Per-dimension statistical validation
│   │   ├── visualization/      #     Dashboard & chart generation
│   │   ├── generation/         #     Synthetic graph generation
│   │   ├── benchmark/          #     Benchmarking services
│   │   └── explanation/        #     Human-readable CLI & report formatting
│   ├── tests/                  #   Pytest unit & integration tests
│   └── requirements.txt        #   Python dependencies
│
├── examples/                   # Annotated Python usage examples (see examples/README.md)
├── input/                      # Topology JSON & YAML scenario configs (8 scenarios)
├── output/                     # Pipeline output artefacts (dashboards, reports)
├── results/                    # Validation results from previous runs
├── benchmarks/                 # Benchmark data
└── docs/                       # Per-step methodology documentation
    ├── graph-model.md          #   Step 1: Modeling
    ├── structural-analysis.md  #   Step 2: Analysis
    ├── prediction.md           #   Step 3: Prediction
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