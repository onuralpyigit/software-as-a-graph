# Software-as-a-Graph (SaG)

> **Predict which components in a distributed system will cause the most damage when they fail — using only its architecture.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB)](https://react.dev/)
[![Docker](https://img.shields.io/badge/docker-compose-blue)](https://www.docker.com/)
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-green.svg)](https://neo4j.com/)
[![IEEE RASSE 2025](https://img.shields.io/badge/IEEE-RASSE%202025-orange.svg)](#citation)

---

## Table of Contents

1. [The Problem](#the-problem)
2. [The Methodology](#the-methodology)
   - [Core Insight](#core-insight)
   - [Graph Construction (Import)](#graph-construction-import)
   - [Analysis](#analyze-stage-step-2)
   - [Prediction](#predict-stage-step-3)
   - [Failure Simulation](#failure-simulation-step-4)
   - [Statistical Validation](#statistical-validation-step-5)
3. [Empirical Results](#empirical-results)
4. [Supported Platforms](#supported-platforms)
5. [Quick Start (Docker)](#quick-start-docker)
6. [Web Interface — Genieus](#web-interface--genieus)
7. [Development Setup (CLI)](#development-setup-cli)
8. [Local Development](#local-development)
9. [The Pipeline](#the-pipeline)
10. [RMAV Formulas Reference](#rmav-formulas-reference)
11. [Anti-Pattern Detection](#anti-pattern-detection)
12. [Python SDK (`saag`)](#python-sdk-saag)
13. [Project Structure](#project-structure)
14. [Research Context](#research-context)
15. [Citation](#citation)
16. [License](#license)

---

## The Problem

In distributed publish-subscribe systems (ROS 2, Apache Kafka, MQTT, and others), some components are structurally far more critical than others. When they fail, failures cascade through the system. Traditional approaches to identifying these weak points require either expensive runtime monitoring or waiting for production incidents.

This reactive posture has two fundamental problems:

- **Runtime monitoring adds overhead** to production systems that are often latency-sensitive or safety-critical.
- **By the time a critical failure is discovered in production**, the damage — data loss, service disruption, financial loss, safety risk — has already occurred.

---

## The Methodology

### Core Insight

> **A component's position in the dependency graph reliably predicts its real-world failure impact — without any runtime data.**

Software-as-a-Graph (SaG) operationalises this insight into a six-step analytical pipeline. The fundamental claim is that **topological structure alone** — how components are connected, what they depend on, and how strongly — encodes enough information to rank components by their potential failure impact with high statistical fidelity (Spearman ρ > 0.87, F1 > 0.90).

---

### Graph Construction (Import)

The first step converts a system architecture JSON into a formal weighted directed graph G = (V, E, τ_V, τ_E, w):

**Vertex types (V):** Applications, Brokers, Topics, Infrastructure Nodes, Libraries.

**Structural edges (E_structural):** Six edge types imported directly from the topology description:

| Edge Type | Direction | Semantics |
|-----------|-----------|-----------|
| `PUBLISHES_TO` | App/Lib → Topic | Component sends messages |
| `SUBSCRIBES_TO` | App/Lib → Topic | Component receives messages |
| `ROUTES` | Broker → Topic | Broker routes this topic |
| `RUNS_ON` | App/Broker → Node | Component is hosted here |
| `CONNECTS_TO` | Node → Node | Network reachability |
| `USES` | App/Lib → Library | Code-level shared dependency |

**Derived dependency edges (E_dependency):** The critical transformation step. Six rules infer *logical* DEPENDS_ON edges (direction: *dependent → dependency*) from the structural graph, making hidden failure paths explicit:

| Rule | Dependency Type | Source Pattern | Interpretation |
|------|----------------|----------------|----------------|
| **1** | `app_to_app` | App_sub → App_pub via shared Topic | Subscriber depends on publisher; losing the publisher starves the subscriber |
| **2** | `app_to_broker` | App → Broker routing its topics | App depends on its broker; broker failure silences all its topics |
| **3** | `node_to_node` | Lifted from Rule 1 — host-level | Two hosts inherit their apps' pub-sub dependency |
| **4** | `node_to_broker` | Lifted from Rule 2 — host-level | A host inherits its apps' broker dependency |
| **5** | `app_to_lib` | App → Library (USES edge) | Shared-library failure is a *simultaneous blast*, not a cascade — all consumers fail in one event |
| **6** | `broker_to_broker` | Bidirectional between co-located brokers | Two brokers sharing a physical node share the same hardware fate |

**Edge weights:** Each DEPENDS_ON edge carries a `weight ∈ [0, 1]` derived from the QoS properties of the mediating topic (Reliability × 0.30, Durability × 0.40, Transport Priority × 0.30, modulated by message size). A RELIABLE/PERSISTENT/URGENT topic produces w ≈ 0.85; a BEST_EFFORT/VOLATILE/LOW topic produces w ≈ 0.01. High weight = high-stakes coupling.

**Layer projections:** The graph is filtered into four analytical views:

| Layer | CLI flag | Vertices | Edge types |
|-------|----------|----------|------------|
| Application | `app` | App, Library | `app_to_app`, `app_to_lib` |
| Infrastructure | `infra` | Node | `node_to_node` |
| Middleware | `mw` | App, Broker, Node | `app_to_broker`, `node_to_broker`, `broker_to_broker` |
| System | `system` | All five types | All six types |

---

### Analysis

Step 2 is **deterministic and interpretable**: given the same graph, it always produces the same output. It has two sub-phases that run together as a single stage.

**Sub-phase 2a — Structural metrics.** Computes a structural metric vector **M(v)** for every component in the layer-projected DEPENDS_ON graph. Thirteen metrics are computed across four theoretical families:

| Metric | Symbol | Theoretical family | RMAV role |
|--------|--------|--------------------|-----------|
| Reverse PageRank (on G^T) | RPR | Random walk | R(v) — cascade reach |
| In-Degree normalized | DG_in | Local degree | R(v) — immediate blast radius |
| Multi-Path Coupling Index | MPCI | Structural coupling | R(v) — amplifier via CDPot_enh |
| Fan-Out Criticality | FOC | Topic-specific | R(v) — Topic broadcast risk |
| Betweenness Centrality | BT | Path-based | M(v) — structural bottleneck |
| QoS-Weighted Out-Degree | w_out | QoS-weighted degree | M(v) — efferent coupling |
| Clustering Coefficient | CC | Local topology | M(v) — local redundancy (as 1−CC) |
| Directed AP Score | AP_c_directed | Resilience | A(v) — directed SPOF severity |
| Bridge Ratio | BR | Resilience | A(v) — non-redundant edge fraction |
| Connectivity Degradation Index | CDI | Resilience | A(v) — path elongation on removal |
| Reverse Eigenvector Centrality | REV | Random walk | V(v) — downstream value of dependents |
| Reverse Closeness Centrality | RCL | Path-based | V(v) — adversarial propagation speed |
| QoS-Weighted In-Degree | w_in | QoS-weighted degree | V(v) — attack-surface weight |

**Sub-phase 2b — RMAV scoring.** Maps M(v) to four quality dimensions using AHP-derived weights. This is the rule-based model: a closed-form function of topology and metadata with no learned parameters. Anti-pattern detection (SPOF, FAILURE_HUB, GOD_COMPONENT, etc.) also runs here, on the RMAV scores.

**Why thirteen metrics?** No single metric captures all structural risk dimensions. A component can be a wide cascade propagator (high RPR, low AP_c) but not a SPOF — or be the sole connector between two clusters (high AP_c) without having many direct dependents (low DG_in). The thirteen metrics are deliberately designed to be **orthogonal**: each feeds exactly one RMAV dimension, with no metric shared across dimensions.

**MPCI** is a novel metric introduced in this work. It uses the `path_count` attribute on DEPENDS_ON edges (the number of distinct shared topics mediating a dependency) to quantify *multi-channel coupling intensity*. When the same dependent pair shares three topics, each is an independent failure vector — the cascade depth is amplified accordingly.

---

### Prediction

Step 3 is **inductive**: a HeteroGAT trained on simulation ground truth learns patterns that the AHP-weighted composite cannot encode — nonlinear interactions, multi-hop motifs, cross-type embedding effects. It consumes the `StructuralAnalysisResult` produced by Step 2 (no repository access) and emits GNN-derived criticality ranks blended with RMAV via a learnable ensemble coefficient α.

This stage is **optional**. The Analyze stage alone (Step 2) achieves Spearman ρ > 0.87 and F1 > 0.90. Step 3 refines those predictions after simulation-derived labels become available.

#### RMAV Formulas (produced by Step 2 Analyze)

The RMAV formulas below are part of the Analyze stage. They use AHP-derived weights (Analytic Hierarchy Process, consistency ratio CR < 0.02 for all matrices):

#### Reliability — R(v): *How broadly does failure propagate?*

```
R(v) = 0.45·RPR + 0.30·DG_in + 0.25·CDPot_enh

CDPot_enh(v) = min( CDPot_base(v) × (1 + MPCI(v)), 1.0 )
CDPot_base(v) = ((RPR + DG_in) / 2) × (1 − min(DG_out_raw / DG_in_raw, 1))
```

For Topic nodes (which have no DEPENDS_ON in-degree), a topic-specific formula is used:
```
R_topic(v) = 0.50·FOC + 0.50·CDPot_topic
```

#### Maintainability — M(v): *How hard is this to change safely?*

```
M(v) = 0.35·BT + 0.30·w_out + 0.15·CQP + 0.12·CouplingRisk_enh + 0.08·(1 − CC)

CQP(v)              = 0.40·complexity_norm + 0.35·instability_code + 0.25·lcom_norm
CouplingRisk_enh(v) = min(1.0, CouplingRisk_base × (1 + 0.10 × path_complexity))
```

CQP is zero for non-Application/Library node types (backward-compatible graceful degradation). Two distinct instability signals are intentional: `instability_code` captures static-code-level fragility; `CouplingRisk_enh` captures runtime-topology-level fragility. They frequently diverge.

#### Availability — A(v): *Is this a structural SPOF?*

```
A(v) = 0.35·AP_c_directed + 0.25·QSPOF + 0.25·BR + 0.10·CDI + 0.05·w(v)

QSPOF(v) = AP_c_directed(v) × w(v)   (QoS-weighted SPOF severity)
```

AP_c_directed uses a *directed* articulation point score computed on the DEPENDS_ON graph — it correctly captures directed cut vertices rather than the undirected AP, which can both over-report and under-report in pub-sub systems.

#### Vulnerability — V(v): *How attractive a target for adversarial attack?*

```
V(v) = 0.40·REV + 0.35·RCL + 0.25·w_in
```

REV and RCL are computed on G^T (the transposed graph) so they measure how far compromise propagates *outward* through v's dependents, not how far v itself can reach its dependencies.

#### Composite Score — Q*(v)

```
Q*(v) = 0.43·A(v) + 0.24·R(v) + 0.17·M(v) + 0.16·V(v)
```

AHP cross-dimension weights (CR ≈ 0.02): Availability (0.43) is dominant because structural SPOF failure — an articulation point with BR = 1.0 — partitions the graph with certainty, while cascade reach and coupling risk are probabilistic.

**Weight shrinkage:** AHP weights are blended toward a uniform prior with shrinkage factor λ = 0.7:
```
w_final(d) = 0.70 × w_AHP(d) + 0.30 × 0.25
```
λ = 0.70 was selected empirically via a sensitivity sweep; Spearman ρ plateaus in the λ ∈ [0.65, 0.75] region.

#### Criticality Classification

Five-level adaptive box-plot thresholding based on the system's own score distribution:

| Level | Threshold |
|-------|-----------|
| **CRITICAL** | score > Q3 + 1.5 × IQR |
| **HIGH** | Q3 < score ≤ upper fence |
| **MEDIUM** | Median < score ≤ Q3 |
| **LOW** | Q1 < score ≤ Median |
| **MINIMAL** | score ≤ Q1 |

> For graphs with fewer than 12 components, a fixed-percentile fallback is used (top 10% → CRITICAL).

Classification is applied **independently per RMAV dimension and for Q*(v)**. A component can be CRITICAL on Availability (structural SPOF) but MINIMAL on Vulnerability — this decomposition is exactly the information needed to direct targeted remediation.

#### GNN Ensemble (Predict Stage — Step 3)

The Predict stage refines the Analyze-stage scores using a Graph Attention Network trained on simulation ground truth I(v). Predictions are blended via a per-dimension learned ensemble:

```
Q_ensemble(v) = α · Q_GNN(v) + (1−α) · Q_RMAV(v)    α ∈ ℝ⁵, learned per-dimension
```

The HeteroGAT architecture processes 23-dimensional node feature vectors (18 topological + 5 code-quality for App/Library types), 8-dimensional edge features, and uses 3 message-passing layers with 4 attention heads. Multi-task prediction heads produce per-RMAV-dimension outputs; the composite head also receives the four dimension predictions as input.

---

### Failure Simulation (Step 4)

Step 4 produces **independent** ground-truth impact scores I(v) using physically grounded failure simulators. This stage also generates the labelled training data consumed by the Predict stage (Step 3) when running GNN training. "Independent" is critical: Step 5 measures the agreement between Q*(v) from Analyze and I(v) from Simulate.

Four simulators run in parallel, each aligned to one RMAV dimension:

| Simulator | Ground Truth | Formula |
|-----------|-------------|---------|
| `FaultInjector` (BFS cascade) | I*(v) overall | 0.35·reachability_loss + 0.25·fragmentation + 0.25·throughput_loss + 0.15·flow_disruption |
| `FaultInjector` (cascade trace) | IR(v) reliability | 0.45·CascadeReach + 0.35·WeightedCascadeImpact + 0.20·NormalizedCascadeDepth |
| `ChangePropagationSimulator` | IM(v) maintainability | 0.45·ChangeReach + 0.35·WeightedChangeImpact + 0.20·NormalizedChangeDepth |
| Connectivity analysis | IA(v) availability | 0.50·WeightedReachabilityLoss + 0.35·WeightedFragmentation + 0.15·PathBreakingThroughputLoss |
| `CompromisePropagationSimulator` | IV(v) vulnerability | 0.40·AttackReach + 0.35·WeightedAttackImpact + 0.25·HighValueContamination |

The `FaultInjector` runs exhaustive BFS for every candidate node: Wave 0 directly orphans topics; subsequent waves propagate the cascade through subscriber → publisher chains until fixpoint. Multi-seed averaging (default: seeds 42, 123, 456, 789, 2024) dampens stochastic variance.

A complementary `MessageFlowSimulator` (SimPy discrete-event) models real-time delivery rates, QoS enforcement (deadline, lifespan, reliability policy), and per-subscriber queue dynamics.

**Methodological independence guarantee:** Q*(v) is computed using only G_analysis(l) (DEPENDS_ON graph topology). I(v) is computed using only G_structural (raw pub-sub edges, no DEPENDS_ON). The two pipelines have no shared data path. Measuring their agreement in Step 5 is a genuine empirical test — not a consistency check.

---

### Statistical Validation (Step 5)

Step 5 closes the methodological loop. It measures the statistical agreement between Q*(v) and I*(v) using a compound test battery:

| Metric | Role | Target |
|--------|------|--------|
| Spearman ρ | Primary gate: global rank ordering | ≥ 0.80 (medium/dense topologies) |
| Kendall τ | Conservative cross-check; |ρ−τ| > 0.15 flags outlier-driven agreement | — |
| Bootstrap 95% CI | Non-parametric uncertainty (B=2,000 resamples) | CI above gate threshold |
| F1@K | Top-K critical component identification quality | ≥ 0.70 |
| SPOF-F1 | Articulation-point detection quality | ≥ 0.65 |
| Predictive Gain (PG) | ρ(Q*,I*) − ρ(degree_centrality, I*) | ≥ 0.03 |
| False Top Rate (FTR) | Fraction of predicted-critical that are actually safe | ≤ 0.25 |
| Wilcoxon signed-rank | Q*(v) statistically closer to I*(v) than degree baseline | p < 0.05 |

Gates are **topology-adaptive**: sparse 12-node systems face softer thresholds than dense hub-spoke architectures with 100+ components.

---

## Empirical Results

Validated across 8 domain scenario datasets:

| Metric | Target | Achieved |
|--------|--------|----------|
| Spearman ρ(Q*, I*) overall | ≥ 0.85 | **> 0.87** |
| Spearman ρ(Q*, I*) at large scale (150–300+ nodes) | — | **0.943** |
| Overall F1-score | ≥ 0.90 | **> 0.90** |
| Predictive Gain (PG) vs. degree baseline | > 0.03 | **> 0.03** |
| Best prediction layer | — | Application layer outperforms infrastructure layer |
| Scale effect | — | Prediction accuracy improves with system size |

Validation domains include ROS 2 autonomous vehicles, IoT smart cities, financial trading platforms, healthcare systems, and Air Traffic Management (ATM) systems.

---

## Supported Platforms

The graph model maps naturally to any pub-sub middleware:

| Graph Concept | ROS 2 / DDS | Apache Kafka | MQTT |
|---------------|-------------|--------------|------|
| Application | ROS Node | Producer / Consumer | MQTT Client |
| Topic | ROS Topic | Kafka Topic | MQTT Topic |
| Broker | DDS Participant | Kafka Broker | MQTT Broker |
| Infrastructure Node | Host / Container | Broker Host | Broker Server |
| Library | ROS package dep | Maven artifact | Paho client lib |

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

`--network host` allows the container to connect to Neo4j running on the host (port 7687).

---

## Web Interface — Genieus

The **Genieus** dashboard (Next.js 16, React 19, TypeScript, Tailwind CSS 4) provides an interactive frontend for the full analysis pipeline:

1. **Dashboard** — High-level KPIs, criticality distribution heatmap, and top critical component list.
2. **Graph Explorer** — Interactive 2D/3D force-directed graph. Filter by layer (app / infra / mw / system), search components, and inspect dependency details.
3. **Analysis** — Trigger structural analysis and RMAV quality scoring for a selected layer.
4. **Simulation** — Simulate component failures and visualise cascade propagation paths.
5. **Statistics** — Extras dashboard: QoS risk scatter, topic fanout, node communication load, criticality I/O, domain diversity, and more.
6. **Settings** — Configure the Neo4j connection (URI, credentials, database name).

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
# Step 1 — Import: generate synthetic topology & import into Neo4j
python bin/generate_graph.py --scale medium --output input/system.json
python bin/import_graph.py --input input/system.json --clear

# Step 2 — Analyze: structural metrics (13 topological indicators) + RMAV/Q scoring + anti-patterns
#   Deterministic: given the same graph, always produces the same Q(v).
python bin/analyze_graph.py --layer system --predict

# Step 3 — Predict (optional): inductive GNN forecasting beyond the closed-form RMAV
#   Requires Step 4 simulation results for training labels first.
python bin/train_graph.py --layer system                              # train GNN
python bin/predict_graph.py --layer system --gnn-model output/gnn_checkpoints/best_model

# Step 4 — Simulate: failure simulation (ground-truth I(v) + training labels for Step 3)
python bin/simulate_graph.py fault-inject --input input/system.json --export-json

# Step 5 — Validate: statistical validation (Spearman ρ, F1-score, per-RMAV metrics)
python bin/validate_graph.py report --input input/system.json --qos

# Step 6 — Visualize: interactive HTML dashboard (self-contained, no server needed)
python bin/visualize_graph.py --layer system --output output/dashboard.html --open
```

The `--layer` flag accepts `app`, `infra`, `mw`, or `system` (all layers combined). The `app` layer includes both Application and Library nodes — library blast-radius risk (a shared library used by N applications is a SPOF for all N) is visible at this layer.

Additional utility scripts:

```bash
# Standalone anti-pattern detection (CI/CD gating)
python bin/detect_antipatterns.py --layer system --output output/antipatterns.json

# Export graph data from Neo4j
python bin/export_graph.py --output output/graph_export.json

# Benchmark across all scale presets
python bin/benchmark.py

# Run the full pipeline across all 8 domain scenarios
bash bin/run_scenarios.sh

# Ground the SPOF threshold empirically across all 8 scenarios
python bin/ground_threshold.py

# Statistics dashboard (CLI): topology & communication pattern analytics
python bin/statistics_graph.py --layer system
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
| 3 | `examples/example_analysis.py` | Analyze stage: structural metrics, RMAV scoring & anti-pattern detection | Yes |
| 4 | `examples/example_simulation.py` | Simulate stage: exhaustive failure simulations (ground-truth I(v)) | Yes |
| 5 | `examples/example_validation.py` | Validate stage: comparing Analyze output against ground truth | Yes |
| 6 | `examples/example_prediction.py` | Predict stage: GNN-based inductive criticality refinement | Yes |
| 7 | `examples/example_visualization.py` | Visualize stage: generating self-contained HTML dashboards | Yes |
| 8 | `examples/example_end_to_end.py` | Full pipeline (Import → Analyze → Predict → Simulate → Validate → Visualize) | Yes |
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

## The Pipeline

```
Architecture JSON
      │
      ▼
┌─────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  Step 1     │    │  Step 2              │    │  Step 3 (optional)  │
│  Import     │───▶│  Analyze             │───▶│  Predict            │
│             │    │  (M(v) + RMAV/Q(v)  │    │  (GNN ensemble;     │
│             │    │   + Anti-Patterns)   │    │   inductive only)   │
└─────────────┘    └──────────────────────┘    └─────────────────────┘
                            │                             │
       ┌────────────────────┘                            │
       ▼                                                 ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐
│  Step 4     │    │  Step 5     │    │  Step 6                     │
│  Simulate   │───▶│  Validate   │───▶│  Visualize                  │
│  (I(v) GT)  │    │  (ρ, F1)    │    │                             │
└─────────────┘    └─────────────┘    └─────────────────────────────┘
                                                     │
                                                     ▼
                                              HTML Dashboard
```

| Step | What It Does | Key Output | Docs |
|------|-------------|------------|------|
| **1. Import** | Converts topology JSON to a weighted directed graph G(V, E, w); derives DEPENDS_ON edges via six rules; computes QoS-derived weights | G_structural and G_analysis(l) | [graph-model.md](docs/graph-model.md) |
| **2. Analyze** | Deterministic, closed-form. Computes 13 structural metrics M(v); maps them to RMAV dimension scores and Q*(v) via AHP-weighted formulas; detects anti-patterns. Given the same graph, always produces the same output. | M(v) metric vector, RMAV/Q*(v) scores, five-level classification, anti-pattern report | [structural-analysis.md](docs/structural-analysis.md) · [prediction.md](docs/prediction.md) |
| **3. Predict** | Inductive, optional. A HeteroGAT trained on simulation labels I(v) learns interactions the AHP composite cannot encode. Consumes Analyze output only — no repository access. | GNN criticality ranks, edge criticality, ensemble-blended Q_ens(v) | [prediction.md](docs/prediction.md) |
| **4. Simulate** | Runs four parallel simulators (cascade, change-propagation, connectivity-loss, compromise-propagation). Provides training labels for Step 3 and ground truth for Step 5. | Per-dimension ground-truth IR(v), IM(v), IA(v), IV(v) and composite I*(v) | [failure-simulation.md](docs/failure-simulation.md) |
| **5. Validate** | Computes Spearman ρ and Kendall τ between Q*(v) (from Analyze) or Q_ens(v) (from Predict) and I*(v); evaluates F1, PG, SPOF-F1, FTR, Bootstrap CI, Wilcoxon | Statistical evidence of predictive validity | [validation.md](docs/validation.md) |
| **6. Visualize** | Renders interactive dashboards with network graphs, dependency matrices, cascade heatmaps, and RMAV radar charts | `dashboard.html` (fully self-contained) | [visualization.md](docs/visualization.md) |

> **Note — Generate is not part of the SaG methodology.** The `--generate` CLI flag and `bin/generate_graph.py` script produce synthetic pub-sub topologies for evaluation and benchmarking purposes only. Real deployments start at Step 1 (Import) with an actual architecture description. Synthetic graphs are useful for reproducible experiments, scale sweeps, and CI regression tests, but they are not inputs the methodology assumes or requires. Reviewers asking "is this validated on real or synthetic data?" should note that the published Spearman and F1 results use eight domain-scenario topologies in `input/`, not generated data.

### Scale Presets

The synthetic generator supports five scale presets for rapid experimentation:

| Preset | Apps | Topics | Brokers | Nodes | Libs | Typical Use |
|--------|------|--------|---------|-------|------|-------------|
| `tiny` | 5 | 5 | 1 | 2 | 2 | Unit tests |
| `small` | 15 | 10 | 2 | 4 | 5 | Quick checks |
| `medium` | 50 | 30 | 3 | 8 | 10 | Development |
| `large` | 150 | 100 | 6 | 20 | 30 | Integration tests |
| `xlarge` | 500 | 300 | 10 | 50 | 100 | Performance benchmarks |

---

## RMAV Formulas Reference

Quality scores are computed per component v. AHP weights use a shrinkage factor λ = 0.7.

### Reliability — R(v)

```
R(v) = 0.45·RPR + 0.30·DG_in + 0.25·CDPot_enh

CDPot_enh(v) = min( CDPot_base(v) × (1 + MPCI(v)), 1.0 )
```

| Term | Description |
|------|-------------|
| **RPR** | Reverse PageRank — fault propagation reach on G^T |
| **DG_in** | Normalised in-degree — direct dependent count |
| **CDPot_enh** | Enhanced cascade depth potential; amplified by multi-path coupling |

### Maintainability — M(v)

```
M(v) = 0.35·BT + 0.30·w_out + 0.15·CQP + 0.12·CouplingRisk_enh + 0.08·(1 − CC)

CQP(v) = 0.40·complexity_norm + 0.35·instability_code + 0.25·lcom_norm  (App/Lib only; 0 otherwise)
```

| Term | Description |
|------|-------------|
| **BT** | Betweenness centrality — structural bottleneck position |
| **w_out** | QoS-weighted efferent coupling (outgoing dependency weight) |
| **CQP** | Code Quality Penalty: cyclomatic complexity, Martin instability, LCOM |
| **CouplingRisk_enh** | `1 − |2·Instability − 1|` modulated by path complexity; peaks at 0.5 (embedded on both sides) |
| **(1−CC)** | Inverse clustering coefficient — low local redundancy intensifies coupling uniqueness |

### Availability — A(v)

```
A(v) = 0.35·AP_c_directed + 0.25·QSPOF + 0.25·BR + 0.10·CDI + 0.05·w(v)

QSPOF(v) = AP_c_directed(v) × w(v)
```

| Term | Description |
|------|-------------|
| **AP_c_directed** | `max(AP_c_out, AP_c_in)` — directed articulation point score |
| **QSPOF** | QoS-scaled SPOF severity — doubly penalises high-priority SPOFs |
| **BR** | Bridge ratio — fraction of incident edges that are bridges |
| **CDI** | Connectivity Degradation Index — normalised path elongation on removal |
| **w(v)** | Pure operational priority weight from QoS derivation |

### Vulnerability — V(v)

```
V(v) = 0.40·REV + 0.35·RCL + 0.25·w_in
```

| Term | Description |
|------|-------------|
| **REV** | Reverse eigenvector centrality on G^T — strategic attack reach into high-value targets |
| **RCL** | Reverse closeness centrality on G^T — how quickly adversarial paths converge on v |
| **w_in** | QoS-weighted in-degree (QADS) — direct high-SLA attack surface |

### Overall Quality Score — Q*(v)

```
Q*(v) = 0.43·A(v) + 0.24·R(v) + 0.17·M(v) + 0.16·V(v)
```

Availability is dominant (0.43) because SPOF failure in a dependency graph is the most directly measurable, structurally certain failure mode in a pre-deployment topology.

### Criticality Classification (Adaptive Box-Plot)

| Level | Threshold |
|-------|-----------|
| **CRITICAL** | score > Q3 + 1.5 × IQR |
| **HIGH** | score > Q3 |
| **MEDIUM** | score > Median |
| **LOW** | score > Q1 |
| **MINIMAL** | score ≤ Q1 |

> For graphs with fewer than 12 components, a fixed-percentile fallback is used (top 10% → CRITICAL, etc.).

### RMAV Interpretation Patterns

| Pattern | R | M | A | V | Risk type | Recommended action |
|---------|:-:|:-:|:-:|:-:|-----------|-------------------|
| Full hub | H | H | H | H | Catastrophic | Redundancy + circuit breakers + hardening |
| Reliability hub | H | L | L | L | Wide cascade | Retry logic, back-pressure, graceful degradation |
| God Component | L | H | L | L | Change fragility | Reduce coupling; extract interface |
| SPOF | L | L | H | L | Availability loss | Redundant instance, active-passive failover |
| High-value target | L | L | L | H | Compromise propagation | Zero-trust boundaries, network isolation |
| Multi-path sink | H | M | M | L | Deep multi-channel cascade | Reduce shared-topic count between same pair |
| Leaf | L | L | L | L | None | Standard monitoring |

### GNN Ensemble (Predict Stage — Step 3)

```
Q_ensemble(v) = α · Q_GNN(v) + (1−α) · Q_RMAV(v)
```

α is a 5-dimensional per-RMAV-dimension learnable blending coefficient (α = sigmoid(logit), initialised at 0.5). The HeteroGAT model uses 3 layers, 4 attention heads, hidden dimension D = 64.

```bash
# Train or retrain the GNN on the current dataset (requires Step 4 simulation results)
python bin/train_graph.py --layer system

# Run GNN inference on a new graph (ensemble blends GNN with RMAV from Analyze stage)
python bin/predict_graph.py --layer system --mode ensemble
```

---

## Anti-Pattern Detection

After RMAV scoring, the `AntiPatternDetector` audits results and flags architectural smells across 10 categories. Run standalone via `bin/detect_antipatterns.py` or integrated in CI/CD pipelines (see `examples/example_antipatterns.py`).

| Anti-Pattern | Trigger Condition | Severity |
|---|---|---|
| **SPOF** | Component is a directed articulation point | CRITICAL |
| **FAILURE_HUB** | R(v) ≥ CRITICAL threshold | CRITICAL |
| **GOD_COMPONENT** | M(v) ≥ CRITICAL and betweenness > 0.3 | CRITICAL |
| **TARGET** | V(v) ≥ CRITICAL threshold | CRITICAL |
| **SYSTEMIC_RISK** | CRITICAL components > 20% of system | CRITICAL |
| **BRIDGE_EDGE** | Edge is a graph bridge | HIGH |
| **EXPOSURE** | V(v) == HIGH and closeness > 0.6 | HIGH |
| **CYCLE** | Strongly Connected Component ≥ 2 nodes | HIGH |
| **HUB_AND_SPOKE** | Clustering < 0.1 and degree > 3 | MEDIUM |
| **CHAIN** | Weakly connected sequence ≥ 4 nodes | MEDIUM |

Anti-pattern results carry CI-ready exit codes (0 = clean, 1 = smells detected), making them suitable as pre-merge quality gates.

---

## Python SDK (`saag`)

The `saag` package provides a fluent Python API for building analysis pipelines without touching the CLI. It is the recommended interface for notebook-based and programmatic workflows.

```python
import saag

# Analyze + Simulate + Validate in 5 lines
result = (
    saag.Pipeline.from_json("input/system.json", clear=True)
        .analyze(layer="app")          # deterministic: structural + RMAV + anti-patterns
        .simulate(layer="app", mode="exhaustive")
        .validate()
        .visualize(output="output/report.html")
        .run()
)

print(f"Spearman ρ = {result.validation.overall.spearman:.3f}")
print(f"F1-Score   = {result.validation.overall.f1:.3f}")

# Optionally add the inductive Predict stage after simulation labels are available
result2 = (
    saag.Pipeline.from_json("input/system.json")
        .analyze(layer="app")
        .predict(mode="ensemble")      # GNN; requires prior training run
        .simulate(layer="app")
        .validate()
        .run()
)
```

**Key classes:**

| Class | Purpose |
|-------|---------|
| `saag.Pipeline` | Fluent builder that chains and executes pipeline stages |
| `saag.Client` | Low-level service wrapper (analyze, predict, simulate, validate, visualize) |
| `saag.AnalysisResult` | Analyze stage output: M(v) structural metrics, RMAV/Q*(v) scores, anti-patterns |
| `saag.PredictionResult` | Predict stage output: GNN criticality ranks, ensemble-blended scores |
| `saag.ValidationResult` | Validate stage output: Spearman ρ, F1-score, and per-RMAV correlations |

**Independence guarantee:** `Q*(v)` (RMAV, from the Analyze stage) is computed using only graph topology. `I*(v)` (from the Simulate stage) is computed using only cascade propagation rules and never reads `Q*(v)`. Measuring their agreement in the Validate stage is therefore a genuine empirical test — not a consistency check.

---

## Project Structure

```
.
├── bin/                        # CLI pipeline scripts
│   ├── run.py                  #   Master pipeline orchestrator (--all flag)
│   ├── generate_graph.py       #   Synthetic topology generator
│   ├── import_graph.py         #   Step 1: Import — Neo4j import & dependency derivation
│   ├── analyze_graph.py        #   Step 2: Analyze — structural metrics + RMAV/Q scoring + anti-patterns
│   ├── train_graph.py          #   Step 3: Predict — GNN training (optional; requires Step 4 labels)
│   ├── predict_graph.py        #   Step 3: Predict — GNN inference on a new graph
│   ├── detect_antipatterns.py  #   Standalone anti-pattern / CI gate
│   ├── simulate_graph.py       #   Step 4: Simulate (fault-inject | message-flow | combined)
│   ├── validate_graph.py       #   Step 5: Validate (single | sweep | report | compare)
│   ├── visualize_graph.py      #   Step 6: Visualize
│   ├── statistics_graph.py     #   Statistics dashboard (topology & communication analytics)
│   ├── export_graph.py         #   Export graph data from Neo4j
│   ├── benchmark.py            #   Benchmark across scale presets
│   ├── run_scenarios.sh        #   Full pipeline across 8 domain scenarios
│   └── ground_threshold.py     #   Empirical SPOF threshold grounding
│
├── tools/                      # Standalone tooling (no Neo4j dependency)
│   └── generation/             #   Statistical pub-sub topology generator
│       ├── generator.py        #     Core generator (structural edges only)
│       ├── service.py          #     High-level GenerationService wrapper
│       ├── models.py           #     Scale presets & statistical config
│       └── datasets.py         #     Domain-specific naming & QoS mappings
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
│   │   │   ├── layers.py       #       Canonical LAYER_DEFINITIONS & DEPENDENCY_TO_LAYER
│   │   │   └── neo4j_repo.py   #       Graph import, 5-phase construction, 6 derivation rules
│   │   ├── analysis/           #     Structural metrics + anti-pattern detection
│   │   ├── prediction/         #     RMAV quality scoring + GNN service + ensemble blending
│   │   ├── simulation/         #     Four parallel failure/event simulators
│   │   ├── validation/         #     Per-dimension statistical validation
│   │   ├── visualization/      #     Dashboard & chart generation
│   │   ├── generation/         #     Synthetic graph generation
│   │   └── benchmark/          #     Benchmarking services
│   ├── tests/                  #   Pytest unit & integration tests
│   └── requirements.txt        #   Python dependencies
│
├── examples/                   # Annotated Python usage examples (see examples/README.md)
├── input/                      # Topology JSON & YAML scenario configs (8 scenarios)
│   └── scenarios/              #   scenario_00_*.yaml … scenario_09_atm_system.yaml
├── output/                     # Pipeline output artefacts (dashboards, reports)
├── results/                    # Validation results from previous runs
└── docs/                       # Per-step methodology documentation
    ├── graph-model.md          #   Step 1 (Import): Formal graph model & dependency derivation rules
    ├── structural-analysis.md  #   Step 2 (Analyze): Structural metric catalogue and normalization
    ├── prediction.md           #   Step 2–3 (Analyze+Predict): RMAV formulas, AHP, GNN architecture
    ├── failure-simulation.md   #   Step 4 (Simulate): Fault injection & message flow simulation
    ├── validation.md           #   Step 5 (Validate): Statistical battery & topology-class gate system
    ├── visualization.md        #   Step 6 (Visualize): Dashboard & chart reference
    ├── antipatterns.md         #   Anti-pattern catalogue & detection heuristics
    ├── statistics.md           #   Statistics calculator module reference
    ├── scenario.md             #   Domain scenario library
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

Key methodological contributions:
- **Six dependency derivation rules** that make hidden failure paths in pub-sub systems explicit in a formal graph.
- **The RMAV quality model** — a four-dimensional AHP-weighted composite that decomposes criticality into orthogonal, actionable dimensions.
- **The MPCI metric** (Multi-Path Coupling Index) — a novel measure of intensified coupling through redundant shared channels.
- **The directed AP_c score** — a continuous articulation-point measure on directed graphs that correctly captures asymmetric SPOF risk.
- **Adaptive box-plot classification** — system-relative criticality thresholds that remain meaningful across all system sizes.
- **Empirical independence guarantee** — G_analysis(l) and G_structural are structurally separate, ensuring Q*(v) and I*(v) are genuinely independent.

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