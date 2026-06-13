# Software-as-a-Graph (SaG)

**Predict which components in a distributed system will cause the most damage when they fail — using only its architecture.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB)](https://react.dev/)
[![Docker](https://img.shields.io/badge/docker-compose-blue)](https://www.docker.com/)
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-green.svg)](https://neo4j.com/)

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Core Methodology & Pipeline](#core-methodology--pipeline)
3. [RMAV Quality Model & Formulas Reference](#rmav-quality-model--formulas-reference)
4. [Empirical Results](#empirical-results)
5. [Supported Platforms](#supported-platforms)
6. [Web Interface — SMART](#web-interface--smart)
7. [Installation & Development Setup](#installation--development-setup)
8. [Anti-Pattern Detection](#anti-pattern-detection)
9. [Python SDK (`saag`)](#python-sdk-saag)
10. [Project Structure](#project-structure)
11. [Research Context](#research-context)
12. [License](#license)

---

## The Problem

In distributed publish-subscribe systems (such as ROS 2, Apache Kafka, MQTT, and others), some components are structurally far more critical than others. When they fail, failures cascade through the system. Traditional approaches to identifying these weak points require either expensive runtime monitoring or waiting for production incidents.

This reactive posture has two fundamental problems:
- **Runtime monitoring adds overhead** — dynamic instrumentation imposes latency penalty on production systems that are often latency-sensitive or safety-critical.
- **Production incident reliance** — by the time a critical failure is discovered in production, the damage (data loss, service disruption, financial loss, safety risk) has already occurred.

---

## Core Methodology & Pipeline

> [!IMPORTANT]
> **Core Insight:** A component's position in the dependency graph reliably predicts its real-world failure impact — without any runtime data.

Software-as-a-Graph (SaG) operationalizes this insight into a 6-step core analytical pipeline, supported by an offline input preparation stage (Generate). The fundamental claim is that **topological structure alone** — how components are connected, what they depend on, and how strongly — encodes enough information to rank components by their potential failure impact with high statistical fidelity.

```
        ┌─────────────┐
        │  Offline    │
        │  Generate   │  (synthetic topologies for experiments & benchmarks)
        └──────┬──────┘
               │ topology JSON
               ▼
┌─────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  Step 1     │    │  Step 2              │    │  Step 3 (optional)  │
│  Model      │───▶│  Analyze             │───▶│  Predict            │
│  (import +  │    │  (M(v) + RMAV/Q(v)  │    │  (GNN ensemble;     │
│   export)   │    │   + Anti-Patterns)   │    │   inductive only)   │
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

| Step | What It Does | Key Output | Documentation |
|:---|:---|:---|:---|
| **Offline Prep: Generate** | Produces a synthetic pub-sub topology for experiments, benchmarks, or CI regression tests | Topology JSON (`data/system.json`) | [graph-generation.md](docs/graph-generation.md) |
| **1. Model** | Converts topology JSON into a formal weighted directed graph $G = (V, E, \tau_V, \tau_E, w)$ in Neo4j; derives logical `DEPENDS_ON` edges via six dependency rules; computes QoS-derived weights | $G_{\text{structural}}$ and $G_{\text{analysis}}(l)$ | [graph-model.md](docs/graph-model.md) |
| **2. Analyze** | Deterministic, closed-form. Computes 14 Tier-1 structural metrics $M(v)$; maps them to RMAV dimension scores and $Q^*(v)$ via AHP-weighted formulas; detects anti-patterns. | $M(v)$ metric vector, RMAV/$Q^*(v)$ scores, five-level classification, anti-pattern report | [structural-analysis.md](docs/structural-analysis.md) |
| **3. Predict** | Inductive, optional. A 3-layer `EdgeAwareHGTConv` (HGT) model trained on simulation labels $I(v)$ learns patterns the AHP composite cannot encode. | GNN criticality ranks, edge criticality, ensemble-blended $Q_{\text{ens}}(v)$ | [prediction.md](docs/prediction.md) |
| **4. Simulate** | Runs four parallel simulators (cascade, change-propagation, connectivity-loss, compromise-propagation). Provides training labels for Step 3 and ground truth for Step 5. | Per-dimension ground-truth $I_R(v)$, $I_M(v)$, $I_A(v)$, $I_V(v)$ and composite $I^*(v)$ | [failure-simulation.md](docs/failure-simulation.md) |
| **5. Validate** | Computes Spearman $\rho$ and Kendall $\tau$ between predictions and ground truth; evaluates F1, PG, SPOF-F1, FTR, Bootstrap CI, Wilcoxon | Statistical evidence of predictive validity | [validation.md](docs/validation.md) |
| **6. Visualize** | Renders interactive dashboards with network graphs, dependency matrices, cascade heatmaps, and RMAV radar charts | `dashboard.html` (fully self-contained) | [visualization.md](docs/visualization.md) |

### Scale Presets

The offline synthetic generator supports six scale presets for rapid experimentation:

| Preset | Apps | Topics | Brokers | Nodes | Libs | Typical Use |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `tiny` | 5 | 5 | 1 | 2 | 2 | Unit tests |
| `small` | 15 | 10 | 2 | 4 | 5 | Quick checks |
| `medium` | 50 | 30 | 3 | 8 | 10 | Development |
| `large` | 150 | 100 | 6 | 20 | 30 | Integration tests |
| `jumbo` | 300 | 120 | 10 | 40 | 50 | Large-scale benchmarks |
| `xlarge` | 500 | 300 | 10 | 50 | 100 | Performance benchmarks |

> [!NOTE]
> The `cli/generate_graph.py` script produces synthetic pub-sub topologies for evaluation, benchmarking, and reproducible experiments. Real deployments start at Step 1 (Model) with an actual architecture description.

---

## RMAV Quality Model & Formulas Reference

The RMAV quality model decomposes criticality into four orthogonal, actionable dimensions. It operates on the derived dependency graph $G_{\text{analysis}}(l)$ where edges point from *dependent* to *dependency* (e.g., subscriber $\rightarrow$ publisher, application $\rightarrow$ broker).

| Dimension | Question Answered | High Score Means | Primary Stakeholder |
|:---|:---|:---|:---|
| **R — Reliability** | How broadly does failure propagate? | Failure cascades widely and is hard to contain | Reliability Engineer |
| **M — Maintainability** | How hard is this to change safely? | Tightly coupled; structural bottleneck | Software Architect |
| **A — Availability** | Is this a structural single point of failure? | Removing it partitions the dependency graph | DevOps / SRE |
| **V — Vulnerability** | How attractive a target is this for attack? | Central, reachable, high-value downstream | Security Engineer |

Quality scores are computed per component $v$. Individual input metrics are normalized to $[0, 1]$ using rank-based robust normalization. AHP-derived weights are blended with a uniform prior using a shrinkage factor $\lambda = 0.70$.

### Reliability — $R(v)$

Reliability measures how broadly and deeply a component's failure propagates. Because Topic nodes have no in-degree in the dependency graph, a topic-specific formula is applied:

- **Standard Formula** (Application, Broker, Node, Library):
  $$R(v) = 0.45 \times \text{RPR}(v) + 0.30 \times \text{DG}_{\text{in}}(v) + 0.25 \times \text{CDPot}_{\text{enh}}(v)$$
  Where:
  - $\text{RPR}(v)$ — Reverse PageRank computed on $G^T$ (captures global cascade reach in the failure propagation direction).
  - $\text{DG}_{\text{in}}(v)$ — Normalized in-degree (captures immediate blast radius / direct dependents).
  - $\text{CDPot}_{\text{enh}}(v)$ — Enhanced Cascade Depth Potential, tracking depth $\times$ breadth of the cascade:
    $$\text{CDPot}_{\text{base}}(v) = \frac{\text{RPR}(v) + \text{DG}_{\text{in}}(v)}{2} \times \left(1 - \min\left(\frac{\text{DG}_{\text{out\_raw}}(v)}{\max(\text{DG}_{\text{in\_raw}}(v), 10^{-9})}, 1.0\right)\right)$$
    $$\text{CDPot}_{\text{enh}}(v) = \min\left(\text{CDPot}_{\text{base}}(v) \times (1 + \text{MPCI}(v)), 1.0\right)$$
  - $\text{MPCI}(v)$ — Multi-Path Coupling Index, summing extra shared message channels establishing dependencies:
    $$\text{MPCI}(v) = \frac{1}{|V|-1} \sum_{e \in \text{InEdges}(v)} \max(\text{path\_count}(e) - 1, 0)$$

- **Topic Formula** (Topic nodes only):
  $$R_{\text{topic}}(v) = 0.50 \times \text{FOC}(v) + 0.50 \times \text{CDPot}_{\text{topic}}(v)$$
  Where:
  - $\text{FOC}(v)$ — Fan-Out Criticality (log-scaled message frequency modulated by subscriber count).
  - $\text{CDPot}_{\text{topic}}(v)$ — Topic-specific cascade potential, penalizing topics with many subscribers but low publisher redundancy:
    $$\text{CDPot}_{\text{topic}}(v) = \text{FOC}(v) \times (1 - \min(\text{publisher}_{\text{count\_norm}}(v), 1.0))$$

### Maintainability — $M(v)$

Maintainability measures how structurally embedded a component is in the topology, capturing static code fragility and deployment-coupling risk:
$$M(v) = 0.35 \times \text{BT}(v) + 0.30 \times \text{w}_{\text{out}}(v) + 0.15 \times \text{CQP}(v) + 0.12 \times \text{CouplingRisk}_{\text{enh}}(v) + 0.08 \times (1 - \text{CC}(v))$$

Where:
- $\text{BT}(v)$ — Betweenness centrality computed using inverted weights ($1/w$) as path distances.
- $\text{w}_{\text{out}}(v)$ — QoS-weighted efferent coupling (sum of outgoing dependency weights).
- $\text{CQP}(v)$ — Code Quality Penalty, incorporating cyclomatic complexity, lines of code, LCOM, and package imports:
  $$\text{CQP}(v) = 0.10 \times \text{loc}_{\text{norm}}(v) + 0.35 \times \text{complexity}_{\text{norm}}(v) + 0.30 \times \text{instability}_{\text{code}}(v) + 0.25 \times \text{lcom}_{\text{norm}}(v)$$
  *(CQP is zero for non-Application/Library types, degrading gracefully).*
- $\text{CouplingRisk}_{\text{enh}}(v)$ — Topological instability index modulated by path complexity ($\text{PC}(v)$):
  $$\text{CouplingRisk}_{\text{enh}}(v) = \min\left(1.0, \left(1 - |2 \cdot \text{Instability}(v) - 1|\right) \times \left(1 + 0.10 \times \text{PC}(v)\right)\right)$$
- $\text{CC}(v)$ — Clustering coefficient computed on the undirected projection of the layer graph.

### Availability — $A(v)$

Availability measures whether a component is a structural single point of failure (SPOF):
$$A(v) = 0.35 \times \text{AP}_{\text{c\_directed}}(v) + 0.25 \times \text{QSPOF}(v) + 0.25 \times \text{BR}(v) + 0.10 \times \text{CDI}(v) + 0.05 \times w(v)$$

Where:
- $\text{AP}_{\text{c\_directed}}(v)$ — Directed Articulation Point score, capturing graph fragmentation when $v$ is removed.
- $\text{QSPOF}(v)$ — QoS-scaled SPOF severity: $\text{AP}_{\text{c\_directed}}(v) \times w(v)$.
- $\text{BR}(v)$ — Bridge Ratio (fraction of incident edges that are graph bridges).
- $\text{CDI}(v)$ — Connectivity Degradation Index, measuring average path length elongation in the surviving graph.
- $w(v)$ — Pure operational priority weight derived from QoS properties (Reliability, Durability, Priority).

### Vulnerability — $V(v)$

Vulnerability measures how attractive a component is as an adversarial target:
$$V(v) = 0.40 \times \text{REV}(v) + 0.35 \times \text{RCL}(v) + 0.25 \times \text{w}_{\text{in}}(v)$$

Where:
- $\text{REV}(v)$ — Reverse eigenvector centrality computed on $G^T$ (captures downstream reach into other important components).
- $\text{RCL}(v)$ — Reverse closeness centrality computed on $G^T$ (captures entry propagation speed).
- $\text{w}_{\text{in}}(v)$ — QoS-weighted in-degree (QADS - QoS-weighted Attack-Dependent Surface).

### Overall Quality Score — $Q^*(v)$

The overall criticality score combines the four dimensions:
$$Q^*(v) = 0.43 \times A(v) + 0.24 \times R(v) + 0.17 \times M(v) + 0.16 \times V(v)$$

Availability dominates ($0.43$) because structural SPOF failure partitions the graph with certainty, whereas cascade propagation and coupling risks are probabilistic.

### Criticality Classification (Adaptive Box-Plot)

Scores are mapped to five criticality tiers using adaptive box-plot thresholding derived from the system's own score distribution:

| Level | Threshold |
|:---|:---|
| **CRITICAL** | $score > Q3 + 1.5 \times IQR$ |
| **HIGH** | $Q3 < score \le upper\ fence$ |
| **MEDIUM** | $Median < score \le Q3$ |
| **LOW** | $Q1 < score \le Median$ |
| **MINIMAL** | $score \le Q1$ |

> [!NOTE]
> For small topologies (fewer than 12 components), the box-plot method falls back to fixed-percentile classification (top 10% $\rightarrow$ CRITICAL).

### RMAV Interpretation Triage Patterns

The four-dimensional criticality signature profiles the exact nature of architectural risk:

| Pattern | R | M | A | V | Risk Type | Recommended Action |
|:---|:---:|:---:|:---:|:---:|:---|:---|
| **Total Hub** | H | H | H | H | Catastrophic Hub | Introduce structural redundancy + network isolation + hardening |
| **Reliability Hub** | H | L | L | L | Wide Cascade Propagator | Implement retry logic, circuit breakers, and back-pressure |
| **Bottleneck** | L | H | L | L | High Change Fragility | Extract clean interface boundaries and reduce efferent coupling |
| **SPOF** | L | L | H | L | High Availability Risk | Introduce redundant instances or active-passive failover |
| **Attack Target** | L | L | L | H | High Attack Exposure | Enforce zero-trust boundaries and isolate network paths |
| **Fragile Hub** | H | L | H | L | Multi-Channel Cascade Sink | Reduce shared-topic count between subscriber/publisher pairs |
| **Exposed Bottleneck** | L | H | L | H | Vulnerable Bottleneck | Refactor code quality issues and shield component interfaces |

### GNN Ensemble (Step 3 Predict)

The optional inductive Predict stage refines the deterministic RMAV scores using a Graph Attention Network trained on simulation ground-truth impact $I(v)$:
$$Q_{\text{ensemble}}(v) = \alpha \cdot Q_{\text{GNN}}(v) + (1 - \alpha) \cdot Q_{\text{RMAV}}(v)$$
Where $\alpha \in \mathbb{R}^5$ is a learned blending coefficient vector initialized at $0.5$.

---

## Empirical Results

The methodology has been validated across eight domain scenarios (including autonomous vehicles, IoT, high-frequency trading, healthcare, and air traffic management):

| Metric | Target | Achieved |
|:---|:---:|:---:|
| Spearman $\rho(Q^*, I^*)$ overall | $\ge 0.85$ | **> 0.87** |
| Spearman $\rho(Q^*, I^*)$ at large scale (150–300+ nodes) | — | **0.943** |
| Overall F1-score | $\ge 0.90$ | **> 0.90** |
| Predictive Gain (PG) vs. degree baseline | $> 0.03$ | **> 0.03** |
| Best prediction layer | — | Application layer outperforms infrastructure layer |
| Scale effect | — | Prediction accuracy improves with system size |

---

## Supported Platforms

The graph model maps naturally to any publish-subscribe middleware:

| Graph Concept | ROS 2 / DDS | Apache Kafka | MQTT |
|:---|:---|:---|:---|
| **Application** | ROS Node | Producer / Consumer | MQTT Client |
| **Topic** | ROS Topic | Kafka Topic | MQTT Topic |
| **Broker** | DDS Participant | Kafka Broker | MQTT Broker |
| **Infrastructure Node** | Host / Container | Broker Host | Broker Server |
| **Library** | ROS package dependency | Maven artifact dependency | Paho client library |

---

## Web Interface — SMART

The **SMART** web interface (built using Next.js 16, React 19, TypeScript, and Tailwind CSS) provides an interactive dashboard:
1. **Dashboard** — High-level KPIs, criticality heatmaps, and lists of top critical components.
2. **Graph Explorer** — Interactive 2D/3D force-directed dependency graph displaying system layers.
3. **Analysis** — Configurable structural analysis and RMAV quality scoring triggers.
4. **Simulation** — Failure injection animation showing real-time cascade propagation paths.
5. **Statistics** — QoS risk scatter plots, topic fan-out distributions, and communication loading.
6. **Settings** — Configuration interface for Neo4j database connections.

---

## Installation & Development Setup

### Prerequisites

- **Python 3.9+** (Virtual environment recommended)
- **Neo4j 5.x** (With GDS and APOC plugins enabled)
- **Node.js 18+** (Frontend dashboard execution only)

### 1. Neo4j Database Setup

The recommended way to run Neo4j locally with GDS (Graph Data Science) and APOC is via Docker:

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:2026.02.2
```

> [!WARNING]
> The default credentials (`neo4j` / `password`) are for local development only. Change these in the root `.env` file before shared deployments.

### 2. Backend & CLI Installation

Initialize a virtual environment, install PyTorch (with PyG dependencies), and install the CLI/SDK package in editable mode:

```bash
# Set up virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install PyTorch and PyG (CPU version)
pip install torch==2.5.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

# Install package with all extras
pip install -e ".[all]"
```

To start the FastAPI REST server:
```bash
uvicorn api.main:app --reload --port 8000
```
API docs are available at `http://localhost:8000/docs`.

### 3. Frontend Installation

Navigate to the dashboard directory, install dependencies, and start the Next.js development server:

```bash
cd smart
npm install
npm run dev
```
Dashboard is available at `http://localhost:7000`.

### 4. Running the Pipeline via CLI

Run the full pipeline (Model $\rightarrow$ Analyze $\rightarrow$ Predict $\rightarrow$ Simulate $\rightarrow$ Validate $\rightarrow$ Visualize) on a layer in a single command:

```bash
python cli/run.py --all --layer system
```

#### Individual Pipeline Scripts

All CLI scripts are stored in the [cli/](cli/) folder. Run them from the project root:

```bash
# Generate a synthetic pub-sub topology JSON
python cli/generate_graph.py --scale medium --output data/system.json

# Step 1: Import topology to Neo4j and derive DEPENDS_ON edges
python cli/import_graph.py --input data/system.json --clear

# Step 2: Compute structural metrics, RMAV/Q scores, and anti-patterns
python cli/analyze_graph.py --layer system --predict

# Step 3: Optional GNN training and inference
python cli/train_graph.py --layer system
python cli/predict_graph.py --layer system --gnn-model output/gnn_checkpoints/best_model

# Step 4: Run failure simulations to obtain ground truth I(v)
python cli/simulate_graph.py fault-inject --input data/system.json --export-json

# Step 5: Compute Spearman correlation and validation gate metrics
python cli/validate_graph.py report --input data/system.json --qos

# Step 6: Renders interactive HTML dashboard report
python cli/visualize_graph.py --layer system --output output/dashboard.html --open
```

#### Pipeline Utilities

```bash
# Standalone anti-pattern detection (CI/CD quality gating)
python cli/detect_antipatterns.py --layer system --output output/antipatterns.json

# Export database graph representation to JSON
python cli/export_graph.py --output output/graph_export.json

# Run scalability performance benchmark
python cli/benchmark.py

# Execute full pipeline validation across all 8 scenarios
bash cli/run_scenarios.sh

# Run CLI-based statistics dashboard
python cli/statistics_graph.py --layer system
```

### 5. Running Tests

```bash
pytest                # Run all tests
pytest -x             # Halt execution on first failure
pytest -k "reliability" # Run tests matching name pattern
```

---

## Anti-Pattern Detection

The `AntiPatternDetector` reviews RMAV scoring results and flags structural deficiencies in CI/CD pipelines.

| Anti-Pattern | Trigger Condition | Severity |
|:---|:---|:---|
| **SPOF** | Component is a directed articulation point | CRITICAL |
| **FAILURE_HUB** | $R(v) \ge$ CRITICAL threshold | CRITICAL |
| **GOD_COMPONENT** | $M(v) \ge$ CRITICAL and betweenness centrality $> 0.3$ | CRITICAL |
| **TARGET** | $V(v) \ge$ CRITICAL threshold | CRITICAL |
| **SYSTEMIC_RISK** | CRITICAL components account for $> 20\%$ of system | CRITICAL |
| **BRIDGE_EDGE** | Edge is a graph bridge | HIGH |
| **EXPOSURE** | $V(v) ==$ HIGH and closeness centrality $> 0.6$ | HIGH |
| **CYCLE** | Strongly Connected Component size $\ge 2$ nodes | HIGH |
| **HUB_AND_SPOKE** | Clustering coefficient $< 0.1$ and degree centrality $> 3$ | MEDIUM |
| **CHAIN** | Weakly connected sequence length $\ge 4$ nodes | MEDIUM |

> [!TIP]
> Detection runs return exit codes (0: clean, 1: warnings/smells, 2: critical/high patterns detected) suitable for pre-merge gates.

---

## Python SDK (`saag`)

The `saag` package exposes a fluent programmatic builder API for custom scripting:

```python
import saag

# Run Analyze + Simulate + Validate + Visualize in a fluent chain
result = (
    saag.Pipeline.from_json("data/system.json", clear=True)
        .analyze(layer="app")          # Deterministic structural metrics & RMAV
        .simulate(layer="app", mode="exhaustive") # Ground-truth simulation
        .validate()                    # Statistical validation
        .visualize(output="output/report.html") # Renders dashboard
        .run()
)

print(f"Spearman ρ = {result.validation.overall.spearman:.3f}")
print(f"F1-Score   = {result.validation.overall.f1:.3f}")
```

### Key SDK Classes

| Class | Location | Purpose |
|:---|:---|:---|
| [Pipeline](saag/pipeline.py#L12) | `saag.Pipeline` | Fluent builder to sequence and execute the pipeline |
| [Client](saag/client.py#L9) | `saag.Client` | Low-level service facade wrapper |
| [AnalysisResult](saag/models.py#L102) | `saag.AnalysisResult` | Step 2 scoring output structural metrics and RMAV |
| [PredictionResult](saag/models.py#L180) | `saag.PredictionResult` | Step 3 scoring output GNN ensemble-blended scores |
| [ValidationResult](saag/models.py#L369) | `saag.ValidationResult` | Step 5 validation output including Spearman and gate metrics |

---

## Project Structure

```
.
├── cli/                        # CLI pipeline scripts
│   ├── run.py                  #   Master pipeline orchestrator (--all flag)
│   ├── generate_graph.py       #   Offline Prep: Generate — synthetic pub-sub topology
│   ├── import_graph.py         #   Step 1a: Model (Import) — Neo4j import & dependency derivation
│   ├── export_graph.py         #   Step 1b: Model (Export) — export graph from Neo4j to JSON
│   ├── analyze_graph.py        #   Step 2: Analyze — structural metrics + RMAV/Q scoring + anti-patterns
│   ├── train_graph.py          #   Step 3a: Predict (Train) — GNN training (optional; requires Step 4 labels)
│   ├── predict_graph.py        #   Step 3b: Predict (Inference) — GNN inference on a new graph
│   ├── detect_antipatterns.py  #   Standalone anti-pattern / CI gate
│   ├── simulate_graph.py       #   Step 4: Simulate — fault-inject | message-flow | combined
│   ├── validate_graph.py       #   Step 5: Validate — single | sweep | report | compare
│   ├── visualize_graph.py      #   Step 6: Visualize — interactive HTML dashboard
│   ├── statistics_graph.py     #   Statistics dashboard (topology & communication analytics)
│   ├── benchmark.py            #   Benchmark across scale presets
│   ├── loso_evaluate.py        #   Leave-One-Scenario-Out GNN validation protocol
│   ├── multi_seed_summary.py   #   Aggregate results across seeds
│   └── run_scenarios.sh        #   Full pipeline across 8 domain scenarios
│
├── tools/                      # Standalone tooling (no Neo4j dependency)
│   └── generation/             #   Statistical pub-sub topology generator
│       ├── generator.py        #     Core generator (structural edges only)
│       ├── service.py          #     High-level GenerationService wrapper
│       ├── models.py           #     Scale presets & statistical config
│       └── datasets.py         #     Domain-specific naming & QoS mappings
│
├── saag/                       # Core Python SDK & Logic
│   ├── core/                   #   Domain models, ports, Neo4j & memory repos
│   ├── analysis/               #   Structural metrics + anti-pattern detection
│   ├── prediction/             #   RMAV quality scoring + GNN service + ensemble blending
│   ├── simulation/             #   Four parallel failure/event simulators
│   ├── validation/             #   Per-dimension statistical validation
│   ├── visualization/          #   Dashboard & chart generation
│   ├── infrastructure/         #   Neo4j drivers & data persistence
│   ├── pipeline.py             #   saag.Pipeline — fluent builder
│   ├── client.py               #   saag.Client — service façade
│   └── models.py               #   Result & data model types
│
├── api/                        # FastAPI application (Hexagonal Architecture)
│   ├── presenters/             #   Response formatting & API translation
│   ├── routers/                #   REST endpoints (thin layer)
│   └── dependencies.py         #   Service & Repository injection
│
├── smart/                      # Web Frontend (SMART - Next.js)
├── tests/                      # Pytest unit & integration tests
├── examples/                   # Annotated Python usage examples
├── data/                       # Topology JSON & YAML scenario configs
├── output/                     # Pipeline output artefacts
├── results/                    # Validation results from previous runs
├── models/                     # Trained GNN model checkpoints
└── docs/                       # Per-step methodology documentation
```

Key directories in the workspace:
- [cli/](cli/) — Entry points for the command-line interface.
- [tools/](tools/) — Auxiliary graph generation utilities.
- [saag/](saag/) — Implementation of the core pipeline logic and mathematical models.
- [api/](api/) — FastAPI endpoints for backend integration.
- [smart/](smart/) — Next.js single page dashboard application.
- [docs/](docs/) — Detailed methodology references.

---

## Research Context

<!-- 
This framework is the software artifact for the PhD dissertation **"Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems"** at Istanbul Technical University, Department of Computer Engineering.

The underlying methodology was peer-reviewed and published at:

> **IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE 2025)**
> *A Graph-Based Dependency Analysis Method for Identifying Critical Components in Distributed Publish-Subscribe Systems*
-->

The primary research contribution is the demonstration that **topological graph metrics can reliably predict real-world failure impact without runtime instrumentation**, validated empirically across multiple application domains and scale dimensions.

Key methodological contributions:
- **Six dependency derivation rules** making pub-sub logical dependencies explicit.
- **The RMAV quality model** decomposing criticality into orthogonal, actionable dimensions.
- **The MPCI metric** (Multi-Path Coupling Index) measuring coupling intensity.
- **The directed $AP_c$ score** correctly capturing directed single points of failure.
- **Adaptive box-plot classification** determining system-relative thresholds.
- **Empirical independence guarantee** structurally separating predictor and simulation views.

<!--
### Citation

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
-->

---

## License

See the [LICENSE](LICENSE) file for terms of use.
