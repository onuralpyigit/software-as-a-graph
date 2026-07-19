# Software-as-a-Graph (SaG)

**Predict which components in a distributed system will cause the most damage when they fail — using only its architecture.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB)](https://react.dev/)
[![Docker](https://img.shields.io/badge/docker-compose-blue)](https://www.docker.com/)
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-green.svg)](https://neo4j.com/)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Core Methodology & Pipeline](#core-methodology--pipeline)
3. [RMAV Quality Model](#rmav-quality-model)
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
┌─────────────┐    ┌──────────────────────┐    ┌─────────────────────────────┐
│  Step 1     │    │  Step 2              │    │  Step 3                     │
│  Model      │───▶│  Analyze             │───▶│  Predict (unified)          │
│  (import +  │    │  (M(v) structural    │    │  (RMAV/Q(v) always + GNN    │
│   export)   │    │   metrics only)      │    │   when available + Anti-    │
│             │    │                      │    │   Patterns + Explanation)   │
└─────────────┘    └──────────────────────┘    └─────────────────────────────┘
                             │                             │
        ┌────────────────────┘                            │
        ▼                                                 ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐
│  Step 4     │    │  Step 5     │    │  Step 6     │    │  Step 7                     │
│  Simulate   │───▶│  Validate   │───▶│  Prescribe  │───▶│  Visualize                  │
│  (I(v) GT)  │    │  (ρ, F1)    │    │  (Delta G)  │    │                             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────────────────────┘
                                                                        │
                                                                        ▼
                                                                 HTML Dashboard
```

| Step | What It Does | Key Output | Documentation |
|:---|:---|:---|:---|
| **Offline Prep: Generate** | Produces a synthetic pub-sub topology for experiments, benchmarks, or CI regression tests | Topology JSON (`data/system.json`) | [graph-generation.md](docs/graph-generation.md) |
| **1. Model** | Converts topology JSON into a formal weighted directed graph $G = (V, E, \tau_V, \tau_E, w)$ in Neo4j; derives logical `DEPENDS_ON` edges via six dependency rules; computes QoS-derived weights | $G_{\text{structural}}$ and $G_{\text{analysis}}(l)$ | [graph-model.md](docs/graph-model.md) |
| **2. Analyze** | Deterministic, closed-form. Computes 13 Tier-1 structural metrics $M(v)$ only. | $M(v)$ metric vector | [structural-analysis.md](docs/structural-analysis.md) |
| **3. Predict** | Unified Prediction Step. Always maps $M(v)$ to RMAV dimension scores and $Q^*(v)$ via AHP-weighted formulas (deterministic); blends in a 3-layer `EdgeAwareHGTConv` (HGT) model trained on simulation labels $I(v)$ when a checkpoint is available (falls back to RMAV otherwise); detects anti-patterns; generates a natural-language explanation. | RMAV/$Q^*(v)$ scores, five-level classification, GNN criticality ranks (when available), anti-pattern report, explanation | [prediction.md](docs/prediction.md) |
| **4. Simulate** | Runs the FailureSimulator (producing canonical composite $I^*(v)$ and RM-AV ground truths) or FaultInjector (producing BFS feed-loss $I(v)$). Provides training labels for Step 3 and validation ground truth. | Per-dimension ground-truth $I_R(v)$, $I_M(v)$, $I_A(v)$, $I_V(v)$ and composite $I^*(v)$ / feed-loss $I(v)$ | [failure-simulation.md](docs/failure-simulation.md) |
| **5. Validate** | Computes Spearman $\rho$ and Kendall $\tau$ between predictions and ground truth; evaluates F1, PG, SPOF-F1, FTR, Bootstrap CI, Wilcoxon | Statistical evidence of predictive validity | [validation.md](docs/validation.md) |
| **6. Prescribe** | Generates rule-based architectural optimization recommendations and verifies them in a closed-loop simulation on a mutated graph | `PrescribeResult` with baseline vs mutated SRI | [prescription.md](docs/prescription.md) |
| **7. Visualize** | Renders interactive dashboards with network graphs, dependency matrices, cascade heatmaps, and RMAV radar charts | `dashboard.html` (fully self-contained) | [visualization.md](docs/visualization.md) |

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

## RMAV Quality Model

The RMAV quality model decomposes criticality into four orthogonal, actionable dimensions. It operates on the derived dependency graph where edges point from *dependent* to *dependency* (e.g., subscriber $\rightarrow$ publisher, application $\rightarrow$ broker).

| Dimension | Question Answered | High Score Means | Primary Stakeholder |
|:---|:---|:---|:---|
| **R — Reliability** | How broadly does failure propagate? | Failure cascades widely and is hard to contain | Reliability Engineer |
| **M — Maintainability** | How hard is this to change safely? | Tightly coupled; structural bottleneck | Software Architect |
| **A — Availability** | Is this a structural single point of failure? | Removing it partitions the dependency graph | DevOps / SRE |
| **V — Vulnerability** | How attractive a target is this for attack? | Central, reachable, high-value downstream | Security Engineer |

### Reliability

Reliability measures how broadly and deeply a component's failure propagates. Standard components (Applications, Brokers, Nodes, Libraries) are evaluated based on Reverse PageRank, normalized in-degree, and Enhanced Cascade Depth Potential. Topic components are evaluated using Fan-Out Criticality (FOC) and topic-specific cascade potential.

**Standard Formula (Applications, Brokers, Nodes, Libraries):**

$$
R(v) = 0.45 \times \text{RPR}(v) + 0.30 \times \text{DG-in}(v) + 0.25 \times \text{CDPot-enh}(v)
$$

Where:
- $\text{RPR}(v)$ is the Reverse PageRank (transitive cascade reach).
- $\text{DG-in}(v)$ is the normalized in-degree (immediate blast radius).
- $\text{CDPot-enh}(v)$ is the Enhanced Cascade Depth Potential:

$$
\text{CDPot-enh}(v) = \min\left(\text{CDPot-base}(v) \times (1 + \text{MPCI}(v)), 1.0\right)
$$

$$
\text{CDPot-base}(v) = \frac{\text{RPR}(v) + \text{DG-in}(v)}{2} \times \left(1 - \min\left(\frac{\text{DG-out-raw}(v)}{\max(\text{DG-in-raw}(v), \epsilon)}, 1\right)\right)
$$

  with $\text{MPCI}(v)$ as the Multi-Path Coupling Index, and $\text{DG-out-raw}(v)$ / $\text{DG-in-raw}(v)$ as the raw integer degree metrics.

**Topic Formula:**

$$
R\text{-topic}(v) = 0.50 \times \text{FOC}(v) + 0.50 \times \text{CDPot-topic}(v)
$$

Where:
- $\text{FOC}(v)$ is the Fan-Out Criticality (log1p of message frequency $\times$ subscriber count).
- $\text{CDPot-topic}(v) = \text{FOC}(v) \times (1 - \min(\text{publisher-count-norm}(v), 1))$ represents topic fan-out depth.

### Maintainability

Maintainability measures how structurally embedded a component is in the topology, capturing static code fragility and deployment-coupling risk. It integrates betweenness centrality, efferent coupling, a Code Quality Penalty (CQP—incorporating cyclomatic complexity, lines of code, LCOM, and imports), topological instability, and the clustering coefficient.

**Formula:**

$$
M(v) = 0.35 \times \text{BT}(v) + 0.30 \times w\text{-out}(v) + 0.15 \times \text{CQP}(v) + 0.12 \times \text{CouplingRisk-enh}(v) + 0.08 \times (1 - \text{CC}(v))
$$

Where:
- $\text{BT}(v)$ is the betweenness centrality (fraction of shortest dependency paths passing through $v$).
- $w\text{-out}(v)$ is the QoS-weighted out-degree (efferent coupling).
- $\text{CQP}(v)$ is the Code Quality Penalty (only for Application and Library nodes; 0 otherwise):

$$
\text{CQP}(v) = 0.10 \times \text{loc-norm}(v) + 0.35 \times \text{complexity-norm}(v) + 0.30 \times \text{instability-code}(v) + 0.25 \times \text{lcom-norm}(v)
$$

- $\text{CouplingRisk-enh}(v)$ is the Enhanced Coupling Risk:

$$
\text{CouplingRisk-enh}(v) = \min\left(1.0, \text{CouplingRisk-base}(v) \times (1 + \Delta \times \text{path-complexity}(v))\right)
$$

$$
\text{CouplingRisk-base}(v) = 1 - |2 \times \text{Instability-topo}(v) - 1|
$$

$$
\text{Instability-topo}(v) = \frac{\text{DG-out-raw}(v)}{\text{DG-in-raw}(v) + \text{DG-out-raw}(v) + \epsilon}
$$

  with the coupling path delta $\Delta = 0.10$.
- $\text{CC}(v)$ is the clustering coefficient (measuring local redundancy).

### Availability

Availability measures whether a component is a structural single point of failure (SPOF). It combines directed articulation point scores, QoS-scaled SPOF severity, bridge ratios, connectivity degradation, and operational priority weights.

**Formula:**

$$
A(v) = 0.35 \times \text{AP}\text{-c-directed}(v) + 0.25 \times \text{QSPOF}(v) + 0.25 \times \text{BR}(v) + 0.10 \times \text{CDI}(v) + 0.05 \times w(v)
$$

Where:
- $\text{AP}\text{-c-directed}(v)$ is the directed articulation point score (reachability-based SPOF signal).
- $\text{QSPOF}(v) = \text{AP}\text{-c-directed}(v) \times w(v)$ is the QoS-weighted SPOF severity.
- $\text{BR}(v)$ is the bridge ratio (fraction of non-redundant edges).
- $\text{CDI}(v)$ is the Connectivity Degradation Index (path elongation on removal).
- $w(v)$ is the component's operational QoS weight.

### Vulnerability

Vulnerability measures how attractive a component is as an adversarial target. It is calculated using reverse eigenvector centrality, reverse closeness centrality, and QoS-weighted in-degree (QADS).

**Formula:**

$$
V(v) = 0.40 \times \text{REV}(v) + 0.35 \times \text{RCL}(v) + 0.25 \times w\text{-in}(v)
$$

Where:
- $\text{REV}(v)$ is the Reverse Eigenvector Centrality (transitive compromise reach on $G^T$).
- $\text{RCL}(v)$ is the Reverse Closeness Centrality (adversarial reach speed via harmonic centrality on $G^T$).
- $w\text{-in}(v)$ is the QoS-weighted in-degree (QADS: QoS-weighted Attack-Dependent Surface).

### Overall Criticality Score

The overall criticality score combines the four dimensions. Availability dominates because structural SPOF failure partitions the graph with certainty, whereas cascade propagation and coupling risks are probabilistic.

**AHP-Weighted Composite Formula:**

$$
Q(v) = 0.43 \times A(v) + 0.24 \times R(v) + 0.17 \times M(v) + 0.16 \times V(v)
$$

Alternatively, an equal-weight baseline (0.25 each) can be activated for baseline comparisons:

$$
Q\_{\text{equal}}(v) = 0.25 \times A(v) + 0.25 \times R(v) + 0.25 \times M(v) + 0.25 \times V(v)
$$




### Criticality Classification (Adaptive Box-Plot)

Scores are mapped to five criticality tiers using adaptive box-plot thresholding derived from the system's own score distribution:

| Level | Threshold |
|:---|:---|
| **CRITICAL** | Score > Q3 + 1.5 * IQR |
| **HIGH** | Q3 < Score <= Upper Fence |
| **MEDIUM** | Median < Score <= Q3 |
| **LOW** | Q1 < Score <= Median |
| **MINIMAL** | Score <= Q1 |

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

### GNN Prediction (Step 3 Predict)
 
The optional inductive Predict stage refines the deterministic RMAV scores by utilizing a Graph Neural Network (trained on simulation ground-truth impact) to perform GNN-only criticality predictions.

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

# Step 2: Compute structural metrics only
python cli/analyze_graph.py --layer system

# Step 3: Unified Prediction Step — RMAV always; optional GNN training/inference
python cli/predict_graph.py --layer system
python cli/train_graph.py --layer system
python cli/predict_graph.py --layer system --gnn-model output/gnn_checkpoints/best_model

# Step 4: Run failure simulations to obtain ground truth I(v)
python cli/simulate_graph.py fault-inject --input data/system.json --export-json

# Step 5: Compute Spearman correlation and validation gate metrics
python cli/validate_graph.py report --input data/system.json --qos

# Step 6: Prescribe optimizations and validate in closed loop
python cli/prescribe_graph.py --layer system

# Step 7: Renders interactive HTML dashboard report
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
| **SPOF** | Component is a directed articulation point, or its availability score $A(v)$ exceeds the population's upper fence | CRITICAL |
| **SYSTEMIC_RISK** | CRITICAL components account for $> 20\%$ of system | CRITICAL |
| **GOD_COMPONENT** | $M(v) \ge$ CRITICAL and betweenness centrality $> 0.3$ | CRITICAL |
| **FAILURE_HUB** | $R(v) \ge$ CRITICAL threshold and out-degree above the median | CRITICAL |
| **TARGET** | $V(v) \ge$ CRITICAL threshold | CRITICAL |
| **COMPOUND_RISK** | Component is simultaneously a SPOF and a GOD_COMPONENT/HUB_AND_SPOKE/FAILURE_HUB | CRITICAL |
| **CYCLE** | Strongly Connected Component size $\ge 2$ nodes | HIGH |
| **BRIDGE_EDGE** | Edge is a graph bridge | HIGH |
| **BOTTLENECK_EDGE** | Edge betweenness centrality exceeds the edge-betweenness population's upper fence ($Q_3 + 1.5 \times IQR$) | HIGH |
| **BROKER_OVERLOAD** | Broker's availability $\ge 2\times$ the median broker availability, or the sole broker in the system | HIGH |
| **DEEP_PIPELINE** | Longest dependency chain $\ge 5$ hops (app layer) | HIGH |
| **EXPOSURE** | $V(v) ==$ HIGH and closeness centrality $> 0.6$ | HIGH |
| **CONCENTRATION_RISK** | Top-3 components hold $> 50\%$ of total PageRank (system-wide, $\ge 5$ components) | MEDIUM |
| **TOPIC_FANOUT** | Topic subscriber count exceeds the topic population's upper fence (floor of 5) | MEDIUM |
| **CHATTY_PAIR** | Bidirectional edges with $edge\_score(u \to v) \times edge\_score(v \to u) > 0.25$ | MEDIUM |
| **QOS_MISMATCH** | Publisher QoS weight $w(u) <$ subscriber QoS weight $w(v) - 0.3$ | MEDIUM |
| **ORPHANED_TOPIC** | Topic has no publishers or no subscribers | MEDIUM |
| **UNSTABLE_INTERFACE** | Maintainability score $> 0.80$ and enhanced coupling risk $> 0.80$ | MEDIUM |
| **HUB_AND_SPOKE** | Clustering coefficient $< 0.1$ and degree centrality $> 3$ | MEDIUM |
| **CHAIN** | Weakly connected sequence length $\ge 4$ nodes | MEDIUM |
| **ISOLATED** | Component has no incoming or outgoing dependencies in this layer | MEDIUM |

See [antipatterns.md](docs/antipatterns.md) for the full formal specification of each pattern.

> [!TIP]
> Detection runs return exit codes (0: clean, 1: warnings/smells, 2: critical/high patterns detected) suitable for pre-merge gates.

---

## Python SDK (`saag`)

The `saag` package exposes a fluent programmatic builder API for custom scripting:

```python
import saag

# Run Analyze + Simulate + Validate + Prescribe + Visualize in a fluent chain
result = (
    saag.Pipeline.from_json("data/system.json", clear=True)
        .analyze(layer="app")          # Deterministic structural metrics only
        .simulate(layer="app", mode="exhaustive") # Ground-truth simulation
        .validate()                    # Statistical validation
        .prescribe()                   # Prescriptive remediation generation
        .visualize(output="output/report.html") # Renders dashboard
        .run()
)

print(f"Spearman ρ = {result.validation.overall.spearman:.3f}")
print(f"F1-Score   = {result.validation.overall.f1:.3f}")
if result.prescription:
    print(f"SRI Imp.   = {result.prescription.sri_improvement:.4f}")
```

### Key SDK Classes

| Class | Location | Purpose |
|:---|:---|:---|
| [Pipeline](saag/pipeline.py#L12) | `saag.Pipeline` | Fluent builder to sequence and execute the pipeline |
| [Client](saag/client.py#L9) | `saag.Client` | Low-level service facade wrapper |
| [AnalysisResult](saag/models.py#L102) | `saag.AnalysisResult` | Step 2 output — structural metrics only |
| [PredictionResult](saag/models.py#L180) | `saag.PredictionResult` | Step 3 (unified Predict) output — RMAV/GNN scores, anti-patterns, explanation |
| [ValidationResult](saag/models.py#L369) | `saag.ValidationResult` | Step 5 validation output including Spearman and gate metrics |
| [PrescribeResult](saag/prescription/models.py) | `saag.prescription.PrescribeResult` | Step 6 prescriptive output including compiled policy and SRI delta |

---

## Project Structure

```
.
├── cli/                        # CLI pipeline scripts
│   ├── run.py                  #   Master pipeline orchestrator (--all flag)
│   ├── generate_graph.py       #   Offline Prep: Generate — synthetic pub-sub topology
│   ├── import_graph.py         #   Step 1a: Model (Import) — Neo4j import & dependency derivation
│   ├── export_graph.py         #   Step 1b: Model (Export) — export graph from Neo4j to JSON
│   ├── analyze_graph.py        #   Step 2: Analyze — structural metrics only
│   ├── train_graph.py          #   Step 3a: Predict (Train) — GNN training (optional; requires Step 4 labels)
│   ├── predict_graph.py        #   Step 3b: Predict (Inference) — unified RMAV + GNN + anti-patterns
│   ├── detect_antipatterns.py  #   Standalone anti-pattern / CI gate
│   ├── simulate_graph.py       #   Step 4: Simulate — fault-inject | message-flow | combined
│   ├── validate_graph.py       #   Step 5: Validate — single | sweep | report | compare
│   ├── prescribe_graph.py      #   Step 6: Prescribe — optimize architecture & validate in closed loop
│   ├── visualize_graph.py      #   Step 7: Visualize — interactive HTML dashboard
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
│   ├── analysis/               #   Structural metrics only (Step 2)
│   ├── prediction/             #   Unified Predict step (Step 3): RMAV quality scoring + GNN service + anti-pattern detection + explanations
│   ├── simulation/             #   Four parallel failure/event simulators
│   ├── validation/             #   Per-dimension statistical validation
│   ├── prescription/           #   Stage 6 prescriptive optimization service
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

Released under an open-source license — see [`LICENSE`](LICENSE). Apache-2.0 is the suggested default; all dependencies are license-compatible.
