# Software-as-a-Graph

**Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-green.svg)](https://neo4j.com/)
[![IEEE RASSE 2025](https://img.shields.io/badge/IEEE-RASSE%202025-orange.svg)](#citation)

---

## What This Project Does

This framework predicts **which components in a distributed system are most critical** — meaning their failure would cause the greatest impact — using only the system's architectural structure.

**The key insight**: Graph topological metrics computed on system architecture can reliably predict component criticality *before* deployment, without expensive runtime monitoring or actual failures.

**Validation Results**:
- **Spearman Correlation**: **0.85** (Application Layer, Large Scale)
- **Classification Accuracy**: **0.83** F1-score for critical components
- **Performance**: Analysis is **2.2x faster** than discrete-event simulation
- **Scalability**: Verified on **XLarge systems (500+ components)**

---

## Table of Contents

1. [The Problem](#the-problem)
2. [The Solution: Six-Step Methodology](#the-solution-six-step-methodology)
3. [Graph Model](#graph-model)
4. [Quality Scoring (R, M, A)](#quality-scoring-r-m-a)
5. [Quick Start](#quick-start)
6. [CLI Reference](#cli-reference)
7. [Validation](#validation)
8. [Project Structure](#project-structure)
9. [Citation](#citation)

---

## The Problem

In distributed publish-subscribe systems (ROS 2, Kafka, MQTT, microservices), identifying critical components is challenging:

| Traditional Approach | Limitation |
|---------------------|------------|
| Expert judgment | Subjective, doesn't scale |
| Runtime monitoring | Reactive — requires actual failures |
| Load testing | Expensive, incomplete coverage |
| Code analysis | Misses architectural dependencies |

**Our approach**: Model the system as a graph and use topological metrics to *predict* criticality *before* deployment.

---

## The Solution: Six-Step Methodology

```
┌─────────────────────────────────────────────────────────────────┐
│                    SOFTWARE-AS-A-GRAPH PIPELINE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Step 1: GRAPH MODEL CONSTRUCTION                              │
│           Transform system topology → directed weighted graph   │
│                           ↓                                     │
│   Step 2: STRUCTURAL ANALYSIS                                   │
│           Compute centrality metrics (PageRank, Betweenness)    │
│                           ↓                                     │
│   Step 3: QUALITY SCORING                                       │
│           Calculate R(v), M(v), A(v) → Q(v) criticality score   │
│                           ↓                                     │
│   Step 4: FAILURE SIMULATION                                    │
│           Test each component's actual failure impact I(v)      │
│                           ↓                                     │
│   Step 5: VALIDATION                                            │
│           Compare predicted Q(v) vs actual I(v)                 │
│                           ↓                                     │
│   Step 6: VISUALIZATION                                         │
│           Generate interactive dashboard                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: Graph Model Construction

Transform your distributed system into a formal graph:

**Components become vertices:**
- **Applications** — services that publish/subscribe to topics
- **Topics** — message channels with QoS policies
- **Brokers** — middleware routing infrastructure
- **Nodes** — physical/virtual hosts

**Relationships become edges:**
- `PUBLISHES_TO` / `SUBSCRIBES_TO` — data flow
- `ROUTES` — broker responsibility
- `RUNS_ON` — hosting relationship
- `DEPENDS_ON` — derived logical dependencies

```
              ┌─────────┐
              │ Node-1  │  Infrastructure Layer
              └────┬────┘
                   │ RUNS_ON
              ┌────▼────┐
              │  App-A  │  Application Layer
              └────┬────┘
                   │ PUBLISHES_TO
              ┌────▼────┐
              │ Topic-1 │  Topic Layer
              └─────────┘
```

### Step 2: Structural Analysis

Compute topological metrics that capture different aspects of criticality:

| Metric | What It Measures | High Value Means |
|--------|------------------|------------------|
| **PageRank** | Transitive influence | Many important components depend on this |
| **Reverse PageRank** | Failure propagation | Failure cascades to many downstream |
| **Betweenness** | Communication bottleneck | Lies on many dependency paths |
| **Clustering** | Local modularity | Well-connected neighborhood |
| **Articulation Point** | SPOF indicator | Removal disconnects the graph |

### Step 3: Quality Scoring

Combine metrics into three composite scores:

**Reliability R(v)** — Fault propagation risk:
```
R(v) = 0.45·PageRank + 0.35·ReversePageRank + 0.20·InDegree
```

**Maintainability M(v)** — Change propagation risk:
```
M(v) = 0.45·Betweenness + 0.25·(1-Clustering) + 0.30·Degree
```

**Availability A(v)** — Single point of failure risk:
```
A(v) = 0.50·ArticulationPoint + 0.25·BridgeRatio + 0.25·Criticality
```

**Overall Quality Q(v)**:
```
Q(v) = 0.35·R(v) + 0.30·M(v) + 0.35·A(v)
```

Components are classified using **box-plot statistics** (adaptive thresholds):
- **Critical**: Q > Q3 + 1.5×IQR
- **High**: Q > Q3
- **Medium**: Q > Median
- **Low**: Q ≤ Q1

### Step 4: Failure Simulation

Validate predictions by simulating actual failures:

```python
For each component v:
    1. Remove v from graph
    2. Measure: disconnected components, unreachable paths, weight loss
    3. Compute impact score I(v)
    4. Restore v
```

### Step 5: Validation

The framework was rigorously validated against a ground-truth simulator across multiple scales (Small to XLarge) and domains.

### Latest Benchmark Results (Jan 2026)

| Metric | Application Layer | Infrastructure Layer | Notes |
| :--- | :--- | :--- | :--- |
| **Spearman $\rho$** | **0.85** (Strong) | 0.54 (Moderate) | Model excels at logical software dependencies. |
| **F1-Score** | **0.83** | 0.68 | High precision in identifying critical apps. |
| **Top-5 Overlap** | **62%** | 40% | Successfully finds the most critical nodes. |
| **Speedup** | **2.2x** | 1.2x | Significantly faster than simulation. |

> **Key Insight**: The topological metrics are highly effective for the **Application Layer**, correctly identifying critical software components. Infrastructure prediction is currently limited by physical redundancy patterns not fully captured in the graph.

### Step 6: Visualization

Generate interactive HTML dashboards showing:
- Network topology with criticality coloring
- Score distributions
- Problem detection results
- Validation metrics

---

## Graph Model

### Formal Definition

The system is modeled as a directed weighted graph:

**G = (V, E, τ, w)** where:
- V = vertices (components)
- E = directed edges (relationships)
- τ = type functions for vertices/edges
- w = weight function

### Weight Calculation

All weights flow from **Topic QoS policies**:

```
W_topic = S_reliability + S_durability + S_priority + S_size
```

| QoS Setting | Score |
|-------------|-------|
| RELIABLE | +0.30 |
| PERSISTENT | +0.40 |
| URGENT priority | +0.30 |
| Large message (>64KB) | +0.60 |

Weights propagate upward:
```
Topics → Edges → Applications/Brokers → Nodes → Dependencies
```

### Dependency Derivation

Logical dependencies are automatically derived:

**App-to-App**: If App-A subscribes to Topic-X and App-B publishes to Topic-X:
```
App-A ──DEPENDS_ON──▶ App-B
```

**Node-to-Node**: If any app on Node-1 depends on any app on Node-2:
```
Node-1 ──DEPENDS_ON──▶ Node-2
```

---

## Quality Scoring (R, M, A)

### Why Three Dimensions?

| Score | Focus | Key Question |
|-------|-------|--------------|
| **R** (Reliability) | Fault propagation | What happens if this fails? |
| **M** (Maintainability) | Coupling | How hard is it to change this? |
| **A** (Availability) | SPOF risk | Is this irreplaceable? |

### Interpretation Guide

| Pattern | R | M | A | Interpretation |
|---------|---|---|---|----------------|
| **Hub** | High | High | High | Critical integration point |
| **Bottleneck** | Low | High | Med | Coupling problem — refactor |
| **SPOF** | Med | Low | High | Add redundancy |
| **Leaf** | Low | Low | Low | Low concern |

### Classification Thresholds

Box-plot adaptive classification avoids arbitrary cutoffs:

```
CRITICAL ────── score > Q3 + 1.5×IQR (outliers)
HIGH ────────── score > Q3 (top quartile)
MEDIUM ──────── score > Median
LOW ─────────── score > Q1
MINIMAL ─────── score ≤ Q1
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Neo4j 5.x (Docker recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/software-as-a-graph.git
cd software-as-a-graph

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Start Neo4j

```bash
docker-compose up -d neo4j
```

### Run the Pipeline

```bash
# Option 1: Full pipeline
python run.py --all --layer system

# Option 2: Step by step
python generate_graph.py --scale medium --output data/system.json
python import_graph.py --input data/system.json --clear
python analyze_graph.py --layer system
python simulate_graph.py --exhaustive --layer system
python validate_graph.py --layer system
python visualize_graph.py --layer system --output dashboard.html --open
```

### Example Output

```
══════════════════════════════════════════════════════════════════════
                        VALIDATION RESULTS
══════════════════════════════════════════════════════════════════════

  Layer: system
  Components: 48
  
  Spearman ρ:  0.876  ✓ (target: ≥0.70)
  F1 Score:    0.943  ✓ (target: ≥0.80)
  Precision:   0.952  ✓
  Recall:      0.935  ✓
  
  Top-5 Critical Components:
    1. sensor_fusion     (Q=0.892) CRITICAL
    2. main_broker       (Q=0.856) CRITICAL
    3. planning_node     (Q=0.789) HIGH
    4. gateway           (Q=0.734) HIGH
    5. control_app       (Q=0.678) MEDIUM
  
  Status: ✓ VALIDATION PASSED
```

---

## CLI Reference

### Generate Graph Data

```bash
python generate_graph.py --scale {tiny,small,medium,large,xlarge} --output FILE
```

### Import to Neo4j

```bash
python import_graph.py --input FILE [--clear] [--uri bolt://localhost:7687]
```

### Analyze Graph

```bash
python analyze_graph.py --layer {app,infra,mw-app,mw-infra,system} [--all]
```

**Layers:**
| Layer | Components | Dependencies |
|-------|------------|--------------|
| `app` | Applications | app_to_app |
| `infra` | Nodes | node_to_node |
| `mw-app` | Apps + Brokers | app_to_broker |
| `mw-infra` | Nodes + Brokers | node_to_broker |
| `system` | All | All |

### Simulate Failures

```bash
# Single component
python simulate_graph.py --failure COMPONENT_ID

# Exhaustive (all components)
python simulate_graph.py --exhaustive --layer system

# Generate report
python simulate_graph.py --report --layers app,infra,system
```

### Validate Predictions

```bash
python validate_graph.py --layer system [--spearman 0.70] [--f1 0.80]
```

### Generate Dashboard

```bash
python visualize_graph.py --layers app,infra,system --output dashboard.html --open
```

### Run Benchmarks

```bash
python benchmark.py --scales small,medium,large --layers app,system --runs 5
```

---

## Project Structure

```
software-as-a-graph/
├── src/
│   ├── core/                  # Graph model, import/export
│   ├── analysis/              # Structural metrics, quality scoring
│   ├── simulation/            # Failure simulation
│   ├── validation/            # Statistical validation
│   └── visualization/         # Dashboard generation
├── scripts/                   # CLI entry points (if separated)
├── docs/
│   ├── graph-model.md         # Formal graph definition
│   ├── weight-calculations.md # Weight formulas
│   └── quality-formulations.md # R, M, A score details
├── generate_graph.py          # Generate synthetic data
├── import_graph.py            # Import to Neo4j
├── analyze_graph.py           # Run analysis
├── simulate_graph.py          # Failure simulation
├── validate_graph.py          # Statistical validation
├── visualize_graph.py         # Generate dashboards
├── run.py                     # Pipeline orchestrator
└── benchmark.py               # Benchmark suite
```

---

## Input Format

System topology is defined in JSON:

```json
{
  "nodes": [
    {"id": "N0", "name": "Server-1"}
  ],
  "brokers": [
    {"id": "B0", "name": "MainBroker"}
  ],
  "topics": [
    {
      "id": "T0",
      "name": "/sensors/temperature",
      "size": 256,
      "qos": {
        "durability": "PERSISTENT",
        "reliability": "RELIABLE",
        "transport_priority": "HIGH"
      }
    }
  ],
  "applications": [
    {"id": "A0", "name": "TempSensor", "role": "pub"},
    {"id": "A1", "name": "TempController", "role": "sub"}
  ],
  "relationships": {
    "runs_on": [{"from": "A0", "to": "N0"}],
    "routes": [{"from": "B0", "to": "T0"}],
    "publishes_to": [{"from": "A0", "to": "T0"}],
    "subscribes_to": [{"from": "A1", "to": "T0"}]
  }
}
```

---

## Domain Mapping

The framework supports multiple pub-sub middleware:

| Graph Concept | ROS 2 | Kafka | MQTT |
|---------------|-------|-------|------|
| Application | ROS Node | Producer/Consumer | Client |
| Topic | ROS Topic | Kafka Topic | MQTT Topic |
| Broker | DDS Domain | Kafka Broker | MQTT Broker |
| Node | Host/Container | Pod/VM | Server |

---

## Citation

```bibtex
@INPROCEEDINGS{11315354,
  author={Yigit, Ibrahim Onuralp and Buzluca, Feza},
  booktitle={2025 IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE)}, 
  title={A Graph-Based Dependency Analysis Method for Identifying Critical Components in Distributed Publish-Subscribe Systems}, 
  year={2025},
  volume={},
  number={},
  pages={1-9},
  keywords={Fault tolerance;Architecture;Fault tolerant systems;Microservice architectures;Publish-subscribe;Computer architecture;Solids;Loss measurement;Complexity theory;Topology;Distributed Systems;Publish-Subscribe Architecture;Graph Modeling;Dependency Analysis;Critical Components;Failure Impact},
  doi={10.1109/RASSE64831.2025.11315354}}

```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---