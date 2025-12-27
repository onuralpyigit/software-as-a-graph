# Software-as-a-Graph

## Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems

A methodology and toolkit for identifying critical components in distributed systems through graph-based structural analysis.

**Author**: Ibrahim Onuralp Yigit  
**Institution**: Istanbul Technical University  
**Publication**: IEEE RASSE 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [Research Motivation](#research-motivation)
3. [Six-Step Methodology](#six-step-methodology)
   - [Step 1: Graph Model Construction](#step-1-graph-model-construction)
   - [Step 2: Structural Analysis](#step-2-structural-analysis)
   - [Step 3: Criticality Scoring](#step-3-criticality-scoring)
   - [Step 4: Failure Simulation](#step-4-failure-simulation)
   - [Step 5: Statistical Validation](#step-5-statistical-validation)
   - [Step 6: Visualization](#step-6-visualization)
4. [Quick Start](#quick-start)
5. [Installation](#installation)
6. [CLI Reference](#cli-reference)
7. [Python API](#python-api)
8. [Validation Targets](#validation-targets)
9. [Project Structure](#project-structure)
10. [Examples](#examples)
11. [Testing](#testing)
12. [Publications](#publications)

---

## Introduction

Modern distributed systems—autonomous vehicles, IoT networks, financial trading platforms—rely on **publish-subscribe (pub-sub)** messaging for communication between components. These systems can have hundreds of applications, thousands of topics, and complex multi-layer dependencies that are difficult to analyze using traditional methods.

This project provides a **graph-based approach** to model these systems and identify critical components before failures occur. Instead of relying on qualitative expert judgment, we use **quantitative structural metrics** from graph theory to predict which components would cause the most damage if they failed.

### Key Insight

> **Topological structure predicts system behavior.** Components that sit on many shortest paths (high betweenness centrality) or whose removal disconnects the graph (articulation points) are structurally critical—and this structural criticality correlates with actual failure impact.

### What This Toolkit Provides

| Capability | Description |
|------------|-------------|
| **Graph Modeling** | Multi-layer graph representation of pub-sub systems |
| **Criticality Analysis** | Composite scoring using centrality metrics |
| **Failure Simulation** | Exhaustive campaign with cascade propagation |
| **Statistical Validation** | Rigorous comparison of predictions vs actuals |
| **Interactive Visualization** | Dashboards and multi-layer views |

---

## Research Motivation

### The Problem

Traditional approaches to identifying critical components rely on:
- **Expert intuition**: Subjective and doesn't scale
- **Runtime monitoring**: Reactive, not predictive
- **Code analysis**: Misses runtime dependencies

These methods fail to capture the **structural dependencies** that emerge in distributed systems.

### Our Approach

We model the system as a **multi-layer graph** and apply **graph algorithms** to identify structural vulnerabilities:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MULTI-LAYER GRAPH                          │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 4: Infrastructure    [N1]───────[N2]───────[N3]             │
│                               │         │         │                │
│  Layer 3: Broker            [B1]───────[B2]                        │
│                             /   \     /   \                        │
│  Layer 2: Topic          [T1]  [T2] [T3]  [T4]                     │
│                           │     │    │     │                       │
│  Layer 1: Application   [A1]  [A2] [A3]  [A4]  [A5]                │
└─────────────────────────────────────────────────────────────────────┘
```

### Research Questions

1. Can graph topology predict failure impact?
2. Which centrality metrics best identify critical components?
3. How accurately can we classify critical vs non-critical components?

### Validation Targets

| Metric | Target | Meaning |
|--------|--------|---------|
| **Spearman ρ** | ≥ 0.70 | Predicted ranking correlates with actual ranking |
| **F1-Score** | ≥ 0.90 | High accuracy classifying critical components |
| **Precision** | ≥ 0.80 | Few false positives (predicted critical but actually not) |
| **Recall** | ≥ 0.80 | Few false negatives (actually critical but not predicted) |
| **Top-5 Overlap** | ≥ 60% | Agreement on the most critical components |

---

## Six-Step Methodology

### Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   STEP 1     │────▶│   STEP 2     │────▶│   STEP 3     │
│ Graph Model  │     │  Structural  │     │ Criticality  │
│ Construction │     │  Analysis    │     │   Scoring    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│   STEP 6     │◀────│   STEP 5     │◀────│   STEP 4     │
│Visualization │     │  Statistical │     │   Failure    │
│              │     │  Validation  │     │  Simulation  │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

### Step 1: Graph Model Construction

**Goal**: Transform the pub-sub system into a multi-layer graph.

#### Graph Components

**Vertices (Nodes)**

| Type | Symbol | Description |
|------|--------|-------------|
| Application | A | Software component (publisher, subscriber, or both) |
| Topic | T | Message channel / communication endpoint |
| Broker | B | Message routing infrastructure |
| Node | N | Physical/virtual infrastructure host |

**Edges (Relationships)**

| Type | Direction | Meaning |
|------|-----------|---------|
| PUBLISHES_TO | A → T | Application publishes messages to topic |
| SUBSCRIBES_TO | A → T | Application subscribes to receive messages |
| ROUTES | B → T | Broker handles message routing for topic |
| RUNS_ON | A/B → N | Component executes on infrastructure node |
| CONNECTS_TO | N → N | Network connectivity between nodes |

#### Derived Dependencies

From the basic relationships, we derive **DEPENDS_ON** edges that capture functional dependencies:

| Type | Derivation | Meaning |
|------|------------|---------|
| app_to_app | A₁ publishes to T, A₂ subscribes to T | A₂ depends on A₁ for data |
| app_to_broker | A publishes/subscribes via B | A depends on B for routing |
| app_to_node | A runs on N | A depends on N for execution |

#### Example

```python
from src.core import generate_graph
from src.simulation import SimulationGraph

# Generate a realistic IoT system
graph_data = generate_graph(
    scale="medium",      # 30+ components
    scenario="iot",      # IoT domain patterns
    seed=42              # Reproducible
)

# Load into simulation model
graph = SimulationGraph.from_dict(graph_data)

print(f"Components: {len(graph.components)}")
print(f"Connections: {len(graph.connections)}")
```

**Output**:
```
Components: 77
Connections: 241
```

---

### Step 2: Structural Analysis

**Goal**: Compute graph-theoretic metrics that characterize each component's structural position.

#### Centrality Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Betweenness Centrality** | CB(v) = Σ σst(v)/σst | How many shortest paths pass through v |
| **Degree Centrality** | CD(v) = deg(v)/(n-1) | How connected v is |
| **PageRank** | PR(v) = (1-d)/n + d Σ PR(u)/deg(u) | Importance based on incoming links |

#### Articulation Points

An **articulation point** is a vertex whose removal disconnects the graph:

```
      [A]                    [A]
       │                      
      [B]  ← removal →      [C]    [D]
      / \                   
    [C] [D]              (disconnected!)
```

If B is removed, C and D become unreachable from A. B is an articulation point.

#### Implementation

```python
from src.validation import GraphAnalyzer

analyzer = GraphAnalyzer(graph)

# Individual metrics
betweenness = analyzer.betweenness_centrality()
degree = analyzer.degree_centrality()
pagerank = analyzer.pagerank()
message_path = analyzer.message_path_centrality()

# All metrics at once
all_metrics = analyzer.analyze_all()
```

---

### Step 3: Criticality Scoring

**Goal**: Combine metrics into a single **composite criticality score**.

#### Composite Score Formula

```
C_score(v) = α·CB_norm(v) + β·AP(v) + γ·DC_norm(v) + δ·PR_norm(v)
```

Where:
- **CB_norm(v)** ∈ [0,1]: Normalized betweenness centrality
- **AP(v)** ∈ {0,1}: 1 if v is an articulation point, 0 otherwise
- **DC_norm(v)** ∈ [0,1]: Normalized degree centrality
- **PR_norm(v)** ∈ [0,1]: Normalized PageRank

Default weights: α=0.35, β=0.25, γ=0.20, δ=0.20

#### Classification Levels

Using **box-plot statistical classification** (avoids arbitrary thresholds):

| Level | Threshold | Interpretation |
|-------|-----------|----------------|
| **CRITICAL** | > Q3 + 1.5×IQR | Upper outliers - immediate attention |
| **HIGH** | > Q3 | Top quartile - monitor closely |
| **MEDIUM** | > Median | Above average |
| **LOW** | > Q1 | Below average |
| **MINIMAL** | ≤ Q1 | Bottom quartile |

#### Implementation

```python
composite_scores = analyzer.composite_score(
    weights={"betweenness": 0.35, "degree": 0.20, "pagerank": 0.20}
)

# Classify using box-plot method
from src.analysis import BoxPlotClassifier

classifier = BoxPlotClassifier(k_factor=1.5)
items = [{"id": k, "type": "component", "score": v} 
         for k, v in composite_scores.items()]

result = classifier.classify(items, metric_name="composite")

print(f"Critical: {len(result.by_level['CRITICAL'])}")
print(f"High: {len(result.by_level['HIGH'])}")
```

---

### Step 4: Failure Simulation

**Goal**: Measure **actual impact** when each component fails.

#### Simulation Approach

For each component v in the graph:
1. Remove v from the graph
2. Calculate **reachability loss**: How many message paths are broken?
3. Simulate **cascade propagation**: Do dependent components also fail?
4. Compute **impact score**

#### Impact Score Formula

```
Impact(v) = 0.5 × reachability_loss + 0.3 × fragmentation + 0.2 × cascade_extent
```

Where:
- **reachability_loss**: Fraction of message paths destroyed
- **fragmentation**: Number of disconnected components created
- **cascade_extent**: Number of components that fail due to cascade

#### Cascade Propagation

When component A fails, components that depend on A may also fail:

```
[A] fails
 ↓
[B] (depends on A) → fails if dependency > threshold
 ↓
[C] (depends on B) → may cascade further
```

#### Implementation

```python
from src.simulation import FailureSimulator

simulator = FailureSimulator(
    cascade_threshold=0.5,    # Dependency strength to trigger cascade
    cascade_probability=0.7,  # Probability of cascade occurring
    max_cascade_depth=5,      # Maximum cascade hops
    seed=42
)

# Simulate single failure
result = simulator.simulate_failure(graph, "B1", enable_cascade=True)
print(f"Impact: {result.impact.impact_score:.4f}")
print(f"Cascade failures: {len(result.cascade_failures)}")

# Exhaustive campaign (all components)
batch = simulator.simulate_all_failures(graph, enable_cascade=True)
print(f"Most critical: {batch.critical_components[:5]}")
```

---

### Step 5: Statistical Validation

**Goal**: Compare **predicted criticality** (from Step 3) against **actual impact** (from Step 4).

#### Validation Process

```
Predicted Scores          Actual Impacts
(from topology)           (from simulation)
     ↓                          ↓
┌────────────────────────────────────────┐
│         Statistical Comparison          │
├─────────────────────────────────────────┤
│  • Spearman Rank Correlation           │
│  • Classification Metrics (F1, P, R)   │
│  • Top-k Overlap                        │
│  • Bootstrap Confidence Intervals       │
└─────────────────────────────────────────┘
     ↓
 Validation Result (PASSED / PARTIAL / FAILED)
```

#### Metrics Explained

**Spearman Rank Correlation (ρ)**

Measures whether the predicted **ranking** matches the actual **ranking**:
- ρ = 1.0: Perfect agreement
- ρ = 0.7: Strong positive correlation ✓
- ρ = 0.0: No correlation
- ρ < 0: Inverse correlation

**F1-Score**

Harmonic mean of precision and recall:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Using 80th percentile as threshold for "critical":
- **True Positive**: Predicted critical AND actually critical
- **False Positive**: Predicted critical BUT not actually critical
- **False Negative**: Not predicted critical BUT actually critical

**Top-k Overlap**

What fraction of predicted top-k appear in actual top-k?
```
Top-5 Overlap = |Predicted Top-5 ∩ Actual Top-5| / 5
```

#### Implementation

```python
from src.validation import ValidationPipeline

pipeline = ValidationPipeline(
    spearman_target=0.70,
    f1_target=0.90,
    seed=42
)

result = pipeline.run(graph, analysis_method="composite")

print(f"Spearman ρ: {result.validation.correlation.spearman:.4f}")
print(f"F1-Score:   {result.validation.classification.f1:.4f}")
print(f"Status:     {result.validation.status.value}")
```

**Output**:
```
Spearman ρ: 0.8081
F1-Score:   0.8750
Status:     partial
```

---

### Step 6: Visualization

**Goal**: Generate interactive visualizations for analysis and presentation.

#### Visualization Types

| Type | Description | Use Case |
|------|-------------|----------|
| **Network Graph** | Interactive node-link diagram | Explore topology |
| **Multi-Layer View** | Vertical layer separation | Understand architecture |
| **Criticality Heatmap** | Color-coded by score | Identify hotspots |
| **Dashboard** | Charts + metrics + tables | Comprehensive analysis |

#### Implementation

```python
from src.visualization import GraphRenderer, DashboardGenerator

# Prepare criticality data
criticality = {
    comp_id: {"score": score, "level": level}
    for comp_id, (score, level) in classification_results.items()
}

# Network visualization
renderer = GraphRenderer()
html = renderer.render(graph, criticality)
Path("network.html").write_text(html)

# Multi-layer view
html = renderer.render_multi_layer(graph, criticality)
Path("multi_layer.html").write_text(html)

# Comprehensive dashboard
generator = DashboardGenerator()
html = generator.generate(
    graph,
    criticality=criticality,
    validation=result.validation.to_dict(),
    simulation=batch.to_dict(),
)
Path("dashboard.html").write_text(html)
```

---

## Quick Start

### Option 1: End-to-End Pipeline (Recommended)

```bash
# Run complete pipeline
python run.py --quick

# View results
open output/dashboard.html
```

### Option 2: Step-by-Step CLI

```bash
# Step 1: Generate graph
python generate_graph.py --scale small --scenario iot --output graph.json

# Steps 2-4: Validate (includes analysis and simulation)
python validate_graph.py --input graph.json --output results.json

# Step 6: Visualize
python visualize_graph.py --input graph.json --dashboard --output dashboard.html
```

### Option 3: Python API

```python
from src.core import generate_graph
from src.simulation import SimulationGraph
from src.validation import ValidationPipeline
from src.visualization import DashboardGenerator

# Generate and load
data = generate_graph(scale="small", scenario="iot", seed=42)
graph = SimulationGraph.from_dict(data)

# Analyze and validate
pipeline = ValidationPipeline(seed=42)
result = pipeline.run(graph)

# Visualize
generator = DashboardGenerator()
html = generator.generate(graph, validation=result.validation.to_dict())
Path("dashboard.html").write_text(html)

print(f"Spearman: {result.validation.correlation.spearman:.4f}")
```

---

## Installation

### Requirements

- Python 3.9+
- NetworkX (graph algorithms)
- Neo4j (optional, for GDS-based analysis)

### Install

```bash
# Clone repository
git clone https://github.com/onuralpyigit/software-as-a-graph.git
cd software-as-a-graph

# Install dependencies
pip install networkx

# Optional: Neo4j integration
pip install neo4j
```

### Verify Installation

```bash
python run.py --quick
# Should complete with validation results
```

---

## CLI Reference

### run.py - End-to-End Pipeline

```bash
python run.py [OPTIONS]

Options:
  --scenario {iot,financial,healthcare,smart_city}  Domain scenario
  --scale {small,medium,large}                      System scale
  --quick                                           Quick demo (small scale)
  --skip-generate                                   Use existing graph
  --skip-validate                                   Skip validation
  --skip-visualize                                  Skip visualization
  --input FILE                                      Input graph JSON
  --output DIR                                      Output directory
  --spearman-target FLOAT                           Spearman target (default: 0.70)
  --f1-target FLOAT                                 F1 target (default: 0.90)
```

### generate_graph.py - Graph Generation

```bash
python generate_graph.py [OPTIONS]

Options:
  --scale {tiny,small,medium,large,xlarge}  Scale preset
  --scenario {iot,financial,healthcare,...}  Domain scenario
  --seed INT                                 Random seed
  --antipatterns [god_topic,spof,...]        Inject anti-patterns
  --output FILE                              Output JSON file
```

### validate_graph.py - Validation

```bash
python validate_graph.py [OPTIONS]

Options:
  --input FILE            Input graph JSON (required)
  --output FILE           Output results JSON
  --spearman FLOAT        Spearman target
  --f1 FLOAT              F1-score target
  --method {composite,betweenness,degree,pagerank}  Analysis method
```

### visualize_graph.py - Visualization

```bash
python visualize_graph.py [OPTIONS]

Options:
  --input FILE            Input graph JSON (required)
  --output FILE           Output HTML file
  --dashboard             Generate comprehensive dashboard
  --multi-layer           Generate multi-layer view
  --run-analysis          Compute criticality scores
```

---

## Python API

### Core Module

```python
from src.core import generate_graph, GraphModel

# Generate graph data
data = generate_graph(scale="medium", scenario="iot", seed=42)

# Work with model
model = GraphModel.from_dict(data)
print(model.summary())
```

### Simulation Module

```python
from src.simulation import SimulationGraph, FailureSimulator, EventSimulator

# Load graph
graph = SimulationGraph.from_dict(data)

# Failure simulation
simulator = FailureSimulator(seed=42)
result = simulator.simulate_failure(graph, "B1")
batch = simulator.simulate_all_failures(graph)

# Event simulation
event_sim = EventSimulator(seed=42)
metrics = event_sim.simulate(graph, duration_ms=10000, message_rate=100)
```

### Validation Module

```python
from src.validation import ValidationPipeline, GraphAnalyzer, Validator

# Integrated pipeline
pipeline = ValidationPipeline(seed=42)
result = pipeline.run(graph)

# Manual analysis
analyzer = GraphAnalyzer(graph)
scores = analyzer.composite_score()

# Manual validation
validator = Validator()
validation = validator.validate(predicted_scores, actual_impacts)
```

### Visualization Module

```python
from src.visualization import GraphRenderer, DashboardGenerator

# Graph visualization
renderer = GraphRenderer()
html = renderer.render(graph, criticality)
html = renderer.render_multi_layer(graph, criticality)

# Dashboard
generator = DashboardGenerator()
html = generator.generate(graph, criticality=scores, validation=results)
```

### Analysis Module

```python
from src.analysis import BoxPlotClassifier, CriticalityLevel

# Box-plot classification
classifier = BoxPlotClassifier(k_factor=1.5)
result = classifier.classify(items, metric_name="composite")

# Access results
for item in result.items:
    print(f"{item.id}: {item.score:.4f} ({item.level.value})")
```

---

## Validation Targets

| Metric | Target | Achieved (typical) | Status |
|--------|--------|-------------------|--------|
| Spearman ρ | ≥ 0.70 | 0.75 - 0.85 | ✅ |
| F1-Score | ≥ 0.90 | 0.85 - 0.95 | ✅/⚠️ |
| Precision | ≥ 0.80 | 0.85 - 0.95 | ✅ |
| Recall | ≥ 0.80 | 0.80 - 0.90 | ✅ |
| Top-5 Overlap | ≥ 60% | 60% - 80% | ✅ |

### Interpretation

| Status | Meaning | Action |
|--------|---------|--------|
| **PASSED** | All targets met | Predictions reliable |
| **PARTIAL** | Most targets met | Use with verification |
| **FAILED** | Few targets met | Review methodology |

---

## Project Structure

```
software-as-a-graph/
├── run.py                 # End-to-end pipeline
├── run_tests.py           # Test runner
├── generate_graph.py      # Graph generation CLI
├── validate_graph.py      # Validation CLI
├── visualize_graph.py     # Visualization CLI
├── simulate_graph.py      # Simulation CLI
├── import_graph.py        # Neo4j import CLI
├── analyze_graph.py       # Analysis CLI (Neo4j)
│
├── src/
│   ├── core/              # Graph model and generation
│   ├── simulation/        # Failure and event simulation
│   ├── validation/        # Statistical validation
│   ├── analysis/          # Centrality and classification
│   └── visualization/     # Rendering and dashboards
│
├── tests/                 # Test suite
│   ├── test_core.py
│   ├── test_simulation.py
│   ├── test_validation.py
│   ├── test_analysis.py
│   ├── test_visualization.py
│   └── test_integration.py
│
└── examples/              # Example scripts
    ├── quick_start.py
    ├── cli_reference.py
    └── run_all.py
```

---

## Examples

```bash
# Quick start demo
python examples/quick_start.py

# CLI reference
python examples/cli_reference.py

# Run all examples
python examples/run_all.py
```

---

## Testing

```bash
# Run all tests
python run_tests.py

# Quick tests only
python run_tests.py --quick

# Specific module
python run_tests.py --module validation
```

---

## Publications

### Primary Publication

**Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems**  
Ibrahim Onuralp Yigit  
IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE) 2025

### Citation

```bibtex
@inproceedings{yigit2025graph,
  title={Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems},
  author={Yigit, Ibrahim Onuralp},
  booktitle={IEEE RASSE},
  year={2025}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- Supervisor: Doç. Dr. Feza Buzluca, Istanbul Technical University
- Neo4j Graph Data Science team
- NetworkX, vis.js, and Chart.js communities