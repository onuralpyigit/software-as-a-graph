# Methodology Overview

This document provides a comprehensive explanation of the six-step methodology for graph-based modeling and analysis of distributed publish-subscribe systems.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Methodology Pipeline](#methodology-pipeline)
3. [Step 1: Graph Model Construction](#step-1-graph-model-construction)
4. [Step 2: Structural Analysis](#step-2-structural-analysis)
5. [Step 3: Criticality Scoring](#step-3-criticality-scoring)
6. [Step 4: Failure Simulation](#step-4-failure-simulation)
7. [Step 5: Statistical Validation](#step-5-statistical-validation)
8. [Step 6: Visualization](#step-6-visualization)
7. [End-to-End Example](#end-to-end-example)

---

## Introduction

### Research Motivation

Distributed publish-subscribe (pub-sub) systems are foundational to modern software architectures:

- **Autonomous Vehicles**: ROS 2 uses DDS pub-sub for sensor fusion and control
- **IoT Networks**: MQTT brokers connect thousands of sensors
- **Financial Trading**: Event-driven architectures process market data
- **Microservices**: Message queues decouple service dependencies

These systems present unique challenges:

| Challenge | Description |
|-----------|-------------|
| **Scale** | Hundreds of applications, thousands of topics |
| **Complexity** | Multi-layer dependencies across infrastructure |
| **Dynamism** | Systems evolve over time |
| **Opacity** | Dependencies are implicit in messaging patterns |

### Traditional Approaches and Their Limitations

| Approach | Limitation |
|----------|------------|
| Expert Intuition | Subjective, doesn't scale, knowledge silos |
| Runtime Monitoring | Reactive (not predictive), expensive |
| Code Analysis | Misses runtime dependencies |
| Manual Architecture Reviews | Time-consuming, incomplete |

### Our Approach: Graph-Based Analysis

We transform the pub-sub system into a **multi-layer graph** and apply **graph algorithms** to identify structural vulnerabilities. The key insight:

> **Topological metrics predict failure impact.** Components that are structurally central (high betweenness, articulation points) cause greater damage when they fail.

---

## Methodology Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SIX-STEP METHODOLOGY                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐                    │
│   │  STEP 1    │───▶│  STEP 2    │───▶│  STEP 3    │                    │
│   │   Graph    │    │ Structural │    │Criticality │                    │
│   │   Model    │    │  Analysis  │    │  Scoring   │                    │
│   └────────────┘    └────────────┘    └────────────┘                    │
│         │                                    │                           │
│         │         PREDICTION PATH            │                           │
│         ▼                                    ▼                           │
│   ┌─────────────────────────────────────────────────────┐               │
│   │              Predicted Criticality Scores            │               │
│   └─────────────────────────────────────────────────────┘               │
│                              │                                           │
│                              │ Compare                                   │
│                              ▼                                           │
│   ┌─────────────────────────────────────────────────────┐               │
│   │              STEP 5: Statistical Validation          │               │
│   │         Spearman ρ, F1-Score, Top-k Overlap          │               │
│   └─────────────────────────────────────────────────────┘               │
│                              ▲                                           │
│                              │ Compare                                   │
│                              │                                           │
│   ┌─────────────────────────────────────────────────────┐               │
│   │                Actual Impact Scores                  │               │
│   └─────────────────────────────────────────────────────┘               │
│         ▲                                    ▲                           │
│         │         VALIDATION PATH            │                           │
│         │                                    │                           │
│   ┌────────────┐                       ┌────────────┐                   │
│   │  STEP 4    │                       │  STEP 6    │                   │
│   │  Failure   │                       │Visualize   │                   │
│   │ Simulation │                       │            │                   │
│   └────────────┘                       └────────────┘                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Graph Model Construction

**Goal**: Transform the pub-sub system into a multi-layer graph representation.

### Multi-Layer Architecture

We model the system across four architectural layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MULTI-LAYER GRAPH                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Layer 4: Infrastructure    [N1]───────[N2]───────[N3]             │
│                               │         │         │                │
│  Layer 3: Broker            [B1]───────[B2]                        │
│                             /   \     /   \                        │
│  Layer 2: Topic          [T1]  [T2] [T3]  [T4]                     │
│                           │     │    │     │                       │
│  Layer 1: Application   [A1]  [A2] [A3]  [A4]  [A5]                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| Layer | Components | Description |
|-------|------------|-------------|
| **Application** | Publishers, Subscribers | Software components that produce/consume messages |
| **Topic** | Message Channels | Named endpoints for pub-sub communication |
| **Broker** | Message Routers | Infrastructure handling message distribution |
| **Infrastructure** | Nodes, Hosts | Physical/virtual machines running components |

### Vertex Types

| Type | Symbol | Properties | Description |
|------|--------|------------|-------------|
| Application | A | id, name, role | Publisher, subscriber, or both |
| Topic | T | id, name, qos, size | Message channel with QoS settings |
| Broker | B | id, name | Message routing infrastructure |
| Node | N | id, name | Physical/virtual host |

### Edge Types

| Type | Direction | Meaning |
|------|-----------|---------|
| PUBLISHES_TO | A → T | Application publishes messages to topic |
| SUBSCRIBES_TO | A → T | Application receives messages from topic |
| ROUTES | B → T | Broker handles routing for topic |
| RUNS_ON | A/B → N | Component executes on infrastructure node |
| CONNECTS_TO | N → N | Network connectivity between nodes |

### Derived Dependencies (DEPENDS_ON)

From basic relationships, we derive functional dependencies:

| Type | Derivation | Weight Factors |
|------|------------|----------------|
| app_to_app | A₁ publishes to T, A₂ subscribes to T | Topic count, QoS, message size |
| app_to_broker | A uses B for routing | Number of topics routed |
| app_to_node | A runs on N | Direct infrastructure dependency |
| node_to_node | N₁ connects to N₂ | Network path dependencies |

### Implementation

```python
from src.core import generate_graph
from src.simulation import SimulationGraph

# Generate a realistic pub-sub system
graph_data = generate_graph(
    scale="medium",      # System size preset
    scenario="iot",      # Domain-specific patterns
    seed=42              # Reproducibility
)

# Load into simulation model
graph = SimulationGraph.from_dict(graph_data)

print(f"Components: {len(graph.components)}")
print(f"Connections: {len(graph.connections)}")
print(f"Message Paths: {len(graph.get_message_paths())}")
```

**Output:**
```
Components: 77
Connections: 241
Message Paths: 156
```

---

## Step 2: Structural Analysis

**Goal**: Compute graph-theoretic metrics that characterize each component's structural position.

### Centrality Metrics

We compute multiple centrality metrics to capture different aspects of structural importance:

#### Betweenness Centrality

Measures how often a node lies on shortest paths between other nodes.

```
        σst(v)
CB(v) = Σ ──────
       s≠v≠t  σst
```

Where:
- σst = number of shortest paths from s to t
- σst(v) = number of those paths passing through v

**Interpretation**: High betweenness = bottleneck, potential single point of failure.

#### Degree Centrality

Measures the number of direct connections.

```
         deg(v)
CD(v) = ────────
         n - 1
```

**Interpretation**: High degree = highly coupled, many dependencies.

#### PageRank

Measures importance based on incoming links from other important nodes.

```
              1 - d       PR(u)
PR(v) = ────────── + d Σ ─────────
            n         u∈M(v) deg(u)
```

Where d is the damping factor (typically 0.85).

**Interpretation**: High PageRank = receives data from important sources.

#### Message Path Centrality (Custom)

Measures participation in actual message flow paths.

```
                |paths containing v|
MPC(v) = ────────────────────────────
              |all message paths|
```

**Interpretation**: High MPC = critical for message delivery.

### Articulation Points

An **articulation point** is a vertex whose removal disconnects the graph:

```
Before removal:          After removing B:

    [A]                      [A]
     │                        
    [B]  ← articulation     [C]    [D]
   /   \                   
 [C]   [D]                (disconnected!)
```

Articulation points are structurally critical regardless of other metrics.

### Implementation

```python
from src.validation import GraphAnalyzer

analyzer = GraphAnalyzer(graph)

# Individual metrics
betweenness = analyzer.betweenness_centrality()
degree = analyzer.degree_centrality()
pagerank = analyzer.pagerank()
message_path = analyzer.message_path_centrality()

# Articulation points
articulation_points = analyzer.articulation_points()

# All metrics combined
all_metrics = analyzer.analyze_all()
```

---

## Step 3: Criticality Scoring

**Goal**: Combine metrics into a single composite criticality score.

### Composite Score Formula

```
C_score(v) = α·CB_norm(v) + β·AP(v) + γ·DC_norm(v) + δ·PR_norm(v)
```

Where:
- **CB_norm(v)** ∈ [0,1]: Normalized betweenness centrality
- **AP(v)** ∈ {0,1}: 1 if v is an articulation point, 0 otherwise
- **DC_norm(v)** ∈ [0,1]: Normalized degree centrality
- **PR_norm(v)** ∈ [0,1]: Normalized PageRank

**Default weights**: α=0.35, β=0.25, γ=0.20, δ=0.20

### Normalization

All metrics are normalized to [0,1] using min-max scaling:

```
              x - min(X)
x_norm = ─────────────────
          max(X) - min(X)
```

### Box-Plot Classification

Instead of fixed thresholds, we use box-plot statistical classification:

| Level | Threshold | Interpretation |
|-------|-----------|----------------|
| **CRITICAL** | > Q3 + 1.5×IQR | Statistical outliers - immediate attention |
| **HIGH** | > Q3 | Top quartile - close monitoring |
| **MEDIUM** | > Median | Above average |
| **LOW** | > Q1 | Below average |
| **MINIMAL** | ≤ Q1 | Bottom quartile |

**Advantage**: Thresholds adapt to the score distribution of each system.

### Implementation

```python
from src.analysis import BoxPlotClassifier

# Compute composite scores
composite_scores = analyzer.composite_score(
    weights={"betweenness": 0.35, "degree": 0.20, "pagerank": 0.20}
)

# Classify using box-plot method
classifier = BoxPlotClassifier(k_factor=1.5)
items = [{"id": k, "type": "component", "score": v} 
         for k, v in composite_scores.items()]

result = classifier.classify(items, metric_name="composite")

# View classification summary
for level, components in result.by_level.items():
    print(f"{level}: {len(components)} components")
```

---

## Step 4: Failure Simulation

**Goal**: Measure actual system impact when each component fails.

### Simulation Approach

For each component v:
1. Remove v from the graph
2. Calculate reachability loss
3. Simulate cascade propagation
4. Compute impact score

### Impact Score Formula

```
Impact(v) = 0.5 × reachability_loss + 0.3 × fragmentation + 0.2 × cascade_extent
```

Where:
- **reachability_loss**: Fraction of message paths destroyed
- **fragmentation**: Increase in disconnected components
- **cascade_extent**: Number of components that fail due to cascade

### Cascade Propagation Model

When component A fails, dependent components may also fail:

```
[A] fails
 ↓
[B] depends on A (strength = 0.8)
 ↓ cascade probability check
[B] fails if strength > threshold AND random() < cascade_prob
 ↓
[C] depends on B
 ↓
...continues until max_depth or no more failures
```

### Implementation

```python
from src.simulation import FailureSimulator

simulator = FailureSimulator(
    cascade_threshold=0.5,    # Dependency strength to trigger cascade
    cascade_probability=0.7,  # Probability cascade actually occurs
    max_cascade_depth=5,      # Maximum cascade hops
    seed=42
)

# Single failure
result = simulator.simulate_failure(graph, "B1", enable_cascade=True)
print(f"Impact: {result.impact.impact_score:.4f}")
print(f"Cascade failures: {len(result.cascade_failures)}")

# Exhaustive campaign (all components)
batch = simulator.simulate_all_failures(graph, enable_cascade=True)
print(f"Most critical: {batch.critical_components[:5]}")
```

---

## Step 5: Statistical Validation

**Goal**: Compare predicted criticality scores against actual failure impacts.

### Validation Process

```
Predicted Scores          Actual Impacts
(from Step 3)             (from Step 4)
     │                         │
     └────────┬────────────────┘
              │
              ▼
     ┌─────────────────────┐
     │ Statistical Tests   │
     ├─────────────────────┤
     │ • Spearman ρ        │
     │ • F1-Score          │
     │ • Precision/Recall  │
     │ • Top-k Overlap     │
     └─────────────────────┘
              │
              ▼
     Validation Result
     (PASSED / PARTIAL / FAILED)
```

### Spearman Rank Correlation

Measures whether predicted ranking matches actual ranking:

- ρ = 1.0: Perfect agreement
- ρ = 0.7: Strong positive correlation ✓
- ρ = 0.0: No correlation
- ρ < 0: Inverse correlation

### F1-Score

Harmonic mean of precision and recall:

```
        2 × Precision × Recall
F1 = ─────────────────────────
        Precision + Recall
```

Using 80th percentile as threshold:
- **True Positive (TP)**: Predicted critical AND actually critical
- **False Positive (FP)**: Predicted critical BUT not actually
- **False Negative (FN)**: Not predicted BUT actually critical

### Top-k Overlap

Agreement on the most critical components:

```
                |Predicted Top-k ∩ Actual Top-k|
Top-k Overlap = ─────────────────────────────────
                            k
```

### Research Targets

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Spearman ρ | ≥ 0.70 | Strong ranking correlation |
| F1-Score | ≥ 0.90 | High classification accuracy |
| Precision | ≥ 0.80 | Few false positives |
| Recall | ≥ 0.80 | Few false negatives |
| Top-5 Overlap | ≥ 60% | Agree on most critical |

### Implementation

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

---

## Step 6: Visualization

**Goal**: Generate interactive visualizations for analysis and communication.

### Visualization Types

| Type | Purpose | Use Case |
|------|---------|----------|
| **Network Graph** | Interactive node-link diagram | Explore topology |
| **Multi-Layer View** | Vertical layer separation | Understand architecture |
| **Dashboard** | Charts + metrics + tables | Comprehensive analysis |
| **Criticality Heatmap** | Color-coded by score | Identify hotspots |

### Dashboard Components

1. **Summary Cards**: Key metrics at a glance
2. **Criticality Distribution**: Bar chart of level counts
3. **Score Histogram**: Distribution of composite scores
4. **Correlation Scatter**: Predicted vs actual plot
5. **Component Table**: Sortable, searchable details
6. **Interactive Graph**: vis.js network visualization

### Implementation

```python
from src.visualization import GraphRenderer, DashboardGenerator

# Network visualization
renderer = GraphRenderer()
html = renderer.render(graph, criticality_scores)

# Multi-layer view
html = renderer.render_multi_layer(graph, criticality_scores)

# Comprehensive dashboard
generator = DashboardGenerator()
html = generator.generate(
    graph,
    criticality=criticality_scores,
    validation=result.validation.to_dict(),
)

# Save to file
Path("dashboard.html").write_text(html)
```

---

## End-to-End Example

Complete pipeline in one script:

```python
#!/usr/bin/env python3
"""Complete six-step methodology demonstration."""

from pathlib import Path
from src.core import generate_graph
from src.simulation import SimulationGraph
from src.validation import ValidationPipeline
from src.visualization import DashboardGenerator

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Graph Model Construction
# ═══════════════════════════════════════════════════════════════════
print("Step 1: Generating graph model...")
data = generate_graph(scale="small", scenario="iot", seed=42)
graph = SimulationGraph.from_dict(data)
print(f"  Components: {len(graph.components)}")
print(f"  Connections: {len(graph.connections)}")

# ═══════════════════════════════════════════════════════════════════
# STEPS 2-5: Analysis, Simulation, Validation (via Pipeline)
# ═══════════════════════════════════════════════════════════════════
print("\nSteps 2-5: Running validation pipeline...")
pipeline = ValidationPipeline(seed=42)
result = pipeline.run(graph, analysis_method="composite")

print(f"\n  Analysis method: {result.analysis_method}")
print(f"  Spearman ρ: {result.validation.correlation.spearman:.4f}")
print(f"  F1-Score:   {result.validation.classification.f1:.4f}")
print(f"  Precision:  {result.validation.classification.precision:.4f}")
print(f"  Recall:     {result.validation.classification.recall:.4f}")
print(f"  Status:     {result.validation.status.value}")

# ═══════════════════════════════════════════════════════════════════
# STEP 6: Visualization
# ═══════════════════════════════════════════════════════════════════
print("\nStep 6: Generating dashboard...")
generator = DashboardGenerator()
html = generator.generate(
    graph,
    validation=result.validation.to_dict(),
)
Path("output/dashboard.html").write_text(html)
print("  Dashboard saved to output/dashboard.html")

print("\n✓ Pipeline complete!")
```

**Output:**
```
Step 1: Generating graph model...
  Components: 31
  Connections: 74

Steps 2-5: Running validation pipeline...

  Analysis method: composite
  Spearman ρ: 0.8081
  F1-Score:   0.8750
  Precision:  0.8750
  Recall:     0.8750
  Status:     partial

Step 6: Generating dashboard...
  Dashboard saved to output/dashboard.html

✓ Pipeline complete!
```

---

## Navigation

- **Previous:** [← Documentation Index](index.md)
- **Next:** [Graph Model →](graph-model.md)
