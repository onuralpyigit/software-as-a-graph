# Step 2: Structural Analysis

**Compute topological metrics and quality scores to identify critical components**

This document covers the second step of the Software-as-a-Graph methodology: analyzing the graph model to compute structural metrics, calculate RMA quality scores (Reliability, Maintainability, Availability), classify components by criticality, and detect architectural problems.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Topological Metrics](#2-topological-metrics)
3. [Quality Scoring (RMA)](#3-quality-scoring-rma)
4. [Box-Plot Classification](#4-box-plot-classification)
5. [Edge Criticality](#5-edge-criticality)
6. [Using analyze_graph.py](#6-using-analyze_graphpy)
7. [Understanding Output](#7-understanding-output)
8. [Multi-Layer Analysis](#8-multi-layer-analysis)
9. [Problem Detection](#9-problem-detection)
10. [Worked Example](#10-worked-example)

---

## 1. Overview

### What This Step Does

Structural Analysis transforms the graph model into actionable criticality assessments:

```
┌─────────────────────┐          ┌─────────────────────┐
│  Graph Model        │          │  Analysis Results   │
│  (from Step 1)      │    →     │                     │
│                     │          │  - R, M, A scores   │
│  - Vertices         │          │  - Q(v) criticality │
│  - DEPENDS_ON edges │          │  - Classification   │
│  - Weights          │          │  - Problem alerts   │
└─────────────────────┘          └─────────────────────┘
```

### Analysis Pipeline

```
1. Extract Layer Subgraph
   └── Filter by dependency type (app_to_app, node_to_node, etc.)

2. Compute Topological Metrics
   ├── PageRank (transitive influence)
   ├── Reverse PageRank (failure propagation)
   ├── Betweenness Centrality (bottlenecks)
   ├── Clustering Coefficient (modularity)
   ├── Degree Centrality (interface complexity)
   ├── Articulation Points (SPOFs)
   └── Bridge Edges (critical paths)

3. Normalize Metrics
   └── Min-max scaling to [0, 1]

4. Calculate Quality Scores
   ├── R(v) - Reliability score
   ├── M(v) - Maintainability score
   ├── A(v) - Availability score
   └── Q(v) - Overall criticality

5. Classify Components
   └── Box-plot statistical classification

6. Detect Problems
   └── Identify architectural anti-patterns
```

### Quick Start

```bash
# Analyze complete system
python analyze_graph.py --layer system

# Analyze all layers
python analyze_graph.py --all

# Export results to JSON
python analyze_graph.py --layer system --output results/analysis.json
```

---

## 2. Topological Metrics

The analysis computes seven core metrics that capture different aspects of component criticality.

### Metrics Summary

| Metric | Measures | High Value Indicates |
|--------|----------|---------------------|
| **PageRank** | Transitive influence | Important dependency target |
| **Reverse PageRank** | Failure propagation | Wide downstream impact |
| **Betweenness** | Path centrality | Communication bottleneck |
| **Clustering** | Local modularity | Well-encapsulated component |
| **Degree** | Connection count | Complex interface |
| **Articulation Point** | Graph connectivity | Single point of failure |
| **Bridge Ratio** | Critical edges | Irreplaceable connections |

### PageRank

Measures transitive importance based on incoming dependencies.

```
PR(v) = (1-d)/N + d × Σ PR(u)/L(u)
        for all u pointing to v
```

- **d** = damping factor (default: 0.85)
- **N** = total nodes
- **L(u)** = out-degree of u

**Interpretation**: A component is important if important components depend on it. High PageRank means this is a critical upstream provider.

### Reverse PageRank

PageRank computed on the graph with edge directions reversed.

```
RP(v) = PageRank on G with all edges (u,v) → (v,u)
```

**Interpretation**: Measures failure propagation potential. High Reverse PageRank means if this component fails, many important downstream components are affected.

### Betweenness Centrality

Measures how often a node lies on shortest paths between other nodes.

```
BC(v) = Σ σ_st(v) / σ_st
        for all s ≠ v ≠ t
```

- **σ_st** = number of shortest paths from s to t
- **σ_st(v)** = number of those paths through v

**Interpretation**: High betweenness indicates a bottleneck. Changes to this component affect many communication paths.

### Clustering Coefficient

Measures how interconnected a node's neighbors are.

```
CC(v) = 2 × edges_between_neighbors / (k × (k-1))
```

- **k** = degree of v

**Interpretation**: High clustering means neighbors form cohesive groups (good modularity). Low clustering suggests poor encapsulation.

### Degree Centrality

Normalized count of connections.

```
DC(v) = degree(v) / (N - 1)
```

**Interpretation**: High degree means many interfaces—more integration points to manage during changes.

### Articulation Points

Binary indicator: does removing this node disconnect the graph?

```
AP(v) = 1  if removing v increases connected components
AP(v) = 0  otherwise
```

**Interpretation**: Articulation points are structural single points of failure (SPOFs). Their removal breaks system connectivity.

### Bridge Edges & Bridge Ratio

A bridge is an edge whose removal disconnects the graph.

```
Bridge Ratio = (bridge edges incident to v) / degree(v)
```

**Interpretation**: High bridge ratio means most of a component's connections are irreplaceable.

---

## 3. Quality Scoring (RMA)

Metrics are combined into three orthogonal quality dimensions plus an overall score.

### Score Overview

| Score | Question Answered | Risk Indicator |
|-------|-------------------|----------------|
| **R(v)** Reliability | What happens when this fails? | Failure cascade potential |
| **M(v)** Maintainability | How hard is this to change? | Change propagation risk |
| **A(v)** Availability | Is this a single point of failure? | Service continuity risk |
| **Q(v)** Overall | How critical is this component? | Combined priority |

### Reliability Score R(v)

**Purpose**: Measures fault propagation and system-wide failure impact.

```
R(v) = 0.45 × PR_norm + 0.35 × RP_norm + 0.20 × InDegree_norm
```

| Component | Weight | Why |
|-----------|--------|-----|
| PageRank | 0.45 | Transitive influence—depended on by important components |
| Reverse PageRank | 0.35 | Failure propagation—cascade downstream |
| In-Degree | 0.20 | Direct dependents—immediate impact breadth |

**Interpretation**: Higher R(v) → Higher reliability risk if component fails.

### Maintainability Score M(v)

**Purpose**: Measures coupling, complexity, and change propagation risk.

```
M(v) = 0.45 × BT_norm + 0.25 × (1 - CC_norm) + 0.30 × DC_norm
```

| Component | Weight | Why |
|-----------|--------|-----|
| Betweenness | 0.45 | Bottleneck—changes affect many paths |
| (1 - Clustering) | 0.25 | Poor modularity—changes propagate widely |
| Degree | 0.30 | Interface complexity—more integration points |

**Note**: Clustering is inverted because *low* clustering indicates *poor* maintainability.

**Interpretation**: Higher M(v) → Harder to maintain, higher change propagation risk.

### Availability Score A(v)

**Purpose**: Measures single point of failure (SPOF) risk.

```
A(v) = 0.50 × AP + 0.25 × BR + 0.25 × CR_norm

where CR = PR_norm × DC_norm  (criticality = influence × connectivity)
```

| Component | Weight | Why |
|-----------|--------|-----|
| Articulation Point | 0.50 | Structural SPOF—removal disconnects graph |
| Bridge Ratio | 0.25 | Irreplaceable connections |
| Criticality | 0.25 | Important hub characteristics |

**Interpretation**: Higher A(v) → Higher availability risk (more likely SPOF).

### Overall Quality Score Q(v)

**Purpose**: Single criticality measure combining all dimensions.

```
Q(v) = 0.35 × R(v) + 0.30 × M(v) + 0.35 × A(v)
```

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Reliability | 0.35 | Runtime stability |
| Maintainability | 0.30 | Development effort (slightly lower) |
| Availability | 0.35 | Runtime stability |

**Interpretation**: Higher Q(v) → Higher overall criticality requiring attention.

### Weight Customization

Weights can be adjusted for domain priorities:

```bash
# Safety-critical system: increase reliability weight
python analyze_graph.py --layer system \
    --weights '{"overall": {"reliability": 0.50, "maintainability": 0.20, "availability": 0.30}}'
```

### Default Weights Reference

```python
WEIGHTS = {
    "reliability": {
        "pagerank": 0.45,
        "reverse_pagerank": 0.35,
        "in_degree": 0.20
    },
    "maintainability": {
        "betweenness": 0.45,
        "clustering": 0.25,  # inverted in formula
        "degree": 0.30
    },
    "availability": {
        "articulation_point": 0.50,
        "bridge_ratio": 0.25,
        "criticality": 0.25
    },
    "overall": {
        "reliability": 0.35,
        "maintainability": 0.30,
        "availability": 0.35
    }
}
```

---

## 4. Box-Plot Classification

Instead of arbitrary fixed thresholds, components are classified using statistical box-plot boundaries.

### Classification Levels

```
                         Score Distribution
                                │
    ┌───────────────────────────┼───────────────────────────┐
    │                           │                           │
    │  ┌─────────┐         ┌────┴────┐         ┌─────────┐ │
    │  │ MINIMAL │─────────│   BOX   │─────────│CRITICAL │ │
    │  │  ≤ Q1   │   LOW   │ Q1 - Q3 │  HIGH   │ outlier │ │
    │  └─────────┘  Q1-Med └────┬────┘ Q3-upper└─────────┘ │
    │                      Med-Q3                          │
    │                      MEDIUM                          │
    └──────────────────────────────────────────────────────┘

    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    Upper fence = Q3 + k × IQR  (k = 1.5 by default)
```

### Classification Rules

| Level | Condition | Interpretation |
|-------|-----------|----------------|
| **CRITICAL** | Q(v) > Q3 + 1.5×IQR | Statistical outlier—immediate attention |
| **HIGH** | Q(v) > Q3 | Top quartile—high priority review |
| **MEDIUM** | Q(v) > Median | Above average—regular monitoring |
| **LOW** | Q(v) > Q1 | Below average—standard practices |
| **MINIMAL** | Q(v) ≤ Q1 | Bottom quartile—low concern |

### Why Box-Plot Classification?

| Benefit | Explanation |
|---------|-------------|
| **Adaptive** | Thresholds adjust to each dataset's distribution |
| **No magic numbers** | Avoids arbitrary cutoffs like "0.7 is critical" |
| **Statistically grounded** | Based on well-understood descriptive statistics |
| **Scale-independent** | Works regardless of absolute score magnitudes |

### Adjusting Sensitivity

The k-factor controls outlier sensitivity:

```bash
# More sensitive (more components flagged as critical)
python analyze_graph.py --layer system --k-factor 1.0

# Less sensitive (fewer critical flags)
python analyze_graph.py --layer system --k-factor 2.0
```

---

## 5. Edge Criticality

Dependencies (edges) are also scored for criticality.

### Edge Reliability Score

```
R_e(e) = 0.40 × weight_norm + 0.60 × (R(source) + R(target)) / 2
```

Edges connecting critical components or carrying high-weight data are more critical.

### Edge Availability Score

```
A_e(e) = 0.60 × is_bridge + 0.40 × (A(source) + A(target)) / 2
```

Bridge edges between high-availability-risk components are critical paths.

### Overall Edge Score

```
Q_e(e) = 0.50 × R_e(e) + 0.50 × A_e(e)
```

**Note**: Edge Maintainability is not computed—maintainability is inherently node-centric.

---

## 6. Using analyze_graph.py

### Basic Usage

```bash
# Analyze a specific layer
python analyze_graph.py --layer system

# Analyze all primary layers (app, infra, system)
python analyze_graph.py --all

# Include middleware layers
python analyze_graph.py --all --include-middleware
```

### Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--layer`, `-l` | system | Layer to analyze: app, infra, mw-app, mw-infra, system |
| `--all`, `-a` | false | Analyze all primary layers |
| `--include-middleware` | false | Include mw-app and mw-infra with --all |
| `--output`, `-o` | none | Export results to JSON file |
| `--json` | false | Output JSON to stdout |
| `--uri` | bolt://localhost:7687 | Neo4j connection URI |
| `--user`, `-u` | neo4j | Neo4j username |
| `--password`, `-p` | password | Neo4j password |
| `--k-factor`, `-k` | 1.5 | Box-plot IQR multiplier |
| `--damping`, `-d` | 0.85 | PageRank damping factor |
| `--quiet`, `-q` | false | Minimal output |
| `--verbose`, `-v` | false | Debug output |

### Layer Options

| Layer | Components | Dependencies | Use Case |
|-------|------------|--------------|----------|
| `app` | Applications | app_to_app | Service-level analysis |
| `infra` | Nodes | node_to_node | Infrastructure analysis |
| `mw-app` | Apps + Brokers | app_to_broker | Middleware coupling |
| `mw-infra` | Nodes + Brokers | node_to_broker | Infra-middleware coupling |
| `system` | All | All | Complete system analysis |

### Example Commands

```bash
# Basic system analysis
python analyze_graph.py --layer system

# Application layer with JSON export
python analyze_graph.py --layer app --output results/app_analysis.json

# All layers with verbose output
python analyze_graph.py --all --verbose

# Custom PageRank damping
python analyze_graph.py --layer system --damping 0.90

# More sensitive classification
python analyze_graph.py --layer system --k-factor 1.0
```

---

## 7. Understanding Output

### Terminal Output Structure

```
══════════════════════════════════════════════════════════════════════════════
                              system Analysis
══════════════════════════════════════════════════════════════════════════════
  Complete system analysis including all component types and dependencies

>> Graph Summary
──────────────────────────────────────────────────────────────────────────────
  Nodes:               48
  Edges:               127
  Density:             0.0562
  Avg Degree:          5.29
  Avg Clustering:      0.1847
  Connected:           Yes
  Components:          1
  Articulation Pts:    3
  Bridges:             5
  Health:              MODERATE

  Node Types: Application: 25, Broker: 2, Node: 6, Topic: 15
  Edge Types: DEPENDS_ON: 127

>> Classification Summary
──────────────────────────────────────────────────────────────────────────────

  Components (48 total):
    critical   ████████ 4
    high       ████████████ 6
    medium     ████████████████████████ 12
    low        ████████████████████████████████ 16
    minimal    ████████████████████ 10

>> Top Components by Criticality
──────────────────────────────────────────────────────────────────────────────
  ID                        Type         R       M       A       Q       Level
  -------------------------------------------------------------------------
  sensor_fusion             Application  0.847   0.892   0.825   0.854   CRITICAL ●
  main_broker               Broker       0.823   0.756   0.900   0.826   CRITICAL ●
  planning_node             Application  0.756   0.834   0.650   0.745   HIGH
  gateway_node              Node         0.698   0.623   0.800   0.707   HIGH
  control_app               Application  0.634   0.712   0.575   0.639   MEDIUM
  ...

  Legend: R=Reliability, M=Maintainability, A=Availability, Q=Overall
          ● = Articulation Point (SPOF)

>> Critical Edges
──────────────────────────────────────────────────────────────────────────────
  sensor_fusion → lidar_driver [app_to_app] Score: 0.823 CRITICAL 🌉
  planning_node → sensor_fusion [app_to_app] Score: 0.756 HIGH
  ...

>> Detected Problems
──────────────────────────────────────────────────────────────────────────────

  Total: 3 problems
  By Severity: CRITICAL: 1  HIGH: 2

  -------------------------------------------------------------------------
  [CRITICAL] Single Point of Failure
           Entity: sensor_fusion (Application)
           Category: availability
           Issue: Component is an articulation point with high betweenness
                  (0.892). Removal disconnects the dependency graph.
           Fix: Add redundant instance or decouple into smaller services.
  -------------------------------------------------------------------------
  [    HIGH] Excessive Coupling
           Entity: main_broker (Broker)
           Category: maintainability
           Issue: Component has very high degree centrality (0.756) with
                  low clustering (0.12), indicating tight coupling.
           Fix: Consider introducing abstraction layer or message filtering.
  -------------------------------------------------------------------------
```

### Output Sections Explained

| Section | Content |
|---------|---------|
| **Graph Summary** | Basic topology stats, connectivity health |
| **Classification Summary** | Distribution of criticality levels |
| **Top Components** | Ranked list with R, M, A, Q scores |
| **Critical Edges** | High-criticality dependencies |
| **Detected Problems** | Architectural anti-patterns with recommendations |

### JSON Output Structure

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "layer": "system",
  "layer_name": "Complete System",
  "structural": {
    "graph_summary": {
      "nodes": 48,
      "edges": 127,
      "density": 0.0562,
      "avg_degree": 5.29,
      "avg_clustering": 0.1847,
      "is_connected": true,
      "num_components": 1,
      "num_articulation_points": 3,
      "num_bridges": 5,
      "connectivity_health": "MODERATE"
    }
  },
  "quality": {
    "components": [
      {
        "id": "sensor_fusion",
        "type": "Application",
        "scores": {
          "reliability": 0.847,
          "maintainability": 0.892,
          "availability": 0.825,
          "overall": 0.854
        },
        "levels": {
          "reliability": "critical",
          "maintainability": "critical",
          "availability": "high",
          "overall": "critical"
        },
        "structural": {
          "pagerank": 0.089,
          "reverse_pagerank": 0.112,
          "betweenness": 0.234,
          "clustering": 0.125,
          "degree": 12,
          "is_articulation_point": true,
          "bridge_ratio": 0.417
        }
      }
    ],
    "edges": [...],
    "classification_summary": {
      "total_components": 48,
      "component_distribution": {
        "critical": 4,
        "high": 6,
        "medium": 12,
        "low": 16,
        "minimal": 10
      }
    }
  },
  "problems": [
    {
      "severity": "CRITICAL",
      "name": "Single Point of Failure",
      "entity_id": "sensor_fusion",
      "entity_type": "Application",
      "category": "availability",
      "description": "Component is an articulation point...",
      "recommendation": "Add redundant instance..."
    }
  ]
}
```

---

## 8. Multi-Layer Analysis

### Analyzing All Layers

```bash
python analyze_graph.py --all
```

This analyzes:
- **app**: Application-to-application dependencies
- **infra**: Node-to-node infrastructure dependencies
- **system**: Complete system with all dependencies

### Including Middleware Layers

```bash
python analyze_graph.py --all --include-middleware
```

Adds:
- **mw-app**: Application-to-broker dependencies
- **mw-infra**: Node-to-broker dependencies

### Layer-Specific Insights

| Layer | Typical Findings |
|-------|------------------|
| **app** | Service bottlenecks, data flow criticality |
| **infra** | Network chokepoints, hardware SPOFs |
| **mw-app** | Broker dependencies, middleware coupling |
| **mw-infra** | Infrastructure-middleware coupling |
| **system** | Overall criticality, cross-cutting concerns |

### Cross-Layer Analysis Output

```
══════════════════════════════════════════════════════════════════════════════
                         Multi-Layer Analysis Results
══════════════════════════════════════════════════════════════════════════════

  Timestamp: 2024-01-15T10:30:00
  Layers: app, infra, system

[... per-layer results ...]

>> Cross-Layer Insights
──────────────────────────────────────────────────────────────────────────────
  • sensor_fusion is critical in both app and system layers
  • gateway_node is an infrastructure SPOF hosting 3 critical applications
  • main_broker appears in top-5 critical components across all layers

>> Overall Summary
──────────────────────────────────────────────────────────────────────────────
  Layers analyzed:     3
  Total components:    48
  Total problems:      5
  Critical problems:   2
```

---

## 9. Problem Detection

The analyzer automatically detects architectural anti-patterns.

### Problem Categories

| Category | Problems Detected |
|----------|-------------------|
| **Availability** | SPOFs, articulation points, bridge edges |
| **Maintainability** | Excessive coupling, low modularity, hub components |
| **Reliability** | High failure propagation, concentrated dependencies |
| **Structural** | Disconnected components, isolated nodes |

### Problem Severity Levels

| Severity | Meaning | Action |
|----------|---------|--------|
| **CRITICAL** | Immediate system risk | Fix before deployment |
| **HIGH** | Significant concern | Schedule remediation |
| **MEDIUM** | Worth addressing | Include in backlog |
| **LOW** | Minor improvement | Consider when convenient |

### Common Problems and Fixes

| Problem | Indicator | Recommended Fix |
|---------|-----------|-----------------|
| **Single Point of Failure** | Articulation point | Add redundancy, decouple |
| **Excessive Coupling** | High degree + low clustering | Introduce abstraction layer |
| **Communication Bottleneck** | High betweenness | Distribute responsibility |
| **Critical Path** | Bridge edge | Add alternative route |
| **Hub Overload** | High in-degree + out-degree | Split into smaller components |

---

## 10. Worked Example

### Scenario

A sensor fusion system with 3 applications:

```
Sensor-A ──pub──▶ /raw ◀──sub── Fusion ──pub──▶ /fused ◀──sub── Display
Sensor-B ──pub──▶ /raw
```

### Step 1: Raw Metrics

| Component | PageRank | Rev. PR | In-Deg | Betweenness | Clustering | Degree | AP? | Bridge Ratio |
|-----------|----------|---------|--------|-------------|------------|--------|-----|--------------|
| Sensor-A | 0.45 | 0.20 | 0 | 0.00 | 0.00 | 1 | No | 0.00 |
| Sensor-B | 0.45 | 0.20 | 0 | 0.00 | 0.00 | 1 | No | 0.00 |
| Fusion | 0.35 | 0.50 | 2 | 0.80 | 0.00 | 3 | Yes | 1.00 |
| Display | 0.20 | 0.30 | 1 | 0.00 | 0.00 | 1 | No | 0.00 |

### Step 2: Normalization

For PageRank: min=0.20, max=0.45, range=0.25

| Component | PR_norm | RP_norm | ID_norm | BT_norm | CC_norm | DC_norm |
|-----------|---------|---------|---------|---------|---------|---------|
| Sensor-A | 1.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.33 |
| Sensor-B | 1.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.33 |
| Fusion | 0.60 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |
| Display | 0.00 | 0.33 | 0.50 | 0.00 | 0.50 | 0.33 |

### Step 3: RMA Calculation for Fusion

**Reliability:**
```
R(Fusion) = 0.45 × 0.60 + 0.35 × 1.00 + 0.20 × 1.00
          = 0.27 + 0.35 + 0.20
          = 0.82
```

**Maintainability:**
```
M(Fusion) = 0.45 × 1.00 + 0.25 × (1 - 0.50) + 0.30 × 1.00
          = 0.45 + 0.125 + 0.30
          = 0.875
```

**Availability** (CR = 0.60 × 1.00 = 0.60):
```
A(Fusion) = 0.50 × 1.0 + 0.25 × 1.00 + 0.25 × 0.60
          = 0.50 + 0.25 + 0.15
          = 0.90
```

**Overall:**
```
Q(Fusion) = 0.35 × 0.82 + 0.30 × 0.875 + 0.35 × 0.90
          = 0.287 + 0.263 + 0.315
          = 0.865
```

### Step 4: Classification

All Q scores: [0.32, 0.32, 0.865, 0.28]

```
Q1 = 0.30, Median = 0.32, Q3 = 0.59
IQR = 0.29
Upper fence = 0.59 + 1.5 × 0.29 = 1.025
```

Fusion (0.865) > Q3 + 1.5×IQR? No, but > Q3? Yes → **HIGH**

(In a larger system, 0.865 would likely be CRITICAL as an outlier)

### Step 5: Interpretation

**Fusion** scores 0.865 overall, driven by:
- **High Availability (0.90)**: Articulation point with all bridge edges
- **High Maintainability (0.875)**: Central bottleneck with high betweenness
- **High Reliability (0.82)**: Significant failure propagation potential

**Recommendation**: Fusion is a critical component requiring:
- Redundant instance for availability
- Health monitoring for reliability
- Careful change management for maintainability

---

## Quick Reference

### Commands

```bash
# Single layer analysis
python analyze_graph.py --layer system

# All layers
python analyze_graph.py --all

# Export to JSON
python analyze_graph.py --layer app --output results.json

# Adjust sensitivity
python analyze_graph.py --layer system --k-factor 1.0
```

### Quality Formulas

```
R(v) = 0.45×PR + 0.35×RP + 0.20×ID         (Reliability)
M(v) = 0.45×BT + 0.25×(1-CC) + 0.30×DC     (Maintainability)
A(v) = 0.50×AP + 0.25×BR + 0.25×CR         (Availability)
Q(v) = 0.35×R + 0.30×M + 0.35×A            (Overall)
```

### Classification Thresholds

```
CRITICAL:  Q > Q3 + 1.5×IQR  (outlier)
HIGH:      Q > Q3            (top quartile)
MEDIUM:    Q > Median        (above average)
LOW:       Q > Q1            (below average)
MINIMAL:   Q ≤ Q1            (bottom quartile)
```

### Key Metrics

| Metric | Used In | Captures |
|--------|---------|----------|
| PageRank | R | Transitive influence |
| Reverse PageRank | R | Failure propagation |
| In-Degree | R | Direct dependents |
| Betweenness | M | Bottleneck |
| Clustering | M | Modularity (inverted) |
| Degree | M | Interface complexity |
| Articulation Point | A | Structural SPOF |
| Bridge Ratio | A | Critical connections |
| Criticality (PR×DC) | A | Important hub |

---

## Next Step

After structural analysis, proceed to **Step 3: Failure Simulation** to validate predictions against actual failure impact.

```bash
python simulate_graph.py --exhaustive --layer system
```

---

## Navigation

- **Previous**: [Step 1: Graph Model Construction](step1-graph-model-construction.md)
- **Next**: [Step 3: Failure Simulation](step3-failure-simulation.md)
- **See Also**: [Step 4: Validation](step4-validation.md)
