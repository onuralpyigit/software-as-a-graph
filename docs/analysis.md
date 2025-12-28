# Structural Analysis

This document explains the graph algorithms and metrics used to analyze component criticality in publish-subscribe systems.

---

## Table of Contents

1. [Overview](#overview)
2. [Centrality Metrics](#centrality-metrics)
3. [Articulation Points](#articulation-points)
4. [Composite Criticality Score](#composite-criticality-score)
5. [Box-Plot Classification](#box-plot-classification)
6. [Anti-Pattern Detection](#anti-pattern-detection)
7. [Implementation](#implementation)

---

## Overview

Structural analysis applies graph algorithms to identify components with high topological importance. The key insight is that **structural centrality correlates with failure impact**—components that are central to the graph's structure cause more damage when they fail.

### Analysis Pipeline

```
┌─────────────────┐
│   Input Graph   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Centrality      │
│ Computation     │
│ • Betweenness   │
│ • Degree        │
│ • PageRank      │
│ • Message Path  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Articulation    │
│ Point Detection │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Composite Score │
│ Calculation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Box-Plot        │
│ Classification  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Criticality     │
│ Levels          │
└─────────────────┘
```

---

## Centrality Metrics

We compute multiple centrality metrics to capture different aspects of structural importance.

### Betweenness Centrality

**Definition**: Measures how often a node lies on shortest paths between other nodes.

**Formula**:
```
          σst(v)
CB(v) = Σ ──────
       s≠v≠t  σst
```

Where:
- σst = total number of shortest paths from s to t
- σst(v) = number of those paths passing through v

**Normalized**:
```
                   CB(v)
CB_norm(v) = ─────────────────────
             (n-1)(n-2)/2
```

**Interpretation**:
- High betweenness → Component is a bottleneck
- Removal disrupts many shortest paths
- Indicates potential single point of failure (SPOF)

**Example**:
```
    [A]     [B]
      \     /
       \   /
        [C]  ← High betweenness (bridges A-B with D-E)
       /   \
      /     \
    [D]     [E]
```

### Degree Centrality

**Definition**: Measures the number of direct connections.

**Formula**:
```
         deg(v)
CD(v) = ────────
         n - 1
```

Where deg(v) = number of edges incident to v.

**For Directed Graphs**:
```
              deg_in(v) + deg_out(v)
CD(v) = ───────────────────────────────
                  2(n - 1)
```

**Interpretation**:
- High degree → Highly coupled component
- Many dependencies (in or out)
- Changes to this component affect many others

**Example**:
```
    [A]──┐
    [B]──┼──▶[C]  ← High in-degree (many depend on C)
    [D]──┘
```

### PageRank

**Definition**: Measures importance based on incoming links from other important nodes.

**Formula (Iterative)**:
```
              1 - d       PR(u)
PR(v) = ────────── + d Σ ─────────
            n         u∈M(v) deg(u)
```

Where:
- d = damping factor (typically 0.85)
- M(v) = set of nodes linking to v
- n = total number of nodes

**Interpretation**:
- High PageRank → Receives data from important sources
- Recursive importance measure
- Captures transitive significance

**Example**:
```
[Important_Publisher]───▶[Topic]───▶[Subscriber]
         ↑                              ↑
    High PageRank               Also gets high PageRank
```

### Message Path Centrality

**Definition**: Custom metric measuring participation in actual message flow paths.

**Formula**:
```
                |paths containing v|
MPC(v) = ────────────────────────────
              |all message paths|
```

A message path is: Publisher → Topic → Subscriber (through broker).

**Interpretation**:
- High MPC → Critical for message delivery
- Domain-specific (not pure graph theory)
- Captures pub-sub communication patterns

**Example**:
```
[Pub1]──▶[Topic1]──▶[Sub1]
              │
              ├────▶[Sub2]
              │
[Pub2]──▶[Topic1]  ← Topic1 has high MPC (on many paths)
```

---

## Articulation Points

### Definition

An **articulation point** (or cut vertex) is a node whose removal disconnects the graph.

### Detection Algorithm

Uses depth-first search (DFS) with low-link values:

```
Algorithm: Find Articulation Points

1. Run DFS, tracking:
   - discovery[v] = time when v was discovered
   - low[v] = lowest discovery time reachable from subtree of v

2. Node v is articulation point if:
   - v is root with ≥2 children, OR
   - v is not root AND has child u where low[u] ≥ discovery[v]
```

### Interpretation

- Articulation points are **structurally critical**
- Their removal creates disconnected subgraphs
- Indicate architectural vulnerabilities
- Binary indicator (is or isn't)

### Example

```
Before:                    After removing B:
    [A]                        [A]
     │                          
    [B] ← Articulation        [C]    [D]
   /   \                     
 [C]   [D]                  (Disconnected!)
```

### Usage in Criticality

Articulation points receive a bonus in composite scoring:

```
AP(v) = 1 if v is articulation point, else 0
```

---

## Composite Criticality Score

We combine metrics into a single score using weighted summation.

### Formula

```
C_score(v) = α·CB_norm(v) + β·AP(v) + γ·DC_norm(v) + δ·PR_norm(v)
```

Where:
- **CB_norm(v)** ∈ [0,1]: Normalized betweenness centrality
- **AP(v)** ∈ {0,1}: Articulation point indicator
- **DC_norm(v)** ∈ [0,1]: Normalized degree centrality
- **PR_norm(v)** ∈ [0,1]: Normalized PageRank

### Default Weights

| Weight | Value | Rationale |
|--------|-------|-----------|
| α (Betweenness) | 0.35 | Strong predictor of bottlenecks |
| β (Articulation) | 0.25 | Critical structural vulnerability |
| γ (Degree) | 0.20 | Indicates coupling |
| δ (PageRank) | 0.20 | Captures transitive importance |

### Normalization

All metrics normalized to [0,1] using min-max scaling:

```
              x - min(X)
x_norm = ─────────────────
          max(X) - min(X)
```

### Example Calculation

For component B1 with:
- Betweenness: 0.45 (normalized)
- Is articulation point: Yes (1)
- Degree: 0.30 (normalized)
- PageRank: 0.25 (normalized)

```
C_score(B1) = 0.35×0.45 + 0.25×1 + 0.20×0.30 + 0.20×0.25
            = 0.1575 + 0.25 + 0.06 + 0.05
            = 0.5175
```

---

## Box-Plot Classification

Instead of fixed thresholds, we use box-plot statistical classification.

### The Sharp Boundary Problem

Fixed thresholds create arbitrary boundaries:

```
Score: 0.799 → LOW
Score: 0.800 → HIGH   ← Only 0.001 difference!
```

This doesn't reflect real significance differences.

### Box-Plot Solution

Classification based on score distribution:

```
                    │
                    │    ┌───────────┐
    CRITICAL ───────┼────│  outliers │ > Q3 + 1.5×IQR
                    │    └───────────┘
                    │    ┌───────────┐
    HIGH ───────────┼────│    top    │ > Q3
                    │    │  quartile │
                    │    ├───────────┤
    MEDIUM ─────────┼────│   above   │ > Median
                    │    │  median   │
                    │    ├───────────┤
    LOW ────────────┼────│   below   │ > Q1
                    │    │  median   │
                    │    ├───────────┤
    MINIMAL ────────┼────│  bottom   │ ≤ Q1
                    │    │ quartile  │
                    │    └───────────┘
```

### Formulas

```
Q1 = 25th percentile
Q2 = 50th percentile (median)
Q3 = 75th percentile
IQR = Q3 - Q1 (interquartile range)
```

**Classification Rules**:

| Level | Condition |
|-------|-----------|
| CRITICAL | score > Q3 + 1.5 × IQR |
| HIGH | score > Q3 |
| MEDIUM | score > Q2 |
| LOW | score > Q1 |
| MINIMAL | score ≤ Q1 |

### Advantages

1. **Adaptive**: Thresholds adjust to each system's distribution
2. **Statistical**: Based on actual data, not arbitrary values
3. **Outlier Detection**: CRITICAL level identifies true outliers
4. **Comparable**: Consistent interpretation across systems

### Implementation

```python
from src.analysis import BoxPlotClassifier

classifier = BoxPlotClassifier(k_factor=1.5)

items = [
    {"id": "B1", "type": "Broker", "score": 0.52},
    {"id": "B2", "type": "Broker", "score": 0.48},
    {"id": "A1", "type": "Application", "score": 0.15},
    # ... more items
]

result = classifier.classify(items, metric_name="composite")

# Access results
print(f"Critical: {len(result.by_level['CRITICAL'])}")
print(f"Thresholds: Q1={result.stats.q1:.3f}, Q3={result.stats.q3:.3f}")
```

---

## Anti-Pattern Detection

The analysis module can detect common architectural anti-patterns.

### God Topic

A topic with too many publishers or subscribers.

**Detection**: Degree centrality significantly above average.

```
[Pub1]──┐
[Pub2]──┼──▶[GodTopic]──┬──▶[Sub1]
[Pub3]──┤               ├──▶[Sub2]
[Pub4]──┘               ├──▶[Sub3]
                        └──▶[Sub4]
```

**Issue**: Single point of failure, scalability bottleneck.

### Single Point of Failure (SPOF)

Component whose failure disconnects the system.

**Detection**: Articulation point with high betweenness.

```
[A]──▶[SPOF]──▶[B]
        │
        └──▶[C]
```

**Issue**: No redundancy, high risk.

### Bottleneck Broker

Broker routing too many topics relative to capacity.

**Detection**: High routing degree compared to other brokers.

```
[T1]──┐
[T2]──┼──▶[BottleneckBroker]  (routes 80% of topics)
[T3]──┤
...   │
[Tn]──┘
```

**Issue**: Performance bottleneck, single point of failure.

### Chatty Application

Application with excessive publish/subscribe relationships.

**Detection**: High message path participation.

```
         ┌──▶[T1]
[Chatty]─┼──▶[T2]
         ├──▶[T3]
         ├──▶[T4]
         └──▶[T5]
```

**Issue**: Noisy, hard to trace issues.

### Implementation

```python
from src.analysis import AntiPatternDetector

detector = AntiPatternDetector(
    god_topic_threshold=0.3,  # % of total connections
    spof_betweenness_threshold=0.5,
    bottleneck_ratio=0.5
)

patterns = detector.detect(graph)

for pattern in patterns:
    print(f"{pattern.type}: {pattern.component}")
    print(f"  Severity: {pattern.severity}")
    print(f"  Recommendation: {pattern.recommendation}")
```

---

## Implementation

### GraphAnalyzer Class

```python
from src.validation import GraphAnalyzer

# Create analyzer
analyzer = GraphAnalyzer(graph)

# Individual metrics
bc = analyzer.betweenness_centrality()  # Dict[str, float]
dc = analyzer.degree_centrality()
pr = analyzer.pagerank()
mpc = analyzer.message_path_centrality()

# Articulation points
aps = analyzer.articulation_points()  # Set[str]

# Composite score
composite = analyzer.composite_score(
    weights={"betweenness": 0.35, "degree": 0.20, "pagerank": 0.20}
)

# All metrics at once
all_metrics = analyzer.analyze_all()
# Returns: {"degree": {...}, "betweenness": {...}, "composite": {...}}
```

### GDSClient (Neo4j)

For large-scale analysis using Neo4j Graph Data Science:

```python
from src.analysis import GDSClient

gds = GDSClient(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Create projection
gds.create_depends_on_projection("analysis")

# Run algorithms
betweenness = gds.betweenness_centrality("analysis")
pagerank = gds.pagerank("analysis")
communities = gds.louvain_community("analysis")

# Cleanup
gds.drop_projection("analysis")
gds.close()
```

### BoxPlotClassifier Class

```python
from src.analysis import BoxPlotClassifier, CriticalityLevel

classifier = BoxPlotClassifier(k_factor=1.5)

# Classify items
items = [{"id": "X", "type": "Component", "score": 0.5}, ...]
result = classifier.classify(items, metric_name="composite")

# Access results
for item in result.items:
    print(f"{item.id}: {item.score:.4f} -> {item.level.value}")

# Statistics
print(f"Q1: {result.stats.q1:.4f}")
print(f"Median: {result.stats.median:.4f}")
print(f"Q3: {result.stats.q3:.4f}")
print(f"Upper fence: {result.stats.upper_fence:.4f}")

# By level
for level in CriticalityLevel:
    count = len(result.by_level.get(level.value, []))
    print(f"{level.value}: {count}")
```

---

## Navigation

- **Previous:** [← Graph Model](graph-model.md)
- **Next:** [Failure Simulation →](simulation.md)
