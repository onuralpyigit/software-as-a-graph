# Quality Assessment Formulations

This document formally defines the composite criticality scores for assessing **Reliability**, **Maintainability**, and **Availability** using graph topological metrics.

---

## Table of Contents

1. [Overview](#overview)
2. [Component Quality Scores](#component-quality-scores)
   - [Reliability Score](#formula-1-reliability-score-rv)
   - [Maintainability Score](#formula-2-maintainability-score-mv)
   - [Availability Score](#formula-3-availability-score-av)
   - [Overall Quality Criticality](#formula-4-overall-quality-criticality-qv)
3. [Edge Quality Scores](#edge-quality-scores)
   - [Edge Reliability Score](#formula-5-edge-reliability-score)
   - [Edge Availability Score](#formula-6-edge-availability-score)
4. [Metrics Definitions](#metrics-definitions)
5. [Normalization](#normalization)
6. [Box-Plot Classification](#box-plot-classification)
7. [Default Weights](#default-weights)
8. [Implementation](#implementation)

---

## Overview

Quality assessment in distributed pub-sub systems uses graph topological metrics to identify critical components and edges. The approach combines multiple metrics into composite scores that capture different quality dimensions:

| Attribute | Focus | Key Question |
|-----------|-------|--------------|
| **Reliability** | Fault propagation | What happens when this component fails? |
| **Maintainability** | Coupling & complexity | How difficult is it to modify this component? |
| **Availability** | Service continuity | Is this a single point of failure? |

The formulations are designed to:
- Identify critical components before deployment
- Prioritize maintenance and monitoring efforts
- Guide architectural improvements

---

## Component Quality Scores

### Formula (1): Reliability Score R(v)

**Purpose**: Measures the potential for fault propagation and system-wide failure impact.

$$R(v) = \omega_{PR} \cdot PR_{norm}(v) + \omega_{FP} \cdot FP_{norm}(v) + \omega_{ID} \cdot ID_{norm}(v)$$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $PR_{norm}(v)$ | [0, 1] | Normalized PageRank (transitive influence) |
| $FP_{norm}(v)$ | [0, 1] | Normalized Reverse PageRank (Failure Propagation) |
| $ID_{norm}(v)$ | [0, 1] | Normalized in-degree (dependency count) |
| $\omega_{PR}$ | [0, 1] | PageRank weight (default: 0.45) |
| $\omega_{FP}$ | [0, 1] | Failure propagation weight (default: 0.35) |
| $\omega_{ID}$ | [0, 1] | In-degree weight (default: 0.20) |

**Interpretation**: Higher R(v) → Higher reliability risk if component v fails.

**Rationale**:
- PageRank captures transitive influence through the dependency graph
- Failure propagation measures potential cascade depth
- In-degree indicates how many components depend on v

---

### Formula (2): Maintainability Score M(v)

**Purpose**: Measures coupling, complexity, and change propagation risk.

$$M(v) = \omega_{BT} \cdot BT_{norm}(v) + \omega_{CC} \cdot (1 - CC_{norm}(v)) + \omega_{DC} \cdot DC_{norm}(v)$$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $BT_{norm}(v)$ | [0, 1] | Normalized betweenness centrality (coupling) |
| $CC_{norm}(v)$ | [0, 1] | Normalized clustering coefficient (modularity) |
| $DC_{norm}(v)$ | [0, 1] | Normalized degree centrality (interface complexity) |
| $\omega_{BT}$ | [0, 1] | Betweenness weight (default: 0.45) |
| $\omega_{CC}$ | [0, 1] | Clustering weight (default: 0.25) |
| $\omega_{DC}$ | [0, 1] | Degree weight (default: 0.30) |

**Interpretation**: Higher M(v) → Harder to maintain, higher change propagation risk.

**Rationale**:
- Betweenness indicates bottleneck/coupling characteristics
- Clustering coefficient is *inverted* because low clustering means poor modularity
- Degree centrality reflects interface complexity

---

### Formula (3): Availability Score A(v)

**Purpose**: Measures single point of failure risk and service continuity impact.

$$A(v) = \omega_{AP} \cdot AP(v) + \omega_{BR} \cdot BR(v) + \omega_{CR} \cdot CR_{norm}(v)$$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $AP(v)$ | {0, 1} | Articulation point indicator (structural SPOF) |
| $BR(v)$ | [0, 1] | Bridge ratio (bridge edges / total incident edges) |
| $CR_{norm}(v)$ | [0, 1] | Normalized Criticality ($PR_{norm} \times Degree_{norm}$) |
| $\omega_{AP}$ | [0, 1] | Articulation point weight (default: 0.50) |
| $\omega_{BR}$ | [0, 1] | Bridge ratio weight (default: 0.25) |
| $\omega_{CR}$ | [0, 1] | Criticality weight (default: 0.25) |

**Interpretation**: Higher A(v) → Higher availability risk (more likely SPOF).

**Rationale**:
- Articulation points are structural single points of failure
- Bridge ratio measures how many critical edges connect through v
- Criticality combines influence and connectivity

---

### Formula (4): Overall Quality Criticality Q(v)

**Purpose**: Composite score combining all quality dimensions.

$$Q(v) = \alpha \cdot R(v) + \beta \cdot M(v) + \gamma \cdot A(v)$$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $R(v)$ | [0, 1] | Reliability score from Formula (1) |
| $M(v)$ | [0, 1] | Maintainability score from Formula (2) |
| $A(v)$ | [0, 1] | Availability score from Formula (3) |
| $\alpha$ | [0, 1] | Reliability weight (default: 0.35) |
| $\beta$ | [0, 1] | Maintainability weight (default: 0.30) |
| $\gamma$ | [0, 1] | Availability weight (default: 0.35) |

**Interpretation**: Higher Q(v) → Higher overall criticality.

---

## Edge Quality Scores

### Formula (5): Edge Reliability Score

**Purpose**: Measure reliability impact of an edge.

For edge $e = (u, v)$:

$$R_e(e) = \omega_w \cdot w(e) + \omega_{ep} \cdot \frac{R(u) + R(v)}{2}$$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $w(e)$ | [0, 1] | Normalized edge weight |
| $R(u), R(v)$ | [0, 1] | Endpoint reliability scores |
| $\omega_w$ | [0, 1] | Edge weight factor (default: 0.40) |
| $\omega_{ep}$ | [0, 1] | Endpoint factor (default: 0.60) |

---

### Formula (6): Edge Availability Score

**Purpose**: Measure availability impact of an edge.

For edge $e = (u, v)$:

$$A_e(e) = \omega_{br} \cdot BR_e(e) + \omega_{ap} \cdot \frac{AP(u) + AP(v)}{2}$$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $BR_e(e)$ | {0, 1} | Bridge indicator (1 if bridge, 0 otherwise) |
| $AP(u), AP(v)$ | {0, 1} | Endpoint Availability Scores (from Formula 3)* |
| $\omega_{br}$ | [0, 1] | Bridge weight (default: 0.60) |
| $\omega_{ap}$ | [0, 1] | Articulation point weight (default: 0.40) |

---

## Metrics Definitions

### PageRank

PageRank measures transitive importance based on incoming connections:

$$PR(v) = \frac{1-d}{N} + d \sum_{u \in B(v)} \frac{PR(u)}{L(u)}$$

Where:
- $d$ = damping factor (default: 0.85)
- $N$ = total number of nodes
- $B(v)$ = set of nodes linking to v
- $L(u)$ = number of outgoing links from u

### Betweenness Centrality

Betweenness measures how often a node lies on shortest paths:

$$BC(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

Where:
- $\sigma_{st}$ = number of shortest paths from s to t
- $\sigma_{st}(v)$ = number of those paths passing through v

### Clustering Coefficient

Local clustering coefficient measures neighbor connectivity:

$$CC(v) = \frac{2 \cdot e_v}{k_v(k_v - 1)}$$

Where:
- $e_v$ = number of edges between neighbors of v
- $k_v$ = degree of v

### Articulation Point

A node v is an articulation point if removing v disconnects the graph:

$$AP(v) = \begin{cases} 1 & \text{if } G - \{v\} \text{ has more connected components than } G \\ 0 & \text{otherwise} \end{cases}$$

### Bridge Edge

An edge e is a bridge if removing it disconnects the graph:

$$BR_e(e) = \begin{cases} 1 & \text{if } G - \{e\} \text{ has more connected components than } G \\ 0 & \text{otherwise} \end{cases}$$

---

## Normalization

All metrics are normalized to [0, 1] using min-max scaling:

$$x_{norm} = \frac{x - \min(X)}{\max(X) - \min(X)}$$

Special cases:
- If $\max(X) = \min(X)$, return 0.5 for all values
- Empty sets return 0.0

---

## Box-Plot Classification

Instead of fixed thresholds, classification uses box-plot statistics:

```
                    │
                    │    ┌───────────┐
    CRITICAL ───────┼────│  outliers │ > Q3 + k×IQR
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

**Classification Rules**:

| Level | Condition |
|-------|-----------|
| CRITICAL | score > Q3 + k × IQR |
| HIGH | score > Q3 |
| MEDIUM | score > Median (Q2) |
| LOW | score > Q1 |
| MINIMAL | score ≤ Q1 |

**Default k-factor**: 1.5 (standard box-plot)

---

## Default Weights

### Component Weights

```python
DEFAULT_WEIGHTS = {
    "reliability": {
        "pagerank": 0.45,
        "failure_propagation": 0.35,
        "in_degree": 0.20,
    },
    "maintainability": {
        "betweenness": 0.45,
        "clustering": 0.25,
        "degree": 0.30,
    },
    "availability": {
        "articulation_point": 0.50,
        "bridge_ratio": 0.25,
        "criticality": 0.25,
    },
    "overall": {
        "reliability": 0.35,
        "maintainability": 0.30,
        "availability": 0.35,
    }
}
```

### Edge Weights

```python
DEFAULT_EDGE_WEIGHTS = {
    "reliability": {
        "weight": 0.40,
        "endpoint_avg": 0.60,
    },
    "availability": {
        "bridge": 0.60,
        "endpoint_spof": 0.40,
    }
}
```

---

## CLI Usage

```bash
# Full quality assessment
python analyze_graph.py

# Reliability-focused analysis
python analyze_graph.py --attribute reliability

# Custom k-factor
python analyze_graph.py --k-factor 2.0

# Export results
python analyze_graph.py --output results.json
```

---

## Navigation

- **Previous**: [Analysis Overview](analysis.md)
- **Next**: [Validation](validation.md)
