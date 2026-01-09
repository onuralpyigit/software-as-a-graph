# Quality Assessment Formulations

This document formally defines the composite criticality scores for assessing **Reliability**, **Maintainability**, and **Availability** (RMA) using graph topological metrics.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Component Quality Scores](#2-component-quality-scores)
   - [Reliability Score R(v)](#21-formula-1-reliability-score-rv)
   - [Maintainability Score M(v)](#22-formula-2-maintainability-score-mv)
   - [Availability Score A(v)](#23-formula-3-availability-score-av)
   - [Overall Quality Criticality Q(v)](#24-formula-4-overall-quality-criticality-qv)
3. [Edge Quality Scores](#3-edge-quality-scores)
   - [Edge Reliability Score](#31-formula-5-edge-reliability-score)
   - [Edge Availability Score](#32-formula-6-edge-availability-score)
   - [Overall Edge Quality](#33-formula-7-overall-edge-quality)
4. [Metrics Definitions](#4-metrics-definitions)
5. [Normalization](#5-normalization)
6. [Box-Plot Classification](#6-box-plot-classification)
7. [Score Interpretation](#7-score-interpretation)
8. [Layer-Specific Considerations](#8-layer-specific-considerations)
9. [Worked Example](#9-worked-example)
10. [Default Weights](#10-default-weights)
11. [Validation Targets](#11-validation-targets)
12. [Implementation](#12-implementation)
13. [References](#13-references)

---

## 1. Overview

Quality assessment in distributed pub-sub systems uses graph topological metrics to identify critical components and edges. The approach combines multiple metrics into composite scores that capture different quality dimensions:

| Attribute | Focus | Key Question | Risk Indicator |
|-----------|-------|--------------|----------------|
| **Reliability** | Fault propagation | What happens when this component fails? | High influence + many dependents |
| **Maintainability** | Coupling & complexity | How difficult is it to modify this component? | High betweenness + low modularity |
| **Availability** | Service continuity | Is this a single point of failure? | Articulation point + bridge edges |

### Design Goals

The formulations are designed to:

1. **Identify critical components** before deployment through static analysis
2. **Prioritize maintenance efforts** by quantifying change propagation risk
3. **Guide architectural improvements** by detecting structural weaknesses
4. **Enable validation** through comparison with simulation results

### Key Principles

- **Separation of Prediction and Validation**: Topological metrics predict criticality; simulation measures actual impact
- **Adaptive Thresholds**: Box-plot classification avoids arbitrary fixed thresholds
- **Multi-Dimensional Assessment**: R, M, A scores capture orthogonal quality concerns
- **Weight Normalization**: All component weights within each formula sum to 1.0

---

## 2. Component Quality Scores

### 2.1 Formula (1): Reliability Score R(v)

**Purpose**: Measures the potential for fault propagation and system-wide failure impact.

$$R(v) = \omega_{PR} \cdot PR_{norm}(v) + \omega_{RP} \cdot RP_{norm}(v) + \omega_{ID} \cdot ID_{norm}(v)$$

**Constraint**: $\omega_{PR} + \omega_{RP} + \omega_{ID} = 1.0$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $PR_{norm}(v)$ | [0, 1] | Normalized PageRank — transitive influence (importance as a dependency target) |
| $RP_{norm}(v)$ | [0, 1] | Normalized Reverse PageRank — failure propagation potential |
| $ID_{norm}(v)$ | [0, 1] | Normalized in-degree — direct dependency count |
| $\omega_{PR}$ | [0, 1] | PageRank weight (default: **0.45**) |
| $\omega_{RP}$ | [0, 1] | Reverse PageRank weight (default: **0.35**) |
| $\omega_{ID}$ | [0, 1] | In-degree weight (default: **0.20**) |

**Interpretation**: Higher $R(v)$ → Higher reliability risk if component $v$ fails.

**Rationale**:

| Metric | Why It Matters |
|--------|----------------|
| PageRank | Captures transitive influence — a component depended upon by important components is itself important |
| Reverse PageRank | Measures failure propagation — computed on reversed graph to model how failures cascade downstream |
| In-Degree | Direct dependency count — more dependents means broader immediate impact |

> **Note on Reverse PageRank**: This is PageRank computed on the graph with edge directions reversed. In the dependency graph where edges point from dependent → provider, reversing captures "if I fail, how many components are affected downstream?"

---

### 2.2 Formula (2): Maintainability Score M(v)

**Purpose**: Measures coupling, complexity, and change propagation risk.

$$M(v) = \omega_{BT} \cdot BT_{norm}(v) + \omega_{CC} \cdot (1 - CC_{norm}(v)) + \omega_{DC} \cdot DC_{norm}(v)$$

**Constraint**: $\omega_{BT} + \omega_{CC} + \omega_{DC} = 1.0$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $BT_{norm}(v)$ | [0, 1] | Normalized betweenness centrality — coupling/bottleneck indicator |
| $CC_{norm}(v)$ | [0, 1] | Normalized clustering coefficient — local modularity |
| $DC_{norm}(v)$ | [0, 1] | Normalized degree centrality — interface complexity |
| $\omega_{BT}$ | [0, 1] | Betweenness weight (default: **0.45**) |
| $\omega_{CC}$ | [0, 1] | Clustering weight (default: **0.25**) |
| $\omega_{DC}$ | [0, 1] | Degree weight (default: **0.30**) |

**Interpretation**: Higher $M(v)$ → Harder to maintain, higher change propagation risk.

**Rationale**:

| Metric | Why It Matters |
|--------|----------------|
| Betweenness | High betweenness indicates a bottleneck — changes here affect many communication paths |
| Clustering (inverted) | Low clustering means neighbors don't interconnect — poor modularity, changes propagate widely |
| Degree | High degree means many interfaces — more integration points to consider during changes |

> **Note on Clustering Inversion**: The formula uses $(1 - CC_{norm})$ because *low* clustering indicates *poor* maintainability. A well-modularized component has high clustering (neighbors form cohesive groups).

---

### 2.3 Formula (3): Availability Score A(v)

**Purpose**: Measures single point of failure (SPOF) risk and service continuity impact.

$$A(v) = \omega_{AP} \cdot AP(v) + \omega_{BR} \cdot BR(v) + \omega_{CR} \cdot CR_{norm}(v)$$

**Constraint**: $\omega_{AP} + \omega_{BR} + \omega_{CR} = 1.0$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $AP(v)$ | {0, 1} | Articulation point indicator — structural SPOF |
| $BR(v)$ | [0, 1] | Bridge ratio — fraction of incident edges that are bridges |
| $CR_{norm}(v)$ | [0, 1] | Normalized criticality — composite importance measure |
| $\omega_{AP}$ | [0, 1] | Articulation point weight (default: **0.50**) |
| $\omega_{BR}$ | [0, 1] | Bridge ratio weight (default: **0.25**) |
| $\omega_{CR}$ | [0, 1] | Criticality weight (default: **0.25**) |

**Criticality Sub-Formula**:

$$CR(v) = PR_{norm}(v) \times DC_{norm}(v)$$

This multiplicative combination identifies nodes that are both globally important (high PageRank) AND locally connected (high degree).

**Interpretation**: Higher $A(v)$ → Higher availability risk (more likely SPOF).

**Rationale**:

| Metric | Why It Matters |
|--------|----------------|
| Articulation Point | Binary indicator — removal disconnects the graph entirely |
| Bridge Ratio | Proportion of critical edges — indicates how many connections are irreplaceable |
| Criticality | Combines influence and connectivity — captures "important hub" characteristics |

---

### 2.4 Formula (4): Overall Quality Criticality Q(v)

**Purpose**: Composite score combining all quality dimensions into a single criticality measure.

$$Q(v) = \alpha \cdot R(v) + \beta \cdot M(v) + \gamma \cdot A(v)$$

**Constraint**: $\alpha + \beta + \gamma = 1.0$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $R(v)$ | [0, 1] | Reliability score from Formula (1) |
| $M(v)$ | [0, 1] | Maintainability score from Formula (2) |
| $A(v)$ | [0, 1] | Availability score from Formula (3) |
| $\alpha$ | [0, 1] | Reliability weight (default: **0.35**) |
| $\beta$ | [0, 1] | Maintainability weight (default: **0.30**) |
| $\gamma$ | [0, 1] | Availability weight (default: **0.35**) |

**Interpretation**: Higher $Q(v)$ → Higher overall criticality requiring attention.

**Weight Rationale**:
- Reliability and Availability weighted equally (0.35 each) as they both concern system stability
- Maintainability slightly lower (0.30) as it concerns development effort rather than runtime behavior
- Weights can be adjusted based on domain priorities (e.g., safety-critical systems may increase $\alpha$)

---

## 3. Edge Quality Scores

Edges (dependencies) are also assessed for criticality. Edge scores help identify critical communication paths that require monitoring or redundancy.

> **Design Note**: Edge Maintainability is not computed because maintainability is inherently a node-centric property (coupling, modularity). Edges only have Reliability and Availability scores.

### 3.1 Formula (5): Edge Reliability Score

**Purpose**: Measure the reliability impact of a dependency edge.

For edge $e = (u, v)$:

$$R_e(e) = \omega_w \cdot w_{norm}(e) + \omega_{ep} \cdot \frac{R(u) + R(v)}{2}$$

**Constraint**: $\omega_w + \omega_{ep} = 1.0$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $w_{norm}(e)$ | [0, 1] | Normalized edge weight (from QoS/size) |
| $R(u), R(v)$ | [0, 1] | Endpoint reliability scores |
| $\omega_w$ | [0, 1] | Edge weight factor (default: **0.40**) |
| $\omega_{ep}$ | [0, 1] | Endpoint factor (default: **0.60**) |

**Interpretation**: Edges connecting critical components or carrying high-weight data are more critical.

---

### 3.2 Formula (6): Edge Availability Score

**Purpose**: Measure the availability impact of a dependency edge.

For edge $e = (u, v)$:

$$A_e(e) = \omega_{br} \cdot BR_e(e) + \omega_{ep} \cdot \frac{A(u) + A(v)}{2}$$

**Constraint**: $\omega_{br} + \omega_{ep} = 1.0$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $BR_e(e)$ | {0, 1} | Bridge indicator — 1 if edge is a bridge, 0 otherwise |
| $A(u), A(v)$ | [0, 1] | Endpoint availability scores (from Formula 3) |
| $\omega_{br}$ | [0, 1] | Bridge weight (default: **0.60**) |
| $\omega_{ep}$ | [0, 1] | Endpoint availability weight (default: **0.40**) |

**Interpretation**: Bridge edges connecting high-availability-risk components are critical paths.

> **Note**: The formula uses full Availability scores $A(u), A(v)$ rather than just articulation point indicators. This provides a more nuanced assessment by considering all availability factors of the endpoints.

---

### 3.3 Formula (7): Overall Edge Quality

**Purpose**: Combine edge reliability and availability into a single score.

$$Q_e(e) = \omega_R \cdot R_e(e) + \omega_A \cdot A_e(e)$$

**Constraint**: $\omega_R + \omega_A = 1.0$

**Variables**:

| Symbol | Range | Description |
|--------|-------|-------------|
| $R_e(e)$ | [0, 1] | Edge reliability score from Formula (5) |
| $A_e(e)$ | [0, 1] | Edge availability score from Formula (6) |
| $\omega_R$ | [0, 1] | Reliability weight (default: **0.50**) |
| $\omega_A$ | [0, 1] | Availability weight (default: **0.50**) |

**Interpretation**: Higher $Q_e(e)$ → More critical dependency requiring attention.

---

## 4. Metrics Definitions

### 4.1 PageRank

PageRank measures transitive importance based on incoming connections:

$$PR(v) = \frac{1-d}{N} + d \sum_{u \in B(v)} \frac{PR(u)}{L(u)}$$

Where:
- $d$ = damping factor (default: 0.85)
- $N$ = total number of nodes
- $B(v)$ = set of nodes linking to $v$ (predecessors)
- $L(u)$ = number of outgoing links from $u$

**In dependency graphs**: High PageRank indicates a component that many important components depend on.

### 4.2 Reverse PageRank (Failure Propagation)

Reverse PageRank is computed by running PageRank on the graph with reversed edge directions:

$$RP(v) = PR_{G^{-1}}(v)$$

Where $G^{-1}$ is the graph with all edges $(u, v)$ replaced by $(v, u)$.

**In dependency graphs**: High Reverse PageRank indicates a component whose failure would propagate to many important downstream components.

### 4.3 Betweenness Centrality

Betweenness measures how often a node lies on shortest paths:

$$BC(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

Where:
- $\sigma_{st}$ = number of shortest paths from $s$ to $t$
- $\sigma_{st}(v)$ = number of those paths passing through $v$

**In dependency graphs**: High betweenness indicates a bottleneck that many dependency chains pass through.

### 4.4 Clustering Coefficient

Local clustering coefficient measures neighbor connectivity:

$$CC(v) = \frac{2 \cdot e_v}{k_v(k_v - 1)}$$

Where:
- $e_v$ = number of edges between neighbors of $v$
- $k_v$ = degree of $v$

**Range**: [0, 1] where 1 means all neighbors are connected to each other.

**In dependency graphs**: High clustering indicates good local modularity — neighbors form cohesive groups.

### 4.5 Degree Centrality

Degree centrality normalizes the node degree:

$$DC(v) = \frac{deg(v)}{N - 1}$$

Where $deg(v)$ is the total degree (in + out for directed graphs).

### 4.6 Articulation Point

A node $v$ is an articulation point if removing it disconnects the graph:

$$AP(v) = \begin{cases} 1 & \text{if } |CC(G - \{v\})| > |CC(G)| \\ 0 & \text{otherwise} \end{cases}$$

Where $CC(G)$ denotes the set of connected components of graph $G$.

### 4.7 Bridge Edge

An edge $e$ is a bridge if removing it disconnects the graph:

$$BR_e(e) = \begin{cases} 1 & \text{if } |CC(G - \{e\})| > |CC(G)| \\ 0 & \text{otherwise} \end{cases}$$

### 4.8 Bridge Ratio

The bridge ratio for a node is the fraction of its incident edges that are bridges:

$$BR(v) = \frac{|\{e \in E : v \in e \land BR_e(e) = 1\}|}{deg(v)}$$

---

## 5. Normalization

All metrics are normalized to [0, 1] using min-max scaling within the analysis context:

$$x_{norm} = \frac{x - \min(X)}{\max(X) - \min(X)}$$

**Special Cases**:

| Condition | Handling |
|-----------|----------|
| $\max(X) = \min(X)$ | Return 0.5 for all values (uniform distribution) |
| Empty set | Return 0.0 |
| Single element | Return 0.5 |

**Context-Specific Normalization**:

Normalization is performed within each analysis scope to ensure fair comparison:

| Analysis Scope | Normalization Population |
|----------------|--------------------------|
| Application Layer | Only Application nodes |
| Infrastructure Layer | Only Node nodes |
| Complete System | All components together |

---

## 6. Box-Plot Classification

Instead of fixed thresholds, classification uses box-plot statistics derived from the actual score distribution:

```
                    │
                    │    ┌───────────┐
    CRITICAL ───────┼────│  outliers │──── score > Q3 + k×IQR
                    │    └───────────┘
                    │    ┌───────────┐
    HIGH ───────────┼────│    top    │──── score > Q3
                    │    │  quartile │
                    │    ├───────────┤
    MEDIUM ─────────┼────│   above   │──── score > Median
                    │    │  median   │
                    │    ├───────────┤
    LOW ────────────┼────│   below   │──── score > Q1
                    │    │  median   │
                    │    ├───────────┤
    MINIMAL ────────┼────│  bottom   │──── score ≤ Q1
                    │    │ quartile  │
                    │    └───────────┘
                    │
```

### Classification Rules

| Level | Condition | Interpretation |
|-------|-----------|----------------|
| **CRITICAL** | score > Q3 + k × IQR | Statistical outlier — requires immediate attention |
| **HIGH** | score > Q3 | Top quartile — high priority for review |
| **MEDIUM** | score > Median | Above average — monitor regularly |
| **LOW** | score > Q1 | Below average — standard monitoring |
| **MINIMAL** | score ≤ Q1 | Bottom quartile — low concern |

### Box-Plot Statistics

| Statistic | Definition |
|-----------|------------|
| Q1 | 25th percentile (first quartile) |
| Median (Q2) | 50th percentile |
| Q3 | 75th percentile (third quartile) |
| IQR | Interquartile range = Q3 - Q1 |
| k | Outlier factor (default: **1.5**) |

### Why Box-Plot Classification?

1. **Adaptive**: Thresholds adjust to each dataset's distribution
2. **No arbitrary cutoffs**: Avoids magic numbers like "0.7 is critical"
3. **Statistically grounded**: Based on well-understood descriptive statistics
4. **Handles scale differences**: Works regardless of absolute score magnitudes

---

## 7. Score Interpretation

### 7.1 Component Score Ranges

| Score Range | General Interpretation | Recommended Action |
|-------------|------------------------|-------------------|
| 0.8 - 1.0 | Extremely critical | Immediate review, add redundancy |
| 0.6 - 0.8 | Highly critical | Priority monitoring, consider refactoring |
| 0.4 - 0.6 | Moderately critical | Regular monitoring |
| 0.2 - 0.4 | Low criticality | Standard practices |
| 0.0 - 0.2 | Minimal criticality | No special attention needed |

### 7.2 Dimension-Specific Interpretation

**High Reliability Score** ($R > 0.7$):
- Component is heavily depended upon
- Failure would cascade widely
- **Action**: Implement circuit breakers, health checks, graceful degradation

**High Maintainability Score** ($M > 0.7$):
- Component is a coupling bottleneck
- Changes would propagate to many others
- **Action**: Consider splitting responsibilities, introduce abstraction layers

**High Availability Score** ($A > 0.7$):
- Component is a single point of failure
- No redundant paths exist
- **Action**: Add redundancy, implement failover mechanisms

### 7.3 Combined Patterns

| Pattern | R | M | A | Interpretation |
|---------|---|---|---|----------------|
| Hub | High | High | High | Critical integration point — top priority |
| Bottleneck | Low | High | Med | Coupling problem — refactor candidate |
| SPOF | Med | Low | High | Availability risk — add redundancy |
| Influencer | High | Low | Low | Important but well-designed — maintain |
| Leaf | Low | Low | Low | Edge component — minimal concern |

---

## 8. Layer-Specific Considerations

Quality metrics behave differently across architectural layers due to structural differences.

### 8.1 Application Layer

**Characteristics**:
- Dense pub-sub connectivity
- Many-to-many topic relationships
- Role differentiation (pub/sub/pubsub)

**Metric Behavior**:
| Metric | Typical Pattern |
|--------|-----------------|
| PageRank | Publishers often rank higher (data sources) |
| Betweenness | Processors (pubsub) tend to be bottlenecks |
| Clustering | Usually low due to topic-mediated communication |
| Articulation Points | Rare in well-designed systems |

**Recommended Focus**: Reliability and Maintainability

### 8.2 Infrastructure Layer

**Characteristics**:
- Sparser connectivity (physical/network topology)
- Aggregated from application dependencies
- Often follows hierarchical patterns

**Metric Behavior**:
| Metric | Typical Pattern |
|--------|-----------------|
| PageRank | Central nodes (gateways, aggregators) rank high |
| Betweenness | Network choke points score high |
| Clustering | Varies by topology (mesh vs. star) |
| Articulation Points | Common in cost-optimized topologies |

**Recommended Focus**: Availability (SPOF detection)

### 8.3 Cross-Layer Analysis

When analyzing the complete system:
- Normalize across all component types together
- Be aware that different types may cluster at different score ranges
- Consider type-specific breakdowns in addition to overall scores

---

## 9. Worked Example

### 9.1 Scenario

Consider a simple system with 3 applications:

| Component | PageRank | Rev. PageRank | In-Degree | Betweenness | Clustering | Degree | Is AP? | Bridge Ratio |
|-----------|----------|---------------|-----------|-------------|------------|--------|--------|--------------|
| App-A | 0.45 | 0.20 | 2 | 0.60 | 0.00 | 3 | No | 0.33 |
| App-B | 0.35 | 0.50 | 1 | 0.80 | 0.00 | 2 | Yes | 1.00 |
| App-C | 0.20 | 0.30 | 1 | 0.10 | 0.50 | 2 | No | 0.00 |

### 9.2 Normalization

For PageRank: min=0.20, max=0.45, range=0.25

| Component | $PR_{norm}$ | $RP_{norm}$ | $ID_{norm}$ | $BT_{norm}$ | $CC_{norm}$ | $DC_{norm}$ |
|-----------|-------------|-------------|-------------|-------------|-------------|-------------|
| App-A | 1.00 | 0.00 | 1.00 | 0.71 | 0.00 | 1.00 |
| App-B | 0.60 | 1.00 | 0.00 | 1.00 | 0.00 | 0.50 |
| App-C | 0.00 | 0.33 | 0.00 | 0.00 | 1.00 | 0.50 |

### 9.3 R, M, A Calculation (Using Default Weights)

**App-B Reliability**:
$$R(B) = 0.45 \times 0.60 + 0.35 \times 1.00 + 0.20 \times 0.00 = 0.27 + 0.35 + 0.00 = 0.62$$

**App-B Maintainability**:
$$M(B) = 0.45 \times 1.00 + 0.25 \times (1 - 0.00) + 0.30 \times 0.50 = 0.45 + 0.25 + 0.15 = 0.85$$

**App-B Availability** (with $CR = 0.60 \times 0.50 = 0.30$):
$$A(B) = 0.50 \times 1.0 + 0.25 \times 1.00 + 0.25 \times 0.30 = 0.50 + 0.25 + 0.075 = 0.825$$

**App-B Overall**:
$$Q(B) = 0.35 \times 0.62 + 0.30 \times 0.85 + 0.35 \times 0.825 = 0.217 + 0.255 + 0.289 = 0.761$$

### 9.4 Interpretation

App-B scores **0.761** overall, driven by:
- High Maintainability (0.85) — central bottleneck with high betweenness
- High Availability (0.825) — articulation point with all bridge edges
- Moderate Reliability (0.62) — significant failure propagation potential

**Recommendation**: App-B is a critical component requiring redundancy and careful change management.

---

## 10. Default Weights

### 10.1 Component Weights

```python
DEFAULT_WEIGHTS = {
    "reliability": {
        "pagerank": 0.45,           # ω_PR
        "reverse_pagerank": 0.35,   # ω_RP (failure propagation)
        "in_degree": 0.20,          # ω_ID
        # Sum: 1.00
    },
    "maintainability": {
        "betweenness": 0.45,        # ω_BT
        "clustering": 0.25,         # ω_CC (inverted in formula)
        "degree": 0.30,             # ω_DC
        # Sum: 1.00
    },
    "availability": {
        "articulation_point": 0.50, # ω_AP
        "bridge_ratio": 0.25,       # ω_BR
        "criticality": 0.25,        # ω_CR
        # Sum: 1.00
    },
    "overall": {
        "reliability": 0.35,        # α
        "maintainability": 0.30,    # β
        "availability": 0.35,       # γ
        # Sum: 1.00
    }
}
```

### 10.2 Edge Weights

```python
DEFAULT_EDGE_WEIGHTS = {
    "reliability": {
        "weight": 0.40,             # ω_w
        "endpoint_avg": 0.60,       # ω_ep
        # Sum: 1.00
    },
    "availability": {
        "bridge": 0.60,             # ω_br
        "endpoint_avg": 0.40,       # ω_ep
        # Sum: 1.00
    },
    "overall": {
        "reliability": 0.50,        # ω_R
        "availability": 0.50,       # ω_A
        # Sum: 1.00
    }
}
```

### 10.3 Classification Parameters

```python
CLASSIFICATION_PARAMS = {
    "k_factor": 1.5,  # Standard box-plot outlier threshold
}
```

---

## 11. Validation Targets

The quality formulations are validated against simulation results. The following targets define acceptable model performance:

| Metric | Target | Description |
|--------|--------|-------------|
| **Spearman Correlation** | ≥ 0.70 | Rank correlation between predicted scores and actual impact |
| **F1 Score** | ≥ 0.80 | Accuracy of critical component identification |
| **Precision** | ≥ 0.80 | Proportion of predicted critical components that are truly critical |
| **Recall** | ≥ 0.80 | Proportion of truly critical components that are predicted |
| **Top-5 Overlap** | ≥ 0.60 | Agreement on the 5 most critical components |
| **Top-10 Overlap** | ≥ 0.50 | Agreement on the 10 most critical components |

**Validation Process**:
1. Compute predicted criticality scores using formulas (1)-(4)
2. Run exhaustive failure simulation to measure actual impact
3. Compare predicted vs. actual using correlation and classification metrics
4. Model passes validation if targets are met

See [Validation Documentation](validation.md) for detailed methodology.

---

## 12. Implementation

### 12.1 Key Files

| File | Purpose |
|------|---------|
| `src/analysis/structural_analyzer.py` | Computes raw topological metrics using NetworkX |
| `src/analysis/quality_analyzer.py` | Implements R, M, A formulas and classification |
| `src/analysis/classifier.py` | Box-plot classification logic |
| `src/analysis/problem_detector.py` | Identifies architectural issues from scores |

### 12.2 CLI Usage

```bash
# Full quality assessment (all layers)
python analyze_graph.py --all

# Analyze specific layer
python analyze_graph.py --layer application
python analyze_graph.py --layer infrastructure

# Analyze specific component type
python analyze_graph.py --type Application

# Export results to JSON
python analyze_graph.py --output results/analysis.json

# Run with validation
python validate_graph.py --layer complete
```

### 12.3 Programmatic Usage

```python
from src.analysis.analyzer import GraphAnalyzer

with GraphAnalyzer(uri="bolt://localhost:7687") as analyzer:
    # Full system analysis
    results = analyzer.analyze()
    
    # Access component scores
    for comp in results["results"].components:
        print(f"{comp.id}: R={comp.scores.reliability:.3f}, "
              f"M={comp.scores.maintainability:.3f}, "
              f"A={comp.scores.availability:.3f}, "
              f"Q={comp.scores.overall:.3f}")
    
    # Access classification
    critical = [c for c in results["results"].components 
                if c.levels.overall.value == "critical"]
```

---

## 13. References

### 13.1 Graph Metrics

1. **PageRank**: Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank Citation Ranking: Bringing Order to the Web*. Stanford InfoLab.

2. **Betweenness Centrality**: Freeman, L. C. (1977). *A Set of Measures of Centrality Based on Betweenness*. Sociometry, 40(1), 35-41.

3. **Clustering Coefficient**: Watts, D. J., & Strogatz, S. H. (1998). *Collective Dynamics of 'Small-World' Networks*. Nature, 393(6684), 440-442.

4. **Articulation Points & Bridges**: Tarjan, R. E. (1972). *Depth-First Search and Linear Graph Algorithms*. SIAM Journal on Computing, 1(2), 146-160.

### 13.2 Software Quality

5. **ISO/IEC 25010**: Systems and Software Engineering — Systems and Software Quality Requirements and Evaluation (SQuaRE).

6. **Graph-Based Software Analysis**: Zimmermann, T., & Nagappan, N. (2008). *Predicting Defects Using Network Analysis on Dependency Graphs*. ICSE '08.

### 13.3 Distributed Systems

7. **Pub-Sub Systems**: Eugster, P. T., Felber, P. A., Guerraoui, R., & Kermarrec, A. M. (2003). *The Many Faces of Publish/Subscribe*. ACM Computing Surveys, 35(2), 114-131.

---

## Navigation

- **Previous**: [Weight Calculations](weight-calculations.md)
- **Next**: [Validation](validation.md)
- **See Also**: [Graph Model](graph-model.md)