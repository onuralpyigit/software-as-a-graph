# Step 3: Predictive Analysis

**Assessing Reliability, Maintainability, and Availability Through Composite Quality Scores**

---

## Table of Contents

1. [Overview](#overview)
2. [From Metrics to Quality Scores](#from-metrics-to-quality-scores)
3. [RMA Score Formulations](#rma-score-formulations)
4. [Box-Plot Classification](#box-plot-classification)
5. [Problem Detection Framework](#problem-detection-framework)
6. [Using analyze_graph.py for Quality Assessment](#using-analyze_graphpy-for-quality-assessment)
7. [Interpreting RMA Scores](#interpreting-rma-scores)
8. [Architectural Anti-Patterns](#architectural-anti-patterns)
9. [Practical Problem Detection Examples](#practical-problem-detection-examples)
10. [Remediation Strategies](#remediation-strategies)
11. [Advanced Techniques](#advanced-techniques)
12. [Best Practices](#best-practices)

---

## Overview

Predictive Analysis is the third step in the Software-as-a-Graph methodology, transforming raw topological metrics (from Step 2) into composite quality scores that predict component criticality and identify systemic architectural problems **before deployment**.

### The Predictive Analysis Promise

```
┌──────────────────────────────────────────────────────────────┐
│         FROM TOPOLOGY TO ACTIONABLE PREDICTIONS              │
└──────────────────────────────────────────────────────────────┘

Traditional Approach:              Our Approach:
────────────────────────            ─────────────────────
Deploy → Monitor → Fail             Analyze → Predict → Fix
    ↓        ↓       ↓                  ↓         ↓       ↓
Reactive  Costly  Late             Proactive  Cheap  Early

❌ Manual inspection                ✅ Quantitative scoring
❌ Subjective judgment              ✅ Statistical classification
❌ Post-failure learning            ✅ Pre-deployment insights
❌ Expensive incidents              ✅ Preventive action
```

### What Predictive Analysis Provides

| Question | Answer | Method |
|----------|--------|--------|
| **Which components will cause cascading failures?** | Reliability Score R(v) | PageRank + Reverse PageRank + In-Degree |
| **Which components are difficult to maintain?** | Maintainability Score M(v) | Betweenness + Clustering + Degree |
| **Which components are single points of failure?** | Availability Score A(v) | Articulation Points + Bridges + Criticality |
| **What is the overall criticality?** | Quality Score Q(v) | Composite RMA score |
| **How do I classify components?** | Critical/High/Medium/Low | Box-plot statistical thresholds |
| **What architectural problems exist?** | Problem Reports | Pattern detection algorithms |

### The Three-Phase Process

```
Phase 1: SCORE COMPUTATION
┌─────────────────────────────────┐
│ Raw Metrics (Step 2)            │
│  ├─ PageRank: 0.0245            │
│  ├─ Betweenness: 0.3421         │
│  ├─ Articulation Point: True    │
│  └─ ...                          │
└─────────────────────────────────┘
           ↓
    Apply Formulas
    R(v) = ω₁·PR + ω₂·RP + ω₃·ID
    M(v) = ω₁·BT + ω₂·(1-CC) + ω₃·DC
    A(v) = ω₁·AP + ω₂·BR + ω₃·CR
    Q(v) = α·R + β·M + γ·A
           ↓
┌─────────────────────────────────┐
│ Quality Scores [0,1]            │
│  ├─ R(v) = 0.762               │
│  ├─ M(v) = 0.845               │
│  ├─ A(v) = 0.913               │
│  └─ Q(v) = 0.841               │
└─────────────────────────────────┘

Phase 2: CLASSIFICATION
┌─────────────────────────────────┐
│ Q(v) scores for all components  │
│  [0.234, 0.456, 0.678, ...]     │
└─────────────────────────────────┘
           ↓
    Box-Plot Analysis
    Q1, Q3, IQR = quartiles(Q)
    Critical: Q > Q3 + 1.5×IQR
    High: Q1 + 1.5×IQR < Q ≤ Q3 + 1.5×IQR
    Medium: Q1 < Q ≤ Q1 + 1.5×IQR
    Low: Q ≤ Q1
           ↓
┌─────────────────────────────────┐
│ Classification Labels           │
│  sensor_fusion: CRITICAL        │
│  planning_node: HIGH            │
│  logger: LOW                    │
└─────────────────────────────────┘

Phase 3: PROBLEM DETECTION
┌─────────────────────────────────┐
│ Scores + Classifications        │
└─────────────────────────────────┘
           ↓
    Pattern Matching
    ├─ Hub-and-Spoke Detection
    ├─ Circular Dependency Detection
    ├─ God Object Detection
    ├─ Bottleneck Detection
    └─ SPOF Detection
           ↓
┌─────────────────────────────────┐
│ Problem Reports                 │
│  ⚠️ 3 Single Points of Failure  │
│  ⚠️ 2 Hub-and-Spoke Patterns    │
│  ⚠️ 5 Maintenance Hotspots      │
└─────────────────────────────────┘
```

### Why Composite Scores?

Single metrics are insufficient:

**Example: Component A**
- High PageRank (0.89) → Seems critical
- But: Not articulation point, Low betweenness
- Conclusion: Important but replaceable

**Example: Component B**
- Moderate PageRank (0.56) → Seems less critical
- But: Articulation point, High betweenness
- Conclusion: Actually critical (SPOF + bottleneck)

**Composite scores capture multi-dimensional criticality** that single metrics miss.

---

## From Metrics to Quality Scores

### The Transformation Pipeline

```
RAW METRICS                    QUALITY DIMENSIONS
(Step 2 Output)                (Step 3 Output)

┌─────────────────┐            ┌──────────────────┐
│ PageRank        │────┐       │                  │
│ Reverse PR      │────┼──────▶│  Reliability     │
│ In-Degree       │────┘       │      R(v)        │
└─────────────────┘            │                  │
                               └──────────────────┘
┌─────────────────┐            ┌──────────────────┐
│ Betweenness     │────┐       │                  │
│ Clustering      │────┼──────▶│ Maintainability  │
│ Degree          │────┘       │      M(v)        │
└─────────────────┘            │                  │
                               └──────────────────┘
┌─────────────────┐            ┌──────────────────┐
│ Articulation Pt │────┐       │                  │
│ Bridge Ratio    │────┼──────▶│  Availability    │
│ Criticality     │────┘       │      A(v)        │
└─────────────────┘            │                  │
                               └──────────────────┘
                                       │
                                       │ Weighted
                                       │ Combination
                                       ▼
                               ┌──────────────────┐
                               │  Overall Quality │
                               │      Q(v)        │
                               └──────────────────┘
```

### Design Principles

#### Principle 1: Separation of Concerns

Each quality dimension addresses orthogonal concerns:

| Dimension | Focus | Risk Type |
|-----------|-------|-----------|
| **Reliability** | Fault propagation | Runtime failure cascades |
| **Maintainability** | Change propagation | Development complexity |
| **Availability** | Service continuity | Downtime and interruptions |

A component can be:
- High reliability risk but low availability risk (many dependents, but alternatives exist)
- High maintainability risk but low reliability risk (complex but not critical)
- High availability risk but low maintainability risk (SPOF but simple)

#### Principle 2: Weighted Aggregation

Multiple metrics contribute to each dimension with tunable weights:

$$R(v) = \omega_{PR} \cdot PR_{norm}(v) + \omega_{RP} \cdot RP_{norm}(v) + \omega_{ID} \cdot ID_{norm}(v)$$

Where $\omega_{PR} + \omega_{RP} + \omega_{ID} = 1.0$

**Rationale**: Different metrics have different importance. Weights allow domain-specific tuning.

#### Principle 3: Normalization

All metrics normalized to [0, 1] before aggregation:

$$\text{metric}_{norm}(v) = \frac{\text{metric}(v) - \min(\text{metric})}{\max(\text{metric}) - \min(\text{metric})}$$

**Rationale**: Ensures comparable scales (e.g., PageRank [0.001, 0.045] vs Degree [1, 34]).

#### Principle 4: Adaptive Classification

Box-plot thresholds adapt to data distribution:

```
Instead of:  Q > 0.75 is "critical"  (arbitrary)
We use:      Q > Q3 + 1.5×IQR        (statistical)
```

**Rationale**: Avoids hard thresholds that fail in different systems.

---

## RMA Score Formulations

### Reliability Score R(v)

**Purpose**: Predicts fault propagation and system-wide failure impact.

**Formula**:
$$R(v) = \omega_{PR} \cdot PR_{norm}(v) + \omega_{RP} \cdot RP_{norm}(v) + \omega_{ID} \cdot ID_{norm}(v)$$

**Default Weights**:
- $\omega_{PR} = 0.45$ (PageRank)
- $\omega_{RP} = 0.35$ (Reverse PageRank)
- $\omega_{ID} = 0.20$ (In-Degree)

**Component Explanations**:

| Component | Weight | Contribution | Why It Matters |
|-----------|--------|--------------|----------------|
| **PageRank** | 45% | Transitive importance | Components depended upon by important components are themselves important |
| **Reverse PageRank** | 35% | Failure propagation | Measures how many downstream components fail when this fails |
| **In-Degree** | 20% | Direct impact | Number of immediate dependents |

**Mathematical Intuition**:

If component $v$ fails:
- High $PR(v)$ → Important components lose a dependency
- High $RP(v)$ → Failure cascades affect many components
- High $ID(v)$ → Many components immediately affected

**Interpretation Scale**:

| R(v) Range | Classification | Interpretation |
|-----------|----------------|----------------|
| 0.80 - 1.00 | Extreme Risk | Critical reliability concern, cascading failure highly likely |
| 0.60 - 0.79 | High Risk | Significant failure propagation potential |
| 0.40 - 0.59 | Moderate Risk | Localized failure impact |
| 0.20 - 0.39 | Low Risk | Limited failure propagation |
| 0.00 - 0.19 | Minimal Risk | Negligible reliability impact |

**Example Calculation**:

```python
Component: sensor_fusion_node
  PageRank (normalized): 0.876
  Reverse PageRank (normalized): 0.943
  In-Degree (normalized): 0.785

R(sensor_fusion) = 0.45 × 0.876 + 0.35 × 0.943 + 0.20 × 0.785
                 = 0.394 + 0.330 + 0.157
                 = 0.881

Interpretation: EXTREME reliability risk
- If sensor_fusion fails, cascading impact affects 94.3% of reachable components
- High transitive importance (88th percentile)
- 78.5% of components depend on it directly or indirectly
```

### Maintainability Score M(v)

**Purpose**: Predicts change propagation risk and maintenance complexity.

**Formula**:
$$M(v) = \omega_{BT} \cdot BT_{norm}(v) + \omega_{CC} \cdot (1 - CC_{norm}(v)) + \omega_{DC} \cdot DC_{norm}(v)$$

**Default Weights**:
- $\omega_{BT} = 0.45$ (Betweenness)
- $\omega_{CC} = 0.25$ (Clustering, inverted)
- $\omega_{DC} = 0.30$ (Degree)

**Component Explanations**:

| Component | Weight | Contribution | Why It Matters |
|-----------|--------|--------------|----------------|
| **Betweenness** | 45% | Coupling indicator | High betweenness = changes affect many communication paths |
| **Clustering (inverted)** | 25% | Modularity measure | Low clustering = poor modularity, changes propagate widely |
| **Degree** | 30% | Interface complexity | High degree = many integration points to maintain |

**Note on Clustering Inversion**: The formula uses $(1 - CC_{norm})$ because:
- High clustering = Good modularity (neighbors interconnected)
- Low clustering = Poor modularity (neighbors isolated)
- We want **high M(v) to indicate high maintenance risk**, so we invert clustering

**Interpretation Scale**:

| M(v) Range | Classification | Interpretation |
|-----------|----------------|----------------|
| 0.80 - 1.00 | Very Hard to Maintain | Extreme coupling, changes propagate system-wide |
| 0.60 - 0.79 | Hard to Maintain | Significant change propagation risk |
| 0.40 - 0.59 | Moderate Complexity | Manageable maintenance burden |
| 0.20 - 0.39 | Easy to Maintain | Well-modularized, localized changes |
| 0.00 - 0.19 | Very Easy | Minimal coupling, isolated component |

**Example Calculation**:

```python
Component: gateway_aggregator
  Betweenness (normalized): 0.923
  Clustering (normalized): 0.123
  Degree (normalized): 0.856

M(gateway_aggregator) = 0.45 × 0.923 + 0.25 × (1 - 0.123) + 0.30 × 0.856
                      = 0.415 + 0.219 + 0.257
                      = 0.891

Interpretation: VERY HARD to maintain
- Extreme bottleneck (92nd percentile betweenness)
- Poor modularity (87.7% of neighbors don't interconnect)
- Complex interface (86th percentile degree)
- Changes to this component require careful system-wide testing
```

### Availability Score A(v)

**Purpose**: Predicts single point of failure (SPOF) risk and service continuity impact.

**Formula**:
$$A(v) = \omega_{AP} \cdot AP(v) + \omega_{BR} \cdot BR(v) + \omega_{CR} \cdot CR_{norm}(v)$$

Where:
$$CR(v) = PR_{norm}(v) \times DC_{norm}(v)$$

**Default Weights**:
- $\omega_{AP} = 0.50$ (Articulation Point)
- $\omega_{BR} = 0.25$ (Bridge Ratio)
- $\omega_{CR} = 0.25$ (Criticality)

**Component Explanations**:

| Component | Weight | Contribution | Why It Matters |
|-----------|--------|--------------|----------------|
| **Articulation Point** | 50% | Structural SPOF | Binary: removal disconnects graph entirely |
| **Bridge Ratio** | 25% | Edge criticality | Fraction of incident edges that are irreplaceable |
| **Criticality** | 25% | Hub importance | Product of influence (PR) and connectivity (DC) |

**Criticality Sub-Formula**:

$$CR(v) = PR_{norm}(v) \times DC_{norm}(v)$$

This multiplicative combination identifies nodes that are **both** globally important (high PageRank) **and** locally connected (high degree) – the "important hubs".

**Interpretation Scale**:

| A(v) Range | Classification | Interpretation |
|-----------|----------------|----------------|
| 0.80 - 1.00 | Critical SPOF | Removal guarantees service disruption |
| 0.60 - 0.79 | High SPOF Risk | Likely service degradation on failure |
| 0.40 - 0.59 | Moderate Risk | Some redundancy exists |
| 0.20 - 0.39 | Low Risk | Multiple alternatives available |
| 0.00 - 0.19 | Minimal Risk | Highly redundant or peripheral |

**Example Calculation**:

```python
Component: main_broker
  Articulation Point: True (1.0)
  Bridge Ratio: 0.833 (10 of 12 edges are bridges)
  PageRank (normalized): 0.678
  Degree (normalized): 0.712
  
  Criticality = 0.678 × 0.712 = 0.483

A(main_broker) = 0.50 × 1.0 + 0.25 × 0.833 + 0.25 × 0.483
               = 0.500 + 0.208 + 0.121
               = 0.829

Interpretation: CRITICAL SPOF
- Confirmed articulation point (removal disconnects graph)
- 83.3% of connections are irreplaceable bridges
- Important hub (67.8% PageRank, 71.2% connectivity)
- Zero redundancy – must implement failover
```

### Overall Quality Criticality Q(v)

**Purpose**: Composite score combining all quality dimensions into unified criticality measure.

**Formula**:
$$Q(v) = \alpha \cdot R(v) + \beta \cdot M(v) + \gamma \cdot A(v)$$

**Default Weights**:
- $\alpha = 0.35$ (Reliability)
- $\beta = 0.30$ (Maintainability)
- $\gamma = 0.35$ (Availability)

**Weight Rationale**:

| Dimension | Weight | Reasoning |
|-----------|--------|-----------|
| Reliability | 35% | Runtime stability concern (high priority) |
| Maintainability | 30% | Development concern (slightly lower priority) |
| Availability | 35% | Service continuity concern (high priority) |

**Note**: Reliability and Availability weighted equally (both runtime concerns), Maintainability slightly lower (development concern not immediate runtime risk).

**Domain-Specific Weight Tuning**:

```python
# Safety-critical system (prioritize reliability)
weights = {"reliability": 0.50, "maintainability": 0.20, "availability": 0.30}

# Rapidly evolving system (prioritize maintainability)
weights = {"reliability": 0.25, "maintainability": 0.50, "availability": 0.25}

# High-availability system (prioritize availability)
weights = {"reliability": 0.30, "maintainability": 0.20, "availability": 0.50}
```

**Complete Example**:

```python
Component: sensor_fusion_node
  R(v) = 0.881 (extreme reliability risk)
  M(v) = 0.645 (hard to maintain)
  A(v) = 0.913 (critical SPOF)

Q(sensor_fusion) = 0.35 × 0.881 + 0.30 × 0.645 + 0.35 × 0.913
                 = 0.308 + 0.194 + 0.320
                 = 0.822

Interpretation: CRITICAL component requiring immediate attention
- All three dimensions score high
- Balanced criticality across reliability, maintainability, availability
- Top priority for redundancy implementation
```

---

## Box-Plot Classification

### Overview

Box-plot classification uses statistical quartiles to adaptively classify components based on their quality scores, avoiding arbitrary fixed thresholds.

### The Box-Plot Method

```
        Lower          Lower    Median   Upper      Upper
        Outlier        Fence              Fence     Outlier
        Boundary        │                  │        Boundary
           │            │                  │           │
           ▼            ▼                  ▼           ▼
    ───────┼────────────┼──────────────────┼───────────┼───────
           │            │        Q2        │           │
           │     Q1     │                  │    Q3     │
           │◄──────────►│◄────────────────►│◄─────────►│
           │            │                  │           │
           │◄───────────┴──────────────────┴──────────►│
                            IQR (Interquartile Range)
                            
    ───────┴────────────┴──────────────────┴───────────┴───────
     LOW      MEDIUM            HIGH            CRITICAL
```

### Classification Formula

Given quality scores $Q = \{Q_1, Q_2, ..., Q_n\}$ for all components:

1. **Compute Quartiles**:
   - $Q1$ = 25th percentile of $Q$
   - $Q2$ = 50th percentile of $Q$ (median)
   - $Q3$ = 75th percentile of $Q$

2. **Compute Interquartile Range**:
   $$IQR = Q3 - Q1$$

3. **Define Classification Boundaries**:
   - **Critical**: $Q(v) > Q3 + k \times IQR$
   - **High**: $Q1 + k \times IQR < Q(v) \leq Q3 + k \times IQR$
   - **Medium**: $Q1 < Q(v) \leq Q1 + k \times IQR$
   - **Low**: $Q(v) \leq Q1$

Where $k = 1.5$ (standard Tukey's outlier detection factor)

### Why Box-Plot Classification?

#### Advantage 1: Adaptive to Data Distribution

**Fixed Threshold Approach** (Bad):
```python
if Q(v) > 0.75: return "critical"
elif Q(v) > 0.50: return "high"
elif Q(v) > 0.25: return "medium"
else: return "low"
```

**Problem**: System A might have max score 0.65, system B might have max 0.95. Same threshold doesn't fit both.

**Box-Plot Approach** (Good):
```python
Q1, Q3 = percentile(scores, [25, 75])
IQR = Q3 - Q1
critical_threshold = Q3 + 1.5 * IQR
```

**Benefit**: Threshold adapts to each system's score distribution.

#### Advantage 2: Statistically Grounded

Box-plot method is Tukey's standard outlier detection:
- Used in statistics for 50+ years
- Well-understood properties
- Handles skewed distributions
- Robust to extreme values

#### Advantage 3: Avoids "Sharp Boundary Problem"

**Sharp Boundary Problem**:
```
Component A: Q = 0.749 → "medium"
Component B: Q = 0.751 → "high"
```
Tiny difference (0.002) causes category jump.

**Box-Plot Solution**: Gradual transitions
- Low → Medium at Q1
- Medium → High at Q1 + 1.5×IQR
- High → Critical at Q3 + 1.5×IQR

**Future Enhancement**: Fuzzy logic for smooth transitions (allows partial membership in multiple categories).

### Classification Example

**System Scores**: [0.234, 0.312, 0.398, 0.456, 0.512, 0.567, 0.623, 0.689, 0.745, 0.823, 0.891]

**Step 1: Compute Quartiles**
```
Q1 = 0.398 (25th percentile)
Q2 = 0.567 (50th percentile, median)
Q3 = 0.745 (75th percentile)
```

**Step 2: Compute IQR**
```
IQR = Q3 - Q1 = 0.745 - 0.398 = 0.347
```

**Step 3: Compute Boundaries**
```
Critical boundary = Q3 + 1.5×IQR = 0.745 + 1.5×0.347 = 1.266 (capped at 1.0)
High boundary = Q1 + 1.5×IQR = 0.398 + 1.5×0.347 = 0.919
Medium boundary = Q1 = 0.398
```

**Step 4: Classify**
```
Q(v) > 0.919  → CRITICAL  (scores: 0.891 borderline)
0.398 < Q(v) ≤ 0.919 → HIGH (scores: 0.456, 0.512, 0.567, 0.623, 0.689, 0.745, 0.823)
Q(v) ≤ 0.398 → LOW (scores: 0.234, 0.312, 0.398)
```

**Result Distribution**:
- Critical: 0 components (none exceed 0.919)
- High: 7 components (64%)
- Medium: 0 components (collapsed due to data distribution)
- Low: 4 components (36%)

### Per-Dimension Classification

Classification can be applied to R, M, A individually:

```python
Component: gateway_node
  
  Overall Q(v) = 0.782 → Classification: HIGH
  
  But per-dimension:
    R(v) = 0.634 → Classification: MEDIUM
    M(v) = 0.891 → Classification: CRITICAL
    A(v) = 0.823 → Classification: HIGH
    
Insight: Maintenance is the primary concern (critical),
         availability is secondary (high),
         reliability is manageable (medium)
         
Action: Focus on refactoring for maintainability
```

---

## Problem Detection Framework

### Overview

Beyond individual component scores, the framework detects **systemic architectural problems** through pattern matching on quality scores and graph structure.

### Problem Categories

```
┌─────────────────────────────────────────────────────────────┐
│                  PROBLEM DETECTION TAXONOMY                  │
└─────────────────────────────────────────────────────────────┘

1. STRUCTURAL PROBLEMS
   ├─ Single Points of Failure (SPOFs)
   ├─ Hub-and-Spoke Patterns
   ├─ Circular Dependencies
   └─ Disconnected Components

2. COMPLEXITY PROBLEMS
   ├─ God Objects / God Services
   ├─ Bottleneck Mediators
   ├─ Interface Complexity
   └─ Deep Dependency Chains

3. QUALITY PROBLEMS
   ├─ Reliability Hotspots
   ├─ Maintainability Hotspots
   ├─ Availability Hotspots
   └─ Balanced vs Imbalanced Criticality

4. SYSTEMIC PROBLEMS
   ├─ Cascading Failure Potential
   ├─ Change Propagation Risk
   ├─ Service Disruption Risk
   └─ Architectural Drift
```

### Detection Algorithms

#### Problem 1: Single Point of Failure (SPOF)

**Definition**: Component whose failure causes service disruption or graph disconnection.

**Detection Criteria**:
```python
def detect_spof(component):
    return (
        component.metrics.is_articulation_point == True or
        component.scores.availability > 0.80 or
        (component.scores.overall > 0.70 and 
         component.metrics.bridge_ratio > 0.60)
    )
```

**Severity Levels**:
- **Critical**: Articulation point + A(v) > 0.80
- **High**: Articulation point OR A(v) > 0.70
- **Medium**: High bridge ratio (>0.60)

**Symptoms**:
```
⚠️ Component 'main_broker' is a CRITICAL SPOF
   - Articulation Point: Yes
   - Availability Score: 0.913
   - Bridge Ratio: 0.833
   - Impact: Removal disconnects 34 of 40 components
   - Recommendation: Deploy standby replica with automated failover
```

#### Problem 2: Hub-and-Spoke Pattern

**Definition**: Centralized coordinator that all other components depend on.

**Detection Criteria**:
```python
def detect_hub_and_spoke(component):
    total_components = len(all_components)
    return (
        component.metrics.betweenness > 0.80 and
        component.metrics.degree_total > 0.70 * total_components and
        component.scores.maintainability > 0.75
    )
```

**Characteristics**:
- Very high betweenness (>0.80)
- Connected to most components (>70%)
- High maintainability score (>0.75)

**Symptoms**:
```
⚠️ Hub-and-Spoke anti-pattern detected
   - Hub: 'central_coordinator'
   - Betweenness: 0.923 (extreme)
   - Connected to: 38 of 40 components (95%)
   - Maintainability Score: 0.891 (critical)
   - Impact: Bottleneck for all communication, hard to maintain
   - Recommendation: Refactor to event-driven or peer-to-peer architecture
```

### Problem Severity Scoring

Each detected problem receives a severity score:

$$\text{Severity} = w_1 \cdot \text{affected\_count} + w_2 \cdot \text{avg\_criticality} + w_3 \cdot \text{pattern\_weight}$$

**Example**:
```python
Problem: SPOF (main_broker)
  Affected Components: 34
  Average Criticality: 0.823
  Pattern Weight: 1.0 (SPOF is most severe)
  
  Severity = 0.4 × (34/40) + 0.3 × 0.823 + 0.3 × 1.0
           = 0.340 + 0.247 + 0.300
           = 0.887 (CRITICAL)
```

---

## Summary

**Step 3: Predictive Analysis** transforms structural metrics into actionable quality assessments and problem reports:

✅ **RMA Scores**: Composite measures of Reliability, Maintainability, Availability

✅ **Box-Plot Classification**: Adaptive statistical thresholds (Critical/High/Medium/Low)

✅ **Problem Detection**: Identifies SPOFs, bottlenecks, anti-patterns, coupling issues

✅ **Actionable Insights**: Specific remediation recommendations with severity

✅ **Pre-Deployment**: Predict critical components **before** failures occur

### Key Formulas

$$R(v) = 0.45 \cdot PR + 0.35 \cdot RP + 0.20 \cdot ID$$
$$M(v) = 0.45 \cdot BT + 0.25 \cdot (1-CC) + 0.30 \cdot DC$$
$$A(v) = 0.50 \cdot AP + 0.25 \cdot BR + 0.25 \cdot CR$$
$$Q(v) = 0.35 \cdot R + 0.30 \cdot M + 0.35 \cdot A$$

---

**Last Updated**: January 2025  
**Part of**: Software-as-a-Graph Research Project  
**Institution**: Istanbul Technical University
