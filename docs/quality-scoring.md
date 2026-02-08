# Step 3: Quality Scoring

**Combine topological metrics into composite quality scores using AHP-derived weights**

---

## Overview

Quality Scoring transforms raw graph metrics into four quality dimensions: Reliability (R), Maintainability (M), Availability (A), and Vulnerability (V).

```
┌─────────────────────┐          ┌─────────────────────┐
│  Structural Metrics │          │  Quality Scores     │
│                     │    →     │                     │
│  - PageRank         │          │  R(v) Reliability   │
│  - Betweenness      │          │  M(v) Maintainability│
│  - Articulation Pt  │          │  A(v) Availability  │
│  - Eigenvector      │          │  V(v) Vulnerability │
│  - ...              │          │  Q(v) Overall       │
└─────────────────────┘          └─────────────────────┘
```

---

## The Four Quality Dimensions

| Dimension | Focus | Key Question |
|-----------|-------|--------------|
| **R** (Reliability) | Fault propagation | What happens if this fails? |
| **M** (Maintainability) | Coupling complexity | How hard is this to change? |
| **A** (Availability) | SPOF risk | Is this irreplaceable? |
| **V** (Vulnerability) | Security exposure | Is this an attractive target? |

---

## Design Principles

### Metric Orthogonality

Each raw structural metric is used by **at most one** RMAV dimension. This prevents
any single metric from accumulating disproportionate effective weight in the overall
composite score Q(v), and ensures the four dimensions capture genuinely independent
quality concerns.

| Metric | Used In | Not Used In | Rationale |
|--------|---------|-------------|-----------|
| PageRank | R(v) | — | Transitive influence → fault propagation |
| Reverse PageRank | R(v) | — | Failure cascade direction |
| In-Degree | R(v) | — | Direct dependents count |
| Betweenness | M(v) | — | Bottleneck / coupling position |
| Out-Degree | M(v), V(v) | — | Efferent coupling (M) / attack surface (V) |
| Clustering | M(v) | — | Local modularity |
| AP Score (continuous) | A(v) | — | Structural SPOF risk |
| Bridge Ratio | A(v) | — | Irreplaceable connections |
| Importance (PR+RPR)/2 | A(v) | — | Critical hub proxy |
| Eigenvector | V(v) | — | Strategic importance (high-value target) |
| Closeness | V(v) | — | Propagation speed |

> **Historical note**: Prior versions used In-Degree in both R(v) and V(v), giving it
> an effective weight of ~13.75% in Q(v) — more than PageRank (10%). The current design
> replaces V(v)'s In-Degree with Out-Degree, which better captures the concept of
> "attack surface" (number of outbound connections an attacker can traverse) while
> keeping the dimensions orthogonal.

### Continuous over Binary

All metrics are continuous in [0, 1]. The binary Articulation Point flag has been
replaced by a continuous fragmentation score (see Availability below) to avoid
step-function discontinuities in the scoring.

---

## Quality Formulas

### Reliability R(v)

Measures fault propagation risk: _What happens if this component fails?_

```
R(v) = w₁×PR + w₂×RPR + w₃×ID
```

| Component | Default Weight | Rationale |
|-----------|----------------|-----------|
| PageRank | 0.40 | Transitive influence |
| Reverse PageRank | 0.35 | Failure cascade |
| In-Degree | 0.25 | Direct dependents |

**Interpretation**: High R(v) → Greater reliability risk if component fails.

---

### Maintainability M(v)

Measures coupling complexity and change risk: _How hard is this component to change?_

```
M(v) = w₁×BT + w₂×OD + w₃×(1-CC)
```

| Component | Default Weight | Rationale |
|-----------|----------------|-----------|
| Betweenness | 0.40 | Bottleneck position |
| Out-Degree | 0.35 | Efferent coupling (change fragility) |
| (1 - Clustering) | 0.25 | Poor modularity |

**Note**: Clustering is inverted—high clustering = better maintainability.

**Design rationale**: Out-Degree (efferent coupling) measures how many components this
one _depends on_, which directly affects change fragility — modifying any upstream
dependency can break this component. This aligns with Martin's Instability metric
`I = Ce / (Ca + Ce)` from software engineering literature. The previous formula used
Total Degree (in + out), which conflated _impact of change_ (already captured by R(v)
via In-Degree and PageRank) with _fragility to change_.

**Interpretation**: High M(v) → Harder to maintain, higher change risk.

---

### Availability A(v)

Measures single point of failure (SPOF) risk: _Is this component irreplaceable?_

```
A(v) = w₁×AP_c + w₂×BR + w₃×Importance
```

Where:
- `AP_c(v)` = Continuous articulation point score (see below)
- `BR(v)` = Bridge ratio
- `Importance = (PR + RPR) / 2`

| Component | Default Weight | Rationale |
|-----------|----------------|-----------|
| AP Score (continuous) | 0.50 | Structural SPOF severity |
| Bridge Ratio | 0.30 | Irreplaceable connections |
| Importance | 0.20 | Critical hub |

#### Continuous Articulation Point Score

The binary AP flag ({0, 1}) creates a step-function discontinuity: a node that _almost_
disconnects the graph receives the same score (0.0) as a leaf node. We replace it with
a continuous metric based on reachability loss:

```
AP_c(v) = 1 - |largest_CC(G \ {v})| / (|V| - 1)
```

Where `largest_CC(G \ {v})` is the largest connected component after removing v.

| Scenario | AP_c(v) | Explanation |
|----------|---------|-------------|
| True AP (splits into equal halves) | 0.50 | Half the nodes become unreachable |
| True AP (splits off 1 node) | 1/(n-1) | Minimal fragmentation |
| Near-AP (doesn't disconnect) | 0.0 | But bridge_ratio may still be high |
| Leaf node | 0.0 | Removal doesn't affect connectivity |

**Complexity**: O(|V| × (|V| + |E|)) — acceptable for target scale (|V| < 1000).

**Interpretation**: High A(v) → Higher availability risk.

---

### Vulnerability V(v)

Measures security exposure and attack surface: _Is this an attractive target?_

```
V(v) = w₁×EV + w₂×CL + w₃×OD
```

| Component | Default Weight | Rationale |
|-----------|----------------|-----------|
| Eigenvector | 0.40 | Strategic importance (high-value target) |
| Closeness | 0.30 | Propagation speed (blast radius) |
| Out-Degree | 0.30 | Attack surface (reachable via outbound edges) |

**Design rationale**: Out-Degree captures the number of outbound connections an attacker
can traverse after compromising this component, making it a natural measure of attack
surface. In-Degree (used in the previous version) measures how many components _depend
on_ this one — that's a reliability/impact concern, not a vulnerability one.

**Interpretation**: High V(v) → Higher security risk.

---

### Overall Quality Q(v)

Combines all dimensions:

```
Q(v) = w_R×R(v) + w_M×M(v) + w_A×A(v) + w_V×V(v)
```

Default: Equal weights (0.25 each).

---

## Edge Quality Scoring

Edges are scored using a simplified RMAV-aligned formula that combines edge-intrinsic
metrics with endpoint component quality scores.

```
R_e = w₁×EB + w₂×W + w₃×max(R_src, R_tgt)
M_e = w₁×EB + w₂×Bridge + w₃×W
A_e = w₁×Bridge + w₂×min(A_src, A_tgt)
V_e = w₁×W + w₂×max(V_src, V_tgt)
```

| Symbol | Meaning |
|--------|---------|
| EB | Edge betweenness centrality |
| W | Edge weight (dependency strength) |
| Bridge | 1.0 if bridge edge, 0.0 otherwise |
| R_src, R_tgt | Reliability scores of source/target components |
| A_src, A_tgt | Availability scores of source/target components |
| V_src, V_tgt | Vulnerability scores of source/target components |

**Note**: Edge scoring incorporates endpoint quality scores to propagate component-level
criticality into the edge analysis, ensuring that edges connecting critical components
are themselves flagged as critical.

---

## AHP Weight Calculation

The Analytic Hierarchy Process (AHP) derives weights from expert pairwise comparisons.

### Saaty's Scale

| Value | Meaning |
|-------|---------|
| 1 | Equal importance |
| 3 | Moderate importance |
| 5 | Strong importance |
| 7 | Very strong importance |
| 9 | Extreme importance |

### Pairwise Comparison Matrix

Example for **Availability** (AP_c, BR, Importance):

```
          AP_c   BR    IM
   AP_c [ 1.0   3.0   5.0 ]   ← AP_c strongly more important than IM
   BR   [ 0.33  1.0   2.0 ]
   IM   [ 0.20  0.5   1.0 ]
```

### Weight Calculation (Geometric Mean)

1. **Compute geometric mean of each row:**
   ```
   GM_AP = ∛(1.0 × 3.0 × 5.0) ≈ 2.47
   GM_BR = ∛(0.33 × 1.0 × 2.0) ≈ 0.87
   GM_IM = ∛(0.20 × 0.5 × 1.0) ≈ 0.46
   ```

2. **Normalize:**
   ```
   Total = 2.47 + 0.87 + 0.46 = 3.80
   
   w_AP = 2.47 / 3.80 ≈ 0.65
   w_BR = 0.87 / 3.80 ≈ 0.23
   w_IM = 0.46 / 3.80 ≈ 0.12
   ```

### Consistency Check

AHP validates matrix consistency using Consistency Ratio (CR):

```
CR = CI / RI    (should be < 0.10)
```

---

## Default AHP Matrices

**Reliability** (PR, RPR, ID):
```
[ 1.0   2.0   2.0 ]
[ 0.5   1.0   1.0 ]
[ 0.5   1.0   1.0 ]
```

**Maintainability** (BT, OD, CC):
```
[ 1.0   2.0   3.0 ]
[ 0.5   1.0   2.0 ]
[ 0.33  0.5   1.0 ]
```

**Availability** (AP_c, BR, IM):
```
[ 1.0   3.0   5.0 ]
[ 0.33  1.0   2.0 ]
[ 0.2   0.5   1.0 ]
```

**Vulnerability** (EV, CL, OD):
```
[ 1.0   2.0   2.0 ]
[ 0.5   1.0   1.0 ]
[ 0.5   1.0   1.0 ]
```

---

## Normalization

Quality Scoring normalizes raw metrics to [0, 1] before applying formulas.

### Max-Normalization (Default)

```
x_norm(v) = x(v) / max(x(u) for all u ∈ V_l)
```

When `max(x) = 0` (all values zero), the normalizer returns 0.0 (safe division guard).

### Robust Normalization (Optional)

For distributions with extreme outliers that compress the majority of values into a
narrow range, a rank-based normalization is available:

```
x_norm(v) = rank(x(v)) / |V_l|
```

This preserves ordinal relationships while distributing values uniformly across [0, 1].
Use `--normalize robust` or set `normalization_method="robust"` in the API.

---

## Box-Plot Classification

Components are classified using statistical quartiles (not arbitrary thresholds):

```
                Score Distribution
    ┌───────────────────────────────────────────┐
    │                                           │
 MINIMAL    LOW     MEDIUM    HIGH     CRITICAL
    │        │         │        │          │
   Q1     Median      Q3    Q3+1.5×IQR  outliers
```

| Level | Condition | Interpretation |
|-------|-----------|----------------|
| **CRITICAL** | Q > Q3 + 1.5×IQR | Statistical outlier |
| **HIGH** | Q > Q3 | Top quartile |
| **MEDIUM** | Q > Median | Above average |
| **LOW** | Q > Q1 | Below average |
| **MINIMAL** | Q ≤ Q1 | Bottom quartile |

### Small-Sample Fallback

Box-plot quartile estimation requires sufficient samples for statistical stability.
When `|V_l| < 12`, the classifier falls back to fixed percentile thresholds:

| Level | Percentile Rank |
|-------|-----------------|
| **CRITICAL** | Top 10% |
| **HIGH** | Top 25% |
| **MEDIUM** | Top 50% |
| **LOW** | Top 75% |
| **MINIMAL** | Bottom 25% |

This prevents unreliable quartile estimates from producing misleading classifications
in small subsystems or single-layer analyses with few components.

---

## Interpretation Patterns

| Pattern | R | M | A | V | Meaning | Action |
|---------|---|---|---|---|---------|--------|
| **Hub** | High | High | High | High | Critical integration point | Redundancy + monitoring |
| **Bottleneck** | Low | High | Med | Med | Coupling problem | Refactor |
| **SPOF** | Med | Low | High | Low | Single point of failure | Add redundancy |
| **Target** | Low | Low | Low | High | Security exposure | Harden |
| **Leaf** | Low | Low | Low | Low | Low concern | Standard practices |

---

## Weight Sensitivity Analysis

AHP weights are derived from expert judgment and carry inherent subjectivity. The
optional sensitivity analysis module measures how robust the final rankings are to
small weight perturbations:

```bash
# Run sensitivity analysis with 200 perturbations, ±5% noise
python bin/analyze_graph.py --layer system --use-ahp --sensitivity --perturbations 200 --noise 0.05
```

### Method

1. Perturb each AHP weight by Gaussian noise (σ = 0.05 by default)
2. Re-normalize weights to maintain sum-to-one constraints
3. Recompute Q(v) rankings for each perturbation
4. Report stability metrics:

| Metric | Meaning | Good Threshold |
|--------|---------|----------------|
| Top-5 Stability | Fraction of trials where top-5 set is unchanged | ≥ 0.80 |
| Mean Kendall τ | Average rank correlation with original | ≥ 0.90 |
| Std Kendall τ | Variability of rank correlation | ≤ 0.05 |

High stability means the methodology is robust to expert judgment uncertainty.
Low stability suggests the AHP matrices should be reviewed or that alternative
weight configurations should be explored.

---

## Commands

```bash
# Default weights
python bin/analyze_graph.py --layer system

# Use AHP-derived weights
python bin/analyze_graph.py --layer system --use-ahp

# Robust normalization
python bin/analyze_graph.py --layer system --use-ahp --normalize robust

# Weight sensitivity analysis
python bin/analyze_graph.py --layer system --use-ahp --sensitivity

# Export results
python bin/analyze_graph.py --layer system --use-ahp --output results/quality.json
```

---

## Output Example

```
═══════════════════════════════════════════════════════════════
  QUALITY SCORING - System Layer
═══════════════════════════════════════════════════════════════

  Top Components by Q(v):
  
  Component          Type        R      M      A      V      Q     Level
  ─────────────────────────────────────────────────────────────────────
  sensor_fusion      Application 0.82   0.88   0.90   0.75   0.84  CRITICAL
  main_broker        Broker      0.78   0.65   0.95   0.80   0.80  CRITICAL
  planning_node      Application 0.71   0.73   0.45   0.68   0.64  HIGH
  
  Classification Summary:
    CRITICAL:  5 components
    HIGH:      8 components
    MEDIUM:   15 components
    LOW:      12 components
    MINIMAL:   8 components
    
  Weight Sensitivity (200 perturbations, σ=0.05):
    Top-5 Stability: 0.92
    Mean Kendall τ:  0.96
    Std Kendall τ:   0.02
```

---

## Custom Weights

```python
from src.domain.services import AHPMatrices, AHPProcessor, QualityAnalyzer

# Define custom comparison matrices
matrices = AHPMatrices(
    criteria_availability=[
        [1.0, 5.0, 7.0],  # AP_c strongly dominates
        [0.2, 1.0, 3.0],
        [0.14, 0.33, 1.0],
    ]
)

processor = AHPProcessor(matrices)
weights = processor.compute_weights()

analyzer = QualityAnalyzer(weights=weights)
```

---

## Next Step

→ [Step 4: Failure Simulation](failure-simulation.md)