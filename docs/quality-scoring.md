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

## Quality Formulas

### Reliability R(v)

Measures fault propagation risk.

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

Measures coupling complexity and change risk.

```
M(v) = w₁×BT + w₂×DG + w₃×(1-CC)
```

| Component | Default Weight | Rationale |
|-----------|----------------|-----------|
| Betweenness | 0.40 | Bottleneck position |
| Degree | 0.35 | Interface complexity |
| (1 - Clustering) | 0.25 | Poor modularity |

**Note**: Clustering is inverted—high clustering = better maintainability.

**Interpretation**: High M(v) → Harder to maintain, higher change risk.

---

### Availability A(v)

Measures single point of failure (SPOF) risk.

```
A(v) = w₁×AP + w₂×BR + w₃×Importance
```

Where: `Importance = (PR + RPR) / 2`

| Component | Default Weight | Rationale |
|-----------|----------------|-----------|
| Articulation Point | 0.50 | Structural SPOF |
| Bridge Ratio | 0.30 | Irreplaceable connections |
| Importance | 0.20 | Critical hub |

**Interpretation**: High A(v) → Higher availability risk.

---

### Vulnerability V(v)

Measures security exposure and attack surface.

```
V(v) = w₁×EV + w₂×CL + w₃×ID
```

| Component | Default Weight | Rationale |
|-----------|----------------|-----------|
| Eigenvector | 0.40 | Strategic importance (high-value target) |
| Closeness | 0.30 | Propagation speed |
| In-Degree | 0.30 | Attack surface |

**Interpretation**: High V(v) → Higher security risk.

---

### Overall Quality Q(v)

Combines all dimensions:

```
Q(v) = w_R×R(v) + w_M×M(v) + w_A×A(v) + w_V×V(v)
```

Default: Equal weights (0.25 each).

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

Example for **Availability** (AP, BR, Importance):

```
         AP    BR    IM
   AP  [ 1.0   3.0   5.0 ]   ← AP strongly more important than IM
   BR  [ 0.33  1.0   2.0 ]
   IM  [ 0.20  0.5   1.0 ]
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

**Maintainability** (BT, DG, CC):
```
[ 1.0   2.0   3.0 ]
[ 0.5   1.0   2.0 ]
[ 0.33  0.5   1.0 ]
```

**Availability** (AP, BR, IM):
```
[ 1.0   3.0   5.0 ]
[ 0.33  1.0   2.0 ]
[ 0.2   0.5   1.0 ]
```

**Vulnerability** (EV, CL, ID):
```
[ 1.0   2.0   2.0 ]
[ 0.5   1.0   1.0 ]
[ 0.5   1.0   1.0 ]
```

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

## Commands

```bash
# Default weights
python analyze_graph.py --layer system

# Use AHP-derived weights
python analyze_graph.py --layer system --use-ahp

# Export results
python analyze_graph.py --layer system --use-ahp --output results/quality.json
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
```

---

## Custom Weights

```python
from src.analysis import AHPMatrices, AHPProcessor, QualityAnalyzer

# Define custom comparison matrices
matrices = AHPMatrices(
    criteria_availability=[
        [1.0, 5.0, 7.0],  # AP strongly dominates
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
