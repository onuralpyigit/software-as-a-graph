# Step 3: Quality Scoring

**Map 13 raw structural metrics into four interpretable quality dimensions and a single criticality classification per component.**

← [Step 2: Structural Analysis](structural-analysis.md) | → [Step 4: Failure Simulation](failure-simulation.md)

---

## Table of Contents

1. [What This Step Does](#what-this-step-does)
2. [The Four Quality Dimensions](#the-four-quality-dimensions)
3. [RMAV Formulas](#rmav-formulas)
   - [Reliability R(v) — Fault Propagation Risk](#reliability-rv--fault-propagation-risk)
   - [Maintainability M(v) — Coupling Complexity](#maintainability-mv--coupling-complexity)
   - [Availability A(v) — SPOF Risk](#availability-av--spof-risk)
   - [Vulnerability V(v) — Security Exposure](#vulnerability-vv--security-exposure)
   - [Overall Quality Q(v)](#overall-quality-qv)
4. [Metric Orthogonality](#metric-orthogonality)
5. [Classification](#classification)
   - [Box-Plot Thresholds (Normal Path)](#box-plot-thresholds-normal-path)
   - [Small-Sample Percentile Fallback](#small-sample-percentile-fallback)
6. [AHP Weight Derivation](#ahp-weight-derivation)
   - [The AHP Procedure](#the-ahp-procedure)
   - [Default Pairwise Matrices](#default-pairwise-matrices)
   - [Weight Sensitivity Analysis](#weight-sensitivity-analysis)
7. [Interpretation Patterns](#interpretation-patterns)
8. [Worked Example](#worked-example)
9. [Output](#output)
10. [Commands](#commands)
11. [What Comes Next](#what-comes-next)

---

## What This Step Does

Quality Scoring takes the 13-element metric vector **M(v)** produced by Step 2 for each component and maps it into four interpretable quality dimensions: **R**eliability, **M**aintainability, **A**vailability, and **V**ulnerability (RMAV). A composite quality score **Q(v)** combines the four dimensions, and **box-plot classification** assigns each component to one of five criticality levels.

```
M(v) from Step 2                 Quality Analyzer                   Output
─────────────────                ─────────────────                ──────────────────────────
PR, RPR, BT, CL, EV,       →    1. Compute R, M, A, V     →    R(v), M(v), A(v), V(v)
DG_in, DG_out, CC,               2. Compute Q(v)                 Q(v)
AP_c, BR, w, w_in, w_out         3. Box-plot classify            Level: CRITICAL / HIGH /
                                                                         MEDIUM / LOW / MINIMAL
```

This step completes the **predictive** half of the methodology. The Q(v) scores are predictions derived purely from topology — no runtime data involved. Step 4 generates ground-truth impact scores I(v) from failure simulation, and Step 5 measures how well Q(v) predicts I(v). Empirically, this step achieves Spearman ρ = 0.876 and F1-score > 0.90 across validated system scales.

---

## The Four Quality Dimensions

Each RMAV dimension answers a different operational question about a component's structural role:

| Dimension | Question | High Score Means | Stakeholder |
|-----------|----------|-----------------|-------------|
| **R — Reliability** | What is the blast radius if this fails? | Failure propagates widely through dependents | Reliability Engineer |
| **M — Maintainability** | How hard is this to change safely? | Tightly coupled, structural bottleneck | Software Architect |
| **A — Availability** | Is this a single point of failure? | Removing it would partition the dependency graph | DevOps / SRE |
| **V — Vulnerability** | Is this an attractive attack target? | Central, reachable, high-value hub | Security Engineer |

The four dimensions are deliberately orthogonal — they capture distinct failure modes a component might exhibit. A component can simultaneously score high on all four (a critical hub), or high on one and low on others (a pure bottleneck, or a pure SPOF). The RMAV breakdown tells you *why* a component is critical, guiding different remediation strategies.

---

## RMAV Formulas

Each dimension is a weighted linear combination of specific metrics from M(v). All inputs are normalized to [0, 1] by Step 2. All RMAV dimension scores are therefore also in [0, 1].

The default weights shown below are derived from the Analytic Hierarchy Process (AHP). See [AHP Weight Derivation](#ahp-weight-derivation) for how they are computed.

### Reliability R(v) — Fault Propagation Risk

```
R(v) = 0.40 × RPR(v) + 0.35 × w_in(v) + 0.25 × CDPot(v)
```

| Term | Weight | Rationale |
|------|--------|-----------|
| RPR(v) | 0.40 | Reverse PageRank — *global* cascade reach; how broadly v's failure propagates in the reverse-dependency direction |
| w_in(v) | 0.35 | QoS-weighted in-degree — the count of direct dependents weighted by their SLA priority; directly captures *immediate* blast radius without reusing a metric from other dimensions |
| CDPot(v) | 0.25 | Cascade Depth Potential — derived from RPR and the in/out-degree ratio: `((RPR + DG_in) / 2) × (1 − min(DG_out/DG_in, 1))`. Absorber nodes (many dependents, few outgoing links) score high; fan-out hubs with wide shallow cascades score low. |

A component with high R(v) is one whose failure would propagate broadly **and deeply** through the dependency graph. The combination of RPR (global cascade reach), w_in (immediate blast radius weighted by SLA priority), and CDPot (cascade depth signal) makes R(v) a comprehensive fault-propagation predictor while respecting strict metric orthogonality.

> [!NOTE]
> **CDPot(v) inline formula:** `CDPot = ((RPR + DG_in) / 2) × (1 − min(DG_out / DG_in, 1))`. Fan-out nodes (DG_out ≫ DG_in) → CDPot ≈ 0 (wide but shallow, cascade absorbed quickly). Absorber nodes (DG_in ≫ DG_out) → CDPot is high (deep, self-reinforcing cascade). This replaces the old `pubsub_betweenness` redistribution hack with a clean, explainable depth signal.

### Maintainability M(v) — Coupling Complexity

```
M(v) = 0.40 × BT(v) + 0.35 × DG_out(v) + 0.25 × (1 − CC(v))
```

| Term | Contribution | Rationale |
|------|-------------|-----------|
| BT(v) | 0.40 | Betweenness identifies *bottleneck* position — v lies on many shortest dependency paths |
| DG_out(v) | 0.35 | Out-degree measures *efferent coupling* — how many components v directly depends on |
| (1 − CC(v)) | 0.25 | Inverted clustering: low clustering means v's neighbors are *not* interconnected, so v cannot be removed without disrupting many unique paths |

Note: a high CC(v) means v's neighbors are dense and interconnected, offering alternative paths — lower maintainability risk. The inversion `(1 − CC)` converts this into a risk score: sparse neighborhood → harder to refactor safely.

### Availability A(v) — SPOF Risk

```
A(v) = 0.50 × AP_c(v) + 0.30 × BR(v) + 0.20 × w(v)
```

| Term | Contribution | Rationale |
|------|-------------|-----------|
| AP_c(v) | 0.50 | Continuous articulation-point score — measures how much the graph fragments upon v's removal |
| BR(v) | 0.30 | Bridge ratio — fraction of v's edges that are bridges; high BR means v's connections are irreplaceable |
| w(v) | 0.20 | QoS-derived component weight— ensures highly-critical components (based on domain priority) contribute to availability risk even if they are not structural SPOFs |

Availability uses structural indicators (AP_c, BR) alongside domain priority (w(v)) to identify components where failure would cause significant service disruption. By using $w(v)$ instead of topological hubness ($PR$), we maintain true metric orthogonality across RMAV dimensions.

### Vulnerability V(v) — Strategic Exposure

```
V(v) = 0.67 × EV(v) + 0.33 × CL(v)
```

| Term | Contribution | Rationale |
|------|-------------|-----------|
| EV(v) | 0.67 | Eigenvector centrality — connection to other high-value hubs signals *strategic* importance |
| CL(v) | 0.33 | Closeness centrality — short average distance to all others means a compromise propagates quickly |

Vulnerability measures how strategically placed a component is within the network. High Eigenvector centrality identifies nodes connected to other important hubs, making them high-value targets. High Closeness measures how effectively a compromise at `v` could pivot to other parts of the system. By removing Out-Degree from this dimension, we establish absolute metric orthogonality.

### Overall Quality Q(v)

```
Q(v) = w_R × R(v) + w_M × M(v) + w_A × A(v) + w_V × V(v)
```

**Default weights:** `w_R = w_M = w_A = w_V = 0.25` (equal weighting across all four dimensions).

The equal default reflects deliberate expert judgment — an AHP pairwise comparison matrix where all dimensions are rated equally important produces exactly this 0.25/0.25/0.25/0.25 split. This is appropriate for general-purpose analysis where no single quality concern dominates.

**Domain-specific adjustments:** For systems where one concern dominates, use `--use-ahp` with a custom matrix. Examples:

| System Type | Suggested Priority | Example Weights |
|-------------|-------------------|-----------------|
| High-availability (medical, aerospace) | A > R > M > V | w_A=0.40, w_R=0.30, w_M=0.20, w_V=0.10 |
| Security-critical (financial, government) | V > A > R > M | w_V=0.40, w_A=0.30, w_R=0.20, w_M=0.10 |
| Actively developed (fast iteration) | M > R > A > V | w_M=0.40, w_R=0.30, w_A=0.20, w_V=0.10 |
| General-purpose | Equal | w_R=w_M=w_A=w_V=0.25 |

---

## Metric Orthogonality

A core design principle is that each raw metric contributes to **at most one** RMAV dimension. This prevents a single structural property from accumulating disproportionate weight in Q(v). Without orthogonality, a metric like PageRank could influence both R(v) and A(v), effectively doubling its impact on Q(v).

| Metric | Symbol | R | M | A | V | Rationale |
|--------|--------|---|---|---|---|-----------|
| Reverse PageRank | RPR | ✓ | | | | Global cascade reach from v (primary R signal) |
| QoS-Weighted In-Degree | w_in | ✓ | | | | Direct dependents × SLA priority — immediate blast radius |
| Cascade Depth Potential | CDPot | ✓ | | | | Derived depth signal (RPR + DG ratio) — orthogonal to M, A, V |
| Betweenness | BT | | ✓ | | | Structural bottleneck position |
| Out-Degree | DG_out | | ✓ | | | Efferent coupling |
| Clustering Coefficient | CC | | ✓ | | | Local redundancy / modularity |
| Articulation Point Score | AP_c | | | ✓ | | Structural SPOF detection |
| Bridge Ratio | BR | | | ✓ | | Irreplaceable connections |
| QoS Weight | w(v) | | | ✓ | | QoS-derived component weight |
| Eigenvector Centrality | EV | | | | ✓ | Strategic hub connectivity |
| Closeness Centrality | CL | | | | ✓ | Propagation speed |

**Absolute Orthogonality:** Every raw metric contributes to **exactly one** RMAV dimension. PageRank (PR) and raw in-degree (DG_in) are no longer used in any dimension — they have been replaced in R(v) by w_in and CDPot, which carry the same signal without reusing metrics from other dimensions.

**QoS Weight metrics:** The weight metrics (w, w_in, w_out) from Step 2 reflect domain-specific priority. `w(v)` is included in A(v) to ensure high-priority components are flagged for availability risk. `w_in(v)` is now the second term in R(v), capturing the SLA-weighted immediate blast radius of a failure. `w_out(v)` is reported but not used in any RMAV formula.

---

## Classification

After computing Q(v) for all components in a layer, components are classified into five criticality levels. The classification uses **adaptive thresholds** based on the actual distribution of Q(v) scores, not static cutoffs.

### Box-Plot Thresholds (Normal Path)

Used when the layer has **≥ 12 components**.

```
Statistics: Q1, Median, Q3  computed from all Q(v) values
            IQR = Q3 − Q1
            Upper fence = Q3 + 1.5 × IQR

Classification:
  Q(v) > upper fence  →  CRITICAL   (statistical outlier — significantly above the 3rd quartile)
  Q(v) > Q3           →  HIGH       (above 75th percentile)
  Q(v) > Median       →  MEDIUM     (above 50th percentile)
  Q(v) > Q1           →  LOW        (above 25th percentile)
  Q(v) ≤ Q1           →  MINIMAL    (bottom 25th percentile)
```

**Why box-plot thresholds?** Static cutoffs (e.g., "Q(v) > 0.7 = CRITICAL") fail when systems have different score distributions. A system where all components are tightly coupled might have Q(v) values clustered between 0.6 and 0.8; a static threshold of 0.7 would misclassify half the system. Box-plot thresholds adapt to each system's actual distribution. The IQR-based upper fence specifically identifies *outliers* — components that are structurally exceptional relative to their peers — which is exactly the definition of "critical."

**Typical expected distribution:**
- CRITICAL: ~5–15% of components (outliers above upper fence)
- HIGH: ~25% of components (top quartile, below upper fence)
- MEDIUM: ~25% of components (second quartile)
- LOW: ~25% of components (third quartile)
- MINIMAL: ~25% of components (bottom quartile)

In practice, systems with many redundant paths produce fewer CRITICAL components; systems with chain-like topologies produce more.

### Small-Sample Percentile Fallback

Used when the layer has **< 12 components** (too few for stable quartile estimation).

| Level | Percentile Threshold |
|-------|---------------------|
| **CRITICAL** | Top 10% (90th percentile and above) |
| **HIGH** | Top 25% (75th–90th percentile) |
| **MEDIUM** | Top 50% (50th–75th percentile) |
| **LOW** | Top 75% (25th–50th percentile) |
| **MINIMAL** | Bottom 25% (below 25th percentile) |

This maps to fixed ranks for small systems: in a 10-component layer, CRITICAL = top 1 component, HIGH = next 1–2, etc.

---

## AHP Weight Derivation

The default weights (0.40, 0.35, 0.25 etc.) are not arbitrary — they are derived from the **Analytic Hierarchy Process (AHP)**, a structured decision-making method that translates expert judgment into numerical weights via pairwise comparisons (Saaty, 1980).

### The AHP Procedure

Given `n` criteria to weight:

```
Step 1 — Construct pairwise comparison matrix A (n × n):
         A[i][j] = "How much more important is criterion i than criterion j?"
         Using Saaty's scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme
         A[j][i] = 1 / A[i][j]  (reciprocal)

Step 2 — Compute geometric mean per row:
         GM[i] = (∏ A[i][j])^(1/n)  for j = 1..n

Step 3 — Normalize to get priority vector (weights):
         w[i] = GM[i] / Σ GM

Step 4 — Consistency check:
         λ_max = average of (A × w)[i] / w[i]
         CI    = (λ_max − n) / (n − 1)
         CR    = CI / RI[n]
         If CR > 0.10: the matrix is inconsistent — revise judgments
```

Random Index values (Saaty, 1980):

| n | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| RI | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 |

### Weight Shrinkage Strategy

To address methodological liability and provide stable rankings, we do not use pure AHP weights directly. Instead, we apply a **Formal Shrinkage Procedure** that blends the AHP priority vector with a uniform distribution (equal weights) using a mixing coefficient $\lambda$:

$$w_{final} = \lambda \cdot w_{AHP} + (1 - \lambda) \cdot w_{uniform}$$

- We use a default **$\lambda = 0.7$**.
- This respects the AHP-derived hierarchy while remaining robust to the extreme dominance of single metrics in small comparison matrices.
- Blending provides a principled, reproducible alternative to ad-hoc "smoothing."

### Final Weight Distributions ($\lambda = 0.7$)

**Reliability** — criteria: [RPR, w_in, CDPot]

```
           RPR   w_in  CDPot
RPR     [ 1.0,  0.67,  2.0 ]   RPR is less important than w_in (direct blast radius is primary)
w_in    [ 1.5,  1.0,   3.0 ]   w_in dominates: SLA-weighted dependents is the clearest signal
CDPot   [ 0.5,  0.33,  1.0 ]   CDPot provides a depth penalty (secondary signal)

→ GM:  [1.100, 1.145, 0.693]  →  Normalized: [0.334, 0.348, 0.166]  (sum to ~0.85 before rounding)
  CR ≈ 0.01  (highly consistent)
```
- AHP: [0.334, 0.500, 0.166]  (pure, before shrinkage)
- **Blend (λ=0.7): [0.334, 0.450, 0.216]**
- *Rationale*: QoS-weighted in-degree (w_in) is the strongest immediate signal for fault propagation blast radius. RPR captures global cascade reach. CDPot adds depth discrimination without reusing PageRank or raw DG_in.

**Availability** — criteria: [AP_c, BridgeRatio, QoS-Weight]

```
          APc   BR    w
APc     [ 1.0,  3.0,  5.0 ]   AP_c strongly dominates (structural SPOF is primary signal)
BR      [ 0.33, 1.0,  2.0 ]
w       [ 0.20, 0.50, 1.0 ]

→ GM:  [2.466, 0.693, 0.368]  →  Normalized: [0.65, 0.23, 0.12]
  CR ≈ 0.02  (highly consistent)
```
- AHP: [0.65, 0.23, 0.12]
- **Blend: [0.55, 0.26, 0.19]**
- *Rationale*: Prevents Articulation Point (AP_c) from completely drowning out Bridge Ratio and component QoS Weight.

**Maintainability** — criteria: [Betweenness, OutDegree, (1−Clustering)]

```
          BT    DGout  (1-CC)
BT      [ 1.0,  2.0,   3.0 ]
DGout   [ 0.5,  1.0,   2.0 ]
(1-CC)  [ 0.33, 0.50,  1.0 ]

→ GM:  [1.817, 0.909, 0.480]  →  Normalized: [0.54, 0.30, 0.16]
  CR ≈ 0.003  (highly consistent)
```
- AHP: [0.54, 0.30, 0.16]
- **Blend: [0.48, 0.31, 0.21]**
- *Rationale*: Betweenness remains the core bottleneck indicator, but Out-Degree (efferent coupling) and Clustering are given weight to reflect local complexity.

**Vulnerability** — criteria: [Eigenvector, Closeness]

```
          EV    CL
EV      [ 1.0,  2.0 ]
CL      [ 0.5,  1.0 ]
```
- AHP: [0.67, 0.33]
- **Blend: [0.57, 0.43]**   (with $\lambda = 0.7$ shrinkage)
- *Rationale*: Eigenvector highlights strategic hubs; Closeness measures propagation speed.

**Overall Q(v)** — criteria: [R, M, A, V]

```
All pairwise comparisons = 1.0  (all dimensions equally important)
→ weights = [0.25, 0.25, 0.25, 0.25]
CR = 0.00
```

### Weight Sensitivity Analysis

The default weights come from expert judgment and contain uncertainty. The sensitivity analysis perturbs the weights and measures how stable the top-ranked components remain:

```bash
python bin/analyze_graph.py --layer system --use-ahp --sensitivity
```

**Procedure:** 200 iterations of Gaussian weight perturbation (σ = 0.05), followed by:
- **Top-5 Stability:** fraction of iterations where the same top 5 components appear in the top 5 (target ≥ 0.80)
- **Mean Kendall τ:** average rank correlation between the baseline ranking and each perturbed ranking (target ≥ 0.90)

A Top-5 Stability ≥ 0.80 means the critical components are identified robustly despite weight uncertainty. Rankings that are unstable under small perturbations indicate that some components are borderline and the analyst should inspect their individual RMAV breakdowns rather than relying on the composite Q(v) alone.

---

## Interpretation Patterns

The RMAV breakdown tells you not just *that* a component is critical, but *why* — and what to do about it:

| Pattern | R | M | A | V | What It Means | Primary Risk | Recommended Action |
|---------|---|---|---|---|--------------|-------------|-------------------|
| **Hub** | High | High | High | High | Critical integration point — central to all concerns | Catastrophic failure | Add redundancy + circuit breakers + monitoring |
| **Reliability Hub** | High | Low | Low | Low | Widely depended upon but not a bottleneck or SPOF | Cascade failure | Add retry logic, graceful degradation |
| **Bottleneck** | Low | High | Low | Low | Coupling problem — change here ripples everywhere | Change fragility | Refactor to reduce coupling (e.g., introduce intermediary) |
| **SPOF** | Low | Low | High | Low | Structural single point of failure | Availability loss | Add redundant instance or failover path |
| **Target** | Low | Low | Low | High | Security-exposed hub | Compromise propagation | Harden, isolate, add access controls |
| **Maintenance Debt** | Med | High | Med | Low | Tightly coupled and in the critical path | Tech debt fragility | Prioritize refactoring in next sprint |
| **Leaf** | Low | Low | Low | Low | Peripheral — low concern | None | Standard practices |

**Reading the breakdown:** When a component is classified CRITICAL or HIGH, examine its individual R, M, A, V scores to identify the dominant concern. A CRITICAL component with A=0.90 but M=0.20 is a SPOF that is easy to change — the remediation is redundancy, not refactoring. A CRITICAL component with M=0.90 but A=0.20 is tightly coupled but not a SPOF — the remediation is architectural decoupling.

---

## Worked Example: Distributed Intelligent Factory (DIF)

This section computes RMAV scores for the **PLC_Controller (A3)** using the metric values from the DIF worked example in Step 2.

**Metric values for A3 (normalized):**

| Metric | Symbol | Value |
|--------|--------|-------|
| PageRank | PR | 0.75 |
| Reverse PageRank | RPR | 0.60 |
| Betweenness | BT | 0.95 |
| In-Degree | DG_in | 0.75 |
| Out-Degree | DG_out | 0.80 |
| Clustering | CC | 0.15 |
| AP_c | AP_c | 0.43 |
| Bridge Ratio | BR | 1.00 |
| Eigenvector | EV | 0.80 |
| Closeness | CL | 0.70 |

**RMAV computation (using default weights):**

First compute CDPot for A3:
```
CDPot = ((RPR + DG_in) / 2) × (1 − min(DG_out / DG_in, 1))
       = ((0.60 + 0.75) / 2) × (1 − min(0.80 / 0.75, 1))
       = 0.675 × (1 − 1.0)   ← DG_out/DG_in > 1, capped at 1
       = 0.675 × 0.0 = 0.00
```
A3 is a fan-out hub: it has more out-degree than in-degree, so CDPot → 0 (wide shallow cascade, not deep).

```
R(A3) = 0.40×RPR + 0.35×w_in + 0.25×CDPot
       = 0.40×0.60 + 0.35×0.75* + 0.25×0.00
       = 0.24 + 0.26 + 0.00 = 0.50
  (*w_in ≈ DG_in for this example; QoS weights assumed uniform)

M(A3) = 0.40×0.95 + 0.35×0.80 + 0.25×(1−0.15) = 0.38 + 0.28 + 0.2125 = 0.87

A(A3) = 0.50×0.43 + 0.30×1.00 + 0.20×0.68* = 0.215 + 0.30 + 0.136 = 0.65
   (*0.68 is the QoS weight derived from domain priority)

V(A3) = 0.67×0.80 + 0.33×0.70 = 0.536 + 0.231 = 0.77

Q(A3) = (0.50 + 0.87 + 0.65 + 0.77) / 4 = 0.70
```

**Interpretation:** **PLC_Controller (A3)** scores **0.70 (HIGH)**. The formula reveals that A3 has moderate reliability risk (R=0.50) because although it has many direct dependents (w_in) and broad reverse PageRank reach, its fan-out topology means cascades are wide but shallow (CDPot=0). The dominant concern remains **Maintainability** (M=0.87) — A3 is the system's structural bottleneck with max betweenness and high efferent coupling.

---

## Output

### Per-Component Quality Scores

For each component v in the selected layer:

| Field | Type | Description |
|-------|------|-------------|
| `reliability` | float [0,1] | R(v) — fault propagation risk |
| `maintainability` | float [0,1] | M(v) — coupling complexity |
| `availability` | float [0,1] | A(v) — SPOF risk |
| `vulnerability` | float [0,1] | V(v) — security exposure |
| `overall` | float [0,1] | Q(v) — composite score |
| `level` | enum | CRITICAL / HIGH / MEDIUM / LOW / MINIMAL |
| `is_articulation_point` | bool | Binary SPOF flag (from AP_c > 0) |

### Layer-Level Classification Summary

| Field | Description |
|-------|-------------|
| `critical_count` | Number of CRITICAL components |
| `high_count` | Number of HIGH components |
| `spof_count` | Number of components with AP_c > 0 |
| `q1`, `median`, `q3` | Box-plot thresholds used for classification |
| `upper_fence` | CRITICAL threshold = Q3 + 1.5×IQR |

### JSON Output Schema

```json
{
  "layer": "app",
  "classification_method": "box_plot",
  "thresholds": {
    "q1": 0.18,
    "median": 0.31,
    "q3": 0.52,
    "iqr": 0.34,
    "upper_fence": 0.73
  },
  "summary": {
    "total": 35,
    "critical": 3,
    "high": 9,
    "medium": 9,
    "low": 9,
    "minimal": 5,
    "spof_count": 3
  },
  "components": {
    "DataRouter": {
      "reliability":     0.88,
      "maintainability": 0.75,
      "availability":    0.92,
      "vulnerability":   0.81,
      "overall":         0.84,
      "level":           "CRITICAL",
      "is_articulation_point": true
    },
    "SensorHub": {
      "reliability":     0.76,
      "maintainability": 0.60,
      "availability":    0.85,
      "vulnerability":   0.71,
      "overall":         0.73,
      "level":           "HIGH",
      "is_articulation_point": true
    }
  }
}
```

---

## Commands

```bash
# Analyze with default equal weights (w_R = w_M = w_A = w_V = 0.25)
python bin/analyze_graph.py --layer app

# Analyze with AHP-derived weights (uses default pairwise matrices)
python bin/analyze_graph.py --layer system --use-ahp

# Export quality scores to JSON
python bin/analyze_graph.py --layer system --output results/quality.json

# Run weight sensitivity analysis
python bin/analyze_graph.py --layer system --use-ahp --sensitivity
# Reports: Top-5 Stability and Mean Kendall τ across 200 perturbations (σ=0.05)

# Analyze all layers and compare
for layer in app infra mw system; do
  python bin/analyze_graph.py --layer $layer --output results/quality_$layer.json
done
```

### Reading the Output

```
Layer: app | 35 components | Default weights (equal)
Classification: box-plot  |  Q1=0.18  Median=0.31  Q3=0.52  Fence=0.73

CRITICAL (3 components):
  DataRouter      Q=0.84  R=0.88  M=0.75  A=0.92  V=0.81  [SPOF]
  SensorHub       Q=0.73  R=0.76  M=0.60  A=0.85  V=0.71  [SPOF]
  CommandBus      Q=0.73  R=0.71  M=0.79  A=0.62  V=0.74

HIGH (9 components):
  ...
```

- **[SPOF]** tag indicates AP_c > 0 — structural single point of failure. Always address these first regardless of overall Q(v).
- Check the RMAV breakdown for CRITICAL components to identify the dominant concern before choosing a remediation strategy.

---

## What Comes Next

At this point, every component has a predicted quality score Q(v) ∈ [0, 1] and a criticality classification derived purely from topology. These are **predictions** — they have not yet been validated against real failure behavior.

Step 4 generates **ground-truth impact scores I(v)** by simulating component failures exhaustively and measuring the cascade damage each causes. Step 5 then computes Spearman ρ between Q(v) and I(v) to quantify how accurately topology predicts real failure impact — closing the loop on the methodology's central empirical claim.

For the **Reliability dimension specifically**, Step 4 also produces **IR(v)** — a cascade-dynamics-specific ground truth computed from the same exhaustive run. Step 5 validates R(v) against IR(v) using Spearman ρ(R, IR), Cascade Capture Rate CCR@5, and Cascade Magnitude Error CME — providing a reliability-specific validation signal that is independent of the global I(v) composite.

---

← [Step 2: Structural Analysis](structural-analysis.md) | → [Step 4: Failure Simulation](failure-simulation.md)