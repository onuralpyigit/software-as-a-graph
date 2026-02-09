# Step 3: Quality Scoring

**Combine raw metrics into four meaningful quality dimensions using AHP-derived weights.**

← [Step 2: Structural Analysis](structural-analysis.md) | → [Step 4: Failure Simulation](failure-simulation.md)

---

## What This Step Does

Quality Scoring takes the 13 raw metrics from Step 2 and maps them into four interpretable quality dimensions: Reliability, Maintainability, Availability, and Vulnerability (RMAV). Each dimension answers a different question about a component, and a composite score Q(v) summarizes overall criticality.

```
Metric Vectors M(v)          Quality            RMAV Scores + Classification
(PageRank, Betweenness,  →   Analyzer    →     R(v), M(v), A(v), V(v), Q(v)
 AP, Eigenvector, ...)                          + CRITICAL / HIGH / MEDIUM / LOW / MINIMAL
```

## The Four Quality Dimensions

| Dimension | Question It Answers | High Score Means |
|-----------|-------------------|-----------------|
| **R — Reliability** | What happens if this fails? | Failure would propagate widely |
| **M — Maintainability** | How hard is this to change? | Tightly coupled, bottleneck position |
| **A — Availability** | Is this irreplaceable? | Single point of failure risk |
| **V — Vulnerability** | Is this an attractive attack target? | High strategic exposure |

## Formulas

Each dimension is a weighted sum of specific metrics. A key design principle is **metric orthogonality**: each raw metric appears in at most one dimension, so no single metric dominates the overall score.

### Reliability R(v) — Fault Propagation Risk

```
R(v) = 0.40 × PageRank + 0.35 × ReversePageRank + 0.25 × InDegree
```

PageRank and Reverse PageRank capture transitive influence (how far failures spread). In-Degree captures direct dependents.

### Maintainability M(v) — Coupling Complexity

```
M(v) = 0.40 × Betweenness + 0.35 × OutDegree + 0.25 × (1 − Clustering)
```

Betweenness identifies bottlenecks. Out-Degree measures efferent coupling. Low clustering means the component's neighbors aren't interconnected (harder to refactor).

### Availability A(v) — SPOF Risk

```
A(v) = 0.50 × AP_c + 0.30 × BridgeRatio + 0.20 × Importance
```

AP_c is the continuous articulation point score. Bridge Ratio measures irreplaceable connections. Importance is the average of PageRank and Reverse PageRank.

### Vulnerability V(v) — Security Exposure

```
V(v) = 0.40 × Eigenvector + 0.30 × Closeness + 0.30 × OutDegree
```

Eigenvector identifies connections to high-value hubs. Closeness measures how quickly a compromise could propagate. Out-Degree represents the attack surface.

### Overall Quality Q(v)

```
Q(v) = 0.25 × R(v) + 0.25 × M(v) + 0.25 × A(v) + 0.25 × V(v)
```

Equal weights by default. Adjust based on system priorities (e.g., increase availability weight for high-availability systems).

## Classification

Components are classified into five criticality levels using **box-plot statistics** on the Q(v) distribution, not fixed thresholds. This avoids the problem of static cutoffs that don't adapt to different system sizes.

| Level | Rule |
|-------|------|
| **CRITICAL** | Score > Q3 + 1.5 × IQR (outliers) |
| **HIGH** | Score > Q3 |
| **MEDIUM** | Score > Median |
| **LOW** | Score > Q1 |
| **MINIMAL** | Score ≤ Q1 |

For small systems (< 12 components), a percentile fallback is used instead.

## Interpretation Patterns

The RMAV breakdown tells you not just *that* a component is critical, but *why*:

| Pattern | R | M | A | V | What It Means | Recommended Action |
|---------|---|---|---|---|--------------|-------------------|
| **Hub** | High | High | High | High | Critical integration point | Add redundancy + monitoring |
| **Bottleneck** | Low | High | Med | Med | Coupling problem | Refactor to reduce coupling |
| **SPOF** | Med | Low | High | Low | Single point of failure | Add redundancy |
| **Target** | Low | Low | Low | High | Security exposure | Harden and isolate |
| **Leaf** | Low | Low | Low | Low | Low concern | Standard practices |

## AHP Weight Derivation

The default weights above come from the Analytic Hierarchy Process (AHP), which derives weights from expert pairwise comparisons using Saaty's scale. This provides a principled, repeatable alternative to arbitrary weight assignment. Custom AHP matrices can be supplied for domain-specific needs.

### Weight Sensitivity

To check that rankings are robust to weight uncertainty:

```bash
python bin/analyze_graph.py --layer system --use-ahp --sensitivity
```

This perturbs weights 200 times with Gaussian noise (σ = 0.05) and reports stability metrics. A Top-5 Stability ≥ 0.80 and Mean Kendall τ ≥ 0.90 indicate robust rankings.

## Commands

```bash
# Default weights
python bin/analyze_graph.py --layer system

# AHP-derived weights
python bin/analyze_graph.py --layer system --use-ahp

# Export results
python bin/analyze_graph.py --layer system --output results/quality.json
```

## What Comes Next

At this point we have predicted quality scores Q(v) for every component. But are these predictions accurate? Step 4 simulates actual failures to produce ground truth impact scores I(v), and Step 5 statistically compares the two.

---

← [Step 2: Structural Analysis](structural-analysis.md) | → [Step 4: Failure Simulation](failure-simulation.md)