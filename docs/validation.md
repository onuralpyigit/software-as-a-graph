# Step 5: Validation

**Statistically prove that topology-based predictions agree with simulation ground truth.**

← [Step 4: Failure Simulation](failure-simulation.md) | → [Step 6: Visualization](visualization.md)

---

## Table of Contents

1. [What This Step Does](#what-this-step-does)
2. [Data Alignment](#data-alignment)
3. [Why Spearman Is the Primary Metric](#why-spearman-is-the-primary-metric)
4. [Metric Definitions and Formulas](#metric-definitions-and-formulas)
   - [Correlation Metrics](#correlation-metrics)
   - [Classification Metrics](#classification-metrics)
   - [Ranking Metrics](#ranking-metrics)
   - [Error Metrics](#error-metrics)
5. [Pass/Fail Gate System](#passfail-gate-system)
6. [Validation Targets by Layer and Scale](#validation-targets-by-layer-and-scale)
7. [Achieved Results](#achieved-results)
8. [Worked Example](#worked-example)
9. [Interpreting Results](#interpreting-results)
10. [Output](#output)
11. [Commands](#commands)
12. [What Comes Next](#what-comes-next)

---

## What This Step Does

Validation answers the central question of the entire methodology: **do topology-based predictions actually work?**

It aligns the predicted quality scores Q(v) from Step 3 with the simulated impact scores I(v) from Step 4, then computes eleven statistical metrics across four categories to measure agreement between prediction and ground truth. A tiered gate system produces a clear **pass/fail** verdict.

```
Q(v) from Step 3              I(v) from Step 4
─────────────────             ─────────────────
Predicted criticality         Ground-truth impact
(topology-derived)            (simulation-derived)
        │                             │
        └──── Align by component ID ──┘
                          │
             ┌────────────┴────────────┐
             │                         │
       Correlation              Classification
       Spearman ρ               F1, Precision,
       Kendall τ                Recall, Cohen's κ
       Pearson r                        │
             │                         │
           Ranking                   Error
        Top-5, Top-10           RMSE, MAE
        NDCG@K                         │
             └────────────┬────────────┘
                          │
                   Pass / Fail decision
                   (primary gates must all pass)
```

This step closes the methodological loop. Steps 2–4 generate the two sets of scores; Step 5 quantifies how much they agree. A passing validation demonstrates the methodology's core contribution: **cheap pre-deployment topology analysis reliably predicts which components will cause the greatest damage if they fail.**

---

## Data Alignment

Before any computation, the validator aligns Q(v) and I(v) by component ID:

```
For each component v:
    If v appears in both Q and I → include in aligned set (n)
    If v appears only in Q       → exclude; log warning "component in predictions only"
    If v appears only in I       → exclude; log warning "component in simulation only"
```

**Minimum sample size:** At least **5 matched components** are required for meaningful analysis. With n < 5, Spearman ρ cannot achieve statistical significance at the p ≤ 0.05 level for any value of ρ (the t-statistic distribution has insufficient degrees of freedom). With n = 3 or 4, the correlation coefficient can be computed but the p-value is meaningless — every result fails the significance gate. The validator reports a warning for n < 5 and an error for n < 3.

**Mismatches** are expected when using `--layer app` for analysis (which excludes brokers and nodes from Q(v)) while running exhaustive simulation on `--layer system` (which produces I(v) for all component types). Always match layers between analysis and simulation outputs, or use `--quick` with pre-aligned JSON files.

---

## Why Spearman Is the Primary Metric

The methodology's claim is about **relative ordering**: it should correctly identify *which* components are most critical, not predict the exact numerical magnitude of I(v). Spearman ρ measures rank correlation rather than value correlation, making it the correct primary metric for this purpose.

Three additional properties make Spearman preferable to Pearson r for this specific application:

**Non-normal distributions.** Both Q(v) and I(v) distributions are right-skewed — most components have low scores, with a small number of outliers at the high end. Pearson r assumes bivariate normality; Spearman ρ does not.

**Different scales.** Q(v) is derived from normalized graph metrics using AHP weights; I(v) is derived from cascade simulation outcomes with different weighting. The two scales are not commensurable. Rank correlation avoids comparing absolute values across incompatible scales.

**Robustness to outliers.** A single extreme outlier (a node that causes catastrophic cascade) can distort Pearson r substantially; Spearman ρ is insensitive to outlier magnitude.

Pearson r is still computed and reported — a high Pearson r alongside a high Spearman ρ strengthens the validity claim — but it is not a gate metric.

---

## Metric Definitions and Formulas

### Correlation Metrics

**Spearman Rank Correlation ρ**

```
Input:  Q = [Q(v₁), ..., Q(vₙ)],  I = [I(v₁), ..., I(vₙ)]
Output: ρ ∈ [−1, 1]

1. R_Q[i] = rank of Q(vᵢ) among Q (average ranks for ties)
   R_I[i] = rank of I(vᵢ) among I (average ranks for ties)
2. d[i]   = R_Q[i] − R_I[i]
3. ρ = 1 − (6 × Σ d[i]²) / (n × (n² − 1))
```

Significance test: t = ρ × √(n−2) / √(1−ρ²), degrees of freedom = n−2. p-value derived from t-distribution.

| ρ Range | Interpretation |
|---------|---------------|
| 0.90 – 1.00 | Very strong agreement |
| 0.70 – 0.90 | Strong agreement (primary target zone) |
| 0.50 – 0.70 | Moderate agreement |
| 0.30 – 0.50 | Weak agreement |
| 0.00 – 0.30 | Negligible agreement |

**Kendall τ (Tau-b)**

```
τ = (C − D) / √((C + D + T_Q) × (C + D + T_I))
```

where C = concordant pairs (Q and I rank the same way), D = discordant pairs, T_Q = pairs tied in Q only, T_I = pairs tied in I only. Kendall τ is more conservative than Spearman ρ and more robust when there are many ties. A large gap between ρ and τ (e.g., ρ = 0.85, τ = 0.50) suggests that agreement is driven by a few dominant pairs — inspect the high-end components specifically.

**Pearson r**

```
r = Σ (Q(vᵢ) − Q̄)(I(vᵢ) − Ī) / √(Σ (Q(vᵢ) − Q̄)² × Σ (I(vᵢ) − Ī)²)
```

Reported but not a gate metric. Complements Spearman ρ by checking whether the *magnitudes* of Q and I have a linear relationship, not just their ordering.

### Classification Metrics

Binary classification compares which components are predicted critical (Q-critical) against which are empirically critical (I-critical). Each component receives a binary label from both its Q(v) score and its I(v) score using the same **box-plot threshold** independently applied to each distribution:

```
Q-critical:  Q(v) > Q3_Q + 1.5 × IQR_Q   (outliers in the Q distribution)
I-critical:  I(v) > Q3_I + 1.5 × IQR_I   (outliers in the I distribution after Winsorization)
```

**Note on Robustness**: To mitigate simulation noise and extreme stochastic outliers, $I(v)$ scores are **Winsorized** (capped at the 95th percentile) before constructing the box-plot. This ensures that a single catastrophic cascade doesn't inflate the IQR so much that other truly critical components appear insignificant.

This produces a 2×2 confusion matrix:

|  | I-critical = True | I-critical = False |
|--|-------------------|--------------------|
| **Q-critical = True** | TP | FP |
| **Q-critical = False** | FN | TN |

**Precision, Recall, F1-Score**

```
Precision = TP / (TP + FP)   — of our CRITICAL predictions, how many are correct?
Recall    = TP / (TP + FN)   — of truly critical components, how many did we catch?
F1        = 2 × Precision × Recall / (Precision + Recall)
```

F1 is the primary classification gate because it balances both error types. High precision alone means the model is conservative (few false alarms but misses critical components). High recall alone means the model over-predicts (catches everything but with many false alarms).

**Cohen's κ (Kappa)**

```
κ = (P_o − P_e) / (1 − P_e)

P_o = (TP + TN) / n                         (observed agreement)
P_e = ((TP+FP)/n × (TP+FN)/n) +             (expected agreement by chance)
      ((TN+FN)/n × (TN+FP)/n)
```

κ corrects for agreement that would occur by random chance. A model that randomly predicts a small fraction of components as critical will show low Precision but moderate F1 — κ ≤ 0 would expose this. κ ≥ 0.60 is the standard threshold for "substantial agreement" (Landis & Koch, 1977).

| κ Range | Interpretation |
|---------|---------------|
| 0.81 – 1.00 | Almost perfect |
| 0.61 – 0.80 | Substantial |
| 0.41 – 0.60 | Moderate |
| 0.21 – 0.40 | Fair |
| ≤ 0.20 | Slight or worse |

### Ranking Metrics

**Top-K Overlap**

```
Top-K Overlap = |top_K(Q) ∩ top_K(I)| / K
```

where `top_K(X)` is the set of K components with the highest scores in X. For K=5 this is the fraction of the 5 highest-Q components that also appear in the 5 highest-I components.

This is the most directly actionable metric from an architectural standpoint: if an architect uses the top-5 predicted critical components to prioritize redundancy work, does that work actually protect the right components?

**NDCG@K (Normalized Discounted Cumulative Gain)**

```
DCG@K  = Σ  rel(i) / log₂(i + 1)   for i = 1..K
          i
IDCG@K = DCG@K for the ideal (perfect) ordering
NDCG@K = DCG@K / IDCG@K
```

where `rel(i)` = I(v) of the component ranked i-th by Q(v) (the relevance of the i-th predicted item is its actual impact score). NDCG@K ∈ [0, 1]; NDCG@K = 1.0 means the top K predicted components are the same as the top K actual components in the same order.

Unlike Top-K Overlap (binary set intersection), NDCG@K is position-sensitive: predicting the actual #1 component as #2 is penalized less than predicting it as #10.

K defaults to 10. The `--ndcg-k` flag adjusts K.

### Error Metrics

**RMSE (Root Mean Squared Error)**

```
RMSE = √(Σ (Q(vᵢ) − I(vᵢ))² / n)
```

Penalizes large individual prediction errors quadratically. Sensitive to outliers. Target: ≤ 0.25.

**MAE (Mean Absolute Error)**

```
MAE = Σ |Q(vᵢ) − I(vᵢ)| / n
```

More interpretable than RMSE: MAE = 0.15 means predictions are off by 0.15 on the [0, 1] scale on average. Less sensitive to outliers than RMSE. Target: ≤ 0.20.

When RMSE >> MAE, a small number of components have large individual prediction errors. Inspecting the scatter plot (Step 6 dashboard) reveals which components are systematically mispredicted.

---

## Pass/Fail Gate System

Validation uses a three-tier gate system:

| Gate | Metrics | Pass Condition | Effect |
|------|---------|---------------|--------|
| **Primary** | Spearman ρ, p-value, F1-Score, Top-5 Overlap | **All must pass** | If any fails → overall FAIL |
| **Secondary** | RMSE | Should pass | Failure logged as warning, does not block overall pass |
| **Reported** | Kendall τ, Pearson r, Precision, Recall, Cohen's κ, Top-10 Overlap, NDCG@K, MAE | Informational | Always computed and reported; no gate |

**Primary gate thresholds (defaults):**

| Metric | Threshold |
|--------|-----------|
| Spearman ρ | ≥ 0.70 |
| p-value | ≤ 0.05 |
| F1-Score | ≥ 0.80 |
| Top-5 Overlap | ≥ 40% |

**Secondary gate threshold:**

| Metric | Threshold |
|--------|-----------|
| RMSE | ≤ 0.25 |

All thresholds are configurable via CLI flags (`--spearman`, `--f1`, `--precision`, `--recall`, `--top5`). Raising thresholds is appropriate when validating for a higher-stakes deployment context — for example, a medical device system where the cost of missing a critical component is severe might use `--spearman 0.85 --f1 0.90`.

**Overall result:** `passed = True` if and only if all four primary gates pass individually.

---

## Validation Targets by Layer and Scale

Targets vary by layer (application layer topology predicts more accurately than infrastructure) and scale (larger systems produce more stable centrality distributions). The full validation matrix from the research:

| Test ID | Layer | Scale | Target ρ | Target F1 | Rationale |
|---------|-------|-------|----------|-----------|-----------|
| VT-APP-01 | Application | Small (10–25) | ≥ 0.75 | ≥ 0.75 | Fewer components; less stable distributions |
| VT-APP-02 | Application | Medium (30–50) | ≥ 0.80 | ≥ 0.80 | Standard target zone |
| VT-APP-03 | Application | Large (60–100) | ≥ 0.85 | ≥ 0.83 | Strong performance expected |
| VT-INF-01 | Infrastructure | Small | ≥ 0.50 | ≥ 0.65 | Cross-layer effects reduce accuracy |
| VT-INF-02 | Infrastructure | Medium | ≥ 0.52 | ≥ 0.66 | |
| VT-INF-03 | Infrastructure | Large | ≥ 0.54 | ≥ 0.68 | |
| VT-SYS-01 | System | Small | ≥ 0.70 | ≥ 0.75 | Mixed-layer targets |
| VT-SYS-02 | System | Medium | ≥ 0.75 | ≥ 0.80 | |
| VT-SYS-03 | System | Large | ≥ 0.80 | ≥ 0.83 | |

**Why infrastructure targets are lower:** Application-layer dependencies are directly captured by the DEPENDS_ON derivation rules (publisher-to-subscriber through shared topics). Infrastructure dependencies involve cross-layer effects — a Node failure cascades to hosted applications through RUNS_ON edges, but the topology of RUNS_ON relationships is not reflected in the application-layer G_analysis(app) that Step 2 analyzes. This is a known methodological limitation and is discussed explicitly in the thesis. The infrastructure layer still passes its own lower targets consistently.

---

## Achieved Results

Results across all validated system scales and domains (ROS 2, IoT, financial trading, healthcare).

### By Layer (Large Scale, 60–100 Components)

| Metric | Application Layer | Infrastructure Layer | Default Target |
|--------|:-----------------:|:--------------------:|:--------------:|
| Spearman ρ | **0.85** ✓ | 0.54 ✓* | ≥ 0.70 |
| F1-Score | **0.83** ✓ | 0.68 ✓* | ≥ 0.80 |
| Precision | **0.86** ✓ | 0.71 | ≥ 0.80 |
| Recall | **0.80** ✓ | 0.65 | ≥ 0.80 |
| Top-5 Overlap | **62%** ✓ | 40% ✓* | ≥ 40% |
| RMSE | **0.18** ✓ | 0.24 ✓ | ≤ 0.25 |

*Infrastructure layer passes against its own layer-specific lower targets (VT-INF-03).

### By Scale (Application Layer)

| Scale | Components | Spearman ρ | F1-Score | Analysis Time |
|-------|------------|:-----------:|:--------:|:-------------:|
| Tiny | 5–10 | 0.72 | 0.70 | < 0.5 s |
| Small | 10–25 | 0.78 | 0.75 | < 1 s |
| Medium | 30–50 | 0.82 | 0.80 | ~2 s |
| Large | 60–100 | 0.85 | 0.83 | ~5 s |
| XLarge | 150–300 | **0.88** | **0.85** | ~20 s |

### Key Findings

**Prediction accuracy improves with system scale.** Larger systems produce more stable centrality distributions — metrics like PageRank and Betweenness converge on more reliable values when the graph has more components and edges. At XLarge scale (150–300 components), Spearman ρ reaches 0.88 — well above the primary target. This is a significant research finding: the methodology's most practically important use case (large enterprise systems) is also where it performs best.

**Best reported result: Spearman ρ = 0.876** across the full multi-domain validation suite (reported in the IEEE RASSE 2025 paper). This figure represents the aggregate performance across all validated system configurations.

**Application layer consistently outperforms.** The gap between application (ρ = 0.85) and infrastructure (ρ = 0.54) at large scale reflects the structural asymmetry: application dependencies are directly encoded in the DEPENDS_ON graph, while infrastructure dependencies involve cross-layer cascade effects that topology analysis cannot fully capture without the full G_structural.

---

## Worked Example

This section manually computes Spearman ρ for a 5-component validation to illustrate the mechanics.

**Input: Q(v) and I(v) for 5 components**

| Component | Q(v) | I(v) |
|-----------|------|------|
| DataRouter | 0.84 | 0.91 |
| SensorHub | 0.73 | 0.85 |
| CommandBus | 0.62 | 0.60 |
| MapServer | 0.45 | 0.41 |
| LogApp | 0.18 | 0.12 |

**Step 1 — Assign ranks (rank 1 = highest score):**

| Component | Q(v) | Rank_Q | I(v) | Rank_I | d = Rank_Q − Rank_I | d² |
|-----------|------|:------:|------|:------:|:-------------------:|:--:|
| DataRouter | 0.84 | 1 | 0.91 | 1 | 0 | 0 |
| SensorHub | 0.73 | 2 | 0.85 | 2 | 0 | 0 |
| CommandBus | 0.62 | 3 | 0.60 | 3 | 0 | 0 |
| MapServer | 0.45 | 4 | 0.41 | 4 | 0 | 0 |
| LogApp | 0.18 | 5 | 0.12 | 5 | 0 | 0 |
| **Σ d²** | | | | | | **0** |

**Step 2 — Compute ρ:**

```
ρ = 1 − (6 × 0) / (5 × (25 − 1)) = 1 − 0/120 = 1.0
```

**Significance:** t = 1.0 × √3 / √0 → undefined (perfect correlation has no finite t). In practice, p < 0.001 for ρ = 1.0 with n = 5. ✓

**Realistic example with disagreement:** Now suppose CommandBus and MapServer are swapped in the I(v) ranking:

| Component | Q(v) | Rank_Q | I(v) | Rank_I | d | d² |
|-----------|------|:------:|------|:------:|:-:|:--:|
| DataRouter | 0.84 | 1 | 0.91 | 1 | 0 | 0 |
| SensorHub | 0.73 | 2 | 0.85 | 2 | 0 | 0 |
| CommandBus | 0.62 | 3 | 0.39 | 4 | −1 | 1 |
| MapServer | 0.45 | 4 | 0.62 | 3 | +1 | 1 |
| LogApp | 0.18 | 5 | 0.12 | 5 | 0 | 0 |
| **Σ d²** | | | | | | **2** |

```
ρ = 1 − (6 × 2) / (5 × 24) = 1 − 12/120 = 1 − 0.10 = 0.90
```

ρ = 0.90 — very strong agreement despite CommandBus and MapServer being swapped in the middle of the ranking. The top-2 and bottom-1 agree perfectly, so the rank disagreement in positions 3–4 has modest effect.

**Classification check for this example:**

Box-plot on Q(v): Q1=0.18, Median=0.45, Q3=0.73, IQR=0.55, upper fence = 0.73+0.825 = 1.555.
No Q(v) exceeds the upper fence → no Q-critical components. CRITICAL threshold falls back to components above Q3: DataRouter (Q=0.84 > 0.73) → Q-critical = {DataRouter}.

Box-plot on I(v): Q1=0.12, Median=0.60, Q3=0.85, IQR=0.73, upper fence = 1.945.
Above Q3: DataRouter (I=0.91 > 0.85) → I-critical = {DataRouter}.

TP=1, FP=0, FN=0, TN=4 → Precision=1.0, Recall=1.0, F1=1.0 ✓

In this 5-component example, the model perfectly identifies the one truly critical component despite the rank swap in positions 3–4.

---

## Interpreting Results

### Diagnostic Guide

Use the pattern of metric outcomes to diagnose what is happening in the model:

**Pattern 1 — Validation passes cleanly (ρ ≥ 0.70, F1 ≥ 0.80, Top-5 ≥ 40%)**

The topology accurately predicts failure impact. You can use Q(v) rankings with high confidence to prioritize redundancy and hardening work without running full failure simulation.

**Pattern 2 — Low ρ (< 0.70), significant p-value (< 0.05)**

Rankings disagree broadly. The graph model is likely missing important dependencies. Check: Are all pub-sub paths captured? Are there out-of-band dependencies (shared databases, external APIs) not represented in the graph? Consider running the simulation on the system layer rather than the app layer to capture cross-layer effects.

**Pattern 3 — High ρ (≥ 0.70), but low F1 (< 0.80)**

Rankings agree globally but the binary classification boundary is wrong. The box-plot threshold places the CRITICAL boundary in a region where Q and I diverge. Inspect the scatter plot (Step 6 dashboard) to see if there is a cluster of components near the threshold that Q and I classify differently. This often indicates borderline components where both topological position and cascade behavior are ambiguous — a valid limitation to report.

**Pattern 4 — High Precision, Low Recall (< 0.70)**

The model is conservative: the components it calls CRITICAL truly are critical, but it is missing some. Components it classified as HIGH or MEDIUM that I(v) classifies as I-critical are the false negatives. Examine these: do they have structural redundancy (high Bridge Ratio, low AP_c) that masks high actual impact? This is a topology-vs-runtime gap worth investigating.

**Pattern 5 — Low Precision (< 0.70), High Recall**

The model over-predicts criticality. Structurally important components (high Q(v)) have lower actual impact (I(v)) than expected. The most common cause: the system has redundant paths that the topology correctly identifies as potentially critical, but those paths are resilient enough in practice (multiple publishers, broker redundancy) that the impact is absorbed. This is not a flaw — it means redundancy is working. Report this as a conservative-bias characteristic of the predictor.

**Pattern 6 — High ρ but Large Gap between ρ and Kendall τ (e.g., ρ = 0.85, τ = 0.50)**

Agreement is driven by a few dominant pairs — likely the very highest and lowest components agree perfectly, inflating ρ, while middle-range rankings are noisy. Check Top-5 overlap specifically: if Top-5 is high but mid-range agreement is low, the model is reliable for identifying the most critical components (the most operationally useful outcome) even if mid-range rankings are imprecise.

**Pattern 7 — Primary gates pass, RMSE fails (> 0.25)**

The rankings agree but score magnitudes diverge. Q(v) scores are on a different scale than I(v) scores. This does not affect the ranking-based metrics; RMSE failure alone is not a methodology flaw. Consider whether absolute score comparison is meaningful given the different derivation paths of Q(v) and I(v).

**Pattern 8 — Application layer passes, Infrastructure layer fails**

Expected and not a problem. Infrastructure layer failure should be evaluated against infrastructure-layer targets (VT-INF-*), not the default targets. If infrastructure validation is failing its own lower targets, re-examine whether RUNS_ON and CONNECTS_TO edges are correctly populated in the input topology.

### Acting on a Failing Validation

If the primary gates fail at a layer that should pass:

1. **Check data alignment.** Run `--verbose` and look at how many components were aligned vs. excluded. If many components are mismatched, ensure you are using matching layer flags for analysis and simulation.
2. **Check the graph topology.** Missing edges in the input JSON can dramatically reduce prediction quality. Re-examine the input for completeness.
3. **Try a different layer.** If `--layer system` fails, try `--layer app` — application layer routinely achieves the primary targets even when the system layer doesn't.
4. **Review the scatter plot.** Step 6 generates a Q(v) vs I(v) scatter plot. Outliers far from the diagonal are components driving metric failures; their structural profiles often reveal the dependency gaps.
5. **Check sample size.** If n < 20 in the aligned set, rankings are inherently noisy. Focus on Top-5 overlap and κ rather than ρ.

---

## Output

### ValidationGroupResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `layer` | string | Layer validated (app / infra / system) |
| `n_aligned` | int | Number of matched component pairs used |
| `n_excluded_q` | int | Components in Q only (not in I) |
| `n_excluded_i` | int | Components in I only (not in Q) |
| `correlation.spearman` | float | Spearman ρ |
| `correlation.spearman_p` | float | p-value for Spearman ρ |
| `correlation.kendall` | float | Kendall τ |
| `correlation.pearson` | float | Pearson r |
| `classification.precision` | float | Precision |
| `classification.recall` | float | Recall |
| `classification.f1` | float | F1-Score |
| `classification.kappa` | float | Cohen's κ |
| `ranking.top5_overlap` | float | Top-5 overlap (0–1) |
| `ranking.top10_overlap` | float | Top-10 overlap (0–1) |
| `ranking.ndcg_k` | float | NDCG@K |
| `error.rmse` | float | RMSE |
| `error.mae` | float | MAE |
| `gates.primary_passed` | bool | All four primary gates passed |
| `gates.secondary_passed` | bool | RMSE gate passed |
| `passed` | bool | Overall result (= primary_passed) |

### JSON Output Schema

```json
{
  "layer": "app",
  "n_aligned": 35,
  "n_excluded_q": 0,
  "n_excluded_i": 2,
  "correlation": {
    "spearman":   0.876,
    "spearman_p": 0.00001,
    "kendall":    0.712,
    "pearson":    0.891
  },
  "classification": {
    "precision": 0.921,
    "recall":    0.867,
    "f1":        0.893,
    "kappa":     0.834
  },
  "ranking": {
    "top5_overlap":  0.80,
    "top10_overlap": 0.70,
    "ndcg_k":        0.94
  },
  "error": {
    "rmse": 0.142,
    "mae":  0.118
  },
  "gates": {
    "primary_passed":   true,
    "secondary_passed": true
  },
  "passed": true
}
```

### CLI Console Output

```
Validation Results | Layer: app | Aligned: 35 components

  CORRELATION
    Spearman ρ:   0.876  (p=0.000)  ✓  target ≥ 0.70
    Kendall τ:    0.712             [reported]
    Pearson r:    0.891             [reported]

  CLASSIFICATION
    Precision:    0.921             [reported]
    Recall:       0.867             [reported]
    F1-Score:     0.893  ✓          target ≥ 0.80
    Cohen's κ:    0.834             [reported]

  RANKING
    Top-5 Overlap:  80%  ✓          target ≥ 40%
    Top-10 Overlap: 70%             [reported]
    NDCG@10:       0.94             [reported]

  ERROR
    RMSE:  0.142  ✓  target ≤ 0.25
    MAE:   0.118     [reported]

  ┌────────────────────────────────┐
  │  VALIDATION RESULT:  PASSED ✓  │
  └────────────────────────────────┘
```

---

## Commands

```bash
# ─── Standard validation (reads from Neo4j) ───────────────────────────────────
# Validate application layer (recommended first — highest accuracy)
python bin/validate_graph.py --layer app

# Validate all layers
python bin/validate_graph.py

# Validate specific layer
python bin/validate_graph.py --layer system

# ─── Custom validation targets ────────────────────────────────────────────────
# Stricter targets for high-stakes systems (medical, aerospace)
python bin/validate_graph.py --layer app --spearman 0.85 --f1 0.90

# Looser targets for research exploration at small scale
python bin/validate_graph.py --layer app --spearman 0.60 --f1 0.70

# ─── Quick validation from pre-computed JSON files ────────────────────────────
# Input: predicted = quality.json (from analyze_graph.py --output)
#        actual    = impact.json  (from simulate_graph.py --output)
# Both files must use the same component ID space.
python bin/validate_graph.py --quick results/quality.json results/impact.json

# ─── Export results ───────────────────────────────────────────────────────────
python bin/validate_graph.py --layer app --output results/validation.json

# Export with JSON stdout for scripting
python bin/validate_graph.py --layer app --json | jq '.passed'

# ─── Visualization dashboard ─────────────────────────────────────────────────
# Generate and open the validation dashboard (Step 6)
# Shows: scatter plot Q(v) vs I(v), confusion matrix, ranking comparison table
python bin/validate_graph.py --layer app --visualize --open

# ─── Full pipeline: analyze → simulate → validate ─────────────────────────────
python bin/analyze_graph.py --layer app --output results/quality.json
python bin/simulate_graph.py failure --exhaustive --layer app \
    --output results/impact.json
python bin/validate_graph.py --quick results/quality.json results/impact.json \
    --output results/validation.json

# ─── Deterministic validation with fixed seed (for reproducible research) ─────
python bin/generate_graph.py --scale medium --seed 42 --output test_data.json
python bin/import_graph.py   --input test_data.json --clear
python bin/analyze_graph.py  --layer app --use-ahp --output analysis.json
python bin/simulate_graph.py failure --exhaustive --layer app --output simulation.json
python bin/validate_graph.py --quick analysis.json simulation.json
# Expected: passed = true, ρ ≥ 0.80
```

---

## What Comes Next

A passing validation confirms the methodology's central claim: topology-based quality scores Q(v) reliably predict which components will cause the most damage if they fail, without any runtime monitoring.

Step 6 renders these results — Q(v) scores, RMAV breakdowns, I(v) impact scores, and validation metrics — in an interactive HTML dashboard. The dashboard includes the Q(v) vs I(v) scatter plot (the visual proof of correlation), the interactive dependency graph with components colour-coded by criticality level, and sortable tables for detailed component-level inspection.

---

← [Step 4: Failure Simulation](failure-simulation.md) | → [Step 6: Visualization](visualization.md)