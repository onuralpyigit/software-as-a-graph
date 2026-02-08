# Step 5: Validation

**Statistically compare predicted quality scores Q(v) against actual impact I(v)**

---

## Overview

Validation answers the critical question: **Do our predictions match reality?**

This step aligns the predicted quality scores Q(v) from Step 3 with the
simulated failure impact scores I(v) from Step 4, then applies a battery of
statistical tests to determine whether the graph-based model is a reliable
predictor of runtime criticality.

```
Q(v) from Step 3          I(v) from Step 4
 {A:0.84, B:0.72,          {B:0.78, C:0.31,
  C:0.35, D:0.11}           D:0.15, E:0.09}
         │                          │
         └──────── Align ───────────┘
                     │
              Common: {B, C, D}
              Warnings: A unmatched (pred)
                        E unmatched (actual)
                     │
              Statistical Tests (n=3)
                     │
         ┌───────────────────────┐
         │  Correlation          │
         │    Spearman ρ         │
         │    Kendall τ          │
         │    Pearson r          │
         │                       │
         │  Classification       │
         │    F1-Score           │
         │    Precision / Recall │
         │    Cohen's κ          │
         │                       │
         │  Ranking              │
         │    Top-K Overlap      │
         │    NDCG@K             │
         │                       │
         │  Error                │
         │    RMSE / NRMSE       │
         │    MAE                │
         └───────────────────────┘
                     │
              Pass / Fail Decision
              + Confidence Intervals
```

### Data Alignment

Before computing any metrics the validator must reconcile the two score
dictionaries. Components present in only one dictionary are excluded and
reported as warnings. At least three matched components are required for
meaningful statistical analysis; below that threshold the validator reports
"Insufficient data" and skips computation.

---

## Validation Targets

| Metric | Target | Purpose | Gate |
|--------|--------|---------|------|
| Spearman ρ | ≥ 0.70 | Ranking correlation | Primary |
| Spearman p-value | ≤ 0.05 | Statistical significance | Primary |
| F1-Score | ≥ 0.80 | Classification accuracy | Primary |
| Top-5 Overlap | ≥ 40% | Agreement on most critical | Primary |
| RMSE | ≤ 0.25 | Prediction error | Secondary |
| Precision | ≥ 0.80 | Avoid false alarms | Reported |
| Recall | ≥ 0.80 | Catch critical components | Reported |
| Cohen's κ | ≥ 0.60 | Chance-corrected agreement | Reported |
| Top-10 Overlap | ≥ 50% | Extended critical set | Reported |
| Kendall τ | ≥ 0.50 | Robust rank correlation | Reported |
| MAE | ≤ 0.20 | Absolute error | Reported |

**Primary gates** must all pass for overall validation to succeed. **Secondary
gates** must also pass but carry lower weight. **Reported** metrics are
computed and included in results for analysis but do not block pass/fail.

---

## Metric 1: Spearman Rank Correlation (ρ)

Measures how well the **ranking** of predictions matches actual impact. This is
our primary validation metric because we care more about correctly ordering
components by criticality than about exact score values.

### Formula

```
ρ = 1 - (6 × Σdᵢ²) / (n × (n² - 1))
```

Where dᵢ = difference in ranks for component i.

### Statistical Significance

Spearman ρ alone is insufficient — a high ρ on very few components may be due
to chance. The p-value quantifies the probability of observing a correlation at
least as extreme under the null hypothesis (no association). We require p ≤ 0.05
as a primary gate.

The p-value is computed via a t-distribution approximation:

```
t = ρ × √((n - 2) / (1 - ρ²))
```

with n - 2 degrees of freedom.

### Bootstrap Confidence Intervals

Point estimates can be misleading with small samples. The validator computes
95% bootstrap confidence intervals (1000 resamples, fixed seed for
reproducibility) for Spearman ρ and F1-Score. A result like
ρ = 0.85 [0.72, 0.93] is far more informative than ρ = 0.85 alone.

### Example

| Component | Q(v) Rank | I(v) Rank | dᵢ | dᵢ² |
|-----------|-----------|-----------|-----|------|
| Fusion | 1 | 1 | 0 | 0 |
| Broker | 2 | 3 | -1 | 1 |
| Gateway | 3 | 2 | 1 | 1 |
| Control | 4 | 4 | 0 | 0 |
| Sensor | 5 | 5 | 0 | 0 |

```
ρ = 1 - (6 × 2) / (5 × 24) = 1 - 0.10 = 0.90 ✓
```

### Interpretation

| ρ Value | Strength |
|---------|----------|
| 0.90 - 1.00 | Very strong |
| 0.70 - 0.89 | Strong (target) |
| 0.50 - 0.69 | Moderate |
| < 0.50 | Weak |

---

## Metric 2: Kendall Rank Correlation (τ)

Kendall's τ counts concordant and discordant pairs and has better statistical
properties than Spearman ρ for small samples (n < 20). It is reported alongside
ρ for robustness.

### Formula

```
τ = (concordant - discordant) / (n × (n - 1) / 2)
```

A pair (i, j) is **concordant** if Q(i) > Q(j) and I(i) > I(j) (or both
reversed), and **discordant** otherwise.

### Interpretation

| τ Value | Strength |
|---------|----------|
| 0.70 - 1.00 | Very strong |
| 0.50 - 0.69 | Strong (target) |
| 0.30 - 0.49 | Moderate |
| < 0.30 | Weak |

Kendall τ values are typically lower than Spearman ρ for the same data; a τ of
0.60 roughly corresponds to ρ of 0.75.

---

## Metric 3: Classification Metrics

Convert continuous scores to binary labels (critical / non-critical) for
classification analysis.

### Thresholding Strategy

Both predicted and actual scores are independently thresholded at the **75th
percentile** (configurable via `critical_percentile`). Components at or above
the threshold are labeled "critical."

```
pred_threshold  = percentile(Q(v) values, 75)
actual_threshold = percentile(I(v) values, 75)
```

This means the top ~25% of components on each side are labeled critical. The
symmetric thresholding guarantees equal class proportions, which simplifies
interpretation but means baseline agreement by chance alone is ~62.5%. This is
why we also report Cohen's κ (see below).

### Confusion Matrix

```
                    Predicted
                 Critical  Non-Critical
Actual  Critical    TP          FN
        Non-Crit    FP          TN
```

### Precision

Of components we predicted as critical, how many actually are?

```
Precision = TP / (TP + FP)
```

High precision → Few false alarms.

### Recall

Of actually critical components, how many did we catch?

```
Recall = TP / (TP + FN)
```

High recall → Few missed critical components.

### F1-Score

Harmonic mean balancing precision and recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Cohen's Kappa (κ)

F1-Score does not account for chance agreement. Cohen's κ corrects for this:

```
κ = (p_o - p_e) / (1 - p_e)
```

Where p_o is observed agreement and p_e is expected agreement by chance. Values
above 0.60 indicate substantial agreement beyond what random labeling would
produce.

| κ Value | Strength |
|---------|----------|
| 0.80 - 1.00 | Almost perfect |
| 0.60 - 0.79 | Substantial (target) |
| 0.40 - 0.59 | Moderate |
| < 0.40 | Fair or poor |

### Example

```
Actually critical: 5 components
Predicted critical: 6 components
Correctly predicted: 4

TP = 4, FP = 2, FN = 1, TN = 13

Precision = 4/6 = 0.67
Recall    = 4/5 = 0.80
F1        = 2 × (0.67 × 0.80) / (0.67 + 0.80) = 0.73

p_o = (4 + 13) / 20 = 0.85
p_e = (6/20 × 5/20) + (14/20 × 15/20) = 0.60
κ   = (0.85 - 0.60) / (1 - 0.60) = 0.625 ✓
```

---

## Metric 4: Top-K Overlap

Measures agreement on the K most critical components. In practice, architects
prioritize the most critical components for redundancy and monitoring, so
agreement on the top of the list matters most.

### Formula

```
Overlap_K = |Top_K(Q) ∩ Top_K(I)| / K
```

### Example

```
Top-5 by Q(v): [Fusion, Broker, Planning, Gateway, Control]
Top-5 by I(v): [Broker, Fusion, Planning, Perception, Localization]

Overlap = {Fusion, Broker, Planning} = 3 components

Top-5 Overlap = 3/5 = 60% ✓
```

### Why Top-K Matters

Even with moderate overall correlation, high Top-K overlap means your
**priority list** is correct — the components you would invest in hardening are
indeed the ones that matter most.

---

## Metric 5: NDCG (Normalized Discounted Cumulative Gain)

NDCG is a more nuanced ranking metric than Top-K overlap. It accounts not only
for whether a truly critical component appears in the predicted top-K, but also
for **where** it appears. A critical component ranked 2nd is better than one
ranked 5th.

### Formula

```
DCG@K  = Σᵢ₌₁ᴷ  rel(i) / log₂(i + 1)
IDCG@K = Σᵢ₌₁ᴷ  rel_ideal(i) / log₂(i + 1)
NDCG@K = DCG@K / IDCG@K
```

Where rel(i) is the actual impact I(v) of the component at predicted rank i,
and rel_ideal(i) uses the ideal ordering (sorted by actual impact).

| NDCG@K Value | Quality |
|-------------|---------|
| 0.90 - 1.00 | Excellent ranking |
| 0.70 - 0.89 | Good (target) |
| 0.50 - 0.69 | Fair |
| < 0.50 | Poor |

---

## Metric 6: Error Metrics

Error metrics measure how far off the predicted scores are in absolute terms.

### RMSE (Root Mean Squared Error)

```
RMSE = √(Σ(Q(v) - I(v))² / n)
```

RMSE penalizes large errors more heavily. Target: ≤ 0.25.

### NRMSE (Normalized RMSE)

RMSE depends on the score range. NRMSE normalizes by the range of actual values
to make error comparable across different scales:

```
NRMSE = RMSE / (max(I) - min(I))
```

An NRMSE < 0.30 indicates predictions track actual values well.

### MAE (Mean Absolute Error)

```
MAE = Σ|Q(v) - I(v)| / n
```

MAE is more robust to outliers than RMSE. Target: ≤ 0.20.

### Interpreting Error vs. Correlation

High Spearman ρ with moderate RMSE is common and acceptable: it means the
**ordering** is correct even though the absolute score values differ. Since the
methodology's primary goal is to identify *which* components are most critical
(not to predict exact failure impact magnitudes), ranking metrics take
precedence over error metrics.

---

## Pass / Fail Logic

Validation passes when **all primary and secondary gates** are satisfied:

```python
passed = (
    spearman >= 0.70 and
    spearman_p <= 0.05 and      # statistically significant
    f1_score >= 0.80 and
    top_5_overlap >= 0.40 and
    rmse <= 0.25                 # secondary gate
)
```

### Low-Power Warning

When 3 ≤ n < 10, the validator adds a warning that statistical power is low.
Results may still pass but should be interpreted with caution: the confidence
intervals will be wide.

---

## Troubleshooting Failed Validation

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Low ρ, high Top-5 | Mid-range ranking noise | Often acceptable for critical component identification; check NDCG |
| High ρ, low F1 | Threshold sensitivity | Try different `critical_percentile` (70 or 80); check Cohen's κ |
| Low ρ, low everything | Model mismatch | Review AHP weight configuration; check if correct layer is used |
| High App layer, low Infra | Expected behavior | Infrastructure has physical redundancy; application layer is primary target |
| High ρ but p > 0.05 | Too few components | Increase system scale; n ≥ 15 recommended for reliable statistics |
| Passed but wide CI | Small sample | Results are suggestive but not conclusive; validate at larger scale |
| High RMSE, high ρ | Score scale mismatch | Acceptable — rankings correct despite value offset; check NRMSE |
| Low Cohen's κ, high F1 | Chance inflation | Symmetric thresholding inflating agreement; examine confusion matrix |

---

## Commands

```bash
# Run validation
python bin/validate_graph.py --layer system

# With visualization
python bin/validate_graph.py --layer system --visualize

# Custom thresholds
python bin/validate_graph.py --layer system \
    --spearman 0.70 \
    --f1 0.80 \
    --top5 0.40

# Export results
python bin/validate_graph.py --layer system --output results/validation.json
```

---

## Output Example

```
═══════════════════════════════════════════════════════════════
  VALIDATION - System Layer
═══════════════════════════════════════════════════════════════

  Data Alignment:
    Predicted: 48 components
    Actual:    45 components
    Matched:   43 components (2 warnings)

  Correlation:
    Spearman ρ  = 0.852  (p = 0.0001) [0.74, 0.92] ✓
    Kendall τ   = 0.691
    Pearson r   = 0.814  (p = 0.0003)

  Classification (threshold: 75th percentile):
    Precision   = 0.857  ✓
    Recall      = 0.800  ✓
    F1-Score    = 0.828  ✓
    Cohen's κ   = 0.643
    Confusion:  TP=8  FP=1  FN=2  TN=32

  Ranking:
    Top-5 Overlap  = 60%   ✓
    Top-10 Overlap = 70%
    NDCG@5         = 0.91
    NDCG@10        = 0.87

  Error:
    RMSE    = 0.182  ✓
    NRMSE   = 0.241
    MAE     = 0.147

  ────────────────────────────────────────────────
  RESULT: PASSED  (all primary + secondary gates met)
  ════════════════════════════════════════════════
```

---

## Benchmark Suite

For comprehensive validation across scales and layers:

```bash
python bin/benchmark.py --scales small,medium,large,xlarge \
                    --layers app,infra,system \
                    --runs 5 \
                    --output results/benchmark
```

Generates:
- CSV data file
- JSON detailed results
- Markdown summary report

---

## Next Step

→ [Step 6: Visualization](visualization.md)

---

## Navigation

← [Step 4: Failure Simulation](failure-simulation.md) | [README](../README.md)