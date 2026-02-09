# Step 5: Validation

**Statistically prove that predicted quality scores match actual failure impact.**

← [Step 4: Failure Simulation](failure-simulation.md) | → [Step 6: Visualization](visualization.md)

---

## What This Step Does

Validation answers the central question of the entire methodology: **do our graph-based predictions actually work?**

It aligns predicted quality scores Q(v) from Step 3 with simulated impact scores I(v) from Step 4, then runs a battery of statistical tests to measure how well they agree.

```
Q(v) from Step 3         I(v) from Step 4
(predicted criticality)   (actual impact)
        │                        │
        └──── Align by component ┘
                     │
              Statistical Tests
                     │
           Pass / Fail Decision
```

## Data Alignment

Before comparing, the validator matches components that appear in both the predicted and simulated score sets. Components present in only one set are excluded and reported as warnings. At least three matched components are required for meaningful analysis.

## What Gets Measured

The validation computes three categories of metrics:

### Correlation — "Do rankings agree?"

| Metric | What It Tests | Target |
|--------|-------------|--------|
| **Spearman ρ** | Do the *rankings* of Q(v) and I(v) match? (rank correlation) | ≥ 0.70 |
| **Kendall τ** | Same idea, but more robust to ties | ≥ 0.50 |
| **Pearson r** | Do the *values* have a linear relationship? | Reported |
| **p-value** | Is the correlation statistically significant? | ≤ 0.05 |

Spearman ρ is the primary metric because we care about *relative ordering* (which components are most critical) more than exact score values.

### Classification — "Do criticality levels agree?"

| Metric | What It Tests | Target |
|--------|-------------|--------|
| **F1-Score** | Balance of precision and recall for critical/non-critical classification | ≥ 0.80 |
| **Precision** | Of components we predicted as critical, how many actually are? | ≥ 0.80 |
| **Recall** | Of components that are actually critical, how many did we catch? | ≥ 0.80 |
| **Cohen's κ** | Agreement corrected for chance | ≥ 0.60 |

### Ranking — "Do we identify the same top components?"

| Metric | What It Tests | Target |
|--------|-------------|--------|
| **Top-5 Overlap** | Do the top 5 predicted and top 5 actual components overlap? | ≥ 40% |
| **Top-10 Overlap** | Same for top 10 | ≥ 50% |
| **NDCG@K** | Normalized discounted cumulative gain (ranking quality) | Reported |

### Error — "How far off are the scores?"

| Metric | What It Tests | Target |
|--------|-------------|--------|
| **RMSE** | Root mean squared error between Q(v) and I(v) | ≤ 0.25 |
| **MAE** | Mean absolute error | ≤ 0.20 |

## Pass/Fail Decision

Validation uses a tiered gate system:

- **Primary gates** (must all pass): Spearman ρ ≥ 0.70, p-value ≤ 0.05, F1 ≥ 0.80, Top-5 Overlap ≥ 40%
- **Secondary gates** (should pass): RMSE ≤ 0.25
- **Reported metrics** (informational): all others

Overall validation passes only when all primary gates pass.

## Achieved Results

Across multiple system scales, the methodology has consistently achieved:

| Metric | Achieved | Target |
|--------|----------|--------|
| Spearman ρ | 0.876 | ≥ 0.70 |
| F1-Score | > 0.90 | ≥ 0.80 |
| Precision | > 0.90 | ≥ 0.80 |

Prediction accuracy improves with system scale — larger systems produce more stable and reliable correlation.

## Commands

```bash
# Validate all layers
python bin/validate_graph.py

# Validate a specific layer
python bin/validate_graph.py --layer system

# Custom targets
python bin/validate_graph.py --layer system --spearman 0.80 --f1 0.85

# Quick validation from pre-computed JSON files
python bin/validate_graph.py --quick predicted.json actual.json

# Export results
python bin/validate_graph.py --layer system --output results/validation.json

# Generate validation dashboard
python bin/validate_graph.py --layer system --visualize --open
```

## Interpreting Results

If validation passes, you can trust the graph-based predictions and skip expensive failure simulations for future system changes. If it fails, consider:

- **Low Spearman ρ**: The graph model may be missing important dependencies. Check if the derivation rules capture all relevant paths.
- **High Precision, Low Recall**: The model is conservative — it correctly identifies critical components but misses some. Consider lowering classification thresholds.
- **Low Precision, High Recall**: The model over-predicts criticality. Some structurally important components may have redundancy that the topology doesn't capture.

---

← [Step 4: Failure Simulation](failure-simulation.md) | → [Step 6: Visualization](visualization.md)