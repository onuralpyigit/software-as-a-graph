# Step 5: Validation

**Statistically compare predicted quality scores Q(v) against actual impact I(v)**

---

## Overview

Validation answers the critical question: **Do our predictions match reality?**

```
┌─────────────────────┐          ┌─────────────────────┐
│  Predicted Q(v)     │    vs    │  Actual I(v)        │
│  (from Step 3)      │          │  (from Step 4)      │
└─────────────────────┘          └─────────────────────┘
                    ↓
            Statistical Tests
                    ↓
         ┌───────────────────┐
         │  Spearman ρ       │
         │  F1-Score         │
         │  Precision/Recall │
         │  Top-K Overlap    │
         └───────────────────┘
```

---

## Validation Targets

| Metric | Target | Purpose |
|--------|--------|---------|
| Spearman ρ | ≥ 0.70 | Ranking correlation |
| F1-Score | ≥ 0.80 | Classification accuracy |
| Precision | ≥ 0.80 | Avoid false alarms |
| Recall | ≥ 0.80 | Catch critical components |
| Top-5 Overlap | ≥ 40% | Agreement on most critical |
| Top-10 Overlap | ≥ 50% | Agreement on critical set |

---

## Metric 1: Spearman Correlation (ρ)

Measures how well the **ranking** of predictions matches actual impact.

### Formula

```
ρ = 1 - (6 × Σdᵢ²) / (n × (n² - 1))
```

Where dᵢ = difference in ranks for component i.

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

## Metric 2: Classification Metrics

Convert scores to binary (critical/non-critical) for classification analysis.

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

### Example

```
Actually critical: 5 components
Predicted critical: 6 components
Correctly predicted: 4

TP = 4, FP = 2, FN = 1

Precision = 4/6 = 0.67
Recall = 4/5 = 0.80
F1 = 2 × (0.67 × 0.80) / (0.67 + 0.80) = 0.73
```

---

## Metric 3: Top-K Overlap

Measures agreement on the K most critical components.

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

In practice, you prioritize the most critical components. Even with moderate overall correlation, high Top-K overlap means your priority list is correct.

---

## Commands

```bash
# Run validation
python bin/validate_graph.py --layer system

# With visualization
python bin/validate_graph.py --layer system --visualize

# Export results
python bin/validate_graph.py --layer system --output results/validation.json
```

---

<!-- lines 178-265 omitted for brevity -->

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

→ [Step 6: Visualization](step6-visualization.md)
