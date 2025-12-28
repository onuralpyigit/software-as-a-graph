# Statistical Validation

This document explains the statistical validation approach for comparing predicted criticality scores against actual failure impacts.

---

## Table of Contents

1. [Overview](#overview)
2. [Validation Process](#validation-process)
3. [Correlation Metrics](#correlation-metrics)
4. [Classification Metrics](#classification-metrics)
5. [Ranking Metrics](#ranking-metrics)
6. [Research Targets](#research-targets)
7. [Validation Pipeline](#validation-pipeline)
8. [Implementation](#implementation)

---

## Overview

Validation answers the central research question:

> **Do topological metrics accurately predict actual system impact when components fail?**

### Approach

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VALIDATION FRAMEWORK                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PREDICTED SCORES              STATISTICAL              ACTUAL IMPACTS │
│   (from topology)               COMPARISON               (from simulation)│
│                                                                         │
│   ┌─────────────┐               ┌──────────┐            ┌─────────────┐│
│   │ Component A │               │          │            │ Component A ││
│   │ Score: 0.82 │──────────────▶│ Spearman │◀───────────│ Impact: 0.78││
│   ├─────────────┤               │   ρ      │            ├─────────────┤│
│   │ Component B │               │          │            │ Component B ││
│   │ Score: 0.65 │──────────────▶│ F1-Score │◀───────────│ Impact: 0.71││
│   ├─────────────┤               │          │            ├─────────────┤│
│   │ Component C │               │ Top-k    │            │ Component C ││
│   │ Score: 0.45 │──────────────▶│ Overlap  │◀───────────│ Impact: 0.42││
│   └─────────────┘               └──────────┘            └─────────────┘│
│                                      │                                  │
│                                      ▼                                  │
│                              ┌──────────────┐                          │
│                              │  Validation  │                          │
│                              │   Result     │                          │
│                              │ PASSED/FAILED│                          │
│                              └──────────────┘                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Validation Matters

Without validation, we cannot know if our predictions are meaningful:

| Scenario | Issue |
|----------|-------|
| High Spearman, Low F1 | Rankings correlate but classifications wrong |
| Low Spearman, High F1 | Lucky guesses, rankings don't match |
| Both Low | Methodology doesn't work |
| **Both High** | **Predictions are reliable** |

---

## Validation Process

### Step-by-Step

1. **Collect Predicted Scores**: Composite criticality scores from structural analysis
2. **Collect Actual Impacts**: Impact scores from failure simulation
3. **Match Components**: Ensure both have scores for same components
4. **Compute Metrics**: Correlation, classification, ranking
5. **Compare to Targets**: Determine validation status

### Data Preparation

```python
# From analysis (Step 3)
predicted = {
    "B1": 0.82,
    "B2": 0.75,
    "A1": 0.45,
    "A2": 0.38,
    "N1": 0.65,
    ...
}

# From simulation (Step 4)
actual = {
    "B1": 0.78,
    "B2": 0.71,
    "A1": 0.42,
    "A2": 0.41,
    "N1": 0.58,
    ...
}

# Must have same keys
common = set(predicted.keys()) & set(actual.keys())
```

---

## Correlation Metrics

Correlation measures the relationship between predicted and actual values.

### Spearman Rank Correlation (ρ)

Measures whether the **ranking** of predictions matches the **ranking** of actuals.

**Formula**:
```
         6 Σ dᵢ²
ρ = 1 - ──────────
        n(n² - 1)
```

Where dᵢ = difference between ranks for component i.

**Interpretation**:

| ρ Value | Interpretation |
|---------|----------------|
| 1.0 | Perfect positive correlation |
| 0.7 - 1.0 | Strong positive correlation ✓ |
| 0.4 - 0.7 | Moderate positive correlation |
| 0.0 - 0.4 | Weak positive correlation |
| 0.0 | No correlation |
| < 0 | Negative correlation (inverse) |

**Why Spearman over Pearson?**

- Works with any monotonic relationship (not just linear)
- Robust to outliers
- Rankings matter more than exact values for prioritization

**Example**:
```
Component   Predicted   Actual   Pred_Rank   Actual_Rank   d    d²
B1          0.82        0.78     1           1             0    0
B2          0.75        0.71     2           2             0    0
N1          0.65        0.58     3           4             1    1
A1          0.45        0.42     4           5             1    1
A2          0.38        0.41     5           3             2    4

Σ d² = 6
n = 5

ρ = 1 - (6 × 6) / (5 × (25 - 1)) = 1 - 36/120 = 0.70
```

### Pearson Correlation (r)

Measures linear relationship between values.

**Formula**:
```
        Σ(xᵢ - x̄)(yᵢ - ȳ)
r = ──────────────────────────
    √[Σ(xᵢ - x̄)²] √[Σ(yᵢ - ȳ)²]
```

**Usage**: Secondary metric; Spearman is primary.

### Kendall's Tau (τ)

Measures ordinal association based on concordant/discordant pairs.

**Formula**:
```
        (concordant - discordant)
τ = ─────────────────────────────────
          n(n-1)/2
```

**Usage**: More robust than Spearman for small samples with ties.

---

## Classification Metrics

Classification metrics evaluate binary classification: critical vs non-critical.

### Threshold Selection

We use the **80th percentile** of each distribution:

```python
pred_threshold = percentile(predicted_scores, 80)
actual_threshold = percentile(actual_impacts, 80)

pred_critical = [c for c in components if predicted[c] >= pred_threshold]
actual_critical = [c for c in components if actual[c] >= actual_threshold]
```

### Confusion Matrix

```
                    Actual Critical    Actual Non-Critical
                  ┌─────────────────┬───────────────────────┐
Predicted         │                 │                       │
Critical          │  True Positive  │   False Positive      │
                  │       (TP)      │        (FP)           │
                  ├─────────────────┼───────────────────────┤
Predicted         │                 │                       │
Non-Critical      │  False Negative │   True Negative       │
                  │       (FN)      │        (TN)           │
                  └─────────────────┴───────────────────────┘
```

### Precision

Fraction of predicted critical that are actually critical.

```
              TP
Precision = ────────
            TP + FP
```

**Interpretation**: How many of our "critical" predictions were correct?

### Recall (Sensitivity)

Fraction of actual critical that were predicted critical.

```
           TP
Recall = ────────
         TP + FN
```

**Interpretation**: How many actual critical components did we identify?

### F1-Score

Harmonic mean of precision and recall.

```
        2 × Precision × Recall
F1 = ─────────────────────────────
        Precision + Recall
```

**Why Harmonic Mean?**

- Penalizes imbalance between precision and recall
- F1 = 1 only if both precision and recall are perfect
- More stringent than arithmetic mean

**Example**:
```
TP = 4, FP = 1, FN = 1, TN = 14

Precision = 4 / (4 + 1) = 0.80
Recall = 4 / (4 + 1) = 0.80
F1 = 2 × 0.80 × 0.80 / (0.80 + 0.80) = 0.80
```

### Accuracy

Overall correct classification rate.

```
              TP + TN
Accuracy = ───────────────────
           TP + TN + FP + FN
```

**Note**: Less informative than F1 for imbalanced classes.

---

## Ranking Metrics

Ranking metrics evaluate agreement on top components.

### Top-k Overlap

Fraction of predicted top-k that appear in actual top-k.

```
                |Predicted_Top_k ∩ Actual_Top_k|
Top_k_Overlap = ─────────────────────────────────
                            k
```

**Example**:
```
Predicted Top-5: [B1, B2, N1, A3, A7]
Actual Top-5:    [B1, N1, B2, A5, A3]

Overlap: {B1, B2, N1, A3} = 4 components

Top-5 Overlap = 4/5 = 0.80 (80%)
```

**Interpretation**:
- 100%: Perfect agreement on top-k
- 60%+: Good agreement ✓
- <50%: Poor agreement, prioritization unreliable

### Why Top-k Matters

In practice, teams focus on the **most critical** components:
- Limited resources for remediation
- Prioritization is essential
- Top-k agreement directly impacts effectiveness

### Multiple k Values

We typically compute for k = 3, 5, 10:

```python
for k in [3, 5, 10]:
    overlap = len(set(pred_top_k[:k]) & set(actual_top_k[:k])) / k
    print(f"Top-{k}: {overlap:.0%}")
```

---

## Research Targets

### Target Values

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Spearman ρ** | ≥ 0.70 | Strong correlation standard in statistics |
| **F1-Score** | ≥ 0.90 | High classification accuracy |
| **Precision** | ≥ 0.80 | Limit false alarms |
| **Recall** | ≥ 0.80 | Catch most critical components |
| **Top-5 Overlap** | ≥ 60% | Agreement on highest priority |

### Validation Status

| Status | Condition |
|--------|-----------|
| **PASSED** | All targets met |
| **PARTIAL** | Most targets met (≥3 of 5) |
| **FAILED** | Few targets met (<3 of 5) |
| **INSUFFICIENT_DATA** | Too few components to validate |

### Interpretation Guidelines

```
┌─────────────┬──────────┬───────────────────────────────────────────┐
│  Spearman   │    F1    │              Interpretation               │
├─────────────┼──────────┼───────────────────────────────────────────┤
│   ≥ 0.70    │  ≥ 0.90  │ ✅ Excellent - Predictions highly reliable │
│   ≥ 0.70    │ 0.70-0.90│ ⚠️ Good - Rankings reliable, verify classes│
│  0.50-0.70  │  ≥ 0.90  │ ⚠️ Partial - Classifications ok, rankings off│
│  0.50-0.70  │ 0.70-0.90│ ⚠️ Moderate - Use with caution             │
│   < 0.50    │  < 0.70  │ ❌ Poor - Consider different methodology   │
└─────────────┴──────────┴───────────────────────────────────────────┘
```

---

## Validation Pipeline

The ValidationPipeline integrates all steps.

### Pipeline Flow

```
┌──────────────────┐
│   Input Graph    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 1. Analyze       │  GraphAnalyzer → predicted scores
│    (topology)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 2. Simulate      │  FailureSimulator → actual impacts
│    (failures)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. Validate      │  Validator → comparison metrics
│    (compare)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. Report        │  Status, metrics, details
│    (output)      │
└──────────────────┘
```

### Usage

```python
from src.validation import ValidationPipeline

pipeline = ValidationPipeline(
    spearman_target=0.70,
    f1_target=0.90,
    cascade_threshold=0.5,
    cascade_probability=0.7,
    seed=42
)

result = pipeline.run(
    graph,
    analysis_method="composite",
    enable_cascade=True
)

# Results
print(f"Status: {result.validation.status.value}")
print(f"Spearman: {result.validation.correlation.spearman:.4f}")
print(f"F1-Score: {result.validation.classification.f1:.4f}")

# Timing
print(f"Analysis: {result.analysis_time_ms:.0f} ms")
print(f"Simulation: {result.simulation_time_ms:.0f} ms")
print(f"Validation: {result.validation_time_ms:.0f} ms")
```

### Comparing Methods

```python
results = pipeline.compare_methods(
    graph,
    methods=["betweenness", "degree", "pagerank", "composite"],
    enable_cascade=True
)

for method, result in results.items():
    print(f"{method}: ρ={result.validation.correlation.spearman:.3f}, "
          f"F1={result.validation.classification.f1:.3f}")
```

---

## Implementation

### Validator Class

```python
from src.validation import Validator, ValidationTargets

targets = ValidationTargets(
    spearman=0.70,
    f1=0.90,
    precision=0.80,
    recall=0.80,
    top_5_overlap=0.60
)

validator = Validator(targets=targets, seed=42)

# Basic validation
result = validator.validate(
    predicted_scores,    # Dict[str, float]
    actual_impacts,      # Dict[str, float]
    component_types      # Optional: Dict[str, str]
)

# With bootstrap confidence intervals
result = validator.validate_with_bootstrap(
    predicted_scores,
    actual_impacts,
    n_iterations=1000,
    confidence=0.95
)
```

### ValidationResult Structure

```python
@dataclass
class ValidationResult:
    status: ValidationStatus           # PASSED, PARTIAL, FAILED
    total_components: int
    
    correlation: CorrelationMetrics    # Spearman, Pearson, Kendall
    classification: ConfusionMatrix    # TP, FP, FN, TN, F1, etc.
    ranking: RankingMetrics            # Top-k overlap
    
    targets: ValidationTargets         # Target values
    achieved: Dict[str, Tuple]         # Achieved vs target
    
    component_validations: List        # Per-component details
    false_positives: List[str]         # Type I errors
    false_negatives: List[str]         # Type II errors
    
    bootstrap_results: Optional[List]  # If bootstrap run
    
    def to_dict(self) -> Dict:
        """Export as dictionary"""
        
    def summary(self) -> Dict:
        """Quick summary stats"""
```

### CorrelationMetrics

```python
@dataclass
class CorrelationMetrics:
    spearman: float          # Spearman coefficient
    spearman_pvalue: float   # Statistical significance
    pearson: float           # Pearson coefficient
    pearson_pvalue: float
    kendall: float           # Kendall tau
    sample_size: int
```

### ConfusionMatrix

```python
@dataclass
class ConfusionMatrix:
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    threshold: float
    
    @property
    def precision(self) -> float
    
    @property
    def recall(self) -> float
    
    @property
    def f1(self) -> float
    
    @property
    def accuracy(self) -> float
```

### CLI Usage

```bash
# Basic validation
python validate_graph.py --input graph.json

# With options
python validate_graph.py \
    --input graph.json \
    --method composite \
    --spearman 0.70 \
    --f1 0.90 \
    --cascade \
    --output results/

# Compare methods
python validate_graph.py \
    --input graph.json \
    --compare \
    --methods betweenness degree pagerank composite
```

---

## Navigation

- **Previous:** [← Failure Simulation](simulation.md)
- **Next:** [Visualization →](visualization.md)
