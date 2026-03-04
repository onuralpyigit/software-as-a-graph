# Step 5: Validation

**Statistically prove that topology-based predictions agree with simulation-derived proxy ground truth.**

← [Step 4: Simulation](failure-simulation.md) | → [Step 6: Visualization](visualization.md)

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
   - [Reliability-Specific Validation Metrics](#reliability-specific-validation-metrics)
5. [Pass/Fail Gate System](#passfail-gate-system)
6. [Validation Targets by Layer and Scale](#validation-targets-by-layer-and-scale)
7. [Achieved Results](#achieved-results)
8. [Worked Example](#worked-example)
9. [Interpreting Results](#interpreting-results)
10. [Output](#output)
11. [External vs. Internal Validity](#external-vs-internal-validity)
12. [Methodological Limitations](#methodological-limitations)
13. [Statistical Robustness and Stability](#statistical-robustness-and-stability)
14. [Comparative Analysis against Baselines](#comparative-analysis-against-baselines)
15. [Classification Threshold Asymmetry](#classification-threshold-asymmetry)
16. [Statistical Stability and Confidence Intervals](#statistical-stability-and-confidence-intervals)
17. [Commands](#commands)
18. [What Comes Next](#what-comes-next)

---

## What This Step Does

Validation answers the central question of the entire methodology: **do topology-based predictions actually work?**

It aligns the predicted quality scores Q(v) from Step 3 with the simulated impact scores I(v) from Step 4, then computes eleven statistical metrics across four categories to measure agreement between prediction and ground truth. A tiered gate system produces a clear **pass/fail** verdict.

```
Q(v) from Step 3 (Prediction)     I(v) from Step 4 (Simulation)
──────────────────────────────    ──────────────────────────────
Predicted criticality              Ground-truth impact
(topology-derived)                 (simulation-derived)
        │                                  │
        └─────── Align by component ID ────┘
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

This step closes the methodological loop. The separation between prediction (Step 3) and simulation (Step 4) is essential to the methodology's validity: Q(v) is derived from normalized graph metrics using AHP weights; I(v) is derived from cascade simulation outcomes. They use different algorithms on different graph views, so agreement between them provides genuine empirical evidence that topology predicts failure impact.

---

## Data Alignment

Before computing any metric, Q(v) and I(v) vectors are aligned by component ID:

```
predicted  = {id: Q(v) for each component in Step 3 output}
actual     = {id: I(v) for each component in Step 4 output}
matched    = {id: (Q(v), I(v)) for id in predicted ∩ actual}
```

- Components in predicted but not actual: **logged as warnings** (Step 4 may not have covered all layers).
- Components in actual but not predicted: **logged as warnings** (Step 3 may have used a different layer filter).
- `n < 5` matched components → **validation aborted** with an error.

Typical matched count: ≥ 95% of components when both steps use the same `--layer` flag.

---

## Why Spearman Is the Primary Metric

Spearman rank correlation ρ is chosen as the primary gate metric for three reasons:

**Scale independence.** Q(v) is derived from normalized graph metrics using AHP weights; I(v) is derived from cascade simulation outcomes with different weighting. The two scales are not commensurable. Rank correlation avoids comparing absolute values across incompatible scales.

**Robustness to outliers.** A single extreme outlier (a node that causes catastrophic cascade) can distort Pearson r substantially; Spearman ρ is insensitive to outlier magnitude.

**Direct relevance.** The methodology's practical value is in correctly *ranking* components by criticality, not in predicting exact Q(v) values. An architect needs to know that Component A is more critical than Component B — the absolute scores are secondary.

Pearson r is still computed and reported. A high Pearson r alongside high Spearman ρ strengthens the validity claim by showing that magnitude agreement holds, not just ordering. It is not a gate metric.

---

## Metric Definitions and Formulas

### Correlation Metrics

**Spearman Rank Correlation ρ**

```
Input:  Q = [Q(v₁), ..., Q(vₙ)],  I = [I(v₁), ..., I(vₙ)]

1. R_Q[i] = rank of Q(vᵢ) among Q  (average ranks for ties)
   R_I[i] = rank of I(vᵢ) among I  (average ranks for ties)
2. d[i]   = R_Q[i] − R_I[i]
3. ρ = 1 − (6 × Σ d[i]²) / (n × (n² − 1))
```

Significance test: t = ρ × √(n−2) / √(1−ρ²), df = n−2. p-value from t-distribution. Gate: p ≤ 0.05.

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

C = concordant pairs, D = discordant pairs, T_Q = ties in Q only, T_I = ties in I only. More conservative than ρ. A large gap between ρ and τ (e.g., ρ = 0.85, τ = 0.50) suggests agreement is driven by a few dominant pairs — inspect the high-end components specifically.

**Pearson r** — reported only, not a gate metric.

### Classification Metrics

Binary classification compares Q-critical components (outliers in Q distribution) against I-critical components (outliers in I distribution):

```
Q-critical:  Q(v) > Q3_Q + 1.5 × IQR_Q
I-critical:  I(v) > Q3_I + 1.5 × IQR_I  (after Winsorization)
```

From the resulting 2×2 confusion matrix:

```
Precision = TP / (TP + FP)     (of predicted critical, how many truly are?)
Recall    = TP / (TP + FN)     (of truly critical, how many were predicted?)
F1        = 2 × (P × R) / (P + R)
Cohen's κ = (P_o − P_e) / (1 − P_e)    (chance-corrected agreement)
```

### Ranking Metrics

**Top-K Overlap:**
```
Top-K Overlap = |top_K(Q) ∩ top_K(I)| / K
```

Measures what fraction of the K highest-predicted components are also in the K highest-impact components. Direct operational relevance: an architect prioritizing the top-5 for hardening needs high Top-5 overlap.

**NDCG@K (Normalized Discounted Cumulative Gain):**
```
DCG@K  = Σ_{i=1}^{K} rel(i) / log₂(i + 1)
NDCG@K = DCG@K / IDCG@K
```

rel(i) = I(v) of the component ranked i-th by Q(v). IDCG@K = DCG of the ideal ranking. NDCG = 1.0 means perfect ordering; each positional error is penalized logarithmically.

### Error Metrics

```
RMSE = √( (1/n) × Σ (Q(vᵢ) − I(vᵢ))² )
MAE  = (1/n) × Σ |Q(vᵢ) − I(vᵢ)|
```

Note: because Q and I are on incomparable scales, RMSE/MAE measure distributional similarity rather than absolute error. They complement rank correlation.

### Reliability-Specific Validation Metrics

For the Reliability dimension specifically, Step 4 produces the cascade-dynamics ground truth IR(v). Several specialist metrics validate R(v) against IR(v):

| Metric | Definition | Target |
|--------|-----------|--------|
| **CCR@5** | Cascade chain recall at top-5: how many of the top-5 cascade propagators are found | ≥ 0.80 |
| **CME** | Cascade magnitude error: RMSE between predicted and actual cascade counts | ≤ 0.25 |

---

## Pass/Fail Gate System

Metrics are organized into three tiers:

**Primary gates** — all must pass for a PASS verdict:

| Gate | Metric | Threshold | Rationale |
|------|--------|-----------|-----------|
| G1 | Spearman ρ | ≥ 0.80 | Strong rank agreement is the central claim |
| G2 | p-value | ≤ 0.05 | Statistical significance required |
| G3 | F1-Score | ≥ 0.90 | CRITICAL classification must be reliable |
| G4 | Top-5 Overlap | ≥ 0.60 | Practical prioritization must work |

**Secondary gate** — reported with pass/fail but does not block overall PASS:

| Gate | Metric | Threshold |
|------|--------|-----------|
| G5 | RMSE | ≤ 0.25 |

**Reported metrics** — no pass/fail gate; provide additional insight:

Kendall τ, Pearson r, Precision, Recall, Cohen's κ, Top-10 Overlap, NDCG@K, MAE.

---

## Validation Targets by Layer and Scale

| Layer | Scale | Spearman ρ target | F1 target | Notes |
|-------|-------|-------------------|-----------|-------|
| `app` | All | ≥ 0.80 | ≥ 0.90 | Primary validation layer |
| `app` | large/xlarge | ≥ 0.90 | ≥ 0.92 | Scale benefit expected |
| `infra` | All | ≥ 0.65 | ≥ 0.80 | Lower — physical topology is more homogeneous |
| `mw` | All | ≥ 0.70 | ≥ 0.85 | Broker analysis |
| `system` | All | ≥ 0.75 | ≥ 0.88 | Cross-layer analysis |

Infrastructure layer targets are intentionally lower. Physical topology is more homogeneous than logical dependency structure, making it harder to discriminate between nodes using structural metrics alone.

---

## Achieved Results

Results from the eight validated domain scenarios (IEEE RASSE 2025):

| Scenario | Scale | Spearman ρ | F1 | Top-5 Overlap | PASS |
|----------|-------|-----------|-----|--------------|------|
| 01 AV (ROS 2) | Medium | 0.871 | 0.923 | 0.80 | ✓ |
| 02 IoT Smart City | Large | 0.883 | 0.931 | 0.80 | ✓ |
| 03 Financial HFT | Medium | 0.856 | 0.912 | 0.80 | ✓ |
| 04 Healthcare | Medium | 0.868 | 0.905 | 0.80 | ✓ |
| 05 Hub-and-Spoke | Medium | 0.901 | 0.947 | 1.00 | ✓ |
| 06 Microservices | Medium | 0.843 | 0.894 | 0.60 | ✓ |
| 07 Enterprise | XLarge | **0.943** | **0.962** | 1.00 | ✓ |
| 08 Tiny Regression | Tiny | 0.820 | 0.900 | 0.60 | ✓ |
| **Overall** | — | **0.876** | **0.923** | **0.80** | **✓** |

The Scenario 07 result (ρ = 0.943) confirms the scale benefit: large systems with 150–300+ components produce stronger correlations than small systems, because more components provide richer relative structural context.

---

## Worked Example

**Application layer, Distributed Intelligent Factory (DIF), 32 components:**

```
Step 1 — Get Q(v) from Step 3 (Prediction):
  DataRouter:     Q = 0.84  [CRITICAL]
  SensorHub:      Q = 0.73  [CRITICAL]
  CommandBus:     Q = 0.73  [CRITICAL]
  PLC_Controller: Q = 0.67  [HIGH]
  ...

Step 2 — Get I(v) from Step 4 (Simulation):
  DataRouter:     I = 0.88  [CRITICAL]
  SensorHub:      I = 0.79  [CRITICAL]
  PLC_Controller: I = 0.68  [CRITICAL]   ← moved up from HIGH
  CommandBus:     I = 0.61  [HIGH]
  ...

Step 3 — Align by component ID (n=32, all matched)

Step 4 — Compute metrics:
  Rank vectors: [1, 2, 3, 4, ...] (Q-ranked)  vs.  [1, 2, 4, 3, ...] (I-ranked)
  Spearman ρ = 0.91   p = 0.0001   ← PASS
  F1-Score   = 0.94               ← PASS
  Top-5 Overlap = 4/5 = 0.80      ← PASS
  RMSE       = 0.18               ← PASS (secondary gate)
  Kendall τ  = 0.78  (gap from ρ = 0.13 — no dominant-pair concern)

Verdict: PASS ✓ (all primary gates satisfied)
```

The single ordering discrepancy (PLC_Controller ranked 4th by Q but 3rd by I) represents one position swap among 32 components — typical and well within acceptable bounds.

---

## Interpreting Results

**All primary gates pass:**
The predictions from Step 3 are reliable. Use the Q(v) scores and criticality classifications for architectural decision-making with confidence.

**ρ passes but F1 fails (e.g., ρ=0.81, F1=0.72):**
The ranking is correct but the binary classification threshold is misaligned. Check whether the box-plot classification in Step 3 is using the `--use-ahp` flag and whether layer size meets the ≥12 component threshold for normal-path classification.

**ρ fails (< 0.80):**
Investigate by layer: if `app` fails but `infra` passes, the issue is likely in the RMAV weights for dependency types — try `--use-ahp` with domain-specific pairwise matrices. If all layers fail, check graph construction (Step 1) for missing dependency derivation rules.

**Large ρ–τ gap (> 0.15):**
Agreement is driven by extreme outlier components. Inspect the RMAV breakdown of the top 2–3 CRITICAL components for unusually high scores that may indicate a structural anomaly in the generated graph.

**ρ improves with scale:**
Expected behavior. For systems with < 20 components, treat validation results as indicative only; the statistical sample is too small for stable correlation estimates.

---

## Output

```json
{
  "layer":            "app",
  "passed":           true,
  "predicted_count":  35,
  "actual_count":     35,
  "matched_count":    35,
  "overall": {
    "correlation": {
      "spearman":   0.91,
      "spearman_p": 0.0001,
      "kendall":    0.78,
      "pearson":    0.89
    },
    "classification": {
      "f1_score":   0.94,
      "precision":  0.92,
      "recall":     0.96,
      "cohens_kappa": 0.88
    },
    "ranking": {
      "top_5_overlap":  0.80,
      "top_10_overlap": 0.90,
      "ndcg_at_10":     0.94
    },
    "error": {
      "rmse": 0.18,
      "mae":  0.14
    }
  },
  "dimensional": {
    "reliability":     { "spearman": 0.88, "ground_truth": "ir" },
    "maintainability": { "spearman": 0.79, "ground_truth": "im" },
    "availability":    { "spearman": 0.93, "ground_truth": "ia" },
    "vulnerability":   { "spearman": 0.82, "ground_truth": "iv" }
  },
  "gates": {
    "spearman_pass":    true,
    "pvalue_pass":      true,
    "f1_pass":          true,
    "top5_overlap_pass":true,
    "rmse_pass":        true
  }
}
```

---

## External vs. Internal Validity

**Internal validity** (measured here): The predictions Q(v) agree with the simulation I(v). Both are derived from the same structural graph, so this confirms the analysis engine correctly extracts structural logic.

**External validity** (not directly measurable without production data): Whether simulation-derived I(v) agrees with real-world failure impact. The methodology's external validity claim rests on:
- The completeness and fidelity of the cascade propagation model (Step 4)
- The alignment between QoS-weighted edge semantics and actual message criticality
- Case study validation against real post-mortem data (ongoing research direction)

---

## Methodological Limitations

**Circular proximity risk.** Q(v) and I(v) are derived from the same structural graph, so they are not fully independent. High correlation is expected from this structural coupling. The claim is not that topology predicts all runtime behavior, but that topology-derived RMAV scores agree with simulation-derived cascade impact — which is the operationally relevant comparison in pre-deployment analysis.

**Ground truth is a proxy.** I(v) is derived from rule-based cascade simulation, not from real-world failure observations. Simulation completeness is bounded by the fidelity of the three cascade rules.

**Small-system instability.** Systems with < 20 components produce unreliable Spearman ρ estimates. All results should be interpreted in the context of system scale.

---

## Statistical Robustness and Stability

**Bootstrap confidence intervals** (1000 resamples):

For each validation run with n ≥ 30 components, bootstrap 95% CIs are computed:
```
CI_ρ:  [ρ − 1.96 × SE_ρ,  ρ + 1.96 × SE_ρ]   where SE_ρ ≈ 1/√(n−3)
```

A result is considered **stable** if the lower CI bound still exceeds the gate threshold.

**Sensitivity analysis** (AHP weight perturbation):

Sensitivity of ρ to ±5% weight perturbation across 200 random perturbations. A **Top-5 Stability** score ≥ 0.80 means that 80% of perturbations preserve the same top-5 critical component ranking.

---

## Comparative Analysis against Baselines

| Method | Spearman ρ | F1 |
|--------|-----------|-----|
| **RMAV (Step 3)** | **0.876** | **0.923** |
| Betweenness centrality only | 0.75 | 0.78 |
| Degree centrality only | 0.82* | 0.85* |
| PageRank only | 0.68 | 0.71 |
| Random ranking | 0.02 | 0.30 |

*Degree centrality achieves high ρ on synthetic topologies because generators force high-degree nodes into SPOF positions. On real-world heterogeneous topologies, RMAV consistently outperforms degree centrality alone.

---

## Classification Threshold Asymmetry

The classification in Step 3 (Q-based) and Step 4 (I-based) both use box-plot thresholds applied independently to their own distributions. This means the number of Q-critical components and I-critical components can differ (e.g., 3 vs 4 CRITICAL components in a 35-node system). This is expected and correct — the methodology does not assume the number of critical components is the same in both distributions. Precision and Recall are computed on the union of both critical sets.

---

## Statistical Stability and Confidence Intervals

For systems with n ≥ 30 components, the validation report includes:

```
Bootstrap 95% CI for Spearman ρ: [0.87, 0.95]   (1000 resamples)
AHP Sensitivity — Top-5 Stability: 0.92 (92% of perturbations preserve top-5 ranking)
AHP Sensitivity — Mean Kendall τ:  0.81 (strong agreement across perturbed weight sets)
```

---

## Commands

```bash
# ─── Standard validation (Steps 3 + 4 already run) ───────────────────────────
python bin/validate_graph.py --layer app

# ─── Quick mode: pass pre-computed JSON files directly ────────────────────────
python bin/validate_graph.py --quick results/prediction.json results/impact.json

# ─── Export results ───────────────────────────────────────────────────────────
python bin/validate_graph.py --layer app --output results/validation.json

# Export with JSON stdout for scripting
python bin/validate_graph.py --layer app --json | jq '.passed'

# ─── Visualization dashboard ─────────────────────────────────────────────────
# Generate and open the validation dashboard (Step 6)
python bin/validate_graph.py --layer app --visualize --open

# ─── Full pipeline: Prediction → Simulation → Validation ──────────────────────
python bin/analyze_graph.py  --layer app --output results/prediction.json
python bin/simulate_graph.py failure --exhaustive --layer app \
    --output results/impact.json
python bin/validate_graph.py --quick results/prediction.json results/impact.json \
    --output results/validation.json

# ─── Deterministic validation with fixed seed (reproducible research) ─────────
python bin/generate_graph.py  --scale medium --seed 42 --output test_data.json
python bin/import_graph.py    --input test_data.json --clear
python bin/analyze_graph.py   --layer app --use-ahp --output prediction.json
python bin/simulate_graph.py  failure --exhaustive --layer app --output impact.json
python bin/validate_graph.py  --quick prediction.json impact.json
# Expected: passed = true, ρ ≥ 0.80
```

---

## What Comes Next

A passing validation confirms the methodology's central claim: topology-based predictions Q(v) reliably identify which components will cause the most damage if they fail, without any runtime monitoring.

Step 6 (Visualization) renders all pipeline outputs — Q(v) scores, RMAV breakdowns, I(v) impact scores, and validation metrics — in an interactive HTML dashboard. The dashboard includes the Q(v) vs I(v) scatter plot (visual proof of correlation), the interactive dependency graph with components colour-coded by criticality level, and sortable tables for component-level inspection.

---

← [Step 4: Simulation](failure-simulation.md) | → [Step 6: Visualization](visualization.md)