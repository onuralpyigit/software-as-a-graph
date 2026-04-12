# Step 5: Validation

**Statistically prove that topology-based predictions agree with simulation-derived proxy ground truth.**

← [Step 4: Simulation](failure-simulation.md) | → [Step 6: Visualization](visualization.md)

---

## Table of Contents

1. [What This Step Does](#what-this-step-does)
2. [Two Ground-Truth Targets](#two-ground-truth-targets)
3. [Data Alignment](#data-alignment)
4. [Why Spearman Is the Primary Metric](#why-spearman-is-the-primary-metric)
5. [Metric Definitions](#metric-definitions)
   - [Correlation Metrics](#correlation-metrics)
   - [Classification Metrics](#classification-metrics)
   - [Ranking Metrics](#ranking-metrics)
   - [Error Metrics](#error-metrics)
6. [Dimension-Specific Specialist Metrics](#dimension-specific-specialist-metrics)
   - [Reliability: CCR@K and CME](#reliability-ccrk-and-cme)
   - [Maintainability: COCR@K, κ_CTA, BP](#maintainability-cocrk-κ_cta-bp)
   - [Availability: SPOF_F1, HSRR, DASA, RRI](#availability-spof_f1-hsrr-dasa-rri)
   - [Vulnerability: AHCR@K, FTR, APAR, CDCC](#vulnerability-ahcrk-ftr-apar-cdcc)
   - [Composite: Predictive Gain PG](#composite-predictive-gain-pg)
7. [Pass/Fail Gate System](#passfail-gate-system)
8. [Multi-Seed Stability Protocol](#multi-seed-stability-protocol)
9. [Node-Type Stratified Reporting](#node-type-stratified-reporting)
10. [Validation Targets by Layer and Scale](#validation-targets-by-layer-and-scale)
11. [Achieved Results](#achieved-results)
12. [Worked Example](#worked-example)
13. [Statistical Robustness](#statistical-robustness)
14. [Interpreting Results](#interpreting-results)
15. [Output Schema](#output-schema)
16. [External vs. Internal Validity](#external-vs-internal-validity)
17. [Commands](#commands)
18. [What Comes Next](#what-comes-next)

---

## What This Step Does

Validation answers the central question of the entire methodology: **do topology-based predictions actually work?**

It aligns predicted quality scores Q(v) from Step 3 with simulation-derived ground-truth scores from Step 4, then computes a structured suite of statistical metrics across four categories to measure agreement. A tiered gate system produces a clear **pass/fail** verdict.

```
Q(v) from Step 3                   I(v) / I*(v) from Step 4
────────────────────               ────────────────────────────
Q_RMAV(v) or Q_GNN(v)              I(v)   — composite cascade impact
R(v), M(v), A(v), V(v)             I*(v)  — per-dimension composite
                                   IR(v), IM(v), IA(v), IV(v)
          │                                     │
          └──────── Align by component ID ───────┘
                              │
          ┌───────────────────┼────────────────────┐
          │                   │                    │
    Correlation          Classification         Ranking
    ρ, τ, r             F1, Prec, Rec, κ      Top-K, NDCG@K
          │                   │                    │
          └───────────────────┼────────────────────┘
                              │
                    Dimension-Specific
                  CCR, CME, COCR, SPOF_F1
                  AHCR, FTR, APAR, CDCC, PG
                              │
                    Pass / Fail verdict
```

This step closes the methodological loop. Q(v) is derived from normalized graph metrics using AHP weights; I(v) is derived from cascade simulation outcomes on a separate graph view. Agreement between them provides genuine empirical evidence that topology alone predicts failure impact.

---

## Two Ground-Truth Targets

Step 4 produces two distinct ground-truth objects. Understanding the difference is essential for correctly interpreting validation results.

**I(v) — Overall simulation composite:**
```
I(v) = 0.35 × RL(v) + 0.25 × FR(v) + 0.25 × TL(v) + 0.15 × FD(v)
```
This is the direct output of the cascade simulation: reachability loss, fragmentation, throughput loss, and flow disruption combined. It answers "how much damage does v's failure cause?"

**I*(v) — Per-dimension composite:**
```
I*(v) = 0.25 × IR(v) + 0.25 × IM(v) + 0.25 × IA(v) + 0.25 × IV(v)
```
This is the equal-weighted combination of the four dimension-specific ground truths produced by the post-passes. IR(v) measures cascade dynamics, IM(v) measures change propagation, IA(v) measures connectivity disruption, IV(v) measures compromise spread. Together they are the simulation-side analog of Q(v)'s four-dimensional RMAV decomposition.

**Two distinct validation pairs:**

| Validation | Prediction | Ground Truth | What it measures |
|-----------|-----------|--------------|-----------------|
| Overall ranking | Q(v) | I(v) | Does topology predict cascade damage? |
| Composite | Q*(v) = Q(v) | I*(v) | Does RMAV decomposition match the full dimension-ground-truth structure? |
| Reliability | R(v) | IR(v) | Does R(v) rank cascade propagators correctly? |
| Maintainability | M(v) | IM(v) | Does M(v) rank change-fragile components correctly? |
| Availability | A(v) | IA(v) | Does A(v) identify SPOFs correctly? |
| Vulnerability | V(v) | IV(v) | Does V(v) rank attack targets correctly? |

> **Why two composites?** I(v) and I*(v) are not identical because they weight different phenomena. I(v) emphasizes reachability (0.35) which is closely tied to cascade breadth — biasing toward IR. I*(v) gives equal weight to all four dimensions. The Predictive Gain metric (§6.5) measures whether Q*(v) outperforms any single dimension predictor against I*(v), which is a strictly stronger claim than matching I(v) alone.

---

## Data Alignment

Before computing any metric, Q(v) and I(v) vectors are aligned by component ID:

```
predicted  = {id: Q(v)  for each component in Step 3 output}
actual     = {id: I(v)  for each component in Step 4 output}
matched    = {id: (Q(v), I(v))  for id in predicted ∩ actual}
```

- Components in predicted but not actual: logged as warnings (Step 4 may not have covered all layers).
- Components in actual but not predicted: logged as warnings (Step 3 may have used a different layer filter).
- `n < 5` matched components → validation aborted with an error.

Typical matched count: ≥ 95% of components when both steps use the same `--layer` flag.

---

## Why Spearman Is the Primary Metric

Spearman rank correlation ρ is the primary gate metric for three reasons:

**Scale independence.** Q(v) uses AHP-weighted topological metrics; I(v) uses cascade simulation weights. The two scales are not commensurable. Rank correlation avoids comparing absolute values across incompatible scales.

**Robustness to outliers.** A single extreme outlier (a node that causes catastrophic cascade) can distort Pearson r substantially; Spearman ρ is insensitive to outlier magnitude.

**Direct operational relevance.** The methodology's practical value is in correctly *ranking* components by criticality. An architect needs to know Component A is more critical than B — absolute scores are secondary.

Pearson r is still computed and reported for completeness. A high Pearson r alongside high Spearman ρ strengthens validity by showing magnitude agreement holds, not just ordering. It is not a gate metric.

---

## Metric Definitions

### Correlation Metrics

**Spearman Rank Correlation ρ:**
```
R_Q[i] = rank of Q(vᵢ) among all Q values  (average ranks for ties)
R_I[i] = rank of I(vᵢ) among all I values
d[i]   = R_Q[i] − R_I[i]
ρ = 1 − (6 × Σ d[i]²) / (n × (n² − 1))
```
Significance test: t = ρ√(n−2)/√(1−ρ²), df = n−2; p-value from two-tailed t-distribution.

Bootstrap 95% CI: 1000 resamples with fixed seed 42 (degraded to point estimate for n < 5).

| ρ Range | Interpretation |
|---------|---------------|
| 0.90–1.00 | Very strong agreement |
| 0.70–0.90 | Strong agreement (primary target zone) |
| 0.50–0.70 | Moderate agreement |
| < 0.50 | Weak/negligible agreement |

**Kendall τ (Tau-b):**
```
τ = (C − D) / √((C + D + T_Q)(C + D + T_I))
```
More conservative than ρ. A large ρ–τ gap (> 0.15) suggests a few dominant pairs are driving agreement — inspect top CRITICAL components specifically.

**Pearson r** — reported only, not a gate metric.

### Classification Metrics

Binary classification defines critical components from each distribution independently:
```
Q-critical: Q(v) > Q3_Q + 1.5 × IQR_Q
I-critical: I(v) > Q3_I + 1.5 × IQR_I
```

From the 2×2 confusion matrix:
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
Cohen's κ = (P_o − P_e) / (1 − P_e)
```

Bootstrap 95% CI on F1: 1000 resamples, seed 42. The CI is reported but does not gate the verdict.

### Ranking Metrics

**Top-K Overlap:**
```
Top-K Overlap = |top_K(Q) ∩ top_K(I)| / K
```
Measured at K=5 (primary gate) and K=10 (reported).

**NDCG@K:**
```
DCG@K  = Σᵢ₌₁ᴷ rel(i) / log₂(i+1)
NDCG@K = DCG@K / IDCG@K
```
rel(i) = I(v) of the component ranked i-th by Q(v). NDCG = 1.0 means perfect rank quality; positional errors are penalized logarithmically.

### Error Metrics

```
RMSE = √((1/n) × Σ (Q(vᵢ) − I(vᵢ))²)
MAE  = (1/n) × Σ |Q(vᵢ) − I(vᵢ)|
```
Because Q and I are on different scales, RMSE/MAE measure distributional similarity rather than absolute error. They complement rank correlation.

---

## Dimension-Specific Specialist Metrics

Each RMAV dimension has a set of specialist metrics that test properties beyond rank correlation. These are computed after the overall metrics and reported in the `dimensional` section of the output.

### Reliability: CCR@K and CME

Validates R(v) against IR(v).

**CCR@K — Cascade Capture Rate at K:**
```
CCR@K = |Top-K(R(v)) ∩ Top-K(IR(v))| / K
```
Measures what fraction of the K highest-predicted cascade propagators are also the K highest actual cascade propagators. Target: CCR@5 ≥ 0.80.

**CME — Cascade Magnitude Error:**
```
CME = (1/n) × Σ |rank_R(v) − rank_IR(v)| / n
```
Mean rank displacement normalized by system size. A CME near 0 means the predicted cascade ranking closely matches the simulation-observed cascade ordering. Target: CME ≤ 0.10.

### Maintainability: COCR@K, κ_CTA, BP

Validates M(v) against IM(v).

**COCR@K — Change Obligation Capture Rate at K:**
```
COCR@K = |Top-K(M(v)) ∩ Top-K(IM(v))| / K
```
Structurally identical to CCR@K for Reliability. Target: COCR@5 ≥ 0.75.

**κ_CTA — Weighted Coupling Tier Agreement:**
Three-tier classification (LOW/MEDIUM/HIGH coupling based on M(v) and IM(v) distributions) compared using weighted Cohen's κ, where tier distance weights adjacent misclassifications less than distant ones. Target: κ_CTA ≥ 0.55.

```
Tiers defined by tertiles of each distribution:
  LOW:    score ≤ 33rd percentile
  MEDIUM: 33rd < score ≤ 66th percentile
  HIGH:   score > 66th percentile

Weight matrix: w[i][j] = 1 − |i−j| / (n_tiers − 1)
κ_weighted = 1 − Σᵢⱼ wᵢⱼ × |obs[i][j] − exp[i][j]| / Σᵢⱼ wᵢⱼ × exp[i][j]
```

**BP — Bottleneck Precision:**
```
BP = |{v : BT(v) > τ_BT AND w_out(v) > τ_wout AND IM(v) > τ_IM}|
      / |{v : BT(v) > τ_BT AND w_out(v) > τ_wout}|
```
Precision of identifying components where high predicted bottleneck score (BT and w_out both above their 75th percentiles) corresponds to high actual change impact (IM(v) > 0.50). Target: BP ≥ 0.70.

### Availability: SPOF_F1, HSRR, DASA, RRI

Validates A(v) against IA(v).

**SPOF_F1 — SPOF Classification F1:**
```
SPOF-predicted:  AP_c_directed(v) > 0  (structural SPOF from Step 2)
SPOF-actual:     IA(v) > IA_threshold   (IA_threshold = Q3_IA + 0.5 × IQR_IA)

SPOF_F1 = 2 × SPOF_Precision × SPOF_Recall / (SPOF_Precision + SPOF_Recall)
```
Target: SPOF_F1 ≥ 0.90.

**HSRR — Hidden SPOF Recovery Rate:**
```
HSRR = |{v : AP_c_directed(v) = 0 AND QSPOF(v) > 0 AND IA(v) > IA_threshold}|
        / |{v : AP_c_directed(v) = 0 AND IA(v) > IA_threshold}|
```
Fraction of high-availability-impact components that are not binary articulation points but are nevertheless caught by QSPOF or CDI. Measures whether the continuous availability metrics (QSPOF, CDI) add value beyond binary AP detection. Target: HSRR ≥ 0.65.

**DASA — Directed SPOF Asymmetry Accuracy:**
```
DASA = |{v : sign(AP_c_out − AP_c_in) = sign(IA_out(v) − IA_in(v))}| / n
```
Checks that the directionality of the SPOF (whether out-reachability or in-reachability dominates) matches the directionality observed in simulation. Target: DASA ≥ 0.70.

**RRI — Redundancy Robustness Index:**
```
RRI = |{v : BR(v) = 0 AND IA(v) < IA_threshold}| / |{v : BR(v) = 0}|
```
Among components with no bridge edges (structurally redundant), what fraction also have low actual availability impact? High RRI confirms that structural redundancy (BR = 0) genuinely protects availability. Target: RRI ≥ 0.80.

### Vulnerability: AHCR@K, FTR, APAR, CDCC

Validates V(v) against IV(v).

**AHCR@K — Attack-Hub Capture Rate at K:**
```
AHCR@K = |Top-K(V(v)) ∩ Top-K(IV(v))| / K
```
Top-K overlap between predicted high-vulnerability components and actual high-compromise-reach components. Target: AHCR@5 ≥ 0.70.

**FTR — False Target Rate:**
```
FTR = |{v : V(v) > V_threshold AND IV(v) < IV_threshold}| / |{v : V(v) > V_threshold}|
```
Where V_threshold = 60th percentile of V(v) and IV_threshold = 10th percentile of IV(v). FTR measures the rate at which predicted high-vulnerability components turn out to have negligible actual compromise reach. Target: FTR ≤ 0.25.

**APAR — Attack Path Agreement Rate:**
```
APAR = |{v : V(v) > V_threshold AND v appears in any critical_path from Step 4}|
        / |{v : V(v) > V_threshold}|
```
Among high-V components, what fraction appear in at least one critical compromise propagation path identified by IV's post-pass? Target: APAR ≥ 0.60.

**CDCC — Cross-Dimensional Contamination Check:**
```
CDCC = ρ(A(v), V(v))
```
Spearman correlation between the Availability and Vulnerability dimension predictions. High CDCC (> 0.70) would indicate that A and V are not measuring orthogonal properties — a structural flaw in the RMAV decomposition. Target: CDCC ≤ 0.40 (lower is better — dimensions should be orthogonal).

### Composite: Predictive Gain PG

**PG — Predictive Gain:**
```
PG = ρ(Q*(v), I*(v)) − max_{d ∈ {R,M,A,V}} ρ(d(v), Id(v))
```
The increase in Spearman ρ achieved by the composite score Q*(v) over the single best-performing dimension predictor. PG > 0.03 confirms that combining all four RMAV dimensions adds predictive value beyond the strongest individual dimension alone. This is the key evidence that the four-dimensional decomposition is justified over a single-score approach.

Target: PG > 0.03.

**Why PG matters for the thesis:** A reviewer might argue that a simpler single-metric predictor (e.g., just RPR) would work as well as the full RMAV composite. A positive PG directly refutes this: it proves that the four-dimension decomposition captures structural information that no single dimension captures alone. Without PG, the multi-dimensional architecture of RMAV is an unverified design choice; with PG, it is an empirically validated one.

---

## Pass/Fail Gate System

Metrics are organized into four tiers. All Tier 1 gates must pass for a PASS verdict.

### Tier 1 — Primary Gates (all must pass)

| Gate | Metric | Threshold | Applies To |
|------|--------|-----------|-----------|
| G1 | Spearman ρ(Q, I) | ≥ 0.80 | Overall, application layer |
| G2 | p-value | ≤ 0.05 | Overall |
| G3 | F1-Score | ≥ 0.90 | Overall |
| G4 | Top-5 Overlap | ≥ 0.60 | Overall |

### Tier 2 — Secondary Gates (reported with pass/fail; do not block PASS)

| Gate | Metric | Threshold | Applies To |
|------|--------|-----------|-----------|
| G5 | RMSE | ≤ 0.25 | Overall |
| G6 | Predictive Gain PG | > 0.03 | Composite vs. I*(v) |
| G7 | SPOF_F1 | ≥ 0.90 | Availability dimension |
| G8 | CCR@5 | ≥ 0.80 | Reliability dimension |

### Tier 3 — Dimension-Specific Targets

These are used for per-dimension assessment and ICSA 2026 evidence set. They do not gate the overall verdict but are required to pass their respective dimension validations.

| Dimension | Metric | Target | Ground Truth |
|-----------|--------|--------|-------------|
| Overall | ρ(Q, I) | ≥ 0.80 | I(v) |
| Reliability | ρ(R, IR) | ≥ 0.75 | IR(v) |
| Reliability | CCR@5 | ≥ 0.80 | IR(v) |
| Reliability | CME | ≤ 0.10 | IR(v) |
| Maintainability | ρ(M, IM) | ≥ 0.72 | IM(v) |
| Maintainability | COCR@5 | ≥ 0.75 | IM(v) |
| Maintainability | κ_CTA | ≥ 0.55 | IM(v) |
| Maintainability | BP | ≥ 0.70 | IM(v) |
| Availability | ρ(A, IA) | ≥ 0.82 | IA(v) |
| Availability | SPOF_F1 | ≥ 0.90 | IA(v) |
| Availability | HSRR | ≥ 0.65 | IA(v) |
| Availability | DASA | ≥ 0.70 | IA(v) |
| Availability | RRI | ≥ 0.80 | IA(v) |
| Vulnerability | ρ(V, IV) | ≥ 0.70 | IV(v) |
| Vulnerability | AHCR@5 | ≥ 0.70 | IV(v) |
| Vulnerability | FTR | ≤ 0.25 | IV(v) |
| Vulnerability | APAR | ≥ 0.60 | IV(v) |
| Vulnerability | CDCC | ≤ 0.40 | Orthogonality |
| Composite | ρ(Q*, I*) | ≥ 0.85 | I*(v) |
| Composite | PG | > 0.03 | I*(v) |

### Tier 4 — Reported Metrics (no gate)

Kendall τ, Pearson r, Precision, Recall, Cohen's κ, Top-10 Overlap, NDCG@K, MAE, all CI bounds.

---

## Multi-Seed Stability Protocol

A single seed (42) provides no evidence of result stability. The thesis defense requires multi-seed testing to demonstrate that validation results are reproducible, not seed-dependent artifacts.

**Required seeds:** {42, 123, 456, 789, 2024}

For each seed s, generate a synthetic topology at the target scale, run the full pipeline (Steps 1–5), and record ρ(Q, I).

**Stability criterion:**
```
μ_ρ = mean(ρ₄₂, ρ₁₂₃, ρ₄₅₆, ρ₇₈₉, ρ₂₀₂₄)
σ_ρ = std(ρ₄₂, ρ₁₂₃, ρ₄₅₆, ρ₇₈₉, ρ₂₀₂₄)

Pass: μ_ρ ≥ 0.80  AND  σ_ρ ≤ 0.05
```

A standard deviation ≤ 0.05 confirms the methodology is stable across different random graph instantiations of the same scale — a pre-condition for the results to be generalizable rather than lucky.

**Expected results at medium scale:**

| Seed | ρ(Q, I) | F1 |
|------|---------|----|
| 42 | ≈ 0.847 | ≈ 0.85 |
| 123 | ≈ 0.831 | ≈ 0.82 |
| 456 | ≈ 0.858 | ≈ 0.88 |
| 789 | ≈ 0.843 | ≈ 0.84 |
| 2024 | ≈ 0.852 | ≈ 0.86 |
| **μ ± σ** | **0.846 ± 0.010** | **0.85 ± 0.02** |

Multi-seed stability at σ ≤ 0.05 is also the minimum evidence required for GNN validation in the ICSA 2026 submission.

**Running the multi-seed protocol:**
```bash
for seed in 42 123 456 789 2024; do
    python bin/generate_graph.py --scale medium --seed $seed --output input/system_s${seed}.json
    python bin/import_graph.py --input input/system_s${seed}.json --clear
    python bin/analyze_graph.py  --layer app --use-ahp --output results/pred_s${seed}.json
    python bin/simulate_graph.py failure --exhaustive --layer app \
                                  --output results/sim_s${seed}.json
    python bin/validate_graph.py results/pred_s${seed}.json results/sim_s${seed}.json \
                           --output results/val_s${seed}.json
done
# Then aggregate: python bin/multi_seed_summary.py results/val_s*.json
```

---

## Node-Type Stratified Reporting

After Step 1's Rule 5 (app_to_lib), Library nodes now appear in the DEPENDS_ON graph with non-zero in-degree. They are scored by Step 3 and simulated by Step 4. Their validation must be reported separately from Application nodes because their failure semantics differ (simultaneous blast vs. sequential cascade).

**Stratified ρ reporting (system layer):**

```
ρ_app   = ρ(Q_Application,  I_Application)   — target ≥ 0.80
ρ_lib   = ρ(Q_Library,      I_Library)        — target ≥ 0.70 (fewer components, less stable)
ρ_broker = ρ(Q_Broker,      I_Broker)         — target ≥ 0.72
ρ_infra  = ρ(Q_Node,        I_Node)           — target ≥ 0.54
```

**Why Library targets are lower:** Library components are fewer in number per system (typically 3–15 at medium scale), making rank correlation estimates statistically noisier. The ≥ 0.70 target is appropriate given sample size. As the Library population grows in larger test systems, this target should be raised toward the Application layer target.

**Reporting guidance:** The overall validation verdict is determined by the application-layer primary gates. Stratified reporting is required in the thesis and ICSA 2026 submission to demonstrate the methodology handles Library nodes correctly and to document the known accuracy gap between node types.

---

## Validation Targets by Layer and Scale

### Layer Targets

| Layer | Spearman ρ | F1 | Notes |
|-------|-----------|-----|-------|
| `app` | ≥ 0.80 | ≥ 0.90 | Primary validation layer — all primary gates apply here |
| `app` (large/xlarge) | ≥ 0.90 | ≥ 0.92 | Scale benefit expected |
| `infra` | ≥ 0.54 | ≥ 0.68 | Physical topology is more homogeneous |
| `mw` | ≥ 0.70 | ≥ 0.85 | Broker-level analysis |
| `system` | ≥ 0.75 | ≥ 0.88 | Cross-layer, includes Topics and Libraries |

### Scale Matrix (Application Layer)

| Test ID | Scale | ρ target | F1 target | Top-5 target | Seed |
|---------|-------|----------|-----------|--------------|------|
| VT-APP-01 | Small (10–25) | ≥ 0.75 | ≥ 0.80 | ≥ 50% | 42 |
| VT-APP-02 | Medium (30–50) | ≥ 0.80 | ≥ 0.88 | ≥ 60% | 42 |
| VT-APP-03 | Large (60–100) | ≥ 0.85 | ≥ 0.90 | ≥ 70% | 42 |
| VT-SYS-01 | System/Small | ≥ 0.70 | ≥ 0.78 | ≥ 50% | 42 |
| VT-SYS-02 | System/Medium | ≥ 0.75 | ≥ 0.85 | ≥ 60% | 42 |
| VT-INF-01 | Infra/Any | ≥ 0.54 | ≥ 0.68 | ≥ 35% | 42 |

---

## Achieved Results

Results from the eight validated domain scenarios (IEEE RASSE 2025, application layer, large scale):

| Scenario | Domain | Scale | ρ(Q,I) | F1 | Top-5 | PASS |
|----------|--------|-------|--------|-----|-------|------|
| 01 | Autonomous Vehicle (ROS 2) | Medium | 0.871 | 0.923 | 0.80 | ✓ |
| 02 | IoT Smart City | Large | 0.883 | 0.931 | 0.80 | ✓ |
| 03 | Financial HFT (Kafka) | Medium | 0.856 | 0.912 | 0.80 | ✓ |
| 04 | Healthcare (MQTT) | Medium | 0.868 | 0.905 | 0.80 | ✓ |
| 05 | Hub-and-Spoke | Medium | 0.901 | 0.947 | 1.00 | ✓ |
| 06 | Microservices mesh | Medium | 0.843 | 0.894 | 0.60 | ✓ |
| 07 | Enterprise (XLarge) | XLarge | **0.943** | **0.962** | 1.00 | ✓ |
| 08 | Tiny regression | Tiny | 0.820 | 0.900 | 0.60 | ✓ |
| **Overall** | — | — | **0.876** | **0.923** | **0.80** | **✓** |

**Scale trend:** ρ improves from ≈0.787 (small) to ≈0.876 (xlarge). This is expected: larger systems provide richer structural context, making relative centrality differences more pronounced and rank correlations more stable. This scale benefit is a key thesis contribution (REQ-ACC-05).

---

## Worked Example

**Application layer, Distributed Intelligent Factory (DIF), 32 components:**

```
Step 1 — Get Q(v) from Step 3:
  DataRouter:     Q = 0.84  [CRITICAL]
  SensorHub:      Q = 0.73  [CRITICAL]
  CommandBus:     Q = 0.73  [CRITICAL]
  PLC_Controller: Q = 0.67  [HIGH]

Step 2 — Get I(v) from Step 4:
  DataRouter:     I = 0.88  [CRITICAL]
  SensorHub:      I = 0.79  [CRITICAL]
  PLC_Controller: I = 0.68  [CRITICAL]   ← one rank higher than predicted
  CommandBus:     I = 0.61  [HIGH]

Step 3 — Align (n=32, all matched)

Step 4 — Compute:
  Spearman ρ   = 0.91     p = 0.0001   G1: PASS
  F1-Score     = 0.94                  G3: PASS
  Top-5 Overlap = 4/5 = 0.80           G4: PASS
  RMSE         = 0.18                  G5: PASS (secondary)
  Kendall τ    = 0.78   gap = 0.13     (acceptable — below 0.15 concern threshold)

  Reliability dim: ρ(R, IR) = 0.88   CCR@5 = 0.80   CME = 0.07
  Availability dim: ρ(A, IA) = 0.92  SPOF_F1 = 0.94

  Composite:  ρ(Q*, I*) = 0.93
  Best single dim: ρ(A, IA) = 0.92
  PG = 0.93 − 0.92 = 0.01           (marginal — A dominates for this system)

Verdict: PASS ✓ — all four primary gates satisfied
```

The single ordering discrepancy (PLC_Controller ranked 4th by Q but 3rd by I) is typical and within bounds for a 32-component system. The small PG (0.01) signals that Availability is the dominant predictive dimension for this factory scenario — consistent with the topology being heavily infrastructure-dependent.

---

## Statistical Robustness

**Bootstrap confidence intervals:** Spearman ρ and F1 both carry 95% bootstrap CIs (1000 resamples, seed 42). A CI that does not cross the gate threshold provides stronger evidence than a point estimate alone.

**ρ–τ gap diagnostic:** When |ρ − τ| > 0.15, agreement is driven by a small number of extreme components. In this case, inspect the top 2–3 CRITICAL components — one of them may be an extreme outlier driving the rank correlation. This is not necessarily a failure mode, but it should be reported and interpreted.

**Classification threshold asymmetry:** The classification boundary (box-plot upper fence) is computed independently from Q(v) and I(v) distributions. This means the number of Q-critical and I-critical components can differ. When |Q-critical count − I-critical count| / n > 0.10, interpret F1 alongside Precision and Recall separately: a low F1 driven by low Recall means the prediction is conservative (misses some critical components); driven by low Precision means it is over-sensitive (flags too many).

**Small-scale instability:** At n < 20, box-plot quartile estimates are unreliable. The percentile fallback classification (top 10% = CRITICAL) is used instead. For these systems, treat F1 as indicative only; ρ is the primary robustness indicator.

**Internal vs. external validity:** Both Q(v) and I(v) are derived from the same structural graph. High ρ confirms the analysis engine correctly extracts structural logic, but does not directly prove that the predicted criticality order matches real-world incident impact. External validation against production post-mortems is the next research direction.

---

## Interpreting Results

**All primary gates pass:** Predictions from Step 3 are reliable for this system. Use Q(v) scores and criticality classifications for architectural decision-making.

**ρ passes but F1 fails (e.g., ρ=0.81, F1=0.72):** The ranking is correct but the binary classification threshold is misaligned. Check whether box-plot classification in Step 3 is using `--use-ahp` and whether layer size meets the ≥12 component threshold for normal-path classification.

**ρ fails (< 0.80):** Investigate by layer. If `app` fails but `infra` passes, the issue is likely in RMAV weights for specific dependency types — try `--use-ahp` with domain-specific pairwise matrices. If all layers fail, check Step 1 graph construction for missing dependency derivation rules.

**Large ρ–τ gap (> 0.15):** Agreement is driven by extreme outlier components. Inspect RMAV breakdown of top 2–3 CRITICAL components for unusually high scores.

**PG ≤ 0.03:** The composite does not outperform the single best dimension. This usually means one RMAV dimension is much more relevant for this system type than the others (e.g., Availability dominates in infrastructure-heavy systems). Report which dimension is dominant and justify why in the thesis.

**Low SPOF_F1 with high ρ:** The global ordering is correct but SPOFs are not being identified accurately. Check that AP_c_directed is being stored from Step 2 (not just the binary is_articulation_point flag) and that CDI is enriching the availability signal.

**ρ improves with scale:** Expected behavior. For systems with < 20 components, treat validation results as indicative only.

---

## Output Schema

```json
{
  "layer": "app",
  "passed": true,
  "predicted_count": 35,
  "actual_count": 35,
  "matched_count": 35,
  "overall": {
    "correlation": {
      "spearman": 0.91, "spearman_p": 0.0001,
      "spearman_ci_lower": 0.84, "spearman_ci_upper": 0.95,
      "kendall": 0.78, "kendall_spearman_gap": 0.13,
      "pearson": 0.89
    },
    "classification": {
      "f1_score": 0.94, "precision": 0.92, "recall": 0.96,
      "cohens_kappa": 0.88,
      "f1_ci_lower": 0.88, "f1_ci_upper": 0.97
    },
    "ranking": {
      "top_5_overlap": 0.80, "top_10_overlap": 0.90, "ndcg_at_10": 0.94
    },
    "error": { "rmse": 0.18, "mae": 0.14 }
  },
  "dimensional": {
    "reliability": {
      "spearman": 0.88, "ground_truth": "IR(v)",
      "ccr_5": 0.80, "cme": 0.07
    },
    "maintainability": {
      "spearman": 0.79, "ground_truth": "IM(v)",
      "cocr_5": 0.78, "weighted_kappa_cta": 0.62, "bottleneck_precision": 0.74
    },
    "availability": {
      "spearman": 0.93, "ground_truth": "IA(v)",
      "spof_f1": 0.94, "spof_precision": 0.95, "spof_recall": 0.93,
      "hsrr": 0.70, "dasa": 0.76, "rri": 0.85
    },
    "vulnerability": {
      "spearman": 0.82, "ground_truth": "IV(v)",
      "ahcr_5": 0.80, "ftr": 0.18, "apar": 0.67, "cdcc": 0.22
    }
  },
  "composite": {
    "spearman_q_star_i_star": 0.93,
    "predictive_gain": 0.01,
    "best_single_dim": "availability",
    "interdim_max_correlation": 0.31
  },
  "gates": {
    "G1_spearman": true, "G2_pvalue": true,
    "G3_f1": true, "G4_top5": true,
    "G5_rmse": true, "G6_predictive_gain": false,
    "G7_spof_f1": true, "G8_ccr5": true
  },
  "node_type_stratified": {
    "Application": { "n": 28, "spearman": 0.91 },
    "Library":     { "n": 4,  "spearman": 0.76 },
    "Broker":      { "n": 3,  "spearman": 0.83 }
  }
}
```

---

## External vs. Internal Validity

**Internal validity** (measured here): Q(v) agrees with I(v). Both derive from the same graph, so this confirms the analysis engine correctly extracts structural logic.

**External validity** (not directly measurable without production data): Whether simulation-derived I(v) agrees with real-world failure impact. External validity is the next research direction. Approaches include: matching against known incidents in open-source ROS 2 systems, comparing predictions against deployment-time SLO breach records in financial trading platforms, and user studies with reliability engineers validating CRITICAL classifications against their intuition.

---

## Commands

```bash
# ─── Single run — topology-only baseline ─────────────────────────────────────
python bin/validate_graph.py single --input input/scenarios/atm_system.json

# ─── Single run — QoS-enriched (v7/v4 formulas) ─────────────────────────────
python bin/validate_graph.py single --input input/scenarios/atm_system.json --qos

# ─── Multi-seed stability sweep ──────────────────────────────────────────────
python bin/validate_graph.py sweep --input input/scenarios/atm_system.json --qos

# ─── Full report (sweep + topology-class gates + node-type strata) ──────────
python bin/validate_graph.py report --input input/scenarios/atm_system.json \
    --output output/validation_report.json --qos

# ─── Ablation Study (Topo-only vs. QoS-enriched + LaTeX Export) ──────────────
python bin/validate_graph.py compare --input input/scenarios/atm_system.json \
    --seeds 42,123,456,789,2024 --latex

# ─── Advanced options ────────────────────────────────────────────────────────
python bin/validate_graph.py report --input input/scenarios/atm_system.json \
    --top-k 10 --cascade 10 --bootstrap 5000 --verbose
```

---

## What Comes Next

Step 5 produces a structured validation report: primary gate verdicts, per-dimension ρ values, specialist metrics for each RMAV dimension, composite PG, multi-seed stability results, and node-type stratified correlation. This report is the empirical evidence base for the dissertation's central claim.

Step 6 (Visualization) renders these results into an interactive HTML dashboard with a correlation scatter plot of Q(v) vs. I(v), a ranked component table with RMAV breakdown and criticality classification, a dependency matrix heatmap sorted by composite score, and the complete validation metric panel. It also renders the anti-pattern catalogue with detected instances highlighted on the topology graph.

---

← [Step 4: Simulation](failure-simulation.md) | → [Step 6: Visualization](visualization.md)