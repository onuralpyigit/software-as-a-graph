# Step 5: Validate

**Statistically prove that topology-based predictions agree with simulation-derived cascade impact.**

← [Step 4: Simulate](failure-simulation.md) | → [Step 6: Visualize](visualization.md)

---

## Table of Contents

1. [What This Step Does](#1-what-this-step-does)
2. [Validation Pipeline Overview](#2-validation-pipeline-overview)
3. [Ground Truth: I(v) from Cascade Simulation](#3-ground-truth-iv-from-cascade-simulation)
4. [RMAV Prediction: Q(v)](#4-rmav-prediction-qv)
5. [Statistical Battery](#5-statistical-battery)
   - 5.1 [Rank Correlation — Spearman ρ and Kendall τ](#51-rank-correlation--spearman-ρ-and-kendall-τ)
   - 5.2 [Bootstrap Confidence Interval](#52-bootstrap-confidence-interval)
   - 5.3 [Classification Metrics — Precision, Recall, F1 @ K](#53-classification-metrics--precision-recall-f1--k)
   - 5.4 [SPOF-F1](#54-spof-f1)
   - 5.5 [Specialist Metrics — ICR@K, BCE, FTR, PG](#55-specialist-metrics--icrk-bce-ftr-pg)
   - 5.6 [Wilcoxon Signed-Rank Test](#56-wilcoxon-signed-rank-test)
6. [Node-Type Stratified Reporting](#6-node-type-stratified-reporting)
7. [Topology-Class Gate System](#7-topology-class-gate-system)
8. [Multi-Seed Stability Sweep](#8-multi-seed-stability-sweep)
9. [Ablation Study: Topology-Only vs. QoS-Enriched](#9-ablation-study-topology-only-vs-qos-enriched)
10. [CLI Reference](#10-cli-reference)
11. [Output Schema](#11-output-schema)
12. [Interpreting Results](#12-interpreting-results)
13. [What Comes Next](#13-what-comes-next)

---

## 1. What This Step Does

Step 5 closes the methodological loop. It aligns two independently derived signals for every
component in the system:

| Signal | Source | What it represents |
|--------|--------|-------------------|
| **Q(v)** | RMAV formula applied to topology (Analyze stage, Step 2) | *Predicted* criticality — computed deterministically from graph structure alone, before any runtime data |
| **Q_ens(v)** | GNN ensemble (Predict stage, Step 3, optional) | *Refined prediction* — inductive GNN blend; compared against I(v) in addition to or instead of Q(v) when available |
| **I(v)** | Stochastic cascade simulation (Simulate stage, Step 4) | *Proxy ground truth* — normalised damage score obtained by injecting each node as the failure origin |

High statistical agreement between Q(v) and I(v) is empirical evidence that **topology alone predicts
failure impact** — the central claim of the Software-as-a-Graph thesis.

```
   Graph (Step 1)
        │
   ┌────┴──────────────────────────────┐
   │ Step 2: Analyze (RMAV)            │  Step 4: Simulate (Cascade)
   │   Q(v) = w·R + w·M + w·A + w·V   │    I(v) = mean impact over N_repeats
   └────┬──────────────────────────────┘             simulation seeds
        │    [optional: Step 3 Predict]
        │    Q_ens(v) = α·Q_GNN + (1-α)·Q(v)
        │                   │
        └────────┬──────────┘
                 │
          Statistical Battery
          ─────────────────────
          Spearman ρ,  Kendall τ
          Bootstrap 95% CI
          F1@K, SPOF-F1, FTR
          ICR@K, BCE, PG
          Wilcoxon vs. degree baseline
                 │
          Gate Evaluation
          (topology-class adaptive)
                 │
          PASS / FAIL verdict
```

A compound test is used because no single metric is sufficient:
- **ρ** confirms the global rank ordering is preserved.
- **F1@K** confirms the top-K critical components are correctly identified.
- **PG** confirms Q(v) outperforms the naïve degree-centrality baseline.
- **SPOF-F1** confirms structural SPOFs are correctly caught.

---

## 2. Validation Pipeline Overview

`cli/validate_graph.py` implements the full pipeline as a single self-contained CLI.  
It loads a graph JSON, derives I(v) by running the `FaultInjector` for every node, computes Q(v) via
the central `QualityAnalyzer`, then runs the statistical battery.

```
load_graph(system.json)
    │
    ├── compute_rmav(G, qos=False|True)   →  Q(v), R(v), M(v), A(v), V(v)
    │       uses QualityAnalyzer + StructuralMetrics
    │
    └── derive_ground_truth(G, seed, n_repeats=5)  →  I(v)
            uses FaultInjector._inject_node() per node
            I(v) = mean(composite_impact) over n_repeats seeds
            │
run_statistical_tests(node_scores, top_k)
    │
    ├── Rank correlation  →  ρ, τ, CI
    ├── Classification    →  Precision, Recall, F1, FTR
    ├── SPOF-F1
    ├── ICR@K, BCE, PG
    └── Wilcoxon vs. degree centrality
    │
stratified_metrics(by node type)
classify_topology(G)    →   "sparse" | "medium" | "dense" | "hub_spoke"
evaluate_gates(vr, topo_class)
```

> **Independence guarantee (composite and R/A dimensions).** Q(v) uses only the graph structure (PageRank, betweenness, degree, articulation points) and optionally QoS contract attributes. The composite I(v), IR(v), and IA(v) are produced by simulations that operate on G_structural (raw pub-sub edges) and have no access to Q(v). Measuring ρ(Q\*, I\*), ρ(R, IR), and ρ(A, IA) is a genuine empirical test — not a consistency check.
>
> IM(v) and IV(v) are derived from the same DEPENDS_ON graph as M(v) and V(v) (`ChangePropagationSimulator` traverses G^T of DEPENDS_ON with an `instability`-based stop condition shared with M(v)'s CouplingRisk; `CompromisePropagationSimulator` traverses the same G^T with a trust-threshold on DEPENDS_ON edge weights used by V(v)'s QADS). ρ(M, IM) and ρ(V, IV) are therefore **internal consistency checks**: they confirm structural alignment between the RMAV predictor and a simulation proxy that shares the same graph substrate. This does not invalidate them — alignment on a shared substrate still provides useful signal — but they cannot claim the same methodological independence as the composite or R/A correlations.

---

## 3. Ground Truth: I(v) from Cascade Simulation

### Simulation Mechanics

For each node v, the `FaultInjector` runs a BFS cascade simulation in two sequential phases per wave:

**Phase A — Direct propagation (Stochastic):** *(optional extension — **disabled by default**)*

> [!NOTE]
> **Phase A is a no-op in the default configuration.** The propagation probability for pure `DEPENDS_ON` / `USES` edges is `prob = 0.0` (see `saag/simulation/fault_injector.py`). When `prob = 0.0`, no node ever fails through a bare dependency edge — Phase A contributes zero cascade events. This means the independence claim (§2 note, H5) holds: $I(v)$ is derived entirely from pub-sub topology (Phase B), which shares no computational path with the RMAV predictor $Q(v)$.
>
> Phase A exists as an extension point for future work where a non-zero `prob` is desirable — e.g. modelling compile-time dependency chains or shared-library ABI breaks. To activate it, set `prob > 0` in the `FaultInjector` constructor. Doing so changes the independence character of ρ(M, IM) and ρ(V, IV) from a consistency check to a partially-coupled measurement; this should be disclosed in any publication.

Failure spreads from failed nodes along `DEPENDS_ON` and `USES` edges stochastically. The propagation probability in pure dependency edges is `prob * depth_damp`. By default, `prob` is set to `0.0` (disabled).

**Phase B — Topic-mediated Soft QoS/Rate-weighted Propagation:**  
1. **Continuous Topic Feed Loss**:
   For each topic $t$, the feed loss $L(t) \in [0.0, 1.0]$ is calculated dynamically:
   - If the topic has publishers:
     $$L(t) = \frac{\sum_{p \in \text{failed\_publishers}(t)} \text{rate\_hz}(p, t)}{\sum_{p \in \text{all\_publishers}(t)} \text{rate\_hz}(p, t)}$$
     where `rate_hz` is the publish rate. If the total rate is 0, it falls back to the fraction of failed publishers.
   - If the topic has no publishers but has broker routers, the loss is the fraction of failed routers:
     $$L(t) = \frac{|\text{failed\_routers}(t)|}{|\text{all\_routers}(t)|}$$
   - The loss is then scaled by the topic's QoS criticality factor and capped at 1.0:
     $$L(t) = \min(1.0, L(t) \times \text{QoS\_factor}(t))$$
     where $\text{QoS\_factor}(t)$ is computed from reliability (`RELIABLE` multiplier `1.2`) and priority (`HIGH`/`CRITICAL`/`URGENT` multiplier `1.15`, `MEDIUM` multiplier `1.05`).
2. **Orphaned Topic and Subscriber Impact Tracking**:
   - If $L(t) > 10^{-6}$ and the topic was not previously orphaned, it is added to `orphaned_topics`.
   - All subscriber applications of $t$ that are not already failed are marked as impacted.
3. **Stochastic Subscriber Failure**:
   For each subscriber application $s$, we compute its average feed loss across all its subscribed topics:
   $$\text{sub\_loss}(s) = \frac{\sum_{t \in \text{subscribed\_topics}(s)} L(t)}{|\text{subscribed\_topics}(s)|}$$
   If $\text{sub\_loss}(s) \ge \text{propagation\_threshold}$ (default `0.2`):
   - The subscriber fails stochastically with probability:
     $$P_{\text{fail}}(s) = \min\left(1.0, \frac{\text{sub\_loss}(s)}{\text{propagation\_threshold}}\right) \times \text{depth\_damp}$$
     Where $\text{depth\_damp} = \max(0.25, 1.0 - \text{wave\_idx} \times 0.15)$ is a depth-based damping factor to prevent runaway cascade propagation.

### Ground Truth Derivation

To obtain the ground truth, the validation pipeline runs the exhaustive fault injection across all candidate nodes:

```python
rng_seeds = [seed + i * 37 for i in range(n_repeats)]   # default n_repeats = 5

for each node v:
    impacts = []
    for s in rng_seeds:
        imp, depth, affected = simulate_cascade(G, v, depth_limit, seed=s)
        impacts.append(imp)
    I(v) = mean(impacts)
```

Averaging across `n_repeats` seeds dampens stochastic variance and yields a stable mean impact estimate. This is the value compared against Q(v) in all subsequent statistical tests.

### What I(v) Represents

`I(v)` is the **normalised cascade impact score** ∈ [0, 1] measuring how much of the system becomes unreachable or impaired when node $v$ fails. It is the `composite_impact` property of `ImpactMetrics`:

$$I(v) = 0.35 \cdot \text{reachability\_loss} + 0.25 \cdot \text{fragmentation} + 0.25 \cdot \text{throughput\_loss} + 0.15 \cdot \text{flow\_disruption}$$

Weights are AHP-derived (see `saag/prediction/weight_calculator.py` `criteria_impact`).

---

## 4. RMAV Prediction: Q(v)

Q(v) is computed by `QualityAnalyzer` using a four-dimensional formula:

```
Q(v) = w_A × A(v)  +  w_R × R(v)  +  w_M × M(v)  +  w_V × V(v)
```

**AHP-derived weights (default):**

| Dimension | Weight | Rationale |
|-----------|:------:|-----------|
| Availability (A) | **0.43** | SPOF severity dominates pre-deployment risk |
| Reliability (R) | **0.24** | Cascade propagation reach |
| Maintainability (M) | **0.17** | Coupling complexity; long-term fragility |
| Vulnerability (V) | **0.16** | Security exposure surface |

### Latest formula versions (Middleware 2026):

#### Reliability R(v) — Fault Propagation Risk
- **Standard formula** (Application, Broker, Node, Library):
  $$R(v) = 0.60 \times PR(v) \times (1 + MPCI(v)) + 0.40 \times DG\_in(v)$$
- **Topic formula** (Topic nodes only):
  $$R_{\text{topic}}(v) = 0.50 \times FOC(v) + 0.50 \times CDPot\_topic(v)$$
  Where $CDPot\_topic(v) = FOC(v) \times (1 - \min(publisher\_count\_norm(v), 1.0))$.

#### Maintainability M(v) — Coupling Complexity
$$M(v) = 0.35 \times BT(v) + 0.30 \times w\_out(v) + 0.15 \times CQP(v) + 0.12 \times CouplingRisk\_enh(v) + 0.08 \times (1 - CC(v))$$
$$CQP(v) = 0.10 \times loc\_norm(v) + 0.35 \times complexity\_norm(v) + 0.30 \times instability\_code(v) + 0.25 \times lcom\_norm(v)$$

#### Availability A(v) — SPOF Risk
$$A(v) = 0.35 \times AP\_c\_directed(v) + 0.25 \times QSPOF(v) + 0.25 \times BR(v) + 0.10 \times CDI(v) + 0.05 \times w(v)$$
Where:
- **AP_c_directed(v)** — Directed Articulation Point score (worst-case directed graph connectivity loss when $v$ is removed).
- **QSPOF(v)** — QoS-amplified SPOF severity: `AP_c_directed(v) × w(v)`.
- **BR(v)** — Bridge Ratio (fraction of edges incident to $v$ that are bridges).
- **CDI(v)** — Connectivity Degradation Index (average path length increase when $v$ is removed).
- **w(v)** — Component QoS weight from Step 1.

#### Vulnerability V(v) — Security Exposure
$$V(v) = 0.40 \times REV(v) + 0.35 \times RCL(v) + 0.25 \times w\_in(v)$$
Where:
- **REV(v)** — Reverse Eigenvector Centrality (downstream attack propagation reach).
- **RCL(v)** — Reverse Closeness Centrality (adversarial entry proximity).
- **w_in(v)** — QoS-weighted in-degree (QADS - QoS-weighted Attack-Dependent Surface).

See [docs/structural-analysis.md](structural-analysis.md) for the complete formula reference.

**Topology-only vs. QoS-enriched modes:**

| Mode | `--qos` flag | PSPOF contribution |
|------|:-----------:|--------------------|
| Topology-only baseline | off (default) | `PSPOF = 0` for all nodes |
| QoS-enriched | on | `PSPOF` computed from pub-sub topology (used as diagnostic or GNN feature) |

The ablation study (`compare` subcommand) measures the predictive lift from the QoS-enriched mode.

---

## 5. Statistical Battery

All statistics are computed on **Application-type nodes** by default, falling back to all nodes only
when fewer than 4 Application nodes exist. This matches the thesis claim: topology predicts
*application-layer* cascade criticality.

### 5.1 Rank Correlation — Spearman ρ and Kendall τ

**Spearman ρ** is the primary gate metric. It measures whether the *rank ordering* of components by
Q(v) matches the rank ordering by I(v).

```
ρ = 1  −  (6 × Σ dᵢ²) / (n × (n² − 1))

where  dᵢ = rank(Q(vᵢ)) − rank(I(vᵢ))
```

Significance test: two-tailed t-distribution with df = n − 2.

The interpretation of an absolute ρ value depends on the **ground-truth source regime**. Two regimes apply:

**Regime A — RMAV pipeline against simulation labels (Q(v) vs. I(v), composite or IA/IR).**
Ground truth is a stochastic cascade simulator; achievable ρ is bounded by simulator noise and topology decoupling. The primary criterion in this regime is G1 (§5.7): pass if ρ ≥ 0.70. Absolute levels above that are informative but not a quality gate.

| ρ Range | Interpretation |
|---------|---------------|
| ≥ 0.85  | Very strong — well above G1 |
| 0.70–0.85 | Acceptable — G1 passes |
| 0.60–0.70 | Borderline — G1 fails; check topology class and node-type filter |
| < 0.60  | Weak — investigation required |

**Regime B — Learned/GNN models against simulation labels (Q*(v) vs. I*(v), Middleware evaluation).**
Against stochastic Sim labels in decoupled pub-sub topologies, absolute ρ is constrained by simulator noise independent of model quality. The meaningful criterion is **lift over the structural baseline** (Δρ = ρ(model) − ρ(Topo-BL)), not the absolute level.

| Δρ vs. Topo-BL | Interpretation |
|---------|---------------|
| ≥ +0.15 | Substantial lift — heterogeneous/learned model adds clear value |
| +0.05 to +0.15 | Meaningful lift — model outperforms structural baseline |
| −0.05 to +0.05 | No clear improvement over structural baseline |
| < −0.05 | Regression — model underperforms structural baseline |

> **Why two regimes?** Absolute thresholds (0.80, 0.75) were calibrated when validation targets shared structural basis with predictors (ρ ≈ 0.94 against reachability proxies). Against honest Sim labels — where the target is produced by a stochastic forward simulation fully decoupled from the predictor graph — the same absolute values are unattainable regardless of model quality. Applying Regime A thresholds to Regime B results condemns results for the wrong reason. Conversely, applying Regime B (relative) bands to RMAV pipeline results masks absolute weakness. Use the regime that matches the ground-truth source.

**Kendall τ** is the conservative cross-check:

```
τ = (C − D) / √((C + D + T_Q)(C + D + T_I))
```

A large |ρ − τ| gap (> 0.15) indicates that agreement is driven by a few extreme outliers. Inspect
the top 2–3 CRITICAL components in that case.

### 5.2 Bootstrap Confidence Interval

Non-parametric bootstrap CI for Spearman ρ (B = 2000 resamples, seed 42):

```
for b in 1..B:
    idx  = sample with replacement from [0..n-1]
    ρ_b  = spearmanr(Q[idx], I[idx])

CI_95 = [percentile(ρ_b, 2.5),  percentile(ρ_b, 97.5)]
```

A CI that does not cross the gate threshold provides stronger evidence than a point estimate alone.
When variance is zero (constant arrays), the CI degenerates to `[0, 0]` and a warning is emitted.

### 5.3 Classification Metrics — Precision, Recall, F1 @ K

`K` defaults to **20% of total node count** (minimum 3, maximum n). Override with `--top-k`.

```
gt_top_k  = top K nodes by I(v)   (ground truth critical set)
pred_top_k = top K nodes by Q(v)  (predicted critical set)

TP = |gt_top_k ∩ pred_top_k|
FP = |pred_top_k − gt_top_k|
FN = |gt_top_k  − pred_top_k|

Precision@K = TP / (TP + FP)
Recall@K    = TP / (TP + FN)
F1@K        = 2 × Precision × Recall / (Precision + Recall)
```

### 5.4 SPOF-F1

Measures the quality of articulation-point detection as an availability indicator.

```
SPOF-actual   = {v : is_articulation_point(v)  AND  I(v) > 0.3}
SPOF-predicted = {v : is_articulation_point(v)}

SPOF-F1 = harmonic mean of SPOF-precision and SPOF-recall
```

A low SPOF-F1 with a high overall ρ means the *global* ordering is correct but the binary SPOF
classification threshold is misaligned with the simulation threshold (0.3). See
[Interpreting Results](#12-interpreting-results).

### 5.5 Multi-Dimensional Validation Framework

Instead of comparing all dimensions against a single global cascade score, the validation pipeline correlates each predictor against a dimension-specific ground truth derived from simulation metrics:

#### 1. Reliability Dimension Validation
- **Predictor**: $R(v)$
- **Ground Truth**: $IR(v)$ (Reliability Impact, representing the propagation potential of the node's failure).
- **Core Metrics**:
  - **Spearman correlation** $\rho(R(v), IR(v))$
  - **Cascade Capture Rate @ 5 (CCR@5)**: The fraction of the top 5 most reliability-critical nodes correctly captured by the top 5 $R(v)$ predictions.
  - **Cascade Magnitude Error (CME)**: Mean absolute difference between predicted reliability and actual reliability impact:
    $$CME = \frac{1}{|V|} \sum_{v \in V} |R(v) - IR(v)|$$

#### 2. Maintainability Dimension Validation *(internal consistency check)*
- **Predictor**: $M(v)$
- **Ground Truth**: $IM(v)$ (Maintainability Impact, measuring the structural coupling fragility). Both $M(v)$ and $IM(v)$ are derived from the DEPENDS_ON graph; this correlation measures structural alignment, not empirical independence.
- **Core Metrics**:
  - **Spearman correlation** $\rho(M(v), IM(v))$
  - **Coupling-Oriented Capture Rate @ 5 (COCR@5)**: The fraction of the top 5 most maintainability-critical nodes correctly captured by the top 5 $M(v)$ predictions.
  - **Weighted Kappa Coupling Tier Agreement ($\kappa_{CTA}$)**: Cohen's Weighted Kappa comparing predicted maintainability tiers against actual impact tiers.
  - **Bottleneck Precision (BP)**: Precision of bottleneck detection based on $BT(v)$ and $w_{out}(v)$ against actual maintainability impact.

#### 3. Availability Dimension Validation
- **Predictor**: $A(v)$
- **Ground Truth**: $IA(v)$ (Availability Impact, representing the structural graph partitioning effect).
- **Core Metrics**:
  - **Spearman correlation** $\rho(A(v), IA(v))$
  - **SPOF-F1**: Articulation point detection F1 score comparing structural articulation points against nodes with actual availability impact exceeding `0.30`.
  - **Directed Articulation Separation Agreement (DASA)**: Compares directional articulation point metrics (`ap_c_out`, `ap_c_in`) with actual directional simulation impacts (`ia_out`, `ia_in`).
  - **Redundancy Recovery Index (RRI)**: Assesses the relationship between the Bridge Ratio ($BR(v)$) and availability recovery.
  - **High-SLA Redundancy Recall (HSRR)**: Measures overlap between QoS-amplified SPOF predictions ($QSPOF$) and high-impact availability failures.

#### 4. Vulnerability Dimension Validation *(internal consistency check)*
- **Predictor**: $V(v)$
- **Ground Truth**: $IV(v)$ (Vulnerability Impact, representing strategic reach and propagation speed). Both $V(v)$ and $IV(v)$ are derived from the DEPENDS_ON graph; this correlation measures structural alignment, not empirical independence.
- **Core Metrics**:
  - **Spearman correlation** $\rho(V(v), IV(v))$
  - **Attack Hub Capture Rate @ 5 (AHCR@5)**: Capture rate for the top 5 most vulnerable nodes.
  - **False Top Rate (FTR)**: The fraction of predicted top-K vulnerable nodes that are false alarms (actual compromise reach is $< 10\%$).
  - **Attack Path Adherence Rate (APAR)**: Measures overlap of high-vulnerability predictions with the simulation's observed critical attack paths.
  - **Cross-Dimensional Contamination Check (CDCC)**: The Spearman correlation between $V(v)$ and $A(v)$. A high value indicates redundancy and path-coupling conflation.

#### 5. Composite Validation and Predictive Gain (PG)
- **Predictor**: $Q(v)$ (overall composite score)
- **Ground Truth**: $I^*(v)$ (Composite Ground Truth, defined as the equal-weighted sum of the four dimensional ground truths):
  $$I^*(v) = 0.25 \times IR(v) + 0.25 \times IM(v) + 0.25 \times IA(v) + 0.25 \times IV(v)$$
- **Predictive Gain (PG)**: Measures whether the composite score outperforms the best single-dimension correlation:
  $$PG = \rho(Q(v), I^*(v)) - \max(\rho(R, IR), \rho(M, IM), \rho(A, IA), \rho(V, IV))$$
  A target of $PG \ge 0.03$ proves that multi-dimensional integration adds genuine predictive value.

---

### 5.6 Wilcoxon Signed-Rank Test

Tests whether $Q(v)$ ranks nodes *better* than the degree centrality (or PageRank) baseline against ground truth $I(v)$:
```
diff_scores = |Q(v) − I(v)| − |DC(v) − I(v)|     for all v
```
A one-sided Wilcoxon signed-rank test is conducted (alternative is 'less', significance level $\alpha = 0.05$). Significance ($p < 0.05$) means the absolute errors of $Q(v)$ are statistically smaller than the baseline. Requires at least 10 nodes; otherwise it defaults to $p = 1.0$.

---

### 5.7 Unified Validation Gates (G1-G9)

The validation service implements a unified 9-gate checklist across three tiers to determine system validation success:

| Gate ID | Metric | Type | Default Threshold | Description |
|---|---|---|:---:|---|
| **Tier 1** | **Primary Gates** | | | *(All must pass for overall validation success)* |
| **G1** | Spearman $\rho$ | Correlation | $\ge 0.70$ | Global rank correlation of Application-type nodes |
| **G2** | F1 @ K | Classification | $\ge 0.75$ | F1-score of the top-K critical set classification |
| **G3** | Precision @ K | Classification | $\ge 0.80$ | Precision of the top-K critical set classification |
| **G4** | Top-5 Overlap | Ranking | $\ge 0.60$ | Overlap of the top 5 predicted vs actual critical nodes |
| **Tier 2** | **Secondary Gates** | | | |
| **G5** | Predictive Gain (PG) | Gain | $> 0.03$ | Lift of composite $\rho$ over single-dimension correlations |
| **G6** | $\kappa_{CTA}$ | Classification | $\ge 0.70$ | Weighted Kappa Coupling Tier Agreement |
| **G7** | $CDCC$ | Correlation | $< 0.30$ | Cross-Dimensional Contamination Check |
| **Tier 3** | **Specialist Gates** | | | |
| **G8** | Bottleneck Precision | Specialist | $\ge 0.70$ | Precision of maintainability bottleneck detection |
| **G9** | False Top Rate (FTR) | Specialist | $\le 0.20$ | FTR of vulnerability exposure |

---

### 5.8 System Health Metrics

The validation service aggregates component-level predictions to calculate system-wide health and risk indicators (all metrics are weighted by the component QoS weights $w(v)$):

1. **Dimensional Health ($H_R, H_M, H_A, H_V \in [0, 1]$)**:
   Measures the system health in each quality dimension, where $1.0$ is perfect and lower scores represent degradation:
   $$H_d = 1.0 - \frac{\sum_{v \in V} \text{score}_d(v) \times w(v)}{\sum_{v \in V} w(v)}$$
2. **System Risk Index (SRI)**:
   A composite risk index reflecting the overall structural vulnerability and instability:
   $$SRI = 0.25 \times (1 - H_R) + 0.25 \times (1 - H_M) + 0.25 \times (1 - H_A) + 0.25 \times (1 - H_V)$$
3. **Risk Concentration Index (RCI)**:
   Computes the Gini coefficient of the composite $Q(v)$ scores to measure whether risk is concentrated in a few components or evenly distributed:
   $$RCI = \frac{\sum_{i=1}^{n} (2i - n - 1) \times Q_{(i)}}{n \sum_{i=1}^{n} Q_{(i)}}$$
   where $Q_{(i)}$ is the sorted overall quality score vector.

---

## 6. Node-Type Stratified Reporting

### Node-Type Stratification

Spearman ρ and F1@K are computed independently for each node type:

```
Application   →  primary validation layer
Broker        →  secondary (broker-layer analysis)
Topic         →  expected zero-variance signal (cascade simulation does not
                 propagate *from* topics); reported with a note
InfraNode     →  infrastructure layer; smaller population
Library       →  library coupling layer; fewer nodes → noisier ρ
```

Strata with fewer than 4 nodes report `"too few nodes for ρ"`.  
Strata with constant I(v) (std < 1e-9) report `"constant signal (not a primary failure type)"`.

**Typical stratification output:**
```
  Application       n=  26  ρ= 0.8320  F1=0.7143
  Broker            n=   5  constant signal (not a primary failure type)
  Topic             n=  27  constant signal (not a primary failure type)
  InfraNode         n=   8  constant signal (not a primary failure type)
  Library           n=   8  constant signal (not a primary failure type)
```

Topics and Brokers are expected to show constant signal: the cascade simulation triggers from a
*source* node's failure, not from a topic. Topic-layer reliability is captured through pub-sub
orphaning (Phase B), but the score accrues to the publisher application, not the topic node itself.

### Topic Frequency-Decile Stratification (Simpson's Paradox Mitigation)

Because distributed systems exhibit highly skewed topic messaging rates (spanning several orders of magnitude), aggregate validation metrics can be subject to **Simpson's paradox**—where global correlations mask strong or weak associations within specific frequency bands.

To address this, the validation pipeline automatically performs **frequency-decile stratified reporting** for all `Topic` nodes:
1. **Topic Binning**: All `Topic` components in the graph are sorted by their raw messaging frequency (`frequency` or `topic_frequency` in Hz) and partitioned into ten deciles.
2. **Spearman ρ per Decile**: For each frequency decile containing $\ge 3$ topics, the Spearman correlation $\rho$ and its corresponding $p$-value are calculated between predictions and simulation-derived ground truth.
3. **Frequency Range Tracking**: The concrete frequency bounds of each decile are reported (e.g., `(10.0, 50.0) Hz`), enabling structural engineers to verify which communication bandwidths have the strongest correlation.

---

## 7. Topology-Class Gate System

Gates are **adaptive** — a sparse 12-node system faces less stringent thresholds than a dense 80-node
hub-spoke architecture.

### Topology Classification

```python
density   = edges / (nodes × (nodes − 1))
hub_ratio = max_degree / mean_degree

"hub_spoke" if hub_ratio > 10  and density < 0.10
"sparse"    if density < 0.05
"dense"     if density > 0.20
"medium"    otherwise
```

### Gate Thresholds

| Class | ρ ≥ | F1 ≥ | SPOF-F1 ≥ | FTR ≤ | PG ≥ |
|-------|:---:|:----:|:---------:|:-----:|:----:|
| `sparse` | 0.75 | 0.65 | 0.60 | 0.30 | 0.02 |
| `medium` | 0.80 | 0.70 | 0.65 | 0.25 | 0.03 |
| `dense` | 0.82 | 0.72 | 0.65 | 0.25 | 0.03 |
| `hub_spoke` | 0.85 | 0.75 | 0.70 | 0.20 | 0.03 |

All five gates must pass for `overall_pass = True`. The exit code is 0 on PASS and 1 on FAIL,
enabling use in CI pipelines.

---

## 8. Multi-Seed Stability Sweep

A single seed run is not sufficient evidence. The `sweep` and `report` subcommands run the full
pipeline across multiple seeds to measure **stability**.

**Default seeds:** `42, 123, 456, 789, 2024`

**Aggregate metrics:**

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| `rho_mean` | mean(ρ across seeds) | Average predictive power |
| `rho_std` | std(ρ across seeds) | Stability; target σ ≤ 0.05 |
| `rho_min / rho_max` | range | Worst and best seed |
| `f1_mean` | mean(F1@K) | Average classification quality |
| `pg_mean` | mean(PG) | Average predictive gain |
| `rcr` | 1 − mean(normalised Kendall distance between seed pairs) | Rank Consistency Rate |
| `all_gates_pass_rate` | fraction of seeds that pass all gates | Reliability of PASS verdict |

**RCR — Rank Consistency Rate:**

```
For each pair of seeds (i, j):
    compute normalised Kendall distance d_{ij} = (1 − τ_{ij}) / 2  ∈ [0, 1]

RCR = 1 − mean(d_{ij})
```

RCR = 1.0 means identical rankings across all seeds. Target: RCR ≥ 0.90 for a stable methodology.

---

## 9. Ablation Study: Topology-Only vs. QoS-Enriched

The `compare` subcommand runs two full sweeps back-to-back — one without QoS (topology-only
baseline) and one with QoS enrichment — and reports the pair-wise deltas.

**Primary claim:** Δρ = ρ(Q_QoS, I) − ρ(Q_topo, I) > 0, *p* < 0.05

This is the evidence that incorporating QoS contract topology (singleton publisher detection, weight
amplification) adds predictive signal beyond purely structural graph metrics.

**Statistical test:** Paired t-test on seed-level ρ series (`alternative='greater'`).  
For fewer than 3 seeds, significance is approximated as `Δρ > 0.01`.

**LaTeX export:**

The `--latex` flag generates a ready-to-paste IEEE booktabs table (`ablation_table.tex`):

```latex
\begin{table}[t]
\centering
\caption{Ablation Study: Topology-Only vs.\ QoS-Enriched Prediction}
\label{tab:ablation}
\begin{tabular}{@{}lSSS@{}}
\toprule
Metric & {Topo-Only} & {QoS-Enr.} & {$\Delta$} \\
\midrule
Spearman $\rho$ (mean) & 0.8123 & 0.8467 & +0.0344 \\
Spearman $\rho$ (std)  & 0.0231 & 0.0198 & -0.0033 \\
F1 @ $K$               & 0.7200 & 0.7600 & +0.0400 \\
Predictive Gain (PG)   & 0.0410 & 0.0620 & +0.0210 \\
RCR                    & 0.9340 & 0.9410 & +0.0070 \\
\midrule
\multicolumn{4}{l}{\small QoS $\rho$-lift: $p < \alpha$, significant} \\
\bottomrule
\end{tabular}
\end{table}
```

The `compare` subcommand exits with code 0 only when Δρ > 0 *and* the test is significant — suitable
for CI gates guarding research claims.

---

## 10. CLI Reference

### Subcommands

| Subcommand | Description |
|-----------|-------------|
| `single` | One-seed run (uses the first seed in the list) |
| `sweep` | Multi-seed stability sweep |
| `report` | Full sweep + per-seed detail + gate JSON output |
| `compare` | Ablation study: topology-only vs. QoS-enriched |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input PATH` | *(required)* | Path to `system.json` or dataset JSON |
| `--qos` | off | Enable QoS-weighted RMAV scoring (adds PSPOF) |
| `--top-k INT` | 20% of nodes | K for classification and specialist metrics |
| `--seeds INTS` | `42,123,456,789,2024` | Comma-separated seed list |
| `--cascade INT` | `5` | Cascade simulation depth limit |
| `--bootstrap INT` | `2000` | Bootstrap resamples for CI |
| `--alpha FLOAT` | `0.05` | Significance level for Wilcoxon test |
| `--output PATH` | *(none)* | Write JSON report to path |
| `--csv` | off | Also write per-node CSV table |
| `--latex` | off | Write LaTeX ablation table (requires `compare` subcommand) |
| `--verbose` | off | Print per-node Q(v)/I(v) scores ranked by Q |
| `--no-color` | off | Disable ANSI colours in console output |

### Examples

```bash
# ── Quick sanity check (topology-only, single seed) ──────────────────────────
PYTHONPATH=. python cli/validate_graph.py single --input data/scenarios/atm_system.json

# ── QoS-enriched single run ───────────────────────────────────────────────────
PYTHONPATH=. python cli/validate_graph.py single --input data/scenarios/atm_system.json --qos --verbose

# ── Multi-seed stability sweep ────────────────────────────────────────────────
PYTHONPATH=. python cli/validate_graph.py sweep --input data/scenarios/atm_system.json --qos

# ── Full report with JSON output ──────────────────────────────────────────────
PYTHONPATH=. python cli/validate_graph.py report \
    --input data/scenarios/atm_system.json \
    --output output/atm_validation.json \
    --qos

# ── Ablation study + LaTeX table ──────────────────────────────────────────────
PYTHONPATH=. python cli/validate_graph.py compare \
    --input data/scenarios/atm_system.json \
    --output output/ablation.json \
    --seeds 42,123,456,789,2024 \
    --latex

# ── Custom K, deeper cascade, finer CI ────────────────────────────────────────
PYTHONPATH=. python cli/validate_graph.py report \
    --input data/scenarios/atm_system.json \
    --top-k 10 --cascade 10 --bootstrap 5000 \
    --qos --output output/fine_report.json

# ── CI-friendly exit codes ────────────────────────────────────────────────────
# Exit 0 = PASS;  Exit 1 = FAIL
PYTHONPATH=. python cli/validate_graph.py single --input data/scenarios/atm_system.json --qos \
    && echo "Validation passed"
```

---

## 11. Output Schema

### Single / Report JSON

```json
{
  "topology_class": "sparse",
  "validation": {
    "seed": 42,
    "qos_enabled": true,
    "n_nodes": 74,
    "n_app_nodes": 26,

    "spearman_rho":    0.8320,
    "spearman_p":      0.0009,
    "kendall_tau":     0.6101,
    "kendall_p":       0.0023,
    "bootstrap_ci_lo": 0.6841,
    "bootstrap_ci_hi": 0.9241,

    "top_k":           14,
    "precision_at_k":  0.7143,
    "recall_at_k":     0.7143,
    "f1_at_k":         0.7143,
    "spof_f1":         0.6250,
    "ftr":             0.2857,

    "icr_at_k":        0.7857,
    "bce":             0.2432,
    "pg":              0.0512,

    "wilcoxon_stat":   164.00,
    "wilcoxon_p":      0.0312,
    "wilcoxon_significant": true,

    "strata": {
      "Application": { "n": 26, "spearman_rho": 0.8320, "spearman_p": 0.0009, "f1_at_k": 0.7143, "k_used": 5 },
      "Broker":      { "n": 5,  "note": "constant signal (not a primary failure type)" },
      "Topic":       { "n": 27, "note": "constant signal (not a primary failure type)" }
    },

    "gates_passed": {
      "rho >= 0.75":      true,
      "f1 >= 0.65":       true,
      "spof_f1 >= 0.60":  true,
      "ftr <= 0.30":      true,
      "pg >= 0.02":       true
    },
    "overall_pass": true
  }
}
```

### Sweep / Report JSON

```json
{
  "sweep": {
    "qos_enabled": true,
    "seeds": [42, 123, 456, 789, 2024],
    "rho_mean":  0.8217,
    "rho_std":   0.0341,
    "rho_min":   0.7810,
    "rho_max":   0.8640,
    "f1_mean":   0.6974,
    "pg_mean":   0.0438,
    "rcr":       0.9231,
    "all_gates_pass_rate": 0.80,
    "per_seed":  [ ... ]
  },
  "topology_class": "sparse",
  "gate_thresholds": [0.75, 0.65, 0.60, 0.30, 0.02]
}
```

### Pipeline Result JSON (ValidationService)

This is the schema produced by the core `ValidationService` (`validate_layers()`):

```json
{
  "timestamp": "2026-06-06T21:40:00",
  "all_passed": true,
  "total_components": 14,
  "layers_passed": 1,
  "targets": {
    "spearman": 0.70,
    "f1_score": 0.75,
    "precision": 0.80,
    "top_5_overlap": 0.60,
    "predictive_gain": 0.03,
    "weighted_kappa_cta": 0.70,
    "cdcc_max": 0.30,
    "bottleneck_precision_target": 0.70,
    "ftr_max": 0.20
  },
  "layers": {
    "app": {
      "layer": "app",
      "layer_name": "Application Layer",
      "passed": true,
      "summary": {
        "spearman": 0.8320,
        "f1_score": 0.7143,
        "top_5_overlap": 0.6000,
        "rmse": 0.2432,
        "reliability_spearman": 0.8120,
        "maintainability_spearman": 0.7320,
        "availability_spearman": 0.8420,
        "vulnerability_spearman": 0.7120,
        "composite_spearman": 0.8520,
        "predictive_gain": 0.0400,
        "system_health": {
          "H_R": 0.8500,
          "H_M": 0.7800,
          "H_A": 0.9200,
          "H_V": 0.8100,
          "SRI": 0.1600,
          "RCI": 0.1200
        }
      },
      "gates": {
        "G1_spearman": true,
        "G2_f1": true,
        "G3_precision": true,
        "G4_top5": true,
        "G5_predictive_gain": true,
        "G6_kappa_cta": true,
        "G7_cdcc": true,
        "G8_bottleneck_precision": true,
        "G9_ftr": true
      },
      "node_type_stratified": {
        "Application": { "n": 26, "spearman": 0.8320, "target_rho": 0.75, "passed": true },
        "Broker":      { "n": 5,  "spearman": 0.0000, "target_rho": 0.70, "passed": false }
      },
      "frequency_decile_stratified": {
        "Decile 1": { "n": 3, "frequency_range": [0.1, 1.0], "spearman": 0.75, "p_value": 0.05 }
      },
      "warnings": []
    }
  },
  "warnings": []
}
```

### CSV Output (`--csv` flag)

When `--csv` is specified, a file `<output>_nodes.csv` is written with one row per node, ranked
by Q(v) descending:

```
rank, node_id, node_type, Q, R, M, A, V, I, cascade_depth, nodes_affected, is_articulation_point, degree_centrality
1, ConflictDetector, Application, 0.8421, 0.7200, 0.6500, 0.9100, 0.4200, 0.8102, 4, 18, True, 0.0321
...
```

---

## 12. Interpreting Results

### ρ is high (≥ gate) but F1@K fails

The global rank ordering is correct, but the binary threshold for "critical" is misaligned.
Check the distribution of Q(v) and I(v):

- If IQR(I) is very small (most nodes have similar impact), the top-K boundary is unstable.
- Consider increasing `--top-k` to use a larger critical set, or reviewing whether the gate
  threshold is appropriate for this system's size.

### ρ is negative

This is the **Inverse Criticality** effect, observed in mission-critical systems where the most
central nodes are the most heavily hardened. High PageRank components have redundant publishers and
backup routes, so their failure impact is *lower* than less-connected components.

Possible causes:
- The system was explicitly designed for resilience (ATM, financial HFT) — high-centrality nodes
  are protected by design intent.
- The cascade simulation depth limit (`--cascade`) is too shallow to propagate through well-connected
  hubs; increase to `--cascade 10` or `--cascade 15`.
- The `--qos` flag is off: PSPOF is zero, so sole-publisher nodes are not penalised.

### PG ≤ 0 (RMAV no better than degree centrality)

RMAV is not adding value beyond the naïve baseline:
- If the graph has very few Application nodes (< 10), the statistical test has high variance.
- Check whether `ap_c_directed` and `mpci` are being populated in `StructuralMetrics` (they default
  to 0 if the upstream analyzer did not run Step 2 first).
- Run `--verbose` to inspect per-node Q(v) vs I(v) to identify systematic mispredictions.

### Wilcoxon not significant with high ρ

This is expected when n < 10. The Wilcoxon test requires at least 10 nodes for reliable results. On
small graphs, treat ρ as the primary evidence and Wilcoxon as inconclusive.

### Strata show "constant signal" for Topics and Brokers

This is the expected and correct behaviour. Topics and Brokers do not generate cascade failures as
*origin* nodes in the `FaultInjector` simulation — their impact accrues to connected Application
nodes via Phase B orphaning. Constant I(v) = 0 for these types does not indicate a bug.

### Large |ρ − τ| gap (> 0.15)

Agreement is driven by a small number of extreme outliers. Inspect the top 2–3 CRITICAL components:
one may be a global hub with disproportionate cascade reach. This is not necessarily a failure — it
often reflects correct detection of a "God Component" in the architecture.

---

## 13. What Comes Next

The validation report produced by Step 5 is the empirical evidence base for the central thesis
claim. Step 6 (Visualization) renders these results into an interactive HTML dashboard:

- **Q(v) vs. I(v) scatter plot** — with quadrant highlighting (TP, TN, FP, FN).
- **Delta heatmap** — topology graph coloured by |Q(v) − I(v)|; high-delta nodes are "architectural
  surprises" where structural intent diverges from simulated behaviour.
- **RMAV radar charts** — per-node multi-dimensional breakdown for critical components.
- **Ranked component table** — with Q, R, M, A, V, I columns and pass/fail gate badge.

For research-grade LaTeX artifacts, run `compare --latex` to generate the ablation table directly
from the pipeline results:

```bash
PYTHONPATH=. python cli/validate_graph.py compare \
    --input data/scenarios/atm_system.json \
    --seeds 42,123,456,789,2024 \
    --output output/ablation_final.json \
    --latex
```

This writes `output/ablation_final_table.tex` — a booktabs-formatted table ready for IEEE/ACM
double-column layout.

---

← [Step 4: Simulate](failure-simulation.md) | → [Step 6: Visualize](visualization.md)