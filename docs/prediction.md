# Step 3: Prediction

**Produce criticality predictions for every component from topology alone — before any runtime data or simulation is available.**

← [Step 2: Analysis](structural-analysis.md) | → [Step 4: Simulation](failure-simulation.md)

---

## Table of Contents

1. [What This Step Does](#what-this-step-does)
2. [Two Prediction Paths](#two-prediction-paths)
3. [Rule-Based Prediction: RMAV](#rule-based-prediction-rmav)
   - [The Four Quality Dimensions](#the-four-quality-dimensions)
   - [RMAV Formulas](#rmav-formulas)
     - [Reliability R(v) — Fault Propagation Risk](#reliability-rv--fault-propagation-risk)
     - [Maintainability M(v) — Coupling Complexity](#maintainability-mv--coupling-complexity)
     - [Availability A(v) — SPOF Risk](#availability-av--spof-risk)
     - [Vulnerability V(v) — Security Exposure](#vulnerability-vv--security-exposure)
     - [Composite Score Q(v)](#composite-score-qv)
   - [Metric Orthogonality](#metric-orthogonality)
   - [AHP Weight Derivation](#ahp-weight-derivation)
     - [The AHP Procedure](#the-ahp-procedure)
     - [Default Pairwise Matrices](#default-pairwise-matrices)
     - [Weight Shrinkage Strategy](#weight-shrinkage-strategy)
   - [Criticality Classification](#criticality-classification)
     - [Box-Plot Thresholds (Normal Path)](#box-plot-thresholds-normal-path)
     - [Small-Sample Percentile Fallback](#small-sample-percentile-fallback)
   - [Interpretation Patterns](#interpretation-patterns)
4. [Learning-Based Prediction: GNN](#learning-based-prediction-gnn)
   - [Motivation](#motivation)
   - [Architecture Overview](#architecture-overview)
   - [Node Feature Construction](#node-feature-construction)
   - [Heterogeneous Graph Attention Network](#heterogeneous-graph-attention-network)
   - [Multi-Task RMAV Prediction Heads](#multi-task-rmav-prediction-heads)
   - [Edge Criticality Prediction](#edge-criticality-prediction)
   - [Ensemble: GNN + RMAV](#ensemble-gnn--rmav)
   - [Training Protocol](#training-protocol)
5. [Comparing the Two Paths](#comparing-the-two-paths)
6. [Worked Example](#worked-example)
7. [Output Schema](#output-schema)
8. [Commands](#commands)
9. [What Comes Next](#what-comes-next)

---

## What This Step Does

Prediction takes the 13-element metric vector **M(v)** produced by Step 2 for every component and produces a criticality prediction Q(v) ∈ [0, 1] along with a five-level classification — without consulting any runtime data.

```
M(v) from Step 2                  Prediction Engine                     Output
─────────────────                 ─────────────────                ──────────────────────────
PR, RPR, BT, CL, EV,       →    Rule-Based (RMAV)           →    R(v), M(v), A(v), V(v)
DG_in, DG_out, CC,               or                               Q_RMAV(v)
AP_c, BR, w, w_in, w_out         Learning-Based (GNN)             Q_GNN(v)
                                  or                               Q_ens(v)  [Ensemble]
                                  Both (Ensemble)                  Level: CRITICAL / HIGH /
                                                                           MEDIUM / LOW / MINIMAL
```

This step completes the **predictive half** of the methodology. Q(v) scores are topology-derived predictions. Step 4 produces ground-truth impact scores I(v) through failure simulation, and Step 5 measures how accurately Q(v) predicts I(v). The rule-based RMAV predictor achieves Spearman ρ = 0.876 and F1-score > 0.90 across validated system scales.

---

## Two Prediction Paths

The pipeline provides two complementary approaches to criticality prediction, which can be used independently or combined via an ensemble:

| | Rule-Based (RMAV) | Learning-Based (GNN) |
|---|---|---|
| **Mechanism** | AHP-weighted linear combination of topological metrics | Heterogeneous Graph Attention Network |
| **Interpretability** | Full — every score decomposes into metric contributions | Partial — attention weights and RMAV heads aid explanation |
| **Data requirement** | None — works on any graph immediately | Requires simulation-labelled training data |
| **New capabilities** | Node-level RMAV scores | Edge criticality scoring; cross-domain transfer |
| **Validation target** | ρ ≥ 0.80, F1 ≥ 0.90 | ρ ≥ 0.70, F1 ≥ 0.90 |
| **When to use** | First analysis; interpretable decision support | After training data exists; system-of-systems validation |

Both paths use the same five-level classification (CRITICAL / HIGH / MEDIUM / LOW / MINIMAL) and are validated against the same I(v) ground truth from Step 4, making their predictions directly comparable.

---

## Rule-Based Prediction: RMAV

### The Four Quality Dimensions

Each RMAV dimension answers a distinct operational question about a component's structural role:

| Dimension | Question | High Score Means | Stakeholder |
|-----------|----------|-----------------|-------------|
| **R — Reliability** | What is the blast radius if this fails? | Failure propagates widely through dependents | Reliability Engineer |
| **M — Maintainability** | How hard is this to change safely? | Tightly coupled, structural bottleneck | Software Architect |
| **A — Availability** | Is this a single point of failure? | Removing it would partition the dependency graph | DevOps / SRE |
| **V — Vulnerability** | Is this an attractive attack target? | Central, reachable, high-value hub | Security Engineer |

The four dimensions are deliberately orthogonal — they capture distinct failure modes. A component can simultaneously score high on all four (a critical hub) or high on only one (a pure SPOF, or a pure bottleneck). The RMAV breakdown tells you *why* a component is critical, guiding targeted remediation strategies.

---

### RMAV Formulas

Each dimension is a weighted linear combination of specific metrics from M(v). All inputs are normalized to [0, 1] by Step 2. All RMAV dimension scores are therefore in [0, 1].

The default weights shown below are derived from the Analytic Hierarchy Process (AHP) with shrinkage factor λ = 0.7. See [AHP Weight Derivation](#ahp-weight-derivation) for the derivation.

#### Reliability R(v) — Fault Propagation Risk

```
R(v) = 0.45 × RPR(v) + 0.30 × DG_in(v) + 0.25 × CDPot(v)
```

| Term | Weight | Rationale |
|------|--------|-----------|
| RPR(v) | 0.45 | Reverse PageRank — *global* cascade reach; how broadly v's failure propagates in the reverse-dependency direction |
| DG_in(v) | 0.30 | In-degree — count of direct dependents; captures *immediate* structural blast radius |
| CDPot(v) | 0.25 | Cascade Depth Potential — `((RPR + DG_in) / 2) × (1 − min(DG_out/DG_in, 1))`. Absorber nodes (many dependents, few outgoing links) score high; fan-out hubs score low |

A component with high R(v) is one whose failure propagates broadly **and deeply** through the dependency graph.

#### Maintainability M(v) — Coupling Complexity

```
M(v) = 0.40 × BT(v) + 0.35 × w_out(v) + 0.15 × CouplingRisk(v) + 0.10 × (1 − CC(v))
```

| Term | Weight | Rationale |
|------|--------|-----------|
| BT(v) | 0.40 | Betweenness — fraction of shortest dependency paths that pass through v; the defining bottleneck signal |
| w_out(v) | 0.35 | QoS-weighted out-degree — efferent coupling weighted by SLA priority; high-priority outgoing dependencies amplify change risk |
| CouplingRisk(v) | 0.15 | Instability-based: `1 − |DG_out − DG_in| / (DG_out + DG_in + ε)`. Peaks when afferent and efferent coupling are balanced — the most fragile coupling regime |
| 1 − CC(v) | 0.10 | Inverse clustering coefficient — low local redundancy means each of v's connections is a unique coupling path |

#### Availability A(v) — SPOF Risk

```
A(v) = 0.45 × QSPOF(v) + 0.30 × BR(v) + 0.15 × AP_c_directed(v) + 0.10 × CDI(v)
```

| Term | Weight | Rationale |
|------|--------|-----------|
| QSPOF(v) | 0.45 | QoS-weighted SPOF: `AP_c(v) × w(v)`. Structural SPOF severity weighted by the component's own QoS importance |
| BR(v) | 0.30 | Bridge Ratio — fraction of incident edges that are non-redundant bridges; losing any bridge edge disconnects a subgraph |
| AP_c_directed(v) | 0.15 | Directed articulation score — fraction of reachable pairs that lose connectivity when v is removed, computed on the directed graph |
| CDI(v) | 0.10 | Connectivity Disruption Index — increase in average shortest-path length upon v's removal |

#### Vulnerability V(v) — Security Exposure

```
V(v) = 0.40 × REV(v) + 0.35 × RCL(v) + 0.25 × w_in(v)
```

| Term | Weight | Rationale |
|------|--------|-----------|
| REV(v) | 0.40 | Reverse Eigenvector Centrality — computed on G^T; high score means v's downstream dependents are themselves highly connected, amplifying compromise reach |
| RCL(v) | 0.35 | Reverse Closeness Centrality — computed on G^T; measures how quickly a compromise at v can reach all downstream dependents |
| w_in(v) | 0.25 | QoS-weighted in-degree — direct dependents weighted by SLA priority; high-priority dependents make v a high-value target |

#### Composite Score Q(v)

```
Q(v) = w_R × R(v) + w_M × M(v) + w_A × A(v) + w_V × V(v)
```

**Default weights:** `w_R = w_M = w_A = w_V = 0.25` (equal weighting, suitable for general-purpose analysis).

**Domain-specific adjustments via `--use-ahp`:**

| System Type | Priority Order | Example Weights |
|-------------|---------------|-----------------|
| High-availability (medical, aerospace) | A > R > M > V | w_A=0.40, w_R=0.30, w_M=0.20, w_V=0.10 |
| Security-critical (financial, government) | V > A > R > M | w_V=0.40, w_A=0.30, w_R=0.20, w_M=0.10 |
| Actively developed (fast iteration) | M > R > A > V | w_M=0.40, w_R=0.30, w_A=0.20, w_V=0.10 |
| General-purpose | Equal | w_R=w_M=w_A=w_V=0.25 |

---

### Metric Orthogonality

Each raw metric contributes to **exactly one** RMAV dimension. This prevents a single structural property from accumulating disproportionate weight in Q(v).

| Metric | Symbol | R | M | A | V | Rationale |
|--------|--------|---|---|---|---|-----------|
| Reverse PageRank | RPR | ✓ | | | | Global cascade reach — primary R signal |
| In-Degree | DG_in | ✓ | | | | Direct dependents — immediate blast radius |
| Cascade Depth Potential | CDPot | ✓ | | | | Depth discrimination (derived from RPR + DG ratio) |
| Betweenness | BT | | ✓ | | | Structural bottleneck position |
| QoS-Weighted Out-Degree | w_out | | ✓ | | | Priority-weighted efferent coupling |
| Coupling Risk | CouplingRisk | | ✓ | | | Afferent/efferent imbalance (derived) |
| Clustering Coefficient | CC | | ✓ | | | Inverse: local redundancy proxy |
| QoS-Weighted SPOF | QSPOF | | | ✓ | | QoS-amplified structural SPOF |
| Bridge Ratio | BR | | | ✓ | | Non-redundant edge fraction |
| Directed AP Score | AP_c_directed | | | ✓ | | Directed reachability loss |
| Connectivity Disruption | CDI | | | ✓ | | Path elongation upon removal |
| Reverse Eigenvector | REV | | | | ✓ | Strategic exposure via downstream hubs |
| Reverse Closeness | RCL | | | | ✓ | Compromise propagation speed |
| QoS-Weighted In-Degree | w_in | | | | ✓ | Priority-weighted attack surface |

> **Note:** `DG_out` (raw out-degree) is not used directly in any dimension; it is replaced by `w_out` (QoS-weighted) and the derived `CouplingRisk`. Raw PageRank (PR) and raw in-degree (DG_in) are reported for reference but do not appear in dimension formulas beyond R(v)'s explicit DG_in term.

---

### AHP Weight Derivation

The default intra-dimension weights (0.45, 0.30, 0.25, etc.) are derived from the **Analytic Hierarchy Process (AHP)** — a structured method that translates expert judgment into numerical weights via pairwise comparisons (Saaty, 1980).

#### The AHP Procedure

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
         If CR > 0.10: the matrix is inconsistent — revise judgments and recompute
```

Random Index values (Saaty, 1980):

| n | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| RI | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 |

#### Default Pairwise Matrices

**Reliability** — criteria: [RPR, DG_in, CDPot]

```
           RPR    DG_in  CDPot
RPR     [ 1.00,  3.00,  5.00 ]   RPR dominates: global cascade reach is the primary signal
DG_in   [ 0.33,  1.00,  2.00 ]   DG_in moderately dominates CDPot
CDPot   [ 0.20,  0.50,  1.00 ]   CDPot is a secondary depth refinement

→ AHP: [0.648, 0.230, 0.122]   CR ≈ 0.003 (highly consistent)
  Final (λ=0.7): [0.45, 0.26, 0.19]  (after shrinkage, rounded for implementation)
```

**Maintainability** — criteria: [BT, w_out, CouplingRisk, CC_inv]

```
              BT    w_out   CR    CC_inv
BT        [ 1.00,  2.00,  3.00,  5.00 ]   Betweenness is the defining bottleneck signal
w_out     [ 0.50,  1.00,  2.00,  3.00 ]   Priority-weighted coupling amplifies change risk
CR        [ 0.33,  0.50,  1.00,  2.00 ]   Coupling imbalance adds structural fragility signal
CC_inv    [ 0.20,  0.33,  0.50,  1.00 ]   Low clustering: unique paths (supplementary signal)

→ AHP: [0.508, 0.271, 0.144, 0.077]   CR ≈ 0.006 (highly consistent)
  Final (λ=0.7): [0.40, 0.35, 0.15, 0.10]
```

**Availability** — criteria: [QSPOF, BR, AP_c_directed, CDI]

```
                QSPOF   BR    AP_c_d  CDI
QSPOF       [ 1.00,  2.00,  3.00,  5.00 ]   QoS-amplified SPOF is the primary availability signal
BR          [ 0.50,  1.00,  2.00,  3.00 ]   Non-redundant edges: structural brittleness
AP_c_dir    [ 0.33,  0.50,  1.00,  2.00 ]   Directed reachability loss (complements QSPOF)
CDI         [ 0.20,  0.33,  0.50,  1.00 ]   Path elongation: soft availability degradation

→ AHP: [0.508, 0.269, 0.145, 0.079]   CR ≈ 0.006 (highly consistent)
  Final (λ=0.7): [0.45, 0.30, 0.15, 0.10]
```

**Vulnerability** — criteria: [REV, RCL, w_in]

```
           REV    RCL   w_in
REV     [ 1.00,  1.00,  2.00 ]   REV and RCL are co-primary; structural symmetry justified
RCL     [ 1.00,  1.00,  2.00 ]   Same reasoning as REV
w_in    [ 0.50,  0.50,  1.00 ]   Direct attack surface (immediate, not propagated)

→ AHP: [0.400, 0.400, 0.200]   CR = 0.000 (perfectly consistent)
  Final (λ=0.7): [0.40, 0.35, 0.25]
```

#### Weight Shrinkage Strategy

Pure AHP weights are not used directly. A **formal shrinkage procedure** blends AHP weights with a uniform distribution (equal weights) via a mixing coefficient λ:

```
w_final = λ · w_AHP + (1 − λ) · w_uniform
```

Default **λ = 0.7**. This blending respects the AHP-derived priority hierarchy while remaining robust to the extreme dominance of single metrics, particularly in small comparison matrices. It provides a principled, reproducible alternative to ad-hoc smoothing.

---

### Criticality Classification

After computing Q(v) for all components in a layer, each component is classified into one of five criticality levels using **adaptive thresholds** derived from the actual distribution of Q(v) scores.

#### Box-Plot Thresholds (Normal Path)

Used when the layer has **≥ 12 components**.

```
Compute:  Q1, Median, Q3  from all Q(v) values
          IQR = Q3 − Q1
          upper_fence = Q3 + 1.5 × IQR

Classify:
  Q(v) > upper_fence  →  CRITICAL   (statistical outlier — significantly above 3rd quartile)
  Q(v) > Q3           →  HIGH       (above 75th percentile)
  Q(v) > Median       →  MEDIUM     (above 50th percentile)
  Q(v) > Q1           →  LOW        (above 25th percentile)
  Q(v) ≤ Q1           →  MINIMAL    (bottom 25th percentile)
```

**Why box-plot thresholds?** Static cutoffs (e.g., "Q(v) > 0.7 = CRITICAL") fail when score distributions vary across system types. Box-plot thresholds adapt to each system's actual distribution, identifying components that are structurally exceptional *relative to their peers* — the definition of "critical."

Typical distribution: CRITICAL ≈ 5–15%, HIGH ≈ 25%, MEDIUM ≈ 25%, LOW ≈ 25%, MINIMAL ≈ 25%.

#### Small-Sample Percentile Fallback

Used when the layer has **< 12 components** (too few for stable quartile estimation).

| Level | Percentile Threshold |
|-------|---------------------|
| **CRITICAL** | Top 10% (90th percentile and above) |
| **HIGH** | Top 25% (75th–90th percentile) |
| **MEDIUM** | Top 50% (50th–75th percentile) |
| **LOW** | Top 75% (25th–50th percentile) |
| **MINIMAL** | Bottom 25% (below 25th percentile) |

---

### Interpretation Patterns

The RMAV breakdown reveals not just *that* a component is critical, but *why* — and what remediation to apply:

| Pattern | R | M | A | V | What It Means | Primary Risk | Recommended Action |
|---------|---|---|---|---|--------------|-------------|-------------------|
| **Hub** | High | High | High | High | Critical integration point — central to all concerns | Catastrophic failure | Add redundancy + circuit breakers + monitoring |
| **Reliability Hub** | High | Low | Low | Low | Widely depended upon but not a bottleneck or SPOF | Cascade failure | Add retry logic, graceful degradation |
| **Bottleneck** | Low | High | Low | Low | Coupling problem — change here ripples everywhere | Change fragility | Refactor to reduce coupling |
| **SPOF** | Low | Low | High | Low | Structural single point of failure | Availability loss | Add redundant instance or failover path |
| **Target** | Low | Low | Low | High | Security-exposed hub | Compromise propagation | Harden, isolate, add access controls |
| **Maintenance Debt** | Med | High | Med | Low | Tightly coupled and in the critical path | Tech debt fragility | Prioritize refactoring in next sprint |
| **Leaf** | Low | Low | Low | Low | Peripheral — low concern | None | Standard practices |

---

## Learning-Based Prediction: GNN

### Motivation

The RMAV predictor combines 13 metrics via fixed AHP-derived weights. This has two inherent limitations:

**Fixed feature interactions.** RMAV cannot discover that, for a particular system topology, the interaction between betweenness and reverse eigenvector centrality is more predictive than either metric alone. The weights are determined before analysis begins.

**Node-only scoring.** RMAV scores nodes; edges are analysed only via structural proxies (bridge ratio, betweenness of endpoints). There is no direct prediction of how critical a *pub-sub relationship* is — which data flows are most dangerous to lose.

The Graph Neural Network predictor addresses both limitations: it learns which metric interactions actually predict failure impact from labelled training data, and introduces direct edge criticality scoring as a new capability.

The GNN is positioned as **Step 3.5** — an extension of Step 3, inserted between RMAV scoring and Failure Simulation — and validated against the same I(v) ground truth, enabling direct comparison with RMAV.

```
Step 3a: Rule-Based Prediction (RMAV)     Q_RMAV(v) — AHP-weighted formula
Step 3b: Learning-Based Prediction (GNN)  Q_GNN(v)  — trained on I(v) ground truth
Step 3c: Ensemble                         Q_ens(v)  = α·Q_GNN + (1−α)·Q_RMAV
```

### Architecture Overview

Three cooperating components form the GNN prediction engine:

```
                    NetworkX DiGraph
                          │
                ┌─────────▼──────────┐
                │   Data Preparation  │  networkx_to_hetero_data()
                │   HeteroData        │  node features (18-dim)
                │   node/edge splits  │  edge features (8-dim)
                │   labels I(v)       │  labels (5-dim RMAV)
                └─────────┬──────────┘
                          │
          ┌───────────────┼──────────────────────┐
          │               │                      │
┌─────────▼──────┐ ┌──────▼─────────┐  ┌────────▼────────┐
│ NodeCriticality│ │ EdgeCriticality │  │  EnsembleGNN    │
│ GNN            │ │ GNN            │  │                 │
│ HeteroGAT      │ │ (shares node   │  │ α·Q_GNN         │
│ 3 layers       │ │  backbone)     │  │ +(1−α)·Q_RMAV   │
│ 4 heads        │ │ Edge MLP head  │  │                 │
│ Output: (N,5)  │ │ Output: (E,5)  │  │ Output: (N,5)   │
└────────────────┘ └────────────────┘  └─────────────────┘
```

### Node Feature Construction

Each node v is represented by an **18-dimensional feature vector** derived from Steps 2 and 3a, ensuring the GNN starts with the same information as the RMAV scorer and learns on top of it.

**Topological metrics (indices 0–12):**

| Index | Metric | RMAV Role | Captures |
|-------|--------|-----------|----------|
| 0 | PageRank (PR) | Reliability | Transitive dependency importance |
| 1 | Reverse PageRank (RPR) | Reliability | Cascade propagation reach |
| 2 | Betweenness Centrality (BT) | Maintainability | Structural bottleneck position |
| 3 | Closeness Centrality (CL) | Vulnerability | Speed of fault propagation |
| 4 | Eigenvector Centrality (EV) | Vulnerability | Influence of high-value neighbours |
| 5 | In-Degree (DG_in) | Reliability | Direct dependent count |
| 6 | Out-Degree (DG_out) | Maintainability | Direct dependency count |
| 7 | Clustering Coefficient (CC) | Maintainability | Local redundancy |
| 8 | Continuous AP score (AP_c) | Availability | SPOF severity |
| 9 | Bridge Ratio (BR) | Availability | Fraction of non-redundant edges |
| 10 | QoS weight aggregate (w) | Availability | Overall QoS criticality |
| 11 | QoS weighted in-degree (w_in) | Vulnerability | Priority-weighted dependents |
| 12 | QoS weighted out-degree (w_out) | Maintainability | Priority-weighted dependencies |

**Node type one-hot (indices 13–17):**

| Index | Node Type |
|-------|-----------|
| 13 | Application |
| 14 | Broker |
| 15 | Topic |
| 16 | Node (infrastructure) |
| 17 | Library |

All metrics are already normalized to [0, 1] by Step 2's min-max normalization — no additional scaling is required.

### Heterogeneous Graph Attention Network

The pub-sub multi-layer graph contains five distinct node types and seven edge types. A homogeneous GNN would conflate semantically distinct relationships (e.g., `PUBLISHES_TO` vs. `RUNS_ON`). A **heterogeneous GNN** maintains separate weight matrices per relation type, preserving these distinctions.

The model uses a **3-layer, 4-head Heterogeneous Graph Attention Network (HeteroGAT)**:

```
Layer 0 — Input projection (type-specific):
  h_v^(0) = GELU( LayerNorm( W_{type(v)} · x_v + b_{type(v)} ) )

Layer k — Message passing per relation type r:
  α_{uv}^r = softmax( a_r^T · [h_u^(k) ‖ h_v^(k)] )  (attention weight)
  m_v^(r,k) = Σ_{u ∈ N_r(v)} α_{uv}^r · W_r^(k) · h_u^(k)

  Aggregate across relation types:
  h_v^(k+1) = GELU( LayerNorm( Σ_r W_{agg,r} · m_v^(r,k) + W_self · h_v^(k) ) )
```

With **hidden dimension D = 64**, dropout p = 0.2, and residual connections between layers.

### Multi-Task RMAV Prediction Heads

Rather than predicting a single composite score, the model uses **four dimension-specific MLP heads** plus one composite head, directly mirroring the RMAV decomposition:

```
R̂(v) = MLP_R( h_v )      — Reliability prediction head
M̂(v) = MLP_M( h_v )      — Maintainability prediction head
Â(v)  = MLP_A( h_v )      — Availability prediction head
V̂(v)  = MLP_V( h_v )      — Vulnerability prediction head

Î*(v) = MLP_C( h_v ‖ R̂(v) ‖ M̂(v) ‖ Â(v) ‖ V̂(v) )   — Composite head
```

The composite head receives the dimension predictions as additional inputs, allowing it to learn data-driven dimension weighting that complements the fixed AHP weighting in RMAV. All outputs pass through sigmoid activation, producing scores in [0, 1] consistent with Q*(v) and I*(v) value ranges.

**Training loss:**

```
L = L_composite + 0.5 · L_RMAV + 0.3 · L_rank

L_composite = (1/N) Σ_v (Î*(v) − I*(v))²                     (primary: match simulation)
L_RMAV      = (1/N) Σ_v Σ_{d∈{R,M,A,V}} (d̂(v) − I_d(v))²   (auxiliary: RMAV alignment)
L_rank      = −(1/N) Σ_v log P(v-th position)                 (ListMLE: optimise Spearman ρ)
```

### Edge Criticality Prediction

For each edge (u, v) with type `rel`:

```
score(u,v) = MLP_E( h_u ‖ h_v ‖ e_{uv} )
```

where `e_{uv}` is an 8-dimensional edge feature vector (QoS policy fields: reliability, durability, history depth, deadline, liveliness, priority, message size, and edge type one-hot). This produces five scores per edge (composite, R, M, A, V), enabling relationship-level RMAV analysis alongside component-level scoring — a capability not available in the rule-based path.

Edge labels for training:
```
I_edge(u,v) = max( I*(u), I*(v) )
```

### Ensemble: GNN + RMAV

The ensemble combines GNN predictions with RMAV scores via a **learnable convex combination**:

```
Q_ens(v) = α · Q_GNN(v) + (1 − α) · Q_RMAV(v)
```

`α ∈ (0, 1)` is a **learned scalar per RMAV dimension** (five scalars total), stored in logit space and initialised to 0.5. The ensemble is fine-tuned on training nodes with labelled simulation ground truth. Example learned weights after training:

```
Learned ensemble weights (α = GNN contribution):
  composite       : GNN  62.3% ████████████░░░░░░░░ RMAV  37.7%
  reliability     : GNN  58.1% ████████████░░░░░░░░ RMAV  41.9%
  maintainability : GNN  44.7% █████████░░░░░░░░░░░ RMAV  55.3%
  availability    : GNN  71.2% ██████████████░░░░░░ RMAV  28.8%
  vulnerability   : GNN  49.3% ██████████░░░░░░░░░░ RMAV  50.7%
```

An α > 0.5 for a dimension means the GNN learned a stronger signal than RMAV for that dimension. High α on Availability indicates the GNN captures SPOF patterns better than the handcrafted AP_c metric; high RMAV contribution on Maintainability indicates the AHP-weighted betweenness + coupling risk formula is hard to improve via learning alone.

### Training Protocol

**Transductive setting (single graph):** Nodes are randomly split 60/20/20 into train/val/test sets. Early stopping halts training when validation Spearman ρ does not improve for 30 consecutive epochs. Best model weights are restored before test evaluation.

**Inductive setting (multiple graphs):** When all eight domain scenarios are available, the trainer supports inductive multi-graph learning — train on a subset of system instances, evaluate generalisation on held-out instances. This is the recommended setting for strong transfer learning capability, configured via `--multi-scenario`.

**Optimiser:** AdamW, `lr = 3×10⁻⁴`, `weight_decay = 10⁻⁴`, cosine annealing schedule, gradient clipping `max_norm = 1.0`.

---

## Comparing the Two Paths

| Property | Rule-Based (RMAV) | Learning-Based (GNN) | Ensemble |
|---|---|---|---|
| Requires training data | No | Yes (I(v) labels from Step 4) | Yes |
| Node criticality | ✓ | ✓ | ✓ |
| Edge criticality | Proxies only | ✓ Direct | ✓ Direct |
| RMAV dimension breakdown | ✓ | ✓ | ✓ |
| Interpretability | Full (formula-derived) | Partial (attention weights) | Partial |
| Generalisation to unseen systems | Immediate | Requires fine-tuning | Requires fine-tuning |
| Typical Spearman ρ | 0.876 | 0.876+ (with sufficient training) | Best of both |
| Typical F1-score | > 0.90 | ≥ 0.90 | ≥ 0.90 |
| Primary use case | First analysis; interpretable decisions | Post-training; system-of-systems | Production deployment |

**Recommended workflow:**
1. Run RMAV first — immediate results, full interpretability, no training required.
2. Run Failure Simulation (Step 4) to generate I(v) ground truth.
3. Train GNN on the labelled data.
4. Use Ensemble for production prediction; compare GNN vs. RMAV per dimension to understand where learning adds value.

---

## Worked Example

Using **PLC_Controller (A3)** from the Distributed Intelligent Factory (DIF) scenario:

**Metric values for A3 (normalized):**

| Metric | Value | Used in |
|--------|-------|---------|
| RPR | 0.60 | R(v) |
| DG_in | 0.75 | R(v) |
| BT | 0.95 | M(v) |
| w_out | 0.68 | M(v) |
| AP_c | 0.43 | A(v) via QSPOF |
| BR | 1.00 | A(v) |
| REV | 0.80 | V(v) |
| RCL | 0.70 | V(v) |
| w_in | 0.75 | V(v) |
| w(v) | 0.72 | A(v) via QSPOF |

**CDPot computation for A3:**
```
CDPot = ((0.60 + 0.75) / 2) × (1 − min(0.80 / 0.75, 1))
      = 0.675 × 0.00   ← DG_out/DG_in > 1, capped at 1
      = 0.00
```
A3 is a fan-out hub — it has more outgoing than incoming dependencies, so cascade depth potential is minimal.

**RMAV computation:**
```
R(A3) = 0.45 × 0.60 + 0.30 × 0.75 + 0.25 × 0.00 = 0.270 + 0.225 + 0.000 = 0.495

CouplingRisk = 1 − |0.80 − 0.75| / (0.80 + 0.75 + ε) ≈ 0.968
M(A3) = 0.40 × 0.95 + 0.35 × 0.68 + 0.15 × 0.968 + 0.10 × (1−0.15)
       = 0.380 + 0.238 + 0.145 + 0.085 = 0.848

QSPOF = 0.43 × 0.72 = 0.310
A(A3) = 0.45 × 0.310 + 0.30 × 1.00 + 0.15 × 0.43 + 0.10 × 0.65
       = 0.140 + 0.300 + 0.065 + 0.065 = 0.570

V(A3) = 0.40 × 0.80 + 0.35 × 0.70 + 0.25 × 0.75
       = 0.320 + 0.245 + 0.188 = 0.753

Q(A3) = 0.25 × 0.495 + 0.25 × 0.848 + 0.25 × 0.570 + 0.25 × 0.753 = 0.667
```

**Interpretation:** A3 scores CRITICAL on M (tight coupling — high BT, high CouplingRisk) and HIGH on V (attractive target via downstream hub reach). The primary risk is change fragility and vulnerability exposure, not cascade blast radius. Remediation should focus on architectural decoupling before security hardening.

---

## Output Schema

```json
{
  "layer": "app",
  "prediction_method": "rmav",
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
    }
  },
  "gnn_scores": {
    "DataRouter": {
      "composite":       0.91,
      "reliability":     0.92,
      "maintainability": 0.84,
      "availability":    0.97,
      "vulnerability":   0.81
    }
  },
  "ensemble_scores": {
    "DataRouter": {
      "composite":  0.87,
      "alpha_used": [0.62, 0.58, 0.45, 0.71, 0.49]
    }
  }
}
```

---

## Commands

```bash
# ─── Rule-Based Prediction (RMAV) ────────────────────────────────────────────

# Predict with equal dimension weights (default)
python bin/analyze_graph.py --layer app

# Predict with AHP-derived weights (recommended for domain-specific analysis)
python bin/analyze_graph.py --layer system --use-ahp

# Export prediction scores to JSON
python bin/analyze_graph.py --layer system --output results/prediction.json

# Run weight sensitivity analysis (200 perturbations, σ=0.05)
python bin/analyze_graph.py --layer system --use-ahp --sensitivity

# Predict across all layers
for layer in app infra mw system; do
  python bin/analyze_graph.py --layer $layer --output results/prediction_$layer.json
done

# ─── Learning-Based Prediction (GNN) ─────────────────────────────────────────

# Train GNN on current graph (requires Step 4 simulation results)
python bin/train_graph.py --layer app

# Train with custom hyperparameters
python bin/train_graph.py --layer system \
    --hidden 128 --heads 8 --layers 4 \
    --epochs 500 --patience 50

# Train from pre-computed results (skip Neo4j)
python bin/train_graph.py \
    --structural results/metrics.json \
    --simulated  results/impact.json \
    --rmav       results/prediction.json \
    --checkpoint output/gnn_checkpoints/

# ─── GNN Inference ───────────────────────────────────────────────────────────

# Predict on current graph using trained checkpoint
python bin/predict_graph.py --layer app --checkpoint output/gnn_checkpoints/

# Side-by-side GNN vs. RMAV comparison with edge scores
python bin/predict_graph.py --layer system \
    --checkpoint output/gnn_checkpoints/ \
    --compare-rmav \
    --show-edges \
    --top-n 20

# Validate GNN predictions against simulation ground truth
python bin/predict_graph.py --layer app \
    --checkpoint output/gnn_checkpoints/ \
    --simulated results/impact.json
```

### Reading the Output

```
Layer: app | 35 components | AHP weights (λ=0.7)
Classification: box-plot  |  Q1=0.18  Median=0.31  Q3=0.52  Fence=0.73

CRITICAL (3 components):
  DataRouter      Q=0.84  R=0.88  M=0.75  A=0.92  V=0.81  [SPOF]
  SensorHub       Q=0.73  R=0.76  M=0.60  A=0.85  V=0.71  [SPOF]
  CommandBus      Q=0.73  R=0.71  M=0.79  A=0.62  V=0.74

HIGH (9 components):
  ...
```

- **[SPOF]** indicates AP_c > 0 — structural single point of failure. Always address SPOFs first regardless of overall Q(v).
- Inspect the RMAV breakdown for CRITICAL components to identify the dominant concern before choosing a remediation strategy.

---

## What Comes Next

At this point, every component has a predicted criticality score Q(v) ∈ [0, 1] and a five-level classification derived purely from topology. These are **predictions** — they have not yet been validated against actual failure behavior.

Step 4 generates **ground-truth impact scores I(v)** by simulating component failures exhaustively and measuring the cascade damage each causes. Step 5 then computes Spearman ρ between Q(v) and I(v) to quantify how accurately topology predicts real failure impact, closing the loop on the methodology's central empirical claim.

For the GNN path, Step 4 simulation results serve a dual purpose: they are the **training labels** for the GNN and simultaneously the **validation ground truth** for both RMAV and GNN predictions.

---

← [Step 2: Analysis](structural-analysis.md) | → [Step 4: Simulation](failure-simulation.md)