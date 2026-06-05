# Step 2–3: Analyze & Predict

**Produce criticality predictions for every component from topology alone — before any runtime data or simulation is available.**

← [Step 2: Analyze (structural metrics)](structural-analysis.md) | → [Step 4: Simulate](failure-simulation.md)

> **Stage separation.** This document covers two distinct pipeline stages that use the same input M(v) but serve different roles:
> - **Analyze (Step 2b)** — deterministic, closed-form RMAV/Q scoring. Rule-based model. Given the same graph, always produces the same Q(v). No training required.
> - **Predict (Step 3)** — inductive GNN forecasting that generalises beyond the closed form. Learns nonlinear interactions and multi-hop motifs that AHP-weighted scoring cannot encode. Requires simulation-labelled training data.

---

## Table of Contents

1. [What These Stages Do](#1-what-these-stages-do)
2. [Two Paths: Analyze vs Predict](#2-two-paths-analyze-vs-predict)
3. [Analyze Stage — Rule-Based RMAV Scoring](#3-analyze-stage--rule-based-rmav-scoring)
   - 3.1 [The Four Quality Dimensions](#31-the-four-quality-dimensions)
   - 3.2 [RMAV Formulas](#32-rmav-formulas)
     - [Reliability R(v)](#reliability-rv--fault-propagation-risk)
     - [Maintainability M(v)](#maintainability-mv--coupling-complexity)
     - [Availability A(v)](#availability-av--spof-risk)
     - [Vulnerability V(v)](#vulnerability-vv--security-exposure)
     - [Composite Score Q(v)](#composite-score-qv)
   - 3.3 [Derived Terms](#33-derived-terms)
   - 3.4 [Metric Orthogonality](#34-metric-orthogonality)
   - 3.5 [AHP Weight Derivation](#35-ahp-weight-derivation)
   - 3.6 [Weight Shrinkage Strategy](#36-weight-shrinkage-strategy)
   - 3.7 [Criticality Classification](#37-criticality-classification)
   - 3.8 [Interpretation Patterns](#38-interpretation-patterns)
4. [Predict Stage — Graph Neural Network](#4-predict-stage--graph-neural-network)
   - 4.1 [Motivation](#41-motivation)
   - 4.2 [Architecture Overview](#42-architecture-overview)
   - 4.3 [Graph Data Preparation](#43-graph-data-preparation)
   - 4.4 [HGT Model (Heterogeneous Graph Transformer)](#44-hgt-model-heterogeneous-graph-transformer)
   - 4.5 [Multi-Task Prediction Heads](#45-multi-task-prediction-heads)
   - 4.6 [Edge Criticality Prediction](#46-edge-criticality-prediction)
   - 4.7 [Ensemble: GNN + RMAV](#47-ensemble-gnn--rmav)
   - 4.8 [Training Protocol](#48-training-protocol)
   - 4.9 [Multi-Seed Stability](#49-multi-seed-stability)
   - 4.10 [Methodological Integrity](#410-methodological-integrity)
5. [Comparing the Two Stages](#5-comparing-the-two-stages)
6. [Worked Example](#6-worked-example)
7. [Output Schema](#7-output-schema)
8. [Commands](#8-commands)
9. [What Comes Next](#9-what-comes-next)

---

## 1. What These Stages Do

The Analyze and Predict stages both take the metric vector **M(v)** produced by the structural sub-phase of Step 2 and produce a criticality prediction Q(v) ∈ [0, 1] with a five-level classification.

```
M(v) from structural analysis        Scoring Engine                     Output
──────────────────────               ──────────────────────         ───────────────────────────
Tier 1 — 13 RMAV inputs:    →       Analyze (deterministic)  →     R(v), M(v), A(v), V(v)
  RPR, DG_in, MPCI, FOC              AHP-weighted formula            Q_RMAV(v) ∈ [0, 1]
  BT, w_out, CC, path_complexity     Anti-pattern detection
  AP_c_dir, BR, CDI               → Predict (inductive, opt) →     Q_GNN(v)
  REV, RCL, w_in                     HeteroGAT trained on I(v)      Q_ens(v)
                                                                     Level ∈ {CRITICAL, HIGH,
Tier 2 — 6 diagnostic:                                                 MEDIUM, LOW, MINIMAL}
  PR, CL, EV, pubsub_ratio,
  in_out_ratio, degree_product
```

**Independence guarantee.** The Analyze stage produces Q(v). The Simulate stage (Step 4) produces ground-truth I(v). The Validate stage (Step 5) measures the correlation between Q(v) and I(v). These pipelines must remain independent: M(v) must not be contaminated by simulation outputs, and I(v) must not read Q(v) as an input. This independence is the methodological foundation of the empirical validation claim.

**Validated performance.** The rule-based RMAV Analyze stage achieves Spearman ρ = 0.876 overall (ρ = 0.943 at large scale, 150–300+ nodes) and F1-score > 0.89 across validated system scales. All results are from internal validation against simulation-derived I(v); external cross-system inductive validation remains an open item.

---

## 2. Two Paths: Analyze vs Predict

| | Analyze — Rule-Based (RMAV) | Predict — Learning-Based (GNN) | Predict — Ensemble |
|---|---|---|---|
| **Pipeline stage** | Step 2 (deterministic) | Step 3 (inductive, optional) | Step 3 (inductive, optional) |
| **Mechanism** | AHP-weighted linear combination of Tier 1 metrics | Heterogeneous Graph Transformer (HGTConv) with EdgeFeatureEncoder | Convex blend α·Q_GNN + (1−α)·Q_RMAV |
| **Interpretability** | Full — every score decomposes into metric contributions | Partial — attention weights and per-head outputs | Partial |
| **Requires training data** | No | Yes (simulation-labelled I(v)) | Yes |
| **Node criticality** | ✓ | ✓ | ✓ |
| **Edge criticality** | Proxies (BR, BT of endpoints) | ✓ Direct | ✓ Direct |
| **Anti-pattern detection** | ✓ (runs with Analyze) | — | — |
| **Topic-type branching** | ✓ (FOC formula) | Learned | Learned |
| **MPCI effect** | ✓ Explicit (CDPot_enh) | Learned | Learned |
| **Generalises to unseen systems** | Immediately | Requires fine-tuning | Requires fine-tuning |
| **Spearman ρ (validated)** | 0.876 | TBD (training on ATM dataset pending) | TBD |
| **Primary use** | First analysis; interpretable; CI gate; fallback when no checkpoint | Default predictor after checkpoint is available | Research comparison; activate with `--mode ensemble` |

Both paths classify components into the same five levels and are validated against the same I(v) ground truth, making their predictions directly comparable under the Validate stage protocol.

**Recommended workflow:**
1. Run the **Analyze** stage — immediate results, no training required, full interpretability. RMAV scores serve as the fallback when no GNN checkpoint exists.
2. Run the **Simulate** stage to generate I(v) ground truth (and GNN training labels).
3. Train the GNN on the labelled graph (`cli/train_graph.py`). The GNN becomes the primary predictor automatically.
4. Run the **Predict** stage (`--mode gnn`, the default once a checkpoint exists). RMAV scores are still computed as regularisation input and ensemble right-hand side.
5. For ablation: compare ρ(Q_RMAV, I*), ρ(Q_GNN, I*), ρ(Q_ens, I*) using `--mode rmav|gnn|ensemble`. Switch to ensemble only if Q_ens delivers predictive gain Δρ > 0.03 over GNN alone.

---

## 3. Analyze Stage — Rule-Based RMAV Scoring

### 3.1 The Four Quality Dimensions

| Dimension | Question answered | High score means | Primary stakeholder |
|-----------|------------------|-----------------|---------------------|
| **R — Reliability** | How broadly and deeply does failure propagate? | Failure cascades widely and is hard to contain | Reliability Engineer |
| **M — Maintainability** | How hard is this to change safely? | Tightly coupled; structural bottleneck | Software Architect |
| **A — Availability** | Is this a structural single point of failure? | Removing it partitions the dependency graph | DevOps / SRE |
| **V — Vulnerability** | How attractive a target is this for attack? | Central, reachable, high-value downstream | Security Engineer |

The four dimensions are deliberately **orthogonal** in metric input: each raw metric feeds exactly one dimension (see [Metric Orthogonality](#34-metric-orthogonality)). This means a component's RMAV breakdown tells you *why* it is critical — a pure SPOF has high A but low R, M, V; a God Component has high M; a cascade hub has high R — enabling targeted remediation instead of blanket hardening.

---

### 3.2 RMAV Formulas

All inputs are normalized to [0, 1] by Step 2's rank normalization unless otherwise noted. All RMAV scores are therefore in [0, 1]. Intra-dimension weights are derived from AHP; see [Section 3.5](#35-ahp-weight-derivation).

---

#### Reliability R(v) — Fault Propagation Risk

R(v) measures how broadly and deeply a component's failure propagates through the DEPENDS_ON dependency graph.

**Standard formula (v7)** — Application, Broker, Infrastructure Node, Library:

```
R(v) = 0.60 × RPR(v) × (1 + MPCI(v))  +  0.40 × DG_in(v)
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| RPR(v) | 0.60 | Reverse PageRank — global significance and reachability of the failure origin; higher score indicates the node is a more central propagator |
| MPCI(v) | (Amp) | Multi-Path Coupling Intensity — amplifies RPR when redundant or complex paths increase failure vector density; multiplicative amplifier on RPR |
| DG_in(v) | 0.40 | In-degree (normalised) — immediate blast radius; number of direct dependents |

**Topic-type formula** — used exclusively for Topic nodes:

Topic nodes have DG_in = 0 in the DEPENDS_ON graph because Topics are not DEPENDS_ON endpoints. Their reliability risk is measured instead through subscriber fan-out:

```
R_topic(v) = 0.50 × FOC(v)  +  0.50 × CDPot_topic(v)

CDPot_topic(v) = FOC(v) × (1 − min(publisher_count_norm(v), 1))
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| FOC(v) | 0.50 | Fan-Out Criticality — how many subscribers simultaneously lose their data source |
| CDPot_topic(v) | 0.50 | Fan-out depth — topics with many subscribers but few publishers are pure blast relays with no publisher-side redundancy to absorb the loss |

> **Type dispatch.** The formula branch is resolved by `τ_V(v)` (the vertex type attribute on the graph node). `τ_V(v) = Topic` → Topic formula; all other node types → standard formula.

---

#### Maintainability M(v) — Coupling Complexity

M(v) measures how structurally embedded a component is in the system, making it fragile to change.

```
M(v) = 0.35 × BT(v)
     + 0.30 × w_out(v)
     + 0.15 × CQP(v)
     + 0.12 × CouplingRisk_enh(v)
     + 0.08 × (1 − CC(v))
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| BT(v) | 0.35 | Betweenness Centrality — fraction of shortest dependency paths that pass through v; the defining structural bottleneck signal |
| w_out(v) | 0.30 | QoS-weighted out-degree — efferent coupling weighted by SLA priority; high-priority outgoing dependencies amplify change risk |
| CQP(v) | 0.15 | Code Quality Penalty — composite of cyclomatic complexity, instability, and LCOM; zero for non-Application/Library types (formula degrades gracefully) |
| CouplingRisk_enh(v) | 0.12 | Topology instability enriched by path complexity — peaks when DG_in ≈ DG_out; intensified when shared topics create complex multi-path coupling (see [Section 3.3](#33-derived-terms)) |
| 1 − CC(v) | 0.08 | Inverse clustering coefficient — low local redundancy means each of v's connections is a structurally unique coupling path |

**CQP formula** (Application and Library nodes only; CQP = 0 for all other node types):
```
CQP(v) = 0.40 × complexity_norm(v)  +  0.35 × instability_code(v)  +  0.25 × lcom_norm(v)
```

**Why two instability signals in M(v)?** M(v) contains two coupling-related terms that may appear redundant: `instability_code` (inside CQP) and `CouplingRisk_enh` (topological). They capture distinct architectural layers:
- `instability_code` measures efferent coupling at the *static code level* (package imports, class dependencies) — technically fragile implementations.
- `CouplingRisk_enh` measures efferent coupling at the *runtime topology level* (USES edges, pub-sub relationships) — structurally fragile deployment roles.

These two often diverge: a library can have high static fan-out but only one consumer in the current deployment, or an application can have simple code but hundreds of pub-sub topics. Both signals are needed to capture the full maintenance risk picture.

---

#### Availability A(v) — SPOF Risk

A(v) measures whether a component is a structural single point of failure, weighted by its QoS priority, bridge redundancy, and path elongation.

```
A(v) = 0.35 × AP_c_directed(v) + 0.25 × QSPOF(v) + 0.25 × BR(v) + 0.10 × CDI(v) + 0.05 × w(v)
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| AP_c_directed(v) | 0.35 | Directed articulation point score — primary SPOF signal; removal partitions the dependency graph flows |
| QSPOF(v) | 0.25 | QoS-weighted SPOF severity — `AP_c_directed × w(v)`; amplifies critical SPOFs that serve high-priority traffic |
| BR(v) | 0.25 | Bridge Ratio — fraction of edges that are non-redundant structural bridges |
| CDI(v) | 0.10 | Connectivity Degradation Index — path elongation on removal; soft SPOF signal |
| w(v) | 0.05 | QoS aggregate weight — direct priority bias on the component's own operational weight |

> **AP_c_directed vs AP_c undirected.** The SPOF detection signal uses `AP_c_directed`, which is computed on the *directed* DEPENDS_ON graph using a worst-case out-reachability / in-reachability measure. This correctly captures directed cut vertices — nodes whose removal breaks directed reachability — rather than the undirected articulation point, which can both over-report (paths that are directionally irrelevant) and under-report (asymmetric directed SPOFs) in pub-sub systems.

---

#### Vulnerability V(v) — Security Exposure

V(v) measures how attractive v is as an attack target and how far a compromise would propagate.

```
V(v) = 0.40 × REV(v)  +  0.35 × RCL(v)  +  0.25 × w_in(v)
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| REV(v) | 0.40 | Reverse Eigenvector Centrality — v's downstream dependents are themselves important hubs; compromise at v cascades into high-value targets |
| RCL(v) | 0.35 | Reverse Closeness Centrality — many components can reach v quickly; adversarial paths to v are short |
| w_in(v) | 0.25 | QoS-weighted in-degree (QADS) — direct high-SLA dependents make v attractive because compromising it disrupts the most operationally critical consumers |

---

#### Composite Score Q(v)

```
Q(v) = w_A × A(v)  +  w_R × R(v)  +  w_M × M(v)  +  w_V × V(v)
```

**AHP-derived weights (recommended):**

| Dimension | Weight | Rationale |
|-----------|:------:|-----------|
| Availability (A) | **0.43** | Strongest structural alignment; SPOF severity dominates pre-deployment risk |
| Reliability (R) | **0.24** | Cascade propagation reach; directly tied to blast radius |
| Maintainability (M) | **0.17** | Coupling complexity; amplifies long-term fragility |
| Vulnerability (V) | **0.16** | Security exposure surface; strategic but secondary to structural concerns |

> **Cross-dimension weight derivation.** The 4×4 AHP matrix above gives CR ≈ 0.02 (well within the 0.10 acceptability threshold). The dominant position of A(v) reflects that SPOF failure is the most directly measurable structural risk in a pre-deployment topology: an articulation point with BR = 1.0 partitions the graph with certainty, whereas the cascade depth and coupling effects of R(v) and M(v) are probabilistic.

**Baseline (equal weights).** An equal-weight alternative (0.25 each) can be activated via `--equal-weights` for sensitivity analysis or reproducibility. The AHP-derived weights produce a meaningfully different top-k ranking than equal weights; this difference is part of what the Step 5 validation measures.

**Sensitivity of λ.** The AHP shrinkage factor λ ∈ [0, 1] blends raw AHP weights toward equal weights:
```
w_final(d) = λ × w_AHP(d) + (1 − λ) × 0.25
```
The default λ = 0.70 was selected based on a sensitivity sweep across λ ∈ {0.50, 0.60, 0.70, 0.80, 0.90, 1.00}. Spearman ρ plateaus around λ = 0.70, suggesting the AHP signal saturates beyond that point. The sensitivity sweep result should be reported in any paper submission that uses AHP weights (reviewers will ask).

---

### 3.3 Derived Terms

These scalars are computed inline within the RMAV formulas at scoring time. They are derived from M(v) fields produced in Step 2; they are not stored as independent graph properties.

#### CDPot_enh — Enhanced Cascade Depth Potential

CDPot_enh captures how deeply a failure propagates in the absorber direction of the dependency graph, amplified by multi-path couplings.

```
CDPot_enh(v) = min( CDPot_base(v) × (1 + MPCI(v)),  1.0 )

CDPot_base(v) = ((RPR(v) + DG_in(v)) / 2)  ×  (1 − min(DG_out_raw(v) / max(DG_in_raw(v), ε), 1))

ε = 1e-9  (division guard)
```

Note: `DG_out_raw` and `DG_in_raw` are the raw integer degree counts, not the normalized versions. Using raw counts preserves the ratio semantics — a node with 10 in-edges and 2 out-edges should behave differently from one with 1 and 0.2, even though both have normalized ratio ≈ 0.2.

| Factor | Interpretation |
|--------|---------------|
| `(RPR + DG_in) / 2` | Average cascade reach: global breadth (RPR) combined with immediate blast radius (DG_in) |
| `1 − min(DG_out_raw / DG_in_raw, 1)` | Depth penalty: absorber nodes (DG_in >> DG_out) score high; fan-out hubs (DG_out >> DG_in) approach 0 |
| `× (1 + MPCI)` | Multi-path amplifier: when the same dependents share multiple topics, each topic is an independent failure vector; cascade depth grows with coupling intensity |

**Why MPCI amplifies depth, not breadth.** MPCI counts redundant channels between existing dependent pairs — it does not add new dependent nodes. The count of dependents (DG_in) and their transitive reach (RPR) are unchanged by MPCI. What changes is the *depth* of impact: when v fails, all `path_count` shared topics with each dependent fail simultaneously, making the cascade harder to absorb. This is a depth effect.

**Behaviour reference:**

| Node type | DG_in | DG_out | MPCI | CDPot_base | CDPot_enh | Interpretation |
|-----------|:-----:|:------:|:----:|:----------:|:---------:|---------------|
| Absorber hub | High | Low | 0 | High | High | Deep cascade, single-channel |
| Absorber + multi-path | High | Low | High | High | Very high | Deep cascade, multiple independent vectors |
| Fan-out hub | Low | High | 0 | ≈ 0 | ≈ 0 | Wide but shallow — quickly absorbed |
| Isolated leaf | 0 | 0 | 0 | 0 | 0 | No cascade potential |

---

#### CouplingRisk_enh — Topology Instability with Path Complexity

```
Instability_topo(v) = DG_out_raw(v) / (DG_in_raw(v) + DG_out_raw(v) + ε)

CouplingRisk_base(v) = 1 − |2 × Instability_topo(v) − 1|

CouplingRisk_enh(v) = min(1.0,  CouplingRisk_base(v) × (1 + Δ × path_complexity(v)))

Δ = 0.10 (COUPLING_PATH_DELTA)
```

| Topology role | Instability | CouplingRisk_base | Interpretation |
|--------------|:-----------:|:-----------------:|---------------|
| Pure source (DG_in = 0) | 1.0 | 0 | No afferent pressure — not fragile from above |
| Pure sink (DG_out = 0) | 0.0 | 0 | No efferent pressure — not fragile from below |
| Balanced (DG_in ≈ DG_out) | ≈ 0.5 | ≈ 1.0 | Maximum fragility — structural pressure from both directions |

The `path_complexity` term is an intensifier: if a node is already coupling-balanced (fragile), having many redundant topics per dependency further increases synchronisation complexity, raising the effective maintenance risk. The result is capped at 1.0.

---

#### QSPOF — QoS-Weighted SPOF Severity

```
QSPOF(v) = AP_c_directed(v) × w(v)
```

Scales the directed articulation point score by the component's operational QoS weight. A component that is structurally a SPOF *and* handles high-priority traffic is a doubly severe availability risk: its removal is both certain to disconnect the graph and certain to affect the most critical data flows.

---

### 3.4 Metric Orthogonality

Each raw metric from M(v) feeds **exactly one** RMAV dimension. No metric appears in more than one formula. Violations would inflate the effective weight of shared metrics relative to the AHP calibration.

| Metric | Symbol | R | M | A | V | Notes |
|--------|--------|:-:|:-:|:-:|:-:|-------|
| Reverse PageRank | RPR | ✓ | | | | Global cascade reach |
| In-Degree (norm) | DG_in | ✓ | | | | Immediate blast radius |
| MPCI | MPCI | ✓ via CDPot | | | | Amplifier only; enters via derived term |
| Fan-Out Criticality | FOC | ✓ Topics | | | | Substitutes for DG_in on Topic nodes |
| Path Complexity | path_complexity | | ✓ via CouplingRisk | | | Structural coupling depth |
| Betweenness | BT | | ✓ | | | Structural bottleneck |
| QoS Out-Degree | w_out | | ✓ | | | Priority-weighted efferent coupling |
| Code Quality Penalty | CQP | | ✓ | | | Complexity + instability + LCOM |
| CouplingRisk_enh | CouplingRisk | | ✓ | | | Derived from DG_in_raw, DG_out_raw |
| Clustering Coefficient | CC | | ✓ as 1−CC | | | Local path redundancy |
| Directed AP Score | AP_c_directed | | | ✓ | | Directly in A(v) and via QSPOF |
| Bridge Ratio | BR | | | ✓ | | Non-redundant edge fraction |
| CDI | CDI | | | ✓ | | Path elongation on removal |
| Reverse Eigenvector | REV | | | | ✓ | Strategic downstream value |
| Reverse Closeness | RCL | | | | ✓ | Adversarial reach speed |
| QoS In-Degree (QADS) | w_in | | | | ✓ | High-SLA attack surface |
| PageRank | PR | — | — | — | — | Diagnostic only (Tier 2) |
| Closeness | CL | — | — | — | — | Diagnostic only (Tier 2) |
| Eigenvector | EV | — | — | — | — | Diagnostic only (Tier 2) |

---

### 3.5 AHP Weight Derivation

Intra-dimension weights are derived from the **Analytic Hierarchy Process (AHP)** using pairwise comparison matrices on Saaty's 1–9 scale.

```
Step 1 — Construct n×n matrix A:  A[i][j] = importance of criterion i relative to j
          Reciprocal constraint:  A[j][i] = 1 / A[i][j]

Step 2 — Geometric mean per row:  GM[i] = ( ∏_j A[i][j] )^(1/n)

Step 3 — Normalise:  w[i] = GM[i] / Σ_j GM[j]

Step 4 — Consistency check:
          λ_max = average of ( (A·w)[i] / w[i] )  for all i
          CI    = (λ_max − n) / (n − 1)
          CR    = CI / RI[n]
          Abort if CR > 0.10
```

Reference RI values (Saaty 1980): n=3 → 0.58, n=4 → 0.90, n=5 → 1.12, n=6 → 1.24.

#### Reliability AHP (3×3: RPR, DG_in, CDPot_enh)

```
            RPR    DG_in  CDPot
RPR      [ 1.00,  1.50,  2.00 ]   RPR: global reach is the primary cascade signal
DG_in    [ 0.67,  1.00,  1.50 ]   DG_in: immediate dependents are secondary
CDPot    [ 0.50,  0.67,  1.00 ]   CDPot: cascade depth is supplementary

→ AHP raw weights:  [0.45,  0.30,  0.25]    CR ≈ 0.001
```

MPCI enters R(v) indirectly through CDPot_enh and does not add a 4th AHP criterion. This preserves the 3×3 matrix and its near-zero CR while capturing the MPCI effect.

#### Maintainability AHP (5×5: BT, w_out, CQP, CouplingRisk, CC_inv)

```
            BT     w_out  CQP    CR     CC_inv
BT       [1.00,  1.17,  2.33,  2.92,  4.38]   BT: primary structural bottleneck
w_out    [0.86,  1.00,  2.00,  2.50,  3.75]   w_out: QoS-weighted efferent coupling
CQP      [0.43,  0.50,  1.00,  1.25,  1.88]   CQP: code-level coupling
CR       [0.34,  0.40,  0.80,  1.00,  1.50]   CouplingRisk: topology instability
CC_inv   [0.23,  0.27,  0.53,  0.67,  1.00]   CC_inv: local redundancy (supplementary)

→ AHP raw weights:  [0.35,  0.30,  0.15,  0.12,  0.08]    CR ≈ 0.000
```

CQP and CouplingRisk receive equal AHP judgement because both measure coupling — CQP at the code level, CouplingRisk at the deployment topology level — and neither dominates the other a priori.

#### Availability AHP (5×5: AP_c_directed, QSPOF, BR, CDI, w)

```
                AP_c   QSPOF    BR     CDI      w
AP_c_directed [1.00,  1.40,  1.40,  3.50,  7.00]   AP_c: primary directed SPOF signal
QSPOF         [0.71,  1.00,  1.00,  2.50,  5.00]   QSPOF: QoS-amplified SPOF severity
BR            [0.71,  1.00,  1.00,  2.50,  5.00]   BR: bridge fraction
CDI           [0.29,  0.40,  0.40,  1.00,  2.00]   CDI: soft SPOF / path elongation
w             [0.14,  0.20,  0.20,  0.50,  1.00]   w: direct QoS priority weight

→ AHP raw weights:  [0.35,  0.25,  0.25,  0.10,  0.05]    CR ≈ 0.001
```

#### Composite Q AHP (4×4: A, R, M, V)

```
       A      R      M      V
A   [1.00,  1.50,  2.50,  2.67]   Availability: dominant (SPOF = certain graph partition)
R   [0.67,  1.00,  1.67,  1.78]   Reliability: propagation reach is second priority
M   [0.40,  0.60,  1.00,  1.07]   Maintainability: coupling fragility is tertiary
V   [0.37,  0.56,  0.93,  1.00]   Vulnerability: security exposure is supplementary

→ AHP raw weights:  [0.43,  0.24,  0.17,  0.16]    CR ≈ 0.02
```

---

### 3.6 Weight Shrinkage Strategy

Raw AHP weights can be extreme on small comparison sets. The shrinkage strategy formally blends them with a uniform prior:

```
w_final(d) = λ × w_AHP(d)  +  (1 − λ) × (1 / n_dimensions)
```

The default λ = 0.70 was selected from a sensitivity sweep across {0.50, 0.60, 0.70, 0.80, 0.90, 1.00}. Spearman ρ on the ATM validation dataset plateaus in the λ ∈ [0.65, 0.75] range. The empirical sensitivity sweep result must be included in any paper submission that cites AHP-derived weights (λ = 0.70 as a point estimate is not sufficient; reviewers will ask about sensitivity).

---

### 3.7 Criticality Classification

RMAV scores are classified using an **adaptive box-plot classifier** that identifies components exceptional relative to the system's own distribution:

```
CRITICAL  :  score > Q3 + 1.5 × IQR   (structural outliers)
HIGH      :  Q3 < score ≤ upper fence   (upper quartile, non-outlier)
MEDIUM    :  median < score ≤ Q3
LOW       :  Q1 < score ≤ median
MINIMAL   :  score ≤ Q1
```

Classification is applied **independently per RMAV dimension and for the composite Q(v)**. A component can be CRITICAL on Availability (structural SPOF) but MINIMAL on Vulnerability — which is exactly the diagnostic information needed to direct remediation.

**Small-sample fallback (n < 12).** Box-plot thresholds become unstable at small node counts. For graphs with fewer than 12 components, percentile thresholds are used instead: CRITICAL = top 10%, HIGH = 75th–90th, MEDIUM = 50th–75th, LOW = 25th–50th, MINIMAL = bottom 25%.

**Typical distribution across validated scenarios:** CRITICAL ≈ 5–15%, HIGH ≈ 25%, MEDIUM ≈ 25%, LOW ≈ 25%, MINIMAL ≈ bottom 10–25%.

---

### 3.8 Interpretation Patterns

The combination of RMAV dimension scores characterises the *type* of risk and directs remediation:

| Pattern | R | M | A | V | Primary risk | Recommended action |
|---------|:-:|:-:|:-:|:-:|-------------|-------------------|
| **Full hub** | H | H | H | H | Catastrophic — all failure modes | Redundancy + circuit breakers + hardening |
| **Reliability hub** | H | L | L | L | Wide cascade | Retry logic, graceful degradation, back-pressure |
| **Bottleneck** | L | H | L | L | Change fragility | Reduce coupling; extract an interface or façade |
| **SPOF** | L | L | H | L | Availability loss | Redundant instance, active-passive failover |
| **High-value target** | L | L | L | H | Compromise propagation | Zero-trust boundaries, audit logs, network isolation |
| **Compound: SPOF + hub** | H | H | H | H | Unreliable *and* unrefactorable | Architecture redesign required before any other mitigation |
| **Multi-path sink** | H (MPCI>0) | M | M | L | Deep multi-channel cascade | Reduce shared-topic count between the same dependent pair |
| **Maintenance debt** | M | H | M | L | Technical debt accumulation | Prioritise refactoring before the next feature sprint |
| **Leaf** | L | L | L | L | None | Standard monitoring |

> **Compound SPOF + God Component.** A component that is simultaneously an articulation point (high A) and a structural bottleneck with high total degree (high M) is the highest-priority compound risk in the catalog. It is unreliable (any failure partitions the graph) *and* untestable/unrefactorable (too many responsibilities to change safely). In the ATM system, `ConflictDetector` (Q ≈ 0.90, AP = true) is the primary compound risk candidate.

---

## 4. Predict Stage — Graph Neural Network

### 4.1 Motivation

The RMAV rule-based Analyze stage has two structural limitations:

**Fixed feature interactions.** RMAV combines metrics via fixed AHP weights determined before analysis. It cannot discover that, for a specific topology, the interaction between BT and RPR is more predictive than either metric alone.

**Node-only scoring.** Edges are scored only via endpoint proxies (BR, BT of endpoints). Direct edge-level criticality — identifying which specific pub-sub relationship is most dangerous to lose — requires edge-level supervision.

The Predict stage learns from simulation ground truth I(v) and produces directly supervised edge criticality scores alongside node scores.

---

### 4.2 Architecture Overview

```
Step 2 — Analyze:  Q_RMAV(v)   — AHP-weighted formula (deterministic, closed-form)
Step 3 — Predict:  Q_GNN(v)    — trained on I(v) simulation ground truth
Step 3 — Ensemble: Q_ens(v)    = α · Q_GNN(v) + (1−α) · Q_RMAV(v)
```

Three cooperating modules:

```
    NetworkX DiGraph (Step 1 output)
              │
   ┌──────────▼───────────────────────┐
   │   Data Preparation               │  Type-specific node features:
   │   networkx_to_hetero_data()      │    App/Lib=23, Broker=19, Topic=22, Node=20
   │   HeteroData + splits            │   16-dim edge features (weight, path_count_norm,
   └──────────┬───────────────────────┘    7-bit type one-hot, 7 QoS decomposition dims)
              │                            5-dim simulation labels y = I*(v)
              │                            5-dim RMAV scores  y_rmav = Q_RMAV(v)
      ┌───────┼────────────┐
      ▼       ▼            ▼
   NodeGNN  EdgeGNN   EnsembleGNN
   3L HGT   TypedEdge  α · Q_GNN     ← Step 3 output (default mode="gnn")
   +EdgeFea  Encoder  +(1-α)·Q_RMAV ← Q_RMAV from Step 2
   +BiDir   (E, 16)   (N, 5)
   (N, 5)
```

All three modules are implemented in `saag/prediction/` and managed by `GNNService`.

---

### 4.3 Graph Data Preparation

`networkx_to_hetero_data()` in `data_preparation.py` converts the Step 1 NetworkX graph to a PyTorch Geometric `HeteroData` object, partitioning nodes and edges by type.

#### Node Feature Vector (type-specific dimensions)

Each node `v` is represented by a feature vector whose dimension depends on its type. The first 18 indices are the shared topological base present for all node types. Type-specific extra features follow at indices 18+.

| Node type | Total dim | Extra features (indices 18+) |
|-----------|:---------:|------------------------------|
| Application | 23 | +5 code quality attributes (indices 18–22) |
| Library | 23 | +5 code quality attributes (indices 18–22) |
| Broker | 19 | +1 `max_connections_norm` (index 18) |
| Topic | 22 | +4 `subscriber_count_norm`, `publisher_count_norm` (indices 18–19), `log1p_frequency_norm` (index 20, per-scenario z-score), `topic_qos_criticality_ord` (index 21, ordinal 0–4) |
| Node (infra) | 20 | +2 `cpu_cores_norm`, `memory_gb_norm` (indices 18–19) |

> **Implementation note.** The HGT architecture handles type-specific projections internally, so a global one-hot node-type vector is **not required** and has been removed to reduce parameter bloat. The constant `NODE_TYPE_TO_DIM` in `data_preparation.py` defines the authoritative widths. Infrastructure extra features (`cpu_cores_norm`, `memory_gb_norm`, `max_connections_norm`) are derived by per-graph min-max normalization of node attributes in `_normalize_infra_features()`. Topic runtime features (`subscriber_count_norm`, `publisher_count_norm`) are derived by counting `SUBSCRIBES_TO`/`PUBLISHES_TO` edges per Topic node and dividing by the graph maximum. `log1p_frequency_norm` uses per-scenario z-score of log1p(Hz) to avoid cross-domain leakage. `topic_qos_criticality_ord` is the ordinal encoding (0–4) of the 5-level QoS urgency label; when all topics in a graph share the same criticality (zero variance), the field is masked to 0.0 to prevent covariate shift across scenarios.

**Topological metrics — indices 0–17 (base for all node types):**

| Index | Metric | RMAV role |
|:-----:|--------|-----------|
| 0 | PageRank (PR) | Diagnostic (Tier 2) |
| 1 | Reverse PageRank (RPR) | R(v) |
| 2 | Betweenness Centrality (BT) | M(v) |
| 3 | Closeness Centrality (CL) | Diagnostic (Tier 2) |
| 4 | Eigenvector Centrality (EV) | Diagnostic (Tier 2) |
| 5 | In-Degree normalized (DG_in) | R(v) |
| 6 | Out-Degree normalized (DG_out) | CouplingRisk_enh |
| 7 | Clustering Coefficient (CC) | M(v) as 1−CC |
| 8 | AP_c Score | Diagnostic (Topological) |
| 9 | Bridge Ratio (BR) | A(v) |
| 10 | QoS aggregate weight (w) | QSPOF, A(v) |
| 11 | QoS weighted in-degree (w_in) | V(v) |
| 12 | QoS weighted out-degree (w_out) | M(v) |
| 13 | MPCI | R(v) via CDPot_enh amplifier |
| 14 | path_complexity | M(v) via CouplingRisk_enh |
| 15 | Fan-Out Criticality (FOC) | R(v) for Topic nodes |
| 16 | AP_c_directed | A(v) directly and via QSPOF |
| 17 | CDI (Connectivity Degradation) | A(v) |

**Code quality metrics — indices 18–22 (Application and Library only):**

| Index | Metric | RMAV role |
|:-----:|--------|-----------|
| 18 | loc_norm | Diagnostic |
| 19 | complexity_norm | M(v) via CQP |
| 20 | instability_code | M(v) via CQP |
| 21 | lcom_norm | M(v) via CQP |
| 22 | code_quality_penalty (CQP) | M(v) directly |

**Infrastructure metrics — indices 18–19 (Broker, Topic, Node only):**

| Index | Metric | Node type | Source |
|:-----:|--------|-----------|--------|
| 18 | max_connections_norm | Broker | Node attribute, normalized per Broker subgraph |
| 18 | subscriber_count_norm | Topic | SUBSCRIBES_TO in-edge count, normalized per graph |
| 19 | publisher_count_norm | Topic | PUBLISHES_TO in-edge count, normalized per graph |
| 18 | cpu_cores_norm | Node | Node attribute, normalized per Node subgraph |
| 19 | memory_gb_norm | Node | Node attribute, normalized per Node subgraph |

#### Edge Features (16 dimensions)

| Index | Feature |
|:-----:|---------|
| 0 | QoS weight w(e) |
| 1 | path_count_norm = log₂(1 + path_count) / log₂(17) — coupling intensity (capped at 16 paths) |
| 2–8 | Edge-type one-hot (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO, USES, DEPENDS_ON) |
| 9 | reliability_score (0.0 BEST_EFFORT / 1.0 RELIABLE) — non-zero for PUBLISHES_TO, SUBSCRIBES_TO, DEPENDS_ON |
| 10 | durability_score (VOLATILE=0.0 / TRANSIENT_LOCAL=0.5 / TRANSIENT=0.6 / PERSISTENT=1.0) — pub/sub only |
| 11 | priority_score (LOW=0.0 / MEDIUM=0.33 / HIGH=0.66 / URGENT=1.0) — pub/sub only |
| 12 | has_deadline (1.0 if finite deadline_ns is set, else 0.0) — pub/sub only |
| 13 | deadline_ns_log = log10(1 + deadline_ns / 1e6), clamped to [0, 1] — pub/sub only |
| 14 | max_blocking_ms_log = log10(1 + max_blocking_ms), clamped to [0, 1] — pub/sub only |
| 15 | qos_heterogeneity_flag (1.0 if edge QoS profile differs from scenario-level modal profile, else 0.0) — pub/sub only |

Dimensions 9–15 are non-zero only for PUBLISHES_TO / SUBSCRIBES_TO edges, where QoS profiles are semantically meaningful. All other edge types receive zeros for these dimensions, preserving backward numerical compatibility.

**Node labels** (`data[type].y`, shape (n, 5)): simulation ground-truth per-dimension impact scores `[I*(v), IR(v), IM(v), IA(v), IV(v)]`.

**RMAV scores** (`data[type].y_rmav`, shape (n, 5)): RMAV quality scores `[Q(v), R(v), M(v), A(v), V(v)]`, stored for ensemble blending. These are *not* used as training labels; they are the ensemble right-hand side in Step 3c.

---

### 4.4 HGT Model (Heterogeneous Graph Transformer)

The model uses a **3-layer Heterogeneous Graph Transformer (HGTConv)** with type-dependent key/query/value projections. For each `(src_type, edge_type, dst_type)` triple, HGT learns separate attention parameters — the correct inductive bias for a graph with 5 node types and 7+ edge types. This is a stronger inductive bias than GAT-within-HeteroConv, which does not model joint source-relation-target type context.

**HGTConv does not accept raw edge_attr tensors.** A dedicated `EdgeFeatureEncoder` bridges this gap by aggregating edge features into destination-node embeddings via scatter-mean before each HGT layer.

```
Layer 0 — Type-specific input projection:
  h_v^(0) = GELU( LayerNorm( W_{type(v)} · x_v ) )

Pre-layer edge injection (EdgeFeatureEncoder, applied per layer k):
  e_v^(k) = scatter_mean_{u → v}( W_e · e_{uv} )     ← 16-dim edge → D-dim
  h_v^(k) ← h_v^(k) + e_v^(k)                        ← edge signal added before MP

Layer k — HGT message passing per (src_type s, edge_type r, dst_type d):
  ATT(u,v) = softmax( (K_s^(k) h_u)^T · (Q_d^(k) h_v) / √D_h · W_{ATT}^(s,r,d) )
  MSG(u,v) = V_s^(k) h_u · W_{MSG}^(s,r,d)
  m_v^(k)  = Σ_{u,r} ATT(u,v) · MSG(u,v)
  h_v^(k+1) = GELU( LayerNorm( W_{agg} · m_v^(k)  +  h_v^(k) ) )   ← residual

Reverse pass (use_bidirectional=True, computed on-the-fly within encode()):
  rev_ei = { (dst, "rev_"+etype, src) : flip(edge_index) }  for each relation
  h_rev  = rev_conv(h, rev_ei)
  h_v   ← h_v + 0.5 · h_rev[v]                              ← upstream signal
```

Residual connections between layers. Hidden dimension D = 64. Dropout p = 0.2. The reverse pass captures upstream (subscriber ← topic failure) and downstream (publisher → consumer cascade) structural signals without modifying data preparation — reverse edge indices are built on-the-fly inside `encode()`.

---

### 4.5 Multi-Task Prediction Heads

```
R̂(v) = MLP_R( h_v )                            — Reliability head
M̂(v) = MLP_M( h_v )                            — Maintainability head
Â(v)  = MLP_A( h_v )                            — Availability head
V̂(v)  = MLP_V( h_v )                            — Vulnerability head
Î*(v) = MLP_C( h_v ‖ R̂(v) ‖ M̂(v) ‖ Â(v) ‖ V̂(v) ) — Composite head
```

All outputs pass through sigmoid activation, producing scores in [0, 1]. The composite head receives dimension predictions as additional input, so the final Î*(v) can incorporate cross-dimension interactions that the linear RMAV composite cannot represent.

---

### 4.6 Edge Criticality Prediction

```
score(u, v) = TypedEdgeEncoder_r( h_u, h_v, e_{uv} )

e_{uv} ∈ ℝ^9: QoS weight + path_count_norm + 7-bit edge-type one-hot
```

The `TypedEdgeEncoder` replaces the former shared MLP. Each relation type `r` has a dedicated linear projection `W_r ∈ ℝ^{16×D}` learned independently. The per-relation projected edge feature is fused with source and destination node embeddings via a shared `[h_src ‖ h_dst ‖ e_proj]` → LayerNorm → GELU layer before the output head. This ensures the model learns distinct edge-criticality semantics for PUBLISHES_TO vs DEPENDS_ON edges rather than forcing them through a common feature space.

**GNN edge feature encoding (implementation detail).** The `EdgeFeatureEncoder` in `NodeCriticalityGNN` aggregates the full 16-dim edge attributes into destination-node embeddings via `scatter_mean` before each HGT layer. HGTConv does not accept `edge_attr` directly, so the encoder projects `e_uv` from 16 to `hidden_channels` and adds the result to the destination node embedding. The edge-specific `TypedEdgeEncoder` used by `EdgeCriticalityGNN` also consumes the full 16-dim vector.

**Edge labels.** Training labels for edges are derived from simulation ground truth with a bridge-aware multiplier:

```
I_edge(u, v) = max( I*(u), I*(v) ) × bridge_multiplier

bridge_multiplier = 1.0   if (u, v) is a structural bridge
                   = 0.1   otherwise
```

This multiplier downweights edges that are not bridges, reducing label noise from redundant relationships. Labels are stored as `data[rel].y_edge` with shape `(E, 5)`, one column per RMAV dimension.

---

### 4.7 Ensemble: GNN + RMAV

```
Q_ens(v) = α · Q_GNN(v)  +  (1−α) · Q_RMAV(v)

α ∈ ℝ^5  — per-dimension learnable blend coefficients, α_d = sigmoid(logit_d)
           Initialised at logit = 0  →  α = 0.5  (equal blend start)
```

The ensemble is a thin `EnsembleGNN` module containing only the 5-dimensional `alpha_logit` parameter. It is fine-tuned after the main GNN training loop using the training-node subset.

> **Default mode.** The default prediction mode is `"gnn"` (GNN-only output) after a checkpoint exists. `"ensemble"` must be explicitly requested via `--mode ensemble`. RMAV remains the fallback when no checkpoint is found (`predict_quality_with_gnn()` in `PredictionService` detects checkpoint availability via `_has_checkpoint()`).

> **Implementation note.** When `mode="ensemble"` is requested but `y_rmav` is absent from the HeteroData or the ensemble module is not initialised, the result falls back to `prediction_mode="gnn_only"` and `ensemble_scores` is left empty. Callers should check `prediction_mode` and `ensemble_scores` to distinguish a true ensemble result from a fallback.

---

### 4.8 Training Protocol

**Transductive (single graph, default).** 60/20/20 train/val/test split applied per node type via `create_node_splits()`. Maximum 300 epochs.

**Label normalization.** Before node splits are created, all simulation labels are normalized in-place via `normalize_labels_robust()`: IQR-based robust normalization `sigmoid((y − median) / IQR)` with the IQR clipped to a minimum of 1e-6, applied jointly across all node types. This reduces the influence of simulation outliers (extreme cascade cascades at small-graph scales) on the training signal.

**Early stopping.** Combined-metric early stopping with patience = 30 epochs:
```
combined = 0.6 × val_rho  +  0.4 × max(0,  1 − val_loss / (best_val_loss + ε))
```
This balances ranking quality (Spearman ρ) with absolute loss improvement, reducing cases where high ρ hides a degraded loss surface or vice versa.

**Loss function:**

```
L = L_composite  +  0.5 × L_dimension  +  0.3 × L_rank  +  0.1 × L_pairwise

L_composite   = MSE( Î*(v),   I*(v) )                  — composite impact
L_dimension   = Σ_d  MSE( d̂(v),  I_d*(v) )            — per-dimension impact
L_rank        = −(1/N) Σ_v  log P(rank of v)           — ListMLE list-level ranking
L_pairwise    = Σ_{i,j: t_i−t_j > m}  max(0, m − (s_i−s_j)) / n_pairs  — pairwise margin
                margin m = 0.05
```

> **Note on L_dimension.** Each per-RMAV head is supervised against the simulation-derived per-dimension impact `I_d*(v)` (not against `Q_RMAV_d(v)` scores). The GNN learns to predict simulation impact, not to reproduce the RMAV formula. `L_RMAV` used in older internal comments is a misnomer; `L_dimension` is the correct term.

> **Note on L_pairwise.** L_pairwise enforces pairwise ordering: if component i should rank strictly above j (target difference > margin), the model is penalised when its predicted scores fail to preserve that ordering. It supplements ListMLE's list-level signal with direct pair-level correctness.

**Optimizer and scheduler:** AdamW, lr = 3×10⁻⁴, weight_decay = 10⁻⁴, gradient clipping max_norm = 1.0. Learning rate schedule: `CosineAnnealingWarmRestarts(T_0 = max(50, epochs//4), T_mult=2, η_min = lr×0.01)`. Warm restarts help escape local minima that are common when training on heterogeneous graphs with varying simulation label density across scenarios.

**Ensemble fine-tuning.** After the main training loop, `EnsembleGNN.alpha_logit` is fine-tuned for 100 epochs using Adam at lr = 1×10⁻³. GNN predictions are computed under `torch.no_grad()` during this step, so gradients flow only through α. This limits ensemble adaptation to a global blend scalar and cannot correct systematic biases in the GNN's direction of error.

---

### 4.9 Multi-Seed Stability

Multi-seed training runs the full train loop independently for each seed in `{42, 123, 456, 789, 2024}`. Training metrics are averaged across seeds and logged. The implementation restores the weights from the **best-performing seed** (by validation Spearman ρ) before final inference or serialization, ensuring that `best_model.pt` reflects the global optimum across the searched seeds.

**Inductive evaluation.** For inductive generalisation claims (train on system A, evaluate on system B), `inductive_graphs` can be passed to `GNNService.train()`. All graphs in this list are added to the DataLoader but only the primary graph has train/val/test splits applied; inductive graphs contribute all their nodes to training.

---

### 4.10 Methodological Integrity

The following table tracks the technical hardening of the GNN pipeline. All high-severity items are resolved in the current production implementation.

| ID | Issue | Severity | Status | Solution / Mitigation |
|----|-------|:--------:|--------|-----------------------|
| G1 | Node feature dim mismatch | High | **Resolved** | Per-type dims enforced (App/Lib=23, Broker=19, Topic=22, Node=20); `feature_version=3` in checkpoint config; old checkpoints load with `strict=False` + warning rather than hard error |
| G2 | Edge labels as node proxies | Medium | Open | Current: `max(I*(u), I*(v))`. Future: direct removal impact per edge |
| G3 | Redundant type encoding | Low | **Resolved** | Zero-padding removed; `KEYS_BY_TYPE` selects only type-appropriate features |
| G4 | Transductive leakage | High | Open | Test nodes are present in MP graph. Requires pure inductive split for SoS claims |
| G5 | Best seed weight saving | High | **Resolved** | `best_state` is now restored before `train()` returns |
| G6 | Ensemble α bias | Medium | Open | α is a global blend; cannot correct local directional GNN errors |
| G7 | Prediction seed overwrite | High | **Resolved** | `predict_from_data()` now uses persisted `self._best_seed` for mask consistency |
| G8 | Ensemble validation gap | High | **Resolved** | `ensemble_metrics` now computed via `evaluate_scores` in prediction path |
| G9 | Silent ensemble fallback | Medium | **Resolved** | Added `prediction_mode` field and explicit fallback logging |
| G10 | Layer compatibility check | Medium | **Resolved** | `from_checkpoint()` validates `layer` alias against checkpoint metadata |
| G11 | Edge feature dim mismatch | Medium | **Resolved** | `EDGE_FEATURE_DIM` changed 8→9; `path_count_norm` added at index 1; existing checkpoints must be retrained (architectural change) |
| G12 | HGTConv edge_attr incompatibility | Medium | **Resolved** | `EdgeFeatureEncoder` (scatter-mean projection) injects edge signals into dst-node embeddings before each HGT layer |
| G13 | Single-cycle LR decay | Low | **Resolved** | `CosineAnnealingWarmRestarts` with T_0, T_mult=2 replaces single-cycle decay; warm restarts escape local minima |
| G14 | Spearman-only early stopping | Low | **Resolved** | Combined score `0.6×ρ + 0.4×loss_improvement`; prevents ρ-high/loss-degraded runs from being saved |

---

## 5. Comparing the Two Stages

| Property | Analyze — RMAV (rule-based) | Predict — GNN (learning-based) | Predict — Ensemble |
|----------|:-----------------:|:--------------------:|:--------:|
| Requires training data | No | Yes | Yes |
| Node criticality | ✓ | ✓ | ✓ |
| Edge criticality | Proxies (BR, BT) | ✓ Direct (approx.) | ✓ Direct |
| Per-dimension decomposition | ✓ Explicit | ✓ Learned heads | ✓ Blended |
| Interpretability | Full | Partial (attention + heads) | Partial |
| Topic-type branching | ✓ Explicit | Learned | Learned |
| MPCI amplification | ✓ Explicit (CDPot_enh) | Learned | Learned |
| Generalises to unseen systems | Immediately | Requires fine-tuning | Requires fine-tuning |
| Spearman ρ (ATM, validated) | 0.876 overall; 0.943 large-scale | Pending | Pending |
| F1-score (ATM, validated) | 0.893 | Pending | Pending |
| Transductive leakage risk | None (formula-based) | Present (G4) | Present |
| Primary use | First analysis; interpretable; CI gate; fallback when no checkpoint | Default predictor after training; RMAV = fallback | Research comparison; use `--mode ensemble` |

**Three-way ablation protocol.** Before claiming that GNN or Ensemble outperforms RMAV, run the controlled ablation using `--mode rmav|gnn|ensemble` in `predict_graph.py`. Use Wilcoxon signed-rank test on per-component ρ values across seeds. Predictive gain threshold for switching: Δρ > 0.03 (one standard deviation on the ATM dataset).

---

## 6. Worked Example

**System from Step 2 worked example** (SensorApp, MonitorApp, MainBroker, NavLib, /temperature).

**Step 2 metric vector inputs:**

```
Component        PR    DG_in  MPCI  FOC   BT    AP_c_dir  BR    CDI   REV   RCL   w_in
──────────────────────────────────────────────────────────────────────────────────────────
SensorApp        0.58  0.25   0.0   0.0   0.40  0.43     1.0   0.2   0.3   0.4   0.0
MonitorApp       0.25  0.0    0.0   0.0   0.0   0.0      0.0   0.0   0.5   0.6   0.0
MainBroker       0.65  0.50   0.0   0.0   0.60  0.65     1.0   0.5   0.6   0.7   0.71
NavLib           0.72  0.50   0.0   0.0   0.50  0.50     1.0   0.4   0.4   0.5   0.71
/temperature     0.0   0.0    0.0   1.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0
```

**R(v) scores (v7 formula: 0.60 × PR × (1 + MPCI) + 0.40 × DG_in):**

```
SensorApp:    R = 0.60×0.58 + 0.40×0.25 = 0.348 + 0.100 = 0.448
MonitorApp:   R = 0.60×0.25 + 0.40×0.0  = 0.150 + 0.000 = 0.150
MainBroker:   R = 0.60×0.65 + 0.40×0.50 = 0.390 + 0.200 = 0.590
NavLib:       R = 0.60×0.72 + 0.40×0.50 = 0.432 + 0.200 = 0.632
/temperature: R_topic = 0.50×1.0 + 0.50×(1.0 × (1 − min(0.0, 1))) = 1.000  ← highest in system
```
Component        RPR   DG_in  MPCI  FOC   BT    AP_c_dir  BR    CDI   REV   RCL   w_in
──────────────────────────────────────────────────────────────────────────────────────────
SensorApp        0.58  0.25   0.0   0.0   0.40  0.43     1.0   0.2   0.3   0.4   0.0
MonitorApp       0.25  0.0    0.0   0.0   0.0   0.0      0.0   0.0   0.5   0.6   0.0
MainBroker       0.65  0.50   0.0   0.0   0.60  0.65     1.0   0.5   0.6   0.7   0.71
NavLib           0.72  0.50   0.0   0.0   0.50  0.50     1.0   0.4   0.4   0.5   0.71
/temperature     0.0   0.0    0.0   1.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0
```

**CDPot_enh calculations** (MPCI = 0 for all; example uses equal raw in/out counts):

```
SensorApp:   CDPot_base = ((0.58+0.25)/2) × (1 − 0)    = 0.415
             CDPot_enh  = 0.415 × (1+0) = 0.415

MainBroker:  CDPot_base = ((0.65+0.50)/2) × (1 − 0)    = 0.575
             CDPot_enh  = 0.575

NavLib:      CDPot_base = ((0.72+0.50)/2) × (1 − 0)    ≈ 0.610
             CDPot_enh  = 0.610
```

**R(v) scores:**

```
SensorApp:    R = 0.45×0.58 + 0.30×0.25 + 0.25×0.415 = 0.261 + 0.075 + 0.104 = 0.440
MonitorApp:   R = 0.45×0.25 + 0.30×0.0  + 0.25×0.0   = 0.113
MainBroker:   R = 0.45×0.65 + 0.30×0.50 + 0.25×0.575 = 0.293 + 0.150 + 0.144 = 0.587
NavLib:       R = 0.45×0.72 + 0.30×0.50 + 0.25×0.610 = 0.324 + 0.150 + 0.153 = 0.627
/temperature: R_topic = 0.50×1.0 + 0.50×(1.0 × 1.0)  = 1.000  ← highest in system
```

Key observations:
- `/temperature` scores R = 1.0 because FOC = 1.0 (the only topic with active subscribers) and there is only one publisher. In a larger system with multiple topics, this would rank relative to peers.
- **NavLib** outranks MainBroker on R because it is depended upon by both application nodes directly (via Rule 5 `app_to_lib` DEPENDS_ON edges), giving it higher RPR.
- **MonitorApp** has R = 0.113 — no components depend on it, so its failure cascades to no one.

**A(v) scores (abbreviated):**

```
MainBroker: A = 0.35×0.65 + 0.25×(0.65×0.71) + 0.25×1.0 + 0.10×0.5 + 0.05×0.71
              = 0.228 + 0.115 + 0.250 + 0.050 + 0.036 = 0.679   → [HIGH]

NavLib:     A = 0.35×0.50 + 0.25×(0.50×0.71) + 0.25×1.0 + 0.10×0.4 + 0.05×0.71
              = 0.175 + 0.089 + 0.250 + 0.040 + 0.036 = 0.590   → [HIGH]
```

Both MainBroker and NavLib are structural SPOFs with BR = 1.0. Adding redundancy for either would be the top remediation priority for this system.

---

## 7. Output Schema

The JSON output from `predict_graph.py --output results/prediction.json` follows this structure:

```json
{
  "layers": {
    "system": {
      "total_components": 35,
      "rmav": {
        "NavLib": {
          "overall":          0.54,
          "reliability":      0.63,
          "maintainability":  0.41,
          "availability":     0.58,
          "vulnerability":    0.52,
          "level":            "HIGH",
          "is_spof":          true,
          "blast_radius":     12,
          "cascade_depth":    4
        },
        "/temperature": {
          "overall":          0.33,
          "reliability":      1.00,
          "maintainability":  0.10,
          "availability":     0.12,
          "vulnerability":    0.08,
          "level":            "MEDIUM",
          "is_spof":          false,
          "blast_radius":     3,
          "cascade_depth":    1,
          "fan_out_criticality": 1.00
        }
      },
      "antipatterns": [
        {
          "entity_id":    "NavLib",
          "entity_type":  "Component",
          "name":         "Single Point of Failure (SPOF)",
          "severity":     "CRITICAL",
          "category":     "Availability",
          "description":  "NavLib is a directed cut vertex. Removing it partitions the dependency graph.",
          "recommendation": "Introduce redundancy: backup instances or alternative paths.",
          "evidence":     { "is_articulation_point": true, "availability_score": 0.58 }
        }
      ],
      "gnn": null
    }
  }
}
```

`blast_radius` is the count of nodes reachable from v in the DEPENDS_ON graph (computed via BFS on the Step 1 structural graph). `cascade_depth` is the length of the longest directed path from v. Both are populated by `predict_graph.py` using `compute_propagation_metrics()`.

---

## 8. Commands

All prediction is now managed through the unified `predict_graph.py` CLI.

```bash
# ─── Standard prediction ──────────────────────────────────────────────────────

# RMAV + anti-pattern scan on system layer (default)
PYTHONPATH=. python cli/predict_graph.py

# Multi-layer
PYTHONPATH=. python cli/predict_graph.py --layer app,system

# AHP-derived weights (recommended for thesis results)
PYTHONPATH=. python cli/predict_graph.py --use-ahp

# Equal-weight baseline (ablation condition)
PYTHONPATH=. python cli/predict_graph.py --equal-weights

# Custom AHP shrinkage factor
PYTHONPATH=. python cli/predict_graph.py --use-ahp --ahp-shrinkage 0.8

# ─── GNN inference ────────────────────────────────────────────────────────────

# First, train the GNN (requires Step 4 simulation results)
PYTHONPATH=. python cli/train_graph.py --layer system

# Multi-seed stability training
PYTHONPATH=. python cli/train_graph.py --layer system --seeds 42 123 456 789 2024

# Multi-graph inductive training (all domain scenarios)
PYTHONPATH=. python cli/train_graph.py --layer system --multi-scenario

# Ensemble prediction using a trained checkpoint
PYTHONPATH=. python cli/predict_graph.py --gnn-model output/gnn_checkpoints/best_model

# ─── Anti-pattern control ─────────────────────────────────────────────────────

# CRITICAL and HIGH patterns only (strict CI gate)
PYTHONPATH=. python cli/predict_graph.py --severity critical,high

# Specific patterns only
PYTHONPATH=. python cli/predict_graph.py --pattern SPOF,FAILURE_HUB,GOD_COMPONENT

# Skip anti-pattern detection (prediction scores only)
PYTHONPATH=. python cli/predict_graph.py --no-antipatterns

# Print full pattern catalog
PYTHONPATH=. python cli/predict_graph.py --catalog

# ─── CI/CD integration ────────────────────────────────────────────────────────
# Exit code 0 = clean, 1 = MEDIUM only, 2 = HIGH or CRITICAL → blocks deployment

PYTHONPATH=. python cli/predict_graph.py \
    --layer system \
    --use-ahp \
    --severity critical,high \
    --output-antipatterns results/antipatterns.json

# Pass antipattern report to visualize_graph.py
PYTHONPATH=. python cli/visualize_graph.py \
    --layers system \
    --antipatterns results/antipatterns.json \
    --output output/dashboard.html

# ─── Full results export ──────────────────────────────────────────────────────

PYTHONPATH=. python cli/predict_graph.py \
    --layer app,system \
    --use-ahp \
    --output results/prediction.json \
    --output-antipatterns results/antipatterns.json
```

---

## 9. What Comes Next

The Analyze stage produces Q(v) ∈ [0, 1] with a five-level RMAV classification, per-component blast-radius and cascade-depth metrics, and an architectural anti-pattern report. These are *topology-derived pre-deployment predictions*. The optional Predict stage refines them to Q_ens(v) after simulation labels become available.

Their empirical accuracy is quantified in the Validate stage (Step 5), which computes Spearman ρ, Kendall τ, F1-score, and specialist metrics by comparing Q(v) (or Q_ens(v)) against the simulation impact I(v) produced by the Simulate stage (Step 4). The pre-deployment independence guarantee — that Q(v) was produced from topology alone, with no access to I(v) — is what makes Step 5 a genuine empirical test rather than a consistency check.

→ [Step 4: Simulate](failure-simulation.md)