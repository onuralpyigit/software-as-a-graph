# Step 3: Prediction

**Produce criticality predictions for every component from topology alone — before any runtime data or simulation is available.**

← [Step 2: Analysis](structural-analysis.md) | → [Step 4: Simulation](failure-simulation.md)

---

## Table of Contents

1. [What This Step Does](#1-what-this-step-does)
2. [Two Prediction Paths](#2-two-prediction-paths)
3. [Rule-Based Prediction: RMAV](#3-rule-based-prediction-rmav)
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
4. [Learning-Based Prediction: GNN](#4-learning-based-prediction-gnn)
   - 4.1 [Motivation](#41-motivation)
   - 4.2 [Architecture Overview](#42-architecture-overview)
   - 4.3 [Graph Data Preparation](#43-graph-data-preparation)
   - 4.4 [HeteroGAT Model](#44-heterogat-model)
   - 4.5 [Multi-Task Prediction Heads](#45-multi-task-prediction-heads)
   - 4.6 [Edge Criticality Prediction](#46-edge-criticality-prediction)
   - 4.7 [Ensemble: GNN + RMAV](#47-ensemble-gnn--rmav)
   - 4.8 [Training Protocol](#48-training-protocol)
   - 4.9 [Multi-Seed Stability](#49-multi-seed-stability)
   - 4.10 [Methodological Integrity](#410-methodological-integrity)
5. [Comparing the Two Paths](#5-comparing-the-two-paths)
6. [Worked Example](#6-worked-example)
7. [Output Schema](#7-output-schema)
8. [Commands](#8-commands)
9. [What Comes Next](#9-what-comes-next)

---

## 1. What This Step Does

Prediction takes the metric vector **M(v)** produced by Step 2 for every component and produces a criticality prediction Q(v) ∈ [0, 1] along with a five-level classification, using topology alone.

```
M(v) from Step 2                     Prediction Engine                  Output
──────────────────────               ──────────────────────         ───────────────────────────
Tier 1 — 13 RMAV inputs:    →       Rule-Based (RMAV)       →      R(v), M(v), A(v), V(v)
  RPR, DG_in, MPCI, FOC              or                              Q_RMAV(v) ∈ [0, 1]
  BT, w_out, CC, path_complexity     Learning-Based (GNN)            Q_GNN(v)
  AP_c_dir, BR, CDI                  or                              Q_ens(v)
  REV, RCL, w_in                     Both (Ensemble)                 Level ∈ {CRITICAL, HIGH,
                                                                        MEDIUM, LOW, MINIMAL}
Tier 2 — 6 diagnostic:
  PR, CL, EV, pubsub_ratio,
  in_out_ratio, degree_product
```

**Prediction–simulation independence guarantee.** This step produces Q(v). Step 4 produces simulation ground-truth I(v). Step 5 measures the correlation between Q(v) and I(v). These pipelines must remain independent: M(v) must not be contaminated by simulation outputs, and I(v) must not read Q(v) as an input. This independence is the methodological foundation of the empirical validation claim.

**Validated performance.** The rule-based RMAV predictor achieves Spearman ρ = 0.876 overall (ρ = 0.943 at large scale, 150–300+ nodes) and F1-score > 0.89 across validated system scales. All results are from internal validation against simulation-derived I(v); external cross-system inductive validation remains an open item.

---

## 2. Two Prediction Paths

| | Rule-Based (RMAV) | Learning-Based (GNN) | Ensemble |
|---|---|---|---|
| **Mechanism** | AHP-weighted linear combination of Tier 1 metrics | Heterogeneous Graph Attention Network | Convex blend α·Q_GNN + (1−α)·Q_RMAV |
| **Interpretability** | Full — every score decomposes into metric contributions | Partial — attention weights and per-head outputs | Partial |
| **Requires training data** | No | Yes (simulation-labelled) | Yes |
| **Node criticality** | ✓ | ✓ | ✓ |
| **Edge criticality** | Proxies (BR, BT of endpoints) | ✓ Direct | ✓ Direct |
| **Topic-type branching** | ✓ (FOC formula) | Learned | Learned |
| **MPCI effect** | ✓ Explicit (CDPot_enh) | Learned | Learned |
| **Generalises to unseen systems** | Immediately | Requires fine-tuning | Requires fine-tuning |
| **Spearman ρ (validated)** | 0.876 | TBD (training on ATM dataset pending) | TBD |
| **Primary use** | First analysis; interpretable decision support | Post-training; system-of-systems scale | Production after ablation comparison |

Both paths classify components into the same five levels and are validated against the same I(v) ground truth, making their predictions directly comparable under the Step 5 protocol.

**Recommended workflow:**
1. Run RMAV — immediate results, no training required, full interpretability.
2. Run Step 4 Simulation to generate I(v) ground truth.
3. Train GNN on the labelled graph.
4. Run three-way ablation: compare ρ(Q_RMAV, I*), ρ(Q_GNN, I*), ρ(Q_ens, I*) using `--mode rmav|gnn|ensemble`.
5. Switch to Ensemble for production if Q_ens delivers predictive gain > 0.03 over RMAV alone.

---

## 3. Rule-Based Prediction: RMAV

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

**Standard formula** — Application, Broker, Infrastructure Node, Library:

```
R(v) = 0.45 × RPR(v)  +  0.30 × DG_in(v)  +  0.25 × CDPot_enh(v)
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| RPR(v) | 0.45 | Reverse PageRank — *global* transitive cascade reach across the full dependency graph |
| DG_in(v) | 0.30 | In-degree — count of direct dependents; *immediate* blast radius |
| CDPot_enh(v) | 0.25 | Enhanced Cascade Depth Potential — absorber nodes score high; MPCI amplifies depth when multi-path couplings exist (see [Section 3.3](#33-derived-terms)) |

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

> **Type dispatch.** The formula branch is resolved by `τ_V(v)` (the vertex type attribute on the graph node). `τ_V(v) = Topic` → Topic formula; all other node types → standard formula. This is justified because Topic failure semantics differ fundamentally from application/broker failure: a Topic failure is always a *simultaneous broadcast loss* to all subscribers, not a sequential cascade through the dependency graph.

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

A(v) measures whether a component is a structural single point of failure, weighted by operational priority.

```
A(v) = 0.35 × AP_c_directed(v)
     + 0.25 × QSPOF(v)
     + 0.25 × BR(v)
     + 0.10 × CDI(v)
     + 0.05 × w(v)
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| AP_c_directed(v) | 0.35 | Directed articulation point score — primary SPOF signal; computed on the directed DEPENDS_ON graph (see [Section 3.3](#33-derived-terms)) |
| QSPOF(v) | 0.25 | QoS-weighted SPOF severity — `AP_c_directed × w(v)`; high-priority SPOFs receive additional weight (see [Section 3.3](#33-derived-terms)) |
| BR(v) | 0.25 | Bridge Ratio — fraction of v's incident edges that are structural bridges; losing any bridge disconnects a subgraph |
| CDI(v) | 0.10 | Connectivity Degradation Index — average path-length increase upon v's removal; catches *soft SPOFs* that degrade service without fully disconnecting the graph |
| w(v) | 0.05 | Component QoS weight — direct contribution of operational priority to availability risk |

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

## 4. Learning-Based Prediction: GNN

### 4.1 Motivation

The RMAV rule-based predictor has two structural limitations:

**Fixed feature interactions.** RMAV combines metrics via fixed AHP weights determined before analysis. It cannot discover that, for a specific topology, the interaction between BT and RPR is more predictive than either metric alone.

**Node-only scoring.** Edges are scored only via endpoint proxies (BR, BT of endpoints). Direct edge-level criticality — identifying which specific pub-sub relationship is most dangerous to lose — requires edge-level supervision.

The GNN adds a **Step 3b** layer that learns from simulation ground truth I(v) and produces directly supervised edge criticality scores alongside node scores.

---

### 4.2 Architecture Overview

```
Step 3a: RMAV (rule-based)      Q_RMAV(v)   — AHP-weighted formula
Step 3b: GNN  (learning-based)  Q_GNN(v)    — trained on I(v) simulation ground truth
Step 3c: Ensemble               Q_ens(v)    = α · Q_GNN(v) + (1−α) · Q_RMAV(v)
```

Three cooperating modules:

```
    NetworkX DiGraph (Step 1 output)
             │
  ┌──────────▼───────────────────┐
  │   Data Preparation           │  Type-specific node features (23/18)
  │   networkx_to_hetero_data()  │   8-dim edge features per relation
  │   HeteroData + splits        │   5-dim simulation labels y = I*(v)
  └──────────┬───────────────────┘   5-dim RMAV scores  y_rmav = Q_RMAV(v)
             │
     ┌───────┼────────────┐
     ▼       ▼            ▼
  NodeGNN  EdgeGNN   EnsembleGNN
  3L 4H    (shares    α · Q_GNN
  HetGAT   NodeGNN   +(1-α)·Q_RMAV
  (N, 5)   backbone)  (N, 5)
           (E, 5)
```

All three modules are implemented in `backend/src/prediction/` and managed by `GNNService`.

---

### 4.3 Graph Data Preparation

`networkx_to_hetero_data()` in `data_preparation.py` converts the Step 1 NetworkX graph to a PyTorch Geometric `HeteroData` object, partitioning nodes and edges by type.

#### Node Feature Vector (23 or 18 dimensions)

Each node `v` is represented by a feature vector whose dimension depends on its type. Code-bearing types (**Application**, **Library**) have **23 dimensions** (Topological + Code Quality). Other types (**Broker**, **Topic**, **Node**) have **18 dimensions** (Topological only).

> **Implementation note.** The HeteroGAT architecture handles type-specific projections internally, so a global one-hot node-type vector is **not required** and has been removed to reduce parameter bloat. The constant `NODE_TYPE_TO_DIM` in `models.py` defines the authoritative widths.

**Topological metrics — indices 0–17 (18 features):**

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

**Code quality metrics — indices 18–22 (5 features; Applications/Libraries only):**

| Index | Metric | RMAV role |
|:-----:|--------|-----------|
| 18 | loc_norm | Diagnostic |
| 19 | complexity_norm | M(v) via CQP |
| 20 | instability_code | M(v) via CQP |
| 21 | lcom_norm | M(v) via CQP |
| 22 | code_quality_penalty (CQP) | M(v) directly |

#### Edge Features (8 dimensions)

| Index | Feature |
|:-----:|---------|
| 0 | QoS weight w(e) |
| 1–7 | Edge-type one-hot (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO, USES, DEPENDS_ON) |

**Node labels** (`data[type].y`, shape (n, 5)): simulation ground-truth per-dimension impact scores `[I*(v), IR(v), IM(v), IA(v), IV(v)]`.

**RMAV scores** (`data[type].y_rmav`, shape (n, 5)): RMAV quality scores `[Q(v), R(v), M(v), A(v), V(v)]`, stored for ensemble blending. These are *not* used as training labels; they are the ensemble right-hand side in Step 3c.

---

### 4.4 HeteroGAT Model

The model uses a **3-layer, 4-head Heterogeneous Graph Attention Network (HeteroGAT)** with separate weight matrices per relation type:

```
Layer 0 — Type-specific input projection:
  h_v^(0) = GELU( LayerNorm( W_{type(v)} · x_v + b_{type(v)} ) )

Layer k — Message passing per relation r ∈ edge_types:
  α_{uv}^r = softmax_u( a_r^T · [ h_u^(k) ‖ h_v^(k) ] )
  m_v^(r,k) = Σ_{u ∈ N_r(v)}  α_{uv}^r · W_r^(k) · h_u^(k)

  Aggregate across relation types:
  h_v^(k+1) = GELU( LayerNorm( Σ_r  W_{agg,r} · m_v^(r,k)  +  W_self · h_v^(k) ) )
```

Residual connections between layers. Hidden dimension D = 64. Dropout p = 0.2. All weight matrices are independent per relation type, ensuring the model learns distinct aggregation semantics for, e.g., PUBLISHES_TO vs DEPENDS_ON edges.

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
score(u, v) = MLP_E( h_u ‖ h_v ‖ e_{uv} )

e_{uv} ∈ ℝ^8: QoS weight + 7-bit edge-type one-hot
```

**Edge labels.** Training labels for edges are derived as:

```
I_edge(u, v) = max( I*(u), I*(v) )
```

> **Known limitation.** This labeling is a node-impact proxy, not direct edge criticality. It assigns a CRITICAL label to every edge connected to a CRITICAL node, regardless of whether the edge is a structural bridge or has redundant alternatives. True edge criticality should reflect the marginal cascade damage caused by *removing that specific edge*, approximated as `I*(u) × bridge_indicator(e)` for bridge edges and near-zero for edges with alternative paths. The current labeling causes the edge model to learn node centrality by proxy rather than genuine edge-level structural risk. This is a known approximation that should be tightened in future work.

---

### 4.7 Ensemble: GNN + RMAV

```
Q_ens(v) = α · Q_GNN(v)  +  (1−α) · Q_RMAV(v)

α ∈ ℝ^5  — per-dimension learnable blend coefficients, α_d = sigmoid(logit_d)
           Initialised at logit = 0  →  α = 0.5  (equal blend start)
```

The ensemble is a thin `EnsembleGNN` module containing only the 5-dimensional `alpha_logit` parameter. It is fine-tuned after the main GNN training loop using the training-node subset.

> **Implementation note.** `EnsembleGNN.forward()` requires `y_rmav` to be present in the HeteroData object. At inference time, if `rmav_scores` are not passed to `networkx_to_hetero_data()`, `y_rmav` will be absent and the ensemble silently falls back to GNN-only output while still labelling the result as `ensemble_scores`. The `GNNAnalysisResult` should expose a `prediction_mode` field (`"ensemble"` | `"gnn_only"`) so callers can distinguish these cases. This is an open implementation gap.

---

### 4.8 Training Protocol

**Transductive (single graph, default).** 60/20/20 train/val/test split applied per node type via `create_node_splits()`. Early stopping on validation Spearman ρ with patience = 30 epochs. Maximum 300 epochs.

**Loss function:**

```
L = L_composite  +  0.5 × L_dimension  +  0.3 × L_rank

L_composite   = MSE( Î*(v),   I*(v) )          — composite impact prediction
L_dimension   = Σ_d  MSE( d̂(v),  I_d*(v) )    — per-dimension impact prediction
L_rank        = −(1/N) Σ_v  log P(rank of v)   — ListMLE ranking loss
```

> **Note on L_dimension.** Each per-RMAV head is supervised against the simulation-derived per-dimension impact `I_d*(v)` (not against `Q_RMAV_d(v)` scores). This is the correct formulation: the GNN learns to predict simulation impact, not to reproduce the RMAV formula. The name `L_RMAV` used in some internal comments can be misleading; `L_dimension` is the correct term.

**Optimizer:** AdamW, lr = 3×10⁻⁴, weight_decay = 10⁻⁴, cosine annealing, gradient clipping max_norm = 1.0.

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
| G1 | `NODE_FEATURE_DIM` mismatch | High | **Resolved** | Enforced 23/18 split; added dimension check in `from_checkpoint()` |
| G2 | Edge labels as node proxies | Medium | Open | Current: `max(I*(u), I*(v))`. Future: direct removal impact |
| G3 | Redundant type encoding | Low | Open | Type-specific projections are used, but zero-padding remains in raw tensors |
| G4 | Transductive leakage | High | Open | Test nodes are present in MP graph. Requires pure inductive split for SoS claims |
| G5 | Best seed weight saving | High | **Resolved** | `best_state` is now restored before `train()` returns |
| G6 | Ensemble α bias | Medium | Open | α is a global blend; cannot correct local directional GNN errors |
| G7 | Predicton seed overwrite | High | **Resolved** | `predict_from_data()` now uses persisted `self._best_seed` for mask consistency |
| G8 | Ensemble validation gap | High | **Resolved** | `ensemble_metrics` now computed via `evaluate_scores` in prediction path |
| G9 | Silent ensemble fallback | Medium | **Resolved** | Added `prediction_mode` field and explicit fallback logging |
| G10 | Layer compatibility check | Medium | **Resolved** | `from_checkpoint()` validates `layer` alias against checkpoint metadata |

---

## 5. Comparing the Two Paths

| Property | RMAV (rule-based) | GNN (learning-based) | Ensemble |
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
| Primary use | First analysis; interpretable; CI gate | Post-training; SoS scale | Production after ablation |

**Three-way ablation protocol.** Before claiming that GNN or Ensemble outperforms RMAV, run the controlled ablation using `--mode rmav|gnn|ensemble` in `predict_graph.py`. Use Wilcoxon signed-rank test on per-component ρ values across seeds. Predictive gain threshold for switching: Δρ > 0.03 (one standard deviation on the ATM dataset).

---

## 6. Worked Example

**System from Step 2 worked example** (SensorApp, MonitorApp, MainBroker, NavLib, /temperature).

**Step 2 metric vector inputs:**

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
python bin/predict_graph.py

# Multi-layer
python bin/predict_graph.py --layer app,system

# AHP-derived weights (recommended for thesis results)
python bin/predict_graph.py --use-ahp

# Equal-weight baseline (ablation condition)
python bin/predict_graph.py --equal-weights

# Custom AHP shrinkage factor
python bin/predict_graph.py --use-ahp --ahp-shrinkage 0.8

# ─── GNN inference ────────────────────────────────────────────────────────────

# First, train the GNN (requires Step 4 simulation results)
python bin/train_graph.py --layer system

# Multi-seed stability training
python bin/train_graph.py --layer system --seeds 42 123 456 789 2024

# Multi-graph inductive training (all domain scenarios)
python bin/train_graph.py --layer system --multi-scenario

# Ensemble prediction using a trained checkpoint
python bin/predict_graph.py --gnn-model output/gnn_checkpoints/best_model

# ─── Anti-pattern control ─────────────────────────────────────────────────────

# CRITICAL and HIGH patterns only (strict CI gate)
python bin/predict_graph.py --severity critical,high

# Specific patterns only
python bin/predict_graph.py --pattern SPOF,FAILURE_HUB,GOD_COMPONENT

# Skip anti-pattern detection (prediction scores only)
python bin/predict_graph.py --no-antipatterns

# Print full pattern catalog
python bin/predict_graph.py --catalog

# ─── CI/CD integration ────────────────────────────────────────────────────────
# Exit code 0 = clean, 1 = MEDIUM only, 2 = HIGH or CRITICAL → blocks deployment

python bin/predict_graph.py \
    --layer system \
    --use-ahp \
    --severity critical,high \
    --output-antipatterns results/antipatterns.json

# Pass antipattern report to visualize_graph.py
python bin/visualize_graph.py \
    --layers system \
    --antipatterns results/antipatterns.json \
    --output output/dashboard.html

# ─── Full results export ──────────────────────────────────────────────────────

python bin/predict_graph.py \
    --layer app,system \
    --use-ahp \
    --output results/prediction.json \
    --output-antipatterns results/antipatterns.json
```

---

## 9. What Comes Next

Step 3 produces Q(v) ∈ [0, 1] with a five-level RMAV classification, per-component blast-radius and cascade-depth metrics, and an architectural anti-pattern report. These are *topology-derived pre-deployment predictions*.

Their empirical accuracy is quantified in Step 5 (Validation), which computes Spearman ρ, Kendall τ, F1-score, and specialist metrics (ICR@K, RCR, BCE, Predictive Gain) by comparing Q(v) against the simulation impact I(v) produced in Step 4. The pre-deployment independence guarantee — that Q(v) was produced from topology alone, with no access to I(v) — is what makes Step 5 a genuine empirical test rather than a consistency check.

→ [Step 4: Simulation](failure-simulation.md)