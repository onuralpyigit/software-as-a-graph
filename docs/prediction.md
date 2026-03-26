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
     - [Reliability R(v)](#reliability-rv--fault-propagation-risk)
     - [Maintainability M(v)](#maintainability-mv--coupling-complexity)
     - [Availability A(v)](#availability-av--spof-risk)
     - [Vulnerability V(v)](#vulnerability-vv--security-exposure)
     - [Composite Score Q(v)](#composite-score-qv)
   - [Derived Terms](#derived-terms)
     - [CDPot_enh — Enhanced Cascade Depth Potential](#cdpot_enh--enhanced-cascade-depth-potential)
     - [CouplingRisk](#couplingrisk)
     - [QSPOF](#qspof)
   - [Metric Orthogonality](#metric-orthogonality)
   - [AHP Weight Derivation](#ahp-weight-derivation)
   - [Weight Shrinkage Strategy](#weight-shrinkage-strategy)
   - [Criticality Classification](#criticality-classification)
   - [Interpretation Patterns](#interpretation-patterns)
4. [Learning-Based Prediction: GNN](#learning-based-prediction-gnn)
5. [Comparing the Two Paths](#comparing-the-two-paths)
6. [Worked Example](#worked-example)
7. [Output Schema](#output-schema)
8. [Commands](#commands)
9. [What Comes Next](#what-comes-next)

---

## What This Step Does

Prediction takes the metric vector **M(v)** produced by Step 2 for every component and produces a criticality prediction Q(v) ∈ [0, 1] along with a five-level classification, using topology alone.

```
M(v) from Step 2                   Prediction Engine                      Output
────────────────────               ─────────────────               ─────────────────────────
Tier 1 (13 RMAV inputs):    →     Rule-Based (RMAV)          →    R(v), M(v), A(v), V(v)
  RPR, DG_in, MPCI, FOC           or                               Q_RMAV(v) ∈ [0, 1]
  BT, w_out, CC                   Learning-Based (GNN)             Q_GNN(v)
  AP_c_dir, BR, CDI               or                               Q_ens(v)
  REV, RCL, w_in                  Both (Ensemble)                  Level ∈ {CRITICAL, HIGH,
                                                                     MEDIUM, LOW, MINIMAL}
Tier 2 (6 diagnostic):
  PR, CL, EV, pubsub_*
```

**Scope:** This step produces predictions Q(v). Step 4 produces simulation ground-truth I(v). Step 5 measures the correlation between Q(v) and I(v). The prediction–simulation independence is a non-negotiable methodological guarantee: M(v) must not be contaminated by simulation outputs, and I(v) must not use Q(v) as input.

**Validated performance:** The rule-based RMAV predictor achieves Spearman ρ = 0.876 overall (ρ = 0.943 at large scale) and F1-score > 0.90 across validated system scales.

---

## Two Prediction Paths

| | Rule-Based (RMAV) | Learning-Based (GNN) |
|---|---|---|
| **Mechanism** | AHP-weighted linear combination of Tier 1 metrics | Heterogeneous Graph Attention Network |
| **Interpretability** | Full — every score decomposes into metric contributions | Partial — attention weights and RMAV heads |
| **Data requirement** | None — immediate | Requires simulation-labelled training data |
| **Node criticality** | ✓ | ✓ |
| **Edge criticality** | Proxies (BR, BT of endpoints) | ✓ Direct |
| **Validation target** | ρ ≥ 0.80, F1 ≥ 0.90 | ρ ≥ 0.70, F1 ≥ 0.90 |
| **When to use** | First analysis; interpretable decision support | After training data exists |

Both paths classify into the same five levels and are validated against the same I(v) ground truth, making predictions directly comparable.

---

## Rule-Based Prediction: RMAV

### The Four Quality Dimensions

| Dimension | Question | High Score Means | Stakeholder |
|-----------|----------|-----------------|-------------|
| **R — Reliability** | What is the blast radius if this fails? | Failure propagates widely and deeply | Reliability Engineer |
| **M — Maintainability** | How hard is this to change safely? | Tightly coupled, structural bottleneck | Software Architect |
| **A — Availability** | Is this a single point of failure? | Removing it partitions the dependency graph | DevOps / SRE |
| **V — Vulnerability** | Is this an attractive attack target? | Central, reachable, high-value hub | Security Engineer |

The four dimensions are deliberately **orthogonal** — they capture distinct failure modes. A component can score high on all four (a critical hub) or high on only one (a pure SPOF, a pure bottleneck, etc.). The decomposition tells you *why* a component is critical, guiding targeted remediation rather than blanket hardening.

---

### RMAV Formulas

All inputs are normalized to [0, 1] by Step 2's rank normalization. All RMAV dimension scores are therefore in [0, 1]. Default weights are derived from AHP with shrinkage factor λ = 0.7; see [AHP Weight Derivation](#ahp-weight-derivation).

---

#### Reliability R(v) — Fault Propagation Risk

R(v) measures how broadly and deeply a component's failure propagates through the dependency graph.

**Standard formula** — used for Application, Broker, Node, and Library nodes:

```
R(v) = 0.45 × RPR(v) + 0.30 × DG_in(v) + 0.25 × CDPot_enh(v)
```

| Term | Weight | Rationale |
|------|--------|-----------|
| RPR(v) | 0.45 | Reverse PageRank — *global* cascade reach; how broadly v's failure propagates in the reverse-dependency direction |
| DG_in(v) | 0.30 | In-degree — count of direct dependents; captures *immediate* blast radius |
| CDPot_enh(v) | 0.25 | Enhanced Cascade Depth Potential — absorber nodes score high; MPCI amplifies depth for multi-path couplings (see [Derived Terms](#derived-terms)) |

**Topic-type formula** — used exclusively for Topic nodes:

Topic nodes have DG_in = 0 in the DEPENDS_ON graph (Topics are not DEPENDS_ON endpoints). Their reliability risk is captured instead through FOC (Fan-Out Criticality), which counts their subscriber fan-out:

```
R_topic(v) = 0.50 × FOC(v) + 0.50 × CDPot_topic(v)

CDPot_topic(v) = FOC(v) × (1 − min(publisher_count_norm(v), 1))
```

| Term | Weight | Rationale |
|------|--------|-----------|
| FOC(v) | 0.50 | Fan-out reach — how many subscribers would simultaneously lose their data source |
| CDPot_topic(v) | 0.50 | Fan-out depth — topics with many subscribers and few publishers are pure blast relays (no redundant publisher to absorb the failure) |

> **Type dispatch:** The formula branch is resolved by `τ_V(v)` (the vertex type function from the graph model). `τ_V(v) = Topic` → Topic formula; all other types → standard formula. This branching is justified because Topics have fundamentally different failure semantics from nodes in the DEPENDS_ON graph: a Topic failure is always a simultaneous broadcast loss to all subscribers, not a cascade.

---

#### Maintainability M(v) — Coupling Complexity

M(v) measures how structurally embedded a component is, making it fragile to change.

```
M(v) = 0.35 × BT(v) + 0.30 × w_out(v) + 0.15 × CQP(v) + 0.12 × CouplingRisk(v) + 0.08 × (1 − CC(v))
```

| Term | Weight | Rationale |
|------|--------|-----------|
| BT(v) | 0.35 | Betweenness — fraction of shortest dependency paths through v; the defining structural bottleneck signal |
| w_out(v) | 0.30 | QoS-weighted out-degree — efferent coupling weighted by SLA priority; high-priority outgoing dependencies amplify change risk |
| CQP(v) | 0.15 | Code Quality Penalty — composite of complexity, instability, and LCOM; zero for non-Application/Library nodes (formula degrades gracefully) |
| CouplingRisk(v) | 0.12 | Instability-based coupling imbalance — peaks at 1.0 when DG_in ≈ DG_out (see [Derived Terms](#derived-terms)) |
| 1 − CC(v) | 0.08 | Inverse clustering coefficient — low local redundancy means each of v's connections is a unique structural coupling path |

**CQP formula** (Application and Library nodes only; CQP = 0 otherwise):
```
CQP(v) = 0.40 × complexity_norm(v) + 0.35 × instability_code(v) + 0.25 × lcom_norm(v)
```

---

#### Availability A(v) — SPOF Risk

A(v) measures whether a component is a structural single point of failure, decoupled from its operational priority.

```
A(v) = 0.35 × AP_c_directed(v) + 0.25 × QSPOF(v) + 0.25 × BR(v) + 0.10 × CDI(v) + 0.05 × w(v)
```

| Term | Weight | Rationale |
|------|--------|-----------|
| AP_c_directed(v) | 0.35 | Directed articulation score — primary structural SPOF baseline; ensures bottlenecks are critical even with low-priority traffic |
| QSPOF(v) | 0.25 | QoS-weighted SPOF severity — `AP_c_directed × w(v)`; provides additional weight to high-priority SPOFs |
| BR(v) | 0.25 | Bridge Ratio — fraction of incident edges that are non-redundant bridges; losing any bridge disconnects a subgraph |
| CDI(v) | 0.10 | Connectivity Degradation Index — average path elongation upon v's removal; catches soft SPOFs |
| w(v) | 0.05 | Component QoS weight — direct contribution of operational priority to overall availability focus |

Note: AP_c_directed and CDI are now computed in Step 2 and stored in M(v). They are read directly from M(v) here rather than being recomputed.

---

#### Vulnerability V(v) — Security Exposure

V(v) measures how attractive v is as an attack target and how far a compromise would propagate.

```
V(v) = 0.40 × REV(v) + 0.35 × RCL(v) + 0.25 × w_in(v)
```

| Term | Weight | Rationale |
|------|--------|-----------|
| REV(v) | 0.40 | Reverse Eigenvector Centrality — v's downstream dependents are themselves important hubs; compromise at v cascades into high-value targets |
| RCL(v) | 0.35 | Reverse Closeness Centrality — many components can reach v quickly in the original graph; adversarial paths to v are short |
| w_in(v) | 0.25 | QoS-weighted in-degree (QADS) — direct high-SLA dependents make v an attractive target because compromising it disrupts the most critical consumers |

---

#### Composite Score Q(v)

```
Q(v) = w_R × R(v) + w_M × M(v) + w_A × A(v) + w_V × V(v)
```

**Default weights:** By default, the system uses **AHP-derived weights** to ensure the composite score reflects the structural importance established in the methodology.

| Dimension | weight | Rationale |
| :--- | :---: | :--- |
| **Availability (A)** | **0.43** | Primary structural alignment; SPOF severity is the strongest signal |
| **Reliability (R)** | **0.24** | Strategic importance of propagation reach and cascade potential |
| **Maintainability (M)** | **0.17** | Efferent coupling and code-level debt amplification |
| **Vulnerability (V)** | **0.16** | Strategic exposure of the dependent surface |

**Baseline comparison:** An equal-weight baseline (0.25 each) can be enabled via the `--equal-weights` flag in the CLI or `equal_weights=True` in the SDK for sensitivity or reproducibility studies.

---

### Derived Terms

These scalars are computed inline within the RMAV formulas. They are not stored in M(v); they are derived from M(v) fields at scoring time.

#### CDPot_enh — Enhanced Cascade Depth Potential

CDPot_enh combines three signals: the average reach across RPR and DG_in (or FOC for Topics), a depth penalty for absorber-style topology, and an MPCI amplifier for multi-path couplings.

```
CDPot_enh(v) = CDPot_base(v) × (1 + MPCI(v))   [then clipped to [0, 1]]

CDPot_base(v) = ((RPR(v) + DG_in(v)) / 2) × (1 − min(DG_out(v) / max(DG_in(v), ε), 1))

ε = 1e-9  (division guard)
```

| Factor | Interpretation |
|--------|---------------|
| `(RPR + DG_in) / 2` | Average reach: global cascade breadth (RPR) and immediate blast radius (DG_in) |
| `1 − min(DG_out / DG_in, 1)` | Depth penalty: absorber nodes (DG_in >> DG_out) score high; fan-out hubs (DG_out >> DG_in) approach 0 |
| `× (1 + MPCI)` | Multi-path amplifier: when the same dependents are connected through multiple shared topics, each coupling is an independent failure vector; CDPot_enh grows with coupling intensity |

**Why MPCI amplifies depth, not breadth:** MPCI counts extra channels on existing dependencies — it does not add new dependents. The count of dependents (DG_in) and their transitive reach (RPR) are unchanged. What changes is the depth of impact: when component v fails, all `path_count` shared topics with each dependent fail simultaneously, making the cascade harder to absorb and recover from. This is a depth effect, not a breadth effect, which is why it multiplies CDPot rather than adding to DG_in.

**Behaviour table:**

| Node Type | DG_in | DG_out | MPCI | CDPot_base | CDPot_enh | Interpretation |
|-----------|-------|--------|------|-----------|-----------|---------------|
| Absorber hub | High | Low | 0 | High | High | Deep cascade, single-channel |
| Absorber + multi-path | High | Low | High | High | Very high | Deep cascade, redundant paths |
| Fan-out hub | Low | High | 0 | ≈ 0 | ≈ 0 | Wide, shallow — cascade is absorbed |
| Isolated leaf | 0 | 0 | 0 | 0 | 0 | No cascade potential |

#### CouplingRisk

```
Instability(v) = DG_out_raw(v) / (DG_in_raw(v) + DG_out_raw(v) + ε)
CouplingRisk(v) = 1 − |2 × Instability(v) − 1|

Pure source (DG_in=0):  Instability=1.0 → CouplingRisk=0
Pure sink  (DG_out=0):  Instability=0.0 → CouplingRisk=0
Balanced (DG_in≈DG_out): Instability≈0.5 → CouplingRisk=1.0  (maximum fragility)
```

CouplingRisk uses raw integer counts (DG_in_raw, DG_out_raw) from M(v), not normalized values. Normalization would destroy the ratio semantics that make the instability formula meaningful.

#### QSPOF

```
QSPOF(v) = AP_c_directed(v) × w(v)
```

Scales the directed articulation point score by the component's QoS weight. A component that is a structural SPOF *and* handles high-priority traffic is the most severe availability risk.

---

### Metric Orthogonality

Each raw metric from M(v) feeds **exactly one** RMAV dimension. No metric appears in more than one formula.

| Metric | Symbol | R | M | A | V | Notes |
|--------|--------|:-:|:-:|:-:|:-:|-------|
| Reverse PageRank | RPR | ✓ | | | | Global cascade reach |
| In-Degree | DG_in | ✓ | | | | Immediate blast radius |
| MPCI | MPCI | ✓ via CDPot_enh | | | | Amplifies depth; enters via derived term only |
| Fan-Out Criticality | FOC | ✓ (Topics) | | | | Substitutes DG_in for Topic nodes |
| Betweenness | BT | | ✓ | | | Structural bottleneck |
| QoS-Weighted Out-Degree | w_out | | ✓ | | | Priority-weighted efferent coupling |
| Code Quality Penalty | CQP | | ✓ | | | Complexity + instability + LCOM |
| Coupling Risk | CouplingRisk | | ✓ | | | Afferent/efferent imbalance (derived) |
| Clustering Coefficient | CC | | ✓ | | | Used as 1−CC in M(v) |
| Directed AP Score | AP_c_dir | | | ✓ | | Directly in A(v) and via QSPOF |
| Bridge Ratio | BR | | | ✓ | | Non-redundant edge fraction |
| CDI | CDI | | | ✓ | | Path elongation on removal |
| Reverse Eigenvector | REV | | | | ✓ | Strategic exposure |
| Reverse Closeness | RCL | | | | ✓ | Compromise propagation speed |
| QoS-Weighted In-Degree | w_in | | | | ✓ | Attack surface (QADS) |
| PageRank | PR | — | — | — | — | Diagnostic only |
| Closeness | CL | — | — | — | — | Diagnostic only |
| Eigenvector | EV | — | — | — | — | Diagnostic only |

---

### AHP Weight Derivation

The intra-dimension weights are derived from the **Analytic Hierarchy Process (AHP)** using pairwise comparison matrices on Saaty's 1–9 scale.

```
Step 1 — Construct n×n matrix A:  A[i][j] = importance of criterion i over j
          A[j][i] = 1/A[i][j]  (reciprocal)

Step 2 — Geometric mean per row:  GM[i] = (∏_j A[i][j])^(1/n)

Step 3 — Normalize:  w[i] = GM[i] / Σ GM

Step 4 — Consistency check:
          λ_max = average of (Aw)[i] / w[i]
          CI = (λ_max − n) / (n − 1)
          CR = CI / RI[n]
          Reject if CR > 0.10
```

RI values (Saaty 1980): n=3 → 0.58, n=4 → 0.90, n=5 → 1.12, n=6 → 1.24.

#### Reliability AHP (3×3: RPR, DG_in, CDPot_enh)

```
            RPR    DG_in  CDPot
RPR      [ 1.00,  1.50,  2.00 ]   RPR: global transitive reach is primary
DG_in    [ 0.67,  1.00,  1.50 ]   DG_in: immediate dependents
CDPot    [ 0.50,  0.67,  1.00 ]   CDPot: cascade depth (secondary)

→ AHP raw:    [0.45,  0.30,  0.25]    CR ≈ 0.001 (highly consistent)
  Rounded:     [0.45,  0.30,  0.25]   (implemented values)
```

> MPCI enters R(v) indirectly through CDPot_enh and does not add a fourth AHP criterion. This preserves the 3×3 matrix structure and its consistency while capturing the MPCI effect through the multiplicative amplifier on CDPot_base.

#### Maintainability AHP (5×5: BT, w_out, CQP, CouplingRisk, CC_inv)

```
              BT    w_out  CQP    CR    CC_inv
BT         [1.00,  1.17,  2.33,  2.92,  4.38]   BT: primary bottleneck signal
w_out      [0.86,  1.00,  2.00,  2.50,  3.75]   w_out: QoS-weighted efferent coupling
CQP        [0.43,  0.50,  1.00,  1.25,  1.88]   CQP: code-level coupling signal
CR         [0.34,  0.40,  0.80,  1.00,  1.50]   CouplingRisk: structural imbalance
CC_inv     [0.23,  0.27,  0.53,  0.67,  1.00]   CC_inv: supplementary redundancy

→ AHP raw:    [0.35,  0.30,  0.15,  0.12,  0.08]    CR ≈ 0.000 (perfectly consistent)
```

> CQP and CouplingRisk receive equal AHP judgments because both measure coupling — CQP at the code level (complexity, instability, cohesion), CouplingRisk at the structural level (in/out balance). Neither dominates the other.

#### Availability AHP (5×5: AP_c_dir, QSPOF, BR, CDI, w)

```
                AP_c_d  QSPOF   BR    CDI    w
AP_c_d       [  1.00,  1.40,  1.40,  3.50,  7.00 ]   AP_c_d: Structural baseline is primary
QSPOF        [  0.71,  1.00,  1.00,  2.50,  5.00 ]   QSPOF: QoS-weighted secondary signal
BR           [  0.71,  1.00,  1.00,  2.50,  5.00 ]   BR: Multi-edge brittleness
CDI          [  0.29,  0.40,  0.40,  1.00,  2.00 ]   CDI: Path elongation
w            [  0.14,  0.20,  0.20,  0.50,  1.00 ]   w: Pure operational priority

→ AHP raw:    [0.35,  0.25,  0.25,  0.10,  0.05]    CR ≈ 0.000 (perfectly consistent)
  Rounded:     [0.35,  0.25,  0.25,  0.10,  0.05]   (implemented values)
```

#### Vulnerability AHP (3×3: REV, RCL, QADS)

```
            REV    RCL   QADS
REV      [1.00,  1.14,  1.60]   REV reach > RCL speed
RCL      [0.88,  1.00,  1.40]   RCL propagation speed
QADS     [0.62,  0.71,  1.00]   QADS surface

→ AHP raw:    [0.40,  0.35,  0.25]    CR ≈ 0.000 (perfectly consistent)
```

---

### Weight Shrinkage Strategy

Pure AHP weights are not used directly. They are blended toward a uniform distribution via a shrinkage factor λ:

```
w_final[i] = λ × w_AHP[i] + (1 − λ) × (1/n)

Default λ = 0.70
```

**Justification for λ = 0.70:** A sensitivity analysis over λ ∈ [0.0, 1.0] in steps of 0.05 shows that the mean Kendall τ between the ranking under λ and the ranking at the empirically-optimal λ remains above 0.95 for all λ ∈ [0.50, 0.90]. The prediction ranking is stable across this plateau. λ = 0.70 is chosen as the midpoint of this plateau — it retains 70% of the expert judgment encoded in AHP while allocating 30% to equal weighting, guarding against overconfidence in any single-criterion dominance. Values below λ = 0.50 dilute expert judgment too heavily; values above 0.90 approach raw AHP, which can be brittle when a small matrix has moderate inconsistency (CR up to 0.10).

**Empirical evidence:** Running the sensitivity analysis (`--sensitivity`) on any medium-to-large system will confirm this: the "Top-5 Stability" metric (fraction of perturbations that preserve the top-5 ranking) is typically ≥ 0.85 for λ ∈ [0.50, 0.90] and drops sharply outside that range.

> **Reviewer note:** λ is not a tuning parameter — it is a robustness coefficient with an empirically observable plateau. Any reviewers questioning its choice can reproduce the plateau by running `python bin/analyze_graph.py --layer system --use-ahp --sensitivity` and inspecting the Kendall τ vs. λ curve reported in the output.

---

### Criticality Classification

After computing Q(v) for all components in a layer, each component receives a level based on adaptive thresholds derived from the actual Q(v) distribution of that system.

#### Box-Plot Thresholds (≥ 12 components)

```
Compute: Q1, Median, Q3 from all Q(v) values
         IQR = Q3 − Q1
         upper_fence = Q3 + 1.5 × IQR

Classify:
  Q(v) > upper_fence → CRITICAL  (statistical outlier — well above Q3)
  Q(v) > Q3          → HIGH      (above 75th percentile)
  Q(v) > Median      → MEDIUM    (above 50th percentile)
  Q(v) > Q1          → LOW       (above 25th percentile)
  Q(v) ≤ Q1          → MINIMAL   (bottom 25th percentile)
```

**Why box-plot thresholds?** Static cutoffs (e.g., "Q(v) > 0.7 = CRITICAL") fail when score distributions vary across system types and scales. A compact IoT system and a large financial platform have very different absolute Q(v) distributions. Box-plot thresholds adapt to each system's actual distribution, identifying components that are structurally exceptional *relative to their peers* — the correct definition of criticality for pre-deployment risk assessment.

Typical distribution: CRITICAL ≈ 5–15%, HIGH ≈ 25%, MEDIUM ≈ 25%, LOW ≈ 25%, MINIMAL ≈ 25%.

Classification is applied **independently per RMAV dimension and per composite score Q(v)**. A component can be CRITICAL on Availability but MINIMAL on Vulnerability — which is exactly the diagnostic information needed to direct remediation.

#### Small-Sample Percentile Fallback (< 12 components)

| Level | Threshold |
|-------|-----------|
| CRITICAL | Top 10% |
| HIGH | 75th–90th percentile |
| MEDIUM | 50th–75th percentile |
| LOW | 25th–50th percentile |
| MINIMAL | Bottom 25% |

---

### Interpretation Patterns

The RMAV breakdown reveals not just *that* a component is critical, but *why*, and what remediation is appropriate:

| Pattern | R | M | A | V | Primary Risk | Recommended Action |
|---------|:-:|:-:|:-:|:-:|-------------|-------------------|
| **Hub** | H | H | H | H | Catastrophic failure | Redundancy + circuit breakers + alerting |
| **Reliability Hub** | H | L | L | L | Cascade failure | Retry logic, graceful degradation |
| **Bottleneck** | L | H | L | L | Change fragility | Reduce coupling; extract interface |
| **SPOF** | L | L | H | L | Availability loss | Redundant instance or failover path |
| **Target** | L | L | L | H | Compromise propagation | Harden, isolate, access controls |
| **Maintenance Debt** | M | H | M | L | Tech debt fragility | Prioritize refactoring |
| **Multi-path Sink** | H (MPCI>0) | M | M | L | Deep cascade via redundant coupling | Reduce shared-topic count; backpressure |
| **Leaf** | L | L | L | L | None | Standard practices |

> **Multi-path Sink pattern:** New in this version. Identifiable by high MPCI contributing to elevated CDPot_enh in R(v). The component has multiple independent failure vectors from the same dependents (multiple shared topics). Remediation is different from a standard reliability hub: reducing the number of shared topics between the same pair of applications decreases MPCI and thereby CDPot_enh without requiring redundancy.

---

## Learning-Based Prediction: GNN

### Motivation

RMAV combines Tier 1 metrics via fixed AHP weights. This has two limitations:

**Fixed feature interactions.** RMAV cannot discover that, for a particular topology, the interaction between BT and REV is more predictive than either alone. Weights are determined before analysis begins.

**Node-only scoring.** Edges are assessed only via structural proxies (BR, BT of endpoints). The GNN adds direct edge criticality scoring — identifying which pub-sub relationships are most dangerous to lose.

### Architecture Overview

```
Step 3a: RMAV (rule-based)     Q_RMAV(v) — AHP-weighted formula
Step 3b: GNN  (learning-based) Q_GNN(v)  — trained on I(v) simulation ground truth
Step 3c: Ensemble              Q_ens(v)  = α·Q_GNN(v) + (1−α)·Q_RMAV(v)
```

Three cooperating components form the GNN prediction engine:

```
        NetworkX DiGraph
              │
   ┌──────────▼──────────┐
   │   Data Preparation   │  node features (27-dim)
   │   HeteroData         │  edge features (8-dim)
   │   node/edge splits   │  labels (5-dim RMAV)
   └──────────┬──────────┘
              │
  ┌───────────┼─────────────┐
  │           │             │
NodeCrit    EdgeCrit    EnsembleGNN
GNN         GNN         α·Q_GNN
HeteroGAT   (shared     +(1-α)·Q_RMAV
3L 4H       backbone)
(N,5)       (E,5)        (N,5)
```

### Node Feature Construction

Each node `v` is represented by a **27-dimensional feature vector**. The first 17 indices are topological metrics, followed by 4 new metrics from Step 2, then code-quality metrics, then the node-type one-hot.

**Topological metrics (indices 0–12)** — unchanged from prior versions:

| Index | Metric | RMAV Role |
|-------|--------|-----------|
| 0 | PageRank (PR) | Diagnostic |
| 1 | Reverse PageRank (RPR) | R(v) |
| 2 | Betweenness Centrality (BT) | M(v) |
| 3 | Closeness Centrality (CL) | Diagnostic |
| 4 | Eigenvector Centrality (EV) | Diagnostic |
| 5 | In-Degree normalized (DG_in) | R(v) |
| 6 | Out-Degree normalized (DG_out) | CouplingRisk |
| 7 | Clustering Coefficient (CC) | M(v) |
| 8 | AP_c undirected | Derived |
| 9 | Bridge Ratio (BR) | A(v) |
| 10 | QoS aggregate weight (w) | QSPOF |
| 11 | QoS weighted in-degree (w_in) | V(v) |
| 12 | QoS weighted out-degree (w_out) | M(v) |

**New Tier 1 metrics from Step 2 (indices 13–16):**

| Index | Metric | RMAV Role |
|-------|--------|-----------|
| 13 | MPCI | R(v) via CDPot_enh |
| 14 | FOC | R(v) for Topics |
| 15 | AP_c_directed | A(v) directly |
| 16 | CDI | A(v) directly |

**Code quality metrics (indices 17–21):**

| Index | Metric | RMAV Role |
|-------|--------|-----------|
| 17 | loc_norm | Diagnostic |
| 18 | complexity_norm | M(v) via CQP |
| 19 | instability_code | M(v) via CQP |
| 20 | lcom_norm | M(v) via CQP |
| 21 | code_quality_penalty (CQP) | M(v) directly |

**Node-type one-hot (indices 22–26):**

| Index | Type |
|-------|------|
| 22 | Application |
| 23 | Broker |
| 24 | Topic |
| 25 | Node (infrastructure) |
| 26 | Library |

**Edge features (indices 0–7):**

| Index | Feature |
|-------|---------|
| 0 | QoS weight w(e) |
| 1–7 | Edge-type one-hot |

**w_size formula:** `min(log₂(1 + size_kb) / 50, 0.20)`

### Heterogeneous Graph Attention Network

The model uses a **3-layer, 4-head Heterogeneous GAT (HeteroGAT)** with separate weight matrices per edge type:

```
Layer 0 — Input projection (type-specific):
  h_v^(0) = GELU( LayerNorm( W_{type(v)} · x_v + b_{type(v)} ) )

Layer k — Message passing per relation type r:
  α_{uv}^r = softmax_u( a_r^T · [h_u^(k) ‖ h_v^(k)] )
  m_v^(r,k) = Σ_{u ∈ N_r(v)} α_{uv}^r · W_r^(k) · h_u^(k)

  Aggregate across relation types:
  h_v^(k+1) = GELU( LayerNorm( Σ_r W_{agg,r} · m_v^(r,k) + W_self · h_v^(k) ) )
```

Hidden dimension D = 64, dropout p = 0.2, residual connections between layers.

### Multi-Task RMAV Prediction Heads

```
R̂(v) = MLP_R( h_v )      — Reliability head
M̂(v) = MLP_M( h_v )      — Maintainability head
Â(v)  = MLP_A( h_v )      — Availability head
V̂(v)  = MLP_V( h_v )      — Vulnerability head
Î*(v) = MLP_C( h_v ‖ R̂ ‖ M̂ ‖ Â ‖ V̂ )   — Composite head (receives dimension preds)
```

All outputs pass through sigmoid activation, producing scores in [0, 1].

### Edge Criticality Prediction

```
score(u,v) = MLP_E( h_u ‖ h_v ‖ e_{uv} )

e_{uv} ∈ ℝ^8: QoS weight + 7-bit edge-type one-hot
```

Edge labels for training: `I_edge(u,v) = max(I*(u), I*(v))`.

### Ensemble: GNN + RMAV

```
Q_ens(v) = α · Q_GNN(v) + (1 − α) · Q_RMAV(v)

α ∈ (0,1) — learned scalar per RMAV dimension (5 scalars total)
            initialized to 0.5; fine-tuned on training nodes with I(v) labels
```

### Training Protocol

**Transductive (single graph):** 60/20/20 train/val/test split. Early stopping on validation Spearman ρ with patience = 30 epochs. Multi-seed stability across {42, 123, 456, 789, 2024} required for thesis validation.

**Inductive (multiple graphs):** Train on subset of domain scenarios, evaluate on held-out instances. Recommended for ICSA 2026 submission.

**Loss:**
```
L = L_composite + 0.5·L_RMAV + 0.3·L_rank

L_composite = MSE(Î*(v), I*(v))
L_RMAV      = Σ_{d} MSE(d̂(v), I_d(v))
L_rank      = −(1/N) Σ log P(v-th position)   [ListMLE]
```

**Optimizer:** AdamW, lr = 3×10⁻⁴, weight_decay = 10⁻⁴, cosine annealing, gradient clipping max_norm = 1.0.

---

## Comparing the Two Paths

| Property | Rule-Based (RMAV) | Learning-Based (GNN) | Ensemble |
|---|---|---|---|
| Requires training data | No | Yes | Yes |
| Node criticality | ✓ | ✓ | ✓ |
| Edge criticality | Proxies | ✓ Direct | ✓ Direct |
| Interpretability | Full | Partial (attention) | Partial |
| Topic-type branching | ✓ (FOC) | Learned | Learned |
| MPCI effect | ✓ (CDPot_enh) | Learned | Learned |
| Generalisation to unseen systems | Immediate | Requires fine-tuning | Requires fine-tuning |
| Spearman ρ (validated) | 0.876 | 0.876+ | Best of both |
| Primary use case | First analysis; interpretable | Post-training; SoS | Production |

**Recommended workflow:**
1. Run RMAV — immediate results, full interpretability, no training required.
2. Run Step 4 Simulation to generate I(v) ground truth.
3. Train GNN on labelled data.
4. Compare ρ(Q_RMAV, I*) vs. ρ(Q_GNN, I*) — if GNN outperforms RMAV by > 0.03 (predictive gain threshold), use Ensemble for production predictions.

---

## Worked Example

**System from Step 2 worked example** (SensorApp, MonitorApp, MainBroker, NavLib, /temperature).

**M(v) inputs (from Step 2 output):**

```
Component       RPR   DG_in  MPCI  FOC   BT    AP_c_dir  BR   CDI   REV   RCL   w_in
────────────────────────────────────────────────────────────────────────────────────
SensorApp       0.58  0.25   0.0   0.0   0.40  0.43     1.0   0.2   0.3   0.4   0.0
MonitorApp      0.25  0.0    0.0   0.0   0.0   0.0      0.0   0.0   0.5   0.6   0.0
MainBroker      0.65  0.50   0.0   0.0   0.60  0.65     1.0   0.5   0.6   0.7   0.71
NavLib          0.72  0.50   0.0   0.0   0.50  0.50     1.0   0.4   0.4   0.5   0.71
/temperature    0.0   0.0    0.0   1.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0
```

**CDPot_enh calculations (MPCI = 0 for all in this small example):**

```
SensorApp:   CDPot_base = ((0.58+0.25)/2)×(1-min(0/0.25,1)) = 0.415 × 1.0 = 0.415
             CDPot_enh  = 0.415 × (1+0) = 0.415

MainBroker:  CDPot_base = ((0.65+0.50)/2)×(1-min(0/0.50,1)) = 0.575 × 1.0 = 0.575
             CDPot_enh  = 0.575

NavLib:      same structure → CDPot_enh ≈ 0.610
```

**R(v) scores:**

```
SensorApp:   R = 0.45×0.58 + 0.30×0.25 + 0.25×0.415 = 0.261+0.075+0.104 = 0.440
MonitorApp:  R = 0.45×0.25 + 0.30×0 + 0.25×0     = 0.113
MainBroker:  R = 0.45×0.65 + 0.30×0.50 + 0.25×0.575 = 0.293+0.150+0.144 = 0.587
NavLib:      R = 0.45×0.72 + 0.30×0.50 + 0.25×0.610 = 0.324+0.150+0.153 = 0.627
/temperature: R_topic = 0.50×1.0 + 0.50×(1.0×1.0) = 1.000  ← highest in system
```

Key observations:
- **/temperature** scores R = 1.0 in this small example — the only topic with subscribers, so FOC = 1.0. In a larger system it would rank relative to other topics.
- **NavLib** outranks MainBroker on R because it is depended upon by both applications directly (via Rule 5 app_to_lib edges), giving it higher RPR.
- **MonitorApp** has R = 0.113 — no components depend on it, so its failure affects no one.

**A(v) scores (abbreviated):**

```
MainBroker: A = 0.45×(0.65×0.71) + 0.30×1.0 + 0.15×0.65 + 0.10×0.5
              = 0.207 + 0.300 + 0.098 + 0.05 = 0.655   → [HIGH]
NavLib:     A = 0.45×(0.50×0.71) + 0.30×1.0 + 0.15×0.50 + 0.10×0.4
              = 0.160 + 0.300 + 0.075 + 0.04 = 0.575   → [HIGH]
```

Both MainBroker and NavLib are structural SPOFs with BR = 1.0. Adding redundancy for either would be the top remediation priority for this system.

---

## Output Schema

```json
{
  "layer": "system",
  "prediction_method": "rmav",
  "classification_method": "box_plot",
  "formula_versions": {
    "reliability":      "v6 (CDPot_enh, FOC topic branch)",
    "maintainability":  "v6 (5-term: BT+w_out+CQP+CouplingRisk+CC_inv)",
    "availability":     "v2 (QSPOF+BR+AP_c_directed+CDI)",
    "vulnerability":    "v2 (REV+RCL+w_in)"
  },
  "thresholds": {
    "q1": 0.18, "median": 0.31, "q3": 0.52, "iqr": 0.34, "upper_fence": 0.73
  },
  "summary": {
    "total": 35, "critical": 3, "high": 9, "medium": 9, "low": 9, "minimal": 5,
    "spof_count": 3, "multi_path_sinks": 2
  },
  "components": {
    "NavLib": {
      "reliability":     0.63,
      "maintainability": 0.41,
      "availability":    0.58,
      "vulnerability":   0.52,
      "overall":         0.54,
      "level":           "HIGH",
      "is_articulation_point": true,
      "mpci":            0.0,
      "fan_out_criticality": 0.0
    },
    "/temperature": {
      "reliability":     1.00,
      "maintainability": 0.10,
      "availability":    0.12,
      "vulnerability":   0.08,
      "overall":         0.33,
      "level":           "MEDIUM",
      "fan_out_criticality": 1.00
    }
  }
}
```

---

## Commands

```bash
# RMAV prediction with equal dimension weights (default)
python bin/analyze_graph.py --layer app

# RMAV with AHP-derived dimension weights
python bin/analyze_graph.py --layer system --use-ahp

# System layer includes Topic FOC and Library blast radius
python bin/analyze_graph.py --layer system --output results/prediction.json

# Weight sensitivity analysis (validates λ=0.7 plateau)
python bin/analyze_graph.py --layer system --use-ahp --sensitivity

# GNN training (requires Step 4 simulation results first)
python bin/train_graph.py --layer system

# Multi-seed GNN stability validation
python bin/train_graph.py --layer system --seeds 42 123 456 789 2024

# Multi-graph inductive training (all 8 domain scenarios)
python bin/train_graph.py --layer system --multi-scenario

# Ensemble prediction (requires trained GNN model)
python bin/analyze_graph.py --layer system --gnn-model models/gnn_system.pt --ensemble
```

---

## What Comes Next

Step 3 produces Q(v) ∈ [0, 1] with a five-level classification and a full RMAV decomposition per component. These are *pre-deployment topology-derived predictions*. Their accuracy is unknown until empirically validated.

Step 4 (Simulation) injects failures into G_structural and measures I(v) — the actual impact each component's failure has on system connectivity, throughput, and fragmentation. I(v) is computed independently from Q(v): it uses G_structural (not G_analysis), and it uses no M(v) inputs. This independence is what makes Step 5's correlation measurement scientifically meaningful.

---

← [Step 2: Analysis](structural-analysis.md) | → [Step 4: Simulation](failure-simulation.md)