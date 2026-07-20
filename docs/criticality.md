# Component and Relationship Criticality

**Define what "criticality" means for a node and for an edge in the dependency graph, and how the project's structural/learned scores relate to stakeholder-facing software quality.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Component (Node) Criticality](#2-component-node-criticality)
   - 2.1 [Definition](#21-definition)
   - 2.2 [The RMAV Model](#22-the-rmav-model)
   - 2.3 [Criticality Classification](#23-criticality-classification)
3. [Relationship (Edge) Criticality](#3-relationship-edge-criticality)
   - 3.1 [Definition](#31-definition)
   - 3.2 [Structural Edge Signals](#32-structural-edge-signals)
   - 3.3 [Learned Edge Scoring (GNN)](#33-learned-edge-scoring-gnn)
   - 3.4 [Ranking Critical Edges](#34-ranking-critical-edges)
4. [Quality-in-Use Framing (ISO/IEC 25010 SQuaRE)](#4-quality-in-use-framing-isoiec-25010-square)
   - 4.1 [The Five Quality-in-Use Characteristics](#41-the-five-quality-in-use-characteristics)
   - 4.2 [Mapping RMAV to Quality-in-Use](#42-mapping-rmav-to-quality-in-use)
   - 4.3 [Proxy, Not Ground Truth](#43-proxy-not-ground-truth)
5. [Where This Fits in the Pipeline](#5-where-this-fits-in-the-pipeline)
6. [References](#6-references)

---

## 1. Overview

Every entity in the graph — a component (node) or a dependency (edge) — carries a **criticality** signal: a score describing how much damage its failure, degradation, or compromise would do to the rest of the system. This document is the conceptual home for that concept. It does not re-derive formulas that already live in [structural-analysis.md](structural-analysis.md) and [prediction.md](prediction.md); it defines the terms, shows where each signal is computed, and explains the theoretical grounding (ISO/IEC 25010 SQuaRE Quality-in-Use) that motivates *why* these particular structural signals are treated as criticality.

Two distinct but related concepts are in scope:

| Concept | Applies to | Primary output |
|:---|:---|:---|
| **Component criticality** | Nodes ($v \in V$: Application, Broker, Topic, Node, Library) | RMAV scores $R(v), M(v), A(v), V(v), Q(v)$ + five-tier classification |
| **Relationship criticality** | Edges ($e \in E$: physical pub-sub links and derived `DEPENDS_ON` edges) | Structural bridge/betweenness signals + GNN edge score $Q_{\text{GNN}}(u,v)$ |

---

## 2. Component (Node) Criticality

### 2.1 Definition

> **Component criticality** is the degree to which a component's failure, degradation, or compromise threatens the health of the rest of the system — measured as the breadth and depth of failure propagation, the difficulty of changing the component safely, whether it is a structural single point of failure, and how attractive a target it is for attack.

Criticality is computed, not asserted: it is derived entirely from a component's position in $G_{\text{analysis}}(l)$ (the layer-projected dependency graph produced by [graph-model.md](graph-model.md)), never from manual tagging.

### 2.2 The RMAV Model

Component criticality is decomposed into four orthogonal dimensions — **Reliability, Maintainability, Availability, Vulnerability (RMAV)** — combined into a composite score $Q(v)$. The full formulas, weights, and derivations are defined in [structural-analysis.md §11](structural-analysis.md#11-analyze-stage--rule-based-rmav-scoring); the summary:

| Dimension | Question Answered | Driven Primarily By |
|:---|:---|:---|
| **R — Reliability** | How broadly/deeply does failure propagate? | Reverse PageRank, in-degree, Cascade Depth Potential |
| **M — Maintainability** | How hard is this to change safely? | Betweenness, efferent coupling, Code Quality Penalty |
| **A — Availability** | Is this a structural single point of failure? | Directed articulation point score, bridge ratio, QoS-SPOF |
| **V — Vulnerability** | How attractive a target is this for attack? | Reverse eigenvector/closeness centrality, QoS-weighted in-degree |

$$
Q(v) = 0.43 \cdot A(v) + 0.24 \cdot R(v) + 0.17 \cdot M(v) + 0.16 \cdot V(v)
$$

Availability is weighted highest because structural SPOF failure partitions the graph *deterministically*, whereas cascade propagation (R), coupling risk (M), and attack exposure (V) are *probabilistic*. See [saag/core/criticality.py](../saag/core/criticality.py) for the `CriticalityRanking` DTO that carries these scores through the pipeline.

### 2.3 Criticality Classification

Raw $Q(v)$ scores are mapped onto five tiers using **adaptive box-plot thresholding**, relative to the system's own score distribution rather than fixed cutoffs — full definition in [structural-analysis.md §11.7](structural-analysis.md#117-criticality-classification):

```
CRITICAL  :  score > Q3 + 1.5 × IQR
HIGH      :  Q3 < score ≤ upper fence
MEDIUM    :  median < score ≤ Q3
LOW       :  Q1 < score ≤ median
MINIMAL   :  score ≤ Q1
```

Implemented by `CriticalityLevel` and `BoxPlotStats` in [saag/core/criticality.py](../saag/core/criticality.py). Classification is applied independently per RMAV dimension and for the composite — a component can be CRITICAL on Availability while MINIMAL on Vulnerability, which is the diagnostic signal that directs remediation.

---

## 3. Relationship (Edge) Criticality

### 3.1 Definition

> **Relationship criticality** is the degree to which a single dependency edge — a specific pub-sub linkage or derived `DEPENDS_ON` relationship — is non-redundant and load-bearing for system connectivity: how much of the system's reachability, path efficiency, or failure containment depends on that one edge existing.

Where component criticality asks "how dangerous is losing this node," relationship criticality asks "how dangerous is losing this specific *link*, independent of the endpoints' own criticality." A high-criticality node can still have many low-criticality (redundant) edges; a low-criticality node can sit at the far end of a single, highly critical bridge edge.

### 3.2 Structural Edge Signals

Relationship criticality is assembled from per-edge structural signals computed in [saag/analysis/structural_analyzer.py](../saag/analysis/structural_analyzer.py) and carried by `EdgeMetrics` / `EdgeQuality` in [saag/core/metrics.py](../saag/core/metrics.py):

- **`is_bridge`** — whether the edge is a graph bridge (cut-edge): `nx.bridges()` over the undirected projection. Removing a bridge disconnects a subgraph from the rest of the system.
- **`betweenness`** — edge betweenness centrality (`nx.edge_betweenness_centrality`, QoS-weighted): the fraction of shortest dependency paths that traverse this specific edge.
- **`weight`** — the edge's QoS-derived weight from [graph-model.md](graph-model.md).

These are distinct from two *node-level* metrics that are easy to mistake for edge scores because they are edge-derived:

- **Bridge Ratio `BR(v)`** ([structural-analysis.md §9.9](structural-analysis.md#99-bridge-ratio-br)) — the *fraction of a node's own connections* that are bridges. It describes a node's exposure to non-redundant edges, not a per-edge score.
- **Multi-Path Coupling Index `MPCI(v)`** ([structural-analysis.md §9.3](structural-analysis.md#93-multi-path-coupling-index-mpci)) — counts *redundant* shared channels feeding into a node. High MPCI means a node's incoming edges are collectively low-criticality (multi-channel, no single edge is a SPOF); low MPCI (with high `DG_in`) means each incoming edge is closer to a single point of failure for that dependency.

### 3.3 Learned Edge Scoring (GNN)

The Predict stage's GNN produces a direct, per-edge criticality prediction rather than relying on endpoint-node proxies — see [prediction.md §2.6](prediction.md#26-edge-criticality-prediction) and [design/SDD.md §6.26](design/SDD.md) for the full architecture:

```
score(u, v) = TypedEdgeEncoder_r( h_u, h_v, e_uv )

e_uv ∈ ℝ^16: QoS weight + path_count_norm + 7-bit edge-type one-hot + 7-bit QoS features
```

Training labels are derived from simulated failure impact with a **bridge-aware multiplier**, so an edge inherits its source's blast radius only when it is structurally non-redundant:

```
I_edge(u, v) = I*(u) × bridge_multiplier

bridge_multiplier = 1.0   if (u, v) is a structural bridge
                   = 0.1   otherwise
```

This is the formal statement of relationship criticality used for training/validation: an edge is critical in proportion to *both* what it connects to (`I*(u)`) *and* whether it is replaceable (`bridge_multiplier`).

### 3.4 Ranking Critical Edges

Edges are ranked for reporting/UI consumption via `get_critical_edges()` in [saag/analysis/service.py](../saag/analysis/service.py) and exposed through [api/routers/components.py](../api/routers/components.py), sorting by `EdgeQuality.scores.overall` (the same RMAV-style composite machinery used for nodes, applied edge-wise).

---

## 4. Quality-in-Use Framing (ISO/IEC 25010 SQuaRE)

### 4.1 The Five Quality-in-Use Characteristics

ISO/IEC 25010 (SQuaRE) defines **Quality-in-Use** as quality measured from the outcome experienced by stakeholders operating a system-of-interest in a specific context of use — as distinct from the structural/product-quality characteristics (reliability, maintainability, etc. as internal attributes). It has five characteristics:

| Characteristic | Sub-characteristics | Meaning |
|:---|:---|:---|
| **Effectiveness** | — | Accuracy/completeness with which stakeholders achieve their goals |
| **Efficiency** | — | Resources expended relative to effectiveness achieved |
| **Satisfaction** | Usefulness, Trust, Pleasure, Comfort | Stakeholder response to using the system |
| **Freedom from risk** | Economic, Health & safety, Environmental risk mitigation | Degree to which the system limits risk of harm |
| **Context coverage** | Context completeness, Flexibility | Degree to which quality-in-use is sustained across all intended contexts |

### 4.2 Mapping RMAV to Quality-in-Use

RMAV and edge criticality are **structural proxies** for quality-in-use loss under failure — they are graph-computable, deterministic, and don't require live stakeholders. This table states which Quality-in-Use characteristic each RMAV dimension primarily operationalizes, and why:

| Quality-in-Use characteristic | Primarily operationalized by | Why |
|:---|:---|:---|
| **Effectiveness** | **A — Availability** | A structural SPOF's removal partitions the graph — dependents cannot complete their function at all. |
| **Efficiency** | **R — Reliability**, **M — Maintainability** | Cascades (R) force retries/failover; tight coupling (M) means every change or incident costs more engineering effort per unit of value delivered. |
| **Satisfaction** | **R + V** | Repeated cascading outages erode trust (R); being a high-value attack target (V) undermines confidence even absent an actual incident. |
| **Freedom from risk** | **A + V** (dominant), **R** | Availability quantifies economic/operational risk (SPOF = certain partition); Vulnerability quantifies security/legal risk (breach exposure); Reliability quantifies propagation risk. This is why $Q(v)$ weights $A$ highest (0.43) — freedom-from-risk is the dominant quality-in-use concern for infrastructure components. |
| **Context coverage** | Cross-scenario/cross-domain stability of the score | A component's criticality ranking should hold across topologies and domains; instability here is a weakness of the *criticality signal itself*, checked via the per-domain repeated stratified k-fold evaluation and multi-scenario batch runs (`cli/run_scenarios.sh`). |

### 4.3 Proxy, Not Ground Truth

Quality-in-use is behavioral and only directly observable via user studies, incident data, or — as used in this project — **simulated failure impact**. RMAV/edge scores are validated against that simulated ground truth ($I_R(v), I_M(v), I_A(v), I_V(v)$ from [failure-simulation.md](failure-simulation.md)) via the correlation, F1, and SPOF-F1 metrics computed in [validation.md](validation.md). That validation step is precisely the question "does our structural criticality proxy actually track quality-in-use loss under failure?" RMAV should be described as **operationalizing Effectiveness and Freedom-from-Risk as computable, validated structural metrics**, not as a complete model of Quality-in-Use — Satisfaction and Context Coverage are only partially covered, and not directly measured from live stakeholder behavior.

---

## 5. Where This Fits in the Pipeline

| Step | Relation to criticality |
|:---|:---|
| [graph-model.md](graph-model.md) | Produces $G_{\text{analysis}}(l)$ and derives `DEPENDS_ON` edges — the substrate both node and edge criticality are computed over. |
| [structural-analysis.md](structural-analysis.md) | Computes the Tier-1 metric vector $M(v)$ and deterministic RMAV scores — see §2.2 and §3.2 above. |
| [prediction.md](prediction.md) | Refines RMAV into GNN-blended node scores and direct edge scores $Q_{\text{GNN}}(u,v)$ — see §3.3 above. |
| [failure-simulation.md](failure-simulation.md) | Produces the simulated ground truth ($I^*(v)$, $I_{R/M/A/V}(v)$) that criticality proxies are trained/validated against. |
| [validation.md](validation.md) | Statistically checks whether structural/learned criticality tracks simulated impact — the empirical check on §4.3. |

## 6. References

- ISO/IEC 25010:2011 / 25010:2023, *Systems and software Quality Requirements and Evaluation (SQuaRE) — System and software quality models*.
- Tarjan, R. (1972). *Depth-first search and linear graph algorithms*. SIAM Journal on Computing, 1(2), 146-160. (bridges / cut-edges, also cited in [structural-analysis.md §9.9](structural-analysis.md#99-bridge-ratio-br))
- Henry, S., & Kafura, D. (1981). *Software structure metrics based on information flow*. IEEE Transactions on Software Engineering, (5), 510-518. (structural coupling, also cited in [structural-analysis.md §9.3](structural-analysis.md#93-multi-path-coupling-index-mpci))
