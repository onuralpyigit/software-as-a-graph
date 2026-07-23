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
   - 3.3 [Edge RMAV Decomposition](#33-edge-rmav-decomposition)
   - 3.4 [Learned Edge Scoring (GNN)](#34-learned-edge-scoring-gnn)
   - 3.5 [Ranking Critical Edges](#35-ranking-critical-edges)
4. [Quality-in-Use Framing (ISO/IEC 25010 SQuaRE)](#4-quality-in-use-framing-isoiec-25010-square)
   - 4.1 [The Five Quality-in-Use Characteristics](#41-the-five-quality-in-use-characteristics)
   - 4.2 [Stakeholders & Context of Use](#42-stakeholders--context-of-use)
   - 4.3 [Mapping RMAV to Quality-in-Use](#43-mapping-rmav-to-quality-in-use)
   - 4.4 [Worked Example: Score to Narrative](#44-worked-example-score-to-narrative)
   - 4.5 [Proxy, Not Ground Truth](#45-proxy-not-ground-truth)
   - 4.6 [Real-World Drivers vs. Structural Proxies](#46-real-world-drivers-vs-structural-proxies)
5. [Where This Fits in the Pipeline](#5-where-this-fits-in-the-pipeline)
6. [References](#6-references)

---

## 1. Overview

Criticality is how much a component or connection matters to the people who rely on the system working — measured by what would go wrong for them if it failed, not by how the code is written internally.

Per ISO/IEC 25010 (SQuaRE), Quality-in-Use is quality as experienced by a stakeholder in a real context of use, not an internal code property. So criticality answers: **"If this fails, how much worse does the outcome get for the people using the system?"** — broken into five plain questions:

| Quality-in-Use characteristic | What a user would notice |
|:---|:---|
| Effectiveness | "Can I still get my task done at all, or does it just stop working?" |
| Efficiency | "Does it now take more time, retries, or resources to get the same result?" |
| Satisfaction | "Do I still trust this system / enjoy using it after this happens?" |
| Freedom from risk | "Does this failure cost money, create safety issues, or expose data?" |
| Context coverage | "Does this hold up everywhere I use it, or only in some situations?" |

A component or edge is **critical** to the extent its failure degrades one or more of these for real stakeholders. The RMAV structural metrics below (Reliability, Maintainability, Availability, Vulnerability) are *proxies* for this — graph-computable stand-ins used because you can't survey real users for every simulated failure. See [§4](#4-quality-in-use-framing-isoiec-25010-square) for the full technical mapping.

Every entity in the graph — a component (node) or a dependency (edge) — carries a **criticality** signal: a score describing how much damage its failure, degradation, or compromise would do to the rest of the system. This document is the conceptual home for that concept. It does not re-derive formulas that already live in [structural-analysis.md](structural-analysis.md) and [prediction.md](prediction.md); it defines the terms, shows where each signal is computed, and explains the theoretical grounding (ISO/IEC 25010 SQuaRE Quality-in-Use) that motivates *why* these particular structural signals are treated as criticality.

Two distinct but related concepts are in scope:

| Concept | Applies to | Primary output | User-side failure signature |
|:---|:---|:---|:---|
| **Component criticality** | Nodes ($v \in V$: Application, Broker, Topic, Node, Library) | RMAV scores $R(v), M(v), A(v), V(v), Q(v)$ + five-tier classification | The node itself goes away. E.g. MainBroker fails → every application routed through it loses its only path to publish/subscribe; the task stops outright, it doesn't degrade (§4.4). |
| **Relationship criticality** | Edges ($e \in E$: physical pub-sub links and derived `DEPENDS_ON` edges) | Structural bridge/betweenness signals + GNN edge score $Q_{\text{GNN}}(u,v)$ | The node survives, but one specific *link* to it breaks. A low-criticality node can still sit behind a single highly critical bridge edge — losing that one relationship is as consequential as losing a much higher-scoring node (§3.1). |

---

## 2. Component (Node) Criticality

### 2.1 Definition

> **Component criticality** is the degree to which the failure, latency, or degradation of a specific software component reduces the system's capacity to enable users to achieve specified goals with effectiveness, efficiency, freedom from risk, and satisfaction within its operational context — and how consistently that impact holds across the contexts in which the system is used.

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

> **Relationship criticality** is the degree to which the failure, latency, or degradation of a specific dependency relationship — a pub-sub linkage or derived `DEPENDS_ON` edge, independent of its endpoints' own criticality — reduces the system's capacity to enable users to achieve specified goals with effectiveness, efficiency, freedom from risk, and satisfaction within its operational context — and how consistently that impact holds across the contexts in which the system is used.

Where component criticality asks "how dangerous is losing this node," relationship criticality asks "how dangerous is losing this specific *link*, independent of the endpoints' own criticality." A high-criticality node can still have many low-criticality (redundant) edges; a low-criticality node can sit at the far end of a single, highly critical bridge edge.

### 3.2 Structural Edge Signals

Relationship criticality is assembled from per-edge structural signals computed in [saag/analysis/structural_analyzer.py](../saag/analysis/structural_analyzer.py) and carried by `EdgeMetrics` / `EdgeQuality` in [saag/core/metrics.py](../saag/core/metrics.py):

- **`is_bridge`** — whether the edge is a graph bridge (cut-edge): `nx.bridges()` over the undirected projection. Removing a bridge disconnects a subgraph from the rest of the system.
- **`betweenness`** — edge betweenness centrality (`nx.edge_betweenness_centrality`, QoS-weighted): the fraction of shortest dependency paths that traverse this specific edge.
- **`weight`** — the edge's QoS-derived weight from [graph-model.md](graph-model.md).

These are distinct from two *node-level* metrics that are easy to mistake for edge scores because they are edge-derived:

- **Bridge Ratio `BR(v)`** ([structural-analysis.md §9.9](structural-analysis.md#99-bridge-ratio-br)) — the *fraction of a node's own connections* that are bridges. It describes a node's exposure to non-redundant edges, not a per-edge score.
- **Multi-Path Coupling Index `MPCI(v)`** ([structural-analysis.md §9.3](structural-analysis.md#93-multi-path-coupling-index-mpci)) — counts *redundant* shared channels feeding into a node. High MPCI means a node's incoming edges are collectively low-criticality (multi-channel, no single edge is a SPOF); low MPCI (with high `DG_in`) means each incoming edge is closer to a single point of failure for that dependency.

### 3.3 Edge RMAV Decomposition

Just as component criticality is decomposed into RMAV (§2.2), each edge is scored on the same four dimensions in [`_score_and_classify_edges`](../saag/analysis/analyzer.py#L412-L483) — an edge is not reduced to a single number, but assessed as reliability, maintainability, availability, and vulnerability risks in its own right, blending the edge's intrinsic structural signals (§3.2) with its endpoints' own RMAV scores:

| Dimension | Question Answered for an Edge | Formula (blend of edge-intrinsic + endpoint context) |
|:---|:---|:---|
| **R — Reliability** | How much does this specific link contribute to fault propagation? | Edge betweenness + edge weight (bridge proxy) + `max(source.R, target.R)` |
| **M — Maintainability** | How much does this link add to coupling/change cost? | Edge betweenness + is-bridge flag + edge weight |
| **A — Availability** | Does losing this specific link partition the graph? | is-bridge flag + `min(source.A, target.A)` |
| **V — Vulnerability** | How much does this link expand the attack surface? | Edge weight (QoS-derived) + `max(source.V, target.V)` |

Two design choices carry meaning:

- **`max()` for R and V, `min()` for A** — a link is only as *reliable/secure* as its riskiest endpoint (failure or compromise on either side propagates through the edge), but it is only as *available* as its weakest endpoint (the edge can't be more resilient than the more fragile side it connects).
- **`is_bridge` appears in both M and A** — a non-redundant edge is expensive to reroute around (raises M) *and* is a structural cut-point if removed (raises A) — the same structural fact, two different consequences.

The four dimension scores are combined into the same overall composite formula used for nodes (§2.2), giving each edge a `QualityScores` record (`reliability`, `maintainability`, `availability`, `security`, `overall`) identical in shape to a component's — see [`EdgeQuality`](../saag/core/metrics.py#L345-L378).

### 3.4 Learned Edge Scoring (GNN)

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

### 3.5 Ranking Critical Edges

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

Restated per characteristic, as a definition of what "high criticality on that characteristic" means for a component or relationship in this graph:

- **Effectiveness criticality** — failure directly prevents a dependent from completing its function, or corrupts the result it produces. This is what a structural SPOF (high $A(v)$, §2.2) operationalizes: removal partitions the graph, so downstream components cannot complete their function at all, not just slower.
- **Efficiency criticality** — failure or added latency forces dependents into retries, failover, or extra resource spend to reach the same outcome. This is what cascade reach ($R$) and coupling cost ($M$) operationalize.
- **Freedom-from-risk criticality** — malfunction exposes the operator or business to economic loss, safety hazard, or security/compliance breach. This is what $A$ (operational risk) and $V$ (security/legal risk) jointly operationalize, and why $Q(v)$ weights $A$ highest (§2.2).
- **Satisfaction criticality** — repeated or high-profile failures erode downstream trust in the system, independent of whether the immediate task technically still completes. Only partially operationalized by RMAV (§4.5).
- **Context-coverage criticality** — whether the impact holds in every deployment/topology this component or relationship appears in, or only in specialized configurations. Checked empirically via cross-scenario/cross-domain stability, not a single RMAV dimension (§4.5).

### 4.2 Stakeholders & Context of Use

ISO/IEC 25010 defines Quality-in-Use relative to *specified* stakeholders operating in a *specified* context of use — it is not a free-floating property. Applied to this project, "stakeholder" actually spans two distinct populations, and conflating them is the easiest way to misread a criticality score:

- **Who is harmed by a failure.** The end users and operators of the *system being modeled* — e.g. the customers of an ATM network, the downstream services subscribing to a Kafka topic, the operators of a ROS 2 robot. The five characteristics in §4.1 describe *their* experience, and that is what a criticality score is ultimately a proxy for.
- **Who acts on the criticality signal.** The engineering role that consumes $Q(v)$/RMAV output to prioritize remediation. Each RMAV dimension has a named primary consumer (see the [RMAV Quality Model table in README.md](../README.md#rmav-quality-model)):

  | RMAV Dimension | Primary Engineering Stakeholder | Whose Quality-in-Use They're Protecting |
  |:---|:---|:---|
  | **R** — Reliability | Reliability Engineer | Effectiveness/Efficiency for end users caught in a cascade |
  | **M** — Maintainability | Software Architect | Efficiency of the engineering team's own change process |
  | **A** — Availability | DevOps / SRE | Effectiveness and Freedom from risk for end users during an outage |
  | **V** — Vulnerability | Security Engineer | Freedom from risk (security/legal exposure) for end users and the business |

A criticality score routes a structural signal to the engineering role equipped to act on it, but the *severity* it encodes is always denominated in harm to the first population — not in convenience for the second. The mapping in §4.3 is stated from that first, end-user perspective.

### 4.3 Mapping RMAV to Quality-in-Use

RMAV and edge criticality are **structural proxies** for quality-in-use loss under failure — they are graph-computable, deterministic, and don't require live stakeholders. This table states which Quality-in-Use characteristic each RMAV dimension primarily operationalizes, and why:

| Quality-in-Use characteristic | Primarily operationalized by | Why |
|:---|:---|:---|
| **Effectiveness** | **A — Availability** | A structural SPOF's removal partitions the graph — dependents cannot complete their function at all. |
| **Efficiency** | **R — Reliability**, **M — Maintainability** | Cascades (R) force retries/failover; tight coupling (M) means every change or incident costs more engineering effort per unit of value delivered. |
| **Satisfaction** | **R + V** | Repeated cascading outages erode trust (R); being a high-value attack target (V) undermines confidence even absent an actual incident. |
| **Freedom from risk** | **A + V** (dominant), **R** | Availability quantifies economic/operational risk (SPOF = certain partition); Vulnerability quantifies security/legal risk (breach exposure); Reliability quantifies propagation risk. This is why $Q(v)$ weights $A$ highest (0.43) — freedom-from-risk is the dominant quality-in-use concern for infrastructure components. |
| **Context coverage** | Cross-scenario/cross-domain stability of the score | A component's criticality ranking should hold across topologies and domains; instability here is a weakness of the *criticality signal itself*, checked via the per-domain repeated stratified k-fold evaluation and multi-scenario batch runs (`cli/run_scenarios.sh`). |

### 4.4 Worked Example: Score to Narrative

The formulas above stay abstract until tied to an instance. [structural-analysis.md §13](structural-analysis.md#13-worked-example) computes `A(MainBroker) = 0.679` → **HIGH**, driven by `AP_c_directed = 0.65` (a directed structural SPOF) and `BR = 1.0` (every one of MainBroker's edges is a bridge — there is no redundant path around it). Read as a Quality-in-Use narrative for the end users of that system:

- **Effectiveness** — if MainBroker fails, both SensorApp and MonitorApp lose their only path to publish/subscribe on `/temperature`. The monitoring task doesn't degrade, it stops: `BR = 1.0` means there is no alternate route left to fall back on.
- **Freedom from risk** — an undetected temperature excursion during that outage is an economic or safety risk to whoever depends on the reading, which is exactly why $A$ carries the highest weight (0.43) in $Q(v)$ (§2.2).
- **Context coverage** — because every one of MainBroker's edges is a bridge, this holds in *every* context this topology is used in; there's no scenario where a redundant path happens to save the day.

A component scoring MINIMAL on $A$ but HIGH on $V$ (e.g. a rarely-invoked but highly-reachable component) would instead read as: normal operation is unaffected (Effectiveness fine), but a compromise there has outsized reach (Freedom from risk driven by breach exposure, not outage) — the same five-characteristic lens, applied to a different RMAV profile.

### 4.5 Proxy, Not Ground Truth

Quality-in-use is behavioral and only directly observable via user studies, incident data, or — as used in this project — **simulated failure impact**. RMAV/edge scores are validated against that simulated ground truth ($I_R(v), I_M(v), I_A(v), I_V(v)$ from [failure-simulation.md](failure-simulation.md)) via the correlation, F1, and SPOF-F1 metrics computed in [validation.md](validation.md). That validation step is precisely the question "does our structural criticality proxy actually track quality-in-use loss under failure?" RMAV should be described as **operationalizing Effectiveness and Freedom-from-Risk as computable, validated structural metrics**, not as a complete model of Quality-in-Use — Satisfaction and Context Coverage are only partially covered, and not directly measured from live stakeholder behavior.

### 4.6 Real-World Drivers vs. Structural Proxies

§4.5 states the gap qualitatively; this section states it dimension-by-dimension. In a live system, each RMAV dimension is really driven by a mix of runtime, code, and security signals — most of which this project has no field for. The graph model ([graph-model.md](graph-model.md)) carries topology, a DDS-style QoS weight (`reliability`/`durability`/`transport_priority` + message size — a delivery-guarantee proxy, not live traffic or security metadata), and static code metrics (LOC, cyclomatic complexity, instability, LCOM). There is no MTTF/MTTR, privilege, encryption, or telemetry field anywhere in the schema — so every RMAV number below is a structural stand-in, never a direct read of the real-world driver.

**Component criticality:**

| Dimension | Real-world driver | What the structural proxy actually captures | Not captured |
|:---|:---|:---|:---|
| **R** | Intrinsic failure rate (MTTF) and severity of an independent failure | Reverse PageRank + in-degree + CDPot (§2.2): how far/deep a failure would propagate *given that it happens* | Whether the node fails often at all — $R(v)$ is purely blast-radius, silent on the component's own failure rate |
| **M** | Change-impact risk: regression likelihood from complexity and code churn | `CQP` (§2.2, structural-analysis.md §11.2) blends `complexity_norm`, `instability_code`, `lcom_norm` from static code metrics, plus topological betweenness/coupling | Code churn as a time-series (commit frequency) — `instability_code` is a point-in-time Martin instability ratio, not a churn rate |
| **A** | SPOF status weighted by MTTR (how long the outage lasts once it starts) | `AP_c_directed` + bridge ratio + `QSPOF` (§2.2): whether removing the node partitions the graph | MTTR — every structural SPOF is scored the same regardless of how fast it would actually be restored |
| **V** | Asset value (PII/secrets handled) and the component's privilege level | Reverse eigenvector/closeness centrality + QoS-weighted in-degree (§2.2): how reachable/central the node is | Data sensitivity, privilege level, or what the component does — $V(v)$ is topology-only; a low-privilege leaf handling PII would still score MINIMAL |

**Relationship criticality:**

| Dimension | Real-world driver | What the structural proxy actually captures | Not captured |
|:---|:---|:---|:---|
| **R** | Cascading-failure probability: synchronous/blocking calls with no circuit breaker | Edge betweenness + bridge factor + `max(source.R, target.R)` (§3.3) | Whether the call is actually synchronous/blocking or backed by a circuit breaker — a runtime/code property invisible to a static graph |
| **M** | Interface/contract volatility: how likely a change on one side breaks the other | Edge betweenness + is-bridge flag + edge weight (§3.3) | Semantic contract coupling (e.g. a shared database schema) — only topological reachability is measured |
| **A** | Traffic bottlenecks: throughput/bandwidth saturation, lack of redundant routing | is-bridge flag + `min(source.A, target.A)` (§3.3): whether this specific link is structurally redundant | Live traffic volume — `weight` (§3.2) is the QoS delivery-guarantee proxy, not measured throughput or bandwidth |
| **V** | Trust-boundary crossing: unencrypted channels, missing mutual TLS, lateral-movement potential | Edge weight (QoS) + `max(source.V, target.V)` (§3.3) | Encryption status, inter-service authentication, or any security-boundary metadata — none of this exists in the schema |

None of this makes the structural proxy wrong — [validation.md](validation.md) is precisely the check that it still tracks simulated failure impact well enough to be useful. It does mean a CRITICAL/HIGH score should be read as *"structurally exposed,"* not as *"this component definitely has a high MTTF/PII/no-circuit-breaker problem"* — those specific root causes still require the engineering stakeholder from §4.2 to inspect the actual component or relationship.

---

## 5. Where This Fits in the Pipeline

| Step | Relation to criticality |
|:---|:---|
| [graph-model.md](graph-model.md) | Produces $G_{\text{analysis}}(l)$ and derives `DEPENDS_ON` edges — the substrate both node and edge criticality are computed over. |
| [structural-analysis.md](structural-analysis.md) | Computes the Tier-1 metric vector $M(v)$ and deterministic RMAV scores — see §2.2 and §3.2 above. |
| [prediction.md](prediction.md) | Refines RMAV into GNN-blended node scores and direct edge scores $Q_{\text{GNN}}(u,v)$ — see §3.3 above. |
| [failure-simulation.md](failure-simulation.md) | Produces the simulated ground truth ($I^*(v)$, $I_{R/M/A/V}(v)$) that criticality proxies are trained/validated against. |
| [validation.md](validation.md) | Statistically checks whether structural/learned criticality tracks simulated impact — the empirical check on §4.5. |

## 6. References

- ISO/IEC 25010:2011 / 25010:2023, *Systems and software Quality Requirements and Evaluation (SQuaRE) — System and software quality models*.
- Tarjan, R. (1972). *Depth-first search and linear graph algorithms*. SIAM Journal on Computing, 1(2), 146-160. (bridges / cut-edges, also cited in [structural-analysis.md §9.9](structural-analysis.md#99-bridge-ratio-br))
- Henry, S., & Kafura, D. (1981). *Software structure metrics based on information flow*. IEEE Transactions on Software Engineering, (5), 510-518. (structural coupling, also cited in [structural-analysis.md §9.3](structural-analysis.md#93-multi-path-coupling-index-mpci))
