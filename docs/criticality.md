# Component and Relationship Criticality

**Define what "criticality" means for a node and for an edge in the dependency graph, grounded in the ISO/IEC 25010 (SQuaRE) Quality-in-Use model, and relate the project's structural/learned scores to that stakeholder-facing definition.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quality-in-Use Foundation (ISO/IEC 25010 SQuaRE)](#2-quality-in-use-foundation-isoiec-25010-square)
   - 2.1 [What Quality-in-Use Is](#21-what-quality-in-use-is)
   - 2.2 [Stakeholders: Who Is Harmed vs. Who Acts](#22-stakeholders-who-is-harmed-vs-who-acts)
   - 2.3 [Context of Use](#23-context-of-use)
   - 2.4 [The Five Criticality Questions](#24-the-five-criticality-questions)
3. [Component (Node) Criticality](#3-component-node-criticality)
   - 3.1 [Definition](#31-definition)
   - 3.2 [User-Side Failure Signature](#32-user-side-failure-signature)
   - 3.3 [The RMAV Model](#33-the-rmav-model)
   - 3.4 [What the Component Carries: the Weight Channel](#34-what-the-component-carries-the-weight-channel)
   - 3.5 [Mapping RMAV to Quality-in-Use](#35-mapping-rmav-to-quality-in-use)
   - 3.6 [Criticality Classification](#36-criticality-classification)
4. [Relationship (Edge) Criticality](#4-relationship-edge-criticality)
   - 4.1 [Definition](#41-definition)
   - 4.2 [Why a Link Needs Its Own Score](#42-why-a-link-needs-its-own-score)
   - 4.3 [Structural Edge Signals](#43-structural-edge-signals)
   - 4.4 [What the Relationship Carries: the Weight Channel](#44-what-the-relationship-carries-the-weight-channel)
   - 4.5 [Edge RMAV Decomposition](#45-edge-rmav-decomposition)
   - 4.6 [Learned Edge Scoring (GNN)](#46-learned-edge-scoring-gnn)
   - 4.7 [Ranking Critical Edges](#47-ranking-critical-edges)
5. [From Score to Stakeholder Narrative](#5-from-score-to-stakeholder-narrative)
   - 5.1 [Worked Example](#51-worked-example)
   - 5.2 [Reading a Score as a Quality-in-Use Statement](#52-reading-a-score-as-a-quality-in-use-statement)
6. [Limits of the Proxy](#6-limits-of-the-proxy)
   - 6.1 [Proxy, Not Ground Truth](#61-proxy-not-ground-truth)
   - 6.2 [Characteristic Coverage](#62-characteristic-coverage)
   - 6.3 [Real-World Drivers vs. Structural Proxies](#63-real-world-drivers-vs-structural-proxies)
7. [Where This Fits in the Pipeline](#7-where-this-fits-in-the-pipeline)
8. [References](#8-references)

---

## 1. Overview

Criticality is defined here **from the stakeholder's side**: how much a component or connection matters to the people who depend on the system working — measured by what would go wrong *for them* if it failed, not by how the code is written internally.

That framing is taken directly from ISO/IEC 25010 (SQuaRE), which separates **product quality** (internal/external attributes of the software itself — reliability, maintainability, security as code properties) from **Quality-in-Use** (the outcome a *specified stakeholder* experiences while operating the system in a *specified context of use*). Criticality in this project is a Quality-in-Use concept, so it answers one question:

> **If this fails, how much worse does the outcome get for the people who depend on the system?**

Every entity in the graph — a component (node) or a dependency (edge) — carries a **criticality** signal answering that question as a score. This document is the conceptual home for the concept. It does not re-derive formulas that already live in [structural-analysis.md](structural-analysis.md) and [prediction.md](prediction.md); it defines the terms in stakeholder-facing language ([§2](#2-quality-in-use-foundation-isoiec-25010-square)), states what each score means for a node ([§3](#3-component-node-criticality)) and for an edge ([§4](#4-relationship-edge-criticality)), and is explicit about where the structural proxy stops tracking real Quality-in-Use ([§6](#6-limits-of-the-proxy)).

Two distinct but related concepts are in scope:

| Concept | Applies to | Primary output | What the stakeholder experiences |
|:---|:---|:---|:---|
| **Component criticality** | Nodes ($v \in V$: Application, Broker, Topic, Node, Library) | RMAV scores $R(v), M(v), A(v), V(v), Q(v)$ + five-tier classification | The component itself goes away. E.g. MainBroker fails → every application routed through it loses its only path to publish/subscribe; the user's task stops outright, it doesn't merely slow down ([§5.1](#51-worked-example)). |
| **Relationship criticality** | Edges ($e \in E$: physical pub-sub links and derived `DEPENDS_ON` edges) | Structural bridge/betweenness signals + edge RMAV composite + GNN edge score $Q_{\text{GNN}}(u,v)$ | Both components survive, but one specific *link* between them breaks. The stakeholder sees a partial outage — one data flow stops while the rest of the system stays up ([§4.2](#42-why-a-link-needs-its-own-score)). |

The RMAV structural metrics (Reliability, Maintainability, Availability, Vulnerability) used throughout are **proxies** for Quality-in-Use loss — graph-computable stand-ins, used because you cannot survey real stakeholders for every simulated failure. Their mapping to the standard is stated in [§3.5](#35-mapping-rmav-to-quality-in-use) and their limits in [§6](#6-limits-of-the-proxy).

Both concepts are scored over the **weighted** graph, never over bare topology: the QoS-derived weights $w(v)$ and $w(e)$ computed in Step 1 carry the declared delivery guarantee of each component and each dependency into every RMAV dimension. Structure says how many stakeholder outcomes route through an element; weight says how strongly each of those outcomes was promised. [§3.4](#34-what-the-component-carries-the-weight-channel) and [§4.4](#44-what-the-relationship-carries-the-weight-channel) trace that channel term by term.

---

## 2. Quality-in-Use Foundation (ISO/IEC 25010 SQuaRE)

### 2.1 What Quality-in-Use Is

ISO/IEC 25010 defines **Quality-in-Use** as the degree to which a product used by specific stakeholders meets their needs to achieve specific goals with effectiveness, efficiency, freedom from risk and satisfaction, in specific contexts of use. Three properties of that definition drive everything below:

1. **It is measured at the outcome, not at the artifact.** A component is not critical because it is complex or centrally placed; it is critical because losing it degrades an outcome someone cares about.
2. **It is relative to named stakeholders.** "Critical" is meaningless without answering *critical to whom* ([§2.2](#22-stakeholders-who-is-harmed-vs-who-acts)).
3. **It is relative to a named context of use.** The same broker is critical in one deployment and replaceable in another ([§2.3](#23-context-of-use)).

The model has five characteristics:

| Characteristic | Sub-characteristics | Meaning |
|:---|:---|:---|
| **Effectiveness** | — | Accuracy and completeness with which stakeholders achieve their goals |
| **Efficiency** | — | Resources expended relative to the effectiveness achieved |
| **Satisfaction** | Usefulness, Trust, Pleasure, Comfort | Stakeholder response to using the system |
| **Freedom from risk** | Economic, Health & safety, Environmental risk mitigation | Degree to which the system limits risk of harm |
| **Context coverage** | Context completeness, Flexibility | Degree to which quality-in-use is sustained across all intended contexts |

> **Standard lineage.** These five characteristics are the Quality-in-Use model of **ISO/IEC 25010:2011**, which this document uses as its operative formulation. The 2023 revision of ISO/IEC 25010 covers the *product quality* model only; the Quality-in-Use model was moved into the separate standard **ISO/IEC 25019:2023**. Citations that need to be current should reference 25019:2023 for Quality-in-Use and 25010:2023 for product quality — the five-characteristic decomposition used here is unaffected.

### 2.2 Stakeholders: Who Is Harmed vs. Who Acts

Quality-in-Use is defined relative to *specified* stakeholders. In this project "stakeholder" spans two distinct populations, and conflating them is the easiest way to misread a criticality score.

**Population 1 — who is harmed by a failure.** The end users and operators of the *system being modeled*: the driver of an autonomous vehicle, the clinician reading a patient monitor, the trader whose order flow crosses a matching engine, the customer at an ATM. The five characteristics in §2.1 describe *their* experience. **A criticality score is denominated in harm to this population.**

**Population 2 — who acts on the criticality signal.** The engineering role that consumes RMAV output to prioritize remediation. Each dimension has a named primary consumer (see the [RMAV Quality Model table in README.md](../README.md#rmav-quality-model)):

| RMAV Dimension | Primary Engineering Stakeholder | Whose Quality-in-Use They Are Protecting |
|:---|:---|:---|
| **R** — Reliability | Reliability Engineer | Effectiveness/Efficiency for end users caught in a cascade |
| **M** — Maintainability | Software Architect | Efficiency of the engineering team's own change process |
| **A** — Availability | DevOps / SRE | Effectiveness and Freedom from risk for end users during an outage |
| **V** — Vulnerability | Security Engineer | Freedom from risk (security/legal exposure) for end users and the business |

A criticality score **routes** a structural signal to the engineering role equipped to act on it, but the **severity** it encodes is always harm to Population 1 — never convenience for Population 2. Maintainability is the one dimension where the two populations partly coincide: the engineering team is itself a stakeholder whose efficiency the score measures.

### 2.3 Context of Use

The same structural position carries different Quality-in-Use weight in different domains, which is why the project validates across the domain scenarios in [scenario.md](scenario.md) rather than a single topology. The dominant characteristic per domain:

| Scenario domain | Primary stakeholder | Dominant Quality-in-Use characteristic | A CRITICAL score there means |
|:---|:---|:---|:---|
| ROS 2 / autonomous vehicle | Vehicle occupants, road users | Freedom from risk (health & safety) | A sensor/perception path failure is a safety hazard, not an inconvenience |
| Healthcare / clinical HIS | Clinicians, patients | Freedom from risk (health & safety) + Effectiveness | A lost vitals stream means care decisions are made on stale data |
| Financial trading (HFT) | Traders, the operating firm | Freedom from risk (economic) + Efficiency | Latency added by failover is itself the loss; downtime is priced per second |
| ATM / aviation surveillance | Customers, controllers | Effectiveness + Freedom from risk (economic) | The transaction or the track simply cannot complete |
| IoT smart city | Residents, city operators | Context coverage + Efficiency | Impact depends on which districts/devices are in scope at the time |
| Enterprise ESB / microservices | Internal service teams, end customers | Efficiency + Satisfaction (trust) | Degradation is felt as slowdown and eroded confidence before it is felt as outage |

The graph model itself does not carry a domain-criticality field: the pipeline computes structure, and the reader supplies the context row above when converting a score into a decision.

### 2.4 The Five Criticality Questions

Restated as the canonical, user-side definition used throughout this document. A component or relationship is **critical on a characteristic** to the extent that its failure produces the effect in the right-hand column:

| Quality-in-Use characteristic | The stakeholder's question | Criticality on that characteristic means |
|:---|:---|:---|
| **Effectiveness** | "Can I still get my task done at all?" | Failure directly prevents a dependent from completing its function, or corrupts the result it produces. The task **stops**, it does not degrade. |
| **Efficiency** | "Does it now cost more time, retries, or resources to get the same result?" | Failure or added latency forces dependents into retries, failover, or extra resource spend to reach the same outcome. |
| **Satisfaction** | "Do I still trust this system after this happens?" | Repeated or high-profile failures erode stakeholder trust and confidence, independent of whether the immediate task technically still completes. |
| **Freedom from risk** | "Does this failure cost money, endanger someone, or expose data?" | Malfunction exposes the operator or business to economic loss, safety hazard, or security/compliance breach. |
| **Context coverage** | "Does this hold up everywhere I use the system, or only in some situations?" | The impact holds in every deployment and topology the component appears in, rather than only in specialized configurations. |

These five questions are the definition. Everything from [§3](#3-component-node-criticality) onward is the machinery for **estimating** the answers from graph structure alone.

---

## 3. Component (Node) Criticality

### 3.1 Definition

> **Component criticality** is the degree to which the failure, latency, or degradation of a specific software component reduces the system's capacity to enable its stakeholders to achieve their goals with effectiveness, efficiency, freedom from risk, and satisfaction within the intended context of use — and how consistently that reduction holds across the contexts in which the system is used.

Stated as an operational rule: **a component is critical in proportion to how many stakeholder outcomes stop being achievable when it stops working.**

That single quantity is not directly computable, so it is **decomposed into four dimensions — Reliability, Maintainability, Availability, and Vulnerability/Security (RMAV)** — each capturing one distinct mechanism by which a component's failure destroys stakeholder value:

$$
\text{criticality}(v) \;=\; f\big(\underbrace{R(v)}_{\text{it spreads}},\; \underbrace{M(v)}_{\text{it resists change}},\; \underbrace{A(v)}_{\text{it stops everything}},\; \underbrace{V(v)}_{\text{it invites attack}}\big)
$$

The four dimensions are *not* four separate definitions of criticality; they are four separable causes of the same stakeholder harm, kept apart because each has a different remedy and a different owner ([§2.2](#22-stakeholders-who-is-harmed-vs-who-acts)). Per-dimension definitions are given in [§3.3](#33-the-rmav-model) and their edge counterparts in [§4.5](#45-edge-rmav-decomposition).

**Two inputs, not one.** Each dimension is computed over the **weighted** dependency graph, so criticality is a function of both where a component sits *and* what it carries:

$$
\text{criticality}(v) \;=\; f\big(\;\underbrace{\text{position of } v \text{ in } G_{\text{analysis}}(l)}_{\text{structure — how many outcomes route through it}},\;\; \underbrace{w(v),\; \{w(e) : e \text{ incident to } v\}}_{\text{weight — how much each of those outcomes is guaranteed}}\;\big)
$$

Structure alone cannot separate a SPOF carrying `RELIABLE`/`PERSISTENT` safety telemetry from a topologically identical SPOF carrying `BEST_EFFORT`/`VOLATILE` debug logs; from the stakeholder's side those are not the same failure. The QoS-derived weights $w(v)$ and $w(e)$ computed in Step 1 ([graph-model.md §4.3, §4.5](graph-model.md#43-phase-3--intrinsic-weight-computation)) are what encode that difference, and [§3.4](#34-what-the-component-carries-the-weight-channel) traces exactly where they enter each dimension.

Criticality is computed, not asserted: no component carries a manually assigned criticality label, and no score is hand-tuned per component. It is derived from the component's position in $G_{\text{analysis}}(l)$ (the layer-projected dependency graph produced by [graph-model.md](graph-model.md)) together with the weights derived from its declared QoS policies. The QoS policy *is* an authored input — so criticality inherits the delivery guarantees the architect declared, while remaining independent of any opinion the architect holds about what is important ([§3.4](#34-what-the-component-carries-the-weight-channel) closes with what that dependency costs).

### 3.2 User-Side Failure Signature

Before any formula, each characteristic has a recognizable failure signature for a component. This is what the score is trying to detect:

| Characteristic | Component-failure signature the stakeholder observes |
|:---|:---|
| **Effectiveness** | A function becomes unreachable. The dependent has no alternative route, so its task returns nothing or returns a wrong/stale result. Structurally this is a **single point of failure**. |
| **Efficiency** | The task still completes, but through retries, a failover path, or a degraded mode — more time and more resource per unit of delivered value. Structurally this is **cascade reach** and **coupling cost**. |
| **Satisfaction** | The stakeholder starts routing around the system, adding manual checks, or escalating — the loss is confidence, and it outlives the incident. |
| **Freedom from risk** | The failure window itself is the harm: an undetected safety excursion, an unbookable transaction, an exposed data path. |
| **Context coverage** | The above holds in every deployment of this topology, not just the one that happened to be measured. |

### 3.3 The RMAV Model

Component criticality is decomposed into four orthogonal structural dimensions — **Reliability, Maintainability, Availability, Vulnerability (RMAV)** — combined into a composite score $Q(v)$. The full formulas, weights, and derivations are defined in [structural-analysis.md §11](structural-analysis.md#11-analyze-stage--rule-based-rmav-scoring); the summary, with each dimension tied to the characteristic it estimates:

| Dimension | Question Answered | Driven Primarily By | Estimates (§2.4) |
|:---|:---|:---|:---|
| **R — Reliability** | How broadly/deeply does failure propagate? | Reverse PageRank *(QoS-weighted)*, in-degree, Cascade Depth Potential | Efficiency, Satisfaction |
| **M — Maintainability** | How hard is this to change safely? | Betweenness *(QoS-weighted)*, QoS-weighted efferent coupling, Code Quality Penalty | Efficiency (engineering-side) |
| **A — Availability** | Is this a structural single point of failure? | Directed articulation point score, bridge ratio, QoS-SPOF *(uses $w(v)$)* | Effectiveness, Freedom from risk |
| **V — Vulnerability** | How attractive a target is this for attack? | Reverse eigenvector *(QoS-weighted)*/closeness centrality, QoS-weighted in-degree | Freedom from risk, Satisfaction |

The italicised terms are where the Step 1 weights enter; [§3.4](#34-what-the-component-carries-the-weight-channel) traces each one.

> **Terminology — Vulnerability vs. Security.** The conceptual dimension is **Vulnerability** ($V$): how exposed a component is to compromise. The serialized field in `QualityScores` and `CriticalityRanking` is named **`security`** ([saag/core/metrics.py#L243-L258](../saag/core/metrics.py#L243-L258)). They are the same dimension viewed from opposite ends — high vulnerability *is* low security — and a **high** `security` score means **worse** (more exposed), matching the direction of the other three dimensions where high always means more critical. This document uses $V$ / "Vulnerability" for the concept and `security` when naming the field.

#### Component criticality per dimension

Each dimension is itself a full criticality definition, scoped to one failure mechanism:

**R — Reliability criticality**
> The degree to which a component's failure **propagates beyond itself**, converting one local fault into a multi-component outage.

| | |
|:---|:---|
| Stakeholder question | "When this breaks, how much else breaks with it?" |
| High score means | The component sits upstream of many transitive dependents; a fault reaches far and deep |
| Structural drivers | Reverse PageRank, normalized in-degree, Enhanced Cascade Depth Potential (Topics: Fan-Out Criticality) |
| Quality-in-Use effect | **Efficiency** (dependents retry/fail over), then **Satisfaction** (repeated cascades erode trust) |
| Acted on by | Reliability Engineer — bulkheads, circuit breakers, cascade containment |

**M — Maintainability criticality**
> The degree to which a component **resists safe change**, so that every modification to it carries disproportionate regression risk for the rest of the system.

| | |
|:---|:---|
| Stakeholder question | "How expensive and risky is it to change or fix this?" |
| High score means | A structural bottleneck with high fan-out coupling and poor internal code quality |
| Structural drivers | Betweenness centrality, efferent coupling/degree, Code Quality Penalty (`complexity_norm`, `instability_code`, `lcom_norm`) |
| Quality-in-Use effect | **Efficiency**, uniquely on the engineering stakeholder ([§2.2](#22-stakeholders-who-is-harmed-vs-who-acts)) rather than the end user — slower fixes, longer incident recovery |
| Acted on by | Software Architect — decoupling, interface extraction, refactoring |

**A — Availability criticality**
> The degree to which a component is a **structural single point of failure**: its removal partitions the dependency graph, leaving dependents with no alternative path.

| | |
|:---|:---|
| Stakeholder question | "If this is down, does anything still work?" |
| High score means | Removing the node disconnects a subgraph — there is no redundant route around it |
| Structural drivers | Directed articulation-point score `AP_c_directed`, Bridge Ratio `BR`, QoS-weighted SPOF `QSPOF` |
| Quality-in-Use effect | **Effectiveness** — the only dimension where the stakeholder's task stops outright — plus **Freedom from risk** (the outage window is itself the harm) |
| Acted on by | DevOps / SRE — redundancy, failover, replication |

**V — Vulnerability / Security criticality**
> The degree to which a component is an **attractive and reachable target**, such that its compromise (rather than its failure) grants disproportionate reach into the system.

| | |
|:---|:---|
| Stakeholder question | "If an attacker owns this, how much do they own?" |
| High score means | Centrally reachable, with many high-QoS flows converging on it |
| Structural drivers | Reverse eigenvector centrality, closeness centrality, QoS-weighted in-degree |
| Quality-in-Use effect | **Freedom from risk** (security/legal/compliance exposure) and **Satisfaction** (confidence loss even without an incident) |
| Acted on by | Security Engineer — hardening, segmentation, access control |

The dimensions are scored and classified **independently** ([§3.6](#36-criticality-classification)), so a component carries a four-way profile rather than one label — the profile, not the composite, is what identifies *which* kind of criticality is present and therefore which remedy applies.

$$
Q(v) = 0.43 \cdot A(v) + 0.24 \cdot R(v) + 0.17 \cdot M(v) + 0.16 \cdot V(v)
$$

**Why $A$ dominates the weighting, in Quality-in-Use terms.** Availability is the only dimension that maps onto **Effectiveness** — the characteristic where the stakeholder's task stops outright rather than costing more. A structural SPOF partitions the graph *deterministically*, whereas cascade propagation (R), coupling risk (M), and attack exposure (V) are *probabilistic* and typically surface as Efficiency or Freedom-from-risk loss. Weighting $A$ at 0.43 is therefore a statement about stakeholder harm — total loss of a goal outranks a more expensive path to it — not merely a statement about graph topology.

See [saag/core/criticality.py](../saag/core/criticality.py) for the `CriticalityRanking` DTO that carries these scores through the pipeline.

### 3.4 What the Component Carries: the Weight Channel

Two components can occupy identical positions in the graph and still differ in criticality, because the *guarantees attached to the data they handle* differ. That difference reaches the RMAV scores through the **weight channel** — the QoS-derived weights computed in Step 1.

#### From declared QoS to a component's weight

```
declared QoS policy + message size          (topology JSON, per Topic)
        │  reliability / durability / transport_priority, AHP-weighted
        ▼
w(t) ∈ [0,1]                                 Step 1, Phase 3
        │  inherited by PUBLISHES_TO / SUBSCRIBES_TO / ROUTES edges
        ▼
w(v) ∈ [0,1]  for App, Library, Broker, Node  Step 1, Phase 5a (hybrid max/mean, fan-out amplified for libraries)
w(e) ∈ [0,1]  for every DEPENDS_ON edge        Step 1, Phases 4 and 5b (worst-case QoS over the mediating topics)
        │
        ├─ w(v), and the per-node sums w_in = Σw(e), w_out = Σw(e)
        │       → rank-normalised across the analysed layer
        └─ w(e) itself → used as-is, as centrality edge weight and in edge scoring
        ▼
weight-bearing terms of R, M, A, V            Step 2
```

Full derivation of $w(t)$, $w(v)$ and $w(e)$ is in [graph-model.md §4.3–§4.5](graph-model.md#43-phase-3--intrinsic-weight-computation). What matters here is the semantics carried along: **$w$ is a delivery-guarantee proxy** — how strongly the system promises that this data arrives — not a measure of traffic volume, revenue, or safety class ([§6.3](#63-real-world-drivers-vs-structural-proxies)).

#### Where weight enters each dimension

Every RMAV dimension is weight-aware, but through different mechanisms. Coefficients are the AHP defaults from [structural-analysis.md §11](structural-analysis.md#11-analyze-stage--rule-based-rmav-scoring):

| Dimension | Weight-bearing term | Mechanism | Reading |
|:---|:---|:---|:---|
| **R** | `RPR` (0.45) | Reverse PageRank uses $w(e)$ as edge *importance* on $G^T$ | Cascade reach is measured along the strongly-guaranteed paths, so a fault that would travel over `RELIABLE`/`PERSISTENT` dependencies scores higher than one travelling over best-effort links |
| **M** | `w_out` (0.30), `BT` (0.35) | `w_out(v) = Σ w(e)` over efferent edges, rank-normalised; betweenness uses $1/w(e)$ as *distance* | Depending on many high-guarantee components makes change expensive — each is an SLA obligation that a change must not break |
| **A** | `QSPOF` (0.25), `w(v)` (0.05) | $\text{QSPOF} = \text{AP\_c\_directed} \times w(v)$, plus a direct additive term | A SPOF is amplified in proportion to what it guarantees: a partition on high-QoS traffic is a worse Effectiveness loss than the same partition on volatile traffic |
| **V** | `QADS` = `w_in` (0.25), `REV` (0.40) | `w_in(v) = Σ w(e)` over afferent edges, rank-normalised; reverse eigenvector uses $w(e)$ as importance | A component is an attractive target in proportion to the guarantees its dependents rely on — compromising it breaks the promises that matter most |

Two structural companions of the weight travel with it:

- **`path_count`** — the number of topics (or shared nodes) mediating one `DEPENDS_ON` edge ([graph-model.md §3](graph-model.md#3-formal-graph-definition)). It is deliberately *not* folded into $w(e)$ — that would break the $w \in [0,1]$ contract — so it enters criticality separately: through `MPCI`, which *amplifies* the cascade-depth term of $R$ (three shared topics between the same pair are three simultaneous failure vectors, a tighter coupling than three independent single-topic links), and through `path_complexity`, which raises $M$'s coupling-risk term. Note the node/edge duality: multiplicity makes the *pair* more fragile while making each *individual* channel more replaceable ([§4.3](#43-structural-edge-signals)).
- **Topic fan-out** — `subscriber_count` / `publisher_count`, computed in Phase 2, drive `FOC` and the Topic-specific form of $R$. Note that Topic reliability is scored from fan-out and message *frequency*, not from $w(t)$: $w(t)$ reaches Topics' neighbours instead, through the aggregation above.

#### Two different things called "weight"

The word is overloaded in this project and the two senses never mix:

| | Graph weight | Dimension weight |
|:---|:---|:---|
| Written | $w(v)$, $w(e)$ | $q_R, q_M, q_A, q_V$ and the per-metric coefficients |
| Comes from | Declared QoS policy + message size, per component/edge | AHP pairwise comparison of the dimensions, once for the whole model |
| Varies | Per component and per edge, within one system | Per model configuration, not per component |
| Answers | "How strongly is *this* flow guaranteed?" | "How much does *this kind* of criticality count?" |

They meet in exactly one place: the composite coefficients $q_*$ are themselves adapted to the system's aggregate QoS profile before scoring (`adapt_qos_weights`, on by default). A system whose topics are predominantly `PERSISTENT`/`RELIABLE`/high-priority shifts weight toward $R$ and $A$; a predominantly `VOLATILE`/`BEST_EFFORT` system shifts it toward $M$ and $V$ — the same structural graph therefore yields a different composite ranking in a mission-critical deployment than in a best-effort one, which is [§2.3](#23-context-of-use)'s context-of-use argument made computable.

#### What this dependency costs

- **Weights are relative, not absolute.** $w(v)$, `w_in` and `w_out` are rank-normalised across the components of the analysed layer before entering RMAV, so the weight channel expresses *ordering within one system*, matching the relative reading of the tiers in [§3.6](#36-criticality-classification). Raising every topic in a system to `RELIABLE`/`PERSISTENT` does not raise everything's criticality; it only flattens the channel's ability to discriminate.
- **A mis-declared QoS policy is a mis-scored component.** If the topology declares `BEST_EFFORT` on a flow the business actually treats as critical, the weight channel faithfully reproduces that error. This is the one place where criticality inherits an authored judgement, and it is the first thing to check when a score contradicts operational experience.
- **The floor is not zero.** $w(t) \geq 0.01$ by construction, so an unguaranteed flow still contributes structure; the weight channel modulates criticality, it never switches it off.

### 3.5 Mapping RMAV to Quality-in-Use

RMAV is a set of **structural proxies** for Quality-in-Use loss under failure — graph-computable, deterministic, and requiring no live stakeholders. This table states which characteristic each dimension primarily operationalizes, and why:

| Quality-in-Use characteristic | Primarily operationalized by | Why |
|:---|:---|:---|
| **Effectiveness** | **A — Availability** | A structural SPOF's removal partitions the graph — dependents cannot complete their function at all. |
| **Efficiency** | **R — Reliability**, **M — Maintainability** | Cascades (R) force retries/failover; tight coupling (M) means every change or incident costs more engineering effort per unit of value delivered. |
| **Satisfaction** | **R + V** | Repeated cascading outages erode trust (R); being a high-value attack target (V) undermines confidence even absent an actual incident. |
| **Freedom from risk** | **A + V** (dominant), **R** | Availability quantifies economic/operational risk (SPOF = certain partition); Vulnerability quantifies security/legal risk (breach exposure); Reliability quantifies propagation risk. |
| **Context coverage** | Cross-scenario/cross-domain stability of the score | A component's criticality ranking should hold across topologies and domains; instability here is a weakness of the *criticality signal itself*, checked via the per-domain repeated stratified k-fold evaluation and multi-scenario batch runs (`cli/run_scenarios.sh`). |

The mapping is many-to-many by design: no single RMAV dimension is a characteristic, and no characteristic is fully captured by one dimension. Coverage gaps are enumerated in [§6.2](#62-characteristic-coverage).

### 3.6 Criticality Classification

Raw $Q(v)$ scores are mapped onto five tiers using **adaptive box-plot thresholding**, relative to the system's own score distribution rather than fixed cutoffs — full definition in [structural-analysis.md §11.7](structural-analysis.md#117-criticality-classification):

```
CRITICAL  :  score > Q3 + 1.5 × IQR
HIGH      :  Q3 < score ≤ upper fence
MEDIUM    :  median < score ≤ Q3
LOW       :  Q1 < score ≤ median
MINIMAL   :  score ≤ Q1
```

Implemented by `CriticalityLevel` and `BoxPlotStats` in [saag/core/criticality.py](../saag/core/criticality.py).

Two consequences matter when reading a tier as a stakeholder statement:

- **Tiers are relative, not absolute.** A CRITICAL label means "an outlier *within this system's* distribution," not "critical in an absolute, cross-system sense." A well-designed redundant system still has a CRITICAL tier; a system full of SPOFs still has a MINIMAL tier. The tier prioritizes attention inside one system; it does not compare two systems.
- **Per-dimension tiers are the diagnostic.** Classification is applied independently per RMAV dimension and for the composite. A component can be CRITICAL on Availability while MINIMAL on Vulnerability — which reads, in the language of §2.4, as "this threatens Effectiveness but not Freedom from risk," and directs remediation accordingly.

---

## 4. Relationship (Edge) Criticality

### 4.1 Definition

> **Relationship criticality** is the degree to which the failure, latency, or degradation of a specific dependency relationship — a pub-sub linkage or derived `DEPENDS_ON` edge, independent of its endpoints' own criticality — reduces the system's capacity to enable its stakeholders to achieve their goals with effectiveness, efficiency, freedom from risk, and satisfaction within the intended context of use — and how consistently that reduction holds across the contexts in which the system is used.

Where component criticality asks *"how dangerous is losing this component,"* relationship criticality asks *"how dangerous is losing this specific link, even though both components are still running."*

It is decomposed along the **same four RMAV dimensions** as a component ([§3.1](#31-definition)), scoped to the link rather than the endpoint:

$$
\text{criticality}(u,v) \;=\; f\big(\underbrace{R(u,v)}_{\text{it conducts faults}},\; \underbrace{M(u,v)}_{\text{it binds the two sides}},\; \underbrace{A(u,v)}_{\text{it is the only route}},\; \underbrace{V(u,v)}_{\text{it is a path in}}\big)
$$

Using one dimension set for both nodes and edges is deliberate: it makes the two comparable in a single ranking, and it lets a remediation owner ([§2.2](#22-stakeholders-who-is-harmed-vs-who-acts)) read node and edge findings in the same vocabulary — an SRE reads $A$ on both, a Security Engineer reads $V$ on both. Per-dimension edge definitions are in [§4.5](#45-edge-rmav-decomposition).

As with a component ([§3.1](#31-definition)), each dimension is computed over the weighted graph, so a link's criticality combines its position with the guarantees of the flow crossing it:

$$
\text{criticality}(u,v) \;=\; f\big(\;\underbrace{\text{position of } (u,v) \text{ in } G_{\text{analysis}}(l)}_{\text{is this link replaceable?}},\;\; \underbrace{w(u,v),\; \text{path\_count}(u,v)}_{\text{what does it guarantee, over how many channels?}},\;\; \underbrace{R,M,A,V \text{ of } u \text{ and } v}_{\text{what does it connect?}}\;\big)
$$

The third argument is what distinguishes an edge score from a node score: an edge is scored *in the context of its endpoints* — it can be no more reliable than its riskiest end, nor more available than its weakest ([§4.5](#45-edge-rmav-decomposition)). The second is traced in [§4.4](#44-what-the-relationship-carries-the-weight-channel).

### 4.2 Why a Link Needs Its Own Score

From the stakeholder's side, an edge failure and a node failure produce different observable symptoms, which is why a separate score is warranted rather than inheriting endpoint scores:

- **A node failure is a total outage of a capability.** Everything the component provides stops.
- **An edge failure is a partial outage.** The component is up, its dashboards are green, its other consumers are fine — but *one* data flow has stopped. For the stakeholder on the far end of that link, Effectiveness is lost just as completely as in a full outage, while the operator sees a healthy system.

This asymmetry produces the two cases the model must handle:

- A **high-criticality node** can have many **low-criticality edges** — a redundantly connected broker, where losing any single link changes nothing for anyone.
- A **low-criticality node** can sit behind a **single highly critical bridge edge** — losing that one relationship is as consequential for its dependents as losing a much higher-scoring component.

Edge criticality is therefore governed by one structural question with a direct Quality-in-Use reading: **is this link replaceable?** A replaceable link degrades Efficiency (traffic reroutes, costs more). A non-replaceable link — a graph bridge — destroys Effectiveness for everything behind it.

### 4.3 Structural Edge Signals

Relationship criticality is assembled from per-edge structural signals computed in [saag/analysis/structural_analyzer.py](../saag/analysis/structural_analyzer.py) and carried by `EdgeMetrics` / `EdgeQuality` in [saag/core/metrics.py](../saag/core/metrics.py):

- **`is_bridge`** — whether the edge is a graph bridge (cut-edge): `nx.bridges()` over the undirected projection. Removing a bridge disconnects a subgraph from the rest of the system — the Effectiveness case above.
- **`betweenness`** — edge betweenness centrality (`nx.edge_betweenness_centrality`) computed over the **inverted-weight** graph, where each edge's length is $1/w(e)$: the fraction of shortest dependency paths that traverse this specific edge — the Efficiency case (how much traffic must reroute). Inversion is what makes strongly-guaranteed dependencies *short*, so they attract the shortest paths rather than repelling them.
- **`weight`** — $w(e)$, the edge's QoS-derived weight from [graph-model.md](graph-model.md): the worst-case guarantee over every topic mediating this dependency, a proxy for how strongly the flow across it is promised. Traced through the edge dimensions in [§4.4](#44-what-the-relationship-carries-the-weight-channel).
- **`path_count`** — how many distinct topics (or shared nodes) establish this one edge. Deliberately kept out of $w(e)$ to preserve $w \in [0,1]$, so it acts as a separate coupling-intensity signal.

These are distinct from two *node-level* metrics that are easy to mistake for edge scores because they are edge-derived:

- **Bridge Ratio `BR(v)`** ([structural-analysis.md §9.9](structural-analysis.md#99-bridge-ratio-br)) — the *fraction of a node's own connections* that are bridges. It describes a node's exposure to non-redundant edges, not a per-edge score.
- **Multi-Path Coupling Index `MPCI(v)`** ([structural-analysis.md §9.3](structural-analysis.md#93-multi-path-coupling-index-mpci)) — counts *redundant* shared channels feeding into a node. High MPCI means a node's incoming edges are collectively low-criticality (multi-channel, no single edge is a SPOF); low MPCI (with high `DG_in`) means each incoming edge is closer to a single point of failure for that dependency.

### 4.4 What the Relationship Carries: the Weight Channel

An edge's weight is not an attribute of the link's shape — it is the guarantee attached to the data crossing it, and it is computed in Step 1 before any criticality is scored:

```
w(e) for a structural edge      = w(t) of the topic it attaches to      (Phase 3, inheritance)
w(e) for a DEPENDS_ON edge      = max w(t) over the mediating topics    (Phase 4, worst case)
      app_to_lib                = w(consuming component)                 (Phase 5b)
      broker_to_broker          = w(shared node)                         (Phase 5b)
path_count(e)                   = number of mediating topics / shared nodes
```

Taking the worst case rather than an average is the conservative reading required by the definition in [§4.1](#41-definition): if *any* strongly-guaranteed flow crosses this link, losing the link breaks that guarantee, regardless of how many weak flows it also carries.

#### Where weight enters each edge dimension

Unlike a component, an edge uses $w(e)$ **directly, un-normalised** — the Step 1 contract already puts it in $[0,1]$, so no rank normalisation is needed. Coefficients are the `e_*` defaults in [`QualityWeights`](../saag/analysis/weight_calculator.py):

| Edge dimension | Weight-bearing term | Reading |
|:---|:---|:---|
| **R** | $w(e)$ at 0.30 — second only to betweenness | A link conducts faults in proportion to what it promises to deliver; a `RELIABLE`/`PERSISTENT` channel that fails breaks a promise a dependent was entitled to rely on |
| **M** | $w(e)$ at 0.15 | A high-guarantee contract is expensive to renegotiate — both sides must move together |
| **A** | **none** | Availability asks only *"is this link replaceable?"*, and redundancy is a topological fact: a bridge is a bridge whether it carries safety telemetry or debug logs. QoS amplification reaches $A$ indirectly, through the endpoints' own `QSPOF` ([§3.4](#34-what-the-component-carries-the-weight-channel)) |
| **V** | $w(e)$ at 0.15 | A channel is worth attacking in proportion to the value of what flows over it |
| *(via betweenness, in R and M)* | $1/w(e)$ as path length | Strongly-guaranteed edges sit on more shortest paths, raising the betweenness of the links the system leans on |

`path_count` does not enter the edge score directly. It shapes the *endpoints'* $R$ and $M$ ([§3.4](#34-what-the-component-carries-the-weight-channel)), and of those only $R$ can reach the edge again — through the `max(source.R, target.R)` term of edge $R$. Edge $M$ carries no endpoint term at all, so a link's maintainability score is blind to how coupled its endpoints already are.

> **Reading note — edge dimension scales.** The four edge dimensions do not draw on the same coefficient mass ($R$ sums to 0.85 of a possible 1.0, $M$ to 0.80, $A$ to 0.50, $V$ to 0.35), so raw edge scores are comparable *within* a dimension but not *across* dimensions. Since classification is box-plot relative within the edge set ([§3.6](#36-criticality-classification)), per-dimension rankings and tiers are unaffected; only the raw magnitudes are, and the composite mixes the dimensions at these differing scales. Read an edge's dimension tiers, not its absolute dimension values.

### 4.5 Edge RMAV Decomposition

Just as component criticality is decomposed into RMAV ([§3.3](#33-the-rmav-model)), each edge is scored on the same four dimensions in [`_score_and_classify_edges`](../saag/analysis/analyzer.py#L412-L483) — an edge is not reduced to a single number, but assessed as reliability, maintainability, availability, and vulnerability risks in its own right, blending the edge's intrinsic structural signals ([§4.3](#43-structural-edge-signals)) with its endpoints' own RMAV scores:

| Dimension | Question Answered for an Edge | Formula (edge-intrinsic + weight channel + endpoint context) |
|:---|:---|:---|
| **R — Reliability** | How much does this specific link contribute to fault propagation? | `0.35·betweenness + 0.30·w(e) + 0.20·max(source.R, target.R)` |
| **M — Maintainability** | How much does this link add to coupling/change cost? | `0.35·betweenness + 0.30·is_bridge + 0.15·w(e)` |
| **A — Availability** | Does losing this specific link partition the graph? | `0.30·is_bridge + 0.20·min(source.A, target.A)` |
| **V — Vulnerability** | How much does this link expand the attack surface? | `0.15·w(e) + 0.20·max(source.V, target.V)` |

#### Relationship criticality per dimension

Each dimension restated as a definition of the link, parallel to the component definitions in [§3.3](#33-the-rmav-model):

**R — Reliability criticality (edge)**
> The degree to which a relationship acts as a **conductor of faults** — the channel along which a failure at one endpoint reaches the other.

| | |
|:---|:---|
| Stakeholder question | "If the upstream side breaks, does this link carry the damage downstream?" |
| High score means | A heavily traversed link whose riskiest endpoint has wide blast radius |
| Structural drivers | Edge betweenness, edge weight, `max(source.R, target.R)` |
| Quality-in-Use effect | **Efficiency** — dependents on the far side retry or fail over |
| Acted on by | Reliability Engineer — timeouts, circuit breakers, backpressure on this specific flow |

**M — Maintainability criticality (edge)**
> The degree to which a relationship **binds its two endpoints together**, so that a change on one side forces a coordinated change on the other.

| | |
|:---|:---|
| Stakeholder question | "Can either side of this link evolve independently?" |
| High score means | A non-redundant, heavily used link — the contract across it cannot be changed unilaterally or routed around |
| Structural drivers | Edge betweenness, `is_bridge`, edge weight |
| Quality-in-Use effect | **Efficiency** for the engineering stakeholder — coordinated releases, higher change cost |
| Acted on by | Software Architect — interface versioning, contract decoupling |

**A — Availability criticality (edge)**
> The degree to which a relationship is the **only route** between what it connects: removing it partitions the graph even though both endpoints stay up.

| | |
|:---|:---|
| Stakeholder question | "If just this connection drops, is anything cut off?" |
| High score means | The edge is a structural bridge — this is the defining case of relationship criticality ([§4.2](#42-why-a-link-needs-its-own-score)) |
| Structural drivers | `is_bridge`, `min(source.A, target.A)` |
| Quality-in-Use effect | **Effectiveness** — total task loss for everything behind the bridge, while the operator's dashboards stay green |
| Acted on by | DevOps / SRE — redundant routing, multi-broker paths, alternate channels |

**V — Vulnerability / Security criticality (edge)**
> The degree to which a relationship is a **usable path into or across the system** — the lateral-movement value of the link itself, distinct from the value of either endpoint.

| | |
|:---|:---|
| Stakeholder question | "If an attacker gets onto this channel, where does it take them?" |
| High score means | A high-QoS flow touching an already-exposed endpoint |
| Structural drivers | Edge weight (QoS), `max(source.V, target.V)` |
| Quality-in-Use effect | **Freedom from risk** — interception, injection, or lateral movement along the channel |
| Acted on by | Security Engineer — channel encryption, mutual authentication, segmentation |

**Node and edge criticality side by side.** The same dimension asks a structurally different question depending on what it is scoring:

| Dimension | Component criticality asks | Relationship criticality asks |
|:---|:---|:---|
| **R** | Does *this component's* failure spread? | Does *this link* carry the spread? |
| **M** | Is *this component* hard to change safely? | Does *this link* force the two sides to change together? |
| **A** | Is *this component* a SPOF? | Is *this link* a SPOF, even with both components healthy? |
| **V** | Is *this component* a valuable target? | Is *this link* a usable route to a target? |

Three design choices in the formulas carry meaning:

- **`max()` for R and V, `min()` for A** — a link is only as *reliable/secure* as its riskiest endpoint (failure or compromise on either side propagates through the edge), but it is only as *available* as its weakest endpoint (the edge cannot be more resilient than the more fragile side it connects).
- **`is_bridge` appears in both M and A** — a non-redundant edge is expensive to reroute around (raises M, an Efficiency cost to the engineering stakeholder) *and* is a structural cut-point if removed (raises A, an Effectiveness loss to the end user) — the same structural fact, two different stakeholder consequences.
- **`w(e)` appears in R, M and V but not A** — the guarantee crossing a link scales how much its loss costs, but not whether it can be lost at all ([§4.4](#44-what-the-relationship-carries-the-weight-channel)). Replaceability is topological; consequence is QoS-weighted.

The four dimension scores are combined with the same composite coefficients used for nodes ([§3.3](#33-the-rmav-model)), including the QoS-profile adaptation described in [§3.4](#34-what-the-component-carries-the-weight-channel), giving each edge a `QualityScores` record (`reliability`, `maintainability`, `availability`, `security`, `overall`) identical in shape to a component's — see [`EdgeQuality`](../saag/core/metrics.py#L345-L378).

### 4.6 Learned Edge Scoring (GNN)

The Predict stage's GNN produces a direct, per-edge criticality prediction rather than relying on endpoint-node proxies — see [prediction.md §2.6](prediction.md#26-edge-criticality-prediction) and [design/SDD.md §6.26](design/SDD.md) for the full architecture:

```
score(u, v) = TypedEdgeEncoder_r( h_u, h_v, e_uv )

e_uv ∈ ℝ^16: QoS weight + path_count_norm + 7-bit edge-type one-hot + 7-bit QoS features
```

Training labels are **measured by removing the edge**, not inferred from its endpoints:

```
I_edge(u, v) = composite_impact(G \ {(u,v)})  −  composite_impact(G)
```

`FailureSimulator.simulate_edge_removal` severs the one relationship, leaves both endpoints active — the partial-outage case of [§4.2](#42-why-a-link-needs-its-own-score) — and recomputes reachability, fragmentation, throughput and flow disruption. Subtracting the no-op control matters: the impact function is non-zero on a pristine graph (topics that already lack a publisher or subscriber count as lost throughput), so a level rather than a delta would give every edge that floor as if it were signal.

This replaces the earlier heuristic `I_edge(u,v) = I*(u) × {1.0 if bridge else 0.1}`, which encoded the §4.2 Effectiveness/Efficiency distinction as a hand-chosen 10× gap rather than observing it. That heuristic remains available for ablation.

> **What the measurement shows, and a caution.** Most individual edges cost almost nothing: on `av_system`, 4 of 40 candidates carry non-zero impact. That is the §4.2 replaceability question answered empirically — most links *are* replaceable. It also exposes a modelling gap the heuristic hid: `RUNS_ON` edges measure exactly 0.0 because the cascade routes no traffic over them ([failure-simulation.md L5](failure-simulation.md#11-known-limitations)), even though bridge detection flags them as non-redundant. A zero there means "this model cannot express that link's failure", not "that link does not matter" — the same caveat that applies to Topic and Node labels (L6).

### 4.7 Ranking Critical Edges

Edges are ranked for reporting/UI consumption via `get_critical_edges()` in [saag/analysis/service.py](../saag/analysis/service.py) and exposed through [api/routers/components.py](../api/routers/components.py), sorting by `EdgeQuality.scores.overall` (the same RMAV-style composite machinery used for nodes, applied edge-wise).

Simulated edge criticality is available separately via `SimulationService.classify_edges()`, which returns `EdgeCriticality` records from the removal sweep ([§4.6](#46-learned-edge-scoring-gnn)). Two fields must be read together:

- `combined_impact` — the measured delta. Comparable to node `composite_impact` because it uses the same weighting.
- `evaluated` — whether the edge was in the candidate set at all. The sweep measures `bridges(G) ∪ top-q edge-betweenness`; everything else returns `evaluated: false` with `combined_impact = 0.0`. **An unevaluated edge is not a harmless edge** — sorting the two together would rank never-measured links alongside measured-as-zero ones.

---

## 5. From Score to Stakeholder Narrative

### 5.1 Worked Example

The formulas stay abstract until tied to an instance. [structural-analysis.md §13](structural-analysis.md#13-worked-example) computes `A(MainBroker) = 0.679` → **HIGH**, driven by `AP_c_directed = 0.65` (a directed structural SPOF) and `BR = 1.0` (every one of MainBroker's edges is a bridge — there is no redundant path around it). Of that score, `0.25 × (0.65 × 0.71) + 0.05 × 0.71 = 0.151` — roughly a fifth — comes from the weight channel: MainBroker routes `/temperature`, a `RELIABLE`/`TRANSIENT_LOCAL`/`HIGH` topic, and that declared guarantee is what makes its normalised weight 0.71. Route only `BEST_EFFORT`/`VOLATILE` traffic through the same topological position and the same SPOF scores materially lower. Read as a Quality-in-Use narrative for the end users of that system:

- **Effectiveness** — if MainBroker fails, both SensorApp and MonitorApp lose their only path to publish/subscribe on `/temperature`. The monitoring task does not degrade, it stops: `BR = 1.0` means there is no alternate route to fall back on.
- **Efficiency** — there is no cheaper-but-slower path to fall back to either; the Efficiency characteristic is not the one at stake here, which is exactly what distinguishes a SPOF from a bottleneck.
- **Freedom from risk** — an undetected temperature excursion during that outage is an economic or safety risk to whoever depends on the reading, which is why $A$ carries the highest weight (0.43) in $Q(v)$ ([§3.3](#33-the-rmav-model)).
- **Context coverage** — because every one of MainBroker's edges is a bridge, this holds in *every* context this topology is used in; there is no scenario where a redundant path happens to save the day.

A component scoring MINIMAL on $A$ but HIGH on $V$ (e.g. a rarely-invoked but highly-reachable component) reads differently under the same lens: normal operation is unaffected (Effectiveness intact), but a compromise there has outsized reach (Freedom from risk driven by breach exposure, not outage).

### 5.2 Reading a Score as a Quality-in-Use Statement

A repeatable template for turning any RMAV profile into a stakeholder-facing statement — the intended way to consume the pipeline's output:

1. **Identify the stakeholder and context** from [§2.3](#23-context-of-use). *"This is a clinical HIS; the harmed party is a clinician making a care decision."*
2. **Take the dominant per-dimension tier**, not the composite. The composite ranks; the dimension explains.
3. **Translate the dimension into its characteristic** via [§3.5](#35-mapping-rmav-to-quality-in-use). *High A → Effectiveness and Freedom from risk.*
4. **State the consequence in the stakeholder's terms** using the failure signature in [§3.2](#32-user-side-failure-signature). *"If this fails, the vitals stream stops entirely; there is no alternate route, so the clinician sees stale data with no indication it is stale."*
5. **Qualify with the proxy's limits** from [§6](#6-limits-of-the-proxy). *"Structurally exposed — this says nothing about how often this component actually fails or how quickly it would be restored."*

The last step is not optional. A CRITICAL tier is a statement about **structural exposure to Quality-in-Use loss**, not a measurement of Quality-in-Use loss itself.

---

## 6. Limits of the Proxy

### 6.1 Proxy, Not Ground Truth

Quality-in-Use is behavioral and only directly observable via user studies, incident data, or — as used in this project — **simulated failure impact**. RMAV/edge scores are validated against that simulated ground truth ($I_R(v), I_M(v), I_A(v), I_V(v)$ from [failure-simulation.md](failure-simulation.md)) via the correlation, F1, and SPOF-F1 metrics computed in [validation.md](validation.md). That validation step is precisely the question *"does our structural criticality proxy actually track Quality-in-Use loss under failure?"*

RMAV should therefore be described as **operationalizing Effectiveness and Freedom-from-risk as computable, validated structural metrics** — not as a complete model of Quality-in-Use.

### 6.2 Characteristic Coverage

Stated per characteristic, so the gap is explicit rather than implied:

| Characteristic | Coverage | Basis and gap |
|:---|:---|:---|
| **Effectiveness** | **Strong** | Directly operationalized by $A$ (structural partition) and `is_bridge`; validated against simulated reachability loss. |
| **Efficiency** | **Moderate** | $R$ and $M$ capture cascade reach and coupling cost, but the *magnitude* of the extra cost (latency, retries, engineer-hours) is not modelled — only that a cost exists. |
| **Freedom from risk** | **Moderate** | Economic/operational risk is well proxied by $A$; safety and environmental risk are not represented at all (no criticality-of-function field), and security risk is topology-only ([§6.3](#63-real-world-drivers-vs-structural-proxies)). |
| **Satisfaction** | **Weak** | Inferred indirectly from $R$ and $V$. Trust erosion is a behavioural response with no structural correlate; nothing in the pipeline measures it. |
| **Context coverage** | **Indirect** | Not a dimension but a property of the *signal*: assessed empirically as cross-scenario/cross-domain ranking stability ([validation.md](validation.md), `cli/run_scenarios.sh`), not computed per component. |

The two weakest rows are inherent, not implementation debt: Satisfaction and Context coverage are defined in the standard over live stakeholder behaviour across real deployments, which a static structural model cannot observe.

### 6.3 Real-World Drivers vs. Structural Proxies

§6.1 states the gap qualitatively; this section states it dimension-by-dimension. In a live system, each RMAV dimension is really driven by a mix of runtime, code, and security signals — most of which this project has no field for. The graph model ([graph-model.md](graph-model.md)) carries topology, a DDS-style QoS weight (`reliability`/`durability`/`transport_priority` + message size — a delivery-guarantee proxy, not live traffic or security metadata), and static code metrics (LOC, cyclomatic complexity, instability, LCOM). There is no MTTF/MTTR, privilege, encryption, or telemetry field anywhere in the schema — so every RMAV number below is a structural stand-in, never a direct read of the real-world driver.

**Component criticality:**

| Dimension | Real-world driver | What the structural proxy actually captures | Not captured |
|:---|:---|:---|:---|
| **R** | Intrinsic failure rate (MTTF) and severity of an independent failure | Reverse PageRank + in-degree + CDPot ([§3.3](#33-the-rmav-model)): how far/deep a failure would propagate *given that it happens* | Whether the node fails often at all — $R(v)$ is purely blast-radius, silent on the component's own failure rate |
| **M** | Change-impact risk: regression likelihood from complexity and code churn | `CQP` ([§3.3](#33-the-rmav-model), structural-analysis.md §11.2) blends `complexity_norm`, `instability_code`, `lcom_norm` from static code metrics, plus topological betweenness/coupling | Code churn as a time-series (commit frequency) — `instability_code` is a point-in-time Martin instability ratio, not a churn rate |
| **A** | SPOF status weighted by MTTR (how long the outage lasts once it starts) | `AP_c_directed` + bridge ratio + `QSPOF` ([§3.3](#33-the-rmav-model)): whether removing the node partitions the graph | MTTR — every structural SPOF is scored the same regardless of how fast it would actually be restored |
| **V** | Asset value (PII/secrets handled) and the component's privilege level | Reverse eigenvector/closeness centrality + QoS-weighted in-degree ([§3.3](#33-the-rmav-model)): how reachable/central the node is, weighted by what its dependents were promised | Data sensitivity, privilege level, or what the component does — a low-privilege leaf handling PII scores MINIMAL unless its QoS declaration happens to be high |

**Relationship criticality:**

| Dimension | Real-world driver | What the structural proxy actually captures | Not captured |
|:---|:---|:---|:---|
| **R** | Cascading-failure probability: synchronous/blocking calls with no circuit breaker | Edge betweenness + bridge factor + `max(source.R, target.R)` ([§4.5](#45-edge-rmav-decomposition)) | Whether the call is actually synchronous/blocking or backed by a circuit breaker — a runtime/code property invisible to a static graph |
| **M** | Interface/contract volatility: how likely a change on one side breaks the other | Edge betweenness + is-bridge flag + edge weight ([§4.5](#45-edge-rmav-decomposition)) | Semantic contract coupling (e.g. a shared database schema) — only topological reachability is measured |
| **A** | Traffic bottlenecks: throughput/bandwidth saturation, lack of redundant routing | is-bridge flag + `min(source.A, target.A)` ([§4.5](#45-edge-rmav-decomposition)): whether this specific link is structurally redundant | Live traffic volume — `weight` ([§4.3](#43-structural-edge-signals)) is the QoS delivery-guarantee proxy, not measured throughput or bandwidth |
| **V** | Trust-boundary crossing: unencrypted channels, missing mutual TLS, lateral-movement potential | Edge weight (QoS) + `max(source.V, target.V)` ([§4.5](#45-edge-rmav-decomposition)) | Encryption status, inter-service authentication, or any security-boundary metadata — none of this exists in the schema |

**What the weight channel does and does not fix.** The QoS weights ([§3.4](#34-what-the-component-carries-the-weight-channel), [§4.4](#44-what-the-relationship-carries-the-weight-channel)) are the one non-topological signal in RMAV, and they close part of every gap above: they distinguish a SPOF on guaranteed traffic from a SPOF on disposable traffic, and an attack surface that breaks strong promises from one that breaks weak ones. They close none of it fully, for three reasons:

- **A declaration, not a measurement.** $w$ encodes what the architecture *promises* about delivery, not what the system *does* — no throughput, latency, failure rate, or data sensitivity is behind it. A `RELIABLE`/`PERSISTENT` topic carrying one message a day outweighs a `BEST_EFFORT` topic carrying the business's entire traffic.
- **Delivery guarantee ≠ business criticality.** Nothing in the schema records safety class, revenue exposure, or regulatory scope, so the weight channel can only approximate those to the extent that architects happened to encode them in QoS policy.
- **Inherited errors are invisible.** A wrong QoS declaration produces a confidently wrong criticality score, with no internal signal that anything is off ([§3.4](#34-what-the-component-carries-the-weight-channel)).

None of this makes the structural proxy wrong — [validation.md](validation.md) is precisely the check that it still tracks simulated failure impact well enough to be useful. It does mean a CRITICAL/HIGH score should be read as *"structurally exposed to Quality-in-Use loss, at the guarantee level this architecture declares,"* not as *"this component definitely has a high MTTF/PII/no-circuit-breaker problem"* — those specific root causes still require the engineering stakeholder from [§2.2](#22-stakeholders-who-is-harmed-vs-who-acts) to inspect the actual component or relationship.

---

## 7. Where This Fits in the Pipeline

| Step | Relation to criticality |
|:---|:---|
| [graph-model.md](graph-model.md) | Produces $G_{\text{analysis}}(l)$ and derives `DEPENDS_ON` edges — the substrate both node and edge criticality are computed over — **and** computes the QoS weights $w(v)$, $w(e)$ and `path_count` that every RMAV dimension consumes ([§3.4](#34-what-the-component-carries-the-weight-channel), [§4.4](#44-what-the-relationship-carries-the-weight-channel)). |
| [structural-analysis.md](structural-analysis.md) | Computes the Tier-1 metric vector $M(v)$ and deterministic RMAV scores, weighting the centralities by $w(e)$ — see [§3.3](#33-the-rmav-model) and [§4.3](#43-structural-edge-signals) above. |
| [prediction.md](prediction.md) | Refines RMAV into GNN-blended node scores and direct edge scores $Q_{\text{GNN}}(u,v)$ — see [§4.6](#46-learned-edge-scoring-gnn) above. |
| [failure-simulation.md](failure-simulation.md) | Produces the simulated ground truth ($I^*(v)$, $I_{R/M/A/V}(v)$) that criticality proxies are trained/validated against — the closest observable stand-in for Quality-in-Use loss. |
| [validation.md](validation.md) | Statistically checks whether structural/learned criticality tracks simulated impact — the empirical check on [§6.1](#61-proxy-not-ground-truth). |

## 8. References

- ISO/IEC 25010:2011, *Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — System and software quality models*. (source of the five-characteristic Quality-in-Use model used in [§2.1](#21-what-quality-in-use-is))
- ISO/IEC 25019:2023, *Systems and software engineering — SQuaRE — Quality-in-use model*. (current home of the Quality-in-Use model after the 2023 revision moved it out of 25010)
- ISO/IEC 25010:2023, *Systems and software engineering — SQuaRE — Product quality model*. (product-quality characteristics, distinguished from Quality-in-Use in [§1](#1-overview))
- ISO/IEC 25022:2016, *Systems and software engineering — SQuaRE — Measurement of quality in use*. (measurement approach that structural proxies stand in for, cf. [§6.1](#61-proxy-not-ground-truth))
- Tarjan, R. (1972). *Depth-first search and linear graph algorithms*. SIAM Journal on Computing, 1(2), 146-160. (bridges / cut-edges, also cited in [structural-analysis.md §9.9](structural-analysis.md#99-bridge-ratio-br))
- Henry, S., & Kafura, D. (1981). *Software structure metrics based on information flow*. IEEE Transactions on Software Engineering, (5), 510-518. (structural coupling, also cited in [structural-analysis.md §9.3](structural-analysis.md#93-multi-path-coupling-index-mpci))
