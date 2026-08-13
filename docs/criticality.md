# Component and Relationship Criticality

**Define what "criticality" means for a node and for an edge in the dependency graph, grounded in the ISO/IEC 25019:2023 (SQuaRE) Quality-in-Use model, and relate the project's structural/learned scores to that stakeholder-facing definition.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [What "Criticality" Means Here](#2-what-criticality-means-here)
   - 2.1 [Three Established Traditions](#21-three-established-traditions)
   - 2.2 [What This Construct Borrows and Rejects](#22-what-this-construct-borrows-and-rejects)
   - 2.3 [Consequence, Not Risk (D3)](#23-consequence-not-risk)
   - 2.4 [Comparison with Classical Graph and Software Metrics](#24-comparison-with-classical-graph-and-software-metrics)
3. [Quality Grounding (SQuaRE)](#3-quality-grounding-square)
   - 3.0 [Three Quality Views: Internal, External, and Quality-in-Use](#30-three-quality-views-internal-external-and-quality-in-use)
   - 3.1 [What Quality-in-Use Is](#31-what-quality-in-use-is)
   - 3.2 [Stakeholders: Primary, Secondary, and Indirect](#32-stakeholders-primary-secondary-and-indirect)
   - 3.3 [Context of Use and Domain Context Vector](#33-context-of-use-and-domain-context-vector)
   - 3.4 [The Criticality Questions](#34-the-criticality-questions)
   - 3.5 [How the Dimensions Bind to External Quality, Dependability, and Quality-in-Use](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)
4. [Component (Node) Criticality](#4-component-node-criticality)
   - 4.1 [Definition](#41-definition)
   - 4.2 [User-Side Failure Signature](#42-user-side-failure-signature)
   - 4.3 [The RMAV Model](#43-the-rmav-model)
   - 4.4 [What the Component Carries: the Weight Channel](#44-what-the-component-carries-the-weight-channel)
   - 4.5 [Mapping RMAV to External Quality and Quality-in-Use](#45-mapping-rmav-to-external-quality-and-quality-in-use)
   - 4.6 [Criticality Classification](#46-criticality-classification)
5. [Relationship (Edge) Criticality](#5-relationship-edge-criticality)
   - 5.1 [Definition](#51-definition)
   - 5.2 [Why a Link Needs Its Own Score](#52-why-a-link-needs-its-own-score)
   - 5.3 [Structural Edge Signals](#53-structural-edge-signals)
   - 5.4 [What the Relationship Carries: the Weight Channel](#54-what-the-relationship-carries-the-weight-channel)
   - 5.5 [Edge RMAV Decomposition](#55-edge-rmav-decomposition)
   - 5.6 [Learned Edge Scoring (GNN)](#56-learned-edge-scoring-gnn)
   - 5.7 [Ranking Critical Edges](#57-ranking-critical-edges)
6. [From Score to Stakeholder Narrative](#6-from-score-to-stakeholder-narrative)
   - 6.1 [Worked Example](#61-worked-example)
   - 6.2 [Reading a Score as a Quality-in-Use Statement](#62-reading-a-score-as-a-quality-in-use-statement)
   - 6.3 [Academic Paper Template and LaTeX Snippets](#63-academic-paper-template-and-latex-snippets)
7. [Validity of the Construct](#7-validity-of-the-construct)
   - 7.1 [The Validation Chain Has Three Links](#71-the-validation-chain-has-three-links)
   - 7.2 [Construct Validity](#72-construct-validity)
   - 7.3 [Characteristic Coverage](#73-characteristic-coverage)
   - 7.4 [Real-World Drivers vs. Structural Proxies](#74-real-world-drivers-vs-structural-proxies)
   - 7.5 [External Validity](#75-external-validity)
   - 7.6 [Empirical Threats to Validity Taxonomy](#76-empirical-threats-to-validity-taxonomy)
8. [Where This Fits in the Pipeline](#8-where-this-fits-in-the-pipeline)
9. [References](#9-references)

---

## 1. Overview

Criticality is defined here **from the stakeholder's side**: how much a component or connection matters to the people who depend on the system working — measured by what would go wrong *for them* if it failed, not by how the code is written internally.

That framing is taken directly from the SQuaRE series, specifically **ISO/IEC 25019:2023** (Quality-in-use model) and **ISO/IEC 25010:2023** (Product quality model). The standard explicitly separates **product quality** — measurable *internally* on the artifact at rest and *externally* on the system while it executes — from **Quality-in-Use** (the outcome a *specified stakeholder* experiences while operating the system in a *specified context of use*). Criticality in this project is a Quality-in-Use concept, so it answers one question:

> **If this fails, how much worse does the outcome get for the people who depend on the system?**

Those three views are not interchangeable, and keeping them apart is what makes the rest of this document precise. In one sentence:

> **Criticality is *computed* from internal quality evidence, *validated* against simulated external quality, and *defined* on Quality-in-Use.**

[§3.0](#30-three-quality-views-internal-external-and-quality-in-use) states each view and what this project holds of it; [§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use) binds every RMAV dimension to a named external quality attribute, a dependability attribute, and a Quality-in-Use harm; [§7.1](#71-the-validation-chain-has-three-links) is explicit about which of the three transitions is measured and which are not.

Every entity in the graph — a component (node) or a dependency (edge) — carries a **criticality** signal answering that question as a score. This document is the conceptual home for the concept. It does not re-derive formulas that already live in [structural-analysis.md](structural-analysis.md) and [prediction.md](prediction.md); it defines the terms in stakeholder-facing language ([§3](#3-quality-grounding-square)), states what each score means for a node ([§4](#4-component-node-criticality)) and for an edge ([§5](#5-relationship-edge-criticality)), and is explicit about where the structural proxy stops tracking real Quality-in-Use ([§7](#7-validity-of-the-construct)).

Two distinct but related concepts are in scope:

| Concept | Applies to | Primary output | What the stakeholder experiences |
|:---|:---|:---|:---|
| **Component criticality** | Nodes ($v \in V$: Application, Broker, Topic, Node, Library) | RMAV scores $R(v), M(v), A(v), V(v), Q(v)$ + five-tier classification | The component itself goes away. E.g. MainBroker fails → every application routed through it loses its only path to publish/subscribe; the user's task stops outright, it doesn't merely slow down ([§6.1](#61-worked-example)). |
| **Relationship criticality** | Edges ($e \in E$: physical pub-sub links and derived `DEPENDS_ON` edges) | Structural bridge/betweenness signals + edge RMAV composite + GNN edge score $Q_{\text{GNN}}(u,v)$ | Both components survive, but one specific *link* between them breaks. The stakeholder sees a partial outage — one data flow stops while the rest of the system stays up ([§5.2](#52-why-a-link-needs-its-own-score)). |

The RMAV structural metrics (Reliability, Maintainability, Availability, Vulnerability) used throughout are **proxies** for Quality-in-Use loss — graph-computable stand-ins, used because you cannot survey real stakeholders for every simulated failure. Where their names come from is settled in [§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use), what each one estimates in [§4.5](#45-mapping-rmav-to-external-quality-and-quality-in-use), and how far the proxy can be trusted in [§7](#7-validity-of-the-construct).

Both concepts are scored over the **weighted** graph, never over bare topology: the QoS-derived weights $w(v)$ and $w(e)$ computed in Step 1 carry the declared delivery guarantee of each component and each dependency into every RMAV dimension. Structure says how many stakeholder outcomes route through an element; weight says how strongly each of those outcomes was promised. [§4.4](#44-what-the-component-carries-the-weight-channel) and [§5.4](#54-what-the-relationship-carries-the-weight-channel) trace that channel term by term.

**Four definitions carry the construct.** They are labelled so that downstream work can cite them rather than paraphrase: **D1** ([§4.1](#41-definition)) defines component criticality, **D2** ([§5.1](#51-definition)) relationship criticality, **D3** ([§2.3](#23-consequence-not-risk)) fixes it as a consequence rather than a risk, and **D4** ([§7](#7-validity-of-the-construct)) fixes it as relative rather than absolute. Everything else in this document either motivates those four statements or operationalizes them.

---

## 2. What "Criticality" Means Here

"Criticality" is not a term this project coined, and it is not an ISO/IEC 25019 term either. It is load-bearing in three established traditions whose definitions are mutually incompatible, so a methodology that uses the word owes the reader an account of which one it means. This section gives that account before [§3](#3-quality-grounding-square) builds the Quality-in-Use grounding on top of it.

### 2.1 Three Established Traditions

**Failure-mode criticality (dependability engineering).** In FMECA, criticality is a quantity attached to a failure *mode*, combining the severity of its effect with how often it occurs. MIL-STD-1629A computes a mode criticality number from the failure-effect probability, the mode ratio, the part failure rate, and the operating time; IEC 60812 carries the same structure into current practice. The defining property is that **likelihood is inside the number**.

**Assigned integrity levels (safety and certification).** IEC 61508 (SIL), ISO 26262 (ASIL), DO-178C (DAL) and MIL-STD-882 severity categories attach a criticality level to a *function* on the basis of hazard and risk analysis. The level then governs process obligations — how much verification rigour the function must receive. The defining property is that the level is **assigned by expert judgement against a hazard catalogue**, not computed from an artifact.

**Critical elements in network science.** The critical node (and critical link) detection problem asks which vertices or edges, when removed, most degrade a graph's connectivity — a purely topological optimization, applied to cascading-failure analysis in power grids, transport and communication networks. The defining property is that criticality is **a property of graph structure alone**.

### 2.2 What This Construct Borrows and Rejects

| Tradition | Its "criticality" is | Relation to the construct defined here |
|:---|:---|:---|
| **FMECA / criticality analysis** (MIL-STD-1629A; IEC 60812) | Severity of a failure mode, weighted by failure rate and mode ratio | Shares the severity axis; **rejects** the likelihood axis ([D3](#23-consequence-not-risk)). What this project computes is closer to the factor FMECA would *multiply* by a failure rate than to a criticality number itself |
| **Assigned integrity levels** (IEC 61508 SIL; ISO 26262 ASIL; DO-178C DAL; MIL-STD-882) | A level assigned to a function by hazard analysis, driving process obligations | Complementary rather than competing: those levels encode what the function *is for*, which no architecture graph contains. An ASIL-D function can sit on a structurally MINIMAL component, and that is not a contradiction — the two answer different questions |
| **Critical node / link detection** (CNDP; cascading-failure analysis) | The vertex or edge set whose removal maximally degrades connectivity | The closest relative, and the mechanical basis of the Availability dimension. **Extends** it in two ways: dependencies are *typed* (six derivation rules with distinct failure semantics, [graph-model.md §4.4](graph-model.md#44-phase-4--dependency-derivation)) and *QoS-weighted* ([§4.4](#44-what-the-component-carries-the-weight-channel)), so two topologically identical cut-vertices carrying different guarantees do not score alike |

Stated positively, criticality in this project is: a **pre-deployment**, **architecture-derived**, **consequence-only** estimate of stakeholder harm — **computed** rather than assigned, and **relative** to one system rather than absolute. Each of those five qualifiers is a deliberate exclusion of one of the traditions above, and each is restated formally in D1–D4.

### 2.3 Consequence, Not Risk

> **D3 — Criticality is a consequence, not a risk.** Under the standard decomposition of risk into likelihood and consequence, criticality as defined here is the **consequence factor alone**. No RMAV dimension estimates how probable it is that a component or relationship fails; every dimension estimates how much is lost *given* that it does.

This is the single property that most often gets misread, so its consequences are worth stating explicitly:

- **Comparisons assume equal likelihood.** Ranking component $u$ above component $v$ says that losing $u$ hurts more, not that $u$ is more likely to be lost. Two components with identical criticality are not equally risky if one fails weekly and the other has never failed.
- **Likelihood must be supplied externally.** To turn this into risk, multiply by a failure-rate estimate from operational history, vendor MTTF data, or a reliability model — none of which exists in the topology schema ([§7.4](#74-real-world-drivers-vs-structural-proxies)).
- **It is why the construct is computable pre-deployment.** Consequence follows from architecture, which exists before the system runs; likelihood follows from behaviour, which does not. Restricting the construct to consequence is what makes it available at design time — the tradeoff the whole framework is built on.

The `weight` channel does not violate this. A QoS policy declares how strongly delivery is *guaranteed*, not how often it fails ([§4.4](#44-what-the-component-carries-the-weight-channel)) — it scales the consequence, and contributes no likelihood term.

### 2.4 Comparison with Classical Graph and Software Metrics

To position this methodology rigorously in academic research, the table below compares the RMAV Quality-in-Use proxy with traditional software metrics and unweighted network centralities:

| Metric Family | Canonical Examples | Primary Focus | Limitation in Complex Pub-Sub Architectures | How RMAV Overcomes the Limitation |
|:---|:---|:---|:---|:---|
| **Object-Oriented Software Metrics** | C&K (WMC, CBO, LCOM), Martin Instability ($I$), Cyclomatic Complexity ($v(G)$) | Code-level complexity, intra-module cohesion, and static coupling. | Blind to publish-subscribe decoupling, topic mediation, physical deployment layers, and delivery guarantees. | Integrates static code penalties ($CQP$) as one factor inside Maintainability ($M$), while measuring multi-layer pub-sub dependencies ($G_{\text{analysis}}$). |
| **Information-Flow Metrics** | Henry & Kafura ($IF = (\text{fan-in} \times \text{fan-out})^2$) | Direct information flow between procedure calls. | Assumes synchronous point-to-point calls; fails to model asynchronous fan-out and multi-topic intermediary nodes. | Differentiates afferent/efferent flows via typed directional edges (`PUBLISHES_TO`, `SUBSCRIBES_TO`) and reverse PageRank propagation ($R$). |
| **Unweighted Graph Centrality** | Degree, Betweenness, Eigenvector, Closeness Centrality | Topological prominence in unweighted graphs. | Treats all connections equally regardless of QoS contracts or delivery guarantees; cannot isolate SPOFs from bottlenecks. | Weight-modulates centralities via QoS weights ($w(v), w(e)$) and partitions mechanisms into $R, M, A, V$ profiles. |
| **Network Critical Node Problem (CNDP)** | Cut-vertices, Vertex Connectivity, Fragment size | Global graph fragmentation under vertex removal. | Purely topological; ignores domain semantics, typed dependencies, and partial link outages. | Combines directed articulation point analysis ($AP_{\text{directed}}$) with QoS amplification ($QSPOF$) and relationship partial-outage scoring ($D2$). |

---

## 3. Quality Grounding (SQuaRE)

### 3.0 Three Quality Views: Internal, External, and Quality-in-Use

SQuaRE does not offer one notion of quality but three, distinguished by **what you have to have in front of you in order to measure it**. The distinction is load-bearing here because this project sits on a different view at each stage of its argument, and conflating them is the most common way to overstate what a criticality score establishes.

| SQuaRE view | Measured on | What this project holds of it | Observed? |
|:---|:---|:---|:---|
| **Internal quality** | The artifact at rest — no execution required. Product-quality attributes read from static structure (ISO/IEC 25010:2023 attributes; ISO/IEC 25023 internal measures) | The typed multigraph $G$ itself, plus the ingested `cm_*` static code metrics ([graph-model.md §4.1](graph-model.md#41-phase-1--entity-modeling)) | **Yes — and this is the *entire* input to RMAV** |
| **External quality** | The system while it executes — the same product-quality attributes, read from observable behaviour (ISO/IEC 25023 external measures) | *Declared* as QoS policy on every topic ([graph-model.md §4.3](graph-model.md#43-phase-3--intrinsic-weight-computation)); *simulated* by the discrete-event engines as delivery rate, latency percentiles and contract conformance ([failure-simulation.md](failure-simulation.md)) | **Simulated only — never observed on a running system** |
| **Quality-in-Use** | A specified stakeholder achieving a specified goal in a specified context of use (ISO/IEC 25019:2023 model; ISO/IEC 25022 measures) | Nothing. No user study, expert elicitation, or incident record exists anywhere in this project | **No** |

**Internal quality has two sources, at two granularities, and they contribute very unequally.** "The typed multigraph $G$ plus the ingested `cm_*` metrics" is one row above because both are static evidence, but they come from different instruments and enter scoring through different paths:

| Internal evidence source | Granularity | Instrument | What it measures |
|:---|:---|:---|:---|
| **Graph model (SSA — Static System Analysis)** | System-level | The topology import itself ([graph-model.md](graph-model.md)) | An element's *position*: centralities, directed articulation, bridges, coupling, QoS-derived weights |
| **Static code analysis (SCA)** | Code-level, Application/Library only | Ingested SonarQube-style metrics ([graph-model.md §4.1](graph-model.md#41-phase-1--entity-modeling)) | A component's *internals*: LOC, complexity, cohesion, coupling |

**Measured, not assumed: SCA is a targeted enrichment of one dimension, not a fourth of the evidence.** In the rule-based path, the five code-derived aliases (`loc`, `cyclomatic_complexity`, `lcom`, `coupling_afferent`, `coupling_efferent` — two ingested fields, `cm_avg_cbo` and `cm_avg_rfc`, are never read by any scoring code) feed exactly one composite, the Code Quality Penalty, which feeds exactly one dimension, Maintainability, at coefficient 0.15 ([structural-analysis.md §11.2](structural-analysis.md#112-rmav-formulas)). R, A and V are purely topological (plus declared QoS) — **zero code-derived inputs**. The effective share of code evidence in the composite $Q(v)$ is $0.17 \times 0.15 = 2.55\%$. Edge dimensions are **entirely code-free**: `e_maintainability` deliberately carries no endpoint-M term, so `CQP` never reaches an edge score ([§5.4](#54-what-the-relationship-carries-the-weight-channel)). The learned (GNN) path is structurally different: the same five code features sit on every Application/Library node vector, and a single shared encoder feeds all four RMAV heads, so code evidence reaches $R$, $A$ and $V$ as well as $M$ — and propagates by message passing onto Broker/Node/Topic nodes, which carry no code metrics of their own and which the rule-based path forbids by construction. A reader comparing the two predictors head to head should read this as a genuine architectural difference in what "internal evidence" means to each, not a detail.

> **A corpus/normalization interaction worth flagging precisely, not just noting.** `StructuralAnalyzer._compute_code_quality_metrics`'s min-max helper returns `1.0` for a population with zero variance — a deliberate hardening decision for the case of a single node with real, non-zero metrics (there is no partner value to normalise against). The same branch fires, without distinguishing the two cases, when an entire population is uniformly zero because it carries **no** `code_metrics` at all: every Library in each of the six real-world scenario graphs (Autoware, Train-Ticket, Cloud-microservices) lacks `code_metrics` entirely, so every one of those 28 libraries receives `CQP = 0.70` — a near-maximal code-quality penalty synthesised from no data — flipping the Maintainability tier of **all 28** from what a `0.0` would produce, and (through the shared box-plot fence across the whole component population) the *composite* tier of roughly a fifth to a quarter of every other component in those same graphs. Measured directly: this does **not** move $\rho(Q, I^*)$ on the Application population in any of the three graphs (Δρ = 0.0000 to four decimal places in each), because Spearman correlation depends only on relative order within the correlated population and the shift is a uniform per-type offset. It **does** make the tier reported for every real-world Library, and a meaningful minority of every other real-world component, a fabricated rather than computed signal — a direct contradiction of the "criticality is computed, never asserted" claim ([§4.1](#41-definition)) for exactly the topologies used as the external-validity case studies. A correct fix needs to distinguish "one real data point" from "many nodes, uniformly absent data," which the current guard cannot do from the normalised values alone; that is a scoped follow-up, not a one-line change, and is not undertaken here.

Three further consequences follow immediately, and each is developed later in this document:

- **The predictor is an internal-quality instrument.** Every RMAV input — centralities, articulation scores, coupling terms, code-quality penalties — is a static property of an artifact that has not run. That is precisely what makes criticality available pre-deployment ([D3](#23-consequence-not-risk)), and precisely what bounds it.
- **The oracle is an external-quality instrument.** What the simulator measures — how much of the delivered service is lost when a component fails — is an external product-quality observation, not a Quality-in-Use observation. It is a *model* of the executing system rather than the executing system itself, but it sits on the external axis, and naming it correctly is what lets [§7.1](#71-the-validation-chain-has-three-links) state what has been established and what has not.
- **The construct is defined on the view nobody here measures.** D1 and D2 are Quality-in-Use statements. That is a deliberate choice — criticality *should* be defined by stakeholder harm rather than by whichever quantity happened to be convenient — but it means the definitions reach one view beyond the evidence, and every claim built on them inherits that reach.

> **Why not simply define criticality on external quality, where the evidence is?** Because the resulting construct would not survive a change of deployment. Losing the same broker is a nuisance in one system and a life-safety event in another, and nothing in "delivery rate fell by 40%" distinguishes them. The stakeholder-facing definition is what makes [§3.3](#33-context-of-use-and-domain-context-vector)'s context sensitivity expressible at all. The cost of that choice is the unmeasured transition in [§7.1](#71-the-validation-chain-has-three-links), and this document pays it explicitly rather than hiding it.

### 3.1 What Quality-in-Use Is

ISO/IEC 25019:2023 defines **Quality-in-Use** as the degree to which a product used by specific stakeholders meets their needs to achieve specific goals with **beneficialness**, **freedom from risk**, and **acceptability**, in specific contexts of use. Three properties of that definition drive everything below:

1. **It is measured at the outcome, not at the artifact.** A component is not critical because it is complex or centrally placed; it is critical because losing it degrades an outcome someone cares about.
2. **It is relative to named stakeholders.** "Critical" is meaningless without answering *critical to whom* ([§3.2](#32-stakeholders-primary-secondary-and-indirect)).
3. **It is relative to a named context of use.** The same broker is critical in one deployment and replaceable in another ([§3.3](#33-context-of-use-and-domain-context-vector)).

Under **ISO/IEC 25019:2023**, Quality-in-Use is structured into three primary characteristics and their sub-characteristics:

| Characteristic | Sub-characteristics | Meaning in Stakeholder Terms |
|:---|:---|:---|
| **Beneficialness** | **Usability** (Effectiveness, Efficiency, Satisfaction, Trust, Comfort, Transparency), **Accessibility**, **Suitability** | Degree to which the system delivers positive utility and enables stakeholders to achieve operational goals accurately and efficiently. |
| **Freedom from risk** | **Freedom from economic risk**, **Freedom from health risk**, **Freedom from human life risk**, **Freedom from environmental & societal risk** | Degree to which the system limits potential economic, safety, human, or environmental harm during operational failure. |
| **Acceptability** | **Experience**, **Trustworthiness**, **Compliance** | Degree to which stakeholders respond favorably to system deployment and maintain confidence in its operation and regulatory adherence. |

> **Standard Lineage & Supersession.** In ISO/IEC 25010:2011, Quality-in-Use was defined as five standalone characteristics (Effectiveness, Efficiency, Satisfaction, Freedom from Risk, Context Coverage). ISO/IEC 25019:2023 restructured this into three macro-characteristics. Crucially, **Usability** under ISO 25019:2023 subsumes Effectiveness, Efficiency, and Satisfaction as key measurement dimensions, preserving their internal measurement utility while placing them within the broader Beneficialness framework.

### 3.2 Stakeholders: Primary, Secondary, and Indirect

Quality-in-Use is defined relative to *specified* stakeholders. ISO/IEC 25019:2023 establishes a formal three-tier stakeholder taxonomy, which this framework adopts directly:

1. **Primary Stakeholders (Population 1 — Harmed Direct Users):** Individuals who interact directly with the software to achieve primary goals (e.g., the driver of an autonomous vehicle, the trader executing an order, the clinician monitoring a patient, the customer at an ATM).
2. **Indirect Stakeholders (Population 1 — Harmed Beneficiaries):** Entities or individuals who receive system outputs or are impacted by system outcomes without direct interaction (e.g., patients whose care depends on monitoring feeds, vehicle occupants, transaction counter-parties).
3. **Secondary Stakeholders (Population 2 — Acting Support Roles):** Individuals who support, maintain, administer, secure, or architect the system (e.g., Software Architects, DevOps/SREs, Reliability Engineers, Security Engineers).

| RMAV Dimension | Primary Secondary Stakeholder (Who Acts) | Protected Stakeholders (Whose Quality-in-Use Is Safeguarded) | Primary ISO 25019:2023 Target |
|:---|:---|:---|:---|
| **R** — Reliability | Reliability Engineer | Primary & Indirect Stakeholders | Beneficialness (Usability: Efficiency / Cascade Prevention) |
| **M** — Maintainability | Software Architect | Secondary Stakeholders (Engineering Team) | Beneficialness (Engineering Efficiency & Modifiability) |
| **A** — Availability | DevOps / SRE | Primary & Indirect Stakeholders | Beneficialness (Usability: Effectiveness) & Freedom from Risk |
| **V** — Vulnerability | Security Engineer | Primary, Indirect, & Business Stakeholders | Freedom from Risk (Security) & Acceptability (Trustworthiness) |

A criticality score **routes** a structural signal to the Secondary Stakeholder equipped to remediate it, but the **severity** it encodes is denominated primarily in harm to Primary and Indirect Stakeholders. Maintainability is the one dimension where Secondary Stakeholders are themselves the direct victims of Quality-in-Use degradation (slower incident fixes, elevated regression risk).

### 3.3 Context of Use and Domain Context Vector

The same structural position carries different Quality-in-Use weight in different deployment contexts. To formalize how context modulates criticality in research studies, the Quality-in-Use harm is parametrized by a **Domain Context Vector** $\vec{\omega}_{\text{context}}$:

$$
Q_{\text{QiU}}(v \mid \text{domain}) \;=\; \sum_{c \in \{\text{Ben, Risk, Acc}\}} \omega_{\text{domain}}(c) \cdot \text{Harm}_c(v)
$$

Where $\vec{\omega}_{\text{domain}} = [\omega_{\text{Ben}}, \omega_{\text{Risk}}, \omega_{\text{Acc}}]$ normalizes the relative gravity of the three ISO/IEC 25019 characteristics per deployment domain:

| Scenario domain | Primary stakeholder | Dominant ISO 25019 Characteristic | Context Vector Priority ($\vec{\omega}_{\text{domain}}$) | A CRITICAL score means |
|:---|:---|:---|:---|:---|
| ROS 2 / autonomous vehicle | Vehicle occupants, road users | Freedom from risk (health & life risk) | $\omega_{\text{Risk}} \gg \omega_{\text{Ben}}$ | A sensor/perception path failure is a life-safety hazard |
| Healthcare / clinical HIS | Clinicians, patients | Freedom from risk + Beneficialness (Usability) | $\omega_{\text{Risk}} \approx \omega_{\text{Ben}} > \omega_{\text{Acc}}$ | Lost vitals stream causes stale diagnostic decisions |
| Financial trading (HFT) | Traders, operating firm | Freedom from risk (economic) + Beneficialness | $\omega_{\text{Risk}} > \omega_{\text{Ben}}$ | Added failover latency directly causes financial loss |
| ATM / aviation surveillance | Customers, controllers | Beneficialness (Effectiveness) + Freedom from risk | $\omega_{\text{Ben}} > \omega_{\text{Risk}}$ | Transaction or flight track cannot complete |
| IoT smart city | Residents, city operators | Beneficialness (Efficiency) + Acceptability | $\omega_{\text{Ben}} > \omega_{\text{Acc}}$ | Service degradation across municipal districts |
| Enterprise ESB / microservices | Internal service teams, customers | Beneficialness (Usability) + Acceptability | $\omega_{\text{Ben}} \approx \omega_{\text{Acc}}$ | Slowdown erodes user trust and operational adoption |

### 3.4 The Criticality Questions

Restated as the canonical, user-side definition used throughout this document. A component or relationship is **critical on a characteristic** to the extent that its failure produces the effect in the right-hand column:

| ISO 25019:2023 Characteristic | Measurement Dimension | The Stakeholder's Question | Criticality on that characteristic means |
|:---|:---|:---|:---|
| **Beneficialness** | **Usability (Effectiveness)** | "Can I still achieve my task at all?" | Failure directly prevents a dependent from completing its function or corrupts the result. The task **stops**. |
| **Beneficialness** | **Usability (Efficiency)** | "Does it cost more time or resources to get the result?" | Failure or added latency forces retries, failover, or extra resource spend. |
| **Acceptability** | **Trustworthiness & Experience** | "Do I still trust and adopt this system?" | Repeated or high-profile failures erode stakeholder trust and confidence. |
| **Freedom from Risk** | **Economic, Health, & Life Risk** | "Does this failure cost money, endanger life, or breach compliance?" | Malfunction exposes stakeholders or the operating firm to financial loss, safety hazards, or regulatory breach. |

### 3.5 How the Dimensions Bind to External Quality, Dependability, and Quality-in-Use

The four dimensions this framework scores — Reliability, Maintainability, Availability, Vulnerability — carry names from SQuaRE's **product quality** model (ISO/IEC 25010:2023), while criticality itself is defined on **Quality-in-Use** (ISO/IEC 25019:2023). That dualism is deliberate, and this table is where the two are joined. Each row reads left to right as one causal sentence: *from this static evidence, we estimate the loss of this externally observable attribute, which is this dependability property, whose degradation harms stakeholders in this way.*

**Criticality is not itself a characteristic — it is a characteristic's sensitivity to element loss.** A quality characteristic is a property of the *product*; criticality is a property of an *element within* the product, so it must always be stated *relative to* a named characteristic:

> Criticality is the sensitivity of a named quality characteristic to the loss of an architectural element. This work **instantiates it primarily on Reliability**, with Maintainability and Security as secondary, thinner instantiations, and nine characteristics of the standard entirely out of scope.

#### The primary association: Reliability

ISO/IEC 25010:2023 states that *availability is a combination of faultlessness (which governs the frequency of failure), fault tolerance and recoverability (which govern the length of down time following each failure)*. That composition is not incidental to this framework — it maps exactly onto scoping decisions RMAV already made, independently of the standard:

| Reliability sub-characteristic (ISO/IEC 25010:2023) | Status in this framework | Why |
|:---|:---|:---|
| **Faultlessness** | **Excluded by definitional choice** | The likelihood-bearing sub-characteristic — how *often* the system fails. [**D3**](#23-consequence-not-risk) ("criticality is a consequence, not a risk") *is* the exclusion of Faultlessness, restated here in the standard's own vocabulary. |
| **Fault tolerance** | **= the R dimension** | Operating despite faults ⇒ how far an error propagates through dependents before containment. |
| **Availability** | **= the A dimension** | Readiness for correct service ⇒ structural partition, the state where the task stops outright. |
| **Recoverability** | **Absent — a declared data gap, not modelled** | Needs MTTR, restart semantics, or replication state; no such field exists in the schema ([§7.4](#74-real-world-drivers-vs-structural-proxies)). Every structural SPOF scores alike regardless of how fast it would actually be restored. |

Two sub-characteristics modelled as dimensions, one excluded by an explicit definitional choice, one a declared gap: a complete account of Reliability's four sub-characteristics, at no added computational cost.

#### Secondary associations, per dimension, per characteristic

| RMAV Dimension | Internal evidence (what is computed) | External quality attribute (ISO/IEC 25010:2023) | Dependability attribute | Quality-in-Use harm (ISO/IEC 25019:2023) |
|:---|:---|:---|:---|:---|
| **R — Reliability** | `RPR`, `DG_in`, `CDPot_enh`; `FOC` for Topics | Reliability → **Fault tolerance** | **Reliability** — continuity of correct service. Scores the *error-propagation* stage of the fault→error→failure chain | Beneficialness (Usability: Efficiency) & Acceptability (Trustworthiness) |
| **M — Maintainability** | `BT`, `w_out`, `CQP`, `CouplingRisk_enh`, `(1 − CC)` | Maintainability → **Modularity**, **Modifiability** — 2 of its 5 sub-characteristics (Reusability, Analysability, Testability unmodelled). *Assessed on the artifact; not observable in execution at all* | **Maintainability** — aptitude to undergo modification | Beneficialness (Engineering Efficiency for Secondary Stakeholders) |
| **A — Availability** | `AP_c_directed`, `QSPOF`, `BR`, `CDI`, `w(v)` | Reliability → **Availability** — readiness for correct service (promoted to peer dimension) | **Availability** | Beneficialness (Usability: Effectiveness) & Freedom from Risk |
| **V — Vulnerability** | `REV`, `RCL`, `QADS` | Security → **Confidentiality**, **Integrity** — 2 of its 6 sub-characteristics (Non-repudiation, Accountability, Authenticity, Resistance need mechanisms invisible to topology). *Requires an adversary model, not a fault model* | **Confidentiality + Integrity** | Freedom from Risk (Security/Legal) & Acceptability (Trustworthiness) |
| **— not covered —** | *none* | **Safety** is a first-class ISO/IEC 25010:2023 characteristic (added in the 2023 revision) — none of its five sub-characteristics (Operational constraint, Risk identification, Fail safe, Hazard warning, Safe integration) is addressed | **Safety** — absence of catastrophic consequences | Freedom from **health risk** and **human life risk** |

**The dependability column is not decoration.** It is the vocabulary in which the fault→error→failure chain is stated, and that chain is what separates $R$ from $A$: a component's *fault* is the injected failure, the *error* is what propagates along `DEPENDS_ON` to its dependents (which is what $R$ measures), and the *failure* is the resulting loss of service to a stakeholder (which is what $A$ measures when the loss is total). Two dimensions that would otherwise both read as "something broke" are, in dependability terms, two different stages of one causal chain — and, per the table above, two constituents of the *same* Reliability characteristic rather than two independent attributes ([§3.5 "Orthogonality" below](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)).

> **The last row is a real coverage gap, correctly stated.** RMAV addresses four of the standard's nine characteristics — Reliability (partially, via Fault tolerance and Availability), Maintainability (partially), and Security (partially) — and **Safety is not one of them**, because the schema carries no functional integrity class, no hazard catalogue, and no safety-criticality field ([§7.4](#74-real-world-drivers-vs-structural-proxies)). Five further characteristics are out of scope entirely and undiscussed elsewhere in this document: Functional suitability, Compatibility, Interaction capability, Flexibility, and — worth flagging separately — **Performance efficiency**, where the discrete-event engine already records time behaviour and capacity ([failure-simulation.md §9](failure-simulation.md#9-what-the-simulator-measures-in-quality-model-terms)) but no RMAV dimension consumes them. This is why the ROS 2 / autonomous-vehicle row of [§3.3](#33-context-of-use-and-domain-context-vector) — whose dominant characteristic is *freedom from health and life risk* — is the one domain whose primary harm no dimension estimates directly. A safety-critical deployment can use these scores to find structural exposure; it cannot use them to discharge a safety argument, and nothing in the tiering should be read as though it could. Assigned integrity levels (SIL/ASIL/DAL) remain the complementary instrument for that ([§2.2](#22-what-this-construct-borrows-and-rejects)).

**The two italicised cells explain an otherwise puzzling asymmetry in the evidence.** $M$ and $V$ are the two dimensions the simulation oracles cannot measure behaviourally ([validation.md §3.1](validation.md#31-notation--three-quantities-three-symbols)), and the reason is visible in this column rather than in the implementation: maintainability is not an externally observable attribute — no amount of watching a system run tells you what it costs to change — and security is not a fault-tolerance property, so a fault injector is the wrong instrument for it by construction. [§7.1](#71-the-validation-chain-has-three-links) develops what follows for the validation claim.

**Mathematical Projection Matrix.** The transformation from the four graph-computable Product Quality metrics $\mathbf{s}_{\text{RMAV}}(v) = [R(v), M(v), A(v), V(v)]^T$ into the ISO 25019 Quality-in-Use harm vector $\mathbf{h}_{\text{QiU}}(v) = [H_{\text{Ben}}, H_{\text{Risk}}, H_{\text{Acc}}]^T$ is formally expressed as:

$$
\mathbf{h}_{\text{QiU}}(v) \;=\; \mathbf{M}_{\text{RMAV} \to \text{QiU}} \cdot \mathbf{s}_{\text{RMAV}}(v)
$$

Where the structural mapping matrix $\mathbf{M}_{\text{RMAV} \to \text{QiU}}$ is:

$$
\mathbf{M}_{\text{RMAV} \to \text{QiU}} \;=\; 
\begin{bmatrix} 
0.35 & 0.25 & 0.40 & 0.00 \\
0.10 & 0.00 & 0.50 & 0.40 \\
0.30 & 0.00 & 0.20 & 0.50 
\end{bmatrix}
$$

> **What the projection matrix is and is not.** Its twelve coefficients are a stated design judgement, not an estimated or validated quantity: nothing in this project measures Quality-in-Use, so nothing in this project could have fitted them. They are useful for making the many-to-many correspondence of [§4.5](#45-mapping-rmav-to-external-quality-and-quality-in-use) arithmetic rather than rhetorical, and they should be cited as an operationalization proposal. The same caution that [§4.3](#43-the-rmav-model) applies to the composite weighting — where a sensitivity sweep withdrew the accuracy claim outright — applies here with less evidence behind it, not more.

#### The layered quality model, and a property of the matrix worth stating precisely

The construct is now expressible as four layers, each with a declared epistemic status, implemented in [`saag/core/quality_model.py`](../saag/core/quality_model.py):

| Layer | Standard | Provenance | Oracle |
|:---|:---|:---|:---|
| 0 — Quality Measure Elements | ISO/IEC 25021 | **MEASURED** | — |
| 1 — Internal quality measures | ISO/IEC 25023 | **DERIVED** | — |
| 2 — External quality attributes (R, M, A, V) | ISO/IEC 25010:2023 | **DERIVED** | IR / IM / IA / IS ([validation/dimensions.py](../saag/validation/dimensions.py)) |
| 3 — Quality-in-use weighting | ISO/IEC 25019:2023 | **DECLARED** | — |

Layer 2 is the only layer with an oracle. Layer 3 is where the projection matrix and the Domain Context Vector $\vec{\omega}_{\text{domain}}$ ([§3.3](#33-context-of-use-and-domain-context-vector)) live — and it is **not a further prediction stage stacked on top of Layer 2**. It cannot be one, for a reason that is a property of the matrix's *shape* rather than of its coefficients:

> **Every row of $\mathbf{M}_{\text{RMAV} \to \text{QiU}}$ sums to 1.0.** For any domain weights $\vec{\omega}$ over $\{\text{Ben}, \text{Risk}, \text{Acc}\}$:
> $$
> Q_{\text{QiU}} \;=\; \vec{\omega} \cdot (\mathbf{M}\,\mathbf{s}) \;=\; (\mathbf{M}^{\mathsf T}\vec{\omega}) \cdot \mathbf{s}, \qquad \textstyle\sum(\mathbf{M}^{\mathsf T}\vec{\omega}) = \sum\vec{\omega} = 1
> $$
> A quality-in-use scalarisation is therefore **algebraically identical to scoring the same RMAV vector under a different composite weighting**. There is no such thing as a "quality-in-use score" that ranks components differently from *some* RMAV weighting — and this project's code does not compute one and report it as an independent quantity (`saag/core/quality_model.py` states and pins this property; `tests/test_quality_model.py::TestQiuCollapseEquivalence` enforces it).

**What Layer 3 is, positively stated: a principled generator of context-dependent RMAV weights.** `derive_rmav_weights(domain)` computes $\mathbf{M}^{\mathsf T}\vec{\omega}_{\text{domain}}$ from the ordinal stakeholder priorities of [§3.3](#33-context-of-use-and-domain-context-vector) and returns it as the composite weighting — replacing the AHP pairwise-comparison judgement (whose *accuracy* claim [§4.3](#43-the-rmav-model) already withdrew) with a derivation traceable to a named domain and a named standard. Unlike the AHP vector, this is directly testable: every scenario in the corpus already carries `metadata.domain`.

**Measured** ([reproduce/domain_weight_comparison.py](../reproduce/domain_weight_comparison.py), `results/domain_weight_comparison.json`, 7 synthetic scenarios + the 3 real-world RQ4 graphs, Spearman $\rho$ against $I^*(v)$):

| Weighting | Mean $\rho$ | vs. static default | vs. equal weights |
|:---|:---|:---|:---|
| Static (current default) | 0.284 | — | — |
| Equal (0.25 each) | 0.329 | — | — |
| **Domain-derived** | **0.338** | **+0.054, $p=0.012$ (Wilcoxon, 8/10 wins)** | +0.009, $p=0.85$ (5/10 wins — a coin flip) |
| AHP $\lambda=0.7$ | 0.299 | — | — |

Domain-derived weighting beats the shipped static default with a nominally significant, if under-powered ($n=10$), paired result. Against **equal weights** — the harder baseline, since equal weights already beat the AHP judgement by 0.111 $\rho$ on the shrinkage sweep ([§4.3](#43-the-rmav-model)) — it is statistically indistinguishable. Read plainly: the domain derivation recovers most of what equal weighting already gets for free, non-arbitrarily and with a stated stakeholder rationale, but is not yet evidence that this specific priority encoding out-ranks the simplest possible baseline. Both figures are reported because the honest picture needs both.

**Why two models at once.** A dimension name identifies the *failure mechanism* — the product-quality attribute whose degradation that mechanism represents, observable externally when the system runs. The rightmost column identifies the *harm* — the Quality-in-Use outcome that degrades as a result. Criticality is defined on the harm ([D1](#41-definition)); the decomposition is organized by mechanism because mechanism, not harm, determines the remedy and its secondary stakeholder owner ([§3.2](#32-stakeholders-primary-secondary-and-indirect)).

**Why Availability is promoted.** In ISO 25010:2023, Availability is not a peer of Reliability — it is one of Reliability's four sub-characteristics, alongside Faultlessness, Fault tolerance and Recoverability ([above](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)). It is raised to a peer *dimension* here — a modelling choice about how to organize the scoring, not a claim about the standard's taxonomy — because structural partition is the dominant failure mode in pub-sub architectures and the only mechanism that maps directly to Usability/Effectiveness — the state where a stakeholder's task stops outright rather than costing more. Demoting it back into a single Reliability score would erase exactly the distinction [§4.5](#45-mapping-rmav-to-external-quality-and-quality-in-use) needs: Effectiveness loss (total, from $A$) versus Efficiency loss (partial, from $R$) are different stakeholder harms that a merged score could not separate.

**Metric-level orthogonality, not attribute-level independence.** Each raw structural metric feeds **exactly one** dimension, never two — that discipline is real and holds throughout [§4.3](#43-the-rmav-model) and [§5.5](#55-edge-rmav-decomposition). What it does *not* mean is that R, M, A and V are four independent quality attributes: R and A are both constituents of the single Reliability characteristic (above), so a component scoring high on both is not evidence of two unrelated problems — it is a single characteristic degraded through two distinct mechanisms (propagation and partition). Orthogonal *inputs* still make a component's four-way profile readable as a diagnostic explanation — a pure SPOF scores high on $A$, a god-component scores high on $M$, a cascade hub scores high on $R$ — and that profile is exactly why the *composite* weights $R$ and $A$ rather than treating either as redundant with the other ([§4.3](#43-the-rmav-model)): weighting two constituents of one characteristic is what the standard's own composition calls for, not an arbitrary choice.

---

## 4. Component (Node) Criticality


### 4.1 Definition

> **D1 — Component criticality.** The degree to which the failure, latency, or functional degradation of a specific software component — directly or transitively — reduces the system's capacity to enable its stakeholders to achieve specified operational goals with beneficialness (usability, accessibility, suitability), freedom from risk (economic, health, life, environmental), and acceptability (experience, trustworthiness, compliance) within its operational context.
>
> **Observable form.** That harm is not directly observable. D1 is estimated through the loss of the **external quality attributes** the component sustains — fault tolerance, availability, modifiability, and confidentiality/integrity, the four bound in [§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use) — computed from **internal quality** evidence alone. In dependability terms, D1 scores the consequence of a *fault* at $v$ propagating as an *error* through $v$'s dependents into a service *failure*.
>
> Realised at layer $l$ as a measure: $\;\mathrm{crit}_l : V_l \to [0,1]^4 \times [0,1]$, mapping each component $v \in V_l$ to its four RMAV dimension scores and their composite $Q(v)$.

D1 therefore spans all three quality views at once, and the clause that does that work is the one added above: the *definition* is quality-in-use, the *estimand* is external quality, the *evidence* is internal quality. [§7.1](#71-the-validation-chain-has-three-links) states which of those transitions has been measured.

Five clauses in the main sentence do real work and are easy to skim past:

- **"failure, latency, or functional degradation"** — three distinct fault modes, not one. The structural estimator does not distinguish them; the simulation oracle does ([§7.2](#72-construct-validity)).
- **"directly or transitively"** — the transitive half is the whole reason Reliability exists as a separate dimension, and it is what extends this construct beyond the purely topological critical-node tradition ([§2.2](#22-what-this-construct-borrows-and-rejects)): the harm is loss of stakeholder outcomes reachable through the component, not loss of graph connectivity.
- **"reduces"** — the statement is counterfactual. It says what follows *if* the fault occurs, and never how likely that is ([D3](#23-consequence-not-risk)).
- **"its stakeholders"** — primary, indirect, and secondary stakeholders as defined in [§3.2](#32-stakeholders-primary-secondary-and-indirect). Maintainability is the dimension where the secondary stakeholder (engineering team) is directly affected, keeping change cost inside the definition.
- **"within its operational context"** — the same structure carries different weight in different deployment domains ([§3.3](#33-context-of-use-and-domain-context-vector)), which is why composite coefficients adapt to the system's QoS profile ([§4.4](#44-what-the-component-carries-the-weight-channel)). *Context coverage* as a property of the criticality signal is treated in [§7.3](#73-characteristic-coverage).

The realization line adds one further relativization the sentence leaves implicit: criticality is defined **per layer projection**, not once per component. The same broker scores differently in the `mw` and `system` layers because a different vertex set is being ranked ([D4](#7-validity-of-the-construct)).

Stated as an operational rule: **a component is critical in proportion to how many stakeholder outcomes stop being achievable when it fails, slows, or degrades.**

That single quantity is not directly computable, so it is **decomposed into four dimensions — Reliability, Maintainability, Availability, and Vulnerability/Security (RMAV)** — each capturing one distinct mechanism by which a component's failure destroys stakeholder value:

$$
\text{criticality}(v) \;=\; f\big(\underbrace{R(v)}_{\text{it spreads}},\; \underbrace{M(v)}_{\text{it resists change}},\; \underbrace{A(v)}_{\text{it stops everything}},\; \underbrace{V(v)}_{\text{it invites attack}}\big)
$$

The four dimensions are *not* four separate definitions of criticality; they are four separable causes of the same stakeholder harm, kept apart because each has a different remedy and a different owner ([§3.2](#32-stakeholders-primary-secondary-and-indirect)). Per-dimension definitions are given in [§4.3](#43-the-rmav-model) and their edge counterparts in [§5.5](#55-edge-rmav-decomposition).

**Two inputs, not one.** Each dimension is computed over the **weighted** dependency graph, so criticality is a function of both where a component sits *and* what it carries:

$$
\text{criticality}(v) \;=\; f\big(\;\underbrace{\text{position of } v \text{ in } G_{\text{analysis}}(l)}_{\text{structure — how many outcomes route through it}},\;\; \underbrace{w(v),\; \{w(e) : e \text{ incident to } v\}}_{\text{weight — how much each of those outcomes is guaranteed}}\;\big)
$$

Structure alone cannot separate a SPOF carrying `RELIABLE`/`PERSISTENT` safety telemetry from a topologically identical SPOF carrying `BEST_EFFORT`/`VOLATILE` debug logs; from the stakeholder's side those are not the same failure. The QoS-derived weights $w(v)$ and $w(e)$ computed in Step 1 ([graph-model.md §4.3, §4.5](graph-model.md#43-phase-3--intrinsic-weight-computation)) are what encode that difference, and [§4.4](#44-what-the-component-carries-the-weight-channel) traces exactly where they enter each dimension.

Criticality is computed, not asserted: no component carries a manually assigned criticality label, and no score is hand-tuned per component. It is derived from the component's position in $G_{\text{analysis}}(l)$ (the layer-projected dependency graph produced by [graph-model.md](graph-model.md)) together with the weights derived from its declared QoS policies. The QoS policy *is* an authored input — so criticality inherits the delivery guarantees the architect declared, while remaining independent of any opinion the architect holds about what is important ([§4.4](#44-what-the-component-carries-the-weight-channel) closes with what that dependency costs).

### 4.2 User-Side Failure Signature

Before any formula, each characteristic has a recognizable failure signature for a component. This is what the score is trying to detect:

| Characteristic | Component-failure signature the stakeholder observes |
|:---|:---|
| **Effectiveness** | A function becomes unreachable. The dependent has no alternative route, so its task returns nothing or returns a wrong/stale result. Structurally this is a **single point of failure**. |
| **Efficiency** | The task still completes, but through retries, a failover path, or a degraded mode — more time and more resource per unit of delivered value. Structurally this is **cascade reach** and **coupling cost**. |
| **Satisfaction** | The stakeholder starts routing around the system, adding manual checks, or escalating — the loss is confidence, and it outlives the incident. |
| **Freedom from risk** | The failure window itself is the harm: an undetected safety excursion, an unbookable transaction, an exposed data path. |
| **Context coverage** | The above holds in every deployment of this topology, not just the one that happened to be measured. |

### 4.3 The RMAV Model

Component criticality is decomposed into four orthogonal structural dimensions — **Reliability, Maintainability, Availability, Vulnerability (RMAV)** — combined into a composite score $Q(v)$. The full formulas, weights, and derivations are defined in [structural-analysis.md §11](structural-analysis.md#11-analyze-stage--rule-based-rmav-scoring); the summary, with each dimension tied to the characteristic it estimates:

| Dimension | Question Answered | Driven Primarily By | Estimates (§3.4) |
|:---|:---|:---|:---|
| **R — Reliability** | How broadly/deeply does failure propagate? | Reverse PageRank *(QoS-weighted)*, in-degree, Cascade Depth Potential | Efficiency, Satisfaction |
| **M — Maintainability** | How hard is this to change safely? | Betweenness *(QoS-weighted)*, QoS-weighted efferent coupling, Code Quality Penalty | Efficiency (engineering-side) |
| **A — Availability** | Is this a structural single point of failure? | Directed articulation point score, bridge ratio, QoS-SPOF *(uses $w(v)$)* | Effectiveness, Freedom from risk |
| **V — Vulnerability** | How attractive a target is this for attack? | Reverse eigenvector *(QoS-weighted)*, reverse closeness, QoS-weighted in-degree | Freedom from risk, Satisfaction |

The italicised terms are where the Step 1 weights enter; [§4.4](#44-what-the-component-carries-the-weight-channel) traces each one.

**The four dimensions partition the mechanisms, not the harm.** Each is [D1](#41-definition) narrowed to one route by which a fault reaches stakeholders, which is why they can be scored independently and read as a profile: no raw metric feeds two dimensions ([§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)), so a component's shape across the four names the mechanism at work. The *harm* is not partitioned — two dimensions can threaten the same Quality-in-Use characteristic, and [§4.5](#45-mapping-rmav-to-external-quality-and-quality-in-use) reads that correspondence in the opposite direction.

> **Terminology — Vulnerability vs. Security.** The serialized field in `QualityScores` and `CriticalityRanking` is named **`security`** ([saag/core/metrics.py#L243-L258](../saag/core/metrics.py#L243-L258)), and a **high** `security` score means **worse** (more exposed). The two are the same dimension viewed from opposite ends; [§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use) explains why the concept keeps the name $V$.

#### Component criticality per dimension

Each dimension is **[D1](#41-definition) restricted to one mechanism** — not a fifth and sixth definition of criticality. D1 fixes the harm; a dimension fixes the route by which a fault produces it. Metric inputs are listed in full; their coefficients live in [structural-analysis.md §11](structural-analysis.md#11-analyze-stage--rule-based-rmav-scoring).

**D1.R — Reliability criticality (component)**
> D1 restricted to **transitive propagation**: the degree to which a component's failure, latency or degradation reaches stakeholders *through* the components that depend on it, converting one local fault into a multi-component loss of goals.

| | |
|:---|:---|
| Stakeholder question | "When this breaks, how much else breaks with it?" |
| High score means | The component sits upstream of many transitive dependents; a fault reaches far and deep |
| Metric inputs | `RPR`, `DG_in`, `CDPot_enh` (itself consuming `DG_out` and `MPCI`). Topics instead use `FOC` and the publisher-count norm |
| External quality attribute | Reliability → **Fault tolerance** (ISO/IEC 25010:2023); dependability attribute **Reliability**. Scores the *error-propagation* stage of the fault→error→failure chain — how far an error travels before it is contained |
| Quality-in-Use effect | **Efficiency** (dependents retry/fail over), then **Satisfaction** (repeated cascades erode trust) |
| Acted on by | Reliability Engineer — bulkheads, circuit breakers, cascade containment |

**D1.M — Maintainability criticality (component)**
> D1 restricted to **change cost**: the degree to which a component resists safe modification, so that every change to it carries disproportionate regression risk for the stakeholders who must keep the system working.

| | |
|:---|:---|
| Stakeholder question | "How expensive and risky is it to change or fix this?" |
| High score means | A structural bottleneck with high QoS-weighted fan-out coupling and poor internal code quality |
| Metric inputs | `BT`, `w_out`, `CQP` (`complexity_norm`, `instability_code`, `lcom_norm`), `CouplingRisk_enh` (consuming `path_complexity`), `(1 − CC)` |
| External quality attribute | Maintainability → **Modularity**, **Modifiability** (ISO/IEC 25010:2023); dependability attribute **Maintainability**. The one dimension whose attribute is *not externally observable*: watching the system execute never reveals what a change to it would cost, so $M$ is an internal-quality estimate of an internal-quality attribute |
| Quality-in-Use effect | **Efficiency**, uniquely on the secondary engineering stakeholder ([§3.2](#32-stakeholders-primary-secondary-and-indirect)) rather than the primary direct user — slower fixes, longer incident recovery |
| Acted on by | Software Architect — decoupling, interface extraction, refactoring |

**D1.A — Availability criticality (component)**
> D1 restricted to **structural partition**: the degree to which a component's failure removes the *only* path by which its dependents reach what they need, so that stakeholder goals become unachievable rather than merely more expensive.

| | |
|:---|:---|
| Stakeholder question | "If this is down, does anything still work?" |
| High score means | Removing the node disconnects a subgraph — there is no redundant route around it |
| Metric inputs | `AP_c_directed`, `QSPOF` (= `AP_c_directed × w(v)`), Bridge Ratio `BR`, `CDI`, `w(v)` |
| External quality attribute | Reliability → **Availability** — readiness for correct service (ISO/IEC 25010:2023); dependability attribute **Availability**. Scores the *failure* stage of the chain: the point at which propagated error becomes service the stakeholder cannot obtain at all |
| Quality-in-Use effect | **Effectiveness** — the only dimension where the stakeholder's task stops outright — plus **Freedom from risk** (the outage window is itself the harm) |
| Acted on by | DevOps / SRE — redundancy, failover, replication |

**D1.V — Vulnerability / Security criticality (component)**
> D1 restricted to **compromise rather than failure**: the degree to which a component is an attractive and reachable target, such that an attacker owning it gains disproportionate reach over the guarantees stakeholders depend on.

| | |
|:---|:---|
| Stakeholder question | "If an attacker owns this, how much do they own?" |
| High score means | Centrally reachable on $G^\top$, with many high-QoS flows converging on it |
| Metric inputs | `REV` (reverse eigenvector), `RCL` (reverse closeness), `QADS` (= `w_in`) |
| External quality attribute | Security → **Confidentiality**, **Integrity** (ISO/IEC 25010:2023); dependability attributes **Confidentiality + Integrity**. Unlike R and A this is not a fault-tolerance property at all: it presumes an adversary rather than a fault, which is why no fault-injection oracle can measure it ([§7.1](#71-the-validation-chain-has-three-links)) |
| Quality-in-Use effect | **Freedom from risk** (security/legal/compliance exposure) and **Satisfaction** (confidence loss even without an incident) |
| Acted on by | Security Engineer — hardening, segmentation, access control |

The dimensions are scored and classified **independently** ([§4.6](#46-criticality-classification)), so a component carries a four-way profile rather than one label — the profile, not the composite, is what identifies *which* kind of criticality is present and therefore which remedy applies.

$$
Q(v) = 0.43 \cdot A(v) + 0.24 \cdot R(v) + 0.17 \cdot M(v) + 0.16 \cdot V(v)
$$

These are the static `QualityWeights` defaults — the vector actually used when `use_ahp=False`, which is the default and the configuration behind the reported results. Two qualifications travel with it:

- **An AHP path exists and does not produce this vector.** With `use_ahp=True`, `AHPProcessor` computes weights from the comparison matrices and blends them toward a uniform prior by a shrinkage factor $\lambda$ (default 0.7). At $\lambda = 1$ the composite is $(0.458, 0.246, 0.169, 0.128)$ and at $\lambda = 0.7$ it is $(0.395, 0.247, 0.193, 0.165)$ — neither equals the static vector above. The AHP path is used for the shrinkage sensitivity sweep, not for production scoring; do not describe the static vector as "the AHP weights".
- **QoS-profile adaptation is applied on top** (`adapt_qos_weights`, on by default), so the *effective* composite is per-system — see [§4.4](#44-what-the-component-carries-the-weight-channel).

**Why $A$ is weighted highest, in Quality-in-Use terms.** Availability is the only dimension that maps onto **Effectiveness** — the characteristic where the stakeholder's task stops outright rather than costing more. A structural SPOF partitions the graph *deterministically*, whereas cascade propagation (R), coupling risk (M), and attack exposure (V) are *probabilistic* and typically surface as Efficiency or Freedom-from-risk loss. Weighting $A$ highest is therefore a statement about stakeholder harm — total loss of a goal outranks a more expensive path to it — not merely a statement about graph topology.

> **This is a mechanism argument, not an accuracy claim.** Measured against simulated impact, the stated weighting does *not* improve ranking: a sweep of the shrinkage parameter shows a monotone decline with no plateau, and equal dimension weights outperform every weighting that expresses the intended ordering. The decomposition is retained as an *attribution* mechanism — the four dimensions have distinct remedies and distinct owners — and the accuracy claim attached to the weighting is withdrawn. A consumer optimising purely for ranking should use equal weights.

See [saag/core/criticality.py](../saag/core/criticality.py) for the `CriticalityRanking` DTO that carries these scores through the pipeline.

### 4.4 What the Component Carries: the Weight Channel

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

Full derivation of $w(t)$, $w(v)$ and $w(e)$ is in [graph-model.md §4.3–§4.5](graph-model.md#43-phase-3--intrinsic-weight-computation). What matters here is the semantics carried along: **$w$ is a delivery-guarantee proxy** — how strongly the system promises that this data arrives — not a measure of traffic volume, revenue, or safety class ([§7.4](#74-real-world-drivers-vs-structural-proxies)).

#### Where weight enters each dimension

Every RMAV dimension is weight-aware, but through different mechanisms. Coefficients are the AHP defaults from [structural-analysis.md §11](structural-analysis.md#11-analyze-stage--rule-based-rmav-scoring):

| Dimension | Weight-bearing term | Mechanism | Reading |
|:---|:---|:---|:---|
| **R** | `RPR` (0.45) | Reverse PageRank uses $w(e)$ as edge *importance* on $G^T$ | Cascade reach is measured along the strongly-guaranteed paths, so a fault that would travel over `RELIABLE`/`PERSISTENT` dependencies scores higher than one travelling over best-effort links |
| **M** | `w_out` (0.30), `BT` (0.35) | `w_out(v) = Σ w(e)` over efferent edges, rank-normalised; betweenness uses $1/w(e)$ as *distance* | Depending on many high-guarantee components makes change expensive — each is an SLA obligation that a change must not break |
| **A** | `QSPOF` (0.25), `w(v)` (0.05) | $\text{QSPOF} = \text{AP\_c\_directed} \times w(v)$, plus a direct additive term | A SPOF is amplified in proportion to what it guarantees: a partition on high-QoS traffic is a worse Effectiveness loss than the same partition on volatile traffic |
| **V** | `QADS` = `w_in` (0.25), `REV` (0.40) | `w_in(v) = Σ w(e)` over afferent edges, rank-normalised; reverse eigenvector uses $w(e)$ as importance | A component is an attractive target in proportion to the guarantees its dependents rely on — compromising it breaks the promises that matter most |

Two structural companions of the weight travel with it:

- **`path_count`** — the number of topics (or shared nodes) mediating one `DEPENDS_ON` edge ([graph-model.md §3](graph-model.md#3-formal-graph-definition)). It is deliberately *not* folded into $w(e)$ — that would break the $w \in [0,1]$ contract — so it enters criticality separately: through `MPCI`, which *amplifies* the cascade-depth term of $R$ (three shared topics between the same pair are three simultaneous failure vectors, a tighter coupling than three independent single-topic links), and through `path_complexity`, which raises $M$'s coupling-risk term. Note the node/edge duality: multiplicity makes the *pair* more fragile while making each *individual* channel more replaceable ([§5.3](#53-structural-edge-signals)).
- **Topic fan-out** — `subscriber_count` / `publisher_count`, computed in Phase 2, drive `FOC` and the Topic-specific form of $R$. Note that Topic reliability is scored from fan-out and message *frequency*, not from $w(t)$: $w(t)$ reaches Topics' neighbours instead, through the aggregation above.

#### Two different things called "weight"

The word is overloaded in this project and the two senses never mix:

| | Graph weight | Dimension weight |
|:---|:---|:---|
| Written | $w(v)$, $w(e)$ | $q_R, q_M, q_A, q_V$ and the per-metric coefficients |
| Comes from | Declared QoS policy + message size, per component/edge | AHP pairwise comparison of the dimensions, once for the whole model |
| Varies | Per component and per edge, within one system | Per model configuration, not per component |
| Answers | "How strongly is *this* flow guaranteed?" | "How much does *this kind* of criticality count?" |

They meet in exactly one place: the composite coefficients $q_*$ are themselves adapted to the system's aggregate QoS profile before scoring (`adapt_qos_weights`, on by default). A system whose topics are predominantly `PERSISTENT`/`RELIABLE`/high-priority shifts weight toward $R$ and $A$; a predominantly `VOLATILE`/`BEST_EFFORT` system shifts it toward $M$ and $V$ — the same structural graph therefore yields a different composite ranking in a mission-critical deployment than in a best-effort one, which is [§3.3](#33-context-of-use-and-domain-context-vector)'s context-of-use argument made computable.

#### What this dependency costs

- **Weights are relative, not absolute.** $w(v)$, `w_in` and `w_out` are rank-normalised across the components of the analysed layer before entering RMAV, so the weight channel expresses *ordering within one system*, matching the relative reading of the tiers in [§4.6](#46-criticality-classification). Raising every topic in a system to `RELIABLE`/`PERSISTENT` does not raise everything's criticality; it only flattens the channel's ability to discriminate.
- **A mis-declared QoS policy is a mis-scored component.** If the topology declares `BEST_EFFORT` on a flow the business actually treats as critical, the weight channel faithfully reproduces that error. This is the one place where criticality inherits an authored judgement, and it is the first thing to check when a score contradicts operational experience.
- **The floor is not zero.** $w(t) \geq 0.01$ by construction, so an unguaranteed flow still contributes structure; the weight channel modulates criticality, it never switches it off.

### 4.5 Mapping RMAV to External Quality and Quality-in-Use

[§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use) read this correspondence one way — from each dimension to the attribute and harm it estimates. This table reads it the other way, from each Quality-in-Use characteristic back to the dimensions that operationalize it, which is the direction a stakeholder asking "what threatens my ability to finish the task?" needs. The middle column names the **external quality attribute** whose loss is the mechanism connecting the two:

| Quality-in-Use characteristic | Loss of which external quality attribute produces it | Primarily operationalized by | Why |
|:---|:---|:---|:---|
| **Effectiveness** | **Availability** | **A — Availability** | A structural SPOF's removal partitions the graph — dependents cannot complete their function at all. |
| **Efficiency** | **Fault tolerance**; **Modifiability** | **R — Reliability**, **M — Maintainability** | Cascades (R) force retries/failover; tight coupling (M) means every change or incident costs more engineering effort per unit of value delivered. |
| **Satisfaction** | **Fault tolerance**; **Confidentiality/Integrity** | **R + V** | Repeated cascading outages erode trust (R); being a high-value attack target (V) undermines confidence even absent an actual incident. |
| **Freedom from risk** (economic, operational, security) | **Availability**; **Confidentiality/Integrity**; **Fault tolerance** | **A + V** (dominant), **R** | Availability quantifies economic/operational risk (SPOF = certain partition); Vulnerability quantifies security/legal risk (breach exposure); Reliability quantifies propagation risk. |
| **Freedom from risk** (health, human life) | **Safety** | *— none —* | The dependability attribute RMAV does not cover ([§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)). Structural exposure is reported; hazard severity is not, and cannot be inferred from topology. |
| **Context coverage** | *(not an attribute — a property of the ranking)* | Cross-scenario/cross-domain stability of the score | A component's criticality ranking should hold across topologies and domains; instability here is a weakness of the *criticality signal itself*, checked via the per-domain repeated stratified k-fold evaluation and multi-scenario batch runs (`cli/run_scenarios.sh`). |

The mapping is many-to-many by design: no single RMAV dimension is a characteristic, and no characteristic is fully captured by one dimension. That is why this table and the per-dimension sub-definitions ([D1.R–D1.V](#43-the-rmav-model), [D2.R–D2.V](#55-edge-rmav-decomposition)) are the *same* correspondence read in opposite directions, not two competing claims: those state which harm each mechanism produces, this states which mechanisms produce each harm. Coverage gaps are enumerated in [§7.3](#73-characteristic-coverage).

### 4.6 Criticality Classification

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
- **Per-dimension tiers are the diagnostic.** Classification is applied independently per RMAV dimension and for the composite. A component can be CRITICAL on Availability while MINIMAL on Vulnerability — which reads, in the language of §3.4, as "this threatens Effectiveness but not Freedom from risk," and directs remediation accordingly.

---

## 5. Relationship (Edge) Criticality

### 5.1 Definition

> **D2 — Relationship criticality.** The degree to which the disruption, latency, or data loss across a specific inter-component interaction or dependency path — with both endpoint components remaining operational — reduces the system's capacity to enable its stakeholders to achieve specified goals with beneficialness (usability, accessibility, suitability), freedom from risk (economic, health, life, environmental), and acceptability (experience, trustworthiness, compliance), in proportion to the absence of redundant or fallback paths around it.
>
> **Observable form.** As with [D1](#41-definition), the harm is estimated through the loss of **external quality attributes** — the same four bound in [§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use), scoped to one channel — computed from **internal quality** evidence alone. In dependability terms a severed link is a *fault* whose *error* is confined to a single channel; the resulting *failure* is therefore **partial** loss of service rather than total, which is what separates the external signature of an edge failure from that of a node failure ([§5.2](#52-why-a-link-needs-its-own-score)).
>
> Realised at layer $l$ as a measure: $\;\mathrm{crit}_l : E_l \to [0,1]^4 \times [0,1]$, mapping each dependency $e = (u,v) \in E_l$ to the same signature as [D1](#41-definition).

Three clauses distinguish D2 from D1 rather than merely restating it for edges:

- **"with both endpoint components remaining operational"** — this is the discriminating clause. It isolates the partial-outage case ([§5.2](#52-why-a-link-needs-its-own-score)), and it is precisely the condition the simulator enforces when it measures edge impact by severing one relationship and leaving both endpoints active ([§5.6](#56-learned-edge-scoring-gnn)).
- **"inter-component interaction or dependency path"** — the two spellings are deliberate, because relationship criticality spans both graph views: raw structural interactions (`PUBLISHES_TO`, `ROUTES`, `RUNS_ON`, …) and the derived `DEPENDS_ON` paths abstracted from them — the two views of [graph-model.md §6](graph-model.md#6-two-graph-views). A single `DEPENDS_ON` edge may stand for several underlying interactions, which is what `path_count` records ([§5.3](#53-structural-edge-signals)).
- **"in proportion to the absence of redundant or fallback paths"** — replaceability scales the harm rather than gating it. A link with no alternative route destroys Usability/Effectiveness for everything behind it; a replaceable one still costs Usability/Efficiency, and still carries whatever guarantee crosses it. That proportionality is exactly how the four edge dimensions divide the work: only Availability is bridge-gated, while Reliability, Maintainability and Vulnerability score replaceable links too ([§5.5](#55-edge-rmav-decomposition)).

Where component criticality asks *"how dangerous is losing this component,"* relationship criticality asks *"how dangerous is losing this specific link, even though both components are still running."*

It is decomposed along the **same four RMAV dimensions** as a component ([§4.1](#41-definition)), scoped to the link rather than the endpoint:

$$
\text{criticality}(u,v) \;=\; f\big(\underbrace{R(u,v)}_{\text{it conducts faults}},\; \underbrace{M(u,v)}_{\text{it binds the two sides}},\; \underbrace{A(u,v)}_{\text{it is the only route}},\; \underbrace{V(u,v)}_{\text{it is a path in}}\big)
$$

Using one dimension set for both nodes and edges is deliberate: it makes the two comparable in a single ranking, and it lets a remediation owner ([§3.2](#32-stakeholders-primary-secondary-and-indirect)) read node and edge findings in the same vocabulary — an SRE reads $A$ on both, a Security Engineer reads $V$ on both. Per-dimension edge definitions are in [§5.5](#55-edge-rmav-decomposition).

As with a component ([§4.1](#41-definition)), each dimension is computed over the weighted graph, so a link's criticality combines its position with the guarantees of the flow crossing it:

$$
\text{criticality}(u,v) \;=\; f\big(\;\underbrace{\text{position of } (u,v) \text{ in } G_{\text{analysis}}(l)}_{\text{is this link replaceable?}},\;\; \underbrace{w(u,v),\; \text{path\_count}(u,v)}_{\text{what does it guarantee, over how many channels?}},\;\; \underbrace{R,M,A,V \text{ of } u \text{ and } v}_{\text{what does it connect?}}\;\big)
$$

The third argument is what distinguishes an edge score from a node score: an edge is scored *in the context of its endpoints* — it can be no more reliable than its riskiest end, nor more available than its weakest ([§5.5](#55-edge-rmav-decomposition)). The second is traced in [§5.4](#54-what-the-relationship-carries-the-weight-channel).

### 5.2 Why a Link Needs Its Own Score

From the stakeholder's side, an edge failure and a node failure produce different observable symptoms, which is why a separate score is warranted rather than inheriting endpoint scores:

- **A node failure is a total outage of a capability.** Everything the component provides stops.
- **An edge failure is a partial outage.** The component is up, its dashboards are green, its other consumers are fine — but *one* data flow has stopped. For the stakeholder on the far end of that link, Effectiveness is lost just as completely as in a full outage, while the operator sees a healthy system.

This asymmetry produces the two cases the model must handle:

- A **high-criticality node** can have many **low-criticality edges** — a redundantly connected broker, where losing any single link changes nothing for anyone.
- A **low-criticality node** can sit behind a **single highly critical bridge edge** — losing that one relationship is as consequential for its dependents as losing a much higher-scoring component.

Edge criticality is therefore governed by one structural question with a direct Quality-in-Use reading: **is this link replaceable?** A replaceable link degrades Efficiency (traffic reroutes, costs more). A non-replaceable link — a graph bridge — destroys Effectiveness for everything behind it. This is the same claim D2 makes as *"in proportion to the absence of redundant or fallback paths"* ([§5.1](#51-definition)); the difference between the two outcomes is a difference of degree, which is why replaceability scales the score instead of switching it on.

### 5.3 Structural Edge Signals

Relationship criticality is assembled from per-edge structural signals computed in [saag/analysis/structural_analyzer.py](../saag/analysis/structural_analyzer.py) and carried by `EdgeMetrics` / `EdgeQuality` in [saag/core/metrics.py](../saag/core/metrics.py):

- **`is_bridge`** — whether the edge is a graph bridge (cut-edge): `nx.bridges()` over the undirected projection. Removing a bridge disconnects a subgraph from the rest of the system — the Effectiveness case above.
- **`betweenness`** — edge betweenness centrality (`nx.edge_betweenness_centrality`) computed over the **inverted-weight** graph, where each edge's length is $1/w(e)$: the fraction of shortest dependency paths that traverse this specific edge — the Efficiency case (how much traffic must reroute). Inversion is what makes strongly-guaranteed dependencies *short*, so they attract the shortest paths rather than repelling them.
- **`weight`** — $w(e)$, the edge's QoS-derived weight from [graph-model.md](graph-model.md): the worst-case guarantee over every topic mediating this dependency, a proxy for how strongly the flow across it is promised. Traced through the edge dimensions in [§5.4](#54-what-the-relationship-carries-the-weight-channel).
- **`path_count`** — how many distinct topics (or shared nodes) establish this one edge. Deliberately kept out of $w(e)$ to preserve $w \in [0,1]$, so it acts as a separate coupling-intensity signal.

These are distinct from two *node-level* metrics that are easy to mistake for edge scores because they are edge-derived:

- **Bridge Ratio `BR(v)`** ([structural-analysis.md §9.9](structural-analysis.md#99-bridge-ratio-br)) — the *fraction of a node's own connections* that are bridges. It describes a node's exposure to non-redundant edges, not a per-edge score.
- **Multi-Path Coupling Index `MPCI(v)`** ([structural-analysis.md §9.3](structural-analysis.md#93-multi-path-coupling-index-mpci)) — counts *redundant* shared channels feeding into a node. High MPCI means a node's incoming edges are collectively low-criticality (multi-channel, no single edge is a SPOF); low MPCI (with high `DG_in`) means each incoming edge is closer to a single point of failure for that dependency.

### 5.4 What the Relationship Carries: the Weight Channel

An edge's weight is not an attribute of the link's shape — it is the guarantee attached to the data crossing it, and it is computed in Step 1 before any criticality is scored:

```
w(e) for a structural edge      = w(t) of the topic it attaches to      (Phase 3, inheritance)
w(e) for a DEPENDS_ON edge      = max w(t) over the mediating topics    (Phase 4, worst case)
      app_to_lib                = w(consuming component)                 (Phase 5b)
      broker_to_broker          = w(shared node)                         (Phase 5b)
path_count(e)                   = number of mediating topics / shared nodes
```

Taking the worst case rather than an average is the conservative reading required by the definition in [§5.1](#51-definition): if *any* strongly-guaranteed flow crosses this link, losing the link breaks that guarantee, regardless of how many weak flows it also carries.

#### Where weight enters each edge dimension

Unlike a component, an edge uses $w(e)$ **directly, un-normalised** — the Step 1 contract already puts it in $[0,1]$, so no rank normalisation is needed. Coefficients are the `e_*` defaults in [`QualityWeights`](../saag/analysis/weight_calculator.py):

| Edge dimension | Weight-bearing term | Reading |
|:---|:---|:---|
| **R** | $w(e)$ at 0.30 — second only to betweenness | A link conducts faults in proportion to what it promises to deliver; a `RELIABLE`/`PERSISTENT` channel that fails breaks a promise a dependent was entitled to rely on |
| **M** | $w(e)$ at 0.15 | A high-guarantee contract is expensive to renegotiate — both sides must move together |
| **A** | **none** | Availability asks only *"is this link replaceable?"*, and redundancy is a topological fact: a bridge is a bridge whether it carries safety telemetry or debug logs. In quality-model terms the two inputs sit on different views — **replaceability is a structural property of the system** (internal quality evidence for an availability claim), while **$w(e)$ is a declared external quality requirement of one flow** ([graph-model.md §4.3](graph-model.md#43-phase-3--intrinsic-weight-computation)). Multiplying them would let a declared requirement change a topological fact. QoS amplification reaches $A$ indirectly instead, through the endpoints' own `QSPOF` ([§4.4](#44-what-the-component-carries-the-weight-channel)) |
| **V** | $w(e)$ at 0.15 | A channel is worth attacking in proportion to the value of what flows over it |
| *(via betweenness, in R and M)* | $1/w(e)$ as path length | Strongly-guaranteed edges sit on more shortest paths, raising the betweenness of the links the system leans on |

`path_count` does not enter the edge score directly. It shapes the *endpoints'* $R$ and $M$ ([§4.4](#44-what-the-component-carries-the-weight-channel)), and of those only $R$ can reach the edge again — through the `max(source.R, target.R)` term of edge $R$. Edge $M$ carries no endpoint term at all, so a link's maintainability score is blind to how coupled its endpoints already are.

> **Reading note — edge dimension scales.** The four edge dimensions do not draw on the same coefficient mass ($R$ sums to 0.85 of a possible 1.0, $M$ to 0.80, $A$ to 0.50, $V$ to 0.35), so raw edge scores are comparable *within* a dimension but not *across* dimensions. Since classification is box-plot relative within the edge set ([§4.6](#46-criticality-classification)), per-dimension rankings and tiers are unaffected; only the raw magnitudes are, and the composite mixes the dimensions at these differing scales. Read an edge's dimension tiers, not its absolute dimension values.

### 5.5 Edge RMAV Decomposition

Just as component criticality is decomposed into RMAV ([§4.3](#43-the-rmav-model)), each edge is scored on the same four dimensions in [`_score_and_classify_edges`](../saag/analysis/analyzer.py#L412-L483) — an edge is not reduced to a single number, but assessed as reliability, maintainability, availability, and vulnerability risks in its own right, blending the edge's intrinsic structural signals ([§5.3](#53-structural-edge-signals)) with its endpoints' own RMAV scores:

| Dimension | Question Answered for an Edge | Formula (edge-intrinsic + weight channel + endpoint context) |
|:---|:---|:---|
| **R — Reliability** | How much does this specific link contribute to fault propagation? | `0.35·betweenness + 0.30·w(e) + 0.20·max(source.R, target.R)` |
| **M — Maintainability** | How much does this link add to coupling/change cost? | `0.35·betweenness + 0.30·is_bridge + 0.15·w(e)` |
| **A — Availability** | Does losing this specific link partition the graph? | `0.30·is_bridge + 0.20·min(source.A, target.A)` |
| **V — Vulnerability** | How much does this link expand the attack surface? | `0.15·w(e) + 0.20·max(source.V, target.V)` |

#### Relationship criticality per dimension

Each dimension is **[D2](#51-definition) restricted to one mechanism**, exactly parallel to the component sub-definitions in [§4.3](#43-the-rmav-model) — same four mechanisms, scoped to the link instead of the endpoint:

**D2.R — Reliability criticality (relationship)**
> D2 restricted to **transitive propagation**: the degree to which a relationship acts as the conductor of a fault — the channel along which disruption, latency or data loss at one endpoint reaches the stakeholders behind the other.

| | |
|:---|:---|
| Stakeholder question | "If the upstream side breaks, does this link carry the damage downstream?" |
| High score means | A heavily traversed link whose riskiest endpoint has wide blast radius |
| Metric inputs | Edge betweenness, $w(e)$, `max(source.R, target.R)` |
| External quality attribute | Reliability → **Fault tolerance**; dependability attribute **Reliability**. The link is the *conduit* of the error rather than its origin — the stage of the chain at which a contained fault stops being contained |
| Quality-in-Use effect | **Efficiency** (dependents on the far side retry or fail over), then **Satisfaction** (repeated cascades erode trust) |
| Acted on by | Reliability Engineer — timeouts, circuit breakers, backpressure on this specific flow |

**D2.M — Maintainability criticality (relationship)**
> D2 restricted to **change cost**: the degree to which a relationship binds its two endpoints together, so that a change on one side forces a coordinated change on the other.

| | |
|:---|:---|
| Stakeholder question | "Can either side of this link evolve independently?" |
| High score means | A non-redundant, heavily used link — the contract across it cannot be changed unilaterally or routed around |
| Metric inputs | Edge betweenness, `is_bridge`, $w(e)$ |
| External quality attribute | Maintainability → **Modularity**, **Modifiability**; dependability attribute **Maintainability**. As with [D1.M](#43-the-rmav-model), not externally observable — the cost of changing a contract is invisible to anything watching the system run |
| Quality-in-Use effect | **Efficiency** for the engineering stakeholder — coordinated releases, higher change cost |
| Acted on by | Software Architect — interface versioning, contract decoupling |

**D2.A — Availability criticality (relationship)**
> D2 restricted to **structural partition**: the degree to which a relationship is the only route between what it connects, so that severing it makes stakeholder goals unachievable *even though both endpoints remain operational*.

| | |
|:---|:---|
| Stakeholder question | "If just this connection drops, is anything cut off?" |
| High score means | The edge is a structural bridge — this is the defining case of relationship criticality ([§5.2](#52-why-a-link-needs-its-own-score)), and the dimension where D2's redundancy clause bites hardest |
| Metric inputs | `is_bridge`, `min(source.A, target.A)` |
| External quality attribute | Reliability → **Availability**; dependability attribute **Availability**. The one edge dimension with a clean external measurement: severing the link and observing the delivery loss on the far side is exactly what `simulate_edge_removal` does ([§5.6](#56-learned-edge-scoring-gnn)) |
| Quality-in-Use effect | **Effectiveness** — total task loss for everything behind the bridge, while the operator's dashboards stay green — plus **Freedom from risk** |
| Acted on by | DevOps / SRE — redundant routing, multi-broker paths, alternate channels |

**D2.V — Vulnerability / Security criticality (relationship)**
> D2 restricted to **compromise rather than disruption**: the degree to which a relationship is a usable path into or across the system — the lateral-movement value of the channel itself, distinct from the value of either endpoint.

| | |
|:---|:---|
| Stakeholder question | "If an attacker gets onto this channel, where does it take them?" |
| High score means | A high-QoS flow touching an already-exposed endpoint |
| Metric inputs | $w(e)$, `max(source.V, target.V)` |
| External quality attribute | Security → **Confidentiality**, **Integrity**; dependability attributes **Confidentiality + Integrity**. Presumes an adversary rather than a fault, so — as for [D1.V](#43-the-rmav-model) — no fault-removal oracle measures it |
| Quality-in-Use effect | **Freedom from risk** (interception, injection, lateral movement) and **Satisfaction** (confidence loss even without an incident) |
| Acted on by | Security Engineer — channel encryption, mutual authentication, segmentation |

#### The mapping in one view

The correspondence is two-dimensional. **The dimension fixes the harm** — which Quality-in-Use characteristic degrades, and therefore who acts, is a property of the mechanism and does not change with scope. **The scope fixes the mechanism** — whether the element *originates* the fault or *conducts* it:

| Dimension | Component mechanism (D1.x) | Relationship mechanism (D2.x) | Quality-in-Use harm | Acted on by |
|:---|:---|:---|:---|:---|
| **R** | Its failure spreads to transitive dependents | It carries the spread between endpoints | Efficiency, then Satisfaction | Reliability Engineer |
| **M** | It resists safe change | It forces the two sides to change together | Efficiency (engineering stakeholder) | Software Architect |
| **A** | Its loss partitions the graph | It is the only route, both endpoints healthy | Effectiveness, Freedom from risk | DevOps / SRE |
| **V** | It is a valuable target | It is a usable route to a target | Freedom from risk, Satisfaction | Security Engineer |

Reading a row left to right gives the same failure mechanism at two scopes; reading a column down gives the four mechanisms that threaten stakeholders at one scope. The harm and owner columns are shared by construction — a dimension that meant different things for a node and an edge would not be one dimension.

Three design choices in the formulas carry meaning:

- **`max()` for R and V, `min()` for A** — a link is only as *reliable/secure* as its riskiest endpoint (failure or compromise on either side propagates through the edge), but it is only as *available* as its weakest endpoint (the edge cannot be more resilient than the more fragile side it connects).
- **Edge-to-Node Aggregation Duality** — while a link's availability is bounded by its weakest endpoint (`min`), a component's availability exposure is bounded by its weakest incoming/outgoing bridge edge. An edge cannot be more resilient than its endpoints, but a node is no more available than its single non-redundant connection.
- **`is_bridge` appears in both M and A** — a non-redundant edge is expensive to reroute around (raises M, an Efficiency cost to the engineering stakeholder) *and* is a structural cut-point if removed (raises A, an Effectiveness loss to the end user) — the same structural fact, two different stakeholder consequences.
- **`w(e)` appears in R, M and V but not A** — the guarantee crossing a link scales how much its loss costs, but not whether it can be lost at all ([§5.4](#54-what-the-relationship-carries-the-weight-channel)). Replaceability is topological; consequence is QoS-weighted.

The four dimension scores are combined with the same composite coefficients used for nodes ([§4.3](#43-the-rmav-model)), including the QoS-profile adaptation described in [§4.4](#44-what-the-component-carries-the-weight-channel), giving each edge a `QualityScores` record (`reliability`, `maintainability`, `availability`, `security`, `overall`) identical in shape to a component's — see [`EdgeQuality`](../saag/core/metrics.py#L345-L378).

### 5.6 Learned Edge Scoring (GNN)

The Predict stage's GNN produces a direct, per-edge criticality prediction rather than relying on endpoint-node proxies — see [prediction.md §2.6](prediction.md#5-edge-criticality) and [design/SDD.md §6.26](design/SDD.md) for the full architecture:

```
score(u, v) = TypedEdgeEncoder_r( h_u, h_v, e_uv )

e_uv ∈ ℝ^16: QoS weight + path_count_norm + 7-bit edge-type one-hot + 7-bit QoS features
```

Training labels are **measured by removing the edge**, not inferred from its endpoints:

```
I_edge(u, v) = composite_impact(G \ {(u,v)})  −  composite_impact(G)
```

`FailureSimulator.simulate_edge_removal` severs the one relationship, leaves both endpoints active — the partial-outage case of [§5.2](#52-why-a-link-needs-its-own-score) — and recomputes reachability, fragmentation, throughput and flow disruption. Subtracting the no-op control matters: the impact function is non-zero on a pristine graph (topics that already lack a publisher or subscriber count as lost throughput), so a level rather than a delta would give every edge that floor as if it were signal.

This replaces the earlier heuristic `I_edge(u,v) = I*(u) × {1.0 if bridge else 0.1}`, which encoded the §5.2 Effectiveness/Efficiency distinction as a hand-chosen 10× gap rather than observing it. That heuristic remains available for ablation.

> **What the measurement shows.** Most individual edges cost almost nothing: on `av_system`, a clean run of `FailureSimulator.simulate_edge_removal_sweep(layer="system")` at its `top_q=50` default returns **50 candidates** (35 `RUNS_ON`, 11 `CONNECTS_TO`, 3 `SUBSCRIBES_TO`, 1 `PUBLISHES_TO`), of which exactly **4 carry non-zero impact** (max `combined_impact` = 0.00504) — the one `PUBLISHES_TO` edge and all three `SUBSCRIBES_TO` edges, each a shared library's link to a topic it produces or consumes. That is the §5.2 replaceability question answered empirically — most links *are* replaceable, both in count and in magnitude. It also confirms the modelling gap the heuristic hid: all 46 `RUNS_ON`/`CONNECTS_TO` candidates — the majority of the set, and structurally non-redundant bridges by construction — measure exactly 0.0, because the cascade routes no traffic over infrastructure-layer relations at all ([failure-simulation.md §12 L5](failure-simulation.md#12-known-limitations)). A zero there means "this model cannot express that link's failure", not "that link does not matter" — the same caveat that applies to Topic and Node labels (L6). This run is regenerated at `results/av_system_edge_removal_sweep.json` (gitignored, like the rest of `results/`, but deterministic and reproducible by rerunning `simulate_edge_removal_sweep` at defaults on a freshly loaded repository — see the ordering caveat below).
>
> ⚠️ **This figure is sensitive to `MemoryRepository` call order, and the sensitivity is itself worth knowing.** `repo.get_graph_data(include_raw=True)` returns a different, larger edge set (4,345 edges on `av_system` rather than 748, including `ROUTES` and even `DEPENDS_ON` relations) if `AnalysisService.analyze_layers` / `PredictionService.predict_quality` have already run against the same `MemoryRepository` instance — derived-projection state appears to leak into what `include_raw=True` is meant to return as pure $G_{\text{structural}}$. Constructing `SimulationGraph` from a repository that has *not* yet been analysed — the order the pipeline-level tests already enforce (`tests/test_pipeline_dag.py`'s "Simulate before Predict") — gives the clean figures above; running analysis first silently produces a different, contaminated candidate set (previously and incorrectly reported as 50 candidates including `DEPENDS_ON`, 48 non-zero, max 0.0087). No test currently catches this ordering dependency — `tests/test_independence_guarantee.py` only checks that `saag/simulation/` does not *import* prediction code, not that a shared repository instance stays uncontaminated at runtime. This is a real gap worth a regression test, independent of anything reported in this document.

### 5.7 Ranking Critical Edges

Edges are ranked for reporting/UI consumption by the `/edges/critical` endpoint in [api/routers/components.py](../api/routers/components.py), sorting by `EdgeQuality.scores.overall` (the same RMAV-style composite machinery used for nodes, applied edge-wise).

Simulated edge criticality is available separately via `SimulationService.classify_edges()`, which returns `EdgeCriticality` records from the removal sweep ([§5.6](#56-learned-edge-scoring-gnn)). Two fields must be read together:

- `combined_impact` — the measured delta. Comparable to node `composite_impact` because it uses the same weighting.
- `evaluated` — whether the edge was in the candidate set at all. The sweep measures `bridges(G) ∪ top-q edge-betweenness`; everything else returns `evaluated: false` with `combined_impact = 0.0`. **An unevaluated edge is not a harmless edge** — sorting the two together would rank never-measured links alongside measured-as-zero ones.

---

## 6. From Score to Stakeholder Narrative

### 6.1 Worked Example

The formulas stay abstract until tied to an instance. [structural-analysis.md §13](structural-analysis.md#13-worked-example) computes `A(MainBroker) = 0.679` → **HIGH**, driven by `AP_c_directed = 0.65` (a directed structural SPOF) and `BR = 1.0` (every one of MainBroker's edges is a bridge — there is no redundant path around it). Of that score, `0.25 × (0.65 × 0.71) + 0.05 × 0.71 = 0.151` — roughly a fifth — comes from the weight channel: MainBroker routes `/temperature`, a `RELIABLE`/`TRANSIENT_LOCAL`/`HIGH` topic, and that declared guarantee is what makes its normalised weight 0.71. Route only `BEST_EFFORT`/`VOLATILE` traffic through the same topological position and the same SPOF scores materially lower. Read as a Quality-in-Use narrative for the end users of that system:

- **Effectiveness** — if MainBroker fails, both SensorApp and MonitorApp lose their only path to publish/subscribe on `/temperature`. The monitoring task does not degrade, it stops: `BR = 1.0` means there is no alternate route to fall back on.
- **Efficiency** — there is no cheaper-but-slower path to fall back to either; the Efficiency characteristic is not the one at stake here, which is exactly what distinguishes a SPOF from a bottleneck.
- **Freedom from risk** — an undetected temperature excursion during that outage is an economic or safety risk to whoever depends on the reading, which is why $A$ carries the highest weight (0.43) in $Q(v)$ ([§4.3](#43-the-rmav-model)).
- **Context coverage** — because every one of MainBroker's edges is a bridge, this holds in *every* context this topology is used in; there is no scenario where a redundant path happens to save the day.

A component scoring MINIMAL on $A$ but HIGH on $V$ (e.g. a rarely-invoked but highly-reachable component) reads differently under the same lens: normal operation is unaffected (Effectiveness intact), but a compromise there has outsized reach (Freedom from risk driven by breach exposure, not outage).

### 6.2 Reading a Score as a Quality-in-Use Statement

A repeatable template for turning any RMAV profile into a stakeholder-facing statement — the intended way to consume the pipeline's output:

1. **Identify the stakeholder and context** from [§3.3](#33-context-of-use-and-domain-context-vector). *"This is a clinical HIS; the harmed party is a clinician making a care decision."*
2. **Take the dominant per-dimension tier**, not the composite. The composite ranks; the dimension explains.
3. **Translate the dimension into its characteristic** via [§4.5](#45-mapping-rmav-to-external-quality-and-quality-in-use). *High A → Effectiveness and Freedom from risk.*
4. **State the consequence in the stakeholder's terms** using the failure signature in [§4.2](#42-user-side-failure-signature). *"If this fails, the vitals stream stops entirely; there is no alternate route, so the clinician sees stale data with no indication it is stale."*
5. **Qualify with the proxy's limits** from [§7](#7-validity-of-the-construct). *"Structurally exposed — this says nothing about how often this component actually fails or how quickly it would be restored."*

The last step is not optional. A CRITICAL tier is a statement about **structural exposure to Quality-in-Use loss**, not a measurement of Quality-in-Use loss itself.

### 6.3 Academic Paper Template and LaTeX Snippets

When writing paper submissions (e.g., IEEE TSE, TOSEM, JSS, AUSE), researchers can copy and cite the following formal LaTeX definitions and mathematical formulations directly into Overleaf or document manuscripts:

```latex
% Definition D1: Component Criticality (ISO/IEC 25019:2023 Grounded)
% Computed from internal quality evidence; estimates loss of external quality
% attributes (ISO/IEC 25010:2023); defined on Quality-in-Use (ISO/IEC 25019:2023).
\begin{definition}[Component Criticality ($D1$)]
Let $G_l = (V_l, E_l, w)$ be a layer-projected dependency graph at projection $l$. The component criticality measure $\mathrm{crit}_l : V_l \to [0,1]^4 \times [0,1]$ maps each component $v \in V_l$ to a metric-orthogonal external quality attribute vector $\mathbf{s}(v) = [R(v), M(v), A(v), V(v)]^T$ --- respectively fault tolerance, modifiability, availability and confidentiality/integrity, computed from internal quality evidence alone --- and composite score $Q(v)$, estimating counterfactual Quality-in-Use loss across Beneficialness, Freedom from Risk, and Acceptability. $R$ and $A$ are constituents of one ISO/IEC 25010:2023 characteristic (Reliability), not independent attributes; "orthogonal" refers to disjoint metric inputs, not attribute independence (see docs/criticality.md \S3.5):
\begin{equation}
Q(v) = q_A A(v) + q_R R(v) + q_M M(v) + q_V V(v)
\end{equation}
where composite weights satisfy $q_A + q_R + q_M + q_V = 1.0$ (stated defaults $q_A = 0.43, q_R = 0.24, q_M = 0.17, q_V = 0.16$, adapted per system from its aggregate QoS profile).
\end{definition}
% NOTE: these are the stated static defaults, not AHP output, and not the shrunk vector.
% See §4.3 before describing them as "AHP-derived" in a manuscript.

% Definition D2: Relationship Criticality
% A severed link is a fault whose error is confined to one channel; the resulting
% service failure is therefore partial rather than total (Avizienis et al., 2004).
\begin{definition}[Relationship Criticality ($D2$)]
Let $e = (u,v) \in E_l$ be an inter-component dependency edge. Relationship criticality measure $\mathrm{crit}_l : E_l \to [0,1]^4 \times [0,1]$ estimates Quality-in-Use loss, through the loss of the same four external quality attributes scoped to a single channel, resulting from link disruption under operational endpoints ($u, v \in V_l$ active):
\begin{equation}
A(u,v) = 0.30 \cdot \mathbf{1}_{\mathrm{bridge}}(e) + 0.20 \cdot \min\left(A(u), A(v)\right)
\end{equation}
\end{definition}

% Quality-in-Use Transformation Matrix
\begin{equation}
\mathbf{h}_{\mathrm{QiU}}(v) = \mathbf{M}_{\mathrm{RMAV} \to \mathrm{QiU}} \cdot \mathbf{s}_{\mathrm{RMAV}}(v) = 
\begin{bmatrix}
0.35 & 0.25 & 0.40 & 0.00 \\
0.10 & 0.00 & 0.50 & 0.40 \\
0.30 & 0.00 & 0.20 & 0.50
\end{bmatrix}
\begin{bmatrix} R(v) \\ M(v) \\ A(v) \\ V(v) \end{bmatrix}
\end{equation}
```

---

## 7. Validity of the Construct

D1 and D2 define criticality as Quality-in-Use loss. Quality-in-Use is behavioural, and this project never observes it. What follows states exactly how far the evidence reaches, in the three standard senses of validity, so that a claim built on this construct can be scoped honestly.

> **D4 — Criticality is relative, not absolute.** Every score and tier is relative to (i) the score distribution of the system $S$ being analysed, since tiers are box-plot thresholds over that distribution ([§4.6](#46-criticality-classification)), and (ii) the layer $l$, since the vertex set being ranked and the weight normalisation both change with the projection ([§4.4](#44-what-the-component-carries-the-weight-channel)). Criticality values are therefore **not comparable across systems or across layers**. A well-designed redundant system still has a CRITICAL tier; a system full of SPOFs still has a MINIMAL tier.

### 7.1 The Validation Chain Has Three Links

The claim "RMAV is validated" is true of one link in a three-link chain, and it is worth being precise about which. The three links are the three quality views of [§3.0](#30-three-quality-views-internal-external-and-quality-in-use), traversed in order:

```
 internal quality evidence ──①──▶ simulated external quality ──②──▶ real external quality ──③──▶ Quality-in-Use loss
   G_analysis(l), cm_* metrics      I*(v), I_comp(v), I_dyn(v)        (a deployed system)         (what D1 and D2 define)
   RMAV scores, GNN scores          MEASURED and reported             not measured                not measured
```

- **Link ① is measured.** Scores are validated against the simulation oracles ([failure-simulation.md](failure-simulation.md)) by correlation, F1 and SPOF-F1 ([validation.md](validation.md)). This is a real, reported, falsifiable result.
- **Link ② is not measured.** The simulator is a *model* of the executing system, not the executing system. Nothing here has been checked against a deployed pub-sub system's actual delivery behaviour.
- **Link ③ is not measured.** No user study, expert elicitation, or production incident data is used anywhere in this project.

Consequently the defensible claim is: *RMAV tracks simulated external quality loss, and simulated external quality loss is our stated operationalization of Quality-in-Use loss.* The stronger claim — that RMAV tracks Quality-in-Use as stakeholders would report it — is **unsupported by anything in this repository**. Closing ③ requires evidence of a different kind: expert ranking studies against the same topologies, or post-hoc comparison against incident records from a deployed system.

**Naming the middle view correctly changes what can be claimed, in both directions.** The older two-link framing of this section treated everything past the score as "Quality-in-Use", which understated link ① and overstated the construct at the same time. Split properly:

- **What is now claimable and was not.** The simulation oracles measure external product quality — service delivered under fault — and this project measures that. `I_dyn(v)`, the delivery-rate oracle, is built and reported, not prospective: mean $\rho(I_{\text{dyn}}, I^*) = 0.765$ with a minimum of 0.548 across the seven-scenario cohort ([failure-simulation.md §9.3](failure-simulation.md#93-i_dynv--the-effectiveness-term-implemented), [validation.md §3.3](validation.md#33-the-behavioural-oracle-i_textdynv)). Because `I_dyn` observes message delivery under load while `I*` walks reachability over edges, their agreement is convergent evidence of a different kind — which is the strongest thing link ① can offer.
- **What is not claimable and previously read as though it were.** Delivery rate and latency are *not* Quality-in-Use measurements. Two stakeholders can experience the same 40% delivery loss as an inconvenience and as a life-safety event; the external measurement cannot separate them, and neither can any score derived from it.

**Two dimensions are not on the external axis at all, and this bounds link ① rather than link ③.** Per [§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use):

| Dimension | External quality attribute | Oracle that measures it | Kind of evidence |
|:---|:---|:---|:---|
| **R** | Fault tolerance | $I^*(v)$ (subscriber feed-loss under cascade); $I_{\text{dyn}}(v)$ (delivery-rate drop under load) | **External** — service delivery observed under an injected fault |
| **A** | Availability | $I_{\text{comp}}$ `reachability_loss` / `fragmentation` | **External** — readiness for service |
| **M** | Modularity / Modifiability | $I_M(v)$ via [`ChangePropagationSimulator`](../saag/simulation/change_propagation.py) — BFS on $G^\top$ | **Internal** — a structural model of change cost, not a behavioural observation |
| **V** | Confidentiality / Integrity | $I_S(v)$ via `_postpass_security` compromise propagation | **Internal** — a structural model of compromise reach; no adversary is simulated |

> **Where the quality-in-use *weighting* (Layer 3, [§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)) sits relative to this chain: outside it, not as a fourth link.** ③ asks whether external quality predicts stakeholder outcomes — a question about what the *scores mean*. Layer 3 answers a different question — which composite *weighting* to score with — and it does so by collapsing algebraically into a Layer-2 reweighting, never by adding a further measurement. It therefore inherits exactly link ①'s validation status (measured, per `reproduce/domain_weight_comparison.py`) and none of ③'s open question; reporting a domain-derived ranking result is a Layer-2 claim about $\rho$ against $I^*(v)$, not a claim about quality-in-use.

This is why $I^*(v)$ supplies labels for `reliability` and `availability` but declares `maintainability` and `security` absent ([validation.md §3.1](validation.md#31-notation--three-quantities-three-symbols)). It is **not** an implementation gap awaiting a better simulator: maintainability is not an externally observable attribute, and security is not a fault-tolerance property, so for $M$ and $V$ link ① compares one structural model against another. Their correlations should be read as internal consistency checks, not as behavioural validation.

**On the remaining Quality-in-Use characteristics.** *Freedom from risk* would need contract-conformance data: deadline and lifespan counters exist and the harness has an oracle slot for them, but no topic in the scenario corpus declares a deadline — 0 of 710 — so the counters never fire ([failure-simulation.md §9](failure-simulation.md#9-what-the-simulator-measures-in-quality-model-terms)). That is blocked by the corpus, not the method, and it would still be an *external* measurement when unblocked. *Satisfaction* has no correlate in a message-flow simulation at all, which permanently bounds what this construct can claim there. *Context coverage* is a property of the ranking across runs rather than of any single fault.

### 7.2 Construct Validity

**What would falsify the construct.** Criticality claims that structural exposure predicts harm under failure. It is falsified if components ranked CRITICAL are, under controlled failure, no more damaging than components ranked MINIMAL — which is precisely what the link ① correlation and top-$K$ agreement statistics test.

**The definitions name three fault modes; the two halves of the pipeline see them differently.** D1 spans failure, latency and functional degradation, and D2 spans disruption, latency and data loss. The **structural estimator** does not distinguish them: RMAV scores an element's *exposure* — how much depends on it, how irreplaceable it is, how strongly its flows are guaranteed — which is why one score covers all three modes rather than three scores covering one each. The **simulation oracle** does distinguish them: it measures end-to-end latency percentiles before and after fault injection, message drops, and QoS deadline and lifespan violations ([failure-simulation.md](failure-simulation.md)). The breadth of the definitions is therefore carried by link ① rather than by the estimator, and a reader comparing D1's wording against the RMAV formulas should read the difference as division of labour, not as overreach.

**View independence is not source independence.** Predictors read $G_{\text{analysis}}$ while ground truth is produced by simulating $G_{\text{structural}}$, and no simulation output is fed back as a predictor feature. That rules out feature–label feedback. It does *not* make the two independent: both views are deterministic functions of the same input topology, so a modelling assumption shared by both would be invisible to this check. The guarantee should be cited for what it is.

**A third of each system is unlabelled.** The cascade model cannot express the failure of a Topic or a physical Node, leaving a substantial fraction of components per scenario with no ground truth. Criticality is still computed for them — D1 is defined for every vertex — but for those components link ① is untested as well as link ②.

**Two oracles, weak agreement.** Where two ground-truth formulations exist, they correlate only weakly with each other. A result established against one is not evidence about the other, and this document's own cross-references should not be read as transitive.

### 7.3 Characteristic Coverage

Coverage has to be stated twice, because the construct spans two views ([§3.0](#30-three-quality-views-internal-external-and-quality-in-use)) and it is much better on one than the other.

**External quality coverage (ISO/IEC 25010:2023 attributes).** What the dimensions estimate, and whether any instrument here observes it:

| External quality attribute | Estimated by | Coverage | Basis and gap |
|:---|:---|:---|:---|
| Reliability → **Availability** | $A$ | **Strong** | Directly operationalized by `AP_c_directed`, `BR` and `is_bridge`; validated against simulated reachability loss and fragmentation. |
| Reliability → **Fault tolerance** | $R$ | **Strong** | Cascade reach validated against two independent oracles, one topological ($I^*$) and one behavioural ($I_{\text{dyn}}$, mean $\rho = 0.765$). |
| Reliability → **Recoverability** | *— none —* | **Absent** | Requires MTTR, restart semantics, or replication state; no such field exists in the schema ([§7.4](#74-real-world-drivers-vs-structural-proxies)). Every structural SPOF scores alike regardless of how fast it would be restored. |
| Performance efficiency → **Time behaviour**, **Capacity** | *— none —* | **Absent** | The discrete-event engine records latency percentiles, but no RMAV dimension consumes them, and `w(e)` is a delivery-*guarantee* proxy rather than a throughput measurement. |
| **Maintainability** → Modularity, Modifiability | $M$ | **Moderate** | Well-founded on internal evidence (`CQP`, coupling, betweenness), but not externally observable in principle ([§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)), so its oracle $I_M$ is a second structural model rather than a behavioural check. |
| **Security** → Confidentiality, Integrity | $V$ | **Weak** | Topological reachability only. No privilege, encryption, authentication, or data-sensitivity field exists, and no adversary is simulated. Non-repudiation, Accountability, Authenticity and Resistance — the other four Security sub-characteristics — are unaddressed. |
| **Safety** | *— none —* | **Absent** | Safety **is** a first-class ISO/IEC 25010:2023 characteristic (added in the 2023 revision); this framework addresses **none** of its five sub-characteristics — no functional integrity class, hazard catalogue, or safety-criticality field exists in the schema ([§7.4](#74-real-world-drivers-vs-structural-proxies)). |

**Nine characteristics, four addressed at all.** Reliability, Maintainability, Security and (nominally) Performance efficiency are the rows above; **Functional suitability, Compatibility, Interaction capability, and Flexibility are entirely out of scope** and appear nowhere in RMAV — no row exists for them because no metric here bears on any of their sub-characteristics.

**Quality-in-Use coverage (ISO/IEC 25019:2023 characteristics).** What D1 and D2 are *defined* on, none of which is observed here ([§7.1](#71-the-validation-chain-has-three-links), link ③):

| Characteristic | Coverage | Basis and gap |
|:---|:---|:---|
| **Beneficialness (Usability: Effectiveness)** | **Strong** | Directly operationalized by $A$ (structural partition) and `is_bridge`; validated against simulated reachability loss. |
| **Beneficialness (Usability: Efficiency)** | **Moderate** | $R$ and $M$ capture cascade reach and coupling cost, but the *magnitude* of the extra cost (latency, retries, engineer-hours) is not modelled — only that a cost exists. |
| **Freedom from Risk** (economic, operational) | **Moderate** | Well proxied by $A$: a structural partition is a certain outage, and outage duration is where the economic exposure sits. Security risk is topology-only ([§7.4](#74-real-world-drivers-vs-structural-proxies)). |
| **Freedom from Risk** (health, human life) | **Absent** | Requires the **Safety** dependability attribute, which no dimension covers ([§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)). This is the characteristic the ROS 2 / autonomous-vehicle and clinical domains of [§3.3](#33-context-of-use-and-domain-context-vector) rank highest, and the one the construct is least able to speak to. |
| **Acceptability (Trustworthiness)** | **Weak** | Inferred indirectly from $R$ and $V$. Trust erosion and user experience are behavioral responses with no direct structural correlate; nothing in the static pipeline measures user sentiment. |
| *Context coverage* (ISO/IEC 25010:**2011**) | **Indirect** | Assessed empirically as cross-scenario/cross-domain ranking stability ([validation.md](validation.md), `cli/run_scenarios.sh`), not computed per component. |

The first five rows are ISO/IEC 25019:2023 characteristics. **Context coverage is not**: it was a standalone Quality-in-Use characteristic in ISO/IEC 25010:2011 and has no direct counterpart in the 2023 model ([§3.1](#31-what-quality-in-use-is)). It is retained here because ranking stability across topologies is a genuine property of the criticality signal worth reporting, but it should be labelled as a 2011 notion rather than presented as 25019:2023 coverage. The same applies wherever this document uses *Effectiveness*, *Efficiency* and *Satisfaction* as shorthand ([§4.2](#42-user-side-failure-signature), [§4.5](#45-mapping-rmav-to-external-quality-and-quality-in-use)): under the 2023 model these are measurement dimensions *within* Beneficialness → Usability, not top-level characteristics.

The two weakest rows are inherent, not implementation debt: Acceptability and live context coverage are defined over live stakeholder behaviour across real deployments, which a static structural model cannot observe directly.

### 7.4 Real-World Drivers vs. Structural Proxies

[§7.1](#71-the-validation-chain-has-three-links) states the gap qualitatively; this section states it dimension-by-dimension. In a live system, each RMAV dimension is really driven by a mix of runtime, code, and security signals — most of which this project has no field for. The graph model ([graph-model.md](graph-model.md)) carries topology, a DDS-style QoS weight (`reliability`/`durability`/`transport_priority` + message size — a delivery-guarantee proxy, not live traffic or security metadata), and static code metrics (LOC, cyclomatic complexity, instability, LCOM). There is no MTTF/MTTR, privilege, encryption, or telemetry field anywhere in the schema — so every RMAV number below is a structural stand-in, never a direct read of the real-world driver.

**Component criticality:**

| Dimension | Real-world driver | What the structural proxy actually captures | Not captured |
|:---|:---|:---|:---|
| **R** | Intrinsic failure rate (MTTF) and severity of an independent failure | Reverse PageRank + in-degree + CDPot ([§4.3](#43-the-rmav-model)): how far/deep a failure would propagate *given that it happens* | Whether the node fails often at all — $R(v)$ is purely blast-radius, silent on the component's own failure rate |
| **M** | Change-impact risk: regression likelihood from complexity and code churn | `CQP` ([§4.3](#43-the-rmav-model), structural-analysis.md §11.2) blends `complexity_norm`, `instability_code`, `lcom_norm` from static code metrics, plus topological betweenness/coupling | Code churn as a time-series (commit frequency) — `instability_code` is a point-in-time Martin instability ratio, not a churn rate |
| **A** | SPOF status weighted by MTTR (how long the outage lasts once it starts) | `AP_c_directed` + bridge ratio + `QSPOF` ([§4.3](#43-the-rmav-model)): whether removing the node partitions the graph | MTTR — every structural SPOF is scored the same regardless of how fast it would actually be restored |
| **V** | Asset value (PII/secrets handled) and the component's privilege level | Reverse eigenvector/closeness centrality + QoS-weighted in-degree ([§4.3](#43-the-rmav-model)): how reachable/central the node is, weighted by what its dependents were promised | Data sensitivity, privilege level, or what the component does — a low-privilege leaf handling PII scores MINIMAL unless its QoS declaration happens to be high |
| **Safety** *(no dimension)* | Hazard severity and functional integrity class of what the component controls | **Nothing.** The schema has no safety-criticality field, no hazard catalogue, and no functional integrity class ([§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)) | All of it. A component whose failure is a life-safety event is indistinguishable, structurally, from one whose failure loses a debug log — unless the architect happened to encode the difference in QoS policy |

**Relationship criticality:**

| Dimension | Real-world driver | What the structural proxy actually captures | Not captured |
|:---|:---|:---|:---|
| **R** | Cascading-failure probability: synchronous/blocking calls with no circuit breaker | Edge betweenness + bridge factor + `max(source.R, target.R)` ([§5.5](#55-edge-rmav-decomposition)) | Whether the call is actually synchronous/blocking or backed by a circuit breaker — a runtime/code property invisible to a static graph |
| **M** | Interface/contract volatility: how likely a change on one side breaks the other | Edge betweenness + is-bridge flag + edge weight ([§5.5](#55-edge-rmav-decomposition)) | Semantic contract coupling (e.g. a shared database schema) — only topological reachability is measured |
| **A** | Traffic bottlenecks: throughput/bandwidth saturation, lack of redundant routing | is-bridge flag + `min(source.A, target.A)` ([§5.5](#55-edge-rmav-decomposition)): whether this specific link is structurally redundant | Live traffic volume — `weight` ([§5.3](#53-structural-edge-signals)) is the QoS delivery-guarantee proxy, not measured throughput or bandwidth |
| **V** | Trust-boundary crossing: unencrypted channels, missing mutual TLS, lateral-movement potential | Edge weight (QoS) + `max(source.V, target.V)` ([§5.5](#55-edge-rmav-decomposition)) | Encryption status, inter-service authentication, or any security-boundary metadata — none of this exists in the schema |

**What the weight channel does and does not fix.** The QoS weights ([§4.4](#44-what-the-component-carries-the-weight-channel), [§5.4](#54-what-the-relationship-carries-the-weight-channel)) are the one non-topological signal in RMAV, and they close part of every gap above: they distinguish a SPOF on guaranteed traffic from a SPOF on disposable traffic, and an attack surface that breaks strong promises from one that breaks weak ones. They close none of it fully, for three reasons:

- **A declaration, not a measurement.** $w$ encodes what the architecture *promises* about delivery, not what the system *does* — no throughput, latency, failure rate, or data sensitivity is behind it. A `RELIABLE`/`PERSISTENT` topic carrying one message a day outweighs a `BEST_EFFORT` topic carrying the business's entire traffic.
- **Delivery guarantee ≠ business criticality.** Nothing in the schema records safety class, revenue exposure, or regulatory scope, so the weight channel can only approximate those to the extent that architects happened to encode them in QoS policy.
- **Inherited errors are invisible.** A wrong QoS declaration produces a confidently wrong criticality score, with no internal signal that anything is off ([§4.4](#44-what-the-component-carries-the-weight-channel)).

None of this makes the structural proxy wrong — [validation.md](validation.md) is precisely the check that it still tracks simulated failure impact well enough to be useful. It does mean a CRITICAL/HIGH score should be read as *"structurally exposed to Quality-in-Use loss, at the guarantee level this architecture declares,"* not as *"this component definitely has a high MTTF/PII/no-circuit-breaker problem"* — those specific root causes still require the secondary engineering stakeholder from [§3.2](#32-stakeholders-primary-secondary-and-indirect) to inspect the actual component or relationship.

### 7.5 External Validity

The construct has been exercised on generated topologies spanning several deployment domains and scales, never on a harvested one. Three consequences bound how far a result about criticality generalises:

- **Synthetic systems share a generator.** Cross-scenario evaluation tests transfer to held-out *architectures*, but all of them come from one generator with one set of structural priors. Regularities the generator imposes cannot be distinguished from regularities of pub-sub systems.
- **No production or expert baseline.** Nothing has been checked against a deployed system, an operator's judgement, or an incident history. This is the same gap as link ② in [§7.1](#71-the-validation-chain-has-three-links), seen from the generalisation side rather than the measurement side.
- **One middleware family.** The model is pub-sub with DDS-style QoS semantics. Whether the construct transfers to request/response, streaming, or service-mesh architectures is untested — the definitions D1–D2 are agnostic, but every operational term in RMAV assumes pub-sub dependency semantics.

Of the three validity dimensions, this is the weakest, and it is the one where additional evidence would most change what can be claimed.

### 7.6 Empirical Threats to Validity Taxonomy

When reporting evaluation studies in software engineering journals (following Wohlin et al.'s guidelines), threats to validity should be structured and declared across four dimensions:

| Threat Category | Specific Methodological Risk | Mitigation / Scoping Statement in Research Papers |
|:---|:---|:---|
| **Construct Validity** | Proxy discrepancy between graph-derived RMAV scores and real stakeholder Quality-in-Use loss (Link ② of §7.1). | Explicitly scope claims to *structural exposure under simulated fault impact* ($I^*(v)$), acknowledging that live human perception is unmeasured. |
| **Internal Validity** | Potential confounding between QoS policy declarations and structural topology in composite scores ($Q(v)$). | Validate $G_{\text{analysis}}$ predictors independently against pristine simulated ground truth ($G_{\text{structural}}$) to ensure zero feature-label feedback. |
| **External Validity** | Generalizability restricted to pub-sub topologies with DDS-style QoS semantics and synthetic generator priors (§7.5). | State explicitly that findings demonstrate transfer across held-out pub-sub architectures, while cross-paradigm generalizability (e.g., gRPC, Kafka) remains future work. |
| **Conclusion Validity** | Non-parametric score distributions violating normality assumptions in criticality tier classification. | Apply adaptive box-plot thresholding ($Q3 + 1.5 \times IQR$) rather than parametric z-scores or static threshold cutoffs (§4.6). |

---

## 8. Where This Fits in the Pipeline

| Step | Relation to criticality |
|:---|:---|
| [graph-model.md](graph-model.md) | Produces $G_{\text{analysis}}(l)$ and derives `DEPENDS_ON` edges — the substrate both node and edge criticality are computed over — **and** computes the QoS weights $w(v)$, $w(e)$ and `path_count` that every RMAV dimension consumes ([§4.4](#44-what-the-component-carries-the-weight-channel), [§5.4](#54-what-the-relationship-carries-the-weight-channel)). |
| [structural-analysis.md](structural-analysis.md) | Computes the Tier-1 metric vector $M(v)$ and deterministic RMAV scores, weighting the centralities by $w(e)$ — see [§4.3](#43-the-rmav-model) and [§5.3](#53-structural-edge-signals) above. |
| [prediction.md](prediction.md) | Refines RMAV into GNN-blended node scores and direct edge scores $Q_{\text{GNN}}(u,v)$ — see [§5.6](#56-learned-edge-scoring-gnn) above. |
| [failure-simulation.md](failure-simulation.md) | Produces the simulated ground truth ($I^*(v)$, $I_{R/M/A/V}(v)$) that criticality proxies are trained/validated against — a model of **external quality** loss, and the closest observable stand-in this project has for Quality-in-Use loss ([§7.1](#71-the-validation-chain-has-three-links)). |
| [validation.md](validation.md) | Statistically checks whether structural/learned criticality tracks simulated impact — link ① of [§7.1](#71-the-validation-chain-has-three-links). For **nodes** only; see the note below. |
| [research/thesis/material/relationship_criticality.md](research/thesis/material/relationship_criticality.md) | The canonical manuscript statement of D2 and the edge RMAV formulas, with the scoping conditions on what validates them. It lives here rather than in the JSS submission: [research/jss/draft.md §4.1](research/jss/draft.md) deliberately *defers* the relationship construction, because the validation gap in the note below cannot be closed within that paper's independence guarantee. |

> **Edge criticality has no link ① yet, and this is a structural property of the current design rather than a pending fix.** Node scores and node labels are defined on the same vertex set, so they can be correlated. Edges are not: attribution ([§5.5](#55-edge-rmav-decomposition)) is scored over derived `DEPENDS_ON` edges of $G_{\text{analysis}}$, while the removal oracle ([§5.6](#56-learned-edge-scoring-gnn)) severs raw structural edges of $G_{\text{structural}}$. On `av_system` that is 3,753 derived edges against a candidate set drawn from `ROUTES`, `SUBSCRIBES_TO` and `PUBLISHES_TO` — populations that barely intersect. Running the sweep directly on `DEPENDS_ON` is not an available fix: the independence guarantee ([§7.1](#71-the-validation-chain-has-three-links)) requires simulation to operate only on $G_{\text{structural}}$, exactly as [CLAUDE.md's invariant](../CLAUDE.md) states. The one route that respects that guarantee — lifting measured structural-edge impact onto the derived edges it mediates, via an aggregation rule over a many-to-many mapping — is future work, not a mechanical follow-up. Until it is undertaken, **edge criticality rests on construction rather than measurement**, and the §5.6 sweep should not be described as validating the §5.5 scores.

## 9. References

**Quality models (the construct's grounding, [§3](#3-quality-grounding-square)):**

- ISO/IEC 25019:2023, *Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Quality-in-use model*. (primary home of the Quality-in-Use model: Beneficialness, Freedom from Risk, Acceptability)
- ISO/IEC 25010:2023, *Systems and software engineering — SQuaRE — Product quality model*. (source of the external quality attributes the RMAV failure mechanisms are denominated in, [§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use))
- ISO/IEC 25010:2011, *Systems and software engineering — SQuaRE — System and software quality models*. (historical antecedent of the 2023 SQuaRE series)
- ISO/IEC 25023:2016, *Systems and software engineering — SQuaRE — Measurement of system and software product quality*. (the internal-measure / external-measure distinction that [§3.0](#30-three-quality-views-internal-external-and-quality-in-use) rests on)
- ISO/IEC 25022:2016, *Systems and software engineering — SQuaRE — Measurement of quality in use*. (measurement approach that structural proxies stand in for, cf. [§7.1](#71-the-validation-chain-has-three-links))

**Dependability taxonomy ([§3.5](#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)):**

- Avizienis, A., Laprie, J.-C., Randell, B., & Landwehr, C. (2004). *Basic concepts and taxonomy of dependable and secure computing*. IEEE Transactions on Dependable and Secure Computing, 1(1), 11–33. (the attribute set — availability, reliability, safety, integrity, maintainability, confidentiality — and the fault→error→failure chain that distinguishes $R$ from $A$)
- Laprie, J.-C. (ed.) (1992). *Dependability: Basic Concepts and Terminology*. Springer. (the antecedent statement of the same taxonomy)

**Prior meanings of "criticality" ([§2](#2-what-criticality-means-here)):**

- MIL-STD-1629A (1980), *Procedures for performing a failure mode, effects and criticality analysis*. (failure-mode criticality number; likelihood-weighted, cf. [D3](#23-consequence-not-risk))
- IEC 60812:2018, *Failure modes and effects analysis (FMEA and FMECA)*. (current standard form of criticality analysis)
- ISO 31000:2018, *Risk management — Guidelines*. (risk as the combination of likelihood and consequence, the decomposition [D3](#23-consequence-not-risk) selects from)
- IEC 61508 (SIL), ISO 26262 (ASIL), RTCA DO-178C (DAL), MIL-STD-882E. (integrity/criticality levels assigned by hazard analysis rather than computed, [§2.2](#22-what-this-construct-borrows-and-rejects))
- Arulselvan, A., Commander, C. W., Elefteriadou, L., & Pardalos, P. M. (2009). *Detecting critical nodes in sparse graphs*. Computers & Operations Research, 36(7), 2193–2200. (critical node detection problem)
- Lalou, M., Tahraoui, M. A., & Kheddouci, H. (2018). *The critical node detection problem in networks: A survey*. Computer Science Review, 28, 92–117. (survey of the topological tradition [§2.1](#21-three-established-traditions) positions this construct against)

**Structural metrics:**

- Tarjan, R. (1972). *Depth-first search and linear graph algorithms*. SIAM Journal on Computing, 1(2), 146-160. (bridges / cut-edges, also cited in [structural-analysis.md §9.9](structural-analysis.md#99-bridge-ratio-br))
- Henry, S., & Kafura, D. (1981). *Software structure metrics based on information flow*. IEEE Transactions on Software Engineering, (5), 510-518. (structural coupling, also cited in [structural-analysis.md §9.3](structural-analysis.md#93-multi-path-coupling-index-mpci))
- Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill. (pairwise-comparison derivation of the dimension coefficients, [§4.3](#43-the-rmav-model))
