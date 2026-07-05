# Graph-Based Detection of Architectural Anti-Patterns and Prescriptive Refactoring in Distributed Publish–Subscribe Systems

**Target venue:** Automated Software Engineering (AuSE), Springer — Special Issue on *Intelligent
techniques for CI/CD, DevOps, software evolution, technical debt analysis, and refactoring
recommendation*.
**SI topic mapping:** this submission covers three of the bullet's five named areas directly —
*technical debt analysis* (the anti-pattern catalog, §4), *refactoring recommendation* (the
prescriptive operators, §6), and *CI/CD/DevOps* (the delta-aware quality gate, §7). No AutoML/NAS/LLM
contribution is claimed.

> **Provenance note (not part of the manuscript).** This draft supersedes the prior prescription-only
> version of the manuscript, realigning it with the revised outline in this directory
> (`docs/research/ause/outline.md`). It folds in the twenty-one-pattern anti-pattern catalog and its
> empirical validation — previously maintained only as a separate technical reference
> (`docs/antipatterns.md`) and cited by companion papers purely as background — as a first-class
> Section 4 of this manuscript, and re-maps the three prescriptive operators (§6.3) explicitly onto
> the anti-patterns they repair. All numbers reported below (Spearman ρ, F1/precision/recall for
> detection; SRI/ΔSRI, operator counts, and runtimes for prescription) are taken as-is from
> `docs/antipatterns.md` and the previously reconciled `docs/prescription.md`; no figures have been
> invented or adjusted to fit the new framing. The title has also been shortened to drop the
> `SaG-Prescribe:` branding prefix and the `Closed-Loop, DevOps-Integrated` qualifier; the underlying
> framework is still referred to as SaG-Prescribe throughout the abstract and body text below.
>
> **Prior provenance (retained for the record).** This draft previously merged three earlier,
> independent drafts that accumulated in this directory — *"Automated Prescriptive Refactoring of
> Distributed Middleware Architectures"*, *"Automated, Simulation-Verified Refactoring
> Recommendation"*, and *"Closed-Loop Prescriptive Optimization of Publish–Subscribe Architectures
> over Heterogeneous Graph Models"* — all of which described the same SaG-Prescribe system with
> overlapping but non-identical framing. Experimental values for the prescriptive pipeline are taken
> from the most recently re-measured source ("Closed-Loop Prescriptive Optimization...", re-measured
> 2026-07-02 against `reproduce/run_prescribe_all.py` and `saag/prescription/`). The metric name is
> standardized to **System Risk Index (SRI)**, matching `docs/validation.md` and `docs/prescription.md`.
> One related-work subsection (§2.3) retains two `[REF: …]` citation-slot placeholders (learning-based
> and LLM-based refactoring recommenders) — these are explicitly *not* invented citations and must be
> populated from a real bibliography before submission.

---

## Abstract

Distributed publish–subscribe middleware (ROS 2, DDS, MQTT) decouples producers and consumers, but
the resulting indirect dependency structure obscures how component failures cascade, and this
architectural technical debt accumulates invisibly to code-level static analysis (SCA) tools. Unlike
object-oriented design, where mature catalogs of named anti-patterns (God Class, Feature Envy,
Shotgun Surgery) give practitioners a shared vocabulary and testable detection rules, distributed
publish–subscribe architectures have no equivalent catalog: problems are discovered reactively,
through postmortems and cascade incidents, rather than proactively at design time. Even where
structural diagnostic frameworks exist, they typically operate *open-loop*: they rank components by
criticality without naming the specific architectural pathology at fault or producing verified
guidance on how to repair it. We address both gaps with **SaG-Prescribe**, a graph-based framework
that (1) detects twenty-one named, severity-tiered publish–subscribe anti-patterns and bad smells —
including Single Point of Failure, God Component, Broker Saturation (Hub-and-Spoke), Chatty Pair, and
QoS Policy Mismatch — via formally specified topological signatures over thirteen structural metrics
and adaptive box-plot thresholds, each validated against independent failure-simulation ground truth
(Spearman $\rho = 0.876$ overall, rising to $\rho = 0.943$ at large scale; $F_1 = 0.923$, precision
$0.912$, recall $0.857$ for critical-tier classification); and (2) compiles the resulting diagnosis —
components and patterns flagged `CRITICAL` or `HIGH` — into a transformation policy $\Delta(G)$ of
three targeted graph-mutation operators (logical topic splitting, physical anti-affinity
reallocation, and transport QoS contract hardening), each re-verified by the same discrete-event
cascade simulator that produced the diagnosis before it is surfaced as a recommendation. The pipeline
is further operationalized as a delta-aware CI/CD quality gate (`detect_antipatterns.py`) that blocks
only newly introduced structural anti-patterns relative to a Git merge base, within sub-minute
execution bounds.

Across eight benchmark scenarios used for detection validation and a seven-scenario subset used for
prescriptive evaluation — spanning autonomous vehicles, IoT, finance, healthcare, and hyper-scale
enterprise topologies — the generated prescriptions reduce the System Risk Index (SRI, lower is
better) in every scenario, with relative risk reductions between 1.4% and 15.9% (Wilcoxon signed-rank
across the seven baseline/mutated pairs: $W=0.0$, $p=0.0156$, significant at $\alpha=0.05$). We report
this result alongside a less flattering but more honest one: component-level failure-impact
reductions average only +4.61% and regress in two of seven scenarios (−31.67% and −25.36%), because
the current prescriptive policy is applied greedily and unconditionally, occasionally introducing new
cascade hops that offset its own gains — a finding that directly motivates a per-edit acceptance
filter as the highest-priority extension. The full generate–verify loop completes in under 65 seconds
for six of the seven prescription scenarios (4.7 s–64.3 s) and in approximately 10.8 minutes for the
300-process enterprise topology, while the standalone CI/CD detection gate completes in under 2
seconds to ~40 seconds across the same scale range — making both proactive anti-pattern detection and
closed-loop prescriptive refactoring practical as merge-request-gated or nightly pre-deployment CI/CD
stages.

**Keywords:** architectural anti-patterns; bad smells; technical debt analysis; refactoring
recommendation; publish–subscribe middleware; CI/CD quality gates; DevOps; failure cascade
simulation; graph mutation; search-based software engineering

---

## 1. Introduction

### 1.1 Context and Motivation

Distributed publish–subscribe middleware frameworks — such as the Robot Operating System (ROS 2), the
Data Distribution Service (DDS), and MQTT — form the communication backbone of modern microservices,
IoT systems, and safety-critical cyber-physical platforms. These architectures achieve spatial,
temporal, and synchronization decoupling among producers and consumers by routing messages through
shared topics and broker intermediaries. However, this decoupling introduces deep, non-linear
structural dependencies that obscure how component-level faults propagate through the wider system.
Hardening these networks against cascading failures requires proactive, continuous, pre-deployment
optimization before configurations are committed to runtime operational fabrics — not remediation
after an incident.

### 1.2 Two Open Gaps: An Unnamed-Pathology Gap and an Open-Loop Refactoring-Recommendation Gap

Automated quality assurance has historically operated at the source-code level, through static code
analysis (SCA) platforms such as SonarQube. This produces an **Architecture-Code Gap**: a system can
have clean source code within every individual module yet remain highly fragile at the topology level
— single points of failure (SPOFs), co-located deployment bottlenecks, or mismatched communication
attributes are invisible to file-scoped analysis. Shifting structural verification "left" into the
CI/CD pipeline requires a paradigm shift from Static Code Analysis to *Static System Analysis*.

This gap is compounded by a second, narrower one that this paper addresses directly. Object-oriented
design has a mature vocabulary for structural pathology — God Class, Feature Envy, Shotgun Surgery —
each with a name, a formal detection rule, and an established refactoring strategy. Microservices
research has begun to build an analogous vocabulary for REST-based architectures (excessive
chattiness, shared databases, distributed monoliths). **No equivalent catalog exists for
publish–subscribe topologies.** Practitioners identify pub-sub-specific problems — broker saturation,
topic fan-out explosion, QoS contract mismatches — reactively, through postmortem reports, performance
regressions, or cascade failures, rather than proactively at design time, because there is no shared
name or testable rule for these conditions to check against.

Even where topology-aware diagnostic frameworks do exist, a further limitation remains. In our
companion paper [1], we introduced **Software-as-a-Graph (SaG)**, a static system analysis framework
that models pub-sub topologies as native heterogeneous graphs and produces diagnostic criticality
rankings $Q(v)$ and failure-impact predictions $I(v)$. SaG closes the Architecture-Code Gap for
diagnosis, but — like refactoring-recommendation research more broadly, from code-smell detectors to
Search-Based Software Engineering (SBSE) — it behaves as an *open-loop diagnostic engine*: it flags
architectural vulnerabilities without synthesizing concrete, verified guidance on how to resolve them,
and it reports a numeric criticality score rather than naming the specific pathology at fault. For
architectural refactorings of distributed topologies, where the quality attribute is resistance to
cascading failure, this dual limitation is particularly hazardous: an architect cannot act on an
unnamed number, and an edit that looks beneficial locally can degrade global resilience if its effect
is never verified. We refer to the combination of these two limitations — no named, testable
publish–subscribe anti-pattern vocabulary, and no verified guidance on how to repair what is found —
as the **detection-and-remediation gap** this paper closes.

### 1.3 Proposed Solution: SaG-Prescribe

To close this gap, we present **SaG-Prescribe**, a graph-based framework unifying three stages: detect,
prescribe, and gate. First, SaG-Prescribe **detects** twenty-one named, severity-tiered
publish–subscribe anti-patterns and bad smells, each given a formal topological detection rule over
thirteen structural metrics and adaptive box-plot thresholds, and each validated against independent
failure-simulation ground truth. Second, it **prescribes**: components and patterns flagged
`CRITICAL` or `HIGH` — itself informed by a **Code Quality Penalty (CQP)** computed from
static-analysis metrics — feed a rule-based prescriptive engine that generates targeted architectural
mutations along three vectors: (1) logical topic splitting, isolating high-fan-out publish channels;
(2) physical anti-affinity reallocation, isolating co-located SPOFs; and (3) transport QoS contract
hardening, upgrading volatile/best-effort channels to reliable, transient-local settings. Crucially,
SaG-Prescribe implements **closed-loop simulation verification**: each mutated candidate topology $G'$
is programmatically re-simulated with the same discrete-event cascade simulator used for detection,
under identical fault conditions, and a policy is reported together with its verified change in the
System Risk Index (SRI) rather than as an unverified suggestion. Third, because the recommend-verify
loop runs against an in-memory repository with no database dependency, the same underlying engine is
operationalized as a **delta-aware CI/CD quality gate** that blocks only structural anti-patterns
newly introduced relative to a Git merge base, leaving pre-existing, risk-accepted debt untouched
unless its severity worsens.

### 1.4 Contributions

1. **A catalog of twenty-one publish–subscribe anti-patterns and bad smells**, each with a formal,
   topology-based detection rule expressed over thirteen structural metrics and adaptive box-plot
   thresholds, organized into three severity tiers and four RMAV quality dimensions, and empirically
   validated against independent failure-simulation ground truth (§4).
2. **A code-quality-augmented, closed-loop prescriptive refactoring pipeline** that translates the
   catalog's findings into counterfactual graph mutations via three named operators, each explicitly
   mapped to the anti-patterns it targets, and verifies each candidate against the same discrete-event
   cascade simulator that produced the diagnosis before it is surfaced as a recommendation (§5–§6).
3. **A DevOps-integrated, delta-aware CI/CD quality gate** (`detect_antipatterns.py`) that surfaces
   the catalog's twenty-one detectors as a blocking check, comparing candidate and merge-base
   topologies to flag only newly introduced anti-patterns, with a three-tier exit-code protocol and an
   auditable waiver register for accepted legacy risk (§7).
4. **A multi-scenario empirical evaluation** across eight detection-validation scenarios and a
   seven-scenario prescriptive-evaluation subset, showing consistent, statistically significant SRI
   reductions of 1.4%–15.9% (Wilcoxon $W=0.0$, $p=0.0156$), an operator-level contribution analysis, a
   **component-level audit that reports mixed results honestly** rather than only the flattering
   system-level aggregate, and a runtime audit confirming compatibility with CI/CD budgets at every
   evaluated scale (§8–§9).

The remainder of the paper is organized as follows. Section 2 surveys related work. Section 3
formalizes the graph model and the code-quality bridge. Section 4 presents the anti-pattern catalog
and its empirical validation. Section 5 defines the closed-loop optimization objective. Section 6
presents the prescriptive pipeline and its operators. Section 7 describes the DevOps/CI-CD
integration. Sections 8 and 9 present the experimental design and results. Section 10 discusses
implications and threats to validity, and Section 11 concludes.

---

## 2. Background and Related Work

### 2.1 Publish–Subscribe Middleware Dependability

Dependability research for message-oriented middleware historically centers on protocol verification,
fault-tolerant replication patterns, network traffic load balancing, and runtime contract validation.
Classical broker-fault-tolerance work replicates or partitions the broker itself to survive crashes
[4, 5], and replicated-log designs such as Apache Kafka generalize this to a durable, partitioned
commit log underlying the pub-sub abstraction [6]. Closer to our DDS/ROS 2 setting, recent work
analyzes the latency and reliability behavior of DDS's QoS-driven retransmission protocol [7] and the
static verifiability of interdependent DDS QoS policies [8]. While Chaos Engineering practices inject
faults into live clusters to evaluate empirical resilience, this occurs late in the lifecycle and
introduces operational risk. SaG-Prescribe instead operates earlier, executing static system analysis
on "Architecture-as-Code" descriptors to proactively detect and remediate structural pathology before
deployment, treating the topology as an open parameter rather than a fixed input.

### 2.2 Anti-Pattern and Code-Smell Catalogs

The most mature body of anti-pattern work addresses object-oriented design. Fowler's refactoring
catalog names recurring code-level pathologies (Long Method, Feature Envy, Shotgun Surgery) alongside
concrete refactoring transformations [13]. Brown et al.'s *AntiPatterns* catalog extends this template
to the architectural and project-management level, formalizing a named pattern, a recognizable
symptom, and a remediation strategy as the standard specification unit [14]. Suryanarayana et al.
systematize design smells with explicit, checkable structural rules rather than purely qualitative
description [15]. Microservices research builds an analogous vocabulary for REST-based
architectures: Richardson catalogs recurring microservices design and deployment patterns [16], and
Taibi et al. propose a taxonomy of microservices-specific anti-patterns (excessive chattiness, shared
databases, distributed monoliths) grounded in practitioner surveys [17].

The catalog presented in this paper (§4) follows the same specification template — a named pattern, a
formal detection rule, a remediation strategy — but targets a domain none of the above cover: the
*publish–subscribe communication topology* rather than object-oriented code structure or
request-response service boundaries. A pub-sub system can be architecturally pathological (a single
broker routing all traffic, a topic with hundreds of unmanaged subscribers) while every individual
component is internally well-structured by OO standards and every service boundary is REST-idiomatic.
The anomalies our catalog targets — SPOF hosts, congested topic hubs, fragile transport contracts —
have no file-level or service-boundary analog and are invisible to both OO and microservices
catalogs. Where those catalogs are grounded primarily in expert judgment and practitioner survey, §4.5
further validates each of our detection rules against independent failure-simulation ground truth.

### 2.3 Refactoring Recommendation and Architectural Technical Debt

Automated refactoring recommendation has been studied extensively at code level: smell-driven
recommenders detect structural anomalies (god classes, feature envy) and propose remediation
transformations, following the catalogs of §2.2 [13, 15]; learning-based recommenders mine
refactoring histories to predict refactoring opportunities [REF: ML-based refactoring prediction]; and
recent approaches employ large language models for refactoring suggestion and explanation [REF:
LLM-based refactoring]. At the architectural level, technical-debt research quantifies the cost of
structural decay and proposes prioritized remediation plans, in the spirit of Brown et al.'s
architectural AntiPatterns [14].

SaG-Prescribe differs from this body of work along two axes. First, its **scope** is the deployed
system topology — applications, brokers, topics, hosts, and their QoS contracts — rather than source
code within a module boundary; the anomalies it detects and remediates have no file-level analog and
are invisible to code-scope recommenders. Second, its **verification model** is closed-loop: whereas
code-level recommenders typically validate suggestions against static quality metrics or historical
acceptance data, SaG-Prescribe re-simulates every candidate topology against a cascade failure model
and surfaces only recommendations with verified risk improvements. In technical-debt terms, the
framework names architectural debt items (anti-patterns with quantified risk via the Code Quality
Penalty and RMAV attribution, §3, and formal specification, §4), proposes repayments (mutation
operators, §6.3), and verifies the repayment's effect before recommending it — a verify-before-recommend
discipline that, to our knowledge, has not been applied to pub-sub topology refactoring.

> *[Citation-slot note, not part of the manuscript: the two remaining `[REF: …]` markers above must be
> populated from a real bibliography — learning-based refactoring-prediction studies and LLM-assisted
> refactoring work — before submission. No references have been invented to fill these slots.]*

### 2.4 Search-Based Software Engineering and Architecture Optimization

Search-Based Software Engineering (SBSE) applies heuristic search to discover architectural refactoring
blueprints [2], and the architecture-optimization sub-field surveyed by Aleti et al. [3] specifically
targets quality-attribute-driven structural redesign. However, classical search-based methods often
operate open-loop, reporting recommendations without verifying their operational efficacy against a
cascade model. SaG-Prescribe combines the multi-dimensional diagnostics of SaG [1] with closed-loop
simulation verification, ensuring that every recommended edit is evaluated for its effect on the System
Risk Index before acceptance.

### 2.5 Diagnostic Foundation (SaG)

We rely on the heterogeneous graph representation, multi-dimensional quality attribution (RMAV), and
discrete-event failure simulator of Software-as-a-Graph [1] as our diagnostic baseline. SaG-Prescribe
builds directly on SaG's hexagonal ports, extending the domain service to name specific structural
pathologies (§4) and to close the loop between diagnostic ranking and prescriptive mutation (§6). The
full mathematical treatment of the diagnostic stages — graph schema, projection rules, RMAV
attribution, and the learned failure-impact predictor — is given in [1] and is not repeated here; §3
summarizes only what the detection and prescriptive engines consume.

### 2.6 Structural Criticality Analysis

Graph-theoretic approaches offer constructs such as betweenness centrality [9], PageRank, closeness,
and articulation-point tests to pinpoint critical bridges; recent work applies centrality measures
directly to microservice dependency graphs to detect architectural anti-patterns [10], and complex-network
analyses of software call graphs report the same small-world, hub-dominated topologies that motivate
criticality analysis in the first place [11]. Design-structure-matrix research on coupling and
modularity [18] and combinatorial network-reliability theory [19] provide the graph-theoretic
foundations that the anti-pattern catalog's structural signatures (§4) build on; Lehman's laws of
software evolution [20] further motivate treating architectural criticality as a property that must be
re-checked continuously as a system evolves, rather than assessed once at initial design. Because
classical centrality metrics assume uniform edge semantics, they degrade on pub-sub layers, where
decoupled endpoints are separated by high-fan-out topics, brokers, and distinct QoS policies. [1]
quantifies this gap and motivates the typed multigraph model that both the anti-pattern catalog and
SaG-Prescribe's mutation operators act on. These approaches identify fragility but, like the
diagnostic layer of SaG, stop short of naming the specific pathology and prescribing and verifying
remediation — the step this paper automates.

---

## 3. System Model and Code-Quality-Augmented Technical Debt Analysis

### 3.1 Heterogeneous Graph Formulation

A distributed publish-subscribe deployment is modeled as a typed, weighted, directed multigraph

$$G = (V, E, \tau_V, \tau_E, w_E, w_V)$$

where $\tau_V : V \to T_V$ partitions vertices into five semantic types,

$$T_V = \{\text{Application}, \text{Library}, \text{Topic}, \text{Broker}, \text{Node}\}$$

* **Application ($V_{\text{app}}$):** active execution processes that produce or consume data.
* **Library ($V_{\text{lib}}$):** shared code modules utilized across applications.
* **Topic ($V_{\text{topic}}$):** named communication channels mediating message exchanges.
* **Broker ($V_{\text{broker}}$):** middleware intermediaries routing message paths.
* **Node ($V_{\text{node}}$):** physical or virtual hosting environments.

and $\tau_E : E \to T_E$ assigns each edge to a structural relation imported from the architecture
description,

$$T_E = \{\text{PUBLISHES\_TO}, \text{SUBSCRIBES\_TO}, \text{ROUTES}, \text{RUNS\_ON}, \text{CONNECTS\_TO}, \text{USES}\}$$

### 3.2 Derived Dependencies: The `DEPENDS_ON` Projection

To uncover logical dependency paths hidden behind decoupled pub-sub structures, the framework derives
explicit `DEPENDS_ON` relations (directed **dependent → dependency**) via typed projection rules:

* **Application-to-Application:** formed when a subscriber depends on a publisher via a shared topic.
* **Application-to-Broker:** maps reliance on the specific broker instance routing an application's topics.
* **Application-to-Library:** models the simultaneous blast radius where a shared library failure
  instantly impacts all consuming applications.
* **Broker-to-Broker:** captures colocation vulnerabilities where multiple brokers share a physical host.

This projection produces $G_{\text{analysis}}$, organized across four architectural layers — **app**
(applications only), **infra** (nodes only), **mw** (applications and brokers), and **system** (all
types) — each providing a different lens for both the anti-pattern detectors of §4 and the RMAV
attribution below.

### 3.3 The Code Quality Penalty (CQP)

To bridge local code quality with system architecture — and to give this paper a direct, explicit tie
to the SI's "software quality evaluation" framing — the framework ingests modular metrics from static
code analysis (SCA) APIs during model import. These features encompass total lines of code
(`cm_total_loc`), Weighted Methods per Class (`cm_avg_wmc`), Lack of Cohesion of Methods
(`cm_avg_lcom`), and the technical debt ratio (`sqale_debt_ratio`). Rank-normalized properties map
directly into a per-component **Code Quality Penalty**, defined for Application and Library vertices:

$$\mathrm{CQP}(v) = 0.10\,\text{loc\_norm} + 0.35\,\text{complexity\_norm} + 0.30\,\text{instability\_code} + 0.25\,\text{lcom\_norm}$$

CQP is the paper's single explicit channel from code-level quality signals into the architecture-level
risk model: it feeds directly into the Maintainability dimension of the RMAV attribution below, so a
module's static-analysis debt is not siloed from its topological criticality.

### 3.4 Multi-Dimensional Quality Attribution (RMAV)

Component criticality is decomposed into four orthogonal dimensions, ensuring that each structural and
code metric feeds exactly one perspective to preserve explanation legibility:

* **Reliability ($R$):** fault-propagation risk via Reverse PageRank (RPR) and fan-out concentration.
* **Maintainability ($M$):** coupling complexity driven by betweenness centrality ($BT$), efferent QoS
  out-degree ($w\_out$), and the CQP metric:

$$M(v) = 0.35\,\mathrm{BT}(v) + 0.30\,\mathrm{w\_out}(v) + 0.15\,\mathrm{CQP}(v) + 0.12\,\mathrm{CouplingRisk\_enh}(v) + 0.08\,(1-\mathrm{CC}(v))$$

* **Availability ($A$):** single-point-of-failure risk via directed cut-vertex tests and QoS-amplified
  SPOF scores.
* **Vulnerability ($V$):** exposure to adversarial reach, mapping attack propagation vectors.

These four profiles blend into a composite criticality score $Q(v)$ using Analytic Hierarchy Process
(AHP) weights [12] mixed with a uniform prior ($\lambda=0.70$) to prevent extreme parameter
concentration, yielding final weights of $(0.38, 0.24, 0.19, 0.19)$ for availability, reliability,
maintainability, and vulnerability respectively. Composite scores are mapped to five criticality tiers
(`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `MINIMAL`) using adaptive box-plot thresholding on the system's
own score distribution (`CRITICAL`: $Q > Q_3 + 1.5\,\mathrm{IQR}$; `HIGH`: $Q_3 < Q \le$ upper fence).
This section's typed graph, RMAV dimensions, and adaptive box-plot machinery are the shared foundation
consumed by both the anti-pattern catalog (§4) and the prescriptive engine (§6).

---

## 4. A Catalog of Architectural Anti-Patterns for Publish–Subscribe Systems

### 4.1 Anti-Patterns vs. Bad Smells

Following the taxonomy established in object-oriented design research (§2.2), this catalog
distinguishes between two categories of finding. An **anti-pattern** is a recognizable structural
configuration known to cause problems: it represents a deliberate or accidental architectural
decision that creates systemic risk and typically requires significant restructuring to resolve. A
**bad smell** is a surface symptom that suggests an underlying problem may exist — not definitively
harmful in every context, but a reliable signal worth investigating, and often addressable with only a
localized change. In practice, the distinction is one of confidence: anti-patterns have well-understood
failure modes, whereas bad smells are heuristics that require human judgment to confirm.

The key enabling insight is that architectural decisions in publish–subscribe systems leave
**measurable structural fingerprints** in the dependency graph: a single broker serving all
applications becomes an articulation point; a component that publishes to and subscribes from
everything exhibits extreme betweenness centrality; a topic with hundreds of subscribers shows
anomalous out-degree in the topic projection. Because these fingerprints are computable from the
system's static architecture — the YAML configuration, the launch file, the infrastructure-as-code —
without running the system at all, detection can occur proactively, at design time or during CI/CD
pipeline execution, before any deployment.

### 4.2 Detection Methodology

Detection operates over a thirteen-element metric vector $M(v)$ computed per component: PageRank
$\mathrm{PR}(v)$, Reverse PageRank $\mathrm{RPR}(v)$, betweenness centrality $\mathrm{BT}(v)$,
closeness centrality $\mathrm{CL}(v)$, eigenvector centrality $\mathrm{EV}(v)$, in- and out-degree
($\mathrm{DG}_{\text{in}}(v)$, $\mathrm{DG}_{\text{out}}(v)$), clustering coefficient $\mathrm{CC}(v)$,
articulation-point score $\mathrm{AP}_c(v)$, bridge ratio $\mathrm{BR}(v)$, and QoS-weighted degree
measures ($w(v)$, $w_{\text{in}}(v)$, $w_{\text{out}}(v)$). Topological metrics use rank-based
normalization by default, since they are typically highly skewed (a single hub-broker may have
betweenness 50$\times$ the median, which would compress all other values under min-max scaling); linear
properties (LOC, complexity, LCOM, hardware capacity) use min-max normalization, since their absolute
magnitude differences are meaningful.

A central design choice for robustness is the use of **adaptive box-plot thresholds** rather than
fixed global constants: for a metric vector $X$, the outlier fence is $Q_3 + 1.5 \times \mathrm{IQR}$,
and a component is flagged when its value exceeds this fence. This gives three properties important
for cross-system detection: **scale invariance** (a "high" betweenness score means something different
in a 10- versus a 300-component system, and the threshold adapts automatically); **distribution
awareness** (the threshold derives from the system's own metric distribution, avoiding both
over-flagging dense systems and under-flagging sparse ones); and **theoretical grounding** (the
$1.5 \times \mathrm{IQR}$ rule identifies genuine statistical outliers relative to a component's
peers, matching the definition of an anti-pattern as a structurally anomalous configuration). Several
patterns additionally target coupling imbalance directly, following Martin's Instability metric [21]
enriched with topological path complexity (`CouplingRisk_enh`, §3.4).

### 4.3 Catalog Overview

The twenty-one patterns are organized into three severity tiers — `CRITICAL` (structural risk
requiring immediate architectural intervention; no production deployment should proceed without
addressing these), `HIGH` (significant risk materially degrading reliability, availability, or
maintainability; should be addressed in the current development cycle), and `MEDIUM` (accumulated
technical debt or localized risk; tracked for medium-term remediation) — and mapped onto the four
RMAV dimensions of §3.4. Table 4.1 summarizes the full catalog; formal detection rules and detailed
remediation strategies for every pattern are given in the companion technical reference
(`docs/antipatterns.md`) and are cited rather than reproduced here in full.

**Table 4.1 — Anti-pattern catalog summary.**

| Pattern | Severity | Primary RMAV Dimension | Detection Signal |
| --- | --- | --- | --- |
| SPOF | CRITICAL | Availability | Articulation point, QoS-weighted SPOF score |
| SYSTEMIC_RISK | CRITICAL | Reliability | Share of CRITICAL-tier components $> 20\%$ |
| CYCLE | HIGH | Architecture (cross-cutting) | Strongly connected component / self-loop |
| GOD_COMPONENT | CRITICAL | Maintainability | Extreme betweenness $\wedge$ CRITICAL maintainability |
| BOTTLENECK_EDGE | HIGH | Availability | Edge betweenness outlier |
| BROKER_OVERLOAD | HIGH | Availability | Broker availability $\ge 2\times$ median, or sole broker |
| DEEP_PIPELINE | HIGH | Reliability | Path length $\ge \max(5, P_{75})$ |
| TOPIC_FANOUT | MEDIUM | Reliability | Topic subscriber out-degree outlier |
| CHATTY_PAIR | MEDIUM | Maintainability | Bidirectional edge-weight product $> \tau$ |
| QOS_MISMATCH | MEDIUM | Reliability | Publisher/subscriber QoS-weight gap $> \tau$ |
| ORPHANED_TOPIC | MEDIUM | Maintainability | Zero in- or out-degree on structural graph |
| UNSTABLE_INTERFACE | MEDIUM | Maintainability | High `CouplingRisk_enh` $\wedge$ high $M(v)$ |
| BRIDGE_EDGE | HIGH | Availability | Graph-theoretic bridge |
| FAILURE_HUB | CRITICAL | Reliability | Reliability outlier $\wedge$ above-median out-degree |
| CONCENTRATION_RISK | MEDIUM | Reliability | Top-3 PageRank share $> 0.5$ |
| HUB_AND_SPOKE | MEDIUM | Maintainability | Low clustering coefficient $\wedge$ degree $> 3$ |
| TARGET | CRITICAL | Vulnerability | Security-criticality tier $\ge$ CRITICAL |
| EXPOSURE | HIGH | Vulnerability | HIGH security tier $\wedge$ high closeness |
| CHAIN | MEDIUM | Architecture (cross-cutting) | Degree-bounded linear weakly-connected subgraph |
| ISOLATED | MEDIUM | Architecture (cross-cutting) | Zero total degree |
| COMPOUND_RISK | CRITICAL | Architecture (cross-cutting) | Co-occurring SPOF + God/Hub/Failure-Hub finding |

### 4.4 Representative Pattern Walkthroughs

We highlight five patterns spanning severity tiers, RMAV dimensions, and detection technique
diversity.

**SPOF (Single Point of Failure).** A component $v$ whose removal disconnects the graph — formally an
articulation point. Detection combines a binary structural test with a continuous
**QoS-weighted SPOF severity** $\mathrm{QSPOF}(v) = \mathrm{AP}_c(v) \times w(v)$, so a flagged SPOF is
both structurally load-bearing and operationally significant. Unlike a performance bottleneck, a SPOF
produces a hard availability cliff: the system works completely until the SPOF fails, at which point
dependent functionality becomes entirely unavailable. Remediation centers on introducing redundancy
(replicated brokers, active-passive failover, stateless horizontally-scalable extraction for
application SPOFs) and circuit-breaker patterns to bound failover latency [22].

**GOD_COMPONENT.** A component simultaneously exhibiting extreme betweenness centrality and
CRITICAL-tier maintainability ($\mathrm{BT}(v) > 0.30 \wedge \mathrm{Level}(M(v)) = \mathrm{CRITICAL}$).
It sits at a disproportionate share of shortest paths while also being the hardest component to change
safely, concentrating change-proneness, failure impact, and cognitive complexity simultaneously.
Remediation follows the Strangler Fig pattern: incrementally extracting cohesive publish/subscribe
responsibility subsets into new, purpose-built components while the original remains functional.

**BROKER_OVERLOAD (Hub-and-Spoke).** The pub-sub-specific instantiation of the classical Hub-and-Spoke
topology anti-pattern: a broker handling at least $2\times$ the median broker's routing load (or the
sole broker in the system, flagged unconditionally). One of our eight validation scenarios
deliberately encodes this anti-pattern with only two brokers serving seventy applications across
twelve nodes; both brokers are correctly classified `CRITICAL`, with broker failure-impact scores
exceeding 50% of total system applications, confirming that the availability metric correctly
identifies broker-level overload independent of any single-application-level signal.

**CHATTY_PAIR.** A pair of application components maintaining a bidirectional, high-weight dependency
through separate topics in each direction: $(u \to v) \wedge (v \to u) \in E_{\text{depends}}$ with
$\mathrm{edge\_score}(u{\to}v) \times \mathrm{edge\_score}(v{\to}u) > \tau_{\text{chatty}}$. This
pattern detects **logical coupling masquerading as decoupling**: the pub-sub layer gives the
appearance of independence, but the communication pattern reveals that the pair cannot be
independently deployed, scaled, or reasoned about, and the coupling is distributed across the broker
rather than visible in code. Remediation introduces a mediator component or applies event-carried
state transfer, replacing the bidirectional conversational pattern with a unidirectional, reactive
one.

**QOS_MISMATCH.** A dependency edge $(u, v)$ where the publisher's QoS weight falls substantially below
the subscriber's expected guarantee level ($w_{\text{publisher}}(u) < w_{\text{subscriber}}(v) -
\tau_{\text{qos}}$). This pattern is unique to QoS-bearing middleware: it detects a **silent
connectivity failure** risk — in DDS/ROS 2 systems, incompatible QoS policies can prevent the endpoint
match from being established at all, with no compile-time warning, while both endpoints appear healthy
in isolation. Remediation includes a QoS policy registry enforced in CI, or a dedicated QoS-bridging
relay component when the publisher's constraints (e.g., a hardware driver limited to `BEST_EFFORT`)
cannot be upgraded directly.

### 4.5 Empirical Validation of Detection

Findings are validated empirically through the failure simulation pipeline: for each flagged
component, the simulated impact score $I(v)$ — computed by exhaustive component removal and cascade
propagation — provides independent evidence that a topological signature corresponds to real
structural risk. Table 4.2 summarizes the headline validation metrics.

**Table 4.2 — Detection validation metrics (overall).**

| Metric | Target | Achieved |
| --- | --- | --- |
| Spearman $\rho$ ($Q$ vs. $I$) | $\ge 0.70$ | **0.876** |
| $F_1$-Score (critical classification) | $\ge 0.90$ | **0.923** |
| Precision | $\ge 0.85$ | **0.912** |
| Recall | $\ge 0.80$ | **0.857** |
| Top-5 Overlap | $\ge 0.70$ | **0.80** |

At large scale (150–300+ components), $\rho$ rises to **0.943**, indicating that prediction accuracy
improves as system scale increases — precisely the regime where manual architectural review becomes
least practical. Pattern-specific evidence corroborates the aggregate: SPOF classification achieves
$F_1 > 0.95$ in application-layer analysis, and the deliberately-encoded Hub-and-Spoke scenario (§4.4)
confirms BROKER_OVERLOAD detection independent of the aggregate correlation figures. A baseline
comparison shows the composite $Q(v)$ score consistently outperforming single-metric alternatives
(betweenness centrality alone: $\rho = 0.75$); degree centrality alone reaches $\rho = 0.95$ in
synthetic graphs, a known artifact of topology generators that structurally force high-degree hubs
into SPOF positions, rather than evidence that degree alone suffices on real-world heterogeneous
topologies.

Detection validation uses an eight-scenario suite (01–08) spanning autonomous-vehicle, IoT, financial,
healthcare, deliberately-anti-pattern (Hub-and-Spoke), microservices, enterprise, and a deterministic
"Tiny Regression" smoke-test topology. Two scenarios play a distinguished role: **Scenario 06**
(sparse microservices mesh) is the primary **precision stress test** — a well-designed topology should
produce few or no findings, confirming detectors do not over-flag well-structured systems — and
**Scenario 07** (300+ components) is the primary **scalability benchmark**, confirming detection
algorithms scale gracefully to enterprise-scale deployments. The seven-scenario subset used for
prescriptive evaluation in §8–§9 excludes the smoke-test fixture, which carries no domain-representative
topology and would not contribute meaningful signal to the prescriptive-efficacy questions asked
there.

---

## 5. Closed-Loop Optimization Objective

The prescriptive task is to compute a transformation policy $\Delta$ producing a mutated topology
$G' = \Delta(G)$ that minimizes the aggregate failure-impact profile across system vertices, subject to
a modification budget:

$$\min_{\Delta} \sum_{v \in V} I^*_{\Delta(G)}(v) \quad \text{subject to} \quad \mathrm{Cost}(\Delta) \le \mathcal{B}$$

where $I^*(v)$ denotes the simulated failure impact of component $v$. The candidate set for $\Delta$
is exactly the components and patterns flagged `CRITICAL` or `HIGH` by the catalog of §4. In the
present implementation the modification budget is unconstrained ($\mathcal{B} = \infty$): the engine
emits every mutation whose triggering rule fires over this candidate set, and the aggregate objective
is tracked through the System Risk Index (SRI, §8.4) computed by the verification stage. A policy is
**accepted** if and only if its verified improvement satisfies $\Delta\mathrm{SRI} > 0$ — the
whole-policy acceptance gate actually implemented in `PrescribeService` (`saag/prescription/service.py`)
and documented canonically in `docs/prescription.md`. A stricter per-edit margin criterion,
$\Delta A > \kappa\,\sigma_{\text{seed}}$, is discussed as future work (§11.2) but is not implemented;
under the deterministic simulator configuration used throughout this evaluation, $\sigma_{\text{seed}}
= 0$ (empirically verified across seeds 42–46, §8.3), so the two criteria coincide for the present
results. Budget-constrained policy search — selecting the best subset of mutations under an explicit
cost model rather than firing every triggered rule — is likewise deferred to future work (§11.2).

---

## 6. The SaG-Prescribe Prescriptive Pipeline

### 6.1 Hexagonal Core Abstraction

The system uses a decoupled hexagonal (ports-and-adapters) design separating domain orchestration from
persistence and communication infrastructure. Persistence services implement the `IGraphRepository`
port: production deployments run the Bolt-driven `Neo4jRepository`, while the verification loop and
test suites use an isolated, thread-safe `MemoryRepository` requiring no database instance. This
substitution is what makes repeated counterfactual re-simulation cheap enough for CI/CD integration
(§9.5).

### 6.2 Pipeline Stages

SaG-Prescribe extends the diagnostic pipeline of [1] with a detect–generate–verify loop:

* **Stages 1–5: Diagnostic foundation and anti-pattern detection.** Ingest JSON/YAML topology
  descriptions, compute multi-layered topological metrics, attribute component criticality (§3.4), run
  the twenty-one anti-pattern detectors of §4 over the resulting metric vectors and RMAV scores, model
  failure cascades with a discrete-event simulator, and validate predictive alignment against
  simulation ground truth (§4.5).
* **Stage 6: Prescriptive recommendation generation (this paper).** The engine consumes components and
  patterns categorized `CRITICAL` or `HIGH` by Stage 5 and compiles a policy $\Delta(G)$ from the three
  operators of §6.3. The compiled policy is applied to an exported copy of the topology, and the
  mutated model $G'$ is passed back into the analysis stage under identical fault scenarios and seeds
  to re-evaluate system risk (§6.4).
* **Stage 7: Review interface.** Baseline metrics, detected anti-patterns, prescribed modifications,
  and verification deltas are serialized for the Next.js dashboard (**SMART**), which renders
  side-by-side interactive topologies so architects can review and approve modifications before code
  commitment. Recommendations remain advisory: the human architect is the final authority (§10.2).

### 6.3 Refactoring Operators, Mapped to the Anti-Patterns They Target

Each operator is formalized as a typed graph mutation rule triggered by specific anti-pattern findings
from §4, making the detect-to-prescribe hand-off explicit rather than an unstated correspondence
between two independently-described stages.

**Operator 1 — Logical topic splitting** targets `TOPIC_FANOUT` and topic-hub contributions to
`GOD_COMPONENT`. For a flagged Topic $t$ with publisher set $P(t) = \{a : (a, t) \in
\text{PUBLISHES\_TO}\}$ and $|P(t)| > 1$, the operator replaces $t$ with dedicated sub-topics
$\{t_a : a \in P(t)\}$, rewiring each publisher to its own sub-topic and re-attaching subscriber edges
to the resulting set. This confines each data feed to its target subscribers, bounding the structural
blast radius of the original high-fan-out hub, and duplicates broker routing links accordingly.

**Operator 2 — Physical anti-affinity reallocation** targets `SPOF`, `BROKER_OVERLOAD` (Hub-and-Spoke),
and co-location contributions to `COMPOUND_RISK`. For a physical Node $n$ hosting multiple flagged
components, the operator emits reallocation constraints $(c, n_{\text{from}}, n_{\text{to}})$ moving
each co-located component $c$ beyond the first to an isolating host $n_{\text{to}}$, rewriting the
corresponding `RUNS_ON` edge and duplicating `CONNECTS_TO` links to preserve network reachability. The
emitted constraints correspond directly to container-orchestration anti-affinity scheduling rules.

**Operator 3 — Transport QoS contract hardening** targets `QOS_MISMATCH` and any `CRITICAL`/`HIGH`
topic with a volatile transport configuration (`BEST_EFFORT` reliability, `VOLATILE` durability). The
operator upgrades the contract to `RELIABLE` reliability and `TRANSIENT` durability (raising transport
priority where applicable), hardening the channel against message loss during cascades.

### 6.4 Closed-Loop Verification

The verification engine executes the following loop, guaranteeing complete independence between
candidate generation and the validation path by separating graph structures from simulated metrics:

1. **Export:** serialize the source topology into a flat JSON schema.
2. **Mutate:** apply the compiled policy $\Delta(G)$ (splits, reallocations, upgrades) unconditionally
   to the exported structures.
3. **Sandbox isolation:** seed a temporary, thread-safe `MemoryRepository` with the mutated JSON and
   re-derive `DEPENDS_ON` edges.
4. **Simulation oracle:** run the full analysis–simulation–validation suite on the sandbox model under
   identical fault scenarios and seeds.
5. **Resilience delta quantification:** compute $\Delta\mathrm{SRI} = \mathrm{SRI}_{\text{baseline}} -
   \mathrm{SRI}_{\text{mutated}}$ to report verified structural alignment.

The core threat to structural predictors is circular leakage, where features inadvertently read data
from downstream labels. The framework mathematically avoids this via a strict **independence
guarantee**: all code metrics, RMAV calculations, and anti-pattern detection operate strictly on
$G_{\text{analysis}}$ (the derived projection layers), whereas ground-truth labels and SRI evaluations
are derived separately from raw $G_{\text{structural}}$ simulation waves. No simulation parameters
ever feed back into diagnostic metrics or candidate generation.

---

## 7. DevOps Integration and Delta-Aware CI/CD Gating

### 7.1 Automated Code Review Architecture

To continuously govern structural quality during rapid code evolution, the same underlying engine is
operationalized as a blocking check script (`detect_antipatterns.py`) in continuous integration and
delivery (CI/CD) pipelines, surfacing the twenty-one detectors of §4 directly as a CI check. Whenever
an engineer alters system structures or configures new messaging topology, the gate parses the
"Architecture-as-Code" descriptors and populates an in-memory graph view — reusing the
`MemoryRepository` substrate that makes the prescriptive loop itself CI/CD-viable (§6.1).

### 7.2 Delta-Aware Regression Semantics

Absolute quality gates that fail builds on any critical structural anti-pattern are unsustainable in
industrial software development, since real architectures frequently contain intentional, risk-accepted
debt (e.g. legacy unreplicated components). To resolve this, the gate uses **delta-aware semantics**:
it compares the pull-request candidate topology against the target branch's merge-base topology, and
isolates and flags only *newly introduced* anti-patterns. Pre-existing findings pass unless their
severity worsens, and intentional anomalies can be bypassed via an auditable, time-bound **waiver
register** — mirroring the "Clean as You Code" discipline familiar from code-scope quality platforms,
applied here at topology scope: the gate blocks new architectural debt, while the prescriptive engine
(§6) proposes verified repayments of existing debt.

### 7.3 Exit-Code Protocol

The quality gate enforces automated code review by terminating with standardized exit codes that
command CI/CD pipeline workers:

* **Exit Code 0:** no new anti-patterns detected; build passes, deployment permitted.
* **Exit Code 1:** new `MEDIUM`-severity anti-patterns or QoS warnings introduced; build passes with
  warnings compiled into the developer's code-review dashboard.
* **Exit Code 2:** new, unwaived `CRITICAL` or `HIGH` severity anomalies (e.g. newly introduced
  un-replicated SPOFs or routing loops) discovered; **the build breaks and deployment is blocked**.

Because the recommend-verify loop of §6 runs within CI time budgets (§9.5), prescriptive repayment
proposals can be regenerated on every merge request, keeping the architectural-debt register
synchronized with the evolving topology rather than only gating new regressions.

---

## 8. Experimental Design

### 8.1 Research Questions

* **RQ1 (Detection efficacy and precision):** Does the anti-pattern catalog correlate with
  ground-truth failure impact, and does it avoid over-flagging well-structured systems?
* **RQ2 (Prescriptive efficacy):** Does the closed-loop engine reduce the System Risk Index across
  heterogeneous scenarios, and are the reductions statistically significant?
* **RQ3 (Operator contributions):** How do the individual refactoring operators contribute to the
  observed improvements across topological regimes?
* **RQ4 (Component-level remediation efficacy):** Do system-level SRI improvements translate uniformly
  into component-level failure-impact reductions, or can the greedy, unconditional policy application
  backfire on individual components?
* **RQ5 (Computational overhead and CI/CD feasibility):** What is the wall-clock execution time of the
  detection gate and of the full prescriptive pipeline as system scale grows — and is either
  compatible with CI/CD budgets?

### 8.2 Benchmark Scenarios

Detection validation (§4.5, RQ1) uses an eight-scenario suite (01–08), including a deterministic
"Tiny Regression" smoke-test fixture used as a CI regression sanity check. Prescriptive evaluation
(RQ2–RQ5) uses the seven-scenario subset (01–07) of that suite, spanning realistic domain verticals and
scale presets; the smoke-test fixture is intentionally excluded from prescriptive evaluation since it
carries no domain-representative topology and would not contribute meaningful signal to questions about
prescriptive efficacy or CI/CD runtime at scale.

**Table 8.1 — Scenario scale and topology summary (seven-scenario prescriptive-evaluation subset).**

| Scenario | Applications | Libraries | Topics | Brokers | Nodes | Structural Edges ($|E|$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| S01 Autonomous Vehicle | 80 | 20 | 40 | 4 | 8 | 797 |
| S02 IoT Smart City | 200 | 10 | 80 | 6 | 30 | 1322 |
| S03 Financial Trading | 60 | 18 | 35 | 5 | 6 | 580 |
| S04 Healthcare | 50 | 12 | 25 | 3 | 8 | 400 |
| S05 Hub-and-Spoke | 70 | 25 | 30 | 2 | 12 | 797 |
| S06 Microservices Mesh | 90 | 30 | 45 | 6 | 15 | 680 |
| S07 Hyper-Scale Enterprise | 300 | 50 | 120 | 10 | 40 | 3245 |

This seven-scenario suite is a subset of the eight-scenario preset library used across companion SaG
materials [1]; the eighth preset (an ATM system) is reserved there for external validation of the
diagnostic model against a non-synthetic reference topology and is not reused here, since this paper's
prescriptive evaluation targets the closed-loop pipeline rather than diagnostic external validity.
Replaying SaG-Prescribe on the ATM topology is future work (§11.2).

### 8.3 Experimental Protocol

For each scenario, the prescriptive engine runs in-memory over the topology, measuring the SRI before
and after mutation together with the counts of the three operators. The discrete-event cascade
simulator runs in its default deterministic configuration (cascade propagation threshold 0.2, cascade
probability 1.0, non-Poisson event arrivals). We empirically verified this determinism: re-running the
full generate–verify loop under five candidate seeds (42–46) produced byte-identical SRI values in
every scenario ($\sigma_{\text{seed}} = 0$), so a single run per scenario is reported and paired
comparisons are not confounded by simulation stochasticity under the current configuration. For RQ2, we
test the baseline-vs-mutated SRI pairs across the seven scenarios with a Wilcoxon signed-rank test.

### 8.4 Metrics

**Detection metrics (RQ1).** Spearman rank correlation $\rho$ between the composite criticality score
$Q(v)$ and simulated impact $I(v)$; $F_1$-score, precision, and recall for `CRITICAL`-tier
classification against simulation ground truth; Top-5 overlap between predicted and simulated
rankings.

**System Risk Index (SRI, RQ2–RQ5).** The primary prescriptive outcome measure is a composite risk
index over the four RMAV health dimensions:

$$\mathrm{SRI} = 0.25\,(1 - H_R) + 0.25\,(1 - H_M) + 0.25\,(1 - H_A) + 0.25\,(1 - H_V)$$

where $H_R, H_M, H_A, H_V$ are the system-level Reliability, Maintainability, Availability, and
Vulnerability health scores. Lower SRI indicates lower composite structural risk. We report
$\Delta\mathrm{SRI} = \mathrm{SRI}_{\text{baseline}} - \mathrm{SRI}_{\text{mutated}}$ (positive =
improvement), per-operator recommendation counts, and — separately — component-level failure-impact
deltas to answer RQ4.

---

## 9. Results

### 9.1 Detection Efficacy and Precision (RQ1)

The catalog's detection rules correlate strongly with independent failure-simulation ground truth
across the eight-scenario detection suite: Spearman $\rho(Q, I) = 0.876$ overall, rising to
$\rho = 0.943$ at large scale (150–300+ components); $F_1 = 0.923$, precision $0.912$, recall $0.857$
for `CRITICAL`-tier classification; and Top-5 overlap of $0.80$ (§4.5, Table 4.2). The precision
stress test (Scenario 06, sparse microservices mesh) confirms detectors do not over-flag well-structured
topologies, and the scalability benchmark (Scenario 07, 300+ components) confirms the metrics above
hold at enterprise scale rather than degrading. Pattern-specific evidence — SPOF $F_1 > 0.95$; the
deliberately-encoded Hub-and-Spoke scenario correctly classifying both brokers `CRITICAL` with
broker-failure impact exceeding 50% of system applications — corroborates that these aggregate figures
are not an artifact of averaging over easy and hard cases.

### 9.2 Prescriptive Efficacy (RQ2)

**Table 9.1 — Prescriptive optimization results across scenarios.**

| Scenario | Baseline SRI | Mutated SRI | ΔSRI | Relative Reduction | Splits | Reallocs | Upgrades |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S01 Autonomous Vehicle | 0.3645 | 0.3535 | 0.0110 | 3.0% | 35 | 121 | 0 |
| S02 IoT Smart City | 0.4206 | 0.3537 | 0.0669 | 15.9% | 58 | 276 | 51 |
| S03 Financial Trading | 0.3675 | 0.3482 | 0.0193 | 5.3% | 31 | 88 | 6 |
| S04 Healthcare | 0.3809 | 0.3757 | 0.0052 | 1.4% | 19 | 74 | 6 |
| S05 Hub-and-Spoke | 0.3595 | 0.3527 | 0.0068 | 1.9% | 30 | 97 | 0 |
| S06 Microservices Mesh | 0.3612 | 0.3542 | 0.0070 | 1.9% | 40 | 123 | 0 |
| S07 Hyper-Scale Enterprise | 0.3614 | 0.3469 | 0.0145 | 4.0% | 119 | 409 | 0 |

In all seven scenarios the mutated topology achieves a lower SRI than the baseline (mean $\Delta$SRI =
0.0187). A Wilcoxon signed-rank test over the seven baseline/mutated pairs yields $W = 0.0$,
$p = 0.0156$ — significant at $\alpha = 0.05$, since all seven pairs share the same sign of difference
(the smallest attainable two-sided $p$-value at $n=7$).

The largest improvement occurs in **S02 (IoT Smart City)**, where SRI drops by 0.0669 (15.9%), driven
by 51 QoS upgrades stabilizing high-loss best-effort links (repairing `QOS_MISMATCH` findings, §6.3)
combined with anti-affinity constraints that partition key microservices. The smallest improvement
occurs in **S04 (Healthcare, 1.4%)**: its centralized monitoring fan-outs are intentional and already
run under long durability horizons, leaving less structural headroom for the current operator set. In
**S05 (Hub-and-Spoke)**, topic splitting mitigates the deliberate broker-pair `BROKER_OVERLOAD`
bottleneck (0.3595 → 0.3527).

### 9.3 Operator Contributions (RQ3)

Operator activity tracks each scenario's topological regime and the anti-patterns each scenario
stresses (§6.3):

* **Logical topic splits** dominate where publisher fan-out concentrates: S07 receives 119 splits,
  alleviating `TOPIC_FANOUT` and `GOD_COMPONENT` congestion in heavily shared publisher–subscriber
  hubs.
* **Physical reallocations** are the most frequently emitted mutation overall (409 in S07, 276 in S02),
  reflecting how often `SPOF` and co-location `COMPOUND_RISK` findings arise in dense deployments;
  anti-affinity constraints resolve them without touching logical structure.
* **QoS upgrades** activate almost exclusively in high-loss profiles — 51 in S02 — where hardening
  `QOS_MISMATCH` volatile/best-effort channels yields the single largest SRI gain observed. Scenarios
  already running reliable contracts (S01, S05, S06, S07) receive none, confirming the rule does not
  fire spuriously.

An ablation applying each operator class in isolation would let us attribute $\Delta$SRI per operator
rather than inferring contribution from counts; we flag this as future work (§11.2) rather than
including a speculative table here.

### 9.4 Remediation Efficacy at Component Boundaries (RQ4)

To thoroughly audit the refactoring engine beyond the system-level aggregate, we track individual
component-level failure impacts before and after applying mutations. Because the current implementation
applies the fully compiled policy unconditionally without a per-edit acceptance filter, it delivers a
mixed result that exposes the structural trade-offs of greedy graph mutation:

* **S02 (IoT Smart City):** +47.01% mean reduction in individual component failure impact.
* **S04 (Healthcare):** +30.94% component-level risk improvement.
* **S05 (Hub-and-Spoke):** −31.67% degradation in mean component metrics.
* **S07 (Hyper-Scale Enterprise):** −25.36% regression at internal component boundaries.

This mixed performance — averaging **+4.61%** across the corpus — shows that greedy operators, such as
physical reallocation, can introduce additional network cascade hops (`CONNECTS_TO` extensions) that
backfire on isolated components even while the system-level SRI improves. We report this result
prominently, rather than only the flattering aggregate, because it is the most direct evidence for the
paper's own highest-priority future-work item: a strict per-edit acceptance filter (§11.2).

### 9.5 Computational Overhead and CI/CD Feasibility (RQ5)

**Table 9.2 — Measured wall-clock time of the full analyze–prescribe–verify loop, single run per scenario.**

| Scenario | Elapsed (s) |
| --- | :---: |
| S01 Autonomous Vehicle | 13.4 |
| S02 IoT Smart City | 64.3 |
| S03 Financial Trading | 9.1 |
| S04 Healthcare | 4.7 |
| S05 Hub-and-Spoke | 18.1 |
| S06 Microservices Mesh | 12.1 |
| S07 Hyper-Scale Enterprise | 649.6 |

For six of the seven scenarios (S01, S03–S06), the full analysis–generation–verification loop completes
in under 20 seconds; S02 (1322 structural edges) completes in 64.3 seconds; S07 (300 processes, 3245
structural edges) completes in 649.6 seconds (~10.8 minutes). Bypassing the Neo4j instance with the
in-memory `MemoryRepository` is the enabling optimization. Measured on an Intel Core i7-1370P (14 cores
/ 20 threads), 32 GB RAM, Ubuntu Linux, Python 3.11.5, single-threaded execution.

The standalone CI/CD detection gate script (`detect_antipatterns.py`, which runs the twenty-one
detectors of §4 but not the full generate–verify loop) is considerably faster, since blocking gates
must survive as sub-minute checks without frustrating engineering teams:

* **Tiny / small scales ($\le 25$ vertices):** $< 2$ seconds.
* **Medium scale (~50 components, S04-like):** $\approx 5$ seconds.
* **Large scale (80–100 components):** $\approx 12$ seconds.
* **Xlarge scale (150–300 components, S07-like):** $\approx 40$ seconds.

Taken together, these figures are compatible with pre-deployment CI/CD gating at every evaluated scale:
the detection gate itself stays under a minute even at hyper-scale, and the heavier generate–verify loop
— run less frequently, e.g. nightly or on-demand per merge request rather than on every commit — fits
comfortably within nightly batch windows even at 300 processes.

---

## 10. Discussion and Threats to Validity

### 10.1 What Naming and Verifying Buys

Two results, read together, motivate the paper's two-stage design. §9.1 shows that a *named,
severity-tiered* finding — not just a scalar criticality number — correlates strongly with independent
ground truth, giving architects a specific pathology to act on rather than an opaque score. §9.4 then
shows why acting on a name is not enough on its own: each recommendation must also be *verified*
against a cascade model before it reaches the architect, converting refactoring recommendation from a
suggestion service into a quality-evaluation instrument. An unverified recommender, naming the correct
anti-patterns but skipping verification, would have reported S05 and S07 as unambiguous wins at the
system level while silently degrading a quarter to a third of their individual components.

### 10.2 Positioning in CI/CD and Technical-Debt Workflows

Two claim boundaries govern responsible deployment of the framework. First, **prescriptive
recommendations are advisory**: the pipeline surfaces verified refactoring blueprints for architect
review (§6.2, Stage 7), but does not auto-apply mutations, since verified-in-simulation does not entail
correct-in-production. Second, **blocking-gate claims are reserved for anti-pattern detection** (§7) —
new, unwaived findings introduced relative to the merge base — while composite criticality rankings and
prescriptive recommendations inform but do not block. The gate blocks new architectural debt; the
recommender proposes verified repayments of existing debt. For teams without full CI/CD automation, the
catalog of §4 additionally functions as a structured architecture-review checklist: twenty-one specific,
testable questions about the system's graph structure, in the spirit of design review by checklist as
practiced in aviation and surgery, rather than an informal "does this look healthy?" pass.

### 10.3 Construct Validity

Both detection and verification are defined relative to the same discrete-event cascade simulator:
detection's validation (§4.5, §9.1) demonstrates that named patterns correlate with *simulated* impact,
and SRI reductions (§9.2) demonstrate that prescriptions improve resilience *as the simulator models
it* — neither necessarily as a production system would experience it. We mitigate this by grounding both
the catalog's patterns and the prescriptive operators in mechanisms meaningful independent of the
simulator: established dependability practice (fan-out reduction, anti-affinity scheduling, QoS
hardening) and graph-theoretic reliability results predating this work [18, 19]. Because the same
simulator produces both diagnosis and verification for the prescriptive loop, that loop is internally
consistent by construction; the transfer of verified gains to operational deployments is an external
question (§10.5).

### 10.4 Internal Validity

The prescriptive engine is rule-based and greedy: it emits every triggered mutation rather than
searching the policy space, so reported system-level gains are a lower bound on what an optimizing
search could achieve, and no optimality claim is made. Verification uses identical fault scenarios and
seeds for baseline and mutated topologies, so paired comparisons are not confounded by scenario
sampling. The independence guarantee of §6.4 rules out circular leakage between diagnostics and
ground-truth labels. Per-type label degeneracy (passive shared libraries pooled with active
applications) can mask signals in aggregate statistics; as in [1], we treat stratified reporting as
mandatory wherever type-level or pattern-level results are given, and both §4.5's pattern-specific
evidence and §9.4's component-level breakdown are instances of this discipline applied to detection and
prescription respectively.

### 10.5 External Validity

All eight (detection) and seven (prescription) scenarios are parameterized synthetic topologies. While
the presets mimic representative domain verticals, they may not capture the runtime complexity of
industrial clusters — dynamic workload shifts, packet-loss bursts, transient hardware faults. The
catalog itself is scoped to the publish–subscribe communication paradigm: systems combining pub-sub
with request-response patterns (hybrid microservices, mixed REST/event architectures) will require
additional patterns addressing the request-response side, and `QOS_MISMATCH` is currently specified for
DDS/ROS 2 and MQTT QoS-weight semantics, requiring adaptation for other middleware platforms. Validating
prescriptions against a real system (e.g. replaying the engine on the ATM topology reserved for external
validation in [1]) is the most direct extension.

### 10.6 Conclusion Validity

The prescriptive-evaluation scenario sample is small ($n=7$); the Wilcoxon result reported in §9.2
($W=0.0$, $p=0.0156$) is the smallest attainable two-sided $p$-value at this sample size, since all
seven deltas share the same sign — it should be read as consistent directional evidence rather than as
a claim of large effect size. Results are single-run per scenario under a verified-deterministic
simulator configuration ($\sigma_{\text{seed}}=0$, §8.3); this determinism is a property of the current
cascade-probability-1.0 configuration and would need re-establishing under the stochastic propagation
model proposed as future work (§11.2).

### 10.7 Engineering Trade-offs

The closed-loop verification step adds computational cost over emitting unverified recommendations.
Since the framework targets pre-deployment optimization rather than runtime alerting, §9.5 shows this
cost stays within CI/CD budgets at every evaluated scale, and we consider verification the feature that
distinguishes a prescription from a suggestion — just as naming a finding against an empirically
validated catalog is the feature that distinguishes a diagnosis from an unlabeled score.

---

## 11. Conclusion and Future Work

### 11.1 Conclusions

This paper presented SaG-Prescribe, a graph-based framework that unifies the detection of named,
severity-tiered architectural anti-patterns with closed-loop, DevOps-integrated prescriptive
refactoring for distributed publish–subscribe architectures. The twenty-one-pattern catalog gives
practitioners a pub-sub-specific vocabulary analogous to established object-oriented and microservices
smell catalogs, with detection rules empirically validated against failure-simulation ground truth
(Spearman $\rho = 0.876$ overall, $0.943$ at scale; $F_1 = 0.923$). By augmenting these topological
diagnostics with a code-quality bridge (CQP) and extending the Software-as-a-Graph baseline, we
introduced an end-to-end pipeline that translates named findings into three concrete mutation operators
— topic splitting, anti-affinity reallocation, and QoS hardening — verifies every compiled policy
against the same discrete-event cascade oracle that produced the diagnosis, and operationalizes the
result as a delta-aware CI/CD quality gate. Across seven benchmark scenarios, the generated
prescriptions reduce the System Risk Index in every case (1.4%–15.9% relative, Wilcoxon $W=0.0$,
$p=0.0156$) within runtime envelopes compatible with merge-request-gated or nightly CI/CD pipelines,
while an honest component-level audit (§9.4) shows the current greedy policy is not yet uniformly
beneficial below the system level.

### 11.2 Future Work

1. **Per-edit acceptance filter (highest priority):** eliminating counterproductive mutations before
   policy execution, directly motivated by the component-level regressions of §9.4.
2. **Budget-constrained policy search:** incorporating explicit cost models and modification budgets
   ($\mathcal{B} < \infty$ in §5), turning the engine from exhaustive rule firing into subset selection
   over candidate mutations.
3. **Stochastic cascade propagation:** the current simulator is deterministic under its default
   configuration (cascade probability 1.0), so seed variance is trivially zero (§8.3). Introducing
   probabilistic propagation ($<1.0$) would make the $\kappa\cdot\sigma_{\text{seed}}$ acceptance
   criterion of §5 load-bearing and let RQ2 test robustness to genuine simulation noise.
4. **Operator ablation and learned policy ordering:** isolating per-operator $\Delta$SRI contributions
   and exploring learned prioritization over the rule-based candidate set.
5. **Catalog extension:** generalizing the pattern set to hybrid REST/event architectures and to
   middleware platforms whose QoS semantics differ from the DDS/ROS 2/MQTT weight formula
   `QOS_MISMATCH` currently assumes (§10.5).
6. **Real-system replication:** applying the engine to the ATM topology of [1] and to harvested
   industrial configurations, closing the external-validity gap of §10.5.
7. **LLM-assisted pull-request generation:** linking large language model code assistants to
   automatically generate pull requests that implement the synthesized architectural refactoring
   blueprints. This is the paper's only LLM-adjacent contribution and should not be oversold beyond
   this single future-work item — the paper makes no LLM or AutoML contribution today.

---

## References

`[1]` [Authors]. *Software-as-a-Graph: A Static System Analysis Framework for Pre-Deployment Quality
Gating and Failure Simulation of Publish-Subscribe Middleware.* Journal of Systems and Software, under
review / to appear. [Update status at submission time; AuSE permits citing companion work under review
with a copy supplied to the editor. Confirm submission status and disclose per AuSE's
originality/overlap policy in the cover letter.]

`[2]` M. Harman, S. A. Mansouri, Y. Zhang, "Search-Based Software Engineering: Trends, Techniques and
Applications," *ACM Computing Surveys*, 45(1), Article 11, 2012.

`[3]` A. Aleti, B. Buhnova, L. Grunske, A. Koziolek, I. Meedeniya, "Software Architecture Optimization
Methods: A Systematic Literature Review," *IEEE Transactions on Software Engineering*, 39(5), 658–683,
2013.

`[4]` S. Pallickara, H. Bulut, G. Fox, "Fault-Tolerant Reliable Delivery of Messages in Distributed
Publish/Subscribe Systems," *Proc. 4th IEEE International Conference on Autonomic Computing (ICAC
2007)*, 2007.

`[5]` T. Chang, S. Duan, H. Meling, S. Peisert, H. Zhang, "P2S: A Fault-Tolerant Publish/Subscribe
Infrastructure," *Proc. 8th ACM International Conference on Distributed Event-Based Systems (DEBS
2014)*, 2014.

`[6]` G. Wang, J. Koshy, S. Subramanian, K. Paramasivam, M. Zadeh, N. Narkhede, J. Rao, J. Kreps, J.
Stein, "Building a Replicated Logging System with Apache Kafka," *Proceedings of the VLDB Endowment*,
8(12), 1654–1655, 2015.

`[7]` S. Lee, H.-S. Park, J. Chae, K.-J. Park, "Probabilistic Latency Analysis of the Data Distribution
Service in ROS 2," *arXiv:2508.10413*, 2025.

`[8]` S. Lee, J. Kang, K.-J. Park, "Dependency Chain Analysis of ROS 2 DDS QoS Policies: From Lifecycle
Tutorial to Static Verification," *arXiv:2509.03381*, 2025.

`[9]` L. C. Freeman, "A set of measures of centrality based on betweenness," *Sociometry*, vol. 40, no.
1, pp. 35–41, 1977.

`[10]` A. Bakhtin, M. Esposito, V. Lenarduzzi, D. Taibi, "Network Centrality as a New Perspective on
Microservice Architecture," *Proc. IEEE International Conference on Software Architecture (ICSA
2025)*, 72–83, 2025.

`[11]` D. H. M. Falci, O. A. Gomes, F. S. Parreiras, "Complex Networks Analysis for Software
Architecture: an Hibernate Call Graph Study," *arXiv:1706.09859*, 2017.

`[12]` T. L. Saaty, *The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation*,
McGraw-Hill, 1980.

`[13]` M. Fowler, *Refactoring: Improving the Design of Existing Code*, Addison-Wesley, 1999.

`[14]` W. H. Brown, R. C. Malveau, H. W. McCormick, T. J. Mowbray, *AntiPatterns: Refactoring Software,
Architectures, and Projects in Crisis*, Wiley, 1998.

`[15]` G. Suryanarayana, G. Samarthyam, T. Sharma, *Refactoring for Software Design Smells: Managing
Technical Debt*, Morgan Kaufmann, 2014.

`[16]` C. Richardson, *Microservices Patterns: With Examples in Java*, Manning, 2018.

`[17]` D. Taibi, V. Lenarduzzi, C. Pahl, "Microservices anti-patterns: A taxonomy," in *Microservices:
Science and Engineering*, Springer, 2020.

`[18]` C. Y. Baldwin, K. B. Clark, *Design Rules, Volume 1: The Power of Modularity*, MIT Press, 2000.

`[19]` C. J. Colbourn, *The Combinatorics of Network Reliability*, Oxford University Press, 1987.

`[20]` M. M. Lehman, "Laws of software evolution revisited," *Proceedings of EWSPT '96*, Springer, 1996.

`[21]` R. C. Martin, *Agile Software Development, Principles, Patterns, and Practices*, Prentice Hall,
2003.

`[22]` M. T. Nygard, *Release It! Design and Deploy Production-Ready Software* (2nd ed.), Pragmatic
Bookshelf, 2018.

> *[Reference-list note, not part of the manuscript: references [2]–[12] were sourced from the
> previously verified prescriptive/SBSE/pub-sub bibliography; references [13]–[22] are sourced from
> `docs/antipatterns.md`'s own verified bibliography and are real, non-invented candidates, though full
> text was not re-read for each in this pass — sanity-check relevance before submission. AuSE reviewers
> will expect ~30–45 references total; the two `[REF: …]` placeholders in §2.3 (learning-based and
> LLM-based refactoring recommenders) still need populating with real citations, and this list should
> otherwise be expanded further before submission.]*

## Declarations

- **Funding:** [to be completed]
- **Competing interests:** [to be completed]
- **Data availability:** A replication package containing scenario configurations, seeds, and the
  prescriptive pipeline implementation will be made available at [URL pending].
- **Ethics approval:** Not applicable.
