# Graph-Based Detection of Architectural Anti-Patterns and Prescriptive Refactoring in Distributed Publish–Subscribe Systems

**Target venue:** Automated Software Engineering (AuSE), Springer — Special Issue on *Intelligent
techniques for CI/CD, DevOps, software evolution, technical debt analysis, and refactoring
recommendation*.
**SI topic mapping:** this submission covers three of the bullet's five named areas directly —
*technical debt analysis* (the anti-pattern catalog, §4), *refactoring recommendation* (the verified
prescriptive operators, §6), and *CI/CD/DevOps* (the quality gate, §7). No AutoML/NAS/LLM contribution
is claimed.

> **Provenance note (not part of the manuscript).** This revision realigns the manuscript with the
> post-revision implementation and re-measures its detection claims. Four changes are substantive:
>
> 1. **Detection validation is now measured here rather than carried forward.** The figures previously
>    reported in §4.5/§9.1 (ρ = 0.876, ρ = 0.943 at scale, F₁ = 0.923, precision 0.912, recall 0.857,
>    Top-5 0.80) originated in a prior conference publication and had no reproducible artifact in this
>    repository. §9.1 now reports measurements produced by `reproduce/detection_validation.py` against
>    a committed oracle, in `results/detection_validation.json` (system layer) and
>    `results/detection_validation_app.json` (app layer). **The measured figures are substantially
>    weaker**, and RQ1 is reported as a negative result.
> 2. **The per-edit acceptance filter is implemented.** §5, §6.4, §9.2–§9.4 and §11.2 previously
>    described it as unimplemented future work and reported unfiltered results (ΔSRI 1.4%–15.9% across
>    seven scenarios, Wilcoxon p = 0.0156; mean component-level +4.61% concealing −31.67% and −25.36%
>    regressions). Those are superseded by the filtered results in `results/prescribe_all.log`.
> 3. **§7 is rescoped.** Delta-aware merge-base semantics and the waiver register are specified but
>    not implemented; §7.2 now says so, and the corresponding evaluation claims are removed from §9.5.
> 4. **§9.5's generate–verify runtimes are withdrawn**, having been measured under the unfiltered
>    design, and replaced with measured detection-gate timings plus the per-edit cost model.
>
> Two items remain open before submission. §2.3 retains two `[REF: …]` citation-slot placeholders
> (learning-based and LLM-based refactoring recommenders) — these are explicitly *not* invented
> citations and must be populated from a real bibliography. And the overlap disclosure in §1.5 must be
> checked against the companion submission's final scope.

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
criticality without naming the architectural pathology at fault or producing verified guidance on how
to repair it. We address both gaps with **SaG-Prescribe**, a graph-based framework that (1) specifies
twenty-one named, severity-tiered publish–subscribe anti-patterns and bad smells — including Single
Point of Failure, God Component, Broker Saturation, Chatty Pair, and QoS Policy Mismatch — as formal
topological signatures over a structural metric vector with adaptive box-plot thresholds; and (2)
compiles the resulting diagnosis into a transformation policy of three graph-mutation operators
(logical topic splitting, physical anti-affinity reallocation, transport QoS contract hardening), in
which **every candidate edit is verified independently** on its own counterfactual graph against a
discrete-event cascade simulator, and only edits whose measured impact reduction exceeds the
simulator's own seed noise at every propagation threshold are surfaced as recommendations.

We evaluate both stages against a committed simulation oracle and report two negative results
prominently, because both change what the framework can be claimed to do. **Detection is weaker than
prior reporting suggested.** Across eight benchmark scenarios the composite criticality score reaches
mean Spearman $\rho = 0.268$ against simulated cascade impact — below betweenness centrality ($0.295$)
and degree centrality ($0.417$) alone — and the catalog implicates a mean of 90.4% of components at
`CRITICAL`/`HIGH` severity, including 70% on the sparse topology chosen as a precision stress test. The
pattern specifications are sound; their thresholds are not yet calibrated to produce a usable
shortlist. **Verification admits far less than generation proposes, and changes the causal account.**
Of 213 candidate edits across six scenarios, 29 (13.6%) survive per-edit verification, all of them
topic splits: no anti-affinity reallocation and no QoS upgrade clears the margin anywhere, contradicting
the previously reported attribution of the suite's best result to QoS hardening. Five of six scenarios
admit nothing; one improves clearly ($\Delta\mathrm{SRI} = +0.0365$); and one shows three
individually verified edits composing into a marginal regression, bounding but not eliminating that
failure mode. The defensible scope for the prescriptive stage is fan-out decomposition where a fan-out
bottleneck actually exists. Detection runs in 0.03–24.6 s from a 12-component fixture to a
300-application enterprise topology, with the detectors themselves accounting for under a second in
total, making per-commit gating feasible once threshold calibration and delta-aware semantics — designed
and reported here, but not yet implemented — are in place.

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
publish–subscribe anti-patterns and bad smells, each given a formal topological detection rule over a
structural metric vector with adaptive box-plot thresholds. Second, it **prescribes**: components and
patterns flagged `CRITICAL` or `HIGH` — a classification itself informed by a **Code Quality Penalty
(CQP)** computed from static-analysis metrics — feed a rule-based engine that generates targeted
architectural mutations along three vectors: (1) logical topic splitting, decomposing high-fan-out
publish channels; (2) physical anti-affinity reallocation, separating co-located critical components;
and (3) transport QoS contract hardening, upgrading volatile/best-effort channels to reliable,
transient-local settings.

Crucially, SaG-Prescribe implements **two-level closed-loop verification**. Each candidate edit is
first applied *alone* to a counterfactual copy of the topology and re-simulated across a propagation-
threshold sweep and a seed set; it is admitted only if its measured cascade-impact reduction exceeds
the simulator's own seed noise at every threshold. Only the admitted subset is then applied jointly
and re-evaluated end to end. A recommendation therefore reaches the architect having been measured
individually, and the framework reports an empty recommendation set — which happens often — rather
than manufacturing one.

Third, because the loop runs against an in-memory repository with no database dependency, the same
underlying engine is operationalized as a **CI/CD quality gate**. We describe the gate's intended
delta-aware semantics, which block only findings newly introduced relative to a Git merge base and
leave pre-existing, risk-accepted debt untouched unless its severity worsens; §7.2 states plainly that
these semantics are specified but not yet implemented, and §9.1 explains why implementing them is a
prerequisite for deployment rather than a refinement.

### 1.4 Contributions

1. **A catalog of twenty-one publish–subscribe anti-patterns and bad smells**, each with a formal,
   topology-based detection rule over a structural metric vector with adaptive box-plot thresholds,
   organized into three severity tiers and four quality dimensions (§4).
2. **A prescriptive refactoring pipeline with per-edit counterfactual verification**, translating the
   catalog's findings into graph mutations via three named operators, each explicitly mapped to the
   patterns it targets, where every candidate is measured alone against a discrete-event cascade
   simulator and admitted only if it beats that simulator's own noise at every propagation threshold
   (§5–§6). We disclose the automation footprint precisely rather than by implication: five of the
   twenty-one patterns are directly wired to an operator; the other sixteen remain advisory.
3. **A CI/CD quality gate**, with a three-tier exit-code protocol and designed delta-aware, waiver-
   registered semantics, reported with its implementation status stated explicitly for each half (§7).
4. **An empirical evaluation that reports two negative results as its principal findings** (§8–§9):
   detection correlates with simulated impact only weakly and is out-performed by degree centrality
   alone, while implicating ~90% of components; and per-edit verification admits 13.6% of generated
   candidates, none of them from two of the three operators, overturning this work's own earlier causal
   account of its best-performing scenario.
5. **An account of the measurement itself.** The detection figures previously reported for this
   framework were inherited from a prior publication without a reproducible artifact. We supply the
   harness (`reproduce/detection_validation.py`), report what it measures, and document a pooling
   effect strong enough to flip the sign of a system-layer correlation — a methodological hazard for
   anyone scoring criticality on a heterogeneous graph.

### 1.5 Relationship to the Authors' Prior Work

This submission builds on two prior efforts by the same authors, and we state the boundaries because
they bear on originality and overlap assessment.

A companion manuscript [1] introduces the Software-as-a-Graph model itself — the typed multigraph, the
`DEPENDS_ON` projection rules, the RMAV attribution, the failure simulators, and a learned
heterogeneous-GNN criticality predictor. This paper *consumes* that model (§3 summarizes only what the
detection and prescription stages need) and contributes what [1] does not contain: the anti-pattern
catalog and its detection methodology (§4), the operator-to-pattern mapping and its disclosed coverage
gaps (§6.3), the CI/CD gate design (§7), and the detection validation of §9.1. The prescriptive
verification machinery of §5–§6 is shared infrastructure described in both; the *evaluation* of it
here is specific to this paper's operator set and scenario subset. Neither manuscript's claims depend
on the other's results.

Earlier detection figures for this framework were published in a conference paper by the same authors.
As §9.1 and the provenance note record, those figures are not reproduced here: we re-measured rather
than restated, and the measured values are weaker. We regard reporting that difference as part of this
paper's contribution rather than an embarrassment to be smoothed over.

### 1.6 Organization

Section 2 surveys related work. Section 3 formalizes the graph model and the code-quality bridge.
Section 4 presents the anti-pattern catalog and its validation methodology. Section 5 defines the
closed-loop optimization objective and its acceptance criteria. Section 6 presents the prescriptive
pipeline and its operators. Section 7 describes the CI/CD gate. Sections 8 and 9 present the
experimental design and results. Section 10 discusses implications and threats to validity, and
Section 11 concludes.

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
re-checked continuously as a system evolves, rather than assessed once at initial design.

The standard argument for moving beyond single-metric centrality is that classical measures assume
uniform edge semantics and should therefore degrade on pub-sub layers, where decoupled endpoints are
separated by high-fan-out topics, brokers, and heterogeneous QoS policies. We state that argument here
because it motivates the typed multigraph model, and then note that **this paper's own measurements do
not confirm it**: on our suite, degree centrality alone out-ranks the four-dimensional composite
against simulated cascade impact (§9.1). Readers should weigh §4's decomposition on the explanatory
grounds developed in §10.1 rather than on an assumed accuracy advantage over simple centrality.

What these approaches share, and what motivates this paper regardless of that result, is that they
identify fragility without naming the specific pathology or verifying a remedy — the step this paper
automates.

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

These four profiles blend into a composite criticality score $Q(v)$ using pairwise-comparison weights
on Saaty's 1–9 scale [12], checked for internal consistency and then mixed with a uniform prior
($\lambda = 0.70$) to prevent extreme parameter concentration. The raw weights are $(0.43, 0.24, 0.17,
0.16)$ and the *applied* weights after shrinkage are $(0.395, 0.247, 0.193, 0.165)$ for availability,
reliability, maintainability and vulnerability respectively; the shrinkage is applied to the
intra-dimension weight vectors as well as to the composite, so $\lambda$ moves every formula above at
once. We describe these as stated design judgements checked for consistency rather than as elicited
from raters: the near-zero consistency ratios are a symptom of matrices written from a target weight
vector, and the framework's own sensitivity sweep finds that equal weights out-perform the calibrated
ones on ranking accuracy. This is a further reason the decomposition's contribution is scoped to
attribution rather than accuracy (§10.1). Composite scores are mapped to five criticality tiers
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

Detection operates over a per-component structural metric vector $M(v)$. Fifteen of its fields are
*Tier-1* metrics — those that feed an RMAV dimension directly — organized by the dimension they
serve: Reverse PageRank $\mathrm{RPR}$, in-degree $\mathrm{DG}_{\text{in}}$, multi-path coupling index
$\mathrm{MPCI}$ and topic fan-out criticality $\mathrm{FOC}$ (Reliability); betweenness
$\mathrm{BT}$, QoS-weighted out-degree $w_{\text{out}}$, clustering coefficient $\mathrm{CC}$ and path
complexity $\mathrm{PC}$ (Maintainability); directed articulation score
$\mathrm{AP}_{c,\text{directed}}$, bridge ratio $\mathrm{BR}$, connectivity degradation index
$\mathrm{CDI}$ and component weight $w$ (Availability); and reverse eigenvector $\mathrm{REV}$,
reverse closeness $\mathrm{RCL}$ and QoS-weighted in-degree $w_{\text{in}}$ (Vulnerability). The Code
Quality Penalty of §3.3 is a sixteenth Tier-1 input, entering through Maintainability. Forward-facing
centralities (PageRank, closeness, eigenvector) are computed but held at Tier 2 — informative for
visualization, deliberately not fed to the RMAV formulas, since their reverse counterparts on $G^T$
are the failure-propagation-relevant direction.

Topological metrics use **rank-based normalization** by default, on the argument that they are highly
skewed (a single hub-broker may have betweenness $50\times$ the median, which min-max scaling would
compress everything else beneath); linear code and hardware properties use min-max, since their
absolute magnitudes are meaningful. We flag one measured consequence of that default rather than
leaving it as design rationale, because it bears directly on §9.1: rank normalization discards
magnitude before the RMAV weighted sum, which makes $Q(v)$ closer to a Borda count over the Tier-1
metrics than to a weighted aggregate of them, and a sweep of the alternatives shows the default costs
roughly $0.195$ Spearman $\rho$ against magnitude-preserving normalization. Because rank-normalized
inputs are near-uniform on $[0,1]$ by construction, the box-plot classifier below also produces a
fairly stable critical fraction almost regardless of topology — which is part of the explanation for
the over-flagging reported in §9.1.

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
topology anti-pattern: a broker whose availability score reaches at least $2\times$ the median
broker's, or the sole broker in a system, flagged unconditionally.

This pattern also supplies the catalog's clearest example of a specification that is defensible in
principle and mis-specified in practice, so we use it rather than a success as the walkthrough's
cautionary case. One of our eight scenarios (S05) deliberately encodes broker saturation: two brokers
serve seventy applications across twelve nodes. The detector finds **nothing** there. Both brokers
score $Q = 0.206$ and classify `MINIMAL`, and their measured cascade impact is $0.171$ and $0.191$
against a suite-wide maximum of $0.331$ — elevated, but nowhere near a level that would single them
out. The reason is visible in the rule: with exactly two comparably-loaded brokers, each *is* the
median, so the $2\times$-median test can never fire, and the sole-broker branch does not apply either.
A rule keyed on within-population relative load is structurally blind to the case where the whole
population is overloaded. Detecting this scenario requires an absolute or cross-system referent —
applications-per-broker, or routing load against broker capacity — which the current specification
does not have. §11.2 folds this into the threshold-recalibration item.

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

### 4.5 Validation Methodology for Detection

Findings are validated empirically against the failure simulation pipeline: for each component, the
simulated impact score $I_{\text{comp}}(v)$ — computed by exhaustive single-component removal and
cascade propagation — provides evidence, independent of the detectors' own inputs, about whether a
topological signature corresponds to structural risk that materializes under failure. The measurement
protocol is given in §8.3, the metric definitions in §8.4, and the results in §9.1. We keep them there
rather than previewing headline figures here, because the results are mixed and a summary at this
point in the paper would have to be either misleading or long enough to belong in §9.

Detection validation uses an eight-scenario suite (S01–S08) spanning autonomous-vehicle, IoT,
financial, healthcare, deliberately-anti-pattern (Hub-and-Spoke), microservices, and enterprise
topologies, plus a deterministic "Tiny Regression" smoke-test fixture. Two scenarios were designed to
play a distinguished role. **S06** (sparse microservices mesh) is the **precision stress test**: a
well-structured topology should produce few findings, so the catalog's finding volume there is a
direct test of whether the detectors over-flag. **S07** (300+ components) is the **scalability
benchmark**. §9.1 reports that S06 does not behave as intended.

The prescriptive evaluation of §9.2–§9.4 uses a six-scenario subset. It excludes the smoke-test
fixture, which carries no domain-representative topology, and — on measured cost grounds stated in
§9.2 — the enterprise topology.

One methodological point deserves stating before the results, because it constrains what any of them
can mean. Detection is validated against the *same class of simulator* that the prescriptive stage
uses for verification. The catalog's rules are computed on $G_{\text{analysis}}$ and the labels on
$G_{\text{structural}}$, so there is no feature–label feedback; but both are deterministic functions
of one topology under one cascade model, so agreement between them is not evidence that either matches
a production system. §10.3 develops this.

---

## 5. Closed-Loop Optimization Objective

The prescriptive task is to compute a transformation policy $\Delta$ producing a mutated topology
$G' = \Delta(G)$ that minimizes the aggregate failure-impact profile across system vertices, subject to
a modification budget:

$$\min_{\Delta} \sum_{v \in V} I^*_{\Delta(G)}(v) \quad \text{subject to} \quad \mathrm{Cost}(\Delta) \le \mathcal{B}$$

where $I(v)$ denotes the simulated failure impact of component $v$. The candidate set for $\Delta$ is
exactly the components and patterns flagged `CRITICAL` or `HIGH` by the catalog of §4. In the present
implementation the modification budget is unconstrained ($\mathcal{B} = \infty$) at the *generation*
stage: the engine emits every mutation whose triggering rule fires over the candidate set. What
constrains the policy is not a budget but a two-level acceptance test, both levels of which are
implemented in `PrescribeService` (`saag/prescription/service.py`) and `EditVerifier`
(`saag/prescription/verifier.py`):

1. **Per-edit acceptance.** Each candidate edit $\delta$ is applied *alone* to a counterfactual copy
   of the graph and simulated across a propagation-threshold sweep $\Theta$ and a seed set $S$. It is
   retained only if

   $$\overline{\Delta I}_\theta(\delta) > \kappa \cdot \sigma_{\text{seed},\theta}(\delta)
   \qquad \text{for every } \theta \in \Theta$$

   where $\overline{\Delta I}_\theta$ is the mean reduction in cascade impact at threshold $\theta$,
   paired against a baseline measured at the same $(\theta, s)$, and $\sigma_{\text{seed},\theta}$ is
   the across-seed standard deviation at that same threshold. Requiring the margin at *every*
   threshold prevents an edit being admitted because it happened to help at the canonical $0.2$
   default; requiring it to exceed $\kappa\sigma$ prevents simulator noise being read as improvement.
   Deliberately, $\sigma$ is estimated per threshold and not pooled across thresholds, since the
   quantity being tested is a per-threshold claim.

2. **Whole-policy acceptance.** The accepted subset — and only that subset — is applied jointly, and
   the resulting topology is re-evaluated end to end. The policy is reported as accepted iff
   $\Delta\mathrm{SRI} = \mathrm{SRI}_{\text{baseline}} - \mathrm{SRI}_{\text{mutated}} > 0$.

The second level is not redundant with the first. Per-edit verification admits edits one at a time;
it cannot observe how an admitted *set* composes. §9.4 reports a scenario where three individually
verified edits together left the System Risk Index marginally worse, which is exactly the case the
whole-policy gate exists to catch and report.

An empty accepted set is a valid and informative outcome, reported as such with the baseline
unchanged rather than dressed up as a no-op improvement. Budget-constrained policy search — selecting
the best subset under an explicit cost model rather than filtering an exhaustively generated candidate
set — remains future work (§11.2).

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
  patterns categorized `CRITICAL` or `HIGH` by Stage 5 and compiles a *candidate* policy $\Delta(G)$
  from the three operators of §6.3. Each candidate is then verified in isolation and only the
  surviving subset is applied and re-evaluated (§6.4).
* **Stage 7: Review interface.** The output of a `prescribe()` call is a remediation blueprint: the
  itemized list of applied changes, the per-edit verdicts for everything declined (each carrying its
  measured $\Delta I$, $\sigma_{\text{seed}}$ and rejection reason), and before/after metrics. It is
  reachable from the SDK and the CLI (`cli/prescribe_graph.py`); unlike the diagnostic stages, Stage 6
  has no REST router and is not rendered in the project's dashboard. Recommendations remain advisory:
  the human architect is the final authority (§10.2).

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

**Operator 3 — Transport QoS contract hardening** fires on any `CRITICAL`/`HIGH` topic, or topic
adjacent to a `CRITICAL`/`HIGH` component, whose transport configuration is volatile (`BEST_EFFORT`
reliability or `VOLATILE` durability). The operator upgrades the contract to `RELIABLE` reliability
and `TRANSIENT` durability, hardening the channel against message loss during cascades. Topics already
running hardened contracts are skipped.

#### Automation coverage is narrower than the catalog

The detect-to-prescribe hand-off is worth stating precisely, because a reader could otherwise assume
that a twenty-one-pattern catalog implies twenty-one automated repairs. It does not. Two independent
signals feed the operator triggers, and only one of them names a catalog entry:

* **Generic criticality tier.** Any component classified `CRITICAL`/`HIGH` on the RMAV dimensional
  scale can trigger any operator, irrespective of which specific anti-pattern — if any — was detected
  on it.
* **Detected-problem name matching.** The only channel that ties a mutation back to a particular
  catalog entry, implemented as substring matching over `DetectedProblem.name`, the human-readable
  `PatternSpec.name`.

Following the second channel through to the catalog, exactly **five of the twenty-one patterns** reach
an operator: `SPOF` → anti-affinity reallocation; `GOD_COMPONENT`, `BOTTLENECK_EDGE`, `FAILURE_HUB`
and `HUB_AND_SPOKE` → topic splitting. Notably, `QOS_MISMATCH` has **no** link to QoS hardening
despite the conceptual overlap; Operator 3 fires only from the generic criticality tier. The remaining
sixteen patterns have no automated operator: their `PatternSpec.recommendation` text is advisory, for
a human to act on.

Part of this boundary is principled — breaking a dependency cycle correctly, or deciding which
pipeline stages are safe to merge, requires knowing *what* a component does, not merely how it is
wired, and only remediations expressible as pure topology or QoS mutations are automatable. Part of it
is an implementation artifact we do not wish to present as design: `DetectedProblem` carries no
`pattern_id` field, so the linkage travels through display strings and silently unbinds if a pattern
is renamed. §10.3 records this as a construct-validity threat rather than a limitation of the
approach.

### 6.4 Closed-Loop Verification

The verification engine executes the following loop. Steps 2 and 6 implement the two acceptance levels
of §5; the intervening steps guarantee that candidate generation and the validation path never share
state.

1. **Baseline.** Run the source graph through analyze → simulate → validate, producing a baseline
   System Risk Index and a per-component cascade impact map $I(v)$.
2. **Per-edit acceptance filter.** Apply each candidate edit *alone* to a counterfactual copy of the
   exported topology and simulate it across the threshold sweep $\Theta$ and seed set $S$. Keep it
   only if $\overline{\Delta I} > \kappa\sigma_{\text{seed}}$ at every threshold. Rejected edits never
   reach the mutated graph, and each carries a verdict recording its measured $\Delta I$,
   $\sigma_{\text{seed}}$ and the binding threshold at which it failed.
3. **Mutate in memory.** Export the graph to flat JSON and apply the *accepted subset* of $\Delta(G)$
   to that JSON — never to the production graph store.
4. **Sandbox isolation.** Load the mutated JSON into a temporary, thread-safe `MemoryRepository` and
   re-derive `DEPENDS_ON` edges from scratch.
5. **Simulation oracle.** Re-run the full analysis–simulation–validation suite on the sandbox model
   under the same fault scenarios and seeds as the baseline.
6. **Whole-policy gate.** Compute $\Delta\mathrm{SRI} = \mathrm{SRI}_{\text{baseline}} -
   \mathrm{SRI}_{\text{mutated}}$ and mark the policy accepted iff $\Delta\mathrm{SRI} > 0$.

A rejected policy is still returned in full, with its before/after metrics and every per-edit verdict,
for the architect to inspect; nothing is silently discarded. One measurement cost is worth naming here
because it dominates the runtime of §9.5: step 2 requires one exhaustive simulation sweep per
$(\text{edit} \times \text{threshold} \times \text{seed})$ triple, so verification is roughly
$|\Theta| \cdot |S|$ times more expensive per candidate than the unfiltered design it replaces.

The core threat to structural predictors is circular leakage, where features inadvertently read data
from downstream labels. The framework avoids this via a strict **independence guarantee**: all code
metrics, RMAV calculations, and anti-pattern detection operate on $G_{\text{analysis}}$ (the derived
projection layers), whereas ground-truth labels and SRI evaluations are derived separately from raw
$G_{\text{structural}}$ simulation waves. No simulation result feeds back into candidate generation
within a run. We note the honest scope of this guarantee in §10.4: it is *view* independence, not
independence of data source, since both views are deterministic functions of the same topology.

---

## 7. DevOps Integration and CI/CD Gating

This section describes the gate's design and reports what is implemented today. We separate the two
explicitly, because the implementation status differs between the two halves of the design and a
reader deciding whether to adopt this needs to know which is which.

### 7.1 Automated Code Review Architecture

To govern structural quality during rapid code evolution, the detection stage is operationalized as a
blocking check in continuous integration and delivery (CI/CD) pipelines, surfacing the catalog's
detectors directly as a CI check. Whenever an engineer alters system structure or configures new
messaging topology, the gate parses the "Architecture-as-Code" descriptors, builds the graph view,
computes the RMAV attribution, and runs the detectors. §9.5 reports the measured wall-clock cost of
exactly this work.

**Implementation status.** The functioning path is the detection stage invoked through
`cli/predict_graph.py`, which runs the detectors and sets the process exit code from the highest
severity found. A separate dedicated gate entry point (`cli/detect_antipatterns.py`) also exists but
currently passes the analysis result — rather than the prediction result that carries the detected
problems — into its detection call, so it reports no findings; the working path is the one measured in
§9.5. Neither path is yet wired into this project's own CI workflow.

### 7.2 Regression Semantics: Absolute Today, Delta by Design

Absolute quality gates that fail a build on *any* critical structural anti-pattern are unsustainable
in industrial development, because real architectures carry intentional, risk-accepted debt — a legacy
unreplicated component that the team has consciously chosen to live with. §9.1 makes this concrete
rather than hypothetical: on these topologies the catalog implicates the large majority of components
at `CRITICAL`/`HIGH` severity, so an absolute gate would block essentially every build.

The design response is **delta-aware semantics**: compare the pull-request candidate topology against
the target branch's merge-base topology and flag only *newly introduced* findings, letting pre-existing
ones pass unless their severity worsens, with intentional anomalies bypassed through an auditable,
time-bound **waiver register** naming entity, rule and expiry. This mirrors the "Clean as You Code"
discipline familiar from code-scope quality platforms, applied at topology scope: the gate blocks new
architectural debt while the prescriptive engine (§6) proposes verified repayments of existing debt.

**Implementation status.** Delta semantics and the waiver register are *specified but not
implemented*. The shipped gate evaluates severities absolutely, against the candidate topology alone,
with no merge-base comparison and no waiver mechanism. We therefore make no empirical claim about
delta-gating precision or recall in this paper, and record its implementation and evaluation as the
CI/CD item in §11.2. The §9.1 base-rate result is what makes this the binding next step rather than a
refinement: without delta semantics or a severity recalibration, the gate as it stands is not
deployable.

### 7.3 Exit-Code Protocol

The gate signals CI/CD pipeline workers through standardized exit codes. This three-tier protocol *is*
implemented, on absolute rather than delta semantics:

* **Exit Code 0:** no findings above the reporting floor; build passes, deployment permitted.
* **Exit Code 1:** `MEDIUM`-severity findings present; build passes with warnings.
* **Exit Code 2:** `CRITICAL` or `HIGH` severity findings present; the build breaks and deployment is
  blocked.

Under delta semantics each tier would be evaluated over *newly introduced, unwaived* findings rather
than over the absolute finding set. That substitution is the whole of the change required at this
layer; the difficulty is in computing and diffing the merge-base topology, not in the protocol.

---

## 8. Experimental Design

### 8.1 Research Questions

* **RQ1 (Detection efficacy and precision):** Does the anti-pattern catalog correlate with
  ground-truth failure impact, and does it avoid over-flagging well-structured systems?
* **RQ2 (Prescriptive efficacy):** Does the closed-loop engine reduce the System Risk Index across
  heterogeneous scenarios, and are the reductions statistically significant?
* **RQ3 (Operator contributions):** How do the individual refactoring operators contribute to the
  observed improvements across topological regimes?
* **RQ4 (What per-edit verification admits):** How many generated candidates survive independent
  counterfactual verification, in which structural regimes, and does an accepted *set* compose — that
  is, do individually verified edits remain beneficial when applied together?
* **RQ5 (Computational overhead and CI/CD feasibility):** What is the wall-clock execution time of the
  detection gate and of the full prescriptive pipeline as system scale grows — and is either
  compatible with CI/CD budgets?

### 8.2 Benchmark Scenarios

Detection validation (RQ1, RQ5) uses the full eight-scenario suite (S01–S08), including the
deterministic "Tiny Regression" smoke-test fixture, which is retained here because a 12-component
topology is a useful lower bound on both runtime and finding volume.

Prescriptive evaluation (RQ2–RQ4) uses a **six-scenario subset (S01–S06)**, for two separate reasons
that we keep separate. S08 is excluded on relevance: it carries no domain-representative topology and
would contribute no meaningful signal about prescriptive efficacy. **S07 is excluded on measured
cost** — approximately 8.7 hours of serial computation under per-edit verification, itemized in §9.2 —
which is a very different kind of exclusion, since it removes the largest topology from precisely the
result whose scale-invariance is in question. §10.5 records the consequence.

**Table 8.1 — Scenario scale and topology summary.**

| Scenario | Applications | Libraries | Topics | Brokers | Nodes | Structural Edges ($|E|$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| S01 Autonomous Vehicle | 80 | 20 | 40 | 4 | 8 | 797 |
| S02 IoT Smart City | 200 | 10 | 80 | 6 | 30 | 1322 |
| S03 Financial Trading | 60 | 18 | 35 | 5 | 6 | 580 |
| S04 Healthcare | 50 | 12 | 25 | 3 | 8 | 400 |
| S05 Hub-and-Spoke | 70 | 25 | 30 | 2 | 12 | 797 |
| S06 Microservices Mesh | 90 | 30 | 45 | 6 | 15 | 680 |
| S07 Hyper-Scale Enterprise | 300 | 50 | 120 | 10 | 40 | 3245 |
| S08 Tiny Regression | 12 | 4 | 8 | 2 | 3 | — |

These presets are drawn from the scenario library shared with the companion SaG materials [1], which
additionally contains an ATM topology reserved there for external validation of the diagnostic model
against a non-synthetic reference. It is not reused here; replaying SaG-Prescribe on it is future work
(§11.2). All are parameterized synthetic topologies from one generator, which §10.5 identifies as the
evaluation's weakest dimension.

### 8.3 Experimental Protocol

**Detection (RQ1, RQ5).** Each scenario is loaded into an in-memory repository, analyzed, scored, and
passed through the catalog detectors; the resulting findings and the $Q(v)$ ranking are scored against
the oracle. Ground truth is $I_{\text{comp}}(v)$, the `FailureSimulator` composite obtained by
exhaustive single-component removal at the canonical propagation threshold $0.2$ and seed $42$. This
is the same oracle the Validate-stage gates and the prescriptive acceptance criterion run on, so
detection and remediation are scored against one yardstick. It is *not* the `FaultInjector` oracle
$I^*(v)$ used for the learned predictors of the companion work, and the two agree only moderately
(mean Spearman $\rho = 0.4046$), so a figure established here does not transfer to a claim measured
against $I^*$. Results are reported at two layer scopes: the **app** layer (Applications and
Libraries), which is homogeneous and is the primary lens, and the **system** layer (all five node
types), which we report to expose a pooling effect discussed in §9.1.

**Prescription (RQ2–RQ4).** The prescriptive engine runs in-memory over each topology. Verification
uses $\kappa = 1.0$, propagation thresholds $\Theta = \{0.1, 0.2, 0.5\}$ and seeds $S = \{42, 123,
456\}$. Baselines are measured once over the full $(\theta, s)$ grid and shared across candidates, and
each mutated run is paired-differenced against the baseline at *its own* $(\theta, s)$, so the
simulator noise common to both largely cancels. Earlier versions of this work claimed that the
simulator's default configuration made $\sigma_{\text{seed}}$ identically zero and the margin
criterion therefore vacuous; that is not what the sweep measures, and §9.4 reports that the margin
rejects the large majority of candidates.

### 8.4 Metrics

**Detection metrics (RQ1).** Spearman rank correlation $\rho$ between the composite criticality score
$Q(v)$ and simulated impact $I_{\text{comp}}(v)$, reported pooled *and* stratified by node type;
precision, recall and $F_1$ for critical-set classification; Top-5 and Top-10 overlap; and, as
baselines, the same correlations for betweenness centrality and degree centrality alone, so the
composite's value is a measured margin rather than an assertion.

Critical-set membership follows the rule already implemented in the framework's validator: the
box-plot top tier ($\ge Q_3$) when $n \ge 20$, and a top-20% percentile mask below that, applied
identically to the prediction and to the truth. One property of this rule must be stated so its
outputs are not over-read: because it thresholds both sides to near-equal sizes, precision and recall
are equal by construction for the $Q(v)$, betweenness and degree predictors. This is a property of the
rule, not a finding. The *catalog* predictor is deliberately not thresholded this way — its flagged
set is whatever the detectors produced — so its precision and recall are independent quantities.

**Prescription metrics (RQ2–RQ4).** The System Risk Index and $\Delta\mathrm{SRI}$ (below); the
per-edit acceptance rate and the distribution of rejection reasons; and per-operator counts split into
*candidates generated* and *edits admitted*, since the two now differ substantially.

**System Risk Index (SRI, RQ2–RQ4).** The primary prescriptive outcome measure is a composite risk
index over the four RMAV health dimensions:

$$\mathrm{SRI} = 0.25\,(1 - H_R) + 0.25\,(1 - H_M) + 0.25\,(1 - H_A) + 0.25\,(1 - H_V)$$

where each $H_d = 1 - \left(\sum_c \mathrm{score}_d(c)\, w_c\right) / \sum_c w_c$ is the
component-weight-normalized system-level health along dimension $d$. Lower SRI indicates lower
composite structural risk, and we report $\Delta\mathrm{SRI} = \mathrm{SRI}_{\text{baseline}} -
\mathrm{SRI}_{\text{mutated}}$, so positive is improvement. SRI and the per-component impact map
$I(v)$ are read from a single simulation sweep, so the two provably describe the same simulation
rather than two runs that happen to share a configuration.

---

## 9. Results

### 9.1 Detection Efficacy and Precision (RQ1)

RQ1 resolves negatively on both of its halves, and we report it as such. Table 9.1 gives the
per-scenario figures at the app layer; all numbers in this subsection come from
`results/detection_validation.json` and `results/detection_validation_app.json`.

**Table 9.1 — Detection validation, app layer, against $I_{\text{comp}}(v)$.** $\rho$, $F_1$ and
Top-5 score the composite $Q(v)$ ranking; catalog P/R score the named `CRITICAL`/`HIGH` findings.

| Scenario | $n$ | $\rho(Q, I)$ | $F_1$ | Top-5 | Catalog P | Catalog R | Implicated % | Gate (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S01 Autonomous Vehicle | 80 | 0.449 | 0.650 | 0.6 | 0.253 | 1.000 | 98.8 | 1.18 |
| S02 IoT Smart City | 200 | 0.097 | 0.300 | 0.4 | 0.308 | 0.800 | 65.0 | 1.03 |
| S03 Financial Trading | 60 | 0.315 | 0.467 | 0.8 | 0.250 | 1.000 | 100.0 | 1.06 |
| S04 Healthcare | 50 | 0.316 | 0.308 | 0.6 | 0.234 | 0.846 | 94.0 | 0.36 |
| S05 Hub-and-Spoke | 70 | 0.307 | 0.333 | 0.4 | 0.257 | 1.000 | 100.0 | 1.55 |
| S06 Microservices Mesh | 90 | 0.178 | 0.391 | 0.4 | 0.286 | 0.783 | 70.0 | 0.55 |
| S07 Hyper-Scale Enterprise | 300 | 0.355 | 0.600 | 0.4 | 0.245 | 0.933 | 95.3 | 24.59 |
| S08 Tiny Regression | 12 | 0.126 | 0.333 | 0.4 | 0.250 | 1.000 | 100.0 | 0.03 |
| **Mean** | | **0.268** | **0.423** | **0.500** | **0.260** | **0.920** | **90.4** | |

**Efficacy: the correlation is weak, and single-metric baselines are not beaten.** Mean $\rho = 0.268$
across the suite, with per-scenario values from $0.097$ to $0.449$ — positive everywhere, but far from
the $\rho \ge 0.70$ target the framework's own validation gates set. The comparison that matters most
is against the baselines the composite is meant to improve on: betweenness centrality alone reaches
$\rho = 0.295$ and **degree centrality alone reaches $\rho = 0.417$**, both above the four-dimensional
composite. On this evidence the RMAV composite does not buy ranking accuracy over the simplest
structural metric available. This does not make the decomposition worthless — its four dimensions
route a finding to the engineering role equipped to act on it, which a scalar degree count cannot —
but that is an explanatory claim, not an accuracy claim, and §10.1 scopes it accordingly.

**Precision: the catalog does not stay quiet.** The named findings behave as a highly sensitive,
weakly specific screen: mean recall $0.920$ against mean precision $0.260$. The precision figure sits
essentially at the base rate, because the catalog implicates a mean of **90.4% of components** at
`CRITICAL`/`HIGH` severity. The intended precision stress test fails in exactly the place it was
supposed to succeed: the sparse microservices mesh (S06), chosen because a well-structured topology
should produce few findings, has 70% of its components implicated — the *lowest* figure in the suite,
but not a quiet one. A screen that flags nine components in ten is not usable as a blocking gate, and
§7.2 records this as the reason delta semantics are the binding next step rather than a refinement.

Two accounting notes keep this figure honest in both directions. First, most `CRITICAL`/`HIGH`
findings are keyed on an edge (`BOTTLENECK_EDGE`, `BRIDGE_EDGE`) or a member list (`CYCLE`) rather
than on a component, and we count a component as implicated when a finding names it as an endpoint or
member — the reading a practitioner would take. Counting only component-keyed findings would put the
flagged fraction near zero and understate the catalog just as badly. Second, one `CYCLE` finding can
name dozens of components at once, so a single detection contributes disproportionately to the
implicated fraction.

**Scale: the "improves at larger scale" claim does not survive.** Splitting by system size gives mean
$\rho = 0.226$ for the two topologies at or above 150 components against $0.282$ below it. The
direction is, if anything, mildly the opposite of the claim previously made, and at this magnitude the
difference is not interpretable either way.

**Node-type pooling is a trap here, and we report both sides of it.** At the system layer, which pools
all five node types, the pooled correlation is $\rho = -0.085$ — *negative* — while every individual
type is positive: Application $0.306$, Node $0.319$, Broker $0.047$. The pooled figure lies outside the
per-type range entirely, which is the signature of a Simpson's-paradox effect: it is not a summary of
the strata but an artifact of between-type offsets in both score and impact scales. We therefore treat
the homogeneous app layer as the primary lens and report the pooled system-layer number only to
document the effect. Any single system-layer correlation quoted for a heterogeneous pub-sub graph
should be treated the same way.

**The deliberately-encoded anti-pattern is not detected.** S05 exists to encode broker saturation
explicitly — two brokers serving seventy applications — and is therefore the suite's only scenario with
a known-by-construction ground-truth pattern. `BROKER_OVERLOAD` produces zero findings on it, for the
structural reason given in §4.4: its $2\times$-median rule cannot fire on a two-broker population in
which both brokers are equally overloaded. This is the sharpest available evidence that the catalog's
weakness is in threshold specification rather than in the pattern definitions, since the pattern is
correctly *described* and simply cannot trigger on the case it describes.

**SPOF-specific validation is not available on these topologies.** The purpose-built SPOF metric
compares the directed articulation score $\mathrm{AP}_{c,\text{directed}}(v)$ against simulated
availability impact, but $\mathrm{AP}_{c,\text{directed}}$ is zero for every component in seven of the
eight scenarios (one component in the eighth), so the predicted-SPOF set is empty and the metric is
**undefined rather than zero**. Separately, the framework's default true-SPOF threshold of
$\mathrm{IA} > 0.50$ is off this oracle's scale: availability impact peaks at $0.39$ across the entire
suite. We report both facts rather than a derived $F_1$ of $0.0$, which would misrepresent a degenerate
measurement as a detection failure.

### 9.2 Prescriptive Efficacy (RQ2)

**Table 9.2 — Prescriptive results under per-edit verification** ($\kappa = 1.0$, $\Theta = \{0.1,
0.2, 0.5\}$, $S = \{42, 123, 456\}$).

| Scenario | Baseline SRI | Mutated SRI | ΔSRI | Candidates | Accepted | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S01 Autonomous Vehicle | 0.3645 | 0.3645 | +0.0000 | 35 | 0 | 35 |
| S02 IoT Smart City | 0.4206 | 0.3841 | **+0.0365** | 58 | 26 | 32 |
| S03 Financial Trading | 0.3675 | 0.3675 | +0.0000 | 31 | 0 | 31 |
| S04 Healthcare | 0.3809 | 0.3809 | +0.0000 | 19 | 0 | 19 |
| S05 Hub-and-Spoke | 0.3595 | 0.3595 | +0.0000 | 30 | 0 | 30 |
| S06 Microservices Mesh | 0.3612 | 0.3623 | **−0.0011** | 40 | 3 | 37 |
| **Total** | | | | **213** | **29** | **184** |

The honest summary is that the operator set rarely produces a change the simulator can distinguish
from its own noise. **Twenty-nine of 213 candidate edits (13.6%) survive verification**, and five of
the six scenarios end with no admitted edit at all — their SRI is reported unchanged because nothing
was applied, not because a mutation was applied and made no difference. One scenario shows a clear
improvement (S02, $\Delta\mathrm{SRI} = +0.0365$), and one shows a small regression discussed in §9.4.

We do not report a significance test on this table. With four of six scenarios at exactly zero, one
positive and one negative, a signed-rank test over six pairs has neither the sample nor the
distributional structure to say anything; quoting one would be a decoration rather than evidence.

**The Enterprise scenario (S07) is excluded from this table on cost grounds**, and we state the cost
rather than the exclusion alone: 119 candidate edits $\times$ 3 thresholds $\times$ 3 seeds at
approximately 29.1 s per exhaustive sweep over 350 labelled components is roughly 8.7 hours of serial
computation. The consequence to be explicit about is that the 13.6% acceptance rate is established
over topologies of 98–326 components and cannot be claimed scale-invariant. Note also that the
exclusion is a reporting decision: the reproduction script does not skip S07, so re-running it
unattended will attempt the full sweep. Parallelising it is recorded in §11.2.

### 9.3 Operator Contributions (RQ3)

Under verification, the operator-contribution question changes shape: what matters is not how often
each operator *fires* but how often its output *survives*. The two now diverge sharply.

| Operator | Candidates generated | Edits admitted |
| --- | ---: | ---: |
| Logical topic splitting | 213 total across all three operators | **29** |
| Physical anti-affinity reallocation | (see below) | **0** |
| Transport QoS contract hardening | | **0** |

**All 29 admitted edits are topic splits.** Not one anti-affinity reallocation and not one QoS
upgrade cleared the acceptance margin in any scenario, despite reallocation being by far the most
frequently *generated* mutation — the unfiltered engine emitted 409 reallocations on the enterprise
topology and 276 on IoT Smart City. This is the single most informative result in the prescriptive
half of the paper, and it directly contradicts a claim carried by earlier versions of this work, which
attributed IoT Smart City's improvement to 51 QoS upgrades "stabilizing high-loss best-effort links".
Those upgrades were generated; none of them survived independent verification, and the improvement
comes entirely from topic splitting.

The mechanism is visible in the operator semantics. Anti-affinity reallocation moves a co-located
process to a fresh host and duplicates `CONNECTS_TO` links to preserve reachability — which adds
network cascade hops. Under whole-policy application those additions were invisible, absorbed into an
aggregate that also contained genuine improvements; under per-edit verification they are measured
individually and rejected. QoS hardening changes edge weights without changing topology, and on this
cascade model that moves impact by less than seed noise.

An ablation applying each operator class in isolation would let us attribute $\Delta\mathrm{SRI}$ per
operator rather than reading it off admission counts; we record it as future work (§11.2) rather than
presenting a speculative table.

### 9.4 What Per-Edit Verification Admits (RQ4)

Three findings, all of which qualify the previous subsection rather than extending it.

**The filter's yield is low and its rejections are informative.** Of 184 rejected candidates, the
rejection reason in each case names the binding propagation threshold at which the mean impact
reduction failed to exceed $\kappa\sigma_{\text{seed}}$. Rejection is therefore not a silent drop but a
measurement the architect can read.

**Individually verified edits can still interact.** S06 (Microservices Mesh) accepted three edits,
each of which cleared the bar on its own counterfactual graph, yet the combination left the System
Risk Index marginally *worse* ($\Delta\mathrm{SRI} = -0.0011$). Per-edit verification bounds this
failure mode without eliminating it — the comparable figure under the previous unfiltered design was a
$-31.67\%$ component-level regression, two orders of magnitude larger — but it does not establish that
an accepted *set* composes. Verification admits singletons, not subsets. Verifying subsets would close
this at combinatorial cost and is recorded as a limitation (§10.4) and future work (§11.2).

**Where the filter admits edits it admits many, and only in one structural regime.** S02 accepts 26 of
58 candidates, all topic splits, and is the suite's only clear improvement. The honest scope for this
stage is therefore narrower than "topology-level hardening": it is closer to *fan-out decomposition
where a fan-out bottleneck actually exists*. Where no such bottleneck exists, the current operator set
has nothing to offer that survives measurement, and it says so by admitting nothing.

### 9.5 Computational Overhead and CI/CD Feasibility (RQ5)

**Detection gate.** Table 9.1 reports measured wall-clock time for the complete gate path — load,
analyze, RMAV attribution, and all active detectors — from `results/detection_validation_app.json`.
It ranges from 0.03 s on the 12-component fixture to 24.6 s on the 300-application enterprise topology
(0.16 s to 33.0 s at the system layer, which analyses all five node types). This is comfortably inside
a blocking-check budget at every scale evaluated.

The cost breakdown is more useful than the totals, and it is not where one would expect. Summed across
all eight scenarios, the twenty active detectors together account for roughly **0.2 seconds**; the
costliest single detector (`CYCLE`) accounts for 0.06 s. Essentially the entire gate budget is spent
in the analysis stage that computes the structural metrics and RMAV scores, not in pattern detection.
Optimizing the detectors would therefore buy nothing; the analysis stage is the only lever.

**One detector is excluded, and the reason is a defect rather than a design choice.** `DEEP_PIPELINE`
enumerates every simple source-to-sink path in the dependency graph and emits one finding per path,
keyed on the path string. On the 29-component tiny fixture this produces **247,761 findings**; on the
50-application healthcare topology it does not terminate within ten minutes. The twenty remaining
detectors are what the figures above measure. Until this is bounded — by capping enumeration, or by
reporting one finding per source-sink pair rather than per path — the full catalog cannot run as a CI
gate at any of the scales in this study. We record the fix in §11.2.

**Generate–verify loop.** Per-edit verification changed this cost regime by construction: it requires
one exhaustive simulation sweep per $(\text{edit} \times \text{threshold} \times \text{seed})$ triple,
so a scenario with $c$ candidates costs $9c$ sweeps at the configuration used here, against a single
end-state evaluation under the previous unfiltered design. The one directly measured point is the
enterprise scenario's sweep cost of approximately 29.1 s over 350 labelled components, giving the
$\approx 8.7$ h serial estimate of §9.2. We do not carry forward the loop runtimes reported in earlier
versions of this work (4.7 s–649.6 s per scenario), because they were measured under the unfiltered
design and do not describe the pipeline evaluated here. Re-measuring the loop under verification, and
parallelising the independent per-candidate sweeps — which are embarrassingly parallel — is recorded
in §11.2.

The practical reading for CI/CD adoption is a split one: the detection gate is a per-commit-viable
check at every scale measured, while the generate–verify loop is a nightly or on-demand batch job, not
a merge-request-blocking one.

---

## 10. Discussion and Threats to Validity

### 10.1 What Naming and Verifying Buys, and What It Does Not

Two results read together, and they cut in different directions.

**Verification earns its cost.** §9.3 is the clearest evidence in the paper for the verify-before-
recommend discipline. An unverified recommender running these same operators would have emitted 409
anti-affinity reallocations on the enterprise topology and 51 QoS upgrades on IoT Smart City, and — as
earlier versions of this work did — attributed the resulting system-level improvement to them.
Independent per-edit measurement shows that *none* of those mutations survives, and that the entire
measurable gain comes from topic splitting. The difference between a suggestion service and a
quality-evaluation instrument is exactly this measurement, and here it changed the paper's causal
account, not merely its confidence interval.

**Naming, on this evidence, does not yet buy what we claimed for it.** The premise of §4 is that a
named, severity-tiered finding is more actionable than an opaque score. That remains true as a design
argument — an architect can act on "Single Point of Failure at broker B2" in a way they cannot act on
"$Q = 0.83$" — but §9.1 shows the current catalog does not deliver it in practice, because it
implicates roughly nine components in ten. A vocabulary that names almost everything conveys almost
nothing, and the honest reading is that the catalog's *specifications* are sound while its *thresholds*
are not yet calibrated to produce a usable shortlist. §4.2's note on rank normalization producing a
near-constant critical fraction points at the mechanism, and §11.2 makes recalibration the first
detection-side item.

Similarly, the composite $Q(v)$ does not out-rank degree centrality on this suite (§9.1). We therefore
scope the RMAV decomposition's contribution to *attribution* — telling a reader which quality dimension
a component is critical along, and hence which engineering role should own it — rather than to ranking
accuracy. That is a real contribution and a narrower one than previously claimed.

### 10.2 Positioning in CI/CD and Technical-Debt Workflows

Two claim boundaries govern responsible deployment. First, **prescriptive recommendations are
advisory**: the pipeline surfaces verified refactoring blueprints for architect review (§6.2, Stage 7)
but does not auto-apply mutations, since verified-in-simulation does not entail correct-in-production.
Second, **blocking-gate claims are reserved for detection** (§7) — but §9.1 constrains even that. At a
90% implication rate an absolute gate blocks every build, so the gate is deployable only once either
delta semantics (§7.2) or threshold recalibration (§11.2) lands. We state this rather than presenting
the gate as ready.

For teams without CI/CD automation, the catalog functions as a structured architecture-review
checklist: twenty-one specific, testable questions about the system's graph structure, in the spirit of
design review by checklist as practiced in aviation and surgery, rather than an informal "does this
look healthy?" pass. This use is unaffected by the threshold-calibration problem, because a human
reviewer working down a checklist supplies the prioritization that the severity tiers currently do not.

### 10.3 Construct Validity

Both detection and verification are defined relative to the same discrete-event cascade simulator:
§9.1 demonstrates that named patterns correlate — weakly — with *simulated* impact, and §9.2's SRI
change demonstrates that an admitted edit improves resilience *as the simulator models it*, neither
necessarily as a production system would experience it. Three specifics bound this.

First, the framework has two simulation oracles and they agree only moderately: mean Spearman
$\rho = 0.4046$ between the `FailureSimulator` composite $I_{\text{comp}}$ used throughout this paper
and the `FaultInjector` measure $I^*$ used for the learned predictors of the companion work, with the
worst scenario at $\rho = 0.06$. A result established against one is therefore not evidence for a claim
measured against the other, and we have kept every figure in this paper on $I_{\text{comp}}$ for that
reason.

Second, we mitigate simulator dependence by grounding the catalog's patterns and the prescriptive
operators in mechanisms meaningful independent of it — established dependability practice (fan-out
reduction, anti-affinity scheduling, QoS hardening) and graph-theoretic reliability results predating
this work [18, 19]. This is an argument, not a measurement.

Third, the operator-to-pattern linkage of §6.3 relies on substring matching over human-readable
detected-problem names rather than a dedicated pattern-ID field. The construct "this mutation repairs
that anti-pattern" is therefore implemented by a string comparison that silently unbinds if a pattern
is renamed, and it covers only five of the twenty-one patterns directly.

### 10.4 Internal Validity

The prescriptive engine generates candidates by exhaustive rule firing rather than by searching the
policy space, so no optimality claim is made; what it reports is what survives filtering, not what an
optimizing search would find. Verification uses identical fault scenarios and seeds for baseline and
mutated topologies, paired at matching $(\theta, s)$, so comparisons are not confounded by simulation
sampling.

Two limits on the acceptance procedure deserve stating plainly. **Verification admits singletons, not
subsets:** each candidate is measured alone, so an accepted set is not guaranteed to compose, and §9.4
reports a case where it did not. **The margin parameter is asserted, not derived:** $\kappa = 1.0$ is
a stated choice, and the acceptance rate is sensitive to it in a way we have not characterized.

The independence guarantee of §6.4 rules out feature–label feedback, and we scope it honestly: it is
*view* independence, not independence of data source. Both $G_{\text{analysis}}$ and
$G_{\text{structural}}$ are deterministic functions of the same topology under one cascade model, so
what is ruled out is a leakage path, not a shared modelling assumption.

Finally, aggregate statistics over a heterogeneous graph can invert their own strata. §9.1 documents a
concrete instance: the pooled system-layer correlation is negative while every node type is positive.
We therefore treat stratified reporting as mandatory rather than optional wherever a type-level result
is available, and report the pooled figure alongside it as a diagnostic rather than as a headline.

### 10.5 External Validity

All scenarios are parameterized synthetic topologies from a single generator. While the presets mimic
representative domain verticals, they may not capture the runtime complexity of industrial clusters —
dynamic workload shifts, packet-loss bursts, transient hardware faults. This is the weakest dimension
of the evaluation and the highest-value follow-up.

The catalog is scoped to the publish–subscribe communication paradigm: systems combining pub-sub with
request-response patterns (hybrid microservices, mixed REST/event architectures) will require
additional patterns for the request-response side, and `QOS_MISMATCH` is specified for DDS/ROS 2 and
MQTT QoS-weight semantics, requiring adaptation elsewhere.

Two scenario-specific limits also bound generalization. The prescriptive acceptance rate of §9.2 is
established over topologies of 98–326 components and cannot be claimed scale-invariant, since the
largest scenario was excluded on cost. And the detection figures of §9.1 are computed with one of the
twenty-one detectors excluded (§9.5), so they characterize a twenty-detector catalog.

### 10.6 Conclusion Validity

We report no significance test. The prescriptive sample is six scenarios, of which four produced no
admitted edit, and a signed-rank test over that structure would be a decoration rather than evidence
(§9.2). Detection figures are single-run per scenario at the canonical seed and propagation threshold;
§9.1's correlations should be read with the propagation-threshold sensitivity of the underlying model
in mind, which spans roughly $0.2$ Spearman $\rho$ across the plausible range — and which is precisely
why §6.4's acceptance filter requires its margin at every threshold rather than at the default alone.

Earlier versions of this work justified single-run reporting by asserting that the simulator's default
configuration made $\sigma_{\text{seed}}$ identically zero. That assertion does not survive
measurement: the per-edit filter estimates $\sigma_{\text{seed}}$ across seeds at every threshold and
rejects 86% of candidates on that margin.

### 10.7 Engineering Trade-offs

Closed-loop verification costs roughly $|\Theta| \cdot |S|$ exhaustive simulation sweeps per candidate
against a single end-state evaluation for an unverified recommender — a factor of nine at the
configuration used here, and the reason the enterprise scenario was excluded (§9.2). We consider that
cost well spent, on the evidence of §9.3: verification did not merely tighten the prescriptive claim,
it corrected it. The corresponding trade-off on the detection side runs the other way — §9.5 shows
detection is nearly free relative to the analysis that feeds it — which is what makes the split
deployment model of §9.5 (per-commit detection, batch prescription) the sensible one.

---

## 11. Conclusion and Future Work

### 11.1 Conclusions

This paper presented SaG-Prescribe, a graph-based framework that unifies the detection of named,
severity-tiered architectural anti-patterns with closed-loop prescriptive refactoring for distributed
publish–subscribe architectures, and evaluated it honestly enough that its limits are as legible as
its contributions.

The twenty-one-pattern catalog gives practitioners a pub-sub-specific vocabulary analogous to
established object-oriented and microservices smell catalogs — a contribution we believe stands on the
specifications themselves. Its empirical validation, measured here for the first time against a
committed oracle rather than carried forward from prior publication, is weaker than previously
reported: mean Spearman $\rho = 0.268$ between the composite criticality score and simulated cascade
impact, below both betweenness ($0.295$) and degree centrality ($0.417$) alone, with the catalog
implicating roughly 90% of components at `CRITICAL`/`HIGH` severity. The specifications are sound; the
thresholds are not yet calibrated to produce a usable shortlist, and we identify that as the binding
detection-side problem rather than presenting the gate as deployable.

The prescriptive contribution is where verification changed the result rather than confirming it.
Compiling named findings into three graph-mutation operators and subjecting every candidate edit to
independent counterfactual verification — $\Delta I > \kappa\sigma_{\text{seed}}$ at every propagation
threshold — admits 29 of 213 candidates (13.6%), all of them topic splits. No anti-affinity
reallocation and no QoS upgrade survives measurement anywhere in the suite, which contradicts the
causal account earlier versions of this work gave for its own best result. Five of six scenarios admit
nothing at all; one shows a clear improvement ($\Delta\mathrm{SRI} = +0.0365$); and one shows that
individually verified edits can still interact badly in combination. The defensible scope for this
stage is narrower than "topology-level hardening" — it is fan-out decomposition where a fan-out
bottleneck actually exists — and we think a negative result of this shape, obtained by measuring what
was previously assumed, is worth more to the field than the aggregate it replaces.

The CI/CD contribution is the least disturbed: detection runs in 0.03–24.6 s across the full scale
range, with the detectors themselves accounting for under a second in total and the analysis stage
carrying essentially the entire budget. The delta-aware gating semantics are specified and unimplemented,
and §9.1 makes implementing them a prerequisite for deployment rather than an enhancement.

### 11.2 Future Work

Ordered by how much each would change the paper's claims.

1. **Recalibrate detection thresholds.** A catalog implicating 90% of components cannot prioritize
   (§9.1). The mechanism is at least partly identified — rank normalization makes the box-plot
   classifier's critical fraction near-constant across topologies (§4.2) — so the first experiment is a
   normalization and fence sweep measured against the same oracle used here.
2. **Bound the `DEEP_PIPELINE` detector.** It enumerates every simple source-to-sink path and does not
   terminate at realistic scale (§9.5), so the full twenty-one-pattern catalog cannot currently run as
   a gate. Reporting one finding per source–sink pair, or capping enumeration, restores it.
3. **Implement and evaluate delta-aware gating.** Merge-base topology diffing, the waiver register,
   and delta-relative exit codes (§7.2), followed by a fault-injection study measuring the gate's
   precision and recall on newly introduced regressions.
4. **Verify subsets, not only singletons.** §9.4's composition failure is the direct motivation.
   Greedy forward selection over the accepted set is the cheapest approach that would catch it.
5. **Derive $\kappa$ rather than assert it.** Estimate the simulator's noise scale across a broader
   sweep and set the acceptance margin from it, and characterize acceptance-rate sensitivity to
   $\kappa$ (§10.4).
6. **Parallelise verification and close the enterprise gap.** Per-candidate sweeps are independent, so
   the $\approx 8.7$ h serial cost that forced the enterprise exclusion (§9.2) is parallelisable
   almost linearly. This would also test whether the 13.6% acceptance rate holds at scale.
7. **Expand and ablate the operator set.** Two of three operators currently admit nothing; either they
   need triggers matched to what the cascade model can express, or the model needs to express what
   they change. An ablation attributing $\Delta\mathrm{SRI}$ per operator would separate these.
8. **Extend the catalog.** Hybrid REST/event architectures, and middleware whose QoS semantics differ
   from the DDS/ROS 2/MQTT weight formula `QOS_MISMATCH` assumes (§10.5).
9. **Real-system replication.** Applying the engine to the ATM topology of [1] and to harvested
   industrial configurations, closing the external-validity gap of §10.5.
10. **LLM-assisted pull-request generation:** linking code assistants to generate pull requests
    implementing verified refactoring blueprints. This is the paper's only LLM-adjacent item and is not
    a contribution claimed here.

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
  detection and prescriptive pipeline implementations will be made available at [URL pending]. Every
  figure in §9 is reproducible from committed artifacts:

  ```
  PYTHONPATH=. python reproduce/detection_validation.py                    # §9.1, §9.5 (system layer)
  PYTHONPATH=. python reproduce/detection_validation.py --layer app \
      --output results/detection_validation_app.json                       # §9.1, Table 9.1
  PYTHONPATH=. python reproduce/run_prescribe_all.py --kappa 1.0           # §9.2–§9.4
  ```

  Note that `run_prescribe_all.py` does not itself skip the enterprise scenario; the exclusion
  described in §9.2 is a reporting decision, and re-running the script unattended will attempt the
  full sweep at the cost stated there.
- **Ethics approval:** Not applicable.
