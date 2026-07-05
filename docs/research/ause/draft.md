# SaG-Prescribe: Closed-Loop, DevOps-Integrated Prescriptive Refactoring of Distributed Publish–Subscribe Middleware Architectures

**Target venue:** Automated Software Engineering (AuSE), Springer — Special Issue on *Intelligent
Techniques for Automated Code Review and Software Quality Evaluation*.
**SI topic mapping:** *Intelligent techniques for CI/CD, DevOps, software evolution, technical debt
analysis, and refactoring recommendation* (primary bullet satisfied). No AutoML/NAS/LLM contribution
is claimed; the DevOps-integrated quality gate (§6) and the closed-loop refactoring-recommendation
pipeline (§5) are what carry this submission into scope.

> **Provenance note (not part of the manuscript).** This draft merges three prior independent drafts
> that accumulated in this directory — *"Automated Prescriptive Refactoring of Distributed Middleware
> Architectures"*, *"Automated, Simulation-Verified Refactoring Recommendation"*, and *"Closed-Loop
> Prescriptive Optimization of Publish–Subscribe Architectures over Heterogeneous Graph Models"* — all
> of which described the same SaG-Prescribe system with overlapping but non-identical framing,
> reference lists, and section coverage. Per the pre-submission action items recorded in this
> directory's outline file, they have been consolidated into this single canonical draft; the source
> files have been removed to eliminate the risk of an overlapping-submission violation. Experimental
> values are taken from the most recently re-measured source ("Closed-Loop Prescriptive
> Optimization...", re-measured 2026-07-02 against `reproduce/run_prescribe_all.py` and
> `saag/prescription/`). The metric name is standardized to **System Risk Index (SRI)**, matching
> `docs/validation.md` and `docs/prescription.md` (one source draft used "System Resilience Index";
> that naming is not used here). One related-work subsection (§4.2) retains `[REF: …]` citation-slot
> placeholders inherited from a source draft — these are explicitly *not* invented citations and must
> be populated from a real bibliography before submission.

---

## Abstract

Distributed publish–subscribe middleware (ROS 2, DDS, MQTT) decouples producers and consumers, but
the resulting indirect dependency structure obscures how component failures cascade, and this
architectural technical debt accumulates invisibly to code-level static analysis (SCA) tools. Existing
structural diagnostic frameworks can rank components by criticality, yet they operate *open-loop*:
they tell architects which components are fragile without producing verified guidance on how to
restructure the topology. We present **SaG-Prescribe**, a closed-loop, DevOps-integrated prescriptive
refactoring system that extends the Software-as-a-Graph (SaG) diagnostic framework. SaG-Prescribe
augments topological diagnostics with a code-quality-penalty signal derived from static-analysis
metrics (LOC, cyclomatic complexity, LCOM, technical-debt ratio), and compiles the resulting
criticality diagnosis — components flagged `CRITICAL` or `HIGH` by adaptive box-plot fences — into a
transformation policy $\Delta(G)$ composed of three graph mutation operators: logical topic splitting,
physical anti-affinity reallocation, and transport QoS contract hardening. Each candidate topology
$G' = \Delta(G)$ is re-evaluated by the same discrete-event cascade simulator that produced the
diagnosis, so every accepted prescription is verified — not merely recommended — before it reaches an
architect or a pull request. The pipeline is further operationalized as a delta-aware CI/CD quality
gate (`detect_antipatterns.py`) that blocks only newly introduced structural regressions relative to a
Git merge base, within sub-minute execution bounds.

Across seven parameterized publish–subscribe scenarios spanning autonomous vehicles, IoT, finance,
healthcare, and hyper-scale enterprise topologies, the generated prescriptions reduce the System Risk
Index (SRI, lower is better) in every scenario, with relative risk reductions between 1.4% and 15.9%
(Wilcoxon signed-rank across the seven baseline/mutated pairs: $W=0.0$, $p=0.0156$, significant at
$\alpha=0.05$). We report this result alongside a less flattering but more honest one: component-level
failure-impact reductions average only +4.61% and regress in two of seven scenarios (−31.67% and
−25.36%), because the current policy is applied greedily and unconditionally, occasionally introducing
new cascade hops that offset its own gains — a finding that directly motivates a per-edit acceptance
filter as the highest-priority extension. The full generate–verify loop completes in under 65 seconds
for six of the seven scenarios (4.7 s–64.3 s) and in approximately 10.8 minutes for the 300-process
enterprise topology, while the CI/CD gate script alone completes in under 2 seconds to ~40 seconds
across the same scale range — making closed-loop prescriptive refactoring practical as a
merge-request-gated or nightly pre-deployment CI/CD stage.

**Keywords:** refactoring recommendation; publish–subscribe middleware; CI/CD quality gates; DevOps;
architectural technical debt; failure cascade simulation; graph mutation; search-based software
engineering

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

### 1.2 The Architecture-Code Gap and the Open-Loop Refactoring-Recommendation Gap

Automated quality assurance has historically operated at the source-code level, through static code
analysis (SCA) platforms such as SonarQube. This produces an **Architecture-Code Gap**: a system can
have clean source code within every individual module yet remain highly fragile at the topology level
— single points of failure (SPOFs), co-located deployment bottlenecks, or mismatched communication
attributes are invisible to file-scoped analysis. Shifting structural verification "left" into the
CI/CD pipeline requires a paradigm shift from Static Code Analysis to *Static System Analysis*.

In our companion paper [1], we introduced **Software-as-a-Graph (SaG)**, a static system analysis
framework that models pub-sub topologies as native heterogeneous graphs and produces diagnostic
criticality rankings $Q(v)$ and failure-impact predictions $I(v)$. SaG closes the Architecture-Code Gap
for diagnosis, but it behaves as an *open-loop diagnostic engine*: it flags architectural
vulnerabilities without synthesizing concrete, verified guidance on how to resolve them. This
limitation is not specific to SaG — refactoring-recommendation research more broadly, from code-smell
detectors to Search-Based Software Engineering (SBSE), predominantly proposes changes without verifying
their effect on the quality attribute of interest before the change enters the development pipeline.
For architectural refactorings of distributed topologies, where the quality attribute is resistance to
cascading failure, this is particularly hazardous: an edit that looks beneficial locally can degrade
global resilience. We refer to the combination of these two limitations as the **open-loop
refactoring-recommendation gap**: architects know *which* components are fragile but lack automated
recommendations on *how* to restructure the topology whose effect on cascading risk has been *verified
before* the recommendation is surfaced.

### 1.3 Proposed Solution: SaG-Prescribe

To close this gap, we present **SaG-Prescribe**, a closed-loop, DevOps-integrated prescriptive
refactoring system extending [1]. SaG-Prescribe consumes SaG's diagnostic output — components flagged
`CRITICAL` or `HIGH` via adaptive box-plot fences, itself informed by a **Code Quality Penalty (CQP)**
computed from static-analysis metrics — and feeds them to a rule-based prescriptive engine that
generates targeted architectural mutations along three vectors: (1) logical topic splitting, isolating
high-fan-out publish channels; (2) physical anti-affinity reallocation, isolating co-located SPOFs; and
(3) transport QoS contract hardening, upgrading volatile/best-effort channels to reliable,
transient-local settings. Crucially, SaG-Prescribe implements **closed-loop simulation verification**:
each mutated candidate topology $G'$ is programmatically re-simulated with SaG's discrete-event cascade
simulator under identical fault conditions, and a policy is reported together with its verified change
in the System Risk Index (SRI) rather than as an unverified suggestion. Because the recommend-verify
loop runs against an in-memory repository with no database dependency, the same underlying engine is
further operationalized as a **delta-aware CI/CD quality gate** that blocks only structural regressions
newly introduced relative to a Git merge base, leaving pre-existing, risk-accepted debt untouched
unless its severity worsens.

### 1.4 Contributions

1. **A code-quality-augmented, closed-loop prescriptive refactoring pipeline** that translates
   topological criticality diagnostics — informed by a Code Quality Penalty bridging SCA metrics into
   the architecture-level risk model — into counterfactual graph mutations, and verifies each candidate
   against the same discrete-event cascade simulator that produced the diagnosis, before it is
   surfaced as a recommendation.
2. **Three publish–subscribe refactoring operators**, formally defined as typed graph mutations
   targeting logical topic congestion, physical hosting SPOFs, and transport QoS fragility (§5.3).
3. **A DevOps-integrated, delta-aware CI/CD quality gate** (`detect_antipatterns.py`) that compares
   candidate and merge-base topologies to block only newly introduced structural regressions, with a
   three-tier exit-code protocol and an auditable waiver register for accepted legacy risk (§6).
4. **A multi-scenario empirical evaluation** across seven realistic pub-sub topologies showing
   consistent, statistically significant SRI reductions of 1.4%–15.9% (Wilcoxon $W=0.0$, $p=0.0156$),
   an operator-level contribution analysis, a **component-level audit that reports mixed results
   honestly** rather than only the flattering system-level aggregate, and a runtime audit confirming
   compatibility with CI/CD budgets at every evaluated scale (§7–§8).

The remainder of the paper is organized as follows. Section 2 surveys related work. Section 3
formalizes the graph model and the code-quality bridge. Section 4 defines the closed-loop optimization
objective. Section 5 presents the prescriptive pipeline and its operators. Section 6 describes the
DevOps/CI-CD integration. Sections 7 and 8 present the experimental design and results. Section 9
discusses implications and threats to validity, and Section 10 concludes.

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
on "Architecture-as-Code" descriptors to proactively evaluate and optimize topologies before
deployment, treating the topology as an open parameter rather than a fixed input.

### 2.2 Refactoring Recommendation and Architectural Technical Debt

Automated refactoring recommendation has been studied extensively at code level: smell-driven
recommenders detect structural anomalies (god classes, feature envy) and propose remediation
transformations [REF: code-smell refactoring recommenders]; learning-based recommenders mine
refactoring histories to predict refactoring opportunities [REF: ML-based refactoring prediction]; and
recent approaches employ large language models for refactoring suggestion and explanation [REF:
LLM-based refactoring]. At the architectural level, technical-debt research quantifies the cost of
structural decay and proposes prioritized remediation plans [REF: architectural technical debt
management].

SaG-Prescribe differs from this body of work along two axes. First, its **scope** is the deployed
system topology — applications, brokers, topics, hosts, and their QoS contracts — rather than source
code within a module boundary; the anomalies it remediates (SPOF hosts, congested topic hubs, fragile
transport contracts) have no file-level analog and are invisible to code-scope recommenders. Second,
its **verification model** is closed-loop: whereas code-level recommenders typically validate
suggestions against static quality metrics or historical acceptance data, SaG-Prescribe re-simulates
every candidate topology against a cascade failure model and surfaces only recommendations with
verified risk improvements. In technical-debt terms, the framework identifies architectural debt items
(anti-patterns with quantified risk via the Code Quality Penalty and RMAV attribution, §3), proposes
repayments (mutation operators, §5.3), and verifies the repayment's effect before recommending it — a
verify-before-recommend discipline that, to our knowledge, has not been applied to pub-sub topology
refactoring.

> *[Citation-slot note, not part of the manuscript: the three `[REF: …]` markers above must be
> populated from a real bibliography — e.g. code-smell/refactoring-recommendation surveys,
> learning-based refactoring-prediction studies, and LLM-assisted refactoring work — before
> submission. No references have been invented to fill these slots.]*

### 2.3 Search-Based Software Engineering and Architecture Optimization

Search-Based Software Engineering (SBSE) applies heuristic search to discover architectural refactoring
blueprints [2], and the architecture-optimization sub-field surveyed by Aleti et al. [3] specifically
targets quality-attribute-driven structural redesign. However, classical search-based methods often
operate open-loop, reporting recommendations without verifying their operational efficacy against a
cascade model. SaG-Prescribe combines the multi-dimensional diagnostics of SaG [1] with closed-loop
simulation verification, ensuring that every recommended edit is evaluated for its effect on the System
Risk Index before acceptance.

### 2.4 Diagnostic Foundation (SaG)

We rely on the heterogeneous graph representation, multi-dimensional quality attribution (RMAV), and
discrete-event failure simulator of Software-as-a-Graph [1] as our diagnostic baseline. SaG-Prescribe
builds directly on SaG's hexagonal ports, extending the domain service to close the loop between
diagnostic ranking and prescriptive mutation. The full mathematical treatment of the diagnostic stages
— graph schema, projection rules, RMAV attribution, and the learned failure-impact predictor — is given
in [1] and is not repeated here; §3 summarizes only what the prescriptive engine consumes.

### 2.5 Structural Criticality Analysis

Graph-theoretic approaches offer constructs such as betweenness centrality [9], PageRank, closeness,
and articulation-point tests to pinpoint critical bridges; recent work applies centrality measures
directly to microservice dependency graphs to detect architectural anti-patterns [10], and complex-network
analyses of software call graphs report the same small-world, hub-dominated topologies that motivate
criticality analysis in the first place [11]. Because classical centrality metrics assume uniform edge
semantics, they degrade on pub-sub layers, where decoupled endpoints are separated by high-fan-out
topics, brokers, and distinct QoS policies. [1] quantifies this gap and motivates the typed multigraph
model that SaG-Prescribe mutates. These approaches identify fragility but, like the diagnostic layer of
SaG, stop short of prescribing and verifying remediation — the step this paper automates.

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
Components in the top two tiers, together with structural anti-pattern flags from the analyzer (e.g.
SPOF hosts, congested topic hubs), are the remediation candidates consumed by the prescriptive engine
(§5).

---

## 4. Closed-Loop Optimization Objective

The prescriptive task is to compute a transformation policy $\Delta$ producing a mutated topology
$G' = \Delta(G)$ that minimizes the aggregate failure-impact profile across system vertices, subject to
a modification budget:

$$\min_{\Delta} \sum_{v \in V} I^*_{\Delta(G)}(v) \quad \text{subject to} \quad \mathrm{Cost}(\Delta) \le \mathcal{B}$$

where $I^*(v)$ denotes the simulated failure impact of component $v$. In the present implementation the
modification budget is unconstrained ($\mathcal{B} = \infty$): the engine emits every mutation whose
triggering rule fires over the `CRITICAL`/`HIGH` candidate set, and the aggregate objective is tracked
through the System Risk Index (SRI, §7.4) computed by the verification stage. A policy is **accepted**
if and only if its verified improvement satisfies $\Delta\mathrm{SRI} > 0$ — the whole-policy acceptance
gate actually implemented in `PrescribeService` (`saag/prescription/service.py`) and documented
canonically in `docs/prescription.md`. A stricter per-edit margin criterion, $\Delta A > \kappa\,
\sigma_{\text{seed}}$, is discussed as future work (§10.2) but is not implemented; under the
deterministic simulator configuration used throughout this evaluation, $\sigma_{\text{seed}} = 0$
(empirically verified across seeds 42–46, §7.3), so the two criteria coincide for the present results.
Budget-constrained policy search — selecting the best subset of mutations under an explicit cost model
rather than firing every triggered rule — is likewise deferred to future work (§10.2).

---

## 5. The SaG-Prescribe Prescriptive Pipeline

### 5.1 Hexagonal Core Abstraction

The system uses a decoupled hexagonal (ports-and-adapters) design separating domain orchestration from
persistence and communication infrastructure. Persistence services implement the `IGraphRepository`
port: production deployments run the Bolt-driven `Neo4jRepository`, while the verification loop and
test suites use an isolated, thread-safe `MemoryRepository` requiring no database instance. This
substitution is what makes repeated counterfactual re-simulation cheap enough for CI/CD integration
(§8.4).

### 5.2 Pipeline Stages

SaG-Prescribe extends the diagnostic pipeline of [1] with a generate–verify loop:

* **Stages 1–5: Diagnostic foundation [1].** Ingest JSON/YAML topology descriptions, compute
  multi-layered topological metrics, attribute component criticality (§3.4), model failure cascades
  with a discrete-event simulator, and validate predictive alignment against simulation ground truth.
  See [1] for the full treatment.
* **Stage 6: Prescriptive recommendation generation (this paper).** The engine consumes components
  categorized `CRITICAL` or `HIGH` and compiles a policy $\Delta(G)$ from the three operators of §5.3.
  The compiled policy is applied to an exported copy of the topology, and the mutated model $G'$ is
  passed back into Stage 4 under identical fault scenarios and seeds to re-evaluate system risk (§5.4).
* **Stage 7: Review interface.** Baseline metrics, detected flaws, prescribed modifications, and
  verification deltas are serialized for the Next.js dashboard (**SMART**), which renders side-by-side
  interactive topologies so architects can review and approve modifications before code commitment.
  Recommendations remain advisory: the human architect is the final authority (§9.2).

### 5.3 Refactoring Operators

**Operator 1 — Logical topic splitting.** A central topic hub connected to multiple publishers and
subscribers is a logical bottleneck and a high-risk failure propagator. For a flagged Topic $t$ with
publisher set $P(t) = \{a : (a, t) \in \text{PUBLISHES\_TO}\}$ and $|P(t)| > 1$, the operator replaces
$t$ with dedicated sub-topics $\{t_a : a \in P(t)\}$, rewiring each publisher to its own sub-topic and
re-attaching subscriber edges to the resulting set. This confines each data feed to its target
subscribers, bounding the structural blast radius of the original high-fan-out hub, and duplicates
broker routing links accordingly.

**Operator 2 — Physical anti-affinity reallocation.** Multiple processes co-located on a single
physical host flagged as an SPOF or critical fail simultaneously if the host fails. For a physical Node
$n$ hosting multiple flagged components, the operator emits reallocation constraints
$(c, n_{\text{from}}, n_{\text{to}})$ moving each co-located component $c$ beyond the first to an
isolating host $n_{\text{to}}$, rewriting the corresponding `RUNS_ON` edge and duplicating
`CONNECTS_TO` links to preserve network reachability. The emitted constraints correspond directly to
container-orchestration anti-affinity scheduling rules.

**Operator 3 — Transport QoS contract hardening.** Critical channels using volatile transport
configurations (`BEST_EFFORT` reliability, `VOLATILE` durability) are fragile under loss. For any topic
that is `CRITICAL`/`HIGH`, or that connects to a critical component with such a configuration, the
operator upgrades the contract to `RELIABLE` reliability and `TRANSIENT` durability (raising transport
priority where applicable), hardening the channel against message loss during cascades.

### 5.4 Closed-Loop Verification

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
guarantee**: all code metrics and RMAV calculations operate strictly on $G_{\text{analysis}}$ (the
derived projection layers), whereas ground-truth labels and SRI evaluations are derived separately from
raw $G_{\text{structural}}$ simulation waves. No simulation parameters ever feed back into diagnostic
metrics or candidate generation.

---

## 6. DevOps Integration and Delta-Aware CI/CD Gating

### 6.1 Automated Code Review Architecture

To continuously govern structural quality during rapid code evolution, the same underlying engine is
operationalized as a blocking check script (`detect_antipatterns.py`) in continuous integration and
delivery (CI/CD) pipelines. Whenever an engineer alters system structures or configures new messaging
topology, the gate parses the "Architecture-as-Code" descriptors and populates an in-memory graph view
— reusing the `MemoryRepository` substrate that makes the prescriptive loop itself CI/CD-viable (§5.1).

### 6.2 Delta-Aware Regression Semantics

Absolute quality gates that fail builds on any critical structural anti-pattern are unsustainable in
industrial software development, since real architectures frequently contain intentional, risk-accepted
debt (e.g. legacy unreplicated components). To resolve this, the gate uses **delta-aware semantics**:
it compares the pull-request candidate topology against the target branch's merge-base topology, and
isolates and flags only *newly introduced* structural regressions. Pre-existing risks pass unless their
severity worsens, and intentional anomalies can be bypassed via an auditable, time-bound **waiver
register** — mirroring the "Clean as You Code" discipline familiar from code-scope quality platforms,
applied here at topology scope: the gate blocks new architectural debt, while the prescriptive engine
(§5) proposes verified repayments of existing debt.

### 6.3 Exit-Code Protocol

The quality gate enforces automated code review by terminating with standardized exit codes that
command CI/CD pipeline workers:

* **Exit Code 0:** no new structural anomalies detected; build passes, deployment permitted.
* **Exit Code 1:** new minor architectural smells or QoS warnings introduced; build passes with
  warnings compiled into the developer's code-review dashboard.
* **Exit Code 2:** new, unwaived `CRITICAL` or `HIGH` severity anomalies (e.g. newly introduced
  un-replicated SPOFs or routing loops) discovered; **the build breaks and deployment is blocked**.

Because the recommend-verify loop of §5 runs within CI time budgets (§8.4), prescriptive repayment
proposals can be regenerated on every merge request, keeping the architectural-debt register
synchronized with the evolving topology rather than only gating new regressions.

---

## 7. Experimental Design

### 7.1 Research Questions

* **RQ1 (Prescriptive efficacy):** Does the closed-loop engine reduce the System Risk Index across
  heterogeneous scenarios, and are the reductions statistically significant?
* **RQ2 (Operator contributions):** How do the individual refactoring operators contribute to the
  observed improvements across topological regimes?
* **RQ3 (Component-level remediation efficacy):** Do system-level SRI improvements translate uniformly
  into component-level failure-impact reductions, or can the greedy, unconditional policy application
  backfire on individual components?
* **RQ4 (Computational overhead and CI/CD feasibility):** What is the wall-clock execution time of the
  closed-loop pipeline, and of the standalone CI/CD gate, as system scale grows — and is either
  compatible with CI/CD budgets?

### 7.2 Benchmark Scenarios

The evaluation suite comprises seven parameterized publish-subscribe topologies spanning realistic
domain verticals and scale presets:

1. **S01 Autonomous Vehicle:** medium-scale ROS 2 network, streaming sensor topologies, reliable and
   transient-local QoS profiles.
2. **S02 IoT Smart City:** large-scale network modeling high-loss, volatile, best-effort endpoints.
3. **S03 Financial Trading:** high-density network with time-critical message loops and strict
   persistent priority settings.
4. **S04 Healthcare Integration:** dense network with centralized real-time patient-monitoring fan-outs
   and long durability horizons.
5. **S05 Hub-and-Spoke:** topology built around a deliberate anti-pattern bottleneck — a centralized
   broker pair — for single-point-of-failure tracking.
6. **S06 Microservices Mesh:** sparse, cloud-native cluster with low coupling, auditing prescription
   behavior at the low-risk boundary.
7. **S07 Hyper-Scale Enterprise:** 300 distinct execution processes, probing the runtime performance
   boundary of the framework.

**Table 7.1 — Scenario scale and topology summary.**

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
evaluation targets the closed-loop prescriptive pipeline rather than diagnostic external validity.
Replaying SaG-Prescribe on the ATM topology is future work (§10.2).

### 7.3 Experimental Protocol

For each scenario, the prescriptive engine runs in-memory over the topology, measuring the SRI before
and after mutation together with the counts of the three operators. The discrete-event cascade
simulator runs in its default deterministic configuration (cascade propagation threshold 0.2, cascade
probability 1.0, non-Poisson event arrivals). We empirically verified this determinism: re-running the
full generate–verify loop under five candidate seeds (42–46) produced byte-identical SRI values in
every scenario ($\sigma_{\text{seed}} = 0$), so a single run per scenario is reported and paired
comparisons are not confounded by simulation stochasticity under the current configuration. For RQ1, we
test the baseline-vs-mutated SRI pairs across the seven scenarios with a Wilcoxon signed-rank test.

### 7.4 Metrics

**System Risk Index (SRI).** The primary outcome measure is a composite risk index over the four RMAV
health dimensions:

$$\mathrm{SRI} = 0.25\,(1 - H_R) + 0.25\,(1 - H_M) + 0.25\,(1 - H_A) + 0.25\,(1 - H_V)$$

where $H_R, H_M, H_A, H_V$ are the system-level Reliability, Maintainability, Availability, and
Vulnerability health scores. Lower SRI indicates lower composite structural risk. We report
$\Delta\mathrm{SRI} = \mathrm{SRI}_{\text{baseline}} - \mathrm{SRI}_{\text{mutated}}$ (positive =
improvement), per-operator recommendation counts, and — separately — component-level failure-impact
deltas to answer RQ3.

---

## 8. Results

### 8.1 Prescriptive Efficacy (RQ1)

**Table 8.1 — Prescriptive optimization results across scenarios.**

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
by 51 QoS upgrades stabilizing high-loss best-effort links combined with anti-affinity constraints that
partition key microservices. The smallest improvement occurs in **S04 (Healthcare, 1.4%)**: its
centralized monitoring fan-outs are intentional and already run under long durability horizons, leaving
less structural headroom for the current operator set. In **S05 (Hub-and-Spoke)**, topic splitting
mitigates the deliberate broker-pair bottleneck (0.3595 → 0.3527).

### 8.2 Operator Contributions (RQ2)

Operator activity tracks each scenario's topological regime:

* **Logical topic splits** dominate where publisher fan-out concentrates: S07 receives 119 splits,
  alleviating congestion in heavily shared publisher–subscriber hubs.
* **Physical reallocations** are the most frequently emitted mutation overall (409 in S07, 276 in S02),
  reflecting how often co-location SPOFs arise in dense deployments; anti-affinity constraints resolve
  them without touching logical structure.
* **QoS upgrades** activate almost exclusively in high-loss profiles — 51 in S02 — where hardening
  volatile/best-effort channels yields the single largest SRI gain observed. Scenarios already running
  reliable contracts (S01, S05, S06, S07) receive none, confirming the rule does not fire spuriously.

An ablation applying each operator class in isolation would let us attribute $\Delta$SRI per operator
rather than inferring contribution from counts; we flag this as future work (§10.2) rather than
including a speculative table here.

### 8.3 Remediation Efficacy at Component Boundaries (RQ3)

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
paper's own highest-priority future-work item: a strict per-edit acceptance filter (§10.2).

### 8.4 Computational Overhead and CI/CD Feasibility (RQ4)

**Table 8.2 — Measured wall-clock time of the full analyze–prescribe–verify loop, single run per scenario.**

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

The standalone CI/CD gate script (`detect_antipatterns.py`, which performs diagnosis only, not the full
generate–verify loop) is considerably faster, since blocking gates must survive as sub-minute checks
without frustrating engineering teams:

* **Tiny / small scales ($\le 25$ vertices):** $< 2$ seconds.
* **Medium scale (~50 components, S04-like):** $\approx 5$ seconds.
* **Large scale (80–100 components):** $\approx 12$ seconds.
* **Xlarge scale (150–300 components, S07-like):** $\approx 40$ seconds.

Taken together, these figures are compatible with pre-deployment CI/CD gating at every evaluated scale:
the gate itself stays under a minute even at hyper-scale, and the heavier generate–verify loop — run
less frequently, e.g. nightly or on-demand per merge request rather than on every commit — fits
comfortably within nightly batch windows even at 300 processes.

---

## 9. Discussion and Threats to Validity

### 9.1 What Closed-Loop Verification Buys

The central result is not that rule-based operators can improve a risk metric — well-chosen
remediations of known anti-patterns are expected to help — but that each recommendation is *verified*
against a cascade model before it reaches the architect. This converts refactoring recommendation from
a suggestion service into a quality-evaluation instrument: the recommendation arrives with quantified,
simulator-confirmed evidence of its effect ($\Delta$SRI per policy), and §8.3 shows why this discipline
matters in practice — an unverified recommender would have reported S05 and S07 as unambiguous wins at
the system level while silently degrading a quarter to a third of their individual components.

### 9.2 Positioning in CI/CD and Technical-Debt Workflows

Two claim boundaries govern responsible deployment of the framework in CI/CD pipelines. First,
**prescriptive recommendations are advisory**: the pipeline surfaces verified refactoring blueprints for
architect review (§5.2, Stage 7), but does not auto-apply mutations, since verified-in-simulation does
not entail correct-in-production. Second, **blocking-gate claims are reserved for structural regression
detection** (§6) — new, unwaived anti-patterns introduced relative to the merge base — while composite
criticality rankings and prescriptive recommendations inform but do not block. The gate blocks new
architectural debt; the recommender proposes verified repayments of existing debt.

### 9.3 Construct Validity

Our optimization target and verification oracle are both defined by the discrete-event cascade
simulator: SRI reductions demonstrate that the prescriptions improve resilience *as the simulator
models it*, not necessarily as a production system would experience it. Because the same simulator
produces both the diagnosis and the verification, the closed loop is internally consistent by
construction; the transfer of verified gains to operational deployments is an external question (§9.5).
We mitigate the risk of optimizing simulator artifacts by using operators grounded in established
dependability practice (fan-out reduction, anti-affinity scheduling, QoS hardening) whose mechanisms
are meaningful independent of the simulator.

### 9.4 Internal Validity

The prescriptive engine is rule-based and greedy: it emits every triggered mutation rather than
searching the policy space, so reported system-level gains are a lower bound on what an optimizing
search could achieve, and no optimality claim is made. Verification uses identical fault scenarios and
seeds for baseline and mutated topologies, so paired comparisons are not confounded by scenario
sampling. The independence guarantee of §5.4 rules out circular leakage between diagnostics and
ground-truth labels. Per-type label degeneracy (passive shared libraries pooled with active
applications) can mask signals in aggregate statistics; as in [1], we treat stratified reporting as
mandatory wherever type-level results are given, and §8.3's component-level breakdown is itself an
instance of this discipline applied to the prescriptive results.

### 9.5 External Validity

All seven scenarios are parameterized synthetic topologies. While the presets mimic representative
domain verticals, they may not capture the runtime complexity of industrial clusters — dynamic workload
shifts, packet-loss bursts, transient hardware faults. Validating prescriptions against a real system
(e.g. replaying the engine on the ATM topology reserved for external validation in [1]) is the most
direct extension.

### 9.6 Conclusion Validity

The scenario sample is small ($n=7$); the Wilcoxon result reported in §8.1 ($W=0.0$, $p=0.0156$) is the
smallest attainable two-sided $p$-value at this sample size, since all seven deltas share the same sign
— it should be read as consistent directional evidence rather than as a claim of large effect size.
Results are single-run per scenario under a verified-deterministic simulator configuration
($\sigma_{\text{seed}}=0$, §7.3); this determinism is a property of the current cascade-probability-1.0
configuration and would need re-establishing under the stochastic propagation model proposed as future
work (§10.2).

### 9.7 Engineering Trade-offs

The closed-loop verification step adds computational cost over emitting unverified recommendations.
Since the framework targets pre-deployment optimization rather than runtime alerting, §8.4 shows this
cost stays within CI/CD budgets at every evaluated scale, and we consider verification the feature that
distinguishes a prescription from a suggestion.

---

## 10. Conclusion and Future Work

### 10.1 Conclusions

This paper presented SaG-Prescribe, a closed-loop, DevOps-integrated prescriptive refactoring system
for distributed publish-subscribe architectures. By augmenting topological diagnostics with a
code-quality bridge (CQP) and extending the Software-as-a-Graph baseline, we introduced an end-to-end
pipeline that translates code-level debt and topological vulnerabilities into three concrete mutation
operators — topic splitting, anti-affinity reallocation, and QoS hardening — verifies every compiled
policy against the same discrete-event cascade oracle that produced the diagnosis, and operationalizes
the result as a delta-aware CI/CD quality gate. Across seven benchmark scenarios, the generated
prescriptions reduce the System Risk Index in every case (1.4%–15.9% relative, Wilcoxon $W=0.0$,
$p=0.0156$) within runtime envelopes compatible with merge-request-gated or nightly CI/CD pipelines,
while an honest component-level audit (§8.3) shows the current greedy policy is not yet uniformly
beneficial below the system level.

### 10.2 Future Work

1. **Per-edit acceptance filter (highest priority):** eliminating counterproductive mutations before
   policy execution, directly motivated by the component-level regressions of §8.3.
2. **Budget-constrained policy search:** incorporating explicit cost models and modification budgets
   ($\mathcal{B} < \infty$ in §4), turning the engine from exhaustive rule firing into subset selection
   over candidate mutations.
3. **Stochastic cascade propagation:** the current simulator is deterministic under its default
   configuration (cascade probability 1.0), so seed variance is trivially zero (§7.3). Introducing
   probabilistic propagation ($<1.0$) would make the $\kappa\cdot\sigma_{\text{seed}}$ acceptance
   criterion of §4 load-bearing and let RQ1 test robustness to genuine simulation noise.
4. **Operator ablation and learned policy ordering:** isolating per-operator $\Delta$SRI contributions
   and exploring learned prioritization over the rule-based candidate set.
5. **Real-system replication:** applying the engine to the ATM topology of [1] and to harvested
   industrial configurations, closing the external-validity gap of §9.5.
6. **LLM-assisted pull-request generation:** linking large language model code assistants to
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

> *[Reference-list note, not part of the manuscript: references [2]–[11] were sourced from the most
> polished prior draft and verified as real, non-invented candidates via literature search; abstracts/
> metadata were checked but full text was not read for each — sanity-check relevance before submission.
> AuSE reviewers will expect ~30–45 references total. The three `[REF: …]` placeholders in §2.2 still
> need populating with real code-level refactoring-recommendation citations; the P. T. Eugster et al.
> pub/sub survey, the DDS and MQTT standards documents, and a GNN benchmark-dataset paper appeared in
> one source draft's reference list but were dropped here — the standards documents and survey are
> reasonable to reinstate if §2.1 needs strengthening, but the GNN benchmark paper's relevance to this
> rule-based, non-learned pipeline was flagged as unclear in the source outline and was not carried
> over.]*

## Declarations

- **Funding:** [to be completed]
- **Competing interests:** [to be completed]
- **Data availability:** A replication package containing scenario configurations, seeds, and the
  prescriptive pipeline implementation will be made available at [URL pending].
- **Ethics approval:** Not applicable.
