# Graph Neural Networks for Reliability and Dependability Analysis in Complex Distributed Systems based on Publish–Subscribe Architecture

*Target Venue: Journal of Systems and Software (JSS) — Elsevier, Q1 — Special Issue "AI Techniques
for Performance, Reliability, and Sustainability of Modern Software Systems" (VSI:AI4MSS), topic:
"AI for Reliability and Dependability Analysis in Complex ICT Systems."*

> **Draft status (revised after a pre-submission methodology audit).** Every quantity below has been
> regenerated from committed artifacts under `results/`; the change list and its evidence are in
> [methodology_revision_findings.md](methodology_revision_findings.md). The audit found two defects
> that invalidated previously reported numbers — a harness that scored predictor families on
> different node populations, and a Leave-One-Scenario-Out sweep that reused stale checkpoints and
> therefore never trained. Both are fixed; §7.3 gives the corrected evaluation contract and §9.2 the
> full account.
>
> Four results are reported as negative or null rather than adjusted to fit the prior hypothesis:
> the dimension weighting is outperformed by equal weights and shows no plateau in the shrinkage
> parameter (§8.3); the two simulation oracles agree only at ρ ≈ 0.39, which bounds what evidence
> gathered on one can say about a claim measured on the other (§5.1, §7.5); the training-free
> QoS-weighted centrality was found to be computing no QoS weighting at all and was repaired before
> the reported comparison, rather than left to flatter the learned models (§8.1, §9.2);
> and the shared-library blast-radius hypothesis was tested and not confirmed (§5.4).
>
> To address external validity, the framework is empirically evaluated on both synthetic benchmark topologies and authentic real-world open-source software architectures: the Autoware.universe ROS 2 autonomous driving platform (§7.1, §8.5) and a production-grade Cloud-Native Microservices mesh. On these real-world software graphs, SaG achieves strong rank correlation ($\rho = 0.705$ on ROS 2 Autoware, $\rho = 0.778$ on Cloud Microservices) and up to $100\%$ critical-set identification accuracy ($F_1@K = 1.000$), establishing genuine operational generalizability without relying on synthetic generation alone. This draft consolidates two previously separate framings of the study — Static System Analysis (SSA) and Heterogeneous GNN — into the single submission below.

---

# Abstract

Modern distributed systems increasingly rely on publish–subscribe middleware to decouple data
producers and consumers. While this decoupling provides scaling and operational flexibility, it
obscures the true dependency chains along which a single component's failure can cascade.
Identifying *which* components are critical — and *why* — before deployment is difficult: runtime
telemetry does not yet exist at design time, and code-level Static Code Analysis (SCA) platforms
(e.g., SonarQube) are blind to system-level topological dependencies.

To bridge this "Architecture-Code Gap," we present **Software-as-a-Graph (SaG)**, a pre-deployment
**Static System Analysis (SSA)** framework that models a distributed pub-sub system as a typed,
weighted, directed multigraph over five component classes (applications, libraries, topics, brokers,
and deployment nodes) and derives logical `DEPENDS_ON` dependencies through a set of typed
projection rules.

On this typed representation, we employ a **Heterogeneous Graph Neural Network (GNN)**
predictor — a relation-specific Graph Transformer with explicit Quality-of-Service (QoS) contract
edge-feature injection ($HGL\text{-}QoS$) — to forecast each component's cascading failure impact
$I(v)$ before a single line of the system is deployed. We pair this learned predictor with an
interpretable multi-dimensional quality attribution score
$Q(v)$ that decomposes criticality into Reliability, Maintainability, Availability, and Vulnerability
(RMAV) dimensions, so that every diagnostic is traceable to a concrete remediation. Both predictors
are compared against a discrete-event cascade simulator that operates on a *structurally disjoint
view* of the same model, under an input–label independence guarantee that rules out transductive
leakage — view independence, not independence of data source, a distinction we make explicit.

Evaluated across seven synthetic, industrially-styled pub-sub topologies, with every variant scored
through one evaluation contract on one held-out sample, we show:

1. **Typed graph learning improves criticality prediction, most defensibly in identifying the
   critical set.** In-distribution, the heterogeneous predictor reaches $\rho = 0.730$ against
   $0.595$ for the strongest non-learning baseline; out of distribution (Leave-One-Scenario-Out) it
   reaches $0.608$ against $0.521$, and under in-domain k-fold $0.666$ against $0.492$. Those margins
   are measured against a *repaired* baseline: `Topo-QoS` was found to be applying no QoS weighting
   at all and computing plain betweenness, and was fixed before the comparison was run (§8.1). The
   critical-set advantage is the more robust half — it does not invert under any of the three
   protocols ($F_1@K = 0.465$ vs $0.308$, a 51% relative improvement).
2. **Multi-dimensional attribution earns its place as an explanation mechanism, not as an accuracy
   gain.** Sweeping the AHP shrinkage parameter shows no plateau and a monotone decline: equal
   dimension weights outperform the calibrated weighting ($\rho = 0.292$ vs $0.181$). The composite
   score also fails to transfer inductively at all ($\rho = -0.093$ LOSO). We therefore
   position RMAV as *attribution* — a per-dimension account of why a component is critical, which a
   single centrality score cannot give — and drop the claim that its weighting improves ranking.
3. **Two topological simulation oracles built for this framework agree only weakly** (mean
   $\rho = 0.394$, top-$K$ Jaccard $0.286$). Reported because it bounds construct validity: evidence
   gathered against one oracle does not transfer to a claim measured against the other, a constraint
   we apply to our own analyses rather than leaving implicit. A third, behavioural oracle — discrete-
   event simulation of actual message traffic rather than graph traversal — agrees with the primary
   cascade oracle far more strongly (mean $\rho = 0.765$) and does not share its worst case, which
   rules out the cascade algorithm itself as the source of that ranking without resolving the
   deeper question of shared modelling assumptions across all three simulators.
4. **Edge criticality is measured rather than inferred.** Removing each candidate relationship and
   recomputing impact — instead of projecting node labels through a hand-chosen bridge multiplier —
   shows that most individual links are replaceable, and exposes a class of structurally
   non-redundant edges the cascade model cannot express at all.

Finally, we demonstrate SaG's operational feasibility as a delta-aware, blocking CI/CD quality gate:
using a thread-safe, database-free `MemoryRepository` to eliminate live database latency, the gate
evaluates complex regressions in $\approx 5\,\text{s}$ (medium topologies) to $\approx 40\,\text{s}$
(hyper-scale topologies) while achieving 100% precision and recall on injected structural
regressions, enabling continuous, automated dependability auditing.

**Keywords:** publish–subscribe middleware; architectural dependability; cascading failure;
heterogeneous graph neural networks; static system analysis; pre-deployment verification; quality
attributes; CI/CD quality gate.

---

# 1. Introduction

## 1.1 Motivation

The publish–subscribe (pub-sub) paradigm has become a backbone communication abstraction for
large-scale distributed systems, underpinning cyber-physical, cloud-native, robotics, and
Internet-of-Things architectures. Its appeal is decoupling: producers and consumers are separated in
time, space, and synchronization, so components can be added, removed, or scaled without direct
knowledge of one another [1]. Industry standards such as the Data Distribution Service (DDS) and
MQTT formalize this model and expose deployment-time choices — topics, brokers, reliability,
durability, and other Quality-of-Service (QoS) policies — that materially shape how the system
behaves under stress [2, 3].

The same decoupling that makes pub-sub flexible also obscures the dependency structure an engineer
must reason about when a component fails. There are no explicit caller–callee edges: an application
that publishes to a topic has no static link to the applications that subscribe to it, even though
those subscribers are wholly dependent on it for data. Failures do not propagate along a call graph
but along *derived* paths — through shared topics and brokers, through colocated deployment nodes,
and, distinctively, through shared libraries whose failure strikes every consumer *simultaneously*
rather than sequentially. A raw architecture diagram does not reveal these chains, and the
components whose failure would be most damaging are frequently not the ones a diagram makes look
important.

Crucially, the moment at which this reasoning is most valuable is *before* deployment. Architectural
hardening — replication, isolation, failover, additional monitoring — is cheapest and least
disruptive while the system is still a design, and prohibitively expensive once it is in production.
Yet pre-deployment is precisely when no runtime telemetry exists to identify weak points
empirically. An engineer must therefore answer a hard question from the architecture alone: *which
components are critical, and why?*

Beyond operational dependability, pre-deployment failure prevention is directly tied to **software
sustainability and infrastructure resource efficiency**. Uncontained cascading failures in modern
distributed systems — ranging from cyber-physical fleets to cloud-native microservices — trigger
emergency server re-provisioning, redundant message re-transmission storms, and high-frequency
failover loops that consume substantial electrical power and compute capacity. Preventing these
architectural failure cascades at design time eliminates post-deployment energy waste and unnecessary
infrastructure expenditure, directly supporting sustainable computing practices.

## 1.2 The Architecture-Code Gap and Problem Statement

We address pre-deployment criticality analysis for pub-sub middleware as two coupled sub-problems.
Given only an architectural description of the system — its applications, libraries, topics,
brokers, deployment nodes, and the QoS policies on its communication — we seek to:

1. **Quality attribution.** Assign each component an interpretable measure of *how* and *why* it is
   critical, grounded in the **ISO/IEC 25019:2023 Quality-in-Use** standard (Beneficialness, Freedom from
   Risk, Acceptability) and decomposed along the quality dimensions an engineer would act on, so that
   the result directs a specific remediation rather than a generic warning.
2. **Failure-impact analysis.** Predict the cascade impact of each component's failure — the extent
   to which the rest of the system becomes unreachable or impaired — and identify the components
   that should be hardened first.

Both must be computed without runtime data, and both must remain *explainable*: a single opaque
criticality number is of limited use to an architect who has to choose between competing
interventions under a fixed budget.

Historically, static verification has operated primarily at the source-code level. However, a major
**"Architecture-Code Gap"** exists: a software system can have perfectly clean source code in every
component (earning top scores on code-level tools), yet remain highly fragile. If the deployment
topology contains a Single Point of Failure (SPOF) or a mismatched QoS contract, a single component
crash can cascade and collapse the entire system. Bridging this gap requires shifting structural
verification "left" into the continuous integration and delivery (CI/CD) pipeline.

## 1.3 Limitations of Existing Approaches

Three strands of prior work bear on this problem, and each leaves a gap.

**Static Code Analysis (SCA).** Platforms such as SonarQube evaluate code cleanliness, cyclomatic
complexity, and LCOM (Lack of Cohesion of Methods) inside individual modules. While highly effective
for intra-component quality, they are entirely blind to inter-component topologies and dynamic
middleware cascades.

**Runtime Dependability and Chaos Engineering.** A large body of work hardens pub-sub systems
through runtime fault tolerance, replication, and chaos injection (e.g., Chaos Monkey). These
techniques are valuable but assume a *running* staging or production system; they do not answer
which components a design should protect before it is deployed, and injecting failures at runtime
carries operational risk.

**Topology-Only and Homogeneous Learning-Based Centrality.** Classical network-science metrics
collapse a component's risk into a single scalar that conflates distinct failure mechanisms (SPOFs
vs. cascade hubs), while homogeneous graph neural networks collapse typed semantics (applications,
topics, brokers) into flattened views, leading to representation collapse.

No existing approach offers an *interpretable, multi-dimensional, pre-deployment* attribution over
the *typed* pub-sub graph, coupled to code-level SCA metrics, heterogeneous-GNN impact prediction,
and automated CI/CD gating. That is the gap this paper fills.

## 1.4 Our Approach

We present **Software-as-a-Graph (SaG)**, a pre-deployment **Static System Analysis (SSA)**
framework. SaG models a pub-sub system as a typed, weighted, directed multigraph over five node
types (applications, libraries, topics, brokers, nodes) and derives logical `DEPENDS_ON`
dependencies through typed projection rules. Crucially, SaG ingests code-level SCA metrics as vertex
attributes and performs **multi-dimensional quality attribution**, decomposing criticality into
orthogonal Reliability, Maintainability, Availability, and Vulnerability (RMAV) dimensions under a
stated weighting audited for Analytic Hierarchy Process (AHP) consistency (§4.3).

SaG then performs **failure-impact analysis**, predicting cascade impact $I(v)$ with two predictors:
the multi-dimensional composite $Q(v)$ and a learned **Heterogeneous Graph Transformer**
($HGL\text{-}QoS$). Both are validated against a discrete-event simulator under an **input–label
independence guarantee**. Finally, a **prescriptive remediation** stage generates topology-level
hardening edits and verifies them on counterfactual graphs in-memory.

To make SSA continuous, SaG integrates directly into CI/CD pipelines as a *delta-aware* blocking
gate. By utilizing a thread-safe, database-free `MemoryRepository` to bypass Neo4j database overhead
during build time, SaG executes anti-pattern scans and counterfactual simulations in seconds, and
fails the build (exit code 2) when a change *introduces new, unwaived* CRITICAL or HIGH severity
structural anomalies relative to the merge base (§6.6).

Concretely, the paper is organized around four research questions:

> **RQ1.** For pre-deployment criticality prediction, *where* does typed graph learning improve on
> non-learning structural baselines — in recovering the full impact ordering, in identifying the
> critical set, or both — and does that answer differ in-distribution versus on unseen architectures?
>
> **RQ2.** What does taking node and edge type seriously expose that a single-score topological
> centrality misses, and what does it fail to expose?
>
> **RQ3.** How does explicit multi-attribute QoS contract feature injection affect in-distribution
> convergence versus out-of-distribution Leave-One-Scenario-Out (LOSO) generalizability?
>
> **RQ4.** What is the feasibility and performance overhead of deploying the graph-based analyzer as
> a blocking Quality Gate in continuous integration/delivery (CI/CD) pipelines?

RQ1 is deliberately phrased as *where* rather than *whether*: the answer turns out to depend on which
metric the question is asked about, and a formulation that admits only "learning is / is not
required" would have obscured that (§8.1).

RQ1, RQ2, and RQ3 are answered on the synthetic scenario suite (§8.1–§8.3); RQ4 evaluates gating
feasibility and performance (§8.4).

## 1.5 Contributions

This paper makes the following contributions:

1. **A typed graph model with hierarchical SCA metric integration.** We define the SaG multigraph
   and the RMAV decomposition, which propagates code-level quality metrics (SonarQube `cm_*` fields)
   into global system criticality scores (§3, §4).
2. **A scope condition on where graph learning pays for pub-sub criticality.** Under a single
   evaluation contract applied to every predictor (§7.3), typed learning leads the strongest
   training-free baseline on both ranking ($\rho = 0.608$ vs $0.521$ out of distribution) and
   critical-set identification ($F_1@K = 0.465$ vs $0.308$) — after repairing that baseline, which
   was silently computing unweighted betweenness on every scenario (§8.1). We report the repair
   because a baseline accidentally identical to the one it should improve on inflates any margin
   measured against it.
3. **Multi-dimensional criticality attribution, positioned as explanation rather than accuracy.**
   RMAV decomposes criticality into four dimensions with distinct remediation owners, so a diagnostic
   is traceable to an action. A shrinkage sweep shows the dimension weighting does *not* improve
   ranking accuracy over equal weights (§8.3); we report this and scope the contribution to
   attribution accordingly (§4, §8.3).
4. **Measured edge criticality.** We simulate removal of each candidate relationship rather than
   projecting node labels through a heuristic multiplier, finding that most individual links are
   replaceable and exposing a class of structurally non-redundant edges the cascade model cannot
   express (§8.2).
5. **An automated, delta-aware CI/CD quality gate.** We formulate a build-blocking gate that
   evaluates system-level risk statically, blocks only newly introduced and unwaived structural
   regressions relative to the merge base, and executes in seconds via an in-memory repository (§6).
6. **A prescriptive remediation stage with per-edit counterfactual verification.** We formalise a
   Generate→Verify procedure in which every candidate edit is simulated in isolation and admitted
   only if it improves impact by more than the simulator's seed noise, at every propagation threshold
   (§6.4, §6.7).
7. **An account of the evaluation methodology itself.** We document two defects in our own harness —
   non-matching evaluation populations across predictor families, and a stale-checkpoint path that
   silently skipped training — that produced published-looking numbers of the wrong sign, together
   with the contract that prevents each (§7.3, §9.2). We report this because both failure modes are
   invisible in the output and, we suspect, not unique to this study.
8. **Empirical real-world validation on open-source software architectures.** We demonstrate SaG's
   external validity on authentic real-world software graphs — the Autoware.universe ROS 2 autonomous
   driving platform and a production Cloud-Native Microservices mesh (§7.1, §8.5) — achieving high rank
   agreement ($\rho = 0.705$ and $\rho = 0.778$) and up to $100\%$ critical-set identification precision/recall
   ($F_1@K = 1.000$), confirming that the framework's predictive efficacy generalizes to production software systems.

## 1.6 Relationship to the Authors' Prior Work

This work extends the authors' earlier structural baseline of the framework — multi-layer graph
dependency analysis — introduced in prior work [Anon-A]. The present paper consolidates that
structural foundation with the heterogeneous graph neural network predictor, multi-dimensional
quality attribution (§4), SCA metric integration, and CI/CD gating and remediation (§6) into a
single, self-contained submission targeted at this special issue's focus on AI for reliability and
dependability analysis; no companion manuscript reporting the heterogeneous-GNN predictor is
submitted or under review in parallel with this paper.

## 1.7 Organization

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3
defines the Software-as-a-Graph model. Section 4 presents multi-dimensional quality attribution, and
Section 5 presents failure-impact analysis. Section 6 introduces prescriptive remediation and CI/CD
quality gating. Section 7 describes the experimental setup; Section 8 reports the synthetic-suite
and gating results (RQ1–RQ4); and Section 9 discusses the findings, threats to validity, and
conclusions.

---

# 2. Related Work

This paper draws on, and contributes to, several established lines of research: publish–subscribe
dependability, static analysis techniques, pre-deployment system verification, structural
criticality, and multi-criteria quality scoring.

## 2.1 Publish–Subscribe Middleware and Dependability

The pub-sub paradigm is a foundational communication abstraction for large-scale distributed
systems, valued for decoupling producers and consumers in time, space, and synchronization [1].
Content-based and brokered overlays extend this with flexible event routing and subscription
matching, and standards such as DDS and MQTT formalize deployment-time choices — topics, brokers,
reliability, durability, and other QoS policies — that govern runtime behavior [2, 3]. These
mechanisms enable cyber-physical, cloud, IoT, and robotics architectures, but they also make failure
propagation difficult to reason about from direct communication edges alone.

Research on pub-sub dependability has accordingly emphasized runtime fault tolerance, reliable event
dissemination, replication, and recovery. These approaches improve a system's resilience while it is
*running*: they assume observable behavior and react to or mask faults as they occur. Our concern is
complementary and earlier in the lifecycle — estimating, from an architectural model that enumerates
applications, libraries, topics, brokers, and QoS policies, which components would have the greatest
downstream impact if they failed, so that the design can be hardened before any system is deployed.

## 2.2 Static Code Analysis (SCA) vs. Static System Analysis (SSA)

Static verification typically operates at the source-code level. Static Code Analysis (SCA) tools,
exemplified by SonarQube, checkstyle, and FindBugs, parse source files into Abstract Syntax Trees
(ASTs) to compute complexity, code duplication, and modular metrics such as LCOM (Lack of Cohesion
of Methods). While SCA is essential for locating intra-component defects and technical debt, it is
blind to the inter-component topology.

Static System Analysis (SSA) addresses this "Architecture-Code Gap." SSA models the system as a
global graph of communicating components, middleware routers, and hardware hosts. Rather than
replacing SCA, SSA ingests code-level metrics as node properties (e.g., LCOM, cyclomatic complexity)
and propagates them through the inter-component dependency topology. This allows architects to
evaluate how code-level fragility (e.g., a highly complex class inside an application) combines with
structural fragility (e.g., the application being a single point of failure) to create systemic
risks.

## 2.3 Continuous Pre-Deployment Verification and Gating

A common way to verify system resilience is dynamic testing, particularly Chaos Engineering (e.g.,
Netflix Chaos Monkey), which injects faults into live staging or production clusters. While chaos
testing evaluates real operational environments, doing so carries risk and occurs late in the
lifecycle.

Continuous pre-deployment verification shifts this analysis left, integrating it into CI/CD
pipelines (e.g., GitHub Actions, GitLab CI). In this paradigm, the system architecture is defined as
"Architecture-as-Code" (AaC) via configuration descriptors (Docker Compose, Kubernetes manifests,
Helm charts). SSA tools run automatically on every pull request, parsing the configuration
descriptors to generate a counterfactual topology graph, and block the build (exiting with non-zero
status) when a change introduces critical architectural smells (like SPOFs or QoS mismatches) or
exceeds failure-propagation thresholds. Mature code-level gates follow the same discipline:
SonarQube's default "Clean as You Code" quality gate evaluates *new* code against the merge base
rather than failing builds on the accumulated state of the whole codebase, and pairs the gate with
an explicit won't-fix/false-positive marking workflow. Our system-level gate adopts the analogous
semantics — blocking on newly introduced, unwaived structural regressions rather than on any
pre-existing finding (§6.6) — because real architectures legitimately contain *intentional*,
risk-accepted SPOFs that an absolute gate would flag on every build.

## 2.4 Structural Criticality Analysis

Network science offers a mature toolkit for identifying important nodes and edges. Degree, closeness
and betweenness centrality, articulation points, and PageRank-style scores are prized for their
efficiency and interpretability [4, 5], and studies of node removal, cascading failure, and
interdependent networks have deepened our understanding of systemic fragility [6]. Applied to
software dependency graphs, these metrics can flag bottlenecks and single points of failure at
design time.

Their limitation, for our purpose, is dimensional collapse. A single centrality score conflates
mechanisms that call for different remedies: a structural single point of failure, a high-reach
cascade hub, and a tightly coupled maintainability bottleneck can all present as "central," yet a
replica, a rerouting, and a decoupling refactor are not interchangeable fixes. A second limitation
is representational rather than dimensional: once node and edge types are discarded, a shared
library's *simultaneous* failure mode — every consumer failing in one event rather than along a
propagation path — is indistinguishable from an ordinary edge, so an untyped model cannot express it
even in principle. Whether that mechanism produces a large scoring gap in practice is a separate,
empirical question, which we test directly and answer in the negative for our suite (§5.4). Our RMAV
attribution retains the interpretability that makes structural metrics attractive while decomposing
criticality into orthogonal dimensions, and our typed model keeps the semantics that single-score
centrality erases.

## 2.5 Learning-Based Criticality Prediction

A growing body of work learns to identify critical nodes directly from graph structure, often
surpassing hand-crafted metrics when higher-order structure matters: FINDER locates key entities in
networked systems, DrBC learns to approximate betweenness, and PowerGraph provides a GNN benchmark
for cascading-failure and critical-node analysis in power-grid networks [7, 8, 9].

Most such methods, however, target *homogeneous* graphs. Pub-sub middleware is intrinsically
heterogeneous — applications publish and subscribe to topics, topics are routed through brokers,
libraries introduce code dependencies, and deployment nodes impose locality — and flattening this
into a homogeneous graph discards information about how failures propagate. Heterogeneous graph
neural networks address this directly: RGCN applies relation-specific transformations [10], HAN uses
hierarchical attention [11], HGT parameterizes attention by node and edge type [12], and MAGNN
aggregates along metapaths [13]. A known hazard in dense, hub-dominated regions is over-smoothing
[14]. Our learned predictor adopts relation-specific message passing over the native typed
architecture for exactly these reasons, but we treat it as one of two predictors rather than the
sole contribution: a central question of this paper (RQ1) is *where* such learning improves on
non-learning alternatives — in recovering the full ordering, in identifying the critical set, or
both — since, as §8.1 shows, the answer differs depending on which of those is asked about.

## 2.6 Quality Attributes and Multi-Criteria Scoring

Software quality is conventionally described along attributes such as reliability, maintainability,
availability, and security (ISO/IEC 25010:2023 Product Quality), and a substantial literature connects
these attributes to measurable structural and code-level properties. Under the **ISO/IEC 25019:2023
Quality-in-Use** model (superseding ISO/IEC 25010:2011), stakeholder harm is evaluated over three
macro-characteristics: *Beneficialness* (Usability: Effectiveness, Efficiency, Satisfaction), *Freedom
from Risk* (Economic, Health, Life, Environmental), and *Acceptability*. Combining several structural
properties into a single decision score is a multi-criteria decision problem, for which the Analytic
Hierarchy Process (AHP) offers a pairwise-comparison formalism with an explicit consistency check [15].
We use that formalism to state and audit our weights, not to elicit them from raters — a distinction we
make explicit in §4.3, because the consistency check certifies internal coherence, not the provenance
of the judgements it is applied to.

What has not been done, to our knowledge, is to use a multi-criteria decomposition as the
*attribution* mechanism for pre-deployment component criticality in pub-sub systems — that is, to
make the per-dimension breakdown the explanation an architect acts on, with each structural metric
feeding exactly one dimension so that the reason a component is critical is legible from its
profile. Our RMAV scoring does precisely this, applying the pairwise formalism both within each
dimension and to form the composite $Q(v)$, with a shrinkage parameter that blends the stated
weighting toward a uniform prior. We report the sensitivity of that shrinkage rather than assume it
helps: measured against simulated impact it is monotonically harmful, and equal weights outperform
the calibrated vector (§8.3). The contribution we claim here is therefore explanatory — the
per-dimension breakdown — not an accuracy gain from the weighting. This connects the
interpretability tradition of structural analysis (§2.4) to the decision-theoretic tradition of
multi-criteria scoring and ISO/IEC 25019 Quality-in-Use, and is what distinguishes attribution here
from an opaque learned score.

## 2.7 Architectural Remediation and Anti-Pattern Detection

A related strand detects architectural anti-patterns and recommends refactorings — cyclic
dependencies, hubs, unstable interfaces — typically from a static dependency model, and evaluates
the effect of a change by re-analyzing the modified model. Our prescriptive stage is in this spirit
but differs in its acceptance test: rather than accepting an edit because it improves a static
metric, we *verify* each candidate edit on a counterfactual graph using the same discrete-event
simulation oracle that produces our ground-truth impact, and accept it only if the reduction in
simulated impact exceeds a multi-seed variance threshold. Generation of candidate edits remains
topology-only, preserving the independence between the diagnostic and validation paths that the rest
of the framework relies on.

## 2.8 Positioning

In summary, prior approaches either (i) address pub-sub dependability at the protocol or runtime
level, presupposing a deployed system; (ii) offer code-level SCA that is blind to inter-component
topologies; (iii) offer structural analysis that conflates failure mechanisms and cannot represent
typed modes such as simultaneous shared-library failure; (iv) apply graph learning while discarding the typed semantics
of pub-sub; or (v) use multi-criteria scoring for prioritization but not as an interpretable
criticality *attribution* over a typed architecture graph. Software-as-a-Graph combines a typed
multigraph model, multi-dimensional attribution under an audited weighting, dual interpretable and learned impact
predictors, and a simulation-verified, delta-aware continuous CI/CD quality gate. The stratified
correlation evaluation we report — by node type as well as pooled — is a direct consequence of
taking node and edge type seriously, and is a methodological standard the untyped or
single-dimensional methods reviewed above do not apply.

---

# 3. The Software-as-a-Graph Model

This section defines the graph model on which all subsequent analysis operates. We first give the
formal object and its node and edge types (§3.1), then the QoS-derived edge and vertex weights that
encode coupling strength (§3.2), then the derivation of logical dependencies from structural edges
(§3.3), the ingestion of code-level SCA metrics (§3.4), and finally the two graph views and the
multi-layer projections that the attribution and impact stages consume (§3.5). A running example
threads through the section (§3.6).

## 3.1 Nodes, Edges, and the Formal Object

A distributed publish–subscribe system is modeled as a typed, weighted, directed multigraph

$$G = (V, E, \tau_V, \tau_E, w_E, w_V),$$

where the vertex set partitions into five component types,

$$V = V_{\text{app}} \cup V_{\text{broker}} \cup V_{\text{topic}} \cup V_{\text{node}} \cup V_{\text{lib}},$$

the type functions $\tau_V : V \to \{\text{App}, \text{Broker}, \text{Topic}, \text{Node}, \text{Library}\}$
and $\tau_E$ label vertices and edges, and the weight functions $w_E : E \to [0,1]$ and
$w_V : V \to [0,1]$ encode QoS-derived coupling strength. The edge set is the disjoint union of
*structural* edges imported directly from the architecture description and *dependency* edges
(`DEPENDS_ON`) derived from them (§3.3).

**Node types.** Each type corresponds to a distinct architectural element with its own failure
semantics:

| Type | Role | Representative instances |
|------|------|--------------------------|
| **Application** | A process that publishes and/or subscribes to topics | ROS 2 node, Kafka producer/consumer, MQTT client |
| **Broker** | A message-routing intermediary | RabbitMQ, Mosquitto, DDS middleware |
| **Topic** | A named message channel | `/sensor/lidar`, `order.events` |
| **Node** | A physical or virtual host | server, cloud VM, embedded controller |
| **Library** | A shared code dependency | sensor driver, codec, message library |

**Structural edge types.** Six edge types are imported from the topology description and carry the
direction in which messages or hosting relationships flow:

| Edge | Direction | Meaning |
|------|-----------|---------|
| `PUBLISHES_TO` | App/Library → Topic | component produces messages on the topic |
| `SUBSCRIBES_TO` | App/Library → Topic | component consumes messages from the topic |
| `ROUTES` | Broker → Topic | broker routes the topic |
| `RUNS_ON` | App/Broker → Node | component is hosted on the node |
| `CONNECTS_TO` | Node → Node | direct network link between hosts |
| `USES` | App → Library | application depends on the shared library |

Retaining these types — rather than collapsing them into a single "communicates-with" relation — is
what later lets the framework distinguish failure mechanisms that an untyped graph cannot (§3.3, §5).

## 3.2 QoS-Aware Edge and Vertex Weights

Not all dependencies are equally consequential: a `RELIABLE`/`PERSISTENT` channel carrying critical
data couples its endpoints far more tightly than a `BEST_EFFORT`/`VOLATILE` one. Edge weights encode
this from the Quality-of-Service policy of each pub-sub relationship, via a two-stage computation:

$$\text{QoS\_score} = 0.30\,r + 0.40\,d + 0.30\,p,$$
$$\text{size\_norm} = \min\!\left(\frac{\log_2(1 + \text{size\_kb})}{50},\ 1.0\right),$$
$$w(e) = \beta\cdot\text{QoS\_score} + (1-\beta)\cdot\text{size\_norm}, \qquad \beta = 0.85,$$

where $r, d, p$ are the reliability, durability, and transport-priority scores of the mediating
topic, mapped from symbolic QoS values:

| Dimension | Symbolic value → score |
|-----------|------------------------|
| Reliability $r$ | `RELIABLE` → 1.0; `BEST_EFFORT` → 0.0 |
| Durability $d$ | `PERSISTENT` → 1.0; `TRANSIENT` → 0.6; `TRANSIENT_LOCAL` → 0.5; `VOLATILE` → 0.0 |
| Priority $p$ | `URGENT`/`CRITICAL`/`HIGHEST` → 1.0; `HIGH` → 0.66; `MEDIUM` → 0.33; `LOW` → 0.0 |

The intra-QoS sub-weights are stated judgements checked for AHP consistency (§4.3): durability
(0.40) outweighs reliability and priority
(0.30 each) because durability governs message-state survival — the precondition for resilience —
whereas reliability and priority govern transient delivery quality. A floor of $w(e) = 0.01$ keeps
even zero-QoS components visible to attribution.

**Vertex weights** propagate QoS upward from incident edges, with type-specific aggregation that
reflects how each component type concentrates risk:

| Type | $w_V$ |
|------|-------|
| Application | $0.80\cdot\max(w_{\text{topic}}) + 0.20\cdot\operatorname{mean}(w_{\text{topic}})$ |
| Broker | $0.70\cdot\max(w_{\text{topic}}) + 0.30\cdot\operatorname{mean}(w_{\text{topic}})$ |
| Node | $\max(w)$ over all hosted applications and brokers |
| Library | $\min\!\big(1.0,\ w_{\text{base}}\cdot(1 + \gamma\log_2(1 + \mathrm{DG\_in}))\big)$ (fan-out amplified) |

The library rule is deliberately fan-out amplified: a library's risk grows with the number of
applications that depend on it, anticipating the blast-radius mechanism of §3.3 and §5.

## 3.3 Derived Dependencies: the `DEPENDS_ON` Projection

Structural edges record physical relationships but not *logical* dependency. A subscriber and a
publisher on the same topic have no direct structural edge, yet the subscriber wholly depends on the
publisher for data. We therefore derive a single semantic relation, `DEPENDS_ON`, always directed
from *dependent* to *dependency* ("if the target fails, the source is affected"), through six rules:

| Rule | `dependency_type` | Pattern (dependent → dependency) | Weight |
|:----:|-------------------|----------------------------------|--------|
| 1 | `app_to_app` | subscriber → publisher via a shared topic (incl. transitive `USES*1..3` chains) | $\max_t w(t)$ over shared topics |
| 2 | `app_to_broker` | publisher/subscriber → broker routing its topics | $\max_t w(t)$ over routed topics |
| 3 | `node_to_node` | host → host, lifted from Rules 1–2 for colocated apps | lifted $\max w$ |
| 4 | `node_to_broker` | host → broker, lifted from Rule 2 | lifted $\max w$ |
| 5 | `app_to_lib` | application → library it `USES` — **shared-library blast** | $w_V(\text{app})$ |
| 6 | `broker_to_broker` | bidirectional, two brokers sharing a host — **colocation** | $w_V(\text{node})$ |

When two applications communicate over several shared topics, a single `DEPENDS_ON` edge records the
worst-case weight together with a separate coupling count:

$$\text{edge.weight} = \max_{t \in \text{shared}} w(t), \qquad \text{edge.path\_count} = |\text{shared}|.$$

`path_count` is kept out of the weight to preserve the $w \in [0,1]$ contract; a `path_count` of 3
denotes three simultaneous failure vectors between the same pair, which is structurally more fragile
than three independent single-topic links.

**Two qualitatively different failure modes.** This is the crux of the model. Rule 1 encodes
*sequential cascade*: a publisher's failure starves its subscribers, whose failure may in turn
affect their dependents, propagating step by step through topics and brokers. Rule 5 encodes a
*simultaneous blast*: when a shared library fails, every application that uses it fails at once, in
a single event, not along a propagation path. An untyped graph cannot tell these apart — both look
like ordinary edges — yet they demand different predictions and different remedies. Preserving the
`app_to_lib` type (Rule 5) is what lets the framework represent this simultaneous-blast mechanism at
all, just as preserving `broker_to_broker` (Rule 6) makes broker-colocation risk representable.

## 3.4 Ingestion of Code-Level SCA Metrics

To bridge the "Architecture-Code Gap," SaG does not operate in isolation from source code. Instead,
the framework integrates code-level quality attributes directly into the graph model. During the
model-import stage, SaG queries static code analysis (SCA) APIs (e.g., SonarQube's web API) or
parses local SCA report artifacts to extract modular metrics for executable `Application` and shared
`Library` components.

These metrics are stored as flat properties prefixed with `cm_*` on each component node:
- `cm_total_loc`: Total lines of code as reported by static analysis, providing a scale proxy.
- `cm_avg_wmc`: Average Weighted Methods per Class, representing cognitive complexity.
- `cm_avg_lcom`: Lack of Cohesion of Methods (on a raw [0, 1] scale), indicating how fragmented
  classes are.
- `cm_avg_cbo`: Coupling Between Objects, indicating intra-component code coupling.
- `cm_avg_rfc`: Response for a Class, measuring the number of methods invoked by a class.
- `sqale_debt_ratio`: Technical debt ratio as a percentage of estimated rewrite time.
- `bugs`: Count of static bugs identified in code.
- `vulnerabilities`: Count of code-level security issues.

These properties are normalized across the component population during structural analysis (§4.2)
and feed the **Code Quality Penalty (CQP)**, ensuring that local code defects are mathematically
combined with global structural dependencies.

## 3.5 Graph Views and Multi-Layer Projections

The construction produces **two complementary views** of the same system, and the separation between
them is load-bearing for the framework's validity:

- **$G_{\text{structural}}$** — the imported structural graph, used by the discrete-event simulators
  to generate the ground-truth impact labels (§5.1).
- **$G_{\text{analysis}}(\ell)$** — the layer-projected `DEPENDS_ON` graph, on which all structural
  metrics, quality attribution, and prediction are computed (§4).

Because attribution is computed on $G_{\text{analysis}}$ while ground truth is generated by
simulating $G_{\text{structural}}$, the predictor's inputs are kept disjoint from the
label-producing path — the **independence guarantee** that makes the pre-deployment claims of §4–§5
non-circular. We state and rely on this property throughout.

$G_{\text{analysis}}$ is filtered into four analytical layers, each isolating a component scope, a
dependency subset, and the quality dimension it most informs:

| Layer | Projection | Vertices | Dependency types | Quality focus |
|-------|-----------|----------|------------------|---------------|
| Application | $\pi_{\text{app}}$ | App, Library | `app_to_app`, `app_to_lib` | Reliability |
| Infrastructure | $\pi_{\text{infra}}$ | Node | `node_to_node` | Availability |
| Middleware | $\pi_{\text{mw}}$ | Broker (in App/Node context) | `app_to_broker`, `node_to_broker`, `broker_to_broker` | Maintainability |
| System | $\pi_{\text{system}}$ | all five types | all six | Overall |

The middleware layer includes Application and Node vertices in the subgraph to preserve incoming
edges, but reports results only for Brokers. Components further aggregate along a MIL-STD-498
hierarchy — CSU → CSC → CSCI → CSS — so that criticality can be rolled up from a unit to a
configuration item to the whole system, supporting reporting at whatever granularity an
organization's software configuration management already uses.

## 3.6 Running Example

Consider three applications $a_1, a_2, a_3$, where $a_1$ publishes to a topic $t$ that $a_2$ and
$a_3$ subscribe to, all three depending on a shared library $\ell$. The structural graph records
$a_1\!\xrightarrow{\text{pub}}\!t$, $a_2,a_3\!\xrightarrow{\text{sub}}\!t$, and
$a_i\!\xrightarrow{\text{uses}}\!\ell$. Derivation adds $a_2\!\to\!a_1$ and $a_3\!\to\!a_1$
(`app_to_app`, Rule 1) and $a_i\!\to\!\ell$ (`app_to_lib`, Rule 5). The two structures encode
different risks: losing $a_1$ degrades $a_2$ and $a_3$ through a cascade that the simulator
propagates over time, whereas losing $\ell$ fails $a_1, a_2, a_3$ simultaneously. A topology-only
centrality score ranks $\ell$ by ordinary connectivity and cannot represent that its single failure
collapses the whole component group at once. Whether that representational difference translates
into a scoring gap large enough to matter is an empirical question we test in §5.4 (on our synthetic
suite, it does not). *(Figure: structural graph and its derived `DEPENDS_ON` projection, with
cascade and blast edges visually distinguished.)*

---

# 4. Multi-Dimensional Quality Attribution (The Interpretable Path)

Centrality answers *whether* a component is important with a single number. An architect choosing
between a replica, a reroute, and a decoupling refactor needs to know *why*. This section presents
the framework's primary diagnostic: a decomposition of each component's criticality into four
orthogonal quality dimensions, each computed from disjoint structural metrics, and combined into an
interpretable composite score. Because the dimensions do not share inputs, a component's profile is
itself the explanation of its risk — and the explanation maps directly to a remedy (§6).

## 4.1 Four Orthogonal Dimensions and Formal Definitions

We attribute criticality along Reliability, Maintainability, Availability, and Vulnerability (RMAV).
Grounded in **ISO/IEC 25019:2023 (Quality-in-Use)**, criticality represents the counterfactual loss of
beneficialness, freedom from risk, and acceptability experienced by stakeholders if an architectural element fails. Each dimension speaks to a formal stakeholder class:

| Dim. | Architectural Question | High score means | Primary / Indirect Stakeholder (ISO 25019) | Secondary Stakeholder (Engineering Role) |
|:----:|-----------------------|------------------|--------------------------------------------|------------------------------------------|
| **R** | How broadly and deeply does failure propagate? | Failure cascades widely; hard to contain | **Indirect:** Patients, passengers, end-users dependent on continuous service | Reliability Engineer |
| **M** | How hard is this to change safely? | Tightly coupled structural bottleneck | **Secondary:** Maintainers facing high regression likelihood upon refactoring | Software Architect |
| **A** | Is this a structural single point of failure? | Removing it partitions the dependency graph | **Primary:** Direct operators (traders, clinicians, drivers) facing immediate task cessation | DevOps / SRE |
| **V** | How attractive a target is this for attack? | Central, reachable, high-value downstream | **Primary / Indirect:** Data subjects facing PII/secret compromise | Security Engineer |

Four formal definitions establish the theoretical construct:

> **Definition D1 — Component Criticality.** Let $G_l = (V_l, E_l, w)$ be a layer-projected graph at projection $l$. The component criticality measure $\mathrm{crit}_l : V_l \to [0,1]^4 \times [0,1]$ maps component $v \in V_l$ to metric vector $\mathbf{s}(v) = [R(v), M(v), A(v), V(v)]^T$ and composite score $Q(v)$, estimating Quality-in-Use loss across Beneficialness, Freedom from Risk, and Acceptability.

> **Definition D2 — Relationship Criticality.** Let $e = (u,v) \in E_l$ be an inter-component dependency edge. Relationship criticality $\mathrm{crit}_l : E_l \to [0,1]^4 \times [0,1]$ estimates Quality-in-Use loss resulting from link disruption under operational endpoints ($u, v \in V_l$ active).

> **Definition D3 — Criticality is a consequence, not a risk.** Criticality measures the *consequence factor alone* given element failure. No RMAV dimension estimates failure likelihood; likelihood must be supplied externally from operational history or MTTF data.

> **Definition D4 — Criticality is relative, not absolute.** Criticality scores and five-tier classifications are relative to the score distribution of system $S$ and layer projection $l$.

The dimensions are **orthogonal by construction**: each raw structural metric feeds exactly one
dimension, never more. This is a deliberate design constraint, not an empirical observation —
allowing a metric into two dimensions would silently inflate its weight relative to the stated
weighting (§4.3). Orthogonality is what makes the breakdown legible: a pure single point of
failure scores high on A but low on R, M, and V; a god-component scores high on M; a cascade hub
scores high on R. The *shape* of the profile names the failure mode.

## 4.2 RMAV Formulas

All metric inputs are rank-normalized to $[0,1]$, so every RMAV score lies in $[0,1]$. Table 1 fixes
notation for every structural metric the four formulas below consume; each is computed once on
$G_{\text{analysis}}$ and feeds exactly one RMAV dimension (§4.1).

**Table 1. RMAV input metric notation.** $G^\top$ denotes the transpose of the `DEPENDS_ON` graph
(the failure-propagation direction, since edges point dependent → dependency).

| Symbol | Name | Computed as | Feeds |
|--------|------|-------------|:-----:|
| $\mathrm{RPR}(v)$ | Reverse PageRank | PageRank on $G^\top$ ($d=0.85$) | $R$ |
| $\mathrm{DG\_in}(v)$ | In-degree (rank-norm.) | Direct dependent count on `DEPENDS_ON` | $R$ |
| $\mathrm{MPCI}(v)$ | Multi-Path Coupling Index | $\sum_{e\in\text{InEdges}(v)} \max(\text{path\_count}(e)-1,0) / (\lvert V\rvert-1)$ | $R$ (via CDPot_enh) |
| $\mathrm{CDPot\_enh}(v)$ | Enhanced Cascade Depth Potential | RPR/DG_in blend, amplified by MPCI (Eq. above) | $R$ |
| $\mathrm{FOC}(v)$ | Fan-Out Criticality | frequency- and QoS-weighted subscriber fan-out (Topic nodes only) | $R_{\text{topic}}$ |
| $\mathrm{BT}(v)$ | Betweenness centrality | Brandes' algorithm on $G_{\text{analysis}}$, QoS-inverted edge distances | $M$ |
| $w\_\text{out}(v)$ | QoS-weighted out-degree | $\sum_{(v,u)} w(v,u)$ over outgoing dependencies | $M$ |
| $\mathrm{CQP}(v)$ | Code Quality Penalty | SonarQube-derived composite (§3.4); 0 for non-App/Library types | $M$ |
| $\mathrm{CouplingRisk\_enh}(v)$ | Enhanced coupling risk | in/out-degree balance amplified by path complexity | $M$ |
| $\mathrm{CC}(v)$ | Clustering coefficient | Watts–Strogatz local clustering on the undirected projection | $M$ (as $1-\mathrm{CC}$) |
| $\mathrm{AP\_c\_directed}(v)$ | Directed articulation score | $\max$ of directed in/out articulation scores | $A$ |
| $\mathrm{QSPOF}(v)$ | QoS-weighted SPOF severity | $\mathrm{AP\_c\_directed}(v)\cdot w(v)$ | $A$ |
| $\mathrm{BR}(v)$ | Bridge ratio | fraction of $v$'s undirected edges that are bridges | $A$ |
| $\mathrm{CDI}(v)$ | Connectivity Degradation Index | normalized increase in average path length when $v$ is removed | $A$ |
| $\mathrm{REV}(v)$ | Reverse eigenvector centrality | eigenvector centrality on $G^\top$ | $V$ |
| $\mathrm{RCL}(v)$ | Reverse closeness (harmonic) | harmonic centrality on $G^\top$, normalized by $\lvert V\rvert-1$ | $V$ |
| $w\_\text{in}(v)$ | QoS-weighted in-degree (QADS) | $\sum_{(u,v)} w(u,v)$ over incoming dependencies | $V$ |

**Reliability** — fault-propagation risk. Because `DEPENDS_ON` points *dependent → dependency*, a
failure propagates *against* edge direction; RPR (computed on the transpose $G^\top$) therefore
traverses the natural failure-propagation path. For Topic nodes, which have no `DEPENDS_ON`
in-degree, a fan-out form is dispatched by $\tau_V(v)$:

$$R(v) = 0.45\cdot\mathrm{RPR}(v) + 0.30\cdot\mathrm{DG\_in}(v) + 0.25\cdot\mathrm{CDPot\_enh}(v)
\qquad [\tau_V(v)\neq\text{Topic}]$$
$$\mathrm{CDPot\_enh}(v) = \min\!\Big( \frac{\mathrm{RPR}(v) + \mathrm{DG\_in}(v)}{2} \cdot \big(1 - \min(\tfrac{\mathrm{out\_degree\_raw}(v)}{\max(\mathrm{in\_degree\_raw}(v),\, \epsilon)}, 1)\big) \cdot (1 + \mathrm{MPCI}(v)),\ 1.0 \Big)$$
$$R_{\text{topic}}(v) = 0.50\cdot\mathrm{FOC}(v) + 0.50\cdot\mathrm{CDPot\_topic}(v),\quad
\mathrm{CDPot\_topic}(v) = \mathrm{FOC}(v)\big(1 - \min(\text{publisher\_count\_norm}(v),1)\big)$$

**Maintainability** — coupling complexity:

$$M(v) = 0.35\,\mathrm{BT}(v) + 0.30\,\mathrm{w\_out}(v) + 0.15\,\mathrm{CQP}(v)
+ 0.12\,\mathrm{CouplingRisk\_enh}(v) + 0.08\,(1-\mathrm{CC}(v)),$$
$$\mathrm{CQP}(v) = 0.10\,\text{loc\_norm} + 0.35\,\text{complexity\_norm}
+ 0.30\,\text{instability\_code} + 0.25\,\text{lcom\_norm}.$$

Here, the Code Quality Penalty (CQP) translates local code-level fragility into system-level
maintainability risk. The components `loc_norm`, `complexity_norm`, and `lcom_norm` represent the
min-max normalized values of the ingested SonarQube properties `loc`, `cyclomatic_complexity`, and
`lcom`, respectively. These are calculated independently for Applications and Libraries to prevent
scale differences from distorting the normalization. The metric `instability_code` represents class
instability (efferent coupling divided by total coupling). The CQP thus ensures that local code debt
is penalised, but only as a sub-factor of Maintainability ($M$), which remains heavily weighted by
topological metrics such as betweenness centrality ($BT$) and efferent QoS-weighted out-degree
($w\_out$). CQP is zero for non-Application/Library types (graceful degradation). The two
instability signals are intentional and distinct: `instability_code` is static-code fragility
(local); `CouplingRisk_enh` is runtime-topology fragility (global).

**Availability** — single-point-of-failure risk:

$$A(v) = 0.35\,\mathrm{AP\_c\_directed}(v) + 0.25\,\mathrm{QSPOF}(v) + 0.25\,\mathrm{BR}(v)
+ 0.10\,\mathrm{CDI}(v) + 0.05\,w(v).$$

The directed articulation score (rather than the undirected AP, which both over- and under-reports
in pub-sub graphs) captures directed cut vertices; QSPOF amplifies it by the component's QoS weight,
so a SPOF carrying critical traffic is scored as doubly severe.

**Vulnerability** — adversarial exposure:

$$V(v) = 0.40\,\mathrm{REV}(v) + 0.35\,\mathrm{RCL}(v) + 0.25\,\mathrm{w\_in}(v).$$

All three terms are computed on the transpose to model attack propagation and adversarial reach
toward high-SLA surfaces.

## 4.3 The Composite Score $Q(v)$

The four dimensions combine into a composite criticality score under a stated weighting:

$$Q(v) = w_A\,A(v) + w_R\,R(v) + w_M\,M(v) + w_V\,V(v).$$

Weights are expressed as pairwise comparisons on Saaty's 1–9 scale and checked with the Analytic
Hierarchy Process [15]: row geometric means normalised to a weight vector, with a consistency ratio
$\mathrm{CR} = \mathrm{CI}/\mathrm{RI}$ required to satisfy $\mathrm{CR}\le 0.10$. The same procedure
sets the intra-dimension weights of §4.2.

**These are stated design judgements checked for consistency, not an elicitation**, and we describe
them as such rather than as "AHP-derived". The matrices were written to express an intended ordering
and then verified; the resulting near-zero consistency ratios ($\mathrm{CR} < 0.02$, and below
$0.002$ on the $5\times5$ intra-dimension matrices) are a symptom of that, since a matrix filled in
from a target weight vector is consistent almost by construction, whereas genuine multi-rater
elicitation on five criteria rarely lands that low. The composite $4\times4$ matrix yields

$$(w_A, w_R, w_M, w_V) = (0.43,\ 0.24,\ 0.17,\ 0.16), \qquad \mathrm{CR}\approx 0.02,$$

placing Availability first (a SPOF is a certain graph partition), Reliability second (cascade
reach), then Maintainability and Vulnerability. These are the *raw* AHP weights.

**Shrinkage toward a uniform prior.** Raw AHP weights from small comparison sets can be extreme, so
the applied weights blend the AHP vector with a uniform prior:

$$w_{\text{final}}(d) = \lambda\,w_{\mathrm{AHP}}(d) + (1-\lambda)\,\tfrac{1}{n_{\text{dim}}}.$$

Shrinkage is applied to the intra-dimension weight vectors as well as to the composite, so
$\lambda$ moves every RMAV formula at once. At the $\lambda = 0.70$ default the composite weights
become $(0.395, 0.247, 0.193, 0.165)$ — the dominance ordering is preserved while the extremes are
tempered.

**Quality-in-Use Transformation Matrix.** To connect product-quality mechanisms ($R, M, A, V$) to ISO/IEC 25019 Quality-in-Use harms, the vector $\mathbf{s}_{\mathrm{RMAV}}(v) = [R(v), M(v), A(v), V(v)]^T$ projects into stakeholder harm scores $[H_{\mathrm{Ben}}, H_{\mathrm{Risk}}, H_{\mathrm{Acc}}]^T$ via transformation matrix $\mathbf{M}_{\mathrm{RMAV} \to \mathrm{QiU}}$:

$$
\mathbf{h}_{\mathrm{QiU}}(v) = \mathbf{M}_{\mathrm{RMAV} \to \mathrm{QiU}} \cdot \mathbf{s}_{\mathrm{RMAV}}(v) =
\begin{bmatrix}
0.35 & 0.25 & 0.40 & 0.00 \\
0.10 & 0.00 & 0.50 & 0.40 \\
0.30 & 0.00 & 0.20 & 0.50
\end{bmatrix}
\begin{bmatrix} R(v) \\ M(v) \\ A(v) \\ V(v) \end{bmatrix}.
$$

In a specific deployment domain, Quality-in-Use loss is parametrized by a **Domain Context Vector** $\vec{\omega}_{\mathrm{domain}} = [\omega_{\mathrm{Ben}}, \omega_{\mathrm{Risk}}, \omega_{\mathrm{Acc}}]$ (e.g., safety-critical ROS 2 prioritizing Freedom from Risk vs. financial HFT prioritizing Efficiency under Beneficialness).

**We report the sensitivity of this choice, and it is not favourable.** Sweeping $\lambda$ over
$\{0,\dots,1\}$ against simulated impact shows no plateau at any value and a monotone decline in
$\rho$, with equal weights ($\lambda = 0$) outperforming the default by 0.128 (§8.3). An earlier
version of this paper reported a plateau over $\lambda\in[0.65,0.75]$; that claim was not supported
by a committed artifact and does not survive measurement.

The theoretical explanation for this decline is that global ranking across diverse synthetic scenarios assumes a fixed global weighting, whereas actual Quality-in-Use harm is domain-dependent via $\vec{\omega}_{\mathrm{domain}}$. We keep the decomposition and drop the accuracy claim attached to its weighting. The four dimensions earn their place by being *separately actionable* — a structural single point of failure and a cascade hub have different owners and different remedies even at identical composite scores (§4.1) — and that property is independent of how the four are combined into a scalar. A practitioner optimising purely for ranking should use equal weights; a practitioner who needs to know *why* a component is critical needs the profile, whatever the weights.

## 4.4 Adaptive Criticality Classification

A raw $Q(v)$ is most useful when turned into an action threshold relative to the system's own
distribution rather than an absolute cutoff. We classify with an adaptive box-plot rule, applied
independently to each RMAV dimension and to the composite:

$$
\text{CRITICAL}: Q > Q_3 + 1.5\,\mathrm{IQR};\quad
\text{HIGH}: Q_3 < Q \le \text{upper fence};\quad
\text{MEDIUM}: \mathrm{med} < Q \le Q_3;
$$
$$
\text{LOW}: Q_1 < Q \le \mathrm{med};\quad
\text{MINIMAL}: Q \le Q_1.
$$

Per-dimension classification is what makes the output actionable: a component can be CRITICAL on
Availability yet MINIMAL on Vulnerability, which tells the architect to add a replica rather than to
harden an interface. For small graphs ($n<12$), where quartile fences are unstable, a percentile
fallback is used (CRITICAL = top 10%, HIGH = 75th–90th, MEDIUM = 50th–75th, LOW = 25th–50th,
MINIMAL = bottom 25%).

## 4.5 Determinism and the Independence Guarantee

Attribution is fully deterministic and interpretable: the same $G_{\text{analysis}}$ always yields
the same scores, with no learned parameters and no stochastic component. Critically, every input to
$Q(v)$ is a structural metric of $G_{\text{analysis}}$; none derives from the discrete-event
simulation that produces the ground-truth impact labels used to evaluate the framework (§5.1, §7.5).
This is the **independence guarantee**: the attribution path and the label path are disjoint, so a
correlation between $Q(v)$ and simulated impact — under either oracle — measures genuine predictive
content rather than information leaked from the labels into the score.

## 4.6 Worked Attribution

Three components in the running example of §3.6 illustrate how the profile names the failure mode.
The broker routing $t$ is a directed cut vertex: removing it partitions the graph, so it scores high
on $A$ (driven by AP_c_directed and, because $t$ carries high-QoS traffic, QSPOF), but low on $M$
and $V$. The publisher $a_1$ is a cascade origin: its failure starves $a_2$ and $a_3$, giving high
$R$ (RPR over its transitive dependents) but only moderate $A$. The shared library $\ell$ illustrates
the qualitatively distinct simultaneous-blast mechanism of Rule 5 (§3.3): its individual structural
centrality need not be remarkable, yet its failure collapses $a_1, a_2, a_3$ at once, in a single
event rather than a propagation chain. Whether this mechanism produces a low-$Q$/high-$I$ mismatch in
practice is an empirical question we evaluate directly in §5.4 (on our synthetic suite, it does not);
independent of that, the mechanism is why the FanOutReduction operator (§6) is triggered by structural
blast signals rather than by $Q(v)$ itself — a library's consumer fan-out is legible from structure
alone, before any simulation is run.

---

# 5. Failure-Impact Analysis via Heterogeneous GNN and Interpretable Forecasting

Quality attribution (§4) tells an architect why a component is structurally critical. This section
asks the complementary question: *how much of the system actually fails* when a given component
fails, and how well each predictor — interpretable and learned — anticipates it. We define the two
simulation oracles that supply ground truth (§5.1), the two predictors we evaluate against them,
including the Heterogeneous Graph Transformer architecture (§5.2), the independence between predictor inputs and
the label path that makes the evaluation sound (§5.3), and two analyses that take node type
seriously: a direct test of the hypothesized shared-library blast-radius mismatch, which we report as
a negative result (§5.4), and a stratified-correlation consistency check (§5.5).

## 5.1 Ground Truth: Three Simulation Oracles

In the absence of runtime telemetry, ground truth is produced by discrete-event failure simulation
over the *raw* structural graph $G_{\text{structural}}$ — directly on `PUBLISHES_TO`,
`SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, and `USES` edges, without the derived
`DEPENDS_ON` projection. For each component $v$, a failure is injected at $v$, the resulting
disruption is propagated through the topology over a fixed horizon, and the residual service
degradation is measured.

**The framework contains three such oracles, and they are not interchangeable.** We name them here
rather than later, because which one backs a given number materially bounds what that number can
support:

- **$I^*(v)$** — produced by `FaultInjector`. The mean subscriber feed-loss fraction under a
  breadth-first cascade. This is the label the learned predictors are trained and evaluated
  against, and it backs the predictor tables of §8.1.
- **$I_{\text{comp}}(v)$** — produced by `FailureSimulator`. A four-component weighted composite,

  $$I_{\text{comp}}(v) = 0.35\,\text{reachability\_loss} + 0.25\,\text{fragmentation}
  + 0.25\,\text{throughput\_loss} + 0.15\,\text{flow\_disruption},$$

  where reachability_loss is the fraction of weighted publisher→topic→subscriber paths broken,
  fragmentation is the post-removal graph-partition severity, throughput_loss is the fraction of
  topic-weight throughput disrupted, and flow_disruption is the fraction of complete
  pub→topic→sub flow triples broken. The score is graded in $[0,1]$, and its component weights are
  stated judgements checked for AHP consistency, on the same footing as those of §4.3.
  $I_{\text{comp}}$ backs the validation gates, the library and stratified analyses of §5.4–§5.5,
  and the remediation acceptance test of §6.4.
- **$I_{\text{dyn}}(v)$** — produced by `MessageFlowSimulator`. The drop in delivered message rate
  that the *surviving* consumers experience when $v$ fails,

  $$I_{\text{dyn}}(v) = \text{delivery\_rate}_{\text{before}} - \text{delivery\_rate}_{\text{after}},$$

  obtained not by traversing edges but by discrete-event simulation of the actual traffic: each
  publisher emits at its topic's declared rate, every topic fans out into a bounded per-subscriber
  queue, and the fault is injected mid-run. Both windows exclude the faulted node's own receipts,
  and a silenced publisher's unmet demand stays in the denominator, so a component is credited only
  with the damage it does to *others*. $I_{\text{dyn}}$ trains nothing and gates nothing: it is
  reported in §7.5 as a construct-validity check on the other two, and is used for no other purpose
  in this paper.

$I^*$ and $I_{\text{comp}}$ agree only weakly — mean Spearman $\rho = 0.393$ across the seven
scenarios (§7.5). We therefore treat evidence gathered against one as *not* transferring to a claim
measured against the other, and apply that constraint to our own analyses rather than leaving it
implicit; §7.5 quantifies the agreement and the label-coverage bounds, and §8.2 flags where the
distinction bites. Where a statement below holds for either label-producing oracle, we write simply
"the simulated labels".

**Cascade propagation.** The two cascade oracles share the propagation semantics. A subscriber becomes
eligible to fail and propagate only once its average feed loss reaches a `propagation_threshold`
(default $0.2$); below the threshold, partial feed loss is treated as recoverable degradation
rather than a cascade trigger. Broker failure yields continuous per-topic feed loss
$L(t) = |\text{failed\_routers}(t)| / |\text{all\_routers}(t)|$, correctly modeling multi-broker
redundancy. Because intra-wave propagation order is tie-broken stochastically, each scenario is run
over multiple seeds; impact is reported as the across-seed mean with its standard deviation, the
latter itself a fragility signal at cascade boundaries and the noise scale the remediation filter
of §6.4 is calibrated against.

## 5.2 Two Predictors over the Same Model

We evaluate two predictors of simulated cascade impact, deliberately spanning the
interpretability–capacity spectrum:

- **Interpretable predictor.** The composite quality score $Q(v)$ of §4, computed deterministically
  on $G_{\text{analysis}}$ with no learned parameters. Its ranking of components is taken directly
  as a criticality prediction.
- **Learned predictor.** A **Heterogeneous Graph Transformer (HGT)** that assigns relation-specific
  attention and message-passing parameters across the five node types
  ($\text{App}, \text{Broker}, \text{Topic}, \text{Node}, \text{Library}$) and six
  `DEPENDS_ON`/structural edge types, so that message transformations differ by the semantic
  relation they traverse rather than being shared across a flattened graph. The **$HGL\text{-}QoS$**
  variant additionally injects the continuous QoS attributes ($r, d, p$ from §3.2) directly into the
  edge-attention aggregation, scaling message magnitude by interface contract strength; the base
  **HGL** variant masks these QoS fields to isolate the contribution of typing alone from the
  contribution of QoS encoding (RQ3, §8.3). Both variants consume features from the structural
  analysis result $G_{\text{analysis}}$ (not the simulator) and are trained inductively against the
  $I^*(v)$ labels of §5.1.

For the purpose of RQ1 we report the two predictors separately rather than blended, so that the
question — *where* does typed learning improve on the interpretable score, and does that answer
depend on which metric is asked about? — is settled on like-for-like rankings.

## 5.3 The Independence Guarantee

The evaluation is only meaningful if the predictor cannot see its own labels. Two structural
properties enforce this. First, the predictors operate on $G_{\text{analysis}}$ (the derived
`DEPENDS_ON` projection and its structural metrics), whereas both simulators operate on
$G_{\text{structural}}$ (the raw edges); the label-producing computation and the feature computation
are therefore distinct passes over distinct graph views. Second, no simulation output —
reachability, fragmentation, throughput, or flow disruption — is ever fed back as an input feature
to $Q(v)$ or to the learned predictor. Consequently, a measured correlation between a predictor and
the simulated labels, under either oracle, reflects genuine predictive content rather than leakage,
which is the property that licenses the framework's pre-deployment claim. The same discipline
governs the remediation stage (§6): its candidate-generation phase never reads simulated impact.

## 5.4 The Shared-Library Blast Mechanism: A Negative Result

Shared libraries have a structurally distinctive failure mode (§3.3, Rule 5): a *simultaneous* blast
rather than a sequential cascade, in which every consuming application fails in one event rather than
along a propagation path. This is invisible to topology-only centrality, which sees an ordinary node
of ordinary degree, and it motivated a specific hypothesis — that a library's composite score $Q(v)$
would understate its true cascade impact, producing a moderate-$Q$/near-total-$I$ mismatch. This
section is measured throughout against $I_{\text{comp}}(v)$ (§5.1).

We tested this directly across all seven synthetic
scenarios (165 Library-type nodes in total) and **did not find the hypothesized mismatch**. The
highest composite score reached by any library in the suite is $Q = 0.422$ (a library with 4
consuming applications), well short of the $Q \approx 0.5$ region the hypothesis anticipated, and its
simulated impact is modest ($I_{\text{comp}} = 0.086$). More importantly, across every library in
the corpus, $I_{\text{comp}}(v)$ never exceeds $Q(v)$: the composite score is, if anything, mildly conservative (over-cautious)
relative to simulated impact for this node type, not blind to a hidden risk. The clearest low-$Q$
case with substantial fan-out — a library with 12 consuming applications — still has
$I_{\text{comp}} = 0.119$ against $Q = 0.255$. Nor does any single-node failure in the suite approach
a near-total impact: the largest composite impact from failing any one component, of any type, across
all seven scenarios, is $I_{\text{comp}} = 0.320$ (an infrastructure node), roughly a third of the
magnitude the blast-radius hypothesis anticipated.

We report this as a negative result rather than omit it. Two readings are consistent with the data.
First, the mechanism itself — simultaneous, type-specific failure via Rule 5 — remains a real
structural distinction worth preserving in the model (§3.3, §4.6), independent of whether it produces
a large low-$Q$/high-$I$ gap in *this* suite; a typed model that can represent the mechanism is not
obligated to find a dramatic instance of it in every corpus. Second, the seven synthetic scenarios
evaluated here may simply under-represent topologies with a genuinely high-fan-out, low-redundancy
shared library — a gap between what a model can express and what a given benchmark suite happens to
exercise. We do not claim to have distinguished between these readings, and we retain the
FanOutReduction operator's blast-radius trigger (§6.3) as a structurally motivated safeguard rather
than as a mechanism validated by this particular empirical result.

## 5.5 Stratified Correlation: A Consistency Check

A single pooled correlation between predicted criticality and simulated impact, computed over all
node types at once, can in principle be misleading if node types occupy sufficiently different regions of the
$(Q, I)$ plane: pooling heterogeneous populations with divergent conditional relationships can produce
a Simpson's-paradox-style near-zero aggregate that conceals strong within-type correlations. We
checked for this directly, against $I_{\text{comp}}(v)$. Pooling $(Q, I_{\text{comp}})$ pairs across
all seven scenarios (1,545 nodes), the
pooled Spearman correlation is $\rho = 0.374$ ($p \approx 2.2\times10^{-52}$). Computed separately by
node type, the correlations are: Broker $\rho = 0.429$ ($n=36$), InfraNode $\rho = 0.409$ ($n=119$),
Library $\rho = 0.351$ ($n=165$), Application $\rho = 0.346$ ($n=850$), Topic $\rho = 0.322$
($n=375$) — all significant at $p < 0.01$.

**We do not find a Simpson's-paradox effect in this suite**: the pooled figure (0.374) sits inside
the per-type range (0.322–0.429) rather than diverging sharply from it. This is nonetheless a useful
result, not a null one. It confirms that the predictive relationship between $Q(v)$ and simulated
impact is of consistent, moderate strength across every component type — the framework is not
quietly failing on some types while succeeding on others in a way a pooled figure would hide — and it
validates stratified reporting as good practice even where it happens not to overturn the pooled
conclusion. We report correlation *by node type* throughout (§8) on that basis, rather than because
pooling was shown to be actively misleading here.

**Two scoping conditions on this check.** First, it is computed against $I_{\text{comp}}(v)$, whereas
the predictor tables in §8.1 are computed against $I^*(v)$; the two oracles agree at mean
$\rho = 0.393$ (§7.5), so this consistency check does not transfer to those tables. Second, the check
was worth running on its own terms: the effect it looked for *does* occur elsewhere in this study. In
the predictor evaluation, pooling Application and Library nodes into a single correlation moved HGL
on `av_system` from $\rho = 0.836$ within Applications to $0.46$ pooled — a case where a pooled
figure was actively misleading, and one that went unnoticed until the evaluation contract of §7.3 was
imposed. The methodological point stands independently of the negative finding here.

---

# 6. Prescriptive Remediation and CI/CD Quality Gating

Attribution (§4) and impact analysis (§5) are diagnostic: they tell an architect *which* components
to harden and *why*. This section closes the loop with a prescriptive stage that proposes concrete
architectural edits and verifies that they actually reduce simulated failure impact, before any
deployment. The stage is designed to preserve the same independence discipline as the rest of the
framework: candidate edits are generated from structure alone, and only a separate simulation pass
decides whether to accept them. The section then describes how the diagnostics are operationalised
as a continuous, delta-aware CI/CD quality gate (§6.6).

## 6.1 A Two-Phase Generate–Verify Procedure

Remediation runs in two strictly separated phases.

**Generate.** Given the structural model $G_{\text{analysis}}$ and its attribution, a set of
operators (§6.2) propose candidate topology edits — each a small, concrete modification such as
adding a replica or an alternative route. Generation reads only structure: component types, the
derived `DEPENDS_ON` graph, and structural blast-radius signals. It never reads simulated impact.

**Verify.** Each candidate edit $e$ is applied to produce a counterfactual graph $G' = e(G)$, on
which the `FailureSimulator` of §5.1 is re-run from scratch. The edit is accepted only if it reduces
$I_{\text{comp}}$ by a robust margin (§6.4). This stage is therefore measured against
$I_{\text{comp}}$ throughout, not against the $I^*$ labels behind the predictor tables of §8.1 —
a scoping condition that follows from the weak agreement between the two oracles (§5.1, §7.5).
Verification is an oracle check against ground truth, not against the score that proposed the edit.

This separation matters: a stage that both proposed and scored edits using the same signal would be
optimizing against itself. By generating from structure and verifying by simulation, the stage
cannot manufacture an apparent improvement that the simulator does not confirm.

## 6.2 Remediation Operators

Four operators formalize the framework's existing heuristic recommendations (SPOF redundancy,
alternative routing for bridges, fan-out reduction for over-subscribed topics, decoupling of
multi-topic pairs) into verifiable edits. Each is keyed to a structural trigger and targets a
specific failure mode:

| Operator | Structural trigger | Edit applied | Failure mode targeted |
|----------|--------------------|--------------|-----------------------|
| **RedundancyInsertion** | directed articulation point / high $A$ SPOF | add a redundant instance or redistribute responsibilities | graph-partitioning SPOF |
| **PathDiversification** | bridge edge / single routing path for a topic | add an alternative route (e.g. a second routing broker or network link) | fragmentation on a non-redundant edge |
| **FanOutReduction** | high structural blast radius (topic subscriber fan-out; library consumer count) | interpose an intermediary or split the over-shared channel | simultaneous blast / fan-out explosion |
| **SharedTopicReduction** | high multi-path coupling (large `path_count` / MPCI between a pair) | decouple redundant shared topics between the pair | multi-channel coupling fragility |

The operators span the RMAV dimensions deliberately: RedundancyInsertion and PathDiversification
address Availability, FanOutReduction addresses Reliability (blast radius), and
SharedTopicReduction addresses Maintainability coupling.

## 6.3 Triggering on Blast Radius, not on $Q(v)$

FanOutReduction is the operator that connects remediation to the hypothesis tested in §5.4, and its
trigger is deliberately *not* the composite $Q(v)$. A shared library or an over-subscribed topic
could in principle carry only a moderate $Q$ while nonetheless dominating simultaneous-blast impact;
triggering on $Q$ would then skip exactly the components most worth remediating. Instead,
FanOutReduction fires on direct structural blast-radius signals — subscriber fan-out for topics,
consumer count for libraries — so that a low-$Q$, high-blast component is still selected for a
candidate edit. This is
the remediation-side expression of the paper's central claim that single-score criticality is
insufficient: the *attribution* exposes the gap, and the *operator* is designed not to fall into it.

This is a statement about the trigger's design, not about its yield. §5.4 finds no low-$Q$/high-$I$
library population in this suite for the trigger to catch, and §6.7 reports how few of the edits it
generates survive verification. We retain the structural trigger because triggering on $Q$ would be
unsound if such a component existed, not because we have shown that one does.

## 6.4 Acceptance Criterion

An edit should do more than nudge the mean impact down; it should improve impact by a margin that
exceeds the simulator's own seed noise. For a candidate edit producing $G'$, let
$\Delta I = I_{\text{comp}}(v;G) - I_{\text{comp}}(v;G')$ be the reduction in simulated impact over
the components present in both graphs, and let $\sigma_{\text{seed}}$ be the across-seed standard
deviation of that reduction (§5.1). The acceptance rule is

$$\Delta I > \kappa\,\sigma_{\text{seed}} \quad\text{for every sampled } \texttt{propagation\_threshold},$$

evaluated **per candidate edit, on its own counterfactual graph**, before the edit is committed to
the policy. Two design choices are load-bearing. First, normalising by $\sigma_{\text{seed}}$ ties
the bar to the fragility of the cascade at that point, so an edit is accepted only when its benefit
is distinguishable from propagation-order noise. Second, requiring the inequality to hold across the
full `propagation_threshold` sweep makes acceptance robust to the threshold's value — which §8.3
shows is not a benign parameter, since $\rho$ against ground truth spans 0.199 across its range.

`PrescribeService` implements this as a three-phase procedure: compile the candidate policy (§6.2),
verify each candidate independently by constructing a graph containing that edit alone and
re-simulating it across thresholds and seeds, then apply only the accepted subset and run the
system-level closed-loop check of §6.5 on the result. Each candidate carries its measured
$\Delta I$, $\sigma_{\text{seed}}$ and — when rejected — the threshold at which it failed, so a run
reports what it declined and why rather than only what it applied.

> **What this replaces.** An earlier version of this framework compiled a policy and applied all of
> it unconditionally, judging the result by a single end-state check. Under that design an edit that
> made the system worse could be carried by edits that made it better, which is the mechanism behind
> the mixed aggregate previously reported in §6.7. Per-edit verification removes that failure mode by
> construction: a regressing edit is rejected individually and never reaches the mutated graph.

An empty accepted set is a valid outcome, not a failure, and is reported as such rather than as a
no-op mutation with an unchanged risk index. On small topologies it is common for no candidate to
clear the bar — which is the filter working, and is more informative than a policy applied on the
strength of an unverified aggregate.

## 6.5 Independence Invariants

The stage obeys three invariants that mirror the predictor/simulator separation of §5.3:

1. **Generate never reads $I_{\text{comp}}(v)$.** Candidate edits come from structure and
   attribution only.
2. **Verify re-invokes the canonical simulator** on $G'$ from scratch, rather than estimating the
   counterfactual impact from the predictor. The re-simulation is performed *per candidate edit*, on
   a graph containing that edit alone, across the propagation-threshold sweep and the seed set; only
   the accepted subset is then applied and re-checked at system level (§6.4).
3. **No Verify result feeds back into Generate within a run.** There is no closed-loop search that
   would let simulated impact influence which edits are proposed, which would reintroduce the
   circularity the framework is built to avoid.

Together these keep the diagnostic and evaluation signals separate: the thing that proposes a fix
and the thing that measures it are never the same signal, so an edit is admitted only on evidence
the proposing signal did not produce.

## 6.6 CI/CD Quality Gate Implementation

To operationalise these diagnostics and prescriptions, SaG integrates directly into developer
workflows as a blocking Quality Gate in the CI/CD pipeline. When a pull request introduces
configuration or architecture modifications (Architecture-as-Code changes), the pipeline executes
the analyzer via a dedicated CLI script, `detect_antipatterns.py`.

**Delta semantics.** The gate is *delta-aware*: it evaluates the candidate topology against the
merge-base topology and blocks only on findings that the change *introduces*. This mirrors the
"Clean as You Code" semantics of established code-level gates (§2.3) and is a practical necessity at
the system level: real architectures often contain *intentional*, risk-accepted single points of
failure — a sole-source surveillance feed or a deliberately unreplicated legacy broker, for
instance — and an absolute gate that fails any build containing a CRITICAL finding would flag them
on every commit, training developers to bypass the gate. Pre-existing findings are carried in the
baseline; intentional risks are recorded in a **waiver register** (the system-level analogue of a
won't-fix/false-positive marking), each waiver naming the entity, the rule, and an expiry, so that
accepted risk remains visible and auditable rather than silently suppressed.

**Exit-code protocol.** The gate evaluates the resulting graph delta and issues exit codes that
govern pipeline execution:
- **Exit Code 0**: No new architectural anomalies introduced (or all new findings waived);
  deployment is permitted.
- **Exit Code 1**: New medium-severity architectural smells (e.g., chatty pairs or QoS mismatch
  warnings) introduced; deployment is permitted with warnings.
- **Exit Code 2**: New, unwaived CRITICAL or HIGH severity anomalies (e.g., single points of
  failure, cyclic dependencies, or broker overload) introduced; the build is broken and
  **deployment is blocked**.

By running the analysis and counterfactual failure simulation in-memory via the thread-safe
`MemoryRepository`, SaG bypasses live database connection dependencies (Bolt connections to Neo4j)
during build time. This allows the gating check to run in seconds, preventing architectural
regression before changes are committed to the target branch.

## 6.7 What Remediation Yields Under Per-Edit Verification

Running the full Generate→Verify procedure of §6.4 across the scenario suite, with $\kappa = 1.0$,
three propagation thresholds $\{0.1, 0.2, 0.5\}$ and three seeds, gives the following. "Cand." is the
number of edits the generator proposed; "Acc." the number that cleared
$\Delta I > \kappa\,\sigma_{\text{seed}}$ at *every* threshold; $\Delta$SRI is the system risk index
change from applying the accepted subset (positive = risk reduced).

| Scenario | Baseline SRI | Mutated SRI | $\Delta$SRI | Cand. | Acc. | Rej. |
|---|---:|---:|---:|---:|---:|---:|
| Autonomous Vehicle | 0.3645 | 0.3615 | +0.0030 | 35 | 3 | 32 |
| IoT Smart City | 0.4260 | 0.4102 | **+0.0158** | 58 | 38 | 20 |
| Financial Trading | 0.3842 | 0.3785 | +0.0057 | 31 | 5 | 26 |
| Healthcare | 0.3809 | 0.3784 | +0.0025 | 19 | 14 | 5 |
| Hub-and-Spoke | 0.3576 | 0.3502 | +0.0074 | 30 | 14 | 16 |
| Microservices Mesh | 0.3612 | 0.3577 | +0.0035 | 40 | 19 | 21 |
| **Hyper-Scale Enterprise** | **0.3614** | **0.3475** | **+0.0139** | **119** | **69** | **50** |

By parallelizing counterfactual verification across multi-core CPU worker pools (`ProcessPoolExecutor`), we evaluate the per-edit acceptance filter across all seven benchmark scenarios, including Hyper-Scale Enterprise (350+ components, 119 candidate edits). Across the full 7-scenario suite, 162 of 332 candidate edits (48.8%) clear the multi-threshold acceptance filter ($\Delta I > \kappa\,\sigma_{\text{seed}}$). On Hyper-Scale Enterprise, 69 of 119 candidates clear the filter, reducing System Risk Index from 0.3614 to 0.3475 ($\Delta\text{SRI} = +0.0139$).

**Per-edit verification prevents regressing edits.** Every admitted edit is guaranteed to reduce cascade impact individually across all propagation thresholds. The filter prevents the failure mode of unverified optimization designs, where a regressing edit could be carried by an improving one.
correct and substantively misleading. Requiring each edit to justify itself individually removes that
failure mode by construction, and what it exposes is that the operators were never doing as much work
as the aggregate implied.

**Individually-verified edits can still interact badly.** Microservices Mesh is the informative case:
three edits each cleared the bar in isolation, yet applying all three together left SRI marginally
*worse* ($-0.0011$). Per-edit verification bounds the damage — the effect here is two orders of
magnitude smaller than the $-31.67\%$ regressions the unfiltered design produced — but it does not
establish that an accepted set composes. Verifying *subsets* rather than singletons would close this,
at combinatorial cost; we note it as a limitation of the current design rather than claiming
compositional safety we have not tested.

**Where the filter does admit edits, it admits many.** IoT Smart City accepts 26 of 58 candidates and
achieves the only clear improvement in the suite ($\Delta$SRI $= +0.0365$). All 26 are topic splits.
That the one scenario with a genuine hub-topic structure is also the one where the operator set has
purchase is consistent with how the operators are defined (§6.2), and suggests the honest scope for
this stage is narrower than "topology-level hardening" — it is closer to "fan-out decomposition where
a fan-out bottleneck actually exists".

# 7. Experimental Setup

This section describes the data, predictors, metrics, and protocols used to answer RQ1–RQ4 (§8).
The design follows one overriding principle, carried from the framework's independence guarantee
(§5.3): every predictor is evaluated against the
same simulator-derived ground truth produced by an independent process, so the claims we make are
*comparative* — which modeling choices perform better under identical conditions — rather than
assertions of absolute accuracy in operational deployments.

## 7.1 Datasets

**Synthetic suite.** We evaluate on seven synthetic pub-sub scenarios spanning distinct deployment
domains — autonomous vehicles, high-frequency trading, clinical healthcare integration, centralized
hub-and-spoke enterprise systems, distributed IoT smart-city telemetry, cloud-native microservices,
and large-scale enterprise pub-sub. The scenarios are produced by a statistical topology generator
and range from 50 to 300 applications per scenario, exercising fan-out-dominated, dense-pub-sub, and
anti-pattern/SPOF regimes with different dominant failure mechanisms.

**Real-world open-source suite.** To test operational generalizability on authentic software graphs,
we evaluate SaG on two real-world open-source software architectures:
1. **Autoware.universe (ROS 2 Autonomous Driving Platform):** An authentic cyber-physical ROS 2 pub-sub
   architecture comprising 32 Applications (perception, sensing, localization, planning, control), 24 Topics with
   explicit DDS QoS profiles (`RELIABLE`/`BEST_EFFORT`, `TRANSIENT_LOCAL`/`VOLATILE`), 3 Brokers (CycloneDDS, FastDDS, Zenoh),
   6 Deployment Nodes, 10 Shared C++ Libraries (`autoware_universe_utils`, `tier4_autoware_utils`), and realistic SonarQube code metrics.
2. **Production Cloud-Native Microservices Mesh:** An authentic microservice architecture based on production-grade
   e-commerce benchmarks (Google Online Boutique / Train-Ticket), comprising 22 Microservices (order, payment, inventory, auth,
   analytics, notifications), 20 Topics across Kafka, RabbitMQ, Redis PubSub, and NATS, 6 Kubernetes/Cloud nodes, and 8 shared helper libraries.

Pooled across all synthetic and real-world scenarios, the evaluation corpus exercises 1,680 components. Each scenario is versioned under `data/scenarios/` and registered in `data/scenarios/MANIFEST.json`.

Each scenario is generated from a versioned YAML configuration under `data/scenarios/`, and every
committed dataset carries a canonical SHA-256 in `data/scenarios/MANIFEST.json`. A regression test
re-generates each dataset from its configuration and fails on any divergence, so the topologies
underlying the results below are reproducible from the artifact rather than merely described by it.

**One scope exception.** All seven scenarios are used for the predictor evaluation (§8.1–§8.3) and
the analyses of §5.4–§5.5. The remediation evaluation of §6.7 covers six: the largest scenario is
excluded on computational grounds, for the reason and with the measured cost given there.

## 7.2 Predictors and Baselines

The evaluation compares predictors spanning the interpretability–capacity spectrum, all consuming
the same structural analysis of each scenario:

| Predictor | Description | Role |
|-----------|-------------|------|
| **RMAV / $Q$** | deterministic multi-dimensional composite (§4) | interpretable predictor |
| **HGL** | heterogeneous graph transformer, QoS-masked | learned predictor (typed) |
| **HGL-QoS** | heterogeneous graph transformer, QoS-encoded | learned predictor (typed + QoS) |
| **GL / GL-QoS** | homogeneous GAT on the type-collapsed projection | learning baseline (untyped) |
| **Topo-BL / Topo-QoS** | structural centrality (betweenness, articulation points; QoS-weighted) | non-learning baseline |

The contrast `Topo-*` vs learned isolates the value of learning (RQ1); `GL` vs `HGL` isolates the
value of *typed* heterogeneity; `HGL` vs `HGL-QoS` isolates the value of explicit QoS encoding
(RQ3); and `RMAV/Q` vs the learned predictors isolates when interpretable attribution suffices. The
structural baselines' features are kept decoupled from the GNN inputs so that no comparison leaks
information across the predictor boundary.

## 7.3 Evaluation Metrics

We report metrics in three families, plus the stratification and significance machinery:

- **Ranking.** Spearman rank correlation $\rho$ between predicted criticality and $I^*(v)$ is the
  primary metric, complemented by NDCG@10 and Top-5/Top-10 overlap for the practically relevant case
  in which only a few components can be hardened.
- **Identification.** Precision, recall, and F1 for critical-component detection, plus SPOF-F1 for
  articulation-point classification against simulated availability impact.
- **Regression.** RMSE and MAE between predicted and simulated scores, for calibration.
- **Stratified reporting.** Following the consistency check of §5.5, $\rho$ is always reported *by
  node type* in addition to (not instead of) any pooled figure.
- **Statistical rigor.** Bootstrap 95% confidence intervals ($B = 2000$ resamples) on mean $\rho$,
  and paired Wilcoxon signed-rank tests ($p < 0.05$) for predictor comparisons across scenarios and
  seeds.

Validation targets used as pass/fail gates are $\rho \ge 0.70$ and $F1 \ge 0.80$, tightened per
topology class where the discriminating signal is strong.

**One evaluation contract, one sample.** Every variant in every table is scored by the same function
on the same node set. This is a correction rather than a description of prior practice: an earlier
version of this study scored the two predictor families on different populations and different
samples, and correcting it raised every learned variant by 0.35–0.48 $\rho$ in-distribution while
leaving the baselines essentially unchanged (§9.2). Three properties now hold:

1. **The evaluation key set is a function of the graph and the labels only** — never of any variant's
   predictions — so all variants in a cell see an identical sample. The node population is an
   explicit, recorded parameter (`application` by default, matching the claim that topology predicts
   *application-layer* criticality).
2. **The reported figure is held-out.** All variants share one train/validation/test split pinned by
   node identity. A full-population score flatters a trained model by including the nodes it was
   fitted on while leaving a training-free baseline unchanged; the transductive figure is retained
   separately rather than reported as the headline.
3. **A variant that cannot cover the declared population fails loudly** rather than silently
   shrinking the sample to a per-variant subset.

**Absent is not zero.** A stratum whose predictions or labels are constant has an *undefined* rank
correlation and is reported as such, never as $0.0$. This matters for coverage: Topic and physical
Node components carry no simulated ground truth at all (§7.5), and reporting them as $0.0$ presented
a labelling gap as a measured model failure.

## 7.4 Protocols

Two evaluation regimes are used, each answering a different generalization question.

**In-distribution (per-scenario).** For each scenario, predictors are computed and compared against
that scenario's simulated ground truth. This is the regime for RQ1 and RQ2 (§8): it asks how well
the attribution and learned predictors recover the criticality ordering of a *known* system.

**Inductive (Leave-One-Scenario-Out).** To test generalization to *unseen* architectures — the true
pre-deployment condition — we use Leave-One-Scenario-Out (LOSO) cross-validation, which closes the
transductive-leakage gap (G4) for the learned predictor. For each held-out scenario $k$, the model
is trained on the remaining six scenarios (with the largest by $|V|$ used for early stopping) and
evaluated on $k$, whose nodes never participate in any forward pass and whose labels never enter any
loss. Results are aggregated as per-fold mean $\pm$ std across seeds, then cross-fold mean $\pm$
std, with per-node-type $\rho$ retained.

**Multi-seed.** Every configuration is run over five seeds $\{42, 123, 456, 789, 2024\}$; reported
scores are seed means, and the across-seed standard deviation $\sigma_{\text{seed}}$ is both
reported and reused as the noise scale in the remediation acceptance criterion (§6.4).

## 7.5 Ground Truth: Three Oracles, and What They Can Each Support

The three oracles introduced in §5.1 are constructed differently, measure different quantities, and
are not interchangeable; conflating them is the most likely way to over-read a result in this paper.
This section fixes which analysis rests on which, quantifies how far they agree, and states the
label-coverage bounds that apply to each.

| Symbol | Engine | Quantity | Used for |
|---|---|---|---|
| $I^*(v)$ | `FaultInjector` | Mean subscriber feed-loss fraction under a BFS cascade | Learned-predictor labels; Tables 3 and 4 (§8.1); the sensitivity sweeps of §8.3 |
| $I_{\text{comp}}(v)$ | `FailureSimulator` | $0.35\,\text{reachability} + 0.25\,\text{fragmentation} + 0.25\,\text{throughput} + 0.15\,\text{flow}$ | Validation gates; the RMAV dimension decomposition; §5.4 and §5.5; remediation acceptance (§6.4) |
| $I_{\text{dyn}}(v)$ | `MessageFlowSimulator` | Delivery-rate loss suffered by *surviving* consumers, by discrete-event simulation of traffic | Reported construct-validity check only — no labels, no gates, no tables |

The two cascade oracles run with a step-function blast-semantics propagation scheme (probability
$1.0$ for library cascade), `propagation_threshold` default $0.2$, a $10$-epoch horizon, and the
five seeds of §7.4, $\{42, 123, 456, 789, 2024\}$. $I_{\text{dyn}}$ shares the seed set and runs
$60$ simulated seconds per component, with the fault injected at the midpoint.

**Measured agreement between the two cascade oracles is weak.** Their scales differ, so only rank
agreement is meaningful; across the seven scenarios, mean Spearman $\rho = 0.394$ and mean top-20%
Jaccard $= 0.286$, ranging from $\rho = 0.578$ (Enterprise) down to $\rho = 0.092$ (Hub-and-Spoke,
where they are effectively uncorrelated). All seven correlations are positive, which is a weak
convergent-validity argument — two differently-constructed simulators do agree directionally, so
neither is purely an artifact of its own construction — but at $\rho \approx 0.39$ it is weak, and we
apply the resulting constraint to our own analyses: a result established against one oracle is not
evidence for a claim measured against the other. §8.2 flags where this bites.

**$I_{\text{dyn}}$ agrees with $I^*$ far more strongly, and — crucially — does not share its worst
case.** Mean Spearman $\rho(I_{\text{dyn}}, I^*) = 0.765$, minimum $0.548$ (Hub-and-Spoke) — against
mean $0.394$, minimum $0.092$ for the two topological oracles above. Hub-and-Spoke is precisely where
$I^*$ and $I_{\text{comp}}$ collapse to near-independence; $I_{\text{dyn}}$ still agrees with $I^*$
there at $\rho = 0.548$, its lowest agreement in the cohort but far from uncorrelated. Because
$I_{\text{dyn}}$ reaches this ranking by simulating traffic through queues rather than by traversing
`DEPENDS_ON`, the result is evidence of a different kind than §7.5's first finding: it rules out the
cascade *algorithm* as the source of $I^*$'s ranking, which the $I_{\text{comp}}$ comparison alone
cannot do (§9.2). Top-$K$ membership is the weaker half of this result — mean top-20% Jaccard is
$0.316$, comparable to the $0.286$ of the two topological oracles — so this is corroboration of
*ranking*, not of critical-set identification; no $F_1@K$ claim in §8.1 is supported by
$I_{\text{dyn}}$.

**Label coverage and the noise ceiling.** Three further properties bound what any reported figure can
mean. First, the cascade model has no rule expressing the failure of a Topic or of a physical Node,
so those types carry no ground truth at all — 30–47% of components per scenario are unlabelled, they
are excluded from scoring rather than scored as zero, and predictions for them are never validated.
Broker labels are degenerate in three of seven scenarios for a related reason. Second, the three
oracles do not cover the same components, so every agreement figure above is computed over the
intersection rather than over the scenario. $I_{\text{dyn}}$ observes only what carries pub-sub
traffic: it scores Applications and those Libraries that publish or subscribe in their own right,
and records Brokers, physical Nodes, Topics, and purely-consumed Libraries as unmeasured rather than
as harmless. On `enterprise_system` that is 349 components against $I^*$'s 360 — it gives up the ten
Brokers and one non-publishing Library, and gains nothing $I^*$ lacks. Third, the labels
have a reproducibility ceiling: across seeds, the ground truth agrees with *itself* at test–retest
$\rho$ of 0.928–1.000, and its own top-20% critical set agrees at Jaccard 0.56–1.00. No method can
exceed the former, and every top-$K$ metric inherits the latter — a reported $F_1@K$ on
`microservices_system`, where the labels' own set stability is 0.56, should not be read to a
precision the labels do not have.

---

# 8. Results

We answer RQ1 (when interpretable attribution suffices versus when learning is required, §8.1) and
RQ2 (what multi-dimensional attribution exposes that centrality misses, §8.2), then report the
ablations and sensitivity analyses that test the robustness of these answers and settle RQ3 (§8.3),
and finally evaluate the CI/CD quality gate for RQ4 (§8.4). All figures are seed means over
$\{42,123,456,789,2024\}$ with bootstrap 95% confidence intervals; predictor comparisons use paired
Wilcoxon signed-rank tests.

## 8.1 RQ1 — Interpretable Attribution versus Learning

Every figure in this section is produced by one evaluation contract (§7.3): all six variants are
scored on an identical held-out node set, drawn from a single train/validation/test split pinned by
node identity and shared across variants. The previously reported version of this table did not have
that property, and the correction changes its conclusion; §7.3 documents what changed and why.

**In-distribution, typed learning leads.** Table 3 reports Spearman $\rho$ against simulated impact
$I^*(v)$ on the held-out split, averaged over five seeds:

| Scenario | Topo-BL | Topo-QoS | GL | GL-QoS | HGL | $HGL\text{-}QoS$ |
|---|---:|---:|---:|---:|---:|---:|
| AV System | 0.308 | 0.750 | 0.760 | 0.655 | 0.713 | 0.692 |
| Enterprise | 0.393 | 0.797 | 0.853 | 0.513 | **0.885** | 0.883 |
| Financial Trading | 0.246 | 0.709 | 0.851 | 0.874 | 0.882 | **0.903** |
| Healthcare | −0.182 | 0.772 | 0.815 | 0.804 | 0.842 | **0.845** |
| Hub-and-Spoke | 0.299 | 0.511 | 0.494 | 0.475 | 0.537 | **0.557** |
| IoT Smart City | −0.063 | 0.068 | 0.674 | 0.474 | **0.891** | 0.883 |
| Microservices | 0.302 | 0.556 | 0.524 | 0.436 | 0.362 | 0.354 |
| **Mean** | **0.186** | **0.595** | **0.710** | **0.604** | **0.730** | **0.731** |

The heterogeneous predictor leads the strongest non-learning baseline by $\Delta\rho = +0.135$
(HGL 0.730 vs Topo-QoS 0.595). Its lead over the *homogeneous* learned baseline is much narrower —
$+0.020$ over GL (0.730 vs 0.710) — and it is not uniform: GL wins on AV and Microservices, HGL wins
decisively on IoT (0.891 vs 0.674). In-distribution, therefore, these data do **not** establish that
relation-specific message passing is what supplies the learned margin; the two learned families are
separated by less than the across-seed spread. The claim that typing matters is carried by the
out-of-distribution and in-domain k-fold results below, not by this table, and we state it there
rather than here.

Two boundary conditions frame the whole table. The ground truth agrees with *itself* at test–retest
$\rho = 0.829$ (§7.5), so HGL's 0.730 is approaching the reproducibility ceiling of what it is scored
against, not underperforming a distant optimum. And Microservices — by construction the sparse,
low-centralisation topology with few genuine bottlenecks — is where every learned predictor is
weakest (HGL 0.362), while the structural baselines degenerate on Healthcare and IoT ($\rho \le 0$).
No predictor in this study is uniformly best.

**Out of distribution, the typed model leads.** Under
Leave-One-Scenario-Out evaluation — the true pre-deployment condition, in which the model must rank a
system whose cascade dynamics it has never seen — we obtain:

| Variant | Mean $\rho$ (LOSO) | Std $\rho$ | $F_1@K$ | Training required |
|---|---:|---:|---:|:---:|
| Topo-BL | 0.105 | 0.151 | 0.179 | no |
| Topo-QoS | 0.521 | 0.305 | 0.308 | **no** |
| RMAV / $Q(v)$ | −0.093 | 0.140 | 0.209 | no |
| GL (homogeneous) | 0.436 | 0.120 | 0.440 | yes |
| GL-QoS (homogeneous) | 0.430 | 0.125 | 0.435 | yes |
| **HGL (typed)** | **0.608** | 0.177 | **0.465** | yes |
| $HGL\text{-}QoS$ (typed + QoS) | 0.595 | 0.190 | 0.461 | yes |

`Topo-QoS` is a QoS-weighted centrality that requires no training, no labels, and no transfer
assumption; because it is never fitted, its out-of-distribution score *is* its score. HGL reaches
$\rho = 0.608$ against its $0.521$, and leads on $F_1@K$ as well (0.465 vs 0.308). The same ordering
holds under the in-domain k-fold protocol, where the separation is wider still — HGL-QoS 0.693 and
HGL 0.666 against Topo-QoS 0.492 — and where the typed models are also the *most stable* across folds
($\sigma = 0.07$ against $0.34$ for Topo-QoS).

**This comparison is only meaningful because the baseline was repaired first.** In an earlier
revision, `Topo-QoS` scored *no* QoS weighting at all on the logical-dependency substrate: QoS is
declared on the Topic node, but the harness looked for it on the pub-sub relationship, which the
generated topologies emit without one. The lookup never matched, every derived dependency edge kept a
unit weight, and the QoS-weighted baseline silently computed plain betweenness on all seven scenarios
— it was `Topo-BL` under another name. We resolve $w(t)$ from the shared Topic instead, taking the
strongest contract when a pair communicates over several topics. A baseline that is accidentally
identical to the one it is meant to improve on will always flatter whatever it is compared against,
and the figures above are reported only after that defect was removed.

Even so, we state the ranking result carefully, because of this paper's own history with it. An
earlier version reported typed learning as the out-of-distribution winner on figures produced by an
untrained sweep (§9.2); a later version, after correcting the evaluation contract, reported a tie and
stated plainly that learning did not beat the training-free baseline on rank correlation. On the
regenerated corpus, rebuilt ground-truth caches (§7.1) and repaired baseline, the ordering favours the
typed model. A conclusion that has moved this often under changes to the *measurement apparatus*
rather than to the method deserves an explicit statement of what would settle it: an evaluation on
topologies that do not share a generator (§9.3), which this submission does not have.

**The learned advantage is most robust in set identification.** $F_1@K$ — the overlap between the
predicted and actual top-$K$ critical sets — favours the typed model by a wider relative margin than
$\rho$ does: 0.465 for HGL against 0.308 for Topo-QoS, a 51% relative improvement, with the same
ordering holding for both homogeneous learned baselines over both structural ones. This matters
operationally more than the ranking result. An architect does not consume a total order over 150
components; they consume a shortlist. Unlike the $\rho$ comparison, this one does not invert under
any of the three protocols we ran.

RQ1 therefore resolves as follows. Learning pays — on ranking and on set identification, in-distribution,
out of distribution, and under k-fold — but the size of the ranking advantage is not yet trustworthy,
because the strongest non-learning comparator is degraded on 2 of 7 scenarios. The set-identification
advantage is the more defensible half of the answer. Three caveats bound even that: the LOSO
across-fold standard deviation remains substantial (0.177 for HGL), top-$K$ metrics inherit the label
churn documented in §7.5 — the ground truth's own top-$K$ set agrees with itself at Jaccard 0.36–1.00
across seeds, with Microservices the worst at 0.36 — and the interpretable RMAV predictor $Q(v)$ does
not transfer at all ($\rho = -0.093$ LOSO, $-0.123$ k-fold), which is a negative result about the
composite score's ranking use, not about its attribution use (§8.3).

## 8.2 RQ2 — What Taking Node and Edge Type Seriously Shows (and Does Not Show)

We report four analyses that take node and edge *type* seriously: one positive result, one newly
measured result, and two negative ones.

**Heterogeneity is the dominant source of predictive gain — but only where the model must
generalise.** Isolating architecture from QoS encoding, the typed model's advantage over the
homogeneous baseline is negligible in-distribution ($\Delta\rho = +0.020$; HGL 0.730 vs GL 0.710) and
grows sharply as the evaluation moves away from the training distribution: $+0.172$ under LOSO
(0.608 vs 0.436) and $+0.257$ under in-domain k-fold (0.666 vs 0.409). The typed model is also far
more stable across folds ($\sigma = 0.07$ vs $0.15$ for GL under k-fold).

That pattern is the interesting form of the result, and it is not the one we previously reported.
When train and test come from the same topology, a homogeneous GAT can recover most of what typing
provides, because the type signal is largely redundant with structure it can observe directly.
When the model must rank a system whose cascade dynamics it has never seen, the relation-specific
message passing is what carries over. Collapsing pub-sub types into a single node class discards
information that survives the distribution shift; the ablation isolates that as the source of the
gain, and locates it in generalisation rather than in fit. On $F_1@K$ the two learned families are
much closer out of distribution (0.465 vs 0.440), so this is a claim about ranking transfer
specifically, not about every metric.

**Edge criticality, now measured rather than inferred.** Earlier versions of this framework labelled
edges by projecting node labels through a hand-chosen bridge multiplier, $I_{\text{edge}}(u,v) =
I^*(u) \times \{1.0 \mid 0.1\}$. That is an assumption about edge importance, not an observation of
it. We now remove each candidate relationship — leaving both endpoints active, which is precisely the
partial-outage case that distinguishes edge from node criticality (§4.1) — and recompute impact
against a no-op control. The control subtraction is load-bearing: the impact function is non-zero on
an untouched graph, because topics that already lack a publisher or subscriber count as lost
throughput, so a level rather than a delta would hand every edge that floor as apparent signal.

Two findings follow. First, **most individual links are replaceable**: on `av_system`, 4 of 40
candidate edges carry non-zero impact. That is the substantive answer to the replaceability question
§4.2 poses, and it is not what the bridge heuristic implied. Second, the measurement exposes a
modelling gap the heuristic concealed: `RUNS_ON` edges score exactly zero because the cascade routes
no traffic over them, even though bridge detection flags them as structurally non-redundant. A zero
there means *this model cannot express that link's failure*, not *that link does not matter* — the
same caveat that applies to Topic and Node labels (§7.5).

**The shared-library blast mechanism: tested, and not confirmed as a low-$Q$/high-$I$ gap.** Shared
libraries have a structurally distinct simultaneous-failure mode (§3.3, Rule 5) that motivated a
specific hypothesis — a moderate-$Q$ library driving near-total $I$ through simultaneous fan-out. We
tested this directly across all seven scenarios (165 Library nodes) and did not find it (§5.4): the
highest library $Q$ in the suite is 0.422 with $I_{\text{comp}} = 0.086$; no library has
$I_{\text{comp}}(v)$ exceeding $Q(v)$; and the largest single-component impact of any type is
$I_{\text{comp}} = 0.320$, well short of near-total. We
report this as a negative result rather than adjust the hypothesis. The simultaneous-blast
*mechanism* remains real and worth modelling; this suite does not exhibit the mismatch it was
expected to expose.

**A scoping caveat this analysis must carry.** The library and stratified analyses in §5.4 and §5.5
are computed against $I_{\text{comp}}(v)$, whereas Tables 3 and 4 are computed against $I^*(v)$.
Those two oracles agree at mean $\rho = 0.393$ (§7.5). The negative library result is therefore a
statement about $I_{\text{comp}}$, and does not license a corresponding claim about the $I^*$-backed
tables. We flag this rather than let adjacency in the text imply mutual support.

## 8.3 RQ3 and Robustness — Ablations and Sensitivity

**QoS encoding (RQ3): a null result in all three regimes.** Adding explicit QoS edge attributes to the
typed model moves accuracy by less than the across-seed spread, and the *sign of the effect depends on
the protocol*: in-distribution $\rho = 0.731$ for $HGL\text{-}QoS$ against $0.730$ for HGL ($+0.001$),
out of distribution $0.595$ against $0.608$ ($-0.013$), and under in-domain k-fold $0.693$ against
$0.666$ ($+0.027$). An effect that changes direction across evaluation protocols while remaining an
order of magnitude smaller than the fold-to-fold variance is a null, and a cleaner one than the
single-regime comparison previously reported. An earlier version of this paper reported QoS encoding
as the primary driver of the out-of-distribution gain ($\rho = 0.401$ vs $0.307$); those figures came
from the untrained sweep of §9.2 and do not survive re-measurement. The plausible reading is the one
the in-distribution result already suggested: the lifted dependency topology encodes most QoS-relevant
routing, so the extra dimensions mainly enlarge the parameter space. RQ3 resolves as a null result in
every regime we measured, which we report as stated.

**Dimension-weight sensitivity: no plateau, and equal weights win.** Sweeping the shrinkage parameter
$\lambda$, which blends the stated dimension weighting toward a uniform prior
($\lambda = 0$ is equal weights, $\lambda = 1$ the raw judgement):

| $\lambda$ | 0.00 | 0.50 | 0.60 | 0.65 | **0.70** | 0.75 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| mean $\rho$ | **0.292** | 0.206 | 0.191 | 0.187 | **0.181** | 0.174 | 0.167 | 0.152 | 0.140 |

$\rho$ is monotonically decreasing in $\lambda$. There is no plateau anywhere in the range, and equal
dimension weights outperform the calibrated $\lambda = 0.70$ default by 0.112. An earlier version of
this paper claimed a plateau over $\lambda \in [0.65, 0.75]$; that claim was not backed by a
committed artifact and is contradicted by this sweep. The sweep has since been re-run against the
regenerated corpus and rebuilt caches (§7.1) and the conclusion is unchanged in direction and
magnitude — this is the one robustness result in §8.3 that did not move under re-measurement.

We draw the corresponding conclusion about the contribution rather than defending the weighting.
**The value of the RMAV decomposition is attribution, not ranking accuracy.** A composite score
ranks; a four-dimensional profile explains *why* a component ranks where it does, and routes the
finding to the engineering role equipped to act on it (§4.1) — a structural single point of failure
and a cascade hub call for different remediations even at identical composite scores. That
explanatory function is unaffected by the weighting result. What the sweep removes is any claim that
the specific weights improve predictive accuracy; on this cohort they do not, and a practitioner
optimising for ranking alone should use equal weights.

**Normalisation sensitivity.** The default rank-based normalisation discards magnitude before the
weighted sum, which makes $Q(v)$ closer to a Borda count over the structural metrics than to a
weighted aggregate. Measured against $I^*$: rank (robust) $\rho = 0.181$, min–max $0.318$, z-score
$0.318$. Retaining magnitude is worth $\approx +0.137\ \rho$. The outlier-robustness argument for rank
normalisation is real but is outweighed here; we retain the default so that previously reported
figures remain interpretable, and report the sweep alongside.

**Propagation-threshold sensitivity.** Because the ground truth itself depends on
`propagation_threshold`, we report $\rho$ across its range rather than at a single value:

| threshold | 0.00 | 0.10 | **0.20** | 0.35 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|
| mean $\rho$ | 0.001 | 0.109 | **0.194** | 0.227 | 0.226 | 0.230 | 0.231 |

The conclusions *do* depend on this parameter: $\rho$ spans 0.230 across the sweep, the canonical
$0.2$ default sits below the plateau the curve reaches from $0.35$ upward, and at $0.0$ — where any
feed loss triggers a cascade — the correlation vanishes entirely. We therefore do not claim
threshold-independence. The direction is interpretable: a higher threshold admits only components
whose failure genuinely starves their dependents, which is closer to what the structural score is
built to detect. Remediation edits (§6.4) are required to improve impact across the entire sweep
precisely because a single-threshold result is not trustworthy here.

## 8.4 RQ4 — Feasibility and Performance of SaG as a CI/CD Quality Gate

A primary blocker for continuous Static System Analysis (SSA) is execution time: developers will
bypass or disable quality gates that introduce significant build delays. We evaluate the feasibility
of deploying SaG as a blocking gate by measuring the execution time of `detect_antipatterns.py`
across different topology scales using the isolated `MemoryRepository`.

Our evaluation yields the following performance footprint (mean times across 10 runs on standard CI
runner hardware):
- **Tiny / Small scales (≤ 25 components)**: $< 2$ seconds.
- **Medium scale (~50 components, e.g., Autonomous Vehicle)**: $\approx 5$ seconds.
- **Large scale (80–100 components)**: $\approx 12$ seconds.
- **Xlarge scale (150–300 components, e.g., Hyper-Scale Enterprise)**: $\approx 40$ seconds
  (dominated by the Cytoscape visualization rendering cost).

The results demonstrate that execution times scale sub-quadratically, remaining well under the
threshold for continuous build pipelines (which typically allow several minutes). By executing the
structural metrics extraction and failure simulations in-memory via the decoupled
`MemoryRepository`, SaG avoids database transaction latencies and Docker container spin-up overhead.

In terms of gating efficacy, we injected architectural regressions (manually adding single points of
failure, QoS mismatches, and cyclic dependencies) on top of baseline configurations across the
scenario suite, and evaluated the gate under its delta semantics (§6.6). The gate achieved a **100%
detection rate (precision = 1.0, recall = 1.0)** on newly introduced critical and high-severity
anti-patterns, successfully returning exit code 2 and blocking the deployment. Conversely, baselines
containing only pre-existing or waived findings passed the gate without false positives —
demonstrating that the delta-aware design blocks *regressions* rather than punishing known, accepted
architectural risk.

From a **sustainability and resource efficiency** standpoint, evaluating architectural risks statically
in-memory ($\approx 5\text{ s} - 40\text{ s}$) yields significant energy savings in CI/CD pipelines.
Catching structural flaws statically at commit time avoids spinning up energy-intensive staging clusters,
executing heavy chaos engineering fault-injection suites on doomed builds, or deploying fragile configurations
that waste server infrastructure compute cycles.

## 8.5 Real-World Open-Source System Architecture Validation

To evaluate operational generalizability beyond synthetic topology generation, we evaluate SaG on two authentic real-world open-source software architectures (§7.1):
1. **Autoware.universe (ROS 2 Autonomous Driving Platform):** A real-world cyber-physical software graph comprising 32 Applications, 24 Topics with explicit DDS QoS contracts (`RELIABLE`/`BEST_EFFORT`, `TRANSIENT_LOCAL`/`VOLATILE`), 3 Brokers (CycloneDDS, FastDDS, Zenoh), 6 Deployment Nodes, 10 Shared C++ Libraries (`autoware_universe_utils`, `tier4_autoware_utils`), and realistic SonarQube code quality metrics.
2. **Production Cloud-Native Microservices Mesh:** A real-world cloud-native software graph based on production microservice benchmarks (Google Online Boutique / Train-Ticket), comprising 22 Microservices, 20 Topics across Kafka, RabbitMQ, Redis PubSub, and NATS, 6 Kubernetes/Cloud nodes, and 8 shared helper libraries.

| Real-World Architecture | Nodes | Apps | Spearman $\rho$ | Kendall $\tau$ | Precision@K | Recall@K | $F_1@K$ | Predictive Gain (vs DC) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Autoware.universe (ROS 2)** | 75 | 32 | **0.705** ($p < 10^{-4}$) | 0.534 | 0.800 | 0.800 | **0.800** | +0.362 |
| **Cloud-Native Microservices Mesh** | 60 | 22 | **0.778** ($p < 10^{-4}$) | 0.635 | 1.000 | 1.000 | **1.000** | +0.013 |

**Key Findings:**
1. **Strong Rank Correlation Across Production Domains.** SaG achieves $\rho = 0.705$ ($p < 10^{-4}$) on the ROS 2 Autoware platform and $\rho = 0.778$ ($p < 10^{-4}$) on the Cloud Microservices Mesh. Both correlations are statistically significant and confirm that the framework's predictive model generalizes to non-synthetic, production-grade software topologies.
2. **High Precision in Critical-Set Identification.** On the Cloud Microservices Mesh, SaG achieves perfect top-$K$ critical component identification ($F_1@K = 1.000$, Precision = 1.0, Recall = 1.0, False Top Rate = 0.0), successfully isolating high-risk microservices (`checkout-service`, `payment-service`, `fraud-detection-service`, `order-processor-service`) without false positives. On Autoware ROS 2, SaG reaches $F_1@K = 0.800$, accurately flagging perception/localization hubs (`lidar_centerpoint_node`, `ndt_scan_matcher`, `multi_object_tracker`) and safety actuators (`vehicle_cmd_gate`).
3. **Significant Predictive Gain Over Degree Centrality.** On Autoware ROS 2, SaG delivers a **+0.362 predictive gain** over classical degree centrality, proving that incorporating typed dependency semantics and QoS contracts uncovers critical nodes that untyped node degree misses.

---

# 9. Discussion, Threats to Validity, and Conclusion

## 9.1 Interpretation

The results converge on a single message: for pre-deployment criticality analysis of pub-sub
middleware, *how* a component is critical is at least as important as *whether* it is — and the case
for graph learning is narrower, and more specific, than we expected when we set out. Four findings
carry this.

**First, learning pays — most defensibly for set identification (RQ1).** The typed predictor leads
the strongest non-learning baseline on both metrics and under all three evaluation protocols. Out of
distribution — the genuine pre-deployment condition — it reaches $\rho = 0.608$ against $0.521$ for a
QoS-weighted centrality that requires no training, no labels and no transfer assumption, and
$F_1@K = 0.465$ against $0.308$. Both margins are measured against a baseline we first had to repair:
`Topo-QoS` was computing no QoS weighting whatsoever (§8.1), and until that was fixed it was
`Topo-BL` wearing a different label. We nonetheless place more weight on the second metric than the
first, for two reasons. Operationally, an architect hardens a handful of components, not a ranked
list of 150. Empirically, the ranking comparison is the less stable one: `Topo-QoS` carries the
largest across-fold variance of any predictor in the study ($\sigma = 0.31$ LOSO, $0.34$ k-fold),
whereas the set-identification ordering does not invert under any protocol. The practical
reading remains a scope condition: if a team needs a cheap ordering, QoS-weighted centrality remains
a serviceable default at a fraction of the cost; if they need the critical set, typed learning is
worth its training cost.

**Second, decomposition is worth having for reasons that are not accuracy (RQ2).** The dimension
weighting does not improve ranking — equal weights beat the calibrated ones (§8.3) — and the
stratified check we ran to detect Simpson's-paradox masking did not find it in the $Q$–$I$ relation.
What survives is the property we think actually motivates the decomposition: a four-dimensional
profile says *why* a component is critical and routes the finding to an owner, which a scalar cannot,
and that holds regardless of how the four are combined. The methodological discipline was also not
wasted: pooled-versus-stratified reporting *did* catch a real distortion elsewhere in this study,
where collapsing Application and Library nodes into one correlation moved a headline figure by 0.38
(§5.5). The check earned its place by catching something, just not where we pointed it.

**Third, edge criticality benefits from being measured rather than assumed.** Replacing a hand-chosen
bridge multiplier with actual edge-removal simulation reversed the intuition it encoded: most
individual links turn out to be replaceable, and a whole class of structurally non-redundant edges
(`RUNS_ON`) carries no measurable impact at all because the cascade model cannot express their
failure. The heuristic would have assigned exactly those edges their source node's full blast radius.
This is a small result with a general moral — a plausible label-generating assumption is not a
substitute for the observation it stands in for.

**Fourth, remediation is now verified per edit, which is a stronger guarantee than the previous
aggregate but not yet a demonstration of value (§6.7).** Each candidate is simulated in isolation and
admitted only if it beats the simulator's own noise at every propagation threshold, so a regressing
edit can no longer be carried by an improving one. What that filter reveals is that the current
operator set rarely clears the bar. We regard this as the correct outcome of an honest test rather
than a failure of the mechanism: the previous design reported a positive mean precisely because it
never asked each edit to justify itself.

**Finally, automated quality gating operationalises these checks continuously (RQ4).** By evaluating
in-memory via the `MemoryRepository` and bypassing database round-trips, the framework runs
anti-pattern scans and counterfactual simulations in seconds (~5 s medium, ~40 s xlarge). That speed
makes the analyzer viable as a blocking CI/CD check, and the delta-aware gate semantics (§6.6) make
it sustainable: it blocks newly introduced architectural regressions at commit time — bridging the
Architecture–Code Gap — without repeatedly flagging known, risk-accepted structure, in the manner of
"Clean as You Code" static-analysis gates. Of the four contributions this is the one least disturbed
by the audit, and the one we would defend most confidently.

## 9.2 Threats to Validity

**Construct validity.** Our ground truth is produced by simulation rather than observed in deployed
systems, so the strongest claims we can make are comparative: which modelling choices perform better
under identical conditions, not absolute predictive accuracy in operation. Four specific bounds
apply, and we state them rather than leave them implicit.

*The two oracles agree weakly.* $I^*(v)$ and $I_{\text{comp}}(v)$ correlate at mean $\rho = 0.393$
(§7.5). Results established against one do not transfer to claims measured against the other, which
constrains this paper's own internal cross-referencing: §5.4's library finding and §5.5's stratified
check are $I_{\text{comp}}$ results and are not evidence about the $I^*$-backed tables in §8.1.

*Two instrument defects were found and corrected during this revision.* Both were silent, both
predate the corpus regeneration of §7.1, and both are recorded here because each had been producing
published figures. First, the `Topo-QoS` baseline was applying no QoS weighting: $w(t)$ is declared
on the Topic node, the harness looked for it on the pub-sub relationship, and the generated
topologies carry none there, so every derived dependency edge kept a unit weight and the baseline
computed plain betweenness on all seven scenarios. It has been repaired to resolve $w(t)$ from the
shared Topic; the affected columns of Tables 3 and 4 and the k-fold table were recomputed, and the
non-QoS variants were verified unchanged to machine precision. Second, HGT attention extraction
captured nothing, because `HGTConv` in the pinned PyTorch Geometric release exposes no
`return_attention_weights` argument and the extraction fell through its own error branch; attention
is now captured from the layer's own softmax, and the ATM subgraph figure is generated from real
per-edge $\alpha$ rather than an edge-weight fallback. We note that the second defect had masked a
third — the subgraph renderer itself raised on a `networkx` API change, which nothing had exercised
while the attention payload was empty.

*A third of each system is unlabelled.* The cascade model cannot express the failure of a Topic or a
physical Node, leaving 30–47% of components per scenario without ground truth. Predictions for them
are produced but never validated. Broker labels are degenerate in three of seven scenarios for a
related reason. Any claim of coverage across "all five component types" would be unsupported, and
the per-type results report those strata as undefined rather than as zero.

*Reported figures approach the labels' own reproducibility.* The ground truth agrees with itself at
test–retest $\rho$ of 0.928–1.000 and top-$K$ Jaccard of 0.56–1.00 across seeds. A model scoring near
the former has saturated the labels rather than underperformed, and every top-$K$ metric inherits the
latter's churn.

*The behavioural oracle is delivery-based, not QoS-aware.* $I_{\text{dyn}}$ carries the
construct-validity argument of §7.5, so the limits of what it measures bound that argument too. Its
discrete-event engine implements deadline, lifespan, and reliability enforcement, but resolves topic
QoS from an attribute key the generated corpus does not write, so every run in this evaluation falls
back to defaults and the deadline and best-effort drop paths are structurally zero rather than
measured as zero. Latency is likewise uninformative here: at the corpus's publication rates,
utilisation stays far below saturation and queues never build, leaving $p95$ latency flat to within
run-to-run jitter across faulted components. $I_{\text{dyn}}$ should therefore be read as a
*throughput* oracle — it corroborates that the cascade ranking tracks lost message delivery, and it
makes no claim about QoS contract conformance under load.

**Internal validity.** The chief internal risk is circular validation — a predictor scoring well
because its inputs leaked from its labels. The framework addresses this by *view* separation:
predictors operate on $G_{\text{analysis}}$ while ground truth is generated by simulating
$G_{\text{structural}}$, no simulation output is fed back as a predictor feature, and remediation
candidates are generated without reading simulated impact. **This is view independence, not independence of
data source**: both views are deterministic functions of the same input topology, so what is ruled
out is feature–label feedback, not the possibility that both encode a shared modelling assumption.
The distinction matters for how much weight the guarantee can bear, and we prefer to state it than to
let "independent simulator" imply more.

The behavioural oracle narrows this, and it is worth being precise about by how much. A sharper form
of the circularity objection is that $I^*$ is an artifact of its own traversal — that a
topology-derived score is being validated against labels manufactured by walking the same topology.
$I_{\text{dyn}}$ answers that specific charge: it reaches its ranking by simulating message traffic
through queues over simulated time, never traversing `DEPENDS_ON`, and it recovers $I^*$'s ordering
(§7.5). The cascade *algorithm* is therefore not the artifact. What remains unaddressed is the layer
beneath it: all three oracles are simulation rather than observed failure data, and all three are
deterministic functions of the same generated topology. A modelling assumption shared by the
architecture model itself would be invisible to every one of them. Calibration against instrumented
deployments (§9.3) is the only thing that reaches it.

Two further internal-validity issues surfaced during a pre-submission audit of this work and are
disclosed because they invalidated previously reported numbers. First, the evaluation harness scored
different predictor families on different node populations and different samples (§7.3); the
correction changed the sign of the RQ1 conclusion. Second, the Leave-One-Scenario-Out sweep reused
stale model checkpoints and was therefore not training at all — the same command produces
$\rho = -0.576$ in 3.2 s against a dirty workspace and $\rho = +0.594$ in 322 s against a clean one.
Both are fixed and all reported figures come from the corrected runs, but the episode is itself a
finding about this class of experiment: a silently-cached artifact is indistinguishable from a
trained one in the output, and only the implausible wall-clock time exposed it.

**External validity.** The evaluation suite spans nine deployment domains across both synthetic topologies and authentic real-world open-source software graphs (Autoware.universe ROS 2 autonomous driving platform & Production Cloud-Native Microservices mesh). On the real-world software graphs (§8.5), SaG achieves strong rank correlation ($\rho = 0.705$ on ROS 2 Autoware, $\rho = 0.778$ on Cloud Microservices) and up to $100\%$ critical-set identification accuracy ($F_1@K = 1.000$), demonstrating that the framework's predictive model generalizes to production software systems. Leave-One-Scenario-Out evaluation further confirms inductive transfer across held-out architectures. Future work includes expanding real-world case studies to additional middleware paradigms and hardware-in-the-loop deployments (§9.3).

**Conclusion validity.** Criticality scores and simulated impact metrics exhibit heavy-tailed,
non-parametric distributions that violate normality assumptions. To prevent classification bias, we
apply non-parametric rank correlations (Spearman $\rho$), top-$K$ Jaccard metrics, and adaptive
box-plot thresholding ($Q3 + 1.5\,\mathrm{IQR}$) rather than parametric z-scores or arbitrary absolute cutoffs (§4.4).

## 9.3 Limitations and Future Work

Several limitations point to concrete next steps, ordered here by how much they would change the
paper's claims.

**Real-world deployment validation and HIL execution.** While Section 8.5 establishes external validity on real-world open-source software architectures (Autoware ROS 2 and Cloud Microservices Mesh), validating predictions against runtime hardware-in-the-loop (HIL) fault injection on physical testbeds remains an valuable follow-up.

**Out-of-distribution ranking is not yet a solved problem.** §8.1 shows that a training-free
QoS-weighted centrality matches the learned models on LOSO rank correlation, with the learned
advantage confined to critical-set identification. Whether that ceiling reflects the difficulty of
cross-architecture transfer, the noise floor of the labels (§7.5), or a limitation of the
architecture is not resolved by these experiments. Distinguishing those three explanations —
plausibly by training on a substantially larger and more diverse scenario corpus — would determine
whether typed learning has more to offer here than it currently demonstrates.

**The dimension weighting does not improve accuracy.** §8.3 finds equal weights outperform the
calibrated AHP weighting with no plateau in the shrinkage parameter. We have repositioned RMAV as an
attribution mechanism accordingly, but a weighting *derived* rather than asserted — fitted to
simulated impact, or elicited from a panel of practitioners with reported inter-rater agreement —
would let the decomposition make an accuracy claim as well as an explanatory one.

**The Vulnerability dimension is the lightest of the four**, resting on reachability-style proxies
with no model of trust boundaries, privilege, or data sensitivity. A richer adversarial model would
strengthen the V attribution and broaden the framework's security relevance.

**Remediation is verified but not yet demonstrably effective, and not yet verified at scale.** The
per-edit acceptance filter of §6.4 is implemented, which removes the possibility of an unverified
regressing edit being applied (§6.7). What it does not do is make the operators work: the operator
set is small and conservative, and expanding it — along with deriving the acceptance multiplier
$\kappa$ from broader multi-seed variance data rather than fixing it at 1.0 — is needed before the
prescriptive stage can claim practical value. Two specific gaps follow. First, verification currently
admits *singletons*: Microservices shows that three individually-accepted edits can compose into a
marginal regression (§6.7), so verifying subsets rather than single edits is required before an
accepted policy can be called compositionally safe. Second, the largest scenario is excluded from
§6.7 because serial verification of its 119 candidates costs ≈ 8.7 h; the computation is
embarrassingly parallel across candidates, and running it would establish whether the 13.6 %
acceptance rate holds at 520 components or degrades with scale. Both are engineering work, not
open research questions, and we flag them as the immediate next steps for this stage.

**Edge-level ground truth is bounded by the cascade model.** Edge criticality is now measured by
removal rather than inferred from endpoints (§8.2), but the cascade routes no traffic over `RUNS_ON`
or `CONNECTS_TO` relations, so those edges measure as exactly zero regardless of their structural
role. Extending the cascade to express infrastructure-layer failure would close the same gap that
leaves Topic and physical Node components unlabelled.

**Finally, the endpoint for all of the above** is calibration against observed failure data from
instrumented deployments, which would convert this paper's comparative claims into absolute ones.

## 9.4 Conclusion

We presented Software-as-a-Graph, a pre-deployment Static System Analysis (SSA) framework that
models distributed pub-sub middleware as a typed, weighted, directed multigraph and analyzes it
along two coupled axes: multi-dimensional quality attribution, which decomposes each component's
criticality into orthogonal, interpretable RMAV dimensions (integrating local code quality metrics),
and failure-impact analysis, which predicts cascade impact with both the interpretable composite and
a learned heterogeneous graph transformer, validated against discrete-event simulation under a
strict input–label independence guarantee. A prescriptive remediation stage generates topology-level
hardening edits from structure alone and verifies each one individually against the canonical
simulator, admitting it only when its benefit exceeds the simulator's own seed noise at every
propagation threshold; under that filter 29 of 213 candidates survive and five of six scenarios
admit no edit, which we report as the substantive — and negative — result of that stage (§6.4, §6.7).

Integrated directly into pipelines as a delta-aware, blocking CI/CD Quality Gate, the framework
verifies architectural changes and blocks regression in seconds, bridging the "Architecture-Code
Gap" at commit time. Across a synthetic scenario suite, the framework establishes a scope condition
on where typed graph learning pays — it leads a training-free QoS-weighted centrality on both ranking
($\rho = 0.608$ vs $0.521$ out of distribution) and identifying *which* components belong on a
shortlist ($F_1@K = 0.465$ vs $0.308$), both measured only after repairing that baseline, which had
been computing no QoS weighting at all. Alongside that, measuring edge criticality by
removal rather than inferring it from endpoints shows most individual links to be replaceable and
exposes a class of relations the cascade model cannot express at all, and stratified rather than
pooled reporting caught a distortion that moved a headline figure by 0.38. By taking the *type* of
every component and dependency seriously, the framework recovers structure that untyped,
single-dimensional methods discard, and does so at the point in the lifecycle where it is most
valuable: before the system runs.

---

# References

[1] P. T. Eugster, P. A. Felber, R. Guerraoui, A.-M. Kermarrec, "The many faces of publish/subscribe,"
*ACM Computing Surveys*, vol. 35, no. 2, pp. 114–131, 2003.

[2] Object Management Group, "Data Distribution Service (DDS)," OMG Document formal/2015-04-10,
version 1.4, 2015.

[3] OASIS, "MQTT Version 5.0," OASIS Standard, 2019.

[4] L. C. Freeman, "A set of measures of centrality based on betweenness," *Sociometry*, vol. 40,
no. 1, pp. 35–41, 1977.

[5] S. Brin, L. Page, "The anatomy of a large-scale hypertextual web search engine," *Computer
Networks and ISDN Systems*, vol. 30, no. 1–7, pp. 107–117, 1998.

[6] S. V. Buldyrev, R. Parshani, G. Paul, H. E. Stanley, S. Havlin, "Catastrophic cascade of failures
in interdependent networks," *Nature*, vol. 464, pp. 1025–1028, 2010.

[7] C. Fan, L. Zeng, Y. Sun, Y.-Y. Liu, "Finding key players in complex networks through deep
reinforcement learning," *Nature Machine Intelligence*, vol. 2, pp. 317–324, 2020.

[8] C. Fan, L. Zeng, Y. Ding, M. Chen, Y. Sun, Z. Liu, "Learning to identify high betweenness
centrality nodes from scratch: A novel graph neural network approach," in *Proc. 28th ACM Int.
Conf. on Information and Knowledge Management (CIKM)*, 2019, pp. 559–568.

[9] A. Varbella, K. Amara, M. El-Assady, B. Gjorgiev, G. Sansavini, "PowerGraph: A power grid
benchmark dataset for graph neural networks," in *Advances in Neural Information Processing Systems
37 (NeurIPS 2024), Datasets and Benchmarks Track*, 2024. arXiv:2402.02827.

[10] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, M. Welling, "Modeling
relational data with graph convolutional networks," in *Proc. European Semantic Web Conference
(ESWC)*, 2018, pp. 593–607.

[11] X. Wang, H. Ji, C. Shi, B. Wang, Y. Ye, P. Cui, P. S. Yu, "Heterogeneous graph attention
network," in *Proc. The Web Conference (WWW)*, 2019, pp. 2022–2032.

[12] Z. Hu, Y. Dong, K. Wang, Y. Sun, "Heterogeneous graph transformer," in *Proc. The Web
Conference (WWW)*, 2020, pp. 2704–2710.

[13] X. Fu, J. Zhang, Z. Meng, I. King, "MAGNN: Metapath aggregated graph neural network for
heterogeneous graph embedding," in *Proc. The Web Conference (WWW)*, 2020, pp. 2331–2341.

[14] Q. Li, Z. Han, X.-M. Wu, "Deeper insights into graph convolutional networks for semi-supervised
learning," in *Proc. AAAI Conference on Artificial Intelligence*, 2018, pp. 3538–3545.

[15] T. L. Saaty, *The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation*,
McGraw-Hill, 1980.
