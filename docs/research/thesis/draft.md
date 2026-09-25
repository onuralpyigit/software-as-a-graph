# Graph Neural Networks for Reliability and Dependability Analysis in Complex Distributed Systems based on Publish–Subscribe Architecture

**Authors.** *[Omitted for double-anonymised review.]*

**Affiliations.** *[Omitted for double-anonymised review.]*

**Corresponding author.** *[Omitted for double-anonymised review.]*

---

# Abstract

Publish–subscribe middleware decouples producers and consumers, improving scalability but obscuring
the dependency chains along which one component's failure cascades. Runtime telemetry does not exist
before deployment and code-level static analysis is blind to system-level topology, so identifying
*which* components are critical — and *why* — remains difficult. We present **Software-as-a-Graph
(SaG)**, a pre-deployment **Static System Analysis** framework that models a pub-sub system as a
typed, weighted, directed multigraph over five component classes and derives logical dependencies
through typed projection rules. On this representation we train heterogeneous (**Heterogeneous Graph
Transformer**) and homogeneous graph neural networks to forecast cascading failure impact, and pair
them with an interpretable score decomposing criticality into Reliability, Maintainability, Availability and Vulnerability (RMAV)
dimensions. Both are compared against discrete-event cascade simulators operating on a structurally
disjoint view of the same model, under an input–label independence guarantee. Across twelve synthetic
topologies, evaluated leave-one-scenario-out, and hand-authored models of five open-source systems, we
report five results. **(1)** The QoS-aware representation carries most of the signal: QoS-weighted
closed-form ranking outperforms unweighted centrality on all twelve held-out architectures
($\rho = 0.553$ vs $0.349$). **(2)** On their own, learned engines are statistically on par with that
closed-form score (`HGT-QoS` $\rho = 0.638$, not significant), but hybrids that learn a correction to
it significantly outperform it ($\rho = 0.657$ and $0.683$, each on 11 of 12 folds, Holm
$p \le 0.0068$). Zero-shot, the learned engines rank the five system models at $\rho = 0.760$–$0.805$
against $0.511$–$0.526$ for every training-free score, though on the components that actually
propagate failures the comparison is unresolved. **(3)** At matched capacity, heterogeneous typing adds
nothing over homogeneous attention ($\Delta\rho = -0.014$); the QoS edge channel does ($+0.073$).
**(4)** Attribution earns its place as explanation rather than accuracy: equal dimension weights
outperform the calibrated weighting, and the composite transfers only weakly ($\rho = 0.205$). **(5)**
The cascade oracles agree only moderately (composite $\rho = 0.395$, behavioural $0.627$), bounding
construct validity; edge criticality is measured by removal rather than inferred, showing most links
replaceable. Finally, SaG operates as a blocking CI/CD quality gate, evaluating a candidate topology
in well under a minute even at 500+ components, though the anti-pattern catalog's agreement with the
cascade oracle is modest (precision 0.24–0.40, Cohen's $\kappa$ from $-0.04$ to $0.30$ across our
corpus), and the release gate's thresholds, calibrated on the synthetic corpus, pass on only one of
the five system models.

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
the multi-dimensional composite $Q(v)$ and a learned **Heterogeneous Graph Transformer** (**HGT**).
We evaluate the learned predictor in two variants — QoS-masked (HGL) and QoS-encoded
($HGL\text{-}QoS$) — to isolate what explicit QoS contract features contribute; §8.3 reports that
contribution as a null, so every headline figure in this paper is the QoS-masked HGL, and we name the
variants separately throughout rather than presenting the QoS-encoded model as the framework's
predictor. Both are validated against a discrete-event simulator under an **input–label independence
guarantee**. Finally, a **prescriptive remediation** stage generates topology-level hardening edits
and verifies them on counterfactual graphs in-memory.

To make SSA continuous, SaG integrates directly into CI/CD pipelines as a blocking gate: a dedicated
CLI script runs the anti-pattern catalog against the candidate topology in seconds and fails the
build (exit code 2) when it finds CRITICAL or HIGH severity structural anomalies (§6.6). The gate is
currently absolute rather than delta-aware — it evaluates the full finding set on every run rather
than diffing against a merge-base baseline — a limitation §6.6 and §9.3 discuss and scope as future
work.

*(Figure 1: end-to-end SaG pipeline — architecture description → typed multigraph → `DEPENDS_ON`
projection → the two predictor paths and the simulation oracle path, with the independence boundary
between them marked → remediation and the CI/CD gate.)*

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
>
> **RQ5.** Does the framework's predictive ranking transfer to architectures it did not generate —
> that is, to systems specified independently of our topology generator?

RQ1 is deliberately phrased as *where* rather than *whether*: the answer turns out to depend on the
evaluation protocol and on whether learning replaces the closed-form score or corrects it, and a
formulation that admits only "learning is / is not required" would have obscured that (§8.1).

RQ1, RQ2, and RQ3 are answered on the twelve-scenario synthetic suite (§8.1–§8.3); RQ4 evaluates
gating feasibility and performance (§8.4); and RQ5 is answered on hand-authored models of five
open-source systems (§8.5). RQ5 carries the paper's external validity and is, correspondingly, the
question our evidence answers most weakly — §8.5 states in full what five hand-built models with
simulated ground truth can and cannot establish.

## 1.5 Contributions

This paper makes the following contributions:

1. **A typed graph model with hierarchical SCA metric integration.** We define the SaG multigraph
   and the RMAV decomposition, which propagates code-level quality metrics (SonarQube `cm_*` fields)
   into global system criticality scores (§3, §4).
2. **A scope condition on where graph learning pays for pub-sub criticality.** Under a single
   evaluation contract applied to every predictor (§7.3), and over twelve held-out architectures,
   learned engines alone are statistically on par with the strongest training-free baseline, a
   QoS-weighted centrality (`HGT-QoS` $\rho = 0.638$ vs $0.553$, not significant), while hybrids that
   learn a correction to that baseline significantly outperform it ($\rho = 0.657$ and $0.683$, each
   on 11 of 12 folds, Holm $p \le 0.0068$), because the learned and closed-form engines fail on
   different architectures (§8.1). All margins are measured after repairing that baseline, which was
   silently computing unweighted betweenness on every scenario; we report the repair because a
   baseline accidentally identical to the one it should improve on inflates any margin measured
   against it. At matched capacity, heterogeneous relation typing adds nothing over homogeneous
   attention ($\Delta\rho = -0.014$), and the QoS edge channel is what helps ($+0.073$; §8.2). The
   contribution is the scope condition, not a win for any single engine.
3. **Multi-dimensional criticality attribution, positioned as explanation rather than accuracy.**
   RMAV decomposes criticality into four dimensions with distinct remediation owners, so a diagnostic
   is traceable to an action. A shrinkage sweep shows the dimension weighting does *not* improve
   ranking accuracy over equal weights (§8.3); we report this and scope the contribution to
   attribution accordingly (§4, §8.3).
4. **Relationship criticality as a first-class measure, and measured edge ground truth.** We give
   inter-component dependencies the same four-dimensional attribution as components (§4.7), so that
   the partial-outage case — one link down, both endpoints healthy — is scored rather than inferred
   from endpoint scores. Separately, we obtain edge ground truth by simulating removal of each
   candidate relationship rather than projecting node labels through a heuristic multiplier, finding
   that most individual links are replaceable and exposing a class of structurally non-redundant
   edges the cascade model cannot express (§8.2). We are explicit that these two are computed over
   different edge populations, so the second does not validate the first (§4.7, §9.3).
5. **An automated CI/CD quality gate.** We formulate a build-blocking gate that evaluates
   system-level structural risk statically and executes in well under a minute on every scenario in
   our corpus (§6, §8.4). The gate is absolute rather than delta-aware in the current implementation;
   §6.6 and §9.3 scope delta-awareness and a waiver register as design work this contribution does
   not yet include.
6. **A prescriptive remediation stage with per-edit counterfactual verification.** We formalise a
   Generate→Verify procedure in which every candidate edit is simulated in isolation and admitted
   only if it improves impact by more than the simulator's seed noise, at every propagation threshold
   (§6.4, §6.7).
7. **An account of the evaluation methodology itself.** We document two defects in our own harness —
   non-matching evaluation populations across predictor families, and a stale-checkpoint path that
   silently skipped training — that produced published-looking numbers of the wrong sign, together
   with the contract that prevents each (§7.3, §9.2). We report this because both failure modes are
   invisible in the output and, we suspect, not unique to this study.
8. **Zero-shot evaluation on models of open-source systems.** We evaluate SaG on hand-authored
   models of five open-source systems — Autoware.universe (ROS 2), EdgeX Foundry, Home Assistant, and
   meshes modelled after Online Boutique and Train-Ticket (§7.1, §8.5) — that contribute nothing to
   training. Learned engines trained only on synthetic scenarios rank them at $\rho = 0.760$
   (`HGT-QoS`) and $0.805$ (`GAT-QoS`) against $0.511$–$0.526$ for every training-free score, and
   roughly double top-$K$ critical-set overlap. Restricted to the components that propagate
   failures, every interval spans zero at five systems, and the framework's release gate passes on
   only one of the five. What the five cases jointly support is that learned ranking transfers to
   architecture models written independently of our generator, not an unqualified success on
   production software systems.

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
matching, and standards such as DDS and MQTT formalize deployment-time choices, alongside log-structured brokers
such as Kafka [44] and robotics middleware such as ROS 2 [45], — topics, brokers,
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
(ASTs) to compute complexity [30], code duplication, and modular metrics such as LCOM (Lack of Cohesion
of Methods) [29, 31]. While SCA is essential for locating intra-component defects and technical debt, it is
blind to the inter-component topology.

Static System Analysis (SSA) addresses this "Architecture-Code Gap." SSA models the system as a
global graph of communicating components, middleware routers, and hardware hosts. Rather than
replacing SCA, SSA ingests code-level metrics as node properties (e.g., LCOM, cyclomatic complexity)
and propagates them through the inter-component dependency topology. This allows architects to
evaluate how code-level fragility (e.g., a highly complex class inside an application) combines with
structural fragility (e.g., the application being a single point of failure) to create systemic
risks.

## 2.3 Continuous Pre-Deployment Verification and Gating

A common way to verify system resilience is dynamic testing, particularly Chaos Engineering [18],
popularised by Netflix's Chaos Monkey, which injects faults into live staging or production clusters.
While chaos testing evaluates real operational environments, doing so carries risk and occurs late in
the lifecycle.

Continuous pre-deployment verification shifts this analysis left, integrating it into CI/CD pipelines
[19, 20]. In this paradigm, the system architecture is defined as
"Architecture-as-Code" (AaC) via configuration descriptors (Docker Compose, Kubernetes manifests,
Helm charts). SSA tools run automatically on every pull request, parsing the configuration
descriptors to generate a counterfactual topology graph, and block the build (exiting with non-zero
status) when a change introduces critical architectural smells (like SPOFs or QoS mismatches) or
exceeds failure-propagation thresholds. Mature code-level gates follow the same discipline:
SonarQube's default "Clean as You Code" quality gate evaluates *new* code against the merge base
rather than failing builds on the accumulated state of the whole codebase, and pairs the gate with
an explicit won't-fix/false-positive marking workflow [21]. Our system-level gate adopts the analogous
semantics — blocking on newly introduced, unwaived structural regressions rather than on any
pre-existing finding (§6.6) — because real architectures legitimately contain *intentional*,
risk-accepted SPOFs that an absolute gate would flag on every build.

## 2.4 Structural Criticality Analysis

Network science offers a mature toolkit for identifying important nodes and edges. Degree, closeness
and betweenness centrality, articulation points, and PageRank-style scores are prized for their
efficiency and interpretability [4, 5, 38, 39], and studies of node removal [36], cascading failure
[37], and interdependent networks [6] have deepened our understanding of systemic fragility. Applied to
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

Most such methods build on the homogeneous message-passing lineage of GCN [40], GraphSAGE [41] and
GAT [42], and therefore target *homogeneous* graphs. Pub-sub middleware is intrinsically
heterogeneous — applications publish and subscribe to topics, topics are routed through brokers,
libraries introduce code dependencies, and deployment nodes impose locality — and flattening this
into a homogeneous graph discards information about how failures propagate. Heterogeneous graph
neural networks address this directly: RGCN applies relation-specific transformations [10], HAN uses
hierarchical attention [11], HGT parameterizes attention by node and edge type [12], and MAGNN
aggregates along metapaths [13]. A known hazard in dense, hub-dominated regions is over-smoothing
[14, 53]. Our learned predictor adopts relation-specific message passing over the native typed
architecture for exactly these reasons, but we treat it as one of two predictors rather than the
sole contribution: a central question of this paper (RQ1) is *where* such learning improves on
non-learning alternatives — in recovering the full ordering, in identifying the critical set, or
both — since, as §8.1 shows, the answer differs depending on which of those is asked about.

## 2.6 Quality Attributes and Multi-Criteria Scoring

Software quality is conventionally described along attributes such as reliability, maintainability,
availability, and security, formalised in the product quality model of ISO/IEC 25010:2023 [16], and a
substantial literature connects these attributes to measurable structural and code-level properties.
The quality-in-use portion of the earlier ISO/IEC 25010:2011 has since been separated into a standard
of its own, **ISO/IEC 25019:2023** [17]; under that model, stakeholder harm is evaluated over three
macro-characteristics: *Beneficialness* (Usability: Effectiveness, Efficiency, Satisfaction), *Freedom
from Risk* (Economic, Health, Life, Environmental), and *Acceptability*. The dependability vocabulary we adopt for failure, fault and
impact follows the standard taxonomy [32], and the architecture-evaluation tradition we position
against is that of scenario-based methods such as ATAM [33, 34]. Combining several structural
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
the effect of a change by re-analyzing the modified model. Catalogues of architectural bad smells
[22, 23] formalise these structures at the component-and-connector level, and the microservice
literature has extended them to distributed deployments, where cyclic and hub-shaped service
dependencies carry the same diagnosis [24, 25]. The underlying dependency metrics — instability,
afferent and efferent coupling — descend from Martin's design-quality criteria [26], and the
technical-debt framing that motivates acting on them from Cunningham [27] and the subsequent
management literature [28]. Our prescriptive stage is in this spirit
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
predictors, and a simulation-verified, blocking CI/CD quality gate. The stratified
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

**Table 1. Node types of the SaG model.** Each type carries distinct failure semantics.

| Type | Role | Representative instances |
|------|------|--------------------------|
| **Application** | A process that publishes and/or subscribes to topics | ROS 2 node, Kafka producer/consumer, MQTT client |
| **Broker** | A message-routing intermediary | RabbitMQ, Mosquitto, DDS middleware |
| **Topic** | A named message channel | `/sensor/lidar`, `order.events` |
| **Node** | A physical or virtual host | server, cloud VM, embedded controller |
| **Library** | A shared code dependency | sensor driver, codec, message library |

**Structural edge types.** Six edge types are imported from the topology description and carry the
direction in which messages or hosting relationships flow:

**Table 2. Structural edge types**, imported directly from the architecture description.

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

**Table 3. QoS symbolic-value to numeric-score mapping** used by the edge weight of §3.2.

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

**Table 4. Type-specific vertex weight aggregation rules.**

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

**Table 5. The six `DEPENDS_ON` projection rules** deriving logical dependencies from structural edges.

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

**Table 6. The four analytical layer projections** and the quality dimension each most informs.

| Layer | Projection | Vertices | Dependency types | Quality focus |
|-------|-----------|----------|------------------|---------------|
| Application | $\pi_{\text{app}}$ | App, Library | `app_to_app`, `app_to_lib` | Reliability |
| Infrastructure | $\pi_{\text{infra}}$ | Node | `node_to_node` | Availability |
| Middleware | $\pi_{\text{mw}}$ | Broker (in App/Node context) | `app_to_broker`, `node_to_broker`, `broker_to_broker` | Maintainability |
| System | $\pi_{\text{system}}$ | all five types | all six | Overall |

The middleware layer includes Application and Node vertices in the subgraph to preserve incoming
edges, but reports results only for Brokers. Components further aggregate along a MIL-STD-498 [52]
hierarchy — CSU → CSC → CSCI → CSS — so that criticality can be rolled up from a unit to a
configuration item to the whole system, supporting reporting at whatever granularity an
organization's software configuration management already uses.

## 3.6 Running Example

Consider three applications $a_1, a_2, a_3$, where $a_1$ publishes to a topic $t$ that $a_2$ and
$a_3$ subscribe to, all three depending on a shared library $\ell$. A single broker $b$ routes $t$,
and one host $n$ runs all four processes. The topic declares
`RELIABLE`/`TRANSIENT_LOCAL`/`HIGH` with a 1 KiB payload, which by §3.2 gives it a weight of
$w(t) = 0.596$. The structural graph records $a_1\!\xrightarrow{\text{pub}}\!t$,
$a_2,a_3\!\xrightarrow{\text{sub}}\!t$, $b\!\xrightarrow{\text{routes}}\!t$,
$a_i, b\!\xrightarrow{\text{runs\_on}}\!n$, and $a_i\!\xrightarrow{\text{uses}}\!\ell$. Derivation
adds $a_2\!\to\!a_1$ and $a_3\!\to\!a_1$ (`app_to_app`, Rule 1), $a_i\!\to\!b$ (`app_to_broker`,
Rule 2), $n\!\to\!b$ (`node_to_broker`, Rule 4) and $a_i\!\to\!\ell$ (`app_to_lib`, Rule 5). The two
structures encode
different risks: losing $a_1$ degrades $a_2$ and $a_3$ through a cascade that the simulator
propagates over time, whereas losing $\ell$ fails $a_1, a_2, a_3$ simultaneously. A topology-only
centrality score ranks $\ell$ by ordinary connectivity and cannot represent that its single failure
collapses the whole component group at once. Whether that representational difference translates
into a scoring gap large enough to matter is an empirical question we test in §5.4 (on our synthetic
suite, it does not). *(Figure 2: the running example's structural graph and its derived `DEPENDS_ON` projection, with
sequential-cascade and simultaneous-blast edges visually distinguished.)*

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

**Table 7. The four RMAV dimensions**, the architectural question each answers, and the stakeholder and engineering role each routes to.

| Dim. | Architectural Question | High score means | Harmed Stakeholder (ISO 25019) | Secondary Stakeholder (Engineering Role) |
|:----:|-----------------------|------------------|--------------------------------------------|------------------------------------------|
| **R** | How broadly and deeply does failure propagate? | Failure cascades widely; hard to contain | **Primary & Indirect:** operators and downstream beneficiaries whose tasks retry, fail over, or degrade | Reliability Engineer |
| **M** | How hard is this to change safely? | Tightly coupled structural bottleneck | **Secondary:** maintainers facing high regression likelihood upon refactoring | Software Architect |
| **A** | Is this a structural single point of failure? | Removing it partitions the dependency graph | **Primary & Indirect:** direct operators (traders, clinicians, drivers) and dependent beneficiaries facing task cessation | DevOps / SRE |
| **V** | How attractive a target is this for attack? | Central and reachable on $G^\top$, with many strongly-guaranteed flows converging on it | **Primary, Indirect & business:** parties relying on the guarantees an attacker would gain control of | Security Engineer |

Maintainability is the one dimension whose direct victim is the secondary stakeholder; the other
three route a finding to the engineering role equipped to act on it while denominating severity in
harm to primary and indirect stakeholders. The $V$ row is deliberately phrased in terms of
*guarantees* rather than asset value: $w_{\text{in}}$ is a delivery-guarantee proxy, and the model
carries no field for data sensitivity, privilege, or PII (§9.2, §9.3).

Four formal definitions establish the theoretical construct. Each is stated in full, because
several clauses that are easy to skim past do real work in what follows.

> **Definition D1 — Component Criticality.** The degree to which the failure, latency, or functional
> degradation of a specific software component — directly or transitively — reduces the system's
> capacity to enable its stakeholders to achieve specified operational goals with beneficialness
> (usability, accessibility, suitability), freedom from risk (economic, health, life, environmental),
> and acceptability (experience, trustworthiness, compliance) within its operational context.
> Realised at layer $l$ as a measure $\mathrm{crit}_l : V_l \to [0,1]^4 \times [0,1]$ mapping each
> $v \in V_l$ to $\mathbf{s}(v) = [R(v), M(v), A(v), V(v)]^T$ and composite $Q(v)$.

*"Failure, latency, or functional degradation"* names three distinct fault modes. The structural
estimator does not separate them — RMAV scores a component's *exposure*, which is why one score
covers all three — whereas the simulation oracle does (§5.1). *"Directly or transitively"* is why
Reliability exists as a dimension separate from Availability: the harm is loss of stakeholder
outcomes reachable *through* the component, not loss of graph connectivity. *"Within its operational
context"* is the clause §4.3 operationalises through the QoS-profile adaptation of the composite
weights.

> **Definition D2 — Relationship Criticality.** The degree to which the disruption, latency, or data
> loss across a specific inter-component interaction or dependency path — **with both endpoint
> components remaining operational** — reduces the system's capacity to enable its stakeholders to
> achieve specified goals with beneficialness, freedom from risk, and acceptability, **in proportion
> to the absence of redundant or fallback paths around it**. Realised at layer $l$ as
> $\mathrm{crit}_l : E_l \to [0,1]^4 \times [0,1]$, the same signature as D1.

The first emphasised clause is what makes D2 more than D1 restated for edges: it isolates the
*partial-outage* case, in which the component is up and its dashboards are green while one data flow
has stopped. It is also exactly the condition the edge oracle enforces (§8.2). The second clause
makes replaceability *scale* the harm rather than gate it, which is why only the Availability
dimension is bridge-gated while R, M and V score replaceable links too (§4.7).

> **Definition D3 — Criticality is a consequence, not a risk.** Under the standard decomposition of
> risk into likelihood and consequence, criticality as defined here is the **consequence factor
> alone**. No RMAV dimension estimates how probable it is that a component or relationship fails;
> every dimension estimates how much is lost *given* that it does.

Two consequences bear directly on how the results of §8 should be read. Ranking $u$ above $v$ says
that losing $u$ hurts more, not that $u$ is more likely to be lost — so every comparison in this
paper holds likelihood fixed. And restricting the construct to consequence is precisely what makes
it computable pre-deployment: consequence follows from architecture, which exists before the system
runs; likelihood follows from behaviour, which does not.

> **Definition D4 — Criticality is relative, not absolute.** Every score and tier is relative to
> (i) the score distribution of the system $S$ being analysed, since tiers are box-plot thresholds
> over that distribution (§4.4), and (ii) the layer $l$, since both the vertex set being ranked and
> the weight normalisation change with the projection. Criticality values are therefore **not
> comparable across systems or across layers**.

A well-designed redundant system still has a CRITICAL tier, and a system full of SPOFs still has a
MINIMAL tier; the tier prioritises attention inside one system rather than comparing two. D4 also
constrains how this paper may aggregate: any figure computed over more than one scenario must be
formed from within-scenario ranks or per-scenario statistics, never from raw scores pooled across
systems. §5.5 and §8.5 carry the corresponding scoping statements.

For **components**, the dimensions are **orthogonal by construction**: each raw structural metric
feeds exactly one dimension, never more. This is a deliberate design constraint, not an empirical
observation — allowing a metric into two dimensions would silently inflate its weight relative to the
stated weighting (§4.3). Orthogonality is what makes the breakdown legible: a pure single point of
failure scores high on A but low on R, M, and V; a god-component scores high on M; a cascade hub
scores high on R. The *shape* of the profile names the failure mode. The constraint is specific to
the component decomposition; the edge formulas of §4.7 deliberately relax it in exchange for endpoint
context, and we say so there rather than letting the claim read as framework-wide.

## 4.2 RMAV Formulas

All metric inputs are rank-normalized to $[0,1]$, so every RMAV score lies in $[0,1]$. Table 8 fixes
notation for every structural metric the four formulas below consume; each is computed once on
$G_{\text{analysis}}$ and feeds exactly one RMAV dimension (§4.1).

**Table 8. RMAV input metric notation.** $G^\top$ denotes the transpose of the `DEPENDS_ON` graph
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

**The weights are stated design judgements, audited for coherence rather than elicited.** Each
comparison matrix is written on Saaty's 1–9 scale to express an intended ordering, then checked with
the Analytic Hierarchy Process [15]: row geometric means normalised to a weight vector, with a
consistency ratio $\mathrm{CR} = \mathrm{CI}/\mathrm{RI}$ required to satisfy
$\mathrm{CR} \le 0.10$. We describe them as "stated and audited" rather than "AHP-derived" because
the resulting near-zero consistency ratios ($\mathrm{CR} < 0.02$, and below $0.002$ on the
$5\times5$ intra-dimension matrices) are a symptom of the construction: a matrix filled in from a
target weight vector is consistent almost by construction, whereas genuine multi-rater elicitation
on five criteria rarely lands that low. The audit certifies internal coherence, not provenance.

**Three weighting paths exist in the implementation, and the reported results use the first.** We
set them out explicitly, because they do not coincide and an earlier version of this paper conflated
them:

**Table 9. The three weighting paths in the implementation.** Reported results use the stated default.

| Path | Composite $(w_A, w_R, w_M, w_V)$ | Intra-dimension | Used by |
|---|---|---|---|
| **Stated default** | $(0.43,\ 0.24,\ 0.17,\ 0.16)$ | exactly the coefficients printed in §4.2 | **all reported results** (§8.1–§8.5) |
| AHP reconstruction, $\lambda = 1$ | $(0.458,\ 0.246,\ 0.169,\ 0.128)$ | matches §4.2 to three decimals | upper endpoint of the §8.3 sweep |
| AHP with shrinkage $\lambda$ | $\lambda\,w_{\mathrm{AHP}} + (1-\lambda)\tfrac{1}{n_{\text{dim}}}$; at $\lambda = 0.70$, $(0.395,\ 0.247,\ 0.193,\ 0.165)$ | shrunk likewise | the §8.3 sensitivity sweep |

All three place Availability first (a SPOF is a certain graph partition), Reliability second (cascade
reach), then Maintainability and Vulnerability. Shrinkage blends toward a uniform prior and is
applied to the intra-dimension vectors as well as to the composite, so $\lambda$ moves every RMAV
formula at once; it exists because weight vectors from small comparison sets can be extreme. The
stated default is *not* a point on that $\lambda$ axis — it is a hand-set vector expressing the same
ordering — which is why §8.3 reports the sweep as a sensitivity analysis of the ordering rather than
as a tuning curve for a deployed parameter.

**A QoS-profile adaptation is applied on top of whichever vector is in force.** Before scoring,
the four composite coefficients are re-derived from the analysed system's aggregate QoS profile and
renormalised to sum to one: a predominantly `PERSISTENT`/`RELIABLE`/high-priority system shifts
weight toward $R$ and $A$, a predominantly `VOLATILE`/`BEST_EFFORT` one toward $M$ and $V$, and a
mixed profile keeps the stated defaults. This is D1's *"within its operational context"* clause made
computable, and it is on by default in every run reported here. Two consequences follow. The
effective composite is therefore **per system**, so the vectors tabulated above are starting points
rather than the coefficients any individual system is scored with — a further sense in which D4's
relativity holds. And it does not disturb the determinism of §4.5: the adaptation is a deterministic
function of the same $G_{\text{analysis}}$, with no learned or stochastic component.

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

In a specific deployment domain, Quality-in-Use loss can be further parametrized by a **Domain
Context Vector** $\vec{\omega}_{\mathrm{domain}} = [\omega_{\mathrm{Ben}}, \omega_{\mathrm{Risk}},
\omega_{\mathrm{Acc}}]$ that reweights the three harm scores — safety-critical ROS 2 prioritising
Freedom from Risk, financial HFT prioritising Efficiency under Beneficialness, and so on.

**Both $\mathbf{M}_{\mathrm{RMAV}\to\mathrm{QiU}}$ and $\vec{\omega}_{\mathrm{domain}}$ are stated
mappings, and neither is used in any result reported in this paper.** They are given here because
D1 and D2 define criticality on Quality-in-Use while the four dimensions are named after product
quality, and a reader is owed an explicit statement of how one is meant to reach the other. But the
coefficients are asserted, not fitted or elicited; unlike the composite weights they carry no
consistency audit; and no table in §8 reports an $\mathbf{h}_{\mathrm{QiU}}$ score. They should be
read as a specification of the intended correspondence, not as a validated instrument. Deriving them
— and testing whether per-domain reweighting recovers the ranking accuracy that the global weighting
does not — is future work (§9.3).

**We report the sensitivity of the composite weighting, and it is not favourable.** Sweeping
$\lambda$ over $\{0,\dots,1\}$ against simulated impact shows no plateau at any value and a monotone
decline in $\rho$, with equal weights ($\lambda = 0$) outperforming the $\lambda = 0.70$ setting by
$0.111$ (§8.3). An earlier version of this paper reported a plateau over $\lambda\in[0.65,0.75]$;
that claim was not supported by a committed artifact and does not survive measurement. Because the
decline is monotone across the whole range, the conclusion applies to the stated default of the
table above as well, even though that vector is not itself a point on the $\lambda$ axis: every
weighting that expresses the intended ordering is beaten by the uniform one on this cohort.

One reading of the decline is that a single global weighting cannot fit scenarios drawn from
domains whose harm profiles genuinely differ, which is what $\vec{\omega}_{\mathrm{domain}}$ is
meant to express. We flag that as a conjecture rather than an explanation: we have not run the
per-domain reweighting that would test it, and until we do, the measured fact is simply that the
stated weighting does not improve ranking. We keep the decomposition and drop the accuracy claim
attached to its weighting. The four dimensions earn their place by being *separately actionable* —
a structural single point of failure and a cascade hub have different owners and different remedies
even at identical composite scores (§4.1) — and that property is independent of how the four are
combined into a scalar. A practitioner optimising purely for ranking should use equal weights; a
practitioner who needs to know *why* a component is critical needs the profile, whatever the weights.

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

Scoring the running example of §3.6 with the pipeline of §4.2–§4.4 gives the following profile. The
point of the table is the divergence between the last two columns:

**Table 10. Worked RMAV attribution for the running example of §3.6.** The divergence between the last two columns is the point.

| Component | $R$ | $M$ | $A$ | $V$ | $Q$ | Composite tier | Dominant dimension tier |
|---|---:|---:|---:|---:|---:|---|---|
| $b$ (broker) | 0.569 | 0.278 | 0.335 | 0.375 | 0.356 | LOW | **CRITICAL on $A$** |
| $t$ (topic) | 0.875 | 0.305 | 0.021 | 0.188 | 0.260 | MINIMAL | **CRITICAL on $R$** |
| $a_1$ (publisher) | 0.500 | 0.627 | 0.021 | 0.667 | 0.428 | MEDIUM | **CRITICAL on $M$** |
| $n$ (host) | 0.300 | 0.405 | 0.271 | 0.438 | 0.357 | MEDIUM | HIGH on $A$ |
| $\ell$ (library) | 0.450 | 0.357 | 0.050 | 0.333 | 0.266 | LOW | LOW on $R$, $M$, $V$ |

Three components illustrate how the profile names the failure mode, and each is a case the composite
alone would mislead on. The broker $b$ is a directed cut vertex: removing it partitions the graph, so
it is CRITICAL on $A$ — driven by the directed articulation score and, because $t$ carries
`RELIABLE`/`TRANSIENT_LOCAL`/`HIGH` traffic at $w(t) = 0.596$, by QSPOF — while scoring MINIMAL on
$M$. Yet its *composite* tier is LOW. An architect reading only $Q(b)$ would deprioritise the one
component whose loss stops every dependent outright; the $A$ tier is what routes it to the SRE for a
second broker. The topic $t$ inverts the same pattern on a different dimension: CRITICAL on $R$
through its subscriber fan-out, MINIMAL overall. The publisher $a_1$ is CRITICAL on $M$ — a
betweenness and efferent-coupling bottleneck the architect should decouple — at a composite of only
MEDIUM.

This is the concrete form of the claim §8.3 arrives at empirically. The composite is a ranking
device and, on this cohort, not a good one; the *profile* is the diagnostic, and the two disagree
often enough that reading only the scalar discards the finding. Reading the broker row as a
stakeholder statement, in the terms of §4.1: if $b$ fails, $a_2$ and $a_3$ lose their only path to
$t$, so the monitoring task does not degrade — it stops. That is a Beneficialness/Effectiveness
loss, and the outage window is itself a Freedom-from-risk exposure. What the score does *not* say is
how often $b$ fails or how fast it would be restored (D3); a CRITICAL tier is a statement about
structural exposure to Quality-in-Use loss, not a measurement of Quality-in-Use loss itself (§9.2).

The edge scores of §4.7 add the complementary reading. Only one dependency in the example is a
bridge — $n \to b$, at $A = 0.354$ and $Q = 0.320$, the highest-scoring edge — while every
`app_to_broker`, `app_to_app` and `app_to_lib` edge scores $A = 0.004$: replaceable links, whose loss
costs Efficiency rather than Effectiveness. Those replaceable edges nonetheless carry
$R \approx 0.29$, because $w(e) = 0.596$ and their endpoints' own reliability reach them through the
endpoint term. That is D2's proportionality clause behaving as specified: redundancy scales the harm
to near zero on $A$ without switching the other three dimensions off.

The shared library $\ell$ illustrates
the qualitatively distinct simultaneous-blast mechanism of Rule 5 (§3.3): its individual structural
centrality need not be remarkable, yet its failure collapses $a_1, a_2, a_3$ at once, in a single
event rather than a propagation chain. Whether this mechanism produces a low-$Q$/high-$I$ mismatch in
practice is an empirical question we evaluate directly in §5.4 (on our synthetic suite, it does not);
independent of that, the mechanism is why the FanOutReduction operator (§6) is triggered by structural
blast signals rather than by $Q(v)$ itself — a library's consumer fan-out is legible from structure
alone, before any simulation is run.

## 4.7 Relationship Criticality

D2 gives edges the same signature as nodes, and this section supplies the corresponding measure. The
motivation is that an edge failure and a node failure produce different observable symptoms. A node
failure is a *total* outage of a capability: everything the component provides stops. An edge failure
is a *partial* outage: the component is up, its other consumers are fine, its dashboards are green —
but one data flow has stopped, and for the stakeholder behind that link, Effectiveness is lost just
as completely as in a full outage. Two cases follow that endpoint scores cannot express. A
high-criticality node may have uniformly low-criticality edges, as with a redundantly connected
broker where losing any single link changes nothing; and a low-criticality node may sit behind a
single highly critical bridge edge, where losing that one relationship is as consequential for its
dependents as losing a much higher-scoring component.

**Structural edge signals.** Four per-edge quantities are computed on $G_{\text{analysis}}$:

**Table 11. Per-edge structural signals** computed on $G_{\text{analysis}}$ for relationship criticality.

| Signal | Computed as | Reads as |
|---|---|---|
| $\mathbf{1}_{\text{bridge}}(e)$ | cut-edge test on the undirected projection | removing $e$ disconnects a subgraph — the Effectiveness case |
| $\mathrm{bt}(e)$ | edge betweenness on **inverted** weights, each edge's length $1/w(e)$ | fraction of shortest dependency paths crossing $e$ — the Efficiency case (how much traffic must reroute) |
| $w(e)$ | worst-case (max) QoS weight over the topics mediating the dependency (§3.3) | how strongly the flow across $e$ is guaranteed |
| $\text{path\_count}(e)$ | number of distinct mediating topics or shared hosts | coupling intensity, kept out of $w(e)$ to preserve $w\in[0,1]$ |

Weight inversion is what makes strongly-guaranteed dependencies *short*, so they attract shortest
paths rather than repelling them. Unlike the node case, $w(e)$ enters **un-normalised**: the §3.2
construction already places it in $[0,1]$.

**Edge RMAV.** Each edge is scored on the same four dimensions, blending its intrinsic signals with
the endpoint scores of §4.2:

$$R(u,v) = 0.35\,\mathrm{bt} + 0.30\,w(e) + 0.20\max\big(R(u), R(v)\big)$$
$$M(u,v) = 0.35\,\mathrm{bt} + 0.30\,\mathbf{1}_{\text{bridge}} + 0.15\,w(e)$$
$$A(u,v) = 0.30\,\mathbf{1}_{\text{bridge}} + 0.20\min\big(A(u), A(v)\big)$$
$$V(u,v) = 0.15\,w(e) + 0.20\max\big(V(u), V(v)\big)$$

combined into $Q(u,v)$ with the same composite coefficients and QoS-profile adaptation as a node
(§4.3), and classified by the same box-plot rule (§4.4) applied within the edge set.

Four design choices carry meaning. **$\max$ for $R$ and $V$, $\min$ for $A$**: a link is only as
reliable or as secure as its *riskiest* endpoint, since failure or compromise on either side
propagates across it, but only as available as its *weakest*, since the edge cannot be more resilient
than the more fragile side it connects. **$\mathbf{1}_{\text{bridge}}$ appears in both $M$ and $A$**:
a non-redundant edge is expensive to route around (an Efficiency cost to the engineering stakeholder)
*and* a structural cut-point if removed (an Effectiveness loss to the end user) — one structural
fact, two stakeholder consequences. **$w(e)$ appears in $R$, $M$ and $V$ but not $A$**: the guarantee
crossing a link scales how much its loss costs, but not whether it can be lost at all. Replaceability
is topological; consequence is QoS-weighted. This is D2's redundancy clause made operational — only
$A$ is bridge-gated, while the other three score replaceable links too. **$\text{path\_count}$ does
not enter the edge score directly**; it shapes the endpoints' $R$ and $M$ (§4.2), of which only $R$
reaches the edge again, through the endpoint term.

**Two scoping conditions.** First, the orthogonality constraint of §4.1 is a property of the *node*
decomposition and does not carry over here: $\mathrm{bt}$ feeds both $R$ and $M$,
$\mathbf{1}_{\text{bridge}}$ feeds both $M$ and $A$, and $w(e)$ feeds three of the four. The edge
formulas trade orthogonality for the endpoint context that distinguishes an edge score from a node
score, and we state the claim as node-scoped rather than framework-wide. Second, the four edge
dimensions do not draw on equal coefficient mass — $R$ sums to $0.85$ of a possible $1.0$, $M$ to
$0.80$, $A$ to $0.50$, $V$ to $0.35$ — so raw edge scores are comparable *within* a dimension but not
*across* dimensions. Because classification is box-plot relative within the edge set, per-dimension
rankings and tiers are unaffected; only the raw magnitudes are. An edge's dimension *tiers* should be
read, not its absolute dimension values.

**What validates this, and what does not.** Relationship attribution is scored over
$G_{\text{analysis}}$ — the derived `DEPENDS_ON` edges — while the edge-removal oracle of §8.2 severs
raw edges of $G_{\text{structural}}$. On `av_system` those are 3,753 derived edges against a
candidate set of 50 raw structural edges drawn predominantly from `RUNS_ON` and `CONNECTS_TO`, with a
handful of `SUBSCRIBES_TO` and `PUBLISHES_TO` relations (§8.2 gives the exact composition), and the
two populations barely intersect. This is not an oversight: it is the independence guarantee of §5.3
operating exactly as designed — predictors and labels must be computed over disjoint graph views —
and the edge case simply has no shared identifier space for the two views to meet on, where the node
case does. **There is therefore no common edge population on which $Q(u,v)$ and the measured edge
impact are both defined**, and the correlation-style validation applied to node scores in §8.1 cannot
be run for edges as the two quantities are currently constructed. We present relationship attribution
as a *defined and implemented* measure that operationalises D2, and the edge-removal measurement of
§8.2 as a separate result about the structural graph — not as a validation of the attribution, and we
do not report or imply a correlation between the two anywhere in this paper. Re-simulating on
`DEPENDS_ON` directly is not an available fix: the framework's independence guarantee (§5.3) requires
simulation to operate only on $G_{\text{structural}}$. The one route that would close the gap without
violating that guarantee — tracking, for each derived edge, which raw structural edges mediate it,
then aggregating their measured impact onto it — is a modelling exercise in its own right (the
mediating relations are many-to-many, so the aggregation rule is a choice, not a formality) and is out
of scope for this submission; we position it as future work in §9.3 rather than as a pending fix.

---

# 5. Failure-Impact Analysis via Heterogeneous GNN and Interpretable Forecasting

Quality attribution (§4) tells an architect why a component is structurally critical. This section
asks the complementary question: *how much of the system actually fails* when a given component
fails, and how well each predictor — interpretable and learned — anticipates it. We define the three
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

$I^*$ and $I_{\text{comp}}$ agree only weakly — mean Spearman $\rho = 0.394$ across the seven
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

*(Figure 6: learned relation-specific attention over a case-study subgraph, with per-edge $\alpha$
from the HGT layer's own softmax.)*

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
which is the property that licenses the framework's pre-deployment claim. For the learned predictor
specifically, the second property is also checked at inference time rather than only held as a design
invariant: the GNN service raises before running a forward pass if a feature tensor carries a target
label attribute, if any node type's feature width collides with the label dimensionality, or if a
label key appears among the input keys — a defensive check against a leakage bug introduced later in
this codebase's lifetime, not a proof that no such bug exists today. The same discipline governs the
remediation stage (§6): its candidate-generation phase never reads simulated impact.

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

*(Figure 3: pooled versus per-node-type Spearman $\rho$ between $Q(v)$ and $I_{\text{comp}}(v)$, with
per-type sample sizes.)*

**We do not find a Simpson's-paradox effect in this suite**: the pooled figure (0.374) sits inside
the per-type range (0.322–0.429) rather than diverging sharply from it. This is nonetheless a useful
result, not a null one. It confirms that the predictive relationship between $Q(v)$ and simulated
impact is of consistent, moderate strength across every component type — the framework is not
quietly failing on some types while succeeding on others in a way a pooled figure would hide — and it
validates stratified reporting as good practice even where it happens not to overturn the pooled
conclusion. We report correlation *by node type* throughout (§8) on that basis, rather than because
pooling was shown to be actively misleading here.

**Three scoping conditions on this check.** First, it is computed against $I_{\text{comp}}(v)$, whereas
the predictor tables in §8.1 are computed against $I^*(v)$; the two oracles agree at mean
$\rho = 0.394$ (§7.5), so this consistency check does not transfer to those tables. Second, the check
was worth running on its own terms: the effect it looked for *does* occur elsewhere in this study. In
the predictor evaluation, pooling Application and Library nodes into a single correlation moved HGL
on `av_system` from $\rho = 0.836$ within Applications to $0.46$ pooled — a case where a pooled
figure was actively misleading, and one that went unnoticed until the evaluation contract of §7.3 was
imposed. The methodological point stands independently of the negative finding here.

Third, and cutting across both: the figures in this subsection aggregate components drawn from seven
different systems, while D4 (§4.1) makes $Q(v)$ comparable only *within* a system's own score
distribution. The aggregate is therefore a diagnostic over the union of seven within-system rankings,
not a criticality measurement over a single population, and it is reported here only to answer the
narrow question it was built for — whether the $Q$–$I$ relation holds at similar strength in every
component type. It should not be read as a cross-system criticality result, and no claim elsewhere in
the paper rests on the pooled value.

---

# 6. Prescriptive Remediation and CI/CD Quality Gating

Attribution (§4) and impact analysis (§5) are diagnostic: they tell an architect *which* components
to harden and *why*. This section closes the loop with a prescriptive stage that proposes concrete
architectural edits and verifies that they actually reduce simulated failure impact, before any
deployment. The stage is designed to preserve the same independence discipline as the rest of the
framework: candidate edits are generated from structure alone, and only a separate simulation pass
decides whether to accept them. The section then describes how the diagnostics are operationalised
as a blocking CI/CD quality gate (§6.6).

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

**Table 12. The four remediation operators**, their structural triggers, and the failure mode each targets.

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
library population in this suite for the trigger to catch, and §6.7 shows that its yield is
concentrated in the two topologies that actually contain a fan-out bottleneck. We retain the
structural trigger because triggering on $Q$ would be unsound if such a component existed, not
because we have shown that one does.

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
shows is not a benign parameter, since $\rho$ against ground truth spans 0.230 across its range.

`PrescribeService` implements this as a three-phase procedure: compile the candidate policy (§6.2),
verify each candidate independently by constructing a graph containing that edit alone and
re-simulating it across thresholds and seeds, then apply only the accepted subset and measure the
System Risk Index before and after on the mutated graph as a whole (§6.7's Table 13). Each candidate carries its measured
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
   the accepted subset is then applied and re-checked at system level (§6.7).
3. **No Verify result feeds back into Generate within a run.** There is no closed-loop search that
   would let simulated impact influence which edits are proposed, which would reintroduce the
   circularity the framework is built to avoid.

Together these keep the diagnostic and evaluation signals separate: the thing that proposes a fix
and the thing that measures it are never the same signal, so an edit is admitted only on evidence
the proposing signal did not produce.

## 6.6 CI/CD Quality Gate Implementation

To operationalise these diagnostics, SaG integrates into developer workflows as a blocking Quality
Gate in the CI/CD pipeline. When a pull request introduces configuration or architecture
modifications (Architecture-as-Code changes), the pipeline executes the analyzer via a dedicated CLI
script, `detect_antipatterns.py`, which runs the full anti-pattern catalog against the candidate
topology and issues an exit code.

**Exit-code protocol.** The gate issues exit codes that govern pipeline execution:
- **Exit Code 0**: No architectural anomalies found; deployment is permitted.
- **Exit Code 1**: Medium-severity architectural smells found (e.g., chatty pairs or QoS mismatch
  warnings); deployment is permitted with warnings.
- **Exit Code 2**: CRITICAL or HIGH severity anomalies found (e.g., single points of failure, cyclic
  dependencies, or broker overload); the build is broken and **deployment is blocked**.

This is an *absolute* gate: every run evaluates the candidate topology's full finding set, not a
diff against a prior baseline. It has a known consequence, which we do not paper over: a real
architecture that carries an intentional, risk-accepted single point of failure — a sole-source
surveillance feed, a deliberately unreplicated legacy broker — fails the build on every commit,
indistinguishable from a genuine regression. A *delta-aware* gate that evaluates the candidate
against a merge-base topology and blocks only on newly introduced findings, together with a waiver
register recording accepted risk (entity, rule, expiry) so it stays visible rather than silently
re-triggering, would close this gap; we describe the design in §9.3 as future work rather than claim
it here, since the mechanism is not implemented in the released tool.

The underlying analysis-and-detection machinery is in-memory and does not require a live database
connection — `saag`'s thread-safe `MemoryRepository` port satisfies the same repository interface
`detect_antipatterns.py` consumes, and is what the timing harness behind §8.4's measurements uses.
Wiring that path into `detect_antipatterns.py` itself, so the packaged CLI does not require a Neo4j
connection during a CI build, is a small remaining integration step we have not made; today the
script connects to a running database.

## 6.7 What Remediation Yields Under Per-Edit Verification

Running the full Generate→Verify procedure of §6.4 across the scenario suite, with $\kappa = 1.0$,
three propagation thresholds $\{0.1, 0.2, 0.5\}$ and the first three of the five canonical seeds
$\{42, 123, 456\}$, gives the following. The seed set is reduced here, and only here, because
acceptance requires a full re-simulation *per candidate edit per threshold*, so the sweep over 332
candidates is the most expensive experiment in the study. The reduction is a compute concession, not
a methodological one, and we flag the consequence rather than argue it away: $\sigma_{\text{seed}}$ is
the quantity the acceptance rule divides by, and a three-seed estimate of it is noisier than a
five-seed one, which makes the filter correspondingly less reliable at the margin. "Cand." is the
number of edits the generator proposed; "Acc." the number that cleared
$\Delta I > \kappa\,\sigma_{\text{seed}}$ at *every* threshold; $\Delta$SRI is the system risk index
change from applying the accepted subset (positive = risk reduced).

**Table 13. Remediation yield under per-edit verification**, $\kappa = 1.0$, thresholds $\{0.1,0.2,0.5\}$, seeds $\{42,123,456\}$.

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

**Per-edit verification prevents regressing edits.** Every admitted edit is guaranteed to reduce
cascade impact individually across all propagation thresholds. This removes by construction the
failure mode of the previous unverified design, in which a regressing edit could be carried by an
improving one — an aggregate that was arithmetically correct and substantively misleading. Under the
per-edit filter no scenario in the suite regresses.

**Individually-verified edits are not shown to compose.** Acceptance is decided on singletons: each
candidate is simulated alone, on a graph containing only that edit. Nothing in the procedure
establishes that a set of individually-accepted edits remains beneficial when applied together, and
the $\Delta$SRI column reports the outcome of applying the accepted subset rather than a verified
prediction of it. Verifying *subsets* rather than singletons would close this at combinatorial cost;
we note it as a limitation of the current design rather than claiming compositional safety we have
not tested.

**The acceptance rate varies widely with topology, and that is the substantive finding.** The filter
admits 3 of 35 candidates on Autonomous Vehicle and 5 of 31 on Financial Trading, but 38 of 58 on
IoT Smart City and 14 of 19 on Healthcare. The two largest absolute risk reductions come from IoT
Smart City ($\Delta$SRI $= +0.0158$) and Hyper-Scale Enterprise ($+0.0139$) — the two scenarios with
pronounced hub-topic and fan-out structure. That the operator set has purchase precisely where a
fan-out bottleneck exists is consistent with how the operators are defined (§6.2), and suggests the
honest scope for this stage is narrower than "topology-level hardening": it is closer to "fan-out
decomposition where a fan-out bottleneck actually exists". Across the suite the improvements are
real but small in absolute terms — between $+0.0025$ and $+0.0158$ SRI — which is the result we
report rather than a demonstration that the prescriptive stage is yet practically valuable (§9.3).

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
we evaluate SaG on three real-world open-source software architectures:
1. **Autoware.universe (ROS 2 Autonomous Driving Platform) [46]:** An authentic cyber-physical ROS 2 pub-sub
   architecture comprising 32 Applications (perception, sensing, localization, planning, control), 24 Topics with
   explicit DDS QoS profiles (`RELIABLE`/`BEST_EFFORT`, `TRANSIENT_LOCAL`/`VOLATILE`), 3 Brokers (CycloneDDS, FastDDS, Zenoh),
   6 Deployment Nodes, 10 Shared C++ Libraries (`autoware_universe_utils`, `tier4_autoware_utils`), and realistic SonarQube code metrics.
2. **Production Cloud-Native Microservices Mesh [48]:** An authentic microservice architecture based on the
   Google Online Boutique e-commerce benchmark, comprising 22 Microservices (order, payment, inventory, auth,
   analytics, notifications), 20 Topics across Kafka, RabbitMQ, Redis PubSub, and NATS, 6 Kubernetes/Cloud nodes, and 8 shared helper libraries.
3. **Train-Ticket Railway Booking Mesh [47]:** An authentic microservice architecture based on the Fudan
   University Train-Ticket benchmark [47], comprising 41 Microservices (order, travel, preserve, route, seat,
   payment, food, security, admin), 30 Topics, 3 Brokers (RabbitMQ, Redis PubSub, Spring Eureka naming
   server), 8 deployment Nodes, and 8 shared Spring/MyBatis libraries. At 90 components it is the largest
   of the three real-world graphs.

Pooled across all synthetic and real-world scenarios, the evaluation corpus exercises 1,770 components.
Every scenario is versioned in the replication package. The synthetic suite is registered in a
manifest carrying a canonical SHA-256 per dataset, and a regression test re-generates each synthetic
dataset from its configuration and fails on any divergence, so the corpus is byte-reproducible from
its configs. The three real-world graphs are not part of that manifest: they are versioned files
produced by a hand-written adapter rather than by the statistical topology generator, so the
byte-identity guarantee applies to the synthetic suite only.

All seven scenarios are used for the predictor evaluation (§8.1–§8.3), the analyses of §5.4–§5.5, and
the remediation evaluation of §6.7, the last of which became tractable at the largest scale by
parallelising counterfactual verification across candidates.

## 7.2 Predictors and Baselines

The evaluation compares predictors spanning the interpretability–capacity spectrum, all consuming
the same structural analysis of each scenario:

**Table 14. Predictors and baselines**, and the factor each contrast isolates.

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
- **Statistical rigor.** Bootstrap 95% confidence intervals [50] ($B = 2000$ resamples) on mean
  $\rho$ [51], and paired Wilcoxon signed-rank tests [49] ($p < 0.05$) for predictor comparisons
  across scenarios.

**Validation gate thresholds.** The implementation carries two gate registries. The release gates
of `saag/validation/models.py` pass a layer when $\rho \ge 0.70$, top-quartile overlap $\ge 0.75$
and top-5 overlap $\ge 0.60$, and report predictive gain, $\kappa_{\text{CTA}}$ and bottleneck
precision alongside without gating on them; a gate value that was never measured is reported as
unmeasured rather than as a failure. The validation CLI (`cli/validation/statistics.py`) applies a
second, topology-class-adjusted registry, which is the one §8.5 reports against. It is parameterised
by topology class, because a threshold that discriminates on a dense topology is either trivial or
unattainable on a sparse one. We tabulate it because §8.5 reports gate *failures*, and a reader
cannot evaluate a failure against an unstated bar:

**Table 15. Topology-class validation gate thresholds.** A graph's class is assigned from the density
and hub ratio of its structural graph (derived `DEPENDS_ON` edges excluded). All four conditions must
pass for the gate to pass.

| Condition | `sparse` | `medium` | `dense` | `hub_spoke` |
|---|---:|---:|---:|---:|
| Spearman $\rho$ vs simulated impact | $\ge 0.75$ | $\ge 0.80$ | $\ge 0.82$ | $\ge 0.85$ |
| Overlap@$K$ (critical-set overlap) | $\ge 0.70$ | $\ge 0.75$ | $\ge 0.75$ | $\ge 0.80$ |
| SPOF-F1 (articulation points vs $I > 0.3$) | $\ge 0.60$ | $\ge 0.65$ | $\ge 0.65$ | $\ge 0.70$ |
| Predictive gain over degree centrality | $\ge 0.02$ | $\ge 0.03$ | $\ge 0.03$ | $\ge 0.03$ |

An earlier version of this registry carried a fifth condition, a false target rate. It was defined
as the complement of the overlap condition with a strictly tighter implied threshold in every class,
so the overlap condition could never bind and the gate was never really five conditions; it has been
retired. The class adjustment is not uniformly a tightening relative to the release gates: $\rho$
rises from $0.70$ to at least $0.75$ in every class, whereas the overlap threshold is *relaxed* to
$0.70$ on the sparse class, because on a sparse graph a handful of components carries the whole
signal and a single disagreement moves the overlap sharply. We tabulate the rule rather than describe
it as a tightening throughout.

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
transductive-leakage gap for the learned predictor. For each held-out scenario $k$, the model
is trained on the remaining six scenarios (with the largest by $|V|$ used for early stopping) and
evaluated on $k$, whose nodes never participate in any forward pass and whose labels never enter any
loss. Results are aggregated as per-fold mean $\pm$ std across seeds, then cross-fold mean $\pm$
std, with per-node-type $\rho$ retained.

**Multi-seed.** Every configuration is run over five seeds $\{42, 123, 456, 789, 2024\}$; reported
scores are seed means, and the across-seed standard deviation $\sigma_{\text{seed}}$ is both
reported and reused as the noise scale in the remediation acceptance criterion (§6.4). The one
exception is the remediation sweep of §6.7, which uses the first three of these seeds for the compute
reason stated there.

## 7.5 Ground Truth: Three Oracles, and What They Can Each Support

The three oracles introduced in §5.1 are constructed differently, measure different quantities, and
are not interchangeable; conflating them is the most likely way to over-read a result in this paper.
This section fixes which analysis rests on which, quantifies how far they agree, and states the
label-coverage bounds that apply to each.

**Table 16. The three simulation oracles**, what each measures, and which results rest on which.

| Symbol | Engine | Quantity | Used for |
|---|---|---|---|
| $I^*(v)$ | `FaultInjector` | Mean subscriber feed-loss fraction under a BFS cascade | Learned-predictor labels; Tables 18 and 20 (§8.1); the sensitivity sweeps of §8.3 |
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
case.** Mean Spearman $\rho(I_{\text{dyn}}, I^*) = 0.907$, minimum $0.748$ (Microservices) — against
mean $0.425$, minimum $-0.044$ for the two topological oracles above. Hub-and-Spoke is precisely where
$I^*$ and $I_{\text{comp}}$ collapse to near-independence ($\rho = -0.044$); $I_{\text{dyn}}$ still
agrees with $I^*$ there at $\rho = 0.883$, well above its cohort minimum. Because
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
$\rho$ of 0.807–1.000, and its own top-20% critical set agrees at Jaccard 0.44–1.00 (deterministic:
`FaultInjector`'s cascade previously iterated an unordered subscriber set while consuming seeded
random draws, so re-running the *same* seed in a different process could still change the label —
this has been fixed and is disclosed as an instrument defect in §9.2, and the figures here are the
post-fix, process-independent ones). No method can exceed the former, and every top-$K$ metric
inherits the latter — a reported $F_1@K$ on `microservices_system`, where the labels' own set
stability is 0.44, should not be read to a precision the labels do not have.

## 7.6 Model Configuration and Implementation

The learned predictors are implemented in PyTorch Geometric [43]. Table 17 fixes every hyperparameter; all
variants share it, so the contrasts of §7.2 isolate architecture and features rather than tuning
budget. The values were fixed before the reported runs and not tuned per scenario — there is no
per-scenario hyperparameter search anywhere in this study, which is a deliberate constraint (a search
per scenario would leak held-out information under the in-distribution protocol) and also a
limitation, since a tuned baseline might close some of the margins in §8.1.

**Table 17. Learned-predictor configuration.** Identical across HGL, $HGL\text{-}QoS$, GL and GL-QoS
except where the architecture differs by construction.

| Component | Setting |
|---|---|
| Convolution | `HGTConv` (heterogeneous); `GATConv` for the homogeneous GL variants |
| Layers | 3 |
| Hidden channels | 64 |
| Attention heads | 4 |
| Dropout | 0.2 |
| Input projection | per-node-type linear → LayerNorm → ReLU |
| Output heads | four RMAV residual MLPs + one composite head, sigmoid-activated |
| Optimizer | AdamW, learning rate $3\times10^{-4}$, weight decay $1\times10^{-4}$ |
| LR schedule | `CosineAnnealingWarmRestarts`, $T_0 = \max(50, \text{epochs}/4)$, $T_{\text{mult}} = 2$, $\eta_{\min} = 0.01\cdot\text{lr}$ |
| Gradient clipping | max-norm 1.0 |
| Epochs / early stopping | 300, patience 30 on validation loss |
| Node splits | 60% train / 20% validation / 20% test, pinned by node identity (§7.3) |
| Loss | composite MSE $+\ 0.5\,$multitask $+\ 0.3\,$ListMLE ranking $+\ 0.1\,$pairwise margin $+\ 0.1\,$RMAV consistency |

**Hardware and runtime.** Training and evaluation were run on a single workstation; the LOSO sweep is
the dominant cost, at roughly 31 minutes for HGL and 36 for $HGL\text{-}QoS$ across all folds and
seeds, against 5–6 minutes for the homogeneous variants and well under a minute for the training-free
baselines. The CI/CD gate measurements of §8.4 were taken on the same machine rather than on hosted
runner hardware, so they should be read as an order-of-magnitude feasibility result rather than as a
calibrated figure for any particular CI provider.

---

# 8. Results

We answer RQ1 (when interpretable attribution suffices versus when learning is required, §8.1) and
RQ2 (what multi-dimensional attribution exposes that centrality misses, §8.2), then report the
ablations and sensitivity analyses that test the robustness of these answers and settle RQ3 (§8.3),
evaluate the CI/CD quality gate for RQ4 (§8.4), and close with the real-world external-validity
evidence for RQ5 (§8.5). All figures are seed means over $\{42,123,456,789,2024\}$. Bootstrap 95%
confidence intervals ($B = 2000$) accompany Table 20, and predictor comparisons are tested with paired
Wilcoxon signed-rank tests across scenarios (Table 19). Where an interval or a test is not reported,
it is because the underlying per-fold artifact was not retained, and we say so at that point rather
than omit it silently.

## 8.1 RQ1 — Interpretable Attribution versus Learning

> **Provenance.** This section reports the twelve-scenario corpus. It replaces the seven-scenario
> figures of an earlier revision, whose Leave-One-Scenario-Out artifact was not retained. Every figure
> is transcribed from the artifact-reconciled JSS supplement ([`supplementary.tex`](../jss/latex/supplementary.tex)
> §§S17, S25 and the registered LOSO sweep), with the artifact behind each table named in its caption;
> per [`outline.md`](outline.md#source-integrity), re-read each figure from its artifact when this
> section moves into the thesis. Predictor names follow the current manuscript: Topo is this draft's
> Topo-BL, HGT and `HGT-QoS` are HGL and $HGL\text{-}QoS$, and RM / $Q(v)$ is the RMAV composite.
> The small untyped GATs are `GAT-S-P` and `GAT-S-P-w` in distribution (this draft's GL and GL-QoS,
> reading the Application–Library projection) and `GAT-S` and `GAT-S-w` under LOSO (reading the
> native multigraph).

Every figure in this section is produced by one evaluation contract (§7.3): each predictor is scored
on the same Application node set. In distribution, that set is a held-out 60/20/20 node split, redrawn
per seed and shared across variants. Out of distribution, it is the complete Application population of
the held-out scenario (26 to 300 nodes, $K = \mathrm{round}(0.2\,|V_{\text{app}}|)$ from 5 to 60).

**In-distribution, typed learning leads on the point estimate.** Table 18 reports Spearman $\rho$
against simulated impact $I^*(v)$ on the held-out split, averaged over five seeds, with the held-out
sample size $n$ on which each row's correlations are computed:

**Table 18. In-distribution held-out Spearman $\rho$ against $I^*(v)$**, seed means over
$\{42,123,456,789,2024\}$; $n$ is the number of held-out Application components. The last row gives
the mean Overlap@$K$. Artifact: `results/main_table.json` (Supplementary §S17).

| Scenario | $n$ | Topo | Topo-QoS | GAT-S-P | GAT-S-P-w | HGT | `HGT-QoS` |
|---|---:|---:|---:|---:|---:|---:|---:|
| ATM | 5 | 0.538 | **0.557** | −0.393 | −0.080 | 0.492 | 0.348 |
| AV System | 16 | 0.188 | 0.797 | **0.816** | 0.465 | 0.637 | 0.558 |
| Enterprise | 60 | 0.443 | 0.793 | 0.779 | 0.481 | 0.861 | **0.878** |
| Enterprise Integration (ESB) | 14 | 0.179 | 0.429 | 0.363 | −0.156 | 0.421 | **0.476** |
| Financial Trading | 12 | 0.387 | 0.512 | 0.565 | 0.666 | 0.693 | **0.730** |
| Healthcare | 10 | 0.291 | 0.399 | **0.725** | 0.575 | 0.575 | 0.607 |
| Industrial SCADA | 28 | 0.601 | 0.710 | 0.656 | 0.478 | 0.787 | **0.839** |
| IoT Smart City | 40 | 0.320 | 0.397 | 0.580 | 0.538 | 0.849 | **0.850** |
| Logistics Fleet | 22 | 0.511 | 0.652 | 0.746 | 0.780 | 0.796 | **0.815** |
| Microservices | 18 | 0.219 | 0.344 | 0.351 | 0.363 | 0.141 | **0.664** |
| Real-Time Gaming | 15 | 0.360 | **0.802** | 0.464 | 0.471 | 0.651 | 0.641 |
| Telecom RAN | 24 | 0.402 | 0.422 | **0.608** | 0.350 | 0.591 | 0.526 |
| **Mean $\rho$** | — | 0.370 | 0.568 | 0.522 | 0.411 | 0.624 | **0.661** |
| **Mean Overlap@$K$** | — | 0.379 | 0.390 | 0.391 | 0.346 | 0.501 | **0.503** |

`HGT-QoS` leads the strongest non-learning baseline by $\Delta\rho = +0.093$ (0.661 against
`Topo-QoS` 0.568), and the typed pair leads every other predictor on critical-set overlap (0.50
against 0.35–0.39). Its lead over the small homogeneous GATs is larger still ($+0.118$ and $+0.222$, Table 19), but
this table cannot attribute that margin to typing: the typed pair reads the native multigraph and the
homogeneous pair reads the Application–Library projection, so the difference mixes relation-specific
message passing with multi-entity visibility. The typing question is settled by the capacity- and
channel-matched control of §9.1.1 (Table 26), not by this table, and there the answer is that typing
adds nothing.

**Significance testing, and what it does and does not license.** Table 19 reports the paired
Wilcoxon signed-rank test promised in §7.3, computed across the twelve scenarios on the per-scenario
mean $\rho$. We report it in full because it qualifies our own headline:

**Table 19. Paired Wilcoxon signed-rank tests across the twelve in-distribution scenarios**
($n = 12$; two-sided). Artifact: `results/main_table.json` (Supplementary, In-Distribution
Significance Tests).

| Comparison | $\Delta\rho$ | Scenarios won | $W$ | $p$ | |
|---|---:|:---:|---:|---:|:---|
| `HGT-QoS` vs Topo | +0.291 | 11/12 | 2.0 | 0.0015 | significant |
| Topo-QoS vs Topo | +0.198 | 12/12 | 0.0 | 0.0005 | significant |
| `HGT-QoS` vs GAT-S-P-w | +0.222 | 11/12 | 3.0 | 0.0024 | significant (substrate-confounded) |
| `HGT-QoS` vs GAT-S-P | +0.118 | 9/12 | 21.0 | 0.176 | n.s. |
| `HGT-QoS` vs Topo-QoS | +0.093 | 9/12 | 23.0 | 0.233 | n.s. |
| HGT vs GAT-S-P | +0.082 | 8/12 | 29.0 | 0.470 | n.s. |
| `HGT-QoS` vs HGT | +0.037 | 8/12 | 32.0 | 0.622 | n.s. |
| GAT-S-P-w vs Topo-QoS | −0.129 | 4/12 | 21.0 | 0.176 | n.s. |

Both comparisons against unweighted centrality are significant, and so is QoS weighting of the
closed-form score itself, on all twelve scenarios. **The in-distribution margin over `Topo-QoS` is
not established by a paired test across scenarios** ($+0.093$, $p = 0.233$). At $n = 12$ the
smallest attainable two-sided $p$ is $0.00049$, so significance no longer requires a clean sweep as
it did at seven scenarios, and a 9-of-12 lead that does not reach it remains unconfirmed. We therefore state the in-distribution ranking
result as a point-estimate lead that this design cannot confirm.

**On the interpretable predictor's absence from Table 18.** $Q(v)$ appears in the LOSO table below
but not in Table 18. The in-distribution harness scores it under a separate path that does not emit
the held-out per-scenario correlations the other six variants produce, so adding a column here would
mean quoting a figure computed under a different contract, which is exactly the defect §7.3 documents
and corrects. We leave the cell empty rather than fill it inconsistently; §8.3's normalisation and
shrinkage sweeps characterise $Q(v)$'s in-distribution ranking behaviour directly, and its inductive
result is in Table 20.

Two boundary conditions frame the whole table. The ground truth agrees with *itself* at test–retest
$\rho$ of 0.811–1.000 (median 0.982; §7.5), so `HGT-QoS`'s mean 0.661 sits well inside what the labels
can support rather than near a ceiling. And the held-out sets are small on several scenarios: $n = 5$
on ATM gives $K = 1$, so a single rank swap moves $\rho$ substantially, and the per-scenario cells on
the smaller topologies are coarsely quantised. The row means, not individual cells, carry the
comparison. No predictor is uniformly best: `Topo-QoS` wins on ATM and Real-Time Gaming, a small GAT
on AV, Healthcare and Telecom RAN, and the typed pair on the remaining seven.

**Out of distribution, the learned engines lead but do not significantly outperform the
closed-form score.** Under Leave-One-Scenario-Out evaluation, the true pre-deployment condition in
which the model must rank a system whose cascade dynamics it has never seen, the registered sweep
gives:

**Table 20. Inductive Leave-One-Scenario-Out evaluation, registered sweep.** Twelve folds, five
seeds, Application population. Fold score = mean over seeds; CI = bootstrap over folds
($B = 2000$); $\Delta\rho$ is paired by fold against `Topo-QoS`; fold $\sigma$ = spread of the twelve
fold means; seed $\sigma$ = median within-fold spread; $\rho_{>0}$ = the same predictions scored only
on components with positive impact. Artifacts: `results/loso_all_variants_v5.json`,
`results/loso_significance_v5.json` (Supplementary, Registered LOSO Sweep and §S25).

| Variant | Mean $\rho$ [95% CI] | $\Delta\rho$ vs Topo-QoS [95% CI] | Fold $\sigma$ | Seed $\sigma$ | Overlap@$K$ | $\rho_{>0}$ | Training |
|---|---|---|---:|---:|---:|---:|:---:|
| Topo | 0.349 [0.254, 0.452] | −0.204 [−0.286, −0.122] | 0.173 | — | 0.366 | 0.181 | no |
| Topo-QoS | 0.553 [0.443, 0.657] | — | 0.192 | — | 0.388 | 0.280 | **no** |
| RM / $Q(v)$ | 0.205 [0.092, 0.320] | −0.348 [−0.432, −0.265] | 0.195 | — | 0.322 | 0.102 | no |
| GAT-S (homogeneous) | 0.317 [0.254, 0.381] | −0.236 [−0.342, −0.125] | 0.111 | 0.298 | 0.328 | 0.159 | yes |
| GAT-S-w (homogeneous) | 0.604 [0.538, 0.665] | +0.051 [−0.067, +0.169] | 0.112 | 0.024 | **0.431** | 0.328 | yes |
| HGT (typed) | 0.551 [0.474, 0.617] | −0.002 [−0.066, +0.072] | 0.124 | 0.114 | 0.427 | 0.299 | yes |
| **`HGT-QoS` (typed + QoS)** | **0.638** [0.561, 0.710] | +0.085 [−0.029, +0.194] | 0.133 | 0.052 | 0.424 | **0.356** | yes |

`Topo-QoS` is a QoS-weighted centrality that requires no training, no labels and no transfer
assumption; because it is never fitted, its out-of-distribution score *is* its score. `HGT-QoS`
reaches $\rho = 0.638$ against its $0.553$ and wins 9 of 12 folds, but the registered confirmatory
contrast is not significant ($p = 0.151$, Holm $0.303$), and HGT without the QoS channel ties it
($-0.002$, $p = 0.470$). Every learned interval in the $\Delta\rho$ column spans zero. The learned
engines are, however, markedly more stable across folds than the closed-form score (fold $\sigma$
0.11–0.13 against 0.19), and the two fail in different places: `HGT-QoS` loses substantively on
Enterprise ($0.461$ against $0.795$) and Telecom RAN ($0.407$ against $0.576$), and gains $+0.229$ and
$+0.210$ on Microservices and ATM, where `Topo-QoS` is weakest. Enterprise is the largest graph (520
nodes) with by far the densest derived projection (26,276 edges), yet neither graph size ($\rho =
-0.434$ with the margin, $p = 0.159$) nor density predicts in advance which engine wins on an unseen
architecture. A later CPU sweep of the same protocol, on which the amendments of §9.1.1 were
evaluated, agrees on every conclusion (`HGT-QoS` $0.622$; Table 25).

**This comparison is only meaningful because the baseline was repaired first.** In an earlier
revision, `Topo-QoS` scored *no* QoS weighting at all on the logical-dependency substrate: QoS is
declared on the Topic node, but the harness looked for it on the pub-sub relationship, which the
generated topologies emit without one. The lookup never matched, every derived dependency edge kept a
unit weight, and the QoS-weighted baseline silently computed plain betweenness on every scenario; it
was Topo under another name. We resolve $w(t)$ from the shared Topic instead, taking the strongest
contract when a pair communicates over several topics. A baseline that is accidentally identical to
the one it is meant to improve on will always flatter whatever it is compared against, and the
figures above are reported only after that defect was removed. With it removed, QoS weighting is the
largest single gain in the table: $+0.204$ over unweighted centrality, on all twelve folds.

**The history of this result, and what now settles it.** An earlier version reported typed learning
as the out-of-distribution winner on figures produced by an untrained sweep (§9.2). A later version,
after correcting the evaluation contract, reported a tie. The seven-scenario revision then reported a
typed lead ($0.608$ against $0.521$) from a run whose artifact was overwritten, against a retained
pre-repair run that recorded the opposite ordering. A conclusion that moved this often under changes
to the *measurement apparatus* needed two things before it could be stated: a larger corpus and an
analysis fixed before the result existed. Table 20 has both. Its contrasts were registered before the
twelve-fold harness produced any result (Amendment 3 re-baselined them on this sweep), its artifact is
retained in the replication package, and an independent CPU sweep reproduced its conclusions. On that footing, the answer is the tie, not the lead: learned engines are statistically on
par with the repaired closed-form score out of distribution.

**Set identification no longer separates the engines out of distribution.** In the seven-scenario
revision, $F_1@K$ was the more robust half of the learned advantage. On twelve folds, the learned
engines still lead on Overlap@$K$ (0.424–0.431 against 0.388), but the fold-level difference is not
significant either ($\Delta = +0.037$, $p = 0.470$ for `HGT-QoS`), and top-$K$ metrics inherit the
label churn documented in §7.5: cross-seed Jaccard of the ground truth's own top-$K$ set has median
0.847 and falls to 0.370 on Logistics Fleet. Identification separates the engines clearly only
in distribution (Table 18) and under zero-shot transfer to the five open-source system models, where
the learned engines roughly double closed-form top-$K$ overlap (§9.1.1).

**Half of every correlation is inertness detection.** Between 21% and 52% of each held-out
Application population carries zero simulated impact. Restricted to the components that do propagate
failures, every predictor keeps only 49–56% of its full-population correlation ($\rho_{>0}$ column),
with no separation between learned and training-free families and the method ordering unchanged.

**Learning outperforms the closed-form score only in combination with it.** Two engines registered
after this sweep (Amendments 5 and 6) give a learned model the rank-normalised `Topo-QoS` score as
an input and learn a logit-scale correction to it. On the CPU sweep, Hybrid-HGT reaches $\rho =
0.657$ ($+0.103$) and Hybrid-GAT $0.683$ ($+0.130$) against `Topo-QoS`, each on 11 of 12 folds, and
both survive Holm correction pooled over all eleven registered contrasts (§9.1.1, Table 25). They are
the only engines in this study that significantly outperform closed-form ranking, and they do so
because they keep the closed-form engine's strength on Enterprise and Telecom RAN while adding the
learned engine's gains elsewhere.

The in-domain k-fold protocol reported in the seven-scenario revision has not been re-run on the
twelve-scenario corpus (`make -f reproduce/Makefile kfold`), so no k-fold figure is reported here.

RQ1 therefore resolves as a scope condition rather than a verdict. On its own, learning leads the
strongest non-learning baseline on the point estimate both in and out of distribution, and on
neither protocol is that lead significant. The interpretable composite $Q(v)$ transfers only weakly
($\rho = 0.205$ LOSO, below both centralities), which is a negative result about the composite
score's ranking use, not about its attribution use (§8.3). What learning does establish is
complementarity: it fails on different architectures from the closed-form score, so a hybrid that
corrects the closed-form score significantly outperforms it, and a pure learned engine with the QoS
edge channel transfers best to architectures unlike the training corpus. The practical
recommendation we are willing to defend is correspondingly specific: use `Topo-QoS` as the
training-free default, a hybrid for architectures that resemble the training corpus, and a pure
learned engine with the QoS channel for substantially different ones.

## 8.2 RQ2 — What Taking Node and Edge Type Seriously Shows (and Does Not Show)

We report four analyses that take node and edge *type* seriously: one that reverses our earlier
reading of relation typing, one newly measured result, one negative result, and a scoping caveat.
Figures for the typing analysis come from the twelve-scenario corpus of §8.1 and carry the same
provenance note.

**Relation typing adds nothing once capacity and edge-channel width are matched; the QoS edge
channel does.** The seven-scenario revision reported the typed model's advantage over the
homogeneous baseline as negligible in distribution ($+0.020$) and growing sharply out of
distribution ($+0.172$ under LOSO, $+0.257$ under in-domain k-fold), and read that pattern as relation
typing carrying over under distribution shift. The twelve-scenario registered sweep reproduces the
pattern in the unmatched comparison: HGT leads the small untyped `GAT-S` by $+0.234$ under LOSO, on
12 of 12 folds ($p = 0.0005$). That comparison is confounded. `GAT-S` has 28,168 parameters against
HGT's 434,620 and reads at most a scalar edge weight against HGT's relation one-hot, and it is also
the least stable arm in the study (median within-fold seed spread $0.298$). Neither the Fisher-$z$
transform nor robust seed aggregation could detect the confound, because both hold the same four
unmatched arms fixed.

The control registered in Amendment 2, before any control result existed, removes both differences:
`GAT` and `GAT-QoS` are untyped GATs at HGT's parameter budget (437,496 and 429,992), and `GAT-QoS`
reads the same 16-D edge vector as `HGT-QoS`, relation one-hot included. At matched capacity the
untyped design rises from $0.317$ to $0.563$ under LOSO, and the typing effect disappears: the typing
main effect is $-0.014$ (4 of 12 folds, Holm $p = 0.94$) and the typing $\times$ QoS interaction is
$+0.001$, while the QoS edge channel raises both designs by $+0.073$ on 10 of 12 folds, significantly
for the untyped pair ($+0.072$, $p = 0.016$; §9.1.1, Table 26). The channel also stabilises training:
the median seed spread of the untyped pair falls from $0.083$ to $0.010$.

That is not the result we previously reported, and the difference matters for the thesis claim.
What carries over to unseen architectures is learning over the QoS-annotated graph, not
relation-specific parameters. Because `GAT-QoS` receives each edge's relation type as an input, the
precise statement is that relation-typed *parameters* add nothing beyond relation-typed *inputs*. The
same holds under zero-shot transfer to the five open-source system models, where `GAT-QoS` outperforms
`HGT-QoS` on all five ($\rho = 0.805$ against $0.760$). Three limits bound the statement. Message
directionality remains unmatched, because the registered `HGT-QoS-U` control has not been run. The
in-distribution typed lead of Table 18 cannot speak to typing either way, because the two families
read different substrates there. And $I^*(v)$ is a near-topological target (a topology-only
relabelling recovers its ordering at $\rho = 0.965$), so on this target the QoS channel acts largely as
a relation-identity and coupling-strength signal; an oracle that expressed deadline misses or
durability replay would let the encodings contribute contract semantics as well.

**Edge criticality, now measured rather than inferred.** Earlier versions of this framework labelled
edges by projecting node labels through a hand-chosen bridge multiplier, $I_{\text{edge}}(u,v) =
I^*(u) \times \{1.0 \mid 0.1\}$. That is an assumption about edge importance, not an observation of
it. We now remove each candidate relationship — leaving both endpoints active, which is precisely the
partial-outage case that distinguishes edge from node criticality (§4.1) — and recompute impact
against a no-op control. The control subtraction is load-bearing: the impact function is non-zero on
an untouched graph, because topics that already lack a publisher or subscriber count as lost
throughput, so a level rather than a delta would hand every edge that floor as apparent signal.

The candidate set on `av_system` — structural bridges united with the highest-betweenness structural
edges — contains 50 edges
(35 `RUNS_ON`, 11 `CONNECTS_TO`, 3 `SUBSCRIBES_TO`, 1 `PUBLISHES_TO`), of which exactly 4 carry
non-zero impact: the one `PUBLISHES_TO` edge and all three `SUBSCRIBES_TO` edges, each connecting a
shared library to the topic it produces or consumes. Two findings follow. First, **most individual
links are replaceable, in both magnitude and count**: the largest measured impact of severing any
single relationship is $0.00504$, over an order of magnitude below the largest single-*component*
impact in the suite ($I_{\text{comp}} = 0.320$, §5.4), and 46 of 50 candidates measure exactly zero.
That is the substantive answer to the replaceability question §4.7 poses, and it is not what the
bridge heuristic implied — the heuristic would have assigned a bridge edge its source node's full
blast radius. Second, the measurement confirms a modelling gap the heuristic concealed: every one of
the 46 `RUNS_ON`/`CONNECTS_TO` candidates — the majority of the set, and structurally non-redundant
bridges by construction — scores exactly zero, because the cascade routes no traffic over
infrastructure-layer relations at all. Bridge detection flags these links as non-redundant; the
measurement means *this model cannot express that link's failure*, not *that link does not
matter* — the same caveat that applies to Topic and Node labels (§7.5).

**This measurement does not validate the attribution of §4.7, and is not intended to.** The sweep
severs raw structural edges, whereas $Q(u,v)$ is defined on derived `DEPENDS_ON` edges; the two are
computed over populations that barely intersect by construction (§4.7), so no correlation between
them is reported here and none should be inferred from the two results appearing in the same
section.

**A methodological note on reproducing this figure.** The candidate set above requires the sweep to
be run against a freshly loaded repository, before any structural analysis has touched it. We found
during this revision that running the analysis and prediction stages against the same in-memory
repository instance *before* constructing the simulator's graph view causes derived `ROUTES` and
`DEPENDS_ON` edges to leak into what the simulator receives as $G_{\text{structural}}$ — a repository
state-ordering issue distinct from, and not caught by, the import-level independence check of §5.3.
It is not visible in this paper's other results, since the standard pipeline order is Simulate before
Predict throughout, but it can silently substitute a contaminated candidate set for a clean one in an
ad hoc script, which is precisely how an earlier revision of this figure was produced. We flag it as
a reproducibility hazard for this specific measurement rather than treat it as resolved.

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
are computed against $I_{\text{comp}}(v)$, whereas Tables 18 and 20 are computed against $I^*(v)$.
Those two oracles agree at mean $\rho = 0.395$ over the twelve LOSO topologies (range 0.083–0.653;
§7.5). The negative library result is therefore a
statement about $I_{\text{comp}}$, and does not license a corresponding claim about the $I^*$-backed
tables. We flag this rather than let adjacency in the text imply mutual support.

## 8.3 RQ3 and Robustness — Ablations and Sensitivity

**QoS encoding (RQ3): small in distribution, the working ingredient out of distribution.** Figures
in this paragraph come from the twelve-scenario corpus and carry §8.1's provenance note. In
distribution, adding the 16-D QoS edge channel to the typed model moves ranking by $+0.037$
(`HGT-QoS` against HGT, 8 of 12 scenarios, $p = 0.622$; Table 19), less than the across-seed spread.
Out of distribution the effect is larger and consistent in sign. In the registered LOSO sweep
`HGT-QoS` leads HGT by $+0.087$ on 10 of 12 folds (0.638 against 0.551, $p = 0.204$). In the
capacity- and channel-matched control, the QoS channel's main effect is $+0.073$ on 10 of 12 folds
(CI $[+0.013, +0.120]$, Holm $p = 0.127$), significant for the untyped pair ($+0.072$, $p = 0.016$)
and not for the typed pair ($+0.073$, $p = 0.129$; §8.2). It also stabilises training: the median
within-fold seed spread of the untyped pair falls from $0.083$ to $0.010$. For the closed-form score
the effect of QoS is larger still and unambiguous: QoS-weighted betweenness outperforms unweighted
betweenness by $+0.198$ in distribution and $+0.204$ out of distribution, on 12 of 12 scenarios and
folds.

This reverses the seven-scenario revision, which reported QoS encoding as a null whose sign changed
across protocols ($+0.001$, $-0.013$, $+0.027$). An earlier version still had reported QoS encoding
as the primary driver of the out-of-distribution gain ($\rho = 0.401$ vs $0.307$); those figures came
from the untrained sweep of §9.2 and did not survive re-measurement. Two qualifications bound the
present reading. Not every contrast survives correction: only the untyped pair's is individually
significant, and the matched main effect is not after Holm correction. And $I^*(v)$ is a
near-topological target (a topology-only relabelling recovers its ordering at $\rho = 0.965$), so the
QoS channel acts largely as a relation-identity and coupling-strength signal rather than as contract
semantics. RQ3 therefore resolves as a scope condition: QoS encoding matters little in distribution
and is the component that improves learned ranking out of distribution.

The sensitivity sweeps that follow (Tables 21 and 22) score the RMAV-era $Q(v)$ on the
seven-scenario corpus and have not been re-measured on the twelve-scenario one.

**Dimension-weight sensitivity: no plateau, and equal weights win.** Sweeping the shrinkage parameter
$\lambda$, which blends the stated dimension weighting toward a uniform prior
($\lambda = 0$ is equal weights, $\lambda = 1$ the raw judgement). The QoS-profile adaptation of §4.3
remains active throughout the sweep, as it is in every run reported in this paper: each $\lambda$
therefore fixes the vector that adaptation starts from, not the coefficients any individual scenario
is finally scored with. The sweep is consequently a sensitivity analysis of the *stated ordering*
under the framework's normal operating configuration, and the $\lambda$ labels should be read as
inputs to the weighting path rather than as the applied weights:

**Table 21. AHP shrinkage sensitivity.** Mean $\rho$ against $I^*(v)$ as $\lambda$ blends the stated weighting toward a uniform prior.

| $\lambda$ | 0.00 | 0.50 | 0.60 | 0.65 | **0.70** | 0.75 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| mean $\rho$ | **0.292** | 0.206 | 0.191 | 0.187 | **0.181** | 0.174 | 0.167 | 0.152 | 0.140 |

*(Figure 4: mean $\rho$ against $\lambda$ over the shrinkage sweep, showing the monotone decline and
the absence of a plateau.)*

$\rho$ is monotonically decreasing in $\lambda$. There is no plateau anywhere in the range, and equal
dimension weights outperform the calibrated $\lambda = 0.70$ setting by 0.111. An earlier version of
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

**Table 22. Propagation-threshold sensitivity.** Mean $\rho$ against $I^*(v)$ across the sweep; the canonical default is $0.20$.

| threshold | 0.00 | 0.10 | **0.20** | 0.35 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|
| mean $\rho$ | 0.001 | 0.109 | **0.194** | 0.227 | 0.226 | 0.230 | 0.231 |

*(Figure 5: mean $\rho$ against `propagation_threshold`, with the canonical $0.2$ default marked.)*

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
of deploying SaG as a blocking gate by measuring the wall-clock cost of the structural analysis and
anti-pattern catalog — the mechanism `detect_antipatterns.py` invokes — run via the in-memory
`MemoryRepository`, across all eleven scenarios in our corpus (mean over the five canonical seeds).

The measured footprint:
- **≤ 90 components** (`tiny_system` and all three real-world transcribed architectures): $0.02$–$0.04$ s.
- **98–326 components** (the remaining six generated scenarios): $0.27$–$1.24$ s. Cost does not scale
  monotonically with component count in this range — `hub_and_spoke_system` (139 components, $1.24$ s)
  costs more than `iot_smart_city_system` (326 components, $1.08$ s) — consistent with the catalog's
  cost being driven by specific detectors' complexity (e.g. `DEEP_PIPELINE`'s path enumeration) more
  than by raw component count.
- **`enterprise_system`**, the largest scenario at 520 components: $26.74 \pm 0.32$ s.

All eleven scenarios complete in well under the several-minute budget continuous build pipelines
typically allow.

In terms of gating efficacy: the gate is currently absolute rather than delta-aware (§6.6), so there
is no merge-base diff to evaluate detection against; what we can and do measure is the anti-pattern
catalog's raw agreement with the cascade oracle — the property the gate is a proxy for. Scoring
CRITICAL/HIGH findings (with edge-keyed findings crediting both endpoints) against the oracle's own
critical set gives, across five seeds: precision $0.237 \pm 0.014$, recall $0.887 \pm 0.059$, F1
$0.374 \pm 0.022$, Cohen's $\kappa = -0.036 \pm 0.049$ on the seven generated scenarios ($n=40$
scenario–seed pairs), and precision $0.402 \pm 0.059$, recall $0.861 \pm 0.105$, F1 $0.544 \pm
0.061$, $\kappa = 0.296 \pm 0.100$ on the three real-world architectures ($n=15$). Recall is high and
precision is not: the catalog over-flags relative to the oracle's critical set, and near-zero $\kappa$
on the generated corpus means the agreement it does show is close to what chance flagging at the
catalog's own base rate would produce. We read this as a genuine limitation of the pattern catalog as
a *stand-alone* predictor of simulated impact — a finding pattern catalog and rank predictor serve
different purposes (naming a structural smell versus ranking by predicted impact, §9.1), but the gap
here is wide enough that the catalog's findings should not be read as impact-calibrated. The composite
$Q(v)$ of §4 remains the ranking signal validated against the oracle in §8.1.

From a **sustainability and resource efficiency** standpoint, evaluating architectural risks statically
in-memory ($0.02\text{ s}$–$26.74\text{ s}$ across our corpus) yields energy savings relative to
spinning up staging clusters or running heavier dynamic checks per build, though we have not measured
that comparison directly.

## 8.5 RQ5 — Real-World Open-Source System Architecture Validation

> **Provenance.** This section reports the five open-source system models of the current corpus. It
> replaces an earlier three-model evaluation of the RMAV-era score. Figures are transcribed from the
> artifact-reconciled JSS manuscript and supplement ([`sec7_results.md`](../jss/sections/sec7_results.md)
> Table 9, [`supplementary.tex`](../jss/latex/supplementary.tex) §§S7, S14, S27); per
> [`outline.md`](outline.md#source-integrity), re-read each figure from its artifact when this
> section moves into the thesis. Predictor names follow §8.1.

To evaluate generalisation beyond the synthetic generator, we score SaG on hand-authored models of
five open-source systems. None contributes gradients, checkpoint selection or any other input to
training.

| System model | Original paradigm | $|V|$ | $|V_{\text{app}}|$ | Topics | Brokers |
|---|---|---:|---:|---:|---:|
| Autoware.universe (ROS 2) | publish–subscribe | 75 | 32 | 24 | 3 |
| EdgeX Foundry (Industrial IoT) | publish–subscribe | 63 | 22 | 24 | 3 |
| Home Assistant (Smart Home) | publish–subscribe | 63 | 24 | 22 | 3 |
| Online Boutique (pub-sub model) | gRPC | 60 | 22 | 20 | 4 |
| Train-Ticket booking mesh | RPC | 90 | 41 | 30 | 3 |

Each model was written by one author as a typed multigraph from public documentation
(`saag/adapters/realworld_adapter.py`); none is a mechanical extraction. Brokers, QoS profiles, code
metrics and host specifications are partly assumed, and where a system declares no QoS manifest,
standard middleware defaults (ROS 2 Best-Effort/Reliable, MQTT QoS 0/1) are applied uniformly to
every predictor. Two models depart materially from their originals: the Online Boutique model is a
22-application pub-sub mesh with four brokers, whereas the original is about eleven gRPC services
with no broker, and the Train-Ticket model represents its service-discovery server as a broker.
Neither contains a synchronous call edge. The test is therefore transfer to independently authored
architecture models under simulated reachability, not to deployed systems.

**Zero-shot transfer of the learned engines.** `HGT-QoS` and `GAT-QoS` were trained on all twelve
synthetic scenarios and evaluated without fine-tuning, at the same 3-layer, 300-epoch budget as every
LOSO result, over five seeds. The training-free scores are deterministic and are scored on identical
labels and node sets.

**Table 23. Zero-shot transfer to the five open-source system models.** Spearman $\rho$ against
$I^*(v)$ on the Application population; $\pm$ is the spread over five training seeds; $n_{>0}$ is the
number of Applications with positive impact; the last column restricts `HGT-QoS` to them. Artifacts:
`results/realworld_zeroshot_v7.json` (`HGT-QoS`), `results/realworld_zeroshot_*_cpu.json`
(`GAT-QoS`).

| System model | $|V_{\text{app}}|$ | $n_{>0}$ | Topo | Topo-QoS | `HGT-QoS` | `GAT-QoS` | `HGT-QoS` $\rho_{>0}$ |
|---|---:|---:|---:|---:|---|---|---:|
| Autoware.universe (ROS 2) | 32 | 19 | 0.307 | 0.378 | 0.716 ± 0.081 | **0.758 ± 0.019** | +0.517 |
| EdgeX Foundry | 22 | 10 | 0.534 | 0.534 | 0.793 ± 0.037 | **0.815 ± 0.055** | +0.183 |
| Home Assistant | 24 | 17 | 0.297 | 0.289 | 0.864 ± 0.063 | **0.925 ± 0.023** | +0.702 |
| Online Boutique (pub-sub model) | 22 | 8 | **0.891** | 0.888 | 0.710 ± 0.070 | 0.750 ± 0.119 | −0.031 |
| Train-Ticket booking mesh | 41 | 14 | 0.528 | 0.541 | 0.717 ± 0.096 | **0.777 ± 0.007** | −0.192 |
| **Mean** | — | — | 0.511 | 0.526 | 0.760 | **0.805** | +0.236 |

**Key findings:**

1. **The learned engines transfer to independently authored topologies, and typing is not what
   transfers.** Both learned engines lead on four of five systems, with full-population bootstrap
   intervals that do not overlap those of the training-free scores (`HGT-QoS` $0.760$
   $[0.714, 0.819]$ against `Topo-QoS` $0.526$ $[0.357, 0.699]$ and RM / $Q(v)$ $0.516$
   $[0.343, 0.680]$). The untyped `GAT-QoS` outperforms `HGT-QoS` on all five systems, and also on
   identification (Overlap@$K$ $0.519$ against $0.470$, PR-AUC $0.790$ against $0.713$), consistent
   with §8.2: what transfers is learning over the QoS-annotated graph, not relation-specific
   parameters. Identification separates learned from closed-form scores most clearly: top-$K$ overlap
   averages $0.470$ for `HGT-QoS` against $0.248$ for the closed-form scores, and PR-AUC $0.713$
   against $0.474$–$0.521$. On EdgeX, symmetric adapter-to-broker stars create betweenness ties that
   collapse closed-form triage entirely (Overlap@$K = 0.000$). The one system where closed-form
   ranking wins is the Online Boutique model, where Topo reaches $0.891$.
2. **On the components that propagate failures, the comparison is unresolved.** Restricted to
   Applications with positive impact, `HGT-QoS` keeps a positive mean ($\rho_{>0} = +0.236$,
   $[-0.053, +0.525]$) where every training-free score turns negative (`Topo-QoS` $-0.092$,
   RM / $Q(v)$ $-0.055$), but every interval spans zero at five systems. $\rho_{>0}$ is positive on
   the three models of publish–subscribe systems and non-positive on the two modelled after RPC
   systems. Both RPC-derived models are encoded as publish–subscribe graphs and labelled by the same
   forward-reachability oracle, so this split cannot be attributed to call-tree semantics; it is a
   pattern to test, with synchronous edges and a backward-propagating oracle (§9.3), not a finding.
3. **The primary configuration is the LOSO one, not one chosen for these systems.** An earlier,
   non-blind configuration used 2 layers and 150 epochs, chosen because these meshes are small. That
   choice appealed to a property of the test systems, so it is not reported as primary; it is
   uniformly slightly stronger and changes no conclusion.

**The interpretable score and the release gate on the same systems.** A separate run scores the
deterministic RM / $Q(v)$ with the validation CLI, whose injector caps cascade depth at 5 and averages
five injector repeats per node, so its figures are not commensurable with Table 23's unlimited-depth
labels. Under that configuration $Q(v)$ reaches $\rho = 0.800$ on EdgeX, $0.778$ on the Online
Boutique model, $0.759$ on Train-Ticket, $0.685$ on Autoware and $0.514$ on Home Assistant, and leads
unweighted degree centrality on all five ($+0.014$ to $+0.427$; Wilcoxon $p \ge 0.33$ at $n = 5$).
Because $Q(v)$ is never fitted, this is not a transfer result: it shows that the attribution remains
informative on architectures we did not generate. Its critical-set identification on the Application
population is weak ($F_1@K$ from $0.000$ on EdgeX to $0.625$ on Train-Ticket); pooled over all
entity types it reaches $0.667$–$1.000$, but that figure is inflated by correctly identifying inert
infrastructure and we do not read it. An earlier draft reported the Autoware correlation as unstable
from sweep to sweep at a fixed seed set; the cause was the `FaultInjector` ordering defect of §9.2,
and reruns now agree to the precision shown.

The topology-class validation gate, at the `sparse`-class thresholds of Table 15 that the
supplement applies to all five ($\rho \ge 0.75$, Overlap@$K \ge 0.70$, SPOF-F1 $\ge 0.60$,
predictive gain $\ge 0.02$), passes on one system in five. EdgeX passes all four
conditions on all five seeds, helped by the pooled overlap figure the previous paragraph declines to
read. Autoware fails the $\rho$ and SPOF conditions, the Online Boutique model fails SPOF and
prediction gain, Train-Ticket fails SPOF, and Home Assistant misses the $\rho$ condition despite a
perfect SPOF-F1. What this establishes is a negative result about thresholds: a gate calibrated on
the synthetic corpus does not transfer as shipped, and its absolute cut-offs are domain-specific.

**What these five cases do and do not establish.** Four scoping conditions apply, and they matter
because this is the thesis's only evidence outside the generator.

*They are hand-built models of real architectures, not harvested artifacts.* What transfers is the
*topology and QoS structure* of a documented system as one author read it, not its runtime
behaviour. No second modeller has re-derived any model; `reproduce/model_agreement.py` implements the
re-modelling protocol, with per-type entity and per-relation edge Jaccard, for when one does.

*The ground truth is still simulated.* These correlations are between a structural or learned score
and a simulated impact label, produced by the same machinery as everywhere else in this thesis. No
incident record, operator judgement or observed failure enters. §9.2's construct-validity bound
applies unchanged: this is agreement with a model of harm, not with harm.

*They are small, and D4 forbids comparing them.* At 60 to 90 components (22–41 Applications) these
graphs are smaller than most synthetic scenarios, and because criticality is relative to a system's
own distribution (D4), the per-system $\rho$ values are separate within-system results, not points on
a shared scale.

*They cover two original paradigms, and neither RPC model is modelled natively.* Three models come
from publish–subscribe systems and two from RPC systems, but all five are encoded as
publish–subscribe graphs. What the five cases establish is the narrower claim that learned ranking
over the QoS-annotated graph transfers to architecture models written independently of the scenario
generator, well above every training-free score on the full population. They do not establish that
it ranks the components that actually propagate failures, and they do not settle whether performance
depends on regularities of our own generator (§9.3).

---

# 9. Discussion, Threats to Validity, and Conclusion

## 9.1 Interpretation

The results converge on a single message: for pre-deployment criticality analysis of pub-sub
middleware, *how* a component is critical is at least as important as *whether* it is — and the case
for graph learning is narrower, and more specific, than we expected when we set out. Four findings
carry this.

**First, learning pays in combination with closed-form ranking, not instead of it (RQ1).** Out of
distribution, over twelve held-out architectures, the pure learned engines lead the QoS-weighted
closed-form score only numerically: `HGT-QoS` reaches $\rho = 0.622$ and the untyped `GAT-QoS`
$0.635$, against $0.553$ for a `Topo-QoS` that needs no training, no labels and no transfer
assumption, and neither contrast is significant ($p = 0.266$ and $0.233$; §9.1.1, Table 25). The
engines that do significantly outperform it are the hybrids, which correct the closed-form score
with a learned residual: Hybrid-HGT reaches $\rho = 0.657$ ($+0.103$) and Hybrid-GAT $0.683$
($+0.130$), each on 11 of 12 folds and each surviving Holm correction. The reason is
complementarity. The learned engine loses exactly where the closed-form engine is strongest, most
sharply on the largest and densest fold (Enterprise, $0.426$ against $0.795$), and gains most where
it is weakest (Healthcare, IoT Smart City, ATM, Microservices), so `Topo-QoS`'s per-fold scores
range from $0.265$ to $0.810$ while the hybrids keep most of both strengths (Table 27). All of these
margins are measured against a baseline we first had to repair: `Topo-QoS` was computing no QoS
weighting whatsoever (§8.1), and until that was fixed it was `Topo-BL` wearing a different label.
Under LOSO, set identification no longer carries the case on its own: Overlap@$K$ moves only from
$0.388$ for `Topo-QoS` to $0.426$–$0.450$ for the learned and hybrid engines, and top-$K$ sets are
noisier than rankings in the labels themselves. It does separate the engines under zero-shot
transfer to the five open-source system models, where the pure learned engines reach
$\rho = 0.760$–$0.805$ and top-$K$ overlap $0.470$–$0.519$, against $0.511$–$0.526$ and $0.248$ for
the closed-form scores. The practical reading is therefore a three-way scope condition. For a
cheap ordering in a CI gate, QoS-weighted centrality remains a serviceable default. For
architectures that resemble the training corpus, a hybrid is worth its training cost on the evidence
here. For substantially different architectures, a pure learned engine with the QoS edge channel
transfers best, and because relation typing adds nothing at matched capacity, the untyped `GAT-QoS`
is the simpler choice. What we cannot yet demonstrate is that any engine ranks the components that
actually propagate failures: restricted to them, every predictor loses about half its correlation
(§9.1.1).

**Second, decomposition is worth having for reasons that are not accuracy (RQ2, §8.2; weighting
sensitivity from RQ3's robustness analysis, §8.3).** The dimension
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
edit can no longer be carried by an improving one. What that filter reveals is that the operators'
yield is highly topology-dependent — from 3 of 35 candidates on Autonomous Vehicle to 38 of 58 on IoT
Smart City — and that the resulting risk reductions are uniformly small. We regard this as the
correct outcome of an honest test rather than a failure of the mechanism: the previous design
reported a more favourable aggregate precisely because it never asked each edit to justify itself.

**Finally, automated quality gating operationalises these checks continuously (RQ4).** By evaluating
in-memory via the `MemoryRepository` and bypassing database round-trips, the framework runs
anti-pattern scans and counterfactual simulations in seconds (~5 s medium, ~40 s xlarge). That speed
makes the analyzer viable as a blocking CI/CD check. It is not yet sustainable as one: the gate is
absolute rather than delta-aware (§6.6), so it re-evaluates the full finding set on every run, and a
deliberately accepted single point of failure fails the build on every commit, indistinguishable
from a regression. Evaluating against the merge base and blocking only on newly introduced findings,
with a waiver register for accepted risk, is the change that would make the gate usable on a real
architecture (§9.3). The speed result is the part of this contribution least disturbed by the audit;
the gating semantics are the part still to build.

### 9.1.1 Where Graph Learning Helps, and Where It Does Not

> **Provenance.** Every figure in this subsection is transcribed from the current, artifact-reconciled
> JSS tables ([`sec7_results.md`](../jss/sections/sec7_results.md) Tables 7–10,
> [`supplementary.tex`](../jss/latex/supplementary.tex) §§S17, S23, S25 and the naive $2\times2$),
> which `reproduce/reconcile_manuscript.py` checks against their `results/` artifacts; the artifact
> behind each table is named in its caption. Per the source-integrity rules in
> [`outline.md`](outline.md#source-integrity), re-read each figure from its artifact when this
> subsection moves into the thesis. These figures come from the twelve-scenario corpus that §8.1
> reports. Predictor names follow the current manuscript: `HGT-QoS` is this draft's
> `HGL-QoS`; `GAT` and `GAT-QoS` are untyped GATs matched to HGT in parameter budget, which have no
> counterpart in Table 18.

The question this thesis set out to answer is whether graph learning is a useful instrument for
analysing and predicting failure impact in publish–subscribe systems, and if so whether
heterogeneous (typed) or homogeneous learning is the better choice in particular scenarios,
topologies or system scales. The answer is partly yes, and it is narrower than the one we expected.
Graph neural networks clearly outperform standard centrality and transfer well to system models they
never saw during training. On their own, however, they do not significantly outperform the
QoS-weighted closed-form score, and they only do so when combined with it. At matched capacity,
relation typing adds nothing measurable; what helps is the QoS edge channel. The claims about
topology type and system scale each rest on one to three folds or systems, so we state them as
hypotheses.

**Table 24. Claims about graph learning and the strength of the evidence for each.** "Hypothesis"
marks a pattern that rests on one to three folds or systems.

| Claim | Status | Evidence |
|---|---|---|
| GNNs outperform plain centrality on unseen architectures | Supported | LOSO $\rho$ 0.622–0.635 vs 0.349; zero-shot 0.760–0.805 vs 0.511 |
| A pure GNN outperforms the QoS-weighted closed-form score | Not significant | $+0.069$ / $+0.082$, $p = 0.266$ / $0.233$, 8/12 and 7/12 folds |
| A hybrid (GNN corrected by the closed-form prior) outperforms it | Supported | $+0.103$ / $+0.130$, 11/12 folds, Holm $p \le 0.0068$; survives the pooled 11-contrast correction |
| GNNs transfer to independently authored system models | Supported | `GAT-QoS` 0.805, `HGT-QoS` 0.760 vs 0.511–0.526; Overlap@$K$ 0.470 vs 0.248 |
| GNNs rank the components that actually propagate failures | Not shown | $\rho$ roughly halves on the active stratum for every predictor; zero-shot intervals span zero |
| Heterogeneous (typed) learning outperforms homogeneous learning | Not supported | Matched typing main effect $-0.014$, 4/12 folds, Holm $p = 0.94$ |
| The QoS edge channel improves learned ranking | Supported | $+0.073$ on 10/12 folds, typed or untyped |
| GNNs degrade on large, dense graphs | Hypothesis | One fold (Enterprise, 300 Applications) |
| GNNs help on irregular meshes and symmetric stars | Hypothesis | Healthcare, IoT Smart City, Microservices, ATM; EdgeX zero-shot |
| Performance depends on the original system's paradigm | Hypothesis | 3 pub-sub vs 2 RPC-derived models, all encoded as pub-sub graphs |

#### Are the learned engines successful?

**Table 25. The engines under LOSO and zero-shot transfer.** Spearman $\rho$ against $I^*(v)$ on
the Application population. LOSO: twelve synthetic folds, five seeds, CPU sweeps. Zero-shot: trained
on all twelve scenarios, evaluated on five hand-authored models of open-source systems. $\Delta\rho$
is paired by fold against `Topo-QoS`. Artifacts: `results/loso_hybrid_cpu.json`,
`results/loso_hybrid_gat_cpu.json`, `results/realworld_zeroshot_*_cpu.json`,
`results/omnibus_registered_holm.json`.

| Predictor | Kind | LOSO $\rho$ | $\Delta\rho$ vs `Topo-QoS` | Folds won | Zero-shot $\rho$ | Zero-shot PR-AUC |
|---|---|---:|---:|:---:|---:|---:|
| Topo | Unweighted centrality | 0.349 | $-0.204$ | 0/12 | 0.511 | 0.474 |
| `Topo-QoS` | Closed-form, QoS-weighted | 0.553 | — | — | 0.526 | 0.474 |
| `HGT-QoS` | Heterogeneous GNN | 0.622 | $+0.069$ | 8/12 | 0.760 | 0.713 |
| `GAT-QoS` | Homogeneous GNN | 0.635 | $+0.082$ | 7/12 | **0.805** | **0.790** |
| Hybrid-HGT | `HGT-QoS` + closed-form prior | 0.657 | $+0.103$ | 11/12 | 0.695 | 0.602 |
| Hybrid-GAT | `GAT-QoS` + closed-form prior | **0.683** | $\mathbf{+0.130}$ | 11/12 | 0.662 | 0.600 |

Three readings follow. First, the learned engines lead `Topo-QoS` numerically, but the intervals of
both contrasts span zero, and the registered GPU sweep agrees ($+0.085$, $p = 0.151$). Second, the
hybrids, which feed the rank-normalised closed-form score in as a prior and learn a logit-scale
correction to it, are the only engines in the study that significantly outperform closed-form
ranking. Third, anchoring to the prior trades transfer for in-distribution accuracy: on the five
system models, the pure learned engines lead the hybrids, and both lead every training-free score.
Identification separates the engines most clearly: zero-shot top-$K$ overlap averages 0.470 for
`HGT-QoS` against 0.248 for the closed-form scores.

`HGT-QoS` is also reported as $\rho = 0.638$ in the registered GPU sweep
(`results/loso_all_variants_v5.json`). Learned cells move by up to 0.172 in a fold mean across
devices and code revisions, so every learned figure must name its sweep, and comparisons are made
only within one sweep.

**Success is concentrated in separating inert from active components.** Between 21% and 52% of each
held-out Application population carries zero simulated impact. Restricted to components with
positive impact, every predictor keeps only 49–56% of its full-population correlation, with no
separation between learned and training-free families (`HGT-QoS`: 0.638 to 0.356, GPU sweep;
Supplementary §S25). In zero-shot transfer, `HGT-QoS` keeps a positive active-stratum mean
($\rho_{>0} = +0.236$) where every training-free score turns negative, but every interval spans zero
at five systems. The claim that GNNs rank the components that actually propagate failures is
therefore not established.

#### Homogeneous versus heterogeneous learning

**Table 26. The capacity- and channel-matched $2\times2$ (Amendment 2).** Cell means and effects,
twelve LOSO folds, five seeds, one CPU sweep. Holm correction across the three orthogonal
quantities. Artifacts: `loso_rq2_matched.json`, `loso_significance_rq2_matched.json`.

| | Homogeneous (GAT) | Heterogeneous (HGT) |
|---|---:|---:|
| No QoS channel | 0.563 (437,496 params) | 0.548 (434,620 params) |
| 16-D QoS channel | 0.635 (429,992 params) | 0.622 (434,620 params) |

| Quantity | $\Delta\rho$ | 95% CI | Won | $p_{\text{Holm}}$ |
|---|---:|:---:|:---:|---:|
| Typing (main effect) | $-0.014$ | $[-0.052, +0.023]$ | 4/12 | 0.940 |
| QoS channel (main effect) | $+0.073$ | $[+0.013, +0.120]$ | 10/12 | 0.127 |
| Typing $\times$ QoS interaction | $+0.001$ | $[-0.050, +0.042]$ | 6/12 | 0.940 |

**Relation-typed parameters add nothing beyond relation-typed inputs.** `GAT-QoS` receives each
edge's relation type in its 16-D edge vector, and at matched capacity it performs as well as
`HGT-QoS` in LOSO and better in transfer, on all five system models (0.805 vs 0.760). The QoS edge
channel is the working ingredient: it raises both designs by about 0.07 on 10 of 12 folds, and
significantly for the untyped pair ($+0.072$, CI $[+0.028, +0.109]$, $p = 0.016$). It also stabilises
training: the median within-fold seed spread of the untyped pair falls from 0.083 to 0.010.

**The earlier typing result was a capacity effect.** The unmatched comparison credited typing with
$+0.234$ ($p = 0.0005$, 12/12 folds). It compared HGT (434,620 parameters, 16-D edge channel)
against a 28,168-parameter GAT reading a scalar edge weight. At matched capacity, the same untyped
design rises from 0.317 to 0.563 and the typing gain disappears. Neither the Fisher-$z$ transform nor
robust seed aggregation detected the confound, because both held the same four unmatched arms fixed.
We record this as a methodological lesson: a factorial over model families is interpretable only
when capacity and input width are matched across its rows.

The typed model retains two advantages that this study did not measure. It can score infrastructure
entities and dependency edges through a relation-specific head, and it exposes per-relation attention
for explanation. Every learned result above is scored on Applications only, and the registered
directionality control (`HGT-QoS-U`) has not been run. In-distribution, `HGT-QoS` leads the
projection-based GATs (0.661 against 0.522 and 0.411, Supplementary §S17), but those arms read a
different substrate, so that comparison confounds message passing with multi-entity visibility and
supports no claim about typing.

#### Scenarios, topologies and scale

**Table 27. Per-fold LOSO $\rho$, ordered by the closed-form engine's score.** Fold score = mean over
five seeds, CPU sweeps; $n$ = Applications in the held-out fold. Artifacts:
`results/loso_hybrid_cpu.json`, `results/loso_hybrid_gat_cpu.json`.

| Held-out fold | $n$ | `Topo-QoS` | `HGT-QoS` | Hybrid-HGT | `GAT-QoS` | Hybrid-GAT |
|---|---:|---:|---:|---:|---:|---:|
| Real-Time Gaming | 75 | 0.810 | 0.789 | **0.837** | 0.685 | 0.825 |
| Enterprise | 300 | **0.795** | 0.426 | 0.735 | 0.407 | 0.768 |
| AV System | 80 | 0.753 | 0.704 | 0.782 | 0.732 | **0.793** |
| Logistics Fleet | 110 | 0.741 | 0.771 | 0.792 | 0.654 | **0.806** |
| Industrial SCADA | 140 | 0.650 | 0.684 | 0.758 | 0.721 | **0.768** |
| Financial Trading | 60 | 0.586 | 0.695 | 0.754 | 0.713 | **0.797** |
| Telecom RAN | 120 | 0.576 | 0.427 | 0.648 | 0.574 | **0.656** |
| Enterprise Integration (ESB) | 70 | 0.430 | 0.548 | 0.564 | **0.630** | 0.568 |
| Healthcare | 50 | 0.369 | 0.730 | 0.625 | **0.798** | 0.686 |
| IoT Smart City | 200 | 0.351 | 0.688 | 0.590 | **0.720** | 0.654 |
| ATM | 26 | 0.311 | **0.523** | 0.429 | 0.506 | 0.447 |
| Microservices | 90 | 0.265 | 0.475 | 0.366 | **0.479** | 0.429 |
| **Mean** | — | 0.553 | 0.622 | 0.657 | 0.635 | **0.683** |

**The learned and closed-form engines fail on different architectures.** `HGT-QoS` loses to
`Topo-QoS` on four folds, all where the closed-form engine is strongest: Real-Time Gaming,
Enterprise (0.426 vs 0.795), AV System and Telecom RAN (0.427 vs 0.576). Its largest gains come where
the closed-form engine is weakest: Healthcare, IoT Smart City, ATM and Microservices, $+0.210$ to
$+0.362$. The closed-form prior removes the failure mode on Enterprise (Hybrid-HGT 0.735,
Hybrid-GAT 0.768) and turns Telecom RAN into a win, at the cost of smaller gains on the weakest
folds. This complementarity is the mechanism behind the hybrids' result, and it is the most useful
guidance the corpus gives on engine choice:

- **Lightweight CI gate:** `Topo-QoS`. It needs no training and improves on unweighted centrality
  on 12 of 12 folds ($+0.204$).
- **Architecture resembling the training corpus:** Hybrid-HGT or Hybrid-GAT, which give the best
  LOSO ranking and are significant on 11 of 12 folds. Hybrid-HGT remains the registered
  recommendation because it transfers better.
- **Substantially different architecture:** a pure learned engine with the QoS channel, preferably
  the simpler untyped `GAT-QoS`, which transfers best.

On the zero-shot systems the same pattern appears in topological terms. On EdgeX, symmetric
adapter-to-broker stars create betweenness ties that collapse closed-form triage (Overlap@$K$ =
0.000), and the learned engines reach $\rho$ = 0.793–0.815. On the Online Boutique model the
closed-form scores are strongest (Topo 0.891) and the learned engines trail (0.710–0.750).

**Three patterns are hypotheses, not findings.**

- *Scale.* The claim that learned engines degrade on large graphs rests on Enterprise alone, the
  largest fold (520 nodes, 300 Applications) with the densest projection. The candidate mechanism is
  that three rounds of message passing cover less of a large graph. Graph size, density and
  prediction dispersion do not predict the winner in advance on this corpus. The only controlled
  scale sweep ([`atm_scale_sweep.py`](../../../reproduce/atm_scale_sweep.py), 29–444 components)
  measures anti-pattern detection, not the learned engines, and a matching sweep for the GNNs is the
  experiment that would settle it.
- *Topology.* Gains on dense irregular meshes (Microservices, ATM) and on symmetric stars (EdgeX)
  rest on one to three folds or systems each.
- *Architectural paradigm.* On the active stratum, $\rho_{>0}$ is positive on the three models of
  publish–subscribe systems and non-positive on the two modelled after RPC systems. Both RPC-derived
  models were encoded as publish–subscribe graphs with no synchronous edge and labelled by the same
  forward-propagating oracle, so the split cannot be attributed to call-tree semantics. Testing it
  requires synchronous edges in the schema and a backward-propagating oracle.

#### Why graph learning did not work where it failed

We record the unsuccessful cases because each one names a boundary of the method rather than a
tuning shortfall.

1. **The target is almost topological.** A topology-only relabelling recovers $I^*(v)$'s ordering at
   mean $\rho = 0.965$, and QoS acts mainly at its top-$K$ boundary. A strong closed-form competitor is
   therefore expected, and the learned engines cannot show that they read contract semantics: no
   topic in the corpus declares a deadline, so no oracle exercises deadline, durability or priority
   behaviour.
2. **The value relative to direct simulation is not demonstrated.** The labelling cascade runs
   2.0–17.7$\times$ faster (median 5.6$\times$) than the feature extraction the learned engines
   consume; the forward pass takes 56 ms against 239 s of structural analysis at 2,000 components.
   The motivations for learning instead of simulating (scoring unsimulated infrastructure, robustness
   to missing operational parameters, incremental caching) remain untested. Retargeting the LOSO
   contrasts on the discrete-event oracle $I_{\text{dyn}}$ is the experiment that would give the
   learned engines a task the cascade cannot trivially solve.
3. **Raw publish–subscribe graphs starve the features.** Messages route through topics and brokers,
   so Application betweenness vanishes on the native multigraph. The derived `DEPENDS_ON`
   projection is required, and its direction matters: inverting it flips the structural predictor's
   correlation from $\rho \approx +0.84$ to $-0.79$.
4. **Label noise and inert components bound what any predictor can show.** Oracle test–retest
   $\rho$ is 0.811–1.000 (median 0.982), but top-$K$ Jaccard falls to 0.370 on Logistics Fleet, so
   Overlap@$K$ margins are less stable than $\rho$ margins. Half of every predictor's correlation
   comes from separating inert from active components.
5. **Learned training is fragile.** An untuned 28,168-parameter GAT had a median within-fold seed
   spread of 0.298. Learned cells drift across code revisions and devices through a since-fixed
   PyTorch Geometric device-placement issue, stale checkpoint resumption and non-deterministic CUDA
   reductions, whereas every training-free cell reproduces across devices.
6. **Silent instrument defects invalidated earlier learning results.** Substituting RM scores as
   training labels made the labels a function of the input features and is now disabled by default.
   `Topo-QoS` applied no QoS weighting, the auxiliary RM target was all zeros on one training path,
   `FaultInjector` labels depended on the process hash seed, and the maintainability oracle
   $I_M(v)$ was identically zero. None raised an error. The methodology chapter treats these as a
   finding ([`threats_and_instrument_defects.md`](material/threats_and_instrument_defects.md)).
7. **External validity is narrow.** All twelve training scenarios come from one generator family.
   The five system models are small (22–41 Applications) and were written by one author, with no
   second modeller. No production incident data is used, and no published learned-criticality model
   (FINDER, DrBC) has been reproduced on this corpus.

**Summary.** On QoS-annotated publish–subscribe graphs, graph learning complements rather than
replaces closed-form structural analysis. Hybrids significantly improve in-distribution ranking, and
pure GNNs transfer best to unseen architectures. At matched capacity, heterogeneous typing confers no
measurable advantage over homogeneous attention; the QoS edge channel does. The unsuccessful cases
are specific boundaries, each with its sample size: the large, dense Enterprise fold, the active
stratum, the RPC-derived models, and a target that is almost entirely topological.

## 9.2 Threats to Validity

**Construct validity.** D1 and D2 define criticality as Quality-in-Use loss, and this study never
observes Quality-in-Use. The validation chain has two links, and only the first is measured:

$$\underbrace{\text{structural / learned score}}_{Q(v),\ \text{HGL}}
\;\xrightarrow{\ \text{\textcircled{1}}\ }\;
\underbrace{\text{simulated failure impact}}_{I^*,\ I_{\text{comp}},\ I_{\text{dyn}}}
\;\xrightarrow{\ \text{\textcircled{2}}\ }\;
\underbrace{\text{real Quality-in-Use loss}}_{\text{what D1 and D2 define}}$$

Link ① is what §8 reports: a real, falsifiable result. Link ② is not measured anywhere in this
paper — no user study, expert elicitation, or production incident record is used, and the simulator
is itself a *model* of stakeholder harm rather than an observation of it. The defensible claim is
therefore: *RMAV and the learned predictors track simulated failure impact, and simulated failure
impact is our stated operationalisation of Quality-in-Use loss.* The stronger claim — that these
scores track Quality-in-Use as stakeholders would report it — is not supported by anything here, and
we do not make it. Closing link ② requires evidence of a different kind: expert ranking studies on
the same topologies, or post-hoc comparison against incident records from a deployed system (§9.3).

Two qualifications keep this from being either overstated or unduly bleak. The ISO/IEC 25019
characteristics are not equally out of reach: Effectiveness and Efficiency are in principle
measurable from quantities the simulators already produce — delivery rate before and after a fault,
and the latency shift the discrete-event engine records — so link ② is partly closable by
re-summarising existing output on the Quality-in-Use axis rather than by new instrumentation.
Freedom from risk is blocked by the corpus rather than by the method: deadline and lifespan violation
counters exist and the harness has an oracle slot for them, but no topic in the scenario corpus
declares a deadline, so the counters never fire. Acceptability and Satisfaction are behavioural and
are not measurable by these means at all, which bounds what this construct can ever claim on them.
None of that has been *run*; it establishes measurability, not a measurement.

Because the ground truth is simulated rather than observed, the strongest claims we can make remain
comparative: which modelling choices perform better under identical conditions, not absolute
predictive accuracy in operation. Four further bounds apply, and we state them rather than leave them
implicit.

*The two oracles agree weakly.* $I^*(v)$ and $I_{\text{comp}}(v)$ correlate at mean $\rho = 0.395$
over the twelve LOSO topologies (range 0.083–0.653, Application population; §7.5). Results established against one do not transfer to claims measured against the other, which
constrains this paper's own internal cross-referencing: §5.4's library finding and §5.5's stratified
check are $I_{\text{comp}}$ results and are not evidence about the $I^*$-backed tables in §8.1.

*Six instrument defects were found and corrected during this revision.* All were silent — none
raised an exception or produced an obviously wrong number — and all are recorded here because each
had, or could have had, a published figure resting on it. The first two predate the corpus
regeneration of §7.1. First, the `Topo-QoS` baseline was applying no QoS weighting: $w(t)$ is declared
on the Topic node, the harness looked for it on the pub-sub relationship, and the generated
topologies carry none there, so every derived dependency edge kept a unit weight and the baseline
computed plain betweenness on all seven scenarios of the corpus at the time. It has been repaired to
resolve $w(t)$ from the shared Topic; the affected columns of the seven-scenario tables were
recomputed, the non-QoS variants were verified unchanged to machine precision, and every
twelve-scenario figure in §8.1 was produced after the repair. Second, HGT attention extraction
captured nothing, because `HGTConv` in the pinned PyTorch Geometric release exposes no
`return_attention_weights` argument and the extraction fell through its own error branch; attention
is now captured from the layer's own softmax, and the attention subgraph of Figure 6 is generated
from real per-edge $\alpha$ rather than an edge-weight fallback. We note that the second defect had masked a
third — the subgraph renderer itself raised on a `networkx` API change, which nothing had exercised
while the attention payload was empty.

Four further defects were found in a later pass that checked the implementation against this
manuscript directly, rather than against a specific reported number.

Third, and the one that changes reported figures, `FaultInjector`'s cascade iterated an unordered
Python set of subscribers while consuming seeded random draws for each; set iteration order in
Python is salted per-process by `PYTHONHASHSEED`, so the *same* requested seed could assign different
draws to different subscribers across processes, making $I^*(v)$ reproducible only within one
interpreter run, not across runs — exactly the kind of instability the seed-mean-and-standard-deviation
protocol of §5.1 and §7.5 was designed to average over, not diagnose. It has been repaired (the
iteration is now sorted); cross-process reproducibility was verified directly (identical $I^*(v)$
across five different `PYTHONHASHSEED` values, both on the synthetic corpus and on the real-world
Autoware sweep of §8.5). Two figures rest on the pre-fix labels and are flagged rather than silently
carried forward: the label test–retest $\rho$/Jaccard ceiling of §7.5, restated above with the
corrected, now process-independent values (previously reported as $\rho \in [0.928, 1.000]$, Jaccard
$\in [0.56, 1.00]$, both measured within a single process and so blind to this defect); and §8.5's
Autoware row, whose "sweep-to-sweep instability" was reported in an earlier draft as a property of
that graph and is corrected there to what it actually was. The seven-scenario tables of the earlier
revision, and every scenario in Table 13, were unaffected: both use `cascade_depth_limit=0`, the
setting at which the sixth defect below is provably a no-op, and neither exercises the code path this
defect lived in independently of that setting. The twelve-scenario tables of §8.1 reproduce at their
reported precision at fixed code, seeds and device.

Fourth, `extract_rmav_scores_dict` — the function that turns `PredictionService`'s RMAV output into
the GNN's auxiliary training target — keyed its lookup by an attribute (`component_id`) that the
underlying dataclass does not have (it has `id`), so every key fell through to the object's own
`repr()` string and the lookup silently returned nothing usable; the $0.1$-weighted RMAV-consistency
term of Table 17 was training against an all-zero target wherever this function was on the path. It
has been repaired to key by `id` first, matching its sibling function's already-correct convention.
Table 3/5/6/7's reported runs are unaffected: both evaluation harnesses (`cli/loso_evaluate.py`,
`cli/kfold_evaluate.py`) read RMAV scores through a different loader that never called the broken
function. Any GNN checkpoint trained via the standalone `cli/train_graph.py` entry point without an
explicit `--rmav` file did go through the broken path and trained with no RMAV supervision; that
entry point is not what produced the tables in this paper.

Fifth, the parallel worker in the prescription stage's per-edit verifier (§6.4) constructed its own
evaluator with default settings — layer `system`, no GNN checkpoint — regardless of what layer and
checkpoint the run was actually configured with, so a `--layer app` run would score every candidate
edit's counterfactual impact on the `system` layer while its baselines were measured on `app`. It has
been repaired to thread the configured layer and checkpoint into each worker. Table 13 is unaffected:
`reproduce/run_prescribe_all.py` runs at `layer="system"` with no checkpoint, which is exactly what
the unpatched default constructed, so the mismatch could not occur for the reported run.

Sixth, the post-loop computation of $I^*(v)$ read a `topic_loss` variable left over from the cascade's
last executed wave rather than recomputing it against the final set of failed components, so a
subscriber failure in the final wave was not reflected in that subscriber's own reported feed loss.
It has been repaired to recompute once more after the loop terminates. This defect is a no-op when
`cascade_depth_limit=0` (unlimited waves, the default and the setting `reproduce/` and the corpus
generation in §7.1 both use throughout): we verified this directly by running the pre-fix and
post-fix simulators against the same cached topologies and confirming a maximum absolute difference
in $I^*(v)$ of exactly $0$ across three scenarios. It is not a no-op under a finite
`cascade_depth_limit`, which no reported figure in this paper uses.

*A third of each system is unlabelled.* The cascade model cannot express the failure of a Topic or a
physical Node, leaving 30–47% of components per scenario without ground truth. Predictions for them
are produced but never validated. Broker labels are degenerate in three of seven scenarios for a
related reason. Any claim of coverage across "all five component types" would be unsupported, and
the per-type results report those strata as undefined rather than as zero.

*The labels bound what any predictor can show.* Across the twelve LOSO topologies the ground truth
agrees with itself at test–retest $\rho$ of 0.811–1.000 (median 0.982), well above every engine's
mean $\rho$, so the learned engines have not saturated the labels. Top-$K$ sets are noisier: the
cross-seed Jaccard of the ground truth's own top-$K$ set has median 0.847 and falls to 0.370 on
Logistics Fleet, and every top-$K$ metric inherits that churn. Between 21% and 52% of each held-out
Application population carries zero simulated impact, and restricted to the components that do
propagate failures, every predictor keeps only 49–56% of its correlation (§8.1, Table 20).

*The behavioural oracle evaluates delivery drop under calibrated contention.* $I_{\text{dyn}}$ carries the
construct-validity argument of §7.5, so the limits of what it measures bound that argument too. The discrete-event
engine resolves declared topic QoS policies (DDS reliability, history depth, durability replay, and transport priority)
and sizes each subscriber's `ServiceStation` to target operational utilization ($\rho = 0.65$, via $E[S_s] = \rho / \Lambda_s$)
so queue contention is active rather than underloaded. However, tail-latency degradation ($\Delta L_{p95}$) cannot serve
as an architectural criticality signal: empirical multi-seed measurements show that within-node seed variance
($\sigma_{\text{seed}} \approx 79.4\text{ ms}$) dwarfs across-node spread ($\sigma_{\text{across}} \approx 20.9\text{ ms}$),
yielding an uninformative signal-to-noise ratio ($\text{SNR} = 0.26$), compounded by the fact that dropping a chatty publisher
relieves contention and produces negative latency deltas. $I_{\text{dyn}}$ is therefore formulated strictly as unweighted
delivery rate loss ($\text{SNR} = 1.46$). Over the twelve LOSO topologies it agrees with $I^*$ at mean $\rho = 0.627$
(range 0.186–0.953), below $I^*$'s own test–retest, and $\rho^{+} = 0.429$ on components both oracles score non-zero,
so much of the agreement concerns which components are harmless. It is a convergent-validity probe rather than an
independent multidimensional oracle, and it is itself stochastic: on the three folds re-run across seeds its own
test–retest is 0.741–0.972.

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

Two matching conditions bound the learned comparisons. Substrate, training set, depth and early
stopping are matched across learned arms, and every typing conclusion rests on the capacity- and
channel-matched control (§8.2), not on the unmatched comparison it superseded. Message
directionality remains unmatched, because the registered `HGT-QoS-U` control has not been run. No
hyperparameter was tuned on an evaluation split.

*Artifact retention, and why learned figures name their sweep.* Table 18 and the sensitivity sweeps
of §8.3 regenerate exactly from stored result files, a claim that held only approximately before the
determinism defect above was fixed, since a re-run in a fresh process was not guaranteed to reproduce
a stored `FaultInjector` label exactly even at an unchanged seed. The seven-scenario
Leave-One-Scenario-Out table of the earlier revision was not so lucky: its result file was overwritten
during the revision, and the most recent retained log predated the baseline repair and recorded a
different ordering. That table has been withdrawn, not repaired. Table 20 replaces it with the
registered twelve-fold sweep, whose artifact is retained and whose conclusions an independent CPU
sweep reproduced (§8.1). The common mechanism in all three defects of this kind is that an
experiment's *evidence* and its *output* were allowed to come apart (a cached checkpoint, a
mismatched sample, an unretained result file), and in each case the number looked entirely ordinary.
The discipline this study now imposes, and did not impose soon enough, is that no figure enters the
manuscript unless the artifact that produced it is retained and the figure can be recomputed from it.

Retention is not the same as repeatability across environments. At fixed code, seeds and device,
every figure reproduces at its reported precision, and all training-free cells also reproduce across
devices. Learned cells do not: they move across code revisions and devices by up to 0.172 in a fold
mean (`HGT-QoS`: 0.041), through a since-fixed PyTorch Geometric device-placement issue, stale
checkpoint resumption and non-deterministic CUDA reductions. `HGT-QoS` therefore reads 0.638 on the
registered GPU sweep and 0.622 on the CPU sweep of §9.1.1, every learned figure names its sweep, and
every comparison is made within one sweep.

**External validity.** This is the weakest dimension of the study, and we regard it as the
highest-value follow-up (§9.3).

*The synthetic corpus comes from one generator family.* The twelve LOSO topologies span
autonomous vehicles, financial trading, healthcare, industrial SCADA, smart-city IoT, telecom RAN,
logistics, gaming, microservices, enterprise integration and air-traffic management, but all twelve,
and their code metrics, come from one generator. Leave-One-Scenario-Out evaluation therefore confirms
transfer across configurations of that generator, not across independently designed systems.

*The five system models are small and hand-authored.* Autoware.universe (ROS 2), EdgeX Foundry and
Home Assistant, plus meshes modelled after Online Boutique and Train-Ticket, were each written by one
author as typed multigraphs from public documentation (22–41 Applications). Brokers, QoS profiles,
code metrics and host specifications are partly assumed, and no second modeller has re-derived any
model; `reproduce/model_agreement.py` implements the re-modelling protocol for when one does. Two
models depart materially from their originals: the Online Boutique model is a 22-application pub-sub
mesh with four brokers, whereas the original is about eleven gRPC services with no broker, and the
Train-Ticket model represents its service-discovery server as a broker. Zero-shot, the learned engines
rank these models at $\rho = 0.760$ (`HGT-QoS`) and $0.805$ (`GAT-QoS`) against $0.511$–$0.526$ for
every training-free score (§9.1.1), but restricted to components that propagate failures every
interval spans zero at five systems. We read this as evidence that learned ranking transfers to
independently authored architecture models under simulated reachability, not to deployed systems.

*The framework's own release gate passes on one system model in five* (§8.5). Four of the five
fail at least one condition, most often SPOF-F1, so the gate's absolute cut-offs, calibrated on the
synthetic corpus, do not transfer as shipped. This is not a demonstration of production readiness.

*The paradigm split is untested.* $\rho_{>0}$ is positive on the three models of publish–subscribe
systems and non-positive on the two modelled after RPC systems. Both RPC-derived models are encoded
as publish–subscribe graphs with no synchronous edge and labelled by the same forward-propagating
oracle, so the split cannot be attributed to call-tree semantics; testing it requires synchronous
edges in the schema and a backward-propagating oracle (§9.3).

**Conclusion validity.** Criticality scores and simulated impact metrics exhibit heavy-tailed,
non-parametric distributions that violate normality assumptions. To prevent classification bias, we
apply non-parametric rank correlations (Spearman $\rho$), top-$K$ Jaccard metrics, and adaptive
box-plot thresholding ($Q3 + 1.5\,\mathrm{IQR}$) rather than parametric z-scores or arbitrary absolute cutoffs (§4.4).

LOSO folds share ten of their eleven training scenarios, so the paired tests across folds are
anti-conservative and every $p$-value is nominal; we read them alongside fold-level sign consistency
and bootstrap intervals. The primary contrast (`HGT-QoS` against `Topo-QoS`) was registered in the
repository before the twelve-fold harness produced any result, and three amendments registered
further contrasts, each before its run: the matched control (Amendment 2) and the two hybrids
(Amendments 5 and 6). We call this *registered* rather than pre-registered, because the plan has no
third-party timestamp. The sequence was adaptive, since Amendment 6 followed Amendment 2's result, so
we also pool all eleven registered contrasts under one Holm correction: both hybrid primaries remain
significant ($p_{\text{omni}} = 0.016$ and $0.034$), and no other registered contrast reaches
$\alpha = 0.05$ ($p_{\text{omni}} \ge 0.38$).

## 9.3 Limitations and Future Work

Several limitations point to concrete next steps, ordered here by how much they would change the
paper's claims.

**Real-world deployment validation and HIL execution.** Section 8.5 narrows the external-validity gap
on three real-world open-source software architectures (Autoware.universe ROS 2, the Cloud-Native
Microservices Mesh, and Train-Ticket), but does not close it: the graphs are hand-transcribed, their
ground truth is still simulated, and none clears the framework's own validation gate in full.
Validating predictions against runtime hardware-in-the-loop (HIL) fault injection on physical
testbeds, and against harvested rather than transcribed architectures, remains the highest-value
follow-up.

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

**Remediation is verified but not yet demonstrably effective.** The per-edit acceptance filter of
§6.4 is implemented, which removes the possibility of an unverified regressing edit being applied
(§6.7). What it does not do is make the operators work: the risk reductions it achieves are real but
small ($+0.0025$ to $+0.0158$ SRI), and the operator set is narrow enough that its yield depends
heavily on whether the topology happens to contain a fan-out bottleneck. Expanding the operator set —
along with deriving the acceptance multiplier $\kappa$ from broader multi-seed variance data rather
than fixing it at 1.0 — is needed before the prescriptive stage can claim practical value. One
specific gap follows: verification currently admits *singletons*, each simulated on a graph
containing that edit alone, so verifying subsets rather than single edits is required before an
accepted policy can be called compositionally safe. That is engineering work rather than an open
research question, and we flag it as the immediate next step for this stage.

**The CI/CD gate is absolute, not yet delta-aware.** §6.6 implements exit-code gating on a
candidate topology's full finding set; it does not yet diff that set against a merge-base topology,
so an architecture carrying an intentional, previously-accepted risk (a sole-source feed, a
deliberately unreplicated legacy broker) fails the build on every commit rather than only when a
change introduces something new. A delta-aware gate — evaluate candidate and merge-base topologies,
block only on findings absent from the baseline — together with a waiver register recording accepted
risk (entity, rule, expiry) so it stays auditable rather than silently re-suppressed on every run, is
the fix; neither is implemented today, and closing this gap is engineering work rather than an open
research question, similar in kind to the remediation gap above. A related, smaller gap is that
`detect_antipatterns.py` itself requires a live Neo4j connection even though the underlying analysis
machinery is already usable through the database-free `MemoryRepository` — wiring that path into the
packaged CLI is a prerequisite for running the gate without a database in an actual CI job.

**Edge-level ground truth is bounded by the cascade model.** Edge criticality is now measured by
removal rather than inferred from endpoints (§8.2), but the cascade routes no traffic over `RUNS_ON`
or `CONNECTS_TO` relations, so those edges measure as exactly zero regardless of their structural
role. Extending the cascade to express infrastructure-layer failure would close the same gap that
leaves Topic and physical Node components unlabelled.

**Relationship attribution is defined but not validated, and closing that gap is out of scope for
this submission.** §4.7 gives D2 a measure with the same signature as D1, implemented and computed on
every derived dependency. What it does not have — and, as the framework is currently constructed,
cannot have — is the correlation-style evidence §8.1 supplies for nodes: attribution is scored on
`DEPENDS_ON` edges while the removal oracle severs raw structural edges, so the two are never defined
on a common population (§4.7). Re-simulating directly on the derived graph is not an available fix,
since the independence guarantee (§5.3) requires the simulator to operate only on
$G_{\text{structural}}$. The one route that respects that guarantee is to track, for each derived
edge, which raw edges mediate it, and aggregate their measured impact onto it — but the mediating
relations are many-to-many, so this requires a real modelling decision (how to aggregate) rather than
a mechanical lift, and we leave it for future work rather than attempt it here. Until it is done, the
relationship half of the framework rests on construction rather than on measurement, which we regard
as the most significant open gap in the diagnostic path.

**Finally, the endpoint for all of the above** is calibration against observed failure data from
instrumented deployments, which would convert this paper's comparative claims into absolute ones.

## 9.4 Conclusion

We presented Software-as-a-Graph, a pre-deployment Static System Analysis (SSA) framework that
models distributed pub-sub middleware as a typed, weighted, directed multigraph and analyzes it
along two coupled axes: multi-dimensional quality attribution, which decomposes each component's
criticality into orthogonal, interpretable RMAV dimensions (integrating local code quality metrics),
and failure-impact analysis, which predicts cascade impact with the interpretable composite, a
QoS-weighted closed-form score, and learned heterogeneous and homogeneous graph neural networks,
validated against discrete-event simulation under a
strict input–label independence guarantee. A prescriptive remediation stage generates topology-level
hardening edits from structure alone and verifies each one individually against the canonical
simulator, admitting it only when its benefit exceeds the simulator's own seed noise at every
propagation threshold; under that filter 162 of 332 candidates survive across the seven scenarios,
but the resulting risk reductions are small ($+0.0025$ to $+0.0158$ SRI) and concentrated in the two
topologies with pronounced fan-out structure, which we report as the substantive — and qualified —
result of that stage (§6.4, §6.7).

Integrated directly into pipelines as a blocking CI/CD Quality Gate, the framework evaluates a
candidate topology in seconds, bridging the "Architecture-Code Gap" at commit time; the gate is
absolute rather than delta-aware, so blocking only newly introduced findings against the merge base
remains future work (§9.3). Across twelve synthetic architectures and models of five open-source systems,
the framework establishes a scope condition on where graph learning pays. The QoS-aware
representation carries most of the signal: QoS-weighted centrality outperforms unweighted centrality
on every held-out architecture ($\rho = 0.553$ vs $0.349$), measured only after repairing that
baseline, which had been computing no QoS weighting at all. Learned engines alone are statistically
on par with it out of distribution, but hybrids that learn a correction to it significantly
outperform it ($\rho = 0.657$ and $0.683$), and pure learned engines transfer best to independently
authored system models ($\rho = 0.760$–$0.805$ against $0.511$–$0.526$). At matched capacity,
heterogeneous typing adds nothing over homogeneous attention, and the QoS edge channel is what
improves learned ranking. None of the engines is yet shown to rank the components that actually
propagate failures. Alongside that, measuring edge criticality by
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

[16] ISO/IEC 25010:2023, "Systems and software engineering — Systems and software Quality
Requirements and Evaluation (SQuaRE) — Product quality model," International Organization for
Standardization, 2023.

[17] ISO/IEC 25019:2023, "Systems and software engineering — Systems and software Quality
Requirements and Evaluation (SQuaRE) — Quality-in-use model," International Organization for
Standardization, 2023.

[18] A. Basiri, N. Behnam, R. de Rooij, L. Hochstein, L. Kosewski, J. Reynolds, C. Rosenthal,
"Chaos engineering," *IEEE Software*, vol. 33, no. 3, pp. 35–41, 2016.

[19] J. Humble, D. Farley, *Continuous Delivery: Reliable Software Releases through Build, Test, and
Deployment Automation*, Addison-Wesley, 2010.

[20] L. Chen, "Continuous delivery: Huge benefits, but challenges too," *IEEE Software*, vol. 32,
no. 2, pp. 50–54, 2015.

[21] SonarSource, "Clean as You Code," SonarQube documentation, 2024. [Online].

[22] J. Garcia, D. Popescu, G. Edwards, N. Medvidovic, "Toward a catalogue of architectural bad
smells," in *Proc. 5th Int. Conf. on the Quality of Software Architectures (QoSA)*, LNCS 5581, 2009,
pp. 146–162.

[23] J. Garcia, D. Popescu, G. Edwards, N. Medvidovic, "Identifying architectural bad smells," in
*Proc. 13th European Conf. on Software Maintenance and Reengineering (CSMR)*, 2009, pp. 255–258.

[24] D. Taibi, V. Lenarduzzi, "On the definition of microservice bad smells," *IEEE Software*,
vol. 35, no. 3, pp. 56–62, 2018.

[25] N. Dragoni, S. Giallorenzo, A. L. Lafuente, M. Mazzara, F. Montesi, R. Mustafin, L. Safina,
"Microservices: Yesterday, today, and tomorrow," in *Present and Ulterior Software Engineering*,
Springer, 2017, pp. 195–216.

[26] R. C. Martin, *Agile Software Development: Principles, Patterns, and Practices*, Prentice Hall,
2003.

[27] W. Cunningham, "The WyCash portfolio management system," in *Addendum to the Proc. Conf. on
Object-Oriented Programming Systems, Languages, and Applications (OOPSLA)*, 1992, pp. 29–30.

[28] Z. Li, P. Avgeriou, P. Liang, "A systematic mapping study on technical debt and its
management," *Journal of Systems and Software*, vol. 101, pp. 193–220, 2015.

[29] S. R. Chidamber, C. F. Kemerer, "A metrics suite for object oriented design," *IEEE
Transactions on Software Engineering*, vol. 20, no. 6, pp. 476–493, 1994.

[30] T. J. McCabe, "A complexity measure," *IEEE Transactions on Software Engineering*, vol. SE-2,
no. 4, pp. 308–320, 1976.

[31] N. Fenton, J. Bieman, *Software Metrics: A Rigorous and Practical Approach*, 3rd ed., CRC
Press, 2014.

[32] A. Avizienis, J.-C. Laprie, B. Randell, C. Landwehr, "Basic concepts and taxonomy of dependable
and secure computing," *IEEE Transactions on Dependable and Secure Computing*, vol. 1, no. 1,
pp. 11–33, 2004.

[33] L. Bass, P. Clements, R. Kazman, *Software Architecture in Practice*, 3rd ed., Addison-Wesley,
2012.

[34] R. Kazman, M. Klein, M. Barbacci, T. Longstaff, H. Lipson, J. Carriere, "The architecture
tradeoff analysis method," in *Proc. 4th IEEE Int. Conf. on Engineering of Complex Computer Systems
(ICECCS)*, 1998, pp. 68–78.

[35] S. Newman, *Building Microservices: Designing Fine-Grained Systems*, O'Reilly Media, 2015.

[36] R. Albert, H. Jeong, A.-L. Barabási, "Error and attack tolerance of complex networks,"
*Nature*, vol. 406, pp. 378–382, 2000.

[37] A. E. Motter, Y.-C. Lai, "Cascade-based attacks on complex networks," *Physical Review E*,
vol. 66, 065102(R), 2002.

[38] U. Brandes, "A faster algorithm for betweenness centrality," *Journal of Mathematical
Sociology*, vol. 25, no. 2, pp. 163–177, 2001.

[39] M. E. J. Newman, *Networks: An Introduction*, Oxford University Press, 2010.

[40] T. N. Kipf, M. Welling, "Semi-supervised classification with graph convolutional networks," in
*Proc. Int. Conf. on Learning Representations (ICLR)*, 2017.

[41] W. L. Hamilton, R. Ying, J. Leskovec, "Inductive representation learning on large graphs," in
*Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017, pp. 1024–1034.

[42] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio, "Graph attention
networks," in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2018.

[43] M. Fey, J. E. Lenssen, "Fast graph representation learning with PyTorch Geometric," in *ICLR
Workshop on Representation Learning on Graphs and Manifolds*, 2019.

[44] J. Kreps, N. Narkhede, J. Rao, "Kafka: A distributed messaging system for log processing," in
*Proc. 6th Int. Workshop on Networking Meets Databases (NetDB)*, 2011.

[45] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, W. Woodall, "Robot Operating System 2: Design,
architecture, and uses in the wild," *Science Robotics*, vol. 7, no. 66, eabm6074, 2022.

[46] S. Kato, S. Tokunaga, Y. Maruyama, S. Maeda, M. Hirabayashi, Y. Kitsukawa, A. Monrroy,
T. Ando, Y. Fujii, T. Azumi, "Autoware on board: Enabling autonomous vehicles with embedded systems,"
in *Proc. ACM/IEEE 9th Int. Conf. on Cyber-Physical Systems (ICCPS)*, 2018, pp. 287–296.

[47] X. Zhou, X. Peng, T. Xie, J. Sun, C. Ji, W. Li, D. Ding, "Fault analysis and debugging of
microservice systems: Industrial survey, benchmark system, and empirical study," *IEEE Transactions
on Software Engineering*, vol. 47, no. 2, pp. 243–260, 2021.

[48] Google Cloud Platform, "Online Boutique: A cloud-native microservices demo application,"
software artifact. [Online].

[49] F. Wilcoxon, "Individual comparisons by ranking methods," *Biometrics Bulletin*, vol. 1, no. 6,
pp. 80–83, 1945.

[50] B. Efron, R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.

[51] C. Spearman, "The proof and measurement of association between two things," *American Journal
of Psychology*, vol. 15, no. 1, pp. 72–101, 1904.

[52] U.S. Department of Defense, "MIL-STD-498: Software Development and Documentation,"
Military Standard, 1994.

[53] D. Chen, Y. Lin, W. Li, P. Li, J. Zhou, X. Sun, "Measuring and relieving the over-smoothing
problem for graph neural networks from the topological view," in *Proc. AAAI Conference on Artificial
Intelligence*, 2020, pp. 3438–3445.

[Anon-A] Authors' prior work on multi-layer graph dependency analysis for publish–subscribe systems.
*Citation withheld for double-anonymised review.*

---

# Declarations

**CRediT authorship contribution statement.** *[Omitted for double-anonymised review. To be completed
on acceptance with the standard CRediT roles: Conceptualization; Methodology; Software; Validation;
Formal analysis; Investigation; Data curation; Writing — original draft; Writing — review and
editing; Visualization; Supervision.]*

**Declaration of competing interest.** The authors declare that they have no known competing
financial interests or personal relationships that could have appeared to influence the work reported
in this paper.

**Funding.** *[Omitted for double-anonymised review.]*

**Data availability.** The twelve synthetic LOSO datasets (eleven evaluation scenarios and the ATM
case study), their generator configurations, and the manifest of canonical dataset hashes
(`data/scenarios/MANIFEST.json`) are included in the replication package; every synthetic dataset
regenerates byte-identically from its configuration, which continuous integration verifies. The five
open-source system models of §8.5, their configurations and their adapter
(`saag/adapters/realworld_adapter.py`) are included on the same terms. Result artifacts are provided
for the in-distribution evaluation and its significance tests (Tables 18–19), the registered
Leave-One-Scenario-Out sweep (Table 20), the CPU sweeps behind the hybrid and matched-control
contrasts and the omnibus Holm correction (§9.1.1), and the zero-shot evaluation of §8.5 (Table 23),
together with the trained checkpoints. The sensitivity sweeps of §8.3, the edge-removal measurement
of §8.2 and the remediation sweep of §6.7 were measured on the earlier seven-scenario corpus, and
their artifacts are provided as they stand. **One withdrawal is recorded rather than glossed:** the
per-fold artifact behind the seven-scenario Leave-One-Scenario-Out table of an earlier revision was
not retained; that table has been withdrawn and replaced by the registered twelve-fold sweep, whose
artifact is retained (§8.1, §9.2). Learned cells move across devices and code revisions (§9.2), so
new runs should be compared within one sweep rather than against the published cells. A link to the
archived package will be supplied on acceptance.

**Declaration of generative AI use.** *[To be completed by the authors in accordance with the
journal's policy.]*
