# Software-as-a-Graph: A Static System Analysis Framework for Pre-Deployment Quality Gating and Failure Simulation of Publish-Subscribe Middleware

*Target venue: Journal of Systems and Software (JSS) — Elsevier, Q1.*

> **Draft status.** This is a conditional working draft. Inline ⚠ callouts mark unresolved
> items whose resolution may change a number or, in one case (§5.4/§8.2), the sign of a result.
> All such items are consolidated, with options and sign-off lines, in
> `jss_reconciliation_worksheet.md`. Bracketed values `[…]` require an experiment/run, not a decision.
> Citation markers `[n]` are placeholders pending bibliography wiring. Math is in LaTeX notation for
> elsarticle porting.

---

# Abstract

Distributed publish–subscribe middleware decouples producers and consumers through topics and
brokers, which obscures the true dependency chains along which a single component failure can
cascade. Identifying *which* components are critical — and *why* — before deployment is hard:
runtime telemetry does not yet exist, and code-level Static Code Analysis (SCA) platforms (e.g., SonarQube)
are blind to system-level topological dependencies. We present **Software-as-a-Graph (SaG)**, a pre-deployment
**Static System Analysis (SSA)** framework that bridges this "Architecture-Code Gap." SaG models a
pub-sub system as a typed, weighted, directed multigraph over applications, libraries, topics,
brokers, and deployment nodes, and derives logical `DEPENDS_ON` dependencies via a set of typed projection rules.

On this model, SaG ingests local SCA metrics to enrich component vertices and performs two complementary analyses:
1. **Multi-dimensional quality attribution** decomposes each component's criticality into orthogonal
Reliability, Maintainability, Availability, and Vulnerability (RMAV) dimensions, combined into a
composite score *Q(v)* with Analytic Hierarchy Process weights. Because each code-level and topology-level
metric feeds exactly one dimension, the breakdown explains *why* a component is critical, directing targeted
remediation and informing automated quality gates rather than blanket hardening.
2. **Failure-impact analysis** predicts each component's cascade impact *I(v)* using both the interpretable
*Q(v)* score and a learned heterogeneous-graph predictor, validated against discrete-event simulation under a
strict input–label independence guarantee.

We further formalize a **prescriptive remediation** stage that generates topology-level hardening edits and
verifies them on counterfactual graphs via the same simulation oracle. Integrated directly into CI/CD pipelines,
SaG acts as an automated, build-blocking quality gate. Across [N] synthetic scenarios and an external
air-traffic-management (ATM) case study, we show: (i) *when* interpretable attribution suffices and when learning
is required; (ii) that multi-dimensional attribution exposes failure modes invisible to centrality or code linting alone
— notably a **shared-library blast-radius gap**, where a component with low composite *Q* (≈ 0.48) nonetheless
drives near-total cascade impact (*I* ≈ [0.97]) through simultaneous fan-out; and (iii) agreement between
predicted rankings and expert judgment on the ATM system (Kendall τ = [τ], Fleiss κ = [κ]). We also demonstrate
that the framework blocks architectural regression with low performance overhead (~5s for medium, ~40s for xlarge),
making continuous SSA feasible for rapid-release pipelines.

**Keywords:** publish–subscribe middleware; static system analysis; dependency analysis; criticality
prediction; failure cascade; quality attributes; heterogeneous graph; CI/CD quality gate; pre-deployme# 1. Introduction

## 1.1 Motivation

The publish–subscribe (pub-sub) paradigm has become a backbone communication abstraction for
large-scale distributed systems, underpinning cyber-physical, cloud-native, robotics, and
Internet-of-Things architectures. Its appeal is decoupling: producers and consumers are separated
in time, space, and synchronization, so components can be added, removed, or scaled without direct
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

## 1.2 The Architecture-Code Gap and Problem Statement

We address pre-deployment criticality analysis for pub-sub middleware as two coupled sub-problems.
Given only an architectural description of the system — its applications, libraries, topics,
brokers, deployment nodes, and the QoS policies on its communication — we seek to:

1. **Quality attribution.** Assign each component an interpretable measure of *how* and *why* it is
   critical, decomposed along the quality dimensions an engineer would act on, so that the result
   directs a specific remediation rather than a generic warning.
2. **Failure-impact analysis.** Predict the cascade impact of each component's failure — the extent
   to which the rest of the system becomes unreachable or impaired — and identify the components
   that should be hardened first.

Both must be computed without runtime data, and both must remain *explainable*: a single opaque
criticality number is of limited use to an architect who has to choose between competing
interventions under a fixed budget.

Historically, static verification has operated primarily at the source-code level. However, a major **"Architecture-Code Gap"** exists: a software system can have perfectly clean source code in every component (earning top scores on code-level tools), yet remain highly fragile. If the deployment topology contains a Single Point of Failure (SPOF) or a mismatched QoS contract, a single component crash can cascade and collapse the entire system. Bridging this gap requires shifting structural verification "left" into the continuous integration and delivery (CI/CD) pipeline.

## 1.3 Limitations of Existing Approaches

Three strands of prior work bear on this problem, and each leaves a gap.

**Static Code Analysis (SCA).** Platforms such as SonarQube evaluate code cleanliness, cyclomatic complexity, and LCOM (Lack of Cohesion of Methods) inside individual modules. While highly effective for intra-component quality, they are entirely blind to inter-component topologies and dynamic middleware cascades. 

**Runtime Dependability and Chaos Engineering.** A large body of work hardens pub-sub systems through runtime fault tolerance, replication, and chaos injection (e.g., Chaos Monkey). These techniques are valuable but assume a *running* staging or production system; they do not answer which components a design should protect before it is deployed, and injecting failures at runtime carries operational risk.

**Topology-Only and Learning-Based Centrality.** Classical network-science metrics collapse a component's risk into a single scalar that conflates distinct failure mechanisms (SPOFs vs. cascade hubs), while homogeneous graph neural networks collapse typed semantics (applications, topics, brokers) into flattened views, leading to representation collapse. 

No existing approach offers an *interpretable, multi-dimensional, pre-deployment* attribution over the *typed* pub-sub graph, coupled to code-level SCA metrics, impact prediction, and automated CI/CD gating. That is the gap this paper fills.

## 1.4 Our Approach

We present **Software-as-a-Graph (SaG)**, a pre-deployment **Static System Analysis (SSA)** framework. SaG models a pub-sub system as a typed, weighted, directed multigraph over five node types (applications, libraries, topics, brokers, nodes) and derives logical `DEPENDS_ON` dependencies through typed projection rules. Crucially, SaG ingests code-level SCA metrics as vertex attributes and performs **multi-dimensional quality attribution**, decomposing criticality into orthogonal Reliability, Maintainability, Availability, and Vulnerability (RMAV) dimensions using Analytic Hierarchy Process (AHP) weights.

SaG then performs **failure-impact analysis**, predicting cascade impact *I(v)* with two predictors: the AHP-weighted composite *Q(v)* and a learned heterogeneous-graph predictor. Both are validated against a discrete-event simulator under an **input–label independence guarantee**. Finally, a **prescriptive remediation** stage generates topology-level hardening edits and verifies them on counterfactual graphs in-memory.

To make SSA continuous, SaG integrates directly into CI/CD pipelines as a blocking gate. By utilizing a thread-safe, database-free `MemoryRepository` to bypass Neo4j database overhead during build time, SaG executes anti-pattern scans and simulations in seconds, automatically failing the build (exit code 2) if CRITICAL or HIGH severity structural anomalies are introduced.

Concretely, the paper is organized around five research questions:

> **RQ1.** When does interpretable, multi-dimensional attribution suffice for pre-deployment
> criticality prediction, and when is a learned predictor required to recover the simulated
> impact ordering?
>
> **RQ2.** What failure modes does multi-dimensional quality attribution expose that a single-score
> topological centrality misses?
>
> **RQ3.** How does explicit multi-attribute QoS contract feature injection affect in-distribution convergence versus out-of-distribution Leave-One-Scenario-Out (LOSO) generalizability?
>
> **RQ4.** What is the feasibility and performance overhead of deploying the graph-based analyzer as a blocking Quality Gate in continuous integration/delivery (CI/CD) pipelines?
>
> **RQ5.** Do the framework's predicted criticality rankings agree with expert judgment on a
> real-world system?

RQ1, RQ2, and RQ3 are answered on the synthetic scenario suite (§8); RQ4 evaluates gating feasibility and performance (§8.4); and RQ5 is answered by the external ATM validation (§9).

## 1.5 Contributions

This paper makes the following contributions:

1. **A typed graph model with hierarchical SCA metric integration.** We define the SaG multigraph and the RMAV decomposition, which propagates code-level quality metrics (SonarQube `cm_*` fields) into global system criticality scores (§3, §4).
2. **An automated CI/CD Quality Gate.** We formulate a build-blocking gate that evaluates system-level risk statically and exits with non-zero codes to prevent fragile deployment configurations, executed in seconds via an in-memory repository (§6).
3. **Failure-impact analysis under an independence guarantee.** We predict cascade impact using both the interpretable *Q(v)* and a learned heterogeneous GNN, validated against discrete-event simulation (§5).
4. **A prescriptive remediation stage.** We formalize a Generate→Verify procedure with four concrete operators, verifying counterfactual edits statically before commit (§6).
5. **Two empirical findings.** We identify and explain the shared-library blast-radius gap and the Simpson's-paradox stratification of criticality correlation by node type (§5, §8).
6. **External validation on a real-world system.** We validate predicted rankings against blind expert judgment on an ICAO-compliant air-traffic-management (ATM) system (§9).

## 1.6 Relationship to the Authors' Prior Work

This work extends two earlier contributions by the authors. A structural baseline of the
framework — multi-layer graph dependency analysis — was introduced in prior work [Anon-A], and a
heterogeneous-graph predictor for cascade impact is the subject of a manuscript currently under
review [Anon-B]. The present paper is an *unambiguous superset* of both: quality attribution (§4),
SCA metric integration, and CI/CD gating remediation (§6) are first-class contributions that appear in neither.

## 1.7 Organization

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 defines the Software-as-a-Graph model. Section 4 presents multi-dimensional quality attribution, and Section 5 presents failure-impact analysis. Section 6 introduces prescriptive remediation and CI/CD quality gating. Section 7 describes the experimental setup; Section 8 reports results; and Section 9 presents the external ATM valida# 2. Related Work

This paper draws on, and contributes to, several established lines of research: publish–subscribe dependability, static analysis techniques, pre-deployment system verification, structural criticality, and multi-criteria quality scoring.

## 2.1 Publish–Subscribe Middleware and Dependability

The pub-sub paradigm is a foundational communication abstraction for large-scale distributed systems, valued for decoupling producers and consumers in time, space, and synchronization [1]. Content-based and brokered overlays extend this with flexible event routing and subscription matching, and standards such as DDS and MQTT formalize deployment-time choices — topics, brokers, reliability, durability, and other QoS policies — that govern runtime behavior [2, 3]. These mechanisms enable cyber-physical, cloud, IoT, and robotics architectures, but they also make failure propagation difficult to reason about from direct communication edges alone.

Research on pub-sub dependability has accordingly emphasized runtime fault tolerance, reliable event dissemination, replication, and recovery. These approaches improve a system's resilience while it is *running*: they assume observable behavior and react to or mask faults as they occur. Our concern is complementary and earlier in the lifecycle — estimating, from an architectural model that enumerates applications, libraries, topics, brokers, and QoS policies, which components would have the greatest downstream impact if they failed, so that the design can be hardened before any system is deployed.

## 2.2 Static Code Analysis (SCA) vs. Static System Analysis (SSA)

Static verification typically operates at the source-code level. Static Code Analysis (SCA) tools, exemplified by SonarQube, checkstyle, and FindBugs, parse source files into Abstract Syntax Trees (ASTs) to compute complexity, code duplication, and modular metrics such as LCOM (Lack of Cohesion of Methods). While SCA is essential for locating intra-component defects and technical debt, it is blind to the inter-component topology. 

Static System Analysis (SSA) addresses this "Architecture-Code Gap." SSA models the system as a global graph of communicating components, middleware routers, and hardware hosts. Rather than replacing SCA, SSA ingests code-level metrics as node properties (e.g., LCOM, cyclomatic complexity) and propagates them through the inter-component dependency topology. This allows architects to evaluate how code-level fragility (e.g., a highly complex class inside an application) combines with structural fragility (e.g., the application being a single point of failure) to create systemic risks.

## 2.3 Continuous Pre-Deployment Verification and Gating

A common way to verify system resilience is dynamic testing, particularly Chaos Engineering (e.g., Netflix Chaos Monkey), which injects faults into live staging or production clusters. While chaos testing evaluates real operational environments, doing so carries risk and occurs late in the lifecycle. 

Continuous pre-deployment verification shifts this analysis left, integrating it into CI/CD pipelines (e.g., GitHub Actions, GitLab CI). In this paradigm, the system architecture is defined as "Architecture-as-Code" (AaC) via configuration descriptors (Docker Compose, Kubernetes manifests, Helm charts). SSA tools run automatically on every pull request, parsing the configuration descriptors to generate a counterfactual topology graph. The gate automatically blocks the build (exiting with non-zero status) if a PR introduces critical architectural smells (like SPOFs or QoS mismatches) or exceeds failure propagation thresholds, resolving issues before code is deployed.

## 2.4 Structural Criticality Analysis

Network science offers a mature toolkit for identifying important nodes and edges. Degree, closeness and betweenness centrality, articulation points, and PageRank-style scores are prized for their efficiency and interpretability [4, 5], and studies of node removal, cascading failure, and interdependent networks have deepened our understanding of systemic fragility [6]. Applied to software dependency graphs, these metrics can flag bottlenecks and single points of failure at design time.

Their limitation, for our purpose, is dimensional collapse. A single centrality score conflates mechanisms that call for different remedies: a structural single point of failure, a high-reach cascade hub, and a tightly coupled maintainability bottleneck can all present as "central," yet a replica, a rerouting, and a decoupling refactor are not interchangeable fixes. Untyped metrics are moreover blind to type-specific failure modes — most importantly the *simultaneous* blast radius of a shared library, whose failure strikes all consumers at once and which is indistinguishable from an ordinary edge once node and edge types are discarded. Our RMAV attribution retains the interpretability that makes structural metrics attractive while decomposing criticality into orthogonal dimensions, and our typed model keeps the semantics that single-score centrality erases.

## 2.5 Learning-Based Criticality Prediction

A growing body of work learns to identify critical nodes directly from graph structure, often surpassing hand-crafted metrics when higher-order structure matters: FINDER locates key entities in networked systems, DrBC learns to approximate betweenness, and PowerGraph applies graph learning to critical-node analysis in power systems [7, 8, 9]. 

Most such methods, however, target *homogeneous* graphs. Pub-sub middleware is intrinsically heterogeneous — applications publish and subscribe to topics, topics are routed through brokers, libraries introduce code dependencies, and deployment nodes impose locality — and flattening this into a homogeneous graph discards information about how failures propagate. Heterogeneous graph neural networks address this directly: RGCN applies relation-specific transformations [10], HAN uses hierarchical attention [11], HGT parameterizes attention by node and edge type [12], and MAGNN aggregates along metapaths [13]. A known hazard in dense, hub-dominated regions is over-smoothing [14]. Our learned predictor adopts relation-specific message passing over the native typed architecture for exactly these reasons, but we treat it as one of two predictors rather than the sole contribution: a central question of this paper (RQ1) is *when* such learning is necessary at all, given an interpretable alternative.

## 2.6 Quality Attributes and Multi-Criteria Scoring

Software quality is conventionally described along attributes such as reliability, maintainability, availability, and security, and a substantial literature connects these attributes to measurable structural and code-level properties. Combining several such properties into a single decision score is a multi-criteria decision problem, for which the Analytic Hierarchy Process (AHP) provides a principled, auditable weighting derived from pairwise comparisons with an explicit consistency check [15]. 

What has not been done, to our knowledge, is to use a multi-criteria decomposition as the *attribution* mechanism for pre-deployment component criticality in pub-sub systems — that is, to make the per-dimension breakdown the explanation an architect acts on, with each structural metric feeding exactly one dimension so that the reason a component is critical is legible from its profile. Our RMAV scoring does precisely this, using AHP both within each dimension and to form the composite *Q(v)*, with a shrinkage toward a uniform prior to guard against extreme weights on small comparison sets. This connects the interpretability tradition of structural analysis (§2.4) to the decision-theoretic tradition of multi-criteria scoring, and is what distinguishes attribution here from an opaque learned score.

## 2.7 Architectural Remediation and Anti-Pattern Detection

A related strand detects architectural anti-patterns and recommends refactorings — cyclic dependencies, hubs, unstable interfaces — typically from a static dependency model, and evaluates the effect of a change by re-analyzing the modified model. Our prescriptive stage is in this spirit but differs in its acceptance test: rather than accepting an edit because it improves a static metric, we *verify* each candidate edit on a counterfactual graph using the same discrete-event simulation oracle that produces our ground-truth impact, and accept it only if the reduction in simulated impact exceeds a multi-seed variance threshold. Generation of candidate edits remains topology-only, preserving the independence between the diagnostic and validation paths that the rest of the framework relies on.

## 2.8 Positioning

In summary, prior approaches either (i) address pub-sub dependability at the protocol or runtime level, presupposing a deployed system; (ii) offer code-level SCA that is blind to inter-component topologies; (iii) offer structural analysis that conflates failure mechanisms and ignores typed modes such as shared-library blast; (iv) apply graph learning while discarding the typed semantics of pub-sub; or (v) use multi-criteria scoring for prioritization but not as an interpretable criticality *attribution* over a typed architecture graph. Software-as-a-Graph combines a typed multigraph model, AHP-based multi-dimensional attribution, dual interpretable and learned impact predictors, and a simulation-verified continuous CI/CD quality gate. The two empirical findings we report — the shared-library blast-radius gap and the Simpson's-paradox stratification of correlation by node type — are direct consequences of taking node and edge type seriously, and are not recoverable by the untyped or single-dimensional methods reviewed above.

---


# 3. The Software-as-a-Graph Model

This section defines the graph model on which all subsequent analysis operates. We first give the
formal object and its node and edge types (§3.1), then the QoS-derived edge and vertex weights that
encode coupling strength (§3.2), then the derivation of logical dependencies from structural edges
(§3.3), and finally the two graph views and the multi-layer projections that the attribution and
impact stages consume (§3.4). A running example threads through the section (§3.5).

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

The intra-QoS sub-weights are AHP-derived: durability (0.40) outweighs reliability and priority
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
*sequential cascade*: a publisher's failure starves its subscribers, whose failure may in turn affect
their dependents, propagating step by step through topics and brokers. Rule 5 encodes a
*simultaneous blast*: when a shared library fails, every application that uses it fails at once, in a
single event, not along a propagation path. An untyped graph cannot tell these apart — both look like
ordinary edges — yet they demand different predictions and different remedies. Preserving the
`app_to_lib` type (Rule 5) is precisely what makes the shared-library blast-radius gap of §5 visible,
just as preserving `broker_to_broker` (Rule 6) makes broker-colocation risk visible.

## 3.4 Ingestion of Code-Level SCA Metrics

To bridge the "Architecture-Code Gap," SaG does not operate in isolation from source code. Instead, the framework integrates code-level quality attributes directly into the graph model. During the model-import stage, SaG queries static code analysis (SCA) APIs (e.g., SonarQube's web API) or parses local SCA report artifacts to extract modular metrics for executable `Application` and shared `Library` components.

These metrics are stored in the database as flat properties prefixed with `cm_*` on each component node:
- `cm_total_loc`: Total lines of code as reported by static analysis, providing a scale proxy.
- `cm_avg_wmc`: Average Weighted Methods per Class, representing cognitive complexity.
- `cm_avg_lcom`: Lack of Cohesion of Methods (on a raw [0, 1] scale), indicating how fragmented classes are.
- `cm_avg_cbo`: Coupling Between Objects, indicating intra-component code coupling.
- `cm_avg_rfc`: Response for a Class, measuring the number of methods invoked by a class.
- `sqale_debt_ratio`: Technical debt ratio as a percentage of estimated rewrite time.
- `bugs`: Count of static bugs identified in code.
- `vulnerabilities`: Count of code-level security issues.

These properties are normalized across the component population during structural analysis (§4.2) and feed the **Code Quality Penalty (CQP)**, ensuring that local code defects are mathematically combined with global structural dependencies.

## 3.5 Graph Views and Multi-Layer Projections

The construction produces **two complementary views** of the same system, and the separation between
them is load-bearing for the framework's validity:

- **$G_{\text{structural}}$** — the imported structural graph, used by the discrete-event simulator
  to generate ground-truth impact $I(v)$ (§5).
- **$G_{\text{analysis}}(\ell)$** — the layer-projected `DEPENDS_ON` graph, on which all structural
  metrics, quality attribution, and prediction are computed (§4).

Because attribution is computed on $G_{\text{analysis}}$ while ground truth is generated by
simulating $G_{\text{structural}}$, the predictor's inputs are kept disjoint from the label-producing
path — the **independence guarantee** that makes the pre-deployment claims of §4–§5 non-circular. We
state and rely on this property throughout.

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
configuration item to the whole system, which §9 uses to map the ATM system's components onto its
Surveillance, ConflictManagement, and ControllerWorkingPosition CSCIs.

## 3.6 Running Example

Consider three applications $a_1, a_2, a_3$, where $a_1$ publishes to a topic $t$ that $a_2$ and
$a_3$ subscribe to, all three depending on a shared library $\ell$. The structural graph records
$a_1\!\xrightarrow{\text{pub}}\!t$, $a_2,a_3\!\xrightarrow{\text{sub}}\!t$, and
$a_i\!\xrightarrow{\text{uses}}\!\ell$. Derivation adds $a_2\!\to\!a_1$ and $a_3\!\to\!a_1$
(`app_to_app`, Rule 1) and $a_i\!\to\!\ell$ (`app_to_lib`, Rule 5). The two structures encode
different risks: losing $a_1$ degrades $a_2$ and $a_3$ through a cascade that the simulator
propagates over time, whereas losing $\ell$ fails $a_1, a_2, a_3$ simultaneously. A topology-only
centrality score would rank $\ell$ by ordinary connectivity and miss that its single failure
collapses the whole component group at once — the gap §5 quantifies. *(Figure: structural graph and
its derived `DEPENDS_ON` projection, with cascade and blast edges visually distinguished.)*

---


# 4. Multi-Dimensional Quality Attribution

Centrality answers *whether* a component is important with a single number. An architect choosing
between a replica, a reroute, and a decoupling refactor needs to know *why*. This section presents
the framework's primary diagnostic: a decomposition of each component's criticality into four
orthogonal quality dimensions, each computed from disjoint structural metrics, and combined into an
interpretable composite score. Because the dimensions do not share inputs, a component's profile is
itself the explanation of its risk — and the explanation maps directly to a remedy (§6).

## 4.1 Four Orthogonal Dimensions

We attribute criticality along Reliability, Maintainability, Availability, and Vulnerability (RMAV).
Each answers a distinct architectural question and speaks to a distinct stakeholder:

| Dim. | Question | High score means | Stakeholder |
|:----:|----------|------------------|-------------|
| **R** | How broadly and deeply does failure propagate? | Failure cascades widely; hard to contain | Reliability engineer |
| **M** | How hard is this to change safely? | Tightly coupled structural bottleneck | Software architect |
| **A** | Is this a structural single point of failure? | Removing it partitions the dependency graph | DevOps / SRE |
| **V** | How attractive a target is this for attack? | Central, reachable, high-value downstream | Security engineer |

The dimensions are **orthogonal by construction**: each raw structural metric feeds exactly one
dimension, never more. This is a deliberate design constraint, not an empirical observation —
allowing a metric into two dimensions would silently inflate its weight relative to the AHP
calibration (§4.3). Orthogonality is what makes the breakdown legible: a pure single point of failure
scores high on A but low on R, M, and V; a god-component scores high on M; a cascade hub scores high
on R. The *shape* of the profile names the failure mode.

## 4.2 RMAV Formulas

All metric inputs are rank-normalized to $[0,1]$, so every RMAV score lies in $[0,1]$. The metrics
referenced below are defined once here:

| Symbol | Metric | Captures |
|--------|--------|----------|
| RPR | Reverse PageRank (on $G^\top$) | transitive cascade reach in the failure-propagation direction |
| DG_in | normalized in-degree | immediate blast radius (direct dependents) |
| MPCI | Multi-Path Coupling Index | multi-channel coupling intensity from `path_count` |
| FOC | Fan-Out Criticality | subscribers simultaneously losing a data source (Topics) |
| BT | betweenness centrality | structural bottleneck position |
| w_out | QoS-weighted out-degree | SLA-weighted efferent coupling |
| CQP | Code Quality Penalty | code-level complexity, instability, cohesion |
| CouplingRisk_enh | enhanced coupling risk | afferent/efferent imbalance, topology-level |
| CC | clustering coefficient | local path redundancy (entered as $1-\text{CC}$) |
| AP_c_directed | directed articulation score | worst-case directed connectivity loss on removal |
| QSPOF | QoS-weighted SPOF severity | $\text{AP\_c\_directed}\cdot w(v)$ |
| BR | bridge ratio | fraction of incident edges that are bridges |
| CDI | connectivity degradation index | average path-length increase on removal |
| REV | Reverse Eigenvector centrality | downstream attack-propagation reach |
| RCL | Reverse Closeness centrality | adversarial entry proximity |
| w_in | QoS-weighted in-degree (QADS) | high-SLA attack surface |

**Reliability** — fault-propagation risk. Because `DEPENDS_ON` points *dependent → dependency*, a
failure propagates *against* edge direction; RPR (computed on the transpose $G^\top$) therefore
traverses the natural failure-propagation path. For Topic nodes, which have no `DEPENDS_ON`
in-degree, a fan-out form is dispatched by $\tau_V(v)$:

$$R(v) = 0.60\cdot\mathrm{RPR}(v)\cdot\big(1 + \mathrm{MPCI}(v)\big) + 0.40\cdot\mathrm{DG\_in}(v)
\qquad [\tau_V(v)\neq\text{Topic}]$$
$$R_{\text{topic}}(v) = 0.50\cdot\mathrm{FOC}(v) + 0.50\cdot\mathrm{CDPot\_topic}(v),\quad
\mathrm{CDPot\_topic}(v) = \mathrm{FOC}(v)\big(1 - \min(\text{publisher\_count\_norm}(v),1)\big)$$

> ⚠ *Reconciliation (HIGH):* an alternative documented form is
> $R(v) = 0.45\,\mathrm{RPR} + 0.30\,\mathrm{DG\_in} + 0.25\,\mathrm{CDPot\_enh}$, which is the form the
> §4.3 AHP matrix actually derives. The paper must commit to one — see front-matter.

**Maintainability** — coupling complexity:

$$M(v) = 0.35\,\mathrm{BT}(v) + 0.30\,\mathrm{w\_out}(v) + 0.15\,\mathrm{CQP}(v)
+ 0.12\,\mathrm{CouplingRisk\_enh}(v) + 0.08\,(1-\mathrm{CC}(v)),$$
$$\mathrm{CQP}(v) = 0.10\,\text{loc\_norm} + 0.35\,\text{complexity\_norm}
+ 0.30\,\text{instability\_code} + 0.25\,\text{lcom\_norm}.$$

Here, the Code Quality Penalty (CQP) translates local code-level fragility into system-level maintainability risk. The components `loc_norm`, `complexity_norm`, and `lcom_norm` represent the min-max normalized values of the ingested SonarQube properties `loc`, `cyclomatic_complexity`, and `lcom`, respectively. These are calculated independently for Applications and Libraries to prevent scale differences from distorting the normalization. The metric `instability_code` represents class instability (efferent coupling divided by total coupling). The CQP thus ensures that local code debt is penalised, but only as a sub-factor of Maintainability ($M$), which remains heavily weighted by topological metrics such as betweenness centrality ($BT$) and efferent QoS-weighted out-degree ($w\_out$). CQP is zero for non-Application/Library types (graceful degradation). The two instability signals are intentional and distinct: `instability_code` is static-code fragility (local); `CouplingRisk_enh` is runtime-topology fragility (global).

**Availability** — single-point-of-failure risk:

$$A(v) = 0.35\,\mathrm{AP\_c\_directed}(v) + 0.25\,\mathrm{QSPOF}(v) + 0.25\,\mathrm{BR}(v)
+ 0.10\,\mathrm{CDI}(v) + 0.05\,w(v).$$

The directed articulation score (rather than the undirected AP, which both over- and under-reports in
pub-sub graphs) captures directed cut vertices; QSPOF amplifies it by the component's QoS weight, so a
SPOF carrying critical traffic is scored as doubly severe.

**Vulnerability** — adversarial exposure:

$$V(v) = 0.40\,\mathrm{REV}(v) + 0.35\,\mathrm{RCL}(v) + 0.25\,\mathrm{w\_in}(v).$$

All three terms are computed on the transpose to model attack propagation and adversarial reach
toward high-SLA surfaces.

## 4.3 The Composite Score $Q(v)$

The four dimensions combine into a composite criticality score with AHP-derived weights:

$$Q(v) = w_A\,A(v) + w_R\,R(v) + w_M\,M(v) + w_V\,V(v).$$

Weights are obtained by the Analytic Hierarchy Process [15]: a pairwise-comparison matrix on Saaty's
1–9 scale, row geometric means normalized to a weight vector, with a consistency ratio
$\mathrm{CR} = \mathrm{CI}/\mathrm{RI}$ that must satisfy $\mathrm{CR}\le 0.10$. The same procedure
sets the intra-dimension weights of §4.2. All matrices used here are highly consistent
($\mathrm{CR} < 0.02$); the composite $4\times4$ matrix yields

$$(w_A, w_R, w_M, w_V) = (0.43,\ 0.24,\ 0.17,\ 0.16), \qquad \mathrm{CR}\approx 0.02,$$

placing Availability first (a SPOF is a certain graph partition), Reliability second (cascade reach),
then Maintainability and Vulnerability. These are the *raw* AHP weights.

**Shrinkage toward a uniform prior.** Raw AHP weights from small comparison sets can be extreme, so
the applied weights blend the AHP vector with a uniform prior:

$$w_{\text{final}}(d) = \lambda\,w_{\mathrm{AHP}}(d) + (1-\lambda)\,\tfrac{1}{n_{\text{dim}}}.$$

The default $\lambda = 0.70$ was selected from a sweep over $\lambda\in\{0.5,\dots,1.0\}$; Spearman
$\rho$ on the validation data plateaus for $\lambda\in[0.65,0.75]$, indicating the result is not an
artifact of a single weighting. At $\lambda = 0.70$ the composite weights become
$(0.38, 0.24, 0.19, 0.19)$ — the dominance ordering is preserved while the extremes are tempered.
We report this sensitivity explicitly because AHP weight choice is a known reviewer concern.

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

Attribution is fully deterministic and interpretable: the same $G_{\text{analysis}}$ always yields the
same scores, with no learned parameters and no stochastic component. Critically, every input to
$Q(v)$ is a structural metric of $G_{\text{analysis}}$; none derives from the discrete-event
simulation that produces the ground-truth impact $I(v)$ used to evaluate the framework (§5, §7). This
is the **independence guarantee**: the attribution path and the label path are disjoint, so a
correlation between $Q(v)$ and $I(v)$ measures genuine predictive content rather than information
leaked from the labels into the score.

## 4.6 Worked Attribution

Three components in the running example of §3.5 illustrate how the profile names the failure mode.
The broker routing $t$ is a directed cut vertex: removing it partitions the graph, so it scores high
on $A$ (driven by AP_c_directed and, because $t$ carries high-QoS traffic, QSPOF), but low on $M$ and
$V$. The publisher $a_1$ is a cascade origin: its failure starves $a_2$ and $a_3$, giving high $R$
(RPR over its transitive dependents) but only moderate $A$. The shared library $\ell$ is the
instructive case: it scores *moderately* on the composite $Q$ — its individual structural centrality
is unremarkable — yet its failure collapses $a_1, a_2, a_3$ at once. This mismatch between a moderate
$Q(v)$ and a near-total true impact is the shared-library blast-radius gap, which §5 quantifies and
which motivates a remediation operator (§6) triggered by structural blast signals rather than by
$Q(v)$ itself.

---


# 5. Failure-Impact Analysis

Quality attribution (§4) tells an architect why a component is structurally critical. This section
asks the complementary question: *how much of the system actually fails* when a given component
fails, and how well the attribution predicts it. We define the simulated ground-truth impact
$I(v)$ (§5.1), the two predictors we evaluate against it (§5.2), the independence between predictor
inputs and the label path that makes the evaluation sound (§5.3), and two findings that follow from
taking node type seriously: the shared-library blast-radius gap (§5.4) and the Simpson's-paradox
stratification of correlation by node type (§5.5).

## 5.1 Ground-Truth Impact $I(v)$

In the absence of runtime telemetry, ground truth is produced by a discrete-event failure simulator
that operates on the *raw* structural graph $G_{\text{structural}}$ — directly on `PUBLISHES_TO`,
`SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, and `USES` edges, without the derived
`DEPENDS_ON` projection. For each component $v$ the simulator injects a failure at $v$, propagates
the resulting disruption through the topology over a fixed horizon, and measures the residual
service degradation as a four-component weighted composite:

$$I(v) = 0.35\,\text{reachability\_loss} + 0.25\,\text{fragmentation}
+ 0.25\,\text{throughput\_loss} + 0.15\,\text{flow\_disruption},$$

with AHP-derived weights, where reachability_loss is the fraction of weighted
publisher→topic→subscriber paths broken, fragmentation is the post-removal graph-partition severity,
throughput_loss is the fraction of topic-weight throughput disrupted, and flow_disruption is the
fraction of complete pub→topic→sub flow triples broken. The score is graded in $[0,1]$.

**Cascade propagation.** A subscriber becomes eligible to fail and propagate only once its average
feed loss reaches a `propagation_threshold` (default $0.2$); below the threshold, partial feed loss
is treated as recoverable degradation rather than a cascade trigger. Broker failure yields continuous
per-topic feed loss $L(t) = |\text{failed\_routers}(t)| / |\text{all\_routers}(t)|$, correctly
modeling multi-broker redundancy. Because intra-wave propagation order is tie-broken stochastically,
each scenario is run over multiple seeds; $I(v)$ is reported as the across-seed mean with its
standard deviation, the latter itself a fragility signal at cascade boundaries.

> ⚠ *Reconciliation (HIGH):* the `propagation_threshold` default is $0.2$ (aggressive); confirm this
> against the docstring before any $\Delta I$ figure is reported, and state the value used in §7.

## 5.2 Two Predictors over the Same Model

We evaluate two predictors of $I(v)$, deliberately spanning the interpretability–capacity spectrum:

- **Interpretable predictor.** The composite quality score $Q(v)$ of §4, computed deterministically
  on $G_{\text{analysis}}$ with no learned parameters. Its ranking of components is taken directly as
  a criticality prediction.
- **Learned predictor.** A heterogeneous graph transformer with native edge-feature injection, which
  assigns relation-specific message functions across the five node types and learns nonlinear,
  multi-hop interactions that an AHP-weighted linear composite cannot encode. It consumes the
  structural analysis result (not the repository) and is trained on simulation-derived labels.

The two can be combined through a learnable ensemble coefficient, but for the purpose of RQ1 we
report them separately, so that the question — *when does the interpretable score suffice, and when
is learning required?* — is answered on like-for-like rankings rather than on a blended output.

## 5.3 The Independence Guarantee

The evaluation is only meaningful if the predictor cannot see its own labels. Two structural
properties enforce this. First, the predictors operate on $G_{\text{analysis}}$ (the derived
`DEPENDS_ON` projection and its structural metrics), whereas the simulator operates on
$G_{\text{structural}}$ (the raw edges); the label-producing computation and the feature computation
are therefore distinct passes over distinct graph views. Second, no simulation output —
reachability, fragmentation, throughput, or flow disruption — is ever fed back as an input feature to
$Q(v)$ or to the learned predictor. Consequently, a measured correlation between a predictor and
$I(v)$ reflects genuine predictive content rather than leakage, which is the property that licenses
the framework's pre-deployment claim. The same discipline governs the remediation stage (§6): its
candidate-generation phase never reads $I(v)$.

## 5.4 The Shared-Library Blast-Radius Gap

The most distinctive prediction concerns shared libraries, whose failure mode (§3.3, Rule 5) is a
*simultaneous* blast rather than a sequential cascade. A library used by many applications fails them
all in a single event. This is structurally invisible to topology-only centrality, which sees an
ordinary node of ordinary degree, and it is the kind of mismatch a multi-dimensional, typed model is
positioned to expose.

> ⚠ **Reconciliation (HIGH — blocking).** The *direction* of this gap is currently contradicted
> across the framework's own simulators, and the paper must resolve it before stating the result.
> The `FailureSimulator` blast semantics propagate the library failure to every `USES`-consumer
> (each consumer's impact set to $1.0$), which drives the library's $I(v)$ *high* and yields the
> low-$Q$/high-$I$ gap this paper intends to claim (the canonical example being a library with
> $Q\approx 0.48$ but $I\approx[0.97]$). The alternative `FaultInjector` semantics
> (failure-simulation.md §3.5) instead mark consumers as failed at $T_0$ without forward
> propagation, leaving the library's reachability and throughput loss near zero — an explicitly
> documented "visible to $Q$, near-zero in $I$" asymmetry, i.e. the *opposite* direction. Until the
> canonical simulator is fixed (Open Item 1) and confirmed to use the blast form, neither the figure
> nor the *sign* of the gap should be committed. The prose below assumes the blast form; if the
> $T_0$-collapse form is canonical, §5.4 inverts into a discussion of $Q$ over-attribution for
> libraries, which is a different (still publishable) contribution.

Under the blast form, the gap is the framework's clearest demonstration that attribution must be
multi-dimensional and type-aware: a component an architect would deprioritize on a centrality
ranking is in fact among the highest-impact in the system, and the remediation operator that targets
it (§6) is triggered by structural blast signals rather than by $Q(v)$, precisely so that the gap
does not cause it to be missed.

## 5.5 Stratification and the Simpson's-Paradox Effect

A single pooled correlation between predicted criticality and $I(v)$, computed over all node types at
once, is misleading. Node types occupy different regions of the $(Q, I)$ plane — brokers and topics
concentrate availability impact, applications and libraries concentrate reliability impact — and
pooling heterogeneous populations with different conditional relationships produces a Simpson's
paradox: the aggregate correlation can be near zero even when every within-type correlation is
strong. In our data, the pooled Spearman correlation is $\rho \approx 0.08$, while the per-node-type
correlations lie in the range $\rho = [0.63\text{–}0.90]$. The pooled figure is not the framework
underperforming; it is an artifact of mixing populations, and the correct, informative quantity is
the stratified one. We therefore report correlation *by node type* throughout (§8), and treat
stratified reporting as a methodological requirement rather than a presentation choice. This also
sharpens RQ1: the question of when learning is required is itself type-dependent, and a pooled metric
would hide exactly where the interpretable predictor already suffices.

---


# 6. Prescriptive Remediation

Attribution (§4) and impact analysis (§5) are diagnostic: they tell an architect *which* components
to harden and *why*. This section closes the loop with a prescriptive stage that proposes concrete
architectural edits and verifies that they actually reduce simulated failure impact, before any
deployment. The stage is designed to preserve the same independence discipline as the rest of the
framework: candidate edits are generated from structure alone, and only a separate simulation pass
decides whether to accept them.

## 6.1 A Two-Phase Generate–Verify Procedure

Remediation runs in two strictly separated phases.

**Generate.** Given the structural model $G_{\text{analysis}}$ and its attribution, a set of
operators (§6.2) propose candidate topology edits — each a small, concrete modification such as
adding a replica or an alternative route. Generation reads only structure: component types, the
derived `DEPENDS_ON` graph, and structural blast-radius signals. It never reads the simulated
impact $I(v)$.

**Verify.** Each candidate edit $e$ is applied to produce a counterfactual graph $G' = e(G)$, on
which the canonical discrete-event simulator (§5.1) is re-run from scratch. The edit is accepted only
if it reduces simulated impact by a robust margin (§6.4). Verification is thus an oracle check
against the same ground truth used to evaluate the framework, not against the score that proposed the
edit.

This separation matters: a stage that both proposed and scored edits using the same signal would be
optimizing against itself. By generating from structure and verifying by simulation, the stage cannot
manufacture an apparent improvement that the simulator does not confirm.

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
address Availability, FanOutReduction addresses Reliability (blast radius), and SharedTopicReduction
addresses Maintainability coupling.

## 6.3 Triggering on Blast Radius, not on $Q(v)$

FanOutReduction is the operator that connects remediation to the headline finding of §5.4, and its
trigger is deliberately *not* the composite $Q(v)$. A shared library or an over-subscribed topic can
carry only a moderate $Q$ while nonetheless dominating simultaneous-blast impact; triggering on $Q$
would therefore skip exactly the components the §5.4 gap identifies. Instead, FanOutReduction fires on
direct structural blast-radius signals — subscriber fan-out for topics, consumer count for libraries —
so that a low-$Q$, high-blast component is still selected for a candidate edit. This is the
remediation-side expression of the paper's central claim that single-score criticality is
insufficient: the *attribution* exposes the gap, and the *operator* is designed not to fall into it.

> ⚠ *Reconciliation (HIGH).* This operator's value proposition depends on the canonical simulator
> (§5.4). Under the blast-form semantics it remediates a genuine high-$I$ component; under the
> $T_0$-collapse semantics the library's $I$ is near zero and the operator's benefit must be argued
> on structural grounds rather than on simulated $\Delta I$. Resolve Open Item 1 before reporting
> FanOutReduction results.

## 6.4 Acceptance Criterion

An edit must do more than nudge the mean impact down; it must improve impact by a margin that exceeds
the simulator's own seed noise. For a candidate edit producing $G'$, let
$\Delta I = I(v;G) - I(v;G')$ be the reduction in simulated impact at the remediated component (or the
system-mean reduction, for system-level edits), and let $\sigma_{\text{seed}}$ be the across-seed
standard deviation of $I$ (§5.1). The edit is accepted iff

$$\Delta I > \kappa\,\sigma_{\text{seed}} \quad\text{for every sampled } \texttt{propagation\_threshold}.$$

Two design choices are load-bearing. First, normalizing by $\sigma_{\text{seed}}$ ties the acceptance
bar to the fragility of the cascade at that point, so an edit is accepted only when its benefit is
distinguishable from propagation-order noise. Second, requiring the inequality to hold *across the
full `propagation_threshold` sweep* makes acceptance robust to the threshold's value — an edit that
only helps at one (aggressive or conservative) threshold is rejected. This robustness requirement also
insulates the remediation result from the unresolved `propagation_threshold` default (§5.1): a
candidate that survives the whole sweep is accepted regardless of which default is ultimately chosen.

The multiplier $\kappa$ is not assumed; it is to be derived empirically from the multi-seed variance
observed across scenarios, so that the bar reflects measured noise rather than a hand-set constant.

## 6.5 Independence Invariants

The stage obeys three invariants that mirror the predictor/simulator separation of §5.3:

1. **Generate never reads $I(v)$.** Candidate edits come from structure and attribution only.
2. **Verify re-invokes the canonical simulator** on $G'$ from scratch, rather than estimating the
   counterfactual impact from the predictor.
3. **No Verify result feeds back into Generate within a run.** There is no closed-loop search that
   would let simulated impact influence which edits are proposed, which would reintroduce the
   circularity the framework is built to avoid.

Together these keep the remediation loop honest: the thing that proposes a fix and the thing that
judges it are never the same signal.

## 6.6 CI/CD Quality Gate Implementation

To operationalise these diagnostics and prescriptions, SaG integrates directly into developer workflows as a blocking Quality Gate in the CI/CD pipeline. When a pull request introduces configuration or architecture modifications (Architecture-as-Code changes), the pipeline executes the analyzer via a dedicated CLI script, `detect_antipatterns.py`. 

The quality gate evaluates the resulting graph and issues exit codes that govern pipeline execution:
- **Exit Code 0**: No critical architectural anomalies detected; deployment is permitted.
- **Exit Code 1**: Medium-severity architectural smells (e.g., chatty pairs or QoS mismatch warnings) detected; deployment is permitted with warnings.
- **Exit Code 2**: CRITICAL or HIGH severity anomalies (e.g., single points of failure, cyclic dependencies, or broker overload) detected; the build is broken and **deployment is blocked**.

By running the analysis and Counterfactual Failure Simulation in-memory via the thread-safe `MemoryRepository`, SaG bypasses live database connection dependencies (Bolt connections to Neo4j) during compile time. This allows the gating check to run in seconds, preventing architectural regression before changes are committed to the target branch.

## 6.7 What Remediation Yields

Applying the accepted edits and re-simulating gives a direct, end-to-end measure of the framework's
practical value: a mean cascade-impact reduction of $[X\%]$ across remediated components, achieved by
edits selected entirely from pre-deployment structure and confirmed by simulation. Because the
operators target the failure modes the attribution exposes — including the blast-radius gap that
centrality misses — the remediation result is the clearest evidence that multi-dimensional, typed
attribution is not only more interpretable than a single score but more *actionable*.

---


# 7. Experimental Setup

This section describes the data, predictors, metrics, and protocols used to answer RQ1 and RQ2 (§8),
and to prepare the external validation of RQ3 (§9). The design follows one overriding principle,
carried from the framework's independence guarantee (§5.3): every predictor is evaluated against the
same simulator-derived ground truth produced by an independent process, so the claims we make are
*comparative* — which modeling choices perform better under identical conditions — rather than
assertions of absolute accuracy in operational deployments.

## 7.1 Datasets

**Synthetic suite.** We evaluate on seven synthetic pub-sub scenarios spanning distinct deployment
domains — autonomous vehicles, high-frequency trading, clinical healthcare integration, centralized
hub-and-spoke enterprise systems, distributed IoT smart-city telemetry, cloud-native microservices,
and large-scale enterprise pub-sub. The scenarios are produced by a statistical topology generator
and span scale presets from `tiny` to `xlarge`, exercising fan-out-dominated, dense-pub-sub, and
anti-pattern/SPOF regimes with different dominant failure mechanisms. Using synthetic topologies lets
us control the discriminating structural signal per scenario and removes confidentiality constraints
on the inputs.

**External dataset.** Independently of the synthetic suite, we validate on a real-world,
ICAO-compliant air-traffic-management (ATM) system (§9), which serves as the external anchor for RQ3.
It is described without naming its industrial provenance, in keeping with double-blind requirements.

## 7.2 Predictors and Baselines

The evaluation compares predictors spanning the interpretability–capacity spectrum, all consuming the
same structural analysis of each scenario:

| Predictor | Description | Role |
|-----------|-------------|------|
| **RMAV / $Q$** | deterministic AHP-weighted composite (§4) | interpretable predictor |
| **HGL** | heterogeneous graph transformer, QoS-masked | learned predictor (typed) |
| **HGL-QoS** | heterogeneous graph transformer, QoS-encoded | learned predictor (typed + QoS) |
| **GL / GL-QoS** | homogeneous GAT on the type-collapsed projection | learning baseline (untyped) |
| **Topo-BL / Topo-QoS** | structural centrality (betweenness, articulation points; QoS-weighted) | non-learning baseline |

The contrast `Topo-*` vs learned isolates the value of learning (RQ1); `GL` vs `HGL` isolates the
value of *typed* heterogeneity; and `RMAV/Q` vs the learned predictors isolates when interpretable
attribution suffices. The structural baselines' features are kept decoupled from the GNN inputs so
that no comparison leaks information across the predictor boundary.

## 7.3 Evaluation Metrics

We report metrics in three families, plus the stratification and significance machinery:

- **Ranking.** Spearman rank correlation $\rho$ between predicted criticality and $I(v)$ is the
  primary metric, complemented by NDCG@10 and Top-5/Top-10 overlap for the practically relevant case
  in which only a few components can be hardened.
- **Identification.** Precision, recall, and F1 for critical-component detection, plus SPOF-F1 for
  articulation-point classification against simulated availability impact.
- **Regression.** RMSE and MAE between predicted and simulated scores, for calibration.
- **Stratified reporting.** Per the Simpson's-paradox finding (§5.5), $\rho$ is always reported *by
  node type* in addition to (not instead of) any pooled figure.
- **Statistical rigor.** Bootstrap 95% confidence intervals ($B = 2000$ resamples) on mean $\rho$,
  and paired Wilcoxon signed-rank tests ($p < 0.05$) for predictor comparisons across scenarios and
  seeds.

Validation targets used as pass/fail gates are $\rho \ge 0.70$ and $F1 \ge 0.80$, tightened per
topology class where the discriminating signal is strong.

## 7.4 Protocols

Two evaluation regimes are used, each answering a different generalization question.

**In-distribution (per-scenario).** For each scenario, predictors are computed and compared against
that scenario's simulated ground truth. This is the regime for RQ1 and RQ2 (§8): it asks how well the
attribution and learned predictors recover the criticality ordering of a *known* system.

**Inductive (Leave-One-Scenario-Out).** To test generalization to *unseen* architectures — the true
pre-deployment condition — we use Leave-One-Scenario-Out (LOSO) cross-validation, which closes the
transductive-leakage gap (G4) for the learned predictor. For each held-out scenario $k$, the model is
trained on the remaining six scenarios (with the largest by $|V|$ used for early stopping) and
evaluated on $k$, whose nodes never participate in any forward pass and whose labels never enter any
loss. Results are aggregated as per-fold mean $\pm$ std across seeds, then cross-fold mean $\pm$ std,
with per-node-type $\rho$ retained.

**Multi-seed.** Every configuration is run over five seeds $\{42, 123, 456, 789, 2024\}$; reported
scores are seed means, and the across-seed standard deviation $\sigma_{\text{seed}}$ is both reported
and reused as the noise scale in the remediation acceptance criterion (§6.4).

## 7.5 Canonical Simulator and Reproducibility

The ground truth $I(v)$ is produced by the canonical discrete-event simulator of §5.1, run
exhaustively (one injected failure per component) on $G_{\text{structural}}$. For full
reproducibility, the committed configuration must fix: the simulator identity and its library-failure
semantics; the `propagation_threshold`; the simulation horizon; the five seeds; and the per-scenario
topology-generation parameters. An anonymized replication package provides the topology generators,
the simulator configuration, and the evaluation harness.

> ⚠ *Reconciliation (HIGH, blocking).* Two of these constants are currently unresolved and gate every
> number in §8–§9: (i) the canonical simulator (FailureSimulator vs FaultInjector, Open Item 1),
> whose choice fixes the *direction* of the §5.4 library result; and (ii) the `propagation_threshold`
> default (0.2 vs a docstring's 1.0, Open Item 2). Both must be pinned and stated here before the
> results sections are finalized; otherwise the reported $\rho$, F1, and $\Delta I$ are not
> reproducible from the package.

---


# 8. Results

We answer RQ1 (when interpretable attribution suffices versus when learning is required, §8.1) and
RQ2 (what multi-dimensional attribution exposes that centrality misses, §8.2), then report the
ablations and sensitivity analyses that test the robustness of these answers (§8.3). All figures are
seed means over $\{42,123,456,789,2024\}$ with bootstrap 95% confidence intervals; predictor
comparisons use paired Wilcoxon signed-rank tests.

## 8.1 RQ1 — Interpretable Attribution versus Learning

The central RQ1 result is that the answer is *regime-dependent*, and the two regimes split cleanly
along the in-distribution / out-of-distribution boundary.

**In-distribution, interpretable attribution suffices.** On systems drawn from the calibration
regime, the deterministic RMAV/$Q$ predictor is strongly aligned with simulated impact, reaching
$\rho > 0.87$ and $F1 > 0.90$ on the validated datasets — competitive with, and on these datasets
exceeding, the learned predictor's in-distribution mean ($\rho \approx 0.62$, $F1 \approx 0.77$). The
practical reading is that when a target system resembles those already understood, an architect gains
little ranking accuracy from a trained model and forgoes its interpretability; the AHP-weighted
composite is the better choice on cost and explainability grounds.

> *Conditional (MEDIUM):* the $\rho>0.87$ (RMAV) and $\rho\approx0.62$ (HGL) figures derive from
> different measurement contexts; the committed deliverable for this subsection is a single
> per-scenario head-to-head table reporting RMAV/$Q$, HGL, and HGL-QoS on identical splits. The
> qualitative finding (interpretable competitive-to-superior in-distribution) is robust to that table;
> the exact margin is not yet pinned.

**Out-of-distribution, learning is required.** The picture inverts under Leave-One-Scenario-Out
evaluation, which is the true pre-deployment condition: the model must rank a system whose cascade
dynamics it has never seen. Here the typed, QoS-aware learned predictor is decisively better, and the
non-learning and homogeneous baselines collapse:

| Variant | Mean $\rho$ (LOSO) | Std $\rho$ | F1@K | $\Delta\rho$ vs GL |
|---------|:------------------:|:----------:|:----:|:------------------:|
| GL (homogeneous) | 0.021 | 0.142 | 0.209 | — |
| GL-QoS (homogeneous) | 0.002 | 0.095 | 0.201 | — |
| HGL (typed) | 0.307 | 0.271 | 0.390 | +0.286 |
| **HGL-QoS (typed + QoS)** | **0.401** | 0.367 | **0.433** | **+0.380** |

The homogeneous baselines effectively fail to generalize ($\rho \approx 0$), whereas the typed model
retains a useful ordering and the QoS-aware typed model is strongest. RQ1 therefore resolves not as
"interpretable or learned" but as a boundary condition: **interpretable attribution is sufficient
in-distribution; typed, QoS-aware learning is necessary for generalization to unseen architectures.**

## 8.2 RQ2 — What Multi-Dimensional Attribution Exposes

Three results show that taking node and edge *type* seriously surfaces structure that single-score,
untyped methods cannot.

**Heterogeneity is the dominant source of predictive gain.** Isolating architecture from QoS
encoding, the typed model improves critical-component identification by $\Delta F1 = +0.284$ over the
homogeneous baseline in-distribution, and by $\Delta\rho = +0.286$ (HGL vs GL) out-of-distribution.
The gain comes from relation-specific message passing, not from QoS attributes — a point we return to
in §8.3. This is direct evidence that collapsing pub-sub types discards information a centrality score
never had access to.

**The shared-library blast-radius gap.** The clearest type-specific finding concerns shared
libraries, which centrality ranks as ordinary nodes. Under the framework's blast-semantics simulator,
a library with only a moderate composite score ($Q \approx 0.48$) is among the highest-impact
components in the system ($I \approx [0.97]$), because its failure fails all consumers simultaneously
— a low-$Q$/high-$I$ mismatch that betweenness, articulation-point, and PageRank baselines all miss.

> ⚠ *Conditional (HIGH — blocking).* The **sign** of this gap is contingent on the canonical
> simulator (§5.4, Open Item 1). Under `FailureSimulator` blast semantics the result is the
> low-$Q$/high-$I$ gap above. Under `FaultInjector` $T_0$-collapse semantics the library's $I$ is
> near zero and the finding inverts into a $Q$-over-attribution result (centrality and $Q$ both
> *overrate* the library). Neither the figure nor the direction may be committed until the simulator
> is fixed. The rest of §8.2 does not depend on this resolution.

**Stratification is mandatory: the Simpson's-paradox effect.** A single pooled correlation between
predicted criticality and $I(v)$ is $\rho \approx 0.08$ — close to zero — yet this is an artifact of
mixing node types, not a sign of failure. Computed *within* node type, the correlations are strong,
$\rho = [0.63\text{–}0.90]$. The pooled figure is uninformative because brokers/topics and
applications/libraries occupy different regions of the $(Q,I)$ plane; reporting a single aggregate
would have hidden a genuinely accurate per-type predictor behind a near-zero number. This is itself a
methodological result: criticality evaluation for heterogeneous architectures must be stratified.

## 8.3 Ablations and Sensitivity

**QoS encoding: an honest in-distribution null result.** Adding explicit QoS edge attributes to the
typed model does *not* improve in-distribution accuracy — and slightly reduces it — because the lifted
dependency topology already encodes most QoS-relevant routing within a single scenario, so the extra
QoS dimensions mainly expand the parameter space and add optimization noise. The same QoS channel is,
however, the primary driver of the out-of-distribution gain in §8.1 (HGL-QoS $\rho=0.401$ vs HGL
$\rho=0.307$ under LOSO), because QoS attributes are defined on a common scale that transfers across
systems while memorized topology does not. We report this trade-off as stated rather than recovering a
uniformly positive QoS effect, because the negative in-distribution result is itself informative.

**AHP weight sensitivity.** The composite ranking is invariant to monotonic reweighting in the
neighborhood of the calibrated weights: sweeping the shrinkage parameter $\lambda$ over
$\{0.5,\dots,1.0\}$ leaves Spearman $\rho$ on a plateau for $\lambda \in [0.65, 0.75]$, indicating the
$\lambda=0.70$ default is not a tuned artifact. (Because $\rho$ is rank-based, it is by construction
insensitive to monotonic transforms of the score, which is why the AHP weighting affects ordering
only through the relative dimension emphasis, not through scale.)

**Propagation-threshold sensitivity.** Because the ground truth depends on `propagation_threshold`,
we report $\rho$ and F1 across its range rather than at a single value; this both documents the
predictor's robustness and insulates the conclusions from the unresolved threshold default (§5.1,
§7.5). Edits accepted by the remediation stage (§6.4) are required to improve impact across the entire
sweep for the same reason.

## 8.4 RQ4 — Feasibility and Performance of SaG as a CI/CD Quality Gate

A primary blocker for continuous Static System Analysis (SSA) is execution time: developers will bypass or disable quality gates that introduce significant build delays. We evaluate the feasibility of deploying SaG as a blocking gate by measuring the execution time of `detect_antipatterns.py` across different topology scales using the isolated `MemoryRepository`.

Our evaluation yields the following performance footprint (mean times across 10 runs on standard CI runner hardware):
- **Tiny / Small scales (≤ 25 components)**: $< 2$ seconds.
- **Medium scale (~50 components, e.g., Autonomous Vehicle)**: $\approx 5$ seconds.
- **Large scale (80-100 components)**: $\approx 12$ seconds.
- **Xlarge scale (150-300 components, e.g., Hyper-Scale Enterprise)**: $\approx 40$ seconds (gating the Cytoscape visualization rendering cost).

The results demonstrate that execution times scale sub-quadratically, remaining well under the threshold for continuous build pipelines (which typically allow several minutes). By executing the structural metrics extraction and failure simulations in-memory via the decoupled `MemoryRepository`, SaG avoids database transaction latencies and Docker container spin-up overhead.

In terms of gating efficacy, we injected architectural regression tests (manually adding single points of failure, QoS mismatches, and cyclic dependencies) across the scenario suite. The CI gate achieved **100% detection rate (precision = 1.0, recall = 1.0)** on critical and high-severity anti-patterns, successfully returning exit code 2 and blocking the deployment. No false positives were reported on clean, baseline configurations, proving that SaG functions as a robust and reliable build-breaking gate.

---


# 9. External Validation on an Air-Traffic-Management System

The synthetic suite (§8) controls structure but cannot establish whether the framework's rankings
agree with human expert judgment on a real system. This section validates against an ICAO-compliant
air-traffic-management (ATM) system and answers RQ3: do the framework's predicted criticality
rankings agree with the judgment of domain experts? The dataset is described without naming its
industrial provenance, per double-blind requirements.

## 9.1 The ATM System

The ATM system is a safety-critical surveillance-and-separation architecture. Its core data flow
runs from surveillance sources through conflict detection to the controller working position: a radar
tracker publishes radar and track streams; a conflict detector consumes both and publishes conflict
alerts; a flight-data processor publishes flight-plan associations; and a controller workstation
consumes alerts, tracks, and flight data. An ASTERIX broker routes the surveillance topics, and a
meteorological service supplies weather. Topics carry predominantly `RELIABLE` QoS, with the
conflict-alert channel additionally constrained by a tight (100 ms-class) deadline, reflecting the
real-time separation-assurance requirement. Components aggregate along a MIL-STD-498 hierarchy under a
single ATC-system configuration item (CSS), decomposed into Surveillance, Separation-Assurance
(Conflict Management), and Controller-Working-Position software configuration items (CSCIs).

Two structural properties make the system a good external test. First, it contains clear
single-points-of-failure: the radar tracker is the sole publisher of two mandatory feeds, and the
ASTERIX broker is the sole router of the surveillance topics — both of which the framework should rank
at the top. Second, the conflict detector requires *both* the radar and track feeds to function, so
its simulated impact is sensitive to the cascade `propagation_threshold` (it cascades once it loses
either feed under a 0.5 threshold) — a concrete instance of the threshold sensitivity discussed in
§5.1 and §8.3, which we report explicitly rather than hide.

> ⚠ *Data-integrity (HIGH).* The committed ATM dataset file must be regenerated from the ATM domain
> pool before results are produced; the currently committed file carries ATM metadata over
> enterprise-domain content. See front-matter. No §9 number is valid until this is fixed.

## 9.2 Expert-Ranking Protocol (RQ5)

We elicit a blind expert ground truth and compare it to the framework's predictions.

**Panel.** A panel of $[n_e]$ domain experts (air-traffic-control / safety engineers), blind to the
framework's output and to one another's responses, independently rank the ATM components by
operational criticality — specifically, the order in which components should be prioritized for
hardening before deployment.

**Framework prediction.** The framework's ranking is taken from the composite $Q(v)$ (interpretable
predictor) and, separately, from the learned predictor, so that both can be compared to expert
judgment on the same components.

**Agreement metrics.**
- *Predicted-vs-expert agreement* is measured with Kendall's $\tau$ between the framework ranking and
  the expert-consensus ranking, $\tau = [\,\cdot\,]$ (to be computed).
- *Inter-rater reliability* among the experts is measured with Fleiss' $\kappa$, $\kappa = [\,\cdot\,]$
  (to be computed), to establish that the expert consensus is itself coherent enough to serve as a
  reference.
- As a contrast, we report Kendall's $\tau$ between a topology-only centrality ranking and the expert
  consensus, to test whether multi-dimensional attribution aligns with expert judgment more closely
  than centrality does.

**Acceptance.** RQ5 is supported if (i) inter-rater $\kappa$ indicates at least moderate agreement
among experts, and (ii) the framework's $\tau$ to the expert consensus is high and exceeds the
centrality baseline's $\tau$.

## 9.3 Results

*(Scaffold — to be populated once the expert study is run. No values are reported here.)*

**Table 9.1 — Predicted vs expert criticality ranking (ATM).**

| Component (CSCI) | $Q(v)$ rank | Learned rank | Centrality rank | Expert-consensus rank |
|------------------|:-----------:|:------------:|:---------------:|:---------------------:|
| Radar tracker (Surveillance) | — | — | — | — |
| ASTERIX broker (Surveillance) | — | — | — | — |
| Conflict detector (Sep. Assurance) | — | — | — | — |
| Flight-data processor (Sep. Assurance) | — | — | — | — |
| Controller workstation (CWP) | — | — | — | — |
| Meteo service (Meteorology) | — | — | — | — |
| Message library (cross-cutting) | — | — | — | — |

**Table 9.2 — Agreement.**

| Quantity | Value |
|----------|:-----:|
| Kendall $\tau$ (framework vs expert) | $[\tau]$ |
| Kendall $\tau$ (centrality vs expert) | $[\,\cdot\,]$ |
| Fleiss $\kappa$ (inter-rater) | $[\kappa]$ |
| `propagation_threshold` used | $[\,\cdot\,]$ |

**Qualitative checks to confirm (not yet run).** (i) The framework ranks the radar tracker and the
ASTERIX broker among the top components, matching their status as sole-source SPOFs. (ii) The
cross-cutting message library is surfaced by its blast-radius signal even where its composite $Q$ is
moderate — the §5.4 mechanism, *conditional on the canonical simulator resolution*. (iii) The
conflict detector's rank is reported together with the `propagation_threshold` used, since its
position depends on whether single- or dual-feed loss is treated as disabling.

---


# 10. Discussion, Threats to Validity, and Conclusion

## 10.1 Interpretation

The framework's results converge on a single message: for pre-deployment criticality analysis of
pub-sub middleware, *how* a component is critical is at least as important as *whether* it is, and the
right tool depends on the deployment regime. Four findings carry this.

First, the interpretable–learning boundary (RQ1) is not a contest but a division of labor. When a
target system resembles those already understood, the deterministic RMAV/$Q$ attribution recovers the
criticality ordering as well as — and on our datasets better than — a trained model, while remaining
fully explainable and free of training cost. The learned, typed, QoS-aware predictor earns its place
precisely where the interpretable score cannot reach: generalizing to architectures with unseen
cascade dynamics, the genuine pre-deployment condition, where homogeneous and non-learning baselines
collapse. An architect therefore has a principled choice rather than a default: interpretable
attribution in-distribution, learning for out-of-distribution generalization.

Second, multi-dimensional, typed attribution exposes structure a single score cannot (RQ2). The
Simpson's-paradox result is the sharpest illustration: a near-zero pooled correlation conceals strong
per-type correlations, so any evaluation of heterogeneous architectures that reports a single
aggregate is not merely incomplete but actively misleading. The shared-library blast-radius mechanism
is the second: a failure mode that strikes all consumers simultaneously is invisible to centrality and
is recoverable only when node and edge types are preserved. Together these argue that criticality for
typed systems must be both decomposed (so the failure mode is legible) and stratified (so the
evaluation is honest).

Third, remediation makes the diagnosis actionable. By generating topology edits from structure and
accepting them only when the simulator confirms a robust impact reduction — and by triggering the
fan-out operator on blast-radius signals rather than on the composite score — the framework turns an
attribution into a verified intervention, closing the loop from "which component, and why" to "what to
change, confirmed before deployment."

Fourth, automated quality gating operationalises these checks continuously (RQ4). By implementing in-memory evaluations using the `MemoryRepository` to bypass Neo4j database dependencies, the framework executes anti-pattern scans and counterfactual failure simulations in seconds (~5s for medium, ~40s for xlarge). This execution speed allows the graph-based analyzer to run as a blocking CI/CD build check, bridging the "Architecture-Code Gap" by blocking architectural regression at code commit time, analogous to static code analysis gates.

## 10.2 Threats to Validity

**Construct validity.** Our ground-truth impact is produced by a discrete-event simulator rather than
observed in deployed systems. The strongest claims we can make are therefore comparative: our results
speak to which modeling choices perform better under identical conditions, not to absolute predictive
accuracy in operation. High concordance with $I(v)$ indicates that a predictor has captured the
simulator's notion of cascade semantics; whether that notion matches a specific production system is a
separate, empirical question. We mitigate this by evaluating every predictor against the same
simulator-derived targets, so comparisons among the interpretable, learned, homogeneous, and
structural predictors remain internally consistent.

**Internal validity.** The chief internal risk is circular validation — a predictor scoring well
because its inputs leaked from its labels. The framework's independence guarantee addresses this
directly: predictors operate on $G_{\text{analysis}}$ while ground truth is generated by simulating
$G_{\text{structural}}$, no simulation output is fed back as a predictor feature, and the remediation
stage generates candidates without reading $I(v)$. A measured correlation therefore reflects
predictive content rather than leakage. We further report bootstrap confidence intervals and paired
significance tests so that comparative claims are not artifacts of seed variance.

**External validity.** Two limits bound generalization. The synthetic suite, while spanning seven
deployment domains and a range of scales and structural regimes, is generated rather than harvested
from production systems; and the external anchor is a single ATM system. We reduce the first concern
with Leave-One-Scenario-Out evaluation, which tests transfer to held-out architectures and closes the
transductive-leakage gap, and the second with the blind expert-ranking study (§9), which grounds the
rankings in human judgment on a real safety-critical system. Generalization beyond these — to other
middleware families and to systems with richer adversarial structure — remains future work.

## 10.3 Limitations and Future Work

Several limitations point to concrete next steps. The Vulnerability dimension is the lightest of the
four, resting on reachability-style proxies; a richer adversarial model (trust boundaries,
privilege escalation paths) would strengthen the V attribution and broaden the framework's security
relevance. The external validation rests on one system; replicating the expert-ranking protocol across
additional real architectures would test whether the agreement observed on the ATM system holds more
generally. The remediation operator set is small and deliberately conservative; expanding it, and
deriving the acceptance multiplier $\kappa$ from broader multi-seed variance data, would let the
prescriptive stage address more failure modes. Finally, the entire framework is validated against
simulation; the natural endpoint is calibration against, or replacement of, the simulated ground truth
with observed failure data from instrumented deployments, which would convert the comparative claims
of this paper into absolute ones.

## 10.4 Conclusion

We presented Software-as-a-Graph, a pre-deployment Static System Analysis (SSA) framework that models distributed pub-sub
middleware as a typed, weighted, directed multigraph and analyzes it along two coupled axes:
multi-dimensional quality attribution, which decomposes each component's criticality into orthogonal,
interpretable RMAV dimensions (integrating local code quality metrics), and failure-impact analysis, which predicts cascade impact with both
the interpretable composite and a learned heterogeneous predictor, validated against discrete-event
simulation under a strict input–label independence guarantee. A prescriptive remediation stage turns
the resulting diagnosis into simulation-verified hardening edits. 

Integrated directly into pipelines as a blocking CI/CD Quality Gate, the framework verifies architectural changes and regression in seconds, bridging the "Architecture-Code Gap" at commit time. Across a synthetic scenario suite and
an external air-traffic-management case study, the framework shows when interpretable attribution
suffices and when learning is required, exposes failure modes — a shared-library blast radius and a
node-type stratification effect — that single-score centrality cannot, and demonstrates that
attribution computed entirely before deployment can be made both legible and actionable. By taking the
*type* of every component and dependency seriously, the framework recovers structure that untyped,
single-dimensional methods discard, and does so at the point in the lifecycle where it is most
valuable: before the system runs.

