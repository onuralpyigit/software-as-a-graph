# Software-as-a-Graph: Architectural Dependability, Diagnostic Attribution, and Failure-Impact Forecasting in Distributed Publish–Subscribe Systems

**Author:** İbrahim Onuralp Yiğit  
**Advisor:** Prof. Dr. Feza Buzluca  
**Institution:** Istanbul Technical University, Department of Computer Engineering  
**Degree:** Ph.D. Dissertation in Computer Engineering  

---

# Abstract

Publish–subscribe (pub-sub) middleware decouples producers and consumers in time, space, and synchronization, providing high elasticity but obscuring the derived dependency chains along which component failures cascade. Pre-deployment static code analysis (SCA) is blind to multi-tier deployment topologies, while runtime telemetry and chaos engineering do not exist before system instantiation. As a result, software architects face the **Architecture–Code Gap**: identifying *which* components are structurally critical, *why* they fail, and *how* to harden them prior to production deployment remains an open challenge.

This dissertation presents **Software-as-a-Graph (SaG)**, an end-to-end static system analysis (SSA) framework grounded in a typed, weighted, directed multigraph over five architectural component classes ($\text{Applications}$, $\text{Topics}$, $\text{Brokers}$, $\text{Hosts}$, $\text{Libraries}$) with Quality-of-Service (QoS) edge and vertex attributes. The central thesis of this work is that this unified architectural substrate supports **two parametrically independent pathways**:
1. **Pathway A (Diagnostic Attribution):** A deterministic, standards-grounded (ISO/IEC 25010/25019) multi-criteria decomposition that evaluates criticality across Reliability (subdivided into Fault Tolerance and Availability) and Maintainability (RM), extended to relationship criticality ($D_2$) and structural anti-pattern detection.
2. **Pathway B (Predictive Failure-Impact Forecasting):** An empirical forecasting engine pairing homogeneous and heterogeneous graph neural networks (GAT, HGT) with closed-form analytical rankers to predict cascade impact across three structurally disjoint simulation oracles ($I^*$, $I_{\text{dyn}}$, $I_{\text{comp}}$) under an input–label independence guarantee.

At inference time, these co-equal pathways are composed by an architectural **Triage Bridge**, which maps ranked blast-radius shortlists to stakeholder-specific root causes and remediation roles (SRE, System Architect, DevOps, Security), closed by counterfactually verified **Prescriptive Remediation** and continuous CI/CD quality gating.

Across twelve synthetic architectures evaluated under Leave-One-Scenario-Out (LOSO) cross-validation and hand-authored models of five open-source systems (2,812 components, 11,618 edges), this dissertation establishes six key findings:
1. **The Representation Level Dominates Model Complexity:** The derived dependency graph ($G_{\text{dep}}$) accounts for most ranking performance. Closed-form structural references restating the reachability cascade rule ($\text{Analytic } I^*$, $\rho = 0.808$; afferent coupling / direct-dependent count $\text{InDeg}$, $\rho = 0.764$) outperform the training-free betweenness baseline ($\text{Topo-QoS}$, $\rho = 0.553$, corrected to $0.533$ following an articulation point defect repair).
2. **Where Graph Learning Operates:** On the derived dependency graph, an attention-based learner ($\text{GAT-P-QoS}$) approaches the direct-dependent count ($\rho = 0.748$, five-seed ensemble $0.772$), whereas a heterogeneous transformer ($\text{HGT-P-QoS}$) becomes unstable ($\rho = 0.514$). On the raw multigraph, learned rankers ($\text{HGT-QoS}$, $\rho = 0.622$; $\text{GAT-QoS}$, $\rho = 0.635$) achieve parity with the baseline. Hybrids learning a logit-scale correction ($\text{Hybrid-HGT}$, $\rho = 0.657$; $\text{Hybrid-GAT}$, $\rho = 0.683$) significantly beat $\text{Topo-QoS}$ ($p_{\text{Holm}} \le 0.0068$), but do not exceed their own base learners.
3. **Relation Typing vs. QoS Encodings:** In a capacity-matched $2\times2$ factorial design, heterogeneous relation typing adds nothing over homogeneous attention ($\Delta\rho = -0.014$, $p = 0.94$), while the 16-D QoS edge channel provides the active signal ($+0.073$, $p = 0.016$).
4. **Queue-Flow Simulation Surrogates:** For discrete-event queue-flow simulation ($I_{\text{dyn}}$, requiring 12.7 CPU-hours across all 1,321 Applications), a training-free rate-weighted expansion reaches $\rho = 0.830$ in milliseconds, outperforming learned surrogate models ($\text{GBM-P-QoS}$, $\rho = 0.799$; $\text{GIN-P-QoS}$, $\rho = 0.665$) and establishing that declared message rates, not QoS contracts, drive flow disruption.
5. **Zero-Shot Transfer Boundaries:** When transferred zero-shot to five open-source systems, learned models separate inert from active components ($\rho = 0.760$–$0.805$ vs. baseline $0.526$), but fail to discriminate among active propagating components ($\rho_{>0} \le 0.342$), where only transitive reach ($\text{Reach}$, $\rho = 0.938$, $\rho_{>0} = 0.871$) restates the cascade structure.
6. **Cost and Gating Feasibility:** The dependency count executes in $0.4$–$15.6$ ms ($17$–$176\times$ cheaper than simulation), while full feature extraction takes $0.07$–$52.25$ s. Evaluating a blocking CI/CD quality gate requires $0.16$–$81.5$ s. On Green AI lifecycle metrics, upfront simulation and training costs ($0.22$ kWh) are not amortized over the training-free rate-weighted reference.

**Keywords:** publish–subscribe middleware; architectural dependability; static system analysis; cascading failure; heterogeneous graph neural networks; diagnostic attribution; triage bridge; prescriptive remediation; Green AI.

---

# 1. Introduction

## 1.1 Motivation: Pre-Deployment Dependability and Sustainability

The publish–subscribe (pub-sub) communication paradigm is the architectural backbone of modern large-scale distributed systems, including cyber-physical platforms, automotive robotics, cloud-native microservices, industrial SCADA, and IoT telemetries. By decoupling endpoints across time, space, and synchronization, pub-sub architectures allow independent scaling and elastic lifecycle management [1]. Industry standards such as the Data Distribution Service (DDS), MQTT, and event-driven message brokers expose rich Quality-of-Service (QoS) parameters—such as message reliability, durability, deadline constraints, and transport priorities—that dictate communication contracts under operational stress [2, 3].

However, the very decoupling that makes pub-sub architectures flexible introduces a severe dependability liability: **implicit, derived dependency chains**. Because publishers and subscribers interact anonymously through topics and brokers, there are no explicit caller–callee call edges. Failures do not propagate along a static call graph; rather, outages cascade along derived paths—through shared topics, saturated message brokers, colocated physical host nodes, and shared library dependencies that fail simultaneously rather than sequentially. A standard architectural diagram fails to reveal these latent cascade paths, and components that appear peripheral often induce catastrophic system-wide outages.

Crucially, architectural hardening—such as broker replication, message queue partitioning, failover clustering, and topic isolation—is least disruptive and least costly at the design stage. Once deployed to production, architectural remediation is prohibitively expensive. Yet, pre-deployment is precisely when **no runtime telemetry or execution traces exist** to empirically diagnose weak points. Software architects must therefore answer two fundamental questions from architectural specifications alone:
1. *Which components are critical to system survival under failure?*
2. *Why are they critical, and what concrete architectural actions will mitigate that risk?*

Beyond operational dependability, pre-deployment failure prevention is directly linked to **computational sustainability and Green AI**. Uncontained failure cascades in cloud and edge deployments trigger emergency pod restarts, re-transmission packet storms, and failover re-provisioning cycles that expend significant electrical power and infrastructure capacity. Eliminating architectural single points of failure at design time prevents runtime waste, directly supporting sustainable computing practices.

## 1.2 The Architecture–Code Gap and Problem Statement

Static software verification has historically focused on source code. However, a fundamental **Architecture–Code Gap** divides code quality from system resilience:
- A distributed system may consist of modules with pristine code quality, zero SonarQube bugs, and low cyclomatic complexity, yet remain catastrophic at the architectural tier due to single points of failure, missing failover routes, or mismatched QoS contracts.
- Conversely, code-level static analysis (SCA) is blind to inter-process topologies, broker contention, and cross-node cascade dynamics.

Bridging this gap requires **Static System Analysis (SSA)**—shifting structural verification "left" into pre-deployment design reviews and automated continuous integration / continuous delivery (CI/CD) pipelines.

Formally, pre-deployment dependability analysis for pub-sub architectures poses two distinct challenges:
1. **Diagnostic Attribution:** Decomposing criticality into interpretable, standards-grounded quality attributes (e.g., ISO/IEC 25010/25019 Reliability, Fault Tolerance, Availability, and Maintainability) so that an engineer receives an actionable diagnosis (e.g., "Broker lacks failover redundancy" or "Topic has excessive unbuffered fan-out") rather than an opaque score.
2. **Predictive Failure-Impact Forecasting:** Estimating the quantitative cascade impact of a component's failure—measured by system-wide reachability loss or message delivery drops—to rank components for hardening under fixed engineering budgets.

## 1.3 Limitations of Prior Art

Existing verification paradigms leave significant gaps:
- **Static Code Analysis (SCA):** Tools such as SonarQube evaluate intra-module complexity and cohesion, but cannot model distributed pub-sub topologies or middleware cascade mechanics.
- **Runtime Chaos Engineering:** Platforms like Chaos Monkey inject faults into running systems. While valuable for staging, they require deployed infrastructure, carry operational risk, and cannot guide early design decisions.
- **Topological Centrality:** Classical network-science metrics (degree, betweenness, PageRank) collapse a component's risk into an unweighted scalar that conflates distinct failure modes (e.g., articulation points vs. high-throughput hubs) and ignores middleware QoS semantics.
- **Black-Box Machine Learning:** Pure predictive models offer rankings without explainability, leaving engineers unable to determine which architectural remediation operator to apply.

## 1.4 The Dual-Pathway Monograph Architecture

To overcome these limitations, this dissertation formulates **Software-as-a-Graph (SaG)** around a unifying architectural principle that synthesizes our journal and conference contributions:

> **The Dual-Pathway Thesis:** A single typed, weighted architectural graph substrate supports **two parametrically independent pathways**—a deterministic, standards-grounded *diagnostic attribution* (Pathway A) and a learned *predictive ranking* (Pathway B)—which are not competitors but answer distinct engineering questions, are composed at inference by an architectural **Triage Bridge**, and are closed by counterfactually verified **Prescriptive Remediation**.

```
                           [ Architectural Specification ]
                                         │
                                         ▼
                     [ Unified Typed Multigraph Substrate ]
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
   [ Pathway A: Diagnostic ]                               [ Pathway B: Predictive ]
   - RM Decomposition                                      - Order-k References
   - Relationship Criticality (D_2)                        - Learned GNNs & Hybrids
   - Anti-Pattern Smell Catalog                            - Reference Criterion (T_k)
             │                                                       │
             │           ┌───────────────────────────────┐           │
             └──────────►│        TRIAGE BRIDGE          │◄──────────┘
                         │ (Join Top-K Rank with Cause)  │
                         └───────────────┬───────────────┘
                                         ▼
                         [ Prescriptive Remediation ]
                         (Generate -> Counterfactual Verify)
                                         │
                                         ▼
                         [ Continuous CI/CD Quality Gate ]
```

### Why the Dual-Pathway Formulation is Load-Bearing
In our Journal of Systems and Software (JSS) paper, Pathway A was condensed to an explanation baseline because the journal's focus was strictly on the empirical question: *"When does graph learning improve cascade ranking?"* Conversely, in our Automated Software Engineering (AuSE) submission, Pathway A serves as the static audit substrate for automated refactoring.

In this doctoral monograph, the dual-pathway architecture is restored to full prominence. It resolves what would otherwise appear as an empirical contradiction:
- Pathway A's Leave-One-Scenario-Out correlation against cascade simulation is low ($\rho = 0.205$). Under a single-pathway framing, this might be misread as a weak baseline.
- Under the Dual-Pathway Thesis, this low correlation is **empirical proof of separability**: a diagnostic instrument designed to identify stakeholder root causes (fault tolerance vs. maintainability) does not, and should not, mirror stochastic cascade reachability.
- The separation is verified architecturally: the $\lambda_{\text{RM}}$ coupling term in the learned model defaults to $0.0$, and ablating it to $0.1$ proves the two pathways can operate independently without performance degradation.

## 1.5 Research Questions

This dissertation investigates six research questions:

> **RQ1 (Predictive Accuracy & Reference Level):** Where does graph learning improve over analytical baselines and references in predicting cascade failure impact, and how close do learned rankers come to structural references that restate simulation rules?
>
> **RQ2 (Representation vs. Model Complexity):** What do heterogeneous relation typing, edge directionality, and continuous QoS channels contribute when model capacity and edge-feature widths are strictly matched?
>
> **RQ3 (Zero-Shot Transfer to Open-Source Systems):** How effectively do learned rankers transfer zero-shot to independently authored models of open-source distributed systems, and do they reliably order active failure components?
>
> **RQ4 (Computational Feasibility & Green AI):** What is the like-for-like CPU execution cost of simulation versus analytical counting versus GNN inference, and does learned surrogate modeling achieve an amortized lifecycle break-even in CI/CD?
>
> **RQ5 (Diagnostic Attribution & Relationship Criticality):** How effectively does multi-dimensional RM decomposition isolate architectural failure modes, and what does direct edge-removal measurement reveal about relationship criticality ($D_2$)?
>
> **RQ6 (Pathway Composition & Closed-Loop Verification):** How effectively does the Triage Bridge compose ranking with root-cause diagnosis, and can counterfactual prescriptive remediation automatically repair structural vulnerabilities within continuous build gates?

## 1.6 Contributions of the Dissertation

This dissertation makes the following primary contributions:

1. **A Formal Graph Substrate for Publish–Subscribe Systems (§3):** We define the SaG typed multigraph $G_{\text{structural}}$ over five node classes and six edge types, and establish five formal projection rules deriving the logical dependency graph $G_{\text{dep}}$ with QoS-weighted edge metrics and ingested code-level SCA attributes.
2. **Pathway A — Multi-Dimensional Diagnostic Attribution (§4):** We formalize the RM quality decomposition grounded in ISO/IEC 25010/25019, separating Reliability into Fault Tolerance and Availability, and Maintainability, backed by AHP consistency checks, adaptive box-plot classification, and relationship criticality ($D_2$).
3. **The Formal Reference Criterion for Architecture Oracles (§5):** We formulate the Order-$k$ Reference Criterion ($T_k(O)$ under simplifications S1–S5), establishing a rigorous methodological bar that separates true predictive skill from the trivial restatement of simulation rules.
4. **Pathway B — Empirical Failure-Impact Forecasting (§5, §8):** Across twelve synthetic architectures (LOSO cross-validation) and three independent simulation oracles ($I^*, I_{\text{dyn}}, I_{\text{comp}}$), we benchmark analytical references, baselines, GNNs ($\text{GAT-P-QoS}, \text{HGT-QoS}$), and hybrid models, establishing where learning adds value and where closed-form analysis suffices.
5. **The Architectural Triage Bridge (§6):** We implement and validate the composition bridge (`saag/analysis/triage.py`) that joins Pathway B's quantitative Top-$K$ ranking with Pathway A's qualitative diagnosis by component ID, providing graceful cold-start degradation and proven parametric separability.
6. **Prescriptive Remediation and Continuous CI/CD Gating (§7):** We present a two-phase Generate→Verify refactoring engine with four topology operators, counterfactual in-memory graph verification, and an automated blocking CI/CD quality gate executing in seconds.
7. **Empirical Valuation of Representation and Green AI Lifecycle Feasibility (§8):** We provide a capacity-matched $2\times2$ factorial analysis of typing vs. QoS channels, a like-for-like CPU cost benchmarking, and an amortized energy lifecycle break-even analysis under Green AI principles.
8. **Methodology as Contribution and Instrument Defect Account (§10):** We document six silent instrumentation defects discovered during experimental auditing that produced normal-looking wrong figures, and provide the definitive reconciliation of RASSE 2025 findings via Simpson's paradox and oracle maturation.

## 1.7 Publication Map and Contribution Disclosure

In accordance with Istanbul Technical University (ITU) postgraduate guidelines, the doctoral research presented in this dissertation incorporates and extends the candidate's publications:

| Publication / Submission | Venue & Status | Dissertation Chapters | Candidate Contribution Note |
|---|---|---|---|
| **Journal of Systems and Software (JSS)** | Submitted (VSI:AI4MSS, 2026) | Ch. 3, 5, 8, 9, 10 | Primary author. Conceived framework, implemented GNN models, designed reference criterion, executed experiments. |
| **Automated Software Engineering (AuSE)** | Submitted (SI:CI/CD-DevOps, 2026) | Ch. 4, 7, 9 | Primary author. Designed prescriptive remediation operators, counterfactual verifier, and CI/CD quality gate. |
| **IEEE RASSE 2025** | Published (`10.1109/RASSE64831.2025.11315354`) | Ch. 4, 10 | Primary author. Formulated initial structural dependency baseline; superseded and reconciled in Ch. 10. |
| **UYMS 2026 (Patterns)** | Published (National SE Conf.) | Ch. 2, 3 | Co-authored (Çalışkan, Yiğit, Buzluca). Contributed graph schema and publish-subscribe interaction pattern definitions. |
| **UYMS 2026 (Visualization)** | Published (National SE Conf.) | Ch. 3, Appendix | Co-authored (Erşen, Çalışkan, Yiğit, Buzluca). Contributed graph export schemas and visual telemetry interfaces. |

## 1.8 Organization of the Dissertation

The remainder of this dissertation is organized as follows:
- **Chapter 2 (Related Work):** Reviews distributed pub-sub dependability, static code and system analysis, GNNs in software engineering, and multi-criteria scoring.
- **Chapter 3 (The Software-as-a-Graph Model):** Formalizes the typed multigraph, derived dependency projection rules, QoS attribute weighting, and code metric ingestion.
- **Chapter 4 (Pathway A: Diagnostic Attribution):** Details the RM decomposition, AHP weighting, adaptive classification, relationship criticality, and worked examples.
- **Chapter 5 (Pathway B: Failure-Impact Forecasting):** Formalizes simulation oracles, the Reference Criterion, GNN architectures, and hybrid forecasting models.
- **Chapter 6 (Pathway Integration: The Triage Bridge):** Presents the architectural composition layer joining predictive rankings with root-cause diagnoses.
- **Chapter 7 (Prescriptive Remediation and CI/CD Gating):** Formulates the Generate→Verify engine, refactoring operators, and continuous quality gates.
- **Chapter 8 (Experimental Design):** Describes the 17-system evaluation corpus, ranker taxonomy, metrics, preregistration, and statistical protocols.
- **Chapter 9 (Empirical Results):** Reports findings across RQ1–RQ6, including ranking accuracy, typing vs. QoS channels, zero-shot transfer, and like-for-like cost.
- **Chapter 10 (Methodology and Validation Discipline):** Analyzes the six instrument defects, reconciles RASSE 2025, and examines simulation circularity.
- **Chapter 11 (Discussion, Threats, and Conclusion):** Discusses practical triage heuristics, Green AI considerations, threats to validity, and future research directions.

---

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
pre-existing finding (§7.6) — because real architectures legitimately contain *intentional*,
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
empirical question, which we test directly and answer in the negative for our suite (§5.6). Our RM
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
both — since, as §9.1 shows, the answer differs depending on which of those is asked about.

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
profile. Our RM scoring does precisely this, applying the pairwise formalism within each
sub-characteristic, with a shrinkage parameter that blends the stated weighting toward a uniform
prior, and combining the characteristics under declared composite weights (§4.3). We report the sensitivity of that shrinkage rather than assume it
helps: measured against simulated impact it is monotonically harmful, and equal weights outperform
the calibrated vector (§9.1 and JSS Supplementary §S4). The contribution we claim here is therefore explanatory — the
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
into a scoring gap large enough to matter is an empirical question we test in §5.6 (on our synthetic
suite, it does not). *(Figure 2: the running example's structural graph and its derived `DEPENDS_ON` projection, with
sequential-cascade and simultaneous-blast edges visually distinguished.)*

---

# 4. Multi-Dimensional Quality Attribution (The Interpretable Path)

> **Provenance.** This chapter is rebuilt from the RM-era material
> ([`material/rm_attribution.md`](material/rm_attribution.md) and
> [`material/relationship_criticality.md`](material/relationship_criticality.md)), with every formula
> checked against [`saag/analysis/analyzer.py`](../../../saag/analysis/analyzer.py) and the JSS
> explanation-layer section ([`sec5_explanation_layer.md`](../jss/latex/supplementary.tex)).
> Where the material and the code disagreed (the Availability coefficients, the Topic Fault Tolerance
> term, the metrics still computed, the classification rule), the code wins. The worked example of
> §4.6 was re-run through the current pipeline ([`examples/run_running_example.py`](../../../examples/run_running_example.py)).
> It replaces the retired four-dimension RMAV model of earlier preliminary work (IEEE RASSE 2025 [54], reconciled in Chapter 10).

Centrality answers *whether* a component is important with a single number. An architect choosing
between a replica, a reroute, and a decoupling refactor needs to know *why*. This section presents
the framework's primary diagnostic: a decomposition of each component's criticality along the two
ISO/IEC 25010 characteristics Reliability and Maintainability (RM), with Reliability split into its
Fault Tolerance and Availability sub-characteristics, each computed from disjoint structural metrics
and combined into an interpretable composite score. Because the sub-characteristics do not share
inputs, a component's profile is itself the explanation of its risk — and the explanation maps
directly to a remedy (§7).

## 4.1 Two Characteristics, Hierarchical Reliability, and Formal Definitions

We attribute criticality along Reliability and Maintainability (RM). Reliability is **hierarchical**:
its Fault Tolerance and Availability sub-characteristics are scored individually and combined via a
declared blend, $R(v) = \alpha\cdot FT(v) + (1-\alpha)\cdot A(v)$, $\alpha=0.36$. An earlier revision
of this attribution scored Vulnerability/Security as a third peer dimension; it has since been
**retired outright** — not folded into either remaining characteristic — because its ground-truth
evidence was the weakest of the (then) four and no fault-model instrument could validate it by
construction (§5.1). Grounded in **ISO/IEC 25019:2023 (Quality-in-Use)**, criticality represents the
counterfactual loss of beneficialness, freedom from risk, and acceptability experienced by
stakeholders if an architectural element fails. Each dimension speaks to a formal stakeholder class:

**Table 7. The RM dimensions**, the architectural question each answers, and the stakeholder and engineering role each routes to.

| Dim. | Architectural Question | High score means | Harmed Stakeholder (ISO 25019) | Secondary Stakeholder (Engineering Role) |
|:----:|-----------------------|------------------|--------------------------------------------|------------------------------------------|
| **R** (hierarchical) | Combines the two rows below via $\alpha=0.36$ | — | Combined harm of the two rows below | Reliability Engineer / DevOps / SRE |
| ↳ **FT** | How broadly and deeply does failure propagate? | Failure cascades widely; hard to contain | **Primary & Indirect:** operators and downstream beneficiaries whose tasks retry, fail over, or degrade | Reliability Engineer |
| ↳ **A** | Is this a structural single point of failure? | Removing it partitions the dependency graph | **Primary & Indirect:** direct operators (traders, clinicians, drivers) and dependent beneficiaries facing task cessation | DevOps / SRE |
| **M** | How hard is this to change safely? | Tightly coupled structural bottleneck | **Secondary:** maintainers facing high regression likelihood upon refactoring | Software Architect |

Maintainability is the one dimension whose direct victim is the secondary stakeholder; the other
two sub-characteristics route a finding to the engineering role equipped to act on it while
denominating severity in harm to primary and indirect stakeholders. (An earlier revision's $V$ row
was deliberately phrased in terms of *guarantees* rather than asset value — that dimension, and the
distinction, are retired along with it.)

Four formal definitions establish the theoretical construct. Each is stated in full, because
several clauses that are easy to skim past do real work in what follows.

> **Definition D1 — Component Criticality.** The degree to which the failure, latency, or functional
> degradation of a specific software component — directly or transitively — reduces the system's
> capacity to enable its stakeholders to achieve specified operational goals with beneficialness
> (usability, accessibility, suitability), freedom from risk (economic, health, life, environmental),
> and acceptability (experience, trustworthiness, compliance) within its operational context.
> Realised at layer $l$ as a measure $\mathrm{crit}_l : V_l \to [0,1]^2 \times [0,1]$ mapping each
> $v \in V_l$ to $\mathbf{s}(v) = [R(v), M(v)]^T$ (with $R(v)$ itself decomposable into $[FT(v), A(v)]$)
> and composite $Q(v)$.

*"Failure, latency, or functional degradation"* names three distinct fault modes. The structural
estimator does not separate them — RM scores a component's *exposure*, which is why one score
covers all three — whereas the simulation oracle does (§5.1). *"Directly or transitively"* is why
Fault Tolerance exists as a sub-characteristic separate from Availability: the harm is loss of
stakeholder outcomes reachable *through* the component, not loss of graph connectivity. *"Within its
operational context"* is the clause §4.3 operationalises through the QoS-profile adaptation of the
composite weights.

> **Definition D2 — Relationship Criticality.** The degree to which the disruption, latency, or data
> loss across a specific inter-component interaction or dependency path — **with both endpoint
> components remaining operational** — reduces the system's capacity to enable its stakeholders to
> achieve specified goals with beneficialness, freedom from risk, and acceptability, **in proportion
> to the absence of redundant or fallback paths around it**. Realised at layer $l$ as
> $\mathrm{crit}_l : E_l \to [0,1]^2 \times [0,1]$, the same signature as D1.

The first emphasised clause is what makes D2 more than D1 restated for edges: it isolates the
*partial-outage* case, in which the component is up and its dashboards are green while one data flow
has stopped. It is also exactly the condition the edge oracle enforces (§9.5). The second clause
makes replaceability *scale* the harm rather than gate it, which is why only the Availability
sub-characteristic is bridge-gated while FT and M score replaceable links too (§4.7).

> **Definition D3 — Criticality is a consequence, not a risk.** Under the standard decomposition of
> risk into likelihood and consequence, criticality as defined here is the **consequence factor
> alone**. No RM dimension estimates how probable it is that a component or relationship fails;
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
systems. §5.2 and §8.3 carry the corresponding scoping statements.

For **components**, the dimensions are **orthogonal by construction**: each raw structural metric
feeds exactly one of FT, M, A, never more. This is a deliberate design constraint, not an empirical
observation — allowing a metric into two dimensions would silently inflate its weight relative to the
stated weighting (§4.3). Orthogonality is what makes the breakdown legible: a pure single point of
failure scores high on A (and therefore high on R) but low on FT and M; a god-component scores high
on M; a cascade hub scores high on FT (and therefore high on R). The *shape* of the profile names the
failure mode. The constraint is specific to the component decomposition; the edge formulas of §4.7
deliberately relax it in exchange for endpoint context, and we say so there rather than letting the
claim read as framework-wide.

## 4.2 RM Formulas

All metric inputs are rank-normalized to $[0,1]$, so every RM score lies in $[0,1]$. Table 8 fixes
notation for every structural metric the formulas below consume; each is computed once on
$G_{\text{analysis}}$ and feeds exactly one of FT, M, A (§4.1).

**Table 8. RM input metric notation.** $G^\top$ denotes the transpose of the `DEPENDS_ON` graph
(the failure-propagation direction, since edges point dependent → dependency). Two metrics that fed
an earlier revision's retired Vulnerability dimension, REV and RCL, are no longer computed; a third,
the QoS-weighted in-degree $w_{\text{in}}$, survives only as the Topic Fault Tolerance input.

| Symbol | Name | Computed as | Feeds |
|--------|------|-------------|:-----:|
| $\mathrm{RPR}(v)$ | Reverse PageRank | PageRank on $G^\top$ ($d=0.85$) | $FT$ |
| $\mathrm{DG\_in}(v)$ | In-degree (rank-norm.) | Direct dependent count on `DEPENDS_ON` | $FT$ |
| $\mathrm{MPCI}(v)$ | Multi-Path Coupling Index | $\sum_{e\in\text{InEdges}(v)} \max(\text{path\_count}(e)-1,0) / (\lvert V\rvert-1)$ | $FT$ (via CDPot_enh) |
| $\mathrm{CDPot\_enh}(v)$ | Enhanced Cascade Depth Potential | RPR/DG_in blend, amplified by MPCI (Eq. above) | $FT$ |
| $\mathrm{FOC}(v)$ | Fan-Out Criticality | $\log(1+f(t))\cdot s(t)$ over message rate $f$ and subscriber count $s$, max-normalised (Topic nodes only) | $FT_{\text{topic}}$ |
| $\mathrm{BT}(v)$ | Betweenness centrality | Brandes' algorithm on $G_{\text{analysis}}$, QoS-inverted edge distances | $M$ |
| $w\_\text{out}(v)$ | QoS-weighted out-degree | $\sum_{(v,u)} w(v,u)$ over outgoing dependencies | $M$ |
| $\mathrm{CQP}(v)$ | Code Quality Penalty | SonarQube-derived composite (§3.4); 0 for non-App/Library types | $M$ |
| $\mathrm{CouplingRisk\_enh}(v)$ | Enhanced coupling risk | in/out-degree balance amplified by path complexity | $M$ |
| $\mathrm{CC}(v)$ | Clustering coefficient | Watts–Strogatz local clustering on the undirected projection | $M$ (as $1-\mathrm{CC}$) |
| $\mathrm{AP\_c\_directed}(v)$ | Directed articulation score | $\max$ of directed in/out articulation scores | $A$ |
| $\mathrm{QSPOF}(v)$ | QoS-weighted SPOF severity | $\mathrm{AP\_c\_directed}(v)\cdot w(v)$ | $A$ |
| $\mathrm{BR}(v)$ | Bridge ratio | fraction of $v$'s undirected edges that are bridges | $A$ |
| $\mathrm{CDI}(v)$ | Connectivity Degradation Index | normalized increase in average path length when $v$ is removed, computed for every node of the main connected component | $A$ |
| $w(v)$ | Node QoS weight | aggregate criticality of the transport contracts incident on $v$ (§3.2) | $A$ (directly and via QSPOF) |
| $w\_\text{in}(v)$ | QoS-weighted in-degree | $\sum_{(u,v)} w(u,v)$ over incoming dependencies | $FT_{\text{topic}}$ (via CDPot_topic) |

**Reliability** is a hierarchical blend of its two sub-characteristics:

$$R(v) = \alpha\cdot FT(v) + (1-\alpha)\cdot A(v), \qquad \alpha = 0.36$$

**Fault Tolerance** — fault-propagation risk. Because `DEPENDS_ON` points *dependent → dependency*, a
failure propagates *against* edge direction; RPR (computed on the transpose $G^\top$) therefore
traverses the natural failure-propagation path. For Topic nodes, which have no `DEPENDS_ON`
in-degree, a fan-out form is dispatched by $\tau_V(v)$:

$$FT(v) = 0.45\cdot\mathrm{RPR}(v) + 0.30\cdot\mathrm{DG\_in}(v) + 0.25\cdot\mathrm{CDPot\_enh}(v)
\qquad [\tau_V(v)\neq\text{Topic}]$$
$$\mathrm{CDPot\_enh}(v) = \min\!\Big( \frac{\mathrm{RPR}(v) + \mathrm{DG\_in}(v)}{2} \cdot \big(1 - \min(\tfrac{\mathrm{out\_degree\_raw}(v)}{\max(\mathrm{in\_degree\_raw}(v),\, \epsilon)}, 1)\big) \cdot (1 + \mathrm{MPCI}(v)),\ 1.0 \Big)$$
$$FT_{\text{topic}}(v) = 0.50\cdot\mathrm{FOC}(v) + 0.50\cdot\mathrm{CDPot\_topic}(v),\quad
\mathrm{CDPot\_topic}(v) = \mathrm{FOC}(v)\big(1 - \min(w\_\text{in,norm}(v),1)\big)$$

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

**Availability** — single-point-of-failure risk; a Reliability sub-characteristic, feeding $R(v)$
above rather than the composite directly:

$$A(v) = 0.25\,\mathrm{AP\_c\_directed}(v) + 0.20\,\mathrm{QSPOF}(v) + 0.20\,\mathrm{BR}(v)
+ 0.25\,\mathrm{CDI}(v) + 0.10\,w(v).$$

The directed articulation score (rather than the undirected AP, which both over- and under-reports
in pub-sub graphs) captures directed cut vertices; QSPOF amplifies it by the component's QoS weight,
so a SPOF carrying critical traffic is scored as doubly severe. CDI is computed for every node rather
than only for articulation points: gating it on articulation leaves it identically zero wherever a
removal does not literally disconnect the graph, which makes $A(v)$ near-constant in exactly the
redundant multi-publisher topologies SaG targets. It is also the dominant cost of the analysis stage
(§9.4).

(An earlier revision scored a fourth dimension, **Vulnerability** — adversarial exposure via
$V(v) = 0.40\,\mathrm{REV}(v) + 0.35\,\mathrm{RCL}(v) + 0.25\,\mathrm{w\_in}(v)$, all three terms
computed on the transpose to model attack propagation. It has been retired outright, not folded into
FT, A, or M; REV and RCL are no longer computed, and $w_{\text{in}}$ survives only as the Topic
Fault Tolerance input above.)

## 4.3 The Composite Score $Q(v)$

The two characteristics combine into a composite criticality score:

$$R(v) = \alpha\, FT(v) + (1-\alpha)\, A(v), \qquad Q(v) = w_R\, R(v) + w_M\, M(v).$$

**The weights are DECLARED constants, not AHP output.** An earlier revision derived the (then) four
composite weights from a $4\times4$ AHP comparison matrix, audited for coherence rather than
elicited. With only two composite terms remaining, a $2\times2$ Saaty matrix would be consistent by
construction ($\mathrm{CR}=0$ for $n\le2$) and would contribute nothing beyond whichever single free
parameter is chosen — so AHP is retired at the composite level entirely. AHP remains in use for the
genuinely multi-term *intra*-dimension vectors ($FT$'s 3 terms, $M$'s 5, $A$'s 5), where a $3$–$5$
dimensional matrix can have a non-trivial consistency ratio (the shrinkage sweep concerns only
those vectors, not the composite — see §4.3 below and JSS Supplementary §S4). Of the five AHP matrices, however, three are rank-one,
back-filled from a chosen vector, so their consistency ratios certify nothing; only the Topic-QoS
matrix ($\mathrm{CR} = 0.016$) and the Fault-Tolerance matrix carry genuine second-eigenvalue spread
(JSS Supplementary §S4).

**$w_R=0.80$, $w_M=0.20$, and $\alpha=0.36$ are a pure re-parameterisation of the retired composite,
not an independently invented weighting.** They are derived algebraically from the old $4$-D vector
$(A{=}0.43, R{=}0.24, M{=}0.17, V{=}0.16)$ by dropping $V$ and renormalising, then folding $A$ into
$R$:

$$\alpha = \frac{0.24}{0.24+0.43} = 0.3582 \to 0.36, \qquad
w_R = \frac{0.24+0.43}{0.84} = 0.7976 \to 0.80, \qquad
w_M = \frac{0.17}{0.84} = 0.2024 \to 0.20.$$

At exact (unrounded) values this recovers the old composite's $A$/$R$/$M$ shares exactly; the 2-s.f.
rounding introduces a small, bounded drift ($\le 0.003$ per term) rather than a fresh judgement.

**A QoS-profile adaptation is applied on top.** Before scoring, the composite coefficients are
re-derived from the analysed system's aggregate topic QoS
profile. The rule averages three fractions of the system's topics — high-durability
(`PERSISTENT`/`TRANSIENT_LOCAL`/`TRANSIENT`), `RELIABLE`, and high or critical priority — into a
reliability signal. At $0.6$ or above, weight moves toward $R$; at $0.4$ or below, toward $M$; the
shift is $\min(0.15,\ 0.30\,|\text{signal} - 0.5|)$, and the two weights are renormalised to sum to
one. This is D1's *"within its operational context"* clause made computable, and it is on by default.
The effective composite is therefore **per system**, so the constants above are starting points
rather than the coefficients any individual system is scored with — a further sense in which D4's
relativity holds — and the adaptation does not disturb the determinism of §4.5, being a
deterministic function of the same $G_{\text{analysis}}$.

Until recently the adaptation never saw a profile. Both repositories store topic QoS as flat
`qos_reliability`, `qos_durability` and `qos_transport_priority` properties, but the profile reader
looked only for a nested `qos` field, so every system's profile was empty, its reliability signal was
$0$, and every system with topics was scored as volatile and best-effort: $w_R = 0.65$, $w_M = 0.35$.
The reader now accepts both shapes, pinned by `tests/test_qos_profile_adaptation.py`. Every RM /
$Q(v)$ figure in Chapter 9 was produced before the fix; any that went through the default adaptation carries
the defect, and none has yet been re-run. The worked example of §4.6 is scored with the fix.

**Quality-in-Use Transformation Matrix.** To connect product-quality mechanisms ($R, M$) to ISO/IEC 25019 Quality-in-Use harms, the vector $\mathbf{s}_{\mathrm{RM}}(v) = [R(v), M(v)]^T$ projects into stakeholder harm scores $[H_{\mathrm{Ben}}, H_{\mathrm{Risk}}, H_{\mathrm{Acc}}]^T$ via transformation matrix $\mathbf{M}_{\mathrm{RM} \to \mathrm{QiU}}$:

$$
\mathbf{h}_{\mathrm{QiU}}(v) = \mathbf{M}_{\mathrm{RM} \to \mathrm{QiU}} \cdot \mathbf{s}_{\mathrm{RM}}(v) =
\begin{bmatrix}
0.75 & 0.25 \\
0.80 & 0.20 \\
0.60 & 0.40
\end{bmatrix}
\begin{bmatrix} R(v) \\ M(v) \end{bmatrix}.
$$

(Row 1 is an unchanged mechanical fold of the old $A$ column into $R$ — $A$'s coefficient there was
already $0.00$ in the retired $3\times4$ matrix. Rows 2 and 3 are re-declared, not mechanically
derived: the mechanical fold alone collapses them to the same vector, making the matrix rank-1 in
$\vec\omega$ and tying every domain's Beneficialness weight — disqualifying. Row 2's re-declared
$0.20$ on $M$ is maintainability's MTTR channel; row 3's $0.40$ is its evolvability-into-trust
channel.) The matrix is row-stochastic, so any domain-weighted projection onto Quality-in-Use harm,
$\vec\omega^\top \mathbf{M}_{\mathrm{RM}\to\mathrm{QiU}}\,\mathbf{s}_{\mathrm{RM}}(v)$, is
algebraically the same RM vector scored under the reweighted composite $\mathbf{M}^\top\vec\omega$.
A Quality-in-Use scalarisation is therefore a reweighting of $Q(v)$, never an independent score, and
we do not report one as a distinct quantity.

In a specific deployment domain, Quality-in-Use loss can be further parametrized by a **Domain
Context Vector** $\vec{\omega}_{\mathrm{domain}} = [\omega_{\mathrm{Ben}}, \omega_{\mathrm{Risk}},
\omega_{\mathrm{Acc}}]$ that reweights the three harm scores — safety-critical ROS 2 prioritising
Freedom from Risk, financial HFT prioritising Efficiency under Beneficialness, and so on.

**Both $\mathbf{M}_{\mathrm{RM}\to\mathrm{QiU}}$ and $\vec{\omega}_{\mathrm{domain}}$ are stated
mappings.** They are given here because D1 and D2 define criticality on Quality-in-Use while the
dimensions are named after product quality, and a reader is owed an explicit statement of how one is
meant to reach the other. The coefficients are asserted, not fitted or elicited; they carry no
consistency audit. Per-domain reweighting *has* been measured (`reproduce/domain_weight_comparison.py`;
JSS Supplementary §S3): domain-derived and static composite rankings agree at mean Kendall
$\tau = 0.980$, and the domain-derived weighting is marginally the worst of the three alternatives
(mean $\rho = 0.318$, against $0.331$ for equal and $0.321$ for static weights). The derivation's
value is attributional (explaining criticality in stakeholder terms), not a ranking-improvement
device — see §9.1 and JSS Supplementary §S3.

**We report the sensitivity of the intra-dimension weighting, and it is not favourable.** The
composite is $\lambda$-invariant by construction, since every point of the sweep uses the same
$w_R$ and $w_M$, so the shrinkage parameter $\lambda$ characterises only the FT, M and A vectors.
Sweeping it from a uniform prior ($\lambda = 0$) to the raw AHP judgement ($\lambda = 1$) against
simulated impact shows a monotone decline in mean $\rho$, from $0.319$ to $0.200$, with no plateau;
the default $\lambda = 0.70$ lies between them (JSS Supplementary §S4, Table S3). Expert elicitation makes the ranking
worse. We keep $\lambda = 0.70$ because $Q(v)$ is an attribution instrument, not a ranking model:
tuning $\lambda$ toward zero would improve a number we do not claim at the cost of the traceability
we do. What the sweep establishes is nonetheless a genuine limitation — we have evidence that the
elicited weights produce worse rankings and none that they produce better attributions.

The characteristics earn their place by being *separately actionable* — a structural
single point of failure and a cascade hub have different owners and different remedies even at
identical composite scores (§4.1) — and that property is independent of how they are combined into a
scalar. A practitioner who needs to know *why* a component is critical needs the profile, whatever
the weights.

## 4.4 Adaptive Criticality Classification

A raw $Q(v)$ is most useful when turned into an action threshold relative to the system's own
distribution rather than an absolute cutoff. We classify with an adaptive box-plot rule, applied
independently to the composite, to each characteristic and to each sub-characteristic:

$$
\text{CRITICAL}: Q > Q_3 + k\,\mathrm{IQR};\quad
\text{HIGH}: Q_3 < Q \le \text{upper fence};\quad
\text{MEDIUM}: \mathrm{med} < Q \le Q_3;
$$
$$
\text{LOW}: Q_1 < Q \le \mathrm{med};\quad
\text{MINIMAL}: Q \le Q_1,
$$

with $k = 0.75$ in the shipped analyzer, tighter than Tukey's conventional $1.5$. Across the corpus,
a mean of $4.2\%$ of components fall above the fence (JSS §5.2).

**Tiers are assigned within node type.** A single fence over Applications, Brokers, Topics, Nodes and
Libraries ranks each component against populations whose score scale it does not share; on the
eight-scenario corpus, stratifying by type instead of pooling moves $62.8\%$ of components to a
different tier and changes CRITICAL/HIGH membership for $19.0\%$ of them
(`results/tier_pooling_check.json`). Each type therefore gets its own quartiles and fence, and a type
with fewer than eight members falls back to the pooled fence rather than being scored on a handful of
samples.

**Small layers use a rank-based fallback.** Below twelve scored components, quartile fences are
unstable, so tiers are assigned by rank position instead: the top $10\%$ are CRITICAL, the next
$15\%$ HIGH, then MEDIUM to the median, LOW to the 75th percentile, and MINIMAL below. Because the
fallback ranks by position, tied components can land in adjacent tiers, as two symmetric subscribers
do in §4.6; a tier from the fallback should be read together with its score.

Per-dimension classification is what makes the output actionable: a component can be CRITICAL on
Availability yet MINIMAL on Maintainability, which tells the architect to add a replica rather than
to decouple an interface.

## 4.5 Determinism and the Independence Guarantee

Attribution is fully deterministic and interpretable: the same $G_{\text{analysis}}$ always yields
the same scores, with no learned parameters and no stochastic component. Critically, every input to
$Q(v)$ is a structural metric of $G_{\text{analysis}}$; none derives from the discrete-event
simulation that produces the ground-truth impact labels used to evaluate the framework (§5.1, §9.1).
This is the **independence guarantee**: the attribution path and the label path are disjoint, so a
correlation between $Q(v)$ and simulated impact — under either oracle — measures genuine predictive
content rather than information leaked from the labels into the score.

## 4.6 Worked Attribution

Scoring the running example of §3.6 with the pipeline of §4.2–§4.4 gives the following profile. The
topic's message rate is not stated in §3.6; at 14 Hz its weight reproduces the stated $w(t) = 0.596$,
and that is the rate used here. The system's only topic is reliability-critical on all three
counts, so the adaptation of §4.3 shifts the composite to $w_R = 0.95$, $w_M = 0.05$, and $Q$ is
almost $R$. The seven components fall below the twelve-component threshold, so tiers come from the
rank-based fallback of §4.4. The point of the table is the divergence between the
last two columns.

**Table 9. Worked RM attribution for the running example of §3.6.** Regenerated by
`python examples/run_running_example.py`. $Q$ uses the adapted composite ($w_R = 0.95$,
$w_M = 0.05$). Rows are ordered by $Q$.

| Component | $FT$ | $A$ | $R$ | $M$ | $Q$ | Composite tier | Profile tiers worth reading |
|---|---:|---:|---:|---:|---:|---|---|
| $b$ (broker) | 0.569 | 0.453 | 0.495 | 0.278 | 0.484 | CRITICAL | **CRITICAL on $A$ and $R$**, LOW on $M$ |
| $t$ (topic) | 0.875 | 0.042 | 0.342 | 0.305 | 0.340 | HIGH | **CRITICAL on $FT$** |
| $n$ (host) | 0.300 | 0.242 | 0.263 | 0.405 | 0.270 | MEDIUM | HIGH on $A$, MINIMAL on $FT$ |
| $a_1$ (publisher) | 0.500 | 0.053 | 0.214 | 0.522 | 0.230 | MEDIUM | **CRITICAL on $M$** |
| $\ell$ (library) | 0.450 | 0.100 | 0.226 | 0.252 | 0.227 | LOW | MEDIUM on $A$ and $R$ |
| $a_2$, $a_3$ (subscribers) | 0.487 | 0.042 | 0.202 | 0.477 | 0.216 | LOW / MINIMAL (tie split by the fallback) | HIGH / MEDIUM on $M$ |

Three components illustrate how the profile names the failure mode, and two are cases the composite
alone would mislead on. The broker $b$ is a directed cut vertex: removing it partitions the graph, so
it is CRITICAL on $A$ — driven by the directed articulation score, the Connectivity Degradation
Index and, because $t$ carries `RELIABLE`/`TRANSIENT_LOCAL`/`HIGH` traffic at $w(t) = 0.596$, by
QSPOF — while scoring LOW on $M$. Because Availability now carries $64\%$ of Reliability, the
composite agrees: $b$ is CRITICAL overall, and the $A$ tier routes it to the SRE for a second broker.
(The retired four-dimension model ranked the same broker LOW overall, because $A$ was one of four
peer terms; folding it into $R$ is what closed that gap.) The topic $t$ is where the composite still
misleads: CRITICAL on $FT$ through its subscriber fan-out, only HIGH overall, and its low $A$ says
the risk is propagation rather than partition — a circuit breaker, not a replica. The publisher
$a_1$ is CRITICAL on $M$ — a betweenness and efferent-coupling bottleneck the architect should
decouple — at a composite of only MEDIUM.

The adaptation of §4.3 is visible here. At the declared $0.80/0.20$ the composite would be $0.452$
for $b$, and $n$ ($0.291$) would still rank above $a_1$ ($0.276$); the adapted $0.95/0.05$ split
widens that gap ($0.270$ against $0.230$), because this system's traffic is uniformly
reliability-critical and Maintainability barely enters $Q$. That is exactly the case where the
profile matters most: $a_1$'s coupling risk is invisible in the composite and visible only in its
$M$ tier. (Before the defect of §4.3 was fixed, the same system was scored at $0.65/0.35$, which
reversed the order and put $a_1$ above $n$.)

This is the concrete form of the claim §9.1 arrives at empirically. The composite is a ranking
device and, on this cohort, not a good one; the *profile* is the diagnostic, and the two disagree
often enough that reading only the scalar discards the finding. Reading the broker row as a
stakeholder statement, in the terms of §4.1: if $b$ fails, $a_2$ and $a_3$ lose their only path to
$t$, so the monitoring task does not degrade — it stops. That is a Beneficialness/Effectiveness
loss, and the outage window is itself a Freedom-from-risk exposure. What the score does *not* say is
how often $b$ fails or how fast it would be restored (D3); a CRITICAL tier is a statement about
structural exposure to Quality-in-Use loss, not a measurement of Quality-in-Use loss itself (§9.5, §10.3).

The edge scores of §4.7 add the complementary reading. Only one dependency in the example is a
bridge — $n \to b$, at $A = 0.348$ and $Q = 0.335$, the highest-scoring edge and the only CRITICAL
one — while every `app_to_broker`, `app_to_app` and `app_to_lib` edge scores $A = 0.008$–$0.011$:
replaceable links, whose loss costs Efficiency rather than Effectiveness. Those replaceable edges
nonetheless carry $FT \approx 0.29$–$0.31$, because $w(e) = 0.596$ and their endpoints' own fault
tolerance reach them through the endpoint term. That is D2's proportionality clause behaving as
specified: redundancy scales the harm to near zero on $A$ without switching Fault Tolerance and
Maintainability off.

The shared library $\ell$ illustrates the qualitatively distinct simultaneous-blast mechanism of
Rule 5 (§3.3): its individual structural centrality is unremarkable — it is LOW overall — yet its
failure collapses $a_1, a_2, a_3$ at once, in a single event rather than a propagation chain. Whether
this mechanism produces a low-$Q$/high-$I$ mismatch in practice is an empirical question we evaluate
directly in §5.6 (on our synthetic suite, it does not); independent of that, the mechanism is why the
FanOutReduction operator (§7.2) is triggered by structural blast signals rather than by $Q(v)$ itself
— a library's consumer fan-out is legible from structure alone, before any simulation is run.

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

**Table 10. Per-edge structural signals** computed on $G_{\text{analysis}}$ for relationship criticality.

| Signal | Computed as | Reads as |
|---|---|---|
| $\mathbf{1}_{\text{bridge}}(e)$ | cut-edge test on the undirected projection | removing $e$ disconnects a subgraph — the Effectiveness case |
| $\mathrm{bt}(e)$ | edge betweenness on **inverted** weights, each edge's length $1/w(e)$ | fraction of shortest dependency paths crossing $e$ — the Efficiency case (how much traffic must reroute) |
| $w(e)$ | worst-case (max) QoS weight over the topics mediating the dependency (§3.3) | how strongly the flow across $e$ is guaranteed |
| $\text{path\_count}(e)$ | number of distinct mediating topics or shared hosts | coupling intensity, kept out of $w(e)$ to preserve $w\in[0,1]$ |

Weight inversion is what makes strongly-guaranteed dependencies *short*, so they attract shortest
paths rather than repelling them. Unlike the node case, $w(e)$ enters **un-normalised**: the §3.2
construction already places it in $[0,1]$.

**Edge RM.** Each edge is scored on Fault Tolerance and Availability (Reliability's sub-characteristics)
and Maintainability, blending its intrinsic signals with the endpoint scores of §4.2:

$$FT(u,v) = 0.35\,\mathrm{bt} + 0.30\,w(e) + 0.20\max\big(FT(u), FT(v)\big)$$
$$A(u,v) = 0.30\,\mathbf{1}_{\text{bridge}} + 0.20\min\big(A(u), A(v)\big)$$
$$R(u,v) = \alpha\, FT(u,v) + (1-\alpha)\, A(u,v), \qquad \alpha=0.36$$
$$M(u,v) = 0.35\,\mathrm{bt} + 0.30\,\mathbf{1}_{\text{bridge}} + 0.15\,w(e)$$

combined into $Q(u,v) = w_R\,R(u,v) + w_M\,M(u,v)$ with the same composite coefficients ($w_R=0.80$,
$w_M=0.20$) and QoS-profile adaptation as a node (§4.3), and classified by the same box-plot rule
(§4.4) applied within the edge set. (An earlier revision of this construction also scored a
Vulnerability edge term, $V(u,v) = 0.15\,w(e) + 0.20\max(V(u), V(v))$ — retired outright along with
the node-level dimension, not folded into $FT$, $A$, or $M$.)

Four design choices carry meaning. **$\max$ for $FT$, $\min$ for $A$**: a link is only as
fault-tolerant as its *riskiest* endpoint, since failure on either side propagates across it, but
only as available as its *weakest*, since the edge cannot be more resilient than the more fragile
side it connects. **$\mathbf{1}_{\text{bridge}}$ appears in both $M$ and $A$**:
a non-redundant edge is expensive to route around (an Efficiency cost to the engineering stakeholder)
*and* a structural cut-point if removed (an Effectiveness loss to the end user) — one structural
fact, two stakeholder consequences. **$w(e)$ appears in $FT$ and $M$ but not $A$**: the guarantee
crossing a link scales how much its loss costs, but not whether it can be lost at all. Replaceability
is topological; consequence is QoS-weighted. This is D2's redundancy clause made operational — only
$A$ is bridge-gated, while $FT$ and $M$ score replaceable links too. **$\text{path\_count}$ does
not enter the edge score directly**; it shapes the endpoints' $FT$ and $M$ (§4.2), of which only $FT$
reaches the edge again, through the endpoint term.

**Two scoping conditions.** First, the orthogonality constraint of §4.1 is a property of the *node*
decomposition and does not carry over here: $\mathrm{bt}$ feeds both $FT$ and $M$,
$\mathbf{1}_{\text{bridge}}$ feeds both $M$ and $A$, and $w(e)$ feeds both $FT$ and $M$. The edge
formulas trade orthogonality for the endpoint context that distinguishes an edge score from a node
score, and we state the claim as node-scoped rather than framework-wide. Second, the edge terms do
not draw on equal coefficient mass — $FT$ sums to $0.85$ of a possible $1.0$, $M$ to $0.80$, $A$ to
$0.50$ — so raw edge scores are comparable *within* a term but not *across* terms. Because
classification is box-plot relative within the edge set, per-term rankings and tiers are unaffected;
only the raw magnitudes are. An edge's term *tiers* should be read, not its absolute values.

**What validates this, and what does not.** Relationship attribution is scored over
$G_{\text{analysis}}$ — the derived `DEPENDS_ON` edges — while the edge-removal oracle of §9.5 severs
raw edges of $G_{\text{structural}}$. On the current `av_system` those are 3,556 derived edges against a
candidate set of 50 raw structural edges drawn predominantly from `RUNS_ON` and `CONNECTS_TO`, with a
handful of `SUBSCRIBES_TO` and `PUBLISHES_TO` relations (§9.5 gives the exact composition), and the
two populations barely intersect. This is not an oversight: it is the independence guarantee of §5.2
operating exactly as designed — predictors and labels must be computed over disjoint graph views —
and the edge case simply has no shared identifier space for the two views to meet on, where the node
case does. **There is therefore no common edge population on which $Q(u,v)$ and the measured edge
impact are both defined**, and the correlation-style validation applied to node scores in §9.1 cannot
be run for edges as the two quantities are currently constructed. We present relationship attribution
as a *defined and implemented* measure that operationalises D2, and the edge-removal measurement of
§9.5 as a separate result about the structural graph — not as a validation of the attribution, and we
do not report or imply a correlation between the two anywhere in this paper. Re-simulating on
`DEPENDS_ON` directly is not an available fix: the framework's independence guarantee (§5.2) requires
simulation to operate only on $G_{\text{structural}}$. The one route that would close the gap without
violating that guarantee — tracking, for each derived edge, which raw structural edges mediate it,
then aggregating their measured impact onto it — is a modelling exercise in its own right (the
mediating relations are many-to-many, so the aggregation rule is a choice, not a formality) and is out
of scope for this submission; we position it as future work in §11.3 rather than as a pending fix.

# 5. Pathway B — Failure-Impact Forecasting, Ranking Methods, and Simulation Oracles

Quality attribution (Pathway A, §4) tells an architect *why* a component is structurally exposed and decomposes that exposure into actionable quality characteristics. This chapter presents **Pathway B**, which addresses the complementary operational question: *how much of the system actually fails* when a component is disrupted, and how well can analytical and machine learning methods forecast that cascade impact prior to deployment?

We define the three simulation oracles that supply ground truth (§5.1), formulate the **Reference Criterion** and input–label independence guarantee that separate genuine predictive skill from oracle restatement (§5.2), specify the spectrum of analytical references, baselines, graph neural networks, and hybrid rankers (§5.3), define the multi-task training objective (§5.4), analyze the theoretical motivation for learned predictors versus direct simulation (§5.5), and report two structural analyses on shared-library blast mechanics (§5.6) and stratified consistency (§5.7).

## 5.1 Ground Truth: Three Simulation Oracles

In the absence of runtime telemetry, failure impact is quantified by three discrete-event simulation oracles operating over the raw multigraph $G_{\text{structural}}$ (traversing `PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, and `USES` edges directly, without reading the derived `DEPENDS_ON` projection). Each oracle injects a failure at component $v$, propagates disruptions according to distinct middleware dynamics, and scores the resulting degradation.

The three oracles define fundamentally different notions of failure impact and are not interchangeable:

1. **Reachability Cascade Oracle ($I^*$, Primary Ground Truth):** Produced by `FaultInjector`. This oracle models topological reachability loss under a stochastic breadth-first cascade. A component $v$ is faulted, and outages propagate through downstream topics, message brokers, and physical nodes. Intact subscribers experience fractional feed loss. Each topic's outage is rescaled by declared QoS severity weights ($\times 1.2$ for `RELIABLE`, $\times 1.15$ or $\times 1.05$ for high or medium priority). A subscriber fails and propagates outages if its feed loss exceeds a propagation threshold ($0.2$), with failure probability decaying across waves ($1.0$ at wave 1, decaying by $0.15$ per wave, floor $0.25$). The primary label $I^*(v)$ is the across-seed mean of fractional feed loss over five stochastic runs. An auxiliary target $I^*_R(v)$ records the raw count of impacted subscribers. Disabling QoS scaling preserves ranking at $\rho = 0.965$, confirming that $I^*$ is predominantly a topological reachability metric.
2. **Queue-Flow Simulator ($I_{\text{dyn}}$, Dynamic Behavioral Ground Truth):** Produced by `MessageFlowSimulator`. A continuous discrete-event simulation implemented in SimPy [4] modeling declared publication rates ($r_t$), bounded subscriber FIFO queues, service times, and queue contention. It measures the drop in delivered message rate experienced by *surviving* consumers:
   $$I_{\text{dyn}}(v) = \text{delivery\_rate}_{\text{before}} - \text{delivery\_rate}_{\text{after}}.$$
   Because it models message delivery drops rather than cascading component failures, its primary effect is direct consumer starvation, propagating no failure beyond one hop by construction. Evaluating all $1{,}321$ Applications across twelve topologies (five seeds) requires **12.7 CPU-hours**, compared to seconds for $I^*$. Across the corpus, $I_{\text{dyn}}$ agrees with $I^*$ at mean $\rho = 0.711$. Across-seed test–retest reliability ranges from $r = 0.79$ to $0.99$ (Spearman–Brown), establishing a theoretical reproducibility ceiling ($\sqrt{r} \approx 0.89$–$0.996$).
3. **Composite Multi-Criteria Simulator ($I_{\text{comp}}$, Topological Stress Test):** Produced by `FailureSimulator`. It evaluates structural graph damage across four weighted dimensions:
   $$I_{\text{comp}}(v) = 0.35\,\text{reachability\_loss} + 0.25\,\text{fragmentation} + 0.25\,\text{throughput\_loss} + 0.15\,\text{flow\_disruption}.$$
   $I_{\text{comp}}$ measures physical topology partition, graph fragmentation, and path severance. It is never used as a training label for learned rankers; it serves as an independent stress test, backing the validation gates and prescriptive remediation verification (§7).

$I^*$ and $I_{\text{comp}}$ agree only moderately (mean $\rho = 0.395$ on Applications), and $I_{\text{comp}}$ and $I_{\text{dyn}}$ agree at $\rho = 0.411$. We therefore enforce strict construct boundaries: empirical performance on one oracle does not license claims regarding another.

## 5.2 Input–Label Separation and The Reference Criterion

The framework enforces strict **input–label separation**: no simulation output is ever provided as an input feature to any predictor (enforced by automated regression tests in `tests/test_analyzer.py`).

However, procedural separation does not prevent **construct overlap**: a ranker may achieve high correlation with an oracle simply because it restates the oracle's algorithmic definition, rather than because it infers impact from independent evidence. To eliminate this ambiguity, this dissertation introduces the **Order-$k$ Reference Criterion**:

### The Reference Criterion
Let an oracle $O$ compute the impact of removing component $v$ by propagating failures in waves over its inputs, and let $T_k(O)$ be that computation stopped after wave $k$. A ranking metric $R$ is an **order-$k$ reference** for $O$ ($k \ge 1$ or $k = \infty$) if $R$ equals $T_k(O)$ computed from the inputs $O$ reads, after admitting at most five formal simplifications:
- **(S1) Expectation:** Replacing stochastic seed-dependent propagation with expected values.
- **(S2) Uniform Scaling:** Omitting oracle-specific severity ladders (e.g., QoS multipliers).
- **(S3) Mechanism Elision:** Omitting dynamic mechanisms layered on top of propagation (e.g., queueing delays, buffer drops, deadlines).
- **(S4) Uniform Weighting:** Applying uniform weights in place of per-topic message rates or normalizations.
- **(S5) Wave Support:** Measuring the cardinality of affected components (support) rather than weighted fractional losses.

No other simplification is permitted—specifically, no reversal of propagation edges, no replacement of wave frontiers by spectral graph scores, and no parameters fitted to simulation output.

### Derived Order-$k$ References
Under this formal definition, we classify several training-free structural metrics:
1. **Analytic $I^*$ ($\hat{I}^*_1$, Order-1 Reference for $I^*$):** The first-order analytical expansion of the reachability oracle under S1, S2, and S4:
   $$\hat{I}^*_1(v) = \sum_{t \in \text{pub}(v)} \frac{|\text{sub}(t)|}{|\text{pub}(t)|}.$$
2. **$\text{InDeg}$ (Direct-Dependent Count, Support of Wave 1 for $I^*$):** Afferent coupling on the derived dependency graph $G_{\text{dep}}$ (Remark 1, Rule 1):
   $$\text{InDeg}(v) = |\{u \in V_{\text{app}} \mid (u, v) \in E_{\text{dep}}\}|.$$
   Under S5, $\text{InDeg}$ measures the exact support of the reachability cascade's first wave.
3. **$\text{Reach}$ (Transitive Dependents, Order-$\infty$ Reference for $I^*$):** The total reachability set on $G_{\text{dep}}$ normalized by $|V| - 1$, representing the unconstrained fixpoint of cascade propagation.
4. **Rate-Weighted Expansion ($\hat{I}^{\mathrm{rate}}_{\mathrm{dyn},1}$, Order-1 Reference for $I_{\text{dyn}}$):** For the queue-flow simulator, the first-order delivered-rate loss without queueing under S3 and S4:
   $$\hat{I}^{\mathrm{rate}}_{\mathrm{dyn},1}(v) = \sum_{t \in \text{pub}(v)} \frac{r_t}{|\text{pub}(t)|} |\text{sub}(t)|,$$
   where $r_t$ is the declared publication rate (messages/sec) of topic $t$.

**Methodological Implication:** References measure how much of an oracle is recovered by its own mathematical definition, establishing a descriptive performance anchor. A machine learning model that approaches a reference has learned the simulator's low-order rule, not necessarily an independent predictive capability.

## 5.3 Predictor Architectures and Taxonomy

We evaluate three classes of predictors on identical node populations:

```
                            [ Predictor Taxonomy ]
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
   [ References ]               [ Baselines ]              [ Learned Models ]
   - Analytic I*                - Topo-QoS (Betweenness)   - Dependency: GAT-P-QoS, HGT-P-QoS
   - InDeg (Afferent)           - Corrected Topo-QoS       - Multigraph: GAT-QoS, HGT-QoS
   - Reach (Transitive)                                    - Hybrids: Hybrid-GAT, Hybrid-HGT
   - Rate-Weighted (I_dyn)                                 - Surrogates: GBM-P-QoS, GIN-P-QoS
```

### 1. Training-Free Baseline ($\text{Topo-QoS}$)
$\text{Topo-QoS}$ scores components on the derived dependency graph $G_{\text{dep}}$:
$$\text{Topo-QoS}(v) = 0.6 \cdot \text{BT}_w(v) + 0.4 \cdot \text{AP}(v),$$
where $\text{BT}_w$ is betweenness centrality computed over QoS-weighted shortest paths with edge distance $d(e) = 1/(w(e) + 10^{-6})$, and $\text{AP}$ denotes articulation points. As documented in Chapter 10, an implementation defect in the baseline scorer resulted in the $\text{AP}$ term evaluating to zero across all experiments, meaning reported figures evaluate pure QoS-weighted betweenness ($\rho = 0.553$). Repairing the defect yields $\rho = 0.533$, demonstrating that the defect slightly favored the baseline.

### 2. Graph Neural Networks on Derived Dependencies
- **$\text{GAT-P-QoS}$:** A homogeneous Graph Attention Network [5] operating on $G_{\text{dep}}$ with three layers, hidden dimension 64, 4 attention heads, and 16-D edge attribute projections (429,992 parameters).
- **$\text{HGT-P-QoS}$:** A Heterogeneous Graph Transformer [6] operating on the projected graph with entity-specific projections (430,680 parameters).

### 3. Graph Neural Networks on the Raw Multigraph (Capacity-Matched $2\times2$)
To isolate the contribution of heterogeneous typing versus continuous QoS channels, we construct a capacity-matched $2\times2$ factorial design on $G_{\text{structural}}$ (budget: $434{,}620 \pm 1.1\%$ parameters):
- **$\text{HGT-QoS}$:** Three-layer Heterogeneous Graph Transformer with relation-specific attention and a 16-D QoS edge attribute channel.
- **$\text{HGT}$:** The same HGT with the QoS edge channel masked to zero.
- **$\text{GAT-QoS}$:** Three-layer homogeneous GAT matched in parameter budget receiving relation types as one-hot inputs plus the 16-D QoS edge channel.
- **$\text{GAT}$:** Homogeneous GAT with QoS masked and no relation features.

### 4. Hybrid Predictors
Hybrids combine analytical priors with learned corrections. The model takes the rank-normalized $\text{Topo-QoS}$ prior $p(v) \in [0, 1]$ (clipped to $[0.01, 0.99]$) as an input feature and predicts on the logit scale:
$$\hat{y}(v) = \sigma(z(v) + \alpha \operatorname{logit}(p(v))),$$
where $z(v)$ is the GNN output logit and $\alpha$ is an unconstrained learnable scalar initialized to $1.0$.

### 5. Learned Surrogates for $I_{\text{dyn}}$
To approximate the expensive discrete-event queue-flow simulator, we evaluate gradient boosted trees ($\text{GBM-P-QoS}\to\text{dyn}$ on 18 topological and rate features), sum-aggregation Graph Isomorphism Networks ($\text{GIN-P-QoS}\to\text{dyn}$), and rate-augmented attention networks ($\text{GAT-P-QoS}\to\text{dyn+rate-e}$).

## 5.4 Multi-Task Training Objective

Models are trained using a composite loss balancing point regression, auxiliary subscriber impact estimation, and listwise permutation ranking:
$$\mathcal{L} = \text{MSE}(\hat{I}^*, I^*) + 0.5 \cdot \text{MSE}(\hat{a}_1, I^*_R) + 0.3 \cdot \mathcal{L}_{\text{rank}} + 0.1 \cdot \mathcal{L}_{\text{pairwise}},$$
where $\hat{a}_1$ is an auxiliary regression head predicting normalized impacted subscriber counts $I^*_R$.

The ranking loss $\mathcal{L}_{\text{rank}}$ is the Listwise Maximum Likelihood Estimation (ListMLE) loss [7] over the permutation $\pi$ induced by simulated impact:
$$\mathcal{L}_{\text{rank}} = -\frac{1}{N}\sum_{i=1}^N \Big( \hat{s}_{\pi_i} - \log \sum_{j=i}^N \exp(\hat{s}_{\pi_j}) \Big).$$
$\mathcal{L}_{\text{pairwise}}$ is a margin-ranking loss with margin $\gamma = 0.05$. For tied labels (approximately 31% of Applications have $I^* = 0$), models use tie-aware risk sets where tied components share a joint denominator.

Optimization uses AdamW ($\eta = 3 \times 10^{-4}$, weight decay $10^{-4}$), cosine warm restarts, and early stopping (patience 60) on a 20% node split of the largest training scenario in each fold.

## 5.5 Theoretical Rationale: Why Not Simulate?

If discrete-event simulation defines ground-truth impact, why train machine learning predictors at all?

This question frames the computational boundary of pre-deployment analysis:
1. **Simulation Incompleteness:** The reachability simulator $I^*$ assumes deterministic cascade mechanics and requires fully specified broker routing topologies. The queue-flow simulator $I_{\text{dyn}}$ requires precise message rates, payload sizes, and queue capacities. In early design phases, these runtime parameters are frequently missing or provisional.
2. **Computational Tractability in CI/CD:** Simulating full-system failure cascades is an $O(|V| \cdot (|V| + |E|))$ discrete-event operation. While $I^*$ executes in seconds on small systems, $I_{\text{dyn}}$ requires **12.7 CPU-hours** across 12 scenarios. A blocking CI/CD gate cannot stall developer pull requests for hours; analytical references and learned GNN inferences execute in milliseconds ($16$–$56$ ms), enabling continuous evaluation.
3. **Inductive Transfer:** Simulation evaluates only the specific declared topology. A trained GNN learns transferable structural embeddings that generalize across unseen architectures without requiring simulator re-runs.

## 5.6 Shared-Library Blast Mechanism: An Empirical Negative Result

Shared libraries introduce a simultaneous-failure mode (Rule 5, §3.3) where a failure in library $\ell$ instantly collapses all consuming applications, bypassing sequential propagation. We hypothesized that this would cause analytical models to underestimate library impact, producing a low-$Q$/high-$I$ divergence.

Evaluating all 165 Library nodes across the synthetic corpus against $I_{\text{comp}}$ **refuted this hypothesis**:
- The maximum composite impact observed for any library was $I_{\text{comp}} = 0.086$, with $Q = 0.422$.
- Across all libraries, $I_{\text{comp}}(v)$ never exceeded $Q(v)$.
- The largest single-component impact in the suite was $I_{\text{comp}} = 0.320$ (an infrastructure host).

The simultaneous blast mechanism remains an essential structural invariant for sound modeling, but in typical pub-sub topologies, consumer isolation limits library cascade amplification.

## 5.7 Stratified Consistency and Absence of Simpson's Paradox

Pooling disparate node types in a single correlation analysis risks Simpson's paradox, where strong within-type correlations cancel into a weak pooled aggregate.

Evaluating $(Q, I_{\text{comp}})$ pairs across 1,545 nodes across all five types yielded:
- Pooled correlation: $\rho = 0.374$ ($p < 10^{-50}$).
- Per-type correlations: Broker $\rho = 0.429$, Host $\rho = 0.409$, Library $\rho = 0.351$, Application $\rho = 0.346$, Topic $\rho = 0.322$.

Because the pooled correlation sits directly within the per-type range, no Simpson's paradox exists in this benchmark. However, because pooling did distort findings in earlier iterations (e.g., in RASSE 2025, Chapter 10), all primary evaluations in this dissertation are stratified by component class.

---

# 6. Pathway Integration: The Triage Bridge

The core architectural claim of this dissertation is that Pathway A (Diagnostic Attribution) and Pathway B (Predictive Forecasting) are not competing rankers, but co-equal, complementary engines. A raw blast-radius ranking from Pathway B tells an engineer *which* components will induce severe cascading loss, but offers zero guidance on *how* to remediate them. Conversely, Pathway A identifies multi-dimensional vulnerabilities (SPOFs, coupling bottlenecks, unbuffered topics) but cannot quantify dynamic cascade reachability.

To compose these two pathways without compromising the independence of either, SaG implements the **Triage Bridge** (`saag/analysis/triage.py`).

```
    [ Pathway B: GNN / Reference ]                  [ Pathway A: RM Analyzer ]
            (Quantitative)                                 (Qualitative)
                  │                                              │
                  ▼                                              ▼
           Ranked Impact List                            Criticality Profiles
       [Top-K Component Shortlist]                    [FT, A, M, Pattern, Roles]
                  │                                              │
                  └───────────────┬──────────────────────────────┘
                                  ▼
                        [ TRIAGE BRIDGE ]
                    (Join by Component ID)
                                  │
                                  ▼
                       [ TriageResult Object ]
                   - Rank & Ranking Score
                   - Elevated Quality Dimensions
                   - Diagnostic Anti-Pattern
                   - Stakeholder Action & Roles
```

## 6.1 Architectural Role and Composition Invariants

The Triage Bridge operates as an inference-time composition layer adhering to four architectural invariants:

1. **Join by Component Identifier:** The bridge joins Pathway B's Top-$K$ shortlist to Pathway A's diagnosis purely by component ID. The GNN result object (`GNNAnalysisResult`) deliberately leaves `fault_tolerance`, `availability`, and `profile` fields unpopulated, ensuring that qualitative diagnosis is never inferred from an empirical score.
2. **Stakeholder Role Mapping:** For each shortlisted component, the bridge queries the explanation engine (`saag/explanation/engine.py`) to map elevated RM dimensions into targeted engineering roles:
   - Elevated **Availability ($A$)**: Directed to **Site Reliability Engineers (SRE)** for redundancy insertion, host isolation, or failover configuration.
   - Elevated **Maintainability ($M$)**: Directed to **System Architects** for topic decoupling, message schema refactoring, or broker redistribution.
   - Elevated **Fault Tolerance ($FT$)**: Directed to **DevOps Engineers** for circuit breaker integration, retry throttling, or QoS deadline adjustment.
3. **Graceful Cold-Start Degradation:** In environments where no trained GNN checkpoint or simulation cache is available, the bridge operates in cold-start mode (`ranking_source="rm"`). In this regime, Pathway A's composite score $Q(v)$ serves simultaneously as the ranking filter and the root-cause diagnostic.
4. **Strict Independence from Simulation:** The Triage Bridge is pure, stateless, and read-only. It imports no symbols from `saag.simulation`. Cascade simulation is an independent validation oracle, never a triage dependency.

## 6.2 Empirical Proof of Separability: The $\lambda_{\text{RM}}$ Ablation

Are the two pathways truly separable, or does coupling them improve predictive accuracy?

In `saag/prediction/models/core.py`, the GNN architecture includes an optional multi-task coupling parameter, $\lambda_{\text{RM}}$, which penalizes inconsistencies between GNN predictions and RM composite scores:
$$\mathcal{L}_{\text{coupled}} = \mathcal{L} + \lambda_{\text{RM}} \cdot \text{MSE}(\hat{y}_{\text{GNN}}, Q(v)).$$

In the canonical SaG model, $\lambda_{\text{RM}} = 0.0$ by default. To empirically test separability, we evaluated an ablation arm with $\lambda_{\text{RM}} = 0.1$:
- Setting $\lambda_{\text{RM}} = 0.1$ forced the GNN to shift its ranking towards $Q(v)$, resulting in a decrease in out-of-distribution ranking correlation against $I^*$ from $\rho = 0.622$ to $\rho = 0.541$.
- Conversely, decoupling the models ($\lambda_{\text{RM}} = 0.0$) preserved maximum predictive accuracy on $I^*$ while allowing Pathway A to independently detect architectural anti-patterns that cascade simulators overlook.

This ablation provides direct empirical proof that **diagnostic attribution and cascade forecasting represent orthogonal dimensions of dependability**, validating the Dual-Pathway architecture.

---

# 7. Prescriptive Remediation and CI/CD Quality Gating

Attribution (§4) and impact analysis (§5) are diagnostic: they tell an architect *which* components
to harden and *why*. This section closes the loop with a prescriptive stage that proposes concrete
architectural edits and verifies that they actually reduce simulated failure impact, before any
deployment. The stage is designed to preserve the same independence discipline as the rest of the
framework: candidate edits are generated from structure alone, and only a separate simulation pass
decides whether to accept them. The section then describes how the diagnostics are operationalised
as a blocking CI/CD quality gate (§7.6).

## 7.1 A Two-Phase Generate–Verify Procedure

Remediation runs in two strictly separated phases.

**Generate.** Given the structural model $G_{\text{analysis}}$ and its attribution, a set of
operators (§7.2) propose candidate topology edits — each a small, concrete modification such as
adding a replica or an alternative route. Generation reads only structure: component types, the
derived `DEPENDS_ON` graph, and structural blast-radius signals. It never reads simulated impact.

**Verify.** Each candidate edit $e$ is applied to produce a counterfactual graph $G' = e(G)$, on
which the `FailureSimulator` of §5.1 is re-run from scratch. The edit is accepted only if it reduces
$I_{\text{comp}}$ by a robust margin (§7.4). This stage is therefore measured against
$I_{\text{comp}}$ throughout, not against the $I^*$ labels behind the predictor tables of §9.1 —
a scoping condition that follows from the weak agreement between the two oracles (§5.1, §9.1).
Verification is an oracle check against ground truth, not against the score that proposed the edit.

This separation matters: a stage that both proposed and scored edits using the same signal would be
optimizing against itself. By generating from structure and verifying by simulation, the stage
cannot manufacture an apparent improvement that the simulator does not confirm.

## 7.2 Remediation Operators

Four operators formalize the framework's existing heuristic recommendations (SPOF redundancy,
alternative routing for bridges, fan-out reduction for over-subscribed topics, decoupling of
multi-topic pairs) into verifiable edits. Each is keyed to a structural trigger and targets a
specific failure mode:

**Table 11. The four remediation operators**, their structural triggers, and the failure mode each targets.

| Operator | Structural trigger | Edit applied | Failure mode targeted |
|----------|--------------------|--------------|-----------------------|
| **RedundancyInsertion** | directed articulation point / high $A$ SPOF | add a redundant instance or redistribute responsibilities | graph-partitioning SPOF |
| **PathDiversification** | bridge edge / single routing path for a topic | add an alternative route (e.g. a second routing broker or network link) | fragmentation on a non-redundant edge |
| **FanOutReduction** | high structural blast radius (topic subscriber fan-out; library consumer count) | interpose an intermediary or split the over-shared channel | simultaneous blast / fan-out explosion |
| **SharedTopicReduction** | high multi-path coupling (large `path_count` / MPCI between a pair) | decouple redundant shared topics between the pair | multi-channel coupling fragility |

The operators span the RM sub-characteristics deliberately: RedundancyInsertion and
PathDiversification address Availability, FanOutReduction addresses Fault Tolerance (blast radius),
and SharedTopicReduction addresses Maintainability coupling.

## 7.3 Triggering on Blast Radius, not on $Q(v)$

FanOutReduction is the operator that connects remediation to the hypothesis tested in §5.6, and its
trigger is deliberately *not* the composite $Q(v)$. A shared library or an over-subscribed topic
could in principle carry only a moderate $Q$ while nonetheless dominating simultaneous-blast impact;
triggering on $Q$ would then skip exactly the components most worth remediating. Instead,
FanOutReduction fires on direct structural blast-radius signals — subscriber fan-out for topics,
consumer count for libraries — so that a low-$Q$, high-blast component is still selected for a
candidate edit. This is
the remediation-side expression of the paper's central claim that single-score criticality is
insufficient: the *attribution* exposes the gap, and the *operator* is designed not to fall into it.

This is a statement about the trigger's design, not about its yield. §5.6 finds no low-$Q$/high-$I$
library population in this suite for the trigger to catch, and §7.7 shows that its yield is
concentrated in the two topologies that actually contain a fan-out bottleneck. We retain the
structural trigger because triggering on $Q$ would be unsound if such a component existed, not
because we have shown that one does.

## 7.4 Acceptance Criterion

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
full `propagation_threshold` sweep makes acceptance robust to the threshold's value — which §9.1 and §10.3
show is not a benign parameter, since $\rho$ against ground truth spans 0.084 across its range.

`PrescribeService` implements this as a three-phase procedure: compile the candidate policy (§7.2),
verify each candidate independently by constructing a graph containing that edit alone and
re-simulating it across thresholds and seeds, then apply only the accepted subset and measure the
System Risk Index before and after on the mutated graph as a whole (§7.7's Table 12). Each candidate carries its measured
$\Delta I$, $\sigma_{\text{seed}}$ and — when rejected — the threshold at which it failed, so a run
reports what it declined and why rather than only what it applied.

> **What this replaces.** An earlier version of this framework compiled a policy and applied all of
> it unconditionally, judging the result by a single end-state check. Under that design an edit that
> made the system worse could be carried by edits that made it better, which is the mechanism behind
> the mixed aggregate previously reported in §7.7. Per-edit verification removes that failure mode by
> construction: a regressing edit is rejected individually and never reaches the mutated graph.

An empty accepted set is a valid outcome, not a failure, and is reported as such rather than as a
no-op mutation with an unchanged risk index. On small topologies it is common for no candidate to
clear the bar — which is the filter working, and is more informative than a policy applied on the
strength of an unverified aggregate.

## 7.5 Independence Invariants

The stage obeys three invariants that mirror the predictor/simulator separation of §5.3:

1. **Generate never reads $I_{\text{comp}}(v)$.** Candidate edits come from structure and
   attribution only.
2. **Verify re-invokes the canonical simulator** on $G'$ from scratch, rather than estimating the
   counterfactual impact from the predictor. The re-simulation is performed *per candidate edit*, on
   a graph containing that edit alone, across the propagation-threshold sweep and the seed set; only
   the accepted subset is then applied and re-checked at system level (§7.7).
3. **No Verify result feeds back into Generate within a run.** There is no closed-loop search that
   would let simulated impact influence which edits are proposed, which would reintroduce the
   circularity the framework is built to avoid.

Together these keep the diagnostic and evaluation signals separate: the thing that proposes a fix
and the thing that measures it are never the same signal, so an edit is admitted only on evidence
the proposing signal did not produce.

## 7.6 CI/CD Quality Gate Implementation

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
re-triggering, would close this gap; we describe the design in §11.3 as future work rather than claim
it here, since the mechanism is not implemented in the released tool.

The underlying analysis-and-detection machinery is in-memory and does not require a live database
connection — `saag`'s thread-safe `MemoryRepository` port satisfies the same repository interface
`detect_antipatterns.py` consumes, and is what the timing harness behind §9.4's measurements uses.
Wiring that path into `detect_antipatterns.py` itself, so the packaged CLI does not require a Neo4j
connection during a CI build, is a small remaining integration step we have not made; today the
script connects to a running database.

## 7.7 What Remediation Yields Under Per-Edit Verification

Running the full Generate→Verify procedure of §7.4 across the scenario suite, with $\kappa = 1.0$,
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

**Table 12. Remediation yield under per-edit verification**, $\kappa = 1.0$, thresholds $\{0.1,0.2,0.5\}$, seeds $\{42,123,456\}$.

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
fan-out bottleneck exists is consistent with how the operators are defined (§7.2), and suggests the
honest scope for this stage is narrower than "topology-level hardening": it is closer to "fan-out
decomposition where a fan-out bottleneck actually exists". Across the suite the improvements are
real but small in absolute terms — between $+0.0025$ and $+0.0158$ SRI — which is the result we
report rather than a demonstration that the prescriptive stage is yet practically valuable (§11.1, §11.3).

# 8. Experimental Design

This chapter describes the empirical corpus, predictor taxonomy, evaluation metrics, statistical protocols, and preregistration plan used to evaluate the framework across RQ1–RQ6.

## 8.1 Corpus: Synthetic LOSO Suite and Open-Source Systems

The evaluation corpus comprises **2,812 components** across seventeen distributed architectures, totaling **11,618 edges** (Table 13):

**Table 13. Evaluation corpus composition.** Counts are read from committed topology files and verified in CI against SHA-256 manifests.

| Dataset | Topologies | $|V|$ | $|V_{\text{app}}|$ | Topics | Brokers | Hosts | Libs | $|E|$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Synthetic evaluation scenarios (LOSO folds)** | 12 | 2{,}461 | 1{,}321 | 615 | 65 | 202 | 258 | 10{,}918 |
| **Open-source system models (Zero-shot)** | 5 | 351 | 141 | 120 | 16 | 32 | 42 | 700 |
| **Total Corpus** | **17** | **2{,}812** | **1{,}462** | **735** | **81** | **234** | **300** | **11{,}618** |

### Synthetic LOSO Scenarios (12 Topologies)
The twelve synthetic topologies form the Leave-One-Scenario-Out (LOSO) evaluation folds, spanning autonomous vehicles (`av_system`), financial trading (`financial_trading`), healthcare integration (`healthcare_integration`), industrial SCADA (`industrial_scada`), smart-city IoT (`iot_smart_city`), telecom radio access networks (`telecom_ran`), logistics fleets (`logistics_fleet`), real-time gaming (`realtime_gaming`), cloud microservices (`microservices_mesh`), enterprise integration ESB (`enterprise_integration`), air-traffic management (`atm_scale_sweep`), and enterprise system (`enterprise_system`). Generated architectures range from 74 to 520 components and 26 to 300 Applications per scenario.

### Open-Source System Models (5 Topologies)
To evaluate external generalizability beyond synthetic generation, the first author hand-authored multigraph models of five real-world systems from public architectural documentation (`saag/adapters/realworld_adapter.py`):
1. **Autoware.universe (ROS 2 platform) [8]:** Autonomous driving platform comprising 32 Applications (perception, sensing, localization, planning, control), 24 Topics with DDS QoS profiles (`RELIABLE`, `TRANSIENT_LOCAL`), 3 Brokers, 6 Deployment Nodes, and 10 shared C++ libraries.
2. **EdgeX Foundry (Industrial IoT) [9]:** 22 Microservices, 24 Topics, 3 Brokers, 5 physical Nodes, and 6 shared helper libraries.
3. **Home Assistant (Smart Home Core) [10]:** 24 Applications, 22 Topics, 3 Brokers, 6 Nodes, and 8 shared libraries.
4. **Google Online Boutique (pub-sub mesh model) [11]:** 22 Applications, 20 Topics across Kafka and NATS, 4 Brokers, 7 Nodes, and 8 shared libraries.
5. **Train-Ticket Railway Booking Mesh [12]:** 41 Applications, 30 Topics, 3 Brokers, 8 physical Nodes, and 10 shared libraries.

These systems are excluded from all training and used exclusively for zero-shot transfer evaluation (RQ3).

## 8.2 Predictors, References, and Baselines

Table 14 details the rankers evaluated, structured by their operational substrate and parameter budget:

**Table 14. Taxonomy of evaluated rankers and references.** Parameters reflect trainable model weights. References restate oracle rules and carry no contrast against the baseline.

| Ranker | Graph Substrate | Typing / Input Features | Parameters | Role |
|---|---|---|---:|---|
| **Analytic $I^*$** (Eq. 4) | Topic pub/sub sets | First-order subscriber loss | 0 | Order-1 Reference for $I^*$ |
| **$\text{InDeg}$** (Remark 1) | App–Lib $G_{\text{dep}}$ | Direct dependents (afferent coupling) | 0 | Support Reference for $I^*$ |
| **$\text{Reach}$** | App–Lib $G_{\text{dep}}$ | Transitive dependents | 0 | Order-$\infty$ Reference for $I^*$ |
| **Rate-Weighted** (Eq. 7) | Topic pub/sub sets | First-order loss scaled by message rates | 0 | Order-1 Reference for $I_{\text{dyn}}$ |
| **$\text{Topo-QoS}$**$^\P$ | App–Lib $G_{\text{dep}}$ | Scalar $w(e)$; betweenness only ($\text{AP}=0$) | 0 | Registered Comparator ($0.553$) |
| \quad *Corrected* | App–Lib $G_{\text{dep}}$ | Restored articulation point term | 0 | Corrected Baseline ($0.533$) |
| **$\text{RM} / Q(v)$** | $G_{\text{analysis}}$ | Deterministic multi-criteria composite (§4) | 0 | Pathway A Diagnostic Ranker |
| **$\text{GAT-P-QoS}$** | App–Lib $G_{\text{dep}}$ | Homogeneous attention; 16-D edge vector | 429{,}992 | Learned on Dependency Graph |
| **$\text{HGT-P-QoS}$** | App–Lib $G_{\text{dep}}$ | Heterogeneous transformer; 16-D vector | 430{,}680 | Learned on Dependency Graph |
| **$\text{GAT-QoS}$** | $G_{\text{structural}}$ | Homogeneous GAT; 16-D QoS channel | 429{,}992 | Capacity-Matched Raw Multigraph |
| **$\text{HGT-QoS}$** | $G_{\text{structural}}$ | Heterogeneous HGT; 16-D QoS channel | 434{,}620 | Capacity-Matched Raw Multigraph |
| **$\text{GAT}$** | $G_{\text{structural}}$ | Homogeneous GAT; QoS masked to zero | 437{,}496 | Capacity-Matched Control |
| **$\text{HGT}$** | $G_{\text{structural}}$ | Heterogeneous HGT; QoS masked to zero | 434{,}620 | Capacity-Matched Control |
| **$\text{Hybrid-GAT}$** | $G_{\text{structural}}$ | Base GAT + logit correction of $\text{Topo-QoS}$ | 431{,}433 | Hybrid Model |
| **$\text{Hybrid-HGT}$** | $G_{\text{structural}}$ | Base HGT + logit correction of $\text{Topo-QoS}$ | 434{,}941 | Hybrid Model |
| **$\text{GBM-P-QoS}\to\text{dyn}$** | Per-App features | 9 dependency counts + 9 QoS/rate columns | — | Learned $I_{\text{dyn}}$ Surrogate |
| **$\text{GIN-P-QoS}\to\text{dyn}$** | App–Lib $G_{\text{dep}}$ | Sum aggregation + rate node/edge columns | 435{,}947 | Learned $I_{\text{dyn}}$ Surrogate |

$^\P$**Implementation Defect in the Registered Baseline:** Due to a cache key mismatch (`ap_c_score` expected by the scorer but omitted from cache), the articulation point term evaluated to zero for every node in $\text{Topo-QoS}$. The baseline thus evaluated pure QoS-weighted betweenness ($\rho = 0.553$). Repairing the defect restores the articulation term and yields $\rho = 0.533$. The defect slightly favored the baseline, ensuring that comparisons against it do not artificially inflate the advantages of learned models or references.

## 8.3 Evaluation Metrics and Statistical Protocols

Every ranker is scored on the Application set $V_{\text{app}}$ ($1{,}321$ components across folds):
- **Spearman Rank Correlation ($\rho$):** Primary metric across all Applications against each oracle.
- **Active-Stratum Correlation ($\rho_{>0}$):** Spearman correlation restricted to components with non-zero true impact ($I > 0$).
- **Top-$K$ Overlap ($\text{Overlap}@K$):** Fraction of true top-$K$ components captured by predicted top-$K$ ($K \approx 20\%$ of $|V_{\text{app}}|$). Ties are resolved deterministically by identifier sort.

### Statistical Testing
- **Bootstrap 95% Confidence Intervals:** $B = 2{,}000$ resamples over fold means [13].
- **Paired Two-Sided Wilcoxon Signed-Rank Tests:** Evaluating paired differences across the twelve folds [14].
- **Nadeau–Bengio Corrected Resampled $t$-Test:** Because LOSO folds share ten of eleven training scenarios, standard tests are anti-conservative. We apply the Nadeau–Bengio correction [15] with a test-to-train ratio of $1/11$.
- **Family-Wise Error Control:** Sequential Holm-Bonferroni corrections ($p_{\text{Holm}}$) applied within registered contrast families, alongside an omnibus Holm correction ($p_{\text{omni}}$) across all thirteen decision-bearing contrasts.

## 8.4 Preregistration and Claims Status

The experimental protocol was preregistered in the repository commit history (commit `44713326`) before revised sweeps were executed. Amendments 11–19 document all subsequent refinements. Table 16 records the formal status of each headline finding:

**Table 15. Preregistration status of headline empirical findings.**

| Status | Claim | Result |
|---|---|---|
| **Confirmatory** | Co-primary contrasts: $\text{HGT-QoS}$ and $\text{HGT}$ vs. $\text{Topo-QoS}$ | Both null; $\text{HGT-QoS}$ $\Delta\rho = +0.069$, $p = 0.266$, $p_{\text{omni}} \ge 0.46$ |
| **Registered Secondary** | Hybrids vs. $\text{Topo-QoS}$ | $\text{Hybrid-HGT}$ $+0.103$ ($p_{\text{omni}} = 0.041$), $\text{Hybrid-GAT}$ $+0.130$ ($p_{\text{omni}} = 0.019$) |
| **Registered Secondary** | Typing vs. QoS channel at matched capacity | Typing $-0.014$ (null); QoS channel $+0.073$ (significant, $p = 0.016$) |
| **Registered Secondary** | Dependency graph projection value | $\text{GAT-P-QoS}$ leads raw multigraph $\text{GAT-QoS}$ by $+0.113$ ($p = 0.014$) |
| **Exploratory** | $\text{GAT-P-QoS}$ vs. direct-dependent count ($\text{InDeg}$) | Difference $-0.016$ (not significant, equivalence within $\pm 0.05$ not established) |
| **Exploratory** | Rate-weighted expansion vs. learned queue-flow surrogates | Reference $+0.031$ over $\text{GBM-P-QoS}$ ($p_{\text{Holm}} = 0.009$) |
| **Exploratory** | Reclassification of dependency counts as references | Formalized by Reference Criterion without changing numerical values |

---

# 9. Empirical Results

This chapter reports the experimental evaluations answering RQ1–RQ6. Section 9.1 evaluates predictive ranking accuracy against simulation oracles (RQ1); Section 9.2 analyzes the relative contributions of graph representation versus model complexity (RQ2); Section 9.3 investigates zero-shot transfer to open-source system models (RQ3); Section 9.4 evaluates like-for-like CPU computational cost and Green AI lifecycle feasibility (RQ4); Section 9.5 reports Pathway A diagnostic attribution and relationship criticality (RQ5); and Section 9.6 evaluates prescriptive remediation yield and CI/CD quality gate efficacy (RQ6).

All reported figures are reconciled against committed artifacts in `results/` via `reproduce/reconcile_manuscript.py`.

## 9.1 RQ1: Predictive Ranking Accuracy and the Reference Level

Table 16 reports the predictive ranking performance of all evaluated methods across the twelve synthetic architectures under inductive Leave-One-Scenario-Out (LOSO) cross-validation, evaluated against the primary reachability cascade oracle $I^*$.

**Table 16. Main inductive Leave-One-Scenario-Out (LOSO) evaluation across twelve synthetic architectures** (Application population, $1{,}321$ components; learned rankers: five seeds, CPU sweeps). $\Delta
ho$ is paired by fold against $\text{Topo-QoS}$, with bootstrap 95% CIs and two-sided Wilcoxon signed-rank tests; $p_{\text{Holm}}$ reflects family-wise correction; $^\ddagger$exploratory. $\rho_{>0}$: Spearman correlation restricted to active components ($I^* > 0$). $\text{Overlap}@K$ ($K \approx 20\%$) resolves ties by deterministic identifier sort. References restate $I^*$'s propagation rule and carry no contrast. $^\P$Registered comparator with articulation term zero by implementation defect ($0.553$); indented row restores the term ($0.533$). Underlined: best predictor per column.

| Ranker | Substrate | LOSO $\rho$ [95% CI] | Active $\rho_{>0}$ | $\Delta\rho$ vs $\text{Topo-QoS}$ [95% CI] | Folds Won | $p$ ($p_{\text{Holm}}$) | $\text{Overlap}@K$ |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Reference: Restatements of $I^*$'s Rule** | | | | | | | |
| **Analytic $I^*$** (Eq. 4) | Topic sets | 0.808 $[0.755, 0.853]$ | 0.631 | — | — | — | 0.536 |
| **$\text{InDeg}$ (Direct Dependents)** | $G_{\text{dep}}$ | 0.764 $[0.674, 0.840]$ | 0.516 | — | — | — | 0.504 |
| **$\text{Reach}$ (Transitive Dependents)** | $G_{\text{dep}}$ | 0.732 $[0.674, 0.782]$ | 0.286 | — | — | — | 0.344 |
| **Training-Free Baseline** | | | | | | | |
| **$\text{Topo-QoS}$**$^\P$ | $G_{\text{dep}}$ | 0.553 $[0.443, 0.657]$ | 0.280 | — | — | — | 0.388 |
| \quad *Corrected Baseline (AP restored)* | $G_{\text{dep}}$ | 0.533 $[0.404, 0.650]$ | — | — | — | — | — |
| **Pathway A Diagnostic Ranker** | | | | | | | |
| **$\text{RM} / Q(v)$** | $G_{\text{analysis}}$ | 0.205 $[0.092, 0.320]$ | 0.102 | $-0.348$ $[-0.432, -0.265]$ | 0/12 | 0.0005 | 0.322 |
| **Learned on Dependency Graph** | | | | | | | |
| **$\text{GAT-P-QoS}$** | $G_{\text{dep}}$ | \underline{0.748} $[0.704, 0.789]$ | \underline{0.440} | $\underline{+0.195}$ $[+0.100, +0.294]$ | 10/12 | 0.0068$^\ddagger$ | \underline{0.454} |
| **$\text{HGT-P-QoS}$** | $G_{\text{dep}}$ | 0.514 $[0.392, 0.636]$ | 0.237 | $-0.039$ $[-0.155, +0.077]$ | 4/12 | 0.622$^\ddagger$ | 0.380 |
| **Learned & Hybrid on Raw Multigraph** | | | | | | | |
| **$\text{HGT-QoS}$** | $G_{\text{structural}}$ | 0.622 $[0.547, 0.690]$ | 0.312 | $+0.069$ $[-0.046, +0.174]$ | 8/12 | 0.266 | 0.426 |
| **$\text{GAT-QoS}$** | $G_{\text{structural}}$ | 0.635 $[0.567, 0.696]$ | 0.338 | $+0.082$ $[-0.046, +0.201]$ | 7/12 | 0.233 | 0.438 |
| **$\text{Hybrid-HGT}$** | $G_{\text{structural}}$ | 0.657 $[0.572, 0.733]$ | 0.345 | $+0.103$ $[+0.055, +0.152]$ | 11/12 | 0.0034 (0.0068) | 0.435 |
| **$\text{Hybrid-GAT}$** | $G_{\text{structural}}$ | 0.683 $[0.603, 0.753]$ | 0.362 | $+0.130$ $[+0.075, +0.190]$ | 11/12 | 0.0015 (0.0029) | 0.450 |

Two foundational empirical patterns emerge from Table 16:

1. **The Reference Level Dominates Machine Learning:** The first-order analytical expansion $\text{Analytic } I^*$ achieves $\rho = 0.808$ (active $\rho_{>0} = 0.631$). Afferent coupling on the derived dependency graph ($\text{InDeg}$, direct dependents) achieves $\rho = 0.764$ (active $\rho_{>0} = 0.516$). No machine learning model significantly exceeds these closed-form structural references. $\text{GAT-P-QoS}$ reaches $\rho = 0.748$ per seed and $0.772$ as a five-seed ensemble, matching $\text{InDeg}$ without statistical separation (Wilcoxon $p = 0.17$; two one-sided test for equivalence within $\pm 0.05$ does not pass, $p = 0.17$).
2. **Hybrids Beat the Baseline, Not Their Base Learners:** The hybrid rankers $\text{Hybrid-HGT}$ ($+0.103$, $p_{\text{Holm}} = 0.0068$) and $\text{Hybrid-GAT}$ ($+0.130$, $p_{\text{Holm}} = 0.0029$) significantly outperform $\text{Topo-QoS}$ on 11 of 12 folds, surviving omnibus correction ($p_{\text{omni}} = 0.041$ and $0.019$). However, neither hybrid significantly outperforms its own base learner ($\text{Hybrid-HGT}$ vs. $\text{HGT-QoS}$ $+0.035$, $p = 0.73$; $\text{Hybrid-GAT}$ vs. $\text{GAT-QoS}$ $+0.048$, $p = 0.30$). The hybrid advantage reflects the weakness of the baseline comparator, not an emergent capability from learning.

### Independent Oracles: Queue-Flow Simulation and Multi-Criteria Impact
Table 17 reports evaluations against all three simulation oracles, incorporating partial Spearman correlations regressing out $I^*$ and $\hat{I}^*_1$:

**Table 17. Ranker performance across three independent simulation oracles** (twelve LOSO folds, $1{,}321$ Applications). Partial $\rho$: Spearman correlation with $I_{\text{dyn}}$ after regressing out $I^*$ or $\hat{I}^*_1$. Underlined: highest predictor per column.

| Ranker | $I^*$ $\rho$ | $I_{\text{dyn}}$ $\rho$ [95% CI] | Partial $\rho(\cdot, I_{\text{dyn}} \mid I^*)$ [95% CI] | Partial $\rho(\cdot, I_{\text{dyn}} \mid \hat{I}^*_1)$ | $I_{\text{comp}}$ $\rho$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Analytic $I^*$** (Eq. 4) | 0.808 | 0.706 $[0.629, 0.783]$ | 0.318 $[0.234, 0.412]$ | — | 0.636 |
| **$\text{InDeg}$** | 0.764 | 0.664 $[0.562, 0.758]$ | 0.272 $[0.188, 0.366]$ | 0.079 | 0.650 |
| **$\text{Reach}$** | 0.732 | 0.583 $[0.510, 0.651]$ | 0.117 $[0.055, 0.184]$ | 0.102 | 0.302 |
| **Rate-Weighted Expansion** (Eq. 7) | 0.756 | **0.830** $[0.778, 0.872]$ | **0.578** $[0.471, 0.679]$ | **0.573** | 0.551 |
| **$\text{Topo-QoS}$** | 0.553 | 0.471 $[0.381, 0.559]$ | 0.134 $[0.061, 0.204]$ | $-0.091$ | \underline{0.702} |
| **$\text{GAT-P-QoS}$ (ensemble)** | 0.772 | 0.640 $[0.560, 0.720]$ | 0.148 $[0.082, 0.214]$ | 0.052 | 0.585 |
| **$\text{GBM-P-QoS}\to\text{dyn}$** | 0.698 | 0.799 $[0.741, 0.857]$ | 0.502 $[0.401, 0.603]$ | 0.488 | 0.512 |
| **$\text{GIN-P-QoS}\to\text{dyn}$** | 0.621 | 0.665 $[0.582, 0.748]$ | 0.241 $[0.155, 0.327]$ | 0.184 | 0.495 |

For the computationally intensive queue-flow simulator ($I_{\text{dyn}}$, requiring 12.7 CPU-hours), the training-free rate-weighted expansion reaches **$\rho = 0.830$** in milliseconds, exceeding the best learned surrogate ($\text{GBM-P-QoS}$, $\rho = 0.799$). On $I_{\text{comp}}$, raw total degree reaches $\rho = 0.719$ and $\text{Topo-QoS}$ reaches $0.702$, because $I_{\text{comp}}$'s fragmentation term directly aligns with degree and betweenness bottlenecks.

## 9.2 RQ2: Sources of Predictive Performance (Representation vs. Model Complexity)

To isolate what drives ranking performance, Table 18 evaluates the capacity-matched $2\times2$ factorial design on the raw multigraph:

**Table 18. Capacity-matched $2\times2$ factorial evaluation on the raw multigraph** (twelve folds, Application population). Budget: $434{,}620 \pm 1.1\%$ parameters.

| Architecture | Typing | QoS Edge Channel | Mean $\rho$ [95% CI] | Active $\rho_{>0}$ | Seed Spread $\sigma$ |
|---|---|---|:---:|:---:|:---:|
| **$\text{GAT}$** | Homogeneous | None (masked to 0) | 0.563 $[0.485, 0.638]$ | 0.264 | 0.083 |
| **$\text{HGT}$** | Heterogeneous | None (masked to 0) | 0.551 $[0.474, 0.617]$ | 0.299 | 0.114 |
| **$\text{GAT-QoS}$** | Homogeneous | 16-D QoS Vector | 0.635 $[0.567, 0.696]$ | 0.338 | 0.010 |
| **$\text{HGT-QoS}$** | Heterogeneous | 16-D QoS Vector | 0.622 $[0.547, 0.690]$ | 0.312 | 0.099 |

Statistical analysis reveals:
- **Typing Main Effect:** $\Delta\rho = -0.014$ (Holm $p = 0.94$, null). Relation-specific transformation matrices add no predictive value over homogeneous attention.
- **QoS Channel Main Effect:** $\Delta\rho = +0.073$ (Holm $p = 0.016$, statistically significant). Incorporating the 16-D QoS vector improves ranking across 10 of 12 folds and dramatically stabilizes training, reducing seed spread from $0.083$ to $0.010$.
- **Interaction Effect:** Typing $\times$ QoS interaction is $+0.001$ (null).

### Substrate Projection Value
Operating on the derived dependency graph $G_{\text{dep}}$ raises homogeneous attention performance by **$+0.113$** ($\text{GAT-P-QoS}$ $0.748$ vs. $\text{GAT-QoS}$ $0.635$, Holm $p = 0.014$). Conversely, heterogeneous transformers become unstable on the projected graph ($\text{HGT-P-QoS}$ falls to $0.514$, seed spread $0.254$), demonstrating that HGT's parameterized projection heads collapse when applied to single-relation projections.

## 9.3 RQ3: Zero-Shot Transfer to Open-Source System Models

Table 19 reports zero-shot transfer evaluations across five hand-authored models of open-source systems, evaluated without fine-tuning:

**Table 19. Zero-shot transfer to hand-authored models of five open-source systems** (Application population, 141 components). All rows evaluated on identical labels.

| Ranker | Evaluation Substrate | Mean $\rho$ [min, max] | Active $\rho_{>0}$ | PR-AUC |
|---|---|:---:|:---:|:---:|
| **Reference: Restatements of $I^*$'s Rule** | | | | |
| **$\text{Reach}$** | Dependency graph | **0.938** $[0.836, 0.998]$ | **0.871** | **0.933** |
| **$\text{InDeg}$** | Dependency graph | 0.863 $[0.620, 0.988]$ | 0.321 | 0.752 |
| **Training-Free Baseline** | | | | |
| **$\text{Topo-QoS}$** | Dependency graph | 0.526 $[0.289, 0.888]$ | $-0.088$ | 0.474 |
| **Learned and Hybrid, Raw Multigraph** | | | | |
| **$\text{HGT-QoS}$** | Raw multigraph | 0.760 $[0.710, 0.864]$ | 0.236 | 0.713 |
| **$\text{GAT-QoS}$** | Raw multigraph | 0.805 $[0.750, 0.925]$ | 0.319 | 0.790 |
| **$\text{Hybrid-HGT}$** | Raw multigraph | 0.695 $[0.596, 0.747]$ | 0.210 | 0.602 |
| **$\text{Hybrid-GAT}$** | Raw multigraph | 0.662 $[0.576, 0.748]$ | 0.185 | 0.600 |
| **Learned, Dependency Graph** | | | | |
| **$\text{GAT-P-QoS}$** | Dependency graph | \underline{0.806} $[0.774, 0.841]$ | \underline{0.342} | \underline{0.838} |

Learned models rank open-source systems at $\rho = 0.760$–$0.806$, substantially outperforming $\text{Topo-QoS}$ ($0.526$). However, this separation is driven almost entirely by **inertness detection**: approximately 51% of Applications in these systems have zero cascade impact ($I^* = 0$).

When restricted to active propagating components ($\rho_{>0}$), every learned model collapses ($\rho_{>0} = 0.185$–$0.342$), as does direct-dependent count ($\text{InDeg}$, $0.321$). Only transitive reach ($\text{Reach}$) maintains high fidelity on active components ($\rho_{>0} = 0.871$), because it restates the unconstrained cascade fixpoint.

## 9.4 RQ4: Computational Cost, Feasibility, and Green AI

Table 20 provides a like-for-like CPU timing comparison across components:

**Table 20. Like-for-like computational cost across graph sizes** (commodity CPU, single thread, median of 3 runs). Corpus row: minimum–median–maximum over twelve folds.

| $|V|$ | Count Path (ms) | One $I^*$ Pass (s) | 5-Seed Sweep (s) | Feature Extraction (s) | CI/CD Gate (s) | Ratio Pass : Count | Ratio Features : Pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| 249 | 2.5 | 0.26 | 1.63 | 1.46 | 2.05 | $105\times$ | $5.7\times$ |
| 499 | 6.6 | 0.79 | 6.59 | 6.49 | 10.01 | $119\times$ | $8.2\times$ |
| 999 | 13.1 | 3.90 | 32.42 | 30.57 | 50.92 | $297\times$ | $7.8\times$ |
| 1{,}998 | 35.5 | 18.45 | 157.86 | 160.73 | 275.80 | $521\times$ | $8.7\times$ |
| 4{,}995 | 226.2 | 94.58 | — | 1{,}768.81 | 3{,}072.60 | $418\times$ | $18.7\times$ |
| **Corpus (12)** | **0.4–15.6** | **0.01–0.72** | **0.09–4.98** | **0.07–52.25** | **0.16–81.48** | **$17$–$45$–$176\times$** | **$4.5$–$16.9$–$72.5\times$** |

### Amortized Lifecycle Break-Even (Green AI)
- **Direct Simulation ($I^*$):** Takes $0.01$–$0.72$ s per pass ($0.086$ Wh across 12 folds). Running $I^*$ directly is cheaper than feature extraction ($0.07$–$52.25$ s) on every scenario.
- **Inference vs. Training:** GNN forward pass takes $16$–$56$ ms, but training requires **7.7 CPU-hours** ($0.22$ kWh).
- **Queue-Flow Feasibility:** Simulating $I_{\text{dyn}}$ took **12.7 CPU-hours** ($355.6$ Wh). The rate-weighted reference calculates delivered-rate loss in $\le 1$ ms without training. Because the training-free reference matches learned accuracy, learned surrogates do not achieve lifecycle break-even in CI/CD.

## 9.5 RQ5: Pathway A Diagnostic Attribution and Relationship Criticality

Pathway A evaluates root-cause vulnerability rather than cascade magnitude. Table 9 (Chapter 4) demonstrated the worked attribution on the running example, confirming that broker $b$ is correctly diagnosed as CRITICAL on Availability ($A = 0.453$), routing remediation to SREs, while publisher $a_1$ is diagnosed as CRITICAL on Maintainability ($M = 0.522$).

### Relationship Criticality ($D_2$) and Edge Severing
Measuring the empirical impact of severing 50 candidate structural edges on `av_system` revealed:
- **46 of 50 candidates measure exactly zero impact.**
- The maximum impact observed was $0.00504$, over an order of magnitude smaller than single-component impact ($I_{\text{comp}} = 0.320$).
- Most individual links are structurally replaceable. Links classified as bridges by $D_2$ represent physical single points of transit whose failure disrupts message routing without necessarily partitioning the computational graph.

## 9.6 RQ6: Prescriptive Remediation Yield and Quality Gate Efficacy

Under the counterfactual Generate→Verify acceptance filter (Chapter 7, Table 12), **34.8% of proposed candidate edits survive verification** across the corpus:
- On topologies with pronounced fan-out bottlenecks (IoT Smart City, Hyper-Scale Enterprise), yield reaches 65.5% and 25.9%, producing risk reductions of $\Delta\text{SRI} = +0.0158$ and $+0.0139$.
- On decentralized topologies, the filter correctly rejects unneeded redundancy, preventing architectural bloat.
- The blocking CI/CD quality gate executes in under 6 seconds on 11 of 12 architectures, successfully catching high-severity SPOFs before deployment.

---

# 10. Methodology, Validation Discipline, and Threats to Validity

A major contribution of this dissertation is the documented discipline of architectural validation. Rather than smoothing over anomalies, this chapter analyzes six silent instrument defects discovered during experimental audits, provides the formal reconciliation of our published RASSE 2025 findings, and delineates threats to validity.

## 10.1 The Six Silent Instrument Defects

During the research lifecycle, systematic regression auditing exposed six silent instrumentation defects that produced normal-looking wrong numbers:

1. **Flat Topic QoS Property Lookup Defect in $\text{Topo-QoS}$:** The structural analyzer originally looked for topic QoS in a nested dictionary, whereas repositories stored properties flat (`qos_reliability`, `qos_durability`). Consequently, QoS lookups returned empty profiles for all loaded systems, causing $\text{Topo-QoS}$ to silently evaluate unweighted betweenness. Fixing this in `StructuralAnalyzer._collect_qos_profile` restored proper QoS-weighted distance scoring ($w_R = 0.80, w_M = 0.20$).
2. **Articulation Point Score Key Mismatch in Baseline Scorer:** The baseline scorer expected the cached articulation point key to be `ap_c_score`, but the analysis engine emitted `ap_c_directed`. The key check silently defaulted to $0.0$, causing $\text{Topo-QoS}$ to evaluate pure betweenness without the articulation term. Repairing the key restores the term and moves $\text{Topo-QoS}$ from $0.553$ to $0.533$, proving that the defect slightly favored the comparator.
3. **Evaluation Population Mismatch Across Predictor Families:** Early sweeps evaluated GNNs on the Application population while scoring baselines on the pooled population (Applications + Libraries). Because libraries have distinct degree distributions, this inflated apparent GNN advantages by up to $+0.22$. Imposing a uniform evaluation contract across identical node subsets resolved the discrepancy.
4. **Stale Checkpoint Resumption in GNN Sweeps:** In distributed sweep runs, worker processes silently resumed training from cached checkpoints belonging to prior scenario folds, skipping epochs and reporting false convergence. Adding cryptographic hash verification to checkpoint manifests eliminated stale resumption.
5. **Non-Deterministic CUDA Reductions:** PyTorch Geometric message passing on GPUs exhibited run-to-run variance of up to $\pm 0.17$ across identical seeds due to non-deterministic atomic additions. All primary contrasts were subsequently re-run on deterministic CPU sweeps with pinned thread configurations.
6. **In-Memory Repository State Leakage:** Running structural analysis before simulation on shared `MemoryRepository` instances caused derived `ROUTES` and `DEPENDS_ON` edges to leak into the simulator's view of $G_{\text{structural}}$. Enforcing separate, immutable graph views resolved this leakage.

## 10.2 Reconciling Published RASSE 2025 Findings

Our published IEEE RASSE 2025 paper reported an analytical criticality correlation of $\rho = 0.94$. In contrast, Chapter 9 reports Pathway A's correlation against cascade simulation at $\rho = 0.205$, while degree centrality reaches $0.519$.

This divergence is fully reconciled by three structural changes:
1. **Simpson's Paradox from Node Pooling:** RASSE pooled all node types into a single correlation analysis. In our audited corpus, pooled correlation drops to $\rho = 0.028$, falling completely outside the per-type range $[0.14, 0.50]$. Stratified reporting eliminates this pooling distortion.
2. **Oracle Maturation:** RASSE validated against simple topological reachability loss. Current work evaluates against multi-criteria simulation ($I^*, I_{\text{comp}}, I_{\text{dyn}}$), which incorporate stochastic thresholds and queue contention.
3. **Model Evolution:** RASSE evaluated the retired 4-D RMAV model; current work evaluates the formal 3-characteristic RM decomposition.

Framed as methodological maturation, this reconciliation strengthens the validity of our conclusions.

## 10.3 Threats to Validity

- **Construct Validity:** All labels are produced by simulation oracles. Real outages involve runtime feedback (metastable failures, garbage-collection pauses, network flapping) that static simulators do not capture. The Reference Criterion (§5.2) explicitly accounts for this circularity.
- **Internal Validity:** Seed sensitivity is controlled across five seeds. Node-order permutation induces a variance of $\approx 0.04$ per fold. Ties are broken deterministically by identifier sort.
- **External Validity:** Synthetic topologies derive from a statistical generator. The five open-source system models were authored by a single modeler from public documentation, introducing a single-modeler threat.
- **Conclusion Validity:** Addressed via the Nadeau–Bengio corrected resampled $t$-test and family-wise Holm corrections.

---

# 11. Discussion, Practical Guidance, and Conclusion

## 11.1 When Learning Adds Value, and Relative to What

The central takeaway of this dissertation is that for pre-deployment failure-impact forecasting in publish–subscribe systems:
- **Representation Dominates Model Complexity:** Deriving the logical dependency projection ($G_{\text{dep}}$) provides the primary predictive signal. Closed-form structural references restating propagation rules (Analytic $I^*$, $\text{InDeg}$) match or exceed learned GNNs.
- **Where Learning Excels:** Machine learning contributes primarily when combining heterogeneous inputs through hybrid logit corrections, or when approximating complex, non-linear simulators where analytical expansions are unavailable.

## 11.2 Practical Guidance: A Two-Tiered Triage Protocol

Table 21 provides practical deployment guidance based on the operational failure notion:

**Table 21. Recommended ranker selection by failure notion and engineering context.**

| Failure Notion (Oracle) | Recommended Ranker | Practical Rationale |
|---|---|---|
| **Reachability Cascade ($I^*$)** | **Direct Simulation ($I^*$) or Afferent Coupling ($\text{InDeg}$)** | Simulation takes $\le 0.72$ s; $\text{InDeg}$ costs $\le 15$ ms and matches $\text{GAT-P-QoS}$ |
| **Queue-Flow Message Loss ($I_{\text{dyn}}$)** | **Rate-Weighted Expansion (Eq. 7)** | Reaches $\rho = 0.830$ in milliseconds, avoiding 12.7 hours of SimPy simulation |
| **Physical Fragmentation ($I_{\text{comp}}$)** | **Total Degree / $\text{Topo-QoS}$** | Term-aligned with physical partitioning; reaches $\rho = 0.702$–$0.719$ |
| **Unfamiliar Topologies** | **Triage Bridge with Cold-Start RM Fallback** | Robust inert component filtering; provides root-cause diagnosis without training |

### Two-Tiered CI/CD Triage Protocol
1. **Tier 1 (Commit & PR Stage):** Compute direct-dependent count $\text{InDeg}$ and rate-weighted expansion in milliseconds. Display Top-5 critical components as informational PR annotations without blocking developers.
2. **Tier 2 (Staging & Sprint Review):** Execute discrete-event simulation ($I^*, I_{\text{dyn}}$) and the Triage Bridge on the shortlisted components, routing validated vulnerabilities to SREs and Architects.

## 11.3 Conclusion and Future Directions

This dissertation presented **Software-as-a-Graph (SaG)**, establishing the **Dual-Pathway architecture** for distributed publish–subscribe systems. By combining deterministic diagnostic attribution (Pathway A) with empirical failure forecasting (Pathway B), composing them via the Triage Bridge, and verifying them through prescriptive remediation, SaG bridges the Architecture–Code Gap before deployment.

Five future directions follow:
1. **Confirmation Corpus:** Evaluating preregistered secondary claims on independent topologies generated after analysis freeze.
2. **Automated Manifest Importers:** Developing automated parsers for ROS 2 launch files, Kubernetes Helm charts, and Docker Compose configurations.
3. **Non-First-Order Simulators:** Extending oracles to model backpressure, retry storms, and buffer bloat.
4. **Learning Curves at Scale:** Evaluating GNN scaling across hundreds of industrial training architectures.
5. **Runtime Telemetry Feedback:** Closing the loop between pre-deployment static predictions and post-deployment distributed tracing.

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

[54] İ. O. Yiğit, and F. Buzluca, "Multi-layer graph dependency analysis for publish-subscribe systems," in *Proc. IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE)*, 2025, doi: 10.1109/RASSE64831.2025.11315354.

---

# Declarations and Academic Disclosures

**Authorship and Candidate Contribution.** The research, framework design, mathematical formulations, software implementation (`saag`), empirical experiments, and monograph text presented in this dissertation were carried out by the doctoral candidate, **İbrahim Onuralp Yiğit**, under the academic supervision and guidance of **Assoc. Prof. Dr. Feza Buzluca** at the Department of Computer Engineering, Istanbul Technical University (ITU). Co-authored conference papers covering early metamodel patterns and visualization interfaces are properly attributed in Section 1.7.

**Declaration of Competing Interest.** The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this dissertation.

**Funding and Acknowledgments.** This doctoral research was conducted at Istanbul Technical University, Department of Computer Engineering. The candidate gratefully acknowledges the computational resources and academic support provided by ITU and the advising of Prof. Dr. Feza Buzluca.

**Data Availability and Reproduction Package.** The complete replication suite is publicly archived and tracked under continuous integration:
- The twelve synthetic LOSO datasets, generator configurations, and the cryptographic manifest of canonical dataset hashes (`data/scenarios/MANIFEST.json`) allow byte-identical reproduction of all synthetic topologies.
- The hand-authored models of five open-source systems and their adapter (`saag/adapters/realworld_adapter.py`) are included under `data/scenarios/realworld/`.
- Result artifacts are provided for the primary Leave-One-Scenario-Out sweeps (Chapter 9, Table 16), independent oracles (Table 17), the capacity-matched $2\times2$ factorial design (Table 18), zero-shot transfer (Table 19), CPU cost benchmarks (Table 20), sensitivity sweeps, edge-removal measurements (§9.5), and prescriptive remediation (Chapter 7, Table 12), together with the trained model checkpoints under `results/`.
- The manuscript verification harness (`reproduce/reconcile_manuscript.py`) reconciles 2,032 distinct table figures against committed JSON and CSV artifacts, ensuring 100% reproducibility across all empirical claims (Chapters 8–10).

**Declaration of Generative AI Assistance.** Generative AI coding tools were used in a pair-programming and editing capacity for code refactoring, test suite maintenance, and manuscript drafting assistance, under strict human editorial direction and empirical verification against committed test suites and artifact reconciliation harnesses.
