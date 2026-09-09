# Software-as-a-Graph: Heterogeneous Graph Learning for Pre-Deployment Dependability Analysis of Complex Distributed Systems

**Authors.** Ibrahim Onuralp Yigit, Feza Buzluca

**Affiliation.** Department of Computer Engineering, Istanbul Technical University, 34469 Maslak, Istanbul, Turkey

**Corresponding author.** Ibrahim Onuralp Yigit — yigiti@itu.edu.tr

> **Review model.** JSS uses a **single-anonymised** review process (confirmed against the
> Elsevier Guide for Authors, September 2026), so the manuscript body names its authors and
> `latex/title_page.tex` is uploaded as a separate Editorial Manager file. An earlier version
> of `outline.md` claimed double-anonymised review; that was wrong and has been corrected.

---

# Abstract

Assessing distributed system dependability before deployment is hindered by absent runtime telemetry and static code analysis's blindness to asynchronous communication topology. We present Software-as-a-Graph (SaG), a static system analysis framework that constructs typed multigraphs over five entity types from Architecture-as-Code manifests. SaG pairs two parameter-independent pathways: a relation-specific Heterogeneous Graph Transformer (HGT) with Quality-of-Service (QoS) edge encodings forecasting cascade blast radii, and an interpretable ISO/IEC 25010 layer attributing fragility to Availability or Fault Tolerance to guide repairs.

Across twelve synthetic scenarios and five open-source systems labelled by simulation: (1) relation typing gains $\Delta\rho = +0.114$ over homogeneous learning under distribution shift, winning 11 of 12 folds ($p = 0.0122$), while in-distribution both are indistinguishable ($-0.023$)---typing acts as an inductive bias for unseen architectures rather than added capacity; (2) QoS edge encoding contributes independently ($+0.054$, $p = 0.0093$); (3) against training-free QoS-weighted centrality, the model is \emph{not} demonstrably superior---its nominal $+0.127$ margin ($p = 0.077$) rests on a single fold, falling to $+0.078$ without it; (4) zero-shot transfer reaches $\rho = 0.680$ overall, but drops to $+0.160$ on active failure-propagating components, inverting on two microservice call-tree systems; (5) the neural forward pass is the cheapest pipeline stage by three orders of magnitude, leaving deterministic structural analysis as the dominant cost. Our contribution establishes this empirical boundary on a fully reproducible corpus: typed QoS-aware learning reproducibly improves upon untyped learning under distribution shift, yet remains competitive with closed-form centrality.

**Keywords:** Heterogeneous graph neural networks; Distributed systems dependability; Publish–subscribe architecture; Cascading failures; Static system analysis; Explainable AI.

---

# 1. Introduction

## 1.1 Motivation

Modern large-scale distributed software systems increasingly rely on asynchronous, event-driven, and publish–subscribe (pub-sub) architectures. Across diverse domains—from autonomous driving (ROS 2 [1]) and enterprise event streams (Apache Kafka [2]) to cyber-physical backbones (DDS [3]), IoT fleets (MQTT [4]), cloud-native microservices [5, 6], and distributed AI/LLM serving clusters—pub-sub decouples producers and consumers in space, time, and synchronization [7]. Components interact indirectly through intermediate message topics and brokers without maintaining direct static references. Furthermore, modern middleware specifications allow engineers to configure deployment-time Quality-of-Service (QoS) policies—such as reliability guarantees, durability, message priorities, and delivery deadlines—to govern how traffic behaves under peak load and network stress.

While this architectural decoupling confers elastic scalability and operational flexibility, it creates a formidable **visibility barrier** for system performance, reliability, and computational sustainability:

-   **Indirect Failure and Degradation Pathways:** In traditional synchronous architectures (e.g., RESTful HTTP or gRPC), component interactions follow explicit caller–callee invocation paths. In asynchronous pub-sub and event meshes, publishers and subscribers share no direct references. Cascading failures, queue head-of-line blocking, and backpressure propagate across hidden logical paths spanning brokers, shared topics, colocated execution nodes, and shared libraries.

-   **Distinct Degradation Mechanisms:** Disturbances in complex distributed systems do not propagate in a uniform manner. They manifest either as *sequential cascades* (e.g., a slow subscriber causing broker queue saturation and upstream backpressure) or as *simultaneous blast radii* (e.g., a shared runtime library crash, memory exhaustion, or host machine outage instantly disabling multiple colocated services). Conventional architectural diagrams and static call graphs fail to represent these multi-layer dependencies.

Addressing these architectural vulnerabilities is most effective and cost-efficient **prior to deployment**, during design and Continuous Integration / Continuous Delivery (CI/CD), adhering to established foundational principles of dependable computing [8]. However, at design and build time, **no runtime telemetry, distributed tracing, or operational logs exist**. Consequently, software architects, performance engineers, and Site Reliability Engineers (SREs) face two fundamental questions without operational data:

1.  *Which components, message topics, and communication links are systemically critical to system dependability and performance?*

2.  *Why are they critical, and what specific architectural repair (such as replicating a message broker, decoupling an over-subscribed topic, or sandboxing a shared library) will most effectively eliminate that risk?*

The same questions bear on computational sustainability, though we are careful about the form of the claim. Analysing an architecture from a manifest requires no provisioned cluster, no running services and no fault-injection harness — the resource it eliminates is infrastructure rather than CPU time. What it does not do is compute less: the deterministic structural analysis this framework depends on is expensive, and on our own corpus it costs substantially more than the discrete-event simulation it produces labels from (§§7.5 and 8.2). We measure wall-clock latency rather than energy counters, and we make no claim that the pipeline is computationally cheap in absolute terms. The narrow and supported statement is that the *learned* component is negligible within it — a $56\,\text{ms}$ forward pass against minutes of feature extraction — so applying graph neural networks to pre-deployment analysis does not itself introduce a meaningful cost.

## 1.2 Problem Statement: The Architecture–Code Gap and the Black-Box AI Challenge

We formulate pre-deployment dependability and performance analysis around two distinct, complementary tasks:

1.  **Failure-Impact Forecasting (Predictive Pathway) — the primary task.** We forecast dynamic cascading failure blast radii and identify critical components using a data-driven, relation-specific model over learned topological representations. Closed-form topological metrics capture broad connectivity cheaply; whether they also resolve multi-hop, relation-dependent cascade spread across heterogeneous channels is an empirical question, which we test directly against such a baseline (§7.1). Our predictive pathway is trained and evaluated against independent simulation ground truth as a ranking and critical-set identification model.

2.  **Explainable Criticality Attribution (Explanation Layer) — what a rank alone cannot say.** A ranked shortlist indicates *where* risk lies, but not *how to fix it*. We therefore pair the predictor with an interpretable structural quality profile grounded in ISO/IEC 25010 [9] and ISO/IEC 25019 [10]. This layer diagnoses the *qualitative root cause* of vulnerability—distinguishing, for instance, an unreplicated single point of failure from a high-coupling maintainability bottleneck—to guide concrete repairs. It serves strictly as an attribution model, not a ranking model.

This separation is architectural rather than merely presentational: both pathways operate on the same graph but share no parameters, and neither is trained on the other’s output. The coupling term that could connect them is disabled by default and reported only as an ablation (§4.2). Maintaining this independence allows SaG to identify components that are structurally central yet operationally low-impact—a nuanced diagnosis unattainable by either pathway alone.

Existing software engineering approaches fail to bridge what we define as the **Architecture–Code Gap**: *a distributed system can have pristine, bug-free source code within each individual service, yet remain fragile to catastrophic global outages caused by hidden architectural single points of failure (SPOFs) or mismatched middleware Quality-of-Service (QoS) contracts.* Classical architecture evaluation such as ATAM [11, 12], and the literature on architectural technical debt [13] and bad smells [14], identify architectural risks but rely on manual stakeholder elicitation rather than quantitative structural analysis. The automated paradigms each leave a different part of the gap unaddressed: static code analysis [15, 16, 17, 18] cannot see message queues or cross-host propagation; chaos engineering [19] needs a provisioned cluster and arrives after the architecture is fixed; and homogeneous centrality [20, 21, 22, 23] flattens the system into an untyped graph in which a topic, a library and a host are indistinguishable. §2 develops each in turn.

Furthermore, while machine learning has demonstrated remarkable success across software engineering, contemporary AI approaches applied to system dependability often function as uninterpretable black boxes. Deep neural models frequently output scalar risk scores or latent embeddings without providing transparent, actionable rationales for their predictions. In mission-critical software engineering, an opaque risk score is inadequate: developers and SREs cannot refactor code or reconfigure infrastructure without understanding *why* a component is vulnerable and *which* architectural mechanism is compromised.

## 1.3 The Software-as-a-Graph (SaG) Approach

To bridge the Architecture–Code Gap while overcoming the black-box AI challenge, this work introduces **Software-as-a-Graph (SaG)**, an AI-driven pre-deployment **Static System Analysis (SSA)** framework. SaG ingests Architecture-as-Code manifests and executes a four-stage pipeline:

1.  **Typed Multigraph Formulation:** SaG models the distributed architecture as a typed, directed multigraph over five core entity types: Applications, Brokers, Topics, Execution Nodes, and Shared Libraries (§3.1).

2.  **QoS-Aware Logical Dependency Projection:** Using six formal projection rules, SaG derives a semantic `DEPENDS_ON` dependency layer that captures both sequential cascades (via topics and brokers) and simultaneous blast radii (via shared libraries and node colocation), weighted by declared QoS contracts (§3.2).

3.  **Heterogeneous Graph Learning for Failure Forecasting (Predictive Pathway):** SaG trains a **Heterogeneous Graph Transformer (HGT)** whose relation-specific attention lets a `USES` edge into a shared library propagate differently from a `PUBLISHES_TO` edge into a topic. It forecasts cascading blast radii, ranks critical components, and outputs per-relationship criticality alongside multi-task quality outputs (§4).

4.  **Explainable Quality Attribution (Explanation Layer):** To explain *why* a flagged component is critical, SaG combines code-level SCA metrics with topological properties into a deterministic **Reliability–Maintainability (RM)** attribution model (§5). Reliability decomposes into **Fault Tolerance** (error propagation depth) and **Availability** (single-point-of-failure exposure), pointing to distinct repairs. Because it is a linear, propagation-free aggregate by design, it explains *why* a component is vulnerable rather than how far a cascade travels; its standalone rank correlation is correspondingly modest (§7.1).

To ensure methodological rigor, SaG enforces a strict **input–label independence guarantee**: learned models and attribution baselines operate exclusively on the analytical graph $G_{\text{analysis}}$, while ground-truth failure impacts are generated by independent discrete-event simulators operating on the raw structural topology $G_{\text{structural}}$ (§4.4).

Figure 1 shows how the two pathways relate. The predictive pathway is the primary one and the only pathway validated against the simulation oracle — the oracle scores rankings, which a quality profile is not — and that oracle is strictly an offline training-and-validation component, never a dependency of online inference. The explanation layer then characterises what the predictor flagged and the remediation that implies. The single link between them is triage rather than data flow: the architect applies the explanation to whatever the predictor ranked. The remediation guidance closes a loop of its own, in which each candidate edit is re-simulated on its own mutated copy of $G_{\text{structural}}$ and kept only if it beats the simulator’s seed-to-seed noise.

```
+-----------------------------------------------------------------------------------+
|                            Software-as-a-Graph (SaG)                              |
+-----------------------------------------------------------------------------------+
|  Architecture Descriptor (Apps, Topics, Brokers, Hosts, Libraries, QoS Policies)   |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
                  +----------------------------------------------+ ------+
                  |     Raw Structural Graph (G_structural)      |       | (Independent
                  +----------------------+-----------------------+       |  simulation,
                                         |                               |   §4.4)
                     [Typed projection & QoS weighting, §3.2]            |
                                         v                               |
                  +----------------------------------------------+       |
                  |       Analysis Multigraph (G_analysis)       |       |
                  |   (Derived DEPENDS_ON edges + typed node     |       |
                  |    features, §3.4)                           |       |
                  +----------------------+-----------------------+       |
                                         |                               |
            +----------------------------+----------------------------+  |
            |                                                         |  |
            v  PREDICTIVE PATHWAY (§4)            EXPLANATION LAYER (§5)  |
+-------------------------------------+   +-------------------------------------+
|  Heterogeneous Graph Transformer    |   |   Explainable Quality Attribution   |
|  - Relation-specific attention      |   |  - Fault Tolerance (cascade depth)  |
|  - 16-D QoS edge embedding          |   |  - Availability (SPOF/articulation) |
|  - Multi-task risk & ranking heads  |   |  - Maintainability (coupling + SCA) |
+------------------+------------------+   +------------------+------------------+
                   |                                         |                 |
                   v                                         v                 |
+-------------------------------------+   +-------------------------------------+
|  Top-K Critical Component Set       |-->|  Root-Cause Diagnostic Profile      |
|  - Blast radius C-hat(v) (§4.2)     |   |  - SPOF exposure (high A)           |
|  - Out-of-distribution ranking      |   |  - Cascade hub (high FT)            |
+------------------+------------------+   +------------------+------------------+
                   |  (Triage: A explains what B flagged)     |                 |
                   v                                         v                 |
+-------------------------------------+   +-------------------------------------+
| Ground-Truth Simulation Oracle I*(v)|<-+|  Remediation Guidance (§5, §8.1)    |
|  (FaultInjector on G_structural)    |   |  - Replication (DevOps/SRE)         |
|  [offline: training & validation]   |   |  - Circuit breakers (architect)     |
+-------------------------------------+   |  - Refactoring (developers)         |
        [scores B's ranking only]         +-------------------------------------+
```

![Figure 1](latex/figures/Figure_1.png)

> **Figure numbering.** Figures 1–4 are numbered as in the LaTeX submission sources: Figure 1 pipeline (`Figure_1`), Figure 2 running example (`Figure_2`), Figure 3 results at a glance (`Figure_5`), Figure 4 HGT attention (`Figure_3`). The ASCII schematics and Figure M1 are specific to this document. Supplementary Sections S1–S7 live in `latex/supplementary.tex` and are not reproduced here.

*Figure 1. End-to-end architecture of the SaG framework. A shared front end (manifest ingestion → typed multigraph → QoS-weighted DEPENDS_ON projection → typed node features) feeds two pathways that share no parameters: the predictive pathway (§4), which emits a ranked critical set and per-relationship criticality, and the explanation layer (§5), which emits a standards-grounded quality profile. The simulation oracle scores only the former and runs on Gstructural alone.*

#### Rationale for Graph Learning vs. Direct Simulation

Since discrete-event simulation $I^*(v)$ defines ground-truth criticality here, it is fair to ask why train a graph model at all rather than run simulation sweeps or closed-form heuristics. Three reasons motivate the design, and §7.1 tests them adversarially. A trained model scores entity types no simulator sweep was run for, since message passing generalizes across labelled and unlabelled entities alike. Cascade simulation is stochastic and seed-sensitive (label standard deviation reaches $0.416$), whereas a trained model learns a smooth, threshold-marginalized surrogate that re-scores an already-analysed architecture cheaply. And dynamic simulators need runnable containers or communication harnesses, whereas graph learning scores Architecture-as-Code manifests before any runtime infrastructure exists. A fourth motivation — that neither a simulator nor an unaugmented GNN returns a root cause in standardized quality terms — motivates the explanation layer rather than the predictor, and is taken up in §5. Whether these motivations are borne out empirically is a separate question, and §7.1 answers it only partly in the framework’s favour.

## 1.4 Research Questions

This empirical study investigates five research questions:

> **RQ1 (Predictive Efficacy):** *How accurately does heterogeneous graph learning predict cascading failure impact and identify the critical component set, compared with traditional, non-learning network metrics?*
>
> **RQ2 (Value of Architectural Typing):** *Does modeling distinct entity and dependency types (applications, topics, brokers, hosts, and libraries) yield better failure predictions than homogeneous graph models, and does that advantage hold on architectures the model has never seen?*
>
> **RQ3 (QoS Encoding and Robustness):** *Do middleware Quality-of-Service contracts carry signal a purely structural score discards, do the framework’s simulation oracles agree with one another, and are the reported orderings robust to the free parameters of the scorer and of the ground truth?*
>
> **RQ4 (Real-World Generalization):** *How effectively does the framework transfer zero-shot to authentic, real-world distributed systems across autonomous driving (ROS 2), cloud-native microservices, smart home IoT, and industrial edge computing?*
>
> **RQ5 (Analysis Cost):** *What does pre-deployment analysis cost at CI/CD time, which pipeline stage dominates that footprint, and how does it compare against the discrete-event simulation it is intended to displace?*

## 1.5 Key Contributions

This paper presents four principal contributions:

1.  **Heterogeneous Graph Learning for Pre-Deployment Dependability:** A relation-specific Heterogeneous Graph Transformer that forecasts cascading blast radii from Architecture-as-Code alone, with a 16-dimensional edge feature vector carrying 7 QoS dimensions and multi-task heads for component and relationship criticality (§4). Our central empirical claim concerns *when* architectural typing pays. With matched training sets, substrate, depth and model selection, typed learning leads untyped learning by $\Delta\rho = +0.114$ under inductive distribution shift, winning 11 of 12 folds ($p = 0.0122$), while in-distribution the two are indistinguishable and the best mean belongs to the *untyped* model (§7.2). Relational typing is therefore an inductive bias for unfamiliar topologies rather than added capacity — which is the only regime a pre-deployment gate ever operates in. The QoS edge encoding contributes independently ($+0.054$, 11/12, $p = 0.0093$; §7.3.1). We establish the boundary just as explicitly: against an unparameterized QoS-weighted centrality baseline, out-of-distribution ranking is *not* surpassed, and the nominal margin rests almost entirely on a single fold (§7.1).

2.  **A Formal Typed Architecture Model:** A multigraph representation that derives logical dependencies from physical pub-sub linkages and distinguishes sequential cascade propagation from simultaneous multi-consumer library failures, supplying the typed substrate the predictor consumes (§3).

3.  **A Standards-Grounded Explanation Layer:** An interpretable Reliability–Maintainability model grounded in ISO/IEC 25010/25019 that turns the predictor’s ranked output into an actionable diagnosis, separating single-point-of-failure exposure from error-propagation depth—two distinct failure modes requiring different repairs (§5).

4.  **Empirical Benchmark, Real-World Evaluation, and Cost Characterization:** An evaluation across twelve synthetic topologies (2,461 components) and five open-source reference systems (351 components) under strict graph-view separation, establishing both where typed graph learning helps and the boundary where it does not, with a per-stage cost profile that locates the pipeline’s cost in its deterministic graph-analysis stage rather than in the learned model, and that measures the framework’s own gate against its simulation oracle — finding the gate the more expensive of the two, and withdrawing on that basis the computational-efficiency claim made in the conference version (§§6–7).

#### Relationship to the authors’ prior work

An earlier, shorter version was presented at a peer-reviewed conference [24], covering the preliminary typed multigraph formulation and the deterministic quality-attribution model on the synthetic corpus alone. This manuscript is a substantially extended version meeting the JSS extension policy. New here are: the entire predictive pathway (the HGT, its 16-D QoS edge encoding, the multi-task masked-loss heads, and every learned result in §7); the inductive LOSO protocol and cross-architecture transfer analysis (§7.2); zero-shot evaluation on five open-source reference systems (§7.4); the cost characterization answering RQ5 (§7.5); the four-oracle convergent-validity taxonomy and graph-view separation (§§4.3–4.4); global sensitivity analysis over all ten weight constants (§7.3); and the homogeneous-versus-heterogeneous comparison under substrate parity (§7.2). Material retained from the conference version is limited to preliminary formalisms in §§3 and 5, both restructured and expanded. No companion manuscript from this work is under consideration elsewhere.

## 1.6 Paper Organization

The remainder of this paper is organized as follows: §2 reviews related work on distributed systems dependability, performance engineering, static system analysis, and graph representation learning. §3 formalizes the Software-as-a-Graph architectural model, the dependency projection rules, and the typed node features consumed by both pathways. §4 presents the Heterogeneous Graph Transformer, its multi-task heads, the simulation oracles that supply its labels, and the input–label independence guarantee. §5 introduces the interpretable RM explanation layer. §6 describes the experimental setup, benchmark corpus, and evaluation protocols. §7 presents empirical results for RQ1–RQ5. §8 discusses the practical implications of those results, what the cost measurements do and do not establish for computational sustainability, threats to validity, and the limitations of the study. §9 concludes.

# 2. Related Work

This work builds upon and connects four foundational research areas: (1) dependability, performance, and sustainability in distributed software systems; (2) static code and system analysis; (3) software quality measurement and multi-criteria evaluation; and (4) graph representation learning and explainable AI (XAI).

## 2.1 Dependability, Performance, and Sustainability in Distributed Software Systems

The publish–subscribe (pub-sub) and asynchronous event-driven paradigms decouple communicating entities in space, time, and synchronization, enabling elastic scalability and high throughput [7]. Modern middleware standards—such as ROS 2 [1], Apache Kafka [2], DDS [3], and MQTT [4]—govern these exchanges through fine-grained Quality-of-Service (QoS) policies that regulate message durability, transport reliability, priorities, and delivery deadlines. In cloud-native microservice meshes and distributed AI/LLM serving backbones, asynchronous message passing and queueing topologies form the primary communication substrate, directly shaping tail latencies, throughput bottlenecks, and hardware resource utilization.

Prior dependability and performance research has focused predominantly on **runtime mechanisms**, including dynamic consensus protocols, broker clustering, adaptive backpressure throttling, autoscaling, and automated failover. In parallel, **chaos engineering and runtime verification** [19] inject faults or latency into staging or production clusters to observe degradation and recovery. While runtime fault injection delivers operational validation that no static method can match, it requires a fully provisioned cluster, carries the risk of real service disruption, and consumes cluster-hours per sweep — which places it, alongside model training, among the development-time computations whose energy cost green software engineering has argued should be accounted for rather than assumed away [25, 26, 27]. In practice this precludes its use during architectural design or lightweight commit-level CI/CD.

Our work addresses the complementary **pre-deployment phase**: predicting systemic cascading vulnerabilities and performance degradation directly from Architecture-as-Code descriptors before runtime infrastructure is provisioned. From a green software engineering perspective the relevant property is that no runtime infrastructure need exist for the analysis to run: the input is a manifest, not a deployment. We are deliberately careful about how far that argument reaches. It is a claim about what must be provisioned, not a claim that the analysis consumes less computation than the alternatives — a distinction our own measurements force, since the static gate proves more expensive than the simulation oracle it was intended to displace (§§7.5.1 and 8.2). The further argument that catching cascading failure hubs at design time avoids the restart storms and connection-pool thrashing that waste datacenter capacity in production is plausible and, in this study, unmeasured; we treat it as motivation rather than as a result.

#### Architecture-Based Reliability Prediction

Predicting dependability from an architectural description before deployment is not a new ambition, and SaG should be read against the tradition that pursued it analytically. Cheung’s absorbing-Markov-chain model [28] derives system reliability from component reliabilities and a transfer-of-control graph; Goseva-Popstojanova and Trivedi [29] systematize the state-based, path-based and additive families that followed, and Immonen and Niemelä [30] survey the resulting methods from the architectural perspective. Model-driven descendants such as the Palladio Component Model [31] and layered queueing networks [32] predict performance and reliability from parameterized component models with well-understood solution techniques.

A parallel tradition annotates an architecture description with fault behaviour directly rather than inferring it. The AADL Error Model Annex [33] lets an architect declare component error states, fault propagation paths and composite failure behaviour on the same model used for design, from which fault trees, FMEA tables and Markov reliability models can be generated automatically. That is a strictly richer description of failure than SaG’s, and where a team is willing to author it the resulting analysis answers questions ours cannot. The difference is what must be written down: an error annex is a deliberate modelling artifact requiring per-component failure semantics, whereas SaG derives its dependency structure from deployment manifests that already exist for other reasons. We target the case where no fault model has been authored, not the case where one has.

These methods are complementary to ours rather than superseded by it, and the distinction is one of required inputs rather than of accuracy. They need per-component failure probabilities, transition probabilities or service demands — parameters that are themselves estimated from operational profiles, measurement or expert judgement, and that are unavailable in the setting we target (a manifest at commit time, with no telemetry). SaG asks a narrower question in exchange: not what the system’s reliability *is*, but which components’ failures would propagate furthest through the declared topology. Where those parameters can be obtained, an analytical model answers a stronger question than a ranking does, and we make no claim to displace it.

#### Data-Driven Failure Prediction and Root-Cause Analysis in Microservices

A large recent literature localizes faults in microservice systems from operational data. Seer [34] and Sage [35] predict and debug QoS violations from traces and hardware telemetry; MicroRCA [36] and TraceRCA [37] localize root causes over service-dependency and trace graphs; DeepTraLog [38] and Eadro [39] combine traces, logs and metrics under graph-based deep models. This line is the closest methodological neighbour to our predictive pathway, and it consistently outperforms what a purely static analysis can achieve — because it observes the running system. That is precisely the boundary: every one of these approaches requires a deployed system emitting traces, logs or metrics, and therefore cannot answer a question posed at design or pull-request time. SaG occupies the pre-deployment complement, and accepts a correspondingly weaker evidential basis: simulated rather than observed failures, and topology rather than behaviour.

## 2.2 Static Code Analysis (SCA) vs. Static System Analysis (SSA)

Traditional **Static Code Analysis (SCA)** tools (e.g., SonarQube [15]) inspect source code Abstract Syntax Trees (ASTs) within individual services. They evaluate cyclomatic complexity [16], class cohesion, module coupling (e.g., Lack of Cohesion in Methods [LCOM], Coupling Between Objects [CBO]) [17, 18], and code duplication to flag internal code smells and defect-prone modules [40, 41, 42, 43]. However, SCA cannot observe runtime communication topology: it is blind to inter-service messaging channels, message broker queue saturation, and cross-host failure propagation.

Recovering system-level structure statically is, however, an active area in its own right, and we do not claim the idea as novel. A body of work reconstructs microservice architecture from source and deployment artifacts without running the system: Bushong et al. [44] derive communication diagrams and bounded contexts from static code analysis of a service mesh, and a recent multivocal review compares nine such recovery tools and finds their outputs complementary enough that combining them improves detection [45]. That literature and ours differ in what the recovered graph is *for*: architecture recovery aims to reproduce a faithful description of the system as built, typically for comprehension or drift detection, whereas we take a declared topology as given and ask which of its components a failure would propagate furthest from. Recovery is, in that sense, an upstream complement — it could supply the manifests SaG consumes for a system whose Architecture-as-Code description is incomplete.

To bridge this “Architecture–Code Gap,” **Static System Analysis (SSA)** extends static analysis from single-service source code to the global system architecture. By modeling distributed applications, message topics, brokers, execution nodes, and shared libraries as a connected multigraph, SSA propagates code-level quality metrics across architectural dependencies. This allows engineering teams to detect structural anti-patterns [46, 47] and architectural technical debt [48] early during continuous integration (CI/CD) [49, 50], before defective topologies enter production.

## 2.3 Software Quality Models and Multi-Criteria Evaluation

Software product quality is standardized by the **ISO/IEC 25010:2023** product quality model [9] and the **ISO/IEC 25019:2023** Quality-in-Use model [10]. ISO/IEC 25010:2023 defines three closely intertwined characteristics critical to modern distributed systems:

-   **Reliability:** The degree to which a system performs specified functions under stated conditions, comprising Faultlessness, Availability, Fault Tolerance and Recoverability.

-   **Maintainability:** The degree of effectiveness and efficiency with which software can be modified, comprising Modularity, Reusability, Analysability, Modifiability and Testability.

-   **Performance Efficiency:** Performance relative to resource consumption under stated conditions, comprising Time Behavior (latency, response time), Resource Utilization (CPU, memory, bandwidth), and Capacity.

SaG operationalizes a strict subset of these: Availability and Fault Tolerance under Reliability, and Modularity, Modifiability and Analysability under Maintainability (§5.1). Faultlessness, Recoverability, Reusability and Testability are not derivable from deployment topology alone and are outside the scope of this work.

Software engineering measurement explicitly distinguishes between *internal quality* (measured on static artifacts at rest) and *external quality* (measured on executing software systems) [51, 52]. In distributed architectures, architectural debt (such as over-centralized message topics or unreplicated brokers) degrades internal quality and precipitates severe external performance bottlenecks, queue congestion, and outages.

Aggregating multi-attribute structural metrics into an auditable quality score constitutes a classic Multi-Criteria Decision Making (MCDM) problem. The **Analytic Hierarchy Process (AHP)** [53] delivers a structured pairwise-comparison method with an explicit Consistency Ratio ($CR \le 0.10$) intended to certify that elicited judgements are mutually coherent. That statistic detects *in*consistency; it cannot detect a matrix filled in from an answer already chosen, which is a limitation we take seriously for our own weights and quantify in Supplementary §S4. This study applies AHP to construct an audited, explainable Reliability–Maintainability (RM) quality baseline, in conjunction with learned graph models.

## 2.4 Graph Representation Learning and Explainable AI

Network science provides established centrality metrics to identify critical nodes, such as degree, closeness, betweenness centrality [20, 22], articulation points, and PageRank [21, 23]. Foundational studies on network robustness [54], cascading overloads [55], and interdependent networks [56] model how disruptions propagate across connected systems. These robustness models are a natural source of baselines for a task like ours, and we note explicitly that we did not use them as such: our training-free comparators are centrality-based (§6.2), not percolation-based. A targeted-attack percolation baseline would test a different and arguably more apt hypothesis — that criticality is fragmentation potential rather than centrality — and its absence is a limitation of our baseline set rather than evidence about it.

However, standard network metrics suffer from two major limitations when applied to software architectures:

1.  **Dimensional Collapse:** A single centrality scalar cannot distinguish *why* a component is critical—for instance, whether it is a single point of failure (SPOF), an error-propagating cascade hub, or an over-shared library.

2.  **Semantic Collapse:** Standard metrics treat all nodes and edges identically. They conflate fundamentally different architectural entities, such as an asynchronous message topic, a shared library, and a physical execution host.

To overcome the limits of hand-engineered metrics, recent research has applied machine learning to network vulnerability (e.g., FINDER [57], DrBC [58], and PowerGraph [59]). However, most available models rely on **homogeneous message passing** (GCN [60], GraphSAGE [61], GAT [62]), which averages signals across all connections indiscriminately. Because distributed software architectures are inherently **heterogeneous** (comprising distinct entity types and relationship rules), homogeneous models blur critical architectural boundaries and fail to generalize out-of-distribution.

Heterogeneous Graph Neural Networks (RGCN [63], HAN [64], HGT [65], MAGNN [66]) resolve this by employing relation-specific transformations. We build upon the **Heterogeneous Graph Transformer (HGT)** architecture [65] to preserve typed relational semantics when forecasting cascading failure blast radii and performance degradation.

#### Explainable AI (XAI) vs. The Black-Box Barrier

A critical hurdle in applying modern AI to software engineering is the **black-box barrier**: deep neural models output risk scores or continuous embeddings without explaining underlying structural causality. In production software engineering, uninterpretable risk rankings hinder actionable decision-making: developers and SREs cannot determine whether to replicate a host, configure circuit breakers, or refactor shared libraries.

Existing GNN explanation techniques, such as GNNExplainer [67] and PGExplainer [68], identify influential subgraphs through edge masking or parameterized learning. Although useful, these methods explain the model using internal latent representations rather than standardized software engineering concepts. SaG resolves this limitation through a decoupled dual-pathway design: the predictive HGT pathway reveals typed mutual-attention distributions indicating *which* architectural relations propagated the cascade (§7.3), while the deterministic explanation layer attributes fragility to standardized ISO/IEC quality sub-characteristics (§5), translating raw predictions into actionable, cost-effective remediations.

# 3. The Software-as-a-Graph (SaG) Architectural Model

This section formalizes the Software-as-a-Graph multigraph representation (§3.1), the QoS-aware weighting and logical dependency derivation rules (§3.2), the dual graph views (§3.3), and the typed node feature encodings (§3.4).

## 3.1 Formal Multigraph Definition

A complex distributed software system is formally modeled as a typed, weighted, directed multigraph: $$\mathcal{G} = (V, E, \tau_V, \tau_E, w_V, w_E)$$ where:

-   $V$ is the set of system entities, partitioned into five disjoint entity types: $$V = V_{\text{app}} \cup V_{\text{broker}} \cup V_{\text{topic}} \cup V_{\text{node}} \cup V_{\text{lib}}$$

-   $E$ is the set of directed edges connecting entities.

-   $\tau_V: V \to \mathcal{T}_V$ and $\tau_E: E \to \mathcal{T}_E$ are typing functions assigning node and edge categories.

-   $w_V: V \to [0, 1]$ and $w_E: E \to [0, 1]$ are weighting functions representing entity criticality and connection strength.

Table 1 summarizes the five entity types and six structural edge types formalized in the SaG model, along with their semantics and representative concrete distributed systems implementations.

**Table 1.** Entity and structural edge types in the SaG model.

| **Entity Type ($\mathcal{T}_V$)**     | **Architectural Role**                             | **Concrete System Examples**                   |
|:--------------------------------------|:---------------------------------------------------|:-----------------------------------------------|
| **Application** ($V_{\text{app}}$)    | Autonomous process producing/consuming messages    | ROS 2 node, Kafka microservice, MQTT client    |
| **Broker** ($V_{\text{broker}}$)      | Message routing and queuing intermediary           | RabbitMQ exchange, Mosquitto, EMQX broker      |
| **Topic** ($V_{\text{topic}}$)        | Named logical communication channel                | `/sensor/lidar`, `orders.payment.completed`    |
| **Node** ($V_{\text{node}}$)          | Physical host or virtualized execution environment | Bare-metal server, Kubernetes worker, Cloud VM |
| **Library** ($V_{\text{lib}}$)        | Shared software package or runtime dependency      | `librdkafka`, OpenCV, Protobuf runtime         |
| **Structural Edge ($\mathcal{T}_E$)** | **Direction**                                      | **Semantic Meaning**                           |
| `PUBLISHES_TO`                        | App/Library $\to$ Topic                            | Component publishes messages to topic          |
| `SUBSCRIBES_TO`                       | App/Library $\to$ Topic                            | Component consumes messages from topic         |
| `ROUTES`                              | Broker $\to$ Topic                                 | Broker manages and routes topic traffic        |
| `RUNS_ON`                             | App/Broker $\to$ Node                              | Process is hosted on physical/virtual host     |
| `CONNECTS_TO`                         | Node $\to$ Node                                    | Physical network link between hosts            |
| `USES`                                | App $\to$ Library                                  | Application links to shared library dependency |

Application and Library entities additionally incorporate static code metrics computed via Static Code Analysis (SCA) tools (`cm_` attributes: lines of code, cyclomatic complexity, coupling between objects, LCOM), linking code-level fragility directly to topological analysis.

## 3.2 QoS-Aware Weights and Logical Dependency Derivation

In distributed middleware, communication links vary in coupling strength based on their Quality-of-Service (QoS) contracts. For instance, a `RELIABLE` topic with `TRANSIENT_LOCAL` durability binds communicating services substantially more tightly than a `BEST_EFFORT` telemetry stream.

Each topic $t$ carries an intrinsic criticality weight $w(t) \in [0, 1]$ combining its declared QoS semantics with two runtime-stress modulators: payload size and publication frequency: $$\label{eq:3}
w(t) = \beta \cdot \text{QoS}(t) + \alpha \cdot \text{SizeNorm}(t) + \psi \cdot \text{FreqNorm}(t),
\quad (\beta, \alpha, \psi) = (0.75,\, 0.15,\, 0.10)$$ where the QoS term is an AHP-weighted aggregate of the declared contract: $$\text{QoS}(t) = w_{\text{rel}} \cdot q_{\text{rel}} + w_{\text{dur}} \cdot q_{\text{dur}} + w_{\text{prio}} \cdot q_{\text{prio}},
\quad (w_{\text{rel}}, w_{\text{dur}}, w_{\text{prio}}) = (0.24,\, 0.62,\, 0.14)$$ Here, $q_{\text{rel}}, q_{\text{dur}}, q_{\text{prio}} \in [0, 1]$ represent normalized reliability, durability, and transport-priority scores. Durability dominates because it governs whether data persists across restarts and network partitions. Reliability and transport priority both govern in-flight delivery quality, with reliability receiving higher weight because unconditional delivery guarantees precede message scheduling. The sub-weight vector is the geometric-mean priority vector of an independently stated Saaty pairwise-comparison matrix, printed with its consistency computation in Supplementary §S4; $CR = 0.016$, small but non-zero. The modulators are logarithmically compressed and clamped to $[0, 1]$: $\text{SizeNorm}(t) = \min(1.0, \log_2(1 + \text{bytes})/20)$ (a 1 MiB design envelope, representing the practical DDS sample ceiling before RTPS fragmentation dominates) and $\text{FreqNorm}(t) = \min(1.0, \log_{10}(1 + \text{Hz})/3)$. The final weight $w(t)$ is clamped to $[0.01, 1]$, ensuring that best-effort edges remain visible to graph traversals. Every structural communication edge incident on $t$ (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`) inherits $w_E(e) = w(t)$ along with the topic’s QoS vector.

The outer split $(\beta, \alpha, \psi)$ is a declared convex combination. Sweeping it over the full simplex changes the induced ordering of $w(t)$ by at most $\rho = 0.076$ and downstream rank correlation by at most $0.020$, so it is a documented convention rather than a tuned parameter (Supplementary §S1).

### Logical Dependency Projection (`DEPENDS_ON`)

Structural edges capture explicit deployment connections but omit implicit runtime dependencies. For example, a subscriber depends upon a publisher, yet no direct edge connects them in pub-sub architectures. We therefore derive a single unified semantic relation, `DEPENDS_ON`, directed from *dependent* to *dependency* (“if target fails, source is impacted”), according to the six projection rules detailed in Table 2:

**Table 2.** The six `DEPENDS_ON` logical dependency projection rules.

| **Rule** | **Dependency Category** | **Structural Pattern ($\text{Dependent} \to \text{Dependency}$)**                    | **Derived Weight ($w$)**              |
|:--------:|:------------------------|:-------------------------------------------------------------------------------------|:--------------------------------------|
|  **1**   | `app_to_app`            | Subscriber $\to$ Publisher (via shared Topic, incl. transitive `USES`)               | $1 - \prod_{t \in T}(1 - w(t))$       |
|  **2**   | `app_to_broker`         | Publisher/Subscriber $\to$ Broker routing its topics                                 | $1 - \prod_{t \in T}(1 - w(t))$       |
|  **3**   | `node_to_node`          | Host $\to$ Host (lifted from inter-host app dependencies)                            | Lifted $\max w$                       |
|  **4**   | `node_to_broker`        | Host $\to$ Broker (lifted from hosted app dependencies)                              | Lifted $\max w$                       |
|  **5**   | `app_to_lib`            | Application $\to$ Shared Library it `USES`                                           | $H(w_V(\text{app}), w_V(\text{lib}))$ |
|  **6**   | `broker_to_broker`      | Broker $\leftrightarrow$ Broker (shared physical fault-domain colocation, symmetric) | $w_V(\text{node})$                    |

Rules 1 and 2 aggregate the set of topics $T$ connecting a component pair using a probabilistic union rather than a maximum [69, 70, 71]. This guarantees that additional parallel failure vectors increase coupling monotonically while keeping $w \in (0, 1]$. Rule 5 applies the harmonic mean $H(x, y) = 2xy/(x+y)$ [72] to combine the consuming Application’s and the shared Library’s vertex weights, balancing caller and dependency criticality. Rules 3 and 4 assign the maximum weight among component-level dependencies crossing the host boundary.

### Sequential Cascades vs. Simultaneous Blasts

A foundational principle of the SaG model is distinguishing between two fundamentally different degradation modes:

-   **Sequential Cascade (Rule 1):** When an application publisher fails, downstream subscribers suffer message starvation. The failure propagates hop by hop through message queues and topic buffers.

-   **Simultaneous Blast (Rule 5):** When a shared software library or execution node crashes, all consuming applications and colocated brokers fail *instantaneously* in a single shared-fate event.

Preserving architectural entity types and relation-specific projection rules enables SaG to model both mechanisms, whereas untyped homogeneous graphs collapse them into indistinguishable edges.

Rule 6 is intentionally the only symmetric projection rule. It does not imply that one broker functionally depends on another, but rather that two brokers colocated on the same host share that host’s physical failure domain. This follows the same simultaneous-blast principle as Rule 5, which is why the derived weight equals the shared Node’s weight and the relation is bidirectional. In production middleware deployments, colocated brokers compete for host resources (CPU cores, page cache, file descriptors, and NIC bandwidth); a host outage takes down all colocated instances simultaneously. Operational best practices for Kafka, RabbitMQ, and EMQX recommend distributing brokers across fault domains. Rule 6 does not model directional intra-cluster broker coupling (e.g., partition replication, controller quorum election, federation, or shovel links), which do not require physical colocation; extending the schema to capture these interactions is reserved for future work. Rule 6 applies in four of the eight scenarios forming the detection benchmark subset — the seven core synthetic topologies plus the ATM case study, the subset identified in Table 5 — and contributes only 12 directed edges across them. Because the simulation oracles operate strictly on $G_{\text{structural}}$ (§4.4), Rule 6 has zero influence on ground-truth failure labels.

## 3.3 Dual Graph Views and Architectural Layers

The SaG framework maintains two distinct representations of the system:

1.  **Structural Graph ($G_{\text{structural}}$):** The raw deployment graph containing physical and structural relations (such as `PUBLISHES_TO`, `ROUTES`, `RUNS_ON`, and `USES`). Discrete-event simulators consume this view exclusively to execute unbiased failure injections (§4.3).

2.  **Analysis Graph ($G_{\text{analysis}}$):** The projected graph containing derived `DEPENDS_ON` edges annotated with QoS weights and ingested SCA code metrics. All GNN feature representations, graph embeddings, and analytical metrics are computed on $G_{\text{analysis}}$.

Figure 2 illustrates this duality on a running example, contrasting the raw structural graph against the derived `DEPENDS_ON` projection.

![Figure 2](latex/figures/Figure_2.png)

*Figure 2. Running example: the raw structural graph (left) and the DEPENDS_ON projection derived from it (right). The projection makes implicit runtime dependencies explicit—a subscriber depends on the publishers of its topics even though no structural edge joins them—while the simulators continue to operate on the structural view alone.*

$G_{\text{analysis}}$ is further structured into four analytical layers (Application, Middleware, Infrastructure, and Global System), enabling evaluation of criticality at subsystem levels, consistent with hierarchical frameworks such as MIL-STD-498 [73].

## 3.4 Typed Node Feature Encoding

Both pathways read the same typed node properties from $G_{\text{analysis}}$: the predictive pathway (§4) projects them per entity type before heterogeneous message passing, and the explanation layer (§5) aggregates them into its quality profile. All five entity types share indices 0–17, a common block of topological metrics (in/out degree, betweenness, closeness, reverse PageRank, clustering coefficient, articulation score, bridge load) produced by the deterministic analysis stage whose cost is characterized in §7.5. Type-specific features extend that block:

-   **Application (23 dims):** indices 18–22 add source-code metrics from SCA — lines of code, cyclomatic complexity, Martin’s instability $I_{\text{code}} = C_e/(C_a + C_e)$ [74], Lack of Cohesion in Methods, and the composite Code Quality Penalty (CQP).

-   **Library (25 dims):** the Application block plus two library-specific blast-radius drivers (23–24): the normalized size of the transitive reverse-`USES` closure, and the normalized count of distinct subscribers reachable from topics published within that closure.

-   **Broker (19 dims):** index 18 is normalized queue buffer capacity.

-   **Topic (22 dims):** indices 18–21 are publisher count, subscriber count, log message frequency $\log(1 + \text{freq})$, and ordinal QoS criticality.

-   **Infrastructure Node (20 dims):** indices 18–19 are normalized CPU core allocation and physical memory.

# 4. Graph Learning for Failure-Impact Prediction

Cascading failure impact in distributed software systems is inherently non-linear, multi-hop, and relation-dependent. Outages propagate not merely based on neighbor count, but through architectural relations and dependencies extending multiple hops beyond the initial fault. Whether a closed-form combination of standard centrality metrics can capture these compound dynamics is an empirical question rather than a settled one. The primary predictive pathway of §1.2 therefore employs a learned graph model, and §7.1 evaluates it against exactly such a closed-form baseline — which, on out-of-distribution ranking, it does not significantly surpass.

This section details the Heterogeneous Graph Transformer (HGT) architecture and its typed edge encodings (§4.1), the multi-task prediction heads and dimension-masked loss formulation (§4.2), the ground-truth simulation oracles (§4.3), and the input–label independence guarantee that prevents data leakage (§4.4).

## 4.1 Heterogeneous Graph Transformer Architecture

Because distributed systems comprise heterogeneous entity types (Applications, Libraries, Brokers, Topics, Infrastructure Nodes) and diverse interaction semantics (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`, `DEPENDS_ON`), we employ a three-layer **Heterogeneous Graph Transformer (HGT)** architecture [65], implemented within PyTorch Geometric [75], with hidden dimension $D = 64$ and $H = 4$ attention heads. This architecture ensures that typed relations, rather than simple adjacency, govern failure-impact forecasting.

```
+-----------------------------------------------------------------------------------+
|               Heterogeneous Graph Transformer (HGT) Architecture                  |
+-----------------------------------------------------------------------------------+
|  Node Features (19-25 dims) + 16-dim QoS Edge Encodings injected into destinations|
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
| 1. Type-Specific Input Projection:                                               |
|    h_v^(0) = LayerNorm( GELU( W_tau(v) * x_v ) )                                 |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
| 2. Relational Mutual Attention & Edge Feature Ingestion (L Layers):               |
|    e_uv' = W_edge * e_uv,   h_tilde_v = h_v + e_uv'                               |
|    Attention(u, e, v) = Softmax_u ( ( K(u) * W_att,phi(e) * Q(h_tilde_v)^T ) / d )|
|    Message(u, e, v)   = V(u) * W_msg,phi(e)                                       |
|    h_v^(l) = LayerNorm( h_v^(l-1) + Dropout( Sum_u Attention(u,e,v)*Message(u,e,v) ) )|
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
| 3. Multi-Task Residual Output Heads:                                              |
|    - Reliability Head (R):       y_R(v) = Sigmoid( MLP_R( h_v^(L) ) )             |
|    - Maintainability Head (M):   y_M(v) = Sigmoid( MLP_M( h_v^(L) ) )             |
|    - Global Cascade Impact Head: I_pred(v) = Sigmoid( MLP_C( h_v || y_R || y_M ) )|
|    - Edge Criticality Head:      Q(u,v) = Sigmoid( TypedEdgeEncoder(h_u, h_v, e) )|
+-----------------------------------------------------------------------------------+
```

*Figure M1 (this document only). Layered architecture of the Heterogeneous Graph Transformer predictor. The LaTeX sources carry no counterpart; the equations it summarises are those of §4.1.2 and §4.2.*

### 4.1.1 Continuous-Categorical Edge Feature Encoding (16-D)

To capture continuous QoS constraints and channel semantics, SaG encodes each directed edge $e = (u,v)$ as a 16-dimensional continuous-categorical vector $e_{uv} \in \mathbb{R}^{16}$. Index 0 is the scalar coupling weight $w_E(e) \in (0,1]$ of §3.2; index 1 is the normalized count of simple paths through $e$ in $G_{\text{analysis}}$; indices 2–8 one-hot encode the seven structural and derived relations (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`, `DEPENDS_ON`). Indices 9–15 encode middleware QoS parameters on `PUBLISHES_TO` and `SUBSCRIBES_TO` edges (zeroed elsewhere). Four of these dimensions are active in our evaluation corpus: reliability ($0$ best-effort, $1$ reliable); durability ($0$ volatile, $0.5$ transient-local, $0.6$ transient, $1$ persistent); message priority ($0$, $0.33$, $0.66$, $1$); and a heterogeneity flag set when the edge’s QoS triple departs from its scenario’s modal profile. The remaining three indices (deadline-active flag, $\log_{10}(1 + \text{deadline\_ns}/10^6)$, and $\log_{10}(1 + \text{max\_blocking\_ms})$) are architectural schema provisions designed to accommodate hard real-time DDS and ROS 2 microsecond profiles, and are zero-initialized in the current benchmarks.

An edge projection module maps $e_{uv}$ into the hidden space: $e_{uv}' = W_{\text{edge}} e_{uv}$. Prior to relational attention computation, this projection vector is incorporated directly into the target node representation: $\tilde{h}_v = h_v + e_{uv}'$.

### 4.1.2 Type-Specific Projection and Heterogeneous Message Passing

For each source node $u$ and target node $v$ connected by meta-relation $\tau(e) = (\tau(u), \phi(e), \tau(v))$:

1.  **Type-Specific Projection:** Node feature vectors $x_v$ (of dimension 19–25 depending on entity type $\tau(v)$) are mapped into the shared $D$-dimensional hidden space: $$h_v^{(0)} = \text{LayerNorm}\big(\text{GELU}(W_{\tau(v)} x_v)\big)$$

2.  **Relational Mutual Attention:** Type-parameterized Query ($Q$), Key ($K$), and Value ($V$) projections calculate relation-specific attention. For head $i \in \{1, \dots, H\}$, with the softmax taken over the incoming neighbourhood $\mathcal{N}(v)$: $$\text{Attn}^{\,i}(u, e, v) = \underset{u \in \mathcal{N}(v)}{\text{Softmax}}\left( K^i(u)\, W^i_{\text{att},\phi(e)}\, Q^i(\tilde{h}_v)^\top \cdot \frac{\mu_{\langle \tau(u), \phi(e), \tau(v)\rangle}}{\sqrt{D/H}} \right)$$ where $\mu_{\langle \tau(u), \phi(e), \tau(v)\rangle}$ is the learned per-meta-relation scaling prior of Hu et al. [65], which lets the model weight an entire relation triple up or down independently of the node pair. We retain it: it is the parameter that most directly expresses “this relation type matters more than that one”, and the typing effect of §7.2 is what it exists to capture. The implementation is PyTorch Geometric’s `HGTConv` [75], whose `p_rel` parameter is this term. $$\text{Msg}(u, e, v) = V(u) W_{\text{msg},\phi(e)}$$

3.  **Bidirectional Message Passing:** To capture downstream consumer starvation and upstream backpressure simultaneously, message passing is executed over both forward and transposed relation views ($G_{\text{analysis}}$ and $G_{\text{analysis}}^\top$).

4.  **Residual Aggregation and Layer Normalization:** Target node representations are updated across layers $l \in \{1, \dots, L\}$ via residual connections and layer normalization: $$h_v^{(l)} = \text{LayerNorm}\left( h_v^{(l-1)} + \text{Dropout}\left(\sum_{u \in \mathcal{N}(v)} \text{Attn}(u, e, v) \cdot \text{Msg}(u, e, v)\right)\right)$$

#### Training Protocol and Optimization Hyperparameters

Models are optimized end-to-end using AdamW with initial learning rate $\eta = 3 \times 10^{-4}$, weight decay $10^{-4}$, and dropout probability $p = 0.10$ applied post-attention. Learning rates follow a cosine annealing schedule with warm restarts ($\text{CosineAnnealingWarmRestarts}$, $T_0 = 75$, $T_{\text{mult}} = 2$, $\eta_{\min} = 3 \times 10^{-6}$). Training executes for a maximum of 300 epochs with early stopping governed by a patience of 30 epochs monitored on validation loss over labeled nodes. Inductive subgraphs are processed per scenario using full-graph inductive packing without mini-batch subsampling, with validation masks isolating held-out nodes to prevent information leakage across data splits. Five independent random seeds $\{42, 123, 456, 789, 2024\}$ are evaluated across all runs, redrawing both partition masks and initializations. *Selection protocol:* the architectural hyperparameters ($D = 64$, $H = 4$, dropout, learning rate and schedule) follow values conventional for HGT [65]. The loss coefficients of Equation (9) have no such precedent — the objective is bespoke to this task — and were set by judgement and left untuned; we state this rather than appeal to a convention that does not exist for a five-term multi-task loss. Neither group was tuned against the in-distribution test split or the LOSO folds; no search over them was performed there. The real-world evaluation of §7.4.1 is a separate case and is documented separately: it runs at a different depth and epoch budget from every other learned result in this paper, and §7.4.1 states that configuration and how it was arrived at. This avoids selection leakage, at the cost of leaving open whether either family is reported near its own optimum — a comparison between untuned configurations, which we state rather than treat as a like-for-like optimum comparison.

## 4.2 Multi-Task Prediction Heads and Dimension Masking

From the final node embeddings $h_v^{(L)}$, SaG utilizes specialized multi-task prediction heads:

-   **Reliability Head:** $\hat{R}(v) = \sigma(\text{MLP}_R(h_v)) \in [0, 1]$

-   **Maintainability Head:** $\hat{M}(v) = \sigma(\text{MLP}_M(h_v)) \in [0, 1]$

-   **Composite Failure Impact Head:** $\hat{I}^*(v) = \sigma(\text{MLP}_C(h_v \parallel \hat{R}(v) \parallel \hat{M}(v))) \in [0, 1]$

-   **Relationship Criticality Head:** $\hat{Q}(u,v) = \sigma(\text{TypedEdgeEncoder}_{\phi(e)}(h_u, h_v, e_{uv})) \in [0, 1]$

### 4.2.1 Dimension-Masked Loss Formulation

The combined optimization objective integrates regression accuracy, multi-task dimension learning, ranking fidelity, pairwise ordering, and edge prediction: $$\label{eq:loss}
\mathcal{L} = \mathcal{L}_{\text{composite}} + 0.5 \cdot \mathcal{L}_{\text{dimension}} + 0.3 \cdot \mathcal{L}_{\text{rank}} + 0.1 \cdot \mathcal{L}_{\text{pairwise}} + 0.3 \cdot \mathcal{L}_{\text{edge}} + \lambda_{\text{RM}} \cdot \mathcal{L}_{\text{consistency}}$$ where $I^*(v)$ is the simulated cascade impact defined by the primary oracle (§4.3), $\mathcal{L}_{\text{composite}} = \text{MSE}(\hat{I}^*(v), I^*(v))$, $\mathcal{L}_{\text{rank}}$ is the ListMLE ranking loss [76], $\mathcal{L}_{\text{pairwise}}$ is margin-ranking loss, and $\mathcal{L}_{\text{consistency}} = \text{MSE}\big([\hat{R}(v), \hat{M}(v)]_{v \in \text{unlabeled}}, [R_{\text{RM}}(v), M_{\text{RM}}(v)]_{v \in \text{unlabeled}}\big)$ regresses predicted heads toward the diagnostic pathway’s baseline (§5) on unlabeled nodes. Headline results use $\lambda_{\text{RM}} = 0$, guaranteeing that the predictive and explanatory pathways remain strictly independent.

**Dimension Masking:** Because dynamic cascade simulation ($I^*(v)$ via `FaultInjector`) observes runtime failure reachability rather than source-code maintainability, maintainability ground truth is unobserved during dynamic simulation. A separate change-propagation oracle $I_M(v)$ evaluates static structural change ripple at the Validate stage, but is never used as a training label to avoid circular supervision. We introduce a boolean dimension mask $m = [m_R, m_M] = [1, 0]$: $$\mathcal{L}_{\text{dimension}} = \frac{1}{\sum_{d} m_d} \sum_{d \in \{R, M\}} m_d \cdot \text{MSE}(\hat{d}(v), d^*(v))$$ This mask ensures the unobserved maintainability head is not artificially penalized or driven toward zero during backpropagation.

**What the reliability label is.** The surviving term deserves to be stated plainly, because it is weaker than a multi-task objective normally implies. `FaultInjector` emits a single scalar per component, and the label extractor assigns that same scalar to both the composite and the reliability columns: $R^*(v) = I^*(v)$ identically. $\mathcal{L}_{\text{dimension}}$ under $m = [1,0]$ therefore regresses $\hat{R}$ toward the very target $\mathcal{L}_{\text{composite}}$ regresses $\hat{I}^*$ toward. The two terms are not redundant — they train separate heads, and $\hat{R}$ re-enters the composite head as an input ($\hat{I}^* = \sigma(\text{MLP}_C(h_v \parallel \hat{R} \parallel \hat{M}))$), so the term acts as an auxiliary path to the same supervision rather than as a second source of information. But no dimension of the label is decomposed by this oracle: a reliability score distinct from overall impact would require an oracle that separates fault-tolerance from availability, which $I^*(v)$ does not. We report the objective as implemented rather than describing it as multi-dimensional supervision it does not receive.

### Domain-Reweighted Criticality ($Q_{\text{domain}}$)

To ground predictions in ISO/IEC 25019 Context of Use ($\vec{\omega} = [q_R, q_M]^\top$), the composite score can be evaluated as: $$Q_{\text{domain}}(v) = q_R \cdot \hat{R}(v) + q_M \cdot M_{\text{static}}(v)$$ where $M_{\text{static}}(v)$ is obtained directly from the structural analyzer’s maintainability baseline, combining learned dynamic reliability with static source-code maintainability. Because maintainability is unobserved in dynamic simulation ($m = [1, 0]$), headline results report $\hat{I}^*(v)$ directly, while domain reweighting sensitivity is evaluated against the static RM baseline in Supplementary §S3.

## 4.3 Ground-Truth Simulation Oracles

To evaluate predictive accuracy prior to deployment without relying on production runtime telemetry, SaG executes discrete-event failure simulations over the raw structural multigraph $G_{\text{structural}}$. We establish a formal taxonomy of four component-level oracles and one relationship-level oracle:

-   **Cascade Reachability Oracle ($I^*(v)$)**, via `FaultInjector`: crashes component $v$, propagates outages across dependent topics, brokers and links by breadth-first traversal, and returns the mean fractional feed loss over the subscriber population, each subscriber contributing the unweighted mean loss of the topics it subscribes to. A topic’s feed loss is the fraction of its publishers that have failed (for a topic with no publisher, the fraction of its failed routers), scaled by a QoS ladder ($\times 1.2$ `RELIABLE`, $\times 1.15$ high/urgent priority, $\times 1.05$ medium) and clamped to $[0,1]$. The denominator is the full subscriber set of the intact graph, so subscribers that themselves fail are retained in the average rather than excluded. The implementation admits a per-publisher rate weighting, but rates are declared per topic throughout our corpus, so it reduces exactly to this publisher fraction. This is the **primary continuous target label** throughout.

-   **Multi-Metric Composite Oracle ($I_{\text{comp}}(v)$)**, via `FailureSimulator`: $$I_{\text{comp}}(v) = 0.35 \cdot \Delta\text{Reachability} + 0.25 \cdot \Delta\text{Fragmentation} + 0.25 \cdot \Delta\text{Throughput} + 0.15 \cdot \Delta\text{FlowDisruption}$$ each term weighted by operational severity $s(t) = w(t) \cdot \text{rate}(t)$. These four coefficients are AHP-derived on the same footing as the rest of the framework’s weights, from a Saaty pairwise-comparison matrix over the four impact criteria (Supplementary §S4): the raw priority vector is $(0.389, 0.255, 0.255, 0.100)$, and the $\lambda = 0.7$ shrinkage toward a uniform prior that the framework applies throughout yields $(0.347, 0.254, 0.254, 0.145)$, which the shipped constants round. That matrix is rank-one, so it records where the constants came from without independently justifying them. They are not fitted to any evaluation, and they are not swept in the supplementary sensitivity analysis — a gap we note because $I_{\text{comp}}$ supplies the labels for the real-world evaluation of the explanation layer (Supplementary §S7). $\text{rate}(t)$ reads from a runtime telemetry profile when one is attached; none is attached anywhere in this paper, so $s(t) = w(t)$ throughout and the rate channel should be read as an interface for deployment-time calibration rather than an active term. Publication frequency and payload size still reach $s(t)$ through $w(t)$’s own modulators (§3.2).

-   **Dynamic Queue-Flow Oracle ($I_{\text{dyn}}(v)$)**, via `MessageFlowSimulator` on SimPy [77]: simulates emission rates, stochastic latencies, broker buffer saturation and queue drops under fault injection, extracting the drop in delivered message rate to surviving consumers.

-   **Change-Propagation Oracle ($I_M(v)$)**, via `ChangePropagationSimulator`: a deterministic reverse-dependency traversal quantifying maintenance change impact, $$I_M(v) = 0.45\,\text{ChangeReach}(v) + 0.35\,\text{WeightedChangeImpact}(v) + 0.20\,\text{NormalizedChangeDepth}(v).$$

-   **Relationship (Edge) Removal Oracle ($I_{\text{edge}}(u,v)$):** the systemic impact of severing one dependency while both endpoints stay operational. Writing $\bar{I}_{\text{comp}}(G)$ for the mean composite impact over $G$: $$\label{eq:edge_crit}
        I_{\text{edge}}(u,v) = \bar{I}_{\text{comp}}\big(G \setminus \{(u,v)\}\big) - \bar{I}_{\text{comp}}(G)$$

#### Topic Criticality Label Masking

`FailureSimulator` can blend a declared `Topic.criticality` into its severity term; this is disabled, because that field is a GNN input feature () and consuming it would score the predictor against a transformation of its own input.

**Primary Oracle Declaration and Role Assignment.** Because the three reliability-facing oracles ($I^*$, $I_{\text{comp}}$, $I_{\text{dyn}}$) measure distinct operational constructs, we designate **$I^*(v)$ (`FaultInjector`) as the primary oracle** for all predictive ranking results (Tables 6–8, RQ1–RQ3). $I_{\text{comp}}(v)$ is reserved for Validate-stage quality gates, $I_{\text{dyn}}(v)$ serves as an independent convergent-validity probe, and $I_M(v)$ serves as a structural maintainability reference.

**Cross-Oracle Convergent Validity and Critical-Set Bounds.** Measured across seven benchmark scenarios and five random seeds on the Application population, the mean Spearman rank correlation is $\rho = 0.907$ for $(I_{\text{dyn}}, I^*)$, $\rho = 0.425$ for $(I_{\text{comp}}, I^*)$, and $\rho = 0.427$ for $(I_{\text{comp}}, I_{\text{dyn}})$. The strong rank agreement ($\rho = 0.907$) between the behavioral queue-flow oracle and the topological cascade injector provides independent convergent evidence across distinct simulation paradigms. However, agreement on the top-$K$ critical set ($K = 0.2n$) is more conservative (mean Jaccard overlap of $0.49$ for the strongest pair and $0.24$–$0.28$ for the two $I_{\text{comp}}$ pairs, vs. $0.111$ expected by chance), highlighting the intrinsic sensitivity of discrete thresholding in non-linear cascades. Consequently, results established against one oracle are never transferred to another; every evaluation metric explicitly references its underlying simulation oracle.

## 4.4 Input–Label Independence Guarantee

To eliminate data leakage and ensure rigorous evaluation, SaG enforces strict architectural separation between inputs and labels:

-   **Feature Space:** Constructed exclusively from $G_{\text{analysis}}$ using static structural topology, static code analysis (SCA) metrics, and declared QoS contracts.

-   **Label Space:** Evaluated exclusively on raw $G_{\text{structural}}$ through independent simulation oracles (`FaultInjector`, `FailureSimulator`, `MessageFlowSimulator`).

No simulation outputs, failure trace histories, or dynamic execution telemetry are ever exposed as input features to the GNN or the explanation layer.

#### What this guarantee does and does not establish

The separation rules out circular feature construction: no predictor can read a transformation of the quantity it is scored against. It does not establish statistical independence between features and labels, and we do not claim it does. $G_{\text{analysis}}$ is a deterministic projection of $G_{\text{structural}}$ (§3.2), so the labels are — up to simulator seed — a deterministic function of the same topology the features are computed from. Two consequences follow, and both bound the results of §7.

First, $I^*(v)$ is itself a topological functional: a breadth-first reachability computation over $G_{\text{structural}}$ scaled by a QoS ladder. The predictive task is therefore to recover a closed-form function of a graph from features of that same graph. Read that way, it is unsurprising that an unparameterized centrality score is competitive with a trained model (§7.1); the parity we report there is the expected outcome of the setup rather than a surprising failure of graph learning, and we flag it here so the reader does not have to infer it.

Second, and more restrictively, no result in this paper is validated against an observed failure. Every label — on synthetic topologies and on the five open-source systems alike — is simulator-derived. What the evaluation can establish is whether a learned model recovers a simulator’s ordering on architectures it was not trained on. Whether that ordering corresponds to which components actually fail in production is a question this design cannot answer, and §8.3 treats it as the study’s principal construct-validity threat.

# 5. The Explanation Layer: Standards-Grounded Criticality Attribution

The predictor of §4 answers *where* to act. It does not answer *what to do*, and neither does the oracle that scores it — both return impact, not cause attributed in standardized quality terms. A component may be critical because it is a single point of failure, because it propagates errors widely, or because it is a high-coupling maintenance bottleneck. These diagnoses call for distinct architectural repairs: replicating the host or broker, decoupling the topic, or refactoring the module. This section presents the layer supplying that diagnosis. SaG decomposes component and relationship criticality into a standards-grounded quality profile, computed over the same typed node features (§3.4) but sharing no parameters with the predictor, and applied to whatever the predictor flagged — triage rather than data flow (Figure 1).

## 5.1 Grounding in ISO/IEC Standards

In accordance with **ISO/IEC 25010:2023** (Product Quality Model) [9], **ISO/IEC 25019:2023** (Quality-in-Use) [10], and **ISO/IEC 25022:2016** (Measurement of Quality-in-Use) [78], SaG formalizes two primary criticality constructs:

-   **Component Criticality ($D_1$):** The degree to which the sudden failure, unexpected termination, or severe degradation of an individual component reduces the system’s capacity to deliver required services within its operational context of use.

-   **Relationship Criticality ($D_2$):** The degree of systemic service degradation resulting from the severance, partitioning, or failure of a specific dependency or communication channel while both endpoint components remain operational.

Criticality is evaluated across two orthogonal quality characteristics: **Reliability ($R$)** and **Maintainability ($M$)**. The two are separated because they imply different repairs, not because they predict different runtime quantities: this layer attributes structural fragility and does not forecast latency, throughput or queue occupancy. Table 3 outlines this Reliability–Maintainability (RM) quality decomposition, mapping ISO/IEC sub-characteristics to architectural questions, underlying graph metrics, and targeted engineering remediations.

**Table 3.** The Reliability–Maintainability (RM) quality decomposition.

| **Dimension**             | **Sub-Characteristic**       | **Architectural Question**          | **Underlying Graph Metrics**                                                                    | **Role / Remediation**                             |
|:--------------------------|:-----------------------------|:------------------------------------|:------------------------------------------------------------------------------------------------|:---------------------------------------------------|
| **Reliability ($R$)**     | **Fault Tolerance ($FT$)**   | How broadly does failure propagate? | Reverse PageRank on $G^\top$, in-degree, cascade depth                                          | Reliability Eng.: add redundancy, circuit breakers |
|                           | **Availability ($A$)**       | Is this a single point of failure?  | Directed articulation score (raw + QoS-weighted), bridge ratio, connectivity degradation        | DevOps/SRE: replicate host/broker                  |
| **Maintainability ($M$)** | **Modularity/Modifiability** | How complex and coupled is this?    | Betweenness, QoS-weighted out-degree, Code Penalty, coupling-risk imbalance, inverse clustering | Architect: refactor code, decouple                 |

*Coverage Scope:* SaG focuses specifically on Reliability and Maintainability. Safety (which requires domain-specific hazard logs, such as ISO 26262 Automotive Safety Integrity Level [ASIL] ratings) and Security (which requires explicit threat models, such as STRIDE [Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege]) fall outside purely structural topology analysis and are reserved for domain-specific extensions.

## 5.2 Composite Quality Score Formulation

All raw topological and code metrics are rank-normalized to $[0, 1]$. Quality sub-characteristics are formulated hierarchically using the Analytic Hierarchy Process (AHP) [53]:

1.  **Fault Tolerance ($FT(v)$):** Measures error cascade potential on the transpose graph $G_{\text{analysis}}^\top$ (where edges follow failure propagation from dependency to dependent): $$FT(v) = 0.45 \cdot \text{RPR}(v) + 0.30 \cdot \text{Deg}_{\text{in}}(v) + 0.25 \cdot \text{CDPot}_{\text{enh}}(v)$$ where $\text{RPR}(v)$ is Reverse PageRank, $\text{Deg}_{\text{in}}(v)$ is normalized in-degree, and $\text{CDPot}_{\text{enh}}(v)$ is the enhanced Cascade Depth Potential term.

2.  **Availability ($A(v)$):** Identifies structural single points of failure (SPOFs) across five terms — directed articulation severity, its QoS-weighted variant, edge-level irrecoverability, connectivity degradation, and the component’s own QoS weight: $$\label{eq:availability}
        A(v) = 0.2563 \cdot \text{AP}_c^{\text{dir}}(v) + 0.1998 \cdot \text{QSPOF}(v) + 0.1998 \cdot \text{BR}(v) + 0.2563 \cdot \text{CDI}(v) + 0.0878 \cdot w(v)$$ where $\text{AP}_c^{\text{dir}}(v)$ is Directed Articulation Point severity, $\text{QSPOF}(v)$ is QoS-weighted Single Point of Failure severity, $\text{BR}(v)$ is Bridge Ratio (edge-level irrecoverability), $\text{CDI}(v)$ is Connectivity Degradation Index, and $w(v)$ is the component’s intrinsic QoS weight.

3.  **Reliability ($R(v)$):** Blends Fault Tolerance and Availability hierarchically: $$R(v) = r_\alpha \cdot FT(v) + (1 - r_\alpha) \cdot A(v), \quad r_\alpha = 0.36$$ The blend weight is written $r_\alpha$ throughout to distinguish it from the payload-size coefficient $\alpha$ of Equation (3); the two are unrelated parameters and are screened separately (Supplementary §S1). The shipped intra-dimension weights apply a $\lambda = 0.70$ shrinkage blend between the raw AHP-derived vector and a uniform prior (§7.3 reports ranking sensitivity to $\lambda$). We do not report consistency ratios for the Availability, Fault Tolerance and Maintainability matrices as evidence of soundness, because for three of them the statistic is uninformative: the matrices are rank-one, meaning each row is a scalar multiple of a single vector, which is the structure produced by writing down a priority vector and filling in the implied ratios rather than by eliciting pairwise judgements independently. A consistency ratio computed on such a matrix is near zero by construction — the Availability matrix returns a slightly *negative* $CR$, which is impossible for a genuinely elicited comparison since $\lambda_{\max} \ge n$ always. Supplementary §S4 states the matrices, the diagnostic, and which of them the property affects. These vectors should therefore be read as *declared constants with documented internal structure*, on the same footing as the topic-weight split of Equation (3), and not as independently elicited expert judgement validated by a consistency test.

4.  **Maintainability ($M(v)$):** Evaluates structural coupling combined with code-level static analysis across five terms — betweenness, QoS-weighted efferent coupling, the Code Quality Penalty, an afferent/efferent coupling-risk imbalance term, and inverse clustering: $$M(v) = 0.35 \cdot \text{BT}(v) + 0.30 \cdot w_{\text{out}}(v) + 0.15 \cdot \text{CQP}(v) + 0.12 \cdot \text{CouplingRisk}_{\text{enh}}(v) + 0.08 \cdot (1 - \text{CC}(v))$$ where $\text{BT}(v)$ is Betweenness Centrality, $w_{\text{out}}(v)$ is QoS-weighted efferent coupling (out-degree), $\text{CQP}(v)$ is the Code Quality Penalty, $\text{CouplingRisk}_{\text{enh}}(v)$ is an afferent/efferent coupling-risk imbalance term, and $\text{CC}(v)$ is the local Clustering Coefficient.

The baseline composite quality score $Q(v)$ combines both dimensions: $$Q(v) = 0.80 \cdot R(v) + 0.20 \cdot M(v)$$

When evaluating under a specific ISO/IEC 25019 Context of Use vector $\vec{\omega} = [q_R, q_M]^\top$, the score is reweighted dynamically: $$Q_{\text{domain}}(v) = q_R \cdot R(v) + q_M \cdot M_{\text{static}}(v)$$

Components are categorized into adaptive criticality tiers using box-plot quartile thresholds:

-   **CRITICAL:** $Q > Q_3 + 1.5 \cdot \text{IQR}$

-   **HIGH:** $Q_3 < Q \le Q_3 + 1.5 \cdot \text{IQR}$

-   **MEDIUM:** $Q_1 < Q \le Q_3$

-   **MINIMAL:** $Q \le Q_1$

This yields a directly actionable distinction: a component scoring high on $A$ but low on $FT$ is a single point of failure calling for horizontal replication, whereas one scoring high on $FT$ is an error-cascade hub calling for circuit breakers, rate limiting and bulkhead isolation. Whether acting on these diagnoses improves any measured runtime property is not evaluated here (§8.4).

# 6. Experimental Setup

## 6.1 Datasets and System Corpus

The evaluation corpus comprises 2,812 components across seventeen system architectures — twelve synthetic topologies that form the inductive cross-validation folds, and five real-world reference systems withheld entirely from training — as detailed in Table 4:

**Table 4.** Experimental evaluation corpus. The twelve synthetic topologies are the inductive Leave-One-Scenario-Out folds of Table 8; the five real-world systems are withheld from every training fold and used only for zero-shot transfer (§7.4). Counts are read from the committed topology files rather than from the generator configurations.

| **Dataset / Architecture**                                            | **System Paradigm**       | **$|V|$** | **$|V_{\text{app}}|$** | **Topics** | **Brokers** | **Hosts** | **Libs** |  **$|E|$** |
|:----------------------------------------------------------------------|:--------------------------|----------:|-----------------------:|-----------:|------------:|----------:|---------:|-----------:|
| *Synthetic evaluation scenarios (`evaluation` role in the manifest)*  |                           |           |                        |            |             |           |          |            |
| **Autonomous Vehicle (AV)**                                           | ROS 2 Cyber-Physical      |       152 |                     80 |         40 |           4 |         8 |       20 |        773 |
| **Enterprise Pub-Sub**                                                | Kafka Event Mesh          |       520 |                    300 |        120 |          10 |        40 |       50 |      3,389 |
| **Financial Trading**                                                 | Low-Latency Pub-Sub       |       124 |                     60 |         35 |           5 |         6 |       18 |        583 |
| **Healthcare Integration**                                            | HL7/FHIR Event Mesh       |        98 |                     50 |         25 |           3 |         8 |       12 |        428 |
| **Hub-and-Spoke Enterprise**                                          | Broker-Centric Messaging  |       139 |                     70 |         30 |           2 |        12 |       25 |        751 |
| **IoT Smart City**                                                    | MQTT Telemetry Mesh       |       326 |                    200 |         80 |           6 |        30 |       10 |      1,331 |
| **Microservices Mesh**                                                | Cloud-Native Services     |       186 |                     90 |         45 |           6 |        15 |       30 |        680 |
| **Telecom RAN**                                                       | 5G Radio Access Network   |       225 |                    120 |         55 |           8 |        20 |       22 |        858 |
| **Industrial SCADA**                                                  | Plant Control Telemetry   |       254 |                    140 |         70 |           4 |        25 |       15 |        818 |
| **Real-Time Gaming**                                                  | Multiplayer State Sync    |       158 |                     75 |         38 |           5 |        12 |       28 |        643 |
| **Logistics Fleet**                                                   | Vehicle Telematics Mesh   |       205 |                    110 |         50 |           7 |        18 |       20 |        792 |
| *Synthetic case study (LOSO fold; `case_study` role in the manifest)* |                           |           |                        |            |             |           |          |            |
| **Air Traffic Management (ATM)**                                      | ICAO Global ATM Concept   |        74 |                     26 |         27 |           5 |         8 |        8 |        271 |
| *Real-world reference systems (never used as training folds)*         |                           |           |                        |            |             |           |          |            |
| **Autoware.universe [79]**                                          | Real-World ROS 2 Autoware |        75 |                     32 |         24 |           3 |         6 |       10 |        179 |
| **Cloud Microservices [80]**                                        | Real-World GCP Boutique   |        60 |                     22 |         20 |           4 |         6 |        8 |        128 |
| **Train-Ticket [81]**                                               | Real-World Microservices  |        90 |                     41 |         30 |           3 |         8 |        8 |        162 |
| **Home Assistant [82]**                                             | Real-World Smart Home     |        63 |                     24 |         22 |           3 |         6 |        8 |        119 |
| **EdgeX Foundry [83]**                                              | Real-World Industrial IoT |        63 |                     22 |         24 |           3 |         6 |        8 |        112 |
| **Synthetic subtotal (12 LOSO folds)**                                |                           | **2,461** |                  1,321 |        615 |          65 |       202 |      258 | **11,317** |
| **Real-world subtotal (5 systems)**                                   |                           |   **351** |                    141 |        120 |          16 |        32 |       42 |    **700** |
| **Total**                                                             |                           | **2,812** |                        |            |             |           |          | **12,017** |

Here, $|V|$ is the sum of all five entity-type counts per scenario. The twelve synthetic topologies total 2,461 components; the eleven carrying the manifest’s `evaluation` role account for 2,387 of these and the ATM case study for the remaining 74. The five real-world systems add 351. $|E|$ counts every raw structural relationship instance recorded in the scenario specification — the native substrate that simulation oracles traverse — rather than the derived `DEPENDS_ON` projection constructed for GNN training. All six structural relation types are included in that count (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`); we note the convention explicitly because the generator’s own build log omits `CONNECTS_TO` and therefore reports slightly lower per-scenario totals. Every count in Table 4 is read directly from the committed topology files, whose byte-identical regeneration from configuration is asserted in continuous integration (§6.1.1).

Five real-world architectures were transcribed from authentic open-source repositories using dedicated architectural adapters. The synthetic scenarios were produced by a parameterized topology generator: each is fully defined by a committed configuration specifying a random seed, per-entity-type counts, seven-number summaries (mean, median, standard deviation, minimum, maximum, $Q_1$, $Q_3$) for application publish and subscribe fan-out, applications per host, library fan-in, topic payload size, and categorical distributions over the three QoS dimensions. Graph degree distributions and clustering emerge directly from these parameters rather than being synthetically forced. Supplementary §S5 reports the generative parameters governing each topology’s shape; complete configurations are included in the replication package.

**Corpus Composition across Experimental Regimes.** The experimental evaluation spans three complementary regimes with precisely bounded corpora:

1.  *In-Distribution Evaluation (Table 6):* Evaluated on the seven core synthetic domains from Table 4 using stratified 60% train / 20% validation / 20% test node splits over five random seeds.

2.  *Inductive Leave-One-Scenario-Out (LOSO) Cross-Validation (Table 8):* Evaluated across twelve distinct inductive folds totaling 2,461 components: the seven core synthetic scenarios, four extended domain topologies (Telecom RAN, Industrial SCADA, Realtime Gaming, and Logistics Fleet) — 2,387 components between them — and an Air Traffic Management (ATM) network scenario contributing the remaining 74. In each fold, models are trained on eleven graphs and tested zero-shot on the held-out twelfth graph.

3.  *Real-World Architectural Transfer (Table 11; Supplementary §S7):* The five open-source real-world systems (Autoware.universe, Cloud Microservices, Train-Ticket, Home Assistant, EdgeX Foundry) are never used as training folds; they are withheld entirely and used strictly for zero-shot architectural transfer validation.

Not every analysis in §7 runs on the full corpus: the sensitivity sweeps and the detection benchmark predate the four extended domains and operate on smaller cached subsets. Because a reader comparing figures across subsections would otherwise have no way to tell which population a number belongs to, Table 5 states the subset used by each analysis. Comparisons are made only within a row.

**Table 5.** Which corpus subset backs each analysis. Figures from different rows are not directly comparable, and we do not compare them.

| **Analysis**                           | **Scenario subset**                 | **$n$** | **Reported in**    |
|:---------------------------------------|:------------------------------------|:-------:|:-------------------|
| In-distribution ranking                | Seven core synthetic domains        |    7    | Tables 6–7         |
| Inductive LOSO                         | Eleven evaluation scenarios $+$ ATM |   12    | Table 8, §§7.1–7.2 |
| QoS edge-feature ablation              | Same twelve LOSO folds              |   12    | §7.3.1             |
| Weight sweeps and Morris screening     | Six–seven core synthetic domains    |   6–7   | Supplementary S1   |
| Cross-oracle convergent validity       | Seven core synthetic domains        |    7    | Table 10           |
| Anti-pattern detection, stratification | Seven core domains $+$ ATM          |    8    | §7.3.3             |
| Relational attention illustration      | ATM case study alone                |    1    | Figure 4           |
| Real-world zero-shot transfer          | Five open-source systems            |    5    | Table 11, Supp. S7 |

### 6.1.1 Reproducibility of the Corpus

The benchmark corpus is designed to be fully regenerable rather than merely statically archived. Each dataset is deterministically generated from its configuration file via:

> `python cli/generate_graph.py batch –input-dir data/scenarios –output-dir <dir>`

A companion manifest records the random seed, entity counts, git commit hash, and a SHA-256 cryptographic digest for each emitted topology. Continuous integration regression tests assert that every committed dataset regenerates *byte-identically* from its configuration and that all disk digests match the manifest. This guarantees that third parties can reproduce the exact graphs used in our experiments, rather than simply sampling from similar distributions.

## 6.2 Baselines and Evaluated Predictors

We evaluate four primary predictor configurations drawn from three families. Predictor names state the family and the substrate: an `-N` suffix marks a model trained on the *native* multigraph, its absence the derived Application–Library flow projection, and a `-QoS` suffix marks a configuration that consumes declared QoS contracts. *SaG* throughout denotes the framework, never an individual predictor.

1.  **Heterogeneous graph learning (typed HGT).** **HGT-QoS** (proposed): relation-specific Heterogeneous Graph Transformer (§4) ingesting the complete native multigraph with 16-dimensional continuous-categorical edge features that encode middleware QoS contracts. Its ablation **HGT**, which masks those QoS dimensions, is reported in §7.3.1.

2.  **Homogeneous graph learning (untyped GAT).** **GAT-N-QoS**: homogeneous Graph Attention Network [62] trained on the identical native multigraph substrate with per-type input projections, but untyped, single-relation message passing. Its edge channel carries the scalar QoS aggregate $w(e)$ — dimension $0$ of the same 16-D encoding HGT-QoS consumes — rather than the per-dimension decomposition; no homogeneous architecture in our suite ingests the full 16-D vector. The HGT-QoS–GAT-N-QoS contrast therefore bounds the *joint* contribution of relational typing and per-dimension QoS encoding. §7.3.1 separates the second factor within the typed architecture, and the corresponding unweighted ablation is **GAT-N**.

3.  **Structural baselines (training-free).** **Topo-QoS**: QoS-weighted topological centrality evaluated on the derived application flow projection.

4.  **Structural baselines (training-free).** **Topo**: structural centrality combining unweighted betweenness centrality and articulation point scoring on the flow projection.

In addition, the out-of-distribution evaluation (Table 8) reports **RM** ($Q(v)$, the deterministic hierarchical quality attribution model of §5) as a diagnostic reference baseline. RM is not fitted to rank failure impact; its inclusion demonstrates how much learned relational prediction adds over static structural attribution (§1.2). Furthermore, deterministic RM scoring drives every sensitivity sweep in §7.3, where closed-form formulations isolate parameter effects from neural training stochasticity.

### 6.2.1 Evaluation Substrates and Parity Guarantee

Predictors in this study operate on matched substrates within their respective families:

-   **Graph Learning Models (GAT-N-QoS, HGT-QoS):** Both learned neural predictors ingest the complete native typed multigraph across all five entity types in both in-distribution (Table 6) and out-of-distribution Leave-One-Scenario-Out (Table 8) evaluations — which is what the shared `-N` suffix records — so the comparison carries no multi-entity visibility confound. Node type reaches homogeneous GAT-N-QoS only through its per-type input projection layer, while message passing uses untyped GATConv with shared weights across all edges; in contrast, heterogeneous HGT-QoS employs relation-specific HGTConv weight matrices per edge triple alongside edge-type encodings. Substrate and node features are matched; the edge channel is not, since GAT-N-QoS consumes the scalar QoS aggregate $w(e)$ where HGT-QoS consumes all 16 dimensions (§6.2). Comparisons between the two therefore isolate relation-specific parameterization jointly with per-dimension QoS encoding, and we report the QoS factor separately in §7.3.1 rather than attributing the whole margin to typing.

-   **Training-Free Structural Baselines (Topo, Topo-QoS):** Topological baselines are evaluated on the derived Application–Library `DEPENDS_ON` projection (§3.2). This projected substrate is necessary because in raw publish–subscribe multigraphs, Application nodes never route messages directly, resulting in near-zero betweenness and bridge ratios that yield degenerate, uninformative scores.

-   **Ground-Truth Independence Guarantee:** Crucially, **no predictor in either group accesses $G_{\text{structural}}$**: that raw topology is strictly reserved as the substrate for ground-truth simulation oracles (§4.4), a guarantee formally verified by `tests/test_independence_guarantee.py`.

Regardless of substrate, all variants are scored on an identical, independently resolved Application node set ($V_{\text{app}}$, §6.3).

## 6.3 Evaluation Metrics and Protocols

**Figure accessibility.** All figures in this paper use the high-contrast, colorblind-safe Okabe–Ito palette together with distinct marker, hatching and node-shape encodings, so that every distinction carried by colour is also carried by form and remains legible in monochrome.

-   **Ranking Precision:** Evaluated via Spearman rank correlation ($\rho$) and Kendall’s rank correlation ($\tau$) between predicted component rankings and ground-truth simulated impact $I^*(v)$ from the primary oracle (§4.3).

-   **Critical-Set Identification:** Measured via $F_1@K$, Precision@$K$, and Recall@$K$ for top-$K$ critical components, where $K = \text{round}(0.20 \cdot |V_{\text{app}}|)$. Because predicted and ground-truth sets both contain exactly $K$ elements, Precision, Recall, and $F_1$ coincide identically as the top-$K$ set overlap.

-   **Statistical Significance:** Assessed through paired Wilcoxon signed-rank tests [84] ($p < 0.05$) and non-parametric bootstrap 95% confidence intervals over folds ($B = 2{,}000$) [85, 86], reported for every out-of-distribution predictor in Table 8 and for the paired contrasts in §7.1. Given the sample size, we regard those intervals rather than the $p$-values as the primary evidence. The unit of analysis is the scenario (in-distribution, $n = 7$) or the fold (LOSO, $n = 12$). The in-distribution design sits at the floor of its own resolving power — the smallest attainable two-sided $p$ is $0.0156$ at $n = 7$, so only a near-unanimous sign pattern can register at all. The twelve-fold LOSO design is materially better resourced: its floor is $0.00049$, and it tolerates as many as four lost folds while still reaching $\alpha = 0.05$, provided those losses are the smallest in magnitude. Where a LOSO comparison nonetheless fails to reach significance (§7.1), the cause is therefore the size of the losing folds rather than the size of the corpus, and enlarging the corpus would not remedy it. We report $p$-values uncorrected and state here what a correction would do to them. In distribution the question is moot: with eight comparisons in Table 7, the $0.0156$ floor above already sits well outside $\alpha/8 = 0.00625$, so no in-distribution result could survive a correction regardless of the data. Out of distribution it bites. Applying Holm’s step-down correction across the ten full-population LOSO rank contrasts reported in §§7.1–7.3.1, three survive — Topo-QoS over Topo ($p = 0.0010$), Topo over RM ($p = 0.0005$), and the *unweighted* typing contrast, HGT over GAT-N ($p = 0.0010$) — while the QoS-weighted typing contrast ($p = 0.0122$) and the QoS edge-encoding ablation ($p = 0.0093$) do not clear their adjusted thresholds. We report both readings rather than the more favourable one: the typing effect is corrected-significant in the pair that isolates it most cleanly and nominally significant in the other, and directional consistency across folds — 11 of 12 in both pairs — is what we ask the reader to weigh alongside either $p$.

**Pre-registration.** The primary out-of-distribution contrast — HGT-QoS against the training-free `Topo-QoS` baseline under LOSO, with the fold as the unit of analysis, the five seeds fixed in advance, and Holm correction across the primary and one secondary comparison — was registered before any result under the revised harness existed, together with the commitment that the outcome would be reported as measured rather than re-run under a different configuration if the margin failed to reach significance. It did fail to reach significance (§7.1), and it is reported here as registered. The registration document, including an amendment recording a reduction in the inner selection budget made before any outer result was available, is part of the replication package.

### Evaluation Population

Every predictor within a given evaluation table is scored on an identical node population, resolved strictly from scenario topology and simulation ground truth — never from any model’s predictions. Unless otherwise noted, this population is the **Application** set ($V_{\text{app}}$). This aligns with the framework’s primary objective (forecasting application-layer cascading failures) and ensures a fair common denominator across both typed and untyped predictors. Pooling node types into a single global ranking conflates distinct base rates and impact distributions, shifting the resulting rank correlation outside the envelope of per-type correlations (§7.3). We therefore report stratified, single-population metrics throughout and explicitly identify any pooled figures.

### Evaluation Protocols

-   **In-Distribution Evaluation:** 60% train / 20% validation / 20% test node splits over five seeds $\{42, 123, 456, 789, 2024\}$. Each split is a deterministic function of node identity *and* seed, so a seed redraws the partition as well as the initialisation. The consequence for Table 6 is that dispersion across seeds is split noise on 10–60 held-out nodes for every variant — including the training-free baselines, whose scores are otherwise deterministic — and additionally training noise for the learned predictors. Per-seed standard deviations are reported in that table for this reason.

-   **Inductive Leave-One-Scenario-Out (LOSO):** Models are trained on eleven scenarios and evaluated zero-shot on the held-out twelfth, testing zero-shot generalization across distinct architectural domains. Three protocol properties bear on the validity of the comparison, and all three apply identically to every variant. First, every learned variant receives the identical training set of $N-1$ graphs: one scenario is designated the *primary* graph and the remainder are passed as additional inductive graphs. Second, message-passing depth is fixed at three layers for every fold rather than derived from the primary graph’s size, so capacity does not vary with which scenario is held out. Third, early stopping and checkpoint selection are driven by a validation split *within* the primary training graph. We flag this as a weakness of the protocol rather than a feature of it: selecting on a split of a graph the model is already fitting selects for in-distribution fit, under a protocol whose entire purpose is distribution shift, and it will if anything flatter every learned variant relative to the training-free baselines. The released harness implements the alternative — holding one training scenario (deterministically the median-sized inductive graph) out of the loss entirely and selecting on its whole labelled population, available as `--inner-val-scenario auto` — but the runs reported in this paper do not use it, and we describe what was run rather than what we would recommend (§8.4). Crucially for the comparisons that follow, the rule is identical for every variant, so it cannot account for a difference between them. The outer holdout participates in none of these choices.

-   **Real-World Architectural Transfer:** Evaluating models trained on synthetic corpora zero-shot on authentic open-source distributed systems without fine-tuning.

# 7. Results and Empirical Analysis

This section presents empirical results for RQ1–RQ5 across the twelve-fold inductive benchmark and five authentic open-source distributed systems. Evaluated populations are strictly stratified on the Application service set ($V_{\text{app}}$) under the input–label independence guarantee (§4.4).

## 7.1 RQ1: Graph Learning vs. Structural Baselines

Table 6 presents in-distribution held-out Spearman rank correlation ($\rho$) against simulated cascade impact $I^*(v)$ across seven representative distributed architecture domains.

**Table 6.** In-distribution held-out $\rho$ against $I^*(v)$: mean over five seeds $\pm$ spread; $n$ = held-out Application count. All four learned variants use the identical native multigraph, differing only in typing and edge channel (§6.2); topological baselines use the flow projection. Each seed redraws the 60/20/20 split as well as the initialization. Read with Table 7.

| **Scenario**          | **$n$** |              **Topo**               |         **Topo-QoS**         |          **GAT-N**           |      **GAT-N-QoS**       |           **HGT**            |         **HGT-QoS**          |
|:----------------------|--------:|:-----------------------------------:|:----------------------------:|:----------------------------:|:------------------------:|:----------------------------:|:----------------------------:|
| **AV System**         |      16 |      0.362 $\pm$0.169       |   0.762 $\pm$0.159   |   0.632 $\pm$0.224   | 0.758 $\pm$0.110 | **0.797** $\pm$0.092 |   0.737 $\pm$0.126   |
| **Enterprise**        |      60 |      0.421 $\pm$0.089       |   0.803 $\pm$0.060   |   0.638 $\pm$0.355   | 0.864 $\pm$0.033 |   0.726 $\pm$0.381   | **0.903** $\pm$0.014 |
| **Financial Trading** |      12 |      0.169 $\pm$0.375       |   0.636 $\pm$0.125   | **0.864** $\pm$0.047 | 0.573 $\pm$0.483 |   0.758 $\pm$0.189   |   0.590 $\pm$0.245   |
| **Healthcare**        |      10 | -0.275 $\pm$0.267 |   0.660 $\pm$0.190   | **0.863** $\pm$0.039 | 0.802 $\pm$0.068 |   0.475 $\pm$0.387   |   0.478 $\pm$0.327   |
| **Hub-and-Spoke**     |      14 |      0.025 $\pm$0.230       | **0.545** $\pm$0.260 |   0.538 $\pm$0.219   | 0.441 $\pm$0.308 |   0.367 $\pm$0.571   |   0.425 $\pm$0.462   |
| **IoT Smart City**    |      40 |      0.124 $\pm$0.129       |   0.361 $\pm$0.082   |   0.790 $\pm$0.103   | 0.639 $\pm$0.191 |   0.825 $\pm$0.053   | **0.848** $\pm$0.045 |
| **Microservices**     |      18 |      0.337 $\pm$0.202       | **0.635** $\pm$0.136 |   0.514 $\pm$0.105   | 0.497 $\pm$0.209 |   0.421 $\pm$0.301   |   0.428 $\pm$0.327   |
| **Mean**              |       — |              **0.166**              |          **0.629**           |          **0.691**           |        **0.653**         |          **0.624**           |          **0.630**           |

**Table 7.** Paired Wilcoxon signed-rank tests across in-distribution scenarios ($n = 7$, two-sided).

| **Comparison**             | **$\Delta\rho$** | **Won** | **Wilcoxon $W$** | **$p$-value** | **Significance**             |
|:---------------------------|-----------------:|:-------:|-----------------:|:-------------:|:-----------------------------|
| **HGT-QoS vs. Topo**       |       **+0.464** |   7/7   |              0.0 |  **0.0156**   | **Significant** ($p < 0.05$) |
| **Topo-QoS vs. Topo**      |       **+0.463** |   7/7   |              0.0 |  **0.0156**   | **Significant** ($p < 0.05$) |
| **GAT-N vs. HGT-QoS**      |         $+0.061$ |   4/7   |              9.0 |     0.469     | Not significant              |
| **HGT-QoS vs. GAT-N-QoS**  |         $-0.023$ |   3/7   |             12.0 |     0.813     | Not significant              |
| **HGT vs. GAT-N**          |         $-0.067$ |   3/7   |              8.0 |     0.375     | Not significant              |
| **HGT-QoS vs. HGT**        |         $+0.006$ |   5/7   |             11.0 |     0.688     | Not significant              |
| **HGT-QoS vs. Topo-QoS**   |         $+0.001$ |   2/7   |             10.0 |     0.578     | Not significant              |
| **GAT-N-QoS vs. Topo-QoS** |         $+0.024$ |   3/7   |             13.0 |     0.938     | Not significant              |

### Out-of-Distribution (LOSO) Generalization

In inductive Leave-One-Scenario-Out (LOSO) cross-validation, models are evaluated on their capacity to predict cascading criticality over completely unseen system topologies:

**Table 8.** Inductive LOSO evaluation, Application population, twelve folds. Learned variants share training set, substrate, depth and selection rule (§6.3), differing only in typing and edge channel. Fold score = mean over five seeds; **Fold $\sigma$** = spread of the twelve fold means; **Seed $\sigma$** = median within-fold spread (zero by construction for deterministic baselines); CI = bootstrap over folds ($B = 2{,}000$).

| **Predictor / Reference**                                            | **Mean LOSO $\rho$** |    **95% CI**    | **Fold $\sigma$** | **Seed $\sigma$** | **Critical-Set $F_1@K$** | **Requires Training** |     |
|:---------------------------------------------------------------------|:--------------------:|:----------------:|:-----------------:|:-----------------:|:------------------------:|:---------------------:|:---:|
| *Training-free structural baselines*                                 |                      |                  |                   |                   |                          |                       |     |
| **Topo**                                                             |        0.250         | $[0.128, 0.356]$ |       0.200       |         —         |          0.306           |          No           |     |
| **Topo-QoS**                                                         |        0.568         | $[0.418, 0.686]$ |       0.237       |         —         |          0.353           |          No           |     |
| *Learned predictors (shared native substrate, matched training set)* |                      |                  |                   |                   |                          |                       |     |
| **GAT-N**                                                            |        0.493         | $[0.437, 0.541]$ |     **0.092**     |       0.158       |          0.417           |          Yes          |     |
| **GAT-N-QoS**                                                        |        0.581         | $[0.493, 0.657]$ |       0.141       |     **0.017**     |          0.474           |          Yes          |     |
| **HGT**                                                              |        0.640         | $[0.565, 0.716]$ |       0.138       |       0.069       |          0.466           |          Yes          |     |
| **HGT-QoS**                                                          |      **0.695**       | $[0.631, 0.748]$ |       0.103       |       0.053       |        **0.507**         |          Yes          |     |
| *Diagnostic reference — not a ranking model*                         |                      |                  |                   |                   |                          |                       |     |
| **RM / $Q(v)$**                                                      |        0.133         | $[0.009, 0.247]$ |       0.215       |         —         |          0.258           |          No           |     |

Twelve LOSO folds are reported, comprising the eleven synthetic evaluation scenarios and the ATM case study, all specified in Table 4 (§6.1). In each fold, one scenario is held out for zero-shot testing while the model is trained exclusively on the remaining eleven. All variants are evaluated on the identical Application node set per fold (§6.3); paired Wilcoxon tests are conducted across the twelve folds, where the smallest attainable two-sided $p$ is $0.00049$. Per-fold evaluated populations range from 26 to 300 Application nodes, so $K = \text{round}(0.20\,|V_{\text{app}}|)$ ranges from 5 to 60. On $F_1@K$, HGT-QoS beats Topo-QoS in 8 of 12 folds ($\Delta = +0.154$, $W = 12.0$, $p = 0.034$) but separates from untyped GAT-N-QoS in 8 of 12 without reaching significance ($\Delta = +0.034$, $W = 25.0$, $p = 0.301$): critical-set identification distinguishes the typed learned model from the training-free baseline, but not from untyped learning.

**Label-noise ceiling.** These correlations are bounded by the reproducibility of the target they are scored against. Re-running the ground-truth oracle across the five seeds gives a test–retest rank correlation between $0.880$ and $1.000$ across the twelve folds (median $0.979$; ten of twelve at or above $0.95$), with Microservices the least reproducible at $0.880$. HGT-QoS’s $\rho = 0.695$ therefore recovers roughly $71\%$ of the attainable signal against the median ceiling, and no predictor in Table 8 can exceed the reproducibility of its own labels. Top-$K$ critical sets are the noisier construct by a wide margin: their cross-seed Jaccard has a median of $0.709$ and falls to $0.400$ (Microservices), $0.467$ (Telecom RAN) and $0.481$ (Logistics Fleet). That instability is the main reason the $F_1@K$ margins are less stable than the ranking margins, and it bounds how much weight any single critical-set comparison can carry. Notably, Microservices is both the least reproducible fold and one of the two on which typed learning loses (§7.2.1) — part of that deficit may be label noise rather than model failure.

Figure 3 summarises these results alongside critical-set identification and inter-oracle agreement.

**Key Insights for RQ1:**

1.  **Typed learning is the best configuration, but not demonstrably better than the QoS baseline.** HGT-QoS leads all predictors out-of-distribution ($\rho = 0.695$), and the typed-vs-untyped margin excludes zero decisively (§7.2). Against training-free *Topo-QoS*, HGT-QoS is $+0.127$ (9/12, $W = 16.0$, $p = 0.077$, CI $[+0.011, +0.255]$) and HGT is $+0.073$ (10/12, $p = 0.129$, CI $[-0.038, +0.196]$). We decline to read the first as a win: the margin is carried almost entirely by ATM, where Topo-QoS fails outright ($\rho = -0.086$, its only negative fold) against HGT-QoS’s $0.579$. Excluding that fold the margin falls to $+0.078$ (8/11, $p = 0.148$). A bootstrap interval that excludes zero on the strength of one fold is not evidence of general superiority, and we report interval and sensitivity together rather than quoting whichever is more favourable.

2.  **A QoS-weighted structural score is a genuinely strong baseline.** Topo-QoS reaches $\rho = 0.568$ zero-shot, beating unweighted Topo on 11 of 12 folds ($+0.318$, $p = 0.0010$) and remaining indistinguishable from untyped learning in both directions (GAT-N trails by $0.075$, $p = 0.077$; GAT-N-QoS leads by $0.013$, $p = 0.850$). Any claim that graph learning is *required* must be made against this baseline, not against unweighted centrality.

3.  **Power is not the limiting factor.** At $n = 12$ the design tolerates four lost folds and still reaches $\alpha = 0.05$, provided the losses are smallest in magnitude. HGT-QoS’s are not: it loses Enterprise ($-0.268$), Microservices ($-0.080$) and Real-Time Gaming ($-0.004$), and Enterprise is the largest $|\Delta|$ among them — which is what holds $W$ at $16.0$. Enlarging the corpus will not resolve this; the two substantive inversions must be understood instead (§7.2.1).

4.  **The explanation layer is weakly predictive, not noise.** RM/$Q(v)$ reaches $\rho = 0.133$, losing to unweighted Topo on all twelve folds ($-0.117$, $p = 0.0005$), so no ranking claim is made for it. Its interval $[0.009, 0.247]$ stays above zero and it supplies interpretable diagnostics without training (§5). It appears in Table 8 as a reference point, not a competitor.

![Figure 3](latex/figures/Figure_5.png)

*Figure 3. Results at a glance, Application population. (A) Out-of-distribution rank correlation per predictor across the twelve LOSO folds. (B) Critical-set identification at K = 20%. (C) Pairwise rank agreement between the three simulation oracles, against the chance baseline. Panels A and B are read directly from the same artifact as Table 8 and panel C from that behind Table 10; the ordering shown is whatever the data gives, which is why the untrained Topo-QoS baseline sits third rather than last.*

### 7.1.2 The Active Stratum: Ranking Quality Separated From Inertness Detection

Every LOSO figure above is a correlation over the full held-out Application population, and between $21\%$ (Microservices) and $52\%$ (Healthcare) of that population carries exactly zero simulated impact depending on the fold. A predictor can therefore score well by separating components that can propagate a failure from those that cannot, without ordering the propagating ones correctly. Because these are different capabilities with different operational value, we re-score all twelve folds on the *active stratum* — the $n_{>0}$ components with strictly positive ground-truth impact — using the same predictions, folds and seeds. Table 9 reports both.

**Table 9.** LOSO means over twelve folds on the full Application population ($\rho$) and restricted to components with strictly positive ground-truth impact ($\rho_{>0}$). Same predictions, folds and seeds; only the evaluated subset differs. Training-free baselines lose most of their apparent accuracy under the restriction; learned models lose far less.

| **Predictor**   | **$\rho$ (full)** | **$\rho_{>0}$ (active)** | **Retained** |
|:----------------|:-----------------:|:------------------------:|:------------:|
| **RM / $Q(v)$** |      $0.133$      |         $0.014$          |    $11\%$    |
| **Topo**        |      $0.250$      |         $0.064$          |    $26\%$    |
| **Topo-QoS**    |      $0.568$      |         $0.183$          |    $32\%$    |
| **GAT-N**       |      $0.493$      |         $0.322$          |    $65\%$    |
| **GAT-N-QoS**   |      $0.581$      |         $0.338$          |    $58\%$    |
| **HGT**         |      $0.640$      |         $0.382$          |    $60\%$    |
| **HGT-QoS**     | $\mathbf{0.695}$  |     $\mathbf{0.407}$     |    $59\%$    |

Three consequences follow, and they do not all favour the proposed model.

1.  **The training-free baselines are substantially weaker than their headline figures suggest.** Topo-QoS retains only $32\%$ of its full-population correlation on the active stratum, against $59\%$ for HGT-QoS, and it turns negative on three folds (ATM $-0.556$, Healthcare $-0.248$, and Topo on Financial Trading $-0.185$). Much of what a QoS-weighted centrality score contributes is the detection of inert components, which the degree-zero structure of the projection makes nearly free. This is the clearest evidence in the paper that learning contributes something a structural heuristic does not.

2.  **RQ1’s verdict is nonetheless unchanged.** On the active stratum HGT-QoS leads Topo-QoS by $+0.224$ (9/12, $W = 18.0$, $p = 0.110$) — a margin nearly twice the full-population $+0.127$, but with enough fold-level variance that it still does not reach significance. We report it as a widened but statistically undemonstrated margin, and the RQ1 conclusion of §7.1 stands as written.

3.  **The typed-versus-untyped result survives; the QoS ablation does not.** Typing remains significant under the restriction (HGT-QoS vs. GAT-N-QoS $+0.069$, 10/12, $p = 0.043$; HGT vs. GAT-N $+0.060$, 10/12, $p = 0.027$), so the central claim of §7.2 is not an artifact of zero-inflation. The QoS *encoding*, by contrast, loses its significance entirely: the $+0.054$ typed gain ($p = 0.009$) falls to $+0.025$ ($p = 0.151$), and the untyped gain from $+0.088$ ($p = 0.034$) to $+0.017$ ($p = 0.791$). We therefore narrow the RQ3 ablation claim in §7.3: declared QoS attributes help identify *which* components can propagate a failure, and we have no evidence that they help rank the ones that do.

## 7.2 RQ2: Value of Typed Heterogeneity

To evaluate the specific contribution of node and edge typing, we contrast the relation-specific Heterogeneous Graph Transformer against the homogeneous Graph Attention Network on the shared native multigraph substrate, with the identical training set, depth and model-selection rule:

-   **In-Distribution Fitting (Table 6): typing does not help.** On familiar architectures, typed message passing carries no advantage at all. HGT-QoS reaches $\rho = 0.630$ against GAT-N-QoS’s $0.653$ ($\Delta\rho = -0.023$, won in 3 of 7, $W = 12.0$, $p = 0.813$), and the unweighted pair runs the same way (HGT $0.624$ vs. GAT-N $0.691$, $-0.067$, $p = 0.375$). The best in-distribution mean in the table belongs to *untyped* GAT-N. None of these differences is significant at $n = 7$, and per-seed dispersion is large (median within-scenario $\sigma$ of $0.105$–$0.301$ across the learned variants, reaching $0.571$ on Hub-and-Spoke), so the honest reading is parity rather than a reversal — but there is no in-distribution typing benefit to report.

-   **Out-of-Distribution Generalization (LOSO, Table 8):** Under inductive distribution shift, the typed advantage is the most robust effect in this study. HGT-QoS outperforms GAT-N-QoS by $+0.114$ ($\rho = 0.695$ vs. $0.581$), winning 11 of 12 folds ($W = 8.0$, $p = 0.0122$; 95% CI $[+0.048, +0.170]$). The unweighted pair replicates it and slightly exceeds it: HGT over GAT-N by $+0.147$, 11 of 12 folds, $W = 1.0$, $p = 0.0010$, CI $[+0.101, +0.185]$.

**Typing is an inductive bias, not a capacity advantage.** The contrast between the two regimes is the finding. Given training data from the same generator as the test split, untyped message passing matches typed and nominally exceeds it. Asked to transfer to an unseen architecture, the typed model wins 11 folds of 12 by $+0.114$, while the untyped model loses $0.088$ of its in-distribution standing ($0.653 \rightarrow 0.581$) and the typed model gains $0.065$ ($0.630 \rightarrow 0.695$). The reading is that relation-specific parameters add no fitting capacity — with enough same-distribution data an untyped attention mechanism recovers the same structure — but encode which distinctions survive a change of topology. Distinguishing `PUBLISHES_TO` dissemination from `RUNS_ON` placement is a constraint rather than extra expressiveness, and constraints pay off exactly where distribution shift would otherwise mislead. This is a more specific claim than we set out to test, and a more useful one for a pre-deployment gate, where every architecture analysed is by construction new.

The single fold typed learning loses is the same in both contrasts: ATM, the smallest graph in the corpus (26 Applications) and the only one designated a case study. Excluding it, both contrasts win every remaining fold ($+0.139$ and $+0.161$, 11/11, $p = 0.0010$). We report the twelve-fold figure as the headline regardless, because excluding an inconvenient fold to recover unanimity is precisely the move a reader should distrust.

#### Methodological Controls and Substrate Parity

To ensure that the typed–untyped margin reflects genuine architectural inductive biases rather than experimental artifacts, all learned predictors in Table 8 operate under strict substrate and training-set parity: every model receives all $N-1$ training graphs, message-passing depth is fixed at three layers, and checkpoint selection follows the same rule — a validation split within the primary training graph (§6.3) — for every variant alike. Under these controlled conditions, homogeneous GAT-N reaches $\rho = 0.493$, so the observed heterogeneous advantage ($\Delta\rho = +0.114$, won in 11 of 12 folds, $p = 0.0122$) is attributable to relational typing rather than to substrate, training set, depth or selection rule.

### 7.2.1 Where Typed Learning Fails, and How It Can Be Detected

HGT-QoS loses to Topo-QoS on three folds, but only two of them are substantive: Enterprise ($\rho = 0.569$ vs. $0.838$) and Microservices ($0.483$ vs. $0.563$). The third, Real-Time Gaming ($0.776$ vs. $0.780$), is a tie to within $0.004$ and carries no interpretive weight. It is the two substantive inversions that hold the RQ1 comparison below significance, and both are folds on which the training-free baseline is unusually strong, which suggests the model is discarding structural signal the baseline retains rather than failing to learn.

#### A candidate label-free signature, and why we do not claim it

Let $\hat{\sigma}$ denote the standard deviation of the model’s own predicted scores over the held-out Application population — a quantity computable at inference time, without labels. The intuition is that a model whose output collapses toward a constant on an unfamiliar architecture is signalling its own unreliability. Two observations are consistent with it: Enterprise carries the lowest $\hat{\sigma}$ in the corpus ($0.111$ against a median of $0.173$) and Microservices the third lowest ($0.138$), and those are the two folds HGT-QoS loses.

The correlation does not survive inspection. Across the twelve folds, $\hat{\sigma}$ and the margin over Topo-QoS correlate at $\rho_s = -0.126$ ($p = 0.697$) for HGT-QoS — the wrong sign — and $+0.245$ ($p = 0.443$) for HGT. The only fold set on which the relationship is significant belongs to the untyped baseline (GAT-N, $\rho_s = +0.706$, $p = 0.010$), which is not the model we propose. The counterexample is Healthcare: it carries the second-lowest $\hat{\sigma}$ in the corpus ($0.123$) and one of the largest positive margins ($+0.284$). A low-dispersion prediction is therefore not a reliable warning; on this corpus it is associated with the two worst folds and one of the best.

We report this negative result rather than the two-fold coincidence that motivated it. A label-free confidence signal for pre-deployment gating would be a genuinely valuable property, and we had one in an earlier analysis, but it does not replicate on the present corpus and we do not build on it. The zero-shot dispersion figures of §7.4.1 must be read against this: whatever $\hat{\sigma}$ tracks across five real systems, it does not track fold difficulty across twelve synthetic ones.

Graph size does not explain the failure either. The rank correlation between evaluated fold size and margin is $-0.357$ ($p = 0.255$) — weakly negative, but not significant and not consistent in sign across the fold set: the three largest wins are the smallest graph in the corpus (ATM, 26 Applications, $+0.665$), the second largest (IoT Smart City, 200, $+0.332$) and the second smallest (Healthcare, 50, $+0.284$). Enterprise is the largest fold and one of the two losses, but Microservices is mid-sized, so size alone orders neither the wins nor the losses.

Feature *scale* is a different matter from graph size, and the cross-scenario scale drift documented in `results/feature_shift_diagnostic.md` remains the most plausible account of the Enterprise deficit: it is the corpus’s largest graph, and within-graph feature rank normalisation is the transform whose effect should be largest there. We offer this as an explanation consistent with the evidence rather than a tested one: the configuration reported in Table 8 is fixed across folds by design (§6.3), so this study does not isolate the contribution of that transform on the Enterprise fold.

What we cannot offer is a way to tell in advance which architectures these are. The $\hat{\sigma}$ signature above does not provide it, and no other label-free indicator we examined — graph size, edge density, entity-type mix — orders the folds either. A practitioner applying the model to an unseen architecture therefore has no reliable warning that it is one of the two on which a training-free baseline would have served better. We regard this, rather than the mean correlation, as the most consequential open limitation of the predictive pathway for deployment as a gate (§8.4).

## 7.3 RQ3: Ablations and Sensitivity Analysis

This section reports the ablations that bear on a headline claim — the QoS edge encoding, cross-oracle agreement, and the per-type stratification that governs how every other result in this paper is read. The parameter-sensitivity sweeps over the explanation layer’s ten declared weight constants establish robustness rather than any finding of their own, and are reported in full in the supplementary material (Supplementary §§S1–S2). Their collective result is stated here so the body remains self-contained: of the ten constants, only the Fault-Tolerance/Availability blend $r_\alpha$ and the AHP shrinkage $\lambda$ carry appreciable influence on $\rho$ ($\mu^* = 0.144$ and $0.117$ under Morris screening, against $\le 0.023$ for the remaining eight), and no setting of the topic-weight or QoS sub-weight constants would change any comparison reported above. The elicited AHP weights are, notably, *anti*-predictive: rank correlation falls monotonically from $0.262$ under a uniform prior to $0.166$ under raw AHP judgement. We retain them because RM is an attribution instrument rather than a ranking model, and discuss that trade in §8.4.

### 7.3.1 QoS Feature Ablation

To isolate the specific empirical contribution of the continuous-categorical QoS edge features (§4.1.1), we evaluated **HGT**, an un-augmented ablation of HGT-QoS whose edge features contain only scalar coupling and relation one-hot encodings.

Under the inductive LOSO evaluation, the QoS edge encodings carry a ranking benefit in both architectures on the full held-out population. Mean LOSO rank correlation is $\rho = 0.695$ with the encodings and $0.640$ without them ($\Delta\rho = +0.054$, won in 11 of 12 folds, $W = 7.0$, $p = 0.0093$; 95% bootstrap CI $[+0.021, +0.091]$). The homogeneous architecture agrees in direction and magnitude: GAT-N-QoS achieves $0.581$ against GAT-N $0.493$ ($\Delta\rho = +0.088$, won in 9 of 12 folds, $W = 12.0$, $p = 0.034$, CI $[+0.021, +0.151]$). The typed result is robust to dropping the ATM fold ($+0.049$, 10 of 11, $p = 0.019$); the homogeneous one is not ($+0.071$, 8 of 11, $p = 0.067$), so we treat the typed evidence as the finding and the homogeneous replication as supporting.

#### The gain does not survive restriction to the active stratum, and we do not claim it does

Re-scoring the identical folds on the $n_{>0}$ components that carry strictly positive impact (§7.1.2) removes the effect in both architectures: the typed gain falls from $+0.054$ ($p = 0.0093$) to $+0.025$ (8 of 12, $W = 20.0$, $p = 0.151$), and the homogeneous gain from $+0.088$ ($p = 0.034$) to $+0.017$ (6 of 12, $W = 35.0$, $p = 0.791$) — the latter no better than a coin flip across folds. The full-population effect is therefore real but narrower than it first appears: declared QoS attributes help the model decide *which* components can propagate a failure at all, and this evaluation provides no evidence that they help order the components that do. That is a weaker claim than the one an earlier version of this section made, and it is the claim the data support. It matters for deployment, because a practitioner reading Table 8 would otherwise expect the QoS channel to sharpen a criticality ranking, which is the use it cannot be shown to serve. The typed-versus-untyped result, by contrast, does survive the same restriction (§7.1.2), which is why we treat typing rather than QoS encoding as this paper’s load-bearing architectural claim.

The encodings also improve optimisation reproducibility, though less symmetrically than the ranking result. The median within-fold standard deviation across five random seeds is $0.053$ for HGT-QoS against $0.069$ for un-augmented HGT, and $0.017$ for GAT-N-QoS against $0.158$ for GAT-N — a large effect in the homogeneous pair and a modest one in the typed pair. We read the ranking gain as the primary result and the variance reduction as a secondary property, rather than the reverse.

#### A note on what makes this ablation informative

An earlier version of the generator emitted a near-constant QoS profile in most scenarios, which left the QoS channel with almost no variance to encode and produced a null result here. The corpus reported in this paper does not have that property: modal QoS shares range from 29% to 89% across the twelve scenarios (Supplementary §S5), so every fold carries genuine variation in declared reliability, durability and priority. The ablation is meaningful only under that condition, and we state it because a QoS ablation run on a QoS-degenerate corpus measures nothing. As noted in §4.1.1, three schema dimensions (`has_deadline`, `deadline_ns_log`, `max_blocking_ms_log`) remain zero throughout the corpus and are reserved extension points; the gains reported here come from the four active dimensions alone.

### Convergent Validity Over Simulation Oracles

We evaluated inter-oracle agreement across $I^*(v)$ (`FaultInjector`), $I_{\text{comp}}(v)$ (`FailureSimulator`), and $I_{\text{dyn}}(v)$ (`MessageFlowSimulator`) over seven scenarios, summarized in Table 10:

**Table 10.** Inter-oracle agreement across simulation paradigms (chance top-$K$ Jaccard is $0.111$) over 7 diverse benchmark topologies. $I_{\text{dyn}}$ denotes the queue-flow discrete-event simulation oracle, $I^*$ denotes the graph topological cascade injection oracle, and $I_{\text{comp}}$ denotes the multi-criteria composite oracle. The negative lower bounds on $I_{\text{comp}}$ reflect a single scenario (`hub_and_spoke`) driven by tied zero-inflation rather than directional inversion (Supplementary §S2).

| **Oracle pair**                        |      **Mean $\rho$ (range)**       | **Mean $\tau$** | **Jaccard@$K$** | **Tie-robust** |
|:---------------------------------------|:----------------------------------:|:---------------:|:---------------:|:--------------:|
| $I_{\text{dyn}}$ vs. $I^*$             | $\mathbf{0.907}$ ($0.748$–$0.985$) |     $0.788$     |     $0.486$     |    $0.509$     |
| $I_{\text{comp}}$ vs. $I^*$            |     $0.425$ ($-0.044$–$0.654$)     |     $0.311$     |     $0.240$     |    $0.258$     |
| $I_{\text{comp}}$ vs. $I_{\text{dyn}}$ |     $0.427$ ($-0.037$–$0.654$)     |     $0.312$     |     $0.276$     |    $0.275$     |

Ordering converges strongly between the queue-flow simulator and topological cascade oracle ($\rho = 0.907$). Top-$K$ set agreement is more conservative (Jaccard $0.486$), bounded by discrete threshold sensitivity in non-linear cascades.

### 7.3.3 Node-Type Stratification

One result from the detection benchmark governs how every other number in this paper is read. Measured against $I_{\text{comp}}(v)$ over the eight scenarios of that benchmark, stratified RM rank correlations are $\rho = 0.515$ (Application), $0.183$ (Broker) and $0.149$ (Node), while pooling all types collapses the correlation to $\rho = 0.057$ — below every per-type value it aggregates, which is Simpson’s paradox in its textbook form. This is why every evaluation in this paper is reported on a single stratum, and why pooled critical-set figures should be read as inflated wherever they appear (§7.4). The rule-based anti-pattern catalog evaluated on the same benchmark, its behaviour under scaling, and the comparison against degree centrality are reported in Supplementary §S6; its summary is that the catalog flags $94.3\%$ of scored components and therefore does not discriminate, so critical-set identification is delegated to the continuous rankers of §§7.1–7.2.

### HGT Attention Weight Analysis

![Figure 4](latex/figures/Figure_3.png)

*Figure 4. Relational attention from the trained HGT on the ATM case study, one seed. Mean α spans only 0.15–0.23 across relation types and is driven substantially by in-degree. A qualitative illustration on one topology — not evidence of a relation ordering, nor of why typing helps.*

Aggregated by relation type over the ATM case study, first-layer mean attention orders `USES` into libraries ($0.227$, $0.215$) above `ROUTES` ($0.194$), `SUBSCRIBES_TO` ($0.176$), `PUBLISHES_TO` ($0.163$) and `RUNS_ON` ($0.153$), with the same ordering to within $0.02$ in layers two and three. Two cautions bound what this supports. The spread across all eight relation types is narrow ($0.15$–$0.23$), so this is a tendency, not a separation; and because $\alpha$ is a softmax over each destination’s incoming edges, mean $\alpha$ is driven substantially by in-degree — the top-ranked `Library`$\to$`Library` relation carries 4 edges, and the largest weight in the graph ($\alpha_{uv} = 1.00$) sits at a destination of in-degree one, where $\alpha = 1$ holds by construction. Figure 4 is therefore a qualitative illustration on one topology at one seed without a significance test, not evidence for why typing helps.

## 7.4 RQ4: Real-World Distributed Architecture Validation

We evaluated the framework on five open-source distributed systems transcribed from their public repositories by dedicated architectural adapters. Online Boutique is a vendor demonstration application and Train-Ticket an academic benchmark; Home Assistant is an authentic open-source IoT automation platform and EdgeX Foundry an industrial edge computing reference implementation. Each is an abstraction of its upstream repository rather than a complete transcription, and all five carry labels from the same simulation oracles used throughout — so what follows tests topological generalization, not agreement with observed field failures.

Two separate evaluations run on these systems and they are not commensurable. The deterministic explanation layer $Q(v)$, scored against $I_{\text{comp}}(v)$, attains $\rho = 0.800$ (EdgeX), $0.778$ (Cloud Microservices), $0.759$ (Train-Ticket), $0.685$ (Autoware) and $0.514$ (Home Assistant), with a margin over unweighted degree centrality on all five. Because $Q(v)$ is a closed-form function rather than a fitted model, this involves no transfer in the inductive sense of §7.2: nothing was trained and no distribution shift is being survived, and the figures are not comparable with Table 8’s RM column ($\rho = 0.133$), which is scored against a different oracle. That evaluation, its stratified-versus-pooled critical-set analysis, and the framework’s five-condition acceptance gate — which passes on one of the five architectures — are reported in full in Supplementary §S7. Its two conclusions that bear on the rest of this paper are that pooled top-$K$ figures are inflated by inert infrastructure and should not be read (§7.3.3), and that absolute gating thresholds calibrated on the synthetic corpus do not transfer.

One gap in that evaluation must be recorded here because it bounds a headline comparison. `Topo-QoS`, the strongest training-free baseline on the synthetic corpus, cannot be scored on these systems at all. The obstacle is not missing QoS data — every topic in all five adapters carries declared `durability`, `reliability` and `transport_priority` profiles, and projecting $w(t)$ onto the structural edges yields non-unit weights on $45.3$–$62.2\%$ of them. It is that `Topo` reads betweenness off the cached `DEPENDS_ON` projection, whereas the QoS-weighted variant must recompute it on the raw multigraph, where Applications never route messages and betweenness is identically zero for all of them — the same degeneracy §6.2.1 gives as the reason topological baselines run on the projection at all. We record the gap rather than substitute a number for it; the consequence, that the real-world margin below is measured against the weaker training-free reference, is stated in §8.4.

### 7.4.1 Zero-Shot Transfer of the Learned Model

The typing result of §7.2 was established on scenarios from our parameterized generator. To test whether graph learning transfers zero-shot to architectures we did not author, we trained HGT-QoS on all twelve synthetic scenarios and evaluated it on the five open-source systems against $I^*(v)$ (`FaultInjector`) across five seeds ($\{42, 123, 456, 789, 2024\}$).

#### Configuration, stated in full

This evaluation does not use the configuration of Table 8, and the difference must be on the record before the numbers are read. It employs within-graph rank normalization of node features and labels, a depth of two message-passing layers rather than three, and a budget of 150 epochs rather than 300. The motivation is the cross-scenario feature-scale drift documented in `results/feature_shift_diagnostic.md`: the five transcribed architectures differ in scale from the generated corpus more sharply than the generated scenarios differ from each other, and unnormalised features transfer poorly across that gap.

That motivation is a reason to expect the transform to help; it is not evidence that the specific depth and epoch budget were fixed independently of the results they produce. We therefore make no zero-shot-purity claim for the numbers in Table 11 beyond the literal one: no real-world system contributed a training gradient, and none was used for early stopping or checkpoint selection. Whether the configuration itself was chosen with knowledge of these five systems is a question the released harness answers and the reader should check; we flag it rather than let the phrase “zero-shot” carry more weight than it can bear. Table 11 presents the comparison against the training-free references scored on the identical population and oracle.

**Table 11.** Zero-shot transfer to five open-source systems, scored against $I^*(v)$ on the Application population. HGT-QoS trains on all twelve synthetic scenarios ($\pm$ = spread over five seeds); RM and Topo are training-free, scored on identical labels and nodes. **$\rho_{>0}$ restricts the correlation to the $n_{>0}$ components that actually propagate a failure and is the column to read for ranking quality**; full-population $\rho$ conflates that with separating active from inert. `Topo-QoS` is absent (§8.4).

| **Real-World Architecture**        | **$|V_{\text{app}}|$** | **RM $\rho$** | **Topo $\rho$**  |         **HGT-QoS $\rho$**          | **HGT-QoS $\rho_{>0}$** | **$n_{>0}$** | **$F_1@K$** |
|:-----------------------------------|-----------------------:|:-------------:|:----------------:|:-----------------------------------:|:-----------------------:|:------------:|:-----------:|
| **Cloud Microservices Mesh**       |                     22 |    $0.777$    | $\mathbf{0.891}$ |     $0.492$ $\pm$0.199      |    $\mathbf{-0.314}$    |      18      |   $0.200$   |
| **Train-Ticket Booking Mesh**      |                     41 |    $0.713$    |     $0.528$      | $\mathbf{0.702}$ $\pm$0.071 |    $\mathbf{-0.244}$    |      22      |   $0.375$   |
| **Autoware.universe (ROS 2)**      |                     32 |    $0.357$    |     $0.307$      | $\mathbf{0.715}$ $\pm$0.067 |        $+0.549$         |      28      |   $0.567$   |
| **EdgeX Foundry (Industrial IoT)** |                     22 |    $0.470$    |     $0.534$      | $\mathbf{0.688}$ $\pm$0.086 |        $+0.247$         |      19      |   $0.500$   |
| **Home Assistant (Smart Home)**    |                     24 |    $0.265$    |     $0.297$      | $\mathbf{0.803}$ $\pm$0.032 |        $+0.564$         |      23      |   $0.480$   |
| **Mean**                           |                      — |    $0.516$    |     $0.511$      |          $\mathbf{0.680}$           |        $+0.160$         |      —       |   $0.424$   |

**Key Insights for Real-World Transfer:**

1.  **The full-population figure is not a ranking result.** On the whole Application population HGT-QoS reaches $\rho = 0.680$ against $0.511$ for Topo and $0.516$ for RM, leading on four of five systems. Read alone, that looks like successful zero-shot transfer. It is not, because between $4\%$ (Home Assistant) and $46\%$ (Train-Ticket) of Applications in these architectures carry exactly zero simulated impact, and a correlation computed over a population that is heavily tied at zero rewards separating the inert from the active at least as much as ordering the active correctly.

2.  **Restricted to components that actually propagate failures, transfer is not established.** On the active stratum the mean falls from $0.680$ to $+0.160$, and two of the five systems invert: Cloud Microservices to $\rho_{>0} = -0.314$ ($n = 18$) and Train-Ticket to $-0.244$ ($n = 22$). Both are the microservice call-tree architectures. The three pub-sub systems hold up ($+0.549$ Autoware, $+0.564$ Home Assistant, $+0.247$ EdgeX), so the pattern is architectural rather than random — but a method that anti-correlates with the truth on two of five held-out systems has not demonstrated zero-shot generalisation, and we do not claim it. **We therefore report RQ4 as a negative result:** learned relational transfer to authentic open-source architectures is not established by this evidence.

3.  **What the full-population number does support.** Separating components that propagate failures from those that do not is itself the operationally useful half of the task — a gate that correctly identifies which two thirds of a system cannot cause a cascade has narrowed the review surface, even if it orders the remainder poorly. The $F_1@K$ column ($0.424$ mean) is the honest expression of that capability, and it is what a practitioner would act on. We separate the two claims rather than let the first stand in for the second.

4.  **Why the microservice architectures invert.** Both are deep synchronous RPC call trees rather than pub-sub meshes: a large fraction of services are terminal sinks whose failure reaches nobody, and among the minority that do propagate, impact is governed by position in a call hierarchy that the model, trained entirely on generated pub-sub topologies, has never seen. Topo captures the upstream bottleneck structure directly and leads on Cloud Microservices ($\rho = 0.891$). This is a domain-shift limit of the training corpus, not a defect of typed message passing, and it marks the boundary of what our synthetic corpus can prepare a model for.

The $\hat{\sigma}$ dispersion values that an earlier version of this analysis used to flag the Cloud Microservices failure are not reported here as a diagnostic: the same quantity fails to track fold difficulty on the twelve-fold synthetic corpus, where it carries the wrong sign for this model (§7.2.1). A correlation of $\rho_s = +0.600$ over five systems — which cannot reach significance at $n = 5$ — is not sufficient to reinstate it.

## 7.5 RQ5: Analysis Cost and Its Comparison Against Simulation

RQ5 quantifies computational overhead and sustainability during CI/CD evaluation, with per-stage latencies reported in Table 12:

**Table 12.** Per-stage latency of the inference pipeline across scaling graph sizes (CPU, median of 3 runs).

| **$|V|$** | **$|E|$** | **Analyse (s)** | **Graph$\to$tensor (s)** | **HGT forward (ms)** | **Analyse : forward** |
|----------:|----------:|----------------:|-------------------------:|---------------------:|----------------------:|
|       249 |     1,127 |            1.74 |                    0.010 |                 26.5 |            66$\times$ |
|       499 |     2,402 |            8.32 |                    0.022 |                 16.4 |           509$\times$ |
|       999 |     6,422 |           44.54 |                    0.056 |                 21.1 |         2,108$\times$ |
|     1,998 |    19,301 |          239.34 |                    0.157 |                 56.2 |     **4,259$\times$** |

**The neural model is the cheapest stage, and the deterministic one is expensive.** At 2,000 components the HGT forward pass takes $56\,\text{ms}$ against $239\,\text{s}$ for deterministic structural analysis — a ratio of $4{,}259\times$. That ratio locates cost *inside* the pipeline; it is not the cost of evaluating an architecture. Indices 0–17 of every node feature vector (§3.4) — betweenness, closeness, reverse PageRank, articulation and bridge scores — are products of that same analysis stage, so the forward pass cannot run without it. End-to-end evaluation of an unseen 2,000-component architecture costs about four minutes, of which the learned model is $0.02\%$; the $56\,\text{ms}$ figure is the marginal cost of re-scoring an already-analysed graph. Across the eight-scenario detection benchmark the complete gate (structural analysis plus 18 anti-pattern detectors) runs in $0.04$–$82.7\,\text{s}$, the upper bound being the 520-component Enterprise mesh.

**Cost is dominated by one metric, and it grew.** Measured cost now tracks the stage’s $O(|V|^2 + |V||E|)$ bound closely: from 249 to 1,998 components wall-clock rises $138\times$ against a $137\times$ growth in $|V||E|$. The dominant term is the Connectivity Degradation Index, which is computed for every node in the main connected component rather than for articulation points alone. That choice is deliberate and is a correctness requirement rather than an oversight: gating CDI to articulation points leaves it identically zero for every node whose removal does not literally disconnect the graph, which drives $A(v)$ to a near-constant in the redundant multi-publisher topologies this framework targets. The cost is the price of a non-degenerate Availability score, and we report it rather than the cheaper gated variant we could have measured.

### 7.5.1 The Gate Is Not Cheaper Than the Simulation It Replaces

The framing that motivated this analysis — static gating as a low-cost substitute for dynamic simulation — does not survive measurement against our own oracle. Timing the `FaultInjector` labelling sweep (five seeds, node types Application/Broker/Library, the full ground-truth run) on the same corpus and the same idle hardware gives $0.14$–$7.2\,\text{s}$ per scenario, against $0.04$–$82.7\,\text{s}$ for the analysis gate. Both maxima belong to the 520-component Enterprise mesh, so the largest scenario compares $7.2\,\text{s}$ of simulation against $82.7\,\text{s}$ of static analysis: **the gate costs roughly eleven times more than the simulation it is meant to displace.**

Two things follow, and we state both. First, no claim in this paper that static analysis is computationally cheaper than simulation is supported, and we withdraw it. Our cascade oracle is an in-process breadth-first traversal, and traversing a graph is simply cheaper than computing per-node connectivity degradation over it. Second, the argument that survives is narrower and concerns *what* the two require rather than what they cost: the oracle needs a topology with declared failure semantics and produces labels for components it can express, whereas chaos engineering and hardware-in-the-loop validation — the practices a pre-deployment gate would actually displace in industry — need provisioned clusters, carry operational risk, and run at cluster-hour scale. That comparison is plausible and is the one made in §2.1, but we have not measured it, and it should not be read as established here.

The quantity measured throughout is wall-clock time on a single CPU core. We deliberately do not convert it into an energy or carbon figure (§8.2). What the measurements support is the narrow claim that pre-deployment analysis of a few-hundred-component architecture fits inside a pull-request budget without provisioning any runtime infrastructure. They do not support a claim of general computational efficiency, and at the scales invoked in §1.1 — an order of magnitude beyond what we measured, against an $O(|V|^2 + |V||E|)$ stage — the present implementation would not fit that budget at all.

# 8. Discussion, Threats to Validity, and Limitations

## 8.1 Discussion and Practical Implications

#### When to use Topo-QoS, and when to use HGT-QoS

The results do not support a simple recommendation of the learned model over the closed-form one, and we set out the trade-off as measured rather than as hoped.

1.  **Training-free ranking (`Topo-QoS`).** It requires no training, no checkpoint storage and no retraining as a corpus evolves, reaching $\rho = 0.568$ zero-shot across twelve unseen synthetic architectures and $\approx 0.51$ across five open-source systems. Nothing in this study establishes that a learned model beats it on ranking: the margin is $+0.127$ ($p = 0.077$) and collapses to $+0.078$ when one fold is removed (§7.1). For a team that wants a criticality ordering and nothing more, this is the defensible default, and we say so despite proposing the alternative.

2.  **Learned relational prediction (`HGT-QoS`).** Its established advantage is over *untyped* learning under distribution shift ($+0.114$, 11 of 12 folds), not over closed-form centrality. It also leads on critical-set identification, where it does separate from the training-free baseline ($F_1@K$ $+0.154$, $p = 0.034$). Two further capabilities have no closed-form counterpart, and they are the substantive reason to prefer it where the extra machinery is affordable:

    -   *Typed relational attention* exposes *which* channels mediate a cascade, rather than only which components rank highly. Figure 4 illustrates this on one topology and is explicitly not evidence of a general effect (§7.3).

    -   *Relationship-level criticality* ($I_{\text{edge}}$, Eq. 13) scores individual dependencies rather than components, which is what circuit-breaker or bulkhead placement actually requires, and which a node ranking cannot express. We report this as a property of the formulation rather than an evaluated result: this paper defines the edge oracle and the model’s edge head but presents no evaluation of edge-level predictions against it, so the capability is available and untested.

    -   *QoS-conditioned ranking.* The 16-D edge encoding improves out-of-distribution ranking by $+0.054$ within the typed architecture (11 of 12 folds, $p = 0.0093$) and $+0.088$ within the untyped one, so declared middleware contracts carry signal a purely structural score discards.

#### A gate we can no longer recommend

An earlier version of this work proposed a tiered gate: run the learned model by default, and fall back to `Topo-QoS` when the model’s own prediction dispersion $\hat{\sigma}$ fell below a threshold, on the evidence that $\hat{\sigma}$ tracked transfer quality. That mechanism does not survive the present corpus. Across twelve folds, $\hat{\sigma}$ correlates with the margin over `Topo-QoS` at $\rho_s = -0.126$ for HGT-QoS — the wrong sign — and the relationship is significant only for the untyped baseline (§7.2.1). We withdraw the recommendation rather than restate it with softer wording, and we report the withdrawal because a label-free confidence signal is exactly the kind of claim a deployment would act on.

What remains is a weaker but honest deployment position. Both engines are cheap enough to run together (§7.5), the ranking they produce agrees on most architectures, and the folds where they disagree are the ones a practitioner most needs warning about — with no reliable way, on present evidence, to tell which those are in advance. Running both and escalating disagreement to human review is the procedure our results support; automatic arbitration between them is not.

#### Role of the Explanation Layer

The RM attribution profile ($Q(v)$, §5) provides transparent, standards-compliant architectural diagnostics aligned with ISO/IEC 25010. By separating single-point-of-failure exposure (Availability) from wide error propagation reach (Fault Tolerance), RM provides qualitative remediation guidance (e.g., distinguishing whether a component requires replication or decoupling) that purely numeric rankers and simulation oracles cannot provide.

## 8.2 Performance and Computational Sustainability Implications

#### What sustainability means for a pre-deployment gate

Green software engineering distinguishes the energy a system consumes from the energy its *development and assurance* consume [25], and the machine-learning literature has concentrated on the second: the cost of training and serving models [87, 88, 89], the frameworks that mediate it [90], and how to account for it at all [26, 27]. Reliability assurance sits in the same category and is rarely measured: chaos engineering, hardware-in-the-loop benches and staging-cluster fault injection consume cluster-hours per sweep, and they do so on every pull request that triggers them. Two findings in this study bear on that, and we state them at the strength the measurements support.

The first is about the learned component, and it is unambiguous. *Within* the pipeline the neural model is negligible: the HGT forward pass is $56\,\text{ms}$ on a 2,000-component system against $239\,\text{s}$ for deterministic structural feature extraction, a ratio of $4{,}259\times$ (§7.5). Whatever pre-deployment dependability analysis costs, adding a graph neural network is not what makes it expensive. For a special issue asking whether AI techniques can be afforded in a sustainability-conscious engineering process, that is the directly relevant result: here, the AI is the cheap part by three orders of magnitude.

The second is about what the framework removes, and it is a difference in kind rather than in degree. The resource a manifest-time analysis eliminates is *infrastructure*, not CPU seconds. Chaos engineering and fault injection on a staging cluster require a provisioned, running system; SaG requires a file. Nothing must be deployed, kept warm, or torn down, and no fault is injected into anything a user could be holding. That is the honest form of the sustainability argument — but we have not measured a chaos-engineering baseline, so its magnitude is an assertion rather than a result.

#### The efficiency claim we withdraw

An earlier version of this work made a stronger claim: that static graph-based gating delivers an orders-of-magnitude reduction in execution cost relative to dynamic simulation. Measurement contradicts it. On our own corpus and hardware a five-seed `FaultInjector` sweep costs $0.14$–$7.2\,\text{s}$ per scenario while the static gate costs $0.04$–$82.7\,\text{s}$; on the largest architecture the gate is roughly eleven times the more expensive of the two (§7.5.1). We withdraw the claim rather than qualify it. It is worth being precise about what the comparison shows: our oracle is an in-process breadth-first traversal, not a deployed cluster, so this measures static analysis against *simulation*, not against the chaos engineering a gate would displace in industry. Both readings are in §7.5.1; neither supports a general efficiency claim.

#### Where the cost actually sits, and why we paid it

The expense is concentrated in one metric and is a deliberate accuracy purchase. The Connectivity Degradation Index dominates because it is computed for every node in the main component rather than for articulation points alone; gating it to articulation points would restore roughly an order of magnitude of speed at the price of a degenerate Availability score (§7.5). A deployment that needed the speed more than the single-point-of-failure sensitivity could make the opposite trade, and we would rather document that choice than hide it behind a faster number. This also identifies where an optimisation effort should go: the sustainability of this framework is a question about one graph-theoretic routine, not about its use of machine learning.

Settling any of this properly requires energy rather than wall-clock time — running-average power counters and cluster-level accounting of the kind the green-computing literature has standardised [26] — which is the measurement this study lacks and §8.4 proposes.

#### A note on avoided runtime failure

It is tempting to argue that preventing cascading failures saves the datacenter compute that retry storms and restart loops would have consumed. We find the argument plausible and have no evidence for it: nothing in this study observes a production incident, an averted one, or the compute either would involve. We flag it as motivation rather than contribution.

## 8.3 Threats to Validity

#### Construct Validity

Our primary ground-truth impact oracle $I^*(v)$ is derived from discrete-event cascade simulation on structural models rather than observing live production outages. To evaluate construct divergence, we compared $I^*(v)$ against the queue-flow discrete-event simulator $I_{\text{dyn}}(v)$ and the multi-criteria composite oracle $I_{\text{comp}}(v)$ across diverse topologies (Table 10). The rank correlation between $I^*$ and $I_{\text{dyn}}$ is high ($\rho = 0.907$), but we are careful about how much it establishes: both oracles traverse the same structural graph under the same failure semantics, differing in how they score the consequences rather than in what they model, so their agreement is close to a consistency check and is weak evidence of construct validity. The more informative comparison is with $I_{\text{comp}}$, which differs in construct — and it agrees with $I^*$ at only $\rho = 0.425$. That is the figure that should temper confidence here, particularly because $I_{\text{comp}}$ supplies the labels for the explanation layers real-world evaluation (Supplementary §S7). Discrepancies in top-$K$ critical-set Jaccard ($0.24$–$0.49$) stem from discrete thresholding sensitivity in non-linear cascades and tied zero-inflation, which we analyse explicitly in Supplementary §S2. Convergent validity across simulation paradigms is therefore partially supported at the level of ordering and not established at the level of critical-set membership; no oracle in this study is validated against an observed production failure (§4.4). For Maintainability, $I_M(v)$ traverses the same derived dependency topology from which $M(v)$ is scored; independent validation against repository change histories or version-control churn remains future work.

#### Internal Validity

Potential feature leakage is prevented by strict graph view separation: predictors operate exclusively on $G_{\text{analysis}}$, whereas ground-truth simulation oracles operate on $G_{\text{structural}}$, formally asserted in continuous integration. Substrate parity is rigorously maintained: learned models (HGT-QoS, GAT-N-QoS) share identical training sets, depths, and held-out scenario early-stopping protocols. In the QoS schema, four dimensions (reliability, durability, transport priority, heterogeneity flag) capture active operational middleware configurations, while three dimensions (deadline, max blocking) are reserved extension points that remain zero in standard static descriptors.

#### External Validity

Our evaluation spans twelve synthetic architectures and five authentic open-source distributed systems, and the boundary it establishes is sharper than the coverage suggests. Learned transfer to the real systems is *not* demonstrated: restricted to components that actually propagate failures, mean rank correlation falls to $+0.160$ and inverts on the two microservice call-tree architectures (§7.4.1). Every synthetic scenario in the training corpus is a pub-sub mesh emitted by one parameterized generator, so a model trained on it has never seen a deep synchronous RPC hierarchy, and the two systems it fails on are exactly those. Corpus diversity, not corpus size, is the limiting factor, and no amount of additional generated pub-sub topologies would address it.

A second limit concerns QoS coverage. An earlier version of the generator emitted near-constant QoS profiles, which would have made the QoS ablation of §7.3.1 uninformative; the corpus reported here carries genuine variation (modal shares $29$–$89\%$), but all of it is generator-produced, and we have no evidence about the distribution of QoS declarations in real deployments. Finally, the largest system we evaluate has 520 components and the largest we time has 2,000; the architectures motivating this work in §1.1 are an order of magnitude larger, and the dominant pipeline stage is $O(|V|^2 + |V||E|)$, so neither the accuracy nor the cost results should be extrapolated to that scale.

#### Conclusion Validity

Given heavy-tailed impact distributions, statistical analyses use non-parametric rank correlation (Spearman $\rho$, Kendall $\tau$), bootstrap confidence intervals ($B = 2{,}000$) and paired Wilcoxon signed-rank tests, with the fold or scenario as the unit of analysis. Two hazards recur and shape how we report. Pooling across heterogeneous entity types triggers Simpson’s paradox — pooled $\rho = 0.057$ sits below every per-type value it aggregates ($0.149$–$0.515$) — so all headline figures are stratified on a single population. And rank correlation over zero-inflated labels conflates ordering the active components with separating them from inert ones, which is why we report zero-excluded correlations alongside full-population ones wherever the label distribution permits. Where the two disagree, as on the real-world systems, we read the zero-excluded figure as the ranking result.

## 8.4 Limitations and Future Work

#### An incomplete baseline on the real-world systems

`Topo-QoS` — the strongest training-free baseline on the synthetic corpus, and the one whose competitiveness bounds our central claim — is not scored in Table 11. It requires QoS-weighted betweenness recomputed on the projection graph, which the real-world evaluation cache does not carry, and we chose to leave the cell empty rather than publish an unreproducible figure beside reproducible ones. The consequence is that the real-world comparison is made against the weaker `Topo`, and the margin reported there should be read accordingly. Closing this gap requires only extending the cache, and it is the first thing we would add.

#### The explanation layer is not validated as an explanation

SaG’s attribution layer separates Availability from Fault Tolerance on the claim that they imply different repairs. Nothing in this paper tests that claim. We do not show that the two sub-scores rank components differently in practice, that practitioners find the distinction actionable, or that applying a recommended repair reduces measured failure impact. The elicited AHP weights, meanwhile, are measurably worse than a uniform prior at ranking (§7.3), and while we argue that ranking is not what they are for, we have no positive evidence for what they *are* for. Three studies would settle this: a discriminant analysis of $A(v)$ against $FT(v)$ over the corpus, a counterfactual evaluation in which recommended edits are applied and re-simulated, and a practitioner study of whether the diagnoses change what an engineer would do. Until then, the layer’s contribution is a principled mapping, not a validated instrument.

#### Model selection is made on the training distribution

Early stopping and checkpoint selection in every reported LOSO run use a validation split within the primary training graph (§6.3). Under a protocol whose purpose is distribution shift this is the wrong place to select from: it optimises for fit to a distribution the holdout does not come from. Two things bound how much it matters. The rule is identical across variants, so it cannot manufacture the typed-versus-untyped margin of §7.2; and its expected direction is to favour the learned models over the training-free baselines, which is the opposite of the direction in which our headline comparison against `Topo-QoS` fails. The harness implements selection on a held-out training scenario, and re-running the sweep under it — reporting both, rather than substituting one for the other — is the first robustness check we would add.

#### No label-free reliability signal

We previously reported that the model’s prediction dispersion flagged the architectures it ranked poorly, which would have made the predictor safe to deploy behind an automatic fallback. That relationship does not hold on the present corpus (§7.2.1). A practitioner applying the model to an unseen architecture currently has no reliable indication of whether it is one of the cases where a closed-form baseline would serve better. Finding such a signal is, in our view, more valuable to the pre-deployment setting than any further gain in mean correlation.

#### Distributed AI and LLM Serving Topologies

Modern AI infrastructure relies on distributed LLM serving systems (e.g., vLLM, Triton, DeepSpeed) characterized by complex tensor, pipeline, and expert parallelism across GPU clusters, dynamic KV-cache routing, and disaggregated prefill-decode architectures. A straggler or failing GPU in a pipeline-parallel ring induces severe head-of-line blocking and massive GPU idle power dissipation. Extending the SaG multigraph schema to model distributed AI serving topologies (representing GPU worker nodes, tensor communication channels, and KV-cache transfer fabrics) offers a high-impact direction for sustainable AI systems engineering.

#### Empirical Power and Hardware Testbeds

The sustainability argument of §8.2 rests on wall-clock time and on what infrastructure each approach requires, neither of which is an energy measurement. Settling it needs running-average power counters (Intel/AMD RAPL, NVIDIA NVML) and IPMI sensors read on a Kubernetes cluster during live fault injection, so that a static gate and a chaos-engineering sweep can be compared in joules rather than seconds. Hardware-in-the-loop validation on cyber-physical testbeds — a ROS 2 CAN-bus autonomous-driving compute platform, for instance — would extend the same comparison to the domain where our largest transfer gains appear.

#### Automated Refactoring and Self-Healing

Extending SaG from predictive analysis to prescriptive synthesis—automatically generating pull requests that reconfigure QoS policies, insert circuit-breakers, and add redundant broker pathways to eliminate single points of failure.

# 9. Conclusion

This work introduced **Software-as-a-Graph (SaG)**, a pre-deployment Static System Analysis framework that combines a relation-specific Heterogeneous Graph Transformer for failure-impact forecasting with an interpretable ISO/IEC 25010 Reliability–Maintainability attribution layer, both operating on a typed multigraph derived from Architecture-as-Code manifests with no runtime telemetry.

The central empirical finding is about *when* architectural typing pays. Given training data from the same distribution as the test split, typed and untyped message passing are indistinguishable, and the best in-distribution mean belongs to the untyped model. Asked to rank an architecture it has never seen, the typed model wins 11 of 12 inductive folds by $\Delta\rho = +0.114$ ($p = 0.0122$), and the 16-D QoS edge encoding adds a further $+0.054$ ($p = 0.0093$) independently of typing. Relation-specific parameters therefore act as an inductive bias for unfamiliar topologies rather than as additional fitting capacity — which is the regime a pre-deployment gate always operates in, since every architecture it sees is by construction new.

We are equally clear about what is not established. Against an unparameterized QoS-weighted centrality score, learned ranking is not superior: the nominal margin of $+0.127$ ($p = 0.077$) rests almost entirely on a single fold and falls to $+0.078$ without it. On the five open-source systems, full-population correlation of $\rho = 0.680$ drops to $+0.160$ once restricted to components that actually propagate failures, and inverts on both microservice call-tree architectures, so zero-shot transfer to authentic systems is a negative result on this evidence. A label-free confidence signal we previously reported does not replicate. The explanation layer’s elicited AHP weights measurably worsen ranking relative to a uniform prior, and we have no independent evidence that they improve attribution — validating the attribution itself remains its principal open question.

What the work does contribute is a reproducible typed-multigraph formulation of pub-sub architecture, a corpus that regenerates byte-identically from committed configurations, a clean measurement of where relational typing helps and where it does not, and a pre-deployment analysis pipeline whose learned component is the cheapest stage by three orders of magnitude. Whether that pipeline predicts failures that actually occur — rather than failures a simulator produces — is the question we most want answered next, and it requires field data no static corpus can supply.

---

# References

[1] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, W. Woodall, Robot operating
  system 2: Design, architecture, and uses in the wild, Science Robotics 7 (66)
  (2022) eabm6074.

[2] J. Kreps, N. Narkhede, J. Rao, Kafka: A distributed messaging system for log
  processing, in: Proc. 6th Int. Workshop on Networking Meets Databases
  (NetDB), 2011.

[3] Object Management Group, Data distribution service (dds), Tech. Rep.
  formal/2015-04-10, version 1.4, Object Management Group (2015).

[4] OASIS, MQTT version 5.0, OASIS Standard (2019).

[5] N. Dragoni, S. Giallorenzo, A. L. Lafuente, M. Mazzara, F. Montesi,
  R. Mustafin, L. Safina, Microservices: Yesterday, today, and tomorrow, in:
  Present and Ulterior Software Engineering, Springer, 2017, pp. 195--216.

[6] S. Newman, Building Microservices: Designing Fine-Grained Systems, O'Reilly
  Media, 2015.

[7] P. T. Eugster, P. A. Felber, R. Guerraoui, A.-M. Kermarrec, The many faces of
  publish/subscribe, ACM Computing Surveys 35 (2) (2003) 114--131.

[8] A. Avizienis, J.-C. Laprie, B. Randell, C. Landwehr, Basic concepts and
  taxonomy of dependable and secure computing, IEEE Transactions on Dependable
  and Secure Computing 1 (1) (2004) 11--33.

[9] International Organization for Standardization, ISO/IEC 25010:2023 ---
  systems and software engineering --- systems and software quality
  requirements and evaluation (square) --- product quality model, Tech. rep.,
  International Organization for Standardization (2023).

[10] International Organization for Standardization, ISO/IEC 25019:2023 ---
  systems and software engineering --- systems and software quality
  requirements and evaluation (square) --- quality-in-use model, Tech. rep.,
  International Organization for Standardization (2023).

[11] R. Kazman, M. Klein, M. Barbacci, T. Longstaff, H. Lipson, J. Carriere, The
  architecture tradeoff analysis method, in: Proc. 4th IEEE Int. Conf. on
  Engineering of Complex Computer Systems (ICECCS), 1998, pp. 68--78.

[12] L. Bass, P. Clements, R. Kazman, Software Architecture in Practice, 3rd
  Edition, Addison-Wesley, 2012.

[13] W. Cunningham, The WyCash portfolio management system, in: Addendum to the
  Proc. Conf. on Object-Oriented Programming Systems, Languages, and
  Applications (OOPSLA), 1992, pp. 29--30.

[14] J. Garcia, D. Popescu, G. Edwards, N. Medvidovic, Identifying architectural bad
  smells, in: Proc. 13th European Conf. on Software Maintenance and
  Reengineering (CSMR), 2009, pp. 255--258.

[15] SonarSource, Clean as you code, SonarQube documentation (2024).

[16] T. J. McCabe, A complexity measure, IEEE Transactions on Software Engineering
  SE-2 (4) (1976) 308--320.

[17] S. R. Chidamber, C. F. Kemerer, A metrics suite for object oriented design,
  IEEE Transactions on Software Engineering 20 (6) (1994) 476--493.

[18] N. Fenton, J. Bieman, Software Metrics: A Rigorous and Practical Approach, 3rd
  Edition, CRC Press, 2014.

[19] A. Basiri, N. Behnam, R. de Rooij, L. Hochstein, L. Kosewski, J. Reynolds,
  C. Rosenthal, Chaos engineering, IEEE Software 33 (3) (2016) 35--41.

[20] L. C. Freeman, A set of measures of centrality based on betweenness, Sociometry
  40 (1) (1977) 35--41.

[21] S. Brin, L. Page, The anatomy of a large-scale hypertextual web search engine,
  Computer Networks and ISDN Systems 30 (1--7) (1998) 107--117.

[22] U. Brandes, A faster algorithm for betweenness centrality, Journal of
  Mathematical Sociology 25 (2) (2001) 163--177.

[23] M. E. J. Newman, Networks: An Introduction, Oxford University Press, 2010.

[24] I. O. Yigit, F. Buzluca, A graph-based dependency analysis method for
  identifying critical components in distributed publish--subscribe systems,
  in: Proc. IEEE Int. Conf. on Recent Advances in Systems Science and
  Engineering (RASSE), 2025, pp. 1--8.
https://doi.org/10.1109/RASSE64831.2025.11315354
  `doi:10.1109/RASSE64831.2025.11315354`.

[25] C. Calero, M. Piattini (Eds.), Green in Software Engineering, Springer, Cham,
  Switzerland, 2015.
https://doi.org/10.1007/978-3-319-08581-4
  `doi:10.1007/978-3-319-08581-4`.

[26] L. Lannelongue, J. Grealey, M. Inouye, Green algorithms: Quantifying the carbon
  footprint of computation, Advanced Science 8 (12) (2021) 2100707.
https://doi.org/10.1002/advs.202100707
  `doi:10.1002/advs.202100707`.

[27] R. Verdecchia, J. Sallou, L. Cruz, A systematic review of Green AI, WIREs
  Data Mining and Knowledge Discovery 13 (4) (2023) e1507.
https://doi.org/10.1002/widm.1507
  `doi:10.1002/widm.1507`.

[28] R. C. Cheung, A user-oriented software reliability model, IEEE Transactions on
  Software Engineering SE-6 (2) (1980) 118--125.

[29] K. Goseva-Popstojanova, K. S. Trivedi, Architecture-based approach to
  reliability assessment of software systems, Performance Evaluation 45 (2--3)
  (2001) 179--204.

[30] A. Immonen, E. Niemel\"a, Survey of reliability and availability prediction
  methods from the architectural perspective, Software and Systems Modeling
  7 (1) (2008) 49--65.

[31] S. Becker, H. Koziolek, R. Reussner, The Palladio component model for
  model-driven performance prediction, Journal of Systems and Software 82 (1)
  (2009) 3--22.

[32] G. Franks, T. Al-Omari, M. Woodside, O. Das, S. Derisavi, Enhanced modeling and
  solution of layered queueing networks, IEEE Transactions on Software
  Engineering 35 (2) (2009) 148--161.

[33] J. Delange, P. H. Feiler, Architecture fault modeling with the AADL
  error-model annex, in: 2014 40th EUROMICRO Conference on Software Engineering
  and Advanced Applications (SEAA), IEEE, 2014, pp. 361--368.
https://doi.org/10.1109/SEAA.2014.20
  `doi:10.1109/SEAA.2014.20`.

[34] Y. Gan, Y. Zhang, K. Hu, D. Cheng, Y. He, M. Pancholi, C. Delimitrou, Seer:
  Leveraging big data to navigate the complexity of performance debugging in
  cloud microservices, in: Proc. ACM Int. Conf. on Architectural Support for
  Programming Languages and Operating Systems (ASPLOS), 2019.

[35] Y. Gan, M. Liang, S. Dev, D. Lo, C. Delimitrou, Sage: Practical and scalable
  ML-driven performance debugging in microservices, in: Proc. ACM Int. Conf.
  on Architectural Support for Programming Languages and Operating Systems
  (ASPLOS), 2021.

[36] L. Wu, J. Tordsson, E. Elmroth, O. Kao, MicroRCA: Root cause localization of
  performance issues in microservices, in: Proc. IEEE/IFIP Network Operations
  and Management Symposium (NOMS), 2020.

[37] Z. Li, J. Chen, R. Jiao, N. Zhao, Z. Wang, S. Zhang, Y. Wu, L. Jiang, L. Yan,
  Z. Wang, Z. Chen, W. Zhang, X. Nie, K. Sui, D. Pei, Practical root cause
  localization for microservice systems via trace analysis, in: Proc. IEEE/ACM
  Int. Symposium on Quality of Service (IWQoS), 2021.

[38] C. Zhang, X. Peng, C. Sha, K. Zhang, Z. Fu, X. Wu, Q. Lin, D. Zhang,
  DeepTraLog: Trace-log combined microservice anomaly detection through
  graph-based deep learning, in: Proc. IEEE/ACM Int. Conf. on Software
  Engineering (ICSE), 2022.

[39] C. Lee, T. Yang, Z. Chen, Y. Su, Y. Yang, M. R. Lyu, Eadro: An end-to-end
  troubleshooting framework for microservices on multi-source data, in: Proc.
  IEEE/ACM Int. Conf. on Software Engineering (ICSE), 2023.

[40] V. R. Basili, L. C. Briand, W. L. Melo, A validation of object-oriented design
  metrics as quality indicators, IEEE Transactions on Software Engineering
  22 (10) (1996) 751--761.

[41] N. Nagappan, T. Ball, Static analysis tools as early indicators of pre-release
  defect density, in: Proc. 27th Int. Conf. on Software Engineering (ICSE),
  2005, pp. 580--586.

[42] T. Zimmermann, R. Premraj, A. Zeller, Predicting defects for Eclipse, in:
  Proc. 3rd Int. Workshop on Predictor Models in Software Engineering
  (PROMISE), 2007.

[43] T. Menzies, J. Greenwald, A. Frank, Data mining static code attributes to learn
  defect predictors, IEEE Transactions on Software Engineering 33 (1) (2007)
  2--13.

[44] V. Bushong, D. Das, A. Al Maruf, T. Cerny, Using static analysis to address
  microservice architecture reconstruction, in: 2021 36th IEEE/ACM
  International Conference on Automated Software Engineering (ASE), IEEE, 2021.
https://doi.org/10.1109/ASE51524.2021.9678749
  `doi:10.1109/ASE51524.2021.9678749`.

[45] S. Schneider, A. Bakhtin, X. Li, J. Soldani, A. Brogi, T. Cerny,
  R. Scandariato, D. Taibi, Comparison of static analysis architecture recovery
  tools for microservice applications, preprint (2024).
http://arxiv.org/abs/2412.08352 `arXiv:2412.08352`.

[46] J. Garcia, D. Popescu, G. Edwards, N. Medvidovic, Toward a catalogue of
  architectural bad smells, in: Proc. 5th Int. Conf. on the Quality of Software
  Architectures (QoSA), LNCS 5581, 2009, pp. 146--162.

[47] D. Taibi, V. Lenarduzzi, On the definition of microservice bad smells, IEEE
  Software 35 (3) (2018) 56--62.

[48] Z. Li, P. Avgeriou, P. Liang, A systematic mapping study on technical debt and
  its management, Journal of Systems and Software 101 (2015) 193--220.

[49] J. Humble, D. Farley, Continuous Delivery: Reliable Software Releases through
  Build, Test, and Deployment Automation, Addison-Wesley, 2010.

[50] L. Chen, Continuous delivery: Huge benefits, but challenges too, IEEE Software
  32 (2) (2015) 50--54.

[51] International Organization for Standardization, ISO/IEC 25023:2016 ---
  systems and software engineering --- systems and software quality
  requirements and evaluation (square) --- measurement of system and software
  product quality, Tech. rep., International Organization for Standardization
  (2016).

[52] International Organization for Standardization, ISO/IEC 25021:2012 ---
  systems and software engineering --- systems and software quality
  requirements and evaluation (square) --- quality measure elements, Tech.
  rep., International Organization for Standardization (2012).

[53] T. L. Saaty, The Analytic Hierarchy Process: Planning, Priority Setting,
  Resource Allocation, McGraw-Hill, 1980.

[54] R. Albert, H. Jeong, A.-L. Barab\'asi, Error and attack tolerance of complex
  networks, Nature 406 (2000) 378--382.

[55] A. E. Motter, Y.-C. Lai, Cascade-based attacks on complex networks, Physical
  Review E 66 (2002) 065102(R).

[56] S. V. Buldyrev, R. Parshani, G. Paul, H. E. Stanley, S. Havlin, Catastrophic
  cascade of failures in interdependent networks, Nature 464 (2010) 1025--1028.

[57] C. Fan, L. Zeng, Y. Sun, Y.-Y. Liu, Finding key players in complex networks
  through deep reinforcement learning, Nature Machine Intelligence 2 (2020)
  317--324.

[58] C. Fan, L. Zeng, Y. Ding, M. Chen, Y. Sun, Z. Liu, Learning to identify high
  betweenness centrality nodes from scratch: A novel graph neural network
  approach, in: Proc. 28th ACM Int. Conf. on Information and Knowledge
  Management (CIKM), 2019, pp. 559--568.

[59] A. Varbella, K. Amara, M. El-Assady, B. Gjorgiev, G. Sansavini, PowerGraph: A
  power grid benchmark dataset for graph neural networks, in: Advances in
  Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and
  Benchmarks Track, 2024, arXiv:2402.02827.

[60] T. N. Kipf, M. Welling, Semi-supervised classification with graph convolutional
  networks, in: Proc. Int. Conf. on Learning Representations (ICLR), 2017.

[61] W. L. Hamilton, R. Ying, J. Leskovec, Inductive representation learning on
  large graphs, in: Advances in Neural Information Processing Systems 30
  (NeurIPS), 2017, pp. 1024--1034.

[62] P. Velickovi\'c, G. Cucurull, A. Casanova, A. Romero, P. Li\`o,
  Y. Bengio, Graph attention networks, in: Proc. Int. Conf. on Learning
  Representations (ICLR), 2018.

[63] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, M. Welling,
  Modeling relational data with graph convolutional networks, in: Proc.
  European Semantic Web Conference (ESWC), 2018, pp. 593--607.

[64] X. Wang, H. Ji, C. Shi, B. Wang, Y. Ye, P. Cui, P. S. Yu, Heterogeneous graph
  attention network, in: Proc. The Web Conference (WWW), 2019, pp. 2022--2032.

[65] Z. Hu, Y. Dong, K. Wang, Y. Sun, Heterogeneous graph transformer, in: Proc. The
  Web Conference (WWW), 2020, pp. 2704--2710.

[66] X. Fu, J. Zhang, Z. Meng, I. King, MAGNN: Metapath aggregated graph neural
  network for heterogeneous graph embedding, in: Proc. The Web Conference
  (WWW), 2020, pp. 2331--2341.

[67] Z. Ying, D. Bourgeois, J. You, M. Zitnik, J. Leskovec, GNNExplainer:
  Generating explanations for graph neural networks, in: Advances in Neural
  Information Processing Systems (NeurIPS), Vol. 32, 2019, pp. 9244--9255.

[68] D. Luo, W. Cheng, D. Xu, W. Yu, B. Zong, H. Chen, X. Zhang, Parameterized
  explainer for graph neural network, in: Advances in Neural Information
  Processing Systems (NeurIPS), Vol. 33, 2020, pp. 19620--19631.

[69] J. Pearl, Probabilistic Reasoning in Intelligent Systems: Networks of Plausible
  Inference, Morgan Kaufmann, 1988.

[70] G. Beliakov, A. Pradera, T. Calvo, Aggregation functions: A guide for
  practitioners, Studies in Fuzziness and Soft Computing 221 (2007).

[71] R. R. Yager, On ordered weighted averaging aggregation operators in
  multicriteria decisionmaking, IEEE Transactions on Systems, Man, and
  Cybernetics 18 (1) (1988) 183--190.

[72] G. H. Hardy, J. E. Littlewood, G. P\'olya, Inequalities, 2nd Edition,
  Cambridge University Press, 1952.

[73] U.S. Department of Defense, MIL-STD-498: Software development and
  documentation, Military standard, U.S. Department of Defense (1994).

[74] R. C. Martin, Agile Software Development: Principles, Patterns, and Practices,
  Prentice Hall, 2003.

[75] M. Fey, J. E. Lenssen, Fast graph representation learning with PyTorch
  geometric, in: ICLR Workshop on Representation Learning on Graphs and
  Manifolds, 2019.

[76] F. Xia, T.-Y. Liu, J. Wang, W.-S. Zhang, H. Li, Listwise approach to learning
  to rank: Theory and algorithm, in: Proc. 25th Int. Conf. on Machine Learning
  (ICML), 2008, pp. 1192--1199.

[77] Team SimPy, Simpy: Event discrete simulation for Python,
  <https://simpy.readthedocs.io> (2020).

[78] International Organization for Standardization, ISO/IEC 25022:2016 ---
  systems and software engineering --- systems and software quality
  requirements and evaluation (square) --- measurement of quality in use, Tech.
  rep., International Organization for Standardization (2016).

[79] S. Kato, S. Tokunaga, Y. Maruyama, S. Maeda, M. Hirabayashi, Y. Kitsukawa,
  A. Monrroy, T. Ando, Y. Fujii, T. Azumi, Autoware on board: Enabling
  autonomous vehicles with embedded systems, in: Proc. ACM/IEEE 9th Int. Conf.
  on Cyber-Physical Systems (ICCPS), 2018, pp. 287--296.

[80] Google Cloud Platform, Online boutique: A cloud-native microservices demo
  application, Software artifact (2024).

[81] X. Zhou, X. Peng, T. Xie, J. Sun, C. Ji, W. Li, D. Ding, Fault analysis and
  debugging of microservice systems: Industrial survey, benchmark system, and
  empirical study, IEEE Transactions on Software Engineering 47 (2) (2021)
  243--260.

[82] Home Assistant Community, Home assistant: Open source home automation that
  puts local control and privacy first, Software artifact,
  <https://www.home-assistant.io/> (2024).

[83] Linux Foundation LF Edge, Edgex foundry: An open, vendor-neutral edge iot
  middleware platform, Software artifact, <https://www.edgexfoundry.org/>
  (2024).

[84] F. Wilcoxon, Individual comparisons by ranking methods, Biometrics Bulletin
  1 (6) (1945) 80--83.

[85] B. Efron, R. J. Tibshirani, An Introduction to the Bootstrap, Chapman \& Hall,
  1993.

[86] C. Spearman, The proof and measurement of association between two things,
  American Journal of Psychology 15 (1) (1904) 72--101.

[87] R. Schwartz, J. Dodge, N. A. Smith, O. Etzioni, Green AI, Communications of
  the ACM 63 (12) (2020) 54--63.
https://doi.org/10.1145/3381831 `doi:10.1145/3381831`.

[88] E. Strubell, A. Ganesh, A. McCallum, Energy and policy considerations for deep
  learning in NLP, in: Proceedings of the 57th Annual Meeting of the
  Association for Computational Linguistics (ACL), Florence, Italy, 2019, pp.
  3645--3650.
https://doi.org/10.18653/v1/P19-1355
  `doi:10.18653/v1/P19-1355`.

[89] D. Patterson, J. Gonzalez, Q. Le, C. Liang, L.-M. Munguia, D. Rothchild, D. So,
  M. Texier, J. Dean, Carbon emissions and large neural network training, arXiv
  preprint arXiv:2104.10350 (2021).
https://doi.org/10.48550/arXiv.2104.10350
  `doi:10.48550/arXiv.2104.10350`.

[90] S. Georgiou, M. Kechagia, T. Sharma, F. Sarro, Y. Zou, Green AI: Do deep
  learning frameworks have different costs?, in: Proceedings of the 44th
  International Conference on Software Engineering (ICSE), 2022, pp.
  1082--1094.
https://doi.org/10.1145/3510003.3510221
  `doi:10.1145/3510003.3510221`.

[91] I. O. Yigit, F. Buzluca, Software-as-a-graph: Replication package (datasets,
  generator configurations, simulation harnesses, model checkpoints, and
  analysis scripts), <https://doi.org/10.5281/zenodo.14922108> (2026).
https://doi.org/10.5281/zenodo.14922108
  `doi:10.5281/zenodo.14922108`.

---

# Declarations

**CRediT authorship contribution statement.** **Ibrahim Onuralp Yigit:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing — original draft, Visualization. **Feza Buzluca:** Conceptualization, Methodology, Validation, Writing — review and editing, Supervision, Project administration.

**Declaration of competing interest.** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Funding.** This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

**Data availability.** The complete replication package — including synthetic scenario datasets, generator configurations, simulation harnesses, real-world architecture adapters, trained model checkpoints, and all analysis scripts — is openly available on Zenodo under DOI [10.5281/zenodo.14922108](https://doi.org/10.5281/zenodo.14922108) and cited as [91] in compliance with Option C of the Elsevier research data policy. The synthetic corpus is regenerable: each dataset carries its random seed and SHA-256 cryptographic digest in a committed manifest, with automated tests asserting byte-identical regeneration from configuration files (§6.1). Every table and figure is produced deterministically from committed artifacts by reproducible scripts; none of the reported values is transcribed manually.

**Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.** During the preparation of this work, the authors used AI-assisted language tools to check grammar, improve readability, and support LaTeX typesetting. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.
