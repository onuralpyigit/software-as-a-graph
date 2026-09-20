# Software-as-a-Graph: Heterogeneous Graph Learning for Pre-Deployment Dependability Analysis of Asynchronous and Event-Driven Distributed Systems

**Authors.** Ibrahim Onuralp Yigit, Feza Buzluca

**Affiliation.** Department of Computer Engineering, Istanbul Technical University, 34469 Istanbul, Turkey

**Corresponding author.** Ibrahim Onuralp Yigit — yigiti@itu.edu.tr

---

# Abstract

Modern asynchronous publish–subscribe architectures decouple components across space and time, but conceal cascading failure paths behind message brokers and topics. Whether graph neural networks can forecast failure blast radii from pre-deployment Architecture-as-Code manifests, without runtime telemetry, remains an open empirical question. This study investigates the viability and architectural boundaries of heterogeneous graph learning through Software-as-a-Graph (SaG), a framework transforming manifests into typed multigraphs. We evaluate a Heterogeneous Graph Transformer with Quality-of-Service edge encodings (HGT-QoS) against closed-form network baselines, paired with an interpretable ISO/IEC 25010/25019 attribution layer.

Evaluation across twelve inductive scenarios and five open-source systems reveals three boundaries. First, the learned model gains no significant advantage over an unparameterized QoS-weighted centrality baseline ($\rho = 0.638$ vs. $0.553$, $p = 0.151$); since the simulated target is a deterministic functional of the graph the predictors read, a strong closed-form comparator is expected, bounding what this parity says about graph learning generally. Second, relational typing and QoS encodings act as substitutes rather than complements (interaction $-0.199$, and $-0.233$ after variance stabilisation, negative on 12/12 folds, $p = 0.0005$); typing also carries a $15.4\times$ capacity gap that registered, unrun control arms would separate. Third, transfer to transcribed systems is established over the full component population ($\rho = 0.760$, CI $[0.714, 0.819]$, against $0.51$–$0.53$) but unresolved over actively propagating components, where only the typed model stays positive in point estimate ($\rho_{>0} = +0.236$, CI spanning zero) and both synchronous RPC call trees are non-positive. Computationally, cold feature extraction makes static analysis $2$–$18\times$ costlier than discrete-event simulation (median $5.6\times$), making incremental caching a prerequisite for CI/CD use.

**Keywords:** Heterogeneous graph neural networks; Distributed systems dependability; Publish–subscribe architecture; Cascading failures; Static system analysis; Explainable AI

---

# 1. Introduction

## 1.1 Motivation

Modern large-scale distributed software systems progressively employ asynchronous, event-driven, publish–subscribe (pub-sub) architectures. These architectures are widely used within autonomous driving (ROS 2 [1]), enterprise event streams (Apache Kafka [2]), cyber-physical systems (DDS [3]), IoT deployments (MQTT [4]), cloud-native microservices [5, 6], and distributed AI/LLM serving clusters. Pub-sub architectures decouple producers and consumers across space, time, and synchronization [7]. Components interact indirectly via message topics and brokers, removing the requirement for direct static references. Contemporary middleware specifications also allow deployment-time Quality-of-Service (QoS) policies, including reliability guarantees, durability, message priorities, and delivery deadlines, to manage system performance under peak load and network stress.

While this decoupling supports elastic scalability, it also creates a substantial visibility barrier. Unlike synchronous architectures such as RESTful HTTP or gRPC, which expose interactions by explicit caller–callee paths, publish–subscribe publishers and subscribers do not maintain direct references. Consequently, chain failures, head-of-line blocking, and backpressure can propagate along concealed logical paths involving brokers, shared topics, colocated hosts, and shared libraries [8, 9]. These failures occur mainly through two mechanisms: sequential cascades, where a slow subscriber saturates a broker queue and incrementally throttles its publishers [10]; and simultaneous blast radii, where a shared library crash or host outage takes down all colocated services. Conventional architecture diagrams and static call graphs cannot represent these dissemination processes. Reducing such vulnerabilities is most effective before deployment, particularly during design and continuous integration, consistent with dependable computing principles [11, 12]. At these stages, runtime telemetry, distributed tracing, and operational logs are unavailable. Accordingly, architects and Site Reliability Engineers must identify systemically critical components, topics, and links, ascertain the root causes of their criticality, and implement focused interventions such as broker replication, decoupling over-subscribed topics, or isolating shared libraries to reduce associated risks. Here, systemic criticality denotes an entity’s tendency to serve either as an architectural single point of failure, disrupting downstream connectivity upon failure, or as an error-propagation hub capable of triggering extensive multi-hop cascading outages within the asynchronous message mesh.

By conducting architectural analysis directly on configuration manifests, pre-deployment Static System Analysis eliminates the need for provisioned staging clusters, active container fleets, and live fault-injection harnesses, minimizing staging infrastructure overhead. Nevertheless, pre-deployment static analysis does not surpass in-process simulation in raw CPU speed: deterministic topological feature extraction requires $79.3\,\text{s}$ for a 520-component enterprise mesh and $239.3\,\text{s}$ for 2,000 components. By comparison, the in-process discrete-event cascade simulator completes in $0.08$–$4.5\,\text{s}$ (§§7.5.1 and 8.2). Although the learned GNN forward pass is minimal ($56\,\text{ms}$), cold static analysis costs $2$–$18\times$ the in-process simulation across the twelve scenarios (median $5.6\times$, reaching its maximum on the largest mesh), establishing that static analysis provides an infrastructure-avoidance benefit rather than an in-process CPU execution advantage unless paired with incremental topology caching.

## 1.2 Problem Statement: The Architecture–Code Gap and the Black-Box AI Challenge

Pre-deployment dependability analysis in distributed architectures involves two distinct, complementary tasks. The first task is predictive forecasting: determining whether a data-driven, relation-specific graph neural network can forecast cascading failure blast radii and identify structurally critical components better than closed-form topological metrics. While closed-form metrics capture broad network connectivity, their ability to resolve multi-hop, relation-dependent cascade spread across heterogeneous channels remains an empirical question, which we test directly against such a baseline (§7.1). We train and evaluate the predictive pathway against independent simulation ground truth as a ranking and critical-set identification model.

Explainable Criticality Attribution (Explanation Layer) tackles the limitations of ranking alone. While a ranked shortlist identifies where risk is concentrated, it does not indicate how to remediate it. Accordingly, the predictor is paired with an interpretable structural quality profile grounded in ISO/IEC 25010 [13] and ISO/IEC 25019 [14]. This layer diagnoses structural risk archetypes grounded in standardized software quality models—distinguishing, for example, an unreplicated single point of failure from an error-propagating cascade hub or a high-coupling maintainability bottleneck—to guide targeted architectural remediations. It functions strictly as an attribution model, not a ranking model.

This separation is architectural rather than merely presentational: both pathways operate on the same graph but do not share parameters, and neither is trained on the other’s output. The coupling term that could connect them is turned off by default and reported only as an ablation (§4.2). This independence allows Software-as-a-Graph (SaG) to identify components that are structurally central yet operationally low-impact, enabling a nuanced diagnosis unattainable by either pathway alone.

The distance between an architecture as designed and as realized is long-established: Perry and Wolf [15] formulated architectural erosion three decades ago, and the architectural-technical-debt literature has tracked it since [16, 17]. What we term the **Architecture–Code Gap** specializes this concept to asynchronous middleware, where the challenge is that failure semantics were never expressible in artifacts a build pipeline can inspect: *a distributed system can have pristine, bug-free source code within each service, yet remain fragile to catastrophic global outages caused by hidden single points of failure (SPOFs) or mismatched middleware Quality-of-Service (QoS) contracts.* This fragility is acute in pub-sub architectures where producers and consumers interact indirectly without static references. Classical architecture evaluations (e.g., ATAM [18, 12]) rely on manual elicitation; static code analysis [19, 20, 21, 22] inspects individual services in isolation; chaos engineering [23] requires provisioned staging clusters; and homogeneous centrality [24, 25, 26, 27] flattens typed topologies into untyped graphs. §2 develops each paradigm in detail.

Moreover, although machine learning has achieved significant progress in software engineering, current AI approaches to system dependability frequently operate as uninterpretable black boxes. Deep neural models generally produce scalar risk scores or latent embeddings without transparent, actionable rationales for their predictions. In mission-critical software engineering, this lack of interpretability is inadequate; developers and Site Reliability Engineers require clear explanations of component vulnerabilities and compromised architectural mechanisms to refactor code or reconfigure infrastructure effectively.

## 1.3 The Software-as-a-Graph (SaG) Approach

To address both the Architecture–Code Gap and the black-box AI challenge, this work puts forward **Software-as-a-Graph (SaG)**, an AI-driven pre-deployment **Static System Analysis (SSA)** framework for asynchronous and event-driven distributed systems. SaG provides the modeling and experimental foundation to investigate whether pre-deployment static representations can identify systemically critical components and attribute their structural root causes. SaG realizes Architecture-as-Code through a unified four-stage pipeline: (1) formulating the architecture as a typed, directed multigraph over Applications, Brokers, Topics, Execution Hosts, and Shared Libraries (§3.1); (2) projecting explicit physical connections into a QoS-weighted semantic `DEPENDS_ON` layer capturing sequential cascades and simultaneous blasts (§3.2); (3) training a Heterogeneous Graph Transformer (HGT) that forecasts multi-hop cascading blast radii (§4); and (4) evaluating an explainable, standards-grounded Reliability–Maintainability (RM) attribution profile (§5) to pinpoint why components are fragile.

Crucially, SaG enforces a strict **input–label independence guarantee**: learned models and attribution baselines operate exclusively on the analytical graph $G_{\text{analysis}}$. At the same time, ground-truth failure impacts are generated by independent discrete-event simulators operating on the raw structural topology $G_{\text{structural}}$ (§4.4). This separation supports the core claim by keeping prediction and evaluation independent. §3 formalizes the complete architectural flow and pipeline interactions (Figure 1).

#### Rationale for Graph Learning vs. Direct Simulation

Since discrete-event simulation $I^*(v)$ defines ground-truth criticality and completes in $0.08$–$4.5\,\text{s}$, the rationale for training a graph model needs stating. Manifest analysis avoids provisioning staging clusters and chaos harnesses altogether — infrastructure avoidance, not a measured energy saving; §8.2 bounds the energy the analysis itself spends. In CI/CD, an incremental caching design could reduce per-commit latency to the sub-second forward pass ($56\,\text{ms}$) by extracting feature deltas only for modified subgraphs, but that is an engineering hypothesis: delta latency depends on where in the topology a pull request lands, and uncached static analysis is more expensive than simulation on every scenario we measured. Message passing does generalize across labeled and unlabeled entities, scoring shared libraries and hosts that a node-level simulation sweep leaves unscored. Two further rationales are not supported: cascade simulation is near-deterministic on this corpus (median test–retest $0.982$), and manifest-level fault injection needs no runnable containers. §7.1 evaluates whether graph learning offers ranking advantages over closed-form baselines.

## 1.4 Research Questions

This empirical study considers five research questions:

-   **RQ1 (Predictive Efficacy):** *How accurately does heterogeneous graph learning predict cascading failure impact and identify the critical component set, compared with traditional, non-learning network indicators?*

-   **RQ2 (Value of Architectural Typing):** *Does modeling distinct entity and dependency types yield better failure predictions than homogeneous graph models on architectures the model has never seen — and does whatever advantage it confers depend on what other relational signal the model already has?*

-   **RQ3 (QoS Encoding and Robustness):** *(i) Do middleware Quality-of-Service contracts carry signal a purely structural score discards, (ii) does that signal compose with or substitute for architectural typing, (iii) do the framework’s simulation oracles agree with one another, and (iv) are the reported orderings robust to the free parameters of the scorer and of the ground truth?*

-   **RQ4 (Real-World Generalization):** *How effectively does the framework transfer zero-shot to authentic, real-world distributed systems across autonomous driving (ROS 2), cloud-native microservices, smart home IoT, and industrial edge computing?*

-   **RQ5 (Analysis Cost):** *What does pre-deployment analysis cost at CI/CD time, which pipeline stage dominates that footprint, and how does it compare against the discrete-event simulation it is intended to displace?*

## 1.5 Key Contributions

This paper makes three principal contributions:

1.  **Heterogeneous Graph Learning for Dependability, and Its Boundaries:** A relation-specific Heterogeneous Graph Transformer that forecasts cascading blast radii from Architecture-as-Code manifests, incorporating 16-D QoS edge encodings and an auxiliary component reliability head (§4). Ablated separately under inductive distribution shift across twelve architectures, relation-specific HGT demonstrates out-of-distribution gains ($\Delta\rho = +0.234$ over an untyped baseline, $p = 0.0005$, reflecting the joint transition to the parameterized bidirectional HGT architecture). Crucially, typing and continuous QoS encodings act as empirical substitutes rather than complements — an interaction that survives transformation to a variance-stabilised scale, so it is a property of the mechanisms rather than of bounded $\rho$ — though the typing factor also carries a $15.4\times$ capacity gap and a change in message-passing directionality that the reported controls do not separate. Against an unparameterized QoS-weighted centrality baseline, learned ranking does not establish a statistically significant advantage ($+0.085$, $p = 0.151$). That parity should be read against the target: $I^*(v)$ is a deterministic functional of the same graph the predictors read, so a closed-form centrality is an expected-strong comparator and the result bounds this design more than it bounds graph learning. We report the substitution effect and the empirical boundary as primary findings (§§7.1–7.2).

2.  **A Formal Typed Architecture Model:** A multigraph representation that derives logical dependencies from physical pub-sub linkages and distinguishes sequential cascade propagation from simultaneous multi-consumer library failures (§3).

3.  **Empirical Benchmark, Real-World Transfer, and Cost Profile:** A study across seventeen system architectures totaling 2,812 components: twelve synthetic topologies (2,461 components across LOSO folds) and five open-source reference systems (351 components) under strict graph-view separation. We characterize pipeline costs, showing that the neural forward pass accounts for only $0.02\%$ of runtime, whereas cold deterministic feature extraction dominates and costs $2$–$18\times$ the simulation it was meant to displace (median $5.6\times$ over twelve scenarios, $79.3\,\text{s}$ against $4.5\,\text{s}$ at its maximum), refuting the premise that static analysis is inherently faster than in-process simulation without incremental caching (§§6–7).

One property of these results should be visible before any of them is quoted. Learned figures in this paper are revision-sensitive: between the two most recent committed sweeps, at identical seeds and an identical corpus, the training-free cells reproduce exactly while the learned fold means move by up to $0.172$ — twice the $+0.085$ margin the primary contrast turns on (§8.3). The sign of that contrast is stable across every configuration we have run; its magnitude should be read as pinned to the artifacts we ship rather than to the method.

#### Relationship to the authors’ prior work

A previous conference paper [28] introduced the preliminary multigraph formulation and deterministic quality model on synthetic topologies. This JSS manuscript substantially extends that work through incorporating the complete predictive HGT pathway with 16-dimensional QoS edge encoding (§4); inductive LOSO cross-validation (§7.2); zero-shot evaluation throughout five open-source systems (§7.4); empirical cost and sustainability characterization (§7.5); multi-oracle convergent validity and graph-view separation (§§4.3–4.4); and system-wide sensitivity analyses (§7.3). We retained formalisms from the conference paper only in restructured portions of §§3 and 5. Two of its conclusions do not survive the re-measurement reported here, and we state that explicitly rather than leaving the earlier record standing: the deterministic quality model is not a competitive ranking instrument on this corpus ($\rho = 0.205$, below unweighted centrality, §7.1), and its elicited weights rank worse than a uniform prior (§7.3). The corpus, the oracles and the evaluation population all differ from [28], so these are corrections to what we now believe rather than a failure to reproduce a fixed experiment.

## 1.6 Paper Organization

The remainder of this paper is organized as follows. §2 reviews related work. §3 formalizes the SaG multigraph model and dependency projections. §4 details the Heterogeneous Graph Transformer and simulation oracles. §5 presents the ISO/IEC-grounded explanation layer. §6 outlines the experimental protocol, and §7 reports empirical results for RQ1–RQ5. §8 discusses practical consequences, sustainability, threats to validity, and limitations. §9 concludes.

# 2. Related Work

The central claim of this study is that joining four core research domains (1) dependability, performance, and sustainability in distributed software systems; (2) static code and system analysis; (3) software quality analysis and multi-criteria evaluation; and (4) graph representation learning and explainable artificial intelligence (XAI)—enables prediction of systemic cascading vulnerabilities and performance degradation from Architecture-as-Code descriptors before runtime infrastructure provisioning.

## 2.1 Dependability, Performance, and Sustainability in Distributed Software Systems

The publish–subscribe (pub-sub) and asynchronous event-driven paradigms decouple communicating entities in space, time, and synchronization, enabling elastic scalability and high throughput [7]. Contemporary middleware standards, including ROS 2 [1], Apache Kafka [2], DDS [3], and MQTT [4], regulate these exchanges through fine-grained Quality-of-Service (QoS) policies. In cloud-native microservice meshes and distributed AI/LLM serving infrastructures, asynchronous message passing and queueing topologies form the primary communication substrate and influence tail latencies, throughput bottlenecks, and hardware utilization.

Prior work on dependability and performance has emphasized runtime mechanisms — dynamic consensus, broker clustering, backpressure throttling, autoscaling, automated failover — and, in parallel, chaos engineering and runtime fault injection [23], which disrupt live clusters to observe degradation and recovery. Runtime testing validates realistically but needs a provisioned cluster, risks service disruption and costs compute, which limits it during early design and commit-level CI/CD and places it among the development-time computations green software engineering scrutinises [29, 30, 31, 32, 33]. This study instead analyses dependability from Architecture-as-Code descriptors before any infrastructure is provisioned.

Predicting dependability from architectural descriptions has a rich analytical lineage. Cheung’s absorbing-Markov-chain model [34] derives system reliability from component reliabilities and transfer-of-control graphs; Goseva-Popstojanova and Trivedi [35] systematize subsequent state-based, path-based, and additive families, as Immonen and Niemelä [36] survey. Model-driven frameworks, such as the Palladio Component Model [37] and layered queueing networks [38], predict performance and reliability from parameterized component specifications, while annotation-based approaches such as the AADL Error Model Annex [39] generate fault trees from declared error states. These methods address broader questions than those considered in this study, but require pre-calibrated operational profiles and failure rates unavailable at commit time. In contrast, SaG asks a single targeted question from manifests alone: which components’ failures propagate furthest through the declared topology?

A substantial body of literature localizes faults in microservices using operational telemetry: Seer [40] and Sage [41] predict QoS violations from hardware counters and traces; MicroRCA [42] and TraceRCA [43] isolate root causes over service-dependency graphs; and DeepTraLog [44], Eadro [45], and MicroCause [46] apply graph neural networks to multimodal traces, logs, and metrics (surveyed across 98 papers by Zhang et al. [47]). In an industrial benchmark study, Zhou et al. [48] show that cascading outages in synchronous microservices result from thread-pool exhaustion, RPC timeouts, and recursive retry storms propagating along call trees. In contrast, pub-sub failures propagate via broker queue saturation and message starvation. All these approaches require a running cluster emitting runtime telemetry, whereas SaG addresses the pre-deployment complement by operating on static manifests before code execution.

## 2.2 Static Code Analysis (SCA) vs. Static System Analysis (SSA)

Traditional **Static Code Analysis (SCA)** tools (e.g., SonarQube [19]) inspect source code Abstract Syntax Trees (ASTs) within individual services. They evaluate cyclomatic complexity [20], class cohesion, module coupling (e.g., Lack of Cohesion in Methods [LCOM], Coupling Between Objects [CBO]) [21, 22], and code duplication to flag internal code smells and defect-prone modules [49, 50, 51, 52]. However, SCA cannot observe runtime communication topology: it does not capture inter-service messaging channels, message broker queue saturation, or cross-host failure propagation.

Static recovery of system-level structure is an active area of complementary research. Bushong et al. [53] derive communication diagrams from static code analysis, and a recent review evaluates nine architecture recovery tools [54]. In robotics and publish-subscribe ecosystems, specialized static analysis tools such as HAROS [55] extract computation models to detect structural defects prior to system launch. This body of literature aims to construct accurate representations of implemented systems to facilitate comprehension and drift detection. In contrast, SaG uses a declared topology as input to forecast cascade blast radii.

To bridge the “Architecture–Code Gap,” **Static System Analysis (SSA)** — not to be confused with static single assignment — extends static analysis from single-service source code to the global system architecture. Through modeling distributed applications, message topics, brokers, execution nodes, and shared libraries as a connected multigraph, SSA propagates code-level quality metrics across architectural dependencies. This approach enables engineering teams to detect structural anti-patterns [56, 57] and architectural technical debt [58] early during continuous integration (CI/CD) [59, 60], before defective topologies enter production.

## 2.3 Software Quality Models and Multi-Criteria Evaluation

Software product quality is standardized by the **ISO/IEC 25010:2023** product quality model [13] and the **ISO/IEC 25019:2023** Quality-in-Use model [14]. Of the characteristics ISO/IEC 25010:2023 defines under Reliability, Maintainability and Efficiency of Performance, SaG operationalizes the strict subset derivable from deployment topology: Availability and Fault Tolerance, plus Modularity, Modifiability and Analyzability (§5.1). Faultlessness, Recoverability, Reusability and Testability lie outside topological analysis.

Software engineering measurement explicitly distinguishes between *internal quality* (inbuilt structural attributes assessed on static artifacts at rest) and *external quality* (runtime dependability and behavioral characteristics noted during system execution) [61, 62]. In distributed architectures, architectural technical debt, such as over-centralized message topics or unreplicated brokers, degrades internal quality and can precipitate severe external performance bottlenecks, queue congestion, and outages.

Aggregating multi-attribute structural metrics into an auditable quality score presents a classic Multi-Criteria Decision Making (MCDM) challenge. The Analytic Hierarchy Process (AHP) [63] supplies a structured pairwise-comparison methodology with an explicit Consistency Ratio ($CR \le 0.10$) to guarantee consistency among elicited judgments. While this statistic detects inconsistency, it does not identify matrices completed from predetermined answers, a limitation addressed for the weights in this study (§5.2). This work applies AHP to establish an audited, explainable Reliability–Maintainability (RM) quality baseline together with learned graph models.

## 2.4 Graph Representation Learning and Explainable AI

Network science offers established centrality indices for identifying critical nodes, such as degree, closeness, betweenness centrality [24, 26], articulation points, and PageRank [25, 27]. Fundamental studies on network robustness [10], cascading overloads [8], and interdependent networks [9] model disruption propagation over interconnected topologies. While percolation models provide natural theoretical comparators for network disintegration, analytical bond and site percolation thresholds assume statistically homogeneous, undirected connectivity. They cannot be calculated in closed form over directed multigraphs with asymmetric entity semantics, heterogeneous channel capacities, and continuous QoS contracts. Consequently, this study’s training-free baselines are centrality-based (§6.2), and we propose targeted multi-type percolation fragmentation as a valuable direction for future baseline formulations.

Standard network measures present two major shortcomings in the context of software architectures. First, Dimensional Collapse arises when a single centrality value does not distinguish the underlying reasons for a component’s criticality, such as differentiating between an isolated single point of failure, an error-propagating cascade hub, or an over-shared library. Second, Semantic Collapse occurs when unweighted metrics treat all nodes and edges equivalently, conflating fundamentally distinct architectural entities, such as asynchronous message topics, shared libraries, and physical execution hosts.

To tackle the limitations of hand-engineered metrics, recent research applies machine learning to network vulnerability analysis (e.g., FINDER [64], DrBC [65], PowerGraph [66]). However, most models employ homogeneous message passing (GCN [67], GraphSAGE [68], GAT [69]) and indiscriminately average signals across connection types. Given the inherent heterogeneity of distributed software architectures, such models obscure entity boundaries and fail to generalize to out-of-distribution scenarios. Heterogeneous Graph Neural Networks (RGCN [70], HAN [71], HGT [72], MAGNN [73]) address this issue through relation-specific transformations. The present study uses the Heterogeneous Graph Transformer (HGT) [72] to preserve typed relational semantics when forecasting cascade blast radii. Graph learning has also been directly applied to microservice topologies; for instance, Khodabandeh et al. [74] predict future service interactions using graph attention over temporally segmented call graphs. However, that approach forecasts edge existence relying on observed interaction history. In contrast, the current method accepts a declared topology as input and predicts the blast radius resulting from node removal.

A major challenge in applying modern artificial intelligence to software engineering is the black-box barrier: deep neural models generate risk scores or continuous embeddings without elucidating basic structural causality. In production environments, these uninterpretable risk rankings hinder actionable decision-making, as developers and site reliability engineers (SREs) cannot determine whether to replicate hosts, configure circuit breakers, or refactor shared libraries.

Existing graph neural network (GNN) explanatory methods, such as GNNExplainer [75] and PGExplainer [76], identify influential subgraphs through edge masking or parameterized learning. Although valuable, these approaches interpret models using internal hidden representations rather than standardized software engineering concepts. SaG tackles this drawback through a decoupled dual-pathway design. The predictive HGT pathway reveals typed mutual-attention distributions that indicate which architectural relations propagated the cascade (§7.3.3). At the same time, the deterministic explanation layer attributes fragility to standardized ISO/IEC quality sub-characteristics (§5), translating raw predictions into implementable, cost-effective remediations.

# 3. The Software-as-a-Graph (SaG) Architectural Model

Figure 1 presents the end-to-end architecture of the SaG framework. The shared front end processes Architecture-as-Code manifests, constructs a typed multigraph, projects implicit runtime interactions throughout a QoS-weighted logical dependency layer, and extracts typed node properties. These features feed two independent pathways with no shared parameters. The predictive pathway (§4) forecasts cascading failure blast radii using a Heterogeneous Graph Transformer. The explanation layer (§5) decomposes fragility into Reliability and Maintainability quality profiles.

![Figure 1](latex/figures/Figure_1.png)

*Figure 1. End-to-end architecture of the SaG framework. The predictive pathway runs left to right: manifest ingestion, typed multigraph, QoS-weighted DEPENDS_ON projection, typed node properties, heterogeneous graph learning, ranked critical set. The dashed edge marks the ground-truth simulation oracle, which operates only on Gstructural, trains the predictor offline and takes no part in inference. The explanation layer re-enters from the analysis multigraph and shares no parameters with the predictor, reaching flagged components through triage rather than data flow.*

This section formalizes the Software-as-a-Graph multigraph representation (§3.1), the QoS-aware weighting and logical dependency derivation rules (§3.2), the dual graph views (§3.3), and the typed node feature encodings (§3.4).

## 3.1 Formal Multigraph Definition

A complex distributed software system is formally represented as a typed, weighted, directed multigraph:

$$\tag{1}
\mathcal{G} = (V, E, \tau_V, \tau_E, w_V, w_E)$$

where:

-   $V$ is the set of system entities, partitioned into five disjoint categories $\mathcal{T}_V = \{\text{app}, \text{broker}, \text{topic}, \text{host}, \text{lib}\}$ such that $V = V_{\text{app}} \cup V_{\text{broker}} \cup V_{\text{topic}} \cup V_{\text{host}} \cup V_{\text{lib}}$. To prevent conflation with physical compute machines, $V_{\text{host}}$ is designated as *Execution Hosts* (physical hosts or virtualized nodes).

-   $E$ is the set of directed edges connecting entities.

-   $\tau_V: V \to \mathcal{T}_V$ and $\tau_E: E \to \mathcal{T}_E$ are typing functions assigning entity and relationship categories.

-   $w_V: V \to (0, 1]$ and $w_E: E \to (0, 1]$ are weighting functions representing entity criticality and connection strength. For applications and shared libraries, $w_V(v)$ is initialized from static code metrics as $w_V(v) = 1 - \text{CQP}(v)$ (§3.4). It defaults to $1.0$ when static code metrics are absent or for infrastructure entities ($w_V(\text{host}) = 1.0$).

Table 1 summarizes the five entity types and six structural edge types defined in the SaG model, together with their semantics and representative distributed-system implementations.

**Table 1.** Entity and structural edge types in the SaG model.

| **Entity Type ($\mathcal{T}_V$)**      | **Architectural Role**                             | **Concrete System Examples**                   |
|:---------------------------------------|:---------------------------------------------------|:-----------------------------------------------|
| **Application** ($V_{\text{app}}$)     | Autonomous process producing/consuming messages    | ROS 2 node, Kafka microservice, MQTT client    |
| **Broker** ($V_{\text{broker}}$)       | Message routing and queuing intermediary           | RabbitMQ exchange, Mosquitto, EMQX broker      |
| **Topic** ($V_{\text{topic}}$)         | Named logical communication channel                | `/sensor/lidar`, `orders.payment.completed`    |
| **Execution Host** ($V_{\text{host}}$) | Physical host or virtualized execution environment | Bare-metal server, Kubernetes worker, Cloud VM |
| **Library** ($V_{\text{lib}}$)         | Shared software package or runtime dependency      | Kafka client, OpenCV, Protobuf runtime         |
| **Structural Edge ($\mathcal{T}_E$)**  | **Direction**                                      | **Semantic Meaning**                           |
| `PUBLISHES_TO`                         | App/Library $\to$ Topic                            | Component publishes messages to topic          |
| `SUBSCRIBES_TO`                        | App/Library $\to$ Topic                            | Component consumes messages from topic         |
| `ROUTES`                               | Broker $\to$ Topic                                 | Broker manages and routes topic traffic        |
| `RUNS_ON`                              | App/Broker $\to$ Host                              | Process is hosted on physical/virtual host     |
| `CONNECTS_TO`                          | Host $\to$ Host                                    | Physical network link between hosts            |
| `USES`                                 | App $\to$ Library                                  | Application links to shared library dependency |

Application and Library entities also incorporate static code metrics generated by Static Code Analysis (SCA) tools: lines of code, cyclomatic complexity, coupling between objects, and lack of cohesion in methods. These metrics link code-level fragility directly to topological analysis.

**Notation.** Entity and edge types: Table 1. Simulation oracles: Table 3.

|                         |                                          |                                            |                                                    |
|:------------------------|:-----------------------------------------|:-------------------------------------------|:---------------------------------------------------|
| $G_{\text{structural}}$ | Raw multigraph; oracles only             | $Q(v)$                                     | RM composite quality score                         |
| $G_{\text{analysis}}$   | `DEPENDS_ON` projection; predictor input | $\rho$                                     | Spearman $\rho$, full population                   |
| $V_{\text{app}}$        | Application nodes; the scored population | $\rho_{>0}$                                | Spearman $\rho$, active stratum ($I^* > 0$)        |
| $w(t)$, $w(e)$          | QoS topic weight, edge weight            | $F_1@K$                                    | Critical-set overlap, $K = 0.20\,|V_{\text{app}}|$ |
| $I^*(v)$                | Primary cascade-reachability oracle      | $I_{\text{comp}}$, $I_{\text{dyn}}$, $I_M$ | Further oracles (Table 3)                          |

## 3.2 QoS-Aware Weights and Logical Dependency Derivation

In distributed middleware, communication links differ in strength according to their Quality-of-Service (QoS) contracts. For example, a `RELIABLE` topic with `TRANSIENT_LOCAL` durability creates a stronger binding between communicating services than a `BEST_EFFORT` telemetry stream.

Each topic $t$ carries an intrinsic criticality weight $w(t) \in (0, 1]$ combining its declared QoS semantics with two runtime-stress modulators: payload size and publication frequency:

$$\tag{2}
w(t) = \alpha_{\text{top}} \cdot \text{QoS}(t) + \beta_{\text{top}} \cdot \text{SizeNorm}(t) + \gamma_{\text{top}} \cdot \text{FreqNorm}(t),
\quad (\alpha_{\text{top}},\, \beta_{\text{top}},\, \gamma_{\text{top}}) = (0.75,\, 0.15,\, 0.10)$$ where $(\alpha_{\text{top}}, \beta_{\text{top}}, \gamma_{\text{top}})$ is a convex combination satisfying $\alpha_{\text{top}} + \beta_{\text{top}} + \gamma_{\text{top}} = 1.0$. The QoS term is an AHP-weighted aggregate of the declared contract:

$$\tag{3}
\text{QoS}(t) = w_{\text{rel}} \cdot q_{\text{rel}} + w_{\text{dur}} \cdot q_{\text{dur}} + w_{\text{prio}} \cdot q_{\text{prio}},
\quad (w_{\text{rel}}, w_{\text{dur}}, w_{\text{prio}}) = (0.24,\, 0.62,\, 0.14)$$

Here, $q_{\text{rel}}, q_{\text{dur}}, q_{\text{prio}} \in [0, 1]$ represent normalized reliability, durability, and transport-priority scores mapped from manifest policies: $q_{\text{rel}} \in \{0.0, 1.0\}$ (best-effort vs. reliable), $q_{\text{dur}} \in \{0.0, 0.5, 1.0\}$ (volatile, transient-local, persistent), and $q_{\text{prio}} \in \{0.0, 0.5, 1.0\}$ (low, medium, high). Durability carries highest weight as it determines state preservation across restarts. The sub-weights derive from a Saaty pairwise matrix with consistency ratio $CR = 0.016 \le 0.10$. §§2.3 and 5.2 note that $CR$ is uninformative for a matrix back-filled from a chosen priority vector, so we state which case this is: the Topic QoS matrix is one of the two in the framework that carry genuine second-eigenvalue spread, and its $CR$ therefore reports consistency rather than construction (Supplementary Table S4).

The modulators $\text{SizeNorm}(t)$ and $\text{FreqNorm}(t)$ are logarithmically compressed and clamped to $[0, 1]$:

$$\tag{4}
\text{SizeNorm}(t) = \min\left(1.0, \frac{\log_2(1 + B(t))}{20}\right), \quad
\text{FreqNorm}(t) = \min\left(1.0, \frac{\log_{10}(1 + F(t))}{3}\right)$$

Here, $B(t)$ denotes the message payload size in bytes (with a 1 MiB design envelope, which represents the practical DDS sample ceiling before RTPS fragmentation becomes dominant), and $F(t)$ represents the nominal publication frequency in Hertz. The final weight $w(t)$ is clamped to $[0.01, 1]$ to ensure that best-effort edges remain visible during graph traversals. Each structural communication edge incident on $t$ (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`) inherits $w_E(e) = w(t)$ together with the topic’s QoS vector.

Sweeping the full $(\alpha_{\text{top}}, \beta_{\text{top}}, \gamma_{\text{top}})$ simplex leaves induced orderings intact ($\rho \ge 0.919$), moving downstream ranking by $0.031$ (`Topo-QoS`) and $0.007$ (RM), and Morris screening finds none of the three load-bearing ($\mu^* \le 0.025$): the split is a documented convention, not a sensitive parameter.

### Logical Dependency Projection (`DEPENDS_ON`)

Structural edges represent explicit deployment connections but do not capture implicit runtime dependencies. For instance, a subscriber depends on a publisher, yet no direct edge connects them in publish-subscribe architectures. To address this, a single unified semantic relation, `DEPENDS_ON`, is derived and directed from *dependent* to *dependency* ("if target fails, source is impacted"), following the six projection rules detailed in Table 2. The resulting weight $w \in (0, 1]$ quantifies the magnitude of operational coupling and indicates the conditional likelihood that a disruption in the dependency propagates to the dependent.

**Table 2.** The six `DEPENDS_ON` logical dependency projection rules.

| **Rule** | **Dependency Category** | **Structural Pattern ($\text{Dependent} \to \text{Dependency}$)**                    | **Derived Weight ($w$)**                                                                    |
|:--------:|:------------------------|:-------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|
|  **1**   | `app_to_app`            | Subscriber $\to$ Publisher (via shared Topic, incl. transitive `USES`)               | $1 - \prod_{t \in T}(1 - w(t))$                                                             |
|  **2**   | `app_to_broker`         | Publisher/Subscriber $\to$ Broker routing its topics                                 | $1 - \prod_{t \in T}(1 - w(t))$                                                             |
|  **3**   | `host_to_host`          | Host $\to$ Host (lifted from inter-host app dependencies)                            | $\max_{u \in \text{hosted}(h_1), v \in \text{hosted}(h_2)} w_{\text{DEPENDS\_ON}}(u \to v)$ |
|  **4**   | `host_to_broker`        | Host $\to$ Broker (lifted from hosted app dependencies)                              | $\max_{u \in \text{hosted}(h)} w_{\text{DEPENDS\_ON}}(u \to b)$                             |
|  **5**   | `app_to_lib`            | Application $\to$ Shared Library it `USES`                                           | $H(w_V(\text{app}), w_V(\text{lib}))$                                                       |
|  **6**   | `broker_to_broker`      | Broker $\leftrightarrow$ Broker (shared physical fault-domain colocation, symmetric) | $w_V(\text{host})$                                                                          |

Rules 1 and 2 aggregate the set of topics $T$ connecting a component pair using a probabilistic union rather than a maximum [77, 78, 79]. This approach confirms that additional parallel failure vectors increase coupling monotonically while maintaining $w \in (0, 1]$. Rule 5 employs the harmonic mean $H(x, y) = 2xy/(x+y)$ [80] to combine the vertex weights of the consuming Application and the shared Library, consequently balancing caller and dependency criticality. Rules 3 and 4 extend application-level dependencies across host boundaries using the maximum coupling weight.

### Sequential Cascades vs. Simultaneous Blasts

A central principle of the SaG model is the distinction between two degradation modes. In a **Sequential Cascade (Rule 1)**, a failed publisher starves downstream subscribers sequentially through message queues and topic buffers. In a **Simultaneous Blast (Rule 5)**, a crashed library or execution host causes all consuming applications and colocated brokers to fail instantaneously in a single shared-fate event. Preserving entity types and relation-specific projection rules lets the SaG model represent both mechanisms, whereas untyped homogeneous graphs collapse them into indistinguishable edges.

Rule 6 is the only symmetric rule: two brokers colocated on the same host share that host’s physical failure domain ($w = w_V(\text{host})$). Colocated brokers compete for resources, and host crashes halt them simultaneously. Rule 6 applies in four of the eight scenarios of a companion study’s detection benchmark (12 directed edges); that suite is not this paper’s corpus, and the figure is quoted only to indicate how rarely the rule fires. Because simulation oracles operate strictly on $G_{\text{structural}}$ (§4.4), Rule 6 has zero influence on ground-truth failure labels.

## 3.3 Dual Graph Views and Architectural Layers

The SaG framework consists of two distinct representations. The **Structural Graph** ($G_{\text{structural}}$) is the raw deployment graph containing physical and structural relations (`PUBLISHES_TO`, `ROUTES`, `RUNS_ON`, `USES`); it preserves the untransformed deployment topology. The **Analysis Graph** ($G_{\text{analysis}}$) is the projected graph containing derived `DEPENDS_ON` edges annotated with QoS weights and ingested SCA metrics. We compute all GNN embeddings and analytical metrics on the analysis graph (Supplementary Figure S3).

$G_{\text{analysis}}$ is further organized into four analytical layers (Application, Middleware, Infrastructure, and Global System), enabling criticality evaluation at subsystem levels consistent with hierarchical frameworks such as MIL-STD-498 [81].

## 3.4 Typed Node Feature Encoding

Both the predictive pathway (§4) and the explanation layer (§5) read the same typed node properties from $G_{\text{analysis}}$: the predictor projects them per entity type before message passing, the explanation layer aggregates them into a quality profile. All five entity types share an 18-dimensional block (indices 0–17) of topological metrics — PageRank and its reverse, betweenness, closeness, eigenvector and degree centralities, clustering coefficient, undirected and directed articulation scores, bridge ratio, three QoS weight aggregates, multi-path connectivity, path complexity, fan-out criticality, and the Connectivity Degradation Index (CDI). The full schema is in Supplementary §S11. CDI is by far the most expensive of these and dominates the analysis cost of §7.5; because it is a predictor input and not only a term of the Availability score, it cannot be gated away without changing both pathways. Every metric in the block is normalized to $[0, 1]$ within its graph, which stops raw graph size from driving the per-type projections under cross-scenario transfer. Type-specific blocks extend the vector to 19–25 dimensions, adding source-code metrics and the Code Quality Penalty for Applications, reverse-`USES` blast-radius drivers for Libraries, queue capacity for Brokers, publisher/subscriber counts and ordinal QoS criticality for Topics, and CPU and memory allocation for Execution Hosts.

The shared block provides the GNN with global structural and positional context—analogous to positional and structural encodings in Graph Transformers—enabling relational message passing to modulate multi-hop representations based on global network role rather than local immediate adjacency alone. Crucially, because betweenness, closeness, reverse PageRank, and articulation scores are topological summaries computed before model evaluation, a learned model is not the only way to derive a criticality score from structure. This design guarantees that the closed-form baselines serve as fair, competitive comparators rather than strawman alternatives, while making deterministic feature extraction the dominant computational bottleneck ($O(|V|^2 + |V||E|)$) analyzed in §7.5.

# 4. Graph Learning for Failure-Impact Prediction

Cascading failure impact in distributed software systems indicates non-linear, multi-hop, and relation-dependent characteristics. Outages propagate through architectural relations and dependencies beyond immediate neighbors. Whether a closed-form combination of standard centrality indices can adequately capture these complicated dynamics remains an open empirical question. Consequently, the primary predictive approach described in §1.2 utilizes a learned graph model, which is evaluated in §7.1 against a closed-form baseline. The learned model does not manifest significant improvement over the baseline in out-of-distribution ranking.

This section presents the Heterogeneous Graph Transformer (HGT) architecture and its typed edge encodings (§4.1), the multi-task prediction heads and dimension-masked loss formulation (§4.2), the ground-truth simulation oracles (§4.3), and the input–label independence guarantee designed to prevent data leakage (§4.4).

## 4.1 Heterogeneous Graph Transformer Architecture

Distributed systems comprise heterogeneous entity types (Applications, Libraries, Brokers, Topics, Execution Hosts) and diverse interaction semantics (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`, `DEPENDS_ON`). We therefore employ a three-layer **Heterogeneous Graph Transformer (HGT)** architecture [72], implemented within PyTorch Geometric [82], with hidden dimension $D = 64$ and $H = 4$ attention heads. This architecture guarantees that typed relations, rather than simple adjacency, govern failure-impact forecasting.

### 4.1.1 Continuous-Categorical Edge Feature Encoding (16-D)

To capture continuous QoS constraints and channel semantics, SaG encodes each directed edge $e = (u,v)$ as a 16-dimensional continuous-categorical vector $e_{uv} \in \mathbb{R}^{16}$. Index 0 is the scalar coupling weight $w_E(e) \in (0,1]$ of §3.2; index 1 is the normalized count of simple paths through $e$; indices 2–8 one-hot encode the seven structural and derived relations; and indices 9–15 carry middleware QoS parameters on `PUBLISHES_TO` and `SUBSCRIBES_TO` edges, zeroed elsewhere. Six QoS dimensions are active in our corpus — reliability, durability, message priority, a heterogeneity flag raised when an edge’s QoS triple departs from its scenario’s modal profile, and the deadline pair (an active flag and $\log_{10}(1 + \text{deadline\_ns}/10^6)$, populated on $463$ of $615$ topics $75\%$). The seventh, $\log_{10}(1 + \text{max\_blocking\_ms})$, is a schema provision for hard real-time DDS and ROS 2 profiles and is zero throughout.

An edge projection module maps $e_{uv}$ into the hidden space: $e_{uv}' = W_{\text{edge}} e_{uv}$. Before relational attention computation, the current projection vector is incorporated directly into the target node representation: $\tilde{h}_v = h_v + e_{uv}'$.

### 4.1.2 Type-Specific Projection and Heterogeneous Message Passing

For each source node $u$ and target node $v$ connected by meta-relation $\tau(e) = (\tau(u), \phi(e), \tau(v))$, SaG follows the Heterogeneous Graph Transformer formulation of Hu et al. [72] implemented via PyTorch Geometric’s `HGTConv` [82]. Entity-specific projections $W_{\tau(v)}$ first map raw features $x_v \in \mathbb{R}^{19\text{--}25}$ into the shared $D$-dimensional hidden space: $h_v^{(0)} = \text{LayerNorm}(\text{GELU}(W_{\tau(v)} x_v))$. Relational mutual attention across $H$ heads incorporates type-parameterized Key ($K(u) = h_u^{(l-1)} W_K^{\tau(u)}$), Query ($Q(v) = \tilde{h}_v^{(l-1)} W_Q^{\tau(v)}$), and Value ($V(u) = h_u^{(l-1)} W_V^{\tau(u)}$) projections along with the edge representation $\tilde{h}_v = h_v + e_{uv}'$. Crucially, attention scores scale by a learned per-meta-relation prior $\mu_{\langle \tau(u), \phi(e), \tau(v)\rangle}$ (`p_rel` in PyG), which lets the model weight an entire relation triple up or down independently of individual node embeddings; this parameter directly captures the relational typing effect evaluated in §7.2. Message passing operates bidirectionally across both forward and transposed relation views ($G_{\text{analysis}}$ and $G_{\text{analysis}}^\top$) to capture downstream starvation and upstream backpressure simultaneously, followed by residual aggregation, dropout ($p=0.10$), and layer normalization across layers $l \in \{1, \dots, L\}$.

#### Training Protocol and Optimization Hyperparameters

Models are optimized end-to-end with AdamW ($\eta = 3 \times 10^{-4}$, weight decay $10^{-4}$, post-attention dropout $0.10$) under cosine decay with warm restarts ($T_0 = 75$, $T_{\text{mult}} = 2$, $\eta_{\min} = 3 \times 10^{-6}$), for up to 300 epochs with early stopping at patience 30 on validation loss over labeled nodes. Inductive subgraphs are packed per scenario without mini-batch subsampling, and validation masks isolate held-out nodes. Five seeds $\{42, 123, 456, 789, 2024\}$ are evaluated throughout, redrawing partitions and initializations for each. The architectural hyperparameters ($D = 64$, $H = 4$, dropout, learning rate, schedule) follow conventional HGT values [72]; the loss coefficients of Equation 5 were set by informed judgment, there being no convention for a five-term multi-task loss. Neither was tuned against the in-distribution test split or the LOSO folds, and no search was run over them. This avoids selection leakage but does not place either configuration near its own optimum: the comparison is between untuned configurations, and we state it as such.

## 4.2 Multi-Task Prediction Heads and Dimension Masking

From the final node embeddings $h_v^{(L)}$, SaG utilizes specialized multi-task prediction heads:

-   **Reliability Head:** $\hat{R}(v) = \sigma(\text{MLP}_R(h_v)) \in [0, 1]$

-   **Maintainability Head:** $\hat{M}(v) = \sigma(\text{MLP}_M(h_v)) \in [0, 1]$

-   **Composite Failure Impact Head:** $\hat{I}^*(v) = \sigma(\text{MLP}_C(h_v \parallel \hat{R}(v) \parallel \hat{M}(v))) \in [0, 1]$

-   **Relationship Criticality Head:** $\hat{Q}(u,v) = \sigma(\text{TypedEdgeEncoder}_{\phi(e)}(h_u, h_v, e_{uv})) \in [0, 1]$. Disabled throughout the evaluation reported here; every harness instantiates the model with edge prediction switched off, so $\hat{Q}(u,v)$ is neither trained nor scored. Described for completeness.

### 4.2.1 Dimension-Masked Loss Formulation

The combined optimization objective integrates regression accuracy, multi-task dimension learning, ranking fidelity, and pairwise ordering:

$$\tag{5}
\mathcal{L} = \mathcal{L}_{\text{composite}} + 0.5 \cdot \mathcal{L}_{\text{dimension}} + 0.3 \cdot \mathcal{L}_{\text{rank}} + 0.1 \cdot \mathcal{L}_{\text{pairwise}} + \lambda_{\text{RM}} \cdot \mathcal{L}_{\text{consistency}}$$

where $I^*(v)$ is the simulated cascade impact defined by the primary oracle (§4.3), $\mathcal{L}_{\text{composite}} = \text{MSE}(\hat{I}^*(v), I^*(v))$, $\mathcal{L}_{\text{rank}}$ is the ListMLE listwise ranking loss [83] parameterized by temperature $\tau$:

$$\tag{6}
\mathcal{L}_{\text{rank}} = -\frac{1}{N}\sum_{i=1}^N \left( \frac{\hat{s}_{\pi_i}}{\tau} - \log \sum_{j=i}^N \exp\left(\frac{\hat{s}_{\pi_j}}{\tau}\right) \right)$$

where $\pi = (\pi_1, \dots, \pi_N)$ denotes the permutation of nodes sorted in descending order of ground-truth impact $I^*(v)$, and $\hat{s}_v = \hat{I}^*(v)$. At the baseline default $\tau = 1.0$, the formulation reduces to standard ListMLE; the temperature parameter $\tau < 1.0$ is a configurable hyperparameter that sharpens probability distributions over narrow prediction margins. Pairwise ordering fidelity is guided by margin-ranking loss $\mathcal{L}_{\text{pairwise}} = \frac{1}{|P|} \sum_{(u,v) \in P} \max\big(0, \gamma - (\hat{s}_u - \hat{s}_v)\big)$ with margin $\gamma = 0.05$ over pairs $P = \{(u, v) \mid I^*(u) - I^*(v) > \gamma\}$, and $\mathcal{L}_{\text{consistency}} = \text{MSE}\big([\hat{R}(v), \hat{M}(v)]_{v \in \text{unlabeled}}, [R_{\text{RM}}(v), M_{\text{RM}}(v)]_{v \in \text{unlabeled}}\big)$ regresses predicted heads toward the diagnostic pathway’s baseline on unlabeled nodes, where $R_{\text{RM}}(v)$ and $M_{\text{RM}}(v)$ denote the deterministic Reliability and Maintainability scores from the explanation layer (§5). Headline results use $\lambda_{\text{RM}} = 0$, guaranteeing that the predictive and elucidative pathways remain strictly independent.

The coefficients in Eq. 5 ($0.5$ dimension, $0.3$ listwise rank, $0.1$ pairwise margin) balance composite regression with relative node ordering. Permutation-level ListMLE ($\mathcal{L}_{\text{rank}}$) provides the primary gradient force for global rank monotonicity ($\rho$), while pairwise margin ($\mathcal{L}_{\text{pairwise}}$) penalizes small-margin inversions among adjacent components. Across all seeds, gradient norms remain well-conditioned, preventing individual objectives from overpowering optimization.

**Dimension Masking and Head Roles:** Because dynamic cascade simulation ($I^*(v)$ via discrete-event cascade fault injection) observes runtime failure reachability rather than source-code maintainability, maintainability ground truth is unobserved during dynamic simulation. A separate change-propagation oracle $I_M(v)$ evaluates static structural change ripple at the Validate stage, but is never used as a training label to avoid circular supervision. We introduce a boolean dimension mask $m = [m_R, m_M] = [1, 0]$:

$$\tag{7}
\mathcal{L}_{\text{dimension}} = \frac{1}{\sum_{d} m_d} \sum_{d \in \{R, M\}} m_d \cdot \text{MSE}(\hat{d}(v), d^*(v))$$

This mask ensures the unobserved maintainability head is not artificially penalized or driven toward zero during backpropagation.

**Auxiliary Nature of the Reliability Head and Active Loss:** Under the headline experimental protocol, this general formulation simplifies significantly. With $\lambda_{\text{RM}} = 0$ (preserving strict pathway independence) and dimension mask $m = [1, 0]$, the cascade fault injection oracle emits a single scalar assigned to both targets: $R^*(v) = I^*(v)$ identically. $\mathcal{L}_{\text{dimension}}$ regresses $\hat{R}$ toward the same target as $\mathcal{L}_{\text{composite}}$. The effectively optimized loss reduces to:

$$\tag{8}
\mathcal{L}_{\text{active}} = \text{MSE}(\hat{I}^*(v), I^*(v)) + 0.5 \cdot \text{MSE}(\hat{R}(v), I^*(v)) + 0.3 \cdot \mathcal{L}_{\text{rank}} + 0.1 \cdot \mathcal{L}_{\text{pairwise}}$$

Here, $\hat{R}$ re-enters the composite head as an input ($\hat{I}^* = \sigma(\text{MLP}_C(h_v \parallel \hat{R} \parallel \hat{M}))$), functioning as a feature-enrichment auxiliary pathway rather than independent multi-task supervision. We report the objective as implemented rather than claiming multi-dimensional supervisory ground truth.

### 4.2.2 Domain-Reweighted Criticality

ISO/IEC 25019’s Context of Use specifies that the relative weighting of reliability and maintainability is determined by deployment requirements rather than being fixed. The framework delivers a reweighting $\hat{Q}_{\text{domain}}(v) = q_R \hat{R}(v) + q_M M_{\text{static}}(v)$ to capture this flexibility. Because maintainability is unobserved during dynamic simulation ($m = [1,0]$), headline results report $\hat{I}^*(v)$ directly; sensitivity relative to the static RM baseline is evaluated in §7.3.

## 4.3 Ground-Truth Simulation Oracles

To evaluate predictive accuracy before deployment without relying on production runtime telemetry, SaG executes discrete-event failure simulations over the raw structural multigraph $G_{\text{structural}}$. Table 3 summarizes the four component-level oracles and their distinct roles across the evaluation program.

**Table 3.** Simulation oracles, operational constructs, and evaluation roles.

| **Oracle**           | **Physical Mechanism**                         | **Nature**    | **Role in Evaluation**                 |
|:---------------------|:-----------------------------------------------|:--------------|:---------------------------------------|
| $I^*(v)$             | BFS cascade reachability + QoS ladder          | Deterministic | Primary ranking target (RQ1, RQ2, RQ4) |
| $I_{\text{comp}}(v)$ | Severity mixture: reachability + fragmentation | Deterministic | Explanation layer / Validate gate      |
| $I_{\text{dyn}}(v)$  | Discrete-event SimPy message queuing           | Stochastic    | Convergent-validity probe (RQ3)        |
| $I_M(v)$             | Reverse `DEPENDS_ON` traversal                 | Deterministic | Unsupervised maintainability reference |

-   **Cascade Reachability Oracle ($I^*(v)$)**, evaluated via discrete-event cascade fault injection: crashes component $v$, propagates outages across dependent topics, brokers, and links by breadth-first traversal, and returns the mean fractional feed loss over the subscriber population, each subscriber contributing the unweighted mean loss of the topics it subscribes to. A topic’s feed loss is the fraction of its publishers that have failed (for a topic with no publisher, the fraction of its failed routers), scaled by a QoS ladder ($\times 1.2$ `RELIABLE`, $\times 1.15$ high/urgent priority, $\times 1.05$ medium) and clamped to $[0,1]$. The denominator is the full subscriber set of the intact graph, so subscribers that themselves fail are retained in the average rather than excluded. The implementation admits a per-publisher rate weighting, but rates are declared per topic throughout our corpus, so it reduces exactly to this publisher fraction. This is the **primary continuous target label** throughout.

    *How much QoS is in this label.* The ladder reads reliability and transport priority only; durability never enters $I^*$, despite carrying the largest of the three elicited sub-weights ($0.62$; §3.2). That omission does not bound the label’s QoS content, because disabling QoS scaling entirely leaves the Application ordering nearly intact — mean Spearman $\rho = 0.965$ against the ladder across the twelve folds (range $0.891$–$0.999$) — and substituting a durability-aware $w(t)$ scaling moves it less still ($\rho = 0.977$). The top-$K$ set is the sensitive construct: ladder and topology-only labels agree at mean Jaccard $0.678$, so QoS changes *which* components are named critical without changing their order. $I^*$ is therefore a near-topological target carrying a QoS term at its threshold boundaries, which bounds what any QoS-encoding result can be crediting (§7.3.1).

-   **Multi-Metric Composite Oracle ($I_{\text{comp}}(v)$)**, evaluated via multi-metric failure simulation: a severity-weighted mixture of reachability loss, fragmentation, throughput loss and flow disruption, with AHP-derived coefficients $(0.35, 0.25, 0.25, 0.15)$. Those coefficients come from a rank-one comparison matrix, so they record their origin without independently justifying them. They are not swept in our sensitivity analysis — a gap worth naming because $I_{\text{comp}}$ supplies the labels for the explanation layer’s evaluation. It is reserved for Validate-stage gates and prescriptive verification, never for forecasting ranking.

-   **Dynamic Queue-Flow Oracle ($I_{\text{dyn}}(v)$)**, evaluated via discrete-event message-flow queue simulation (built on SimPy [84]): simulates emission rates, stochastic latencies, broker buffer saturation, and queue drops under fault injection, extracting the drop in delivered message rate to surviving consumers. The engine instruments Google SRE’s *Four Golden Signals* across pre- and post-fault windows — latency decomposed into queue wait and service time at p50/p95/p99, traffic in Hz and KB/s, errors spanning deadline violations and overflow discards, and time-weighted mean queue depth for saturation — as diagnostic telemetry. They are kept out of $I_{\text{dyn}}(v)$ rather than mixed into it, because crashing a high-rate publisher clears downstream subscriber queues (*contention relief*, $\rho = -0.499$ between delivery loss and tail latency delta). An additive composite would mathematically cancel delivery damage with latency reduction. $I_{\text{dyn}}(v)$ is therefore kept strictly 1-dimensional, serving as an independent convergent-validity probe (§7.3.2).

-   **Change-Propagation Oracle ($I_M(v)$)**, evaluated via structural change-propagation analysis: a deterministic reverse-dependency traversal over the transpose of the six-rule `DEPENDS_ON` projection, blending change reach, weighted change impact, and normalized depth. It is a structural maintainability reference and is never used as a training label, which would make the supervision circular.

#### Topic Criticality Label Masking

The multi-metric failure simulator can incorporate declared topic criticality into its severity term. This feature is disabled, however, because topic criticality is a GNN input feature: using it would result in the predictor being measured against a transformation of its own input.

**Primary Oracle Declaration and Role Assignment.** The three reliability-facing oracles measure separate constructs, so we designate **$I^*(v)$ the primary continuous target** for all predictive ranking results (Tables 6–7, RQ1–RQ3): it has zero seed-to-seed variance, which is what deterministic CI/CD gating requires, whereas $I_{\text{dyn}}$ adds stochastic latencies, bursty arrivals and synthetic buffer limits. That choice has a cost, since it makes the target a topological functional of the predictors’ own input (§4.4). $I_{\text{comp}}(v)$ is reserved for Validate-stage gates and prescriptive verification, $I_{\text{dyn}}(v)$ is an independent convergent-validity probe (§7.3.2), and $I_M(v)$ a structural maintainability reference.

**Cross-Oracle Convergent Validity.** As detailed in §7.3.2, the three reliability oracles show substantial but sub-ceiling agreement on Applications ($\rho = 0.627$ for $(I_{\text{dyn}}, I^*)$ against a $0.811$–$1.000$ label noise floor), confirming distinct constructs. Consequently, results established against one oracle are never transferred to another; every evaluation explicitly references its underlying simulation oracle.

## 4.4 Input–Label Independence Guarantee

To prevent data leakage, SaG applies strict architectural separation. Feature Space is constructed exclusively from $G_{\text{analysis}}$ using static structural topology, static code metrics, and declared QoS contracts. Label Space is evaluated exclusively on raw $G_{\text{structural}}$ through independent simulation oracles (cascade reachability injection, composite failure simulation, and dynamic message-flow simulation). No simulation outputs, failure trace histories, or dynamic execution telemetry are exposed as input attributes to the GNN or the explanation layer.

### What this guarantee does and does not establish

The separation rules out circular feature construction: no predictor can read a transformation of the quantity it is scored against. It does not establish probabilistic independence between features and labels, and we do not claim it does. $G_{\text{analysis}}$ is a deterministic projection of $G_{\text{structural}}$ (§3.2). Hence, the labels are — up to the simulator seed — a deterministic function of the same topology from which the features are computed. Two consequences follow, and both bound the results of §7.

First, $I^*(v)$ is a topological functional, defined as a breadth-first reachability computation over $G_{\text{structural}}$ scaled by a QoS ladder. The predictive task is therefore to recover a closed-form graph function from features derived from the same graph. Because the 18-dimensional feature block provides pre-computed global centrality indicators, the GNN’s role is synthesizing these positional and structural cues across heterogeneous relations rather than discovering global topology from scratch. As a result, an unparameterized centrality score is expected to perform competitively with a trained model (§7.1). The observed parity is an anticipated outcome of the experimental design rather than an unexpected limitation of graph learning. We state this explicitly to guarantee clarity.

Second, and more restrictively, no result in this paper is validated against an observed failure. Every label — on synthetic topologies and on the five open-source systems alike — is simulator-derived. The evaluation can establish whether a learned model recovers a simulator’s ordering on architectures it was not trained on. Whether that ordering corresponds to which components actually fail in production is a question this design cannot answer, and §8.3 treats it as the study’s principal construct-validity threat.

# 5. The Explanation Layer: Standards-Grounded Criticality Attribution

The predictor of §4 locates risk but says nothing about remedy. A component may be critical because it is an unreplicated single point of failure, an error-propagating cascade hub, or a high-coupling maintainability bottleneck, and each calls for a different intervention: replication, a circuit breaker, or refactoring. This section formalizes the layer that attributes those causes, decomposing criticality into a standards-grounded quality profile computed from the same typed node properties (§3.4) without sharing parameters with the predictor, and reaching flagged components through triage rather than data flow (Figure 1).

**What this section is, and is not.** This layer is a design pattern, not a validated contribution, and it is deliberately absent from the list of contributions in §1.5. Three results bound it. Its standalone rank correlation is low ($\rho = 0.205$, below unweighted centrality on every fold, §7.1); its elicited AHP weights rank worse than a uniform prior, monotonically so as they approach raw elicited judgment (§7.3); and no human-subject study has tested whether its archetypes change what a developer does. What it offers is a worked mapping from topological properties to standardized ISO/IEC sub-characteristics, and an existence proof that such a mapping can be computed from a manifest. Whether the mapping is *right* is the open question, and §8.4 names the two experiments that would answer it.

## 5.1 Grounding in ISO/IEC Standards

In accordance with ISO/IEC 25010:2023 [13] and ISO/IEC 25019:2023 [14], SaG formalizes two primary criticality dimensions. Component Criticality ($D_1$) is service loss upon component failure; Relationship Criticality ($D_2$) is service decline upon channel severance.

Criticality is assessed across two orthogonal characteristics: **Reliability ($R$)** and **Maintainability ($M$)**. Reliability is divided into **Fault Tolerance ($FT$)** and **Availability ($A$)**. $FT$ uses Reverse PageRank, in-degree, and cascade depth potential to inform redundancy and circuit breaker strategies. $A$ uses directed articulation points, bridge ratios, and connectivity degradation to inform replication strategies. Maintainability ($M$) assesses structural coupling and code-level complexity, using betweenness, QoS-weighted fan-out, code quality penalties, and clustering to guide decoupling and refactoring. This partition maps each ISO/IEC sub-characteristic to its graph metrics and remediation roles. Safety and security considerations that require specialized hazard logs are excluded from purely structural topology analysis.

## 5.2 Composite Quality Score Formulation

All raw metrics are rank-normalized to the interval $[0, 1]$ within the graph. Quality sub-characteristics are formulated hierarchically using the Analytic Hierarchy Process (AHP) [63]:

-   **Fault Tolerance ($FT(v)$):** Evaluates error cascade potential on transpose graph $G_{\text{analysis}}^\top$: $FT(v) = 0.45 \cdot \text{RPR}(v) + 0.30 \cdot \text{Deg}_{\text{in}}(v) + 0.25 \cdot \text{CDPot}_{\text{enh}}(v)$, where $\text{RPR}(v)$ is Reverse PageRank, $\text{Deg}_{\text{in}}(v) = d_{\text{in}}(v)/(|V|-1)$ is normalized in-degree, and $\text{CDPot}_{\text{enh}}(v) = \text{depth}(v) / \max_{u \in V} \text{depth}(u)$ is normalized cascade depth potential on $G_{\text{analysis}}^\top$.

-   **Availability ($A(v)$):** Identifies structural single points of failure across five terms: $A(v) = 0.25 \cdot \text{AP}_c^{\text{dir}}(v) + 0.20 \cdot \text{QSPOF}(v) + 0.20 \cdot \text{BR}(v) + 0.25 \cdot \text{CDI}(v) + 0.10 \cdot w(v)$, where $\text{AP}_c^{\text{dir}}(v)$ is Directed Articulation Point severity, $\text{QSPOF}(v)$ is QoS-weighted SPOF severity, $\text{BR}(v)$ is Bridge Ratio, $\text{CDI}(v)$ is Connectivity Degradation Index, and $w(v)$ is intrinsic QoS weight.

-   **Reliability ($R(v)$):** Blends Fault Tolerance and Availability: $R(v) = r_{\text{FT}} \cdot FT(v) + (1 - r_{\text{FT}}) \cdot A(v)$ with $r_{\text{FT}} = 0.36$. Intra-dimension weights apply $\lambda = 0.70$ shrinkage blending with a uniform prior. Three of the framework’s five comparison matrices (Impact, Maintainability, Availability) are rank-one by construction, so their Consistency Ratios are uninformative; the remaining two (Topic QoS and Fault Tolerance) carry genuine second-eigenvalue spread and their $CR$ figures do mean what $CR$ normally means. Supplementary §S4 separates them. Elicited AHP weights rank worse than a uniform prior against dynamic simulation (§7.3), and we recommend the uniform prior. The shipped default nonetheless remains $\lambda = 0.70$, because the same constant parameterises the $I_{\text{comp}}$ severity weights: changing it would re-label the composite oracle and with it every figure scored against that oracle. Decoupling the two is a prerequisite for changing the default, not a reason to defend it.

-   **Maintainability ($M(v)$):** Blends structural coupling with static code analysis: $M(v) = 0.35 \cdot \text{BT}(v) + 0.30 \cdot w_{\text{out}}(v) + 0.15 \cdot \text{CQP}(v) + 0.12 \cdot \text{CouplingRisk}_{\text{enh}}(v) + 0.08 \cdot (1 - \text{CC}(v))$, where $\text{BT}(v)$ is Betweenness Centrality, $w_{\text{out}}(v)$ is QoS-weighted efferent coupling, $\text{CQP}(v)$ is Code Quality Penalty, and $\text{CC}(v)$ is local Clustering Coefficient.

The baseline composite quality score integrates both dimensions as follows: $Q(v) = 0.80 \cdot R(v) + 0.20 \cdot M(v)$. When evaluated under an ISO/IEC 25019 Context of Use vector $\vec{\omega} = [q_R, q_M]^\top$, the score is dynamically reweighted: $Q_{\text{domain}}(v) = q_R \cdot R(v) + q_M \cdot M_{\text{static}}(v)$. Components are partitioned into Tukey tiers: CRITICAL ($Q > Q_3 + 1.5 \cdot \text{IQR}$), HIGH, MEDIUM, and MINIMAL. Across the benchmark topologies, this conservative Tukey upper fence flags an empirical mean of $4.2\%$ of components (range $1.8\%$–$8.3\%$), deliberately isolating the extreme right tail of architectural risk to prioritize developer intervention. High Availability ($A$) combined with low Fault Tolerance ($FT$) indicates a single point of failure that necessitates replication. In contrast, high Fault Tolerance ($FT$) identifies an error-cascade hub that requires circuit breakers (§8.4). Supplementary Table S16 shows a representative Diagnostic Remediation Card, illustrating how the sub-characteristics translate topological metrics into refactoring actions.

## 5.3 Prescriptive Remediation and Counterfactual Verification

After attributing root causes, candidate repairs (broker replication, circuit breaker insertion, topic decoupling) are generated and counterfactually verified in memory. Mutations are accepted only if they reduce systemic impact beyond simulation seed noise ($\Delta \bar{I}_{\text{comp}} > \kappa \cdot \sigma_{\text{seed}}$, $\kappa \ge 1.0$) without introducing new articulation points. This counterfactual verification loop illustrates the architectural pattern linking diagnosis to remediation; formal developer studies and automated patch synthesis benchmarks are reserved for future work.

# 6. Experimental Setup

## 6.1 Datasets and System Corpus

The evaluation corpus comprises 2,812 components across seventeen system architectures, including twelve synthetic topologies that form the inductive cross-validation folds and five real-world reference systems withheld from all training procedures, as detailed in Table 4. The synthetic scenarios span diverse operational domains (autonomous vehicles, financial trading, healthcare integration, industrial SCADA, smart-city IoT, telecom RAN, cloud microservices, and enterprise application integration via centralized broker hubs/ESB; detailed in Supplementary Table S12).

**Table 4.** Overview of the evaluation corpus. The twelve synthetic topologies correspond to the inductive Leave-One-Scenario-Out folds described in Table 6, and the five real-world systems are excluded from all training folds and used exclusively for zero-shot transfer (§7.4). Per-scenario entity and edge counts are obtained from the committed topology files and verified through continuous integration.

| **Dataset**                            | **$|V|$** | **$|V_{\text{app}}|$** | **Topics** | **Brokers** | **Hosts** | **Libs** |  **$|E|$** |
|:---------------------------------------|----------:|-----------------------:|-----------:|------------:|----------:|---------:|-----------:|
| Synthetic evaluation scenarios (11)    |     2,387 |                  1,295 |        588 |          60 |       194 |      250 |     10,657 |
| Synthetic case study — ATM (1)         |        74 |                     26 |         27 |           5 |         8 |        8 |        261 |
| **Synthetic subtotal (12 LOSO folds)** | **2,461** |              **1,321** |    **615** |      **65** |   **202** |  **258** | **10,918** |
| **Real-world subtotal (5 systems)**    |   **351** |                **141** |    **120** |      **16** |    **32** |   **42** |    **700** |
| **Total**                              | **2,812** |              **1,462** |    **735** |      **81** |   **234** |  **300** | **11,618** |

### 6.1.1 Reproducibility of the Corpus

The benchmark corpus is designed to be fully regenerable rather than statically archived. Each dataset is deterministically generated from its configuration file via `python cli/generate_graph.py batch –input-dir data/scenarios –output-dir <path>`. A companion manifest records the random seed, entity counts, git commit hash, and a SHA-256 cryptographic digest for each emitted topology. Continuous integration regression tests verify that every committed dataset regenerates byte-identically from its configuration and that all disk digests match the manifest. This procedure makes sure that third parties can reproduce the exact graphs used in these experiments, rather than sampling from similar distributions.

## 6.2 Baselines and Evaluated Predictors

Four primary predictor configurations, drawn from three distinct families, are evaluated alongside reference baselines. Table 5 provides a structured taxonomy of the evaluated predictors, their underlying graph representations, edge feature encodings, and empirical roles. Predictor names indicate the model family and substrate: an `-N` infix denotes a model trained on the complete native multigraph, its absence denotes the obtained Application–Library flow projection, and a `-QoS` suffix denotes a configuration that consumes declared QoS contracts. SaG throughout denotes the overall framework, not an individual predictor.

**Table 5.** Taxonomy of evaluated predictors and reference baselines. The `-N` infix denotes execution on the complete native multigraph; `-QoS` indicates inclusion of middleware QoS attributes.

| **Predictor**   | **Evaluation Substrate**                |   **Typing**   | **Edge Features** | **Parameters** | **Trained?** | **Empirical Role**                                         |
|:----------------|:----------------------------------------|:--------------:|:-----------------:|:--------------:|:------------:|:-----------------------------------------------------------|
| **Topo**        | $G_{\text{analysis}}$ (Flow Projection) |       No       |       None        |       0        |      No      | Unweighted topological baseline                            |
| **Topo-QoS**    | $G_{\text{analysis}}$ (Flow Projection) |       No       |   Scalar $w(e)$   |       0        |      No      | QoS-weighted closed-form benchmark                         |
| **GAT**         | $G_{\text{analysis}}$ (Flow Projection) |  Homogeneous   |       None        |     28,168     |     Yes      | In-distribution counterpart of GAT-N (Supp. Table S15)     |
| **GAT-QoS**     | $G_{\text{analysis}}$ (Flow Projection) |  Homogeneous   |   Scalar $w(e)$   |     28,168     |     Yes      | In-distribution counterpart of GAT-N-QoS (Supp. Table S15) |
| **GAT-N**       | Native Multigraph                       |  Homogeneous   |       None        |     28,168     |     Yes      | Untyped, unweighted GNN floor                              |
| **GAT-N-QoS**   | Native Multigraph                       |  Homogeneous   |   Scalar $w(e)$   |     28,168     |     Yes      | Isolates QoS channel without typing                        |
| **HGT**         | Native Multigraph                       | Heterogeneous  |  Relation 1-hot   |    434,620     |     Yes      | Isolates relational typing without QoS                     |
| **HGT-QoS**     | Native Multigraph                       | Heterogeneous  |  16-D QoS Vector  |    434,620     |     Yes      | Proposed full learned model                                |
| **RM / $Q(v)$** | $G_{\text{analysis}}$ (Analysis Graph)  | Per-type rules |   Scalar $w(e)$   |       0        |      No      | Diagnostic attribution reference                           |

Table 5 summarizes the evaluated predictors, graph representations, and empirical roles across three families: heterogeneous graph learning (`HGT-QoS` with 16-D QoS edge vectors, and its ablation `HGT`), homogeneous graph learning (`GAT-N-QoS` with scalar $w(e)$, and unweighted `GAT-N`), and training-free structural baselines (`Topo-QoS` and unweighted `Topo` on the `DEPENDS_ON` flow projection). The `-N` infix denotes the native multigraph substrate; on flow projections, homogeneous variants are denoted `GAT`/`GAT-QoS` (Supplementary Table S15). The out-of-distribution evaluation (Table 6) additionally reports **RM** ($Q(v)$, §5) as a diagnostic reference baseline. Deterministic RM scoring also drives sensitivity sweeps in §7.3.

### 6.2.1 Evaluation Substrates and Parity Guarantee

Predictors in this study operate on matched substrates within their respective families:

**Graph Learning Models (`GAT-N-QoS`, `HGT-QoS`):** Under Leave-One-Scenario-Out (Table 6), both learned predictors ingest the complete native typed multigraph across all five entity types (recorded using the shared `-N` infix). In-distribution (Supplementary Table S15), GAT/GAT-QoS consume the Application–Library `DEPENDS_ON` projection, confounding typing with multi-entity visibility. `GAT-N-QoS` uses per-type projection with untyped GATConv across edges, whereas `HGT-QoS` uses relation-specific `HGTConv` weights. The edge channel also differs: `GAT-N-QoS` consumes a scalar $w(e)$, while `HGT-QoS` consumes all 16 dimensions. The parameter budget ($434{,}620$ vs. $28{,}168$) and directionality also remain open confounds (§8.4).

**Training-Free Structural Baselines (Topo, `Topo-QoS`):** We evaluate topological baselines on the obtained Application–Library `DEPENDS_ON` projection (§3.2), since raw multigraphs route messages via topics/brokers, leaving Application nodes with near-zero betweenness. Because these two baselines carry every headline contrast in §7, we state them in closed form rather than by name. Both are convex combinations of a path-traversal term and a cut-vertex term:

$$\text{Topo}(v) = 0.6 \cdot \text{BT}(v) + 0.4 \cdot \text{AP}(v),
\qquad
\text{Topo-QoS}(v) = 0.6 \cdot \text{BT}_{w}(v) + 0.4 \cdot \text{AP}(v)$$

where $\text{BT}(v)$ is normalized betweenness centrality on the projection, $\text{AP}(v) \in \{0, 1\}$ indicates whether $v$ is an articulation point of the projection’s undirected form, and $\text{BT}_{w}(v)$ is betweenness computed over edge *distances* $d(e) = 1/(w(e) + \varepsilon)$ with $\varepsilon = 10^{-6}$, so that strongly coupled edges are short and attract shortest paths. The projection itself carries two of the six rules of Table 2: Rule 1 joins a subscriber to each publisher of a topic it consumes, at $w = 1 - \prod_{t \in T}(1 - w(t))$, and Rule 5 joins an application to each library it uses, at the median topic weight.

Two properties of this pair matter for how §7 should be read. First, *only the traversal term is QoS-weighted*: the articulation term is identical in both, so the entire `Topo-QoS` margin is carried by re-weighting shortest paths. Second, the weighting *degenerates gracefully*: when no edge of a graph carries a non-unit weight, $\text{BT}_{w}$ reduces to $\text{BT}$ and `Topo-QoS` coincides with Topo on that graph. The $0.6/0.4$ split is a declared convention, not a fitted parameter, and is not tuned per scenario.

**Ground-Truth Independence Guarantee:** Crucially, **no predictor in either group accesses $G_{\text{structural}}$**: that raw topology is strictly reserved for simulation oracles (§4.4), verified by `tests/test_independence_guarantee.py`. Regardless of substrate, all variants are scored on the same independently resolved Application node set ($V_{\text{app}}$, §6.3).

## 6.3 Evaluation Metrics and Protocols

**Figure accessibility.** All figures employ the Okabe–Ito palette with distinct markers and hatchings to ensure monochrome legibility.

**Ranking Precision:** Evaluated via Spearman $\rho$ against the simulated impact $I^*(v)$ from the primary oracle (§4.3). **Critical-Set Identification:** Measured via $F_1@K$ for top-$K$ components ($K = \text{round}(0.20 \cdot |V_{\text{app}}|)$). Because both the predicted and reference sets have exactly $K$ members, precision, recall and $F_1@K$ coincide identically: the column is top-$K$ set overlap, not an $F_1$ score, and it is reported under that reading everywhere it appears. Identification at operating points where precision and recall are free to differ — $F_1@\tau$ against the labels’ own critical set, threshold-free PR-AUC, and rank-weighted nDCG@10 — is reported separately in Supplementary Table S13, because those quantities answer a question $F_1@K$ structurally cannot. **Statistical Significance:** Paired Wilcoxon signed-rank tests [85] ($p < 0.05$) and bootstrap 95% CIs ($B = 2{,}000$) over folds [86, 87]. In the 12-fold LOSO design, the power floor is $p = 0.00049$.

**One confirmatory family, and everything else exploratory.** The confirmatory family is the pre-registered one and contains two contrasts: `HGT-QoS` against `Topo-QoS`, and `HGT` against `Topo-QoS`, Holm-corrected across those two alone. Neither reaches $\alpha = 0.05$ (§7.1). Every other contrast in this paper — the remaining full-population comparisons against `Topo-QoS`, the three orthogonal quantities of the $2\times2$ (Table 8), and the four simple effects — was formulated after the results existed and is reported as exploratory. Earlier versions of this manuscript corrected two overlapping families, which made the same contrast survive under one correction and not the other; the $2\times2$ quantities are still Holm-corrected within their own block, and that block is labelled post-hoc where it appears. For orientation, the largest exploratory effects are `Topo-QoS` over Topo ($+0.204$, 12/12, $p = 0.0005$) and unweighted typing, `HGT` over `GAT-N` ($+0.234$, 12/12, $p = 0.0005$), while the two contrasts that carry the QoS channel into an already-typed model do not separate: typing with the QoS channel present reaches $p = 0.1294$ (9/12 folds) and the QoS edge ablation under typing $p = 0.2036$ (10/12).

**Registered analysis plan.** The primary out-of-distribution contrast (`HGT-QoS` vs. `Topo-QoS` under LOSO, 5 fixed seeds, fold as unit of analysis) was registered in the replication package, with its protocol, statistic, unit of analysis and reporting commitment fixed, before the revised harness produced any result. We describe it as registered rather than pre-registered, and the distinction is not cosmetic: the plan is a file in our own repository with no third-party timestamp, and a prior eight-fold run of the same contrast — since withdrawn, because it reproduced from no commit — predates it and is what motivated writing the plan down. What the registration establishes is that the analysis was not selected after seeing the twelve-fold result; what it cannot establish is that the question was asked in ignorance of any earlier estimate. As reported in §7.1, the margin did not reach statistical significance, which is the outcome the plan committed to reporting.

### Evaluation Population and Protocols

Each predictor within an evaluation table is scored on an identical node population, resolved strictly from scenario topology and ground truth, specifically the **Application** set ($V_{\text{app}}$) unless otherwise noted. Pooling node types conflates distinct base rates and can trigger Simpson’s paradox (§7.3).

**In-Distribution Evaluation:** Stratified 60% train / 20% val / 20% test node splits over five seeds $\{42, 123, 456, 789, 2024\}$, redrawing partitions and initializations. **Inductive LOSO Cross-Validation:** Models are trained on eleven scenarios and test zero-shot on the held-out twelfth across all 12 folds under equal 3-layer depth and inner-split early stopping (§8.4). **Real-World Architectural Transfer:** Synthetic-trained models are evaluated zero-shot on five open-source systems without fine-tuning.

# 7. Results and Empirical Analysis

Empirical results for RQ1–RQ5 are presented across the twelve-fold inductive benchmark and five authentic open-source distributed systems. The evaluated populations are strictly stratified on the Application service set ($V_{\text{app}}$) under the input–label independence guarantee (§4.4).

## 7.1 RQ1: Graph Learning vs. Structural Baselines

#### RQ1 Summary (Predictive Efficacy):

*`HGT-QoS` ($\rho = 0.638$) achieves highest numerical correlation but does not significantly outperform the unparameterized `Topo-QoS` baseline ($\rho = 0.553$, $\Delta\rho = +0.085$, $p = 0.151$, 95% CI spanning zero). Because ground-truth reachability $I^*(v)$ is a topological functional, closed-form QoS-weighted centrality captures broad failure propagation without training. Restricting evaluation to active components roughly halves correlation for every predictor, learned or not ($\rho_{>0} \in [0.102, 0.356]$ against $\rho \in [0.205, 0.638]$), showing that full-population correlation largely reflects separation of inert sink nodes.*

**What the in-distribution results can and cannot support.** Per-scenario in-distribution figures are reported in Supplementary Table S15 rather than here, because no comparison can be drawn down their columns: `GAT`/`GAT-QoS` consume the Application–Library projection while `HGT`/`HGT-QoS` consume the native multigraph (§6.2.1), so a difference between the families confounds message passing with multi-entity visibility. Within a predictor the cells are still informative. In Healthcare, `Topo-QoS` achieves $\rho = 0.399$ but fails at critical triage ($F_1 = 0.000$); in the synthetic Microservices fold, unaugmented `HGT` suffers degradation ($\rho = 0.141, F_1 = 0.300$), whereas `HGT-QoS` recovers both ranking monotonicity ($\rho = 0.664$) and top-$K$ overlap ($F_1@K = 0.600$).

### 7.1.1 Out-of-Distribution (LOSO) Generalization

Inductive Leave-One-Scenario-Out cross-validation asks each model to predict cascading criticality on an entirely unseen topology, and the twelve folds are this paper’s primary anchor for generalization across architectural archetypes. Per-fold breakdowns are in the Supplementary Material (Table S12, §S10); the main text reports the cross-fold summary (Table 6) and the active-stratum contrast (Table 7):

**Table 6.** Inductive LOSO evaluation, Application population, twelve folds. Learned variants share training set, the native multigraph substrate (hence the `-N` infix; §6.2), depth, and selection rule (§6.3), differing in typing and edge channel; parameter budget and message-passing directionality remain unmatched confounds that Table 8 controls for (§8.4). Fold score = mean over five seeds; **Fold $\sigma$** = spread of the twelve fold means; **Seed $\sigma$** = median within-fold spread (zero by construction for deterministic baselines); CI = bootstrap over folds ($B = 2{,}000$). **$\Delta\rho$** is paired by fold against `Topo-QoS`, the pre-registered comparator (§6.3), with its own bootstrap interval; an interval spanning zero means the contrast is not resolved at twelve folds, which is the case for every learned variant.

| **Predictor / Reference**                                            | **Mean LOSO $\rho$** |    **95% CI**    |   **$\Delta\rho$ vs `Topo-QoS`**    | **Fold $\sigma$** | **Seed $\sigma$** | **Critical-Set $F_1@K$** | **Requires Training** |
|:---------------------------------------------------------------------|:--------------------:|:----------------:|:-----------------------------------:|:-----------------:|:-----------------:|:------------------------:|:---------------------:|
| *Training-free structural baselines*                                 |                      |                  |                                     |                   |                   |                          |                       |
| **Topo**                                                             |        0.349         | $[0.254, 0.452]$ | -0.204 $[-0.286, -0.122]$ |       0.173       |         —         |          0.366           |          No           |
| **Topo-QoS**                                                         |        0.553         | $[0.443, 0.657]$ |            — (reference)            |       0.192       |         —         |          0.388           |          No           |
| *Learned predictors (shared native substrate, matched training set)* |                      |                  |                                     |                   |                   |                          |                       |
| **GAT-N**                                                            |        0.317         | $[0.254, 0.381]$ | -0.236 $[-0.342, -0.125]$ |     **0.111**     |       0.298       |          0.328           |          Yes          |
| **GAT-N-QoS**                                                        |        0.604         | $[0.538, 0.665]$ | $+$0.051 $[-0.067, +0.169]$ |       0.112       |     **0.024**     |        **0.431**         |          Yes          |
| **HGT**                                                              |        0.551         | $[0.474, 0.617]$ | -0.002 $[-0.066, +0.072]$ |       0.124       |       0.114       |          0.427           |          Yes          |
| **HGT-QoS**                                                          |      **0.638**       | $[0.561, 0.710]$ | $+$0.085 $[-0.029, +0.194]$ |       0.133       |       0.052       |          0.424           |          Yes          |
| *Diagnostic reference — not a ranking model*                         |                      |                  |                                     |                   |                   |                          |                       |
| **RM / $Q(v)$**                                                      |        0.205         | $[0.092, 0.320]$ | -0.348 $[-0.432, -0.265]$ |       0.195       |         —         |          0.322           |          No           |

Each of the twelve folds holds out one scenario for zero-shot testing and trains on the remaining eleven, with all variants scored on the same Application node set (26 to 300 nodes, giving $K$ between 5 and 60) and paired Wilcoxon tests over folds. For $F_1@K$, `HGT-QoS` achieved $0.424$ compared to `Topo-QoS`’s $0.388$ ($\Delta = +0.037$, prevailing in 7 of 12 folds, $W = 29.0$, $p = 0.470$). The untyped `GAT-N-QoS` attained $0.431$ ($\Delta = -0.006$, prevailing in 5 of 12 folds, 1 tie, $W = 27.5$, $p = 0.653$), indicating that critical-set identification does not statistically distinguish the typed model from either untyped learning or the QoS-weighted baseline.

**Label-noise ceiling.** No predictor can exceed the reproducibility of its own labels. Re-running the oracle across the five seeds gives a test–retest rank correlation of $0.811$–$1.000$ (median $0.982$), so `HGT-QoS`’s $\rho = 0.638$ recovers roughly $65\%$ of the attainable signal. Top-$K$ sets are far noisier — cross-seed Jaccard median $0.847$, falling to $0.370$ on Logistics Fleet — which is why the $F_1@K$ margins are less stable than the ranking ones. The least reproducible fold (Microservices, $0.811$) is not one the typed model loses; on this corpus the low-ceiling folds and the lost folds are disjoint.

**Key Insights concerning RQ1:**

1.  **Typed learning is the best configuration, but not demonstrably better than the QoS baseline.** `HGT-QoS` leads all predictors out-of-distribution ($\rho = 0.638$). Against training-free `Topo-QoS` it is $+0.085$ (9/12, $W = 20.0$, $p = 0.151$, CI $[-0.029, +0.194]$), an interval that includes zero; un-augmented HGT is indistinguishable from the baseline outright ($-0.002$, 3/12, $p = 0.470$). Neither pre-registered contrast reaches significance, and we report that as the answer rather than as a near miss.

2.  **A QoS-weighted structural score is a genuinely strong baseline — and untyped learning is worse than it.** `Topo-QoS` reaches $\rho = 0.553$ zero-shot, beating unweighted Topo on all twelve folds ($+0.204$, $p = 0.0005$). More pointedly, the untyped, unweighted learned model loses to it decisively (`GAT-N`, $-0.236$, 2/12, $p = 0.0024$). On this task, a homogeneous graph network trained on eleven architectures does not reach what a closed-form centrality score achieves with no training at all. Any claim that graph learning is required must be made against this baseline.

3.  **Critical-set identification does not favor the typed model.** On $F_1@K$, `HGT-QoS` scores $0.424$ against `GAT-N-QoS`’s $0.431$ and HGT’s $0.427$ — a three-way tie within noise — while all three numerically lead `Topo-QoS` ($0.388$), though without statistical significance throughout folds ($\Delta = +0.037$, $p = 0.470$).

4.  **Power is not the limiting factor.** At $n = 12$, the design tolerates four lost folds and still reaches $\alpha = 0.05$, provided the losses are smallest in magnitude. `HGT-QoS`’s are not: it loses Enterprise ($-0.335$) and Telecom RAN ($-0.169$) to `Topo-QoS` by the two largest margins in the set, which is what holds $W$ at $20.0$. Enlarging the corpus will not resolve this; we must understand the inversions instead (§7.2.1).

5.  **The explanation layer is weakly predictive, not noise.** RM/$Q(v)$ reaches $\rho = 0.205$, losing to unweighted Topo on every fold ($-0.144$), so no ranking claim is made for it. Its interval $[0.092, 0.320]$ stays above zero, and it supplies interpretable diagnostics without training (§5). Table 6 lists it as a reference point, not a competitor.

### 7.1.2 The Active Stratum: Ranking Quality Separated From Inertness Detection

Between $21\%$ (Microservices) and $52\%$ (Healthcare) of each held-out Application population carries exactly zero simulated impact, so a predictor can score well by separating components that can propagate a failure from those that cannot, without ordering the propagating ones correctly. These are different capabilities, so we re-score all twelve folds on the *active stratum* — the $n_{>0}$ components with strictly positive impact — using the same predictions, folds and seeds. Table 7 reports both.

**Table 7.** LOSO means over twelve folds on the full Application population ($\rho$) and restricted to components with strictly positive ground-truth impact ($\rho_{>0}$). Same predictions, folds, and seeds; only the evaluated subset differs. Retained is defined as the active-stratum ratio $\rho_{>0}/\rho$ (percentage of full-population correlation preserved).

| **Predictor**   | **$\rho$ (full)** | **$\rho_{>0}$ (active)** | **Retained** |
|:----------------|:-----------------:|:------------------------:|:------------:|
| **RM / $Q(v)$** |      $0.205$      |         $0.102$          |    $49\%$    |
| **Topo**        |      $0.349$      |         $0.181$          |    $52\%$    |
| **Topo-QoS**    |      $0.553$      |         $0.280$          |    $51\%$    |
| **GAT-N**       |      $0.317$      |         $0.159$          |    $50\%$    |
| **GAT-N-QoS**   |      $0.604$      |         $0.328$          |    $54\%$    |
| **HGT**         |      $0.551$      |         $0.299$          |    $54\%$    |
| **HGT-QoS**     | $\mathbf{0.638}$  |     $\mathbf{0.356}$     |    $56\%$    |

Two consequences follow.

1.  **The restriction costs every predictor roughly half its correlation, learned or not.** Retained fractions run from $49\%$ (RM) to $56\%$ (`HGT-QoS`) with no systematic separation between the training-free and learned families (`Topo-QoS` $51\%$, `GAT-N` $50\%$, HGT $54\%$). Roughly half of every predictor’s full-population correlation reflects separating inert components from active ones, a property of the label distribution rather than a discriminator between methods.

2.  **The method ordering is unchanged, and so are the verdicts.** On the active stratum `HGT-QoS` still leads ($\rho_{>0} = 0.356$), ahead of `Topo-QoS` ($0.280$) and `GAT-N-QoS` ($0.328$). Nothing in §7.1 or §7.2 turns on whether zero-impact components are included; we report both columns because they answer alternative operational questions.

## 7.2 RQ2: Value of Typed Heterogeneity

#### RQ2 Summary (Typing vs. QoS):

*Relational typing and continuous QoS edge encodings act as empirical substitutes rather than complements. While each mechanism individually exhibits a significant main effect over unweighted baselines ($+0.134$ and $+0.187$, Holm $p = 0.0015$), their factorial interaction is sub-additive ($\Delta\rho = -0.199$, negative on all 12 folds, $p = 0.0005$). In pub-sub architectures, QoS contracts align with communication relations; thus, edge features and relation-specific weights supply redundant traversal indicators, yielding diminishing returns when combined.*

Whether relation typing improves prediction over homogeneous message passing depends entirely on the comparator it is ablated against:

-   **Both factors have a real main effect.** Averaged over the other factor’s levels, relation typing is worth $\Delta\rho = +0.134$ (12/12 folds, $p = 0.0005$, Holm $0.0015$) and the QoS edge channel $+0.187$ (11/12, Holm $0.0015$), subject to the capacity and directionality confounds below.

-   **But they interact, strongly and sub-additively.** The difference of differences — what typing buys with the QoS channel present, minus what it buys without — is $-0.199$, negative on all twelve folds ($W = 0.0$, $p = 0.0005$, Holm $0.0015$, CI $[-0.258, -0.147]$). The simple effects show it from both sides: typing is worth $+0.234$ without the QoS channel and $+0.035$ with it; the QoS channel $+0.287$ without typing and $+0.087$ with it.

-   **In-distribution fitting is not evidence either way** (Supplementary Table S15): the homogeneous pair reads the Application–Library projection while the typed pair reads the native multigraph, confounding message passing with multi-entity visibility.

**Typing and QoS encoding are substitutes, not complements.** Each mechanism alone lifts the plain baseline from $\rho = 0.317$ to $0.55$–$0.60$; together they reach $0.638$, barely more than either achieves alone. Both supply identical information: which relation an edge traverses. Weight matrices encode it for typing; edge vectors encode it for QoS. Neither channel can track impact the oracle does not express, which bounds what either can be crediting (§7.3.1).

**Reference arm and confounders.** Both large simple effects are measured against `GAT-N`, an unweighted floor ($\rho = 0.317$, seed $\sigma = 0.298$). Substrate, training set, depth and early stopping are held constant, but parameter capacity ($434{,}620$ against $28{,}168$, $15.4\times$) and message directionality ($103{,}725$ reverse-relation parameters in `HGTConv`) are not, so the $+0.134$ typing margin reflects a joint architectural transition rather than typing alone. Four control arms that would separate these — `GAT-N-C` and `GAT-N-QoS-C` (capacity-matched untyped, $437{,}496$ and $439{,}272$ parameters), `GAT-N-QoS16-C` (edge-channel width) and `HGT-QoS-U` (unidirectional, $330{,}895$) — are implemented and registered as Amendment 2 of the analysis plan, with a decision rule fixed before any control result exists, but none has been run at the reported budget. The confound therefore stands as stated, and on present evidence the $15.4\times$ capacity gap buys a margin that does not reach significance once the QoS channel is present.

**Table 8.** The $2 \times 2$ over relation typing (T) and the QoS edge channel (Q), whose four cells are the four reported learned arms: GAT-N ($\neg$T$\neg$Q), HGT (T$\neg$Q), GAT-N-QoS ($\neg$TQ), HGT-QoS (TQ). **Holm correction is applied across the three orthogonal quantities in the upper block only**; the four simple effects below are algebraically determined by those three and are reported descriptively (§7.3.1). Main effects average over the other factor’s levels; the typing effect reflects the joint transition to HGT and is confounded (§8.4). **Won** counts folds with $\Delta > 0$; the interaction is *negative* on all twelve. All quantities are post-hoc, and none were pre-registered. Fold overlap makes the reported $p$-values nominal (§8.3).

| **Quantity**                                                                     | **Contrast**              |  **$\Delta\rho$** |  **Won**  | **$W$** |  **$p$**   | **$p_{\text{Holm}}$** |
|:---------------------------------------------------------------------------------|:--------------------------|------------------:|:---------:|:-------:|:----------:|:----------------------|
| *The $2\times2$: three orthogonal quantities, Holm-corrected across these three* |                           |                   |           |         |            |                       |
| **Typing (main effect)**                                                         | averaged over Q           | $\mathbf{+0.134}$ | **12/12** |   0.0   | **0.0005** | **0.0015**            |
| **QoS channel (main effect)**                                                    | averaged over T           | $\mathbf{+0.187}$ |   11/12   |   2.0   | **0.0015** | **0.0015**            |
| **Typing $\times$ QoS interaction**                                              | difference of differences | $\mathbf{-0.199}$ |   0/12    |   0.0   | **0.0005** | **0.0015**            |
| *Simple effects — descriptive, not separately corrected*                         |                           |                   |           |         |            |                       |
| **Typing, QoS absent**                                                           | HGT vs. GAT-N             |          $+0.234$ |   12/12   |   0.0   |   0.0005   | —                     |
| **Typing, QoS present**                                                          | HGT-QoS vs. GAT-N-QoS     |          $+0.035$ |   9/12    |  19.0   |   0.1294   | —                     |
| **QoS channel, typing absent**                                                   | GAT-N-QoS vs. GAT-N       |          $+0.287$ |   11/12   |   1.0   |   0.0010   | —                     |
| **QoS channel, typing present**                                                  | HGT-QoS vs. HGT           |          $+0.087$ |   10/12   |  22.0   |   0.2036   | —                     |

**The interaction is not an artefact of the correlation scale.** A difference of differences computed on Spearman $\rho$ is not scale-free. Rho is bounded on $[-1, 1]$ and compresses as it approaches either end, so two mechanisms that each move a predictor toward the attainable ceiling can appear to interact sub-additively even when they contribute independently under any monotone rescaling. Because the substitution claim is exactly a claim about an interaction, we recomputed all three quantities with each fold’s $\rho$ passed through $\operatorname{arctanh}$ first, which removes that compression. The interaction survives and grows: $\Delta z = -0.233$, negative on all twelve folds, $W = 0.0$, $p = 0.0005$ (Holm $0.0015$), bootstrap 95% CI $[-0.309, -0.165]$, against $-0.199$ on the raw scale. Both main effects likewise survive ($+0.186$ typing, $+0.269$ QoS channel). Supplementary Table S14 reports the transformed block in full. Sub-additivity is therefore a property of the two mechanisms on this corpus, not of the metric used to measure them — which is what licenses reading them as substitutes, subject to the capacity and directionality confounds above.

### 7.2.1 Where Typed Learning Fails, and How It Can Be Detected

HGT-QoS loses to Topo-QoS on three folds, but only two of them are substantive: Enterprise ($\rho = 0.461$ vs. $0.795$) and Telecom RAN ($0.407$ vs. $0.576$). The third, AV System ($0.722$ vs. $0.753$), is a near-tie at $-0.030$ and carries little interpretive weight. The two substantive inversions are what keep the RQ1 comparison below significance. All three losses fall where `Topo-QoS` is at its own strongest. Enterprise, AV and Telecom RAN are its 2nd, 3rd and 7th best folds of twelve ($0.795$, $0.753$, $0.576$), each at or above its mean of $0.553$. This supports reading the inversions as the learned model discarding structural signal the baseline retains, rather than as folds that are intrinsically hard — on the two synthetic folds where `Topo-QoS` is weakest (Microservices $0.265$, ATM $0.311$) the typed model wins by $+0.229$ and $+0.210$.

#### Absence of a Label-Free Confidence Signature

We tested whether the dispersion $\hat{\sigma}$ of predicted scores over held-out applications could flag untrustworthy folds without labels. It cannot: the lowest-dispersion folds include ones with large positive margins, so no threshold on $\hat{\sigma}$ separates wins from losses. We report the direction of this result rather than its coefficients. The dispersion diagnostic was computed against a prediction export that the twelve-fold artifact behind Tables 6–8 supersedes, and no export exists at that fidelity, so the per-fold $\hat{\sigma}$ values are withheld pending a re-run and no coefficient is quoted for them anywhere in this paper. Graph size is measurable on the current artifact and also fails to flag difficulty in advance (rank correlation with margin $-0.434$, $p = 0.159$), as does connection density.

Feature scale drift across scenarios remains the primary explanation for the Enterprise deficit, as Enterprise is the largest graph ($520$ nodes) and feature-scaling disparities are most acute in this case. Because the protocol strictly holds model hyperparameters constant across folds, there is currently no automated, label-free signal to predetermine whether an unseen architecture will favor the learned model or the training-free baseline (§8.4).

## 7.3 RQ3: Ablations and Sensitivity Analysis

#### RQ3 Summary (Ablations and Sensitivity):

*Only two of the explanation layer’s ten declared constants matter under Morris screening, and no configuration alters any comparison above. The elicited AHP weights are *anti*-predictive against a uniform prior, and the three reliability oracles agree substantially but below the label-noise ceiling.*

This section presents ablations relevant to the primary claims, including the QoS edge encoding, cross-oracle agreement, and per-type stratification that informs the interpretation of the results that follow. Of the ten constants, only the AHP shrinkage $\lambda$ and the Fault-Tolerance/Availability blend $r_{\text{FT}}$ exhibit appreciable influence on $\rho$ ($\mu^* = 0.134$ and $0.132$ under Morris screening, compared to $\le 0.025$ for the remaining eight). No configuration of the topic-weight or QoS sub-weight constants alters any comparison reported above. The elicited AHP weights are, notably, *anti*-predictive: rank correlation decreases monotonically from $0.319$ under a uniform setting to $0.200$ under raw AHP judgment. We retain these weights because RM serves as an attribution instrument rather than a ranking model, a trade-off discussed in §8.4.

### 7.3.1 QoS Feature Ablation

To isolate the contribution of the continuous-categorical QoS edge features (§4.1.1) we evaluated **HGT**, an ablation whose edge features carry only scalar coupling and the relation one-hot. The result is not a second finding but Table 8 read from the other side: because the interaction is symmetric in its two factors, a conditional effect of typing on the QoS channel is necessarily a conditional effect of the QoS channel on typing. *Without typing the encoding is decisive* — `GAT-N-QoS` $0.604$ against `GAT-N`’s $0.317$, $+0.287$, 11/12 folds, $p = 0.0010$, CI $[+0.207, +0.365]$. *With typing it is not significant* — `HGT-QoS` $0.638$ against `HGT`’s $0.551$, $+0.087$, 10/12, $p = 0.2036$. The significance of the *difference* between these rests on the interaction, not on the difference between their $p$-values. We previously treated the typed gain as an independent contribution on top of typing; on the reconciled twelve-fold corpus it does not reach significance, and that independence claim is withdrawn.

The encodings also stabilise optimization, and there the asymmetry runs the other way: the median within-fold standard deviation over five seeds is $0.024$ for `GAT-N-QoS` against $0.298$ for `GAT-N`, and $0.052$ for `HGT-QoS` against $0.114$ for `HGT`. A model given neither channel is by far the least stable, and either one fixes it — consistent with both telling the model which relation an edge belongs to.

#### The target is nearly QoS-free, which bounds what either channel can credit.

These gains are earned against $I^*(v)$, whose ordering a topology-only relabeling recovers at mean $\rho = 0.965$ across the same twelve folds with no QoS term in the labeler (§4.3). The QoS edge channel cannot help the model track QoS-driven impact the oracle does not express, which is the strongest evidence we have for reading it as a relation-identity channel rather than a contract-semantics one. The label does move under QoS, but only at its top-$K$ boundary (Jaccard $0.678$), not in its ranking. A corpus whose oracle expressed QoS-driven impact in its *ordering* — deadline misses, durability replay, priority inversion — would be a stronger test of the encodings than the one we report.

#### QoS Parameter Variance

Modal QoS shares range from 29% to 89% across the twelve scenarios, making sure that every fold carries genuine variation in declared reliability, durability, and priority. As noted in §4.1.1, one schema dimension (`max_blocking_ms_log`) remains zero throughout the corpus as a reserved extension point, while the declared deadline populates the other two (`has_deadline`, `deadline_ns_log`) on $75\%$ of topics. Reported gains therefore stem from six active dimensions.

### 7.3.2 Convergent Validity Over Simulation Oracles

The three reliability-facing oracles measure distinct constructs, so we checked whether they agree before treating any as ground truth. Over the twelve inductive folds on the Application population, the behavioural queue-flow oracle and the topological cascade injector agree at mean Spearman $\rho = 0.627$ (top-$K$ Jaccard $0.370$ against $0.111$ expected by chance), against $I^*$’s own seed-to-seed test–retest of $0.811$–$1.000$. The agreement is therefore substantial but distinctly below label noise, which is the reading we want: an oracle reproducing another to within its own reproducibility would be re-measuring the same topology rather than corroborating it. Two boundaries qualify this — a large share of the agreement is the two oracles concurring on which components are *harmless*, and $I_{\text{dyn}}$ has a measured noise floor of its own that the headline does not correct for.

The Four Golden Signals captured during execution show the mechanism behind the separation: crashing a critical publisher degrades delivery to surviving consumers ($I_{\text{dyn}} > 0$) while *reducing* their queue waits through contention relief ($\rho = -0.499$ against tail-latency delta). Since $I^*$ is a deterministic breadth-first reachability computation and $I_{\text{dyn}}$ a stochastic queue simulation, their moderate agreement is convergent validity between two independent formulations rather than re-measurement, and it places the learned predictors as surrogates for topological cascade reach, not detectors of queue dynamics (§8.3).

### 7.3.3 Node-Type Stratification and Attention

One result governs how every other number in this paper is read. Measured against $I_{\text{comp}}(v)$ over the twelve scenarios of the corpus, stratified RM rank correlations are $\rho = 0.597$ (Application), $0.317$ (Broker), and $0.138$ (Node), while pooling all types gives $\rho = 0.217$ — less than two fifths of the Application figure it is supposed to summarise. This is why every evaluation is reported on a single stratum, and why pooled figures should be read as inflated wherever they appear. We claim only the weaker form: pooled correlation sits above the Execution Host stratum ($0.138$), so this is aggregation bias rather than a strict Simpson reversal, which holds on a companion study’s eight-scenario suite but not on this corpus. A global sensitivity sweep of $I_{\text{comp}}$’s four severity weights ($N = 1{,}000$ Dirichlet draws; Supplementary §S1.2) shows Application correlation exceeding pooled on *all* draws ($\rho \in [0.441, 0.599]$ vs. $[0.096, 0.351]$), while the strict condition holds on only $13.9\%$ of the simplex. The stratification argument rests on the former, which is weight-invariant. A rule-based anti-pattern catalog on the same benchmark flags $93.4\%$ of scored components and so does not discriminate; critical-set identification is delegated to the continuous rankers of §§7.1–7.2.

Aggregated by relation type over the ATM case study, first-layer mean HGT attention orders `USES` into libraries ($0.227$) above publish–subscribe channels ($0.163$–$0.176$). However, the spread across all seven relation types is narrow ($0.15$–$0.23$) and driven substantially by destination in-degree artifacts. Typed attention remains active across relation types without establishing a statistically distinct ordering.

## 7.4 RQ4: Out-of-Family Topological Transfer on Transcribed Open-Source Architectures

#### RQ4 Summary (Real-World Transfer):

*Full-population transfer is established and substantial: under the protocol used everywhere else in this paper the learned model reaches $\rho = 0.760$ $[0.714, 0.819]$ against $0.51$–$0.53$ for the three training-free baselines, with non-overlapping bootstrap intervals over the five systems. Active-stratum transfer is not established. Restricted to components that actually propagate failures, every predictor loses most of its correlation; the learned model is the only one whose point estimate stays positive ($\rho_{>0} = +0.236$ against $-0.055$ to $-0.092$), but its interval $[-0.053, +0.525]$ spans zero and so does every baseline’s. Five systems cannot resolve this comparison, and we report it as unresolved rather than as a qualified positive. The one consistent pattern is directional: active correlation is positive on all three asynchronous pub-sub systems and non-positive on both synchronous call trees.*

We evaluated the framework on five open-source distributed systems transcribed from public repositories: Online Boutique, Train-Ticket, Home Assistant, Autoware Universe (ROS 2), and EdgeX Foundry. All carry labels from the same simulation oracles used throughout, testing out-of-family topological transfer against simulated reachability rather than agreement with field incident telemetry.

Two evaluations are conducted. The first is an exploratory evaluation of the closed-form explanation layer $Q(v)$ against the composite oracle $I_{\text{comp}}(v)$ ($\rho = 0.514$–$0.800$); as noted in §4.3, $I_{\text{comp}}$’s four severity weights are unswept heuristics. The second, §7.4.1, evaluates learned predictors zero-shot against $I^*(v)$ (where RM achieves a lower mean $\rho = 0.516$). `Topo-QoS` is scored here on the flow projection, for the reason given in §8.4. Standard middleware default contract profiles (e.g., ROS 2 Best-Effort/Reliable, MQTT QoS 0/1) were applied uniformly across edges lacking explicit manifests to ensure identical graph representations across baselines.

### 7.4.1 Zero-Shot Transfer of the Learned Model

To assess generalization outside the generator, `HGT-QoS` was trained on all twelve synthetic scenarios and evaluated zero-shot across five open-source systems versus $I^*(v)$ (five seeds; no open-source graph contributed training gradients or checkpoint selection). Tables 9–10 report this arm at the 3-layer, 300-epoch budget used for every other learned result in this paper. An earlier configuration used 2 layers and 150 epochs, chosen to limit over-smoothing across the smaller diameters of these transcribed meshes ($|V_{\text{app}}| \le 41$); that reasoning appeals to a property of the evaluation targets, so although no target label or gradient reached the model, the configuration was not blind to the test systems and we do not report it as the primary result. It is uniformly slightly stronger ($\rho = 0.792$ against $0.760$; $\rho_{>0} = +0.281$ against $+0.236$; $F_1@K = 0.533$ against $0.470$) and leaves every qualitative conclusion unchanged, which is the sensitivity we draw from it.

**Table 9.** Zero-shot out-of-family transfer to five transcribed open-source systems, scored against $I^*(v)$ on the Application population under the protocol used throughout this paper (3 layers, 300 epochs; five seeds, $\pm$ = spread over seeds). No open-source graph contributed gradients or checkpoint selection. Training-free references are deterministic and are scored on identical labels, populations and node sets. Systems are grouped by communication paradigm: active-stratum correlation $\rho_{>0}$ is positive on all three asynchronous pub-sub systems and non-positive on both synchronous call trees. Where a system declares no explicit QoS manifest, default middleware contract profiles (ROS 2 Best-Effort/Reliable, MQTT QoS 0/1) are applied uniformly across baselines. Identification metrics for the same runs are in Supplementary Table S13.

|                                                                   |                        |              |        |           |              |                       |                  |
|:------------------------------------------------------------------|-----------------------:|-------------:|-------:|----------:|-------------:|:---------------------:|-----------------:|
| **Real-World Architecture**                                       | **$|V_{\text{app}}|$** | **$n_{>0}$** | **RM** |  **Topo** | **Topo-QoS** |      **HGT-QoS**      |  **$\rho_{>0}$** |
|                                                                   |                        |              | $\rho$ |    $\rho$ |       $\rho$ |        $\rho$         |        (HGT-QoS) |
| *Asynchronous pub-sub systems — forward failure cascades*         |                        |              |        |           |              |                       |                  |
| **Autoware.universe (ROS 2)**                                     |                     32 |           19 |  0.357 |     0.307 |        0.378 | **0.716 $\pm$ 0.081** | $+$0.517 |
| **EdgeX Foundry (Industrial IoT)**                                |                     22 |           10 |  0.470 |     0.534 |        0.534 | **0.793 $\pm$ 0.037** | $+$0.183 |
| **Home Assistant (Smart Home)**                                   |                     24 |           17 |  0.265 |     0.297 |        0.289 | **0.864 $\pm$ 0.063** | $+$0.702 |
| *Synchronous microservice call trees — backward timeout cascades* |                        |              |        |           |              |                       |                  |
| **Online Boutique (microservices)**                               |                     22 |            8 |  0.777 | **0.891** |        0.888 |   0.710 $\pm$ 0.070   | -0.031 |
| **Train-Ticket Booking Mesh**                                     |                     41 |           14 |  0.713 |     0.528 |        0.541 | **0.717 $\pm$ 0.096** | -0.192 |
| **Mean**                                                          |                      — |            — |  0.516 |     0.511 |        0.526 |       **0.760**       | $+$0.236 |

**Table 10.** Means over the five transcribed systems on the full Application population ($\rho$) and restricted to components with strictly positive ground-truth impact ($\rho_{>0}$), with percentile bootstrap intervals over the five systems ($B = 2{,}000$). All four predictors are scored on identical labels, populations and node sets from one run, so the columns are commensurable. At $n = 5$ the intervals are descriptive and carry no significance claim. **The full-population intervals separate the learned model from all three baselines; every active-stratum interval spans zero**, which is why RQ4 is reported as established on the full population and unresolved on the active one.

| **Predictor**   |     **$\rho$ (full), 95% CI**     |   **$\rho_{>0}$ (active), 95% CI**   |
|:----------------|:---------------------------------:|:------------------------------------:|
| **RM / $Q(v)$** |     $0.516$ $[0.343, 0.680]$      |     $-0.055$ $[-0.292, +0.213]$      |
| **Topo**        |     $0.511$ $[0.346, 0.703]$      |     $-0.083$ $[-0.269, +0.108]$      |
| **Topo-QoS**    |     $0.526$ $[0.357, 0.699]$      |     $-0.092$ $[-0.268, +0.094]$      |
| **HGT-QoS**     | $\mathbf{0.760}$ $[0.714, 0.819]$ | $\mathbf{+0.236}$ $[-0.053, +0.525]$ |

**Key Insights for Real-World Transfer:**

1.  **The full-population figure is not a ranking result.** On all Applications, `HGT-QoS` reaches $\rho = 0.760$ $[0.714, 0.819]$ against $0.511$ (Topo), $0.526$ (`Topo-QoS`) and $0.516$ (RM), leading on 4 of 5 systems with a non-overlapping interval. However, between $29\%$ (Home Assistant) and $66\%$ (Train-Ticket) of Applications carry zero simulated impact, so full correlation rewards separating inert components from active ones rather than ranking the active ones.

2.  **On the active stratum, five systems cannot resolve the question.** The learned model is the only predictor whose active-stratum estimate stays positive ($\rho_{>0} = +0.236$ against $-0.055$ to $-0.092$), but its interval $[-0.053, +0.525]$ spans zero and so does every baseline’s (Table 10). We therefore claim only that its point estimate is positive, not that it transfers. The one consistent pattern is directional: $\rho_{>0}$ is positive on all three asynchronous pub-sub systems (Home Assistant $+0.702$, Autoware $+0.517$, EdgeX $+0.183$) and non-positive on both synchronous call trees (Online Boutique $-0.031$, Train-Ticket $-0.192$). A 3–2 split is a pattern worth naming, not a tested effect.

3.  **Identification is where the learned model separates most clearly.** On top-$K$ overlap it averages $0.470$ $[0.410, 0.540]$ against $0.248$ $[0.09, 0.46]$ for the structural baselines, whose interval overlaps it; on the threshold-free measures of Supplementary Table S13 the gap is wider ($F_1@\tau$ $0.520$ vs. $0.29$–$0.33$; PR-AUC $0.749$ vs. $0.47$–$0.52$). The mechanism is visible on EdgeX, where symmetric star connections from peripheral adapters to brokers produce identical betweenness ties that collapse structural triage entirely ($F_1@K = 0.000$) while relational attention still separates components.

4.  **Architectural boundary condition and ingestion boundary.** Synchronous RPC architectures propagate failures *backward* along invocation trees via timeouts and thread starvation [48], whereas pub-sub architectures cascade forward via queue saturation. Trained on pub-sub semantics, the model’s directional bias does not transfer to call trees. We therefore state an ingestion boundary: SaG’s learned pipeline applies to asynchronous, event-driven architectures (ROS 2, Kafka, DDS, MQTT), and static call-graph reachability tools should be used for synchronous RPC/REST meshes. The closed-form baselines are not the substitute there either — `Topo-QoS` leads on Online Boutique only on the full population ($0.888$) and is negative on its active components ($-0.072$), so its advantage lies in identifying inert services rather than ranking propagating ones.

## 7.5 RQ5: Analysis Cost and Its Comparison Against Simulation

#### RQ5 Summary (Analysis vs. Simulation Cost):

*While the neural forward pass executes in milliseconds ($56\,\text{ms}$ for 2,000 nodes), cold deterministic feature extraction requires up to $239.3\,\text{s}$, dominated by the $O(|V|^2 + |V||E|)$ Connectivity Degradation Index. Cold static gating costs $2$–$18\times$ the discrete-event simulation it was meant to displace across the twelve scenarios (median $5.6\times$), reaching its maximum only on the 520-node Enterprise mesh ($79.3\,\text{s}$ vs. $4.5\,\text{s}$). The premium tracks the density of the derived dependency projection rather than component count, and at no scenario size is the gate cheaper — refuting the assumption that static analysis is inherently cheaper than simulation without incremental graph caching.*

RQ5 quantifies computing overhead and sustainability during CI/CD evaluation, with per-stage latencies reported in Table 11:

**Table 11.** Per-stage latency of the inference pipeline across scaling graph sizes (CPU, median of 3 runs).

| **$|V|$** | **$|E|$** | **Analyze (s)** | **Graph $\to$ Tensor (s)** | **HGT Forward (ms)** | **Analyze : Forward** |
|:---------:|:---------:|:---------------:|:--------------------------:|:--------------------:|:---------------------:|
|    249    |   1,127   |      1.74       |           0.010            |         26.5         |          66×          |
|    499    |   2,402   |      8.32       |           0.022            |         16.4         |         509×          |
|    999    |   6,422   |      44.54      |           0.056            |         21.1         |        2,108×         |
|   1,998   |  19,301   |     239.34      |           0.157            |         56.2         |      **4,259×**       |

The neural stage is the cheapest by a wide margin and the deterministic one is not. At 2,000 components the HGT forward pass takes $56\,\text{ms}$ against $239\,\text{s}$ for structural analysis, a ratio of $4{,}259\times$ — but that is a ratio between pipeline stages, not a cost of evaluation, because indices 0–17 of every node feature vector are produced by the analysis stage the forward pass depends on. End-to-end evaluation of an unseen 2,000-component architecture takes about four minutes, of which the learned model is $0.02\%$; the $56\,\text{ms}$ is the marginal cost of re-scoring an already-analysed graph. Across the corpus the complete gate (structural analysis plus 18 anti-pattern detectors, whose share is $\le 0.19\,\text{s}$) executes in $0.16$–$79.3\,\text{s}$ (Table 12).

**One metric dominates cost.** Across the endpoints of Table 11 the measured cost is consistent with the stage’s $O(|V|^2 + |V||E|)$ bound: from 249 to 1,998 components wall-clock rises $138\times$ against a $137\times$ growth in $|V||E|$. That is an endpoint coincidence rather than a tracked curve: between the middle rows the component count doubles ($499 \to 999$) while wall-clock rises $5.4\times$, faster than the bound requires, and the series varies $|V|$ and $|E|$ together so it cannot separate them — Table 12 does that on the corpus. The dominant term is the Connectivity Degradation Index, computed for every node in the main connected component rather than for articulation points alone. That is a correctness requirement, not an oversight: gating CDI to articulation points leaves it identically zero wherever removal does not literally disconnect the graph, driving $A(v)$ to a near-constant in exactly the redundant multi-publisher topologies this system targets. The cost is the price of a non-degenerate Availability score.

**Table 12.** Static analysis gate against the cascade-reachability oracle, per scenario, from one paired measurement session. **Gate** is structural analysis plus the 18 anti-pattern detectors, whose own share is negligible ($\le 0.19\,\text{s}$ everywhere). **Oracle** is the full five-seed ground-truth labelling sweep over Application, Broker and Library nodes. $|E_{\text{proj}}|$ is the number of derived `DEPENDS_ON` edges in the Application–Library projection the analysis stage traverses, read from the committed cache. The ratio is $2.0$–$17.7\times$ with a median of $5.6\times$: the eighteen-fold figure quoted elsewhere in this section is the maximum, not the typical case. Ordering by $|E_{\text{proj}}|$ rather than by $|V|$ is what makes the column monotone — Enterprise carries 300 applications over only 120 topics, so its Rule-1 projection is near-complete, while IoT Smart City has more components and a sixth of the edges at a third of the cost.

| **Scenario**                     | **$|E_{\text{proj}}|$** | **Gate (s)** | **Oracle (s)** |       **Ratio** |
|:---------------------------------|------------------------:|-------------:|---------------:|----------------:|
| **Enterprise**                   |                  26,276 |        79.27 |          4.487 |    17.7$\times$ |
| **Enterprise Integration (ESB)** |                   4,641 |         3.17 |          0.245 |    12.9$\times$ |
| **AV System**                    |                   3,073 |         2.61 |          0.355 |     7.4$\times$ |
| **Financial Trading**            |                   2,657 |         1.71 |          0.239 |     7.1$\times$ |
| **Real-Time Gaming**             |                   2,318 |         2.43 |          0.379 |     6.4$\times$ |
| **Healthcare**                   |                   1,590 |         0.92 |          0.149 |     6.2$\times$ |
| **IoT Smart City**               |                   1,835 |         5.48 |          1.104 |     5.0$\times$ |
| **Telecom RAN**                  |                   2,116 |         3.70 |          1.065 |     3.5$\times$ |
| **Logistics Fleet**              |                   1,660 |         2.74 |          0.939 |     2.9$\times$ |
| **Microservices (synthetic)**    |                   1,524 |         2.29 |          0.828 |     2.8$\times$ |
| **Industrial SCADA**             |                     897 |         2.43 |          1.191 |     2.0$\times$ |
| **ATM System**                   |                     155 |         0.16 |          0.080 |     2.0$\times$ |
| **Median**                       |                       — |            — |              — | **5.6$\times$** |

### 7.5.1 The Gate Is Not Cheaper Than the Simulation It Replaces

The framing that motivated this analysis — static gating as a low-cost substitute for dynamic simulation — does not survive measurement against our own oracle. Timing the cascade reachability labeling sweep (five seeds, node types Application/Broker/Library, the full ground-truth run) over all twelve scenarios, on the same machine, at the same commit and in the same measurement session as the gate, gives $0.08$–$4.49\,\text{s}$ per scenario against $0.16$–$79.3\,\text{s}$ for the analysis gate. Both maxima occur in the 520-component Enterprise mesh, so the largest scenario compares $4.5\,\text{s}$ of simulation against $79.3\,\text{s}$ of static analysis: there the gate costs roughly eighteen times as much as the simulation it is meant to replace. That is the extreme, not the centre. Pairing the sweeps scenario by scenario (Table 12) gives $2.0$–$17.7\times$, median $5.6\times$. Two things follow, in opposite directions: the headline is weaker than one number suggests, and the finding is stronger, because the gate is more expensive on *all twelve* scenarios. What predicts the premium is the derived projection’s size, not the component count — the ratio correlates with $|E_{\text{proj}}|$ at $\rho = 0.95$ against $0.79$ for cost against $|V|$. Enterprise is the outlier because its 300 applications share only 120 topics, so Rule 1 derives a near-complete graph of $26{,}276$ edges, while IoT Smart City has more components, an eighth of the edges and a third of the cost. These are wall-clock figures on one commodity CPU, read to one significant figure: across sessions the gate maximum ranges over $77$–$83\,\text{s}$ and the oracle maximum over $4.5$–$4.8\,\text{s}$, giving $16.7\times$ and $17.7\times$ on two independently paired sessions. Two earlier pairings are superseded rather than reconciled, one unstamped and one pairing this corpus’s oracle against a companion study’s eight-scenario gate; Table 12 uses only the twelve-scenario pairing, whose halves share a commit and corpus digest.

This finding refutes the assumption that static analysis is computationally cheaper than in-process simulation: breadth-first cascade traversal is simpler than computing all-pairs connectivity degradation ($O(|V|^2 + |V||E|)$). Two deployment trade-offs survive it. Static analysis scores components (shared libraries, hosts) and dependency edges that node-level simulation leaves unscored, and its deterministic metrics can in principle be cached across commits, recomputing only the $k$-hop neighbourhood a pull request touches. Table 11 times full from-scratch recomputation; caching is not implemented, and without it direct simulation is strictly faster.

# 8. Discussion, Threats to Validity, and Limitations

## 8.1 Discussion and Practical Consequences

**When to use Topo-QoS, and when to use HGT-QoS.** The findings do not support an unequivocal recommendation of the learned model over the closed-form alternative; we present empirical trade-offs directly.

-   **Training-free ranking (`Topo-QoS`) vs. Simulation.** `Topo-QoS` needs no training, checkpoints or tuning, and reaches $\rho = 0.553$ zero-shot across the twelve synthetic architectures and $0.526$ across the five open-source ones, with no significant learned advantage over it ($+0.085$, $p = 0.151$). On the one large-diameter architecture we have it is also clearly better ($0.795$ vs. $0.461$ on the 520-node Enterprise ESB), where global shortest paths resolve bottlenecks that fixed 3-hop convolutions localize. It is therefore the zero-maintenance default for scalar ranking in resource-constrained CI. Where queue parameters exist and graphs are small ($N < 150$), direct simulation ($0.08$–$4.5\,\text{s}$) gives exact ground truth instead.

-   **When to deploy `HGT-QoS`.** `HGT-QoS` is justified when systems feature dense, irregular topologies with multi-tenant shared library or host dependencies under $N < 150$, where relational attention gains $+0.210$ to $+0.229$ over `Topo-QoS` (the synthetic Microservices and ATM folds, §7.2.1; note that on the transcribed Online Boutique mesh the ordering reverses on the full population, $0.769$ against $0.888$), and when teams require scoring of unsimulated non-runnable entities (physical hosts, shared packages) that node-level simulation leaves unscored. Otherwise, untyped QoS-weighted GNNs offer a superior efficiency trade-off ($\rho = 0.604$ with $28{,}168$ parameters vs. `HGT-QoS`’s $0.638$ with $434{,}620$), as combining typing and QoS yields minimal additive benefit.

-   **Capabilities without a closed-form counterpart.** Per-relation attention (§7.3.3) offers a typed diagnostic, but measured spread across relation types is narrow ($0.15$–$0.23$) and governed by destination in-degree, making it active but diagnostically unreliable on present evidence.

-   **Ingestion boundary.** Deployment must condition on communication synchrony (Table 13): on asynchronous pub-sub systems (ROS 2, DDS, MQTT, Kafka) `HGT-QoS` transfers positively on active components, whereas on the synchronous Train-Ticket call tree it inverts and should not be deployed — use static call-graph reachability tools instead. `Topo-QoS` is not the substitute: it is negative on the active components of both RPC meshes.

**Architectural and Systemic Drivers of Graph Learning Success.** Across seventeen architectures, graph neural network performance is governed by the five systemic factors in Table 13.

**Table 13.** Systemic drivers of graph-learning performance across the seventeen evaluated architectures, with the mechanism each reflects and the evidence for it on this corpus.

| **Factor**                   | **Mechanism**                                                                                                                | **Evidence on this corpus**                                                                                                                  |
|:-----------------------------|:-----------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| **Communication synchrony**  | Directional bias must match physical failure flow: pub-sub starves forward, synchronous call trees time out backward [48]. | Pub-sub $\rho_{>0} > 0$ on 3/3; RPC non-positive on 2/2, with every training-free baseline negative on both.                                 |
| **Scale and diameter**       | Fixed 3-layer message passing covers the diameter of small graphs ($N < 150$) but localizes at $N \ge 500$.                  | 520-node Enterprise: $0.461$ vs. $0.795$. One scenario at this scale, so the threshold is a conjecture, not a measured breakpoint.           |
| **Topology and symmetry**    | Isomorphic 1-hop neighborhoods yield near-identical embeddings; dense irregular meshes do not.                               | Broker-hub ESB $\rho = 0.472$; on the synthetic Microservices and ATM folds, $+0.210$ to $+0.229$ over `Topo-QoS`. Single-fold observations. |
| **Relational heterogeneity** | Typing is essential where shared hosts or libraries induce blast radii, but substitutes for QoS once contracts are declared. | Untyped GNNs drop to $\rho = 0.317$ (§7.2).                                                                                                  |
| **Inert-node base rates**    | Zero-impact sinks inflate full-population correlation through trivial inertness filtering.                                   | $21\%$–$52\%$ of applications carry $I^*(v) = 0$; $\rho_{>0}/\rho \approx 51\%$–$56\%$.                                                      |

**Dual-Engine Consensus Protocol.** Prediction dispersion does not separate the folds the learned model loses from those it wins (§7.2.1), so automated fallback is withdrawn. SaG instead runs `HGT-QoS` and `Topo-QoS` concurrently, flagging unanimous top-$K$ nodes and surfacing divergent rankings for review.

**Role of the Explanation Layer.** The RM profile ($Q(v)$, §5) decomposes structural mechanics under ISO/IEC 25010 to inform refactoring — distinguishing Availability-driven replication from Fault-Tolerance circuit breakers — while neural and centrality predictors govern triage priority. Its elicited AHP weights rank worse than a uniform prior, which we recommend instead pending user studies.

## 8.2 Performance and Computational Sustainability Implications

**What sustainability means for a pre-deployment gate.** Green software engineering assesses energy across development, assurance, and execution [29, 88, 89, 90, 91, 30, 31]. Live chaos sweeps require cluster-hours across provisioned nodes; pre-deployment static analysis avoids provisioning that cluster at all. We state the benefit as infrastructure avoidance rather than as a measured energy saving, because we have not measured the avoided side and cannot put a figure on it. What we can bound is the side we do spend. Taking the measured wall-clock of Table 12 at the processor’s published base power ($28\,\text{W}$ for the measurement machine’s SoC), one gate pass over the entire twelve-scenario corpus costs at most $3.0\,\text{kJ}$ ($0.83\,\text{Wh}$), of which the largest scenario is $2.2\,\text{kJ}$, against at most $0.31\,\text{kJ}$ ($0.086\,\text{Wh}$) for the oracle sweep. These are upper bounds and not measurements: the workload is single-threaded, so charging whole-package power to it overstates the draw by roughly the loaded core count, and wall-clock includes interpreter start-up and I/O. Direct measurement in joules requires hardware counters (RAPL, NVML) and profilers [30, 32, 33], which we outline as an instrumentation protocol for future testbeds. The honest summary is that the computation a pre-deployment gate spends is under a watt-hour for this corpus, which is negligible beside a staging cluster but is not, on this evidence, a quantified saving.

**Where the cost actually sits.** We withdraw the efficiency claim: cold static analysis does not reduce CPU computation over simulation. The gate costs $2$–$18\times$ the cascade traversal across the twelve scenarios (median $5.6\times$; Table 12), rising to its maximum on Enterprise ($79.3\,\text{s}$ against $4.5\,\text{s}$) where CDI’s $O(|V|^2 + |V||E|)$ cost meets the densest derived projection in the corpus. It is more expensive on every scenario, so the direction of the comparison does not depend on which one is quoted. In CI, sustainability requires incremental caching: caching base metrics across commits and extracting features only for PR delta subgraphs reduces latency to the sub-second forward pass ($56\,\text{ms}$). We emphasize this is an engineering hypothesis: delta latency depends on the PR diff’s topological centrality (leaf worker vs. central broker), and benchmarking commit-diff extraction across PR topologies is prioritized.

## 8.3 Threats to Validity

**Construct Validity.** Ground-truth impact $I^*(v)$ derives from simulation rather than live outages. While $I^*$ correlates with dynamic queue flow $I_{\text{dyn}}$ ($\rho = 0.627$ against a $0.811$–$1.000$ ceiling, §7.3.2), top-$K$ Jaccard reaches only $0.27$–$0.37$, and $I^*(v)$ is recovered at $\rho = 0.965$ by topology-only relabeling, confirming the target captures reachability rather than runtime queue collapse or retry storms. More fundamentally, $I^*(v)$ is a deterministic functional of the graph the predictors read, so RQ1 asks a learned model to recover a closed-form graph computation from features that already summarise that graph — a task on which a strong closed-form comparator is expected, which bounds what the parity result says about graph learning generally. The remedy is internal: $I_{\text{dyn}}$ agrees with $I^*$ at only $\rho = 0.627$ and so carries the non-topological structure this design lacks. Retargeting the LOSO contrasts on it needs per-node predictions at the twelve-fold artifact’s fidelity, hence a full re-run, and is the first experiment we intend to report. Validating topological cascades against production incident telemetry remains the central empirical frontier. Transcribing the open-source architectures also relied on standardized heuristics applied by a single coder, so there is no inter-coder agreement figure. Those five systems carry the whole of RQ4, and a transcription choice that systematically favoured one predictor would be invisible to us; independent re-transcription of at least two of them is required before these numbers read as firmer than indicative.

**Internal Validity.** Feature leakage is prevented by strict graph separation: predictors consume $G_{\text{analysis}}$, while simulation oracles traverse $G_{\text{structural}}$ (CI-asserted). Parity is maintained via matched datasets, depths, and early stopping. Comparing `HGT-QoS` to `GAT-N-QoS` conflates typing with a $15.4\times$ capacity gap ($434{,}620$ vs. $28{,}168$) and bidirectional passing ($103{,}725$ reverse parameters in `HGTConv`), which four registered control arms would isolate (`GAT-N-C`, `GAT-N-QoS-C`, `GAT-N-QoS16-C`, `HGT-QoS-U`; §8.4). They are implemented in the replication package and registered with a decision rule fixed in advance, but none has been run at the reported budget, so this confound is disclosed rather than resolved. Six active QoS dimensions govern profiles, with one reserved extension point.

**External Validity.** The corpus comprises 2,812 components across seventeen architectures spanning six domains: robotics (ROS 2), smart cities, healthcare, finance, enterprise ESB, and edge IoT (EdgeX Foundry, Home Assistant). Heterogeneity exposes the active-stratum boundary condition: pub-sub cascades propagate forward via starvation ($\rho_{>0} \in [+0.262, +0.770]$, Table 9), whereas synchronous RPC microservices cascade backward along call trees via timeouts [48], where active correlation inverts ($-0.071$) and baselines fail (Table 10). Directional biases must condition on synchrony. Scaling beyond 2,000 nodes requires incremental caching or mini-batching (GraphSAINT [92]).

**Conclusion Validity.** Distributions are evaluated using a non-parametric correlation (Spearman $\rho$), bootstrap intervals ($B = 2{,}000$), and Wilcoxon signed-rank tests. Analyses are stratified because pooling node types conflates disparate strata (Execution Host $\rho = 0.138$ vs. Application $0.597$), leaving the pooled figure ($0.217$) at less than two fifths of the evaluated stratum. Because folds share training graphs, nominal Wilcoxon $p$-values are optimistic and interpreted descriptively alongside 12-fold sign consistency; synthetic graphs derive from a single generator family.

**Repeatability.** Seed spread and run-to-run displacement are distinct quantities; only the former is reported in Table 6. At fixed code and seeds the pipeline is deterministic: consecutive runs reproduced every reported figure identically. Across code revisions it is not, and that matters more. Between the two most recent committed sweeps — different commits, identical corpus digest, so the difference is attributable to code — all $180$ training-free cells reproduce exactly ($|\Delta\rho| = 0.000000$) while the $240$ learned cells move by up to $0.519$ at seed level and $0.172$ at fold mean (`GAT-N` on Healthcare, $0.600 \rightarrow 0.428$). The movement is largest for the weakest arm and smallest for `HGT-QoS` ($0.041$), which is what keeps the primary contrast interpretable — but only just, since $0.172$ is twice the $+0.085$ margin RQ1 turns on. Read the sign of that result as robust and its magnitude as pinned to these artifacts. The Table 8 contrasts are better placed, being differences within one sweep. The per-arm ledger is in the replication package (`reproduce/rerun_drift.py`). Revisions also altered the active stratum ($n_{>0}$ shift $26\%$–$56\%$), making $\rho_{>0}$ non-comparable across revisions. Learned figures are pinned to commit artifacts; provenance tracking in Data Availability makes this auditable.

## 8.4 Limitations and Future Work

**Correction of the Real-World Baseline.** On raw multigraphs, betweenness vanishes for Application nodes; evaluating on `DEPENDS_ON` flow projections enables `Topo-QoS` across all five real-world systems (Table 9). **Explanation Layer Validation.** The explanation layer is presented as a design pattern rather than a validated contribution (§5): its elicited AHP weights rank worse than a uniform prior (§7.3), it trails unweighted centrality, and no user study has tested whether its archetypes help a developer act. We recommend the uniform prior and retain the elicited constant as the shipped default for one reason, stated so it is not mistaken for inertia: the same shrinkage constant parameterises the $I_{\text{comp}}$ severity weights, so moving it would silently re-label the composite oracle and every figure scored against it. Decoupling the two, then a mutation benchmark (inject a known single point of failure or cascade hub and test whether the layer names the right archetype) and a developer study, are the prerequisites for any evidential claim here. **Uncontrolled Confounds in Typing.** Table 8 holds substrate, depth, and training splits constant, but capacity ($434{,}620$ vs. $28{,}168$) and reverse directionality ($103{,}725$ parameters in `HGTConv`) remain unmatched; the $+0.134$ effect reflects the joint transition to HGT. Four control arms that isolate these factors — `GAT-N-C` and `GAT-N-QoS-C` (capacity-matched untyped models), `GAT-N-QoS16-C` (edge-channel width) and `HGT-QoS-U` (unidirectional passing) — are implemented in the replication package and registered in the analysis plan together with a decision rule fixed before any of them was run. None has yet been run at the reported budget. Running them is the single change that would most alter what §7.2 is entitled to claim, and the registered rule commits us to narrowing that claim if the margin does not survive. **No Comparison Against a Published Method.** Every comparator in this paper is either our own closed-form score or an ablation of our own model. §2.4 reviews learned node-criticality models (FINDER [64], DrBC [65], PowerGraph [66]) without reproducing any of them, so the RQ1 finding is a statement about this heterogeneous transformer against these baselines, not about graph learning for criticality ranking in general. Reproducing DrBC or FINDER on this corpus — both address the ranking task directly — is the comparison that would generalise the claim, and its absence is the clearest gap in the evaluation. **Absence of Live Incident Telemetry.** Simulation oracles ($I^*, I_{\text{comp}}, I_{\text{dyn}}, I_M$) ensure determinism but do not replace incident logs or chaos tests; validating topological cascades against production outages remains essential. **Model Selection and Incremental Caching.** Early stopping uses an inner validation split; held-out scenario validation is a prioritized extension. Prediction dispersion fails as an out-of-distribution fallback signal (§7.2.1). Practical CI deployment requires incremental graph caching over PR diffs. **Future Directions.** Key extensions include: (1) modeling hybrid distributed architectures (e.g., synchronous REST/gRPC frontends feeding asynchronous Kafka pipelines) by conditioning edge directional semantics on synchrony (`SYNCHRONOUS_CALL` with backward timeout cascading vs. `PUBLISH_SUBSCRIBE` with forward starvation); (2) modeling distributed LLM serving backbones (vLLM, DeepSpeed); (3) measuring hardware energy via RAPL/NVML in joules, replacing the bounded estimate of §8.2; (4) retargeting the RQ1 and RQ2 contrasts on $I_{\text{dyn}}$, whose queueing structure is not recoverable in closed form from the input graph, so that the comparison tests graph learning rather than the recovery of a topological functional; (5) separating whether $w(e)$ carries contract semantics or acts as a traffic proxy, by holding the declared QoS triple fixed while varying payload size and publication frequency and scoring `Topo-QoS` against a topology-only relabelling of the oracle; (6) reproducing a published learned-criticality baseline; and (7) advancing to prescriptive synthesis generating pull requests with circuit breakers.

# 9. Conclusion

This study investigated the viability and boundaries of heterogeneous graph neural networks for pre-deployment dependability analysis in asynchronous distributed systems. Using Software-as-a-Graph (SaG) to transform Architecture-as-Code manifests into typed multigraphs, we evaluated a relation-specific Heterogeneous Graph Transformer with QoS edge encodings against unparameterized structural baselines and an ISO/IEC 25010/25019 attribution layer without runtime telemetry.

The primary empirical finding is that relation typing and 16-D QoS edge encodings substitute rather than complement each other. While each exhibits a significant main effect under inductive distribution shift ($\Delta\rho = +0.134$ and $+0.187$, Holm $p = 0.0015$, typing reflecting the joint transition to HGT), their interaction is $-0.199$ across all twelve folds ($p = 0.0005$), and $-0.233$ once the bounded correlation scale is removed by a Fisher transformation, so the sub-additivity is a property of the mechanisms rather than of the metric: typing contributes $+0.234$ without QoS and $+0.035$ with it. Combining both yields minimal benefit, consistent with both channels encoding redundant traversal semantics against a near-topological simulation target (recovered at $\rho = 0.965$ without QoS). The typing factor also carries a $15.4\times$ capacity gap and a change in message-passing directionality; four registered control arms that would separate these are implemented but not yet run, so the attribution to typing itself remains provisional.

These findings carry direct practical implications: teams requiring scalar rankings should consider the simpler untyped QoS-weighted baseline ($\rho = 0.604$ at $28{,}168$ parameters vs. HGT-QoS’s $0.638$ at $434{,}620$), as relational attention is narrow and degree-sensitive (§§7.3.3 and 8.1). Boundary conditions are explicit: learned ranking does not significantly outperform unparameterized QoS centrality ($+0.085$, $p = 0.151$), zero-shot transfer on the active components of the five transcribed systems is weak for all predictors and only the typed model stays positive there ($+0.281$ against negative training-free baselines, inverting on synchronous call trees; on the synthetic active stratum every predictor remains positive), prediction dispersion fails as an OOD indicator, and elicited AHP weights rank poorer than a uniform prior. While neural forward passes execute in $56\,\text{ms}$, deterministic feature extraction costs $2$–$18\times$ the simulation it was meant to displace (median $5.6\times$ over twelve scenarios), and is more expensive on every one of them unless incremental caching is applied across commits. Two frontiers follow from this, one internal and one external. Internally, the target these predictors are scored against is a deterministic functional of their own input, so the sharpest next test is to retarget the comparison on the queue-flow oracle, whose behaviour is not recoverable in closed form from the graph. Externally, and more importantly, no result here is validated against an observed failure: whether such pre-deployment pipelines predict live production outages remains the central empirical question, and nothing in this design can answer it.

---

# Declarations

**CRediT Authorship Contribution Statement.** **Ibrahim Onuralp Yigit:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Visualization. **Feza Buzluca:** Conceptualization, Methodology, Validation, Writing – review and editing, Supervision. **Declaration of Competing Interest.** The authors declare no competing financial interests or personal relationships that could have influenced this work. **Funding.** This research received no external grant.

**Data Availability.** The replication package (datasets, harnesses, checkpoints, scripts) is available on Zenodo under DOI [10.5281/zenodo.14922108](https://doi.org/10.5281/zenodo.14922108) [93] with `uv`/`pip` environments. Synthetic datasets regenerate byte-identically. The deposit ships all artifacts backing reported tables as a dated bundle (`SaG_JSS_Results_<stamp>`) with a `MANIFEST.json` recording SHA-256 digests, commit hashes, and corpus provenance. Five artifacts carry no provenance block, where correspondence is asserted by the bundle. The verification script (`reproduce/reconcile_manuscript.py`) runs standalone against the deposit, mechanically verifying all 415 table quantities against JSON artifacts.

# Declaration of Generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used Anthropic’s Claude to assist with typesetting, LaTeX formatting, and the development of analysis and reporting scripts in the replication package. After using this tool the authors reviewed and edited the content as needed and take full responsibility for the content of the published article. The study design, the choice of experiments, the interpretation of results, and all scientific claims are the authors’ own.

---

# References

[1] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, W. Woodall, Robot operating system 2: Design, architecture, and uses in the wild, Science Robotics 7 (66) (2022) eabm6074.

[2] J. Kreps, N. Narkhede, J. Rao, Kafka: A distributed messaging system for log processing, in: Proc. 6th Int. Workshop on Networking Meets Databases (NetDB), 2011.

[3] Object Management Group, Data distribution service (dds), Tech. Rep. formal/2015-04-10, version 1.4, Object Management Group (2015).

[4] OASIS, MQTT version 5.0, OASIS Standard, <https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html> (accessed 9 September 2026) (2019).

[5] N. Dragoni, S. Giallorenzo, A. L. Lafuente, M. Mazzara, F. Montesi, R. Mustafin, L. Safina, Microservices: Yesterday, today, and tomorrow, in: Present and Ulterior Software Engineering, Springer, 2017, pp. 195--216.

[6] S. Newman, Building Microservices: Designing Fine-Grained Systems, O'Reilly Media, 2015.

[7] P. T. Eugster, P. A. Felber, R. Guerraoui, A.-M. Kermarrec, The many faces of publish/subscribe, ACM Computing Surveys 35 (2) (2003) 114--131.

[8] A. E. Motter, Y.-C. Lai, Cascade-based attacks on complex networks, Physical Review E 66 (2002) 065102(R).

[9] S. V. Buldyrev, R. Parshani, G. Paul, H. E. Stanley, S. Havlin, Catastrophic cascade of failures in interdependent networks, Nature 464 (2010) 1025--1028.

[10] R. Albert, H. Jeong, A.-L. Barab\'asi, Error and attack tolerance of complex networks, Nature 406 (2000) 378--382.

[11] A. Avizienis, J.-C. Laprie, B. Randell, C. Landwehr, Basic concepts and taxonomy of dependable and secure computing, IEEE Transactions on Dependable and Secure Computing 1 (1) (2004) 11--33.

[12] L. Bass, P. Clements, R. Kazman, Software Architecture in Practice, 3rd Edition, Addison-Wesley, 2012.

[13] International Organization for Standardization, ISO/IEC 25010:2023 --- systems and software engineering --- systems and software quality requirements and evaluation (square) --- product quality model, Tech. rep., International Organization for Standardization (2023).

[14] International Organization for Standardization, ISO/IEC 25019:2023 --- systems and software engineering --- systems and software quality requirements and evaluation (square) --- quality-in-use model, Tech. rep., International Organization for Standardization (2023).

[15] D. E. Perry, A. L. Wolf, Foundations for the study of software architecture, ACM SIGSOFT Software Engineering Notes 17 (4) (1992) 40--52.

[16] W. Cunningham, The WyCash portfolio management system, in: Addendum to the Proc. Conf. on Object-Oriented Programming Systems, Languages, and Applications (OOPSLA), 1992, pp. 29--30.

[17] J. Garcia, D. Popescu, G. Edwards, N. Medvidovic, Identifying architectural bad smells, in: Proc. 13th European Conf. on Software Maintenance and Reengineering (CSMR), 2009, pp. 255--258.

[18] R. Kazman, M. Klein, M. Barbacci, T. Longstaff, H. Lipson, J. Carriere, The architecture tradeoff analysis method, in: Proc. 4th IEEE Int. Conf. on Engineering of Complex Computer Systems (ICECCS), 1998, pp. 68--78.

[19] SonarSource, Clean as you code, SonarQube documentation, <https://docs.sonarsource.com/sonarqube-server/latest/core-concepts/clean-as-you-code/introduction/> (accessed 9 September 2026) (2024).

[20] T. J. McCabe, A complexity measure, IEEE Transactions on Software Engineering SE-2 (4) (1976) 308--320.

[21] S. R. Chidamber, C. F. Kemerer, A metrics suite for object oriented design, IEEE Transactions on Software Engineering 20 (6) (1994) 476--493.

[22] N. Fenton, J. Bieman, Software Metrics: A Rigorous and Practical Approach, 3rd Edition, CRC Press, 2014.

[23] A. Basiri, N. Behnam, R. de Rooij, L. Hochstein, L. Kosewski, J. Reynolds, C. Rosenthal, Chaos engineering, IEEE Software 33 (3) (2016) 35--41.

[24] L. C. Freeman, A set of measures of centrality based on betweenness, Sociometry 40 (1) (1977) 35--41.

[25] S. Brin, L. Page, The anatomy of a large-scale hypertextual web search engine, Computer Networks and ISDN Systems 30 (1--7) (1998) 107--117.

[26] U. Brandes, A faster algorithm for betweenness centrality, Journal of Mathematical Sociology 25 (2) (2001) 163--177.

[27] M. E. J. Newman, Networks: An Introduction, Oxford University Press, 2010.

[28] I. O. Yigit, F. Buzluca, A graph-based dependency analysis method for identifying critical components in distributed publish--subscribe systems, in: Proc. IEEE Int. Conf. on Recent Advances in Systems Science and Engineering (RASSE), 2025, pp. 1--8. [doi:10.1109/RASSE64831.2025.11315354](https://doi.org/10.1109/RASSE64831.2025.11315354).

[29] C. Calero, M. Piattini (Eds.), Green in Software Engineering, Springer, Cham, Switzerland, 2015. [doi:10.1007/978-3-319-08581-4](https://doi.org/10.1007/978-3-319-08581-4).

[30] L. Lannelongue, J. Grealey, M. Inouye, Green algorithms: Quantifying the carbon footprint of computation, Advanced Science 8 (12) (2021) 2100707. [doi:10.1002/advs.202100707](https://doi.org/10.1002/advs.202100707).

[31] R. Verdecchia, J. Sallou, L. Cruz, A systematic review of Green AI, WIREs Data Mining and Knowledge Discovery 13 (4) (2023) e1507. [doi:10.1002/widm.1507](https://doi.org/10.1002/widm.1507).

[32] V. Schmidt, K. Goyal, A. Joshi, B. Feld, L. Conell, N. Laskaris, D. Sarthou, H. Verreault, J. Blank, S. Zhang, Codecarbon: Estimate and track carbon emissions from machine learning computing, Journal of Open Source Software (2021).

[33] A. Noureddine, Joularjx: A java-based software power monitoring tool, SoftwareX 14 (2021) 100688.

[34] R. C. Cheung, A user-oriented software reliability model, IEEE Transactions on Software Engineering SE-6 (2) (1980) 118--125.

[35] K. Goseva-Popstojanova, K. S. Trivedi, Architecture-based approach to reliability assessment of software systems, Performance Evaluation 45 (2--3) (2001) 179--204.

[36] A. Immonen, E. Niemel\"a, Survey of reliability and availability prediction methods from the architectural perspective, Software and Systems Modeling 7 (1) (2008) 49--65.

[37] S. Becker, H. Koziolek, R. Reussner, The Palladio component model for model-driven performance prediction, Journal of Systems and Software 82 (1) (2009) 3--22.

[38] G. Franks, T. Al-Omari, M. Woodside, O. Das, S. Derisavi, Enhanced modeling and solution of layered queueing networks, IEEE Transactions on Software Engineering 35 (2) (2009) 148--161.

[39] J. Delange, P. H. Feiler, Architecture fault modeling with the AADL error-model annex, in: 2014 40th EUROMICRO Conference on Software Engineering and Advanced Applications (SEAA), IEEE, 2014, pp. 361--368. [doi:10.1109/SEAA.2014.20](https://doi.org/10.1109/SEAA.2014.20).

[40] Y. Gan, Y. Zhang, K. Hu, D. Cheng, Y. He, M. Pancholi, C. Delimitrou, Seer: Leveraging big data to navigate the complexity of performance debugging in cloud microservices, in: Proc. ACM Int. Conf. on Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2019.

[41] Y. Gan, M. Liang, S. Dev, D. Lo, C. Delimitrou, Sage: Practical and scalable ML-driven performance debugging in microservices, in: Proc. ACM Int. Conf. on Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2021.

[42] L. Wu, J. Tordsson, E. Elmroth, O. Kao, MicroRCA: Root cause localization of performance issues in microservices, in: Proc. IEEE/IFIP Network Operations and Management Symposium (NOMS), 2020.

[43] Z. Li, J. Chen, R. Jiao, N. Zhao, Z. Wang, S. Zhang, Y. Wu, L. Jiang, L. Yan, Z. Wang, Z. Chen, W. Zhang, X. Nie, K. Sui, D. Pei, Practical root cause localization for microservice systems via trace analysis, in: Proc. IEEE/ACM Int. Symposium on Quality of Service (IWQoS), 2021.

[44] C. Zhang, X. Peng, C. Sha, K. Zhang, Z. Fu, X. Wu, Q. Lin, D. Zhang, DeepTraLog: Trace-log combined microservice anomaly detection through graph-based deep learning, in: Proc. IEEE/ACM Int. Conf. on Software Engineering (ICSE), 2022.

[45] C. Lee, T. Yang, Z. Chen, Y. Su, Y. Yang, M. R. Lyu, Eadro: An end-to-end troubleshooting framework for microservices on multi-source data, in: Proc. IEEE/ACM Int. Conf. on Software Engineering (ICSE), 2023.

[46] X. Meng, P. Shen, Y. Sun, D. Liu, J. Lu, S. Zhang, D. Pei, Microcause: Root cause analysis for microservice systems through graph neural networks, in: Proc. IEEE International Conference on Software Maintenance and Evolution (ICSME), 2020, pp. 403--414.

[47] S. Zhang, S. Xia, W. Fan, B. Shi, X. Xiong, Z. Zhong, M. Ma, Y. Sun, D. Pei, Failure diagnosis in microservice systems: A comprehensive survey and analysis, arXiv preprint arXiv:2407.01710 (2024).

[48] X. Zhou, X. Peng, T. Xie, J. Sun, C. Ji, W. Li, D. Ding, Fault analysis and debugging of microservice systems: Industrial survey, benchmark system, and empirical study, IEEE Transactions on Software Engineering 47 (2) (2021) 243--260.

[49] V. R. Basili, L. C. Briand, W. L. Melo, A validation of object-oriented design metrics as quality indicators, IEEE Transactions on Software Engineering 22 (10) (1996) 751--761.

[50] N. Nagappan, T. Ball, Static analysis tools as early indicators of pre-release defect density, in: Proc. 27th Int. Conf. on Software Engineering (ICSE), 2005, pp. 580--586.

[51] T. Zimmermann, R. Premraj, A. Zeller, Predicting defects for Eclipse, in: Proc. 3rd Int. Workshop on Predictor Models in Software Engineering (PROMISE), 2007.

[52] T. Menzies, J. Greenwald, A. Frank, Data mining static code attributes to learn defect predictors, IEEE Transactions on Software Engineering 33 (1) (2007) 2--13.

[53] V. Bushong, D. Das, A. Al Maruf, T. Cerny, Using static analysis to address microservice architecture reconstruction, in: 2021 36th IEEE/ACM International Conference on Automated Software Engineering (ASE), IEEE, 2021. [doi:10.1109/ASE51524.2021.9678749](https://doi.org/10.1109/ASE51524.2021.9678749).

[54] S. Schneider, A. Bakhtin, X. Li, J. Soldani, A. Brogi, T. Cerny, R. Scandariato, D. Taibi, Comparison of static analysis architecture recovery tools for microservice applications, arXiv preprint (2024). [arXiv:2412.08352](http://arxiv.org/abs/2412.08352), [doi:10.48550/arXiv.2412.08352](https://doi.org/10.48550/arXiv.2412.08352).

[55] A. Santos, A. Cunha, N. Macedo, Statistical and model-driven static analysis of ROS systems, IEEE Transactions on Software Engineering 47 (10) (2019) 2200--2218.

[56] J. Garcia, D. Popescu, G. Edwards, N. Medvidovic, Toward a catalogue of architectural bad smells, in: Proc. 5th Int. Conf. on the Quality of Software Architectures (QoSA), LNCS 5581, 2009, pp. 146--162.

[57] D. Taibi, V. Lenarduzzi, On the definition of microservice bad smells, IEEE Software 35 (3) (2018) 56--62.

[58] Z. Li, P. Avgeriou, P. Liang, A systematic mapping study on technical debt and its management, Journal of Systems and Software 101 (2015) 193--220.

[59] J. Humble, D. Farley, Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation, Addison-Wesley, 2010.

[60] L. Chen, Continuous delivery: Huge benefits, but challenges too, IEEE Software 32 (2) (2015) 50--54.

[61] International Organization for Standardization, ISO/IEC 25023:2016 --- systems and software engineering --- systems and software quality requirements and evaluation (square) --- measurement of system and software product quality, Tech. rep., International Organization for Standardization (2016).

[62] International Organization for Standardization, ISO/IEC 25021:2012 --- systems and software engineering --- systems and software quality requirements and evaluation (square) --- quality measure elements, Tech. rep., International Organization for Standardization (2012).

[63] T. L. Saaty, The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation, McGraw-Hill, 1980.

[64] C. Fan, L. Zeng, Y. Sun, Y.-Y. Liu, Finding key players in complex networks through deep reinforcement learning, Nature Machine Intelligence 2 (2020) 317--324.

[65] C. Fan, L. Zeng, Y. Ding, M. Chen, Y. Sun, Z. Liu, Learning to identify high betweenness centrality nodes from scratch: A novel graph neural network approach, in: Proc. 28th ACM Int. Conf. on Information and Knowledge Management (CIKM), 2019, pp. 559--568.

[66] A. Varbella, K. Amara, M. El-Assady, B. Gjorgiev, G. Sansavini, PowerGraph: A power grid benchmark dataset for graph neural networks, in: Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track, 2024, arXiv:2402.02827.

[67] T. N. Kipf, M. Welling, Semi-supervised classification with graph convolutional networks, in: Proc. Int. Conf. on Learning Representations (ICLR), 2017.

[68] W. L. Hamilton, R. Ying, J. Leskovec, Inductive representation learning on large graphs, in: Advances in Neural Information Processing Systems 30 (NeurIPS), 2017, pp. 1024--1034.

[69] P. Velickovi\'c, G. Cucurull, A. Casanova, A. Romero, P. Li\`o, Y. Bengio, Graph attention networks, in: Proc. Int. Conf. on Learning Representations (ICLR), 2018.

[70] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, M. Welling, Modeling relational data with graph convolutional networks, in: Proc. European Semantic Web Conference (ESWC), 2018, pp. 593--607.

[71] X. Wang, H. Ji, C. Shi, B. Wang, Y. Ye, P. Cui, P. S. Yu, Heterogeneous graph attention network, in: Proc. The Web Conference (WWW), 2019, pp. 2022--2032.

[72] Z. Hu, Y. Dong, K. Wang, Y. Sun, Heterogeneous graph transformer, in: Proc. The Web Conference (WWW), 2020, pp. 2704--2710.

[73] X. Fu, J. Zhang, Z. Meng, I. King, MAGNN: Metapath aggregated graph neural network for heterogeneous graph embedding, in: Proc. The Web Conference (WWW), 2020, pp. 2331--2341.

[74] G. Khodabandeh, A. Ezaz, M. Babaei, N. Ezzati-Jivan, Utilizing graph neural networks for effective link prediction in microservice architectures, in: Proceedings of the 16th ACM/SPEC International Conference on Performance Engineering (ICPE), 2025.

[75] Z. Ying, D. Bourgeois, J. You, M. Zitnik, J. Leskovec, GNNExplainer: Generating explanations for graph neural networks, in: Advances in Neural Information Processing Systems (NeurIPS), Vol. 32, 2019, pp. 9244--9255.

[76] D. Luo, W. Cheng, D. Xu, W. Yu, B. Zong, H. Chen, X. Zhang, Parameterized explainer for graph neural network, in: Advances in Neural Information Processing Systems (NeurIPS), Vol. 33, 2020, pp. 19620--19631.

[77] J. Pearl, Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference, Morgan Kaufmann, 1988.

[78] G. Beliakov, A. Pradera, T. Calvo, Aggregation functions: A guide for practitioners, Studies in Fuzziness and Soft Computing 221 (2007).

[79] R. R. Yager, On ordered weighted averaging aggregation operators in multicriteria decisionmaking, IEEE Transactions on Systems, Man, and Cybernetics 18 (1) (1988) 183--190.

[80] G. H. Hardy, J. E. Littlewood, G. P\'olya, Inequalities, 2nd Edition, Cambridge University Press, 1952.

[81] U.S. Department of Defense, MIL-STD-498: Software development and documentation, Military standard, U.S. Department of Defense (1994).

[82] M. Fey, J. E. Lenssen, Fast graph representation learning with PyTorch geometric, in: ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

[83] F. Xia, T.-Y. Liu, J. Wang, W.-S. Zhang, H. Li, Listwise approach to learning to rank: Theory and algorithm, in: Proc. 25th Int. Conf. on Machine Learning (ICML), 2008, pp. 1192--1199.

[84] Team SimPy, Simpy: Discrete event simulation for Python, Software, <https://simpy.readthedocs.io> (accessed 9 September 2026) (2020).

[85] F. Wilcoxon, Individual comparisons by ranking methods, Biometrics Bulletin 1 (6) (1945) 80--83.

[86] B. Efron, R. J. Tibshirani, An Introduction to the Bootstrap, Chapman \& Hall, 1993.

[87] C. Spearman, The proof and measurement of association between two things, American Journal of Psychology 15 (1) (1904) 72--101.

[88] R. Schwartz, J. Dodge, N. A. Smith, O. Etzioni, Green AI, Communications of the ACM 63 (12) (2020) 54--63. [doi:10.1145/3381831](https://doi.org/10.1145/3381831).

[89] E. Strubell, A. Ganesh, A. McCallum, Energy and policy considerations for deep learning in NLP, in: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), Florence, Italy, 2019, pp. 3645--3650. [doi:10.18653/v1/P19-1355](https://doi.org/10.18653/v1/P19-1355).

[90] D. Patterson, J. Gonzalez, Q. Le, C. Liang, L.-M. Munguia, D. Rothchild, D. So, M. Texier, J. Dean, Carbon emissions and large neural network training, arXiv preprint arXiv:2104.10350 (2021). [doi:10.48550/arXiv.2104.10350](https://doi.org/10.48550/arXiv.2104.10350).

[91] S. Georgiou, M. Kechagia, T. Sharma, F. Sarro, Y. Zou, Green AI: Do deep learning frameworks have different costs?, in: Proceedings of the 44th International Conference on Software Engineering (ICSE), 2022, pp. 1082--1094. [doi:10.1145/3510003.3510221](https://doi.org/10.1145/3510003.3510221).

[92] H. Zeng, H. Zhou, A. Srivastava, R. Kannan, V. Prasanna, GraphSAINT: Graph sampling based inductive engine, in: Proc. International Conference on Learning Representations (ICLR), 2020.

[93] I. O. Yigit, F. Buzluca, [dataset] software-as-a-graph: Replication package (datasets, generator configurations, simulation harnesses, model checkpoints, and analysis scripts), <https://doi.org/10.5281/zenodo.14922108> (2026). [doi:10.5281/zenodo.14922108](https://doi.org/10.5281/zenodo.14922108).
