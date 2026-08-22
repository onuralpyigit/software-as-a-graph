# Graph Learning for Reliability and Dependability Analysis in Complex Distributed Systems

**Authors.** *[Omitted for double-anonymised review.]*

**Affiliations.** *[Omitted for double-anonymised review.]*

**Corresponding author.** *[Omitted for double-anonymised review.]*

---

# Abstract

Modern distributed software systems—including publish–subscribe middleware and microservice meshes—heavily decouple components in space and time. While this decoupling enables horizontal scalability and autonomous lifecycle management, it conceals the complex dependency pathways along which cascading failures propagate. Because operational telemetry does not exist prior to deployment, and source-code static analysis cannot observe distributed topology, identifying systemically critical components and explaining their vulnerability remains a fundamental challenge. 

We present **Software-as-a-Graph (SaG)**, a pre-deployment Static System Analysis framework that models complex distributed architectures as typed, weighted, directed multigraphs over five entity classes (Applications, Brokers, Topics, Nodes, and Libraries) with logical dependencies derived through typed projection rules. On this representation, we formulate a relation-specific **Heterogeneous Graph Transformer (HGL)** to forecast cascading failure impact and detect critical components, complemented by an interpretable **Reliability–Maintainability (RM)** attribution baseline grounded in the ISO/IEC 25010/25019 quality models. Both predictors are evaluated against discrete-event cascade simulators operating on a structurally disjoint view of the system under a strict input–label independence guarantee.

Across seven synthetic topologies and three authentic open-source distributed systems (Autoware.universe ROS 2, a Cloud-Native Microservices mesh, and the Train-Ticket benchmark), our empirical evaluation yields four principal findings:
1. **Graph Learning Efficacy:** Typed graph learning consistently outperforms non-learning structural baselines, demonstrating its strongest advantage in critical-set identification ($F_1@K = 0.467$ vs. $0.339$, a $+37.8\%$ relative gain) and inductive out-of-distribution ranking ($\rho = 0.668$ vs. $0.521$).
2. **Value of Heterogeneity and the QoS Trade-off:** Relation-specific message passing is the primary driver of cross-architectural generalisation, with typed GNNs outperforming homogeneous graph attention networks by $+0.197$ in rank correlation under Leave-One-Scenario-Out (LOSO) validation. Explicit QoS edge encoding has an asymmetric effect: it mildly reduces in-distribution accuracy ($\Delta\rho = -0.072$) yet is the single largest driver of LOSO generalisation, making the QoS-aware variant the best-performing configuration overall under distribution shift.
3. **Multi-Dimensional Attribution:** The hierarchical RM profile successfully disentangles error propagation (fault tolerance) from single-point-of-failure vulnerabilities (availability), directing engineers toward targeted architectural remediations — though RM's standalone ranking accuracy remains weak (LOSO $\rho \approx -0.01$), so its role is diagnostic attribution rather than a competing predictor.
4. **Real-World Transferability:** SaG successfully generalizes to complex real-world architectures, achieving strong rank agreement ($\rho = 0.688$ to $0.778$) and robust critical-set containment across both cyber-physical and cloud-native paradigms.

**Keywords:** Graph representation learning; heterogeneous graph neural networks; publish–subscribe middleware; distributed systems dependability; cascading failures; static system analysis; architectural quality models.

---

# 1. Introduction

## 1.1 Motivation

The publish–subscribe (pub-sub) and asynchronous event-driven paradigms have become core architectural abstractions for large-scale distributed systems, spanning cyber-physical systems, autonomous vehicles, cloud-native microservices, robotics, and the Internet of Things (IoT). By decoupling producers and consumers in time, space, and synchronization, pub-sub architectures allow independent components to scale, evolve, and fail without direct coordination [1]. Standardized middleware—such as the Data Distribution Service (DDS) [2], MQTT [3], Apache Kafka [43], and ROS 2 [44]—exposes rich deployment-time Quality-of-Service (QoS) configurations (including reliability, durability, and deadline policies) that govern message routing and system behavior under stress.

However, the very decoupling that makes asynchronous distributed systems flexible also creates a profound visibility barrier when reasoning about dependability. In pub-sub and event-driven architectures, there are no explicit caller–callee invocations: a publishing process maintains no direct static reference to its consumers, even when those consumers are entirely dependent upon its data stream. Consequently, cascading failures do not follow traditional call graphs. Instead, failures propagate along *derived* logical paths across message brokers, shared topics, colocated execution nodes, and shared software libraries. When a shared library crashes, it induces a *simultaneous failure* across all consuming applications rather than a sequential cascade. Traditional architectural block diagrams and static call graphs fail to reveal these multi-layer failure pathways, meaning that the components whose failure would cause the most catastrophic downstream outage are rarely the ones that appear prominent in high-level diagrams.

Addressing this vulnerability is most cost-effective *prior to deployment*, during the design and continuous integration (CI/CD) stages, where structural refactoring and redundancy insertion are cheapest. Yet, pre-deployment is precisely the phase where no runtime telemetry, distributed tracing, or operational logs exist. Software architects and reliability engineers must therefore answer two critical questions solely from design-time descriptors:
1. *Which components and communication links are systemically critical to system availability and reliability?*
2. *Why are they critical, and what specific architectural intervention (e.g., broker replication, topic decoupling, library sandboxing) will most effectively mitigate that risk?*

Uncontained cascading failures in production also carry substantial sustainability and operational costs: emergency failovers, retransmission storms, and compute-heavy restart loops waste energy and infrastructure resources that proactive architectural hardening avoids.

## 1.2 Problem Statement and Limitations of Existing Work

We formulate pre-deployment dependability analysis for complex distributed systems as two coupled tasks:
1. **Interpretable Criticality Attribution:** Computing a multi-dimensional, standards-grounded quality profile for every component and dependency—derived from ISO/IEC 25010 [16] and ISO/IEC 25019 [17]—to explain *how* and *why* a component is vulnerable.
2. **Failure-Impact Forecasting:** Accurately predicting the global cascade blast radius and ranking the most failure-critical components using learned topological representations, enabling engineers to prioritize hardening under budget constraints.

Existing techniques fall short of bridging what we term the **"Architecture–Code Gap"**: a distributed system may exhibit pristine, bug-free source code within each component, yet remain systemically fragile due to hidden topological single points of failure (SPOFs) or mismatched middleware QoS contracts. Three prevailing paradigms leave this gap unaddressed:

- **Static Code Analysis (SCA):** Tools such as SonarQube evaluate code complexity [29], cohesion, and modularity [28, 30] at the individual service level. However, SCA is entirely blind to inter-component distributed topology, middleware message queues, and cross-host failure propagation.
- **Runtime Chaos Engineering and Telemetry:** Approaches such as Chaos Monkey [18] and distributed tracing inject real faults into running staging or production clusters. While valuable, they require fully operational infrastructure, carry operational risks, and operate too late in the software development lifecycle to guide design-time architecture.
- **Homogeneous and Topology-Only Network Analysis:** Traditional graph centrality measures (betweenness, PageRank, degree) [4, 5, 37, 38] collapse complex systems into flat, unweighted networks. They discard node and edge semantics, conflating entirely distinct failure modes (such as sequential event propagation vs. simultaneous library crashes). Similarly, homogeneous Graph Neural Networks (GNNs) [39, 40, 41] flatten heterogeneous middleware interactions, losing relation-specific propagation patterns.

No existing approach provides a unified, pre-deployment framework that combines typed multigraph representations, code-level quality ingestion, heterogeneous graph learning, and interpretable quality attribution.

## 1.3 The Software-as-a-Graph (SaG) Approach

To overcome these limitations, we introduce **Software-as-a-Graph (SaG)**, a pre-deployment **Static System Analysis (SSA)** framework. SaG operates on system architecture descriptions (expressed as Architecture-as-Code manifests) and executes a four-stage analysis pipeline:

1. **Typed Multigraph Formulation:** SaG models the distributed system as a typed, weighted, directed multigraph over five core entity types: Applications, Brokers, Topics, Execution Nodes, and Shared Libraries (§3.1).
2. **QoS-Aware Logical Dependency Projection:** Using six formal projection rules, SaG derives a semantic `DEPENDS_ON` dependency layer that explicitly captures both sequential cascades (via topics and brokers) and simultaneous blast radii (via shared libraries and node colocation), weighted by declared QoS contracts (§3.2).
3. **Interpretable Quality Attribution:** SaG integrates code-level SCA metrics with topological properties into a hierarchical **Reliability–Maintainability (RM)** attribution model (§4). Reliability decomposes into **Fault Tolerance** (error propagation depth) and **Availability** (structural articulation and SPOF exposure), providing explainable diagnostics for reliability engineers.
4. **Graph Learning for Failure Forecasting:** SaG deploys a **Heterogeneous Graph Transformer (HGL)** that performs relation-specific attention and message passing across typed entities to forecast cascading failure impacts (§5).

To guarantee methodological rigor, the framework enforces an **input–label independence guarantee**: the learned and attribution predictors operate strictly on the derived analytical graph $G_{\text{analysis}}$, whereas ground-truth failure impacts are generated by discrete-event simulators executing over the raw structural topology $G_{\text{structural}}$ (§5.3).

```
+-----------------------------------------------------------------------------------+
|                            Software-as-a-Graph (SaG)                              |
+-----------------------------------------------------------------------------------+
|  Architecture Descriptor (Apps, Topics, Brokers, Hosts, Libraries, QoS Policies)   |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       | Raw Structural Graph (G_structural)   |
                       +-------------------+-------------------+
                                           |
                   [Typed Projection Rules | & QoS Weighting]
                                           v
                       +---------------------------------------+
                       |  Analysis Multigraph (G_analysis)     |
                       |       (Derived DEPENDS_ON Edges)      |
                       +-------------------+-------------------+
                                           |
                 +-------------------------+-------------------------+
                 |                                                   |
                 v                                                   v
+---------------------------------+                 +---------------------------------+
|   Interpretable RM Baseline     |                 |  Heterogeneous Graph Transformer|
|  - Fault Tolerance (Propagation)|                 |             (HGL)               |
|  - Availability (SPOF / Bridge) |                 |  - Relation-Specific Attention  |
|  - Maintainability + Code SCA   |                 |  - Inductive Impact Forecasting |
+----------------+----------------+                 +----------------+----------------+
                 |                                                   |
                 +-------------------------+-------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|             Predicted Criticality Rankings & Architectural Diagnoses               |
+------------------------------------------+----------------------------------------+
                                           |
           [Validated Against Independent Simulation on G_structural]
                                           v
+-----------------------------------------------------------------------------------+
| Ground-Truth Cascade Simulation: Reachability Loss, Queue Latency, Feed Disruption|
+-----------------------------------------------------------------------------------+
```
*Figure 1. End-to-end architecture of the Software-as-a-Graph (SaG) framework.*

## 1.4 Research Questions

Our investigation evaluates the predictive power, interpretability, and generalizability of SaG across four research questions:

- **RQ1 (Predictive Efficacy):** How accurately does typed graph learning predict cascading failure impact and identify critical component sets compared to non-learning topological baselines, both in-distribution and under inductive cross-scenario evaluation?
- **RQ2 (Value of Heterogeneity):** What failure mechanisms and structural vulnerabilities does typed heterogeneity reveal that homogeneous graph representations and single-scalar centralities obscure?
- **RQ3 (Ablations and Sensitivity):** How do explicit QoS feature encodings and multi-attribute weighting calibrations affect model performance and ranking stability?
- **RQ4 (Real-World Generalization):** Does the graph learning framework successfully transfer to authentic, real-world open-source distributed software architectures across different architectural paradigms?

## 1.5 Key Contributions

This paper makes the following primary contributions:
1. **A Formal Typed Architecture Model:** A multigraph representation of distributed pub-sub and event-driven systems that derives logical dependencies and distinguishes sequential cascade propagation from simultaneous multi-consumer library failures (§3).
2. **Heterogeneous Graph Learning for Dependability:** A relation-specific Heterogeneous Graph Transformer (HGL) tailored for pre-deployment cascading failure prediction and critical component detection (§5).
3. **Standards-Grounded Interpretable Attribution:** An explainable Reliability–Maintainability (RM) quality baseline grounded in ISO/IEC 25010/25019 that bridges code-level SCA metrics with system-level topological criticality (§4).
4. **Empirical Evaluation and Scope Conditions:** A rigorous empirical evaluation across seven synthetic topologies (1,545 components) and three real-world open-source systems (Autoware.universe ROS 2, Cloud-Native Microservices, and Train-Ticket; 225 components) under strict input–label independence, establishing where typed graph learning provides decisive advantages (§6–§7).

## 1.6 Paper Organization

The remainder of this paper is structured as follows: Section 2 reviews related work in distributed systems dependability, static system analysis, and graph representation learning. Section 3 formalizes the Software-as-a-Graph architectural model and dependency projection rules. Section 4 presents the interpretable RM attribution baseline. Section 5 details the Heterogeneous Graph Transformer formulation and simulation ground truth. Section 6 describes the experimental setup, datasets, and protocols. Section 7 presents empirical results for RQ1–RQ4. Section 8 discusses architectural implications, threats to validity, and conclusions.

---

# 2. Related Work

This research intersects four major domains: distributed systems dependability, static system analysis, graph neural networks for network vulnerability, and software quality models.

## 2.1 Dependability in Distributed and Pub-Sub Systems

The publish–subscribe paradigm provides foundational communication decoupling for distributed systems [1]. Formal middleware specifications—such as DDS [2], MQTT [3], ROS 2 [44], and distributed commit logs like Kafka [43]—enable fine-grained QoS policies governing durability, liveliness, and transport reliability. Prior research on pub-sub dependability has predominantly focused on runtime mechanisms: fault-tolerant event routing, broker clustering, dynamic consensus, and adaptive message retransmission.

In parallel, runtime verification and chaos engineering [18] systematically inject faults into staging or production clusters to observe steady-state recovery. While essential for validating operational resilience, runtime techniques operate late in the software lifecycle, require complete infrastructure deployments, and cannot evaluate architectural alternatives during initial design. Our work addresses the complementary, pre-deployment phase: predicting cascading vulnerability from architectural models *before* systems are deployed.

## 2.2 Static Code Analysis vs. Static System Analysis

Traditional Static Code Analysis (SCA) analyzes source code abstract syntax trees (ASTs) to measure cyclomatic complexity [29], module coupling (e.g., LCOM, CBO) [28, 30], and code duplication. While SCA effectively identifies internal code smells and defect-prone modules [55, 56, 57, 58], it is entirely oblivious to inter-process communication, container placement, and middleware topologies. 

To bridge this "Architecture–Code Gap," **Static System Analysis (SSA)** elevates static analysis to the system architecture level. By modeling distributed components, message channels, and infrastructure hosts as a global dependency graph, SSA frameworks propagate code-level quality attributes across topological links. This enables continuous pre-deployment verification in modern CI/CD pipelines [19, 20], catching structural anti-patterns [21, 22, 23, 24] and architectural degradation [26, 27] at commit time.

## 2.3 Network Science and Graph Representation Learning

Network science offers established metrics for identifying critical network elements, including degree, closeness, betweenness centrality [4, 37], articulation points, and PageRank [5, 38]. Studies on random failure tolerance [35], cascading overload [36], and interdependent networks [6] have provided deep mathematical foundations for systemic vulnerability. However, standard network metrics suffer from two fundamental limitations when applied to software architectures:
1. *Dimensional Collapse:* A single centrality score cannot distinguish a structural Single Point of Failure (SPOF) from an error-propagating cascade hub or a high-maintenance bottleneck.
2. *Semantic Collapse:* Discarding node and edge types treats an asynchronous pub-sub topic the same as a physical host or a shared library, masking distinct failure propagation mechanics.

To overcome the limitations of hand-engineered graph metrics, recent research has applied machine learning on graphs for network dismantling and criticality estimation (e.g., FINDER [7], DrBC [8], and PowerGraph [9]). However, most existing models rely on homogeneous message passing (GCN [39], GraphSAGE [40], GAT [41]). Because distributed middleware is inherently multi-typed and heterogeneous, homogeneous aggregation mixes fundamentally incompatible semantic relationships. Heterogeneous Graph Neural Networks (RGCN [10], HAN [11], HGT [12], MAGNN [13]) address this by parameterizing relation-specific transformations. We build upon the Heterogeneous Graph Transformer (HGT) architecture [12] to maintain typed relational semantics during cascade impact forecasting.

## 2.4 Software Quality Models and Multi-Criteria Evaluation

Software product quality is standardized by the ISO/IEC 25010:2023 product quality model [16] (comprising nine characteristics including Reliability, Maintainability, and Security) and the ISO/IEC 25019:2023 Quality-in-Use standard [17] (evaluating stakeholder harm over Beneficialness, Freedom from Risk, and Acceptability). Quality evaluation distinguishes between *internal quality* (measured statically on software artifacts at rest) and *external quality* (measured dynamically on executing systems) [53, 59]. 

Synthesizing multi-attribute structural metrics into actionable quality indices constitutes a classic Multi-Criteria Decision Making (MCDM) problem. The Analytic Hierarchy Process (AHP) [15] provides a mathematically rigorous pairwise-comparison framework equipped with an explicit Consistency Ratio ($CR \le 0.10$) to validate expert weighting schemes. In this work, we use AHP to construct an audited, interpretable Reliability–Maintainability attribution baseline that serves as a transparent benchmark for our learned GNN models.

---

# 3. The Software-as-a-Graph (SaG) Architectural Model

This section formalizes the Software-as-a-Graph multigraph representation (§3.1), the QoS-aware weighting and logical dependency derivation rules (§3.2), and the dual graph views utilized throughout the framework (§3.3).

## 3.1 Formal Multigraph Definition

We model a complex distributed system as a typed, weighted, directed multigraph:

$$\mathcal{G} = (V, E, \tau_V, \tau_E, w_V, w_E)$$

where:
- $V$ is the set of system entities, partitioned into five disjoint types:
  $$V = V_{\text{app}} \cup V_{\text{broker}} \cup V_{\text{topic}} \cup V_{\text{node}} \cup V_{\text{lib}}$$
- $E$ is the set of directed edges connecting entities.
- $\tau_V: V \to \mathcal{T}_V$ and $\tau_E: E \to \mathcal{T}_E$ are typing functions assigning node and edge categories.
- $w_V: V \to [0, 1]$ and $w_E: E \to [0, 1]$ are weighting functions representing entity criticality and coupling strength.

**Table 1. Entity and structural edge types in the SaG model.**

| Entity Type ($\mathcal{T}_V$) | Architectural Role | Concrete System Examples |
|:---|:---|:---|
| **Application** ($V_{\text{app}}$) | Autonomous process that produces or consumes messages | ROS 2 node, Kafka microservice, MQTT client |
| **Broker** ($V_{\text{broker}}$) | Message routing and queuing intermediary | RabbitMQ exchange, Mosquitto, EMQX broker |
| **Topic** ($V_{\text{topic}}$) | Named logical communication channel | `/sensor/lidar`, `orders.payment.completed` |
| **Node** ($V_{\text{node}}$) | Physical host or virtualized execution environment | Bare-metal server, Kubernetes worker, Cloud VM |
| **Library** ($V_{\text{lib}}$) | Shared software package or driver dependency | `librdkafka`, OpenCV, protocol buffer runtime |

| Structural Edge ($\mathcal{T}_E$) | Direction | Semantic Meaning |
|:---|:---|:---|
| `PUBLISHES_TO` | Application/Library $\to$ Topic | Component publishes messages to topic |
| `SUBSCRIBES_TO` | Application/Library $\to$ Topic | Component consumes messages from topic |
| `ROUTES` | Broker $\to$ Topic | Broker manages and routes topic traffic |
| `RUNS_ON` | Application/Broker $\to$ Node | Process is hosted on physical/virtual host |
| `CONNECTS_TO` | Node $\to$ Node | Physical network link between hosts |
| `USES` | Application $\to$ Library | Application links to shared library dependency |

Application and Library entities additionally ingest static code metrics computed via SCA tools (`cm_*` attributes: lines of code, cyclomatic complexity, coupling between objects, LCOM), directly bridging code-level fragility with topological analysis.

## 3.2 QoS-Aware Weights and Logical Dependency Derivation

In distributed middleware, communication links exhibit varying degrees of coupling based on their Quality-of-Service (QoS) contracts. For example, a `RELIABLE` topic with `TRANSIENT_LOCAL` durability binds communicating services far more tightly than a `BEST_EFFORT` telemetry stream.

For each structural communication edge $e$, we compute an edge coupling weight $w_E(e) \in [0, 1]$ derived from the QoS profile of the mediating topic:

$$w_E(e) = 0.85 \cdot \left( w_{\text{rel}} \cdot q_{\text{rel}} + w_{\text{dur}} \cdot q_{\text{dur}} + w_{\text{prio}} \cdot q_{\text{prio}} \right) + 0.15 \cdot q_{\text{payload}}$$

where $q_{\text{rel}}, q_{\text{dur}}, q_{\text{prio}} \in [0, 1]$ represent normalized reliability, durability, and transport priority scores, with AHP weights $(0.30, 0.40, 0.30)$ ($CR < 0.05$). Durability is assigned the highest weight because it dictates message persistence across network partitions. A minimum floor $w_E(e) \ge 0.01$ ensures that best-effort edges remain visible to graph traversals.

### Logical Dependency Projection (`DEPENDS_ON`)
Structural edges capture explicit deployment connections but omit implicit runtime dependencies. For example, a subscriber depends upon a publisher, yet no direct edge connects them in pub-sub architectures. We derive a single unified semantic relation, `DEPENDS_ON`, directed from *dependent* to *dependency* ("if target fails, source is impacted"):

**Table 2. The six `DEPENDS_ON` logical dependency projection rules.**

| Rule | Dependency Category | Structural Pattern ($\text{Dependent} \to \text{Dependency}$) | Derived Weight ($w$) |
|:---:|:---|:---|:---|
| **1** | `app_to_app` | Subscriber $\to$ Publisher (via shared Topic, incl. transitive `USES`) | $\max_{t} w_E(t)$ |
| **2** | `app_to_broker` | Publisher/Subscriber $\to$ Broker routing its topics | $\max_{t} w_E(t)$ |
| **3** | `node_to_node` | Host $\to$ Host (lifted from inter-host app dependencies) | Lifted $\max w$ |
| **4** | `node_to_broker` | Host $\to$ Broker (lifted from hosted app dependencies) | Lifted $\max w$ |
| **5** | `app_to_lib` | Application $\to$ Shared Library it `USES` | $w_V(\text{app})$ |
| **6** | `broker_to_broker` | Broker $\leftrightarrow$ Broker (bidirectional, colocated on same host) | $w_V(\text{node})$ |

### Sequential Cascades vs. Simultaneous Blasts
A key insight of the SaG model is distinguishing between two fundamentally different failure modes:
- **Sequential Cascade (Rule 1):** When an application publisher fails, downstream subscribers experience data starvation. The failure propagates step-by-step through topics and message queues.
- **Simultaneous Blast (Rule 5):** When a shared software library or execution node crashes, all consuming applications and colocated brokers fail *instantaneously* in a single event. 

Retaining entity types and relation-specific dependency rules allows SaG to model both mechanisms, whereas untyped graphs collapse them into identical edges.

## 3.3 Dual Graph Views and Architectural Layers

The SaG framework maintains two distinct representations of the system:
1. **Structural Graph ($G_{\text{structural}}$):** The raw graph containing physical and structural edges (`PUBLISHES_TO`, `ROUTES`, `RUNS_ON`, `USES`). This view is consumed exclusively by discrete-event simulators to execute unbiased failure injections (§5.1).
2. **Analysis Graph ($G_{\text{analysis}}$):** The projected graph containing derived `DEPENDS_ON` edges annotated with QoS weights and ingested SCA code metrics. All GNN feature representations, graph embeddings, and analytical metrics are computed on $G_{\text{analysis}}$.

$G_{\text{analysis}}$ is further organized into four analytical layers (Application, Middleware, Infrastructure, and Global System), allowing architects to evaluate criticality at subsystem levels (e.g., following MIL-STD-498 hierarchical structures).

---

# 4. Interpretable Attribution as a Quality Baseline

To provide actionable architectural explanations alongside data-driven predictive rankings, SaG decomposes component and relationship criticality into a multi-dimensional, standards-grounded quality attribution profile.

## 4.1 Grounding in ISO/IEC Standards

Following **ISO/IEC 25010:2023** (Product Quality Model) [16] and **ISO/IEC 25019:2023** (Quality-in-Use) [17], we formulate two core formal criticality constructs:
- **Component Criticality ($D_1$):** The degree to which the sudden failure, unexpected termination, or severe degradation of an individual component reduces the system's capacity to deliver required services within its operational context of use.
- **Relationship Criticality ($D_2$):** The degree of systemic service degradation resulting from the severance, partitioning, or failure of a specific dependency or communication channel while both endpoint components remain operational.

Criticality is evaluated primarily across two orthogonal quality characteristics: **Reliability ($R$)** and **Maintainability ($M$)**.

**Table 3. The Reliability–Maintainability (RM) quality decomposition.**

| Dimension | Sub-Characteristic | Architectural Question | Underlying Graph Metrics | Role / Remediation |
|:---|:---|:---|:---|:---|
| **Reliability ($R$)** | **Fault Tolerance ($FT$)** | How broadly does failure propagate? | Transpose Reverse PageRank on $G^\top$, in-degree, cascade depth potential | Reliability Engineer: add fallback/redundancy |
| | **Availability ($A$)** | Is this a structural single point of failure? | Directed articulation point score, bridge centrality, QoS load | SRE / DevOps: replicate host / cluster broker |
| **Maintainability ($M$)** | **Modularity / Modifiability** | How complex and tightly coupled is this component? | Betweenness centrality, out-degree, Code Quality Penalty (SCA) | Software Architect: refactor code, decouple APIs |

*Coverage Scope:* SaG focuses on Reliability and Maintainability. Safety (which requires domain-specific hazard logs, e.g., ISO 26262 ASIL ratings) and Security (which requires explicit threat models, e.g., STRIDE) are declared external to purely structural analysis and require domain-specific hazard analysis (e.g., ISO 26262 ASIL).

## 4.2 Typed Node Feature Representation

SaG extracts rich feature vectors tailored to the 5 distinct entity types in the software multigraph:
- **Application ($|V_{\text{app}}|$, 23 dims):** Indices 0--17 represent shared topological metrics (in/out degree, betweenness, closeness, reverse PageRank, clustering coefficient, articulation score, bridge load). Indices 18--22 capture source code metrics extracted via Static Code Analysis (SCA): Lines of Code (LOC), Cyclomatic Complexity, Martin's Instability metric ($I_{\text{code}} = \frac{C_e}{C_a + C_e}$), Lack of Cohesion in Methods (LCOM), and composite Code Quality Penalty (CQP).
- **Library ($|V_{\text{lib}}|$, 25 dims):** Same shared topological (0--17) and code quality (18--22) metrics as Application, plus two library-specific extras (indices 23--24): the size of the transitive reverse-`USES` closure (normalized) and the count of distinct subscribers reachable from that closure's published topics (normalized) — the two structural drivers of a library's blast radius under both simulators' cascade rules that the code-quality metrics alone do not capture.
- **Broker ($|V_{\text{broker}}|$, 19 dims):** Indices 0--17 shared topological metrics; index 18 represents normalized queue buffer capacity.
- **Topic ($|V_{\text{topic}}|$, 22 dims):** Indices 0--17 shared topological metrics; indices 18--21 capture publisher count, subscriber count, log message frequency $\log(1 + \text{freq})$, and ordinal QoS criticality.
- **Infrastructure Node ($|V_{\text{node}}|$, 20 dims):** Indices 0--17 shared topological metrics; indices 18--19 capture normalized CPU core allocation and physical memory (RAM).

## 4.3 Composite Quality Score Formulation

All raw topological and code metrics are rank-normalized to $[0, 1]$. The quality sub-characteristics are formulated hierarchically using the Analytic Hierarchy Process (AHP) [15]:

1. **Fault Tolerance ($FT(v)$):** Measures error cascade potential on the transpose graph $G_{\text{analysis}}^\top$ (where edges follow failure propagation from dependency to dependent):
   $$FT(v) = 0.45 \cdot \text{RPR}(v) + 0.30 \cdot \text{Deg}_{\text{in}}(v) + 0.25 \cdot \text{CDPot}_{\text{enh}}(v)$$
   where RPR is Reverse PageRank and $\text{CDPot}_{\text{enh}}$ is an enhanced Cascade Depth Potential term.
2. **Availability ($A(v)$):** Identifies structural single points of failure (SPOFs) across five terms — directed articulation severity, its QoS-weighted variant, edge-level irrecoverability, connectivity degradation, and the component's own QoS weight:
   $$A(v) = 0.35 \cdot \text{AP}_c^{\text{dir}}(v) + 0.25 \cdot \text{QSPOF}(v) + 0.25 \cdot \text{BR}(v) + 0.10 \cdot \text{CDI}(v) + 0.05 \cdot w(v)$$
3. **Reliability ($R(v)$):** Blends Fault Tolerance and Availability hierarchically:
   $$R(v) = \alpha \cdot FT(v) + (1 - \alpha) \cdot A(v), \quad \alpha = 0.36$$
   Intra-dimension pairwise comparison matrices are audited against Saaty's consistency ratio and measure $CR = 0.001$ (Fault Tolerance), $CR = 0.001$ (Availability), and $CR = 0.000$ (Maintainability) — all well within the $CR \le 0.10$ acceptability threshold. The shipped intra-dimension weights are a $\lambda = 0.70$ shrinkage blend between the raw AHP-derived vector and a uniform prior (§7.3 reports the sensitivity of ranking accuracy to $\lambda$).
4. **Maintainability ($M(v)$):** Evaluates structural coupling combined with code-level static analysis across five terms — betweenness, QoS-weighted efferent coupling, the Code Quality Penalty, an afferent/efferent coupling-risk imbalance term, and inverse clustering:
   $$M(v) = 0.35 \cdot \text{BT}(v) + 0.30 \cdot w_{\text{out}}(v) + 0.15 \cdot \text{CQP}(v) + 0.12 \cdot \text{CouplingRisk}_{\text{enh}}(v) + 0.08 \cdot (1 - \text{CC}(v))$$

The baseline composite quality score $Q(v)$ combines both dimensions:
$$Q(v) = 0.80 \cdot R(v) + 0.20 \cdot M(v)$$

When evaluating under a specific ISO/IEC 25019 Context of Use vector $\vec{\omega} = [q_R, q_M]^\top$, the score is reweighted dynamically:
$$Q_{\text{domain}}(v) = q_R \cdot R(v) + q_M \cdot M_{\text{static}}(v)$$

Components are categorized into adaptive criticality tiers using box-plot quartile thresholds:
- **CRITICAL:** $Q > Q_3 + 1.5 \cdot \text{IQR}$
- **HIGH:** $Q_3 < Q \le Q_3 + 1.5 \cdot \text{IQR}$
- **MEDIUM:** $Q_1 < Q \le Q_3$
- **MINIMAL:** $Q \le Q_1$

This enables targeted diagnostics: a service scoring high on $A$ but low on $FT$ is diagnosed as a pure SPOF requiring horizontal replication, whereas a service scoring high on $FT$ is an error cascade hub requiring circuit breakers and bulkhead isolation.

---

# 5. Graph Learning for Failure-Impact Prediction

While the interpretable RM baseline provides explainable quality profiles, complex non-linear failure interactions and non-local cascade propagations benefit significantly from graph representation learning. This section details our Heterogeneous Graph Transformer (HGT) architecture, 16-dimensional edge representation, multi-task prediction heads, dimension-masked loss formulation, ground-truth simulation oracles, and the strict input--label independence guarantee.

## 5.1 Ground-Truth Simulation Oracles

To evaluate predictive accuracy prior to deployment without relying on production runtime telemetry, SaG executes discrete-event failure simulations over the raw structural multigraph $G_{\text{structural}}$. We establish a formal three-oracle taxonomy:

- **Cascade Reachability Oracle ($I^*(v)$):** Implemented via `FaultInjector`, this oracle injects an unexpected node crash at component $v \in V$, propagates cascading failures across dependent topics, brokers, and network links using breadth-first dynamic traversal, and computes the fraction of surviving subscriber feeds severed by the outage. $I^*(v) \in [0, 1]$ serves as the primary continuous target label for training and evaluating GNN predictors.
- **Multi-Metric Composite Oracle ($I_{\text{comp}}(v)$):** Implemented via `FailureSimulator`, this oracle evaluates a multi-faceted failure impact vector:
  $$I_{\text{comp}}(v) = 0.35 \cdot \Delta\text{Reachability} + 0.25 \cdot \Delta\text{Fragmentation} + 0.25 \cdot \Delta\text{Throughput} + 0.15 \cdot \Delta\text{FlowDisruption}$$
  $I_{\text{comp}}(v)$ serves as the canonical oracle for architectural quality gate verification.
- **Dynamic Queue-Flow Oracle ($I_{\text{dyn}}(v)$):** Implemented via `MessageFlowSimulator` using the SimPy discrete-event simulation framework, this oracle models message emission rates, stochastic network latencies, broker buffer saturation, and dropped delivery counts under fault injection.
- **Relationship (Edge) Removal Oracle ($I_{\text{edge}}(u,v)$):** Evaluates the systemic impact of severing an individual dependency or communication channel while keeping both endpoint components alive:
  $$I_{\text{edge}}(u,v) = I_{\text{comp}}(G \setminus \{(u,v)\}) - I_{\text{comp}}(G)$$

## 5.2 Heterogeneous Graph Transformer Architecture

Because distributed systems comprise heterogeneous entity types (Applications, Libraries, Brokers, Topics, Infrastructure Nodes) and diverse interaction semantics (`CALLS`, `PUBLISHES_TO`, `SUBSCRIBES_TO`, `HOSTED_ON`, etc.), we implement a 3-layer **Heterogeneous Graph Transformer (HGT)** architecture [12] with hidden dimension $D = 64$ and 4 attention heads.

```
+-----------------------------------------------------------------------------------+
|               Heterogeneous Graph Transformer (HGT) Architecture                  |
+-----------------------------------------------------------------------------------+
|  Node Features (19-23 dims) + 16-dim QoS Edge Encodings injected into destinations|
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
*Figure 2. Layered architecture of the Heterogeneous Graph Transformer (HGT) predictor.*

### 5.2.1 Edge Feature Encoding (16-Dimensional)

To capture continuous QoS constraints and channel semantics, SaG encodes each directed edge $e = (u, v)$ as a 16-dimensional continuous-categorical vector $e_{uv} \in \mathbb{R}^{16}$:
- **Index 0:** Scalar QoS weight $w(e) \in (0, 1]$, computed via the harmonic mean of reliability, durability, priority, deadline, and blocking multipliers.
- **Index 1:** Normalized path count through edge $e$ in $G_{\text{analysis}}$.
- **Indices 2--8:** 7-bit one-hot encoding for edge relationship types (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`, `DEPENDS_ON`).
- **Indices 9--15:** 7 explicit QoS profile dimensions, non-zero only for `PUBLISHES_TO`/`SUBSCRIBES_TO` edges (all other edge types receive zeros): (9) Reliability policy score (0.0 best-effort / 1.0 reliable); (10) Durability policy score (0.0 volatile / 0.5 transient-local / 0.6 transient / 1.0 persistent); (11) Normalized message priority (0.0/0.33/0.66/1.0 for low/medium/high/urgent); (12) Deadline constraint active flag; (13) Log deadline $\log_{10}(1 + \text{deadline\_ns} / 10^6)$; (14) Log max blocking time $\log_{10}(1 + \text{max\_blocking\_ms})$; and (15) a QoS heterogeneity flag (1.0 if the edge's reliability/durability/priority triple deviates from the scenario's modal QoS profile, else 0.0).

The `EdgeFeatureEncoder` projects $e_{uv}$ into the hidden space ($e_{uv}' = W_{\text{edge}} e_{uv}$) and adds it directly to the destination node representation prior to relational attention: $\tilde{h}_v = h_v + e_{uv}'$.

## 5.3 Multi-Task Prediction Heads and Dimension Masking

From the final node embeddings $h_v^{(L)}$, SaG branches into specialized multi-task prediction heads:
- **Reliability Head:** $\hat{R}(v) = \sigma(\text{MLP}_R(h_v)) \in [0, 1]$
- **Maintainability Head:** $\hat{M}(v) = \sigma(\text{MLP}_M(h_v)) \in [0, 1]$
- **Composite Failure Impact Head:** $\hat{I}^*(v) = \sigma(\text{MLP}_C(h_v \parallel \hat{R}(v) \parallel \hat{M}(v))) \in [0, 1]$
- **Relationship Criticality Head:** $\hat{Q}(u,v) = \sigma(\text{TypedEdgeEncoder}_{\phi(e)}(h_u, h_v, e_{uv})) \in [0, 1]$

### Dimension-Masked Loss Formulation

The joint optimization objective balances regression accuracy, multi-task dimension learning, ranking fidelity, pairwise ordering, and edge prediction:
$$\mathcal{L} = \mathcal{L}_{\text{composite}} + 0.5 \cdot \mathcal{L}_{\text{dimension}} + 0.3 \cdot \mathcal{L}_{\text{rank}} + 0.1 \cdot \mathcal{L}_{\text{pairwise}} + 0.1 \cdot \mathcal{L}_{\text{consistency}} + 0.3 \cdot \mathcal{L}_{\text{edge}}$$
where $\mathcal{L}_{\text{composite}} = \text{MSE}(\hat{I}^*(v), I^*(v))$, $\mathcal{L}_{\text{rank}}$ is ListMLE ranking loss [49], $\mathcal{L}_{\text{pairwise}}$ is margin-ranking loss, and $\mathcal{L}_{\text{consistency}} = \text{MSE}(\hat{I}^*(v), 0.8\hat{R}(v) + 0.2\hat{M}(v))$.

**Dimension Masking:** Because dynamic cascade simulation ($I^*(v)$ via `FaultInjector`) observes runtime failure reachability rather than static code maintainability, maintainability ground truth is unobserved during dynamic simulation. We introduce a boolean dimension mask $m = [m_R, m_M] = [1, 0]$:
$$\mathcal{L}_{\text{dimension}} = \frac{1}{\sum_{d} m_d} \sum_{d \in \{R, M\}} m_d \cdot \text{MSE}(\hat{d}(v), d^*(v))$$
This prevents the unmeasured maintainability head from being penalized or driven toward zero by backpropagation.

### Domain-Reweighted Criticality ($Q_{\text{domain}}$)

To ground predictions in ISO/IEC 25019 Context of Use ($\vec{\omega} = [q_R, q_M]^\top$), the composite score is evaluated as:
$$Q_{\text{domain}}(v) = q_R \cdot \hat{R}(v) + q_M \cdot M_{\text{static}}(v)$$
where $M_{\text{static}}(v)$ is drawn directly from the structural analyzer's maintainability baseline ($y_{\text{rm}}$ maintainability column), combining learned dynamic reliability with static source-code maintainability. The implementation deliberately refuses to emit $Q_{\text{domain}}(v)$ unless the checkpoint's maintainability head actually received real supervision — under the sole labeler used in every experiment reported here (`FaultInjector`, $m = [1, 0]$), maintainability is unmeasured, so $Q_{\text{domain}}(v)$ is never populated in these results. Domain reweighting sensitivity is instead evaluated directly against the static RM baseline, reported as a dedicated ablation (§7.3).

## 5.4 Input–Label Independence Guarantee

To guarantee experimental rigor and eliminate data leakage, SaG enforces strict architectural decoupling:
- **Feature Space:** Derived exclusively from $G_{\text{analysis}}$ (using static structural topology, SCA metrics, and declared QoS contracts).
- **Label Space:** Evaluated exclusively on raw $G_{\text{structural}}$ through independent simulation oracles (`FaultInjector`, `FailureSimulator`, `MessageFlowSimulator`).

No simulation outputs are ever exposed as input features to the GNN or attribution baseline.

---

# 6. Experimental Setup

## 6.1 Datasets and System Corpus

Our evaluation corpus comprises 1,770 components across ten distributed system architectures:

**Table 4. Experimental evaluation corpus.**

| Dataset / Architecture | System Paradigm | Total Nodes ($|V|$) | Apps ($|V_{\text{app}}|$) | Topics | Brokers | Hosts | Libraries | Relationships ($|E|$) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| **Autonomous Vehicle (AV)** | ROS 2 Cyber-Physical | 152 | 80 | 40 | 4 | 8 | 20 | 797 |
| **Enterprise Pub-Sub** | Kafka Event Mesh | 520 | 300 | 120 | 10 | 40 | 50 | 3,245 |
| **Financial Trading** | Low-Latency Pub-Sub | 124 | 60 | 35 | 5 | 6 | 18 | 580 |
| **Healthcare Integration** | HL7/FHIR Event Mesh | 98 | 50 | 25 | 3 | 8 | 12 | 400 |
| **Hub-and-Spoke Enterprise** | Broker-Centric Messaging | 139 | 70 | 30 | 2 | 12 | 25 | 797 |
| **IoT Smart City** | MQTT Telemetry Mesh | 326 | 200 | 80 | 6 | 30 | 10 | 1,322 |
| **Microservices Mesh** | Cloud-Native Services | 186 | 90 | 45 | 6 | 15 | 30 | 680 |
| **Autoware.universe [45]** | Real-World ROS 2 Autoware | 75 | 32 | 24 | 3 | 6 | 10 | 179 |
| **Cloud Microservices [47]** | Real-World GCP Boutique | 60 | 22 | 20 | 4 | 6 | 8 | 128 |
| **Train-Ticket Benchmark [46]** | Real-World Microservices | 90 | 41 | 30 | 3 | 8 | 8 | 162 |
| **Total** | | **1,770** | | | | | | |

$|V|$ is the sum of all five entity-type counts per scenario, confirmed against `data/scenarios/MANIFEST.json` and the three real-world adapter outputs; it sums to exactly the corpus total stated above. $|E|$ counts every raw structural relationship instance recorded in the scenario file (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`), i.e. the substrate the simulation oracles traverse — not the derived `DEPENDS_ON` projection used for GNN training, which is denser (e.g. 3,753 edges for AV at the `system` layer).

The seven synthetic scenarios are generated using statistical topology generators spanning diverse degree distributions, clustering coefficients, and middleware QoS configurations. The three real-world architectures are hand-transcribed from open-source repositories using dedicated architectural adapters.

## 6.2 Baselines and Evaluated Predictors

We compare five primary predictor configurations:
1. **Topo-BL:** Unweighted structural centrality (betweenness centrality and articulation point scoring).
2. **Topo-QoS:** QoS-weighted topological centrality baseline.
3. **RM / $Q(v)$:** Deterministic hierarchical quality attribution baseline (§4).
4. **GL / GL-QoS:** Homogeneous Graph Attention Networks (GAT) [41] trained on the flattened, untyped graph projection.
5. **HGL / HGL-QoS:** Relation-specific Heterogeneous Graph Transformers (§5).

## 6.3 Evaluation Metrics and Protocols

- **Ranking Accuracy:** Spearman rank correlation ($\rho$) and Kendall's tau ($\tau$) between predicted rankings and ground-truth simulated impact $I^*(v)$.
- **Critical-Set Identification:** $F_1@K$, Precision@$K$, and Recall@$K$ for top-$K$ critical components, where $K = \lceil 0.20 \cdot |V_{\text{app}}| \rceil$.
- **Statistical Significance:** Paired Wilcoxon signed-rank tests [48] ($p < 0.05$) and bootstrap 95% confidence intervals ($B = 2,000$) [49, 50].

### Evaluation Protocols
- **In-Distribution Evaluation:** 60% train / 20% validation / 20% test node splits pinned by node identity within each scenario, evaluated over five random seeds $\{42, 123, 456, 789, 2024\}$.
- **Inductive Leave-One-Scenario-Out (LOSO):** Across all eight cached scenarios (the seven synthetic scenarios plus the ATM case study, §7.4), models are trained on the remaining seven and evaluated zero-shot on the held-out scenario, for eight folds total, to test out-of-distribution generalizability.
- **Real-World Architectural Transfer:** Evaluating zero-shot transfer on authentic open-source architectures.

---

# 7. Results and Empirical Analysis

## 7.1 RQ1: Graph Learning vs. Structural Baselines

Table 5 presents the in-distribution held-out Spearman rank correlation ($\rho$) against simulated cascade impact $I^*(v)$ across all seven synthetic scenarios.

**Table 5. In-distribution held-out Spearman rank correlation ($\rho$) against $I^*(v)$** (seed means over 5 seeds with bootstrap 95% CIs; $n$ is held-out Application count).

| Scenario | $n$ | Topo-BL | Topo-QoS | GL (Homogeneous) | GL-QoS | HGL (Typed) | HGL-QoS |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **AV System** | 16 | 0.283 [−0.01, 0.54] | 0.701 [0.46, 0.87] | **0.790** [0.71, 0.87] | 0.689 [0.44, 0.87] | 0.789 [0.69, 0.88] | 0.649 [0.57, 0.75] |
| **Enterprise** | 60 | 0.420 [0.34, 0.51] | 0.789 [0.75, 0.83] | 0.875 [0.84, 0.91] | 0.459 [−0.10, 0.79] | 0.880 [0.84, 0.91] | **0.891** [0.86, 0.92] |
| **Financial Trading** | 12 | 0.289 [0.01, 0.56] | 0.700 [0.57, 0.82] | 0.672 [0.26, 0.91] | 0.536 [−0.09, 0.89] | **0.854** [0.77, 0.91] | 0.770 [0.50, 0.92] |
| **Healthcare** | 10 | −0.101 [−0.33, 0.14] | **0.772** [0.59, 0.88] | 0.590 [0.34, 0.76] | 0.463 [−0.12, 0.83] | 0.652 [0.46, 0.81] | 0.645 [0.28, 0.85] |
| **Hub-and-Spoke** | 14 | 0.234 [0.04, 0.34] | 0.359 [0.08, 0.62] | **0.554** [0.37, 0.72] | 0.052 [−0.35, 0.46] | 0.534 [0.51, 0.56] | 0.296 [−0.24, 0.59] |
| **IoT Smart City** | 40 | −0.063 [−0.17, 0.06] | 0.073 [−0.06, 0.20] | 0.238 [−0.20, 0.65] | 0.259 [−0.18, 0.63] | **0.881** [0.85, 0.91] | 0.842 [0.77, 0.90] |
| **Microservices** | 18 | 0.401 [0.15, 0.65] | **0.707** [0.54, 0.86] | 0.564 [0.39, 0.72] | 0.571 [0.46, 0.70] | 0.483 [0.19, 0.71] | 0.476 [0.18, 0.71] |
| **Mean** | — | **0.209** [0.11, 0.31] | **0.586** [0.48, 0.68] | **0.612** [0.49, 0.71] | **0.433** [0.25, 0.59] | **0.725** [0.65, 0.79] | **0.653** [0.52, 0.75] |

**Table 6. Paired Wilcoxon signed-rank tests across scenarios** ($n = 7$, two-sided).

| Comparison | $\Delta\rho$ | Scenarios Won | Wilcoxon $W$ | $p$-value | Significance |
|:---|---:|:---:|---:|---:|:---|
| **HGL vs. Topo-BL** | **+0.516** | 7/7 | 0.0 | **0.0156** | **Statistically Significant** ($p < 0.05$) |
| **HGL vs. GL-QoS** | **+0.292** | 6/7 | 1.0 | **0.0312** | **Statistically Significant** ($p < 0.05$) |
| **HGL vs. Topo-QoS** | **+0.139** | 5/7 | 9.0 | 0.469 | Not significant |
| **GL vs. Topo-QoS** | **+0.026** | 4/7 | 11.0 | 0.688 | Not significant |
| **HGL vs. GL** | **+0.113** | 4/7 | 9.0 | 0.469 | Not significant |
| **HGL-QoS vs. HGL** | **−0.072** | 1/7 | 3.0 | 0.078 | Not significant (marginal; HGL-QoS trails HGL in-distribution) |

### Out-of-Distribution (LOSO) Generalization
Under inductive Leave-One-Scenario-Out (LOSO) cross-validation, the model must predict cascading criticality on an unseen system topology:

**Table 7. Inductive Leave-One-Scenario-Out (LOSO) evaluation.**

| Model Variant | Mean LOSO $\rho$ | Std $\rho$ | Critical-Set $F_1@K$ | Requires Training |
|:---|---:|---:|---:|:---:|
| **Topo-BL** | 0.109 | 0.150 | 0.222 | No |
| **Topo-QoS** | 0.522 | 0.285 | 0.339 | No |
| **RM / $Q(v)$ Baseline** | −0.014 | 0.147 | 0.237 | No |
| **GL (Homogeneous)** | 0.381 | 0.164 | 0.436 | Yes |
| **GL-QoS (Homogeneous)** | 0.501 | 0.167 | 0.442 | Yes |
| **HGL (Typed Heterogeneous)** | 0.578 | **0.116** | **0.467** | Yes |
| **HGL-QoS (Typed + QoS)** | **0.668** | 0.096 | 0.457 | Yes |

Eight LOSO folds are reported (the seven synthetic scenarios plus the ATM
case study of §7.4), each holding out one scenario and training on the
remaining seven.

### Key Findings for RQ1:
1. **Critical-Set Detection Advantage:** HGL achieves a 37.8% relative improvement in critical-set identification ($F_1@K = 0.467$ vs. $0.339$ for Topo-QoS).
2. **Out-of-Distribution Superiority:** Under LOSO validation, the QoS-aware typed model (HGL-QoS) is the best-performing configuration overall ($\rho = 0.668$ vs. $0.521$ for Topo-QoS), substantially outperforming both non-learning baselines and homogeneous GNNs ($\rho = 0.381$--$0.501$). The RM baseline is essentially non-predictive under distribution shift ($\rho \approx -0.01$), confirming that RM's value lies in interpretable attribution rather than standalone ranking.
3. **In-Distribution Scope Condition:** In-distribution, both HGL ($\rho = 0.725$) and GL ($\rho = 0.612$) outperform structural baselines on point estimates, but their margin over Topo-QoS is constrained by topology variance. Notably, in-distribution the QoS-aware variant (HGL-QoS, $\rho = 0.653$) trails the base typed model rather than matching it — see §7.3's QoS Feature Encoding discussion for the full in-distribution/LOSO trade-off.

## 7.2 RQ2: Value of Typed Heterogeneity

To evaluate the contribution of node and edge typing, we compare relation-specific HGL against homogeneous GL across different validation regimes:

- **In-Distribution:** Heterogeneous HGL leads homogeneous GL by $\Delta\rho = +0.113$ (0.725 vs. 0.612; not statistically significant, $p = 0.469$, 4/7 scenarios won). On familiar topologies, homogeneous GNNs can partially approximate type distinctions through structural degree signatures, but the typed model still holds a consistent point-estimate edge.
- **Out-of-Distribution (LOSO):** Heterogeneous HGL outperforms homogeneous GL by **$+0.197$** ($\rho = 0.578$ vs. $0.381$) and exhibits substantially lower variance across folds ($\sigma = 0.116$ vs. $0.164$). When encountering unseen topologies, relation-specific message passing is essential for generalizing failure mechanics — and the gap widens further to $+0.167$ when comparing the QoS-aware variants (HGL-QoS $0.668$ vs. GL-QoS $0.501$), the best-performing configuration in either family.

### Empirical Edge-Removal Analysis
We further probed edge criticality by simulating the removal of candidate relationship channels while keeping both endpoint components alive. On the `av_system` topology across 50 candidate bridge edges, 46 edges exhibited zero downstream cascade impact, while the 4 edges with non-zero impact were direct library communication channels (`PUBLISHES_TO` / `SUBSCRIBES_TO`). This demonstrates that individual communication links are largely redundant, whereas component-level failures and shared library dependencies induce disproportionate systemic disruption.

## 7.3 RQ3: Ablations and Sensitivity Analysis

### QoS Feature Encoding
Explicitly encoding continuous QoS attributes in the GNN (HGL-QoS) does not yield a uniform effect — it trades in-distribution accuracy for out-of-distribution generalization. In-distribution, HGL-QoS *trails* the base typed model ($\rho = 0.653$ vs. $0.725$; $\Delta\rho = -0.072$, marginal at $p = 0.078$, HGL-QoS wins only 1/7 scenarios), consistent with QoS features adding redundant signal (and mild overfitting risk) on topologies the model has already seen, since the derived `DEPENDS_ON` graph already embeds much of the same routing and coupling information. Under LOSO, the relationship reverses sharply: HGL-QoS is the single best-performing configuration in the entire study ($\rho = 0.668$ vs. $0.578$ for HGL; $\Delta\rho = +0.090$). When the topology itself is unseen, the explicit QoS signal generalizes in a way the structural encoding alone does not. We read this as evidence that QoS encoding is not "redundant" so much as *situational*: valuable insurance against distribution shift, at a small in-distribution cost.

### Intra-Dimension AHP Weight Shrinkage
We evaluated the sensitivity of the RM attribution baseline by sweeping a shrinkage parameter $\lambda \in [0, 1]$ blending internal term weights from uniform prior ($\lambda = 0$) to calibrated AHP judgement ($\lambda = 1$):

**Table 8. Sensitivity of RM rank correlation to AHP shrinkage parameter $\lambda$.**

| $\lambda$ Setting | 0.00 (Uniform) | 0.50 | 0.70 (Shipped Default) | 0.80 | 1.00 (Raw AHP Judgement) |
|:---|---:|---:|---:|---:|---:|
| **Mean Rank Correlation ($\rho$)** | −0.0514 | −0.0256 | −0.0190 | −0.0142 | **−0.0069** |

As illustrated in Figure 4, the calibrated AHP judgement ($\lambda = 1.00$) is monotonically less anti-correlated with ground truth than uniform weighting, improving by $+0.045\,\rho$ across the sweep and validating the internal consistency of the expert hierarchy. We note explicitly that every $\lambda$ setting remains negative on this population: the RM baseline is weakly *anti*-correlated with $I^*(v)$ across the whole shrinkage range, consistent with its near-zero-to-negative LOSO performance (Table 7). AHP calibration makes the RM baseline less wrong, not accurate; its primary role remains explainable attribution rather than standalone ranking.

### Convergent Validity Across Simulation Oracles
To test construct validity, we compared node criticality rankings across our three simulation engines: the topological cascade reachability oracle $I^*(v)$ (`FaultInjector`), the multi-metric composite oracle $I_{\text{comp}}(v)$ (`FailureSimulator`), and the discrete-event queue-flow simulator $I_{\text{dyn}}(v)$ (`MessageFlowSimulator`).

Across all seven synthetic scenarios, the behavioural SimPy simulator ($I_{\text{dyn}}$) and the topological cascade reachability oracle ($I^*$) exhibited strong rank convergence:
- **$I_{\text{dyn}}$ vs. $I^*(v)$:** Mean Spearman $\rho = \mathbf{0.7647}$ (ranging from $0.5497$ to $0.9378$), with mean Kendall $\tau = 0.633$ and top-$K$ Jaccard overlap of $0.312$.
- **$I_{\text{comp}}$ vs. $I_{\text{dyn}}$:** Mean Spearman $\rho = 0.4646$ (ranging from $0.1208$ to $0.6575$).
- **$I_{\text{comp}}$ vs. $I^*(v)$:** Mean Spearman $\rho = 0.3970$ (ranging from $0.0974$ to $0.5778$).

This strong agreement between independent behavioural queue simulations and structural cascade reachability confirms that our continuous target label $I^*(v)$ captures genuine dynamic service disruption.

### Domain-Specific Weighting Sensitivity
We investigated the impact of Context of Use reweighting $\vec{\omega} = [q_R, q_M]^\top$ across 10 evaluation scenarios by sweeping the composite reliability weight $w_R$ over its full range and reading off three points on that one curve: the shipped static default ($w_R = 0.80$), an unweighted equal split ($w_R = 0.50$), and the domain-derived value per scenario. Domain-derived weighting improves mean rank correlation by $\Delta\rho = +0.0264$ over the static default, but *underperforms* equal weighting by $\Delta\rho = -0.1097$ — domain derivation is not, on this evidence, a ranking-accuracy improvement over the simplest possible baseline. The reason is structural rather than a modelling failure: across all six declared domains, the domain-derived $w_R$ sits within $0.04$ of the static $0.80$ default (range $[0.70, 0.76]$), so mean Kendall rank fidelity between domain-derived and static rankings is $\tau = 0.9677$ — the two weightings rank components almost identically, and the free parameter Context of Use actually moves is too small for any weighting confined to it to move $\rho$ substantially. We therefore reframe this ablation's conclusion: operational Context of Use reweighting is an *attributional* mechanism — explaining why a component matters in stakeholder terms — not a ranking-improvement device, consistent with the scoping in the quality model (§4.1).

### Threshold and Normalization Sensitivity
We swept 7 scenarios $\times$ a cascade propagation-threshold parameter $\in \{0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0\}$ controlling the ground-truth oracle's eligibility cutoff for cascade propagation, and separately swept 3 label-normalization techniques (min-max, standard, rank-quantile) holding the threshold fixed. Ranking performance is more sensitive to the propagation threshold ($\Delta\rho = 0.1896$ across the extreme settings; mean $\rho$ ranges from $-0.173$ at threshold $0$ to $+0.017$ at threshold $\geq 0.35$, plateauing thereafter) than to normalization choice ($\Delta\rho = 0.0158$ across the three schemes). A small spread under either sweep means the reported ordering is stable under that free parameter of the ground truth and scorer — it is not, by itself, evidence that the ordering is *correct*.

### Anti-Pattern Detection and CI/CD Quality Gates
Validating our rule-based architectural anti-pattern catalog against the multi-metric composite oracle $I_{\text{comp}}(v)$ yielded a mean detection $F_1 = 0.3781$ across 8 system scenarios — but $F_1$ alone understates what the catalog does: mean precision is $0.239$ against mean recall $0.900$, meaning the catalog flags $94.2\%$ of all components on average and trades precision for near-exhaustive coverage of true positives. This is a deliberate high-recall CI/CD posture (a missed critical component is more costly than a false positive requiring manual triage) rather than an accuracy shortfall, but should be read as such rather than via $F_1$ in isolation. One detector, `DEEP_PIPELINE`, is excluded from this evaluation: it enumerates every simple source-to-sink path and does not terminate within ten minutes on the 50-application Healthcare topology — the blowup is itself a reportable scalability limit of exhaustive path enumeration, not a measurement gap. Analysis execution times for the remaining 18 detectors ranged from $0.04\,\text{s}$ (small scenarios, 50 nodes) to $54.85\,\text{s}$ (enterprise event meshes, 300 nodes), confirming that SaG runs comfortably within standard pre-commit and pull-request CI/CD quality gate budgets.

We additionally stratify the RM composite's rank correlation by node type: $\rho = 0.503$ (Application), $0.395$ (Broker), $0.142$ (Node), while the *pooled* correlation across all types is $\rho = 0.028$ — outside the per-type range entirely. This is a Simpson's-paradox effect (aggregating across populations with different scales/base rates reverses the within-group trend), and it means the pooled figure is not a summary of the per-type figures: only the stratified, per-type numbers should be quoted when discussing RM's structural predictive power.

### HGT Attention Weight Analysis
Extracting relational attention matrices from the trained HGT model on the ATM Case Study (Figure 3) revealed that the model places its highest single attention weight ($\alpha_{uv} = 1.00$) on a `USES` edge (an Application's dependency on a shared Library), with the next tier ($\alpha_{uv} \approx 0.50$) split between a `ROUTES` edge (Broker $\to$ Topic) and further `USES`/`SUBSCRIBES_TO` edges. This pattern — dominant attention on library-dependency and broker-routing channels rather than direct publish/subscribe message flow — reflects the shared-library blast-radius and broker-centrality failure pathways identified elsewhere in this study (§7.2's edge-removal analysis, §3's simultaneous-failure semantics for shared libraries) rather than sequential pub-sub cascade propagation.

## 7.4 RQ4: Real-World Distributed Software Architecture Validation

To assess external validity on authentic production architectures, we evaluated SaG on three open-source distributed systems.

**Table 9. Empirical validation on authentic real-world distributed software architectures.**

| Real-World Architecture | Total Nodes | Apps ($|V_{\text{app}}|$) | Spearman $\rho$ (Mean $\pm$ Std) | Kendall $\tau$ | Critical-Set $F_1@K$ | Tie-Robust $F_1@K$ | Non-Zero Impact Apps | Predictive Gain (vs. Degree) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Autoware.universe (ROS 2)** | 75 | 32 | **0.688 $\pm$ 0.009** | 0.517 | **0.800** | 0.800 | 19 / 32 | +0.360 |
| **Cloud Microservices Mesh** | 60 | 22 | **0.778 $\pm$ 0.001** | 0.639 | **1.000** | 0.760 | 8 / 22 | +0.014 |
| **Train-Ticket Booking Mesh** | 90 | 41 | **0.759 $\pm$ 0.001** | 0.605 | **1.000** | 0.810 | 14 / 41 | +0.264 |

### Key Findings for RQ4:
1. **Strong Real-World Rank Agreement:** SaG achieves high rank correlation on Cloud Microservices ($\rho = 0.778$) and Train-Ticket ($\rho = 0.759$), and solid agreement on Autoware.universe ($\rho = 0.688$).
2. **Critical-Set Containment:** Every single application with non-zero cascading impact in Cloud Microservices and Train-Ticket is successfully captured within the predicted top-$K$ critical set (tie-robust $F_1@K = 0.760$ and $0.810$).
3. **Substantial Predictive Gain:** SaG outperforms raw degree centrality by $+0.360$ on Autoware and $+0.264$ on Train-Ticket, demonstrating that typed dependency derivation captures critical architectural semantics beyond superficial connectivity.

---

# 8. Discussion, Threats to Validity, and Conclusion

## 8.1 Discussion and Practical Implications

Our empirical findings provide clear guidance for software architects and reliability engineers:
- **When to Use Graph Learning:** Heterogeneous GNNs provide the greatest return when the objective is identifying the critical top-$K$ shortlist of components for architectural hardening ($F_1@K = 0.467$, HGL), and when evaluating new, unseen system architectures out of distribution ($\rho = 0.668$, HGL-QoS). The choice between the base typed model and its QoS-aware variant is itself a design decision: HGL for in-distribution accuracy, HGL-QoS when the deployment target is expected to differ structurally from the training corpus (§7.3).
- **When to Use Interpretable Attribution:** For root-cause diagnosis and remediation planning, the deterministic RM profile is indispensable. Distinguishing whether a service is critical due to single-point-of-failure exposure (Availability) or wide error propagation (Fault Tolerance) directly dictates whether to introduce load-balanced replicas or circuit-breaker policies.

## 8.2 Threats to Validity

- **Construct Validity:** Our ground truth is generated via discrete-event cascade simulation on structural models rather than observing live production outages. To mitigate construct divergence, we evaluated multiple independent simulation engines (`FaultInjector`, `FailureSimulator`, `MessageFlowSimulator`) and confirmed strong convergent validity between dynamic queue-flow simulation and cascade reachability ($\rho = 0.765$).
- **Internal Validity:** Potential feature leakage is prevented by strict graph view separation: predictors operate on $G_{\text{analysis}}$, whereas ground-truth simulators operate on $G_{\text{structural}}$. We additionally identified and fixed a normalization defect in the code-quality scoring pipeline: a min-max helper previously assigned the maximal Code Quality Penalty to any zero-variance population, which is correct for one genuine node but was indistinguishable from "many nodes carrying no code-quality data at all" — the exact shape of every Library in our three real-world scenarios, which carry no source-level `code_metrics`. The fix (scoring an undifferentiated multi-node population as zero rather than maximal penalty) changes Library Maintainability tier classifications in the real-world corpus but, as expected for a fix confined to a population with no variance to rank, leaves Application-population rank correlation against $I^*(v)$ effectively unchanged (Table 9: $\Delta\rho \in \{-0.003, 0.000, 0.000\}$ across the three real-world scenarios).
- **External Validity:** While our corpus spans ten distinct system domains and three authentic open-source architectures (ROS 2 and microservices), future work should evaluate larger enterprise deployments with hundreds of microservices.
- **Conclusion Validity:** Given the heavy-tailed, non-normal distribution of cascading failure impacts, all statistical comparisons utilize non-parametric rank correlation (Spearman $\rho$, Kendall $\tau$), bootstrap confidence intervals ($B = 2,000$), and paired Wilcoxon signed-rank tests. We note one aggregation hazard directly: the RM composite's rank correlation pooled across node types ($\rho = 0.028$) falls outside the range spanned by its own per-type correlations ($\rho \in [0.14, 0.50]$, §7.3) — a Simpson's-paradox effect that would mislead if the pooled figure were quoted on its own. We report and interpret only stratified figures for this reason.

## 8.3 Limitations and Future Work

1. **Safety and Security Integration:** SaG currently focuses on Reliability and Maintainability. Incorporating formal hazard categories (e.g., ISO 26262 ASIL ratings) and security threat models into the multigraph schema represents an important direction for future research.
2. **Hardware-in-the-Loop (HIL) Validation:** Validating SaG's predictions against real-time physical fault-injection testbeds in cyber-physical environments.
3. **Automated Architectural Refactoring:** Extending SaG from predictive analysis to prescriptive synthesis—automatically generating pull requests that reconfigure QoS policies and add redundancy.

## 8.4 Conclusion

We presented **Software-as-a-Graph (SaG)**, a pre-deployment Static System Analysis framework that models complex distributed pub-sub and microservice architectures as typed, weighted multigraphs. By combining relation-specific Heterogeneous Graph Transformers with an interpretable Reliability–Maintainability quality attribution model, SaG bridges the Architecture–Code Gap, forecasting cascading failure risks before systems are deployed. 

Our empirical results across synthetic and authentic open-source distributed systems demonstrate that heterogeneous graph learning provides significant advantages in identifying critical components ($F_1@K = 0.467$) and generalizing across unseen architectures ($\rho = 0.668$), while delivering actionable architectural diagnostics for resilient distributed systems engineering.

---

# References

[1] P. T. Eugster, P. A. Felber, R. Guerraoui, A.-M. Kermarrec, "The many faces of publish/subscribe," *ACM Computing Surveys*, vol. 35, no. 2, pp. 114–131, 2003.

[2] Object Management Group, "Data Distribution Service (DDS)," OMG Document formal/2015-04-10, version 1.4, 2015.

[3] OASIS, "MQTT Version 5.0," OASIS Standard, 2019.

[4] L. C. Freeman, "A set of measures of centrality based on betweenness," *Sociometry*, vol. 40, no. 1, pp. 35–41, 1977.

[5] S. Brin, L. Page, "The anatomy of a large-scale hypertextual web search engine," *Computer Networks and ISDN Systems*, vol. 30, no. 1–7, pp. 107–117, 1998.

[6] S. V. Buldyrev, R. Parshani, G. Paul, H. E. Stanley, S. Havlin, "Catastrophic cascade of failures in interdependent networks," *Nature*, vol. 464, pp. 1025–1028, 2010.

[7] C. Fan, L. Zeng, Y. Sun, Y.-Y. Liu, "Finding key players in complex networks through deep reinforcement learning," *Nature Machine Intelligence*, vol. 2, pp. 317–324, 2020.

[8] C. Fan, L. Zeng, Y. Ding, M. Chen, Y. Sun, Z. Liu, "Learning to identify high betweenness centrality nodes from scratch: A novel graph neural network approach," in *Proc. 28th ACM Int. Conf. on Information and Knowledge Management (CIKM)*, 2019, pp. 559–568.

[9] A. Varbella, K. Amara, M. El-Assady, B. Gjorgiev, G. Sansavini, "PowerGraph: A power grid benchmark dataset for graph neural networks," in *Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track*, 2024. arXiv:2402.02827.

[10] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, M. Welling, "Modeling relational data with graph convolutional networks," in *Proc. European Semantic Web Conference (ESWC)*, 2018, pp. 593–607.

[11] X. Wang, H. Ji, C. Shi, B. Wang, Y. Ye, P. Cui, P. S. Yu, "Heterogeneous graph attention network," in *Proc. The Web Conference (WWW)*, 2019, pp. 2022–2032.

[12] Z. Hu, Y. Dong, K. Wang, Y. Sun, "Heterogeneous graph transformer," in *Proc. The Web Conference (WWW)*, 2020, pp. 2704–2710.

[13] X. Fu, J. Zhang, Z. Meng, I. King, "MAGNN: Metapath aggregated graph neural network for heterogeneous graph embedding," in *Proc. The Web Conference (WWW)*, 2020, pp. 2331–2341.

[14] Q. Li, Z. Han, X.-M. Wu, "Deeper insights into graph convolutional networks for semi-supervised learning," in *Proc. AAAI Conference on Artificial Intelligence*, 2018, pp. 3538–3545.

[15] T. L. Saaty, *The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation*, McGraw-Hill, 1980.

[16] ISO/IEC 25010:2023, "Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Product quality model," International Organization for Standardization, 2023.

[17] ISO/IEC 25019:2023, "Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Quality-in-use model," International Organization for Standardization, 2023.

[18] A. Basiri, N. Behnam, R. de Rooij, L. Hochstein, L. Kosewski, J. Reynolds, C. Rosenthal, "Chaos engineering," *IEEE Software*, vol. 33, no. 3, pp. 35–41, 2016.

[19] J. Humble, D. Farley, *Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation*, Addison-Wesley, 2010.

[20] L. Chen, "Continuous delivery: Huge benefits, but challenges too," *IEEE Software*, vol. 32, no. 2, pp. 50–54, 2015.

[21] J. Garcia, D. Popescu, G. Edwards, N. Medvidovic, "Toward a catalogue of architectural bad smells," in *Proc. 5th Int. Conf. on the Quality of Software Architectures (QoSA)*, LNCS 5581, 2009, pp. 146–162.

[22] J. Garcia, D. Popescu, G. Edwards, N. Medvidovic, "Identifying architectural bad smells," in *Proc. 13th European Conf. on Software Maintenance and Reengineering (CSMR)*, 2009, pp. 255–258.

[23] D. Taibi, V. Lenarduzzi, "On the definition of microservice bad smells," *IEEE Software*, vol. 35, no. 3, pp. 56–62, 2018.

[24] N. Dragoni, S. Giallorenzo, A. L. Lafuente, M. Mazzara, F. Montesi, R. Mustafin, L. Safina, "Microservices: Yesterday, today, and tomorrow," in *Present and Ulterior Software Engineering*, Springer, 2017, pp. 195–216.

[25] R. C. Martin, *Agile Software Development: Principles, Patterns, and Practices*, Prentice Hall, 2003.

[26] W. Cunningham, "The WyCash portfolio management system," in *Addendum to the Proc. Conf. on Object-Oriented Programming Systems, Languages, and Applications (OOPSLA)*, 1992, pp. 29–30.

[27] Z. Li, P. Avgeriou, P. Liang, "A systematic mapping study on technical debt and its management," *Journal of Systems and Software*, vol. 101, pp. 193–220, 2015.

[28] S. R. Chidamber, C. F. Kemerer, "A metrics suite for object oriented design," *IEEE Transactions on Software Engineering*, vol. 20, no. 6, pp. 476–493, 1994.

[29] T. J. McCabe, "A complexity measure," *IEEE Transactions on Software Engineering*, vol. SE-2, no. 4, pp. 308–320, 1976.

[30] N. Fenton, J. Bieman, *Software Metrics: A Rigorous and Practical Approach*, 3rd ed., CRC Press, 2014.

[31] A. Avizienis, J.-C. Laprie, B. Randell, C. Landwehr, "Basic concepts and taxonomy of dependable and secure computing," *IEEE Transactions on Dependable and Secure Computing*, vol. 1, no. 1, pp. 11–33, 2004.

[32] L. Bass, P. Clements, R. Kazman, *Software Architecture in Practice*, 3rd ed., Addison-Wesley, 2012.

[33] R. Kazman, M. Klein, M. Barbacci, T. Longstaff, H. Lipson, J. Carriere, "The architecture tradeoff analysis method," in *Proc. 4th IEEE Int. Conf. on Engineering of Complex Computer Systems (ICECCS)*, 1998, pp. 68–78.

[34] S. Newman, *Building Microservices: Designing Fine-Grained Systems*, O'Reilly Media, 2015.

[35] R. Albert, H. Jeong, A.-L. Barabási, "Error and attack tolerance of complex networks," *Nature*, vol. 406, pp. 378–382, 2000.

[36] A. E. Motter, Y.-C. Lai, "Cascade-based attacks on complex networks," *Physical Review E*, vol. 66, 065102(R), 2002.

[37] U. Brandes, "A faster algorithm for betweenness centrality," *Journal of Mathematical Sociology*, vol. 25, no. 2, pp. 163–177, 2001.

[38] M. E. J. Newman, *Networks: An Introduction*, Oxford University Press, 2010.

[39] T. N. Kipf, M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2017.

[40] W. L. Hamilton, R. Ying, J. Leskovec, "Inductive representation learning on large graphs," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017, pp. 1024–1034.

[41] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio, "Graph attention networks," in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2018.

[42] M. Fey, J. E. Lenssen, "Fast graph representation learning with PyTorch Geometric," in *ICLR Workshop on Representation Learning on Graphs and Manifolds*, 2019.

[43] J. Kreps, N. Narkhede, J. Rao, "Kafka: A distributed messaging system for log processing," in *Proc. 6th Int. Workshop on Networking Meets Databases (NetDB)*, 2011.

[44] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, W. Woodall, "Robot Operating System 2: Design, architecture, and uses in the wild," *Science Robotics*, vol. 7, no. 66, eabm6074, 2022.

[45] S. Kato, S. Tokunaga, Y. Maruyama, S. Maeda, M. Hirabayashi, Y. Kitsukawa, A. Monrroy, T. Ando, Y. Fujii, T. Azumi, "Autoware on board: Enabling autonomous vehicles with embedded systems," in *Proc. ACM/IEEE 9th Int. Conf. on Cyber-Physical Systems (ICCPS)*, 2018, pp. 287–296.

[46] X. Zhou, X. Peng, T. Xie, J. Sun, C. Ji, W. Li, D. Ding, "Fault analysis and debugging of microservice systems: Industrial survey, benchmark system, and empirical study," *IEEE Transactions on Software Engineering*, vol. 47, no. 2, pp. 243–260, 2021.

[47] Google Cloud Platform, "Online Boutique: A cloud-native microservices demo application," software artifact. [Online].

[48] F. Wilcoxon, "Individual comparisons by ranking methods," *Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945.

[49] B. Efron, R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.

[50] C. Spearman, "The proof and measurement of association between two things," *American Journal of Psychology*, vol. 15, no. 1, pp. 72–101, 1904.

[51] U.S. Department of Defense, "MIL-STD-498: Software Development and Documentation," Military Standard, 1994.

[52] D. Chen, Y. Lin, W. Li, P. Li, J. Zhou, X. Sun, "Measuring and relieving the over-smoothing problem for graph neural networks from the topological view," in *Proc. AAAI Conference on Artificial Intelligence*, 2020, pp. 3438–3445.

[53] ISO/IEC 25023:2016, "Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Measurement of system and software product quality," International Organization for Standardization, 2016.

[54] ISO/IEC 25022:2016, "Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Measurement of quality in use," International Organization for Standardization, 2016.

[55] V. R. Basili, L. C. Briand, W. L. Melo, "A validation of object-oriented design metrics as quality indicators," *IEEE Transactions on Software Engineering*, vol. 22, no. 10, pp. 751–761, 1996.

[56] N. Nagappan, T. Ball, "Static analysis tools as early indicators of pre-release defect density," in *Proc. 27th Int. Conf. on Software Engineering (ICSE)*, 2005, pp. 580–586.

[57] T. Zimmermann, R. Premraj, A. Zeller, "Predicting defects for Eclipse," in *Proc. 3rd Int. Workshop on Predictor Models in Software Engineering (PROMISE)*, 2007.

[58] T. Menzies, J. Greenwald, A. Frank, "Data mining static code attributes to learn defect predictors," *IEEE Transactions on Software Engineering*, vol. 33, no. 1, pp. 2–13, 2007.

[59] ISO/IEC 25021:2012, "Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Quality measure elements," International Organization for Standardization, 2012.

[Anon-A] Authors' prior work on multi-layer graph dependency analysis for publish–subscribe systems. *Citation withheld for double-anonymised review.*

---

# Declarations

**CRediT authorship contribution statement.** *[Omitted for double-anonymised review. To be completed on acceptance with the standard CRediT roles: Conceptualization; Methodology; Software; Validation; Formal analysis; Investigation; Data curation; Writing — original draft; Writing — review and editing; Visualization; Supervision.]*

**Declaration of competing interest.** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Funding.** *[Omitted for double-anonymised review.]*

**Data availability.** The replication package—including the seven synthetic scenario datasets, topology generators, simulation harnesses, and real-world architecture adapters—will be made openly available upon publication.

**Declaration of generative AI use.** *[To be completed by the authors in accordance with the journal's policy.]*
