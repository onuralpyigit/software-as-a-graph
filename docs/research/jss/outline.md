# Closed-Loop Prescriptive Architecture Optimization for Distributed Publish-Subscribe Middleware Using Heterogeneous Graph Learning

## 1. Introduction

### 1.1 Context and Motivation
Distributed publish-subscribe middleware frameworks—such as the Robot Operating System (ROS2), Data Distribution Service (DDS), and MQTT—form the communication backbone of modern microservices, IoT systems, and safety-critical cyber-physical platforms. These architectures provide loose temporal, spatial, and synchronization decoupling among producers and consumers. However, this asynchronous decoupling introduces deep, non-linear structural dependencies that obscure how component-level faults propagate. Hardening these networks against cascading service disruptions requires proactive, pre-deployment optimization before configurations are committed to runtime operational fabrics.

### 1.2 Problem Statement
Existing software reliability frameworks struggle with two core limitations when applied to publish-subscribe topologies:
1. **Semantic and Representation Collapse:** Conventional static analysis and homogeneous graph neural networks (GNNs) collapse node and relationship types into uniform, flattened views. By washing away the structural boundaries separating an executable *Application*, a shared software *Library*, a messaging *Topic*, a *Broker* router, and a compute *Node*, these methods suffer from representation collapse and over-smoothing. They fail to capture relation-specific message transformations that dictate cascade blast radiuses at runtime.
2. **The Open-Loop Diagnostic Gap:** State-of-the-art tools operate primarily as passive, open-loop diagnostic engines. While they can flag high-risk anomalies, they do not automate or synthesize concrete, actionable architectural prescriptions to resolve them. Engineers are left with a critical cognitive bottleneck: identifying vulnerabilities without explicit guidance on how to mathematically restructure the topology to alleviate risk.

### 1.3 Proposed Solution
To resolve these challenges, we introduce **SaG-Prescribe (Software-as-a-Graph for Prescriptive Engineering)**, a complete closed-loop optimization system. SaG-Prescribe models multi-tier topologies as native heterogeneous graphs. It processes features using a relation-specific Heterogeneous Graph Transformer (HGT) to infer pre-deployment criticality profiles without using runtime telemetry. Nodes classified with high or critical anomalies via adaptive box-plot outlier-fence thresholds are mapped to automated graph-transformation policies. These policies perform targeted mutations—such as logical topic splitting, infrastructure anti-affinity constraint generation, and transport Quality-of-Service (QoS) contract hardening. The resulting optimized candidates are re-routed through a decoupled discrete-event simulator to verify improvements before deployment.

### 1.4 Key Contributions
The contributions of this study are:
1. **Native Heterogeneous Representation Space:** A multi-tier schema formulation for publish-subscribe environments embedding multidimensional topological and static features across five node types.
2. **Isolation of Typological Heterogeneity Benefits:** A controlled study validating an HGT architecture against homogeneous setups, establishing an F1-score increase of $+0.284$ driven by preserving relation semantics.
3. **Closed-Loop Prescriptive Optimization Engine:** Design and verification of an automated architectural mutation system that lowers cascading fault vulnerabilities, yielding statistically significant gains in the System Resilience Index (SRI).

---

## 2. Background and Related Work

### 2.1 Publish-Subscribe Middleware Dependability
Dependability research for message-oriented middleware historically centers on protocol verification, fault-tolerant replication patterns, network traffic load balancing, and contract verification at runtime. While useful for mitigation post-failure, these approaches treat topologies as fixed inputs, rather than open parameters that can be statically optimized before system delivery.

### 2.2 Structural Criticality Analysis
Graph-theoretic approaches offer mathematical constructs such as betweenness centrality, PageRank, closeness metrics, and articulation boundary tests to pinpoint critical bridges. However, because classical network centrality metrics assume uniform edge semantics, they fail when applied to pub-sub layers, where decoupled endpoints are separated by high-fan-out topics, message brokers, and distinct QoS policies.

### 2.3 Graph Neural Networks in Software Architecture
Homogeneous GNN frameworks are regularly utilized for binary dependency tracking, change propagation analysis, and automated code flaw scanning. When processing complex environments containing hardware hosts, middleware routers, and software modules, homogeneous message passing causes representation over-smoothing. Aggregating distinct relation features into a uniform vector space distorts structural context and degrades predictive performance.

### 2.4 Search-Based and Prescriptive Software Engineering
Search-Based Software Engineering (SBSE) applies heuristic search algorithms to discover ideal architectural refactoring blueprints. However, classic genetic algorithms and deterministic optimizations struggle with large-scale multi-tier networks due to high computational overhead. SaG-Prescribe combines deep heterogeneous feature learning with rule-based prescriptive mutation mechanics to bridge the diagnostic gap, providing scalable, targeted remediation.

---

## 3. Conceptual Framework and Formal System Model

### 3.1 Mathematical Graph Formulation
A distributed publish-subscribe deployment is formally modeled as a directed, multi-relational heterogeneous graph:

$$G = (V, E, \tau_V, \tau_E, \mathbf{x}_v, \mathbf{e}_{uv})$$

Where $V$ represents the set of system vertices, $E \subseteq V \times V$ defines directed links, $\tau_V : V \to T_V$ partitions nodes into semantic classifications, and $\tau_E : E \to T_E$ assigns specific transport and routing relations to edges.

### 3.2 Node and Edge Vocabularies
The multi-tier system schema enforces distinct typological structural vocabularies:

$$T_V = \{\text{Application}, \text{Library}, \text{Topic}, \text{Broker}, \text{Node}\}$$

$$T_E = \{\text{PUBLISHES\_TO}, \text{SUBSCRIBES\_TO}, \text{ROUTES}, \text{RUNS\_ON}, \text{CONNECTS\_TO}, \text{USES}, \text{DEPENDS\_ON}\}$$

Here, `DEPENDS_ON` functions as a derived logical relationship derived in the pre-analysis stage. The direction follows the dependency convention: **dependent $\to$ dependency**.

### 3.3 Multidimensional Feature Tensors
* **Node Feature Tensors ($\mathbf{x}_v$):** Consist of an 18-dimensional base topological embedding vector common to all elements, augmented by category-specific signals. Applications append 5 static code metrics ($cm\_*$ parameters); Topics append 4 publication counts; physical Nodes include normalized hardware capacity features (CPU cores/memory allocation metrics).
* **Edge Feature Tensors ($\mathbf{e}_{uv}$):** Formulate a 16-dimensional matrix comprising a scalar structural weight, path counts, a 7-dimensional relational one-hot edge vector, and 7 explicit QoS contract attributes (Reliability flags, Durability constraints, and Transport Priorities).

### 3.4 Closed-Loop Optimization Strategy
The prescriptive task is defined as computing a transformation policy $\Delta$ such that:

$$G' = \Delta(G)$$

The target objective function aims to minimize the downstream failure vulnerability profile ($I^*(v)$) computed over the global graph topology, constrained by a maximum modification cost budget:

$$\min_{\Delta} \sum_{v \in V} I^*_{\Delta(G)}(v) \quad \text{subject to} \quad \text{Cost}(\Delta) \le \mathcal{B}$$

---

## 4. The SaG-Prescribe Architectural Pipeline

### 4.1 Hexagonal Core Framework Abstraction
The system utilizes a decoupled hexagonal (ports and adapters) design pattern to separate domain orchestration from database infrastructures and communication protocols. Persistence services implement the `IGraphRepository` interface port. Production networks run the concrete Bolt-driven `Neo4jRepository`, while test suites leverage an isolated, thread-safe `MemoryRepository` for rapid verification without a database instance.

### 4.2 Pipeline Architecture Stages

#### Stage 1: Graph Modeling and Dependency Derivation (Model)
The pipeline ingests raw JSON/YAML configuration representations of pub-sub topologies. The ingestion engine maps vertices into the target persistent graph database space and extracts implicit dependency parameters. It parses publisher-subscriber paths to construct logical `DEPENDS_ON` edges across layered projections (`app`, `infra`, `mw`, `system`), laying the groundwork for subsequent analytical passes.

#### Stage 2: Multi-Layered Centrality Diagnostics (Analyze)
Operating over isolated layer subgraphs, the `StructuralAnalyzer` calculates network centrality distributions—including PageRank, Betweenness, Harmonic Closeness, and Eigenvector centralities. These metrics are augmented with domain-specific pub-sub signals, such as the Fan-Out Coefficient (FOC) and Component Dependency Index (CDI). This stage compiles a comprehensive multidimensional structural metrics matrix $\mathbf{M}$.

#### Stage 3: Inductive Criticality Inference (Predict)
The system feeds matrix $\mathbf{M}$ and topological feature matrices into an inductive Heterogeneous Graph Transformer (HGT) built with stacked `EdgeAwareHGTConv` layers. The GNN evaluates relation-specific parameters to infer continuous node and edge criticality profiles $Q^*(v)$. An adaptive `BoxPlotClassifier` processes these distributions to categorize components into discrete risk categories (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `MINIMAL`) using Interquartile Range (IQR) fences.

#### Stage 4: Discrete-Event Dynamic Failure Simulation (Simulate)
Concurrently and independently—maintaining strict functional decoupling to prevent validation leakage—a discrete-event cascade simulator maps dynamic failure tracks over the architecture graph. The engine sequentially injects simulated faults (such as process crashes or message degradation), tracking cascade propagation along transport routes. This evaluation outputs a continuous, empirical ground-truth label array $I^*(v)$ derived via rate-weighted message loss distributions.

#### Stage 5: Non-Parametric Model Alignment Validation (Validate)
The `ValidationService` maps the inferred predictive indicators ($Q^*(v)$) directly against the empirical ground-truth response fields ($I^*(v)$) using non-parametric ranking algorithms (Spearman $\rho$, Kendall $\tau$). It processes continuous values across a structured 9-gate verification tier to evaluate the system's calibration profile. This phase establishes the baseline System Resilience Index (SRI) for the scenario under review.

#### Stage 6: Prescriptive Remediation Generation (Prescribe)
The prescriptive layer translates identified risks into actionable optimization blueprints, processing elements categorized as `CRITICAL` or `HIGH` by the box-plot threshold filter. The engine evaluates structural anomalies using rule-based mappings to formulate an optimization policy $\Delta(G)$ across three distinct architectural vectors:
1. **Logical Subgraph Refactoring:** Suggests partitioning boundaries for high-fan-out components flagged with a `GOD_COMPONENT` or congested topic hub smell, splitting monolithic message lines into discrete, granular sub-topics.
2. **Physical Locality Anti-Affinity Rules:** Targets physical cluster configurations flagged with a Single Point of Failure (`SPOF`), computing container scheduling placement constraints to isolate co-located processes across separate physical host instances.
3. **Middleware Transport Contract Hardening:** Identifies critical communication channels utilizing volatile or loose QoS settings, generating code configurations that upgrade transport properties to reliable and transient-local contracts.

**Closed-Loop Simulation Verification:** The framework applies the compiled transformation policy $\Delta(G)$ to update the Neo4j graph structure. The mutated model $G'$ is programmatically passed back into Stage 4 (Dynamic Simulation) to re-evaluate system resilience under identical fault scenarios, verifying the optimization policy before deployment.

#### Stage 7: SMART Interface Project Layout (Visualize)
The system serializes the baseline metrics, detected architectural flaws, optimized prescriptive modifications, and simulation verification deltas into accessible formats. The Next.js dashboard (**SMART**) projects side-by-side interactive network topologies. This interface visualizes candidate remediation pathways, allowing software architects to review and approve topological modifications before code commitment.

---

## 5. Experimental Setup and Design

### 5.1 Research Questions
To evaluate the diagnostic accuracy and prescriptive efficacy of SaG-Prescribe, we structure our investigation around three focal research questions:
* **RQ1 (Diagnostic Parity):** Does heterogeneous graph learning capture multi-type logical middleware semantics more accurately than non-learning structural measures and uniform homogeneous GNN baselines?
* **RQ2 (Cross-Scenario Generalization):** How does explicit multi-attribute QoS contract feature injection affect in-distribution convergence versus out-of-distribution Leave-One-Scenario-Out (LOSO) generalizability?
* **RQ3 (Prescriptive Optimization Efficiency):** Does the closed-loop prescriptive mutation engine achieve a statistically significant delta reduction in global cascade vulnerability metrics and improve the System Resilience Index (SRI)?

### 5.2 Target Domain Scenario Profiles
The evaluation suite comprises seven distinct parameterized publish-subscribe topologies spanning realistic domain verticals and architectural scale presets:
1. **Scenario 01 (Autonomous Vehicle System):** Medium-scale ROS2 network running streaming sensor topologies with reliable and transient-local QoS profiles.
2. **Scenario 02 (IoT Smart City System):** Large-scale network modeling high-loss volatile and best-effort endpoints.
3. **Scenario 03 (Financial Trading System):** High-density network running time-critical message loops with strict persistent priority settings.
4. **Scenario 04 (Healthcare Integration System):** Dense network characterized by centralized real-time patient monitoring fan-outs and long durability horizons.
5. **Scenario 05 (Hub-and-Spoke Architecture):** Interconnected topology built with a structural anti-pattern bottleneck constraining message distribution paths through a centralized broker pair to evaluate single-point failure tracking.
6. **Scenario 06 (Microservices Mesh):** Sparse, cloud-native cluster deployment with low architectural coupling designed to audit prediction boundary stability.
7. **Scenario 07 (Hyper-Scale Enterprise Architecture):** Hyper-scale system containing 300 distinct execution processes to evaluate the runtime performance boundaries of the framework.

### 5.3 Controlled Factorial Evaluation Space
To isolate performance drivers, we enforce a rigorous $2 \times 3$ factorial experimental framework (Architecture vs. Feature Encoding Matrix) yielding 210 distinct evaluation cells across 5 independent random serialization seeds:
* **HGL-QoS (Proposed Method):** Native multi-type Heterogeneous Graph Attention Network processing full 16-dimensional edge vectors.
* **HGL (Ablated Baseline):** Heterogeneous network layout with all transport QoS edge indices masked to isolate purely topological structural gains.
* **GL-QoS (Homogeneous Comparison):** Uniform graph network model operating over a collapsed logical dependency subgraph where edge values are reduced to single scalar QoS weights.
* **GL (Flattened Comparison):** Standard homogeneous graph setup running over unweighted, flattened connectivity matrices.
* **Topo-QoS (Non-Learning Centrality):** Structural analysis using QoS-derived, weighted betweenness algorithms.
* **Topo-BL (Non-Learning Centrality):** Standard structural baseline computing unweighted centralities and articulation points.

---

## 6. Experimental Evaluation and Results Analysis

### 6.1 Diagnostic Accuracy and Component Ranking (RQ1)
The experimental results demonstrate that preserving native heterogeneous typologies is critical for accurate risk assessment. HGL and HGL-QoS consistently achieve superior predictive accuracy. On the critical component identification task (selecting the top-$K$ most impactful components), the heterogeneous GAT model achieves a mean F1-score of **0.765**, outperforming the homogeneous GL baseline (**0.481**) by a substantial margin of $\Delta\text{F1} = +0.284$. Homogeneous configurations underperform across several key benchmarks—for instance, dropping to an F1 of **0.000** in the Hub-and-Spoke layout and **0.100** in sparse Microservices—proving their vulnerability to representation collapse in the presence of hub-dominated transport structures.

### 6.2 Ablation and Generalization Mechanics (RQ2)
The factorial setup reveals a subtle trade-off regarding QoS attribute encoding. Within the same domain configuration, the explicit inclusion of multi-attribute QoS dimensions provides marginal returns and can increase optimization overhead, slightly reducing average in-distribution accuracy (`HGL-QoS` vs. `HGL`: $\Delta\rho = -0.044$). This occurs because the typed topological routing structure already implicitly encodes most QoS dynamics, making explicit attributes redundant when training and testing environments align.

However, in Leave-One-Scenario-Out (LOSO) cross-validation on entirely unseen topologies, this behavior reverses. The QoS-aware variant (`HGL-QoS`) completely dominates out-of-distribution evaluations, achieving a mean cross-scenario correlation of $\rho = \mathbf{0.4009}$ compared to purely structural models (`HGL`: $\rho = 0.3073$; `GL`: $\rho = 0.0208$). Thus, while explicit QoS feature injection introduces optimization noise in-distribution, it acts as a critical anchor that prevents topological overfitting, enabling successful generalization to entirely unobserved systems.

### 6.3 Prescriptive Efficacy and Resilience Optimization (RQ3)
We evaluate the prescriptive efficacy of SaG-Prescribe by measuring the delta change in the global System Resilience Index (SRI) before and after executing the computed topological modifications. The closed-loop verification results show that applying automated refactoring transformations consistently yields a statistically significant reduction in cascading failure propagation. For example, in the Hub-and-Spoke scenario, the topic-splitting and anti-affinity placement policies eliminate single points of failure, increasing the scenario's baseline resilience score. Across all seven target domain presets, the programmatic loop successfully mitigates structural hazards, demonstrating that linking predictive risk maps to automated graph mutations can systematically harden distributed middleware prior to software delivery.

---

## 7. Discussion and Threats to Validity

### 7.1 Construct and Internal Validity
* **Simulator-Derived Reference Targets:** Because our ground-truth labels $I^*(v)$ are generated via an independent discrete-event cascade simulator rather than production deployment telemetry, high model alignment shows that our system has successfully learned the simulator's cascade rules. To address this threat, all variants are evaluated against identical targets, ensuring internal consistency across comparative benchmarks.
* **Per-Type Informative Parity (Simpson's Paradox):** Shared software libraries are treated as passive elements during cascade propagation, meaning their label distributions exhibit near-zero variance. Pooling these degenerate distributions with highly active application node metrics creates a statistical anomaly reminiscent of Simpson's Paradox, where type-specific signals are masked in the global average. We mitigate this threat by using stratified, type-level performance reporting throughout our evaluation.

### 7.2 External Validity
Our experiments are conducted across seven parameterized, synthetically generated scenarios. While these vertical presets mimic representative real-world middleware distributions, they may not fully capture the runtime complexities of industrial Kubernetes clusters or live enterprise systems—such as dynamic workload shifts, network packet loss spikes, or transient hardware faults.

### 7.3 System Engineering Trade-offs
The relation-aware transformations computed by the HGT and the multi-iteration closed-loop simulation validation steps introduce additional computational overhead compared to flat, non-learning structural centralities. However, since this framework is explicitly designed for pre-deployment optimization rather than real-world runtime alerting, this initial computational cost represents an acceptable trade-off for systematic reliability improvements.

---

## 8. Conclusion and Future Directions

### 8.1 Key Conclusions Summary
This study presents SaG-Prescribe, an automated, closed-loop prescriptive optimization pipeline designed to secure distributed publish-subscribe architectures against cascading service failures. By running relation-aware message passing over a native heterogeneous graph model, the framework avoids the representation collapse typical of homogeneous GNN alternatives. More importantly, the system bridges the diagnostic gap by mapping topological vulnerabilities to automated structural refactoring transformations, placement constraints, and contract hardening policies. Empirical validation confirms significant, verifiable gains in the global System Resilience Index (SRI), proving that closed-loop architectural mutation can successfully secure decoupled networks before production delivery.

### 8.2 Future Pathways
In future research, we aim to extend SaG-Prescribe along three main lines:
1. **Live Operational Telemetry Hook:** Integrating real-world performance metrics—such as message throughput latency, container CPU throttles, and network packet drop counts—directly into the Neo4j feature space to enable continuous optimization.
2. **Search Bounds Expansion:** Incorporating explicit cost constraints, financial budgets, and compute resource limits into the prescriptive algorithm to balance reliability goals against resource footprints.
3. **Cross-Model Neural Benchmarking:** Conducting comparative validation studies against alternative heterogeneous graph network architectures—such as HAN, RGCN, and MAGNN—to further map the heterogeneous GNN design space for software reliability engineering.