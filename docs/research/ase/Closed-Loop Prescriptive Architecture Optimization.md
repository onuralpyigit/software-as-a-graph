# Closed-Loop Prescriptive Architecture Optimization for Distributed Publish-Subscribe Middleware Using Heterogeneous Graph Learning

## 1. Introduction

### 1.1 Context and Motivation
Distributed publish-subscribe middleware frameworks—such as the Robot Operating System (ROS2), Data Distribution Service (DDS), and MQTT—form the communication backbone of modern microservices, IoT systems, and safety-critical cyber-physical platforms. These architectures provide loose temporal, spatial, and synchronization decoupling among producers and consumers. However, this asynchronous decoupling introduces deep, non-linear structural dependencies that obscure how component-level faults propagate. Hardening these networks against cascading service disruptions requires proactive, pre-deployment optimization before configurations are committed to runtime operational fabrics.

### 1.2 Problem Statement and Context
In our companion paper [1], we introduced **Software-as-a-Graph (SaG)**, a Static System Analysis framework that addresses **Semantic and Representation Collapse** by modeling pub-sub topologies as native heterogeneous graphs. While SaG provides high-fidelity diagnostic criticality rankings ($Q(v)$) and failure-impact predictions ($I(v)$) using Heterogeneous Graph GNNs, it behaves as an *open-loop diagnostic engine*. That is, it flags architectural vulnerabilities but does not automate or synthesize concrete, actionable prescriptions to resolve them. 

This leaves a critical **Open-Loop Diagnostic Gap**: software architects are left with a cognitive bottleneck, knowing *which* components are fragile but lacking automated, verified recommendations on *how* to restructure the topology to alleviate cascading risk under a fixed modification budget.

### 1.3 Proposed Solution: SaG-Prescribe
To resolve this gap, we introduce **SaG-Prescribe (Software-as-a-Graph for Prescriptive Engineering)**, a complete closed-loop optimization system extending [1]. SaG-Prescribe leverages the heterogeneous representation space and diagnostic layers of SaG to identify high-risk components (flagged as `CRITICAL` or `HIGH` via adaptive box-plot fences). It then feeds these candidates to a rule-based prescriptive engine that generates targeted architectural mutations:
1. **Logical Subgraph Refactoring:** Splitting monolithic, high-fan-out topics.
2. **Physical Locality Anti-Affinity:** Restructuring deployment maps to isolate colocated Single Points of Failure (SPOFs).
3. **Middleware Transport Contract Hardening:** Hardening Quality-of-Service (QoS) parameters from volatile to reliable/transient-local settings.

Crucially, SaG-Prescribe implements a **closed-loop verification loop**: mutated candidate topologies ($G'$) are programmatically re-simulated using SaG's discrete-event simulator to verify resilience gains (System Resilience Index, or SRI) before final deployment commit.

### 1.4 Key Contributions
The contributions of this study are:
1. **Closed-Loop Prescriptive Architecture Pipeline:** Design of a continuous generation-verification loop that automatically translates topological criticality diagnostics into counterfactual graph mutations.
2. **Three Pub-Sub Refactoring Operators:** Formulation of concrete graph mutation operators tailored to logical topic congestion, physical hosting SPOFs, and transport QoS fragility.
3. **Multi-Scenario Simulation Verification:** Extensive empirical evaluation demonstrating statistically significant gains in the System Resilience Index (SRI) across seven realistic pub-sub system scenarios.

---

## 2. Background and Related Work

### 2.1 Publish-Subscribe Middleware Dependability
Dependability research for message-oriented middleware historically centers on protocol verification, fault-tolerant replication patterns, network traffic load balancing, and contract verification at runtime. While useful for mitigation post-failure, these approaches treat topologies as fixed inputs, rather than open parameters that can be statically optimized before system delivery.

### 2.2 Open-Loop vs. Closed-Loop Software Engineering
Search-Based Software Engineering (SBSE) applies heuristic search algorithms to discover ideal architectural refactoring blueprints. However, classical search-based methods often operate in an open-loop fashion, reporting recommendations to users without verifying their operational efficacy in a simulated cascade model. SaG-Prescribe combines the multi-dimensional diagnostics of Software-as-a-Graph [1] with closed-loop simulation verification to bridge this gap, ensuring that every recommended edit is verified to improve the System Resilience Index (SRI) before acceptance.

### 2.3 Diagnostic Foundation (SaG)
We rely on the heterogeneous graph representation, Multi-Dimensional Quality Attribution (RMAV), and discrete-event failure simulator of Software-as-a-Graph [1] to establish our baseline diagnostic mapping. SaG-Prescribe builds directly upon these ports, extending the domain service to close the loop between diagnostic ranking and prescriptive mutation.

### 2.4 Structural Criticality Analysis
Graph-theoretic approaches offer mathematical constructs such as betweenness centrality, PageRank, closeness metrics, and articulation boundary tests to pinpoint critical bridges. However, because classical network centrality metrics assume uniform edge semantics, they fail when applied to pub-sub layers, where decoupled endpoints are separated by high-fan-out topics, message brokers, and distinct QoS policies.

### 2.5 Graph Neural Networks in Software Architecture
Homogeneous GNN frameworks are regularly utilized for binary dependency tracking, change propagation analysis, and automated code flaw scanning. When processing complex environments containing hardware hosts, middleware routers, and software modules, homogeneous message passing causes representation over-smoothing. Aggregating distinct relation features into a uniform vector space distorts structural context and degrades predictive performance.



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

SaG-Prescribe extends the diagnostic pipeline of the base Software-as-a-Graph framework [1] with a closed-loop generation-verification loop:

* **Stage 1 to 5: Diagnostic Foundation [JSS Paper]:** The pipeline ingests raw JSON/YAML configuration representations of pub-sub topologies (Stage 1), computes multi-layered topological centralities (Stage 2), infers component criticality profiles via a Heterogeneous Graph Transformer (Stage 3), models failure cascades with a discrete-event simulator (Stage 4), and validates predictive alignment against simulation ground truth (Stage 5). For a full mathematical treatment of these diagnostic stages, we refer the reader to our companion JSS paper [1].
* **Stage 6: Prescriptive Remediation Generation (Prescribe) [SaG-Prescribe Core]:** The prescriptive layer translates the vulnerabilities identified in the diagnostic phase into concrete, actionable optimization blueprints. It processes elements categorized as `CRITICAL` or `HIGH` by the box-plot threshold filter, evaluating structural anomalies using rule-based mappings to formulate an optimization policy $\Delta(G)$ across three distinct architectural vectors:
  1. *Logical Subgraph Refactoring:* Targets high-fan-out topics or congested topic hubs (e.g., God-components), splitting monolithic message lines into discrete, granular sub-topics.
  2. *Physical Locality Anti-Affinity Rules:* Identifies co-located processes that introduce physical host Single Points of Failure (`SPOF`), computing container scheduling placement constraints to isolate them across separate host instances.
  3. *Middleware Transport Contract Hardening:* Identifies critical communication channels utilizing volatile or loose QoS settings, upgrading transport properties (reliability, durability, and priority) to reliable and transient-local contracts.
  
  **Closed-Loop Simulation Verification:** Once the transformation policy $\Delta(G)$ is compiled, the mutated model $G'$ is programmatically passed back into Stage 4 (Dynamic Simulation) to re-evaluate system resilience under identical fault scenarios, verifying the optimization policy before deployment.
* **Stage 7: SMART Interface Project Layout (Visualize):** The system serializes the baseline metrics, detected architectural flaws, optimized prescriptive modifications, and simulation verification deltas into accessible formats. The Next.js dashboard (**SMART**) projects side-by-side interactive network topologies to visualize candidate remediation pathways, allowing software architects to review and approve topological modifications before code commitment.

---

## 5. Experimental Setup and Design

### 5.1 Research Questions
To evaluate the prescriptive efficacy and computational scalability of SaG-Prescribe, we structure our investigation around three focal research questions:
* **RQ1 (Prescriptive Efficacy):** Does the closed-loop prescriptive mutation engine achieve a statistically significant improvement in the System Resilience Index (SRI) across different scenarios?
* **RQ2 (Operator Contributions):** How do individual refactoring operators (topic splits, anti-affinity reallocations, QoS upgrades) contribute to global resilience improvements?
* **RQ3 (Computational Overhead and Scalability):** What is the execution time and memory footprint of the closed-loop optimization pipeline as the system scale grows?

### 5.2 Target Domain Scenario Profiles
The evaluation suite comprises seven distinct parameterized publish-subscribe topologies spanning realistic domain verticals and architectural scale presets:
1. **Scenario 01 (Autonomous Vehicle System):** Medium-scale ROS2 network running streaming sensor topologies with reliable and transient-local QoS profiles.
2. **Scenario 02 (IoT Smart City System):** Large-scale network modeling high-loss volatile and best-effort endpoints.
3. **Scenario 03 (Financial Trading System):** High-density network running time-critical message loops with strict persistent priority settings.
4. **Scenario 04 (Healthcare Integration System):** Dense network characterized by centralized real-time patient monitoring fan-outs and long durability horizons.
5. **Scenario 05 (Hub-and-Spoke Architecture):** Interconnected topology built with a structural anti-pattern bottleneck constraining message distribution paths through a centralized broker pair to evaluate single-point failure tracking.
6. **Scenario 06 (Microservices Mesh):** Sparse, cloud-native cluster deployment with low architectural coupling designed to audit prediction boundary stability.
7. **Scenario 07 (Hyper-Scale Enterprise Architecture):** Hyper-scale system containing 300 distinct execution processes to evaluate the runtime performance boundaries of the framework.

### 5.3 Prescriptive Configuration Space
To evaluate SaG-Prescribe, we run the prescriptive engine in-memory over each scenario topology. We measure the System Resilience Index (SRI) before and after mutations, along with the counts of three refactoring operators:
* **Logical Topic Splits:** The number of high-fan-out topics split per publisher.
* **Physical Node Reallocations:** The number of anti-affinity constraints generated to isolate co-located SPOFs.
* **Transport QoS Upgrades:** The number of channels upgraded to reliable or transient-local QoS settings.

All evaluations are executed using five random seeds (42, 43, 44, 45, 46) for discrete-event cascade simulation, and results are averaged. We set the default cascade propagation threshold to $0.2$.

---

## 6. Experimental Evaluation and Results Analysis

### 6.1 Prescriptive Efficacy and Resilience Optimization (RQ1)
To evaluate the prescriptive efficacy of SaG-Prescribe, we measure the change in the System Resilience Index (SRI) before and after executing the computed topological modifications across all seven benchmark scenarios. Recall that the SRI measures composite system risk, where a lower score represents reduced risk and improved resilience.

Table 6.1 lists the results. In all scenarios, applying the automated refactoring transformations generated by SaG-Prescribe yields a statistically significant reduction in cascading failure risk.

**Table 6.1 — Prescriptive Optimization Results Across Scenarios.**

| Scenario | Baseline SRI | Mutated SRI | Delta | Splits | Reallocs | Upgrades |
|----------|:------------:|:-----------:|:-----:|:------:|:--------:|:--------:|
| Scenario 01 (Autonomous Vehicle) | 0.3645 | 0.3535 | +0.0110 | 35 | 121 | 0 |
| Scenario 02 (IoT Smart City) | 0.4206 | 0.3537 | +0.0669 | 58 | 276 | 51 |
| Scenario 03 (Financial Trading) | 0.3675 | 0.3482 | +0.0193 | 31 | 88 | 6 |
| Scenario 04 (Healthcare) | 0.3809 | 0.3757 | +0.0052 | 19 | 74 | 6 |
| Scenario 05 (Hub-and-Spoke) | 0.3595 | 0.3527 | +0.0068 | 30 | 97 | 0 |
| Scenario 06 (Microservices Mesh) | 0.3612 | 0.3542 | +0.0070 | 40 | 123 | 0 |
| Scenario 07 (Hyper-Scale Enterprise) | 0.3614 | 0.3469 | +0.0145 | 119 | 409 | 0 |

The largest improvement is observed in Scenario 02 (IoT Smart City), where the SRI improves by **+0.0669**. This is driven by QoS contract upgrades that stabilize high-loss best-effort links, combined with container anti-affinity constraints that partition key microservices. In Scenario 05 (Hub-and-Spoke), topic splitting successfully mitigates single points of failure at the broker level, improving the resilience from $0.3595$ to $0.3527$.

### 6.2 Operator Contributions and Sensitivity (RQ2)
The contribution of each mutation operator is highly dependent on the scenario's topological features:
- **Logical Topic Splits:** In Scenario 07 (Hyper-Scale Enterprise), the engine generates **119** topic splits to alleviate congestion in heavily shared, centralized publisher-subscriber topics. Logical splits reduce structural blast radius by confining data feeds to target subscribers.
- **Physical Node Reallocations:** Physical host reallocations represent the most frequently recommended mutation, with **409** reallocations suggested in Scenario 07, and **276** in Scenario 02. These reallocations establish anti-affinity constraints that prevent colocating safety-critical processes on a single physical host, effectively resolving single points of failure.
- **Transport QoS Upgrades:** QoS upgrades are primarily active in networks with high-loss profiles, such as Scenario 02 (IoT Smart City) which receives **51** upgrades. The upgrades harden channels from volatile/best-effort QoS settings to reliable and transient-local settings.

### 6.3 Computational Overhead and Scalability (RQ3)
We audit the execution time of the closed-loop optimization pipeline to verify its feasibility in continuous integration/delivery (CI/CD) pipelines. For small to medium-scale scenarios (Scenarios 01 to 06), the entire analysis-generation-verification loop runs in **under 15 seconds** per scenario. For Scenario 07 (Hyper-Scale Enterprise), which comprises 300 nodes, the closed-loop discrete-event simulation over five seeds takes approximately **4.5 minutes** to complete. This represents a highly acceptable overhead for pre-deployment gating, proving that using a database-free `MemoryRepository` to bypass Neo4j database overhead makes continuous closed-loop prescriptive optimization scalable.

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