# Automated Prescriptive Refactoring of Distributed Middleware Architectures via DevOps-Integrated Graph Analytics

**Target Venue:** *Automated Software Engineering (AuSE) — Special Issue on Intelligent Techniques for Automated Code Review and Software Quality Evaluation*

---

## Abstract

Modern distributed middleware architectures, such as publish–subscribe networks utilizing Data Distribution Service (DDS), ROS 2, or MQTT, accumulate severe architectural technical debt during continuous software evolution. While traditional code-level static analysis platforms effectively flag localized maintainability defects, they are entirely blind to how localized code fragility combines with global multi-relational topologies to induce systemic risk. Furthermore, existing structural evaluation frameworks operate in an open-loop fashion, highlighting architectural vulnerabilities without synthesizing verified, actionable guidance on how to restructure the topology.

To bridge this architecture-code gap, this paper presents a closed-loop prescriptive refactoring system extending the **Software-as-a-Graph (SaG)** ecosystem. The framework ingests code-level metrics to compute a specialized **Code Quality Penalty (CQP)** and propagates these penalties across a multi-layered heterogeneous dependency graph. Vulnerable components are compiled into a graph mutation policy $\Delta(G)$ leveraging three structural refactoring operators: logical topic splitting, physical anti-affinity reallocation, and transport Quality-of-Service (QoS) contract hardening. Each mutated candidate topology is re-evaluated via an in-memory discrete-event cascade simulator to verify improvements in a System Resilience Index (SRI) before deployment.

To support continuous software delivery, the framework is operationalized as a delta-aware CI/CD quality gate (`detect_antipatterns.py`) that blocks newly introduced structural regressions relative to a merge base within sub-minute execution bounds. Across seven parameterized software topologies spanning autonomous vehicles, smart cities, and hyper-scale enterprises, our closed-loop architecture achieves significant simulation-verified risk reductions, making automated prescriptive architecture optimization practical for rapid-release pipelines.

---

## 1. Introduction

### 1.1 Context and Motivation

Distributed publish–subscribe middleware frameworks—such as the Robot Operating System (ROS 2), the Data Distribution Service (DDS), and message-broker networks—form the communication backbone of modern safety-critical cyber-physical systems, microservices meshes, and Internet-of-Things (IoT) ecosystems. These paradigms achieve spatial, temporal, and synchronization decoupling by abstracting communication channels into shared topics and message-routing intermediaries.

However, this decoupling introduces deep, non-linear structural dependencies that obscure how component-level failures propagate through the wider system. Hardening these networks against cascading failures requires proactive, continuous, pre-deployment optimization before configurations are committed to runtime operational fabrics.

### 1.2 The Architecture-Code Gap & The Open-Loop Gap

Historically, automated quality assurance has operated primarily at the source-code level through Static Code Analysis (SCA) platforms like SonarQube. This induces an **"Architecture-Code Gap"**: a system can have perfectly clean source code within individual modules yet remain highly fragile due to topology-level single points of failure (SPOFs), colocated deployment bottlenecks, or mismatched communication attributes. Shifting structural verification "left" into the continuous integration and delivery (CI/CD) pipeline requires a paradigm shift from traditional SCA to Static System Analysis (SSA).

Even when frameworks can statically diagnose vulnerabilities, they suffer from an **open-loop diagnostic gap**. Existing diagnostic platforms rank components by criticality but operate without counterfactual verification; they inform architects *which* components are fragile without synthesizing automated, verified prescriptions explaining *how* to refactor the topology to minimize cascading risk.

### 1.3 Proposed Solution: SaG-Prescribe

To bridge these parallel gaps, we introduce **SaG-Prescribe**, a closed-loop prescriptive optimization system extending the Software-as-a-Graph (SaG) framework. SaG-Prescribe maps the structural and code-level characteristics of distributed networks onto typed heterogeneous graphs. High-risk components are isolated via adaptive box-plot fences and compiled into a graph mutation policy composed of three refactoring operators: logical topic splitting, physical anti-affinity reallocation, and transport QoS contract hardening.

Crucially, SaG-Prescribe executes a programmatic *generate–verify loop*: each candidate topology is re-simulated in-memory with an event-driven cascade simulator under identical fault conditions, ensuring that optimizations are explicitly verified rather than merely suggested. Finally, the framework acts as a continuous CI/CD quality gate, evaluating structural regressions relative to a Git merge base to ensure zero architectural degradation during rapid software evolution.

---

## 2. Background and Related Work

### 2.1 Publish–Subscribe Middleware Dependability

Dependability research for message-oriented middleware traditionally concentrates on protocol verification, fault-tolerant broker replication patterns, network load balancing, and runtime contract validation. While runtime frameworks like Chaos Engineering inject faults into active clusters to evaluate empirical resilience, these approaches occur late in the software lifecycle and introduce operational risk. Our approach operates earlier, executing static system analysis on "Architecture-as-Code" configuration descriptors to proactively evaluate and optimize topologies before deployment.

### 2.2 Search-Based Software Engineering and Architecture Optimization

Search-Based Software Engineering (SBSE) applies heuristic search techniques to discover refactoring blueprints across software patterns. However, classical architectural optimization methods often operate open-loop, presenting structural changes without quantifying their concrete operational efficacy against cascading failures. SaG-Prescribe resolves this limitation by evaluating every recommended refactoring operator against a simulated ground-truth failure model.

### 2.3 Structural Technical Debt Analysis

Network-science metrics (e.g., betweenness centrality, PageRank, and articulation point tests) are frequently applied to software call graphs to isolate technical debt and design anti-patterns. However, classical centralities degrade on publish–subscribe networks because they assume uniform edge semantics. They fail to represent typed middleware boundaries, QoS configurations, or the simultaneous blast radius induced by a shared code library. The Software-as-a-Graph framework addresses this by sustaining typed node and edge vocabularies across distinct architectural perspectives.

---

## 3. Multi-Dimensional Technical Debt Analysis

### 3.1 Heterogeneous Graph Formulation

We model a distributed publish–subscribe system as a typed, weighted, directed multigraph:

$$G = (V, E, \tau_V, \tau_E, w_E, w_V)$$

The vertex set partitions into five distinct semantic node types:

$$T_V = \{\text{Application}, \text{Library}, \text{Topic}, \text{Broker}, \text{Node}\}$$

* **Application ($V_{\text{app}}$):** Active execution processes that produce or consume data.
* **Library ($V_{\text{lib}}$):** Shared code modules utilized across applications.
* **Topic ($V_{\text{topic}}$):** Named communication channels mediating message exchanges.
* **Broker ($V_{\text{broker}}$):** Middleware intermediaries routing message paths.
* **Node ($V_{\text{node}}$):** Physical or virtual hosting environments.

Structural links partition into a distinct type vocabulary mapping physical and communication relations:

$$T_E = \{\text{PUBLISHES\_TO}, \text{SUBSCRIBES\_TO}, \text{ROUTES}, \text{RUNS\_ON}, \text{CONNECTS\_TO}, \text{USES}\}$$

### 3.2 Derived Dependencies: The `DEPENDS_ON` Projection

To uncover logical dependency paths hidden behind decoupled pub-sub structures, the framework derives explicit `DEPENDS_ON` relations (directed from dependent to dependency) via typed projection rules:

* **Application-to-Application:** Formed when a subscriber depends on a publisher via a shared topic channel.
* **Application-to-Broker:** Maps reliance on a specific broker instance routing an application's topics.
* **Application-to-Library:** Models the simultaneous blast radius where a shared library failure instantly impacts all consuming applications.
* **Broker-to-Broker:** Captures colocation vulnerabilities where multiple brokers share the same physical node host.

### 3.3 The Code Quality Penalty (CQP)

To bridge local code quality with system architecture, the framework ingests modular metrics from static code analysis (SCA) APIs during model import. These features encompass total lines of code (`cm_total_loc`), Weighted Methods per Class (`cm_avg_wmc`), Lack of Cohesion of Methods (`cm_avg_lcom`), and the technical debt ratio (`sqale_debt_ratio`). Rank-normalized properties map directly into the **Code Quality Penalty (CQP)** for Applications and Libraries:

$$\mathrm{CQP}(v) = 0.10\,\text{loc\_norm} + 0.35\,\text{complexity\_norm} + 0.30\,\text{instability\_code} + 0.25\,\text{lcom\_norm}$$

### 3.4 Multi-Dimensional Quality Attribution (RMAV)

Component criticality is decomposed into four orthogonal dimensions, ensuring that raw structural and code metrics feed exactly one perspective to guarantee explanation legibility:

* **Reliability ($R$):** Fault-propagation risk calculated via Reverse PageRank (RPR) and fan-out concentration.
* **Maintainability ($M$):** Coupling complexity driven by betweenness centrality ($BT$), efferent QoS out-degree ($w\_out$), and the CQP metric:

$$M(v) = 0.35\,\mathrm{BT}(v) + 0.30\,\mathrm{w\_out}(v) + 0.15\,\mathrm{CQP}(v) + 0.12\,\mathrm{CouplingRisk\_enh}(v) + 0.08\,(1-\mathrm{CC}(v))$$


* **Availability ($A$):** Single-point-of-failure risk evaluating directed cut vertices and QoS-amplified SPOF scores.
* **Vulnerability ($V$):** Exposure to adversarial reach mapping attack propagation vectors.

These profiles blend into a composite criticality score $Q(v)$ utilizing Analytic Hierarchy Process (AHP) weights mixed with a uniform prior ($\lambda=0.70$) to prevent extreme parameter concentration, yielding final weights of $(0.38, 0.24, 0.19, 0.19)$ for availability, reliability, maintainability, and vulnerability respectively.

---

## 4. The SaG-Prescribe Refactoring Engine

### 4.1 Optimization Objective

The prescriptive optimization task aims to synthesize a transformation policy $\Delta$ producing an optimized topology $G' = \Delta(G)$ that systematically minimizes the aggregate failure-impact profile across system vertices:

$$\min_{\Delta} \sum_{v \in V} I^*_{\Delta(G)}(v)$$

The optimization objective is evaluated through the System Resilience Index (SRI) derived via counterfactual discrete-event failure simulations.

### 4.2 Graph Mutation Operators

Components categorized as `CRITICAL` or `HIGH` by adaptive box-plot fences trigger the rule-based execution of three distinct graph mutation operators:

* **Operator 1 — Logical Topic Splitting:** Targets overloaded topics with multiple publishers ($|P(t)| > 1$). It replaces the centralized topic with isolated per-publisher sub-topics, rewiring subscriber links to narrow the structural cascade blast radius.
* **Operator 2 — Physical Anti-Affinity Reallocation:** Addresses co-location single points of failure where a single physical host node runs multiple critical applications or brokers. It synthesizes container scheduler anti-affinity rules, rewriting `RUNS_ON` edges to map components to isolated physical infrastructure.
* **Operator 3 — Transport QoS Contract Hardening:** Targets fragile or volatile transport configurations. It upgrades volatile/best-effort connections to reliable and transient-local settings, increasing baseline system tolerance against packet disruptions during failure waves.

### 4.3 Closed-Loop Verification Pipeline

The verification architecture guarantees complete independence between candidate generation and the validation path by separating graph structures from simulated metrics:

1. **Export:** Serializes the current source topology description into a flat JSON schema.
2. **Mutate:** Applies the compiled mutation policy $\Delta(G)$ unconditionally to the exported structures.
3. **Sandbox Isolation:** Ingests the mutated schema into a separate, thread-safe `MemoryRepository` instance, re-projecting logical dependencies.
4. **Simulation Oracle:** Executes the canonical event-driven failure simulator on the mutated sandbox model under identical seeds and thresholds.
5. **Resilience Delta Quantification:** Computes the change in the System Resilience Index ($\Delta\text{SRI} = \text{SRI}_{\text{baseline}} - \text{SRI}_{\text{mutated}}$) to report verified structural alignment.

---

## 5. DevOps Integration & Delta-Aware Gating

### 5.1 Automated Code Review Architecture

To continuously govern structural quality during rapid code evolution, the framework is operationalized as a blocking check script (`detect_antipatterns.py`) in continuous integration and delivery (CI/CD) pipelines. Whenever an engineer alters system structures or configures new messaging topology, the gate parses the "Architecture-as-Code" descriptors and populates an in-memory graph view.

### 5.2 Delta-Aware Regression Semantics

Absolute quality gates that fail builds on any critical structural anti-pattern are unsustainable in industrial software development, as real architectures frequently contain intentional, risk-accepted risks (e.g., legacy unreplicated components). To resolve this, our gate uses **delta-aware semantics**: it compares the pull request candidate topology against the target branch merge-base topology.

It isolates and flags only *newly introduced* structural regressions. Pre-existing risks are passed unless their thresholds change, and intentional anomalies can be bypassed via an auditable, time-bound **waiver register**.

### 5.3 Exit-Code Protocol

The quality gate enforces code review automation by terminating execution with standardized exit codes that command CI/CD pipeline workers:

* **Exit Code 0:** No new structural anomalies or software design defects detected; build passes, and deployment is permitted.
* **Exit Code 1:** New minor architectural smells or QoS warnings introduced; build passes with warnings compiled into the developer's code review assistant dashboard.
* **Exit Code 2:** New, unwaived `CRITICAL` or `HIGH` severity anomalies (e.g., newly introduced un-replicated SPOFs or routing loops) discovered; **the build breaks, and deployment is blocked**.

---

## 6. Experimental Evaluation

### 6.1 Evaluation Suite Scale

The framework was evaluated across a standardized benchmark suite comprising seven synthetic scenarios modeling diverse distributed middleware verticals and scale categories:

* **S01 (Autonomous Vehicle):** Medium-scale ROS 2 network with streaming sensor fields.
* **S02 (IoT Smart City):** Large-scale, high-loss configuration with best-effort endpoints.
* **S03 (Financial Trading):** Time-critical loops with rigid persistent priority setups.
* **S04 (Healthcare):** Dense clinical network with centralized patient monitors.
* **S05 (Hub-and-Spoke):** High-bottleneck model featuring a central broker pair.
* **S06 (Microservices Mesh):** Sparse, low-coupling, cloud-native configuration.
* **S07 (Hyper-Scale Enterprise):** Extensively scaled architecture featuring 300 processes.

### Table 6.1 — Scenario Scale and Topology Summary

| Scenario | Applications | Libraries | Topics | Brokers | Nodes | Structural Edges ($|E|$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S01 Autonomous Vehicle** | 80 | 20 | 40 | 4 | 8 | 797 |
| **S02 IoT Smart City** | 200 | 10 | 80 | 6 | 30 | 1322 |
| **S03 Financial Trading** | 60 | 18 | 35 | 5 | 6 | 580 |
| **S04 Healthcare** | 50 | 12 | 25 | 3 | 8 | 400 |
| **S05 Hub-and-Spoke** | 70 | 25 | 30 | 2 | 12 | 797 |
| **S06 Microservices Mesh** | 90 | 30 | 45 | 6 | 15 | 680 |
| **S07 Hyper-Scale Enterprise** | 300 | 50 | 120 | 10 | 40 | 3245 |

### 6.2 Prescriptive Optimization Results

All experimental runs were compiled against the canonical deterministic failure simulator over seeds 42–46, producing identical values ($\sigma_{\text{seed}}=0$), which isolates paired optimization deltas from simulation stochasticity.

### Table 6.2 — Architectural Refactoring Optimization Quantities

| Scenario | Baseline SRI | Mutated SRI | SRI Delta ($\Delta$) | Relative Reduction | Splits | Reallocs | Upgrades |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **S01 Autonomous Vehicle** | 0.3645 | 0.3535 | 0.0110 | 3.0% | 35 | 121 | 0 |
| **S02 IoT Smart City** | 0.4206 | 0.3537 | 0.0669 | 15.9% | 58 | 276 | 51 |
| **S03 Financial Trading** | 0.3675 | 0.3482 | 0.0193 | 5.3% | 31 | 88 | 6 |
| **S04 Healthcare** | 0.3809 | 0.3757 | 0.0052 | 1.4% | 19 | 74 | 6 |
| **S05 Hub-and-Spoke** | 0.3595 | 0.3527 | 0.0068 | 1.9% | 30 | 97 | 0 |
| **S06 Microservices Mesh** | 0.3612 | 0.3542 | 0.0070 | 1.9% | 40 | 123 | 0 |
| **S07 Hyper-Scale Enterprise** | 0.3614 | 0.3469 | 0.0145 | 4.0% | 119 | 409 | 0 |

A Wilcoxon signed-rank test over the baseline-vs-mutated pairs confirms statistical significance ($W=0.0, p=0.0156$ at $\alpha=0.05$), indicating a consistent systemic risk reduction across all topological scales.

The single largest improvement is observed in **S02 (IoT Smart City)** with a 15.9% reduction, heavily catalyzed by 51 transport QoS hardening actions that stabilized highly fragile best-effort links. Physical reallocations via container anti-affinity emerged as the most frequently emitted refactoring operator overall, ledgering 409 moves in the hyper-scale system to resolve hardware SPOF concentrations.

### 6.3 Remediation Efficacy at Component Boundaries

To thoroughly audit the refactoring engine, we track individual component-level failure impacts before and after applying mutations. Because the current implementation applies the fully compiled policy unconditionally without an individual component filter, it delivers a mixed result that exposes the structural trade-offs of greedy graph mutation:

* **S02 (IoT Smart City):** Achieves a **+47.01%** mean reduction in individual component failure impact.
* **S04 (Healthcare):** Records a **+30.94%** component risk improvement.
* **S05 (Hub-and-Spoke):** Suffers a **−31.67%** degradation in mean component metrics.
* **S07 (Hyper-Scale Enterprise):** Experiences a **−25.36%** regression at internal component boundaries.

This mixed performance—averaging **+4.61%** across the corpus—unveils that greedy operators, such as physical reallocation, can introduce additional network cascade hops (`CONNECTS_TO` extensions) that backfire on isolated components, highlighting the absolute necessity of building a strict per-edit acceptance filter.

### 6.4 CI/CD Gating Feasibility and Performance Overheads

Continuous software analytics tools must possess sub-minute execution boundaries to survive as blocking quality gates without frustrating engineering teams. We audited the execution time of `detect_antipatterns.py` using an isolated `MemoryRepository` on standard runner hardware across scales:

* **Tiny / Small Scales ($\le 25$ vertices):** Runs in **$< 2$ seconds**.
* **Medium Scale (~50 components, S01):** Completes execution in **$\approx 5$ seconds**.
* **Large Scale (80–100 components):** Completes in **$\approx 12$ seconds**.
* **Xlarge Scale (150–300 components, S07):** Completes analysis and cascade simulation in **$\approx 40$ seconds** (largely dominated by layout rendering metrics).

The complete generate–verify loop for the full prescriptive pipeline completes in under 65 seconds for six of the seven standard scenarios, and takes 10.8 minutes (649.6s) to exhaustively process the 3,245 structural edges of the hyper-scale enterprise fabric. This proves that continuous static system analysis is computationally viable for nightly integration pipelines or merge-request gating stages.

---

## 7. Discussion and Threats to Validity

### 7.1 Construct Validity

Our primary optimization targets and verification oracles are defined internally by a discrete-event cascade simulator operating on synthetic models. SRI reductions demonstrate that prescriptions optimize system attributes *as the simulator represents them*, which may not perfectly transfer to real physical networks featuring dynamic runtime traffic congestion. We address this validity threat by utilizing operators derived from verified dependable computing habits—such as anti-affinity container topologies and strict QoS specifications—whose engineering validity holds independently of the simulation environment.

### 7.2 Internal Validity and the Independence Guarantee

The core threat to structural predictors is circular leakage, where features inadvertently read data from downstream labels. Our framework mathematically avoids this via a strict **independence guarantee**: all code metrics and multi-dimensional RMAV calculations operate strictly on $G_{\text{analysis}}$ (the derived projection layers), whereas the ground-truth labels and SRI evaluations are derived separately from raw $G_{\text{structural}}$ simulation waves. No simulation parameters ever feedback into the diagnostic metrics or candidate generation, preserving rigorous architectural isolation.

---

## 8. Conclusion and Future Work

This paper has presented an automated prescriptive refactoring and quality gating framework for distributed publish–subscribe middleware architectures. By extending the Software-as-a-Graph baseline, we introduced an end-to-end pipeline that translates code-level debt and topological vulnerabilities into actionable, simulation-verified refactoring commands. Implemented as a delta-aware CI/CD quality gate, the platform blocks architectural degradation within a fast performance envelope perfectly compatible with modern high-velocity software engineering workflows.

Future work will focus on integrating a per-edit simulation filter to eliminate counterproductive mutations before policy execution, introducing stochastic cascade models to validate robustness under genuine channel noise, and linking Large Language Model (LLM) code assistants to automatically generate pull requests that implement the synthesized architectural refactoring blueprints.

---

## References

* `[1]` P. T. Eugster, P. A. Felber, R. Guerraoui, A.-M. Kermarrec, "The many faces of publish/subscribe," *ACM Computing Surveys*, vol. 35, no. 2, pp. 114–131, 2003.
* `[2]` Object Management Group, "Data Distribution Service (DDS)," OMG Document formal/2015-04-10, version 1.4, 2015.
* `[3]` OASIS, "MQTT Version 5.0," OASIS Standard, 2019.
* `[4]` M. Harman, S. A. Mansouri, Y. Zhang, "Search-Based Software Engineering: Trends, Techniques and Applications," *ACM Computing Surveys*, 45(1), Article 11, 2012.
* `[5]` A. Aleti, B. Buhnova, L. Grunske, A. Koziolek, I. Meedeniya, "Software Architecture Optimization Methods: A Systematic Literature Review," *IEEE Transactions on Software Engineering*, 39(5), 658–683, 2013.
* `[6]` L. C. Freeman, "A set of measures of centrality based on betweenness," *Sociometry*, vol. 40, no. 1, pp. 35–41, 1977.
* `[7]` T. L. Saaty, *The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation*, McGraw-Hill, 1980.
* `[8]` A. Varbella, K. Amara, M. El-Assady, B. Gjorgiev, G. Sansavini, "PowerGraph: A power grid benchmark dataset for graph neural networks," *NeurIPS 2024 Datasets and Benchmarks Track*, 2024.