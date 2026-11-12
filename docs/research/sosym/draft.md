# Software-as-a-Graph: A Model-Driven Framework for Graph-Based Architectural Modeling, Diagnostic Analysis, and Counterfactual Refactoring of Publish–Subscribe Systems

* **Target Journal:** Software and Systems Modeling (SoSyM) — Springer
* **Target Venue:** Tool Paper Call
* **Authors:** Ibrahim Onuralp Yigit, Feza Buzluca

---

## Abstract

Distributed publish--subscribe architectures---such as the Robot Operating System (ROS 2), Data Distribution Service (DDS), and MQTT---form the backbone of modern robotics, cyber-physical systems, and event-driven microservices. While asynchronous topic-based communication provides spatial and temporal decoupling, it obscures topological dependencies and failure propagation pathways, creating an **Architecture-Implementation Gap** where pristine component codebases can harbor severe system-level vulnerabilities. In this tool paper, we present **Software-as-a-Graph (SaG)**, an extensible, open-source model-driven engineering (MDE) framework for graph-based architectural modeling, diagnostic quality analysis, and counterfactually verified refactoring of publish--subscribe software systems.

SaG extracts typed, multi-layer multigraph models from Architecture-as-Code descriptors, folds source-level static code metrics into a **Code Quality Penalty (CQP)**, and computes an explainable, deterministic quality attribution model grounded in ISO/IEC 25010 and ISO/IEC 25019 standards across Reliability (Fault Tolerance and Availability) and Maintainability. Over these architectural models, SaG formalizes and evaluates a catalog of nineteen publish--subscribe anti-patterns using adaptive box-plot thresholds. To remediate diagnosed technical debt, the tool features a prescriptive engine that compiles findings into typed graph-mutation refactoring operators---logical topic splitting, physical host anti-affinity reallocation, and transport QoS contract hardening. Crucially, SaG enforces a closed-loop **"verify-before-transform"** discipline: candidate refactorings are counterfactually evaluated in an in-memory simulation sandbox, admitting only edits whose cascade-impact reduction exceeds stochastic seed noise across propagation thresholds ($\overline{\Delta I} > \kappa \sigma_{\text{seed}}$).

SaG is engineered as a decoupled hexagonal architecture featuring a dual-repository pattern: a graph database adapter (`Neo4jRepository`) for persistent querying and interactive exploration, and a high-performance in-memory adapter (`MemoryRepository`) that executes over 14,000 counterfactual simulation sweeps in under 3 hours. The tool provides multi-modal interfaces serving diverse stakeholder personas, including an interactive web dashboard and graph explorer, a Python SDK, a modular CLI suite, a REST API, and automated CI/CD regression gates. We evaluate SaG across eight multi-domain benchmark scenarios (29--520 vertices) and three transcribed real-world open-source systems (Autoware.Universe, Apollo Cyber RT, and an MQTT cloud mesh). We candidly report key empirical findings and calibration realities, including ranking saturation among structural predictors (NDCG@10 $\approx 0.82$--$0.86$, demonstrating that domain-grounded attribution is the primary value over scalar centrality), catalog discrimination improvements on real-world topologies ($\kappa = +0.296$ vs. chance on synthetic graphs), and a $71.0\%$ refactoring admission rate. Finally, we articulate five transferable lessons learnt spanning metamodel substrate adequacy, in-memory sandboxing, and simulation-in-the-loop verification to benefit the broader systems modeling community.

**Keywords:** Model-driven engineering; Software and systems modeling; Architectural models; Graph metamodeling; Anti-pattern detection; Counterfactual refactoring; Publish--subscribe middleware; Simulation-based verification; Tool paper.

---

## 1. Introduction

### 1.1 Context and Domain Motivation: Modeling Publish--Subscribe Systems
Distributed publish--subscribe (pub--sub) middleware frameworks---including the Robot Operating System (ROS 2), the Object Management Group's Data Distribution Service (DDS), Apache Kafka, and MQTT---form the foundational communication substrate of modern robotics, autonomous vehicles, industrial IoT, and distributed microservices. By decoupling message producers and consumers across space (anonymous communication), time (asynchronous message delivery), and synchronization (non-blocking publish and subscribe operations) via intermediary topics and brokers, these architectures enable high scalability, modularity, and operational flexibility.

However, this decoupling introduces a fundamental challenge for Model-Driven Engineering (MDE) and software architecture analysis: *it obscures the end-to-end topological dependencies of the system*. In publish--subscribe environments, components do not maintain explicit point-to-point references to one another; instead, dependencies are mediated dynamically through topic contracts, message brokers, and Quality of Service (QoS) transport profiles. Consequently, cascading failures, buffer overflows, broker bottlenecks, and timing jitter propagate along indirect architectural pathways that remain invisible when analyzing individual source code files or static component interfaces in isolation.

### 1.2 Open Challenges in Architectural Modeling and Tool Support
Despite the proliferation of model-based tools for software design and dependability analysis, engineering teams managing publish--subscribe systems face three critical tool and methodological gaps:
1. **The Architecture-Implementation Gap in MDE:** Classical static analysis platforms operate at the file, class, or method level. Conversely, conventional Architecture Description Languages (ADLs) and MDE frameworks (e.g., AADL, Palladio Component Model) have historically been optimized for synchronous Remote Procedure Call (RPC) or procedural client-server paradigms. As a result, an event-driven system can achieve pristine unit-level code quality while harboring severe architectural vulnerabilities.
2. **Lack of Domain-Specific Anti-Pattern Catalogs:** In object-oriented programming and microservices, mature catalogs formalize bad smells. In contrast, publish--subscribe systems lack a formalized catalog of architectural anti-patterns stated over their native abstractions (topics, brokers, QoS contracts).
3. **Open-Loop Model Refactoring vs. Verified Transformation:** While search-based model engineering (SBSE) and automated refactoring tools propose structural modifications, they almost universally operate *open-loop*: applying syntactic transformation rules without verifying whether the mutated architecture actually prevents failure propagation under dynamic message workloads.

### 1.3 The SaG Framework: An MDE Tool for Publish--Subscribe Systems
To resolve these challenges, we introduce **Software-as-a-Graph (SaG)**, an open-source, extensible Model-Driven Engineering framework designed to model, diagnose, and counterfactually refactor distributed publish--subscribe architectures. SaG extracts typed, multi-layer multigraph models from Architecture-as-Code descriptors, evaluates architectural health against international quality standards, and executes verified graph-mutation refactorings.

The framework integrates two coordinated modeling pathways within a unified, hexagonal architecture:
* **The Diagnostic Modeling Pathway:** Ingests declarative architecture manifests and static code metrics, synthesizing them into a Code Quality Penalty (CQP). It constructs a multi-layer heterogeneous dependency graph ($G_{\text{analysis}}$) and computes a hierarchical Reliability--Maintainability (RM) attribution model grounded in ISO/IEC 25010 and ISO/IEC 25019. Over these models, it detects nineteen formalized publish--subscribe anti-patterns using adaptive box-plot thresholds.
* **The Counterfactual Prescriptive Pathway:** Compiles diagnosed architectural pathologies into three formal graph-mutation operators (logical topic splitting, physical host anti-affinity reallocation, transport QoS contract hardening), enforcing a closed-loop "verify-before-transform" discipline using in-memory simulation sandboxes.

### 1.4 Target Audience and Tool Use Cases
1. **Software and Systems Architects:** Model event-driven topologies before deployment, visualize cross-layer dependencies, and evaluate verified refactoring blueprints.
2. **Reliability and SRE Engineers:** Decompose risks into Fault Tolerance and Availability without destructive production testing.
3. **MDE and Systems Modeling Researchers:** Leverage an extensible, open-source experimental platform for evaluating novel graph metrics and model transformation algorithms on realistic benchmarks.
4. **DevOps and Quality Assurance Teams:** Integrate automated quality gates into CI/CD pipelines.

---

## 2. Background and Related Modeling Tools

### 2.1 Model-Driven Engineering and Architectural Description Languages
Model-Driven Engineering (MDE) elevates high-level system models to primary engineering artifacts. Architectural Description Languages such as AADL and the Palladio Component Model (PCM) formalize structural hierarchies and execution platforms for dependability and performance forecasting. However, existing ADLs are primarily tailored for synchronous RPC, client-server, or hardware-bus systems. In event-driven pub-sub systems, communication is mediated through anonymous topics, brokers, and QoS transport contracts. SaG advances MDE by formalizing a domain-specific, typed multigraph metamodel specifically engineered to capture publish--subscribe topology semantics.

### 2.2 Architectural Smell and Anti-Pattern Detection Tools
Specialized tools such as Arcan, Designite, and DV8 detect dependency smells from reverse-engineered source-code graphs. However, these tools operate on module, package, or class dependency graphs recovered from source code; they possess no representation for topics, message brokers, partition queues, or QoS transport policies. Furthermore, empirical studies by Lenarduzzi et al. show that static rule-based tools often correlate weakly with real-world fault proneness. SaG addresses this by grounding its 19-pattern catalog in a multi-layer graph metamodel and evaluating detection against dynamic cascade simulation.

### 2.3 Model Refactoring: Open-Loop Search vs. Counterfactual Verification
Automated model refactoring has been approached through rule-based model transformations, search-based software engineering (SBSE), and simulation-in-the-loop optimization (e.g., PerOpteryx). Despite these advances, the vast majority of architectural refactoring engines operate open-loop with respect to failure propagation. SaG introduces a closed-loop "verify-before-transform" discipline: candidate graph mutations are executed in an in-memory sandbox and evaluated against a discrete-event cascade simulator across multiple seeds and thresholds ($\overline{\Delta I} > \kappa \sigma_{\text{seed}}$).

### 2.4 Advance Over State of the Art

| Capability / Characteristic | Palladio / PerOpteryx | AADL | Arcan / Designite | SonarQube | SaG (This Tool) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Pub--Sub Semantics (Topics, Brokers, QoS) | No | Partial | No | No | **Yes (First-Class)** |
| Multi-Layer Heterogeneous Metamodel | Component/Hardware | Component/Bus | Class/Package | File/Line | **Yes (4 Layers, 5 Types)** |
| Standardized Quality Model | Custom metrics | Custom annexes | Smells only | Technical Debt | **ISO/IEC 25010 & 25019** |
| Domain-Specific Anti-Pattern Catalog | Performance only | Fault trees | OO / Microservices | Code rules | **19 Pub--Sub Patterns** |
| Code-to-Architecture Bridge (CQP) | No | No | No | Native (AST) | **Yes (Continuous CQP)** |
| Refactoring Verification Discipline | Open-loop / Sim | Analytic / Open | Open-loop | Advisory only | **Closed-Loop Simulation** |
| In-Memory Sandboxed Transformation | No | No | No | No | **Yes (`MemoryRepository`)** |
| Multi-Modal Interfaces (Web, SDK, CLI) | Eclipse IDE | OSATE / Eclipse | CLI / Desktop | Web Dashboard | **Web, SDK, CLI, REST** |
| CI/CD Regression Quality Gating | No | No | Partial | Yes | **Yes (3-Tier Exit Codes)** |
| Open Source License | EPL | Open source | Proprietary/Academic | LGPL / Freemium | **Apache 2.0 (Open)** |

---

## 3. Graph Metamodel and Multi-Layer System Representation

### 3.1 Heterogeneous Graph Metamodel
In SaG, a distributed publish--subscribe software architecture is formalized as a typed, attributed, directed multigraph:
$$G = (V, E, \tau_V, \tau_E, w_E, w_V)$$
where:
* $T_V = \{\text{Application}, \text{Library}, \text{Topic}, \text{Broker}, \text{Node}\}$
* $T_E = \{\text{PUBLISHES\_TO}, \text{SUBSCRIBES\_TO}, \text{ROUTES}, \text{RUNS\_ON}, \text{CONNECTS\_TO}, \text{USES}\}$

### 3.2 Derived Architectural Dependencies: The `DEPENDS_ON` Projection
SaG projects $G$ into a directed analysis graph $G_{\text{analysis}} = (V, E_{\text{depends}})$ via 7 formal derivation rules:
1. Application-to-Application: $(A_1 \to A_2) \in E_{\text{depends}}$ if $A_1$ subscribes to topic $T$ published by $A_2$.
2. Application-to-Broker: $(A \to B) \in E_{\text{depends}}$ if broker $B$ routes topic $T$.
3. Application-to-Library: $(A \to L) \in E_{\text{depends}}$ if application $A$ links shared library $L$.
4. Host Co-location: $(A_i \to N) \in E_{\text{depends}}$ capturing shared compute node vulnerabilities.

Projections span four architectural layers:
* Application Layer ($V_{\text{app}} \cup V_{\text{lib}}$)
* Infrastructure Layer ($V_{\text{node}}$)
* Middleware Layer ($V_{\text{app}} \cup V_{\text{broker}}$)
* System Layer (All 5 entity types)

### 3.3 The Code Quality Penalty (CQP)
Synthesizes normalized source-level static code metrics:
$$\mathrm{CQP}(v) = 0.10\,\text{loc\_norm}(v) + 0.35\,\text{complexity\_norm}(v) + 0.30\,\text{instability\_code}(v) + 0.25\,\text{lcom\_norm}(v)$$

### 3.4 Hierarchical Reliability--Maintainability (RM) Quality Model
* **Fault Tolerance ($FT$):** Estimates cascade reach via Reverse PageRank ($\mathrm{RPR}$), normalized in-degree ($\mathrm{DG}_{\text{in}}$), and cascade depth potential ($\mathrm{CDPot}$):
  $$FT(v) = 0.45\,\mathrm{RPR}(v) + 0.30\,\mathrm{DG}_{\text{in}}(v) + 0.25\,\mathrm{CDPot}(v)$$
  For Topics: $FT_{\text{topic}}(t) = 0.50\,\mathrm{FOC}(t) + 0.50\,\mathrm{FOC}(t)(1 - \min(w_{\text{in}}(t), 1))$
* **Availability ($A$):** Evaluates single-point-of-failure exposure:
  $$A(v) = 0.25\,\mathrm{AP}_{c,\text{directed}}(v) + 0.20\,\mathrm{QSPOF}(v) + 0.20\,\mathrm{BR}(v) + 0.25\,\mathrm{CDI}(v) + 0.10\,w(v)$$
* **Maintainability ($M$):** Blends betweenness ($\mathrm{BT}$), efferent QoS weight ($w_{\text{out}}$), $\mathrm{CQP}$, coupling risk, and clustering ($\mathrm{CC}$):
  $$M(v) = 0.35\,\mathrm{BT}(v) + 0.30\,w_{\text{out}}(v) + 0.15\,\mathrm{CQP}(v) + 0.12\,\mathrm{CouplingRisk\_enh}(v) + 0.08\,(1-\mathrm{CC}(v))$$
* **Reliability Composite ($R$):** $R(v) = 0.36\,FT(v) + 0.64\,A(v)$
* **Composite Criticality ($Q$):** $Q(v) = 0.80\,R(v) + 0.20\,M(v)$
* **Stakeholder Harm Projection ($\mathbf{h}_{\text{QiU}}$):** Maps $[R(v), M(v)]^T$ into ISO 25019 harm categories via row-stochastic matrix $\mathbf{M}_{\text{RM} \to \text{QiU}}$.

### 3.5 Substrate Adequacy
Availability predicates (such as articulation points) depend on the graph projection. On dense application layers without brokers or hardware nodes, cut-vertices do not exist ($F_1 = 0.0$). Availability analysis must be evaluated on Middleware or Infrastructure layers where bottlenecks physically exist.

---

## 4. Architectural Anti-Pattern Catalog

SaG formalizes nineteen architectural anti-patterns evaluated with adaptive box-plot fences ($Q_3 + k\,\mathrm{IQR}$, $k=1.5$):

| Pattern | Severity | Category | Scope | Detection Signal | Automated Refactoring | Triggered on Benchmark |
|:---|:---:|:---:|:---:|:---|:---|:---:|
| `SPOF` | CRITICAL | Availability | Component | Articulation point $\wedge$ QoS SPOF score | Anti-affinity reallocation | Yes |
| `SYSTEMIC_RISK` | CRITICAL | Reliability | System | Share of CRITICAL components $> 20\%$ | Advisory | No |
| `CYCLE` | HIGH | Architecture | System | Strongly connected component / loop | Advisory | Yes |
| `GOD_COMPONENT` | CRITICAL | Maintainability | Component | Extreme betweenness $\wedge$ CRITICAL $M(v)$ | Logical topic splitting | No |
| `BOTTLENECK_EDGE` | HIGH | Availability | Edge | Edge betweenness centrality outlier | Logical topic splitting | Yes |
| `BROKER_OVERLOAD` | HIGH | Availability | Component | Broker availability $\ge 2\times$ median load | Anti-affinity reallocation | No |
| `DEEP_PIPELINE` | HIGH | Reliability | System | Path length $\ge \max(5, P_{75})$ | Advisory | Excluded |
| `TOPIC_FANOUT` | MEDIUM | Reliability | Component | Topic subscriber out-degree outlier | Logical topic splitting | No |
| `CHATTY_PAIR` | MEDIUM | Maintainability | Pair | Bidirectional weight product $> \tau_{\text{chatty}}$ | Advisory | No |
| `QOS_MISMATCH` | MEDIUM | Reliability | Edge | Pub/Sub QoS-weight gap $> \tau_{\text{qos}}$ | Advisory (Relay) | Yes |
| `ORPHANED_TOPIC` | MEDIUM | Maintainability | Component | Zero in- or out-degree on graph | Advisory | No |
| `UNSTABLE_INTERFACE` | MEDIUM | Maintainability | Component | High coupling risk $\wedge$ high $M(v)$ | Advisory | Yes |
| `BRIDGE_EDGE` | HIGH | Availability | Edge | Graph-theoretic bridge cut-edge | Advisory | Yes |
| `FAILURE_HUB` | CRITICAL | Reliability | Component | Reliability outlier $\wedge$ high out-degree | Logical topic splitting | No |
| `CONCENTRATION_RISK` | MEDIUM | Reliability | System | Top-3 PageRank share $> 0.50$ | Advisory | No |
| `HUB_AND_SPOKE` | MEDIUM | Maintainability | Component | Low clustering $\wedge$ degree $> 3$ | Logical topic splitting | Yes |
| `CHAIN` | MEDIUM | Architecture | System | Degree-bounded linear subgraph | Advisory | No |
| `ISOLATED` | MEDIUM | Architecture | Component | Zero total degree | Advisory | Yes |
| `COMPOUND_RISK` | CRITICAL | Architecture | Component | Co-occurring SPOF + God/Hub finding | Anti-affinity reallocation | No |

---

## 5. Tool Architecture and Implementation Techniques

### 5.1 Hexagonal Architecture and Dual-Repository Design
SaG decouples domain modeling from graph storage via the `IGraphRepository` port:
* `Neo4jRepository`: Persistent database adapter using Bolt and Cypher for storage and visualization.
* `MemoryRepository`: High-performance in-memory repository featuring vectorized adjacency tables and copy-on-write sandboxing, executing counterfactual simulations in microseconds.

### 5.2 Seven Pipeline Stages
Ingest $\to$ Project $\to$ Attribute $\to$ Detect $\to$ Compile $\to$ Verify $\to$ Present/Gate.

### 5.3 Graph-Mutation Refactoring Operators
1. **Logical Topic Splitting (Operator 1):** Decomposes multi-publisher topics into segregated sub-topics.
2. **Physical Anti-Affinity Reallocation (Operator 2):** Migrates co-located critical applications to isolated physical hosts.
3. **Transport QoS Contract Hardening (Operator 3):** Upgrades volatile best-effort topics to reliable, persistent contracts.

### 5.4 Closed-Loop Counterfactual Verification Engine
* **Level 1 (Per-Edit Filter):** An edit $\delta$ is admitted iff $\overline{\Delta I}_\theta(\delta) > \kappa \cdot \sigma_{\text{seed},\theta}(\delta)$ across all propagation thresholds $\Theta = \{0.1, 0.2, 0.5\}$ and seeds $S = \{42, 123, 456\}$.
* **Level 2 (Whole-Policy Gate):** The joint admitted policy must satisfy $\Delta\mathrm{SRI} > 0$.

---

## 6. Tool Interfaces, Usability, and Engineering Ecosystem

### 6.1 Multi-Modal Interfaces
1. **Interactive Web Dashboard & Graph Explorer:** Developed in Next.js/React and zero-dependency standalone HTML (`saag.visualization.dashboard`), rendering 2D/3D force-directed layouts, cascade heatmaps, RM radar charts, and dependency matrices.
2. **Python SDK & Client API:** `SaagClient` provides programmatic control for researchers and automation scripts.
3. **Modular CLI Suite:** Headless Unix commands (`saag model`, `saag diagnose`, `saag prescribe`, `saag simulate`, `saag validate`).
4. **RESTful API Service:** Production-ready asynchronous FastAPI service with OpenAPI 3.0 specification.

### 6.2 DevOps Integration and Quality Gating
* **Regression Modes:** Absolute Gating (rigid, blocks on pre-existing legacy debt) vs. Delta-Aware Gating (evaluates pull requests against target branch merge-base with waiver registers).
* **Three-Tier Exit Codes:** 0 (Pass), 1 (Warn), 2 (Block).

---

## 7. Empirical Application in Realistic Scenarios

### 7.1 Corpora
* **Synthetic Benchmark Scenarios (S01--S08):** 8 multi-domain scenarios (29 to 520 vertices across Autonomous Vehicles, Smart Cities, Financial Trading, Healthcare, Hub-and-Spoke, Microservices, Hyper-Scale, Regression).
* **Transcribed Real-World Architectures:** Autoware.Universe (ROS 2/DDS autonomous driving stack), Train-Ticket microservices, Cloud IoT Mesh (MQTT).

### 7.2 Key Results
* **RQ1 (Diagnostic Ranking Efficacy):** Predictor saturation across $Q(v)$ ($\rho = 0.485$, NDCG@10 $= 0.830$), betweenness (NDCG@10 $= 0.859$), and degree centrality ($\rho = 0.519$). No single predictor dominates; deterministic role-specific attribution is the primary value.
* **RQ2 (Catalog Calibration):** Synthetic topologies yield chance agreement ($\kappa = -0.036$) due to $98.4\%$ edge-scoped findings on dense uniform graphs. On transcribed real-world systems, chance-corrected agreement improves markedly to $\mathbf{\kappa = +0.296}$ with $40.2\%$ precision and $56.3\%$ component implication.
* **RQ3 (Substrate Adequacy):** Application-layer articulation points yield $F_1 = 0.0$ across all 8 scenarios, demonstrating the necessity of layer-aware evaluation. Stratification resolves Simpson's paradox ($\rho = 0.028$ pooled vs. $0.503$ on applications).
* **RQ4 (Prescriptive Refactoring):** 1128 of 1589 compiled candidates ($71.0\%$) survive counterfactual verification (Reallocation $77.7\%$, QoS Hardening $58.0\%$, Topic Splitting $49.7\%$). Five of seven scenarios achieve net cascade-impact reduction (mean $+3.52\%$, reaching $+18.9\%$ in Healthcare).
* **RQ5 (Tool Performance & Latency):** Diagnostic gate executes in 0.01--20.98 s (sub-second on small/medium topologies). Prescriptive verification of 14,301 simulation runs completed in 2 h 47 min on a 20-core workstation.

---

## 8. Community Importance and Lessons Learnt

### 8.1 Importance for Systems and Software Modeling
1. **Fills the MDE Asynchronous Pub--Sub Blind Spot:** Formalizes domain metamodels and smell detection for ROS 2, DDS, and MQTT.
2. **Standardized Open Benchmark Testbed:** Delivers 8 synthetic and 3 real-world open architectures with simulation ground truths.
3. **Reusable Anti-Pattern & Operator Catalog:** Formalizes 19 pub-sub anti-patterns and 3 mutation operators for the MDE community.
4. **Standardized Quality Grounding:** Operationalizes ISO/IEC 25010/25019 in graph models.

### 8.2 Five Transferable Lessons Learnt
1. **Lesson 1 (Designing — Metamodel Substrate Adequacy):** Metric validity depends on the projection substrate; discrete cut-vertex metrics fail on decoupled application layers and require Middleware or Infrastructure projections.
2. **Lesson 2 (Building — Dual Repositories for Model Transformation):** Graph databases excel at persistence, but in-memory copy-on-write sandboxes (`MemoryRepository`) are mandatory for microsecond simulation-in-the-loop sweeps.
3. **Lesson 3 (Testing — "Verify-Before-Transform" Discipline):** $29.0\%$ of compiled refactoring candidates failed verification; rule-based refactorings cannot be applied open-loop without cascade simulation.
4. **Lesson 4 (Evaluating — Attribution vs. Centrality Ranking):** Complex composite metrics saturate at comparable ranking quality with degree centrality; the true contribution is domain-grounded, explainable attribution.
5. **Lesson 5 (Disseminating — Multi-Modal Ergonomics):** Tool adoption requires synchronized web dashboards for architects, Python SDKs for researchers, and headless CLI gates for DevOps.

---

## 9. Discussion and Tool Availability

### 9.1 Threats to Validity
Construct validity (whole-policy gating circularity, substrate adequacy), internal validity (in-memory sandboxing), external validity (synthetic vs. real-world generalization), and conclusion validity ($n=7$ sample size caveats).

### 9.2 Tool Availability, Reproducibility, and Open Science
* **Open-Source License:** Apache License 2.0.
* **Source Code Repository:** Publicly available on GitHub.
* **Persistent Archival & DOI:** Permanently archived on Zenodo with DOI.
* **Packaging & Containerization:** Python package (`pyproject.toml`) and Docker Compose environment (`docker-compose.yml`).
* **Reproduction Package:** Automated scripts under `reproduce/` regenerate all reported tables and metrics.

---

## 10. Conclusion and Future Work
SaG establishes an extensible, model-driven engineering framework for distributed publish--subscribe software architectures, unifying multi-layer graph modeling, ISO-grounded diagnostic attribution, 19 formal anti-patterns, and counterfactually verified refactoring. Future work includes automated IDL extraction, layer-aware availability metrics, and multi-objective Pareto search optimization.
