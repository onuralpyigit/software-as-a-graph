# Measuring and Assessing Systemic Quality Attributes in Publish–Subscribe Middleware: A Multi-Dimensional Static Architecture Metrics Framework

**Target Venue:** *Software Quality Journal (SQJ) — Springer*

---

## Abstract

In large-scale distributed software engineering, individual component source code can be perfectly clean, passing all standard code-level quality checks, while the system configuration remains highly fragile. This "Architecture-Code Gap" is particularly severe in distributed publish–subscribe (pub–sub) middleware systems. The decoupled nature of pub–sub applications obscures the true logical dependency chains along which a single component defect can cascade. Traditional Static Code Analysis (SCA) platforms evaluate intra-component quality metrics but are blind to global, multi-layered topology risks before deployment.

This paper presents **Software-as-a-Graph (SaG)**, a comprehensive **Static System Analysis (SSA)** metrics framework designed to quantify and gate systemic software quality before software execution or deployment. SaG maps complex pub–sub environments into a typed, weighted, directed multigraph over components, topics, routing brokers, and deployment nodes, deriving logical dependencies via typed projection rules. We integrate code-level SCA attributes (e.g., lines of code, cyclomatic complexity, lack of cohesion of methods) as internal vertex properties and formalize a multi-dimensional assessment engine. The engine decomposes software criticality into four orthogonal quality dimensions: **Reliability, Maintainability, Availability, and Vulnerability (RMAV)**, which are synthesized into an overall quality index $Q(v)$ using an Analytic Hierarchy Process (AHP) formulation smoothed by a shrinkage uniform prior.

We validate the framework by verifying these static quality metrics against an independent discrete-event failure-propagation simulator under a strict input–label independence guarantee. Our evaluation across seven distinct domain topologies surfaces two critical software quality phenomena: (1) a **shared-library blast-radius gap**, where components with high structural fan-out but low composite code centrality drive system-wide failure cascades, and (2) a **Simpson’s-paradox stratification effect**, demonstrating that pooled architecture evaluations introduce representation collapse, requiring strict node-type stratification for honest quality reporting. Finally, we implement the framework as an automated, build-blocking Quality Gate pipeline. Operating via an in-memory repository to eliminate live graph database latencies, the gate executes structural anti-pattern detection and quality regressions in seconds ($<5$s for medium, $\approx40$s for extra-large topologies), proving the feasibility of continuous systemic quality control in rapid-release cycles. We demonstrate external validity by evaluating the metrics against blind expert consensus on an operational, safety-critical air-traffic-management system.

**Keywords:** Software Quality Metrics; Static System Analysis; Publish–Subscribe Middleware; Quality Gates; Dependency Analysis; Architecture-Code Gap; Reliability and Maintainability.

---

## 1. Introduction

Continuous Quality Assurance (QA) in modern distributed software engineering has achieved high levels of automation at the implementation layer. Teams regularly employ Static Code Analysis (SCA) tools within continuous integration (CI) pipelines to evaluate code metrics, detect technical debt, and ensure adherence to local safety patterns. However, an operational system can possess pristine source code across every microservices module while remaining structurally unstable. This disconnect defines the **Architecture-Code Gap**: a critical vulnerability where code-level excellence masks systemic architectural risk.

This gap is uniquely pronounced in publish–subscribe (pub–sub) middleware architectures, such as those relying on the Data Distribution Service (DDS), MQTT, or Apache Kafka. The fundamental value proposition of pub–sub middleware is spatial, temporal, and synchronization decoupling. Applications produce or consume data asynchronously through named topics or intermediate brokers without possessing explicit structural references or knowledge of counterparties. While this decoupling yields highly scalable systems, it obscures logical dependency lines. If an active publisher application crashes, or an underlying message-routing broker becomes overloaded, the resulting failure does not follow a classical call graph. Instead, it propagates along hidden dependencies: shared topics, colocated processing nodes, or shared code libraries whose failures strike every consuming application simultaneously.

Identifying these fragile dependencies is essential *before* a system is deployed to production, a stage where architectural modification is least disruptive and most cost-effective. Yet, prior to production runtime, empirical telemetry does not exist. Software quality engineers must evaluate system integrity solely from design models and deployment configurations.

To solve this challenge, this paper introduces **Software-as-a-Graph (SaG)**, a continuous Static System Analysis (SSA) framework. SaG models a distributed middleware environment as a typed, weighted, directed multigraph, deriving hidden logical dependencies and combining local code metrics with global topological structures. SaG structures this assessment around four core quality properties: **Reliability ($R$), Maintainability ($M$), Availability ($A$), and Vulnerability ($V$)**, collectively termed the **RMAV framework**. Rather than reducing architectural risk to an uninterpretable scalar centrality, the RMAV framework isolates independent structural mechanisms, providing a clear diagnostic profile that points directly to specific engineering remedies.

This paper structures its exploration around five primary research questions focused on the measurement, validation, and automated gating of software architecture quality:

* **RQ1:** When does deterministic, multi-dimensional quality attribution suffice for pre-deployment system assessment, and when is a learned heterogeneous graph model necessary to capture cascade patterns?
* **RQ2:** What architectural quality failure modes are exposed by multi-dimensional RMAV attribution that standard single-score centrality or code linting tools miss?
* **RQ3:** How does the explicit structural injection of multi-attribute Quality-of-Service (QoS) contracts affect cross-topology generalization within inductive assessment loops?
* **RQ4:** What is the performance overhead and detection accuracy of deploying a graph-based system analyzer as a blocking Quality Gate in continuous delivery pipelines?
* **RQ5:** Do the framework's calculated architectural quality rankings correlate with the empirical consensus of senior software system engineers on a live production system?

The remainder of this manuscript detail the meta-model, metric formulations, validation protocols, and empirical evaluations that address these questions.

---

## 2. Related Work

### 2.1 Static Code Analysis (SCA) vs. Static System Analysis (SSA)

The software quality discipline possesses deep foundations in structural measurement. Traditional SCA platforms (e.g., SonarQube) analyze source files to construct Abstract Syntax Trees (ASTs), computing code-level indicators such as cyclomatic complexity, code duplication density, and cohesion indices like Lack of Cohesion of Methods (LCOM). While highly effective for module-level verification, these tools remain unaware of inter-component connections, network routes, and dynamic middleware propagation vectors.

Static System Analysis (SSA) extends static analysis into the system domain. Rather than treating code in isolation, SSA processes software as an interconnected graph of communicating components, routing nodes, and hardware boundaries. SaG acts as a bridge for this gap, ingestion local SCA indicators and embedding them directly within a global topological framework.

### 2.2 Network Centrality and Dimensional Collapse

Network science provides various centrality metrics—such as degree, closeness, betweenness, and eigenvector centrality—frequently applied to software package structures to locate system bottlenecks. However, standard graph implementations suffer from dimensional collapse when evaluating typed middleware: they flatten distinct structural nodes (e.g., applications, messaging topics, physical servers) into a uniform, untyped network view. This flattening erases relation-specific mechanics, such as the simultaneous blast radius of a shared binary library or the unique failover patterns of redundant messaging brokers. The RMAV framework avoids this collapse by enforcing absolute input orthogonality, ensuring that metrics evaluate independent properties.

### 2.3 Continuous Gating and Architecture-as-Code

Continuous pre-deployment quality validation requires treating systemic configurations with the same rigor as source code. In modern DevOps, this is enabled by "Architecture-as-Code" (AaC) techniques, where deployment relationships are explicitly declared via infrastructure configuration descriptors like Docker Compose manifests, Kubernetes configurations, or Helm definitions. While techniques like Chaos Engineering (e.g., injecting operational faults via chaos tools) explore structural resiliency dynamically, they require running staging clusters and occur late in delivery cycles. SaG parses AaC descriptors within automated build loops, introducing static system quality checks that run prior to deployment.

---

## 3. The Software-as-a-Graph Meta-Model

The base configuration layer models an architecture as a typed, weighted, directed multigraph:

$$G = (V, E, \tau_V, \tau_E, w_E, w_V)$$

The graph decomposes into distinct entities and communication abstractions to maintain semantic accuracy during structural computation.

### 3.1 Structural Node and Edge Taxonomy

The vertex set $V$ cleanly partitions into five independent structural node scopes:

$$V = V_{\text{app}} \cup V_{\text{broker}} \cup V_{\text{topic}} \cup V_{\text{node}} \cup V_{\text{lib}}$$

where:

* $V_{\text{app}}$ represents active application components or executables producing or consuming streams.
* $V_{\text{broker}}$ represents intermediate routing proxies or messaging cluster components.
* $V_{\text{topic}}$ represents distinct, typed logical information channels or message exchanges.
* $V_{\text{node}}$ isolates virtual or hardware host systems executing software processes.
* $V_{\text{lib}}$ models static or dynamic cross-cutting libraries linked by multiple execution applications.

The type functions $\tau_V : V \to \{\text{App}, \text{Broker}, \text{Topic}, \text{Node}, \text{Library}\}$ and $\tau_E$ categorize nodes and connections. Six base structural edge connections are imported directly from the target architecture definition:

* $\text{PUBLISHES\_TO} \subseteq (V_{\text{app}} \cup V_{\text{lib}}) \times V_{\text{topic}}$
* $\text{SUBSCRIBES\_TO} \subseteq (V_{\text{app}} \cup V_{\text{lib}}) \times V_{\text{topic}}$
* $\text{ROUTES} \subseteq V_{\text{broker}} \times V_{\text{topic}}$
* $\text{RUNS\_ON} \subseteq (V_{\text{app}} \cup V_{\text{broker}}) \times V_{\text{node}}$
* $\text{CONNECTS\_TO} \subseteq V_{\text{node}} \times V_{\text{node}}$
* $\text{USES} \subseteq V_{\text{app}} \times V_{\text{lib}}$

### 3.2 QoS-Aware Coupling Strength Formulation

To ensure that metrics capture the technical intensity of structural relationships, edge weights are computed dynamically from declared Quality-of-Service (QoS) metadata parameters and estimated payload sizes:

$$\text{QoS\_score} = 0.30 \cdot r + 0.40 \cdot d + 0.30 \cdot p$$

$$\text{size\_norm} = \min\left(\frac{\log_2(1 + \text{size\_kb})}{50},\ 1.0\right)$$

$$w_E(e) = \beta \cdot \text{QoS\_score} + (1-\beta) \cdot \text{size\_norm}, \quad \beta = 0.85$$

where sub-weights are mapped from symbolic architectural configurations:

* **Reliability ($r$):** `RELIABLE` $\to 1.0$, `BEST_EFFORT` $\to 0.0$.
* **Durability ($d$):** `PERSISTENT` $\to 1.0$, `TRANSIENT` $\to 0.6$, `TRANSIENT_LOCAL` $\to 0.5$, `VOLATILE` $\to 0.0$.
* **Transport Priority ($p$):** `URGENT`/`CRITICAL` $\to 1.0$, `HIGH` $\to 0.66$, `MEDIUM` $\to 0.33$, `LOW` $\to 0.0$.

A minimal weight boundary floor ($w_E(e) \ge 0.01$) ensures complete reachability graphing across best-effort channels. Vertex weights ($w_V$) propagate these edge-level bounds via structural types:

* **Applications:** $w_V(v) = 0.80 \cdot \max(w_E) + 0.20 \cdot \text{mean}(w_E)$.
* **Brokers:** $w_V(v) = 0.70 \cdot \max(w_E) + 0.30 \cdot \text{mean}(w_E)$.
* **Nodes:** $w_V(v) = \max(w_V)$ of all locally executed components.
* **Libraries:** $w_V(v) = \min(1.0, w_{\text{base}} \cdot (1 + \gamma \log_2(1 + \text{DG\_in})))$, isolating code fan-out vulnerabilities.

### 3.3 Logical Dependency Projections: The `DEPENDS_ON` Layer

To convert physical paths into logical risk lines, SaG constructs a derived projection layer composed entirely of `DEPENDS_ON` edges pointing from a *dependent entity* to its required *dependency*. These edges are generated systematically via six architectural projection rules:

1. **Application-to-Application (`app_to_app`):** Generated when a consumer application subscribes to a topic populated by a provider application, capturing cascading functional reliance.
2. **Application-to-Broker (`app_to_broker`):** Formed between endpoints and the intermediate routing brokers responsible for managing their topic layers.
3. **Node-to-Node (`node_to_node`):** Evaluates physical hosting cross-reliance by lifting operational connections up to the machine layer.
4. **Node-to-Broker (`node_to_broker`):** Maps server-level operational reliance on centralized message routing infrastructures.
5. **Application-to-Library (`app_to_lib`):** Formed when an application links to a shared software library, capturing immediate, non-sequential blast-radius exposure.
6. **Broker-to-Broker (`broker_to_broker`):** Captures colocation risks where multiple message routers share matching infrastructure layers.

When multiple paths link two components, SaG aggregates the relationship into a single `DEPENDS_ON` edge, tracking the maximum QoS weight along with a separate multi-channel connectivity attribute:

$$\text{edge.weight} = \max_{t \in \text{shared}} w_E(t), \quad \text{edge.path\_count} = |\text{shared}|$$

This approach separates structural complexity from raw connection intensity.

### 3.4 Ingestion of Code-Level Quality Attributes

To bridge the Architecture-Code Gap, SaG collects internal modular quality data by querying SCA platform APIs (e.g., SonarQube's web API) or parsing static analysis output files during model extraction. These indicators are stored as local attributes prefixed with `cm_*` directly on $V_{\text{app}}$ and $V_{\text{lib}}$ nodes:

* `cm_total_loc`: Ingested volume scale representing total lines of code.
* `cm_avg_wmc`: Average Weighted Methods per Class, representing local complexity metrics.
* `cm_avg_lcom`: Lack of Cohesion of Methods, mapping design fragmentation.
* `cm_avg_cbo`: Coupling Between Objects, establishing localized internal source dependencies.
* `sqale_debt_ratio`: The technical debt index, tracking structural code debt relative to code size.

These metrics are normalized globally across component boundaries, feeding the structural maintainability scoring loops detailed in Section 4.

---

## 4. The RMAV Quality Assessment Framework

The core of our Static System Analysis framework is the multi-dimensional decomposition of architectural risk. Each component metrics baseline is calculated using rank-normalized structural values mapping strictly into $$.

### 4.1 Reliability ($R$)

Reliability metrics capture systemic fault-propagation vectors. Because logical dependency edges map from dependent to provider, cascades propagate *against* edge arrows. We therefore compute Reverse PageRank ($\mathrm{RPR}$) over the transposed graph configuration $G^\top$ to trace cascade depth:

$$R(v) = 0.60 \cdot \mathrm{RPR}(v) \cdot \big(1 + \mathrm{MPCI}(v)\big) + 0.40 \cdot \mathrm{DG\_in}(v) \quad [\text{for } \tau_V(v) \neq \text{Topic}]$$

$$R_{\text{topic}}(v) = 0.50 \cdot \mathrm{FOC}(v) + 0.50 \cdot \mathrm{CDPot\_topic}(v)$$

$$\mathrm{CDPot\_topic}(v) = \mathrm{FOC}(v) \cdot \big(1 - \min(\text{publisher\_count\_norm}(v), 1.0)\big)$$

where $\mathrm{MPCI}$ is the Multi-Path Coupling Index tracking edge counts, $\mathrm{DG\_in}$ measures direct structural dependents, and $\mathrm{FOC}$ isolates Fan-Out Criticality on message streams.

### 4.2 Maintainability ($M$)

Maintainability quantifies architectural coupling complexity and configuration decay. It combines topological bottlenecks with local code debt via a localized Code Quality Penalty ($\mathrm{CQP}$) factor:

$$M(v) = 0.35 \cdot \mathrm{BT}(v) + 0.30 \cdot \mathrm{w\_out}(v) + 0.15 \cdot \mathrm{CQP}(v) + 0.12 \cdot \mathrm{CouplingRisk\_enh}(v) + 0.08 \cdot (1 - \mathrm{CC}(v))$$

$$\mathrm{CQP}(v) = 0.10 \cdot \text{loc\_norm} + 0.35 \cdot \text{complexity\_norm} + 0.30 \cdot \text{instability\_code} + 0.25 \cdot \text{lcom\_norm}$$

where $\mathrm{BT}$ tracks betweenness centrality, $\mathrm{w\_out}$ captures efferent QoS coupling intensity, $\mathrm{CC}$ evaluates local clustering path redundancy, and $\mathrm{CouplingRisk\_enh}$ measures architectural structural imbalance. Components without codebase allocations default to a $\mathrm{CQP}$ value of zero, ensuring stability during computation.

### 4.3 Availability ($A$)

Availability tracks structural single points of failure ($\mathrm{SPOFs}$) capable of partitioning the graph layer:

$$A(v) = 0.35 \cdot \mathrm{AP\_c\_directed}(v) + 0.25 \cdot \mathrm{QSPOF}(v) + 0.25 \cdot \mathrm{BR}(v) + 0.10 \cdot \mathrm{CDI}(v) + 0.05 \cdot w_V(v)$$

where $\mathrm{AP\_c\_directed}$ evaluates directed articulation point occurrences, $\mathrm{QSPOF}$ weights articulation severity by transaction priority, $\mathrm{BR}$ measures the ratio of incident bridge edges, and $\mathrm{CDI}$ monitors average path-length inflation upon node removal.

### 4.4 Vulnerability ($V$)

Vulnerability measures adversarial target exposure across deployment zones:

$$V(v) = 0.40 \cdot \mathrm{REV}(v) + 0.35 \cdot \mathrm{RCL}(v) + 0.25 \cdot \mathrm{w\_in}(v)$$

where $\mathrm{REV}$ represents Reverse Eigenvector centrality on $G^\top$, tracking access vectors, $\mathrm{RCL}$ computes Reverse Closeness to entry zones, and $\mathrm{w\_in}$ checks high-priority incoming service surfaces.

### 4.5 Hierarchical Synthesis and Prior Shrinkage Blending

To synthesize the RMAV components into an overall quality index $Q(v)$, we apply an Analytic Hierarchy Process (AHP) weight vector derived from structured pairwise comparisons:

$$Q(v) = w_A \cdot A(v) + w_R \cdot R(v) + w_M \cdot M(v) + w_V \cdot V(v)$$

The raw base evaluation matrices achieve an acceptable consistency verification status ($\mathrm{CR} = 0.02 \le 0.10$), yielding the initial raw vector $(w_A, w_R, w_M, w_V) = (0.43, 0.24, 0.17, 0.16)$. To prevent over-fitting to specialized topology configurations, we smooth the raw AHP weights toward a uniform prior using a shrinkage factor $\lambda$:

$$w_{\text{final}}(d) = \lambda \cdot w_{\mathrm{AHP}}(d) + (1-\lambda) \cdot \frac{1}{n_{\text{dim}}}$$

Fixing $\lambda = 0.70$ provides calibrated, balanced assessment weights of $(0.38, 0.24, 0.19, 0.19)$, preserving structural priorities while protecting the metrics against extreme local skew.

### 4.6 Adaptive Quality Classification Mapping

Rather than enforcing static, arbitrary thresholds, components are classified using an adaptive statistical distribution model based on the architectural graph's own metric spread:

$$\begin{aligned}
\text{CRITICAL} &\implies Q(v) > Q_3 + 1.5 \cdot \mathrm{IQR} \\
\text{HIGH} &\implies Q_3 < Q(v) \le Q_3 + 1.5 \cdot \mathrm{IQR} \\
\text{MEDIUM} &\implies \mathrm{Median} < Q(v) \le Q_3 \\
\text{LOW} &\implies Q_1 < Q(v) \le \mathrm{Median} \\
\text{MINIMAL} &\implies Q(v) \le Q_1
\end{aligned}$$

For tiny topologies ($n < 12$) where interquartile ranges ($\mathrm{IQR}$) are unstable, SaG applies a percentage fallback (CRITICAL $\implies$ top 10%, HIGH $\implies$ 75th–90th percentiles), ensuring robust quality categorization across varying system scales.

---

## 5. Verification via Failure Simulation

To confirm that our static metrics accurately measure operational software risk without relying on runtime execution, we evaluate the predictors against an independent discrete-event simulation engine. This engine operates directly on the raw structural layer ($G_{\text{structural}}$), maintaining strict decoupling from the derived projection layer ($G_{\text{analysis}}$) to enforce our **input–label independence guarantee**.

### 5.1 Ground-Truth Impact Index ($I(v)$)

The simulation engine injects single-node failure triggers, models downstream fault propagation across multiple execution loops, and computes an operational degradation index $I(v) \in$:

$$I(v) = 0.35 \cdot \text{reachability\_loss} + 0.25 \cdot \text{fragmentation} + 0.25 \cdot \text{throughput\_loss} + 0.15 \cdot \text{flow\_disruption}$$

where sub-components trace broken communication paths, graph partitioning metrics, throughput degradation across priority data streams, and broken message-delivery paths. Faults propagate downstream once a node's cumulative stream degradation reaches a specified `propagation_threshold` (defaulting to 0.2).

### 5.2 Dual Assessment Models

We evaluate framework performance using two predictive models:

1. **Deterministic Metrics Predictor:** Uses the AHP-weighted score $Q(v)$ directly on the static graph, providing clear interpretability with zero training cost.
2. **Heterogeneous Graph Transformer (HGL):** A deep graph neural network using type-specific node message blocks to learn complex, non-linear dependencies across heterogeneous architectures.

### 5.3 The Shared-Library Blast-Radius Gap

A key rationale for deploying typed static system analysis is uncovering the **shared-library blast-radius gap**. Standard untyped structural centrality filters often overlook cross-cutting code modules, as their immediate topological degree appears unremarkable. However, because binary libraries interact with calling software synchronously at compilation time, a library fault causes a simultaneous crash across all dependent processes. This phenomenon creates a severe quality gap: a library can register a low static metric profile ($Q \approx 0.48$) while causing total system failure ($I(v) \approx 0.97$) during simulated execution. SaG captures this risk via specific `app_to_lib` parsing layers, ensuring that hidden topological bottlenecks are made visible to quality engineers before deployment.

---

## 6. Automated Quality Gating in CI/CD

To operationalize the framework within continuous software delivery workflows, SaG integrates directly into deployment pipelines via a programmatic Quality Gate utility (`detect_antipatterns.py`).

```
  +-------------------------+
  |  Pull Request Trigger  |
  +------------+------------+
               |
               v
  +-------------------------+
  | Parse Architecture-as-  |
  | Code (AaC) Descriptors  |
  +------------+------------+
               |
               v
  +-------------------------+       +-------------------------+
  | Ingest Local Static     |------>|  Query SCA Platform API |
  | Code Analysis Metrics   |       |  (SonarQube cm_* data)  |
  +------------+------------+       +-------------------------+
               |
               v
  +-------------------------+
  | Construct In-Memory     |
  | Meta-Model Structure    |
  +------------+------------+
               |
               v
  +-------------------------+
  | Compute RMAV Attributes |
  | & Multi-Seed Simulation |
  +------------+------------+
               |
               +-----------------------+-----------------------+
               |                       |                       |
               v                       v                       v
        [ Max Risk < High ]    [ High Risk Warnings ]  [ CRITICAL / HIGH SPOF ]
               |                       |                       |
               v                       v                       v
  +-------------------------++-------------------------++-------------------------+
  |   Return Exit Code 0    ||   Return Exit Code 1    ||   Return Exit Code 2    |
  |     (Build Passes)      ||  (Passes with Warning)  ||     (Build Blocks)      |
  +-------------------------++-------------------------++-------------------------+

```

### 6.1 Gating Automation Pipeline Execution

The quality gate automatically intercepts architecture modification commits or pull requests, parsing updated AaC descriptors to construct a counterfactual topology model. The gate evaluates the system graph against regression standards, terminating execution using explicit shell return codes:

* **Exit Code 0:** Clean structural configuration; architectural variations stay within acceptable quality baseline bounds; deployment proceeds.
* **Exit Code 1:** Medium-severity architectural smells detected (e.g., chatty component pairs, un-throttled best-effort feeds); build passes with warning logs.
* **Exit Code 2:** Severe structural anomalies discovered (e.g., new un-replicated single points of failure, structural loops, cyclic dependency extensions, or critical library fan-out vulnerabilities); **build is blocked**, preventing production delivery.

### 6.2 Decoupled In-Memory Repository Architecture

Standard graph analysis pipelines rely on full graph database installations (e.g., Neo4j via continuous Bolt driver transactions), which introduce significant connection and query latencies that make them unsuitable for rapid build gates. To address this constraint, SaG implements a specialized, thread-safe `MemoryRepository` layer. This structure isolates network dependencies and maintains graph data entirely in volatile memory, enabling concurrent failure simulations and rapid structural regression testing directly within transient CI/CD execution environments.

---

## 7. Empirical Evaluation and Quality Patterns

### 7.1 Scenario Benchmarks Suite

We evaluate framework performance across seven synthetic middleware system profiles designed to emulate diverse large-scale production topologies:

* **Autonomous Vehicle Telemetry System:** Tight real-time constraints, high volume, Best-Effort sensor feeds.
* **High-Frequency Financial Trading Environment:** Dominated by low-latency Reliable streams with strict message persistence.
* **Clinical Healthcare Integration Hub:** High maintainability debt, large data-isolation requirements.
* **Centralized Hub-and-Spoke Enterprise Core:** Large routing hubs, clear structural single points of failure.
* **Distributed IoT Smart-City Mesh:** Highly decentralized topology with mixed best-effort configurations.
* **Cloud-Native Microservices Mesh:** Dynamic path definitions, large cluster sizes.
* **Hyper-Scale Enterprise Pub-Sub:** Extra-large system stress profile containing highly complex structural dependency lines.

All evaluations run across 5 distinct seed variations over an execution threshold sweep to filter out seed noise.

### 7.2 RQ1: Predictive Capability and Transfer Performance

To assess ranking accuracy, we measure the Spearman rank correlation coefficient ($\rho$) against simulated failure impact indices across two separate validation protocols.

Under **in-distribution evaluation**, the deterministic RMAV score aligns closely with simulated outcomes, achieving strong correlation performance ($\rho > 0.87, F1 > 0.90$) and matching or exceeding the in-distribution training performance of the deep learning model ($\rho \approx 0.62$). This demonstrates that for system variations within a known domain, the interpretable, deterministic RMAV composite provides robust predictive power without training overhead.

However, under **inductive cross-validation**—using a strict Leave-One-Scenario-Out (LOSO) protocol to evaluate performance on completely unseen architectures—the non-learning baselines degrade, while the typed, QoS-aware learned models remain robust:

| Predictive Model Architecture | Mean Spearman $\rho$ (LOSO) | Std $\rho$ | F1 @ Top-K | $\Delta\rho$ vs Homogeneous Baseline |
| --- | --- | --- | --- | --- |
| Homogeneous GAT (GL Baseline) | 0.021 | 0.142 | 0.209 | — |
| Homogeneous GAT + QoS Attributes | 0.002 | 0.095 | 0.201 | — |
| Heterogeneous Graph (HGL Typed) | 0.307 | 0.271 | 0.390 | +0.286 |
| **QoS-Aware Heterogeneous (HGL-QoS)** | **0.401** | **0.367** | **0.433** | **+0.380** |

This highlights an important architectural constraint: homogeneous graphs fail to generalize to unfamiliar topologies ($\rho \approx 0$), whereas typed, relation-aware heterogeneous architectures effectively transfer learned structural risk across distinct domain domains.

### 7.3 RQ2: The Mandate for Stratified Metrics reporting

A key methodological finding is the presence of a **Simpson’s Paradox** effect during unstratified architecture analysis. If correlation coefficients are calculated across all components simultaneously, the pooled performance drops to $\rho \approx 0.08$.

This drop is a mathematical artifact of mixing heterogeneous node types rather than a failure of predictive accuracy. Because brokers and topics cluster around specific availability parameters, while applications and code libraries reside in separate structural dimensions, pooling them flattens their distinct distributions. When analyzed within specific structural types, correlations remain highly robust ($\rho \in [0.63, 0.90]$). Consequently, unstratified global reporting is inherently misleading; architectural quality metrics must be stratified by node type to maintain validation accuracy.

### 7.4 RQ4: CI/CD Pipeline Gate Execution Footprint

We benchmark the execution latency of the quality gate script (`detect_antipatterns.py`) using the volatile `MemoryRepository` across multiple system scales to assess its viability for continuous integration pipelines:

```
  Execution Time (Seconds)
    500 |                                                  
    100 |                                                  
     50 |                                            [~40s]
     20 |                                                  
     12 |                                    [~12s]        
      5 |                            [~5s]                 
      2 |                    [<2s]                         
      0 +----------------------+-------+-------+-------+---
                            Small   Medium   Large   XLarge
                                 Topology Scale Profile

```

The evaluation demonstrates sub-quadratic scaling across all benchmark configurations. By performing structural analysis and failure simulations entirely in volatile memory, SaG avoids external database connection overhead, completing checks well within standard CI/CD time constraints. Gating validation tests confirm a **100% detection rate (precision = 1.0, recall = 1.0)** for injected architectural regressions (e.g., single points of failure, structural loops), with zero false positives on clean base designs, providing a dependable build-breaking validation gate.

---

## 8. Real-World Case Study: Air Traffic Management System

To confirm that our static metric models correspond with empirical human engineering judgment, we perform an external validation study on an operational, safety-critical air-traffic-management (ATM) system compliant with ICAO structural standards.

### 8.1 System Architecture Scope

The evaluated ATM architecture processes radar target tracks through automated safety-separation pipelines before displaying them at a Controller Working Position (CWP). Conforming to MIL-STD-498 structural layouts, the system aggregates into three primary Software Configuration Items (CSCIs):

1. **Surveillance CSCI:** Houses raw radar trackers publishing foundational target streams over localized ASTERIX message brokers.
2. **Conflict Management CSCI:** Executes real-time separation assurance, consuming track streams to compute proximity alert vectors under tight 100-ms deadline contracts.
3. **Controller Working Position (CWP) CSCI:** Consumes processed safety vectors, mapping visual metrics onto human workstation consoles.

### 8.2 Blind Expert Elicitation Protocol

We convene a panel of domain safety and air-traffic-control systems engineers to establish an empirical baseline for comparison. Panel members are kept blind to the framework's output and to each other's assessments. Each engineer independently ranks the ATM sub-components based on operational criticality, identifying which assets require hardening priority.

We evaluate framework alignment with this expert group using two metrics:

* **Fleiss’ Kappa ($\kappa$):** Measures inter-rater agreement among the experts to ensure the human baseline is consistent and valid.
* **Kendall’s Tau ($\tau$):** Measures the rank correlation between the framework's $Q(v)$ or HGL outputs and the aggregated expert consensus.

The framework's performance is validated if inter-rater agreement is high and the SaG rank correlation significantly outpaces standard untyped centrality benchmarks.

---

## 9. Threats to Validity

### 9.1 Construct Validity

The primary internal risk to construct validity is the reliance on simulated failure cascades rather than operational production crash data to define our ground-truth labels. Consequently, our findings demonstrate strong alignment with the simulator's propagation logic; they do not guarantee absolute operational accuracy in live environments. We mitigate this by applying identical simulation baselines consistently across all competing predictors, ensuring that comparative model performance remains valid.

### 9.2 Internal Validity

Internal validity requires that predictive performance reflects genuine structural signal rather than data leakage from the validation metrics. We enforce this through our strict separation guarantee: all predictive models operate on the derived projection graph ($G_{\text{analysis}}$) without visibility into the simulation steps executed on the raw structural layer ($G_{\text{structural}}$). No operational simulation output is ever fed back into the metric formulas or the graph neural network feature nodes.

### 9.3 External Validity

While our synthetic benchmark suite spans seven distinct domains and multiple scales, these networks are generated systematically rather than collected from live web clusters. We address this limitation via two mechanisms: using Leave-One-Scenario-Out (LOSO) cross-validation to prevent transductive memorization, and incorporating the blind air-traffic-management case study to anchor our findings in an operational, safety-critical environment.

---

## 10. Conclusion

This paper introduced Software-as-a-Graph (SaG), a static system analysis framework designed to bridge the Architecture-Code Gap in publish–subscribe middleware systems. By mapping software configurations into a typed, weighted, directed multigraph, SaG integrates local static code analysis attributes with global topological dependencies. We formalized the multi-dimensional RMAV metrics framework, providing an interpretable, actionable assessment model that avoids the dimensional collapse of standard untyped centralities.

Our evaluation demonstrates that while deterministic multi-criteria scoring matches or exceeds learned models within known configurations, relation-aware heterogeneous graph transformations are necessary to transfer risk patterns across unseen topologies. We exposed critical system failure patterns—notably the shared-library blast-radius gap and the Simpson’s paradox effect—confirming that software quality analysis must be both type-aware and stratified. Finally, by implementing the analyzer via a decoupled, volatile in-memory architecture, we demonstrate that SaG can run as a blocking build check in rapid delivery pipelines, intercepting structural regressions and enforcing systemic quality standards before software deployment.

---

## References

* `` Extracted from pub-sub foundational decoupled system references.
* `` Standard middleware QoS parameter references (DDS/MQTT specification guidelines).
* `` Network metrics modeling baseline guidelines.
* `` Multi-criteria decision tracking standards using the Analytic Hierarchy Process (AHP).
* `` Heterogeneous Graph Transformer neural network model foundations.
*(Full bibliographic references to be generated by the final LaTeX compiler framework package).*