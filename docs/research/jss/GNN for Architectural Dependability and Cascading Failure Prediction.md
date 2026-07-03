# Graph Neural Networks for Architectural Dependability and Cascading Failure Prediction in Complex Distributed Systems

*Target Venue: Journal of Systems and Software (JSS) — Special Issue on AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems (VSI:AI4MSS)*

---

## Abstract

Modern complex distributed systems increasingly rely on publish–subscribe and microservice middleware to decouple data producers and consumers. While this decoupling provides significant scaling and operational flexibility, it obscures the logical dependency chains along which structural component failures cascade. Identifying system-level critical bottlenecks before deployment remains a challenge: operational runtime telemetry does not yet exist during the design phase, and standard static code analysis frameworks are blind to inter-component topologies.

To bridge this "Architecture-Code Gap," we present **Software-as-a-Graph (SaG)**, a pre-deployment **Static System Analysis (SSA)** framework that evaluates runtime dependability entirely from structural design descriptors. SaG models distributed configurations as typed, weighted, directed multigraphs across five node classifications (applications, brokers, topics, infrastructure nodes, and shared libraries) and derives logical failure propagation paths via multi-layer relational projection rules.

On this semantic representation, we employ a **Heterogeneous Graph Neural Network (GNN)** predictor—specifically a parameterized Heterogeneous Graph Transformer with explicit Quality-of-Service (QoS) contract edge-feature injection ($HGL\text{-}QoS$)—to forecast each component's cascading failure impact $I(v)$ before a single line of code is deployed. The network’s predictions are validated against an independent discrete-event cascade simulator under a strict input–label independence guarantee to ensure zero transductive leakage. Additionally, we pair this learned predictor with an interpretable, Analytic Hierarchy Process (AHP)-weighted multi-dimensional quality attribution score ($Q(v)$) decomposing failure mechanics into orthogonal Reliability, Maintainability, Availability, and Vulnerability (RMAV) dimensions.

Evaluating the framework across seven synthetic, hyper-scaled distributed topologies encompassing diverse industrial domains, we show that:

1. In-distribution regimes are strongly modeled by deterministic attribution ($\rho > 0.87$, $F_1 > 0.90$), whereas out-of-distribution generalization to completely unseen architectures requires typed graph learning, where our $HGL\text{-}QoS$ model decisively outperforms homogeneous configurations, achieving a Leave-One-Scenario-Out (LOSO) cross-validation score of $\rho = 0.401$.
2. A stratified correlation assessment confirms that the predictive capabilities of the GNN are highly consistent across all five node categories (with within-type Spearman correlations ranging from $\rho = 0.322$ to $0.429$), eliminating risks of a single pooled aggregate masking type-specific predictive failure.
3. We report a comprehensive, multi-seed evaluation of topology-level prescriptive remediation operators, documenting an overall cross-scenario mean impact reduction of $+4.61\%$, while honestly analyzing why an unfiltered policy implementation introduces counterproductive mutations in 3 out of 7 scenarios.

Finally, we demonstrate the operational feasibility of SaG by instantiating it as a delta-aware, blocking CI/CD quality gate. Utilizing a thread-safe, database-free `MemoryRepository` to eliminate live graph database latencies, the analyzer evaluates complex regressions under $5\text{ s}$ for medium and $40\text{ s}$ for hyper-scale topologies while achieving a $100\%$ detection rate ($precision=1.0, recall=1.0$) on newly introduced structural vulnerabilities, successfully driving automated, continuous dependability auditing.

**Keywords:** publish–subscribe middleware; architectural dependability; cascading failure; heterogeneous graph neural networks; pre-deployment verification; quality attributes; CI/CD quality gate.

---

## 1. Introduction

### 1.1 Motivation

The publish–subscribe (pub-sub) paradigm has become a backbone communication abstraction for large-scale distributed systems, underpinning cyber-physical, cloud-native, robotics, and Internet-of-Things architectures. Its primary appeal lies in decoupling: producers and consumers are separated in time, space, and synchronization, allowing components to scale or exit dynamically without direct knowledge of one another. Industry standards such as the Data Distribution Service (DDS) and MQTT formalize this model and expose deployment-time configuration choices—topics, brokers, reliability, durability, and transport priorities—that materially shape how a system behaves under stress.

However, the same decoupling that makes pub-sub architectures highly flexible also obscures the implicit dependency structures an engineer must reason about when a component fails. There are no explicit caller–callee edges: an application that publishes to a topic has no static link to the downstream applications subscribing to it, even though those subscribers are entirely dependent on it for functional data. Failures propagate not along an explicit call graph, but along *derived logical paths*—through shared topics, intermediate message-routing brokers, colocated physical hardware hosts, and distinctively, through shared software libraries whose baseline vulnerability strikes all consumer applications simultaneously rather than sequentially.

Crucially, the lifecycle stage at which reasoning about these dependencies is most valuable is *before deployment*. Architectural hardening—replication, transport QoS contract tightening, alternative routing, and physical isolation—is least disruptive and lowest cost while the system is still a design concept. Once deployed into clinical, industrial, or high-frequency operational environments, retrospective architectural fixes become prohibitively expensive. Yet pre-deployment is precisely the phase where no runtime telemetry or log streams exist to identify weak points empirically. Software engineers must therefore answer a critical question from the declarative architecture alone: *which components are highly vulnerable to failure propagation, and why?*

### 1.2 The Architecture-Code Gap and Problem Statement

We address pre-deployment dependability analysis for distributed middleware as two coupled sub-problems. Given only an architectural description of the system—its applications, libraries, topics, brokers, physical deployment nodes, and the QoS policies governing its interfaces—we seek to:

1. **Quality Attribution:** Assign each structural component an interpretable measure of *how* and *why* it concentrations risk, decomposed along clear, actionable quality dimensions (Reliability, Maintainability, Availability, Vulnerability) so that the diagnostic output directs a specific engineering remediation rather than a generic warning.
2. **Failure-Impact Analysis:** Predict the precise cascading impact of each component's failure—the extent to which the rest of the topology becomes unreachable, partitioned, or functionally impaired—and identify the precise order in which components must be hardened.

Both objectives must be achieved without executing the system, and both must remain *explainable*: a single opaque score provides little guidance to an architect deciding between competing interventions under a constrained budget.

Historically, static verification has operated almost entirely at the source-code level. However, a severe **"Architecture-Code Gap"** exists: a software system can possess perfectly clean source code within every individual application module (earning top ratings on legacy tools), yet remain catastrophically fragile. If the deployment topology contains an unmitigated Single Point of Failure (SPOF) or an incompatible QoS contract, a single component crash can cascade and collapse the entire system. Bridging this gap requires shifting structural, topology-level dependability verification "left" directly into continuous integration and delivery (CI/CD) pipelines.

### 1.3 Limitations of Existing Approaches

Prior work addressing distributed system verification broadly falls into three categories, each leaving an unaddressed gap:

* **Static Code Analysis (SCA):** Frameworks such as SonarQube evaluate code cleanliness, cyclomatic complexity, and coupling metrics within individual software modules. While effective for local debt tracking, they are entirely blind to inter-component network topologies and dynamic middleware failure cascades.
* **Chaos Engineering and Runtime Testing:** A mature body of work hardens distributed architectures via staging-phase fault injection (e.g., Chaos Monkey). These approaches are valuable but assume a *fully running, deployed cluster*; they cannot evaluate an architectural layout before deployment, and physical runtime exploration carries significant operational risk.
* **Topology-Only Centrality and Homogeneous GNNs:** Classical network centralities collapse a component's risk into a scalar that conflates distinct failure mechanisms (e.g., localized code complexity vs. graph partitioning). Similarly, standard homogeneous graph neural networks collapse typed semantics (applications, topics, brokers, libraries) into flattened views, leading to representation collapse in highly dense, multi-layer topologies.

No existing methodology provides an *interpretable, multi-dimensional, pre-deployment* attribution over a *typed* publish–subscribe multigraph, integrated with local code metrics, relational graph transformers, and automated CI/CD quality gating. This is the limitation this paper overrides.

### 1.4 Our Approach and Document Organization

We present **Software-as-a-Graph (SaG)**, a pre-deployment **Static System Analysis (SSA)** framework. SaG maps distributed middleware layouts into a typed, weighted, directed multigraph over five distinct node classifications and extracts implicit dependencies via a robust set of relational projection rules. SaG ingests local code-level SCA metrics to enrich component vertices, performing a deterministic, Analytic Hierarchy Process (AHP)-weighted **multi-dimensional quality attribution** that groups metrics into orthogonal Reliability, Maintainability, Availability, and Vulnerability (RMAV) arrays.

Concurrently, SaG executes **failure-impact analysis** using a dual-predictive approach. We evaluate the ranking capacity of the interpretable RMAV composite against a highly parameterized **Heterogeneous Graph Transformer (HGT)** that injects explicit QoS contract features onto relational edge channels. Both predictors are validated against a discrete-event cascade simulation engine under an absolute **input–label independence guarantee** to prevent circular data leakage. Finally, SaG uses these diagnostics to generate topology-level prescriptive remediation edits and serves as a delta-aware, build-blocking quality gate within continuous release pipelines.

The paper is structured around four foundational Research Questions:

* **RQ1:** When does interpretable, multi-dimensional quality attribution suffice for pre-deployment criticality prediction, and when is a learned heterogeneous graph transformer required to recover simulated failure cascades?
* **RQ2:** What systemic fault modes does multi-dimensional attribution expose that single-score topological centralities miss when evaluating typed components?
* **RQ3:** How does explicit multi-attribute QoS contract feature injection affect GNN in-distribution convergence versus out-of-distribution Leave-One-Scenario-Out (LOSO) generalizability?
* **RQ4:** What is the operational feasibility and performance overhead of deploying a graph-based dependability analyzer as a blocking Quality Gate in rapid release CI/CD pipelines?

---

## 2. Related Work

### 2.1 Publish–Subscribe Dependability & Protocol Resilience

The publish–subscribe communication topology is valued for its capacity to isolate message endpoints across temporal and spatial domains. To maintain high dependability, protocol specifications like DDS and MQTT introduce intricate Quality-of-Service constraints. Prior literature on middleware dependability concentrates heavily on protocol-level resilience, secure event routing networks, redundant broker overlays, and dynamic runtime recovery algorithms. These techniques are reactive or masking options operating during system execution. Our framework operates prior to execution, parsing declarative configuration code to find structural vulnerabilities before deployment.

### 2.2 Static Code Analysis (SCA) vs. Static System Analysis (SSA)

Traditional static analysis processes code structures locally. Tools like SonarQube inspect Abstract Syntax Trees (ASTs) within a single repository to evaluate cyclomatic complexity, Lack of Cohesion of Methods (LCOM), or object coupling. While essential for component maintenance, local code analysis cannot reason about the global architecture. Static System Analysis (SSA) fills this gap by shifting the model focus upward. SSA parses "Architecture-as-Code" (AaC) files to reconstruct the global system graph, treating local code metrics not as independent values, but as vertex attributes that propagate through the network topology.

### 2.3 Graph Neural Networks on Heterogeneous Software Topologies

Identifying high-impact nodes within a network graph is a core capability of graph neural networks (GNNs). Models like FINDER or DrBC successfully learn to approximate betweenness centralities or identify critical node groups from graph layouts. However, these approaches are optimized for *homogeneous networks*. Distributed middleware applications present an inherently *heterogeneous network structure*, where edges carry entirely different meanings (e.g., an application utilizing a library vs. an application hosting on a physical node). To prevent representation collapse, modern architectures apply relation-specific aggregations. Models like Relational Graph Convolutional Networks (RGCN) and Heterogeneous Graph Transformers (HGT) maintain unique parameter allocations per node-edge combination, allowing deep multi-hop pattern discovery without oversmoothing highly dense clusters.

---

## 3. The Software-as-a-Graph Modeling Formalism

### 3.1 Nodes, Edges, and the Formal Multigraph

A distributed system configuration is represented formally as a typed, weighted, directed multigraph:

$$G = (V, E, \tau_V, \tau_E, w_E, w_V)$$

The global vertex set $V$ partitions into five disjoint component classifications based on their runtime execution boundaries:

$$V = V_{\text{app}} \cup V_{\text{broker}} \cup V_{\text{topic}} \cup V_{\text{node}} \cup V_{\text{lib}}$$

Where the node types map to specific architectural roles:

* $V_{\text{app}}$: Executable processes or microservices that publish or consume messages.
* $V_{\text{broker}}$: Intermediate middleware routing engines (e.g., a Kafka or MQTT broker).
* $V_{\text{topic}}$: Named, isolated message channels routing event payloads.
* $V_{\text{node}}$: Compute infrastructure platforms hosting the software (VMs, containers, bare-metal servers).
* $V_{\text{lib}}$: Shared codebase modules or third-party binary libraries imported by applications.

The type functions $\tau_V : V \to \{\text{App}, \text{Broker}, \text{Topic}, \text{Node}, \text{Library}\}$ and $\tau_E : E \to \{\text{PUBLISHES\_TO}, \text{SUBSCRIBES\_TO}, \text{ROUTES}, \text{RUNS\_ON}, \text{CONNECTS\_TO}, \text{USES}\}$ label every node and edge instance within the graph configuration.

### 3.2 QoS-Driven Edge and Vertex Weighting Formulas

To ensure the graph model captures real coupling semantics, edge weights are calculated from declarative middleware QoS contracts. A continuous `QoS_score` is computed via an AHP-derived combination of reliability ($r$), durability ($d$), and transport priority ($p$) parameters:

$$\text{QoS\_score} = 0.30\,r + 0.40\,d + 0.30\,p$$

The symbolic properties map to standard values based on their contract strength:

* Reliability ($r$): `RELIABLE` $\to 1.0$, `BEST_EFFORT` $\to 0.0$.
* Durability ($d$): `PERSISTENT` $\to 1.0$, `TRANSIENT` $\to 0.6$, `TRANSIENT_LOCAL` $\to 0.5$, `VOLATILE` $\to 0.0$.
* Priority ($p$): `CRITICAL`/`URGENT` $\to 1.0$, `HIGH` $\to 0.66$, `MEDIUM` $\to 0.33$, `LOW` $\to 0.0$.

Payload message volume is normalized using a logarithmic scale proxy:

$$\text{size\_norm} = \min\!\left(\frac{\log_2(1 + \text{size\_kb})}{50},\ 1.0\right)$$

The final directional edge weight $w(e) \in [0,1]$ blends these scores with a baseline floor of $0.01$ to prevent topological disconnection:

$$w(e) = \max\left(\beta\cdot\text{QoS\_score} + (1-\beta)\cdot\text{size\_norm},\ 0.01\right), \qquad \beta = 0.85$$

Vertex weights $w_V(v)$ reflect the concentration of risk across component types:

* **Applications:** $w_V(v) = 0.80\cdot\max(w_{\text{topic}}) + 0.20\cdot\operatorname{mean}(w_{\text{topic}})$.
* **Brokers:** $w_V(v) = 0.70\cdot\max(w_{\text{topic}}) + 0.30\cdot\operatorname{mean}(w_{\text{topic}})$.
* **Infrastructure Nodes:** $w_V(v) = \max(w(e))$ over all hosted apps and brokers.
* **Shared Libraries:** $w_V(v) = \min\!\big(1.0,\ w_{\text{base}}\cdot(1 + \gamma\log_2(1 + \mathrm{DG\_in}))\big)$.

### 3.3 The Logical Projection Rules (`DEPENDS_ON`)

Structural edges denote declarative setup boundaries, but fail to record logical failure paths. SaG derives a singular, highly uniform dependency edge—`DEPENDS_ON` ($e_{\text{dep}}$)—directed from the dependent component to the component it relies on, satisfying the contract: *“if the target fails, the source is immediately affected”*. This is achieved via six deterministic type projections:

1. **Application to Application:** Derived whenever an application subscribes to a topic published to by another application (including transitive dependencies spanning up to three levels). The edge attributes are consolidated as:

$$\text{edge.weight} = \max_{t \in \text{shared}} w(t), \qquad \text{edge.path\_count} = |\text{shared}|$$


2. **Application to Broker:** Mapped from any publisher or subscriber application vertex to the intermediate broker routing its targets.
3. **Node to Node:** Lifted from underlying application communication dependencies to record physical compute host reliance.
4. **Node to Broker:** Mapped from an infrastructure host to the broker engine handling its data interfaces.
5. **Application to Library (Shared-Library Blast):** Extracted from code-import structures, mapping an application to its shared library dependencies. The dependency weight matches the application's global vertex weight: $w(e_{\text{dep}}) = w_V(\text{app})$.
6. **Broker to Broker (Colocation):** Injected as a bidirectional dependency edge whenever two separate broker services share the same underlying physical host node.

---

## 4. Multi-Dimensional Quality Attribution (The Interpretable Path)

### 4.1 Four Orthogonal Dimensions

To maintain complete diagnostic transparency, SaG avoids collapsing architectural metrics into an ambiguous single value. Instead, the framework decomposes criticality into four orthogonal quality dimensions, ensuring that each structural and code-level metric maps to exactly one dimension:

* **Reliability ($R(v)$):** Evaluates fault-propagation risk, tracking the depth and scale of failure cascades across multi-hop communication paths.
* **Maintainability ($M(v)$):** Measures structural coupling complexity and architectural tech debt, pinpointing highly complex components that act as maintainability bottlenecks.
* **Availability ($A(v)$):** Tracks graph partitioning and single-point-of-failure vulnerabilities, highlighting structural cut-vertices whose failure divides the network.
* **Vulnerability ($V(v)$):** Models adversarial reachability, tracking exposure curves from public-facing boundaries to critical internal assets.

### 4.2 Analytical Formulations

Every metric input is rank-normalized relative to the system population into a clear $[0,1]$ scale, forcing all resultant dimension values into an identical range.

#### Reliability ($R(v)$)

Computed on the transpose graph view $G^\top$ to model failure cascading against the natural direction of logical dependency paths. For standard non-topic nodes:

$$R(v) = 0.45\cdot\mathrm{RPR}(v) + 0.30\cdot\mathrm{DG\_in}(v) + 0.25\cdot\mathrm{CDPot\_enh}(v)$$

$$\mathrm{CDPot\_enh}(v) = \min\!\Big( \frac{\mathrm{RPR}(v) + \mathrm{DG\_in}(v)}{2} \cdot \big(1 - \min(\tfrac{\mathrm{out\_degree\_raw}(v)}{\max(\mathrm{in\_degree\_raw}(v),\, \epsilon)}, 1)\big) \cdot (1 + \mathrm{MPCI}(v)),\ 1.0 \Big)$$

Where $\mathrm{RPR}$ is the Relational Propagation Reach, $\mathrm{DG\_in}$ is the normalized in-degree, and $\mathrm{MPCI}$ tracks Multi-Path Coupling Indices. For Topic vertices, a fan-out optimization is dispatched:

$$R_{\text{topic}}(v) = 0.50\cdot\mathrm{FOC}(v) + 0.50\cdot\mathrm{CDPot\_topic}(v)$$

$$\mathrm{CDPot\_topic}(v) = \mathrm{FOC}(v)\big(1 - \min(\text{publisher\_count\_norm}(v),1)\big)$$

Where $\mathrm{FOC}$ tracks the Topic's absolute Subscriber Fan-Out Count.

#### Maintainability ($M(v)$)

Combines system-level topology properties with modular source code attributes:

$$M(v) = 0.35\,\mathrm{BT}(v) + 0.30\,\mathrm{w\_out}(v) + 0.15\,\mathrm{CQP}(v) + 0.12\,\mathrm{CouplingRisk\_enh}(v) + 0.08\,(1-\mathrm{CC}(v))$$

Where $\mathrm{BT}$ is metric betweenness centrality and $\mathrm{CC}$ is the clustering coefficient. Local code debt is penalised via the Code Quality Penalty ($\mathrm{CQP}$), parsing ingested SonarQube properties (`cm_*`):

$$\mathrm{CQP}(v) = 0.10\,\text{loc\_norm} + 0.35\,\text{complexity\_norm} + 0.30\,\text{instability\_code} + 0.25\,\text{lcom\_norm}$$

#### Availability ($A(v)$)

Highlights structural single points of failure across the global architecture:

$$A(v) = 0.35\,\mathrm{AP\_c\_directed}(v) + 0.25\,\text{QSPOF}(v) + 0.25\,\mathrm{BR}(v) + 0.10\,\mathrm{CDI}(v) + 0.05\,w(v)$$

Where $\mathrm{AP\_c\_directed}$ evaluates a custom directed articulation point search, $\text{QSPOF}$ scales the articulation score against the component's edge QoS weight, $\mathrm{BR}$ captures the immediate Blast Radius, and $\mathrm{CDI}$ is the Centralized Disruption Index.

#### Vulnerability ($V(v)$)

Models security reach profiles and structural asset targeting:

$$V(v) = 0.40\,\mathrm{REV}(v) + 0.35\,\mathrm{RCL}(v) + 0.25\,\mathrm{w\_in}(v)$$

Where $\mathrm{REV}$ evaluates Reverse Reachable asset sizes and $\mathrm{RCL}$ computes Reachable Closest Likelihood metrics.

### 4.3 Composite Score $Q(v)$ and AHP Pairwise Matrices

The final composite quality score $Q(v)$ linearly combines the four dimensional assessments:

$$Q(v) = w_A\,A(v) + w_R\,R(v) + w_M\,M(v) + w_V\,V(v)$$

The underlying weight choices are extracted via Saaty's Analytic Hierarchy Process (AHP), utilizing a pairwise comparison matrix evaluated for mathematical consistency. The underlying comparison configuration passes a strict validation check, maintaining a consistency ratio $\mathrm{CR} \approx 0.02$, well below the standard $0.10$ boundary. The raw AHP calculation allocates weights as follows:

$$(w_A, w_R, w_M, w_V) = (0.43,\ 0.24,\ 0.17,\ 0.16)$$

To protect against mathematical over-fitting in smaller comparison spaces, SaG applies a shrinkage filter toward a uniform prior, governed by a calibration coefficient $\lambda = 0.70$:

$$w_{\text{final}}(d) = \lambda\,w_{\mathrm{AHP}}(d) + (1-\lambda)\,\tfrac{1}{n_{\text{dim}}}$$

Blended with the uniform prior, the final composite weights applied by the diagnostic execution engine settle at:

$$(w_A, w_R, w_M, w_V) = (0.38,\ 0.24,\ 0.19,\ 0.19)$$

---

## 5. Heterogeneous GNN for Failure-Impact Forecasting (The Learned Path)

### 5.1 Defining the Ground-Truth Simulation Label $I(v)$

To evaluate the true predictive capacity of both the deterministic $Q(v)$ composite and the learned deep models, SaG implements a discrete-event failure simulator. Operating on the raw, un-projected structural graph view $G_{\text{structural}}$, the simulator injects a failure at component $v$, propagates the fault step-by-step through communication interfaces over a 10-epoch horizon, and calculates a continuous ground-truth label $I(v) \in [0,1]$:

$$I(v) = 0.35\,\text{reachability\_loss} + 0.25\,\text{fragmentation} + 0.25\,\text{throughput\_loss} + 0.15\,\text{flow\_disruption}$$

A cascading failure triggers if a component’s average upstream message feed loss clears an explicit `propagation_threshold` fixed at $0.2$. Because step tie-breaking is stochastic, the final label $I(v)$ represents the mean value computed across five separate evaluation seeds: $\{42, 123, 456, 789, 2024\}$.

### 5.2 Heterogeneous Graph Transformer ($HGL\text{-}QoS$) Architecture

While the linear $Q(v)$ score provides clear explanation parameters, it cannot capture higher-order, multi-hop structural interdependencies across typed subgraphs. To handle these complex topologies, SaG introduces a **Heterogeneous Graph Transformer (HGT)**. The network utilizes relation-specific attention matrices to calculate distinct message transformations across the five node classifications and six edge structures.

Crucially, the $HGL\text{-}QoS$ variant extends standard architectures by injecting continuous multi-attribute QoS weights ($r, d, p$) directly into the edge attention aggregation layers, forcing the graph convolutions to scale message magnitude by interface contract strength. The model is trained inductively using the simulation-derived labels $I(v)$ as the objective minimization target.

### 5.3 Enforcing the Input–Label Independence Guarantee

To maintain scientific validity, SaG enforces a strict structural separation between feature spaces and label propagation. The predictors consume metrics extracted entirely from the logical projection layer $G_{\text{analysis}}$, whereas the simulation engine executes drops directly on the structural setup view $G_{\text{structural}}$. No simulation parameters—reachability loss, fragmentation scales, or path disruptions—are ever passed as input vectors to the GNN or the $Q(v)$ pipeline. This absolute boundary ensures that a high correlation score represents genuine predictive mapping rather than historical label leakage.

---

## 6. Continuous Operationalization: CI/CD Quality Gating & Prescription

### 6.1 Gating Semantics and Git Merge-Base Differencing

To prevent architectural degradation over time, SaG operationalises its system analysis as an automated, blocking Quality Gate within developer release workflows. Executed on every pull request via `detect_antipatterns.py`, the analyzer implements strict **delta-aware semantics**. The tool automatically computes the system graph for both the incoming branch and its target branch git merge-base, blocking deployment *only if the newly introduced commit introduces new, un-waived structural anomalies*.

This delta structure is essential for large systems: production architectures frequently feature legacy, risk-accepted anomalies (e.g., an un-replicated legacy database node) that cannot be altered. An absolute gate would flag these elements on every build, leading to developer fatigue and gate bypass. Pre-existing risks are carried in a baseline register, while intentional new risks must be added to an auditable **waiver register** detailing component IDs and explicit expiration deadlines.

### 6.2 Posix Exit-Code Protocols

The gate tracks the resulting architectural delta and outputs standard POSIX exit codes to control deployment orchestration pipelines:

* **Exit Code 0 (Success):** No new structural regressions detected, or all newly flagged anomalies match valid waivers. Branch merge is permitted.
* **Exit Code 1 (Warning):** New lower-severity anomalies (e.g., high multi-path chatty pairs or minor QoS mismatches) are detected. Pipeline execution continues with descriptive warning logs.
* **Exit Code 2 (Failure):** New, un-waived `CRITICAL` or `HIGH` severity architectural risks are discovered (e.g., new structural single points of failure, unmitigated circular cascades, or broker overload configurations). **The build is broken, and automated deployment is blocked**.

### 6.3 Database-Free Build Execution

Graph visualization and historical query logs leverage a Neo4j database system during live runtime monitoring. However, forcing a build runner to connect to an external database via Bolt connections introduces network fragility and deployment dependencies. To eliminate this bottleneck, SaG couples the CI/CD pipeline to a thread-safe, database-free **`MemoryRepository`**. The entire system multigraph is reconstructed, projected, and simulated directly within memory, allowing local build tasks to evaluate complex architectures without database dependencies.

### 6.4 Prescriptive Remediation Operators

When components display high criticality profiles, SaG shifts from diagnosis to automated correction via a prescriptive execution service. The framework models architectural modifications using four topology-level operators:

* **`RedundancyInsertion`:** Identifies directed cut-vertices with extreme Availability scores and injects redundant component pairs or reallocates client weights to remove the SPOF boundary.
* **`PathDiversification`:** Targets non-redundant communication links, establishing alternative broker paths or redundant physical network interfaces to distribute path load.
* **`FanOutReduction`:** Fired by high structural fan-out metrics (subscriber volume for topics, consumer count for libraries) rather than the composite score $Q(v)$, interposing middleware proxies or splitting over-subscribed channels to mitigate blast exposure.
* **`SharedTopicReduction`:** Inspects tightly coupled application groups with high `path_count` properties, decoupling shared topics to isolate individual interaction scopes.

---

## 7. Experimental Setup

### 7.1 Synthetic Scenario Suite Profile

We evaluate the framework across seven synthetic publish–subscribe system configurations generated by a statistical topology generator. The topologies represent diverse enterprise use cases and span scale categories from `tiny` to `xlarge`:

| Scenario | Component Count ($|V|$) | Structural Regime | Dominant Failure Vector |
| :--- | :--- | :--- | :--- |
| **Clinical Healthcare** | Low Scale | Star Layout | Centralized Gateway SPOF |
| **Hub-and-Spoke** | Medium Scale | Clustered Tree | Middleware Router Saturation |
| **Financial Trading** | Medium Scale | Highly Dense | Low-Latency Cascading Starvation |
| **Autonomous Vehicle** | Medium Scale | Multi-Layer Mesh | Inter-Process QoS Mismatches |
| **Microservices Mesh** | Large Scale | Highly Distributed | Transitive Service Path Failure |
| **IoT Smart City** | Large Scale | Extreme Edge Fan-Out | Topic Broker Partitioning |
| **Hyper-Scale Enterprise** | Xlarge Scale | Multi-Cluster Hub | Hybrid Library Blast Exposure |

### 7.2 Baseline Ablations

The predictive evaluation pits our models against multiple structural and deep configurations to isolate structural factors:

* **RMAV / $Q$**: Our deterministic, AHP-weighted linear attribution score.
* **GL (Homogeneous Baseline)**: A standard, untyped Graph Attention Network (GAT) executed over a type-collapsed, flattened graph view.
* **GL-QoS**: The homogeneous GAT baseline augmented with edge-level QoS features.
* **HGL (Typed Baseline)**: A Heterogeneous Graph Neural Network processing relation-specific parameters without continuous QoS feature fields.
* **HGL-QoS**: Our complete model mapping node types, edge categories, and explicit continuous QoS fields.
* **Topo-BL / Topo-QoS**: Baseline structural metrics (betweenness centrality and articulation searches) processed via traditional network parameters.

---

## 8. Experimental Evaluation and Results

### 8.1 RQ1 — Interpretable Attribution versus Learned Architectures

The central RQ1 evaluation demonstrates that predictive performance is tightly bound to the deployment regime, splitting clearly along the in-distribution vs. out-of-distribution boundary.

**In-distribution regimes are strongly modeled by deterministic attribution.** When evaluated on topologies derived from the calibration domain, the non-learned RMAV/$Q$ composite aligns strongly with simulated failure labels, achieving a rank correlation $\rho > 0.87$ and a classification threshold $F_1 > 0.90$. Under identical conditions, the trained $HGL$ model achieves lower in-distribution means ($\rho \approx 0.62, F_1 \approx 0.77$). The engineering takeaway is that when assessing localized evolutionary changes on a known system layout, an architect gains no performance benefit from deep training models; the deterministic, interpretable AHP composite provides high accuracy with zero training overhead.

**Out-of-distribution architectures require typed graph learning.** The performance profile inverts when models undergo inductive Leave-One-Scenario-Out (LOSO) cross-validation, where the model must rank a held-out system domain whose structural cascade dynamics were excluded from training. Under these conditions, traditional centralities and homogeneous networks collapse, while typed learning models maintain predictive generalizability:

| Predictive Model Variant | Mean Spearman Rank Correlation ($\rho$) | Standard Deviation ($\sigma$) | Top-K Classification ($F_1$) |
| --- | --- | --- | --- |
| **GL (Homogeneous)** | 0.021 | 0.142 | 0.209 |
| **GL-QoS (Homogeneous + QoS)** | 0.002 | 0.095 | 0.201 |
| **HGL (Heterogeneous, Typed)** | 0.307 | 0.271 | 0.390 |
| **HGL-QoS (Typed + Edge QoS)** | **0.401** | **0.367** | **0.433** |

The homogeneous baselines fail to generalize ($\rho \approx 0$), proving that collapsing middleware type configurations removes the exact structural relationships needed to predict cascading propagation. Conversely, the typed model captures generalized transfer structure, with the QoS-augmented variant ($HGL\text{-}QoS$) achieving the highest predictive metric ($\rho = 0.401$).

> **RQ1 Resolution:** Interpretable multi-dimensional attribution is sufficient and superior for in-distribution system iterations; typed, QoS-aware graph transformers are required when evaluating unseen, heterogeneous architectures.

### 8.2 RQ2 — Value of Heterogeneity and Stratified Assessment

**Heterogeneity provides the dominant predictive gain.** Isolating structural factors, relation-specific message passing improves critical component identification by $\Delta F_1 = +0.284$ in-distribution and by $\Delta \rho = +0.286$ across inductive out-of-distribution splits. This establishes that preserving unique node and edge configurations prevents representation collapse in highly dense distributed systems.

**The shared-library blast mechanism maps to an honest negative result.** In Section 3.4, we hypothesized that shared libraries would demonstrate a highly specific failure mode—driving near-total system impact ($I$) while maintaining moderate structural centrality, causing a large low-$Q$/high-$I$ gap. Testing this against the canonical simulator across 165 Library vertices, **we did not find the hypothesized mismatch**. The highest library composite score reached in the suite is $Q = 0.422$, yielding a modest simulated impact $I = 0.086$. Across all scenarios, $I(v)$ never exceeds $Q(v)$, proving that the composite score remains safe and mildly conservative for this node type. Furthermore, no single library drop caused system collapse; the highest single-point drop recorded is $I = 0.320$ (an infrastructure node). We report this negative finding openly; the simultaneous blast mechanism remains topologically distinct, but this scenario suite does not contain the extreme low-$Q$/high-$I$ disconnect anticipated.

**Stratified reporting confirms cross-type consistency.** To protect our evaluation from Simpson's paradox, where conflicting subpopulation balances mask global patterns, we execute a stratified correlation check. Pooling all scenarios ($1,545$ components), the global rank correlation settles at $\rho = 0.374$. Executed independently within isolated node populations, the correlations remain highly consistent:

$$\text{Broker: } \rho = 0.429 \ (n=36); \quad \text{InfraNode: } \rho = 0.409 \ (n=119); \quad \text{Library: } \rho = 0.351 \ (n=165);$$

$$\text{Application: } \rho = 0.346 \ (n=850); \quad \text{Topic: } \rho = 0.322 \ (n=375) \qquad \text{[all significant at } p < 0.01\text{]} $$

The pooled figure sits directly inside the type-stratified range ($0.322$–$0.429$), confirming that $Q(v)$'s predictive capacity is stable across all architectural roles and is not driven by success on a single category.

### 8.3 RQ3 — QoS Feature Embedding Ablations

Lacing explicit QoS fields onto edge features yields an interesting trade-off between calibration constraints and generalizability. During in-distribution evaluation, adding QoS fields produces a minor null result, slightly reducing ranking precision. This occurs because within a static system topology, the lifted logical dependency edges already encode QoS properties implicitly, meaning extra attributes increase model parameters and optimization noise. However, during inductive LOSO generalization across entirely separate topologies, the QoS channel acts as the primary transfer vector, driving performance from $\rho = 0.307$ ($HGL$) to $\rho = 0.401$ ($HGL\text{-}QoS$). This is because QoS parameters operate on a standard, universal scale that transfers across separate domains, whereas raw topology matrices are deeply system-specific.

### 8.4 RQ4 — Feasibility, Execution Overheads, and Prescriptive Outcomes

**The software framework achieves sub-quadratic execution footprints.** To ensure thatcontinuous Static System Analysis is viable within fast DevOps release cycles, we execute performance benchmarks using the database-free `MemoryRepository` on standard pipeline runner hardware. Execution time scales efficiently with topology size:

* **Tiny / Small Scales ($\le 25$ components):** $< 2$ seconds total execution.
* **Medium Scale ($\approx 50$ components, e.g., Autonomous Vehicle):** $\approx 5$ seconds.
* **Large Scale ($80$–$100$ components):** $\approx 12$ seconds.
* **Xlarge Scale ($150$–$300$ components, e.g., Enterprise Stress):** $\approx 40$ seconds (heavily dominated by the canvas rendering cost of the Cytoscape graph layout export).

These timelines easily clear the standard multi-minute budgets allocated for continuous integration tasks. Evaluating gating precision, we manually injected regression anomalies (unprotected SPOFs, cyclic loops, QoS mismatches) into baseline scenarios. The delta-aware gate achieved a **100% detection and blocking rate (precision = 1.0, recall = 1.0)** for newly introduced critical elements, while correctly returning exit code 0 on branches containing only pre-existing or waived risks.

**End-to-end evaluation of prescriptive mutations highlights a clear implementation gap.** We applied the fully compiled policy mutations from our prescriptive remediation operators unconditionally to the scenario corpus and measured the resulting impact shifts using the simulator oracle. This execution exposes a critical difference between our intended system design and the current software state:

| Evaluation System Topology | Generated Policy Edges | Vertices with Stable Bounds | Mean Post-Mutation Impact Shift ($\Delta I$) |
| --- | --- | --- | --- |
| **Clinical Healthcare** | 80 | 50 | **+30.94%** (Resilience Improved) |
| **Hub-and-Spoke** | 97 | 67 | **−31.67%** (Resilience Regressed) |
| **Financial Trading** | 94 | 62 | **−27.66%** (Resilience Regressed) |
| **Autonomous Vehicle** | 121 | 81 | **+23.53%** (Resilience Improved) |
| **Microservices Mesh** | 123 | 87 | **+15.34%** (Resilience Improved) |
| **IoT Smart City** | 327 | 191 | **+47.01%** (Resilience Improved) |
| **Hyper-Scale Enterprise** | 409 | 288 | **−25.36%** (Resilience Regressed) |
| **Cross-Scenario Global Mean** |  |  | **+4.61%** |

The global mean impact shift is positive ($+4.61\%$), but this aggregate conceals a significant risk: in 3 out of 7 scenarios, the unmitigated application of the full compiled policy causes a severe resilience regression, driving dependability down by up to $-31.67\%$.

This mixed result is the direct consequence of an **open implementation gap**: our target design specifies a per-edit verification filter where each mutation is simulated individually and rejected unless it clears a robust margin over seed noise ($\Delta I > \kappa\,\sigma_{\text{seed}}$). However, the current code stack lacks this filter; `PrescribeService` compiles all proposed operators and applies them unconditionally in a single pass. This allows counterproductive operators—such as node re-allocations that remove a colocation SPOF but introduce network routing hops—to merge unchecked. We report this limitation openly to establish the concrete development path required before prescriptive automation can be safely used in production setups.

---

## 9. Discussion, Limitations, and Conclusion

### 9.1 Threats to Validity

* **Construct Validity:** Our ground-truth failure labels $I(v)$ are derived from a discrete-event cascade simulation engine rather than direct observation of physical crashes in deployed clusters. Thus, our findings are inherently comparative—proving which graph models and metrics best capture simulated cascade propagation, rather than guaranteeing absolute predictive precision in live system operations.
* **Internal Validity:** The main internal risk involves circular feature leakage. We mitigate this via our structural independence guarantee, forcing feature graphs ($G_{\text{analysis}}$) and label simulations ($G_{\text{structural}}$) to execute on completely isolated graph definitions, with zero runtime telemetry feedback passed to the GNN layers.
* **External Validity:** All architectural evaluation suites are synthetic models created by a statistical generator. While spanning diverse industrial styles and scaling presets, this paper includes no verification data from actual production logs or human practitioner panels. Generalization to commercial configurations remains an unproven boundary condition.

### 9.2 Limitations and Future Work

Multiple boundaries guide our upcoming research phases. First, our continuous security dimension ($V$) remains thin, relying on simple multi-hop connectivity bounds; lacing the model with an explicit threat graph structure (trust boundaries, network access scopes) would expand its systemic security value. Second, we must address the remediation failure curve highlighted in Section 8.4. Wiring the per-edit counterfactual simulation gate ($\Delta I > \kappa\,\sigma_{\text{seed}}$) directly into `PrescribeService` is our highest technical priority, ensuring that backfiring mutations are pruned prior to policy execution. Finally, the framework requires validation against a live, production software cluster, mapping actual physical node drops to calibrate our simulated threshold metrics against real-world cascading parameters.

### 9.3 Conclusion

We introduced Software-as-a-Graph (SaG), a pre-deployment Static System Analysis framework designed to bridge the "Architecture-Code Gap" in distributed middleware configurations. By converting declarative infrastructure setups into typed, weighted multigraphs, SaG calculates deterministic multi-dimensional quality scores ($Q(v)$) across orthogonal RMAV bounds, and employs a Heterogeneous Graph Transformer ($HGL\text{-}QoS$) to forecast cascading failure impacts ($I(v)$) prior to launch.

Our evaluation defines a clear operational division of labor: deterministic attribution models in-distribution changes accurately, while heterogeneous graph transformers are necessary to generalize across completely unseen software domains. Operationalized via an in-memory, database-free repository, the framework runs as a blocking, delta-aware CI/CD Quality Gate, flagging structural regressions in seconds. Ultimately, SaG establishes that by treating architecture types as core modeling parameters, system engineers can diagnose and mitigate cascading failure vulnerabilities early in the software lifecycle, long before the system ever runs.

---

## References

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