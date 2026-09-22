# 3. The Software-as-a-Graph (SaG) Architectural Model

Figure 1 shows the SaG framework's complete architecture and core claim: the front end processes Architecture-as-Code manifests, builds a typed multigraph, maps runtime interactions using a QoS-weighted dependency layer, and extracts node properties. Two separate pathways then use these outputs and do not share parameters, with the predictive pathway (§4) using a Heterogeneous Graph Transformer to forecast cascading failure blast radii. Meanwhile, the explanation layer (§5) breaks down fragility into Reliability and Maintainability quality profiles.

![Figure 1](latex/figures/Figure_1.png)

*Figure 1. End-to-end architecture of the SaG framework and its core claim. The predictive pathway proceeds from left to right: manifest ingestion, typed multigraph construction, QoS-weighted DEPENDS_ON projection, extraction of typed node properties, heterogeneous graph learning, and identification of the ranked critical set. The dashed edge indicates the ground-truth simulation oracle, which operates exclusively on $G_{structural}$, trains the predictor offline, and does not participate in inference. The explanation layer re-enters from the analysis multigraph, shares no parameters with the predictor, and identifies flagged components through triage rather than data flow.*

This section first defines the Software-as-a-Graph multigraph representation (§3.1). It then explains the QoS-aware weighting and logical dependency rules (§3.2), describes the dual graph views (§3.3), and introduces the typed node feature encodings (§3.4).

## 3.1 Formal Multigraph Definition

A complex distributed software system can be explicitly described as a typed, weighted, and directed multigraph:

$$\tag{1}
\mathcal{G} = (V, E, \tau_V, \tau_E, w_V, w_E)$$

where:

$V$ is the set of system entities, divided into five categories: $\mathcal{T}_V = \{\text{app}, \text{broker}, \text{topic}, \text{host}, \text{lib}\}$, so $V = V_{\text{app}} \cup V_{\text{broker}} \cup V_{\text{topic}} \cup V_{\text{host}} \cup V_{\text{lib}}$. To avoid confusion with physical machines, $V_{\text{host}}$ refers to *Execution Hosts*, which can be physical or virtual nodes.

-   $E$ is the set of directed edges connecting entities.

-   $\tau_V: V \to \mathcal{T}_V$ and $\tau_E: E \to \mathcal{T}_E$ are typing functions assigning entity and relationship categories.

$w_V: V \to (0, 1]$ and $w_E: E \to (0, 1]$ are weighting functions that show how critical an entity is and how strong a connection is. For applications and shared libraries, $w_V(v)$ is set using static code metrics: $w_V(v) = 1 - \text{CQP}(v)$ (§3.4). When static code metrics are unavailable, or for infrastructure entities, set the weight to $1.0$ ($w_V(\text{host}) = 1.0$).

Table 1 summarizes the five entity types and six structural edge types in the SaG model, together with their semantics and representative distributed-system implementations.

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

Application and Library entities use static code metrics from Static Code Analysis (SCA) tools, including lines of code, cyclomatic complexity, coupling between objects, and method cohesion. Together, these metrics show how code-level fragility connects to topological analysis.

**Notation.** Entity and edge types: Table 1. Simulation oracles: Table 3.

|                         |                                          |                                            |                                                    |
|:------------------------|:-----------------------------------------|:-------------------------------------------|:---------------------------------------------------|
| $G_{\text{structural}}$ | Raw multigraph; oracles only             | $Q(v)$                                     | RM composite quality score                         |
| $G_{\text{analysis}}$   | `DEPENDS_ON` projection; predictor input | $\rho$                                     | Spearman $\rho$, full population                   |
| $V_{\text{app}}$        | Application nodes; the scored population | $\rho_{>0}$                                | Spearman $\rho$, active stratum ($I^* > 0$)        |
| $w(t)$, $w(e)$          | QoS topic weight, edge weight            | $F_1@K$                                    | Critical-set overlap, $K = 0.20\,|V_{\text{app}}|$ |
| $I^*(v)$                | Primary cascade-reachability oracle      | $I_{\text{comp}}$, $I_{\text{dyn}}$, $I_M$ | Further oracles (Table 3)                          |

## 3.2 QoS-Aware Weights and Logical Dependency Derivation

In distributed middleware, a communication link's strength depends on Quality-of-Service (QoS) contracts. For example, a `RELIABLE` topic with `TRANSIENT_LOCAL` durability creates a stronger connection between services than a `BEST_EFFORT` telemetry stream. This difference underpins the following weighting rules.

Each topic $t$ carries an intrinsic criticality weight $w(t) \in (0, 1]$ combining its declared QoS semantics with two runtime-stress modulators: payload size and publication frequency:

$$\tag{2}
w(t) = \alpha_{\text{top}} \cdot \text{QoS}(t) + \beta_{\text{top}} \cdot \text{SizeNorm}(t) + \gamma_{\text{top}} \cdot \text{FreqNorm}(t),
\quad (\alpha_{\text{top}},\, \beta_{\text{top}},\, \gamma_{\text{top}}) = (0.75,\, 0.15,\, 0.10)$$ where $(\alpha_{\text{top}}, \beta_{\text{top}}, \gamma_{\text{top}})$ is a convex combination satisfying $\alpha_{\text{top}} + \beta_{\text{top}} + \gamma_{\text{top}} = 1.0$. The QoS term is an AHP-weighted aggregate of the declared contract:

$$\tag{3}
\text{QoS}(t) = w_{\text{rel}} \cdot q_{\text{rel}} + w_{\text{dur}} \cdot q_{\text{dur}} + w_{\text{prio}} \cdot q_{\text{prio}},
\quad (w_{\text{rel}}, w_{\text{dur}}, w_{\text{prio}}) = (0.24,\, 0.62,\, 0.14)$$

Here, $q_{\text{rel}}, q_{\text{dur}}, q_{\text{prio}} \in [0, 1]$ represent normalized reliability, durability, and transport-priority scores mapped from manifest policies: $q_{\text{rel}} \in \{0.0, 1.0\}$ (best-effort vs. reliable), $q_{\text{dur}} \in \{0.0, 0.5, 1.0\}$ (volatile, transient-local, persistent), and $q_{\text{prio}} \in \{0.0, 0.5, 1.0\}$ (low, medium, high). Durability carries the highest weight because it determines state maintenance across restarts. The sub-weights derive from a Saaty pairwise matrix with consistency ratio $CR = 0.016 \le 0.10$. We note that $CR$ is uninformative for a matrix back-filled from a chosen priority vector, so we state which case this is: the Topic QoS matrix is one of the two in the framework that carry genuine second-eigenvalue spread. Its $CR$ therefore reports consistency rather than construction (Supplementary Table S4). The modulators $\text{SizeNorm}(t)$ and $\text{FreqNorm}(t)$ are logarithmically compressed and clamped to $[0, 1]$.


$$\tag{4}
\text{SizeNorm}(t) = \min\left(1.0, \frac{\log_2(1 + B(t))}{20}\right), \quad
\text{FreqNorm}(t) = \min\left(1.0, \frac{\log_{10}(1 + F(t))}{3}\right)$$
Here, $B(t)$ is the message payload size in bytes. The design envelope is 1 MiB, which is the practical DDS sample limit before RTPS fragmentation becomes a problem. $F(t)$ is the normal publication frequency in Hertz. The final weight $w(t)$ is limited to $[0.01, 1]$ so that best-effort edges remain visible during graph traversals. Each structural communication edge connected to $t$ (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`) uses $w_E(e) = w(t)$ and the topic’s QoS vector. The full $(\alpha_{\text{top}}, \beta_{\text{top}}, \gamma_{\text{top}})$ simplex keeps the orderings ($\rho \ge 0.919$), with small ranking changes of $0.031$ (`Topo-QoS`) and $0.007$ (RM). Morris screening shows that none of the three parameters are critical ($\mu^* \le 0.025$), so the split is a documented choice, not a sensitive parameter.

### Logical Dependency Projection (`DEPENDS_ON`)

Structural edges represent explicit deployment connections but do not capture implicit runtime dependencies. For example, a subscriber depends on a publisher, yet no direct edge connects them in publish-subscribe architectures. To tackle this limitation, a single unified semantic relation, `DEPENDS_ON`, is derived and directed from *dependent* to *dependency* ("if target fails, source is impacted"), following the six projection rules detailed in Table 2. The resulting weight $w \in (0, 1]$ quantifies the magnitude of operational coupling and indicates the conditional likelihood that a disruption in the dependency propagates to the dependent.

**Table 2.** The six `DEPENDS_ON` logical dependency projection rules.

| **Rule** | **Dependency Category** | **Structural Pattern ($\text{Dependent} \to \text{Dependency}$)**                    | **Derived Weight ($w$)**                                                                    |
|:--------:|:------------------------|:-------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|
|  **1**   | `app_to_app`            | Subscriber $\to$ Publisher (via shared Topic, incl. transitive `USES`)               | $1 - \prod_{t \in T}(1 - w(t))$                                                             |
|  **2**   | `app_to_broker`         | Publisher/Subscriber $\to$ Broker routing its topics                                 | $1 - \prod_{t \in T}(1 - w(t))$                                                             |
|  **3**   | `host_to_host`          | Host $\to$ Host (lifted from inter-host app dependencies)                            | $\max_{u \in \text{hosted}(h_1), v \in \text{hosted}(h_2)} w_{\text{DEPENDS\_ON}}(u \to v)$ |
|  **4**   | `host_to_broker`        | Host $\to$ Broker (lifted from hosted app dependencies)                              | $\max_{u \in \text{hosted}(h)} w_{\text{DEPENDS\_ON}}(u \to b)$                             |
|  **5**   | `app_to_lib`            | Application $\to$ Shared Library it `USES`                                           | $H(w_V(\text{app}), w_V(\text{lib}))$                                                       |
|  **6**   | `broker_to_broker`      | Broker $\leftrightarrow$ Broker (shared physical fault-domain colocation, symmetric) | $w_V(\text{host})$                                                                          |

Rules 1 and 2 combine the set of topics $T$ connecting a pair of components using a probabilistic union instead of a maximum [77, 78, 79]. This method confirms that more parallel failure paths always increase coupling, while keeping $w$ within $(0, 1]$. Rule 5 uses the harmonic mean $H(x, y) = 2xy/(x+y)$ [80] to combine the weights of the Application and the shared Library, balancing their importance. Rules 3 and 4 extend dependencies across hosts by using the highest coupling weight.

### Sequential Cascades vs. Simultaneous Blasts

A key idea in the SaG model is the difference between two types of degradation. In a **Sequential Cascade (Rule 1)**, if a publisher fails, it gradually starves downstream subscribers through message queues and topic buffers. In a **Simultaneous Blast (Rule 5)**, if a library or execution host crashes, all connected applications and brokers fail at once in a single event. By keeping entity types and specific projection rules, the SaG model can show both mechanisms, while untyped graphs cannot make this distinction.

Rule 6 is the only symmetric rule: two brokers colocated on the same host share that host’s physical failure domain ($w = w_V(\text{host})$). Colocated brokers compete for resources, and host crashes halt them simultaneously. Rule 6 applies in four of the eight scenarios of a companion study’s detection benchmark (12 directed edges); that suite is not this paper’s corpus, and the figure is quoted only to indicate how rarely the rule fires. Because simulation oracles operate strictly on $G_{\text{structural}}$ (§4.4), Rule 6 has zero influence on ground-truth failure labels.

## 3.3 Dual Graph Views and Architectural Layers

The SaG framework uses two main representations. The **Structural Graph** ($G_{\text{structural}}$) is the basic deployment graph, showing physical and structural relationships like `PUBLISHES_TO`, `ROUTES`, `RUNS_ON`, and `USES`, and keeps the base deployment layout. The **Analysis Graph** ($G_{\text{analysis}}$) is a version of the graph that adds `DEPENDS_ON` edges with QoS weights and SCA metrics. We calculate all GNN embeddings and analytical metrics on the analysis graph (see Supplementary Figure S3). The graph is further organized into four analytical layers (Application, Middleware, Infrastructure, and Global System), enabling criticality evaluation at subsystem levels consistent with hierarchical frameworks such as MIL-STD-498 [81].

## 3.4 Typed Node Feature Encoding

Both the predictive pathway (§4) and the explanation layer (§5) use the same typed node properties from $G_{\text{analysis}}$. The predictor uses these properties for each entity type before message passing, while the explanation layer combines them into a quality profile. All five entity types share a standard 18-dimensional base block of topological metrics, including PageRank, Reverse PageRank, Betweenness Centrality, Closeness Centrality, Eigenvector Centrality, In-Degree and Out-Degree Centrality, Local Clustering Coefficient, Articulation Score, Bridge Ratio, Node QoS Weight, QoS-Weighted In-Degree and Out-Degree, Multi-Path Coupling Index, Path Complexity, Fan-Out Criticality, Directed Articulation Point, and Connectivity Degradation Index (CDI). Supplementary Table S11 (Supplementary § S11) gives the full definitions and normalizations for these metrics. CDI is the most computationally expensive and is a key part of the analysis in §7.5. Because it is used as a predictor input, not just for the Availability score, we cannot remove it without changing both pathways. All metrics are normalized to $[0, 1]$ within each graph, so graph size does not affect projections. Type-specific blocks add more features, increasing the vector to 19–25 dimensions. These include source-code metrics and the Code Quality Penalty for Applications, reverse-`USES` blast-radius drivers for Libraries, queue capacity for Brokers, publisher/subscriber counts and QoS criticality for Topics, and CPU and memory allocation for Execution Hosts.

The shared block gives the GNN a global view of structure and position, similar to how Graph Transformers use positional and structural encodings. This setup lets relational message passing adjust multi-hop representations based on the entire network role, not just local connections. We calculate measures including betweenness, closeness, reverse PageRank, and articulation scores before model evaluation, showing that a learned model is not the only way to obtain a criticality score from structure. This method confirms that closed-form baselines are fair and competitive, while deterministic feature extraction is the main computational bottleneck ($O(|V|^2 + |V||E|)$), as discussed in §7.5.
