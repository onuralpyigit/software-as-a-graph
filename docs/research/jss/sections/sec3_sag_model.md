# 3. The Software-as-a-Graph (SaG) Architectural Model

Figure 1 presents the end-to-end architecture of the SaG framework. The shared front end processes Architecture-as-Code manifests, constructs a typed multigraph, projects implicit runtime interactions throughout a QoS-weighted logical dependency layer, and extracts typed node properties. These features feed two independent pathways with no shared parameters: the predictive pathway (§4), which forecasts cascading failure blast radii and per-relationship criticality using a Heterogeneous Graph Transformer; and the explanation layer (§5), which decomposes fragility into Reliability and Maintainability quality profiles.

![Figure 1](latex/figures/Figure_1.png)

*Figure 1. End-to-end architecture of the SaG framework. The central assertion is that the predictive pathway (§4) forms the primary sequence: manifest ingestion → typed multigraph → QoS-weighted DEPENDS_ON projection → typed node properties → heterogeneous graph learning → a ranked critical set with per-relationship criticality → the ground-truth simulation oracle (§4.3) that evaluates it. The oracle completes the predictive pathway’s training loop and operates solely on Gstructural; it functions strictly offline and is excluded from inference (§4.4), as indicated by the dashed edge. The explanation layer (§5) differs from this sequence: it re-enters from the analysis multigraph, generates a standards-based quality profile from the same typed features without sharing parameters with the predictor, and is accessed through triage instead of direct data flow.*

This section formalizes the Software-as-a-Graph multigraph representation (§3.1), the QoS-aware weighting and logical dependency derivation rules (§3.2), the dual graph views (§3.3), and the typed node feature encodings (§3.4).

## 3.1 Formal Multigraph Definition

A complex distributed software system is formally represented as a typed, weighted, directed multigraph:

$$\tag{1}
\mathcal{G} = (V, E, \tau_V, \tau_E, w_V, w_E)$$

where:

-   $V$ is the set of system entities, partitioned into five disjoint categories $\mathcal{T}_V = \{\text{app}, \text{broker}, \text{topic}, \text{host}, \text{lib}\}$ such that $V = \bigcup_{t \in \mathcal{T}_V} V_t$:

    $$\tag{2}
    V = V_{\text{app}} \cup V_{\text{broker}} \cup V_{\text{topic}} \cup V_{\text{host}} \cup V_{\text{lib}}$$

    To prevent conflation of graph vertices with physical compute machines, $V_{\text{host}}$ is designated as *Execution Hosts* (physical hosts or virtualized execution nodes).

-   $E$ is the set of directed edges connecting entities.

-   $\tau_V: V \to \mathcal{T}_V$ and $\tau_E: E \to \mathcal{T}_E$ are typing functions assigning entity and relationship categories.

-   $w_V: V \to (0, 1]$ and $w_E: E \to (0, 1]$ are weighting functions representing entity criticality and connection strength. For applications and shared libraries, $w_V(v)$ is initialized from static code metrics as $w_V(v) = 1 - \text{CQP}(v)$ (where $\text{CQP}$ denotes the Code Quality Penalty, §3.4). It defaults to $1.0$ when static code metrics are absent or for infrastructure entities ($w_V(\text{host}) = 1.0$).

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

Application and Library entities also incorporate static code metrics generated by Static Code Analysis (SCA) tools (lines of code, cyclomatic complexity, coupling between objects, lack of cohesion in methods), thereby linking code-level fragility directly to topological analysis.

## 3.2 QoS-Aware Weights and Logical Dependency Derivation

In distributed middleware, communication links differ in strength according to their Quality-of-Service (QoS) contracts. For example, a `RELIABLE` topic with `TRANSIENT_LOCAL` durability creates a stronger binding between communicating services than a `BEST_EFFORT` telemetry stream.

Each topic $t$ carries an intrinsic criticality weight $w(t) \in (0, 1]$ combining its declared QoS semantics with two runtime-stress modulators: payload size and publication frequency:

$$\tag{3}
w(t) = \alpha_{\text{top}} \cdot \text{QoS}(t) + \beta_{\text{top}} \cdot \text{SizeNorm}(t) + \gamma_{\text{top}} \cdot \text{FreqNorm}(t),
\quad (\alpha_{\text{top}},\, \beta_{\text{top}},\, \gamma_{\text{top}}) = (0.75,\, 0.15,\, 0.10)$$

where the QoS term is an AHP-weighted aggregate of the declared contract:

$$\tag{4}
\text{QoS}(t) = w_{\text{rel}} \cdot q_{\text{rel}} + w_{\text{dur}} \cdot q_{\text{dur}} + w_{\text{prio}} \cdot q_{\text{prio}},
\quad (w_{\text{rel}}, w_{\text{dur}}, w_{\text{prio}}) = (0.24,\, 0.62,\, 0.14)$$

Here, $q_{\text{rel}}, q_{\text{dur}}, q_{\text{prio}} \in [0, 1]$ represent normalized reliability, durability, and transport-priority scores mapped directly from declared manifest policies: $q_{\text{rel}} \in \{0.0, 1.0\}$ (best-effort vs. reliable), $q_{\text{dur}} \in \{0.0, 0.5, 1.0\}$ (volatile, transient-local, persistent), and $q_{\text{prio}} \in \{0.0, 0.5, 1.0\}$ (low, medium, high). Durability dominates because it determines whether data survives restarts and network partitions. Reliability and transport priority both govern in-flight delivery quality, with reliability receiving higher weight because unconditional delivery guarantees precede message scheduling. The sub-weight vector is the geometric-mean priority vector of an independently stated Saaty pairwise-comparison matrix with consistency ratio $CR = 0.016$, well within Saaty's consistency threshold ($CR \le 0.10$).

The modulators $\text{SizeNorm}(t)$ and $\text{FreqNorm}(t)$ are logarithmically compressed and clamped to $[0, 1]$:

$$\tag{5}
\text{SizeNorm}(t) = \min\left(1.0, \frac{\log_2(1 + B(t))}{20}\right), \quad
\text{FreqNorm}(t) = \min\left(1.0, \frac{\log_{10}(1 + F(t))}{3}\right)$$

Here, $B(t)$ denotes the message payload size in bytes (with a 1 MiB design envelope, which represents the practical DDS sample ceiling before RTPS fragmentation becomes dominant), and $F(t)$ represents the nominal publication frequency in Hertz. The final weight $w(t)$ is clamped to $[0.01, 1]$ to ensure that best-effort edges remain visible during graph traversals. Each structural communication edge incident on $t$ (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`) inherits $w_E(e) = w(t)$ together with the topic’s QoS vector.

The outer split $(\alpha_{\text{top}}, \beta_{\text{top}}, \gamma_{\text{top}})$ constitutes a declared convex combination. Sweeping the full simplex leaves the induced $w(t)$ ordering essentially intact — rank correlation against the shipped split never falls below $\rho = 0.919$ — and moves the downstream ranking by $0.031$ (`Topo-QoS`) and $0.007$ (RM) across the seven sampled points. Under Morris screening none of the three constants is load-bearing ($\mu^* \le 0.025$). The split is therefore a documented convention rather than a sensitive parameter.

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

Rules 1 and 2 aggregate the set of topics $T$ connecting a component pair using a probabilistic union rather than a maximum [72, 73, 74]. This approach confirms that additional parallel failure vectors increase coupling monotonically while maintaining $w \in (0, 1]$. Rule 5 employs the harmonic mean $H(x, y) = 2xy/(x+y)$ [75] to combine the vertex weights of the consuming Application and the shared Library, consequently balancing caller and dependency criticality. Rules 3 and 4 extend application-level dependencies across host boundaries using the maximum coupling weight.

### Sequential Cascades vs. Simultaneous Blasts

A central principle of the SaG model is the distinction between two degradation modes: (1) **Sequential Cascades (Rule 1)**, in which a failed publisher starves downstream subscribers sequentially through message queues and topic buffers; and (2) **Simultaneous Blasts (Rule 5)**, in which a crashed library or execution host causes all consuming applications and colocated brokers to fail instantaneously in a single shared-fate event. Preserving entity types and relation-specific projection rules lets the SaG model represent both mechanisms, whereas untyped homogeneous graphs collapse them into indistinguishable edges.

Rule 6 is the only symmetric projection rule: two brokers colocated on the same execution host share that host’s physical failure domain (following the simultaneous-blast principle of Rule 5, with $w = w_V(\text{host})$). In production deployments, colocated brokers compete for CPU, memory, and I/O; a host crash halts all colocated instances simultaneously. Rule 6 does not model logical intra-cluster broker coupling (e.g., partition replication, quorum election, or shovel links, which do not require physical colocation). It applies in four of the eight detection benchmark contexts and contributes only 12 directed edges. Because simulation oracles operate strictly on $G_{\text{structural}}$ (§4.4), Rule 6 has zero influence on ground-truth failure labels.

## 3.3 Dual Graph Views and Architectural Layers

The SaG framework consists of two distinct representations: (1) **Structural Graph** ($G_{\text{structural}}$), which is the raw deployment graph containing physical and structural relations (`PUBLISHES_TO`, `ROUTES`, `RUNS_ON`, `USES`) and preserves the untransformed deployment topology; and (2) **Analysis Graph** ($G_{\text{analysis}}$), which is the projected graph containing derived `DEPENDS_ON` edges annotated with QoS weights and ingested SCA metrics. We compute all GNN embeddings and analytical metrics on the analysis graph (Figure 2).

![Figure 2](latex/figures/Figure_2.png)

*Figure 2. Running example: the raw structural graph (left) and the `DEPENDS_ON` projection derived from it (right). The projection makes implicit runtime dependencies explicit—a subscriber depends on the publishers of its topics even though no structural edge joins them—while the simulators continue to operate on the structural view alone.*

$G_{\text{analysis}}$ is further organized into four analytical layers (Application, Middleware, Infrastructure, and Global System), enabling criticality evaluation at subsystem levels consistent with hierarchical frameworks such as MIL-STD-498 [76].

## 3.4 Typed Node Feature Encoding

Within the SaG architecture, both the predictive pathway (§4) and the explanation layer (§5) utilize the same unified typed node properties from $G_{\text{analysis}}$. The predictive pathway projects these properties per entity type before heterogeneous message passing, while the explanation layer aggregates them into its quality profile. All five entity types share indices 0–17, a common block of topological metrics: in/out degree, PageRank and reverse PageRank, betweenness, closeness, eigenvector centrality, clustering coefficient, articulation score, bridge ratio, path complexity, fan-out criticality, the multi-path connectivity index, and the Connectivity Degradation Index (CDI, index 17). CDI is the single most expensive metric in the block and dominates the deterministic analysis cost characterized in §7.5; because it is a predictor input and not only a term of the Availability score (§5.2), it cannot be gated away without changing both pathways. The deterministic analysis stage produces these metrics, and § 7.5 describes its computationally demanding cost. All topological metrics in this block are normalized to $[0, 1]$ within each graph: degrees are normalized by $|V|-1$, betweenness and closeness follow standard network formulations, and reverse PageRank is normalized to unit sum. This normalization prevents raw graph size and component counts from controlling multi-layer perceptron projections during cross-scenario inductive transfer. Type-specific blocks extend the feature set to between 19 and 25 dimensions: source-code metrics and the Code Quality Penalty for Applications, two reverse-`USES` blast-radius drivers for Libraries, queue capacity for Brokers, publisher/subscriber counts and ordinal QoS criticality for Topics, and CPU and memory allocation for Execution Hosts ($V_{\text{host}}$).

The shared block includes the graph composition, which is relevant to understanding §7: betweenness, closeness, reverse PageRank, and articulation score are topological summaries computed before model evaluation. Consequently, a learned model is not the only way to derive a criticality score from structure, which ensures that the closed-form baselines serve as fair comparators rather than strawman alternatives.
