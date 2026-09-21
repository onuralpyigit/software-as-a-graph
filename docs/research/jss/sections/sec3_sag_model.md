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

Both the predictive pathway (§4) and the explanation layer (§5) read the same typed node properties from $G_{\text{analysis}}$: the predictor projects them per entity type before message passing, the explanation layer aggregates them into a quality profile. All five entity types share a common, deterministic 18-dimensional base block (indices 0–17) of topological metrics: (0) PageRank ($PR$), (1) Reverse PageRank ($RPR$), (2) Betweenness Centrality ($BT$), (3) Closeness Centrality ($CL$), (4) Eigenvector Centrality ($EV$), (5) In-Degree Centrality ($DG_{\text{in}}$), (6) Out-Degree Centrality ($DG_{\text{out}}$), (7) Local Clustering Coefficient ($CC$), (8) Undirected Articulation Score ($AP$), (9) Bridge Ratio ($BR$), (10) Node QoS Weight ($w$), (11) QoS-Weighted In-Degree ($w_{\text{in}}$), (12) QoS-Weighted Out-Degree ($w_{\text{out}}$), (13) Multi-Path Coupling Index ($MPCI$), (14) Path Complexity ($PC$), (15) Fan-Out Criticality ($FOC$), (16) Directed Articulation Point ($AP_c^{\text{dir}}$), and (17) Connectivity Degradation Index ($CDI$). The full mathematical schema, definitions, and normalizations for each of these eighteen metrics are detailed in Supplementary Table S11 (Supplementary §S11). CDI is by far the most expensive of these and dominates the analysis cost of §7.5; because it is a predictor input and not only a term of the Availability score, it cannot be gated away without changing both pathways. Every metric in the block is normalized to $[0, 1]$ within its graph, which stops raw graph size from driving the per-type projections under cross-scenario transfer. Type-specific blocks extend the vector to 19–25 dimensions, adding source-code metrics and the Code Quality Penalty for Applications, reverse-`USES` blast-radius drivers for Libraries, queue capacity for Brokers, publisher/subscriber counts and ordinal QoS criticality for Topics, and CPU and memory allocation for Execution Hosts.

The shared block provides the GNN with global structural and positional context—analogous to positional and structural encodings in Graph Transformers—enabling relational message passing to modulate multi-hop representations based on global network role rather than local immediate adjacency alone. Crucially, because betweenness, closeness, reverse PageRank, and articulation scores are topological summaries computed before model evaluation, a learned model is not the only way to derive a criticality score from structure. This design guarantees that the closed-form baselines serve as fair, competitive comparators rather than strawman alternatives, while making deterministic feature extraction the dominant computational bottleneck ($O(|V|^2 + |V||E|)$) analyzed in §7.5.
