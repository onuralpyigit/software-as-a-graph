# 3. The Software-as-a-Graph (SaG) Architectural Model

Figure 1 shows the SaG framework's full architecture. The front end processes Architecture-as-Code manifests, builds a typed multigraph, maps runtime interactions using a QoS-weighted logical dependency layer, and extracts typed node properties. From these results, the framework takes two separate paths with no shared parameters. In the predictive path (§4), a Heterogeneous Graph Transformer predicts how failures might spread. In the explanation layer (§5), the system breaks down fragility into Reliability and Maintainability quality profiles.

![Figure 1](latex/figures/Figure_1.png)

*Figure 1. End-to-end architecture of the SaG framework. The core claim is that the predictive pathway proceeds from left to right: manifest ingestion, typed multigraph construction, QoS-weighted DEPENDS_ON projection, extraction of typed node properties, heterogeneous graph learning, and identification of the ranked critical set. The dashed edge indicates the ground-truth simulation oracle, which operates exclusively on Gstructural, trains the predictor offline, and does not participate in inference. The explanation layer re-enters from the analysis multigraph, remains parameter-independent of the predictor, and identifies flagged components through triage rather than data flow.*

This section starts by describing the Software-as-a-Graph multigraph representation (§3.1), then explains the rules for QoS-aware weighting and logical dependency derivation (§3.2). It then introduces the two graph views (§3.3) and the typed node feature encodings (§3.4).

## 3.1 Formal Multigraph Definition

A complex distributed software system is modeled as a typed, weighted, and directed multigraph:

$$\tag{1}
\mathcal{G} = (V, E, \tau_V, \tau_E, w_V, w_E)$$

-   $V$ is the set of system entities, divided into five categories: $\mathcal{T}_V = \{\text{app}, \text{broker}, \text{topic}, \text{host}, \text{lib}\}$, so $V = V_{\text{app}} \cup V_{\text{broker}} \cup V_{\text{topic}} \cup V_{\text{host}} \cup V_{\text{lib}}$. To avoid confusion with physical machines, $V_{\text{host}}$ refers to *Execution Hosts*, which can be physical or virtual nodes.
-   $E$ is the set of directed edges connecting entities.
-   $w_V: V \to (0, 1]$ and $w_E: E \to (0, 1]$ are weighting functions that quantify entity criticality and connection strength, respectively. For applications and shared libraries, $w_V(v)$ is determined using static code metrics: $w_V(v) = 1 - \text{CQP}(v)$ (§3.4). Otherwise, we set the weight to $1.0$.

-   $E$ is the set of directed edges connecting entities.

-   $\tau_V: V \to \mathcal{T}_V$ and $\tau_E: E \to \mathcal{T}_E$ are typing functions assigning entity and relationship categories.

-   $w_V: V \to (0, 1]$ and $w_E: E \to (0, 1]$ are weighting functions that quantify entity criticality and connection strength, respectively. For applications and shared libraries, $w_V(v)$ is determined using static code metrics: $w_V(v) = 1 - \text{CQP}(v)$ (§3.4). For all other entities, we set the weight to $1.0$.

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

Application and Library entities use static code metrics from Static Code Analysis (SCA) tools, such as lines of code, cyclomatic complexity, coupling between objects, and method cohesion. These metrics connect code-level fragility to network analysis and help bridge the two views.

**Notation.** Entity and edge types: Table 1. Simulation oracles: Table 3.

|                         |                                          |                                            |                                                    |
|:------------------------|:-----------------------------------------|:-------------------------------------------|:---------------------------------------------------|
| $G_{\text{structural}}$ | Raw multigraph; oracles only             | $Q(v)$                                     | RM composite quality score                         |
| $G_{\text{analysis}}$   | `DEPENDS_ON` projection; predictor input | $\rho$                                     | Spearman $\rho$, full population                   |
| $V_{\text{app}}$        | Application nodes; the scored population | $\rho_{>0}$                                | Spearman $\rho$, active stratum ($I^* > 0$)        |
| $w(t)$, $w(e)$          | QoS topic weight, edge weight            | $F_1@K$                                    | Critical-set overlap, $K = 0.20\,|V_{\text{app}}|$ |
| $I^*(v)$                | Primary cascade-reachability oracle      | $I_{\text{comp}}$, $I_{\text{dyn}}$, $I_M$ | Further oracles (Table 3)                          |

## 3.2 QoS-Aware Weights and Logical Dependency Derivation

In distributed middleware, Quality-of-Service (QoS) contracts determine how strong a communication link is. For example, a `RELIABLE` topic with `TRANSIENT_LOCAL` durability creates a stronger connection between services than a `BEST_EFFORT` telemetry stream. This difference underpins the following weighting rules.

Each topic $t$ carries an intrinsic criticality weight $w(t) \in (0, 1]$ combining declared QoS semantics with two runtime-stress modulators: payload size and publication frequency:

$$\tag{2}
w(t) = \alpha_{\text{top}} \cdot \text{QoS}(t) + \beta_{\text{top}} \cdot \text{SizeNorm}(t) + \gamma_{\text{top}} \cdot \text{FreqNorm}(t),
\quad (\alpha_{\text{top}},\, \beta_{\text{top}},\, \gamma_{\text{top}}) = (0.75,\, 0.15,\, 0.10)$$ where $(\alpha_{\text{top}}, \beta_{\text{top}}, \gamma_{\text{top}})$ is a convex combination satisfying $\alpha_{\text{top}} + \beta_{\text{top}} + \gamma_{\text{top}} = 1.0$. The QoS term is an AHP-weighted aggregate of the declared contract:

$$\tag{3}
\text{QoS}(t) = w_{\text{rel}} \cdot q_{\text{rel}} + w_{\text{dur}} \cdot q_{\text{dur}} + w_{\text{prio}} \cdot q_{\text{prio}},
\quad (w_{\text{rel}}, w_{\text{dur}}, w_{\text{prio}}) = (0.24,\, 0.62,\, 0.14)$$

Here, $q_{\text{rel}}, q_{\text{dur}}, q_{\text{prio}} \in [0, 1]$ represent normalized reliability, durability, and transport-priority scores mapped from manifest policies: $q_{\text{rel}} \in \{0.0, 1.0\}$ (best-effort vs. reliable), $q_{\text{dur}} \in \{0.0, 0.5, 1.0\}$ (volatile, transient-local, persistent), and $q_{\text{prio}} \in \{0.0, 0.5, 1.0\}$ (low, medium, high). Durability carries the highest weight because it determines state maintenance and historical message replay across node restarts. Distinguish this structural coupling weight $w(t)$ from the instantaneous cascade reachability oracle $I^*(v)$ (§4.3): while $I^*(v)$ scales immediate reachability loss via reliability and priority tiers, durability governs whether failure states persist and propagate across lifecycle boundaries. Derive the sub-weights from a Saaty pairwise matrix with consistency ratio $CR = 0.016 \le 0.10$. The Topic QoS matrix is one of the two in the framework that carry genuine second-eigenvalue spread, so its $CR$ reports consistency rather than construction (Supplementary Table S4). Compress $\text{SizeNorm}(t)$ and $\text{FreqNorm}(t)$ logarithmically and clamp them to $[0, 1]$.

$$\tag{4}
\text{SizeNorm}(t) = \min\left(1.0, \frac{\log_2(1 + B(t))}{20}\right), \quad
\text{FreqNorm}(t) = \min\left(1.0, \frac{\log_{10}(1 + F(t))}{3}\right)$$

Here, $B(t)$ is the message payload size in bytes. The design limit is 1 MiB, which is the practical DDS sample size before RTPS fragmentation becomes an issue. $F(t)$ is the normal publication frequency in Hertz. The final weight $w(t)$ is kept between $[0.01, 1]$ so that best-effort edges are still visible during graph traversals. Each structural communication edge connected to $t$ (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`) uses $w_E(e) = w(t)$ and the topic’s QoS vector. The full $(\alpha_{\text{top}}, \beta_{\text{top}}, \gamma_{\text{top}})$ simplex keeps the orderings ($\rho \ge 0.919$), with only small ranking changes of $0.031$ (`Topo-QoS`) and $0.007$ (RM). Morris screening shows that none of the three parameters are critical ($\mu^* \le 0.025$), so this split is a documented choice, not a sensitive parameter.

### Logical Dependency Projection (`DEPENDS_ON`)

Structural edges show clear deployment connections but do not capture hidden runtime dependencies. For example, a subscriber depends on a publisher, but publish-subscribe architectures have no direct edge between them. To address this, we define a single semantic relation, `DEPENDS_ON`, from the dependent to the dependency ("if target fails, source is impacted"), using the six projection rules in Table 2. The resulting weight $w \in (0, 1]$ shows how much the dependent relies on the dependency and the chance that a disruption will spread.x `DEPENDS_ON` logical dependency projection rules.

| **Rule** | **Dependency Category** | **Structural Pattern ($\text{Dependent} \to \text{Dependency}$)**                    | **Derived Weight ($w$)**                                                                    |
|:--------:|:------------------------|:-------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|
|  **1**   | `app_to_app`            | Subscriber $\to$ Publisher (via shared Topic, incl. transitive `USES`)               | $1 - \prod_{t \in T}(1 - w(t))$                                                             |
|  **2**   | `app_to_broker`         | Publisher/Subscriber $\to$ Broker routing its topics                                 | $1 - \prod_{t \in T}(1 - w(t))$                                                             |
|  **3**   | `host_to_host`          | Host $\to$ Host (lifted from inter-host app dependencies)                            | $\max_{u \in \text{hosted}(h_1), v \in \text{hosted}(h_2)} w_{\text{DEPENDS\_ON}}(u \to v)$ |
|  **4**   | `host_to_broker`        | Host $\to$ Broker (lifted from hosted app dependencies)                              | $\max_{u \in \text{hosted}(h)} w_{\text{DEPENDS\_ON}}(u \to b)$                             |
|  **5**   | `app_to_lib`            | Application $\to$ Shared Library it `USES`                                           | $H(w_V(\text{app}), w_V(\text{lib}))$                                                       |
|  **6**   | `broker_to_broker`      | Broker $\leftrightarrow$ Broker (shared physical fault-domain colocation, symmetric) | $w_V(\text{host})$                                                                          |

Rules 1 and 2 combine the set of topics $T$ connecting two components using a probabilistic union instead of a maximum [77, 78, 79]. This approach shows that more parallel failure paths always increase coupling, while keeping $w$ in $(0, 1]$. Rule 5 uses the harmonic mean $H(x, y) = 2xy/(x+y)$ [80] to combine the Application and shared Library weights, balancing their importance. Rules 3 and 4 extend dependencies across hosts by choosing the highest coupling weight.

### Sequential Cascades vs. Simultaneous Blasts

A key idea in the SaG model is the difference between two types of degradation. In a **Sequential Cascade (Rule 1)**, if a publisher fails, downstream subscribers are gradually starved through message queues and topic buffers. In a **Simultaneous Blast (Rule 5)**, if a library or execution host fails, all consuming applications and colocated brokers fail at the same time in a shared-fate event. By keeping entity types and specific projection rules, the SaG model can show both mechanisms, while untyped graphs cannot distinguish them.

Rule 6 is the only symmetric rule: two brokers colocated on the same host share that host’s physical failure domain ($w = w_V(\text{host})$). Colocated brokers compete for resources, and host crashes halt them simultaneously. Rule 6 applies in four of the eight scenarios of a companion study’s detection benchmark (12 directed edges); that suite is not this paper’s corpus, and the figure is quoted only to indicate how rarely the rule fires. Because simulation oracles operate strictly on $G_{\text{structural}}$ (§4.4), Rule 6 has zero influence on ground-truth failure labels.

## 3.3 Dual Graph Views and Architectural Layers

The SaG framework uses two main representations. The **Structural Graph** ($G_{\text{structural}}$) is the raw deployment graph that shows physical and structural relations (`PUBLISHES_TO`, `ROUTES`, `RUNS_ON`, `USES`), keeping the actual deployment layout. The **Analysis Graph** ($G_{\text{analysis}}$) is a projected graph that adds derived `DEPENDS_ON` edges, QoS weights, and SCA metrics. We calculate all GNN embeddings and analytical metrics on the analysis graph (see Supplementary Figure S3).

$G_{\text{analysis}}$ is further organized into four analytical layers (Application, Middleware, Infrastructure, and Global System), enabling criticality evaluation at subsystem levels according to hierarchical frameworks such as MIL-STD-498 [81].

## 3.4 Typed Node Feature Encoding

Both the predictive pathway (§4) and the explanation layer (§5) use the same typed node properties from $G_{\text{analysis}}$: the predictor projects them per entity type before message passing, while the explanation layer combines them into a quality profile. All five entity types share a common 18-dimensional base block (indices 0–17) of topological metrics: (0) PageRank ($PR$), (1) Reverse PageRank ($RPR$), (2) Betweenness Centrality ($BT$), (3) Closeness Centrality ($CL$), (4) Eigenvector Centrality ($EV$), (5) In-Degree Centrality ($DG_{\text{in}}$), (6) Out-Degree Centrality ($DG_{\text{out}}$), (7) Local Clustering Coefficient ($CC$), (8) Undirected Articulation Score ($AP$), (9) Bridge Ratio ($BR$), (10) Node QoS Weight ($w$), (11) QoS-Weighted In-Degree ($w_{\text{in}}$), (12) QoS-Weighted Out-Degree ($w_{\text{out}}$), (13) Multi-Path Coupling Index ($MPCI$), (14) Path Complexity ($PC$), (15) Fan-Out Criticality ($FOC$), (16) Directed Articulation Point ($AP_c^{\text{dir}}$), and (17) Connectivity Degradation Index ($CDI$). Supplementary Table S11 (Supplementary §S11) gives the full mathematical schema, definitions, and normalizations for these metrics. CDI is the most computationally expensive and dominates the analysis cost in §7.5. Because it is a predictor input and not just part of the Availability score, we cannot remove it without changing both pathways. Each metric is normalized to $[0, 1]$ within its graph, preventing raw graph size from affecting per-type projects across scenarios. Type-speciflocks extend the vector to 19–25 dimensions, adding source-code metrics and the Code Quality Penalty for Applications, reverse-`USES` blast-radius drivers for Libraries, queue capacity for Brokers, publisher/subscriber counts and ordinal QoS criticality for Topics, and CPU and memory allocation for Execution Hosts.

The shared block gives the GNN global structural and positional context, similar to positional and structural encodings in Graph Transformers. This allows relational message passing to adjust multi-hop representations based on each node's role in the whole network, not just its immediate neighbors. Since betweenness, closeness, reverse PageRank, and articulation scores are calculated before model evaluation, a learned model is not the only way to get a criticality score from the structure. This shows that closed-form baselines are fair and competitive, and points out that deterministic feature extraction is the main computational bottleneck ($O(|V|^2 + |V||E|)$) discussed in §7.5.
