# 3. The Software-as-a-Graph (SaG) Architectural Model

Figure 1 presents the end-to-end architecture of the SaG framework. The shared front end processes Architecture-as-Code manifests, constructs a typed multigraph, projects implicit runtime interactions throughout a logical dependency layer, and extracts typed node properties. These features feed the ranking engines (§4) and, separately and with no shared parameters, the proposed explanation layer.

![Figure 1](../latex/figures/Figure_1.png)

*Figure 1. End-to-end architecture of the SaG framework. The predictive pathway runs down the center: manifest ingestion, typed multigraph, DEPENDS_ON projection with typed node properties, the ranking engines (training-free baseline, learned and hybrid; Figure 3), and the ranked critical set. The dashed edge marks the ground-truth simulation oracles, which operate only on Gstructural, train the predictor offline and take no part in inference. The proposed explanation layer (not evaluated) reads the same analysis multigraph, shares no parameters with the predictor, and is applied to components after they have been ranked; no output of the predictor flows into it.*

## 3.1 Multigraph Definition

A distributed system is described as a typed, weighted, directed multigraph:

$$\tag{1}
\mathcal{G} = (V, E, \tau_V, \tau_E, w_V, w_E)$$

where:

-   $V = V_{\text{app}} \cup V_{\text{broker}} \cup V_{\text{topic}} \cup V_{\text{host}} \cup V_{\text{lib}}$ holds the five entity types $\mathcal{T}_V$ of Table 1; $V_{\text{host}}$ denotes physical or virtual *Execution Hosts*.

-   $E$ is the set of directed edges, and $\tau_V: V \to \mathcal{T}_V$ and $\tau_E: E \to \mathcal{T}_E$ assign entity and relation types.

-   $w_V: V \to (0, 1]$ and $w_E: E \to (0, 1]$ weight entity criticality and connection strength. For Applications and Libraries, $w_V(v) = 1 - \text{CQP}(v)$, where CQP is a code-quality penalty computed from static code metrics (lines of code, cyclomatic complexity, coupling and cohesion); otherwise $w_V(v) = 1.0$.

The raw multigraph is denoted $G_{\text{structural}}$ (used exclusively by simulation oracles), while $G_{\text{analysis}}$ denotes the logical `DEPENDS_ON` projection consumed by predictors. A comprehensive reference of mathematical notation is provided in the replication repository.

**Table 1.** Entity types and structural edge types in the SaG model.

| **Entity Type ($\mathcal{T}_V$)**      | **Architectural Role**                        | **Concrete System Examples**                   |
|:---------------------------------------|:----------------------------------------------|:-----------------------------------------------|
| **Application** ($V_{\text{app}}$)     | Process producing/consuming messages          | ROS 2 node, Kafka microservice, MQTT client    |
| **Broker** ($V_{\text{broker}}$)       | Message routing and queuing intermediary      | RabbitMQ exchange, Mosquitto, EMQX broker      |
| **Topic** ($V_{\text{topic}}$)         | Named logical communication channel           | `/sensor/lidar`, `orders.payment.completed`    |
| **Execution Host** ($V_{\text{host}}$) | Physical or virtualized execution environment | Bare-metal server, Kubernetes worker, Cloud VM |
| **Library** ($V_{\text{lib}}$)         | Shared software package or runtime dependency | Kafka client, OpenCV, Protobuf runtime         |
| **Structural Edge ($\mathcal{T}_E$)**  | **Direction**                                 | **Semantic Meaning**                           |
| `PUBLISHES_TO` / `SUBSCRIBES_TO`       | App/Library $\to$ Topic                       | Component publishes to / consumes from topic   |
| `ROUTES`                               | Broker $\to$ Topic                            | Broker manages and routes topic traffic        |
| `RUNS_ON`                              | App/Broker $\to$ Host                         | Process is hosted on host                      |
| `CONNECTS_TO`                          | Host $\to$ Host                               | Network link between hosts                     |
| `USES`                                 | App $\to$ Library                             | Application links to shared library            |

## 3.2 Quality-of-Service Link Weighting

A link’s strength depends on its Quality-of-Service (QoS) contract: a `RELIABLE` topic with `TRANSIENT_LOCAL` durability couples services more strongly than a `BEST_EFFORT` telemetry stream. Each topic $t$ carries an aggregate weight $w(t) \in (0, 1]$ combining declared QoS policies (reliability, durability, priority) with payload size and publication frequency. The sub-weights of reliability, durability and priority come from an Analytic Hierarchy Process (AHP) pairwise-comparison matrix that was stated independently rather than back-solved from a target vector ($CR = 0.016$, non-degenerate; the framework’s other AHP matrices also encode a declared vector).

Whether these weights help a learned ranker is measured directly: with the weighted in-degree held in both arms, the QoS inputs add nothing measurable to the graph neural networks on the reachability oracle (§6.2), whereas on the queue-flow oracle they carry the learned surrogate’s gain (§6.1). How the weighting affects the training-free baseline is archived in the replication repository. The reference dependency counts reported in this paper are unweighted and do not read $w(t)$.

## 3.3 Logical Dependency Projection (`DEPENDS_ON`)

Structural edges do not directly reflect failure propagation: a subscriber depends on a publisher, yet no direct edge joins them in pub-sub topologies. SaG derives an explicit semantic relation, `DEPENDS_ON`, directed from *dependent* to *dependency* (“if the target fails, the source is impacted”), via the rules of Table 2.

**Table 2.** The `DEPENDS_ON` logical dependency projection rules. $^\dagger$Defined for completeness; not exercised by the Application-level evaluation.

|    **Rule**     | **Dependency Category** | **Structural Pattern ($\text{Dependent} \to \text{Dependency}$)**           | **Derived Weight ($w$)**                                                 |
|:---------------:|:------------------------|:----------------------------------------------------------------------------|:-------------------------------------------------------------------------|
|      **1**      | `app_to_app`            | Subscriber $\to$ Publisher (via shared Topic)                               | $1 - \prod_{t \in T}(1 - w(t))$                                          |
| **2**$^\dagger$ | `app_to_broker`         | Publisher/Subscriber $\to$ Broker routing its topics                        | $1 - \prod_{t \in T}(1 - w(t))$                                          |
| **3**$^\dagger$ | `host_to_host`          | Host $\to$ Host (lifted from inter-host app dependencies)                   | $\max_{u \in \text{hosted}(h_1), v \in \text{hosted}(h_2)} w_E(u \to v)$ |
| **4**$^\dagger$ | `host_to_broker`        | Host $\to$ Broker (lifted from hosted app dependencies)                     | $\max_{u \in \text{hosted}(h)} w_E(u \to b)$                             |
|      **5**      | `app_to_lib`            | Application $\to$ Shared Library it `USES`                                  | $H(w_V(\text{app}), w_V(\text{lib}))$                                    |
| **6**$^\dagger$ | `broker_to_broker`      | Broker $\leftrightarrow$ Broker (shared fault-domain colocation, symmetric) | $w_V(\text{host})$                                                       |

Rules 1 and 2 combine topics $T$ joining a pair by probabilistic union [67], ensuring parallel failure paths increase coupling. Rule 5 uses the harmonic mean $H(x, y) = 2xy/(x+y)$ [68], and Rules 3 and 4 lift dependencies to hosts by maximum.

**Sequential cascades and simultaneous blasts.** Rule 1 captures sequential cascades, where a failed publisher starves subscribers through queues and buffers. Rule 5 captures simultaneous blasts, where a crashed library takes down all dependent applications at once. Libraries that publish or subscribe are endpoints of Rule 1 in their own right, and an Application reaches a library’s topics transitively through Rule 5. The projection on which every count, `Topo-QoS` and every `-P` learner is computed applies Rule 1 to direct subscriptions only; the repository derivation that produces the node features additionally follows up to three `USES` hops, so an Application subscribing through a library it uses is a Rule 1 dependent there. Rules 2, 3, 4 and 6 represent infrastructural and broker dependencies (Table 2).

**Remark 1 (the dependency count is a typed two-hop count).** Let $G_{\text{flow}}$ be the Application–Library projection (Rules 1 and 5) and $v$ an Application. By construction, $$\tag{2}
\texttt{InDeg}(v) \;=\; \bigl|\{\,u \neq v : \exists t \in V_{\text{topic}},\; (v, t) \in \texttt{PUBLISHES\_TO} \wedge (u, t) \in \texttt{SUBSCRIBES\_TO}\,\}\bigr|.$$ Since Rule 5 edges end at Libraries, every edge into an Application is a Rule 1 edge $u \to v$, which exists exactly when $u \neq v$ (an Application or a Library) directly subscribes to a topic that $v$ publishes; parallel topics collapse into one edge.

The right-hand side of Eq. 2 is publish–subscribe afferent coupling (fan-in, the Absolute Importance of the Service [26, 27]), and it is a typed two-hop query on the raw multigraph. `InDeg` therefore needs no projection to compute; the projection states which typed query to ask. The equality is also checked on all seventeen corpus graphs (maximum absolute difference $0$). What the projection adds beyond this query is measured separately: Rule 5 raises transitive reach by $+0.058$ (§6.1). Because the primary oracle’s first propagation wave is exactly this set (§4.4), `InDeg` and its transitive counterpart `Reach` are used in this paper as references that restate the oracle, not as predictors.

![Figure 2](../latex/figures/Figure_2.png)

*Figure 2. Running example. (a) Three applications share topic t (routed by broker b) and library ℓ, and all run on host n. No structural edge joins two applications. (b) The derived DEPENDS_ON edges make the hidden dependencies explicit: subscribers a2, a3 depend on publisher a1 (Rule 1), all applications depend on ℓ (Rule 5), and each application depends on the broker (Rule 2). Simulation oracles run on view (a); predictors read view (b).*

## 3.4 Dual Graph Views

The **structural graph** $G_{\text{structural}}$ is the raw deployment topology. The **analysis graph** $G_{\text{analysis}}$ adds the derived `DEPENDS_ON` edges and code metrics (Figure 2). Predictor features are computed on $G_{\text{analysis}}$, while simulation oracles run strictly on $G_{\text{structural}}$ (§4.4). The two views also differ for a learner: on $G_{\text{structural}}$ every relation points away from Applications, so forward message passing never reaches them, whereas on the `DEPENDS_ON` graph each Application receives messages from its dependents (§6.2).

## 3.5 Typed Node Feature Encoding

Both the predictive pathway (§4) and the proposed explanation layer read typed node properties from $G_{\text{analysis}}$. All five entity types share an 18-dimensional base block of normalized topological metrics ($[0, 1]$): PageRank, Reverse PageRank, betweenness, closeness, eigenvector centrality, in- and out-degree, clustering, articulation, bridge ratio, the node QoS weight, incoming and outgoing dependency weights, multi-path coupling, path complexity, fan-out criticality, and the Connectivity Degradation Index (CDI). Four centrality features (PageRank, Reverse PageRank, betweenness and closeness) are computed over weighted edges and so carry QoS information; the nominal QoS-off condition is therefore not completely QoS-free. The in-degree feature, and the incoming dependency weight $w_{\text{in}}$, a QoS-weighted version of it, mean that every learner is given a quantity closely related to `InDeg` as an input, so the learners inherit part of the reference counts’ overlap with the oracle through their features (§6.2); a learner without both columns was not run. Type-specific blocks extend the vector to 19–25 dimensions (full schema documented in the replication repository). CDI evaluates structural connectivity loss via a fixed-size breadth-first sample to control computation time (§6.4); its sensitivity to hash-seeded tie breaking is examined in §7.3.
