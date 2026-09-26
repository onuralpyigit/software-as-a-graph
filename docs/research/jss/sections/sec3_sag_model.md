# 3. The Software-as-a-Graph (SaG) Architectural Model

Figure 1 presents the end-to-end architecture of the SaG framework. The shared front end processes Architecture-as-Code manifests, constructs a typed multigraph, projects implicit runtime interactions throughout a logical dependency layer, and extracts typed node properties. These features feed the ranking engines (§4) and, separately and with no shared parameters, the explanation layer (§5).

![Figure 1](../latex/figures/Figure_1.png)

*Figure 1. End-to-end architecture of the SaG framework. The predictive pathway runs down the centre: manifest ingestion, typed multigraph, DEPENDS_ON projection with typed node properties, the ranking engines (closed-form, learned and hybrid; Figure 3), and the ranked critical set. The dashed edge marks the ground-truth simulation oracles, which operate only on Gstructural, train the predictor offline and take no part in inference. The proposed explanation layer (§5, not evaluated) reads the same analysis multigraph, shares no parameters with the predictor, and is applied to components after they have been ranked; no output of the predictor flows into it.*

## 3.1 Multigraph Definition

A distributed system is described as a typed, weighted, directed multigraph:

$$\tag{1}
\mathcal{G} = (V, E, \tau_V, \tau_E, w_V, w_E)$$

where:

-   $V = V_{\text{app}} \cup V_{\text{broker}} \cup V_{\text{topic}} \cup V_{\text{host}} \cup V_{\text{lib}}$ holds the five entity types $\mathcal{T}_V$ of Table 1; $V_{\text{host}}$ denotes physical or virtual *Execution Hosts*.

-   $E$ is the set of directed edges, and $\tau_V: V \to \mathcal{T}_V$ and $\tau_E: E \to \mathcal{T}_E$ assign entity and relation types.

-   $w_V: V \to (0, 1]$ and $w_E: E \to (0, 1]$ weight entity criticality and connection strength. For Applications and Libraries, $w_V(v) = 1 - \text{CQP}(v)$, where CQP is a code-quality penalty computed from static code metrics (lines of code, cyclomatic complexity, coupling and cohesion); otherwise $w_V(v) = 1.0$.

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

**Table 2.** Notation used throughout the paper.

| **Symbol**              | **Description**                                                   |
|:------------------------|:------------------------------------------------------------------|
| $G_{\text{structural}}$ | Raw multigraph (used exclusively by simulation oracles)           |
| $G_{\text{analysis}}$   | Logical `DEPENDS_ON` projection (input to predictors)             |
| $V_{\text{app}}$        | Application nodes (the primary scored population)                 |
| $w(t)$, $w(e)$          | QoS topic weight and derived dependency edge weight               |
| $I^*(v)$                | Primary cascade-reachability simulation oracle                    |
| $I_{\text{dyn}}(v)$     | Dynamic queue-flow discrete-event simulation oracle               |
| $I_{\text{comp}}(v)$    | Multi-criteria composite simulation oracle                        |
| $Q(v)$                  | Reliability–Maintainability composite quality score               |
| $\rho$                  | Spearman rank correlation coefficient (full population)           |
| $\rho_{>0}$             | Spearman rank correlation restricted to active stratum ($I > 0$)  |
| Overlap@$K$             | Top-$K$ identification set overlap ($K = 0.20\,|V_{\text{app}}|$) |

## 3.2 Quality-of-Service Link Weighting

A link’s strength depends on its Quality-of-Service (QoS) contract: a `RELIABLE` topic with `TRANSIENT_LOCAL` durability couples services more strongly than a `BEST_EFFORT` telemetry stream. Each topic $t$ carries an aggregate weight $w(t) \in (0, 1]$ combining declared QoS policies (reliability, durability, priority) with payload size and publication frequency. The sub-weights of reliability, durability and priority come from an Analytic Hierarchy Process (AHP) pairwise-comparison matrix that was stated independently rather than back-solved from a target vector ($CR = 0.016$, non-degenerate; Supplementary §S4, which also shows that three of the framework’s other AHP matrices do encode a declared vector).

On the reachability oracle, QoS weighting does not improve closed-form ranking: unweighted betweenness on the Application–Library projection scores $\rho = 0.591$, against $0.553$ for QoS-weighted betweenness (`Topo-QoS`; §7.1). On the multi-criteria oracle the order reverses, and `Topo-QoS` is the best training-free ranker (§7.1). The dependency counts evaluated in this paper are unweighted and do not read $w(t)$.

## 3.3 Logical Dependency Projection (`DEPENDS_ON`)

Structural edges do not directly reflect failure propagation: a subscriber depends on a publisher, yet no direct edge joins them in pub-sub topologies. SaG derives an explicit semantic relation, `DEPENDS_ON`, directed from *dependent* to *dependency* (“if the target fails, the source is impacted”), via the rules of Table 3.

**Table 3.** The `DEPENDS_ON` logical dependency projection rules.

| **Rule** | **Dependency Category** | **Structural Pattern ($\text{Dependent} \to \text{Dependency}$)**           | **Derived Weight ($w$)**                                                                    |
|:--------:|:------------------------|:----------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|
|  **1**   | `app_to_app`            | Subscriber $\to$ Publisher (via shared Topic, incl. transitive `USES`)      | $1 - \prod_{t \in T}(1 - w(t))$                                                             |
|  **2**   | `app_to_broker`         | Publisher/Subscriber $\to$ Broker routing its topics                        | $1 - \prod_{t \in T}(1 - w(t))$                                                             |
|  **3**   | `host_to_host`          | Host $\to$ Host (lifted from inter-host app dependencies)                   | $\max_{u \in \text{hosted}(h_1), v \in \text{hosted}(h_2)} w_{\text{DEPENDS\_ON}}(u \to v)$ |
|  **4**   | `host_to_broker`        | Host $\to$ Broker (lifted from hosted app dependencies)                     | $\max_{u \in \text{hosted}(h)} w_{\text{DEPENDS\_ON}}(u \to b)$                             |
|  **5**   | `app_to_lib`            | Application $\to$ Shared Library it `USES`                                  | $H(w_V(\text{app}), w_V(\text{lib}))$                                                       |
|  **6**   | `broker_to_broker`      | Broker $\leftrightarrow$ Broker (shared fault-domain colocation, symmetric) | $w_V(\text{host})$                                                                          |

Rules 1 and 2 combine topics $T$ joining a pair by probabilistic union [88, 89, 90], ensuring parallel failure paths increase coupling. Rule 5 uses the harmonic mean $H(x, y) = 2xy/(x+y)$ [91], and Rules 3 and 4 lift dependencies to hosts by maximum.

**Sequential cascades and simultaneous blasts.** Rule 1 captures sequential cascades, where a failed publisher starves subscribers through queues and buffers. Rule 5 captures simultaneous blasts, where a crashed library takes down all dependent applications at once. Libraries that publish or subscribe are endpoints of Rule 1 in their own right, and an Application reaches a library’s topics transitively through Rule 5. Rules 2, 3, 4 and 6 represent infrastructural and broker dependencies; they are defined for completeness but are not exercised by the Application-level evaluation in this paper.

**Proposition 1 (dependency count = typed two-hop count).** Let $G_{\text{flow}}$ be the Application–Library projection (Rules 1 and 5) and $v$ an Application. Then $$\tag{2}
\texttt{InDeg}(v) \;=\; \bigl|\{\,u \neq v : \exists t \in V_{\text{topic}},\; (v, t) \in \texttt{PUBLISHES\_TO} \wedge (u, t) \in \texttt{SUBSCRIBES\_TO}\,\}\bigr|.$$ *Proof.* Rule 5 edges end at Libraries, so every edge into an Application is a Rule 1 edge $u \to v$, which exists exactly when $u \neq v$ subscribes to a topic that $v$ publishes; parallel topics collapse into one edge. $\square$

The right-hand side of Eq. 2 is publish–subscribe afferent coupling (fan-in, the Absolute Importance of the Service [23, 24]), and it is a typed two-hop query on the raw multigraph. `InDeg` therefore needs no projection to compute; the projection states which typed query to ask. The equality is also checked on all seventeen corpus graphs (maximum absolute difference $0$). What the projection adds beyond this query is measured separately: Rule 5 raises transitive reach by $+0.058$ (§7.1).

![Figure 2](../latex/figures/Figure_2.png)

*Figure 2. Running example. (a) Three applications share topic t (routed by broker b) and library ℓ, and all run on host n. No structural edge joins two applications. (b) The derived DEPENDS_ON edges make the hidden dependencies explicit: subscribers a2, a3 depend on publisher a1 (Rule 1), all applications depend on ℓ (Rule 5), and each application depends on the broker (Rule 2). Simulation oracles run on view (a); predictors read view (b).*

## 3.4 Dual Graph Views

The **structural graph** $G_{\text{structural}}$ is the raw deployment topology. The **analysis graph** $G_{\text{analysis}}$ adds the derived `DEPENDS_ON` edges and code metrics (Figure 2). Predictor features are computed on $G_{\text{analysis}}$, while simulation oracles run strictly on $G_{\text{structural}}$ (§4.4).

## 3.5 Typed Node Feature Encoding

Both the predictive pathway (§4) and the explanation layer (§5) read typed node properties from $G_{\text{analysis}}$. All five entity types share an 18-dimensional base block of normalized topological metrics ($[0, 1]$): PageRank, Reverse PageRank, betweenness, closeness, eigenvector centrality, in- and out-degree, clustering, articulation, bridge ratio, the node QoS weight, incoming and outgoing dependency weights, multi-path coupling, path complexity, fan-out criticality, and the Connectivity Degradation Index (CDI). Four centrality features (PageRank, Reverse PageRank, betweenness and closeness) are computed over weighted edges and so carry QoS information; the nominal QoS-off condition is therefore not completely QoS-free. The in-degree feature means that every learner is given a quantity closely related to `InDeg` as an input (§7.2). Type-specific blocks extend the vector to 19–25 dimensions (full schema in Supplementary §S11). CDI evaluates structural connectivity loss via a fixed-size breadth-first sample to control computation time (§7.4); its sensitivity to hash-seeded tie breaking is examined in §8.3.
