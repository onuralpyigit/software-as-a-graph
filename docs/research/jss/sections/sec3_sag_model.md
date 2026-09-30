# 3. The Software-as-a-Graph (SaG) Architectural Model

Figure 1 shows the full SaG framework architecture. The shared front end handles Architecture-as-Code manifests, builds a typed multigraph, derives a logical dependency layer that makes runtime failure paths explicit, and extracts typed node properties. This representation supports graph-learning pipelines and makes software dependencies explicit, so analytical, hybrid, and learned cascade-impact rankings can be computed from the same architectural representation (§§4 and 5.2). This common representation enables controlled comparisons between methods that differ in complexity but operate on identical dependency information. The proposed explanation layer reads the same properties separately, with no shared parameters.

![Figure 1](../latex/figures/Figure_1.png)

*Figure 1. End-to-end architecture of the SaG framework. The ranking pathway runs down the center: manifest ingestion, typed structural multigraph (Gstructural), DEPENDS_ON logical dependency projection (Ganalysis) with typed node properties, the ranking methods (analytical, hybrid, and learned; Figure 3), and the ranked critical set. The same dependency graph supports all three ranking methods. The dashed edge marks the simulation oracles, which operate strictly on Gstructural, provide training and evaluation labels offline, and do not participate in inference. The proposed explanation layer (not evaluated) reads Ganalysis, shares no parameters with the rankers, and applies to components after they have been ranked; no ranker output flows into it.*

## 3.1 Multigraph Definition

A distributed system is described as a typed, weighted, directed multigraph:

$$\tag{1}
\mathcal{G} = (V, E, \tau_V, \tau_E, w_V, w_E)$$

where:

-   $V = V_{\text{app}} \cup V_{\text{broker}} \cup V_{\text{topic}} \cup V_{\text{host}} \cup V_{\text{lib}}$ holds the five entity types $\mathcal{T}_V$ of Table 1; $V_{\text{host}}$ denotes physical or virtual *Execution Hosts*.

-   $E$ is the set of directed edges, and $\tau_V: V \to \mathcal{T}_V$ and $\tau_E: E \to \mathcal{T}_E$ assign entity and relation types.

-   $w_V: V \to (0, 1]$ and $w_E: E \to (0, 1]$ weight entity criticality along with connection strength. For Applications and Libraries, $w_V(v) = 1 - \text{CQP}(v)$, where CQP is a code-quality penalty computed from static code metrics (lines of code, cyclomatic complexity, coupling and cohesion); otherwise $w_V(v) = 1.0$.

The raw multigraph is denoted $G_{\text{structural}}$ (used exclusively by simulation oracles), while $G_{\text{analysis}}$ denotes the logical `DEPENDS_ON` projection from which all rankers, analytical and learned, are computed. A comprehensive reference of mathematical notation is provided in the replication repository.

**Scope of Evaluation and Entity Roles.** While the multigraph formalizes five entity types, this study concentrates its empirical study specifically on *Applications* ($V_{\text{app}}$). In this work, the term *Application* is adopted to align with publish–subscribe standards (such as OMG DDS and ROS 2 participant nodes); conceptually, it corresponds directly to individual microservice containers, autonomous service agents, or communicating daemon processes in distributed architectures. Applications represent the developer-authored, code-level services that are often changed, refactored, and tested in CI/CD pipelines, so they are the main focus for pre-deployment triage. Brokers, Topics, Execution Hosts, and Libraries are also modeled and included because they are important for understanding indirect failure propagation. For example, we cannot analyze message routing, shared fault-domain colocation, or library-related failures without them. However, ranking infrastructure components like Brokers and Hosts is more relevant to operational provisioning than to software triage at commit time, so we treat this as a separate operational issue. Consequently, the empirical evaluation in this paper focuses strictly on ranking developer-authored Applications ($V_{\text{app}}$); evaluating broker queue bottlenecks and physical host outages remains an important operational extension (§7.6).

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

A link’s strength is conceptually determined by its Quality-of-Service (QoS) contract. For example, a `RELIABLE` topic with `TRANSIENT_LOCAL` durability creates a stronger connection between services than a `BEST_EFFORT` telemetry stream. Each topic $t$ has an overall weight $w(t) \in (0, 1]$ that combines its QoS policies (reliability, durability, priority) with payload size and publication frequency. We set the reliability, durability, and priority weights employing an Analytic Hierarchy Process (AHP) pairwise-comparison matrix, which we defined independently and did not adjust to fit a target vector ($CR = 0.016$, non-degenerate; the framework’s other AHP matrices also use a declared vector).

Although we formalize this multi-criteria weighting to capture declared middleware contracts, the evaluation in §6 tests whether this contractual information improves cascade ranking. It does not: on the reachability and queue-flow simulators, declared QoS policy parameters carried no measurable signal. On the reachability oracle, more than half of the apparent QoS effect came from the QoS-weighted in-degree feature, and the remainder was not significant (§6.2). On the queue-flow oracle, the gains came from declared publication rates and payload sizes, while QoS-policy shares added $-0.005$ ($p = 0.68$; §6.1). We retain the full AHP formalization for architectural completeness and reproducibility; in this corpus, omitting QoS policy weighting did not reduce ranking accuracy. The dependency counts reported in this paper are unweighted and do not use $w(t)$.

## 3.3 Logical Dependency Projection (`DEPENDS_ON`)

Structural edges do not directly show how failures spread. For example, a subscriber depends on a publisher, but pub-sub topologies have no direct edge between them. SaG therefore derives an explicit semantic relation, `DEPENDS_ON`, from the dependent to the dependency: if the target fails, the source is affected. The derived edges expose the architectural pathways through which failures may propagate, and the resulting dependency graph is the common input for analytical rankings, hybrid methods, and graph-learning models. Table 2 lists the rules.

**Table 2.** The `DEPENDS_ON` logical dependency projection rules. $^\dagger$Defined for completeness; not exercised by the Application-level evaluation.

|    **Rule**     | **Dependency Category** | **Structural Pattern ($\text{Dependent} \to \text{Dependency}$)**           | **Derived Weight ($w$)**                                                 |
|:---------------:|:------------------------|:----------------------------------------------------------------------------|:-------------------------------------------------------------------------|
|      **1**      | `app_to_app`            | Subscriber $\to$ Publisher (via shared Topic)                               | $1 - \prod_{t \in T}(1 - w(t))$                                          |
| **2**$^\dagger$ | `app_to_broker`         | Publisher/Subscriber $\to$ Broker routing its topics                        | $1 - \prod_{t \in T}(1 - w(t))$                                          |
| **3**$^\dagger$ | `host_to_host`          | Host $\to$ Host (lifted from inter-host app dependencies)                   | $\max_{u \in \text{hosted}(h_1), v \in \text{hosted}(h_2)} w_E(u \to v)$ |
| **4**$^\dagger$ | `host_to_broker`        | Host $\to$ Broker (lifted from hosted app dependencies)                     | $\max_{u \in \text{hosted}(h)} w_E(u \to b)$                             |
|      **5**      | `app_to_lib`            | Application $\to$ Shared Library it `USES`                                  | $H(w_V(\text{app}), w_V(\text{lib}))$                                    |
| **6**$^\dagger$ | `broker_to_broker`      | Broker $\leftrightarrow$ Broker (shared fault-domain colocation, symmetric) | $w_V(\text{host})$                                                       |

Rules 1 and 2 combine topics $T$ that connect a pair using probabilistic union [67], so that having multiple failure paths increases the connection strength. Rule 5 uses the harmonic mean $H(x, y) = 2xy/(x+y)$ [68]. Rules 3 and 4 assign dependencies to hosts by taking the maximum value.

**Sequential cascades and simultaneous blasts.** Rule 1 describes sequential cascades, where if a publisher fails, its subscribers are affected because they do not receive messages. Rule 5 describes simultaneous blasts, where if a library crashes, all applications that depend on it fail at the same time. Neither mechanism is stated explicitly in a deployment manifest; each rule makes one of them visible. Shared-library dependencies are included because they create simultaneous blast failures that message-flow topology alone does not represent. Libraries that publish or subscribe are also endpoints for Rule 1, and an Application can reach a library’s topics through Rule 5. For counting, `Topo-QoS` and all `-P` learners use Rule 1 only for direct subscriptions. The repository’s method for creating node properties also follows up to three `USES` links, so if an Application subscribes through a library it uses, it is considered a Rule 1 dependent. Rules 2, 3, 4, and 6 cover infrastructure and broker dependencies (see Table 2), which define physical host failure domains and broker routing bottlenecks. Since this evaluation focuses on Application-level code, these four rules are not used in the ranking benchmarks but are included for completeness.

**Remark 1 (the dependency count is a typed two-hop count).** Let $G_{\text{flow}}$ be the Application–Library projection (Rules 1 and 5) and $v$ an Application. By construction, $$\tag{2}
\texttt{InDeg}(v) \;=\; \bigl|\{\,u \neq v : \exists t \in V_{\text{topic}},\; (v, t) \in \texttt{PUBLISHES\_TO} \wedge (u, t) \in \texttt{SUBSCRIBES\_TO}\,\}\bigr|.$$ Since Rule 5 edges end at Libraries, every edge into an Application is a Rule 1 edge $u \to v$, which exists exactly when $u \neq v$ (an Application or a Library) directly subscribes to a topic that $v$ publishes; in the presence of multiple parallel topics between the same pair, parallel topics collapse into a single directed dependency edge in $G_{\text{flow}}$.

The right side of Eq. 2 represents publish–subscribe afferent coupling (fan-in, or the Absolute Importance of the Service [47, 48]), and is a typed two-hop query on the raw multigraph. `InDeg` can be computed directly without projection; the projection defines which typed query to use. We checked this equality on all seventeen corpus graphs, with a maximum absolute difference of $0$. We measure the value of the derivation separately: including Rule 5 raises the agreement of transitive reach with $I^*$ by $+0.058$ (9 of 12 folds, Holm $p = 0.0068$; §6.1). Since the primary oracle’s first propagation wave matches this set (§4.4), `InDeg` and its transitive version `Reach` are used as reference points in this paper, not as predictors.

![Figure 2](../latex/figures/Figure_2.png)

*Figure 2. Running example. (a) Three applications share topic t (routed by broker b) and library ℓ, and all run on host n. No structural edge joins two applications. (b) The derived DEPENDS_ON edges make the hidden dependencies explicit: subscribers a2, a3 depend on publisher a1 (Rule 1), all applications depend on ℓ (Rule 5), and each application depends on the broker (Rule 2). Simulation oracles run on view (a); all rankers, analytical and learned, read view (b).*

## 3.4 Dual Graph Views

The **structural graph** $G_{\text{structural}}$ is the raw deployment topology. The **analysis graph** $G_{\text{analysis}}$ adds the `DEPENDS_ON` edges and code metrics (see Figure 2). All ranker inputs are computed on $G_{\text{analysis}}$, while simulation oracles run strictly on $G_{\text{structural}}$ (§4.4). The two views also differ for a learner: on $G_{\text{structural}}$, every relation points away from Applications, so forward message passing does not reach them. In the `DEPENDS_ON` graph, each Application receives messages from its dependents (§6.2). The raw multigraph is therefore also a comparison point in the evaluation: it exposes communication structure but not cascading dependencies, which allows the effect of the representation to be separated from the effect of the learning architecture (§6.2).

## 3.5 Typed Node Feature Encoding

Both the ranking pathway (§4) and the proposed explanation layer use typed node properties from $G_{\text{analysis}}$. All five entity types share a base set of 18 normalized topological metrics (ranging from $[0, 1]$): PageRank, Reverse PageRank, betweenness, closeness, eigenvector centrality, in-degree, out-degree, clustering, articulation, bridge ratio, node QoS weight, incoming and outgoing dependency weights, multi-path coupling, path complexity, fan-out criticality, and the Connectivity Degradation Index (CDI). We calculate four centrality features (PageRank, Reverse PageRank, betweenness, and closeness) over weighted edges, so they include QoS information. This means the so-called QoS-off condition is not entirely free of QoS effects. The in-degree feature and the incoming dependency weight $w_{\text{in}}$ (a QoS-weighted version) ensure that all learners receive information closely related to `InDeg`, so the learners inherit some overlap with the oracle’s reference counts (§6.2). We did not use any learner missing both columns. Type-specific blocks add more features, extending the vector to 19–25 dimensions (the full schema is in the replication repository). CDI measures structural connectivity loss using a fixed-scale breadth-first sample to keep computation time reasonable (§6.4). We discuss its sensitivity to hash-seeded tie-breaking in §7.5.

In this study, SaG therefore serves two roles. It provides a dependency representation from which analytical rankings can be computed directly, and it provides a graph on which learned and hybrid methods operate. The evaluation in §§5–7 examines how much ranking performance arises from this representation and how much from the models applied to it.
