# 1. Introduction

## 1.1 Motivation and Problem

Large distributed systems increasingly communicate through asynchronous publish–subscribe (pub-sub) middleware: ROS 2 in autonomous driving [1], Apache Kafka in enterprise event streams [2], DDS in cyber-physical systems [3], MQTT in IoT [4], and cloud-native microservices [5, 6]. Pub-sub decouples producers and consumers in space, time and synchronization [7]. Components interact through topics and brokers rather than direct references, and deployment-time Quality-of-Service (QoS) policies govern reliability, durability, priority and deadlines.

The same decoupling hides how failures spread. Publishers and subscribers share no direct link, so outages, head-of-line blocking and backpressure propagate along concealed paths through brokers, shared topics, colocated hosts and shared libraries [8, 9]. These failures take two forms. In *sequential cascades*, a slow subscriber fills a broker queue and gradually starves its publishers [10]. In *simultaneous blasts*, a shared-library crash or host outage takes down every colocated service at once. Neither architecture diagrams nor static call graphs show these mechanisms. The cheapest time to reduce the risk is before deployment, at design and continuous-integration time [11, 12], when no runtime telemetry exists. Architects therefore need to know, from configuration manifests alone, which components, topics and links are systemically critical and why.

Existing practice leaves this gap open, which we call the **Architecture–Code Gap**: a system can have bug-free code in every service and still be fragile through hidden single points of failure or mismatched QoS contracts [13, 14]. Architecture evaluations such as ATAM rely on manual elicitation [15]. Static code analysis inspects services in isolation [16, 17]. Chaos engineering [18] needs a provisioned cluster, while pre-production alternatives like service-level fault injection testing [19] and microservice dependency tracing [20] require runnable execution environments. Homogeneous centrality flattens typed topologies into untyped graphs [21, 22]. Learned models, graph neural networks in particular, could combine these structural cues. What is missing is a representation that makes pub-sub failure paths explicit, and evidence on which analyzer to trust on it: whether graph learning adds anything over the coupling metrics that architecture research already has. This paper provides that evidence for one well-defined task, ranking components by simulated cascade impact before deployment.

## 1.2 The Software-as-a-Graph (SaG) Approach

**Software-as-a-Graph (SaG)** is a pre-deployment static analysis framework for event-driven architectures (Figure 1). It (1) models an architecture as a typed, directed multigraph over Applications, Brokers, Topics, Execution Hosts and Shared Libraries (§3.1); (2) derives from it, through publish–subscribe rules, an explicit `DEPENDS_ON` dependency graph (§3.3); and (3) ranks components by predicted cascade impact with training-free rankers, graph-learning engines, and hybrid engines in which a learned model corrects a closed-form score (§4). Predictors read only the analysis graph. Ground truth comes from three simulators that run on the raw structural topology (§4.3).

**What the ground truth is, and is not.** All ground truth in this study is simulated. The primary oracle $I^*$ is a reachability simulation that runs on the same manifest the predictors read, and its first propagation wave is exactly the dependency count this paper recommends (§4.4). The study therefore measures agreement between cheap static rankers and simulators, not the prediction of observed outages. We bound the circularity with two further simulators and with partial correlations, and report the cost of running $I^*$ itself (§7.4); validation against observed failures remains open (§8.3).

**Findings in brief.** The registered primary contrast, a heterogeneous graph transformer against QoS-weighted centrality, was null ($+0.069$, $p = 0.266$). In exploratory analyses registered after it, the simplest ranker won: the number of a component’s direct dependents (`InDeg`), which is publish–subscribe afferent coupling [23, 24], ranks $I^*$ at $\rho = 0.764$ and beats centrality on all twelve synthetic architectures. It is a typed two-hop query on the raw graph (Proposition 1): untyped raw-graph metrics do not recover it, but merely counting the topics a component publishes comes within $0.033$ of it. Graph attention networks on the dependency graph match the count ($0.748$) but never exceed it. On a queue-flow simulator the count keeps a partial correlation of $0.259$ after the reachability oracle’s rank is removed (on a 30-Application sample per fold); on a multi-criteria simulator QoS-weighted centrality ranks above it. Flagging enough components to catch $80\%$ of the true top-$20\%$ set takes the top $45\%$ by the count. The result is a benchmark with a largely negative outcome for graph learning on this task.

## 1.3 Research Questions

-   **RQ1 (Ranking accuracy):** *How accurately do SaG’s closed-form, learned and hybrid engines rank components by cascading-failure impact on unseen architectures, compared with structural baselines and with dependency counts on the derived graph?*

-   **RQ2 (What learning needs):** *What do learned engines need: which graph they read, relation-specific (typed) parameters, or QoS inputs, once model capacity and edge-channel width are matched?*

-   **RQ3 (Transfer):** *How well do engines trained on synthetic architectures transfer zero-shot to independently authored models inspired by five open-source systems?*

-   **RQ4 (Cost):** *What does the analysis cost at CI/CD time, which stage dominates, and how does it compare with running the simulation directly?*

The evaluation follows a registered protocol. Only the primary contrast, the matched $2\times2$ and the two hybrid contrasts are confirmatory; the dependency counts, the dependency-graph learners, the raw-graph baselines and the cross-oracle analyses were registered in later amendments, after the primary result was known, and are reported as exploratory (§6.3). Supplementary §S25 logs every amendment and maps these four research questions onto the registered plan.

## 1.4 Contributions

Evaluated under leave-one-scenario-out (LOSO) cross-validation over twelve synthetic architectures and on five hand-authored models of open-source systems, this paper contributes:

1.  **A benchmark of static rankers for pre-deployment cascade-impact ranking in publish–subscribe systems** (§§6–7): seventeen architectures (2,812 components), three simulation oracles, and nineteen training-free, learned and hybrid rankers plus a first-order reference, with a registered analysis plan, its amendments, and artifacts that regenerate byte-identically.

2.  **A largely negative result for graph learning on this task** (§§7.1–7.2). Afferent coupling, computed as a typed two-hop count, ranks the reachability oracle at $\rho = 0.764$ and beats closed-form centrality on 12/12 architectures ($+0.211$); graph neural networks gain $+0.08$ to $+0.11$ by reading the dependency graph instead of the raw multigraph, reaching $0.748$, but never exceed the count, and on the raw multigraph they act as per-node models. Relation typing adds nothing at matched capacity (main effect $-0.014$), and in the single untuned configuration tested the heterogeneous transformer is unstable across seeds.

3.  **A bounded account of what the count measures** (§§4.4 and 7.1). Its agreement with the reachability oracle is largely construction, and the paper says so; on a queue-flow simulator it retains $\rho = 0.610$, and a partial correlation of $0.259$ $[0.143, 0.367]$ beyond the reachability oracle, on a 30-Application sample per fold; on a multi-criteria simulator it is outranked by QoS-weighted centrality ($0.650$ vs. $0.702$). Its critical-set recall is moderate: the top $45\%$ is needed to catch $80\%$ of the true top $20\%$.

4.  **Explicit dependency rules for publish–subscribe architectures** (§3), which state the typed query behind afferent coupling and add library-mediated blasts; the library rule raises transitive reach by $+0.058$.

5.  **A cost profile** (§7.4): on the corpus the counting path (projection plus count) takes $1$–$29$ ms per architecture against $0.08$–$4.5$ s for one run of the reachability oracle, and neural feature extraction costs median $5.6\times$ the oracle.

A previous conference paper [25] introduced the preliminary multigraph and deterministic quality model on synthetic topologies. This paper adds the dependency rules’ evaluation, the dependency counts, the learned and hybrid engines, the QoS edge encoding, LOSO and zero-shot evaluation, the matched control, the cross-oracle analyses and the cost profile; the earlier quality model survives only as the proposed explanation layer of §5.

§2 reviews related work, §§3–5 present the model, the rankers and the proposed explanation layer, §§6–7 the evaluation, §8 the discussion and threats, and §9 concludes.
