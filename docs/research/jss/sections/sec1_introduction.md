# 1. Introduction

## 1.1 Motivation and Problem

Large distributed systems increasingly communicate through asynchronous publish–subscribe (pub-sub) middleware: ROS 2 in autonomous driving [1], Apache Kafka in enterprise event streams [2], DDS in cyber-physical systems [3], MQTT in IoT [4], and cloud-native microservices [5, 6]. Pub-sub decouples producers and consumers in space, time and synchronization [7]. Components interact through topics and brokers rather than direct references, and deployment-time Quality-of-Service (QoS) policies govern reliability, durability, priority and deadlines.

The same decoupling hides how failures spread. Publishers and subscribers share no direct link, so outages, head-of-line blocking and backpressure propagate along concealed paths through brokers, shared topics, colocated hosts and shared libraries [8, 9]. These failures take two forms. In *sequential cascades*, a slow subscriber fills a broker queue and gradually starves its publishers [10]. In *simultaneous blasts*, a shared-library crash or host outage takes down every colocated service at once. Neither architecture diagrams nor static call graphs show these mechanisms. The cheapest time to reduce the risk is before deployment, at design and continuous-integration time [11, 12], when no runtime telemetry exists. Architects therefore need to know, from configuration manifests alone, which components, topics and links are systemically critical and why.

Existing practice leaves this gap open, which we call the **Architecture–Code Gap**: a system can have bug-free code in every service and still be fragile through hidden single points of failure or mismatched QoS contracts [13, 14]. Architecture evaluations such as ATAM rely on manual elicitation [15]. Static code analysis inspects services in isolation [16, 17]. Chaos engineering [18] needs a provisioned cluster, while pre-production alternatives like service-level fault injection testing [19] and microservice dependency tracing [20] require runnable execution environments. Homogeneous centrality flattens typed topologies into untyped graphs [21, 22]. Learned models, graph neural networks in particular, could combine these structural cues. What is missing is a representation that makes pub-sub failure paths explicit, and evidence on which analyzer to trust on it: whether graph learning adds anything over the coupling metrics that architecture research already has. This paper provides that evidence for one well-defined task, ranking components by simulated cascade impact before deployment.

## 1.2 The Software-as-a-Graph (SaG) Approach

**Software-as-a-Graph (SaG)** is a pre-deployment static analysis framework for event-driven architectures (Figure 1). It (1) models an architecture as a typed, directed multigraph over Applications, Brokers, Topics, Execution Hosts and Shared Libraries (§3.1); (2) derives from it, through publish–subscribe rules, an explicit `DEPENDS_ON` dependency graph (§3.3); and (3) ranks components by predicted cascade impact with training-free rankers, graph-learning engines, and hybrid engines in which a learned model corrects a closed-form score (§4). Predictors read only the analysis graph. Ground truth comes from three simulators that run on the raw structural topology (§4.3).

**What the ground truth is, and is not.** All ground truth in this study is simulated. The primary oracle $I^*$ is a reachability simulation that runs on the same manifest the predictors read, and its first propagation wave is exactly the number of a component’s direct dependents (§4.4). Dependency counts are therefore reported as references that restate the oracle, never as predictors. The study measures agreement between cheap static rankers and simulators, not the prediction of observed outages. Because ranking against $I^*$ rewards restating its propagation rule, we bound the circularity with two further simulators and with partial correlations, and report the cost of running $I^*$ itself (§7.4); validation against observed failures remains open (§8.3).

**Findings in brief.** The registered primary contrast, a heterogeneous graph transformer against QoS-weighted centrality, was null ($+0.069$, $p = 0.266$). The only confirmatory gains are two hybrids, learned engines corrected by the centrality prior, which beat centrality on 11 of 12 synthetic architectures ($+0.103$ and $+0.130$, Holm-significant). Rankings that restate $I^*$’s propagation rule set a reference level on it: the oracle’s first-order expansion reaches $\rho = 0.808$ and the number of direct dependents (`InDeg`, publish–subscribe afferent coupling [23, 24]) $0.764$. In exploratory analyses, graph attention networks on the dependency graph approach that reference ($0.748$) without exceeding it, and they receive an in-degree feature, so they do not escape the circularity either. On a multi-criteria simulator, QoS-weighted centrality ($0.702$) ranks above every learned engine. The result is a benchmark with a largely negative outcome for graph learning on this task.

## 1.3 Research Questions

-   **RQ1 (Ranking accuracy):** *How accurately do SaG’s closed-form, learned and hybrid engines rank components by cascading-failure impact on unseen architectures, compared with structural baselines, and how close do they come to reference rankings that restate the oracle’s propagation rule?*

-   **RQ2 (What learning needs):** *What do learned engines need: which graph they read, relation-specific (typed) parameters, or QoS inputs, once model capacity and edge-channel width are matched?*

-   **RQ3 (Transfer):** *How well do engines trained on synthetic architectures transfer zero-shot to independently authored models inspired by five open-source systems?*

-   **RQ4 (Cost):** *What does the analysis cost at CI/CD time, which stage dominates, and how does it compare with running the simulation directly?*

The evaluation follows a registered protocol. Only the primary contrast, the matched $2\times2$ and the two hybrid contrasts are confirmatory; the dependency-count references, the dependency-graph learners, the raw-graph baselines and the cross-oracle analyses were registered in later amendments, after the primary result was known, and are reported as exploratory (§6.3). Supplementary §S25 logs every amendment and maps these four research questions onto the registered plan.

## 1.4 Contributions

Evaluated under leave-one-scenario-out (LOSO) cross-validation over twelve synthetic architectures and on five hand-authored models of open-source systems, this paper contributes:

1.  **A benchmark of static rankers for pre-deployment cascade-impact ranking in publish–subscribe systems** (§§6–7): seventeen architectures (2,812 components), three simulation oracles, and training-free, learned and hybrid rankers, with dependency counts and the oracle’s first-order expansion reported as references that restate the oracle (Amendment 13), with a registered analysis plan, its amendments, and artifacts that regenerate byte-identically.

2.  **A largely negative result for graph learning on this task** (§§7.1–7.2). The registered primary contrast is null; only the two hybrids beat closed-form centrality under correction. Graph neural networks gain $+0.08$ to $+0.11$ by reading the dependency graph instead of the raw multigraph, reaching $0.748$, but never exceed a count that restates the oracle’s first wave ($0.764$), and on the raw multigraph they act as per-node models. Relation typing adds nothing at matched capacity (main effect $-0.014$), and in the single untuned configuration tested the heterogeneous transformer is unstable across seeds.

3.  **A bounded account of the oracle circularity** (§§4.4 and 7.1). Ranking against a reachability oracle rewards restating its propagation rule, which is why the dependency counts are reported as references and not as predictors. The paper measures how far the references reach on $I^*$, how much of them survives on a queue-flow simulator beyond $I^*$ (partial correlation $0.259$ $[0.143, 0.367]$ for the direct count, none for transitive reach, on a 30-Application sample per fold), and that on a multi-criteria simulator QoS-weighted centrality ranks above both the references and every learned engine.

4.  **Explicit dependency rules for publish–subscribe architectures** (§3), which state the typed query behind afferent coupling and add library-mediated blasts.

5.  **A cost profile** (§7.4): one run of the reachability oracle takes $0.08$–$4.5$ s per corpus architecture, and neural feature extraction costs median $5.6\times$ that, so where the oracle’s ranking is wanted, running it directly is cheaper than approximating it.

A previous conference paper [25] introduced the preliminary multigraph and deterministic quality model on synthetic topologies. This paper adds the dependency rules’ evaluation, the reference counts, the learned and hybrid engines, the QoS edge encoding, LOSO and zero-shot evaluation, the matched control, the cross-oracle analyses and the cost profile; the earlier quality model survives only as the proposed explanation layer of §5.

§2 reviews related work, §§3–5 present the model, the rankers and the proposed explanation layer, §§6–7 the evaluation, §8 the discussion and threats, and §9 concludes.
