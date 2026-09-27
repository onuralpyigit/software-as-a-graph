# 1. Introduction

## 1.1 Motivation and Problem

Large distributed systems increasingly communicate through asynchronous publish–subscribe (pub-sub) middleware: ROS 2 in autonomous driving [1], Apache Kafka in enterprise event streams [2], DDS in cyber-physical systems [3], MQTT in IoT [4], and cloud-native microservices [5, 6]. Pub-sub decouples producers and consumers in space, time and synchronization [7]. Components interact through topics and brokers rather than direct references, and deployment-time Quality-of-Service (QoS) policies govern reliability, durability, priority and deadlines.

The same decoupling hides how failures spread. Publishers and subscribers share no direct link, so outages, head-of-line blocking and backpressure propagate along concealed paths through brokers, shared topics, colocated hosts and shared libraries [8, 9]. These failures take two forms. In *sequential cascades*, a slow subscriber fills a broker queue and gradually starves its publishers [10]. In *simultaneous blasts*, a shared-library crash or host outage takes down every colocated service at once. Neither architecture diagrams nor static call graphs show these mechanisms. The cheapest time to reduce the risk is before deployment, at design and continuous-integration time [11, 12], when no runtime telemetry exists. Architects therefore need to know, from configuration manifests alone, which components, topics and links are systemically critical and why.

Existing practice leaves this gap open, which we call the **Architecture–Code Gap**: a system can have bug-free code in every service and still be fragile through hidden single points of failure or mismatched QoS contracts [13, 14]. Architecture evaluations such as ATAM rely on manual elicitation [15]. Static code analysis inspects services in isolation [16, 17]. Chaos engineering [18] needs a provisioned cluster, while pre-production alternatives like service-level fault injection testing [19] and microservice dependency tracing [20] require runnable execution environments. Homogeneous centrality flattens typed topologies into untyped graphs [21, 22]. Learned models, graph neural networks in particular, could combine these structural cues. What is missing is a representation that makes pub-sub failure paths explicit, and evidence on which analyzer to trust on it: whether graph learning adds anything over the coupling metrics that architecture research already has. This paper provides that evidence for one well-defined task, ranking components by simulated cascade impact before deployment.

## 1.2 The Software-as-a-Graph (SaG) Approach

**Software-as-a-Graph (SaG)** is a pre-deployment static analysis framework for event-driven architectures (Figure 1). It (1) models an architecture as a typed, directed multigraph over Applications, Brokers, Topics, Execution Hosts and Shared Libraries (§3.1); (2) derives from it, through publish–subscribe rules, an explicit `DEPENDS_ON` dependency graph (§3.3); and (3) ranks components by predicted cascade impact with training-free rankers, graph-learning engines, and hybrid engines in which a learned model corrects a closed-form score (§4). Predictors read only the analysis graph. Ground truth comes from three simulators that run on the raw structural topology (§4.3).

**What the ground truth is, and is not.** All ground truth in this study is simulated. The primary oracle $I^*$ is a reachability simulation that runs on the same manifest the predictors read, and its first propagation wave is exactly the number of a component’s direct dependents (§4.4). Dependency counts are therefore reported as references that restate the oracle, never as predictors. The study measures agreement between cheap static rankers and simulators, not the prediction of observed outages. Because ranking against $I^*$ rewards restating its propagation rule, we bound the circularity with two further simulators and with partial correlations, and report the cost of running $I^*$ itself (§6.4); validation against observed failures remains open (§7.3).

**Findings in brief.** The registered primary contrast, a heterogeneous graph transformer against QoS-weighted centrality, was null ($+0.069$, $p = 0.266$), and applying the plan’s selection rule afterwards does not change that. Two hybrids, learned engines corrected by the centrality prior, beat centrality on 11 of 12 synthetic architectures ($+0.103$ and $+0.130$, Holm-significant), but not their own base learners. Rankings that restate $I^*$’s propagation rule set a reference level on it: the oracle’s first-order expansion reaches $\rho = 0.808$ and the number of direct dependents (publish–subscribe afferent coupling [23, 24]) $0.764$. Graph attention networks on the dependency graph reach $0.748$, statistically indistinguishable from that count but not equivalent to it, and only with their degree features: without them they lose $0.14$. On a queue-flow simulator whose labels cost $12.7$ CPU-hours, a learned surrogate trained on those labels ranks held-out architectures above every closed-form approximation ($0.799$ against $0.706$), and it is the only place where declared QoS contracts carry measurable signal; a GNN given the same labels does not. The result is a benchmark with a largely negative outcome for graph learning where the oracle is cheap, and a positive one for learned surrogates where it is not.

## 1.3 Research Questions

-   **RQ1 (Ranking accuracy):** *How accurately do SaG’s closed-form, learned and hybrid engines rank components by cascading-failure impact on unseen architectures, compared with structural baselines, and how close do they come to reference rankings that restate the oracle’s propagation rule?*

-   **RQ2 (What learning needs):** *What do learned engines need: which graph they read, relation-specific (typed) parameters, or QoS inputs, once model capacity and edge-channel width are matched?*

-   **RQ3 (Transfer):** *How well do engines trained on synthetic architectures transfer zero-shot to independently authored models inspired by five open-source systems?*

-   **RQ4 (Cost):** *What does the analysis cost at CI/CD time, which stage dominates, and how does it compare with running the simulation directly?*

The evaluation follows a version-controlled analysis plan. Only the plan’s two contrasts are confirmatory; the matched $2\times2$, the hybrids, the reference counts, the dependency-graph learners, the full-population queue-flow labels and the round-8 arms were registered in later amendments, each before its own arms ran but after the primary result was known, and are reported as registered secondary results; the rest is exploratory (§5.3). One deviation from the plan, its unapplied selection rule, is disclosed and tested. Supplementary §S25 logs every amendment and maps these four research questions onto the plan.

## 1.4 Contributions

Evaluated under leave-one-scenario-out (LOSO) cross-validation over twelve synthetic architectures and on five hand-authored models of open-source systems, this paper contributes:

1.  **A benchmark of static rankers for pre-deployment cascade-impact ranking in publish–subscribe systems** (§§5–6): seventeen architectures (2,812 components), three simulation oracles labelled for every Application, and training-free, learned and hybrid rankers, with a version-controlled analysis plan, its amendments and one disclosed deviation, and artifacts that regenerate byte-identically.

2.  **A largely negative result for graph learning where the oracle is cheap** (§§6.1–6.2). The registered primary contrast is null; the hybrids beat the registered centrality comparator but not their base learners. Graph neural networks gain $+0.08$ to $+0.11$ by reading the dependency graph, reaching $0.748$, indistinguishable from the direct-dependent count; the gain rests on the degree features they are given, and relation typing adds nothing at matched capacity.

3.  **A positive result for learned surrogates where the oracle is expensive** (§6.1). A gradient-boosted surrogate trained on queue-flow labels beats every closed-form approximation of that oracle on held-out architectures, its QoS features carry the gain, and an attention-based GNN given the same labels fails.

4.  **One operational account of oracle circularity** (§4.4): a criterion for which rankings are references for which oracle, applied to all three, with partial correlations that show how much of each ranking survives outside the oracle it restates.

5.  **Explicit dependency rules for publish–subscribe architectures** (§3), extending the application-level rule of [25] with library-mediated blasts and infrastructure rules, and stating the typed query behind afferent coupling.

6.  **A like-for-like cost profile** (§6.4) of the counting path, the oracle and the learned pipeline on the same graphs.

A previous conference paper [25] introduced the publish–subscribe multigraph, the Application-level subscriber$\to$publisher dependency (Rule 1 of Table 3) and a closed-form betweenness–articulation score of the same form as `Topo` (Eq. 5, with weights $0.7$/$0.3$), and validated that score against a reachability-loss simulation on synthetic topologies and two ROS 2 benchmarks. Everything else is new in this paper: the Library and infrastructure rules (Rules 2–6, including library-mediated blasts), QoS weighting, the reference counts, the learned and hybrid engines, the QoS edge encoding, LOSO and zero-shot evaluation, the second and third oracles, the matched control and the cost profile.

§2 reviews related work, §§3–4 present the model and the rankers, §§5–6 the evaluation, §7 the discussion and threats, and §8 concludes.
