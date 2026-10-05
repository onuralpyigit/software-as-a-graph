# 1. Introduction

## 1.1 Motivation and Problem

Large distributed systems increasingly use asynchronous publish-subscribe (pub-sub) middleware: ROS 2 for autonomous driving [1], Apache Kafka for enterprise event streams [2], DDS in cyber-physical systems [3], MQTT in IoT [4], and cloud-native microservices [5, 6]. Pub-sub separates producers and consumers in space, time and synchronization [7]; components communicate via topics and brokers, and Quality-of-Service (QoS) policies set at deployment control reliability, durability, priority and deadlines. This decoupling also hides how failures spread. Outages and backpressure travel through brokers, shared topics, colocated hosts and shared libraries [8, 9], in sequential cascades (a slow subscriber fills a broker queue and starves its publishers [10]) and in simultaneous blasts (a library crash takes down every service that uses it). Neither is visible in architecture diagrams or call graphs. The cheapest time to address these risks is before deployment [11, 12], when no telemetry exists, so critical components must be identified from configuration manifests alone.

Current practice leaves a gap between service-level analysis and system dependability: a system can have no defect in any service yet fail through hidden single points of failure or mismatched QoS contracts [13, 14]. ATAM depends on manual input [15], static code analysis inspects services in isolation [16, 17], and chaos engineering [18], fault injection [19] and dependency tracing [20] need a running system and considerable computation [21, 22, 23]. Graph representations have long supported reliability and defect analysis [24, 25], and graph learning has recently been applied to network criticality [26, 27] and microservices [28]. It remains unclear whether such models’ performance comes from the learning algorithm or from the dependency representation it reads, and when learning adds value beyond analytical methods derived from the same dependencies. Software-as-a-Graph (SaG) addresses this question.

## 1.2 The Software-as-a-Graph (SaG) Approach

SaG models an event-driven architecture as a typed, directed multigraph over Applications, Brokers, Topics, Execution Hosts and Shared Libraries (Figure 1), derives an explicit `DEPENDS_ON` dependency graph through publish–subscribe rules (§3.3), and supports analytical, hybrid and learned cascade-impact rankings on it (§§4 and 5.2). Labels come from three simulators on the raw topology that encode different failure notions: structural reachability ($I^*$), discrete-event queue saturation ($I_{\text{dyn}}$) and multi-criteria fragmentation ($I_{\text{comp}}$) (§4.3). Because simulators and rankers read the same architecture, some analytical rankings restate a simulator’s rule; a reference criterion (§4.4) reports these as measures of circularity, not as predictors. The reachability simulator is cheap, so we use it to calibrate the method; the queue-flow simulator is expensive, so approximating it is a genuine surrogate-modelling problem.

## 1.3 Findings in Brief

*Making dependencies explicit enables graph learning.* On the raw multigraph every relation points away from Applications, so message passing never reaches the scored nodes, and the registered primary contrast (a heterogeneous graph transformer against the training-free baseline) was null ($\Delta\rho = +0.069$, $p = 0.266$). On the dependency graph, graph attention reaches Spearman $\rho = 0.748$ against the reachability simulator (`GAT-P-QoS`), against $0.635$ on the raw multigraph. Neither edge direction ($+0.072$ over a GAT that passes every raw edge both ways, Holm $p = 0.014$) nor the features that compute part of the simulator account for the gain: with all of them removed, the dependency-graph GAT beats an equally stripped reverse-edge control by $+0.231$ (Holm $p = 0.0068$).

*The same representation enables simple analytical rankings that learned models do not exceed.* Counting direct dependents, publish–subscribe afferent coupling, reaches $0.764$ and is not distinguishable from the best learned model. On the queue-flow simulator ($12.7$ CPU-hours to label the corpus), a rate-weighted first-order approximation reaches $0.830$ in milliseconds and exceeds a gradient-boosted approximation trained on the simulator’s labels ($0.799$; exploratory). Learners that start from it do not improve on it: as an extra input it leaves a gradient-boosted model at its value, and as a prior it leaves a graph attention network below it ($-0.018$).

*Learning adds measurable value only relative to a weak comparator.* Hybrids that correct a closed-form prior outperform the training-free baseline but not their own base learners. Learned models transfer zero-shot to five stylized models of open-source systems better than that baseline ($\rho \approx 0.81$ vs. $0.53$), while dependency counts rank higher still and ordering among active components remains weak. All three simulators are well approximated by low-order functions of the dependency graph, so the benchmark did not exercise a regime in which learning could exceed an aligned analytical ranking. In practice, recovering 80% of the critical set requires reviewing about 40–45% of Applications, so rankings suit prioritized review rather than blocking gates (§7.3).

## 1.4 Research Questions

Establishing when learning is unnecessary is as informative as showing when it succeeds: software engineering research has repeatedly shown that simple metrics can match more complex models when the right information is available [25, 29, 30, 31]. We therefore evaluate learned models against analytical rankings derived from the same dependencies:

-   **RQ1 (Ranking accuracy):** *How accurately do analytical, hybrid, and learned approaches rank components by cascading-failure impact on unseen architectures, and how close do they come to reference rankings that restate a simulator’s propagation rule?*

-   **RQ2 (Sources of predictive performance):** *At matched model capacity, which factors contribute most to ranking performance: the dependency representation, degree information, relation typing, QoS information, or the model family?*

-   **RQ3 (Transfer):** *How well do learned models trained on synthetic architectures transfer zero-shot to stylized models of five open-source systems, and does their ordering of components that actually propagate failures hold?*

-   **RQ4 (Cost):** *What do analytical and learned approaches cost at CI/CD time in latency and estimated energy, and how does this compare with running each simulator directly?*

The evaluation follows an analysis plan pre-specified in the replication repository. Only its two co-primary contrasts are confirmatory, and both were null; the headline findings above are registered secondary or exploratory (§5.3).

## 1.5 Contributions

1.  **Dependency derivation for publish–subscribe architectures** (§3): typed rules that derive explicit dependency graphs from deployment manifests, extending the Application-level rule of [32] with library-mediated dependencies.

2.  **An evaluation guideline for simulator-labeled benchmarks** (§4.4): a reference criterion, a truncation of the simulator’s own computation, against which learned rankers are compared rather than credited for agreement they encode.

3.  **A controlled comparison of analytical, hybrid, and learned rankers** (§§6.1–6.3) that separates the effects of representation, degree information, relation typing, QoS information and model family.

4.  **Analytical and learned approximations of an expensive simulator** (§6.1): a rate-weighted first-order approximation of a queue-flow simulator (Eq. 7), compared with learned approximations trained on its labels.

5.  **A reproducible benchmark, cost analysis and practitioner guidance** (§§5.1, 6.4 and 7.3) over seventeen architectures and three simulators.

A previous conference paper [32] introduced the publish–subscribe multigraph, the Application-level subscriber-to-publisher dependency (Rule 1 of Table 2) and a closed-form betweenness–articulation score similar to the training-free baseline (Eq. 5), validated against a reachability-loss simulation. Everything else here is new: the Library and infrastructure rules, QoS weighting, the reference criterion, the learned and hybrid rankers, the simulator approximations, LOSO and zero-shot evaluation, the second and third oracles, the matched controls and the cost analysis. Under this broader evaluation, the closed-form score family of [32] is the weakest analytical ranker, and the present results supersede that paper’s implicit recommendation of it.

§2 reviews related work, §§3 and 4 present the model and the rankers, §§5 and 6 the evaluation, §7 the discussion and threats, and §8 concludes.
