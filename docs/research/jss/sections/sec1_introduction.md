# 1. Introduction

## 1.1 Motivation and Problem

Large distributed systems now often use asynchronous publish-subscribe (pub-sub) middleware. Examples include ROS 2 for autonomous driving [1], Apache Kafka for enterprise event streams [2], DDS in cyber-physical systems [3], MQTT in IoT [4], and cloud-native microservices [5, 6]. Pub-sub separates producers and consumers by space, time, and synchronization [7]. Instead of direct references, components communicate via topics and brokers, and Quality-of-Service (QoS) policies set at deployment control reliability, durability, priority, and deadlines.

This decoupling also hides how failures spread. Outages, blocking and backpressure travel through brokers, shared topics, colocated hosts and shared libraries [8, 9]: in sequential cascades a slow subscriber fills a broker queue and starves its publishers [10], and in simultaneous blasts a library crash or host outage takes down every colocated service at once. Neither is visible in architecture diagrams or call graphs. The cheapest time to address these risks is before deployment, during design and continuous integration [11, 12], when no telemetry exists, so critical components must be identified from configuration manifests alone.

Current practice leaves a gap between service-level static analysis and system dependability: a system can have no defect in any service yet fail through hidden single points of failure or mismatched QoS contracts [13, 14]. ATAM depends on manual input [15], static code analysis inspects services in isolation [16, 17], and chaos engineering [18], fault injection [19] and dependency tracing [20] need a running system and, in CI, considerable computation and energy [21, 22, 23]. Graph representations have long supported reliability and defect analysis [24, 25], and graph learning has recently been applied to network criticality [26, 27] and microservices [28]. It remains unclear whether such models’ performance comes from the learning algorithm or from the dependency representation it reads, and when learning adds value beyond analytical methods derived from the same dependencies. Software-as-a-Graph (SaG) addresses this question.

## 1.2 The Software-as-a-Graph (SaG) Approach

SaG models an event-driven architecture as a typed, directed multigraph over Applications, Brokers, Topics, Execution Hosts and Shared Libraries (Figure 1; §3.1), derives an explicit DEPENDS\_ON dependency graph through publish–subscribe rules (§3.3), and supports analytical, hybrid and learned cascade-impact rankings on it (§§4 and 5.2).

Rankers never read simulator output. Labels come from three simulators on the raw topology (§4.3) that encode different failure notions: structural reachability ($I^*$), discrete-event queue saturation ($I_{\text{dyn}}$) and multi-criteria fragmentation ($I_{\text{comp}}$); every ranker is evaluated against all three. Because simulators and rankers read the same architecture, some analytical rankings restate a simulator’s rule; a reference criterion (§4.4) reports these as measures of circularity, not as predictors. All conclusions are conditional on simulation fidelity (§7.5).

The simulators give learning different roles. The reachability simulator ranks a corpus architecture in under a second, so a learned approximation of it has no practical use; we study it to calibrate the method, measuring how much of a simulator a learner and a restatement of its rule each recover. The queue-flow simulator is expensive, so approximating it is a genuine surrogate-modelling problem, and that is where we test learners that start from the analytical approximation (§6.1).

## 1.3 Findings in Brief

Explicit dependency representations provide most of the predictive signal for the reachability and queue-flow simulators.

Making dependencies explicit enables graph learning. On the raw multigraph every relation points away from Applications, so message passing never reaches the scored nodes, and the registered primary contrast (a heterogeneous graph transformer against the training-free baseline) was null ($\Delta\rho = +0.069$, $p = 0.266$). On the dependency graph, graph attention reaches Spearman $\rho = 0.748$ against the reachability simulator (`GAT-P-QoS`), against $0.635$ on the raw multigraph. Edge direction does not account for the gain (the derived graph adds $+0.072$ over a raw-graph GAT that passes every edge both ways, Holm $p = 0.014$), nor do the features that compute part of the simulator: with all of them removed, the dependency-graph GAT beats an equally stripped reverse-edge control by $+0.231$ (Holm $p = 0.0068$), while the raw-graph learners collapse.

The same representation enables simple analytical rankings that the learned models do not exceed. Counting direct dependents, publish–subscribe afferent coupling, reaches $0.764$ and is not distinguishable from the best learned model, which approaches it through its degree features. On the queue-flow simulator ($12.7$ CPU-hours to label the corpus), a rate-weighted first-order approximation reaches $0.830$ in milliseconds and exceeds a gradient-boosted approximation trained on the simulator’s labels ($0.799$; exploratory); declared rates and payload sizes, not QoS policies, carry the extra signal.

Learning adds measurable value only relative to a weak comparator. Hybrids that correct a closed-form prior outperform the training-free baseline but not their own base learners, so the gain belongs to the baseline’s weakness. Learners that start from the rate-weighted approximation do not improve on it: as an extra input it leaves a gradient-boosted model at its value, and as a prior it leaves a graph attention network below it ($-0.018$). Learned models transfer zero-shot to five stylized models of open-source systems better than the training-free baseline ($\rho \approx 0.81$ vs. $0.53$), mainly by separating inert from active components, while dependency counts rank higher still and ordering among active components remains weak ($\rho_{>0} \le 0.342$). All three simulators are well approximated by low-order functions of the dependency graph, so the benchmark did not exercise a regime in which learning could exceed an aligned analytical ranking.

In practice, the results favor training-free dependency analysis for routine use. Recovering 80% of the true critical set requires reviewing about 40–45% of Applications, so rankings suit prioritized review rather than blocking gates (§7.4).

## 1.4 Research Questions

Establishing when learning is unnecessary is as informative as showing when it succeeds. Software engineering research has repeatedly shown that simple metrics can match the performance of more complex models when the right information is available [25, 29, 30, 31]. Analytical rankings can also be easier to explain, validate, and run in Continuous Integration and Continuous Deployment (CI/CD) pipelines. We therefore evaluate learned models against analytical rankings derived from the same dependencies, not only against structural baselines. Four research questions guide the study:

-   **RQ1 (Ranking accuracy):** *How accurately do analytical, hybrid, and learned approaches rank components by cascading-failure impact on unseen architectures, and how close do they come to reference rankings that restate a simulator’s propagation rule?*

-   **RQ2 (Sources of predictive performance):** *At matched model capacity, which factors contribute most to ranking performance: the dependency representation, degree information, relation typing, Quality-of-Service (QoS) information, or the model family (attention or sum aggregation; graph neural network or gradient-boosted trees)?*

-   **RQ3 (Transfer):** *How well do learned models trained on synthetic architectures transfer zero-shot to stylized models of five open-source systems, hand-authored by a single modeler, and does their ordering of components that actually propagate failures hold?*

-   **RQ4 (Cost):** *What do analytical and learned approaches cost at CI/CD time in latency and estimated energy, which stage dominates, and how does this compare with running each simulator directly?*

The evaluation follows an analysis plan pre-specified in the replication repository, where each entry carries a commit timestamp; it was not lodged with a third-party registry, and “registered” in this paper means pre-specified in that plan. Only the plan’s two co-primary contrasts are confirmatory, and both were null. We registered the matched $2\times2$, the hybrids, the reference counts, the dependency-graph learners (whose contrasts were declared exploratory), the full-population queue-flow labels, the sensitivity arms and the controls of §6.2 as planned extensions. We registered each before its own tests ran but after we knew the main result, and we report them as secondary results; no confirmation corpus generated after the analysis was frozen has yet tested them (§7.5). The rest of the analysis is exploratory (§5.3). One change from the plan, not applying the registered nested hyperparameter selection, is disclosed and tested. The replication repository records the revision history and shows how these four research questions fit into the plan.

## 1.5 Contributions

This paper makes five contributions:

1.  **Dependency derivation for publish–subscribe architectures** (§3). A method for deriving explicit dependency graphs from deployment manifests using typed architectural rules, extending the Application-level rule of [32] with library-mediated dependencies (Rule 5); infrastructure rules are defined but not evaluated.

2.  **An evaluation guideline for simulator-labeled benchmarks** (§4.4). An operational criterion, a truncation of the simulator’s own computation, that identifies rankings restating a simulator’s propagation rule, so that learned rankers are compared with them rather than credited for agreement they encode.

3.  **A controlled comparison of analytical, hybrid, and learned ranking methods** (§§5.2 and 6.1–6.3). A study that separates the effects of dependency representation, degree information, relation typing, QoS information, and model family on cascade-impact ranking.

4.  **Analytical and learned approximations of an expensive simulator** (§§5.2 and 6.1). A rate-weighted first-order approximation of a queue-flow simulator (Eq. 7), compared with learned approximations trained on the simulator’s labels.

5.  **A reproducible benchmark, cost analysis and practitioner guidance** (§§5.1, 6.4 and 7.4). A reproducible benchmark of seventeen architectures with three simulators, a like-for-like cost analysis, and recommendations for pre-deployment cascade-impact analysis.

A previous conference paper [32] introduced the publish–subscribe multigraph, the Application-level subscriber-to-publisher dependency (Rule 1 of Table 2), and a closed-form betweenness–articulation score similar to the training-free baseline (Eq. 5, unweighted and with weights $0.7$/$0.3$), which it validated against a reachability-loss simulation on synthetic topologies and two ROS 2 benchmarks. This paper adds the Library and infrastructure rules, QoS weighting and the QoS edge encoding, the reference criterion, the learned and hybrid rankers, the analytical and learned simulator approximations, LOSO and zero-shot evaluation, the second and third oracles, the matched control, and the cost analysis. The closed-form score family validated in [32] is the weakest analytical ranker in the present evaluation, which adds reference rankings, the Application population and three oracles; the present results supersede that paper’s implicit recommendation of it.

§2 reviews related work. §§3 and 4 present the model and the rankers. §§5 and 6 cover the evaluation. §7 discusses the findings and threats, and §8 concludes.
