# 1. Introduction

## 1.1 Motivation and Problem

Large distributed systems increasingly communicate through asynchronous publish–subscribe (pub-sub) middleware: ROS 2 in autonomous driving [1], Apache Kafka in enterprise event streams [2], DDS in cyber-physical systems [3], MQTT in IoT [4], and cloud-native microservices [5, 6]. Pub-sub decouples producers and consumers in space, time and synchronization [7]. Components interact through topics and brokers rather than direct references, and deployment-time Quality-of-Service (QoS) policies govern reliability, durability, priority and deadlines.

The same decoupling hides how failures spread. Publishers and subscribers share no direct link, so outages, head-of-line blocking and backpressure propagate along concealed paths through brokers, shared topics, colocated hosts and shared libraries [8, 9]. These failures take two forms. In *sequential cascades*, a failed or slow publisher starves the subscribers that depend on its topics [10]. In *simultaneous blasts*, a shared-library crash or host outage takes down every colocated service at once. Neither architecture diagrams nor static call graphs show these mechanisms, because the dependency between two services exists only through the topic, broker or library they share.

The cheapest time to reduce this risk is before deployment, at design and continuous-integration time [11, 12], when no runtime telemetry exists. The information needed is increasingly available then: Architecture-as-Code manifests, ROS 2 launch files and broker configurations already declare which component publishes and subscribes to which topic, which broker routes it, which host runs it and which libraries it links. What existing practice lacks is a way to turn that declaration into a ranking of systemic risk. Architecture evaluations such as ATAM rely on manual elicitation [13]. Static code analysis inspects services in isolation [14, 15], so a system can have clean code in every service and still be fragile through a hidden single point of failure, a gap between architecture and code that architectural-smell research documents [16, 17]. Chaos engineering needs a provisioned cluster [18]. Centrality on untyped graphs flattens the distinction between topics, libraries and hosts [19, 20]. Learned models could combine structural cues, but it is open whether they add anything once the dependencies are explicit.

## 1.2 The Software-as-a-Graph (SaG) Approach

**Software-as-a-Graph (SaG)** is a pre-deployment static analysis framework for event-driven architectures (Figure 1). It (1) models an architecture as a typed, directed multigraph over Applications, Brokers, Topics, Execution Hosts and Shared Libraries (§3.1); (2) derives a `DEPENDS_ON` projection that makes the hidden dependencies explicit and distinguishes sequential cascades from simultaneous blasts (§3.2); (3) ranks components by cascading-failure impact with training-free, learned and hybrid engines (§§4 and 6.2); and (4) profiles flagged components along ISO/IEC 25010 reliability and maintainability sub-characteristics (§5). Predictors read only the analysis graph; ground-truth impact comes from simulators that run on the raw structural topology (§4.4).

The central thesis is that making dependencies explicit is the decisive investment. Once the projection exists, the question “whose failure reaches furthest?” becomes a question about how many components depend on whom, which a count answers directly and which learned engines, hybrids and QoS weights can then be judged against.

## 1.3 Research Questions

- **RQ1 (Ranking accuracy):** *How accurately do training-free, learned and hybrid engines on SaG’s graphs rank components by cascading-failure impact on unseen architectures, and what drives the accuracy of the training-free engines?*

- **RQ2 (What learning needs):** *Does the learned engines’ accuracy come from relation-specific (typed) parameters or from the edge encoding, once model capacity and edge-channel width are matched?*

- **RQ3 (Transfer):** *How well do these rankers transfer to independently authored models of five open-source systems?*

- **RQ4 (Cost):** *What does the analysis cost at CI/CD time, and how does it compare with running the simulation directly?*

Every headline contrast was registered with a decision rule before its results existed: the primary contrast, the matched control, both hybrid engines, and the training-free dependency counts with their QoS-attribution controls. §6.4 and Supplementary §S24 log every amendment.

## 1.4 Key Findings at a Glance

1.  **Counting dependents ranks failure impact best.** On SaG’s dependency projection, the number of a component’s dependents ranks its simulated cascade impact at Spearman $\rho = 0.764$ on held-out architectures, above the registered betweenness engine on all twelve ($+0.211$) and above every learned and hybrid engine ($0.622$–$0.683$). It identifies the components whose failure reaches no one with $94\%$ balanced accuracy (§7.1).

2.  **The dependency projection, not QoS weighting, carries the signal.** Registered controls attribute the closed-form engine’s gain over the registered centrality baseline ($0.349 \to 0.553$) to the projection: constant or permuted topic weights rank as well as the declared QoS contracts (§7.1.2).

3.  **Learning helps over betweenness, and simple models suffice.** Hybrid engines are the only learned engines that significantly beat the closed-form engine ($+0.103$ and $+0.130$, 11 of 12 folds), and an untyped attention network matches a heterogeneous transformer at equal capacity (§§7.1 and 7.2).

4.  **Rankings transfer.** Trained only on synthetic data, learned engines transfer to five open-source system models at $\rho = 0.760$–$0.805$, and dependency counts reach $0.86$–$0.94$ with no training at all (§7.3).

5.  **The best ranker is the cheapest.** Deriving the projection and counting dependents takes under a tenth of a second on the largest architecture, against seconds for the simulator and minutes for the learned engines’ features (§7.4).

## 1.5 Contributions

1.  **A typed architecture model with an explicit dependency projection** for pub-sub systems, whose six derivation rules turn publish, subscribe, routing, hosting and library relations into dependencies and separate sequential cascades from simultaneous blasts (§3). On it, training-free dependency counts are accurate, transferable and effectively free.

2.  **A controlled comparison of what learning adds** (§§4 and 7): heterogeneous and homogeneous graph neural networks at matched capacity, hybrid engines that correct a closed-form prior, and training-free dependency counts, all under the same leave-one-scenario-out protocol and zero-shot transfer.

3.  **Registered attribution of where accuracy comes from**, including controls that separate the dependency projection from the QoS contracts and that overturned an earlier interpretation of this study (§§7.1.2 and 6.4).

4.  **A standards-grounded explanation layer** (§5) that profiles each flagged component along ISO/IEC 25010 Availability, Fault Tolerance and Maintainability and names a remediation class, presented as a design proposal for validation.

5.  **An open benchmark and replication package** in the spirit of the JSS Open Science initiative: seventeen architectures totaling 2,812 components, twelve of which regenerate byte-identically from committed configurations, a torch-free harness for every training-free result, and a reconciler that mechanically checks every reported table value against the released artifacts.

A previous conference paper [21] introduced the preliminary multigraph and deterministic quality model on synthetic topologies. This paper adds the learned, hybrid and dependency-count engines, the registered leave-one-scenario-out and zero-shot evaluation, the matched and QoS-attribution controls, and the cost profile, and it repositions that quality model as the explanation layer.

§2 reviews related work, §§3–5 present the model, engines and explanation layer, §§6–7 the evaluation, §8 the implications and threats, and §9 concludes.
