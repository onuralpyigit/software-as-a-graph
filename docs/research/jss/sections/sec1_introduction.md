# 1. Introduction

## 1.1 Motivation and Problem

Large distributed systems increasingly communicate through asynchronous publish–subscribe (pub-sub) middleware: ROS 2 in autonomous driving [1], Apache Kafka in enterprise event streams [2], DDS in cyber-physical systems [3], MQTT in IoT [4], and cloud-native microservices [5, 6]. Pub-sub decouples producers and consumers in space, time and synchronization [7]. Components interact through topics and brokers rather than direct references, and deployment-time Quality-of-Service (QoS) policies govern reliability, durability, priority and deadlines.

The same decoupling hides how failures spread. Publishers and subscribers share no direct link, so outages, head-of-line blocking and backpressure propagate along concealed paths through brokers, shared topics, colocated hosts and shared libraries [8, 9]. These failures take two forms. In *sequential cascades*, a slow subscriber fills a broker queue and gradually starves its publishers [10]. In *simultaneous blasts*, a shared-library crash or host outage takes down every colocated service at once. Neither architecture diagrams nor static call graphs show these mechanisms. The cheapest time to reduce the risk is before deployment, at design and continuous-integration time [11, 12], when no runtime telemetry exists. Architects therefore need to know, from configuration manifests alone, which components, topics and links are systemically critical and why.

Existing practice leaves this gap open, which we call the **Architecture–Code Gap**: a system can have bug-free code in every service and still be fragile through hidden single points of failure or mismatched QoS contracts [13, 14]. Architecture evaluations such as ATAM rely on manual elicitation [15]. Static code analysis inspects services in isolation [16, 17]. Chaos engineering needs a provisioned cluster [18]. Homogeneous centrality flattens typed topologies into untyped graphs [19, 20]. Learned models could combine these structural cues, but on their own they produce risk scores without actionable explanations. What is missing is a representation that makes pub-sub failure paths explicit, and evidence on which analyzer to trust on it.

## 1.2 The Software-as-a-Graph (SaG) Approach

**Software-as-a-Graph (SaG)** is a pre-deployment static system analysis framework for event-driven architectures (Figure 1). It (1) models an architecture as a typed, directed multigraph over Applications, Brokers, Topics, Execution Hosts and Shared Libraries (§3.1); (2) derives a QoS-weighted `DEPENDS_ON` projection that captures both sequential cascades and simultaneous blasts (§3.2); (3) ranks components by predicted cascade impact with a training-free closed-form engine, graph-learning engines, and hybrid engines in which a learned model corrects the closed-form score (§4); and (4) explains flagged components with an ISO/IEC 25010 Reliability–Maintainability (RM) profile (§5). Predictors read only the analysis graph. Ground-truth impact comes from independent simulators that run on the raw structural topology (§4.4). The central thesis is that the representation, not the choice of analyzer, is the decisive investment: once declared QoS contracts are part of the graph, closed-form and learned engines both rank failure impact better, and each engine is the best choice in a distinct deployment context.

## 1.3 Research Questions

-   **RQ1 (Ranking accuracy):** *How accurately do SaG’s closed-form, learned and hybrid engines rank components by cascading-failure impact on unseen architectures, compared with standard structural baselines?*

-   **RQ2 (What learning needs):** *Where does the learned engines’ accuracy come from—relation-specific (typed) parameters, message passing, the QoS edge encoding, or the per-component features SaG extracts—once model capacity and edge-channel width are matched?*

-   **RQ3 (Transfer):** *How well do engines trained on synthetic architectures transfer zero-shot to independently authored models of five open-source systems?*

-   **RQ4 (Cost):** *What does the analysis cost at CI/CD time, which stage dominates, and how does it compare with running the simulation directly?*

The primary contrast, the matched control and both hybrid engines were each registered with a decision rule before their results existed. Supplementary §S24 logs every later change as an amendment and maps these four questions onto the five of the registered analysis plan. The attribution controls of §7.2, which separate message passing and the QoS node features from the edge encoding, were added after the matched control’s result. They are exploratory and are logged as Amendment 7; the feature-only regressor among them was declared in Amendment 3, also post hoc, and first run here.

## 1.4 Contributions

Evaluated under leave-one-scenario-out (LOSO) cross-validation over twelve synthetic architectures and zero-shot on five open-source system models, this paper contributes:

1.  **A QoS-aware typed architecture model** that derives logical dependencies from physical pub-sub linkages, weights them by declared QoS contracts, and distinguishes sequential cascades from simultaneous blasts (§3). This representation is the largest single gain in the study: ranking on it raises Spearman correlation with simulated cascade impact from $0.349$ (unweighted centrality) to $0.553$, on all twelve held-out architectures ($+0.204$, $p = 0.0005$).

2.  **Complementary closed-form, learned and hybrid engines** (§4). Closed-form and learned engines are strongest on different architectures, and neither significantly outperforms the other alone. Hybrid engines that learn a correction to the closed-form score exploit this complementarity. They are the most accurate engines on held-out architectures ($\rho = 0.657$ and $0.683$; $+0.103$ and $+0.130$ on 11 of 12 folds). Each meets the decision rule registered before its run and remains significant under a Holm correction pooled over all eleven registered contrasts.

3.  **Evidence on what the learned engines need and how far they transfer** (§7). Matched controls show that learned accuracy comes from the per-component features SaG’s analysis computes on the QoS-weighted graph, not from relation-specific weights, message passing or the QoS edge encoding. A gradient-boosted regressor on the same features matches the learned engines ($\rho = 0.642$ against $0.622$–$0.635$), and the untyped attention network’s gain from QoS is carried by three declared-coupling node features ($+0.095$), not by the 16-D edge channel ($-0.023$). Trained only on synthetic data, learned models transfer zero-shot to five open-source system models at $\rho = 0.757$–$0.831$, against $0.511$–$0.526$ for every training-free score, and raise top-$K$ critical-set overlap from $0.248$ to $0.40$–$0.55$.

4.  **A standards-grounded explanation layer** (§5) that attributes each flagged component to ISO/IEC 25010 Availability, Fault Tolerance or Maintainability, and so names the remediation it calls for: replication, circuit breakers or decoupling.

5.  **A reproducible benchmark and cost profile**: seventeen architectures totaling 2,812 components, twelve of which regenerate byte-identically from committed configurations. Every reported table value is mechanically reconciled against released artifacts. Neural inference takes $56\,\text{ms}$ on a 2,000-component architecture, and one structural metric dominates cost (§7.4).

A previous conference paper [21] introduced the preliminary multigraph and deterministic quality model on synthetic topologies. This paper adds the learned and hybrid engines, the QoS edge encoding, LOSO and zero-shot evaluation, the matched control, and the cost profile, and it repositions that quality model as the explanation layer.

§2 reviews related work, §§3–5 present the model, engines and explanation layer, §§6–7 the evaluation, §8 the discussion and threats, and §9 concludes.
