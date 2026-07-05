# Title: Graph Neural Networks for Architectural Dependability and Cascading Failure Prediction in Complex Distributed Systems

## Abstract

Modern complex distributed systems increasingly rely on publish–subscribe and microservice middleware to decouple data producers and consumers. While this decoupling provides significant scaling and operational flexibility, it obscures the logical dependency chains along which structural component failures cascade. Identifying system-level critical bottlenecks before deployment remains a challenge: operational runtime telemetry does not yet exist during the design phase, and standard static code analysis frameworks are blind to inter-component topologies.

To bridge this "Architecture-Code Gap," we present **Software-as-a-Graph (SaG)**, a pre-deployment Static System Analysis framework that evaluates runtime dependability entirely from structural design descriptors. SaG models distributed configurations as typed, weighted, directed multigraphs across five node classifications (applications, brokers, topics, infrastructure nodes, and shared libraries) and derives logical failure propagation paths via multi-layer relational projection rules.

On this semantic representation, we employ a **Heterogeneous Graph Neural Network (GNN)** predictor—specifically a parameterized Heterogeneous Graph Transformer with explicit Quality-of-Service (QoS) contract edge-feature injection ($HGL\text{-}QoS$)—to forecast each component's cascading failure impact $I(v)$ before single-line deployment. The network’s predictions are validated against an independent discrete-event cascade simulator under a strict input–label independence guarantee to ensure zero transductive leakage. Additionally, we pair this learned predictor with an interpretable, Analytic Hierarchy Process (AHP)-weighted multi-dimensional quality attribution score ($Q(v)$) decomposing failure mechanics into orthogonal Reliability, Maintainability, Availability, and Vulnerability (RMAV) dimensions.

Evaluating the framework across seven synthetic, hyper-scaled distributed topologies encompassing diverse industrial domains, we show that:

1. In-distribution regimes are strongly modeled by deterministic attribution ($\rho > 0.87$, $F_1 > 0.90$), whereas out-of-distribution generalization to completely unseen architectures requires typed graph learning, where our $HGL\text{-}QoS$ model decisely outperforms homogeneous configurations, achieving a Leave-One-Scenario-Out (LOSO) cross-validation score of $\rho = 0.401$.
2. A stratified correlation assessment confirms that the predictive capabilities of the GNN are highly consistent across all five node categories (with within-type Spearman correlations ranging from $\rho = 0.322$ to $0.429$), eliminating risks of a single pooled aggregate masking type-specific predictive failure.
3. We report a comprehensive, multi-seed evaluation of topology-level prescriptive remediation operators, documenting an overall cross-scenario mean impact reduction of $+4.61\%$, while honestly analyzing why an un-filtered policy implementation introduces counterproductive mutations in 3 out of 7 scenarios.

Finally, we demonstrate the operational feasibility of SaG by instantiating it as a delta-aware, blocking CI/CD quality gate. Utilizing a thread-safe, database-free `MemoryRepository` to eliminate live graph database latencies, the analyzer evaluates complex regressions under $5\text{ s}$ for medium and $40\text{ s}$ for hyper-scale topologies while achieving a $100\%$ detection rate ($precision=1.0, recall=1.0$) on newly introduced structural vulnerabilities, successfully driving automated, continuous dependability auditing.

**Keywords:** publish–subscribe middleware; architectural dependability; cascading failure; heterogeneous graph neural networks; pre-deployment verification; quality attributes; CI/CD quality gate.

---

## Manuscript Outline

### 1. Introduction

* **1.1 Motivation:** The growth of high-availability distributed computing (ROS 2, Kafka, MQTT) and the architectural challenge of tracking implicit, decoupled dependency structures.
* **1.2 The Architecture-Code Gap:** Why clean local source code (SCA tools like SonarQube) fails to prevent catastrophic system-level runtime failures when structural single points of failure (SPOFs) or QoS contract mismatches exist.
* **1.3 Problem Statement & Limitations of Prior Art:** Structural single-score centralities collapse typed semantics; chaos engineering is fragile and happens too late in the lifecycle.
* **1.4 Research Questions (Pivoted to Dependability Tracking):**
* *RQ1 (Interpretable vs. Learned):* When does a deterministic multi-attribute quality scoring framework suffice for predicting failure propagation, and when is a heterogeneous GNN needed to generalize to unseen architectures?
* *RQ2 (Typing & Structural Anomalies):* What systemic vulnerabilities does a typed architectural breakdown reveal that classic homogeneous centralities conceal?
* *RQ3 (QoS Impact):* How does lifting explicit middleware QoS contracts (reliability, durability, priority) into edge-feature encoders impact GNN convergence vs. out-of-distribution generalizability?
* *RQ4 (CI/CD Operationalization):* Can graph-based dependability simulation and GNN forecasting be executed sub-quadratically to run as a continuous blocking build gate?


* **1.5 Summary of Key Contributions & Document Structure**

### 2. Related Work

* **2.1 Publish–Subscribe Dependability & Protocol Resilience**
* **2.2 From Static Code Analysis (SCA) to Static System Analysis (SSA)**
* **2.3 Graph Neural Networks on Heterogeneous Software Topologies**
* **2.4 Cascading Failure Simulation and Structural Criticality Modeling**
* **2.5 Multi-Criteria Decision Metrics and Quality Gates in DevOps**

### 3. The Software-as-a-Graph (SaG) Modeling Formalism

* **3.1 Formal Mathematical Definition:** $G = (V, E, \tau_V, \tau_E, w_E, w_V)$ multigraph over 5 node types and 6 structural communication edge types.
* **3.2 QoS-Driven Edge and Vertex Weighting Formulas:** Mathematical mapping of reliability ($r$), durability ($d$), priority ($p$), and message volume (`size_norm`) into continuous coupling weights.
* **3.3 The Logical Projection Rules (`DEPENDS_ON`):** Synthesizing implicit paths, handling application-to-broker relations, application-to-application interactions, and node-level colocations.
* **3.4 Shared-Library Blast Semantics vs. Sequential Cascades:** Modeling the qualitative structural shift between step-by-step propagation and immediate multi-node failures via shared libraries.
* **3.5 Integration of Local Static Code Quality Vectors:** Ingesting SonarQube metrics (`cm_*` properties: complexity, LCOM, lines of code) onto application and library vertices.
* **3.6 Core Separation:** $G_{\text{structural}}$ (Simulator Path) vs. $G_{\text{analysis}}$ (Predictor/GNN Path) ensuring the structural independence guarantee.

### 4. Deterministic Multi-Dimensional Quality Attribution (The Interpretable Path)

* **4.1 The Orthogonal RMAV Dimensions:** Rationale for restricting metric domains to ensure mathematical orthogonality.
* **4.2 Mathematical Metrics Formulation:**
* *Reliability ($R$):* Failure Propagation Risk (RPR) calculated over transposed graph sweeps.
* *Maintainability ($M$):* Metric betweenness ($BT$) and Code Quality Penalty ($CQP$) formulas incorporating class instability and technical debt ratios.
* *Availability ($A$):* Directed cut-vertex tracking via directed articulation points ($AP\_c\_directed$) and QoS-weighted Single Points of Failure ($QSPOF$).
* *Vulnerability ($V$):* Attack surface exposure via Reverse Reachable Closet Likelihood ($REV$/$RCL$) calculations.


* **4.3 Decision-Theoretic Composite Score ($Q(v)$):** Pairwise comparison calibration using the Analytic Hierarchy Process (AHP); implementation of shrinkage toward a uniform prior ($\lambda=0.70$) to prevent extreme parameter weights.

### 5. Heterogeneous GNN for Failure-Impact Forecasting (The Learned Path)

* **5.1 Define Ground-Truth Impact Label $I(v)$:** The discrete-event cascade simulation combining reachability loss, structural fragmentation, throughput degradation, and flow disruption.
* **5.2 Heterogeneous Graph Transformer ($HGL\text{-}QoS$) Architecture:** Type-specific parameter matrices, relation-specific message passing, and QoS attribute masking/injection mechanics.
* **5.3 Enforcing the Zero-Leakage Independence Contract:** Complete mathematical decoupling of feature graphs from label-generating engines.
* **5.4 Evaluation of the Shared-Library Blast Mismatch (Negative Result Discussion):** Direct empirical breakdown of library performance against canonical simulation models.
* **5.5 Type-Stratified vs. Pooled Ranking Analysis Methodology**

### 6. Continuous Operationalization: CI/CD Quality Gating & Prescription

* **6.1 Architecture-as-Code Gating Semantics:** Implementing delta-aware analysis against git merge bases to avoid repeating legacy architectural anomalies.
* **6.2 The Registry & Waiver System:** Managing risk-accepted SPOFs via structured exception listings.
* **6.3 Script Execution Architecture:** The database-free, thread-safe `MemoryRepository` stack driving `detect_antipatterns.py` and outputting standard POSIX exit-codes ($0, 1, 2$).
* **6.4 Topology-Level Prescriptive Remediation Operators:** Formulating `RedundancyInsertion`, `PathDiversification`, `FanOutReduction`, and `SharedTopicReduction`.
* **6.5 Open Implementation Gap (Honest Reporting):** The target design of per-edit multi-seed simulation acceptance checks ($\Delta I > \kappa\,\sigma_{\text{seed}}$) vs. current unconditional full-policy mutations.

### 7. Experimental Setup

* **7.1 Synthetic Scenario Suite Profile:** Scale presets (`tiny` to `xlarge`), structural properties, and domain contexts across the 7 scenarios (Healthcare, AV, Trading, etc.).
* **7.2 Baselines and Model Ablations (GL, GL-QoS, HGL, HGL-QoS, Topo-BL)**.
* **7.3 Evaluation Frameworks:** Spearman rank correlation ($\rho$), NDCG@10, top-K overlap indices, and classification F1-scores.
* **7.4 Inductive Testing Protocol:** Leave-One-Scenario-Out (LOSO) cross-validation framework over five deterministic seeds.

### 8. Experimental Evaluation and Results

* **8.1 Answers to RQ1 (Interpretable vs. Learned):** Success of RMAV/$Q$ in-distribution ($\rho > 0.87$); failure of homogeneous networks ($\rho \approx 0.02$) and success of $HGL\text{-}QoS$ ($\rho = 0.401$) under out-of-distribution LOSO tracking.
* **8.2 Answers to RQ2 (Taking Node Types Seriously):** Gain of typed heterogeneity over homogeneous embeddings ($\Delta F1 = +0.284$). Full presentation of the shared-library negative result and stratified consistency profiles.
* **8.3 Answers to RQ3 (QoS Impact Analysis):** Explaining the in-distribution parameter noise vs. out-of-distribution transfer learning capability provided by explicit QoS encoding.
* **8.4 Answers to RQ4 (Gating Performance Footprint):** Execution benchmarks showing sub-quadratically scaling profiles ($5\text{ s}$ for medium topologies, $40\text{ s}$ for xlarge) and a $100\%$ precision/recall profile on injected regressions.
* **8.5 End-to-End Evaluation of Prescriptive Remediation:** Honest reporting of cross-scenario outcomes (the overall mean $+4.61\%$ shift but negative regressions across 3 specific scenario suites due to unconditional execution).

### 9. Discussion, Limitations, and Conclusion

* **9.1 Deep Interpretation of Results and Architectural Tradeoffs**
* **9.2 Threats to Validity:** Construct validity (simulation-derived labels), Internal validity (independence checks), and External validity (synthetic generation limits).
* **9.3 Actionable Next Steps (Future Work):** Convening the blind domain-expert ranking panel, integrating real production failure logs to replace simulated labels, and implementing the per-edit counterfactual optimization filter inside `PrescribeService`.
* **9.4 Closing Summary**

---