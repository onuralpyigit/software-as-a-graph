# Graph Neural Networks for Reliability and Dependability Analysis in Complex Distributed Systems based on Publish–Subscribe Architecture

* **Target Journal:** Journal of Systems and Software (JSS) — Elsevier
* **Target Venue:** Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems" (VSI:AI4MSS)
* **Target Topic:** *AI for Reliability and Dependability Analysis in Complex ICT Systems*

## Abstract

Publish-subscribe middleware decouples data producers and consumers, improving scalability but obscuring the dependency chains along which a component's failure can cascade. Because runtime telemetry does not exist before deployment, and static code analysis tools are blind to system-level topology, identifying which components are critical, and why, remains difficult. We address this gap with Software-as-a-Graph (SaG), a static system analysis framework that models a publish-subscribe system as a typed, weighted, directed multigraph over five component classes and derives logical dependencies through typed projection rules. On this representation, we train a relation-specific heterogeneous graph neural network (GNN) with explicit quality-of-service (QoS) edge-feature injection ($HGL\text{-}QoS$) to forecast each component's cascading failure impact before deployment. We pair this learned predictor with an interpretable, Analytic Hierarchy Process-weighted composite score decomposing criticality into four orthogonal quality dimensions, so every diagnostic maps to a concrete remediation. Both predictors are validated against an independent discrete-event cascade simulator under a strict input-label independence guarantee. Across seven synthetic publish-subscribe topologies, the interpretable score suffices for known architectures (rank correlation above 0.87), while the heterogeneous predictor is required to generalize to unseen architectures (0.401 versus near zero for homogeneous baselines). A stratified analysis confirms consistent predictive strength across component types, and an end-to-end evaluation of prescriptive remediation operators yields a mixed result, exposing a gap between the intended per-edit acceptance test and the current implementation. Finally, the framework operates as a blocking continuous-integration and continuous-deployment (CI/CD) quality gate, evaluating regressions in seconds with perfect precision and recall on injected structural faults.

## Highlights

* Heterogeneous GNN ($HGL\text{-}QoS$) generalizes cascade prediction to unseen architectures.
* AHP-weighted composite score explains criticality via four orthogonal dimensions.
* Typed multigraph model bridges the Architecture-Code Gap before deployment.
* Delta-aware CI/CD gate blocks structural regressions in seconds, pre-deployment.

## Keywords

heterogeneous graph neural networks; publish-subscribe systems; architectural dependability; cascading failure; static system analysis; pre-deployment verification; CI/CD quality gate.

---

## Detailed Section Outline

### 1. Introduction

* **1.1 Motivation:** The rapid proliferation of publish–subscribe-backed distributed topologies (DDS, MQTT, ROS 2, Kafka) across high-availability domains. The core paradox: spatial, temporal, and synchronization decoupling that enables horizontal scaling simultaneously obscures logical failure-propagation paths. Architectural hardening is cheapest and least disruptive while a system is still a design, making pre-deployment estimation—before runtime telemetry exists—highly valuable.
* **1.2 The Architecture-Code Gap & Problem Statement:** Formulates the two coupled sub-problems solved without runtime execution data: (i) *quality attribution* (interpretable, per-dimension criticality metrics) and (ii) *failure-impact analysis* (predicted cascade reach). Explains the "Architecture-Code Gap" wherein clean, static-code-analysis-passing source code can still sit atop a fragile deployment topology prone to systemic collapse.
* **1.3 Limitations of Existing Approaches:** Identifies three structural gaps: (1) Static Code Analysis (SCA) platforms are blind to global inter-component topology; (2) runtime chaos engineering and fault-tolerance mechanisms carry operational risk and require a deployed system; (3) topology-only and homogeneous learning centralities collapse typed semantics (applications, topics, brokers) into flattened scalars, causing representation collapse.
* **1.4 Our Approach & Research Questions:** Introduces SaG's typed multigraph, multi-dimensional quality attribution (RMAV), dual interpretable/learned impact predictors, prescriptive remediation, and continuous CI/CD gating. Formally frames the four research questions:
* *RQ1 (Interpretable vs. Learned):* When does deterministic multi-attribute scoring suffice, and when is a heterogeneous GNN required to recover the simulated impact ordering on unseen architectures?
* *RQ2 (Typing & Structural Anomalies):* What failure modes does multi-dimensional, typed attribution expose that a single-score topological centrality conceals?
* *RQ3 (QoS Impact):* How does explicit multi-attribute QoS contract edge-feature injection affect in-distribution convergence versus out-of-distribution Leave-One-Scenario-Out (LOSO) generalizability?
* *RQ4 (CI/CD Operationalization):* What is the feasibility and performance overhead of running this analysis as a blocking CI/CD quality gate?


* **1.5 Contributions:** Enumerates the five key contributions: (1) typed graph model with hierarchical code-level SCA metric integration; (2) automated, delta-aware CI/CD quality gate; (3) failure-impact analysis under a strict view-based independence guarantee ($Q(v)$ and $HGL\text{-}QoS$); (4) a Generate→Verify prescriptive remediation framework, exposing an honestly reported implementation gap; (5) a stratified (per-node-type) evaluation of the $Q$–$I$ correlation.
* **1.6 Relationship to the Authors' Prior Work:** Positions this submission as a consolidation and expansion of an initial structural baseline framing (multi-layer graph dependency analysis) [Anon-A] with the heterogeneous-GNN predictor, RMAV attribution, and CI/CD gating into a single standalone paper. Clarifies that no companion GNN manuscript is under parallel review and that placeholder expert panel datasets have been fully purged.
* **1.7 Organization:** Standard structural overview of the remaining sections.

### 2. Related Work

* **2.1 Publish–Subscribe Middleware and Dependability:** Reviews classical runtime fault tolerance, reliable event dissemination overlays, and consensus protocols in DDS/MQTT/Kafka. Re-positions SaG as a complementary, earlier-lifecycle intervention focused on design-time estimation rather than runtime mitigation.
* **2.2 Static Code Analysis (SCA) vs. Static System Analysis (SSA):** Delineates AST-level code metrics (cyclomatic complexity, LCOM) from system-level topological dependencies, showing how SaG bridges the gap by ingesting SCA metrics as vertex properties to evaluate systemic fragility.
* **2.3 Continuous Pre-Deployment Verification and Gating:** Contrasts late-lifecycle chaos engineering with shifting architectural verification left. Establishes the precedent of SonarQube's delta-gating ("Clean as You Code") and demonstrates how SaG maps this paradigm onto topological descriptors.
* **2.4 Structural Criticality Analysis:** Evaluates classical graph centrality metrics (betweenness, degree, articulation points) and cascading-failure models. Highlights their limitation regarding dimensional collapse and their blindness to type-specific simultaneous blast modes.
* **2.5 Learning-Based Criticality Prediction:** Reviews deep reinforcement learning and GNN centralities (FINDER, DrBC, PowerGraph). Argues for relation-specific heterogeneous graph architectures (RGCN, HAN, HGT, MAGNN) to prevent representation and over-smoothing collapse in hub-dominated topologies.
* **2.6 Quality Attributes and Multi-Criteria Scoring:** Grounds software quality dimensions (RMAV) in the multi-criteria decision-theoretic structure of the Analytic Hierarchy Process (AHP), detailing the use of multi-criteria decomposition as an explicit *attribution* and explanation mechanism.
* **2.7 Architectural Remediation and Anti-Pattern Detection:** Compares heuristic refactoring recommendation systems with SaG's counterfactual simulation oracle validation scheme.
* **2.8 Positioning:** Synthesis of the five-way gap closed jointly by SaG's multi-layered pre-deployment pipeline.

### 3. The Software-as-a-Graph (SaG) Model

* **3.1 Nodes, Edges, and the Formal Object:** Formal definition of the typed, weighted, directed multigraph $G = (V, E, \tau_V, \tau_E, w_E, w_V)$. Outlines the five node types ($V_{\text{app}}, V_{\text{broker}}, V_{\text{topic}}, V_{\text{node}}, V_{\text{lib}}$) and six structural edge relations (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`).
* **3.2 QoS-Aware Edge and Vertex Weights:** Formulates the two-stage edge weight allocation using AHP-derived weights for QoS parameters: $\text{QoS\_score} = 0.30r + 0.40d + 0.30p$ (mapping `RELIABLE`, `PERSISTENT`, etc.), combined with log-scale size normalization ($\text{size\_norm}$) via $\beta=0.85$ to set a composite weight $w(e) \in [0.01, 1.0]$. Details type-specific vertex weight ($w_V$) aggregation rules, highlighting the fan-out-amplified library weighting.
* **3.3 Derived Dependencies — the `DEPENDS_ON` Projection:** Details the six typed projection rules mapping raw configurations to directional logical dependencies (e.g., subscriber-to-publisher via a topic). Explains the qualitative division between sequential cascades (Rule 1) and simultaneous blasts (Rule 5, shared libraries). Formulates multi-topic consolidation where $\text{edge.weight} = \max w(t)$ and `path_count` measures multi-channel connectivity.
* **3.4 Ingestion of Code-Level SCA Metrics:** Details the schema for importing SonarQube `cm_*` properties (LOC, WMC, LCOM, CBO, RFC, technical debt, bugs, vulnerabilities) as local vertex attributes that feed the Code Quality Penalty (CQP).
* **3.5 Graph Views and Multi-Layer Projections:** Articulates the structural separation enforcing the independence guarantee: $G_{\text{structural}}$ drives the simulator while $G_{\text{analysis}}(\ell)$ drives attribution. Outlines the four layer scopes (Application, Infrastructure, Middleware, System) and their mapping to a MIL-STD-498 reporting rollup hierarchy.
* **3.6 Running Example:** Walkthrough of a three-app, one-topic, one-shared-library system contrasting sequential cascade paths with simultaneous blast footprints.

### 4. Multi-Dimensional Quality Attribution (The Interpretable Path)

* **4.1 Four Orthogonal Dimensions:** Defines Reliability, Maintainability, Availability, and Vulnerability (RMAV), showing how disjoint structural metric allocations guarantee that a component's profile directly explains the nature of its vulnerability.
* **4.2 RMAV Formulas:** Complete formal mathematical presentation of the dimension metrics:
* *Reliability $R(v)$:* Reverse PageRank ($\mathrm{RPR}$), in-degree ($\mathrm{DG\_in}$), and Enhanced Cascade Depth Potential ($\mathrm{CDPot\_enh}$), plus Topic-specific fan-out criticality ($\mathrm{FOC}$).
* *Maintainability $M(v)$:* Betweenness centrality ($\mathrm{BT}$), efferent QoS-weighted out-degree ($w\_\text{out}$), Code Quality Penalty ($\mathrm{CQP}$) integrating normalized LOC/Complexity/LCOM, enhanced coupling risk ($\mathrm{CouplingRisk\_enh}$), and local clustering ($\mathrm{CC}$).
* *Availability $A(v)$:* Directed articulation score ($\mathrm{AP\_c\_directed}$), QoS-weighted SPOF severity ($\mathrm{QSPOF}$), bridge ratios ($\mathrm{BR}$), and the Connectivity Degradation Index ($\mathrm{CDI}$).
* *Vulnerability $V(v)$:* Reverse eigenvector centrality ($\mathrm{REV}$), reverse harmonic closeness ($\mathrm{RCL}$), and afferent QoS-weighted in-degree ($w\_\text{in}$).


* **4.3 The Composite Score $Q(v)$:** Defines the AHP matrix aggregation. Shows the raw weights $(0.43, 0.24, 0.17, 0.16)$ with consistent $\mathrm{CR}<0.02$. Explains the application of a shrinkage parameter ($\lambda=0.70$) to prevent extreme weights over small comparison sets, yielding operational weights of $(0.38, 0.24, 0.19, 0.19)$.
* **4.4 Adaptive Criticality Classification:** Box-plot quartile fence rule ($Q_3 + 1.5\,\mathrm{IQR}$) mapping scores to discrete tiers (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `MINIMAL`) with a percentile fallback for sparse graphs.
* **4.5 Determinism and the Independence Guarantee:** Formal definition of the structural decoupling that prevents label-to-feature feedback or transductive leakage.
* **4.6 Worked Attribution:** Applying the formulas to the running example (§3.6) to demonstrate distinct RMAV shapes for a SPOF broker, a cascade origin application, and a shared library.

### 5. Failure-Impact Analysis via Heterogeneous GNN and Interpretable Forecasting

* **5.1 Ground-Truth Impact $I(v)$:** Formulates the discrete-event simulation metric on $G_{\text{structural}}$ combining reachability loss, graph fragmentation, throughput loss, and flow disruption. Outlines the cascade conditions: a $0.2$ propagation threshold, 10-epoch horizon, and multi-seed averaging.
* **5.2 Two Predictors over the Same Model:** Juxtaposes the deterministic $Q(v)$ path with the learned inductive Heterogeneous Graph Transformer (HGT), defining the relation-specific parameterization of the base **HGL** (QoS-masked) and **$HGL\text{-}QoS$** (QoS-encoded) variants.
* **5.3 The Independence Guarantee:** Re-enforces view-based isolation ensuring simulation outputs never feed back into feature states.
* **5.4 The Shared-Library Blast Mechanism — A Negative Result:** Documents a rigorous empirical test across 165 library nodes revealing that $I(v) \le Q(v)$ globally (highest library $Q=0.422$ with an impact of $I=0.086$), showing that the composite score is conservative rather than blind to library risks. Discloses why this is reported as an honest negative result.
* **5.5 Stratified Correlation — A Consistency Check:** Evaluates pooled Spearman correlation ($\rho = 0.374$) against type-stratified ranges ($\rho = 0.322$–$0.429$), ruling out Simpson's-paradox masking and validating consistent predictive alignment across all classes.

### 6. Prescriptive Remediation and CI/CD Quality Gating

* **6.1 A Two-Phase Generate–Verify Procedure:** Outlines the separation between structural candidate generation and simulation-driven verification on counterfactual subgraphs.
* **6.2 Remediation Operators:** Defines the four formal operators: `RedundancyInsertion`, `PathDiversification`, `FanOutReduction`, and `SharedTopicReduction`, mapped to structural triggers and target dimensions.
* **6.3 Triggering on Blast Radius, not $Q(v)$:** Justifies why `FanOutReduction` triggers directly on structural fan-out and consumer counts to catch high-blast, lower-$Q$ libraries or channels.
* **6.4 Acceptance Criterion (Target Design) and the Open Implementation Gap:** Details the target statistical filter ($\Delta I > \kappa\,\sigma_{\text{seed}}$ across threshold sweeps) and discloses the implementation gap where `PrescribeService` currently applies compiled policies unconditionally without an active per-edit filter.
* **6.5 Independence Invariants:** Formalizes the three immutable invariants governing candidate isolation and verification passes within any run.
* **6.6 CI/CD Quality Gate Implementation:** Details the deployment of `detect_antipatterns.py` using delta semantics against the merge base, a waiver register for accepted risks, the three-tier exit-code protocol (0=pass, 1=warning, 2=blocked), and the role of the database-free `MemoryRepository` in eliminating live Neo4j network latencies.
* **6.7 What Remediation Currently Yields — A Mixed Result:** Presents the cross-scenario evaluation table showing a $+4.61\%$ cross-scenario mean that masks severe resilience regressions (up to $-31.67\%$) in 3 of 7 scenarios, directly demonstrating the downstream impact of the unbuilt per-edit filter.

### 7. Experimental Setup

* **7.1 Datasets:** Describes the seven synthetic, industrially-styled publish–subscribe topologies (Autonomous Vehicle, Financial Trading, Healthcare, Hub-and-Spoke, IoT Smart City, Microservices Mesh, Large-Scale Enterprise) encompassing 1,545 total components. Acknowledges the explicit exclusion of real-world or expert-ranking datasets.
* **7.2 Predictors and Baselines:** Fixes the experimental matrix: RMAV/$Q$, HGL, $HGL\text{-}QoS$, homogeneous GL/GL-QoS, and structural centralities Topo-BL/Topo-QoS.
* **7.3 Evaluation Metrics:** Sets ranking metrics (Spearman $\rho$, NDCG@10, Top-K overlap), classification metrics (F1, SPOF-F1), calibration metrics (RMSE, MAE), bootstrap confidence intervals ($B=2000$), and Wilcoxon signed-rank tests. Targets validation passes at $\rho \ge 0.70$ and $F1 \ge 0.80$.
* **7.4 Protocols:** Delineates in-distribution per-scenario training from inductive Leave-One-Scenario-Out (LOSO) cross-validation; establishes 5-seed configuration averages.
* **7.5 Canonical Simulator and Reproducibility:** Explicitly parameterizes the environment: step-function blast semantics, propagation threshold of $0.2$, 10-epoch horizon, and fixed seed seeds $\{42, 123, 456, 789, 2024\}$.

### 8. Results

* **8.1 RQ1 — Interpretable Attribution vs. Learning:** Reports a split outcome: in-distribution, deterministic $Q(v)$ attribution dominates ($\rho > 0.87$, $F1 > 0.90$), matching or beating the learned model. Out-of-distribution (LOSO), $HGL\text{-}QoS$ decisively outperforms all baselines ($\rho = 0.401$ vs. near-zero for homogeneous GATs). Documents the measurement context caveat between in-distribution splits.
* **8.2 RQ2 — What Taking Type Seriously Shows (and Does Not Show):** Quantifies structural heterogeneity gains ($\Delta F1 = +0.284$ in-distribution, $\Delta\rho = +0.286$ LOSO). Re-evaluates the shared-library blast-radius result and type-stratified correlations.
* **8.3 RQ3 and Robustness — Ablations and Sensitivity:** Discloses an in-distribution null result for QoS encoding vs. its significant cross-architecture transfer value ($\rho=0.401$ vs. $0.307$). Reports sensitivity trends for AHP shrinkage ($\lambda$) and propagation thresholds.
* **8.4 RQ4 — Feasibility and Performance of the CI/CD Quality Gate:** Details execution footprint across presets ($<2\,\text{s}$ tiny, $\approx 5\,\text{s}$ medium, $\approx 40\,\text{s}$ xlarge). Reports perfect precision and recall (1.0/1.0) on structural regression blocks with zero false positives on waived conditions.

### 9. Discussion, Threats to Validity, and Conclusion

* **9.1 Interpretation:** Key takeaways regarding the complementary division of labor between attribution and graph learning, the discipline of stratification, the implementation gap in remediation, and the continuous integration of SSA gates.
* **9.2 Threats to Validity:** Formulates Construct validity (simulation limits), Internal validity (independence guarantees), and External validity (synthetic transfer boundaries).
* **9.3 Limitations and Future Work:** Outlines 5 discrete actionable paths: (1) like-for-like head-to-head tracking table; (2) enriched Vulnerability proxies; (3) real-world/expert-panel validation; (4) implementing the per-edit counterfactual filter in `PrescribeService`; (5) operational failure data calibration.
* **9.4 Conclusion:** Final summary of SaG's capacity to bridge the Architecture-Code Gap pre-deployment via a typed multigraph paradigm.

### 10. References

* Establishes a verified, real-world citation listing spanning 15 entries across foundational pub-sub specifications [1–3], network-science centrality algorithms [4–6], graph deep learning centralities [7–9], relation-specific heterogeneous graph architectures [10–13], over-smoothing hazards [14], and AHP frameworks [15].