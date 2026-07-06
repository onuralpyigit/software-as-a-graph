# Graph Neural Networks for Reliability and Dependability Analysis in Complex Distributed Systems based on Publish–Subscribe Architecture

* **Target Journal:** Journal of Systems and Software (JSS) — Elsevier
* **Target Venue:** Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems" (VSI:AI4MSS) — topic *"AI for Reliability and Dependability Analysis in Complex ICT Systems"*

## Abstract

Publish-subscribe middleware decouples data producers and consumers, improving scalability but
obscuring the dependency chains along which a component's failure can cascade. Because runtime
telemetry does not exist before deployment, and static code analysis tools are blind to
system-level topology, identifying which components are critical, and why, remains difficult. We
address this gap with Software-as-a-Graph (SaG), a static system analysis framework that models a
publish-subscribe system as a typed, weighted, directed multigraph over five component classes and
derives logical dependencies through typed projection rules. On this representation, we train a
relation-specific heterogeneous graph neural network (GNN) with explicit quality-of-service (QoS)
edge-feature injection (HGL-QoS) to forecast each component's cascading failure impact before
deployment. We pair this learned predictor with an interpretable, Analytic Hierarchy
Process-weighted composite score decomposing criticality into four orthogonal quality dimensions,
so every diagnostic maps to a concrete remediation. Both predictors are validated against an
independent discrete-event cascade simulator under a strict input-label independence guarantee.
Across seven synthetic publish-subscribe topologies, the interpretable score suffices for known
architectures (rank correlation above 0.87), while the heterogeneous predictor is required to
generalize to unseen architectures (0.401 versus near zero for homogeneous baselines). A stratified
analysis confirms consistent predictive strength across component types, and an end-to-end
evaluation of prescriptive remediation operators yields a mixed result, exposing a gap between the
intended per-edit acceptance test and the current implementation. Finally, the framework operates
as a blocking continuous-integration and continuous-deployment (CI/CD) quality gate, evaluating
regressions in seconds with perfect precision and recall on injected structural faults.

## Highlights

* Heterogeneous GNN (HGL-QoS) generalizes cascade prediction to unseen architectures.
* AHP-weighted composite score explains criticality via four orthogonal dimensions.
* Typed multigraph model bridges the Architecture-Code Gap before deployment.
* Delta-aware CI/CD gate blocks structural regressions in seconds, pre-deployment.
* Blast-mismatch and remediation gaps are reported as honest negative results.

## Keywords

heterogeneous graph neural networks; publish-subscribe systems; architectural dependability;
cascading failure; static system analysis; pre-deployment verification; CI/CD quality gate.

## Outline

### 1. Introduction

* **1.1 Motivation:** The rapid growth of pub-sub-backed high-availability distributed architectures (DDS, MQTT, ROS 2, Kafka). The core paradox: spatial/temporal decoupling that enables scaling simultaneously obscures logical failure-propagation paths, and pre-deployment — before any telemetry exists — is exactly when hardening is cheapest and this reasoning is most valuable.
* **1.2 The Architecture-Code Gap & Problem Statement:** Two coupled sub-problems posed without runtime data — (i) *quality attribution* (interpretable, per-dimension criticality) and (ii) *failure-impact analysis* (predicted cascade impact) — and why clean, SCA-passing source code can still sit atop a fragile deployment topology.
* **1.3 Limitations of Existing Approaches:** Three gaps — Static Code Analysis is blind to inter-component topology; runtime/chaos-engineering techniques presuppose a deployed system and carry operational risk; topology-only and homogeneous learning-based centrality collapse distinct failure mechanisms (SPOF vs. cascade hub vs. maintainability bottleneck) into one scalar.
* **1.4 Our Approach & Research Questions:** Introduces SaG's typed multigraph, RMAV attribution, dual interpretable/learned impact predictors, prescriptive remediation, and CI/CD gating; states the four research questions:
  * *RQ1 (Interpretable vs. Learned):* When does deterministic multi-attribute attribution suffice, and when is a heterogeneous GNN required to recover the simulated impact ordering on unseen architectures?
  * *RQ2 (Typing & Structural Anomalies):* What failure modes does multi-dimensional, typed attribution expose that a single-score topological centrality conceals?
  * *RQ3 (QoS Impact):* How does explicit multi-attribute QoS edge-feature injection affect in-distribution convergence versus out-of-distribution (LOSO) generalizability?
  * *RQ4 (CI/CD Operationalization):* What is the feasibility and performance overhead of running this analysis as a blocking CI/CD quality gate?
* **1.5 Contributions:** (1) typed graph model with hierarchical SCA-metric integration; (2) automated, delta-aware CI/CD quality gate; (3) failure-impact analysis under an independence guarantee (interpretable $Q(v)$ + $HGL\text{-}QoS$); (4) a Generate→Verify prescriptive remediation stage, with an honestly reported implementation gap; (5) a stratified (per-node-type) evaluation of the $Q$–$I$ correlation.
* **1.6 Relationship to the Authors' Prior Work:** Positions this submission as a consolidation of an earlier structural-baseline framing (multi-layer graph dependency analysis) with the heterogeneous-GNN predictor, RMAV attribution, and CI/CD gating into one self-contained paper for this special issue; no companion GNN manuscript is under parallel review.
* **1.7 Organization:** One-paragraph section-by-section roadmap.

### 2. Related Work

* **2.1 Publish–Subscribe Middleware and Dependability:** Runtime fault tolerance, reliable dissemination, and recovery in DDS/MQTT/Kafka ecosystems, and how this work's concern is complementary and earlier in the lifecycle (pre-deployment estimation, not runtime reaction).
* **2.2 Static Code Analysis (SCA) vs. Static System Analysis (SSA):** The historical expansion from AST-level intra-component metrics toward global, inter-component topology analysis that ingests SCA metrics as node properties rather than replacing them.
* **2.3 Continuous Pre-Deployment Verification and Gating:** Chaos engineering's runtime/late-lifecycle limitations vs. shifting verification left into CI/CD via Architecture-as-Code, and the "Clean as You Code" delta-gating precedent this paper's gate adopts.
* **2.4 Structural Criticality Analysis:** Classical centralities, articulation points, and cascading-failure/interdependent-network studies; their core limitation — dimensional collapse and blindness to the shared-library simultaneous-blast mode.
* **2.5 Learning-Based Criticality Prediction:** FINDER, DrBC, PowerGraph, and the homogeneous-graph assumption most such methods make; the case for relation-specific heterogeneous architectures (RGCN, HAN, HGT, MAGNN) and the over-smoothing hazard in dense, hub-dominated graphs.
* **2.6 Quality Attributes and Multi-Criteria Scoring:** AHP as a principled, auditable multi-criteria weighting method; the gap this paper fills — using multi-criteria decomposition itself as the *attribution* mechanism, not merely a prioritization aid.
* **2.7 Architectural Remediation and Anti-Pattern Detection:** Prior refactoring-recommendation work vs. this paper's counterfactual-simulation acceptance test for candidate edits.
* **2.8 Positioning:** Synthesis of the five-way gap (runtime-only, code-only, untyped-structural, untyped-learned, prioritization-only multi-criteria) that Software-as-a-Graph closes jointly.

### 3. The Software-as-a-Graph (SaG) Model

* **3.1 Nodes, Edges, and the Formal Object:** Definition of the typed, weighted, directed multigraph $G = (V, E, \tau_V, \tau_E, w_E, w_V)$; the five node types (App, Broker, Topic, Node, Library) and six structural edge types (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`).
* **3.2 QoS-Aware Edge and Vertex Weights:** $\text{QoS\_score} = 0.30r + 0.40d + 0.30p$; $\text{size\_norm}$; composite $w(e) = \beta\cdot\text{QoS\_score} + (1-\beta)\cdot\text{size\_norm}$ ($\beta = 0.85$, floor $0.01$); type-specific $w_V$ aggregation, including fan-out-amplified library weighting.
* **3.3 Derived Dependencies — the `DEPENDS_ON` Projection:** The six projection rules (`app_to_app`, `app_to_broker`, `node_to_node`, `node_to_broker`, `app_to_lib`, `broker_to_broker`); multi-topic consolidation ($\text{edge.weight} = \max_t w(t)$, `path_count`); the qualitative contrast between Rule 1's *sequential cascade* and Rule 5's *simultaneous blast*.
* **3.4 Ingestion of Code-Level SCA Metrics:** SonarQube-derived `cm_*` vertex properties (LOC, WMC, LCOM, CBO, RFC, `sqale_debt_ratio`, bugs, vulnerabilities) feeding the Code Quality Penalty (§4.2).
* **3.5 Graph Views and Multi-Layer Projections:** The two-view separation — $G_{\text{structural}}$ (simulator input) vs. $G_{\text{analysis}}(\ell)$ (attribution/prediction input) — as the load-bearing independence guarantee; the four analytical layers (Application, Infrastructure, Middleware, System) and MIL-STD-498 rollup hierarchy.
* **3.6 Running Example:** A three-application, one-topic, one-shared-library walkthrough contrasting cascade vs. blast failure semantics.

### 4. Multi-Dimensional Quality Attribution (The Interpretable Path)

* **4.1 Four Orthogonal Dimensions:** Reliability, Maintainability, Availability, Vulnerability (RMAV); each fed by disjoint structural metrics by design, so a component's profile is itself its explanation.
* **4.2 RMAV Formulas:** $R(v)$ (RPR, DG_in, CDPot_enh, with a topic-specific fan-out form); $M(v)$ (betweenness, $w\_out$, Code Quality Penalty, CouplingRisk, class cohesion); $A(v)$ (directed articulation score, QSPOF, bridge risk, CDI, weight); $V(v)$ (reverse-reachability REV, RCL, $w\_in$, on the transpose graph).
* **4.3 The Composite Score $Q(v)$:** AHP-derived weights $(w_A, w_R, w_M, w_V) = (0.43, 0.24, 0.17, 0.16)$ at $\mathrm{CR}\approx 0.02$; shrinkage toward a uniform prior ($\lambda = 0.70$) yielding operational weights $(0.38, 0.24, 0.19, 0.19)$.
* **4.4 Adaptive Criticality Classification:** Box-plot fences ($Q_3 + 1.5\,\mathrm{IQR}$) mapping to `CRITICAL`/`HIGH`/`MEDIUM`/`LOW`/`MINIMAL`, applied per-dimension and to the composite, with a percentile fallback for small graphs.
* **4.5 Determinism and the Independence Guarantee:** Formal statement that every $Q(v)$ input is a structural metric of $G_{\text{analysis}}$, disjoint from the simulation that produces $I(v)$.
* **4.6 Worked Attribution:** Applying §3.6's running example to show how a SPOF, a cascade origin, and a shared library each produce a distinct RMAV profile.

### 5. Failure-Impact Analysis via Heterogeneous GNN and Interpretable Forecasting

* **5.1 Ground-Truth Impact $I(v)$:** Discrete-event simulation on $G_{\text{structural}}$; $I(v) = 0.35\,\text{reachability\_loss} + 0.25\,\text{fragmentation} + 0.25\,\text{throughput\_loss} + 0.15\,\text{flow\_disruption}$; cascade propagation threshold ($0.2$), 10-epoch horizon, 5-seed averaging.
* **5.2 Two Predictors over the Same Model:** The deterministic $Q(v)$ vs. the Heterogeneous Graph Transformer, with QoS-masked (**HGL**) and QoS-encoded (**$HGL\text{-}QoS$**) variants, both trained inductively against simulator labels.
* **5.3 The Independence Guarantee:** Two structural properties — distinct graph views for features vs. labels, and no simulation output ever fed back as a feature — that license the paper's pre-deployment claims.
* **5.4 The Shared-Library Blast Mechanism — A Negative Result:** Direct empirical test against the hypothesized low-$Q$/high-$I$ mismatch across 165 library nodes; not found ($Q_{\max} = 0.422$, $I = 0.086$; $I(v) \le Q(v)$ globally); reported honestly as a negative result rather than adjusted to fit.
* **5.5 Stratified Correlation — A Consistency Check:** Pooled $\rho = 0.374$ vs. per-type $\rho = 0.322$–$0.429$; no Simpson's-paradox-style masking found, validating stratified reporting as sound practice independent of outcome.

### 6. Prescriptive Remediation and CI/CD Quality Gating

* **6.1 A Two-Phase Generate–Verify Procedure:** Generate reads structure/attribution only, never $I(v)$; Verify re-runs the canonical simulator on a counterfactual graph $G' = e(G)$.
* **6.2 Remediation Operators:** `RedundancyInsertion`, `PathDiversification`, `FanOutReduction`, `SharedTopicReduction`, each keyed to a structural trigger and an RMAV dimension.
* **6.3 Triggering on Blast Radius, not $Q(v)$:** Why `FanOutReduction` fires on raw structural fan-out signals rather than the composite score, so it still catches low-$Q$/high-blast components.
* **6.4 Acceptance Criterion (Target Design) and the Open Implementation Gap:** The intended per-edit test $\Delta I > \kappa\,\sigma_{\text{seed}}$ across the `propagation_threshold` sweep; honest disclosure that `PrescribeService` currently applies the compiled policy unconditionally, with no per-edit filter wired in.
* **6.5 Independence Invariants:** Generate never reads $I(v)$; Verify re-invokes the canonical simulator from scratch; no Verify result feeds back into Generate within a run.
* **6.6 CI/CD Quality Gate Implementation:** Delta-aware evaluation against the merge base; the waiver register for intentional, risk-accepted SPOFs; the exit-code protocol (0 = clean/waived, 1 = medium warning, 2 = blocked); the `MemoryRepository`'s role in eliminating live-database latency.
* **6.7 What Remediation Currently Yields — A Mixed Result:** Per-scenario $\Delta I$ table; cross-scenario mean $+4.61\%$ masking $-31.67\%$/$-27.66\%$/$-25.36\%$ regressions in 3 of 7 scenarios, directly attributed to the §6.4 implementation gap.

### 7. Experimental Setup

* **7.1 Datasets:** Seven synthetic, industrially-styled pub-sub scenarios (autonomous vehicle, financial trading, healthcare, hub-and-spoke enterprise, IoT smart city, microservices mesh, large-scale enterprise), `tiny`–`xlarge` scale presets, no real-world validation set in this submission.
* **7.2 Predictors and Baselines:** RMAV/$Q$, HGL, $HGL\text{-}QoS$, GL/GL-QoS (homogeneous), Topo-BL/Topo-QoS (non-learning centrality), and what each contrast isolates (RQ1–RQ3).
* **7.3 Evaluation Metrics:** Spearman $\rho$ (primary), NDCG@10, Top-K overlap, precision/recall/F1 (incl. SPOF-F1), RMSE/MAE, mandatory per-node-type stratification, bootstrap CIs ($B=2000$), paired Wilcoxon tests; validation gates $\rho \ge 0.70$, $F_1 \ge 0.80$.
* **7.4 Protocols:** In-distribution per-scenario evaluation vs. inductive Leave-One-Scenario-Out (LOSO) cross-validation; five-seed averaging with reported $\sigma_{\text{seed}}$.
* **7.5 Canonical Simulator and Reproducibility:** `FailureSimulator` configuration (step-function blast semantics, `propagation_threshold = 0.2`, 10 epochs, seeds $\{42, 123, 456, 789, 2024\}$) shared identically across §5–§8.

### 8. Results

* **8.1 RQ1 — Interpretable Attribution vs. Learning:** In-distribution, $Q(v)$ suffices ($\rho > 0.87$, $F_1 > 0.90$) and matches/exceeds the learned predictor's in-distribution mean; out-of-distribution (LOSO), $HGL\text{-}QoS$ decisively wins ($\rho = 0.401$ vs. $\approx 0$ for homogeneous baselines) — a regime-dependent, not either/or, answer.
* **8.2 RQ2 — What Taking Type Seriously Shows (and Does Not Show):** Heterogeneity as the dominant source of predictive gain ($\Delta F_1 = +0.284$ in-distribution, $\Delta\rho = +0.286$ LOSO); the shared-library blast test (negative result); the stratified-correlation consistency check (no Simpson's-paradox masking).
* **8.3 RQ3 and Robustness — Ablations and Sensitivity:** QoS encoding's in-distribution null result vs. its out-of-distribution transfer benefit ($\rho = 0.401$ vs. $0.307$); AHP shrinkage-parameter ($\lambda$) sensitivity sweep; propagation-threshold sensitivity.
* **8.4 RQ4 — Feasibility and Performance of the CI/CD Quality Gate:** Execution-time footprint by scale ($<2\,\text{s}$ to $\approx 40\,\text{s}$); 100% precision/recall on injected structural regressions; zero false positives on waived/pre-existing findings.

### 9. Discussion, Threats to Validity, and Conclusion

* **9.1 Interpretation:** Four synthesizing points — the interpretable/learned boundary is a division of labor, not a contest; typing is a methodological discipline whose value does not require a dramatic reversal; remediation is not yet closed-loop and we say so; CI/CD gating operationalizes the diagnostics continuously.
* **9.2 Threats to Validity:** Construct validity (simulation-derived ground truth; comparative not absolute claims), Internal validity (independence guarantee against circular validation; bootstrap CIs and significance tests), External validity (synthetic-only suite; LOSO tests synthetic-to-synthetic transfer only).
* **9.3 Limitations and Future Work:** Thin Vulnerability-dimension proxies; need for real-world/expert-ranking validation (explicitly not claimed as executed in this submission); expanding the remediation operator set and deriving $\kappa$ empirically; implementing the missing per-edit acceptance filter in `PrescribeService` as the single most concrete next step; eventual calibration against observed production failure data.
* **9.4 Conclusion:** Closing synthesis — SaG bridges the Architecture-Code Gap pre-deployment via typed multigraph modeling, dual interpretable/learned impact prediction under an independence guarantee, and continuous CI/CD gating; negative and mixed results are reported as findings, not smoothed over.

### 10. References

* 15 numbered references spanning pub-sub foundations [1–3], network-science centrality and cascading-failure theory [4–6], learning-based key-node identification and benchmarks [7–9], heterogeneous GNN architectures [10–13], over-smoothing [14], and AHP [15]. All entries verified as real, published works (including PowerGraph [9], confirmed via its NeurIPS 2024 Datasets and Benchmarks Track listing).
