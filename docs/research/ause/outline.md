# Graph-Based Detection of Architectural Anti-Patterns and Prescriptive Refactoring in Distributed Publish–Subscribe Systems

* **Target Journal:** Automated Software Engineering (AuSE) — Springer
* **Target Venue:** Special Issue "Intelligent techniques for CI/CD, DevOps, software evolution, technical debt analysis, and refactoring recommendation."
* **Target Topic:** *Technical debt analysis* (the anti-pattern catalog, §4), *refactoring recommendation* (the prescriptive operators, §6), and *CI/CD/DevOps* (the delta-aware quality gate, §7) — three of the special issue's five named areas; no AutoML/NAS/LLM contribution is claimed.

## Abstract

Distributed publish–subscribe middleware (ROS 2, DDS, MQTT) decouples producers and consumers, but the resulting indirect dependency structure obscures how component failures cascade, and this architectural technical debt accumulates invisibly to code-level static analysis (SCA) tools. Unlike object-oriented design, which has mature anti-pattern catalogs, publish–subscribe architectures have no equivalent catalog, and existing structural diagnostic frameworks typically operate open-loop, ranking components by criticality without naming the responsible pathology or verifying a remedy. We address both gaps with SaG-Prescribe, a graph-based framework that (1) detects twenty-one named, severity-tiered publish–subscribe anti-patterns and bad smells, including Single Point of Failure, God Component, Broker Saturation, Chatty Pair, and QoS Policy Mismatch, via topological signatures over thirteen structural metrics and adaptive thresholds, validated against independent failure-simulation ground truth (Spearman rank correlation of 0.876; F1 score of 0.923); and (2) compiles the flagged components and patterns into a transformation policy of three targeted graph-mutation operators, logical topic splitting, physical anti-affinity reallocation, and transport QoS contract hardening, each re-verified by the same cascade simulator before being surfaced as a recommendation. The pipeline is operationalized as a delta-aware CI/CD quality gate blocking only newly introduced anti-patterns relative to a Git merge base. Across eight benchmark scenarios, the generated prescriptions reduce the System Risk Index in every case, with relative reductions between 1.4% and 15.9% (Wilcoxon signed-rank p = 0.0156). Component-level failure-impact reductions average only +4.61% and regress in two scenarios, motivating a per-edit acceptance filter as future work. The full generate-verify loop completes within CI/CD-compatible time budgets for most scenarios.

## Keywords

architectural anti-patterns; technical debt analysis; refactoring recommendation; publish-subscribe middleware; CI/CD quality gates; failure cascade simulation

---

## Outline

### 1. Introduction

* **1.1 Context and motivation:** Pub-sub decoupling as the backbone of cyber-physical/microservice/IoT systems; decoupling obscures failure propagation and the resulting architectural technical debt.
* **1.2 Two open gaps:** (a) the Architecture-Code Gap and the absence of a named, testable anti-pattern catalog for pub-sub topologies (SCA tools are topology-blind; OO/microservices catalogs don't transfer to broker-mediated communication); (b) the open-loop refactoring-recommendation gap — even topology-aware diagnostics (including our own prior SaG framework) rank criticality without naming the fault or verifying its remediation.
* **1.3 Proposed solution:** A unified detect→prescribe→gate pipeline — twenty-one-pattern catalog with empirically validated detection rules, feeding a closed-loop mutation-and-resimulation engine, operationalized as a delta-aware CI/CD gate.
* **1.4 Contributions:** Four explicit, numbered items: (1) the empirically-validated anti-pattern catalog and detection methodology; (2) the closed-loop prescriptive pipeline and its three operators, with an honestly-scoped mapping from operators to the five catalog patterns they directly automate (of twenty-one total; see §6.3); (3) the CI/CD gate; (4) the multi-scenario evaluation, including the honest component-level result.
* **1.5 Relationship to the Authors' Prior Work:** Positions this submission relative to the companion Software-as-a-Graph JSS submission and the RASSE 2025 conference paper on graph-based critical-component identification; clarifies that the catalog and prescriptive material here are original to this manuscript and not duplicated review.
* **1.6 Organization:** Standard structural overview of the remaining sections.

### 2. Background and Related Work

* **2.1 Publish–Subscribe Middleware Dependability:** Broker fault tolerance, Kafka, DDS QoS/latency literature.
* **2.2 Anti-Pattern and Code-Smell Catalogs:** Object-oriented design (Fowler's refactoring catalog; Brown et al.'s architectural AntiPatterns; Suryanarayana et al.'s design smells) and microservices smells (Richardson; Taibi et al.), establishing the template this catalog follows — named pattern, formal detection rule, refactoring strategy — while operating on system topology rather than code or REST call graphs.
* **2.3 Refactoring Recommendation and Architectural Technical Debt:** Code-scope recommenders (smell-driven, learning-based, LLM-based) vs. this paper's topology scope and verify-before-recommend model. Two of three prior citation slots are now fillable from §2.2's real references; the learning-based and LLM-based recommender slots remain pending, with no invented references used.
* **2.4 Search-Based Software Engineering and Architecture Optimization:** Open-loop vs. this paper's closed-loop verification.
* **2.5 Diagnostic Foundation (SaG):** The heterogeneous graph model and RMAV attribution this paper builds on; what is summarized vs. not repeated from the companion diagnostic paper.
* **2.6 Structural Criticality Analysis:** Why classical single-metric centrality (degree, betweenness alone) degrades on typed pub-sub graphs relative to the composite Q(v) score.

### 3. System Model and Code-Quality-Augmented Quality Attribution

* **3.1 Heterogeneous Graph Formulation:** Five node types, six structural edge types.
* **3.2 Derived DEPENDS_ON Projection:** App-to-App, App-to-Broker, App-to-Library, and Broker-to-Broker dependency edges, and the four architectural layers (app, infra, mw, system).
* **3.3 Code Quality Penalty (CQP):** The paper's explicit bridge from SCA metrics to architecture-level risk; feeds the Maintainability dimension below.
* **3.4 Multi-Dimensional Quality Attribution (RMAV):** Reliability/Maintainability/Availability/Vulnerability, AHP-weighted composite Q(v), adaptive box-plot criticality tiers. This section is the shared foundation consumed by both the detection catalog (§4) and the prescriptive engine (§6).

### 4. A Catalog of Architectural Anti-Patterns for Publish–Subscribe Systems

* **4.1 Anti-Patterns vs. Bad Smells:** Taxonomy and confidence distinction — structural configurations with well-understood failure modes vs. heuristic surface symptoms requiring human judgment.
* **4.2 Detection Methodology:** Thirteen structural metrics per component (PageRank, Reverse PageRank, betweenness, closeness, eigenvector centrality, in/out-degree, clustering coefficient, articulation point score, bridge ratio, QoS-weighted degree measures); rank-based vs. min-max normalization; adaptive box-plot thresholds (scale-invariant, distribution-aware, statistically grounded) in place of fixed global constants.
* **4.3 Catalog Overview:** Twenty-one patterns organized into three severity tiers (critical, high, medium) across four RMAV dimensions; summary table of pattern, severity, primary RMAV dimension, and formal detection rule, with full specifications and remediation strategies given in the companion technical reference (`docs/antipatterns.md`), cited rather than reproduced in full.
* **4.4 Representative Pattern Walkthroughs:** Five patterns selected to span severity tiers, RMAV dimensions, and detection technique diversity: Single Point of Failure (articulation-point-based, Availability), God Component (betweenness plus maintainability gate), Broker Overload / Hub-and-Spoke (the pub-sub instantiation of a classical topology anti-pattern, validated on a deliberately-encoded scenario), Chatty Pair (bidirectional edge-weight product, hidden coupling behind nominal decoupling), and QoS Mismatch (transport-contract-boundary detection unique to QoS-bearing middleware).
* **4.5 Empirical Validation of Detection:** Validation methodology (simulated failure impact as ground truth); headline metrics (Spearman correlation of 0.876 overall, rising to 0.943 at large scale; F1 of 0.923, precision of 0.912, recall of 0.857, Top-5 overlap of 0.80); pattern-specific evidence (Single Point of Failure F1 above 0.95; Hub-and-Spoke scenario confirmation); baseline comparison against single-metric centrality; the eight-scenario validation suite, foregrounding Scenario 06 (sparse microservices mesh) as the precision stress test — a well-structured topology should produce few or no findings — and Scenario 07 as the scalability benchmark.

### 5. Closed-Loop Optimization Objective

* **5.1 Formal Objective:** A minimization objective under a currently unconstrained modification budget; acceptance criterion of a positive change in the System Risk Index as implemented in `PrescribeService`; explicit statement that the candidate set is exactly the critical/high components and patterns identified in §4; note on the deferred margin-based criterion.

### 6. The SaG-Prescribe Prescriptive Pipeline

* **6.1 Hexagonal Core Abstraction:** `IGraphRepository`, `Neo4jRepository` vs. `MemoryRepository`.
* **6.2 Pipeline Stages 1–7:** Diagnostic foundation and anti-pattern detection, prescriptive generation, and review interface; explicit hand-off from §4's named findings to operator selection.
* **6.3 Three Refactoring Operators:** Logical topic splitting, physical anti-affinity reallocation, and transport QoS contract hardening — each formalized as a typed graph mutation rule, triggered by two independent signals: a generic RMAV criticality tier (any critical/high component can trigger any operator) and detected-problem name matching, the only channel that ties a mutation back to a specific catalog pattern. Following the name-matching channel, only five of the twenty-one patterns are directly wired to an operator — Single Point of Failure to anti-affinity reallocation; God Component, Bottleneck Edge, Failure Hub, and Hub-and-Spoke to topic splitting. Notably, QoS Mismatch has no wiring to QoS hardening despite the conceptual overlap — QoS hardening fires only from the generic criticality tier. The remaining sixteen patterns have no automated operator and stay advisory-only; this asymmetry between the twenty-one-pattern catalog and the narrower five-pattern automation footprint is disclosed explicitly rather than implied away.
* **6.4 Closed-Loop Verification Procedure:** Export, mutate, sandbox, resimulate, and measure the change in the System Risk Index; policies are accepted or rejected as a whole — there is no per-edit filter (see §5.1) — with the independence guarantee against circular leakage.

### 7. DevOps Integration and Delta-Aware CI/CD Gating

*(This section is the paper's most direct link to the special issue's CI/CD/DevOps bullet — kept prominent.)*

* **7.1 Automated Code Review Architecture:** `detect_antipatterns.py` surfacing §4's twenty-one detectors as a blocking CI check.
* **7.2 Delta-Aware Regression Semantics:** Merge-base diffing, waiver register for accepted legacy risk.
* **7.3 Exit-Code Protocol:** 0 (pass), 1 (warn), 2 (block).

### 8. Experimental Design

* **8.1 Research Questions:** Five research questions: RQ1 (detection efficacy and precision — does the catalog correlate with ground-truth impact, and does it avoid over-flagging well-structured systems?); RQ2 (prescriptive efficacy); RQ3 (operator contributions); RQ4 (component-level remediation efficacy); RQ5 (computational overhead and CI/CD feasibility for both detection and prescription).
* **8.2 Scenario Suites:** Eight scenarios (01–08) for detection validation, including the deterministic "Tiny Regression" smoke-test fixture (§4.5); the seven-scenario subset (01–07, autonomous vehicle through hyper-scale enterprise) used for prescriptive evaluation, explicitly noting and justifying the count mismatch — the smoke-test fixture carries no domain-representative signal for prescriptive evaluation.
* **8.3 Experimental Protocol:** Deterministic simulator, verified zero seed variance across five seeds.
* **8.4 Metrics:** Detection: Spearman correlation, F1/precision/recall, Top-k overlap. Prescription: System Risk Index formula, change in System Risk Index, operator counts, component-level deltas.

### 9. Results

* **9.1 Detection Efficacy and Precision (RQ1):** Headline validation metrics from §4.5; Scenario 06 precision-stress confirmation (few/no findings on a well-structured topology); Scenario 07 scalability confirmation.
* **9.2 Prescriptive Efficacy (RQ2):** System Risk Index improves in all seven scenarios (1.4%–15.9%), Wilcoxon signed-rank p = 0.0156.
* **9.3 Operator Contributions (RQ3):** Anti-affinity reallocation most frequent overall; QoS upgrades concentrate in high-loss profiles; contribution counts cross-referenced against the five name-matched patterns and the broader generic-criticality trigger described in §6.3.
* **9.4 Remediation Efficacy at Component Boundaries (RQ4):** Honest negative result — mean +4.61% component-level improvement but regressions in Hub-and-Spoke (−31.67%) and Hyper-Scale (−25.36%), traced to greedy unconditional policy application; motivates the per-edit acceptance filter.
* **9.5 Computational Overhead and CI/CD Feasibility (RQ5):** Detection gate sub-2 seconds to roughly 40 seconds across scales; full generate-verify loop under 65 seconds for six of seven scenarios, roughly 10.8 minutes at hyper-scale.

### 10. Discussion, Threats to Validity, and Conclusion

* **10.1 What Naming and Verifying Buys:** Over an unnamed criticality score — §9.1 and §9.4 read together.
* **10.2 Positioning in CI/CD and Technical-Debt Workflows:** Advisory prescriptions vs. blocking detection gate; the catalog as an architecture-review checklist (design review by checklist, as in aviation or surgery) for teams without full CI/CD automation.
* **10.3 Threats to Validity:** Construct validity (detection and verification both defined relative to the same discrete-event simulator; mitigation via operators/patterns grounded in established dependability practice; the operator-to-pattern linkage relies on substring matching of human-readable detected-problem names rather than a dedicated pattern-ID field, which is brittle to pattern-name changes and covers only five of the twenty-one catalog patterns directly, per §6.3); internal validity (greedy, unconditional whole-policy application with no per-edit filter; independence guarantee against circular leakage; stratified reporting discipline); external validity (synthetic parameterized topologies; catalog scope limited to the pub-sub communication paradigm; QoS-mismatch detection currently specified for DDS/ROS 2 and MQTT weight semantics); conclusion validity (small scenario sample of seven for the Wilcoxon test; single-run, verified-deterministic simulator configuration).
* **10.4 Limitations and Future Work:** Outlines seven prioritized next steps: (1) per-edit acceptance filter, highest priority, directly motivated by §9.4; (2) budget-constrained policy search; (3) stochastic cascade propagation; (4) operator ablation and learned policy ordering; (5) catalog extension to hybrid REST/event and non-DDS/MQTT QoS semantics; (6) real-system replication (ATM topology); (7) LLM-assisted PR generation, kept to a single sentence and not oversold.
* **10.5 Conclusion:** Summary of the end-to-end detect→prescribe→gate pipeline and headline results.

### 11. References
