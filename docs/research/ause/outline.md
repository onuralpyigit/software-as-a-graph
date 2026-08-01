# Graph-Based Detection of Architectural Anti-Patterns and Prescriptive Refactoring in Distributed Publish–Subscribe Systems

> **`draft.md` is the authoritative manuscript text.** This outline is a companion map of its
> structure and headline figures, revised against the post-measurement draft; where the two disagree,
> `draft.md` wins. Detection figures come from `results/detection_validation.json` (system layer) and
> `results/detection_validation_app.json` (app layer), produced by
> [reproduce/detection_validation.py](../../../reproduce/detection_validation.py); prescription
> figures from `results/prescribe_all.log`.

* **Target Journal:** Automated Software Engineering (AuSE) — Springer
* **Target Venue:** Special Issue "Intelligent techniques for CI/CD, DevOps, software evolution, technical debt analysis, and refactoring recommendation."
* **Target Topic:** *Technical debt analysis* (the anti-pattern catalog, §4), *refactoring recommendation* (the verified prescriptive operators, §6), and *CI/CD/DevOps* (the quality gate, §7) — three of the special issue's five named areas; no AutoML/NAS/LLM contribution is claimed.

## Abstract

Distributed publish–subscribe middleware (ROS 2, DDS, MQTT) decouples producers and consumers, but the resulting indirect dependency structure obscures how component failures cascade, and this architectural technical debt accumulates invisibly to code-level static analysis (SCA) tools. Unlike object-oriented design, publish–subscribe architectures have no equivalent catalog of named anti-patterns, and existing structural diagnostic frameworks typically operate open-loop, ranking components by criticality without naming the responsible pathology or verifying a remedy. We address both gaps with SaG-Prescribe, a graph-based framework that (1) specifies twenty-one named, severity-tiered publish–subscribe anti-patterns as formal topological signatures with adaptive box-plot thresholds, and (2) compiles the resulting diagnosis into three graph-mutation operators — logical topic splitting, physical anti-affinity reallocation, transport QoS contract hardening — in which every candidate edit is verified independently on its own counterfactual graph and admitted only if its measured cascade-impact reduction exceeds the simulator's own seed noise at every propagation threshold. We report two negative results as principal findings. **Detection is weaker than prior reporting suggested:** mean Spearman ρ = 0.268 against simulated cascade impact — below betweenness (0.295) and degree centrality (0.417) alone — with the catalog implicating a mean 90.4% of components at CRITICAL/HIGH severity, and the deliberately-encoded broker-saturation scenario going entirely undetected. **Verification admits far less than generation proposes:** 29 of 213 candidates (13.6%) survive, all topic splits; no anti-affinity reallocation and no QoS upgrade clears the margin anywhere, contradicting this work's own earlier attribution of its best result to QoS hardening. Five of six scenarios admit nothing; one improves clearly (ΔSRI = +0.0365); one shows three individually verified edits composing into a marginal regression. Detection runs in 0.03–24.6 s across the full scale range, with the detectors themselves under a second in total.

## Keywords

architectural anti-patterns; technical debt analysis; refactoring recommendation; publish-subscribe middleware; CI/CD quality gates; failure cascade simulation; counterfactual verification

---

## Outline

### 1. Introduction

* **1.1 Context and motivation:** Pub-sub decoupling as the backbone of cyber-physical/microservice/IoT systems; decoupling obscures failure propagation and the resulting architectural technical debt.
* **1.2 Two open gaps:** (a) the Architecture-Code Gap and the absence of a named, testable anti-pattern catalog for pub-sub topologies; (b) the open-loop refactoring-recommendation gap — even topology-aware diagnostics rank criticality without naming the fault or verifying its remediation.
* **1.3 Proposed solution:** A detect→prescribe→gate pipeline — twenty-one-pattern catalog feeding a two-level verification engine (per-edit margin test, then whole-policy gate), operationalized as a CI/CD gate whose delta-aware layer is designed but unimplemented.
* **1.4 Contributions:** Five numbered items: (1) the catalog; (2) the prescriptive pipeline with per-edit verification and its disclosed 5-of-21 automation footprint; (3) the CI/CD gate with per-half implementation status; (4) an evaluation whose *principal findings are two negative results*; (5) an account of the measurement itself, including the pooling effect that flips the sign of a system-layer correlation.
* **1.5 Relationship to the Authors' Prior Work:** Positions this submission against the companion Software-as-a-Graph submission (which owns the graph model, RMAV, simulators and the learned predictor) and the prior conference paper whose detection figures are *re-measured rather than restated* here.
* **1.6 Organization.**

### 2. Background and Related Work

* **2.1 Publish–Subscribe Middleware Dependability:** Broker fault tolerance, Kafka, DDS QoS/latency literature.
* **2.2 Anti-Pattern and Code-Smell Catalogs:** OO design (Fowler; Brown et al.; Suryanarayana et al.) and microservices smells (Richardson; Taibi et al.), establishing the specification template this catalog follows while operating on system topology.
* **2.3 Refactoring Recommendation and Architectural Technical Debt:** Code-scope recommenders vs. this paper's topology scope and verify-before-recommend model. **Two `[REF: …]` citation slots remain open** (learning-based and LLM-based recommenders) and must be populated before submission; no references have been invented.
* **2.4 Search-Based Software Engineering and Architecture Optimization:** Open-loop vs. this paper's measured-per-edit verification.
* **2.5 Diagnostic Foundation (SaG):** The heterogeneous graph model and RMAV attribution this paper builds on; what is summarized vs. not repeated.
* **2.6 Structural Criticality Analysis:** Why classical single-metric centrality is expected to degrade on typed pub-sub graphs — a motivation §9.1 then fails to confirm, since degree centrality out-performs the composite on this suite.

### 3. System Model and Code-Quality-Augmented Technical Debt Analysis

* **3.1 Heterogeneous Graph Formulation:** Five node types, six structural edge types.
* **3.2 Derived DEPENDS_ON Projection:** App-to-App, App-to-Broker, App-to-Library, Broker-to-Broker edges; four architectural layers (app, infra, mw, system).
* **3.3 Code Quality Penalty (CQP):** The explicit bridge from SCA metrics to architecture-level risk; feeds Maintainability.
* **3.4 Multi-Dimensional Quality Attribution (RMAV):** Four dimensions; composite Q(v) with raw weights (0.43, 0.24, 0.17, 0.16) and *applied* post-shrinkage weights (0.395, 0.247, 0.193, 0.165) at λ = 0.70. Framed as **stated design judgements checked for consistency, not elicited**, with the unfavourable sensitivity result (equal weights out-perform the calibrated ones) disclosed here rather than deferred.

### 4. A Catalog of Architectural Anti-Patterns for Publish–Subscribe Systems

* **4.1 Anti-Patterns vs. Bad Smells:** Taxonomy and confidence distinction.
* **4.2 Detection Methodology:** The **fifteen Tier-1 structural metrics** (plus CQP as a sixteenth input) grouped by the RMAV dimension each serves — replacing the previous, unsupported "thirteen metrics" claim; why forward centralities are held at Tier 2; rank vs. min-max normalization, **with the measured cost of the rank default (≈0.195 ρ) stated**, and the observation that rank-normalized inputs make the box-plot classifier's critical fraction near-constant across topologies — part of the mechanism behind §9.1's over-flagging.
* **4.3 Catalog Overview:** Twenty-one patterns across three severity tiers and four RMAV dimensions; summary table of pattern, severity, dimension, and detection signal.
* **4.4 Representative Pattern Walkthroughs:** Five patterns spanning tiers, dimensions and detection techniques: SPOF, God Component, **Broker Overload — used as the cautionary case rather than a success**, since its 2×-median rule is structurally unable to fire on the two-broker scenario built to encode broker saturation — Chatty Pair, and QoS Mismatch.
* **4.5 Validation Methodology for Detection:** Protocol, oracle choice and scenario roles only; headline figures deliberately deferred to §9.1 rather than previewed, because the results are mixed. States up front that detection and prescriptive verification share one simulator class, which bounds what any agreement between them can mean.

### 5. Closed-Loop Optimization Objective

* **5.1 Formal Objective and Two-Level Acceptance:** Minimization under an unconstrained generation budget, constrained instead by (1) the **implemented** per-edit criterion `mean ΔI > κ·σ_seed at every threshold`, and (2) the whole-policy gate ΔSRI > 0. Explains why the second is not redundant with the first (§9.4's composition failure). Empty accepted sets are valid, reported outcomes. Budget-constrained search remains future work.

### 6. The SaG-Prescribe Prescriptive Pipeline

* **6.1 Hexagonal Core Abstraction:** `IGraphRepository`, `Neo4jRepository` vs. `MemoryRepository`.
* **6.2 Pipeline Stages 1–7:** Diagnostic foundation, candidate generation, per-edit verification, and the review interface — which is SDK/CLI-reachable and, unlike the diagnostic stages, has **no REST router and no dashboard rendering**.
* **6.3 Three Refactoring Operators, and Their Coverage:** Topic splitting, anti-affinity reallocation, QoS hardening, each a typed graph-mutation rule. Discloses the two trigger channels (generic criticality tier; substring matching on detected-problem names) and that **only 5 of 21 patterns** reach an operator — with `QOS_MISMATCH` notably *not* wired to QoS hardening. Separates the principled part of that boundary from the implementation artifact (`DetectedProblem` carries no `pattern_id`).
* **6.4 Closed-Loop Verification Procedure:** Six steps — baseline → per-edit filter → mutate accepted subset → sandbox → re-run → whole-policy gate. States the cost consequence (one exhaustive sweep per edit × threshold × seed) that dominates §9.5, and scopes the independence guarantee as *view* independence, not independence of data source.

### 7. DevOps Integration and CI/CD Gating

*(This section addresses the special issue's CI/CD/DevOps bullet, and separates design from implementation status throughout.)*

* **7.1 Automated Code Review Architecture:** The gate path and what it computes. **Status:** the functioning path runs through `cli/predict_graph.py`; the dedicated `cli/detect_antipatterns.py` entry point passes the wrong result object and reports no findings; neither is wired into CI.
* **7.2 Regression Semantics — Absolute Today, Delta by Design:** Why absolute gating is unsustainable, made concrete by §9.1's 90% implication rate. **Status:** merge-base diffing and the waiver register are specified but **not implemented**; no empirical delta-gating claim is made.
* **7.3 Exit-Code Protocol:** 0 / 1 / 2, implemented on absolute rather than delta semantics.

### 8. Experimental Design

* **8.1 Research Questions:** RQ1 (detection efficacy and precision); RQ2 (prescriptive efficacy); RQ3 (operator contributions); **RQ4, reframed** (what per-edit verification admits, in which regimes, and whether an accepted set composes); RQ5 (computational overhead and CI/CD feasibility).
* **8.2 Scenario Suites:** Eight scenarios (S01–S08) for detection; a **six-scenario subset** for prescription — S08 excluded on relevance, **S07 excluded on measured cost (≈8.7 h serial)**, with the two reasons kept separate because the second removes the largest topology from the very result whose scale-invariance is in question.
* **8.3 Experimental Protocol:** Detection scored against I_comp at threshold 0.2, seed 42, reported at both app and system layer scopes; prescription at κ = 1.0, Θ = {0.1, 0.2, 0.5}, S = {42, 123, 456}, paired per (θ, s). Explicitly retracts the earlier σ_seed = 0 justification.
* **8.4 Metrics:** Detection — Spearman ρ pooled *and* stratified, precision/recall/F1, Top-K overlap, plus betweenness and degree baselines. **States that the shared critical-set rule makes precision and recall equal by construction for the ranking predictors, and that the catalog predictor is deliberately not thresholded that way.** Prescription — SRI/ΔSRI, acceptance rate, per-operator candidates-vs-admitted.

### 9. Results

* **9.1 Detection Efficacy and Precision (RQ1) — negative:** Table of eight scenarios at the app layer. Mean ρ = 0.268 (range 0.097–0.449), **below betweenness (0.295) and degree (0.417)**; mean F1 = 0.423, Top-5 = 0.500. Catalog behaves as a sensitive, unspecific screen: recall 0.920 against precision 0.260, implicating a mean 90.4% of components — 70% on the sparse topology chosen as the precision stress test. The "improves at scale" claim does not survive (0.226 at ≥150 components vs 0.282 below). **Pooling trap documented:** system-layer pooled ρ = −0.085 lies *outside* the per-type range (Application 0.306, Node 0.319, Broker 0.047) — a Simpson's-paradox effect. The deliberately-encoded broker-saturation scenario is not detected at all. SPOF-specific validation is **undefined rather than zero** (AP_c_directed is zero everywhere; the framework's IA > 0.50 threshold is off this oracle's scale, which peaks at 0.39).
* **9.2 Prescriptive Efficacy (RQ2):** Six-scenario table; 29/213 (13.6%) candidates admitted; four scenarios at ΔSRI = +0.0000 because *nothing was applied*; IoT Smart City +0.0365; Microservices −0.0011. **No significance test reported**, with the reason given. Enterprise exclusion itemized (119 × 3 × 3 sweeps at 29.1 s ≈ 8.7 h) and flagged as a reporting decision the script does not enforce.
* **9.3 Operator Contributions (RQ3):** Candidates-generated vs. edits-admitted. **All 29 admitted edits are topic splits; zero reallocations and zero QoS upgrades survive anywhere** — directly contradicting the earlier attribution of IoT Smart City's gain to 51 QoS upgrades. Mechanism explained via `CONNECTS_TO` duplication in reallocation.
* **9.4 What Per-Edit Verification Admits (RQ4):** Rejections carry the binding threshold and are readable, not silent. Composition failure at S06 (three verified edits, ΔSRI = −0.0011) bounds but does not eliminate the failure mode — two orders of magnitude smaller than the −31.67% the unfiltered design produced. Honest scope: fan-out decomposition where a fan-out bottleneck actually exists.
* **9.5 Computational Overhead and CI/CD Feasibility (RQ5):** Measured gate times 0.03–24.6 s (app layer) / 0.16–33.0 s (system layer). **The twenty active detectors total ≈0.2 s across all eight scenarios** — the entire budget is the analysis stage, so optimizing detectors buys nothing. **`DEEP_PIPELINE` is excluded as a defect**: it enumerates every simple source-to-sink path, producing 247,761 findings on a 29-component fixture and failing to terminate on a 50-application topology. The earlier generate–verify loop runtimes (4.7 s–649.6 s) are **withdrawn** as pre-filter measurements and replaced by the per-edit cost model.

### 10. Discussion, Threats to Validity, and Conclusion

* **10.1 What Naming and Verifying Buys, and What It Does Not:** Verification earns its cost — it corrected the paper's causal account, not merely its confidence. Naming does not yet deliver, because a vocabulary implicating nine components in ten conveys little; specifications sound, thresholds uncalibrated. RMAV scoped to attribution, not ranking accuracy.
* **10.2 Positioning in CI/CD and Technical-Debt Workflows:** Prescriptions advisory; gate not yet deployable at a 90% implication rate; the catalog as an architecture-review checklist, a use unaffected by the calibration problem since a human supplies the prioritization.
* **10.3 Construct Validity:** Shared-simulator dependence; the two oracles agree at mean ρ = 0.4046 (worst scenario 0.06), so cross-oracle transfer is not licensed; operator-to-pattern linkage implemented by substring matching over display strings.
* **10.4 Internal Validity:** Exhaustive generation, no optimality claim; **verification admits singletons, not subsets**; **κ = 1.0 is asserted, not derived**; independence guarantee scoped as view independence; stratified reporting mandatory given §9.1's sign-flipping pooled correlation.
* **10.5 External Validity:** One generator, synthetic topologies; catalog scoped to pub-sub; acceptance rate established over 98–326 components only; detection figures characterize a twenty-detector catalog.
* **10.6 Conclusion Validity:** No significance test available; single-run detection at canonical settings, read against a propagation-threshold sensitivity spanning ≈0.2 ρ; the earlier σ_seed = 0 justification explicitly retracted.
* **10.7 Engineering Trade-offs:** Verification costs ≈9× per candidate and was worth it; detection is nearly free relative to its own analysis stage — hence the split deployment model (per-commit detection, batch prescription).

### 11. Conclusion and Future Work

* **11.1 Conclusions:** The catalog's specifications stand; its thresholds do not. Verification changed the prescriptive result rather than confirming it. CI/CD timing is the least disturbed contribution.
* **11.2 Future Work,** ordered by how much each would change the paper's claims: (1) recalibrate detection thresholds; (2) bound `DEEP_PIPELINE`; (3) implement and evaluate delta-aware gating; (4) verify subsets, not singletons; (5) derive κ rather than assert it; (6) parallelise verification and close the enterprise gap; (7) expand and ablate the operator set; (8) extend the catalog; (9) real-system replication; (10) LLM-assisted PR generation, one sentence, not claimed as a contribution.

### 12. References

* Twenty-two entries covering pub-sub dependability, SBSE and architecture optimization, centrality and network reliability, and the OO/microservices anti-pattern catalogs. **Two `[REF: …]` slots in §2.3 remain unpopulated**; AuSE reviewers will expect roughly 30–45 references, so the list needs expansion before submission.
