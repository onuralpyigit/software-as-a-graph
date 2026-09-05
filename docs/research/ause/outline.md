# SaG-Prescribe: Unifying Diagnostic Pathways with Counterfactual Refactoring for Automated Code Review and Software Quality Evaluation in Distributed Publish–Subscribe Systems

* **Target Journal:** Automated Software Engineering (AuSE) — Springer
* **Target Venue:** Special Issue on *Intelligent Techniques for Automated Code Review and Software Quality Evaluation*
* **Core Topic Mapping:**
  - *Software Quality Evaluation & Technical Debt Analysis:* Deterministic, explainable **Diagnostic Pathway** bridging file-level SCA metrics (CQP) with multi-layer topology ($G_{\text{analysis}}$) under ISO/IEC 25010/25019 RM attribution (§3); 19-pattern anti-pattern & smell catalog with scope/coverage disclosure (§4); a calibration and validity analysis (agreement, ranking saturation, substrate adequacy) that a purely accuracy-framed evaluation would omit (§9.1–§9.3).
  - *Automated Code Review Assistance & CI/CD Quality Gating:* Pre-merge review bot with 0.01–20.98s execution, 3-tier exit codes (0/1/2), delta-aware merge-base regression semantics (§7), and the measured absolute-gating outcome (Exit Code 2 on 8/8 scenarios) that motivates it.
  - *Refactoring Recommendation & Counterfactual Verification:* Compiling diagnostic findings into 3 mutation operators verified on sandboxed counterfactual graphs against cascade noise (§5–§6).

---

## Abstract

Leads with the Diagnostic Pathway's contribution as deterministic, role-attributed *explanation*, not superior ranking accuracy: three structural predictors (RM composite, betweenness, degree) saturate around comparable ranking quality (NDCG@10 ≈ 0.82–0.86, Cohen's κ ≈ 0.41–0.48). Reports, with full candor, a mechanistically explained re-measurement of the authors' own prior published criticality-ranking claim (ρ = 0.94 → 0.485) under an independent cascade oracle; a catalog calibration finding (κ ≈ 0 despite recall 0.942, because 98.4% of active findings are not component-scoped); and an Availability-dimension substrate-adequacy finding (SPOF F₁ = 0.0 in 8/8 scenarios at the application layer). Formalizes 19 publish–subscribe anti-patterns, links them to 3 verified refactoring operators, and reports prescriptive empirical findings: $71.0\%$ verification survival rate (1128/1589 edits admitted), statistically significant risk reduction (Wilcoxon $p = 0.0156$), and sub-second CI/CD gating feasibility.

## Keywords

Automated code review; software quality evaluation; diagnostic pathway; architectural anti-patterns; bad smells; technical debt analysis; refactoring recommendation; publish–subscribe middleware; CI/CD quality gates; counterfactual verification.

---

## Section-by-Section Outline

### 1. Introduction
* **1.1 Context and Motivation:** Pub-sub middleware (ROS 2, DDS, MQTT) in modern distributed systems; how asynchronous decoupling obscures failure propagation and architectural technical debt.
* **1.2 Two Open Gaps:**
  - *The Architecture-Code Gap:* Clean unit-level code vs. fragile topological dependencies invisible to file-scoped SCA.
  - *The Unnamed-Pathology and Open-Loop Review Gap:* Lack of pub-sub anti-pattern catalogs; open-loop numeric ranking without actionable remediation.
* **1.3 Proposed Solution (SaG-Prescribe):** Two-pathway architecture diagram (Figure 1). Diagnostic Pathway's value stated explicitly as attribution, not ranking accuracy.
* **1.4 Contributions:** Six numbered contributions (Diagnostic Model; 19-Pattern Catalog with scope/coverage; Calibration & Validity Analysis [new]; Prescriptive Refactoring Pipeline; CI/CD Gate; Empirical Validation).
* **1.5 Relationship to Prior Work:** Two-axis disclosure — (a) RASSE 2025 [35] as the prior diagnostic-pathway conference paper, with the ρ = 0.94 → 0.485 delta stated and mechanistically previewed; (b) companion JSS work [1] (schema, simulation kernels, GNN).
* **1.6 Organization.**

### 2. Background and Related Work
* **2.1 Publish–Subscribe Middleware Dependability:** Protocol verification, broker fault tolerance, DDS/ROS 2 QoS policies.
* **2.2 Automated Code Review and Static System Analysis:** Moving from AST/file-scoped SCA to global Static System Analysis (SSA).
* **2.3 Software Quality Models and Explainable Diagnostic Pathways:** Score decomposition vs. post-hoc attribution, distinguished explicitly; ISO/IEC 25010/25019 grounding.
* **2.4 Anti-Pattern and Bad Smell Catalogs:** OO smells (Fowler, Brown, Suryanarayana) and microservices smells (Taibi, Richardson) vs. pub-sub topological smells.
* **2.5 Refactoring Recommendation and Technical Debt Management:** Metric-based, SBSE, learning-based, LLM-assisted refactoring vs. verify-before-recommend.
* **2.6 Structural Criticality and Graph Centrality:** Centrality baselines; RASSE 2025 [35] positioned as closest prior work; empirical re-measurement of its claim.

### 3. System Model and The Multi-Level Diagnostic Pathway
* **3.1 Heterogeneous Graph Formulation:** Typed multigraph $G = (V, E, \tau_V, \tau_E, w_E, w_V)$ (5 vertex types, 6 edge types).
* **3.2 Derived Dependencies (`DEPENDS_ON`):** 4 architectural layer projections (app, infra, mw, system).
* **3.3 Code Quality Penalty (CQP):** Corrected input list (LOC, WMC-proxy complexity, LCOM, coupling-derived instability); per-type min-max normalization with zero-variance caveat; explicit disclosure that SQALE debt ratio is ingested but never scored (Table 3.0).
* **3.4 Hierarchical RM Quality Attribution:**
  - Formal Definitions D1–D4.
  - Fault Tolerance ($FT$), Availability ($A$), Reliability blend ($R = 0.36 FT + 0.64 A$), Maintainability ($M$).
  - ISO/IEC 25019 Stakeholder Harm Mapping Matrix ($\mathbf{M}_{\text{RM} \to \text{QiU}}$).
  - Composite Criticality $Q(v) = 0.80 R(v) + 0.20 M(v)$ and adaptive box-plot tiers.
* **3.5 Substrate Adequacy [new]:** States, as a design-time property, that $A(v)$'s discrete cut-structure terms require a projection where they occur; forward-references the app-layer degeneracy measured in §9.3.

### 4. A Catalog of Architectural Anti-Patterns for Publish–Subscribe Systems
* **4.1 Anti-Patterns vs. Bad Smells in Automated Review:** Taxonomy, confidence tiers, and topological signatures.
* **4.2 Detection Methodology:** Fifteen scoring metrics (14 structural + CQP), reconciled against the metric registry; fixed $k=1.5$ IQR fence disclosed as not swept in this work.
* **4.3 Catalog Overview:** Table 4.1 detailing 19 patterns across Reliability, Availability, Maintainability, and Architecture, now with **Scope** (Component/Edge/System/Pair) and **Trig.** (fires on corpus) columns.
* **4.4 Representative Pattern Walkthroughs:** SPOF, God Component, Broker Overload (cautionary case), Chatty Pair, QoS Mismatch.
* **4.5 Validation Methodology for Detection:** Cascade simulation ground truth $I_{\text{comp}}$.
* **4.6 Catalog Coverage on the Benchmark Corpus [new]:** Only 8 of 18 active patterns trigger on the corpus; of 6329 findings, 98.4% are not component-scoped — the mechanism behind §9.2's calibration result.

### 5. Closed-Loop Optimization Objective
* **5.1 Mathematical Objective:** Minimizing cascade impact under budget $\mathcal{B}$; opening sentence makes explicit that candidate compilation consumes §3–§4's diagnostic output.
* **5.2 Two-Level Acceptance Filtering:**
  - Level 1: Per-edit counterfactual margin ($\overline{\Delta I}_\theta > \kappa \sigma_{\text{seed}}$ $\forall \theta$).
  - Level 2: Whole-policy gate ($\Delta\mathrm{SRI} > 0$).

### 6. The SaG-Prescribe Prescriptive Pipeline
* **6.1 Hexagonal Core Abstraction:** `IGraphRepository`, `Neo4jRepository`, `MemoryRepository`.
* **6.2 Seven Pipeline Stages:** Ingest $\to$ Project $\to$ Attribute $\to$ Detect $\to$ Compile $\to$ Verify $\to$ Review.
* **6.3 Refactoring Mutation Operators:** Topic Splitting, Anti-Affinity Reallocation, QoS Hardening; automation footprint (5 of 19) cross-referenced against §4.6's corpus-coverage finding and the underlying rule-trigger mechanism.
* **6.4 Closed-Loop Verification Procedure:** Sandboxed memory mutation and view independence guarantee.

### 7. DevOps Integration and CI/CD Quality Gating
* **7.1 Automated Code Review Bot Architecture:** Sub-second pre-merge architectural review bot.
* **7.2 Regression Semantics:** Absolute evaluation vs. delta-aware merge-base diffing with waiver registers; measured absolute-gating outcome (Exit Code 2 on 8/8 scenarios) stated as the mechanical consequence of §9.1/§9.2's implication rate.
* **7.3 Three-Tier Exit-Code Protocol:** Exit codes 0 (Pass), 1 (Warn), 2 (Block).

### 8. Experimental Design
* **8.1 Research Questions:** RQ1 (Diagnostic Ranking Efficacy), RQ2 (Catalog Screening & Calibration), RQ3 (Attribution Validity & Substrate Adequacy), RQ4 (Prescriptive Efficacy & Operator Survival), RQ5 (Verification Yield, Compositionality & CI/CD Feasibility).
* **8.2 Benchmark Scenarios:** Table 8.1 (8 scenarios, S01–S08, 6 domains, 29 to 520 vertices, 12–300 applications; S08 |E| corrected to 101).
* **8.3 Experimental Protocol & Metrics:** Adds Cohen's κ, NDCG@10, Top-$k$; discloses the equal-sized-set precision=recall=F₁ artifact; states the two-oracle distinction (this paper's $I_{\text{comp}}$ vs. RASSE 2025's Reachability Loss) that §9.3 reconciles.

### 9. Empirical Results
* **9.1 Diagnostic Ranking Efficacy (RQ1):** Table 9.1 + new Table 9.1b (ρ/κ/NDCG@10/Top-10 for all three predictors — no predictor dominates); by-scale split (large 0.599 vs. small 0.447); catalog screening summary.
* **9.2 Catalog Screening and Calibration (RQ2) [new subsection]:** Cohen's κ ≈ 0 at both layers despite high recall; mechanism traced to §4.6's component-vs-non-component finding split.
* **9.3 Attribution Validity and Substrate Adequacy (RQ3) [new subsection]:** SPOF F₁ = 0.0 in 8/8; RASSE 2025 reconciliation (oracle substrate, sample size, composite design); Simpson's paradox pooling result (moved here from old §9.1).
* **9.4 Prescriptive Refactoring Efficacy and Operator Survival (RQ4):** Tables 9.2–9.3 (merged old RQ2+RQ3); Reallocation $77.7\%$ tied to its SPOF/RM-critical trigger precondition.
* **9.5 Verification Yield, Compositionality & CI/CD Feasibility (RQ5):** Merged old RQ4+RQ5; gate latency 0.01–20.98s, `DEEP_PIPELINE` bounding, 2h 47m full verification sweep.

### 10. Discussion and Threats to Validity
* **10.1 Diagnostic Explainability vs. Threshold Calibration:** Rebuilt around the §9.2 calibration result — what $k$ recalibration buys vs. what only delta-aware gating buys.
* **10.2 The Verify-Before-Recommend Discipline:** Preventing harmful mutations in CI/CD; ties the discipline to why the reallocation result is trustworthy, not just plausible.
* **10.3 Construct Validity: Substrate Adequacy [new]:** Promotes the $A(v)$ degeneracy to a first-class, generalizable threat for any RM-style composite on a typed graph.
* **10.4 Comparison with Prior Self-Reported Results [new]:** RASSE 2025 reconciliation in discussion register; general lesson about connectivity-derived oracles.
* **10.5 Further Threats to Validity:** Construct, Internal, External, and Conclusion Validity (the last one new — Wilcoxon $n=7$ significance-ceiling caveat).

### 11. Conclusion and Future Work
* **11.1 Conclusion:** Restates the diagnostic contribution as attribution + calibration evidence, not ranking accuracy.
* **11.2 Future Work:** 11 prioritized directions — threshold-$k$ sweep and layer-aware Availability scoring promoted to items 1–2; RASSE 2025 replication added to the industrial-telemetry item.

### 12. References
* 35 complete, verified bibliographic entries (adds RASSE 2025 [35]).
