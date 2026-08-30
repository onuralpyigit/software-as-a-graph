# SaG-Prescribe: Unifying Diagnostic Pathways with Counterfactual Refactoring for Automated Code Review and Software Quality Evaluation in Distributed Publish–Subscribe Systems

* **Target Journal:** Automated Software Engineering (AuSE) — Springer
* **Target Venue:** Special Issue on *Intelligent Techniques for Automated Code Review and Software Quality Evaluation*
* **Core Topic Mapping:**
  - *Software Quality Evaluation & Technical Debt Analysis:* Deterministic, explainable **Diagnostic Pathway** bridging file-level SCA metrics (CQP) with multi-layer topology ($G_{\text{analysis}}$) under ISO/IEC 25010/25019 RM attribution (§3); 19-pattern anti-pattern & smell catalog (§4).
  - *Automated Code Review Assistance & CI/CD Quality Gating:* Pre-merge review bot with 0.01–20.98s execution, 3-tier exit codes (0/1/2), and delta-aware merge-base regression semantics (§7).
  - *Refactoring Recommendation & Counterfactual Verification:* Compiling diagnostic findings into 3 mutation operators verified on sandboxed counterfactual graphs against cascade noise (§5–§6).

---

## Abstract

Highlights the Diagnostic Pathway as the deterministic, explainable bridge between code-level Static Code Analysis (SCA) and distributed architectural topology. Formalizes 19 publish–subscribe anti-patterns, links them to 3 verified refactoring operators, and reports comprehensive empirical findings: detection correlation (mean $\rho = 0.485$), Simpson's paradox pooling effect, $71.0\%$ prescriptive verification survival rate (1128/1589 edits admitted), statistically significant risk reduction (Wilcoxon $p = 0.0156$), and sub-second CI/CD gating feasibility.

## Keywords

Automated code review; software quality evaluation; diagnostic pathway; architectural anti-patterns; bad smells; technical debt analysis; refactoring recommendation; publish–subscribe middleware; CI/CD quality gates; counterfactual verification.

---

## Section-by-Section Outline

### 1. Introduction
* **1.1 Context and Motivation:** Pub-sub middleware (ROS 2, DDS, MQTT) in modern distributed systems; how asynchronous decoupling obscures failure propagation and architectural technical debt.
* **1.2 Two Open Gaps:**
  - *The Architecture-Code Gap:* Clean unit-level code vs. fragile topological dependencies invisible to file-scoped SCA.
  - *The Unnamed-Pathology and Open-Loop Review Gap:* Lack of pub-sub anti-pattern catalogs; open-loop numeric ranking without actionable remediation.
* **1.3 Proposed Solution (SaG-Prescribe):**
  - *The Diagnostic Pathway:* Deterministic, explainable quality attribution and smell detection.
  - *The Prescriptive Pathway:* Counterfactual verification of candidate refactorings.
* **1.4 Contributions:** Five numbered contributions (Diagnostic Model, 19-Pattern Catalog, Prescriptive Refactoring Pipeline, CI/CD Gate, Empirical Validation).
* **1.5 Relationship to Prior Work:** Delineating contributions relative to companion work [1] and prior conference publications.
* **1.6 Organization.**

### 2. Background and Related Work
* **2.1 Publish–Subscribe Middleware Dependability:** Protocol verification, broker fault tolerance, DDS/ROS 2 QoS policies.
* **2.2 Automated Code Review and Static System Analysis:** Moving from AST/file-scoped SCA to global Static System Analysis (SSA).
* **2.3 Software Quality Models and Explainable Diagnostic Pathways:** ISO/IEC 25010 and ISO/IEC 25019 Quality-in-Use models; deterministic explainability.
* **2.4 Anti-Pattern and Bad Smell Catalogs:** OO smells (Fowler, Brown, Suryanarayana) and microservices smells (Taibi, Richardson) vs. pub-sub topological smells.
* **2.5 Refactoring Recommendation and Technical Debt Management:** Metric-based, SBSE, learning-based, LLM-assisted refactoring vs. verify-before-recommend.
* **2.6 Structural Criticality and Graph Centrality:** Centrality baselines and why explainable diagnostic attribution is essential.

### 3. System Model and The Multi-Level Diagnostic Pathway
* **3.1 Heterogeneous Graph Formulation:** Typed multigraph $G = (V, E, \tau_V, \tau_E, w_E, w_V)$ (5 vertex types, 6 edge types).
* **3.2 Derived Dependencies (`DEPENDS_ON`):** 4 architectural layer projections (app, infra, mw, system).
* **3.3 Code Quality Penalty (CQP):** Bridging file-level SCA metrics (LOC, WMC, LCOM, SQALE) to architecture.
* **3.4 Hierarchical RM Quality Attribution:**
  - Formal Definitions D1–D4.
  - Fault Tolerance ($FT$), Availability ($A$), Reliability blend ($R = 0.36 FT + 0.64 A$), Maintainability ($M$).
  - ISO/IEC 25019 Stakeholder Harm Mapping Matrix ($\mathbf{M}_{\text{RM} \to \text{QiU}}$).
  - Composite Criticality $Q(v) = 0.80 R(v) + 0.20 M(v)$ and adaptive box-plot tiers.

### 4. A Catalog of Architectural Anti-Patterns for Publish–Subscribe Systems
* **4.1 Anti-Patterns vs. Bad Smells in Automated Review:** Taxonomy, confidence tiers, and topological signatures.
* **4.2 Detection Methodology:** 16-element metric vector with adaptive box-plot thresholds ($Q_3 + 1.5\,\mathrm{IQR}$).
* **4.3 Catalog Overview:** Table 4.1 detailing 19 patterns across Reliability, Availability, Maintainability, and Architecture.
* **4.4 Representative Pattern Walkthroughs:** SPOF, God Component, Broker Overload (cautionary case), Chatty Pair, QoS Mismatch.
* **4.5 Validation Methodology for Detection:** Cascade simulation ground truth $I_{\text{comp}}$.

### 5. Closed-Loop Optimization Objective
* **5.1 Mathematical Objective:** Minimizing cascade impact under budget $\mathcal{B}$.
* **5.2 Two-Level Acceptance Filtering:**
  - Level 1: Per-edit counterfactual margin ($\overline{\Delta I}_\theta > \kappa \sigma_{\text{seed}}$ $\forall \theta$).
  - Level 2: Whole-policy gate ($\Delta\mathrm{SRI} > 0$).

### 6. The SaG-Prescribe Prescriptive Pipeline
* **6.1 Hexagonal Core Abstraction:** `IGraphRepository`, `Neo4jRepository`, `MemoryRepository`.
* **6.2 Seven Pipeline Stages:** Ingest $\to$ Project $\to$ Attribute $\to$ Detect $\to$ Compile $\to$ Verify $\to$ Review.
* **6.3 Refactoring Mutation Operators:** Topic Splitting, Anti-Affinity Reallocation, QoS Hardening; automation footprint disclosure (5 of 19 directly automated).
* **6.4 Closed-Loop Verification Procedure:** Sandboxed memory mutation and view independence guarantee.

### 7. DevOps Integration and CI/CD Quality Gating
* **7.1 Automated Code Review Bot Architecture:** Sub-second pre-merge architectural review bot.
* **7.2 Regression Semantics:** Absolute evaluation vs. delta-aware merge-base diffing with waiver registers.
* **7.3 Three-Tier Exit-Code Protocol:** Exit codes 0 (Pass), 1 (Warn), 2 (Block).

### 8. Experimental Design
* **8.1 Research Questions:** RQ1 (Detection & Precision), RQ2 (Prescriptive Efficacy), RQ3 (Operator Survival), RQ4 (Verification Yield & Composition), RQ5 (Computational Overhead).
* **8.2 Benchmark Scenarios:** Table 8.1 (8 scenarios, S01–S08, 12 to 520 components).
* **8.3 Experimental Protocol & Oracles:** Ground truth $I_{\text{comp}}$ vs. learned oracle $I^*$.
* **8.4 Evaluation Metrics:** $\rho$, $F_1$, Top-5, Catalog P/R, $\mathrm{SRI}$, $\Delta\mathrm{SRI}$, Operator survival.

### 9. Empirical Results
* **9.1 Diagnostic Detection Efficacy and Precision (RQ1):** Table 9.1; mean $\rho = 0.485$, degree baseline ($0.519$), catalog screening (recall $0.942$, precision $0.253$, implicated $94.8\%$), Simpson's paradox in node pooling ($\rho=0.028$ pooled vs. $0.503$ app).
* **9.2 Prescriptive Refactoring Efficacy (RQ2):** Table 9.2; 1128/1589 edits admitted ($71.0\%$), Wilcoxon $W=0, p=0.0156 (n=7)$.
* **9.3 Operator Survival and Contributions (RQ3):** Table 9.3; Reallocation $77.7\%$, QoS Hardening $58.0\%$, Topic Splitting $49.7\%$.
* **9.4 Verification Yield and Compositionality (RQ4):** Informative threshold rejections, sub-additive composition dynamics in S03 and S06.
* **9.5 Computational Overhead and CI/CD Feasibility (RQ5):** Gate latency 0.01–20.98s, `DEEP_PIPELINE` bounding analysis, 2h 47m full verification sweep.

### 10. Discussion and Threats to Validity
* **10.1 Diagnostic Explainability vs. Threshold Calibration:** Role-specific attribution vs. over-flagging fences.
* **10.2 The Verify-Before-Recommend Discipline:** Preventing harmful mutations in CI/CD.
* **10.3 Construct Validity:** Simulator grounding and dependability patterns.
* **10.4 Internal Validity:** View independence and memory sandboxing.
* **10.5 External Validity:** Synthetic benchmarks and pub-sub scope.
* **10.6 Conclusion Validity:** Statistical significance and threshold sensitivity.
* **10.7 Engineering Trade-offs:** Split deployment (real-time gate vs. batch verification).

### 11. Conclusion and Future Work
* **11.1 Conclusion:** Unifying the Diagnostic Pathway with counterfactual refactoring for automated review.
* **11.2 Future Work:** 10 prioritized research and engineering directions.

### 12. References
* 34 complete, verified bibliographic entries.
