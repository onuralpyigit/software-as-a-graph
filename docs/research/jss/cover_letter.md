# Submission Cover Letter and Academic Disclosures

**To:** Editorial Board  
**Journal:** Journal of Systems and Software (JSS) — Elsevier  
**Title:** *Software-as-a-Graph: A Static System Analysis Framework for Pre-Deployment Quality Gating and Failure Simulation of Publish-Subscribe Middleware*  

Dear Editor-in-Chief and Editorial Board Members,

We submit our manuscript, titled **"Software-as-a-Graph: A Static System Analysis Framework for Pre-Deployment Quality Gating and Failure Simulation of Publish-Subscribe Middleware,"** for consideration as a first-class research article in the Journal of Systems and Software (JSS).

In accordance with JSS policies on prior publications and parallel submissions, we declare that this manuscript extends and coordinates with two conference-level works by the same authors:

1. **Middleware 2026 Submission:** A conference paper detailing the core multi-layer graph representation and projection semantics (`DEPENDS_ON` projection patterns) for publish-subscribe distributed systems.
2. **ASE 2026 Companion Submission:** A conference manuscript under review that focuses exclusively on the closed-loop prescriptive optimization engine (**SaG-Prescribe**) and its associated mutation operators.

We provide a detailed boundary map below to establish that the JSS manuscript constitutes an *unambiguous superset* of the Middleware paper and remains distinct from the companion ASE paper, containing over 50% new technical content.

---

### 1. Relationship to Middleware 2026 Submission (Prior Conference Baseline)
The Middleware conference paper introduced the foundational *Software-as-a-Graph* (SaG) graph projection schema, project patterns, and structural projecting filters. The JSS manuscript builds upon this base representation space and introduces substantial, first-class extensions:
* **Multi-Dimensional Quality Attribution (RMAV):** We introduce the formal formulation of the Reliability, Maintainability, Availability, and Vulnerability (RMAV) framework. This includes the Analytic Hierarchy Process (AHP) matrix calibration, the shrinkage toward a uniform prior, and the rank-normalized mathematical metrics.
* **Code-Level SCA Ingestion (CQP):** We introduce the Code Quality Penalty (CQP) metric, which ingests raw code-level static analysis variables (e.g., LOC, cyclomatic complexity, LCOM) as vertex properties and propagates them through the global inter-component graph topology, bridging the "Architecture-Code Gap."
* **Continuous CI/CD Quality Gating:** We detail the continuous pipeline-blocking gate architecture, including the exit-code protocol and the database-free `MemoryRepository` designed to run in-memory within seconds.

---

### 2. Boundary Mapping to ASE 2026 Companion Submission
The under-review ASE 2026 manuscript is a *companion conference paper* designed with a clear, non-overlapping boundary:
* **JSS Scope (Static Diagnostic & Gating Framework):** Focuses on the pre-deployment static diagnostic system—defining the graph model, the code-to-architecture quality attribution (RMAV/CQP), failure-impact simulation (interpretable vs. learned HGT predictions), and continuous gating pipelines.
* **ASE Scope (Closed-Loop Prescriptive Optimization Engine):** Focuses exclusively on the closed-loop prescriptive optimization engine (**SaG-Prescribe**), the three graph mutation operators (logical topic splitting, host anti-affinity container reallocations, and transport QoS contract hardening), and the visual dashboard (**SMART**). 

The ASE companion paper cites the JSS manuscript as its foundational diagnostic and simulation baseline, focusing its own evaluation entirely on prescriptive mutation efficiency and computational scalability.

We hope this disclosure assists the Editorial Board in evaluating the manuscript. We are happy to provide copies of the conference manuscripts upon request to aid the review process.

Sincerely,  
*The Authors*
