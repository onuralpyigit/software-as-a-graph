# Software-as-a-Graph (SaG) — SoSyM Tool Paper Submission Reading Map and Outline

* **Target Journal:** Software and Systems Modeling (SoSyM) — Springer
* **Target Venue / Collection:** **Tool Paper Call**
* **Core Topic Mapping:**
  - *Domain Modeling of Publish–Subscribe Systems:* Formal typed multigraph metamodel ($G = (V, E, \tau_V, \tau_E, w_V, w_E)$) capturing applications, libraries, topics, brokers, and compute hosts across 4 layer projections governed by 7 dependency derivation rules (§3).
  - *Code-to-Architecture Quality Attribution:* Synthesizing static code metrics via the continuous Code Quality Penalty (CQP) into an ISO/IEC 25010/25019 Reliability–Maintainability (RM) model with stakeholder harm projection (§3).
  - *Model-Based Anti-Pattern Detection:* Formal catalog of 19 publish–subscribe anti-patterns evaluated with adaptive box-plot fences ($Q_3 + k\,\mathrm{IQR}$) with explicit finding-scope taxonomy (§4).
  - *Tool Architecture & Dual-Repository Pattern:* Hexagonal ports-and-adapters architecture featuring persistent Neo4j storage for exploration and high-performance in-memory sandboxing (`MemoryRepository`) for rapid counterfactual verification (§5).
  - *Closed-Loop Counterfactual Refactoring:* 3 graph-mutation operators (topic splitting, anti-affinity reallocation, QoS contract hardening) verified against a discrete-event cascade simulator under seed-noise rejection margins (§5).
  - *Multi-Modal Tool Usability & CI/CD Gating:* Interactive web dashboard (Next.js/React & HTML), Python SDK, modular CLI suite, REST API, and CI/CD quality gate with 3-tier exit codes (§6).
  - *Empirical Validation in Realistic Scenarios:* Evaluated over 8 benchmark scenarios (29–520 vertices) and 3 transcribed real-world open-source systems (Autoware.Universe, Train-Ticket, Cloud IoT Mesh) (§7).
  - *Community Importance & Lessons Learnt:* Dedicated section addressing why the tool matters for MDE and MBSE, plus 5 transferable lessons learnt across designing, building, testing, evaluating, and disseminating (§8).

---

## Abstract Summary

Presents **Software-as-a-Graph (SaG)**, an extensible, open-source model-driven software and systems engineering framework for distributed publish–subscribe architectures (ROS 2, DDS, MQTT). Formalizes a multi-layer multigraph metamodel, bridges code metrics to architecture via CQP, and computes deterministic ISO/IEC 25010/25019 RM attributions. Drives a 19-pattern anti-pattern catalog and compiles findings into 3 refactoring mutation operators verified in-memory against a discrete-event cascade simulator ("verify-before-transform"). Evaluated across 8 benchmark scenarios and 3 transcribed real-world systems, reporting predictor saturation (NDCG@10 $\approx 0.82$--$0.86$, showing attribution is the true value over ranking), real-world catalog discrimination ($\kappa = +0.296$), $71.0\%$ refactoring admission, and 5 key lessons learnt.

---

## Section-by-Section Outline

### 1. Introduction
* **1.1 Context and Domain Motivation:** Ubiquity of publish–subscribe middleware in cyber-physical systems, autonomous robotics, and IoT; how asynchronous decoupling obscures topological dependencies and failure cascades.
* **1.2 Open Challenges in Architectural Modeling:** (1) The Architecture-Implementation Gap in MDE; (2) Lack of pub-sub anti-pattern catalogs; (3) Open-loop refactoring vs. verified transformation.
* **1.3 The SaG Framework:** Dual pathways (Diagnostic Modeling Pathway and Prescriptive Refactoring Pathway) within a unified hexagonal architecture.
* **1.4 Target Audience and Tool Use Cases:** Systems architects, reliability/SRE engineers, MDE researchers, DevOps teams.
* **1.5 Contributions to the Modeling Community:** Aligned with SoSyM Tool Paper criteria (Metamodel, 19-pattern catalog, tool architecture, simulation verifier, usability, evaluation, lessons learnt, open availability).
* **1.6 Relationship to Prior Work:** Conference precursor (RASSE 2025 [35]) re-measurement; companion GNN manuscript distinction.
* **1.7 Organization.**

### 2. Background and Related Modeling Tools
* **2.1 Model-Driven Engineering & ADLs:** AADL, Palladio Component Model (PCM/PerOpteryx), UML/SysML; limitations in capturing pub-sub semantics.
* **2.2 Architectural Smell & Anti-Pattern Detection Tools:** Arcan, Designite, DV8, SonarQube; contrast between code-derived graphs and pub-sub multi-layer models.
* **2.3 Model Refactoring: Open-Loop Search vs. Counterfactual Verification:** Graph transformation, SBSE (NSGA-II) vs. SaG's closed-loop simulation verification.
* **2.4 Explainable Quality Models & Graph Centrality:** Intrinsic score decomposition vs. post-hoc explainers; ISO 25010/25019 grounding.
* **2.5 Advance Over State of the Art:** Detailed feature comparison matrix (Table 1).

### 3. Graph Metamodel and Multi-Layer System Representation
* **3.1 Heterogeneous Graph Metamodel:** Typed multigraph $G = (V, E, \tau_V, \tau_E, w_V, w_E)$ (5 vertex types, 6 edge types).
* **3.2 Derived Dependencies (`DEPENDS_ON`):** 4 layer projections (Application, Infrastructure, Middleware, System) and 7 derivation rules.
* **3.3 Code Quality Penalty (CQP):** Bridging file-level metrics (LOC, WMC, LCOM, coupling instability) into architectural models.
* **3.4 Hierarchical RM Quality Model:** Definitions D1–D4; explicit formulas for Fault Tolerance ($FT$), Availability ($A$, including $\mathrm{QSPOF}$ interaction), Maintainability ($M$), and Composite Criticality ($Q$); row-stochastic Quality-in-Use harm projection $\mathbf{M}_{\text{RM} \to \text{QiU}}$.
* **3.5 Substrate Adequacy:** Formal theorem on layer requirements for topological cut-vertex metrics.

### 4. A Catalog of Architectural Anti-Patterns for Publish–Subscribe Systems
* **4.1 Anti-Patterns vs. Bad Smells in Systems Modeling:** Taxonomy and confidence tiers.
* **4.2 Detection Methodology:** 15 continuous metrics evaluated with adaptive box-plot fences ($Q_3 + k\,\mathrm{IQR}$).
* **4.3 Catalog Overview:** Table detailing 19 patterns across Reliability, Availability, Maintainability, and Architecture, with finding scope (Component/Edge/Pair/System) and benchmark trigger flags.
* **4.4 Representative Walkthroughs:** SPOF, God Component, Broker Overload, Chatty Pair, QoS Mismatch.
* **4.5 Validation Methodology:** Cascade simulator oracle $I_{\text{comp}}$.
* **4.6 Catalog Coverage on Benchmark Corpus:** Disclosure that 8 patterns trigger on synthetic corpus; 98.4% of findings are edge- or system-scoped.

### 5. Tool Architecture and Implementation Techniques
* **5.1 Hexagonal Architecture & Dual-Repository Design:** Core domain layer decoupled via `IGraphRepository` port; `Neo4jRepository` for persistent queries and visualization; `MemoryRepository` for microsecond in-memory copy-on-write sandboxing.
* **5.2 Seven Pipeline Stages:** Ingest $\to$ Project $\to$ Attribute $\to$ Detect $\to$ Compile $\to$ Verify $\to$ Present/Gate.
* **5.3 Graph-Mutation Refactoring Operators:** Topic Splitting, Anti-Affinity Reallocation, QoS Hardening; automation footprint disclosure (5 direct triggers, 14 advisory).
* **5.4 Closed-Loop Counterfactual Verification Engine:** Mathematical formulation of the screening objective; Level 1 per-edit margin filter ($\overline{\Delta I}_\theta > \kappa \sigma_{\text{seed}}$); Level 2 whole-policy gate ($\Delta\mathrm{SRI} > 0$).
* **5.5 Substrate Independence and Sandboxed Execution:** Independence between diagnostic candidate generation and structural simulation verification.

### 6. Tool Interfaces, Usability, and Engineering Ecosystem
* **6.1 Multi-Modal Interfaces:**
  - *Interactive Web Dashboard & Graph Explorer:* Next.js/React and standalone HTML dashboards (2D/3D force-directed layout, cascade heatmaps, RM radar charts).
  - *Python SDK & Client API:* `SaagClient` for programmatic workflows and Jupyter notebooks.
  - *Modular CLI Suite:* Headless command-line utilities for pipeline automation.
  - *RESTful API Service:* Asynchronous FastAPI microservice with OpenAPI 3.0 specification.
* **6.2 DevOps Integration and CI/CD Quality Gating:** Absolute vs. delta-aware regression semantics, waiver registers, and 3-tier exit code protocol (0: Pass, 1: Warn, 2: Block).

### 7. Empirical Application in Realistic Scenarios
* **7.1 Benchmark Scenarios and Real-World Corpora:** Table of 8 synthetic scenarios (S01–S08, 29–520 vertices) + 3 transcribed real-world open-source systems (Autoware.Universe, Train-Ticket, Cloud IoT Mesh).
* **7.2 Research Questions and Evaluation Protocol:** RQ1 to RQ5; metrics, seed protocols, and oracle definitions.
* **7.3 Diagnostic Ranking Efficacy & Predictor Saturation (RQ1):** Tables comparing $Q(v)$, betweenness, and degree; saturation around NDCG@10 $\approx 0.82$--$0.86$.
* **7.4 Catalog Calibration on Synthetic vs. Transcribed Topologies (RQ2):** Chance agreement on synthetic ($\kappa = -0.036$) vs. fair discrimination on real-world systems ($\mathbf{\kappa = +0.296}$).
* **7.5 Attribution Validity & Substrate Adequacy (RQ3):** Availability degeneracy at the app layer; reconciliation with RASSE 2025 claim; Simpson's paradox in multi-type node pooling ($\rho = 0.028$ vs. $0.503$).
* **7.6 Prescriptive Refactoring Performance & Operator Survival (RQ4):** Tables of refactoring yield (1128/1589 edits admitted, $71.0\%$), per-operator survival (Reallocation $77.7\%$, QoS $58.0\%$, Topic Splitting $49.7\%$), and independent cascade impact reduction ($+3.52\%$).
* **7.7 Tool Performance, Scalability, and Latency (RQ5):** Diagnostic gate latency 0.01–20.98s; full 7-scenario prescriptive sweep (14,301 simulation runs) in 2h 47m on 20 cores.

### 8. Community Importance and Lessons Learnt (MANDATORY FOR SOSYM)
* **8.1 Importance for Systems and Software Modeling:**
  1. Fills MDE blind spot for asynchronous pub-sub architectures.
  2. Open-source experimental benchmark testbed with declarative models.
  3. Reusable catalog of formal anti-patterns and mutation operators.
  4. Operationalization of ISO/IEC 25010/25019 standards in graph models.
* **8.2 Five Transferable Lessons Learnt:**
  1. *Designing:* Metamodel Substrate Adequacy & Layer Awareness.
  2. *Building:* Dual Repositories for Counterfactual Model Transformation.
  3. *Testing:* The "Verify-Before-Transform" Discipline ($29.0\%$ candidate rejection).
  4. *Evaluating:* Explanatory Attribution vs. Centrality Ranking.
  5. *Disseminating:* Multi-Modal Ergonomics for Diverse Modeling Personas.

### 9. Discussion and Tool Availability
* **9.1 Threats to Validity:** Construct, internal, external, and conclusion validity.
* **9.2 Tool Availability, Reproducibility, and Open Science:**
  - Apache 2.0 open-source license.
  - Public GitHub repository.
  - Permanent Zenodo archive with DOI.
  - Pip package and Docker Compose deployment.
  - Automated reproduction package under `reproduce/`.

### 10. Conclusion and Future Work
* **10.1 Conclusion:** Synthesis of tool contributions.
* **10.2 Future Work:** Layer-aware availability metrics, search-based multi-objective optimization (SBSE), combinatorial subset verification, external tool benchmarking, automated IDL extraction, LLM-assisted refactoring PR generation.

### Declarations
* Funding, Competing Interests, Overlap Disclosure, Data Availability, Author Contributions, Ethics Approval.
