# Software-as-a-Graph (SaG) — SoSyM Tool Paper Outline

* **Title:** *Software-as-a-Graph: An Open-Source Tool for Multi-Layer Graph Modeling, Quality Diagnosis, and Simulation-Verified Refactoring of Publish–Subscribe Architectures*
* **Target Venue:** *Software and Systems Modeling* (SoSyM, Springer) — **Tool Paper** collection
* **License:** GNU GPL v3.0
* **Authoritative Source:** `latex/manuscript.tex` and `latex/sections/` (empirical macros generated in `latex/numbers.tex` from commit `58172b5f`, corpus digest `3afa81f0cddc`)

---

## Abstract

Publish–subscribe middleware such as ROS 2/DDS, Kafka and MQTT decouples components through topics, brokers and Quality-of-Service contracts. That decoupling hides the dependencies along which failures propagate, and mainstream architecture modeling tools represent topics, brokers and QoS only indirectly. We present Software-as-a-Graph (SaG), an open-source tool for modeling and analysing publish–subscribe architectures. SaG's modeling language is a typed multigraph metamodel with six vertex and seven edge types, a JSON concrete syntax, four layer projections, and seven declared rules that derive component dependencies from declared structure. Over these models the tool computes an explainable Reliability–Maintainability quality model grounded in ISO/IEC 25010 and detects nineteen publish–subscribe anti-patterns. It gates changes in continuous integration against their merge-base with a waiver register, and it simulates failure cascades. Its refactoring engine compiles findings into three graph-mutation operators and admits a candidate edit only if its simulated benefit exceeds seed-to-seed noise at every propagation threshold. SaG has a hexagonal architecture with a Neo4j adapter for persistent exploration and an in-memory adapter for throw-away model copies, and its architectural invariants are enforced by about 1,600 tests. It is used through a command-line suite, a Python API, a REST service, a web application and self-contained HTML reports. We apply SaG to a running example, five architecture models of open-source systems and eight synthetic scenarios. Diagnosis and gating take 0.04–79 seconds per model. The delta-aware gate separates new findings from existing debt. 32% of rule-generated refactoring edits fail verification. We report eight lessons learnt, most traceable to defects we found: state on which model view a metric is meaningful, make unmeasured values unrepresentable, test transformation rules on the model at hand, and tie every reported number to the artefact that produced it. SaG is available under GPL-3.0.

**Keywords:** Architecture modeling; Graph metamodel; Publish–subscribe middleware; Anti-pattern detection; Model refactoring; Simulation-based verification; Quality gates; Tool paper.

---

## Paper Structure

### 1. Introduction (`sec1_introduction.tex`)
* Motivation: why pub–sub decoupling hides failure-propagation dependencies and where existing ADLs/smell detectors fall short.
* **§1.1 Why a tool, and for whom:** Four target user groups — software/systems architects, reliability engineers, DevOps teams, and modeling researchers.
* **§1.2 Contributions:** Six tool-paper contributions (modeling language, analysis pipeline, hexagonal architecture & invariants, interfaces & extension points, scenario applications, and defect-grounded lessons learnt).
* **§1.3 Relationship to our other publications:** Explicit boundary against RASSE 2025, two UYMS 2026 papers (Turkish), a companion JSS manuscript under review (graph learning), and an unpublished companion manuscript (*SaG-Prescribe*, held in reserve).

### 2. Related Modeling and Analysis Tools (`sec2_related_work.tex`)
* **§2.1 Architecture description languages and component models:** AADL/OSATE and Palladio/PerOpteryx vs. SaG's first-class pub–sub elements.
* **§2.2 Architecture-smell and anti-pattern detectors:** Arcan, Designite, DV8, SonarQube, and performance/scalability anti-pattern detectors vs. topic/broker/QoS-aware patterns.
* **§2.3 Model refactoring and verification:** Graph transformation and SBSE/PerOpteryx vs. SaG's simulation-filtered rule operators.
* **§2.4 Quality models and centrality:** ISO/IEC 25010/25019 intrinsic decomposition vs. simple graph centralities (reporting both for explainability and infrastructure transparency).
* **§2.5 Positioning (Table 1):** Capability comparison matrix across Palladio/PerOpteryx, AADL/OSATE, Arcan/Designite, SonarQube, and SaG.

### 3. The SaG Modeling Language (`sec3_modeling_language.tex`)
* **§3.1 Running example — `mini-ATM` (Fig. 1):** 15-element air-traffic surveillance architecture (`examples/mini_atm.json`) carried through §§3–7.
* **§3.2 Abstract syntax — the metamodel (Fig. 2):** Typed, attributed, directed multigraph with 6 vertex types (`Application`, `Library`, `Topic`, `Broker`, `Node`, `NetworkDevice`) and 7 edge types (6 structural + derived `DEPENDS_ON`), plus QoS/size/frequency AHP-based weights.
* **§3.3 Concrete syntax (Listing 1):** Declarative JSON schema produced by hand, by the YAML scenario generator, or by external extractors.
* **§3.4 Derived dependencies (Table 2):** Seven explicit derivation rules (`app_to_app`, `app_to_broker`, `node_to_node`, `node_to_broker`, `app_to_lib`, `broker_to_broker`, `node_to_device`), yielding 26 derived edges from 28 structural relations on `mini-ATM`.
* **§3.5 Layer projections & Proposition 1 (Substrate Adequacy):** Four projections (`app`, `infra`, `mw`, `system`) and why cut-based metrics lose discriminative power on projections lacking cut vertices.

### 4. Model Analyses (`sec4_analyses.tex`)
* **§4.1 Structural quality — the RM model (Fig. 3):** ISO/IEC 25010 three-level decomposition (`FT`, `A`, `M` with `CQP` → `R` → `Q`), box-plot criticality classification, template-based explanations, and `mini-ATM` breakdown.
* **§4.2 Anti-pattern detection (Table 3):** 19 declarative patterns (`PatternSpec`) with relative box-plot fences, including pub–sub-specific `QOS_MISMATCH` and `CHATTY_PAIR`, plus `mini-ATM` findings and the known `ISOLATED`-topic false positive at the `system` layer.
* **§4.3 Quality gate:** Absolute vs. delta-aware CI gating (`--baseline`), exit codes `0/1/2`, 10% fence margin, warn-only `BOTTLENECK_EDGE` in delta mode, and expiring YAML/JSON waiver register (`--waivers`).
* **§4.4 Failure simulation:** Structural-only `FailureSimulator` cascade oracle (kept separate from `FaultInjector` training labeler), plus optional message-flow, event-driven, and CycloneDDS testbed simulators.
* **§4.5 Verified refactoring (`Prescribe`):** Three operators (*Topic split*, *Anti-affinity reallocation*, *QoS hardening*) guarded by counterfactual simulation across propagation thresholds $\Theta = \{0.1, 0.2, 0.5\}$ and seeds $S = \{42, 123, 456\}$ (Eq. 1 acceptance rule), plus SRI consistency check.
* **§4.6 Further analyses:** Brief overview of *Predict* (heterogeneous GNNs), *Triage* (top-$K$ + RM root-cause profiles), and *Deploy* (`saag-deploy` MIP/greedy placement).

### 5. Tool Architecture and Implementation (`sec5_tool_architecture.tex`)
* **§5.1 Hexagonal structure (Fig. 4):** ~65k LOC Python core (`saag/`, CLI, FastAPI) + ~24k LOC TypeScript/React front end (`smart/`); domain services decoupled via `IGraphRepository` and use-case boundaries.
* **§5.2 Two repository adapters:** `Neo4jRepository` (persistent Cypher exploration) vs. `MemoryRepository` (fast, isolated deep-copy sandbox for parallel refactoring verification).
* **§5.3 Pipeline and data flow:** File-based JSON step boundaries with provenance stamps (git commit, corpus digest, topology fingerprint).
* **§5.4 Invariants enforced by tests:** Nine architectural invariants guarded by ~1,600 tests across 130 files (oracle independence, labeler/oracle separation, `None` vs `0.0` unmeasured gates, SHA-256 corpus regeneration, weight consistency, Eq. 1 acceptance).

### 6. Using and Extending the Tool (`sec6_tool_usage.tex`)
* **§6.1 Command line:** 10 console scripts (`saag`, `saag-import`, `saag-analyze`, `saag-diagnose`, `saag-simulate`, `saag-visualize`, etc.) and `cli/prescribe_graph.py`.
* **§6.2 Python API:** Fluent `saag.Pipeline` builder (supports `MemoryRepository` without Neo4j) and step-by-step `saag.Client`.
* **§6.3 REST service and web dashboard (Fig. 5a–c):** FastAPI OpenAPI service, Next.js/React + ECharts web app (`docker compose up`), and self-contained HTML reports (`DashboardGenerator`).
* **§6.4 Walkthroughs (Fig. 5d):** Nine executable `examples/` scripts on `mini-ATM` that regenerate interactive pages from live tool output and self-check computations.
* **§6.5 Extension points:** Adding an anti-pattern, storage backend, refactoring operator, JSON input adapter, or registered metric.

### 7. Application in Scenarios (`sec7_application.tex`)
* **§7.1 Protocol:** Exact reproduction commands, 20-core workstation setup, 5-seed sweep, and exclusion of `DEEP_PIPELINE`.
* **§7.2 Running example end to end:** `mini-ATM` full pipeline (`< 1 s`) and 10-app ATM refactoring walkthrough (32 candidates → 8 admitted; 19 rejected edits would have increased simulated impact).
* **§7.3 Architecture models of open-source systems (Table 4):** Five hand-authored models (*Autoware.Universe*, *Train-Ticket*, *Cloud e-commerce mesh*, *EdgeX Foundry*, *Home Assistant*; 60–90 components).
* **§7.4 Synthetic corpus — cost, agreement, and refactoring (Tables 5–7):** Eight scenarios (29–520 components) covering scalability, RM-vs-centrality oracle agreement, and verified refactoring yield/composition.
* **§7.5 CI scenario (Table 8):** Feedback-loop change on `mini-ATM` plus 90 random single-subscription edits across 9 models comparing absolute vs. delta-aware gates.

### 8. Importance for the Community and Lessons Learnt (`sec8_community_and_lessons_learnt.tex`)
* **§8.1 Why the tool matters:** First-class pub–sub metamodel, reproducible oracle-backed experimental platform, simulation-as-verification for model transformations, and PR-level practitioner integration.
* **§8.2 Lessons learnt:** Eight transferable engineering and research lessons traced to real defects encountered while building and evaluating SaG.

### 9. Limitations, Threats to Validity, and Availability (`sec9_limitations_availability.tex`)
* **§9.1 Limitations:** Declared-model input (no built-in ROS 2/DDS/Kafka extractor), structural rather than behavioral metrics, known `ISOLATED` topic false positive, greedy individual-edit refactoring, CLI/REST coverage gaps, and `prescribe_graph.py` packaging note.
* **§9.2 Threats to validity:** Simulation-oracle construct validity, hand-authored open-source models and lack of user study (external), small scenario count $n = 8$ (conclusion), hardware-dependent timings.
* **§9.3 Availability:** GPL-3.0 GitHub repository, Zenodo archive (DOI pending), `pip install -e .`, Docker Compose stack, and reproduction scripts.

### 10. Conclusion and Future Work (`sec10_conclusion.tex`)
* Summary of the tool, empirical findings, and trust-oriented lessons.
* Future work: automated config extractors, layer-aware Availability, subset-selection refactoring, Ecore/EMF export, full REST coverage, and practitioner user studies.

### Declarations & Appendix (`declarations.tex`, `appendix_formulas.tex`)
* **Declarations:** Funding (TÜBİTAK), competing interests, full overlap disclosure (RASSE 2025, UYMS 2026 ×2, JSS, reserved *SaG-Prescribe*), code/data availability.
* **Appendix A — RM Model Formulas:** Complete mathematical specification of `CQP`, structural inputs, `FT`, `A`, `M`, `R`, `Q`, ISO/IEC 25019 QiU projection, and box-plot classification thresholds.

---

## Key Points & Headline Takeaways

### Core Technical Innovations
1. **First-class pub–sub metamodel with derived dependencies:** Separates declared topology (6 structural edge types) from derived failure-propagation structure (`DEPENDS_ON` via 7 explicit rules) across 4 explicit layer projections.
2. **Explainable quality scoring + pub–sub anti-patterns:** Linear ISO/IEC 25010 RM decomposition paired with 19 scale-free (box-plot) anti-pattern detectors and a practical delta-aware CI gate with fence margins and expiring waivers.
3. **Counterfactual verification of refactoring rules:** Treats every model transformation as a hypothesis tested against an independent structural failure simulator in isolated in-memory copies.
4. **Test-enforced research integrity:** ~1,600 tests enforce oracle independence, labeler separation, explicit `None` for unmeasured gates, and byte-identical corpus regeneration.

### Key Empirical Findings
* **Runtime & CI feasibility:** Diagnosis takes **0.05–0.11 s** on open-source models (60–90 components), **0.04–5.53 s** on synthetic models up to 326 components, and **78.93 s** at 520 components (detectors $\le 0.17\text{ s}$).
* **Open-source vs. synthetic catalog precision:** Direct findings flag **14–22%** of components on open-source models, with higher oracle agreement ($\kappa = +0.26$, precision $0.39$) than denser synthetic models ($\kappa = -0.03$, precision $0.24$).
* **RM vs. simple centralities:** On the `app` layer, $Q(v)$ achieves mean Spearman $\rho = 0.62$ vs. $0.50$ (betweenness) and $0.49$ (degree), beating degree in **8 of 8** scenarios (Wilcoxon $p = 0.008$). At the `system` layer, $Q$ is strong for applications ($\rho = 0.64$) but weak for hosts ($0.21$) and brokers ($0.12$).
* **Refactoring verification:** **1,028 / 1,510** candidates admitted (**68.1%**; **32%** rejected as noise or harmful). Survival by operator: reallocation **72.4%**, QoS hardening **67.3%**, topic split **55.3%**. Combining individually admitted edits improves mean impact in **4 of 7** scenarios and regresses in **3 of 7** (while SRI improves in **7 of 7**), proving that composition must be verified separately.
* **CI gating (90 random edits):** Absolute gate blocks **100%** of changes; delta-aware gate blocks **26%**, warns on **47%**, and passes **28%**.

### Eight Lessons Learnt
1. **State the view for every metric** (`SPOF` has $F_1 = 0.01$ on the `app` layer because it lacks cut vertices).
2. **Make "not measured" unrepresentable as `0.0`** (use `None` and test oracle liveness).
3. **A test double can become the key production adapter** (`MemoryRepository` enabled 13,653 parallel counterfactual simulations).
4. **Never trust a transformation rule—or the composition of admitted rules—without simulation.**
5. **Tie every number to versioned, provenance-stamped artefacts** (fixing the oracle and corpus reversed an earlier RM-vs-degree conclusion).
6. **Design CI checks for the merge decision** (delta diffs, fence margins, waivers, and bounding path-enumeration cost).
7. **Executable walkthroughs act as self-testing documentation** that catches code–doc drift missed by unit tests.
8. **Preserve multigraph semantics end to end** so parallel pub–sub edges never collapse in downstream graph libraries.
