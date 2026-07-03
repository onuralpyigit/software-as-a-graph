# Outline: "Graph Neural Networks for Architectural Dependability and Cascading Failure Prediction in Complex Distributed Systems"

*Target: JSS Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems" (VSI:AI4MSS). Deadline: 30 September 2026.*

SI topic mapping: **AI for Reliability and Dependability Analysis in Complex ICT Systems** (primary); **Explainable, Interpretable, and Robust AI in Performance Analysis and Optimization** (secondary, via the $Q(v)$-vs-$HGL$ dual-path design); **AI for Automated Performance Tasks** (secondary, via the CI/CD quality gate).

---

## Abstract
- Problem: pre-deployment cascade-impact prediction in pub-sub middleware, no runtime telemetry available.
- Approach: SaG typed multigraph (5 node types) + AHP-weighted RMAV composite $Q(v)$ + Heterogeneous Graph Transformer ($HGL$-QoS) $I(v)$ predictor, validated against an independent discrete-event simulator under input–label independence.
- Headline results: in-distribution $\rho>0.87$, $F_1>0.90$ (attribution) vs. LOSO $\rho=0.401$ (HGL-QoS); stratified $\rho$ 0.322–0.429 across 5 node types; prescriptive remediation mean $+4.61\%$ (mixed, negative in 3/7 scenarios); gate 100% precision/recall, ~5s medium / ~40s xlarge.

## 1. Introduction
- 1.1 Motivation — pub-sub decoupling hides dependency chains; DDS/MQTT QoS shapes behavior under stress.
- 1.2 Architecture-Code Gap / problem statement — quality attribution + failure-impact analysis, both explainable, both pre-execution.
- 1.3 Limitations of existing approaches — SCA (blind to topology), chaos engineering (post-deployment only), topology-only centrality / homogeneous GNNs (representation collapse).
- 1.4 Approach + 4 RQs:
  - **RQ1** — interpretable attribution vs. learned HGT: when does each suffice?
  - **RQ2** — fault modes multi-dimensional attribution exposes that single-score centrality misses.
  - **RQ3** — QoS edge-feature injection: in-distribution convergence vs. LOSO generalizability.
  - **RQ4** — operational feasibility/overhead as a blocking CI/CD quality gate.

## 2. Related Work
- 2.1 Pub-sub dependability & protocol resilience (reactive/runtime-only).
- 2.2 SCA vs. SSA (local code metrics vs. global architecture-as-code).
- 2.3 GNNs on heterogeneous software topologies (FINDER, DrBC, RGCN, HGT) — homogeneous-network limitation.
- **Gap flagged in project notes:** this section is thin (~15 refs across the whole paper) relative to SI reviewer expectations. Needs a dedicated pass citing 2022–2026 AI-for-dependability work (learned failure prediction in microservice meshes, GNN-based root-cause analysis / anomaly detection on distributed traces, ML-assisted reliability assessment) to differentiate SaG's pre-deployment/architecture-only stance from runtime-observability approaches. *(Action item, not yet drafted.)*

## 3. The Software-as-a-Graph Modeling Formalism
- 3.1 Typed multigraph $G=(V,E,\tau_V,\tau_E,w_E,w_V)$ over App/Broker/Topic/Node/Library; 6 edge types.
- 3.2 QoS-driven edge/vertex weighting — `QoS_score` (reliability/durability/priority AHP blend), size-norm, $w(e)$ blend ($\beta=0.85$), per-type $w_V(v)$ formulas.
- 3.3 Six `DEPENDS_ON` logical projection rules (App→App, App→Broker, Node→Node, Node→Broker, App→Library "shared-library blast", Broker→Broker colocation).

## 4. Multi-Dimensional Quality Attribution (Interpretable Path)
- 4.1 Four orthogonal dimensions: Reliability, Maintainability, Availability, Vulnerability.
- 4.2 Analytical formulations — $R(v)$ (transpose-graph cascade, topic fan-out variant), $M(v)$ (betweenness + CQP code-debt), $A(v)$ (directed articulation points, QSPOF, blast radius), $V(v)$ (reverse reachability).
- 4.3 Composite $Q(v)$ via AHP pairwise matrix (CR≈0.02); raw weights $(0.43,0.24,0.17,0.16)$ shrunk ($\lambda=0.70$) to final $(0.38,0.24,0.19,0.19)$.

## 5. Heterogeneous GNN for Failure-Impact Forecasting (Learned Path)
- 5.1 Ground-truth label $I(v)$ from discrete-event simulator (reachability loss/fragmentation/throughput loss/flow disruption blend), 5-seed mean, `propagation_threshold=0.2`.
- 5.2 $HGL$-QoS architecture — relation-specific HGT attention over 5 node / 6 edge types, QoS weights injected into edge attention.
- 5.3 Input–label independence guarantee — features from $G_{\text{analysis}}$, labels from $G_{\text{structural}}$, no leakage path.

## 6. Continuous Operationalization: CI/CD Quality Gating & Prescription
- 6.1 Delta-aware gating via git merge-base diffing; baseline register + waiver register.
- 6.2 POSIX exit-code protocol (0 pass / 1 warn / 2 block).
- 6.3 Database-free `MemoryRepository` execution for build-runner independence.
- 6.4 Four prescriptive operators: `RedundancyInsertion`, `PathDiversification`, `FanOutReduction`, `SharedTopicReduction`.

## 7. Experimental Setup
- 7.1 Seven synthetic scenarios (Clinical Healthcare → Hyper-Scale Enterprise), tiny→xlarge, distinct dominant failure vectors per scenario.
- 7.2 Baselines: RMAV/$Q$, GL, GL-QoS, HGL, HGL-QoS, Topo-BL/Topo-QoS.

## 8. Experimental Evaluation and Results
- 8.1 **RQ1** — ID: attribution $\rho>0.87$, $F_1>0.90$ beats HGL ($\rho\approx0.62$); OOD (LOSO): HGL-QoS $\rho=0.401$ vs. homogeneous $\rho\approx0.02$ (table of 4 variants).
- 8.2 **RQ2** — heterogeneity gain $\Delta F_1=+0.284$ ID / $\Delta\rho=+0.286$ OOD; shared-library blast hypothesis **not confirmed** (honest negative, max library $Q=0.422\to I=0.086$); stratified check: pooled $\rho=0.374$, per-type range $0.322$–$0.429$ across 1,545 components (5 types, all $p<0.01$).
- 8.3 **RQ3** — QoS features: minor ID null result, decisive OOD driver ($\rho$ 0.307→0.401).
- 8.4 **RQ4** — runtime scaling (<2s tiny/small → ~40s xlarge); gate precision/recall = 1.0/1.0; prescriptive remediation table (7 scenarios, mean $+4.61\%$, regressions in 3/7) tied to the open `PrescribeService` implementation gap (no per-edit acceptance filter).

## 9. Discussion, Limitations, and Conclusion
- 9.1 Threats to validity — construct (simulated vs. real crashes), internal (leakage mitigated by independence guarantee), external (synthetic scenarios only, no production/practitioner validation).
- 9.2 Limitations/future work — thin security dimension, unwired per-edit remediation filter (highest priority), no live-cluster validation.
- 9.3 Conclusion — division of labor between deterministic attribution (ID) and typed GNN (OOD); operational feasibility as a blocking gate.

## References
- 15 entries currently (pub-sub surveys, DDS/MQTT standards, centrality, cascading-failure network science, GNN critical-node ID, RGCN/HAT/HGT/MAGNN, AHP). No SI-specific AI-for-reliability citations yet — see Related Work gap above.

---

## Pre-Submission Action Items
1. **Literature pass** (§2) — add 6–10 citations on AI for reliability/dependability of ICT systems (2022–2026) to meet SI reviewer expectations; this is the main structural gap in an otherwise complete draft.
2. **Disjointness/disclosure check** — confirm final scope stays clearly separated from the flagship regular-track submission ("A Static System Analysis Framework...") and disclose any shared preprint/dataset lineage in the cover letter per SI originality rules.
3. **Guide-for-Authors formatting pass** — convert to elsarticle class, verify reference style, select article type "VSI:AI4MSS" at submission.
4. No open blockers or bracketed placeholders remain in the body text itself — draft is otherwise submission-track-ready pending the above.
