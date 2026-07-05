# Outline: SaG-Prescribe — Closed-Loop, DevOps-Integrated Prescriptive Refactoring of Distributed Publish–Subscribe Middleware Architectures

*Target: Automated Software Engineering (AuSE), Springer — Special Issue on "Intelligent Techniques for
Automated Code Review and Software Quality Evaluation."*
*SI topic mapping: **Intelligent techniques for CI/CD, DevOps, software evolution, technical debt
analysis, and refactoring recommendation** (primary bullet). No AutoML/NAS/LLM contribution is
claimed — the DevOps-integrated CI/CD gate and the closed-loop refactoring-recommendation pipeline
carry this submission into scope.*

This document summarizes the merged manuscript at
[SaG-Prescribe - Closed-Loop DevOps-Integrated Prescriptive Refactoring.md](SaG-Prescribe%20-%20Closed-Loop%20DevOps-Integrated%20Prescriptive%20Refactoring.md),
which consolidates three prior overlapping drafts in this directory into a single canonical submission
draft.

---

## Title

**SaG-Prescribe: Closed-Loop, DevOps-Integrated Prescriptive Refactoring of Distributed
Publish–Subscribe Middleware Architectures**

---

## Abstract

Distributed publish–subscribe middleware (ROS 2, DDS, MQTT) decouples producers and consumers, but the
resulting indirect dependency structure obscures how component failures cascade, and this architectural
technical debt accumulates invisibly to code-level static analysis (SCA) tools. Existing structural
diagnostic frameworks can rank components by criticality, yet they operate *open-loop*: they tell
architects which components are fragile without producing verified guidance on how to restructure the
topology. We present **SaG-Prescribe**, a closed-loop, DevOps-integrated prescriptive refactoring
system that extends the Software-as-a-Graph (SaG) diagnostic framework. SaG-Prescribe augments
topological diagnostics with a code-quality-penalty signal derived from static-analysis metrics (LOC,
cyclomatic complexity, LCOM, technical-debt ratio), and compiles the resulting criticality diagnosis —
components flagged `CRITICAL` or `HIGH` by adaptive box-plot fences — into a transformation policy
$\Delta(G)$ composed of three graph mutation operators: logical topic splitting, physical anti-affinity
reallocation, and transport QoS contract hardening. Each candidate topology $G' = \Delta(G)$ is
re-evaluated by the same discrete-event cascade simulator that produced the diagnosis, so every accepted
prescription is verified — not merely recommended — before it reaches an architect or a pull request.
The pipeline is further operationalized as a delta-aware CI/CD quality gate
(`detect_antipatterns.py`) that blocks only newly introduced structural regressions relative to a Git
merge base, within sub-minute execution bounds.

Across seven parameterized publish–subscribe scenarios spanning autonomous vehicles, IoT, finance,
healthcare, and hyper-scale enterprise topologies, the generated prescriptions reduce the System Risk
Index (SRI, lower is better) in every scenario, with relative risk reductions between 1.4% and 15.9%
(Wilcoxon signed-rank across the seven baseline/mutated pairs: $W=0.0$, $p=0.0156$, significant at
$\alpha=0.05$). We report this result alongside a less flattering but more honest one: component-level
failure-impact reductions average only +4.61% and regress in two of seven scenarios (−31.67% and
−25.36%), because the current policy is applied greedily and unconditionally, occasionally introducing
new cascade hops that offset its own gains — a finding that directly motivates a per-edit acceptance
filter as the highest-priority extension. The full generate–verify loop completes in under 65 seconds
for six of the seven scenarios (4.7 s–64.3 s) and in approximately 10.8 minutes for the 300-process
enterprise topology, while the CI/CD gate script alone completes in under 2 seconds to ~40 seconds
across the same scale range — making closed-loop prescriptive refactoring practical as a
merge-request-gated or nightly pre-deployment CI/CD stage.

**Keywords:** refactoring recommendation; publish–subscribe middleware; CI/CD quality gates; DevOps;
architectural technical debt; failure cascade simulation; graph mutation; search-based software
engineering

---

## Paper Outline

**1. Introduction**
1.1 Context and motivation — pub-sub decoupling as the backbone of cyber-physical/microservice/IoT
systems; decoupling obscures failure propagation.
1.2 The Architecture-Code Gap and the open-loop refactoring-recommendation gap — SCA tools are
topology-blind; existing structural diagnostics (including our own prior SaG framework) rank
criticality but don't prescribe or verify remediation.
1.3 Proposed solution: SaG-Prescribe — CQP-augmented diagnostics → mutation policy $\Delta(G)$ →
closed-loop resimulation → DevOps quality gate.
1.4 Contributions — four explicit, numbered items (pipeline, operators, CI/CD gate, evaluation incl.
the honest component-level result).

**2. Background and Related Work**
2.1 Publish–subscribe middleware dependability (broker fault tolerance, Kafka, DDS QoS/latency
literature).
2.2 Refactoring recommendation and architectural technical debt — code-scope recommenders (smell-driven,
learning-based, LLM-based) vs. this paper's topology scope and verify-before-recommend model.
*(citation slots pending — no invented references)*
2.3 Search-Based Software Engineering and architecture optimization — open-loop vs. this paper's
closed-loop verification.
2.4 Diagnostic foundation (SaG) — what this paper builds on and does not repeat.
2.5 Structural criticality analysis — why classical centrality degrades on typed pub-sub graphs.

**3. System Model and Code-Quality-Augmented Technical Debt Analysis**
3.1 Heterogeneous graph formulation — 5 node types, 6 structural edge types.
3.2 Derived `DEPENDS_ON` projection (App↔App, App↔Broker, App↔Library, Broker↔Broker).
3.3 Code Quality Penalty (CQP) — the paper's explicit bridge from SCA metrics to architecture-level
risk; feeds the Maintainability dimension below.
3.4 Multi-dimensional quality attribution (RMAV) — Reliability/Maintainability/Availability/
Vulnerability, AHP-weighted composite $Q(v)$, adaptive box-plot criticality tiers.

**4. Closed-Loop Optimization Objective**
Formal $\min_\Delta$ objective under a (currently unconstrained) modification budget; acceptance
criterion $\Delta\mathrm{SRI}>0$ as implemented in `PrescribeService`; note on the deferred
margin-based criterion.

**5. The SaG-Prescribe Prescriptive Pipeline**
5.1 Hexagonal core abstraction (`IGraphRepository`, `Neo4jRepository` vs. `MemoryRepository`).
5.2 Pipeline stages 1–7 (diagnostic foundation → prescriptive generation → review interface).
5.3 Three refactoring operators — logical topic splitting, physical anti-affinity reallocation,
transport QoS contract hardening — each formalized as a typed graph mutation rule.
5.4 Closed-loop verification procedure (export → mutate → sandbox → resimulate → $\Delta$SRI), with the
independence guarantee against circular leakage.

**6. DevOps Integration and Delta-Aware CI/CD Gating**
*(This section is the paper's most direct link to the SI's CI/CD/DevOps bullet — kept prominent.)*
6.1 Automated code review architecture — `detect_antipatterns.py` as a blocking CI check.
6.2 Delta-aware regression semantics — merge-base diffing, waiver register for accepted legacy risk.
6.3 Exit-code protocol — 0 (pass) / 1 (warn) / 2 (block).

**7. Experimental Design**
7.1 Four research questions (efficacy, operator contributions, component-level remediation efficacy,
computational overhead/CI-CD feasibility).
7.2 Seven benchmark scenarios (autonomous vehicle → hyper-scale enterprise) with scale table.
7.3 Experimental protocol — deterministic simulator, verified $\sigma_{\text{seed}}=0$ across 5 seeds.
7.4 Metrics — SRI formula, $\Delta$SRI, operator counts, component-level deltas.

**8. Results**
8.1 Prescriptive efficacy (RQ1) — SRI improves in all 7 scenarios (1.4%–15.9%), Wilcoxon $W=0.0$,
$p=0.0156$.
8.2 Operator contributions (RQ2) — anti-affinity reallocation most frequent overall; QoS upgrades
concentrate in high-loss profiles.
8.3 Remediation efficacy at component boundaries (RQ3) — **honest negative result**: mean +4.61%
component-level improvement but regressions in Hub-and-Spoke (−31.67%) and Hyper-Scale (−25.36%),
traced to greedy unconditional policy application; motivates the per-edit acceptance filter.
8.4 Computational overhead and CI/CD feasibility (RQ4) — full loop <65s for 6/7 scenarios, ~10.8 min at
hyper-scale; standalone CI/CD gate script sub-2s to ~40s across all scales.

**9. Discussion and Threats to Validity**
9.1 What closed-loop verification buys, framed against the §8.3 result.
9.2 Positioning in CI/CD and technical-debt workflows — advisory recommendations vs. blocking
regression gate.
9.3–9.6 Construct, internal, external, and conclusion validity.
9.7 Engineering trade-offs of the verification step.

**10. Conclusion and Future Work**
10.1 Summary of the end-to-end pipeline and headline results.
10.2 Future work, in priority order: (1) per-edit acceptance filter — highest priority, directly
motivated by §8.3; (2) budget-constrained policy search; (3) stochastic cascade propagation; (4)
operator ablation and learned policy ordering; (5) real-system replication (ATM topology); (6)
LLM-assisted PR generation — kept to a single sentence, not oversold.

**References** — 11 populated, real, verified (non-invented) citations spanning pub-sub dependability,
SBSE/architecture optimization, DDS QoS literature, centrality analysis, and AHP; 3 citation slots in
§2.2 (code-level refactoring-recommendation literature) remain to be populated before submission.

---

## Pre-Submission Checklist (carried over from consolidation)

1. ~~Consolidate three overlapping drafts into one and delete the duplicates~~ — **done**; see the
   provenance note at the top of the merged manuscript.
2. **Populate the three `[REF: …]` slots in §2.2** with real code-level refactoring-recommendation /
   architectural technical-debt citations (smell-driven, learning-based, and LLM-based recommenders).
   No references have been invented to fill these.
3. **Literature pass:** expand from 11 toward the ~30–45 references AuSE reviewers typically expect,
   once §2.2's slots are filled.
4. **Confirm companion-paper status** ([1], the Software-as-a-Graph JSS submission) at submission time
   and disclose per AuSE's originality/overlap policy in the cover letter.
5. **Do not expand the LLM future-work mention (§10.2 item 6)** beyond a single sentence — the paper has
   no LLM/AutoML contribution, and overselling it risks a scope-mismatch desk rejection.
6. Fill in **Funding**, **Competing interests**, and **Data availability** declarations before
   submission.
