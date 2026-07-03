# Outline: "Automated Prescriptive Refactoring of Distributed Middleware Architectures via DevOps-Integrated Graph Analytics"

*Target: Automated Software Engineering (Springer) — Thematic Collection "Intelligent Techniques for
Automated Code Review and Software Quality Evaluation." Deadline: 01 October 2026.*

SI topic mapping: **Intelligent techniques for CI/CD, DevOps, software evolution, technical debt
analysis, and refactoring recommendation** (primary — the only bullet this paper directly satisfies).
No AutoML/NAS/meta-learning/LLM contribution is present; §5 (DevOps-integrated quality gate) and the
refactoring-recommendation framing of §4 are what carry this submission into scope. The cover letter
should lean on this bullet explicitly rather than imply an AutoML/LLM angle the paper doesn't have.

---

## Abstract
- Problem: pub-sub middleware (DDS/ROS 2/MQTT) accumulates architectural technical debt invisible to
  code-level static analysis (SCA); existing structural diagnostics are open-loop — they flag risk
  without verified remediation guidance.
- Approach: SaG-Prescribe — code-quality-penalty-augmented heterogeneous graph diagnostics → rule-based
  mutation policy $\Delta(G)$ (topic splitting, anti-affinity reallocation, QoS hardening) → closed-loop
  discrete-event resimulation → delta-aware CI/CD quality gate (`detect_antipatterns.py`).
- Headline results: SRI improves in all 7 scenarios (1.4%–15.9% relative, Wilcoxon $p=0.0156$); mixed
  component-level results (mean +4.61%, regressions in 2/7 scenarios — reported honestly, not smoothed
  over); gate runtime <2s–~40s (tiny→xlarge), full generate–verify loop <65s for 6/7 scenarios, ~10.8 min
  at hyper-scale (300 processes).

## 1. Introduction
- 1.1 Context and motivation — pub-sub decoupling backbone for cyber-physical/microservice/IoT systems;
  decoupling obscures failure propagation; hardening requires pre-deployment optimization.
- 1.2 Architecture-Code Gap & Open-Loop Gap — SCA tools (SonarQube-style) are topology-blind; existing
  structural diagnostics rank criticality but don't prescribe or verify remediation.
- 1.3 Proposed solution: SaG-Prescribe — generate–verify loop over three mutation operators, sandboxed
  resimulation, CI/CD operationalization.
- 1.4 Contributions (currently implicit in prose — **should be extracted into an explicit numbered list**
  to match AuSE's expected structure; see Action Items).

## 2. Background and Related Work
- 2.1 Pub-sub middleware dependability — protocol verification, broker replication, chaos engineering
  (reactive/runtime-only); SaG-Prescribe operates earlier, on Architecture-as-Code descriptors.
- 2.2 Search-Based Software Engineering & architecture optimization — SBSE discovers refactoring
  blueprints but typically open-loop, no counterfactual verification against a failure model.
- 2.3 Structural technical debt analysis — centrality/articulation-point metrics degrade on typed
  pub-sub graphs (uniform-edge-semantics assumption); motivates the typed multigraph model.
- **Gap flagged relative to the other two drafts in this directory:** this version carries only 8
  references vs. 10 in "Closed-Loop Prescriptive Optimization.md," and has no dedicated subsection on
  code-level refactoring recommendation / LLM-assisted refactoring / architectural technical-debt
  management literature — the "Simulation-Verified Refactoring Recommendation.md" draft's §2.2 covers
  this ground (with `[REF]` placeholders) and should be merged in with real citations before submission,
  since AuSE reviewers will expect the paper to engage with code-scope refactoring-recommendation work
  even though this paper's scope is topology-level.

## 3. Multi-Dimensional Technical Debt Analysis
- 3.1 Heterogeneous graph formulation — $G=(V,E,\tau_V,\tau_E,w_E,w_V)$; 5 node types (Application,
  Library, Topic, Broker, Node); 6 structural edge types.
- 3.2 Derived `DEPENDS_ON` projection — App→App, App→Broker, App→Library (shared-library blast radius),
  Broker→Broker (colocation).
- 3.3 Code Quality Penalty (CQP) — rank-normalized SCA metrics (LOC, WMC, LCOM, `sqale_debt_ratio`)
  blended $0.10/0.35/0.30/0.25$ into a single per-component penalty; this is the paper's one explicit
  bridge from code-level quality signals to architecture-level risk, worth foregrounding for the SI's
  "software quality evaluation" framing.
- 3.4 Multi-dimensional quality attribution (RMAV) — Reliability/Maintainability/Availability/
  Vulnerability, AHP-weighted composite $Q(v)$, final weights $(0.38,0.24,0.19,0.19)$.

## 4. The SaG-Prescribe Refactoring Engine
- 4.1 Optimization objective — $\min_\Delta \sum_v I^*_{\Delta(G)}(v)$, evaluated via System Resilience
  Index (SRI) from counterfactual simulation.
- 4.2 Three mutation operators — logical topic splitting, physical anti-affinity reallocation, transport
  QoS contract hardening; triggered by adaptive box-plot criticality fences (`CRITICAL`/`HIGH`).
- 4.3 Closed-loop verification pipeline — export → mutate → sandbox (`MemoryRepository`) → resimulate →
  $\Delta\text{SRI}$ quantification, with a strict independence guarantee (diagnostics from
  $G_{\text{analysis}}$, ground truth from $G_{\text{structural}}$; no leakage).

## 5. DevOps Integration & Delta-Aware Gating
*(This section is this draft's differentiator vs. the other two SaG-Prescribe drafts — it's the
concrete "automated code review" artifact the SI is asking for. Keep it prominent; don't let a later
merge with the other drafts dilute it.)*
- 5.1 Automated code review architecture — blocking check script `detect_antipatterns.py` parsing
  Architecture-as-Code descriptors in CI.
- 5.2 Delta-aware regression semantics — merge-base diffing; only *new* regressions block; waiver
  register for accepted legacy risk.
- 5.3 Exit-code protocol — 0 (pass) / 1 (warn, surfaced to review dashboard) / 2 (block).

## 6. Experimental Evaluation
- 6.1 Evaluation suite — 7 synthetic scenarios (Autonomous Vehicle, IoT Smart City, Financial Trading,
  Healthcare, Hub-and-Spoke, Microservices Mesh, Hyper-Scale Enterprise), tiny→xlarge scale range.
- 6.2 Prescriptive optimization results — SRI improves in all 7 (1.4%–15.9%), Wilcoxon $W=0.0$,
  $p=0.0156$; largest gain in IoT Smart City (QoS hardening on best-effort links); anti-affinity
  reallocation is the most frequently emitted operator overall.
- 6.3 Remediation efficacy at component boundaries — **honest negative result**: mean +4.61% component-
  level improvement but regressions in Hub-and-Spoke (−31.67%) and Hyper-Scale (−25.36%), traced to
  greedy unconditional policy application introducing new cascade hops; motivates future per-edit
  acceptance filter. Keep this section — it pre-empts the most likely reviewer pushback on "did you just
  cherry-pick the aggregate metric?"
- 6.4 CI/CD gating feasibility — sub-2s (tiny/small) to ~40s (xlarge) for the gate script; full
  generate–verify loop <65s for 6/7 scenarios, ~10.8 min at hyper-scale.

## 7. Discussion and Threats to Validity
- 7.1 Construct validity — SRI is simulator-defined; operators chosen for engineering validity
  independent of the simulator (anti-affinity, QoS hardening are established dependability practice).
- 7.2 Internal validity — independence guarantee rules out circular leakage between diagnostics and
  ground-truth labels.
- *(Missing relative to sibling drafts: no External Validity subsection. Both other SaG-Prescribe drafts
  in this directory carry one — synthetic-only scenarios, no industrial replication. Add for
  consistency and because AuSE reviewers will ask.)*

## 8. Conclusion and Future Work
- Summary: end-to-end pipeline from code-level debt + topological vulnerability to simulation-verified,
  CI/CD-gated refactoring commands.
- Future work: per-edit simulation filter (highest priority, directly motivated by §6.3), stochastic
  cascade models, LLM-assisted PR generation from synthesized refactoring blueprints (currently the
  paper's *only* mention of LLMs — do not oversell this in the abstract/intro).

## References
- 8 entries currently (pub-sub surveys, DDS/MQTT standards, SBSE, architecture optimization survey,
  betweenness centrality, AHP, one GNN benchmark dataset paper whose relevance to this paper is unclear
  and should be checked). Thin relative to AuSE norms (~30–45 expected) and relative to the "Closed-Loop
  Prescriptive Optimization.md" sibling draft's 10, which adds concrete DDS-QoS and microservice-
  centrality citations worth pulling in.

---

## Pre-Submission Action Items
1. **Consolidate with sibling drafts, then delete them.** All three files in `docs/research/ause/`
   describe the same SaG-Prescribe system. Submitting more than one — or leaving multiple near-identical
   drafts sitting in the repo — creates a real risk of an overlapping-submission violation under AuSE's
   policy. Pull the real references from "Closed-Loop Prescriptive Optimization.md" (§2, refs [2]–[10])
   and the §2.2 code-refactoring-recommendation framing from "Simulation-Verified Refactoring
   Recommendation.md" into this draft, then remove the other two files once merged.
2. **Literature pass (§2, References).** Expand from 8 to ~20–30+ citations: pull in the sibling drafts'
   references, add code-level refactoring-recommendation / architectural technical-debt literature so
   the paper visibly engages with the SI's "refactoring recommendation" bullet at the code scope too, not
   only the architecture scope.
3. **Add explicit numbered contributions list in §1.4** and an **External Validity subsection in §7** —
   both present in the sibling drafts, both expected by AuSE reviewers, both currently missing here.
4. **Resolve the CQP/QoS-only positioning question:** since the SI explicitly wants "software quality
   evaluation," consider whether §3.3 (Code Quality Penalty) deserves more prominence — it's currently
   one formula in a subsection, but it's the paper's only direct tie to code-level quality metrics.
5. **Do not expand the LLM mention in §8** beyond a single future-work sentence — the paper has no LLM
   contribution, and overselling it in the cover letter or abstract risks a scope-mismatch desk rejection.
6. **Disclosure check.** The companion diagnostic framework (`[1]`, referenced throughout as the
   "companion paper") appears to be under review at JSS ("Software-as-a-Graph..."). Confirm its status
   at submission time and disclose in the AuSE cover letter per their originality/overlap policy.
7. No bracketed placeholders remain in this draft's body text — it is otherwise the most submission-ready
   of the three, pending the consolidation and literature-expansion work above.
