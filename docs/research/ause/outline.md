# Outline: Graph-Based Detection of Architectural Anti-Patterns and Prescriptive Refactoring in Distributed Publish–Subscribe Systems

*Target: Automated Software Engineering (AuSE), Springer — Special Issue on
**"Intelligent techniques for CI/CD, DevOps, software evolution, technical debt analysis, and
refactoring recommendation."***
*SI topic mapping: this submission covers three of the bullet's five named areas directly —
**technical debt analysis** (the anti-pattern catalog, §4), **refactoring recommendation** (the
prescriptive operators, §6), and **CI/CD/DevOps** (the delta-aware quality gate, §7). No AutoML/NAS/
LLM contribution is claimed.*

This document revises the prior SaG-Prescribe outline to make **anti-pattern detection** a
first-class, explicitly evaluated contribution of this paper rather than an input inherited from a
companion diagnostic paper. The catalog of twenty-one publish–subscribe anti-patterns and its
empirical validation (`docs/antipatterns.md`) already exist as a mature, separately-maintained
artifact in this repository but were not previously folded into an AuSE-targeted manuscript; this
revision does so, pairing them with the existing closed-loop prescriptive-refactoring pipeline
(`docs/prescription.md`, `docs/research/ause/draft.md`) under one narrative: **detect named,
severity-tiered architectural anti-patterns → prescribe targeted, verified refactorings → gate CI/CD
on newly introduced regressions.**

> **Consistency note.** The manuscript body at [draft.md](draft.md) has been rewritten to match this
> outline's detect→prescribe→gate structure (catalog as Section 4, renumbered sections following).
> The title above also reflects the shortened, un-branded form the paper now uses — dropping the
> `SaG-Prescribe:` prefix and the `Closed-Loop, DevOps-Integrated` qualifier from the title itself,
> while the underlying framework is still referred to as SaG-Prescribe in the abstract and body text.

---

## Title

**Graph-Based Detection of Architectural Anti-Patterns and Prescriptive Refactoring in Distributed
Publish–Subscribe Systems**

---

## Abstract

Distributed publish–subscribe middleware (ROS 2, DDS, MQTT) decouples producers and consumers, but
the resulting indirect dependency structure obscures how component failures cascade, and this
architectural technical debt accumulates invisibly to code-level static analysis (SCA) tools. Unlike
object-oriented design, where mature catalogs of named anti-patterns (God Class, Feature Envy,
Shotgun Surgery) give practitioners a shared vocabulary and testable detection rules, distributed
publish–subscribe architectures have no equivalent catalog: problems are discovered reactively,
through postmortems and cascade incidents, rather than proactively at design time. Even where
structural diagnostic frameworks exist, they typically operate *open-loop*: they rank components by
criticality without naming the specific architectural pathology at fault or producing verified
guidance on how to repair it. We address both gaps with **SaG-Prescribe**, a graph-based framework
that (1) detects twenty-one named, severity-tiered publish–subscribe anti-patterns and bad smells —
including Single Point of Failure, God Component, Broker Saturation (Hub-and-Spoke), Chatty Pair, and
QoS Policy Mismatch — via formally specified topological signatures over thirteen structural metrics
and adaptive box-plot thresholds, each validated against independent failure-simulation ground truth
(Spearman $\rho = 0.876$ overall, rising to $\rho = 0.943$ at large scale; $F_1 = 0.923$, precision
$0.912$, recall $0.857$ for critical-tier classification); and (2) compiles the resulting diagnosis —
components and patterns flagged `CRITICAL` or `HIGH` — into a transformation policy $\Delta(G)$ of
three targeted graph-mutation operators (logical topic splitting, physical anti-affinity
reallocation, and transport QoS contract hardening), each re-verified by the same discrete-event
cascade simulator that produced the diagnosis before it is surfaced as a recommendation. The pipeline
is further operationalized as a delta-aware CI/CD quality gate (`detect_antipatterns.py`) that blocks
only newly introduced structural anti-patterns relative to a Git merge base, within sub-minute
execution bounds.

Across eight benchmark scenarios used for detection validation and a seven-scenario subset used for
prescriptive evaluation — spanning autonomous vehicles, IoT, finance, healthcare, and hyper-scale
enterprise topologies — the generated prescriptions reduce the System Risk Index (SRI, lower is
better) in every scenario, with relative risk reductions between 1.4% and 15.9% (Wilcoxon
signed-rank across the seven baseline/mutated pairs: $W=0.0$, $p=0.0156$, significant at
$\alpha=0.05$). We report this result alongside a less flattering but more honest one: component-level
failure-impact reductions average only +4.61% and regress in two of seven scenarios (−31.67% and
−25.36%), because the current prescriptive policy is applied greedily and unconditionally,
occasionally introducing new cascade hops that offset its own gains — a finding that directly
motivates a per-edit acceptance filter as the highest-priority extension. The full generate–verify
loop completes in under 65 seconds for six of the seven prescription scenarios (4.7 s–64.3 s) and in
approximately 10.8 minutes for the 300-process enterprise topology, while the standalone CI/CD
detection gate completes in under 2 seconds to ~40 seconds across the same scale range — making both
proactive anti-pattern detection and closed-loop prescriptive refactoring practical as
merge-request-gated or nightly pre-deployment CI/CD stages.

**Keywords:** architectural anti-patterns; bad smells; technical debt analysis; refactoring
recommendation; publish–subscribe middleware; CI/CD quality gates; DevOps; failure cascade
simulation; graph mutation; search-based software engineering

---

## Paper Outline

**1. Introduction**
1.1 Context and motivation — pub-sub decoupling as the backbone of cyber-physical/microservice/IoT
systems; decoupling obscures failure propagation and the resulting architectural technical debt.
1.2 Two open gaps: **(a)** the Architecture-Code Gap and the absence of a named, testable anti-pattern
catalog for pub-sub topologies (SCA tools are topology-blind; OO/microservices catalogs don't transfer
to broker-mediated communication); **(b)** the open-loop refactoring-recommendation gap — even
topology-aware diagnostics (including our own prior SaG framework) rank criticality without naming the
fault or verifying its remediation.
1.3 Proposed solution: a unified detect→prescribe→gate pipeline — twenty-one-pattern catalog with
empirically validated detection rules, feeding a closed-loop mutation-and-resimulation engine,
operationalized as a delta-aware CI/CD gate.
1.4 Contributions — four explicit, numbered items: (1) the empirically-validated anti-pattern
catalog and detection methodology; (2) the closed-loop prescriptive pipeline and its three operators,
explicitly mapped to the patterns they repair; (3) the CI/CD gate; (4) the multi-scenario evaluation,
including the honest component-level result.

**2. Background and Related Work**
2.1 Publish–subscribe middleware dependability (broker fault tolerance, Kafka, DDS QoS/latency
literature).
2.2 Anti-pattern and code-smell catalogs — object-oriented design (Fowler's refactoring catalog;
Brown et al.'s architectural AntiPatterns; Suryanarayana et al.'s design smells) and microservices
smells (Richardson; Taibi et al.), establishing the template this catalog follows — named pattern,
formal detection rule, refactoring strategy — while operating on system *topology* rather than code
or REST call graphs.
2.3 Refactoring recommendation and architectural technical debt — code-scope recommenders
(smell-driven, learning-based, LLM-based) vs. this paper's topology scope and verify-before-recommend
model. *(two of three prior citation slots now fillable from 2.2's real references; learning-based and
LLM-based recommender slots remain pending — no invented references.)*
2.4 Search-Based Software Engineering and architecture optimization — open-loop vs. this paper's
closed-loop verification.
2.5 Diagnostic foundation (SaG) — the heterogeneous graph model and RMAV attribution this paper
builds on; what is summarized vs. not repeated from the companion diagnostic paper.
2.6 Structural criticality analysis — why classical single-metric centrality (degree, betweenness
alone) degrades on typed pub-sub graphs relative to the composite $Q(v)$ score.

**3. System Model and Code-Quality-Augmented Quality Attribution**
3.1 Heterogeneous graph formulation — 5 node types, 6 structural edge types.
3.2 Derived `DEPENDS_ON` projection (App↔App, App↔Broker, App↔Library, Broker↔Broker) and the four
architectural layers (app, infra, mw, system).
3.3 Code Quality Penalty (CQP) — the paper's explicit bridge from SCA metrics to architecture-level
risk; feeds the Maintainability dimension below.
3.4 Multi-dimensional quality attribution (RMAV) — Reliability/Maintainability/Availability/
Vulnerability, AHP-weighted composite $Q(v)$, adaptive box-plot criticality tiers. This section is the
shared foundation consumed by both the detection catalog (§4) and the prescriptive engine (§6).

**4. A Catalog of Architectural Anti-Patterns for Publish–Subscribe Systems** *(new detection
contribution)*
4.1 Anti-patterns vs. bad smells — taxonomy and confidence distinction (structural configurations with
well-understood failure modes vs. heuristic surface symptoms requiring human judgment).
4.2 Detection methodology — thirteen structural metrics per component (PageRank, Reverse PageRank,
betweenness, closeness, eigenvector centrality, in/out-degree, clustering coefficient, articulation
point score, bridge ratio, QoS-weighted degree measures); rank-based vs. min-max normalization; adaptive
box-plot thresholds (scale-invariant, distribution-aware, statistically grounded) in place of fixed
global constants.
4.3 Catalog overview — twenty-one patterns organized into three severity tiers (`CRITICAL`, `HIGH`,
`MEDIUM`) across four RMAV dimensions; summary table of pattern, severity, primary RMAV dimension, and
formal detection rule (full specifications and remediation strategies given in the companion technical
reference, `docs/antipatterns.md`, cited rather than reproduced in full).
4.4 Representative pattern walkthroughs — five patterns selected to span severity tiers, RMAV
dimensions, and detection technique diversity: `SPOF` (articulation-point-based, Availability),
`GOD_COMPONENT` (betweenness + maintainability gate), `BROKER_OVERLOAD`/Hub-and-Spoke (the pub-sub
instantiation of a classical topology anti-pattern, validated on a deliberately-encoded scenario),
`CHATTY_PAIR` (bidirectional edge-weight product, hidden coupling behind nominal decoupling), and
`QOS_MISMATCH` (transport-contract-boundary detection unique to QoS-bearing middleware).
4.5 Empirical validation of detection — validation methodology (simulated failure impact $I(v)$ as
ground truth); headline metrics (Spearman $\rho(Q,I) = 0.876$ overall, $0.943$ at large scale;
$F_1 = 0.923$, precision $0.912$, recall $0.857$, Top-5 overlap $0.80$); pattern-specific evidence
(SPOF $F_1 > 0.95$; Hub-and-Spoke scenario confirmation); baseline comparison against single-metric
centrality; the eight-scenario validation suite, foregrounding Scenario 06 (sparse microservices mesh)
as the **precision stress test** — a well-structured topology should produce few or no findings — and
Scenario 07 as the **scalability benchmark**.

**5. Closed-Loop Optimization Objective**
Formal $\min_\Delta$ objective under a (currently unconstrained) modification budget; acceptance
criterion $\Delta\mathrm{SRI}>0$ as implemented in `PrescribeService`; explicit statement that the
candidate set for $\Delta$ is exactly the `CRITICAL`/`HIGH` components and patterns identified in §4;
note on the deferred margin-based criterion.

**6. The SaG-Prescribe Prescriptive Pipeline**
6.1 Hexagonal core abstraction (`IGraphRepository`, `Neo4jRepository` vs. `MemoryRepository`).
6.2 Pipeline stages 1–7 (diagnostic foundation and anti-pattern detection → prescriptive generation →
review interface); explicit hand-off from §4's named findings to operator selection.
6.3 Three refactoring operators — logical topic splitting, physical anti-affinity reallocation,
transport QoS contract hardening — each formalized as a typed graph mutation rule and explicitly mapped
to the anti-patterns it targets (topic splitting → `TOPIC_FANOUT`/`GOD_COMPONENT`; anti-affinity
reallocation → `SPOF`/`BROKER_OVERLOAD`/`HUB_AND_SPOKE`; QoS hardening → `QOS_MISMATCH` and
QoS-fragile `CRITICAL`/`HIGH` channels).
6.4 Closed-loop verification procedure (export → mutate → sandbox → resimulate → $\Delta$SRI), with the
independence guarantee against circular leakage.

**7. DevOps Integration and Delta-Aware CI/CD Gating**
*(This section is the paper's most direct link to the SI's CI/CD/DevOps bullet — kept prominent.)*
7.1 Automated code review architecture — `detect_antipatterns.py` surfacing §4's twenty-one detectors
as a blocking CI check.
7.2 Delta-aware regression semantics — merge-base diffing, waiver register for accepted legacy risk.
7.3 Exit-code protocol — 0 (pass) / 1 (warn) / 2 (block).

**8. Experimental Design**
8.1 Five research questions: **RQ1** detection efficacy and precision (does the catalog correlate with
ground-truth impact, and does it avoid over-flagging well-structured systems?); **RQ2** prescriptive
efficacy; **RQ3** operator contributions; **RQ4** component-level remediation efficacy; **RQ5**
computational overhead / CI-CD feasibility for both detection and prescription.
8.2 Scenario suites — eight scenarios (01–08) for detection validation, including the deterministic
"Tiny Regression" smoke-test fixture (§4.5); the seven-scenario subset (01–07, autonomous vehicle →
hyper-scale enterprise) used for prescriptive evaluation, explicitly noting and justifying the count
mismatch (the smoke-test fixture carries no domain-representative signal for prescriptive evaluation).
8.3 Experimental protocol — deterministic simulator, verified $\sigma_{\text{seed}}=0$ across 5 seeds.
8.4 Metrics — detection: Spearman $\rho$, $F_1$/precision/recall, Top-$k$ overlap; prescription: SRI
formula, $\Delta$SRI, operator counts, component-level deltas.

**9. Results**
9.1 Detection efficacy and precision (RQ1) — headline validation metrics from §4.5; Scenario 06
precision-stress confirmation (few/no findings on a well-structured topology); Scenario 07 scalability
confirmation.
9.2 Prescriptive efficacy (RQ2) — SRI improves in all 7 scenarios (1.4%–15.9%), Wilcoxon $W=0.0$,
$p=0.0156$.
9.3 Operator contributions (RQ3) — anti-affinity reallocation most frequent overall; QoS upgrades
concentrate in high-loss profiles; contribution counts cross-referenced against the anti-patterns each
operator targets (§6.3).
9.4 Remediation efficacy at component boundaries (RQ4) — **honest negative result**: mean +4.61%
component-level improvement but regressions in Hub-and-Spoke (−31.67%) and Hyper-Scale (−25.36%),
traced to greedy unconditional policy application; motivates the per-edit acceptance filter.
9.5 Computational overhead and CI/CD feasibility (RQ5) — detection gate sub-2s to ~40s across scales;
full generate–verify loop <65s for 6/7 scenarios, ~10.8 min at hyper-scale.

**10. Discussion and Threats to Validity**
10.1 What naming and verifying buys over an unnamed criticality score — §9.1 and §9.4 read together.
10.2 Positioning in CI/CD and technical-debt workflows — advisory prescriptions vs. blocking detection
gate; the catalog as an architecture-review checklist (design review by checklist, as in aviation/
surgery) for teams without full CI/CD automation.
10.3 Construct validity — detection and verification both defined relative to the same discrete-event
simulator; mitigation via operators/patterns grounded in established dependability practice.
10.4 Internal validity — greedy, unconditional policy application; independence guarantee against
circular leakage; stratified reporting discipline (per-type and per-pattern, not only pooled).
10.5 External validity — synthetic parameterized topologies; catalog scope limited to the pub-sub
communication paradigm (hybrid REST/event architectures need additional patterns); QoS-mismatch
detection currently specified for DDS/ROS 2 and MQTT weight semantics.
10.6 Conclusion validity — small scenario sample ($n=7$ for the Wilcoxon test); single-run,
verified-deterministic simulator configuration.
10.7 Engineering trade-offs of the verification step.

**11. Conclusion and Future Work**
11.1 Summary of the end-to-end detect→prescribe→gate pipeline and headline results.
11.2 Future work, in priority order: (1) per-edit acceptance filter — highest priority, directly
motivated by §9.4; (2) budget-constrained policy search; (3) stochastic cascade propagation; (4)
operator ablation and learned policy ordering; (5) catalog extension to hybrid REST/event and
non-DDS/MQTT QoS semantics; (6) real-system replication (ATM topology); (7) LLM-assisted PR generation
— kept to a single sentence, not oversold.

**References** — the merged reference list draws on the eleven previously verified prescriptive/SBSE/
pub-sub citations plus the anti-pattern catalog's own real, verified references (Fowler 1999; Brown et
al. 1998; Suryanarayana et al. 2014; Richardson 2018; Taibi et al. 2020; Baldwin & Clark 2000; Lehman
1996; Martin 2003; Nygard 2018; Colbourn 1987; Saaty 1980 — deduplicated against the existing AHP
citation). One citation slot in §2.3 (learning-based refactoring-opportunity mining) and one (LLM-based
refactoring suggestion) remain genuinely open and are **not** filled with invented references.

---

## Pre-Submission Checklist

1. **Populate the two remaining `[REF: …]` slots** in §2.3 with real learning-based and LLM-based
   refactoring-recommendation citations. The smell-driven and architectural-technical-debt slots are
   now filled from `docs/antipatterns.md`'s own verified bibliography (Fowler 1999; Suryanarayana et
   al. 2014; Brown et al. 1998) — no references have been invented for the remaining two.
2. **Literature pass:** expand from the current ~20 references (11 prior + ~9 newly merged from the
   catalog, after deduplicating Saaty) toward the ~30–45 references AuSE reviewers typically expect.
3. **Confirm companion-paper status** ([1], the Software-as-a-Graph JSS submission, and the RASSE 2025
   conference paper on graph-based critical-component identification) at submission time and disclose
   per AuSE's originality/overlap policy in the cover letter. Verify the RASSE 2025 paper's scope does
   not overlap the newly added catalog material in §4 beyond citation-level reuse of the shared RMAV/
   criticality foundation.
4. **Reconcile scenario counts explicitly in §8.2** — eight scenarios for detection validation vs.
   seven for prescriptive evaluation is intentional (per `docs/antipatterns.md` §6.3's note) but must be
   stated plainly on first mention to avoid a reviewer inconsistency flag.
5. **Do not expand the LLM future-work mention** (§11.2, item 7) beyond a single sentence — the paper
   has no LLM/AutoML contribution, and overselling it risks a scope-mismatch desk rejection.
6. Fill in **Funding**, **Competing interests**, and **Data availability** declarations before
   submission.
