# Automated, Simulation-Verified Refactoring Recommendation for Distributed Publish-Subscribe Middleware

**Target:** Automated Software Engineering (Springer) — Thematic Collection "Intelligent Techniques for Automated Code Review and Software Quality Evaluation"
**Topic:** Intelligent techniques for CI/CD, DevOps, software evolution, technical debt analysis, and refactoring recommendation
**Status:** DRAFT v0.1 — placeholders bracketed; citation slots marked `[REF: …]`; do not submit before resolving the open items in the change log.

---

## Abstract

Distributed publish-subscribe middleware (DDS, MQTT, ROS 2) decouples components at the cost of opaque failure-propagation paths: architects learn *which* components are fragile from diagnostic tools, but receive no verified guidance on *how* to restructure the topology. Existing search-based refactoring recommenders operate open-loop, proposing changes without confirming their effect on cascading-failure risk. We present SaG-Prescribe, a closed-loop refactoring-recommendation system that extends the Software-as-a-Graph diagnostic framework. It translates high-criticality diagnostics into three graph-mutation operators — splitting high-fan-out topics, generating anti-affinity placement constraints for co-located single points of failure, and hardening volatile QoS transport contracts — and re-simulates every mutated topology with a discrete-event failure simulator, accepting only recommendations that measurably improve the System Resilience Index (SRI). The pipeline runs database-free, enabling integration into CI/CD workflows as a pre-deployment quality stage. Across [seven] benchmark scenarios spanning autonomous vehicles, IoT, finance, healthcare, and hyper-scale enterprise topologies, prescribed refactorings reduced SRI in every scenario (mean absolute reduction [0.019], up to [0.067], i.e. [15.9%] of baseline risk) [Wilcoxon signed-rank, p = [p]], while completing the full recommend-verify loop in under 15 seconds for small-to-medium systems and approximately 4.5 minutes at 300-process scale.

**Keywords:** refactoring recommendation; publish-subscribe middleware; CI/CD quality gates; failure simulation; graph-based analysis; architectural technical debt

---

## 1. Introduction

### 1.1 Context and Motivation

Distributed publish-subscribe middleware frameworks — such as the Robot Operating System (ROS 2), Data Distribution Service (DDS), and MQTT — form the communication backbone of modern microservices, IoT systems, and safety-critical cyber-physical platforms. These architectures provide loose temporal, spatial, and synchronization decoupling among producers and consumers. However, this asynchronous decoupling introduces deep, non-linear structural dependencies that obscure how component-level faults propagate. Hardening these networks against cascading service disruptions requires proactive, pre-deployment optimization before configurations are committed to runtime operational fabrics.

### 1.2 The Open-Loop Refactoring-Recommendation Gap

In our companion paper [1], we introduced **Software-as-a-Graph (SaG)**, a static system analysis framework that models pub-sub topologies as native heterogeneous graphs and produces diagnostic criticality rankings $Q(v)$ and failure-impact predictions. While SaG provides high-fidelity diagnostics, it behaves as an *open-loop diagnostic engine*: it flags architectural vulnerabilities but does not synthesize concrete, actionable recommendations to resolve them.

This gap is not specific to SaG. Refactoring-recommendation research — from code-smell-driven recommenders to search-based software engineering (SBSE) — predominantly operates *open-loop*: a recommender proposes a change, and the burden of verifying that the change actually improves the quality attribute of interest falls back on the engineer, typically after the change has already entered the development pipeline. For architectural refactorings of distributed topologies, where the quality attribute is resistance to cascading failure, unverified recommendations are particularly hazardous: a topology edit that looks beneficial locally can degrade global resilience.

We refer to this as the **open-loop refactoring-recommendation gap**: architects know *which* components are fragile but lack automated recommendations on *how* to restructure the topology whose effect on cascading risk has been *verified before* the recommendation is surfaced.

### 1.3 Proposed Solution: SaG-Prescribe

To close this gap, we present **SaG-Prescribe**, a closed-loop refactoring-recommendation system extending [1]. SaG-Prescribe consumes the diagnostic output of SaG — components flagged `CRITICAL` or `HIGH` via adaptive box-plot fences — and feeds them to a rule-based prescriptive engine that generates targeted architectural mutations along three vectors:

1. **Logical Subgraph Refactoring:** splitting monolithic, high-fan-out topics into per-publisher sub-topics;
2. **Physical Locality Anti-Affinity:** restructuring deployment maps to isolate co-located Single Points of Failure (SPOFs);
3. **Middleware Transport Contract Hardening:** upgrading Quality-of-Service (QoS) parameters from volatile/best-effort to reliable/transient-local settings.

Crucially, SaG-Prescribe implements **closed-loop simulation verification**: each mutated candidate topology $G'$ is programmatically re-simulated with SaG's discrete-event failure simulator, and a recommendation is accepted only if it improves the System Resilience Index (SRI). Because the entire recommend-verify loop runs against an in-memory repository with no database dependency, it is deployable as a pre-deployment quality stage inside CI/CD pipelines.

### 1.4 Contributions

1. **A closed-loop refactoring-recommendation pipeline** that automatically translates topological criticality diagnostics into counterfactual graph mutations and verifies each recommendation against a discrete-event cascade simulator before surfacing it.
2. **Three pub-sub refactoring operators** formulated as concrete graph mutations tailored to logical topic congestion, physical hosting SPOFs, and transport QoS fragility.
3. **Multi-scenario simulation verification** demonstrating consistent, simulation-verified SRI improvements across [seven] realistic pub-sub scenarios [pending Wilcoxon: "with statistically significant gains (p = [p])"], with runtimes compatible with CI/CD budgets.

The remainder of the paper is organized as follows. Section 2 positions the work within refactoring recommendation, architectural technical debt, and SBSE. Section 3 formalizes the problem. Section 4 presents the pipeline and operators. Sections 5 and 6 describe the experimental design and results. Section 7 discusses implications for CI/CD workflows, Section 8 examines threats to validity, and Section 9 concludes.

---

## 2. Background and Related Work

### 2.1 Publish-Subscribe Middleware Dependability

Dependability research for message-oriented middleware historically centers on protocol verification, fault-tolerant replication patterns, network traffic load balancing, and contract verification at runtime [REF: middleware dependability surveys]. While useful for mitigation post-failure, these approaches treat topologies as fixed inputs, rather than open parameters that can be statically optimized before system delivery.

### 2.2 Refactoring Recommendation and Architectural Technical Debt

Automated refactoring recommendation has been studied extensively at code level: smell-driven recommenders detect structural anomalies (god classes, feature envy) and propose remediation transformations [REF: code-smell refactoring recommenders]; learning-based recommenders mine refactoring histories to predict refactoring opportunities [REF: ML-based refactoring prediction]; and modern approaches employ large language models for refactoring suggestion and explanation [REF: LLM-based refactoring]. At the architectural level, technical-debt research quantifies the cost of structural decay and proposes prioritized remediation plans [REF: architectural technical debt management].

SaG-Prescribe differs from this body of work along two axes. First, its **scope** is the deployed system topology — applications, brokers, topics, hosts, and their QoS contracts — rather than source code within a module boundary; the anomalies it remediates (SPOF hosts, congested topic hubs, fragile transport contracts) have no file-level analog and are invisible to code-scope recommenders. Second, its **verification model** is closed-loop: whereas code-level recommenders validate suggestions against static quality metrics or historical acceptance data, SaG-Prescribe re-simulates every candidate topology against a cascade failure model and surfaces only recommendations with verified resilience gains. In technical-debt terms, the framework identifies architectural debt items (anti-patterns with quantified risk), proposes repayments (mutation operators), and verifies the repayment's effect before it is recommended — a verify-before-recommend discipline that, to our knowledge, has not been applied to pub-sub topology refactoring.

### 2.3 Open-Loop vs. Closed-Loop Search-Based Software Engineering

Search-Based Software Engineering applies heuristic search algorithms to discover architectural refactoring blueprints [REF: SBSE surveys]. However, classical search-based methods often operate in an open-loop fashion, reporting recommendations to users without verifying their operational efficacy in a simulated cascade model. SaG-Prescribe combines the multi-dimensional diagnostics of Software-as-a-Graph [1] with closed-loop simulation verification to bridge this gap, ensuring that every recommended edit is verified to improve the System Resilience Index before acceptance.

### 2.4 Diagnostic Foundation

We rely on the heterogeneous graph representation, multi-dimensional quality attribution (RMAV), and discrete-event failure simulator of Software-as-a-Graph [1] to establish our baseline diagnostic mapping. SaG-Prescribe builds directly upon these ports, extending the domain service to close the loop between diagnostic ranking and prescriptive mutation. The diagnostic stages themselves — including the learned criticality predictors — are contributions of the companion work and are outside the scope of this paper.

### 2.5 Structural Criticality Analysis

Graph-theoretic approaches offer mathematical constructs such as betweenness centrality, PageRank, closeness metrics, and articulation-point tests to pinpoint critical bridges [REF: structural criticality analysis]. These identify fragility but, like the diagnostic layer of SaG, stop short of prescribing and verifying remediation — the step this paper automates.

---

## 3. Problem Formulation

### 3.1 System Model

A distributed publish-subscribe system is modeled as a typed, weighted, directed multigraph $G = (V, E, \tau_V, \tau_E, w)$, where the vertex set partitions into five component types — Application, Broker, Topic, Node (host), and Library — and edges carry six structural types imported from the architecture description (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`) plus derived `DEPENDS_ON` dependencies. Edge weights encode QoS-derived coupling strength. The full model, including QoS weight derivation and dependency-derivation rules, is defined in the companion paper [1]; here we summarize only what the prescriptive engine consumes.

### 3.2 Diagnostic Input

The prescriptive engine consumes two diagnostic products of the SaG pipeline [1]:

- **Criticality tiers.** Composite criticality scores $Q(v)$ are mapped to five tiers (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `MINIMAL`) using adaptive box-plot thresholding on the system's own score distribution (`CRITICAL`: $Q > Q_3 + 1.5\,\mathrm{IQR}$; `HIGH`: $Q_3 < Q \le$ upper fence). Components in the top two tiers are remediation candidates.
- **Anti-pattern flags.** Structural smells detected by the anti-pattern analyzer (e.g., SPOF hosts, congested topic hubs) supplement the tier filter as remediation triggers.

### 3.3 Closed-Loop Optimization Objective

The prescriptive task is to compute a transformation policy $\Delta$ such that $G' = \Delta(G)$ minimizes the downstream failure-vulnerability profile computed over the mutated topology, subject to a modification budget:

$$\min_{\Delta} \sum_{v \in V} I^*_{\Delta(G)}(v) \quad \text{subject to} \quad \mathrm{Cost}(\Delta) \le \mathcal{B}$$

where $I^*(v)$ denotes the simulated failure impact of component $v$. In the implemented system, the aggregate objective is operationalized through the System Resilience Index (§5.4), and a policy is **accepted** if and only if its verified improvement satisfies the acceptance criterion [DECISION PENDING: $\Delta\mathrm{SRI} > 0$ as implemented, vs. $\Delta A > \kappa\,\sigma_{\text{seed}}$ per formal documentation — state one canonically here and reconcile the other in a single sentence].

---

## 4. The SaG-Prescribe Pipeline

### 4.1 Architectural Foundation

The system uses a decoupled hexagonal (ports-and-adapters) design that separates domain orchestration from database infrastructure. Persistence services implement the `IGraphRepository` interface port: production networks run a Bolt-driven Neo4j adapter, while the prescriptive verification loop uses an isolated, thread-safe in-memory repository (`MemoryRepository`) requiring no database instance. This design choice is what makes the closed-loop verification cheap enough for CI/CD integration (§6.3).

### 4.2 Pipeline Stages

SaG-Prescribe extends the diagnostic pipeline of SaG [1] with a generation-verification loop:

- **Stages 1–5: Diagnostic foundation [1].** The pipeline ingests JSON/YAML configuration representations of pub-sub topologies, computes multi-layered topological metrics, attributes component criticality, models failure cascades with a discrete-event simulator, and validates predictive alignment against simulation ground truth. For the full treatment of these stages we refer the reader to the companion paper [1].
- **Stage 6: Prescriptive recommendation generation (this paper).** The prescriptive layer processes elements categorized `CRITICAL` or `HIGH`, evaluating structural anomalies with rule-based mappings to compile an optimization policy $\Delta(G)$ across the three operators of §4.3, then verifies the policy in the closed loop of §4.4.
- **Stage 7: Review interface.** The system serializes baseline metrics, detected flaws, prescribed modifications, and verification deltas; a web dashboard (SMART) renders side-by-side interactive topologies so architects can review and approve modifications before code commitment. Recommendations remain advisory: the human architect is the final authority (§7.2).

### 4.3 Refactoring Operators

**Operator 1 — Logical topic splitting.** A central topic hub connected to multiple publishers and subscribers is a logical bottleneck and a high-risk failure propagator. For a congested topic $T$: for each publisher $P_i$ publishing to $T$, create a dedicated sub-topic $T_{P_i}$ and re-route $P_i \rightarrow T_{P_i}$; re-route each subscriber of $T$ to the set of sub-topics; duplicate broker routing links accordingly. This bounds failure propagation by separating independent logical communication channels.

**Operator 2 — Physical anti-affinity reallocation.** Multiple processes co-located on a single physical host flagged as SPOF or critical fail simultaneously if the host fails. For each critical host $N$ hosting multiple processes: allocate separate node instances $N_{C_i}$ for each co-located process $C_i$ beyond the first, update `RUNS_ON` relationships, and duplicate `CONNECTS_TO` links to preserve network reachability. The output is a set of anti-affinity scheduling constraints directly translatable to container-orchestration placement rules.

**Operator 3 — Transport QoS contract hardening.** Critical channels using volatile transport configurations (e.g., `BEST_EFFORT` reliability, `VOLATILE` durability) are fragile under loss. For any topic that is `CRITICAL`/`HIGH` or connects to a critical component with such a configuration, the operator upgrades the contract to `RELIABLE` reliability and `TRANSIENT` durability.

### 4.4 Closed-Loop Verification

Once the policy $\Delta(G)$ is compiled, the verification engine executes: (1) export the source graph to a flat JSON schema; (2) apply the compiled splits, reallocations, and QoS upgrades; (3) seed a temporary `MemoryRepository` with the mutated JSON and derive its dependency edges; (4) run the full analysis-simulation-validation suite on the sandbox; (5) compute $\Delta\mathrm{SRI} = \mathrm{SRI}_{\text{baseline}} - \mathrm{SRI}_{\text{mutated}}$ under identical fault scenarios and seeds. Only policies meeting the acceptance criterion (§3.3) are surfaced as recommendations.

---

## 5. Experimental Design

### 5.1 Research Questions

- **RQ1 (Prescriptive efficacy):** Does the closed-loop engine improve the System Resilience Index across scenarios[, and is the improvement statistically significant]? [Bracketed clause included only if the Wilcoxon test is run before submission.]
- **RQ2 (Operator contributions):** How do the individual refactoring operators contribute to global resilience improvements?
- **RQ3 (Computational overhead and scalability):** What is the execution time of the closed-loop pipeline as system scale grows, and is it compatible with CI/CD budgets?

### 5.2 Benchmark Scenarios

The evaluation suite comprises [seven] parameterized publish-subscribe topologies spanning realistic domain verticals and scale presets [pending 7-vs-8 reconciliation with the companion manuscript]:

1. **Autonomous Vehicle System** — medium-scale ROS 2 network, streaming sensor topologies, reliable/transient-local QoS profiles.
2. **IoT Smart City System** — large-scale network modeling high-loss, volatile, best-effort endpoints.
3. **Financial Trading System** — high-density network with time-critical message loops and strict persistent priority settings.
4. **Healthcare Integration System** — dense network with centralized real-time patient-monitoring fan-outs and long durability horizons.
5. **Hub-and-Spoke Architecture** — topology built around a structural anti-pattern bottleneck (centralized broker pair) for SPOF tracking.
6. **Microservices Mesh** — sparse, cloud-native cluster with low coupling, auditing prediction boundary stability.
7. **Hyper-Scale Enterprise Architecture** — 300 distinct execution processes, probing runtime performance boundaries.

### 5.3 Configuration

The prescriptive engine runs in-memory over each scenario topology. All evaluations use five random seeds ({42, 43, 44, 45, 46}) for discrete-event cascade simulation, with results averaged; the cascade propagation threshold is 0.2. [DISCLOSURE PENDING: these seeds differ from the canonical seed set {42, 123, 456, 789, 2024} used in the companion manuscript's validation experiments — state whether this constitutes an independent replication set or align the sets.]

### 5.4 Metrics

**System Resilience Index (SRI).** The primary outcome measure is a composite risk index over the four RMAV health dimensions of the diagnostic framework [1]:

$$\mathrm{SRI} = 0.25\,(1 - H_R) + 0.25\,(1 - H_M) + 0.25\,(1 - H_A) + 0.25\,(1 - H_V)$$

where $H_R, H_M, H_A, H_V$ are the system-level Reliability, Maintainability, Availability, and Vulnerability health scores. Lower SRI indicates lower composite structural risk. [NAMING: internal documentation titles this metric "System Risk Index," which matches its lower-is-better semantics; this manuscript and the companion draft use "System Resilience Index." Standardize before submission.] We report $\Delta\mathrm{SRI} = \mathrm{SRI}_{\text{baseline}} - \mathrm{SRI}_{\text{mutated}}$ (positive = improvement), alongside per-operator recommendation counts: topic splits, node reallocations, and QoS upgrades.

---

## 6. Results

### 6.1 Prescriptive Efficacy (RQ1)

Table 1 reports baseline and mutated SRI per scenario. In all [seven] scenarios, applying the prescribed refactorings reduces composite risk. [If Wilcoxon run: "A paired Wilcoxon signed-rank test over per-scenario deltas confirms the improvement is statistically significant (W = [W], p = [p])." If not run: "The reduction is consistent across all scenarios and all five simulation seeds."]

**Table 1 — Prescriptive optimization results across scenarios.**

| Scenario | Baseline SRI | Mutated SRI | ΔSRI | Splits | Reallocs | Upgrades |
|----------|:------------:|:-----------:|:----:|:------:|:--------:|:--------:|
| 01 Autonomous Vehicle | 0.3645 | 0.3535 | +0.0110 | 35 | 121 | 0 |
| 02 IoT Smart City | 0.4206 | 0.3537 | +0.0669 | 58 | 276 | 51 |
| 03 Financial Trading | 0.3675 | 0.3482 | +0.0193 | 31 | 88 | 6 |
| 04 Healthcare | 0.3809 | 0.3757 | +0.0052 | 19 | 74 | 6 |
| 05 Hub-and-Spoke | 0.3595 | 0.3527 | +0.0068 | 30 | 97 | 0 |
| 06 Microservices Mesh | 0.3612 | 0.3542 | +0.0070 | 40 | 123 | 0 |
| 07 Hyper-Scale Enterprise | 0.3614 | 0.3469 | +0.0145 | 119 | 409 | 0 |

The largest improvement occurs in Scenario 02 (IoT Smart City), where SRI improves by +0.0669 — [15.9%] of baseline risk — driven by QoS contract upgrades that stabilize high-loss best-effort links combined with anti-affinity constraints that partition key microservices. In Scenario 05 (Hub-and-Spoke), topic splitting mitigates single points of failure at the broker level, improving SRI from 0.3595 to 0.3527. [All Table 1 values pend re-verification under the pinned configuration before unbracketing the abstract aggregates.]

### 6.2 Operator Contributions (RQ2)

The contribution of each operator depends on the scenario's topological features:

- **Logical topic splits.** In Scenario 07 (Hyper-Scale Enterprise), the engine generates 119 splits to alleviate congestion in heavily shared, centralized topics, confining data feeds to target subscribers and reducing structural blast radius.
- **Physical node reallocations.** Reallocation is the most frequently recommended mutation (409 in Scenario 07, 276 in Scenario 02), establishing anti-affinity constraints that prevent co-locating safety-critical processes on a single host.
- **Transport QoS upgrades.** Upgrades concentrate in high-loss profiles: Scenario 02 (IoT Smart City) receives 51 upgrades hardening volatile/best-effort channels to reliable, transient-local contracts.

[OPTIONAL STRENGTHENING, not blocking: per-operator ablation — each operator disabled in turn — would isolate causal contribution rather than inferring it from counts; flag as future work if not run.]

### 6.3 Computational Overhead and Scalability (RQ3)

For small-to-medium scenarios (01–06), the entire analysis-generation-verification loop completes in **under 15 seconds** per scenario. For Scenario 07 (300 processes), the closed-loop discrete-event simulation over five seeds takes approximately **4.5 minutes**. Both figures fit comfortably within typical CI/CD stage budgets — comparable to a standard static-analysis or integration-test stage — confirming that the database-free `MemoryRepository` makes continuous closed-loop recommendation feasible as a pipeline step rather than an offline batch process.

---

## 7. Discussion

### 7.1 What Closed-Loop Verification Buys

The central result is not that rule-based operators can improve a resilience metric — well-chosen remediations of known anti-patterns are expected to help — but that each recommendation is *verified* against a cascade model before it reaches the architect. This converts refactoring recommendation from a suggestion service into a quality-evaluation instrument: the recommendation arrives with quantified, simulator-confirmed evidence of its effect ($\Delta\mathrm{SRI}$ per policy), and policies that fail verification are silently discarded rather than surfaced as noise. For distributed topologies, where local intuition about global cascade behavior is unreliable, this verify-before-recommend discipline is the difference between advice and evidence.

### 7.2 Positioning in CI/CD and Technical-Debt Workflows

Two claim boundaries govern responsible deployment of the framework in CI/CD pipelines. First, **prescriptive recommendations are advisory**: the pipeline surfaces verified refactoring blueprints for architect review (Stage 7), but does not auto-apply mutations, since verified-in-simulation does not entail correct-in-production. Second, following the delta-aware gating semantics of the companion framework [1], **blocking-gate claims are reserved for structural regression detection** — new, unwaived anti-patterns introduced relative to the merge base — while composite criticality rankings and prescriptive recommendations inform but do not block. This division mirrors the "Clean as You Code" discipline familiar from code-scope quality platforms, applied at topology scope: the gate blocks new architectural debt, and the recommender proposes verified repayments of existing debt. Because the recommend-verify loop runs within CI time budgets (§6.3), repayment proposals can be regenerated on every merge request, keeping the architectural-debt register synchronized with the evolving topology.

### 7.3 Trade-offs

The closed-loop simulation verification steps introduce computational overhead compared to surfacing unverified structural recommendations. Since the framework targets pre-deployment optimization rather than runtime alerting, this cost is an acceptable trade-off for verified reliability improvements: the expensive path (full re-simulation) runs once per candidate policy, not per message or per deployment.

---

## 8. Threats to Validity

### 8.1 Construct and Internal Validity

Ground-truth impact is produced by a discrete-event cascade simulator rather than observed in deployed systems; a verified $\Delta\mathrm{SRI}$ therefore demonstrates improvement *with respect to the simulator's cascade rules*, not directly with respect to production incident rates. All baseline and mutated topologies are evaluated against identical fault scenarios, seeds, and simulator configuration, ensuring internal consistency of the comparison. Because shared libraries behave as passive elements during cascade propagation, pooled metrics can mask type-specific signals (a Simpson's-paradox effect); the diagnostic layer mitigates this with stratified, type-level reporting [1].

### 8.2 External Validity

Experiments use [seven] parameterized, synthetically generated scenarios. While the vertical presets mimic representative middleware distributions, they may not capture runtime complexities of industrial deployments — dynamic workload shifts, network loss spikes, transient hardware faults. Validation against an industrial topology remains future work.

### 8.3 Conclusion Validity

The scenario sample is small (n = [7]), so we [report exact Wilcoxon signed-rank statistics with effect sizes / refrain from significance claims and report per-scenario, per-seed consistency instead — resolve with the Wilcoxon decision]. Results are averaged over five simulation seeds; seed-level variance is [reported in the replication package / to be added].

---

## 9. Conclusion and Future Work

This paper presented SaG-Prescribe, a closed-loop refactoring-recommendation system for distributed publish-subscribe architectures. The framework translates graph-based criticality diagnostics into three concrete refactoring operators — topic splitting, anti-affinity reallocation, and QoS contract hardening — and verifies every candidate policy against a discrete-event cascade simulator before recommending it, achieving consistent SRI improvements across [seven] benchmark scenarios at CI/CD-compatible runtimes.

Future work proceeds along three lines: (1) integrating operational telemetry (throughput latency, resource throttling, packet loss) into the feature space to enable continuous re-optimization; (2) extending the prescriptive search with explicit cost, budget, and resource constraints to balance reliability against footprint; (3) per-operator ablation studies and validation against industrial topologies to strengthen causal attribution and external validity.

---

## Declarations

- **Funding:** [to be completed]
- **Competing interests:** [to be completed]
- **Data availability:** A replication package containing scenario configurations, seeds, and the prescriptive pipeline implementation will be made available at [URL pending]. [Confirm what can be released given the Middleware 2026 double-blind status at submission time.]
- **Ethics approval:** Not applicable.

## References

[1] [Companion JSS manuscript — Software-as-a-Graph: A Static System Analysis Framework for Pre-Deployment Quality Gating and Failure Simulation of Publish-Subscribe Middleware. Cite per its submission status at AuSE submission time.]
[REF: …] — placeholder slots marked inline in §2; populate from your bibliography. No citations invented.

---
---

# Change Log (not part of the manuscript)

**Basis:** `docs/research/ase/Closed-Loop Prescriptive Architecture Optimization.md`, restructured per the approved outline (`ause_si_abstract_outline.md`).

1. **HIGH — "statistically significant" removed from §6.1 body and RQ1** and replaced with conditional Wilcoxon/consistency phrasing. The original draft asserts significance with no test reported. Owner action: run paired Wilcoxon; expected exact two-sided p = 0.0156 if all 7 deltas positive.
2. **HIGH — acceptance criterion left as explicit DECISION PENDING in §3.3** (ΔSRI > 0 as implemented vs. ΔA > κ·σ_seed in formal docs). One canonical statement required before submission.
3. **HIGH — scenario count bracketed [seven] throughout** pending 7-vs-8 reconciliation with JSS manuscript.
4. **MEDIUM — seed-set disclosure inserted in §5.3** ({42–46} vs. JSS canonical {42, 123, 456, 789, 2024}); decide framing.
5. **MEDIUM — SRI naming discrepancy flagged in §5.4**: `docs/validation.md` defines "System **Risk** Index" (semantics match lower-is-better); ASE draft uses "System **Resilience** Index." Standardize across manuscript, companion, and docs.
6. **MEDIUM — new §2.2 (refactoring recommendation & architectural technical debt)** written with `[REF]` placeholders only; no citations invented. Owner action: populate from bibliography.
7. **MEDIUM — HGT/feature-tensor detail removed from problem formulation** (original §3.3): the learned diagnostic layer is attributed to the companion papers; this paper's §3.2 now states only what the prescriptive engine consumes (tiers + anti-pattern flags). Keeps the contribution boundary clean and consistent with the JSS cover-letter boundary map.
8. **LOW — title, abstract, keywords replaced** per approved versions; abstract aggregates ([0.019], [0.067], [15.9%]) computed from committed Table 6.1 values, bracketed pending pinned-config re-verification.
9. **LOW — Threats to Validity promoted to standalone §8** with new conclusion-validity subsection; discussion §7.2 (CI/CD/technical-debt positioning) is new, consistent with the delta-aware gating and advisory-vs-blocking claim boundaries established for the JSS manuscript.
10. **LOW — operator descriptions in §4.3 aligned to `docs/prescription.md` §2 semantics** (exact rewiring steps) rather than the draft's shorter summaries.