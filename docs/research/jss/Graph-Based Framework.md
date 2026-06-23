---
title: "A Graph-Based Framework for Pre-Deployment Quality Attribution and Failure-Impact Analysis of Publish-Subscribe Middleware"
venue: "Journal of Systems and Software (JSS) — Elsevier, Q1"
positioning: "Framework-first. Unambiguous superset of the Middleware 2026 HGL paper: quality attribution (§4) and remediation (§6) are first-class sections the conference paper does not contain."
status: "DRAFT scaffold — numeric placeholders in [brackets] are unresolved (see §Open Items). Verify all figures against committed simulator outputs before submission."
---

# Abstract

Distributed publish–subscribe middleware decouples producers and consumers through topics and
brokers, which obscures the true dependency chains along which a single component failure can
cascade. Identifying *which* components are critical — and *why* — before deployment is hard:
runtime telemetry does not yet exist, and topology-only centrality metrics conflate distinct
failure mechanisms into a single score. We present **Software-as-a-Graph (SaG)**, a pre-deployment
framework that models a pub-sub system as a typed, weighted, directed multigraph over applications,
libraries, topics, brokers, and deployment nodes, and derives logical `DEPENDS_ON` dependencies via
a set of typed projection rules. On this model SaG performs two complementary analyses. First,
**multi-dimensional quality attribution** decomposes each component's criticality into orthogonal
Reliability, Maintainability, Availability, and Vulnerability (RMAV) dimensions, combined into a
composite score *Q(v)* with Analytic Hierarchy Process weights; because each raw metric feeds
exactly one dimension, the breakdown explains *why* a component is critical and directs targeted
remediation rather than blanket hardening. Second, **failure-impact analysis** predicts each
component's cascade impact *I(v)* using both the interpretable *Q(v)* score and a learned
heterogeneous-graph predictor, validated against discrete-event simulation under a strict
input–label independence guarantee. We further formalize a **prescriptive remediation** stage that
generates topology-level hardening edits and verifies them on counterfactual graphs via the same
simulation oracle. Across [N] synthetic scenarios and an external air-traffic-management (ATM)
case study, we show: (i) *when* interpretable attribution suffices and when learning is required;
(ii) that multi-dimensional attribution exposes failure modes invisible to centrality — notably a
**shared-library blast-radius gap**, where a component with low composite *Q* (≈ 0.48) nonetheless
drives near-total cascade impact (*I* ≈ [0.97]) through simultaneous fan-out; and (iii) agreement
between predicted rankings and expert judgment on the ATM system (Kendall τ = [τ], Fleiss κ = [κ]).
The learned predictor improves critical-component identification by ΔF1 = +0.284 over a
homogeneous-graph baseline while preserving ranking quality (ρ = 0.620). A pooled correlation of
ρ ≈ 0.08 masks per-node-type correlations of ρ = [0.63–0.90] — a Simpson's-paradox effect that
makes stratified reporting mandatory rather than optional.

**Keywords:** publish–subscribe middleware; software architecture; dependency analysis; criticality
prediction; failure cascade; quality attributes; heterogeneous graph; pre-deployment analysis.

---

# Outline

## 1. Introduction
- 1.1 Motivation: pub-sub decoupling hides dependency chains; pre-deployment is exactly when
  hardening is cheapest, yet telemetry is unavailable.
- 1.2 Problem statement: predict component criticality and failure impact from architecture alone,
  and explain it well enough to act on.
- 1.3 Limitations of existing approaches: protocol/runtime dependability assumes a running system;
  topology-only centrality gives a single opaque score that conflates failure mechanisms.
- 1.4 Contributions (framework-first):
  1. A typed multigraph model + RMAV multi-dimensional **quality attribution** that is interpretable
     by construction (each metric feeds one dimension).
  2. **Failure-impact analysis** with two predictors — interpretable *Q(v)* and learned HGL — under
     a load-bearing input–label independence guarantee.
  3. A first-class **prescriptive remediation** stage (Generate→Verify) with four operators.
  4. Two empirical findings of independent interest: the **shared-library blast-radius gap** and the
     **Simpson's-paradox stratification** of correlation by node type.
  5. **External validation** on a real-world ATM system against blind expert ranking.
- 1.5 Relationship to prior work by the authors: this paper is a superset of the HGL cascade-prediction
  study; attribution (§4) and remediation (§6) are new. *(Cover-letter disclosure point.)*
- 1.6 Paper organization.

## 2. Background and Related Work
- 2.1 Publish–subscribe middleware and dependability (decoupling; DDS/MQTT QoS; fault tolerance vs.
  pre-deployment prediction).
- 2.2 Structural criticality analysis (centrality, articulation points, PageRank; percolation /
  interdependent-network failure).
- 2.3 Learning-based critical-component prediction (FINDER, DrBC, PowerGraph) and why homogeneous
  abstractions discard pub-sub semantics.
- 2.4 Software quality attributes and multi-criteria scoring (AHP); gap: no interpretable,
  multi-dimensional, pre-deployment attribution for typed pub-sub graphs.
- 2.5 Positioning: what SaG adds over each strand.

## 3. The Software-as-a-Graph Model
- 3.1 Five node types (Application, Library, Topic, Broker, Node) and six structural edge types
  (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO, USES).
- 3.2 Derived `DEPENDS_ON` projection via the six typed rules; Rule 4 (shared-library simultaneous
  blast) and Rule 6 (broker colocation) called out — these encode the failure modes §5 exploits.
- 3.3 QoS edge-weight formula (β = 0.85; durability factor 0.40) and component weight *w(v)*.
- 3.4 Multi-layer projections (π_app, π_infra, π_mw, π_system) and the MIL-STD-498 hierarchy
  (CSU→CSC→CSCI→CSS).
- 3.5 *Running example figure:* small topology + its DEPENDS_ON projection.

## 4. Multi-Dimensional Quality Attribution  *(primary lead)*
- 4.1 The four orthogonal dimensions and their stakeholder semantics (R/M/A/V).
- 4.2 RMAV formulas (per-type variants for Topic vs. Application/Broker/Node/Library).
- 4.3 Composite *Q(v)* = 0.43·A + 0.24·R + 0.17·M + 0.16·V; AHP derivation + consistency (CR ≤ 0.10);
  λ = 0.70 weight shrinkage and its sensitivity sweep.
- 4.4 Adaptive box-plot (IQR) classifier; per-dimension + composite classification; small-sample
  percentile fallback (n < 12).
- 4.5 **Independence guarantee:** *Q(v)* computed on `G_analysis`; never reads simulation state.
- 4.6 Worked attribution: a pure SPOF (high A, low R/M/V) vs. a god-component (high M) — *why*,
  not just *that*.

## 5. Failure-Impact Analysis
- 5.1 Ground-truth impact *I(v)* = 0.35·reachability_loss + 0.25·fragmentation +
  0.25·throughput_loss + 0.15·flow_disruption, from the canonical discrete-event simulator.
- 5.2 Two predictors over the same model: interpretable *Q(v)* and learned heterogeneous-graph
  predictor (relation-specific message passing).
- 5.3 The input–label independence path (predictor features never enter the simulation ground-truth
  path) — the property that makes pre-deployment claims valid.
- 5.4 **Shared-library blast-radius gap:** low-*Q* / high-*I* component (canonical example:
  *I* ≈ [0.97] vs *Q* ≈ 0.48), structurally invisible to topology-only centrality.
- 5.5 **Simpson's-paradox stratification:** pooled ρ ≈ 0.08 vs per-type ρ = [0.63–0.90]; stratified
  reporting is required.

## 6. Prescriptive Remediation  *(promoted to first-class)*
- 6.1 Two-phase structure: **Generate** topology-only candidate edits → **Verify** on counterfactual
  *G′* via the canonical simulator oracle.
- 6.2 Four operators: RedundancyInsertion, PathDiversification, FanOutReduction, SharedTopicReduction.
- 6.3 FanOutReduction triggers on structural blast-radius signals (not *Q(v)*) — this is how it catches
  the low-*Q* library case from §5.4.
- 6.4 Acceptance criterion: ΔA > κ·σ_seed across the full propagation_threshold sweep; how κ is
  empirically derived from multi-seed variance.
- 6.5 Remediation independence: Generate never reads *I(v)*; no Verify result feeds back into Generate
  within a run.
- 6.6 Result: [X%] mean cascade-impact reduction on remediated components.

## 7. Experimental Setup
- 7.1 Datasets: [N] synthetic scenarios (domains: autonomous vehicles, HFT, healthcare, enterprise
  hub-and-spoke, IoT smart-city, cloud microservices, large-scale pub-sub) + the ATM dataset.
- 7.2 Metrics: Spearman ρ (ranking), F1 / precision / recall / Top-K / NDCG@10 (identification),
  with bootstrap 95% CIs and paired Wilcoxon tests.
- 7.3 Baselines: topology centrality (betweenness, articulation points; QoS-weighted variant) and
  homogeneous graph learning.
- 7.4 Canonical simulator configuration + reproducibility (seeds, horizon, propagation_threshold).
  *(Resolve simulator/threshold ambiguity before committing — see Open Items.)*

## 8. Results
- 8.1 **RQ1 — When does interpretable attribution suffice vs. learning?** *Q(v)* vs learned predictor
  across scenarios; where the gap is small (interpretable wins on cost/explainability) and where
  learning is required.
- 8.2 **RQ2 — What does multi-dimensional attribution expose that centrality misses?** The
  blast-radius gap (§5.4) and the stratification result (§5.5), quantified against centrality
  baselines. ΔF1 = +0.284, ρ = 0.620 for the learned predictor.
- 8.3 Ablations / sensitivity: AHP weights, λ shrinkage, propagation_threshold.

## 9. External Validation on an Air-Traffic-Management System
- 9.1 ATM system description (ICAO-compliant; double-blind — no industrial collaborator named):
  components (ConflictDetector, ASTERIX_Broker, RadarTracker, FlightDataProcessor, ATCWorkstation,
  MeteoService), 8 topics with mixed QoS down to a 100 ms deadline.
- 9.2 **RQ3 — Predicted rankings vs. expert judgment.** Blind expert ranking protocol; agreement via
  Kendall τ = [τ]; inter-rater reliability via Fleiss κ = [κ].
- 9.3 Qualitative: does the framework surface the known SPOFs (ConflictDetector Q ≈ 0.90,
  ASTERIX_Broker) and the library blast-radius case experts would otherwise miss?

## 10. Discussion, Threats to Validity, and Conclusion
- 10.1 Interpretation: interpretability-vs-accuracy boundary; remediation as the actionable payoff.
- 10.2 Threats: construct (simulator fidelity), internal (independence guarantee), external
  (generalization beyond evaluated domains; LOSO/G4 gap), and the honest QoS-encoding null result.
- 10.3 Limitations and future work (inductive generalization, richer V dimension).
- 10.4 Conclusion.

---

# Open Items (resolve before committing numbers)

1. **Canonical simulator ambiguity (HIGH).** `FaultInjector` (mean feed-loss *I*) vs `FailureSimulator`
   (four-component composite *I*). The library blast-radius headline (*I* ≈ 0.97) is producible only by
   `FailureSimulator`'s USES cascade rule. Pin the canonical simulator and confirm §5.1's composite
   formula attribution before any *I(v)* or ΔI figure is final. Affects abstract, §5.4, §8.2, §9.
2. **propagation_threshold default (HIGH).** Docstring claims 1.0 (conservative); actual default 0.2
   (aggressive). Reconcile and report the value used; it gates ΔI reproducibility and the §6.4
   acceptance sweep.
3. **Unfilled placeholders.** `[N]` scenario count; `[X%]` remediation reduction (needs §6 run);
   `[τ]`, `[κ]` (need the ATM expert-ranking study, the §9 gate); `[0.63–0.90]` per-type ρ range
   (confirm against committed stratified results); `[0.97]` library *I* (pending item 1).
4. **Middleware overlap.** Confirm whether the library blast-radius mechanism is already claimed in the
   submitted Middleware manuscript; this governs how much of §8.2 is "new" vs. cited self-work, and
   the cover-letter superset framing.
5. **RQ4 decision.** Keep remediation as an applied contribution outside the RQ structure (current
   choice), or promote to an explicit RQ4.
