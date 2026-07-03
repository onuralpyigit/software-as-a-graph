# Heterogeneous Graph Learning for Pre-Deployment Reliability Analysis of Publish-Subscribe Middleware

*Target venue: Journal of Systems and Software — Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems" (VSI:AI4MSS). Submission deadline: 30 September 2026.*

> **Draft status (v0.1, working draft — NOT submission-ready).**
> This is a journal extension of the Middleware 2026 conference submission
> ("Heterogeneous Graph Learning for Cascade Impact Prediction in Distributed
> Publish-Subscribe Middleware", currently under double-blind review).
> **Submission precondition:** Middleware acceptance (or withdrawal) must occur
> before this manuscript can be submitted; the SI requires disclosure of the
> prior version, a delta document, and ≥30% new technical content.
>
> **Open decisions (must be resolved before submission):**
> - **D1 [HIGH]** — Confirm the flagship SaG paper returns to the JSS *regular track*; two JSS submissions must have disjoint contributions and mutual cover-letter disclosure.
> - **D2 [HIGH]** — Canonical scenario count: conference used 7; repository defines 11 (incl. Tiny/XLarge/ATM/Redundancy). Decide whether the journal suite is 7 (unchanged), or expanded (e.g., +Broker-Redundancy, +Tiny), and reconcile the 7-vs-8 discrepancy noted across the JSS/ASE manuscripts.
> - **D3 [MEDIUM]** — Include or drop §9 (ATM case study), depending on overlap risk with the flagship's §9 expert-ranking validation. If retained, §9 here must be scoped to HGL predictive evaluation only.
> - **D4 [LOW]** — Canonicalize architecture naming: conference text says "heterogeneous GAT"; implementation docs describe a 3-layer EdgeAwareHGTConv (Heterogeneous Graph Transformer). One name must be used consistently and match the code.
>
> **Placeholder registry:** all `[bracketed]` values are unmeasured. No placeholder
> may be filled without a committed experimental run. Sections marked **[NEW]**
> contain journal-only content; **[EXT]** = extended; **[CONF]** = carried from
> the conference version (numbers committed).

---

## Abstract

Distributed publish-subscribe middleware decouples producers and consumers through topics, brokers, and QoS contracts, making failure cascades difficult to predict from the architecture alone — precisely when prediction matters most: before deployment, when no runtime telemetry exists. We present a heterogeneous graph learning (HGL) framework that predicts component-level cascade impact directly from the system's native, typed architecture graph spanning Applications, Libraries, Topics, Brokers, and deployment Nodes, using relation-specific message passing with explicit QoS edge attribution. To avoid circular validation, ground-truth impact is obtained from discrete-event simulation under a strict input–label independence guarantee. Across a controlled 2×3 factorial evaluation on [seven | D2] representative pub-sub scenarios, HGL achieves the strongest mean ranking correlation (Spearman ρ = 0.620) and critical-component identification (F1 = 0.765, ΔF1 = +0.284 over homogeneous graph learning), demonstrating that typed heterogeneity — not explicit QoS features — is the primary driver of in-distribution accuracy, while QoS attribution is decisive for out-of-distribution generalization under Leave-One-Scenario-Out validation (mean ρ = 0.4009 for HGL-QoS vs. 0.0208 for the homogeneous baseline). Extending our prior conference study, this article contributes: (i) infrastructure-tier prediction targets, extending simulation labels to Broker and deployment-Node failures; (ii) an interpretability analysis of learned attention weights, exposing which typed relations carry cascade signal; (iii) seed-level statistical testing of all pairwise variant comparisons ([Wilcoxon results]); and (iv) [an evaluation on a real-world ICAO-compliant air-traffic-management system | D3]. The results position typed heterogeneous graph learning as a practical AI technique for pre-deployment reliability and dependability analysis of complex ICT systems.

**Keywords:** publish-subscribe middleware; heterogeneous graph neural networks; reliability analysis; dependability; cascade failure prediction; pre-deployment verification; explainable AI

---

## 1. Introduction [EXT]

### 1.1 Motivation

Publish-subscribe (pub-sub) middleware — DDS, MQTT, ROS 2 — is the communication backbone of cyber-physical, IoT, financial, and safety-critical systems. Its defining virtue, the decoupling of producers and consumers in time, space, and synchronization, is also its defining hazard for dependability analysis: the paths along which a single component failure cascades through topics, brokers, shared libraries, and co-located hosts are not visible in any single component's interface. When a system is safety-critical — an air-traffic-management deployment, a clinical monitoring network — the question *"which components, if they fail, take the system down?"* must be answered **before** deployment, at a point in the lifecycle where no runtime telemetry exists to answer it empirically.

### 1.2 Problem Statement

We study pre-deployment critical-component identification: given only the system's architectural model — its applications, shared libraries, topics, brokers, deployment nodes, and QoS contracts — predict, for each component *v*, a criticality score *Q\*(v)* that recovers the ordering of the true cascade impact *I\*(v)* that a failure of *v* would produce at runtime. Because runtime data is unavailable by construction, the problem is one of **learning under label scarcity with a hard input-side constraint**: predictive inputs must be derivable from the static architecture alone, and ground-truth labels must come from a source that shares no features with those inputs, or the pre-deployment claim collapses into circular validation.

### 1.3 Gap

Three strands of prior work each leave part of this problem open. Runtime dependability techniques (replication, reliable dissemination, chaos engineering) assume an observable running system and act too late for design-time hardening. Structural centrality metrics (betweenness, articulation points) are interpretable and telemetry-free, but collapse a component's risk into a single scalar computed on a type-collapsed projection, conflating distinct failure mechanisms. Homogeneous graph neural networks learn richer functions of topology but likewise collapse the typed semantics — Application vs. Topic vs. Broker, publication vs. library-usage vs. host-deployment — that govern how failures actually propagate in pub-sub systems, leading to representation collapse on these sparse, hub-dominated graphs. No prior approach learns cascade impact **on the native typed architecture graph, under an explicit input–label independence guarantee, with validated out-of-distribution behavior** — the combination that pre-deployment reliability analysis requires.

### 1.4 Approach

We formulate pre-deployment critical-component identification as a **heterogeneous graph learning (HGL)** problem. The system is modeled as a typed, weighted, directed multigraph over five node types (Application, Library, Topic, Broker, Node) and seven edge types; each pub-sub edge carries a 7-dimensional QoS attribute vector (reliability, durability, transport priority, deadline, blocking, heterogeneity). A relation-specific attention model learns a dedicated message function per typed relation, so that publication, subscription, routing, library usage, and host deployment each contribute distinct propagation semantics. Ground truth is produced by a discrete-event failure simulator that injects component failures and measures downstream message-flow degradation; simulator outputs never enter the model's input features (the independence guarantee). A controlled 2×3 factorial design — {structural, homogeneous GNN, heterogeneous GNN} × {QoS-masked, QoS-aware} — isolates exactly which design decision buys which capability.

### 1.5 Contributions

Contributions carried from the conference version [CONF]:

1. A formulation of pre-deployment critical-component identification as heterogeneous graph learning over the native typed architecture graph, with QoS attributes as first-class edge features.
2. Empirical evidence that typed heterogeneity is the principal driver of predictive accuracy: mean Spearman ρ = 0.620 and F1 = 0.765, a ΔF1 of +0.284 over an equivalently trained homogeneous baseline.
3. The finding that explicit QoS encoding trades a small in-distribution penalty for decisive out-of-distribution generalization (LOSO mean ρ = 0.4009 vs. 0.3073 QoS-masked and 0.0208 homogeneous), and an explanation of this trade-off in terms of scenario-transferable vs. scenario-specific signal.

New contributions in this article [NEW]:

4. **Infrastructure-tier prediction targets** (§4, §7): we extend the simulation oracle to attribute cascade impact to Broker and deployment-Node failures, converting two tiers that were previously contextual embedding layers into validated prediction targets — closing the principal limitation acknowledged in the conference version.
5. **Interpretability analysis** (§8): we analyze learned attention weights across typed relations to expose *which* architectural semantics carry cascade signal, addressing the black-box objection that limits the actionability of GNN-based reliability predictors.
6. **Seed-level statistical testing** (§5.4, §6): paired Wilcoxon signed-rank tests across all variant comparisons, with [p-values, effect sizes].
7. [**Real-world case study** (§9): HGL predictive evaluation on an ICAO-compliant air-traffic-management system. | D3]

### 1.6 Relationship to the Conference Version

This article extends our Middleware 2026 paper [Anon-B — replace with full citation upon acceptance]. The conference version established the HGL formulation, the 2×3 factorial evaluation on application-level targets, and the LOSO generalization result. Sections 4, 7, 8 [, 9 | D3], the extended statistical protocol in §5, and the expanded related-work treatment in §2 are new; we estimate [40–50]% new technical content. A detailed delta document accompanies this submission per SI requirements.

> **[BLOCKER — do not submit before resolving]** The conference paper is under
> double-blind review. All self-citations remain anonymized ([Anon-A], [Anon-B])
> until acceptance; this section and the cover letter must be rewritten with full
> citations and the required prior-version PDF at submission time.

### 1.7 Organization

§2 reviews related work. §3 defines the system model and HGL architecture. §4 introduces the infrastructure-tier extension of the simulation oracle. §5 details the evaluation methodology. §6 reports application-level results; §7 reports infrastructure-tier results; §8 presents the interpretability analysis. [§9 presents the ATM case study. | D3] §10 discusses threats to validity, and §11 concludes.

---

## 2. Related Work [EXT]

### 2.1 Publish-Subscribe Dependability

The pub-sub paradigm decouples producers and consumers through topics and brokered overlays; standards such as DDS and MQTT bind deployment-time QoS contracts (reliability, durability, deadlines) that govern runtime behavior. Dependability research in this space has concentrated on *runtime* mechanisms — reliable event dissemination, replication, broker failover, and chaos-engineering-style fault injection. These techniques presuppose a running, observable system; our concern is complementary and earlier: estimating from the architectural model alone which components most warrant hardening before any system exists to observe.

### 2.2 Structural Criticality Analysis

Classical network-science metrics — betweenness centrality, articulation points, k-core — identify structurally important vertices in O(V·E) time and are fully interpretable. Applied to software dependency graphs, they provide useful pre-deployment signals; our own prior structural baseline achieved strong topology-impact correlation on single fixed systems [Anon-A]. Their limitation is representational: computed on type-collapsed projections with uniform edge semantics, they cannot distinguish a broker SPOF from a shared-library fan-out from a high-priority publisher, and they degrade sharply when transferred across topologies (§6.4).

### 2.3 Graph Learning for Critical-Node Identification

Learning-based approaches (FINDER, DrBC, PowerGraph) train GNNs to identify critical nodes or predict cascading failures, primarily on homogeneous graphs from network science and power systems. Directly adapting them to middleware requires collapsing the typed architecture into a single-type dependency projection — exactly the reduction that discards pub-sub semantics. Heterogeneous GNN architectures (RGCN, HAN, HGT, MAGNN) provide relation-specific message passing but have not previously been formulated, implemented, and validated for pre-deployment cascade prediction in pub-sub middleware. Our contribution is that middleware-specific formulation and its controlled evaluation, not a new generic architecture.

### 2.4 AI for Reliability and Dependability of ICT Systems [NEW]

[EXPAND for SI audience: position against recent AI-for-dependability work — learned failure prediction in microservice meshes, GNNs for root-cause analysis and anomaly detection in distributed traces, ML-assisted reliability assessment. Key differentiator to articulate: those approaches consume *runtime* observability data (traces, metrics, logs); HGL operates strictly pre-deployment on architecture models, which changes both the feature space and the validation obligations. 6–10 citations needed from 2022–2026 literature. **Owner action: literature pass.**]

### 2.5 Explainable and Robust Graph Learning [NEW]

[EXPAND for SI audience: attention-based explanation of GNN predictions (GNNExplainer, attention-weight analysis and its known caveats), robustness/OOD generalization for GNNs. This grounds §8's method and pre-empts the "attention is not explanation" objection — cite and address it explicitly. **Owner action: literature pass.**]

---

## 3. System Model and HGL Architecture [CONF]

### 3.1 Typed Architecture Graph

Each deployment scenario is modeled as a heterogeneous directed graph

$$G = (V, E, \tau_V, \tau_E, w, \mathrm{QoS})$$

with node-type vocabulary $T_V = \{\text{Application}, \text{Library}, \text{Topic}, \text{Broker}, \text{Node}\}$ and edge-type vocabulary $T_E = \{\text{PUBLISHES\_TO}, \text{SUBSCRIBES\_TO}, \text{ROUTES}, \text{RUNS\_ON}, \text{CONNECTS\_TO}, \text{USES}, \text{DEPENDS\_ON}\}$. Six edge types are structural, imported directly from the topology description; DEPENDS_ON is a derived logical projection used only by the homogeneous baselines, which consume a type-collapsed view of the same system (an edge B→A when application B subscribes to a topic A publishes; A→L when A uses library L).

Each edge carries a 16-dimensional feature vector: one normalized structural-weight/path-count feature, a 7-dimensional edge-type one-hot, and 7 QoS-derived features — reliability score, durability score, transport priority, deadline presence, log-scaled deadline, log-scaled max-blocking, and a QoS-heterogeneity flag (non-zero only on pub-sub edges, where QoS is semantically meaningful; zeroed elsewhere). The QoS-masked variants zero dimensions 9–15 prior to training, isolating the contribution of explicit QoS attribution.

### 3.2 Relation-Specific Message Passing

HGL assigns a dedicated message function to each typed relation $(\text{src\_type}, \text{edge\_type}, \text{dst\_type})$, so publication, subscription, routing, library usage, and host deployment are each uniquely parameterized rather than absorbed into a single uniform transformation. Edge features are projected directly into the per-edge key/value representations before multi-head attention, avoiding the information smoothing that arises when edge patterns are aggregated indiscriminately across incoming paths. Message passing operates over the complete five-type graph; the model therefore uses Topic, Broker, and Node tiers as a contextual embedding layer even where prediction targets are application-level. [D4: fix architecture naming — "heterogeneous GAT" (conference text) vs. 3-layer EdgeAwareHGTConv (implementation) — and report layer count, heads, and hidden dimensions from the pinned configuration.]

### 3.3 Ground Truth and the Independence Guarantee

The target $I^*(v) \in [0,1]$ is the cumulative downstream message-flow degradation caused by failing $v$, measured by a discrete-event simulator over a fixed horizon. Two properties make this a legitimate pre-deployment validation: (i) simulator outputs never appear among model inputs — inputs are architecture-derived only — and (ii) the structural baselines' topology features are fully decoupled from the GNN feature pipeline. This **input–label independence guarantee** is enforced in the implementation by storage separation and static import-separation tests run as a hard CI gate, so the guarantee is a property of the artifact, not merely of the experimental protocol.

### 3.4 Model Variants

| Variant | Architecture | QoS encoding | Purpose |
|---|---|---|---|
| HGL-QoS | Heterogeneous attention GNN | 7-dim vector | Proposed, full |
| HGL | Heterogeneous attention GNN | masked | Isolate typed-structure gains |
| GL-QoS | Homogeneous GAT on projection | scalar weight | Homogeneous + QoS |
| GL | Homogeneous GAT on projection | none | Homogeneous baseline |
| Topo-QoS | QoS-weighted betweenness | QoS-weighted | Strongest structural baseline |
| Topo-BL | Betweenness + articulation points | none | Structural baseline |

Comparisons GL↔HGL (heterogeneity), HGL↔HGL-QoS (explicit QoS), and Topo-BL↔Topo-QoS (non-learned QoS recovery) disentangle the two design axes.

---

## 4. Extended Prediction Targets: Infrastructure Tiers [NEW]

The conference version validated predictions for Application (and Library) nodes only: the simulator quantified message-flow degradation among applications and attributed no independent impact to Topic, Broker, or Node tiers, which therefore served as contextual embedding layers without validated criticality claims. This section closes that gap — explicitly identified as the principal limitation of the prior study.

### 4.1 Simulator Extension

We extend the discrete-event oracle to inject failures at the Broker and deployment-Node tiers and to attribute the resulting application-level message-flow degradation back to the failed infrastructure component:

- **Broker failure**: all ROUTES edges through the broker are severed for the failure interval; impact is the normalized degradation of end-to-end flows whose routing traversed it, accounting for redundant-broker failover where the topology provides it.
- **Node failure**: all applications with RUNS_ON edges to the node fail simultaneously (a correlated multi-application failure); impact is the induced cascade beyond the co-located set itself.

[**Owner actions before this section is real:** (a) specify and implement the label semantics above against the pinned simulator configuration; (b) resolve R1 — the FailureSimulator vs. FaultInjector canonical-simulator ambiguity and the USES-cascade treatment of Library components — *before* generating any new labels, since infrastructure-tier labels inherit whatever cascade rules the canonical simulator applies; (c) re-run the independence-guarantee import checks over the extended label pipeline.]

### 4.2 Learning Setup

No architectural change is required: the heterogeneous model already embeds Broker and Node tiers. The extension adds prediction heads and labels for the two tiers, trained under the same protocol (§5). This is itself a finding worth stating: typed heterogeneous modeling makes multi-tier prediction an *output-side* extension, whereas the homogeneous projection would require re-engineering the graph reduction.

---

## 5. Evaluation Methodology [EXT]

### 5.1 Research Questions

- **RQ1 [CONF].** Does graph learning outperform structural-centrality baselines (betweenness, articulation points, QoS-weighted variants) for critical-component prediction in pub-sub topologies?
- **RQ2 [CONF].** Does preserving heterogeneous node/relation semantics outperform an equivalently trained homogeneous baseline on a type-collapsed projection?
- **RQ3 [CONF].** Does explicit QoS edge attribution add predictive value beyond what typed routing structure already encodes — in-distribution and out-of-distribution?
- **RQ4 [NEW].** Do the accuracy patterns established at the application level transfer to infrastructure-tier (Broker, Node) prediction targets?
- **RQ5 [NEW].** Which typed relations carry the cascade signal, as measured by learned attention mass, and are these attributions consistent with domain-known failure mechanisms?

### 5.2 Scenario Suite

The conference evaluation used seven synthesized scenarios spanning six topology classes: fan-out-dominated AV/ROS 2 (medium scale, RELIABLE/TRANSIENT_LOCAL), IoT smart city (large, VOLATILE/BEST_EFFORT), financial trading (dense medium, PERSISTENT/CRITICAL), healthcare integration (dense medium, PERSISTENT/RELIABLE), hub-and-spoke SPOF anti-pattern (two-broker constraint), sparse cloud-native microservices (precision stressor), and a hyper-scale enterprise platform (up to 300 applications). Each is generated from parameterized configurations with fixed seeds. [D2: the journal suite is [seven, unchanged | expanded to N, adding e.g. broker-redundancy (over-provisioned, suppressed-SPOF) and additional scale presets]. Whatever is chosen must be reconciled with the scenario counts stated in the companion JSS-regular-track and ASE manuscripts.]

### 5.3 Metrics

Ranking: Spearman ρ between predicted and simulated orderings (primary). Identification: F1@K on the top-K critical set, with precision/recall; accuracy reported only as a secondary metric due to prevalence sensitivity. Regression: RMSE/MAE, secondary, since the absolute scale of I\*(v) is simulator-specific.

### 5.4 Statistical Protocol

Each variant × scenario cell is evaluated over five independent seeds {42, 123, 456, 789, 2024}; cell means with bootstrap 95% CIs. For the conference design this yields 210 evaluation cells (140 trained models, 70 structural computations). **[NEW]** All pairwise variant comparisons are tested with paired Wilcoxon signed-rank tests at the seed level, with Holm–Bonferroni correction across the comparison family and rank-biserial effect sizes. [**Owner action:** run the Wilcoxon suite — this is the same run that unblocks the significance-language decision in the ASE manuscript; results: [p-values], [effect sizes].]

### 5.5 In-Distribution and LOSO Protocols

In-distribution: per-scenario train/validation/test splits over nodes. Out-of-distribution: Leave-One-Scenario-Out — train on all-but-one scenario, evaluate on the held-out one — approximating the deployment reality that a pre-trained predictor meets a *new* architecture with unobserved cascade dynamics.

---

## 6. Results: Application-Level Prediction [CONF]

### 6.1 Main Comparison (RQ1, RQ2)

Across the factorial design, HGL attains the best mean global ranking correlation (Spearman ρ = 0.620) and the best critical-component identification (mean F1 = 0.765). Against the equivalently trained homogeneous baseline, preserving typed node and relation semantics yields **ΔF1 = +0.284** — with QoS features *masked* in both, so the gain is attributable to heterogeneity alone. Structural baselines remain useful interpretable references but are dominated by the learned variants on identification. [Port full per-variant results table (Table 5 of conference version) with CIs; add Wilcoxon annotations from §5.4 once run.]

### 6.2 The QoS Trade-off (RQ3)

Explicit QoS attribution slightly *degrades* in-distribution accuracy relative to the QoS-masked HGL, yet is the primary driver of out-of-distribution generalization (§6.4). The mechanism: within a single scenario, the typed routing structure already encodes most QoS-relevant signal, so the added QoS dimensions mainly expand the parameter space and add optimization noise under limited per-scenario training nodes. Across scenarios, structure becomes scenario-specific and non-transferable, while QoS attributes (reliability, durability, deadline, priority) live on a common scale — a transferable channel that prevents representation collapse when memorized structural patterns are unavailable.

### 6.3 Node-Type Stratification

Aggregate correlations mask strong per-type signal — an instance of Simpson's paradox. Library nodes exhibit apparent zero variance under the application-level labeling regime while remaining vital structural context; stratified reporting by node type is therefore methodologically required, not cosmetic. [Note: the infrastructure-tier extension (§7) changes the stratification picture; recompute and report the stratified table under the new labels.]

### 6.4 LOSO Generalization

| Variant | Mean ρ | Std ρ | F1@K |
|---|---|---|---|
| GL | 0.0208 | 0.1418 | 0.2086 |
| GL-QoS | 0.0024 | 0.0946 | 0.2008 |
| HGL | 0.3073 | 0.2708 | 0.3896 |
| **HGL-QoS** | **0.4009** | 0.3672 | **0.4326** |

Homogeneous variants collapse to chance-level ranking on unseen scenarios; heterogeneous variants transfer, and QoS attribution adds a further +0.09 mean ρ. For the SI framing this is the load-bearing result: an AI reliability predictor that only works on the topology it was trained on has no pre-deployment use case.

---

## 7. Results: Infrastructure-Tier Prediction [NEW — all values pending]

[To be populated after §4 owner actions complete. Planned reporting:]

- Per-tier tables (Broker, Node) mirroring §6.1: [ρ_broker], [F1_broker], [ρ_node], [F1_node] per variant, 5 seeds, CIs, Wilcoxon annotations.
- **Hypothesis H7.1** (stated in advance, per pre-registration discipline): heterogeneity gains persist or widen at infrastructure tiers, because broker/node failure semantics are exactly the typed information the homogeneous projection discards. [Confirmed / rejected: pending.]
- **Hypothesis H7.2:** redundancy-aware scenarios (over-provisioned broker topologies) are where Topo baselines fail worst at the Broker tier, since betweenness cannot represent failover semantics. [Pending.]
- Stratified (Simpson-aware) reporting across all five node types under the extended labels.

---

## 8. Interpretability Analysis [NEW — method fixed, values pending]

Black-box opacity is a recognized barrier to acting on AI performance/reliability insights; for a predictor whose recommendations drive hardening investment, *why* a component scores high matters as much as the score. We analyze the trained HGL-QoS models' attention weights aggregated per relation type:

1. **Relation-level attention mass.** For each $(\text{src}, \text{rel}, \text{dst})$ triple, aggregate normalized attention mass across heads, layers, seeds, and scenarios: which typed relations does the model rely on? [Table: attention mass per relation × scenario class.]
2. **Consistency with known mechanisms.** Test whether attributions align with domain-known failure mechanisms — e.g., [does USES attention concentrate on shared libraries with large simultaneous fan-out, consistent with the library blast-radius mechanism? does ROUTES attention drop in redundant-broker scenarios?].
3. **Caveats.** Attention weights are an imperfect explanation proxy; we report them as *relation-level tendencies* across seeds — not per-prediction explanations — and cross-check stability across the five seeds. [Cite and address the "attention is not explanation" line of work in §2.5.]

[**Owner action:** implement attention-extraction hooks over the pinned checkpoints; verify the pinned architecture actually exposes per-relation attention (depends on D4 resolution).]

---

## 9. Case Study: Air-Traffic-Management System [NEW — inclusion pending D3]

> **Boundary constraint (D3):** the flagship JSS regular-track paper uses the ATM
> system for *expert-ranking validation of the interpretable Q(v) composite*.
> If this section is retained, it must be scoped strictly to **HGL predictive
> evaluation** — no expert-panel data, no τ/κ statistics, no Q(v) attribution
> claims — and both cover letters must disclose the shared system.

The subject is an ICAO-compliant ATM architecture: surveillance processing (radar tracking, ASTERIX-format brokering), flight-data processing, conflict detection, controller working positions, and meteorological services, communicating over eight pub-sub topics with mixed QoS (VOLATILE/BEST_EFFORT surveillance tracks through RELIABLE/CRITICAL conflict alerts with a 100 ms deadline). This combines the safety-critical QoS profile and mixed-durability topology that the synthetic suite covers only piecewise, on a real system model.

Evaluation: LOSO-style — models trained on the synthetic suite only, applied to the ATM graph — measuring [ρ_ATM], [F1@K_ATM] per variant. This is the sternest available test of the paper's deployment claim: a predictor trained on synthesized topologies confronting a real architecture. [Values pending; note that the simulator run against the ATM model is the same run that resolves R1 for the flagship — one committed run serves both papers, but its numbers must be reported in only one.]

---

## 10. Threats to Validity [EXT]

**Construct.** Ground truth is simulated, not observed; all claims are relative to the simulator's cascade semantics. The independence guarantee removes input-label circularity but not simulator-model mismatch with real deployments. [NEW: the infrastructure-tier labels add simulator design decisions (failover semantics, correlated co-location failure) that must be disclosed as modeling choices; document the pinned configuration and cascade rules — contingent on R1 resolution.]

**Internal.** The factorial design holds scenarios, seeds, graph data, and targets fixed across variants, so pairwise comparisons are controlled; the added Wilcoxon protocol quantifies seed-level uncertainty. Residual risk: hyperparameter budget parity across variants [document search protocol].

**External.** Scenarios are synthesized, parameterized from real deployment constraints; [the ATM case study provides one real-architecture data point | D3]. Generalization beyond pub-sub middleware (e.g., service meshes, actor systems) is untested.

**Conclusion validity.** With [N] scenarios and 5 seeds, per-cell samples are small; we report CIs and corrected non-parametric tests and avoid claims that rest on single-scenario deltas.

---

## 11. Conclusion [EXT]

We presented a heterogeneous graph learning framework for pre-deployment reliability analysis of publish-subscribe middleware, learning cascade impact directly on the native typed architecture graph under a strict input-label independence guarantee. Typed heterogeneity — not explicit QoS attribution — drives in-distribution accuracy (ρ = 0.620, ΔF1 = +0.284 over homogeneous learning), while QoS attribution is decisive out-of-distribution (LOSO ρ = 0.4009), a trade-off we explain via transferable versus scenario-specific signal. The journal extension [converts Broker and Node tiers from contextual embeddings into validated prediction targets, exposes which typed relations carry cascade signal via attention analysis, and adds seed-level statistical confirmation — results pending the committed runs registered above]. Together these position typed heterogeneous graph learning as a practical, interpretable AI technique for reliability and dependability analysis of complex ICT systems at the point in the lifecycle where hardening is cheapest: before deployment.

---

## References

[Port the conference reference list; add §2.4/§2.5 literature-pass citations; replace [Anon-A] (RASSE 2025 structural baseline) and [Anon-B] (Middleware 2026) with full citations upon acceptance.]

---

## Appendix A — Delta Document Skeleton (SI requirement)

| # | Section | Status | Content |
|---|---|---|---|
| 1 | §1, §2.1–2.3, §3, §5.1–5.3, §5.5, §6 | Carried/extended | Formulation, factorial design, application-level results |
| 2 | §2.4–2.5 | New | AI-for-dependability and explainability positioning |
| 3 | §4, §7 | New | Infrastructure-tier targets and results |
| 4 | §8 | New | Attention-based interpretability analysis |
| 5 | §5.4, Wilcoxon annotations | New | Seed-level statistical testing |
| 6 | §9 | New (pending D3) | Real-world ATM evaluation |

Estimated new technical content: [40–50]% (≥30% required). Submit with prior-version PDF once de-anonymized.