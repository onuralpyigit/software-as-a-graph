# Graph Neural Networks for Reliability and Dependability Analysis in Complex Distributed Systems based on Publish–Subscribe Architecture

> **`draft.md` is the authoritative manuscript text.** This outline is a section-by-section reading
> map of it — what each subsection argues, what backs the claim, what caveats travel with it, where a
> reviewer is likeliest to push, and what it depends on or feeds elsewhere. Where the two disagree,
> `draft.md` wins; this file is regenerated from it, not the other way around. Condensation history
> (what was cut and where it went) is in the [Appendix](#appendix-condensation-history) rather than
> the body, since that is no longer what a reader opening this file usually wants.

* **Target Journal:** Journal of Systems and Software (JSS) — Elsevier
* **Target Venue:** Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems" (VSI:AI4MSS)
* **Target Topic:** *AI for Reliability and Dependability Analysis in Complex ICT Systems*
* **Review model:** double-anonymised (`[Anon-A]` withheld; repo paths and tool names scrubbed)

## Highlights

*(verbatim from [`latex/highlights.tex`](latex/highlights.tex) — the submitted artifact; this list is
the one to keep in sync if either file changes)*

* Typed multigraph model predicts pub-sub cascading failure before deployment.
* Heterogeneous GNN leads a training-free QoS-weighted centrality on critical-set F1.
* AHP weighting slightly outranks uniform weights; RM is a baseline, not a ranker.
* Edge criticality is measured by simulated removal, not inferred from node labels.
* SaG runs as a blocking CI/CD gate in well under a minute, even at 500+ components.

## Keywords

publish–subscribe middleware; architectural dependability; cascading failure; heterogeneous graph
neural networks; static system analysis; pre-deployment verification; quality attributes; CI/CD
quality gate.

---

## Headline figures (as reported in the draft)

| Quantity | Value | Section |
|---|---|---|
| In-distribution mean $\rho$, HGL vs Topo-QoS | 0.730 vs 0.595 | §8.1 (Table 8) |
| Paired Wilcoxon, HGL vs Topo-QoS ($n=7$) | $p = 0.375$, **n.s.** | §8.1 (Table 9) |
| Paired Wilcoxon, HGL vs Topo-BL | $p = 0.016$, significant | §8.1 (Table 9) |
| LOSO mean $\rho$, HGL vs Topo-QoS | 0.608 vs 0.521 *(artifact not retained)* | §8.1 (Table 10) |
| LOSO $F_1@K$, HGL vs Topo-QoS | 0.465 vs 0.308 | §8.1 (Table 10) |
| In-domain k-fold, HGL vs Topo-QoS | 0.666 vs 0.492 | §8.2 |
| QoS encoding (RQ3) | null in all three regimes | §8.3 |
| AHP shrinkage, $\lambda=0$ vs $\lambda=0.70$ | 0.292 vs 0.181, monotone decline | §8.3 (Table 11) |
| Normalisation, rank vs min–max / z-score | 0.181 vs 0.318 ($+0.137$) | §8.3 |
| Threshold sweep span | 0.230 (0.001 at $t=0$ → 0.231 at $t=1$) | §8.3 (Table 12) |
| Oracle agreement, $I^*$ vs $I_{\text{comp}}$ | $\rho = 0.394$, Jaccard 0.286 | §7.4 |
| Oracle agreement, $I_{\text{dyn}}$ vs $I^*$ | $\rho = 0.765$, min 0.548 | §7.4 |
| Remediation acceptance | 162 / 332 (48.8%), 7 scenarios | §6.1, §8.4 |
| $\Delta$SRI range | $+0.0025$ to $+0.0158$ | §6.1 |
| Edge removal, `av_system` | 4 of 50 candidates non-zero | §8.2 |
| Library blast hypothesis | not confirmed (165 libraries) | §5.4 |
| Real-world $\rho$ (Autoware / Cloud / Train-Ticket) | 0.688 / 0.778 / 0.759 | §8.5 (Table 13) |
| Real-world gate | 5 of 15 checks fail; all three fail SPOF-F1 | §8.5 (Table 13) |
| Corpus | 1,545 synthetic + 225 real-world = 1,770 components | §7.1 |
| CI/CD gate runtime | <2 s tiny → ~27 s enterprise; catalog precision 0.24–0.40 | §8.4 |

**Two standing caveats that travel with every figure above.** The ranking half of RQ1 is not
established: it fails the paired test in-distribution and its LOSO artifact was not retained (§8.1,
§9.2). And $I_{\text{comp}}$-backed results (§5.4, §8.2's edge-criticality finding) do not transfer to
$I^*$-backed tables (§8.1), because the two oracles agree only at $\rho = 0.394$ (§7.4).

---

## Section-by-section map

Each entry: what the subsection argues, the evidence that carries it, the caveats/definitions that
are load-bearing there, where a reviewer is likeliest to push, and its dependencies on other sections.

### 1. Introduction

#### §1.1 Motivation
**Argues.** Pub-sub decoupling (producers/consumers separated in time, space, synchronization) is
what makes the paradigm scale, and exactly what hides the dependency chains a failure propagates
along — there are no caller–callee edges, so a diagram's apparent importance and a component's actual
blast radius can diverge. Pre-deployment is when hardening is cheapest, yet it is also when no runtime
telemetry exists.
**Evidence.** Prose only, framed against DDS/MQTT QoS policy surface [2, 3] and pub-sub theory [1]; a
two-sentence sustainability framing (cascades cost re-provisioning and retransmission energy).
**Load-bearing caveats.** None yet — this section motivates rather than claims.
**Reviewer risk.** Low; standard motivating framing.
**Depends on / feeds.** Sets up the "architecture alone, no telemetry" constraint that governs every
later oracle and predictor design (§3, §5).

#### §1.2 Problem Statement and Limitations of Existing Approaches
**Argues.** Two coupled sub-problems: (1) interpretable attribution of *how/why* a component is
critical, grounded in ISO/IEC 25019:2023 Quality-in-Use; (2) prediction of cascade impact. Both must
run pre-deployment and stay explainable. Names the **Architecture-Code Gap**: clean code can still sit
inside a fragile deployment topology. Three prior strands each leave the gap open — SCA (blind to
topology), runtime chaos engineering (needs a running system), topology-only/homogeneous learned
centrality (collapses distinct failure mechanisms into one scalar).
**Evidence.** Prose positioning only; no numbers.
**Load-bearing caveats.** "Both must be computed without runtime data" is the constraint §5.1's three
oracles and §4's deterministic $Q(v)$ are built to satisfy.
**Reviewer risk.** A reviewer familiar with the SSA literature may ask why this three-strand framing
is exhaustive; §2 is where the literature review actually substantiates it.
**Depends on / feeds.** Directly motivates the six contributions of §1.4 and the four-paragraph
related-work structure of §2.

#### §1.3 Our Approach
**Argues.** Introduces SaG's shape end-to-end: the typed multigraph (§3), hierarchical RM attribution
(§4) audited for AHP consistency at the sub-characteristic level, the two failure-impact predictors — composite $Q(v)$ and HGT-based HGL,
evaluated as HGL vs $HGL\text{-}QoS$ to isolate QoS contribution — validated against a discrete-event
simulator under the **input–label independence guarantee** (§5), prescriptive remediation (§6.1), and
the CI/CD gate (§6.2, currently absolute not delta-aware). States all five research questions in full.
**Evidence.** Figure 1 (pipeline diagram, prose placeholder — not yet a rendered figure in `draft.md`
beyond the caption; the rendered asset is `latex/figures/Figure_1.{pdf,png}`).
**Load-bearing caveats.** RQ1 is deliberately phrased as *where*, not *whether* — this framing is what
lets §8.1 land on "scope condition, not verdict" without that reading as a retreat.
**Reviewer risk.** The section states "organized around **four** research questions" (draft.md:122)
immediately before listing RQ1–RQ5 — a numbering slip a copyeditor or reviewer will flag; see
[Draft inconsistencies](#draft-inconsistencies-recorded-not-fixed).
**Depends on / feeds.** RQ1→§8.1, RQ2→§8.2, RQ3→§8.3, RQ4→§8.4, RQ5→§8.5 (full traceability table
below). RQ1–RQ3 run on synthetic scenarios, RQ4 is the gate, RQ5 is the only real-world evidence.

#### §1.4 Contributions
**Argues.** Six contributions: (1) typed model + hierarchical RM + HGT (§3–§5); (2) a **scope
condition** on where learning pays — leads on both ranking and critical-set ID out of distribution,
but the ranking margin is the contested half (§8.1); (3) RM repositioned as **attribution, not
accuracy** — intra-dimension shrinkage sweep shows a small, monotone edge for the calibrated judgement
over uniform weights, not the reverse (§4, §8.3); (4) edge ground truth by **measured**
removal, not inferred multiplier (§8.2); (5) remediation + CI/CD gate, absolute not delta-aware
(§6, §8.4); (6) real-world validation on three architectures across two paradigms (§7.1, §8.5).
**Evidence.** Each contribution states its own headline number inline (e.g. contribution 2 quotes
$\rho = 0.608$ vs $0.521$ LOSO and $F_1@K = 0.465$ vs $0.308$ — the *out-of-distribution* figures, not
the in-distribution ones from the abstract; this is a deliberate and correct choice, not a
duplication, since §8.1 treats OOD as the more defensible evidence).
**Load-bearing caveats.** Contribution 2 explicitly states "the contribution is the scope condition,
not the win" — this sentence is the one to preserve verbatim in any abstract/highlights rewrite, since
it is the paper's own guard against overclaiming RQ1.
**Reviewer risk.** §9.1 later refers to "of the **four** contributions" (draft.md:1204) when discussing
only the CI/CD gate's standing — another numbering slip against this list of six; see
[Draft inconsistencies](#draft-inconsistencies-recorded-not-fixed).
**Depends on / feeds.** Each contribution is a compressed pointer into §3–§8; contribution 2 and 3 are
the ones with the sharpest caveats and are worth re-reading together with §9.2's threats.

#### §1.5 Prior Work and Organization
**Argues.** Positions this paper as consolidating the authors' prior structural-baseline work
`[Anon-A]` with the HGT predictor, attribution, SCA integration, and CI/CD gating into one submission;
states no companion manuscript is under parallel review. Closes with the standard section roadmap.
**Evidence.** None — administrative/roadmap paragraph.
**Load-bearing caveats.** The "no companion manuscript in parallel" statement matters for
double-anonymised review compliance; keep it accurate if `[Anon-A]`'s status changes before
submission.
**Reviewer risk.** None inherent; standard for the venue.
**Depends on / feeds.** N/A.

### 2. Related Work

**Argues.** Four themed paragraphs, unsubdivided (no numbered `2.x`), covering: (i) pub-sub
dependability, runtime fault tolerance, and Chaos Engineering [18] as reactive/late-lifecycle; (ii)
SCA tools [29–31] as topology-blind vs. SSA's Architecture-Code Gap framing and continuous
pre-deployment gating [19–21], explicitly citing SonarQube's delta-aware "Clean as You Code" gate as
the model §6.2's future delta-aware design would follow; (iii) structural centrality [4–6, 36–39] and
learned criticality (FINDER [7], DrBC [8], PowerGraph [9]) built on homogeneous message-passing
(GCN/GraphSAGE/GAT [40–42]) vs. heterogeneous GNNs (RGCN/HAN/HGT/MAGNN [10–13]) as the direct
motivation for relation-specific message passing, with over-smoothing [14, 53] noted as a known
hazard; (iv) ISO/IEC 25010/25019 quality attributes [16, 17], AHP [15] used here to *state and audit*
rather than elicit weights, ATAM-style scenario methods [33–35] as the architecture-evaluation
tradition being positioned against, and architectural anti-pattern/refactoring work [22–28] contrasted
with this paper's simulation-verified acceptance criterion.
**Evidence.** All 53 numbered references retained; no tables/figures.
**Load-bearing caveats.** The closing summary sentence ("prior approaches address … but not … an
interpretable attribution over a typed graph") is the section's thesis statement and should be checked
against §1.2's three-strand framing for consistency if either is edited.
**Reviewer risk.** A JSS reviewer for this special issue will likely check citation currency (PowerGraph
[9] is 2024, everything else pre-2023) and whether SonarQube's "Clean as You Code" citation [21] is a
proper reference (currently an "[Online]" doc citation, not peer-reviewed) — minor, but worth a glance.
**Depends on / feeds.** Feeds §6.2's delta-aware future-work framing (SonarQube analogy) and §5's
heterogeneous-GNN citations that motivate HGT specifically.

### 3. The SaG Model

#### §3.1 Nodes, Edges, and the Formal Object
**Argues.** Defines $G = (V, E, \tau_V, \tau_E, w_E, w_V)$ over five node types (Application, Broker,
Topic, Node, Library) and six structural edge types. States that retaining these types — not
collapsing to one "communicates-with" relation — is what later lets the framework distinguish failure
mechanisms an untyped graph cannot. One sentence on `cm_*` SonarQube metric ingestion feeding §4.2's
Code Quality Penalty.
**Evidence.** **Table 1** (node types + structural edge types, merged).
**Load-bearing caveats.** None new; this is the formal foundation §3.2–§9 build on.
**Reviewer risk.** Low — this is definitional. A reviewer might ask why exactly these five node types
and not more (e.g., separate Producer/Consumer roles); not addressed here.
**Depends on / feeds.** Every later table (2–13) implicitly assumes this typing; §4.1's orthogonality
claim depends on these types being genuinely disjoint.

#### §3.2 QoS-Aware Weights and Derived Dependencies
**Argues.** Edge weights $w(e) \in [0,1]$ from topic QoS (reliability/durability/priority weighted
0.30/0.40/0.30, durability dominant since it governs message-state survival) blended 0.85/0.15 with
payload size, floor 0.01. Six `DEPENDS_ON` projection rules derive logical dependency from structural
edges, always dependent→dependency. **The crux of the model**: Rule 1 (`app_to_app`) encodes
*sequential cascade*; Rule 5 (`app_to_lib`) encodes a *simultaneous blast* — every consumer of a
failed library fails at once, not along a propagation path. An untyped graph cannot represent this
distinction.
**Evidence.** **Table 2** (six projection rules, pattern, weight formula).
**Load-bearing caveats.** `path_count` (multi-topic coupling) is kept *separate* from the $[0,1]$
weight contract deliberately — conflating them would break the weight semantics used everywhere else.
**Reviewer risk.** The blast-radius mechanism motivated here is the one §5.4 tests and does *not*
confirm on the synthetic suite (max library $Q = 0.422$) — a reviewer reading only §3.2 might expect a
stronger empirical payoff than §5.4/§8.2 deliver; the outline flags this tension explicitly so it
isn't missed on a first pass.
**Depends on / feeds.** Rule 5 → §5.4's shared-library blast test (negative result) → §6.1's
FanOutReduction operator trigger (kept as a *structurally motivated* safeguard despite §5.4's result).
Rule 6 (`broker_to_broker`) → makes broker-colocation risk representable at all.

#### §3.3 Graph Views and a Running Example
**Argues.** Two views: $G_{\text{structural}}$ (raw, feeds the simulators) and
$G_{\text{analysis}}(\ell)$ (layer-projected `DEPENDS_ON`, feeds attribution/prediction). Their
separation is the **independence guarantee** — the load-bearing property that makes every correlation
reported in §8 non-circular. Four analytical layers (app/infra/mw/system) roll up along a MIL-STD-498
[52] hierarchy. A three-application running example illustrates sequential-cascade vs
simultaneous-blast losing $a_1$ vs losing $\ell$.
**Evidence.** Figure 2 (running example diagram, prose placeholder; rendered as
`latex/figures/Figure_2.{pdf,png}`).
**Load-bearing caveats.** **The independence guarantee is the single most cited structural property in
the paper** — reused verbatim in §5.3, §9.2, and implicitly underwrites every $\rho$ reported in §8. If
this section's claim were wrong, essentially every quantitative result in the paper would need
re-reading as potentially circular.
**Reviewer risk.** §9.2 itself later qualifies this as "view independence, not source independence"
(both views are deterministic functions of the same input topology) — a subtlety a careless reader of
§3.3 alone would miss; make sure any summary of §3.3 also carries that qualification rather than
stating the guarantee unconditionally.
**Depends on / feeds.** Feeds §4.2 (attribution is deterministic and independent of simulation output),
§5.3 (restates and extends the guarantee with an inference-time leakage check), §9.2 (states the
"view not source" caveat).

### 4. Interpretable Attribution as a Baseline

#### §4.1 Two Dimensions and Formal Definitions
**Argues.** Criticality is not itself a quality characteristic but a characteristic's sensitivity to
element loss, and is instantiated **primarily on Reliability**: ISO/IEC 25010:2023 composes
Reliability from faultlessness (frequency), fault tolerance and recoverability (duration, jointly with
faultlessness composing availability) — Reliability itself blends fault tolerance and availability
hierarchically, $R(v) = \alpha \cdot FT(v) + (1-\alpha)\cdot A(v)$, $\alpha = 0.36$; faultlessness is
excluded by the same consequence-not-risk logic as D3, and recoverability is an explicit data gap (no
MTTR field). Maintainability is a secondary, thinner instantiation (2/5 sub-characteristics). RM —
Reliability (hierarchical) and Maintainability — each answers a distinct architectural question, is
denominated in a named **external quality attribute** (ISO/IEC 25010:2023) and a **dependability
attribute**, and routes to a distinct remediation owner. The section opens with the three-view framing
that governs the whole paper: criticality is **computed** from internal quality evidence, **validated**
against simulated external quality, and **defined** on Quality-in-Use — the distinction §8.2's
three-link chain rests on. The FT/A sub-terms are **orthogonal by construction at the metric level**
(disjoint raw-metric inputs, a design constraint not an empirical finding) but **not
attribute-independent**: both are Reliability sub-characteristics, so a high score on both is one
characteristic degraded two ways, not two unrelated problems — which is why they are blended rather
than reported as independent peers. Two definitions do real work downstream: **criticality is a
consequence, not a risk** (no dimension estimates failure probability, only loss given failure — what
makes the construct computable pre-deployment); and **criticality is relative, not absolute (D4)** —
scores/tiers are relative to the analysed system's own distribution and layer, so **scores are not
comparable across systems or layers**. Relationship-level (edge) attribution is defined with the same
two dimensions but explicitly *not* developed further here, since it cannot currently be validated
against the edge-removal measurement of §8.2 (disjoint populations). Security is not instantiated as a
scored dimension.
**Evidence.** **Table 3** (two dimensions, Reliability's FT/A sub-terms broken out × architectural
question × external quality attribute × dependability attribute × remediation owner), plus explicit
rows correctly stating that **Safety** — a first-class ISO/IEC 25010:2023 characteristic since the
2023 revision, not something outside the standard — and **Security** are not among the characteristics
this framework instantiates, because an architecture description carries no hazard class or threat
model (see also §8.3).
**Load-bearing caveats.** **D4 is the most cited definition in the paper outside the independence
guarantee.** It is what forbids §5.4's stratified check and §8.5's three real-world $\rho$ values from
being pooled into one cross-system number — both sections state the resulting scoping explicitly.
Any future aggregation across scenarios must check D4 first.
**Reviewer risk.** A reviewer could ask whether "orthogonal by construction" is falsifiable, since it's
a modelling choice rather than a tested property — the text itself concedes this ("a deliberate design
constraint, not an empirical observation"), so the honest framing is already in the draft.
**Depends on / feeds.** D4 → §5.4, §8.5 scoping statements. Relationship-attribution deferral →
§9.3's limitations list (closing it needs a many-to-many mediating-relation design, left as future
work).

#### §4.2 The Composite Score, Classification, and Determinism
**Argues.** Defines the sub-term formulas ($FT$: Reverse PageRank on $G^\top$ + in-degree +
cascade-depth; $A$: directed articulation score amplified by QoS weight; blended into Reliability
$R(v) = \alpha \cdot FT(v) + (1-\alpha)\cdot A(v)$, $\alpha = 0.36$; Maintainability: betweenness +
QoS-weighted out-degree + Code Quality Penalty, formula unchanged from the retired model). Composite
$Q(v) = w_R R(v) + w_M M(v)$ under declared weights $(w_R, w_M) = (0.80, 0.20)$ — algebraically derived
from the retired four-dimension AHP composite $(0.43, 0.24, 0.17, 0.16)$, audited for AHP consistency
($\mathrm{CR} \le 0.10$) at the sub-characteristic level rather than freshly elicited. Adaptive
box-plot classification (CRITICAL at $Q_3 + 1.5\,\mathrm{IQR}$). Attribution is **fully deterministic**
— no input derives from the simulation that produces ground truth, which is what makes a measured
correlation evidence rather than leakage. States the shrinkage-sensitivity result directly: **the raw
AHP judgement edges out uniform intra-dimension weights by $\approx 0.032\,\rho$, a small but
consistent effect**, with the conclusion drawn immediately: *the value of RM remains attribution, not
ranking accuracy*.
**Evidence.** The $\approx 0.032\,\rho$ figure is stated here in prose; the full sweep (**Table 11**,
Figure 4) is in §8.3. This is a deliberate structural choice — the paper states its robustness result
in the method section itself rather than saving it for results, which reads as intentional
transparency rather than something to flag as a defect.
**Load-bearing caveats.** "The value of the RM decomposition remains attribution, not ranking
accuracy" is the sentence that reframes contribution 3 (§1.4) and should be treated as a fixed point —
any edit to §8.3's numbers should re-check this sentence still follows from them.
**Reviewer risk.** A reviewer could ask why the AHP-derived weights are retained given the small
effect size — the text's answer (attribution ≠ ranking, so the profile's *shape* still matters
regardless of weights, and the small effect now favours rather than undercuts the calibrated setting)
is present but somewhat implicit; worth having ready as a rebuttal point.
**Depends on / feeds.** The determinism claim depends on §3.3's independence guarantee. The shrinkage
result is fully unpacked in §8.3 (Table 11, Figure 4) and echoed in §9.1's second finding and §9.3's
"dimension weighting does not improve accuracy" limitation.

### 5. Failure-Impact Analysis

#### §5.1 Ground Truth: Three Simulation Oracles
**Argues.** **Three oracles exist and are not interchangeable** — the section's central warning.
$I^*(v)$ (`FaultInjector`, BFS cascade, mean subscriber feed-loss) is what trains/evaluates the learned
predictors and backs §8.1's tables. $I_{\text{comp}}(v)$ (`FailureSimulator`, four-component weighted
composite: 0.35 reachability + 0.25 fragmentation + 0.25 throughput + 0.15 flow) backs validation
gates, §5.4's checks, and remediation acceptance (§6.1). $I_{\text{dyn}}(v)$ (`MessageFlowSimulator`,
discrete-event traffic simulation) trains and gates nothing — it is a construct-validity check only
(§7.4). All three share `propagation_threshold` (default 0.2) and run over five seeds, with
across-seed std reused as the noise scale for §6.1's acceptance criterion.
**Evidence.** Formulas given in full for $I_{\text{comp}}$; agreement figures deferred to §7.4.
**Load-bearing caveats.** $I^*$ vs $I_{\text{comp}}$ agree only at mean $\rho = 0.394$ — stated here and
quantified in §7.4 — which is the single most consequential scoping fact in the paper, since it means
evidence gathered against one oracle does not transfer to claims measured against the other.
**Reviewer risk.** Three non-interchangeable ground-truth oracles is unusual and a reviewer may ask why
not standardize on one — the paper's own answer is that each is doing a different, incommensurable job
(labels vs. gates vs. construct check), which is defensible but should be stated crisply if challenged.
**Depends on / feeds.** This is the foundational section for all of §5.4, §6.1, §7.4, §8.1, §8.2, §8.5
— essentially every quantitative result in the paper cites back to one of these three oracles by name.

#### §5.2 Two Predictors over the Same Model
**Argues.** $Q(v)$ (§4, deterministic) and HGT-based HGL (relation-specific attention/message-passing
across five node types and structural/`DEPENDS_ON` edge types) span the interpretability–capacity
spectrum deliberately. $HGL\text{-}QoS$ additionally injects continuous QoS attributes into edge
attention; base HGL masks them — this contrast *is* RQ3 (§8.3). Both consume $G_{\text{analysis}}$
features and train inductively against $I^*(v)$. Predictors reported separately, not blended, so RQ1
is answered on like-for-like rankings.
**Evidence.** Figure 3 (per-edge attention $\alpha$ case study, prose placeholder; rendered as
`latex/figures/Figure_3.{pdf,png}`).
**Load-bearing caveats.** The attention weights shown in Figure 3 were, per §9.2, initially captured as
nothing at all due to a pinned PyTorch Geometric release lacking `return_attention_weights` — repaired
before this figure was produced; worth knowing if Figure 3 is regenerated.
**Reviewer risk.** None specific beyond the general HGT-architecture questions §7.5's fixed
hyperparameters are meant to pre-empt.
**Depends on / feeds.** HGL vs $HGL\text{-}QoS$ → §8.3 RQ3 (null result). HGL vs $Q(v)$ → §8.1 RQ1.

#### §5.3 The Independence Guarantee
**Argues.** Restates and operationalises §3.3's guarantee: predictors on $G_{\text{analysis}}$,
simulators on $G_{\text{structural}}$, distinct passes; no simulation output ever feeds back as a
predictor feature. For the learned predictor specifically, this is checked *at inference time* — the
GNN service raises if a feature tensor carries a target-label attribute. Remediation's Generate phase
obeys the same discipline (never reads simulated impact).
**Evidence.** No new numbers; a structural/procedural guarantee.
**Load-bearing caveats.** The inference-time check is explicitly framed as "a defensive check against a
leakage bug introduced later … rather than a proof none exists today" — an honest, non-overclaiming
statement worth preserving if this section is ever tightened.
**Reviewer risk.** Low, given the explicit hedge above pre-empts the obvious "how do you know there's no
leakage" question.
**Depends on / feeds.** Underwrites every correlation in §8; explicitly extended to §6.1's remediation
Generate phase.

#### §5.4 Two Checks That Take Node Type Seriously
**Argues.** Two analyses, both against $I_{\text{comp}}(v)$, both **negative or consistency results**,
not headline findings. **(1) Shared-library blast**: hypothesis that $Q(v)$ understates library
cascade impact — **not confirmed** across 165 library nodes (max $Q = 0.422$; $I_{\text{comp}}$ never
exceeds $Q$ for any library). The mechanism (Rule 5, §3.2) is retained as structurally motivated
regardless. **(2) Stratified correlation**: checks for a Simpson's-paradox-style masking effect;
**not found** — pooled $\rho = 0.374$ sits inside the per-type range (0.322–0.429). But the *same
effect does occur elsewhere*: pooling Application and Library nodes moved HGL on `av_system` from
$\rho = 0.836$ (within Applications) to $0.46$ pooled (§8.2) — a case where pooling was genuinely
misleading, caught only once the evaluation contract of §7.2 was imposed.
**Evidence.** 165 libraries, $Q_{\max} = 0.422$; 1,545 nodes pooled $\rho = 0.374$ vs 0.322–0.429
per-type, all $p < 0.01$.
**Load-bearing caveats.** Both checks are **$I_{\text{comp}}$ results** — per the §5.1/§7.4 scoping,
they do not transfer to or corroborate the $I^*$-backed tables of §8.1, and §8.2 explicitly flags this
rather than letting textual adjacency imply mutual support. Per D4 (§4.1), the pooled $\rho = 0.374$
is a diagnostic over the union of seven within-system rankings, not a cross-system result.
**Reviewer risk.** A reviewer skimming only §5.4's headline sentences ("not confirmed", "not found")
might read this as two failed checks; the actual framing — retained mechanism despite one negative
result, real distortion caught by the discipline that produced the other — is more nuanced and worth
stating explicitly in any summary.
**Depends on / feeds.** §8.2 restates both, adjacent to the edge-criticality finding, with the oracle
caveat repeated. The `av_system` pooling distortion is echoed in §9.1's second finding.

### 6. Prescriptive Remediation and CI/CD Quality Gating

#### §6.1 Generate–Verify and the Remediation Operators
**Argues.** Two strictly separated phases. **Generate**: four operators propose edits from structure
alone, never reading simulated impact (**Table 4**: RedundancyInsertion, PathDiversification,
FanOutReduction, SharedTopicReduction). **Verify**: each candidate is applied to a counterfactual
graph, `FailureSimulator` re-run from scratch, accepted only if $\Delta I > \kappa\,\sigma_{\text{seed}}$
($\kappa = 1.0$) at *every* sampled propagation threshold. No Verify result feeds back into Generate
within a run — closing off any closed-loop search that would reintroduce circularity.
**Evidence.** 162/332 candidates (48.8%) accepted, no scenario regresses under the filter; acceptance
varies sharply by topology (3/35 Autonomous Vehicle vs 38/58 IoT Smart City); $\Delta$SRI real but
small ($+0.0025$ to $+0.0158$).
**Load-bearing caveats.** **Acceptance is decided on singletons, not subsets** — nothing establishes
that a set of individually-accepted edits remains beneficial applied together; this is stated as an
open limitation, not glossed over. FanOutReduction's trigger is deliberately a direct structural
blast-radius signal, *not* $Q(v)$ — the remediation-side expression of "single-score criticality is
insufficient," independent of whether §5.4 confirms the library-blast mechanism in this suite.
**Reviewer risk.** The singleton-not-subset limitation is exactly the kind of gap a JSS reviewer
evaluating "practical value" will probe; §9.3 already lists it as future work, so the honest framing is
consistent across the paper.
**Depends on / feeds.** Requires §5.1's $I_{\text{comp}}$ and its seed-noise $\sigma$ (§7.3). Yield
figures are absorbed into §8.4's RQ4 discussion; echoed in §9.1's fourth finding and §9.3.

#### §6.2 The CI/CD Quality Gate
**Argues.** SaG runs as a blocking gate via `detect_antipatterns.py`, exit codes 0 (clean) / 1 (medium
warnings) / 2 (CRITICAL/HIGH — build broken, deployment blocked). Runs in-memory via the thread-safe
`MemoryRepository` port with no live database dependency — this is explicitly what makes §8.4's
runtimes achievable. **This is an absolute gate, not delta-aware**: every run evaluates the full
finding set, not a diff against a prior baseline, so an intentional, risk-accepted SPOF fails the build
on every commit, indistinguishable from a genuine regression. The fix (delta-aware gate + waiver
register) is described as a design, not implemented, deferred to §9.3.
**Evidence.** No new numbers here; runtime numbers are in §8.4.
**Load-bearing caveats.** The absolute/delta-aware distinction is stated correctly here and must stay
consistent with §1.3, §9.1, and §9.3 — an earlier revision had §9.1/§9.4 mis-describing the gate as
delta-aware, since corrected (§9.1's note flags this explicitly).
**Reviewer risk.** The known-and-disclosed consequence (real architectures with accepted SPOFs fail
every build) is a legitimate adoption blocker a reviewer may weigh heavily against RQ4's "feasibility"
claim — the paper does not minimise this, which strengthens credibility but the limitation itself is
real and unresolved in the current submission.
**Depends on / feeds.** §2's SonarQube "Clean as You Code" citation is the explicit model for the
undelivered delta-aware design. Feeds §8.4 (runtime measurement), §9.1 (interpretation), §9.3 (listed
as engineering work, not open research).

### 7. Experimental Setup

#### §7.1 Datasets
**Argues.** Seven synthetic scenarios (50–300 apps, statistical topology generator, ten deployment
domains) + three real-world architectures — Autoware.universe (75 components incl. 32 apps, ROS 2
[46]), Cloud-Native Microservices Mesh (60 components incl. 22 apps, based on Google Online Boutique
[48]), Train-Ticket (90 components incl. 41 apps, [47]). Pooled corpus: 1,770 components. Synthetic
suite is byte-reproducible from configs (SHA-256 manifest, regression-tested); the three real-world
graphs are hand-transcribed by a dedicated adapter, so the byte-identity guarantee applies to the
synthetic suite **only**.
**Evidence.** Component/topic/broker/node/library counts per real-world graph given explicitly.
**Load-bearing caveats.** The synthetic-vs-real-world reproducibility asymmetry is stated plainly and
matters for §9.2's artifact-retention discussion and §9.3's "hand-transcribed, not harvested" limitation
on real-world validity.
**Reviewer risk.** "Hand-transcribed" real-world graphs (vs. mined/harvested from actual repos) is a
external-validity weak point the paper itself flags repeatedly (§8.5, §9.2, §9.3) rather than only
here — consistent self-scoping, but still the single biggest external-validity target for a reviewer.
**Depends on / feeds.** Feeds every table in §8; the CLAUDE.md invariant that
`data/scenarios/*_system.json` regenerates byte-identically from configs is the enforcement mechanism
for the synthetic half of this claim (`tests/test_scenario_corpus.py`).

#### §7.2 Predictors, Baselines, and Evaluation Metrics
**Argues.** **Table 5** lists six predictors/baselines and the factor each contrast isolates
(Topo-* vs learned → value of learning; GL vs HGL → value of typing; HGL vs HGL-QoS → RQ3; RM/Q vs
learned → when interpretable attribution suffices). Three metric families: ranking ($\rho$ primary,
plus NDCG@10, Top-5/10 overlap), identification (precision/recall/F1, SPOF-F1), statistical rigor
(bootstrap 95% CIs, paired Wilcoxon). $\rho$ always reported by node type in addition to pooled.
**One evaluation contract, one sample** — this subsection's central methodological claim, stated as a
*correction*: an earlier version scored predictor families on different populations, and fixing it
raised every learned variant by 0.35–0.48 $\rho$ in-distribution while leaving baselines essentially
unchanged. **Absent is not zero**: strata with constant labels get an undefined correlation, never a
quoted 0.0.
**Evidence.** The 0.35–0.48 $\rho$ correction magnitude; no other new numbers here (results are §8).
**Load-bearing caveats.** "One evaluation contract, one sample" is what makes Table 8 (§8.1) trustworthy
at all — this is arguably the single most important methodological fix disclosed in the paper, since it
*changed the sign of the RQ1 conclusion* (per §9.2).
**Reviewer risk.** A reviewer will likely ask *how* different populations were scored previously and
whether any other undisclosed inconsistency remains — the paper's answer is the six-instrument-defect
disclosure in §9.2, which this section should be read alongside.
**Depends on / feeds.** Directly enables §8.1's Table 8/9/10; §9.2's internal-validity discussion
names this correction explicitly as one of two pre-submission-audit findings.

#### §7.3 Protocols
**Argues.** Two regimes: **in-distribution** (per-scenario, for RQ1/RQ2) and **inductive LOSO**
(train on six scenarios, evaluate on the unseen seventh — the true pre-deployment condition). Five
seeds $\{42, 123, 456, 789, 2024\}$ throughout; scores are seed means, across-seed std reused by
§6.1's acceptance criterion. The remediation sweep (§6.1) is the one exception, using three seeds for
compute-cost reasons.
**Evidence.** No new numbers; protocol definition.
**Load-bearing caveats.** LOSO is explicitly framed as "the true pre-deployment condition" — this
framing is why §8.1 and §9.1 weight LOSO's $F_1@K$ result more heavily than the in-distribution table
despite the in-distribution numbers being individually larger.
**Reviewer risk.** Low; standard ML evaluation protocol description.
**Depends on / feeds.** LOSO protocol → §8.1 Table 10 (whose artifact was *not* retained, §9.2) and
§8.2's generalisation-gap finding.

#### §7.4 Ground Truth: Three Oracles and Their Agreement
**Argues.** **Table 6** restates the three oracles with their measured quantity and which results rest
on which — the single reference table for "which number came from where." Quantifies oracle agreement:
$I^*$ vs $I_{\text{comp}}$ mean $\rho = 0.394$, Jaccard 0.286 (range 0.578 Enterprise down to 0.092
Hub-and-Spoke — near-independence); $I_{\text{dyn}}$ vs $I^*$ mean $\rho = 0.765$ (min 0.548), which
rules out the cascade *algorithm* as the ranking's source since $I_{\text{dyn}}$ reaches similar
ranking by simulating traffic through queues rather than traversing `DEPENDS_ON` — but top-20% Jaccard
is only 0.316, so this corroborates ranking, not critical-set identification. Label coverage: 30–47%
of components per scenario carry no ground truth (Topic/Node types unmodelled by the cascade); test–
retest reproducibility ceiling $\rho = 0.807$–$1.000$, Jaccard 0.44–1.00.
**Evidence.** **Table 6**; all agreement figures above.
**Load-bearing caveats.** This section's opening line — "conflating [the oracles] is the most likely
way to over-read a result in this paper" — is the paper's own warning label; every scoping-caveat
sentence elsewhere in the draft (§5.4, §8.2, §9.2) traces back to the $\rho = 0.394$ figure computed
here. The test–retest ceiling bounds every top-$K$ metric in §8.
**Reviewer risk.** $\rho = 0.092$ at Hub-and-Spoke (near-independence between the two cascade oracles on
one scenario) is a striking number a reviewer will likely zero in on — worth having the "all seven
correlations are positive, so weak convergent validity, not zero" framing ready.
**Depends on / feeds.** This is the source data for every "oracle X vs oracle Y" scoping caveat used
throughout §5, §6, §8, §9 — arguably the second most cross-referenced section after §3.3's independence
guarantee.

#### §7.5 Model Configuration and Implementation
**Argues.** **Table 7** fixes every learned-predictor hyperparameter identically across HGL, HGL-QoS,
GL, GL-QoS, values fixed *before* runs and not tuned per scenario (avoiding held-out leakage under the
in-distribution protocol). LOSO sweep dominates runtime (~31 min HGL, ~36 min HGL-QoS across folds and
seeds).
**Evidence.** **Table 7**; runtime figures.
**Load-bearing caveats.** CI/CD gate measurements (§8.4) were taken on this same single workstation,
not hosted CI runner hardware — explicitly flagged as an order-of-magnitude feasibility result, not a
calibrated per-provider figure.
**Reviewer risk.** Single-workstation, non-tuned-per-scenario hyperparameters is a reasonable and
disclosed choice, but a reviewer may ask about sensitivity to the fixed hyperparameter values
themselves (as opposed to the RM weight sensitivity, which *is* swept in §8.3) — not addressed for
the learned predictor's own hyperparameters.
**Depends on / feeds.** Feeds §8.4's runtime caveat directly.

### 8. Results

#### §8.1 RQ1 — Interpretable Attribution versus Learning
**Argues.** All figures use the single evaluation contract of §7.2. **In-distribution**: typed
learning leads on point estimate (**Table 8**: HGL 0.730 vs Topo-QoS 0.595 mean $\rho$), but HGL's lead
over the *homogeneous* GL baseline is narrow (+0.020) and not uniform (GL wins on AV/Microservices).
**Significance** (**Table 9**, paired Wilcoxon, $n=7$): only HGL vs Topo-BL reaches significance
($p=0.016$); the headline HGL vs Topo-QoS margin does **not** ($p=0.375$). **Out-of-distribution**
(**Table 10**, LOSO): HGL leads both $\rho$ (0.608 vs 0.521) and $F_1@K$ (0.465 vs 0.308); in-domain
k-fold widens this further (0.666 vs 0.492, HGL more stable across folds, $\sigma=0.07$ vs $0.34$).
This comparison is only meaningful **after** repairing Topo-QoS, which had silently been computing
plain unweighted betweenness (a QoS-lookup bug targeting the wrong graph element) — disclosed as one of
six instrument defects (§9.2). **Resolves as a scope condition, not a verdict**: the critical-set
($F_1@K$) advantage is the defensible half; the ranking margin is not.
**Evidence.** Table 8 (in-dist $\rho$, bootstrap CIs, $n$ per scenario), Table 9 (paired Wilcoxon),
Table 10 (LOSO $\rho$/std/$F_1@K$).
**Load-bearing caveats.** **Table 10's underlying run was not persisted to a retained artifact** — the
most recent retained pre-repair log shows a near-tie (Topo-QoS 0.609 vs HGL 0.597), and three changes
are confounded between that run and the reported one (baseline repair, corpus regeneration, cache
rebuild), so the shift cannot be attributed to the repair alone. This is the single largest open
evidentiary gap in the paper and is explicitly named as the first item of outstanding work (outline
front matter, §9.2, Data-availability declaration).
**Reviewer risk.** (1) $n=7$ paired Wilcoxon is underpowered — smallest attainable two-sided $p$ is
0.016, so only a clean 7/7 sweep can ever reach significance; a reviewer versed in nonparametric tests
will likely raise this regardless of the paper's own acknowledgment. (2) The unretained LOSO artifact
is a reproducibility red flag that a rigorous reviewer may treat as disqualifying for that specific
table until re-run. (3) Microservices is the weakest scenario for every learned predictor (HGL 0.362) —
worth understanding why before a reviewer asks.
**Depends on / feeds.** Requires §7.2's evaluation contract and §7.4's oracle/label-ceiling context.
Feeds §9.1's first (headline) finding, §9.2's threats (unretained artifact, instrument defect), and
§9.3's "out-of-distribution ranking is not yet a solved problem" limitation.

#### §8.2 RQ2 — What Taking Node and Edge Type Seriously Shows (and Does Not Show)
**Argues.** **Heterogeneity's advantage is concentrated in generalisation, not in-distribution fit**:
typed-vs-homogeneous gap grows from $+0.020$ in-distribution to $+0.172$ LOSO to $+0.257$ k-fold, with
typed models also far more stable across folds ($\sigma=0.07$ vs $0.15$). **Edge criticality is now
measured, not inferred**: removing each candidate structural relationship on `av_system` (50
candidates), only 4 carry non-zero impact (max $0.00504$, over an order of magnitude below the largest
single-*component* impact); all 46 `RUNS_ON`/`CONNECTS_TO` candidates score exactly zero — meaning the
cascade model **cannot express** infrastructure-layer link failure, not that such links don't matter.
Restates §5.4's two negative results with their oracle caveat, without re-deriving figures.
**Evidence.** Heterogeneity-gap deltas above; edge-removal 4/50 finding; a methodological note that the
removal sweep must run against a freshly-loaded repository (a repository state-ordering hazard
distinct from §5.3's import-level check, not visible elsewhere since Simulate-before-Predict is the
standard pipeline order).
**Load-bearing caveats.** **Explicit scoping caveat**: this section's edge-measurement finding and
§5.4's checks are $I_{\text{comp}}$ results; Tables 8 and 10 are $I^*$ results; the two oracles agree
only at $\rho = 0.394$ (§7.4) — flagged here specifically so textual adjacency does not imply mutual
support between RQ1's tables and RQ2's findings.
**Reviewer risk.** The "cannot express" vs. "does not matter" distinction for `RUNS_ON`/`CONNECTS_TO`
edges is subtle and easy to misread as "infrastructure edges are unimportant" — a reviewer skimming may
draw the wrong conclusion; worth stating the distinction explicitly in any summary or rebuttal.
**Depends on / feeds.** Requires §5.4 and §7.4. Feeds §9.1's third finding (edge criticality) and §9.3
(relationship-level attribution gap).

#### §8.3 RQ3 and Robustness — Ablations and Sensitivity
**Argues.** **QoS encoding (RQ3): null in all three regimes**, with the *sign* of the effect flipping
by protocol (in-dist $+0.001$, OOD $-0.013$, k-fold $+0.027$) — an effect an order of magnitude smaller
than fold-to-fold variance is a null; plausibly because the lifted `DEPENDS_ON` topology already
encodes most QoS-relevant routing. **Intra-dimension weight sensitivity** (**Table 11**, Figure 4): the
composite weights $(w_R, w_M)$ are declared and $\lambda$-invariant; only the $FT$/$A$/$M$ intra-dimension
term weights shrink. $\rho$ rises monotonically in $\lambda$ from uniform intra-dimension weights
($-0.051$ at $\lambda=0$) to the raw AHP judgement ($-0.007$ at $\lambda=1$), with the shipped default
($-0.019$ at $\lambda=0.70$) partway along — no plateau, raw judgement beats uniform weights by
$\approx 0.032$, a small but consistent effect that reverses an earlier version's opposite-signed
finding. **Normalisation**: rank-based (default) $\rho=-0.019$ vs min–max/z-score $-0.035$ — under RM,
retaining magnitude now *costs* $\approx 0.016\,\rho$ rather than gaining it, reversing the earlier
$+0.137$ finding; default retained anyway, now both the strongest of the three and the choice that
keeps prior figures interpretable. **Propagation-threshold**
(**Table 12**): $\rho$ spans 0.230 across the sweep (0.001 at $t=0$ to 0.231 at $t=1$); conclusions *do*
depend on this parameter, which is why remediation (§6.1) requires improvement across the *entire*
sweep rather than trusting a single threshold.
**Evidence.** Table 11 + Figure 4 (shrinkage sweep), Table 12 (threshold sweep), normalisation
comparison inline.
**Load-bearing caveats.** This section is where §4.2's early-stated shrinkage claim gets its full
numeric backing — cross-check that both stay numerically consistent if either is edited. The threshold
sensitivity is explicitly *not* papered over: "we therefore do not claim threshold-independence."
**Reviewer risk.** A reviewer could ask why the rank-normalisation default was ever demonstrably worse
under an earlier version of this sweep — now, under RM, it is the strongest of the three normalisation
choices, which should be stated plainly rather than defended as a debatable trade-off.
**Depends on / feeds.** Table 11 feeds §9.1's second finding and §9.3's "composite weighting is
declared, not fitted" limitation (deriving rather than asserting weights is future work). Table 12
feeds the remediation multi-threshold requirement in §6.1.

#### §8.4 RQ4 — Feasibility and Performance of SaG as a CI/CD Quality Gate
**Argues.** Runtime: $\le 90$ components cost 0.02–0.04 s; 98–326 components cost 0.27–1.24 s, **not
monotonic** in component count (`hub_and_spoke_system` at 139 components costs more than
`iot_smart_city_system` at 326 — detector complexity, not raw count, drives cost); `enterprise_system`
(520 components, the largest) costs $26.74 \pm 0.32$ s — all well under typical CI budgets. **Gate
efficacy is harder to state** since the gate is absolute not delta-aware (§6.2) — no merge-base diff to
evaluate detection against — so the section measures the anti-pattern catalog's raw agreement with the
cascade oracle instead: precision $0.237$/recall $0.887$/$\kappa=-0.036$ on synthetic scenarios,
precision $0.402$/recall $0.861$/$\kappa=0.296$ on real-world graphs. High recall, low precision — the
catalog over-flags, and near-zero $\kappa$ on synthetic means the agreement is close to chance-level.
Remediation yield (162/332, 48.8%) restated here as absorbed content. A sustainability claim (in-memory
evaluation plausibly saves energy vs. staging clusters) is explicitly flagged as **not measured
directly**.
**Evidence.** Runtime figures per scenario size; precision/recall/$\kappa$ by corpus split.
**Load-bearing caveats.** The near-zero Cohen's $\kappa$ on the synthetic corpus is a genuinely weak
result the paper does not minimise — framed as "a genuine limitation of the pattern catalog as a
*stand-alone* predictor," with $Q(v)$ (§4/§8.1) remaining the validated ranking signal.
**Reviewer risk.** Precision 0.237–0.402 for CRITICAL/HIGH findings is low enough that a reviewer may
question RQ4's "feasibility" framing on efficacy grounds even though runtime feasibility is strong — the
paper's own separation of "runs fast" from "flags accurately" pre-empts this but a reviewer may still
push on whether a gate this imprecise is usable in practice. The unmeasured sustainability claim is
explicitly hedged ("plausibly … though we have not measured that comparison directly") — good practice,
but also an easy target ("why include an unmeasured claim at all").
**Depends on / feeds.** Requires §6.2 (gate mechanics), §5.1 ($I_{\text{comp}}$ as the comparison
oracle), §6.1 (remediation yield). Feeds §9.1's fourth finding (least disturbed by the audit, most
confidently defended) and §9.3 (delta-aware gate as engineering work, not open research).

#### §8.5 RQ5 — Real-World Open-Source System Architecture Validation
**Argues.** Three real-world architectures (**Table 13**), five seeds, against the component-level
cascade oracle: $\rho = 0.688$ (Autoware), $0.778$ (Cloud), $0.759$ (Train-Ticket) — strong on two of
three; Autoware misses the framework's own $\rho \ge 0.75$ gate threshold by 0.062 and carries the most
seed-to-seed variance. **All three fail the gate overall on SPOF-F1** ($\ge 0.6$ threshold: Autoware
0.500, Cloud 0.333, Train-Ticket 0.571 — closest, short by 0.029); Cloud additionally fails predictive
gain ($\ge 0.02$: only $0.014$). **5 of 15 total gate checks fail.** $F_1@K=1.000$ on two of three is
**partly a tie-breaking artifact**: $K$ exceeds the non-zero-impact population, so ties at $I=0$ pad
both predicted and actual sets identically under the same stable sort; tie-robust re-sorting (200
shuffles) gives 0.760 (Cloud) and 0.810 (Train-Ticket) — checked and confirmed **not** to affect the
synthetic-suite $F_1@K$ figures of §8.1. The genuine finding is **set containment**: every non-zero-
impact Application falls inside the predicted top-$K$ on two of three graphs. Predictive gain over
degree centrality is real but small and graph-dependent ($+0.360$/$+0.264$/$+0.014$), failing its own
threshold on Cloud.
**Evidence.** **Table 13** (full metrics per graph); tie-robust $F_1@K$ recomputation.
**Load-bearing caveats.** **Four scoping conditions, stated explicitly and all matter**: (1) hand-built
transcriptions, not harvested artifacts — what transfers is topology/QoS structure, not runtime
behaviour; (2) ground truth is still simulated, §9.2's construct-validity bound applies unchanged; (3)
graphs are smaller than five of the seven synthetic scenarios, and per D4 (§4.1) the three $\rho$
values are separate within-system results, not points on a shared scale; (4) **only two paradigms, not
three** — Cloud and Train-Ticket are both microservice meshes, so cyber-physical pub-sub is represented
by Autoware alone.
**Reviewer risk.** "All three fail the gate" alongside "strong rank correlation" is an unusual and
honest combination to report together — a reviewer may either credit this transparency or use the
gate-failure as grounds to discount RQ5 entirely; the paper's own framing ("evidence that predictive
ranking transfers … not a demonstration of production readiness") is the right response to have ready.
The tie-breaking artifact in $F_1@K=1.000$ is exactly the kind of thing a careful reviewer catches
independently — better that the paper discloses it first, which it does.
**Depends on / feeds.** Requires §7.1 (dataset provenance), §5.1 (oracle), D4 (§4.1, scoping). Feeds
§9.1 (not among the five main findings directly, RQ5 is discussed in §9.2's external-validity
subsection instead), §9.2's external-validity assessment ("the weakest dimension of the study"), §9.3's
first-listed (highest-value) limitation.

### 9. Discussion, Threats to Validity, Conclusion

#### §9.1 Interpretation
**Argues.** Five findings synthesising §8: (1) learning pays, most defensibly for set identification —
weight placed on $F_1@K$ over $\rho$ for three stated reasons (practical shortlist size, Topo-QoS's high
variance, the paired-test failure); (2) decomposition is worth having for reasons that are not
accuracy — the stratified-reporting discipline caught the real `av_system` pooling distortion even
though it didn't find the Simpson's-paradox effect it was designed to catch; (3) edge criticality
benefits from being measured — the heuristic's intuition was reversed by direct measurement; (4)
remediation is now verified per edit, a stronger guarantee than before but not yet a value
demonstration; (5) CI/CD gating operationalises the checks continuously — explicitly **corrected** here
to state the gate is absolute, not delta-aware (an earlier revision's §9.1/§9.4 both mis-described it
as delta-aware, contradicting §1.3/§6.2/§9.3; fixed in this draft).
**Evidence.** Synthesises numbers already established in §8; no new figures.
**Load-bearing caveats.** The explicit self-correction note (absolute-not-delta-aware, previously
mis-stated) is a signal of careful revision discipline worth preserving in any summary — it demonstrates
the paper catching and fixing its own internal contradiction.
**Reviewer risk.** Low for this subsection itself; it is a synthesis, and the risks it discusses were
already flagged at their source sections.
**Depends on / feeds.** Synthesises §8.1–§8.5; feeds directly into §9.3's ordered limitations list.

#### §9.2 Threats to Validity
**Argues.** **Construct validity**: the D1/D2 (§4.1) Quality-in-Use construct is never directly
observed. The chain has **three** links, matching the three quality views of §4 — ① internal quality
evidence → simulated external quality, which *is* measured; ② simulated → real (deployed) external
quality, not measured, since the simulator is a model of the executing system rather than the system;
and ③ external quality → Quality-in-Use loss, not measured at all. Naming the middle view corrects an
error in both directions: the earlier two-link framing *understated* link ① (the delivery-rate oracle
$I_{\text{dyn}}$ is built and reported, not prospective) while *overstating* the construct (delivery
rate and latency are external product-quality measures, not Quality-in-Use measures). Freedom from
risk is corpus-blocked (0 of 710 topics declare a deadline) and would still be external when
unblocked; Satisfaction is not measurable by these means at all. The two oracles' weak agreement ($\rho=0.394$)
bounds transferability between $I_{\text{comp}}$- and $I^*$-backed results. **Six silent instrument
defects** disclosed in full: Topo-QoS's QoS-lookup bug (repaired, Tables 8/10 recomputed); HGT
attention silently capturing nothing due to a missing PyG argument (repaired, Figure 3 now real);
`FaultInjector`'s `PYTHONHASHSEED`-dependent iteration order (repaired, verified across five hash
seeds); a training-target lookup keying on a nonexistent attribute (repaired); a parallel remediation
worker ignoring configured layer/checkpoint (repaired, reported Table 4 run unaffected — see
[Draft inconsistencies](#draft-inconsistencies-recorded-not-fixed) on this number); a stale post-loop
variable read, a no-op under the setting every reported figure uses (repaired). **Only the third defect
changes reported figures** — retroactively bearing on the §7.4 test–retest ceiling and §8.5's Autoware
row. **Internal validity**: view independence (§3.3/§5.3) is *not* source independence — both graph
views are deterministic functions of the same input topology, so feature–label feedback is ruled out
but not a shared modelling assumption; $I_{\text{dyn}}$ narrows but does not eliminate this concern.
Two further pre-submission-audit findings: the population-mismatch bug (§7.2, changed RQ1's conclusion
sign) and a stale-checkpoint LOSO bug caught only by implausible wall-clock timing (3.2 s dirty vs.
322 s clean for the "same" run). **Artifact retention is uneven**: Table 8 and §8.3's sweeps regenerate
exactly from stored files; **Table 10's LOSO artifact does not exist** and is the one outstanding item.
**External validity**: explicitly named **the weakest dimension** — only three architectures are not
the authors' own generator, spanning two paradigms not three, none clears the framework's own gate.
**Conclusion validity**: heavy-tailed/non-parametric distributions justify the paper's consistent use
of Spearman $\rho$, Jaccard, and box-plot thresholds over parametric alternatives.
**Evidence.** All figures restated from their originating sections; no new numbers introduced here.
**Load-bearing caveats.** This is the section that makes every other section's caveats *legible as a
system* — a reader who reads only §9.2 gets an accurate, if compressed, map of every scoping condition
in the paper. The "six defects, only one changes figures" framing is the load-bearing distinction —
conflating "found and fixed" with "changed a reported number" would misrepresent the audit's actual
impact.
**Reviewer risk.** This is simultaneously the paper's strongest credibility asset (unusually thorough
self-audit) and its biggest single reviewer-risk concentration point — a reviewer who reads only this
section may conclude the paper is less reliable than it is, since every caveat is gathered in one place
without the balancing evidence sitting alongside it in §8. Recommend any reviewer response draw
explicitly on §8's positive results when responding to concerns raised against this section.
**Depends on / feeds.** Synthesises threats from every prior section; the two headline items (unretained
Table 10, weak-agreement oracles) are the ones most likely to recur in reviewer comments and are already
listed as the first item of outstanding work.

#### §9.3 Limitations and Future Work
**Argues.** Six limitations, **explicitly ordered by how much they would change the paper's claims**:
(1) real-world/HIL validation — highest-value follow-up; (2) OOD ranking not yet solved — Topo-QoS
matches learned models on LOSO $\rho$; (3) dimension weighting doesn't improve accuracy — repositioned
as attribution, a *derived* (not asserted) weighting is future work; (4) remediation verified but not
yet demonstrably effective — subset verification and broader $\kappa$-derivation needed; (5) CI/CD gate
absolute not delta-aware — framed as engineering work, not open research; (6) relationship-level
attribution defined but unvalidated, closing it needs a real many-to-many modelling decision. Closes
with calibration against observed failure data as "the endpoint for all of the above."
**Evidence.** No new numbers; restates and orders prior sections' open items.
**Load-bearing caveats.** The explicit ordering-by-impact is itself a claim worth preserving verbatim
if this section is edited — it signals the authors' own priority ranking, which a reviewer or advisor
may want to interrogate directly ("why is real-world validation ranked above the weighting-accuracy
gap?").
**Reviewer risk.** A reviewer may propose a different ordering (e.g., ranking the CI/CD gate's
production-readiness gap higher, since it's the paper's most operationally-facing contribution) — worth
having a justification ready for the stated order.
**Depends on / feeds.** Directly restates open items from §8.1 (item 2), §4.2/§8.3 (item 3), §6.1 (item
4), §6.2 (item 5), §4.1 (item 6), §8.5/§9.2 (item 1).

#### §9.4 Conclusion
**Argues.** Closing synthesis: SaG as typed multigraph + hierarchical RM attribution + HGT-based failure-impact
prediction, validated under the independence guarantee; remediation with the 162/332 acceptance
result; CI/CD gate (absolute, not delta-aware) running in seconds; the scope-condition framing for RQ1
restated one final time ($\rho=0.608$ vs $0.521$, $F_1@K=0.465$ vs $0.308$, both after baseline repair);
edge-criticality and stratified-reporting findings restated briefly. Closes on the framework's core
thesis: taking type seriously recovers structure untyped, single-dimensional methods discard, at the
point in the lifecycle where it is most valuable.
**Evidence.** Restates headline figures already established; no new numbers.
**Load-bearing caveats.** None new — this section's job is fidelity to what §1–§8 actually established,
and it holds that discipline (e.g., still qualifying the RQ1 figures as post-repair, still stating
"absolute rather than delta-aware").
**Reviewer risk.** Low if the rest of the paper is internally consistent; any residual inconsistency
(e.g., the "four contributions"/"four research questions" slips flagged above) would be most visible if
it propagated into this closing section, and it does not appear to.
**Depends on / feeds.** N/A — terminal section.

### References and Declarations
**Argues.** 53 numbered entries + `[Anon-A]` (withheld for anonymised review), unchanged from the full
draft. Declarations block: CRediT (stub, pending de-anonymisation), competing interest (none declared),
funding (stub), data availability, generative-AI use (stub).
**Evidence.** N/A — bibliographic and administrative.
**Load-bearing caveats.** **Data availability explicitly names the Table 10 exception**: the LOSO
per-fold artifact "was not retained and cannot be regenerated from the archive without re-running the
sweep" — this is the same gap flagged in §8.1 and §9.2, stated a third time here because a data-
availability statement asserting full reproducibility while one table is not reproducible would be a
serious integrity problem if left unstated.
**Reviewer risk.** JSS reviewers and editors check data-availability statements closely; the explicit
Table 10 carve-out here is the correct and necessary disclosure — do not let a future edit quietly drop
it while "cleaning up" the declarations.
**Depends on / feeds.** Cross-references §8.1's and §9.2's Table 10 discussion; must stay in sync with
whichever table numbering is current if the paper is re-condensed or re-expanded again.

---

## Table and figure inventory

13 tables, 4 figures — each owned by exactly one subsection. Useful as a checksum against renumbering
during the pending LaTeX re-conversion (see [Appendix](#appendix-condensation-history)).

| # | Owning section | Content |
|:--:|---|---|
| Table 1 | §3.1 | Node and structural edge types |
| Table 2 | §3.2 | Six `DEPENDS_ON` projection rules |
| Table 3 | §4.1 | Two RM dimensions (Reliability's FT/A sub-terms broken out), question, remediation owner |
| Table 4 | §6.1 | Four remediation operators, trigger, edit, failure mode |
| Table 5 | §7.2 | Predictors/baselines and the factor each isolates |
| Table 6 | §7.4 | Three oracles, quantity measured, results each backs |
| Table 7 | §7.5 | Learned-predictor hyperparameters |
| Table 8 | §8.1 | In-distribution held-out Spearman $\rho$ vs $I^*(v)$ |
| Table 9 | §8.1 | Paired Wilcoxon signed-rank tests ($n=7$) |
| Table 10 | §8.1 | Inductive LOSO evaluation (**artifact not retained**) |
| Table 11 | §8.3 | AHP shrinkage sensitivity |
| Table 12 | §8.3 | Propagation-threshold sensitivity |
| Table 13 | §8.5 | Real-world architecture validation |
| Figure 1 | §1.3 | End-to-end SaG pipeline diagram |
| Figure 2 | §3.3 | Running example: structural graph + `DEPENDS_ON` projection |
| Figure 3 | §5.2 | Learned relation-specific attention case study |
| Figure 4 | §8.3 | Mean $\rho$ vs. shrinkage $\lambda$ |

All four figures exist in `draft.md` only as inline prose placeholders (`*(Figure N: …)*`); the
rendered assets are [`latex/figures/`](latex/figures/) `Figure_1.{pdf,png}` … `Figure_4.{pdf,png}`.

## RQ traceability

| RQ | Question (abbreviated) | Primary section | Primary evidence | Standing caveat |
|:--:|---|---|---|---|
| RQ1 | Where does typed learning improve on non-learning baselines? | §8.1 | Tables 8–10 | Ranking margin fails paired test in-dist; LOSO artifact unretained |
| RQ2 | What does taking type seriously expose/fail to expose? | §8.2 | Heterogeneity-gap deltas; 4/50 edge removal | Restated results are $I_{\text{comp}}$-scoped, don't transfer to $I^*$ tables |
| RQ3 | Does QoS feature injection help convergence vs. generalisation? | §8.3 | Table 11 (Figure 4), Table 12 | Null result, sign flips by protocol |
| RQ4 | Feasibility/overhead of the CI/CD gate? | §8.4 | Runtime figures; precision/recall/$\kappa$ | Gate is absolute not delta-aware; catalog $\kappa\approx0$ on synthetic |
| RQ5 | Does predictive ranking transfer to unseen architectures? | §8.5 | Table 13 | All three fail the gate; only two paradigms represented |

---

## Appendix: Condensation history

*(This appendix preserves what earlier revisions of this file tracked as the primary content. It is
now background, not the reason to open this file.)*

**This is the condensed submission draft.** `draft.md` was cut from ~30,100 words (23 tables, 6
figures) to ~16,800 words (13 tables, 4 figures) to fit JSS's ≤36-single-column-page guidance,
refocused on the paper's graph-learning-and-dependability claim (hierarchical Reliability/
Maintainability (RM) attribution is now a baseline the results are read against, not a co-equal
contribution). Nothing was deleted outright: the pre-condensation text and every cut subsection are
preserved verbatim in **[`../thesis/`](../thesis/)** — see that folder's `README.md` for the
section-by-section map of what moved where and why. `../thesis/jss_draft_full.md` is a frozen,
byte-identical snapshot of `draft.md` from immediately before the condensation pass and, per its own
`README.md`, is not edited in place — it (and the extracted material files below) still use the
retired four-dimension "RMAV" terminology this migration replaced in `draft.md` itself, since they
predate the RM migration and are preserved as historical record, not updated to track it. The six
extracted material files, all confirmed present, are:

- [`../thesis/material/model_details.md`](../thesis/material/model_details.md) — full `cm_*` metric
  list and QoS weight formulas cut from §3.1/§3.2.
- [`../thesis/material/rm_attribution.md`](../thesis/material/rm_attribution.md) — full RMAV (as
  originally derived, pre-migration) formula derivation, three-weighting-paths account, Quality-in-Use
  transformation matrix, worked example, cut from §4.
- [`../thesis/material/relationship_criticality.md`](../thesis/material/relationship_criticality.md)
  — relationship (edge-level) criticality, cut entirely from §4 since it is not validated by the
  edge-removal measurement.
- [`../thesis/material/oracles_and_labels.md`](../thesis/material/oracles_and_labels.md) — full
  original §5.4/§5.5 treatment, including the dropped pooled-vs-per-type correlation figure.
- [`../thesis/material/remediation_and_gating.md`](../thesis/material/remediation_and_gating.md) —
  full original seven-subsection §6 treatment, including the remediation yield table.
- [`../thesis/material/threats_and_instrument_defects.md`](../thesis/material/threats_and_instrument_defects.md)
  — full six-defect account, including how each was found, condensed to one paragraph in §9.2.

**Outcome of the condensation pass:**
1. Length: 30,106 → ~16,800 words (body ~15,300, references 1,280, declarations 220), 23 → 13 tables,
   6 → 4 figures. Estimated 32–35 single-column pages once typeset — to be confirmed against a real
   build (see LaTeX status below).
2. **Table 10's artifact is still not retained** — re-running the LOSO sweep under the final apparatus
   is the first item of outstanding work (§8.1, §9.2). Unchanged since condensation.
3. Author block, CRediT roles, funding, and the archive link remain stubs pending de-anonymisation —
   see [`latex/title_page.tex`](latex/title_page.tex) and
   [`latex/sections/declarations.tex`](latex/sections/declarations.tex).

**LaTeX conversion status.** The submission-ready Elsevier `elsarticle` source lives in
[`latex/`](latex/). **It has not yet been re-converted from this condensed `draft.md`** —
`latex/sections/*.tex` still reflect the pre-condensation, 23-table/6-figure version, and the
previously-recorded 75-page build (under the double-spaced `review` class option) describes that older
text, not this one. See [`latex/README.md`](latex/README.md) for build instructions and what remains a
placeholder pending de-anonymisation. Re-converting the sections against this draft, and switching to a
single-spaced class option to check the real page count, is the next step before submission-ready.

## Outstanding work

1. Re-run the LOSO sweep under the final apparatus and retain the artifact — the one item that would
   settle RQ1's ranking half (§8.1, §9.2).
2. Re-convert `latex/sections/*.tex` from the condensed draft; rebuild single-spaced for a real page
   count.
3. De-anonymise `latex/title_page.tex` and `latex/sections/declarations.tex`.
4. Resolve the draft inconsistencies below (editorial fixes to `draft.md`, not yet made).

## Draft inconsistencies recorded, not fixed

Found while building this map; recorded here for whoever next edits `draft.md`, but `draft.md` itself
is unchanged — none of these were in scope for this rewrite.

- [draft.md:122](draft.md#L122) — "organized around **four** research questions," immediately followed
  by RQ1–RQ5.
- [draft.md:661](draft.md#L661) — §7 states it supports "RQ1–**RQ4**"; §8 answers RQ1–RQ5.
- [draft.md:1204](draft.md#L1204) — §9.1 refers to "of the **four** contributions"; §1.4 lists six.
- [draft.md:1102](draft.md#L1102) vs. [draft.md:1136](draft.md#L1136) — Autoware predictive gain is
  `+0.360` in Table 13 but `+0.361` in the surrounding prose.
- [draft.md:1241](draft.md#L1241) — "the reported **Table 4** run is unaffected" reads as leftover
  pre-condensation numbering; Table 4 is now the remediation-operators table, and the run actually
  meant is §6.1's remediation sweep.
