# Software-as-a-Graph: Heterogeneous Graph Learning for Pre-Deployment Reliability and Dependability Analysis of Complex Distributed Systems

> **`draft.md` is the authoritative manuscript text.** This outline is a section-by-section reading
> map of it — what each subsection argues, what backs the claim, what caveats travel with it, where a
> reviewer is likeliest to push, and what it depends on or feeds elsewhere. Where the two disagree,
> `draft.md` wins; this file is regenerated from it, not the other way around.
>
> **`latex/` is now in sync with `draft.md`** (8 sections, 9 tables), which was not true when this
> file was last written. [`latex/sections/`](latex/sections/) still contains four *orphan* `.tex`
> files from the retired 9-section structure that `manuscript.tex` no longer `\input`s — see
> [Outstanding work](#outstanding-work).

* **Target Journal:** Journal of Systems and Software (JSS) — Elsevier
* **Target Venue:** Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems" (VSI:AI4MSS); submission deadline 30 September 2026
* **Target Topic:** *AI for Reliability and Dependability Analysis in Complex ICT Systems* — with a secondary claim on *Explainable, Interpretable, and Robust AI* (the RM attribution half)
* **Review model:** double-anonymised (`[Anon-A]` withheld; repo paths and tool names scrubbed)
* **Scale:** ~11,100 words, 8 sections, 9 tables, 4 nominal figures (1 rendered — see inventory), 59 numbered references + `[Anon-A]`

### Title rationale

The previous title, *"Graph Learning for Reliability and Dependability Analysis in Complex Distributed
Systems"*, restated the CFP topic bullet almost verbatim and named neither the artifact nor the
novelty. The current title keeps the CFP phrase **"Reliability and Dependability Analysis"** for
special-issue fit while adding the three things that make the paper distinct:

| Element | Why it is in the title |
|---|---|
| **Software-as-a-Graph** | Names the artifact; every section refers back to it |
| **Heterogeneous** | Signals RQ2, the paper's strongest and most defensible result (§7.2) |
| **Pre-Deployment** | The governing constraint — no telemetry exists yet (§1.1), which is what separates this from chaos engineering and from every runtime-trace approach in §2.1 |

The title must stay byte-identical across [`draft.md`](draft.md#L1),
[`latex/manuscript.tex`](latex/manuscript.tex#L30), and
[`latex/title_page.tex`](latex/title_page.tex#L9).

## Highlights

*(verbatim from [`latex/highlights.tex`](latex/highlights.tex) — the submitted artifact; this list is
the one to keep in sync if either file changes)*

* Software-as-a-Graph models distributed pub-sub and microservices architectures as typed multigraphs.
* Heterogeneous Graph Transformers predict cascading failure impacts before deployment.
* Relation-specific graph learning provides significant gains in critical-set identification and cross-architecture generalization.
* Interpretable Reliability-Maintainability attribution disentangles error propagation from single points of failure.
* Empirical validation on real-world systems (ROS 2 Autoware, Cloud-Native, Train-Ticket) confirms zero-shot transferability.

> ⚠ Bullet 3's word *"significant"* is not supported as a statistical claim: HGL's ranking margin over
> Topo-QoS fails the paired Wilcoxon test in-distribution ($p = 0.469$, Table 6). Reword at the
> highlights pass — see [Draft inconsistencies](#draft-inconsistencies-recorded-not-fixed) item 7.

## Keywords

*(verbatim from [`latex/manuscript.tex`](latex/manuscript.tex#L40-L48); `draft.md`'s keyword line matches)*

Graph representation learning; heterogeneous graph neural networks; publish–subscribe middleware;
distributed systems dependability; cascading failures; static system analysis; architectural quality
models.

---

## Headline figures (as reported in the draft)

| Quantity | Value | Section |
|---|---|---|
| In-distribution mean $\rho$ — HGL / GL / Topo-QoS / Topo-BL | 0.725 / 0.612 / 0.586 / 0.209 | §7.1 (Table 5) |
| In-distribution mean $\rho$ — HGL-QoS / GL-QoS | 0.653 / 0.433 | §7.1 (Table 5) |
| Paired Wilcoxon, HGL vs Topo-BL ($n=7$) | $\Delta\rho = +0.516$, $p = 0.0156$, **significant** | §7.1 (Table 6) |
| Paired Wilcoxon, HGL vs GL-QoS | $\Delta\rho = +0.292$, $p = 0.0312$, **significant** | §7.1 (Table 6) |
| Paired Wilcoxon, **HGL vs Topo-QoS** | $\Delta\rho = +0.139$, $p = 0.469$, **n.s.** | §7.1 (Table 6) |
| Paired Wilcoxon, HGL-QoS vs HGL | $\Delta\rho = -0.072$, $p = 0.078$, n.s. (marginal) | §7.1 (Table 6) |
| LOSO mean $\rho$ — HGL-QoS / HGL / Topo-QoS / GL-QoS / GL / RM / Topo-BL | 0.668 / 0.578 / 0.522 / 0.501 / 0.381 / −0.014 / 0.109 | §7.1 (Table 7) |
| LOSO $F_1@K$ — HGL / Topo-QoS | 0.467 vs 0.339 (**+37.8 %** relative) | §7.1 (Table 7) |
| Heterogeneity gap, LOSO (HGL − GL) | $+0.197$; $\sigma$ 0.116 vs 0.164 | §7.2 |
| Heterogeneity gap, LOSO (HGL-QoS − GL-QoS) | $+0.167$ | §7.2 |
| Edge removal, `av_system` | 4 of 50 candidates non-zero, all library channels | §7.2 |
| QoS encoding, in-dist vs LOSO | $-0.072$ vs $+0.090$ — **sign flips by protocol** | §7.3 |
| AHP shrinkage, $\lambda=0 \to 1$ | $-0.0514 \to -0.0069$ ($+0.045$), **negative throughout** | §7.3 (Table 8) |
| Oracle agreement, $I_{\text{dyn}}$ vs $I^*$ | $\rho = 0.765$ (range 0.550–0.938), Jaccard 0.312 | §5.1, §7.3 |
| Oracle agreement, $I_{\text{comp}}$ vs $I_{\text{dyn}}$ | $\rho = 0.465$ (range 0.121–0.658) | §5.1, §7.3 |
| Oracle agreement, $I_{\text{comp}}$ vs $I^*$ | $\rho = 0.397$ (range 0.097–0.578) | §5.1, §7.3 |
| Domain reweighting | $+0.026$ vs static default, **$-0.110$ vs equal split**; $\tau = 0.968$ | §7.3 |
| Threshold sweep | $\Delta\rho = 0.190$ ($-0.173$ at $t{=}0$ → $+0.017$ at $t \ge 0.35$) | §7.3 |
| Normalisation sweep | $\Delta\rho = 0.016$ across three schemes | §7.3 |
| Anti-pattern catalog | $F_1 = 0.378$, precision 0.239 / recall 0.900, flags 94.2 % of components | §7.3 |
| Detector runtime (18 of 19) | 0.04 s (50 nodes) → 54.85 s (300 nodes); `DEEP_PIPELINE` excluded (>10 min) | §7.3 |
| RM stratified $\rho$ | 0.503 App / 0.395 Broker / 0.142 Node, **pooled 0.028** (Simpson's) | §7.3 |
| Real-world $\rho$ (Autoware / Cloud / Train-Ticket) | 0.688 / 0.778 / 0.759 | §7.4 (Table 9) |
| Real-world predictive gain vs degree | $+0.360$ / $+0.014$ / $+0.264$ | §7.4 (Table 9) |
| Corpus | 1,770 components across 10 architectures (7 synthetic + 3 real-world) | §6.1 (Table 4) |

**Two standing caveats that travel with every figure above.**

1. **RQ1's in-distribution ranking margin is not established.** HGL beats Topo-QoS on the point
   estimate ($0.725$ vs $0.586$) but fails the paired test ($p = 0.469$, 5/7 scenarios won). The two
   defensible halves of RQ1 are the **critical-set gain** ($F_1@K$ $0.467$ vs $0.339$) and the
   **LOSO gain** ($0.668$ vs $0.522$). Any abstract, highlights, or conclusion sentence that reads
   "graph learning outperforms structural baselines" without one of those two qualifiers is
   overclaiming against the paper's own Table 6.
2. **Results are oracle-scoped and do not transfer between oracles.** $\rho(I_{\text{comp}}, I^*) =
   0.397$ with top-$K$ Jaccard 0.26–0.31. Findings measured against $I_{\text{comp}}$ (§7.2's
   edge-removal analysis, §7.3's anti-pattern catalog) are **not** evidence for claims measured
   against $I^*$ (Tables 5–7, 9). §5.1 and §8.2 both state this correctly — keep it that way.

A third, weaker caveat worth carrying: **the RM baseline is anti-correlated with $I^*$ at every
$\lambda$** (Table 8) and non-predictive under LOSO ($\rho = -0.014$). The paper handles this honestly
by repositioning RM as attribution rather than ranking, but the word "baseline" in §4's title invites
a reviewer to read it as a competing predictor. That framing is load-bearing and fragile.

---

## Section-by-section map

Each entry: what the subsection argues, the evidence that carries it, the caveats/definitions that
are load-bearing there, where a reviewer is likeliest to push, and its dependencies on other sections.

### 1. Introduction

#### §1.1 Motivation
**Argues.** Pub-sub decoupling (producers/consumers separated in time, space, synchronisation) is what
makes the paradigm scale and exactly what hides the dependency chains a failure propagates along —
there are no caller–callee edges, so a diagram's apparent importance and a component's actual blast
radius diverge. Introduces the shared-library *simultaneous failure* mode as distinct from a
sequential cascade. Frames pre-deployment as when hardening is cheapest and, simultaneously, when no
telemetry exists. Closes with the two questions the architect must answer from design-time descriptors
alone (*which* components are critical, and *why* / what intervention follows).
**Evidence.** Prose only, framed against DDS [2], MQTT [3], Kafka [43], ROS 2 [44] and pub-sub theory [1].
**Load-bearing caveats.** None yet — this section motivates rather than claims. The sequential-vs-
simultaneous distinction introduced here is what §3.2 formalises and §7.2's edge-removal result later
corroborates.
**Reviewer risk.** Low. The one soft spot is the single-sentence sustainability framing (emergency
failovers, retransmission storms, restart loops waste energy) — it is the paper's *only* engagement
with the special issue's sustainability theme, and it is unquantified. A guest editor screening for
theme fit may find that thin; the reliability/dependability bullet is where this paper actually lands.
**Depends on / feeds.** Sets up the "architecture alone, no telemetry" constraint that governs every
later oracle and predictor design (§3, §5).

#### §1.2 Problem Statement and Limitations of Existing Work
**Argues.** Two coupled tasks: (1) *interpretable criticality attribution* grounded in ISO/IEC
25010 [16] and 25019 [17], explaining *how* and *why*; (2) *failure-impact forecasting* — predicting
blast radius and ranking components for budget-constrained hardening. Names the **Architecture–Code
Gap**: pristine per-service code can still sit inside a fragile topology. Three prior strands each
leave the gap open — SCA (blind to topology), runtime chaos engineering (needs a running system, too
late in the lifecycle), homogeneous/topology-only network analysis (*dimensional collapse* and
*semantic collapse*).
**Evidence.** Prose positioning only; no numbers.
**Load-bearing caveats.** "Both must be computed without runtime data" is the constraint §5.1's oracles
and §4's deterministic $Q(v)$ are built to satisfy.
**Reviewer risk.** The closing sentence — "No existing approach provides a unified, pre-deployment
framework that combines typed multigraph representations, code-level quality ingestion, heterogeneous
graph learning, and interpretable quality attribution" — is a four-conjunct novelty claim. Conjunctive
novelty claims are the easiest kind to accept and the easiest kind to dismiss as engineering
integration; §2 must carry the weight, and it is short (see §2's entry).
**Depends on / feeds.** Directly motivates the four contributions of §1.5 and the four-paragraph
structure of §2.

#### §1.3 The Software-as-a-Graph (SaG) Approach
**Argues.** Introduces SaG's shape end-to-end as a four-stage pipeline: typed multigraph formulation
(§3.1) → QoS-aware logical dependency projection via six rules (§3.2) → interpretable RM attribution
(§4) → HGT-based failure forecasting (§5). States the **input–label independence guarantee**:
predictors read $G_{\text{analysis}}$, oracles execute on $G_{\text{structural}}$ (§5.4).
**Evidence.** Figure 1 (pipeline). In `draft.md` this is an inline ASCII block with a caption; in
LaTeX it is the only actually-rendered figure ([`figures/Figure_1.pdf`](latex/figures/Figure_1.pdf),
`sec1_introduction.tex:72-76`).
**Load-bearing caveats.** The independence guarantee is the paper's answer to the leakage question and
is enforced in the codebase by `tests/test_independence_guarantee.py`. It is stated three times
(§1.3, §5.4, §8.2) — deliberately, not redundantly.
**Reviewer risk.** Low; this is the clearest section in the paper.
**Depends on / feeds.** Every subsequent section is a expansion of one of the four stages.

#### §1.3.1 Why a Predictor Rather Than the Oracle
**Argues.** Confronts the obvious objection head-on rather than deferring it: **simulation alone
suffices to identify critical components**, given a complete model and enough compute to sweep it. No
predictor can exceed its own oracle on the oracle's terms, and none is claimed to — every correlation
reported is *surrogate fidelity*, not accuracy against observed production failure. Learning is
motivated instead by three properties of the oracle: it is **incomplete by construction** (the cascade
model cannot express direct Topic or Node failure, so 30–47 % of components per scenario carry no
ground truth and are excluded rather than scored zero); it is **one draw from a family of disagreeing,
parameter-sensitive simulators** (per-node label $\sigma$ reaches 0.416 on a $[0,1]$ target, and the
propagation threshold is a free parameter *of the ground truth*); and its **cost falls in the wrong
place** (sweeping the graph as it stands is affordable; sweeping the space of candidate repairs is
not). Attribution is defended separately: simulation returns a *magnitude*, not a *reason*.
**Evidence.** The 30–47 % unlabelled figure, the 0.416 label $\sigma$, forward-references to §5.1's
oracle disagreement.
**Load-bearing caveats.** This subsection is the paper's single most important defensive move and is
what makes RM's weak standalone ranking (§7.1, §7.3) non-fatal. Preserve it verbatim in any
condensation pass.
**Reviewer risk.** Paradoxically *lowers* risk by pre-empting Reviewer 2's first question. The residual
risk: the cost argument's payoff (cheap proposal + expensive verification for remediation) points at
machinery that **was cut from this submission** — the remediation section no longer exists. The
third justification therefore now argues for something the paper does not deliver. Consider trimming
it to the first two, or re-scoping it as future work aligned with §8.3 item 3.
**Depends on / feeds.** §5.1 (oracle family), §4 (attribution defence), §7.1/§7.3 (why RM's $\rho$ is
not the verdict on RM).

#### §1.4 Research Questions
**Argues.** Four RQs: RQ1 predictive efficacy vs non-learning baselines, in-distribution and inductive;
RQ2 what typed heterogeneity reveals that homogeneous representations obscure; RQ3 ablations and
sensitivity (QoS encoding, weighting calibration); RQ4 real-world transfer across paradigms.
**Evidence.** N/A — statement of scope.
**Load-bearing caveats.** RQ1 is phrased as *"how accurately … compared to"*, which is what lets §7.1
land on a scope condition rather than a verdict.
**Reviewer risk.** **RQ3 as stated under-covers its own results section.** It promises "explicit QoS
feature encodings and multi-attribute weighting calibrations", but §7.3 additionally delivers oracle
convergent validity, threshold/normalisation sensitivity, anti-pattern detection with CI/CD runtimes,
a Simpson's-paradox stratification, and an attention case study. Either widen RQ3's wording or
redistribute §7.3's contents — see [Draft inconsistencies](#draft-inconsistencies-recorded-not-fixed)
item 10.
**Depends on / feeds.** RQ1→§7.1, RQ2→§7.2, RQ3→§7.3, RQ4→§7.4 (traceability table below).

#### §1.5 Key Contributions
**Argues.** Four contributions: (1) formal typed architecture model distinguishing sequential cascade
from simultaneous blast (§3); (2) relation-specific HGT for pre-deployment cascade prediction (§5);
(3) standards-grounded interpretable RM attribution bridging SCA and topology (§4); (4) empirical
evaluation with scope conditions across 7 synthetic + 3 real-world systems under strict input–label
independence (§6–§7).
**Evidence.** Pointers into the body; no inline numbers (a change from the retired six-contribution
list, which quoted headline figures inline).
**Load-bearing caveats.** Contribution 4's phrase "**establishing where** typed graph learning
provides decisive advantages" is the scope-condition framing — the paper's own guard against
overclaiming RQ1. Keep it.
**Reviewer risk.** Contribution 3 claims RM "bridges code-level SCA metrics with system-level
topological criticality"; the evidence for the *bridge* is that CQP is one of five terms in $M(v)$ at
weight 0.15 (§4.3) — and per the repo's own gotcha table, `cm_avg_cbo` / `cm_avg_rfc` are ingested but
never scored. A reviewer who reads §4.3 closely will find the SCA contribution thinner than the
contribution statement implies.
**Depends on / feeds.** Compressed pointers into §3–§7.

#### §1.6 Paper Organization
**Argues.** Standard roadmap, §2 through §8.
**Evidence.** N/A.
**Load-bearing caveats.** **The retired §1.5 "Prior Work and Organization" is gone**, and with it the
statement positioning this paper against the authors' prior work `[Anon-A]` and confirming no
companion manuscript is under parallel review. `[Anon-A]` now appears in the reference list
(`draft.md:761`) but is **cited nowhere in the body**. For a special issue that explicitly requires
disclosure of previously published related material and ≥30 % new content for conference extensions,
this is a compliance gap, not a copyediting nit — see [Outstanding work](#outstanding-work).
**Reviewer risk.** Editorial-desk risk rather than reviewer risk, which is worse.
**Depends on / feeds.** N/A.

### 2. Related Work

**Argues.** Four strands. **§2.1 Dependability in distributed/pub-sub systems** — prior work is
predominantly *runtime* (fault-tolerant routing, broker clustering, consensus, retransmission) or
chaos-engineering [18]; both need deployed infrastructure and arrive too late to guide design.
**§2.2 SCA vs SSA** — SCA [28, 29, 30, 55–58] is blind to IPC, container placement, and middleware
topology; **Static System Analysis** elevates it to architecture level, propagating code-level quality
across topological links for CI/CD-time verification [19–27]. **§2.3 Network science and graph
representation learning** — centrality [4, 5, 37, 38] and cascade/interdependent-network theory
[6, 35, 36] suffer *dimensional collapse* (one scalar cannot separate a SPOF from a cascade hub from a
maintenance bottleneck) and *semantic collapse* (a topic, a host and a library become the same node);
learned dismantling (FINDER [7], DrBC [8], PowerGraph [9]) mostly uses homogeneous message passing
(GCN [39], GraphSAGE [40], GAT [41]), whereas heterogeneous GNNs (RGCN [10], HAN [11], HGT [12],
MAGNN [13]) parameterise relation-specific transforms — HGT [12] is the chosen substrate.
**§2.4 Quality models and MCDM** — ISO/IEC 25010:2023 [16], ISO/IEC 25019:2023 [17], the
internal/external quality distinction [53, 59], and AHP [15] with its $CR \le 0.10$ consistency audit.
**Evidence.** Bibliographic; no numbers.
**Load-bearing caveats.** The *dimensional collapse* / *semantic collapse* pair is the analytical
frame the whole paper rests on: dimensional collapse justifies §4's multi-dimensional RM profile,
semantic collapse justifies §5's typed HGT. RQ2's result (§7.2) is the empirical test of the second;
**there is no equivalent empirical test of the first** — RM's dimensional decomposition is never shown
to beat a scalar at anything measurable, only to be *diagnostically* richer.
**Reviewer risk.** Highest structural risk in the front half. At ~6 KB of `.tex` this is a short
related-work section for JSS, and it must substantiate §1.2's four-conjunct novelty claim. Two
concrete gaps a reviewer may name: (a) no engagement with architecture-level dependability analysis in
the ADL / AADL / error-annex tradition, which is the closest prior art to "predict failure propagation
from an architecture model with no telemetry"; (b) no engagement with microservice dependency-graph
and failure-diagnosis literature, which is where a JSS reviewer's own citations most likely live. Also
note the special issue is explicitly **not** accepting surveys — so expanding §2 must add positioning,
not coverage.
**Depends on / feeds.** §2.3 chooses HGT [12] for §5.2; §2.4 supplies the standards and AHP machinery
of §4.

### 3. The Software-as-a-Graph (SaG) Architectural Model

#### §3.1 Formal Multigraph Definition
**Argues.** $\mathcal{G} = (V, E, \tau_V, \tau_E, w_V, w_E)$ over five disjoint entity types
(Application, Broker, Topic, Node, Library) and six structural edge types (`PUBLISHES_TO`,
`SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`). Application and Library entities
additionally ingest `cm_*` SCA attributes, which is the mechanical form of the Architecture–Code
bridge.
**Evidence.** Table 1 (two sub-tables: entity types, structural edge types), each with concrete
middleware examples.
**Load-bearing caveats.** The five-type partition is the unit of "heterogeneity" that RQ2 tests. The
edge-type list here is the normative one — §5.2 contradicts it (item 8 below).
**Reviewer risk.** Low. A reviewer may ask why Topics are nodes rather than hyperedges; the answer
(they carry QoS state and can themselves be scored) is implicit but never stated.
**Depends on / feeds.** Table 1's edge vocabulary is consumed by §3.2's rules, §5.2.1's 7-bit one-hot,
and every simulator in §5.1.

#### §3.2 QoS-Aware Weights and Logical Dependency Derivation
**Argues.** Edge coupling weight
$w_E(e) = 0.85(w_{\text{rel}}q_{\text{rel}} + w_{\text{dur}}q_{\text{dur}} + w_{\text{prio}}q_{\text{prio}}) + 0.15\,q_{\text{payload}}$
with AHP weights $(0.30, 0.40, 0.30)$ at $CR < 0.05$ and a floor $w_E \ge 0.01$ so best-effort edges
stay traversable. Durability carries the highest weight because it governs persistence across
partitions. Six projection rules derive a single `DEPENDS_ON` relation directed dependent → dependency.
Closes with the **sequential cascade (Rule 1) vs simultaneous blast (Rule 5)** distinction that
untyped graphs cannot express.
**Evidence.** Table 2 (six rules: `app_to_app`, `app_to_broker`, `node_to_node`, `node_to_broker`,
`app_to_lib`, `broker_to_broker`), each with structural pattern and derived weight.
**Load-bearing caveats.** The $(0.30, 0.40, 0.30)$ triple is a **repo invariant** — `QoSPolicy`'s
constants must match `AHPMatrices.criteria_topic_qos`'s priority vector within `abs=0.01` at
$CR < 0.10$, enforced by `tests/test_ahp_shrinkage.py`. Do not edit the paper's numbers without the
code, or vice versa.
**Reviewer risk.** The sequential/simultaneous distinction is asserted as "a key insight" but its only
direct empirical support is §7.2's edge-removal result (4/50 non-zero edges, all library channels) —
which is $I_{\text{comp}}$-scoped and therefore does not transfer to the $I^*$-backed tables. The
distinction is architecturally convincing and empirically under-tested; expect a reviewer to say so.
**Depends on / feeds.** Produces $G_{\text{analysis}}$ (§3.3), whose density differs sharply from
$G_{\text{structural}}$ (AV: 797 structural vs 3,753 derived edges at the `system` layer, §6.1).

#### §3.3 Dual Graph Views and Architectural Layers
**Argues.** Two representations: $G_{\text{structural}}$ (raw, consumed **exclusively** by simulators)
and $G_{\text{analysis}}$ (derived `DEPENDS_ON` + QoS weights + SCA metrics, consumed by all
predictors). $G_{\text{analysis}}$ is further organised into four layers (Application, Middleware,
Infrastructure, Global System).
**Evidence.** Prose. **No running-example figure** — the retired outline recorded one here
(`figures/src/figure2_running_example.dot` still exists on disk), but the current draft has none.
**Load-bearing caveats.** This separation *is* the independence guarantee (§5.4) at the data-model
level; it is enforced in the codebase by `tests/test_independence_guarantee.py`.
**Reviewer risk.** Moderate and worth pre-empting: the two views are not independent *in the
statistical sense* — $G_{\text{analysis}}$ is a deterministic function of $G_{\text{structural}}$, so
"disjoint views" prevents label leakage but does not make features and labels independent. The paper
claims the former; a reviewer may read it as claiming the latter. The wording in §5.4 ("no simulation
outputs are ever exposed as input features") is the precise version — make sure every restatement is
that precise.
**Depends on / feeds.** §5.1 (oracle substrate), §5.4 (guarantee), §7 (every experiment).

### 4. Interpretable Attribution as a Quality Baseline

#### §4.1 Grounding in ISO/IEC Standards
**Argues.** Two formal constructs — **Component Criticality $D_1$** (impact of a component's own
failure) and **Relationship Criticality $D_2$** (impact of severing a channel while both endpoints
live) — evaluated over two orthogonal characteristics, **Reliability** (decomposing into *Fault
Tolerance* = how broadly failure propagates, and *Availability* = is this a structural SPOF) and
**Maintainability**. Each row names the remediation owner (Reliability Engineer / SRE / Architect),
which is the section's actual product.
**Evidence.** Table 3 (dimension → sub-characteristic → architectural question → graph metrics →
remediation owner).
**Load-bearing caveats.** *Coverage scope* is declared explicitly: Safety (ISO 26262 ASIL) and
Security (STRIDE) are out of scope because they need hazard logs and threat models that structural
analysis does not have. This declaration is what makes "RM" a principled reduction rather than an
arbitrary two-of-nine pick.
**Reviewer risk.** A reviewer who knows the repo history may notice RM is a *re-parameterisation* of a
retired four-dimension AHP composite (per the codebase invariant: $r_\alpha = 0.36$ and
$(0.80, 0.20)$ are derived by dropping Vulnerability from $(A{=}0.43, R{=}0.24, M{=}0.17, V{=}0.16)$
and renormalising). The paper presents the constants as designed rather than derived. That is not
wrong, but the derivation is a *stronger* justification than what is currently written — consider
stating it.
**Depends on / feeds.** §4.3's formulas; §8.3 item 1 revisits Safety/Security as future work.

#### §4.2 Typed Node Feature Representation
**Argues.** Per-type feature vectors: Application 23 dims (0–17 shared topological, 18–22 SCA),
Library 25 (same + reverse-`USES` closure size and reachable-subscriber count — the two structural
drivers of library blast radius), Broker 19 (+ queue capacity), Topic 22 (+ publisher/subscriber
counts, log frequency, ordinal QoS criticality), Node 20 (+ CPU cores, RAM).
**Evidence.** Index-level enumeration.
**Load-bearing caveats.** The two Library extras are the only features engineered specifically for the
simultaneous-blast mechanism of §3.2.
**Reviewer risk.** Low, though listing exact index ranges in prose is unusual for a journal and reads
as implementation documentation; a table would serve better if space allows.
**Depends on / feeds.** Feeds both §4.3 (RM) and §5.2 (HGT input projection) — the *same* feature
space serves the interpretable and the learned predictor, which is what makes the comparison fair.

#### §4.3 Composite Quality Score Formulation
**Argues.** Rank-normalise everything to $[0,1]$, then compose hierarchically:
$FT(v) = 0.45\,\text{RPR} + 0.30\,\text{Deg}_{\text{in}} + 0.25\,\text{CDPot}_{\text{enh}}$;
$A(v) = 0.35\,\text{AP}_c^{\text{dir}} + 0.25\,\text{QSPOF} + 0.25\,\text{BR} + 0.10\,\text{CDI} + 0.05\,w$;
$R(v) = \alpha FT + (1-\alpha)A$ with $\alpha = 0.36$;
$M(v) = 0.35\,\text{BT} + 0.30\,w_{\text{out}} + 0.15\,\text{CQP} + 0.12\,\text{CouplingRisk} + 0.08(1-\text{CC})$;
$Q(v) = 0.80R + 0.20M$; domain reweighting $Q_{\text{domain}} = q_R R + q_M M_{\text{static}}$.
Tiers assigned by box-plot quartiles (CRITICAL above $Q_3 + 1.5\,\text{IQR}$, …). Closes with the
diagnostic payoff: high $A$ / low $FT$ ⇒ pure SPOF ⇒ replicate; high $FT$ ⇒ cascade hub ⇒ circuit
breakers and bulkheads.
**Evidence.** AHP consistency audit: $CR = 0.001$ (FT), $0.001$ (A), $0.000$ (M), all $\le 0.10$.
Shipped intra-dimension weights are a $\lambda = 0.70$ shrinkage blend toward a uniform prior, with
§7.3 reporting the full sweep.
**Load-bearing caveats.** $\alpha = 0.36$ and $(0.80, 0.20)$ are **declared constants under a repo
invariant** (`tests/test_quality_model.py::TestCompositeReparameterisation`) — not free parameters.
A second invariant forbids reporting a quality-in-use scalarisation as a quantity distinct from a
reweighted $Q(v)$ (`TestQiuCollapseEquivalence`); §4.3's $Q_{\text{domain}}$ presentation is
consistent with this, and §7.3's domain-sweep conclusion ("attributional, not ranking-improvement")
is the paper-side statement of the same fact.
**Reviewer risk.** Highest-risk subsection in the front half. Roughly a dozen hand-set constants
appear across four formulas. The AHP audit answers *internal consistency* ($CR$) but not *external
validity* — a consistent matrix can still encode a wrong judgement, and Table 8 shows the whole family
is anti-correlated with ground truth. Expect: "why should I believe these weights?" The honest answer
is already in the paper (§7.3: AHP makes RM *less wrong*, not accurate; its role is attribution) —
make sure §4 forward-references that rather than letting §4 read as if $Q(v)$ were a validated ranker.
**Depends on / feeds.** §7.1 (RM's LOSO $\rho = -0.014$), §7.3 (shrinkage, domain reweighting,
stratified $\rho$), §8.1 (when to use attribution vs learning).

### 5. Graph Learning for Failure-Impact Prediction

#### §5.1 Ground-Truth Simulation Oracles
**Argues.** Announces a "formal three-oracle taxonomy" and then defines **four**: $I^*(v)$ cascade
reachability (`FaultInjector`) — the primary continuous training target; $I_{\text{comp}}(v)$
multi-metric composite (`FailureSimulator`) $= 0.35\Delta\text{Reach} + 0.25\Delta\text{Frag} +
0.25\Delta\text{Through} + 0.15\Delta\text{Flow}$ — canonical for quality-gate verification;
$I_{\text{dyn}}(v)$ SimPy queue-flow (`MessageFlowSimulator`); and $I_{\text{edge}}(u,v)$ removal
oracle. Then states plainly that **the oracles are not interchangeable and their disagreement bounds
what any one can support**.
**Evidence.** Seven scenarios × five seeds: $\rho = 0.765$ / $0.465$ / $0.397$ for
$(I_{\text{dyn}},I^*)$ / $(I_{\text{comp}},I_{\text{dyn}})$ / $(I_{\text{comp}},I^*)$, per-scenario
minima $0.550$ / $0.121$ / $0.097$; top-$K$ Jaccard 0.26–0.31 for every pair.
**Load-bearing caveats.** The sentence *"the instruction to 'simply simulate' is under-specified — it
does not say which simulator, at which propagation threshold, under which seed"* is the paper's second
major defensive move, paired with §1.3.1. `FaultInjector` (Predict-stage labeler) and
`FailureSimulator` (Validate-stage oracle) are **non-interchangeable by repo invariant**
(`tests/test_groundtruth_contract.py`).
**Reviewer risk.** This section is unusually honest and that honesty is an asset — but it hands a
reviewer the strongest available objection in the paper's own words. The mitigation is that the
objection is *scoped*, not dismissed: §8.2 repeats it as a construct-validity bound. Keep those two
statements consistent with each other and with §7.3's convergent-validity paragraph.
**Depends on / feeds.** $I^*$ labels every table in §7.1 and §7.4; $I_{\text{comp}}$ backs §7.2's
edge removal and §7.3's anti-pattern catalog; $I_{\text{dyn}}$ appears only as a validity check.

#### §5.2 Heterogeneous Graph Transformer Architecture
**Argues.** 3-layer HGT [12], hidden $D = 64$, 4 attention heads: type-specific input projection →
relational mutual attention with edge-feature ingestion ($\tilde{h}_v = h_v + W_{\text{edge}}e_{uv}$)
→ multi-task residual heads.
**Evidence.** Figure 2 (ASCII layer diagram with the attention and message equations inline).
**Load-bearing caveats.** Edge features are added to the **destination** representation before
attention — this is the mechanism by which QoS enters, and therefore the mechanism whose ablation
produces RQ3's sign-flip result.
**Reviewer risk.** The opening sentence lists interaction semantics as "`CALLS`, `PUBLISHES_TO`,
`SUBSCRIBES_TO`, `HOSTED_ON`, etc." — neither `CALLS` nor `HOSTED_ON` exists in Table 1 (which has
`RUNS_ON`). A reviewer checking the model against its own formalism will find this immediately (item 8
below). Separately: hyperparameters are stated in prose with no tuning protocol, no search space, and
no justification for $D = 64$ / 3 layers / 4 heads. The retired outline recorded a dedicated
hyperparameter table (old Table 7) that the condensation dropped; JSS reviewers will likely want it
back, at minimum as an appendix.
**Depends on / feeds.** §5.2.1 (edge encoding), §5.3 (heads), §7.1–§7.4 (everything measured).

#### §5.2.1 Edge Feature Encoding (16-Dimensional)
**Argues.** $e_{uv} \in \mathbb{R}^{16}$: index 0 scalar QoS weight; index 1 normalised path count;
indices 2–8 seven-bit one-hot over edge types; indices 9–15 seven explicit QoS dimensions (reliability,
durability, priority, deadline flag, log deadline, log max-blocking, QoS-heterogeneity flag), non-zero
only on `PUBLISHES_TO` / `SUBSCRIBES_TO`.
**Evidence.** Index-level specification with the exact policy-score encodings.
**Load-bearing caveats.** Indices 9–15 are precisely what the "-QoS" variants add; the base variants
zero them. This is the operational definition of the RQ3 ablation.
**Reviewer risk.** **Index 0 contradicts §3.2.** Here $w(e)$ is "the harmonic mean of reliability,
durability, priority, deadline, and blocking multipliers"; in §3.2 it is a weighted arithmetic sum
over three terms plus payload. Two incompatible definitions of the same symbol (item 9 below).
**Depends on / feeds.** §7.3's QoS ablation is meaningless without this section being right.

#### §5.3 Multi-Task Prediction Heads and Dimension Masking
**Argues.** Four heads (Reliability, Maintainability, composite impact $\hat{I}^*$, edge criticality
$\hat{Q}(u,v)$). Joint loss
$\mathcal{L} = \mathcal{L}_{\text{comp}} + 0.5\mathcal{L}_{\text{dim}} + 0.3\mathcal{L}_{\text{rank}}
+ 0.1\mathcal{L}_{\text{pair}} + 0.1\mathcal{L}_{\text{cons}} + 0.3\mathcal{L}_{\text{edge}}$ with
ListMLE [49] ranking loss. **Dimension masking** $m = [1, 0]$: because `FaultInjector` observes
runtime reachability and not static maintainability, the M head is unsupervised and masked out rather
than driven to zero. Consequently $Q_{\text{domain}}$ is **never populated in any reported result** —
the implementation refuses to emit it without real M supervision, so domain reweighting is evaluated
against the static RM baseline instead (§7.3).
**Evidence.** Formulas; the explicit non-emission statement.
**Load-bearing caveats.** The refusal-to-emit is a genuine engineering-integrity detail and worth
keeping — it prevents a reader from assuming the learned $Q_{\text{domain}}$ was measured.
**Reviewer risk.** Six loss terms with five hand-set coefficients and no ablation over any of them.
Combined with §5.2's unjustified hyperparameters, this is the paper's largest *unexamined* design
surface — and it sits directly opposite §4.3, where every constant *is* audited. A reviewer may
reasonably ask why the AHP weights get a consistency audit and a shrinkage sweep while the loss
weights get neither.
**Depends on / feeds.** §7.3's domain-reweighting ablation exists in its current form *because* of the
masking; §8's claims about the R head rest on it.

#### §5.4 Input–Label Independence Guarantee
**Argues.** Feature space from $G_{\text{analysis}}$ only; label space from $G_{\text{structural}}$
only, via the three named simulators. No simulation output is ever a model input.
**Evidence.** Four sentences — the shortest subsection in the paper.
**Load-bearing caveats.** Backed in the codebase by three separate invariants
(`test_independence_guarantee.py`, `test_groundtruth_contract.py`,
`test_predict_simulate_separation.py`). None of that enforcement is mentioned in the paper.
**Reviewer risk.** The claim is strong and the support is four sentences. Since the enforcement is
real and mechanical, saying so — "enforced by architectural tests in the replication package" — would
convert an assertion into a verifiable one at the cost of a single clause.
**Depends on / feeds.** §8.2's internal-validity paragraph is its only restatement.

### 6. Experimental Setup

#### §6.1 Datasets and System Corpus
**Argues.** 1,770 components across ten architectures: seven synthetic (AV 152, Enterprise 520,
Financial 124, Healthcare 98, Hub-and-Spoke 139, IoT 326, Microservices 186) and three real-world
(Autoware.universe 75 [45], Cloud Microservices 60 [47], Train-Ticket 90 [46]). Synthetic scenarios
come from statistical topology generators spanning degree distributions, clustering coefficients, and
QoS configurations; real-world ones are hand-transcribed via architectural adapters.
**Evidence.** Table 4, with per-type breakdowns; $|V|$ confirmed against the scenario manifest, $|E|$
counting raw structural relationships (explicitly *not* the denser derived projection — AV: 797 vs
3,753).
**Load-bearing caveats.** The manifest reconciliation is a **repo invariant**: every
`data/scenarios/*_system.json` must regenerate byte-identically from its config, and stale
`output/loso_cache/` has published wrong numbers before (`tests/test_scenario_corpus.py`).
**Reviewer risk.** **Table 4 does not list the ATM case study**, yet ATM is the 8th LOSO fold (§6.3,
Table 7), the subject of the §7.3 attention analysis, and one of the "8 system scenarios" in the
anti-pattern evaluation. The corpus table and the experiment counts do not reconcile (item 4 below).
Separately, "hand-transcribed" invites a fidelity question the paper does not answer: how was
transcription validated against the real repositories?
**Depends on / feeds.** Every experiment in §7.

#### §6.2 Baselines and Evaluated Predictors
**Argues.** Five configurations across seven columns: Topo-BL (unweighted betweenness + articulation),
Topo-QoS (QoS-weighted centrality), RM/$Q(v)$ (§4), GL / GL-QoS (homogeneous GAT [41] on the flattened
projection), HGL / HGL-QoS (relation-specific HGT, §5).
**Evidence.** Enumeration only. The retired outline recorded a table here (old Table 5) mapping each
predictor to *the factor it isolates*; the condensation dropped it.
**Load-bearing caveats.** The design is a clean 2×2 — {homogeneous, typed} × {QoS, no QoS} — plus two
training-free baselines. That factorial structure is what licenses RQ2's and RQ3's causal readings
(typing contributes $+0.197$; QoS contributes $\pm$ depending on protocol). Saying so explicitly would
strengthen both.
**Reviewer risk.** Topo-QoS is the strongest competitor and the one HGL fails to beat significantly
in-distribution; it deserves more than one line of definition. A reviewer will want to know exactly
how QoS weights enter the centrality computation before accepting the comparison.
**Depends on / feeds.** Column headers of Tables 5–7.

#### §6.3 Evaluation Metrics and Protocols
**Argues.** Ranking by Spearman $\rho$ / Kendall $\tau$ against $I^*(v)$; critical-set by $F_1@K$,
P@K, R@K with $K = \lceil 0.20|V_{\text{app}}|\rceil$; significance by paired Wilcoxon [48] at
$p < 0.05$ with bootstrap 95 % CIs ($B = 2{,}000$) [49, 50]. Three protocols: **in-distribution**
(60/20/20 node splits pinned by identity, five seeds $\{42, 123, 456, 789, 2024\}$); **inductive
LOSO** (8 cached scenarios, train on 7, zero-shot the held-out one); **real-world transfer**.
**Evidence.** Protocol specification.
**Load-bearing caveats.** Splits are *node-level within scenario* for the in-distribution protocol —
which is why LOSO is the more meaningful generalisation test and why the paper leans on it for its
defensible RQ1 claim.
**Reviewer risk.** Two things. (a) The ATM cross-reference is wrong — "the ATM case study, §7.4" points
at real-world validation; ATM is in §7.3 (item 3 below). (b) $n = 7$ paired Wilcoxon has a minimum
achievable two-sided $p$ of $0.0156$, so the two "significant" rows in Table 6 are at the floor of
what this design can detect and nothing subtler than a 7/7 or 6/7 sweep can ever reach significance.
That is a real limitation of the protocol and the paper does not name it.
**Depends on / feeds.** All of §7.

### 7. Results and Empirical Analysis

#### §7.1 RQ1 — Graph Learning vs Structural Baselines
**Argues.** Three findings, in the order the paper puts them: (1) **critical-set advantage** —
$F_1@K = 0.467$ (HGL) vs $0.339$ (Topo-QoS), $+37.8\%$ relative; (2) **out-of-distribution
superiority** — HGL-QoS $\rho = 0.668$ vs Topo-QoS $0.522$, with homogeneous GNNs at $0.381$–$0.501$
and RM essentially non-predictive at $-0.014$; (3) **in-distribution scope condition** — HGL ($0.725$)
and GL ($0.612$) beat the structural baselines on point estimates, but the margin over Topo-QoS
($0.586$) is "constrained by topology variance", and HGL-QoS *trails* HGL in-distribution.
**Evidence.** Table 5 (per-scenario in-distribution $\rho$ with bootstrap CIs), Table 6 (paired
Wilcoxon), Table 7 (LOSO, 8 folds). Table 7 is backed by a retained artifact
(`results/loso_all_variants.json`, `results/table4_loso_results.md`, `output/loso_cache/` with 8
scenario directories) — **this resolves the retired outline's headline reproducibility gap.**
**Load-bearing caveats.** Finding 3's phrase "constrained by topology variance" is doing the work of
"$p = 0.469$, not significant". The honest version is in Table 6 and the paper does print it; the prose
softens it. Note also Table 5 shows Topo-QoS *winning outright* on Healthcare ($0.772$) and
Microservices ($0.707$), and HGL collapsing to $0.483$ on Microservices — the very paradigm the
real-world validation then celebrates.
**Reviewer risk.** Highest in the paper. Three specific pushes to expect: (a) "your headline claim
fails your own significance test in-distribution"; (b) "$F_1@K$ 0.467 is a *low absolute* number —
you are shortlisting correctly less than half the time"; (c) "IoT Smart City goes $0.073$ (Topo-QoS)
→ $0.881$ (HGL) while Hub-and-Spoke goes $0.359 \to 0.534$ — what property of a topology determines
when typing pays?" (c) is the most valuable and the paper does not answer it; a short analysis
relating the gap to type-mix or degree heterogeneity would substantially strengthen RQ1.
**Depends on / feeds.** Tables 5–7 are the paper's spine; §8.1's practitioner guidance is read
directly off them.

#### §7.2 RQ2 — Value of Typed Heterogeneity
**Argues.** Typing helps most exactly where it should: in-distribution HGL − GL $= +0.113$ (n.s.,
$p = 0.469$, 4/7 won) because homogeneous models can approximate type distinctions from degree
signatures on familiar topologies; **out-of-distribution HGL − GL $= +0.197$** ($0.578$ vs $0.381$)
with substantially lower fold variance ($\sigma$ 0.116 vs 0.164), widening to $+0.167$ between the
QoS-aware variants. Edge-removal probe on `av_system`: of 50 candidate bridge edges, 46 have zero
downstream impact and the 4 non-zero ones are all library communication channels — individual links
are largely redundant; component and shared-library failures dominate.
**Evidence.** Deltas against Tables 5 and 7; the 4/50 edge-removal measurement.
**Load-bearing caveats.** Edge criticality here is **measured by simulated removal**, not inferred
from node labels — a genuine methodological strength worth stating more prominently than it currently
is. But the measurement runs against $I_{\text{comp}}$, so it **does not transfer** to the $I^*$-backed
tables (standing caveat 2).
**Reviewer risk.** This is the paper's strongest result and the one to lead with. Two pushes: the
edge-removal probe is a single topology and 50 candidates (why `av_system`? why 50? how selected?),
and the variance-reduction claim ($\sigma$ 0.116 vs 0.164) over 8 folds is suggestive rather than
tested.
**Depends on / feeds.** Corroborates §3.2's simultaneous-blast semantics and §7.3's attention finding
(both point at `USES` edges); feeds §8.1's "when to use graph learning".

#### §7.3 RQ3 — Ablations and Sensitivity Analysis
**Argues.** Seven distinct sub-analyses. **QoS encoding**: the sign flips by protocol — $-0.072$
in-distribution ($p = 0.078$, wins 1/7) but $+0.090$ under LOSO, making HGL-QoS the single best
configuration in the study; read as *situational insurance against distribution shift*, not
redundancy. **AHP shrinkage**: $\rho$ improves monotonically $-0.0514 \to -0.0069$ as $\lambda: 0 \to 1$
($+0.045$), but is **negative at every setting** — AHP makes RM less wrong, not accurate.
**Convergent validity**: $\rho(I_{\text{dyn}}, I^*) = 0.765$ supports the narrower claim that $I^*$'s
*ordering* tracks dynamic disruption, while Jaccard $0.312$ and $\rho(I_{\text{comp}}, I^*) = 0.397$
deny critical-set agreement. **Domain reweighting**: $+0.026$ over the static default but $-0.110$
against an equal split; the cause is structural (all six declared domains land within 0.04 of the
static $w_R = 0.80$; $\tau = 0.968$ between the two rankings), so Context-of-Use reweighting is
reframed as *attributional, not ranking-improving*. **Threshold/normalisation**: $\Delta\rho = 0.190$
across propagation thresholds vs $0.016$ across normalisation schemes — stability under a free
parameter, explicitly *not* evidence of correctness. **Anti-patterns/CI-CD**: $F_1 = 0.378$ over 8
scenarios decomposes into precision $0.239$ / recall $0.900$ — a deliberate high-recall gate flagging
94.2 % of components; `DEEP_PIPELINE` excluded for non-termination (>10 min on 50 apps), itself a
reportable scalability limit; 18 detectors run 0.04 s–54.85 s. **Stratification**: RM's per-type $\rho$
is $0.503$/$0.395$/$0.142$ (App/Broker/Node) while *pooled* $\rho = 0.028$ falls outside that range
entirely — a Simpson's-paradox effect, so only stratified figures may be quoted. **Attention**: on the
ATM case study the top weight ($\alpha = 1.00$) is a `USES` edge, next tier ($\approx 0.50$) `ROUTES`
and further `USES`/`SUBSCRIBES_TO` — library-blast and broker-centrality pathways, not sequential
pub-sub flow.
**Evidence.** Table 8 (shrinkage), the sweeps above, Figure 4 (cited, uncaptioned), Figure 3 (cited,
uncaptioned).
**Load-bearing caveats.** Three sentences here are the paper's intellectual honesty on display and
must survive any edit: *"AHP calibration makes the RM baseline less wrong, not accurate"*; *"A small
spread … is not, by itself, evidence that the ordering is correct"*; *"only the stratified, per-type
numbers should be quoted."*
**Reviewer risk.** Two structural problems, both fixable without new experiments. (a) **Scope
mismatch** — RQ3 as stated in §1.4 covers only QoS encoding and weighting calibration, so five of the
seven sub-analyses answer a question the paper never asked (item 10). (b) **Mis-filing** — the
Simpson's-paradox stratification, which is about the RM baseline, sits under the heading "Anti-Pattern
Detection and CI/CD Quality Gates", which is about something else entirely (item 11). Also: the
anti-pattern and CI/CD material is the **only surviving trace of the retired §6 remediation/gating
section**, and it now sits in an ablation subsection with no RQ of its own — a reader will wonder why
runtime benchmarks appear in a sensitivity analysis.
**Depends on / feeds.** The QoS result is what makes HGL-QoS the recommended deployment configuration
in §8.1; the convergent-validity paragraph is the evidence base for §8.2's construct-validity bound.

#### §7.4 RQ4 — Real-World Distributed Software Architecture Validation
**Argues.** Zero-shot transfer to three open-source architectures succeeds: Autoware.universe
$\rho = 0.688 \pm 0.009$ / $\tau = 0.517$ / $F_1@K = 0.800$; Cloud Microservices $0.778 \pm 0.001$ /
$0.639$ / $1.000$; Train-Ticket $0.759 \pm 0.001$ / $0.605$ / $1.000$. Every application with non-zero
cascading impact in the two microservice systems is contained in the predicted top-$K$ (tie-robust
$F_1@K$ $0.760$ and $0.810$). Predictive gain over raw degree centrality: $+0.360$ / $+0.014$ /
$+0.264$.
**Evidence.** Table 9 (8 columns per system).
**Load-bearing caveats.** The non-zero-impact denominators are small — 19/32, 8/22, 14/41 — so
$F_1@K = 1.000$ on Cloud Microservices means "8 of 8 in the top-5". The tie-robust column is the
honest one and the paper reports it; keep both.
**Reviewer risk.** Three pushes. (a) **Cloud Microservices' gain over degree centrality is $+0.014$**
— on that system, degree centrality is essentially as good as the whole framework, and the prose
("substantial predictive gain") quotes only Autoware and Train-Ticket. This is the single most
attackable number in §7. (b) "Real-world" here means *real architecture, simulated failures* — no
production outage data validates any of it; §8.2 says so, but §7.4's framing ("authentic") invites the
stronger reading. (c) Two of three systems are microservice benchmarks and one is ROS 2; the claim of
spanning "cyber-physical and cloud-native paradigms" rests on a single cyber-physical instance.
**Depends on / feeds.** The external-validity leg of §8.2; the transferability highlight.

### 8. Discussion, Threats to Validity, and Conclusion

#### §8.1 Discussion and Practical Implications
**Argues.** Two decision rules for practitioners. **When to use graph learning**: top-$K$ shortlisting
for hardening ($F_1@K = 0.467$) and unseen out-of-distribution architectures ($\rho = 0.668$), with
the base-vs-QoS choice itself a design decision — HGL in-distribution, HGL-QoS under expected shift.
**When to use interpretable attribution**: root-cause diagnosis and remediation planning, where
separating SPOF exposure (Availability) from wide error propagation (Fault Tolerance) dictates
replicas vs circuit breakers.
**Evidence.** Read directly off §7.1 and §7.3.
**Load-bearing caveats.** The word "indispensable" for RM sits awkwardly beside RM's LOSO $\rho = -0.014$
and its anti-correlation at every $\lambda$. The claim is defensible *as attribution* — §1.3.1 and §4
build the case — but the sentence as written does not carry its own justification.
**Reviewer risk.** Short (two bullets) for a JSS discussion section. It reads as a summary rather than
a discussion: there is no engagement with *why* typing helps out-of-distribution but not
in-distribution, no comparison back to the §2 literature, and no account of what the results imply for
the Architecture–Code Gap framing that opened the paper. This is the clearest place where condensation
cut too deep — expect a reviewer to ask for interpretation, not recapitulation.
**Depends on / feeds.** §7 entire.

#### §8.2 Threats to Validity
**Argues.** Four categories, each with a named mitigation or an admitted bound. **Construct**: the
oracle disagreement is reported *as a bound rather than a mitigation* ($0.765$ / $0.397$, Jaccard
0.26–0.31, 30–47 % of components unlabelled), with the explicit rule that a result against one oracle
is not a claim about another and none is a claim about production outages. **Internal**: leakage
prevented by the $G_{\text{analysis}}$ / $G_{\text{structural}}$ split; discloses a **fixed
normalisation defect** — a min-max helper assigned maximal Code Quality Penalty to any zero-variance
population, indistinguishable from "no code-quality data at all", which is exactly every Library in
the three real-world scenarios. **External**: ten domains and three authentic architectures; larger
enterprise deployments remain future work. **Conclusion**: non-parametric throughout (Spearman,
Kendall, bootstrap $B = 2{,}000$, paired Wilcoxon), plus the Simpson's-paradox hazard stated directly.
**Evidence.** Cross-references §5.1 and §7.3; quotes the post-fix deltas as
$\Delta\rho \in \{-0.003, 0.000, 0.000\}$.
**Load-bearing caveats.** Disclosing a *found and fixed* instrument defect, with its post-fix impact
quantified, is a strong integrity signal for JSS and should not be edited away for length.
**Reviewer risk.** The defect-disclosure paragraph attributes the $\Delta\rho$ figures to "Table 9",
but Table 9 has no $\Delta\rho$ column — the cited support does not exist in the cited place (item 13).
Also "ten distinct system domains" here vs "six declared domains" in §7.3's sweep (item 12). Both are
small, and both are in the section reviewers read most sceptically.
**Depends on / feeds.** Restates §5.1 and §5.4; bounds every claim in §7.

#### §8.3 Limitations and Future Work
**Argues.** Three directions: Safety/Security integration (ISO 26262 ASIL, STRIDE) into the multigraph
schema; hardware-in-the-loop validation against physical fault-injection testbeds; automated
architectural refactoring — extending SaG from predictive to prescriptive, generating pull requests
that reconfigure QoS and add redundancy.
**Evidence.** N/A.
**Load-bearing caveats.** Item 3 is where the retired §6 remediation work went. Since that machinery
*exists* (preserved in [`../thesis/material/remediation_and_gating.md`](../thesis/material/remediation_and_gating.md)),
describing it purely as future work understates what was built — but expanding it re-opens the length
problem that motivated the cut. Leave as future work; the honest framing is that it is out of scope
for *this* submission.
**Reviewer risk.** Notable *omissions* from the limitations list, each of which a reviewer may raise
as an unacknowledged limitation rather than a known one: the absence of production-failure validation,
the small $n = 7$ significance design, the unexamined loss/hyperparameter surface (§5.2, §5.3), and
the fact that no result establishes that RM's *dimensional* decomposition beats a scalar at anything
measurable. Naming two or three of these here would cost a paragraph and remove the sting.
**Depends on / feeds.** Closes the arc opened by §4.1's coverage-scope declaration.

#### §8.4 Conclusion
**Argues.** Restates SaG, the HGT + RM combination, the Architecture–Code Gap framing, and closes on
$F_1@K = 0.467$ and $\rho = 0.668$ as the two headline numbers.
**Evidence.** N/A — recapitulation.
**Load-bearing caveats.** Choosing $F_1@K$ and LOSO $\rho$ as the closing numbers is **correct** —
these are exactly the two defensible halves of RQ1, and the conclusion avoids the in-distribution
figure that fails its significance test. This is deliberate; do not "improve" it by adding $0.725$.
**Reviewer risk.** Low. One nit: it quotes Topo-QoS at $0.521$ where Table 7 and the retained artifact
both say $0.522$ (item 1).
**Depends on / feeds.** Mirrors the abstract; the two must stay numerically identical.

### References and Declarations
**Argues.** 59 numbered entries plus `[Anon-A]` (withheld for anonymised review). Declarations block:
CRediT (stub pending de-anonymisation), competing interest (none declared), funding (stub), data
availability (replication package on publication), generative-AI use (stub).
**Evidence.** N/A — bibliographic and administrative.
**Load-bearing caveats.** The data-availability statement is now **clean**: the retired Table 10
carve-out ("artifact not retained") is gone because the LOSO artifact *is* retained
(`results/loso_all_variants.json`). That removal is correct, not a lapse.
[`latex/refs.bib`](latex/refs.bib) holds 63 entries against 59 cited — some are unused.
**Reviewer risk.** `[Anon-A]` is listed but cited nowhere in the body (see §1.6). An uncited reference
is a copyediting flag; a *missing prior-work disclosure* is an editorial-compliance issue for this
special issue specifically.
**Depends on / feeds.** Must stay in sync with `latex/sections/declarations.tex`, which currently
matches `draft.md` verbatim.

---

## Table and figure inventory

9 tables, 4 nominal figures — each owned by exactly one subsection. Useful as a checksum against
renumbering.

| # | Owning section | Content | LaTeX label |
|:--:|---|---|---|
| Table 1 | §3.1 | Entity types + structural edge types (two sub-tables) | `tab:1` |
| Table 2 | §3.2 | Six `DEPENDS_ON` projection rules | `tab:2` |
| Table 3 | §4.1 | RM quality decomposition (FT/A/M, question, metrics, owner) | `tab:3` |
| Table 4 | §6.1 | Experimental evaluation corpus (10 architectures, 1,770 components) | `tab:4` |
| Table 5 | §7.1 | In-distribution held-out Spearman $\rho$ vs $I^*(v)$, 7 columns | `tab:5` |
| Table 6 | §7.1 | Paired Wilcoxon signed-rank tests ($n = 7$) | `tab:6` |
| Table 7 | §7.1 | Inductive LOSO evaluation (8 folds; **artifact retained**) | `tab:7` |
| Table 8 | §7.3 | AHP shrinkage sensitivity ($\lambda$ sweep) | `tab:8` |
| Table 9 | §7.4 | Real-world architecture validation (3 systems) | `tab:9` |
| Figure 1 | §1.3 | End-to-end SaG pipeline | `fig:1` — **rendered** |
| Figure 2 | §5.2 | HGT layered architecture | ASCII in `draft.md`; **not in LaTeX** |
| Figure 3 | §7.3 | Relation-specific attention case study (ATM) | cited only; **no caption, not in LaTeX** |
| Figure 4 | §7.3 | Mean $\rho$ vs shrinkage $\lambda$ | cited only; **no caption, not in LaTeX** |

**The figure situation needs a decision, not just a fix.** Only `Figure_1` is `\includegraphics`'d by
the live LaTeX. [`latex/figures/`](latex/figures/) holds `Figure_2/3/4.{pdf,png}`, but
`figures/src/figure2_running_example.dot` draws the **running example** — the §3.3 figure the current
draft no longer has — not the HGT layer stack that `draft.md` now calls Figure 2. So the on-disk
assets are numbered for the retired structure. Either renumber the assets to the current draft or
renumber the draft's figures to the assets; doing neither ships a paper with one figure.

## RQ traceability

| RQ | Question (abbreviated) | Primary section | Primary evidence | Standing caveat |
|:--:|---|---|---|---|
| RQ1 | How accurately does typed learning predict impact and identify critical sets vs non-learning baselines? | §7.1 | Tables 5–7 | Ranking margin **n.s.** in-distribution ($p = 0.469$); the defensible halves are $F_1@K$ and LOSO |
| RQ2 | What does typed heterogeneity reveal that homogeneous representations obscure? | §7.2 | LOSO deltas $+0.197$/$+0.167$; 4/50 edge removal | Edge-removal result is $I_{\text{comp}}$-scoped, does not transfer to $I^*$ tables |
| RQ3 | How do QoS encoding and weighting calibration affect performance and stability? | §7.3 | Table 8, QoS sign-flip, five further sweeps | RQ3's stated scope covers only 2 of 7 sub-analyses (item 10) |
| RQ4 | Does the framework transfer zero-shot to real open-source architectures? | §7.4 | Table 9 | Gain over degree is $+0.014$ on one of three; simulated failures, not production outages |

---

## Appendix: condensation history

*(Background. `draft.md` is authoritative; this records where the cut material went.)*

`draft.md` was cut from ~30,100 words (23 tables, 6 figures) to **~11,100 words (9 tables,
4 nominal figures)** to fit JSS's ≤36-single-column-page guidance, refocused on the graph-learning-and-
dependability claim, with hierarchical RM attribution repositioned as a baseline the results are read
against rather than a co-equal contribution. A second pass removed the prescriptive-remediation and
CI/CD-gating section entirely (retired §6), dropping RQ5 and reducing six contributions to four.

Nothing was deleted outright. The pre-condensation text is preserved in
[`../thesis/`](../thesis/) — see that folder's `README.md` for the section-by-section map.
`../thesis/jss_draft_full.md` is a frozen snapshot from immediately before the first condensation and
is not edited in place; it (and the extracted material files) still use the retired four-dimension
"RMAV" terminology that `draft.md` has since replaced with RM. The seven extracted material files, all
confirmed present:

- [`../thesis/material/model_details.md`](../thesis/material/model_details.md) — full `cm_*` metric list and QoS weight formulas cut from §3.1/§3.2.
- [`../thesis/material/rm_attribution.md`](../thesis/material/rm_attribution.md) — full RMAV (pre-migration) derivation, three weighting paths, Quality-in-Use transformation matrix, worked example.
- [`../thesis/material/relationship_criticality.md`](../thesis/material/relationship_criticality.md) — edge-level criticality, cut from §4 since it is not validated by the edge-removal measurement.
- [`../thesis/material/oracles_and_labels.md`](../thesis/material/oracles_and_labels.md) — full original oracle/label treatment, including the dropped pooled-vs-per-type correlation figure.
- [`../thesis/material/remediation_and_gating.md`](../thesis/material/remediation_and_gating.md) — the full seven-subsection retired §6, including the remediation yield table.
- [`../thesis/material/threats_and_instrument_defects.md`](../thesis/material/threats_and_instrument_defects.md) — the full six-defect account condensed to one paragraph in §8.2.
- [`../thesis/material/why_not_simulate.md`](../thesis/material/why_not_simulate.md) — the extended predictor-vs-oracle argument condensed into §1.3.1.

**Status changes since this file was last regenerated:**
1. **LaTeX is now converted.** [`latex/sections/`](latex/sections/) matches `draft.md`; `manuscript.tex`
   `\input`s exactly 8 section files and the build produces `manuscript.pdf`.
2. **The LOSO artifact is retained** (`results/loso_all_variants.json`,
   `results/table4_loso_results.md`, `output/loso_cache/` with 8 scenario directories, matching
   Table 7 to 4 decimal places). The retired outline's first outstanding item is closed.
3. **Retired §6 (remediation + CI/CD gating) and RQ5 are gone**; the only surviving trace is the
   anti-pattern/runtime paragraph now sitting inside §7.3.

## Outstanding work

1. **Restore the prior-work disclosure.** `[Anon-A]` is in the reference list but cited nowhere; the
   §1.5 paragraph that positioned this paper against it, and stated no companion manuscript is under
   parallel review, was lost in condensation. The special issue explicitly requires disclosure of
   previously published related material — this is an editorial-compliance item, not a copyediting one.
2. **Resolve the figure numbering** (see the inventory note): three of four figures are cited but not
   rendered, and the on-disk assets are numbered for the retired structure.
3. **Delete or archive the four orphan `.tex` files** in [`latex/sections/`](latex/sections/) that
   `manuscript.tex` no longer `\input`s — `sec6_remediation_gating.tex`, `sec7_experimental_setup.tex`,
   `sec8_results.tex`, `sec9_discussion.tex`. They are the retired 9-section structure and will
   mislead the next editor into treating them as live sources. Content is preserved in
   `../thesis/material/` and in git history. *(Flagged, not actioned.)*
4. **Rebuild single-spaced and confirm the real page count** against JSS's ≤36-single-column-page
   guidance. At ~11,100 words the paper may now be *under*-length for JSS, which reopens room for the
   §2 positioning, §8.1 discussion, and §5 hyperparameter table that condensation removed.
5. **De-anonymise** [`latex/title_page.tex`](latex/title_page.tex) and
   [`latex/sections/declarations.tex`](latex/sections/declarations.tex) before submission; prune the
   4 uncited entries from [`latex/refs.bib`](latex/refs.bib) (63 entries vs 59 cited).
6. **Resolve the draft inconsistencies below** — editorial fixes to `draft.md` *and* the matching
   `latex/sections/*.tex`, since the two are now in sync and must stay that way.

## Draft inconsistencies recorded, not fixed

Each verified against the current `draft.md` while regenerating this file. None is fixed here; they
belong to the section-by-section revision passes. Fixes must be applied to **both** `draft.md` and the
corresponding `latex/sections/*.tex`.

1. **`0.521` vs `0.522`.** Topo-QoS's LOSO $\rho$ is `0.522` in Table 7 and `0.5220` in the retained
   artifact, but `0.521` in §7.1's Key Finding 2, in §8.4's conclusion, and in the abstract — in both
   `draft.md` and [`latex/sections/abstract.tex`](latex/sections/abstract.tex). Four occurrences;
   `0.522` is correct.
2. **The abstract's finding (1) mixes two model variants silently.** `$F_1@K = 0.467$` is HGL;
   `$\rho = 0.668$` is HGL-QoS. Both are correct in isolation and the sentence reads as one system.
   Naming the variants costs four words.
3. **Wrong cross-reference to the ATM case study.** §6.3 and the note under Table 7 both cite it as
   "§7.4"; §7.4 is the real-world validation. ATM appears in §7.3's attention analysis.
4. **ATM is an undocumented 8th scenario.** LOSO reports 8 folds ("the seven synthetic scenarios plus
   the ATM case study") and the anti-pattern evaluation covers "8 system scenarios", but Table 4's
   corpus lists 10 architectures (7 synthetic + 3 real-world) with no ATM row, and the abstract says
   "seven synthetic topologies". `output/loso_cache/` confirms `atm_system` is a real 8th scenario.
   Either add it to Table 4 or explain its exclusion.
5. **Figure inventory is broken** — three of four figures cited without captions or rendered assets;
   the on-disk `Figure_2` draws the running example, not the HGT stack. See the inventory note.
6. **§5.1 announces a "formal three-oracle taxonomy" and then defines four** (Cascade Reachability,
   Multi-Metric Composite, Dynamic Queue-Flow, Relationship Removal). §7.3's convergent-validity
   paragraph consistently says three — the edge oracle is arguably a different kind of instrument, but
   the section must say so rather than miscount.
7. **`highlights.tex` bullet 3 claims "significant gains"** — unsupported for HGL vs Topo-QoS
   ($p = 0.469$). Reword to the critical-set and cross-architecture framings, which are supported.
8. **§5.2 names edge types that do not exist.** It lists "`CALLS`, `PUBLISHES_TO`, `SUBSCRIBES_TO`,
   `HOSTED_ON`, etc."; Table 1 defines `PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`,
   `CONNECTS_TO`, `USES`. `CALLS` and `HOSTED_ON` appear nowhere else in the paper.
9. **Two incompatible definitions of $w(e)$.** §3.2: weighted arithmetic sum,
   $0.85(w_{\text{rel}}q_{\text{rel}} + w_{\text{dur}}q_{\text{dur}} + w_{\text{prio}}q_{\text{prio}}) + 0.15q_{\text{payload}}$.
   §5.2.1 index 0: "harmonic mean of reliability, durability, priority, deadline, and blocking
   multipliers." Different operator, different operand set, same symbol.
10. **RQ3's stated scope under-covers §7.3.** §1.4 promises "explicit QoS feature encodings and
    multi-attribute weighting calibrations"; §7.3 delivers those plus oracle convergent validity,
    threshold/normalisation sensitivity, anti-pattern detection with CI/CD runtimes, a
    Simpson's-paradox stratification, and an attention case study. Widen RQ3 or redistribute §7.3.
11. **Mis-filed subsection content.** The RM stratified/pooled Simpson's-paradox paragraph sits under
    the heading "Anti-Pattern Detection and CI/CD Quality Gates" in §7.3, which is about a different
    analysis entirely.
12. **"Ten distinct system domains" (§8.2) vs "all six declared domains" (§7.3).** The domain-weighting
    sweep runs over "10 evaluation scenarios" with six declared domains; the external-validity claim
    says ten domains. Scenarios and domains are being used interchangeably.
13. **§8.2 cites Table 9 for numbers Table 9 does not contain.** The normalisation-defect paragraph
    attributes $\Delta\rho \in \{-0.003, 0.000, 0.000\}$ to Table 9, but Table 9's columns are $\rho$,
    $\tau$, $F_1@K$, tie-robust $F_1@K$, non-zero-impact apps, and predictive gain — no $\Delta\rho$.
    Either add the column or drop the citation.
14. **§1.3.1's third justification argues for machinery the paper no longer contains.** The
    cost-asymmetry argument ("evaluating the space of candidate architectural repairs is not
    [affordable], which is why remediation is structured as cheap proposal followed by expensive
    simulated verification") describes the retired §6. Trim to the first two justifications, or
    re-scope it explicitly as motivation for §8.3's future work.
