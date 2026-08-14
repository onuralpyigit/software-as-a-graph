# RM attribution: full treatment cut from the JSS condensation

> **Provenance.** Verbatim from `docs/research/jss/draft.md` §4.1–§4.6 (everything up to but not
> including §4.7, which has its own file, [relationship_criticality.md](relationship_criticality.md)),
> as they stood at commit `f0cba41822820a79ebdab123d54a76072b8f1689`. The condensed JSS `draft.md`
> retitles this section "Interpretable Attribution as a Baseline" and compresses it to ~1,200 words
> and one table (Tables 7–10 below become a single merged table); §4.6 Worked Attribution is removed
> entirely. Nothing here has been reworded (beyond this pass's RMAV→RM migration) — this is the
> source the condensed section was cut from.
>
> **This file was updated for the RMAV→RM model migration** (Vulnerability/Security retired outright;
> Availability demoted from a peer dimension to a Reliability sub-characteristic, combined via a
> declared blend $R(v)=\alpha\cdot FT(v)+(1-\alpha)\cdot A(v)$, $\alpha=0.36$). Formulas, tables and
> measured figures below reflect the new model; §4.6's worked-attribution table uses a running-example
> topology this pass could not re-run and is marked `TODO(needs re-measurement)` rather than guessed.

---

## 4.1 Two Characteristics, Hierarchical Reliability, and Formal Definitions

We attribute criticality along Reliability and Maintainability (RM). Reliability is **hierarchical**:
its Fault Tolerance and Availability sub-characteristics are scored individually and combined via a
declared blend, $R(v) = \alpha\cdot FT(v) + (1-\alpha)\cdot A(v)$, $\alpha=0.36$. An earlier revision
of this attribution scored Vulnerability/Security as a third peer dimension; it has since been
**retired outright** — not folded into either remaining characteristic — because its ground-truth
evidence was the weakest of the (then) four and no fault-model instrument could validate it by
construction (§5.1). Grounded in **ISO/IEC 25019:2023 (Quality-in-Use)**, criticality represents the
counterfactual loss of beneficialness, freedom from risk, and acceptability experienced by
stakeholders if an architectural element fails. Each dimension speaks to a formal stakeholder class:

**Table 7. The RM dimensions**, the architectural question each answers, and the stakeholder and engineering role each routes to.

| Dim. | Architectural Question | High score means | Harmed Stakeholder (ISO 25019) | Secondary Stakeholder (Engineering Role) |
|:----:|-----------------------|------------------|--------------------------------------------|------------------------------------------|
| **R** (hierarchical) | Combines the two rows below via $\alpha=0.36$ | — | Combined harm of the two rows below | Reliability Engineer / DevOps / SRE |
| ↳ **FT** | How broadly and deeply does failure propagate? | Failure cascades widely; hard to contain | **Primary & Indirect:** operators and downstream beneficiaries whose tasks retry, fail over, or degrade | Reliability Engineer |
| ↳ **A** | Is this a structural single point of failure? | Removing it partitions the dependency graph | **Primary & Indirect:** direct operators (traders, clinicians, drivers) and dependent beneficiaries facing task cessation | DevOps / SRE |
| **M** | How hard is this to change safely? | Tightly coupled structural bottleneck | **Secondary:** maintainers facing high regression likelihood upon refactoring | Software Architect |

Maintainability is the one dimension whose direct victim is the secondary stakeholder; the other
two sub-characteristics route a finding to the engineering role equipped to act on it while
denominating severity in harm to primary and indirect stakeholders. (An earlier revision's $V$ row
was deliberately phrased in terms of *guarantees* rather than asset value — that dimension, and the
distinction, are retired along with it.)

Four formal definitions establish the theoretical construct. Each is stated in full, because
several clauses that are easy to skim past do real work in what follows.

> **Definition D1 — Component Criticality.** The degree to which the failure, latency, or functional
> degradation of a specific software component — directly or transitively — reduces the system's
> capacity to enable its stakeholders to achieve specified operational goals with beneficialness
> (usability, accessibility, suitability), freedom from risk (economic, health, life, environmental),
> and acceptability (experience, trustworthiness, compliance) within its operational context.
> Realised at layer $l$ as a measure $\mathrm{crit}_l : V_l \to [0,1]^2 \times [0,1]$ mapping each
> $v \in V_l$ to $\mathbf{s}(v) = [R(v), M(v)]^T$ (with $R(v)$ itself decomposable into $[FT(v), A(v)]$)
> and composite $Q(v)$.

*"Failure, latency, or functional degradation"* names three distinct fault modes. The structural
estimator does not separate them — RM scores a component's *exposure*, which is why one score
covers all three — whereas the simulation oracle does (§5.1). *"Directly or transitively"* is why
Fault Tolerance exists as a sub-characteristic separate from Availability: the harm is loss of
stakeholder outcomes reachable *through* the component, not loss of graph connectivity. *"Within its
operational context"* is the clause §4.3 operationalises through the QoS-profile adaptation of the
composite weights.

> **Definition D2 — Relationship Criticality.** The degree to which the disruption, latency, or data
> loss across a specific inter-component interaction or dependency path — **with both endpoint
> components remaining operational** — reduces the system's capacity to enable its stakeholders to
> achieve specified goals with beneficialness, freedom from risk, and acceptability, **in proportion
> to the absence of redundant or fallback paths around it**. Realised at layer $l$ as
> $\mathrm{crit}_l : E_l \to [0,1]^2 \times [0,1]$, the same signature as D1.

The first emphasised clause is what makes D2 more than D1 restated for edges: it isolates the
*partial-outage* case, in which the component is up and its dashboards are green while one data flow
has stopped. It is also exactly the condition the edge oracle enforces (§8.2). The second clause
makes replaceability *scale* the harm rather than gate it, which is why only the Availability
sub-characteristic is bridge-gated while FT and M score replaceable links too (§4.7).

> **Definition D3 — Criticality is a consequence, not a risk.** Under the standard decomposition of
> risk into likelihood and consequence, criticality as defined here is the **consequence factor
> alone**. No RM dimension estimates how probable it is that a component or relationship fails;
> every dimension estimates how much is lost *given* that it does.

Two consequences bear directly on how the results of §8 should be read. Ranking $u$ above $v$ says
that losing $u$ hurts more, not that $u$ is more likely to be lost — so every comparison in this
paper holds likelihood fixed. And restricting the construct to consequence is precisely what makes
it computable pre-deployment: consequence follows from architecture, which exists before the system
runs; likelihood follows from behaviour, which does not.

> **Definition D4 — Criticality is relative, not absolute.** Every score and tier is relative to
> (i) the score distribution of the system $S$ being analysed, since tiers are box-plot thresholds
> over that distribution (§4.4), and (ii) the layer $l$, since both the vertex set being ranked and
> the weight normalisation change with the projection. Criticality values are therefore **not
> comparable across systems or across layers**.

A well-designed redundant system still has a CRITICAL tier, and a system full of SPOFs still has a
MINIMAL tier; the tier prioritises attention inside one system rather than comparing two. D4 also
constrains how this paper may aggregate: any figure computed over more than one scenario must be
formed from within-scenario ranks or per-scenario statistics, never from raw scores pooled across
systems. §5.5 and §8.5 carry the corresponding scoping statements.

For **components**, the dimensions are **orthogonal by construction**: each raw structural metric
feeds exactly one of FT, M, A, never more. This is a deliberate design constraint, not an empirical
observation — allowing a metric into two dimensions would silently inflate its weight relative to the
stated weighting (§4.3). Orthogonality is what makes the breakdown legible: a pure single point of
failure scores high on A (and therefore high on R) but low on FT and M; a god-component scores high
on M; a cascade hub scores high on FT (and therefore high on R). The *shape* of the profile names the
failure mode. The constraint is specific to the component decomposition; the edge formulas of §4.7
deliberately relax it in exchange for endpoint context, and we say so there rather than letting the
claim read as framework-wide.

## 4.2 RM Formulas

All metric inputs are rank-normalized to $[0,1]$, so every RM score lies in $[0,1]$. Table 8 fixes
notation for every structural metric the formulas below consume; each is computed once on
$G_{\text{analysis}}$ and feeds exactly one of FT, M, A (§4.1).

**Table 8. RM input metric notation.** $G^\top$ denotes the transpose of the `DEPENDS_ON` graph
(the failure-propagation direction, since edges point dependent → dependency). Three metrics
(REV, RCL, $w_{\text{in}}$/QADS) that fed an earlier revision's retired Vulnerability dimension are
still computed by `StructuralAnalyzer` but no longer read by any RM formula — listed here for
completeness, marked accordingly.

| Symbol | Name | Computed as | Feeds |
|--------|------|-------------|:-----:|
| $\mathrm{RPR}(v)$ | Reverse PageRank | PageRank on $G^\top$ ($d=0.85$) | $FT$ |
| $\mathrm{DG\_in}(v)$ | In-degree (rank-norm.) | Direct dependent count on `DEPENDS_ON` | $FT$ |
| $\mathrm{MPCI}(v)$ | Multi-Path Coupling Index | $\sum_{e\in\text{InEdges}(v)} \max(\text{path\_count}(e)-1,0) / (\lvert V\rvert-1)$ | $FT$ (via CDPot_enh) |
| $\mathrm{CDPot\_enh}(v)$ | Enhanced Cascade Depth Potential | RPR/DG_in blend, amplified by MPCI (Eq. above) | $FT$ |
| $\mathrm{FOC}(v)$ | Fan-Out Criticality | frequency- and QoS-weighted subscriber fan-out (Topic nodes only) | $FT_{\text{topic}}$ |
| $\mathrm{BT}(v)$ | Betweenness centrality | Brandes' algorithm on $G_{\text{analysis}}$, QoS-inverted edge distances | $M$ |
| $w\_\text{out}(v)$ | QoS-weighted out-degree | $\sum_{(v,u)} w(v,u)$ over outgoing dependencies | $M$ |
| $\mathrm{CQP}(v)$ | Code Quality Penalty | SonarQube-derived composite (§3.4); 0 for non-App/Library types | $M$ |
| $\mathrm{CouplingRisk\_enh}(v)$ | Enhanced coupling risk | in/out-degree balance amplified by path complexity | $M$ |
| $\mathrm{CC}(v)$ | Clustering coefficient | Watts–Strogatz local clustering on the undirected projection | $M$ (as $1-\mathrm{CC}$) |
| $\mathrm{AP\_c\_directed}(v)$ | Directed articulation score | $\max$ of directed in/out articulation scores | $A$ |
| $\mathrm{QSPOF}(v)$ | QoS-weighted SPOF severity | $\mathrm{AP\_c\_directed}(v)\cdot w(v)$ | $A$ |
| $\mathrm{BR}(v)$ | Bridge ratio | fraction of $v$'s undirected edges that are bridges | $A$ |
| $\mathrm{CDI}(v)$ | Connectivity Degradation Index | normalized increase in average path length when $v$ is removed | $A$ |
| $\mathrm{REV}(v)$ | Reverse eigenvector centrality | eigenvector centrality on $G^\top$ | *retired* |
| $\mathrm{RCL}(v)$ | Reverse closeness (harmonic) | harmonic centrality on $G^\top$, normalized by $\lvert V\rvert-1$ | *retired* |
| $w\_\text{in}(v)$ | QoS-weighted in-degree (QADS) | $\sum_{(u,v)} w(u,v)$ over incoming dependencies | *retired* |

**Reliability** is a hierarchical blend of its two sub-characteristics:

$$R(v) = \alpha\cdot FT(v) + (1-\alpha)\cdot A(v), \qquad \alpha = 0.36$$

**Fault Tolerance** — fault-propagation risk. Because `DEPENDS_ON` points *dependent → dependency*, a
failure propagates *against* edge direction; RPR (computed on the transpose $G^\top$) therefore
traverses the natural failure-propagation path. For Topic nodes, which have no `DEPENDS_ON`
in-degree, a fan-out form is dispatched by $\tau_V(v)$:

$$FT(v) = 0.45\cdot\mathrm{RPR}(v) + 0.30\cdot\mathrm{DG\_in}(v) + 0.25\cdot\mathrm{CDPot\_enh}(v)
\qquad [\tau_V(v)\neq\text{Topic}]$$
$$\mathrm{CDPot\_enh}(v) = \min\!\Big( \frac{\mathrm{RPR}(v) + \mathrm{DG\_in}(v)}{2} \cdot \big(1 - \min(\tfrac{\mathrm{out\_degree\_raw}(v)}{\max(\mathrm{in\_degree\_raw}(v),\, \epsilon)}, 1)\big) \cdot (1 + \mathrm{MPCI}(v)),\ 1.0 \Big)$$
$$FT_{\text{topic}}(v) = 0.50\cdot\mathrm{FOC}(v) + 0.50\cdot\mathrm{CDPot\_topic}(v),\quad
\mathrm{CDPot\_topic}(v) = \mathrm{FOC}(v)\big(1 - \min(\text{publisher\_count\_norm}(v),1)\big)$$

**Maintainability** — coupling complexity:

$$M(v) = 0.35\,\mathrm{BT}(v) + 0.30\,\mathrm{w\_out}(v) + 0.15\,\mathrm{CQP}(v)
+ 0.12\,\mathrm{CouplingRisk\_enh}(v) + 0.08\,(1-\mathrm{CC}(v)),$$
$$\mathrm{CQP}(v) = 0.10\,\text{loc\_norm} + 0.35\,\text{complexity\_norm}
+ 0.30\,\text{instability\_code} + 0.25\,\text{lcom\_norm}.$$

Here, the Code Quality Penalty (CQP) translates local code-level fragility into system-level
maintainability risk. The components `loc_norm`, `complexity_norm`, and `lcom_norm` represent the
min-max normalized values of the ingested SonarQube properties `loc`, `cyclomatic_complexity`, and
`lcom`, respectively. These are calculated independently for Applications and Libraries to prevent
scale differences from distorting the normalization. The metric `instability_code` represents class
instability (efferent coupling divided by total coupling). The CQP thus ensures that local code debt
is penalised, but only as a sub-factor of Maintainability ($M$), which remains heavily weighted by
topological metrics such as betweenness centrality ($BT$) and efferent QoS-weighted out-degree
($w\_out$). CQP is zero for non-Application/Library types (graceful degradation). The two
instability signals are intentional and distinct: `instability_code` is static-code fragility
(local); `CouplingRisk_enh` is runtime-topology fragility (global).

**Availability** — single-point-of-failure risk; a Reliability sub-characteristic, feeding $R(v)$
above rather than the composite directly:

$$A(v) = 0.35\,\mathrm{AP\_c\_directed}(v) + 0.25\,\mathrm{QSPOF}(v) + 0.25\,\mathrm{BR}(v)
+ 0.10\,\mathrm{CDI}(v) + 0.05\,w(v).$$

The directed articulation score (rather than the undirected AP, which both over- and under-reports
in pub-sub graphs) captures directed cut vertices; QSPOF amplifies it by the component's QoS weight,
so a SPOF carrying critical traffic is scored as doubly severe.

(An earlier revision scored a fourth dimension, **Vulnerability** — adversarial exposure via
$V(v) = 0.40\,\mathrm{REV}(v) + 0.35\,\mathrm{RCL}(v) + 0.25\,\mathrm{w\_in}(v)$, all three terms
computed on the transpose to model attack propagation. It has been retired outright, not folded into
FT, A, or M; REV/RCL/$w_{\text{in}}$ are still computed by `StructuralAnalyzer` but read by no RM
formula.)

## 4.3 The Composite Score $Q(v)$

The two characteristics combine into a composite criticality score:

$$R(v) = \alpha\, FT(v) + (1-\alpha)\, A(v), \qquad Q(v) = w_R\, R(v) + w_M\, M(v).$$

**The weights are DECLARED constants, not AHP output.** An earlier revision derived the (then) four
composite weights from a $4\times4$ AHP comparison matrix, audited for coherence rather than
elicited. With only two composite terms remaining, a $2\times2$ Saaty matrix would be consistent by
construction ($\mathrm{CR}=0$ for $n\le2$) and would contribute nothing beyond whichever single free
parameter is chosen — so AHP is retired at the composite level entirely. AHP remains in use for the
genuinely multi-term *intra*-dimension vectors ($FT$'s 3 terms, $M$'s 5, $A$'s 5), where a $3$–$5$
dimensional matrix has a non-trivial consistency ratio and still earns its place (§8.3's shrinkage
sweep concerns only those vectors, not the composite — see below).

**$w_R=0.80$, $w_M=0.20$, and $\alpha=0.36$ are a pure re-parameterisation of the retired composite,
not an independently invented weighting.** They are derived algebraically from the old $4$-D vector
$(A{=}0.43, R{=}0.24, M{=}0.17, V{=}0.16)$ by dropping $V$ and renormalising, then folding $A$ into
$R$:

$$\alpha = \frac{0.24}{0.24+0.43} = 0.3582 \to 0.36, \qquad
w_R = \frac{0.24+0.43}{0.84} = 0.7976 \to 0.80, \qquad
w_M = \frac{0.17}{0.84} = 0.2024 \to 0.20.$$

At exact (unrounded) values this recovers the old composite's $A$/$R$/$M$ shares exactly; the 2-s.f.
rounding introduces a small, bounded drift ($\le 0.003$ per term) rather than a fresh judgement.

**A QoS-profile adaptation is applied on top.** Before scoring, the composite coefficients are
re-derived from the analysed system's aggregate QoS profile and renormalised to sum to one: a
predominantly `PERSISTENT`/`RELIABLE`/high-priority system shifts weight toward $R$, a predominantly
`VOLATILE`/`BEST_EFFORT` one toward $M$, and a mixed profile keeps the stated defaults. This is D1's
*"within its operational context"* clause made computable, and it is on by default in every run
reported here. Two consequences follow. The effective composite is therefore **per system**, so the
constants above are starting points rather than the coefficients any individual system is scored
with — a further sense in which D4's relativity holds. And it does not disturb the determinism of
§4.5: the adaptation is a deterministic function of the same $G_{\text{analysis}}$, with no learned
or stochastic component.

**Quality-in-Use Transformation Matrix.** To connect product-quality mechanisms ($R, M$) to ISO/IEC 25019 Quality-in-Use harms, the vector $\mathbf{s}_{\mathrm{RM}}(v) = [R(v), M(v)]^T$ projects into stakeholder harm scores $[H_{\mathrm{Ben}}, H_{\mathrm{Risk}}, H_{\mathrm{Acc}}]^T$ via transformation matrix $\mathbf{M}_{\mathrm{RM} \to \mathrm{QiU}}$:

$$
\mathbf{h}_{\mathrm{QiU}}(v) = \mathbf{M}_{\mathrm{RM} \to \mathrm{QiU}} \cdot \mathbf{s}_{\mathrm{RM}}(v) =
\begin{bmatrix}
0.75 & 0.25 \\
0.80 & 0.20 \\
0.60 & 0.40
\end{bmatrix}
\begin{bmatrix} R(v) \\ M(v) \end{bmatrix}.
$$

(Row 1 is an unchanged mechanical fold of the old $A$ column into $R$ — $A$'s coefficient there was
already $0.00$ in the retired $3\times4$ matrix. Rows 2 and 3 are re-declared, not mechanically
derived: the mechanical fold alone collapses them to the same vector, making the matrix rank-1 in
$\vec\omega$ and tying every domain's Beneficialness weight — disqualifying. Row 2's re-declared
$0.20$ on $M$ is maintainability's MTTR channel; row 3's $0.40$ is its evolvability-into-trust
channel.)

In a specific deployment domain, Quality-in-Use loss can be further parametrized by a **Domain
Context Vector** $\vec{\omega}_{\mathrm{domain}} = [\omega_{\mathrm{Ben}}, \omega_{\mathrm{Risk}},
\omega_{\mathrm{Acc}}]$ that reweights the three harm scores — safety-critical ROS 2 prioritising
Freedom from Risk, financial HFT prioritising Efficiency under Beneficialness, and so on.

**Both $\mathbf{M}_{\mathrm{RM}\to\mathrm{QiU}}$ and $\vec{\omega}_{\mathrm{domain}}$ are stated
mappings.** They are given here because D1 and D2 define criticality on Quality-in-Use while the
dimensions are named after product quality, and a reader is owed an explicit statement of how one is
meant to reach the other. The coefficients are asserted, not fitted or elicited; they carry no
consistency audit. Unlike an earlier revision of this section, per-domain reweighting *has* now been
measured (`reproduce/domain_weight_comparison.py`): mean Kendall $\tau$ between domain-derived and
static composite rankings is $0.968$ across all six declared domains — domain-derived $w_R$ sits
within $0.04$ of the static default across every domain, so the two weightings rank components almost
identically. The derivation's value remains attributional (explaining criticality in stakeholder
terms), not a ranking-improvement device — see §8.3.

**We report the sensitivity of the (now intra-dimension-only) shrinkage weighting, and the finding's
shape has changed, not just its numbers.** Sweeping $\lambda$ over $\{0,\dots,1\}$ against simulated
impact now shows a *monotone increase* in mean $\rho$ (not a decrease, and not a plateau): the raw
AHP judgement at $\lambda=1$ (mean $\rho=-0.0067$) is mildly better than uniform intra-dimension
weights at $\lambda=0$ (mean $\rho=-0.0512$), a lift of $+0.0324$. The composite itself is
$\lambda$-invariant by construction — every row of the sweep used the identical $w_R=0.80$,
$w_M=0.20$ — so this sweep characterises only the FT/M/A/Impact intra-dimension vectors, not the
R-vs-M weighting. An earlier revision of this paper reported the opposite shape (monotone decrease,
no plateau, equal weights beating the default by $0.111$) under the retired 4-D composite, where
$\lambda=0$ meant "equal weights over four composite dimensions" — that comparison no longer exists.
All values in the new sweep are near-zero-to-slightly-negative, consistent with closed-form,
non-learned $Q(v)$ being weak at this scale throughout this cohort (§8.3), not evidence the migration
broke something.

The four (now two) dimensions earn their place by being *separately actionable* — a structural
single point of failure and a cascade hub have different owners and different remedies even at
identical composite scores (§4.1) — and that property is independent of how they are combined into a
scalar. A practitioner who needs to know *why* a component is critical needs the profile, whatever
the weights.

## 4.4 Adaptive Criticality Classification

A raw $Q(v)$ is most useful when turned into an action threshold relative to the system's own
distribution rather than an absolute cutoff. We classify with an adaptive box-plot rule, applied
independently to each RM dimension/sub-characteristic and to the composite:

$$
\text{CRITICAL}: Q > Q_3 + 1.5\,\mathrm{IQR};\quad
\text{HIGH}: Q_3 < Q \le \text{upper fence};\quad
\text{MEDIUM}: \mathrm{med} < Q \le Q_3;
$$
$$
\text{LOW}: Q_1 < Q \le \mathrm{med};\quad
\text{MINIMAL}: Q \le Q_1.
$$

Per-dimension classification is what makes the output actionable: a component can be CRITICAL on
Availability yet MINIMAL on Maintainability, which tells the architect to add a replica rather than
to decouple an interface. For small graphs ($n<12$), where quartile fences are unstable, a percentile
fallback is used (CRITICAL = top 10%, HIGH = 75th–90th, MEDIUM = 50th–75th, LOW = 25th–50th,
MINIMAL = bottom 25%).

## 4.5 Determinism and the Independence Guarantee

Attribution is fully deterministic and interpretable: the same $G_{\text{analysis}}$ always yields
the same scores, with no learned parameters and no stochastic component. Critically, every input to
$Q(v)$ is a structural metric of $G_{\text{analysis}}$; none derives from the discrete-event
simulation that produces the ground-truth impact labels used to evaluate the framework (§5.1, §7.5).
This is the **independence guarantee**: the attribution path and the label path are disjoint, so a
correlation between $Q(v)$ and simulated impact — under either oracle — measures genuine predictive
content rather than information leaked from the labels into the score.

## 4.6 Worked Attribution

Scoring the running example of §3.6 with the pipeline of §4.2–§4.4 gives the following profile. The
point of the table is the divergence between the last two columns.

> `TODO(needs re-measurement)`: **Table 10 below is the pre-migration (RMAV) worked example,
> retained for structural reference only — its numbers are stale and must not be cited.** This pass
> could not re-run the specific running-example topology of §3.6 (not part of the fresh
> `reproduce/` artifacts this migration produced) to regenerate $FT(v)$, $A(v)$, $R(v)$, $M(v)$,
> $Q(v)$ for these five components under the new hierarchical model. Whoever next touches this
> section should re-run the §3.6 topology through the current pipeline (see
> `examples/run_structural_analysis.py` for the calling pattern) and replace the table and the
> prose below it, which still describes the old $A$-as-peer-dimension framing throughout.

**Table 10 (STALE — pre-migration). Worked RMAV attribution for the running example of §3.6.** The divergence between the last two columns is the point.

| Component | $R$ | $M$ | $A$ | $V$ | $Q$ | Composite tier | Dominant dimension tier |
|---|---:|---:|---:|---:|---:|---|---|
| $b$ (broker) | 0.569 | 0.278 | 0.335 | 0.375 | 0.356 | LOW | **CRITICAL on $A$** |
| $t$ (topic) | 0.875 | 0.305 | 0.021 | 0.188 | 0.260 | MINIMAL | **CRITICAL on $R$** |
| $a_1$ (publisher) | 0.500 | 0.627 | 0.021 | 0.667 | 0.428 | MEDIUM | **CRITICAL on $M$** |
| $n$ (host) | 0.300 | 0.405 | 0.271 | 0.438 | 0.357 | MEDIUM | HIGH on $A$ |
| $\ell$ (library) | 0.450 | 0.357 | 0.050 | 0.333 | 0.266 | LOW | LOW on $R$, $M$, $V$ |

*(Under the new model, $V$ no longer exists and $A$ is no longer an independent composite column —
it is a sub-characteristic feeding $R$. The table above cannot be mechanically translated; it needs
a fresh pipeline run, not a relabelling.)*

Three components illustrate how the profile names the failure mode, and each is a case the composite
alone would mislead on. The broker $b$ is a directed cut vertex: removing it partitions the graph, so
it is CRITICAL on $A$ — driven by the directed articulation score and, because $t$ carries
`RELIABLE`/`TRANSIENT_LOCAL`/`HIGH` traffic at $w(t) = 0.596$, by QSPOF — while scoring MINIMAL on
$M$. Yet its *composite* tier is LOW. An architect reading only $Q(b)$ would deprioritise the one
component whose loss stops every dependent outright; the $A$ tier is what routes it to the SRE for a
second broker. The topic $t$ inverts the same pattern on a different dimension: CRITICAL on $FT$
through its subscriber fan-out, MINIMAL overall. The publisher $a_1$ is CRITICAL on $M$ — a
betweenness and efferent-coupling bottleneck the architect should decouple — at a composite of only
MEDIUM.

This is the concrete form of the claim §8.3 arrives at empirically. The composite is a ranking
device and, on this cohort, not a good one; the *profile* is the diagnostic, and the two disagree
often enough that reading only the scalar discards the finding. Reading the broker row as a
stakeholder statement, in the terms of §4.1: if $b$ fails, $a_2$ and $a_3$ lose their only path to
$t$, so the monitoring task does not degrade — it stops. That is a Beneficialness/Effectiveness
loss, and the outage window is itself a Freedom-from-risk exposure. What the score does *not* say is
how often $b$ fails or how fast it would be restored (D3); a CRITICAL tier is a statement about
structural exposure to Quality-in-Use loss, not a measurement of Quality-in-Use loss itself (§9.2).

The edge scores of §4.7 add the complementary reading. Only one dependency in the example is a
bridge — $n \to b$, at $A = 0.354$ and $Q = 0.320$, the highest-scoring edge — while every
`app_to_broker`, `app_to_app` and `app_to_lib` edge scores $A = 0.004$: replaceable links, whose loss
costs Efficiency rather than Effectiveness. Those replaceable edges nonetheless carry
$R \approx 0.29$, because $w(e) = 0.596$ and their endpoints' own reliability reach them through the
endpoint term. That is D2's proportionality clause behaving as specified: redundancy scales the harm
to near zero on $A$ without switching the other three dimensions off.

The shared library $\ell$ illustrates
the qualitatively distinct simultaneous-blast mechanism of Rule 5 (§3.3): its individual structural
centrality need not be remarkable, yet its failure collapses $a_1, a_2, a_3$ at once, in a single
event rather than a propagation chain. Whether this mechanism produces a low-$Q$/high-$I$ mismatch in
practice is an empirical question we evaluate directly in §5.4 (on our synthetic suite, it does not);
independent of that, the mechanism is why the FanOutReduction operator (§6) is triggered by structural
blast signals rather than by $Q(v)$ itself — a library's consumer fan-out is legible from structure
alone, before any simulation is run.
