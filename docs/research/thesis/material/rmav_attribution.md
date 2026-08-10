# RMAV attribution: full treatment cut from the JSS condensation

> **Provenance.** Verbatim from `docs/research/jss/draft.md` §4.1–§4.6 (everything up to but not
> including §4.7, which has its own file, [relationship_criticality.md](relationship_criticality.md)),
> as they stood at commit `f0cba41822820a79ebdab123d54a76072b8f1689`. The condensed JSS `draft.md`
> retitles this section "Interpretable Attribution as a Baseline" and compresses it to ~1,200 words
> and one table (Tables 7–10 below become a single merged table); §4.6 Worked Attribution is removed
> entirely. Nothing here has been reworded — this is the source the condensed section was cut from.

---

## 4.1 Four Orthogonal Dimensions and Formal Definitions

We attribute criticality along Reliability, Maintainability, Availability, and Vulnerability (RMAV).
Grounded in **ISO/IEC 25019:2023 (Quality-in-Use)**, criticality represents the counterfactual loss of
beneficialness, freedom from risk, and acceptability experienced by stakeholders if an architectural element fails. Each dimension speaks to a formal stakeholder class:

**Table 7. The four RMAV dimensions**, the architectural question each answers, and the stakeholder and engineering role each routes to.

| Dim. | Architectural Question | High score means | Harmed Stakeholder (ISO 25019) | Secondary Stakeholder (Engineering Role) |
|:----:|-----------------------|------------------|--------------------------------------------|------------------------------------------|
| **R** | How broadly and deeply does failure propagate? | Failure cascades widely; hard to contain | **Primary & Indirect:** operators and downstream beneficiaries whose tasks retry, fail over, or degrade | Reliability Engineer |
| **M** | How hard is this to change safely? | Tightly coupled structural bottleneck | **Secondary:** maintainers facing high regression likelihood upon refactoring | Software Architect |
| **A** | Is this a structural single point of failure? | Removing it partitions the dependency graph | **Primary & Indirect:** direct operators (traders, clinicians, drivers) and dependent beneficiaries facing task cessation | DevOps / SRE |
| **V** | How attractive a target is this for attack? | Central and reachable on $G^\top$, with many strongly-guaranteed flows converging on it | **Primary, Indirect & business:** parties relying on the guarantees an attacker would gain control of | Security Engineer |

Maintainability is the one dimension whose direct victim is the secondary stakeholder; the other
three route a finding to the engineering role equipped to act on it while denominating severity in
harm to primary and indirect stakeholders. The $V$ row is deliberately phrased in terms of
*guarantees* rather than asset value: $w_{\text{in}}$ is a delivery-guarantee proxy, and the model
carries no field for data sensitivity, privilege, or PII (§9.2, §9.3).

Four formal definitions establish the theoretical construct. Each is stated in full, because
several clauses that are easy to skim past do real work in what follows.

> **Definition D1 — Component Criticality.** The degree to which the failure, latency, or functional
> degradation of a specific software component — directly or transitively — reduces the system's
> capacity to enable its stakeholders to achieve specified operational goals with beneficialness
> (usability, accessibility, suitability), freedom from risk (economic, health, life, environmental),
> and acceptability (experience, trustworthiness, compliance) within its operational context.
> Realised at layer $l$ as a measure $\mathrm{crit}_l : V_l \to [0,1]^4 \times [0,1]$ mapping each
> $v \in V_l$ to $\mathbf{s}(v) = [R(v), M(v), A(v), V(v)]^T$ and composite $Q(v)$.

*"Failure, latency, or functional degradation"* names three distinct fault modes. The structural
estimator does not separate them — RMAV scores a component's *exposure*, which is why one score
covers all three — whereas the simulation oracle does (§5.1). *"Directly or transitively"* is why
Reliability exists as a dimension separate from Availability: the harm is loss of stakeholder
outcomes reachable *through* the component, not loss of graph connectivity. *"Within its operational
context"* is the clause §4.3 operationalises through the QoS-profile adaptation of the composite
weights.

> **Definition D2 — Relationship Criticality.** The degree to which the disruption, latency, or data
> loss across a specific inter-component interaction or dependency path — **with both endpoint
> components remaining operational** — reduces the system's capacity to enable its stakeholders to
> achieve specified goals with beneficialness, freedom from risk, and acceptability, **in proportion
> to the absence of redundant or fallback paths around it**. Realised at layer $l$ as
> $\mathrm{crit}_l : E_l \to [0,1]^4 \times [0,1]$, the same signature as D1.

The first emphasised clause is what makes D2 more than D1 restated for edges: it isolates the
*partial-outage* case, in which the component is up and its dashboards are green while one data flow
has stopped. It is also exactly the condition the edge oracle enforces (§8.2). The second clause
makes replaceability *scale* the harm rather than gate it, which is why only the Availability
dimension is bridge-gated while R, M and V score replaceable links too (§4.7).

> **Definition D3 — Criticality is a consequence, not a risk.** Under the standard decomposition of
> risk into likelihood and consequence, criticality as defined here is the **consequence factor
> alone**. No RMAV dimension estimates how probable it is that a component or relationship fails;
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
feeds exactly one dimension, never more. This is a deliberate design constraint, not an empirical
observation — allowing a metric into two dimensions would silently inflate its weight relative to the
stated weighting (§4.3). Orthogonality is what makes the breakdown legible: a pure single point of
failure scores high on A but low on R, M, and V; a god-component scores high on M; a cascade hub
scores high on R. The *shape* of the profile names the failure mode. The constraint is specific to
the component decomposition; the edge formulas of §4.7 deliberately relax it in exchange for endpoint
context, and we say so there rather than letting the claim read as framework-wide.

## 4.2 RMAV Formulas

All metric inputs are rank-normalized to $[0,1]$, so every RMAV score lies in $[0,1]$. Table 8 fixes
notation for every structural metric the four formulas below consume; each is computed once on
$G_{\text{analysis}}$ and feeds exactly one RMAV dimension (§4.1).

**Table 8. RMAV input metric notation.** $G^\top$ denotes the transpose of the `DEPENDS_ON` graph
(the failure-propagation direction, since edges point dependent → dependency).

| Symbol | Name | Computed as | Feeds |
|--------|------|-------------|:-----:|
| $\mathrm{RPR}(v)$ | Reverse PageRank | PageRank on $G^\top$ ($d=0.85$) | $R$ |
| $\mathrm{DG\_in}(v)$ | In-degree (rank-norm.) | Direct dependent count on `DEPENDS_ON` | $R$ |
| $\mathrm{MPCI}(v)$ | Multi-Path Coupling Index | $\sum_{e\in\text{InEdges}(v)} \max(\text{path\_count}(e)-1,0) / (\lvert V\rvert-1)$ | $R$ (via CDPot_enh) |
| $\mathrm{CDPot\_enh}(v)$ | Enhanced Cascade Depth Potential | RPR/DG_in blend, amplified by MPCI (Eq. above) | $R$ |
| $\mathrm{FOC}(v)$ | Fan-Out Criticality | frequency- and QoS-weighted subscriber fan-out (Topic nodes only) | $R_{\text{topic}}$ |
| $\mathrm{BT}(v)$ | Betweenness centrality | Brandes' algorithm on $G_{\text{analysis}}$, QoS-inverted edge distances | $M$ |
| $w\_\text{out}(v)$ | QoS-weighted out-degree | $\sum_{(v,u)} w(v,u)$ over outgoing dependencies | $M$ |
| $\mathrm{CQP}(v)$ | Code Quality Penalty | SonarQube-derived composite (§3.4); 0 for non-App/Library types | $M$ |
| $\mathrm{CouplingRisk\_enh}(v)$ | Enhanced coupling risk | in/out-degree balance amplified by path complexity | $M$ |
| $\mathrm{CC}(v)$ | Clustering coefficient | Watts–Strogatz local clustering on the undirected projection | $M$ (as $1-\mathrm{CC}$) |
| $\mathrm{AP\_c\_directed}(v)$ | Directed articulation score | $\max$ of directed in/out articulation scores | $A$ |
| $\mathrm{QSPOF}(v)$ | QoS-weighted SPOF severity | $\mathrm{AP\_c\_directed}(v)\cdot w(v)$ | $A$ |
| $\mathrm{BR}(v)$ | Bridge ratio | fraction of $v$'s undirected edges that are bridges | $A$ |
| $\mathrm{CDI}(v)$ | Connectivity Degradation Index | normalized increase in average path length when $v$ is removed | $A$ |
| $\mathrm{REV}(v)$ | Reverse eigenvector centrality | eigenvector centrality on $G^\top$ | $V$ |
| $\mathrm{RCL}(v)$ | Reverse closeness (harmonic) | harmonic centrality on $G^\top$, normalized by $\lvert V\rvert-1$ | $V$ |
| $w\_\text{in}(v)$ | QoS-weighted in-degree (QADS) | $\sum_{(u,v)} w(u,v)$ over incoming dependencies | $V$ |

**Reliability** — fault-propagation risk. Because `DEPENDS_ON` points *dependent → dependency*, a
failure propagates *against* edge direction; RPR (computed on the transpose $G^\top$) therefore
traverses the natural failure-propagation path. For Topic nodes, which have no `DEPENDS_ON`
in-degree, a fan-out form is dispatched by $\tau_V(v)$:

$$R(v) = 0.45\cdot\mathrm{RPR}(v) + 0.30\cdot\mathrm{DG\_in}(v) + 0.25\cdot\mathrm{CDPot\_enh}(v)
\qquad [\tau_V(v)\neq\text{Topic}]$$
$$\mathrm{CDPot\_enh}(v) = \min\!\Big( \frac{\mathrm{RPR}(v) + \mathrm{DG\_in}(v)}{2} \cdot \big(1 - \min(\tfrac{\mathrm{out\_degree\_raw}(v)}{\max(\mathrm{in\_degree\_raw}(v),\, \epsilon)}, 1)\big) \cdot (1 + \mathrm{MPCI}(v)),\ 1.0 \Big)$$
$$R_{\text{topic}}(v) = 0.50\cdot\mathrm{FOC}(v) + 0.50\cdot\mathrm{CDPot\_topic}(v),\quad
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

**Availability** — single-point-of-failure risk:

$$A(v) = 0.35\,\mathrm{AP\_c\_directed}(v) + 0.25\,\mathrm{QSPOF}(v) + 0.25\,\mathrm{BR}(v)
+ 0.10\,\mathrm{CDI}(v) + 0.05\,w(v).$$

The directed articulation score (rather than the undirected AP, which both over- and under-reports
in pub-sub graphs) captures directed cut vertices; QSPOF amplifies it by the component's QoS weight,
so a SPOF carrying critical traffic is scored as doubly severe.

**Vulnerability** — adversarial exposure:

$$V(v) = 0.40\,\mathrm{REV}(v) + 0.35\,\mathrm{RCL}(v) + 0.25\,\mathrm{w\_in}(v).$$

All three terms are computed on the transpose to model attack propagation and adversarial reach
toward high-SLA surfaces.

## 4.3 The Composite Score $Q(v)$

The four dimensions combine into a composite criticality score under a stated weighting:

$$Q(v) = w_A\,A(v) + w_R\,R(v) + w_M\,M(v) + w_V\,V(v).$$

**The weights are stated design judgements, audited for coherence rather than elicited.** Each
comparison matrix is written on Saaty's 1–9 scale to express an intended ordering, then checked with
the Analytic Hierarchy Process [15]: row geometric means normalised to a weight vector, with a
consistency ratio $\mathrm{CR} = \mathrm{CI}/\mathrm{RI}$ required to satisfy
$\mathrm{CR} \le 0.10$. We describe them as "stated and audited" rather than "AHP-derived" because
the resulting near-zero consistency ratios ($\mathrm{CR} < 0.02$, and below $0.002$ on the
$5\times5$ intra-dimension matrices) are a symptom of the construction: a matrix filled in from a
target weight vector is consistent almost by construction, whereas genuine multi-rater elicitation
on five criteria rarely lands that low. The audit certifies internal coherence, not provenance.

**Three weighting paths exist in the implementation, and the reported results use the first.** We
set them out explicitly, because they do not coincide and an earlier version of this paper conflated
them:

**Table 9. The three weighting paths in the implementation.** Reported results use the stated default.

| Path | Composite $(w_A, w_R, w_M, w_V)$ | Intra-dimension | Used by |
|---|---|---|---|
| **Stated default** | $(0.43,\ 0.24,\ 0.17,\ 0.16)$ | exactly the coefficients printed in §4.2 | **all reported results** (§8.1–§8.5) |
| AHP reconstruction, $\lambda = 1$ | $(0.458,\ 0.246,\ 0.169,\ 0.128)$ | matches §4.2 to three decimals | upper endpoint of the §8.3 sweep |
| AHP with shrinkage $\lambda$ | $\lambda\,w_{\mathrm{AHP}} + (1-\lambda)\tfrac{1}{n_{\text{dim}}}$; at $\lambda = 0.70$, $(0.395,\ 0.247,\ 0.193,\ 0.165)$ | shrunk likewise | the §8.3 sensitivity sweep |

All three place Availability first (a SPOF is a certain graph partition), Reliability second (cascade
reach), then Maintainability and Vulnerability. Shrinkage blends toward a uniform prior and is
applied to the intra-dimension vectors as well as to the composite, so $\lambda$ moves every RMAV
formula at once; it exists because weight vectors from small comparison sets can be extreme. The
stated default is *not* a point on that $\lambda$ axis — it is a hand-set vector expressing the same
ordering — which is why §8.3 reports the sweep as a sensitivity analysis of the ordering rather than
as a tuning curve for a deployed parameter.

**A QoS-profile adaptation is applied on top of whichever vector is in force.** Before scoring,
the four composite coefficients are re-derived from the analysed system's aggregate QoS profile and
renormalised to sum to one: a predominantly `PERSISTENT`/`RELIABLE`/high-priority system shifts
weight toward $R$ and $A$, a predominantly `VOLATILE`/`BEST_EFFORT` one toward $M$ and $V$, and a
mixed profile keeps the stated defaults. This is D1's *"within its operational context"* clause made
computable, and it is on by default in every run reported here. Two consequences follow. The
effective composite is therefore **per system**, so the vectors tabulated above are starting points
rather than the coefficients any individual system is scored with — a further sense in which D4's
relativity holds. And it does not disturb the determinism of §4.5: the adaptation is a deterministic
function of the same $G_{\text{analysis}}$, with no learned or stochastic component.

**Quality-in-Use Transformation Matrix.** To connect product-quality mechanisms ($R, M, A, V$) to ISO/IEC 25019 Quality-in-Use harms, the vector $\mathbf{s}_{\mathrm{RMAV}}(v) = [R(v), M(v), A(v), V(v)]^T$ projects into stakeholder harm scores $[H_{\mathrm{Ben}}, H_{\mathrm{Risk}}, H_{\mathrm{Acc}}]^T$ via transformation matrix $\mathbf{M}_{\mathrm{RMAV} \to \mathrm{QiU}}$:

$$
\mathbf{h}_{\mathrm{QiU}}(v) = \mathbf{M}_{\mathrm{RMAV} \to \mathrm{QiU}} \cdot \mathbf{s}_{\mathrm{RMAV}}(v) =
\begin{bmatrix}
0.35 & 0.25 & 0.40 & 0.00 \\
0.10 & 0.00 & 0.50 & 0.40 \\
0.30 & 0.00 & 0.20 & 0.50
\end{bmatrix}
\begin{bmatrix} R(v) \\ M(v) \\ A(v) \\ V(v) \end{bmatrix}.
$$

In a specific deployment domain, Quality-in-Use loss can be further parametrized by a **Domain
Context Vector** $\vec{\omega}_{\mathrm{domain}} = [\omega_{\mathrm{Ben}}, \omega_{\mathrm{Risk}},
\omega_{\mathrm{Acc}}]$ that reweights the three harm scores — safety-critical ROS 2 prioritising
Freedom from Risk, financial HFT prioritising Efficiency under Beneficialness, and so on.

**Both $\mathbf{M}_{\mathrm{RMAV}\to\mathrm{QiU}}$ and $\vec{\omega}_{\mathrm{domain}}$ are stated
mappings, and neither is used in any result reported in this paper.** They are given here because
D1 and D2 define criticality on Quality-in-Use while the four dimensions are named after product
quality, and a reader is owed an explicit statement of how one is meant to reach the other. But the
coefficients are asserted, not fitted or elicited; unlike the composite weights they carry no
consistency audit; and no table in §8 reports an $\mathbf{h}_{\mathrm{QiU}}$ score. They should be
read as a specification of the intended correspondence, not as a validated instrument. Deriving them
— and testing whether per-domain reweighting recovers the ranking accuracy that the global weighting
does not — is future work (§9.3).

**We report the sensitivity of the composite weighting, and it is not favourable.** Sweeping
$\lambda$ over $\{0,\dots,1\}$ against simulated impact shows no plateau at any value and a monotone
decline in $\rho$, with equal weights ($\lambda = 0$) outperforming the $\lambda = 0.70$ setting by
$0.111$ (§8.3). An earlier version of this paper reported a plateau over $\lambda\in[0.65,0.75]$;
that claim was not supported by a committed artifact and does not survive measurement. Because the
decline is monotone across the whole range, the conclusion applies to the stated default of the
table above as well, even though that vector is not itself a point on the $\lambda$ axis: every
weighting that expresses the intended ordering is beaten by the uniform one on this cohort.

One reading of the decline is that a single global weighting cannot fit scenarios drawn from
domains whose harm profiles genuinely differ, which is what $\vec{\omega}_{\mathrm{domain}}$ is
meant to express. We flag that as a conjecture rather than an explanation: we have not run the
per-domain reweighting that would test it, and until we do, the measured fact is simply that the
stated weighting does not improve ranking. We keep the decomposition and drop the accuracy claim
attached to its weighting. The four dimensions earn their place by being *separately actionable* —
a structural single point of failure and a cascade hub have different owners and different remedies
even at identical composite scores (§4.1) — and that property is independent of how the four are
combined into a scalar. A practitioner optimising purely for ranking should use equal weights; a
practitioner who needs to know *why* a component is critical needs the profile, whatever the weights.

## 4.4 Adaptive Criticality Classification

A raw $Q(v)$ is most useful when turned into an action threshold relative to the system's own
distribution rather than an absolute cutoff. We classify with an adaptive box-plot rule, applied
independently to each RMAV dimension and to the composite:

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
Availability yet MINIMAL on Vulnerability, which tells the architect to add a replica rather than to
harden an interface. For small graphs ($n<12$), where quartile fences are unstable, a percentile
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
point of the table is the divergence between the last two columns:

**Table 10. Worked RMAV attribution for the running example of §3.6.** The divergence between the last two columns is the point.

| Component | $R$ | $M$ | $A$ | $V$ | $Q$ | Composite tier | Dominant dimension tier |
|---|---:|---:|---:|---:|---:|---|---|
| $b$ (broker) | 0.569 | 0.278 | 0.335 | 0.375 | 0.356 | LOW | **CRITICAL on $A$** |
| $t$ (topic) | 0.875 | 0.305 | 0.021 | 0.188 | 0.260 | MINIMAL | **CRITICAL on $R$** |
| $a_1$ (publisher) | 0.500 | 0.627 | 0.021 | 0.667 | 0.428 | MEDIUM | **CRITICAL on $M$** |
| $n$ (host) | 0.300 | 0.405 | 0.271 | 0.438 | 0.357 | MEDIUM | HIGH on $A$ |
| $\ell$ (library) | 0.450 | 0.357 | 0.050 | 0.333 | 0.266 | LOW | LOW on $R$, $M$, $V$ |

Three components illustrate how the profile names the failure mode, and each is a case the composite
alone would mislead on. The broker $b$ is a directed cut vertex: removing it partitions the graph, so
it is CRITICAL on $A$ — driven by the directed articulation score and, because $t$ carries
`RELIABLE`/`TRANSIENT_LOCAL`/`HIGH` traffic at $w(t) = 0.596$, by QSPOF — while scoring MINIMAL on
$M$. Yet its *composite* tier is LOW. An architect reading only $Q(b)$ would deprioritise the one
component whose loss stops every dependent outright; the $A$ tier is what routes it to the SRE for a
second broker. The topic $t$ inverts the same pattern on a different dimension: CRITICAL on $R$
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
