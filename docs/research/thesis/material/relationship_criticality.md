# Relationship criticality: full treatment cut from the JSS condensation

> **Provenance.** Verbatim from `docs/research/jss/draft.md` §4.7, as it stood at commit
> `f0cba41822820a79ebdab123d54a76072b8f1689`. The condensed JSS `draft.md` removes this subsection
> entirely — it is, by the draft's own admission below, *defined but not validated*: §8.2 states
> outright that the edge-removal measurement does not validate it, because the two are computed over
> populations that barely intersect. Removing it from the journal paper costs no evidence. It is
> preserved here in full because the edge-level RMAV construction (the $\max$/$\min$ combination
> rules, the bridge-gating design, the endpoint-context tradeoff against orthogonality) is exactly
> the kind of design material a thesis chapter on multi-dimensional attribution would want, even
> though the journal paper cannot currently support it as a validated result.

---

## 4.7 Relationship Criticality

D2 gives edges the same signature as nodes, and this section supplies the corresponding measure. The
motivation is that an edge failure and a node failure produce different observable symptoms. A node
failure is a *total* outage of a capability: everything the component provides stops. An edge failure
is a *partial* outage: the component is up, its other consumers are fine, its dashboards are green —
but one data flow has stopped, and for the stakeholder behind that link, Effectiveness is lost just
as completely as in a full outage. Two cases follow that endpoint scores cannot express. A
high-criticality node may have uniformly low-criticality edges, as with a redundantly connected
broker where losing any single link changes nothing; and a low-criticality node may sit behind a
single highly critical bridge edge, where losing that one relationship is as consequential for its
dependents as losing a much higher-scoring component.

**Structural edge signals.** Four per-edge quantities are computed on $G_{\text{analysis}}$:

**Table 11. Per-edge structural signals** computed on $G_{\text{analysis}}$ for relationship criticality.

| Signal | Computed as | Reads as |
|---|---|---|
| $\mathbf{1}_{\text{bridge}}(e)$ | cut-edge test on the undirected projection | removing $e$ disconnects a subgraph — the Effectiveness case |
| $\mathrm{bt}(e)$ | edge betweenness on **inverted** weights, each edge's length $1/w(e)$ | fraction of shortest dependency paths crossing $e$ — the Efficiency case (how much traffic must reroute) |
| $w(e)$ | worst-case (max) QoS weight over the topics mediating the dependency (§3.3) | how strongly the flow across $e$ is guaranteed |
| $\text{path\_count}(e)$ | number of distinct mediating topics or shared hosts | coupling intensity, kept out of $w(e)$ to preserve $w\in[0,1]$ |

Weight inversion is what makes strongly-guaranteed dependencies *short*, so they attract shortest
paths rather than repelling them. Unlike the node case, $w(e)$ enters **un-normalised**: the §3.2
construction already places it in $[0,1]$.

**Edge RMAV.** Each edge is scored on the same four dimensions, blending its intrinsic signals with
the endpoint scores of §4.2:

$$R(u,v) = 0.35\,\mathrm{bt} + 0.30\,w(e) + 0.20\max\big(R(u), R(v)\big)$$
$$M(u,v) = 0.35\,\mathrm{bt} + 0.30\,\mathbf{1}_{\text{bridge}} + 0.15\,w(e)$$
$$A(u,v) = 0.30\,\mathbf{1}_{\text{bridge}} + 0.20\min\big(A(u), A(v)\big)$$
$$V(u,v) = 0.15\,w(e) + 0.20\max\big(V(u), V(v)\big)$$

combined into $Q(u,v)$ with the same composite coefficients and QoS-profile adaptation as a node
(§4.3), and classified by the same box-plot rule (§4.4) applied within the edge set.

Four design choices carry meaning. **$\max$ for $R$ and $V$, $\min$ for $A$**: a link is only as
reliable or as secure as its *riskiest* endpoint, since failure or compromise on either side
propagates across it, but only as available as its *weakest*, since the edge cannot be more resilient
than the more fragile side it connects. **$\mathbf{1}_{\text{bridge}}$ appears in both $M$ and $A$**:
a non-redundant edge is expensive to route around (an Efficiency cost to the engineering stakeholder)
*and* a structural cut-point if removed (an Effectiveness loss to the end user) — one structural
fact, two stakeholder consequences. **$w(e)$ appears in $R$, $M$ and $V$ but not $A$**: the guarantee
crossing a link scales how much its loss costs, but not whether it can be lost at all. Replaceability
is topological; consequence is QoS-weighted. This is D2's redundancy clause made operational — only
$A$ is bridge-gated, while the other three score replaceable links too. **$\text{path\_count}$ does
not enter the edge score directly**; it shapes the endpoints' $R$ and $M$ (§4.2), of which only $R$
reaches the edge again, through the endpoint term.

**Two scoping conditions.** First, the orthogonality constraint of §4.1 is a property of the *node*
decomposition and does not carry over here: $\mathrm{bt}$ feeds both $R$ and $M$,
$\mathbf{1}_{\text{bridge}}$ feeds both $M$ and $A$, and $w(e)$ feeds three of the four. The edge
formulas trade orthogonality for the endpoint context that distinguishes an edge score from a node
score, and we state the claim as node-scoped rather than framework-wide. Second, the four edge
dimensions do not draw on equal coefficient mass — $R$ sums to $0.85$ of a possible $1.0$, $M$ to
$0.80$, $A$ to $0.50$, $V$ to $0.35$ — so raw edge scores are comparable *within* a dimension but not
*across* dimensions. Because classification is box-plot relative within the edge set, per-dimension
rankings and tiers are unaffected; only the raw magnitudes are. An edge's dimension *tiers* should be
read, not its absolute dimension values.

**What validates this, and what does not.** Relationship attribution is scored over
$G_{\text{analysis}}$ — the derived `DEPENDS_ON` edges — while the edge-removal oracle of §8.2 severs
raw edges of $G_{\text{structural}}$. On `av_system` those are 3,753 derived edges against a
candidate set of 50 raw structural edges drawn predominantly from `RUNS_ON` and `CONNECTS_TO`, with a
handful of `SUBSCRIBES_TO` and `PUBLISHES_TO` relations (§8.2 gives the exact composition), and the
two populations barely intersect. This is not an oversight: it is the independence guarantee of §5.3
operating exactly as designed — predictors and labels must be computed over disjoint graph views —
and the edge case simply has no shared identifier space for the two views to meet on, where the node
case does. **There is therefore no common edge population on which $Q(u,v)$ and the measured edge
impact are both defined**, and the correlation-style validation applied to node scores in §8.1 cannot
be run for edges as the two quantities are currently constructed. We present relationship attribution
as a *defined and implemented* measure that operationalises D2, and the edge-removal measurement of
§8.2 as a separate result about the structural graph — not as a validation of the attribution, and we
do not report or imply a correlation between the two anywhere in this paper. Re-simulating on
`DEPENDS_ON` directly is not an available fix: the framework's independence guarantee (§5.3) requires
simulation to operate only on $G_{\text{structural}}$. The one route that would close the gap without
violating that guarantee — tracking, for each derived edge, which raw structural edges mediate it,
then aggregating their measured impact onto it — is a modelling exercise in its own right (the
mediating relations are many-to-many, so the aggregation rule is a choice, not a formality) and is out
of scope for this submission; we position it as future work in §9.3 rather than as a pending fix.
