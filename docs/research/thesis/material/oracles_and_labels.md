# Simulation oracles and label quality: full treatment cut from the JSS condensation

> **Provenance.** Verbatim from `docs/research/jss/draft.md` §5.4 and §5.5 (in full — §5.1 and §7.5
> below are the ground-truth sections these depend on, included for self-containedness) and §7.5, as
> they stood at commit `f0cba41822820a79ebdab123d54a76072b8f1689`. The condensed JSS `draft.md` keeps
> §5.1 in full (the three-oracle definitions are core dependability content and are not cut) and
> §7.5's headline agreement figures (ρ = 0.394, ρ = 0.765, the 30–47% unlabelled bound), but
> compresses §5.4 and §5.5 to one paragraph each — both were already restated nearly in full in §8.2,
> so the condensation removes ~700 words of duplication rather than removing a finding. This file
> preserves the full original treatment, including Figure 3 and the per-scenario walkthroughs that
> the condensed draft drops. Nothing here has been reworded.

---

## 5.1 Ground Truth: Three Simulation Oracles

In the absence of runtime telemetry, ground truth is produced by discrete-event failure simulation
over the *raw* structural graph $G_{\text{structural}}$ — directly on `PUBLISHES_TO`,
`SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, and `USES` edges, without the derived
`DEPENDS_ON` projection. For each component $v$, a failure is injected at $v$, the resulting
disruption is propagated through the topology over a fixed horizon, and the residual service
degradation is measured.

**The framework contains three such oracles, and they are not interchangeable.** We name them here
rather than later, because which one backs a given number materially bounds what that number can
support:

- **$I^*(v)$** — produced by `FaultInjector`. The mean subscriber feed-loss fraction under a
  breadth-first cascade. This is the label the learned predictors are trained and evaluated
  against, and it backs the predictor tables of §8.1.
- **$I_{\text{comp}}(v)$** — produced by `FailureSimulator`. A four-component weighted composite,

  $$I_{\text{comp}}(v) = 0.35\,\text{reachability\_loss} + 0.25\,\text{fragmentation}
  + 0.25\,\text{throughput\_loss} + 0.15\,\text{flow\_disruption},$$

  where reachability_loss is the fraction of weighted publisher→topic→subscriber paths broken,
  fragmentation is the post-removal graph-partition severity, throughput_loss is the fraction of
  topic-weight throughput disrupted, and flow_disruption is the fraction of complete
  pub→topic→sub flow triples broken. The score is graded in $[0,1]$, and its component weights are
  stated judgements checked for AHP consistency, on the same footing as those of §4.3.
  $I_{\text{comp}}$ backs the validation gates, the library and stratified analyses of §5.4–§5.5,
  and the remediation acceptance test of §6.4.
- **$I_{\text{dyn}}(v)$** — produced by `MessageFlowSimulator`. The drop in delivered message rate
  that the *surviving* consumers experience when $v$ fails,

  $$I_{\text{dyn}}(v) = \text{delivery\_rate}_{\text{before}} - \text{delivery\_rate}_{\text{after}},$$

  obtained not by traversing edges but by discrete-event simulation of the actual traffic: each
  publisher emits at its topic's declared rate, every topic fans out into a bounded per-subscriber
  queue, and the fault is injected mid-run. Both windows exclude the faulted node's own receipts,
  and a silenced publisher's unmet demand stays in the denominator, so a component is credited only
  with the damage it does to *others*. $I_{\text{dyn}}$ trains nothing and gates nothing: it is
  reported in §7.5 as a construct-validity check on the other two, and is used for no other purpose
  in this paper.

$I^*$ and $I_{\text{comp}}$ agree only weakly — mean Spearman $\rho = 0.394$ across the seven
scenarios (§7.5). We therefore treat evidence gathered against one as *not* transferring to a claim
measured against the other, and apply that constraint to our own analyses rather than leaving it
implicit; §7.5 quantifies the agreement and the label-coverage bounds, and §8.2 flags where the
distinction bites. Where a statement below holds for either label-producing oracle, we write simply
"the simulated labels".

**Cascade propagation.** The two cascade oracles share the propagation semantics. A subscriber becomes
eligible to fail and propagate only once its average feed loss reaches a `propagation_threshold`
(default $0.2$); below the threshold, partial feed loss is treated as recoverable degradation
rather than a cascade trigger. Broker failure yields continuous per-topic feed loss
$L(t) = |\text{failed\_routers}(t)| / |\text{all\_routers}(t)|$, correctly modeling multi-broker
redundancy. Because intra-wave propagation order is tie-broken stochastically, each scenario is run
over multiple seeds; impact is reported as the across-seed mean with its standard deviation, the
latter itself a fragility signal at cascade boundaries and the noise scale the remediation filter
of §6.4 is calibrated against.

## 5.4 The Shared-Library Blast Mechanism: A Negative Result

Shared libraries have a structurally distinctive failure mode (§3.3, Rule 5): a *simultaneous* blast
rather than a sequential cascade, in which every consuming application fails in one event rather than
along a propagation path. This is invisible to topology-only centrality, which sees an ordinary node
of ordinary degree, and it motivated a specific hypothesis — that a library's composite score $Q(v)$
would understate its true cascade impact, producing a moderate-$Q$/near-total-$I$ mismatch. This
section is measured throughout against $I_{\text{comp}}(v)$ (§5.1).

We tested this directly across all seven synthetic
scenarios (165 Library-type nodes in total) and **did not find the hypothesized mismatch**. The
highest composite score reached by any library in the suite is $Q = 0.422$ (a library with 4
consuming applications), well short of the $Q \approx 0.5$ region the hypothesis anticipated, and its
simulated impact is modest ($I_{\text{comp}} = 0.086$). More importantly, across every library in
the corpus, $I_{\text{comp}}(v)$ never exceeds $Q(v)$: the composite score is, if anything, mildly conservative (over-cautious)
relative to simulated impact for this node type, not blind to a hidden risk. The clearest low-$Q$
case with substantial fan-out — a library with 12 consuming applications — still has
$I_{\text{comp}} = 0.119$ against $Q = 0.255$. Nor does any single-node failure in the suite approach
a near-total impact: the largest composite impact from failing any one component, of any type, across
all seven scenarios, is $I_{\text{comp}} = 0.320$ (an infrastructure node), roughly a third of the
magnitude the blast-radius hypothesis anticipated.

We report this as a negative result rather than omit it. Two readings are consistent with the data.
First, the mechanism itself — simultaneous, type-specific failure via Rule 5 — remains a real
structural distinction worth preserving in the model (§3.3, §4.6), independent of whether it produces
a large low-$Q$/high-$I$ gap in *this* suite; a typed model that can represent the mechanism is not
obligated to find a dramatic instance of it in every corpus. Second, the seven synthetic scenarios
evaluated here may simply under-represent topologies with a genuinely high-fan-out, low-redundancy
shared library — a gap between what a model can express and what a given benchmark suite happens to
exercise. We do not claim to have distinguished between these readings, and we retain the
FanOutReduction operator's blast-radius trigger (§6.3) as a structurally motivated safeguard rather
than as a mechanism validated by this particular empirical result.

## 5.5 Stratified Correlation: A Consistency Check

A single pooled correlation between predicted criticality and simulated impact, computed over all
node types at once, can in principle be misleading if node types occupy sufficiently different regions of the
$(Q, I)$ plane: pooling heterogeneous populations with divergent conditional relationships can produce
a Simpson's-paradox-style near-zero aggregate that conceals strong within-type correlations. We
checked for this directly, against $I_{\text{comp}}(v)$. Pooling $(Q, I_{\text{comp}})$ pairs across
all seven scenarios (1,545 nodes), the
pooled Spearman correlation is $\rho = 0.374$ ($p \approx 2.2\times10^{-52}$). Computed separately by
node type, the correlations are: Broker $\rho = 0.429$ ($n=36$), InfraNode $\rho = 0.409$ ($n=119$),
Library $\rho = 0.351$ ($n=165$), Application $\rho = 0.346$ ($n=850$), Topic $\rho = 0.322$
($n=375$) — all significant at $p < 0.01$.

*(Figure 3: pooled versus per-node-type Spearman $\rho$ between $Q(v)$ and $I_{\text{comp}}(v)$, with
per-type sample sizes.)*

**We do not find a Simpson's-paradox effect in this suite**: the pooled figure (0.374) sits inside
the per-type range (0.322–0.429) rather than diverging sharply from it. This is nonetheless a useful
result, not a null one. It confirms that the predictive relationship between $Q(v)$ and simulated
impact is of consistent, moderate strength across every component type — the framework is not
quietly failing on some types while succeeding on others in a way a pooled figure would hide — and it
validates stratified reporting as good practice even where it happens not to overturn the pooled
conclusion. We report correlation *by node type* throughout (§8) on that basis, rather than because
pooling was shown to be actively misleading here.

**Three scoping conditions on this check.** First, it is computed against $I_{\text{comp}}(v)$, whereas
the predictor tables in §8.1 are computed against $I^*(v)$; the two oracles agree at mean
$\rho = 0.394$ (§7.5), so this consistency check does not transfer to those tables. Second, the check
was worth running on its own terms: the effect it looked for *does* occur elsewhere in this study. In
the predictor evaluation, pooling Application and Library nodes into a single correlation moved HGL
on `av_system` from $\rho = 0.836$ within Applications to $0.46$ pooled — a case where a pooled
figure was actively misleading, and one that went unnoticed until the evaluation contract of §7.3 was
imposed. The methodological point stands independently of the negative finding here.

Third, and cutting across both: the figures in this subsection aggregate components drawn from seven
different systems, while D4 (§4.1) makes $Q(v)$ comparable only *within* a system's own score
distribution. The aggregate is therefore a diagnostic over the union of seven within-system rankings,
not a criticality measurement over a single population, and it is reported here only to answer the
narrow question it was built for — whether the $Q$–$I$ relation holds at similar strength in every
component type. It should not be read as a cross-system criticality result, and no claim elsewhere in
the paper rests on the pooled value.

---

## 7.5 Ground Truth: Three Oracles, and What They Can Each Support

The three oracles introduced in §5.1 are constructed differently, measure different quantities, and
are not interchangeable; conflating them is the most likely way to over-read a result in this paper.
This section fixes which analysis rests on which, quantifies how far they agree, and states the
label-coverage bounds that apply to each.

**Table 16. The three simulation oracles**, what each measures, and which results rest on which.

| Symbol | Engine | Quantity | Used for |
|---|---|---|---|
| $I^*(v)$ | `FaultInjector` | Mean subscriber feed-loss fraction under a BFS cascade | Learned-predictor labels; Tables 18 and 20 (§8.1); the sensitivity sweeps of §8.3 |
| $I_{\text{comp}}(v)$ | `FailureSimulator` | $0.35\,\text{reachability} + 0.25\,\text{fragmentation} + 0.25\,\text{throughput} + 0.15\,\text{flow}$ | Validation gates; the RMAV dimension decomposition; §5.4 and §5.5; remediation acceptance (§6.4) |
| $I_{\text{dyn}}(v)$ | `MessageFlowSimulator` | Delivery-rate loss suffered by *surviving* consumers, by discrete-event simulation of traffic | Reported construct-validity check only — no labels, no gates, no tables |

The two cascade oracles run with a step-function blast-semantics propagation scheme (probability
$1.0$ for library cascade), `propagation_threshold` default $0.2$, a $10$-epoch horizon, and the
five seeds of §7.4, $\{42, 123, 456, 789, 2024\}$. $I_{\text{dyn}}$ shares the seed set and runs
$60$ simulated seconds per component, with the fault injected at the midpoint.

**Measured agreement between the two cascade oracles is weak.** Their scales differ, so only rank
agreement is meaningful; across the seven scenarios, mean Spearman $\rho = 0.394$ and mean top-20%
Jaccard $= 0.286$, ranging from $\rho = 0.578$ (Enterprise) down to $\rho = 0.092$ (Hub-and-Spoke,
where they are effectively uncorrelated). All seven correlations are positive, which is a weak
convergent-validity argument — two differently-constructed simulators do agree directionally, so
neither is purely an artifact of its own construction — but at $\rho \approx 0.39$ it is weak, and we
apply the resulting constraint to our own analyses: a result established against one oracle is not
evidence for a claim measured against the other. §8.2 flags where this bites.

**$I_{\text{dyn}}$ agrees with $I^*$ far more strongly, and — crucially — does not share its worst
case.** Mean Spearman $\rho(I_{\text{dyn}}, I^*) = 0.765$, minimum $0.548$ (Hub-and-Spoke) — against
mean $0.394$, minimum $0.092$ for the two topological oracles above. Hub-and-Spoke is precisely where
$I^*$ and $I_{\text{comp}}$ collapse to near-independence; $I_{\text{dyn}}$ still agrees with $I^*$
there at $\rho = 0.548$, its lowest agreement in the cohort but far from uncorrelated. Because
$I_{\text{dyn}}$ reaches this ranking by simulating traffic through queues rather than by traversing
`DEPENDS_ON`, the result is evidence of a different kind than §7.5's first finding: it rules out the
cascade *algorithm* as the source of $I^*$'s ranking, which the $I_{\text{comp}}$ comparison alone
cannot do (§9.2). Top-$K$ membership is the weaker half of this result — mean top-20% Jaccard is
$0.316$, comparable to the $0.286$ of the two topological oracles — so this is corroboration of
*ranking*, not of critical-set identification; no $F_1@K$ claim in §8.1 is supported by
$I_{\text{dyn}}$.

**Label coverage and the noise ceiling.** Three further properties bound what any reported figure can
mean. First, the cascade model has no rule expressing the failure of a Topic or of a physical Node,
so those types carry no ground truth at all — 30–47% of components per scenario are unlabelled, they
are excluded from scoring rather than scored as zero, and predictions for them are never validated.
Broker labels are degenerate in three of seven scenarios for a related reason. Second, the three
oracles do not cover the same components, so every agreement figure above is computed over the
intersection rather than over the scenario. $I_{\text{dyn}}$ observes only what carries pub-sub
traffic: it scores Applications and those Libraries that publish or subscribe in their own right,
and records Brokers, physical Nodes, Topics, and purely-consumed Libraries as unmeasured rather than
as harmless. On `enterprise_system` that is 349 components against $I^*$'s 360 — it gives up the ten
Brokers and one non-publishing Library, and gains nothing $I^*$ lacks. Third, the labels
have a reproducibility ceiling: across seeds, the ground truth agrees with *itself* at test–retest
$\rho$ of 0.807–1.000, and its own top-20% critical set agrees at Jaccard 0.44–1.00 (deterministic:
`FaultInjector`'s cascade previously iterated an unordered subscriber set while consuming seeded
random draws, so re-running the *same* seed in a different process could still change the label —
this has been fixed and is disclosed as an instrument defect in §9.2, and the figures here are the
post-fix, process-independent ones). No method can exceed the former, and every top-$K$ metric
inherits the latter — a reported $F_1@K$ on `microservices_system`, where the labels' own set
stability is 0.44, should not be read to a precision the labels do not have.
