# Why a predictor rather than the oracle: the case for the Analyse and Predict stages

> **Provenance.** **New material — not a condensation lift.** Unlike its sibling files in this
> folder, this chapter section has no upstream in `jss_draft_full.md`: the argument it makes does
> not appear anywhere in the JSS, AUSE, or Middleware drafts. It was written to answer a question
> the corpus raises and never resolves — [`docs/research/jss/outline.md`](../../jss/outline.md)
> records it as a live reviewer risk ("a reviewer may ask why not standardize on one … should be
> stated crisply if challenged") — and to supply the upstream text that
> [`docs/criticality.md` §7.2](../../../criticality.md) and the JSS introduction condense from.
> Every figure cited here is traced to a committed artifact; none required a new experimental run.

---

## The question, in its strongest form

If `I*(v)` — the mean subscriber feed-loss fraction under a breadth-first cascade, produced by
`FaultInjector` — is what this work *means* by component criticality, then the critical set is
already computable. Run the simulator over every component, sort, take the top-K. Why introduce a
structural analysis stage that scores components by proxy, and a graph learning stage that is
trained to approximate the very oracle we could have queried directly?

The question deserves the concession before the defence, because the concession is total.

## The concession

**Simulation alone is sufficient to identify critical components, given a complete and correctly
parameterised model of the system and enough compute to sweep it.** Nothing in this thesis
contradicts that, and the framework is built so that it cannot: `I*(v)` is the definition of the
target, so the simulator that produces it attains ρ = 1.0 against it by construction. No predictor
evaluated in this work can exceed its own oracle on the oracle's own terms, and none is claimed to.

Two consequences follow, and both are stated here rather than deferred to a threats section.

First, **every correlation this thesis reports is surrogate fidelity, not accuracy against
reality.** When HGL-QoS attains ρ = 0.6676 under leave-one-system-out evaluation
([`results/table4_loso_results.md`](../../../../results/table4_loso_results.md)), the claim
supported is that the learned model reproduces the simulator's ordering on a system it never saw —
not that either the model or the simulator reproduces the ordering a production outage would
reveal. That second link is unmeasured, and is marked as such in the three-link validation chain of
[`docs/criticality.md` §7.1](../../../criticality.md).

Second, **the interpretable attribution score is not a competitive ranker across systems.** Under
the same LOSO protocol, `RM / Q(v)` attains ρ = **−0.0142** — indistinguishable from, and
nominally worse than, random ordering. This thesis therefore does not position `Q(v)` as a
prioritisation instrument. Its role is attribution, and §4 below argues that attribution is a
distinct and non-substitutable function rather than a weaker form of ranking.

To this must be added the sharpest self-criticism the corpus already contains, which the
input–label independence guarantee does *not* escape: predictors read `G_analysis` while oracles
execute on `G_structural`, but both views are deterministic functions of the same input topology.
What the guarantee rules out is feature–label feedback. It does not rule out a modelling assumption
shared by both views, which would be invisible to every oracle in the framework. This is view
independence, not independence of data source.

Having conceded all of that, the defence is not that the surrogate is more accurate than the
oracle. It is that the oracle is **expensive in the one place it is most needed, brittle in a way
that makes "just simulate" under-specified, and incomplete over roughly a third of every system** —
and that the analysis stage performs a function simulation does not perform at all.

---

## 1. Expensive — but only where it actually matters

The naive form of this argument is false and should not be made. A trained model does not deliver
an order-of-magnitude end-to-end speedup over simulation, because the model's inputs are not free.
From [`results/inference_latency.json`](../../../../results/inference_latency.json), at |V| = 2000
the GNN forward pass takes 254 ms, but the structural analysis that produces its 18–25-dimensional
features takes **56.72 s** — a ratio of 223×, with Analyse, not inference, as the bottleneck. One
exhaustive simulation sweep at comparable scale is of the same order. A claim of cheap prediction
that quietly omits the feature-extraction cost would not survive scrutiny, and is not made here.

Where the cost asymmetry is real is **counterfactual search**. Diagnosis asks one question of the
graph as it stands; remediation asks a question of every graph that a candidate edit could produce.
Because acceptance must not be an artifact of a single propagation threshold or a single seed, each
candidate edit is verified over a (threshold × seed) grid: the corrected sweep across all seven
scenarios covered 1589 candidate edits and **14,301 exhaustive simulation sweeps, at approximately
2 h 47 min of measured wall-clock time on twenty cores**. Against this, the entire LOSO training
sweep — every fold, every seed — costs roughly 31 minutes for HGL and 36 for HGL-QoS. The training
cost is paid once and amortised over every subsequent query; the verification cost is paid per
candidate, per threshold, per seed, and recurs on every architectural revision.

This is why the framework is structured as **propose–verify** rather than as search-by-simulation.
The learned model and the attribution score are cheap enough to score a large candidate space;
`FailureSimulator` is the expensive, authoritative check applied only to the survivors. Simulation
is not displaced from the loop — it is moved to the position where its cost buys the most, which is
adjudication rather than enumeration. The measured yield vindicates this: 1128 of 1589 candidates
(71.0%) survive verification, meaning roughly 29% of proposals would have been accepted in error by
an unverified recommender.

The combinatorial point sharpens this further. The simulator, as implemented, evaluates **one
component failure per run**. Pairwise evaluation exists (`simulate_pairwise`) at n(n−1)/2 cost and
is not exposed through the CLI; no k ≥ 3 enumeration exists anywhere. But the object this thesis is
ultimately about — the critical *set* — is not the set of individually critical components. Two
components each individually survivable may be jointly catastrophic if they are the only two paths
to a subscriber. Exhaustive simulation cannot reach that regime at any realistic system scale,
whereas a model that has learned typed structural regularities across many systems can at least
score candidate sets. This thesis does not claim to have solved the k-subset problem; it observes
that simulation forecloses it by cost, which is a reason for a cheap approximator to exist.

## 2. Brittle — the oracle is a parameter choice, not a fact

"Just simulate" presumes a simulator. This framework has three, and they do not agree.

From [`results/convergent_validity.json`](../../../../results/convergent_validity.json), measured
across seven scenarios and five seeds:

| Oracle pair | Mean Spearman ρ | Min ρ | Mean top-K Jaccard |
|:---|---:|---:|---:|
| `I_dyn` vs `I*` | 0.7647 | 0.5497 | 0.3117 |
| `I_comp` vs `I_dyn` | 0.4646 | 0.1208 | 0.3133 |
| `I_comp` vs `I*` | 0.3970 | 0.0974 | 0.2617 |

Two readings matter. The first is that oracle agreement spans from strong to negligible depending
on which pair is chosen, bottoming out at ρ = 0.097 on `hub_and_spoke_system`. A result established
against one oracle is therefore not evidence for a claim measured against another. The second is
less comfortable and is not currently stated anywhere in the corpus: **even the best-agreeing pair
shares only 31% of its top-K set.** The oracles corroborate each other's *ordering* moderately
well, and corroborate each other's *critical set* barely at all — and the critical set, not the
ordering, is the artefact an architect acts on.

The parameterisation compounds this. `propagation_threshold = 0.2` is a free parameter *of the
ground truth* rather than of any model, which is why the threshold-sensitivity harness exists at
all: a result that only holds at one setting of it is a result about the setting. Seed variance is
likewise non-trivial — from
[`results/label_stability.json`](../../../../results/label_stability.json), test–retest ρ ranges
0.807–1.0 and top-K Jaccard falls to 0.44, with a per-node standard deviation reaching **0.416 on a
[0, 1]-bounded label** in `financial_trading_system`. The exhaustive sweep runs `n_trials = 1`: a
single draw, with no variance estimate attached.

So the instruction "just simulate" is under-specified. It does not say *which* simulator, at *which*
propagation threshold, under *which* seed — and the answer materially changes the critical set. A
model fitted across systems, seeds, and scenarios is, in this light, not merely a cheaper copy of
one simulator run. It is a variance-reduced and threshold-marginalised estimate of what the
oracles agree on, which is a different and arguably better-posed estimand than any single sweep
provides. This is the strongest available defence of the Predict stage, and it is an argument about
estimator quality rather than about speed.

## 3. Incomplete — simulation cannot label a third of the system

The cascade model derives its propagation paths from publish/subscribe and `USES` relations. It
therefore has no way to express the direct failure of a `Topic` or of a physical `Node`: those
types are not merely hard to simulate, they score a degenerate 0.0 everywhere, and both the
injector and the CLI exclude them deliberately on the grounds that a type scoring zero everywhere
is not a finding about the system but an artefact of the cascade having no path to express it.
Training on such a block teaches a model to predict a constant.

The consequence is that **30–47% of components per scenario carry no ground truth at all** — a
figure corroborated by the live artifact in `output/simulation/impact_scores.json`, where 38 of 101
components (37.6%) appear in `unlabeled_node_ids` rather than in `records`. Simulation, as the sole
instrument, would return no criticality verdict for more than a third of the architecture. The
framework's response is to carry an explicit `dimension_mask` so that "measured as harmless" is
never conflated with "never measured", and to score the unlabelled remainder through the structural
and learned paths instead.

The completeness gap is not only about node types. `RuntimeTelemetryProfile` is the only channel by
which observed runtime data — message rates, per-edge failure correlation, per-application
starvation bounds — can enter `FailureSimulator` at all, and it is never constructed anywhere in
the pipeline outside its own unit test. In every reported result, message rates therefore default
to 1.0 uniformly. Separately, unless baseline flows are primed from `EventSimulator`, the
`flow_disruption` term is hard-zero and 0.15 of the composite silently vanishes — a path taken by
`run_failure_simulation_exhaustive` but not by the Monte-Carlo or pairwise entry points. The oracle
is thus not a window onto the running system; it is a model of the running system, whose fidelity
is bounded by inputs the pre-deployment setting does not supply.

## 4. Why the Analyse stage is separately non-substitutable

The preceding three arguments defend Predict against Simulate. They do not, on their own, defend
Analyse — and Analyse is the more expensive of the two stages, so it needs its own justification.
It has three, of which the second is load-bearing.

**(a) It manufactures the model's input.** The GNN consumes 18 shared topological dimensions plus
type-specific extensions, all produced by structural analysis. Without Analyse there is no Predict.
This is a dependency, not an argument for the design, but it explains why the two stages cannot be
costed separately.

**(b) Attribution selects the repair operator, and no scalar impact score can.** This is the
decisive point. Simulation returns a magnitude: *component X, impact 0.72*. It does not return a
reason, and a reason is what determines the remediation. The policy compiler
([`saag/prescription/rules.py`](../../../../saag/prescription/rules.py)) dispatches on
*dimension-level* structure, not on aggregate score: an articulation point with no broker
redundancy is a `NodeReallocation`; a topic concentrating too many subscribers is a `TopicSplit`; a
mismatch between declared QoS and structural exposure is a `QosUpgrade`. These are different
repairs with different costs, and a single number — however accurate — cannot discriminate among
them. The `CriticalityProfile` taxonomy (SPOF, Bottleneck, Total Hub, Fragile Hub) exists precisely
to make that dispatch possible. This is why `Q(v)`'s LOSO ranking failure (ρ = −0.0142) is
survivable rather than fatal: the score is not being asked to rank, it is being asked to explain,
and the explanation feeds a discrete operator choice that is then verified by simulation.

**(c) It is the falsifiability instrument.** The independence guarantee requires a predictor
computed on `G_analysis` and an oracle executed on `G_structural`. Without a structurally derived
score, the validation stage would be comparing simulation against itself, and the framework's
central empirical claim would be a tautology. The Analyse stage is what makes the Validate stage a
test rather than a restatement — subject to the shared-topology caveat conceded above.

---

## The defensible claim, stated once

Simulation can identify critical components, and in this framework it defines what they are. The
Analyse and Predict stages are not present because simulation fails at that task. They are present
because:

1. the oracle's cost is prohibitive precisely where architectural decisions are made — over the
   space of candidate repairs, not over a single fixed graph — so a cheap proposer and an expensive
   verifier outperform search-by-simulation at equal fidelity;
2. the oracle is one draw from a family of disagreeing, parameter-sensitive simulators, so a model
   fitted across that family estimates a better-posed quantity than any single sweep;
3. the oracle is silent on 30–47% of every system by construction; and
4. attribution answers a question — *which repair* — that impact magnitude does not address at all.

What is *not* claimed: that the learned model is more accurate than its oracle, that either is
validated against production outages, or that view independence amounts to independent evidence.
