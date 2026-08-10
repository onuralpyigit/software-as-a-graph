# Remediation and CI/CD gating: full treatment cut from the JSS condensation

> **Provenance.** Verbatim from `docs/research/jss/draft.md` §6 (§6.1–§6.7, in full) and §8.4 (the
> RQ4 results section that reports the gate's runtime and detection efficacy), as they stood at
> commit `f0cba41822820a79ebdab123d54a76072b8f1689`. The condensed JSS `draft.md` collapses §6's
> seven subsections into two (~700 words total, keeping Table 12's four operators and the
> acceptance criterion), and reduces §6.7's 687-word yield analysis plus Table 13 to two sentences
> folded into the condensed §8.4. Nothing here has been reworded.

---

# 6. Prescriptive Remediation and CI/CD Quality Gating

Attribution (§4) and impact analysis (§5) are diagnostic: they tell an architect *which* components
to harden and *why*. This section closes the loop with a prescriptive stage that proposes concrete
architectural edits and verifies that they actually reduce simulated failure impact, before any
deployment. The stage is designed to preserve the same independence discipline as the rest of the
framework: candidate edits are generated from structure alone, and only a separate simulation pass
decides whether to accept them. The section then describes how the diagnostics are operationalised
as a continuous, delta-aware CI/CD quality gate (§6.6).

## 6.1 A Two-Phase Generate–Verify Procedure

Remediation runs in two strictly separated phases.

**Generate.** Given the structural model $G_{\text{analysis}}$ and its attribution, a set of
operators (§6.2) propose candidate topology edits — each a small, concrete modification such as
adding a replica or an alternative route. Generation reads only structure: component types, the
derived `DEPENDS_ON` graph, and structural blast-radius signals. It never reads simulated impact.

**Verify.** Each candidate edit $e$ is applied to produce a counterfactual graph $G' = e(G)$, on
which the `FailureSimulator` of §5.1 is re-run from scratch. The edit is accepted only if it reduces
$I_{\text{comp}}$ by a robust margin (§6.4). This stage is therefore measured against
$I_{\text{comp}}$ throughout, not against the $I^*$ labels behind the predictor tables of §8.1 —
a scoping condition that follows from the weak agreement between the two oracles (§5.1, §7.5).
Verification is an oracle check against ground truth, not against the score that proposed the edit.

This separation matters: a stage that both proposed and scored edits using the same signal would be
optimizing against itself. By generating from structure and verifying by simulation, the stage
cannot manufacture an apparent improvement that the simulator does not confirm.

## 6.2 Remediation Operators

Four operators formalize the framework's existing heuristic recommendations (SPOF redundancy,
alternative routing for bridges, fan-out reduction for over-subscribed topics, decoupling of
multi-topic pairs) into verifiable edits. Each is keyed to a structural trigger and targets a
specific failure mode:

**Table 12. The four remediation operators**, their structural triggers, and the failure mode each targets.

| Operator | Structural trigger | Edit applied | Failure mode targeted |
|----------|--------------------|--------------|-----------------------|
| **RedundancyInsertion** | directed articulation point / high $A$ SPOF | add a redundant instance or redistribute responsibilities | graph-partitioning SPOF |
| **PathDiversification** | bridge edge / single routing path for a topic | add an alternative route (e.g. a second routing broker or network link) | fragmentation on a non-redundant edge |
| **FanOutReduction** | high structural blast radius (topic subscriber fan-out; library consumer count) | interpose an intermediary or split the over-shared channel | simultaneous blast / fan-out explosion |
| **SharedTopicReduction** | high multi-path coupling (large `path_count` / MPCI between a pair) | decouple redundant shared topics between the pair | multi-channel coupling fragility |

The operators span the RMAV dimensions deliberately: RedundancyInsertion and PathDiversification
address Availability, FanOutReduction addresses Reliability (blast radius), and
SharedTopicReduction addresses Maintainability coupling.

## 6.3 Triggering on Blast Radius, not on $Q(v)$

FanOutReduction is the operator that connects remediation to the hypothesis tested in §5.4, and its
trigger is deliberately *not* the composite $Q(v)$. A shared library or an over-subscribed topic
could in principle carry only a moderate $Q$ while nonetheless dominating simultaneous-blast impact;
triggering on $Q$ would then skip exactly the components most worth remediating. Instead,
FanOutReduction fires on direct structural blast-radius signals — subscriber fan-out for topics,
consumer count for libraries — so that a low-$Q$, high-blast component is still selected for a
candidate edit. This is
the remediation-side expression of the paper's central claim that single-score criticality is
insufficient: the *attribution* exposes the gap, and the *operator* is designed not to fall into it.

This is a statement about the trigger's design, not about its yield. §5.4 finds no low-$Q$/high-$I$
library population in this suite for the trigger to catch, and §6.7 shows that its yield is
concentrated in the two topologies that actually contain a fan-out bottleneck. We retain the
structural trigger because triggering on $Q$ would be unsound if such a component existed, not
because we have shown that one does.

## 6.4 Acceptance Criterion

An edit should do more than nudge the mean impact down; it should improve impact by a margin that
exceeds the simulator's own seed noise. For a candidate edit producing $G'$, let
$\Delta I = I_{\text{comp}}(v;G) - I_{\text{comp}}(v;G')$ be the reduction in simulated impact over
the components present in both graphs, and let $\sigma_{\text{seed}}$ be the across-seed standard
deviation of that reduction (§5.1). The acceptance rule is

$$\Delta I > \kappa\,\sigma_{\text{seed}} \quad\text{for every sampled } \texttt{propagation\_threshold},$$

evaluated **per candidate edit, on its own counterfactual graph**, before the edit is committed to
the policy. Two design choices are load-bearing. First, normalising by $\sigma_{\text{seed}}$ ties
the bar to the fragility of the cascade at that point, so an edit is accepted only when its benefit
is distinguishable from propagation-order noise. Second, requiring the inequality to hold across the
full `propagation_threshold` sweep makes acceptance robust to the threshold's value — which §8.3
shows is not a benign parameter, since $\rho$ against ground truth spans 0.230 across its range.

`PrescribeService` implements this as a three-phase procedure: compile the candidate policy (§6.2),
verify each candidate independently by constructing a graph containing that edit alone and
re-simulating it across thresholds and seeds, then apply only the accepted subset and measure the
System Risk Index before and after on the mutated graph as a whole (§6.7's Table 13). Each candidate carries its measured
$\Delta I$, $\sigma_{\text{seed}}$ and — when rejected — the threshold at which it failed, so a run
reports what it declined and why rather than only what it applied.

> **What this replaces.** An earlier version of this framework compiled a policy and applied all of
> it unconditionally, judging the result by a single end-state check. Under that design an edit that
> made the system worse could be carried by edits that made it better, which is the mechanism behind
> the mixed aggregate previously reported in §6.7. Per-edit verification removes that failure mode by
> construction: a regressing edit is rejected individually and never reaches the mutated graph.

An empty accepted set is a valid outcome, not a failure, and is reported as such rather than as a
no-op mutation with an unchanged risk index. On small topologies it is common for no candidate to
clear the bar — which is the filter working, and is more informative than a policy applied on the
strength of an unverified aggregate.

## 6.5 Independence Invariants

The stage obeys three invariants that mirror the predictor/simulator separation of §5.3:

1. **Generate never reads $I_{\text{comp}}(v)$.** Candidate edits come from structure and
   attribution only.
2. **Verify re-invokes the canonical simulator** on $G'$ from scratch, rather than estimating the
   counterfactual impact from the predictor. The re-simulation is performed *per candidate edit*, on
   a graph containing that edit alone, across the propagation-threshold sweep and the seed set; only
   the accepted subset is then applied and re-checked at system level (§6.7).
3. **No Verify result feeds back into Generate within a run.** There is no closed-loop search that
   would let simulated impact influence which edits are proposed, which would reintroduce the
   circularity the framework is built to avoid.

Together these keep the diagnostic and evaluation signals separate: the thing that proposes a fix
and the thing that measures it are never the same signal, so an edit is admitted only on evidence
the proposing signal did not produce.

## 6.6 CI/CD Quality Gate Implementation

To operationalise these diagnostics, SaG integrates into developer workflows as a blocking Quality
Gate in the CI/CD pipeline. When a pull request introduces configuration or architecture
modifications (Architecture-as-Code changes), the pipeline executes the analyzer via a dedicated CLI
script, `detect_antipatterns.py`, which runs the full anti-pattern catalog against the candidate
topology and issues an exit code.

**Exit-code protocol.** The gate issues exit codes that govern pipeline execution:
- **Exit Code 0**: No architectural anomalies found; deployment is permitted.
- **Exit Code 1**: Medium-severity architectural smells found (e.g., chatty pairs or QoS mismatch
  warnings); deployment is permitted with warnings.
- **Exit Code 2**: CRITICAL or HIGH severity anomalies found (e.g., single points of failure, cyclic
  dependencies, or broker overload); the build is broken and **deployment is blocked**.

This is an *absolute* gate: every run evaluates the candidate topology's full finding set, not a
diff against a prior baseline. It has a known consequence, which we do not paper over: a real
architecture that carries an intentional, risk-accepted single point of failure — a sole-source
surveillance feed, a deliberately unreplicated legacy broker — fails the build on every commit,
indistinguishable from a genuine regression. A *delta-aware* gate that evaluates the candidate
against a merge-base topology and blocks only on newly introduced findings, together with a waiver
register recording accepted risk (entity, rule, expiry) so it stays visible rather than silently
re-triggering, would close this gap; we describe the design in §9.3 as future work rather than claim
it here, since the mechanism is not implemented in the released tool.

The underlying analysis-and-detection machinery is in-memory and does not require a live database
connection — `saag`'s thread-safe `MemoryRepository` port satisfies the same repository interface
`detect_antipatterns.py` consumes, and is what the timing harness behind §8.4's measurements uses.
Wiring that path into `detect_antipatterns.py` itself, so the packaged CLI does not require a Neo4j
connection during a CI build, is a small remaining integration step we have not made; today the
script connects to a running database.

## 6.7 What Remediation Yields Under Per-Edit Verification

Running the full Generate→Verify procedure of §6.4 across the scenario suite, with $\kappa = 1.0$,
three propagation thresholds $\{0.1, 0.2, 0.5\}$ and the first three of the five canonical seeds
$\{42, 123, 456\}$, gives the following. The seed set is reduced here, and only here, because
acceptance requires a full re-simulation *per candidate edit per threshold*, so the sweep over 332
candidates is the most expensive experiment in the study. The reduction is a compute concession, not
a methodological one, and we flag the consequence rather than argue it away: $\sigma_{\text{seed}}$ is
the quantity the acceptance rule divides by, and a three-seed estimate of it is noisier than a
five-seed one, which makes the filter correspondingly less reliable at the margin. "Cand." is the
number of edits the generator proposed; "Acc." the number that cleared
$\Delta I > \kappa\,\sigma_{\text{seed}}$ at *every* threshold; $\Delta$SRI is the system risk index
change from applying the accepted subset (positive = risk reduced).

**Table 13. Remediation yield under per-edit verification**, $\kappa = 1.0$, thresholds $\{0.1,0.2,0.5\}$, seeds $\{42,123,456\}$.

| Scenario | Baseline SRI | Mutated SRI | $\Delta$SRI | Cand. | Acc. | Rej. |
|---|---:|---:|---:|---:|---:|---:|
| Autonomous Vehicle | 0.3645 | 0.3615 | +0.0030 | 35 | 3 | 32 |
| IoT Smart City | 0.4260 | 0.4102 | **+0.0158** | 58 | 38 | 20 |
| Financial Trading | 0.3842 | 0.3785 | +0.0057 | 31 | 5 | 26 |
| Healthcare | 0.3809 | 0.3784 | +0.0025 | 19 | 14 | 5 |
| Hub-and-Spoke | 0.3576 | 0.3502 | +0.0074 | 30 | 14 | 16 |
| Microservices Mesh | 0.3612 | 0.3577 | +0.0035 | 40 | 19 | 21 |
| **Hyper-Scale Enterprise** | **0.3614** | **0.3475** | **+0.0139** | **119** | **69** | **50** |

By parallelizing counterfactual verification across multi-core CPU worker pools (`ProcessPoolExecutor`), we evaluate the per-edit acceptance filter across all seven benchmark scenarios, including Hyper-Scale Enterprise (350+ components, 119 candidate edits). Across the full 7-scenario suite, 162 of 332 candidate edits (48.8%) clear the multi-threshold acceptance filter ($\Delta I > \kappa\,\sigma_{\text{seed}}$). On Hyper-Scale Enterprise, 69 of 119 candidates clear the filter, reducing System Risk Index from 0.3614 to 0.3475 ($\Delta\text{SRI} = +0.0139$).

**Per-edit verification prevents regressing edits.** Every admitted edit is guaranteed to reduce
cascade impact individually across all propagation thresholds. This removes by construction the
failure mode of the previous unverified design, in which a regressing edit could be carried by an
improving one — an aggregate that was arithmetically correct and substantively misleading. Under the
per-edit filter no scenario in the suite regresses.

**Individually-verified edits are not shown to compose.** Acceptance is decided on singletons: each
candidate is simulated alone, on a graph containing only that edit. Nothing in the procedure
establishes that a set of individually-accepted edits remains beneficial when applied together, and
the $\Delta$SRI column reports the outcome of applying the accepted subset rather than a verified
prediction of it. Verifying *subsets* rather than singletons would close this at combinatorial cost;
we note it as a limitation of the current design rather than claiming compositional safety we have
not tested.

**The acceptance rate varies widely with topology, and that is the substantive finding.** The filter
admits 3 of 35 candidates on Autonomous Vehicle and 5 of 31 on Financial Trading, but 38 of 58 on
IoT Smart City and 14 of 19 on Healthcare. The two largest absolute risk reductions come from IoT
Smart City ($\Delta$SRI $= +0.0158$) and Hyper-Scale Enterprise ($+0.0139$) — the two scenarios with
pronounced hub-topic and fan-out structure. That the operator set has purchase precisely where a
fan-out bottleneck exists is consistent with how the operators are defined (§6.2), and suggests the
honest scope for this stage is narrower than "topology-level hardening": it is closer to "fan-out
decomposition where a fan-out bottleneck actually exists". Across the suite the improvements are
real but small in absolute terms — between $+0.0025$ and $+0.0158$ SRI — which is the result we
report rather than a demonstration that the prescriptive stage is yet practically valuable (§9.3).

---

# 8.4 (full, pre-condensation) RQ4 — Feasibility and Performance of SaG as a CI/CD Quality Gate

A primary blocker for continuous Static System Analysis (SSA) is execution time: developers will
bypass or disable quality gates that introduce significant build delays. We evaluate the feasibility
of deploying SaG as a blocking gate by measuring the wall-clock cost of the structural analysis and
anti-pattern catalog — the mechanism `detect_antipatterns.py` invokes — run via the in-memory
`MemoryRepository`, across all eleven scenarios in our corpus (mean over the five canonical seeds).

The measured footprint:
- **≤ 90 components** (`tiny_system` and all three real-world transcribed architectures): $0.02$–$0.04$ s.
- **98–326 components** (the remaining six generated scenarios): $0.27$–$1.24$ s. Cost does not scale
  monotonically with component count in this range — `hub_and_spoke_system` (139 components, $1.24$ s)
  costs more than `iot_smart_city_system` (326 components, $1.08$ s) — consistent with the catalog's
  cost being driven by specific detectors' complexity (e.g. `DEEP_PIPELINE`'s path enumeration) more
  than by raw component count.
- **`enterprise_system`**, the largest scenario at 520 components: $26.74 \pm 0.32$ s.

All eleven scenarios complete in well under the several-minute budget continuous build pipelines
typically allow.

In terms of gating efficacy: the gate is currently absolute rather than delta-aware (§6.6), so there
is no merge-base diff to evaluate detection against; what we can and do measure is the anti-pattern
catalog's raw agreement with the cascade oracle — the property the gate is a proxy for. Scoring
CRITICAL/HIGH findings (with edge-keyed findings crediting both endpoints) against the oracle's own
critical set gives, across five seeds: precision $0.237 \pm 0.014$, recall $0.887 \pm 0.059$, F1
$0.374 \pm 0.022$, Cohen's $\kappa = -0.036 \pm 0.049$ on the seven generated scenarios ($n=40$
scenario–seed pairs), and precision $0.402 \pm 0.059$, recall $0.861 \pm 0.105$, F1 $0.544 \pm
0.061$, $\kappa = 0.296 \pm 0.100$ on the three real-world architectures ($n=15$). Recall is high and
precision is not: the catalog over-flags relative to the oracle's critical set, and near-zero $\kappa$
on the generated corpus means the agreement it does show is close to what chance flagging at the
catalog's own base rate would produce. We read this as a genuine limitation of the pattern catalog as
a *stand-alone* predictor of simulated impact — a finding pattern catalog and rank predictor serve
different purposes (naming a structural smell versus ranking by predicted impact, §9.1), but the gap
here is wide enough that the catalog's findings should not be read as impact-calibrated. The composite
$Q(v)$ of §4 remains the ranking signal validated against the oracle in §8.1.

From a **sustainability and resource efficiency** standpoint, evaluating architectural risks statically
in-memory ($0.02\text{ s}$–$26.74\text{ s}$ across our corpus) yields energy savings relative to
spinning up staging clusters or running heavier dynamic checks per build, though we have not measured
that comparison directly.
