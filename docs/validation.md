# Step 5: Validate

**Statistically prove that topology-based predictions agree with simulation-derived cascade impact.**

← [Step 4: Simulate](failure-simulation.md) | → [Step 6: Prescribe](prescription.md)

For the full CLI flag reference (`validate_graph.py`), see [cli-pipeline-guide.md — Step 5](cli-pipeline-guide.md#step-5-validate). This document specifies *what* is measured and *why* those measurements are trustworthy.

---

## Table of Contents

1. [What This Step Does](#1-what-this-step-does)
2. [Two Entry Points, Two Gate Systems](#2-two-entry-points-two-gate-systems)
3. [Ground Truth: I(v)](#3-ground-truth-iv)
   - 3.1 [Notation — three quantities, three symbols](#31-notation--three-quantities-three-symbols)
   - 3.2 [Measured agreement between the oracles](#32-measured-agreement-between-the-oracles)
   - 3.3 [The behavioural oracle I_dyn(v)](#33-the-behavioural-oracle-i_textdynv)
   - 3.4 [Simulation mechanics](#34-simulation-mechanics)
   - 3.5 [Ground truth derivation](#35-ground-truth-derivation)
4. [Prediction: Q(v)](#4-prediction-qv)
5. [Statistical Battery](#5-statistical-battery)
   - 5.1 [The one-population evaluation contract](#51-the-one-population-evaluation-contract)
   - 5.2 [Rank correlation and the label noise ceiling](#52-rank-correlation-and-the-label-noise-ceiling)
   - 5.3 [Bootstrap confidence intervals](#53-bootstrap-confidence-intervals)
   - 5.4 [Classification metrics at K](#54-classification-metrics-at-k)
   - 5.5 [Per-dimension validation](#55-per-dimension-validation)
   - 5.6 [Composite validation and predictive gain](#56-composite-validation-and-predictive-gain)
   - 5.7 [Wilcoxon signed-rank test](#57-wilcoxon-signed-rank-test)
   - 5.8 [System health metrics](#58-system-health-metrics)
6. [Gate Systems](#6-gate-systems)
   - 6.1 [Library gates G1–G6, G8](#61-library-gates-g1g6-g8)
   - 6.2 [CLI topology-class gates](#62-cli-topology-class-gates)
7. [Stratified Reporting](#7-stratified-reporting)
8. [Methodological Guards](#8-methodological-guards)
   - 8.1 [The guard harness](#81-the-guard-harness)
   - 8.2 [Multi-seed stability sweep](#82-multi-seed-stability-sweep)
   - 8.3 [QoS ablation](#83-qos-ablation)
9. [Output Schema](#9-output-schema)
10. [Interpreting Results](#10-interpreting-results)
11. [Known Limitations](#11-known-limitations)
12. [What Comes Next](#12-what-comes-next)

---

## 1. What This Step Does

Step 5 closes the methodological loop. It aligns two independently derived signals for every
component in the system:

| Signal | Source | What it represents |
|--------|--------|-------------------|
| **Q(v)** | RM formula over structural metrics (Predict stage, Step 3) | *Predicted* criticality — computed deterministically from graph structure alone, before any runtime data |
| **Q_gnn(v)** | GNN prediction (Predict stage, Step 3, optional) | *Refined prediction* — inductive GNN node scores, compared against I(v) in addition to or instead of Q(v) |
| **I(v)** | Stochastic cascade simulation (Simulate stage, Step 4) | *Proxy ground truth* — normalised damage score obtained by injecting each node as the failure origin |

High statistical agreement between Q(v) and I(v) is empirical evidence that **topology alone predicts
failure impact** — the central claim of the Software-as-a-Graph thesis.

```
 Graph (Step 1) ──▶ Step 2: Analyze ──▶ Step 3: Predict          Step 4: Simulate
                    M(v) metrics         Q(v) = w_R·R + w_M·M      I(v) = mean impact
                                          R = α·FT + (1−α)·A        over n_repeats seeds
                                                    │                      │
                                                    └──────────┬───────────┘
                                                               │
                                                     Statistical Battery
                                                     Spearman ρ, Kendall τ
                                                     Bootstrap 95% CI
                                                     F1@K, SPOF-F1, FTR, PG
                                                     Wilcoxon vs. degree baseline
                                                               │
                                                       Gate Evaluation
                                                               │
                                                      PASS / FAIL verdict
```

A compound test is used because no single metric is sufficient:

- **ρ** confirms the global rank ordering is preserved.
- **F1@K** confirms the top-K critical components are correctly identified.
- **PG** confirms the composite Q(v) outperforms its own best single dimension.
- **SPOF-F1** confirms structural SPOFs are correctly caught.

> [!NOTE]
> **Scope.** This document covers the Validate stage only. Formula definitions for R/M/A/V live in
> [structural-analysis.md](structural-analysis.md); the cascade engines live in
> [failure-simulation.md](failure-simulation.md).

---

## 2. Two Entry Points, Two Gate Systems

Validation has two independent implementations. They answer the same question with different
oracles and different thresholds, and **results must name which one produced them**.

| | Library path | CLI path |
|---|---|---|
| Invoked by | `saag --validate`, `saag.Client.validate()`, `POST /api/v1/validation/run-pipeline` | `saag-validate` (`cli/validate_graph.py`) |
| Implementation | [saag/validation/](../saag/validation/) | [cli/validation/](../cli/validation/) |
| Ground truth | `FailureSimulator` → $I_{\text{comp}}(v)$ + two dimensions | `FaultInjector` → $I^*(v)$ |
| Gates | 7 gates (G1–G6, G8; G7/G9 retired), fixed thresholds ([§6.1](#61-library-gates-g1g6-g8)) | 5 gates, topology-class adaptive ([§6.2](#62-cli-topology-class-gates)) |
| Output | `PipelineResult` → `LayerValidationResult` per layer | `ValidationResult` / `SweepReport` JSON |
| Scope | Per-layer (`app`, `infra`, `mw`, `system`), both RM dimensions + FT/A sub-characteristic diagnostics | Whole graph, composite only, multi-seed |

**Which to use.** The library path is the one the pipeline and API run, and the only one that
produces per-dimension validation. The CLI path is the research harness: multi-seed sweeps,
topology-class gates, QoS ablation, and LaTeX export.

**Library path call flow:**

```
ValidationService.validate_layers(layers)
    ├── analysis.analyze_layers(layers)          # DEPENDS_ON derived once for the whole run
    └── per layer: validate_single_layer(layer)
            ├── prediction.predict_quality(...)                 → Q(v), R/M (+ FT/A sub-characteristic diagnostics)
            ├── simulation.run_failure_simulation_exhaustive()  → I_comp(v) + IR (= α·IFT+(1−α)·IA)/IM
            └── validate_single_layer_from_results(...)
                    ├── Validator.validate(...)         → overall ρ, F1, top-K, RMSE
                    ├── per dimension (DIMENSION_SPECS) → ρ + specialist metrics
                    ├── composite I*(v)                 → ρ(Q*, I*), predictive gain
                    ├── gates G1–G6, G8
                    └── stratified reporting
```

**CLI path call flow:**

```
load_graph(system.json)
    ├── compute_rm(G, qos=...)  or  compute_gnn_scores(G, checkpoint)   → Q(v)
    └── derive_ground_truth(G, n_repeats=5)                              → I*(v)
            └── run_statistical_tests()  → ρ, τ, CI, F1@K, SPOF-F1, ICR, BCE, PG, Wilcoxon
                    ├── stratified_metrics(by node type)
                    ├── classify_topology(G)  → sparse | medium | dense | hub_spoke
                    └── evaluate_gates(vr, topo_class)
```

> [!IMPORTANT]
> **Independence guarantee.** Q(v) uses only graph structure (PageRank, betweenness, degree,
> articulation points) and optionally QoS contract attributes. The composite I(v), IR(v) and IA(v)
> are produced by simulations over $G_{\text{structural}}$ (raw pub-sub edges) with no access to
> Q(v). Measuring ρ(Q\*, I\*), ρ(R, IR) and ρ(A, IA) is therefore a genuine empirical test.
>
> **IM(v) is an internal consistency check, not an independent test.** It derives from the
> same `DEPENDS_ON` graph as M(v): `ChangePropagationSimulator` traverses $G^T$ with an
> `instability`-based stop condition shared with M(v)'s CouplingRisk. Alignment on a shared
> substrate is still useful signal, but it cannot claim the same methodological independence.
> (An earlier revision of this framework had a second such check, IS(v) against S(v)'s QADS via
> `CompromisePropagationSimulator` — that simulator was deleted along with the Vulnerability/Security
> dimension, not retained as an unused consistency check.)

---

## 3. Ground Truth: I(v)

### 3.1 Notation — three quantities, three symbols

Three different things have been written `I*` across this documentation set. They are not
interchangeable and each result must name the one it used.

| Symbol | Engine | Definition | Backs |
|---|---|---|---|
| **I\*(v)** | `FaultInjector` | Mean subscriber feed-loss fraction | GNN training labels; the main table, LOSO and k-fold tables; the CLI gates |
| **I_comp(v)** | `FailureSimulator` | `0.35·reachability + 0.25·fragmentation + 0.25·throughput + 0.15·flow_disruption` | The library gates ([§6.1](#61-library-gates-g1g6-g8)) and the IR/IM decomposition |
| **I_RM(v)** | `FailureSimulator` | `0.5·(IR + IM)` — equal-weighted dimension sum, where `IR(v) = α·IFT(v) + (1−α)·IA(v)`, α=0.36 | Predictive Gain only ([§5.6](#56-composite-validation-and-predictive-gain)) |

**Dimension coverage.** $I^*(v)$ is a single scalar. It maps onto the `composite` and `reliability`
label columns (`LABEL_COLS = {composite, reliability, maintainability}`, 3-wide); `maintainability` is
the sole **unmeasured** dimension for that engine, declared absent via the artifact's
`labeled_dimensions`, not filled with zeros. (Availability is not a separate label column at all — it
is folded into `reliability` via the $\alpha$-blend before labelling, so $I^*(v)$'s reliability column
already reflects both Fault Tolerance and Availability ground truth without needing its own column.)
Only $I_{\text{comp}}(v)$'s engine supplies both RM dimensions, decomposed further into
`IFT`/`IA`/`IM`. See [failure-simulation.md §6.1](failure-simulation.md#61-impact_scoresjson).

**Why maintainability is the one gap, and why a better simulator would not change it.** The dimension
$I^*(v)$ does cover is denominated in an *externally observable* quality attribute — Reliability,
hierarchical over fault tolerance and availability — which is what a fault injector watching service
delivery can see ([criticality.md §3.5](criticality.md#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)).
Maintainability is not observable that way at all: **it is assessed on the artifact, not in
execution** — no amount of running a system reveals what changing it would cost. This is why the
$I_M$ ground truth in $I_{\text{comp}}$'s engine is a change-propagation BFS over $G^\top$ rather than
a behavioural observation — it is a *structural model*, not a behavioural one. Per-dimension
agreement on `maintainability` should therefore be read as an internal-consistency check between two
structural computations, not as behavioural validation
([criticality.md §7.1](criticality.md#71-the-validation-chain-has-three-links)). (An earlier revision
of this framework had a second, symmetric gap here — security presumes an adversary rather than a
fault, so fault injection was the wrong instrument for it too — but Vulnerability/Security has since
been retired outright rather than kept as a second unmeasured dimension.)

Mixing the two oracles within one stage is a correctness error, guarded by
[`tests/test_groundtruth_contract.py`](../tests/test_groundtruth_contract.py).

### 3.2 Measured agreement between the oracles

The oracles are on different scales, so only their rank agreement is meaningful. Measured across
the seven-scenario cohort
([results/convergent_validity.json](../results/convergent_validity.json), produced by
[reproduce/convergent_validity.py](../reproduce/convergent_validity.py)), for
$I^*(v)$ against $I_{\text{comp}}(v)$:

| Scenario | Spearman ρ | Kendall τ | top-20 % Jaccard | n |
|---|---:|---:|---:|---:|
| enterprise_system | 0.578 | 0.453 | 0.393 | 310 |
| av_system | 0.521 | 0.397 | 0.308 | 84 |
| healthcare_system | 0.424 | 0.312 | 0.294 | 53 |
| financial_trading_system | 0.405 | 0.318 | 0.300 | 65 |
| microservices_system | 0.382 | 0.284 | 0.226 | 96 |
| iot_smart_city_system | 0.359 | 0.274 | 0.262 | 206 |
| hub_and_spoke_system | 0.092 | 0.093 | 0.217 | 72 |
| **mean** | **0.394** | — | **0.286** | — |

> [!WARNING]
> **The two oracles agree only moderately.** Mean ρ = 0.394 and mean top-K Jaccard = 0.286; on
> `hub_and_spoke_system` they are effectively uncorrelated (ρ = 0.092, τ = 0.093). This is a
> **construct-validity bound**: an argument established against one oracle does not transfer to a
> claim measured against the other. Any result that moves between them — for example a stratified
> analysis run on $I_{\text{comp}}$ used to interpret a table computed on $I^*$ — must either be
> re-run on the matching oracle or state the gap explicitly.
>
> Read positively, this is the honest form of the convergent-validity argument: two
> differently-constructed simulators do agree *directionally* (all seven ρ are positive), which is
> evidence that neither is purely an artifact of its own construction.

**QoS is not the source of the disagreement.** Both oracles are QoS-weighted, raising the question
of whether the shared `w(t)` inflates their agreement. Measured with
`reproduce/convergent_validity.py --no-qos` versus the default:

| | mean Spearman ρ | mean top-K Jaccard |
|---|---:|---:|
| QoS disabled in both oracles | 0.4214 | 0.2815 |
| QoS enabled in both oracles | 0.4050 | 0.2831 |

Rank agreement did not improve; it fell slightly. The disagreement is structural — between a mean
subscriber feed-loss and a weighted composite of four connectivity terms.

> [!NOTE]
> This ablation pair predates the 2026-08-04 regeneration that produced the table above and has not
> been re-run against it. The default-QoS row (0.4050) is close to the current default-QoS mean
> (0.394) but was not recomputed from the same run; treat this sub-table as indicative pending a
> `--no-qos` re-run rather than as re-verified.

### 3.3 The behavioural oracle $I_{\text{dyn}}(v)$

$I^*$ and $I_{\text{comp}}$ are both topological cascade engines over the same substrate, so §3.2
measures agreement between two constructions that share their assumptions. It cannot answer the
sharper objection: that a topology-derived $Q(v)$ is being validated against topology-derived
labels.

$I_{\text{dyn}}(v)$ is the third oracle, produced by
[`MessageFlowSimulator`](../saag/simulation/message_flow_simulator.py) and reported by the same
[reproduce/convergent_validity.py](../reproduce/convergent_validity.py). It is the drop in
delivered message rate that **surviving** consumers experience when $v$ fails, measured by
discrete-event simulation of actual traffic rather than by reachability over edges:

$$I_{\text{dyn}}(v) = \text{delivery\_rate}_{\text{before}} - \text{delivery\_rate}_{\text{after}}$$

Both windows exclude the faulted node's own receipts and its share of the fan-out, and a silenced
publisher's demand stays in the denominator. Measured across the same cohort at `duration=60`:

| Scenario | ρ($I_{\text{dyn}}$, $I^*$) | n | ρ($I^*$, $I_{\text{comp}}$) from §3.2 |
|---|---:|---:|---:|
| healthcare_system | 0.938 | 61 | 0.424 |
| iot_smart_city_system | 0.875 | 210 | 0.359 |
| financial_trading_system | 0.780 | 78 | 0.405 |
| av_system | 0.779 | 100 | 0.521 |
| enterprise_system | 0.774 | 349 | 0.578 |
| microservices_system | 0.662 | 118 | 0.382 |
| hub_and_spoke_system | 0.548 | 95 | 0.092 |
| **mean** | **0.765** | — | **0.394** |
| **min** | **0.548** | — | **0.092** |

Two things matter here, and the second matters more than the first.

**The mean is higher** — 0.765 against 0.394. **The minimum is far higher** — 0.548 against 0.092.
$I^*$ and $I_{\text{comp}}$ collapse to near-independence on exactly one scenario,
`hub_and_spoke_system` (§3.2); $I_{\text{dyn}}$ still agrees with $I^*$ there at 0.548 — its lowest
agreement in the cohort, but nowhere near uncorrelated. That is what makes §3.2's warning readable:
where the two topological engines disagree most, simulated traffic still sides with $I^*$, so the
disagreement is a property of how $I_{\text{comp}}$ is constructed rather than a genuine ambiguity in
which components are critical.

**What this is and is not evidence of.** $I_{\text{dyn}}$ and $I^*$ are close in *construct* —
both are ultimately about subscriber feed loss — so their agreement is convergent validity across
**methods**, not across constructs. It rules out the cascade *algorithm* (BFS over derived
`DEPENDS_ON`) being the artifact, since discrete-event simulation of real traffic reproduces its
ranking. It does not independently corroborate the prior assumption that criticality *means*
feed loss. Claiming more than that would overstate it.

**Top-K membership is the weaker half.** Mean top-20% Jaccard is 0.316 for
$I_{\text{dyn}}$/$I^*$ — close to the 0.286 of $I^*$/$I_{\text{comp}}$, not meaningfully better.
Rank correlation is strong while the identity of the top-K set still churns, which is consistent
with the ~40% top-K churn across label seeds noted in §3.6 of
[failure-simulation.md](failure-simulation.md). Read the ρ, not the set overlap, as the result.

> [!IMPORTANT]
> $I_{\text{dyn}}$ is **delivery-based and QoS-agnostic**. It does not measure latency or deadline
> conformance: topic QoS never resolves on this corpus (L13) and latency carries no signal at
> corpus load (L14), both in
> [failure-simulation.md §12](failure-simulation.md#12-known-limitations). It is also **not** a
> validation gate and does **not** produce training labels — it is a reported construct-validity
> check, and it scores only components carrying pub/sub traffic, with Brokers and Nodes recorded
> in `unlabeled_node_ids` rather than scored 0.0.
>
> The accounting this rests on is load-bearing and was wrong until recently: counting the faulted
> node's own lost feed as damage, and letting a silenced publisher's messages leave the numerator
> and denominator together, together put ρ($I_{\text{dyn}}$, $I^*$) at **−0.25** on `atm_system`
> (publisher population, outside this cohort), driven by a +0.75 correlation with the faulted
> node's own subscription count.
> [tests/test_message_flow_oracle.py](../tests/test_message_flow_oracle.py) pins both properties.

### 3.4 Simulation mechanics

For each node v, the `FaultInjector` runs a BFS cascade in two sequential phases per wave.

**Phase A — direct propagation along `DEPENDS_ON` / `USES` edges.**

| Edge kind | Probability | Effect |
|---|---|---|
| `app_to_app` | `prob · depth_damp`, `prob = 0.0` by default | Disabled unless explicitly enabled |
| `app_to_lib` (or the failed node is a `Library`) | `1.0` | Deterministic: library failure fails **every** consuming Application at wave 0, which then orphans the topics they solely publish |

> [!IMPORTANT]
> **Phase A qualifies the independence claim for Library nodes only.** Q(v)'s Reliability dimension
> includes a normalised in-degree term counting `app_to_lib` edges, and a Library's I(v) is driven
> through those same edges by Phase A. Both quantities increase with the number of consuming
> Applications. This is **not** label leakage — I(v) remains a simulation output, not a function of
> the feature vector — but ρ(Q, I) restricted to Library nodes is a partially-coupled measurement
> rather than a clean empirical test. Report `per_type_rho` so Library and Application correlations
> can be read separately. The claim holds unchanged for Application and Broker nodes.

**Phase B — topic-mediated, QoS/rate-weighted propagation.**

1. **Continuous topic feed loss.** For each topic $t$:
   - With publishers: $L(t) = \dfrac{\sum_{p \in \text{failed}} \text{rate\_hz}(p, t)}{\sum_{p \in \text{all}} \text{rate\_hz}(p, t)}$, falling back to the fraction of failed publishers when the total rate is 0.
   - With no publishers but broker routers: $L(t) = \dfrac{|\text{failed routers}(t)|}{|\text{all routers}(t)|}$.
   - Scaled by QoS criticality and capped: $L(t) = \min(1.0,\ L(t) \times \text{QoS\_factor}(t))$, where the factor combines reliability (`RELIABLE` ×1.2) and priority (`HIGH`/`CRITICAL`/`URGENT` ×1.15, `MEDIUM` ×1.05).
2. **Orphan tracking.** If $L(t) > 10^{-6}$ the topic is marked orphaned and its not-yet-failed subscribers are marked impacted.
3. **Stochastic subscriber failure.** For subscriber $s$ with mean feed loss $\text{sub\_loss}(s)$ over its subscribed topics, if $\text{sub\_loss}(s) \ge 0.2$ (the propagation threshold) it fails with probability $\min\left(1.0, \frac{\text{sub\_loss}(s)}{0.2}\right) \times \text{depth\_damp}$, where $\text{depth\_damp} = \max(0.25,\ 1.0 - \text{wave\_idx} \times 0.15)$ prevents runaway cascades.

### 3.5 Ground truth derivation

```python
rng_seeds = [seed + i * 37 for i in range(n_repeats)]   # default n_repeats = 5

for each node v:
    impacts, depths, affected = [], [], []
    for s in rng_seeds:
        impact, depth, n_affected = simulate_cascade(G, v, depth_limit, seed=s)
        impacts.append(impact); depths.append(depth); affected.append(n_affected)
    I(v)            = mean(impacts)
    cascade_depth   = max(depths)        # worst case observed
    nodes_affected  = mean(affected)
```

Averaging across `n_repeats` seeds dampens stochastic variance and yields a stable mean impact
estimate. This is the value compared against Q(v) in every subsequent test.

---

## 4. Prediction: Q(v)

Q(v) is produced by the Predict stage (Step 3), not by Validate. It is a DECLARED (not AHP-weighted)
combination of two characteristic scores, where Reliability is itself a hierarchical blend of two
sub-characteristics:

```
R(v) = 0.36 × FT(v)  +  0.64 × A(v)
Q(v) = 0.80 × R(v)   +  0.20 × M(v)
```

| Term | Weight | Rationale |
|-----------|:------:|-----------|
| Reliability (R) | **0.80** | Re-parameterisation of the retired 4-D composite's (A+R) share; see [structural-analysis.md §11.2](structural-analysis.md#composite-score-qv) |
| Maintainability (M) | **0.17 → 0.20** | Coupling complexity; long-term fragility (share rose slightly on renormalisation after Vulnerability/Security was dropped) |
| ↳ Fault Tolerance (FT), within R | **0.36** | Cascade propagation reach |
| ↳ Availability (A), within R | **0.64** | SPOF severity — still the largest single share of any sub-term, now expressed within Reliability rather than as a peer dimension |

`w_R=0.80`, `w_M=0.20`, and `α=0.36` are DECLARED constants, not an AHP composite output — see
[structural-analysis.md §11.2](structural-analysis.md#composite-score-qv) for the re-parameterisation
derivation from the retired 4-D vector. The complete formula reference for each dimension — including
the Topic-specific Fault Tolerance variant and every sub-term — is in
[structural-analysis.md](structural-analysis.md). It is not duplicated here.

**Topology-only vs. QoS-enriched modes:**

| Mode | `--qos` flag | PSPOF contribution |
|------|:-----------:|--------------------|
| Topology-only baseline | off (default) | `PSPOF = 0` for all nodes |
| QoS-enriched | on | `PSPOF` computed from pub-sub topology |

The ablation study ([§8.3](#83-qos-ablation)) measures the predictive lift from the enriched mode.

---

## 5. Statistical Battery

All CLI statistics are computed on **Application-type nodes** by default, falling back to all nodes
only when fewer than 4 Application nodes exist. This matches the thesis claim: topology predicts
*application-layer* cascade criticality, not generic structural centrality of topics and brokers.

### 5.1 The one-population evaluation contract

Every predictor variant in every reported table is scored by a single function,
[`saag.evaluation.metrics.compute_inductive_metrics`](../saag/evaluation/metrics.py), on a single
node set resolved by `resolve_eval_keys`. This exists because the previously published Table 3 did
not have it, and the resulting comparison was invalid:

| | Structural baselines (Topo-BL / Topo-QoS) | Learned variants (GL / HGL family) |
|---|---|---|
| Node types scored | Applications (DEPENDS_ON projection) | Applications **and** Libraries, pooled into one ρ |
| Sample | **every** node | the **20 % test split** only |

Two estimators measured on two different samples is not a comparison. The effect was large enough to
invert the paper's RQ1 conclusion: pooling Applications with Libraries dragged HGL on `av_system`
from ρ = 0.81 within-type down to 0.46 — the Simpson pattern this document warns about in
[§8.1](#81-the-guard-harness) — while the baselines were unaffected because their key set contained
too few Libraries to pool.

Three rules now hold across Tables 3, 4 and 5:

1. **The key set is a function of the graph and labels only** — never of any variant's predictions —
   so all variants in a cell see an identical sample. `--eval-population` (`application` by default,
   `app_lib`, `labeled`) records the choice in `main_table.json["config"]`.
2. **The reported figure is the held-out one.** All variants share one train/val/test split pinned by
   node id via `resolve_cell_split` and applied through
   [`apply_external_splits`](../saag/prediction/data_preparation.py). A full-population score
   flatters a trained model by including the nodes it was fitted on while leaving a training-free
   baseline unchanged. The transductive figure is retained alongside, under `full_population`.
3. **A variant that cannot cover the population fails loudly.** Scoring returns
   `error: "incomplete_coverage"` rather than silently shrinking the sample.

> **Absent is not zero.** A stratum whose predictions or labels are constant has an *undefined* rank
> correlation. It is reported as `"undefined"` with a `reason` (`constant_signal` /
> `too_few_nodes`), never as `0.0`, and `aggregate_per_type` preserves that through seed and fold
> averaging. Reporting 0.0 made Topic, Node and Library appear as measured failures when they carry
> no ground truth at all — a coverage gap masquerading as a model result.

### 5.2 Rank correlation and the label noise ceiling

**Spearman ρ** is the primary gate metric: it measures whether the rank ordering of components by
Q(v) matches the ordering by I(v). Both implementations delegate to `scipy.stats.spearmanr`;
constant input yields ρ = 0.0, p = 1.0 by convention rather than NaN.

**Kendall τ** is the conservative cross-check. A large |ρ − τ| gap (> 0.15) indicates agreement
driven by a few extreme outliers — inspect the top 2–3 CRITICAL components.

The interpretation of an absolute ρ depends on the **ground-truth regime**:

**Regime A — RM pipeline against simulation labels.** Achievable ρ is bounded by simulator noise
and topology decoupling. The criterion is G1: pass if ρ ≥ 0.70.

| ρ Range | Interpretation |
|---------|---------------|
| ≥ 0.85  | Very strong — well above G1 |
| 0.70–0.85 | Acceptable — G1 passes |
| 0.60–0.70 | Borderline — G1 fails; check topology class and node-type filter |
| < 0.60  | Weak — investigation required |

**Regime B — learned/GNN models against simulation labels.** Absolute ρ is constrained by simulator
noise independent of model quality, so the meaningful criterion is **lift over the structural
baseline**, Δρ = ρ(model) − ρ(Topo-BL).

| Δρ vs. Topo-BL | Interpretation |
|---------|---------------|
| ≥ +0.15 | Substantial lift |
| +0.05 to +0.15 | Meaningful lift |
| −0.05 to +0.05 | No clear improvement |
| < −0.05 | Regression — model underperforms the structural baseline |

> **Why two regimes?** Absolute thresholds were calibrated when validation targets shared structural
> basis with predictors (ρ ≈ 0.94 against reachability proxies). Against honest Sim labels the same
> absolute values are unattainable regardless of model quality. Applying Regime A thresholds to
> Regime B condemns results for the wrong reason; applying Regime B bands to RM pipeline results
> masks absolute weakness.

**The label noise ceiling.** Both regimes appeal to "simulator noise". That bound is measured, not
asserted, and travels with the labels in the artifact's `label_stability` block
([failure-simulation.md §3.6](failure-simulation.md#36-multi-seed-stability-label-noise-and-reproducibility)).

`test_retest_spearman` is the worst pairwise agreement between two seeds' label vectors — how well
the ground truth reproduces *itself*. **No method can exceed it.** A model reporting ρ = 0.93
against labels self-consistent at 0.93 has saturated the labels; treating that as a shortfall
against a 0.95 target is a category error. Measured across the cohort it ranges 0.928
(`microservices_system`) to 1.000 (`iot_smart_city_system`), and `cli/loso_evaluate.py` prints the
worst value in `summary.md` next to the achieved ρ.

**Rank stability and set stability differ sharply.** `topk_jaccard` — the worst pairwise overlap of
the top-20 % critical sets across seeds — falls to **0.56** on `microservices_system` and 0.625 on
`atm_system`, meaning roughly 40 % of the "critical set" changes between seeds of the *same*
labeler. Rank-based metrics (ρ, NDCG) are largely immune; every top-K-cut metric (Overlap@K,
Precision@τ, SPOF-F1) inherits that churn directly. Read `topk_jaccard` before quoting any of them.

> [!CAUTION]
> A single-seed cache cannot establish a ceiling. `label_stability.test_retest_spearman` is `null`
> in that case with an explanatory `note`, and any ρ computed against it is unbounded above by
> construction. Regenerate with the five recommended seeds before publication.

### 5.3 Bootstrap confidence intervals

Non-parametric percentile bootstrap (CLI: B = 2000, seed 42; library: B = 1000, seed 42):

```
for b in 1..B:
    idx = sample indices with replacement from [0..n-1]
    θ_b = statistic(x[idx], y[idx])

CI_95 = [percentile(θ_b, 2.5), percentile(θ_b, 97.5)]
```

A CI that does not cross the gate threshold is stronger evidence than a point estimate alone. When
variance is zero the CI degenerates to `[0, 0]` and a warning is emitted. The library applies the
same resample-sort-percentile core to ρ, F1 and top-5 overlap.

### 5.4 Classification metrics at K

`K` defaults to **20 % of total node count** (minimum 3, maximum n). Override with `--top-k`.

```
gt_top_k   = top K nodes by I(v)   (ground-truth critical set)
pred_top_k = top K nodes by Q(v)   (predicted critical set)
```

> [!IMPORTANT]
> **At equal K, Precision@K, Recall@K and F1@K are one number, not three.** Because
> `|gt_top_k| = |pred_top_k| = K`, it follows that `FP = FN`, so all three are identically
> `|gt_top_k ∩ pred_top_k| / K`, and FTR is its complement. Reporting them as separate evidence
> overstates how much has been measured, and a gate on F1@K plus a gate on Precision@K
> ([§6.1](#61-library-gates-g1g6-g8), G2 and G3) test the same quantity twice.
>
> The code names this honestly: `cli.validation.statistics.top_k_agreement` computes it once and
> the three legacy field names are populated from it. `cli/loso_evaluate.py` emits `overlap_at_k`.

To obtain precision and recall that genuinely diverge, size the truth set from the data rather than
fixing it at K. The evaluation CLI reports a **τ-threshold** critical set alongside the top-K window:

```
true_critical = {v : I(v) ≥ τ}          τ = 0.5 · max I(v)
Precision@τ   = |pred_top_k ∩ true_critical| / K
Recall@τ      = |pred_top_k ∩ true_critical| / |true_critical|
```

The threshold is relative to the maximum because label magnitude is **not comparable across
scenarios**: $I^*(v)$ is a mean over *all* subscribers, so it decays roughly as
$1/|\text{subscribers}|$ — max I(v) ranges from 0.223 on `iot_smart_city_system` to 0.960 on
`healthcare_system`, a ~4.3× spread. An absolute constant would select nearly every node in one
scenario and none in another.

**PR-AUC** (average precision over the full ranking against the τ set) is the preferred single
summary for cross-scenario comparison — it needs no K and no prediction-side threshold.

**A caution on small critical sets.** The τ set typically contains only 2–9 nodes; a single ranking
error moves Recall@τ by 30–50 points. Read `n_true_critical` alongside the value.

**RMSE and MAE against raw labels are not interpretable.** They compare sigmoid-scale predictions
against labels whose maximum varies ~4× across scenarios. Use `rmse_scaled` / `mae_scaled`, which
min-max both vectors first, and read `label_scale_max` for context.

**Coverage must be reported with every metric.** `n_predicted`, `n_labeled` and `n_evaluated` state
how many nodes the model scored, how many carry ground truth, and how many the metric was computed
over. Nodes without labels are dropped from the inner join — they are evidence neither for nor
against the model. Currently 30–47 % of components per scenario are unlabelled (Topic and Node; see
[failure-simulation.md §12](failure-simulation.md#12-known-limitations)).

**SPOF-F1** measures articulation-point detection as an availability indicator:

```
SPOF-actual    = {v : is_articulation_point(v)  AND  I(v) > 0.3}
SPOF-predicted = {v : is_articulation_point(v)}
SPOF-F1        = harmonic mean of SPOF-precision and SPOF-recall
```

A low SPOF-F1 with a high overall ρ means the global ordering is correct but the binary SPOF
threshold is misaligned with the simulation threshold.

### 5.5 Per-dimension validation

Rather than comparing every dimension against one global cascade score, each predictor is
correlated against its own simulation-derived ground truth. The two composite predictors (R, M)
follow an identical procedure — align keys, require n ≥ 3, compute ρ, then compute dimension-specific
specialist metrics — so they are declared as data in
[`saag/validation/dimensions.py`](../saag/validation/dimensions.py)'s `DIMENSION_SPECS`. Fault
Tolerance and Availability are Reliability's sub-characteristics, reported as diagnostics via a
separate `SUBCHARACTERISTIC_SPECS` — excluded from the composite gates (they already feed R via the
α-blend; including them too would double-count) but still individually correlated and worth reading.

| Dimension | Predictor | Ground truth | Specialist metrics |
|---|---|---|---|
| **Reliability** | R(v) | IR(v) = α·IFT(v)+(1−α)·IA(v) — blended cascade+partition impact | **CCR@5** capture rate; **CME** mean rank distance, normalised by system size |
| ↳ **Fault Tolerance** *(diagnostic)* | FT(v) | IFT(v) — cascade propagation potential | Same family, reported not gated |
| ↳ **Availability** *(diagnostic)* | A(v) | IA(v) — partitioning effect | **SPOF-F1** (+ precision/recall); **HSRR** hidden-SPOF recovery; **DASA** directional agreement; **RRI** redundancy robustness |
| **Maintainability** *(consistency check)* | M(v) | IM(v) — coupling fragility | **COCR@5** capture rate; **κ_CTA** weighted-κ over 3 coupling tiers; **BP** bottleneck precision |

CCR@5 and COCR@5 are **the same statistic under two names** — the top-K overlap between predictor and
ground truth over the components both score. They are computed by one function,
`calculate_capture_rate_at_k`, and differ only in which dimension they are applied to and what
target they are held to. (An earlier revision of this framework had a third name, AHCR@5, for the
Security dimension — retired along with it.)

> [!NOTE]
> **ρ(FT, A) is reported as a diagnostic, not an orthogonality gate.** An earlier revision of this
> framework computed **CDCC**, correlating two *predictors* (Security against Availability) as a
> cross-dimension contamination check gated by G7 — both S(v) and G7 are retired. ρ(FT, A)'s reading
> is the *opposite* of CDCC's: since `R = α·FT + (1−α)·A`, a **low** ρ(FT, A) is the interesting case
> — it means the two sub-characteristics genuinely disagree, so blending them is doing real work
> rather than being redundant. It is excluded from `max_interdim_correlation` for that reason, not
> gated at all.

> [!WARNING]
> **HSRR, DASA and RRI are not currently measured.** They read the structural metrics
> `qspof`, `ap_c_out`, `ap_c_in` and `bridge_score`, which `StructuralMetrics` does not yet
> populate, so they evaluate against all-zero predictors. Read them as "not yet measured" rather
> than as failing scores; they are listed in `dimensions.UNPOPULATED_STRUCTURAL_METRICS`. SPOF-F1
> is unaffected — it uses `is_articulation_point`, which is populated.

### 5.6 Composite validation and predictive gain

The composite ground truth is the equal-weighted sum of the two dimensional ground truths — **equal,
not (0.80, 0.20)**, deliberately: using the scoring weights to build the ground-truth composite would
make ρ(Q*, I*) partly circular.

$$I_{\text{RM}}(v) = 0.5 \cdot IR(v) + 0.5 \cdot IM(v)$$

Note this is $I_{\text{RM}}$ in the notation of [§3.1](#31-notation--three-quantities-three-symbols),
*not* $I^*(v)$ — the two dimensions are scaled onto a common range before summation, which is why
they are scaled once, together, in `_extract_ground_truths`. (An earlier revision of this framework
summed four dimensions at 0.25 each; Fault Tolerance and Availability are not separate terms here —
`IR(v)` already carries the α-blend of both, per [§3.1](#31-notation--three-quantities-three-symbols).)

**Predictive Gain** measures whether combining dimensions beats the best single one:

$$PG = \rho(Q(v), I_{\text{RM}}(v)) - \max\big(\rho(R, IR),\ \rho(M, IM)\big)$$

PG > 0.03 (gate G5) is the evidence that multi-dimensional integration adds genuine predictive
value rather than reproducing its strongest component. PG now maxes over 2 candidates rather than 4,
which raises PG mechanically relative to the pre-migration model — the 0.03 threshold was calibrated
against a 4-candidate max and has not been re-derived for the new 2-candidate one; read PG's absolute
value with that in mind rather than assuming it is directly comparable across the migration.

**Orthogonality.** The single composite predictor pair (R×M) is correlated. Any |ρ| above
`max_interdim_correlation` (0.40) logs an orthogonality violation: two dimensions that rank
components identically are not measuring distinct quality attributes. (ρ(FT, A) is also reported,
per [§5.5](#55-per-dimension-validation), but excluded from this gate for the reason stated there.)

### 5.7 Wilcoxon signed-rank test

Tests whether Q(v) ranks nodes *better* than a degree-centrality baseline against I(v):

```
diff_scores = |Q(v) − I(v)| − |DC(v) − I(v)|     for all v
```

One-sided (`alternative='less'`), α = 0.05. Significance means Q(v)'s absolute errors are
statistically smaller than the baseline's. Requires at least 10 nodes; otherwise p = 1.0.

### 5.8 System health metrics

Component-level predictions are aggregated into system-wide indicators, weighted by component QoS
weight w(v):

| Metric | Formula | Meaning |
|---|---|---|
| **H_d** (per dimension) | $1 - \dfrac{\sum_v \text{score}_d(v) \cdot w(v)}{\sum_v w(v)}$ | Health in dimension d ∈ {R, M, FT, A}; 1.0 is perfect. H_FT and H_A are reported alongside for diagnostic visibility but excluded from SRI (below) — they already feed H_R via the α-blend, so summing them too would double-count |
| **SRI** | $w_R \cdot (1 - H_R) + w_M \cdot (1 - H_M)$, $w_R{=}0.5, w_M{=}0.5$ | System Risk Index — overall structural vulnerability. Sums only H_R and H_M — the two composite dimensions |
| **RCI** | $\dfrac{\sum_i (2i - n - 1) \cdot Q_{(i)}}{n \sum_i Q_{(i)}}$ | Risk Concentration Index — Gini coefficient of Q(v); high means risk sits in few components |

---

## 6. Gate Systems

The two entry points ([§2](#2-two-entry-points-two-gate-systems)) apply different gates. Neither
subsumes the other; a report must say which it used.

### 6.1 Library gates G1–G6, G8

Fixed thresholds from [`ValidationTargets`](../saag/validation/models.py), evaluated per layer.
**Only Tier 1 determines `passed`**; Tiers 2 and 3 are reported.

**G7 and G9 are retired**, along with the Vulnerability/Security dimension both were built to gate
(G7 = CDCC, cross-dimensional contamination between Security and Availability; G9 = FTR, Security
false top rate — see [§5.5](#55-per-dimension-validation)). The gap in the numbering is intentional;
do not renumber the survivors or reuse G7/G9 for a future gate.

| Gate | Metric | Threshold | Description |
|---|---|:---:|---|
| **Tier 1 — primary** (all must pass for `passed = True`) | | | |
| G1 | Spearman ρ | ≥ 0.70 | Global rank correlation |
| G2 | F1@K | ≥ 0.75 | Top-K critical set classification |
| G3 | Precision@K | ≥ 0.80 | Top-K critical set precision |
| G4 | Top-5 overlap | ≥ 0.60 | Overlap of top 5 predicted vs. actual |
| **Tier 2 — secondary** | | | |
| G5 | Predictive Gain | > 0.03 | Lift of composite ρ over the best single dimension (not yet re-derived for the 2-candidate max — see [§5.6](#56-composite-validation-and-predictive-gain)) |
| G6 | κ_CTA | ≥ 0.70 | Weighted-κ coupling tier agreement |
| **Tier 3 — specialist** | | | |
| G8 | Bottleneck Precision | ≥ 0.70 | Maintainability bottleneck detection |

> [!NOTE]
> **G2 and G3 cannot disagree.** As [§5.4](#54-classification-metrics-at-k) shows, F1@K and
> Precision@K are the same number at equal K, so G3 (≥ 0.80) strictly dominates G2 (≥ 0.75):
> whenever G3 passes, G2 passes. Two of the four Tier-1 gates therefore carry one gate's worth of
> evidence. This is recorded rather than silently fixed because changing it would move every
> published verdict.

Two further gates are evaluated inside `Validator._validate_group` and reported alongside:
`G5_rmse` (RMSE ≤ 0.25) and `p_value_pass` (ρ's p ≤ 0.05).

### 6.2 CLI topology-class gates

The CLI adapts its thresholds to graph shape — a sparse 12-node system faces less stringent
requirements than a dense hub-spoke architecture.

```python
density   = edges / (nodes × (nodes − 1))
hub_ratio = max_degree / mean_degree

"hub_spoke" if hub_ratio > 10 and density < 0.10
"sparse"    if density < 0.05
"dense"     if density > 0.20
"medium"    otherwise
```

| Class | ρ ≥ | F1 ≥ | SPOF-F1 ≥ | FTR ≤ | PG ≥ |
|-------|:---:|:----:|:---------:|:-----:|:----:|
| `sparse` | 0.75 | 0.65 | 0.60 | 0.30 | 0.02 |
| `medium` | 0.80 | 0.70 | 0.65 | 0.25 | 0.03 |
| `dense` | 0.82 | 0.72 | 0.65 | 0.25 | 0.03 |
| `hub_spoke` | 0.85 | 0.75 | 0.70 | 0.20 | 0.03 |

All five must pass for `overall_pass = True`. Exit code is 0 on PASS and 1 on FAIL, for CI use.

Note that the CLI's ρ thresholds (0.75–0.85) are all **stricter** than the library's G1 (0.70), and
that its `f1` and `ftr` gates are complements of one another ([§5.4](#54-classification-metrics-at-k)).

---

## 7. Stratified Reporting

Pooled correlations can hide a predictor that works on one population and fails on another. Both
paths therefore report stratified results.

**By node type.** ρ (and F1@K in the CLI) computed independently per type, each against its own
target:

| Node type | Library target ρ | Role |
|---|:---:|---|
| Application | 0.75 | Primary validation layer |
| Broker | 0.70 | Secondary, broker-layer |
| Node (infra) | 0.65 | Infrastructure; smaller population |
| Library | 0.60 | Coupling layer; fewer nodes → noisier ρ |
| *(other)* | 0.70 | Default |

Strata with fewer than 4 nodes report `"too few nodes for ρ"`; strata with constant I(v)
(std < 1e-9) report `"constant signal (not a primary failure type)"`. Typical output:

```
  Application       n=  26  ρ= 0.8320  F1=0.7143
  Broker            n=   5  constant signal (not a primary failure type)
  Topic             n=  27  constant signal (not a primary failure type)
  InfraNode         n=   8  constant signal (not a primary failure type)
  Library           n=   8  constant signal (not a primary failure type)
```

Topics and Brokers are *expected* to show constant signal: the cascade triggers from a source node's
failure, not from a topic. Topic-layer reliability is captured through pub-sub orphaning (Phase B),
but the score accrues to the publishing application, not the topic node.

**By topic frequency decile.** Distributed systems have topic message rates spanning orders of
magnitude, so aggregate metrics are exposed to Simpson's paradox. All `Topic` components are sorted
by raw frequency (`frequency` or `topic_frequency`, Hz) and partitioned into ten equal-count bands;
each band with ≥ 3 topics reports ρ, its p-value, and its concrete frequency bounds. This shows
which communication bandwidths the prediction actually works on.

---

## 8. Methodological Guards

### 8.1 The guard harness

`saag-validate harness` operates on **pre-computed** Q(v) and I(v) JSON files rather than deriving
them from a graph. It wraps five guards that are orthogonal to the statistical battery:

| Guard | What it checks |
|-------|---------------|
| **Stratified ρ (Simpson's paradox guard)** | Computes ρ and τ *per node type* and flags the pooled correlation as potentially misleading. Pools mixing node types with different (Q, I) regimes can yield near-zero global ρ even when every per-type ρ is strongly positive — observed on the ATM dataset at pooled ρ ≈ 0.075 against per-type ρ 0.63–0.90. |
| **Convergent validity** | Cross-correlates multiple `--ground-truth` sources. Strong inter-oracle agreement is the convergent-validity argument; weak agreement limits what either oracle can claim. Requires at least two sources. |
| **Independence ledger** | Each source declares whether it shares structural basis with Q(v) (`:qos` tag). A coupled source emits an ablation caveat rather than silently printing a number. |
| **Rank-displacement outliers** | Surfaces nodes where I(v) ranks a component far more critical than Q(v) — structural blind spots, the library blast-radius gap being canonical. |
| **Multi-seed spread** | With `per_seed` scores, reports mean ± std of pooled ρ across seeds. `std > 0` is labelled cascade-order fragility, not hidden. |

Use `single` / `sweep` / `report` / `compare` to compute Q(v) and I(v) from scratch; use `harness`
to validate artifacts that already exist, to bring in a second independent oracle (e.g. Δlatency
`I_dyn(v)`), or to run the journal-submission checklist.

### 8.2 Multi-seed stability sweep

A single seed is not sufficient evidence. `sweep` and `report` run the full pipeline across seeds
(default `42, 123, 456, 789, 2024`):

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| `rho_mean` | mean(ρ across seeds) | Average predictive power |
| `rho_std` | std(ρ across seeds) | Stability; target σ ≤ 0.05 |
| `rho_min` / `rho_max` | range | Worst and best seed |
| `f1_mean` | mean(F1@K) | Average classification quality |
| `pg_mean` | mean(PG) | Average predictive gain |
| `rcr` | 1 − mean(normalised Kendall distance between seed pairs) | Rank Consistency Rate |
| `all_gates_pass_rate` | fraction of seeds passing all gates | Reliability of the PASS verdict |

RCR = 1.0 means identical rankings across all seeds; target ≥ 0.90 for a stable methodology.

### 8.3 QoS ablation

`compare` runs two full sweeps back to back — topology-only and QoS-enriched — and reports the
pairwise deltas. **Primary claim:** Δρ = ρ(Q_QoS, I) − ρ(Q_topo, I) > 0, *p* < 0.05, via a paired
t-test on the seed-level ρ series (for fewer than 3 seeds, significance is approximated as
Δρ > 0.01). `compare` exits 0 only when Δρ > 0 *and* the test is significant.

> [!WARNING]
> **This ablation varies the predictor while holding the label fixed, so on its own it cannot
> establish that the lift is real.** QoS already propagates into the predictor: `w(t)` inherits onto
> every edge and from there into QSPOF, the QoS-weighted in/out degrees, and QoS-weighted
> betweenness. Once QoS also enters the *label*, the two share a term and Δρ rises whether or not
> anything was learned.
>
> [`reproduce/qos_label_ablation.py`](../reproduce/qos_label_ablation.py) varies the **label**
> instead — computing I(v) under `--qos-factor none` / `ladder` / `wt` and scoring each predictor
> against all three. The diagnostic is the *spread of Δρ across predictors*, not any single Δρ:
> roughly uniform means the enriched label is simply a better target; concentrated on `topo_qos`
> means construct overlap. Measured across the cohort, `topo_qos` gains relative to the QoS-blind
> `topo_bl` control in 9 of 10 scenarios, by a small but systematic margin (|Δρ| < 0.05 on all real
> scenarios). Report both numbers together.

**QoS-stratified reporting.** Two fields in [`saag/evaluation/metrics.py`](../saag/evaluation/metrics.py)
answer "does the model rank the components carrying critical channels correctly?":

| Field | Meaning |
|---|---|
| `per_qos_tier_rho` | ρ stratified by the QoS tier of the mass a component carries (`Σ w(t)` over topics it publishes or routes). Guards the Simpson risk on the axis the QoS claim is about. A stratum below `_MIN_STRATUM = 3` or with a constant vector is `undefined`, never `0.0`. |
| `critical_topic_coverage_at_k` | Share of the system's total QoS mass covered by the top-K predicted components, plus `lift` against what K random components would cover. This is what a hardening budget actually asks; a global ρ cannot express it, since a model can rank well overall and still miss the few components carrying critical topics. |

Subscriptions are deliberately excluded from the exposure sum: losing a subscriber does not silence
the channel for anyone else.

---

## 9. Output Schema

### Library path — `PipelineResult`

```
timestamp, all_passed, total_components, layers_passed, targets, warnings
layers: { <layer>: LayerValidationResult }
```

Each `LayerValidationResult` serialises to:

| Field | Type | Meaning |
|---|---|---|
| `layer`, `layer_name` | str | Layer identity |
| `passed` | bool | G1 ∧ G2 ∧ G3 ∧ G4 |
| `summary.spearman` / `f1_score` / `top_5_overlap` / `rmse` | float | Overall metrics |
| `summary.{reliability,maintainability,fault_tolerance,availability}_spearman` | float | Per-dimension ρ (fault_tolerance/availability are Reliability sub-characteristic diagnostics) |
| `summary.composite_spearman`, `summary.predictive_gain` | float | ρ(Q\*, I_RM) and PG |
| `summary.system_health` | dict | `H_R`, `H_M`, `H_FT`, `H_A`, `SRI`, `RCI` |
| `validation_result` | object | Full `ValidationResult`: `overall` and `by_type` groups, each with `correlation` / `error` / `classification` / `ranking` metric blocks |
| `gates` | dict | `G1_spearman` … `G8_bottleneck_precision` (no `G7`/`G9` — retired), plus `G5_rmse` and `p_value_pass` |
| `node_type_stratified` | dict | Per type: `n`, `spearman`, `target_rho`, `passed` |
| `frequency_decile_stratified` | dict | Per decile: `n`, `frequency_range`, `spearman`, `p_value` |
| `dimensional_validation` | dict | Per dimension: `spearman`, `spearman_p`, specialist metrics, `n`, `ground_truth` label; plus a `composite` entry with `interdim_rhos` and `system_health` |
| `dimensional_scatter` | dict | Per dimension: `(id, predicted, actual, level)` tuples for plotting |
| `confidence_intervals` | dict | Per dimension: `(ci_lower, ci_upper)` on ρ |
| `rule_based_baseline_metrics` / `gnn_forecasting_metrics` | dict | `spearman`, `macro_f1`, `ndcg_10`, `passed`, `targets` (GNN adds `bce_loss` and `regression_curve`) |

The API wraps this through [`api/presenters/validation_presenter.py`](../api/presenters/validation_presenter.py),
which adds a `data` block (`predicted_components`, `simulated_components`, `matched_components`) and
`precision`/`recall`/`passed` to `summary`.

### CLI path — `ValidationResult` / `SweepReport`

`single` writes `{"validation": <ValidationResult>, "topology_class": str}`; `report` writes
`{"sweep": <SweepReport>, "topology_class": str, "gate_thresholds": [...]}`.

| Block | Fields |
|---|---|
| Rank correlation | `spearman_rho`, `spearman_p`, `kendall_tau`, `kendall_p`, `bootstrap_ci_lo`, `bootstrap_ci_hi` |
| Classification | `top_k`, `precision_at_k`, `recall_at_k`, `f1_at_k`, `spof_f1`, `ftr` — note the first three are equal by construction ([§5.4](#54-classification-metrics-at-k)) |
| Specialist | `icr_at_k`, `bce`, `pg` |
| Wilcoxon | `wilcoxon_stat`, `wilcoxon_p`, `wilcoxon_significant` |
| Strata / gates | `strata`, `gates_passed`, `overall_pass` |

`SweepReport` adds `rho_mean/std/min/max`, `f1_mean`, `pg_mean`, `rcr`, `all_gates_pass_rate` and
the per-seed `ValidationResult` list.

### Harness report

`pooled.rho` is always reported but flagged as Simpson-contaminated; the `per_type` entries are the
primary evidence. A `Corr` entry with `n < 3` has `NaN` values and is not printed.

### CSV output (`--csv`)

One row per node, ranked by Q(v) descending:

```
rank, node_id, node_type, Q, R, M, A, S, I, cascade_depth, nodes_affected, is_articulation_point, degree_centrality
1, ConflictDetector, Application, 0.8421, 0.7200, 0.6500, 0.9100, 0.4200, 0.8102, 4, 18, True, 0.0321
```

---

## 10. Interpreting Results

| Symptom | Likely cause | Action |
|---|---|---|
| ρ high (≥ gate) but F1@K fails | The global ordering is right but the binary "critical" threshold is misaligned; if IQR(I) is small the top-K boundary is unstable | Inspect the Q(v) and I(v) distributions; consider a larger `--top-k`, or check the gate is appropriate for this system's size |
| **ρ is negative** | **Inverse criticality** — in mission-critical systems the most central nodes are the most hardened, so high-PageRank components have redundant publishers and backup routes and their failure impact is *lower* | Expected for ATM/HFT designs. Otherwise: raise `--cascade` (10–15) so cascades propagate through well-connected hubs, and enable `--qos` so sole-publisher nodes are penalised |
| PG ≤ 0 | The composite adds nothing over its best single dimension | With < 10 Application nodes the test has high variance. Check `ap_c_directed` and `mpci` are populated in `StructuralMetrics` (they default to 0 if Step 2 did not run). Use `--verbose` to find systematic mispredictions |
| Wilcoxon not significant despite high ρ | n < 10 — the test needs at least 10 nodes | Treat ρ as primary evidence and Wilcoxon as inconclusive |
| Strata show "constant signal" for Topics/Brokers | Expected: these do not generate cascades as *origin* nodes; their impact accrues to connected Applications via Phase B orphaning | Not a bug. Constant I(v) = 0 for these types is correct |
| Large \|ρ − τ\| gap (> 0.15) | Agreement driven by a few extreme outliers | Inspect the top 2–3 CRITICAL components; often correct detection of a "God Component" rather than a failure |
| HSRR / DASA / RRI are 0.0 | Their input metrics are not populated ([§5.5](#55-per-dimension-validation)) | Read as "not measured", not as a failing score |

---

## 11. Known Limitations

| Limitation | Consequence |
|---|---|
| G2 and G3 measure the same quantity ([§6.1](#61-library-gates-g1g6-g8)) | Tier 1 carries three gates' worth of independent evidence, not four |
| HSRR, DASA and RRI have no data source ([§5.5](#55-per-dimension-validation)) | Three of the four availability specialist metrics are unmeasured |
| $I^*(v)$ and $I_{\text{comp}}(v)$ agree at mean ρ = 0.405 ([§3.2](#32-measured-agreement-between-the-oracles)) | Results are not transferable between the two entry points |
| IM(v) shares a substrate with M(v) ([§2](#2-two-entry-points-two-gate-systems)) | That correlation is a consistency check, not an independent test |
| Library-node I(v) is partially coupled to R(v) via Phase A ([§3.4](#34-simulation-mechanics)) | ρ restricted to Libraries is not a clean empirical test |
| 30–47 % of components carry no label | Topic and Node strata report `undefined`, not a score |
| Top-K set membership churns ~40 % between label seeds on some scenarios ([§5.2](#52-rank-correlation-and-the-label-noise-ceiling)) | Every top-K-cut metric inherits that variance |

---

## 12. What Comes Next

The validation report is the empirical evidence base for the central thesis claim. Two stages
consume it:

- **[Step 6: Prescribe](prescription.md)** reads `system_health.SRI` as the baseline against which
  each candidate architectural edit is scored in a closed-loop counterfactual simulation.
- **[Step 7: Visualize](visualization.md)** renders the results: the Q(v)-vs-I(v) scatter with
  quadrant highlighting (TP/TN/FP/FN), a delta heatmap over the topology coloured by |Q(v) − I(v)|
  to surface "architectural surprises", per-node RM radar charts, and a ranked component table
  with gate badges. It consumes `dimensional_scatter` and `confidence_intervals` directly.

For research artifacts, `compare --latex` writes a booktabs table ready for IEEE/ACM double-column
layout:

```bash
saag-validate compare \
    --input data/scenarios/atm_system.json \
    --seeds 42,123,456,789,2024 \
    --output output/ablation_final.json \
    --latex
```

---

← [Step 4: Simulate](failure-simulation.md) | → [Step 6: Prescribe](prescription.md)
