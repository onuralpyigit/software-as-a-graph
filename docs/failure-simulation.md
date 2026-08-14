# Step 4: Simulate — Failure Simulation

**Produces the ground-truth impact I(v) that the predicted criticality Q(v) is validated against, by simulating failures on the raw structural graph.**

← [Step 3: Predict](prediction.md) | → [Step 5: Validate](validation.md)

> [!NOTE]
> **Scope of this document.** It covers the two modes exposed by [`cli/simulate_graph.py`](../cli/simulate_graph.py) — `fault-inject` and `message-flow` — and the engines behind them, [`fault_injector.py`](../saag/simulation/fault_injector.py) and [`message_flow_simulator.py`](../saag/simulation/message_flow_simulator.py). The `saag/simulation/` package holds several further engines (§2) that are driven by `SimulationService` rather than by this CLI; the simulation modes themselves are enumerated by `SimulationMode` in [`saag/usecases/models.py`](../saag/usecases/models.py) and dispatched by [`saag/usecases/simulate_graph.py`](../saag/usecases/simulate_graph.py). See [ARCHITECTURE.md](../ARCHITECTURE.md) for the full engine inventory.

---

## Table of Contents

1. [Motivation and Design Rationale](#1-motivation-and-design-rationale)
2. [Architecture Overview](#2-architecture-overview)
   - 2.1 [Which engine is canonical for what](#21-which-engine-is-canonical-for-what)
3. [Mode 1 — Fault Injection](#3-mode-1--fault-injection)
   - 3.1 [Algorithm](#31-algorithm)
   - 3.2 [I(v) Formula](#32-iv-formula)
   - 3.3 [Cascade Propagation](#33-cascade-propagation)
   - 3.4 [Broker Failure Semantics](#34-broker-failure-semantics)
   - 3.5 [Library Blast-Radius Asymmetry](#35-library-blast-radius-asymmetry)
   - 3.6 [Multi-Seed Stability, Label Noise, and Reproducibility](#36-multi-seed-stability-label-noise-and-reproducibility)
4. [Mode 2 — Message Flow Simulation](#4-mode-2--message-flow-simulation)
   - 4.1 [Discrete-Event Model](#41-discrete-event-model)
   - 4.2 [Fan-Out Queue Architecture](#42-fan-out-queue-architecture)
   - 4.3 [QoS Enforcement](#43-qos-enforcement)
   - 4.4 [Fault Injection at Runtime](#44-fault-injection-at-runtime)
5. [CLI Reference — simulate\_graph.py](#5-cli-reference--simulate_graphpy)
   - 5.1 [Shared Flags](#51-shared-flags)
   - 5.2 [fault-inject](#52-fault-inject)
   - 5.3 [message-flow](#53-message-flow)
   - 5.4 [combined](#54-combined)
6. [Output Files](#6-output-files)
   - 6.1 [impact\_scores.json](#61-impact_scoresjson)
   - 6.2 [message\_flow\_results.json](#62-message_flow_resultsjson)
7. [Worked Examples — ATM Dataset](#7-worked-examples--atm-dataset)
8. [Integration with the RM Validation Pipeline](#8-integration-with-the-rm-validation-pipeline)
9. [What the Simulator Measures, in Quality-Model Terms](#9-what-the-simulator-measures-in-quality-model-terms)
   - 9.1 [Coverage by characteristic](#91-coverage-by-characteristic)
   - 9.2 [The two constraints](#92-the-two-constraints)
   - 9.3 [`I_dyn(v)` — the effectiveness term, implemented](#93-i_dynv--the-effectiveness-term-implemented)
10. [Input Graph Format Requirements](#10-input-graph-format-requirements)
11. [Python API](#11-python-api)
12. [Known Limitations](#12-known-limitations)
13. [Resolved Issues](#13-resolved-issues)
14. [What Comes Next](#14-what-comes-next)

---

## 1. Motivation and Design Rationale

The SaG framework predicts component criticality **before deployment** using topology-derived metrics (Q(v)). Validating those predictions requires a ground-truth impact score I(v) to correlate against. Because no runtime monitoring data is available pre-deployment, I(v) must itself be derived from simulation.

Two complementary simulation strategies are provided:

| Mode | When to use | Primary output |
|------|-------------|----------------|
| **Fault injection** | Producing I(v) ground truth for Spearman ρ validation | `impact_scores.json` |
| **Message flow** | Observing timing, delivery rates, and QoS violations at runtime | `message_flow_results.json` |

Both modes are **pre-deployment** — they require only the static graph JSON, never runtime monitoring data. This preserves the core claim of the SaG methodology: topology alone is sufficient to predict criticality.

---

## 2. Architecture Overview

```
cli/simulate_graph.py  (CLI entry point)
├── fault-inject subcommand (wraps FaultInjector)
│   └── saag/simulation/fault_injector.py
│       ├── _PubSubIndex          (O(1) lookup structures over PUBLISHES_TO / SUBSCRIBES_TO / ROUTES)
│       ├── FaultInjector.run()   (iterates over candidate nodes)
│       └── FaultInjector._cascade()  (BFS wave propagation per node per seed)
│
├── message-flow subcommand (wraps MessageFlowSimulator)
│   └── saag/simulation/message_flow_simulator.py
│       ├── TopicFanout           (per-topic fan-out manager)
│       ├── SubscriberQueue       (per-(topic, subscriber) SimPy Store)
│       ├── _publisher_process()  (SimPy generator: emits messages at rate_hz)
│       ├── _subscriber_process() (SimPy generator: dequeues, checks QoS)
│       └── MessageFlowSimulator.run()
│
└── combined      subcommand
    (runs fault-inject then message-flow in sequence — note the differing
     defaults documented in §5.3)
```

The full package. The two stacks below do not share a graph representation or a
result model, and the split is stated in the package docstring
([`__init__.py`](../saag/simulation/__init__.py)):

| Module | Role |
|---|---|
| **Predict-stage labeler** | |
| [`fault_injector.py`](../saag/simulation/fault_injector.py) | `FaultInjector` — canonical I\*(v) labeler. Runs on a raw `networkx.DiGraph` |
| [`message_flow_simulator.py`](../saag/simulation/message_flow_simulator.py) | `MessageFlowSimulator` — SimPy timing/QoS simulator |
| [`simulation_results.py`](../saag/simulation/simulation_results.py) | Result dataclasses for both of the above, plus their JSON `save()` |
| **Validate-stage oracle** | |
| [`failure_simulator.py`](../saag/simulation/failure_simulator.py) | `FailureSimulator` — canonical RM oracle (`ImpactMetrics`) |
| [`event_simulator.py`](../saag/simulation/event_simulator.py) | `EventSimulator` — discrete-event run supplying the baseline flows |
| [`change_propagation.py`](../saag/simulation/change_propagation.py) | `ChangePropagationSimulator` — IM(v) sub-metrics |
| [`graph.py`](../saag/simulation/graph.py) | `SimulationGraph` — state-aware view over raw structural edges |
| [`models.py`](../saag/simulation/models.py) | `ImpactMetrics`, scenarios, enums for this stack |
| [`service.py`](../saag/simulation/service.py) | `SimulationService` — orchestration and reporting |
| [`processor.py`](../saag/simulation/processor.py) | `ComplexityProcessor` — complexity-derived processing latency |
| **Shared / standalone** | |
| [`_stats.py`](../saag/simulation/_stats.py) | The two percentile estimators, which are not interchangeable |
| [`traffic_simulator.py`](../saag/simulation/traffic_simulator.py) | `TrafficSimulator` — closed-form bandwidth calculator (§12 L11) |

### 2.1 Which engine is canonical for what

`FaultInjector` and `FailureSimulator` both emit something called "impact", and the two
quantities are **not** interchangeable — see the warning in
[`saag/simulation/models.py`](../saag/simulation/models.py). Each owns one pipeline stage:

| Stage | Engine | Output | Consumed by |
|-------|--------|--------|-------------|
| **Predict** (supervised labels) | `FaultInjector` | `impact_scores.json` → `I*(v)`, a scalar | GNN training, k-fold / LOSO evaluation |
| **Validate** (quality oracle) | `FailureSimulator` | `ImpactMetrics` → composite + IR/IM (IR hierarchical: fault_tolerance_impact/availability_impact sub-terms) | `saag/validation/` gates |

`FaultInjector` is the labeler because it is deterministic, multi-seed, and records
per-node variance — the properties a training label needs. `FailureSimulator` supplies
the two-dimensional RM decomposition (plus the fault_tolerance/availability
sub-characteristic diagnostics) that the validation gates are written against.

**The two must never be mixed inside one stage.** This is enforced by
[`tests/test_groundtruth_contract.py`](../tests/test_groundtruth_contract.py), which also
checks that the emitted artifact names its own labeler, so a cache can always be traced
back to the engine that wrote it.

The CLI uses a **subcommand pattern** so fault injection and message flow share a common `--input` / `--output` / `--export-json` / `--verbose` interface while each exposes its own mode-specific flags.

---

## 3. Mode 1 — Fault Injection

### 3.1 Algorithm

The fault injector runs a **BFS cascade simulation** on the pub-sub graph for every candidate node. It operates over the pub-sub edges (`PUBLISHES_TO`, `SUBSCRIBES_TO`, and `ROUTES`) as well as the derived dependency edges (`DEPENDS_ON`), which are dynamically derived during initialization if absent from the input graph.

#### Dynamic DEPENDS_ON Derivation

If the input graph does not contain any `DEPENDS_ON` edges, the `FaultInjector` derives them dynamically in its constructor based on other relationships:
1. **App-to-App dependencies**: If an Application node $A_{sub}$ subscribes to a topic $T$ that is published to by Application $A_{pub}$, a `DEPENDS_ON` edge from $A_{sub}$ to $A_{pub}$ is created with `dependency_type="app_to_app"`, `weight=1.0`, and the QoS profile from the edges.
2. **App-to-Library dependencies**: If an Application node $A$ uses a library/dependency $L$ (via `USES` relationship), a `DEPENDS_ON` edge from $A$ to $L$ is created with `dependency_type="app_to_lib"`, `weight=1.0`.

Before any injection begins, `_PubSubIndex` builds six lookup dictionaries from the graph in $O(E)$:

| Dictionary | Maps |
|---|---|
| `topic_publishers` | topic → set of publisher application IDs |
| `topic_subscribers` | topic → set of subscriber application IDs |
| `app_publishes` | application → set of topic IDs it publishes to |
| `app_subscribes` | application → set of topic IDs it subscribes to |
| `broker_routes` | broker → set of topic IDs it routes |
| `topic_routers` | topic → set of broker IDs that route it (inverse of `broker_routes`) |

For each candidate node $v$ and seed, the cascade runs as follows:

**Wave 0, 1, 2, ... — Cascade Waves**

For each wave (starting with the injected node $v$ in wave 0), the simulator executes two sequential phases:

##### Phase A: Direct DEPENDS_ON Propagation
For each node $u$ in the current wave's frontier:
1. Find all incoming edges $(v_{dep}, u)$ in the graph representing $v_{dep} \xrightarrow{\text{DEPENDS\_ON}} u$ (meaning $v_{dep}$ depends on $u$).
2. If $v_{dep}$ is not already failed, it fails with probability `prob`, which takes exactly two values:

   | Condition on the edge $(v_{dep}, u)$ | `prob` |
   |---|---|
   | `dependency_type == "app_to_lib"`, **or** $u$ is itself typed `Library` | `1.0` — the dependent always fails |
   | anything else (i.e. `app_to_app`) | `0.0` — no propagation |

   Phase A applies **no depth damping**: `depth_damp` is used only in Phase B. Because the only non-zero value is `1.0`, this phase is in practice deterministic — a Library failure takes every consumer with it at wave 0, and no `app_to_app` edge ever propagates here. See [§3.5](#35-library-blast-radius-asymmetry) for what that means for Library labels.

##### Phase B: Topic-mediated Soft QoS/Rate-weighted Propagation
1. **Continuous Topic Feed Loss**:
   For each topic $t$, the feed loss $L(t) \in [0.0, 1.0]$ is calculated dynamically based on failed publishers or failed brokers:
   - If the topic has publishers:
     $$L(t) = \frac{\sum_{p \in \text{failed\_publishers}(t)} \text{rate\_hz}(p, t)}{\sum_{p \in \text{all\_publishers}(t)} \text{rate\_hz}(p, t)}$$
     where `rate_hz` is the publish rate (defaults to 10.0 Hz). If the total rate is 0, it falls back to the fraction of failed publishers: $\frac{|\text{failed\_publishers}(t)|}{|\text{all\_publishers}(t)|}$.
   - If the topic has no publishers but has broker routers, the loss is the fraction of failed routers:
     $$L(t) = \frac{|\text{failed\_routers}(t)|}{|\text{all\_routers}(t)|}$$
   - The loss is then scaled by the topic's QoS criticality factor and capped at 1.0:
     $$L(t) = \min(1.0, L(t) \times \text{QoS\_factor}(t))$$
     The factor is selected by `--qos-factor` / `FaultInjector(qos_factor_mode=...)`:

     | Mode | $\text{QoS\_factor}(t)$ |
     |---|---|
     | `ladder` (default) | `1.0`, ×`1.2` if reliability is `RELIABLE`, ×`1.15` if transport priority is `HIGH`/`CRITICAL`/`URGENT`/`HIGHEST`, ×`1.05` if `MEDIUM` |
     | `wt` | $\max(0,\ 1 + \kappa\,(w(t) - \overline{w}))$ with $\kappa = 0.5$ — the same $w(t)$ the rest of the codebase uses, so **durability** participates (it carries the largest AHP sub-weight, 0.40, and the ladder ignores it entirely) |
     | `none` | `1.0` — the topology-only arm used by [`reproduce/qos_label_ablation.py`](../reproduce/qos_label_ablation.py) |

QoS is resolved through `QoSPolicy.from_node_attrs`, which accepts both the flat
(`qos_transport_priority`) and nested (`qos: {...}`) attribute shapes — see
[`tests/test_qos_resolution.py`](../tests/test_qos_resolution.py). Artifacts generated before
this resolution was fixed carry `--qos-factor none` labels regardless of what their
provenance block claims ([§13 R1](#13-resolved-issues)).

2. **Orphaned Topic and Subscriber Impact Tracking**:
   - If $L(t) > 10^{-6}$ and the topic was not previously orphaned, it is added to `orphaned_topics`. If this occurs during Wave 0, the topic is also added to `directly_orphaned_topics`.
   - All subscriber applications of $t$ that are not already failed are marked as impacted.

3. **Stochastic Subscriber Failure**:
   For each subscriber application $s$, we compute its average feed loss across all its subscribed topics:
   $$\text{sub\_loss}(s) = \frac{\sum_{t \in \text{subscribed\_topics}(s)} L(t)}{|\text{subscribed\_topics}(s)|}$$
   If $\text{sub\_loss}(s) \ge \text{propagation\_threshold}$ (and $\text{sub\_loss}(s) > 10^{-6}$):
   - The subscriber fails stochastically with probability:
     $$P_{\text{fail}}(s) = \min\left(1.0, \frac{\text{sub\_loss}(s)}{\text{propagation\_threshold}}\right) \times \text{depth\_damp}$$
     Where:
     - $\text{depth\_damp} = \max(0.25, 1.0 - \text{wave\_idx} \times 0.15)$ is a depth-based damping factor to prevent runaway cascade propagation.
   - If the random check succeeds, $s$ is added to the next wave's frontier.

---

### 3.2 I(v) Formula

There are two parallel ground-truth definitions computed by the simulation suite:

1. **`FaultInjector` (BFS feed-loss / diagnostic simulator)**:
   Computes the average subscriber feed loss across all system subscribers:
   $$I(v) = \frac{\sum_{s \in \text{all\_subscribers}} \text{sub\_loss}(s)}{|\text{all\_subscribers}|}$$
   This is the metric computed dynamically in the CLI `fault-inject` subcommand and legacy validation wrappers (`cli/validate_graph.py`), and saved to `impact_scores.json`.

2. **`FailureSimulator` (Canonical composite simulator)**:
   Computes the four-component weighted composite $I^*(v)$ returned by `ImpactMetrics.composite_impact`:
   $$I^*(v) = 0.35 \cdot \text{reachability\_loss} + 0.25 \cdot \text{fragmentation} + 0.25 \cdot \text{throughput\_loss} + 0.15 \cdot \text{flow\_disruption}$$

   The weights are AHP-derived and now come from `AHPProcessor.compute_weights()`
   rather than a literal, so the pairwise matrix in
   [`weight_calculator.py`](../saag/analysis/weight_calculator.py) actually drives the
   scorer. The exact values are $0.3472 / 0.2538 / 0.2538 / 0.1453$; the figures
   above are those rounded.

   Each term is weighted by the **QoS severity** $s(t) = w(t)\cdot\text{rate}(t)$ of
   what was actually lost, rather than counting broken paths equally
   (`qos_weighting=True`, the default; `False` restores the count-based form and is
   the topology-only arm of the label ablation):

   - **reachability\_loss**: fraction of weighted pub-sub path capacity broken. Already
     QoS-weighted before this change, since path capacity uses edge weights that
     inherit $w(t)$.
   - **fragmentation**: $0.70\cdot(\text{new islands}/\text{max islands}) + 0.30\cdot(\text{QoS mass stranded off the largest island})$,
     measured **relative to the healthy graph** so a topology that is already
     disconnected does not charge every component for a pre-existing island.
   - **throughput\_loss**: $\sum_t s(t)\cdot\ell(t) / \sum_t s(t)$ where
     $\ell(t) = \max(\text{failed publisher fraction},\ \text{failed router fraction})$.
     Continuous and broker-aware — the previous form was binary on
     publishers/subscribers only, so a topic whose sole routing broker died was
     scored as fully delivering.
   - **flow\_disruption**: $\sum_{\text{broken}} s(t) / \sum_{\text{all}} s(t)$ over the
     baseline Pub→Topic→Sub triples, rather than a plain broken/total count.

> [!NOTE]
> **QoS is a severity weight, not a fifth term.** It is not an independent dimension
> of *what breaks* — it is how much the same breakage costs. Adding it additively
> would score a component above zero on QoS while nothing it touches has broken, and
> would double-count against the $w(t)$ already inside `reachability_loss`.

   This composite score is computed by the GNN training services (`cli/train_graph.py`) and validation services (`saag/validation/service.py`) to provide the main Middleware 2026 and RASSE evaluation metrics.

> [!NOTE]
> **Starvation signal role**. In `FaultInjector`, the average subscriber feed loss $\text{sub\_loss}(s)$ is directly aggregated into the final $I(v)$ score. In `FailureSimulator`, however, feed loss and starvation are strictly internal propagation signals used to determine cascade eligibility (§3.1, Stochastic Subscriber Failure); the final $I^*(v)$ is computed using the structural and flow-based metrics shown above.

> [!NOTE]
> **Start-node inclusion in `reachability_loss`.** The failed node $v$ itself is counted as a lost subscriber in the reachability loss calculation, which could give subscriber-heavy nodes a modest advantage in I(v). To quantify this, we ran an exclusion sweep on `data/system.json`: excluding $v$ from its own feed-loss denominator shifts the system-layer Spearman $\rho$ by $+0.0077$ (0.7856 → 0.7933) and leaves the top-5 critical-node ranking identical (minor rank swaps only). The bias is therefore negligible and the current behaviour is retained for implementation simplicity.

---

### 3.3 Cascade Propagation

The `propagation_threshold` parameter (default `0.2`, range $[0.0, 1.0]$) controls the minimum average feed loss required before a subscriber is eligible to fail stochastically and propagate the cascade.

| `propagation_threshold` | Semantic |
|---|---|
| `0.2` (default) | A subscriber is eligible to fail when its average feed loss is $\ge 20\%$. Aggressive default. |
| `0.5` | A subscriber is eligible to fail when its average feed loss is $\ge 50\%$. |
| `1.0` | A subscriber only cascades when it has lost $100\%$ of its feeds (completely starved). Conservative. |
| `0.0` | Any single feed loss triggers eligibility to cascade. Extremely aggressive. |

For the ATM dataset, `ConflictDetector` requires both `T_radar` **and** `T_tracks` to function (both are mandatory inputs to the conflict algorithm). Setting `--propagation-threshold 0.5` will model this correctly: losing either feed alone is sufficient to silence `ConflictDetector`.

> [!NOTE]
> **$P_{\text{fail}}$ step-function discontinuity at `propagation_threshold`.** Because eligibility is gated by `sub_loss >= propagation_threshold`, the cascade probability function is a **step function**: it is exactly $0.0$ for any `sub_loss` below the threshold, then jumps immediately to $P = \text{depth\_damp}$ (constant) for all $\text{sub\_loss} \ge \text{propagation\_threshold}$ (since the ratio $\text{sub\_loss} / \text{propagation\_threshold} \ge 1.0$ is clamped by $\min(1.0, \dots)$). This means there is no gradual ramp or scaling in the eligible region — a subscriber is either completely ineligible or immediately assigned a constant probability of $1.0 \times \text{depth\_damp}$. An alternative design would use a **linear ramp** (no guard condition; `prob = sub_loss / threshold * depth_damp` for all `sub_loss > 0`) or a **sigmoid** to produce smooth eligibility. The current step-function is a deliberate conservative choice: partial feed loss below the threshold is treated as recoverable degradation, not a cascade trigger. Reviewers who prefer the linear ramp may pass `--propagation-threshold 0.0` to approximate it.

---

### 3.4 Broker Failure Semantics

When a Broker node fails, the injector computes the continuous topic feed loss as the fraction of failed brokers that route each topic:
$$L(t) = \frac{|\text{failed\_routers}(t)|}{|\text{all\_routers}(t)|}$$
This correctly handles multi-broker redundancy: if a topic is routed by two brokers, the failure of one broker results in a continuous feed loss of $0.5$ (50%) rather than a complete binary failure ($1.0$ loss). If all routing brokers for a topic fail, the feed loss becomes $1.0$ (100%).

---

### 3.5 Library Blast-Radius Asymmetry

Libraries occupy an asymmetric position between $Q(v)$ (structural quality prediction) and $I(v)$ (simulation ground truth):

**Visible to $Q(v)$:** The structural analyzer creates `app_to_lib` (`DEPENDS_ON`) edges from every consuming Application to the Library. These edges contribute to the Library's in-degree, betweenness, and Reliability dimension score. A widely-used library therefore scores high on the $R(v)$ dimension — its blast radius is structurally significant.

**Also visible to `FaultInjector` $I(v)$.** `FaultInjector.__init__` derives
`DEPENDS_ON(app → lib, dependency_type="app_to_lib")` from `USES` edges when the input
graph carries none, and `_cascade` propagates those edges at `prob = 1.0`. A Library
failure therefore fails every consuming Application at wave 0, and those Applications then
orphan the topics they solely publish — the blast radius does reach subscribers.

Measured on the regenerated LOSO caches (five seeds, `--node-types Application,Broker,Library`):

| Scenario | Library nodes | non-zero | mean $I(v)$ | max $I(v)$ |
|----------|--------------:|---------:|------------:|-----------:|
| `atm_system` | 8 | 6 | 0.400 | 0.705 |
| `healthcare_system` | 12 | 12 | 0.922 | 0.960 |
| `microservices_system` | 30 | 30 | 0.428 | 0.514 |

Libraries are consistently among the *highest*-impact node types, which matches their
structural footprint rather than contradicting it.

**$T_0$ Step-Function Collapse in FailureSimulator**: `FailureSimulator` models library failure as a **$T_0$ step-function collapse**: all consuming Applications that use the Library fail immediately at depth 0. The subsequent propagation of these Application failures forward through the pub-sub topic graph is more restricted than in `FaultInjector`, so the two engines rank libraries differently. That divergence is expected — they measure different quantities (§2.1).

> [!NOTE]
> The Fault Tolerance $FT(v)$ formula (documented in [structural-analysis.md](structural-analysis.md#fault-tolerance-ftv--fault-propagation-risk) — $FT(v)$ is a term of $R(v)$, not $R(v)$ itself) already includes the normalized in-degree term $DG\_in(v)$, which captures the number of direct consumers (blast radius) for both Applications and Libraries. This is the correct place to tune the Library's structural influence if the asymmetry is considered too large.

---

### 3.6 Multi-Seed Stability, Label Noise, and Reproducibility

The cascade propagation order within a wave is non-deterministic when multiple nodes are eligible to propagate simultaneously (tie-breaking). Each seed produces a different shuffle of the wave candidates, testing whether I(v) depends on this ordering.

With N seeds:
- `impact_score` is the **mean** I(v) across all seeds.
- `impact_score_std` is the **standard deviation** across seeds.
- The cascade trace (waves, orphaned topics, impacted subscribers) in the JSON record is from the **seed whose impact score is closest to the mean** (median-representative seed), giving the most stable trace for human inspection.

**Interpreting std values:**
- `std = 0.0` — each seed produced an identical result for that node.
- `std > 0` indicates that I(v) is sensitive to the propagation order, typically at the boundary of a cascade — a signal of fragility that is itself worth reporting.

> [!NOTE]
> **Stochasticity limits on shallow cascades.** Because the depth damping factor at wave 0 is exactly `1.0` (causing all eligible subscribers to fail deterministically) and stochastic propagation through pure `DEPENDS_ON` edges is disabled (`prob = 0.0`), standard deviation is always `0.0` for shallow cascades resolving completely at wave 0. Multi-seed averaging only affects deep cascades resolving at waves $\ge 1$ where `depth_damp < 1.0` introduces probabilistic failures.

Recommended seeds — and the CLI default: `42,123,456,789,2024`.

#### The `label_stability` block

Per-node `impact_score_std` answers "is this node's score stable?". It does not answer
"how much can I trust a correlation computed against this whole label set?". Every artifact
therefore carries an aggregate `label_stability` block:

```json
"label_stability": {
  "n_seeds": 5,
  "n_nodes": 39,
  "k_frac": 0.20,
  "mean_std": 0.026726,
  "max_std": 0.1856,
  "test_retest_spearman": 0.980215,
  "topk_jaccard": 0.625
}
```

| Field | Meaning |
|-------|---------|
| `test_retest_spearman` | **Worst** pairwise Spearman ρ between any two seeds' label vectors. This is the ceiling on any reported ρ. |
| `topk_jaccard` | **Worst** pairwise overlap of the top-`k_frac` critical sets across seeds. |
| `mean_std` / `max_std` | Mean and worst per-node standard deviation. |
| `n_seeds` | Number of seeds. With one seed, the correlation fields are `null` and a `note` explains why — a single seed cannot establish a ceiling, and reporting `1.0` would overstate label quality. |

**Both aggregates report the worst pair, not the mean.** The ceiling is set by the weakest
agreement; averaging hides it.

Measured across the scenario cohort:

| Scenario | `test_retest_spearman` | `topk_jaccard` |
|----------|----------------------:|---------------:|
| `microservices_system` | 0.928 | 0.560 |
| `atm_system` | 0.980 | 0.625 |
| `av_system` | 0.985 | 0.714 |
| `financial_trading_system` | 0.990 | 0.647 |
| `hub_and_spoke_system` | 0.996 | 0.947 |
| `enterprise_system` | 0.996 | 1.000 |
| `healthcare_system` | 0.998 | 0.923 |
| `iot_smart_city_system` | 1.000 | 1.000 |

**Read the two columns separately.** Rank correlation is high everywhere (≥ 0.93), but the
*critical set* is much less stable: on `microservices_system` roughly 44% of the top-20%
changes between seeds. Metrics defined on a top-K cut — Overlap@K, P@τ, R@τ — inherit that
churn, while ρ and NDCG largely do not. A reported Overlap@K of 0.60 on that scenario is
within the labels' own noise.

`cli/loso_evaluate.py` propagates this block into `summary.md`, so the ceiling is printed
next to the achieved ρ rather than having to be looked up.

#### Reproducibility

`FaultInjector` seeds a fresh `random.Random(seed)` per (node, seed) pair, so results are
reproducible across runs and processes.

`FailureSimulator.simulate_exhaustive` takes a `seed` argument (default `42`) and derives a
per-component seed as `run_seed ^ zlib.crc32(component_id)`. Two properties matter here:

- **`zlib.crc32`, not `hash()`** — `hash(str)` is salted by `PYTHONHASHSEED`, which would
  make labels differ between processes.
- **Derived from the component id, not its index** — a component's label must not shift
  because a LOSO fold changed how many other components share the sweep.

Pass `seed=None` to restore free-running behaviour. Before seeding, identical exhaustive
sweeps disagreed with each other: on `healthcare_system`, run-to-run ρ fell to 0.909 with
8% of the top-20% set churning between runs — a noise floor barely above the ρ ≥ 0.85 gate
it was being used to enforce. See
[`tests/test_label_determinism.py`](../tests/test_label_determinism.py).


---

## 4. Mode 2 — Message Flow Simulation

### 4.1 Discrete-Event Model

The message flow simulator uses **SimPy** (https://simpy.readthedocs.io) — a process-based discrete-event simulation library. Simulated time is in seconds, mapping 1-to-1 to the real-world time units of the modelled system.

Three types of SimPy process are spawned for the topology:

**Publisher process** (one per `PUBLISHES_TO` edge):
1. **Determine Publish Interval**: The publish rate (`rate_hz`) is resolved using `generate_workload(topic_id)`. If multiple publishers publish to the same topic, the topic's configured frequency is divided equally among all active publishers to maintain the aggregate topic frequency:
   $$\text{rate\_hz} = \frac{\text{base\_rate}}{\text{num\_publishers}}$$
   The simulator yields a timeout interval:
   - **Poisson Workload**: If `workload_type` on the Topic node is `"poisson"`, the interval is sampled stochastically from an exponential distribution: `rng.expovariate(rate_hz)`.
   - **Periodic Workload**: Otherwise, the interval is deterministic: `1.0 / rate_hz`.
2. **Failure Check**: If `app_id in failed_nodes`, the publisher process exits.
3. **Processing Delay**: Yields a publisher-side compute delay if `processing_time` is configured.
4. **Publish Message**: Creates a `Message` and calls `fanout.publish(msg, failed_nodes)` to place the message in all live subscriber queues.
5. **Window Counters**: To track delivery rates before and after the fault, the publisher increments the appropriate time-window publish counter (`pre` or `post` fault) based on whether `env.now < fault_time`.

**Subscriber process** (one per `SUBSCRIBES_TO` edge):
1. **Pre-dequeue Failure Check**: Checks `app_id in failed_nodes` **before** calling `get()`. If failed, it exits immediately.
2. **Dequeue**: Dequeues a message from the subscriber's private queue: `msg = yield sq.get()`.
3. **Post-dequeue Failure Check**: Checks `app_id in failed_nodes` again. If failed, the message is marked as missed and the process exits.
4. **Subscriber Processing**: Yields a subscriber-side processing delay (models application compute overhead).
5. **End-to-End Latency**: Calculates end-to-end latency *after* subscriber processing:
   $$\text{e2e\_latency\_ms} = (\text{env.now} - \text{msg.created\_at}) \times 1000$$
6. **QoS Verification**: Evaluates lifespan and deadline checks against `e2e_latency_ms`.
7. **Delivery Logging**: If all QoS checks pass, increments topic delivery stats and logs the latency sample. Increments `pre` or `post` time-window delivery counters.

**Fault process** (one per simulation, if `--fault-node` is set):
1. `yield env.timeout(fault_time)`.
2. Adds `fault_node` to `failed_nodes` set. Publisher and subscriber processes observe this on their next loop iteration.

All three process types share the same `failed_nodes: Set[str]` object, which serves as the inter-process fault broadcast channel.

> **Latency windowing.** The subscriber process also buckets each delivered-message end-to-end latency into a shared `latency_windows` dict (`"pre"` / `"post"` keys) keyed on whether `arrival_time < fault_time`. After `env.run()`, the four summary percentiles (`latency_p50_before`, `latency_p50_after`, `latency_p95_before`, `latency_p95_after`) are aggregated via a linear-interpolated percentile helper and written to `FaultEventRecord`. These fields are `None` when no fault was injected or when a window received no deliveries. Their primary use is as an independent I_dyn(v) ground-truth candidate for convergent validity (see [validation.md §10](validation.md#10-interpreting-results)).

### 4.2 Fan-Out Queue Architecture

Standard pub-sub semantics require that **every subscriber receives every message**. A naive single `simpy.Store` per topic would instead route each message to exactly one subscriber (first-come-first-served dequeue), halving — or worse — per-subscriber delivery counts.

The simulator uses a two-level architecture:

```
Publisher
    │
    │  fanout.publish(msg, failed_nodes)
    ▼
TopicFanout
    ├──▶ SubscriberQueue[Sub1]  (simpy.Store, capacity = queue_size)
    ├──▶ SubscriberQueue[Sub2]
    └──▶ SubscriberQueue[Sub3]
              │
              │  sq.get()
              ▼
         Subscriber process
```

`TopicFanout.publish()` iterates over every registered subscriber queue and places a copy of the message in each live subscriber's `SubscriberQueue`. Overflow policy (BEST_EFFORT drop vs. RELIABLE head-drop) is applied independently per subscriber queue.

`TopicFlowStats.total_published` is incremented **once per message** (not once per subscriber that receives it). `total_delivered` counts individual (message × subscriber) deliveries. The system delivery rate is normalised accordingly:

```
system_delivery_rate = total_delivered / (Σ_topic total_published(topic) × num_subscribers(topic))
```

### 4.3 QoS Enforcement

QoS attributes are read from two sources in priority order:

1. The Topic node (`qos_profile` attribute) — topic-level policy; `deadline_ms` takes precedence over the edge-level value when set.
2. The `SUBSCRIBES_TO` edge (`qos_profile` attribute) — subscriber-side policy.

Both sources follow the same structure:

```json
{
  "reliability": "RELIABLE",
  "durability":  "VOLATILE",
  "deadline_ms": 100,
  "lifespan_ms": null,
  "queue_size":  50,
  "history_depth": 10
}
```

**Reliability** (`RELIABLE` / `BEST_EFFORT`) governs overflow behaviour in each `SubscriberQueue`:
- `RELIABLE` — when the queue is full, the **oldest** message is dropped (head-drop) to make room for the newest. This models DDS KEEP\_LAST semantics with backpressure.
- `BEST_EFFORT` — when the queue is full, the **incoming** message is dropped. The overflow event is counted in `total_dropped_best_effort`.

**Deadline** (`deadline_ms`) is enforced as an **end-to-end** check, measured after the subscriber processing delay:
```
e2e_latency_ms = (env.now_after_processing - msg.created_at) × 1000
if e2e_latency_ms > deadline_ms:
    → deadline violation; message counted as missed
```
This matches the DDS definition: the deadline is the maximum acceptable age of a data sample at the point it is consumed by the application.

> [!IMPORTANT]
> **Topic-Level Overrides:** If a `deadline_ms` is set on the Topic node, it takes precedence and overrides any `deadline_ms` defined on the `SUBSCRIBES_TO` edge.

**Lifespan** (`lifespan_ms`) is applied before the deadline check. Messages older than their lifespan at the time of dequeue are silently discarded.

**Durability** (`TRANSIENT_LOCAL`) is noted in the QoS profile but is not fully modelled in the current simulator (no late-joiner history replay). This is documented in [Known Limitations](#12-known-limitations).

### 4.4 Fault Injection at Runtime

The `_fault_process` yields until `fault_time`, then adds `fault_node` to the shared `failed_nodes` set. Publishers and subscribers observe this lazily on their next loop iteration:

- **Publisher**: checks `app_id in failed_nodes` at the top of the loop after each interval wait. The publisher silently exits, stopping all further messages to any topic it published to.
- **Subscriber**: checks `app_id in failed_nodes` before issuing `get()` (fast exit), and again immediately after receiving a message (handles races where the fault was injected while the subscriber was blocked in the queue wait).

Post-simulation, the cascade annotation identifies:
- **Orphaned topics**: topics where the faulted node was the **sole** publisher (verified by checking remaining PUBLISHES\_TO edges for that topic).
- **Impacted subscribers**: all subscribers of orphaned topics.
- **Delivery rate before/after**: computed from per-topic time-window publish and delivery counters accumulated by publisher and subscriber processes respectively.

---

## 5. CLI Reference — simulate_graph.py

Pipeline context and how this step fits with the others is in
[cli-pipeline-guide.md §Step 4](cli-pipeline-guide.md). This section is the flag reference.

### 5.1 Shared Flags

All three subcommands accept these:

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | *(none)* | Path to the graph JSON file. Optional — resolved from `--layer` when omitted. |
| `--layer NAME` | *(none)* | Resolve the input as `data/<layer>.json` instead of passing `--input`. |
| `--output DIR` | `output/simulation/` | Output directory; created if absent. |
| `--export-json` | off | Write the JSON result files and their `.txt` summaries to `--output`. |
| `--verbose` / `-v` | off | Enable DEBUG logging (per-node I(v), per-topic stats). |

### 5.2 `fault-inject`

Runs BFS cascade fault injection and produces `impact_scores.json`.

```bash
PYTHONPATH=. python cli/simulate_graph.py fault-inject [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--nodes ID1,ID2,...` | all matching `--node-types` | Comma-separated node IDs to inject. Overrides `--node-types`. |
| `--node-types TYPE1,TYPE2` | `Application,Broker,Library` | Node types eligible for injection. Types omitted here get **no** ground truth and are listed in the artifact's `unlabeled_node_ids`. |
| `--seeds 42,123,...` | `42,123,456,789,2024` | Comma-separated integer seeds. Labels are the per-node mean; ≥ 2 seeds are required for `label_stability` to be measurable. |
| `--cascade-depth N` | `0` (unlimited) | Maximum cascade wave depth. |
| `--propagation-threshold F` | `0.2` | Minimum average feed loss before a subscriber is eligible to cascade ([§3.3](#33-cascade-propagation)). |
| `--qos-factor MODE` | `ladder` | `ladder` \| `wt` \| `none` — how topic QoS scales feed loss ([§3.1](#31-algorithm)). `none` is the topology-only ablation arm. |

> [!WARNING]
> **Do not add `Topic` or `Node` to `--node-types`.** The cascade derives `DEPENDS_ON` only
> from `PUBLISHES_TO`, `SUBSCRIBES_TO` and `USES`, so it has no way to express the failure
> of a Topic or a physical Node: **every** instance of either scores exactly $I(v) = 0$.
> Those are not measurements of "no impact" — they are the absence of a model. Including
> them adds a block of 25–45 constant-zero labels per scenario (see [§12 L6](#12-known-limitations)) and trains the
> model toward a constant.
>
> `FaultInjector.run()` detects this and emits a `DEGENERATE LABELS` warning naming any node
> type whose entire label set came out zero. If you see that warning, remove the type.

**Example — full ATM dataset, five seeds:**

```bash
PYTHONPATH=. python cli/simulate_graph.py fault-inject \
    --input data/scenarios/atm_system.json \
    --output output/simulation/ \
    --seeds 42,123,456,789,2024 \
    --export-json
```

### 5.3 `message-flow`

Runs the SimPy discrete-event message flow simulation.

```bash
PYTHONPATH=. python cli/simulate_graph.py message-flow [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--duration SECONDS` | `100.0` | Simulation duration in simulated seconds. |
| `--fault-node NODE_ID` | none | Node to fault during simulation. Omit for a clean baseline run. |
| `--fault-time SECONDS` | `duration / 2` | When to inject the fault. Independent of *which* node is faulted. |
| `--seed INT` | `42` | Random seed for publish jitter and processing time variation. |
| `--default-rate HZ` | `10.0` | Fallback publish rate when absent from graph metadata. |
| `--default-queue-size N` | `100` | Fallback per-subscriber queue capacity. |

**Example — fault a broker at the midpoint:**

```bash
PYTHONPATH=. python cli/simulate_graph.py message-flow \
    --input data/scenarios/atm_system.json \
    --duration 300 \
    --fault-node ASTERIX_Broker \
    --fault-time 150 \
    --export-json
```

### 5.4 `combined`

Runs `fault-inject` and `message-flow` in sequence. `--fault-node` serves both modes: it
selects the message-flow fault target and can be combined with `--nodes` to restrict
fault-inject to the same node.

```bash
PYTHONPATH=. python cli/simulate_graph.py combined \
    --input data/scenarios/atm_system.json \
    --seeds 42,123,456,789,2024 \
    --duration 300 --fault-node ASTERIX_Broker --fault-time 150 \
    --export-json
```

> [!WARNING]
> **`combined` does not inherit the subcommand defaults.** Two of its own defaults differ,
> and both silently degrade the output:
>
> | Flag | `fault-inject` default | `combined` default | Consequence |
> |---|---|---|---|
> | `--node-types` | `Application,Broker,Library` | `Application,Broker` | Libraries get no ground truth, reproducing the exclusion [§3.5](#35-library-blast-radius-asymmetry) describes as fixed |
> | `--seeds` | `42,123,456,789,2024` | `42` | One seed makes every `label_stability` correlation `null`, voiding the publication gate in [§8](#8-integration-with-the-rm-validation-pipeline) |
>
> Pass both flags explicitly on any `combined` run whose output you intend to publish.

---

## 6. Output Files

### 6.1 `impact_scores.json`

Written by `fault-inject`. This is the canonical I(v) ground-truth file consumed by the RM validation pipeline.

```
output/simulation/
├── impact_scores.json          ← full result with all records
└── impact_scores_summary.txt   ← human-readable ranked table
```

**Top-level structure:**

```json
{
  "schema_version": "2.1",
  "graph_id": "atm_system",
  "total_nodes_injected": 39,
  "total_application_nodes": 26,
  "total_broker_nodes": 5,
  "total_subscribers": 21,
  "seeds_used": [42, 123, 456, 789, 2024],

  "labeler": "FaultInjector",
  "labeled_node_types": ["Application", "Broker", "Library"],
  "labeled_dimensions": ["composite", "reliability"],
  "unlabeled_node_ids": ["N0", "N1", "N2", "N3", "N4", "..."],
  "label_stability": { "...": "see §3.6" },

  "top_k_by_impact": [ ... ],
  "records": { ... }
}
```

**Provenance fields (added in schema 2.1):**

| Field | Why it exists |
|-------|---------------|
| `labeler` | Names the engine that wrote the file. Two engines emit differently-scaled "impact" (§2.1); a consumer that cannot tell them apart cannot know what its numbers mean. |
| `labeled_node_types` | The types actually injected. |
| `labeled_dimensions` | The label dimensions this engine genuinely **measured**. `FaultInjector` emits one scalar, which maps onto `composite` / `reliability` (`reliability` is itself the α-blend of fault_tolerance and availability, so this labeler does not additionally declare `availability` as a separate measured dimension) — it says nothing about maintainability, so that is **absent**, not zero. |
| `unlabeled_node_ids` | Nodes present in the graph but never injected. Makes the coverage gap explicit instead of letting it vanish in a downstream set intersection. |
| `label_stability` | The labels' own reproducibility — the ceiling on any ρ reported against them. See §3.6. |

> [!IMPORTANT]
> **Absent is not zero.** The artifact distinguishes *not measured* from *measured as zero*,
> in two places, and consumers must preserve the distinction:
>
> - **Dimensions.** `extract_simulation_dict` emits only the dimensions the labeler declared
>   in `labeled_dimensions`; `networkx_to_hetero_data` derives a `dimension_mask` from them
>   so unmeasured columns are excluded from the multitask loss. Emitting `0.0` for
>   `maintainability` instead would train a prediction head against a
>   constant ([§13 R3](#13-resolved-issues)).
> - **Nodes.** A node the simulator targeted and scored `0.0` is a real observation at the
>   low end of the ranking; a node never targeted is missing data. `label_mask` carries
>   presence-in-the-artifact explicitly rather than inferring it from `|y_composite| > 1e-6`,
>   which would drop genuine zeros from the loss while still scoring the model on them
>   (7–115 nodes per scenario; 37% on `enterprise_system`).

**Backward compatibility.** Schema 2.0 files still parse. They carry no provenance fields,
so consumers fall back to the historical behaviour and `label_stability` is unavailable.
Regenerate with `scripts/populate_loso_cache.sh` to pick up the new fields.

**`top_k_by_impact`** — ranked list (top 20 by default):

```json
[
  {
    "rank": 1,
    "node_id": "RadarTracker",
    "node_type": "Application",
    "node_name": "RadarTracker",
    "impact_score": 1.0,
    "cascade_depth": 1,
    "orphaned_topics": 4,
    "impacted_subscribers": 3,
    "impact_score_std": 0.0
  },
  ...
]
```

**`records`** — full detail per node:

```json
{
  "RadarTracker": {
    "node_id": "RadarTracker",
    "node_type": "Application",
    "impact_score": 1.0,
    "total_orphaned_topics": 4,
    "total_impacted_subscribers": 3,
    "total_subscribers": 3,
    "cascade_depth": 1,
    "directly_orphaned_topics": ["T_radar", "T_tracks"],
    "all_orphaned_topics": ["T_conflicts", "T_fpa", "T_radar", "T_tracks"],
    "impacted_subscriber_ids": ["ATCWorkstation", "ConflictDetector", "FlightDataProcessor"],
    "per_subscriber_feed_loss": {
      "ATCWorkstation": 1.0,
      "ConflictDetector": 1.0,
      "FlightDataProcessor": 1.0
    },
    "cascade_waves": [
      {
        "wave_index": 0,
        "newly_orphaned_topics": ["T_radar", "T_tracks"],
        "newly_impacted_subscribers": ["ATCWorkstation", "ConflictDetector", "FlightDataProcessor"],
        "newly_failed_publishers": ["RadarTracker"]
      },
      {
        "wave_index": 1,
        "newly_orphaned_topics": ["T_conflicts", "T_fpa"],
        "newly_impacted_subscribers": [],
        "newly_failed_publishers": ["ConflictDetector", "FlightDataProcessor"]
      }
    ],
    "seed_impact_scores": {"42": 1.0, "123": 1.0, "456": 1.0, "789": 1.0, "2024": 1.0},
    "impact_score_std": 0.0
  }
}
```

**Key fields:**

| Field | Type | Description |
|-------|------|-------------|
| `impact_score` | float [0,1] | Mean I(v) across seeds. Primary validation target. |
| `impact_score_std` | float | Standard deviation across seeds. 0.0 = deterministic. |
| `cascade_depth` | int | Number of cascade waves that fired (0 = no cascade). |
| `directly_orphaned_topics` | list | Topics orphaned by removing v alone (wave 0). |
| `all_orphaned_topics` | list | All topics orphaned, including cascaded waves. |
| `per_subscriber_feed_loss` | dict | Per-subscriber feed-loss fraction (diagnostic; drives cascade propagation, not aggregated as I(v)). |
| `cascade_waves` | list | Full per-wave trace for debugging and visualisation. |

### 6.2 `message_flow_results.json`

Written by `message-flow`. Contains per-topic and per-subscriber statistics, plus a fault event record if a fault was injected.

```
output/simulation/
├── message_flow_results.json    ← full result
└── message_flow_summary.txt     ← human-readable table
```

**Top-level structure:**

```json
{
  "schema_version": "2.0",
  "graph_id": "atm_system",
  "simulation_duration": 300.0,
  "seed": 42,
  "fault_event": { ... },
  "system_delivery_rate": 0.9975,
  "system_drop_rate": 0.0025,
  "total_messages_published": 5820,
  "total_messages_delivered": 9730,
  "total_deadline_violations": 0,
  "total_queue_overflows": 0,
  "topic_stats": { ... },
  "subscriber_stats": { ... }
}
```

**`fault_event`** (null when no fault was injected):

```json
{
  "fault_time": 150.0,
  "faulted_node_id": "ConflictDetector",
  "faulted_node_type": "Application",
  "cascade_silenced_publishers": ["ConflictDetector"],
  "cascade_orphaned_topics": ["T_conflicts"],
  "cascade_impacted_subscribers": ["ATCWorkstation"],
  "delivery_rate_before": 0.9977,
  "delivery_rate_after": 0.9962,
  "latency_p50_before": 2.1,
  "latency_p50_after": 8.7,
  "latency_p95_before": 3.4,
  "latency_p95_after": 15.2
}
```

The four `latency_p*` fields hold system-wide end-to-end latency (ms) for messages delivered in each fault window. `null` is written when a window received no deliveries. The post-fault inflation `Δp50 = latency_p50_after − latency_p50_before` is the basis for the I_dyn(v) independent ground-truth signal consumed by `cli/validate_graph.py harness`.

**`topic_stats`** — per topic:

```json
{
  "T_radar": {
    "topic_id": "T_radar",
    "topic_name": "T_radar",
    "reliability_policy": "RELIABLE",
    "deadline_ms": 100,
    "durability_policy": "TRANSIENT_LOCAL",
    "total_published": 2990,
    "total_delivered": 5980,
    "total_dropped_queue_full": 0,
    "total_dropped_deadline": 0,
    "total_dropped_best_effort": 0,
    "delivery_rate": 1.0,
    "drop_rate": 0.0,
    "latency_p50_ms": 2.1,
    "latency_p95_ms": 3.4,
    "latency_p99_ms": 3.9
  }
}
```

**`subscriber_stats`** — per subscriber:

```json
{
  "ATCWorkstation": {
    "subscriber_id": "ATCWorkstation",
    "subscribed_topics": ["T_tracks", "T_conflicts", "T_fpa"],
    "received_per_topic": {"T_tracks": 1495, "T_conflicts": 148, "T_fpa": 599},
    "missed_per_topic": {"T_tracks": 0, "T_conflicts": 0, "T_fpa": 0},
    "deadline_violations_per_topic": {"T_tracks": 0, "T_conflicts": 0, "T_fpa": 0},
    "total_received": 2242,
    "total_missed": 0,
    "overall_delivery_rate": 1.0,
    "received_post_fault": 1050
  }
}
```

**Note.** `total_delivered` in `topic_stats` counts individual (message, subscriber) deliveries — i.e., for a topic with two subscribers, each message delivered to both counts as two deliveries. The per-topic `delivery_rate` is `total_delivered / (total_published × num_subscribers)`.

---

## 7. Worked Examples — ATM Dataset

The ATM Air Traffic Management dataset has the following pub-sub topology:

```
RadarTracker  ──PUBLISHES_TO──▶  T_radar   ──SUBSCRIBES_TO──▶  ConflictDetector
              ──PUBLISHES_TO──▶  T_tracks  ──SUBSCRIBES_TO──▶  ConflictDetector
                                            ──SUBSCRIBES_TO──▶  ATCWorkstation
                                            ──SUBSCRIBES_TO──▶  FlightDataProcessor

FlightDataProcessor ──PUBLISHES_TO──▶  T_fpa  ──SUBSCRIBES_TO──▶  ATCWorkstation

ConflictDetector ──PUBLISHES_TO──▶  T_conflicts ──SUBSCRIBES_TO──▶  ATCWorkstation

MeteoService ──PUBLISHES_TO──▶  T_meteo  (no subscribers)

ASTERIX_Broker ──ROUTES──▶  T_radar, T_tracks, T_conflicts, T_meteo, T_fpa
```

### 7.1 Expected fault-inject results

| Node | I(v) | Cascade depth | Why |
|------|------|---------------|-----|
| `RadarTracker` | 1.000 | 1 | Sole publisher of T_radar and T_tracks; ConflictDetector and FlightDataProcessor both lose all feeds → cascade → T_conflicts and T_fpa also orphaned; all 3 subscribers lose 100% of their feeds |
| `ASTERIX_Broker` | 1.000 | 1 | Sole router of all 5 topics; same total loss |
| `ConflictDetector` | 0.111 | 0 | Orphans only T_conflicts; ATCWorkstation loses 1/3 feeds (T_conflicts only); other subscribers unaffected |
| `FlightDataProcessor` | 0.111 | 0 | Orphans only T_fpa; ATCWorkstation loses 1/3 feeds |
| `ATCWorkstation` | 0.000 | 0 | Not a publisher; removing it harms no downstream subscriber |
| `MeteoService` | 0.000 | 0 | Orphans T_meteo but T_meteo has no subscribers |

> With `propagation_threshold=0.5`: ConflictDetector losing T_radar alone (1/2 feeds = 50%) would trigger a cascade to T_conflicts → ATCWorkstation also loses T_conflicts → ConflictDetector's I(v) rises.

### 7.2 Running the full validation workflow

```bash
# Step 1: Generate ground-truth I(v)
PYTHONPATH=. python cli/simulate_graph.py fault-inject \
    --input data/scenarios/atm_system.json \
    --output output/simulation/ \
    --seeds 42,123,456,789,2024 \
    --export-json

# Step 2: Run analysis to get Q(v) predictions
PYTHONPATH=. python cli/analyze_graph.py \
    --input data/scenarios/atm_system.json \
    --output output/analysis/ \
    --export-json

# Step 3: Validate Q(v) vs I(v) with methodological guards
# cli/validate_graph.py harness reads pre-computed prediction and impact JSON
PYTHONPATH=. python cli/validate_graph.py harness \
    --predictions output/analysis/predictions.json \
    --ground-truth cascade=output/simulation/impact_scores.json \
    --out output/harness_report.json
```

### 7.3 Message flow: observing the ConflictDetector fault

```bash
PYTHONPATH=. python cli/simulate_graph.py message-flow \
    --input data/scenarios/atm_system.json \
    --duration 300 \
    --fault-node ConflictDetector \
    --fault-time 150 \
    --seed 42 \
    --export-json
```

Expected observations in `message_flow_results.json`:
- `T_conflicts.delivery_rate` drops to ~0.5 (only pre-fault messages delivered).
- `ATCWorkstation.received_per_topic.T_conflicts` is ~150 messages (rate 1 Hz × 150 s).
- `ATCWorkstation.received_per_topic.T_tracks` and `.T_fpa` are unaffected (~full duration).
- `fault_event.delivery_rate_after` is lower than `delivery_rate_before` due to loss of T_conflicts stream.

---

## 8. Integration with the RM Validation Pipeline

The fault injector's `impact_scores.json` is designed to slot directly into the existing SaG validation pipeline:

```
impact_scores.json
    │
    │  records[node_id].impact_score  →  I(v) vector
    │
    ▼
ValidateGraphUseCase  (saag/usecases/validate_graph.py)
    │
    │  Spearman ρ(Q(v), I(v))   ← primary gate metric (threshold ρ ≥ 0.70)
    │  F1 @ top-k               ← secondary gate
    │  ICR@K, RCR, BCE          ← specialist metrics
    │  Predictive Gain (PG)     ← must exceed 0.03 over degree baseline
    │
    ▼
Validation report
```

**Pairing keys.** Both `analysis_results.json` and `impact_scores.json` use the node ID (string matching the graph node name) as the primary key. The validation script inner-joins on this key. Nodes present in only one file are dropped — but that drop must be **reported, not silent**: consult `unlabeled_node_ids` and the `n_predicted` / `n_labeled` / `n_evaluated` counts that `compute_inductive_metrics` now returns. A model scored on 65 of 98 nodes has not been evaluated on the other 33, and that is neither evidence for nor against it.

**Node-type stratified reporting.** The `node_type` field in each record allows the Spearman ρ to be computed separately per type. This matters more now that Libraries are labelled: overall ρ can be driven by *between-type* separation (Libraries score systematically high, Applications low) rather than by correct ranking *within* a type. Always read `per_type_rho` alongside the headline ρ — they can point in opposite directions.

**Multi-seed stability gate.** Before using I(v) for publication:

1. Confirm `label_stability.n_seeds` ≥ 2. With one seed the labels' reproducibility is unmeasured and ρ has no stated ceiling.
2. Read `label_stability.test_retest_spearman` as the ceiling on any reported ρ. A model at 0.93 against labels self-consistent at 0.93 has **saturated** the labels, not underperformed.
3. Read `label_stability.topk_jaccard` before quoting any top-K metric. Where it is ~0.6, Overlap@K and P@τ inherit ~40% churn from the labels themselves.
4. Check per-node `impact_score_std` for boundary fragility (suggested threshold: 0.02).

---

## 9. What the Simulator Measures, in Quality-Model Terms

Every impact quantity defined above — `I*(v)`, `composite_impact`, `IR/IM/IA/IS` — answers the question *"how much of the graph broke?"*. Criticality is defined on a different question: *"how much worse did the outcome get for stakeholders?"* — the **Quality-in-Use** axis of ISO/IEC 25019:2023 that [criticality.md](criticality.md#41-definition) D1 and D2 are written on. This section places what the simulator produces on the right axis, and asks how far it reaches toward the second question.

**The simulator measures external quality, not Quality-in-Use.** Delivery rate, latency percentiles and contract conformance are observations of the *behaviour of the executing system* — external product-quality measures in the sense of ISO/IEC 25023 — made here on a model of that system rather than on a deployment ([criticality.md §3.0](criticality.md#30-three-quality-views-internal-external-and-quality-in-use)). That is a substantive distinction, not a labelling one: the same 40% delivery loss is an inconvenience in one deployment and a hazard in another, and no quantity below separates those two cases. Quality-in-Use measurement in the standard's sense (ISO/IEC 25022) requires a specified stakeholder pursuing a specified goal in a specified context, none of which exists anywhere in this project.

**What is reachable is nonetheless more than it looks.** The discrete-event engine already records delivery, latency, and contract-violation data per fault; those observations are simply aggregated along the RM axis (to mirror the predictor) rather than re-summarised per characteristic. §9.1 audits, characteristic by characteristic, which of them the existing fields can speak to — with the standing caveat that every "measurable" verdict below means *measurable as external quality*, one view short of the construct.

### 9.1 Coverage by characteristic

Read "measurable" throughout as *measurable as external quality* — the corresponding Quality-in-Use characteristic is what it stands in for, never what is observed.

| Quality-in-Use characteristic | External quality attribute actually observed | Status | Existing outputs that measure it |
|:---|:---|:---|:---|
| **Effectiveness** — is the goal achievable at all? | Reliability → **availability**, **fault tolerance** | **Measurable now** | `FaultEventRecord.delivery_rate_before` / `.delivery_rate_after`, `.cascade_impacted_subscribers`, `.cascade_orphaned_topics`; `SubscriberFlowStats.missed_per_topic`, `.missed_post_fault`, `.overall_delivery_rate` ([`simulation_results.py`](../saag/simulation/simulation_results.py)); `ImpactMetrics.reachability_loss`, `.fragmentation`, `.flow_disruption` ([`models.py`](../saag/simulation/models.py)) |
| **Efficiency** — same goal, more resource? | Performance efficiency → **time behaviour**, **capacity** | **Measurable now** | `FaultEventRecord.latency_p50_before/after` and `.latency_p95_before/after` — already positioned as an independent `I_dyn(v)` oracle candidate ([§4.4](#44-fault-injection-at-runtime)); `MessageFlowResult.total_queue_overflows`; `ImpactMetrics.throughput_loss`; `RuntimeMetrics.avg_latency`, `.p99_latency`, `.throughput` |
| **Freedom from risk** — is a contract breached? | Reliability → **fault tolerance** (QoS contract conformance) | **Blocked by the corpus, not by the method** | The machinery exists: deadline and lifespan checks against end-to-end latency ([§4.3](#43-qos-enforcement)), `TopicFlowStats.total_dropped_deadline`, `SubscriberFlowStats.deadline_violations_per_topic`, `MessageFlowResult.total_deadline_violations`, and a `deadline=…:qos` oracle slot in the validation harness. But **no topic in the scenario corpus declares `deadline_ms` — 0 of 710** across all ten scenarios, so every counter is structurally zero |
| **Satisfaction** | *— none —* | **Not measurable** | Behavioural, and no correlate exists in a message-flow simulation. Repeat-outage frequency is the nearest proxy and is not the same construct |
| **Context coverage** | *(a property of the ranking, not an attribute)* | **Across runs, not per fault** | Not a per-fault quantity at all: it is the stability of the impact ranking across scenarios and domains, already exercised by the LOSO and multi-scenario batch runs |

**Two of five are available today; a third is one generator change away; one is permanently out of reach; one is a cross-run property rather than a per-fault measurement.** All of the available ones are external quality measurements standing in for the characteristic named beside them, which is what [criticality.md §7.1](criticality.md#71-the-validation-chain-has-three-links) counts as the measured link of a three-link chain.

### 9.2 The two constraints

**Cost.** `MessageFlowSimulator(graph, duration, fault_node, fault_time, seed, …)` injects **one fault per run** ([§11.2](#112-messageflowsimulator)). A per-component Effectiveness/Efficiency oracle therefore costs one discrete-event run per candidate component, which is materially more expensive than the graph-based sweep that produces `I*(v)` — the reason the cheap oracle remains the one wired into the training pipeline.

Measured, the cost is tolerable at validation scale but not at training scale: at `duration=60`, sweeping `atm_system` (20 publishers) takes well under a minute, `healthcare_system` (39) and `microservices_system` (92) complete in a few minutes, while `iot_smart_city_system` exceeds two. Use `--max-candidates` in [reproduce/convergent_validity.py](../reproduce/convergent_validity.py) on the larger scenarios. This bounds `I_dyn` to a periodically-reported validity check rather than a per-epoch label source.

**Unlocking Freedom from risk is not free.** Every topic in the corpus carries `frequency` (Hz) — 710 of 710 — so a deadline is derivable from the publication period. But emitting `deadline_ms` changes generated topology bytes: all `data/scenarios/*.json` would regenerate and the golden SHA-256 in `tests/test_generation_service.py` would need re-baselining, and simulation outputs produced before and after would not be comparable. That is a deliberate change to the Generate capability, not a side effect to slip into a simulation run.

### 9.3 `I_dyn(v)` — the effectiveness term, implemented

The effectiveness half of the sketch below is now built and reported as `I_dyn(v)`, the third oracle in [reproduce/convergent_validity.py](../reproduce/convergent_validity.py). The other two terms remain unavailable for the reasons given under each:

```
I_QiU(v) = ( effectiveness_loss , efficiency_loss [, risk_loss] )

effectiveness_loss = delivery_rate_before − delivery_rate_after   ← implemented as I_dyn(v)
efficiency_loss    = (latency_p95_after − latency_p95_before) / latency_p95_before
                     ← no signal at corpus load; see §12
risk_loss          = deadline_violations_after / messages_delivered_after   ← always 0 today
```

`I_dyn(v)` measures the loss suffered by the components that **survived** the fault. Both the faulted node's own receipts and its share of the fan-out are excluded from the before and after windows, and a silenced publisher's demand stays in the denominator. Neither is a detail: without them the score inverts, tracking how much a component consumed instead of how much depended on it (`ρ(I_dyn, I*) = −0.25` on `atm_system`, driven by `ρ = +0.75` against the faulted node's own subscription count). [tests/test_message_flow_oracle.py](../tests/test_message_flow_oracle.py) pins both properties.

Two properties matter for it to be usable:

- **It is a third quantity, not a replacement.** `I*(v)` (the Predict-stage labeler) and `composite_impact` (the Validate-stage oracle) are already distinct and must not be conflated ([§2.1](#21-which-engine-is-canonical-for-what)); `I_dyn` is a third named quantity under the same rule, leaving both existing meanings untouched. It is **not** a gate and does **not** produce training labels.
- **No new validation machinery is needed.** `cli/validate_graph.py harness` already accepts repeated `--ground-truth NAME=PATH[:qos]` sources and computes a convergent-validity block between every pair of them. Agreement between `I_dyn` and the cascade oracles is therefore a *reported result* rather than an assumption — which is exactly the check that [criticality.md §7.1](criticality.md#71-the-validation-chain-has-three-links) says is missing.

Why it is worth the cost: `I*(v)` and `composite_impact` are both topological cascade engines over the same substrate, so their agreement cannot rule out a shared construction artifact — the objection that a topology-derived `Q(v)` is being validated against topology-derived labels. `I_dyn` observes message delivery under load instead of reachability over edges, so its agreement with `I*` is evidence of a different kind. Measured across the seven-scenario cohort at `duration=60`: mean `ρ(I_dyn, I*)` = **0.765** with a **minimum of 0.548**, against mean `ρ(I*, I_comp)` = 0.394 with a minimum of 0.092. The lifted floor is the substantive part — `I_dyn`'s worst case (Hub-and-Spoke, 0.548) is far from uncorrelated, while the two topological oracles fall to near-independence on that same scenario. Full table and the limits of the claim: [validation.md §3.3](validation.md#33-the-behavioural-oracle-i_textdynv).

`I_dyn` is **delivery-based and QoS-agnostic** in this study. It does not measure latency or deadline conformance; see §12 for why.

---

## 10. Input Graph Format Requirements

The `--input` file must be a JSON file compatible with the SaG schema.
[`_load_graph`](../cli/simulate_graph.py) reads these keys directly from either the top level
of the JSON or a nested `"relationships"` object (to support exported schemas like the ATM
dataset):

| Key | Type | Description |
|-----|------|-------------|
| `applications` | list | Each item: `{"id": "...", "name": "...", "processing_time": 0.002, ...}` |
| `brokers` | list | Each item: `{"id": "...", "name": "..."}` |
| `topics` | list | Each item: `{"id": "...", "name": "...", "qos_profile": {...}}` |
| `nodes` | list | Infrastructure nodes (optional for simulation) |
| `publishes_to` | list | Each: `{"from": "...", "to": "...", "rate_hz": 10.0, ...}` (also supports `publishes`, `publish_edges`, `source`/`target`) |
| `subscribes_to` | list | Each: `{"from": "...", "to": "...", ...}` (also supports `subscribes`, `subscribe_edges`, `source`/`target`) |
| `routes` | list | Each: `{"from": "...", "to": "..."}` (also supports legacy `broker_routes` dictionary) |
| `runs_on` | list | Each: `{"from": "...", "to": "..."}` (Application/Broker mapping to Node) |

**QoS profile fields:**

```json
{
  "reliability":   "RELIABLE",
  "durability":    "TRANSIENT_LOCAL",
  "deadline_ms":   100,
  "lifespan_ms":   null,
  "queue_size":    50,
  "history_depth": 10
}
```

All QoS fields are optional; defaults are `RELIABLE`, `VOLATILE`, no deadline, no lifespan, `queue_size=100`.

**`processing_time`** on Application nodes (seconds). Used by the message-flow simulator as
the per-component compute latency. Read straight from the node attributes; when absent it
falls back to `MessageFlowSimulator(default_processing_time_s=…)`, which defaults to
`0.001 s` and is **not** exposed as a CLI flag.

> [!NOTE]
> A separate mechanism derives processing latency from static-analysis complexity:
> [`ComplexityProcessor`](../saag/simulation/processor.py) computes
> `pt(v) = base_latency × (1 + α·c_norm(v)) + β·Σ c_norm(lib)` and injects it as
> `processing_latency` on the component. It belongs to the `EventSimulator` stack, not to
> the message-flow one, and it activates only when components carry a flat `complexity`
> property — the scenario generator currently nests it under `code_metrics.complexity`, so
> in practice it is a no-op on the shipped corpus.

---

## 11. Python API

Both simulators can be used as Python libraries without going through the CLI.

### 11.1 FaultInjector

```python
from saag.simulation.fault_injector import FaultInjector
import networkx as nx

# graph: NetworkX DiGraph with PUBLISHES_TO, SUBSCRIBES_TO, ROUTES edges
injector = FaultInjector(
    graph=graph,
    seeds=[42, 123, 456, 789, 2024],
    cascade_depth_limit=0,          # 0 = unlimited
    propagation_threshold=0.2,      # default 0.2
)

# Inject the three labelable types. Topic and Node would score 0 everywhere (§12 L6).
result = injector.run(node_types=["Application", "Broker", "Library"])

# Inject specific nodes only
result = injector.run(node_ids=["ConflictDetector", "ASTERIX_Broker"])

# Save to disk
from pathlib import Path
result.save(Path("output/simulation/impact_scores.json"))

# Access per-node records
for node_id, rec in result.records.items():
    print(f"{node_id}: I(v)={rec.impact_score:.4f}  depth={rec.cascade_depth}")

# Access ranked summary
for row in result.top_k_by_impact:
    print(f"#{row['rank']}  {row['node_id']}  {row['impact_score']:.4f}")

# Check the labels before trusting anything computed against them
stab = result.label_stability
print(f"ceiling on any reported rho: {stab['test_retest_spearman']}")
print(f"top-K critical set stability: {stab['topk_jaccard']}")

# Coverage: which nodes have no ground truth at all
print(f"{len(result.unlabeled_node_ids)} nodes unlabeled: {result.unlabeled_node_ids[:5]}")
```

> [!NOTE]
> Passing a graph whose Library nodes lack a `type` attribute silently excludes them from
> `node_types` matching. If you build the graph yourself rather than via
> `cli/simulate_graph.py::_load_graph`, set `type="Library"` explicitly — implicit creation
> through `USES` edges leaves the attribute unset.

### 11.2 MessageFlowSimulator

```python
from saag.simulation.message_flow_simulator import MessageFlowSimulator

sim = MessageFlowSimulator(
    graph=graph,
    duration=300.0,
    fault_node="ConflictDetector",  # None for baseline (no fault)
    fault_time=150.0,               # defaults to duration / 2
    seed=42,
    default_queue_size=100,
    default_publish_rate_hz=10.0,
    default_processing_time_s=0.001,
    max_latency_samples=10_000,
)

result = sim.run()
result.save(Path("output/simulation/message_flow_results.json"))

# Inspect per-topic stats
for tid, ts in result.topic_stats.items():
    print(f"{ts.topic_name}: delivery={ts.delivery_rate:.4f}  "
          f"P50={ts.latency_p50:.1f}ms  deadline_viol={ts.total_dropped_deadline}")

# Inspect fault event
if result.fault_event:
    fe = result.fault_event
    print(f"Fault at t={fe.fault_time:.1f}s: {fe.faulted_node_id}")
    print(f"  Orphaned:  {fe.cascade_orphaned_topics}")
    print(f"  Impacted:  {fe.cascade_impacted_subscribers}")
    print(f"  Rate before: {fe.delivery_rate_before:.4f}")
    print(f"  Rate after:  {fe.delivery_rate_after:.4f}")
    # Latency windowing (I_dyn(v) source — may be None if a window had no deliveries)
    if fe.latency_p50_before is not None and fe.latency_p50_after is not None:
        delta_p50 = fe.latency_p50_after - fe.latency_p50_before
        print(f"  Δp50 latency: {delta_p50:+.1f} ms  "
              f"(before={fe.latency_p50_before:.1f}, after={fe.latency_p50_after:.1f})")
```

---

## 12. Known Limitations

**L1 — Broker routing model is binary.** The fault injector models broker failure as "topic routed by the failed broker is orphaned if no other live broker routes it." In practice, DDS routing is more nuanced — a broker failure mid-message can cause partial delivery even with redundant routing. The current model is conservative and correct for single-broker topologies (ADVENT, ATM datasets).

**L2 — TRANSIENT\_LOCAL durability not fully simulated.** The message flow simulator notes the `TRANSIENT_LOCAL` QoS policy but does not implement late-joiner history replay. A subscriber that joins after the publisher starts will not receive historical samples. This affects correctness for simulations modelling late-joining controllers.

**L3 — Single fault per simulation run.** Both simulators model at most one node failure at a time. Correlated failures (e.g., a power loss taking down all nodes in a rack) require running the combined mode with explicit topology modifications or extending the fault injector with a `fault_group` parameter.

**L4 — Publisher-side processing time is not included in end-to-end latency.** The message `created_at` timestamp is set after the publisher's processing delay, meaning publisher processing is not part of the reported latency. Total pipeline latency = publisher processing + queue transit + subscriber processing; only the latter two are captured. This is consistent with DDS measurement conventions (publication timestamp is at the point of writing to the middleware).

**L5 — Infrastructure layer metrics not used in cascade.** RUNS\_ON and CONNECTS\_TO edges are not used in the fault cascade. A network partition that isolates a set of physical nodes from each other is not modelled. This is consistent with the known weak correlation of infrastructure-layer Q(v) (ρ ≈ 0.54) and is flagged as a gap in the thesis.

**L6 — Topic and Node cannot be labelled.** Following from L5, `FaultInjector` derives `DEPENDS_ON` only from `PUBLISHES_TO`, `SUBSCRIBES_TO` and `USES`. It has no rule that expresses the failure of a Topic (a topic is orphaned *by* a publisher or broker failing, never injected directly) or of a physical Node (no `RUNS_ON → DEPENDS_ON` derivation exists). Injecting either yields $I(v) = 0$ for **every** instance. These types are therefore excluded from `--node-types` and recorded in `unlabeled_node_ids`, leaving 33–160 nodes per scenario without ground truth (≈ 30–47% of components). The GNN still *predicts* scores for them; those predictions are simply never validated. Closing this requires adding the missing derivation rules to the cascade, not merely widening `--node-types`.

**L7 — Only two of three label dimensions are measured.** `FaultInjector` emits a single scalar, so `maintainability` has no ground truth from this engine (`composite` and `reliability` are covered). It is declared absent via `labeled_dimensions` and excluded from the loss via `dimension_mask` (§6.1). The two-dimensional RM decomposition (plus the fault_tolerance/availability sub-characteristic diagnostics) exists only in `FailureSimulator`, which serves the Validate stage (§2.1). Unifying them would require one engine to also produce a maintainability ground truth.

**L8 — Edge labels are measured, with three caveats.** `EdgeCriticality` is populated by
`FailureSimulator.simulate_edge_removal`, swept by `simulate_edge_removal_sweep`, which severs one
relationship — leaving both endpoints active — and recomputes the same reachability / fragmentation
/ throughput / flow quantities that back `ImpactMetrics`.

Three properties of the implementation matter when reading the numbers:

- **Deltas, not levels.** `_calculate_impact` is *not* zero on a pristine graph: topics that already
  lack a publisher or a subscriber are counted as lost throughput regardless of what failed
  (composite 0.0061 on `av_system`). Every edge quantity is differenced against that null
  observation, so an edge that costs nothing measures as exactly zero instead of inheriting the floor.
- **Bounded candidate set.** Sweeping every edge costs one full impact recomputation per edge, so the
  default candidate set is `bridges(G) ∪ top-q edge-betweenness`. Edges outside it carry
  `evaluated: false` — *not measured* is distinct from *measured as harmless* and must not be read as
  a zero.
- **No cascade.** Only the edge is removed. The question an edge label answers is "what does this link
  carry", not "what else fails afterwards"; cascade effects belong to the node labels.

Measured consequence worth reporting: on `av_system`, of 40 candidates only 4 carry non-zero impact,
and all four are `PUBLISHES_TO`/`SUBSCRIBES_TO`. `RUNS_ON` edges measure exactly 0.0 because this
cascade model routes no traffic over them (L5) — the bridge-based candidate selection surfaces them
as structurally non-redundant while the simulation correctly scores them as carrying nothing. The
prior heuristic labels (`I*(u) × {1.0, 0.1}`) would have assigned them their source node's full blast
radius.

**L9 — Broker labels are topology-dependent and frequently degenerate.** When computing topic feed loss, the cascade uses routing-broker failure as the loss fraction *only* when the topic has no publishers at all; otherwise loss comes from publisher rates and the routing brokers are ignored entirely. Combined with the redundancy rule (a topic is orphaned only if *all* its routing brokers fail), this means a Broker scores $I(v) = 0$ whenever every topic it routes either has a live publisher or has a redundant router.

This is not a corner case. Across the eight regenerated LOSO caches:

| Broker labels | Scenarios |
|---------------|-----------|
| **All zero** — no signal at all | `enterprise_system` (10 brokers), `financial_trading_system` (5), `healthcare_system` (3) |
| Partial (some zero) | `atm_system` (3/5 non-zero), `av_system` (2/4), `iot_smart_city_system` (4/6, max 0.029) |
| Full signal | `hub_and_spoke_system` (2/2, mean 0.897), `microservices_system` (6/6, mean 0.497) |

**Broker labels are therefore usable in some scenarios and absent in others**, and the same
graph can flip between the two depending on how redundantly it is routed — the cohort caches
carry slightly denser `ROUTES` sets than the raw `data/scenarios/*.json` files, and
`healthcare_system` has non-zero broker labels in the latter (max 0.801) but all-zero in the
former. `FaultInjector` emits a `DEGENERATE LABELS` warning naming the affected type, so this
is visible per run rather than silent; treat it as a signal to exclude `Broker` from
`--node-types` for that scenario, or to read per-type ρ with `Broker` excluded. Fixing it
properly means making broker failure contribute to feed loss even for topics that have live
publishers.

**L10 — No timeout / retry modelling.** For RELIABLE QoS, the head-drop policy prevents queue overflow but does not model TCP-style retransmission or DDS heartbeat/acknowledgement. The modelled delivery rates will be optimistic relative to real network conditions.

**L11 — `TrafficSimulator` bypasses the repository port.** It is the only module in
`saag/simulation/` that reaches into `repo.driver.session(...)` and hand-writes Cypher, so it
is coupled to a live Neo4j deployment and is untestable without one — it currently has no test
coverage at all. It is a closed-form bandwidth/message-rate calculator rather than a
simulator, and nothing in the labelling or validation path depends on it; only
[`api/routers/traffic.py`](../api/routers/traffic.py) does. Bringing it behind
`IGraphRepository` is an architectural change, not a cleanup.

**L12 — `FaultInjector` labels depend on `PYTHONHASHSEED`.** Per-component seeds are derived
with `zlib.crc32` precisely so that labels do not vary with Python's salted string hashing
([`failure_simulator.py`](../saag/simulation/failure_simulator.py) `_derive_seed`), but the
cascade still iterates `idx.all_subscribers` — a `set` — while drawing random numbers inside
the loop. Set iteration order therefore permutes which subscriber consumes which draw.
Measured on `atm_system` across two runs in the same environment: `mean_std` 0.0263 → 0.0295,
`test_retest_spearman` 0.918 → 0.887, individual `impact_score` values shifting in the third
decimal.

> Running with `PYTHONHASHSEED=0` reproduces labels exactly. The one-line fix is to sort that
> iteration, but doing so changes every published label value, so it is deliberately **not**
> applied here — it is a re-baselining decision for the experiment owner, not a refactor.

**L13 — Topic QoS never resolves on any scenario in the corpus.** `_extract_qos`
([`message_flow_simulator.py`](../saag/simulation/message_flow_simulator.py)) reads `qos_profile`
or `qos_policy` from the node's attributes, but the topology JSON states topic QoS under `qos`
(`data/scenarios/atm_system.json`), and no file under `data/` uses either key that the simulator
looks for. **Every** message-flow run in this corpus therefore resolves to the defaults:
`RELIABLE`, `queue_size=100`, `deadline_ms=None`, `lifespan_ms=None`. The consequence is that
`total_dropped_deadline`, `total_dropped_best_effort` and the lifespan path are structurally zero
rather than measured-as-zero, and §4.3's QoS enforcement is untested against real inputs.

Edge-level QoS does not compensate: `_project_topic_qos_onto_edges` writes `qos_profile` onto the
pub/sub edges, but queues are constructed from *topic*-level QoS
(`TopicFanout.register`), so edge `reliability` and `queue_size` reach nothing, and
`QoSPolicy.to_dict()` emits only `durability`/`reliability`/`transport_priority` — never a
deadline. `I_dyn(v)` is consequently a **delivery-based, QoS-agnostic** measure, and should be
described as such rather than as a QoS-aware runtime oracle.

**L14 — Latency carries no signal at corpus load.** At the corpus's publication rates (~1 Hz
typical) against `default_processing_time_s = 0.001`, utilisation is far below saturation and
queues never build. Measured on `atm_system`, `latency_p95` sits at ≈1.2 ms both before and after
the fault for every faulted node, varying by less than the run-to-run jitter. `latency_p50_before/after`
and `latency_p95_before/after` on `FaultEventRecord` are therefore **not usable as an impact
dimension** — the `efficiency_loss` term in §9.3 would be dividing noise by noise. Making them
meaningful needs a load multiplier that drives utilisation near saturation, which is an
experimental-design change, not a bug fix.

---

## 13. Resolved Issues

Behaviours that earlier revisions of this document described differently. Recorded so that
artifacts produced before each fix can be interpreted correctly.

| # | Issue | Resolution |
|---|---|---|
| **R1** | Topic QoS was read with flat keys only (`qos_reliability` / `qos_priority`), but the canonical property is `qos_transport_priority` and the research loader leaves QoS nested. No key matched, so I\*(v) was numerically independent of QoS — flipping every `atm_system` topic from `PERSISTENT/RELIABLE/CRITICAL` to `VOLATILE/BEST_EFFORT/LOW` moved all 39 labels by `0.000000`. | Resolved through `QoSPolicy.from_node_attrs`, which accepts both shapes. **Any artifact generated before this is a `--qos-factor none` label** whatever its provenance block says. |
| **R2** | §3.5 stated that a Library injection yields I(v) = 0 because DEPENDS_ON propagation is disabled at `prob = 0.0`. That holds only for `app_to_app`; `app_to_lib` is special-cased to `prob = 1.0`. | Libraries were absent for two unrelated reasons, both fixed: not in the default `--node-types`, and the CLI loader had no `libraries` block, so Library nodes were created implicitly by their `USES` edges with `type=None` and matched no filter. |
| **R3** | `extract_simulation_dict` emitted `"maintainability": 0.0` and `"security": 0.0` for every record; the fabricated zeros were indistinguishable from measurements, so two prediction heads trained against a constant. (`security` no longer exists as a label dimension at all; `maintainability` remains unmeasured by this engine.) | The parser emits only declared dimensions; `dimension_mask` and `label_mask` carry absence explicitly ([§6.1](#61-impact_scoresjson)). |
| **R4** | `SimulationService.classify_edges()` always returned `[]`; edge labels were a projection of node labels through a hand-picked bridge multiplier. | Replaced by real edge-removal measurement ([§12 L8](#12-known-limitations)), pinned by [`tests/test_edge_removal.py`](../tests/test_edge_removal.py). |
| **R5** | `EventType` was missing the `FAIL_COMPONENT` / `RECOVER_COMPONENT` members that `EventSimulator` dispatches on, so any `EventScenario(failure_rate > 0)` raised `AttributeError`. It went unnoticed because the test file carried its own private copy of the simulator that *did* define them. | Members added; the test file now exercises the real `saag.simulation` classes. |
| **R6** | A brokered topic whose brokers had all failed fell through to the brokerless (DDS) direct-delivery path, so a broker outage silently repaired itself and produced zero drops. | `SimulationGraph.has_configured_brokers` distinguishes *no brokers configured* from *all brokers failed*; only the former delivers directly. |

---

## 14. What Comes Next

The labels this step produces are consumed in two directions:

- **[Step 3: Predict](prediction.md)** trains against `impact_scores.json` — read from disk,
  never by importing this package (the separation is enforced by
  [`tests/test_predict_simulate_separation.py`](../tests/test_predict_simulate_separation.py)).
- **[Step 5: Validate](validation.md)** correlates the predicted Q(v) against I(v) and gates
  on Spearman ρ. Before trusting any reported ρ, check `label_stability`
  ([§3.6](#36-multi-seed-stability-label-noise-and-reproducibility)) — it is the ceiling on
  what any correlation against these labels can mean.

← [Step 3: Predict](prediction.md) | → [Step 5: Validate](validation.md)