# Failure Simulation

This document describes the two simulation **CLI modes** available in `simulate_graph.py` and the Python modules that back them: `saag/simulation/fault_injector.py` and `saag/simulation/message_flow_simulator.py`.

> [!NOTE]
> **Scope of this document.** `simulate_graph.py` exposes a two-subcommand CLI surface (`fault-inject` and `message-flow`) aimed at pre-deployment ground-truth collection and message-timing analysis. The full `saag/simulation/` package additionally contains `ChangePropagationSimulator` (produces IM(v)) and `CompromisePropagationSimulator` (produces IV(v)), which are invoked by `saag.simulation.SimulationService` in EXHAUSTIVE / MONTE_CARLO / PAIRWISE modes. The README Step 4 row and SRS REQ-FS-03 refer to these **four simulation engines** in aggregate; this document covers only the two that are exposed through `simulate_graph.py`. See `ARCHITECTURE.md §simulation/` for the full engine inventory.

---

## Contents

1. [Motivation and Design Rationale](#1-motivation-and-design-rationale)
2. [Architecture Overview](#2-architecture-overview)
   - [Which engine is canonical for what](#21-which-engine-is-canonical-for-what)
3. [Mode 1 — Fault Injection](#3-mode-1--fault-injection)
   - [Algorithm](#31-algorithm)
   - [I(v) Formula](#32-iv-formula)
   - [Cascade Propagation](#33-cascade-propagation)
   - [Broker Failure Semantics](#34-broker-failure-semantics)
   - [Library Blast-Radius Asymmetry](#35-library-blast-radius-asymmetry)
   - [Multi-Seed Stability, Label Noise, and Reproducibility](#36-multi-seed-stability-label-noise-and-reproducibility)
4. [Mode 2 — Message Flow Simulation](#4-mode-2--message-flow-simulation)
   - [Discrete-Event Model](#41-discrete-event-model)
   - [Fan-Out Queue Architecture](#42-fan-out-queue-architecture)
   - [QoS Enforcement](#43-qos-enforcement)
   - [Fault Injection at Runtime](#44-fault-injection-at-runtime)
5. [CLI Reference — simulate\_graph.py](#5-cli-reference--simulate_graphpy)
   - [fault-inject](#51-fault-inject)
   - [message-flow](#52-message-flow)
   - [combined](#53-combined)
   - [Shared Flags](#54-shared-flags)
6. [Output Files](#6-output-files)
   - [impact\_scores.json](#61-impact_scoresjson)
   - [message\_flow\_results.json](#62-message_flow_resultsjson)
7. [Worked Examples — ATM Dataset](#7-worked-examples--atm-dataset)
8. [Integration with the RMAV Validation Pipeline](#8-integration-with-the-rmav-validation-pipeline)
9. [Input Graph Format Requirements](#9-input-graph-format-requirements)
10. [Python API](#10-python-api)
11. [Known Limitations](#11-known-limitations)

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
simulate_graph.py  (CLI entry point)
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
    (runs fault-inject then message-flow in sequence)

saag/simulation/  (core simulation engine modules)
├── fault_injector.py        (FaultInjector: canonical Predict-stage labeler, I*(v))
├── failure_simulator.py      (FailureSimulator: canonical Validate-stage RMAV oracle)
├── message_flow_simulator.py (MessageFlowSimulator: discrete-event timing simulator)
└── simulation_results.py    (shared dataclasses for all modes)
    ├── FaultInjectionResult / FaultInjectionRecord / CascadeWave
    └── MessageFlowResult / TopicFlowStats / SubscriberFlowStats / FaultEventRecord
```

### 2.1 Which engine is canonical for what

`FaultInjector` and `FailureSimulator` both emit something called "impact", and the two
quantities are **not** interchangeable — see the warning in
[`saag/simulation/models.py`](../saag/simulation/models.py). Each owns one pipeline stage:

| Stage | Engine | Output | Consumed by |
|-------|--------|--------|-------------|
| **Predict** (supervised labels) | `FaultInjector` | `impact_scores.json` → `I*(v)`, a scalar | GNN training, k-fold / LOSO evaluation |
| **Validate** (quality oracle) | `FailureSimulator` | `ImpactMetrics` → composite + IR/IM/IA/IS | `saag/validation/` gates |

`FaultInjector` is the labeler because it is deterministic, multi-seed, and records
per-node variance — the properties a training label needs. `FailureSimulator` supplies
the four-dimensional RMAV decomposition that the validation gates are written against.

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

##### Phase A: Direct DEPENDS_ON Propagation (Stochastic)
For each node $u$ in the current wave's frontier:
1. Find all incoming edges $(v_{dep}, u)$ in the graph representing $v_{dep} \xrightarrow{\text{DEPENDS\_ON}} u$ (meaning $v_{dep}$ depends on $u$).
2. If $v_{dep}$ is not already failed, it fails stochastically with probability:
   $$P_{\text{dep}}(v_{dep}) = \text{prob} \times \text{depth\_damp}$$
   Where:
   - `prob` is currently set to `0.0` in the codebase (meaning stochastic propagation through pure dependency edges is disabled by default).
   - $\text{depth\_damp} = \max(0.25, 1.0 - \text{wave\_idx} \times 0.15)$ is a wave-depth damping factor.

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
     where $\text{QoS\_factor}(t)$ is:
     - Starts at `1.0`.
     - Multiplied by `1.2` if `qos_reliability` is `"RELIABLE"`.
     - Multiplied by `1.15` if `qos_priority` is `"HIGH"`, `"CRITICAL"`, or `"URGENT"`.
     - Multiplied by `1.05` if `qos_priority` is `"MEDIUM"`.

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
   Where:
   - **reachability\_loss**: fraction of weighted pub-sub paths (publisher → topic → subscriber) that are broken.
   - **fragmentation**: graph partition severity after removing $v$ (weighted connected-component disruption).
   - **throughput\_loss**: fraction of total topic-weight throughput disrupted.
   - **flow\_disruption**: fraction of complete Pub→Topic→Sub flow triples broken.
   
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

> [!IMPORTANT]
> **Corrects an earlier claim.** This section previously stated that a Library injection
> yields $I(v) = 0$ because `DEPENDS_ON` propagation is disabled at `prob = 0.0`. That is
> true only for `app_to_app` dependency edges; `app_to_lib` is explicitly special-cased to
> `prob = 1.0`. Libraries were absent from results for two unrelated reasons, both now
> fixed: they were not in the default `--node-types`, and the CLI's fallback graph loader
> had no `libraries` block, so Library nodes were created implicitly by their `USES` edges
> with `type=None` and matched no type filter at all.

**$T_0$ Step-Function Collapse in FailureSimulator**: `FailureSimulator` models library failure as a **$T_0$ step-function collapse**: all consuming Applications that use the Library fail immediately at depth 0. The subsequent propagation of these Application failures forward through the pub-sub topic graph is more restricted than in `FaultInjector`, so the two engines rank libraries differently. That divergence is expected — they measure different quantities (§2.1).

> [!NOTE]
> The standard Reliability $R(v)$ formula (documented in [structural-analysis.md](structural-analysis.md#reliability-rv--fault-propagation-risk)) already includes the normalized in-degree term $DG\_in(v)$, which captures the number of direct consumers (blast radius) for both Applications and Libraries. This is the correct place to tune the Library's structural influence if the asymmetry is considered too large.

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

> **Latency windowing.** The subscriber process also buckets each delivered-message end-to-end latency into a shared `latency_windows` dict (`"pre"` / `"post"` keys) keyed on whether `arrival_time < fault_time`. After `env.run()`, the four summary percentiles (`latency_p50_before`, `latency_p50_after`, `latency_p95_before`, `latency_p95_after`) are aggregated via a linear-interpolated percentile helper and written to `FaultEventRecord`. These fields are `None` when no fault was injected or when a window received no deliveries. Their primary use is as an independent I_dyn(v) ground-truth candidate for convergent validity (see [validation.md §10](validation.md#10-cli-reference)).

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

**Durability** (`TRANSIENT_LOCAL`) is noted in the QoS profile but is not fully modelled in the current simulator (no late-joiner history replay). This is documented in [Known Limitations](#11-known-limitations).

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

### 5.1 `fault-inject`

Runs BFS cascade fault injection and produces `impact_scores.json`.

```
python simulate_graph.py fault-inject [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | *(required)* | Path to the graph JSON file. |
| `--output DIR` | `output/simulation/` | Output directory; created if absent. |
| `--export-json` | off | Write `impact_scores.json` and `impact_scores_summary.txt` to `--output`. |
| `--nodes ID1,ID2,...` | all matching `--node-types` | Comma-separated node IDs to inject. Overrides `--node-types`. |
| `--node-types TYPE1,TYPE2` | `Application,Broker,Library` | Node types eligible for injection. Types omitted here get **no** ground truth and are listed in the artifact's `unlabeled_node_ids`. |
| `--seeds 42,123,...` | `42,123,456,789,2024` | Comma-separated integer seeds. Labels are the per-node mean; ≥ 2 seeds are required for `label_stability` to be measurable. |
| `--cascade-depth N` | `0` (unlimited) | Maximum cascade wave depth. |
| `--verbose` / `-v` | off | Enable DEBUG logging. |

> **Propagation threshold** is currently only configurable via the Python API (`FaultInjector(propagation_threshold=0.5)`), not the CLI. This is intentional — it is a research parameter that should be set deliberately, not accidentally via a flag.

> [!WARNING]
> **Do not add `Topic` or `Node` to `--node-types`.** The cascade derives `DEPENDS_ON` only
> from `PUBLISHES_TO`, `SUBSCRIBES_TO` and `USES`, so it has no way to express the failure
> of a Topic or a physical Node: **every** instance of either scores exactly $I(v) = 0$.
> Those are not measurements of "no impact" — they are the absence of a model. Including
> them adds a block of 25–45 constant-zero labels per scenario (see §11 L6) and trains the
> model toward a constant.
>
> `FaultInjector.run()` detects this and emits a `DEGENERATE LABELS` warning naming any node
> type whose entire label set came out zero. If you see that warning, remove the type.

**Example — full ATM dataset, five seeds:**

```bash
python simulate_graph.py fault-inject \
    --input data/atm_system.json \
    --output output/simulation/ \
    --seeds 42,123,456,789,2024 \
    --export-json
```

**Example — single node, unlimited cascade:**

```bash
python simulate_graph.py fault-inject \
    --input data/atm_system.json \
    --nodes ConflictDetector \
    --output output/simulation/ \
    --export-json -v
```

**Example — brokers only, max two cascade waves:**

```bash
python simulate_graph.py fault-inject \
    --input data/atm_system.json \
    --node-types Broker \
    --cascade-depth 2 \
    --seeds 42,123,456 \
    --export-json
```

### 5.2 `message-flow`

Runs the SimPy discrete-event message flow simulation.

```
python simulate_graph.py message-flow [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | *(required)* | Path to the graph JSON file. |
| `--output DIR` | `output/simulation/` | Output directory; created if absent. |
| `--export-json` | off | Write `message_flow_results.json` and `message_flow_summary.txt`. |
| `--duration SECONDS` | `100.0` | Simulation duration in simulated seconds. |
| `--fault-node NODE_ID` | none | Node to fault during simulation. |
| `--fault-time SECONDS` | `duration / 2` | When to inject the fault. |
| `--seed INT` | `42` | Random seed for publish jitter and processing time variation. |
| `--default-rate HZ` | `10.0` | Fallback publish rate when absent from graph metadata. |
| `--default-queue-size N` | `100` | Fallback per-subscriber queue capacity. |
| `--verbose` / `-v` | off | Enable DEBUG logging. |

> `--fault-node` and `--fault-time` are independent: `--fault-node` controls **which** node is faulted; `--fault-time` controls **when**. Omitting `--fault-node` runs a clean baseline simulation with no fault.

**Example — baseline, no fault:**

```bash
python simulate_graph.py message-flow \
    --input data/atm_system.json \
    --duration 300 \
    --seed 42 \
    --export-json
```

**Example — fault ConflictDetector at the midpoint:**

```bash
python simulate_graph.py message-flow \
    --input data/atm_system.json \
    --duration 300 \
    --fault-node ConflictDetector \
    --fault-time 150 \
    --seed 42 \
    --export-json
```

**Example — broker fault with custom queue size and rate:**

```bash
python simulate_graph.py message-flow \
    --input data/atm_system.json \
    --duration 200 \
    --fault-node ASTERIX_Broker \
    --fault-time 100 \
    --default-queue-size 50 \
    --default-rate 20.0 \
    --seed 123 \
    --export-json
```

### 5.3 `combined`

Runs `fault-inject` and `message-flow` in sequence using a merged set of flags.

```
python simulate_graph.py combined [options]
```

All flags from both `fault-inject` and `message-flow` are available. The `--fault-node` flag serves both modes: it selects the node for the message-flow fault injection **and** can be combined with `--nodes` to restrict fault-inject to the same node.

**Example — full combined run for ATM:**

```bash
python simulate_graph.py combined \
    --input data/atm_system.json \
    --output output/simulation/ \
    --seeds 42,123,456,789,2024 \
    --duration 300 \
    --fault-node ASTERIX_Broker \
    --fault-time 150 \
    --export-json
```

### 5.4 Shared Flags

All three subcommands accept these flags:

| Flag | Description |
|------|-------------|
| `--input PATH` | Path to the input graph JSON file. Required. |
| `--output DIR` | Output directory. Default: `output/simulation/`. |
| `--export-json` | Write JSON result files to `--output`. |
| `--verbose` / `-v` | Enable DEBUG-level logging (shows per-node I(v) and per-topic stats). |

---

## 6. Output Files

### 6.1 `impact_scores.json`

Written by `fault-inject`. This is the canonical I(v) ground-truth file consumed by the RMAV validation pipeline.

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
  "labeled_dimensions": ["composite", "reliability", "availability"],
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
| `labeled_dimensions` | The label dimensions this engine genuinely **measured**. `FaultInjector` emits one scalar, which maps onto `composite` / `reliability` / `availability` — it says nothing about maintainability or security, so those are **absent**, not zero. |
| `unlabeled_node_ids` | Nodes present in the graph but never injected. Makes the coverage gap explicit instead of letting it vanish in a downstream set intersection. |
| `label_stability` | The labels' own reproducibility — the ceiling on any ρ reported against them. See §3.6. |

> [!IMPORTANT]
> **Absent is not zero.** `extract_simulation_dict` previously emitted
> `"maintainability": 0.0` and `"security": 0.0` for every record. Those fabricated zeros
> were indistinguishable from real measurements, so two of the model's five prediction heads
> were trained and scored against a constant. The parser now emits only the dimensions the
> labeler declared, and `networkx_to_hetero_data` derives a `dimension_mask` from them so the
> unmeasured columns are excluded from the multitask loss.
>
> The same distinction applies to **nodes**: a node the simulator targeted and scored `0.0`
> is a real observation at the low end of the ranking, while a node that was never targeted
> is missing data. Both used to collapse to a `0.0` label, and the training code identified
> labelled nodes with the proxy `|y_composite| > 1e-6` — which excluded genuine zeros from
> the loss while still scoring the model on them (7–115 nodes per scenario; 37% on
> `enterprise_system`). An explicit `label_mask` now carries presence-in-the-artifact.

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
python simulate_graph.py fault-inject \
    --input data/atm_system.json \
    --output output/simulation/ \
    --seeds 42,123,456,789,2024 \
    --export-json

# Step 2: Run analysis to get Q(v) predictions
python analyze_graph.py \
    --input data/atm_system.json \
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
python simulate_graph.py message-flow \
    --input data/atm_system.json \
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

## 8. Integration with the RMAV Validation Pipeline

The fault injector's `impact_scores.json` is designed to slot directly into the existing SaG validation pipeline:

```
impact_scores.json
    │
    │  records[node_id].impact_score  →  I(v) vector
    │
    ▼
ValidateUseCase / validate_topology_classes.py
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

## 9. Input Graph Format Requirements

The `--input` file must be a JSON file compatible with the SaG schema. The CLI loader handles two paths automatically:

**Path 1 (preferred):** If `saag/core/graph_builder.py` and `saag/core/graph_exporter.py` are importable, they are used. This supports the full schema including MIL-STD-498 hierarchy metadata, Jira enrichment, and code metrics.

**Path 2 (fallback):** A lightweight inline loader reads these keys directly from either the top-level of the JSON or a nested `"relationships"` object (to support exported schemas like the ATM dataset):

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

**`processing_time`** on Application nodes (seconds). Used by the message-flow simulator as the per-component compute latency. Set by `ProcessingTimeEnricher` as `base_latency × (1 + α × c_norm(v))` where `c_norm(v)` is the normalised cyclomatic complexity from SonarQube. Falls back to `--default-processing-time` (default 0.001 s) when absent.

---

## 10. Python API

Both simulators can be used as Python libraries without going through the CLI.

### 10.1 FaultInjector

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

# Inject the three labelable types. Topic and Node would score 0 everywhere (§11 L6).
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

### 10.2 MessageFlowSimulator

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

## 11. Known Limitations

**L1 — Broker routing model is binary.** The fault injector models broker failure as "topic routed by the failed broker is orphaned if no other live broker routes it." In practice, DDS routing is more nuanced — a broker failure mid-message can cause partial delivery even with redundant routing. The current model is conservative and correct for single-broker topologies (ADVENT, ATM datasets).

**L2 — TRANSIENT\_LOCAL durability not fully simulated.** The message flow simulator notes the `TRANSIENT_LOCAL` QoS policy but does not implement late-joiner history replay. A subscriber that joins after the publisher starts will not receive historical samples. This affects correctness for simulations modelling late-joining controllers.

**L3 — Single fault per simulation run.** Both simulators model at most one node failure at a time. Correlated failures (e.g., a power loss taking down all nodes in a rack) require running the combined mode with explicit topology modifications or extending the fault injector with a `fault_group` parameter.

**L4 — Publisher-side processing time is not included in end-to-end latency.** The message `created_at` timestamp is set after the publisher's processing delay, meaning publisher processing is not part of the reported latency. Total pipeline latency = publisher processing + queue transit + subscriber processing; only the latter two are captured. This is consistent with DDS measurement conventions (publication timestamp is at the point of writing to the middleware).

**L5 — Infrastructure layer metrics not used in cascade.** RUNS\_ON and CONNECTS\_TO edges are not used in the fault cascade. A network partition that isolates a set of physical nodes from each other is not modelled. This is consistent with the known weak correlation of infrastructure-layer Q(v) (ρ ≈ 0.54) and is flagged as a gap in the thesis.

**L6 — Topic and Node cannot be labelled.** Following from L5, `FaultInjector` derives `DEPENDS_ON` only from `PUBLISHES_TO`, `SUBSCRIBES_TO` and `USES`. It has no rule that expresses the failure of a Topic (a topic is orphaned *by* a publisher or broker failing, never injected directly) or of a physical Node (no `RUNS_ON → DEPENDS_ON` derivation exists). Injecting either yields $I(v) = 0$ for **every** instance. These types are therefore excluded from `--node-types` and recorded in `unlabeled_node_ids`, leaving 33–160 nodes per scenario without ground truth (≈ 30–47% of components). The GNN still *predicts* scores for them; those predictions are simply never validated. Closing this requires adding the missing derivation rules to the cascade, not merely widening `--node-types`.

**L7 — Only three of five label dimensions are measured.** `FaultInjector` emits a single scalar, so `maintainability` and `security` have no ground truth from this engine. They are declared absent via `labeled_dimensions` and excluded from the loss via `dimension_mask` (§6.1). The four-dimensional RMAV decomposition exists only in `FailureSimulator`, which serves the Validate stage (§2.1). Unifying them would require one engine to produce all five dimensions.

**L8 — No edge-removal simulation.** `EdgeCriticality` is declared in `saag/simulation/models.py` but never populated: `all_edge_criticality` is initialised empty and passed straight through, so `SimulationService.classify_edges()` always returns `[]`. Edge criticality labels used for training are a *projection of node labels* through a hand-chosen bridge multiplier (1.0 for bridges, 0.1 otherwise), not an observation of what happens when an edge fails. Reported edge metrics are therefore validated against a heuristic rather than ground truth and should not be read as evidence of predictive accuracy for edges; node-level results are unaffected. See [prediction.md §2.6](prediction.md#26-edge-criticality-prediction). Closing this means simulating removal of each candidate edge (bridges ∪ high edge-betweenness) and populating `EdgeCriticality` from the resulting reachability and fragmentation deltas.

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

**L6 — No timeout / retry modelling.** For RELIABLE QoS, the head-drop policy prevents queue overflow but does not model TCP-style retransmission or DDS heartbeat/acknowledgement. The modelled delivery rates will be optimistic relative to real network conditions.