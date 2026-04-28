# Failure Simulation

This document describes the two simulation modes available in `simulate_graph.py` and the Python modules that back them: `src/simulation/fault_injector.py` and `src/simulation/message_flow_simulator.py`.

---

## Contents

1. [Motivation and Design Rationale](#1-motivation-and-design-rationale)
2. [Architecture Overview](#2-architecture-overview)
3. [Mode 1 — Fault Injection](#3-mode-1--fault-injection)
   - [Algorithm](#31-algorithm)
   - [I(v) Formula](#32-iv-formula)
   - [Cascade Propagation](#33-cascade-propagation)
   - [Broker Failure Semantics](#34-broker-failure-semantics)
   - [Multi-Seed Stability](#35-multi-seed-stability)
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
├── fault-inject  subcommand
│   └── src/simulation/fault_injector.py
│       ├── _PubSubIndex          (O(1) lookup structures over PUBLISHES_TO / SUBSCRIBES_TO / ROUTES)
│       ├── FaultInjector.run()   (iterates over candidate nodes)
│       └── FaultInjector._cascade()  (BFS wave propagation per node per seed)
│
├── message-flow  subcommand
│   └── src/simulation/message_flow_simulator.py
│       ├── TopicFanout           (per-topic fan-out manager)
│       ├── SubscriberQueue       (per-(topic, subscriber) SimPy Store)
│       ├── _publisher_process()  (SimPy generator: emits messages at rate_hz)
│       ├── _subscriber_process() (SimPy generator: dequeues, checks QoS)
│       └── MessageFlowSimulator.run()
│
└── combined  subcommand
    (runs fault-inject then message-flow in sequence)

src/simulation/simulation_results.py  (shared dataclasses for both modes)
├── FaultInjectionResult / FaultInjectionRecord / CascadeWave
└── MessageFlowResult / TopicFlowStats / SubscriberFlowStats / FaultEventRecord
```

The CLI uses a **subcommand pattern** so fault injection and message flow share a common `--input` / `--output` / `--export-json` / `--verbose` interface while each exposes its own mode-specific flags.

---

## 3. Mode 1 — Fault Injection

### 3.1 Algorithm

The fault injector runs a **BFS cascade simulation** on the pub-sub graph for every candidate node. The pub-sub graph is the projection of the SaG graph onto its PUBLISHES\_TO, SUBSCRIBES\_TO, and ROUTES edges — infrastructure (RUNS\_ON, CONNECTS\_TO) and derived (DEPENDS\_ON) edges are not used.

Before any injection begins, `_PubSubIndex` builds six lookup dictionaries from the graph in O(E):

| Dictionary | Maps |
|---|---|
| `topic_publishers` | topic → set of publisher application IDs |
| `topic_subscribers` | topic → set of subscriber application IDs |
| `app_publishes` | application → set of topic IDs it publishes to |
| `app_subscribes` | application → set of topic IDs it subscribes to |
| `broker_routes` | broker → set of topic IDs it routes |
| `topic_routers` | topic → set of broker IDs that route it (inverse of `broker_routes`) |

For each candidate node v the cascade runs as follows:

**Wave 0 — direct orphaning.**
The node v is added to `failed_nodes`. For each topic t that v published to (or routed, for brokers), the algorithm checks whether any other live publisher (or live broker router) still serves t. If none remains, t is orphaned. All subscribers of each orphaned topic lose that feed.

**Waves 1, 2, … — cascade propagation.**
A subscriber is added to the next wave's propagation set when:
- it has lost a fraction of its feeds ≥ `propagation_threshold` (default 1.0), **and**
- it was itself a publisher on at least one topic.

In the next wave each propagation candidate is added to `failed_nodes`, then its published topics are re-evaluated for orphaning. This continues until no new orphaning occurs (fixpoint) or the `cascade_depth_limit` is reached.

A **set-based pending queue** (not a list) is used for propagation candidates so that a subscriber losing feeds from two topics in the same wave is not processed twice.

### 3.2 I(v) Formula

For each subscriber application a, the **feed-loss fraction** is:

```
feed_loss_fraction(a) = |lost_feeds(a)| / |all_subscribed_feeds(a)|
```

The **proxy ground-truth impact score** I(v) is the mean over all subscriber applications:

```
I(v) = (1 / |Subscribers|) × Σ_{a ∈ Subscribers} feed_loss_fraction(a)
```

This is a graded score in [0, 1]. A subscriber that loses half its feeds contributes 0.5 to the sum, not 1.0. This more faithfully models partial degradation than a binary "is impacted?" measure.

`Subscribers` is the set of Application nodes that have at least one SUBSCRIBES\_TO edge; this denominator is fixed across all injections so scores are comparable.

> **Note.** The failed node v is excluded from the impacted subscriber count even if it also has subscriptions. This avoids self-referential inflation of the score.

### 3.3 Cascade Propagation

The `propagation_threshold` parameter (default 1.0, range [0.0, 1.0]) controls how much feed loss is required before a subscriber is considered failed and begins spreading the cascade.

| `propagation_threshold` | Semantic |
|---|---|
| `1.0` (default) | A subscriber only cascades when it has lost **all** its feeds (completely starved). Conservative — minimises false cascades. |
| `0.5` | A subscriber cascades when it has lost ≥ 50% of its feeds. |
| `0.0` | Any single feed loss triggers a cascade. Aggressive — maximises spread. |

For the ATM dataset, `ConflictDetector` requires both `T_radar` **and** `T_tracks` to function (both are mandatory inputs to the conflict algorithm). Setting `--propagation-threshold 0.5` will model this correctly: losing either feed alone is sufficient to silence `ConflictDetector`.

### 3.4 Broker Failure Semantics

When a Broker node fails, the injector uses the `topic_routers` inverse index to check whether any **other live broker** also routes each topic. A topic is only orphaned if all of its routing brokers are in `failed_nodes`.

This correctly handles multi-broker redundancy: in a deployment where `Broker-A` and `Broker-B` both route `T_radar`, failing `Broker-A` alone does not orphan `T_radar`. In single-broker topologies (typical for the ATM dataset), the behaviour is unchanged — the broker is always the sole router and every topic it routes is orphaned.

### 3.5 Multi-Seed Stability

The cascade propagation order within a wave is non-deterministic when multiple nodes are eligible to propagate simultaneously (tie-breaking). Each seed produces a different shuffle of the wave candidates, testing whether I(v) depends on this ordering.

With N seeds:
- `impact_score` is the **mean** I(v) across all seeds.
- `impact_score_std` is the **standard deviation** across seeds.
- The cascade trace (waves, orphaned topics, impacted subscribers) in the JSON record is from the **seed whose impact score is closest to the mean** (median-representative seed), giving the most stable trace for human inspection.

**Interpreting std values:**
- `std = 0.0` on a deterministic topology (most real systems): each seed produces identical results.
- `std > 0` indicates that I(v) is sensitive to the propagation order, typically at the boundary of a cascade — a signal of fragility that is itself worth reporting.

Recommended seeds for thesis experiments: `42,123,456,789,2024`.

---

## 4. Mode 2 — Message Flow Simulation

### 4.1 Discrete-Event Model

The message flow simulator uses **SimPy** (https://simpy.readthedocs.io) — a process-based discrete-event simulation library. Simulated time is in seconds, mapping 1-to-1 to the real-world time units of the modelled system.

Three types of SimPy process are spawned for each edge in the graph:

**Publisher process** (one per PUBLISHES\_TO edge):
1. `yield env.timeout(1.0 / rate_hz)` — wait one publish interval.
2. If `app_id in failed_nodes`, stop.
3. Optionally yield a processing delay (from the `processing_time` node attribute).
4. Create a `Message(msg_id, topic_id, publisher_id, created_at=env.now)`.
5. Call `fanout.publish(msg, failed_nodes)` to fan the message to all live subscriber queues.
6. Increment the appropriate time-window publish counter (`pre` or `post` fault).

**Subscriber process** (one per SUBSCRIBES\_TO edge):
1. Check `app_id in failed_nodes` **before** issuing a `get()`. Exit if failed.
2. `msg = yield sq.get()` — block until a message arrives in the private queue.
3. Check `app_id in failed_nodes` again (may have been faulted during the wait).
4. Optionally yield a subscriber-side processing delay.
5. Compute end-to-end latency: `(env.now - msg.created_at) × 1000 ms`.
6. Apply QoS checks (lifespan, deadline) using the end-to-end latency.
7. If delivered: increment `total_delivered`, store latency sample.

**Fault process** (one per simulation, if `--fault-node` is set):
1. `yield env.timeout(fault_time)`.
2. Add `fault_node` to `failed_nodes`. Publisher and subscriber processes observe this on their next loop iteration.

All three process types share the same `failed_nodes: Set[str]` object, which serves as the inter-process fault broadcast channel.

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

1. The SUBSCRIBES\_TO edge (`qos_profile` attribute) — subscriber-side policy.
2. The Topic node (`qos_profile` attribute) — topic-level policy; `deadline_ms` takes precedence over the edge-level value when set.

Both sources follow the same structure:

```json
{
  "reliability": "RELIABLE",
  "durability":  "TRANSIENT_LOCAL",
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
| `--node-types TYPE1,TYPE2` | `Application,Broker` | Node types eligible for injection. |
| `--seeds 42,123,...` | `42` | Comma-separated integer seeds for multi-seed stability. |
| `--cascade-depth N` | `0` (unlimited) | Maximum cascade wave depth. |
| `--verbose` / `-v` | off | Enable DEBUG logging. |

> **Propagation threshold** is currently only configurable via the Python API (`FaultInjector(propagation_threshold=0.5)`), not the CLI. This is intentional — it is a research parameter that should be set deliberately, not accidentally via a flag.

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
  "schema_version": "2.0",
  "graph_id": "atm_system",
  "total_nodes_injected": 6,
  "total_application_nodes": 5,
  "total_broker_nodes": 1,
  "total_subscribers": 3,
  "seeds_used": [42, 123, 456, 789, 2024],
  "top_k_by_impact": [ ... ],
  "records": { ... }
}
```

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
| `per_subscriber_feed_loss` | dict | Per-subscriber feed-loss fraction contributing to I(v). |
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
  "delivery_rate_after": 0.9962
}
```

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

### Expected fault-inject results

| Node | I(v) | Cascade depth | Why |
|------|------|---------------|-----|
| `RadarTracker` | 1.000 | 1 | Sole publisher of T_radar and T_tracks; ConflictDetector and FlightDataProcessor both lose all feeds → cascade → T_conflicts and T_fpa also orphaned; all 3 subscribers lose 100% of their feeds |
| `ASTERIX_Broker` | 1.000 | 1 | Sole router of all 5 topics; same total loss |
| `ConflictDetector` | 0.111 | 0 | Orphans only T_conflicts; ATCWorkstation loses 1/3 feeds (T_conflicts only); other subscribers unaffected |
| `FlightDataProcessor` | 0.111 | 0 | Orphans only T_fpa; ATCWorkstation loses 1/3 feeds |
| `ATCWorkstation` | 0.000 | 0 | Not a publisher; removing it harms no downstream subscriber |
| `MeteoService` | 0.000 | 0 | Orphans T_meteo but T_meteo has no subscribers |

> With `propagation_threshold=0.5`: ConflictDetector losing T_radar alone (1/2 feeds = 50%) would trigger a cascade to T_conflicts → ATCWorkstation also loses T_conflicts → ConflictDetector's I(v) rises.

### Running the full validation workflow

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

# Step 3: Compute Spearman ρ between Q(v) and I(v)
# (handled by ValidateUseCase / bin/validate_topology_classes.py,
#  which reads output/analysis/analysis_results.json and
#  output/simulation/impact_scores.json)
python bin/validate_topology_classes.py \
    --analysis output/analysis/analysis_results.json \
    --impact   output/simulation/impact_scores.json
```

### Message flow: observing the ConflictDetector fault

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

**Pairing keys.** Both `analysis_results.json` and `impact_scores.json` use the node ID (string matching the graph node name) as the primary key. The validation script should inner-join on this key, discarding nodes present in only one file (e.g., Topic nodes that appear in analysis but are not injection candidates).

**Node-type stratified reporting.** The `node_type` field in each record allows the Spearman ρ to be computed separately for Application nodes and Broker nodes, which is important because the infrastructure layer has historically shown weaker correlation (ρ ≈ 0.54) than the application layer (ρ = 0.876).

**Multi-seed stability gate.** Before using I(v) for publication, verify that `impact_score_std` is below a threshold (suggested: 0.02) for all nodes. High std values indicate topology boundary fragility that should be investigated.

---

## 9. Input Graph Format Requirements

The `--input` file must be a JSON file compatible with the SaG schema. The CLI loader handles two paths automatically:

**Path 1 (preferred):** If `src/core/graph_builder.py` and `src/core/graph_exporter.py` are importable, they are used. This supports the full schema including MIL-STD-498 hierarchy metadata, Jira enrichment, and code metrics.

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

### FaultInjector

```python
from src.simulation.fault_injector import FaultInjector
import networkx as nx

# graph: NetworkX DiGraph with PUBLISHES_TO, SUBSCRIBES_TO, ROUTES edges
injector = FaultInjector(
    graph=graph,
    seeds=[42, 123, 456, 789, 2024],
    cascade_depth_limit=0,          # 0 = unlimited
    propagation_threshold=1.0,      # 1.0 = completely starved
)

# Inject all Application and Broker nodes
result = injector.run(node_types=["Application", "Broker"])

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
```

### MessageFlowSimulator

```python
from src.simulation.message_flow_simulator import MessageFlowSimulator

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
```

---

## 11. Known Limitations

**L1 — Broker routing model is binary.** The fault injector models broker failure as "topic routed by the failed broker is orphaned if no other live broker routes it." In practice, DDS routing is more nuanced — a broker failure mid-message can cause partial delivery even with redundant routing. The current model is conservative and correct for single-broker topologies (ADVENT, ATM datasets).

**L2 — TRANSIENT\_LOCAL durability not fully simulated.** The message flow simulator notes the `TRANSIENT_LOCAL` QoS policy but does not implement late-joiner history replay. A subscriber that joins after the publisher starts will not receive historical samples. This affects correctness for simulations modelling late-joining controllers.

**L3 — Single fault per simulation run.** Both simulators model at most one node failure at a time. Correlated failures (e.g., a power loss taking down all nodes in a rack) require running the combined mode with explicit topology modifications or extending the fault injector with a `fault_group` parameter.

**L4 — Publisher-side processing time is not included in end-to-end latency.** The message `created_at` timestamp is set after the publisher's processing delay, meaning publisher processing is not part of the reported latency. Total pipeline latency = publisher processing + queue transit + subscriber processing; only the latter two are captured. This is consistent with DDS measurement conventions (publication timestamp is at the point of writing to the middleware).

**L5 — Infrastructure layer metrics not used in cascade.** RUNS\_ON and CONNECTS\_TO edges are not used in the fault cascade. A network partition that isolates a set of physical nodes from each other is not modelled. This is consistent with the known weak correlation of infrastructure-layer Q(v) (ρ ≈ 0.54) and is flagged as a gap in the thesis.

**L6 — No timeout / retry modelling.** For RELIABLE QoS, the head-drop policy prevents queue overflow but does not model TCP-style retransmission or DDS heartbeat/acknowledgement. The modelled delivery rates will be optimistic relative to real network conditions.