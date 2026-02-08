# Step 4: Failure Simulation

**Measure actual failure impact to establish ground truth for validation**

---

## Overview

Failure Simulation injects faults into the system graph and measures the actual impact. This provides ground truth $I(v)$ to validate our predicted quality scores $Q(v)$.

```
┌─────────────────────┐          ┌─────────────────────┐
│  Graph Model        │          │  Impact Scores      │
│  (G_structural)     │    →     │                     │
│  For each v:        │          │  I(v) = actual      │
│    - Remove v       │          │  failure impact     │
│    - Cascade        │          │                     │
│    - Measure impact │          │  (Ground Truth)     │
│    - Restore v      │          │                     │
└─────────────────────┘          └─────────────────────┘
```

The simulation operates on $G_{structural}$ — the full graph with raw relationships (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO) — rather than the derived $G_{analysis}$ used by Steps 2–3. This preserves physical topology for realistic cascade propagation.

---

## Why Simulate Failures?

| Predicted $Q(v)$ | Simulated $I(v)$ |
|------------------|-------------------|
| Based on graph topology (centrality, articulation points) | Based on actual failure effects (path loss, fragmentation) |
| Fast to compute | Slower but empirically grounded |
| Theoretical importance | Operational impact |

**Validation**: Compare $Q(v)$ rankings with $I(v)$ rankings to verify that topological metrics reliably predict runtime failure impact (→ [Step 5](validation.md)).

---

## Algorithm

```
Algorithm 1: Failure Simulation with Cascade Propagation
─────────────────────────────────────────────────────────
Input:  G_structural = (V, E), target component v
Output: Impact score I(v), cascade sequence

 1  BASELINE ← capture_state(G)          // paths, connected components, throughput
 2  mark v as FAILED
 3  cascade_set ← {v}
 4  queue ← [(v, depth=0)]
 5
 6  WHILE queue is not empty DO
 7      (u, d) ← dequeue(queue)
 8      IF d ≥ max_cascade_depth THEN CONTINUE
 9
10      // Physical cascade: Node → hosted components
11      IF type(u) = Node THEN
12          FOR EACH c hosted on u DO
13              IF c ∉ cascade_set AND rand() < p_cascade THEN
14                  mark c as FAILED
15                  cascade_set ← cascade_set ∪ {c}
16                  enqueue(queue, (c, d+1))
17
18      // Logical cascade: Broker → unroutable topics
19      IF type(u) = Broker THEN
20          FOR EACH topic t routed through u DO
21              IF no active brokers remain for t THEN
22                  mark t as FAILED
23                  cascade_set ← cascade_set ∪ {t}
24                  enqueue(queue, (t, d+1))
25
26      // Logical cascade: Publisher → data starvation
27      IF type(u) = Application THEN
28          FOR EACH topic t published by u DO
29              IF no active publishers remain for t THEN
30                  mark t as FAILED (starved)
31                  cascade_set ← cascade_set ∪ {t}
32                  enqueue(queue, (t, d+1))
33
34      // Network cascade: Node → isolated neighbors
35      IF type(u) = Node THEN
36          FOR EACH neighbor n connected to u DO
37              IF n has no remaining active connections THEN
38                  mark n as FAILED (partitioned)
39                  cascade_set ← cascade_set ∪ {n}
40                  enqueue(queue, (n, d+1))
41
42  // Measure impact
43  R ← reachability_loss(G, BASELINE)    // broker-aware path counting
44  F ← fragmentation(G, BASELINE)        // connected components increase
45  T ← throughput_loss(G, BASELINE)      // QoS-weighted topic capacity
46
47  I(v) ← w_r × R + w_f × F + w_t × T
48
49  RESTORE G to BASELINE state
50  RETURN I(v), cascade_set
```

---

## Cascade Rules

### Physical Cascade

When a **Node** fails, all hosted components fail:

```
Node-1 fails
   ↓
App-A (on Node-1) fails       ← via RUNS_ON
Broker-1 (on Node-1) fails    ← via RUNS_ON
   ↓
(Broker-1 failure triggers logical cascade...)
```

**Rationale**: Hardware failure is deterministic — when a physical host goes down, all software running on it becomes unavailable. This is the most common initiator of deep cascade chains.

### Logical Cascade

When a **Broker** fails, topics exclusively routed through it become unreachable:

```
Broker-1 fails
   ↓
Topic-X (only routed by Broker-1) unreachable    ← no active brokers
   ↓
Subscribers to Topic-X receive no data
```

When a **Publisher** fails, subscribers may be starved:

```
App-A (sole publisher) fails
   ↓
Topic-X has no active publishers    ← publisher starvation
   ↓
App-B, App-C (subscribers) receive no data
```

**Rationale**: In publish-subscribe systems, message delivery requires an active publisher, an active routing broker, AND an active subscriber. Failure at any point in this chain breaks the communication path. The logical cascade ensures that broker failures correctly propagate to dependent topics and applications.

### Network Cascade

When a **Node** fails, connected nodes that lose all connections become isolated:

```
Node-1 fails (sole connection to Node-2)
   ↓
Node-2 has no remaining active connections    ← network partition
   ↓
Node-2 marked as partitioned
   ↓
All components hosted on Node-2 cascade...
```

**Rationale**: In distributed systems, network connectivity is a prerequisite for communication. An isolated node is functionally equivalent to a failed node for all remote interactions.

---

## Impact Metrics

### Reachability Loss

Fraction of **deliverable** pub-sub paths broken by the failure. A path $(\text{pub} \to \text{topic} \to \text{sub})$ is deliverable only when all three conditions hold:

1. The publisher is active
2. The subscriber is active
3. **At least one routing broker for the topic is active**

$$
\text{Reachability Loss} = 1 - \frac{|\text{remaining deliverable paths}|}{|\text{initial deliverable paths}|}
$$

The broker-awareness condition (3) is critical: without it, broker failures would show artificially low reachability loss even though messages cannot actually be delivered.

### Fragmentation

Normalized increase in **weakly-connected components** of the active subgraph after failure.

$$
\text{Fragmentation} = \frac{\max(0,\; CC_{after} - CC_{before})}{|V_{initial}| - 1}
$$

where $CC$ denotes the number of connected components. This measures true topology disruption — how many disconnected "islands" the failure creates — rather than simple component loss ratio. The denominator normalizes to [0, 1] where 1.0 means every component became its own island.

**Design note**: Fragmentation is intentionally orthogonal to reachability loss. Reachability measures broken communication paths; fragmentation measures structural disconnection of the topology. A failure can break many paths without fragmenting the graph (e.g., a heavily-published topic losing its sole publisher), or fragment the graph without breaking paths (e.g., an isolated monitoring node).

### Throughput Loss

**QoS-weighted** reduction in message delivery capacity. Topics with higher QoS weight (derived from reliability, durability, and transport priority in Step 1) contribute more to the loss.

$$
\text{Throughput Loss} = \frac{\sum_{t \in \text{affected}} w(t)}{\sum_{t \in \text{all}} w(t)}
$$

where $w(t)$ is the QoS-derived weight of topic $t$, and a topic is "affected" if it has lost any part of its delivery chain (no active publishers, no active brokers, or no active subscribers).

### Composite Impact Score $I(v)$

$$
I(v) = w_r \times \text{Reachability} + w_f \times \text{Fragmentation} + w_t \times \text{Throughput}
$$

Default weights: $w_r = 0.4$, $w_f = 0.3$, $w_t = 0.3$.

| Weight | Dimension | Rationale |
|--------|-----------|-----------|
| 0.4 | Reachability | Most direct measure of communication disruption in pub-sub |
| 0.3 | Fragmentation | Captures topology disruption not visible in path counting |
| 0.3 | Throughput | QoS-weighted capacity loss prioritizes critical channels |

Weights are configurable via `impact_weights` on `ImpactMetrics` and can be justified using AHP or validated via sensitivity analysis (see below).

---

## Cascade Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cascade_probability` | 1.0 | Probability that each cascade step propagates. Set < 1.0 for stochastic (Monte Carlo) simulation. |
| `max_cascade_depth` | 10 | Maximum BFS depth for cascade propagation. Prevents infinite loops in cyclic topologies. |
| `cascade_rule` | ALL | Which cascade types to apply: PHYSICAL, LOGICAL, NETWORK, or ALL. |

---

## Commands

### Single Component Failure

```bash
python bin/simulate_graph.py failure --target main_broker --layer system
```

### Exhaustive Analysis

Simulate failure of every component and rank by impact:

```bash
python bin/simulate_graph.py failure --layer system --exhaustive
```

### Monte Carlo (Stochastic)

Run N trials with probabilistic cascade propagation:

```bash
python bin/simulate_graph.py failure --target main_broker --layer system \
    --monte-carlo --trials 200 --cascade-probability 0.8
```

### Impact Weight Sensitivity

Verify that $I(v)$ rankings are robust to weight perturbation:

```bash
python bin/simulate_graph.py failure --layer system --exhaustive --weight-sensitivity
```

This runs 200 perturbations ($\sigma = 0.05$) and reports Top-5 Stability, Mean Kendall $\tau$, and Std Kendall $\tau$.

---

## Simulation Modes

### Failure Mode: Crash (default)

Complete component removal — the component and all its relationships become inactive.

```bash
python bin/simulate_graph.py failure --target X --mode crash
```

### Failure Mode: Degraded

Partial failure — reduced capacity. The component remains active but operates at diminished throughput (e.g., 50% message processing rate).

```bash
python bin/simulate_graph.py failure --target X --mode degraded
```

### Failure Mode: Partition

Network partition — the component is unreachable from parts of the network but continues operating internally.

```bash
python bin/simulate_graph.py failure --target X --mode partition
```

### Failure Mode: Overload

Resource exhaustion — increased latency and probabilistic message drops under load.

```bash
python bin/simulate_graph.py failure --target X --mode overload
```

---

## Layer-Specific Analysis

```bash
# Application layer only
python bin/simulate_graph.py failure --exhaustive --layer app

# Infrastructure layer only
python bin/simulate_graph.py failure --exhaustive --layer infra

# Middleware layer only
python bin/simulate_graph.py failure --exhaustive --layer mw

# Complete system
python bin/simulate_graph.py failure --exhaustive --layer system
```

---

## Export Results

```bash
# Export to JSON for validation
python bin/simulate_graph.py failure --exhaustive --layer system --output results/simulation.json
```

**JSON Structure:**
```json
{
  "metadata": {
    "layer": "system",
    "total_components": 48,
    "timestamp": "2025-07-15T10:30:00Z"
  },
  "results": [
    {
      "target_id": "main_broker",
      "target_type": "Broker",
      "scenario": "Exhaustive failure: main_broker",
      "impact": {
        "reachability": {
          "initial_paths": 240,
          "remaining_paths": 36,
          "loss_percent": 85.0
        },
        "fragmentation": {
          "initial_components": 48,
          "failed_components": 12,
          "initial_connected_components": 1,
          "final_connected_components": 4,
          "fragmentation_percent": 6.38
        },
        "throughput": {
          "initial_throughput": 45.6,
          "remaining_throughput": 9.8,
          "loss_percent": 78.5
        },
        "affected": {
          "topics": 18,
          "subscribers": 22,
          "publishers": 15
        },
        "cascade": {
          "count": 11,
          "depth": 3,
          "by_type": {"Topic": 8, "Application": 3}
        },
        "composite_impact": 0.5975
      },
      "cascaded_failures": ["Topic-1", "Topic-2", "App-A", "..."],
      "cascade_sequence": [
        {"id": "main_broker", "type": "Broker", "cause": "initial_failure", "depth": 0},
        {"id": "Topic-1", "type": "Topic", "cause": "no_active_brokers:main_broker", "depth": 1},
        {"id": "Topic-2", "type": "Topic", "cause": "no_active_brokers:main_broker", "depth": 1}
      ],
      "layer_impacts": {
        "app": 0.12,
        "infra": 0.0,
        "mw": 0.33,
        "system": 0.08
      },
      "related_components": ["Routes: /sensor/lidar", "Routes: /cmd_vel"]
    }
  ]
}
```

---

## Key Insights

1. **Ground Truth**: $I(v)$ provides empirical validation for $Q(v)$ predictions. High correlation between predicted and simulated rankings ($\rho > 0.8$) proves the graph model is a valid predictor.

2. **Cascade Effects**: Physical infrastructure failures (Node → hosted components → broker routing) often cascade more severely than isolated application failures, because they trigger all three cascade types simultaneously.

3. **Broker Criticality**: Brokers that exclusively route high-weight topics produce disproportionately high $I(v)$ scores. The broker-aware reachability metric ensures these components are correctly identified as SPOFs.

4. **SPOF Detection**: Components with $I(v) > 0.5$ are empirical single points of failure. Cross-referencing with $Q(v) > 0.5$ identifies components where both topology and simulation agree on criticality.

5. **Layer Differences**: Application layer predictions typically correlate better with simulation ($\rho \approx 0.85$) than infrastructure ($\rho \approx 0.75$), because application dependencies are more directly captured by the DEPENDS_ON derivation rules.

---

## Implementation Notes

### Baseline Caching

In exhaustive mode, the baseline state (initial paths, connected components, total topic weight) is computed once and reused across all $N$ simulations. This reduces the complexity from $O(N \times P)$ to $O(N + P)$ where $P$ is the cost of path enumeration.

### Cascade Tracking

Every cascade event is recorded with its cause and depth, enabling:

- **Cascade tree visualization** via `FailureResult.cascade_to_graph()` (→ [Step 6](visualization.md))
- **Cause attribution**: distinguish `hosted_on`, `no_active_brokers`, `publisher_starvation`, and `network_partition` causes
- **Depth analysis**: identify components that trigger deep vs. shallow cascades

### Monte Carlo Mode

When `cascade_probability < 1.0`, each cascade step propagates with the given probability. Running $N$ trials produces a distribution of $I(v)$ values, yielding:

- Mean and standard deviation of impact
- 95% confidence interval
- Variance analysis across cascade paths

This is useful for systems where failure propagation is non-deterministic (e.g., retry mechanisms, circuit breakers).

---

## Next Step

→ [Step 5: Validation](validation.md)