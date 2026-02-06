# Step 4: Failure Simulation

**Measure actual failure impact to establish ground truth for validation**

---

## Overview

Failure Simulation injects faults into the system graph and measures the actual impact. This provides ground truth I(v) to validate our predicted quality scores Q(v).

```
┌─────────────────────┐          ┌─────────────────────┐
│  Graph Model        │          │  Impact Scores      │
│                     │    →     │                     │
│  For each v:        │          │  I(v) = actual      │
│    - Remove v       │          │  failure impact     │
│    - Measure impact │          │                     │
│    - Restore v      │          │  (Ground Truth)     │
└─────────────────────┘          └─────────────────────┘
```

---

## Why Simulate Failures?

| Predicted Q(v) | Simulated I(v) |
|----------------|----------------|
| Based on graph topology | Based on actual failure effects |
| Fast to compute | Slower but accurate |
| Theoretical importance | Empirical impact |

**Validation**: Compare Q(v) rankings with I(v) rankings to verify predictions.

---

## Simulation Process

For each component v:

```
1. Capture baseline state
   └── Count paths, components, throughput

2. Inject failure
   └── Remove component v from graph

3. Propagate cascade
   ├── Physical: Node → hosted Apps fail
   ├── Logical: Broker → Topics unreachable
   └── Network: Partitions propagate

4. Measure impact
   ├── Reachability loss (broken paths)
   ├── Fragmentation (disconnected islands)
   └── Throughput loss (capacity reduction)

5. Compute I(v)
   └── Composite impact score

6. Restore component v
```

---

## Cascade Rules

### Physical Cascade

When a **Node** fails, all hosted components fail:

```
Node-1 fails
   ↓
App-A (on Node-1) fails
Broker-1 (on Node-1) fails
```

### Logical Cascade

When a **Broker** fails, its topics become unreachable:

```
Broker-1 fails
   ↓
Topic-X (routed by Broker-1) unreachable
   ↓
Subscribers to Topic-X affected
```

### Application Cascade

When a **Publisher** fails, subscribers may be starved:

```
App-A (publisher) fails
   ↓
Topic-X has no publishers
   ↓
App-B, App-C (subscribers) receive no data
```

---

## Impact Metrics

### Reachability Loss

Percentage of pub-sub paths broken by the failure.

```
Reachability Loss = (initial_paths - remaining_paths) / initial_paths
```

### Fragmentation

Increase in disconnected graph components.

```
Fragmentation = failed_components / initial_components
```

### Throughput Loss

Reduction in message delivery capacity (based on topic weights).

```
Throughput Loss = lost_weight / total_weight
```

### Composite Impact Score I(v)

```
I(v) = 0.4×Reachability + 0.3×Fragmentation + 0.3×Throughput
```

---

## Commands

### Single Component Failure

```bash
python bin/simulate_graph.py --failure main_broker --layer system
```

<!-- ... -->

### Exhaustive Analysis

Simulate failure of every component and rank by impact:

```bash
python bin/simulate_graph.py --exhaustive --layer system
```

<!-- ... -->

## Simulation Modes

### Failure Mode: Crash

Complete component removal (default).

```bash
python bin/simulate_graph.py --failure X --mode crash
```

### Failure Mode: Degraded

Partial failure—reduced capacity.

```bash
python bin/simulate_graph.py --failure X --mode degraded
```

### Failure Mode: Partition

Network partition—component unreachable but running.

```bash
python bin/simulate_graph.py --failure X --mode partition
```

---

## Layer-Specific Analysis

```bash
# Application layer only
python bin/simulate_graph.py --exhaustive --layer app

# Infrastructure layer only
python bin/simulate_graph.py --exhaustive --layer infra

# Complete system
python bin/simulate_graph.py --exhaustive --layer system
```

---

## Export Results

```bash
# Export to JSON for validation
python bin/simulate_graph.py --exhaustive --layer system --output results/simulation.json
```

**JSON Structure:**
```json
{
  "results": [
    {
      "target_id": "main_broker",
      "target_type": "Broker",
      "impact": {
        "composite_impact": 0.9125,
        "reachability_loss": 0.85,
        "fragmentation": 0.333,
        "throughput_loss": 0.785
      },
      "cascade_count": 4,
      "cascaded_failures": ["Topic-1", "App-A", ...]
    }
  ]
}
```

---

## Key Insights

1. **Ground Truth**: I(v) provides empirical validation for Q(v) predictions.

2. **Cascade Effects**: Physical infrastructure failures often cascade more severely than application failures.

3. **SPOF Detection**: Components with I(v) > 0.5 are true single points of failure.

4. **Layer Differences**: Application layer predictions correlate better (ρ=0.85) than infrastructure (ρ=0.54).

---

## Next Step

→ [Step 5: Validation](validation.md)
