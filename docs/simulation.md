# Failure Simulation

This document explains the failure simulation framework used to measure actual system impact when components fail.

---

## Table of Contents

1. [Overview](#overview)
2. [Simulation Modes](#simulation-modes)
3. [Impact Score Calculation](#impact-score-calculation)
4. [Cascade Propagation](#cascade-propagation)
5. [Exhaustive Campaign](#exhaustive-campaign)
6. [Attack Simulation](#attack-simulation)
7. [Event-Driven Simulation](#event-driven-simulation)
8. [Implementation](#implementation)

---

## Overview

Failure simulation measures the **actual impact** of component failures. This serves as ground truth for validating predictions from structural analysis.

### Purpose

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   PREDICTION (from topology)      VALIDATION        ACTUAL IMPACT  │
│                                                     (from simulation)│
│   ┌─────────────────┐              Compare         ┌───────────────┐│
│   │ Criticality     │ ─────────────────────────────│ Impact Score  ││
│   │ Score           │                              │               ││
│   └─────────────────┘                              └───────────────┘│
│                                                                     │
│   Do topological metrics accurately predict failure impact?        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Component Failure** | Removing a component from the active system |
| **Reachability Loss** | Message paths that become unavailable |
| **Fragmentation** | Disconnected subgraphs created |
| **Cascade** | Dependent components that also fail |
| **Impact Score** | Composite metric of failure severity |

---

## Simulation Modes

The framework supports multiple simulation approaches.

### Single Failure

Simulate the failure of one specific component.

```python
from src.simulation import FailureSimulator, SimulationGraph

graph = SimulationGraph.from_dict(graph_data)
simulator = FailureSimulator(seed=42)

result = simulator.simulate_failure(
    graph,
    component_id="B1",
    enable_cascade=True
)

print(f"Impact: {result.impact.impact_score:.4f}")
print(f"Paths lost: {result.impact.paths_lost}")
print(f"Cascade failures: {len(result.cascade_failures)}")
```

### Batch Failure

Simulate failure of multiple components simultaneously.

```python
result = simulator.simulate_batch_failure(
    graph,
    component_ids=["B1", "N2"],
    enable_cascade=True
)
```

### Exhaustive Campaign

Simulate failure of every component individually.

```python
batch = simulator.simulate_all_failures(
    graph,
    enable_cascade=True
)

print(f"Total simulations: {len(batch.results)}")
print(f"Most critical: {batch.critical_components[:5]}")
```

### Attack Simulation

Simulate targeted attacks on high-criticality components.

```python
from src.simulation import AttackStrategy

result = simulator.simulate_attack(
    graph,
    strategy=AttackStrategy.HIGHEST_BETWEENNESS,
    count=3,  # Attack top 3
    enable_cascade=True
)
```

---

## Impact Score Calculation

The impact score quantifies the severity of a failure.

### Formula

```
Impact(v) = w₁ × reachability_loss + w₂ × fragmentation + w₃ × cascade_extent
```

Default weights: w₁=0.5, w₂=0.3, w₃=0.2

### Components

#### Reachability Loss

Fraction of message paths destroyed by the failure.

```
                    |broken_paths|
reachability_loss = ─────────────────
                    |total_paths|
```

A message path is: Publisher → Topic → Subscriber.

**Example**:
```
Before failure:
  [A1]──▶[T1]──▶[A2]  ✓ Path 1
  [A1]──▶[T1]──▶[A3]  ✓ Path 2
  [A4]──▶[T2]──▶[A5]  ✓ Path 3
  Total: 3 paths

After B1 fails (routes T1):
  [A1]──▶[T1]──▶[A2]  ✗ Broken
  [A1]──▶[T1]──▶[A3]  ✗ Broken
  [A4]──▶[T2]──▶[A5]  ✓ Still works
  Broken: 2 paths

reachability_loss = 2/3 = 0.667
```

#### Fragmentation

Increase in disconnected components after failure.

```
                    components_after - components_before
fragmentation = ─────────────────────────────────────────
                           total_nodes - 1
```

**Example**:
```
Before: 1 connected component
After:  3 connected components

fragmentation = (3 - 1) / (10 - 1) = 0.222
```

#### Cascade Extent

Fraction of components that fail due to cascade.

```
                    |cascade_failures|
cascade_extent = ─────────────────────
                 |remaining_components|
```

**Example**:
```
Primary failure: B1
Cascade failures: A1, A2, A3
Remaining after B1: 9 components

cascade_extent = 3/9 = 0.333
```

### Combined Example

```
reachability_loss = 0.667
fragmentation = 0.222
cascade_extent = 0.333

Impact = 0.5 × 0.667 + 0.3 × 0.222 + 0.2 × 0.333
       = 0.334 + 0.067 + 0.067
       = 0.468
```

---

## Cascade Propagation

Cascade simulation models failure spreading through dependencies.

### Model

When component A fails, each dependent component B may also fail based on:

1. **Dependency Strength**: Weight of DEPENDS_ON edge
2. **Cascade Threshold**: Minimum strength to trigger cascade
3. **Cascade Probability**: Random chance of cascade occurring

```
[A] fails
 │
 ▼
[B] dependency_strength = 0.8
 │
 ├─ strength > threshold (0.5)? YES
 │
 ├─ random() < cascade_prob (0.7)? 
 │   │
 │   ├─ YES → [B] fails → continue cascade
 │   │
 │   └─ NO → [B] survives
 │
 ▼
[C] depends on B...
```

### Configuration

```python
simulator = FailureSimulator(
    cascade_threshold=0.5,     # Minimum dependency strength
    cascade_probability=0.7,   # Probability of cascade
    max_cascade_depth=5,       # Maximum hop count
    seed=42                    # Reproducibility
)
```

### Cascade Depth

Cascade propagates through dependency chains up to `max_cascade_depth`:

```
Depth 0: [B1] fails (primary)
Depth 1: [A1, A2] fail (depend on B1)
Depth 2: [A3] fails (depends on A1)
Depth 3: [A4] fails (depends on A3)
...
```

### Example Output

```python
result = simulator.simulate_failure(graph, "B1", enable_cascade=True)

print(f"Primary failures: {result.primary_failures}")
# ['B1']

print(f"Cascade failures: {result.cascade_failures}")
# [('A1', 1), ('A2', 1), ('A3', 2)]  # (component, depth)

print(f"Total affected: {result.impact.total_affected}")
# 4
```

---

## Exhaustive Campaign

Simulates failure of every component to get complete impact data.

### Process

```
For each component v in graph:
    1. Create graph copy
    2. Remove v
    3. Calculate impact score
    4. Record results

Sort by impact score → ranking
```

### Usage

```python
batch = simulator.simulate_all_failures(
    graph,
    component_types=[ComponentType.APPLICATION, ComponentType.BROKER],
    enable_cascade=True
)

# Results
print(f"Simulations run: {len(batch.results)}")
print(f"Total time: {batch.total_time_ms:.0f} ms")

# Most critical
print("Top 5 critical components:")
for i, comp in enumerate(batch.critical_components[:5], 1):
    result = batch.get_result(comp)
    print(f"  {i}. {comp}: {result.impact.impact_score:.4f}")

# By component type
for comp_type in batch.by_type:
    type_results = batch.by_type[comp_type]
    avg_impact = sum(r.impact.impact_score for r in type_results) / len(type_results)
    print(f"{comp_type}: avg impact = {avg_impact:.4f}")
```

### Output Structure

```python
@dataclass
class BatchSimulationResult:
    results: List[FailureResult]     # All individual results
    critical_components: List[str]   # Sorted by impact (high to low)
    by_type: Dict[str, List]         # Grouped by component type
    total_time_ms: float             # Execution time
    
    def get_result(self, component_id: str) -> FailureResult:
        """Get result for specific component"""
        
    def to_dict(self) -> Dict:
        """Export as dictionary"""
```

---

## Attack Simulation

Simulates targeted attacks that remove multiple high-value components.

### Attack Strategies

| Strategy | Description |
|----------|-------------|
| `HIGHEST_BETWEENNESS` | Target highest betweenness centrality |
| `HIGHEST_DEGREE` | Target highest degree (most connected) |
| `HIGHEST_PAGERANK` | Target highest PageRank |
| `ARTICULATION_POINTS` | Target articulation points first |
| `RANDOM` | Random component selection |

### Usage

```python
from src.simulation import AttackStrategy

# Attack top 3 by betweenness
result = simulator.simulate_attack(
    graph,
    strategy=AttackStrategy.HIGHEST_BETWEENNESS,
    count=3,
    enable_cascade=True
)

print(f"Targeted: {result.targeted_components}")
print(f"Total impact: {result.total_impact:.4f}")
print(f"System survival: {result.system_survival_rate:.2%}")
```

### Incremental vs Simultaneous

**Incremental** (default): Remove components one by one, recalculating after each.

```
Step 1: Remove B1 (highest BC) → recalculate
Step 2: Remove B2 (now highest) → recalculate
Step 3: Remove N1 (now highest) → done
```

**Simultaneous**: Remove all at once.

```python
result = simulator.simulate_attack(
    graph,
    strategy=AttackStrategy.HIGHEST_BETWEENNESS,
    count=3,
    incremental=False  # All at once
)
```

---

## Event-Driven Simulation

Full discrete-event simulation of message flow.

### Features

- Message publication and delivery
- QoS-aware routing (reliable vs best-effort)
- Component failures during simulation
- Load testing with ramping rates
- Chaos engineering with random failures

### Basic Usage

```python
from src.simulation import EventSimulator, QoSLevel

event_sim = EventSimulator(seed=42)

result = event_sim.simulate(
    graph,
    duration_ms=10000,      # 10 seconds
    message_rate=100,       # 100 msg/sec
    qos=QoSLevel.AT_LEAST_ONCE
)

print(f"Events processed: {result.events_processed}")
print(f"Messages sent: {result.metrics['messages_sent']}")
print(f"Messages delivered: {result.metrics['messages_delivered']}")
print(f"Delivery rate: {result.metrics['delivery_rate']:.2%}")
```

### Load Testing

```python
result = event_sim.simulate_load_test(
    graph,
    duration_ms=30000,
    initial_rate=10,        # Start at 10/sec
    peak_rate=500,          # Ramp to 500/sec
    ramp_time_ms=10000      # Over 10 seconds
)

print(f"Peak throughput: {result.metrics['peak_throughput']:.0f}/sec")
print(f"Latency P99: {result.metrics['latency_p99']:.2f} ms")
```

### Chaos Engineering

```python
result = event_sim.simulate_chaos(
    graph,
    duration_ms=30000,
    message_rate=100,
    failure_probability=0.01,    # 1% chance per check
    recovery_probability=0.1     # 10% chance to recover
)

print(f"Failures injected: {result.failures_injected}")
print(f"System uptime: {result.metrics['uptime']:.2%}")
```

---

## Implementation

### FailureSimulator Class

```python
from src.simulation import FailureSimulator

simulator = FailureSimulator(
    cascade_threshold=0.5,
    cascade_probability=0.7,
    max_cascade_depth=5,
    seed=42
)

# Single failure
result = simulator.simulate_failure(graph, "B1")

# Batch failure
result = simulator.simulate_batch_failure(graph, ["B1", "B2"])

# Exhaustive campaign
batch = simulator.simulate_all_failures(graph)

# Attack simulation
attack = simulator.simulate_attack(
    graph, 
    AttackStrategy.HIGHEST_BETWEENNESS, 
    count=3
)
```

### FailureResult Structure

```python
@dataclass
class FailureResult:
    simulation_id: str
    primary_failures: List[str]
    cascade_failures: List[Tuple[str, int]]  # (component, depth)
    impact: ImpactMetrics
    affected_paths: int
    affected_components: List[str]
    
@dataclass
class ImpactMetrics:
    impact_score: float
    reachability_loss: float
    fragmentation: float
    cascade_extent: float
    paths_lost: int
    total_affected: int
```

### SimulationGraph Class

```python
from src.simulation import SimulationGraph, ComponentType

# Load
graph = SimulationGraph.from_dict(data)

# Query
component = graph.get_component("B1")
apps = graph.get_components_by_type(ComponentType.APPLICATION)
paths = graph.get_message_paths()

# Modify (for simulation)
subgraph = graph.copy()
subgraph.remove_component("B1")
remaining_paths = subgraph.get_message_paths()
```

---

## Navigation

- **Previous:** [← Structural Analysis](analysis.md)
- **Next:** [Statistical Validation →](validation.md)
