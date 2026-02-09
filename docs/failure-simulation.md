# Step 4: Failure Simulation

**Inject actual faults into the system graph to measure real impact — this is the ground truth.**

← [Step 3: Quality Scoring](quality-scoring.md) | → [Step 5: Validation](validation.md)

---

## What This Step Does

Failure Simulation removes each component from the graph one at a time, propagates the cascading effects, and measures how much damage results. This produces an empirical impact score I(v) for every component — the "actual" criticality that our predicted scores Q(v) should correlate with.

```
For each component v:
    1. Remove v from G_structural
    2. Propagate cascading failures
    3. Measure impact (reachability loss, fragmentation, throughput drop)
    4. Record I(v)
    5. Restore v
```

## Why Simulate?

Steps 2–3 predict criticality from graph topology. But predictions are only useful if they match reality. Simulation gives us ground truth to validate against:

| Predicted Q(v) | Simulated I(v) |
|---------------|----------------|
| Based on graph structure | Based on actual failure effects |
| Fast to compute | Slower but empirically grounded |
| "This component *looks* important" | "This component's failure *actually* causes damage" |

If Q(v) and I(v) agree, we've proven that cheap topological analysis can replace expensive failure testing.

## Cascade Propagation

When a component fails, the effects don't stop there. The simulation models three types of cascading failures:

**Infrastructure cascades**: When a Node fails, everything hosted on it (applications, brokers) also fails.

**Broker cascades**: When a Broker fails, topics it exclusively routes become unavailable. Applications that depend solely on those topics lose connectivity.

**Application cascades**: When a publisher fails, subscribers that have no other source for that data are effectively starved.

### Cascade Depth

Cascades propagate iteratively until no new failures occur. Each round, the simulation checks whether any surviving component has lost all its critical dependencies. If so, it too fails, and the next round begins. The cascade depth (number of rounds) reveals whether a failure causes a shallow, contained disruption or a deep, system-wide collapse.

## Impact Metrics

After cascading, three metrics quantify the damage:

| Metric | What It Measures | Formula |
|--------|-----------------|---------|
| **Reachability Loss** | What fraction of previously reachable component pairs are now disconnected | 1 − (reachable pairs after) / (reachable pairs before) |
| **Fragmentation** | How badly the graph has broken apart | 1 − |largest component after| / |V − 1| |
| **Throughput Drop** | Loss of message-passing capacity, weighted by topic importance | 1 − (total surviving topic weight) / (total original topic weight) |

These combine into the composite impact score:

```
I(v) = α × ReachabilityLoss + β × Fragmentation + γ × ThroughputDrop
```

## Simulation Modes

| Mode | What It Does | When to Use |
|------|-------------|-------------|
| **Exhaustive** | Simulate failure of every component | Full validation (default) |
| **Targeted** | Simulate failure of specific components | Quick checks |
| **Monte Carlo** | Randomized cascade propagation with probability parameter | Non-deterministic systems (retries, circuit breakers) |

In Monte Carlo mode, each cascade step propagates with a configurable probability, and multiple trials produce a distribution of I(v) values with confidence intervals.

## Key Findings

Empirical observations from simulation across multiple system scales:

- **Broker criticality**: Brokers that exclusively route high-weight topics produce disproportionately high impact scores, because their failure cuts off critical data flows.
- **Infrastructure cascades are the deepest**: Node failures trigger all cascade types simultaneously (infrastructure → broker → application), often causing the most severe system-wide damage.
- **SPOF detection**: Components with I(v) > 0.5 are empirical single points of failure. Cross-referencing with Q(v) > 0.5 identifies components where both topology and simulation agree on criticality.
- **Layer differences**: Application layer predictions correlate better with simulation (ρ ≈ 0.85) than infrastructure (ρ ≈ 0.75), because application dependencies are more directly captured by the derivation rules.

## Commands

```bash
# Exhaustive simulation (every component)
python bin/simulate_graph.py failure --layer system --exhaustive

# Targeted simulation (specific components)
python bin/simulate_graph.py failure --layer system --target sensor_fusion,main_broker

# Monte Carlo mode (100 trials, 80% cascade probability)
python bin/simulate_graph.py failure --layer system --exhaustive --monte-carlo --trials 100 --cascade-prob 0.8

# Export results
python bin/simulate_graph.py failure --layer system --exhaustive --output results/impact.json
```

## Performance

Baseline computation (initial paths, components, topic weights) runs once and is reused across all simulations. For a system with N components, exhaustive simulation complexity is O(N × (V + E)) rather than O(N × P) where P is path enumeration cost.

## What Comes Next

We now have two sets of scores: predicted Q(v) from Steps 2–3 and simulated I(v) from this step. Step 5 statistically compares them to answer: **do our predictions actually work?**

---

← [Step 3: Quality Scoring](quality-scoring.md) | → [Step 5: Validation](validation.md)