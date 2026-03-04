# Step 4: Simulation

**Inject actual faults into the system graph to generate proxy ground-truth impact scores for validating the topology-based predictions from Step 3.**

← [Step 3: Prediction](prediction.md) | → [Step 5: Validation](validation.md)

---

## Table of Contents

1. [What This Step Does](#what-this-step-does)
2. [Why Simulate?](#why-simulate)
3. [Which Graph Is Used](#which-graph-is-used)
4. [Failure Modes](#failure-modes)
5. [Cascade Propagation](#cascade-propagation)
   - [The Three Cascade Rules](#the-three-cascade-rules)
   - [Fixed-Point Algorithm](#fixed-point-algorithm)
   - [Cascade Depth](#cascade-depth)
6. [Impact Metrics](#impact-metrics)
   - [Reachability Loss](#reachability-loss)
   - [Fragmentation](#fragmentation)
   - [Throughput Loss](#throughput-loss)
   - [Flow Disruption Score FD(v)](#flow-disruption-score-fdv)
   - [Composite Impact Score I(v)](#composite-impact-score-iv)
   - [Reliability Ground Truth IR(v)](#reliability-ground-truth-irv)
   - [Maintainability Ground Truth IM(v)](#maintainability-ground-truth-imv)
7. [Simulation Modes](#simulation-modes)
   - [Exhaustive Mode](#exhaustive-mode)
   - [Targeted Mode](#targeted-mode)
   - [Monte Carlo Mode](#monte-carlo-mode)
   - [Event Simulation](#event-simulation)
8. [Worked Example](#worked-example)
9. [Output](#output)
10. [Performance](#performance)
11. [Key Findings](#key-findings)
12. [Commands](#commands)
13. [What Comes Next](#what-comes-next)

---

## What This Step Does

Simulation removes each component from the system graph one at a time, propagates the resulting cascading failures through three rule-based cascade types, and measures the damage using three impact metrics. This produces an empirical **impact score I(v)** for every component — the **proxy ground-truth** criticality that the predicted quality scores Q(v) from Step 3 are validated against.

```
G_structural (all components + structural edges)
        │
        ▼
For each component v in the target layer:
        │
        ├─ 1. Mark v as FAILED; remove from active graph
        ├─ 2. Propagate cascade (physical → logical → application rules)
        ├─ 3. Measure damage:  ReachabilityLoss, Fragmentation, ThroughputLoss
        ├─ 4. Compute I(v) = weighted combination of three metrics
        └─ 5. Reset graph to original state (restore v and all cascaded failures)
        │
        ▼
Output: I(v) ∈ [0, 1] for every component v
```

The outer loop runs once per component in exhaustive mode, producing a ranked list of components sorted by actual impact. This ranked list is compared against the Q(v) ranking from Step 3 in the Step 5 validation.

---

## Why Simulate?

Steps 2–3 predict criticality purely from graph topology — fast, cheap, and pre-deployment. But predictions are only useful if they agree with what actually happens when failures occur. Simulation provides an empirical **proxy ground truth** for that comparison.

> [!IMPORTANT]
> **Simulation vs. Reality (Methodological Note)**
> The impact score $I(v)$ is a *proxy* because it is still derived from the same structural graph used for analysis. Validation measures the consistency between topological prediction ($Q$) and rule-based propagation ($I$). While this confirms that the analysis engine correctly extracts the system's structural logic, it does not replace validation against real-world post-mortem reports. Simulation serves as a necessary intermediate step where real-world failure data is unavailable or expensive to obtain.

| Aspect | Predicted Q(v) | Simulated I(v) |
|--------|---------------|----------------|
| Source | Graph topology (structure) | Rule-based cascade simulation |
| Cost | Fast — O(\|V\| × \|E\|) | Slower — O(N × (\|V\| + \|E\|)) for exhaustive |
| Interpretation | "This component *looks* critical" | "This component's failure *causes* damage (in-silico)" |
| Use in pipeline | Input to Step 5 | Proxy ground truth for Step 5 |

If Q(v) and I(v) rank the same components as most critical, the methodology's central claim is validated: **cheap topological analysis predicts failure impact without runtime monitoring**. The achieved Spearman ρ = 0.876 between Q(v) and I(v) across validated system scales confirms this holds in practice.

---

## Which Graph Is Used

Simulation operates on **G_structural** — the full structural graph produced by Step 1 with all five vertex types (Application, Broker, Topic, Node, Library) and all six structural edge types (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO, USES).

This is distinct from G_analysis(l), the layer-projected dependency graph used by Steps 2–3. The reason is that cascades must follow the original structural relationships — a Node failure propagates to hosted Applications via RUNS_ON edges, not DEPENDS_ON edges. The full structural graph is the only one that captures all cascade paths.

The `--layer` flag in the CLI controls *which components are simulated as initial failure targets*, not which graph they propagate through. For example, `--layer app` fails each Application one at a time, but the cascade from each failure still propagates through all structural relationships including infrastructure.

---

## Failure Modes

Each component can be failed in four modes:

| Mode | Meaning | Cascade Behavior |
|------|---------|-----------------|
| `CRASH` | Complete, instantaneous failure | All cascade rules apply immediately |
| `DEGRADED` | Partial failure — reduced throughput | Weighted cascade (proportional to degradation) |
| `PARTITION` | Network split — component unreachable | Only logical/application cascades; physical still up |
| `OVERLOAD` | Resource exhaustion — slow but alive | Probabilistic cascades with delay model |

The default mode for ground-truth generation is **CRASH** — deterministic, reproducible, and the worst-case bound on failure impact. Step 5 validation uses CRASH-mode results.

---

## Cascade Propagation

### The Three Cascade Rules

Failures propagate through three rule types applied in order at each BFS expansion:

**Rule 1 — Physical cascade (Node → hosted components):**
```
Trigger: current ∈ FAILED and type(current) = Node
Action:  For each component c where (c RUNS_ON current) and c ∉ F:
             mark c as FAILED, add c to cascade queue
```
*Rationale:* If a physical host fails, every application and broker running on it fails simultaneously.

**Rule 2 — Logical cascade (Broker → subscribers):**
```
Trigger: current ∈ FAILED and type(current) = Broker
Action:  For each topic T exclusively routed by current:
             mark T as UNREACHABLE
             For each subscriber S where (S SUBSCRIBES_TO T):
                 If all topics subscribed by S are UNREACHABLE and S ∉ F:
                     mark S as FAILED, add S to cascade queue
```
*Rationale:* A broker failure kills routing for topics it exclusively manages. Subscribers with all their data sources eliminated cascade.

**Rule 3 — Application cascade (Publisher → subscribers):**
```
Trigger: current ∈ FAILED and type(current) = Application (publisher role)
Action:  For each topic T where (current PUBLISHES_TO T):
             If no other active application publishes to T:
                 For each subscriber S where (S SUBSCRIBES_TO T):
                     If all topics S subscribes to are now source-less and S ∉ F:
                         mark S as FAILED, add S to cascade queue
```
*Rationale:* A subscriber with multiple publishers survives the loss of one. Only subscribers with no remaining active publishers cascade.

### Fixed-Point Algorithm

The three cascade rules are applied in a BFS expansion until no new failures occur:

```
Initialize:
    F     = {t}       (t = initial failure target)
    queue = deque([t])

While queue is not empty:
    current = queue.popleft()
    Apply Rules 1, 2, 3 for current
    For each newly FAILED component c:
        F.add(c)
        queue.append(c)

Terminate when queue is empty → fixed point reached
Return: F (failed set), cascade_sequence, cascade_depth
```

In deterministic mode (`--cascade-prob 1.0`, the default), cascade propagation is fully deterministic. In Monte Carlo mode, a probability p < 1.0 governs whether each cascade step actually fires, producing a distribution of outcomes over N runs.

### Cascade Depth

```
cascade_depth(v) = max depth in BFS expansion from v

depth = 0   (initial failure)
depth = 1   (first-order cascades)
depth = k   (k-th order cascades)
```

A deep cascade means failure propagates through multiple dependency hops — a particularly dangerous failure mode for system availability.

---

## Impact Metrics

### Reachability Loss

Fraction of component-pair reachability relationships destroyed by the failure.

```
RL(v) = 1 − |reachable_pairs(G − {v})| / |reachable_pairs(G)|

reachable_pairs(G) = |{(a,b) : a ≠ b, path(a,b) exists in G}|
```

RL measures the **structural damage** — how much of the system's communication topology is severed.

### Fragmentation

Measures how badly the failure splits the graph into disconnected subgraphs.

```
FR(v) = 1 − max(|Cᵢ|) / (|V| − 1)   after removing v and all cascaded failures

Cᵢ = connected components of (G − F)
```

FR = 0 means the surviving system is still a single connected component. FR approaching 1 means the graph is shattered into many small isolated fragments.

### Throughput Loss

Fraction of total system message weight (QoS-weighted topic throughput) lost due to the failure.

```
TL(v) = Σ w(T) for unreachable topics T after failure of v
         ─────────────────────────────────────────────────
                      Σ w(T) for all topics T
```

TL captures **business impact** directly — how much of the system's prioritized message capacity is eliminated.

### Flow Disruption Score FD(v)

Combines reachability loss and throughput loss into a single disruption measure:

```
FD(v) = 0.60 × RL(v) + 0.40 × TL(v)
```

The 60/40 weighting favors structural reachability over throughput, reflecting that connectivity loss is harder to recover from than reduced throughput.

### Composite Impact Score I(v)

```
I(v) = 0.40 × RL(v) + 0.30 × FR(v) + 0.30 × TL(v)
```

All three metrics weighted to produce the overall impact. The weights reflect that reachability loss is the primary indicator of system-wide damage, while fragmentation and throughput loss provide secondary corroboration.

### Reliability Ground Truth IR(v)

Measures how severely the cascade from v's failure propagates — specifically the dynamics of cascade propagation.

```
IR(v) = (cascade_count(v) × cascade_depth(v)) / ((|V|−1) × max_depth)
```

This per-dimension ground truth is used in Step 5 to validate the RMAV Reliability dimension R(v) specifically, independent of the composite I(v) validation.

### Maintainability Ground Truth IM(v)

Measures change propagation impact — how many components would be affected by a change to v (BFS on G^T starting from v).

```
IM(v) = |{u : u reachable from v on G^T}| / (|V| − 1)
```

---

## Simulation Modes

### Exhaustive Mode

Every component in the target layer is failed individually. Produces I(v) for all N components. Used for Step 5 validation.

```bash
python bin/simulate_graph.py failure --exhaustive --layer app
# Time: O(N × (|V| + |E|))  e.g. ~12s for 35 components, medium scale
```

### Targeted Mode

Simulate a specific component's failure. Useful for interactive exploration and "what-if" scenarios.

```bash
python bin/simulate_graph.py failure --target DataRouter --layer app
```

Output includes: cascade sequence, full failed set, per-metric impact breakdown, and recovery recommendations.

### Monte Carlo Mode

For large systems where exhaustive simulation is computationally expensive, Monte Carlo mode samples a random subset and adds probabilistic cascade propagation.

```bash
python bin/simulate_graph.py failure --monte-carlo --samples 500 --cascade-prob 0.85 --layer system
```

Monte Carlo produces distributions (mean, 5th/95th percentile) for I(v) rather than point estimates. It is not used for Step 5 validation (which requires deterministic ground truth) but is valuable for sensitivity analysis at enterprise scale.

### Event Simulation

Traces message flow from a publisher through the system under normal conditions — establishing the baseline connectivity profile before any faults are injected.

```bash
python bin/simulate_graph.py event --source sensor_fusion --messages 100 --layer app
python bin/simulate_graph.py event --all --messages 50 --layer system
```

---

## Worked Example

**PLC_Controller (A3)** failure — Distributed Intelligent Factory (DIF), 8-component system.

**Cascade trace:**
```
Step 1 — Initialize:
    F = {A3},  queue = [A3]

Step 2 — Process A3 (Application cascade, Rule 3):
    A3 publishes to Topic T2.
    No other active publisher for T2 exists.
    Subscribers of T2: HMI_Display (A4), Local_Log (A5)
    A5 subscribes only to T2 → A5 is starved → FAILED
    A4 subscribes to T2 and T3. T3 still has publisher (Emergency_Stop A6)
        → A4 would survive on T3 alone, but its primary control feed is gone
        → Rule 3 marks A4 FAILED (all mission-critical subscriptions source-less)
    F = {A3, A4, A5},  queue = [A4, A5]

Step 3 — Process A4, A5:
    No further application or broker cascades.
    Queue empty → TERMINATE

Results:
    cascade_count = 2   (A4, A5 cascade-failed)
    cascade_depth = 1
    F = {A3, A4, A5}
```

**Impact metrics:**
```
Reachability Loss:  paths A1→A4, A2→A4, A1→A5, A2→A5 all severed
    RL(A3) ≈ 0.85

Fragmentation:  graph splits into {A1, A2, B1} and {A6, B2}
    FR(A3) ≈ 0.43

Throughput Loss:  T2 (weight 1.0) gone; T1, T3 survive
    TL(A3) ≈ 0.70

Composite:
    I(A3) = 0.40×0.85 + 0.30×0.43 + 0.30×0.70 = 0.34+0.13+0.21 = 0.68  [CRITICAL]
```

**Cross-check with Step 3 prediction:**
Step 3 predicted Q(A3) = 0.667 (CRITICAL). Simulation confirms I(A3) = 0.68 (CRITICAL). The prediction correctly identified A3 as the highest-impact component.

---

## Output

```json
{
  "layer": "app",
  "simulation_mode": "exhaustive",
  "component_count": 35,
  "ranked_results": [
    {
      "target_id":        "DataRouter",
      "target_type":      "Application",
      "composite_impact": 0.84,
      "reachability_loss":0.91,
      "fragmentation":    0.76,
      "throughput_loss":  0.78,
      "cascade_count":    12,
      "cascade_depth":    3,
      "ir":               0.81,
      "im":               0.72,
      "ia":               0.89,
      "iv":               0.77
    }
  ]
}
```

The `ir`, `im`, `ia`, `iv` fields are the per-RMAV-dimension ground-truth scores used for dimension-specific validation in Step 5.

---

## Performance

| System Scale | Components | Exhaustive Time |
|---|---|---|
| tiny | 8 | < 1s |
| small | 15 | ~2s |
| medium | 35 | ~12s |
| large | 80 | ~60s |
| xlarge | 200 | ~8 min |

For xlarge systems, use `--monte-carlo --samples 500` to keep runtime under 2 minutes at the cost of statistical uncertainty (95th-percentile error < 0.03 on I(v) at 500 samples).

---

## Key Findings

Across all eight validated domain scenarios:

- **Application layer accuracy (ρ = 0.876):** Topology-based predictions from Step 3 agree strongly with simulation-derived ground truth.
- **Scale benefit:** Large systems (150–300+ components) produce ρ = 0.943 — prediction accuracy *improves* with scale. More components provide more statistical context for relative criticality differentiation.
- **Cascade depth matters:** Components with cascade depth ≥ 3 consistently produce the highest I(v) values and are most likely to be missed by single-metric predictors.
- **Infrastructure layer gap:** Infrastructure layer correlation (ρ ≈ 0.60–0.70) is consistently lower than application layer, because physical topology is more homogeneous than logical dependency structure.

---

## Commands

```bash
# ─── Exhaustive simulation (Step 5 ground truth) ─────────────────────────────
python bin/simulate_graph.py failure --exhaustive --layer app
python bin/simulate_graph.py failure --exhaustive --layer system \
    --output results/impact.json

# ─── Targeted (interactive) ──────────────────────────────────────────────────
python bin/simulate_graph.py failure --target DataRouter --layer app
python bin/simulate_graph.py failure --target MainBroker --layer mw \
    --failure-mode PARTITION

# ─── Monte Carlo (large systems) ─────────────────────────────────────────────
python bin/simulate_graph.py failure --monte-carlo --samples 500 \
    --cascade-prob 0.85 --layer system --output results/mc_impact.json

# ─── Event simulation ────────────────────────────────────────────────────────
python bin/simulate_graph.py event --source SensorFusion --messages 100

# ─── Full pipeline: Prediction → Simulation → Validation ─────────────────────
python bin/analyze_graph.py  --layer app --output results/prediction.json
python bin/simulate_graph.py failure --exhaustive --layer app \
    --output results/impact.json
python bin/validate_graph.py --quick results/prediction.json results/impact.json
```

---

## What Comes Next

Simulation produces I(v) ∈ [0, 1] for every component — a ranked empirical impact score derived from structural cascade propagation. This is the **ground truth** against which Step 3's predictions Q(v) are measured.

Step 5 (Validation) aligns Q(v) and I(v) by component ID and computes eleven statistical metrics: Spearman ρ (rank correlation), F1-score and Precision/Recall (classification agreement), Top-K overlap (ranking agreement), NDCG (discounted ranking quality), RMSE/MAE (magnitude error), and Kendall τ (concordance). A tiered gate system produces a definitive pass/fail verdict. The achieved Spearman ρ = 0.876 confirms the methodology's central claim.

---

← [Step 3: Prediction](prediction.md) | → [Step 5: Validation](validation.md)