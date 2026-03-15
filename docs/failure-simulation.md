# Step 4: Simulation

**Inject faults into the system graph to generate proxy ground-truth impact scores for validating the topology-based predictions from Step 3.**

← [Step 3: Prediction](prediction.md) | → [Step 5: Validation](validation.md)

---

## Table of Contents

1. [What This Step Does](#what-this-step-does)
2. [Why Simulate?](#why-simulate)
3. [Independence Guarantee](#independence-guarantee)
4. [Which Graph Each Simulator Uses](#which-graph-each-simulator-uses)
5. [Execution Architecture](#execution-architecture)
6. [Failure Modes](#failure-modes)
7. [Cascade Propagation](#cascade-propagation)
   - [Rule 1 — Physical Cascade](#rule-1--physical-cascade)
   - [Rule 2 — Logical Cascade](#rule-2--logical-cascade)
   - [Rule 3 — Application Cascade](#rule-3--application-cascade)
   - [Rule 4 — Library Cascade (new)](#rule-4--library-cascade)
   - [Fixed-Point Algorithm](#fixed-point-algorithm)
8. [Ground-Truth Metrics](#ground-truth-metrics)
   - [Base Metrics](#base-metrics)
   - [I(v) — Composite Impact Score](#iv--composite-impact-score)
   - [IR(v) — Reliability Ground Truth](#irv--reliability-ground-truth)
   - [IM(v) — Maintainability Ground Truth](#imv--maintainability-ground-truth)
   - [IA(v) — Availability Ground Truth](#iav--availability-ground-truth)
   - [IV(v) — Vulnerability Ground Truth](#ivv--vulnerability-ground-truth)
   - [Ground-Truth Summary](#ground-truth-summary)
9. [Simulation Modes](#simulation-modes)
10. [Worked Example](#worked-example)
11. [Output Schema](#output-schema)
12. [Performance](#performance)
13. [Key Findings](#key-findings)
14. [Commands](#commands)
15. [What Comes Next](#what-comes-next)

---

## What This Step Does

Simulation removes each component from the system graph one at a time, propagates cascading failures through four structural cascade rules, measures damage across five dimensions, and packages the result as an empirical impact score. The output is five per-component ground-truth scalars — I(v), IR(v), IM(v), IA(v), IV(v) — one for each RMAV dimension plus the composite.

```
G_structural (all components + 6 structural edge types)
        │
        ▼  ── STEP A: Run event simulation to establish baseline flows FD(v)
        │
        ▼  ── STEP B: Main exhaustive loop (one iteration per component v)
        │       1. Mark v as FAILED; remove from active graph
        │       2. Propagate cascade (Rules 1–4)
        │       3. Measure RL(v), FR(v), TL(v), FD(v) → store in ImpactMetrics
        │       4. Reset graph to original state
        │
        ▼  ── STEP C: Four post-passes (each independent of Q(v))
        │       Post-pass 1: IR(v) — cascade dynamics from main loop data
        │       Post-pass 2: IM(v) — change propagation on G^T (G_analysis)
        │       Post-pass 3: IA(v) — QoS-weighted connectivity on G_structural
        │       Post-pass 4: IV(v) — compromise propagation on G^T (G_analysis)
        │
        ▼
ImpactMetrics per component: I(v), IR(v), IM(v), IA(v), IV(v)
```

---

## Why Simulate?

Steps 2–3 predict criticality purely from graph topology — fast, cheap, and pre-deployment. But predictions are only useful if they agree with what actually happens when failures occur. Simulation provides an empirical **proxy ground truth** for that comparison.

> **Simulation vs. Reality (Methodological Note):** I(v) is a *proxy* because it is derived from the same structural graph used for analysis. Validation measures consistency between topological prediction Q(v) and rule-based propagation I(v). While this confirms the analysis engine correctly extracts the system's structural logic, it does not replace validation against real post-mortem reports. Simulation is the necessary intermediate step when runtime failure data is unavailable.

| Aspect | Predicted Q(v) | Simulated I(v) |
|--------|---------------|----------------|
| Source | Graph topology (structure only) | Rule-based cascade on G_structural |
| Cost | Fast — O(\|V\| × \|E\|) | Slower — O(N × (\|V\| + \|E\|)) exhaustive |
| Meaning | "This component *looks* critical" | "This component's failure *causes* damage (in-silico)" |
| Role in pipeline | Input to Step 5 | Proxy ground truth for Step 5 |

If Q(v) and I(v) rank the same components as most critical, the central claim is validated: **cheap topology-only analysis predicts failure impact without runtime monitoring**. The achieved Spearman ρ = 0.876 confirms this holds in practice.

---

## Independence Guarantee

**This is the most important methodological constraint in the entire pipeline.**

```
HARD RULE: I(v) must not read Q(v), R(v), M(v), A(v), V(v), or any
           field derived from Step 3 (Prediction) as an input.
```

The independence guarantee is what makes Step 5 validation scientifically meaningful. If simulation used prediction scores to guide cascade propagation or to weight impact metrics, the correlation between Q(v) and I(v) would be circular — it would measure how consistently the method agrees with itself, not whether topology predicts impact.

**What simulation IS allowed to use:**
- G_structural (the raw structural graph from Step 1)
- G_analysis as G^T (the DEPENDS_ON graph from Step 1, reversed, for IM and IV)
- QoS edge weights (from Step 1 Phase 4 weight assignment)
- Component type information (vertex type τ_V(v))

**What simulation is NOT allowed to use:**
- M(v) (metric vector from Step 2)
- Q(v), R(v), M(v), A(v), V(v) (prediction scores from Step 3)
- CDPot_enh, MPCI, FOC, or any derived term from Step 3

Simulation and prediction are two **independent views** of the same graph. Step 5 measures the agreement between these views.

---

## Which Graph Each Simulator Uses

Different simulators require different graph projections:

| Simulator | Graph Used | Justification |
|-----------|-----------|---------------|
| **Main loop (cascade)** | G_structural | Cascades follow physical structural edges (RUNS_ON, ROUTES, PUBLISHES_TO, USES) — not abstract DEPENDS_ON edges |
| **IR(v) post-pass** | Main loop data | Derived from cascade_count and cascade_depth collected during the main loop |
| **IM(v) post-pass** | G_analysis reversed (G^T) | Change propagation follows *dependency direction reversed* — if A depends on B, a change to B may force A to adapt |
| **IA(v) post-pass** | G_structural | Connectivity disruption requires the full structural graph to measure QoS-weighted path loss |
| **IV(v) post-pass** | G_analysis reversed (G^T) | Compromise propagates from dependency to dependent (reverse direction of DEPENDS_ON) |

> **G^T construction for IM and IV:** If edge `(A, B) ∈ DEPENDS_ON` (A depends on B), then G^T contains edge `(B, A)` — meaning "a change or compromise at B can reach A." This is the standard dependency inversion for change propagation analysis.

---

## Execution Architecture

The simulation step runs in five sequential stages. Each stage is independent from Q(v).

**Stage A — Event simulation (prerequisite for FD(v)):**
Run discrete-event message flow simulation from all publishers. This establishes the set of baseline flows — (publisher, topic, subscriber) triples that succeed under normal conditions. These flows are stored and used in Stage B to compute FD(v): the fraction of baseline flows that are disrupted when v fails.

**Stage B — Main exhaustive loop:**
For each component v in the target layer, in O(|V| + |E|) per iteration:
1. Remove v from the active graph
2. Run cascade propagation (Rules 1–4) to fixed point
3. Measure RL(v), FR(v), TL(v), FD(v)
4. Record cascade_count(v), cascade_depth(v), cascade_sequence(v)
5. Restore graph to original state

**Stage C — Four post-passes (run once, over all N results):**

Post-pass 1 (IR): Normalize cascade_count and cascade_depth across all N results to produce CascadeReach, WeightedCascadeImpact, NormalizedCascadeDepth.

Post-pass 2 (IM): Run ChangePropagationSimulator on G^T for all N components. For each v, BFS from v on G^T collecting components that must adapt, stopping at loose-coupling and stable-interface boundaries.

Post-pass 3 (IA): For each result, compute QoS-weighted reachability loss and fragmentation from the main loop's structural damage data.

Post-pass 4 (IV): Run CompromisePropagationSimulator on G^T for all N components. For each v, BFS from v on G^T collecting reachable components above the trust threshold θ_trust = 0.30.

Post-passes 2 and 4 use G^T and are fully independent from the main loop — they do not re-simulate failures, they analyze propagation paths over a different graph projection.

---

## Failure Modes

| Mode | Meaning | Cascade Behavior |
|------|---------|-----------------|
| `CRASH` | Complete, instantaneous failure | All four cascade rules apply immediately |
| `DEGRADED` | Partial failure — reduced throughput | Weighted cascade; starvation threshold SL < 0.30 |
| `PARTITION` | Network split — component unreachable | Logical and application cascades only; physical still up |
| `OVERLOAD` | Resource exhaustion — slow but alive | Probabilistic cascades with delay model |

The default mode for ground-truth generation is **CRASH** — deterministic, reproducible, worst-case bound. Step 5 validation always uses CRASH-mode results.

---

## Cascade Propagation

Cascade propagation runs as a BFS from the initially failed component v, applying the four rules at each dequeued component until no new failures occur (fixed point).

### Rule 1 — Physical Cascade

```
Trigger: current ∈ F AND τ_V(current) = Node
Action:  For each component c where (c −[RUNS_ON]→ current) AND c ∉ F:
             F.add(c);  queue.append(c)
```

**Rationale:** A physical host failure takes down all co-located applications and brokers simultaneously. This is a broadcast failure — the number of hosted components is the immediate blast radius.

### Rule 2 — Logical Cascade

```
Trigger: current ∈ F AND τ_V(current) = Broker
Action:  For each topic T where (current −[ROUTES]→ T):
             If no other active broker routes T:
                 mark T as UNREACHABLE
                 For each subscriber S where (S −[SUBSCRIBES_TO]→ T):
                     If ALL topics subscribed by S are UNREACHABLE AND S ∉ F:
                         F.add(S);  queue.append(S)
```

**Rationale:** A broker failure makes all exclusively-routed topics unreachable. Subscribers that have lost all their data sources — no alternative topic remains — are starved and cascade. Subscribers with at least one surviving topic survive this rule.

### Rule 3 — Application Cascade

```
Trigger: current ∈ F AND τ_V(current) = Application (publisher role)
Action:  For each topic T where (current −[PUBLISHES_TO]→ T):
             If NO other active application publishes to T:
                 For each subscriber S where (S −[SUBSCRIBES_TO]→ T):
                     If ALL topics subscribed by S are now source-less AND S ∉ F:
                         F.add(S);  queue.append(S)
```

**Rationale:** A publisher failure orphans its topics only when no other active publisher exists. Subscribers with multiple publishers survive the loss of one. This rule encodes the pub-sub redundancy model: N-to-1 fan-in on a topic provides publisher fault tolerance.

### Rule 4 — Library Cascade

```
Trigger: current ∈ F AND τ_V(current) = Library
Action:  For each application A where (A −[USES]→ current) AND A ∉ F:
             F.add(A);  queue.append(A)
```

**Rationale:** A library failure produces a **simultaneous blast** — all consuming applications crash at once, immediately, because the library is a loaded in-process dependency. This is fundamentally different from Rules 1–3: there is no partial survival, no topic-level redundancy, and no host-level isolation. Every application that has loaded the library fails. After applications in F cascade, they are processed by Rules 2 and 3, potentially triggering further subscriber starvation.

> **Library vs. Application cascade semantics:** Rule 3 (Application cascade) can be avoided if a topic has multiple publishers. Rule 4 (Library cascade) cannot — there is no "multiple library" redundancy mechanism in standard pub-sub systems. A library used by 15 applications has an immediate blast radius of 15.

### Fixed-Point Algorithm

```python
F     = {v}         # initially failed set
queue = deque([v])  # BFS frontier

while queue:
    current = queue.popleft()
    apply Rule 1 if τ_V(current) = Node
    apply Rule 2 if τ_V(current) = Broker
    apply Rule 3 if τ_V(current) = Application (publisher)
    apply Rule 4 if τ_V(current) = Library

    for each newly failed component c:
        F.add(c)
        queue.append(c)

# Terminates when no new failures are discovered
cascade_count = |F| - 1    (excludes the initial failure v)
cascade_depth = max BFS depth reached
```

Termination is guaranteed because the graph is finite and components can only transition from ACTIVE → FAILED (never back). The BFS explores at most |V| − 1 additional components.

**Cascade depth:**
```
depth = 0   → initial failure (v)
depth = 1   → directly triggered failures
depth = k   → k-th hop in the BFS expansion
```

---

## Ground-Truth Metrics

### Base Metrics

These are measured directly in the main loop for each component v:

**Reachability Loss RL(v):**
```
RL(v) = 1 − |reachable_pairs(G \ F)| / |reachable_pairs(G)|

reachable_pairs(H) = |{(a,b) : a ≠ b, path(a→b) exists in H}|
```
Measures the fraction of component-to-component path connectivity destroyed by the failure and its cascade. RL = 0 means the surviving graph is fully connected; RL = 1 means all reachability is lost.

**Fragmentation FR(v):**
```
FR(v) = 1 − max(|Cᵢ|) / (|V| − 1)    over connected components Cᵢ of G \ F
```
Measures how badly the failure shatters the graph. FR = 0 means the survivors form one connected component; FR → 1 means the graph is shattered into many isolated fragments.

**Throughput Loss TL(v):**
```
TL(v) = Σ w(T) for topics T unreachable after F / Σ w(T) for all topics T
```
Measures the fraction of QoS-weighted message capacity eliminated. w(T) is the topic weight from Step 1 Phase 4.

**Flow Disruption FD(v):**
```
FD(v) = |{(pub, topic, sub) ∈ baseline_flows : pub ∈ F OR topic unreachable}|
         / |baseline_flows|
```
Measures what fraction of the event-simulation baseline flows are interrupted. Requires Stage A (event simulation) to run first to establish `baseline_flows`. FD = 0 if no baseline flows are disrupted; FD = 1 if all are.

---

### I(v) — Composite Impact Score

```
I(v) = 0.35 × RL(v) + 0.25 × FR(v) + 0.25 × TL(v) + 0.15 × FD(v)
```

| Term | Weight | Rationale |
|------|--------|-----------|
| RL(v) | 0.35 | Reachability loss is the primary indicator of system-wide damage |
| FR(v) | 0.25 | Fragmentation captures long-term partition effects beyond path loss |
| TL(v) | 0.25 | Throughput loss measures business-level impact via QoS weights |
| FD(v) | 0.15 | Flow disruption grounds the score in observed message delivery, not just topology |

> **On FD(v) weight 0.15:** FD depends on Stage A event simulation completing first. When event simulation is skipped (e.g., `--no-event`), FD(v) = 0 and the I(v) formula effectively renormalizes: `I(v) ≈ (0.35·RL + 0.25·FR + 0.25·TL) / 0.85`. The pipeline handles this gracefully without requiring a separate code path.

---

### IR(v) — Reliability Ground Truth

IR(v) measures fault propagation dynamics — how rapidly and broadly v's failure cascades. It is the ground truth that validates R(v) from Step 3.

```
IR(v) = 0.45 × CascadeReach(v) + 0.35 × WeightedCascadeImpact(v) + 0.20 × NormalizedCascadeDepth(v)
```

| Sub-metric | Definition |
|------------|-----------|
| CascadeReach(v) | `cascade_count(v) / (|V| − 1)` — fraction of all other components that cascade-failed |
| WeightedCascadeImpact(v) | `Σ w(c) for c ∈ cascaded_failures(v) / Σ w(all)` — importance-weighted cascade breadth |
| NormalizedCascadeDepth(v) | `cascade_depth(v) / max_depth` — relative depth across all simulation runs |

**IR(v) vs. IA(v) orthogonality:** IR(v) measures *propagation dynamics* (how the failure spreads step by step through the cascade). IA(v) measures *structural connectivity loss* (how removing v changes the graph's reachability structure). A component can have high IR but low IA (a publisher that starves many subscribers but does not partition the graph) or high IA but low IR (an articulation point that disconnects the graph but has no cascade chain). This orthogonality is intentional and mirrors the R/A distinction in the RMAV prediction model.

---

### IM(v) — Maintainability Ground Truth

IM(v) measures development-time change propagation — how many components would need to adapt if v's interface changed. It is computed by Post-pass 2 using G^T (reversed DEPENDS_ON graph) and is the ground truth that validates M(v).

```
IM(v) = 0.45 × ChangeReach(v) + 0.35 × WeightedChangeImpact(v) + 0.20 × NormalizedChangeDepth(v)
```

| Sub-metric | Definition |
|------------|-----------|
| ChangeReach(v) | Fraction of reachable components on G^T (with stop conditions) |
| WeightedChangeImpact(v) | Importance-weighted adaptation cost |
| NormalizedChangeDepth(v) | Relative BFS depth normalized across all components |

**BFS stop conditions on G^T:**

```
Stop propagation at edge (v → u) on G^T if EITHER:
  (a) edge_weight < θ_loose = 0.20    (loose coupling — dependent absorbs the change)
  (b) Instability(u) < θ_stable = 0.20  (stable interface — u has many afferent dependents
                                           and few efferent dependencies; it absorbs changes)
```

**Why BFS on G^T, not G_structural:** IM(v) models *change propagation*, not *runtime failure cascade*. If `A DEPENDS_ON B` (A depends on B), a change to B's interface may force A to adapt — so the change propagates from B to A, which is the reverse of the DEPENDS_ON edge. G^T is precisely the "change propagation graph" over the DEPENDS_ON structure.

**Why stop conditions:** Not all downstream dependencies propagate changes. A low-weight (BEST_EFFORT, VOLATILE) dependency means the dependent is loosely contracted and can absorb the change without modification. A stable component (many dependents, few of its own dependencies) acts as an architectural boundary that absorbs change obligations.

---

### IA(v) — Availability Ground Truth

IA(v) measures QoS-weighted structural connectivity disruption — how much of the high-priority connectivity structure is lost when v is removed. It is the ground truth that validates A(v).

```
IA(v) = 0.50 × WeightedReachabilityLoss(v) + 0.35 × WeightedFragmentation(v) + 0.15 × PathBreakingThroughputLoss(v)
```

| Sub-metric | Definition |
|------------|-----------|
| WeightedReachabilityLoss(v) | RL(v) weighted by the QoS importance of lost paths |
| WeightedFragmentation(v) | FR(v) weighted by the criticality of isolated fragments |
| PathBreakingThroughputLoss(v) | Throughput lost specifically via PARTITION_LOSS events (structural path breaks, not cascade starvation) |

**IA(v) vs. IR(v) orthogonality:** PathBreakingThroughputLoss is strictly the throughput lost because paths are structurally severed — it excludes throughput lost because publishers cascaded and starved subscribers. This separation ensures IR(v) and IA(v) measure complementary phenomena rather than the same effect with different weights.

---

### IV(v) — Vulnerability Ground Truth

IV(v) measures adversarial compromise propagation — how far a compromise at v would spread through the trusted dependency graph. It is computed by Post-pass 4 using G^T and is the ground truth that validates V(v).

```
IV(v) = 0.40 × AttackReach(v) + 0.35 × WeightedAttackImpact(v) + 0.25 × HighValueContamination(v)
```

| Sub-metric | Definition |
|------------|-----------|
| AttackReach(v) | Fraction of components reachable from v on G^T above trust threshold |
| WeightedAttackImpact(v) | Importance-weighted sum of contaminated components |
| HighValueContamination(v) | Distance-discounted sum of high-importance components reached |

**Trust threshold θ_trust = 0.30:** A compromise propagates along G^T edges only when the edge weight exceeds 0.30 (i.e., the dependency is at least TRANSIENT or has some Priority weight). Low-QoS BEST_EFFORT / VOLATILE edges are not trusted as compromise vectors — they represent loose, unguaranteed dependencies that would not carry automated credential or state propagation.

**Why G^T for IV(v):** If `A DEPENDS_ON B` (A depends on B, meaning B is a dependency of A), then compromising B can affect A. In G^T, this is an edge `B → A`, so BFS from B on G^T reaches A. This is the correct direction for adversarial propagation: the attack starts at the compromised component and spreads outward to everything that trusts it.

---

### Ground-Truth Summary

All five ground-truth scores and their RMAV correspondences:

| Score | Formula | Measures | Validates | Graph |
|-------|---------|---------|-----------|-------|
| I(v) | 0.35·RL + 0.25·FR + 0.25·TL + 0.15·FD | Overall structural damage | Q(v) composite | G_structural |
| IR(v) | 0.45·CascadeReach + 0.35·WCI + 0.20·NCD | Cascade propagation dynamics | R(v) | Main loop data |
| IM(v) | 0.45·ChangeReach + 0.35·WChI + 0.20·NChD | Change propagation reach | M(v) | G^T (DEPENDS_ON reversed) |
| IA(v) | 0.50·WRL + 0.35·WFR + 0.15·PBTL | QoS-weighted connectivity loss | A(v) | G_structural |
| IV(v) | 0.40·AttackReach + 0.35·WAI + 0.25·HVC | Adversarial compromise spread | V(v) | G^T (DEPENDS_ON reversed) |

**Weight justifications:**
- **RL dominates I(v) at 0.35:** Reachability loss is the most direct proxy for "how broken is the system" — it measures whether components can communicate at all, which is more fundamental than how many fragments form or how much throughput is lost.
- **CascadeReach leads IR(v) at 0.45:** The count of cascade-failed components is the clearest measure of blast radius, aligning directly with R(v)'s DG_in and RPR inputs.
- **WeightedReachabilityLoss leads IA(v) at 0.50:** Availability is fundamentally about whether the system can route messages at all. QoS weighting ensures high-priority paths dominate the score, consistent with the QSPOF and w(v) terms in A(v).

---

## Simulation Modes

### Exhaustive Mode

Every component in the target layer is failed individually. Produces all five ground-truth scores for all N components. Required for Step 5 validation.

```bash
python bin/simulate_graph.py failure --exhaustive --layer system
# Time: O(N × (|V| + |E|))  e.g. ~12s for medium, ~8min for xlarge
```

Run event simulation first if FD(v) is needed:

```bash
python bin/simulate_graph.py event --all --messages 50 --layer system
python bin/simulate_graph.py failure --exhaustive --layer system --output results/impact.json
```

### Targeted Mode

Simulate a specific component's failure for interactive exploration. Returns all five impact scores plus the full cascade sequence.

```bash
python bin/simulate_graph.py failure --target MainBroker --layer system
python bin/simulate_graph.py failure --target NavLib --layer system  # Library Rule 4 visible
```

### Monte Carlo Mode

Samples a random subset with probabilistic cascade propagation. Produces distributions rather than point estimates. Not used for Step 5 validation (which requires deterministic CRASH-mode ground truth).

```bash
python bin/simulate_graph.py failure --monte-carlo --samples 500 --cascade-prob 0.85 --layer system
```

### Event Simulation

Establishes baseline flows for FD(v) computation. Should be run before exhaustive simulation when FD(v) is desired in I(v).

```bash
python bin/simulate_graph.py event --all --messages 50 --layer system
```

---

## Worked Example

**NavLib** failure — system from Step 2/3 worked example (SensorApp, MonitorApp, MainBroker, NavLib, /temperature).

This example illustrates Rule 4 (Library cascade) which was absent from the original simulation document.

**Cascade trace:**
```
Initialize: F = {NavLib}, queue = [NavLib]

Process NavLib (Rule 4 — Library cascade):
    NavLib is used by: SensorApp (USES), MonitorApp (USES)
    Both applications in → F immediately (simultaneous blast)
    F = {NavLib, SensorApp, MonitorApp}, queue = [SensorApp, MonitorApp]

Process SensorApp (Rule 3 — Application cascade):
    SensorApp PUBLISHES_TO /temperature.
    No other active publisher for /temperature.
    MonitorApp SUBSCRIBES_TO /temperature — but MonitorApp is already in F.
    No new failures triggered.

Process MonitorApp (Rule 3 — Application cascade):
    MonitorApp has no publisher role. No cascade.

Queue empty → Fixed point reached.

cascade_count = 2  (SensorApp, MonitorApp)
cascade_depth = 1
F = {NavLib, SensorApp, MonitorApp}
```

**Impact metrics:**
```
RL(NavLib):  All paths through SensorApp and MonitorApp are severed
              ≈ 0.80

FR(NavLib):  Surviving graph = {MainBroker, /temperature} — mostly isolated
              ≈ 0.60

TL(NavLib):  /temperature unreachable after SensorApp fails
              ≈ 0.71

I(NavLib) = 0.35×0.80 + 0.25×0.60 + 0.25×0.71 + 0.15×FD
           = 0.280 + 0.150 + 0.178 + 0.15×FD  ≈ 0.64–0.70  [CRITICAL]
```

**Post-pass IR(v):**
```
CascadeReach = 2 / (5−1) = 0.50
NormalizedCascadeDepth = 1 / max_depth_in_system
IR(NavLib) = 0.45×0.50 + 0.35×WCI + 0.20×NCD  ≈ 0.42+
```

**Cross-check with Step 3:** Step 3 predicted NavLib as HIGH/CRITICAL (R = 0.627). The simulation confirms cascade-count = 2, consistent with DG_in = 2 in the prediction model.

---

## Output Schema

```json
{
  "layer": "system",
  "simulation_mode": "exhaustive",
  "cascade_rules": ["physical", "logical", "application", "library"],
  "component_count": 35,
  "ranked_results": [
    {
      "target_id":           "NavLib",
      "target_type":         "Library",
      "composite_impact":    0.68,
      "reachability_loss":   0.80,
      "fragmentation":       0.60,
      "throughput_loss":     0.71,
      "flow_disruption":     0.65,
      "cascade_count":       2,
      "cascade_depth":       1,
      "cascaded_failures":   ["SensorApp", "MonitorApp"],
      "ir":                  0.44,
      "im":                  0.38,
      "ia":                  0.61,
      "iv":                  0.29
    }
  ]
}
```

---

## Performance

| Scale | Components | Exhaustive Time |
|-------|-----------|-----------------|
| tiny | 8 | < 1s |
| small | 15 | ~2s |
| medium | 35 | ~12s |
| large | 80 | ~60s |
| xlarge | 200 | ~8 min |

Post-passes add approximately 20–30% overhead on top of the main loop. For xlarge systems, use `--monte-carlo --samples 500` to reduce main loop time; post-passes still run on the sampled subset.

---

## Key Findings

Across all eight validated domain scenarios:

- **Application layer accuracy (ρ = 0.876):** Topology-based predictions agree strongly with simulation-derived ground truth.
- **Scale benefit:** Large systems (150–300+ components) produce ρ = 0.943 — prediction accuracy improves with scale.
- **Library cascades are high-impact:** Components failing via Rule 4 (Library cascade) consistently produce the highest I(v) values per cascade step — because the blast is simultaneous rather than sequential, all path metrics spike together rather than propagating gradually.
- **Cascade depth ≥ 3:** Components whose failure reaches depth ≥ 3 are most likely to be missed by single-metric predictors. CDPot_enh (Step 3) is the primary predictor that captures these deep cascades.
- **Infrastructure layer gap:** Infrastructure layer correlation (ρ ≈ 0.60–0.70) is lower because physical topology is more homogeneous than logical dependency structure.

---

## Commands

```bash
# ─── Recommended full sequence ───────────────────────────────────────────────
# Step A: establish baseline flows for FD(v)
python bin/simulate_graph.py event --all --messages 50 --layer system

# Step B+C: exhaustive simulation + all four post-passes
python bin/simulate_graph.py failure --exhaustive --layer system \
    --output results/impact.json

# ─── Targeted (interactive exploration) ──────────────────────────────────────
python bin/simulate_graph.py failure --target NavLib --layer system
python bin/simulate_graph.py failure --target MainBroker --layer mw \
    --failure-mode PARTITION

# ─── Monte Carlo (large systems) ─────────────────────────────────────────────
python bin/simulate_graph.py failure --monte-carlo --samples 500 \
    --cascade-prob 0.85 --layer system --output results/mc_impact.json

# ─── Full pipeline: Prediction → Simulation → Validation ─────────────────────
python bin/analyze_graph.py  --layer system --output results/prediction.json
python bin/simulate_graph.py event --all --messages 50 --layer system
python bin/simulate_graph.py failure --exhaustive --layer system \
    --output results/impact.json
python bin/validate_graph.py results/prediction.json results/impact.json
```

---

## What Comes Next

Simulation produces five ground-truth scores per component: I(v), IR(v), IM(v), IA(v), IV(v). Each is independent from Q(v) by construction.

Step 5 (Validation) aligns Q(v) and I(v) by component ID and computes eleven statistical metrics per dimension: Spearman ρ and Kendall τ (rank correlation), F1-score (classification agreement), Top-K overlap and NDCG@K (ranking quality), RMSE/MAE (magnitude error), and dimension-specific specialist metrics (CCR@5 for Reliability, COCR@5 for Maintainability, SPOF_F1 for Availability, AHCR@5 for Vulnerability). A tiered gate system produces a definitive pass/fail verdict. The achieved Spearman ρ = 0.876 confirms the methodology's central claim.

---

← [Step 3: Prediction](prediction.md) | → [Step 5: Validation](validation.md)