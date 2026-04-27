# Step 1: Import

**Turn your system architecture into a graph that captures who depends on whom and how strongly.**

[README](../README.md) | → [Step 2: Analyze](structural-analysis.md)

---

## Table of Contents

1. [What This Step Does](#what-this-step-does)
2. [Why a Graph?](#why-a-graph)
3. [Formal Graph Definition](#formal-graph-definition)
4. [Construction Phases](#construction-phases)
   - [Phase 1 — Entity Modeling](#phase-1--entity-modeling)
   - [Phase 2 — Structural Graph](#phase-2--structural-graph)
   - [Phase 3 — Intrinsic Weight Computation](#phase-3--intrinsic-weight-computation)
   - [Phase 4 — Dependency Derivation](#phase-4--dependency-derivation)
   - [Phase 5 — Aggregate Weight Propagation](#phase-5--aggregate-weight-propagation)
5. [Layer Projections](#layer-projections)
6. [Two Graph Views](#two-graph-views)
7. [Input Format](#input-format)
8. [Worked Example](#worked-example)
9. [Domain Mapping](#domain-mapping)
10. [Complexity](#complexity)
11. [CLI Reference: Importing Graph Data](#cli-reference-importing-graph-data)
12. [CLI Reference: Exporting Graph Data](#cli-reference-exporting-graph-data)
13. [Export–Import Roundtrip](#exportimport-roundtrip)
14. [What Comes Next](#what-comes-next)

---

## What This Step Does

Modeling takes a distributed publish-subscribe system — its applications, topics, brokers, infrastructure nodes, and shared libraries — and converts it into a formal weighted directed graph. This graph becomes the foundation for all subsequent steps.

The process runs in five sequential phases inside `Neo4jRepository.save_graph()`:

```
                Phase 1         Phase 2         Phase 3         Phase 4         Phase 5
                  │               │               │               │               │
System JSON ──▶  │  Entity    ──▶ │  Structural ──▶ │  Intrinsic ──▶ │  Dependency ──▶ │  Aggregate
                 │  Modeling      │  Edges         │  Weights       │  Derivation     │  Weights
                 │                │                │                │                 │
                 │  5 vertex      │  6 edge types  │  Topic QoS     │  6 DEPENDS_ON   │  App, Broker,
                 │  types         │  imported;     │  weights;      │  rules applied; │  Node, Library
                 │  created       │  fan-out attrs │  edge weights  │  path_count     │  weights
                 │                │  computed      │  inherited     │  recorded       │  propagated
```

The output is two complementary graph views — **G_structural** for simulation and **G_analysis(l)** for analysis and prediction — described in [Two Graph Views](#two-graph-views).

---

## Why a Graph?

In a pub-sub system, applications don't call each other directly — they communicate through topics and brokers. A raw architecture diagram doesn't reveal the true dependency chains. By deriving logical DEPENDS_ON relationships, we make hidden dependencies explicit:

- If App A publishes to Topic T and App B subscribes to Topic T, then **B depends on A**. If A crashes, B is starved of data.
- If two applications share a broker, they have an **infrastructure dependency**. If the broker fails, both are affected.
- If two applications run on the same host node and their host's broker fails, they have a **cross-layer dependency** captured by the node-to-broker rule.
- If two brokers share a physical node, they have a **colocation dependency** — a node failure takes both down simultaneously (Rule 6).
- If multiple applications share a library, they have a **code dependency**. A library crash or incompatible update causes a **simultaneous blast** — all consumers fail at once, not sequentially. This pattern is qualitatively different from pub-sub cascade propagation and is made visible by Rule 5.

These derived dependencies are what make the graph useful for predicting failure impact before any system is deployed.

---

## Formal Graph Definition

```
G = (V, E, τ_V, τ_E, w) where:

V = V_app ∪ V_broker ∪ V_topic ∪ V_node ∪ V_lib
    (Applications, Brokers, Topics, Infrastructure Nodes, Libraries)

E_structural ⊆ V × V    (6 structural edge types — imported from topology JSON)
E_dependency ⊆ V × V    (DEPENDS_ON edges — derived by 6 derivation rules)

τ_V : V → {App, Broker, Topic, Node, Library}                    (vertex type function)
τ_E : E → {structural edge types} ∪ {DEPENDS_ON}                 (edge type function)

w : E → [0, 1]    (QoS-derived edge weight)
w : V → [0, 1]    (QoS-derived vertex weight, propagated from incident edges)
```

**Selected vertex attributes relevant to reliability prediction:**

| Vertex Type | Attribute | Description |
|-------------|-----------|-------------|
| Topic | `subscriber_count` | Number of distinct subscribing applications/libraries (fan-out) |
| Topic | `publisher_count` | Number of distinct publishing applications/libraries (fan-in) |
| Application | `weight` | Hybrid: 0.80 × max(w_topic) + 0.20 × mean(w_topic) |
| Broker | `weight` | Hybrid: 0.70 × max(w_topic) + 0.30 × mean(w_topic) |
| Node | `weight` | max(w) over all hosted applications and brokers |
| Library | `weight` | Fan-out amplified: min(1.0, base_w × (1 + γ × log₂(1 + DG_in))) |

**Selected edge attributes on DEPENDS_ON edges:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `weight` | float ∈ [0,1] | Max QoS weight over all topics mediating this dependency |
| `dependency_type` | string | One of: `app_to_app`, `app_to_broker`, `node_to_node`, `node_to_broker`, `app_to_lib`, `broker_to_broker` |
| `path_count` | int ≥ 1 | Number of shared topics (for `app_to_app`) or shared nodes (for `broker_to_broker`) establishing this dependency |

> **On `path_count`:** When two components are connected through multiple shared topics, `path_count` captures coupling intensity. A `path_count = 3` dependency means three simultaneous failure vectors between the same pair — structurally more fragile than three independent single-topic links. Step 3 uses this to refine cascade depth potential (CDPot) computations.

---

## Construction Phases

### Phase 1 — Entity Modeling

Each entity in the topology JSON becomes a vertex in G. Five vertex types are created in this order: infrastructure nodes, brokers, topics, applications, libraries.

| Vertex Type | JSON Array | Core Properties |
|-------------|------------|-----------------|
| **Node** | `nodes[]` | `id`, `name` |
| **Broker** | `brokers[]` | `id`, `name` |
| **Topic** | `topics[]` | `id`, `name`, `size`, `qos_reliability`, `qos_durability`, `qos_transport_priority`; `subscriber_count` and `publisher_count` added in Phase 2 |
| **Application** | `applications[]` | `id`, `name`, `role`, `app_type`, `version`, `criticality`; code-metric flat properties (`cm_*`); system-hierarchy flat properties |
| **Library** | `libraries[]` | `id`, `name`, `version`; code-metric and system-hierarchy properties |

**Optional code-quality attributes** on Application and Library vertices (all default to `0`/`0.0` when absent):

| Field | Type | Description |
|-------|------|-------------|
| `cm_total_loc` | int | Total lines of code |
| `cm_avg_wmc` | float | Average Weighted Methods per Class |
| `cm_avg_lcom` | float | Average Lack of Cohesion of Methods (raw SonarQube scale) |
| `cm_avg_cbo` | float | Average Coupling Between Objects |
| `cm_avg_rfc` | float | Average Response for a Class |
| `cm_avg_fanin` | float | Average afferent coupling (Library: internal static analysis) |
| `cm_avg_fanout` | float | Average efferent coupling (Library: internal static analysis) |

These attributes feed the **Code Quality Penalty (CQP)** composite used in Step 3's Maintainability M(v) term. When absent, M(v) falls back to the topology-only formula.

> **Why `subscriber_count` and `publisher_count` are listed under Phase 1 but computed in Phase 2:** These are properties of Topic vertices, but their values depend on SUBSCRIBES_TO and PUBLISHES_TO edges which don't exist until Phase 2. They are computed at the end of Phase 2 and written back onto each Topic vertex.

**Uniqueness constraints** are created for all five vertex labels (`Application`, `Broker`, `Topic`, `Node`, `Library`) on the `id` property. Constraints are created as `IF NOT EXISTS` and should be present whether or not `--clear` is used.

---

### Phase 2 — Structural Graph

Six structural edge types are imported directly from the topology JSON. Each edge represents an explicit, observable relationship in the system.

| Edge Type | Direction | Meaning |
|-----------|-----------|---------|
| `PUBLISHES_TO` | Application / Library → Topic | Component sends messages to this topic |
| `SUBSCRIBES_TO` | Application / Library → Topic | Component receives messages from this topic |
| `ROUTES` | Broker → Topic | Broker is responsible for routing this topic |
| `RUNS_ON` | Application / Broker → Node | Component is hosted on this infrastructure node |
| `CONNECTS_TO` | Node → Node | Direct network connectivity between hosts |
| `USES` | Application / Library → Library | Component depends on this shared code module |

Together these six types form **G_structural**, which Step 4 (Simulation) uses for cascade propagation. They are never modified after import.

**Fan-out augmentation (Phase 2 post-step):** After all structural edges are imported, each Topic vertex is updated with:

```
subscriber_count(t) = |{ a ∈ V_app ∪ V_lib : (a, t) ∈ SUBSCRIBES_TO }|
publisher_count(t)  = |{ a ∈ V_app ∪ V_lib : (a, t) ∈ PUBLISHES_TO  }|
```

`subscriber_count` is the primary fan-out signal for single-point-of-failure analysis — a Topic with high fan-out is a natural distribution bottleneck.

---

### Phase 3 — Intrinsic Weight Computation

Topic weights are computed from QoS properties and message size. These weights are the foundational signal for all downstream weight propagation.

#### Topic Weight Formula

```
w(topic) = max(MIN_WEIGHT, β × QoS_score + (1−β) × size_norm)

QoS_score  = 0.30 × reliability_score + 0.40 × durability_score + 0.30 × priority_score
size_norm  = min(log₂(1 + size_kb) / 50, 1.0)     where size_kb = size_bytes / 1024
β          = 0.85
MIN_WEIGHT = 0.01
```

**AHP justification for β:** QoS semantics are the primary signal for dependency criticality; payload size is a secondary amplifier. The 0.85 weight preserves the primacy of the QoS contract while allowing message volume to modulate the final score.

**AHP justification for QoS sub-weights:** Durability (0.40) outweighs Reliability and Priority (0.30 each) because durability defines message state survival — fundamental for resilience — while reliability and priority govern transient delivery quality.

| Component | Symbolic Value | Score |
|-----------|----------------|-------|
| **reliability_score** | `RELIABLE` | 1.0 |
| | `BEST_EFFORT` | 0.0 |
| **durability_score** | `PERSISTENT` | 1.0 |
| | `TRANSIENT` | 0.6 |
| | `TRANSIENT_LOCAL` | 0.5 |
| | `VOLATILE` | 0.0 |
| **priority_score** | `HIGHEST` | 1.0 |
| | `CRITICAL` | 1.0 |
| | `URGENT` | 1.0 |
| | `HIGH` | 0.66 |
| | `MEDIUM` | 0.33 |
| | `LOW` | 0.0 |

**Edge weight inheritance:** After topic weights are computed, structural edge weights are updated by one-pass inheritance:

```
∀ e = (a, t) ∈ PUBLISHES_TO ∪ SUBSCRIBES_TO ∪ ROUTES :  e.weight = t.weight
```

---

### Phase 4 — Dependency Derivation

Structural edges reveal physical relationships but not logical dependencies. This phase derives **DEPENDS_ON** edges — directed edges meaning "if the target fails, the source is affected."

**Edge direction:** DEPENDS_ON always points from the *dependent* to the *dependency* (e.g., subscriber → publisher, application → broker).

Six derivation rules are applied, producing six `dependency_type` values:

| Rule | `dependency_type` | Source Pattern | Weight |
|------|-------------------|----------------|--------|
| 1 | `app_to_app` | App/Lib `SUBSCRIBES_TO` → Topic ← `PUBLISHES_TO` App/Lib; also transitive via `USES*1..3` library chains | `max(t.weight)` over shared topics |
| 2 | `app_to_broker` | App/Lib `PUBLISHES_TO` or `SUBSCRIBES_TO` → Topic ← `ROUTES` Broker; also transitive via `USES*1..3` chains | `max(t.weight)` over routed topics |
| 3 | `node_to_node` | Lifted from `app_to_app` and `app_to_broker` DEPENDS_ON edges: Node_B → Node_A when their hosted apps share one of those dependency types | lifted `max(d.weight)` over matching edges |
| 4 | `node_to_broker` | Lifted from Rule 2: Node → Broker when a hosted app has an `app_to_broker` edge | lifted `max(dep.weight)` |
| 5 | `app_to_lib` | App/Lib `USES` → Library | `src.weight` (set in Phase 5, after aggregate propagation) |
| 6 | `broker_to_broker` | Bidirectional colocation edge between two Brokers sharing a physical Node | `node.weight` (set in Phase 5) |

> **Phase ordering note for Rules 5 and 6:** `app_to_lib` edge weights are set from `app.weight` and `broker_to_broker` edge weights are set from `node.weight`. Both of those vertex weights are computed in **Phase 5**, which runs after Phase 4. Rules 5 and 6 derivation queries therefore read placeholder weights (≈ 0.01) at derivation time; a second-pass update query in Phase 5 corrects the edge weights once vertex weights are finalized.

**Multi-path coupling:** When two applications communicate through multiple shared topics, a single DEPENDS_ON edge is created with:

```
edge.weight      = max(w(t) for t in shared_topics)   # worst-case QoS
edge.path_count  = len(shared_topics)                  # coupling intensity
```

`path_count` is not folded into the weight to preserve the `w ∈ [0,1]` contract.

**Library blast semantics vs. pub-sub cascade:** Rule 5 captures a qualitatively different failure mode. A library failure causes a *simultaneous* blast — all consuming applications fail at once. This contrasts with pub-sub cascade propagation (Rule 1), which flows step-by-step through topics and brokers. Step 4 (Simulation) handles this distinction at the cascade propagation layer. Rule 5 simply records the structural dependency so that `DG_in(Library)` is non-zero and visible to R(v) in Step 3.

**Derivation trace example:**

```
Given:
  SensorApp    --[PUBLISHES_TO]--> /temperature
  MonitorApp   --[SUBSCRIBES_TO]--> /temperature
  MainBroker   --[ROUTES]---------> /temperature
  BackupBroker --[ROUTES]---------> /temperature   (redundant router)
  SensorApp    --[RUNS_ON]--------> ComputeNode1
  MonitorApp   --[RUNS_ON]--------> ComputeNode2
  MainBroker   --[RUNS_ON]--------> ComputeNode1
  BackupBroker --[RUNS_ON]--------> ComputeNode1
  SensorApp    --[USES]-----------> NavLib
  MonitorApp   --[USES]-----------> NavLib

Derived DEPENDS_ON edges:
  MonitorApp  --[app_to_app,    w=w(/temp)]-->  SensorApp    (Rule 1)
  MonitorApp  --[app_to_broker, w=w(/temp)]-->  MainBroker   (Rule 2)
  SensorApp   --[app_to_broker, w=w(/temp)]-->  MainBroker   (Rule 2)
  ComputeNode2 --[node_to_node, w=lifted]-->    ComputeNode1 (Rule 3)
  ComputeNode2 --[node_to_broker,w=lifted]-->   MainBroker   (Rule 4)
  SensorApp   --[app_to_lib,   w=app.weight]--> NavLib       (Rule 5)
  MonitorApp  --[app_to_lib,   w=app.weight]--> NavLib       (Rule 5)
  MainBroker  --[broker_to_broker,w=node.w]-->  BackupBroker (Rule 6, bidirectional)

After derivation:
  DG_in(NavLib)     = 2   → visible to R(v) formula
  DG_in(SensorApp)  = 1
  DG_in(MainBroker) = 2
```

---

### Phase 5 — Aggregate Weight Propagation

Once topic weights are established (Phase 3) and DEPENDS_ON edges exist (Phase 4), vertex weights for Applications, Brokers, Nodes, and Libraries are computed by propagating topic weights upward through the component hierarchy. This phase also corrects the `app_to_lib` and `broker_to_broker` edge weights that could only be assigned a placeholder in Phase 4.

#### Application Weight

```
w(app) = 0.80 × max{ w(t) : app PUBLISHES_TO t OR app SUBSCRIBES_TO t }
       + 0.20 × mean{ w(t) : app PUBLISHES_TO t OR app SUBSCRIBES_TO t }
```

The hybrid formula reflects that an application's criticality is primarily bounded by its most critical data stream (0.80 × max), but a dense subscription footprint of medium-weight topics adds cumulative risk (0.20 × mean). When `max = mean` (single-topic app), the formula collapses to `w = w(t)`.

**Library-mediated pass (step 1.5):** The formula above only counts topics directly connected to the application via `PUBLISHES_TO` or `SUBSCRIBES_TO`. Applications that communicate exclusively through shared libraries (no direct topic edges) would receive `w(app) = 0.01` from the first pass — making them invisible to RMAV scoring even if they indirectly handle high-weight data.

After Library weights are computed in step 2, a second pass corrects this for any application still at the default floor:

```
For all apps where w(app) ≤ 0.01:
  w(app) = max{ w(l) : app USES l }
```

This propagates library importance back to the consuming application. Only applications with no direct topics AND at least one USES edge are affected; all other applications retain their step-1 weights unchanged.

#### Library Weight

```
w(lib) = min(1.0, base_w × (1 + γ × log₂(1 + DG_in)))
         where base_w = max( max{ w(t) : lib PUBLISHES_TO t OR lib SUBSCRIBES_TO t },
                             max{ w(app) : app USES lib } )
               γ      = 0.15
```

The fan-out multiplier reflects simultaneous blast semantics. A library consumed by 15 applications has a blast radius of 15 — this must be visible even before any of those applications have high individual weights. The log₂ term prevents extreme fan-out from producing weights > 1.0. When `DG_in = 0` (unused library), `w(lib) = base_w × 1.0 = base_w`.

#### Broker Weight

```
w(broker) = 0.70 × max{ w(t) : broker ROUTES t }
           + 0.30 × mean{ w(t) : broker ROUTES t }
```

A broker routing 20 medium-weight topics carries more cumulative risk than one routing a single high-weight topic. The hybrid captures both worst-case exposure and accumulated routing load.

#### Node Weight

```
w(node) = max{ w(v) : v RUNS_ON node }
```

A node's hardware failure takes down all hosted components simultaneously; the worst-case hosted component determines the node's criticality tier.

#### Edge Weight Corrections (Phase 5 post-step)

After vertex weights are set, two edge types receive their final weights:

```cypher
// Rule 5: app_to_lib edges inherit the application's QoS weight
MATCH (app)-[d:DEPENDS_ON {dependency_type: 'app_to_lib'}]->(lib:Library)
SET d.weight = coalesce(app.weight, 0.01)

// Rule 6: broker_to_broker edges inherit the shared node's weight
MATCH (b1:Broker)-[d:DEPENDS_ON {dependency_type: 'broker_to_broker'}]->(b2:Broker)
MATCH (b1)-[:RUNS_ON]->(n:Node)<-[:RUNS_ON]-(b2)
WITH d, max(n.weight) as node_w
SET d.weight = coalesce(node_w, 0.01)
```

---

## Layer Projections

The graph supports four layer projections, each filtering vertices and DEPENDS_ON edges to a specific architectural concern.

| Layer | CLI name | Vertex Types | `dependency_type` values |
|-------|----------|-------------|--------------------------|
| Application | `app` | Application, Library | `app_to_app`, `app_to_lib` |
| Infrastructure | `infra` | Node | `node_to_node` |
| Middleware | `mw` | Application, Broker, Node | `app_to_broker`, `node_to_broker`, `broker_to_broker` |
| System | `system` | All five types | All six types |

> **`app_to_lib` and Library nodes:** These are available in the `system` layer and can be isolated by filtering `dependency_type = 'app_to_lib'`. They are intentionally excluded from the `app` layer to keep Application-layer analysis focused on pub-sub data flow. Library blast-radius analysis is performed at the `system` layer.

**Legacy layer aliases** (backward compatible, resolved internally):

| Alias | Canonical name |
|-------|----------------|
| `application` | `app` |
| `infrastructure` | `infra` |
| `app_broker` | `mw` |
| `complete` | `system` |

---

## Two Graph Views

| Graph | Contains | Used By |
|-------|----------|---------|
| **G_structural** | All vertices + 6 structural edge types (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`) | Step 4 (Simulation): cascade propagation follows physical paths |
| **G_analysis(l)** | Layer-filtered vertices + `DEPENDS_ON` edges only | Steps 2–3 (Analysis + Prediction): centrality metrics operate on abstract dependency graph |

The separation is deliberate and methodologically essential: **prediction and simulation must remain independent**. Centrality metrics in Step 2 must not be contaminated by simulation outcomes, and Step 4 simulation must not use prediction scores as inputs. Using separate graph views enforces this contract structurally.

---

## Input Format

The topology JSON uses a **dict-of-lists** structure for relationships. Each key under `"relationships"` is the snake_case name of a structural edge type, and the value is a list of `{ "from": id, "to": id }` objects.

```json
{
  "metadata": {
    "scale": { "apps": 5, "topics": 3, "brokers": 1, "nodes": 2, "libs": 2 },
    "seed": 42,
    "generation_mode": "statistical",
    "domain": null,
    "scenario": null
  },
  "nodes": [
    { "id": "N0", "name": "ComputeNode1" },
    { "id": "N1", "name": "ComputeNode2" }
  ],
  "brokers": [
    { "id": "B0", "name": "MainBroker" }
  ],
  "topics": [
    {
      "id": "T0",
      "name": "/temperature",
      "size": 64,
      "qos": {
        "reliability": "RELIABLE",
        "durability": "TRANSIENT_LOCAL",
        "transport_priority": "HIGH"
      }
    }
  ],
  "applications": [
    {
      "id": "A0",
      "name": "SensorApp",
      "role": "pub",
      "app_type": "sensor",
      "version": "1.0.0",
      "criticality": true,
      "system_hierarchy": {
        "csc_name": "Sensor Platform",
        "csci_name": "Perception Software",
        "css_name": "Environmental Sensing",
        "csms_name": "Temperature Monitor"
      },
      "code_metrics": {
        "size":       { "total_loc": 1200, "total_classes": 12, "total_methods": 95, "total_fields": 30 },
        "complexity": { "total_wmc": 120,  "avg_wmc": 10.0, "max_wmc": 22 },
        "cohesion":   { "avg_lcom": 18.4,  "max_lcom": 42.0 },
        "coupling":   { "avg_cbo": 5.2, "max_cbo": 9, "avg_rfc": 22.1, "max_rfc": 38,
                        "avg_fanin": 3.1, "max_fanin": 7, "avg_fanout": 4.8, "max_fanout": 11 }
      }
    },
    {
      "id": "A1",
      "name": "MonitorApp",
      "role": "sub",
      "app_type": "monitor",
      "version": "2.1.0",
      "criticality": false
    }
  ],
  "libraries": [
    {
      "id": "L0",
      "name": "NavLib",
      "version": "3.2.1",
      "system_hierarchy": {
        "csc_name": "Navigation Platform",
        "csci_name": "Navigation Software",
        "css_name": "Path Planning",
        "csms_name": "Core Navigation"
      },
      "code_metrics": {
        "size":       { "total_loc": 4500, "total_classes": 42, "total_methods": 360, "total_fields": 128 },
        "complexity": { "avg_wmc": 14.2, "max_wmc": 38 },
        "cohesion":   { "avg_lcom": 29.3, "max_lcom": 87.1 },
        "coupling":   { "avg_cbo": 8.1, "avg_rfc": 31.4, "avg_fanin": 7.2, "avg_fanout": 5.6 }
      }
    }
  ],
  "relationships": {
    "runs_on":      [
      { "from": "A0", "to": "N0" },
      { "from": "A1", "to": "N1" },
      { "from": "B0", "to": "N0" }
    ],
    "routes":       [{ "from": "B0", "to": "T0" }],
    "publishes_to": [{ "from": "A0", "to": "T0" }],
    "subscribes_to":[{ "from": "A1", "to": "T0" }],
    "connects_to":  [{ "from": "N0", "to": "N1" }],
    "uses":         [
      { "from": "A0", "to": "L0" },
      { "from": "A1", "to": "L0" }
    ]
  }
}
```

**Schema notes:**

- `"qos"` and `"qos_policy"` are both accepted as the QoS sub-object key; `"qos"` is canonical.
- QoS string values must be **uppercase** (`"RELIABLE"`, `"TRANSIENT_LOCAL"`, `"HIGH"`, etc.). They are stored and exported in uppercase. The Cypher weight CASE statements perform case-sensitive matching, so lowercase values (`"reliable"`) would fall through to the `ELSE 0.0` branch and silently produce `w = 0.01` for every affected topic.
- All fields except `"id"` are optional. Missing fields receive defaults (`role = "pubsub"`, `app_type = "service"`, `qos_reliability = "BEST_EFFORT"`, etc.).
- The `"metadata"` block is optional but strongly recommended for provenance tracking. Generated files include it automatically.
- Relationship edges may also use `"source"` and `"target"` keys as aliases for `"from"` and `"to"`.

---

## Worked Example

**Given topology:** SensorApp publishes to `/temperature`; MonitorApp subscribes. Both use NavLib. MainBroker routes `/temperature`. `/temperature` has QoS `RELIABLE / TRANSIENT_LOCAL / HIGH`.

**Phase 1 — Entities created:**

```
Vertices: SensorApp (Application), MonitorApp (Application),
          MainBroker (Broker), /temperature (Topic), NavLib (Library)
```

**Phase 2 — Structural edges imported:**

```
SensorApp  --[PUBLISHES_TO]-->  /temperature
MonitorApp --[SUBSCRIBES_TO]--> /temperature
MainBroker --[ROUTES]---------> /temperature

/temperature.subscriber_count = 1
/temperature.publisher_count   = 1
```

**Phase 3 — Intrinsic weights:**

```
QoS_score(/temperature) = 0.30×1.0 + 0.40×0.5 + 0.30×0.66 = 0.298 + 0.200 + 0.198 = 0.696
size_norm(64 bytes)      = log₂(1 + 0.0625) / 50 ≈ 0.0017   (negligible)
w(/temperature)          = max(0.01, 0.85×0.696 + 0.15×0.0017) ≈ 0.592

PUBLISHES_TO.weight  = 0.592
SUBSCRIBES_TO.weight = 0.592
ROUTES.weight        = 0.592
```

**Phase 4 — Dependency derivation:**

```
MonitorApp  --[DEPENDS_ON, app_to_app,    w=0.592, path_count=1]--> SensorApp
MonitorApp  --[DEPENDS_ON, app_to_broker, w=0.592, path_count=1]--> MainBroker
SensorApp   --[DEPENDS_ON, app_to_broker, w=0.592, path_count=1]--> MainBroker
SensorApp   --[DEPENDS_ON, app_to_lib,   w=placeholder]-----------> NavLib
MonitorApp  --[DEPENDS_ON, app_to_lib,   w=placeholder]-----------> NavLib
```

**Phase 5 — Aggregate weights:**

```
w(SensorApp)  = 0.80×0.592 + 0.20×0.592 = 0.592  (single topic)
w(MonitorApp) = 0.592
w(MainBroker) = 0.70×0.592 + 0.30×0.592 = 0.592

w(NavLib)  base_w = max(0.592, 0.592) = 0.592
           DG_in  = 2
           w      = min(1.0, 0.592 × (1 + 0.15 × log₂(3))) ≈ 0.592 × 1.238 ≈ 0.733

app_to_lib edge weights corrected:
  SensorApp  → NavLib: w = 0.592
  MonitorApp → NavLib: w = 0.592
```

---

## Domain Mapping

The model maps naturally to different pub-sub middleware technologies:

| Graph Concept | ROS 2 / DDS | Apache Kafka | MQTT |
|---------------|-------------|--------------|------|
| Application | ROS Node | Producer / Consumer | MQTT Client |
| Topic | ROS Topic | Kafka Topic | MQTT Topic |
| Broker | DDS Participant | Kafka Broker | MQTT Broker |
| Node | Host / Container | Broker Host | Broker Server |
| Library | ROS package dep | Maven artifact | Paho client lib |
| `PUBLISHES_TO` | `publish()` call | `produce()` call | `publish()` call |
| `SUBSCRIBES_TO` | `subscription()` call | `consume()` call | `subscribe()` call |
| `USES` | `package.xml` dep | `pom.xml` dep | `requirements.txt` |

---

## Complexity

| Phase | Operation | Complexity | Notes |
|-------|-----------|------------|-------|
| Phase 1 | Vertex creation | O(&#124;V&#124;) | One vertex per entity |
| Phase 2 | Structural edge import | O(&#124;E_S&#124;) | One pass over relationships |
| Phase 2 post-step | Fan-out augmentation | O(&#124;E_S&#124;) | One pass over `SUBSCRIBES_TO` and `PUBLISHES_TO` |
| Phase 3 | Topic weight computation | O(&#124;V_topic&#124;) | One Cypher pass |
| Phase 3 | Edge weight inheritance | O(&#124;E_S&#124;) | One pass over pub/sub/routes edges |
| Phase 4 | `app_to_app` derivation | O(&#124;Apps&#124;² × &#124;Topics&#124;) | All subscriber–publisher pairs per topic; bounded by fan-out in practice |
| Phase 4 | `app_to_lib` derivation | O(&#124;USES edges&#124;) | Sparse in practice |
| Phase 4 | Rules 2–6 | O(&#124;E_S&#124;) | One pass per rule |
| Phase 5 | Aggregate weight propagation | O(&#124;V&#124; + &#124;E_S&#124;) | One Cypher pass per vertex type |

The dominant cost is Phase 4 `app_to_app` derivation. In practice, topic fan-out is bounded (typically 1–12 subscribers), so the effective cost is much lower than the worst case. Critically, all five phases run **once at design time**, with zero runtime monitoring overhead.

---

## CLI Reference: Importing Graph Data

`bin/import_graph.py` reads a topology JSON file and runs all five construction phases against a Neo4j instance.

### Synopsis

```
python bin/import_graph.py --input <file> [options]
```

### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--input` | path | **yes** | — | Path to the topology JSON file |
| `--clear` | flag | no | off | Wipe the entire database before importing. Recommended when loading a new topology to avoid stale data |
| `--dry-run` | flag | no | off | Validate the input JSON and report expected counts without touching the database |
| `--uri` | string | no | `bolt://localhost:7687` | Neo4j Bolt connection URI (env: `NEO4J_URI`) |
| `--user` / `-u` | string | no | `neo4j` | Neo4j username (env: `NEO4J_USER`) |
| `--password` / `-p` | string | no | `password` | Neo4j password (env: `NEO4J_PASSWORD`) |
| `--layer` / `-l` | string | no | `system` | Reserved for future use; currently unused by the import path |
| `--verbose` / `-v` | flag | no | off | Enable debug logging and print tracebacks on error |
| `--quiet` / `-q` | flag | no | off | Suppress non-essential console output |
| `--output` / `-o` | path | no | — | Write the returned import statistics to a JSON file |

### Call Chain

```
bin/import_graph.py
  └─ saag.Client.import_topology(filepath, clear)
       └─ src.usecases.model_graph.ModelGraphUseCase.execute()
            └─ Neo4jRepository.save_graph()
                 ├─ Phase 1: _import_entities()
                 ├─ Phase 2: _import_relationships()
                 ├─ Phase 3: _calculate_intrinsic_weights()
                 ├─ Phase 4: _derive_dependencies()
                 └─ Phase 5: _calculate_aggregate_weights()
```

### Usage Examples

```bash
# Basic import (appends to existing database)
python bin/import_graph.py \
  --input input/system.json

# Import with database wipe (recommended for fresh runs)
python bin/import_graph.py \
  --input input/system.json \
  --clear

# Import against a non-default Neo4j instance
python bin/import_graph.py \
  --input input/sample_topology.json \
  --clear \
  --uri bolt://neo4j-host:7687 \
  --user admin \
  --password secret \
  --verbose

# Save import statistics to a file for CI verification
python bin/import_graph.py \
  --input input/system.json \
  --clear \
  --output output/import_stats.json
```

### Output

On success, the console prints an import summary. The returned statistics dict (also written to `--output` if specified) contains:

| Key | Description |
|-----|-------------|
| `nodes_imported` | Total vertex count in Neo4j after import (excludes the internal `:Metadata` node) |
| `edges_imported` | Total relationship count after import (structural + DEPENDS_ON) |
| `duration_ms` | Total import duration in milliseconds |
| `application_count`, `broker_count`, … | Per-label vertex counts |
| `app_to_app_count`, `app_to_broker_count`, … | Per-type DEPENDS_ON edge counts |
| `success` | `true` on success |
| `message` | Human-readable status string |

**Dry-run output** (`--dry-run` flag): no database writes occur. The returned dict instead contains:

| Key | Description |
|-----|-------------|
| `nodes_imported` | Total vertex count parsed from the input JSON |
| `edges_imported` | Count of structural relationship entries in the input JSON |
| `structural_edges` | Same as `edges_imported` — the six structural edge types only |
| `estimated_depends_on` | Estimated lower bound of DEPENDS_ON edges that would be derived (sum of `publishes_to` + `subscribes_to` + `uses` entries, which drive Rules 1, 2, and 5) |
| `note` | Reminder that `edges_imported` covers structural relationships only; DEPENDS_ON edges are derived at import time |
| `dry_run` | `true` |

### Notes and Caveats

**`--clear` is strongly recommended** when importing a new topology. Without it, uniqueness constraints prevent duplicate nodes, but stale DEPENDS_ON edges from a previous import can remain and silently inflate centrality scores.

**No transactional rollback.** If a phase fails mid-way (e.g., due to a Neo4j memory error on a large topology), the database is left in a partially constructed state: entities may be present but weights and DEPENDS_ON edges absent. Re-run with `--clear` to recover.

**Referential integrity is enforced in Phase 2.** Before any edge is created, each batch of relationships is validated: source and target IDs must exist as Neo4j vertices. A missing entity raises a `ValueError` and rolls back the entire transaction. The error message names up to five offending IDs. If you see this error, check your JSON for typos or missing entries in the `nodes`, `applications`, etc. arrays.

**The REST API equivalent** is `POST /api/v1/graph/import` with `ImportGraphRequest` body. The CLI and REST path share the same `ModelGraphUseCase` and produce identical Neo4j state.

---

## CLI Reference: Exporting Graph Data

`bin/export_graph.py` reads the current Neo4j database and writes a topology JSON file that mirrors the input format, suitable for archiving, sharing, or re-importing into a fresh database.

### Synopsis

```
python bin/export_graph.py --output <file> [options]
```

### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--output` / `-o` | path | **yes** | — | Path for the output JSON file. Parent directories are created if absent |
| `--format` | choice | no | `persistence` | `persistence` — nested JSON re-importable by `import_graph.py`; `analysis` — flat `components`/`edges` dict for downstream tooling |
| `--layer` / `-l` | string | no | `system` | Layer filter applied to `--format analysis` only. One of `app`, `infra`, `mw`, `system` (or legacy aliases). Scopes which vertex types and dependency types are included |
| `--include-structural` | flag | no | off | Include raw structural edges (`PUBLISHES_TO`, `SUBSCRIBES_TO`, etc.) alongside `DEPENDS_ON` edges in `--format analysis` output |
| `--uri` | string | no | `bolt://localhost:7687` | Neo4j Bolt connection URI (env: `NEO4J_URI`) |
| `--user` / `-u` | string | no | `neo4j` | Neo4j username (env: `NEO4J_USER`) |
| `--password` / `-p` | string | no | `password` | Neo4j password (env: `NEO4J_PASSWORD`) |
| `--verbose` / `-v` | flag | no | off | Enable debug logging and print tracebacks on error |
| `--quiet` / `-q` | flag | no | off | Suppress non-essential console output |

### Call Chains

**Persistence format** (`--format persistence`, default):
```
bin/export_graph.py
  └─ saag.Client.export_topology()
       └─ Neo4jRepository.export_json()
            └─ get_graph_data(include_raw=True) + _get_metadata_dict()
                 └─ serialization.reconstruct_export_payload()
                      └─ Reconstructed topology dict → json.dump()
```

**Analysis format** (`--format analysis`):
```
bin/export_graph.py
  └─ saag.Client.get_graph_data(component_types, dependency_types, include_raw)
       └─ Neo4jRepository.get_graph_data()
            └─ GraphData.to_dict() → json.dump()
```

The layer filter (`--layer`) is applied to the analysis format only, scoping `component_types` and `dependency_types` through `LAYER_DEFINITIONS`. The persistence format always exports all five vertex types.

### Usage Examples

```bash
# Export full topology — nested persistence format (re-importable)
python bin/export_graph.py \
  --output output/snapshot.json

# Export application-layer view — flat analysis format
python bin/export_graph.py \
  --output output/app_layer.json \
  --format analysis \
  --layer app

# Export system analysis view including raw structural edges
python bin/export_graph.py \
  --output output/system_full.json \
  --format analysis \
  --layer system \
  --include-structural

# Export from a remote Neo4j instance
python bin/export_graph.py \
  --output output/sample_snapshot.json \
  --uri bolt://neo4j-host:7687 \
  --user admin \
  --password secret \
  --verbose
```

### Output Format

**Persistence format** (default): the exported file uses the same dict-of-lists structure as the input and is suitable for direct re-import with `import_graph.py`. See [Input Format](#input-format) for the full schema.

One difference from the input: the persistence export adds a `"depends_on"` key inside `"relationships"` containing a snapshot of the currently derived DEPENDS_ON edges. This key is **informational only** — `import_graph.py` ignores it and always re-derives DEPENDS_ON from structural edges. The export summary reports these separately as "Derived (DEPENDS_ON)" to avoid inflating the structural edge count.

**Analysis format** (`--format analysis`): outputs `{ "components": [...], "edges": [...] }`. This format is consumed by the Genieus frontend and downstream analysis scripts. It is **not re-importable** by `import_graph.py`.

### Notes and Caveats

See [Export–Import Roundtrip](#exportimport-roundtrip) below for a full accounting of what is and is not preserved.

**The REST API equivalents** are:

| Endpoint | Method | Output shape | Re-importable? |
|----------|--------|-------------|----------------|
| `POST /api/v1/graph/export-neo4j-data` | `repo.export_json()` | Input-file shape | **Yes** |
| `POST /api/v1/graph/export` | `repo.get_graph_data()` | `components`/`edges` analysis shape | No — analysis view only |
| `POST /api/v1/graph/export-limited` | `repo.get_limited_graph_data()` | Truncated analysis shape | No |

Use `/export-neo4j-data` (not `/export`) when a re-importable snapshot is required. The `/export` endpoint produces an analysis-layer view for the Genieus frontend — its `components`/`edges` envelope is not accepted by `import_graph.py`.

---

## Export–Import Roundtrip

Running `export_graph.py` (persistence format) followed by `import_graph.py` on the output is a faithful roundtrip for all user-supplied data. The following table documents what is preserved and what is intentionally re-computed.

### Preserved

| Data | Notes |
|------|-------|
| All entity IDs and names | `id`, `name` for all five vertex types |
| Topic QoS and size | `qos_reliability`, `qos_durability`, `qos_transport_priority`, `size` — exported and re-imported in uppercase |
| Application `role`, `app_type`, `version`, `criticality` | Exported conditionally (only if non-null/non-empty) |
| All six structural relationship types | `runs_on`, `routes`, `publishes_to`, `subscribes_to`, `connects_to`, `uses` |
| `code_metrics` block (Applications and Libraries) | Flat `cm_*` properties in Neo4j are reconstructed into the nested `code_metrics` structure on export and re-flattened on re-import. CQP scores are fully reproducible after a roundtrip. |
| `system_hierarchy` block | Flat `csms_name`, `css_name`, `csc_name`, `csci_name` properties are reconstructed and re-imported correctly |
| `metadata` block | Stored in the `:Metadata` singleton node; reconstructed and included in the export |

### Not Preserved (Re-computed on Re-import)

| Data | What Happens Instead |
|------|----------------------|
| Computed `weight` properties | All five construction phases re-run on re-import from QoS data; computed weights are always fresh and do not need to be stored |
| `DEPENDS_ON` edges | Re-derived automatically from structural edges on re-import; the persistence export includes them as an informational `"depends_on"` key but this key is ignored by the importer |

### Roundtrip Validation Test

To verify roundtrip integrity for a given topology, run:

```bash
# Import original
python bin/import_graph.py --input input/system.json --clear

# Export persistence snapshot
python bin/export_graph.py --output output/snapshot.json

# Re-import snapshot
python bin/import_graph.py --input output/snapshot.json --clear

# Verify: node and edge counts, per-label counts, and topic weights
# should all match between the original and re-import runs.
# code_metrics, system_hierarchy, and metadata are fully preserved.
```

A full roundtrip integration test that asserts identical node/edge counts, identical per-label vertex counts, and identical topic weights after re-import is a recommended CI addition.

---

## What Comes Next

Step 1 produces two graph views: G_structural (for simulation) and G_analysis(l) (for analysis). Step 2 operates on G_analysis(l) to compute a structural metric vector M(v) for every component.

→ [Step 2: Structural Analysis](structural-analysis.md)