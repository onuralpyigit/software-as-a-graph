# Step 1: Modeling

**Turn your system architecture into a graph that captures who depends on whom and how strongly.**

[README](../README.md) | → [Step 2: Analysis](structural-analysis.md)

---

## Table of Contents

1. [What This Step Does](#what-this-step-does)
2. [Why a Graph?](#why-a-graph)
3. [Formal Graph Definition](#formal-graph-definition)
4. [Construction Phases](#construction-phases)
   - [Phase 1 — Entity Modeling](#phase-1--entity-modeling)
   - [Phase 2 — Structural Graph](#phase-2--structural-graph)
   - [Phase 3 — Dependency Derivation](#phase-3--dependency-derivation)
   - [Phase 4 — Weight Assignment](#phase-4--weight-assignment)
5. [Layer Projections](#layer-projections)
6. [Two Graph Views](#two-graph-views)
7. [Input Format](#input-format)
8. [Worked Example](#worked-example)
9. [Domain Mapping](#domain-mapping)
10. [Complexity](#complexity)
11. [Commands](#commands)
12. [What Comes Next](#what-comes-next)

---

## What This Step Does

Modeling takes a distributed publish-subscribe system — its applications, topics, brokers, infrastructure nodes, and shared libraries — and converts it into a formal weighted directed graph. This graph becomes the foundation for all subsequent steps.

The process has four phases:

```
                  Phase 1           Phase 2           Phase 3           Phase 4
                    │                 │                 │                 │
System Topology  ──▶│  Entity      ──▶│  Structural  ──▶│  Dependency  ──▶│  Weighted
   (JSON input)     │  Modeling       │  Graph          │  Derivation     │  Graph
                    │                 │                 │                 │
                    │  Vertices for   │  6 structural   │  5 DEPENDS_ON   │  QoS-based
                    │  each entity    │  edge types     │  rules applied  │  weights on
                    │  type; Topic    │  imported;      │  to derive      │  all edges
                    │  degree attrs   │  fan_out attrs  │  dependencies   │  and vertices;
                    │  noted          │  computed       │                 │  path_count
                    │                 │                 │                 │  on edges
```

The output is two complementary graphs — **G_structural** for simulation and **G_analysis(l)** for analysis — described in [Two Graph Views](#two-graph-views).

---

## Why a Graph?

In a pub-sub system, applications don't call each other directly — they communicate through topics and brokers. A raw architecture diagram doesn't reveal the true dependency chains. By deriving logical DEPENDS_ON relationships, we make hidden dependencies explicit:

- If App A publishes to Topic T and App B subscribes to Topic T, then **B depends on A**. If A crashes, B is starved of data.
- If two applications share a broker, they have an **infrastructure dependency**. If the broker fails, both are affected.
- If two brokers share a host node, they have a **colocation dependency**. A node failure takes both down.
- If multiple applications share a library, they have a **code dependency**. A library crash or incompatible update causes a **simultaneous blast** — all consumers fail at once, not sequentially. This pattern is qualitatively different from the pub-sub cascade and is made visible by Rule 5.

These derived dependencies are what make the graph useful for predicting failure impact.

---

## Formal Graph Definition

```
G = (V, E, τ_V, τ_E, w, φ) where:

V = V_app ∪ V_broker ∪ V_topic ∪ V_node ∪ V_lib
    (Applications, Brokers, Topics, Infrastructure Nodes, Libraries)

E_structural ⊆ V × V     (6 structural edge types — imported from topology JSON)
E_dependency ⊆ V × V     (DEPENDS_ON edges — derived by 5 derivation rules)

τ_V : V → {App, Broker, Topic, Node, Library}   (vertex type function)
τ_E : E → {structural edge types} ∪ {DEPENDS_ON}  (edge type function)

w : E → [0, 1]             (QoS-derived edge weight)
w : V → [0, 1]             (QoS-derived vertex weight, propagated from incident edges)

φ : V_topic → ℕ × ℕ       (fan_out function: Topic → (subscriber_count, publisher_count))
```

**Selected vertex attributes relevant to reliability prediction:**

| Vertex Type | Attribute | Description |
|-------------|-----------|-------------|
| Topic | `subscriber_count` | Number of distinct subscribing applications (fan-out) |
| Topic | `publisher_count` | Number of distinct publishing applications (fan-in) |
| Application | `qos_weight` | Propagated from max incident topic weight |
| Broker | `qos_weight` | Hybrid: 0.70 × max(w_t) + 0.30 × mean(w_t) |

**Selected edge attributes on DEPENDS_ON edges:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `weight` | float ∈ [0,1] | Max QoS weight over all topics mediating this dependency |
| `dependency_type` | string | One of: app_to_app, app_to_broker, node_to_node, node_to_broker, **app_to_lib** |
| `path_count` | int ≥ 1 | Number of distinct topics (for app_to_app) or USES edges (for app_to_lib) that jointly establish this dependency |

> **Reliability note on `path_count`:** When two components are connected through multiple shared topics, `path_count` captures the intensity of coupling. A `path_count = 3` dependency is structurally more fragile than three independent single-topic links — it represents three simultaneous failure vectors between the same pair. Step 3 (Prediction) can use this to refine CDPot computations.

---

## Construction Phases

### Phase 1 — Entity Modeling

Each entity in the system topology JSON becomes a vertex in G. Five vertex types are created:

| Vertex Type | Source | Properties |
|-------------|--------|------------|
| **Application** | `applications[]` | id, name, role (pub/sub/pubsub), app_type, loc *(opt)*, cyclomatic_complexity *(opt)*, coupling_afferent *(opt)*, coupling_efferent *(opt)*, lcom *(opt)* |
| **Broker** | `brokers[]` | id, name |
| **Topic** | `topics[]` | id, name, size, QoS policy; `subscriber_count` and `publisher_count` computed in Phase 2 |
| **Node** | `nodes[]` | id, name |
| **Library** | `libraries[]` | id, name, version, loc *(opt)*, cyclomatic_complexity *(opt)*, coupling_afferent *(opt)*, coupling_efferent *(opt)*, lcom *(opt)* |

**Code-level quality attributes** (all optional, all default to `0`/`0.0`):

| Field | Type | Description |
|-------|------|-------------|
| `loc` | `int` | Lines of code — raw size proxy |
| `cyclomatic_complexity` | `float` | Average cyclomatic complexity per method |
| `coupling_afferent` | `int` | Ca — modules that *depend on* this one (fan-in) |
| `coupling_efferent` | `int` | Ce — modules this one *depends on* (fan-out) |
| `lcom` | `float ∈ [0,1]` | Lack of Cohesion of Methods (0 = fully cohesive) |

These attributes feed the **Code Quality Penalty (CQP)** composite metric used in Step 3 (Prediction) to improve the Maintainability M(v) signal for **Application and Library** nodes. Library nodes are normalised independently from Application nodes (separate population min-max) because their typical LOC/CC scales differ significantly. When absent or zero, M(v) falls back to the topology-only formula (fully backward-compatible).

Library vertices model shared code dependencies. Their failure (e.g., a shared library crash or incompatible update) triggers a **simultaneous blast** — all consuming applications fail at the same moment, not in a cascade sequence. This is distinct from pub-sub cascade propagation and is captured by Rule 5 in Phase 3.

> **Why `subscriber_count` and `publisher_count` are listed here but computed in Phase 2:**
> These are properties of Topic vertices, but their values depend on SUBSCRIBES_TO and PUBLISHES_TO edges which don't exist until Phase 2. They are therefore computed at the end of Phase 2 and written back onto each Topic vertex. They appear here to document the complete vertex schema.

### Phase 2 — Structural Graph

Six structural edge types are imported directly from the topology JSON. Each edge represents an explicit, observable relationship in the system.

| Edge Type | Direction | Meaning |
|-----------|-----------|---------| 
| `PUBLISHES_TO` | Application → Topic | App sends messages to topic |
| `SUBSCRIBES_TO` | Application → Topic | App receives messages from topic |
| `ROUTES` | Broker → Topic | Broker is responsible for routing this topic |
| `RUNS_ON` | Application / Broker → Node | Component is hosted on this infrastructure node |
| `CONNECTS_TO` | Node → Node | Direct network connectivity between hosts |
| `USES` | Application → Library | App depends on this shared code module |

Together, these six types capture the full physical topology of the system. They form **G_structural**, which is used directly by Step 4 (Simulation) for cascade propagation.

**Fan-out augmentation (Phase 2 post-step):**
After all structural edges are imported, each Topic vertex is augmented with two derived attributes:

```
subscriber_count(t) = |{ a ∈ V_app : (a, t) ∈ SUBSCRIBES_TO }|
publisher_count(t)  = |{ a ∈ V_app : (a, t) ∈ PUBLISHES_TO  }|
```

`subscriber_count` is the primary fan-out signal for reliability analysis — a Topic with high fan-out is a natural single point of failure for data distribution. These attributes are subsequently available to Step 2 (Analysis) as raw node features and to Step 3 (Prediction) for cascade depth weighting.

### Phase 3 — Dependency Derivation

Structural edges reveal physical relationships but not logical dependencies. This phase computes **DEPENDS_ON** edges — directed edges meaning "if the target fails, the source is affected." Five derivation rules are applied:

| Rule | Pattern | Resulting DEPENDS_ON Edge | Dependency Type |
|------|---------|--------------------------|-----------------| 
| **app_to_app** | App_sub `SUBSCRIBES_TO` → Topic ← `PUBLISHES_TO` App_pub | App_sub → App_pub | Data dependency |
| **app_to_broker** | App `PUBLISHES_TO` or `SUBSCRIBES_TO` → Topic ← `ROUTES` Broker | App → Broker | Routing dependency |
| **node_to_node** | Node_a hosts App_pub → Topic ← App_sub hosted on Node_b | Node_b → Node_a | Infrastructure dependency |
| **node_to_broker** | Node `RUNS_ON` App → Topic ← `ROUTES` Broker | Node → Broker | Cross-layer dependency |
| **app_to_lib** | App `USES` → Library | App → Library | Code dependency |

**Reading the edge direction:** DEPENDS_ON points from the *dependent* to the *dependency*. `App_sub → App_pub` means the subscriber depends on the publisher — if the publisher fails, the subscriber loses its data source. `App → Library` means the application depends on the library — if the library fails, the application is immediately affected.

**Why Rule 5 matters for reliability:** Without it, Library nodes have `DG_in = 0` from the dependency graph's perspective, making `R(Library) ≈ 0` regardless of how many applications use them. A library consumed by 15 applications has a blast radius of 15 — larger than most application-level failures. Rule 5 makes this visible to Steps 2 and 3 without requiring any additional input or runtime data.

**Library failure semantics vs. pub-sub cascade:** Library failures produce a *simultaneous* multi-consumer blast — all dependent applications fail at the same instant, not sequentially. This is structurally different from pub-sub cascade propagation (Rule 1), which flows through topics and brokers step by step. The Step 4 (Simulation) cascade propagation model handles this distinction at the simulation layer; the Modeling step simply needs to record the dependency.

These derived edges form the **G_analysis(l)** graph used by Steps 2–3. The separation is deliberate: centrality analysis needs abstracted dependency edges; cascade simulation needs the raw structural graph.

**Example derivation trace:**

```
Given:
  SensorApp    --[PUBLISHES_TO]--> /temperature
  MonitorApp   --[SUBSCRIBES_TO]--> /temperature
  MainBroker   --[ROUTES]---------> /temperature
  SensorApp    --[USES]-----------> NavLib
  MonitorApp   --[USES]-----------> NavLib

Derived DEPENDS_ON edges:
  MonitorApp --[DEPENDS_ON, app_to_app]-->    SensorApp   (data dependency)
  MonitorApp --[DEPENDS_ON, app_to_broker]--> MainBroker  (routing dependency)
  SensorApp  --[DEPENDS_ON, app_to_broker]--> MainBroker  (routing dependency)
  SensorApp  --[DEPENDS_ON, app_to_lib]-->    NavLib      (code dependency) ← NEW
  MonitorApp --[DEPENDS_ON, app_to_lib]-->    NavLib      (code dependency) ← NEW
```

After derivation: `DG_in(NavLib) = 2`, `DG_in(SensorApp) = 1`. Both are now visible to the R(v) formula.

**Multi-path coupling and `path_count`:**
When the same pair (App_sub, App_pub) communicates through multiple topics, a single DEPENDS_ON edge is created with:

```python
edge.weight      = max(w(t) for t in shared_topics)   # worst-case QoS weight
edge.path_count  = len(shared_topics)                  # coupling intensity
```

`path_count > 1` indicates that disrupting any one of the shared topics independently degrades the dependency, not just disrupts it entirely. This is recorded as edge metadata for downstream use by Step 3.

### Phase 4 — Weight Assignment

Weights encode dependency strength. A `DEPENDS_ON` edge with weight 1.0 represents a critical, high-priority, reliable data stream — disrupting it has maximum impact. A weight near 0 represents a low-priority, best-effort stream — less critical.

#### Topic Weight Formula

```
w(topic) = max(MIN_WEIGHT, β × QoS_score + (1−β) × size_norm)

QoS_score  = 0.30 × reliability_score + 0.40 × durability_score + 0.30 × priority_score
size_norm  = min(log₂(1 + size_kb) / 50, 1.0)     where size_kb = size_bytes / 1024
β          = 0.85
MIN_WEIGHT = 0.01
```

**AHP justification for β:** QoS semantics are the primary signal for dependency criticality; payload size is a secondary amplifier. The 0.85 weighting preserves the primacy of the QoS contract while allowing message volume to modulate the final score within the [0, 1] range.

| Component | Symbolic Value | Score |
|-----------|----------------|-------|
| **reliability_score** | `RELIABLE` | 1.0 |
| | `BEST_EFFORT` | 0.0 |
| **durability_score** | `PERSISTENT` | 1.0 |
| | `TRANSIENT` | 0.6 |
| | `TRANSIENT_LOCAL` | 0.5 |
| | `VOLATILE` | 0.0 |
| **priority_score** | `URGENT` | 1.0 |
| | `HIGH` | 0.66 |
| | `MEDIUM` | 0.33 |
| | `LOW` | 0.0 |

**AHP justification for QoS weights:** Durability (0.40) outweighs Reliability and Priority (0.30 each) because durability defines message state survival — fundamental for resilience — while reliability and priority govern transient delivery quality.

#### Weight Propagation

Once topics have weights, those weights propagate to the vertices that depend on them:

- **Application weight** = `max(w(t))` over all topics t the application publishes to or subscribes to
- **Broker weight** = `0.70 × max(w(t)) + 0.30 × mean(w(t))` over all topics t the broker routes
- **Node weight** = `max(w(v))` over all applications and brokers hosted on the node

**Application** uses `max()` because an application's criticality is bounded by the most critical stream it handles — a single URGENT/PERSISTENT stream makes the application critical regardless of how many LOW/BEST_EFFORT streams it also handles.

**Broker** uses a hybrid formula because brokers aggregate system-wide routing exposure. A broker routing 20 medium-weight topics carries materially more cumulative risk than one routing a single high-weight topic — yet `max()` alone assigns them the same weight. The hybrid captures both worst-case exposure (0.70 × max) and accumulated routing load (0.30 × mean). When a broker routes only one topic, `mean = max` and the formula collapses to `w = max`, preserving backward compatibility.

**Node** uses `max()` because a node's hardware failure takes down all its hosted components simultaneously; the worst-case hosted component determines the node's criticality tier.

#### DEPENDS_ON Edge Weight

```
w(App_sub → App_pub) = max(w(t) for t in shared_topics(App_sub, App_pub))
```

The `max()` captures the worst-case data flow. `path_count` (the count of shared topics) is stored as a separate edge attribute — see [Phase 3](#phase-3--dependency-derivation) — and is not folded into the weight to preserve the [0,1] range contract.

For `app_to_lib` edges, the edge weight is propagated from the application's own QoS weight:

```
w(App → Library) = w(App)
```

This reflects the severity of the applications that depend on the library. A library consumed only by high-priority applications carries a higher weight than one consumed only by low-priority applications.

---

## Layer Projections

The graph supports four layer projections, each filtering vertices and edges to a specific architectural concern.

| Layer | Label | Vertex Types | DEPENDS_ON Types |
|-------|-------|-------------|-----------------|
| Application | `app` | Application | app_to_app |
| Infrastructure | `infra` | Node | node_to_node |
| Middleware | `mw` | Application, Broker, Node | app_to_broker, node_to_broker |
| System | `system` | All five types | All five types |

> **Note:** `app_to_lib` dependencies are included in the `system` layer and can be isolated by filtering `dependency_type = 'app_to_lib'`. They are intentionally excluded from the `app` layer to keep Application-layer analysis focused on pub-sub data flow; Library blast-radius analysis is performed at the `system` layer.

---

## Two Graph Views

| Graph | Contains | Used By |
|-------|----------|---------|
| **G_structural** | All vertices + 6 structural edge types (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO, USES) | Step 4 (Simulation): cascade propagation follows physical paths |
| **G_analysis(l)** | Layer-filtered vertices + DEPENDS_ON edges only | Steps 2–3 (Analysis + Prediction): centrality metrics operate on abstract dependency graph |

The separation is deliberate and methodologically important: **prediction and simulation must remain independent**. Centrality metrics in Step 2 must not be contaminated by simulation outcomes, and simulation in Step 4 must not use prediction scores as inputs. Using separate graph views enforces this contract structurally.

---

## Input Format

The topology JSON schema:

```json
{
  "applications": [
    {
      "id": "sensor_app",
      "name": "SensorApp",
      "role": "publisher",
      "app_type": "sensor",
      "loc": 1200,
      "cyclomatic_complexity": 3.2
    }
  ],
  "brokers": [
    { "id": "main_broker", "name": "MainBroker" }
  ],
  "topics": [
    {
      "id": "temp_topic",
      "name": "/temperature",
      "size_bytes": 64,
      "qos": {
        "reliability": "RELIABLE",
        "durability": "TRANSIENT_LOCAL",
        "transport_priority": "HIGH"
      }
    }
  ],
  "nodes": [
    { "id": "compute_node_1", "name": "ComputeNode1" }
  ],
  "libraries": [
    {
      "id": "nav_lib",
      "name": "NavLib",
      "version": "2.1.0",
      "loc": 4500,
      "cyclomatic_complexity": 5.8,
      "lcom": 0.12
    }
  ],
  "relationships": [
    { "from": "sensor_app", "to": "temp_topic",      "type": "PUBLISHES_TO" },
    { "from": "monitor_app","to": "temp_topic",      "type": "SUBSCRIBES_TO" },
    { "from": "main_broker","to": "temp_topic",      "type": "ROUTES" },
    { "from": "sensor_app", "to": "compute_node_1",  "type": "RUNS_ON" },
    { "from": "sensor_app", "to": "nav_lib",         "type": "USES" },
    { "from": "monitor_app","to": "nav_lib",         "type": "USES" }
  ]
}
```

---

## Worked Example

**Given topology:** SensorApp publishes to `/temperature`; MonitorApp subscribes. Both use NavLib. MainBroker routes `/temperature`. `/temperature` has QoS `RELIABLE / TRANSIENT_LOCAL / HIGH`.

**Phase 1 — Entity Modeling:**
```
Vertices created: SensorApp (App), MonitorApp (App), /temperature (Topic), MainBroker (Broker), NavLib (Library)
Topic /temperature: subscriber_count = TBD (computed in Phase 2), publisher_count = TBD
```

**Phase 2 — Structural Graph:**
```
PUBLISHES_TO:  SensorApp  → /temperature
SUBSCRIBES_TO: MonitorApp → /temperature
ROUTES:        MainBroker → /temperature
USES:          SensorApp  → NavLib
USES:          MonitorApp → NavLib

Fan-out augmentation:
  /temperature.subscriber_count = 1  (MonitorApp)
  /temperature.publisher_count  = 1  (SensorApp)
```

**Phase 3 — Dependency Derivation (all 5 rules applied):**
```
Rule 1 (app_to_app):    MonitorApp --[DEPENDS_ON]--> SensorApp  (w = w(/temperature))
Rule 2 (app_to_broker): MonitorApp --[DEPENDS_ON]--> MainBroker (w = w(/temperature))
Rule 2 (app_to_broker): SensorApp  --[DEPENDS_ON]--> MainBroker (w = w(/temperature))
Rule 5 (app_to_lib):    SensorApp  --[DEPENDS_ON]--> NavLib     (w = w(SensorApp))
Rule 5 (app_to_lib):    MonitorApp --[DEPENDS_ON]--> NavLib     (w = w(MonitorApp))
```

**Phase 4 — Weight Assignment:**
```
QoS_score(/temperature) = 0.30×1.0 + 0.40×0.5 + 0.30×0.66 = 0.30 + 0.20 + 0.198 = 0.698
size_norm(/temperature) ≈ 0.0017  [64 bytes]
w(/temperature) = max(0.01, 0.85 × 0.698 + 0.15 × 0.0017) ≈ 0.59

w(SensorApp)  = max(w(/temperature)) = 0.59
w(MonitorApp) = max(w(/temperature)) = 0.59
w(MainBroker) = 0.70 × 0.59 + 0.30 × 0.59 = 0.59   [single topic; mean = max]
w(NavLib)     = 0.59  [propagated from consuming apps]
```

**DEPENDS_ON edge weights:**
```
MonitorApp → SensorApp:  0.59
MonitorApp → MainBroker: 0.59
SensorApp  → MainBroker: 0.59
SensorApp  → NavLib:     0.59
MonitorApp → NavLib:     0.59
```

**Resulting reliability-relevant vertex properties:**
```
DG_in(SensorApp)  = 1  (MonitorApp depends on it)
DG_in(MainBroker) = 2  (both apps depend on it)
DG_in(NavLib)     = 2  (both apps depend on it)  ← enabled by Rule 5
```

These are the inputs that will drive high R(v) scores in Step 3 Prediction.

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
| PUBLISHES_TO | publish() call | produce() call | publish() call |
| SUBSCRIBES_TO | subscription() call | consume() call | subscribe() call |
| USES | package.xml dep | pom.xml dep | requirements.txt |

---

## Complexity

| Phase | Operation | Complexity | Notes |
|-------|-----------|------------|-------|
| Phase 1 | Vertex creation | O(\|V\|) | One vertex per entity |
| Phase 2 | Structural edge import | O(\|E_s\|) | One pass over relationships |
| Phase 2 post-step | Fan-out augmentation | O(\|E_s\|) | One pass over SUBSCRIBES_TO and PUBLISHES_TO |
| Phase 3 | app_to_app derivation | O(\|Apps\|² × \|Topics\|) | All subscriber–publisher pairs per topic |
| Phase 3 | app_to_lib derivation | O(\|Apps\| × \|Libs\|) | Bounded by USES edge count in practice |
| Phase 3 | Other rules | O(\|E_s\|) | One pass per rule |
| Phase 4 | Weight propagation | O(\|E_s\|) | One pass over structural edges |

The dominant cost is Phase 3 app_to_app (dependency derivation). In practice, topic fan-out is bounded (typically 1–12 subscribers), so the effective cost is much lower than the worst case. app_to_lib is bounded by the number of USES edges (typically sparse), adding negligible overhead. Critically, this cost is incurred **once at design time**, in contrast to continuous runtime monitoring overhead.

---

## Commands

```bash
# Generate a synthetic system topology
python bin/generate_graph.py --scale medium --output input/system.json

# Or from a custom YAML config
python bin/generate_graph.py --config input/config/medium_scale.yaml --output input/system.json

# Import into Neo4j — runs all four construction phases automatically
python bin/import_graph.py --input input/system.json --clear

# Verify the result in Neo4j Browser (http://localhost:7474)
```

```cypher
-- Count vertices by type
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS count ORDER BY count DESC;

-- Check structural edges
MATCH ()-[r]->() WHERE type(r) <> 'DEPENDS_ON'
RETURN type(r) AS edge_type, count(*) AS count ORDER BY count DESC;

-- Inspect top derived dependencies by weight
MATCH (a)-[d:DEPENDS_ON]->(b)
RETURN a.name AS dependent, d.dependency_type, b.name AS dependency, d.weight, d.path_count
ORDER BY d.weight DESC LIMIT 20;

-- Verify Rule 5: inspect library in-degree
MATCH (lib:Library)
OPTIONAL MATCH (app)-[d:DEPENDS_ON {dependency_type: 'app_to_lib'}]->(lib)
RETURN lib.name AS library, count(d) AS dependent_app_count
ORDER BY dependent_app_count DESC;

-- Find multi-path couplings (path_count > 1)
MATCH (a)-[d:DEPENDS_ON]->(b)
WHERE d.path_count > 1
RETURN a.name AS dependent, b.name AS dependency, d.path_count, d.weight
ORDER BY d.path_count DESC LIMIT 10;

-- Find high-weight topics with large fan-out (reliability hotspots)
MATCH (t:Topic)
RETURN t.name, t.weight, t.subscriber_count, t.publisher_count,
       t.weight * t.subscriber_count AS blast_potential
ORDER BY blast_potential DESC LIMIT 10;

-- Verify broker hybrid weight (should exceed max topic weight when routing multiple topics)
MATCH (b:Broker)-[:ROUTES]->(t:Topic)
WITH b, max(t.weight) AS max_w, avg(t.weight) AS mean_w, count(t) AS n_topics
RETURN b.name, n_topics, max_w, mean_w,
       round(0.70 * max_w + 0.30 * mean_w, 4) AS expected_hybrid_weight,
       b.weight AS actual_weight;
```

### Scale Presets

| Scale | Apps | Topics | Brokers | Nodes | Typical Use |
|-------|------|--------|---------|-------|-------------|
| `tiny` | 5–8 | 3–5 | 1 | 2–3 | Unit tests |
| `small` | 10–15 | 8–12 | 2 | 3–4 | Quick validation |
| `medium` | 20–35 | 15–25 | 3–5 | 5–8 | Development |
| `large` | 50–80 | 30–50 | 5–8 | 8–12 | Integration tests |
| `xlarge` | 100–200 | 60–100 | 8–15 | 15–25 | Performance tests |

---

## What Comes Next

Step 1 produces the weighted dependency graph G_analysis(l). The five-rule DEPENDS_ON graph, enriched with `subscriber_count` on Topics, `path_count` on edges, and hybrid-weighted Broker vertices, gives Step 2 (Analysis) the densest possible structural signal for computing Reverse PageRank, betweenness centrality, bridge detection, and articulation point scores.

Step 2 (Analysis) applies graph centrality algorithms to this graph — Reverse PageRank, betweenness centrality, bridge detection, articulation point analysis, and eight more metrics — to build a complete structural profile of each component.

The key insight is that these purely topological metrics, computed without any runtime data, are strong predictors of real-world failure impact. The remainder of the pipeline quantifies exactly how strong.

---

[README](../README.md) | → [Step 2: Analysis](structural-analysis.md)