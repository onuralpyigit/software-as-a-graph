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

Modeling takes a distributed publish-subscribe system — its applications, topics, brokers, and infrastructure nodes — and converts it into a formal weighted directed graph. This graph becomes the foundation for all subsequent steps.

The process has four phases:

```
                  Phase 1           Phase 2           Phase 3           Phase 4
                    │                 │                 │                 │
System Topology  ──▶│  Entity      ──▶│  Structural  ──▶│  Dependency  ──▶│  Weighted
   (JSON input)     │  Modeling       │  Graph          │  Derivation     │  Graph
                    │                 │                 │                 │
                    │  Vertices for   │  6 structural   │  4 DEPENDS_ON   │  QoS-based
                    │  each entity    │  edge types     │  rules applied  │  weights on
                    │  type           │  imported       │  to derive      │  all edges
                    │                 │  from JSON      │  dependencies   │  and vertices
```

The output is two complementary graphs — **G_structural** for simulation and **G_analysis(l)** for analysis — described in [Two Graph Views](#two-graph-views).

---

## Why a Graph?

In a pub-sub system, applications don't call each other directly — they communicate through topics and brokers. A raw architecture diagram doesn't reveal the true dependency chains. By deriving logical DEPENDS_ON relationships, we make hidden dependencies explicit:

- If App A publishes to Topic T and App B subscribes to Topic T, then **B depends on A**. If A crashes, B is starved of data.
- If two applications share a broker, they have an **infrastructure dependency**. If the broker fails, both are affected.
- If two brokers share a host node, they have a **colocation dependency**. A node failure takes both down.

These derived dependencies are what make the graph useful for predicting failure impact.

---

## Formal Graph Definition

```
G(V, E, w) where:

V = V_app ∪ V_broker ∪ V_topic ∪ V_node ∪ V_lib
    (Applications, Brokers, Topics, Infrastructure Nodes, Libraries)

E_structural ⊆ V × V     (6 structural edge types — imported from topology JSON)
E_dependency ⊆ V × V     (DEPENDS_ON edges — derived by 4 derivation rules)

w: E → [0, 1]             (QoS-derived edge weight)
w: V → [0, 1]             (QoS-derived vertex weight, propagated from incident edges)
```

---

## Construction Phases

### Phase 1 — Entity Modeling

Each entity in the system topology JSON becomes a vertex in G. Five vertex types are created:

| Vertex Type | Source | Properties |
|-------------|--------|------------|
| **Application** | `applications[]` | id, name, role (pub/sub/pubsub), app_type, loc *(opt)*, cyclomatic_complexity *(opt)*, coupling_afferent *(opt)*, coupling_efferent *(opt)*, lcom *(opt)* |
| **Broker** | `brokers[]` | id, name |
| **Topic** | `topics[]` | id, name, size, QoS policy |
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

Library vertices model shared code dependencies. Their failure (e.g., a shared library crash or incompatible update) can trigger simultaneous failures across many applications — a pattern this model makes visible.

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

### Phase 3 — Dependency Derivation

Structural edges reveal physical relationships but not logical dependencies. This phase computes **DEPENDS_ON** edges — directed edges meaning "if the target fails, the source is affected." Four derivation rules are applied:

| Rule | Pattern | Resulting DEPENDS_ON Edge | Dependency Type |
|------|---------|--------------------------|-----------------| 
| **app_to_app** | App_sub `SUBSCRIBES_TO` → Topic ← `PUBLISHES_TO` App_pub | App_sub → App_pub | Data dependency |
| **app_to_broker** | App `PUBLISHES_TO` or `SUBSCRIBES_TO` → Topic ← `ROUTES` Broker | App → Broker | Routing dependency |
| **node_to_node** | Node_a hosts App_pub → Topic ← App_sub hosted on Node_b | Node_b → Node_a | Infrastructure dependency |
| **node_to_broker** | Node `RUNS_ON` App → Topic ← `ROUTES` Broker | Node → Broker | Cross-layer dependency |

**Reading the edge direction:** DEPENDS_ON points from the *dependent* to the *dependency*. `App_sub → App_pub` means the subscriber depends on the publisher — if the publisher fails, the subscriber loses its data source.

These derived edges form the **G_analysis(l)** graph used by Steps 2–3. The separation is deliberate: centrality analysis needs abstracted dependency edges; cascade simulation needs the raw structural graph.

**Example derivation trace:**

```
Given:
  SensorApp  --[PUBLISHES_TO]--> /temperature
  MonitorApp --[SUBSCRIBES_TO]--> /temperature
  MainBroker --[ROUTES]---------> /temperature

Derived DEPENDS_ON edges:
  MonitorApp --[DEPENDS_ON, app_to_app]-->    SensorApp   (data dependency)
  MonitorApp --[DEPENDS_ON, app_to_broker]--> MainBroker  (routing dependency)
  SensorApp  --[DEPENDS_ON, app_to_broker]--> MainBroker  (routing dependency)
```

### Phase 4 — Weight Assignment

Weights encode dependency strength. A `DEPENDS_ON` edge with weight 1.0 represents a critical, high-priority, reliable data stream — disrupting it has maximum impact. A weight near 0 represents a low-priority, best-effort stream — less critical.

#### Topic Weight Formula

```
w(topic) = max(MIN_WEIGHT, QoS_score + size_weight)

QoS_score  = 0.30 × reliability_score + 0.40 × durability_score + 0.30 × priority_score
size_weight = min(log₂(1 + size_kb) / 50, 0.20)     where size_kb = size_bytes / 1024
MIN_WEIGHT  = 0.01
```

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
- **Broker weight** = `max(w(t))` over all topics t the broker routes
- **Node weight** = `max(w(v))` over all applications and brokers hosted on the node

The `max` aggregation reflects a conservative risk model: a component's criticality is bounded by the most critical data stream it handles. This weight then augments the centrality-based prediction score in Step 3.

#### DEPENDS_ON Edge Weight

```
w(App_sub → App_pub) = w(topic T through which the dependency flows)
```

If multiple topics connect two components, the edge weight is the maximum over those topics.

---

## Layer Projections

The graph supports four layer projections, each filtering vertices and edges to a specific architectural concern.

| Layer | Analysed Component Types | DEPENDS_ON Subtypes Included | Quality Focus | What It Reveals |
|-------|--------------------------|------------------------------|---------------|-----------------|
| **`app`** | Application | `app_to_app` | Reliability | Application-level coupling and data flow |
| **`infra`** | Node | `node_to_node` | Availability | Infrastructure topology and colocation risk |
| **`mw`** | Broker | `app_to_broker`, `node_to_broker` | Maintainability | Middleware routing bottlenecks and broker coupling |
| **`system`** | Application, Broker, Node | All four subtypes | Overall | Full cross-layer critical component picture |

> **Implementation note:** The `mw` layer subgraph includes Application and Node vertices temporarily to preserve incoming DEPENDS_ON edges that point to Broker vertices. Only Broker components appear in the final analysis results.

Research has shown that the **application layer consistently produces the strongest predictions** (highest Spearman ρ and F1-score), because software dependencies are more tightly coupled to failure impact than infrastructure topology.

---

## Two Graph Views

Construction produces two complementary graphs that are used by different downstream steps:

| Graph | Contains | Purpose |
|-------|----------|---------| 
| **G_structural** | All 5 vertex types + all 6 structural edge types | Step 4: Simulation — cascade propagation needs physical topology |
| **G_analysis(l)** | Layer-filtered vertices + derived DEPENDS_ON edges only | Steps 2–3: Centrality analysis and prediction |

The separation is methodologically important. Using raw structural edges for centrality analysis would mix physical and logical relationships in ways that distort metric interpretation. Using derived edges for simulation would miss physical cascade paths (e.g., a node failure taking down all hosted apps simultaneously).

---

## Input Format

System topology is supplied as a JSON file with six top-level sections:

```json
{
  "nodes": [
    { "id": "N0", "name": "Server-1" }
  ],
  "brokers": [
    { "id": "B0", "name": "MainBroker" }
  ],
  "topics": [
    {
      "id": "T0",
      "name": "/sensors/temperature",
      "size": 256,
      "qos": {
        "durability":         "PERSISTENT",
        "reliability":        "RELIABLE",
        "transport_priority": "HIGH"
      }
    }
  ],
  "libraries": [
    { "id": "L0", "name": "ros2_common", "version": "1.2.0" }
  ],
  "applications": [
    {
      "id": "A0", "name": "TempSensor", "role": "pub", "app_type": "sensor",
      "loc": 256, "cyclomatic_complexity": 3.2,
      "coupling_afferent": 0, "coupling_efferent": 2, "lcom": 0.12
    },
    { "id": "A1", "name": "TempController", "role": "sub", "app_type": "controller" }
  ],
  "relationships": {
    "runs_on":      [{"from": "A0", "to": "N0"}, {"from": "A1", "to": "N0"}, {"from": "B0", "to": "N0"}],
    "routes":       [{"from": "B0", "to": "T0"}],
    "publishes_to": [{"from": "A0", "to": "T0"}],
    "subscribes_to":[{"from": "A1", "to": "T0"}],
    "connects_to":  [],
    "uses":         [{"from": "A0", "to": "L0"}, {"from": "A1", "to": "L0"}]
  }
}
```

> **Note:** The `relationships` section uses `from`/`to` identifiers that must reference IDs defined in their respective entity sections. Missing references are logged as warnings during import and the edge is skipped.

---

## Worked Example

**Distributed Intelligent Factory (DIF) — minimal excerpt:**

```
Entities:
  PLC_Controller (A3)  — publishes to /control_commands (RELIABLE, PERSISTENT, URGENT, 512 B)
  HMI_Display    (A5)  — subscribes to /control_commands
  Local_Log      (A7)  — subscribes to /control_commands
  MainBroker     (B0)  — routes /control_commands
  Server_A       (N0)  — hosts A3, A5, A7, B0

Structural edges:
  A3 --[PUBLISHES_TO]--> /control_commands
  A5 --[SUBSCRIBES_TO]--> /control_commands
  A7 --[SUBSCRIBES_TO]--> /control_commands
  B0 --[ROUTES]--> /control_commands
  A3, A5, A7, B0 --[RUNS_ON]--> N0

Topic weight for /control_commands:
  QoS_score  = 0.30×1.0 + 0.40×1.0 + 0.30×1.0 = 1.0
  size_weight = min(log₂(1 + 0.5) / 50, 0.20)  ≈ 0.012
  w(T)       = 1.0   (capped at 1.0)

Derived DEPENDS_ON edges (all weight = 1.0):
  A5 → A3   (A5 depends on A3 via /control_commands)
  A7 → A3   (A7 depends on A3 via /control_commands)
  A5 → B0   (A5 depends on B0 for routing)
  A7 → B0   (A7 depends on B0 for routing)
  A3 → B0   (A3 depends on B0 for routing)
```

The PLC_Controller (A3) has high in-degree (A5 and A7 depend on it) and is connected to a maximum-weight topic. These are the inputs that will drive high R(v) and V(v) scores in Step 3 Prediction.

---

## Domain Mapping

The model maps naturally to different pub-sub middleware technologies:

| Graph Concept | ROS 2 / DDS | Apache Kafka | MQTT |
|---------------|-------------|--------------|------|
| Application | ROS Node | Producer / Consumer | MQTT Client |
| Topic | ROS Topic | Kafka Topic | MQTT Topic |
| Broker | DDS Participant | Kafka Broker | MQTT Broker |
| Node | Host / Container | Broker Host | Broker Server |
| PUBLISHES_TO | publish() call | produce() call | publish() call |
| SUBSCRIBES_TO | subscription() call | consume() call | subscribe() call |

---

## Complexity

| Phase | Operation | Complexity | Notes |
|-------|-----------|------------|-------|
| Phase 1 | Vertex creation | O(\|V\|) | One vertex per entity |
| Phase 2 | Structural edge import | O(\|E_s\|) | One pass over relationships |
| Phase 3 | Dependency derivation | O(\|Apps\|² × \|Topics\|) | All subscriber–publisher pairs per topic |
| Phase 4 | Weight propagation | O(\|E_s\|) | One pass over structural edges |

The dominant cost is Phase 3 (dependency derivation). In practice, topic fan-out is bounded (typically 1–12 subscribers), so the effective cost is much lower than the worst case. Critically, this cost is incurred **once at design time**, in contrast to continuous runtime monitoring overhead.

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

-- Inspect the top derived dependencies by weight
MATCH (a)-[d:DEPENDS_ON]->(b)
RETURN a.name AS dependent, d.dependency_type, b.name AS dependency, d.weight
ORDER BY d.weight DESC LIMIT 20;

-- Find high-weight topics (potential critical paths)
MATCH (t:Topic) RETURN t.name, t.weight ORDER BY t.weight DESC LIMIT 10;
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

Step 1 produces the weighted dependency graph G_analysis(l). Step 2 (Analysis) applies graph centrality algorithms to this graph — Reverse PageRank, betweenness centrality, bridge detection, articulation point analysis, and eight more metrics — to build a complete structural profile of each component.

The key insight is that these purely topological metrics, computed without any runtime data, are strong predictors of real-world failure impact. The remainder of the pipeline quantifies exactly how strong.

---

[README](../README.md) | → [Step 2: Analysis](structural-analysis.md)