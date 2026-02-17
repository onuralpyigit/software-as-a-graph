# Step 1: Graph Model Construction

**Turn your system architecture into a graph that captures who depends on whom and how strongly.**

[README](../README.md) | → [Step 2: Structural Analysis](structural-analysis.md)

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

Graph Model Construction takes a distributed publish-subscribe system — its applications, topics, brokers, and infrastructure nodes — and converts it into a formal weighted directed graph. This graph becomes the foundation for all subsequent analysis.

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

These derived dependencies are what make the graph useful for predicting failure impact. A component's position in this dependency graph — how many things depend on it, whether removing it splits the graph, how many shortest paths flow through it — reliably predicts the real-world blast radius of its failure.

---

## Formal Graph Definition

A pub-sub system is modeled as a directed weighted graph:

```
G = (V, E, w)
```

where:

- **V** is a finite set of typed vertices partitioned into five disjoint types:

  ```
  V = V_app ∪ V_topic ∪ V_broker ∪ V_node ∪ V_lib
  ```

- **E ⊆ V × V** is a set of typed directed edges, partitioned into structural edges E_s and derived edges E_d:

  ```
  E = E_s ∪ E_d,   E_s ∩ E_d = ∅
  ```

- **w: E → [0, 1]** is a weight function assigning each edge a normalized strength, derived from the QoS policies of the topics involved.

---

## Construction Phases

### Phase 1 — Entity Modeling

Each component in the system topology becomes a vertex with a type label and a unique identifier.

| Vertex Type | Symbol | What It Represents | Examples |
|-------------|--------|--------------------|----------|
| **Application** | V_app | Software component that publishes or subscribes to topics | ROS node, Kafka consumer/producer, MQTT client |
| **Topic** | V_topic | Named message channel with associated QoS settings | `/sensor/lidar`, `orders.created` |
| **Broker** | V_broker | Message routing middleware | DDS participant, Kafka broker, MQTT broker |
| **Node** | V_node | Physical or virtual infrastructure host | Container, VM, bare-metal server |
| **Library** | V_lib | Shared code dependency | ROS package, shared module |

Library vertices represent shared code that multiple applications depend on. Their failure (e.g., a shared library crash or incompatible update) can trigger simultaneous failures across many applications — a pattern this model makes visible.

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

Together, these six types capture the full physical topology of the system. They form **G_structural**, which is used directly by Step 4 (Failure Simulation) for cascade propagation.

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
  SensorApp  --[PUBLISHES_TO]-->  /temperature
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

Each topic receives an intrinsic weight derived from its QoS settings:

```
w(topic) = reliability_score + durability_score + priority_score + size_score
```

where each component is defined as:

| Component | Formula | Values |
|-----------|---------|--------|
| **reliability_score** | 0.30 if RELIABLE, else 0.0 | RELIABLE → 0.30, BEST_EFFORT → 0.0 |
| **durability_score** | 0.40 if PERSISTENT, 0.20 if TRANSIENT_LOCAL, else 0.0 | PERSISTENT → 0.40, TRANSIENT_LOCAL → 0.20, VOLATILE → 0.0 |
| **priority_score** | 0.30 if URGENT, 0.20 if HIGH, 0.10 if MEDIUM, else 0.0 | URGENT → 0.30, HIGH → 0.20, MEDIUM → 0.10, LOW → 0.0 |
| **size_score** | min(log₂(1 + size_bytes / 1024) / 10, 1.0) | 1 KB → 0.10, 8 KB → 0.32, 64 KB → 0.60, 1 MB → 1.0 |

The total is capped at 1.0. A default topic (VOLATILE, BEST_EFFORT, LOW priority, 1 KB payload) scores approximately 0.10. A fully critical topic (PERSISTENT, RELIABLE, URGENT, 1 MB payload) scores 1.0.

#### Weight Propagation

Once topics have weights, those weights propagate to the vertices that depend on them:

- **Application weight** = `max(w(t))` over all topics t the application publishes to or subscribes to
- **Broker weight** = `max(w(t))` over all topics t the broker routes
- **Node weight** = `max(w(v))` over all applications and brokers hosted on the node

The `max` aggregation reflects a conservative risk model: a component's criticality is bounded by the most critical data stream it handles. This weight then augments the centrality-based quality score in Step 3.

#### DEPENDS_ON Edge Weight

The weight of a derived DEPENDS_ON edge is set to the weight of the topic through which the dependency is established:

```
w(App_sub → App_pub) = w(topic T through which the dependency flows)
```

If multiple topics connect two components, the edge weight is the maximum over those topics.

---

## Layer Projections

The graph supports four layer projections, each filtering vertices and edges to a specific architectural concern. Analysis in Step 2 can be performed on any single layer or on all layers combined (`system`).

| Layer | Vertices Included | Structural Edges Included | What It Reveals |
|-------|-------------------|---------------------------|-----------------|
| **`app`** | Application, Library | PUBLISHES_TO, SUBSCRIBES_TO, USES | Application-level coupling and data flow |
| **`infra`** | Node | RUNS_ON, CONNECTS_TO | Infrastructure topology and colocation risk |
| **`mw`** | Broker | ROUTES, PUBLISHES_TO, SUBSCRIBES_TO | Middleware routing structure |
| **`system`** | All vertex types | All structural edge types | Full cross-layer picture |

For G_analysis(l), only the DEPENDS_ON edges relevant to the selected layer's vertex types are included. Research has shown that the **application layer consistently produces the strongest predictions** (highest Spearman ρ and F1-score), because software dependencies are more tightly coupled to failure impact than infrastructure topology.

---

## Two Graph Views

Construction produces two complementary graphs that are used by different downstream steps:

| Graph | Contains | Purpose |
|-------|----------|---------|
| **G_structural** | All 5 vertex types + all 6 structural edge types | Step 4: Failure Simulation — cascade propagation needs physical topology |
| **G_analysis(l)** | Layer-filtered vertices + derived DEPENDS_ON edges only | Steps 2–3: Centrality analysis and quality scoring |

The separation is methodologically important. Using raw structural edges for centrality analysis would mix physical and logical relationships in ways that distort metric interpretation. Using derived edges for simulation would miss physical cascade paths (e.g., a node failure taking down all hosted apps simultaneously).

---

## Input Format

System topology is supplied as a JSON file with six top-level sections:

```json
{
  "nodes": [
    {
      "id": "N0",         // Unique identifier (required)
      "name": "Server-1"  // Human-readable name (required)
    }
  ],
  "brokers": [
    {
      "id": "B0",
      "name": "MainBroker"
    }
  ],
  "topics": [
    {
      "id": "T0",
      "name": "/sensors/temperature",
      "size": 256,          // Message payload size in bytes (required for weight calculation)
      "qos": {
        "durability":        "PERSISTENT",   // PERSISTENT | TRANSIENT_LOCAL | VOLATILE
        "reliability":       "RELIABLE",     // RELIABLE | BEST_EFFORT
        "transport_priority": "HIGH"         // URGENT | HIGH | MEDIUM | LOW
      }
    }
  ],
  "libraries": [
    {
      "id": "L0",
      "name": "ros2_common",
      "version": "1.2.0"  // Optional
    }
  ],
  "applications": [
    {
      "id": "A0",
      "name": "TempSensor",
      "role": "pub",       // pub | sub | pubsub
      "app_type": "sensor" // sensor | service | controller | monitor | ...
    },
    {
      "id": "A1",
      "name": "TempController",
      "role": "sub",
      "app_type": "controller"
    }
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

This section traces a minimal system through all four construction phases to make each step concrete.

**System:** A temperature sensor application publishes readings to a critical topic. A controller subscribes and issues commands back. Both run on the same node, served by a single broker.

### System Topology (JSON input)

```json
{
  "nodes":    [{"id": "N0", "name": "EdgeServer"}],
  "brokers":  [{"id": "B0", "name": "MainBroker"}],
  "topics":   [
    {"id": "T0", "name": "/temp/reading",  "size": 64,   "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE",    "transport_priority": "URGENT"}},
    {"id": "T1", "name": "/temp/setpoint", "size": 32,   "qos": {"durability": "VOLATILE",   "reliability": "BEST_EFFORT", "transport_priority": "LOW"}}
  ],
  "applications": [
    {"id": "A0", "name": "TempSensor",     "role": "pub"},
    {"id": "A1", "name": "TempController", "role": "pubsub"}
  ],
  "relationships": {
    "runs_on":       [{"from": "A0", "to": "N0"}, {"from": "A1", "to": "N0"}, {"from": "B0", "to": "N0"}],
    "routes":        [{"from": "B0", "to": "T0"}, {"from": "B0", "to": "T1"}],
    "publishes_to":  [{"from": "A0", "to": "T0"}, {"from": "A1", "to": "T1"}],
    "subscribes_to": [{"from": "A1", "to": "T0"}]
  }
}
```

### Phase 1 — Vertices created

```
V = { N0:Node, B0:Broker, T0:Topic, T1:Topic, A0:Application, A1:Application }
```

### Phase 2 — Structural edges created (G_structural)

```
A0 --[PUBLISHES_TO]-->  T0
A1 --[PUBLISHES_TO]-->  T1
A1 --[SUBSCRIBES_TO]--> T0
B0 --[ROUTES]---------> T0
B0 --[ROUTES]---------> T1
A0 --[RUNS_ON]--------> N0
A1 --[RUNS_ON]--------> N0
B0 --[RUNS_ON]--------> N0
```

### Phase 3 — Dependency derivation (G_analysis)

Applying the four derivation rules:

```
Rule app_to_app:    A1 subscribes T0, A0 publishes T0  →  A1 --[DEPENDS_ON]--> A0
Rule app_to_broker: A0 publishes T0, B0 routes T0      →  A0 --[DEPENDS_ON]--> B0
Rule app_to_broker: A1 subscribes T0, B0 routes T0     →  A1 --[DEPENDS_ON]--> B0
Rule app_to_broker: A1 publishes T1, B0 routes T1      →  A1 --[DEPENDS_ON]--> B0  (same edge, max weight)
Rule node_to_broker: N0 hosts A0 → T0 ← B0 routes     →  N0 --[DEPENDS_ON]--> B0
```

Result: A1 depends on both A0 and B0. A0 depends on B0. N0 depends on B0.

### Phase 4 — Weight calculation

**Topic weights:**

```
T0 (/temp/reading):  PERSISTENT(0.40) + RELIABLE(0.30) + URGENT(0.30) + size(0.06) = 1.00  (capped)
T1 (/temp/setpoint): VOLATILE(0.00)   + BEST_EFFORT(0.00) + LOW(0.00) + size(0.05) = 0.05
```

**Component weights (max over their topics):**

```
TempSensor    (A0): publishes T0          → weight = max(1.00) = 1.00
TempController(A1): publishes T1, subs T0 → weight = max(1.00, 0.05) = 1.00
MainBroker    (B0): routes T0, T1        → weight = max(1.00, 0.05) = 1.00
EdgeServer    (N0): hosts A0, A1, B0     → weight = max(1.00, 1.00, 1.00) = 1.00
```

**Interpretation:** Every component in this tiny system is maximally weighted because it participates in the high-priority URGENT/RELIABLE/PERSISTENT stream T0. In larger systems with diverse QoS settings, weights differentiate components meaningfully.

---

## Domain Mapping

The model is designed for pub-sub systems but maps naturally to different middleware technologies:

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

For a system with 100 applications and 50 topics, the worst case is 100² × 50 = 500,000 pairs. With average fan-out of 5 subscribers, the practical count is closer to 5 × 50 = 250 DEPENDS_ON edges derived.

---

## Commands

```bash
# Generate a synthetic system topology
python bin/generate_graph.py --scale medium --output data/system.json

# Or from a custom YAML config
python bin/generate_graph.py --config config/medium_scale.yaml --output data/system.json

# Import into Neo4j — runs all four phases automatically
python bin/import_graph.py --input data/system.json --clear

# Verify the result in Neo4j
# (run these in Neo4j Browser at http://localhost:7474)
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

Step 1 produces the weighted dependency graph G_analysis(l). Step 2 applies graph centrality algorithms to this graph — PageRank, betweenness centrality, articulation point detection, and eight more metrics — to build a complete structural profile of each component.

The key insight is that these purely topological metrics, computed without any runtime data, are strong predictors of real-world failure impact. The remainder of the pipeline quantifies exactly how strong.

---

[README](../README.md) | → [Step 2: Structural Analysis](structural-analysis.md)