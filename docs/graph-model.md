# Step 1: Graph Model Construction

**Turn your system architecture into a graph that captures who depends on whom and how strongly.**

[README](../README.md) | → [Step 2: Structural Analysis](structural-analysis.md)

---

## What This Step Does

Graph Model Construction takes a distributed publish-subscribe system — its applications, topics, brokers, and infrastructure nodes — and converts it into a formal weighted directed graph. This graph becomes the foundation for all subsequent analysis.

The process has four phases:

```
System Topology  →  Structural Graph  →  Dependency Graph  →  Weighted Graph
  (your input)       (raw entities        (+ derived           (+ QoS-based
                      and edges)           DEPENDS_ON)          weights)
```

## Why a Graph?

In a pub-sub system, applications don't call each other directly — they communicate through topics and brokers. A raw architecture diagram doesn't reveal the true dependency chains. By deriving logical DEPENDS_ON relationships, we make hidden dependencies explicit:

- If App A publishes to Topic T and App B subscribes to Topic T, then **B depends on A**.
- If two applications share a broker, they have an infrastructure dependency.
- If two brokers share a host node, they have a colocation dependency.

These derived dependencies are what make the graph useful for predicting failure impact.

## The Graph Model

A pub-sub system is modeled as a directed weighted graph **G = (V, E, w)** with typed vertices and edges.

### Vertex Types

| Type | What It Represents | Examples |
|------|-------------------|----------|
| **Application** | A software component that publishes or subscribes to topics | ROS node, Kafka consumer/producer, MQTT client |
| **Topic** | A named message channel with QoS settings | `/sensor/lidar`, `orders.created` |
| **Broker** | Message routing middleware | DDS participant, Kafka broker, MQTT broker |
| **Node** | Physical or virtual host | Container, VM, bare-metal server |
| **Library** | Shared code dependency | ROS package, shared module |

### Edge Types

Edges are divided into two categories:

**Structural edges** come directly from the system topology:

| Edge | Meaning |
|------|---------|
| `PUBLISHES_TO` | Application → Topic |
| `SUBSCRIBES_TO` | Application → Topic |
| `ROUTES` | Broker → Topic |
| `RUNS_ON` | Application/Broker → Node |
| `CONNECTS_TO` | Node → Node |
| `USES` | Application → Library |

**Derived edges** are computed from structural paths:

| Derived Edge | Rule | Meaning |
|-------------|------|---------|
| `app_to_app` | App₁ publishes → Topic ← App₂ subscribes | Data dependency |
| `app_to_broker` | App publishes → Topic ← Broker routes | Routing dependency |
| `node_to_node` | Node₁ hosts Broker₁ → Topic ← Broker₂ on Node₂ | Infrastructure dependency |
| `node_to_broker` | Node hosts App → Topic ← Broker routes | Cross-layer dependency |

### Edge Weights

Weights reflect dependency strength, derived from the QoS policies of the topics involved:

| QoS Property | Higher Weight When... |
|-------------|----------------------|
| Reliability | RELIABLE > BEST_EFFORT |
| Durability | PERSISTENT > VOLATILE |
| Priority | URGENT > NORMAL |
| Message Size | Larger payloads |

The weight formula combines these factors: `w(topic) = α·reliability + β·durability + γ·priority + δ·size`, then propagates to derived DEPENDS_ON edges as the maximum topic weight along the path.

## Two Graph Views

Construction produces two complementary graphs for use by later steps:

| Graph | Contains | Used By |
|-------|----------|---------|
| **G_structural** | All vertices and raw structural edges | Step 4 (Failure Simulation) — needs physical topology for cascade propagation |
| **G_analysis(l)** | Layer-filtered vertices and derived DEPENDS_ON edges only | Steps 2–3 (Analysis, Scoring) — needs abstracted dependencies for centrality metrics |

The layer projection `l` can be `app`, `infra`, `mw`, or `system` (all combined).

## Domain Mapping

The model is designed for pub-sub systems but maps naturally to different technologies:

| Graph Concept | ROS 2 / DDS | Apache Kafka | MQTT |
|--------------|-------------|--------------|------|
| Application | ROS Node | Producer / Consumer | MQTT Client |
| Topic | ROS Topic | Kafka Topic | MQTT Topic |
| Broker | DDS Participant | Kafka Broker | MQTT Broker |
| Node | Host / Container | Broker Host | Broker Server |

## Commands

```bash
# Generate a synthetic system topology
python bin/generate_graph.py --config config/medium_scale.yaml --output data/system.json

# Or use a scale preset (tiny / small / medium / large / xlarge)
python bin/generate_graph.py --scale medium --output data/system.json

# Import into Neo4j (performs all four construction phases automatically)
python bin/import_graph.py --input data/system.json --clear
```

### Scale Presets

| Scale | Apps | Topics | Brokers | Nodes | Use Case |
|-------|------|--------|---------|-------|----------|
| tiny | 5–8 | 3–5 | 1 | 2–3 | Unit tests |
| small | 10–15 | 8–12 | 2 | 3–4 | Quick validation |
| medium | 20–35 | 15–25 | 3–5 | 5–8 | Development |
| large | 50–80 | 30–50 | 5–8 | 8–12 | Integration tests |
| xlarge | 100–200 | 60–100 | 8–15 | 15–25 | Performance tests |

### Verify the Import

```cypher
-- Count entities by type
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS count;

-- Check derived dependencies
MATCH (a)-[d:DEPENDS_ON]->(b)
RETURN a.name, d.dependency_type, b.name, d.weight
ORDER BY d.weight DESC LIMIT 10;
```

## Complexity

The dominant cost is dependency derivation, which considers all subscriber–publisher pairs per topic: **O(|Apps|² × |Topics|)**. In practice, topic fan-out is bounded (typically 1–12 subscribers), so effective cost is much lower. Importantly, this cost is incurred once at design time, compared to continuous runtime monitoring costs.

---

[README](../README.md) | → [Step 2: Structural Analysis](structural-analysis.md)