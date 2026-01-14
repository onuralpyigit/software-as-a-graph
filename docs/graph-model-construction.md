# Graph Model Construction

**Transform distributed pub-sub system topology into a weighted directed graph**

This document covers the first step of the Software-as-a-Graph methodology: constructing the graph model from system architecture, including data generation, Neo4j import, weight calculation, and dependency derivation.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Graph Model Definition](#2-graph-model-definition)
3. [Component Types (Vertices)](#3-component-types-vertices)
4. [Relationship Types (Edges)](#4-relationship-types-edges)
5. [Weight Calculation](#5-weight-calculation)
6. [Dependency Derivation](#6-dependency-derivation)
7. [Generating Graph Data](#7-generating-graph-data)
8. [Importing to Neo4j](#8-importing-to-neo4j)
9. [Multi-Layer Architecture](#9-multi-layer-architecture)
10. [Input Format Reference](#10-input-format-reference)
11. [Domain Mapping](#11-domain-mapping)

---

## 1. Overview

### What This Step Does

Graph Model Construction transforms a distributed publish-subscribe system into a formal graph representation:

```
┌─────────────────────┐          ┌─────────────────────┐
│  System Topology    │          │  Weighted Directed  │
│                     │    →     │  Graph in Neo4j     │
│  - Applications     │          │                     │
│  - Topics + QoS     │          │  - Vertices (N,B,T,A)│
│  - Brokers          │          │  - Structural Edges │
│  - Infrastructure   │          │  - Derived DEPENDS_ON│
│  - Relationships    │          │  - Computed Weights │
└─────────────────────┘          └─────────────────────┘
```

### Pipeline

```bash
# Step 1a: Generate synthetic data (or provide your own JSON)
python generate_graph.py --scale medium --output data/system.json

# Step 1b: Import into Neo4j with weight calculation and dependency derivation
python import_graph.py --input data/system.json --clear
```

### What Gets Created

| Entity | Description | Count (Medium Scale) |
|--------|-------------|---------------------|
| **Vertices** | Applications, Topics, Brokers, Nodes | ~50-80 |
| **Structural Edges** | PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON | ~100-200 |
| **Derived Edges** | DEPENDS_ON (4 subtypes) | ~50-150 |
| **Weights** | QoS-based weights on all edges and vertices | All entities |

---

## 2. Graph Model Definition

### Formal Definition

The system is modeled as a directed weighted graph:

```
G = (V, E, τ, w)
```

Where:
- **V** = set of vertices (system components)
- **E** ⊆ V × V = set of directed edges (relationships)
- **τ** = type functions for vertices and edges
- **w** = weight function

### Vertex Types

```
T_V = { Application, Broker, Topic, Node }
```

### Edge Types

```
T_E = T_structural ∪ T_derived

T_structural = { PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO }
T_derived    = { DEPENDS_ON }
```

### Dependency Subtypes

DEPENDS_ON edges have a `dependency_type` property:

```
D = { app_to_app, app_to_broker, node_to_node, node_to_broker }
```

---

## 3. Component Types (Vertices)

### Application

A software service that publishes and/or subscribes to topics.

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier (e.g., "App-0") |
| `name` | string | Human-readable name |
| `role` | enum | `pub`, `sub`, or `pubsub` |
| `weight` | float | Computed criticality weight |

**Roles:**
- `pub` — Only publishes messages (data source)
- `sub` — Only subscribes to messages (data sink)
- `pubsub` — Both publishes and subscribes (data processor)

### Topic

A named channel for message exchange with QoS policies.

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier |
| `name` | string | Topic name/path (e.g., "/sensors/lidar") |
| `size` | integer | Message payload size in bytes |
| `qos_reliability` | enum | `RELIABLE` or `BEST_EFFORT` |
| `qos_durability` | enum | `VOLATILE`, `TRANSIENT_LOCAL`, `TRANSIENT`, `PERSISTENT` |
| `qos_transport_priority` | enum | `LOW`, `MEDIUM`, `HIGH`, `URGENT` |
| `weight` | float | Computed from QoS and size |

### Broker

Middleware component responsible for message routing.

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier |
| `name` | string | Broker instance name |
| `weight` | float | Sum of routed topic weights |

**Examples:** ROS 2 DDS daemon, MQTT broker, Kafka broker, RabbitMQ

### Node

Physical or virtual infrastructure hosting applications and brokers.

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier |
| `name` | string | Hostname or instance name |
| `weight` | float | Sum of hosted component weights |

**Examples:** Kubernetes pod, Docker container, VM, physical server, edge device

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                     │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   │
│   │ Node-0  │   │ Node-1  │   │ Node-2  │   │ Node-3  │   │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘   │
│        │ RUNS_ON     │             │             │         │
│        ▼             ▼             ▼             ▼         │
│   ┌─────────────────────────────────────────────────────┐ │
│   │                APPLICATION LAYER                     │ │
│   │  ┌───────┐  ┌───────┐  ┌───────┐  ┌────────┐       │ │
│   │  │ App-A │  │ App-B │  │ App-C │  │ Broker │       │ │
│   │  └───┬───┘  └───┬───┘  └───┬───┘  └────┬───┘       │ │
│   │      │ PUB/SUB  │          │ PUB/SUB   │ ROUTES    │ │
│   │      ▼          ▼          ▼           ▼           │ │
│   │  ┌───────────────────────────────────────────────┐ │ │
│   │  │                 TOPIC LAYER                   │ │ │
│   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐      │ │ │
│   │  │  │ Topic-1 │  │ Topic-2 │  │ Topic-3 │      │ │ │
│   │  │  └─────────┘  └─────────┘  └─────────┘      │ │ │
│   │  └───────────────────────────────────────────────┘ │ │
│   └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Relationship Types (Edges)

### Structural Relationships

Explicit connections defined in system configuration.

| Relationship | Direction | Description |
|--------------|-----------|-------------|
| `PUBLISHES_TO` | Application → Topic | App sends messages to topic |
| `SUBSCRIBES_TO` | Application → Topic | App receives messages from topic |
| `ROUTES` | Broker → Topic | Broker handles message routing |
| `RUNS_ON` | App/Broker → Node | Component hosted on infrastructure |
| `CONNECTS_TO` | Node → Node | Network connectivity |

### Derived Relationships

Logical dependencies computed during import.

| Subtype | Direction | Derivation |
|---------|-----------|------------|
| `app_to_app` | Subscriber → Publisher | Via shared topics |
| `app_to_broker` | Application → Broker | Via routed topics |
| `node_to_node` | Node A → Node B | Via hosted app dependencies |
| `node_to_broker` | Node → Broker | Via hosted app-broker deps |

### Edge Summary Diagram

```
STRUCTURAL RELATIONSHIPS
========================

Application ──PUBLISHES_TO──▶ Topic
Application ──SUBSCRIBES_TO─▶ Topic
Broker ──────ROUTES─────────▶ Topic
Application ──RUNS_ON───────▶ Node
Broker ──────RUNS_ON────────▶ Node
Node ────────CONNECTS_TO────▶ Node


DERIVED DEPENDENCIES (computed during import)
=============================================

Subscriber App ──DEPENDS_ON──▶ Publisher App    [app_to_app]
Application ─────DEPENDS_ON──▶ Broker           [app_to_broker]
Node ────────────DEPENDS_ON──▶ Node             [node_to_node]
Node ────────────DEPENDS_ON──▶ Broker           [node_to_broker]
```

---

## 5. Weight Calculation

Weights represent component criticality and flow from Topics upward through the hierarchy.

### Weight Flow Hierarchy

```
Topics (QoS + Size)
    ↓ inherit
Structural Edges (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES)
    ↓ aggregate
Applications/Brokers (intrinsic weight)
    ↓ aggregate
Nodes (intrinsic weight)
    ↓ derive
DEPENDS_ON relationships (dependency weight)
    ↓ sum
Final Component Weight (intrinsic + centrality)
```

### Topic Weight (Base Metric)

The fundamental unit of weight, computed from QoS policies and message size:

```
W_topic = S_reliability + S_durability + S_priority + S_size
```

**QoS Scoring Tables:**

| Reliability | Score | | Durability | Score |
|-------------|-------|-|------------|-------|
| RELIABLE | 0.30 | | PERSISTENT | 0.40 |
| BEST_EFFORT | 0.00 | | TRANSIENT | 0.25 |
| | | | TRANSIENT_LOCAL | 0.20 |
| | | | VOLATILE | 0.00 |

| Priority | Score | | Message Size | Score |
|----------|-------|-|--------------|-------|
| URGENT | 0.30 | | 256 bytes | 0.03 |
| HIGH | 0.20 | | 1 KB | 0.10 |
| MEDIUM | 0.10 | | 8 KB | 0.32 |
| LOW | 0.00 | | 64 KB | 0.60 |
| | | | 1 MB+ | 1.00 (cap) |

**Size Score Formula** (logarithmic scaling):
```
S_size = min( log₂(1 + size/1024) / 10, 1.0 )
```

**Example Topic Weight:**
```
Topic: /sensors/lidar
- Reliability: RELIABLE       → 0.30
- Durability: TRANSIENT_LOCAL → 0.20
- Priority: HIGH              → 0.20
- Size: 64KB                  → 0.60
────────────────────────────────────
W_topic = 1.30
```

### Edge Weights

Structural edges inherit weights from connected topics:

```
W_PUBLISHES_TO  = W_connected_topic
W_SUBSCRIBES_TO = W_connected_topic
W_ROUTES        = W_routed_topic
W_RUNS_ON       = 1.0 (default)
W_CONNECTS_TO   = 1.0 (default)
```

### Component Intrinsic Weights

Intrinsic weight represents base importance before network effects.

**Application:**
```
W_app = Σ W_topic  (for all pub/sub topics)
```

**Broker:**
```
W_broker = Σ W_topic  (for all routed topics)
```

**Node:**
```
W_node = Σ W_app + Σ W_broker  (for all hosted components)
```

### Dependency Edge Weights

Dependency weights combine topic count and weight sum:

```
W_dep = |T_shared| + Σ W_topic
```

This ensures that many low-weight topics still create strong coupling.

### Final Component Weight

```
W_final(v) = W_intrinsic(v) + W_centrality(v)

W_centrality(v) = Σ W_dep(outgoing) + Σ W_dep(incoming)
```

**Why sum both directions?**
- **Incoming deps**: Impact — how many depend on this component
- **Outgoing deps**: Vulnerability — exposure to upstream failures

Hub components with high scores in both directions are critical integration points.

---

## 6. Dependency Derivation

Dependencies are automatically derived during import based on pub-sub patterns.

### App-to-App Dependencies

**Rule:** If App-A subscribes to Topic-X and App-B publishes to Topic-X, then App-A depends on App-B.

```
App-A (subscriber) ──DEPENDS_ON──▶ App-B (publisher)
```

**Direction rationale:** Subscriber depends on publisher because:
- Subscriber's functionality requires data from publisher
- Publisher failure causes subscriber data starvation
- Models failure propagation direction

**Example:**

```
Sensor-A ──PUBLISHES_TO──▶ /raw/a ◀──SUBSCRIBES_TO── Fusion
Sensor-B ──PUBLISHES_TO──▶ /raw/b ◀──SUBSCRIBES_TO── Fusion
Fusion ───PUBLISHES_TO──▶ /fused ◀──SUBSCRIBES_TO── Display
                                 ◀──SUBSCRIBES_TO── Logger

Derived Dependencies:
  Fusion ──DEPENDS_ON──▶ Sensor-A   (via /raw/a)
  Fusion ──DEPENDS_ON──▶ Sensor-B   (via /raw/b)
  Display ─DEPENDS_ON──▶ Fusion     (via /fused)
  Logger ──DEPENDS_ON──▶ Fusion     (via /fused)
```

### App-to-Broker Dependencies

**Rule:** If App uses any topic routed by Broker, then App depends on Broker.

```
Application ──DEPENDS_ON──▶ Broker
```

### Node-to-Node Dependencies

**Rule:** If any app on Node-A depends on any app on Node-B, then Node-A depends on Node-B.

```
Node-A ──DEPENDS_ON──▶ Node-B
```

Weight is aggregated from all app-to-app dependencies between the nodes.

### Node-to-Broker Dependencies

**Rule:** If any app on Node depends on Broker, then Node depends on Broker.

```
Node ──DEPENDS_ON──▶ Broker
```

### Derivation Algorithm (Pseudocode)

```python
# App-to-App Dependencies
for topic in topics:
    publishers = get_publishers(topic)
    subscribers = get_subscribers(topic)
    for sub in subscribers:
        for pub in publishers:
            if sub != pub:
                create_dependency(sub, pub, "app_to_app", topic.weight)

# App-to-Broker Dependencies
for app in applications:
    for topic in app.topics:
        broker = get_broker(topic)
        if broker:
            create_dependency(app, broker, "app_to_broker", topic.weight)

# Node-to-Node Dependencies (aggregation)
for dep in app_to_app_dependencies:
    node_a = get_node(dep.source)
    node_b = get_node(dep.target)
    if node_a != node_b:
        add_or_merge_dependency(node_a, node_b, "node_to_node", dep.weight)

# Node-to-Broker Dependencies (aggregation)
for dep in app_to_broker_dependencies:
    node = get_node(dep.source)
    add_or_merge_dependency(node, dep.target, "node_to_broker", dep.weight)
```

---

## 7. Generating Graph Data

### Using generate_graph.py

Generate synthetic pub-sub system graphs for testing and benchmarking.

```bash
python generate_graph.py --scale medium --output data/system.json --seed 42
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--scale` | tiny, small, medium, large, xlarge | medium | System size |
| `--output` | path | required | Output JSON file |
| `--seed` | integer | 42 | Random seed for reproducibility |

### Scale Definitions

| Scale | Apps | Topics | Brokers | Nodes | Use Case |
|-------|------|--------|---------|-------|----------|
| `tiny` | 5-8 | 3-5 | 1 | 2-3 | Unit tests |
| `small` | 10-15 | 8-12 | 1-2 | 3-4 | Quick validation |
| `medium` | 20-35 | 15-25 | 2-3 | 5-8 | Development |
| `large` | 50-80 | 30-50 | 3-5 | 8-12 | Integration tests |
| `xlarge` | 100-200 | 60-100 | 5-10 | 15-25 | Performance tests |

### Output Example

```bash
$ python generate_graph.py --scale medium --output data/system.json

Generating 'medium' graph (Seed: 42)...
Success! Saved to data/system.json
Stats: 6 Nodes, 25 Apps, 18 Topics, 2 Brokers
```

### Generated Graph Characteristics

The generator creates realistic pub-sub topologies with:

- **Mixed application roles**: Publishers, subscribers, and processors
- **Varied QoS policies**: Distribution across reliability/durability/priority levels
- **Realistic message sizes**: Range from small control messages to large sensor data
- **Connected topology**: Ensures graphs are connected (no isolated components)
- **Broker assignment**: Topics distributed across available brokers
- **Node placement**: Applications distributed across infrastructure nodes

---

## 8. Importing to Neo4j

### Using import_graph.py

Import generated or custom JSON data into Neo4j, computing weights and deriving dependencies.

```bash
python import_graph.py --input data/system.json --clear
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | required | Input JSON file path |
| `--uri` | bolt://localhost:7687 | Neo4j Bolt URI |
| `--user` | neo4j | Neo4j username |
| `--password` | password | Neo4j password |
| `--db` | neo4j | Database name |
| `--clear` | false | Clear existing data before import |

### Import Process

The importer performs these operations in order:

```
1. Clear Database (if --clear)
   └── Delete all nodes and relationships

2. Create Vertices
   ├── Nodes (infrastructure)
   ├── Brokers (middleware)
   ├── Topics (with QoS weights calculated)
   └── Applications (with role)

3. Create Structural Edges
   ├── RUNS_ON (app/broker → node)
   ├── ROUTES (broker → topic)
   ├── PUBLISHES_TO (app → topic)
   ├── SUBSCRIBES_TO (app → topic)
   └── CONNECTS_TO (node → node)

4. Calculate Weights
   ├── Topic weights (from QoS + size)
   ├── Edge weights (inherit from topics)
   ├── Application weights (aggregate topics)
   ├── Broker weights (aggregate routed topics)
   └── Node weights (aggregate hosted components)

5. Derive Dependencies
   ├── app_to_app (via shared topics)
   ├── app_to_broker (via routed topics)
   ├── node_to_node (aggregate app deps)
   └── node_to_broker (aggregate app-broker deps)

6. Update Final Weights
   └── Add centrality (dependency) weights to components
```

### Output Example

```bash
$ python import_graph.py --input data/system.json --clear

Reading data/system.json...
Connecting to Neo4j at bolt://localhost:7687...

Import & Derivation Complete!
------------------------------
Entities Imported:
  Nodes:       6
  Brokers:     2
  Topics:      18
  Apps:        25
------------------------------
Dependencies Derived:
  App->App:    42
  App->Broker: 25
  Node->Node:  15
  Node->Broker:6
------------------------------
Weight Calculation:
  - Intrinsic weights (QoS/Size) applied to Topics/Edges.
  - Aggregate weights applied to Apps/Brokers/Nodes.
  - Final criticality scores (Intrinsic + Centrality) updated.
------------------------------
```

### Neo4j Schema

After import, the database contains:

**Node Labels:**
```cypher
(:Application {id, name, role, weight})
(:Broker {id, name, weight})
(:Topic {id, name, size, qos_reliability, qos_durability, qos_transport_priority, weight})
(:Node {id, name, weight})
```

**Relationships:**
```cypher
// Structural
(:Application)-[:PUBLISHES_TO {weight}]->(:Topic)
(:Application)-[:SUBSCRIBES_TO {weight}]->(:Topic)
(:Broker)-[:ROUTES {weight}]->(:Topic)
(:Application)-[:RUNS_ON {weight}]->(:Node)
(:Broker)-[:RUNS_ON {weight}]->(:Node)
(:Node)-[:CONNECTS_TO {weight}]->(:Node)

// Derived
(:Application)-[:DEPENDS_ON {dependency_type: 'app_to_app', weight}]->(:Application)
(:Application)-[:DEPENDS_ON {dependency_type: 'app_to_broker', weight}]->(:Broker)
(:Node)-[:DEPENDS_ON {dependency_type: 'node_to_node', weight}]->(:Node)
(:Node)-[:DEPENDS_ON {dependency_type: 'node_to_broker', weight}]->(:Broker)
```

### Verifying Import

Query Neo4j to verify the import:

```cypher
// Count all entities
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS count;

// View dependencies
MATCH (a)-[d:DEPENDS_ON]->(b)
RETURN a.name, d.dependency_type, b.name, d.weight
ORDER BY d.weight DESC
LIMIT 10;

// Check weight distribution
MATCH (a:Application)
RETURN a.name, a.role, a.weight
ORDER BY a.weight DESC;
```

---

## 9. Multi-Layer Architecture

The graph supports analysis at multiple architectural layers.

### Layer Definitions

| Layer | Components | Dependencies | Analysis Focus |
|-------|------------|--------------|----------------|
| `app` | Applications | app_to_app | Service-level reliability |
| `infra` | Nodes | node_to_node | Network topology resilience |
| `mw-app` | Apps + Brokers | app_to_broker | Middleware dependency |
| `mw-infra` | Nodes + Brokers | node_to_broker | Infrastructure-middleware coupling |
| `system` | All | All | System-wide analysis |

### Layer Extraction

For analysis, layers are extracted as subgraphs:

```cypher
// Application layer
MATCH (a1:Application)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(a2:Application)
RETURN a1, d, a2;

// Infrastructure layer
MATCH (n1:Node)-[d:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2:Node)
RETURN n1, d, n2;

// Complete system
MATCH (a)-[d:DEPENDS_ON]->(b)
RETURN a, d, b;
```

### Cross-Layer Dependencies

Layers are connected through hosting and routing:

```
Infrastructure ←─RUNS_ON─── Applications
                           ├──────────── app_to_broker ──▶ Brokers
Infrastructure ←─RUNS_ON───────────────────────────────── Brokers
```

---

## 10. Input Format Reference

### JSON Schema

```json
{
  "nodes": [
    {
      "id": "N0",
      "name": "Server-1"
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
      "size": 256,
      "qos": {
        "durability": "PERSISTENT",
        "reliability": "RELIABLE",
        "transport_priority": "HIGH"
      }
    }
  ],
  "applications": [
    {
      "id": "A0",
      "name": "TempSensor",
      "role": "pub"
    },
    {
      "id": "A1",
      "name": "TempController",
      "role": "sub"
    }
  ],
  "relationships": {
    "runs_on": [
      {"from": "A0", "to": "N0"},
      {"from": "A1", "to": "N0"},
      {"from": "B0", "to": "N0"}
    ],
    "routes": [
      {"from": "B0", "to": "T0"}
    ],
    "publishes_to": [
      {"from": "A0", "to": "T0"}
    ],
    "subscribes_to": [
      {"from": "A1", "to": "T0"}
    ],
    "connects_to": []
  }
}
```

### Field Descriptions

**nodes[]**
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | Yes | string | Unique identifier |
| `name` | Yes | string | Display name |

**brokers[]**
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | Yes | string | Unique identifier |
| `name` | Yes | string | Display name |

**topics[]**
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | Yes | string | Unique identifier |
| `name` | Yes | string | Topic path/name |
| `size` | No | integer | Message size in bytes (default: 1024) |
| `qos.durability` | No | enum | VOLATILE, TRANSIENT_LOCAL, TRANSIENT, PERSISTENT |
| `qos.reliability` | No | enum | BEST_EFFORT, RELIABLE |
| `qos.transport_priority` | No | enum | LOW, MEDIUM, HIGH, URGENT |

**applications[]**
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | Yes | string | Unique identifier |
| `name` | Yes | string | Display name |
| `role` | No | enum | pub, sub, pubsub (default: pubsub) |

**relationships**
| Field | Type | Description |
|-------|------|-------------|
| `runs_on` | array | App/Broker → Node hosting |
| `routes` | array | Broker → Topic routing |
| `publishes_to` | array | App → Topic publishing |
| `subscribes_to` | array | App → Topic subscription |
| `connects_to` | array | Node → Node connectivity |

---

## 11. Domain Mapping

The graph model maps to various pub-sub middleware:

### ROS 2 / DDS

| Graph Concept | ROS 2 Equivalent |
|---------------|------------------|
| Application | ROS Node |
| Topic | ROS Topic |
| Broker | DDS Domain Participant |
| Node | Host / Container |
| PUBLISHES_TO | Publisher |
| SUBSCRIBES_TO | Subscription |
| QoS Policy | ROS 2 QoS settings |

### Apache Kafka

| Graph Concept | Kafka Equivalent |
|---------------|------------------|
| Application | Producer / Consumer |
| Topic | Kafka Topic (+Partition) |
| Broker | Kafka Broker |
| Node | Broker host / K8s pod |

### MQTT

| Graph Concept | MQTT Equivalent |
|---------------|------------------|
| Application | MQTT Client |
| Topic | MQTT Topic (hierarchical) |
| Broker | MQTT Broker |
| Node | Broker server |

### Generic Microservices

| Graph Concept | Microservices Equivalent |
|---------------|--------------------------|
| Application | Service instance |
| Topic | Event channel / Message queue |
| Broker | Message broker / Event bus |
| Node | Container / Pod / VM |

---

## Quick Reference

### Commands

```bash
# Generate graph data
python generate_graph.py --scale medium --output data/system.json

# Import to Neo4j
python import_graph.py --input data/system.json --clear

# Verify in Neo4j Browser
# Open http://localhost:7474 and run:
MATCH (n) RETURN n LIMIT 50;
```

### What Happens During Import

1. **Vertices created**: Nodes, Brokers, Topics, Applications
2. **Structural edges created**: RUNS_ON, ROUTES, PUBLISHES_TO, SUBSCRIBES_TO
3. **Weights calculated**: Topic QoS → Edges → Components
4. **Dependencies derived**: app_to_app, app_to_broker, node_to_node, node_to_broker
5. **Final weights updated**: Intrinsic + centrality weights

### Key Formulas

```
Topic Weight:     W_topic = S_rel + S_dur + S_pri + S_size
App Weight:       W_app = Σ W_topics
Dependency Weight: W_dep = |T_shared| + Σ W_topics
Final Weight:     W_final = W_intrinsic + Σ W_deps_in + Σ W_deps_out
```

---

## Next Step

After graph model construction, proceed to **Structural Analysis & Quality Scoring**. to compute topological metrics (PageRank, Betweenness, Articulation Points, etc.) that feed into the quality scoring formulas.

```bash
python analyze_graph.py --layer system
```

---

## Navigation

- **Previous**: [README](README.md)
- **Next**: [Structural Analysis & Quality Scoring](structural-analysis-quality-scoring.md)
