# Graph Model for Distributed Publish-Subscribe Systems

This document defines the formal graph model used to represent distributed publish-subscribe (pub-sub) systems. The model captures both structural relationships (physical topology) and logical dependencies (data flow patterns) to enable comprehensive quality assessment.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Formal Definition](#2-formal-definition)
3. [Component Types (Vertices)](#3-component-types-vertices)
4. [Relationship Types (Edges)](#4-relationship-types-edges)
5. [Multi-Layer Architecture](#5-multi-layer-architecture)
6. [Dependency Derivation](#6-dependency-derivation)
7. [Domain Mapping](#7-domain-mapping)
8. [Example Graph](#8-example-graph)
9. [Implementation](#9-implementation)

---

## 1. Overview

The Software-as-a-Graph approach models distributed pub-sub systems as directed graphs where:

- **Vertices** represent system components (applications, brokers, topics, infrastructure nodes)
- **Edges** represent relationships between components (data flow, hosting, routing)

This graph-based representation enables:

| Capability | Description |
|------------|-------------|
| **Structural Analysis** | Identify critical paths, bottlenecks, and single points of failure |
| **Quality Assessment** | Compute reliability, maintainability, and availability scores |
| **Failure Simulation** | Model cascade effects when components fail |
| **Architecture Optimization** | Guide refactoring and redundancy decisions |

### Design Principles

1. **Separation of Concerns**: Structural relationships (physical) are distinguished from derived dependencies (logical)
2. **Multi-Layer Analysis**: Different architectural layers can be analyzed independently or together
3. **Domain Agnostic**: The model applies to ROS 2, MQTT, Kafka, DDS, and other pub-sub middleware
4. **Weight-Aware**: All edges carry weights derived from QoS policies and message characteristics

---

## 2. Formal Definition

### 2.1 Graph Structure

The system is modeled as a directed weighted multigraph:

$$G = (V, E, \tau_V, \tau_E, w)$$

Where:
- $V$ = set of vertices (components)
- $E \subseteq V \times V$ = set of directed edges (relationships)
- $\tau_V : V \rightarrow T_V$ = vertex type function
- $\tau_E : E \rightarrow T_E$ = edge type function
- $w : E \rightarrow \mathbb{R}^+$ = edge weight function

### 2.2 Type Sets

**Vertex Types** ($T_V$):
$$T_V = \{\text{Application}, \text{Broker}, \text{Topic}, \text{Node}\}$$

**Edge Types** ($T_E$):
$$T_E = T_E^{structural} \cup T_E^{derived}$$

Where:
$$T_E^{structural} = \{\text{PUBLISHES\_TO}, \text{SUBSCRIBES\_TO}, \text{ROUTES}, \text{RUNS\_ON}, \text{CONNECTS\_TO}\}$$

$$T_E^{derived} = \{\text{DEPENDS\_ON}\}$$

### 2.3 Dependency Subtypes

Derived dependencies are further classified:

$$D = \{\text{app\_to\_app}, \text{app\_to\_broker}, \text{node\_to\_node}, \text{node\_to\_broker}\}$$

For each derived edge $e \in E$ where $\tau_E(e) = \text{DEPENDS\_ON}$, there exists a subtype function:

$$\delta : E_{derived} \rightarrow D$$

---

## 3. Component Types (Vertices)

### 3.1 Application

**Definition**: A software service that publishes and/or subscribes to topics.

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier |
| `name` | string | Human-readable name |
| `role` | enum | `pub`, `sub`, or `pubsub` |
| `weight` | float | Computed criticality weight |

**Roles**:
- `pub` — Only publishes messages (data source)
- `sub` — Only subscribes to messages (data sink)
- `pubsub` — Both publishes and subscribes (data processor)

**Symbol**: $a \in A$ where $A \subset V$

### 3.2 Topic

**Definition**: A named channel for message exchange with associated QoS policies.

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier |
| `name` | string | Topic name/path |
| `size` | integer | Message payload size (bytes) |
| `qos_reliability` | enum | `RELIABLE` or `BEST_EFFORT` |
| `qos_durability` | enum | `VOLATILE`, `TRANSIENT_LOCAL`, `TRANSIENT`, `PERSISTENT` |
| `qos_transport_priority` | enum | `LOW`, `MEDIUM`, `HIGH`, `URGENT` |
| `weight` | float | Computed from QoS and size |

**Symbol**: $t \in T$ where $T \subset V$

**Note**: Topics are intermediate vertices in the data flow graph. They do not have outgoing edges in the structural graph (only applications publish/subscribe to them).

### 3.3 Broker

**Definition**: Middleware component responsible for message routing and delivery.

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier |
| `name` | string | Broker instance name |
| `weight` | float | Sum of routed topic weights |

**Symbol**: $b \in B$ where $B \subset V$

**Examples**: ROS 2 DDS daemon, MQTT broker, Kafka broker, RabbitMQ node

### 3.4 Node

**Definition**: Physical or virtual infrastructure hosting applications and brokers.

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier |
| `name` | string | Hostname or instance name |
| `weight` | float | Sum of hosted component weights |

**Symbol**: $n \in N$ where $N \subset V$

**Examples**: Kubernetes pod, Docker container, VM instance, physical server, edge device

### 3.5 Component Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Node-0  │  │ Node-1  │  │ Node-2  │  │ Node-3  │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │
│       │            │            │            │          │
│       │ RUNS_ON    │ RUNS_ON    │ RUNS_ON    │ RUNS_ON  │
│       ▼            ▼            ▼            ▼          │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Application Layer                   │   │
│  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌────────┐      │   │
│  │  │ App-A │ │ App-B │ │ App-C │ │ Broker │      │   │
│  │  └───┬───┘ └───┬───┘ └───┬───┘ └────┬───┘      │   │
│  │      │         │         │          │           │   │
│  │      │ PUB/SUB │ PUB/SUB │ PUB/SUB  │ ROUTES   │   │
│  │      ▼         ▼         ▼          ▼           │   │
│  │  ┌─────────────────────────────────────────┐   │   │
│  │  │            Topic Layer                   │   │   │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │   │
│  │  │  │ Topic-1 │  │ Topic-2 │  │ Topic-3 │  │   │   │
│  │  │  └─────────┘  └─────────┘  └─────────┘  │   │   │
│  │  └─────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Relationship Types (Edges)

### 4.1 Structural Relationships

Structural relationships represent explicit, physical connections defined in the system configuration.

#### PUBLISHES_TO

**Definition**: Application sends messages to a topic.

| Property | Description |
|----------|-------------|
| Direction | Application → Topic |
| Weight | Inherited from topic weight |
| Cardinality | Many-to-many |

$$e = (a, t) \text{ where } a \in A, t \in T$$

#### SUBSCRIBES_TO

**Definition**: Application receives messages from a topic.

| Property | Description |
|----------|-------------|
| Direction | Application → Topic |
| Weight | Inherited from topic weight |
| Cardinality | Many-to-many |

$$e = (a, t) \text{ where } a \in A, t \in T$$

**Note**: The edge direction represents the subscription relationship, not data flow. For data flow analysis, this is traversed in reverse.

#### ROUTES

**Definition**: Broker is responsible for routing messages on a topic.

| Property | Description |
|----------|-------------|
| Direction | Broker → Topic |
| Weight | Inherited from topic weight |
| Cardinality | Many-to-many |

$$e = (b, t) \text{ where } b \in B, t \in T$$

#### RUNS_ON

**Definition**: Application or Broker is hosted on a Node.

| Property | Description |
|----------|-------------|
| Direction | Application/Broker → Node |
| Weight | 1.0 (default) |
| Cardinality | Many-to-one |

$$e = (c, n) \text{ where } c \in A \cup B, n \in N$$

#### CONNECTS_TO

**Definition**: Network connectivity between infrastructure nodes.

| Property | Description |
|----------|-------------|
| Direction | Node → Node (bidirectional in practice) |
| Weight | 1.0 (default, can represent bandwidth/latency) |
| Cardinality | Many-to-many |

$$e = (n_1, n_2) \text{ where } n_1, n_2 \in N, n_1 \neq n_2$$

### 4.2 Derived Relationships

Derived relationships are computed during graph import to model logical dependencies that emerge from pub-sub patterns.

#### DEPENDS_ON

**Definition**: Logical dependency between components based on data flow or hosting.

| Property | Description |
|----------|-------------|
| Direction | Dependent → Provider |
| Weight | Computed (see [Weight Calculations](weight-calculations.md)) |
| Subtype | One of: `app_to_app`, `app_to_broker`, `node_to_node`, `node_to_broker` |

**Subtypes**:

| Subtype | Source | Target | Derivation Logic |
|---------|--------|--------|------------------|
| `app_to_app` | Subscriber App | Publisher App | Subscriber depends on publisher via shared topic |
| `app_to_broker` | Application | Broker | App uses topic routed by broker |
| `node_to_node` | Node A | Node B | App on A depends on app on B |
| `node_to_broker` | Node | Broker | Hosted app depends on broker |

### 4.3 Edge Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                    Structural Relationships                       │
├──────────────────┬───────────────┬───────────────┬───────────────┤
│ PUBLISHES_TO     │ SUBSCRIBES_TO │ ROUTES        │ RUNS_ON       │
│ App → Topic      │ App → Topic   │ Broker → Topic│ App/Broker→Node│
├──────────────────┴───────────────┴───────────────┴───────────────┤
│ CONNECTS_TO: Node ↔ Node                                         │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    Derived Dependencies                           │
├─────────────────┬─────────────────┬──────────────────────────────┤
│ app_to_app      │ app_to_broker   │ Computed from pub-sub        │
│ Sub → Pub       │ App → Broker    │ patterns and routing         │
├─────────────────┼─────────────────┼──────────────────────────────┤
│ node_to_node    │ node_to_broker  │ Aggregated from hosted       │
│ Node → Node     │ Node → Broker   │ application dependencies     │
└─────────────────┴─────────────────┴──────────────────────────────┘
```

---

## 5. Multi-Layer Architecture

The graph supports analysis at multiple architectural layers, enabling focused assessment of specific concerns.

### 5.1 Layer Definitions

| Layer | Components | Dependencies | Focus |
|-------|------------|--------------|-------|
| **Application** | Application | app_to_app | Service-level reliability |
| **Infrastructure** | Node | node_to_node | Network topology resilience |
| **App-Broker** | Application, Broker | app_to_broker | Middleware dependency |
| **Node-Broker** | Node, Broker | node_to_broker | Infrastructure-middleware coupling |
| **Complete** | All | All | System-wide analysis |

### 5.2 Layer Extraction

For a given layer $L$, the subgraph is extracted as:

$$G_L = (V_L, E_L)$$

Where:
- $V_L = \{v \in V : \tau_V(v) \in \text{ComponentTypes}(L)\}$
- $E_L = \{e \in E : \delta(e) \in \text{DependencyTypes}(L)\}$

### 5.3 Layer Relationships

```
                    ┌─────────────────────┐
                    │   Complete System   │
                    │   (All Components)  │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Application     │ │  Infrastructure  │ │  Broker          │
│  Layer           │ │  Layer           │ │  Layers          │
│                  │ │                  │ │                  │
│  • Applications  │ │  • Nodes         │ │  • App-Broker    │
│  • app_to_app    │ │  • node_to_node  │ │  • Node-Broker   │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

### 5.4 Cross-Layer Dependencies

Layers are not fully independent—cross-layer dependencies capture coupling:

- **Application ↔ Infrastructure**: Apps run on Nodes (`RUNS_ON`)
- **Application ↔ Broker**: Apps depend on Brokers (`app_to_broker`)
- **Infrastructure ↔ Broker**: Nodes depend on Brokers (`node_to_broker`)

---

## 6. Dependency Derivation

### 6.1 Derivation Process

Dependencies are automatically derived during graph import:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Raw Topology   │     │  Derive Deps    │     │  Weighted       │
│                 │────▶│                 │────▶│  Dependency     │
│  - Components   │     │  - app_to_app   │     │  Graph          │
│  - Pub/Sub/Run  │     │  - node_to_node │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 6.2 App-to-App Derivation

**Logic**: If application $a_{sub}$ subscribes to topic $t$ and application $a_{pub}$ publishes to topic $t$, then $a_{sub}$ depends on $a_{pub}$.

$$\forall a_{pub}, a_{sub} \in A, t \in T:$$
$$(a_{pub}, t) \in E_{pub} \land (a_{sub}, t) \in E_{sub} \implies (a_{sub}, a_{pub}) \in E_{dep}^{app}$$

**Direction Rationale**: The subscriber depends on the publisher because:
1. Subscriber's functionality requires data from publisher
2. Publisher failure causes subscriber data starvation
3. This models failure propagation direction

### 6.3 Node-to-Node Derivation

**Logic**: If any app on node $n_1$ depends on any app on node $n_2$, then $n_1$ depends on $n_2$.

$$\forall n_1, n_2 \in N:$$
$$(\exists a_1, a_2 \in A: (a_1, n_1) \in E_{runs} \land (a_2, n_2) \in E_{runs} \land (a_1, a_2) \in E_{dep}^{app})$$
$$\implies (n_1, n_2) \in E_{dep}^{node}$$

### 6.4 Broker Dependencies

**App-to-Broker**: App depends on broker if it uses a topic routed by that broker.

$$\forall a \in A, b \in B:$$
$$(\exists t \in T: ((a, t) \in E_{pub} \lor (a, t) \in E_{sub}) \land (b, t) \in E_{routes})$$
$$\implies (a, b) \in E_{dep}^{broker}$$

**Node-to-Broker**: Aggregated from hosted app dependencies.

---

## 7. Domain Mapping

### 7.1 ROS 2 (DDS)

| Graph Concept | ROS 2 Equivalent |
|---------------|------------------|
| Application | ROS Node |
| Topic | ROS Topic |
| Broker | DDS Participant / Domain |
| Node | Host machine / Container |
| PUBLISHES_TO | Publisher declaration |
| SUBSCRIBES_TO | Subscription declaration |
| QoS Policy | ROS 2 QoS settings |

**Example Mapping** (from `mont_blanc.json`):

```json
{
  "node_name": "hamburg",
  "subscribers": [
    {"topic_name": "nile", "msg_type": "stamped4_int32"},
    {"topic_name": "tigris", "msg_type": "stamped4_float32"}
  ],
  "publishers": [
    {"topic_name": "parana", "msg_type": "stamped3_float32", "period_ms": 10}
  ]
}
```

Maps to:
- Application: `hamburg` (role: `pubsub`)
- Topics: `nile`, `tigris`, `parana`
- Edges: `hamburg` → SUBSCRIBES_TO → `nile`, `tigris`
- Edges: `hamburg` → PUBLISHES_TO → `parana`

### 7.2 Apache Kafka

| Graph Concept | Kafka Equivalent |
|---------------|------------------|
| Application | Producer / Consumer |
| Topic | Kafka Topic (+ Partition) |
| Broker | Kafka Broker |
| Node | Broker host / K8s pod |

### 7.3 MQTT

| Graph Concept | MQTT Equivalent |
|---------------|------------------|
| Application | MQTT Client |
| Topic | MQTT Topic (hierarchical) |
| Broker | MQTT Broker |
| Node | Broker server |

### 7.4 Generic Microservices

| Graph Concept | Microservices Equivalent |
|---------------|--------------------------|
| Application | Service instance |
| Topic | Event channel / Message queue |
| Broker | Message broker / Event bus |
| Node | Container / Pod / VM |

---

## 8. Example Graph

### 8.1 Simple Pub-Sub System

**Scenario**: Two publishers, one processor, two consumers, single broker.

```
Publishers:          Processor:         Consumers:
┌──────────┐        ┌──────────┐        ┌──────────┐
│ Sensor-A │        │ Fusion   │        │ Display  │
└────┬─────┘        └────┬─────┘        └────┬─────┘
     │                   │                   │
     │ pub               │ sub    pub        │ sub
     ▼                   ▼        │          ▼
┌─────────┐         ┌─────────┐  │     ┌─────────┐
│ /raw/a  │         │ /raw/a  │  │     │ /fused  │
└─────────┘         │ /raw/b  │  │     └─────────┘
                    └─────────┘  │
┌──────────┐             ▲       │     ┌──────────┐
│ Sensor-B │             │       ▼     │ Logger   │
└────┬─────┘             │  ┌─────────┐└────┬─────┘
     │ pub               │  │ /fused  │     │ sub
     ▼                   │  └─────────┘     ▼
┌─────────┐              │            ┌─────────┐
│ /raw/b  │──────────────┘            │ /fused  │
└─────────┘                           └─────────┘
```

### 8.2 Derived Dependencies

From the above topology, the following dependencies are derived:

```
Fusion ──DEPENDS_ON──▶ Sensor-A    (via /raw/a)
Fusion ──DEPENDS_ON──▶ Sensor-B    (via /raw/b)
Display ──DEPENDS_ON──▶ Fusion     (via /fused)
Logger ──DEPENDS_ON──▶ Fusion      (via /fused)
```

### 8.3 Graph Statistics

For a typical medium-scale system:

| Metric | Typical Range |
|--------|---------------|
| Applications | 15-50 |
| Topics | 10-30 |
| Brokers | 1-3 |
| Nodes | 4-8 |
| Structural Edges | 50-150 |
| Derived Dependencies | 30-100 |
| Graph Density | 0.05-0.15 |

---

## 9. Implementation

### 9.1 Data Model Classes

```python
from enum import Enum
from dataclasses import dataclass

class VertexType(str, Enum):
    APPLICATION = "Application"
    BROKER = "Broker"
    TOPIC = "Topic"
    NODE = "Node"

class EdgeType(str, Enum):
    RUNS_ON = "RUNS_ON"
    ROUTES = "ROUTES"
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    CONNECTS_TO = "CONNECTS_TO"
    DEPENDS_ON = "DEPENDS_ON"

class DependencyType(str, Enum):
    APP_TO_APP = "app_to_app"
    NODE_TO_NODE = "node_to_node"
    APP_TO_BROKER = "app_to_broker"
    NODE_TO_BROKER = "node_to_broker"

@dataclass
class QoSPolicy:
    durability: str = "VOLATILE"
    reliability: str = "BEST_EFFORT"
    transport_priority: str = "MEDIUM"
```

### 9.2 Neo4j Schema

**Node Labels**:
```cypher
CREATE CONSTRAINT FOR (a:Application) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT FOR (b:Broker) REQUIRE b.id IS UNIQUE;
CREATE CONSTRAINT FOR (t:Topic) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT FOR (n:Node) REQUIRE n.id IS UNIQUE;
```

**Relationship Types**:
```cypher
// Structural
(a:Application)-[:PUBLISHES_TO]->(t:Topic)
(a:Application)-[:SUBSCRIBES_TO]->(t:Topic)
(b:Broker)-[:ROUTES]->(t:Topic)
(a:Application)-[:RUNS_ON]->(n:Node)
(b:Broker)-[:RUNS_ON]->(n:Node)
(n1:Node)-[:CONNECTS_TO]->(n2:Node)

// Derived
(a1:Application)-[:DEPENDS_ON {dependency_type: 'app_to_app'}]->(a2:Application)
(a:Application)-[:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(b:Broker)
(n1:Node)-[:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2:Node)
(n:Node)-[:DEPENDS_ON {dependency_type: 'node_to_broker'}]->(b:Broker)
```

### 9.3 JSON Input Format

```json
{
  "nodes": [
    {"id": "N0", "name": "Node-0"}
  ],
  "brokers": [
    {"id": "B0", "name": "Broker-0"}
  ],
  "topics": [
    {
      "id": "T0",
      "name": "Topic-0",
      "size": 1024,
      "qos": {
        "durability": "PERSISTENT",
        "reliability": "RELIABLE",
        "transport_priority": "HIGH"
      }
    }
  ],
  "applications": [
    {"id": "A0", "name": "App-0", "role": "pub"}
  ],
  "relationships": {
    "runs_on": [{"from": "A0", "to": "N0"}],
    "routes": [{"from": "B0", "to": "T0"}],
    "publishes_to": [{"from": "A0", "to": "T0"}],
    "subscribes_to": [],
    "connects_to": []
  }
}
```

### 9.4 Key Files

| File | Purpose |
|------|---------|
| `src/core/graph_model.py` | Data class definitions |
| `src/core/graph_importer.py` | Neo4j import and dependency derivation |
| `src/core/graph_exporter.py` | Graph data retrieval |
| `src/core/graph_generator.py` | Synthetic graph generation |

---

## Navigation

- **Next**: [Weight Calculations](weight-calculations.md)
- **See Also**: [Quality Formulations](quality-formulations.md)
