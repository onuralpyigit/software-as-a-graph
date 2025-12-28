# Graph Model

This document describes the multi-layer graph model used to represent distributed publish-subscribe systems.

---

## Table of Contents

1. [Overview](#overview)
2. [Multi-Layer Architecture](#multi-layer-architecture)
3. [Vertex Types](#vertex-types)
4. [Edge Types](#edge-types)
5. [Derived Dependencies](#derived-dependencies)
6. [QoS-Aware Weight Calculation](#qos-aware-weight-calculation)
7. [Graph Generation](#graph-generation)
8. [Domain Scenarios](#domain-scenarios)
9. [Implementation Details](#implementation-details)

---

## Overview

We model publish-subscribe systems as **directed multi-layer graphs** where:

- **Vertices** represent system components (applications, topics, brokers, nodes)
- **Edges** represent relationships (publishes, subscribes, routes, runs_on)
- **Layers** capture architectural hierarchy (application → topic → broker → infrastructure)

This representation enables applying graph algorithms to analyze structural properties and identify critical components.

### Formal Definition

A pub-sub system graph G = (V, E, L) where:
- V = set of vertices (components)
- E = set of directed edges (relationships)
- L = {Application, Topic, Broker, Infrastructure} (layer assignment)

Each vertex v ∈ V has:
- type(v) ∈ {Application, Topic, Broker, Node}
- layer(v) ∈ L
- properties(v) = domain-specific attributes

Each edge e = (u, v) ∈ E has:
- type(e) ∈ {PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO, DEPENDS_ON}
- weight(e) = dependency strength (for DEPENDS_ON edges)

---

## Multi-Layer Architecture

The graph spans four architectural layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LAYER 4: INFRASTRUCTURE                                               │
│  ┌──────┐         ┌──────┐         ┌──────┐                           │
│  │  N1  │─────────│  N2  │─────────│  N3  │  Physical/virtual hosts   │
│  └──┬───┘         └──┬───┘         └──┬───┘                           │
│     │                │                │                                │
├─────│────────────────│────────────────│────────────────────────────────┤
│     │                │                │                                │
│  LAYER 3: BROKER                                                       │
│  ┌──▼───┐         ┌──▼───┐                                            │
│  │  B1  │─────────│  B2  │         Message routing infrastructure     │
│  └──┬───┘         └──┬───┘                                            │
│    /│\              /│\                                                │
├───/─│─\────────────/─│─\───────────────────────────────────────────────┤
│  /  │  \          /  │  \                                              │
│  LAYER 2: TOPIC                                                        │
│ ┌▼┐ ┌▼┐ ┌▼┐    ┌▼┐ ┌▼┐ ┌▼┐                                           │
│ │T1│ │T2│ │T3│  │T4│ │T5│ │T6│     Message channels with QoS          │
│ └┬┘ └┬┘ └┬┘    └┬┘ └┬┘ └┬┘                                           │
│  │   │   │      │   │   │                                              │
├──│───│───│──────│───│───│──────────────────────────────────────────────┤
│  │   │   │      │   │   │                                              │
│  LAYER 1: APPLICATION                                                  │
│ ┌▼┐ ┌▼┐ ┌▼┐    ┌▼┐ ┌▼┐ ┌▼┐                                           │
│ │A1│ │A2│ │A3│  │A4│ │A5│ │A6│     Publishers and subscribers         │
│ └──┘ └──┘ └──┘  └──┘ └──┘ └──┘                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer Descriptions

| Layer | Components | Role | Examples |
|-------|------------|------|----------|
| **Application** | Publishers, Subscribers | Produce/consume messages | Sensor driver, control node, logger |
| **Topic** | Message Channels | Named communication endpoints | /sensor/temperature, /cmd/velocity |
| **Broker** | Message Routers | Route messages between apps | MQTT broker, DDS participant |
| **Infrastructure** | Nodes, Hosts | Execution environment | Edge device, cloud VM, container |

### Cross-Layer Relationships

- **Application → Topic**: PUBLISHES_TO, SUBSCRIBES_TO
- **Broker → Topic**: ROUTES
- **Application → Node**: RUNS_ON
- **Broker → Node**: RUNS_ON
- **Node → Node**: CONNECTS_TO

---

## Vertex Types

### Application

Software components that produce and/or consume messages.

```python
@dataclass
class Application:
    id: str              # Unique identifier (e.g., "A1", "sensor_driver")
    name: str            # Human-readable name
    role: str            # "publisher", "subscriber", or "pubsub"
```

**Roles:**
- `publisher`: Only publishes messages
- `subscriber`: Only subscribes to topics
- `pubsub`: Both publishes and subscribes

### Topic

Message channels with QoS (Quality of Service) settings.

```python
@dataclass
class Topic:
    id: str              # Unique identifier (e.g., "T1", "/sensor/temp")
    name: str            # Topic name
    qos: QoSPolicy       # Quality of service settings
    size: int            # Typical message size in bytes
```

**QoS Settings:**

| Property | Values | Description |
|----------|--------|-------------|
| Durability | VOLATILE, TRANSIENT_LOCAL, TRANSIENT, PERSISTENT | Message persistence |
| Reliability | BEST_EFFORT, RELIABLE | Delivery guarantee |
| Priority | LOW, MEDIUM, HIGH, URGENT | Transport priority |

### Broker

Message routing infrastructure components.

```python
@dataclass
class Broker:
    id: str              # Unique identifier (e.g., "B1", "mqtt_broker")
    name: str            # Broker name
```

### Node

Physical or virtual infrastructure hosts.

```python
@dataclass
class Node:
    id: str              # Unique identifier (e.g., "N1", "edge_device")
    name: str            # Node name
```

---

## Edge Types

### PUBLISHES_TO

Application publishes messages to a topic.

```
[Application] ──PUBLISHES_TO──▶ [Topic]
```

Properties:
- Directed: Application → Topic
- Implies message production

### SUBSCRIBES_TO

Application subscribes to receive messages from a topic.

```
[Application] ──SUBSCRIBES_TO──▶ [Topic]
```

Properties:
- Directed: Application → Topic
- Implies message consumption

### ROUTES

Broker handles message routing for a topic.

```
[Broker] ──ROUTES──▶ [Topic]
```

Properties:
- Directed: Broker → Topic
- One broker typically routes many topics

### RUNS_ON

Component executes on an infrastructure node.

```
[Application/Broker] ──RUNS_ON──▶ [Node]
```

Properties:
- Directed: Component → Node
- Multiple components can run on one node

### CONNECTS_TO

Network connectivity between infrastructure nodes.

```
[Node] ──CONNECTS_TO──▶ [Node]
```

Properties:
- Can be bidirectional (modeled as two directed edges)
- Represents network reachability

---

## Derived Dependencies

From basic relationships, we derive **DEPENDS_ON** edges that capture functional dependencies.

### app_to_app

Subscriber depends on publisher through shared topic.

```
[Publisher] ──PUBLISHES_TO──▶ [Topic] ◀──SUBSCRIBES_TO── [Subscriber]
                    │
                    ▼ derive
[Publisher] ◀──DEPENDS_ON── [Subscriber]
```

**Logic**: If A₁ publishes to topic T and A₂ subscribes to T, then A₂ depends on A₁.

### app_to_broker

Application depends on broker for message routing.

```
[Application] ──PUBLISHES/SUBSCRIBES──▶ [Topic] ◀──ROUTES── [Broker]
                                              │
                                              ▼ derive
                    [Application] ──DEPENDS_ON──▶ [Broker]
```

**Logic**: If A uses topic T and B routes T, then A depends on B.

### app_to_node

Application depends on node for execution.

```
[Application] ──RUNS_ON──▶ [Node]
         │
         ▼ derive
[Application] ──DEPENDS_ON──▶ [Node]
```

**Logic**: If A runs on N, then A depends on N (directly from RUNS_ON).

### node_to_node

Transitive infrastructure dependency through network paths.

```
[N1] ──CONNECTS_TO──▶ [N2] ──CONNECTS_TO──▶ [N3]
            │
            ▼ derive
       [N1] ──DEPENDS_ON──▶ [N2]
```

**Logic**: Network reachability creates infrastructure dependencies.

---

## QoS-Aware Weight Calculation

DEPENDS_ON edges include a **weight** property reflecting dependency strength:

```
weight = topic_count + qos_score + size_factor
```

### QoS Score Components

| Property | Value | Weight Contribution |
|----------|-------|---------------------|
| **Durability** | PERSISTENT | +0.40 |
| | TRANSIENT | +0.25 |
| | TRANSIENT_LOCAL | +0.20 |
| | VOLATILE | +0.00 |
| **Reliability** | RELIABLE | +0.30 |
| | BEST_EFFORT | +0.00 |
| **Priority** | URGENT | +0.30 |
| | HIGH | +0.20 |
| | MEDIUM | +0.10 |
| | LOW | +0.00 |

### Size Factor

```
size_factor = min(message_size / 10000, 0.5)
```

Larger messages indicate more critical data dependencies.

### Example Calculation

Dependency through 3 topics with:
- PERSISTENT durability (+0.40 each)
- RELIABLE reliability (+0.30 each)
- Average message size 1KB (+0.10 each)

```
Weight = 3 + (3 × 0.40) + (3 × 0.30) + (3 × 0.10) = 5.40
```

---

## Graph Generation

The toolkit includes a graph generator for creating realistic test systems.

### Scale Presets

| Scale | Applications | Brokers | Topics | Nodes |
|-------|-------------|---------|--------|-------|
| tiny | 5 | 1 | 8 | 2 |
| small | 10 | 2 | 20 | 4 |
| medium | 30 | 4 | 60 | 8 |
| large | 100 | 8 | 200 | 20 |
| xlarge | 300 | 16 | 600 | 50 |

### Usage

```python
from src.core import generate_graph

# Generate with defaults
graph = generate_graph(scale="medium", scenario="iot")

# With options
graph = generate_graph(
    scale="large",
    scenario="financial",
    seed=42,
    antipatterns=["god_topic", "spof"]
)
```

### CLI

```bash
# Generate medium IoT system
python generate_graph.py --scale medium --scenario iot --output system.json

# Preview without saving
python generate_graph.py --scale small --preview

# With anti-patterns
python generate_graph.py --scale medium --antipatterns god_topic spof
```

---

## Domain Scenarios

The generator supports domain-specific topology patterns.

### IoT (Internet of Things)

```python
scenario="iot"
```

- Many sensors publishing telemetry
- Few actuators subscribing to commands
- Hierarchical topic structure
- Mixed QoS (telemetry: best-effort, commands: reliable)

**Typical Topics**: `/sensor/temperature`, `/sensor/humidity`, `/actuator/command`, `/alert/critical`

### Financial Trading

```python
scenario="financial"
```

- Market data publishers (high volume)
- Order management subscribers
- High reliability requirements
- Low latency priority

**Typical Topics**: `/market/quotes`, `/orders/new`, `/orders/status`, `/risk/alerts`

### Healthcare

```python
scenario="healthcare"
```

- Patient monitoring sensors
- Alert notification systems
- Strict reliability (RELIABLE)
- Data persistence (PERSISTENT)

**Typical Topics**: `/patient/vitals`, `/monitor/ecg`, `/alert/emergency`, `/records/update`

### Autonomous Vehicle (ROS 2)

```python
scenario="autonomous_vehicle"
```

- Sensor fusion topics
- Control command topics
- Real-time QoS requirements
- Strict timing constraints

**Typical Topics**: `/lidar/pointcloud`, `/camera/image`, `/cmd_vel`, `/odom`

### Smart City

```python
scenario="smart_city"
```

- Distributed sensor networks
- Traffic management
- Environmental monitoring
- Mixed criticality levels

**Typical Topics**: `/traffic/flow`, `/environment/air_quality`, `/parking/availability`

---

## Implementation Details

### GraphModel Class

```python
from src.core import GraphModel

# Load from dictionary
model = GraphModel.from_dict(graph_data)

# Access components
for app in model.applications:
    print(f"{app.id}: {app.role}")

for topic in model.topics:
    print(f"{topic.id}: {topic.qos}")

# Query relationships
publishers = model.get_publishers(topic_id="T1")
subscribers = model.get_subscribers(topic_id="T1")
topics_routed = model.get_routed_topics(broker_id="B1")

# Summary
print(model.summary())
```

### SimulationGraph Class

```python
from src.simulation import SimulationGraph

# Load from dictionary
graph = SimulationGraph.from_dict(graph_data)

# Component access
component = graph.get_component("A1")
components = graph.get_components_by_type(ComponentType.APPLICATION)

# Path queries
paths = graph.get_message_paths()
reachable = graph.get_reachable_from("A1")

# Graph operations
subgraph = graph.copy()
subgraph.remove_component("B1")
```

### JSON Format

```json
{
  "nodes": [
    {"id": "N1", "name": "edge_device_1"},
    {"id": "N2", "name": "cloud_server"}
  ],
  "brokers": [
    {"id": "B1", "name": "mqtt_broker"}
  ],
  "topics": [
    {
      "id": "T1",
      "name": "/sensor/temperature",
      "size": 256,
      "qos": {
        "durability": "TRANSIENT_LOCAL",
        "reliability": "RELIABLE",
        "priority": "HIGH"
      }
    }
  ],
  "applications": [
    {"id": "A1", "name": "temp_sensor", "role": "publisher"},
    {"id": "A2", "name": "controller", "role": "subscriber"}
  ],
  "relationships": {
    "publishes_to": [
      {"from": "A1", "to": "T1"}
    ],
    "subscribes_to": [
      {"from": "A2", "to": "T1"}
    ],
    "routes": [
      {"from": "B1", "to": "T1"}
    ],
    "runs_on": [
      {"from": "A1", "to": "N1"},
      {"from": "A2", "to": "N2"},
      {"from": "B1", "to": "N2"}
    ],
    "connects_to": [
      {"from": "N1", "to": "N2"}
    ]
  }
}
```

---

## Navigation

- **Previous:** [← Methodology Overview](methodology.md)
- **Next:** [Structural Analysis →](analysis.md)
