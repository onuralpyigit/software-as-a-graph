# Step 1: Graph Model Construction

**Transform distributed pub-sub system topology into a weighted directed graph**

---

## Overview

Graph Model Construction converts your system architecture into a formal graph representation stored in Neo4j.

```
┌─────────────────────┐          ┌─────────────────────┐
│  System Topology    │          │  Weighted Graph     │
│                     │    →     │                     │
│  - Applications     │          │  - Vertices (N,B,T,A)│
│  - Topics + QoS     │          │  - Structural edges │
│  - Brokers          │          │  - DEPENDS_ON edges │
│  - Infrastructure   │          │  - Computed weights │
└─────────────────────┘          └─────────────────────┘
```

---

## Formal Definition

The system is modeled as a **directed weighted graph**:

```
G = (V, E, τ, w)
```

| Symbol | Description |
|--------|-------------|
| V | Vertices (system components) |
| E | Directed edges (relationships) |
| τ | Type function for vertices/edges |
| w | Weight function |

---

## Component Types (Vertices)

| Type | Description | Example |
|------|-------------|---------|
| **Node** | Physical/virtual host | Server, VM, Container |
| **Broker** | Message routing middleware | DDS Participant, Kafka Broker |
| **Topic** | Message channel with QoS | `/sensors/lidar`, `orders.created` |
| **Application** | Service that pub/sub to topics | ROS Node, Microservice |
| **Library** | Shared code dependency that pub/sub to topics | Navigation Library, Data Processing Library|

### Hierarchy

```
Node (Infrastructure)
  └── Broker (Middleware)
        └── Topic (Data Channel)
              └── Application (Software)
                    └── Library (Shared Code)
```

---

## Relationship Types (Edges)

### Structural Edges (From Input)

| Edge | From → To | Meaning |
|------|-----------|---------|
| `RUNS_ON` | App/Broker → Node | Deployed on host |
| `ROUTES` | Broker → Topic | Manages topic routing |
| `PUBLISHES_TO` | App → Topic | Sends messages |
| `SUBSCRIBES_TO` | App → Topic | Receives messages |
| `CONNECTS_TO` | Node → Node | Network connection |
| `USES` | App/Lib -> Lib | Shared code |

### Derived Edges (Computed)

Logical dependencies are automatically derived during import:

| Edge | Rule |
|------|------|
| `app_to_app` | App-A subscribes to Topic-X, App-B publishes to Topic-X → A depends on B |
| `app_to_broker` | App uses Topic managed by Broker → App depends on Broker |
| `node_to_node` | Apps on Node-1 depend on Apps on Node-2 → Node-1 depends on Node-2 |
| `node_to_broker` | Apps on Node use Broker → Node depends on Broker |

**Example derivation:**

```
App-A ──SUBSCRIBES_TO──▶ Topic-X ◀──PUBLISHES_TO── App-B

                    ↓ derives

App-A ────────────DEPENDS_ON────────────▶ App-B
```

---

## Weight Calculation

Weights represent component importance and propagate from Topic QoS upward.

### Step 1: Topic Weight

```
W_topic = S_reliability + S_durability + S_priority + S_size
```

| QoS Setting | Condition | Score |
|-------------|-----------|-------|
| Reliability | RELIABLE | +0.30 |
| Durability | PERSISTENT | +0.40 |
| Priority | URGENT | +0.30 |
| Message Size | Logarithmic | 0.0 - 1.0 |

**Size Score Formula:**
```
S_size = min(log₂(1 + size_bytes / 1024) / 10, 1.0)
```

Example values:
- 1 KB → 0.10
- 8 KB → 0.32
- 64 KB → 0.60
- 1 MB → 1.00 (capped)

### Step 2: Weight Propagation

```
Topic QoS → Topic Weight → Edge Weight → Component Weight → Dependency Weight
```

- **Edge weight** = Topic weight
- **App weight** = Sum of connected topic weights + Sum of used library weights
- **Lib weight** = Sum of shared topic weights
- **Broker weight** = Sum of routed topic weights
- **Node weight** = Sum of hosted component weights (Apps + Brokers)
- **Dependency weight** = Shared topics count × average topic weight


---

## Analysis Layers

The graph supports multi-layer analysis by filtering dependencies:

| Layer | Components Analyzed | Dependencies Used | Focus |
|-------|---------------------|-------------------|-------|
| **app** | Applications | app_to_app | Software dependencies & reliability |
| **infra** | Nodes | node_to_node | Hardware dependencies & availability |
| **mw** | Brokers | app_to_broker, node_to_broker | Middleware coupling & maintainability |
| **system** | All Components | All | Complete system criticality |

---

## Commands

```bash
# Generate synthetic graph data
python bin/generate_graph.py --config config/medium_scale.yaml --output data/system.json

# Import to Neo4j (computes weights, derives dependencies)
python bin/import_graph.py --input data/system.json --clear
```

### Scale Options

| Scale | Apps | Topics | Nodes | Use Case |
|-------|------|--------|-------|----------|
| tiny | 5-8 | 3-5 | 2-3 | Unit tests |
| small | 10-15 | 8-12 | 3-4 | Quick validation |
| medium | 20-35 | 15-25 | 5-8 | Development |
| large | 50-80 | 30-50 | 8-12 | Integration tests |
| xlarge | 100-200 | 60-100 | 15-25 | Performance tests |

---

## Domain Mapping

| Graph Concept | ROS 2 | Kafka | MQTT |
|---------------|-------|-------|------|
| Application | ROS Node | Producer/Consumer | MQTT Client |
| Topic | ROS Topic | Kafka Topic | MQTT Topic |
| Broker | DDS Participant | Kafka Broker | MQTT Broker |
| Node | Host/Container | Broker Host | Broker Server |

---

## Verify Import

Query Neo4j to verify:

```cypher
-- Count entities
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS count;

-- View dependencies
MATCH (a)-[d:DEPENDS_ON]->(b)
RETURN a.name, d.dependency_type, b.name, d.weight
ORDER BY d.weight DESC LIMIT 10;
```

---

## Next Step

→ [Step 2: Structural Analysis](structural-analysis.md)
