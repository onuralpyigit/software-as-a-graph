# Step 1: Graph Model Construction

**Building Distributed Publish-Subscribe System Graphs for Critical Component Analysis**

---

## Table of Contents

1. [Overview](#overview)
2. [Graph Generation with generate_graph.py](#graph-generation-with-generate_graphpy)
3. [Graph Import with import_graph.py](#graph-import-with-import_graphpy)
4. [JSON Schema Reference](#json-schema-reference)
5. [Dependency Derivation Algorithm](#dependency-derivation-algorithm)
6. [Practical Examples](#practical-examples)
7. [Validation and Verification](#validation-and-verification)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Overview

Graph Model Construction is the foundation of the Software-as-a-Graph methodology. This step transforms distributed publish-subscribe system architectures into formal graph representations that enable topological analysis and critical component identification.

### Two-Phase Construction Process

```
Phase 1: GENERATION                    Phase 2: IMPORT
┌──────────────────────┐              ┌──────────────────────┐
│  generate_graph.py   │              │  import_graph.py     │
│                      │              │                      │
│  • Synthetic graphs  │              │  • Neo4j storage     │
│  • Custom topologies │  JSON File   │  • Weight calc       │
│  • Domain presets    │ ──────────▶  │  • Dependency derive │
│  • Validation        │              │  • Constraint setup  │
└──────────────────────┘              └──────────────────────┘
         │                                       │
         │                                       │
         ▼                                       ▼
  system_topology.json                   Neo4j Graph Database
  ┌────────────────┐                    ┌────────────────────┐
  │ - nodes        │                    │ (a:Application)    │
  │ - brokers      │                    │ (b:Broker)         │
  │ - topics       │                    │ (t:Topic)          │
  │ - applications │                    │ (n:Node)           │
  │ - relationships│                    │                    │
  └────────────────┘                    │ [:PUBLISHES_TO]    │
                                        │ [:SUBSCRIBES_TO]   │
                                        │ [:ROUTES]          │
                                        │ [:RUNS_ON]         │
                                        │ [:DEPENDS_ON] ✨   │
                                        └────────────────────┘
```

### What Happens in Step 1

1. **Generate/Prepare Topology**: Create JSON describing system architecture
2. **Import to Neo4j**: Parse JSON and create graph nodes/relationships
3. **Calculate Weights**: Compute topic weights from QoS policies
4. **Derive Dependencies**: Generate logical `DEPENDS_ON` relationships
5. **Validate Structure**: Ensure graph integrity and completeness

### Key Features

| Feature | Description |
|---------|-------------|
| **Synthetic Generation** | Create test systems with configurable scale and patterns |
| **Domain Presets** | Pre-configured templates (ROS 2, IoT, Microservices, Kafka) |
| **Custom Import** | Load hand-crafted or extracted system topologies |
| **Automatic Derivation** | Infer logical dependencies from structural relationships |
| **Neo4j Integration** | Native graph database with Cypher query support |
| **Validation Checks** | Structural integrity verification |

---

## Graph Generation with generate_graph.py

### Purpose

The `generate_graph.py` script creates synthetic distributed pub-sub system topologies for testing, demonstration, and research purposes. It generates realistic system architectures with configurable scale and communication patterns.

### Basic Usage

```bash
python scripts/generate_graph.py [OPTIONS]
```

### Command-Line Options

#### Required Options (One of)

```bash
# Generate from preset
--preset {ros2_autonomous, iot_smart_city, microservices_trading, kafka_pipeline, generic}

# Generate with custom parameters
--num-apps INT          # Number of applications
--num-topics INT        # Number of topics
--num-brokers INT       # Number of brokers (default: 1)
--num-nodes INT         # Number of infrastructure nodes
```

#### Optional Configuration

```bash
--output PATH           # Output JSON file path (default: generated_graph.json)
--size {small, medium, large}  # Preset size configuration
--seed INT              # Random seed for reproducibility
--validate              # Run validation checks before output
--pretty                # Pretty-print JSON with indentation
--verbose               # Show detailed generation logs
```

### Size Presets

| Size | Applications | Topics | Brokers | Nodes | Typical Use Case |
|------|-------------|--------|---------|-------|------------------|
| **small** | 10-15 | 6-10 | 1-2 | 3-4 | Unit testing, demos |
| **medium** | 30-40 | 15-25 | 2-3 | 6-8 | Integration testing, small systems |
| **large** | 60-80 | 30-50 | 3-5 | 10-15 | Stress testing, enterprise systems |

### Domain Presets

#### 1. ROS 2 Autonomous Vehicle (`ros2_autonomous`)

**Characteristics**:
- Sensor publishers (cameras, LiDAR, IMU, GPS)
- Fusion and processing nodes (pubsub role)
- Control and actuation subscribers
- High-priority topics with RELIABLE QoS
- Realistic ROS 2 naming conventions

**Example**:
```bash
python scripts/generate_graph.py \
    --preset ros2_autonomous \
    --size medium \
    --output examples/autonomous_vehicle.json
```

**Generated Structure**:
```
Sensors (Publishers):
├── /camera/front       (Image, 1MB, RELIABLE/VOLATILE)
├── /lidar/points       (PointCloud, 2MB, RELIABLE/VOLATILE)
├── /imu/data           (IMU, 256B, RELIABLE/TRANSIENT_LOCAL)
└── /gps/fix            (GPS, 512B, RELIABLE/TRANSIENT)

Processing (PubSub):
├── /fusion/obstacles   (Object List)
├── /planning/path      (Trajectory)
└── /localization/pose  (PoseStamped)

Actuators (Subscribers):
├── /control/steering
└── /control/throttle
```

#### 2. IoT Smart City (`iot_smart_city`)

**Characteristics**:
- Many sensor devices (temperature, traffic, air quality)
- Gateway aggregation nodes
- Cloud processing services
- Mixed QoS: critical alarms (RELIABLE) vs metrics (BEST_EFFORT)
- Hierarchical topic structure

**Example**:
```bash
python scripts/generate_graph.py \
    --preset iot_smart_city \
    --size large \
    --output examples/smart_city.json
```

**Generated Structure**:
```
Edge Sensors (Publishers):
├── /city/zone-a/traffic/count
├── /city/zone-a/env/temp
├── /city/zone-a/env/air_quality
├── /city/zone-b/traffic/count
└── ... (50+ sensors)

Gateways (PubSub):
├── /aggregated/zone-a/metrics
└── /aggregated/zone-b/metrics

Cloud Services (Subscribers):
├── Analytics Service
├── Dashboard Service
└── Alert Service
```

#### 3. Microservices Trading (`microservices_trading`)

**Characteristics**:
- High-frequency data streams
- Order processing pipeline
- Market data distribution
- Critical PERSISTENT durability for orders
- Large message sizes for market depth

**Example**:
```bash
python scripts/generate_graph.py \
    --preset microservices_trading \
    --size medium \
    --output examples/trading_platform.json
```

**Generated Structure**:
```
Market Data (Publishers):
├── /market/stocks/prices    (RELIABLE, VOLATILE, URGENT)
├── /market/options/quotes   (RELIABLE, VOLATILE, HIGH)
└── /market/depth            (BEST_EFFORT, VOLATILE, MEDIUM)

Order Processing (PubSub):
├── /orders/new              (RELIABLE, PERSISTENT, URGENT)
├── /orders/fills            (RELIABLE, PERSISTENT, URGENT)
└── /orders/cancels          (RELIABLE, TRANSIENT, HIGH)

Risk & Analytics (Subscribers):
├── Position Tracker
├── Risk Monitor
└── P&L Calculator
```

#### 4. Kafka Pipeline (`kafka_pipeline`)

**Characteristics**:
- Multiple broker cluster
- Partitioned topics
- Consumer groups (modeled as applications)
- Stream processing applications

**Example**:
```bash
python scripts/generate_graph.py \
    --preset kafka_pipeline \
    --size medium \
    --output examples/kafka_system.json
```

#### 5. Generic (`generic`)

**Characteristics**:
- Random but realistic topology
- Balanced pub/sub/pubsub distribution
- Mixed QoS policies
- Suitable for algorithm testing

### Generation Algorithm

The script follows this process:

```python
1. Initialize Parameters
   ├── Parse command-line arguments
   ├── Load preset configuration if specified
   └── Validate parameter combinations

2. Generate Infrastructure Layer
   ├── Create N nodes (servers, containers, pods)
   └── Assign unique IDs and names

3. Generate Broker Layer
   ├── Create B brokers
   ├── Distribute brokers across nodes (RUNS_ON)
   └── Assign broker responsibilities

4. Generate Topic Layer
   ├── Create T topics with QoS policies
   │   ├── Sample durability distribution
   │   ├── Sample reliability distribution
   │   ├── Sample priority distribution
   │   └── Sample message sizes
   ├── Assign topics to brokers (ROUTES)
   └── Calculate topic weights

5. Generate Application Layer
   ├── Create A applications
   ├── Assign roles (pub: 30%, sub: 40%, pubsub: 30%)
   ├── Distribute apps across nodes (RUNS_ON)
   └── Create pub/sub relationships
       ├── Publishers: Connect to 1-3 topics
       └── Subscribers: Connect to 1-4 topics

6. Validate Structure
   ├── Check connectivity
   ├── Verify no isolated components
   ├── Ensure role consistency
   └── Validate relationship cardinality

7. Export JSON
   ├── Serialize graph structure
   ├── Format with metadata
   └── Write to output file
```

### Advanced Examples

#### Example 1: Reproducible Generation

```bash
# Generate with fixed seed for reproducibility
python scripts/generate_graph.py \
    --preset ros2_autonomous \
    --size medium \
    --seed 42 \
    --output test_system_seed42.json \
    --validate \
    --verbose
```

#### Example 2: Custom Topology

```bash
# Create specific topology for testing
python scripts/generate_graph.py \
    --num-apps 25 \
    --num-topics 15 \
    --num-brokers 2 \
    --num-nodes 6 \
    --output custom_topology.json \
    --pretty \
    --validate
```

#### Example 3: Batch Generation

```bash
# Generate multiple systems for statistical analysis
for size in small medium large; do
    for seed in {1..10}; do
        python scripts/generate_graph.py \
            --preset generic \
            --size $size \
            --seed $seed \
            --output "batch/system_${size}_${seed}.json"
    done
done
```

### Output File Structure

The generated JSON follows this schema:

```json
{
  "metadata": {
    "generator": "generate_graph.py",
    "version": "1.0",
    "generated_at": "2025-01-11T10:30:00Z",
    "preset": "ros2_autonomous",
    "size": "medium",
    "seed": 42,
    "statistics": {
      "num_nodes": 6,
      "num_brokers": 2,
      "num_topics": 20,
      "num_applications": 35,
      "num_relationships": 142
    }
  },
  "nodes": [...],
  "brokers": [...],
  "topics": [...],
  "applications": [...],
  "relationships": {
    "runs_on": [...],
    "routes": [...],
    "publishes_to": [...],
    "subscribes_to": [...],
    "connects_to": [...]
  }
}
```

---

## Graph Import with import_graph.py

### Purpose

The `import_graph.py` script performs the critical transformation from JSON topology to Neo4j graph database. It handles:

1. **Schema Creation**: Establishes Neo4j constraints and indexes
2. **Data Import**: Creates nodes and structural relationships
3. **Weight Calculation**: Computes topic and component weights from QoS
4. **Dependency Derivation**: Generates logical `DEPENDS_ON` relationships
5. **Validation**: Verifies graph integrity

### Basic Usage

```bash
python scripts/import_graph.py [OPTIONS]
```

### Command-Line Options

#### Required Options

```bash
--input PATH            # Path to JSON topology file
```

#### Neo4j Connection Options

```bash
--uri URI               # Neo4j Bolt URI (default: bolt://localhost:7687)
--user USERNAME         # Neo4j username (default: neo4j)
--password PASSWORD     # Neo4j password (default: password)
--database DATABASE     # Neo4j database name (default: neo4j)
```

#### Import Behavior Options

```bash
--clear                 # Delete all existing data before import
--skip-dependencies     # Skip DEPENDS_ON derivation (faster, incomplete)
--batch-size INT        # Batch size for bulk operations (default: 1000)
--validate              # Run validation checks after import
--verbose               # Show detailed progress logs
--dry-run               # Parse JSON but don't import to Neo4j
```

### Import Pipeline

The import process follows this sequence:

```
┌─────────────────────────────────────────────────────────────┐
│                    IMPORT PIPELINE                           │
└─────────────────────────────────────────────────────────────┘

1. PRE-IMPORT VALIDATION
   ├── Check JSON schema validity
   ├── Verify all IDs are unique
   ├── Validate relationship references
   └── Check QoS policy values

2. NEO4J SETUP
   ├── Connect to database
   ├── (Optional) Clear existing data if --clear
   ├── Create constraints
   │   ├── UNIQUE (a:Application {id})
   │   ├── UNIQUE (b:Broker {id})
   │   ├── UNIQUE (t:Topic {id})
   │   └── UNIQUE (n:Node {id})
   └── Create indexes for performance

3. IMPORT INFRASTRUCTURE LAYER
   ├── CREATE (n:Node) nodes
   └── Log: "Imported 8 nodes"

4. IMPORT BROKER LAYER
   ├── CREATE (b:Broker) nodes
   ├── CREATE [:RUNS_ON] relationships (Broker → Node)
   └── Log: "Imported 3 brokers"

5. IMPORT TOPIC LAYER
   ├── CREATE (t:Topic) nodes with QoS properties
   ├── CALCULATE topic weights
   │   └── W_topic = S_reliability + S_durability + S_priority + S_size
   └── Log: "Imported 25 topics, computed weights"

6. IMPORT APPLICATION LAYER
   ├── CREATE (a:Application) nodes
   ├── CREATE [:RUNS_ON] relationships (Application → Node)
   ├── CREATE [:PUBLISHES_TO] relationships (Application → Topic)
   ├── CREATE [:SUBSCRIBES_TO] relationships (Application → Topic)
   ├── ASSIGN edge weights from connected topics
   └── Log: "Imported 40 applications"

7. IMPORT ROUTING RELATIONSHIPS
   ├── CREATE [:ROUTES] relationships (Broker → Topic)
   ├── ASSIGN edge weights from routed topics
   └── Log: "Imported routing relationships"

8. CALCULATE COMPONENT WEIGHTS
   ├── Application.weight = Σ(connected topic weights)
   ├── Broker.weight = Σ(routed topic weights)
   ├── Node.weight = Σ(hosted component weights)
   └── Log: "Calculated intrinsic weights"

9. DERIVE DEPENDENCIES ⭐
   ├── App-to-App Dependencies
   │   └── For each topic: link subscribers → publishers
   ├── App-to-Broker Dependencies
   │   └── For each app: link to brokers routing its topics
   ├── Node-to-Node Dependencies
   │   └── Aggregate from app-level dependencies
   ├── Node-to-Broker Dependencies
   │   └── Aggregate from app-broker dependencies
   └── Log: "Derived 127 DEPENDS_ON relationships"

10. POST-IMPORT VALIDATION
    ├── Verify node counts match JSON
    ├── Check for orphaned nodes
    ├── Validate dependency weights
    └── Log: "Validation passed ✓"

11. GENERATE SUMMARY REPORT
    └── Print statistics and completion time
```

### Connection Management

The script uses Neo4j Python Driver with connection pooling:

```python
from neo4j import GraphDatabase

# Connection is established with context manager
with GraphDatabase.driver(uri, auth=(user, password)) as driver:
    with driver.session(database=database) as session:
        # All import operations happen here
        pass
```

**Connection Best Practices**:
- Use environment variables for credentials
- Enable connection pooling for large imports
- Set appropriate timeouts for bulk operations

### Schema Creation

Before import, the script establishes database constraints:

```cypher
-- Node uniqueness constraints
CREATE CONSTRAINT app_id_unique IF NOT EXISTS
FOR (a:Application) REQUIRE a.id IS UNIQUE;

CREATE CONSTRAINT broker_id_unique IF NOT EXISTS
FOR (b:Broker) REQUIRE b.id IS UNIQUE;

CREATE CONSTRAINT topic_id_unique IF NOT EXISTS
FOR (t:Topic) REQUIRE t.id IS UNIQUE;

CREATE CONSTRAINT node_id_unique IF NOT EXISTS
FOR (n:Node) REQUIRE n.id IS UNIQUE;

-- Performance indexes
CREATE INDEX app_name_idx IF NOT EXISTS
FOR (a:Application) ON (a.name);

CREATE INDEX topic_name_idx IF NOT EXISTS
FOR (t:Topic) ON (t.name);
```

### Weight Calculation Details

#### Topic Weight Calculation

During import, topic weights are computed from QoS policies:

```python
def calculate_topic_weight(topic):
    """
    W_topic = S_reliability + S_durability + S_priority + S_size
    """
    # Reliability score
    if topic.qos.reliability == "RELIABLE":
        s_reliability = 0.3
    else:  # BEST_EFFORT or other
        s_reliability = 0.0
    
    # Durability score
    durability_scores = {
        "PERSISTENT": 0.4,
        "TRANSIENT": 0.25,
        "TRANSIENT_LOCAL": 0.2,
        "VOLATILE": 0.0
    }
    s_durability = durability_scores.get(topic.qos.durability, 0.0)
    
    # Priority score
    priority_scores = {
        "URGENT": 0.3,
        "HIGH": 0.2,
        "MEDIUM": 0.1,
        "LOW": 0.0
    }
    s_priority = priority_scores.get(topic.qos.transport_priority, 0.0)
    
    # Size score (logarithmic scaling)
    s_size = min(math.log2(1 + topic.size / 1024) / 10, 1.0)
    
    return s_reliability + s_durability + s_priority + s_size
```

#### Component Weight Calculation

After structural import, component intrinsic weights are computed:

```cypher
-- Application weights (sum of connected topic weights)
MATCH (a:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
WITH a, sum(t.weight) as total_weight
SET a.weight = total_weight;

-- Broker weights (sum of routed topic weights)
MATCH (b:Broker)-[:ROUTES]->(t:Topic)
WITH b, sum(t.weight) as total_weight
SET b.weight = total_weight;

-- Node weights (sum of hosted component weights)
MATCH (n:Node)<-[:RUNS_ON]-(component)
WHERE component:Application OR component:Broker
WITH n, sum(component.weight) as total_weight
SET n.weight = total_weight;
```

### Basic Examples

#### Example 1: Simple Import

```bash
# Import generated graph with defaults
python scripts/import_graph.py \
    --input examples/autonomous_vehicle.json
```

#### Example 2: Clean Import with Validation

```bash
# Clear existing data and validate after import
python scripts/import_graph.py \
    --input examples/smart_city.json \
    --clear \
    --validate \
    --verbose
```

#### Example 3: Custom Neo4j Instance

```bash
# Import to remote Neo4j instance
python scripts/import_graph.py \
    --input production_topology.json \
    --uri bolt://production-neo4j:7687 \
    --user admin \
    --password $NEO4J_PASSWORD \
    --database production_db
```

#### Example 4: Dry Run (Validation Only)

```bash
# Validate JSON without importing
python scripts/import_graph.py \
    --input untrusted_topology.json \
    --dry-run \
    --verbose
```

---

## Dependency Derivation Algorithm

### Overview

The `DEPENDS_ON` relationship derivation is the **core innovation** that transforms structural graphs into dependency graphs suitable for criticality analysis. These relationships capture **logical dependencies** that emerge from pub-sub communication patterns.

### Why Derive Dependencies?

**Structural relationships alone are insufficient** because:

1. Publishers and subscribers are **indirectly connected** via topics
2. Applications depend on brokers for **routing services**
3. Infrastructure nodes have **transitive dependencies** through hosted apps
4. Failure propagation follows **logical dependencies**, not just physical connections

### Derivation Types

```
┌────────────────────────────────────────────────────────────┐
│           DEPENDENCY DERIVATION HIERARCHY                   │
└────────────────────────────────────────────────────────────┘

Level 1: APP-TO-APP DEPENDENCIES
         Subscribers depend on Publishers (via shared topics)
         
Level 2: APP-TO-BROKER DEPENDENCIES
         Applications depend on Brokers (for routing)
         
Level 3: NODE-TO-NODE DEPENDENCIES
         Aggregated from app-level dependencies
         
Level 4: NODE-TO-BROKER DEPENDENCIES
         Aggregated from app-broker dependencies
```

### Algorithm Details

#### Type 1: App-to-App Dependencies

**Rule**: For each topic, all subscribers depend on all publishers of that topic.

**Cypher Query**:
```cypher
// Find all publisher-subscriber pairs through shared topics
MATCH (pub:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(sub:Application)
WHERE pub.id <> sub.id  // Don't create self-dependencies

// Calculate dependency weight
WITH sub, pub, collect(t) as shared_topics
WITH sub, pub, shared_topics,
     size(shared_topics) + reduce(w = 0, t IN shared_topics | w + t.weight) as dep_weight

// Create DEPENDS_ON relationship
MERGE (sub)-[d:DEPENDS_ON]->(pub)
SET d.dependency_type = 'app_to_app',
    d.weight = dep_weight,
    d.shared_topics = [t IN shared_topics | t.id],
    d.topic_count = size(shared_topics);
```

**Weight Formula**:
$$W_{dep(sub \to pub)} = |T_{shared}| + \sum_{t \in T_{shared}} W_t$$

**Example**:
```
Before:
  SensorFusion ──SUBSCRIBES_TO──> /camera/front
  Camera        ──PUBLISHES_TO──> /camera/front
  
  SensorFusion ──SUBSCRIBES_TO──> /lidar/points
  LiDAR         ──PUBLISHES_TO──> /lidar/points

After Derivation:
  SensorFusion ──DEPENDS_ON──> Camera   (weight: 1 + W_camera)
  SensorFusion ──DEPENDS_ON──> LiDAR    (weight: 1 + W_lidar)
```

#### Type 2: App-to-Broker Dependencies

**Rule**: Applications depend on brokers that route their topics.

**Cypher Query**:
```cypher
// Find apps and brokers connected through routed topics
MATCH (app:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(broker:Broker)

// Calculate dependency weight
WITH app, broker, collect(DISTINCT t) as routed_topics
WITH app, broker, routed_topics,
     size(routed_topics) + reduce(w = 0, t IN routed_topics | w + t.weight) as dep_weight

// Create DEPENDS_ON relationship
MERGE (app)-[d:DEPENDS_ON]->(broker)
SET d.dependency_type = 'app_to_broker',
    d.weight = dep_weight,
    d.routed_topics = [t IN routed_topics | t.id],
    d.topic_count = size(routed_topics);
```

**Example**:
```
Before:
  TempSensor    ──PUBLISHES_TO──> /sensors/temp
  MainBroker    ──ROUTES──>       /sensors/temp
  
  TempSensor    ──SUBSCRIBES_TO──> /config/sensors
  MainBroker    ──ROUTES──>        /config/sensors

After Derivation:
  TempSensor ──DEPENDS_ON──> MainBroker (weight: 2 + W_temp + W_config)
```

#### Type 3: Node-to-Node Dependencies

**Rule**: If applications on Node A depend on applications on Node B, then Node A depends on Node B.

**Cypher Query**:
```cypher
// Find node-level dependencies through hosted apps
MATCH (nodeA:Node)<-[:RUNS_ON]-(appA:Application)
MATCH (appA)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(appB:Application)
MATCH (appB)-[:RUNS_ON]->(nodeB:Node)
WHERE nodeA.id <> nodeB.id

// Aggregate dependency weights
WITH nodeA, nodeB, collect(d) as app_deps
WITH nodeA, nodeB, app_deps,
     reduce(w = 0, dep IN app_deps | w + dep.weight) as total_weight

// Create aggregated node-level dependency
MERGE (nodeA)-[d:DEPENDS_ON]->(nodeB)
SET d.dependency_type = 'node_to_node',
    d.weight = total_weight,
    d.app_dependency_count = size(app_deps);
```

**Example**:
```
Before:
  Node-1 ──RUNS_ON── App-A
  Node-2 ──RUNS_ON── App-B
  App-A  ──DEPENDS_ON──> App-B (weight: 5.0)

After Derivation:
  Node-1 ──DEPENDS_ON──> Node-2 (weight: 5.0, from 1 app dependency)
```

#### Type 4: Node-to-Broker Dependencies

**Rule**: If applications on a Node depend on a Broker, the Node depends on that Broker.

**Cypher Query**:
```cypher
// Find node-broker dependencies through hosted apps
MATCH (node:Node)<-[:RUNS_ON]-(app:Application)
MATCH (app)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker:Broker)

// Aggregate by node and broker
WITH node, broker, collect(d) as app_deps
WITH node, broker, app_deps,
     reduce(w = 0, dep IN app_deps | w + dep.weight) as total_weight

// Create aggregated dependency
MERGE (node)-[d:DEPENDS_ON]->(broker)
SET d.dependency_type = 'node_to_broker',
    d.weight = total_weight,
    d.app_dependency_count = size(app_deps);
```

### Derivation Order

The derivation **must** follow this order due to dependencies:

```
1. App-to-App       (base dependencies from pub-sub)
       ↓
2. App-to-Broker    (independent of app-to-app)
       ↓
3. Node-to-Node     (requires app-to-app to exist)
       ↓
4. Node-to-Broker   (requires app-to-broker to exist)
```

### Performance Optimization

For large graphs (>1000 components), derivation uses batching:

```python
def derive_dependencies_batched(session, batch_size=1000):
    """Derive dependencies in batches to avoid memory issues."""
    
    # Type 1: App-to-App (batched by topic)
    topics = session.run("MATCH (t:Topic) RETURN t.id as id").data()
    for batch in chunks(topics, batch_size):
        topic_ids = [t['id'] for t in batch]
        session.run("""
            MATCH (pub:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(sub:Application)
            WHERE t.id IN $topic_ids AND pub.id <> sub.id
            // ... rest of query
        """, topic_ids=topic_ids)
    
    # Type 2-4: Similar batching strategy
    # ...
```

### Validation Checks

After derivation, the script validates:

```cypher
-- Check 1: No self-dependencies
MATCH (n)-[d:DEPENDS_ON]->(n)
RETURN count(d) as self_deps;
// Expected: 0

-- Check 2: All weights are positive
MATCH ()-[d:DEPENDS_ON]->()
WHERE d.weight <= 0
RETURN count(d) as invalid_weights;
// Expected: 0

-- Check 3: Dependency types are valid
MATCH ()-[d:DEPENDS_ON]->()
WHERE NOT d.dependency_type IN ['app_to_app', 'app_to_broker', 'node_to_node', 'node_to_broker']
RETURN count(d) as invalid_types;
// Expected: 0

-- Check 4: Count derived relationships
MATCH ()-[d:DEPENDS_ON]->()
RETURN d.dependency_type as type, count(d) as count
ORDER BY type;
```

---

## JSON Schema Reference

### Complete Schema Definition

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["nodes", "brokers", "topics", "applications", "relationships"],
  "properties": {
    
    "metadata": {
      "type": "object",
      "description": "Optional metadata about graph generation",
      "properties": {
        "generator": {"type": "string"},
        "version": {"type": "string"},
        "generated_at": {"type": "string", "format": "date-time"},
        "preset": {"type": "string"},
        "size": {"type": "string", "enum": ["small", "medium", "large"]},
        "seed": {"type": "integer"},
        "statistics": {
          "type": "object",
          "properties": {
            "num_nodes": {"type": "integer"},
            "num_brokers": {"type": "integer"},
            "num_topics": {"type": "integer"},
            "num_applications": {"type": "integer"},
            "num_relationships": {"type": "integer"}
          }
        }
      }
    },
    
    "nodes": {
      "type": "array",
      "description": "Infrastructure nodes (servers, containers, VMs)",
      "items": {
        "type": "object",
        "required": ["id", "name"],
        "properties": {
          "id": {
            "type": "string",
            "description": "Unique identifier (e.g., 'N0', 'node-1')"
          },
          "name": {
            "type": "string",
            "description": "Human-readable name (e.g., 'Server-1', 'Edge-Device-A')"
          },
          "properties": {
            "type": "object",
            "description": "Optional custom properties",
            "properties": {
              "cpu_cores": {"type": "integer"},
              "memory_gb": {"type": "number"},
              "location": {"type": "string"}
            }
          }
        }
      }
    },
    
    "brokers": {
      "type": "array",
      "description": "Message brokers (middleware routing infrastructure)",
      "items": {
        "type": "object",
        "required": ["id", "name"],
        "properties": {
          "id": {
            "type": "string",
            "description": "Unique identifier (e.g., 'B0', 'broker-main')"
          },
          "name": {
            "type": "string",
            "description": "Human-readable name (e.g., 'MainBroker', 'EdgeBroker-1')"
          }
        }
      }
    },
    
    "topics": {
      "type": "array",
      "description": "Message topics/channels with QoS policies",
      "items": {
        "type": "object",
        "required": ["id", "name", "size", "qos"],
        "properties": {
          "id": {
            "type": "string",
            "description": "Unique identifier (e.g., 'T0', 'topic-sensors-temp')"
          },
          "name": {
            "type": "string",
            "description": "Topic name/path (e.g., '/sensors/temperature', 'orders.new')"
          },
          "size": {
            "type": "integer",
            "minimum": 0,
            "description": "Message payload size in bytes"
          },
          "qos": {
            "type": "object",
            "required": ["durability", "reliability", "transport_priority"],
            "properties": {
              "durability": {
                "type": "string",
                "enum": ["VOLATILE", "TRANSIENT_LOCAL", "TRANSIENT", "PERSISTENT"],
                "description": "Data persistence policy"
              },
              "reliability": {
                "type": "string",
                "enum": ["BEST_EFFORT", "RELIABLE"],
                "description": "Delivery guarantee"
              },
              "transport_priority": {
                "type": "string",
                "enum": ["LOW", "MEDIUM", "HIGH", "URGENT"],
                "description": "Message priority level"
              }
            }
          }
        }
      }
    },
    
    "applications": {
      "type": "array",
      "description": "Applications (services, nodes, clients)",
      "items": {
        "type": "object",
        "required": ["id", "name", "role"],
        "properties": {
          "id": {
            "type": "string",
            "description": "Unique identifier (e.g., 'A0', 'app-sensor-temp')"
          },
          "name": {
            "type": "string",
            "description": "Application name (e.g., 'TemperatureSensor', 'OrderProcessor')"
          },
          "role": {
            "type": "string",
            "enum": ["pub", "sub", "pubsub"],
            "description": "Publisher, Subscriber, or Both"
          }
        }
      }
    },
    
    "relationships": {
      "type": "object",
      "required": ["runs_on", "routes", "publishes_to", "subscribes_to"],
      "properties": {
        
        "runs_on": {
          "type": "array",
          "description": "Applications/Brokers hosted on Nodes",
          "items": {
            "type": "object",
            "required": ["from", "to"],
            "properties": {
              "from": {
                "type": "string",
                "description": "Application or Broker ID"
              },
              "to": {
                "type": "string",
                "description": "Node ID"
              }
            }
          }
        },
        
        "routes": {
          "type": "array",
          "description": "Brokers routing Topics",
          "items": {
            "type": "object",
            "required": ["from", "to"],
            "properties": {
              "from": {
                "type": "string",
                "description": "Broker ID"
              },
              "to": {
                "type": "string",
                "description": "Topic ID"
              }
            }
          }
        },
        
        "publishes_to": {
          "type": "array",
          "description": "Applications publishing to Topics",
          "items": {
            "type": "object",
            "required": ["from", "to"],
            "properties": {
              "from": {
                "type": "string",
                "description": "Application ID (must have role 'pub' or 'pubsub')"
              },
              "to": {
                "type": "string",
                "description": "Topic ID"
              }
            }
          }
        },
        
        "subscribes_to": {
          "type": "array",
          "description": "Applications subscribing to Topics",
          "items": {
            "type": "object",
            "required": ["from", "to"],
            "properties": {
              "from": {
                "type": "string",
                "description": "Application ID (must have role 'sub' or 'pubsub')"
              },
              "to": {
                "type": "string",
                "description": "Topic ID"
              }
            }
          }
        },
        
        "connects_to": {
          "type": "array",
          "description": "Optional: Network connections between Nodes",
          "items": {
            "type": "object",
            "required": ["from", "to"],
            "properties": {
              "from": {
                "type": "string",
                "description": "Node ID"
              },
              "to": {
                "type": "string",
                "description": "Node ID"
              },
              "properties": {
                "type": "object",
                "properties": {
                  "bandwidth_mbps": {"type": "number"},
                  "latency_ms": {"type": "number"}
                }
              }
            }
          }
        }
      }
    }
  }
}
```

### Minimal Valid Example

```json
{
  "nodes": [
    {"id": "N0", "name": "Server-1"}
  ],
  "brokers": [
    {"id": "B0", "name": "Broker-1"}
  ],
  "topics": [
    {
      "id": "T0",
      "name": "/data/stream",
      "size": 1024,
      "qos": {
        "durability": "VOLATILE",
        "reliability": "BEST_EFFORT",
        "transport_priority": "MEDIUM"
      }
    }
  ],
  "applications": [
    {"id": "A0", "name": "Publisher", "role": "pub"},
    {"id": "A1", "name": "Subscriber", "role": "sub"}
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

---

## Practical Examples

### Example 1: ROS 2 Sensor Fusion System

#### Step 1: Generate the Topology

```bash
python scripts/generate_graph.py \
    --preset ros2_autonomous \
    --size small \
    --output examples/sensor_fusion.json \
    --pretty \
    --validate
```

#### Step 2: Inspect the JSON

```bash
cat examples/sensor_fusion.json | jq '.applications[] | select(.role == "pubsub")'
```

Output:
```json
{
  "id": "A5",
  "name": "sensor_fusion_node",
  "role": "pubsub"
}
```

#### Step 3: Import to Neo4j

```bash
python scripts/import_graph.py \
    --input examples/sensor_fusion.json \
    --clear \
    --validate \
    --verbose
```

Expected Output:
```
[INFO] Connecting to bolt://localhost:7687...
[INFO] Clearing existing data...
[INFO] Creating constraints and indexes...
[INFO] Importing 4 nodes...
[INFO] Importing 1 brokers...
[INFO] Importing 8 topics...
[INFO] Calculating topic weights...
[INFO]   Topic '/camera/front': weight=1.432
[INFO]   Topic '/lidar/points': weight=1.687
[INFO]   ...
[INFO] Importing 12 applications...
[INFO] Creating structural relationships...
[INFO]   runs_on: 13 relationships
[INFO]   routes: 8 relationships
[INFO]   publishes_to: 18 relationships
[INFO]   subscribes_to: 24 relationships
[INFO] Calculating component intrinsic weights...
[INFO] Deriving DEPENDS_ON relationships...
[INFO]   App-to-App: 35 dependencies
[INFO]   App-to-Broker: 12 dependencies
[INFO]   Node-to-Node: 6 dependencies
[INFO]   Node-to-Broker: 4 dependencies
[INFO] Running validation checks...
[INFO] ✓ No self-dependencies found
[INFO] ✓ All weights positive
[INFO] ✓ Node count matches (4 == 4)
[INFO] ✓ Application count matches (12 == 12)
[SUCCESS] Import completed in 3.42 seconds
```

#### Step 4: Query the Results

```cypher
// Find sensor fusion node dependencies
MATCH (fusion:Application {name: 'sensor_fusion_node'})
MATCH (fusion)-[d:DEPENDS_ON]->(provider)
RETURN provider.name, d.dependency_type, d.weight
ORDER BY d.weight DESC;
```

Result:
```
╒════════════════════╤═══════════════════╤══════════╕
│ provider.name      │ d.dependency_type │ d.weight │
╞════════════════════╪═══════════════════╪══════════╡
│ "lidar_node"       │ "app_to_app"      │ 2.687    │
│ "camera_node"      │ "app_to_app"      │ 2.432    │
│ "imu_node"         │ "app_to_app"      │ 1.320    │
│ "main_broker"      │ "app_to_broker"   │ 5.439    │
╘════════════════════╧═══════════════════╧══════════╛
```

### Example 2: IoT Smart City (Large Scale)

#### Generate Large System

```bash
python scripts/generate_graph.py \
    --preset iot_smart_city \
    --size large \
    --seed 2025 \
    --output examples/smart_city_large.json
```

#### Import with Performance Optimization

```bash
python scripts/import_graph.py \
    --input examples/smart_city_large.json \
    --clear \
    --batch-size 2000 \
    --verbose
```

This system will have:
- 60-80 applications (sensors, gateways, services)
- 30-50 topics (hierarchical topic structure)
- 3-5 brokers (distributed cluster)
- 10-15 nodes (edge + cloud infrastructure)

Expected import time: 8-12 seconds

### Example 3: Custom Healthcare System

#### Step 1: Create Custom JSON

```json
{
  "nodes": [
    {"id": "hospital_server", "name": "Hospital-Datacenter"},
    {"id": "edge_icu", "name": "ICU-Edge-Device"},
    {"id": "edge_er", "name": "ER-Edge-Device"}
  ],
  "brokers": [
    {"id": "central_broker", "name": "Central-MQTT-Broker"}
  ],
  "topics": [
    {
      "id": "vitals_icu",
      "name": "/hospital/icu/vitals",
      "size": 512,
      "qos": {
        "durability": "PERSISTENT",
        "reliability": "RELIABLE",
        "transport_priority": "URGENT"
      }
    },
    {
      "id": "alerts",
      "name": "/hospital/alerts",
      "size": 256,
      "qos": {
        "durability": "PERSISTENT",
        "reliability": "RELIABLE",
        "transport_priority": "URGENT"
      }
    },
    {
      "id": "logs",
      "name": "/hospital/logs",
      "size": 1024,
      "qos": {
        "durability": "TRANSIENT",
        "reliability": "BEST_EFFORT",
        "transport_priority": "LOW"
      }
    }
  ],
  "applications": [
    {"id": "monitor_icu", "name": "ICU-Monitor", "role": "pub"},
    {"id": "alert_system", "name": "Alert-System", "role": "pubsub"},
    {"id": "dashboard", "name": "Dashboard", "role": "sub"},
    {"id": "archive", "name": "Archive-Service", "role": "sub"}
  ],
  "relationships": {
    "runs_on": [
      {"from": "central_broker", "to": "hospital_server"},
      {"from": "monitor_icu", "to": "edge_icu"},
      {"from": "alert_system", "to": "hospital_server"},
      {"from": "dashboard", "to": "hospital_server"},
      {"from": "archive", "to": "hospital_server"}
    ],
    "routes": [
      {"from": "central_broker", "to": "vitals_icu"},
      {"from": "central_broker", "to": "alerts"},
      {"from": "central_broker", "to": "logs"}
    ],
    "publishes_to": [
      {"from": "monitor_icu", "to": "vitals_icu"},
      {"from": "alert_system", "to": "alerts"},
      {"from": "alert_system", "to": "logs"}
    ],
    "subscribes_to": [
      {"from": "alert_system", "to": "vitals_icu"},
      {"from": "dashboard", "to": "vitals_icu"},
      {"from": "dashboard", "to": "alerts"},
      {"from": "archive", "to": "logs"}
    ],
    "connects_to": []
  }
}
```

#### Step 2: Import and Analyze

```bash
# Import
python scripts/import_graph.py \
    --input healthcare_system.json \
    --clear \
    --validate

# Query critical dependencies
cypher-shell -u neo4j -p password << EOF
MATCH (alert:Application {name: 'Alert-System'})
MATCH (alert)-[d:DEPENDS_ON]->(dep)
RETURN dep.name, d.dependency_type, d.weight, d.shared_topics
ORDER BY d.weight DESC;
EOF
```

### Example 4: Batch Processing Multiple Systems

```bash
#!/bin/bash
# batch_import.sh - Import multiple system configurations

systems=(
    "ros2_autonomous:medium"
    "iot_smart_city:large"
    "microservices_trading:medium"
    "kafka_pipeline:small"
)

for system_config in "${systems[@]}"; do
    IFS=':' read -r preset size <<< "$system_config"
    
    echo "Processing $preset ($size)..."
    
    # Generate
    python scripts/generate_graph.py \
        --preset "$preset" \
        --size "$size" \
        --output "batch/${preset}_${size}.json"
    
    # Import to separate databases
    python scripts/import_graph.py \
        --input "batch/${preset}_${size}.json" \
        --database "${preset}_db" \
        --clear
done

echo "Batch import complete!"
```

---

## Validation and Verification

### Pre-Import Validation

The script performs these checks before importing:

#### 1. JSON Schema Validation

```python
def validate_json_schema(data):
    """Validate JSON structure and required fields."""
    errors = []
    
    # Check required top-level keys
    required_keys = ['nodes', 'brokers', 'topics', 'applications', 'relationships']
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    
    # Check ID uniqueness
    all_ids = set()
    for component_type in ['nodes', 'brokers', 'topics', 'applications']:
        for item in data.get(component_type, []):
            if 'id' not in item:
                errors.append(f"Missing 'id' in {component_type}")
            elif item['id'] in all_ids:
                errors.append(f"Duplicate ID: {item['id']}")
            else:
                all_ids.add(item['id'])
    
    return errors
```

#### 2. Relationship Reference Validation

```python
def validate_relationships(data):
    """Ensure all relationship references point to existing components."""
    errors = []
    
    # Build ID sets
    node_ids = {n['id'] for n in data['nodes']}
    broker_ids = {b['id'] for b in data['brokers']}
    topic_ids = {t['id'] for t in data['topics']}
    app_ids = {a['id'] for a in data['applications']}
    
    # Validate runs_on
    for rel in data['relationships']['runs_on']:
        if rel['from'] not in app_ids | broker_ids:
            errors.append(f"RUNS_ON: Unknown source {rel['from']}")
        if rel['to'] not in node_ids:
            errors.append(f"RUNS_ON: Unknown target {rel['to']}")
    
    # Validate routes
    for rel in data['relationships']['routes']:
        if rel['from'] not in broker_ids:
            errors.append(f"ROUTES: Unknown broker {rel['from']}")
        if rel['to'] not in topic_ids:
            errors.append(f"ROUTES: Unknown topic {rel['to']}")
    
    # Similar checks for publishes_to, subscribes_to
    
    return errors
```

#### 3. Role Consistency Validation

```python
def validate_app_roles(data):
    """Check that app roles match their relationships."""
    errors = []
    
    # Get publishers and subscribers
    publishers = {rel['from'] for rel in data['relationships']['publishes_to']}
    subscribers = {rel['from'] for rel in data['relationships']['subscribes_to']}
    
    # Check role consistency
    for app in data['applications']:
        app_id = app['id']
        role = app['role']
        
        is_publisher = app_id in publishers
        is_subscriber = app_id in subscribers
        
        if role == 'pub' and not is_publisher:
            errors.append(f"App {app_id} has role 'pub' but doesn't publish")
        if role == 'sub' and not is_subscriber:
            errors.append(f"App {app_id} has role 'sub' but doesn't subscribe")
        if role == 'pubsub' and not (is_publisher and is_subscriber):
            errors.append(f"App {app_id} has role 'pubsub' but missing pub or sub")
    
    return errors
```

### Post-Import Validation

After import, the script verifies graph integrity:

#### 1. Component Count Verification

```cypher
// Check imported counts match JSON
MATCH (n:Node) WITH count(n) as node_count
MATCH (b:Broker) WITH node_count, count(b) as broker_count
MATCH (t:Topic) WITH node_count, broker_count, count(t) as topic_count
MATCH (a:Application) WITH node_count, broker_count, topic_count, count(a) as app_count
RETURN node_count, broker_count, topic_count, app_count;
```

#### 2. Orphaned Node Detection

```cypher
// Find components with no relationships
MATCH (n)
WHERE NOT (n)--()
RETURN labels(n)[0] as type, n.id as id, n.name as name;
// Expected: Empty result set
```

#### 3. Weight Validation

```cypher
// Check for invalid weights
MATCH (n)
WHERE n.weight IS NULL OR n.weight < 0
RETURN labels(n)[0] as type, n.id, n.weight;
// Expected: Empty result set

// Check dependency weights
MATCH ()-[d:DEPENDS_ON]->()
WHERE d.weight IS NULL OR d.weight <= 0
RETURN type(d), d.weight, d.dependency_type;
// Expected: Empty result set
```

#### 4. Dependency Statistics

```cypher
// Summary of derived dependencies
MATCH ()-[d:DEPENDS_ON]->()
RETURN d.dependency_type as type,
       count(d) as count,
       min(d.weight) as min_weight,
       max(d.weight) as max_weight,
       avg(d.weight) as avg_weight
ORDER BY type;
```

Example Output:
```
╒═════════════════╤═══════╤════════════╤════════════╤════════════╕
│ type            │ count │ min_weight │ max_weight │ avg_weight │
╞═════════════════╪═══════╪════════════╪════════════╪════════════╡
│ "app_to_app"    │ 42    │ 1.250      │ 8.432      │ 3.856      │
│ "app_to_broker" │ 18    │ 2.100      │ 12.340     │ 6.234      │
│ "node_to_node"  │ 8     │ 3.450      │ 15.670     │ 9.112      │
│ "node_to_broker"│ 4     │ 5.600      │ 18.900     │ 11.234     │
╘═════════════════╧═══════╧════════════╧════════════╧════════════╛
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Connection Refused

**Error**:
```
neo4j.exceptions.ServiceUnavailable: Unable to connect to bolt://localhost:7687
```

**Solution**:
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# If not running, start it
docker start neo4j-graph

# Verify connectivity
cypher-shell -u neo4j -p password "RETURN 1"
```

#### Issue 2: Authentication Failed

**Error**:
```
neo4j.exceptions.AuthError: The client is unauthorized due to authentication failure.
```

**Solution**:
```bash
# Reset Neo4j password
docker exec neo4j-graph cypher-shell -u neo4j -p neo4j \
    "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'newpassword'"

# Or use environment variables
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
python scripts/import_graph.py --input graph.json
```

#### Issue 3: Duplicate ID Constraint Violation

**Error**:
```
neo4j.exceptions.ConstraintError: Node already exists with label Application and id 'A5'
```

**Solution**:
```bash
# Use --clear flag to delete existing data
python scripts/import_graph.py --input graph.json --clear

# Or manually clear specific label
cypher-shell "MATCH (a:Application {id: 'A5'}) DELETE a"
```

#### Issue 4: Invalid QoS Values

**Error**:
```
ValueError: Invalid durability value: 'DURABLE'. Must be one of: VOLATILE, TRANSIENT_LOCAL, TRANSIENT, PERSISTENT
```

**Solution**:
Fix the JSON file:
```json
{
  "qos": {
    "durability": "PERSISTENT",  // Not "DURABLE"
    "reliability": "RELIABLE",   // Not "GUARANTEED"
    "transport_priority": "HIGH" // Not "PRIORITY_HIGH"
  }
}
```

#### Issue 5: Memory Issues with Large Graphs

**Error**:
```
java.lang.OutOfMemoryError: Java heap space
```

**Solution**:
```bash
# Increase Neo4j heap size
docker run \
    --name neo4j-graph \
    -e NEO4J_dbms_memory_heap_max__size=4G \
    -e NEO4J_dbms_memory_heap_initial__size=2G \
    neo4j:5-community

# Use batching for import
python scripts/import_graph.py \
    --input large_graph.json \
    --batch-size 500
```

#### Issue 6: Slow Dependency Derivation

**Symptom**: Derivation takes >5 minutes

**Solution**:
```bash
# Ensure indexes exist
cypher-shell << EOF
CREATE INDEX app_id_idx IF NOT EXISTS FOR (a:Application) ON (a.id);
CREATE INDEX topic_id_idx IF NOT EXISTS FOR (t:Topic) ON (t.id);
CREATE INDEX broker_id_idx IF NOT EXISTS FOR (b:Broker) ON (b.id);
EOF

# Use batching
python scripts/import_graph.py \
    --input graph.json \
    --batch-size 1000
```

---

## Best Practices

### 1. JSON Design

**DO**:
- Use descriptive, unique IDs (e.g., `sensor_temp_001` not `A1`)
- Include metadata for traceability
- Validate JSON before import (`--dry-run`)
- Use semantic topic names (e.g., `/robots/robot1/sensors/lidar`)

**DON'T**:
- Use special characters in IDs (spaces, quotes)
- Create circular RUNS_ON relationships
- Forget QoS policies (they affect weights significantly)
- Mix ID formats (stick to one convention)

### 2. Import Strategy

**For Development**:
```bash
# Always use --clear during development
python scripts/import_graph.py --input test.json --clear --validate
```

**For Production**:
```bash
# Use specific database, no --clear, with backup
neo4j-admin database dump neo4j --to-path=/backups/
python scripts/import_graph.py \
    --input production.json \
    --database production_db \
    --validate \
    --verbose
```

### 3. Performance Optimization

**Small Graphs (<100 components)**:
- Use default settings
- Single transaction import

**Medium Graphs (100-1000 components)**:
- Batch size: 1000
- Enable indexes
- Monitor memory usage

**Large Graphs (>1000 components)**:
- Batch size: 500
- Increase Neo4j heap
- Consider parallel import (future feature)
- Use Neo4j Enterprise for clustering

### 4. Validation Workflow

```bash
# 1. Validate JSON schema
python scripts/import_graph.py --input system.json --dry-run

# 2. Import to test database
python scripts/import_graph.py \
    --input system.json \
    --database test_db \
    --clear \
    --validate

# 3. Run queries to verify
cypher-shell -d test_db < verification_queries.cypher

# 4. If OK, import to production
python scripts/import_graph.py \
    --input system.json \
    --database production_db \
    --validate
```

### 5. Documentation

Always document your system topology:

```json
{
  "metadata": {
    "description": "Production ROS 2 autonomous vehicle system",
    "version": "2.1.0",
    "date": "2025-01-11",
    "author": "Robotics Team",
    "notes": [
      "Added new LiDAR sensor node",
      "Updated fusion algorithm QoS to RELIABLE",
      "Removed deprecated /diagnostics topic"
    ]
  },
  ...
}
```

### 6. Version Control

```bash
# Track topology changes in git
git add examples/production_system.json
git commit -m "Update topology: add redundant broker for HA"
git tag v2.1.0
```

---

## Summary

**Step 1: Graph Model Construction** establishes the foundation for critical component analysis by:

1. **Generating** system topologies (synthetic or custom)
2. **Importing** to Neo4j with weight calculation
3. **Deriving** logical dependencies from structural relationships
4. **Validating** graph integrity and completeness

The output is a **queryable graph database** ready for structural analysis (Step 2) and quality assessment (Step 3).

### Key Takeaways

✅ **generate_graph.py**: Creates realistic test systems with domain presets  
✅ **import_graph.py**: Transforms JSON → Neo4j with dependency derivation  
✅ **Weight Calculation**: QoS policies → quantitative criticality  
✅ **Dependency Derivation**: Pub-Sub patterns → logical dependencies  
✅ **Validation**: Ensures structural integrity and correctness  

### Next Steps

With the graph model constructed, proceed to:

- **Step 2**: [Structural Analysis](step-2-structural-analysis.md) - Compute centrality metrics
- **Step 3**: [Predictive Analysis](step-3-predictive-analysis.md) - Calculate RMA scores
- **Documentation**: Review [Graph Model](graph-model.md) for formal definitions

---

## References

- [Graph Model Documentation](graph-model.md)
- [Weight Calculations](weight-calculations.md)
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/)
- [JSON Schema Specification](https://json-schema.org/)

