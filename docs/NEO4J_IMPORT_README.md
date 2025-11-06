# Neo4j Graph Import 🗄️

## 🎉 Complete Neo4j Integration

Import generated DDS pub-sub system graphs into **Neo4j graph database** for advanced graph analytics, visualization, and querying.

## ✅ What's Been Implemented

### Neo4j Importer
- **[neo4j_importer.py](computer:///mnt/user-data/outputs/neo4j_importer.py)** (750+ lines)
  - Automatic schema creation
  - Batch import for large graphs
  - Node types: Node, Application, Topic, Broker
  - Relationship types: RUNS_ON, PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, DEPENDS_ON
  - Constraint and index creation
  - Sample queries
  - Statistics

### Docker Compose
- **[docker-compose-neo4j.yml](computer:///mnt/user-data/outputs/docker-compose-neo4j.yml)**
  - Neo4j 5.15
  - APOC plugins
  - Persistent storage
  - Optimized configuration

## 🎯 Quick Start

### 1. Install Neo4j Driver

```bash
pip install neo4j
# Or with system packages
pip install neo4j --break-system-packages
```

### 2. Start Neo4j

**Option A: Docker Compose**
```bash
docker-compose -f docker-compose-neo4j.yml up -d

# Check status
docker ps | grep neo4j

# View logs
docker logs dds-neo4j
```

**Option B: Docker Run**
```bash
docker run -d \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    --name dds-neo4j \
    neo4j:5.15
```

**Option C: Local Installation**
```bash
# Download from neo4j.com/download
# Start Neo4j
./bin/neo4j start
```

### 3. Import Graph

```bash
# Generate graph
python generate_graph.py --scale medium --output system.json

# Import to Neo4j
python neo4j_importer.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --input system.json
```

### 4. Access Neo4j Browser

Open: **http://localhost:7474**
- Username: `neo4j`
- Password: `password`

## 📊 Graph Schema

### Node Types

**Node** (Infrastructure)
```
Properties:
- id: string
- name: string
- cpu_capacity: float
- memory_gb: float
- network_bandwidth_mbps: float
- zone: string
- region: string
```

**Application**
```
Properties:
- id: string
- name: string
- type: string (PRODUCER/CONSUMER/PROSUMER)
- criticality: string (LOW/MEDIUM/HIGH/CRITICAL)
- replicas: int
- cpu_request: float
- memory_request_mb: float
```

**Topic**
```
Properties:
- id: string
- name: string
- message_size_bytes: int
- expected_rate_hz: int
- qos_durability: string
- qos_reliability: string
- qos_history_depth: int
- qos_deadline_ms: int
- qos_transport_priority: string
```

**Broker**
```
Properties:
- id: string
- name: string
- max_topics: int
- max_connections: int
```

### Relationship Types

**RUNS_ON** (Application → Node)
```
Direction: Application to Node
Properties: None
```

**PUBLISHES_TO** (Application → Topic)
```
Direction: Application to Topic
Properties:
- period_ms: int
- msg_size: int
```

**SUBSCRIBES_TO** (Application → Topic)
```
Direction: Application to Topic
Properties: None
```

**ROUTES** (Broker → Topic)
```
Direction: Broker to Topic
Properties: None
```

**DEPENDS_ON** (Application → Application)
```
Direction: Consumer to Producer
Properties: None
(Derived: Consumer depends on Producer if it subscribes to Producer's topics)
```

## 💡 Usage Examples

### Example 1: Basic Import

```bash
# Generate graph
python generate_graph.py --scale small --output small.json

# Import
python neo4j_importer.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --input small.json
```

### Example 2: Clear and Import

```bash
# Import with clear
python neo4j_importer.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --input system.json \
    --clear
```

### Example 3: Import with Sample Queries

```bash
# Import and run queries
python neo4j_importer.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --input system.json \
    --queries
```

### Example 4: Large Graph Import

```bash
# Generate large graph
python generate_graph.py --scale xlarge --output large.json

# Import with larger batch size
python neo4j_importer.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --input large.json \
    --batch-size 500
```

## 🔍 Sample Queries

### 1. View All Applications

```cypher
MATCH (a:Application)
RETURN a
LIMIT 25
```

### 2. Find Critical Applications

```cypher
MATCH (a:Application)
WHERE a.criticality = 'CRITICAL'
RETURN a.id, a.name, a.replicas, a.type
ORDER BY a.name
```

### 3. Pub-Sub Relationships

```cypher
MATCH (a:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
RETURN a, r, t
LIMIT 100
```

### 4. Most Connected Applications

```cypher
MATCH (a:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
WITH a, count(DISTINCT t) as topic_count
RETURN a.id, a.name, a.type, topic_count
ORDER BY topic_count DESC
LIMIT 10
```

### 5. Most Popular Topics

```cypher
MATCH (t:Topic)<-[:SUBSCRIBES_TO]-(a:Application)
WITH t, count(a) as subscriber_count
RETURN t.id, t.name, subscriber_count, t.expected_rate_hz
ORDER BY subscriber_count DESC
LIMIT 10
```

### 6. Broker Load Analysis

```cypher
MATCH (b:Broker)-[:ROUTES]->(t:Topic)
WITH b, count(t) as topic_count
RETURN b.id, b.name, topic_count, b.max_topics,
       round(100.0 * topic_count / b.max_topics, 2) as utilization_pct
ORDER BY utilization_pct DESC
```

### 7. Application Dependencies

```cypher
MATCH (consumer:Application)-[:SUBSCRIBES_TO]->(t:Topic)
      <-[:PUBLISHES_TO]-(producer:Application)
WHERE consumer.id <> producer.id
RETURN consumer.id, producer.id, collect(t.name) as shared_topics
ORDER BY size(shared_topics) DESC
LIMIT 20
```

### 8. Dependency Chains

```cypher
MATCH path = (a1:Application)-[:DEPENDS_ON*1..3]->(a2:Application)
RETURN a1.id, a2.id, length(path) as chain_length,
       [node in nodes(path) | node.id] as chain
ORDER BY chain_length DESC
LIMIT 10
```

### 9. Single Points of Failure

```cypher
MATCH (a:Application)-[:DEPENDS_ON]->(critical:Application)
WHERE critical.replicas = 1
WITH critical, count(DISTINCT a) as dependent_count
WHERE dependent_count > 5
RETURN critical.id, critical.name, critical.criticality,
       critical.replicas, dependent_count
ORDER BY dependent_count DESC
```

### 10. Cross-Zone Dependencies

```cypher
MATCH (a1:Application)-[:RUNS_ON]->(n1:Node),
      (a1)-[:DEPENDS_ON]->(a2:Application)-[:RUNS_ON]->(n2:Node)
WHERE n1.zone <> n2.zone
RETURN n1.zone as from_zone, n2.zone as to_zone,
       count(*) as dependency_count
ORDER BY dependency_count DESC
```

### 11. High-Throughput Topics

```cypher
MATCH (t:Topic)
WITH t, t.expected_rate_hz * t.message_size_bytes / 1024.0 as throughput_kb_s
WHERE throughput_kb_s > 100
RETURN t.id, t.name, t.expected_rate_hz,
       round(throughput_kb_s, 2) as throughput_kb_s
ORDER BY throughput_kb_s DESC
```

### 12. Reliable Topics

```cypher
MATCH (t:Topic)
WHERE t.qos_reliability = 'RELIABLE'
  AND t.qos_durability IN ['PERSISTENT', 'TRANSIENT_LOCAL']
RETURN t.id, t.name, t.qos_reliability, t.qos_durability,
       t.qos_deadline_ms, t.qos_history_depth
ORDER BY t.qos_deadline_ms ASC
```

### 13. Application Deployment Distribution

```cypher
MATCH (a:Application)-[:RUNS_ON]->(n:Node)
WITH n.zone as zone, count(a) as app_count
RETURN zone, app_count
ORDER BY app_count DESC
```

### 14. Topic Publisher/Subscriber Ratio

```cypher
MATCH (t:Topic)
OPTIONAL MATCH (t)<-[:PUBLISHES_TO]-(pub:Application)
OPTIONAL MATCH (t)<-[:SUBSCRIBES_TO]-(sub:Application)
WITH t, count(DISTINCT pub) as publishers, count(DISTINCT sub) as subscribers
WHERE publishers > 0 AND subscribers > 0
RETURN t.id, t.name, publishers, subscribers,
       round(1.0 * subscribers / publishers, 2) as fanout_ratio
ORDER BY fanout_ratio DESC
LIMIT 20
```

### 15. Find Communication Bottlenecks

```cypher
MATCH (b:Broker)-[:ROUTES]->(t:Topic)<-[r:PUBLISHES_TO]-(a:Application)
WITH b, t, sum(1000.0 / r.period_ms) as message_rate
WHERE message_rate > 100
RETURN b.id as broker, t.id as topic, round(message_rate, 2) as msg_per_sec
ORDER BY message_rate DESC
LIMIT 20
```

## 📊 Advanced Analytics

### Centrality Analysis

**PageRank (Application Importance)**
```cypher
CALL gds.pageRank.stream('myGraph')
YIELD nodeId, score
MATCH (a:Application) WHERE id(a) = nodeId
RETURN a.id, a.name, score
ORDER BY score DESC
LIMIT 10
```

**Betweenness Centrality (Critical Paths)**
```cypher
CALL gds.betweenness.stream('myGraph')
YIELD nodeId, score
MATCH (a:Application) WHERE id(a) = nodeId
WHERE score > 0
RETURN a.id, a.name, score
ORDER BY score DESC
LIMIT 10
```

### Community Detection

**Louvain (Application Clusters)**
```cypher
CALL gds.louvain.stream('myGraph')
YIELD nodeId, communityId
MATCH (a:Application) WHERE id(a) = nodeId
RETURN communityId, collect(a.id) as applications, count(*) as size
ORDER BY size DESC
```

### Path Finding

**Shortest Path Between Applications**
```cypher
MATCH (a1:Application {id: 'A1'}), (a2:Application {id: 'A10'})
MATCH path = shortestPath((a1)-[:DEPENDS_ON*]-(a2))
RETURN path, length(path) as hops
```

**All Paths (Impact Analysis)**
```cypher
MATCH path = (a1:Application {id: 'A1'})-[:DEPENDS_ON*1..5]->(a2:Application)
RETURN a1.id, a2.id, length(path) as distance,
       [node in nodes(path) | node.id] as path_nodes
ORDER BY distance
LIMIT 50
```

## 🎨 Visualization

### Neo4j Browser

Access at **http://localhost:7474**

**Example Visualization Queries:**

```cypher
// Full graph (small systems only)
MATCH (n) RETURN n LIMIT 100

// Application dependencies
MATCH (a:Application)-[r:DEPENDS_ON]->(b:Application)
RETURN a, r, b
LIMIT 50

// Pub-sub network
MATCH path = (a:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
RETURN path
LIMIT 100

// Broker topology
MATCH (b:Broker)-[:ROUTES]->(t:Topic)<-[r]-(a:Application)
RETURN b, t, a, r
LIMIT 100
```

### Style Configuration

```javascript
// In Neo4j Browser Settings
node {
  diameter: 50px;
  color: #A5ABB6;
  border-color: #9AA1AC;
  border-width: 2px;
  text-color-internal: #FFFFFF;
}

relationship {
  color: #A5ABB6;
  shaft-width: 1px;
  font-size: 8px;
  padding: 3px;
}

node.Application {
  color: #4C8EDA;
  caption: '{id}';
}

node.Topic {
  color: #DA7194;
  caption: '{name}';
}

node.Broker {
  color: #FCC940;
  caption: '{name}';
}

node.Node {
  color: #68BDF6;
  caption: '{zone}';
}
```

## 🔧 Integration Examples

### Example 1: Generate → Import → Query

```bash
# Generate IoT system
python generate_graph.py \
    --scale large \
    --scenario iot \
    --ha \
    --output iot_system.json

# Import to Neo4j
python neo4j_importer.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --input iot_system.json \
    --clear

# Query in Python
python << 'QUERY_EOF'
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", 
                              auth=("neo4j", "password"))

with driver.session() as session:
    result = session.run("""
        MATCH (a:Application)
        WHERE a.criticality = 'CRITICAL'
        RETURN count(a) as critical_count
    """)
    print(f"Critical apps: {result.single()['critical_count']}")

driver.close()
QUERY_EOF
```

### Example 2: Anti-Pattern Detection

```bash
# Generate with anti-patterns
python generate_graph.py \
    --scale medium \
    --antipatterns spof broker_overload \
    --output antipattern.json

# Import
python neo4j_importer.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --input antipattern.json \
    --clear \
    --queries
```

Then query SPOFs:
```cypher
MATCH (a:Application)-[:DEPENDS_ON]->(spof:Application)
WHERE spof.replicas = 1
WITH spof, count(a) as dependents
WHERE dependents > 10
RETURN spof.id, spof.name, dependents
ORDER BY dependents DESC
```

### Example 3: Time-Series Analysis

```bash
# Generate baseline
python generate_graph.py --scale medium --seed 42 --output t1.json

# Generate evolved system
python generate_graph.py --scale medium --seed 43 --output t2.json

# Import to different databases
python neo4j_importer.py --uri bolt://localhost:7687 \
    --database t1 --input t1.json --user neo4j --password password

python neo4j_importer.py --uri bolt://localhost:7687 \
    --database t2 --input t2.json --user neo4j --password password

# Compare in Neo4j Browser
```

## 📈 Performance Tips

### Optimization

**1. Batch Size**
```bash
# Larger batches for large graphs
python neo4j_importer.py ... --batch-size 1000
```

**2. Indexes**
```cypher
// Automatically created by importer
CREATE INDEX app_criticality IF NOT EXISTS 
FOR (a:Application) ON (a.criticality);
```

**3. Memory Configuration**
```yaml
# In docker-compose-neo4j.yml
environment:
  - NEO4J_dbms_memory_heap_max__size=4G
  - NEO4J_dbms_memory_pagecache_size=2G
```

### Query Optimization

**Use Indexes**
```cypher
// Good - uses index
MATCH (a:Application {criticality: 'CRITICAL'})
RETURN a

// Bad - full scan
MATCH (a:Application)
WHERE a.name CONTAINS 'Sensor'
RETURN a
```

**Limit Results**
```cypher
// Always use LIMIT for exploration
MATCH (a:Application)-[r]->(t:Topic)
RETURN a, r, t
LIMIT 100
```

## 🎉 Summary

**Complete Neo4j Integration:**

✅ **Automatic Import** - Batch processing for large graphs
✅ **Schema Creation** - Constraints and indexes
✅ **Node Types** - Node, Application, Topic, Broker
✅ **Relationships** - 5 types including derived dependencies
✅ **Sample Queries** - 15+ ready-to-use queries
✅ **Docker Support** - Easy deployment
✅ **Visualization** - Neo4j Browser integration
✅ **Analytics** - Centrality, communities, paths

**Perfect for:**
- Graph visualization
- Complex queries
- Dependency analysis
- Path finding
- Impact assessment
- Centrality analysis
- Community detection
- Time-series comparison

**Production-ready Neo4j integration!** 🚀

---

**Quick Commands:**

```bash
# Start Neo4j
docker-compose -f docker-compose-neo4j.yml up -d

# Generate & Import
python generate_graph.py --scale medium --output system.json
python neo4j_importer.py --uri bolt://localhost:7687 \
    --user neo4j --password password --input system.json

# Access Browser
open http://localhost:7474
```

**Full graph database capabilities!**
