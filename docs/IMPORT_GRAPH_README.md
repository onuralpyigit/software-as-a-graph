# Neo4j Graph Import System 🚀

Comprehensive Neo4j integration for importing, analyzing, and querying distributed pub-sub system graphs with advanced features and production-ready reliability.

## 📋 Table of Contents

- [Overview](#overview)
- [What's New](#whats-new)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Schema Reference](#schema-reference)
- [Query Examples](#query-examples)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The enhanced Neo4j import system provides production-ready capabilities for:

- **Graph Import**: Batch processing with transaction management
- **Schema Management**: Automatic constraints and indexes
- **Data Validation**: Pre-import validation and error reporting
- **Analytics**: Built-in queries for criticality analysis, dependency detection, and more
- **Performance**: Optimized for graphs with 10,000+ components
- **Reliability**: Comprehensive error handling and retry logic

## 🆕 What's New

### Enhanced Features

#### 1. **Improved Error Handling**
- Detailed error messages with context
- Graceful degradation on partial failures
- Connection retry logic
- Transaction rollback on errors

#### 2. **Data Validation**
- Pre-import validation of graph structure
- Reference integrity checking
- Orphaned component detection
- Data quality reporting

#### 3. **Progress Reporting**
- Real-time progress bars
- Batch-level statistics
- Performance metrics (items/sec, ETA)
- Detailed logging options

#### 4. **Advanced Analytics**
- Application type distribution
- Criticality analysis
- QoS policy analysis
- Dependency depth analysis
- Throughput calculations
- Network topology insights

#### 5. **Better Performance**
- Optimized batch processing
- Configurable batch sizes
- Parallel transaction support
- Memory-efficient processing

#### 6. **Enhanced Queries**
- 15+ sample queries
- Cross-zone dependency analysis
- Single Point of Failure (SPOF) detection
- Broker load analysis
- High-frequency topic identification

## 🔧 Installation

### Prerequisites

1. **Neo4j Database** (5.x or later)
   ```bash
   # Docker (recommended)
   docker run -d \
       -p 7474:7474 -p 7687:7687 \
       -e NEO4J_AUTH=neo4j/password \
       --name neo4j \
       neo4j:latest
   
   # Or download from: https://neo4j.com/download/
   ```

2. **Python Driver**
   ```bash
   pip install neo4j
   
   # Or with system packages
   pip install neo4j --break-system-packages
   ```

3. **Project Dependencies**
   ```bash
   # If using the full Software-as-a-Graph platform
   pip install -r requirements.txt
   ```

### Verify Installation

```bash
# Test Neo4j connection
python neo4j_utils.py --test

# Should output: ✓ Connection successful
```

## 🚀 Quick Start

### 1. Basic Import

```bash
# Generate a test graph
python generate_graph.py --scale medium --output system.json

# Import to Neo4j
python import_graph.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --input system.json
```

### 2. Import with Validation

```bash
python import_graph.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --input system.json \
    --validate \
    --progress
```

### 3. Import with Analytics

```bash
python import_graph.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --input system.json \
    --queries \
    --analytics \
    --export-stats import_stats.json
```

### 4. Clear and Re-import

```bash
python import_graph.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --input system.json \
    --clear \
    --batch-size 500
```

## 📊 Advanced Usage

### Large Graph Import

For graphs with >1000 components, optimize batch size:

```bash
python import_graph.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --input large_system.json \
    --batch-size 1000 \
    --progress
```

**Performance Tips:**
- **Small graphs (<100 components)**: Use default batch size (100)
- **Medium graphs (100-1000)**: Use batch size 200-500
- **Large graphs (>1000)**: Use batch size 500-1000
- **Very large (>10000)**: Consider splitting or using batch size 1000+

### Multiple Databases

Import to different databases for comparison:

```bash
# Import baseline
python import_graph.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --database baseline \
    --input system_v1.json

# Import updated version
python import_graph.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --database updated \
    --input system_v2.json
```

### Export and Backup

```bash
# Export graph to JSON
python neo4j_utils.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --export backup.json

# Get database info
python neo4j_utils.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --info
```

### Custom Queries

```bash
# Run custom Cypher query
python neo4j_utils.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --query "MATCH (a:Application) WHERE a.criticality = 'CRITICAL' RETURN count(a)"
```

## 📐 Schema Reference

### Node Types

#### **Node** (Infrastructure)
```cypher
CREATE (n:Node {
    id: string,              // Unique identifier
    name: string,            // Human-readable name
    cpu_capacity: float,     // CPU cores
    memory_gb: float,        // RAM in GB
    network_bandwidth_mbps: float,  // Network bandwidth
    zone: string,            // Availability zone
    region: string           // Geographic region
})
```

#### **Application**
```cypher
CREATE (a:Application {
    id: string,              // Unique identifier
    name: string,            // Application name
    type: string,            // PRODUCER/CONSUMER/PROSUMER
    criticality: string,     // CRITICAL/HIGH/MEDIUM/LOW
    replicas: int,           // Number of replicas
    cpu_request: float,      // CPU requirement
    memory_request_mb: float // Memory requirement
})
```

#### **Topic**
```cypher
CREATE (t:Topic {
    id: string,                      // Unique identifier
    name: string,                    // Topic name
    message_size_bytes: int,         // Message size
    expected_rate_hz: int,           // Expected frequency
    qos_durability: string,          // VOLATILE/TRANSIENT_LOCAL/PERSISTENT
    qos_reliability: string,         // BEST_EFFORT/RELIABLE
    qos_history_depth: int,          // History queue depth
    qos_deadline_ms: int,            // Deadline constraint
    qos_lifespan_ms: int,            // Message lifespan
    qos_transport_priority: string   // LOW/MEDIUM/HIGH
})
```

#### **Broker**
```cypher
CREATE (b:Broker {
    id: string,            // Unique identifier
    name: string,          // Broker name
    max_topics: int,       // Topic capacity
    max_connections: int   // Connection capacity
})
```

### Relationship Types

#### **RUNS_ON** (Application → Node)
```cypher
(a:Application)-[:RUNS_ON]->(n:Node)
```
Indicates which physical/virtual node hosts the application.

#### **PUBLISHES_TO** (Application → Topic)
```cypher
(a:Application)-[:PUBLISHES_TO {
    period_ms: int,    // Publishing period
    msg_size: int      // Message size
}]->(t:Topic)
```

#### **SUBSCRIBES_TO** (Application → Topic)
```cypher
(a:Application)-[:SUBSCRIBES_TO]->(t:Topic)
```

#### **ROUTES** (Broker → Topic)
```cypher
(b:Broker)-[:ROUTES]->(t:Topic)
```

#### **DEPENDS_ON** (Application → Application)
```cypher
(consumer:Application)-[:DEPENDS_ON]->(producer:Application)
```
Automatically derived: Consumer depends on Producer if it subscribes to Producer's topics.

### Indexes and Constraints

The system automatically creates:

**Constraints:**
- `node_id`: Unique ID for Node
- `app_id`: Unique ID for Application
- `topic_id`: Unique ID for Topic
- `broker_id`: Unique ID for Broker

**Indexes:**
- `app_type`: Application type
- `app_criticality`: Application criticality
- `app_name`: Application name
- `topic_name`: Topic name
- `node_zone`: Node zone
- `node_region`: Node region
- `broker_name`: Broker name

## 🔍 Query Examples

### Basic Queries

#### 1. View All Applications
```cypher
MATCH (a:Application)
RETURN a
LIMIT 25
```

#### 2. Find Critical Components
```cypher
MATCH (a:Application)
WHERE a.criticality = 'CRITICAL'
RETURN a.name, a.type, a.replicas
ORDER BY a.name
```

#### 3. Pub-Sub Network
```cypher
MATCH (a:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
RETURN a, r, t
LIMIT 100
```

### Dependency Analysis

#### 4. Application Dependencies
```cypher
MATCH (consumer:Application)-[:SUBSCRIBES_TO]->(t:Topic)
      <-[:PUBLISHES_TO]-(producer:Application)
WHERE consumer.id <> producer.id
RETURN consumer.name, producer.name, collect(t.name) as shared_topics
ORDER BY size(shared_topics) DESC
```

#### 5. Dependency Chains
```cypher
MATCH path = (a1:Application)-[:DEPENDS_ON*1..3]->(a2:Application)
RETURN a1.name, a2.name, length(path) as chain_length,
       [node in nodes(path) | node.name] as chain
ORDER BY chain_length DESC
LIMIT 10
```

#### 6. Circular Dependencies
```cypher
MATCH path = (a:Application)-[:DEPENDS_ON*2..5]->(a)
RETURN [node in nodes(path) | node.name] as cycle
LIMIT 10
```

### Critical Component Detection

#### 7. Single Points of Failure (SPOFs)
```cypher
MATCH (a:Application)-[:DEPENDS_ON]->(critical:Application)
WHERE critical.replicas = 1
WITH critical, count(DISTINCT a) as dependent_count
WHERE dependent_count > 5
RETURN critical.name, critical.criticality, 
       critical.replicas, dependent_count
ORDER BY dependent_count DESC
```

#### 8. Most Connected Applications
```cypher
MATCH (a:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
WITH a, count(DISTINCT t) as topic_count,
     count(CASE WHEN type(r) = 'PUBLISHES_TO' THEN 1 END) as pub_count,
     count(CASE WHEN type(r) = 'SUBSCRIBES_TO' THEN 1 END) as sub_count
RETURN a.name, topic_count, pub_count, sub_count
ORDER BY topic_count DESC
LIMIT 10
```

### Infrastructure Analysis

#### 9. Cross-Zone Dependencies
```cypher
MATCH (a1:Application)-[:RUNS_ON]->(n1:Node),
      (a1)-[:DEPENDS_ON]->(a2:Application)-[:RUNS_ON]->(n2:Node)
WHERE n1.zone <> n2.zone
RETURN n1.zone as from_zone, n2.zone as to_zone,
       count(*) as dependency_count
ORDER BY dependency_count DESC
```

#### 10. Broker Load Analysis
```cypher
MATCH (b:Broker)-[:ROUTES]->(t:Topic)
WITH b, count(t) as topic_count
RETURN b.name, topic_count, b.max_topics,
       round(100.0 * topic_count / b.max_topics, 2) as utilization_pct
ORDER BY utilization_pct DESC
```

### QoS and Performance

#### 11. High-Throughput Topics
```cypher
MATCH (t:Topic)
WITH t, t.expected_rate_hz * t.message_size_bytes / 1024.0 / 1024.0 as throughput_mb_s
WHERE throughput_mb_s > 1.0
RETURN t.name, t.expected_rate_hz, t.message_size_bytes,
       round(throughput_mb_s, 2) as throughput_mb_s
ORDER BY throughput_mb_s DESC
```

#### 12. Critical QoS Topics
```cypher
MATCH (t:Topic)
WHERE t.qos_reliability = 'RELIABLE'
  AND t.qos_durability = 'PERSISTENT'
  AND t.qos_deadline_ms < 100
RETURN t.name, t.qos_deadline_ms, t.expected_rate_hz
ORDER BY t.qos_deadline_ms ASC
```

#### 13. QoS Mismatch Detection
```cypher
MATCH (pub:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(sub:Application)
WHERE t.qos_reliability = 'BEST_EFFORT'
  AND sub.criticality = 'CRITICAL'
RETURN pub.name as publisher, t.name as topic, sub.name as subscriber,
       sub.criticality, t.qos_reliability
```

### Advanced Analytics

#### 14. Component Centrality (PageRank approximation)
```cypher
MATCH (a:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
WITH a, count(r) as degree
WITH a, degree, sum(degree) as total_degree
RETURN a.name, a.type, degree,
       round(100.0 * degree / total_degree, 2) as centrality_pct
ORDER BY degree DESC
LIMIT 10
```

#### 15. Topic Popularity
```cypher
MATCH (t:Topic)<-[:SUBSCRIBES_TO]-(sub:Application)
OPTIONAL MATCH (t)<-[:PUBLISHES_TO]-(pub:Application)
WITH t, count(DISTINCT sub) as subscriber_count,
     count(DISTINCT pub) as publisher_count
RETURN t.name, publisher_count, subscriber_count,
       subscriber_count - publisher_count as fan_out
ORDER BY subscriber_count DESC
LIMIT 10
```

## ⚡ Performance Tuning

### Batch Size Optimization

```python
# Small graphs (<100 nodes)
--batch-size 100

# Medium graphs (100-1000 nodes)
--batch-size 300

# Large graphs (1000-10000 nodes)
--batch-size 1000

# Very large graphs (>10000 nodes)
--batch-size 2000
```

### Neo4j Configuration

For large graphs, optimize Neo4j memory settings:

```yaml
# docker-compose.yml
environment:
  - NEO4J_dbms_memory_heap_max__size=4G
  - NEO4J_dbms_memory_pagecache_size=2G
  - NEO4J_dbms_query__cache__size=1000
```

### Query Optimization Tips

1. **Use Indexes**: Ensure queries use indexed properties
2. **Limit Results**: Always use `LIMIT` for exploration
3. **Avoid Cartesian Products**: Use proper `WHERE` clauses
4. **Profile Queries**: Use `PROFILE` to analyze query plans

```cypher
// Good - uses index
MATCH (a:Application {criticality: 'CRITICAL'})
RETURN a

// Bad - full scan
MATCH (a:Application)
WHERE a.name CONTAINS 'Sensor'
RETURN a

// Profile a query
PROFILE MATCH (a:Application)-[:DEPENDS_ON*1..3]->(b)
RETURN count(*)
```

## 🔧 Troubleshooting

### Connection Issues

**Problem**: Cannot connect to Neo4j
```
❌ Error connecting to Neo4j: ServiceUnavailable
```

**Solutions**:
1. Check Neo4j is running: `docker ps | grep neo4j`
2. Verify port 7687 is accessible: `nc -zv localhost 7687`
3. Check credentials are correct
4. Try: `docker logs neo4j` for Neo4j logs

### Import Failures

**Problem**: Import fails mid-process
```
❌ Error during import: Failed to import application batch
```

**Solutions**:
1. Run with `--validate` flag first
2. Check JSON file format is correct
3. Verify references (e.g., app references valid node)
4. Use smaller `--batch-size` (e.g., 50)
5. Run with `--verbose` for detailed logs

### Memory Issues

**Problem**: Import slow or fails on large graphs

**Solutions**:
1. Increase Neo4j heap size:
   ```bash
   docker run -d \
       -e NEO4J_dbms_memory_heap_max__size=8G \
       neo4j:latest
   ```

2. Use larger batch sizes: `--batch-size 1000`

3. Import in stages (split your JSON file)

### Query Performance

**Problem**: Queries are slow

**Solutions**:
1. Check indexes exist:
   ```cypher
   SHOW INDEXES
   ```

2. Profile slow queries:
   ```cypher
   PROFILE MATCH (a:Application)-[:DEPENDS_ON*]->(b)
   RETURN count(*)
   ```

3. Add `LIMIT` clauses to exploratory queries

4. Consider using APOC for complex operations

## 📚 Additional Resources

### Neo4j Browser

Access at: **http://localhost:7474**
- Username: `neo4j`
- Password: (your password)

### Neo4j Documentation
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [Performance Tuning](https://neo4j.com/docs/operations-manual/current/performance/)
- [APOC Procedures](https://neo4j.com/labs/apoc/)

### Project Resources
- `docs/IMPORT_GRAPH_README.md` - Original documentation
- `examples/` - Example usage scenarios
- `src/core/graph_importer.py` - Implementation details

## 🤝 Contributing

Improvements and bug fixes welcome! Key areas:
- Additional analytics queries
- Performance optimizations
- Better error messages
- More export formats

## 📝 License

Part of the Software-as-a-Graph research project.

---

**Questions or Issues?**
- Check the troubleshooting section above
- Review Neo4j logs: `docker logs neo4j`
- Run with `--verbose` flag for detailed output
- Use `--validate` before import to catch issues early

**Happy Graphing! 🎉**
