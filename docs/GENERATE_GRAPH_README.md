# Graph Generation Script 🎲

## 🎉 Comprehensive Graph Generation Tool

A powerful **graph generation script** that creates realistic DDS pub-sub system graphs at multiple scales with domain-specific patterns, anti-patterns, and high availability configurations.

**Fully integrated with the refactored architecture!**

## ✅ What's Been Implemented

### Generate Graph Script
- **[generate_graph.py](computer:///mnt/user-data/outputs/generate_graph.py)** (850+ lines)
  - Multiple scale presets (tiny → extreme)
  - Domain-specific scenarios (IoT, Financial, etc.)
  - Realistic QoS policies
  - Anti-pattern injection
  - High availability patterns
  - Validation integration
  - Multiple export formats

## 🎯 Quick Start

### 1. Basic Usage

```bash
# Generate small system
python generate_graph.py --scale small --output system.json

# Generate large IoT system
python generate_graph.py --scale large --scenario iot --output iot_system.json

# Generate with high availability
python generate_graph.py --scale medium --ha --output ha_system.json
```

### 2. With Anti-Patterns

```bash
# Single point of failure
python generate_graph.py --scale medium --antipatterns spof --output spof_system.json

# Multiple anti-patterns
python generate_graph.py --scale medium \
    --antipatterns spof broker_overload god_object \
    --output multi_antipattern.json
```

### 3. Validate and Export

```bash
# Validate graph
python generate_graph.py --scale medium --validate --output validated.json

# Export to multiple formats
python generate_graph.py --scale small \
    --formats json graphml gexf \
    --output multi_format
```

## 📊 Scale Presets

### Available Scales

| Scale | Nodes | Apps | Topics | Brokers | Use Case |
|-------|-------|------|--------|---------|----------|
| **tiny** | 3 | 5 | 3 | 1 | Testing, demos |
| **small** | 5 | 10 | 8 | 2 | Development |
| **medium** | 15 | 50 | 25 | 3 | Production small |
| **large** | 50 | 200 | 100 | 8 | Production medium |
| **xlarge** | 100 | 500 | 250 | 15 | Production large |
| **extreme** | 200 | 1000 | 500 | 20 | Stress testing |

### Example

```bash
# Tiny system (quick testing)
python generate_graph.py --scale tiny --output tiny.json

# Extreme scale (stress testing)
python generate_graph.py --scale extreme --output extreme.json
```

## 🎭 Domain Scenarios

### Available Scenarios

**1. Generic** (default)
- General-purpose services
- Balanced workload
- Standard QoS

**2. IoT**
- Sensor collectors
- Device managers
- Telemetry aggregators
- High-frequency, low-latency
- Best-effort reliability for sensors
- Reliable for commands

**3. Financial**
- Order processors
- Market data feeds
- Risk engines
- Ultra-low latency
- High durability
- RELIABLE reliability
- Persistent storage

**4. E-Commerce**
- Order services
- Inventory managers
- Payment processors
- Mixed QoS requirements
- Transactional consistency

**5. Analytics**
- Data collectors
- Stream processors
- ML pipelines
- High throughput
- Batch processing
- Best-effort for raw data

### Example

```bash
# IoT system
python generate_graph.py --scale large --scenario iot --output iot_system.json

# Financial trading system
python generate_graph.py --scale medium --scenario financial --output trading_system.json
```

## 🔧 Configuration Options

### Custom Parameters

Override scale defaults:

```bash
python generate_graph.py \
    --nodes 20 \
    --apps 100 \
    --topics 50 \
    --brokers 5 \
    --output custom.json
```

### Edge Density

Control connection density (0.0 - 1.0):

```bash
# Sparse connections (0.2)
python generate_graph.py --scale medium --density 0.2 --output sparse.json

# Dense connections (0.8)
python generate_graph.py --scale medium --density 0.8 --output dense.json
```

### Random Seed

Reproducible generation:

```bash
# Same seed = same graph
python generate_graph.py --scale medium --seed 42 --output graph1.json
python generate_graph.py --scale medium --seed 42 --output graph2.json
# graph1.json == graph2.json
```

## 🏗️ High Availability

Enable HA patterns:

```bash
python generate_graph.py --scale large --ha --output ha_system.json
```

**HA Features:**
- ✅ Application replication (2-5 replicas for critical apps)
- ✅ Multi-zone distribution
- ✅ Cross-region deployment
- ✅ Failover-ready topology

**Example Output:**
```json
{
  "applications": [
    {
      "id": "A1",
      "criticality": "CRITICAL",
      "replicas": 3
    }
  ],
  "nodes": [
    {"id": "N1", "zone": "zone-1", "region": "region-1"},
    {"id": "N2", "zone": "zone-2", "region": "region-1"},
    {"id": "N3", "zone": "zone-3", "region": "region-1"}
  ]
}
```

## ⚠️ Anti-Patterns

### Available Anti-Patterns

**1. SPOF (Single Point of Failure)**
```bash
python generate_graph.py --scale medium --antipatterns spof --output spof.json
```
- Critical app with no replicas
- Many dependencies on single component
- No redundancy

**2. Broker Overload**
```bash
python generate_graph.py --scale medium --antipatterns broker_overload --output overload.json
```
- All topics routed through single broker
- No load distribution
- Performance bottleneck

**3. God Object**
```bash
python generate_graph.py --scale medium --antipatterns god_object --output god.json
```
- One app subscribes to 80% of topics
- Knows everything about system
- Tight coupling

**4. Single Broker**
```bash
python generate_graph.py --scale medium --antipatterns single_broker --output single.json
```
- Only one broker in system
- No broker redundancy
- Single point of failure

**5. Tight Coupling**
```bash
python generate_graph.py --scale medium --antipatterns tight_coupling --output coupling.json
```
- Circular dependencies
- A1 → T1 → A2 → T2 → A3 → T3 → A1
- Cascading failures

### Multiple Anti-Patterns

```bash
python generate_graph.py --scale medium \
    --antipatterns spof broker_overload god_object \
    --output worst_case.json
```

## 📊 QoS Policies

### Scenario-Specific QoS

**Financial Systems:**
```json
{
  "qos": {
    "durability": "PERSISTENT",
    "reliability": "RELIABLE",
    "history_depth": 50,
    "deadline_ms": 10,
    "transport_priority": "URGENT"
  }
}
```

**IoT Telemetry:**
```json
{
  "qos": {
    "durability": "VOLATILE",
    "reliability": "BEST_EFFORT",
    "history_depth": 5,
    "deadline_ms": 100,
    "transport_priority": "MEDIUM"
  }
}
```

**IoT Commands:**
```json
{
  "qos": {
    "durability": "TRANSIENT_LOCAL",
    "reliability": "RELIABLE",
    "history_depth": 10,
    "deadline_ms": 50,
    "transport_priority": "HIGH"
  }
}
```

## ✅ Validation

### Validate Generated Graph

```bash
python generate_graph.py --scale medium --validate --output validated.json
```

**Validation Checks:**
- ✅ All applications have deployment (runs_on)
- ✅ All topics have broker routes
- ✅ No orphaned components
- ✅ Graph structure integrity
- ✅ Relationship consistency

**Example Output:**
```
Validating graph...
✓ Graph validation passed
```

**With Errors:**
```
Validating graph...
✗ Graph validation failed:
  - Applications without deployment: {'A5', 'A12'}
  - Topics without broker routes: {'T8'}

Continue with invalid graph? [y/N]:
```

## 📁 Export Formats

### Available Formats

**1. JSON** (default)
```bash
python generate_graph.py --scale small --output system.json
```

**2. GraphML**
```bash
python generate_graph.py --scale small --formats json graphml --output system
# Creates: system.json, system.graphml
```

**3. GEXF**
```bash
python generate_graph.py --scale small --formats json gexf --output system
# Creates: system.json, system.gexf
```

**4. Pickle**
```bash
python generate_graph.py --scale small --formats json pickle --output system
# Creates: system.json, system.pickle
```

**5. All Formats**
```bash
python generate_graph.py --scale small \
    --formats json graphml gexf pickle \
    --output system
# Creates: system.json, system.graphml, system.gexf, system.pickle
```

## 📊 Output Format

### JSON Structure

```json
{
  "metadata": {
    "version": "2.0",
    "generated_at": "2025-11-05T10:30:00",
    "scenario": "iot",
    "scale": "medium",
    "description": "Medium scale iot system"
  },
  "nodes": [
    {
      "id": "N1",
      "name": "Node1",
      "cpu_capacity": 16.0,
      "memory_gb": 32.0,
      "network_bandwidth_mbps": 5000.0,
      "zone": "zone-1",
      "region": "region-1"
    }
  ],
  "applications": [
    {
      "id": "A1",
      "name": "SensorCollector1",
      "type": "PRODUCER",
      "criticality": "HIGH",
      "replicas": 2,
      "cpu_request": 2.5,
      "memory_request_mb": 1024.0
    }
  ],
  "topics": [
    {
      "id": "T1",
      "name": "telemetry/1",
      "qos": {
        "durability": "VOLATILE",
        "reliability": "BEST_EFFORT",
        "history_depth": 5,
        "deadline_ms": 100,
        "transport_priority": "MEDIUM"
      },
      "message_size_bytes": 256,
      "expected_rate_hz": 50
    }
  ],
  "brokers": [
    {
      "id": "B1",
      "name": "Broker1",
      "max_topics": 150,
      "max_connections": 500
    }
  ],
  "relationships": {
    "publishes_to": [
      {
        "from": "A1",
        "to": "T1",
        "period_ms": 20,
        "msg_size": 256
      }
    ],
    "subscribes_to": [
      {
        "from": "A2",
        "to": "T1"
      }
    ],
    "routes": [
      {
        "from": "B1",
        "to": "T1"
      }
    ],
    "runs_on": [
      {
        "from": "A1",
        "to": "N1"
      }
    ]
  }
}
```

## 🔧 Integration Examples

### 1. Generate and Analyze

```bash
# Generate graph
python generate_graph.py --scale medium --output system.json

# Analyze with refactored tools
python example_refactored.py system.json
```

### 2. Generate and Simulate

```bash
# Generate graph
python generate_graph.py --scale large --scenario iot --output iot_system.json

# Run lightweight simulation
python example_lightweight_simulation.py iot_system.json
```

### 3. Generate and Test Failures

```bash
# Generate with anti-patterns
python generate_graph.py --scale medium --antipatterns spof --output spof_system.json

# Run failure simulation
python example_failure_simulation.py spof_system.json
```

### 4. Generate and Visualize

```bash
# Generate graph
python generate_graph.py --scale medium --output system.json

# Visualize
python example_visualization.py system.json
```

## 📊 Statistics Output

```
============================================================
GRAPH STATISTICS
============================================================

Components:
  Nodes: 15
  Applications: 50
  Topics: 25
  Brokers: 3

Relationships:
  Publishes: 145
  Subscribes: 150
  Routes: 25
  Runs On: 50

Application Types:
  CONSUMER: 14
  PROSUMER: 19
  PRODUCER: 17

Success! Graph saved to system.json
```

## 🎯 Use Cases

### 1. Testing & Development

```bash
# Quick test graph
python generate_graph.py --scale tiny --output test.json

# Development graph
python generate_graph.py --scale small --output dev.json
```

### 2. Anti-Pattern Research

```bash
# Generate baseline
python generate_graph.py --scale medium --output baseline.json

# Generate with SPOF
python generate_graph.py --scale medium --antipatterns spof --output spof.json

# Compare criticality scores
python example_refactored.py baseline.json
python example_refactored.py spof.json
```

### 3. Performance Benchmarking

```bash
# Generate multiple scales
for scale in tiny small medium large xlarge; do
    python generate_graph.py --scale $scale --output ${scale}_system.json
done

# Benchmark analysis time
for file in *_system.json; do
    time python example_refactored.py $file
done
```

### 4. Domain-Specific Systems

```bash
# IoT system
python generate_graph.py --scale large --scenario iot --ha --output iot_prod.json

# Financial system
python generate_graph.py --scale medium --scenario financial --output trading.json

# Analytics pipeline
python generate_graph.py --scale xlarge --scenario analytics --output analytics.json
```

## 💡 Advanced Examples

### Example 1: Custom IoT System

```bash
python generate_graph.py \
    --scenario iot \
    --nodes 30 \
    --apps 150 \
    --topics 80 \
    --brokers 6 \
    --ha \
    --density 0.4 \
    --seed 12345 \
    --validate \
    --formats json graphml \
    --output iot_custom
```

### Example 2: Worst-Case Scenario

```bash
python generate_graph.py \
    --scale large \
    --antipatterns spof broker_overload god_object tight_coupling \
    --output worst_case.json

# Test resilience
python example_failure_simulation.py worst_case.json
```

### Example 3: High-Performance Financial

```bash
python generate_graph.py \
    --scale xlarge \
    --scenario financial \
    --ha \
    --density 0.2 \
    --validate \
    --output trading_system.json

# Analyze criticality
python example_refactored.py trading_system.json
```

### Example 4: Reproducible Research

```bash
# Set 1 (seed 42)
python generate_graph.py --scale medium --seed 42 --output exp1_a.json
python generate_graph.py --scale medium --seed 42 --antipatterns spof --output exp1_b.json

# Set 2 (seed 123)
python generate_graph.py --scale medium --seed 123 --output exp2_a.json
python generate_graph.py --scale medium --seed 123 --antipatterns spof --output exp2_b.json

# Compare results
python example_refactored.py exp1_a.json
python example_refactored.py exp1_b.json
```

## 🎉 Summary

**Comprehensive Graph Generation:**

✅ **Multiple Scales** - Tiny to extreme (1000+ apps)
✅ **Domain Scenarios** - IoT, Financial, E-commerce, Analytics
✅ **Realistic QoS** - Scenario-specific policies
✅ **Anti-Patterns** - SPOF, overload, god object, etc.
✅ **High Availability** - Replication, zones, regions
✅ **Validation** - Built-in graph validation
✅ **Multiple Formats** - JSON, GraphML, GEXF, Pickle
✅ **Reproducible** - Random seed control

**Perfect for:**
- Testing & development
- Anti-pattern research
- Performance benchmarking
- Domain-specific systems
- Simulation scenarios
- Research & education

**Fully integrated with refactored architecture!** 🚀

---

**Quick Commands:**

```bash
# Basic
python generate_graph.py --scale small --output system.json

# IoT with HA
python generate_graph.py --scale large --scenario iot --ha --output iot.json

# With anti-patterns
python generate_graph.py --scale medium --antipatterns spof broker_overload --output bad.json

# Validated multi-format
python generate_graph.py --scale medium --validate --formats json graphml gexf --output system
```

**All scales, all scenarios, production-ready!**
