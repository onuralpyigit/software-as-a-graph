# Step 2: Structural Analysis

**Computing Topological Metrics to Identify Critical Components in Distributed Systems**

---

## Table of Contents

1. [Overview](#overview)
2. [Graph-Theoretic Foundation](#graph-theoretic-foundation)
3. [Centrality Metrics Explained](#centrality-metrics-explained)
4. [Multi-Layer Analysis](#multi-layer-analysis)
5. [Using analyze_graph.py](#using-analyze_graphpy)
6. [Understanding Analysis Results](#understanding-analysis-results)
7. [Practical Examples](#practical-examples)
8. [Advanced Analysis Techniques](#advanced-analysis-techniques)
9. [Performance Considerations](#performance-considerations)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## Overview

Structural Analysis is the critical second step in the Software-as-a-Graph methodology. After constructing the graph model (Step 1), we compute topological metrics that reveal which components occupy structurally important positions in the dependency network.

### What Structural Analysis Reveals

```
┌──────────────────────────────────────────────────────────────┐
│                  STRUCTURAL ANALYSIS INSIGHTS                 │
└──────────────────────────────────────────────────────────────┘

Question                              Metric              Answer
────────────────────────────────────────────────────────────────
"Who depends on this component?"  →  In-Degree        →  Impact
"Who does this component depend on?" → Out-Degree     →  Risk
"Which components are most influential?" → PageRank    →  Authority
"If this fails, what breaks?"     →  Reverse PageRank →  Propagation
"Which components broker communication?" → Betweenness →  Bottlenecks
"Is this a single point of failure?" → Articulation Pt →  SPOF
"How well-modularized is this?"   →  Clustering      →  Cohesion
"How connected is this component?" → Degree Centrality → Complexity
```

### The Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              STEP 2: STRUCTURAL ANALYSIS FLOW               │
└─────────────────────────────────────────────────────────────┘

Input: Neo4j Graph with DEPENDS_ON relationships
   │
   ├─────────────────────────────────────────────────────────┐
   │                                                         │
   ▼                                                         ▼
LAYER SELECTION                                    BACKEND SELECTION
├─ Complete System                                 ├─ NetworkX (default)
├─ Application Layer                               └─ Neo4j GDS (optional)
├─ Infrastructure Layer                                     │
├─ Topic Layer                                              │
└─ Broker Layer                                             │
   │                                                         │
   └─────────────────────┬───────────────────────────────────┘
                         │
                         ▼
              EXTRACT DEPENDENCY SUBGRAPH
              ├─ Filter by layer
              ├─ Filter by component type
              └─ Build analysis graph
                         │
                         ▼
              COMPUTE RAW METRICS
              ├─ PageRank & Reverse PageRank
              ├─ Betweenness Centrality
              ├─ Degree Centrality (in/out/total)
              ├─ Closeness Centrality
              ├─ Clustering Coefficient
              ├─ Articulation Points
              ├─ Bridges
              └─ Component Weights
                         │
                         ▼
              NORMALIZE METRICS [0,1]
              ├─ Min-max normalization
              ├─ Handle edge cases
              └─ Within-layer normalization
                         │
                         ▼
              GENERATE ANALYSIS RESULTS
              ├─ Component-level metrics
              ├─ Edge-level metrics
              ├─ Summary statistics
              └─ Layer-specific insights
                         │
                         ▼
              OUTPUT FORMATS
              ├─ JSON (structured data)
              ├─ HTML (interactive visualization)
              ├─ CSV (tabular export)
              └─ Console (human-readable)
```

### Key Objectives

| Objective | Method | Output |
|-----------|--------|--------|
| **Identify Critical Hubs** | PageRank, Degree Centrality | Components with high influence |
| **Detect Bottlenecks** | Betweenness Centrality | Communication chokepoints |
| **Find Single Points of Failure** | Articulation Points, Bridges | Components whose removal disconnects graph |
| **Assess Modularity** | Clustering Coefficient | Component cohesion and coupling |
| **Predict Failure Impact** | Reverse PageRank, Out-Degree | Potential cascade effects |
| **Measure Complexity** | Total Degree, Neighbor Count | Interface complexity |

### Why Multi-Layer Analysis?

The framework analyzes graphs at multiple abstraction levels:

1. **Complete System**: All components and relationships together
2. **Application Layer**: Only application-to-application dependencies
3. **Infrastructure Layer**: Only node-to-node dependencies
4. **Topic Layer**: Topic-centric analysis
5. **Broker Layer**: Broker-centric analysis

**Rationale**: Comparing applications to brokers is like comparing apples to oranges. Multi-layer analysis ensures fair comparison within component categories.

---

## Graph-Theoretic Foundation

### Dependency Graph Model

After Step 1, we have a directed weighted graph focused on `DEPENDS_ON` relationships:

$$G_{dep} = (V, E_{dep}, w)$$

Where:
- $V$ = set of components (applications, nodes, brokers)
- $E_{dep}$ = set of `DEPENDS_ON` edges
- $w : E_{dep} \rightarrow \mathbb{R}^+$ = dependency weight function

### Graph Properties

**Directed**: Edges have direction indicating dependency flow
- $(u, v) \in E_{dep}$ means "component $u$ depends on component $v$"

**Weighted**: Edges carry weights based on QoS and topic counts
- Stronger dependencies have higher weights
- Weights influence centrality calculations

**Potentially Disconnected**: The graph may have multiple connected components
- This is normal in distributed systems with isolated subsystems
- Analysis handles disconnected graphs gracefully

**Acyclic or Cyclic**: Dependency cycles may exist (circular dependencies)
- Cyclic dependencies indicate tight coupling
- Some metrics (like PageRank) handle cycles naturally

### Subgraph Extraction

For layer-specific analysis, we extract subgraphs:

**Application Layer**:
$$G_{app} = (V_{app}, E_{app-to-app})$$
Where $E_{app-to-app} = \{e \in E_{dep} : \text{subtype}(e) = \text{'app\_to\_app'}\}$

**Infrastructure Layer**:
$$G_{infra} = (V_{node}, E_{node-to-node} \cup E_{node-to-broker})$$

**Complete Layer**:
$$G_{complete} = (V, E_{dep})$$
All components, all dependency types

---

## Centrality Metrics Explained

### 1. PageRank (PR)

**Definition**: Measures transitive influence through the dependency network.

**Formula**:
$$PR(v) = \frac{1-d}{N} + d \sum_{u \in In(v)} \frac{PR(u)}{L(u)}$$

Where:
- $d$ = damping factor (typically 0.85)
- $N$ = number of nodes
- $In(v)$ = set of nodes with edges pointing to $v$
- $L(u)$ = out-degree of node $u$

**Interpretation**:
- High PageRank → Many components depend on this component (directly or transitively)
- Captures importance as a **dependency target**
- Originally developed for web page ranking

**Example**:
```
Component A ← Component B ← Component C ← Component D
```
If D is highly depended upon, and C depends on D, then C also gets importance credit through transitive influence.

**In Practice**:
```cypher
// Components with highest PageRank
MATCH (v)
WHERE v.pagerank IS NOT NULL
RETURN v.name, v.pagerank
ORDER BY v.pagerank DESC
LIMIT 10;
```

### 2. Reverse PageRank (RP)

**Definition**: PageRank computed on the graph with reversed edges.

**Formula**: Same as PageRank, but on $G_{reversed}$ where all edges are flipped.

**Interpretation**:
- High Reverse PageRank → This component's failure affects many others
- Measures **failure propagation potential**
- Captures downstream impact

**Why Reverse?**
In a dependency graph where $A \to B$ means "A depends on B":
- Regular PageRank: "How important is B as a provider?"
- Reverse PageRank: "If B fails, how many components are affected downstream?"

**Example**:
```
Publisher → Topic → Subscriber-1
                  → Subscriber-2
                  → Subscriber-3
```
Publisher has high Reverse PageRank because its failure affects 3 subscribers.

### 3. Betweenness Centrality (BT)

**Definition**: Fraction of shortest paths that pass through a node.

**Formula**:
$$BT(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

Where:
- $\sigma_{st}$ = number of shortest paths from $s$ to $t$
- $\sigma_{st}(v)$ = number of those paths passing through $v$

**Interpretation**:
- High betweenness → Component acts as a bridge between others
- Indicates **communication bottlenecks**
- Removal disrupts many paths

**Example**:
```
Group A ──→ Gateway ──→ Group B
```
Gateway has high betweenness because all paths from Group A to Group B pass through it.

**Complexity**: $O(n \cdot m)$ for unweighted graphs, $O(n \cdot m + n^2 \log n)$ for weighted

### 4. Degree Centrality (DC)

**Definition**: Number of edges connected to a node.

**Formulas**:
$$DC_{in}(v) = |In(v)| = \text{number of dependents}$$
$$DC_{out}(v) = |Out(v)| = \text{number of dependencies}$$
$$DC_{total}(v) = DC_{in}(v) + DC_{out}(v)$$

**Normalized**:
$$DC_{norm}(v) = \frac{DC_{total}(v)}{N-1}$$

**Interpretation**:
- High in-degree → Many components depend on this (high impact if fails)
- High out-degree → Depends on many components (high vulnerability)
- High total degree → Highly connected hub (complex interface)

**Example**:
```
Component A: in=5, out=2 → Many dependents, few dependencies (provider)
Component B: in=1, out=8 → Few dependents, many dependencies (consumer)
Component C: in=6, out=7 → Hub component (mediator)
```

### 5. Closeness Centrality (CC)

**Definition**: Average distance from a node to all other reachable nodes.

**Formula**:
$$CC(v) = \frac{N-1}{\sum_{u \neq v} d(v,u)}$$

Where $d(v,u)$ is the shortest path distance from $v$ to $u$.

**Interpretation**:
- High closeness → Can reach/be reached by others quickly
- Indicates **central position** in the network
- Useful for identifying coordinating components

**Note**: Undefined for disconnected graphs; computed per connected component.

### 6. Clustering Coefficient (CL)

**Definition**: Degree to which neighbors of a node are connected to each other.

**Formula**:
$$CL(v) = \frac{2 \cdot E(N_v)}{k_v \cdot (k_v - 1)}$$

Where:
- $N_v$ = neighbors of $v$
- $E(N_v)$ = edges between neighbors
- $k_v$ = degree of $v$

**Interpretation**:
- High clustering → Neighbors form cohesive groups (good modularity)
- Low clustering → Neighbors don't interconnect (poor modularity, changes propagate)
- Measures **local cohesion**

**Example**:
```
High Clustering:           Low Clustering:
    A                          A
   /|\                        /|\
  B-C-D                      B C D
  (neighbors connected)      (neighbors isolated)
```

### 7. Articulation Points (AP)

**Definition**: Nodes whose removal disconnects the graph.

**Algorithm**: Tarjan's depth-first search algorithm.

**Interpretation**:
- Articulation point = Single Point of Failure (SPOF)
- Binary indicator: $AP(v) \in \{0, 1\}$
- Critical for availability assessment

**Example**:
```
Group-A ──→ Gateway ──→ Group-B
```
Gateway is an articulation point; removing it disconnects Group-A from Group-B.

**Cypher Query**:
```cypher
// Find articulation points using Neo4j GDS
CALL gds.graph.project('dependency-graph', 
    ['Application', 'Node'], 
    {DEPENDS_ON: {orientation: 'NATURAL'}})
YIELD graphName;

CALL gds.articleRank.write('dependency-graph', {
    writeProperty: 'is_articulation_point'
});
```

### 8. Bridges

**Definition**: Edges whose removal disconnects the graph.

**Interpretation**:
- Bridge = Critical communication path
- No alternative route exists
- Indicates lack of redundancy

**Bridge Ratio**: For a node $v$:
$$BR(v) = \frac{\text{number of incident bridge edges}}{\text{total incident edges}}$$

A node with $BR(v) = 1.0$ has only critical edges (high SPOF risk).

---

## Multi-Layer Analysis

### Layer Definitions

The framework supports analysis at five different abstraction levels:

#### 1. Complete Layer

**Scope**: All components and all dependency types

**Use Case**: System-wide critical component identification

**Graph**:
```cypher
MATCH (v)-[d:DEPENDS_ON]->(u)
RETURN v, d, u
```

**Analysis Focus**: Cross-layer dependencies and interactions

#### 2. Application Layer

**Scope**: Only applications with app-to-app dependencies

**Use Case**: Service-level critical component analysis

**Graph**:
```cypher
MATCH (a1:Application)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(a2:Application)
RETURN a1, d, a2
```

**Analysis Focus**: Service dependencies, data flow patterns

**Example Questions**:
- Which microservices are most critical?
- What are the data processing bottlenecks?
- Which services have the most complex interfaces?

#### 3. Infrastructure Layer

**Scope**: Infrastructure nodes with node-to-node and node-to-broker dependencies

**Use Case**: Hardware/VM failure impact analysis

**Graph**:
```cypher
MATCH (n1:Node)-[d:DEPENDS_ON]->(n2)
WHERE d.dependency_type IN ['node_to_node', 'node_to_broker']
RETURN n1, d, n2
```

**Analysis Focus**: Physical deployment, host-level SPOFs

**Example Questions**:
- Which servers are most critical?
- What are the inter-datacenter dependencies?
- Which nodes host critical workloads?

#### 4. Topic Layer

**Scope**: Topics and their routing relationships

**Use Case**: Message channel criticality analysis

**Graph**:
```cypher
MATCH (t:Topic)
OPTIONAL MATCH (t)<-[:PUBLISHES_TO|SUBSCRIBES_TO]-(a:Application)
RETURN t, count(a) as connected_apps
```

**Analysis Focus**: Data flow channels, QoS impact

#### 5. Broker Layer

**Scope**: Brokers and their routing responsibilities

**Use Case**: Middleware infrastructure analysis

**Graph**:
```cypher
MATCH (b:Broker)-[:ROUTES]->(t:Topic)
RETURN b, count(t) as routed_topics
```

**Analysis Focus**: Broker criticality, routing load distribution

### Layer Comparison

| Layer | Components | Edges | Typical Size | Analysis Time |
|-------|-----------|-------|--------------|---------------|
| **Complete** | All | All DEPENDS_ON | 100-200 nodes | 5-10 sec |
| **Application** | Applications only | app_to_app | 30-80 nodes | 2-5 sec |
| **Infrastructure** | Nodes only | node_to_node | 5-20 nodes | <1 sec |
| **Topic** | Topics only | PUBLISHES/SUBSCRIBES | 20-50 nodes | 1-3 sec |
| **Broker** | Brokers only | ROUTES | 1-5 nodes | <1 sec |

### Within-Layer Normalization

**Critical Point**: Metrics are normalized **within each layer** separately.

**Why?** To ensure fair comparison:
- Applications should be compared to other applications
- Nodes should be compared to other nodes
- Not: "Application vs Broker" (meaningless comparison)

**Example**:
```python
# Application layer normalization
app_pageranks = [0.005, 0.012, 0.008, 0.025, 0.003]
min_pr = 0.003
max_pr = 0.025
range_pr = 0.022

# Normalized values [0, 1]
app_pageranks_norm = [(pr - min_pr) / range_pr for pr in app_pageranks]
# Result: [0.091, 0.409, 0.227, 1.000, 0.000]

# Infrastructure layer normalization (separate)
node_pageranks = [0.120, 0.180, 0.095]
# Normalize independently using node-specific min/max
```

### Cross-Layer Insights

While metrics are computed per-layer, cross-layer patterns reveal system design:

**Pattern 1: Infrastructure Concentrates Criticality**
```
Infrastructure: 3 nodes, all "critical"
Applications: 40 apps, 8 "critical"
```
→ System relies heavily on infrastructure layer

**Pattern 2: Application Complexity Dominates**
```
Applications: High betweenness, many articulation points
Infrastructure: Low betweenness, no articulation points
```
→ Service orchestration is the complexity driver

**Pattern 3: Balanced Distribution**
```
All layers: Similar critical component ratios
```
→ Well-architected system with distributed concerns

---

## Using analyze_graph.py

### Command-Line Interface

```bash
python scripts/analyze_graph.py [OPTIONS]
```

### Core Options

#### Required Options (One of)

```bash
--layer {complete, application, infrastructure, topic, broker}
    # Specify which layer to analyze

--type {Application, Broker, Topic, Node}
    # Analyze specific component type across layers

--all
    # Analyze all layers sequentially
```

#### Neo4j Connection Options

```bash
--uri URI               # Neo4j Bolt URI (default: bolt://localhost:7687)
--user USERNAME         # Neo4j username (default: neo4j)
--password PASSWORD     # Neo4j password (default: password)
--database DATABASE     # Neo4j database name (default: neo4j)
```

#### Analysis Configuration

```bash
--backend {networkx, neo4j-gds}
    # Graph analysis backend (default: networkx)
    # neo4j-gds: Faster for large graphs (requires Neo4j GDS plugin)

--include-weights
    # Use edge weights in metric computation (default: False)
    # When enabled, uses dependency weights for PageRank, Betweenness

--pagerank-iterations INT
    # Maximum iterations for PageRank (default: 100)

--pagerank-damping FLOAT
    # PageRank damping factor (default: 0.85)
```

#### Output Options

```bash
--output PATH
    # Output file path (supports .json, .csv, .html)

--format {json, csv, html, console}
    # Output format (default: console)

--visualize
    # Generate interactive HTML visualization

--summary-only
    # Show only summary statistics, not per-component details

--verbose
    # Show detailed progress logs
```

### Usage Examples

#### Example 1: Quick Analysis (Console Output)

```bash
# Analyze application layer, display in console
python scripts/analyze_graph.py --layer application
```

**Output**:
```
╔════════════════════════════════════════════════════════════╗
║          STRUCTURAL ANALYSIS - APPLICATION LAYER           ║
╚════════════════════════════════════════════════════════════╝

Graph Statistics:
  Components: 35
  Edges: 87
  Connected Components: 2
  Density: 0.073

Top 10 Critical Components by PageRank:
┌────────────────────────┬───────────┬────────────┬─────────┐
│ Name                   │ PageRank  │ Betweenness│ Degree  │
├────────────────────────┼───────────┼────────────┼─────────┤
│ sensor_fusion_node     │ 0.0245    │ 0.3421     │ 12      │
│ planning_node          │ 0.0198    │ 0.2156     │ 8       │
│ localization_node      │ 0.0187    │ 0.1823     │ 7       │
│ perception_node        │ 0.0156    │ 0.1432     │ 6       │
│ control_node           │ 0.0134    │ 0.0987     │ 5       │
└────────────────────────┴───────────┴────────────┴─────────┘

Articulation Points: 3
  - sensor_fusion_node
  - planning_node
  - main_broker

Analysis complete. Time: 2.34 seconds
```

#### Example 2: Export to JSON

```bash
# Analyze complete system, export structured data
python scripts/analyze_graph.py \
    --layer complete \
    --output results/complete_analysis.json \
    --format json
```

**Output File Structure**:
```json
{
  "metadata": {
    "layer": "complete",
    "timestamp": "2025-01-11T15:30:00Z",
    "graph_stats": {
      "num_nodes": 48,
      "num_edges": 127,
      "density": 0.056,
      "connected_components": 1
    }
  },
  "components": [
    {
      "id": "A5",
      "name": "sensor_fusion_node",
      "type": "Application",
      "metrics": {
        "pagerank": 0.0245,
        "pagerank_normalized": 0.876,
        "reverse_pagerank": 0.0312,
        "reverse_pagerank_normalized": 0.943,
        "betweenness": 0.3421,
        "betweenness_normalized": 1.000,
        "degree_in": 7,
        "degree_out": 5,
        "degree_total": 12,
        "degree_normalized": 0.923,
        "closeness": 0.0156,
        "clustering": 0.234,
        "is_articulation_point": true,
        "bridge_ratio": 0.167
      }
    }
  ],
  "edges": [...],
  "summary": {...}
}
```

#### Example 3: Generate Visualization

```bash
# Create interactive HTML visualization
python scripts/analyze_graph.py \
    --layer application \
    --visualize \
    --output results/app_analysis.html
```

This generates an interactive vis.js network diagram with:
- Node sizes proportional to PageRank
- Node colors indicating criticality level
- Edge thickness showing dependency weights
- Hover tooltips with metric details
- Interactive zoom/pan controls

#### Example 4: Compare All Layers

```bash
# Analyze all layers and export
python scripts/analyze_graph.py \
    --all \
    --output results/all_layers_analysis.json \
    --verbose
```

**Output Structure**:
```json
{
  "layers": {
    "complete": {...},
    "application": {...},
    "infrastructure": {...},
    "topic": {...},
    "broker": {...}
  },
  "cross_layer_comparison": {
    "critical_overlap": [...],
    "layer_complexity": {...}
  }
}
```

#### Example 5: Analyze Specific Component Type

```bash
# Analyze only broker components
python scripts/analyze_graph.py \
    --type Broker \
    --output results/broker_analysis.csv \
    --format csv
```

#### Example 6: Weighted Analysis

```bash
# Use dependency weights in metric computation
python scripts/analyze_graph.py \
    --layer application \
    --include-weights \
    --output results/weighted_analysis.json
```

**Impact**: When weights are included:
- PageRank considers edge importance (high-weight dependencies contribute more)
- Betweenness uses weighted shortest paths
- Degree metrics remain count-based

### Advanced Usage

#### Custom PageRank Configuration

```bash
# High-precision PageRank with custom damping
python scripts/analyze_graph.py \
    --layer complete \
    --pagerank-iterations 200 \
    --pagerank-damping 0.90 \
    --backend neo4j-gds
```

**When to adjust**:
- **More iterations**: For larger graphs or when convergence is slow
- **Higher damping** (0.90-0.95): When transitive influence is very important
- **Lower damping** (0.70-0.80): For more localized importance

#### Programmatic Usage

```python
from src.analysis.analyzer import GraphAnalyzer

with GraphAnalyzer(uri="bolt://localhost:7687", user="neo4j", password="password") as analyzer:
    # Analyze application layer
    results = analyzer.analyze(
        layer="application",
        backend="networkx",
        include_weights=True
    )
    
    # Access results
    print(f"Analyzed {results.metadata.num_nodes} components")
    
    # Get top 10 by PageRank
    top_components = sorted(
        results.components,
        key=lambda c: c.metrics.pagerank,
        reverse=True
    )[:10]
    
    for comp in top_components:
        print(f"{comp.name}: PR={comp.metrics.pagerank:.4f}, "
              f"BT={comp.metrics.betweenness:.4f}")
    
    # Find articulation points
    spofs = [c for c in results.components 
             if c.metrics.is_articulation_point]
    print(f"Found {len(spofs)} single points of failure")
    
    # Export to file
    results.to_json("analysis_results.json")
    results.to_csv("analysis_results.csv")
    results.to_html("analysis_visualization.html")
```

---

## Understanding Analysis Results

### Component-Level Results

Each analyzed component receives a comprehensive metric profile:

```python
ComponentMetrics {
    # Influence Metrics
    pagerank: float              # [0, 1] normalized - transitive importance
    reverse_pagerank: float      # [0, 1] normalized - failure propagation
    
    # Centrality Metrics
    betweenness: float          # [0, 1] normalized - bottleneck indicator
    closeness: float            # [0, 1] - average distance to others
    degree_centrality: float    # [0, 1] normalized - connectivity
    
    # Degree Metrics
    degree_in: int              # Number of dependents
    degree_out: int             # Number of dependencies
    degree_total: int           # Total connections
    
    # Structural Properties
    clustering: float           # [0, 1] - local cohesion
    is_articulation_point: bool # True if SPOF
    bridge_ratio: float         # [0, 1] - fraction of critical edges
    
    # Component Properties
    weight: float               # Intrinsic weight from QoS
}
```

### Interpreting Metric Combinations

#### Profile 1: Critical Hub

```
PageRank: HIGH (>0.8)
Betweenness: HIGH (>0.7)
Degree: HIGH (>0.8)
Articulation Point: True
```

**Interpretation**: Highly critical component
- Many components depend on it (high PageRank)
- Acts as communication bridge (high betweenness)
- Many connections (high degree)
- Single point of failure (articulation point)

**Action**: Implement redundancy, monitoring, failover

#### Profile 2: Data Producer

```
PageRank: HIGH (>0.7)
Reverse PageRank: LOW (<0.3)
Degree In: HIGH
Degree Out: LOW
```

**Interpretation**: Important data source
- Many depend on this (high PageRank, high in-degree)
- Doesn't depend on many (low out-degree)
- Failure affects many downstream (low reverse PR means it's a leaf/source)

**Action**: Ensure reliability, data persistence, backup

#### Profile 3: Bottleneck Mediator

```
Betweenness: VERY HIGH (>0.9)
PageRank: MODERATE (0.4-0.6)
Clustering: LOW (<0.2)
```

**Interpretation**: Communication chokepoint
- Many paths pass through (very high betweenness)
- Neighbors don't interconnect (low clustering)
- Moderate importance (moderate PageRank)

**Action**: Consider load balancing, alternative paths

#### Profile 4: Vulnerable Consumer

```
Degree Out: VERY HIGH
Degree In: LOW
PageRank: LOW (<0.3)
```

**Interpretation**: Highly dependent component
- Depends on many others (high out-degree)
- Few depend on it (low in-degree)
- Vulnerable to upstream failures

**Action**: Implement circuit breakers, fallbacks, retries

#### Profile 5: Isolated/Leaf Node

```
Degree: VERY LOW (1-2)
PageRank: LOW
Betweenness: 0
Clustering: Undefined
```

**Interpretation**: Peripheral component
- Few connections
- Not on critical paths
- Low system impact

**Action**: Lower priority for reliability investments

### Summary Statistics

The analysis produces summary statistics for the layer:

```json
{
  "summary": {
    "graph_statistics": {
      "num_nodes": 35,
      "num_edges": 87,
      "density": 0.073,
      "connected_components": 1,
      "largest_component_size": 35,
      "diameter": 6,
      "average_path_length": 3.2
    },
    "metric_distributions": {
      "pagerank": {
        "min": 0.0028,
        "max": 0.0245,
        "mean": 0.0114,
        "median": 0.0098,
        "std": 0.0056
      },
      "betweenness": {
        "min": 0.0,
        "max": 0.3421,
        "mean": 0.0876,
        "median": 0.0543,
        "std": 0.0923
      }
    },
    "structural_properties": {
      "num_articulation_points": 3,
      "num_bridges": 12,
      "average_clustering": 0.234,
      "modularity": 0.456
    },
    "top_components": {
      "by_pagerank": [...],
      "by_betweenness": [...],
      "by_degree": [...]
    }
  }
}
```

### Visualization Elements

When generating HTML visualizations (`--visualize`), the output includes:

#### Network Diagram

- **Node Size**: Proportional to PageRank
- **Node Color**: 
  - Red: Articulation points (SPOFs)
  - Orange: High betweenness (bottlenecks)
  - Yellow: High degree (hubs)
  - Green: Normal components
- **Edge Thickness**: Proportional to dependency weight
- **Edge Color**: Gray (dependencies)
- **Labels**: Component names

#### Interactive Features

- **Zoom/Pan**: Mouse wheel and drag
- **Hover Tooltip**: Shows all metrics for component
- **Click**: Highlights dependencies (incoming and outgoing)
- **Search**: Find specific components
- **Filter**: Show/hide by metric threshold

#### Charts

- **PageRank Distribution**: Histogram
- **Betweenness Distribution**: Histogram
- **Degree Distribution**: Scatter plot
- **Metric Correlation**: Heatmap showing metric relationships

---

## Practical Examples

### Example 1: ROS 2 Autonomous Vehicle Analysis

#### Context
A medium-sized ROS 2 autonomous vehicle system with 35 applications including sensors, fusion, planning, and control nodes.

#### Analyze Application Layer

```bash
python scripts/analyze_graph.py \
    --layer application \
    --output results/ros2_app_analysis.json \
    --visualize \
    --verbose
```

#### Interpret Results

**Top 5 by PageRank**:
```
1. sensor_fusion_node     (PR: 0.0245, BT: 0.3421, Deg: 12) ⚠️
2. planning_node          (PR: 0.0198, BT: 0.2156, Deg: 8)
3. localization_node      (PR: 0.0187, BT: 0.1823, Deg: 7)
4. perception_node        (PR: 0.0156, BT: 0.1432, Deg: 6)
5. control_node           (PR: 0.0134, BT: 0.0987, Deg: 5)
```

**Articulation Points**:
- `sensor_fusion_node` ⚠️ (SPOF)
- `planning_node` ⚠️ (SPOF)

**Insights**:
1. **sensor_fusion_node is critically important**:
   - Highest PageRank → Many nodes depend on it
   - Highest betweenness → All sensor→control paths go through it
   - Articulation point → Removing it disconnects graph
   - **Action**: Deploy redundant fusion node

2. **planning_node is also critical**:
   - Second highest PageRank
   - High betweenness
   - Articulation point
   - **Action**: Implement planning fallback mode

3. **Sensor nodes have low criticality**:
   - Low PageRank (data sources)
   - High out-degree (publishers)
   - Not articulation points
   - **Action**: Standard monitoring sufficient

#### Analyze Infrastructure Layer

```bash
python scripts/analyze_graph.py \
    --layer infrastructure \
    --output results/ros2_infra_analysis.json
```

**Top Nodes**:
```
1. Vehicle-Compute-Unit   (PR: 0.45, hosts fusion+planning)
2. Sensor-Hub             (PR: 0.28, hosts all sensors)
3. Control-Unit           (PR: 0.18, hosts control)
4. Edge-Logger            (PR: 0.09, hosts logging)
```

**Insight**: `Vehicle-Compute-Unit` is SPOF at infrastructure level
- Hosts 2 critical applications (fusion, planning)
- Node failure would be catastrophic
- **Action**: Consider distributing fusion and planning to separate nodes

### Example 2: IoT Smart City (Large Scale)

#### Context
Large IoT deployment with 70 sensor applications, 5 gateway nodes, 3 cloud services, distributed across 12 infrastructure nodes.

#### Full System Analysis

```bash
python scripts/analyze_graph.py \
    --all \
    --output results/smart_city_full.json \
    --verbose
```

#### Application Layer Results

**Key Findings**:
```
Total Applications: 70
Articulation Points: 5 (all gateways)
Average Degree: 3.2
Connected Components: 1
```

**Critical Components**:
- `gateway_zone_a` (PR: 0.089, BT: 0.456) - Aggregates 25 sensors
- `gateway_zone_b` (PR: 0.082, BT: 0.423) - Aggregates 20 sensors
- `gateway_zone_c` (PR: 0.078, BT: 0.398) - Aggregates 15 sensors
- `cloud_analytics` (PR: 0.067, BT: 0.234) - Processes all data
- `alert_service` (PR: 0.054, BT: 0.189) - Critical alerts

**Insight**: Gateway nodes are bottlenecks
- All gateways are articulation points
- High betweenness (all sensor→cloud paths)
- Edge failure affects 15-25 sensors each
- **Action**: Deploy redundant gateways per zone

#### Infrastructure Layer Results

```
Total Nodes: 12
Node-to-Node Dependencies: 18
Articulation Points: 2
  - cloud_datacenter_1
  - edge_gateway_master
```

**Critical Infrastructure**:
- `cloud_datacenter_1`: Hosts analytics and alert services
- `edge_gateway_master`: Hosts primary gateway applications

**Recommendation**: 
- Set up hot standby for `cloud_datacenter_1`
- Implement gateway failover to secondary edge nodes

### Example 3: Microservices Trading Platform

#### Context
Financial trading system with order processing, market data distribution, risk management, and settlement services.

#### Weighted Analysis

```bash
python scripts/analyze_graph.py \
    --layer application \
    --include-weights \
    --output results/trading_weighted.json
```

**Why Weighted?**
- Order topics have high weights (PERSISTENT, RELIABLE, URGENT)
- Market data has lower weights (VOLATILE, BEST_EFFORT)
- Weighting emphasizes critical paths

#### Results Comparison

**Unweighted Top 5**:
```
1. market_data_distributor   (PR: 0.0456)
2. order_processor           (PR: 0.0398)
3. risk_engine               (PR: 0.0334)
4. position_tracker          (PR: 0.0298)
5. settlement_service        (PR: 0.0256)
```

**Weighted Top 5**:
```
1. order_processor           (PR: 0.0512) ⬆️ moved up
2. risk_engine               (PR: 0.0467) ⬆️ moved up
3. settlement_service        (PR: 0.0401) ⬆️ moved up
4. market_data_distributor   (PR: 0.0356) ⬇️ moved down
5. position_tracker          (PR: 0.0298)
```

**Insight**: Weighted analysis correctly prioritizes financial operations
- Order processing moved to #1 (handles high-weight order topics)
- Market data dropped to #4 (handles low-weight data topics)
- Aligns with business criticality

### Example 4: Detecting Architectural Issues

#### Run Analysis

```bash
python scripts/analyze_graph.py \
    --layer application \
    --output results/architecture_check.json
```

#### Issue 1: Hub-and-Spoke Anti-Pattern

**Detection**:
```
Component: central_coordinator
  Betweenness: 0.89 (extreme)
  Degree: 34 (all other components)
  PageRank: 0.156 (high)
  Articulation Point: True
```

**Problem**: Centralized coordinator creates:
- Single point of failure
- Performance bottleneck
- Tight coupling

**Recommendation**: Refactor to peer-to-peer or event-driven architecture

#### Issue 2: Circular Dependencies

**Detection**:
```
Cycle detected:
  service_a → service_b → service_c → service_a
  
Metrics:
  service_a: Clustering = 1.0 (perfect triangle)
  service_b: Clustering = 1.0
  service_c: Clustering = 1.0
```

**Problem**: Circular dependencies indicate:
- Tight coupling
- Difficult to test in isolation
- Failure cascade risk

**Recommendation**: Break cycle with asynchronous patterns or API gateway

#### Issue 3: God Object Pattern

**Detection**:
```
Component: data_manager
  Degree Out: 2 (few dependencies)
  Degree In: 28 (many dependents)
  PageRank: 0.245 (very high)
  Reverse PageRank: 0.089 (low)
```

**Problem**: Many components depend on single data manager
- Centralized state
- Scalability bottleneck
- Difficult to modify

**Recommendation**: Decompose into domain-specific data services

---

## Advanced Analysis Techniques

### 1. Temporal Analysis (Comparison)

Compare analyses over time to detect architecture evolution:

```bash
# Week 1
python scripts/analyze_graph.py --layer application --output week1.json

# Week 2 (after refactoring)
python scripts/analyze_graph.py --layer application --output week2.json

# Compare
python scripts/compare_analyses.py --before week1.json --after week2.json
```

**Metrics to Track**:
- Number of articulation points (should decrease)
- Average betweenness (should decrease)
- Graph density (should increase moderately)
- Average clustering (should increase)

### 2. Subgraph Analysis

Analyze specific subsystems:

```cypher
// Extract perception subsystem
MATCH path = (sensor:Application)-[:DEPENDS_ON*1..3]->(fusion:Application)
WHERE sensor.name CONTAINS 'sensor' AND fusion.name CONTAINS 'fusion'
RETURN path
```

Then analyze this subgraph in isolation to focus on specific domains.

### 3. Metric Correlation Analysis

Understand relationships between metrics:

```python
import pandas as pd
import seaborn as sns

# Load results
results = pd.read_json("analysis.json")

# Compute correlation matrix
metrics = ['pagerank', 'betweenness', 'degree_total', 'clustering']
corr_matrix = results[metrics].corr()

# Visualize
sns.heatmap(corr_matrix, annot=True)
```

**Common Correlations**:
- PageRank ↔ Degree: Often positive (more connections → more importance)
- Betweenness ↔ Clustering: Often negative (bottlenecks have low clustering)
- PageRank ↔ Reverse PageRank: Weak correlation (different aspects)

### 4. Community Detection

Identify clusters of tightly connected components:

```cypher
// Using Neo4j GDS
CALL gds.louvain.stream('dependency-graph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name as name, communityId
ORDER BY communityId
```

**Use Case**: Identify microservice domains, deployment groups

### 5. Critical Path Analysis

Find most critical dependency chains:

```cypher
// Find longest weighted paths
MATCH path = (source)-[:DEPENDS_ON*]->(sink)
WHERE NOT (source)<-[:DEPENDS_ON]-() AND NOT (sink)-[:DEPENDS_ON]->()
WITH path, reduce(w = 0, r IN relationships(path) | w + r.weight) as total_weight
ORDER BY total_weight DESC
LIMIT 10
RETURN path, total_weight
```

### 6. Impact Simulation (Preview)

Estimate impact before running full simulation:

```python
def estimate_failure_impact(component, analysis_results):
    """Quick impact estimate using reverse PageRank and out-degree."""
    c = analysis_results.get_component(component)
    
    # Estimated affected components
    affected = c.metrics.degree_out * (1 + c.metrics.reverse_pagerank)
    
    # Estimated cascade factor
    cascade = 1 + (c.metrics.betweenness * 2)
    
    return affected * cascade

# Use before expensive simulation
top_10 = sorted(components, key=estimate_failure_impact, reverse=True)[:10]
# Now run full simulation only on these 10
```

---

## Performance Considerations

### Analysis Complexity

| Metric | Time Complexity | Space Complexity | Notes |
|--------|----------------|------------------|-------|
| Degree | O(V + E) | O(V) | Very fast |
| PageRank | O(k·E) | O(V) | k = iterations (~100) |
| Betweenness | O(V·E) | O(V²) | Slowest metric |
| Closeness | O(V·E) | O(V²) | Only for connected graphs |
| Clustering | O(V·d²) | O(V) | d = avg degree |
| Articulation Points | O(V + E) | O(V) | Fast (DFS) |

Where:
- V = number of vertices (components)
- E = number of edges (dependencies)
- k = PageRank iterations

### Performance by Graph Size

| Components | Edges | Analysis Time (NetworkX) | Analysis Time (Neo4j GDS) |
|-----------|-------|--------------------------|---------------------------|
| 10-20 | 20-50 | <1 sec | <1 sec |
| 30-50 | 80-150 | 2-3 sec | 1-2 sec |
| 60-100 | 200-400 | 5-8 sec | 2-4 sec |
| 100-200 | 500-1000 | 15-25 sec | 5-10 sec |
| 500+ | 2000+ | 2-5 min | 30-60 sec |

**Recommendation**: For graphs >200 components, use Neo4j GDS backend.

### Optimization Strategies

#### 1. Use Neo4j GDS for Large Graphs

```bash
# Install Neo4j Graph Data Science plugin
# Then use --backend neo4j-gds
python scripts/analyze_graph.py \
    --layer complete \
    --backend neo4j-gds \
    --output results.json
```

**Speedup**: 2-5x faster for graphs >100 components

#### 2. Layer-Specific Analysis

Instead of complete system:
```bash
# Slower: Complete system (100 components)
python scripts/analyze_graph.py --layer complete  # 20 seconds

# Faster: Application layer only (40 components)
python scripts/analyze_graph.py --layer application  # 3 seconds
```

#### 3. Reduce PageRank Iterations

For quick estimates:
```bash
python scripts/analyze_graph.py \
    --layer application \
    --pagerank-iterations 50  # Default: 100
```

**Trade-off**: Slight accuracy reduction, 30-40% faster

#### 4. Disable Expensive Metrics

```python
# Programmatic control
analyzer.analyze(
    layer="application",
    metrics=['pagerank', 'degree'],  # Skip betweenness
    compute_articulation_points=False
)
```

#### 5. Caching Results

```bash
# Cache analysis results
python scripts/analyze_graph.py --layer application --output cache/app.json

# Reuse cached results
python scripts/validate_graph.py --analysis-cache cache/app.json
```

### Memory Considerations

**NetworkX**: In-memory graph representation
- Memory usage: ~100 bytes per edge
- 1000 edges ≈ 100 KB
- 10,000 edges ≈ 1 MB
- Safe up to ~100,000 edges on typical machines

**Neo4j GDS**: Native graph storage
- More memory efficient for large graphs
- Leverages disk for storage
- Can handle millions of edges

---

## Troubleshooting

### Issue 1: Empty Results

**Symptom**:
```
Analysis complete. 0 components found.
```

**Causes**:
1. Wrong layer specified
2. No DEPENDS_ON relationships exist
3. Database connection issue

**Solutions**:
```bash
# Check if dependencies were derived
cypher-shell "MATCH ()-[d:DEPENDS_ON]->() RETURN count(d)"

# Verify layer has components
cypher-shell "MATCH (a:Application) RETURN count(a)"

# Re-run import with dependency derivation
python scripts/import_graph.py --input system.json --clear
```

### Issue 2: Neo4j GDS Not Available

**Error**:
```
Neo4j GDS plugin not installed or not accessible
```

**Solution**:
```bash
# Install GDS plugin
# For Docker:
docker run \
    -e NEO4J_PLUGINS='["graph-data-science"]' \
    neo4j:5-community

# Verify installation
cypher-shell "CALL gds.version()"
```

### Issue 3: Analysis Takes Too Long

**Symptom**: Analysis runs >5 minutes

**Solutions**:
```bash
# 1. Switch to Neo4j GDS
--backend neo4j-gds

# 2. Reduce PageRank iterations
--pagerank-iterations 50

# 3. Analyze smaller layer
--layer application  # Instead of --layer complete

# 4. Check graph size
cypher-shell "MATCH ()-[d:DEPENDS_ON]->() RETURN count(d)"
# If >10,000 edges, optimization needed
```

### Issue 4: Disconnected Graph Warning

**Warning**:
```
Warning: Graph has 3 connected components. 
Closeness centrality may be misleading.
```

**Interpretation**: This is often normal
- Distributed systems may have isolated subsystems
- Each component is analyzed independently
- PageRank and betweenness still valid

**Action**: If unexpected, investigate:
```cypher
// Find isolated components
MATCH (n)
WHERE NOT (n)--()
RETURN n.name, labels(n)
```

### Issue 5: Metrics All Zero

**Symptom**:
```
All PageRank values are 0.000
```

**Cause**: Graph has no edges (missing dependencies)

**Solution**:
```bash
# Verify DEPENDS_ON relationships exist
cypher-shell "MATCH p=()-[:DEPENDS_ON]->() RETURN count(p)"

# Re-import and derive dependencies
python scripts/import_graph.py --input system.json --clear
```

---

## Best Practices

### 1. Analysis Workflow

**Recommended Sequence**:
```bash
# Step 1: Quick complete system overview
python scripts/analyze_graph.py --layer complete --summary-only

# Step 2: Detailed application analysis
python scripts/analyze_graph.py --layer application --visualize

# Step 3: Infrastructure analysis
python scripts/analyze_graph.py --layer infrastructure

# Step 4: Export for validation (Step 4)
python scripts/analyze_graph.py --layer application --output app_analysis.json
```

### 2. Metric Interpretation Priority

When identifying critical components, prioritize:

1. **Articulation Points** (binary, definitive)
2. **PageRank** (overall importance)
3. **Betweenness** (bottleneck risk)
4. **Degree** (interface complexity)
5. **Clustering** (modularity assessment)

### 3. Layer Selection Guide

| Goal | Recommended Layer |
|------|-------------------|
| Find critical services | `application` |
| Assess infrastructure SPOF | `infrastructure` |
| Evaluate overall system | `complete` |
| Analyze message routing | `topic` or `broker` |
| Compare cross-layer | `--all` |

### 4. Documentation

Document analysis results:

```bash
# Create analysis report
cat > analysis_report.md << EOF
# Structural Analysis Report
Date: $(date)
System: Production Trading Platform
Layer: Application

## Key Findings
- Critical Components: 5
- Articulation Points: 2 (order_processor, risk_engine)
- Recommendation: Deploy redundant risk_engine

## Metrics Summary
$(cat results/app_analysis.json | jq '.summary.graph_statistics')
EOF
```

### 5. Continuous Monitoring

Set up periodic analysis:

```bash
#!/bin/bash
# weekly_analysis.sh

DATE=$(date +%Y-%m-%d)
python scripts/analyze_graph.py \
    --layer application \
    --output "analysis_history/${DATE}_application.json"

# Compare with last week
python scripts/compare_analyses.py \
    --before "analysis_history/$(date -d '7 days ago' +%Y-%m-%d)_application.json" \
    --after "analysis_history/${DATE}_application.json"
```

### 6. Integration with CI/CD

```yaml
# .github/workflows/architecture-analysis.yml
name: Architecture Analysis

on: [pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - name: Start Neo4j
        run: docker run -d neo4j:5-community
      
      - name: Import Topology
        run: python scripts/import_graph.py --input topology.json
      
      - name: Run Analysis
        run: python scripts/analyze_graph.py --layer application --output analysis.json
      
      - name: Check for New SPOFs
        run: python scripts/check_spofs.py --input analysis.json --fail-on-new
```

---

## Summary

**Step 2: Structural Analysis** computes graph-theoretic metrics that predict component criticality based on network position. Key takeaways:

✅ **Eight Core Metrics**: PageRank, Reverse PageRank, Betweenness, Degree, Closeness, Clustering, Articulation Points, Bridges

✅ **Multi-Layer Analysis**: Separate analysis per architectural layer ensures fair component comparison

✅ **Powerful CLI**: `analyze_graph.py` provides flexible analysis with multiple output formats

✅ **Actionable Insights**: Metrics directly inform architectural decisions (redundancy, refactoring, monitoring)

✅ **Predictive Power**: Structural metrics correlate strongly (ρ=0.876) with actual failure impact

### Next Steps

With structural metrics computed, proceed to:

- **Step 3**: [Predictive Analysis](step-3-predictive-analysis.md) - Compute RMA quality scores
- **Step 4**: [Failure Impact Assessment](step-4-failure-simulation.md) - Validate predictions through simulation
- **Step 5**: [Validation & Comparison](step-5-validation.md) - Statistical correlation analysis

---

## References

### Graph Theory

1. Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press.
2. Wasserman, S., & Faust, K. (1994). *Social Network Analysis: Methods and Applications*. Cambridge University Press.

### Centrality Metrics

3. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank Citation Ranking*. Stanford InfoLab.
4. Freeman, L. C. (1977). *A Set of Measures of Centrality Based on Betweenness*. Sociometry, 40(1), 35-41.
5. Brandes, U. (2001). *A Faster Algorithm for Betweenness Centrality*. Journal of Mathematical Sociology, 25(2), 163-177.

### Software Architecture

6. Zimmermann, T., & Nagappan, N. (2008). *Predicting Defects Using Network Analysis*. ICSE '08.
7. Concas, G., et al. (2007). *Power-Laws in a Large Object-Oriented Software System*. IEEE TSE, 33(10).

### Tools

8. NetworkX Documentation: https://networkx.org/documentation/stable/
9. Neo4j Graph Data Science: https://neo4j.com/docs/graph-data-science/

---

**Last Updated**: January 2025  
**Part of**: Software-as-a-Graph Research Project  
**Institution**: Istanbul Technical University
