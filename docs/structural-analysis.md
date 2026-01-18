# Step 2: Structural Analysis

**Compute topological metrics that capture different aspects of component criticality**

---

## Overview

Structural Analysis computes graph metrics on the DEPENDS_ON subgraph to quantify each component's structural importance.

```
┌─────────────────────┐          ┌─────────────────────┐
│  Graph Model        │          │  Structural Metrics │
│                     │    →     │                     │
│  - Vertices         │          │  - PageRank         │
│  - DEPENDS_ON edges │          │  - Betweenness      │
│  - Weights          │          │  - Articulation Pts │
│                     │          │  - Clustering       │
└─────────────────────┘          └─────────────────────┘
```

---

## Topological Metrics

The analysis computes these metrics for each component:

| Metric | Symbol | Measures | High Value Means |
|--------|--------|----------|------------------|
| **PageRank** | PR | Transitive influence | Depended on by important components |
| **Reverse PageRank** | RPR | Failure propagation | Failure cascades downstream |
| **Betweenness** | BT | Bottleneck centrality | Lies on many shortest paths |
| **Degree** | DG | Connection count | Many direct connections |
| **In-Degree** | ID | Incoming connections | Many dependents |
| **Clustering** | CC | Local modularity | Well-connected neighborhood |
| **Eigenvector** | EV | Strategic importance | Connected to other VIPs |
| **Closeness** | CL | Propagation speed | Can reach others quickly |

### Structural Properties

| Property | Type | Meaning |
|----------|------|---------|
| **Articulation Point** | Boolean | Removal disconnects the graph (SPOF) |
| **Bridge Ratio** | Float | Fraction of edges that are bridges |

---

## Metric Details

### PageRank (PR)

Measures transitive influence—how important are the components that depend on this one.

```
PR(v) = (1-d)/n + d × Σ PR(u)/out(u)
```

Where d = 0.85 (damping factor).

**Use**: Identifies components that are structurally important through their dependencies.

### Reverse PageRank (RPR)

PageRank computed on the reversed graph—measures downstream failure propagation.

**Use**: Identifies components whose failure would cascade to many others.

### Betweenness Centrality (BT)

Fraction of shortest paths that pass through a component.

```
BT(v) = Σ σ(s,t|v) / σ(s,t)
```

**Use**: Identifies bottlenecks and communication hubs.

### Articulation Points (AP)

Components whose removal disconnects the graph into multiple components.

**Use**: Critical for identifying Single Points of Failure (SPOFs).

### Clustering Coefficient (CC)

Measures how connected a component's neighbors are to each other.

```
CC(v) = 2×|edges among neighbors| / (k×(k-1))
```

**Use**: High clustering = better modularity = easier to maintain.

### Eigenvector Centrality (EV)

Like PageRank but without damping—measures connection to other highly-connected nodes.

**Use**: Identifies strategically important nodes (high-value targets).

### Closeness Centrality (CL)

Inverse of average shortest path distance to all other nodes.

**Use**: Identifies nodes that can quickly reach/affect others.

---

## Analysis Pipeline

```
1. Extract Layer Subgraph
   └── Filter by dependency type (app_to_app, node_to_node, etc.)

2. Compute Metrics
   ├── PageRank (forward and reverse)
   ├── Betweenness Centrality
   ├── Degree Centrality (in, out, total)
   ├── Clustering Coefficient
   ├── Eigenvector Centrality
   ├── Closeness Centrality
   └── Articulation Points / Bridges

3. Normalize Metrics
   └── Min-max scaling to [0, 1]
```

---

## Commands

```bash
# Analyze specific layer
python analyze_graph.py --layer system

# Analyze all layers
python analyze_graph.py --all

# Export results to JSON
python analyze_graph.py --layer system --output results/analysis.json
```

---

## Output Example

```
═══════════════════════════════════════════════════════════════
  STRUCTURAL ANALYSIS - System Layer
═══════════════════════════════════════════════════════════════

  Graph Statistics:
    Nodes:              48
    Edges:              127
    Density:            0.056
    Connected:          Yes
    
  Structural Properties:
    Articulation Points: 5
    Bridges:            8
    
  Top Components by PageRank:
    1. sensor_fusion     0.0892
    2. main_broker       0.0756
    3. planning_node     0.0634
    ...
```

---

## Layer Analysis

Different layers reveal different structural patterns:

| Layer | What It Reveals |
|-------|-----------------|
| **app** | Software coupling, data flow bottlenecks |
| **infra** | Hardware SPOFs, network topology issues |
| **mw-app** | Middleware impact on applications |
| **system** | Complete system criticality picture |

---

## Key Formulas Reference

| Metric | Formula |
|--------|---------|
| PageRank | `PR(v) = (1-d)/n + d × Σ PR(u)/out(u)` |
| Betweenness | `BT(v) = Σ σ(s,t|v) / σ(s,t)` |
| Clustering | `CC(v) = 2×edges / (k×(k-1))` |
| Closeness | `CL(v) = (n-1) / Σ d(v,u)` |

---

## Next Step

→ [Step 3: Quality Scoring](quality-scoring.md)
