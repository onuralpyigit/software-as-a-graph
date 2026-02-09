# Step 2: Structural Analysis

**Measure each component's structural importance using graph centrality metrics.**

← [Step 1: Graph Model](graph-model.md) | → [Step 3: Quality Scoring](quality-scoring.md)

---

## What This Step Does

Structural Analysis takes the layer-projected graph G_analysis(l) from Step 1 and computes a set of topological metrics for every component. Each metric captures a different aspect of "importance" — how central a component is, how many paths flow through it, whether removing it would break the graph.

```
G_analysis(l)          Structural          Metric Vectors
(vertices + edges)  →  Analyzer     →     M(v) for each component
                                           + graph-level summary
```

The output feeds directly into Step 3, where these raw metrics are combined into quality scores.

## Why Multiple Metrics?

No single metric captures all aspects of criticality. A component might have high PageRank (many things depend on it transitively) but low Betweenness (it's not a bottleneck). By computing multiple independent metrics, we build a complete picture of each component's role in the system.

## Metric Catalogue

All metrics are normalized to [0, 1] for comparability.

### Centrality Metrics — "How important is this component?"

| Metric | Symbol | What It Measures | High Value Means |
|--------|--------|-----------------|-----------------|
| **PageRank** | PR(v) | Transitive importance via incoming dependencies | Many components depend on it (directly or indirectly) |
| **Reverse PageRank** | RPR(v) | Transitive importance via outgoing dependencies | It depends on many components (failure propagation source) |
| **Betweenness Centrality** | BT(v) | How often v lies on shortest paths between others | Bottleneck — removing it disrupts communication |
| **Closeness Centrality** | CL(v) | Average distance to all other components | Central position — can reach/be reached quickly |
| **Eigenvector Centrality** | EV(v) | Connection to other highly-connected components | Connected to important hubs (strategically valuable) |

### Degree Metrics — "How connected is this component?"

| Metric | Symbol | What It Measures | High Value Means |
|--------|--------|-----------------|-----------------|
| **In-Degree** | DG_in(v) | Number of incoming DEPENDS_ON edges | Many direct dependents |
| **Out-Degree** | DG_out(v) | Number of outgoing DEPENDS_ON edges | Depends on many things (high coupling) |

### Resilience Metrics — "What happens if this component is removed?"

| Metric | Symbol | What It Measures | High Value Means |
|--------|--------|-----------------|-----------------|
| **Clustering Coefficient** | CC(v) | How interconnected v's neighbors are | Well-connected neighborhood (more redundancy) |
| **Articulation Point Score** | AP(v) | Continuous measure of graph fragmentation upon removal | Removing it disconnects the graph (SPOF risk) |
| **Bridge Ratio** | BR(v) | Fraction of v's edges that are bridges | v's connections are irreplaceable |

### Edge and Weight Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Weight** w(v) | Component's own weight (from QoS) |
| **Weighted In-Degree** w_in(v) | Total weight of incoming edges |
| **Weighted Out-Degree** w_out(v) | Total weight of outgoing edges |

## How Articulation Points Are Scored

Traditional articulation point detection is binary (yes/no), which creates sharp discontinuities. We use a continuous score based on reachability loss:

```
AP_c(v) = 1 − |largest connected component after removing v| / (|V| − 1)
```

This means a true articulation point that splits the graph in half scores 0.50, while one that only disconnects a single leaf scores close to 0. Non-articulation points score 0.

## Output

For each component v in the selected layer, the analysis produces a metric vector:

```
M(v) = (PR, RPR, BT, CL, EV, DG_in, DG_out, CC, AP, BR, w, w_in, w_out)
```

Plus a graph-level summary: total vertices, total edges, density, average clustering, number of connected components, and diameter.

## Commands

```bash
# Analyze a specific layer
python bin/analyze_graph.py --layer app
python bin/analyze_graph.py --layer infra
python bin/analyze_graph.py --layer system   # all layers combined

# Export results to JSON
python bin/analyze_graph.py --layer system --output results/metrics.json
```

## What Comes Next

These raw metrics are individually informative, but they answer different questions about different quality concerns. Step 3 combines them into four interpretable quality dimensions — Reliability, Maintainability, Availability, and Vulnerability — using formally derived weights.

---

← [Step 1: Graph Model](graph-model.md) | → [Step 3: Quality Scoring](quality-scoring.md)