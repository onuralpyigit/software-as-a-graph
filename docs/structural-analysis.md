# Step 2: Analysis

**Measure each component's structural importance using graph-theoretic metrics.**

← [Step 1: Modeling](graph-model.md) | → [Step 3: Prediction](prediction.md)

---

## Table of Contents

1. [What This Step Does](#what-this-step-does)
2. [Why Multiple Metrics?](#why-multiple-metrics)
3. [Formal Definitions](#formal-definitions)
   - [Normalization](#normalization)
   - [PageRank](#pagerank)
   - [Reverse PageRank](#reverse-pagerank)
   - [Betweenness Centrality](#betweenness-centrality)
   - [Closeness Centrality](#closeness-centrality)
   - [Eigenvector Centrality](#eigenvector-centrality)
   - [Degree Metrics](#degree-metrics)
   - [Clustering Coefficient](#clustering-coefficient)
   - [Articulation Point Score](#articulation-point-score)
   - [Bridge Ratio](#bridge-ratio)
   - [Weight Metrics](#weight-metrics)
4. [Metric Catalogue Reference](#metric-catalogue-reference)
5. [Output](#output)
6. [Worked Example](#worked-example)
7. [Complexity](#complexity)
8. [Commands](#commands)
9. [What Comes Next](#what-comes-next)

---

## What This Step Does

Analysis takes the layer-projected dependency graph **G_analysis(l)** produced by Step 1 and computes **18 metrics** for every component — 13 topological metrics plus 5 code-quality attributes for Application nodes. Each metric captures a different dimension of structural importance — how central a component is, how many dependency paths flow through it, whether removing it would partition the graph, and how maintainable its internal structure is.

```
G_analysis(l)       StructuralAnalyzer      Metric Vectors
(vertices +     →   (13 algorithms)    →   M(v) for each component v
 DEPENDS_ON             │                  + graph-level summary S(G)
 edges)                 │
                  networkx algorithms
                  (PageRank, Betweenness,
                   Eigenvector, Tarjan, ...)
```

The 13 raw metrics are not criticality scores in themselves — they are structural observations. Step 3 (Prediction) combines them into the RMAV quality dimensions using AHP-derived weights to produce the final criticality prediction Q(v).

---

## Why Multiple Metrics?

No single metric captures all aspects of structural criticality. Consider two components in a typical pub-sub system:

- **Component A**: Many applications depend on it transitively (high PageRank), but it is well-connected in a redundant subgraph (low Betweenness, not an articulation point). It is broadly depended upon but not a bottleneck — a reliability concern, not a SPOF.
- **Component B**: Few components depend on it directly (low PageRank), but it is the only bridge between two clusters of the dependency graph (binary articulation point, Bridge Ratio = 1.0). It is structurally irreplaceable — an availability concern despite low PageRank.

Relying on a single metric would misclassify both. Thirteen independent metrics, each from a different theoretical family (random walk, shortest path, spectral, local topology, resilience), together provide a complete structural fingerprint.

---

## Formal Definitions

### Normalization

All raw metric values are normalized to [0, 1] before being consumed by Step 3.

```
x_norm(v) = (x(v) − min(x)) / (max(x) − min(x))

Edge case: If max(x) = min(x) → x_norm(v) = 0 for all v (no discriminating power)
```

Normalization is applied independently per metric and per layer, so scores are relative within each analysis context.

### PageRank

Captures *transitive importance* — how broadly a component is depended upon, accounting for the importance of its dependents.

```
PR(v) = (1 − d) / |V| + d × Σ_{u: (u,v)∈E} PR(u) / out_degree(u)

d = 0.85 (damping factor),  max iterations = 100,  tolerance = 1e-6
```

**High PR(v) means:** v is a highly depended-upon hub, either directly or transitively. Its failure starves a large fraction of the downstream dependency graph.

### Reverse PageRank

Computes PageRank on the **transposed graph G^T** (all edge directions reversed). Captures *cascade reach* — how far a failure at v propagates in the direction of its dependents.

```
RPR(v) = PageRank on G^T
```

**High RPR(v) means:** If v fails, the failure propagates broadly through the components that depend on it. RPR is the primary input to the Reliability dimension R(v) in Step 3.

### Betweenness Centrality

Measures what fraction of all shortest dependency paths pass through v. A component with high betweenness sits at the intersection of many communication paths — a structural bottleneck.

```
BT(v) = Σ_{s≠v≠t} σ(s, t | v) / σ(s, t)

σ(s, t)       = number of shortest paths from s to t
σ(s, t | v)   = number of those paths that pass through v
```

Computed using Brandes' algorithm: O(|V| × |E|). Normalized to [0, 1] by dividing by (|V|−1)(|V|−2).

**High BT(v) means:** v is a structural bottleneck — many dependency chains pass through it. Changes to v risk disrupting many other components. Primary input to the Maintainability dimension M(v).

### Closeness Centrality

Measures how quickly a component can "reach" all others in the dependency graph — a proxy for how fast information or failures propagate *from* v.

```
CL(v) = (|V| − 1) / Σ_{u≠v} d(v, u)     (Wasserman-Faust normalization for disconnected graphs)

d(v, u) = shortest path distance from v to u
```

**High CL(v) means:** v is topologically close to many others — faults or information originating at v spread quickly.

### Eigenvector Centrality

Measures whether a component is connected to *other important* hubs. A component scores high if its neighbours themselves have high importance.

```
EV(v) = (1/λ) Σ_{u ∈ N(v)} EV(u)     (power iteration, max 100 iterations)

λ = largest eigenvalue of the adjacency matrix
Fallback: in-degree if power iteration fails to converge
```

**High EV(v) means:** v is embedded in a high-value neighbourhood. Its compromise exposes a cluster of important components. Primary input to the Vulnerability dimension V(v).

### Degree Metrics

```
DG_in(v)  = in_degree(v)  / (|V| − 1)     (normalized)
DG_out(v) = out_degree(v) / (|V| − 1)     (normalized)
```

- **High DG_in(v):** Many components depend directly on v — immediate blast radius.
- **High DG_out(v):** v depends directly on many others — high efferent coupling, change fragility.

### Clustering Coefficient

Measures the density of connections among v's immediate neighbours. High clustering indicates local redundancy; low clustering indicates that each of v's connections is a unique, non-redundant path.

```
CC(v) = |{(u,w) ∈ E : u,w ∈ N(v)}| / (deg(v) × (deg(v) − 1))

Computed on undirected projection of G. If deg(v) < 2: CC(v) = 0.
```

**High CC(v) means:** v's neighbours are well-connected among themselves — the local topology is redundant, making v less of a bottleneck. Low CC contributes positively to the Maintainability dimension M(v) (via the `1 − CC` term), indicating that each coupling is a unique structural dependency.

### Articulation Point Score

Binary articulation point detection identifies whether removing v disconnects the graph. The **continuous score AP_c(v)** extends this to measure *how badly* the graph fragments.

```
Binary AP detection (Tarjan's DFS, O(|V| + |E|)):
  v is an articulation point if removing v increases the number of connected components

Continuous AP_c(v):
  Remove v from G; compute connected components C₁, C₂, ..., Cₖ
  AP_c(v) = 1 − max(|Cᵢ|) / (|V| − 1)

  AP_c(v) = 0    → v's removal does not fragment the graph
  AP_c(v) = 1    → v's removal isolates every other component
```

**High AP_c(v) means:** v is a structural single point of failure — its removal splits the dependency graph into fragments that can no longer communicate.

### Bridge Ratio

Measures what fraction of v's incident edges are **bridges** — edges whose removal disconnects the graph.

```
BR(v) = |{e ∈ bridges(G) : v ∈ e}| / degree_undirected(v)

If degree_undirected(v) = 0: BR(v) = 0
```

Bridge detection: `networkx.bridges()` — O(|V| + |E|).

**High BR(v) means:** Most of v's connections are non-redundant — there is no alternative path if any of these edges is severed. Combined with AP_c, this is the primary signal for Availability A(v).

### Weight Metrics

QoS-derived weight aggregates propagated from Step 1.

```
w(v)     = component's own QoS-derived importance weight (from Step 1)
w_in(v)  = Σ w(e) for incoming DEPENDS_ON edges  (normalized)
w_out(v) = Σ w(e) for outgoing DEPENDS_ON edges  (normalized)
```

- **High w(v):** v handles high-priority, reliable, or large-payload topics — intrinsically important regardless of structural position.
- **High w_in(v):** v's incoming dependencies carry high-criticality data streams.
- **High w_out(v):** v's outgoing dependencies are high-stakes data sources.

These weight metrics feed directly into Step 3's Prediction formulas (w_out → M(v), w(v) × AP_c → A(v) via QSPOF, w_in → V(v)).

### Code-Quality Metrics

Optional code-level attributes supplied on **Application nodes** in the topology JSON. All default to `0`/`0.0` when absent; non-Application nodes always receive `0.0`.

| Field | Symbol | Range | Definition |
|-------|--------|-------|-----------|
| Lines of Code (normalised) | `loc_norm` | [0,1] | Population min-max of `loc` across all Application nodes in the layer |
| Cyclomatic Complexity (normalised) | `complexity_norm` | [0,1] | Population min-max of `cyclomatic_complexity` |
| Martin Instability | `instability_code` | [0,1] | `Ce / (Ca + Ce)` where Ca = coupling_afferent, Ce = coupling_efferent |
| LCOM (normalised) | `lcom_norm` | [0,1] | Population min-max of `lcom` |
| **Code Quality Penalty** | **CQP** | [0,1] | `0.40·complexity_norm + 0.35·instability_code + 0.25·lcom_norm` |

CQP feeds the Maintainability dimension M(v) in Step 3 as a 5th criterion. Because `loc_norm` captures size risk separately from CQP, LOC is available for reporting/GNN features but is not included in CQP to avoid double-counting with complexity.

---

## Metric Catalogue Reference

All 18 metrics at a glance, with their formal symbols, the quality dimension they primarily contribute to in Step 3, and their computational source.

| # | Metric | Symbol | Category | RMAV Dimension | Implementation |
|---|--------|--------|----------|---------------|----------------|
| 1 | PageRank | PR(v) | Centrality | Reported only | `networkx.pagerank` |
| 2 | Reverse PageRank | RPR(v) | Centrality | R — Reliability | `networkx.pagerank(G^T)` |
| 3 | Betweenness Centrality | BT(v) | Centrality | M — Maintainability | `networkx.betweenness_centrality` |
| 4 | Closeness Centrality | CL(v) | Centrality | Reported only | `networkx.closeness_centrality` |
| 5 | Eigenvector Centrality | EV(v) | Centrality | Reported only | `networkx.eigenvector_centrality` |
| 6 | In-Degree | DG_in(v) | Degree | R — Reliability | `G.in_degree` |
| 7 | Out-Degree | DG_out(v) | Degree | M — Maintainability | `G.out_degree` |
| 8 | Clustering Coefficient | CC(v) | Local topology | M — Maintainability | `networkx.clustering` |
| 9 | Articulation Point Score | AP_c(v) | Resilience | A — Availability | Tarjan DFS + reachability |
| 10 | Bridge Ratio | BR(v) | Resilience | A — Availability | `networkx.bridges` |
| 11 | Component Weight | w(v) | QoS | A — Availability | From Step 1 QoS |
| 12 | Weighted In-Degree | w_in(v) | QoS | V — Vulnerability | Σ incident edge weights |
| 13 | Weighted Out-Degree | w_out(v) | QoS | M — Maintainability | Σ incident edge weights |
| 14 | LOC (normalised) | loc_norm | Code Quality | GNN features | Population min-max |
| 15 | Cyclomatic Complexity | complexity_norm | Code Quality | M — Maintainability (via CQP) | Population min-max |
| 16 | Martin Instability | instability_code | Code Quality | M — Maintainability (via CQP) | Ce/(Ca+Ce) |
| 17 | LCOM (normalised) | lcom_norm | Code Quality | M — Maintainability (via CQP) | Population min-max |
| 18 | Code Quality Penalty | CQP | Code Quality | M — Maintainability | 0.40·CC + 0.35·I + 0.25·LCOM |

> **Note on DG_out:** Out-degree contributes to Maintainability (efferent coupling in change risk context) only. The CL and EV metrics feed the GNN node features but are not used directly in RMAV formulas; REV and RCL (computed on G^T in Step 3) are the active Vulnerability signals. Metrics 14–18 are 0.0 for all non-Application node types.

---

## Output

### Per-Component Metric Vector

For each component v in the selected layer:

```
M(v) = (PR(v), RPR(v), BT(v), CL(v), EV(v),
        DG_in(v), DG_out(v), CC(v), AP_c(v), BR(v),
        w(v), w_in(v), w_out(v),
        loc_norm(v), complexity_norm(v), instability_code(v), lcom_norm(v), CQP(v))
```

All values are in [0, 1]. Metrics 14–18 are 0.0 for non-Application component types. The `is_articulation_point` boolean flag is additionally reported alongside AP_c(v) for binary filtering (e.g., instant SPOF detection).

### Graph-Level Summary S(G)

```json
{
  "layer": "app",
  "component_count": 35,
  "edge_count": 87,
  "density": 0.073,
  "avg_clustering": 0.21,
  "articulation_point_count": 3,
  "bridge_count": 11,
  "spof_count": 3
}
```

### JSON Output Structure

```json
{
  "layer": "app",
  "summary": { "total": 35, "articulation_points": 3, "bridges": 11 },
  "components": {
    "DataRouter": {
      "pagerank":              0.88,
      "reverse_pagerank":      0.76,
      "betweenness":           0.79,
      "closeness":             0.65,
      "eigenvector":           0.72,
      "in_degree":             0.75,
      "out_degree":            0.23,
      "clustering_coefficient":0.15,
      "ap_score":              0.62,
      "is_articulation_point": true,
      "bridge_ratio":          0.83,
      "weight":                0.92,
      "weighted_in_degree":    0.80,
      "weighted_out_degree":   0.31
    }
  }
}
```

---

## Worked Example

**PLC_Controller (A3)** from the Distributed Intelligent Factory (DIF) scenario — application layer, 32 components:

```
Raw values (before normalization):
  PageRank:              0.045  (mid-range — not the top PageRank hub)
  Reverse PageRank:      0.082  (high — failure propagates to many downstream)
  Betweenness:           0.310  (very high — lies on many critical paths)
  Closeness:             0.610  (high — central in the dependency graph)
  Eigenvector:           0.540  (high — connected to important neighbours)
  In-Degree:             6      (6 components directly depend on A3)
  Out-Degree:            5      (A3 depends on 5 components)
  Clustering Coeff.:     0.18   (low — connections are not well-interconnected)
  AP_c (undirected):     0.43   (43% of the graph fragments without A3)
  Bridge Ratio:          0.67   (2/3 of A3's connections are bridges)
  w(v):                  0.92   (routes a PERSISTENT+URGENT+RELIABLE topic)
  w_in:                  0.88   (dependents carry high-priority data streams)
  w_out:                 0.74   (dependencies are high-priority sources)

After min-max normalization across all 32 components:
  RPR = 0.60,  BT = 0.95,  DG_in = 0.75,  AP_c = 0.43,  BR = 1.00
  w   = 0.92,  w_in = 0.88, w_out = 0.68
```

A3's normalized betweenness of 0.95 immediately identifies it as the structural bottleneck of the application layer. The combination of high AP_c (0.43) and Bridge Ratio = 1.0 further marks it as a structural SPOF. These three signals together will drive CRITICAL-level predictions across the M(v) and A(v) RMAV dimensions in Step 3.

---

## Complexity

| Algorithm | Complexity | Notes |
|-----------|------------|-------|
| PageRank / RPR | O(I × \|E\|) | I = iterations (≤100); sparse power iteration |
| Betweenness Centrality | O(\|V\| × \|E\|) | Brandes' algorithm |
| Closeness Centrality | O(\|V\| × (\|V\| + \|E\|)) | BFS from each vertex |
| Eigenvector Centrality | O(I × \|E\|) | Power iteration; falls back to in-degree |
| Articulation Points | O(\|V\| + \|E\|) | Tarjan's DFS |
| AP_c (continuous) | O(\|V\| × (\|V\| + \|E\|)) | Repeated reachability computation |
| Bridge Detection | O(\|V\| + \|E\|) | DFS-based |
| Min-Max Normalization | O(\|V\|) | Per metric |

**Overall:** O(|V|² + |V|×|E|) dominated by AP_c and Closeness. An `xlarge` system (200 components, ~600 edges) takes approximately 20 seconds.

---

## Commands

```bash
# Analyze the application layer (recommended first)
python bin/analyze_graph.py --layer app

# Analyze all layers
python bin/analyze_graph.py --layer system

# Export metric vectors to JSON for inspection or post-processing
python bin/analyze_graph.py --layer system --output results/metrics.json

# Run with AHP-derived weights (affects prediction scoring in Step 3, not metric computation here)
python bin/analyze_graph.py --layer system --use-ahp

# Run sensitivity analysis on AHP weights (requires --use-ahp)
python bin/analyze_graph.py --layer system --use-ahp --sensitivity
# Reports: Top-5 Stability, Mean Kendall τ across 200 weight perturbations (σ=0.05)

# Specify Neo4j connection explicitly
python bin/analyze_graph.py --layer app --uri bolt://localhost:7687 --user neo4j --password secret
```

### Interpreting the Output

```
Layer: app | 35 components | 87 edges | density: 0.073
Articulation points: 3  |  Bridges: 11  |  SPOFs detected: 3

Top Critical Components (by Q(v)):
  1. DataRouter      [CRITICAL]  Q=0.91  RPR=0.76  AP_c=0.62  BT=0.79
  2. SensorHub       [CRITICAL]  Q=0.87  RPR=0.71  AP_c=0.50  BT=0.71
  3. CommandGateway  [HIGH]      Q=0.74  RPR=0.48  AP_c=0.00  BT=0.83
```

- **CRITICAL** components with non-zero AP_c are structural SPOFs — top priority for redundancy.
- **CRITICAL** components with high BT but AP_c = 0 are bottlenecks but not SPOFs — consider decoupling or load balancing.
- A high **bridge count** relative to edge count indicates a brittle topology with few redundant paths.

---

## What Comes Next

Step 2 produces a metric vector M(v) for every component — 13 numbers per component, each capturing a different structural property. These numbers are precise but not directly actionable: "betweenness = 0.79" does not tell an architect what to do.

Step 3 (Prediction) maps these raw metrics into four interpretable quality dimensions — Reliability, Maintainability, Availability, Vulnerability — using AHP-derived weights. The output is a single criticality classification per component (CRITICAL, HIGH, MEDIUM, LOW, or MINIMAL), backed by a quantitative score and decomposable by dimension to explain *why* a component is critical. For systems where labelled training data exists, a Graph Neural Network predictor can learn improved metric combinations alongside the rule-based RMAV scorer.

---

← [Step 1: Modeling](graph-model.md) | → [Step 3: Prediction](prediction.md)