# Step 2: Structural Analysis

**Measure each component's structural importance using graph-theoretic metrics.**

← [Step 1: Graph Model](graph-model.md) | → [Step 3: Quality Scoring](quality-scoring.md)

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

Structural Analysis takes the layer-projected dependency graph **G_analysis(l)** produced by Step 1 and computes 13 topological metrics for every component. Each metric captures a different dimension of structural importance — how central a component is, how many dependency paths flow through it, whether removing it would partition the graph.

```
G_analysis(l)       StructuralAnalyzer      Metric Vectors
(vertices +     →   (13 algorithms)    →   M(v) for each component v
 DEPENDS_ON             │                  + graph-level summary S(G)
 edges)                 │
                  networkx algorithms
                  (PageRank, Betweenness,
                   Eigenvector, Tarjan, ...)
```

The 13 raw metrics are not criticality scores in themselves — they are structural observations. Step 3 combines them into the RMAV quality dimensions using AHP-derived weights to produce the final criticality prediction Q(v).

---

## Why Multiple Metrics?

No single metric captures all aspects of structural criticality. Consider two components in a typical pub-sub system:

- **Component A**: Many applications depend on it transitively (high PageRank), but it is well-connected in a redundant subgraph (low Betweenness, not an articulation point). It is broadly depended upon but not a bottleneck — a reliability concern, not a SPOF.
- **Component B**: Few components depend on it directly (low PageRank), but it is the only bridge between two clusters of the dependency graph (binary articulation point, Bridge Ratio = 1.0). It is structurally irreplaceable — an availability concern despite low PageRank.

Relying on a single metric would misclassify both. Thirteen independent metrics, each from a different theoretical family (random walk, shortest path, spectral, local topology, resilience), together provide a complete structural fingerprint. The metric-to-RMAV mapping in Step 3 then interprets each metric through the lens of the quality dimension it most directly predicts.

---

## Formal Definitions

All metrics are computed on the **G_analysis(l)** directed graph of DEPENDS_ON edges. For resilience metrics (articulation point, bridge ratio), the undirected version of the graph is used, since connectivity loss is symmetric. All continuous metrics are normalized to [0, 1] after computation using [min-max scaling](#normalization).

### Normalization

All continuous metrics use min-max normalization across the component set:

```
metric_norm(v) = (metric(v) − min_u metric(u)) / (max_u metric(u) − min_u metric(u))
```

When all values are equal (max = min), all normalized values are set to 0. Integer degree metrics (in-degree, out-degree) are also normalized in this way before entering RMAV formulas.

### PageRank

PageRank measures transitive importance via incoming DEPENDS_ON edges. A component with high PageRank is one that many other components depend upon — directly or through chains of dependencies.

```
PR(v) = (1−d)/|V| + d × Σ  PR(u) / out_degree(u)
                       u ∈ N⁻(v)
```

where `d = 0.85` (damping factor), `N⁻(v)` is the set of in-neighbors of v, and the equation is iterated to convergence (tolerance 1×10⁻⁶, max 100 iterations). The damping factor models the probability that a random walker following dependency edges continues rather than jumping to a random node.

**High PR(v) means:** Many components depend on v, directly or transitively. If v fails, the failure propagates widely through downstream dependents.

Delegated to `networkx.pagerank(G, alpha=0.85)`.

### Reverse PageRank

Reverse PageRank (RPR) is PageRank computed on the **transposed** graph G^T (all edges reversed). In G^T, an edge A→B from G becomes B→A, so RPR accumulates score from *outgoing* dependency paths in the original graph.

```
RPR(v) = PageRank(v, graph=G^T)
```

**High RPR(v) means:** v sits at the head of long dependency chains — it is a *source* from which failures can propagate downstream. While PageRank identifies components that are *depended upon*, Reverse PageRank identifies components from which *failure propagates outward*.

The distinction matters for failure propagation direction. In the RMAV framework, both PR and RPR contribute to Reliability R(v) because fault propagation depends on both how much depends on v (PR) and how many things v's failure can cascade to (RPR).

### Betweenness Centrality

Betweenness centrality measures how often a component lies on the shortest dependency path between other pairs of components. A high-betweenness component is a structural **bottleneck**: removing it forces many dependency paths to take longer routes or break entirely.

```
BT(v) = Σ  σ(s,t | v) / σ(s,t)
       s≠v≠t
```

where σ(s,t) is the total number of shortest paths from s to t, and σ(s,t | v) is the number of those paths that pass through v. Normalized by dividing by (|V|−1)(|V|−2) for directed graphs.

Computed using Brandes' algorithm (O(|V|×|E|)). Delegated to `networkx.betweenness_centrality(G)`.

**High BT(v) means:** v is a structural bottleneck. Removing it disrupts many dependency paths and degrades system connectivity.

### Closeness Centrality

Closeness centrality measures how quickly a component can reach — or be reached by — all others. A component with high closeness occupies a central position in the dependency topology.

For directed graphs with potentially unreachable vertex pairs, the **Wasserman-Faust** normalization is used:

```
CL(v) = (|R(v)| / (|V| − 1)) × (|R(v)| / Σ dist(v, u))
                                              u ∈ R(v)
```

where `R(v)` is the set of vertices reachable from v, and `dist(v, u)` is the shortest path length. The leading factor (|R(v)|/(|V|−1)) penalizes components that can reach only a small fraction of the graph.

Delegated to `networkx.closeness_centrality(G)`.

**High CL(v) means:** v can reach most of the graph quickly. Compromise of v (e.g., by injection of malicious messages) could propagate to many components with few hops.

### Eigenvector Centrality

Eigenvector centrality measures a component's connection to other highly-connected components — not just the count of neighbors, but the *quality* of those connections.

```
EV(v) = (1/λ) × Σ  EV(u)
                u ∈ N⁻(v)
```

where λ is the largest eigenvalue of the adjacency matrix. This means being connected to one high-EV hub contributes more than being connected to many low-EV periphery nodes.

Delegated to `networkx.eigenvector_centrality(G, max_iter=1000)`. If the algorithm does not converge (possible for certain directed graph structures), the result falls back to in-degree centrality with a warning.

**High EV(v) means:** v is connected to strategically important hubs. It is embedded in the most influential part of the dependency network — a high-value target.

### Degree Metrics

Raw degree counts, normalized to [0, 1] using min-max scaling.

```
DG_in(v)  = |{u : u → v  ∈ E}|   (number of incoming DEPENDS_ON edges)
DG_out(v) = |{u : v → u  ∈ E}|   (number of outgoing DEPENDS_ON edges)
```

**High DG_in(v) means:** Many components directly depend on v. Equivalent to immediate fan-out of v's failure.

**High DG_out(v) means:** v depends on many other components (high efferent coupling). In quality scoring, this indicates both maintainability risk (harder to change without breaking dependencies) and vulnerability (more outbound paths for an attacker to exploit).

### Clustering Coefficient

The clustering coefficient measures how interconnected a component's neighbors are. For a directed graph, the local clustering coefficient is computed as:

```
CC(v) = |{(u,w) : u→w ∈ E, u ∈ N(v), w ∈ N(v)}| / (k(v) × (k(v) − 1))
```

where `N(v)` is the union of in- and out-neighbors of v, and `k(v) = |N(v)|`. The numerator counts actual directed edges among v's neighbors; the denominator counts maximum possible directed edges.

Delegated to `networkx.clustering(G.to_undirected())` (undirected, as is standard for topological analysis).

**High CC(v) means:** v's neighbors are densely interconnected. This suggests a well-connected local cluster with redundant paths — *lower* structural risk in isolation. In Maintainability scoring, (1 − CC(v)) is used: a component embedded in a sparse neighborhood (low clustering) has less redundancy and is harder to safely refactor.

**CC(v) = 0** when v has fewer than 2 neighbors or when no edges exist between neighbors.

### Articulation Point Score

A traditional articulation point (AP) detection returns a binary result: v is or is not an articulation point of the undirected graph. This binary classification creates a sharp discontinuity — a component that would only disconnect a single leaf node upon removal gets the same score as one that would split the graph in half.

To address this, we use a **continuous fragmentation score**:

```
AP_c(v) = 1 − |largest connected component of G \ {v}| / (|V| − 1)
```

where `G \ {v}` is the graph with vertex v and all its edges removed.

Interpretation:
- v is **not** an articulation point → AP_c(v) = 0 (graph remains connected)
- v's removal disconnects one small leaf (2 nodes, rest connected) → AP_c(v) ≈ 1/(|V|−1) ≈ 0
- v's removal splits the graph roughly in half → AP_c(v) ≈ 0.5
- v's removal isolates all but one remaining vertex → AP_c(v) → 1.0

This requires running a connected-component check for each vertex: **O(|V| × (|V| + |E|))**. The binary articulation point set (used for reporting) is computed once via Tarjan's DFS algorithm in O(|V| + |E|).

**High AP_c(v) means:** Removing v would cause severe graph fragmentation — v is a structural single point of failure (SPOF). This is the strongest individual predictor of availability risk.

### Bridge Ratio

A **bridge** is an edge whose removal disconnects the graph (the edge equivalent of an articulation point). The Bridge Ratio measures what fraction of a component's incident edges are bridges:

```
BR(v) = |{e ∈ E : e is a bridge, v ∈ e}| / degree(v)
```

where `degree(v)` is the total undirected degree of v. If `degree(v) = 0`, then `BR(v) = 0`.

Bridge detection uses DFS-based identification in O(|V| + |E|), computed on the undirected graph. Delegated to `networkx.bridges()`.

**High BR(v) means:** Most of v's connections are irreplaceable — they are bridges whose removal would partition the graph. Even if v itself is not an articulation point, its edges may be critical load-bearing connections in the dependency topology.

**BR(v) = 1.0** means every edge incident to v is a bridge — v is connected to the rest of the graph through a "spine" of single-point connections.

### Weight Metrics

Weight metrics incorporate the QoS-derived edge weights from Step 1:

```
w(v)     = intrinsic component weight (from QoS of handled topics)
w_in(v)  = Σ  w(u → v)    (sum of weights of incoming DEPENDS_ON edges)
           u
w_out(v) = Σ  w(v → u)    (sum of weights of outgoing DEPENDS_ON edges)
           u
```

All three are normalized to [0, 1] using min-max scaling across the component set.

**High w(v) means:** v handles high-priority, reliable, or large-payload topics. It is intrinsically important regardless of its structural position.

**High w_in(v) means:** v's incoming dependencies carry high-criticality data streams — the components that depend on v do so for mission-critical communication.

**High w_out(v) means:** v's outgoing dependencies are high-stakes — the components v depends on are mission-critical data sources.

These weight metrics are not currently included in the RMAV formulas (which use purely topological metrics), but they are reported in the output for analyst inspection and inform the component-level weight stored in Step 1.

---

## Metric Catalogue Reference

All 13 metrics at a glance, with their formal symbols, the quality dimension they primarily contribute to in Step 3, and their computational source.

| # | Metric | Symbol | Category | RMAV Dimension | Implementation |
|---|--------|--------|----------|---------------|----------------|
| 1 | PageRank | PR(v) | Centrality | R — Reliability | `networkx.pagerank` |
| 2 | Reverse PageRank | RPR(v) | Centrality | R — Reliability | `networkx.pagerank(G^T)` |
| 3 | Betweenness Centrality | BT(v) | Centrality | M — Maintainability | `networkx.betweenness_centrality` |
| 4 | Closeness Centrality | CL(v) | Centrality | V — Vulnerability | `networkx.closeness_centrality` |
| 5 | Eigenvector Centrality | EV(v) | Centrality | V — Vulnerability | `networkx.eigenvector_centrality` |
| 6 | In-Degree | DG_in(v) | Degree | R — Reliability | `G.in_degree` |
| 7 | Out-Degree | DG_out(v) | Degree | M, V | `G.out_degree` |
| 8 | Clustering Coefficient | CC(v) | Local topology | M — Maintainability | `networkx.clustering` |
| 9 | Articulation Point Score | AP_c(v) | Resilience | A — Availability | Tarjan DFS + reachability |
| 10 | Bridge Ratio | BR(v) | Resilience | A — Availability | `networkx.bridges` |
| 11 | Component Weight | w(v) | QoS | (reported only) | From Step 1 QoS |
| 12 | Weighted In-Degree | w_in(v) | QoS | (reported only) | Σ incident edge weights |
| 13 | Weighted Out-Degree | w_out(v) | QoS | (reported only) | Σ incident edge weights |

Note that Out-Degree (DG_out) contributes to both Maintainability M(v) and Vulnerability V(v) with distinct semantic rationale: efferent coupling in the context of change risk (M), and attack surface in the context of security exposure (V). It is the only metric shared between two RMAV dimensions.

---

## Output

### Per-Component Metric Vector

For each component v in the selected layer, the analysis produces a 13-element metric vector:

```
M(v) = (PR(v), RPR(v), BT(v), CL(v), EV(v),
        DG_in(v), DG_out(v), CC(v), AP_c(v), BR(v),
        w(v), w_in(v), w_out(v))
```

All values are in [0, 1]. The `is_articulation_point` boolean flag is additionally reported alongside AP_c(v) for binary filtering.

### Graph-Level Summary

In addition to per-component metrics, the analysis reports a summary **S(G)** for the entire layer graph:

| Field | Description |
|-------|-------------|
| `vertex_count` | Total number of vertices in G_analysis(l) |
| `edge_count` | Total number of DEPENDS_ON edges |
| `density` | Edge count / (|V| × (|V|−1)) — fraction of possible directed edges present |
| `avg_clustering` | Mean clustering coefficient across all vertices |
| `weakly_connected_components` | Number of weakly connected components |
| `strongly_connected_components` | Number of strongly connected components |
| `diameter` | Longest shortest path (−1 if graph is not strongly connected) |
| `articulation_point_count` | Number of vertices identified as binary articulation points |
| `bridge_count` | Number of edges identified as bridges |
| `avg_in_degree` | Mean number of incoming DEPENDS_ON edges per vertex |
| `avg_out_degree` | Mean number of outgoing DEPENDS_ON edges per vertex |

### JSON Output Schema

```json
{
  "layer": "app",
  "graph_summary": {
    "vertex_count": 35,
    "edge_count": 87,
    "density": 0.073,
    "avg_clustering": 0.21,
    "weakly_connected_components": 1,
    "strongly_connected_components": 8,
    "diameter": -1,
    "articulation_point_count": 3,
    "bridge_count": 11,
    "avg_in_degree": 2.49,
    "avg_out_degree": 2.49
  },
  "components": {
    "SensorApp": {
      "pagerank": 0.82,
      "reverse_pagerank": 0.41,
      "betweenness": 0.67,
      "closeness": 0.55,
      "eigenvector": 0.73,
      "in_degree": 0.60,
      "out_degree": 0.20,
      "clustering_coefficient": 0.15,
      "ap_score": 0.50,
      "is_articulation_point": true,
      "bridge_ratio": 0.75,
      "weight": 1.00,
      "weighted_in_degree": 0.88,
      "weighted_out_degree": 0.22
    }
  }
}
```

---

## Worked Example

This section traces the 6-component system from the Step 1 worked example through structural analysis to show which metrics fire for which components.

**Graph** (application layer only, DEPENDS_ON edges):

```
SensorApp ←── MonitorApp
    ↑               ↑
MainBroker ─────────┘
```

In DEPENDS_ON notation:
- MonitorApp → SensorApp  (app_to_app: subscriber depends on publisher)
- MonitorApp → MainBroker (app_to_broker: subscriber depends on routing broker)
- SensorApp  → MainBroker (app_to_broker: publisher depends on routing broker)

This is a simple 3-vertex, 3-edge directed graph. Let G = {S=SensorApp, M=MonitorApp, B=MainBroker}.

### PageRank

Following the random walk on DEPENDS_ON edges (direction: edges point toward dependencies):

- **MainBroker (B)** receives score from both S and M → highest PR
- **SensorApp (S)** receives score from M → medium PR
- **MonitorApp (M)** receives no incoming edges → lowest PR

PR order: B > S > M (B most depended upon, M depends on others but nothing depends on it)

### Reverse PageRank

On the transposed graph (edges reversed — pointing from dependency back to dependent):

- **MonitorApp (M)** has incoming edges from both B and S in G^T → highest RPR
- S and B have one incoming edge each → lower RPR

RPR order: M > S ≈ B (M's failure propagates to nobody — it's a sink; but M *depends on* both S and B, so in G^T, M accumulates score)

> Note: High RPR means "this component depends on many things, and its failure affects it widely" — not a source of cascades but a component whose own operation is most exposed to upstream failures.

### Betweenness

The only path between M and B passes through... itself (M → B is direct). S to M: no path in G. No vertex acts as a bridge between disconnected pairs, so **BT is low for all vertices** in this tiny example. In larger systems, a broker that sits between multiple isolated application clusters would have very high BT.

### Articulation Point Score

Removing MainBroker (B) from the 3-vertex graph leaves {S, M} with the edge M→S — still connected (weakly). AP_c(B) = 1 − 2/2 = 0. No vertex is a true articulation point in this small example.

In a more realistic system: a broker that is the sole middleware between two application clusters scores AP_c ≈ 0.5 or higher.

### Bridge Ratio

The edge M→B is a bridge (removing it disconnects M from B; no alternative path exists). The edge S→B is also a bridge. M→S is a bridge too.

- SensorApp (S): 2 incident edges (M→S incoming, S→B outgoing), both bridges → BR(S) = 1.0
- MainBroker (B): 2 incident edges (S→B, M→B), both bridges → BR(B) = 1.0
- MonitorApp (M): 2 incident edges (M→S, M→B), both bridges → BR(M) = 1.0

All edges are bridges in a tree-like topology. In larger systems with cycles, components embedded in cycles have BR < 1.0 because the cycle provides an alternative path.

### Summary Table for This Example

| Metric | SensorApp | MonitorApp | MainBroker |
|--------|-----------|------------|------------|
| PR (norm) | 0.50 | 0.00 | 1.00 |
| RPR (norm) | 0.50 | 1.00 | 0.00 |
| BT (norm) | 0.00 | 0.00 | 0.00 |
| AP_c | 0.00 | 0.00 | 0.00 |
| BR | 1.00 | 1.00 | 1.00 |

**Interpretation:** MainBroker is the most depended-upon component (highest PR). MonitorApp is the most exposed to upstream failure (highest RPR). All Bridge Ratios are 1.0 — this chain-topology system has no redundant connections. In Step 3, these metrics will classify MainBroker as having the highest Reliability risk and all components as high Availability risk (high BR).

---

## Complexity

| Algorithm | Complexity | Notes |
|-----------|------------|-------|
| PageRank | O((|V|+|E|) × iterations) | ~20–50 iterations typical; max 100 |
| Reverse PageRank | O((|V|+|E|) × iterations) | Same, on transposed graph |
| Betweenness Centrality | O(|V| × |E|) | Brandes algorithm; dominant cost for dense graphs |
| Closeness Centrality | O(|V| × (|V|+|E|)) | BFS from each vertex |
| Eigenvector Centrality | O((|V|+|E|) × iterations) | ~100 iterations; fallback if no convergence |
| In/Out-Degree | O(|V|+|E|) | Single pass |
| Clustering Coefficient | O(|V| × d²) | d = average degree |
| Articulation Point Detection | O(|V|+|E|) | Tarjan DFS (binary) |
| AP_c Score | O(|V| × (|V|+|E|)) | Connected-component check per vertex |
| Bridge Detection | O(|V|+|E|) | DFS-based |
| Bridge Ratio | O(|V|+|E|) | Per vertex from bridge set |
| **Total (full system layer)** | **O(|V| × |E|)** | Dominated by Betweenness and AP_c |

For reference: a `medium` scale system (35 apps, 87 edges) completes full structural analysis in approximately 2 seconds on a standard laptop. An `xlarge` system (200 components, ~600 edges) takes approximately 20 seconds.

---

## Commands

```bash
# Analyze the application layer (recommended first)
python bin/analyze_graph.py --layer app

# Analyze all layers
python bin/analyze_graph.py --layer system

# Export metric vectors to JSON for inspection or post-processing
python bin/analyze_graph.py --layer system --output results/metrics.json

# Run with AHP-derived weights (affects quality scoring in Step 3, not metric computation)
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
  1. DataRouter      [CRITICAL]  Q=0.91  PR=0.88  AP_c=0.62  BT=0.79
  2. SensorHub       [CRITICAL]  Q=0.87  PR=0.82  AP_c=0.50  BT=0.71
  3. CommandGateway  [HIGH]      Q=0.74  PR=0.51  AP_c=0.00  BT=0.83
```

- **CRITICAL** components with non-zero AP_c are structural SPOFs — top priority for redundancy.
- **CRITICAL** components with high BT but AP_c = 0 are bottlenecks but not SPOFs — consider decoupling or load balancing.
- A high **bridge count** relative to edge count indicates a brittle topology with few redundant paths.

---

## What Comes Next

Step 2 produces a metric vector M(v) for every component — 13 numbers per component, each capturing a different structural property. These numbers are precise but not directly actionable: "betweenness = 0.79" doesn't tell an architect what to do.

Step 3 maps these raw metrics into four interpretable quality dimensions (Reliability, Maintainability, Availability, Vulnerability) using AHP-derived weights. The output is a single criticality classification per component — CRITICAL, HIGH, MEDIUM, LOW, or MINIMAL — backed by a quantitative score and decomposable by dimension to explain *why* a component is critical.

---

← [Step 1: Graph Model](graph-model.md) | → [Step 3: Quality Scoring](quality-scoring.md)