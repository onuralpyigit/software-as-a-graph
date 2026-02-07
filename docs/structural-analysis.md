# Step 2: Structural Analysis

**Compute topological metrics that capture different aspects of component criticality within layer-projected dependency subgraphs.**

---

## 2.1 Overview

Structural Analysis is the second step of the six-step methodology. It takes the
layer-projected analysis graph G_analysis(l) from Step 1 and computes a comprehensive
metric vector for each component, quantifying its structural importance from multiple
independent perspectives.

```
┌─────────────────────┐                              ┌─────────────────────┐
│  G_analysis(l)      │                              │  Metric Vectors     │
│                     │                              │                     │
│  - Vertices (V_l)   │   ┌──────────────────────┐   │  For each v ∈ V_l:  │
│  - DEPENDS_ON edges │──▶│  Structural Analyzer  │──▶│  M(v) ∈ [0,1]^k    │
│  - Edge weights     │   │                      │   │                     │
│  - Component types  │   │  1. Centrality        │   │  + Edge metrics     │
│                     │   │  2. Degree            │   │  + Graph summary    │
│                     │   │  3. Resilience        │   │                     │
└─────────────────────┘   └──────────────────────┘   └─────────────────────┘
```

The output of this step feeds directly into Step 3 (Quality Scoring), where raw
structural metrics are mapped to RMAV quality dimensions using AHP-derived weights.

---

## 2.2 Formal Definition

### Definition 3: Structural Analysis Input

Structural Analysis operates on the layer-projected analysis graph produced by Step 1:

```
Input:   G_analysis(l) = (V_l, E_l, w)
         where l ∈ { app, infra, mw, system }
         E_l contains only DEPENDS_ON edges filtered by layer projection π_l
         w : V_l ∪ E_l → ℝ⁺ (weight function from Step 1)
```

### Definition 4: Metric Vector

For each component v ∈ V_l, the analysis produces a metric vector:

```
M(v) = ( PR(v), RPR(v), BT(v), CL(v), EV(v),
         DG_in(v), DG_out(v),
         CC(v), AP(v), BR(v),
         w(v), w_in(v), w_out(v) )
```

where all continuous metrics are normalized to [0, 1]:

```
Output:  M = { M(v) : v ∈ V_l },  M(v) ∈ [0, 1]^k
         E_M = { E_M(e) : e ∈ E_l }  (edge metric vectors)
         S = GraphSummary  (graph-level statistics)
```

---

## 2.3 Metric Catalogue

The analysis computes 13 component-level metrics organized into three categories.
Each metric captures a distinct aspect of structural importance.

### 2.3.1 Centrality Metrics

Centrality metrics quantify a component's importance relative to the overall graph topology.

| # | Metric | Symbol | Graph | Weight | Complexity | Captures |
|---|--------|--------|-------|--------|------------|----------|
| 1 | PageRank | PR | Directed | Importance | O(\|V\|+\|E\|) per iter | Transitive influence |
| 2 | Reverse PageRank | RPR | Directed (reversed) | Importance | O(\|V\|+\|E\|) per iter | Failure propagation |
| 3 | Betweenness | BT | Directed | Distance (inverted) | O(\|V\|·\|E\|) | Bottleneck position |
| 4 | Harmonic Closeness | CL | Directed | — | O(\|V\|·\|E\|) | Reachability speed |
| 5 | Eigenvector / Katz | EV | Directed | Importance | O(\|V\|²) | Strategic importance |

#### PageRank (PR)

Measures transitive influence — how important are the components that depend on this one.

```
PR(v) = (1 - d) / |V| + d × Σ PR(u) / out(u)
                              u ∈ in_neighbors(v)
```

Where d = 0.85 (damping factor). PageRank is computed on the **directed** DEPENDS_ON graph.
Edge weights are interpreted as **importance** (higher weight = stronger dependency = more
influence transferred).

**Interpretation**: High PR indicates a component that is depended upon — directly or
transitively — by many other important components. In a pub-sub system, a shared data
processing node that multiple subscribers depend on will have high PR.

**Reference**: Brin & Page (1998). The Anatomy of a Large-Scale Hypertextual Web Search Engine.

#### Reverse PageRank (RPR)

PageRank computed on the **reversed** graph G^R, where all edge directions are flipped.

```
RPR(v) = PR_{G^R}(v)
```

**Interpretation**: High RPR indicates a component whose failure would cascade to many
important downstream components. While PR measures "who depends on me", RPR measures
"what do I depend on" — a component that depends on many critical services has high RPR,
making it a propagation vector for failures originating at its dependencies.

**Rationale for inclusion alongside PR**: PR and RPR capture complementary failure
dynamics. A leaf subscriber has high RPR (depends on many) but low PR (nothing depends
on it). A shared library has high PR (many depend on it) but potentially low RPR.

#### Betweenness Centrality (BT)

Fraction of all shortest paths that pass through a component.

```
BT(v) = Σ σ(s,t|v) / σ(s,t)      for all s ≠ v ≠ t ∈ V
```

Where σ(s,t) is the number of shortest paths from s to t, and σ(s,t|v) is the number
that pass through v. Computed using Brandes' algorithm (2001) on the **directed** graph.

**Weight semantics — distance inversion**: NetworkX's betweenness centrality interprets
edge weights as **distances** (higher weight = longer path = less preferred for shortest
paths). Since our DEPENDS_ON weights represent dependency **strength** (higher = more
important), we invert weights before computing betweenness:

```
w_distance(e) = 1 / w_importance(e)
```

This ensures that strongly-weighted dependencies are treated as "closer" (preferred
for shortest paths), so nodes that lie on paths between strongly-coupled components
receive higher betweenness scores.

**Interpretation**: High BT identifies bottlenecks and communication hubs — components
that many dependency chains must pass through. These are structurally critical because
their failure disrupts many communication paths simultaneously.

**Reference**: Brandes (2001). A Faster Algorithm for Betweenness Centrality.

#### Harmonic Closeness Centrality (CL)

Harmonic mean of distances from a component to all other reachable components.

```
CL_H(v) = (1 / (|V| - 1)) × Σ  1 / d(v, u)      for all u ≠ v ∈ V
```

Where d(v, u) is the shortest path distance from v to u. If u is unreachable, the
term contributes 0 (since 1/∞ = 0).

**Rationale for harmonic over classical closeness**: Classical closeness CL(v) = (n-1) / Σd(v,u)
is undefined for disconnected graphs because Σd(v,u) = ∞ when any node is unreachable.
The Wasserman-Faust variant scales by component size, but harmonic closeness is
mathematically cleaner — it naturally handles infinite distances and doesn't require
special-casing for disconnected components. Following Boldi & Vigna (2014), we adopt
harmonic closeness as the standard.

Computed on the **directed** graph without weight parameters, measuring topological
proximity rather than weighted distance.

**Interpretation**: High CL indicates components that can quickly reach (or be reached by)
many others. In failure propagation terms, these components have the widest "blast radius."

**Reference**: Boldi & Vigna (2014). Axioms for Centrality. Internet Mathematics.

#### Eigenvector Centrality (EV) with Katz Fallback

Measures influence via connection to other highly-connected nodes.

```
EV(v) = (1/λ) × Σ EV(u)      for all u ∈ neighbors(v)
```

Where λ is the largest eigenvalue of the adjacency matrix. Unlike PageRank, eigenvector
centrality uses no damping factor — it purely measures recursive influence.

**Convergence issue on DAGs**: Eigenvector centrality requires the existence of a dominant
eigenvalue, which is not guaranteed for directed acyclic graphs (DAGs). Many dependency
graphs approximate DAGs, causing the power iteration to fail to converge. When this occurs,
we fall back to **Katz centrality**:

```
Katz(v) = α × Σ A^k × 1      (matrix form: x = α(A^T)x + β·1)
               k=1..∞
```

With attenuation factor α < 1/λ_max, Katz centrality converges on any graph including
DAGs. It assigns each node a base score (β) plus attenuated contributions from all
paths leading to it, providing a meaningful alternative when eigenvector centrality
is undefined.

**Interpretation**: High EV/Katz identifies strategically important nodes — those connected
to other important nodes. This is the "VIP network" effect: a component may have few
direct connections, but if those connections are to critical hubs, it is strategically
important (and a high-value target for adversarial failure).

**Reference**: Bonacich (1987). Power and Centrality: A Family of Measures; Katz (1953).
A New Status Index Derived from Sociometric Analysis.

### 2.3.2 Degree Metrics

Degree metrics measure direct connectivity without considering global graph structure.

| # | Metric | Symbol | Formula | Captures |
|---|--------|--------|---------|----------|
| 6 | In-Degree | DG_in | \|{u : (u,v) ∈ E}\| / (\|V\|-1) | Number of dependents |
| 7 | Out-Degree | DG_out | \|{u : (v,u) ∈ E}\| / (\|V\|-1) | Number of dependencies |

Both raw counts and normalized values (divided by |V|-1) are stored. The normalized
values enable fair comparison across graphs of different sizes.

**Interpretation**: High in-degree means many components directly depend on this one
(fan-in). High out-degree means this component depends on many others (fan-out). A
component with high in-degree and low out-degree is a **provider**; the reverse is a
**consumer**. The `is_hub`, `is_source`, and `is_sink` properties on `StructuralMetrics`
derive from these values.

### 2.3.3 Resilience Metrics

Resilience metrics assess a component's role in graph connectivity and fault tolerance.

| # | Metric | Symbol | Graph | Captures |
|---|--------|--------|-------|----------|
| 8 | Clustering Coefficient | CC | Undirected | Local redundancy |
| 9 | Articulation Point | AP | Undirected | Single Point of Failure |
| 10 | Bridge Count | BR_count | Undirected | Critical edge exposure |
| 11 | Bridge Ratio | BR_ratio | Undirected | Edge vulnerability fraction |

#### Clustering Coefficient (CC)

Measures how connected a component's neighbors are to each other.

```
CC(v) = 2 × |edges among neighbors of v| / (k × (k - 1))
```

Where k is the degree of v in the undirected view.

**Interpretation**: High CC means the component's neighbors can communicate even if this
component fails — indicating local redundancy and better fault tolerance. Low CC combined
with high degree creates a hub-and-spoke anti-pattern where the hub is a critical SPOF.

#### Articulation Point (AP)

A boolean flag indicating whether removing this component disconnects the graph.

```
AP(v) = 1  if G \ {v} has more connected components than G
AP(v) = 0  otherwise
```

**Directed vs. undirected decision**: Articulation points are computed on the **undirected**
view of the dependency graph. This is a deliberate design choice:

- **Strong articulation points** (on directed graphs) require both strong connectivity
  in the original and fragmentation upon removal. This is too restrictive for dependency
  graphs, which are rarely strongly connected.
- **Weak articulation points** (on the undirected projection) capture any component whose
  removal partitions the reachability structure, which aligns with our failure model:
  if component v fails, any undirected path through v is broken, regardless of the
  original edge direction.

For disconnected graphs, articulation points are computed per connected component
(only for components with ≥3 nodes) and then unioned.

**Interpretation**: AP = True identifies Single Points of Failure (SPOFs) — the most
critical finding for system architects. These components require redundancy planning.

#### Bridge Count and Bridge Ratio

Bridge count is the number of bridge edges incident to a component. Bridge ratio is the
fraction of a component's edges that are bridges.

```
BR_count(v) = |{ e ∈ bridges(G) : v ∈ e }|
BR_ratio(v) = BR_count(v) / degree(v)      (0 if degree = 0)
```

A bridge is an edge whose removal disconnects the graph (computed on the undirected view,
per connected component for disconnected graphs).

**Interpretation**: High bridge ratio means most of a component's connections are
irreplaceable — there are no alternative paths. This is distinct from AP status: a
component can have all bridge edges without being an AP (e.g., a leaf node), and vice
versa.

### 2.3.4 Weight Metrics

Weight metrics carry forward the component and edge weights computed in Step 1.

| # | Metric | Symbol | Source | Captures |
|---|--------|--------|--------|----------|
| 12 | Component Weight | w(v) | Step 1 | Intrinsic importance from QoS |
| 13 | Dependency Weight In/Out | w_in(v), w_out(v) | Step 1 | Weighted connectivity |

```
w_in(v)  = Σ w(e)    for all e = (u, v) ∈ E_l
w_out(v) = Σ w(e)    for all e = (v, u) ∈ E_l
```

These are not normalized — they preserve the absolute QoS-derived importance for use in
quality scoring.

---

## 2.4 Analysis Pipeline

The analysis follows a six-phase pipeline applied to each requested layer:

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. Extract   │   │ 2. Centrality│   │ 3. Degree    │   │ 4. Resilience│   │ 5. Edge      │   │ 6. Summary   │
│ Layer        │──▶│ Metrics      │──▶│ Metrics      │──▶│ Metrics      │──▶│ Metrics      │──▶│ & Assembly   │
│ Subgraph     │   │              │   │              │   │              │   │              │   │              │
│              │   │ Directed G   │   │ Directed G   │   │ Undirected U │   │ Directed G   │   │ Both G and U │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

### Phase 1: Extract Layer Subgraph

Filter G_analysis(l) to include only components and DEPENDS_ON edges matching the
layer definition. Each `AnalysisLayer` specifies:

- `component_types`: which vertex types to include (e.g., Application for app layer)
- `dependency_types`: which DEPENDS_ON subtypes to include (e.g., app_to_app)
- `types_to_analyze`: which types appear in the output (may be a subset)

### Phase 2: Compute Centrality Metrics (on directed G)

```python
PageRank        ← nx.pagerank(G, alpha=0.85, weight="weight")
Reverse PR      ← nx.pagerank(G.reverse(), alpha=0.85, weight="weight")
Betweenness     ← nx.betweenness_centrality(G, weight="weight")   # inverted weights
Harmonic Close. ← nx.harmonic_centrality(G)
Eigenvector     ← nx.eigenvector_centrality(G)  or  nx.katz_centrality(G)  on failure
```

### Phase 3: Compute Degree Metrics (on directed G)

```python
in_degree_raw   ← G.in_degree(v)
out_degree_raw  ← G.out_degree(v)
in_degree_norm  ← in_degree_raw / (|V| - 1)     # normalized to [0, 1]
out_degree_norm ← out_degree_raw / (|V| - 1)
degree_norm     ← (in_degree_raw + out_degree_raw) / (2 × (|V| - 1))
```

### Phase 4: Compute Resilience Metrics (on undirected U)

```python
U               ← G.to_undirected()
Clustering      ← nx.clustering(U)
Art. Points     ← nx.articulation_points(U)      # per component if disconnected
Bridges         ← nx.bridges(U)                   # per component if disconnected
Bridge Count    ← count bridges incident to each node
Bridge Ratio    ← bridge_count / degree           # per node
```

### Phase 5: Compute Edge Metrics (on directed G)

```python
Edge Betweenness ← nx.edge_betweenness_centrality(G, weight="weight")  # inverted
Bridge Flag      ← (u,v) ∈ bridges or (v,u) ∈ bridges
```

### Phase 6: Assemble Summary

Graph-level statistics computed on both directed and undirected views:

```python
Density          ← nx.density(G)
Components       ← number of weakly connected components
Diameter         ← diameter of largest connected component (undirected)
Avg Path Length  ← average shortest path in largest component (undirected)
Assortativity    ← nx.degree_assortativity_coefficient(G)
Avg Clustering   ← average clustering coefficient across all nodes
AP Count         ← total number of articulation points
Bridge Count     ← total number of bridges
```

---

## 2.5 Directed vs. Undirected Decisions

A key methodological choice is which metrics are computed on the directed dependency
graph versus its undirected projection. The decision follows this principle:

| Aspect | Graph Type | Rationale |
|--------|-----------|-----------|
| **Influence & flow** | Directed | Dependency direction matters for propagation |
| **Connectivity & resilience** | Undirected | Failure breaks paths regardless of direction |

**Directed metrics** (PageRank, RPR, Betweenness, Closeness, Degree): These capture
directional semantics inherent to dependencies. A depends on B ≠ B depends on A.
Failure propagation, influence flow, and bottleneck analysis are all directional.

**Undirected metrics** (Clustering, Articulation Points, Bridges): These capture
structural connectivity. When a component fails, *all* its edges (incoming and outgoing)
are severed. Whether A→B or B→A, the path through the failed component is broken.
Articulation points on the undirected view capture any component whose failure
partitions the communication structure.

This mixed approach is consistent with prior work in network analysis of software
systems (e.g., Zimmermann & Nagappan, 2008; Bavota et al., 2013).

---

## 2.6 Weight Interpretation

Edge weights in the DEPENDS_ON graph represent dependency **strength** (derived from
Topic QoS in Step 1). However, different NetworkX algorithms interpret the `weight`
parameter differently:

| Algorithm | Weight Semantics | Our Handling |
|-----------|-----------------|--------------|
| **PageRank** | Importance (higher = more influence) | Pass raw weights |
| **Reverse PageRank** | Same as PageRank | Pass raw weights |
| **Betweenness** | Distance (higher = longer path) | **Invert**: w_dist = 1/w |
| **Edge Betweenness** | Distance | **Invert**: w_dist = 1/w |
| **Harmonic Closeness** | — | No weights (topological) |
| **Eigenvector / Katz** | Importance | Pass raw weights |

**Why inversion matters for betweenness**: Without inversion, a strong dependency
(weight = 2.0) would be treated as a longer path, making it *less* likely to be on
a shortest path. This is counterintuitive — strongly coupled components should have
*higher* betweenness for nodes on their paths. Inverting ensures that strong dependencies
are treated as short (preferred) paths.

**Why closeness uses no weights**: Harmonic closeness measures topological proximity —
how many hops away other components are. Using QoS-derived weights would conflate
structural distance with dependency strength, muddying the metric's interpretation.
Topological closeness better captures the "blast radius" concept: how many components
are structurally nearby regardless of dependency strength.

---

## 2.7 Normalization Strategy

All continuous metrics must be normalized to [0, 1] before use in Step 3 (Quality
Scoring). The normalization strategy varies by metric type.

### Pre-Normalized Metrics

Some metrics are inherently bounded or normalized by the algorithm:

| Metric | Range | Notes |
|--------|-------|-------|
| Betweenness | [0, 1] | NetworkX `normalized=True` divides by (n-1)(n-2)/2 |
| Clustering | [0, 1] | Defined as ratio of actual to possible triangles |
| Articulation Point | {0, 1} | Binary flag |
| Bridge Ratio | [0, 1] | Fraction of edges that are bridges |
| Degree (normalized) | [0, 1] | Divided by |V|-1 |

### Post-Normalized Metrics (Min-Max Scaling)

Metrics with unbounded or graph-dependent ranges are normalized by the `QualityAnalyzer`
in Step 3 using max-normalization:

```
x_norm(v) = x(v) / max(x(u) for all u ∈ V_l)
```

| Metric | Raw Range | Normalization |
|--------|-----------|---------------|
| PageRank | [0, 1] (sums to 1) | Max-norm (per-node range varies with |V|) |
| Reverse PageRank | [0, 1] | Max-norm |
| Harmonic Closeness | [0, |V|-1] | Max-norm |
| Eigenvector / Katz | [0, 1] or varies | Max-norm |
| In-degree (raw) | [0, |V|-1] | Max-norm (for RMAV formulas) |
| Total degree (raw) | [0, 2(|V|-1)] | Max-norm |

### Edge Case: Uniform Values

When max(x) = 0 (all values are zero, e.g., single-node graph or no shortest paths),
the normalizer returns 0.0 for all components. This is handled in `QualityAnalyzer._normalize()`
with a safe division guard.

---

## 2.8 Metric Correlation and Independence

Not all 13 metrics provide independent information. Understanding correlations helps
justify the metric set and informs interpretation.

### Expected High Correlations

| Pair | Reason | Implication |
|------|--------|-------------|
| PR ↔ EV | Both measure recursive influence | EV serves as validation for PR |
| PR ↔ DG_in | In-degree is local; PR is its transitive generalization | PR subsumes DG_in for importance |
| BT ↔ AP | APs tend to have high betweenness | AP provides binary confirmation of BT |

### Expected Low Correlations (Independent Information)

| Pair | Reason | What Each Adds |
|------|--------|----------------|
| PR ↔ BT | Hub ≠ Bottleneck | PR: importance; BT: structural position |
| PR ↔ CC | Global ≠ Local | PR: system-wide role; CC: local redundancy |
| BT ↔ CC | Often inversely correlated | BT: central position; CC: mesh connectivity |
| CL ↔ PR | Proximity ≠ Importance | CL: blast radius; PR: recursive influence |
| RPR ↔ PR | Complementary directions | RPR: vulnerability; PR: criticality |

### Justification for Metric Set Size

The 8 centrality/degree metrics map to 4 RMAV quality dimensions in Step 3, with each
dimension using 3 metrics. This requires at least 8 distinct metrics (some shared across
dimensions). The resilience metrics (CC, AP, BR) provide independent structural
information not captured by any centrality metric. The combined set ensures each RMAV
dimension has sufficient independent signal while maintaining interpretability.

---

## 2.9 Computational Complexity

| Phase | Algorithm | Complexity | Bottleneck |
|-------|-----------|------------|------------|
| Centrality | PageRank (×2) | O(k × (\|V\| + \|E\|)) | k iterations (~100) |
| | Betweenness (Brandes) | O(\|V\| × \|E\|) | **Dominant for dense graphs** |
| | Harmonic Closeness | O(\|V\| × \|E\|) | BFS from each node |
| | Eigenvector / Katz | O(\|V\|²) or O(k × \|E\|) | Power iteration |
| Degree | In/Out degree | O(\|V\|) | Trivial |
| Resilience | Clustering | O(\|V\| × d²_max) | d_max = max degree |
| | Articulation Points | O(\|V\| + \|E\|) | Single DFS |
| | Bridges | O(\|V\| + \|E\|) | Single DFS |
| Edges | Edge Betweenness | O(\|V\| × \|E\|) | Same as node betweenness |
| Summary | Diameter + Avg Path | O(\|V\| × \|E\|) | BFS from each node |

**Overall**: O(|V| × |E|) dominated by betweenness centrality. For typical pub-sub
systems (|V| < 300, |E| < 1000), this completes in under 1 second. For enterprise
graphs (|V| > 1000), betweenness can be approximated using k-sample estimation.

---

## 2.10 Graph-Level Summary

Beyond component metrics, the analyzer produces a `GraphSummary` with system-level
statistics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Density | 2\|E\| / (\|V\|×(\|V\|-1)) | Fraction of possible edges present |
| Components | # weakly connected components | Fragmentation indicator |
| Diameter | max shortest path in largest CC | Communication depth |
| Avg Path Length | mean shortest path in largest CC | Average propagation distance |
| Assortativity | Pearson correlation of degree-degree | Mixing pattern indicator |
| SPOF Ratio | # articulation points / \|V\| | Structural vulnerability |
| Bridge Ratio | # bridges / \|E\| | Edge vulnerability |

**Assortativity interpretation**: Degree assortativity measures whether high-degree
nodes tend to connect to other high-degree nodes (assortative, r > 0) or to low-degree
nodes (disassortative, r < 0). Distributed systems with strongly negative assortativity
exhibit hub-and-spoke patterns vulnerable to targeted attacks on high-degree hubs. This
single coefficient captures important structural vulnerability information at the
system level.

---

## Layer Analysis

Different layers reveal different structural patterns. The same analyzer is applied
to each layer with different component/edge filters:

| Layer | Components | Edge Types | What It Reveals |
|-------|-----------|------------|-----------------|
| **app** | Application | app_to_app | Software coupling, data flow bottlenecks |
| **infra** | Node | node_to_node | Hardware SPOFs, network topology issues |
| **mw** | Broker | (broker edges) | Middleware bottlenecks and broker criticality |
| **system** | All types | All DEPENDS_ON | Complete system criticality picture |

---

## Commands

```bash
# Analyze specific layer
python bin/analyze_graph.py --layer system

# Analyze all layers
python bin/analyze_graph.py --all

# Export results to JSON
python bin/analyze_graph.py --layer system --output results/analysis.json

# Analyze with AHP weights for quality scoring
python bin/analyze_graph.py --all --use-ahp
```

---

## Output Example

```json
{
  "layer": "app",
  "components": {
    "sensor_fusion": {
      "pagerank": 0.142,
      "reverse_pagerank": 0.089,
      "betweenness": 0.534,
      "closeness": 0.667,
      "eigenvector": 0.421,
      "in_degree_raw": 5,
      "out_degree_raw": 2,
      "degree": 0.292,
      "in_degree": 0.208,
      "out_degree": 0.083,
      "clustering_coefficient": 0.133,
      "is_articulation_point": true,
      "bridge_count": 2,
      "bridge_ratio": 0.286,
      "weight": 1.45,
      "dependency_weight_in": 4.20,
      "dependency_weight_out": 1.85
    }
  },
  "graph_summary": {
    "nodes": 25,
    "edges": 48,
    "density": 0.080,
    "num_components": 1,
    "diameter": 6,
    "avg_path_length": 2.84,
    "assortativity": -0.32,
    "num_articulation_points": 3,
    "num_bridges": 5,
    "avg_clustering": 0.21
  }
}
```

---

## Key Formulas Reference

| Metric | Formula | Weight Handling |
|--------|---------|-----------------|
| PageRank | PR(v) = (1-d)/n + d × Σ PR(u)/out(u) | Raw (importance) |
| Betweenness | BT(v) = Σ σ(s,t\|v) / σ(s,t) | Inverted (1/w as distance) |
| Harmonic CL | CL_H(v) = (1/(n-1)) × Σ 1/d(v,u) | None (topological) |
| Clustering | CC(v) = 2×edges / (k×(k-1)) | N/A (undirected) |
| Katz | Katz(v) = Σ_{k=1}^∞ α^k × (A^k·1)_v | Raw (importance) |

---

## Navigation

← [Step 1: Graph Model Construction](graph-model.md) | [Step 3: Quality Scoring →](quality-scoring.md)