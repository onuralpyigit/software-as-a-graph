# Step 2: Analysis

**Compute every component's structural fingerprint — the set of numbers that explain how it can fail and who it takes down with it.**

← [Step 1: Modeling](graph-model.md) | → [Step 3: Prediction](prediction.md)

---

## Table of Contents

1. [What This Step Does](#what-this-step-does)
2. [Analysis Pipeline](#analysis-pipeline)
3. [Layer Projections](#layer-projections)
4. [Cross-Layer Analysis](#cross-layer-analysis)
5. [Topological Analysis Flow](#topological-analysis-flow)
6. [Why Multiple Metrics?](#why-multiple-metrics)
7. [Metric Taxonomy](#metric-taxonomy)
8. [Normalization](#normalization)
9. [Formal Definitions](#formal-definitions)
   - [Reverse PageRank (RPR)](#reverse-pagerank-rpr)
   - [In-Degree (DG_in)](#in-degree-dg_in)
   - [Multi-Path Coupling Index (MPCI)](#multi-path-coupling-index-mpci)
   - [Fan-Out Criticality (FOC)](#fan-out-criticality-foc)
   - [Betweenness Centrality (BT)](#betweenness-centrality-bt)
   - [QoS-Weighted Out-Degree (w_out)](#qos-weighted-out-degree-w_out)
   - [Clustering Coefficient (CC)](#clustering-coefficient-cc)
   - [Directed AP Score (AP_c_directed)](#directed-ap-score-ap_c_directed)
   - [Bridge Ratio (BR)](#bridge-ratio-br)
   - [Connectivity Degradation Index (CDI)](#connectivity-degradation-index-cdi)
   - [Reverse Eigenvector Centrality (REV)](#reverse-eigenvector-centrality-rev)
   - [Reverse Closeness Centrality (RCL)](#reverse-closeness-centrality-rcl)
   - [QoS-Weighted In-Degree (w_in / QADS)](#qos-weighted-in-degree-w_in--qads)
   - [Path Complexity (PC)](#path-complexity-pc)
   - [Diagnostic Metrics](#diagnostic-metrics)
10. [Metric Catalogue Reference](#metric-catalogue-reference)
11. [Output: M(v) and S(G)](#output-mv-and-sg)
12. [Worked Example](#worked-example)
13. [Complexity](#complexity)
14. [Commands](#commands)
15. [What Comes Next](#what-comes-next)

---

## What This Step Does

Analysis takes the layer-projected dependency graph **G_analysis(l)** produced by Step 1 and computes a structural metric vector **M(v)** for every component. Each metric captures a structurally independent aspect of how a component is embedded in the system topology — how broadly its failure would propagate, whether removing it would partition the graph, how it is coupled to its neighbors, and how quickly faults could travel through it.

```
G_analysis(l)          StructuralAnalyzer           Output
(DEPENDS_ON graph)  →  13 RMAV-input metrics    →   M(v) per component
                        4 diagnostic metrics         S(G) graph summary
                        3 raw/derived counts
                        — all stored in M(v) —
```

**Scope of this step:** M(v) contains structural observations only. No criticality scores are computed here. Step 3 (Prediction) is the sole consumer of M(v) and applies AHP-derived weights to produce criticality predictions Q(v). The steps are kept separate to preserve the prediction–simulation independence guarantee: structural features must not be contaminated by simulation outcomes.

---

## Analysis Pipeline

The analysis step involves three layers of code. Understanding the call chain prevents confusion about where normalization, QoS profiling, and layer filtering happen.

```
bin/analyze_graph.py          ← CLI entry point
│   argparse flags:
│     --layer, --norm, --winsorize
│     --use-ahp, --equal-weights, --ahp-shrinkage
│     --sensitivity
│
├── saag.Client.analyze(layer, **kwargs)
│       Thin façade — wires dependencies, returns AnalysisResult
│
└── src.analysis.service.AnalysisService.analyze_layer(layer)
        │
        ├── AnalysisLayer.from_string(layer)      ← canonical layer resolution
        ├── IGraphRepository.get_graph_data()     ← load components & edges from Neo4j
        │
        ├── StructuralAnalyzer.analyze(graph_data, layer)
        │       Returns StructuralAnalysisResult
        │         .components : Dict[id, StructuralMetrics]   ← M(v)
        │         .edges      : Dict[(src,tgt), EdgeMetrics]
        │         .graph_summary : GraphSummary                ← S(G)
        │         .graph      : nx.DiGraph (retained for viz)
        │         .qos_profile: QoS distribution across topics
        │         .rcm_order  : bandwidth-minimized node order (RCM)
        │
        ├── PredictionService.predict_quality(struct_result)  ← Step 3 inputs
        ├── AntiPatternDetector.detect(quality_result)        ← smell detection
        └── ExplanationEngine.explain_system(...)             ← human-readable text
```

`AnalysisResult` (returned by `client.analyze()`) wraps `LayerAnalysisResult.raw`, which embeds both `StructuralAnalysisResult` and the immediate prediction derived from it. The CLI's `--output` flag calls `result.save(path)` to persist the full JSON.

---

## Layer Projections

Every call to `analyze()` targets exactly one **analysis layer** (π_l). The layer determines which component types appear in the subgraph and which DEPENDS_ON subtypes are included as edges. This is the same projection defined in Step 1:

| Layer flag | Layer name | Analyzed types | Dependency types | Quality focus |
|-----------|------------|---------------|-----------------|---------------|
| `app` | Application Layer | Application, Library | `app_to_app`, `app_to_lib` | Reliability |
| `infra` | Infrastructure Layer | Node | `node_to_node` | Availability |
| `mw` | Middleware Layer | Broker *(Apps & Nodes in subgraph to preserve edges)* | `app_to_broker`, `node_to_broker`, `broker_to_broker` | Maintainability |
| `system` | Complete System | Application, Broker, Node, Topic, Library | All six subtypes | Overall |

**Scope constraint (app layer):** The `app` layer includes `app_to_lib` edges so shared-library blast-radius is visible without requiring `--layer system`. A Library used by N apps has DG_in = 0 without these edges.

**Middleware layer note:** The `mw` subgraph includes Application and Node vertices only to preserve incoming edges to Brokers. Only Broker components appear in M(v) and S(G) results.

**The `all` shorthand** expands to `["app", "infra", "mw", "system"]` and runs all four layers sequentially. When `--output` is combined with `all` or a comma-separated list, output files are named `<base>_<layer>.<ext>` (e.g. `metrics_app.json`, `metrics_system.json`).

**Layer aliases** accepted by `AnalysisLayer.from_string()`:

| Alias | Resolves to |
|-------|------------|
| `application` | `app` |
| `infrastructure` | `infra` |
| `middleware`, `mw-app`, `mw-infra`, `broker`, `brokers`, `app_broker` | `mw` |
| `complete`, `all` | `system` |

---

## Topological Analysis Flow

`StructuralAnalyzer.analyze()` runs seven internal phases in a fixed order:

```
Phase 1  extract_layer_subgraph()
         │  Filter graph_data by layer's component_types and dependency_types
         │  Build nx.DiGraph G with node attrs: component_type, name, weight,
         │    subscriber_count, loc, cyclomatic_complexity, coupling_*,
         │    lcom, ip_address, cpu_cores, ...
         │  Build G_dist (inverted weights for distance-based algorithms)
         │  Build G_rev (G transposed — failure-propagation direction)

Phase 2  Centrality metrics  (all on directed G / G_rev / G_dist)
         │  PageRank(G, d=0.85, weight)              → pagerank
         │  PageRank(G_rev, d=0.85, weight)          → reverse_pagerank (RPR)
         │  betweenness_centrality(G_dist, weight)   → betweenness (BT)
         │  harmonic_centrality(G) / (n-1)           → closeness (CL)
         │  harmonic_centrality(G_rev) / (n-1)       → reverse_closeness (RCL)
         │  _safe_eigenvector(G)                     → eigenvector (EV)
         │    fallback chain: eigenvector → Katz → zeros
         │  _safe_eigenvector(G_rev)                 → reverse_eigenvector (REV)

Phase 3  Degree & new Tier 1 metrics
         │  in_degree, out_degree per node (raw + normalized by n-1)
         │  MPCI(v) = Σ max(path_count(e)-1, 0) / (n-1) over InEdges(v)
         │  path_complexity(v) = mean(log2(1+path_count(e))) over OutEdges(v)
         │  FOC(t) = subscriber_count(t) / max_subscriber_count (Topic nodes only)

Phase 4  AP_c_directed & CDI
         │  _compute_continuous_ap_scores(G):
         │    For each node v:
         │      AP_c_out(v) = 1 - |largest_CC(G_undirected \ v)| / (n-1)
         │      AP_c_in(v)  = 1 - |largest_CC(G_T_undirected \ v)| / (n-1)
         │      AP_c_directed(v) = max(AP_c_out, AP_c_in)
         │      CDI(v) = min((avg_L(G\v) - avg_L(G)) / avg_L(G), 1.0)
         │    Optimization: BFS sampled from 50 "core" nodes for |V| > 300

Phase 5  Resilience metrics  (all on undirected projection U)
         │  clustering_coefficient   via nx.clustering(U)
         │  is_articulation_point    via nx.articulation_points(U)
         │  bridges                  via nx.bridges(U)
         │  bridge_ratio = bridge_count(v) / degree(v)
         │  (Disconnected graphs: AP/bridge detection runs per connected component)

Phase 6  Pub-sub topology & QoS
         │  _compute_pubsub_metrics():
         │    Build bipartite app-topic graph from PUBLISHES_TO / SUBSCRIBES_TO edges
         │    pubsub_degree     = degree in bipartite graph / max_degree
         │    pubsub_betweenness = betweenness_centrality(bipartite graph)
         │    broker_exposure   = avg distinct brokers routing app's topics / max
         │  _collect_qos_profile():
         │    Aggregate durability, reliability, transport_priority distributions
         │    across all Topic nodes — passed to QualityAnalyzer for RMAV weight adjustment

Phase 7  Assemble & normalize
         │  Assemble StructuralMetrics per node (only for types_to_analyze)
         │  _compute_code_quality_metrics():
         │    Min-max normalize loc, cyclomatic_complexity, lcom independently
         │    per Application population and per Library population
         │    CQP = 0.10·LOC_norm + 0.35·CC_norm + 0.30·instability + 0.25·LCOM_norm
         │  Assemble EdgeMetrics per edge
         │  RCM ordering: reverse_cuthill_mckee for bandwidth minimization
         │  _build_summary() → GraphSummary S(G)
```

> **Weight semantics reminder:** Edge `weight` on DEPENDS_ON edges represents dependency *strength* (importance). PageRank, Eigenvector, and Katz use weights directly. Distance-based algorithms (Betweenness, CDI path length) use inverted weights (`1/w`) so that high-QoS dependencies are treated as "close" — the algorithm preferentially routes through critical edges.

---

## Why Multiple Metrics?

No single metric captures all aspects of structural criticality. Two components illustrate why:

- **Component A** has many transitive dependents (high RPR) but sits in a well-connected, redundant subgraph (low BT, AP_c_directed = 0, BR ≈ 0). It is a broad reliability risk but not a SPOF.
- **Component B** has few direct dependents (low RPR) but is the single connection between two graph clusters (AP_c_directed = 0.82, BR = 1.0). It is a structural single point of failure despite low blast radius.

A single metric misclassifies both. Thirteen RMAV-input metrics, drawn from four different theoretical families — random walk, local topology, resilience, and QoS-weighted degree — together produce a complete and orthogonal structural fingerprint.

---

## Metric Taxonomy

Every field in M(v) belongs to exactly one of three tiers. This taxonomy is the key to understanding which fields feed which later computation.

| Tier | Purpose | Metrics |
|------|---------|---------| 
| **Tier 1 — RMAV inputs** | Directly feed R(v), M(v), A(v), or V(v) in Step 3 | RPR, DG_in, MPCI, FOC, BT, w_out, CC, AP_c_directed, BR, CDI, REV, RCL, w_in, PC |
| **Tier 2 — Diagnostic** | Computed for visualization, output reports, and GNN features; do not feed RMAV formulas | PR, CL, EV, pubsub_degree, pubsub_betweenness, broker_exposure |
| **Tier 3 — Raw / inline-derived** | Integer counts and inline-derived scalars used only within Step 3 formulas; not stored as normalized metrics | DG_in_raw, DG_out_raw, is_articulation_point, bridge_count, CDPot, CouplingRisk_enh, QSPOF |

**Why PR, CL, EV are Tier 2:** The *forward* variants (PageRank, Closeness, Eigenvector) measure how much a component itself is influenced by others — they are informative for dependency visualization but do not directly capture failure propagation outward. Their reverse counterparts (RPR, RCL, REV), computed on G^T, capture how failures at v spread to v's dependents — the reliability-relevant direction. Computing both gives the full picture for dashboards while the RMAV formulas use only the reverse variants.

**Why pubsub_degree, pubsub_betweenness, broker_exposure are Tier 2:** These are computed on the raw bipartite app-topic graph (using PUBLISHES_TO / SUBSCRIBES_TO edges, not DEPENDS_ON edges). They enrich the Genieus visualization dashboard and serve as GNN features, but the RMAV formulas operate on the DEPENDS_ON graph where the same information is captured via DG_in, BT, and RPR respectively.

---

## Normalization

All Tier 1 metrics are normalized to [0, 1] before being consumed by Step 3. The **default method is `robust` normalization** (interquartile range scaling):

```
x_robust(v) = (x(v) − median) / IQR      then clipped to [0, 1]

IQR = Q75 − Q25 over all components in the layer
```

Alternatively, **rank normalization** can be selected with `--norm rank`:

```
x_rank(v) = rank(v) / (|V| − 1)

rank(v) = position of v when all components sorted by ascending x(v)
          (0-based; average-rank tie-breaking)
```

**Why rank normalization (when selected):** Min-max normalization is sensitive to outliers. In a system with one highly-central hub and 50 peripheral nodes, min-max assigns 1.0 to the hub and compresses all other values near 0 — the relative ordering among peripherals is lost. Rank normalization preserves the full ordinal structure regardless of extreme values. This is particularly important for betweenness centrality, which is typically sparse (most nodes have BT near 0, one or two have very high BT).

**Supported normalization methods** (passed via `--norm`):

| Flag value | Method | Notes |
|-----------|--------|-------|
| `robust` | IQR scaling | **Default.** Outlier-resistant; recommended for production. |
| `rank` | Ordinal rank / (n-1) | Preserves ordering; use when outlier distribution is extreme. |
| `minmax` | Min-max (x − min) / (max − min) | Precise relative magnitudes; sensitive to outliers. |
| `zscore` | Z-score (x − μ) / σ | Gaussian assumptions; use only when metrics are roughly normal. |

```
Edge case: If all components have identical raw values → normalized value = 0 for all v
           (no discriminating power; uniform prior for that metric in this layer)
```

Normalization is applied **independently per metric and per layer**. A component's rank score is relative to the population of the current analysis layer (app, infra, mw, or system).

### Normalization Caveats (Hardening Phase)

**1. Solitary Populations (Single-Node Layers):**
If a layer or node type contains only a single component (e.g., one core Library), the min-max span is zero. To preserve the intrinsic complexity signal, the system defaults to a normalized value of **1.0** (most critical) for that component rather than zeroing it out. This ensures large singleton components are still flagged for maintenance risk.

**2. Type-Split Normalization:**
Applications and Libraries are normalized as separate populations before being mixed in the $M(v)$ Maintainability dimension. This prevents a massive legacy monolithic application from compressing the complexity signal of all libraries to near-zero. However, this means a "0.80 complexity" Application is not directly comparable to a "0.80 complexity" Library in absolute terms.

**3. Library Ca/Ce Semantics:**
For Library nodes, `instability_code` uses static analysis coupling (CBO/Fan-in/Fan-out) rather than topological `DEPENDS_ON` edges. This captures the internal stability of the package logic, whereas topological coupling captures system-level blast radius.

**Optional winsorization:** Before rank normalization, raw values above the 95th percentile can be capped (`--winsorize`). This prevents a single extreme outlier from being ranked above all others while the 2nd–99th percentile occupy a single rank bucket.

---

## Formal Definitions

All definitions below operate on G_analysis(l) — the layer-projected DEPENDS_ON graph with QoS-derived edge weights. G^T denotes the transposed graph (all edge directions reversed).

### Reverse PageRank (RPR)

*Tier 1 → R(v)*

Computes PageRank on G^T. Captures **cascade reach** — how broadly a failure at v propagates in the direction of v's dependents.

```
RPR(v) = PageRank(G^T, d=0.85)[v]

d = damping factor (0.85), max iterations = 100, tolerance = 1e-6
```

**High RPR(v) means:** Failure at v reaches a large fraction of the system through the transitive dependency chain. RPR is the primary input to the Reliability dimension R(v).

> **Directional note:** In the DEPENDS_ON graph, edges point from dependent to dependency (App_sub → App_pub). Reversing the graph therefore gives edges pointing *from* publisher to subscribers — the natural failure-propagation direction. RPR on G^T thus counts how many nodes a failure would reach if it propagated outward from v through subscribers.

### In-Degree (DG_in)

*Tier 1 → R(v)*

```
DG_in(v) = in_degree(v) / (|V| − 1)     (normalized)
```

**High DG_in(v) means:** Many components directly depend on v — the immediate blast radius if v fails. DG_in measures *local* propagation; RPR measures *global* propagation. Both are needed because a highly-central hub may have a small local blast radius but a large transitive one.

### Multi-Path Coupling Index (MPCI)

*Tier 1 → R(v)*

A **new metric** added in this version. Uses the `path_count` attribute on DEPENDS_ON edges produced by Step 1's Phase 3. For a given component v, `path_count` on each incoming edge counts the number of distinct shared topics (for app_to_app dependencies) or distinct USES edges (for app_to_lib dependencies) that independently establish that dependency.

```
MPCI(v) = Σ_{e ∈ InEdges(v)} max(path_count(e) − 1, 0) / (|V| − 1)

InEdges(v) = set of incoming DEPENDS_ON edges to v
path_count(e) = number of topics (or USES edges) jointly establishing edge e
```

**Why `path_count − 1`:** A dependency with `path_count = 1` is a single coupling — baseline. Each additional shared topic is an *extra* coupling vector. MPCI sums these extra vectors across all dependents.

**High MPCI(v) means:** Multiple components are coupled to v through redundant shared channels. Each channel is an independent failure vector for those dependents. This amplifies the cascade depth that CDPot (Step 3 derived term) estimates: when a dependency collapses, it does so across all shared channels simultaneously rather than one path at a time.

```
MPCI(v) = 0    → all incoming dependencies are single-channel (baseline)
MPCI(v) > 0    → v has multi-channel coupling; higher values = greater coupling intensity
```

> **Library nodes benefit most from MPCI:** After Step 1's Rule 5 (app_to_lib), libraries now appear as DEPENDS_ON targets. A library used by 10 applications via a single USES edge each has high DG_in but MPCI = 0 (single-channel per dependency). The MPCI signal is non-zero only when the same (App, Lib) pair has multiple USES edges — currently rare — or when (App, App) pairs share multiple topics.

### Fan-Out Criticality (FOC)

*Tier 1 → R(v) for Topic nodes*

A **new metric** added in this version. Topics are not endpoints of DEPENDS_ON edges, so their DG_in and RPR in the dependency graph are 0. FOC provides a reliability signal for Topic nodes by using the `subscriber_count` attribute written by Step 1's Phase 2 fan-out augmentation.

```
FOC(t) = subscriber_count(t) / max_{t' ∈ V_topic} subscriber_count(t')     for Topic nodes
FOC(v) = 0                                                                    for all other types
```

**High FOC(t) means:** Topic t is a data distribution relay for many subscribers. If t becomes unreachable (broker failure, routing failure), all subscribers simultaneously lose their data source. FOC makes this blast relay pattern visible for Topic nodes in system-layer analysis.

> **Usage in R(v) for Topics:** In Step 3, when computing R(v) for a Topic node, the `DG_in` term is replaced with `FOC` because the dependency graph gives Topics no in-degree. The CDPot term uses `FOC` as the reach signal in place of `DG_in` for these nodes.
>
> **Layer restriction:** FOC is non-zero only when `--layer system` is used. Topic nodes are excluded from the `app` and `mw` subgraphs. The CLI will emit a warning when the active layer has no Topic nodes.

### Betweenness Centrality (BT)

*Tier 1 → M(v)*

```
BT(v) = Σ_{s≠v≠t} σ(s,t|v) / σ(s,t)   (Brandes' algorithm, O(|V|×|E|))

σ(s,t) = number of shortest paths from s to t
σ(s,t|v) = number of those paths passing through v

Normalized by (|V|−1)(|V|−2). Shortest paths use inverted weights (1/w) as distances.
```

**High BT(v) means:** v is a structural bottleneck — many dependency chains route through it. Changes to v risk disrupting many other components. The inversion of weights for distance computation means that high-QoS edges (strong dependencies) contribute less to the shortest-path distance — the algorithm preferentially routes through critical edges, making BT sensitive to high-weight dependency chains.

### QoS-Weighted Out-Degree (w_out)

*Tier 1 → M(v)*

```
w_out(v) = Σ_{(v,u) ∈ OutEdges(v)} weight(v,u)    (raw sum, then rank-normalized)
```

**High w_out(v) means:** v depends on many high-priority components. Each outgoing dependency is an efferent coupling; high-QoS couplings amplify change risk because a change to any dependency propagates back to v via its SLA obligations.

### Clustering Coefficient (CC)

*Tier 1 → M(v)*

```
CC(v) = |{(u,w) ∈ E_undirected : u,w ∈ N(v)}| / (deg(v) × (deg(v) − 1))

Computed on undirected projection of G_analysis(l). CC(v) = 0 if deg(v) < 2.
```

**High CC(v) means:** v's neighbors are well-connected among themselves — the local topology is redundant. Low CC (via the `1 − CC` term in M(v)) indicates each of v's couplings is unique and non-redundant, making v harder to safely modify.

### Directed AP Score (AP_c_directed)

*Tier 1 → A(v). Stored in M(v) (previously inline-computed in Step 3).*

The undirected AP_c measures how badly an undirected graph fragments. For a directed dependency graph, the directional variant captures how much of the *reachability structure* is lost when v is removed.

```
Given G' = G_analysis(l) with vertex v and all incident edges removed:

AP_c_out(v) = 1 − |largest weakly reachable set from any vertex in G'| / (|V| − 1)
AP_c_in(v)  = 1 − |largest set that can reach any vertex in G'|         / (|V| − 1)

AP_c_directed(v) = max(AP_c_out(v), AP_c_in(v))

AP_c_directed(v) = 0    → removing v does not fragment the directed dependency structure
AP_c_directed(v) → 1    → removing v severs a large fraction of directed reachability
```

**Why max, not average:** The worst-case direction determines the availability risk. If removing v severs 80% of out-reachability but only 10% of in-reachability, the system loses 80% of its downstream propagation paths — the maximum governs the severity.

> **Implementation note:** AP_c_directed was previously computed inside QualityAnalyzer (_compute_continuous_ap_scores). It is now computed in StructuralAnalyzer and stored in M(v) as `ap_c_directed`. This eliminates duplicate O(|V|²) computation and makes the field available to the GNN feature vector and to Step 5 validation.

### Bridge Ratio (BR)

*Tier 1 → A(v)*

```
BR(v) = |{e ∈ bridges(G_undirected) : v ∈ e}| / undirected_degree(v)

bridge = edge whose removal increases the number of connected components
BR(v) = 0 if degree(v) = 0
```

**High BR(v) means:** A large fraction of v's connections are non-redundant bridges. Losing any bridge edge disconnects a subgraph from the rest of the system.

### Connectivity Degradation Index (CDI)

*Tier 1 → A(v). Stored in M(v) (previously inline-computed in Step 3).*

Catches "soft" SPOF situations where v is not a strict articulation point but its removal still significantly lengthens paths in the surviving graph.

```
Let avg_L(H) = average shortest-path length over all reachable pairs in graph H

CDI(v) = min( (avg_L(G' \ {v}) − avg_L(G)) / avg_L(G),  1.0 )

If G' \ {v} is disconnected: avg_L(G' \ {v}) = ∞ → CDI(v) = 1.0
If |V| ≤ 2 after removal:   CDI(v) = 0

Complexity: O(|V| × (|V| + |E|)) via BFS.
For |V| > 300: sampled BFS from a random subset of source nodes (seed fixed for reproducibility).
```

**High CDI(v) means:** Removing v significantly increases the average path length in the surviving graph — even if the graph remains connected, dependency paths become much longer, indicating v was a shortcut that many routes depended on.

> **Implementation note:** CDI was previously computed inside QualityAnalyzer alongside AP_c_directed. Both are now computed together in StructuralAnalyzer and stored in M(v). The combined computation saves one full graph traversal pass.

### Reverse Eigenvector Centrality (REV)

*Tier 1 → V(v)*

```
REV(v) = eigenvector_centrality(G^T)[v]

Power iteration on G^T, max 500 iterations.
Fallback chain: eigenvector_centrality → katz_centrality(α=0.01) → zeros.
```

**High REV(v) means:** In G^T (failure-propagation direction), v is connected to other high-REV components — meaning v's downstream dependents are themselves important hubs. A compromise at v would cascade into a cluster of high-value targets.

> **Convergence note:** Eigenvector centrality may fail on directed acyclic graphs (DAGs) or nearly-acyclic graphs because the dominant eigenvalue does not exist. The Katz fallback with attenuation factor α = 0.01 handles these cases gracefully. If both fail, zeros are returned and a WARNING is logged.

### Reverse Closeness Centrality (RCL)

*Tier 1 → V(v)*

```
RCL(v) = harmonic_centrality(G^T)[v] / (|V| − 1)

Harmonic closeness is used for robustness to disconnected graphs.
```

**High RCL(v) means:** In G^T, v can reach many other components quickly — meaning, in the original graph, many components can propagate to v in few hops. Adversarial paths from dependents to v are short, amplifying exposure.

### QoS-Weighted In-Degree (w_in / QADS)

*Tier 1 → V(v) as QADS (QoS-weighted Attack-Dependent Surface)*

```
w_in(v) = Σ_{(u,v) ∈ InEdges(v)} weight(u,v)    (raw sum, then rank-normalized)
```

**High w_in(v) means:** Many high-SLA components directly depend on v. v is a high-value target because compromising it disrupts the most critical immediate consumers. The QoS weighting ensures that a dependency from an URGENT/PERSISTENT subscriber counts more than one from a LOW/BEST_EFFORT subscriber.

### Path Complexity (PC)

*Tier 1 → M(v)*

```
PC(v) = mean( log2(1 + path_count(e)) ) over e ∈ OutEdges(v)
```

**High PC(v) means:** v depends on other components through multiple redundant paths (shared topics). While this adds reliability, it increases the **Maintainability** risk (M) because change impact propagation follows all available paths. A change to v's logic may require complex re-synchronization across all paths mediating its efferent dependencies. PC serves as an intensifier for the **Coupling Risk** term in Step 3.


### Diagnostic Metrics

*Tier 2 — computed for visualization and GNN features; do not feed RMAV formulas*

| Metric | Definition | Purpose |
|--------|-----------|---------| 
| PageRank (PR) | Standard PageRank on G | Forward importance; shows which components accumulate the most transitive dependency weight |
| Closeness (CL) | Harmonic closeness on G | Forward propagation speed; complementary view to RCL for dashboards |
| Eigenvector (EV) | Eigenvector centrality on G | Forward influence through neighbors; complementary to REV |
| pubsub_degree | Degree in bipartite app-topic graph | Topic diversity of an application — how many distinct message channels it participates in |
| pubsub_betweenness | Betweenness in bipartite app-topic graph | Applications that bridge separate topic clusters |
| broker_exposure | Avg distinct brokers routing app's topics | Infrastructure blast surface — how many brokers an application's failure would stress |

---

## Metric Catalogue Reference

Complete M(v) field listing. Every field has a tier, a RMAV dimension (or "—" for Tier 2), and a direction (↑ = higher is worse / more critical, ↓ = higher is better).

| Field | Symbol | Tier | RMAV Dim | Dir | Description |
|-------|--------|------|----------|-----|-------------|
| `reverse_pagerank` | RPR | 1 | R | ↑ | Global cascade reach |
| `in_degree` | DG_in | 1 | R | ↑ | Normalized direct dependent count |
| `mpci` | MPCI | 1 | R | ↑ | Multi-path coupling intensity |
| `fan_out_criticality` | FOC | 1 | R | ↑ | Topic fan-out (Topic nodes only) |
| `betweenness` | BT | 1 | M | ↑ | Bottleneck position |
| `dependency_weight_out` | w_out | 1 | M | ↑ | QoS-weighted efferent coupling |
| `clustering_coefficient` | CC | 1 | M | ↓ | Local redundancy (used as 1−CC in M) |
| `ap_c_directed` | AP_c_dir | 1 | A | ↑ | Directed SPOF severity |
| `bridge_ratio` | BR | 1 | A | ↑ | Fraction of non-redundant edges |
| `cdi` | CDI | 1 | A | ↑ | Path elongation on removal |
| `reverse_eigenvector` | REV | 1 | V | ↑ | Strategic compromise reach |
| `reverse_closeness` | RCL | 1 | V | ↑ | Adversarial propagation speed |
| `dependency_weight_in` | w_in | 1 | V | ↑ | QoS-weighted afferent surface |
| `path_complexity` | PC | 1 | M | ↑ | Efferent path count complexity |
| `pagerank` | PR | 2 | — | — | Forward transitive importance |
| `closeness` | CL | 2 | — | — | Forward propagation speed |
| `eigenvector` | EV | 2 | — | — | Forward influence |
| `pubsub_degree` | — | 2 | — | — | Topic participation breadth |
| `pubsub_betweenness` | — | 2 | — | — | Topic cluster bridging |
| `broker_exposure` | — | 2 | — | — | Infrastructure blast surface |
| `in_degree_raw` | — | 3 | — | — | Raw integer in-degree (for CDPot, CouplingRisk_enh) |
| `out_degree_raw` | — | 3 | — | — | Raw integer out-degree (for CouplingRisk_enh) |
| `bridge_count` | — | 3 | — | — | Integer count of bridge edges incident to v |
| `is_articulation_point` | — | 3 | — | — | Binary AP flag (derived from AP_c_directed) |
| `weight` | w | — | A via QSPOF | ↑ | Component QoS weight from Step 1 |

**Code quality metrics** (Application and Library nodes only; 0.0 for all other types):

| Field | Tier | RMAV Dim | Description |
|-------|------|----------|-------------|
| `loc_norm` | 1→CQP | M | Normalized lines of code (min-max within type population) |
| `complexity_norm` | 1→CQP | M | Normalized cyclomatic complexity (min-max within type population) |
| `instability_code` | 1→CQP | M | Martin instability Ce/(Ca+Ce) — already in [0,1], not re-normalized |
| `lcom_norm` | 1→CQP | M | Normalized lack of cohesion (min-max within type population) |
| `code_quality_penalty` | 1 | M | CQP v7: `0.10·loc_norm + 0.35·complexity_norm + 0.30·instability_code + 0.25·lcom_norm` |

---

## Output: M(v) and S(G)

The `StructuralMetrics` dataclass stores all fields above per component. The graph-level summary `S(G)` (`GraphSummary`) provides aggregate topology statistics:

| Field | Type | Description |
|-------|------|-------------|
| `layer` | str | Layer identifier (e.g. `"system"`) |
| `nodes` | int | Number of components in this layer subgraph |
| `edges` | int | Number of DEPENDS_ON edges |
| `density` | float | `edges / (nodes × (nodes−1))` |
| `avg_degree` | float | Mean undirected degree |
| `avg_clustering` | float | Mean clustering coefficient (undirected) |
| `is_connected` | bool | True if graph is weakly connected |
| `num_components` | int | Number of weakly connected components |
| `num_articulation_points` | int | Total strict AP count |
| `num_bridges` | int | Total bridge edge count |
| `diameter` | int | Longest shortest path in largest CC (undirected) |
| `avg_path_length` | float | Average shortest path length in largest CC |
| `assortativity` | float | Pearson degree–degree correlation at edge endpoints |
| `node_types` | dict | `{component_type: count}` breakdown |
| `edge_types` | dict | `{dependency_type: count}` breakdown |
| `connectivity_health` | str | Derived: `ROBUST` / `MODERATE` / `FRAGILE` / `DISCONNECTED` |

> **`spof_count`** (components with `AP_c_directed > 0`) is derived from the components dict at query time, not stored directly in `GraphSummary`.

---

## Worked Example

**System:** SensorApp → `/temperature` ← MonitorApp; both → MainBroker; both → NavLib (after Step 1's Rule 5).

After Step 1 imports: `/temperature` has `subscriber_count = 1`.

After dependency derivation: edges are MonitorApp→SensorApp (path_count=1), MonitorApp→MainBroker (path_count=1), SensorApp→MainBroker (path_count=1), SensorApp→NavLib (path_count=1), MonitorApp→NavLib (path_count=1).

**Computed metrics (system layer):**

```
Component      RPR      DG_in  MPCI  AP_c_dir  BR    BT     w_in  FOC
─────────────────────────────────────────────────────────────────────────
SensorApp      0.58     0.25   0.0   0.43      1.0   0.40   0.0   0.0
MonitorApp     0.25     0.0    0.0   0.0       0.0   0.0    0.0   0.0
MainBroker     0.65     0.50   0.0   0.65      1.0   0.60   0.71  0.0
NavLib         0.72     0.50   0.0   0.50      1.0   0.50   0.71  0.0
/temperature   0.0      0.0    0.0   0.0       0.0   0.0    0.0   1.0
```

**Graph-level summary S(G):**

```
nodes=5, edges=5, density=0.25, avg_degree=2.0, avg_clustering=0.0
is_connected=True, num_components=1
num_articulation_points=3, num_bridges=5, diameter=2, avg_path_length=1.4
assortativity=-0.5       ← negative: high-degree hubs connect to low-degree leaves
```

Key observations:
- **NavLib** has the highest RPR and DG_in despite having no pub-sub connections. This is the Rule 5 effect — both applications failing simultaneously.
- **MainBroker** has the highest BT, reflecting that dependency paths from both applications route through it.
- **/temperature** has RPR = 0 and DG_in = 0 (Topic nodes are not DEPENDS_ON endpoints), but FOC = 1.0 (max fan-out for this system).
- **AP_c_directed(MainBroker) = 0.65** and **AP_c_directed(NavLib) = 0.50**: both are structural SPOFs. MainBroker's removal severs 65% of directed reachability; NavLib's removal severs 50%.
- MPCI = 0.0 everywhere because all dependencies in this small example are single-path. Multi-path MPCI would appear in larger systems where the same (App_sub, App_pub) pair shares multiple topics.
- **Negative assortativity** indicates a hub-and-spoke topology: the two hub nodes (MainBroker, NavLib) connect to lower-degree leaf nodes (SensorApp, MonitorApp).

These metric vectors become the input to Step 3's RMAV formula evaluation.

---

## Complexity

| Algorithm | Complexity | Notes |
|-----------|------------|-------|
| PageRank / RPR | O(I × \|E\|) | I = iterations (≤100) |
| Betweenness | O(\|V\| × \|E\|) | Brandes' algorithm; inverted weights |
| Closeness / RCL | O(\|V\| × (\|V\| + \|E\|)) | Harmonic closeness via BFS |
| Eigenvector / REV | O(I × \|E\|) | Power iteration (≤500 iters); Katz fallback |
| AP_c_directed | O(\|V\| × (\|V\| + \|E\|)) | Reachability removal per vertex |
| CDI | O(\|V\| × (\|V\| + \|E\|)) | APSP removal per vertex; sampled for \|V\| > 300 |
| Bridge detection | O(\|V\| + \|E\|) | DFS-based |
| MPCI | O(\|E\|) | One pass over InEdges per component |
| FOC | O(\|V_topic\|) | One pass over Topic nodes |
| Rank normalization | O(\|V\| log \|V\|) | Per metric sort |
| RCM ordering | O(\|V\| + \|E\|) | Bandwidth minimization for matrix display |

**Overall:** O(|V|² + |V|×|E|), dominated by AP_c_directed and CDI. An `xlarge` system (200 components, ~600 edges) completes in approximately 20–25 seconds. AP_c_directed and CDI together account for roughly 70% of runtime.

> **Performance note:** AP_c_directed and CDI are both computed in StructuralAnalyzer (moved from QualityAnalyzer). This consolidation eliminates one redundant O(|V|²) pass previously performed in Step 3. For enterprise-scale systems (|V| > 300), both use sampled BFS prioritizing "core" component types (Application, Broker, Node) with a fixed random seed of 42 for reproducibility across runs.

---

## Commands

```bash
# Analyze the system layer (default — includes all component types)
python bin/analyze_graph.py

# Analyze the application layer (Apps and Libraries only)
python bin/analyze_graph.py --layer app

# Analyze the middleware layer (Brokers only)
python bin/analyze_graph.py --layer mw

# Analyze the infrastructure layer (Nodes only)
python bin/analyze_graph.py --layer infra

# Analyze all four layers sequentially
python bin/analyze_graph.py --layer all

# Analyze multiple specific layers (comma-separated)
python bin/analyze_graph.py --layer app,system

# Export full metric vectors M(v) to JSON
python bin/analyze_graph.py --layer system --output results/metrics.json

# Multi-layer export: produces metrics_app.json, metrics_system.json
python bin/analyze_graph.py --layer app,system --output results/metrics.json

# Normalization methods (default: robust / IQR scaling)
python bin/analyze_graph.py --layer system                    # robust (default)
python bin/analyze_graph.py --layer system --norm rank        # rank normalization
python bin/analyze_graph.py --layer system --norm minmax      # min-max normalization
python bin/analyze_graph.py --layer system --norm zscore      # z-score normalization

# Enable winsorization to cap extreme outliers at the 95th percentile before ranking
python bin/analyze_graph.py --layer system --winsorize

# Weight modes for Step 3 (prediction) — do not affect M(v) computation
python bin/analyze_graph.py --layer system --use-ahp              # AHP-derived weights
python bin/analyze_graph.py --layer system --equal-weights        # equal 0.25 per dimension
python bin/analyze_graph.py --layer system --use-ahp --ahp-shrinkage 0.5

# Run AHP weight sensitivity analysis (Kendall τ stability report)
python bin/analyze_graph.py --layer system --sensitivity

# Connect to a non-default Neo4j instance
python bin/analyze_graph.py --uri bolt://myhost:7687 --user neo4j --password secret

# Increase logging verbosity
python bin/analyze_graph.py --layer app --verbose
```

### CLI Argument Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--layer`, `-l` | `system` | Layer(s) to analyze. Accepts a single layer, comma-separated list, or `all`. |
| `--norm` | `robust` | Normalization method: `robust`, `rank`, `minmax`, `zscore`. |
| `--winsorize` | off | Cap raw values above the 95th percentile before normalization. |
| `--use-ahp` | off | Use AHP-derived RMAV dimension weights (Step 3 only). |
| `--equal-weights` | off | Use equal 0.25 weights for all RMAV dimensions (baseline). |
| `--ahp-shrinkage` | `0.7` | Shrinkage factor λ ∈ [0, 1] for AHP weight blending. |
| `--sensitivity` | off | Run Kendall τ weight sensitivity analysis after prediction. |
| `--output`, `-o` | — | Path to save full JSON results. |
| `--uri` | `bolt://localhost:7687` | Neo4j Bolt URI (overrides `NEO4J_URI` env var). |
| `--user`, `-u` | `neo4j` | Neo4j username (overrides `NEO4J_USER` env var). |
| `--password`, `-p` | `password` | Neo4j password (overrides `NEO4J_PASSWORD` env var). |
| `--verbose`, `-v` | off | Enable DEBUG-level logging. |
| `--quiet`, `-q` | off | Suppress INFO messages; show only warnings and errors. |

### Interpreting the Output

```
Layer: app | 35 components | 87 edges | density: 0.073
SPOFs: 3  |  Bridges: 11  |  Multi-path couplings: 4

Top Critical Components (by Q(v)):
  1. DataRouter      [CRITICAL]  Q=0.91  RPR=0.89  AP_c_dir=0.62  BT=0.79  MPCI=0.12
  2. SensorHub       [CRITICAL]  Q=0.87  RPR=0.71  AP_c_dir=0.50  BT=0.71  MPCI=0.08
  3. CommandGateway  [HIGH]      Q=0.74  RPR=0.48  AP_c_dir=0.00  BT=0.83  MPCI=0.00

Topic Fan-Out Hotspots (system layer only):
  /sensor/lidar      FOC=1.00  subscribers=12  — blast relay for 12 applications
  /command/velocity  FOC=0.75  subscribers=9   — blast relay for 9 applications
```

Reading the output:
- Components with non-zero `AP_c_dir` are structural SPOFs — top priority for redundancy.
- Components with high `BT` but `AP_c_dir = 0` are bottlenecks but not SPOFs — consider decoupling.
- Components with non-zero `MPCI` have intensified coupling — multiple independent failure vectors reach them from the same dependents.
- Topics with high `FOC` are distribution choke points — if the topic's broker fails, all listed subscribers fail simultaneously.
- **Negative assortativity** (shown in `S(G)`) indicates hub-and-spoke topology — a few highly-critical hubs surrounded by many leaf-level consumers.
- **`connectivity_health`** of `FRAGILE` means one or more articulation points exist; `ROBUST` means no SPOFs were detected.

---

## What Comes Next

Step 2 produces M(v) — a structural fingerprint per component. The 13 Tier 1 metrics are precise but not actionable in isolation: "AP_c_directed = 0.62" does not tell an architect what to do.

Step 3 (Prediction) maps M(v) into four interpretable RMAV quality dimensions — Reliability, Maintainability, Availability, Vulnerability — using AHP-derived weights and box-plot adaptive thresholds. The RMAV decomposition explains *why* a component is critical, enabling targeted remediation rather than blanket "it's risky."

---

← [Step 1: Modeling](graph-model.md) | → [Step 3: Prediction](prediction.md)