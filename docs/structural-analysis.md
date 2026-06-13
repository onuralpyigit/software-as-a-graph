# Step 2: Analyze — Structural Metrics

**Compute every component's structural fingerprint — the set of numbers that explain how it can fail and who it takes down with it.**

← [Step 1: Import](graph-model.md) | → [Step 3: Predict](prediction.md)

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
11. [Analyze Stage — Rule-Based RMAV Scoring](#11-analyze-stage--rule-based-rmav-scoring)
    - 11.1 [The Four Quality Dimensions](#111-the-four-quality-dimensions)
    - 11.2 [RMAV Formulas](#112-rmav-formulas)
      - [Reliability R(v)](#reliability-rv--fault-propagation-risk)
      - [Maintainability M(v)](#maintainability-mv--coupling-complexity)
      - [Availability A(v)](#availability-av--spof-risk)
      - [Vulnerability V(v)](#vulnerability-vv--security-exposure)
      - [Composite Score Q(v)](#composite-score-qv)
    - 11.3 [Derived Terms](#113-derived-terms)
    - 11.4 [Metric Orthogonality](#114-metric-orthogonality)
    - 11.5 [AHP Weight Derivation](#115-ahp-weight-derivation)
    - 11.6 [Weight Shrinkage Strategy](#116-weight-shrinkage-strategy)
    - 11.7 [Criticality Classification](#117-criticality-classification)
    - 11.8 [Interpretation Patterns](#118-interpretation-patterns)
12. [Output: M(v) and S(G)](#12-output-mv-and-sg)
13. [Worked Example](#13-worked-example)
14. [Complexity](#14-complexity)
15. [Commands](#15-commands)
16. [What Comes Next](#16-what-comes-next)

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

**Scope of this step:** M(v) contains structural observations only. Criticality scores are computed in the subsequent sub-phase of Step 2 (Analyze, RMAV sub-phase), which consumes M(v) and applies AHP-derived weights to produce criticality predictions Q(v). The steps are kept separate to preserve the prediction–simulation independence guarantee: structural features must not be contaminated by simulation outcomes.

---

## Analysis Pipeline

The analysis step involves three layers of code. Understanding the call chain prevents confusion about where normalization, QoS profiling, and layer filtering happen.

```
cli/analyze_graph.py          ← CLI entry point
│   argparse flags:
│     --layer, --norm, --winsorize
│     --use-ahp, --equal-weights, --ahp-shrinkage
│     --sensitivity
│
├── saag.Client.analyze(layer, **kwargs)
│      Thin façade — wires dependencies, returns AnalysisResult
│
└── saag.usecases.analyze_graph.AnalyzeGraphUseCase.execute(layer)
      │
      └── saag.analysis.service.AnalysisService.analyze_layer(layer)
            │
            ├── AnalysisLayer.from_string(layer)      ← canonical layer resolution
            ├── IGraphRepository.derive_dependencies() ← derive DEPENDS_ON edges
            ├── IGraphRepository.get_graph_data()     ← load components & edges from Neo4j
            │
            ├── StructuralAnalyzer.analyze(graph_data, layer)
            │       Returns StructuralAnalysisResult
            │         .components : Dict[id, StructuralMetrics]   ← M(v)
            │         .edges      : Dict[(src,tgt), EdgeMetrics]
            │         .graph_summary : GraphSummary              ← S(G)
            │         .graph      : nx.DiGraph (retained for viz)
            │         .qos_profile: QoS distribution across topics
            │         .rcm_order  : bandwidth-minimized node order (RCM)
            │
            ├── PredictionService.predict_quality(struct_result)  ← Step 2 (Analyze, RMAV sub-phase) inputs
            ├── AntiPatternDetector.detect(quality_result)        ← smell detection
            └── ExplanationEngine.explain_system(...)             ← human-readable text
```

`AnalysisResult` (returned by `client.analyze()`) wraps `LayerAnalysisResult.raw`, which embeds both `StructuralAnalysisResult` and the immediate prediction derived from it. The CLI's `--output` flag calls `result.save(path)` to persist the full JSON.

### Pre-Analysis Hook

Before structural analysis begins, `AnalysisService.analyze_layer` automatically triggers the pre-analysis hook:
`self.repository.derive_dependencies()`
This derives the `DEPENDS_ON` edges and establishes their weights, ensuring that structural metrics are calculated on a fresh, fully-derived topology projection.

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

| Alias | Resolves to | Notes |
|-------|------------|-------|
| `application` | `app` | Legacy alias |
| `infrastructure` | `infra` | Legacy alias |
| `middleware`, `mw-app`, `mw-infra`, `broker`, `brokers`, `app_broker` | `mw` | Legacy aliases |
| `complete` | `system` | Legacy alias |
| `all` | `system` (via `from_string`) | Special CLI handling: `--layer all` expands to run all four layers sequentially |

---

## Cross-Layer Analysis

When `--layer all` (or a comma-separated layer list) is used, `AnalysisService.analyze_all_layers()` runs each layer independently and then derives **cross-layer insights** — observations that only become visible by correlating results across two or more layers.

### What cross-layer insights capture

A component that is `CRITICAL` in the `app` layer alone is a service-level reliability risk. The same component also classified as `CRITICAL` in the `infra` layer means its physical host is simultaneously a structural SPOF — an entirely different failure mode. No single-layer analysis surfaces this compound risk; only the multi-layer view can.

Three insight types are produced:

| Insight type | Trigger | Severity |
|---|---|---|
| `compound_critical` | Component is `CRITICAL` or `HIGH` in ≥ 2 distinct layers | `CRITICAL` if any layer classifies it CRITICAL, else `HIGH` |
| `systemic_spof` | Component is an articulation point (`AP_c_directed > 0`) in ≥ 2 distinct layers | `CRITICAL` |
| `layer_concentration` | A single layer has > 30 % of its analysed components classified `CRITICAL` | `HIGH` |

### How the correlation works

After all four layer results are assembled, `_compute_cross_layer_insights()` builds a component-indexed map across all `LayerAnalysisResult` objects:

```
For each component id that appears in ≥ 2 layer results:
  high_layers  = layers where levels.overall ≥ HIGH
  spof_layers  = layers where structural.is_articulation_point == True

  if |high_layers| ≥ 2  → emit compound_critical insight
  if |spof_layers| ≥ 2  → emit systemic_spof insight

For each layer:
  if CRITICAL_count / total_components > 0.30  → emit layer_concentration insight
```

Insights are sorted by severity (`CRITICAL` before `HIGH`) and then by number of affected layers (more layers = higher priority).

### Data model

```python
@dataclass
class CrossLayerInsight:
    component_id:    str        # empty string for layer_concentration insights
    csc_name:  str        # human-readable name from structural metrics
    insight_type:    str        # "compound_critical" | "systemic_spof" | "layer_concentration"
    layers_affected: List[str]  # e.g. ["app", "system"]
    severity:        str        # "CRITICAL" | "HIGH" | "MEDIUM"
    description:     str        # free-text explanation
```

The `MultiLayerAnalysisResult.cross_layer_insights` field carries the full list. It is serialised under the `cross_layer_insights` key in the `--output` JSON.

### Layer membership semantics

A component appears in a given layer only if its type is in that layer's `analyze_types`. This means:

- A `Broker` node can appear in both `mw` results (it is the sole analyzed type) and `system` results — so a broker that is CRITICAL in both layers would produce a `compound_critical` insight.
- An `Application` never appears in `infra` or `mw` results, so no cross-layer signal is possible between those two layers for application nodes. Cross-layer signals for applications are limited to `app` ↔ `system`.
- Nodes (`Node` type) can appear in `infra` and `system`, making infrastructure-level SPOFs detectable across both views.

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
          │  FOC(t) = log1p(f(t)) × s(t) / max_t[log1p(f(t)) × s(t)] (Topic nodes, frequency-weighted)
          │    where f(t) = topic message frequency in Hz, s(t) = subscriber count

Phase 4  AP_c_directed & CDI
         │  _compute_continuous_ap_scores(G):
         │    For each node v:
         │      AP_c_out(v) = 1 - |largest_CC(G_undirected \ v)| / (n-1)
         │      AP_c_in(v)  = 1 - |largest_CC(G_T_undirected \ v)| / (n-1)
         │      AP_c_directed(v) = max(AP_c_out, AP_c_in)
         │      CDI(v) = min((avg_L(G\v) - avg_L(G)) / avg_L(G), 1.0)
         │    Optimization: for |V| > 300 the CDI BFS uses the top-50
         │    highest-degree "core" nodes (Application, Broker, Node),
         │    ranked by in+out degree — deterministic, no randomness

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
         │    CQP = 0.10·loc_norm + 0.35·complexity_norm + 0.30·instability_code + 0.25·lcom_norm
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
| **Tier 1 — RMAV inputs** | Directly feed R(v), M(v), A(v), or V(v) in Step 2 (Analyze, RMAV sub-phase) | RPR, DG_in, MPCI, FOC, BT, w_out, CC, AP_c_directed, BR, CDI, REV, RCL, w_in, PC |
| **Tier 2 — Diagnostic** | Computed for visualization, output reports, and GNN features; do not feed RMAV formulas | PR, CL, EV, pubsub_degree, pubsub_betweenness, broker_exposure |
| **Tier 3 — Raw / inline-derived** | Integer counts and inline-derived scalars used only within Step 2 (Analyze, RMAV sub-phase) formulas; not stored as normalized metrics | DG_in_raw, DG_out_raw, is_articulation_point, bridge_count, CDPot, CouplingRisk_enh, QSPOF |

**Why PR, CL, EV are Tier 2:** The *forward* variants (PageRank, Closeness, Eigenvector) measure how much a component itself is influenced by others — they are informative for dependency visualization but do not directly capture failure propagation outward. Their reverse counterparts (RPR, RCL, REV), computed on G^T, capture how failures at v spread to v's dependents — the reliability-relevant direction. Computing both gives the full picture for dashboards while the RMAV formulas use only the reverse variants.

**Why pubsub_degree, pubsub_betweenness, broker_exposure are Tier 2:** These are computed on the raw bipartite app-topic graph (using PUBLISHES_TO / SUBSCRIBES_TO edges, not DEPENDS_ON edges). They enrich the SMART visualization dashboard and serve as GNN features, but the RMAV formulas operate on the DEPENDS_ON graph where the same information is captured via DG_in, BT, and RPR respectively.

---

## Normalization

All Tier 1 metrics are normalized to [0, 1] before being consumed by Step 2 (Analyze, RMAV sub-phase). The **default method is `robust` normalization** (rank-based scaling):

```
x_robust(v) = rank(v) / (|V| − 1)

rank(v) = position of v when all components sorted by ascending x(v)
        (0-based; average-rank tie-breaking)
```

> **Note on terminology:** The `--norm robust` flag performs rank-based normalization, not IQR scaling as the term "robust" might suggest. This preserves ordinal relationships and is robust to outliers.

**Why rank normalization (default):** Min-max normalization is sensitive to outliers. In a system with one highly-central hub and 50 peripheral nodes, min-max assigns 1.0 to the hub and compresses all other values near 0 — the relative ordering among peripherals is lost. Rank normalization preserves the full ordinal structure regardless of extreme values. This is particularly important for betweenness centrality, which is typically sparse (most nodes have BT near 0, one or two have very high BT).

**Supported normalization methods** (passed via `--norm`):

| Flag value | Method | Notes |
|-----------|--------|-------|
| `robust` | Rank-based normalization | **Default.** Preserves ordinal relationships; robust to outliers. |
| `rank` | Same as `robust` | Provided for explicit clarity. |
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

**High MPCI(v) means:** Multiple components are coupled to v through redundant shared channels. Each channel is an independent failure vector for those dependents. This amplifies the cascade depth that CDPot (Step 2 (Analyze, RMAV sub-phase) derived term) estimates: when a dependency collapses, it does so across all shared channels simultaneously rather than one path at a time.

```
MPCI(v) = 0    → all incoming dependencies are single-channel (baseline)
MPCI(v) > 0    → v has multi-channel coupling; higher values = greater coupling intensity
```

> **Library nodes benefit most from MPCI:** After Step 1's Rule 5 (app_to_lib), libraries now appear as DEPENDS_ON targets. A library used by 10 applications via a single USES edge each has high DG_in but MPCI = 0 (single-channel per dependency). The MPCI signal is non-zero only when the same (App, Lib) pair has multiple USES edges — currently rare — or when (App, App) pairs share multiple topics.

### Fan-Out Criticality (FOC)

*Tier 1 → R(v) for Topic nodes*

A **new metric** added in this version. Topics are not endpoints of DEPENDS_ON edges, so their DG_in and RPR in the dependency graph are 0. FOC provides a reliability signal for Topic nodes by using the `subscriber_count` attribute written by Step 1's Phase 2 fan-out augmentation, combined with topic frequency for QoS-aware weighting.

```
FOC(t) = log1p(f(t)) × s(t) / max_{t' ∈ V_topic}[log1p(f(t')) × s(t')]   for Topic nodes
FOC(v) = 0                                                                   for all other types
```

where `f(t)` = topic message frequency in Hz, `s(t)` = subscriber count.

**High FOC(t) means:** Topic t is a data distribution relay for many subscribers at high message rate. If t becomes unreachable (broker failure, routing failure), all subscribers simultaneously lose their data source. The `log1p` compression handles large frequency variance while preserving monotonicity.

> **Usage in R(v) for Topics:** In Step 2 (Analyze, RMAV sub-phase), when computing R(v) for a Topic node, the `DG_in` term is replaced with `FOC` because the dependency graph gives Topics no in-degree. The CDPot term uses `FOC` as the reach signal in place of `DG_in` for these nodes.
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

*Tier 1 → A(v). Stored in M(v) (previously inline-computed in Step 2 (Analyze, RMAV sub-phase)).*

The undirected AP_c measures how badly an undirected graph fragments. For a directed dependency graph, the directional variant captures how much of the *reachability structure* is lost when v is removed.

```
Given G' = G_analysis(l) with vertex v and all incident edges removed:

AP_c_out(v) = 1 − |largest connected component in undirected (G_analysis(l) \ v)| / (|V| − 1)
AP_c_in(v)  = 1 − |largest connected component in undirected (G_analysis(l)^T \ v)| / (|V| − 1)

AP_c_directed(v) = max(AP_c_out(v), AP_c_in(v))

AP_c_directed(v) = 0    → removing v does not fragment the undirected projection of the layer graph
AP_c_directed(v) → 1    → removing v fragments the undirected projection into small components
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

*Tier 1 → A(v). Stored in M(v) (previously inline-computed in Step 2 (Analyze, RMAV sub-phase)).*

Catches "soft" SPOF situations where v is not a strict articulation point but its removal still significantly lengthens paths in the surviving graph.

```
Let avg_L(H) = average shortest-path length over all reachable pairs in graph H

CDI(v) = min( (avg_L(G' \ {v}) − avg_L(G)) / avg_L(G),  1.0 )

If G' \ {v} is disconnected: avg_L(G' \ {v}) = ∞ → CDI(v) = 1.0
If |V| ≤ 2 after removal:   CDI(v) = 0

Complexity: O(|V| × (|V| + |E|)) via BFS.
For |V| > 300: BFS source nodes are restricted to the top-50 highest-degree
"core" nodes (Application, Broker, Node), ranked by in+out degree. This is
deterministic — no random sampling — so CDI values are identical across runs
on the same graph. High-degree nodes have disproportionate impact on average
path length, making them the most informative BFS sources for CDI estimation.
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

**High PC(v) means:** v depends on other components through multiple redundant paths (shared topics). While this adds reliability, it increases the **Maintainability** risk (M) because change impact propagation follows all available paths. A change to v's logic may require complex re-synchronization across all paths mediating its efferent dependencies. PC serves as an intensifier for the **Coupling Risk** term in Step 2 (Analyze, RMAV sub-phase).


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
| publisher_spof (PSPOF) | `max(w(t) × min(sub_count(t)/5, 1))` over sole-published topics | Sole-publisher risk: if this application is the only publisher on a topic and that topic has active subscribers, PSPOF quantifies the blast if the application goes silent. Available in M(v) for dashboards and GNN features. |

---

## Metric Catalogue Reference

Complete M(v) field listing. Every field has a tier, a RMAV dimension (or "—" for Tier 2), and a direction (↑ = higher is worse / more critical, ↓ = higher is better).

| Field | Symbol | Tier | RMAV Dim | Dir | Description |
|-------|--------|------|----------|-----|-------------|
| `reverse_pagerank` | RPR | 1 | R | ↑ | Global cascade reach |
| `in_degree` | DG_in | 1 | R | ↑ | Normalised direct dependent count |
| `mpci` | MPCI | 1 | R | ↑ | Multi-path coupling intensity |
| `fan_out_criticality` | FOC | 1 | R | ↑ | Topic fan-out (log1p(Hz) × subscribers, Topic nodes only) |
| `betweenness` | BT | 1 | M | ↑ | Bottleneck position |
| `dependency_weight_out` | w_out | 1 | M | ↑ | QoS-weighted efferent coupling |
| `clustering_coefficient` | CC | 1 | M | ↓ | Local redundancy (used as 1−CC in M) |
| `ap_c_directed` | AP_c_dir | 1 | A | ↑ | Directed SPOF severity |
| `bridge_ratio` | BR | 1 | A | ↑ | Fraction of non-redundant edges |
| `cdi` | CDI | 1 | A | ↑ | Path elongation on removal |
| `reverse_eigenvector` | REV | 1 | V | ↑ | Strategic compromise reach |
| `reverse_closeness` | RCL | 1 | V | ↑ | Adversarial propagation speed |
| `dependency_weight_in` | w_in | 1 | V | ↑ | QoS-weighted afferent surface (QADS) |
| `path_complexity` | PC | 1 | M | ↑ | Efferent path count complexity |
| `pagerank` | PR | 2 | — | — | Forward transitive importance |
| `closeness` | CL | 2 | — | — | Forward propagation speed |
| `eigenvector` | EV | 2 | — | — | Forward influence |
| `pubsub_degree` | — | 2 | — | — | Topic participation breadth |
| `pubsub_betweenness` | — | 2 | — | — | Topic cluster bridging |
| `broker_exposure` | — | 2 | — | — | Infrastructure blast surface |
| `publisher_spof` | PSPOF | 2 | — | ↑ | Sole-publisher blast risk (Application nodes; 0.0 otherwise) |
| `in_degree_raw` | — | 3 | — | — | Raw integer in-degree (for CouplingRisk_enh derivation) |
| `out_degree_raw` | — | 3 | — | — | Raw integer out-degree (for CouplingRisk_enh derivation) |
| `bridge_count` | — | 3 | — | — | Integer count of bridge edges incident to v |
| `is_articulation_point` | — | 3 | — | — | Binary AP flag (derived from undirected articulation detection) |
| `is_directed_ap` | — | 3 | — | — | Binary flag for directed articulation (used in cross-layer systemic_spof detection) |
| `blast_radius` | — | 3 | — | — | Number of descendants reachable if v is removed |
| `cascade_depth` | — | 3 | — | — | Longest failure propagation path from v |
| `topic_frequency_hz` | — | 3 | — | — | Raw message rate in Hz (Topic nodes; 0.0 otherwise) |
| `weight` | w | 1 | A | ↑ | Component QoS weight from Step 1; factor in QSPOF |

**Code quality metrics** (Application and Library nodes only; 0.0 for all other types):

| Field | Tier | RMAV Dim | Description |
|-------|------|----------|-------------|
| `loc_norm` | 1→CQP | M | Normalized lines of code (min-max within type population) |
| `complexity_norm` | 1→CQP | M | Normalized cyclomatic complexity (min-max within type population) |
| `instability_code` | 1→CQP | M | Martin instability Ce/(Ca+Ce) — already in [0,1], not re-normalized |
| `lcom_norm` | 1→CQP | M | Normalized lack of cohesion (min-max within type population) |
| `code_quality_penalty` | 1 | M | CQP v7: `0.10·loc_norm + 0.35·complexity_norm + 0.30·instability_code + 0.25·lcom_norm` |

---

## 11. Analyze Stage — Rule-Based RMAV Scoring

### 11.1 The Four Quality Dimensions

| Dimension | Question answered | High score means | Primary stakeholder |
|-----------|------------------|-----------------|---------------------|
| **R — Reliability** | How broadly and deeply does failure propagate? | Failure cascades widely and is hard to contain | Reliability Engineer |
| **M — Maintainability** | How hard is this to change safely? | Tightly coupled; structural bottleneck | Software Architect |
| **A — Availability** | Is this a structural single point of failure? | Removing it partitions the dependency graph | DevOps / SRE |
| **V — Vulnerability** | How attractive a target is this for attack? | Central, reachable, high-value downstream | Security Engineer |

The four dimensions are deliberately **orthogonal** in metric input: each raw metric feeds exactly one dimension (see [Metric Orthogonality](#114-metric-orthogonality)). This means a component's RMAV breakdown tells you *why* it is critical — a pure SPOF has high A but low R, M, V; a God Component has high M; a cascade hub has high R — enabling targeted remediation instead of blanket hardening.

---

### 11.2 RMAV Formulas

All inputs are normalized to [0, 1] by Step 2's rank normalization unless otherwise noted. All RMAV scores are therefore in [0, 1]. Intra-dimension weights are derived from AHP; see [Section 11.5](#115-ahp-weight-derivation).

---

#### Reliability R(v) — Fault Propagation Risk

R(v) measures how broadly and deeply a component's failure propagates through the DEPENDS_ON dependency graph.

**Standard formula (v6)** — Application, Broker, Infrastructure Node, Library:

```
R(v) = 0.45 × RPR(v) + 0.30 × DG_in(v) + 0.25 × CDPot_enh(v)
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| RPR(v) | 0.45 | Reverse PageRank on G^T — transitive cascade reach in the failure-propagation direction (primary signal) |
| DG_in(v) | 0.30 | In-degree (normalised) — immediate blast radius; number of direct dependents |
| CDPot_enh(v) | 0.25 | Enhanced Cascade Depth Potential — depth × breadth of the cascade, amplified by multi-path coupling (see [§11.3](#113-derived-terms)) |

> **Direction argument.** DEPENDS_ON edges point `dependent → dependency` (App_sub → App_pub). A failure at v propagates *against* edge direction — to nodes that depend on v (v's in-neighbours in the DEPENDS_ON graph). RPR reverses G to produce edges `dependency → dependent`, making the failure-propagation path the natural traversal direction. RPR(v) therefore accumulates rank from the nodes v would transitively impair when it fails. Forward PageRank on the DEPENDS_ON graph measures how much accumulated weight v *receives* from its own dependencies — it captures importance as a callee, not cascade reach as a failure origin. This is the answer to "why reverse?" in the committee review.

**Topic-type formula** — used exclusively for Topic nodes:

Topic nodes have DG_in = 0 in the DEPENDS_ON graph because Topics are not DEPENDS_ON endpoints. Their reliability risk is measured instead through subscriber fan-out:

```
R_topic(v) = 0.50 × FOC(v)  +  0.50 × CDPot_topic(v)

CDPot_topic(v) = FOC(v) × (1 − min(publisher_count_norm(v), 1))
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| FOC(v) | 0.50 | Fan-Out Criticality — how many subscribers simultaneously lose their data source |
| CDPot_topic(v) | 0.50 | Fan-out depth — topics with many subscribers but few publishers are pure blast relays with no publisher-side redundancy to absorb the loss |

> **Type dispatch.** The formula branch is resolved by `τ_V(v)` (the vertex type attribute on the graph node). `τ_V(v) = Topic` → Topic formula; all other node types → standard formula.

---

#### Maintainability M(v) — Coupling Complexity

M(v) measures how structurally embedded a component is in the system, making it fragile to change.

```
M(v) = 0.35 × BT(v)
     + 0.30 × w_out(v)
     + 0.15 × CQP(v)
     + 0.12 × CouplingRisk_enh(v)
     + 0.08 × (1 − CC(v))
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| BT(v) | 0.35 | Betweenness Centrality — fraction of shortest dependency paths that pass through v; the defining structural bottleneck signal |
| w_out(v) | 0.30 | QoS-weighted out-degree — efferent coupling weighted by SLA priority; high-priority outgoing dependencies amplify change risk |
| CQP(v) | 0.15 | Code Quality Penalty — composite of cyclomatic complexity, instability, and LCOM; zero for non-Application/Library types (formula degrades gracefully) |
| CouplingRisk_enh(v) | 0.12 | Topology instability enriched by path complexity — peaks when DG_in ≈ DG_out; intensified when shared topics create complex multi-path coupling (see [Section 11.3](#113-derived-terms)) |
| 1 − CC(v) | 0.08 | Inverse clustering coefficient — low local redundancy means each of v's connections is a structurally unique coupling path |

**CQP formula** (Application and Library nodes only; CQP = 0 for all other node types):
```
CQP(v) = 0.10 × loc_norm(v)  +  0.35 × complexity_norm(v)  +  0.30 × instability_code(v)  +  0.25 × lcom_norm(v)
```

**Why two instability signals in M(v)?** M(v) contains two coupling-related terms that may appear redundant: `instability_code` (inside CQP) and `CouplingRisk_enh` (topological). They capture distinct architectural layers:
- `instability_code` measures efferent coupling at the *static code level* (package imports, class dependencies) — technically fragile implementations.
- `CouplingRisk_enh` measures efferent coupling at the *runtime topology level* (USES edges, pub-sub relationships) — structurally fragile deployment roles.

These two often diverge: a library can have high static fan-out but only one consumer in the current deployment, or an application can have simple code but hundreds of pub-sub topics. Both signals are needed to capture the full maintenance risk picture.

---

#### Availability A(v) — SPOF Risk

A(v) measures whether a component is a structural single point of failure, weighted by its QoS priority, bridge redundancy, and path elongation.

```
A(v) = 0.35 × AP_c_directed(v) + 0.25 × QSPOF(v) + 0.25 × BR(v) + 0.10 × CDI(v) + 0.05 × w(v)
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| AP_c_directed(v) | 0.35 | Directed articulation point score — primary SPOF signal; removal partitions the dependency graph flows |
| QSPOF(v) | 0.25 | QoS-weighted SPOF severity — `AP_c_directed × w(v)`; amplifies critical SPOFs that serve high-priority traffic |
| BR(v) | 0.25 | Bridge Ratio — fraction of edges that are non-redundant structural bridges |
| CDI(v) | 0.10 | Connectivity Degradation Index — path elongation on removal; soft SPOF signal |
| w(v) | 0.05 | QoS aggregate weight — direct priority bias on the component's own operational weight |

> **AP_c_directed vs AP_c undirected.** The SPOF detection signal uses `AP_c_directed`, which is computed on the *directed* DEPENDS_ON graph using a worst-case out-reachability / in-reachability measure. This correctly captures directed cut vertices — nodes whose removal breaks directed reachability — rather than the undirected articulation point, which can both over-report (paths that are directionally irrelevant) and under-report (asymmetric directed SPOFs) in pub-sub systems.

---

#### Vulnerability V(v) — Security Exposure

V(v) measures how attractive v is as an attack target and how far a compromise would propagate.

```
V(v) = 0.40 × REV(v)  +  0.35 × RCL(v)  +  0.25 × w_in(v)
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| REV(v) | 0.40 | Reverse Eigenvector Centrality — v's downstream dependents are themselves important hubs; compromise at v cascades into high-value targets |
| RCL(v) | 0.35 | Reverse Closeness Centrality — many components can reach v quickly; adversarial paths to v are short |
| w_in(v) | 0.25 | QoS-weighted in-degree (QADS) — direct high-SLA dependents make v attractive because compromising it disrupts the most operationally critical consumers |

---

#### Composite Score Q(v)

```
Q(v) = w_A × A(v)  +  w_R × R(v)  +  w_M × M(v)  +  w_V × V(v)
```

**AHP-derived weights (recommended):**

| Dimension | Weight | Rationale |
|-----------|:------:|-----------|
| Availability (A) | **0.43** | Strongest structural alignment; SPOF severity dominates pre-deployment risk |
| Reliability (R) | **0.24** | Cascade propagation reach; directly tied to blast radius |
| Maintainability (M) | **0.17** | Coupling complexity; amplifies long-term fragility |
| Vulnerability (V) | **0.16** | Security exposure surface; strategic but secondary to structural concerns |

> **Cross-dimension weight derivation.** The 4×4 AHP matrix above gives CR ≈ 0.02 (well within the 0.10 acceptability threshold). The dominant position of A(v) reflects that SPOF failure is the most directly measurable structural risk in a pre-deployment topology: an articulation point with BR = 1.0 partitions the graph with certainty, whereas the cascade depth and coupling effects of R(v) and M(v) are probabilistic.

**Baseline (equal weights).** An equal-weight alternative (0.25 each) can be activated via `--equal-weights` for sensitivity analysis or reproducibility. The AHP-derived weights produce a meaningfully different top-k ranking than equal weights; this difference is part of what the Step 5 validation measures.

**Sensitivity of λ.** The AHP shrinkage factor λ ∈ [0, 1] blends raw AHP weights toward equal weights:
```
w_final(d) = λ × w_AHP(d) + (1 − λ) × 0.25
```
The default λ = 0.70 was selected from a sensitivity sweep across λ ∈ {0.50, 0.60, 0.70, 0.80, 0.90, 1.00}. Spearman ρ plateaus in the λ ∈ [0.65, 0.75] range, indicating that the AHP signal saturates near the default value. Traceability: the shrinkage implementation is `saag/analysis/weight_calculator.py`; the sweep artifact is tracked in `docs/internal/TODO.md#ahp-shrinkage-sweep-artifact` until committed to `output/`.

---

### 11.3 Derived Terms

These scalars are computed inline within the RMAV formulas at scoring time. They are derived from M(v) fields produced in Step 2; they are not stored as independent graph properties.

#### CDPot_enh — Enhanced Cascade Depth Potential

CDPot_enh captures how deeply a failure propagates in the absorber direction of the dependency graph, amplified by multi-path couplings.

```
CDPot_enh(v) = min( CDPot_base(v) × (1 + MPCI(v)),  1.0 )

CDPot_base(v) = ((RPR(v) + DG_in(v)) / 2)  ×  (1 − min(DG_out_raw(v) / max(DG_in_raw(v), ε), 1))

ε = 1e-9  (division guard)
```

Note: `DG_out_raw` and `DG_in_raw` are the raw integer degree counts, not the normalized versions. Using raw counts preserves the ratio semantics — a node with 10 in-edges and 2 out-edges should behave differently from one with 1 and 0.2, even though both have normalized ratio ≈ 0.2.

| Factor | Interpretation |
|--------|---------------|
| `(RPR + DG_in) / 2` | Average cascade reach: global breadth (RPR) combined with immediate blast radius (DG_in) |
| `1 − min(DG_out_raw / DG_in_raw, 1)` | Depth penalty: absorber nodes (DG_in >> DG_out) score high; fan-out hubs (DG_out >> DG_in) approach 0 |
| `× (1 + MPCI)` | Multi-path amplifier: when the same dependents share multiple topics, each topic is an independent failure vector; cascade depth grows with coupling intensity |

**Why MPCI amplifies depth, not breadth.** MPCI counts redundant channels between existing dependent pairs — it does not add new dependent nodes. The count of dependents (DG_in) and their transitive reach (RPR) are unchanged by MPCI. What changes is the *depth* of impact: when v fails, all `path_count` shared topics with each dependent fail simultaneously, making the cascade harder to absorb. This is a depth effect.

**Behaviour reference:**

| Node type | DG_in | DG_out | MPCI | CDPot_base | CDPot_enh | Interpretation |
|-----------|:-----:|:------:|:----:|:----------:|:---------:|---------------|
| Absorber hub | High | Low | 0 | High | High | Deep cascade, single-channel |
| Absorber + multi-path | High | Low | High | High | Very high | Deep cascade, multiple independent vectors |
| Fan-out hub | Low | High | 0 | ≈ 0 | ≈ 0 | Wide but shallow — quickly absorbed |
| Isolated leaf | 0 | 0 | 0 | 0 | 0 | No cascade potential |

---

#### CouplingRisk_enh — Topology Instability with Path Complexity

```
Instability_topo(v) = DG_out_raw(v) / (DG_in_raw(v) + DG_out_raw(v) + ε)

CouplingRisk_base(v) = 1 − |2 × Instability_topo(v) − 1|

CouplingRisk_enh(v) = min(1.0,  CouplingRisk_base(v) × (1 + Δ × path_complexity(v)))

Δ = 0.10 (COUPLING_PATH_DELTA)
```

| Topology role | Instability | CouplingRisk_base | Interpretation |
|--------------|:-----------:|:-----------------:|---------------|
| Pure source (DG_in = 0) | 1.0 | 0 | No afferent pressure — not fragile from above |
| Pure sink (DG_out = 0) | 0.0 | 0 | No efferent pressure — not fragile from below |
| Balanced (DG_in ≈ DG_out) | ≈ 0.5 | ≈ 1.0 | Maximum fragility — structural pressure from both directions |

The `path_complexity` term is an intensifier: if a node is already coupling-balanced (fragile), having many redundant topics per dependency further increases synchronisation complexity, raising the effective maintenance risk. The result is capped at 1.0.

---

#### QSPOF — QoS-Weighted SPOF Severity

```
QSPOF(v) = AP_c_directed(v) × w(v)
```

Scales the directed articulation point score by the component's operational QoS weight. A component that is structurally a SPOF *and* handles high-priority traffic is a doubly severe availability risk: its removal is both certain to disconnect the graph and certain to affect the most critical data flows.

---

### 11.4 Metric Orthogonality

Each raw metric from M(v) feeds **exactly one** RMAV dimension. No metric appears in more than one formula. Violations would inflate the effective weight of shared metrics relative to the AHP calibration.

| Metric | Symbol | R | M | A | V | Notes |
|--------|--------|:-:|:-:|:-:|:-:|-------|
| Reverse PageRank | RPR | ✓ | | | | Global cascade reach |
| In-Degree (norm) | DG_in | ✓ | | | | Immediate blast radius |
| MPCI | MPCI | ✓ via CDPot | | | | Amplifier only; enters via derived term |
| Fan-Out Criticality | FOC | ✓ Topics | | | | Substitutes for DG_in on Topic nodes |
| Path Complexity | path_complexity | | ✓ via CouplingRisk | | | Structural coupling depth |
| Betweenness | BT | | ✓ | | | Structural bottleneck |
| QoS Out-Degree | w_out | | ✓ | | | Priority-weighted efferent coupling |
| Code Quality Penalty | CQP | | ✓ | | | Complexity + instability + LCOM |
| CouplingRisk_enh | CouplingRisk | | ✓ | | | Derived from DG_in_raw, DG_out_raw |
| Clustering Coefficient | CC | | ✓ as 1−CC | | | Local path redundancy |
| Directed AP Score | AP_c_directed | | | ✓ | | Directly in A(v) and via QSPOF |
| Bridge Ratio | BR | | | ✓ | | Non-redundant edge fraction |
| CDI | CDI | | | ✓ | | Path elongation on removal |
| Reverse Eigenvector | REV | | | | ✓ | Strategic compromise reach |
| Reverse Closeness | RCL | | | | ✓ | Adversarial reach speed |
| QoS In-Degree (QADS) | w_in | | | | ✓ | High-SLA attack surface |
| PageRank | PR | — | — | — | — | Diagnostic only (Tier 2) |
| Closeness | CL | — | — | — | — | Diagnostic only (Tier 2) |
| Eigenvector | EV | — | — | — | — | Diagnostic only (Tier 2) |

---

### 11.5 AHP Weight Derivation

Intra-dimension weights are derived from the **Analytic Hierarchy Process (AHP)** using pairwise comparison matrices on Saaty's 1–9 scale.

```
Step 1 — Construct n×n matrix A:  A[i][j] = importance of criterion i relative to j
          Reciprocal constraint:  A[j][i] = 1 / A[i][j]

Step 2 — Geometric mean per row:  GM[i] = ( ∏_j A[i][j] )^(1/n)

Step 3 — Normalise:  w[i] = GM[i] / Σ_j GM[j]

Step 4 — Consistency check:
          λ_max = average of ( (A·w)[i] / w[i] )  for all i
          CI    = (λ_max − n) / (n − 1)
          CR    = CI / RI[n]
          Abort if CR > 0.10
```

Reference RI values (Saaty 1980): n=3 → 0.58, n=4 → 0.90, n=5 → 1.12, n=6 → 1.24.

#### Reliability AHP (3×3: RPR, DG_in, CDPot_enh)

```
            RPR    DG_in  CDPot
RPR      [ 1.00,  1.50,  2.00 ]   RPR: global reach is the primary cascade signal
DG_in    [ 0.67,  1.00,  1.50 ]   DG_in: immediate dependents are secondary
CDPot    [ 0.50,  0.67,  1.00 ]   CDPot: cascade depth is supplementary

→ AHP raw weights:  [0.45,  0.30,  0.25]    CR ≈ 0.001
```

MPCIs enter R(v) indirectly through CDPot_enh and do not add a 4th AHP criterion. This preserves the 3×3 matrix and its near-zero CR while capturing the MPCI effect.

#### Maintainability AHP (5×5: BT, w_out, CQP, CouplingRisk, CC_inv)

```
            BT     w_out  CQP    CR     CC_inv
BT       [1.00,  1.17,  2.33,  2.92,  4.38]   BT: primary structural bottleneck
w_out    [0.86,  1.00,  2.00,  2.50,  3.75]   w_out: QoS-weighted efferent coupling
CQP      [0.43,  0.50,  1.00,  1.25,  1.88]   CQP: code-level coupling
CR       [0.34,  0.40,  0.80,  1.00,  1.50]   CouplingRisk: topology instability
CC_inv   [0.23,  0.27,  0.53,  0.67,  1.00]   CC_inv: local redundancy (supplementary)

→ AHP raw weights:  [0.35,  0.30,  0.15,  0.12,  0.08]    CR ≈ 0.000
```

CQP and CouplingRisk receive equal AHP judgement because both measure coupling — CQP at the code level, CouplingRisk at the deployment topology level — and neither dominates the other a priori.

#### Availability AHP (5×5: AP_c_directed, QSPOF, BR, CDI, w)

```
                AP_c   QSPOF    BR     CDI      w
AP_c_directed [1.00,  1.40,  1.40,  3.50,  7.00]   AP_c: primary directed SPOF signal
QSPOF         [0.71,  1.00,  1.00,  2.50,  5.00]   QSPOF: QoS-amplified SPOF severity
BR            [0.71,  1.00,  1.00,  2.50,  5.00]   BR: bridge fraction
CDI           [0.29,  0.40,  0.40,  1.00,  2.00]   CDI: soft SPOF / path elongation
w             [0.14,  0.20,  0.20,  0.50,  1.00]   w: direct QoS priority weight

→ AHP raw weights:  [0.35,  0.25,  0.25,  0.10,  0.05]    CR ≈ 0.001
```

#### Composite Q AHP (4×4: A, R, M, V)

```
        A      R      M      V
A   [1.00,  1.50,  2.50,  2.67]   Availability: dominant (SPOF = certain graph partition)
R   [0.67,  1.00,  1.67,  1.78]   Reliability: propagation reach is second priority
M   [0.40,  0.60,  1.00,  1.07]   Maintainability: coupling fragility is tertiary
V   [0.37,  0.56,  0.93,  1.00]   Vulnerability: security exposure is supplementary

→ AHP raw weights:  [0.43,  0.24,  0.17,  0.16]    CR ≈ 0.02
```

---

### 11.6 Weight Shrinkage Strategy

Raw AHP weights can be extreme on small comparison sets. The shrinkage strategy formally blends them with a uniform prior:

```
w_final(d) = λ × w_AHP(d)  +  (1 − λ) × (1 / n_dimensions)
```

The default λ = 0.70 was selected from a sensitivity sweep across λ ∈ {0.50, 0.60, 0.70, 0.80, 0.90, 1.00}. Spearman ρ on the ATM validation dataset plateaus in the λ ∈ [0.65, 0.75] range. Traceability: `saag/analysis/weight_calculator.py` implements the shrinkage formula; the sweep artifact is tracked in `docs/internal/TODO.md#ahp-shrinkage-sweep-artifact` until committed to `output/`.

---

### 11.7 Criticality Classification

RMAV scores are classified using an **adaptive box-plot classifier** that identifies components exceptional relative to the system's own distribution:

```
CRITICAL  :  score > Q3 + 1.5 × IQR   (structural outliers)
HIGH      :  Q3 < score ≤ upper fence   (upper quartile, non-outlier)
MEDIUM    :  median < score ≤ Q3
LOW       :  Q1 < score ≤ median
MINIMAL   :  score ≤ Q1
```

Classification is applied **independently per RMAV dimension and for the composite Q(v)**. A component can be CRITICAL on Availability (structural SPOF) but MINIMAL on Vulnerability — which is exactly the diagnostic information needed to direct remediation.

**Small-sample fallback (n < 12).** Box-plot thresholds become unstable at small node counts. For graphs with fewer than 12 components, percentile thresholds are used instead: CRITICAL = top 10%, HIGH = 75th–90th, MEDIUM = 50th–75th, LOW = 25th–50th, MINIMAL = bottom 25%.

**Typical distribution across validated scenarios:** CRITICAL ≈ 5–15%, HIGH ≈ 25%, MEDIUM ≈ 25%, LOW ≈ 25%, MINIMAL ≈ bottom 10–25%.

---

### 11.8 Interpretation Patterns

The combination of RMAV dimension scores characterises the *type* of risk and directs remediation:

| Pattern | R | M | A | V | Primary risk | Recommended action |
|---------|:-:|:-:|:-:|:-:|-------------|-------------------|
| **Full hub** | H | H | H | H | Catastrophic — all failure modes | Redundancy + circuit breakers + hardening |
| **Reliability hub** | H | L | L | L | Wide cascade | Retry logic, graceful degradation, back-pressure |
| **Bottleneck** | L | H | L | L | Change fragility | Reduce coupling; extract an interface or façade |
| **SPOF** | L | L | H | L | Availability loss | Redundant instance, active-passive failover |
| **High-value target** | L | L | L | H | Compromise propagation | Zero-trust boundaries, audit logs, network isolation |
| **Compound: SPOF + hub** | H | H | H | H | Unreliable *and* unrefactorable | Architecture redesign required before any other mitigation |
| **Multi-path sink** | H (MPCI>0) | M | M | L | Deep multi-channel cascade | Reduce shared-topic count between the same dependent pair |
| **Maintenance debt** | M | H | M | L | Technical debt accumulation | Prioritise refactoring before the next feature sprint |
| **Leaf** | L | L | L | L | None | Standard monitoring |

> **Compound SPOF + God Component.** A component that is simultaneously an articulation point (high A) and a structural bottleneck with high total degree (high M) is the highest-priority compound risk in the catalog. It is unreliable (any failure partitions the graph) *and* untestable/unrefactorable (too many responsibilities to change safely). In the ATM system, `ConflictDetector` (Q ≈ 0.90, AP = true) is the primary compound risk candidate.

---

## 12. Output: M(v) and S(G)

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
| `connectivity_health` | str | Derived: `HEALTHY` / `MODERATE` / `AT_RISK` (based on SPOF ratio and component count) |

> **`spof_count`** (components with `AP_c_directed > 0`) is derived from the components dict at query time, not stored directly in `GraphSummary`.

---

## 13. Worked Example

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

Key structural observations:
- **NavLib** has the highest RPR and DG_in despite having no pub-sub connections. This is the Rule 5 effect — both applications failing simultaneously.
- **MainBroker** has the highest BT, reflecting that dependency paths from both applications route through it.
- **/temperature** has RPR = 0 and DG_in = 0 (Topic nodes are not DEPENDS_ON endpoints), but FOC = 1.0 (max fan-out for this system).
- **AP_c_directed(MainBroker) = 0.65** and **AP_c_directed(NavLib) = 0.50**: both are structural SPOFs. MainBroker's removal severs 65% of directed reachability; NavLib's removal severs 50%.
- MPCI = 0.0 everywhere because all dependencies in this small example are single-path. Multi-path MPCI would appear in larger systems where the same (App_sub, App_pub) pair shares multiple topics.
- **Negative assortativity** indicates a hub-and-spoke topology: the two hub nodes (MainBroker, NavLib) connect to lower-degree leaf nodes (SensorApp, MonitorApp).

**R(v) scores (standard v6 formula: R(v) = 0.45 × RPR + 0.30 × DG_in + 0.25 × CDPot_enh):**

```
SensorApp:    R = 0.45×0.58 + 0.30×0.25 + 0.25×0.0   = 0.261 + 0.075 + 0.0   = 0.336
MonitorApp:   R = 0.45×0.25 + 0.30×0.0  + 0.25×0.0   = 0.113 + 0.0   + 0.0   = 0.113
MainBroker:   R = 0.45×0.65 + 0.30×0.50 + 0.25×0.575 = 0.293 + 0.150 + 0.144 = 0.587
NavLib:       R = 0.45×0.72 + 0.30×0.50 + 0.25×0.61  = 0.324 + 0.150 + 0.153 = 0.627
/temperature: R_topic = 0.50×1.0 + 0.50×(1.0 × (1 − min(0.0, 1))) = 1.000  ← highest in system
```

Key reliability observations:
- `/temperature` scores R = 1.0 because FOC = 1.0 (the only topic with active subscribers) and there is only one publisher. In a larger system with multiple topics, this would rank relative to peers.
- **NavLib** outranks MainBroker on R because it is depended upon by both application nodes directly (via Rule 5 `app_to_lib` DEPENDS_ON edges), giving it higher PageRank.
- **MonitorApp** has R = 0.113 — no components depend on it, so its failure cascades to no one.

**A(v) scores (abbreviated):**

```
MainBroker: A = 0.35×0.65 + 0.25×(0.65×0.71) + 0.25×1.0 + 0.10×0.5 + 0.05×0.71
              = 0.228 + 0.115 + 0.250 + 0.050 + 0.036 = 0.679   → [HIGH]

NavLib:     A = 0.35×0.50 + 0.25×(0.50×0.71) + 0.25×1.0 + 0.10×0.4 + 0.05×0.71
              = 0.175 + 0.089 + 0.250 + 0.040 + 0.036 = 0.590   → [HIGH]
```

Both MainBroker and NavLib are structural SPOFs with BR = 1.0. Adding redundancy for either would be the top remediation priority for this system.

---

## 14. Complexity

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

> **Performance note:** AP_c_directed and CDI are both computed in StructuralAnalyzer (moved from QualityAnalyzer). This consolidation eliminates one redundant O(|V|²) pass previously performed in Step 2 (Analyze, RMAV sub-phase). For enterprise-scale systems (|V| > 300), the CDI BFS is restricted to the top-50 "core" nodes (Application, Broker, Node) ranked by total degree (in + out). This is fully deterministic — the same graph always produces the same CDI values — and prioritises the nodes most likely to have significant path-length impact when removed.

---

## 15. Commands

```bash
# Analyze the system layer (default — includes all component types)
PYTHONPATH=. python cli/analyze_graph.py

# Analyze the application layer (Apps and Libraries only)
PYTHONPATH=. python cli/analyze_graph.py --layer app

# Analyze the middleware layer (Brokers only)
PYTHONPATH=. python cli/analyze_graph.py --layer mw

# Analyze the infrastructure layer (Nodes only)
PYTHONPATH=. python cli/analyze_graph.py --layer infra

# Analyze all four layers sequentially — also produces cross_layer_insights
PYTHONPATH=. python cli/analyze_graph.py --layer all

# Analyze multiple specific layers (comma-separated) — also produces cross_layer_insights
PYTHONPATH=. python cli/analyze_graph.py --layer app,system

# Export full metric vectors M(v) to JSON
PYTHONPATH=. python cli/analyze_graph.py --layer system --output results/metrics.json

# Multi-layer export: produces metrics_app.json, metrics_system.json
PYTHONPATH=. python cli/analyze_graph.py --layer app,system --output results/metrics.json

# Normalization methods
# Note: "robust" (default) performs rank-based normalization, not IQR scaling
PYTHONPATH=. python cli/analyze_graph.py --layer system                 # rank-based (default)
PYTHONPATH=. python cli/analyze_graph.py --layer system --norm rank       # same as default
PYTHONPATH=. python cli/analyze_graph.py --layer system --norm minmax   # min-max normalization
PYTHONPATH=. python cli/analyze_graph.py --layer system --norm zscore     # z-score normalization

# Enable winsorization to cap extreme outliers at the 95th percentile before ranking
PYTHONPATH=. python cli/analyze_graph.py --layer system --winsorize

# Weight modes for Step 2 (Analyze, RMAV sub-phase) — do not affect M(v) computation
PYTHONPATH=. python cli/analyze_graph.py --layer system --use-ahp              # AHP-derived weights
PYTHONPATH=. python cli/analyze_graph.py --layer system --equal-weights        # equal 0.25 per dimension
PYTHONPATH=. python cli/analyze_graph.py --layer system --use-ahp --ahp-shrinkage 0.5

# Run AHP weight sensitivity analysis (Kendall τ stability report)
PYTHONPATH=. python cli/analyze_graph.py --layer system --sensitivity

# Connect to a non-default Neo4j instance
PYTHONPATH=. python cli/analyze_graph.py --uri bolt://myhost:7687 --user neo4j --password secret

# Increase logging verbosity
PYTHONPATH=. python cli/analyze_graph.py --layer app --verbose
```

### CLI Argument Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--layer`, `-l` | `system` | Layer(s) to analyze. Accepts a single layer, comma-separated list, or `all`. |
| `--norm` | `robust` | Normalization method: `robust` (rank-based), `rank` (same as robust), `minmax`, `zscore`. |
| `--winsorize` | off | Cap raw values above the 95th percentile before normalization. |
| `--use-ahp` | off | Use AHP-derived RMAV dimension weights (Step 2 Analyze, RMAV sub-phase only). |
| `--equal-weights` | off | Use equal 0.25 weights for all RMAV dimensions (baseline). |
| `--ahp-shrinkage` | `0.7` | Shrinkage factor λ ∈ [0, 1] for AHP weight blending. |
| `--sensitivity` | off | Run Kendall τ weight sensitivity analysis after prediction. |
| `--output`, `-o` | — | Path to save full JSON results. Parent directory created if absent. |
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

When `--layer all` is used, the output JSON also includes a `cross_layer_insights` array:

```json
"cross_layer_insights": [
  {
    "component_id": "broker-001",
    "csc_name": "MainBroker",
    "insight_type": "systemic_spof",
    "layers_affected": ["infra", "mw"],
    "severity": "CRITICAL",
    "description": "MainBroker is a structural articulation point in 2 layers (infra, mw). Its failure would disconnect subgraphs at multiple architectural levels."
  },
  {
    "component_id": "app-core",
    "csc_name": "DataRouter",
    "insight_type": "compound_critical",
    "layers_affected": ["app", "system"],
    "severity": "CRITICAL",
    "description": "DataRouter is classified CRITICAL in 2 layers (app, system), indicating compound risk that spans architectural boundaries."
  },
  {
    "component_id": "",
    "csc_name": "",
    "insight_type": "layer_concentration",
    "layers_affected": ["mw"],
    "severity": "HIGH",
    "description": "Layer 'mw' has 4/11 (36%) components classified as CRITICAL — high systemic risk concentration in this architectural tier."
  }
]
```

Reading the output:
- Components with non-zero `AP_c_dir` are structural SPOFs — top priority for redundancy.
- Components with high `BT` but `AP_c_dir = 0` are bottlenecks but not SPOFs — consider decoupling.
- Components with non-zero `MPCI` have intensified coupling — multiple independent failure vectors reach them from the same dependents.
- Topics with high `FOC` are distribution choke points — if the topic's broker fails, all listed subscribers fail simultaneously.
- **`systemic_spof` cross-layer insights** identify components whose removal would fragment the graph at multiple architectural levels simultaneously. These are the highest-priority candidates for active redundancy (replica sets, failover routing).
- **`compound_critical` cross-layer insights** identify components that appear as architectural liabilities across more than one layer. A component that is `CRITICAL` at the service level *and* the system level has no layer-scoped mitigation path — the risk is pervasive.
- **`layer_concentration` insights** flag architectural tiers where risk is not distributed. A middleware layer with 40 % `CRITICAL` brokers indicates a design pattern (hub-and-spoke, single broker cluster) rather than individual component problems.
- **Negative assortativity** (shown in `S(G)`) indicates hub-and-spoke topology — a few highly-critical hubs surrounded by many leaf-level consumers.
- **`connectivity_health`** of `AT_RISK` means one or more articulation points exist; `HEALTHY` means no SPOFs were detected.

---

## 16. What Comes Next

Step 2 produces structural metrics and deterministic RMAV quality scores Q_RMAV(v). These rule-based scores represent the baseline criticality of each component.

To generalize these predictions beyond closed-form rules (e.g. learning nonlinear multi-hop motifs and predicting direct edge-level criticalities), the system uses an inductive Graph Neural Network in Step 3.

---

← [Step 1: Import](graph-model.md) | → [Step 3: Predict](prediction.md)