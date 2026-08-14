# Step 2: Analyze — Structural Metrics

**Compute every component's structural fingerprint — the set of numbers that explain how it can fail and who it takes down with it.**

← [Step 1: Import](graph-model.md) | → [Step 3: Predict](prediction.md)

---

## Table of Contents

1. [What This Step Does](#1-what-this-step-does)
2. [Analysis Pipeline](#2-analysis-pipeline)
3. [Layer Projections](#3-layer-projections)
4. [Cross-Layer Analysis](#4-cross-layer-analysis)
5. [Topological Analysis Flow](#5-topological-analysis-flow)
6. [Why Multiple Metrics?](#6-why-multiple-metrics)
7. [Metric Taxonomy](#7-metric-taxonomy)
8. [Normalization](#8-normalization)
9. [Formal Definitions](#9-formal-definitions)
   - [9.1 Reverse PageRank (RPR)](#91-reverse-pagerank-rpr)
   - [9.2 In-Degree (DG_in)](#92-in-degree-dg_in)
   - [9.3 Multi-Path Coupling Index (MPCI)](#93-multi-path-coupling-index-mpci)
   - [9.4 Fan-Out Criticality (FOC)](#94-fan-out-criticality-foc)
   - [9.5 Betweenness Centrality (BT)](#95-betweenness-centrality-bt)
   - [9.6 QoS-Weighted Out-Degree (w_out)](#96-qos-weighted-out-degree-w_out)
   - [9.7 Clustering Coefficient (CC)](#97-clustering-coefficient-cc)
   - [9.8 Directed AP Score (AP_c_directed)](#98-directed-ap-score-ap_c_directed)
   - [9.9 Bridge Ratio (BR)](#99-bridge-ratio-br)
   - [9.10 Connectivity Degradation Index (CDI)](#910-connectivity-degradation-index-cdi)
   - (9.11, 9.12 retired — REV, RCL removed entirely; see the note preceding §9.13)
   - [9.13 QoS-Weighted In-Degree (w_in)](#913-qos-weighted-in-degree-w_in)
   - [9.14 Path Complexity (PC)](#914-path-complexity-pc)
   - [9.15 Diagnostic Metrics](#915-diagnostic-metrics)
10. [Metric Catalogue Reference](#10-metric-catalogue-reference)
11. [Analyze Stage — Rule-Based RM Scoring](#11-analyze-stage--rule-based-rm-scoring)
    - 11.1 [The Two Quality Dimensions](#111-the-two-quality-dimensions)
    - 11.2 [RM Formulas](#112-rm-formulas)
      - [Reliability R(v)](#reliability-rv--hierarchical-composite)
      - [Fault Tolerance FT(v)](#fault-tolerance-ftv--fault-propagation-risk)
      - [Maintainability M(v)](#maintainability-mv--coupling-complexity)
      - [Availability A(v)](#availability-av--spof-risk)
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

## 1. What This Step Does

Analysis takes the layer-projected dependency graph **G_analysis(l)** produced by Step 1 and computes a structural metric vector **M(v)** for every component. Each metric captures a structurally independent aspect of how a component is embedded in the system topology — how broadly its failure would propagate, whether removing it would partition the graph, how it is coupled to its neighbors, and how quickly faults could travel through it.

```
G_analysis(l)          StructuralAnalyzer           Output
(DEPENDS_ON graph)  →  13 RM-input metrics     →   M(v) per component
                        7 diagnostic metrics         S(G) graph summary
                        8 raw/derived counts
                        — all stored in M(v) —
```

This 13/7/8 split covers the core topology + resilience + pub-sub metrics in [§10](#10-metric-catalogue-reference)'s first table; code-quality, infrastructure, and weight fields are catalogued separately in that section. `saag/core/metric_registry.py` is the machine-checked source of truth for which of every `StructuralMetrics` field is scored, detection-only, a GNN feature, or descriptive-only — `tests/test_metric_registry.py` fails if it drifts from the dataclass.

Reverse Eigenvector (REV) and Reverse Closeness (RCL) were RM-input metrics before Vulnerability/Security was retired; both are now removed entirely (see the note preceding [§9.13](#913-qos-weighted-in-degree-w_in)). QoS-Weighted In-Degree (`w_in`) was retired from that same V(v) role but not deleted — it was repurposed as a Topic-only FT(v) input and remains RM-input, not diagnostic (see [§11.4](#114-metric-orthogonality)).

**Scope of this step:** M(v) contains structural observations only. Criticality scores are computed in the **RM sub-phase** of this same step ([§11](#11-analyze-stage--rule-based-rm-scoring)), which consumes M(v) and applies AHP-derived weights to produce criticality predictions Q(v). The steps are kept separate to preserve the prediction–simulation independence guarantee: structural features must not be contaminated by simulation outcomes.

---

## 2. Analysis Pipeline

There are **two entry paths** into structural analysis, and both funnel through the same `AnalysisService.analyze_layers()` method. Knowing which path you are on tells you whether criticality scores will be present in the result.

**Path A — structural only** (`cli/analyze_graph.py`, `saag.Client.analyze`, `Pipeline.analyze`):

```
cli/analyze_graph.py                    ← CLI entry point; flags: --layer, --output
│
├── saag.Client.analyze(layer)
│      Thin façade — wires dependencies, returns AnalysisResult
│
└── saag.usecases.analyze_graph.AnalyzeGraphUseCase.execute(layer)
      │
      └── saag.analysis.service.AnalysisService.analyze_layer(layer)
            └── AnalysisService.analyze_layers([layer])
                  ├── IGraphRepository.derive_dependencies()  ← pre-analysis hook (§2.1)
                  ├── IGraphRepository.get_graph_data()       ← load components & edges
                  └── StructuralAnalyzer.analyze(graph_data, layer)   → per layer
```

The result is a `LayerAnalysisResult` with **only `.structural` populated**; `.quality`, `.problems`, and `.explanation` are `None`/empty. This is the Independence Guarantee — structural features must not be contaminated by scoring or simulation outcomes (enforced by `tests/test_usecases.py`).

**Path B — structural + RM + anti-patterns** (`saag --analyze`, the REST API):

```
api/routers/analysis.py  /  cli/common/dispatcher.py:dispatch_analyze
│
└── saag.usecases.multi_layer_analysis.MultiLayerAnalysisUseCase.execute(layers, …)
      ├── AnalysisService.analyze_layers(layers)      ← same structural pass as Path A
      ├── PredictionService.predict_quality(…)        ← RM scoring  (§11)
      ├── AntiPatternDetector.detect(…)               ← smell detection
      ├── GNNService.predict(…)                       ← optional, when --gnn-model is given
      └── compute_cross_layer_insights(results)       ← cross-layer correlation (§4)
```

`StructuralAnalyzer.analyze` returns a `StructuralAnalysisResult` on both paths:

| Field | Contents |
|---|---|
| `.layer` | The `AnalysisLayer` that was analysed |
| `.components` | `Dict[id, StructuralMetrics]` — **M(v)** |
| `.edges` | `Dict[(src,tgt), EdgeMetrics]` |
| `.graph_summary` | `GraphSummary` — **S(G)** |
| `.graph` | `nx.DiGraph`, retained for visualization and GNN feature extraction |
| `.qos_profile` | QoS distribution across topics |
| `.rcm_order` | Bandwidth-minimized node order (RCM) |

`AnalysisResult` (returned by `client.analyze()`) wraps `LayerAnalysisResult` as `.raw`. The CLI's `--output` flag calls `result.save(path)` to persist the full JSON.

### 2.1 Pre-Analysis Hook

Before structural analysis begins, `AnalysisService.analyze_layers` triggers the pre-analysis hook `self.repository.derive_dependencies()`, which derives the `DEPENDS_ON` edges and establishes their weights. This ensures structural metrics are computed on a fresh, fully-derived topology projection.

The hook and the graph load run **once per call**, not once per layer — analysing all four layers derives dependencies a single time and reuses the loaded `GraphData` for every projection.

---

## 3. Layer Projections

Every call to `analyze()` targets exactly one **analysis layer** (π_l). The layer determines which component types appear in the subgraph and which DEPENDS_ON subtypes are included as edges. This is the same projection defined in Step 1:

| Layer flag | Layer name | Analyzed types | Dependency types | Quality focus |
|-----------|------------|---------------|-----------------|---------------|
| `app` | Application Layer | Application, Library | `app_to_app`, `app_to_lib` | Reliability |
| `infra` | Infrastructure Layer | Node | `node_to_node` | Availability |
| `mw` | Middleware Layer | Broker *(Apps & Nodes in subgraph to preserve edges)* | `app_to_broker`, `node_to_broker`, `broker_to_broker` | Maintainability |
| `system` | Complete System | Application, Broker, Node, Topic, Library | All six subtypes | Overall |

**Scope constraint (app layer):** The `app` layer includes `app_to_lib` edges so shared-library blast-radius is visible without requiring `--layer system`. A Library used by N apps has DG_in = 0 without these edges.

**Middleware layer note:** The `mw` subgraph includes Application and Node vertices only to preserve incoming edges to Brokers. Only Broker components appear in M(v) and S(G) results.

**The `all` shorthand** expands to `["app", "infra", "mw", "system"]` and runs all four layers. When `--output` is combined with `all` or a comma-separated list, `cli/analyze_graph.py` names its output files `<base>_<layer>.<ext>` (e.g. `metrics_app.json`, `metrics_system.json`).

**Layer aliases** accepted by `AnalysisLayer.from_string()`:

| Alias | Resolves to | Notes |
|-------|------------|-------|
| `application` | `app` | Legacy alias |
| `infrastructure` | `infra` | Legacy alias |
| `middleware`, `mw-app`, `mw-infra`, `broker`, `brokers`, `app_broker` | `mw` | Legacy aliases |
| `complete` | `system` | Legacy alias |
| `all` | `system` (via `from_string`) | Special CLI handling: `--layer all` expands to run all four layers sequentially |

---

## 4. Cross-Layer Analysis

When more than one layer is analysed on **Path B** ([§2](#2-analysis-pipeline)), `compute_cross_layer_insights()` ([saag/analysis/cross_layer.py](../saag/analysis/cross_layer.py)) correlates the per-layer results and derives **cross-layer insights** — observations that only become visible by comparing two or more layers.

> **Path A does not produce insights.** Cross-layer insights need RM criticality levels, which the structural-only path never computes. Use `saag --analyze --layers app,system` or the `/api/v1/analysis/layer/{layer}` endpoint; `python cli/analyze_graph.py --layer all` emits structural metrics per layer and nothing else.

### 4.1 What cross-layer insights capture

A component that is `CRITICAL` in the `app` layer alone is a service-level reliability risk. The same component also classified as `CRITICAL` in the `infra` layer means its physical host is simultaneously a structural SPOF — an entirely different failure mode. No single-layer analysis surfaces this compound risk; only the multi-layer view can.

Three insight types are produced:

| Insight type | Trigger | Severity |
|---|---|---|
| `compound_critical` | Component is `CRITICAL` or `HIGH` in ≥ 2 distinct layers | `CRITICAL` if any layer classifies it CRITICAL, else `HIGH` |
| `systemic_spof` | Component is an articulation point (`AP_c_directed > 0`) in ≥ 2 distinct layers | `CRITICAL` |
| `layer_concentration` | A single layer has > 30 % of its analysed components classified `CRITICAL` | `HIGH` |

### 4.2 How the correlation works

After the layer results are assembled, `compute_cross_layer_insights()` builds a component-indexed map across all `LayerAnalysisResult` objects:

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

### 4.3 Data model

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

### 4.4 Layer membership semantics

A component appears in a given layer only if its type is in that layer's `analyze_types`. This means:

- A `Broker` node can appear in both `mw` results (it is the sole analyzed type) and `system` results — so a broker that is CRITICAL in both layers would produce a `compound_critical` insight.
- An `Application` never appears in `infra` or `mw` results, so no cross-layer signal is possible between those two layers for application nodes. Cross-layer signals for applications are limited to `app` ↔ `system`.
- Nodes (`Node` type) can appear in `infra` and `system`, making infrastructure-level SPOFs detectable across both views.

---

## 5. Topological Analysis Flow

`StructuralAnalyzer.analyze()` runs seven phases in a fixed order. Each phase is a private method of the same name, so this diagram and the code can be checked against each other:

```
Phase 1  extract_layer_subgraph()
         │  Filter graph_data by layer's component_types and dependency_types
         │  Build nx.DiGraph G with node attrs: component_type, name, weight,
         │    subscriber_count, loc, cyclomatic_complexity, coupling_*,
         │    lcom, ip_address, cpu_cores, ...
         │  Build G_rev = G.reverse()      (transposed — failure-propagation direction)
         │  Build G_dist = build_distance_graph(G)  (inverted weights, distance semantics)
         │  Empty subgraph → _empty_result(layer) short circuit

Phase 2  _compute_centrality(G, G_rev, G_dist, n)      → _Centrality
         │  PageRank(G, d=0.85, weight)              → pagerank (PR)
         │  PageRank(G_rev, d=0.85, weight)          → reverse_pagerank (RPR)
         │  betweenness_centrality(G_dist, weight)   → betweenness (BT)
         │  harmonic_centrality(G) / (n-1)           → closeness (CL)
         │  _safe_eigenvector(G)                     → eigenvector (EV)
         │    fallback chain: eigenvector → Katz(α=0.01) → zeros
         │  edge_betweenness_centrality(G_dist)      → per-edge betweenness

Phase 3  _compute_coupling(G, n)                       → _Coupling
         │  MPCI(v) = Σ max(path_count(e)-1, 0) / (n-1) over InEdges(v)
         │  path_complexity(v) = mean(log2(1+path_count(e))) over OutEdges(v)
         │  FOC(t) = log1p(f(t)) × s(t) / max_t[log1p(f(t)) × s(t)]  (Topic nodes only)
         │    where f(t) = topic message frequency in Hz, s(t) = subscriber count

Phase 4  _compute_continuous_ap_scores(G)  +  _compute_reachability(G)
         │  For each node v:
         │    AP_c_out(v) = 1 - |largest_CC(G_undirected \ v)| / (n-1)
         │    AP_c_in(v)  = 1 - |largest_CC(G_T_undirected \ v)| / (n-1)
         │    AP_c_directed(v) = max(AP_c_out, AP_c_in)
         │    CDI(v) = min((avg_L(G\v) - avg_L(G)) / avg_L(G), 1.0)
         │    blast_radius(v)  = |descendants(v)|
         │    cascade_depth(v) = longest path in the ego-subgraph rooted at v
         │  Optimization: for |V| > 300 the CDI BFS uses the top-50
         │  highest-degree "core" nodes (Application, Broker, Node),
         │  ranked by in+out degree — deterministic, no randomness

Phase 5  _compute_resilience(G)                         → _Resilience
         │  clustering_coefficient   via nx.clustering(U)
         │  is_articulation_point    via nx.articulation_points(U)
         │  is_directed_ap           via _compute_directed_articulation_points(G)
         │  bridges                  via nx.bridges(U), counted per incident node
         │  (Disconnected graphs: AP/bridge detection runs per connected component)

Phase 6  _compute_pubsub_metrics()  +  _collect_qos_profile()
         │  Build bipartite app-topic graph from PUBLISHES_TO / SUBSCRIBES_TO edges
         │    pubsub_degree      = degree in bipartite graph / max_degree
         │    pubsub_betweenness = betweenness_centrality(bipartite graph)
         │    broker_exposure    = avg distinct brokers routing app's topics / max
         │    publisher_spof     = max(w(t) × min(sub_count(t)/5, 1)) over sole-pub topics
         │  Aggregate durability, reliability, transport_priority distributions
         │    across all Topic nodes — consumed by QualityAnalyzer for weight adjustment

Phase 7  _build_component_metrics()  +  _build_edge_metrics()  +  _build_summary()
         │  Assemble StructuralMetrics per node (only for types_to_analyze)
         │  _compute_code_quality_metrics():
         │    Min-max normalize loc, cyclomatic_complexity, lcom independently
         │    per Application population and per Library population
         │    CQP = 0.10·loc_norm + 0.35·complexity_norm + 0.30·instability_code + 0.25·lcom_norm
         │  Assemble EdgeMetrics per edge with at least one analysed endpoint
         │  _rcm_order(): reverse_cuthill_mckee for bandwidth minimization
         │  _build_summary() → GraphSummary S(G)
```

> **Weight semantics reminder:** Edge `weight` on DEPENDS_ON edges represents dependency *strength* (importance). PageRank, Eigenvector, and Katz use weights directly. Distance-based algorithms (Betweenness, CDI path length) use inverted weights (`1/w`) so that high-QoS dependencies are treated as "close" — the algorithm preferentially routes through critical edges.

---

## 6. Why Multiple Metrics?

No single metric captures all aspects of structural criticality. Two components illustrate why:

- **Component A** has many transitive dependents (high RPR) but sits in a well-connected, redundant subgraph (low BT, AP_c_directed = 0, BR ≈ 0). It is a broad reliability risk but not a SPOF.
- **Component B** has few direct dependents (low RPR) but is the single connection between two graph clusters (AP_c_directed = 0.82, BR = 1.0). It is a structural single point of failure despite low blast radius.

A single metric misclassifies both. The fourteen Tier-1 metrics (listed once, in [§10](#10-metric-catalogue-reference)) are drawn from four different theoretical families — random walk, local topology, resilience, and QoS-weighted degree — and together produce a complete and orthogonal structural fingerprint.

---

## 7. Metric Taxonomy

Every field in M(v) belongs to exactly one of three tiers. This taxonomy is the key to understanding which fields feed which later computation. The `Tier` column of [§10](#10-metric-catalogue-reference) assigns every field to one of these:

| Tier | Purpose |
|------|---------|
| **Tier 1 — RM inputs** | Directly feed FT(v), M(v), or A(v) in the RM sub-phase ([§11](#11-analyze-stage--rule-based-rm-scoring)) — R(v) is a declared blend of FT and A, not a direct metric consumer |
| **Tier 2 — Diagnostic** | Computed for visualization, output reports, and GNN features; do not feed RM formulas |
| **Tier 3 — Raw / inline-derived** | Integer counts and inline-derived scalars used only *within* RM formulas; not stored as normalized metrics |

**Why PR, CL, EV are Tier 2:** The *forward* variants (PageRank, Closeness, Eigenvector) measure how much a component itself is influenced by others — they are informative for dependency visualization but do not directly capture failure propagation outward. Their reverse counterpart RPR, computed on G^T, captures how failures at v spread to v's dependents — the reliability-relevant direction, and is the only reverse-centrality metric an RM formula still reads. (REV and RCL, the reverse-direction counterparts of CL and EV, were retired along with V(v) and are no longer computed at all — see the note preceding [§9.13](#913-qos-weighted-in-degree-w_in).) Computing all of PR/CL/EV gives the full picture for dashboards while FT(v) uses only RPR.

**Why pubsub_degree, pubsub_betweenness, broker_exposure are Tier 2:** These are computed on the raw bipartite app-topic graph (using PUBLISHES_TO / SUBSCRIBES_TO edges, not DEPENDS_ON edges). They enrich the SMART visualization dashboard and serve as GNN features, but the RM formulas operate on the DEPENDS_ON graph where the same information is captured via DG_in, BT, and RPR respectively.

---

## 8. Normalization

All Tier 1 metrics are normalized to [0, 1] before being consumed by the RM sub-phase ([§11](#11-analyze-stage--rule-based-rm-scoring)). The **default method is `robust` normalization** (rank-based scaling):

```
x_robust(v) = rank(v) / (|V| − 1)

rank(v) = position of v when all components sorted by ascending x(v)
        (0-based; average-rank tie-breaking)
```

> **Note on terminology:** The `--norm robust` flag performs rank-based normalization, not IQR scaling as the term "robust" might suggest. This preserves ordinal relationships and is robust to outliers.

**Why rank normalization (default):** Min-max normalization is sensitive to outliers. In a system with one highly-central hub and 50 peripheral nodes, min-max assigns 1.0 to the hub and compresses all other values near 0 — the relative ordering among peripherals is lost. Rank normalization preserves the full ordinal structure regardless of extreme values. This is particularly important for betweenness centrality, which is typically sparse (most nodes have BT near 0, one or two have very high BT).

> [!IMPORTANT]
> **Measured (re-run under the RM model): normalization method barely matters here, and the default is no longer the worst option.** Sweeping the method across the seven scenarios against I*(v) ([results/threshold_sensitivity.json](../results/threshold_sensitivity.json), produced by `PYTHONPATH=. python reproduce/threshold_sensitivity.py --skip-thresholds`):
>
> | `--norm` | `robust` (rank, default) | `minmax` | `zscore` |
> |---|---|---|---|
> | mean ρ | **−0.0188** | **−0.0346** | **−0.0346** |
>
> This supersedes the pre-migration finding, which reported a +0.195 ρ gap favouring `minmax`/`zscore` over the `robust` default. Under the RM model's smaller composite (Q = 0.80R + 0.20M, R itself an α-blend rather than a direct 3-term sum) the spread across methods shrank to 0.0158, and `robust` is now the *least* negative of the three, not the worst. All three are near-zero here for the same reason the closed-form Q(v) scores are near-zero throughout this section — see [§11.6](#116-weight-shrinkage-strategy)'s scope note. Do not carry the old "+0.195, minmax wins" claim forward.
>
> A second consequence worth noting: because rank-normalized inputs are near-uniform on [0,1] by construction, the box-plot classifier in [§11.7](#117-criticality-classification) produces a fairly stable CRITICAL fraction almost regardless of topology, which is the likely explanation for the narrow "typical distribution" band reported there.

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

### 8.1 Normalization Caveats (Hardening Phase)

**1. Solitary Populations (Single-Node Layers):**
If a layer or node type contains only a single component (e.g., one core Library), the min-max span is zero. To preserve the intrinsic complexity signal, the system defaults to a normalized value of **1.0** (most critical) for that component rather than zeroing it out. This ensures large singleton components are still flagged for maintenance risk.

**2. Type-Split Normalization:**
Applications and Libraries are normalized as separate populations before being mixed in the $M(v)$ Maintainability dimension. This prevents a massive legacy monolithic application from compressing the complexity signal of all libraries to near-zero. However, this means a "0.80 complexity" Application is not directly comparable to a "0.80 complexity" Library in absolute terms.

**3. Library Ca/Ce Semantics:**
For Library nodes, `instability_code` uses static analysis coupling (CBO/Fan-in/Fan-out) rather than topological `DEPENDS_ON` edges. This captures the internal stability of the package logic, whereas topological coupling captures system-level blast radius.

**Optional winsorization:** Before rank normalization, raw values above the 95th percentile can be capped (`--winsorize`). This prevents a single extreme outlier from being ranked above all others while the 2nd–99th percentile occupy a single rank bucket.

---

## 9. Formal Definitions

All definitions below operate on G_analysis(l) — the layer-projected DEPENDS_ON graph with QoS-derived edge weights. G^T denotes the transposed graph (all edge directions reversed).

### 9.1 Reverse PageRank (RPR)

*Tier 1 → R(v)*

Computes PageRank on G^T. Captures **cascade reach** — how broadly a failure at v propagates in the direction of v's dependents.

```
RPR(v) = PageRank(G^T, d=0.85)[v]

d = damping factor (0.85), max iterations = 100, tolerance = 1e-6
```

**High RPR(v) means:** Failure at v reaches a large fraction of the system through the transitive dependency chain. RPR is the primary input to the Reliability dimension R(v).

> **Directional note:** In the DEPENDS_ON graph, edges point from dependent to dependency (App_sub → App_pub). Reversing the graph therefore gives edges pointing *from* publisher to subscribers — the natural failure-propagation direction. RPR on G^T thus counts how many nodes a failure would reach if it propagated outward from v through subscribers.

**Literature Citation:** Reversing graph edges ($G^T$) in directed networks to analyze "hub" behavior or start points is referred to as **CheiRank** in complex network studies of software call graphs. See Page et al. (1999) for the foundational PageRank algorithm, and Chepelianskii (2010) or Gleich (2015) for CheiRank / Reverse PageRank applications.
- Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank Citation Ranking: Bringing Order to the Web*. Stanford InfoLab.
- Chepelianskii, A. D. (2010). *Towards physical laws for software architecture*. arXiv:1003.5455.
- Gleich, D. F. (2015). *PageRank beyond the web*. SIAM Review, 57(3), 321-378.

### 9.2 In-Degree (DG_in)

*Tier 1 → R(v)*

```
DG_in(v) = in_degree(v) / (|V| − 1)     (normalized)
```

**High DG_in(v) means:** Many components directly depend on v — the immediate blast radius if v fails. DG_in measures *local* propagation; RPR measures *global* propagation. Both are needed because a highly-central hub may have a small local blast radius but a large transitive one.

**Literature Citation:** Degree centrality is the most fundamental local measure of node importance in social and communication networks.
- Freeman, L. C. (1978). *Centrality in social networks conceptual clarification*. Social Networks, 1(3), 215-239.

### 9.3 Multi-Path Coupling Index (MPCI)

*Tier 1 → R(v)*

A **new metric** added in this version. Uses the `path_count` attribute on DEPENDS_ON edges produced by Step 1's Phase 3. For a given component v, `path_count` on each incoming edge counts the number of distinct shared topics (for app_to_app dependencies) or distinct USES edges (for app_to_lib dependencies) that independently establish that dependency.

```
MPCI(v) = Σ_{e ∈ InEdges(v)} max(path_count(e) − 1, 0) / (|V| − 1)

InEdges(v) = set of incoming DEPENDS_ON edges to v
path_count(e) = number of topics (or USES edges) jointly establishing edge e
```

**Why `path_count − 1`:** A dependency with `path_count = 1` is a single coupling — baseline. Each additional shared topic is an *extra* coupling vector. MPCI sums these extra vectors across all dependents.

**High MPCI(v) means:** Multiple components are coupled to v through redundant shared channels. Each channel is an independent failure vector for those dependents. This amplifies the cascade depth that CDPot (a derived term, [§11.3](#113-derived-terms)) estimates: when a dependency collapses, it does so across all shared channels simultaneously rather than one path at a time.

```
MPCI(v) = 0    → all incoming dependencies are single-channel (baseline)
MPCI(v) > 0    → v has multi-channel coupling; higher values = greater coupling intensity
```

> **Library nodes benefit most from MPCI:** After Step 1's Rule 5 (app_to_lib), libraries now appear as DEPENDS_ON targets. A library used by 10 applications via a single USES edge each has high DG_in but MPCI = 0 (single-channel per dependency). The MPCI signal is non-zero only when the same (App, Lib) pair has multiple USES edges — currently rare — or when (App, App) pairs share multiple topics.

**Literature Citation:** While MPCI is a project-derived metric capturing runtime topic redundancy, structural coupling metrics in software engineering trace back to foundational work on structural information flow and Object-Oriented coupling complexity.
- Henry, S., & Kafura, D. (1981). *Software structure metrics based on information flow*. IEEE Transactions on Software Engineering, (5), 510-518.
- Chidamber, S. R., & Kemerer, C. F. (1994). *A metrics suite for object oriented design*. IEEE Transactions on Software Engineering, 20(6), 476-493.

### 9.4 Fan-Out Criticality (FOC)

*Tier 1 → R(v) for Topic nodes*

A **new metric** added in this version. Topics are not endpoints of DEPENDS_ON edges, so their DG_in and RPR in the dependency graph are 0. FOC provides a reliability signal for Topic nodes by using the `subscriber_count` attribute written by Step 1's Phase 2 fan-out augmentation, combined with topic frequency for QoS-aware weighting.

```
FOC(t) = log1p(f(t)) × s(t) / max_{t' ∈ V_topic}[log1p(f(t')) × s(t')]   for Topic nodes
FOC(v) = 0                                                                   for all other types
```

where `f(t)` = topic message frequency in Hz, `s(t)` = subscriber count.

**High FOC(t) means:** Topic t is a data distribution relay for many subscribers at high message rate. If t becomes unreachable (broker failure, routing failure), all subscribers simultaneously lose their data source. The `log1p` compression handles large frequency variance while preserving monotonicity.

> **Usage in FT(v) for Topics:** In the RM sub-phase, when computing FT(v) for a Topic node, the `DG_in` term is replaced with `FOC` because the dependency graph gives Topics no in-degree. The CDPot term uses `FOC` as the reach signal in place of `DG_in` for these nodes.
>
> **Layer restriction:** FOC is non-zero only when `--layer system` is used. Topic nodes are excluded from the `app` and `mw` subgraphs. The CLI will emit a warning when the active layer has no Topic nodes.

**Literature Citation:** Outbound information flow complexity and its impact on architectural modularity is defined in:
- Henry, S., & Kafura, D. (1981). *Software structure metrics based on information flow*. IEEE Transactions on Software Engineering, (5), 510-518.

### 9.5 Betweenness Centrality (BT)

*Tier 1 → M(v)*

```
BT(v) = Σ_{s≠v≠t} σ(s,t|v) / σ(s,t)   (Brandes' algorithm, O(|V|×|E|))

σ(s,t) = number of shortest paths from s to t
σ(s,t|v) = number of those paths passing through v

Normalized by (|V|−1)(|V|−2). Shortest paths use inverted weights (1/w) as distances.
```

**High BT(v) means:** v is a structural bottleneck — many dependency chains route through it. Changes to v risk disrupting many other components. The inversion of weights for distance computation means that high-QoS edges (strong dependencies) contribute less to the shortest-path distance — the algorithm preferentially routes through critical edges, making BT sensitive to high-weight dependency chains.

**Literature Citation:** Betweenness centrality was defined by Freeman (1977). This implementation leverages Brandes' fast $O(|V||E|)$ algorithm for weighted networks.
- Freeman, L. C. (1977). *A set of measures of centrality based on betweenness*. Sociometry, 35-41.
- Brandes, U. (2001). *A faster algorithm for betweenness centrality*. Journal of Mathematical Sociology, 25(2), 163-177.

### 9.6 QoS-Weighted Out-Degree (w_out)

*Tier 1 → M(v)*

```
w_out(v) = Σ_{(v,u) ∈ OutEdges(v)} weight(v,u)    (raw sum, then rank-normalized)
```

**High w_out(v) means:** v depends on many high-priority components. Each outgoing dependency is an efferent coupling; high-QoS couplings amplify change risk because a change to any dependency propagates back to v via its SLA obligations.

### 9.7 Clustering Coefficient (CC)

*Tier 1 → M(v)*

```
CC(v) = |{(u,w) ∈ E_undirected : u,w ∈ N(v)}| / (deg(v) × (deg(v) − 1))

Computed on undirected projection of G_analysis(l). CC(v) = 0 if deg(v) < 2.
```

**High CC(v) means:** v's neighbors are well-connected among themselves — the local topology is redundant. Low CC (via the `1 − CC` term in M(v)) indicates each of v's couplings is unique and non-redundant, making v harder to safely modify.

**Literature Citation:** Introduced by Watts and Strogatz (1998) to characterize network clustering and small-world properties.
- Watts, D. J., & Strogatz, S. H. (1998). *Collective dynamics of ‘small-world’ networks*. Nature, 393(6684), 440-442.

### 9.8 Directed AP Score (AP_c_directed)

*Tier 1 → A(v). Stored in M(v) (previously inline-computed during RM scoring).*

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

**Literature Citation:** Graph articulation points, biconnectivity, and depth-first search (DFS) structural decomposition algorithms are defined in Tarjan (1972).
- Tarjan, R. (1972). *Depth-first search and linear graph algorithms*. SIAM Journal on Computing, 1(2), 146-160.

### 9.9 Bridge Ratio (BR)

*Tier 1 → A(v)*

```
BR(v) = |{e ∈ bridges(G_undirected) : v ∈ e}| / undirected_degree(v)

bridge = edge whose removal increases the number of connected components
BR(v) = 0 if degree(v) = 0
```

**High BR(v) means:** A large fraction of v's connections are non-redundant bridges. Losing any bridge edge disconnects a subgraph from the rest of the system.

**Literature Citation:** Graph bridges (or cut-edges) represent the most critical single links between subgraphs.
- Tarjan, R. (1972). *Depth-first search and linear graph algorithms*. SIAM Journal on Computing, 1(2), 146-160.

> **Note:** BR(v) describes a *node's* exposure to bridge edges, not a per-edge score. For the direct definition of relationship (edge) criticality, see [criticality.md §5](criticality.md#5-relationship-edge-criticality).

### 9.10 Connectivity Degradation Index (CDI)

*Tier 1 → A(v). Stored in M(v) (previously inline-computed during RM scoring).*

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

**Literature Citation:** Connectivity degradation upon node/edge percolation is a classic benchmark for network error and attack tolerance.
- Albert, R., Jeong, H., & Barabási, A. L. (2000). *Error and attack tolerance of complex networks*. Nature, 406(6794), 378-382.
- Callaway, D. S., Newman, M. E., Strogatz, S. H., & Watts, D. J. (2000). *Network robustness and fragility: Percolation on random graphs*. Physical Review Letters, 85(25), 5468.

*(§9.11 Reverse Eigenvector Centrality and §9.12 Reverse Closeness Centrality — REV and RCL — were Tier 1 → V(v) inputs before Vulnerability/Security was retired. They were reclassified to Tier 2 and kept for reference for a time; both are now fully removed from `StructuralMetrics`, the normalization tables, and the SMART dashboard — nothing computes or reads them. This numbering gap is intentional, the same convention `validation.md` uses for its retired G7/G9 gates.)*

### 9.13 QoS-Weighted In-Degree (w_in)

*Tier 1 → FT(v), Topic nodes only. `StructuralMetrics.dependency_weight_in` was the QADS (QoS-weighted Attack-Dependent Surface) Tier 1 → V(v) input before Vulnerability/Security was retired; the field was not retired with it — it was repurposed as `publisher_norm` in the Topic branch of FT(v) (§11.2). Non-Topic types read 0.0.*

```
w_in(v) = Σ_{(u,v) ∈ InEdges(v)} weight(u,v)    (raw sum, then rank-normalized)
```

For a Topic, `w_in` is the QoS-weighted count of its publishers. Used in `CDPot_topic(v) = FOC(v) × (1 − min(w_in_norm(v), 1))`: a topic with many publishers has redundant sources, so losing one publisher degrades it less than a sole-publisher topic — `w_in` is what makes that redundancy discount possible.

### 9.14 Path Complexity (PC)

*Tier 1 → M(v)*

```
PC(v) = mean( log2(1 + path_count(e)) ) over e ∈ OutEdges(v)
```

**High PC(v) means:** v depends on other components through multiple redundant paths (shared topics). While this adds reliability, it increases the **Maintainability** risk (M) because change impact propagation follows all available paths. A change to v's logic may require complex re-synchronization across all paths mediating its efferent dependencies. PC serves as an intensifier for the **Coupling Risk** term ([§11.3](#113-derived-terms)).


### 9.15 Diagnostic Metrics

*Tier 2 — computed for visualization and GNN features; do not feed RM formulas*

| Metric | Definition | Purpose |
|--------|-----------|---------| 
| PageRank (PR) | Standard PageRank on G | Forward importance; shows which components accumulate the most transitive dependency weight |
| Closeness (CL) | Harmonic closeness on G | Forward propagation speed, for dashboards |
| Eigenvector (EV) | Eigenvector centrality on G | Forward influence through neighbors |
| pubsub_degree | Degree in bipartite app-topic graph | Topic diversity of an application — how many distinct message channels it participates in |
| pubsub_betweenness | Betweenness in bipartite app-topic graph | Applications that bridge separate topic clusters |
| broker_exposure | Avg distinct brokers routing app's topics | Infrastructure blast surface — how many brokers an application's failure would stress |
| publisher_spof (PSPOF) | `max(w(t) × min(sub_count(t)/5, 1))` over sole-published topics | Sole-publisher risk: if this application is the only publisher on a topic and that topic has active subscribers, PSPOF quantifies the blast if the application goes silent. Available in M(v) for dashboards and GNN features. |

---

## 10. Metric Catalogue Reference

Complete M(v) field listing — the single authoritative index. Every field has a tier ([§7](#7-metric-taxonomy)), an RM dimension (or "—" for Tier 2/retired), and a direction (↑ = higher is worse / more critical, ↓ = higher is better). Tier-1 symbols link to their formal definition in [§9](#9-formal-definitions).

| Field | Symbol | Tier | RM Dim | Dir | Notes |
|-------|--------|------|----------|-----|-------|
| `reverse_pagerank` | [RPR](#91-reverse-pagerank-rpr) | 1 | FT | ↑ | |
| `in_degree` | [DG_in](#92-in-degree-dg_in) | 1 | FT | ↑ | |
| `mpci` | [MPCI](#93-multi-path-coupling-index-mpci) | 1 | FT | ↑ | Enters FT(v) via CDPot_enh |
| `fan_out_criticality` | [FOC](#94-fan-out-criticality-foc) | 1 | FT | ↑ | Topic nodes only; substitutes for DG_in |
| `betweenness` | [BT](#95-betweenness-centrality-bt) | 1 | M | ↑ | |
| `dependency_weight_out` | [w_out](#96-qos-weighted-out-degree-w_out) | 1 | M | ↑ | |
| `clustering_coefficient` | [CC](#97-clustering-coefficient-cc) | 1 | M | ↓ | Used as 1−CC in M(v) |
| `path_complexity` | [PC](#914-path-complexity-pc) | 1 | M | ↑ | Enters M(v) via CouplingRisk_enh |
| `ap_c_directed` | [AP_c_dir](#98-directed-ap-score-ap_c_directed) | 1 | A | ↑ | Also a factor in QSPOF |
| `bridge_ratio` | [BR](#99-bridge-ratio-br) | 1 | A | ↑ | |
| `cdi` | [CDI](#910-connectivity-degradation-index-cdi) | 1 | A | ↑ | |
| `weight` | w | 1 | A | ↑ | Component QoS weight from Step 1; factor in QSPOF |
| `dependency_weight_in` | [w_in](#913-qos-weighted-in-degree-w_in) | 1 | FT | ↑ | Topic nodes only — publisher_norm in Topic FT(v); 0.0 for all other types |
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
| `is_articulation_point` | — | 3 | — | — | Binary AP flag (undirected articulation detection) |
| `is_directed_ap` | — | 3 | — | — | Binary directed-articulation flag (used in `systemic_spof` detection) |
| `blast_radius` | — | 3 | — | — | Number of descendants reachable from v |
| `cascade_depth` | — | 3 | — | — | Longest failure propagation path from v |
| `topic_frequency_hz` | — | 3 | — | — | Raw message rate in Hz (Topic nodes; 0.0 otherwise) |

Tier-2 diagnostics are described in [§9.15](#915-diagnostic-metrics).

**Code quality metrics** (Application and Library nodes only; 0.0 for all other types):

| Field | Tier | RM Dim | Description |
|-------|------|----------|-------------|
| `loc_norm` | 1→CQP | M | Normalized lines of code (min-max within type population) |
| `complexity_norm` | 1→CQP | M | Normalized cyclomatic complexity (min-max within type population) |
| `instability_code` | 1→CQP | M | Martin instability Ce/(Ca+Ce) — already in [0,1], not re-normalized |
| `lcom_norm` | 1→CQP | M | Normalized lack of cohesion (min-max within type population) |
| `code_quality_penalty` | 1 | M | CQP v7: `0.10·loc_norm + 0.35·complexity_norm + 0.30·instability_code + 0.25·lcom_norm` |

---

## 11. Analyze Stage — Rule-Based RM Scoring

> See [criticality.md](criticality.md) for the definitions these scores operationalize: component criticality (D1), relationship criticality (D2), why criticality is a consequence rather than a risk (D3), and why the dimension names come from SQuaRE's product-quality model while the harm is measured on Quality-in-Use ([§3.5](criticality.md#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)).

### 11.1 The Two Quality Dimensions

Two ISO/IEC 25010:2023 characteristics are scored. Reliability is **hierarchical**: Fault Tolerance and Availability are its sub-characteristics, reported individually and combined via a declared blend. Vulnerability/Security is not scored at all — it was retired outright, not folded into another dimension (see [criticality.md §4.3](criticality.md#43-the-rm-model) for the full rationale).

| Dimension | Question answered | High score means | Primary stakeholder |
|-----------|------------------|-----------------|---------------------|
| **R — Reliability** (hierarchical) | How broadly does failure propagate, and is this a structural single point of failure? | Failure cascades widely and/or removing it partitions the graph | Reliability Engineer / DevOps / SRE |
| ↳ **FT — Fault Tolerance** (sub-characteristic) | How broadly and deeply does failure propagate? | Failure cascades widely and is hard to contain | Reliability Engineer |
| ↳ **A — Availability** (sub-characteristic) | Is this a structural single point of failure? | Removing it partitions the dependency graph | DevOps / SRE |
| **M — Maintainability** | How hard is this to change safely? | Tightly coupled; structural bottleneck | Software Architect |

The dimensions are deliberately **orthogonal** in metric input: each raw metric feeds exactly one dimension (see [Metric Orthogonality](#114-metric-orthogonality)). This means a component's RM breakdown tells you *why* it is critical — a pure SPOF has high A (and therefore high R) but low FT and M; a God Component has high M; a cascade hub has high FT (and therefore high R) — enabling targeted remediation instead of blanket hardening.

---

### 11.2 RM Formulas

All inputs are normalized to [0, 1] by Step 2's rank normalization unless otherwise noted. All RM scores are therefore in [0, 1]. Intra-dimension weights are stated judgements checked for AHP consistency; see [Section 11.5](#115-ahp-weight-derivation). The composite weights (`w_R`, `w_M`) and the Reliability blend (`r_alpha`) are **not** AHP-derived — see [Composite Score Q(v)](#composite-score-qv) below.

> [!IMPORTANT]
> **The weights printed below are the pre-shrinkage judgement, not the shipped defaults.**
> `AHPProcessor.compute_weights()` ([saag/analysis/weight_calculator.py#L314](../saag/analysis/weight_calculator.py#L314)) applies the shrinkage factor λ to the **intra-dimension** vectors, and the shipped default is λ = 0.70. So every formula in this section runs with weights pulled ~30 % of the way toward uniform:
>
> | Dimension | λ = 0.70 (shipped default) |
> |---|---|
> | FT(v) — RPR, DG_in, CDPot | 0.422, 0.324, 0.255 |
> | M(v) — BT, w_out, CouplingRisk, 1−CC | 0.305, 0.270, 0.144, 0.116 |
>
> The rounded values quoted in the formulas below (0.45 / 0.30 / 0.25 etc.) are the design intent at λ = 1.0, derived from the pairwise matrices in [§11.5](#115-ahp-weight-derivation); use this table when reconciling against runtime output. Reproduce either column with `AHPProcessor(shrinkage_factor=λ).compute_weights()`.
> **Unlike the old 4-D model, `Q(v)`'s composite weights (`w_R=0.80`, `w_M=0.20`) and Reliability's blend (`r_alpha=0.36`) do not move with λ at all** — they are DECLARED constants, not AHP output. See [§11.6](#116-weight-shrinkage-strategy).

---

#### Reliability R(v) — Hierarchical Composite

R(v) is not scored directly from raw metrics. It is a declared blend of its two sub-characteristics:

```
R(v) = r_alpha × FT(v) + (1 − r_alpha) × A(v)          r_alpha = 0.36
```

`r_alpha` is a DECLARED constant, algebraically derived from the retired 4-D AHP composite (`r_alpha = 0.24 / (0.24 + 0.43) = 0.3582 → 0.36`, see [Composite Score Q(v)](#composite-score-qv)) — it is **not** independently AHP-fitted, and is included in the `--equal-weights` / sensitivity-analysis perturbation exactly as `w_R`/`w_M` are (see [§11.6](#116-weight-shrinkage-strategy)).

---

#### Fault Tolerance FT(v) — Fault Propagation Risk

FT(v) measures how broadly and deeply a component's failure propagates through the DEPENDS_ON dependency graph. It is Reliability's fault-tolerance sub-characteristic.

**Standard formula (v6)** — Application, Broker, Infrastructure Node, Library:

```
FT(v) = 0.45 × RPR(v) + 0.30 × DG_in(v) + 0.25 × CDPot_enh(v)
```

| Term | Weight | What it captures |
|------|:------:|-----------------|
| RPR(v) | 0.45 | Reverse PageRank on G^T — transitive cascade reach in the failure-propagation direction (primary signal) |
| DG_in(v) | 0.30 | In-degree (normalised) — immediate blast radius; number of direct dependents |
| CDPot_enh(v) | 0.25 | Enhanced Cascade Depth Potential — depth × breadth of the cascade, amplified by multi-path coupling (see [§11.3](#113-derived-terms)) |

> **Direction argument.** DEPENDS_ON edges point `dependent → dependency` (App_sub → App_pub). A failure at v propagates *against* edge direction — to nodes that depend on v (v's in-neighbours in the DEPENDS_ON graph). RPR reverses G to produce edges `dependency → dependent`, making the failure-propagation path the natural traversal direction. RPR(v) therefore accumulates rank from the nodes v would transitively impair when it fails. Forward PageRank on the DEPENDS_ON graph measures how much accumulated weight v *receives* from its own dependencies — it captures importance as a callee, not cascade reach as a failure origin. This is the answer to "why reverse?" in the committee review.

**Topic-type formula** — used exclusively for Topic nodes:

Topic nodes have DG_in = 0 in the DEPENDS_ON graph because Topics are not DEPENDS_ON endpoints. Their fault-tolerance risk is measured instead through subscriber fan-out:

```
FT_topic(v) = 0.50 × FOC(v)  +  0.50 × CDPot_topic(v)

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

**Two ingested fields are not in this formula.** `cm_avg_cbo` and `cm_avg_rfc` are ingested, flattened onto the vertex, persisted, and shown in the UI, but no scoring code reads them — `CQP(v)` consumes only `loc`, `cyclomatic_complexity` (`avg_wmc`), `instability_code` (from `avg_fanin`/`avg_fanout`), and `lcom`. See [criticality.md §3.0](criticality.md#30-three-quality-views-internal-external-and-quality-in-use) for the fuller accounting of which internal-evidence fields actually reach a score.

**Normalisation caveat.** `loc_norm`, `complexity_norm` and `lcom_norm` are min-max normalised as **separate populations per node type** (Application and Library independently, since their LOC/complexity scales differ) — the min-max helper falls back to `1.0` for a zero-variance population, a deliberate choice for a genuine single-node population but one that also fires, indistinguishably, when a population is uniformly zero because none of its members carry `code_metrics` at all. This is not hypothetical: it is measured to occur across every real-world scenario graph in the committed corpus (see the flagged note in [criticality.md §3.0](criticality.md#30-three-quality-views-internal-external-and-quality-in-use)).

---

#### Availability A(v) — SPOF Risk

A(v) measures whether a component is a structural single point of failure, weighted by its QoS priority, bridge redundancy, and path elongation. It is Reliability's availability sub-characteristic — it is not scored as an independent peer dimension; it feeds `R(v)` via the `r_alpha` blend above.

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

#### Composite Score Q(v)

```
Q(v) = w_R × R(v)  +  w_M × M(v)          w_R = 0.80, w_M = 0.20
```

**`w_R` and `w_M` are DECLARED constants, not AHP-derived.** With only two characteristics, a composite AHP matrix would be a 2×2 Saaty matrix — consistent by construction (CR = 0 for n ≤ 2) and contributing nothing beyond whatever single free parameter is chosen. AHP is retired at the composite level; it is retained for the genuinely multi-term intra-dimension vectors (FT(v)'s 3 terms, M(v)'s 5 terms, A(v)'s 5 terms), which do have non-trivial consistency ratios — see [§11.5](#115-ahp-weight-derivation).

**Provenance.** `w_R = 0.80` and `w_M = 0.20` are a pure **re-parameterisation** of the retired 4-D AHP composite (A = 0.43, R = 0.24, M = 0.17, V = 0.16): drop Vulnerability, renormalise the rest, then fold Availability into Reliability via `r_alpha`:

```
r_alpha = 0.24 / (0.24 + 0.43) = 0.3582 → 0.36
w_R     = (0.24 + 0.43) / 0.84 = 0.7976 → 0.80
w_M     = 0.17 / 0.84           = 0.2024 → 0.20
```

At exact (unrounded) values this re-parameterisation is algebraically identical to "old `Q`, drop `V`, renormalise" — `w_R·r_alpha = 0.24/0.84`, `w_R·(1−r_alpha) = 0.43/0.84`, `w_M = 0.17/0.84`, all exactly. The 2-s.f. rounding is not free — it introduces a small, bounded drift from those exact ratios (≤ 0.003, see `tests/test_quality_model.py::TestCompositeReparameterisation`) — but the composite is a re-derivation of the old judgement, not an independently invented weighting.

**Baseline (equal weights).** `w_R = w_M = 0.5` (and `r_alpha = 0.5`) can be activated via `--equal-weights` for sensitivity analysis or reproducibility.

**Sensitivity of λ — this no longer affects the composite.** The AHP shrinkage factor λ ∈ [0, 1] blends raw AHP weights toward uniform, but since `w_R`/`w_M`/`r_alpha` are declared (not AHP output), **λ now only shrinks the intra-dimension vectors** (FT(v), M(v), A(v), Impact) — the composite is λ-invariant by construction. The measured sweep is in [§11.6](#116-weight-shrinkage-strategy) and committed to [results/ahp_shrinkage_sweep.json](../results/ahp_shrinkage_sweep.json). Read §11.6 before relying on the default — its finding (the curve is monotone increasing in λ, not flat, and the effect is small) supersedes any pre-migration "plateau" claim you may have seen cited elsewhere in this repo's history.

---

### 11.3 Derived Terms

These scalars are computed inline within the RM formulas at scoring time. They are derived from M(v) fields produced in Step 2; they are not stored as independent graph properties.

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

Each raw metric from M(v) feeds **exactly one** of the three scored leaves — Fault Tolerance, Maintainability, Availability (Reliability itself is a declared blend of FT and A, not a direct consumer of raw metrics). No metric appears in more than one formula. Violations would inflate the effective weight of shared metrics relative to the AHP calibration.

| Metric | Symbol | FT | M | A | Notes |
|--------|--------|:-:|:-:|:-:|-------|
| Reverse PageRank | RPR | ✓ | | | Global cascade reach |
| In-Degree (norm) | DG_in | ✓ | | | Immediate blast radius |
| MPCI | MPCI | ✓ via CDPot | | | Amplifier only; enters via derived term |
| Fan-Out Criticality | FOC | ✓ Topics | | | Substitutes for DG_in on Topic nodes |
| QoS In-Degree | w_in | ✓ Topics | | | publisher_norm in Topic FT(v) only; 0.0 for non-Topic types |
| Path Complexity | path_complexity | | ✓ via CouplingRisk | | Structural coupling depth |
| Betweenness | BT | | ✓ | | Structural bottleneck |
| QoS Out-Degree | w_out | | ✓ | | Priority-weighted efferent coupling |
| Code Quality Penalty | CQP | | ✓ | | Complexity + instability + LCOM |
| CouplingRisk_enh | CouplingRisk | | ✓ | | Derived from DG_in_raw, DG_out_raw |
| Clustering Coefficient | CC | | ✓ as 1−CC | | Local path redundancy |
| Directed AP Score | AP_c_directed | | | ✓ | Directly in A(v) and via QSPOF |
| Bridge Ratio | BR | | | ✓ | Non-redundant edge fraction |
| CDI | CDI | | | ✓ | Path elongation on removal |
| PageRank | PR | — | — | — | Diagnostic only (Tier 2) |
| Closeness | CL | — | — | — | Diagnostic only (Tier 2) |
| Eigenvector | EV | — | — | — | Diagnostic only (Tier 2) |

**Fully removed:** Reverse Eigenvector Centrality (REV) and Reverse Closeness Centrality (RCL) were two of `V(v)`'s three terms; neither is computed, stored, or exported any more — see the note preceding [§9.13](#913-qos-weighted-in-degree-w_in). The third term, QoS-Weighted In-Degree (`w_in`, `dependency_weight_in` on `StructuralMetrics`), was **not** retired the same way — it was repurposed as the Topic-only FT(v) input in the table above. Do not confuse it with the still-current `cm_avg_cbo`/`cm_avg_rfc` gotcha in [CLAUDE.md Known Gotchas](../CLAUDE.md#known-gotchas), which is a genuinely different case: those two fields really are computed, persisted, and read by nothing.

---

### 11.5 AHP Weight Derivation

Intra-dimension weights are expressed as pairwise comparison matrices on Saaty's 1–9 scale and checked with the **Analytic Hierarchy Process (AHP)** consistency machinery.

> [!NOTE]
> **These are stated author judgements, not an elicitation.** The matrices below were written to express a design intent and then checked for consistency; they are not the output of a panel exercise. The near-zero consistency ratios (CR ≈ 0.000–0.002 on 5×5 matrices) are a symptom of that — a matrix filled in from a target weight vector is perfectly consistent almost by construction, whereas genuine human elicitation on five criteria rarely lands below CR ≈ 0.02. Describing the result as "AHP-derived" overstates the provenance; it is a principled fixed weighting with a documented rationale and a consistency check.
>
> The empirical case for the weighting therefore rests on [§11.6](#116-weight-shrinkage-strategy), not on the elicitation — and on the cohort measured there, equal weights do better. Treat the matrices as documentation of intent.

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

#### Fault Tolerance AHP (3×3: RPR, DG_in, CDPot_enh)

```
            RPR    DG_in  CDPot
RPR      [ 1.00,  1.50,  2.00 ]   RPR: global reach is the primary cascade signal
DG_in    [ 0.67,  1.00,  1.50 ]   DG_in: immediate dependents are secondary
CDPot    [ 0.50,  0.67,  1.00 ]   CDPot: cascade depth is supplementary

→ AHP raw weights:  [0.45,  0.30,  0.25]    CR ≈ 0.001
```

MPCIs enter FT(v) indirectly through CDPot_enh and do not add a 4th AHP criterion. This preserves the 3×3 matrix and its near-zero CR while capturing the MPCI effect.

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

> **No Composite Q AHP matrix.** The pre-migration model derived `Q(v)`'s dimension weights from a 4×4 AHP matrix (A, R, M, V — CR ≈ 0.02). With only two composite terms remaining, a 2×2 matrix would be consistent by construction (CR = 0) and contribute nothing beyond whichever single free parameter is chosen — so AHP is retired at the composite level entirely. `w_R = 0.80`, `w_M = 0.20` are DECLARED constants re-derived algebraically from that retired 4×4 matrix's A/R/M weights; see [Composite Score Q(v)](#composite-score-qv).

---

### 11.6 Weight Shrinkage Strategy

Raw AHP weights can be extreme on small comparison sets. The shrinkage strategy formally blends them with a uniform prior:

```
w_final(d) = λ × w_AHP(d)  +  (1 − λ) × (1 / n_dimensions)
```

λ = 0 is uniform intra-dimension weights; λ = 1 is the raw stated judgement. **Shrinkage now applies only to the intra-dimension vectors** (FT(v), M(v), A(v), Impact) — since `w_R`, `w_M`, and `r_alpha` are DECLARED constants rather than AHP output ([§11.2](#112-rm-formulas)), the composite is **λ-invariant by construction**: `--equal-weights` and every λ setting in between score `Q(v)` with the identical `w_R=0.80, w_M=0.20` (or `w_R=w_M=0.5, r_alpha=0.5` under `--equal-weights` specifically, which additionally overrides the composite). This is a structural change from the pre-migration model, where λ = 0 meant "equal weights over the four composite dimensions" — that comparison no longer exists.

#### Measured sensitivity — monotone, not a plateau, and the effect is small

The sweep is implemented in [reproduce/ahp_sensitivity.py](../reproduce/ahp_sensitivity.py) and committed to [results/ahp_shrinkage_sweep.json](../results/ahp_shrinkage_sweep.json). Measured across the seven-scenario cohort, scoring Q(v) against `FaultInjector` labels I*(v) on the DEPENDS_ON projection:

| λ | 0.00 | 0.50 | 0.60 | 0.65 | **0.70** | 0.75 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| mean ρ | −0.0512 | −0.0254 | −0.0222 | −0.0202 | **−0.0188** | −0.0151 | −0.0140 | −0.0098 | **−0.0067** |

Three findings, all reported as measured, all superseding whatever this section claimed pre-migration (that comparison — a 4-D composite shrinking toward 4-way equal weights — no longer exists in this model, so read the numbers above as a fresh result, not an update to the old ones):

- **The curve is monotone increasing in λ**, not flat (`monotone_increasing_in_lambda: true`, `monotone_decreasing_in_lambda: false`). There is no plateau to appeal to, but there is also no evidence the shipped λ=0.70 default actively hurts relative to nearby λ — the raw AHP judgement (λ=1.0) is mildly *better* than every shrunk setting on this cohort (`ahp_lift_over_uniform_intra_dim: +0.0324`).
- **The effect is small.** The full sweep spans only 0.0445 ρ (`rho_spread_across_lambda`), and every value in it is near-zero-to-slightly-negative. This is consistent with, not contradictory to, the closed-form-scoring-is-weak-at-this-scale pattern seen throughout this section (compare [§8](#8-normalization)'s normalization sweep, similarly near-zero) — the GNN stages (see [prediction.md](prediction.md)) exist because closed-form `Q(v)` alone does not rank well against `I*(v)` on these cohorts; λ tuning cannot fix that.
- **The composite is unaffected either way** (`composite_weights_lambda_invariant: true`) — every row in the table above used the identical `w_R=0.80, w_M=0.20`. Only the intra-dimension FT/M/A/Impact vectors moved.

> **Scope of the result.** This is measured on the seven-scenario cohort against I*(v) using the closed-form `Q(v)` (no GNN). It does not show the weighting is wrong everywhere; it shows the shipped default is not distinguishable from nearby λ settings on the cohort this repository measures, and that no "plateau in [0.65, 0.75]" claim should be repeated. Because ρ is rank-based, all of this concerns only how the weighting *reorders* components.

Traceability: `AHPProcessor._shrink_weights` in [saag/analysis/weight_calculator.py](../saag/analysis/weight_calculator.py) implements the formula; regenerate the artifact with `PYTHONPATH=. python reproduce/ahp_sensitivity.py`.

---

### 11.7 Criticality Classification

RM scores are classified using an **adaptive box-plot classifier** that identifies components exceptional relative to the system's own distribution:

```
CRITICAL  :  score > Q3 + 1.5 × IQR   (structural outliers)
HIGH      :  Q3 < score ≤ upper fence   (upper quartile, non-outlier)
MEDIUM    :  median < score ≤ Q3
LOW       :  Q1 < score ≤ median
MINIMAL   :  score ≤ Q1
```

Classification is applied **independently per RM dimension/sub-characteristic and for the composite Q(v)**. A component can be CRITICAL on Availability (structural SPOF) but MINIMAL on Maintainability — which is exactly the diagnostic information needed to direct remediation (see [criticality.md's `CriticalityProfile` patterns](criticality.md) for the full FT/A/M combination taxonomy, e.g. "SPOF" = high A, low FT and M).

**Small-sample fallback (n < 12).** Box-plot thresholds become unstable at small node counts. For graphs with fewer than 12 components, percentile thresholds are used instead: CRITICAL = top 10%, HIGH = 75th–90th, MEDIUM = 50th–75th, LOW = 25th–50th, MINIMAL = bottom 25%.

**Typical distribution across validated scenarios:** CRITICAL ≈ 5–15%, HIGH ≈ 25%, MEDIUM ≈ 25%, LOW ≈ 25%, MINIMAL ≈ bottom 10–25%.

---

### 11.8 Interpretation Patterns

The combination of RM dimension scores characterises the *type* of risk and directs remediation:

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
| `layer` | str | Canonical identifier of the analysed layer — `"app"`, `"infra"`, `"mw"`, or `"system"` |
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

**Computed metrics (system layer)** — regenerated this session via `PYTHONPATH=. python examples/run_structural_analysis.py`; reproduce directly rather than hand-editing these numbers:

```
Component      RPR      DG_in  MPCI  AP_c_dir  BR    BT     w_in  FOC
─────────────────────────────────────────────────────────────────────────
SensorApp      0.22     0.25   0.0   0.0       0.0   0.0    0.59  0.0
MonitorApp     0.41     0.0    0.0   0.0       0.0   0.0    0.0   0.0
MainBroker     0.12     0.50   0.0   0.0       0.0   0.0    1.19  0.0
NavLib         0.12     0.50   0.0   0.0       0.0   0.0    1.19  0.0
/temperature   0.12     0.0    0.0   0.0       0.0   0.0    0.0   1.0
```

**Graph-level summary S(G):**

```
nodes=5, edges=5, density=0.25, avg_degree=2.0, avg_clustering=0.667
is_connected=False, num_components=2      ← /temperature is not a DEPENDS_ON endpoint;
                                             it forms its own component in this graph
num_articulation_points=0, num_bridges=0, diameter=2, avg_path_length=1.167
assortativity=-0.408     ← negative: high-degree hubs connect to low-degree leaves
```

> **This graph has zero articulation points and zero bridges.** Among {SensorApp, MonitorApp, MainBroker, NavLib}, 5 of the 6 possible undirected pairs are connected (everything except MainBroker–NavLib), so removing any single node leaves the rest connected — there is no structural SPOF in this tiny example. (An earlier revision of this section reported 3 articulation points, 5 bridges, and AP_c_directed values up to 0.65 for this same topology; that was not correct for the topology `examples/worked_example.json` actually encodes, and is unrelated to the RMAV→RM migration — it is corrected here because this section was being regenerated anyway.) `/temperature` also has RPR = 0.12 (not 0), reflecting the fixed-point PageRank residual `(1−d)/N` a topic node still receives even with in-degree 0.

Key structural observations:
- **MonitorApp** has the highest RPR despite MainBroker/NavLib having the highest DG_in — RPR captures transitive reach on `G^T`, not just immediate in-degree.
- **/temperature** has DG_in = 0 (Topic nodes are not DEPENDS_ON endpoints) but FOC = 1.0 (max fan-out for this system).
- **MainBroker** and **NavLib** have the highest `w_in` (QoS-weighted in-degree) — both applications depend on them directly.
- MPCI = 0.0 everywhere because all dependencies in this small example are single-path. Multi-path MPCI would appear in larger systems where the same (App_sub, App_pub) pair shares multiple topics.
- **Negative assortativity** indicates a hub-and-spoke topology: the two hub nodes (MainBroker, NavLib) connect to lower-degree leaf nodes (SensorApp, MonitorApp).

**Fault Tolerance FT(v), Availability A(v), Reliability R(v), Maintainability M(v), and Composite Q(v)** — pipeline output (rank-normalized internally; not a direct hand-multiplication of the raw table above), regenerated this session:

```
Component      FT(v)    A(v)     R(v)=0.36·FT+0.64·A    M(v)     Q(v)
──────────────────────────────────────────────────────────────────────
SensorApp      0.4875   0.0188   0.1875                 0.6454   0.3478
MonitorApp     0.4875   0.0188   0.1875                 0.5017   0.2975
MainBroker     0.5156   0.0188   0.1976                 0.2500   0.2160
NavLib         0.5156   0.0500   0.2176                 0.3737   0.2723
/temperature   0.9375   0.0188   0.3495                 0.3300   0.3427
```

`R(v) = r_alpha·FT(v) + (1−r_alpha)·A(v)` holds exactly at full precision for every row above (verified to float precision — safe to treat as an exact identity). `Q(v)` does **not** exactly equal `0.80·R(v) + 0.20·M(v)` at this precision (e.g. SensorApp: `0.8×0.1875 + 0.2×0.6454 = 0.279`, but `Q = 0.348`) — `QualityAnalyzer`'s composite weights are QoS-adapted per component around the 0.80/0.20 default (`analyzer.py._derive_qos_weights`, pre-existing behavior, not new to this migration); `Q(v)` is the pipeline's direct output, not a hand-derivable product of the displayed `R(v)`/`M(v)`.

Key observations:
- `/temperature` has the highest FT(v) (0.9375) because FOC = 1.0 dominates the Topic-type formula — the only topic with an active subscriber, and a single publisher, so its removal is maximally disruptive to that one subscriber.
- **NavLib** has the highest A(v) (0.05, driven by its QoS weight term) and, combined with a slightly-above-average FT(v), the highest R(v) among the non-Topic components.
- **SensorApp** has the highest M(v) (0.6454) despite low R(v) — a maintainability-dominant profile, not a reliability-dominant one, illustrating why the two dimensions are reported separately rather than only as the composite.
- No component in this tiny example is a structural SPOF (zero articulation points, see above), so none has an elevated A(v) from that term — A(v) here is driven almost entirely by the QoS-weight term (0.05·w(v)), which is why the values are small and close together.

---

## 14. Complexity

| Algorithm | Complexity | Notes |
|-----------|------------|-------|
| PageRank / RPR | O(I × \|E\|) | I = iterations (≤100) |
| Betweenness | O(\|V\| × \|E\|) | Brandes' algorithm; inverted weights |
| Closeness | O(\|V\| × (\|V\| + \|E\|)) | Harmonic closeness via BFS |
| Eigenvector | O(I × \|E\|) | Power iteration (≤500 iters); Katz fallback |
| AP_c_directed | O(\|V\| × (\|V\| + \|E\|)) | Reachability removal per vertex |
| CDI | O(\|V\| × (\|V\| + \|E\|)) | APSP removal per vertex; sampled for \|V\| > 300 |
| Bridge detection | O(\|V\| + \|E\|) | DFS-based |
| MPCI | O(\|E\|) | One pass over InEdges per component |
| FOC | O(\|V_topic\|) | One pass over Topic nodes |
| Rank normalization | O(\|V\| log \|V\|) | Per metric sort |
| RCM ordering | O(\|V\| + \|E\|) | Bandwidth minimization for matrix display |

**Overall:** O(|V|² + |V|×|E|), dominated by AP_c_directed and CDI. An `xlarge` system (200 components, ~600 edges) completes in approximately 20–25 seconds. AP_c_directed and CDI together account for roughly 70% of runtime.

> **Performance note:** AP_c_directed and CDI are both computed in StructuralAnalyzer (moved from QualityAnalyzer). This consolidation eliminates one redundant O(|V|²) pass previously performed during RM scoring. For enterprise-scale systems (|V| > 300), the CDI BFS is restricted to the top-50 "core" nodes (Application, Broker, Node) ranked by total degree (in + out). This is fully deterministic — the same graph always produces the same CDI values — and prioritises the nodes most likely to have significant path-length impact when removed.

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

# Analyze all four layers
PYTHONPATH=. python cli/analyze_graph.py --layer all

# Analyze multiple specific layers (comma-separated)
PYTHONPATH=. python cli/analyze_graph.py --layer app,system

# Export full metric vectors M(v) to JSON
PYTHONPATH=. python cli/analyze_graph.py --layer system --output results/metrics.json

# Multi-layer export: produces metrics_app.json, metrics_system.json
PYTHONPATH=. python cli/analyze_graph.py --layer app,system --output results/metrics.json

# Connect to a non-default Neo4j instance
PYTHONPATH=. python cli/analyze_graph.py --uri bolt://myhost:7687 --user neo4j --password secret

# Increase logging verbosity
PYTHONPATH=. python cli/analyze_graph.py --layer app --verbose
```

### 15.1 CLI Argument Reference

`cli/analyze_graph.py` computes **M(v) and S(G) only** — it takes no RM options, because scoring belongs to the Predict stage.

| Argument | Default | Description |
|----------|---------|-------------|
| `--layer`, `-l` | `system` | Layer(s) to analyze. Accepts a single layer, comma-separated list, or `all`. |
| `--output`, `-o` | — | Path to save full JSON results. Parent directory created if absent. |
| `--uri` | `bolt://localhost:7687` | Neo4j Bolt URI (overrides `NEO4J_URI` env var). |
| `--user`, `-u` | `neo4j` | Neo4j username (overrides `NEO4J_USER` env var). |
| `--password`, `-p` | `password` | Neo4j password (overrides `NEO4J_PASSWORD` env var). |
| `--verbose`, `-v` | off | Enable DEBUG-level logging. |
| `--quiet`, `-q` | off | Suppress INFO messages; show only warnings and errors. |

**Where the RM options live.** Normalization, weighting, and sensitivity all belong to the RM sub-phase ([§11](#11-analyze-stage--rule-based-rm-scoring)) and are exposed by `saag-predict` (`cli/predict_graph.py`), which runs structural analysis and then scores it:

```bash
# Normalization methods — "robust" (default) is rank-based, not IQR scaling
PYTHONPATH=. python cli/predict_graph.py --layer system --norm minmax
PYTHONPATH=. python cli/predict_graph.py --layer system --norm zscore

# Cap extreme outliers at the 95th percentile before ranking
PYTHONPATH=. python cli/predict_graph.py --layer system --winsorize

# Weight modes — do not affect M(v) computation
PYTHONPATH=. python cli/predict_graph.py --layer system --use-ahp
PYTHONPATH=. python cli/predict_graph.py --layer system --equal-weights
PYTHONPATH=. python cli/predict_graph.py --layer system --use-ahp --ahp-shrinkage 0.5

# Kendall τ weight sensitivity report
PYTHONPATH=. python cli/predict_graph.py --layer system --sensitivity
```

**Where cross-layer insights come from.** They require RM levels, so they are produced by the multi-layer path ([§4](#4-cross-layer-analysis)) — the `saag` orchestrator or the REST API, not `cli/analyze_graph.py`:

```bash
saag --analyze --layers app,infra,mw,system --output results/analysis.json
```

### 15.2 Interpreting the Output

The Q(v) column and the criticality labels below come from the RM sub-phase, so they appear on the `saag-predict` / `saag --analyze` output. `cli/analyze_graph.py` prints the summary line and the raw metric columns only.

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

On the multi-layer path, the output JSON also includes a `cross_layer_insights` array:

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

Step 2 produces structural metrics and deterministic RM quality scores Q(v). These rule-based scores represent the baseline criticality of each component.

To generalize these predictions beyond closed-form rules (e.g. learning nonlinear multi-hop motifs and predicting direct edge-level criticalities), the system uses an inductive Graph Neural Network in Step 3.

---

← [Step 1: Import](graph-model.md) | → [Step 3: Predict](prediction.md)