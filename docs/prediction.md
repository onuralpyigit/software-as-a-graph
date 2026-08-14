# Step 3: Predict — Rule-Based (RM) + Learned (GNN) Criticality

**Predict component and edge criticality by combining deterministic RM scoring with a Heterogeneous Graph Transformer trained on simulation-derived ground truth, plus anti-pattern detection and explanations.**

← [Step 2: Analyze](structural-analysis.md) | → [Step 4: Simulate](failure-simulation.md)

> **Unified Prediction Step.** Step 3 replaces the legacy "Quality Scoring" mechanism that used to live inside Step 2 (Analyze), which is now structural-metrics-only. It always computes deterministic $Q_{\text{RM}}(v)$ scores; when a trained GNN checkpoint is available it additionally runs a learned pass that discovers multi-hop topological patterns and predicts edge-level criticality directly. Anti-pattern detection and human-readable explanations are derived from the RM scores as part of this same step.

---

## Table of Contents

1. [What This Stage Does](#1-what-this-stage-does)
2. [Graph Data Preparation](#2-graph-data-preparation)
3. [Model Architecture](#3-model-architecture)
4. [Training Protocol](#4-training-protocol)
5. [Edge Criticality](#5-edge-criticality)
6. [Comparing the Prediction Modes](#6-comparing-the-prediction-modes)
7. [Output Schema](#7-output-schema)
8. [Commands](#8-commands)
9. [Known Limitations](#9-known-limitations)
10. [What Comes Next](#10-what-comes-next)

---

## 1. What This Stage Does

Step 3 takes the metric vector **M(v)** and graph structure produced by Step 2 and produces:

- Deterministic rule-based node scores $Q_{\text{RM}}(v) \in [0,1]$ — always
- Learned node criticality $Q_{\text{GNN}}(v) \in [0,1]$ — when a checkpoint exists
- Learned edge criticality $Q_{\text{GNN}}(u,v) \in [0,1]$ — when a checkpoint exists and `predict_edges` is set

```
M(v) + graph structure                Prediction Engine                Output
──────────────────────                ─────────────────                ──────────────────
Tier 1 & Tier 2 metrics:       →      RM formulas (always)       →     Q_RM(v) ∈ [0,1]
  PR, RPR, BT, CL, EV,                HGT GNN (with checkpoint)        Q_GNN(v)  ∈ [0,1]
  DG_in, DG_out, CC,                  3 prediction heads               Q_GNN(u,v) ∈ [0,1]
  AP_c_dir, BR, w, w_in,
  w_out, MPCI, PC, FOC, ...
```

Orchestration lives in [`PredictionService`](../saag/prediction/service.py); the learned path is [`GNNService`](../saag/prediction/gnn_service.py).

---

## 2. Graph Data Preparation

[`networkx_to_hetero_data()`](../saag/prediction/data_preparation.py#L498) converts the Step 1 NetworkX graph to a PyTorch Geometric `HeteroData` object, partitioning nodes and edges by type.

### Node features (type-specific dimensions)

Indices 0–17 are the shared topological base present for every node type. Type-specific extras follow at index 18+. `NODE_TYPE_TO_DIM` in [data_preparation.py](../saag/prediction/data_preparation.py#L138) is the authoritative width table.

| Node type | Total dim | Extras (indices 18+) |
|-----------|:---------:|----------------------|
| Application | 23 | 5 code quality attributes (18–22) |
| Library | 23 | 5 code quality attributes (18–22) |
| Broker | 19 | `max_connections_norm` (18) |
| Topic | 22 | `subscriber_count_norm`, `publisher_count_norm` (18–19), `log1p_frequency_norm` (20), `topic_qos_criticality_ord` (21) |
| Node (infra) | 20 | `cpu_cores_norm`, `memory_gb_norm` (18–19) |

HGT handles type-specific projections internally, so a global one-hot node-type vector is **not** required.

**Topological metrics — indices 0–17 (all node types):**

| Idx | Metric | RM role | | Idx | Metric | RM role |
|:---:|--------|-----------|-|:---:|--------|-----------|
| 0 | PageRank (PR) | Diagnostic (Tier 2) | | 9 | Bridge Ratio (BR) | A(v) |
| 1 | Reverse PageRank (RPR) | R(v) | | 10 | QoS aggregate weight (w) | QSPOF, A(v) |
| 2 | Betweenness (BT) | M(v) | | 11 | QoS weighted in-degree (w_in) | Diagnostic — fed the retired V(v); unused by any RM formula |
| 3 | Closeness (CL) | Diagnostic (Tier 2) | | 12 | QoS weighted out-degree (w_out) | M(v) |
| 4 | Eigenvector (EV) | Diagnostic (Tier 2) | | 13 | MPCI | R(v) via CDPot_enh |
| 5 | In-degree norm (DG_in) | R(v) | | 14 | path_complexity | M(v) via CouplingRisk_enh |
| 6 | Out-degree norm (DG_out) | CouplingRisk_enh | | 15 | Fan-Out Criticality (FOC) | R(v) for Topic nodes |
| 7 | Clustering coeff (CC) | M(v) as 1−CC | | 16 | AP_c_directed | A(v) directly and via QSPOF |
| 8 | AP_c Score | Diagnostic (topological) | | 17 | CDI | A(v) |

**Code quality — indices 18–22 (Application and Library only):** `loc_norm`, `complexity_norm`, `instability_code`, `lcom_norm`, `code_quality_penalty` (CQP). All but `loc_norm` feed M(v).

**Derivation notes.** Infrastructure extras (`cpu_cores_norm`, `memory_gb_norm`, `max_connections_norm`) use per-graph min-max normalization. Topic runtime counts divide `SUBSCRIBES_TO`/`PUBLISHES_TO` in-edge counts by the graph maximum. `log1p_frequency_norm` uses a per-scenario z-score of log1p(Hz) to avoid cross-domain leakage. `topic_qos_criticality_ord` is the ordinal (0–4) encoding of the 5-level QoS urgency label, masked to 0.0 when a graph's topics all share one criticality, so zero-variance scenarios cannot induce covariate shift.

> `Application.criticality` (bool, process-level ground truth) and `Topic.criticality` (5-level QoS urgency) are deliberately kept in separate feature dimensions — see the design note at the top of [data_preparation.py](../saag/prediction/data_preparation.py#L26).

### Edge features (16 dimensions)

| Idx | Feature |
|:---:|---------|
| 0 | QoS weight w(e) |
| 1 | `path_count_norm` = log₂(1 + path_count) / log₂(17) — coupling intensity, capped at 16 paths |
| 2–8 | Edge-type one-hot (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO, USES, DEPENDS_ON) |
| 9 | `reliability_score` (BEST_EFFORT 0.0 / RELIABLE 1.0) |
| 10 | `durability_score` (VOLATILE 0.0 / TRANSIENT_LOCAL 0.5 / TRANSIENT 0.6 / PERSISTENT 1.0) |
| 11 | `priority_score` (LOW 0.0 / MEDIUM 0.33 / HIGH 0.66 / URGENT 1.0) |
| 12 | `has_deadline` (1.0 if a finite `deadline_ns` is set) |
| 13 | `deadline_ns_log` = log10(1 + deadline_ns / 1e6), clamped to [0, 1] |
| 14 | `max_blocking_ms_log` = log10(1 + max_blocking_ms), clamped to [0, 1] |
| 15 | `qos_heterogeneity_flag` (1.0 if the edge's QoS profile differs from the scenario modal profile) |

Dimensions 9–15 are non-zero only for `PUBLISHES_TO` / `SUBSCRIBES_TO` edges, where QoS profiles are semantically meaningful; all other edge types receive zeros there.

### Labels

| Tensor | Shape | Contents |
|--------|-------|----------|
| `data[type].y` | (n, 3) | Simulation ground truth `[I*(v), IR(v), IM(v)]` |
| `data[type].y_rm` | (n, 3) | RM scores `[Q(v), R(v), M(v)]` — the consistency-regularisation target, **not** a training label |
| `data[type].label_mask` | (n,) | Which nodes the simulator actually scored — distinct from "scored 0.0" |
| `data[rel].y_edge` | (e, 3) | Per-edge criticality labels ([§5](#5-edge-criticality)) |

---

## 3. Model Architecture

```
    NetworkX DiGraph (Step 1 output)
              │
   ┌──────────▼───────────────────────┐
   │   Data Preparation               │  Type-specific node features:
   │   networkx_to_hetero_data()      │    App/Lib=23, Broker=19, Topic=22, Node=20
   │   HeteroData + splits            │  16-dim edge features
   └──────────┬───────────────────────┘  3-dim node labels y = I*(v)
              │                          3-dim RM targets y_rm
              ▼
   NodeCriticalityGNN ─── 3× (EdgeFeatureEncoder → HGTConv → residual+norm)
              │           optional reverse pass
              ├──────────► 3 prediction heads ──────► node scores (N, 3)
              │
   EdgeCriticalityGNN ─── TypedEdgeEncoder ─────────► edge scores (E, 3)
```

`EdgeCriticalityGNN` wraps a `NodeCriticalityGNN` and reuses its embeddings; both live in [models/core.py](../saag/prediction/models/core.py).

### Message passing

The backbone is a **3-layer stock PyG `HGTConv`** with type-dependent key/query/value projections — one set of attention parameters per `(src_type, edge_type, dst_type)` triple.

`HGTConv` does not accept raw `edge_attr` tensors. Edge features are therefore injected **before** each convolution by [`EdgeFeatureEncoder`](../saag/prediction/models/core.py#L66), which projects the 16-dim edge vector and scatter-means it into the destination node's embedding:

```
Layer 0 — type-specific input projection:
  h_v^(0) = GELU( LayerNorm( W_{type(v)} · x_v ) )

Layer k — edge injection, then HGT message passing:
  h_d  ← h_d + mean_{(u→d) ∈ r}( W_edge^(k) · e_ud )     ← EdgeFeatureEncoder, per dst node
  h'   = HGTConv_k( h, edge_index )
  h_v^(k+1) = Dropout( GELU( LayerNorm_type(v)( h'_v + h_v^(k) ) ) )   ← residual

Reverse pass (use_bidirectional=True, built on the fly inside encode()):
  rev_ei = { (dst, "rev__"+etype, src) : flip(edge_index) }
  h_rev  = rev_conv(h, rev_ei)
  h_v   ← h_v + 0.5 · h_rev[v]                            ← upstream signal
```

Hidden dimension D = 64, 4 attention heads, dropout p = 0.2. The reverse pass gives each node upstream as well as downstream context without duplicating edges in the data.

> **Scatter-mean is a real approximation.** Averaging all incoming edge vectors into one destination-node summary before attention smooths away per-edge distinctions: two incoming links with opposite QoS profiles are indistinguishable to the convolution that follows. Injecting edge features into each edge's own key/value — so attention can weigh links individually — would remove that limitation and is the natural next architectural step, but it is **not implemented**. Only the edge head ([§5](#5-edge-criticality)) sees per-edge features un-averaged.

### Prediction heads

```
R̂(v) = MLP_R( h_v )                            — Reliability
M̂(v) = MLP_M( h_v )                            — Maintainability
Î*(v) = MLP_C( h_v ‖ R̂ ‖ M̂ )                    — Composite
```

Each is a `ResidualMLP`; all outputs pass through a sigmoid, giving scores in [0, 1]. The composite head consumes the two dimension predictions as extra input so it can model non-linear interactions between them.

> **Fault tolerance and availability are not GNN heads.** They are Reliability sub-characteristics scored on the analysis side (`saag/analysis/`), not separate prediction targets — the GNN predicts only the two RM label columns (reliability, maintainability) plus the composite. See [models/core.py](../saag/prediction/models/core.py#L309).

---

## 4. Training Protocol

**Splits.** 60/20/20 train/val/test per node type via `create_node_splits()`, redrawn per seed unless the caller pins an external split (which lets several model variants be scored on an identical sample). Maximum 300 epochs. Loss and metrics are computed over `Application` and `Library` nodes only.

**Label normalization.** `normalize_labels_robust()` maps labels through `sigmoid((y − median) / IQR)` with the IQR clamped to ≥ 1e-6, computed over labelled (non-zero) nodes and preserving zeros. It mutates `.y` in place and so runs exactly once per graph, before the seed loop.

**Loss.**

```
L = L_composite + 0.5·L_dimension + 0.3·L_rank + 0.1·L_pairwise + 0.1·L_consistency + 0.3·L_edge

L_composite   = MSE( Î*(v), I*(v) )                     — labelled nodes
L_dimension   = Σ_d MSE( d̂(v), I_d*(v) )               — labelled nodes; dimensions the
                                                          labeler never measured are dropped
                                                          via `dimension_mask`, not regressed to 0
L_rank        = −(1/N) Σ_v log P(rank of v)             — ListMLE, labelled nodes
L_pairwise    = Σ_{i,j: t_i−t_j > m} max(0, m − (s_i−s_j)) / n_pairs   — margin m = 0.05
L_consistency = MSE( pred_unlabelled, y_rm )            — RM regularisation, unlabelled nodes
L_edge        = mean_r MSE( ŷ_edge[:,0], y_edge[:,0] )  — only when the model predicts edges
```

`L_edge` is averaged across relation types so a graph with many `PUBLISHES_TO` edges and few `DEPENDS_ON` edges does not let one relation dominate. It is skipped entirely for relations without `y_edge`.

**Early stopping.** Combined-metric, patience 30 epochs:

```
combined = 0.6 · val_rho  +  0.4 · max(0, 1 − val_loss / (best_val_loss + ε))
```

**Optimizer.** AdamW, lr = 3×10⁻⁴, weight_decay = 10⁻⁴, gradient clipping at max_norm 1.0. Schedule `CosineAnnealingWarmRestarts(T_0 = max(50, epochs//4), T_mult = 2, η_min = lr × 0.01)`.

**Multi-seed.** Each seed in `{42, 123, 456, 789, 2024}` runs the full loop independently; the weights from the **best seed by validation Spearman ρ** are restored before serialization, and that seed is persisted in `service_config.json` so inference reproduces the same split masks.

**Inductive training.** Passing `inductive_graphs` to `GNNService.train()` adds whole scenarios to the training set; validation and early stopping still track the primary graph via `primary_data`.

---

## 5. Edge Criticality

> See [criticality.md §5](criticality.md#5-relationship-edge-criticality) for the conceptual definition this section implements.

```
score(u, v) = TypedEdgeEncoder_r( h_u, h_v, e_uv )

e_uv ∈ ℝ¹⁶: QoS weight + path_count_norm + 7-bit edge-type one-hot + 7 QoS dims
```

[`TypedEdgeEncoder`](../saag/prediction/models/core.py#L106) learns a relation-specific projection $W_r \in \mathbb{R}^{16 \times D}$ per edge type. The projected edge feature is fused with the endpoint embeddings — `[h_src ‖ h_dst ‖ e_proj]` → Linear → LayerNorm → GELU — before the output head. Unlike the backbone, this path sees each edge's features individually rather than scatter-meaned.

**Edge labels are a heuristic, not a measurement.** [data_preparation.py](../saag/prediction/data_preparation.py#L699) derives them from the source node's simulated impact, discounted by whether the edge is a structural bridge:

```
y_edge(u, v) = I*(u) × bridge_multiplier          bridge_multiplier = 1.0 if bridge else 0.1
```

This is a proxy with two known consequences: every edge out of a high-impact node inherits that node's blast radius regardless of whether traffic actually flows over it, and non-bridge edges are uniformly damped rather than individually assessed.

**The measured alternative is available but not wired in.** `FailureSimulator.simulate_edge_removal` (and `simulate_edge_removal_sweep`) already computes the honest quantity — sever one relationship, leave both endpoints up, recompute impact, and subtract the no-op control:

```
I_edge(u, v) = composite_impact(G \ {(u,v)}) − composite_impact(G)
```

Subtracting matters: `_calculate_impact` returns a non-zero floor on a pristine graph (composite 0.0061 on `av_system`), because topics already lacking a publisher or subscriber count as lost throughput. A level rather than a delta would hand every edge that floor as if it were signal. The sweep is bounded to `bridges(G) ∪ top-q edge-betweenness` since a full pass costs one impact recomputation per edge, and edges outside the candidate set are returned with `evaluated: false` — *not measured* is distinct from *measured as harmless*.

Step 3 does not consume that output today; see [§9](#9-known-limitations).

---

## 6. Comparing the Prediction Modes

| Property | Analyze — RM (rule-based) | Predict — GNN (learned) |
|----------|:---------------------------:|:-----------------------:|
| Requires training data | No | Yes |
| Node criticality | ✓ | ✓ |
| Edge criticality | Proxies (BR, BT) | ✓ Direct, on heuristic labels |
| Per-dimension decomposition | ✓ Explicit | ✓ Learned heads |
| Interpretability | Full | Partial (attention + heads) |
| Topic-type branching | ✓ Explicit | Learned |
| MPCI amplification | ✓ Explicit (CDPot_enh) | Learned |
| Generalises to unseen systems | Immediately | Requires fine-tuning |
| Spearman ρ (published validation) | 0.876 overall; 0.943 large-scale | 0.587 (HGL-QoS, per-domain k-fold) |
| F1@K / F1-score (published validation) | 0.893 | 0.505 (HGL-QoS, per-domain k-fold) |
| Primary use | First analysis; interpretable; CI gate; fallback when no checkpoint | Default predictor after training; RM = fallback |

> **Validation-source note.** The GNN figures are HGL-QoS per-domain repeated k-fold results (`k=5`, 5 seeds, [cli/kfold_evaluate.py](../cli/kfold_evaluate.py)) against simulation labels, evaluated independently within each of seven scenarios and averaged (`ρ = 0.587 ± 0.146`, `F1@K = 0.505`; positive in all seven individually, range `ρ = 0.341–0.781`). This is an *in-domain* metric — trained and evaluated on the same scenario, repeated under resampling to show the result is stable rather than an artifact of one split — not a claim about zero-shot transfer. The cross-scenario Leave-One-Scenario-Out protocol, which does test transfer, remains available ([cli/loso_evaluate.py](../cli/loso_evaluate.py)) and reached `ρ = 0.290` (`F1@K = 0.405`, HGL-QoS). LOSO is retained as a secondary domain-gap analysis rather than the headline metric, since testing transfer between architecturally distinct scenarios (autonomous-vehicle vs. financial-trading vs. hub-and-spoke topologies) conflates model quality with how much structure those unrelated domains happen to share.
>
> Both figures predate the edge-loss fix in [§4](#4-training-protocol) but are unaffected by it: every evaluation harness runs with `predict_edges=False`.

---

## 7. Output Schema

`python cli/predict_graph.py --output results/prediction.json` writes one entry per layer. The `gnn` block is present only when `--gnn-model` was supplied.

```json
{
  "layers": {
    "system": {
      "total_components": 35,
      "rm": {
        "NavLib": {
          "overall":         0.54,
          "reliability":     0.63,
          "maintainability": 0.41,
          "fault_tolerance": 0.59,
          "availability":    0.58,
          "is_spof":         true,
          "blast_radius":    12,
          "cascade_depth":   4
        }
      },
      "antipatterns": [
        {
          "entity_id":      "NavLib",
          "entity_type":    "Component",
          "name":           "Single Point of Failure (SPOF)",
          "severity":       "CRITICAL",
          "category":       "Availability",
          "description":    "NavLib is a directed cut vertex. Removing it partitions the dependency graph.",
          "recommendation": "Introduce redundancy: backup instances or alternative paths.",
          "evidence":       { "is_articulation_point": true, "availability_score": 0.58 }
        }
      ],
      "gnn": {
        "prediction_mode": "gnn_only",
        "node_scores": {
          "NavLib": {
            "component":             "NavLib",
            "composite_score":       0.5432,
            "reliability_score":     0.6321,
            "maintainability_score": 0.4121,
            "criticality_level":     "HIGH",
            "source":                "GNN"
          }
        },
        "edge_scores": [
          {
            "source":                "MonitorApp",
            "target":                "SensorApp",
            "edge_type":             "DEPENDS_ON",
            "composite_score":       0.4512,
            "reliability_score":     0.3211,
            "maintainability_score": 0.2512,
            "criticality_level":     "MEDIUM"
          }
        ],
        "gnn_metrics": {
          "spearman_rho": 0.5871, "f1_score": 0.5052, "macro_f1": 0.0,
          "bce_loss": 0.0, "regression_slope": 0.0, "regression_intercept": 0.0,
          "regression_r2": 0.0, "rmse": 0.0812, "mae": 0.0612,
          "top_5_overlap": 0.6, "top_10_overlap": 0.7, "ndcg_10": 0.9211,
          "precision": 0.0, "recall": 0.0, "accuracy": 0.0,
          "calibration": "rank_matched", "n_critical_in_truth": 0
        }
      }
    }
  }
}
```

`prediction_mode` is `"gnn_only"` or `"rm_only"`. `gnn_metrics` is populated only when evaluation labels were supplied; `criticality_level` comes from a per-scenario box-plot classification of that run's own score distribution, applied to nodes and edges separately.

---

## 8. Commands

```bash
# ─── GNN training (requires Step 4 simulation results) ────────────────────────

PYTHONPATH=. python cli/train_graph.py --layer system                          # single seed
PYTHONPATH=. python cli/train_graph.py --layer system --seeds 42 123 456 789 2024
PYTHONPATH=. python cli/train_graph.py --layer system --multi-scenario         # inductive
PYTHONPATH=. python cli/train_graph.py --layer system --no-edge-model          # nodes only

# ─── GNN inference ────────────────────────────────────────────────────────────

PYTHONPATH=. python cli/predict_graph.py --gnn-model output/gnn_checkpoints/best_model

# ─── Evaluation protocols ─────────────────────────────────────────────────────

PYTHONPATH=. python cli/kfold_evaluate.py    # primary: per-domain repeated k-fold
PYTHONPATH=. python cli/loso_evaluate.py     # secondary: cross-scenario domain gap
```

Full flag reference: [cli-pipeline-guide.md](cli-pipeline-guide.md).

---

## 9. Known Limitations

Documented so they aren't mistaken for working code.

| # | Limitation | Impact |
|:--|------------|--------|
| L1 | **Edge labels are heuristic** ([§5](#5-edge-criticality)). `y_edge` is `I*(source) × bridge_multiplier`, not a measured impact delta. | Edge scores rank by *source-node blast radius discounted by bridge status*, which is a weaker claim than "this link matters". `FailureSimulator.simulate_edge_removal` computes the measured version but Step 3 does not read it. |
| L2 | **Edge features are scatter-meaned in the backbone** ([§3](#message-passing)). `EdgeFeatureEncoder` averages incoming edge vectors per destination node before `HGTConv`. | Per-edge QoS distinctions are smoothed away in the node embeddings. Only `TypedEdgeEncoder` sees edges individually. Projecting edge features into each edge's own key/value would fix this. |
| L3 | **Transductive by default.** Single-graph training lets test nodes contribute neighbourhood context. | Per-domain repeated k-fold ([cli/kfold_evaluate.py](../cli/kfold_evaluate.py)) is the primary validation protocol and excludes held-out fold nodes from that fold's training; LOSO ([cli/loso_evaluate.py](../cli/loso_evaluate.py)) isolates whole scenarios. Read headline numbers from those, not from a bare `train()` call. |
| L4 | **Node loss covers `Application` and `Library` only.** Broker, Topic, and Node embeddings are trained purely through message passing. | Per-type Spearman ρ for infrastructure types is reported by `evaluate()` but those types are never directly supervised. |
| L5 | **Checkpoints below `feature_version` 3 are incompatible.** Broker 18→19, Topic 18→22, Node 18→20. | `from_checkpoint()` raises on a dimension mismatch at `feature_version` ≥ 2 and warns below it. Re-train rather than force-load. |

Fixed in the current revision, recorded because older checkpoints and result files predate them:

- The edge prediction head received no gradient — `y_edge` was written but no loss term read it, so `TypedEdgeEncoder` stayed at random initialisation and all edge scores were noise. `GNNTrainer._edge_loss` now supervises it.
- With `predict_edges=True`, `GNNService` built two independent node networks and reported metrics from the untrained one. `_node_model` is now the edge model's inner `node_gnn`.
- `normalize_labels_robust` ran inside the per-seed loop, compounding its in-place sigmoid squash so seed *N* trained on different labels than seed 1. It now runs once per graph.

---

## 10. What Comes Next

Step 3 has two operational modes. For **inference**, a trained checkpoint lets Step 3 run straight after Step 2 and emit GNN predictions; Steps 4 and 5 then validate those predictions against simulation ground truth. For **training**, Step 4 must come first because the simulation labels `I(v)` are required, followed by Step 5 to measure performance.

→ [Step 4: Simulate](failure-simulation.md)
