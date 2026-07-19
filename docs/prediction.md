# Step 3: Predict — Unified Rule-Based (RMAV) + ML (GNN) Prediction Step

**Predict component and edge criticality by combining deterministic rule-based RMAV scoring with a Heterogeneous Graph Transformer (HGT) trained on simulation-derived ground truth, plus anti-pattern detection and explanations.**

← [Step 2: Analyze](structural-analysis.md) | → [Step 4: Simulate](failure-simulation.md)

> **Unified Prediction Step.** This step replaces the legacy "Quality Scoring" mechanism that used to live inside Step 2 (Analyze), which is now structural-metrics-only. Step 3 always computes deterministic, rule-based Q_RMAV(v) scores; when a trained GNN checkpoint is available it blends in a machine-learning pass that discovers complex, multi-hop topological patterns and predicts direct edge-level criticality (falling back to RMAV otherwise). GNN models are trained on simulation labels I(v) and can generalize predictions to unseen systems without running full runtime simulations. Anti-pattern detection and human-readable explanations are derived from the resulting scores as part of this same step.

---

## Table of Contents

1. [What This Stage Does](#1-what-this-stage-does)
2. [Predict Stage — Graph Neural Network](#2-predict-stage--graph-neural-network)
   - 2.1 [Motivation](#21-motivation)
   - 2.2 [Architecture Overview](#22-architecture-overview)
   - 2.3 [Graph Data Preparation](#23-graph-data-preparation)
   - 2.4 [HGT Model (Heterogeneous Graph Transformer)](#24-hgt-model-heterogeneous-graph-transformer)
   - 2.5 [Multi-Task Prediction Heads](#25-multi-task-prediction-heads)
   - 2.6 [Edge Criticality Prediction](#26-edge-criticality-prediction)
   - 2.7 [Deprecated: Ensemble](#27-deprecated-ensemble)
   - 2.8 [Training Protocol](#28-training-protocol)
   - 2.9 [Multi-Seed Stability](#29-multi-seed-stability)
   - 2.10 [Methodological Integrity](#210-methodological-integrity)
3. [Comparing the Prediction Modes](#3-comparing-the-prediction-modes)
4. [Output Schema](#4-output-schema)
5. [Commands](#5-commands)
6. [What Comes Next](#6-what-comes-next)

---

## 1. What This Stage Does

The Predict stage takes the metric vector **M(v)** and graph structure produced by Step 2 and generates:
- Machine-learned node criticality predictions $Q_{\text{GNN}}(v) \in [0, 1]$
- Direct edge-level criticality predictions $Q_{\text{GNN}}(u, v) \in [0, 1]$

M(v) and Graph structure             Prediction Engine                  Output
────────────────────────             ─────────────────               ─────────────────────
Tier 1 & Tier 2 Metrics:      →      GNN Model (HGT)          →      Q_GNN(v) ∈ [0, 1]
  PR, RPR, BT, CL, EV,               Multi-task prediction           Q_GNN(u, v) ∈ [0, 1]
  DG_in, DG_out, CC,                 heads for RMAV dimensions
  AP_c_dir, BR, w, w_in,             and overall criticality
  w_out, MPCI, PC, FOC, ...
```

---

## 2. Predict Stage — Graph Neural Network

### 2.1 Motivation

While rule-based scoring (RMAV) is highly interpretable, it is limited by:
1. **Fixed Feature Interactions:** Linear or simple non-linear combinations cannot adapt to scenario-specific metric interactions.
2. **Node-Only Supervision:** RMAV cannot directly predict edge criticality (which specific pub-sub relationship is most critical) without endpoint proxies.

The GNN-based Predict stage addresses these limitations by learning directly from simulation ground-truth impact $I(v)$ and structural bridges.

### 2.2 Architecture Overview

```
    NetworkX DiGraph (Step 1 output)
              │
   ┌──────────▼───────────────────────┐
   │   Data Preparation               │  Type-specific node features:
   │   networkx_to_hetero_data()      │    App/Lib=23, Broker=19, Topic=22, Node=20
   │   HeteroData + splits            │   16-dim edge features (weight, path_count_norm,
   └──────────┬───────────────────────┘    7-bit type one-hot, 7 QoS decomposition dims)
              │                            5-dim simulation labels y = I*(v)
              │                            5-dim RMAV scores  y_rmav = Q_RMAV(v)
        ┌───────┴────────────┐
        ▼                    ▼
     NodeGNN              EdgeGNN
     3L HGT               TypedEdge Encoder
     +EdgeFeat +BiDir     (E, 16)
     (N, 5)
```

All prediction modules are managed by `GNNService`.

### 2.3 Graph Data Preparation

`networkx_to_hetero_data()` in `data_preparation.py` converts the Step 1 NetworkX graph to a PyTorch Geometric `HeteroData` object, partitioning nodes and edges by type.

#### Node Feature Vector (type-specific dimensions)

Each node `v` is represented by a feature vector whose dimension depends on its type. The first 18 indices are the shared topological base present for all node types. Type-specific extra features follow at indices 18+.

| Node type | Total dim | Extra features (indices 18+) |
|-----------|:---------:|------------------------------|
| Application | 23 | +5 code quality attributes (indices 18–22) |
| Library | 23 | +5 code quality attributes (indices 18–22) |
| Broker | 19 | +1 `max_connections_norm` (index 18) |
| Topic | 22 | +4 `subscriber_count_norm`, `publisher_count_norm` (indices 18–19), `log1p_frequency_norm` (index 20, per-scenario z-score), `topic_qos_criticality_ord` (index 21, ordinal 0–4) |
| Node (infra) | 20 | +2 `cpu_cores_norm`, `memory_gb_norm` (indices 18–19) |

> **Implementation note.** The HGT architecture handles type-specific projections internally, so a global one-hot node-type vector is **not required**. The constant `NODE_TYPE_TO_DIM` in `data_preparation.py` defines the authoritative widths. Infrastructure extra features (`cpu_cores_norm`, `memory_gb_norm`, `max_connections_norm`) are derived by per-graph min-max normalization of node attributes. Topic runtime features (`subscriber_count_norm`, `publisher_count_norm`) are derived by counting `SUBSCRIBES_TO`/`PUBLISHES_TO` edges per Topic node and dividing by the graph maximum. `log1p_frequency_norm` uses per-scenario z-score of log1p(Hz) to avoid cross-domain leakage. `topic_qos_criticality_ord` is the ordinal encoding (0–4) of the 5-level QoS urgency label; when all topics in a graph share the same criticality (zero variance), the field is masked to 0.0 to prevent covariate shift across scenarios.

**Topological metrics — indices 0–17 (base for all node types):**

| Index | Metric | RMAV role |
|:-----:|--------|-----------|
| 0 | PageRank (PR) | Diagnostic (Tier 2) |
| 1 | Reverse PageRank (RPR) | R(v) |
| 2 | Betweenness Centrality (BT) | M(v) |
| 3 | Closeness Centrality (CL) | Diagnostic (Tier 2) |
| 4 | Eigenvector Centrality (EV) | Diagnostic (Tier 2) |
| 5 | In-Degree normalized (DG_in) | R(v) |
| 6 | Out-Degree normalized (DG_out) | CouplingRisk_enh |
| 7 | Clustering Coefficient (CC) | M(v) as 1−CC |
| 8 | AP_c Score | Diagnostic (Topological) |
| 9 | Bridge Ratio (BR) | A(v) |
| 10 | QoS aggregate weight (w) | QSPOF, A(v) |
| 11 | QoS weighted in-degree (w_in) | V(v) |
| 12 | QoS weighted out-degree (w_out) | M(v) |
| 13 | MPCI | R(v) via CDPot_enh amplifier |
| 14 | path_complexity | M(v) via CouplingRisk_enh |
| 15 | Fan-Out Criticality (FOC) | R(v) for Topic nodes |
| 16 | AP_c_directed | A(v) directly and via QSPOF |
| 17 | CDI (Connectivity Degradation) | A(v) |

**Code quality metrics — indices 18–22 (Application and Library only):**

| Index | Metric | RMAV role |
|:-----:|--------|-----------|
| 18 | loc_norm | Diagnostic |
| 19 | complexity_norm | M(v) via CQP |
| 20 | instability_code | M(v) via CQP |
| 21 | lcom_norm | M(v) via CQP |
| 22 | code_quality_penalty (CQP) | M(v) directly |

**Infrastructure metrics — indices 18–19 (Broker, Topic, Node only):**

| Index | Metric | Node type | Source |
|:-----:|--------|-----------|--------|
| 18 | max_connections_norm | Broker | Node attribute, normalized per Broker subgraph |
| 18 | subscriber_count_norm | Topic | SUBSCRIBES_TO in-edge count, normalized per graph |
| 19 | publisher_count_norm | Topic | PUBLISHES_TO in-edge count, normalized per graph |
| 18 | cpu_cores_norm | Node | Node attribute, normalized per Node subgraph |
| 19 | memory_gb_norm | Node | Node attribute, normalized per Node subgraph |

#### Edge Features (16 dimensions)

| Index | Feature |
|:-----:|---------|
| 0 | QoS weight w(e) |
| 1 | path_count_norm = log₂(1 + path_count) / log₂(17) — coupling intensity (capped at 16 paths) |
| 2–8 | Edge-type one-hot (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO, USES, DEPENDS_ON) |
| 9 | reliability_score (0.0 BEST_EFFORT / 1.0 RELIABLE) — non-zero for PUBLISHES_TO, SUBSCRIBES_TO, DEPENDS_ON |
| 10 | durability_score (VOLATILE=0.0 / TRANSIENT_LOCAL=0.5 / TRANSIENT=0.6 / PERSISTENT=1.0) — pub/sub only |
| 11 | priority_score (LOW=0.0 / MEDIUM=0.33 / HIGH=0.66 / URGENT=1.0) — pub/sub only |
| 12 | has_deadline (1.0 if finite deadline_ns is set, else 0.0) — pub/sub only |
| 13 | deadline_ns_log = log10(1 + deadline_ns / 1e6), clamped to [0, 1] — pub/sub only |
| 14 | max_blocking_ms_log = log10(1 + max_blocking_ms), clamped to [0, 1] — pub/sub only |
| 15 | qos_heterogeneity_flag (1.0 if edge QoS profile differs from scenario-level modal profile, else 0.0) — pub/sub only |

Dimensions 9–15 are non-zero only for PUBLISHES_TO / SUBSCRIBES_TO edges, where QoS profiles are semantically meaningful. All other edge types receive zeros for these dimensions.

**Node labels** (`data[type].y`, shape (n, 5)): simulation ground-truth per-dimension impact scores `[I*(v), IR(v), IM(v), IA(v), IV(v)]`.

**RMAV scores** (`data[type].y_rmav`, shape (n, 5)): RMAV quality scores `[Q(v), R(v), M(v), A(v), V(v)]`, stored for consistency regularization target during training. These are *not* used as training labels.

### 2.4 HGT Model (Heterogeneous Graph Transformer)

The model uses a **3-layer Heterogeneous Graph Transformer with native edge features (EdgeAwareHGTConv)** with type-dependent key/query/value projections. For each `(src_type, edge_type, dst_type)` triple, EdgeAwareHGTConv learns separate attention parameters.

**Standard HGTConv does not accept raw edge_attr tensors.** To avoid information smoothing caused by aggregating edge patterns across incoming paths indiscriminately (as was done by the legacy `EdgeFeatureEncoder`), `EdgeAwareHGTConv` projects the 16-dimensional edge attributes directly into the Key and Value representations of each individual edge before computing the multi-head attention and message-passing aggregates.

```
Layer 0 — Type-specific input projection:
  h_v^(0) = GELU( LayerNorm( W_{type(v)} · x_v ) )

Layer k — Edge-Aware HGT message passing per (src_type s, edge_type r, dst_type d):
  k_uv = K_s^(k) · h_u + K_edge^(s,r,d,k) · e_uv      ← native edge key projection
  v_uv = V_s^(k) · h_u + V_edge^(s,r,d,k) · e_uv      ← native edge value projection
  q_v  = Q_d^(k) · h_v

  ATT(u,v) = softmax_u( (q_v)^T · k_uv / √D_head · μ_(s,r,d) )
  MSG(u,v) = W_{MSG}^(s,r,d) · v_uv
  m_v^(k)  = Σ_{u,r} ATT(u,v) · MSG(u,v)
  h_v^(k+1) = GELU( LayerNorm( W_{agg} · m_v^(k)  +  h_v^(k) ) )   ← residual

Reverse pass (use_bidirectional=True, computed on-the-fly within encode()):
  rev_ei = { (dst, "rev_"+etype, src) : flip(edge_index) }  for each relation
  h_rev  = rev_conv(h, rev_ei, rev_ea)
  h_v   ← h_v + 0.5 · h_rev[v]                              ← upstream signal
```

Residual connections exist between layers. Hidden dimension D = 64. Dropout p = 0.2. The reverse pass captures upstream and downstream structural signals without modifying data preparation.

### 2.5 Multi-Task Prediction Heads

```
R̂(v) = MLP_R( h_v )                            — Reliability head
M̂(v) = MLP_M( h_v )                            — Maintainability head
Â(v)  = MLP_A( h_v )                            — Availability head
V̂(v)  = MLP_V( h_v )                            — Vulnerability head
Î*(v) = MLP_C( h_v ‖ R̂(v) ‖ M̂(v) ‖ Â(v) ‖ V̂(v) ) — Composite head
```

All outputs pass through sigmoid activation, producing scores in [0, 1]. The composite head receives dimension predictions as additional input, allowing the final Î*(v) to incorporate non-linear dimension interactions.

### 2.6 Edge Criticality Prediction

```
score(u, v) = TypedEdgeEncoder_r( h_u, h_v, e_{uv} )

e_{uv} ∈ ℝ^16: QoS weight + path_count_norm + 7-bit edge-type one-hot + 7-bit QoS features
```

The `TypedEdgeEncoder` learns relation-specific linear projections $W_r \in \mathbb{R}^{16 \times D}$. The projected edge feature is fused with source and destination node embeddings via a shared layer `[h_src ‖ h_dst ‖ e_proj]` → LayerNorm → GELU before the output head.

**Edge labels.** Training labels for edges are derived from simulation ground truth with a bridge-aware multiplier:

```
I_edge(u, v) = I*(u) × bridge_multiplier

bridge_multiplier = 1.0   if (u, v) is a structural bridge
                   = 0.1   otherwise
```

This downweights non-bridge edges to reduce label noise from redundant paths.

### 2.7 [DEPRECATED] Ensemble: GNN + RMAV

The ensemble blending step (formerly `EnsembleGNN`) has been deprecated and removed. Criticality predictions are now derived solely from raw GNN outputs.

### 2.8 Training Protocol

**Transductive Split.** 60/20/20 train/val/test split applied per node type via `create_node_splits()`. Maximum 300 epochs.

**Label normalization.** Labels are normalized in-place via `normalize_labels_robust()`: IQR-based robust normalization `sigmoid((y − median) / IQR)` with the IQR clipped to a minimum of 1e-6.

**Early stopping.** Combined-metric early stopping with patience = 30 epochs:
```
combined = 0.6 × val_rho  +  0.4 × max(0,  1 − val_loss / (best_val_loss + ε))
```

**Loss function:**

```
L = L_composite  +  0.5 × L_dimension  +  0.3 × L_rank  +  0.1 × L_pairwise  +  0.1 × L_consistency

L_composite   = MSE( Î*(v),   I*(v) )                  — composite impact (labeled nodes)
L_dimension   = Σ_d  MSE( d̂(v),  I_d*(v) )            — per-dimension impact (labeled nodes)
L_rank        = −(1/N) Σ_v  log P(rank of v)           — ListMLE list-level ranking (labeled nodes)
L_pairwise    = Σ_{i,j: t_i−t_j > m}  max(0, m − (s_i−s_j)) / n_pairs  — pairwise margin (labeled nodes, margin m = 0.05)
L_consistency = MSE( pred_unlabeled_rmav, rmav_target ) — RMAV consistency regularization (unlabeled nodes)
```

**Optimizer and scheduler:** AdamW, lr = 3×10⁻⁴, weight_decay = 10⁻⁴, gradient clipping max_norm = 1.0. Schedule: `CosineAnnealingWarmRestarts(T_0 = max(50, epochs//4), T_mult=2, η_min = lr×0.01)`.

### 2.9 Multi-Seed Stability

Multi-seed training runs the full train loop independently for each seed in `{42, 123, 456, 789, 2024}`. The implementation restores the weights from the **best-performing seed** (by validation Spearman ρ) before final serialization.

**Inductive evaluation.** For cross-scenario generalisation claims, `inductive_graphs` are passed to `GNNService.train()`. Inductive graphs contribute all their nodes to training.

### 2.10 Methodological Integrity

| ID | Issue | Severity | Status | Solution / Mitigation |
|----|-------|:--------:|--------|-----------------------|
| G1 | Node feature dim mismatch | High | **Resolved** | Per-type dims enforced; feature version check in config |
| G2 | Edge labels as node proxies | Medium | Open | Current: bridge-multiplier-based downweighting |
| G3 | Redundant type encoding | Low | **Resolved** | Zero-padding removed; type-appropriate indices selected |
| G4 | Transductive leakage | High | **Resolved** | Per-domain repeated k-fold (`cli/kfold_evaluate.py`) is now the primary validation protocol — held-out fold nodes are excluded from that fold's training within each scenario. The older cross-scenario LOSO protocol (`cli/loso_evaluate.py`) remains available as a secondary/domain-gap analysis (see note below), which used `get_inductive_subgraph` to isolate scenarios during training and validation. |
| G5 | Best seed weight saving | High | **Resolved** | `best_state` restored before `train()` returns |
| G6 | Ensemble α bias | Medium | **Deprecated** | Blending removed. |
| G7 | Prediction seed overwrite | High | **Resolved** | `predict_from_data()` uses persisted best seed for mask consistency |
| G8 | Ensemble validation gap | High | **Deprecated** | Blending removed. |
| G9 | Silent ensemble fallback | Medium | **Deprecated** | Blending removed. |
| G10 | Layer compatibility check | Medium | **Resolved** | `from_checkpoint()` validates layer alias against metadata |
| G11 | Edge feature dim mismatch | Medium | **Resolved** | Historical 8→9 mismatch superseded; active schema is 16 dimensions (`weight`, `path_count_norm`, 7-bit edge type, 7 QoS dims). |
| G12 | HGTConv edge_attr incompatibility | Medium | **Resolved** | EdgeAwareHGTConv projects edge features into relation-specific Key and Value spaces |
| G13 | Single-cycle LR decay | Low | **Resolved** | `CosineAnnealingWarmRestarts` with T_0, T_mult=2 |
| G14 | Spearman-only early stopping | Low | **Resolved** | Combined score `0.6×ρ + 0.4×loss_improvement` |

---

## 3. Comparing the Prediction Modes

| Property | Analyze — RMAV (rule-based) | Predict — GNN (learning-based) |
|----------|:-----------------:|:--------------------:|
| Requires training data | No | Yes |
| Node criticality | ✓ | ✓ |
| Edge criticality | Proxies (BR, BT) | ✓ Direct (approx.) |
| Per-dimension decomposition | ✓ Explicit | ✓ Learned heads |
| Interpretability | Full | Partial (attention + heads) |
| Topic-type branching | ✓ Explicit | Learned |
| MPCI amplification | ✓ Explicit (CDPot_enh) | Learned |
| Generalises to unseen systems | Immediately | Requires fine-tuning |
| Spearman ρ (published validation) | 0.876 overall; 0.943 large-scale | 0.587 (HGL-QoS, per-domain k-fold) |
| F1@K / F1-score (published validation) | 0.893 | 0.505 (HGL-QoS, per-domain k-fold) |
| Transductive leakage risk | None (formula-based) | None (Resolved) |
| Primary use | First analysis; interpretable; CI gate; fallback when no checkpoint | Default predictor after training; RMAV = fallback |

> **Validation-source note:** the GNN values above are the HGL-QoS per-domain repeated k-fold results (`k=5`, 5 seeds, `cli/kfold_evaluate.py`) using simulation labels, evaluated independently within each of seven scenarios and averaged across scenarios (`ρ = 0.587 ± 0.146`, `F1@K = 0.505`; positive in all seven scenarios individually, range `ρ = 0.341–0.781`). This is an *in-domain* metric — the model is trained and evaluated on the same scenario, repeated under resampling to confirm the result is stable rather than an artifact of one split — not a claim about zero-shot transfer to a scenario the model was never trained on. The older cross-scenario Leave-One-Scenario-Out (LOSO) protocol, which does test zero-shot transfer, remains available (`cli/loso_evaluate.py`) and reached `ρ = 0.290` (`F1@K = 0.405`, `HGL-QoS`) under that harder, out-of-domain regime; it is retained as a secondary domain-gap analysis rather than the primary validation metric, since testing transfer between architecturally distinct scenarios (autonomous-vehicle vs. financial-trading vs. hub-and-spoke topologies, etc.) conflates model quality with how much structure those unrelated domains happen to share.

---

## 4. Output Schema

The JSON output from `predict_graph.py --output results/prediction.json` follows this structure when a GNN checkpoint is provided:

```json
{
  "layers": {
    "system": {
      "total_components": 35,
      "rmav": {
        "NavLib": {
          "overall":          0.54,
          "reliability":      0.63,
          "maintainability":  0.41,
          "availability":     0.58,
          "vulnerability":    0.52,
          "level":            "HIGH",
          "is_spof":          true,
          "blast_radius":     12,
          "cascade_depth":    4
        }
      },
      "antipatterns": [
        {
          "entity_id":    "NavLib",
          "entity_type":  "Component",
          "name":         "Single Point of Failure (SPOF)",
          "severity":     "CRITICAL",
          "category":     "Availability",
          "description":  "NavLib is a directed cut vertex. Removing it partitions the dependency graph.",
          "recommendation": "Introduce redundancy: backup instances or alternative paths.",
          "evidence":     { "is_articulation_point": true, "availability_score": 0.58 }
        }
      ],
      "gnn": {
        "prediction_mode": "gnn_only",
        "node_scores": {
          "NavLib": {
            "component": "NavLib",
            "composite_score": 0.5432,
            "reliability_score": 0.6321,
            "maintainability_score": 0.4121,
            "availability_score": 0.5821,
            "vulnerability_score": 0.5211,
            "criticality_level": "HIGH",
            "source": "GNN"
          }
        },
        "edge_scores": [
          {
            "source": "MonitorApp",
            "target": "SensorApp",
            "edge_type": "DEPENDS_ON",
            "composite_score": 0.4512,
            "reliability_score": 0.3211,
            "maintainability_score": 0.2512,
            "availability_score": 0.5121,
            "vulnerability_score": 0.4211,
            "criticality_level": "MEDIUM"
          }
        ],
        "gnn_metrics": {
          "spearman_rho": 0.8912,
          "f1_score": 0.9021,
          "rmse": 0.0812,
          "mae": 0.0612,
          "ndcg_10": 0.9211
        }
      }
    }
  }
}
```

---

## 5. Commands

```bash
# ─── GNN training (requires Step 4 simulation results) ────────────────────────

# Single-seed training
PYTHONPATH=. python cli/train_graph.py --layer system

# Multi-seed stability training
PYTHONPATH=. python cli/train_graph.py --layer system --seeds 42 123 456 789 2024

# Multi-graph inductive training (all domain scenarios)
PYTHONPATH=. python cli/train_graph.py --layer system --multi-scenario

# ─── GNN inference ────────────────────────────────────────────────────────────

# GNN inference using a trained checkpoint
PYTHONPATH=. python cli/predict_graph.py --gnn-model output/gnn_checkpoints/best_model
```

---

## 6. What Comes Next

Step 3 has two operational modes. For inference, a trained checkpoint lets Step 3 run after Step 2 and emit GNN predictions; Step 4 and Step 5 are then used to validate those predictions against simulation ground truth. For GNN training, Step 4 must come first because simulation labels `I(v)` are required, followed by Step 5 validation to measure GNN performance.

→ [Step 4: Simulate](failure-simulation.md)
