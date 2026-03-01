# GNN Integration: Graph Neural Networks for Distributed System Criticality Prediction

**Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems**
Istanbul Technical University, Computer Engineering Department

← [Step 3: Quality Scoring](quality-scoring.md) | → [Step 5: Validation](validation.md)

---

## Table of Contents

1. [Motivation and Position in the Pipeline](#1-motivation-and-position-in-the-pipeline)
2. [Architecture Overview](#2-architecture-overview)
3. [Node Feature Construction](#3-node-feature-construction)
4. [Heterogeneous Graph Attention Network](#4-heterogeneous-graph-attention-network)
5. [Multi-Task Learning: RMAV Dimensions](#5-multi-task-learning-rmav-dimensions)
6. [Edge Criticality Prediction](#6-edge-criticality-prediction)
7. [Ensemble: GNN + RMAV](#7-ensemble-gnn--rmav)
8. [Loss Function](#8-loss-function)
9. [Training Protocol](#9-training-protocol)
10. [Validation and Metric Parity](#10-validation-and-metric-parity)
11. [Relationship to Existing Pipeline](#11-relationship-to-existing-pipeline)
12. [Commands](#12-commands)
13. [Expected Results and Interpretation](#13-expected-results-and-interpretation)
14. [Limitations and Future Work](#14-limitations-and-future-work)

---

## 1. Motivation and Position in the Pipeline

The existing six-step pipeline predicts component criticality using hand-crafted topological metrics assembled by the RMAV quality scorer (Step 3). This approach has strong interpretability — a high RMAV score can always be decomposed into specific metric contributions — and achieves Spearman ρ = 0.876 across validated system scales.

GNN integration extends this in two ways that handcrafted metrics cannot easily achieve:

**Learned feature interactions.** RMAV combines 13 metrics via fixed AHP-derived weights. GNNs learn which combinations of structural signals actually predict failure impact in a data-driven way. The attention mechanism further learns *which neighbours matter most* for each component, enabling context-sensitive weighting that a static formula cannot express.

**Relationship (edge) criticality.** The existing pipeline scores nodes; edges are analysed only via structural proxies (bridge ratio, betweenness of endpoints). The `EdgeCriticalityGNN` directly scores each pub-sub relationship — identifying which data flows are most dangerous to lose — enabling a new class of architectural recommendation.

The GNN is inserted as **Step 3.5** between Quality Scoring and Failure Simulation, and its predictions are validated against the same `I(v)` ground truth as RMAV, ensuring results are directly comparable.

```
Step 1: Graph Model Construction       G(V, E, w)
Step 2: Structural Analysis            M(v) — 13 metrics
Step 3: Quality Scoring (RMAV)         Q*(v) — AHP-weighted
Step 3.5: GNN Criticality Scoring  ←  Q_GNN(v) — LEARNED       [NEW]
          Ensemble                 ←  Q_ens(v) = α·Q_GNN + (1−α)·Q_RMAV  [NEW]
Step 4: Failure Simulation             I*(v) — ground truth
Step 5: Validation                     ρ(Q_GNN, I*) vs ρ(Q_RMAV, I*)     [EXTENDED]
Step 6: Visualization                  Dashboard                            [EXTENDED]
```

---

## 2. Architecture Overview

Three cooperating components form the GNN extension:

```
                    NetworkX DiGraph
                          │
                ┌─────────▼──────────┐
                │   Data Preparation  │  networkx_to_hetero_data()
                │   HeteroData        │  node features (18-dim)
                │   node/edge splits  │  edge features (8-dim)
                │   labels  I(v)      │  labels (5-dim)
                └─────────┬──────────┘
                          │
          ┌───────────────┼──────────────────────┐
          │               │                      │
┌─────────▼──────┐ ┌──────▼─────────┐  ┌────────▼────────┐
│ NodeCriticality│ │ EdgeCriticality │  │  EnsembleGNN    │
│ GNN            │ │ GNN            │  │                 │
│ HeteroGAT      │ │ (shares node   │  │ α·Q_GNN         │
│ 3 layers       │ │  backbone)     │  │ +(1−α)·Q_RMAV   │
│ 4 heads        │ │ Edge MLP head  │  │                 │
│ Output: (N,5)  │ │ Output: (E,5)  │  │ Output: (N,5)   │
└────────────────┘ └────────────────┘  └─────────────────┘
          │               │                      │
          └───────────────┴──────────────────────┘
                          │
                ┌─────────▼──────────┐
                │   GNNAnalysisResult │
                │   node_scores       │
                │   edge_scores       │
                │   ensemble_scores   │
                │   validation metrics│
                └────────────────────┘
```

---

## 3. Node Feature Construction

Each node `v` is represented by an **18-dimensional feature vector** derived directly from the outputs of Steps 2 and 3, ensuring the GNN starts with the same information as the RMAV scorer and learns on top of it.

### 3.1 Topological Metrics (indices 0–12)

| Index | Metric | RMAV Role | Captures |
|-------|--------|-----------|----------|
| 0 | PageRank (PR) | Maintainability | Transitive dependency importance |
| 1 | Reverse PageRank (RPR) | Reliability | Cascade propagation reach |
| 2 | Betweenness Centrality (BT) | Maintainability | Structural bottleneck position |
| 3 | Closeness Centrality (CL) | Vulnerability | Speed of fault propagation |
| 4 | Eigenvector Centrality (EV) | Vulnerability | Influence of high-value neighbours |
| 5 | In-Degree (DG_in) | Reliability | Direct dependent count |
| 6 | Out-Degree (DG_out) | Maintainability | Direct dependency count |
| 7 | Clustering Coefficient (CC) | Maintainability | Local redundancy |
| 8 | Continuous AP score (AP_c) | Availability | SPOF severity |
| 9 | Bridge Ratio (BR) | Availability | Fraction of non-redundant edges |
| 10 | QoS weight aggregate (w) | Availability | Overall QoS criticality |
| 11 | QoS weighted in-degree (w_in) | Reliability | Priority-weighted dependents |
| 12 | QoS weighted out-degree (w_out) | Maintainability | Priority-weighted dependencies |

All metrics are already normalised to `[0, 1]` by Step 2's min-max normalisation, so no additional scaling is required.

### 3.2 Node Type One-Hot (indices 13–17)

| Index | Node Type |
|-------|-----------|
| 13 | Application |
| 14 | Broker |
| 15 | Topic |
| 16 | Node (infrastructure) |
| 17 | Library |

The type encoding enables the GNN to learn type-specific criticality patterns — for example, that Broker failures tend to cause higher cascade impact than Library failures — without requiring separate model instances per type.

---

## 4. Heterogeneous Graph Attention Network

### 4.1 Why Heterogeneous?

The pub-sub multi-layer graph contains five distinct node types and seven edge types. A homogeneous GNN applies the same transformation to all message-passing relations, conflating semantically different relationships like `PUBLISHES_TO` (an application pushing data to a topic) with `RUNS_ON` (an application executing on a physical node). A heterogeneous GNN maintains separate weight matrices per relation type, preserving these distinctions.

### 4.2 Graph Attention (GAT) Layer

Within each heterogeneous message-passing step, the update for node `v` of type `dst` receiving messages from neighbours `u` of type `src` via relation `(src, rel, dst)` is:

```
h_v^{(l+1)} = σ(Σ_{u ∈ N(v)} α_{uv}^{(rel)} · W^{(rel)} · h_u^{(l)})
```

where the attention coefficient is:

```
α_{uv}^{(rel)} = softmax_u( LeakyReLU( a^{(rel)T} · [W^{(rel)} h_u ‖ W^{(rel)} h_v ‖ W_e · e_{uv}] ) )
```

`e_{uv}` is the 8-dimensional edge feature vector (QoS weight + edge type one-hot), incorporated via an edge-attribute projection `W_e`. This allows the model to learn that a `PUBLISHES_TO` edge with `reliability=RELIABLE, durability=TRANSIENT_LOCAL` carries different criticality signal than one with `reliability=BEST_EFFORT`.

### 4.3 Multi-Head Attention

With `H = 4` attention heads and hidden dimension `D = 64`, each head uses `d = D/H = 16` dimensions. Outputs are concatenated and projected:

```
h_v^{(l+1)} = ‖_{h=1}^{H} σ(Σ_{u} α_{uv,h}^{(rel)} · W_h^{(rel)} · h_u^{(l)})
```

### 4.4 HeteroConv Aggregation

When a node `v` receives messages from multiple relation types simultaneously (e.g., an Application node is the target of `DEPENDS_ON` edges *and* a `RUNS_ON` edge), the HeteroConv layer aggregates contributions from all incoming relations via **mean pooling**:

```
h_v^{(l+1)} = mean_{rel ∈ R(dst_type)} h_v^{(l+1, rel)}
```

### 4.5 Residual Connections and Layer Normalisation

After each message-passing step, the update includes a residual connection and layer normalisation to stabilise training on small graphs:

```
h_v^{(l+1)} ← LayerNorm( h_v^{(l+1)} + h_v^{(l)} )
```

with GELU activation and dropout `p = 0.2`.

### 4.6 Input Projection

Before the first message-passing layer, each node type uses a **type-specific linear projection** to map the 18-dimensional feature vector to the hidden dimension:

```
h_v^{(0)} = GELU( LayerNorm( W_{type(v)} · x_v + b_{type(v)} ) )
```

This allows the model to learn different feature combinations per node type before message passing begins.

---

## 5. Multi-Task Learning: RMAV Dimensions

Rather than predicting a single composite criticality score, the model uses **four dimension-specific MLP heads** plus one composite head, mirroring the RMAV decomposition:

```
R̂(v) = MLP_R( h_v )     — Reliability prediction head
M̂(v) = MLP_M( h_v )     — Maintainability prediction head
Â(v) = MLP_A( h_v )     — Availability prediction head
V̂(v) = MLP_V( h_v )     — Vulnerability prediction head

Î*(v) = MLP_C( h_v ‖ R̂(v) ‖ M̂(v) ‖ Â(v) ‖ V̂(v) )   — Composite
```

The composite head receives the dimension predictions as additional inputs, allowing it to learn how to combine them in a data-driven way that complements the fixed AHP weighting in RMAV.

All outputs pass through a sigmoid activation, producing scores in `[0, 1]` consistent with the existing `Q*(v)` and `I*(v)` value ranges.

---

## 6. Edge Criticality Prediction

For each edge `(u, v)` with type `rel`:

```
score(u,v) = MLP_E( h_u ‖ h_v ‖ e_{uv} )
```

where:
- `h_u`, `h_v` are the final node embeddings (dim `D = 64` each)
- `e_{uv}` is the 8-dimensional edge feature vector
- `MLP_E` is a 3-layer MLP with sigmoid output

This produces five scores per edge (composite, R, M, A, V), enabling relationship-level RMAV analysis in addition to component-level.

**Edge labels** for training are derived from simulation results as:

```
I_edge(u,v) = max( I*(u), I*(v) )
```

The max pooling semantics capture that a pub-sub relationship is only as robust as its most critical endpoint.

---

## 7. Ensemble: GNN + RMAV

The ensemble combines GNN predictions with existing RMAV scores via a **learnable convex combination**:

```
Q_ens(v) = α · Q_GNN(v) + (1 − α) · Q_RMAV(v)
```

`α ∈ (0, 1)` is a learned scalar per RMAV dimension (5 scalars total), stored in logit space for unconstrained optimisation:

```
α = sigmoid( α_logit ),   α_logit initialised to 0  → α = 0.5
```

The ensemble is fine-tuned by minimising MSE on training nodes with labelled simulation ground truth. This discovers the optimal per-dimension blending — for example, if GNN learns to predict Availability (SPOF detection) better than RMAV but RMAV is stronger on Vulnerability, the ensemble will weight accordingly.

When RMAV scores are unavailable (pure inference mode), the ensemble passes GNN predictions through unchanged.

---

## 8. Loss Function

Training uses a three-component loss:

### 8.1 Composite MSE

```
L_composite = (1/N) Σ_v (Î*(v) − I*(v))²
```

Primary task: match the simulation-derived composite impact score.

### 8.2 RMAV Auxiliary MSE

```
L_RMAV = (1/N) Σ_v Σ_{d ∈ {R,M,A,V}} (d̂(v) − I_d(v))²
```

Auxiliary task: match each per-dimension simulation ground truth `IR(v)`, `IM(v)`, `IA(v)`, `IV(v)`. Weight coefficient `λ_RMAV = 0.5`.

### 8.3 ListMLE Ranking Loss

```
L_rank = −(1/N) Σ_{v in sorted order} log P(v-th position)
```

Encourages the model to correctly order components by criticality, directly optimising the Spearman rank correlation. Weight coefficient `λ_rank = 0.3`.

### 8.4 Total Loss

```
L = L_composite + 0.5 · L_RMAV + 0.3 · L_rank
```

---

## 9. Training Protocol

### 9.1 Transductive Setting (single graph)

When only one system topology is available, nodes are randomly split 60/20/20 into train/val/test sets. The GNN is trained on train nodes, tuned on val nodes (early stopping), and evaluated on test nodes — a standard inductive-to-transductive protocol.

**Early stopping**: Training halts when validation Spearman ρ does not improve for 30 consecutive epochs. Best model weights are restored before test evaluation.

**Optimiser**: AdamW with `lr = 3×10⁻⁴`, `weight_decay = 10⁻⁴`, cosine annealing schedule.

**Gradient clipping**: `max_norm = 1.0` to handle the sparsity of small graphs.

### 9.2 Inductive Setting (multiple graphs)

When all eight domain scenarios are available (ROS 2, IoT, financial, healthcare, and their scaled variants), the trainer supports inductive multi-graph learning: train on a subset of system instances, evaluate generalisation on held-out instances. This is the recommended setting for strong transfer learning capability and is configured via the `--multi-scenario` flag.

---

## 10. Validation and Metric Parity

GNN predictions are evaluated using exactly the same metric suite as RMAV (Step 5 of the existing pipeline), enabling direct comparison:

| Metric | RMAV Target | GNN Target |
|--------|-------------|------------|
| Spearman ρ (composite) | ≥ 0.70 | ≥ 0.70 |
| F1 (composite) | ≥ 0.90 | ≥ 0.90 |
| RMSE | ≤ 0.25 | ≤ 0.25 |
| MAE | ≤ 0.20 | ≤ 0.20 |
| Top-5 Overlap | — | reported |
| Top-10 Overlap | — | reported |
| NDCG@10 | — | reported |
| Edge Spearman ρ | N/A | new metric |

RMAV achieved Spearman ρ = 0.876 overall (0.943 at large scale). GNN is expected to match or improve on this, with the most significant gains anticipated in:
- **Systems with complex multi-hop dependencies** (GNN captures longer-range structural patterns)
- **Edge criticality** (wholly new capability, no RMAV baseline)
- **Transfer across domains** (GNN learned representations generalise; RMAV does not)

---

## 11. Relationship to Existing Pipeline

The GNN module follows the hexagonal architecture of the existing codebase:

| Existing Component | GNN Counterpart | Relationship |
|-------------------|-----------------|--------------|
| `StructuralAnalyzer` | `data_preparation.py` | Consumes M(v) as node features |
| `AnalysisService.analyze_layer()` | `GNNService.predict()` | Parallel analysis path |
| `SimulationService` | Training labels | Consumed, not replaced |
| `ValidationService` | `trainer.evaluate()` | Same metric protocol |
| `LayerAnalysisResult` | `GNNAnalysisResult` | Complementary result object |
| `CompositeCriticalityScore` | `GNNCriticalityScore` | Same interface for downstream use |

The GNN does **not** replace any existing step. It is an additional analysis path that produces complementary predictions.

---

## 12. Commands

```bash
# ── Train GNN on a system graph ──────────────────────────────────────────────
python bin/train_graph.py --layer app

# Train with custom hyperparameters
python bin/train_graph.py --layer system \\
    --hidden 128 --heads 8 --layers 4 \\
    --epochs 500 --patience 50

# Train from pre-computed results (skip Neo4j)
python bin/train_graph.py \\
    --structural results/metrics.json \\
    --simulated  results/impact.json \\
    --rmav       results/quality.json \\
    --checkpoint output/gnn_checkpoints/

# Export training results
python bin/train_graph.py --layer app --output results/gnn_train_result.json

# ── Inference on new graph ────────────────────────────────────────────────────
python bin/predict_graph.py --layer app --checkpoint output/gnn_checkpoints/

# With RMAV comparison and edge scores
python bin/predict_graph.py --layer system \\
    --checkpoint output/gnn_checkpoints/ \\
    --compare-rmav \\
    --show-edges \\
    --top-n 20

# Validate against simulation ground truth
python bin/predict_graph.py --layer app \\
    --checkpoint output/gnn_checkpoints/ \\
    --simulated results/impact.json

# ── Programmatic API ──────────────────────────────────────────────────────────
from src.gnn import GNNService

# Train
service = GNNService()
result  = service.train(graph, structural_metrics, simulation_results, rmav_scores)

# Save and reload
service.save("output/gnn_checkpoints/")
service2 = GNNService.from_checkpoint("output/gnn_checkpoints/", graph=nx_graph)

# Predict
result = service2.predict(graph, structural_metrics, rmav_scores)
print(result.top_critical_nodes(10))
print(result.top_critical_edges(10))
print(result.summary())
```

---

## 13. Expected Results and Interpretation

### Reading GNN Output

```
Top 10 Critical Components (Ensemble)
  #    Component                      Score   Level      R      M      A      V
  ──────────────────────────────────────────────────────────────────────────────
  1    DataRouter                    0.9142  CRITICAL  0.921  0.843  0.967  0.812
  2    SensorHub                     0.8731  CRITICAL  0.891  0.782  0.923  0.771
  3    CommandGateway                0.7419  HIGH      0.701  0.883  0.652  0.741
  ...
```

Interpreting RMAV dimension scores:
- **R (Reliability)**: High → failure propagates widely through dependents
- **M (Maintainability)**: High → tightly coupled, structural bottleneck
- **A (Availability)**: High → SPOF or near-SPOF; removal partitions the graph
- **V (Vulnerability)**: High → central, reachable, high-value attack target

### Ensemble Alpha Interpretation

```
Learned ensemble weights (α = GNN contribution):
  composite            : GNN  62.3% ████████████░░░░░░░░ RMAV  37.7%
  reliability          : GNN  58.1% ████████████░░░░░░░░ RMAV  41.9%
  maintainability      : GNN  44.7% █████████░░░░░░░░░░░ RMAV  55.3%
  availability         : GNN  71.2% ██████████████░░░░░░ RMAV  28.8%
  vulnerability        : GNN  49.3% ██████████░░░░░░░░░░ RMAV  50.7%
```

An α > 0.5 for a dimension means the GNN learned a stronger signal than RMAV for that dimension. High α on Availability suggests the GNN captures SPOF patterns better than the handcrafted AP_c metric; high RMAV contribution on Maintainability suggests the AHP-weighted betweenness + coupling risk formula is hard to improve via learning alone.

---

## 14. Limitations and Future Work

### Current Limitations

**Data volume**: GNN training is most powerful with many labelled instances. With a single graph of 35 components, the transductive setting limits how much the model can generalise beyond memorisation. Running all eight domain scenarios and using inductive training significantly mitigates this.

**Cold start for new node types**: Adding a new component type (e.g., a new middleware abstraction) requires re-training. The RMAV scorer handles this automatically.

**Graph metadata alignment**: The HeteroData metadata (relation types) must match between training and inference graphs. The service handles this gracefully but requires the same edge types to be present.

### Planned Extensions

**Temporal GNN**: Extend to `T`-graphs representing system topology at different points in time (before/after a deployment change). A temporal GNN can predict how criticality distributions shift as the system evolves, enabling predictive maintenance.

**Inductive transfer across architectural styles**: Train on pub-sub systems and fine-tune on REST microservices, using GraphSAGE-style neighbourhood sampling for inductive generalisation.

**GNN-guided refactoring**: Use the attention weights `α_{uv}` to identify which specific dependencies drive a component's criticality, generating targeted refactoring recommendations ("remove edge X to reduce DataRouter criticality by 23%").

**Uncertainty quantification**: Add MC-Dropout or Deep Ensembles to produce confidence intervals on criticality scores, distinguishing high-confidence CRITICAL predictions from uncertain ones at the class boundary.

---

← [Step 3: Quality Scoring](quality-scoring.md) | → [Step 5: Validation](validation.md)

---

*Software-as-a-Graph Framework v2.3 · March 2026*
*Istanbul Technical University, Computer Engineering Department*
