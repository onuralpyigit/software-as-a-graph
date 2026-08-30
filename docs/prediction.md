# Step 3: Predict — Inductive Blast-Radius Forecasting (HGT)

**Rank architectural components by predicted failure blast radius using a Heterogeneous Graph Transformer (Pathway B: Predictive), with the deterministic ISO-RM composite computed underneath as the model's own input feature and zero-checkpoint fallback.**

← [Step 2: Analyze](structural-analysis.md) | → [Step 4: Diagnose](diagnosis.md)

---

## Table of Contents

1. [Overview & Dual-Pathway Architecture](#1-overview--dual-pathway-architecture)
2. [Pathway B: Inductive Learned Criticality & Blast-Radius Forecasting (HGT)](#2-pathway-b-inductive-learned-criticality--blast-radius-forecasting-hgt)
   - 2.1 [Heterogeneous Graph Transformer (HGT) Message Passing](#21-heterogeneous-graph-transformer-hgt-message-passing)
   - 2.2 [16-Dimensional QoS Edge Encodings](#22-16-dimensional-qos-edge-encodings)
   - 2.3 [Zero-GNN Cold-Start Fallback](#23-zero-gnn-cold-start-fallback)
3. [Graph Data Preparation (PyTorch Geometric `HeteroData`)](#3-graph-data-preparation-pytorch-geometric-heterodata)
   - 3.1 [Node Feature Schema](#31-node-feature-schema)
   - 3.2 [Edge Feature Schema (16 Dimensions)](#32-edge-feature-schema-16-dimensions)
   - 3.3 [Target Labels & Dimension Masking](#33-target-labels--dimension-masking)
4. [Model Architecture & Prediction Heads](#4-model-architecture--prediction-heads)
   - 4.1 [Backbone Structure](#41-backbone-structure)
   - 4.2 [Multi-Task Node Prediction Heads](#42-multi-task-node-prediction-heads)
   - 4.3 [Relation-Specific Edge Prediction Head](#43-relation-specific-edge-prediction-head)
5. [Training Protocol & Multi-Task Loss Formulation](#5-training-protocol--multi-task-loss-formulation)
   - 5.1 [Multi-Task Loss Equation](#51-multi-task-loss-equation)
   - 5.2 [Training Hyperparameters & Optimisation](#52-training-hyperparameters--optimisation)
6. [Programmatic Python SDK & Use Cases](#6-programmatic-python-sdk--use-cases)
   - 6.1 [Pipeline Integration](#61-pipeline-integration)
   - 6.2 [Direct Use Case Execution (`saag.usecases`)](#62-direct-use-case-execution-saagusecases)
7. [CLI Reference & Commands](#7-cli-reference--commands)
8. [Output Schema & Sample JSON](#8-output-schema--sample-json)
9. [Known Limitations & Design Boundaries](#9-known-limitations--design-boundaries)
10. [What Comes Next](#10-what-comes-next)

---

## 1. Overview & Dual-Pathway Architecture

Steps 3 and 4 together are the **core analytical and predictive engine** of the Software-as-a-Graph (SaG) framework, split into two deliberately separate stages rather than forcing a single model to act simultaneously as a statistical estimator and an explainable standards compliance checker. This document covers **Step 3 (Predict, Pathway B)** — the learned ranking engine; see [diagnosis.md](diagnosis.md) for **Step 4 (Diagnose, Pathway A)** — the deterministic root-cause engine and the Triage Bridge that joins the two.

```mermaid
flowchart TD
    M["Step 2 Output<br>StructuralAnalysisResult M(v) & Graph G"] --> PE["Step 3 + Step 4: Prediction Engine"]

    subgraph PathRM["Step 4: Diagnose — Pathway A / ISO-RM (Always Active)"]
        PE --> RM["Closed-Form RM Scoring<br>FT(v), A(v), R(v), M(v), Q*(v)"]
        RM --> AP["19 Anti-Pattern Auditing & Explanations"]
        RM --> OUT_RM["Diagnostic Quality Profiles Q*(v)"]
    end

    subgraph PathGNN["Step 3: Predict — Pathway B / HGL (Opt-In / Checkpoint)"]
        PE --> HGT["Heterogeneous Graph Transformer (HGT)"]
        HGT --> NH["Node Heads: R̂(v), M̂(v), Composite Î*(v)"]
        HGT --> EH["TypedEdgeEncoder: Edge Criticality Q(u,v)"]
        NH --> TOPK["Top-K Critical Components (Shortlist)"]
    end

    TOPK --> TB["TRIAGE BRIDGE (Step 4)<br>(Join on component_id)"]
    OUT_RM --> TB
    TB --> SO["Stakeholder Remediation Action Profiles<br>(DevOps/SRE, Architect, Developer)"]
    SO --> PRESCRIBE["Step 7: Prescribe & Closed-Loop Gating"]
    NH --> VALIDATE["Step 6: Statistical Validation (Gate G1-G6, G8)"]
```

### Key Architectural Invariants
1. **Parameter Independence**: Pathway A (Step 4) and Pathway B (Step 3) share no learned weights; neither is fitted to the other's output.
2. **Offline Oracle Separation**: Simulation (Step 5) acts solely as an offline supervisor generating supervised training labels ($I^*(v)$ via `FaultInjector`) and serving as the validation oracle (`FailureSimulator`). Step 3 consumes labels from disk during training but has zero runtime dependency on the simulation engine.
3. **No Hallucination in Root-Cause Attribution**: The Triage Bridge (Step 4) joins quantitative rankings to qualitative RM diagnostics strictly by component ID, preventing neural models from guessing unverified architectural root causes.

---

## 2. Pathway B: Inductive Learned Criticality & Blast-Radius Forecasting (HGT)

Pathway B evaluates multi-hop, non-linear failure dynamics that exceed closed-form 1-hop and 2-hop structural metrics.

### 2.1 Heterogeneous Graph Transformer (HGT) Message Passing
The machine learning backbone uses stacked `torch_geometric.nn.HGTConv` layers with multi-head attention ($H=4$, $D=64$, dropout $p=0.2$). Heterogeneous node types (`Application`, `Library`, `Broker`, `Topic`, `Node`) and typed edges are parameterized with type-specific projection matrices.

### 2.2 16-Dimensional QoS Edge Encodings
Edge representations capture both topological connectivity and declared transport Quality-of-Service contracts (Reliability, Durability, Priority, Deadlines, Blocking Timeouts, and Heterogeneity Flags). An `EdgeFeatureEncoder` injects these 16-D representations into target nodes via scatter-mean aggregation prior to each convolution layer.

### 2.3 Zero-GNN Cold-Start Fallback
When no trained GNN checkpoint is available on disk, Step 3 falls back gracefully to the same deterministic $Q^*(v)$ ranking Step 4 (Diagnose) computes — the RM composite is always scored first, as both Step 3's own input feature and its fallback. The system never crashes in uncalibrated or cold-start deployments; see [diagnosis.md §2](diagnosis.md#2-pathway-a-deterministic-diagnostic-quality-attribution-iso-rm) for the RM composite itself.

---

## 3. Graph Data Preparation (PyTorch Geometric `HeteroData`)

[`networkx_to_hetero_data()`](../saag/prediction/data_preparation.py) converts the NetworkX graph into a PyTorch Geometric `HeteroData` multigraph structure, partitioning nodes and edges by entity type.

### 3.1 Node Feature Schema

Node vectors combine a **shared 18-dimensional topological base** (indices 0–17) with **type-specific attributes** (indices 18+):

| Node Type | Total Dimensions | Type-Specific Extensions (Indices 18+) |
|:---|:---:|:---|
| `Application` | **23** | 5 Code Quality attributes (`loc_norm`, `complexity_norm`, $I_{\text{code}}$, `lcom_norm`, $CQP$) |
| `Library` | **23** | 5 Code Quality attributes (`loc_norm`, `complexity_norm`, $I_{\text{code}}$, `lcom_norm`, $CQP$) |
| `Broker` | **19** | `max_connections_norm` |
| `Topic` | **22** | `subscriber_count_norm`, `publisher_count_norm`, `log1p_frequency_norm`, `topic_qos_criticality_ord` |
| `Node` (Infra) | **20** | `cpu_cores_norm`, `memory_gb_norm` |

```mermaid
graph LR
    subgraph NodeVec["Node Feature Vector"]
        Base["Indices 0–17:<br>Shared Topological Base<br>(PR, RPR, BT, DG_in, AP_c_dir, CDI, w, etc.)"]
        Ext["Indices 18+:<br>Type-Specific Attributes<br>(Code metrics, HW cores, Topic frequencies)"]
    end
    Base --> Ext
```

#### Shared Topological Base (Indices 0–17 across all Node Types)
- `0`: PageRank ($PR$)
- `1`: Reverse PageRank ($RPR$)
- `2`: Betweenness Centrality ($BT$)
- `3`: Closeness Centrality ($CL$)
- `4`: Eigenvector Centrality ($EV$)
- `5`: In-Degree Normalised ($DG_{in}$)
- `6`: Out-Degree Normalised ($DG_{out}$)
- `7`: Clustering Coefficient ($CC$)
- `8`: Undirected Articulation Score
- `9`: Bridge Ratio ($BR$)
- `10`: Component QoS Weight ($w(v)$)
- `11`: QoS-Weighted In-Degree ($w_{in}$)
- `12`: QoS-Weighted Out-Degree ($w_{out}$)
- `13`: Multi-Path Coupling Index ($MPCI$)
- `14`: Path Complexity ($PC$)
- `15`: Fan-Out Criticality ($FOC$)
- `16`: Directed Articulation Point ($AP_c^{\text{dir}}$)
- `17`: Connectivity Degradation Index ($CDI$)

### 3.2 Edge Feature Schema (16 Dimensions)

Edge features capture both topological connectivity and declared transport QoS delivery guarantees:

| Index | Feature Key | Semantic Meaning |
|:---:|:---|:---|
| **0** | `qos_weight` | Continuous QoS weight $w(e) \in [0, 1]$ |
| **1** | `path_count_norm` | Normalized channel count: $\log_2(1 + \text{path\_count}) / \log_2(17)$ |
| **2–8** | `edge_type_one_hot` | One-hot indicator (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`, `DEPENDS_ON`) |
| **9** | `reliability_score` | QoS Reliability (`BEST_EFFORT` = 0.0, `RELIABLE` = 1.0) |
| **10** | `durability_score` | QoS Durability (`VOLATILE` = 0.0, `TRANSIENT_LOCAL` = 0.5, `TRANSIENT` = 0.6, `PERSISTENT` = 1.0) |
| **11** | `priority_score` | Transport Priority (`LOW` = 0.0, `MEDIUM` = 0.33, `HIGH` = 0.66, `URGENT` = 1.0) |
| **12** | `has_deadline` | Binary flag (1.0 if finite contract deadline is configured) |
| **13** | `deadline_ns_log` | Scaled contract deadline duration |
| **14** | `max_blocking_ms_log`| Scaled blocking timeout duration |
| **15** | `qos_heterogeneity_flag`| Flag indicating edge QoS diverges from system mode |

*(Note: Indices 9–15 are active for pub/sub interaction links; structural links default to 0).*

### 3.3 Target Labels & Dimension Masking

| Tensor Name | Target Shape | Semantic Purpose |
|:---|:---:|:---|
| `data[type].y` | $(N, 3)$ | Simulation ground-truth vectors: $[I^*(v), I_R(v), I_M(v)]$ |
| `data[type].y_rm` | $(N, 3)$ | Rule-based consistency regularization target: $[Q(v), R(v), M(v)]$. Populated whenever `rm_scores` are supplied; only consumed as a training signal when `rm_consistency_weight > 0` (default 0.0 — the diagnostic and predictive pathways are trained independently unless this ablation is opted into explicitly). |
| `data[type].label_mask` | $(N,)$ | Boolean mask indicating which nodes were simulated (excludes unlabelled nodes) |
| `data[type].dimension_mask`| $(3,)$ | Boolean mask indicating measured ground-truth columns (masks unmeasured targets) |
| `data[rel].y_edge` | $(E, 3)$ | Per-edge ground-truth criticality labels |

---

## 4. Model Architecture & Prediction Heads

```mermaid
flowchart TD
    subgraph Input["1. Input Embeddings"]
        X_V["Type-Specific Node Features x_v"] --> LinV["Type Linear Projections"]
        E_UV["16-Dim Edge Features e_uv"] --> EFE["EdgeFeatureEncoder"]
    end

    subgraph Backbone["2. Heterogeneous Message Passing (3 Layers)"]
        LinV --> HGT1["HGT Layer 1 + Residual + LayerNorm"]
        EFE -.->|Scatter-Mean Injection| HGT1
        HGT1 --> HGT2["HGT Layer 2 + Residual + LayerNorm"]
        EFE -.->|Scatter-Mean Injection| HGT2
        HGT2 --> HGT3["HGT Layer 3 + Residual + LayerNorm"]
        EFE -.->|Scatter-Mean Injection| HGT3
        HGT3 --> RevPass["Optional Bidirectional Reverse Pass"]
    end

    subgraph NodeHeads["3. Multi-Task Node Prediction Heads"]
        RevPass --> HeadR["Reliability Head R̂(v)"]
        RevPass --> HeadM["Maintainability Head M̂(v)"]
        RevPass --> Fuse["Concatenate [h_v || R̂ || M̂]"]
        HeadR --> Fuse
        HeadM --> Fuse
        Fuse --> HeadC["Composite Head Î*(v)"]
    end

    subgraph EdgeHead["4. Relation-Specific Edge Prediction Head"]
        RevPass --> TEE["TypedEdgeEncoder(h_u, h_v, e_uv)"]
        E_UV --> TEE
        TEE --> EdgeOut["Edge Criticality Q_GNN(u,v)"]
    end
```

### 4.1 Backbone Structure

The backbone consists of **3 layers of Heterogeneous Graph Transformer (`HGTConv`)** with hidden dimension $D = 64$, 4 attention heads, and dropout $p = 0.2$.

1. **Type-Specific Input Projection**:
   $$\mathbf{h}_v^{(0)} = \text{GELU}(\text{LayerNorm}(\mathbf{W}_{\text{type}(v)} \mathbf{x}_v))$$
2. **Edge Feature Injection & Convolution**:
   Edge attributes are projected and scatter-mean injected into target nodes before each convolution:
   $$\mathbf{h}_d \leftarrow \mathbf{h}_d + \frac{1}{|\mathcal{N}(d)|} \sum_{u \in \mathcal{N}(d)} \mathbf{W}_{\text{edge}}^{(k)} \mathbf{e}_{ud}$$
   $$\mathbf{h}_v^{(k+1)} = \text{Dropout}\left(\text{GELU}\left(\text{LayerNorm}\left(\text{HGTConv}_k(\mathbf{h}^{(k)}, \mathcal{E}) + \mathbf{h}_v^{(k)}\right)\right)\right)$$
3. **Bidirectional Reverse Flow**:
   When `use_bidirectional=True`, an inverted convolution pass transmits upstream signals:
   $$\mathbf{h}_v \leftarrow \mathbf{h}_v + 0.5 \cdot \mathbf{h}_v^{\text{rev}}$$

### 4.2 Multi-Task Node Prediction Heads

All prediction heads utilize `ResidualMLP` networks with final sigmoid activations bounding outputs in $[0, 1]$:

$$\begin{aligned}
\hat{R}(v) &= \text{Sigmoid}(\text{MLP}_R(\mathbf{h}_v)) \quad &\text{(Reliability)} \\
\hat{M}(v) &= \text{Sigmoid}(\text{MLP}_M(\mathbf{h}_v)) \quad &\text{(Maintainability)} \\
\hat{I}^*(v) &= \text{Sigmoid}(\text{MLP}_C([\mathbf{h}_v \parallel \hat{R}(v) \parallel \hat{M}(v)])) \quad &\text{(Composite Blast Radius)}
\end{aligned}$$

*(The composite head explicitly consumes dimension predictions to capture non-linear cross-attribute interactions).*

### 4.3 Relation-Specific Edge Prediction Head

Evaluated directly on individual links via `TypedEdgeEncoder`:

$$Q_{\text{GNN}}(u, v) = \text{Sigmoid}\left(\text{MLP}_{\text{edge}}\left(\left[\mathbf{h}_u \parallel \mathbf{h}_v \parallel \mathbf{W}_r \mathbf{e}_{uv}\right]\right)\right)$$

---

## 5. Training Protocol & Multi-Task Loss Formulation

### 5.1 Multi-Task Loss Equation

The model is trained end-to-end using a balanced composite loss combining point regression, ranking objectives, pairwise margin separation, and rule-based consistency:

$$\mathcal{L} = \mathcal{L}_{\text{composite}} + 0.5 \cdot \mathcal{L}_{\text{dimension}} + 0.3 \cdot \mathcal{L}_{\text{rank}} + 0.1 \cdot \mathcal{L}_{\text{pairwise}} + 0.1 \cdot \mathcal{L}_{\text{consistency}} + 0.3 \cdot \mathcal{L}_{\text{edge}}$$

| Loss Component | Mathematical Formulation | Optimization Target |
|:---|:---|:---|
| **$\mathcal{L}_{\text{composite}}$** | $\text{MSE}(\hat{I}^*(v), I^*(v))$ | Accurate absolute composite prediction on simulated nodes |
| **$\mathcal{L}_{\text{dimension}}$** | $\sum_{d} \text{MSE}(\hat{d}(v), I_d^*(v)) \cdot \text{mask}_d$ | Multi-task alignment on measured sub-dimensions |
| **$\mathcal{L}_{\text{rank}}$** | $-\frac{1}{N} \sum_{v} \log P(\text{rank}(v))$ | ListMLE loss optimizing global Kendall $\tau$ and Spearman $\rho$ |
| **$\mathcal{L}_{\text{pairwise}}$** | $\sum_{i,j: y_i - y_j > m} \frac{\max(0, \; m - (\hat{s}_i - \hat{s}_j))}{|\text{pairs}|}$ | Margin ranking ($m=0.05$) enforcing strict ordinal separation |
| **$\mathcal{L}_{\text{consistency}}$**| $\text{MSE}(\hat{s}_{\text{unlabelled}}, y_{\text{RM}})$ | Semi-supervised regularization on unlabelled nodes toward $Q_{\text{RM}}$ |
| **$\mathcal{L}_{\text{edge}}$** | $\frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \text{MSE}(\hat{y}_{\text{edge}}^{(r)}, y_{\text{edge}}^{(r)})$ | Relation-balanced MSE on edge criticality predictions |

### 5.2 Training Hyperparameters & Optimisation

- **Data Splits**: 60% Train / 20% Validation / 20% Test (stratified per node type).
- **Optimizer**: AdamW ($\text{lr} = 3 \times 10^{-4}$, weight decay $= 10^{-4}$, gradient clipping norm $= 1.0$).
- **Learning Rate Schedule**: `CosineAnnealingWarmRestarts` ($T_0 = 50, T_{\text{mult}} = 2, \eta_{\text{min}} = 3 \times 10^{-6}$).
- **Early Stopping**: 30 epochs patience on combined metric ($0.6 \cdot \rho_{\text{val}} + 0.4 \cdot (1 - \mathcal{L}_{\text{val}} / \mathcal{L}_{\text{best}})$).
- **Multi-Seed Robustness**: Sweeps seeds $\{42, 123, 456, 789, 2024\}$ and saves the best model based on validation Spearman $\rho$.

---

## 6. Programmatic Python SDK & Use Cases

### 6.1 Pipeline Integration

The high-level `Pipeline` builder configures and executes Step 3 (Predict) on its own, or chained into Step 4 (Diagnose) — see [diagnosis.md §6.1](diagnosis.md#61-pipeline-integration) for the full chain including the Triage Bridge:

```python
import saag

pipeline = (
    saag.Pipeline.from_json("data/system.json", clear=True)
        .analyze(layer="system")
        .simulate(layer="system", mode="exhaustive")  # offline ground truth
        .predict()                                    # Step 3: Pathway B ranking (or RM fallback)
        .diagnose(k=10)                               # Step 4: Pathway A + Triage Bridge
        .validate()
        .prescribe()
        .run()
)

result = pipeline
```

`predict()` alone bundles Step 4 in by default (`diagnose=True` is `Client.predict()`'s own default, kept for backward compatibility) when called directly on `Client` rather than through `Pipeline` — `Pipeline.predict()` sets `diagnose=False` so Step 3 and Step 4 are genuinely independent stages when chained explicitly, as above.

### 6.2 Direct Use Case Execution (`saag.usecases`)

For fine-grained, decoupled execution without database dependencies:

```python
from saag.usecases import PredictiveUseCase

# Pathway B: Inductive Blast-Radius Forecasting
pred_uc = PredictiveUseCase(gnn_checkpoint_dir="output/gnn_checkpoints/best_model")
gnn_result = pred_uc.execute(structural_result=struct_result, graph=nx_graph)
```

See [diagnosis.md §6.2](diagnosis.md#62-direct-use-case-execution-saagusecases) for `DiagnosticUseCase` and `TriageUseCase`.

---

## 7. CLI Reference & Commands

### 1. Training GNN Checkpoints (Requires Step 5 Simulation Results)

```bash
# Standard training across 5 random seeds
PYTHONPATH=. python cli/train_graph.py --layer system --seeds 42 123 456 789 2024

# Multi-scenario inductive training
PYTHONPATH=. python cli/train_graph.py --layer system --multi-scenario

# Train node model only (disable edge head)
PYTHONPATH=. python cli/train_graph.py --layer system --no-edge-model
```

### 2. Running Criticality Predictions

```bash
# RM fallback (no checkpoint) — Step 4 (anti-patterns, explanation, Triage) bundled by default
PYTHONPATH=. python cli/predict_graph.py --layer system --triage-k 10

# Full GNN Inference + Step 4 bundled + Triage Bridge
PYTHONPATH=. python cli/predict_graph.py --layer system \
  --gnn-model output/gnn_checkpoints/best_model \
  --triage-k 10 \
  --output output/prediction.json

# Step 3 alone — GNN ranking only, no anti-patterns/explanation/triage/exit-code gating
PYTHONPATH=. python cli/predict_graph.py --layer system --gnn-model output/gnn_checkpoints/best_model --no-diagnose
```

For Step 4 (Diagnose) on its own — no GNN checkpoint required — see [diagnosis.md §5](diagnosis.md#5-cli-reference--commands).

---

## 8. Output Schema & Sample JSON

Running `python cli/predict_graph.py --gnn-model <ckpt> --triage-k 5 --output output/prediction.json` produces the following schema — `rm`, `antipatterns`, and `triage` are Step 4's output, bundled in by default; see [diagnosis.md §6](diagnosis.md#6-output-schema--sample-json) for that section's own schema when run via `saag-diagnose` instead:

```json
{
  "layers": {
    "system": {
      "total_components": 35,
      "rm": {
        "NavLib": {
          "overall": 0.54,
          "reliability": 0.63,
          "maintainability": 0.41,
          "fault_tolerance": 0.59,
          "availability": 0.58,
          "is_spof": true,
          "blast_radius": 12,
          "cascade_depth": 4
        }
      },
      "antipatterns": [
        {
          "entity_id": "NavLib",
          "entity_type": "Component",
          "name": "Single Point of Failure (SPOF)",
          "severity": "CRITICAL",
          "category": "Availability",
          "description": "NavLib is a directed cut vertex. Removing it partitions the dependency graph.",
          "recommendation": "Introduce redundancy: deploy backup instances or redundant routing paths.",
          "evidence": { "is_articulation_point": true, "availability_score": 0.58 }
        }
      ],
      "gnn": {
        "prediction_mode": "gnn_only",
        "node_scores": {
          "NavLib": {
            "component": "NavLib",
            "composite_score": 0.8432,
            "reliability_score": 0.8321,
            "maintainability_score": 0.6121,
            "criticality_level": "CRITICAL",
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
            "criticality_level": "MEDIUM"
          }
        ],
        "gnn_metrics": {
          "spearman_rho": 0.5871,
          "f1_score": 0.5052,
          "ndcg_10": 0.9211,
          "top_5_overlap": 0.60
        }
      },
      "triage": {
        "layer": "system",
        "k": 5,
        "ranking_source": "gnn",
        "population": 35,
        "entries": [
          {
            "component_id": "NavLib",
            "rank": 1,
            "ranking_score": 0.8432,
            "component_type": "Library",
            "pattern": "Single Point of Failure (SPOF)",
            "level": "CRITICAL",
            "priority_action": "Introduce redundancy: deploy backup instances or redundant routing paths.",
            "roles": ["DevOps", "Architect"],
            "elevated_dimensions": ["Availability", "Fault Tolerance"]
          }
        ]
      }
    }
  }
}
```

---

## 9. Known Limitations & Design Boundaries

| # | Boundary / Limitation | Methodological Context & Mitigation |
|:--|:---|:---|
| **L1** | **Heuristic Edge Labels** | Training uses $I^*(u) \times \text{bridge\_multiplier}$. Direct simulation removal ground truth ($I_{\text{edge}}$) is available in Step 5 for validation. |
| **L2** | **Backbone Edge Scatter-Mean** | Node convolutions average incoming edge vectors. Individual per-edge features are preserved un-averaged in `TypedEdgeEncoder`. |
| **L3** | **Unsupervised Infrastructure Nodes** | Loss is computed over `Application` and `Library` nodes; Broker, Topic, and Host nodes learn via graph message passing. |
| **L4** | **Transductive vs. Inductive Scope** | Single-graph training operates transductively. Inductive transfer is validated via `loso_evaluate.py` across distinct scenarios. |
| **L5** | **Feature Version Compatibility** | Checkpoints require `feature_version >= 3` (Broker: 19, Topic: 22, Node: 20 dimensions). Older checkpoints must be retrained. |

---

## 10. What Comes Next

- **For root-cause attribution**: Proceed to **[Step 4: Diagnose](diagnosis.md)** for the deterministic RM composite, 19-pattern anti-pattern audit, and the Triage Bridge scoping this stage's Top-K ranking to a root-cause profile.
- **For Inference validation**: **[Step 6: Validate](validation.md)** statistically evaluates this stage's predicted scores against discrete-event failure injection, and **[Step 7: Prescribe](prescription.md)** compiles verified refactoring blueprints from Step 4's output.
- **For Training**: Execute **[Step 5: Simulate](failure-simulation.md)** first to generate ground-truth impact labels $I^*(v)$ before running `cli/train_graph.py`.

---

← [Step 2: Analyze](structural-analysis.md) | → [Step 4: Diagnose](diagnosis.md)
