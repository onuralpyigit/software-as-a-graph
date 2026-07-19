# Software Requirements Specification (SRS)

## Software-as-a-Graph (saag)

### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 3.0** · **June 2026**  
*Istanbul Technical University, Computer Engineering Department*  
*Conforming to ISO/IEC/IEEE 29148:2018 & ISO/IEC/IEEE 12207:2026*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [Functional Requirements](#3-functional-requirements)
   - 3.1 [Synthetic Graph Generation (Offline Input Preparation)](#31-synthetic-graph-generation-offline-input-preparation)
   - 3.2 [Graph Model Construction (Step 1)](#32-graph-model-construction-step-1)
   - 3.3 [Structural Analysis (Step 2)](#33-structural-analysis-step-2)
   - 3.4 [Unified Prediction Step: Rule-Based (RMAV) + ML (GNN) (Step 3)](#34-unified-prediction-step-rule-based-rmav--ml-gnn-step-3)
   - 3.5 [Failure Simulation (Step 4)](#35-failure-simulation-step-4)
   - 3.6 [Statistical Validation (Step 5)](#36-statistical-validation-step-5)
   - 3.7 [Visualization (Step 6)](#37-visualization-step-6)
   - 3.8 [Multi-Layer Analysis](#38-multi-layer-analysis)
   - 3.9 [System Interface Requirements](#39-system-interface-requirements)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [Data Requirements](#5-data-requirements)
6. [Validation Targets and Achieved Results](#6-validation-targets-and-achieved-results)
7. [ISO/IEC/IEEE 12207:2026 Process Mapping Matrix](#7-isoiecieee-122072026-process-mapping-matrix)
8. [Appendix A: Prediction Formula Reference](#appendix-a-prediction-formula-reference)
9. [Appendix B: Glossary](#appendix-b-glossary)

---

## 1. Introduction

### 1.1 Purpose
This document specifies the software requirements for the **Software-as-a-Graph (saag)** framework—a system that predicts which components in a distributed publish-subscribe environment are critical (i.e., whose failure would cause the greatest cascading impact) using only the system's architectural structure. 

This specification is aligned with **ISO/IEC/IEEE 29148:2018** for requirements engineering and traces directly to **ISO/IEC/IEEE 12207:2026** software life cycle processes. The target audience includes software architects, reliability engineers, security engineers, DevOps professionals, and researchers.

### 1.2 Scope
Software-as-a-Graph transforms a distributed publish-subscribe system's topology description into a weighted heterogeneous directed graph, evaluates its structural characteristics, and applies both rule-based (RMAV) and machine-learning (Heterogeneous Graph Transformer - GNN) approaches to forecast component and link criticality before deployment, without requiring runtime telemetry.

The framework implements a 6-step core analytical pipeline, preceded by an offline input preparation stage:

| Step | Function | Output |
|------|----------|--------|
| **Offline Prep: Generate** | Produce synthetic pub-sub topologies for evaluation | Topology JSON |
| **1. Model** | Load topology JSON into Neo4j; derive logical dependencies | Heterogeneous Graph $G(V, E)$ |
| **2. Analyze** | Compute 18+ topological metrics per component | Metric vectors $\mathbf{M}(v)$ |
| **3. Predict** | Forecast criticality via rule-based (RMAV) and learning-based (GNN) models | Node $Q(v)$ & Edge $Q(e)$ scores |
| **4. Simulate** | Inject cascade failure scenarios to generate labels | Ground-truth labels $\mathbf{I}(v)$ |
| **5. Validate** | Compare predictions $Q$ against simulation results $\mathbf{I}$ | Spearman $\rho$, F1-score, NDCG |
| **6. Visualize**| Generate dashboard reports and interactive viewers | HTML Reports / SMART Web App |

The system is delivered through three interfaces:
1. **Core SDK ([saag])**: Hexagonal-architecture Python package.
2. **CLI Utility Tools ([cli])**: Scripts for automation and batch pipeline execution.
3. **SMART Web Application (smart)**: FastAPI backend + Next.js interactive frontend.

### 1.3 References

| Reference | Version / DOI | Description |
|-----------|---------------|-------------|
| **ISO/IEC/IEEE 12207:2026** | Second Edition | Systems and software engineering — Software life cycle processes |
| **ISO/IEC/IEEE 29148:2018** | First Edition | Systems and software engineering — Life cycle processes — Requirements engineering |
| **IEEE RASSE 2025** | doi:10.1109/RASSE64831.2025.11315354 | Published methodology paper |
| **PyTorch Geometric** | $\ge 2.3$ | Heterogeneous GNN framework library |
| **Neo4j DBMS** | 5.x | Primary graph database and GDS engine |
| **NetworkX** | $\ge 3.0$ | Topological analysis library |
| **FastAPI** | $\ge 0.100$ | REST API backend framework |
| **Next.js** | 16.x | SMART web interface framework |

### 1.4 Document Conventions
- **Shall** denotes a mandatory requirement.
- **Should** denotes a recommended behavior.
- **May** denotes an optional capability.
- Identifiers follow the pattern **REQ-\<STAGE\>-\<NN\>** (e.g., `REQ-GG-01` for Graph Generation, `REQ-GNN-01` for GNN Prediction).

### 1.5 Change History
- **v2.2 (Feb 2026):** Added FastAPI backend, SMART Next.js web application interfaces, and increased target accuracy bounds.
- **v2.3 (Mar 2026):** Refactored backend architecture to follow presenter patterns and updated quality formulations to align with expert weight shifts.
- **v3.0 (Jun 2026 - Current):** Full alignment with **ISO/IEC/IEEE 12207:2026** and **ISO/IEC/IEEE 29148:2018**. Expanded GNN details (Heterogeneous Graph Transformers, bidirectional propagation, custom edge projections, multi-task losses, robust normalization) and added Step 0 (Synthetic Graph Generation) functional requirements. Added ML Operational NFRs and the process mapping matrix.

---

## 2. Overall Description

### 2.1 Product Perspective
Software-as-a-Graph is a standalone, pre-deployment static analysis tool. It reads external system topologies and assesses component failure bounds. It operates as a microservice stack (Docker Compose) or as a local Python script suite.

### 2.2 System Interfaces
- **Graph Database**: Neo4j 5.x reachable via Bolt protocol on port 7687.
- **REST API**: FastAPI server listening on port 8000.
- **Interactive Web App**: Next.js App Router (smart) serving on port 7000.
- **Topology Input**: JSON or GraphML format files describing components and QoS properties.

### 2.3 User Characteristics
- **Software Architect**: Designs topologies and assesses reliability risks.
- **Reliability/DevOps Engineer**: Focuses on single points of failure (SPOF) and redundancy mapping.
- **Security Engineer**: Identifies high-vulnerability attack vectors.
- **Researcher/Data Scientist**: Evaluates and trains GNN models on synthetic datasets.

### 2.4 Product Functions
1. **Synthetic Generation**: Automatically build pub-sub topology benchmarks.
2. **Graph Import**: Ingest topology specifications and project layers in Neo4j.
3. **Metric Extraction**: Calculate structural, QoS, and code-level characteristics.
4. **Predictive Analysis**: Compute closed-form RMAV dimensions and run PyTorch GNN inference.
5. **Cascade Simulation**: Propagate failures across four operational contexts.
6. **Performance Validation**: Check predictions against simulated benchmarks using Spearman $\rho$ gates.

### 2.5 Constraints
- The system performs static analysis; accuracy is bounded by input topology completeness.
- Neural network features rely on correct schema mappings.
- The web interface requires Node.js 20+ and Next.js 16.x compatibility.

### 2.6 Dependencies
- **Python $\ge 3.9$**
- **PyTorch $\ge 2.0$ & PyTorch Geometric $\ge 2.3$** (for HGTConv and HeteroData operations)
- **FastAPI / Uvicorn** (REST API)
- **Node.js $\ge 20$ / React 19 / Next.js 16** (smart)
- **Neo4j 5.x with GDS Plugin** (Persistence and centrality algorithms)

---

## 3. Functional Requirements

### 3.1 Synthetic Graph Generation (Offline Input Preparation)
The system must generate synthetically parameterized topologies representing diverse publish-subscribe environments to enable benchmarking and training.

| ID | Requirement |
|----|-------------|
| **REQ-GG-01** | The system shall generate topologies matching five node types: Application, Broker, Topic, Node (infra), and Library. |
| **REQ-GG-02** | The system shall support six scale presets: TINY, SMALL, MEDIUM, LARGE, HUGE, and ENTERPRISE, restricting component counts to defined bounds. |
| **REQ-GG-03** | The system shall inject domain-specific QoS policies (Reliability, Durability, Priority, and Message Size) based on configuration tables for 8 default scenarios. |
| **REQ-GG-04** | The system shall support seed parameters to guarantee deterministic node and edge distribution across multiple generation cycles. |
| **REQ-GG-05** | The system shall calculate code-quality attributes (LOC, cyclomatic complexity, instability, LCOM) for Application and Library nodes based on statistical distributions. |
| **REQ-GG-06** | The system shall output the generated topology in a single, self-contained JSON schema. |

### 3.2 Graph Model Construction (Step 1)
The system must parse topology representations, populate the graph database, and derive dependency mappings.

| ID | Requirement |
|----|-------------|
| **REQ-GM-01** | The system shall ingest topology specifications in JSON and GraphML formats. |
| **REQ-GM-02** | The system shall perform a multi-phase import into Neo4j: importing vertices, creating structural edges (RUNS_ON, ROUTES, PUBLISHES_TO, SUBSCRIBES_TO, CONNECTS_TO, USES), and mapping QoS properties. |
| **REQ-GM-03** | The system shall derive logical `DEPENDS_ON` edges between components based on operational data-flow rules (app_to_app, app_to_broker, node_to_node, node_to_broker). |
| **REQ-GM-04** | The system shall assign weights to `DEPENDS_ON` edges in the range $[0, 2.0]$, derived from QoS settings and message size bounds (see Appendix A.1). |
| **REQ-GM-05** | The system shall isolate subgraphs for layer-specific analysis, supporting: Application (`app`), Infrastructure (`infra`), Middleware (`mw`), and System (`system`) layers. |

### 3.3 Structural Analysis (Step 2)
The system must extract topological metrics from the projected subgraphs.

| ID | Requirement |
|----|-------------|
| **REQ-SA-01** | The system shall calculate PageRank (PR), Reverse PageRank (RPR), Betweenness (BT), Closeness (CL), Eigenvector (EV), and Degree (In/Out) centralities. |
| **REQ-SA-02** | The system shall identify articulation points (AP), calculate directed articulation point scores (AP_c_directed), and identify structural bridge edges. |
| **REQ-SA-03** | The system shall aggregate QoS weights ($w$, $w_{in}$, $w_{out}$) and compute connectivity degradation indices (CDI). |
| **REQ-SA-04** | The system shall normalize all computed topological and QoS metrics to the interval $[0, 1]$ prior to downstream utilization. |
| **REQ-SA-05** | The system shall compute graph-level statistics (density, clustering coefficient, bridge ratio, and node/edge ratios) for diagnostic reporting. |

### 3.4 Unified Prediction Step: Rule-Based (RMAV) + ML (GNN) (Step 3)
The legacy "Quality Scoring" mechanism (formerly part of Step 2) has been removed and replaced by a single, unified Prediction Step. The system must forecast node and edge criticality using rule-based metrics (always computed) and trained GNN models (blended in when available), and derive anti-pattern reports and explanations from the result.

#### 3.4.1 Rule-Based Quality Scoring (RMAV)
| ID | Requirement |
|----|-------------|
| **REQ-QS-01** | The system shall compute individual dimension scores for Reliability $R(v)$, Maintainability $M(v)$, Availability $A(v)$, and Vulnerability $V(v)$ using the closed-form expressions in Appendix A.2 to A.5. |
| **REQ-QS-02** | The system shall compute the composite score $Q_{RMAV}(v)$ using weighted dimensions derived from the Analytic Hierarchy Process (AHP). |
| **REQ-QS-03** | The system shall validate AHP matrix consistency, ensuring the Consistency Ratio ($CR$) is less than $0.10$ before applying derived weights. |
| **REQ-QS-04** | The system shall isolate and report architectural anti-patterns: Single Points of Failure (SPOF), failure hubs, and high-vulnerability clusters. |

#### 3.4.2 GNN-Based Prediction (Inductive Forecasting)
| ID | Requirement |
|----|-------------|
| **REQ-GNN-01** | The system shall convert NetworkX topology representations into PyTorch Geometric `HeteroData` representations with type-partitioned nodes and edges. |
| **REQ-GNN-02** | The system shall construct node feature tensors consisting of an 18-dimensional base topological vector augmented by type-specific properties (Application/Library: 23-dim, Broker: 19-dim, Topic: 22-dim, Node: 20-dim). |
| **REQ-GNN-03** | The system shall construct 16-dimensional edge feature tensors containing QoS metrics, path counts, and edge-type one-hot encodings (see Appendix A.7). |
| **REQ-GNN-04** | The system shall implement a 3-layer **EdgeAwareHGTConv (HGT)** backbone (`NodeCriticalityGNN`) with relation-specific Key/Query/Value projection matrices to learn type-specific attention weights. |
| **REQ-GNN-05** | The system shall inject 16-dimensional edge attributes directly into the Key and Value representation of each individual edge before multi-head attention, via `EdgeAwareHGTConv`'s edge projection layers (`k_edge_proj`, `v_edge_proj`). The legacy `EdgeFeatureEncoder` (scatter-mean pre-aggregation) is retained for backward checkpoint compatibility but is not used in the active forward pass. |
| **REQ-GNN-06** | The system shall support a bidirectional pass option (`use_bidirectional=True`) to capture upstream and downstream architectural signals during graph convolution. |
| **REQ-GNN-07** | The system shall deploy multi-task prediction heads (MLPs with Sigmoid activations) to predict dimension scores ($\hat{R}$, $\hat{M}$, $\hat{A}$, $\hat{V}$) and a composite score $\hat{I}^*$ concurrently. |
| **REQ-GNN-08** | The system shall feed the outputs of the four dimension heads directly into the composite head alongside the node representation to learn non-linear dimension interactions. |
| **REQ-GNN-09** | The system shall calculate direct edge-level criticality rankings using a `TypedEdgeEncoder` which projects edge features and fuses them with source and destination node embeddings. |

#### 3.4.3 Criticality Classification
| ID | Requirement |
|----|-------------|
| **REQ-GNN-CLS-01** | The system shall classify components into five criticality levels (CRITICAL, HIGH, MEDIUM, LOW, MINIMAL) using box-plot statistical classification ($Q_3 + k \cdot IQR$) with adaptive thresholds rather than static cuts. |
| **REQ-GNN-CLS-02** | The system shall log fallback alerts and default to RMAV scoring if the specified GNN model checkpoint is missing or incompatible with the target layer. |

#### 3.4.4 GNN Training and Optimization
| ID | Requirement |
|----|-------------|
| **REQ-GNN-TR-01** | The system shall utilize AdamW optimization, Cosine Annealing learning rate scheduling with restarts, and gradient norm clipping. |
| **REQ-GNN-TR-02** | The system shall partition nodes using a transductive split (60% Train, 20% Val, 20% Test) per node type, and support fully inductive scenario splits. |
| **REQ-GNN-TR-03** | The system shall optimize GNN parameters using a composite loss function comprising: MSE composite loss, MSE dimension loss, ListMLE ranking loss, pairwise margin loss, and RMAV consistency regularization on unlabeled nodes (see Appendix A.8). |
| **REQ-GNN-TR-04** | The system shall perform robust label normalization in-place on target simulation labels using IQR-scaled sigmoidal bounds to mitigate outlier influence. |
| **REQ-GNN-TR-05** | The system shall support multi-seed training loops (default seeds: 42, 123, 456, 789, 2024), logging validation Spearman $\rho$ per seed, and restoring the best weights before checkpoint serialization. |

### 3.5 Failure Simulation (Step 4)
The system must run cascade simulations to establish target criticality labels.

| ID | Requirement |
|----|-------------|
| **REQ-FS-01** | The system shall support exhaustive failure simulation by sequentially failing each component and evaluating downstream disruptions. |
| **REQ-FS-02** | The system shall support Monte Carlo simulation for large systems to limit execution durations. |
| **REQ-FS-03** | The system shall execute four distinct simulator modes: Reliability (cascade propagation), Maintainability (change impact reach), Availability (network connectivity loss), and Vulnerability (compromise propagation). |
| **REQ-FS-04** | The system shall propagate cascades using four semantic rules: PHYSICAL (RUNS_ON), LOGICAL (Broker routes), NETWORK (CONNECTS_TO), and LIBRARY (USES). |
| **REQ-FS-05** | The system shall output a composite impact score $I(v)$ representing the overall cascading footprint of each component. |

### 3.6 Statistical Validation (Step 5)
The system must validate prediction accuracy against simulation ground truth.

| ID | Requirement |
|----|-------------|
| **REQ-VL-01** | The system shall calculate the Spearman rank correlation coefficient ($\rho$) and its associated p-value between prediction and simulation rankings. |
| **REQ-VL-02** | The system shall calculate F1-Score, Precision, Recall, and Cohen's Kappa ($\kappa$) for critical component classifications. |
| **REQ-VL-03** | The system shall compute ranking-specific validation metrics, including NDCG@K and Top-5/Top-10 overlap percentages. |
| **REQ-VL-04** | The system shall compute Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) between normalized prediction and simulation scores. |
| **REQ-VL-05** | The system shall evaluate metrics against three tiers of validation gates (Primary, Secondary, Specialist) and block downstream deployments upon gate failure. |

### 3.7 Visualization (Step 6)
The system must generate visual reports for end-user analysis.

| ID | Requirement |
|----|-------------|
| **REQ-VZ-01** | The system shall output a self-contained HTML dashboard report containing all extracted metrics, criticality scores, and validation results. |
| **REQ-VZ-02** | The system shall display interactive network graphs (vis.js) with dynamic node coloring based on criticality. |
| **REQ-VZ-03** | The SMART web application (smart) shall provide a Dashboard view, an interactive Graph Explorer (2D/3D force-directed layouts), an Analysis interface, and a Failure Simulator panel. |
| **REQ-VZ-04** | The Graph Explorer side panel shall display detailed node metrics, direct dependency links, and active anti-pattern flags upon component selection. |

### 3.8 Multi-Layer Analysis
| ID | Requirement |
|----|-------------|
| **REQ-ML-01** | The system shall support graph projections and pipeline executions targeted at specific layers: Application (`app`), Infrastructure (`infra`), Middleware (`mw`), and System (`system`). |

### 3.9 System Interface Requirements

#### 3.9.1 REST API Endpoints
All API routers shall serve versioned JSON payloads under the `/api/v1/` prefix.

| ID | Requirement | Endpoint | Method |
|----|-------------|----------|--------|
| **REQ-API-01** | Health Check | `/health` | GET |
| **REQ-API-02** | Graph Summary | `/api/v1/graph/summary` | GET |
| **REQ-API-03** | Topology Import | `/api/v1/graph/import` | POST |
| **REQ-API-04** | Component Search | `/api/v1/graph/search-nodes` | GET |
| **REQ-API-05** | Trigger Analysis | `/api/v1/analysis/layer/{layer}` | POST |
| **REQ-API-06** | Predict Criticality | `/api/v1/prediction/predict` | POST |
| **REQ-API-07** | Train GNN model | `/api/v1/prediction/train` | POST |
| **REQ-API-08** | Run Failure Simulation | `/api/v1/simulation/failure` | POST |
| **REQ-API-09** | Execute Full Pipeline | `/api/v1/validation/run-pipeline` | POST |
| **REQ-API-10** | Retrieve Layers | `/api/v1/validation/layers` | GET |

#### 3.9.2 Command-Line Interface (CLI) Scripts
The system shall expose command-line entry points registered via package configuration.

| ID | Requirement | Entrypoint | Script / Purpose |
|----|-------------|------------|------------------|
| **REQ-CLI-01** | Generate Topology | `saag-generate` | `cli/generate_graph.py` |
| **REQ-CLI-02** | Import Graph | `saag-import` | `cli/import_graph.py` |
| **REQ-CLI-03** | Analyze Graph | `saag-analyze` | `cli/analyze_graph.py` |
| **REQ-CLI-04** | GNN Training | `saag-train` | `cli/train_graph.py` |
| **REQ-CLI-05** | Predict Criticality | `saag-predict` | `cli/predict_graph.py` |
| **REQ-CLI-06** | Fail Components | `saag-simulate` | `cli/simulate_graph.py` |
| **REQ-CLI-07** | Validate Results | `saag-validate` | `cli/validate_graph.py` |
| **REQ-CLI-08** | Visualise Data | `saag-visualize` | `cli/visualize_graph.py` |
| **REQ-CLI-09** | Orchestrate Pipeline | `saag` | `cli/run.py` |

---

## 4. Non-Functional Requirements

### 4.1 Performance
- **REQ-PERF-01**: Graph analysis shall complete in $\le 1.0$ second for systems with under 30 nodes.
- **REQ-PERF-02**: Graph analysis shall complete in $\le 5.0$ seconds for systems with under 100 nodes.
- **REQ-PERF-03**: Graph analysis shall complete in $\le 20.0$ seconds for systems with under 600 nodes.
- **REQ-PERF-04**: REST API analysis requests shall respond in $\le 30.0$ seconds for up to 1,000 components.

### 4.2 Accuracy
- **REQ-ACC-01**: Spearman $\rho$ correlation coefficient shall achieve $\ge 0.80$ at the Application Layer.
- **REQ-ACC-02**: Critical classification F1-Score shall achieve $\ge 0.90$ at the Application Layer.
- **REQ-ACC-03**: Precision and Recall metrics shall each achieve $\ge 0.80$ at the Application Layer.
- **REQ-ACC-04**: Top-5 ranked overlap shall achieve $\ge 60\%$ at the Application Layer.

### 4.3 Scalability
- **REQ-SCAL-01**: The system shall process topologies containing up to 1,000 nodes.
- **REQ-SCAL-02**: The system shall process topologies containing up to 10,000 edges.

### 4.4 Reliability
- **REQ-REL-01**: Input schema errors must be intercepted, reporting diagnostic validation logs.
- **REQ-REL-02**: Database timeouts must trigger reconnection retries before failing.

### 4.5 Portability
- **REQ-PORT-01**: The Python SDK and CLI scripts shall run natively on Linux, macOS, and Windows.
- **REQ-PORT-02**: The GNN model inference code shall run on both CUDA-enabled GPUs and CPU-only fallbacks.

### 4.6 Maintainability
- **REQ-MAINT-01**: Python code shall conform to PEP 8 standards.
- **REQ-MAINT-02**: The SDK module unit test coverage shall be $\ge 80\%$.

### 4.7 Security
- **REQ-SEC-01**: Database authentication credentials shall not be hardcoded; they must be resolved from environment variables.
- **REQ-SEC-02**: The system shall support TLS/SSL encryption for database connections (bolt+s protocol).

### 4.8 Logging and Observability
- **REQ-LOG-01**: The CLI and API components shall log pipeline duration metrics and statistical thresholds.

### 4.9 Machine Learning Operational Constraints
- **REQ-MLOPS-01**: GNN model checkpoints shall be serializable in a single directory containing the weight parameters and layer metadata.
- **REQ-MLOPS-02**: Multi-seed evaluations must enforce weight isolation to prevent gradient leakage.
- **REQ-MLOPS-03**: Training runs must implement early stopping with a combined metric (loss and correlation) to prevent overfitting.

### 4.10 Hardware Requirements
- **REQ-HW-01**: Small configurations ($<100$ nodes): $\ge 4$ GB RAM.
- **REQ-HW-02**: Medium configurations ($100-500$ nodes): $\ge 8$ GB RAM.
- **REQ-HW-03**: Enterprise configurations ($>500$ nodes): $\ge 16$ GB RAM.

---

## 5. Data Requirements

### 5.1 Graph Data Model

#### 5.1.1 Vertex Node Labels
- `Application`: A software task that publishes or subscribes to message channels.
- `Broker`: Message routing middleware (e.g., DDS Daemon, ROS Master, Kafka Broker).
- `Topic`: Named message path with QoS configurations.
- `Node`: Hardware or virtual system host.
- `Library`: Code library dependency.

#### 5.1.2 Edge Relationship Types
- `PUBLISHES_TO`: `Application` $\rightarrow$ `Topic`
- `SUBSCRIBES_TO`: `Application` $\rightarrow$ `Topic`
- `ROUTES`: `Broker` $\rightarrow$ `Topic`
- `RUNS_ON`: `Application` / `Broker` $\rightarrow$ `Node`
- `CONNECTS_TO`: `Node` $\rightarrow$ `Node`
- `USES`: `Application` / `Library` $\rightarrow$ `Library`
- `DEPENDS_ON`: General derived dependency link.

---

## 6. Validation Targets and Achieved Results

The system evaluates all forecast models against simulated failure footprints. The targets and actual performance across validated scenarios are as follows:

| Metric | Target (App Layer) | Achieved (App Layer) | Achieved (Infra Layer) |
|--------|--------------------|-----------------------|------------------------|
| Spearman $\rho$ | $\ge 0.80$ | **0.876** ✓ | 0.54 |
| F1-Score | $\ge 0.90$ | **0.943** ✓ | 0.68 |
| Precision | $\ge 0.80$ | **0.94** ✓ | 0.71 |
| Recall | $\ge 0.80$ | **0.95** ✓ | 0.65 |
| Top-5 Overlap | $\ge 60\%$ | **80%** ✓ | 40% |

---

## 7. ISO/IEC/IEEE 12207:2026 Process Mapping Matrix

This matrix maps the requirements defined in Section 3 to the standard software life cycle processes defined in **ISO/IEC/IEEE 12207:2026**.

| Requirement Group | ISO/IEC/IEEE 12207:2026 Process | Primary Activities / Verification Strategy |
|-------------------|---------------------------------|-------------------------------------------|
| **REQ-GG-01 to 06** | System/Software Requirements Definition | Synthetic generation verification, seed consistency testing. |
| **REQ-GM-01 to 05** | Software Implementation / Design | Database schema assertions, entity loading checks. |
| **REQ-SA-01 to 05** | Software Design Definition | Centrality verification against NetworkX baseline calculations. |
| **REQ-QS-01 to 04** | Software Design Definition | Closed-form metric verification, AHP consistency checks. |
| **REQ-GNN-01 to 09**| Software Implementation / Design | Node/edge dimension checks, HGT layer shapes verification. |
| **REQ-GNN-CLS-01 to 02** | Software Integration / Classification | Criticality tier classification checks, fallback mode tests. |
| **REQ-GNN-TR-01 to 05**| Software Implementation / Training | Optimization checks, loss value decreases validation. |
| **REQ-FS-01 to 05** | Software Verification | Cascade propagation rules checking against test scenarios. |
| **REQ-VL-01 to 05** | Software Validation | Statistical function testing (scipy comparison assertions). |
| **REQ-VZ-01 to 04** | Software Transition | HTML formatting validations, Next.js UI integration tests. |
| **REQ-API-01 to 10**| Software Integration | REST API routing validation, JSON schema assertions. |
| **REQ-CLI-01 to 09**| Software Integration | Command parser checks, script execution testing. |
| **NFR-PERF / NFR-ACC**| Software Validation | Performance profiling and accuracy gate runs. |
| **NFR-MLOPS-01 to 03**| Software Verification / Maintenance | Checkpoint loading validation, seed weight isolation tests. |

---

## Appendix A: Prediction Formula Reference

### A.1 Edge Weight Formula
For a dependency edge $e$, the weight $w(e)$ is computed as:
$$w(e) = w_{\text{reliability}} + w_{\text{durability}} + w_{\text{priority}} + w_{\text{size}}$$
Where:
- $w_{\text{reliability}} = 0.30$ if Reliable, $0.00$ otherwise.
- $w_{\text{durability}} = 0.40$ if Persistent, $0.24$ if Transient, $0.20$ if Transient-Local, $0.00$ if Volatile.
- $w_{\text{priority}} = 0.30$ if Urgent, $0.20$ if High, $0.10$ if Medium, $0.00$ if Low.
- $w_{\text{size}} = \min(\log_2(1 + \text{size\_bytes}/1024)/50, 0.20)$.

### A.2 Reliability Score
$$R(v) = w_1 \times RPR(v) + w_2 \times DG_{in}(v) + w_3 \times CDPot_{enh}(v)$$
*(Weights: $w_1 = 0.45, w_2 = 0.30, w_3 = 0.25$)*

### A.3 Maintainability Score
$$M(v) = w_1 \times BT(v) + w_2 \times w_{out}(v) + w_3 \times CQP(v) + w_4 \times CouplingRisk_{enh}(v) + w_5 \times (1 - CC(v))$$
*(Weights: $w_1 = 0.35, w_2 = 0.30, w_3 = 0.15, w_4 = 0.12, w_5 = 0.08$)*

### A.4 Availability Score
$$A(v) = w_1 \times AP_{c\_directed}(v) + w_2 \times QSPOF(v) + w_3 \times BR(v) + w_4 \times CDI(v) + w_5 \times w(v)$$
*(Weights: $w_1 = 0.35, w_2 = 0.25, w_3 = 0.25, w_4 = 0.10, w_5 = 0.05$)*

### A.5 Vulnerability Score
$$V(v) = w_1 \times REV(v) + w_2 \times RCL(v) + w_3 \times QADS(v)$$
*(Weights: $w_1 = 0.40, w_2 = 0.35, w_3 = 0.25$)*

### A.6 Composite Quality Score (RMAV Baseline)
$$Q_{RMAV}(v) = \alpha \cdot R(v) + \beta \cdot M(v) + \gamma \cdot A(v) + \delta \cdot V(v)$$
*(Default system-layer weights: $\alpha = \beta = \gamma = \delta = 0.25$)*

### A.7 GNN Edge Features (16 Dimensions)
1. QoS weight $w(e)$ (index 0)
2. Normalized path count $\log_2(1 + \text{path\_count})/\log_2(17)$ (index 1)
3. Edge-type one-hot vector (indices 2–8: `PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`, `DEPENDS_ON`)
4. Reliability configuration flag ($1.0$ Reliable, $0.0$ otherwise) (index 9)
5. Durability ordinal configuration (indices 10–15 QoS decomposition attributes)

### A.8 GNN Training Multi-Task Loss Formulation
$$\mathcal{L} = \mathcal{L}_{\text{composite}} + 0.5 \times \mathcal{L}_{\text{dimension}} + 0.3 \times \mathcal{L}_{\text{rank}} + 0.1 \times \mathcal{L}_{\text{pairwise}} + 0.1 \times \mathcal{L}_{\text{consistency}}$$
- $\mathcal{L}_{\text{composite}}$: MSE of composite predictions against simulation labels.
- $\mathcal{L}_{\text{dimension}}$: MSE of individual dimension predictions.
- $\mathcal{L}_{\text{rank}}$: ListMLE rank-loss optimized across node lists.
- $\mathcal{L}_{\text{pairwise}}$: Margin loss ($m=0.05$) enforcing relative node rank constraints.
- $\mathcal{L}_{\text{consistency}}$: Regularization term checking predicting bounds against RMAV baselines on unlabeled nodes.

---

## Appendix B: Glossary

- **AHP**: Analytic Hierarchy Process—a structured weighting methodology.
- **AP**: Articulation Point—a node whose removal disconnects the graph.
- **CDI**: Connectivity Degradation Index—a metric indicating potential isolation footprint.
- **CQP**: Code Quality Penalty—metric incorporating cyclomatic complexity, LOC, and instability.
- **GDS**: Graph Data Science—Neo4j's algorithm framework.
- **GNN**: Graph Neural Network—neural network operating directly on graphs.
- **HeteroData**: PyTorch Geometric's data object containing heterogeneous nodes and edges.
- **HGT / HGTConv / EdgeAwareHGTConv**: Heterogeneous Graph Transformer—GNN layers using type-specific attention. `EdgeAwareHGTConv` is the project’s custom extension that projects edge features directly into relation-specific Key and Value spaces before message passing, avoiding information smoothing.
- **ListMLE**: Listwise maximum likelihood loss for ranking data.
- **RMAV**: Reliability, Maintainability, Availability, and Vulnerability.
- **SPOF**: Single Point of Failure—an active articulation point component.