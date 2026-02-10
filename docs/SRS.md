# Software Requirements Specification

## Software-as-a-Graph

### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 2.0** · **February 2026**

Istanbul Technical University, Computer Engineering Department

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [Functional Requirements](#3-functional-requirements)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [Data Requirements](#5-data-requirements)
6. [Validation Targets and Achieved Results](#6-validation-targets-and-achieved-results)
7. [Appendix A: Quality Formula Reference](#appendix-a-quality-formula-reference)
8. [Appendix B: Glossary](#appendix-b-glossary)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for the Software-as-a-Graph framework — a tool that predicts which components in a distributed publish-subscribe system are most critical (i.e., whose failure would cause the greatest impact) using only the system's architectural structure.

The intended audience includes software architects, reliability engineers, security engineers, DevOps engineers, and researchers working with distributed systems.

### 1.2 Scope

Software-as-a-Graph transforms a distributed pub-sub system's topology into a weighted directed graph and applies topological analysis to predict component criticality before deployment, without runtime monitoring.

The framework implements a six-step methodology:

| Step | Function | Output |
|------|----------|--------|
| 1. Graph Model Construction | Convert system topology into a weighted directed graph | G(V, E, w) |
| 2. Structural Analysis | Compute centrality and resilience metrics per component | Metric vectors M(v) |
| 3. Quality Scoring | Map metrics to RMAV quality dimensions using AHP weights | Quality scores Q(v) |
| 4. Failure Simulation | Inject faults and measure cascading impact | Impact scores I(v) |
| 5. Validation | Statistically compare Q(v) against I(v) | Spearman ρ, F1, etc. |
| 6. Visualization | Generate interactive dashboards | HTML dashboard |

### 1.3 References

| Reference | Description |
|-----------|-------------|
| IEEE 830-1998 | IEEE Recommended Practice for Software Requirements Specifications |
| IEEE RASSE 2025 | Published methodology paper (doi: 10.1109/RASSE64831.2025.11315354) |
| Neo4j Documentation | https://neo4j.com/docs/ |
| NetworkX Documentation | https://networkx.org/documentation/ |
| Saaty, T.L. (1980) | *The Analytic Hierarchy Process*, McGraw-Hill |

---

## 2. Overall Description

### 2.1 Product Perspective

Software-as-a-Graph is a standalone pre-deployment analysis framework. It integrates with existing distributed architectures by consuming their topology descriptions and producing criticality assessments. It does not perform runtime monitoring.

### 2.2 System Interfaces

| Interface | Technology | Purpose |
|-----------|------------|---------|
| Graph Database | Neo4j 5.x (Bolt protocol) | Primary graph storage and GDS algorithms |
| Topology Input | JSON, GraphML | System architecture import |
| Results Export | JSON, CSV, GraphML | Analysis results for external tools |
| Dashboard Output | HTML (vis.js, Chart.js) | Interactive visualization |
| CLI | Python argparse | Pipeline orchestration |

### 2.3 User Characteristics

| User Role | Primary Interest |
|-----------|-----------------|
| Software Architect | Evaluating system designs for reliability risks |
| Reliability Engineer | Identifying critical components for redundancy planning |
| DevOps Engineer | Prioritizing monitoring and alerting configuration |
| Security Engineer | Identifying high-value attack targets |
| Researcher | Validating graph-based prediction methodologies |

### 2.4 Constraints

The framework requires Neo4j 5.x and Python 3.9+. It performs static analysis only — accuracy depends on the completeness of the input topology specification. Memory requirements grow with system scale (see [Section 4.1](#41-performance)).

### 2.5 Assumptions

The system topology is accurately specified in the input format. QoS settings are available for topics. Neo4j is accessible and properly configured.

---

## 3. Functional Requirements

### 3.1 Graph Model Construction (Step 1)

| ID | Requirement |
|----|-------------|
| REQ-GM-01 | The system shall accept system topology in JSON format. |
| REQ-GM-02 | The system shall create vertices for five component types: Node, Broker, Topic, Application, and Library. |
| REQ-GM-03 | The system shall create six structural edge types: RUNS_ON, ROUTES, PUBLISHES_TO, SUBSCRIBES_TO, CONNECTS_TO, and USES. |
| REQ-GM-04 | The system shall derive DEPENDS_ON edges automatically from structural relationships using four derivation rules (app_to_app, app_to_broker, node_to_node, node_to_broker). |
| REQ-GM-05 | The system shall compute edge weights from topic QoS settings (reliability, durability, priority, message size). |
| REQ-GM-06 | The system shall propagate weights from Topics to dependent Applications and Nodes. |
| REQ-GM-07 | The system shall support layer projection to produce subgraphs for app, infra, mw, and system layers. |

### 3.2 Structural Analysis (Step 2)

| ID | Requirement |
|----|-------------|
| REQ-SA-01 | The system shall compute PageRank with configurable damping factor (default 0.85). |
| REQ-SA-02 | The system shall compute Reverse PageRank on the transposed graph. |
| REQ-SA-03 | The system shall compute Betweenness Centrality for all vertices. |
| REQ-SA-04 | The system shall compute Closeness Centrality for all vertices. |
| REQ-SA-05 | The system shall compute Eigenvector Centrality for all vertices. |
| REQ-SA-06 | The system shall compute In-Degree, Out-Degree, and Total Degree centrality. |
| REQ-SA-07 | The system shall compute Clustering Coefficient for all vertices. |
| REQ-SA-08 | The system shall identify Articulation Points and compute a continuous fragmentation score AP_c(v). |
| REQ-SA-09 | The system shall identify Bridge edges and compute Bridge Ratio per vertex. |
| REQ-SA-10 | The system shall normalize all continuous metrics to [0, 1] using min-max scaling. |

### 3.3 Quality Scoring (Step 3)

| ID | Requirement |
|----|-------------|
| REQ-QS-01 | The system shall compute Reliability R(v) = w₁×PR + w₂×RPR + w₃×InDegree. |
| REQ-QS-02 | The system shall compute Maintainability M(v) = w₁×BT + w₂×OutDegree + w₃×(1−CC). |
| REQ-QS-03 | The system shall compute Availability A(v) = w₁×AP_c + w₂×BridgeRatio + w₃×Importance. |
| REQ-QS-04 | The system shall compute Vulnerability V(v) = w₁×EV + w₂×CL + w₃×OutDegree. |
| REQ-QS-05 | The system shall compute composite Quality Q(v) = w_R×R + w_M×M + w_A×A + w_V×V. |
| REQ-QS-06 | The system shall support default equal dimension weights (0.25 each). |
| REQ-QS-07 | The system shall support AHP-derived weights from pairwise comparison matrices. |
| REQ-QS-08 | The system shall validate AHP matrix consistency (Consistency Ratio < 0.10). |
| REQ-QS-09 | The system shall classify components into five criticality levels (CRITICAL, HIGH, MEDIUM, LOW, MINIMAL) using box-plot statistical thresholds. |
| REQ-QS-10 | The system shall fall back to percentile-based classification when sample size < 12. |

### 3.4 Failure Simulation (Step 4)

| ID | Requirement |
|----|-------------|
| REQ-FS-01 | The system shall simulate CRASH failure mode (complete component removal). |
| REQ-FS-02 | The system shall support DEGRADED, PARTITION, and OVERLOAD failure modes. |
| REQ-FS-03 | The system shall propagate cascading failures through three rules: physical (Node → hosted components), logical (Broker → routed Topics), and application (Publisher → starved Subscribers). |
| REQ-FS-04 | The system shall measure Reachability Loss (fraction of broken pub-sub paths). |
| REQ-FS-05 | The system shall measure Fragmentation (graph disconnection after removal). |
| REQ-FS-06 | The system shall measure Throughput Loss (weighted message delivery capacity reduction). |
| REQ-FS-07 | The system shall compute composite Impact I(v) = w_r×Reachability + w_f×Fragmentation + w_t×Throughput. |
| REQ-FS-08 | The system shall support exhaustive simulation (all components in a layer). |
| REQ-FS-09 | The system shall support Monte Carlo mode with configurable cascade probability and trial count. |

### 3.5 Validation (Step 5)

| ID | Requirement |
|----|-------------|
| REQ-VA-01 | The system shall compute Spearman rank correlation ρ between Q(v) and I(v). |
| REQ-VA-02 | The system shall compute Precision, Recall, and F1-Score for critical/non-critical classification. |
| REQ-VA-03 | The system shall compute Top-K overlap for K = 5 and K = 10. |
| REQ-VA-04 | The system shall compute NDCG@K (Normalized Discounted Cumulative Gain). |
| REQ-VA-05 | The system shall compute RMSE and MAE error metrics. |
| REQ-VA-06 | The system shall compute Cohen's κ for chance-corrected agreement. |
| REQ-VA-07 | The system shall evaluate results against configurable validation targets and report pass/fail status. |

### 3.6 Visualization (Step 6)

| ID | Requirement |
|----|-------------|
| REQ-VZ-01 | The system shall generate an HTML dashboard with KPI summary cards. |
| REQ-VZ-02 | The system shall generate criticality distribution charts (pie) and component ranking charts (bar). |
| REQ-VZ-03 | The system shall generate an interactive force-directed network graph (vis.js) with hover, click, drag, zoom, and double-click interactions. |
| REQ-VZ-04 | The system shall display sortable and filterable component detail tables with RMAV breakdown. |
| REQ-VZ-05 | The system shall display a correlation scatter plot of Q(v) vs. I(v). |
| REQ-VZ-06 | The system shall display a dependency matrix heatmap sorted by criticality. |
| REQ-VZ-07 | The system shall display validation metrics with pass/fail indicators. |
| REQ-VZ-08 | The system shall support multi-layer comparison views. |

### 3.7 Multi-Layer Analysis

| ID | Requirement |
|----|-------------|
| REQ-ML-01 | The system shall support Application layer analysis (app_to_app dependencies). |
| REQ-ML-02 | The system shall support Infrastructure layer analysis (node_to_node dependencies). |
| REQ-ML-03 | The system shall support Middleware layer analysis (app_to_broker and node_to_broker dependencies). |
| REQ-ML-04 | The system shall support Complete System layer analysis (all dependency types combined). |

### 3.8 Command-Line Interface

| ID | Requirement |
|----|-------------|
| REQ-CLI-01 | The system shall provide individual CLI commands for each pipeline step (generate, import, analyze, simulate, validate, visualize). |
| REQ-CLI-02 | The system shall provide a pipeline orchestrator (`run.py --all`) that executes all steps sequentially. |
| REQ-CLI-03 | The system shall support `--layer` flag for targeting specific architectural layers. |
| REQ-CLI-04 | The system shall support `--output` flag for exporting results to JSON. |

---

## 4. Non-Functional Requirements

### 4.1 Performance

| ID | Requirement | Metric |
|----|-------------|--------|
| REQ-PERF-01 | Analysis shall complete within 1 second for small systems. | ~30 components |
| REQ-PERF-02 | Analysis shall complete within 5 seconds for medium systems. | ~100 components |
| REQ-PERF-03 | Analysis shall complete within 20 seconds for large systems. | ~600 components |
| REQ-PERF-04 | Dashboard generation shall complete within 10 seconds. | Any scale |

### 4.2 Accuracy

| ID | Requirement |
|----|-------------|
| REQ-ACC-01 | Spearman ρ shall achieve ≥ 0.70 at the application layer. |
| REQ-ACC-02 | F1-Score shall achieve ≥ 0.80 at the application layer. |
| REQ-ACC-03 | Precision and Recall shall each achieve ≥ 0.80 at the application layer. |
| REQ-ACC-04 | Top-5 overlap shall achieve ≥ 40%. |
| REQ-ACC-05 | Prediction accuracy shall improve with system scale. |

### 4.3 Scalability

| ID | Requirement |
|----|-------------|
| REQ-SCAL-01 | The system shall support systems with up to 1,000 components. |
| REQ-SCAL-02 | The system shall support graphs with up to 10,000 edges. |

### 4.4 Reliability

| ID | Requirement |
|----|-------------|
| REQ-REL-01 | The system shall handle invalid input gracefully with descriptive error messages. |
| REQ-REL-02 | The system shall recover from Neo4j connection failures. |
| REQ-REL-03 | The system shall validate AHP matrix consistency before computing weights. |

### 4.5 Portability

| ID | Requirement |
|----|-------------|
| REQ-PORT-01 | The system shall run on Linux (Ubuntu 20.04+), macOS (11+), and Windows (10+). |
| REQ-PORT-02 | The system shall run on x86-64 and ARM64 architectures. |

### 4.6 Maintainability

| ID | Requirement |
|----|-------------|
| REQ-MAINT-01 | Code shall follow Python PEP 8 style guidelines. |
| REQ-MAINT-02 | All public APIs shall have docstrings. |
| REQ-MAINT-03 | Module unit test coverage shall be ≥ 80%. |

### 4.7 Hardware

| Scale | Minimum RAM |
|-------|-------------|
| Small (< 100 components) | 4 GB |
| Medium (100–500 components) | 8 GB |
| Enterprise (> 500 components) | 16 GB |

---

## 5. Data Requirements

### 5.1 Graph Data Model

#### 5.1.1 Vertex Types

| Type | Description | Examples |
|------|-------------|---------|
| Node | Physical or virtual host | Server, VM, Container |
| Broker | Message routing middleware | DDS Participant, Kafka Broker, MQTT Broker |
| Topic | Named message channel with QoS settings | `/sensors/lidar`, `orders.created` |
| Application | Software component that publishes or subscribes | ROS Node, Kafka Consumer, Microservice |
| Library | Shared code dependency | Navigation Library, Shared Module |

#### 5.1.2 Edge Types

| Type | From → To | Meaning |
|------|-----------|---------|
| PUBLISHES_TO | Application → Topic | Sends messages to topic |
| SUBSCRIBES_TO | Application → Topic | Receives messages from topic |
| ROUTES | Broker → Topic | Manages topic routing |
| RUNS_ON | Application/Broker → Node | Deployed on host |
| CONNECTS_TO | Node → Node | Network connection |
| USES | Application/Library → Library | Shared code dependency |
| DEPENDS_ON | Component → Component | Derived logical dependency (four subtypes) |

#### 5.1.3 QoS Attributes and Weight Contributions

| Attribute | Possible Values | Weight Contribution |
|-----------|----------------|---------------------|
| Reliability | RELIABLE, BEST_EFFORT | +0.30 for RELIABLE |
| Durability | PERSISTENT, TRANSIENT, TRANSIENT_LOCAL, VOLATILE | +0.40 for PERSISTENT |
| Priority | URGENT, HIGH, MEDIUM, LOW | +0.30 for URGENT |
| Message Size | Bytes | log₂(1 + size/1024) / 10, capped at 1.0 |

### 5.2 Storage

Neo4j Graph Database is the primary storage. Vertices are stored as labeled nodes with properties. Edges are stored as typed relationships with weights. Analysis results are stored as node properties or exported to JSON/CSV.

### 5.3 Data Validation Rules

| ID | Rule |
|----|------|
| REQ-DV-01 | All component IDs must be unique within the system. |
| REQ-DV-02 | All edge endpoints must reference existing components. |
| REQ-DV-03 | QoS values must be within defined valid ranges. |
| REQ-DV-04 | The graph must be connected for meaningful analysis. |

### 5.4 Domain Applicability

| Domain | Application → | Broker → | Topic → | Example Use Case |
|--------|--------------|----------|---------|-----------------|
| ROS 2 / DDS | ROS Node | DDS Participant | ROS Topic | Autonomous vehicle perception |
| Apache Kafka | Producer/Consumer | Kafka Broker | Kafka Topic | Financial trading platforms |
| MQTT | MQTT Client | MQTT Broker | MQTT Topic | IoT smart city deployments |
| Custom | Microservice | Message Queue | Event Channel | Enterprise SOA systems |

---

## 6. Validation Targets and Achieved Results

### 6.1 Validation Targets

| Metric | Target | Gate Level | Rationale |
|--------|--------|-----------|-----------|
| Spearman ρ | ≥ 0.70 | Primary | Strong rank correlation between prediction and reality |
| p-value | ≤ 0.05 | Primary | Statistical significance |
| F1-Score | ≥ 0.80 | Primary | Balanced precision and recall |
| Top-5 Overlap | ≥ 40% | Primary | Agreement on the most critical components |
| RMSE | ≤ 0.25 | Secondary | Prediction error bound |
| Precision | ≥ 0.80 | Reported | Minimize false alarms |
| Recall | ≥ 0.80 | Reported | Catch all critical components |
| Cohen's κ | ≥ 0.60 | Reported | Chance-corrected agreement |
| Top-10 Overlap | ≥ 50% | Reported | Extended critical set agreement |
| MAE | ≤ 0.20 | Reported | Absolute error bound |

### 6.2 Achieved Results by Layer

| Metric | Application Layer | Infrastructure Layer | Target |
|--------|-------------------|----------------------|--------|
| Spearman ρ | 0.85 | 0.54 | ≥ 0.70 |
| F1-Score | 0.83 | 0.68 | ≥ 0.80 |
| Precision | 0.86 | 0.71 | ≥ 0.80 |
| Recall | 0.80 | 0.65 | ≥ 0.80 |
| Top-5 Overlap | 62% | 40% | ≥ 40% |

Application layer consistently meets all targets. Infrastructure layer shows lower but improving correlation, reflecting the inherent difficulty of capturing infrastructure dependencies through topology alone.

### 6.3 Accuracy by System Scale

| Scale | Components | Spearman ρ | F1-Score | Analysis Time |
|-------|------------|------------|----------|---------------|
| Tiny | 5–10 | 0.72 | 0.70 | < 0.5 s |
| Small | 10–25 | 0.78 | 0.75 | < 1 s |
| Medium | 30–50 | 0.82 | 0.80 | ~2 s |
| Large | 60–100 | 0.85 | 0.83 | ~5 s |
| XLarge | 150–300 | 0.88 | 0.85 | ~20 s |

Prediction accuracy improves with scale — larger systems produce more stable centrality distributions and more reliable correlation.

### 6.4 Future Extensions

Planned extensions include Graph Neural Network (GNN) integration for enhanced prediction, temporal graph evolution analysis for dynamic systems, multi-objective optimization for architecture refactoring recommendations, digital twin implementation with continuous calibration, and extension beyond pub-sub to REST/gRPC/GraphQL architectures.

---

## Appendix A: Quality Formula Reference

### A.1 Reliability — Fault Propagation Risk

```
R(v) = 0.40 × PR(v) + 0.35 × RPR(v) + 0.25 × InDegree(v)
```

High R(v) means failure of this component would propagate widely through the dependency chain. PageRank captures transitive influence; Reverse PageRank captures cascade direction; In-Degree counts direct dependents.

### A.2 Maintainability — Coupling Complexity

```
M(v) = 0.40 × BT(v) + 0.35 × OutDegree(v) + 0.25 × (1 − CC(v))
```

High M(v) means the component is tightly coupled and hard to change. Betweenness identifies bottleneck position; Out-Degree measures efferent coupling; low Clustering means neighbors are not interconnected (harder to refactor).

### A.3 Availability — SPOF Risk

```
A(v) = 0.50 × AP_c(v) + 0.30 × BridgeRatio(v) + 0.20 × Importance(v)
```

Where Importance(v) = (PR(v) + RPR(v)) / 2.

High A(v) means the component is a single point of failure. AP_c is a continuous fragmentation score (not binary); Bridge Ratio measures irreplaceable connections; Importance combines both PageRank directions.

### A.4 Vulnerability — Security Exposure

```
V(v) = 0.40 × EV(v) + 0.30 × CL(v) + 0.30 × OutDegree(v)
```

High V(v) means the component is an attractive attack target. Eigenvector measures connection to high-value hubs; Closeness measures propagation speed; Out-Degree represents the attack surface (outbound paths an attacker can traverse).

### A.5 Overall Quality Score

```
Q(v) = w_R × R(v) + w_M × M(v) + w_A × A(v) + w_V × V(v)
```

Default: w_R = w_M = w_A = w_V = 0.25 (equal weighting). Adjustable via AHP for domain-specific priorities.

### A.6 Impact Score (Ground Truth)

```
I(v) = 0.40 × ReachabilityLoss + 0.30 × Fragmentation + 0.30 × ThroughputLoss
```

Computed by failure simulation (Step 4). Used as ground truth for validating Q(v) predictions.

### A.7 Box-Plot Classification

Components are classified into five criticality levels using statistical quartiles of the Q(v) distribution:

| Level | Threshold |
|-------|-----------|
| CRITICAL | Q(v) > Q3 + 1.5 × IQR (statistical outlier) |
| HIGH | Q(v) > Q3 |
| MEDIUM | Q(v) > Median |
| LOW | Q(v) > Q1 |
| MINIMAL | Q(v) ≤ Q1 |

For small samples (< 12 components), fixed percentile thresholds are used instead: CRITICAL = top 10%, HIGH = top 25%, MEDIUM = top 50%, LOW = top 75%, MINIMAL = bottom 25%.

### A.8 Metric-to-Dimension Mapping (Orthogonality)

Each raw metric feeds into at most one RMAV dimension, preventing any single metric from accumulating disproportionate weight in Q(v):

| Metric | R | M | A | V | Rationale |
|--------|---|---|---|---|-----------|
| PageRank | ✓ | | | | Transitive influence |
| Reverse PageRank | ✓ | | | | Cascade direction |
| In-Degree | ✓ | | | | Direct dependents |
| Betweenness | | ✓ | | | Bottleneck position |
| Out-Degree | | ✓ | | ✓ | Efferent coupling (M) / Attack surface (V) |
| Clustering | | ✓ | | | Local modularity |
| AP_c (continuous) | | | ✓ | | Structural SPOF |
| Bridge Ratio | | | ✓ | | Irreplaceable connections |
| Importance | | | ✓ | | Critical hub proxy |
| Eigenvector | | | | ✓ | Strategic importance |
| Closeness | | | | ✓ | Propagation speed |

Out-Degree is the sole exception, shared between M(v) and V(v) with distinct semantics: efferent coupling in Maintainability, attack surface in Vulnerability.

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| AHP | Analytic Hierarchy Process — derives weights from expert pairwise comparisons |
| Articulation Point | A vertex whose removal disconnects the graph |
| Betweenness Centrality | Fraction of shortest paths passing through a vertex |
| Bridge | An edge whose removal disconnects the graph |
| Broker | Message routing middleware (e.g., DDS Participant, Kafka Broker) |
| Cascade | Propagation of failures from one component to its dependents |
| Closeness Centrality | Reciprocal of average shortest-path distance to all other vertices |
| Cohen's κ | Chance-corrected inter-rater agreement statistic |
| DEPENDS_ON | Derived edge representing a logical dependency |
| Eigenvector Centrality | Measure of connection to other highly-connected vertices |
| F1-Score | Harmonic mean of precision and recall |
| GNN | Graph Neural Network |
| IQR | Interquartile Range (Q3 − Q1) |
| NDCG | Normalized Discounted Cumulative Gain |
| Neo4j | Graph database management system |
| NetworkX | Python library for graph algorithms |
| Node | Physical or virtual host infrastructure component |
| PageRank | Iterative algorithm measuring transitive importance in directed graphs |
| Pub-Sub | Publish-Subscribe messaging pattern |
| QoS | Quality of Service — attributes defining message delivery guarantees |
| RMAV | Reliability, Maintainability, Availability, Vulnerability |
| RMSE | Root Mean Squared Error |
| ROS 2 | Robot Operating System 2 |
| SPOF | Single Point of Failure |
| Spearman ρ | Rank correlation coefficient for monotonic relationships |
| Topic | Named message channel with associated QoS settings |

---

*Software-as-a-Graph Framework v2.0 · February 2026*