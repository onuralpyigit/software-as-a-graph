# Software Requirements Specification

## Software-as-a-Graph
### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 1.0**  
**January 2026**

Istanbul Technical University  
Computer Engineering Department

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [Specific Requirements](#3-specific-requirements)
4. [System Features](#4-system-features)
5. [Data Requirements](#5-data-requirements)
6. [Other Requirements](#6-other-requirements)
7. [Appendix A: Quality Formula Reference](#appendix-a-quality-formula-reference)

---

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) document provides a comprehensive description of the Software-as-a-Graph framework. The framework predicts which components in a distributed publish-subscribe system are most critical—those whose failure would cause the greatest impact—using only the system's architectural structure.

This document is intended for software architects, system engineers, reliability engineers, researchers, and developers who need to understand the requirements, capabilities, and constraints of the Software-as-a-Graph system.

### 1.2 Scope

Software-as-a-Graph is a comprehensive framework for graph-based modeling and analysis of distributed publish-subscribe systems. The system transforms architectural topology into a weighted directed graph and applies topological metrics to predict component criticality before deployment, without expensive runtime monitoring.

The framework provides:
- Graph model construction from system topology definitions
- Multi-layer structural analysis (application, infrastructure, middleware, system)
- RMAV quality scoring (Reliability, Maintainability, Availability, Vulnerability)
- Failure simulation with cascade propagation modeling
- Statistical validation comparing predictions against ground truth
- Interactive visualization dashboards for analysis results

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|------------|
| AHP | Analytic Hierarchy Process - method for deriving weights from pairwise comparisons |
| Articulation Point | A vertex whose removal disconnects a connected graph into multiple components |
| Betweenness Centrality | Measure of how often a node lies on shortest paths between other nodes |
| Broker | Message routing middleware component (e.g., DDS Participant, Kafka Broker) |
| Cascade | Propagation of failures from one component to dependent components |
| DEPENDS_ON | Derived edge representing logical dependency between components |
| Eigenvector Centrality | Measure of connection to other highly-connected nodes |
| F1-Score | Harmonic mean of precision and recall |
| GNN | Graph Neural Network |
| IQR | Interquartile Range - statistical measure for box-plot classification |
| Neo4j | Graph database management system |
| NetworkX | Python library for graph algorithms |
| Node | Physical or virtual host infrastructure component |
| PageRank | Algorithm measuring transitive importance in directed graphs |
| Pub-Sub | Publish-Subscribe messaging pattern |
| QoS | Quality of Service - attributes defining message delivery guarantees |
| RMAV | Reliability, Maintainability, Availability, Vulnerability quality dimensions |
| ROS 2 | Robot Operating System 2 - middleware for robotics applications |
| SPOF | Single Point of Failure |
| Spearman ρ | Rank correlation coefficient measuring monotonic relationships |
| Topic | Named message channel with associated QoS settings |

### 1.4 References

- IEEE 830-1998: IEEE Recommended Practice for Software Requirements Specifications
- IEEE RASSE 2025: Conference proceedings for the published methodology paper
- Neo4j Documentation: https://neo4j.com/docs/
- NetworkX Documentation: https://networkx.org/documentation/
- Saaty, T.L. (1980): The Analytic Hierarchy Process - McGraw-Hill

### 1.5 Overview

The remainder of this document is organized as follows: Section 2 provides an overall description of the system including product perspective, functions, and constraints. Section 3 details specific functional and non-functional requirements. Section 4 describes system features and their specifications. Section 5 covers data requirements and Section 6 addresses additional requirements.

---

## 2. Overall Description

### 2.1 Product Perspective

Software-as-a-Graph is a standalone analysis framework that integrates with existing distributed system architectures. It operates as a pre-deployment analysis tool that transforms system topology specifications into graph models for criticality prediction.

#### 2.1.1 System Interfaces

- **Neo4j Graph Database**: Primary storage for graph models and analysis results
- **JSON/GraphML Import**: System topology input formats
- **HTML Export**: Interactive visualization dashboards
- **CLI Interface**: Command-line tools for pipeline execution

#### 2.1.2 User Interfaces

- Command-line interface for pipeline orchestration and analysis
- HTML dashboard for interactive visualization of results
- JSON/CSV export for integration with external tools

#### 2.1.3 Hardware Interfaces

The system operates on standard computing hardware with sufficient memory for graph processing. Recommended minimum: 8GB RAM, multi-core processor for parallel analysis operations.

#### 2.1.4 Software Interfaces

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.9+ | Primary runtime environment |
| Neo4j | 5.x | Graph database storage and GDS algorithms |
| NetworkX | 2.6+ | Graph algorithms library |
| vis.js | 9.x | Interactive network visualization |
| Node.js | 16+ | Document generation and tooling |

### 2.2 Product Functions

The framework implements a six-step methodology:

#### 2.2.1 Graph Model Construction (Step 1)

Transform distributed pub-sub system topology into a weighted directed graph with derived dependencies. Components include Nodes, Brokers, Topics, Applications, and Libraries with structural relationships (RUNS_ON, ROUTES, PUBLISHES_TO, SUBSCRIBES_TO, CONNECTS_TO, USES) and derived DEPENDS_ON edges.

#### 2.2.2 Structural Analysis (Step 2)

Compute topological metrics including PageRank, Reverse PageRank, Betweenness Centrality, Degree Centrality, Clustering Coefficient, Eigenvector Centrality, Closeness Centrality, and identify Articulation Points and Bridges.

#### 2.2.3 Quality Scoring (Step 3)

Calculate RMAV quality scores using AHP-weighted formulas:
- Reliability R(v) = w₁×PR + w₂×RPR + w₃×ID
- Maintainability M(v) = w₁×BT + w₂×DG + w₃×(1-CC)
- Availability A(v) = w₁×AP + w₂×BR + w₃×Importance
- Vulnerability V(v) = w₁×EV + w₂×CL + w₃×ID
- Overall Q(v) = w_R×R(v) + w_M×M(v) + w_A×A(v) + w_V×V(v)

#### 2.2.4 Failure Simulation (Step 4)

Simulate component failures with cascade propagation. Measure actual impact I(v) through reachability loss, fragmentation, and throughput reduction metrics.

#### 2.2.5 Validation (Step 5)

Statistically compare predicted Q(v) against actual I(v) using Spearman correlation, F1-score, precision, recall, and Top-K overlap metrics.

#### 2.2.6 Visualization (Step 6)

Generate interactive HTML dashboards with KPI cards, charts, network graphs, data tables, and validation metrics.

### 2.3 User Characteristics

Target users include:
- **Software Architects**: Evaluating system designs for reliability risks
- **Reliability Engineers**: Identifying critical components for redundancy planning
- **DevOps Engineers**: Prioritizing monitoring and alerting configurations
- **Security Engineers**: Identifying high-value attack targets
- **Researchers**: Validating graph-based prediction methodologies

### 2.4 Constraints

- Graph database dependency: Requires Neo4j 5.x installation
- Python 3.9+ environment required
- Memory constraints for very large systems (>1000 components)
- Static analysis only - does not perform runtime monitoring
- Accuracy depends on completeness of input topology specification

### 2.5 Assumptions and Dependencies

- System topology is accurately specified in input format
- QoS settings are available for weight calculation
- Neo4j database is accessible and properly configured
- Network connectivity between analysis host and Neo4j instance

---

## 3. Specific Requirements

### 3.1 External Interface Requirements

#### 3.1.1 User Interfaces

- **REQ-UI-001**: The system shall provide a command-line interface for all pipeline operations.
- **REQ-UI-002**: The system shall generate HTML dashboards viewable in modern web browsers.
- **REQ-UI-003**: Dashboard shall display interactive network visualizations using vis.js.
- **REQ-UI-004**: Dashboard shall provide sortable and filterable data tables.

#### 3.1.2 Hardware Interfaces

- **REQ-HW-001**: The system shall operate on x86-64 or ARM64 architecture systems.
- **REQ-HW-002**: The system shall require minimum 4GB RAM for small-scale analysis.
- **REQ-HW-003**: The system shall require minimum 16GB RAM for enterprise-scale analysis.

#### 3.1.3 Software Interfaces

- **REQ-SW-001**: The system shall interface with Neo4j database via Bolt protocol.
- **REQ-SW-002**: The system shall support JSON format for topology import.
- **REQ-SW-003**: The system shall support GraphML format for graph export.
- **REQ-SW-004**: The system shall support CSV format for metrics export.

#### 3.1.4 Communication Interfaces

- **REQ-CI-001**: The system shall communicate with Neo4j via bolt://localhost:7687 by default.
- **REQ-CI-002**: The system shall support configurable Neo4j connection parameters.
- **REQ-CI-003**: The system shall support remote Neo4j connections with authentication.

### 3.2 Functional Requirements

#### 3.2.1 Graph Model Construction

- **REQ-GMC-001**: The system shall accept system topology in JSON format.
- **REQ-GMC-002**: The system shall create vertices for Node, Broker, Topic, Application, and Library components.
- **REQ-GMC-003**: The system shall create structural edges: RUNS_ON, ROUTES, PUBLISHES_TO, SUBSCRIBES_TO, CONNECTS_TO, USES.
- **REQ-GMC-004**: The system shall automatically derive DEPENDS_ON edges from structural relationships.
- **REQ-GMC-005**: The system shall calculate edge weights from Topic QoS settings.
- **REQ-GMC-006**: The system shall support these QoS attributes: reliability, durability, priority, message_size.
- **REQ-GMC-007**: The system shall propagate weights from Topics to Applications to Nodes.
- **REQ-GMC-008**: The system shall derive dependency types: app_to_app, node_to_node, app_to_broker, node_to_broker.

#### 3.2.2 Structural Analysis

- **REQ-SA-001**: The system shall compute PageRank with configurable damping factor (default 0.85).
- **REQ-SA-002**: The system shall compute Reverse PageRank on transposed graph.
- **REQ-SA-003**: The system shall compute Betweenness Centrality for all vertices.
- **REQ-SA-004**: The system shall compute In-Degree, Out-Degree, and Total Degree centrality.
- **REQ-SA-005**: The system shall compute Clustering Coefficient.
- **REQ-SA-006**: The system shall compute Eigenvector Centrality.
- **REQ-SA-007**: The system shall compute Closeness Centrality.
- **REQ-SA-008**: The system shall identify Articulation Points.
- **REQ-SA-009**: The system shall identify Bridge edges.
- **REQ-SA-010**: The system shall normalize all metrics to [0, 1] range using min-max scaling.

#### 3.2.3 Quality Scoring

- **REQ-QS-001**: The system shall compute Reliability score R(v) = w₁×PR + w₂×RPR + w₃×ID.
- **REQ-QS-002**: The system shall compute Maintainability score M(v) = w₁×BT + w₂×DG + w₃×(1-CC).
- **REQ-QS-003**: The system shall compute Availability score A(v) = w₁×AP + w₂×BR + w₃×Importance.
- **REQ-QS-004**: The system shall compute Vulnerability score V(v) = w₁×EV + w₂×CL + w₃×ID.
- **REQ-QS-005**: The system shall compute Overall Quality Q(v) combining all dimensions.
- **REQ-QS-006**: The system shall support default equal weights (0.25 each dimension).
- **REQ-QS-007**: The system shall support AHP-derived weights from pairwise comparison matrices.
- **REQ-QS-008**: The system shall validate AHP consistency using Consistency Ratio (CR < 0.10).
- **REQ-QS-009**: The system shall classify components using box-plot statistical thresholds.
- **REQ-QS-010**: The system shall assign criticality levels: CRITICAL, HIGH, MEDIUM, LOW, MINIMAL.

#### 3.2.4 Failure Simulation

- **REQ-FS-001**: The system shall simulate CRASH failure mode (complete component failure).
- **REQ-FS-002**: The system shall support DEGRADED, PARTITION, and OVERLOAD failure modes.
- **REQ-FS-003**: The system shall apply PHYSICAL cascade rule (Node failure cascades to hosted components).
- **REQ-FS-004**: The system shall apply LOGICAL cascade rule (Broker failure affects Topic routing).
- **REQ-FS-005**: The system shall apply NETWORK cascade rule (partition propagation).
- **REQ-FS-006**: The system shall measure Reachability Loss (broken pub-sub paths).
- **REQ-FS-007**: The system shall measure Fragmentation (disconnected components).
- **REQ-FS-008**: The system shall measure Throughput Loss (message delivery capacity).
- **REQ-FS-009**: The system shall compute Composite Impact I(v) = w_r×reachability + w_f×fragmentation + w_t×throughput.
- **REQ-FS-010**: The system shall support exhaustive simulation for all components in a layer.

#### 3.2.5 Validation

- **REQ-VA-001**: The system shall compute Spearman rank correlation ρ between Q(v) and I(v).
- **REQ-VA-002**: The system shall compute Precision, Recall, and F1-Score.
- **REQ-VA-003**: The system shall compute Top-K overlap for K=5, 10.
- **REQ-VA-004**: The system shall compute NDCG (Normalized Discounted Cumulative Gain).
- **REQ-VA-005**: The system shall compute RMSE and MAE error metrics.
- **REQ-VA-006**: The system shall evaluate against configurable validation targets.
- **REQ-VA-007**: The system shall report pass/fail status for each validation target.

#### 3.2.6 Visualization

- **REQ-VZ-001**: The system shall generate HTML dashboard with KPI summary cards.
- **REQ-VZ-002**: The system shall generate pie charts for criticality distribution.
- **REQ-VZ-003**: The system shall generate bar charts for component rankings.
- **REQ-VZ-004**: The system shall generate interactive network graph visualization.
- **REQ-VZ-005**: The system shall display sortable component tables.
- **REQ-VZ-006**: The system shall display validation metrics with pass/fail indicators.
- **REQ-VZ-007**: The system shall support multi-layer comparison views.

#### 3.2.7 Multi-Layer Analysis

- **REQ-ML-001**: The system shall support Application layer (app_to_app dependencies).
- **REQ-ML-002**: The system shall support Infrastructure layer (node_to_node dependencies).
- **REQ-ML-003**: The system shall support Middleware-Application layer (app_to_broker dependencies).
- **REQ-ML-004**: The system shall support Middleware-Infrastructure layer (node_to_broker dependencies).
- **REQ-ML-005**: The system shall support Complete System layer (all dependencies).

### 3.3 Non-Functional Requirements

#### 3.3.1 Performance Requirements

- **REQ-PERF-001**: Analysis shall complete within 1 second for small systems (~30 components).
- **REQ-PERF-002**: Analysis shall complete within 5 seconds for medium systems (~100 components).
- **REQ-PERF-003**: Analysis shall complete within 20 seconds for large systems (~600 components).
- **REQ-PERF-004**: Dashboard generation shall complete within 10 seconds.

#### 3.3.2 Accuracy Requirements

- **REQ-ACC-001**: Spearman correlation ρ shall achieve ≥0.70 at application layer.
- **REQ-ACC-002**: F1-Score shall achieve ≥0.80 at application layer.
- **REQ-ACC-003**: Precision shall achieve ≥0.80 at application layer.
- **REQ-ACC-004**: Recall shall achieve ≥0.80 at application layer.
- **REQ-ACC-005**: Top-5 overlap shall achieve ≥40%.
- **REQ-ACC-006**: Top-10 overlap shall achieve ≥50%.

#### 3.3.3 Reliability Requirements

- **REQ-REL-001**: The system shall handle invalid input gracefully with error messages.
- **REQ-REL-002**: The system shall recover from Neo4j connection failures.
- **REQ-REL-003**: The system shall validate AHP matrix consistency before use.

#### 3.3.4 Scalability Requirements

- **REQ-SCAL-001**: The system shall support systems with up to 1000 components.
- **REQ-SCAL-002**: The system shall support graphs with up to 10000 edges.
- **REQ-SCAL-003**: Prediction accuracy shall improve with system scale.

#### 3.3.5 Maintainability Requirements

- **REQ-MAINT-001**: Code shall follow Python PEP 8 style guidelines.
- **REQ-MAINT-002**: All public APIs shall have docstrings.
- **REQ-MAINT-003**: Modules shall have unit test coverage ≥80%.

#### 3.3.6 Portability Requirements

- **REQ-PORT-001**: The system shall run on Linux (Ubuntu 20.04+).
- **REQ-PORT-002**: The system shall run on macOS (11+).
- **REQ-PORT-003**: The system shall run on Windows (10+).

---

## 4. System Features

### 4.1 Graph Model Construction

**Description**: Transform system topology into a weighted directed graph stored in Neo4j.

**Priority**: High

| Input | Process | Output |
|-------|---------|--------|
| JSON topology file | Parse and validate topology | Validated components and edges |
| Component definitions | Create Neo4j vertices | Graph vertices with properties |
| Relationship definitions | Create Neo4j edges | Structural edges |
| Structural edges | Derive dependencies | DEPENDS_ON edges |
| QoS settings | Calculate weights | Weighted edges and vertices |

### 4.2 Structural Analysis Engine

**Description**: Compute topological metrics using NetworkX and Neo4j GDS algorithms.

**Priority**: High

| Metric | Algorithm | Interpretation |
|--------|-----------|----------------|
| PageRank | Iterative power method | Transitive importance |
| Betweenness | Brandes algorithm | Bottleneck position |
| Eigenvector | Power iteration | Strategic importance |
| Closeness | Dijkstra shortest paths | Propagation speed |
| Clustering | Triangle counting | Modularity |
| Articulation | DFS-based detection | SPOF identification |

### 4.3 RMAV Quality Scoring

**Description**: Calculate composite quality scores using AHP-weighted metric combinations.

**Priority**: High

| Dimension | Focus | Key Metrics |
|-----------|-------|-------------|
| Reliability R(v) | Fault propagation risk | PageRank, Reverse PageRank, In-Degree |
| Maintainability M(v) | Coupling complexity | Betweenness, Degree, Clustering |
| Availability A(v) | SPOF risk | Articulation Point, Bridge Ratio, Importance |
| Vulnerability V(v) | Security exposure | Eigenvector, Closeness, In-Degree |

### 4.4 Failure Simulation Engine

**Description**: Simulate failures and measure actual impact for validation.

**Priority**: High

| Cascade Type | Trigger | Effect |
|--------------|---------|--------|
| Physical | Node failure | Hosted Apps and Brokers fail |
| Logical | Broker failure | Topics become unreachable |
| Application | Publisher failure | Subscribers starved of data |
| Network | Partition | Isolated components |

### 4.5 Statistical Validation

**Description**: Compare predictions against simulation ground truth.

**Priority**: High

| Metric | Target | Purpose |
|--------|--------|---------|
| Spearman ρ | ≥0.70 | Ranking correlation |
| F1-Score | ≥0.80 | Classification accuracy |
| Precision | ≥0.80 | Avoid false alarms |
| Recall | ≥0.80 | Catch critical components |
| Top-5 Overlap | ≥40% | Agreement on most critical |
| Top-10 Overlap | ≥50% | Agreement on critical set |

### 4.6 Visualization Dashboard

**Description**: Generate interactive HTML dashboards for analysis results.

**Priority**: Medium

| Component | Technology | Purpose |
|-----------|------------|---------|
| KPI Cards | HTML/CSS | High-level metrics summary |
| Pie Charts | Chart.js | Distribution visualization |
| Bar Charts | Chart.js | Rankings and comparisons |
| Network Graph | vis.js | Interactive topology exploration |
| Data Tables | HTML/JS | Detailed component data |
| Validation Box | HTML/CSS | Pass/fail status display |

---

## 5. Data Requirements

### 5.1 Data Dictionary

#### 5.1.1 Vertex Types

| Type | Description | Example |
|------|-------------|---------|
| Node | Physical or virtual host | Server, VM, Container |
| Broker | Message routing middleware | DDS Participant, Kafka Broker |
| Topic | Named message channel with QoS | /sensors/lidar, orders.created |
| Application | Service that pub/sub to topics | ROS Node, Microservice |
| Library | Shared code dependency | Navigation Library |

#### 5.1.2 Edge Types

| Type | From | To | Meaning |
|------|------|-----|---------|
| RUNS_ON | App/Broker | Node | Deployed on host |
| ROUTES | Broker | Topic | Manages topic routing |
| PUBLISHES_TO | App | Topic | Sends messages |
| SUBSCRIBES_TO | App | Topic | Receives messages |
| CONNECTS_TO | Node | Node | Network connection |
| USES | App/Lib | Lib | Shared code dependency |
| DEPENDS_ON | Component | Component | Derived logical dependency |

#### 5.1.3 QoS Attributes

| Attribute | Values | Weight Contribution |
|-----------|--------|---------------------|
| Reliability | RELIABLE, BEST_EFFORT | +0.30 for RELIABLE |
| Durability | PERSISTENT, VOLATILE, TRANSIENT | +0.40 for PERSISTENT |
| Priority | URGENT, HIGH, NORMAL, LOW | +0.30 for URGENT |
| Message Size | Bytes | log₂(1 + size/1024) / 10, max 1.0 |

### 5.2 Data Storage

**Primary storage**: Neo4j Graph Database

- Vertices stored as labeled nodes with properties
- Edges stored as typed relationships with weights
- Analysis results stored as node/relationship properties
- Export formats: JSON, GraphML, CSV

### 5.3 Data Validation

- **REQ-DV-001**: All component IDs must be unique within the system.
- **REQ-DV-002**: All edge endpoints must reference existing components.
- **REQ-DV-003**: QoS values must be within valid ranges.
- **REQ-DV-004**: Graph must be connected for meaningful analysis.

---

## 6. Other Requirements

### 6.1 Validation Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Spearman ρ | ≥0.70 | Strong positive rank correlation required |
| F1-Score | ≥0.80 | Balanced precision/recall |
| Precision | ≥0.80 | Minimize false positives |
| Recall | ≥0.80 | Minimize missed critical components |
| Top-5 Overlap | ≥40% | Critical set agreement |
| Top-10 Overlap | ≥50% | Extended critical set agreement |
| RMSE | ≤0.25 | Prediction error bound |
| MAE | ≤0.20 | Absolute error bound |

### 6.2 Achieved Results

The framework has demonstrated strong empirical validation:

| Metric | Application Layer | Infrastructure Layer | Target |
|--------|-------------------|----------------------|--------|
| Spearman ρ | 0.85 | 0.54 | ≥0.70 |
| F1-Score | 0.83 | 0.68 | ≥0.80 |
| Precision | 0.86 | 0.71 | ≥0.80 |
| Recall | 0.80 | 0.65 | ≥0.80 |
| Top-5 Overlap | 62% | 40% | ≥40% |
| Speedup | 2.2× | 1.2× | N/A |

### 6.3 Scale Performance

| Scale | Components | Spearman ρ | F1-Score | Analysis Time |
|-------|------------|------------|----------|---------------|
| Tiny | 5-10 | 0.72 | 0.70 | <0.5s |
| Small | 10-25 | 0.78 | 0.75 | <1s |
| Medium | 30-50 | 0.82 | 0.80 | ~2s |
| Large | 60-100 | 0.85 | 0.83 | ~5s |
| XLarge | 150-300 | 0.88 | 0.85 | ~20s |

### 6.4 Domain Applicability

The framework supports multiple distributed system domains:

| Domain | Application | Broker | Example Use Case |
|--------|-------------|--------|------------------|
| ROS 2 | ROS Node | DDS Participant | Autonomous vehicle perception |
| Kafka | Producer/Consumer | Kafka Broker | Financial trading platforms |
| MQTT | MQTT Client | MQTT Broker | IoT smart city deployments |
| Custom | Microservice | Message Queue | Enterprise SOA systems |

### 6.5 Future Extensions

- Graph Neural Network (GNN) integration for improved prediction accuracy
- Temporal graph evolution analysis for dynamic systems
- Multi-objective optimization for architecture refactoring recommendations
- Digital twin implementation with continuous calibration
- Heterogeneous multi-layer dependency analysis beyond pub-sub

### 6.6 Documentation Requirements

- **REQ-DOC-001**: User documentation shall include installation guide.
- **REQ-DOC-002**: User documentation shall include CLI reference.
- **REQ-DOC-003**: User documentation shall include methodology overview.
- **REQ-DOC-004**: API documentation shall cover all public modules.
- **REQ-DOC-005**: Each methodology step shall have dedicated documentation.

---

## Appendix A: Quality Formula Reference

### A.1 Reliability Formula

```
R(v) = w₁×PR(v) + w₂×RPR(v) + w₃×ID(v)
```

**Default weights**: w₁=0.40, w₂=0.35, w₃=0.25

**Interpretation**: High R(v) indicates greater reliability risk if component fails.

### A.2 Maintainability Formula

```
M(v) = w₁×BT(v) + w₂×DG(v) + w₃×(1-CC(v))
```

**Default weights**: w₁=0.40, w₂=0.35, w₃=0.25

**Interpretation**: High M(v) indicates harder to maintain, higher change risk.

### A.3 Availability Formula

```
A(v) = w₁×AP(v) + w₂×BR(v) + w₃×Importance(v)
```

Where `Importance = (PR + RPR) / 2`

**Default weights**: w₁=0.50, w₂=0.30, w₃=0.20

**Interpretation**: High A(v) indicates higher single point of failure risk.

### A.4 Vulnerability Formula

```
V(v) = w₁×EV(v) + w₂×CL(v) + w₃×ID(v)
```

**Default weights**: w₁=0.40, w₂=0.30, w₃=0.30

**Interpretation**: High V(v) indicates higher security exposure risk.

### A.5 Composite Quality Formula

```
Q(v) = w_R×R(v) + w_M×M(v) + w_A×A(v) + w_V×V(v)
```

**Default weights**: w_R=w_M=w_A=w_V=0.25 (equal weighting)

### A.6 Impact Score Formula

```
I(v) = w_r×reachability_loss + w_f×fragmentation + w_t×throughput_loss
```

**Default weights**: w_r=0.40, w_f=0.30, w_t=0.30

### A.7 Box-Plot Classification

Components are classified using statistical quartiles:

- **CRITICAL**: Q(v) > Q3 + 1.5×IQR (statistical outlier)
- **HIGH**: Q(v) > Q3 (top quartile)
- **MEDIUM**: Q(v) > Median (above average)
- **LOW**: Q(v) > Q1 (below average)
- **MINIMAL**: Q(v) ≤ Q1 (bottom quartile)

---

*Document generated: January 2026*  
*Software-as-a-Graph Framework v1.0*
