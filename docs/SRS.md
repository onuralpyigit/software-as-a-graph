# Software Requirements Specification

## Software-as-a-Graph

### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 2.1** · **February 2026**

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
| SDD v2.1 | Software Design Description for this project |
| STD v2.1 | Software and System Test Document for this project |

### 1.4 Document Conventions

The following conventions are used throughout this document:

- **Shall** denotes a mandatory requirement.
- **Should** denotes a recommended but non-mandatory behavior.
- **May** denotes an optional or conditional capability.
- Requirement identifiers follow the pattern **REQ-\<AREA\>-\<NN\>** (e.g., REQ-GM-01 for Graph Model requirement 1). Each identifier is unique and stable across versions.
- Mathematical notation follows the conventions defined in Appendix A.
- All metrics are scalar values in [0, 1] unless stated otherwise.

### 1.5 Document Overview

Section 2 describes the product's context, interfaces, users, and constraints. Section 3 lists all functional requirements organized by methodology step. Section 4 lists non-functional requirements for performance, accuracy, scalability, reliability, portability, maintainability, security, and hardware. Section 5 defines the data model, storage, validation rules, and domain applicability. Section 6 presents validation targets alongside empirical results achieved to date. Appendix A provides the complete formula reference for quality scoring. Appendix B is a glossary of technical terms.

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

### 2.4 Product Functions

The framework provides the following principal functions, detailed in Section 3:

- **Import** a system topology from JSON or GraphML into a Neo4j graph database.
- **Analyze** the graph structure with 13 topological metrics per component.
- **Score** each component across four RMAV quality dimensions to produce Q(v).
- **Simulate** failures exhaustively or via Monte Carlo to produce ground-truth I(v).
- **Validate** predictions statistically against simulation results.
- **Visualize** all results in an interactive HTML dashboard.

### 2.5 Constraints

The framework requires Neo4j 5.x and Python 3.9+. It performs static analysis only — accuracy depends on the completeness of the input topology specification. Memory requirements grow with system scale (see [§4.7](#47-hardware)).

### 2.6 Dependencies

| Dependency | Version | Role |
|------------|---------|------|
| Python | ≥ 3.9 | Runtime language |
| NetworkX | ≥ 3.0 | Graph algorithm library |
| Neo4j | 5.x | Graph database and GDS plugin |
| scipy | ≥ 1.9 | Statistical validation (Spearman, Pearson) |
| matplotlib | ≥ 3.6 | Chart generation |
| vis.js | 9.x (CDN) | Interactive network graph rendering |
| Chart.js | 4.x (CDN) | KPI and bar/pie chart rendering |

### 2.7 Assumptions

- The system topology is accurately and completely specified in the input file.
- QoS settings are available for all topics in the input topology.
- Neo4j 5.x is installed, running, and reachable on the configured Bolt URI.
- The analyst has sufficient domain knowledge to interpret criticality predictions relative to operational context.

### 2.8 Future Evolution

The following capabilities are planned for future versions and inform current design decisions without imposing requirements:

- **Graph Neural Network enhancement** — replacing or augmenting metric-based scoring with GNN-learned embeddings to improve cross-domain generalization.
- **Temporal analysis** — tracking how criticality distributions evolve as system topology changes over time.
- **Digital twin integration** — continuous calibration of static predictions against runtime telemetry.
- **Multi-objective refactoring recommendations** — suggesting architectural changes that reduce critical component concentration.
- **Broader architectural styles** — extending validation beyond publish-subscribe to REST-based microservices and event-driven architectures.

---

## 3. Functional Requirements

### 3.1 Graph Model Construction (Step 1)

| ID | Requirement |
|----|-------------|
| REQ-GM-01 | The system shall accept system topology in JSON format. |
| REQ-GM-02 | The system shall accept system topology in GraphML format. |
| REQ-GM-03 | The system shall create vertices for five component types: Node, Broker, Topic, Application, and Library. |
| REQ-GM-04 | The system shall create six structural edge types: RUNS_ON, ROUTES, PUBLISHES_TO, SUBSCRIBES_TO, CONNECTS_TO, and USES. |
| REQ-GM-05 | The system shall derive DEPENDS_ON edges automatically from structural relationships using four derivation rules (app_to_app, app_to_broker, node_to_node, node_to_broker). |
| REQ-GM-06 | The system shall compute edge weights from topic QoS settings (reliability, durability, priority, message size) as specified in [§5.1.3](#513-qos-attributes-and-weight-contributions). |
| REQ-GM-07 | The system shall propagate QoS-derived weights from Topics to their dependent DEPENDS_ON edges. |
| REQ-GM-08 | The system shall support layer projection to produce subgraphs for app, infra, mw, and system layers. |

> **Note on REQ-GM-07:** QoS-derived weights influence DEPENDS_ON edge weights and therefore affect all centrality calculations in Step 2. The three per-vertex weight aggregates (w(v), w\_in(v), w\_out(v)) are reported as supplementary metrics in Step 2 output but are not directly incorporated into the RMAV formulas of Step 3. The AHP-derived RMAV weights in Step 3 serve the separate role of balancing quality dimensions.

### 3.2 Structural Analysis (Step 2)

| ID | Requirement |
|----|-------------|
| REQ-SA-01 | The system shall compute PageRank PR(v) with configurable damping factor (default 0.85). |
| REQ-SA-02 | The system shall compute Reverse PageRank RPR(v) on the transposed graph G^T. |
| REQ-SA-03 | The system shall compute Betweenness Centrality BT(v) for all vertices using Brandes' algorithm. |
| REQ-SA-04 | The system shall compute Closeness Centrality CL(v) using the Wasserman-Faust normalization for directed graphs with unreachable pairs. |
| REQ-SA-05 | The system shall compute Eigenvector Centrality EV(v), falling back to In-Degree centrality with a warning if the algorithm does not converge. |
| REQ-SA-06 | The system shall compute In-Degree DG\_in(v) and Out-Degree DG\_out(v) centrality for all vertices. |
| REQ-SA-07 | The system shall compute Clustering Coefficient CC(v) for all vertices. |
| REQ-SA-08 | The system shall identify Articulation Points and compute a continuous fragmentation score AP\_c(v) reflecting reachability loss upon removal. |
| REQ-SA-09 | The system shall identify Bridge edges and compute Bridge Ratio BR(v) per vertex. |
| REQ-SA-10 | The system shall compute three QoS weight aggregates per vertex: component weight w(v), weighted in-degree w\_in(v), and weighted out-degree w\_out(v). These are reported as supplementary output only. |
| REQ-SA-11 | The system shall normalize all continuous metrics to [0, 1] using min-max scaling. |
| REQ-SA-12 | The system shall report graph-level summary statistics S(G): vertex count, edge count, density, average clustering, weakly and strongly connected component counts, diameter, articulation point count, bridge count, and average in/out-degree. |

### 3.3 Quality Scoring (Step 3)

| ID | Requirement |
|----|-------------|
| REQ-QS-01 | The system shall compute Reliability R(v) = w₁×PR + w₂×RPR + w₃×DG\_in. |
| REQ-QS-02 | The system shall compute Maintainability M(v) = w₁×BT + w₂×DG\_out + w₃×(1−CC). |
| REQ-QS-03 | The system shall compute Availability A(v) = w₁×AP\_c + w₂×BR + w₃×w(v), where w(v) is the QoS-derived component weight from Step 2 (see Appendix A.4). |
| REQ-QS-04 | The system shall compute Vulnerability V(v) = w₁×EV + w₂×CL + w₃×DG\_out. |
| REQ-QS-05 | The system shall compute composite Quality Q(v) = w\_R×R + w\_M×M + w\_A×A + w\_V×V. |
| REQ-QS-06 | The system shall support default equal dimension weights (w\_R = w\_M = w\_A = w\_V = 0.25). |
| REQ-QS-07 | The system shall support AHP-derived weights from pairwise comparison matrices. |
| REQ-QS-08 | The system shall validate AHP matrix consistency (Consistency Ratio < 0.10) before accepting weights. |
| REQ-QS-09 | The system shall classify components into five criticality levels (CRITICAL, HIGH, MEDIUM, LOW, MINIMAL) using box-plot statistical thresholds of the Q(v) distribution. |
| REQ-QS-10 | The system shall fall back to fixed-percentile classification (top 10%/25%/50%/75%) when sample size < 12. |

> **Definition of "Importance" in REQ-QS-03:** The term *w(v)* (component weight) in the Availability formula is the QoS-derived aggregate weight computed by REQ-SA-10. It captures the intrinsic criticality of the data streams handled by component v — components routing high-priority, reliable, or large-payload traffic score higher regardless of their structural position. This is distinct from the AHP-derived dimension weights w\_R, w\_M, w\_A, w\_V.

### 3.4 Failure Simulation (Step 4)

| ID | Requirement |
|----|-------------|
| REQ-FS-01 | The system shall simulate CRASH failure mode (complete component removal). |
| REQ-FS-02 | The system shall support DEGRADED, PARTITION, and OVERLOAD failure modes. |
| REQ-FS-03 | The system shall propagate cascading failures through three rules: physical (Node → hosted components), logical (Broker → routed Topics), and application (Publisher → starved Subscribers). |
| REQ-FS-04 | The system shall measure Reachability Loss (fraction of broken pub-sub paths). |
| REQ-FS-05 | The system shall measure Fragmentation (graph disconnection after removal). |
| REQ-FS-06 | The system shall measure Throughput Loss (weighted message delivery capacity reduction). |
| REQ-FS-07 | The system shall compute composite Impact I(v) = w\_r×ReachabilityLoss + w\_f×Fragmentation + w\_t×ThroughputLoss, with default weights w\_r = 0.40, w\_f = 0.30, w\_t = 0.30. All three weights are configurable. |
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
| REQ-VA-07 | The system shall evaluate results against configurable validation targets and report pass/fail status per metric. |

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
| REQ-ML-01 | The system shall support Application layer analysis (app\_to\_app dependencies). |
| REQ-ML-02 | The system shall support Infrastructure layer analysis (node\_to\_node dependencies). |
| REQ-ML-03 | The system shall support Middleware layer analysis (app\_to\_broker and node\_to\_broker dependencies). |
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

| ID | Requirement | Scale Reference |
|----|-------------|-----------------|
| REQ-PERF-01 | Analysis shall complete within 1 second for small systems. | ~30 components |
| REQ-PERF-02 | Analysis shall complete within 5 seconds for medium systems. | ~100 components |
| REQ-PERF-03 | Analysis shall complete within 20 seconds for large systems. | ~600 components |
| REQ-PERF-04 | Dashboard generation shall complete within 10 seconds for any supported scale. | Any scale |

### 4.2 Accuracy

The following targets apply to the **application layer**, which is the primary validation layer. Infrastructure layer results are reported for informational purposes; see [§6.2](#62-achieved-results-by-layer) for current infrastructure layer performance.

| ID | Requirement |
|----|-------------|
| REQ-ACC-01 | Spearman ρ shall achieve ≥ 0.70 at the application layer. |
| REQ-ACC-02 | F1-Score shall achieve ≥ 0.80 at the application layer. |
| REQ-ACC-03 | Precision and Recall shall each achieve ≥ 0.80 at the application layer. |
| REQ-ACC-04 | Top-5 overlap shall achieve ≥ 40% at the application layer. |
| REQ-ACC-05 | Prediction accuracy (Spearman ρ) shall not decrease as system scale increases within the same architectural domain. |

### 4.3 Scalability

| ID | Requirement |
|----|-------------|
| REQ-SCAL-01 | The system shall support systems with up to 1,000 components. |
| REQ-SCAL-02 | The system shall support graphs with up to 10,000 edges. |

### 4.4 Reliability

| ID | Requirement |
|----|-------------|
| REQ-REL-01 | The system shall handle invalid input gracefully with descriptive error messages identifying the offending field or constraint. |
| REQ-REL-02 | The system shall recover from Neo4j connection failures with retry logic and a clear error message if the connection cannot be established. |
| REQ-REL-03 | The system shall validate AHP matrix consistency (Consistency Ratio < 0.10) before computing weights and abort with a diagnostic message if the check fails. |

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

### 4.7 Security

| ID | Requirement |
|----|-------------|
| REQ-SEC-01 | The system shall support Neo4j authentication via configurable username and password credentials. |
| REQ-SEC-02 | Credentials shall not be stored in source code or committed to version control; they shall be read from environment variables or a configuration file excluded from VCS. |
| REQ-SEC-03 | The system shall support encrypted Bolt connections (bolt+s://) to Neo4j when configured. |

### 4.8 Logging and Observability

| ID | Requirement |
|----|-------------|
| REQ-LOG-01 | The system shall emit structured log output at configurable verbosity levels (DEBUG, INFO, WARNING, ERROR). |
| REQ-LOG-02 | Each pipeline step shall log its start time, completion time, and key output metrics at INFO level. |
| REQ-LOG-03 | Validation pass/fail results shall be logged at INFO level with the metric value and threshold that was evaluated. |

### 4.9 Hardware

| ID | Scale | Minimum RAM |
|----|-------|-------------|
| REQ-HW-01 | Small (< 100 components) | 4 GB |
| REQ-HW-02 | Medium (100–500 components) | 8 GB |
| REQ-HW-03 | Enterprise (> 500 components) | 16 GB |

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

Edge weights are additive. Each attribute contributes a non-negative increment; attributes not matching a listed condition contribute 0.

| Attribute | Condition | Weight Increment |
|-----------|-----------|-----------------|
| Reliability | RELIABLE | +0.30 |
| Reliability | BEST\_EFFORT | +0.00 |
| Durability | PERSISTENT | +0.40 |
| Durability | TRANSIENT or TRANSIENT\_LOCAL | +0.20 |
| Durability | VOLATILE | +0.00 |
| Priority | URGENT | +0.30 |
| Priority | HIGH | +0.20 |
| Priority | MEDIUM | +0.10 |
| Priority | LOW | +0.00 |
| Message Size | Any | log₂(1 + size\_bytes / 1024) / 10, capped at 1.0 |

The minimum possible edge weight is 0.0 (BEST\_EFFORT, VOLATILE, LOW, zero-byte message). The maximum without the size component is 1.0. The size term adds up to 1.0 on top, so the theoretical maximum uncapped total is 2.0; in practice the cap ensures size never exceeds 1.0 alone.

### 5.2 Storage

Neo4j Graph Database is the primary storage. Vertices are stored as labeled nodes with properties. Edges are stored as typed relationships with weights. Analysis results are stored as node properties or exported to JSON/CSV.

### 5.3 Data Validation Rules

| ID | Rule |
|----|------|
| REQ-DV-01 | All component IDs must be unique within the system. |
| REQ-DV-02 | All edge endpoints must reference existing components. |
| REQ-DV-03 | QoS values must be within the defined valid ranges per [§5.1.3](#513-qos-attributes-and-weight-contributions). |
| REQ-DV-04 | The imported graph must be **weakly connected** (i.e., connected when edge directionality is ignored) for layer projection and centrality computation to be meaningful. Disconnected graphs are accepted with a warning; analysis proceeds on each weakly connected component separately. |

> **Rationale for REQ-DV-04:** Directed graphs can be weakly connected without being strongly connected. Requiring strong connectivity would be too restrictive — most real-world pub-sub topologies are not strongly connected. Weakly connected ensures no component is entirely isolated, which would render its centrality metrics trivially zero.

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

All targets in this table apply to the **application layer**. Infrastructure layer targets are currently aspirational; see §6.2 for current results.

### 6.2 Achieved Results by Layer

| Metric | Application Layer | Infrastructure Layer | Target |
|--------|-------------------|----------------------|--------|
| Spearman ρ | 0.85 ✓ | 0.54 | ≥ 0.70 |
| F1-Score | 0.83 ✓ | 0.68 | ≥ 0.80 |
| Precision | 0.86 ✓ | 0.71 | ≥ 0.80 |
| Recall | 0.80 ✓ | 0.65 | ≥ 0.80 |
| Top-5 Overlap | 62% ✓ | 40% ✓ | ≥ 40% |

Application layer consistently meets all targets. Infrastructure layer shows lower but improving correlation, reflecting the inherent difficulty of capturing infrastructure dependencies through topology alone. Improving infrastructure layer accuracy is an active research direction (see §2.8).

### 6.3 Accuracy by System Scale

| Scale | Components | Spearman ρ | F1-Score | Analysis Time |
|-------|------------|------------|----------|---------------|
| Tiny | 5–10 | 0.72 | 0.70 | < 0.5 s |
| Small | 10–25 | 0.78 | 0.75 | < 1 s |
| Medium | 30–50 | 0.82 | 0.80 | ~2 s |
| Large | 60–100 | 0.85 | 0.83 | ~5 s |
| XLarge | 150–300 | 0.88 | 0.85 | ~20 s |
| Enterprise | 300–1,000 | TBD (empirical validation in progress) | TBD | ≤ 20 s (REQ-PERF-03) |

Prediction accuracy improves with scale — larger systems produce more stable centrality distributions and more reliable correlation. The Enterprise row represents the REQ-SCAL-01 upper bound; empirical results at this scale are part of ongoing thesis validation work.

---

## Appendix A: Quality Formula Reference

### A.1 Edge Weight Formula

```
w(e) = w_reliability + w_durability + w_priority + w_size

w_reliability = 0.30  if RELIABLE,  0.00 otherwise
w_durability  = 0.40  if PERSISTENT, 0.20 if TRANSIENT/TRANSIENT_LOCAL, 0.00 if VOLATILE
w_priority    = 0.30  if URGENT, 0.20 if HIGH, 0.10 if MEDIUM, 0.00 if LOW
w_size        = min(log₂(1 + size_bytes / 1024) / 10, 1.0)
```

### A.2 Reliability Score

```
R(v) = w₁ × PR(v) + w₂ × RPR(v) + w₃ × DG_in(v)
```

Default intra-dimension weights: w₁ = 0.40, w₂ = 0.30, w₃ = 0.30.

### A.3 Maintainability Score

```
M(v) = w₁ × BT(v) + w₂ × DG_out(v) + w₃ × (1 − CC(v))
```

Default intra-dimension weights: w₁ = 0.40, w₂ = 0.30, w₃ = 0.30.

### A.4 Availability Score

```
A(v) = w₁ × AP_c(v) + w₂ × BR(v) + w₃ × w(v)
```

where w(v) = normalized QoS-derived component weight from Step 1 (REQ-SA-10). A component routing high-priority, reliable, or large-payload data scores higher in Availability regardless of structural position, reflecting its operational indispensability.

Default intra-dimension weights: w₁ = 0.40, w₂ = 0.35, w₃ = 0.25.

### A.5 Vulnerability Score

```
V(v) = w₁ × EV(v) + w₂ × CL(v) + w₃ × DG_out(v)
```

Default intra-dimension weights: w₁ = 0.40, w₂ = 0.35, w₃ = 0.25.

Note: DG\_out(v) appears in both Maintainability (as efferent coupling) and Vulnerability (as attack surface) with distinct semantic rationale. It is the sole metric shared between two RMAV dimensions.

### A.6 Overall Quality Score

```
Q(v) = w_R × R(v) + w_M × M(v) + w_A × A(v) + w_V × V(v)
```

Default inter-dimension weights: w\_R = w\_M = w\_A = w\_V = 0.25 (equal weighting). Adjustable via AHP for domain-specific priorities; requires Consistency Ratio < 0.10 (REQ-QS-08).

### A.7 Impact Score (Ground Truth)

```
I(v) = w_r × ReachabilityLoss(v) + w_f × Fragmentation(v) + w_t × ThroughputLoss(v)
```

Default weights: w\_r = 0.40, w\_f = 0.30, w\_t = 0.30. All weights are configurable (REQ-FS-07). Computed by failure simulation (Step 4); used as ground truth for validating Q(v) predictions.

### A.8 Box-Plot Classification

Components are classified into five criticality levels using statistical quartiles of the Q(v) distribution:

| Level | Threshold |
|-------|-----------|
| CRITICAL | Q(v) > Q3 + 1.5 × IQR (statistical outlier) |
| HIGH | Q(v) > Q3 |
| MEDIUM | Q(v) > Median |
| LOW | Q(v) > Q1 |
| MINIMAL | Q(v) ≤ Q1 |

For small samples (< 12 components), fixed percentile thresholds are used instead: CRITICAL = top 10%, HIGH = top 25%, MEDIUM = top 50%, LOW = top 75%, MINIMAL = bottom 25%.

### A.9 Metric-to-Dimension Mapping (Orthogonality)

Each raw metric feeds into at most one RMAV dimension, preventing any single metric from accumulating disproportionate weight in Q(v):

| Metric | Symbol | R | M | A | V | Notes |
|--------|--------|---|---|---|---|-------|
| PageRank | PR | ✓ | | | | Transitive influence |
| Reverse PageRank | RPR | ✓ | | | | Cascade direction |
| In-Degree | DG\_in | ✓ | | | | Direct dependents |
| Betweenness | BT | | ✓ | | | Bottleneck position |
| Out-Degree | DG\_out | | ✓ | | ✓ | Efferent coupling (M) / Attack surface (V) |
| Clustering | CC | | ✓ | | | Local modularity |
| AP Score | AP\_c | | | ✓ | | Structural SPOF |
| Bridge Ratio | BR | | | ✓ | | Irreplaceable connections |
| Component Weight | w(v) | | | ✓ | | QoS-derived importance |
| Eigenvector | EV | | | | ✓ | Strategic influence |
| Closeness | CL | | | | ✓ | Propagation speed |
| Weighted In-Degree | w\_in | — | — | — | — | Reported only |
| Weighted Out-Degree | w\_out | — | — | — | — | Reported only |

Out-Degree is the sole exception in the active metrics, shared between M(v) and V(v) with distinct semantics. The three QoS weight aggregates (w, w\_in, w\_out) are part of the 13-metric output vector but do not participate directly in the RMAV formulas (except w(v) in A(v)).

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| AHP | Analytic Hierarchy Process — derives dimension weights from expert pairwise comparisons |
| Articulation Point | A vertex whose removal disconnects the graph (increases the number of weakly connected components) |
| Betweenness Centrality | Fraction of all-pairs shortest paths passing through a given vertex |
| Bridge | An edge whose removal disconnects the graph |
| Broker | Message routing middleware (e.g., DDS Participant, Kafka Broker) |
| Cascade | Propagation of failures from one component to its dependents |
| Closeness Centrality | Reciprocal of average shortest-path distance to all reachable vertices, normalized for partial reachability |
| Cohen's κ | Chance-corrected inter-rater agreement statistic |
| Component Weight w(v) | QoS-derived aggregate weight reflecting the criticality of data streams handled by component v |
| DEPENDS\_ON | Derived edge representing a logical dependency between two components |
| Eigenvector Centrality | Centrality measure that accounts for the quality (not just count) of a vertex's neighbors |
| F1-Score | Harmonic mean of precision and recall |
| GNN | Graph Neural Network |
| IQR | Interquartile Range (Q3 − Q1) |
| NDCG | Normalized Discounted Cumulative Gain — ranking quality metric that penalizes highly-ranked errors more than low-ranked errors |
| Neo4j | Graph database management system |
| NetworkX | Python library for graph algorithms |
| Node | Physical or virtual host infrastructure component |
| PageRank | Iterative random-walk algorithm measuring transitive importance in directed graphs |
| Pub-Sub | Publish-Subscribe messaging pattern |
| QoS | Quality of Service — attributes defining message delivery guarantees (reliability, durability, priority) |
| RMAV | Reliability, Maintainability, Availability, Vulnerability — the four quality dimensions |
| RMSE | Root Mean Squared Error |
| ROS 2 | Robot Operating System 2 |
| SPOF | Single Point of Failure — a component whose removal disconnects or severely degrades the system |
| Spearman ρ | Rank correlation coefficient measuring monotonic agreement between two ordered sequences |
| Topic | Named message channel with associated QoS settings |
| Weakly Connected | A directed graph property: the graph is connected when all edges are treated as undirected |

---

*Software-as-a-Graph Framework v2.1 · February 2026*
*Istanbul Technical University, Computer Engineering Department*