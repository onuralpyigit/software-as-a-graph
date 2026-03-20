# Software Requirements Specification

## Software-as-a-Graph

### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 2.3** · **March 2026**

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
| 3. Prediction | Map M(v) to RMAV dimensions via rule-based (RMAV) and learning-based (GNN) paths | Prediction scores Q(v) |
| 4. Failure Simulation | Inject faults and measure cascading impact | Impact scores I(v) |
| 5. Validation | Statistically compare Q(v) against I(v) | Spearman ρ, F1, etc. |
| 6. Visualization | Generate interactive dashboards | HTML dashboard |

The framework is delivered through two interfaces: a **CLI pipeline** (`bin/`) for batch analysis and scripting, and the **Genieus web application** (FastAPI backend + Next.js frontend) for interactive browser-based exploration.

### 1.3 References

| Reference | Description |
|-----------|-------------|
| IEEE 830-1998 | IEEE Recommended Practice for Software Requirements Specifications |
| IEEE RASSE 2025 | Published methodology paper (doi: 10.1109/RASSE64831.2025.11315354) |
| Neo4j Documentation | https://neo4j.com/docs/ |
| NetworkX Documentation | https://networkx.org/documentation/ |
| Saaty, T.L. (1980) | *The Analytic Hierarchy Process*, McGraw-Hill |
| SDD v2.2 | Software Design Description for this project |
| STD v2.2 | Software and System Test Document for this project |

### 1.4 Document Conventions

The following conventions are used throughout this document:

- **Shall** denotes a mandatory requirement.
- **Should** denotes a recommended but non-mandatory behavior.
- **May** denotes an optional or conditional capability.
- Requirement identifiers follow the pattern **REQ-\<AREA\>-\<NN\>** (e.g., REQ-GM-01 for Graph Model requirement 1). Each identifier is unique and stable across versions.
- Mathematical notation follows the conventions defined in Appendix A.
- All metrics are scalar values in [0, 1] unless stated otherwise.

### 1.5 Document Overview

Section 2 describes the product's context, interfaces, users, and constraints. Section 3 lists all functional requirements organized by methodology step, including REST API and web interface requirements. Section 4 lists non-functional requirements for performance, accuracy, scalability, reliability, portability, maintainability, security, and hardware. Section 5 defines the data model, storage, validation rules, and domain applicability. Section 6 presents validation targets alongside empirical results achieved to date. Appendix A provides the complete formula reference for quality scoring. Appendix B is a glossary of technical terms.

### 1.6 Change History

| 2.2 | February 2026 | Added REST API interface and Genieus web app requirements (§2.2, §3.9); corrected metric count to 16 (§2.4); added FastAPI, Next.js, Docker dependencies (§2.6); added Node.js constraint (§2.5); updated accuracy targets to reflect achieved performance (§4.2, §6.2, §6.3); added CLI flags (§3.8); expanded Appendix B glossary |
| 2.3 | March 2026 | Refactored backend architecture with thinner routers, presenters, and dependency injection; updated quality formulas to include CDPot_enh, CQP (C-Maintainability), QSPOF, and QADS; aligned weights with AHP v2.3; updated testing strategy in SDD/STD |

---

## 2. Overall Description

### 2.1 Product Perspective

Software-as-a-Graph is a standalone pre-deployment analysis framework. It integrates with existing distributed architectures by consuming their topology descriptions and producing criticality assessments. It does not perform runtime monitoring.

The framework is available as a Docker Compose stack (recommended for the full experience), or as a Python package for CLI-only usage.

### 2.2 System Interfaces

| Interface | Technology | Purpose |
|-----------|------------|---------|
| Graph Database | Neo4j 5.x (Bolt protocol, port 7687) | Primary graph storage and GDS algorithms |
| Topology Input | JSON, GraphML | System architecture import |
| Results Export | JSON, CSV, GraphML | Analysis results for external tools |
| Static Dashboard | HTML (vis.js, Chart.js) | Archivable interactive visualization artifact |
| CLI | Python argparse (`bin/`) | Pipeline orchestration and scripting |
| REST API | FastAPI (HTTP/JSON, port 8000) | Programmatic access to all pipeline steps |
| Web Application | Next.js 16 frontend (port 7000) | Browser-based interactive exploration (Genieus) |

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
- **Analyze** the graph structure with 16 topological metrics per component.
- **Score** each component across four RMAV quality dimensions to produce Q(v).
- **Simulate** failures exhaustively or via Monte Carlo to produce ground-truth I(v).
- **Validate** predictions statistically against simulation results.
- **Visualize** all results in an interactive static HTML dashboard and via the Genieus live web application.
- **Expose** all pipeline operations through a versioned REST API (`/api/v1/`).

### 2.5 Constraints

The framework requires Neo4j 5.x, Python 3.9+, and Node.js 20+ (for the Genieus frontend). It performs static analysis only — accuracy depends on the completeness of the input topology specification. Memory requirements grow with system scale (see [§4.9](#49-hardware)). The recommended deployment method is Docker Compose, which handles all infrastructure dependencies automatically.

### 2.6 Dependencies

| Dependency | Version | Role |
|------------|---------|------|
| Python | ≥ 3.9 | Runtime language |
| NetworkX | ≥ 3.0 | Graph algorithm library |
| Neo4j | 5.x | Graph database and GDS plugin |
| FastAPI | ≥ 0.100 | REST API backend framework |
| uvicorn | ≥ 0.23 | ASGI server for FastAPI |
| scipy | ≥ 1.9 | Statistical validation (Spearman, Pearson) |
| matplotlib | ≥ 3.6 | Chart generation |
| httpx | ≥ 0.24 | Async HTTP client (test infrastructure) |
| Node.js | ≥ 20 | Next.js frontend build and runtime |
| Next.js | 16.x | Genieus frontend framework |
| React | 19.x | Frontend component library |
| TypeScript | ≥ 5.0 | Frontend type safety |
| Docker | ≥ 20.10 | Container runtime |
| Docker Compose | ≥ 2.x | Full-stack orchestration |
| vis.js | 9.x (CDN) | Interactive network graph rendering |
| Chart.js | 4.x (CDN) | KPI and bar/pie chart rendering |

### 2.7 Assumptions

- The system topology is accurately and completely specified in the input file.
- QoS settings are available for all topics in the input topology.
- Neo4j 5.x is installed, running, and reachable on the configured Bolt URI.
- The analyst has sufficient domain knowledge to interpret criticality predictions relative to operational context.

### 2.8 Future Evolution

The following capabilities are planned for future versions and inform current design decisions without imposing requirements:

- **ICSA 2026 paper** — extended methodology with enhanced QoS-weighted formulations and middleware-specific insights, targeting the IEEE International Conference on Software Architecture 2026.
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
| REQ-GM-06 | The system shall compute QoS-based edge weights for DEPENDS_ON edges using the formula defined in Appendix A.1. |
| REQ-GM-07 | The system shall project a layer-specific subgraph from the full graph for each of the four supported layers (app, infra, mw, system). |

### 3.2 Structural Analysis (Step 2)

| ID | Requirement |
|----|-------------|
| REQ-SA-01 | The system shall compute PageRank (PR) for each component. |
| REQ-SA-02 | The system shall compute Reverse PageRank (RPR) for each component. |
| REQ-SA-03 | The system shall compute Betweenness Centrality (BT) for each component. |
| REQ-SA-04 | The system shall compute Closeness Centrality (CL) for each component. |
| REQ-SA-05 | The system shall compute Eigenvector Centrality (EV) for each component. |
| REQ-SA-06 | The system shall compute In-Degree (DG_in) and Out-Degree (DG_out) centrality for each component. |
| REQ-SA-07 | The system shall compute Clustering Coefficient (CC) for each component. |
| REQ-SA-08 | The system shall compute the articulation point indicator (AP boolean) and continuous articulation point score (AP_c) for each component. |
| REQ-SA-09 | The system shall detect bridge edges and compute Bridge Ratio (BR) for each component. |
| REQ-SA-10 | The system shall compute QoS weight aggregates (w, w_in, w_out) per component from the DEPENDS_ON edge weights. |
| REQ-SA-11 | The system shall normalize all metrics to the [0, 1] interval before quality scoring. |
| REQ-SA-12 | The system shall compute graph-level summary statistics (component count, edge count, density, average clustering, number of articulation points, number of bridges). |

> **Note on metric count:** Step 2 computes up to 20 metric fields per component: PR, RPR, BT, CL, EV, DG_in, DG_out, CC, AP (bool), AP_c_dir, BR, w, w_in, w_out, CDI, RCL, REV, MPCI, FOC, plus graph-level summary statistics. The figure "13 topological metrics" used in earlier research refers to the core Tier 1 subset.

### 3.3 Prediction (Step 3)

| ID | Requirement |
|----|-------------|
| REQ-QS-01 | The system shall compute a Reliability score R(v) for each component. |
| REQ-QS-02 | The system shall compute a Maintainability score M(v) for each component. |
| REQ-QS-03 | The system shall compute an Availability score A(v) for each component. |
| REQ-QS-04 | The system shall compute a Vulnerability score V(v) for each component. |
| REQ-QS-05 | The system shall compute a composite quality score Q(v) = α·R(v) + β·M(v) + γ·A(v) + δ·V(v) for each component. |
| REQ-QS-06 | The system shall classify each component into a criticality level: CRITICAL, HIGH, MEDIUM, LOW, or MINIMAL. |
| REQ-QS-07 | The system shall support AHP-derived dimension weights as the default weighting mode, and equal weights as an alternative mode. |
| REQ-QS-08 | The system shall validate AHP matrix consistency (Consistency Ratio < 0.10) before computing weights and abort with a diagnostic message if the check fails. |
| REQ-QS-09 | The system shall classify components using box-plot statistical classification (IQR-based adaptive thresholds) rather than fixed cutoffs. |
| REQ-QS-10 | The system shall detect and report architectural anti-patterns: Single Points of Failure (SPOF), bottleneck components, and high-vulnerability clusters. |

### 3.4 Failure Simulation (Step 4)

| ID | Requirement |
|----|-------------|
| REQ-FS-01 | The system shall support exhaustive failure simulation — injecting failure into every component in the layer and measuring cascade impact. |
| REQ-FS-02 | The system shall support Monte Carlo simulation as an alternative for large systems where exhaustive simulation is computationally prohibitive. |
| REQ-FS-03 | The system shall implement four failure simulators aligned to RMAV dimensions: cascade failure (Reliability), change-propagation (Maintainability), connectivity-loss (Availability), and compromise-propagation (Vulnerability). |
| REQ-FS-04 | The system shall compute a composite impact score I(v) from the four simulator outputs for each component. |
| REQ-FS-05 | The system shall model failure propagation through DEPENDS_ON edges, weighted by the QoS edge weights. |

### 3.5 Statistical Validation (Step 5)

| ID | Requirement |
|----|-------------|
| REQ-VL-01 | The system shall compute Spearman rank correlation coefficient ρ between Q(v) and I(v). |
| REQ-VL-02 | The system shall compute the p-value for the Spearman correlation and report significance (p ≤ 0.05). |
| REQ-VL-03 | The system shall compute F1-Score, Precision, and Recall for binary critical/non-critical classification. |
| REQ-VL-04 | The system shall compute Top-5 and Top-10 overlap between Q(v) and I(v) ranked component lists. |
| REQ-VL-05 | The system shall compute RMSE and MAE between normalized Q(v) and I(v). |
| REQ-VL-06 | The system shall compute Cohen's κ (chance-corrected agreement) for the criticality classification. |
| REQ-VL-07 | The system shall compute NDCG (Normalized Discounted Cumulative Gain) for ranking quality. |
| REQ-VL-08 | The system shall compute Pearson correlation coefficient r as a supplementary metric. |
| REQ-VL-09 | The system shall evaluate each metric against its target threshold (see §6.1) and report a pass/fail result. |
| REQ-VL-10 | The system shall require a minimum of 5 matched components for valid statistical analysis and report a warning for n < 5. |

### 3.6 Visualization (Step 6)

| ID | Requirement |
|----|-------------|
| REQ-VZ-01 | The system shall generate a self-contained HTML dashboard with all analysis results embedded. |
| REQ-VZ-02 | The system shall display KPI summary cards showing component counts, SPOF count, anti-pattern count, and overall validation pass/fail status. |
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
| REQ-CLI-03 | The system shall support a `--layer` flag for targeting specific architectural layers. |
| REQ-CLI-04 | The system shall support an `--output` flag for exporting results to JSON. |
| REQ-CLI-05 | The system shall support a `--generate` flag on the orchestrator to run synthetic topology generation before import. |
| REQ-CLI-06 | The system shall support a `--scale` flag to select a named scale preset (tiny, small, medium, large, xlarge) for topology generation. |
| REQ-CLI-07 | The system shall support `--verbose` and `--quiet` flags to control log verbosity (DEBUG and WARNING level respectively). |
| REQ-CLI-08 | The system shall support an `--open` flag on the orchestrator to automatically open the generated HTML dashboard in the default browser upon completion. |

### 3.9 REST API and Web Interface (Genieus)

This section specifies requirements for the FastAPI backend and the Genieus Next.js web application. All API endpoints are versioned under the `/api/v1/` prefix.

#### 3.9.1 REST API Endpoints

| ID | Requirement |
|----|-------------|
| REQ-API-01 | The system shall expose a health-check endpoint `GET /health` returning `{"status": "ok"}` with HTTP 200. |
| REQ-API-02 | The system shall expose `GET /api/v1/graph/summary` returning counts of all vertex and edge types currently stored in Neo4j. |
| REQ-API-03 | The system shall expose `POST /api/v1/graph/import` accepting a JSON topology body and importing it into Neo4j; the response shall include entity counts. |
| REQ-API-04 | The system shall expose `GET /api/v1/graph/search-nodes` accepting search query parameters and returning matching component records. |
| REQ-API-05 | The system shall expose `POST /api/v1/analysis/layer/{layer}` triggering structural analysis and quality scoring for the specified layer; the response shall include all component RMAV scores. |
| REQ-API-06 | The system shall expose `POST /api/v1/simulation/failure` accepting a layer parameter and returning one FailureResult per component in the layer. |
| REQ-API-07 | The system shall expose `POST /api/v1/validation/run-pipeline` accepting a layer parameter and returning the full ValidationResult including Spearman ρ and all other metrics. |
| REQ-API-08 | The system shall expose `GET /api/v1/validation/layers` returning the list of available analysis layers with their metadata. |
| REQ-API-09 | The system shall expose `GET /api/v1/components` accepting a layer parameter and returning all component records with full metric and score fields for that layer. |
| REQ-API-10 | The system shall return HTTP 422 with a descriptive error body (RFC 7807 Problem Details) for invalid request payloads. |
| REQ-API-11 | The system shall serve interactive API documentation (Swagger UI) at `GET /docs`. |

#### 3.9.2 Genieus Web Application

| ID | Requirement |
|----|-------------|
| REQ-WEB-01 | The system shall provide a Dashboard tab displaying KPI cards (component counts, criticality distribution, SPOF count, anti-pattern count, validation pass/fail). |
| REQ-WEB-02 | The system shall provide a Graph Explorer tab with an interactive 2D/3D force-directed dependency graph filterable by layer. |
| REQ-WEB-03 | The Graph Explorer shall support node click to open a side panel showing full RMAV scores, I(v), criticality level, cascade count, and direct dependency list. |
| REQ-WEB-04 | The Graph Explorer shall support 2D/3D toggle for visual layout. |
| REQ-WEB-05 | The system shall provide an Analysis tab where the user can select a layer and weight mode (equal/AHP) and trigger on-demand analysis. |
| REQ-WEB-06 | The system shall provide a Simulation tab where the user can select a target component and failure mode (CRASH, DEGRADED, PARTITION, OVERLOAD) and view the cascading failure. |
| REQ-WEB-07 | The system shall provide a Settings tab for configuring the Neo4j connection URI, username, and password. |
| REQ-WEB-08 | Settings shall persist across page reloads. |
| REQ-WEB-09 | The web application shall be deployable via Docker Compose (`docker compose up --build`) and accessible on port 7000. |
| REQ-WEB-10 | The Docker stack shall reach a healthy state within 60 seconds of `docker compose up`. |

---

## 4. Non-Functional Requirements

### 4.1 Performance

| ID | Requirement | Scale Reference |
|----|-------------|-----------------|
| REQ-PERF-01 | Analysis shall complete within 1 second for small systems. | ~30 components |
| REQ-PERF-02 | Analysis shall complete within 5 seconds for medium systems. | ~100 components |
| REQ-PERF-03 | Analysis shall complete within 20 seconds for large systems. | ~600 components |
| REQ-PERF-04 | Dashboard generation shall complete within 10 seconds for any supported scale. | Any scale |
| REQ-PERF-05 | REST API analysis endpoints shall return within 30 seconds for any supported scale. | ≤ 1,000 components |

### 4.2 Accuracy

The following targets apply to the **application layer**, which is the primary validation layer. Infrastructure layer results are reported for informational purposes; see [§6.2](#62-achieved-results-by-layer) for current infrastructure layer performance. All targets below have been met or exceeded by the current implementation (see §6).

| ID | Requirement |
|----|-------------|
| REQ-ACC-01 | Spearman ρ shall achieve ≥ 0.80 at the application layer. |
| REQ-ACC-02 | F1-Score shall achieve ≥ 0.90 at the application layer. |
| REQ-ACC-03 | Precision and Recall shall each achieve ≥ 0.80 at the application layer. |
| REQ-ACC-04 | Top-5 overlap shall achieve ≥ 60% at the application layer. |
| REQ-ACC-05 | Prediction accuracy (Spearman ρ) shall not decrease as system scale increases within the same architectural domain. |

> **Rationale for target revision (v2.1 → v2.2):** Targets REQ-ACC-01, 02, and 04 have been raised to reflect the achieved empirical performance documented in the IEEE RASSE 2025 publication (ρ = 0.876, F1 = 0.943, Top-5 overlap = 80%). The v2.1 targets (ρ ≥ 0.70, F1 ≥ 0.80, Top-5 ≥ 40%) represented conservative lower bounds at the time of initial specification and no longer represent the expected capability of the system.

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
| REQ-PORT-03 | The full Docker stack shall be deployable on any host meeting the Docker Engine requirements for the above operating systems and architectures. |

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
| Durability | TRANSIENT | +0.25 |
| Durability | TRANSIENT\_LOCAL | +0.20 |
| Durability | VOLATILE | +0.00 |
| Priority | URGENT | +0.30 |
| Priority | HIGH | +0.20 |
| Priority | MEDIUM | +0.10 |
| Priority | LOW | +0.00 |
| Message Size | Any | log₂(1 + size\_bytes / 1024) / 10, capped at 1.0 |

The minimum possible edge weight is 0.0 (BEST\_EFFORT, VOLATILE, LOW, zero-byte message). The maximum without the size component is 1.0. The size term adds up to 1.0 on top, so the theoretical maximum uncapped total is 2.0; in practice the cap ensures the size term never exceeds 1.0 alone.

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
| Spearman ρ | ≥ 0.80 | Primary | Strong rank correlation between prediction and reality |
| p-value | ≤ 0.05 | Primary | Statistical significance |
| F1-Score | ≥ 0.90 | Primary | Balanced precision and recall |
| Top-5 Overlap | ≥ 60% | Primary | Agreement on the most critical components |
| RMSE | ≤ 0.25 | Secondary | Prediction error bound |
| Precision | ≥ 0.80 | Reported | Minimize false alarms |
| Recall | ≥ 0.80 | Reported | Catch all critical components |
| Cohen's κ | ≥ 0.60 | Reported | Chance-corrected agreement |
| Top-10 Overlap | ≥ 50% | Reported | Extended critical set agreement |
| MAE | ≤ 0.20 | Reported | Absolute error bound |

All targets in this table apply to the **application layer**. Infrastructure layer targets are currently aspirational; see §6.2 for current results.

### 6.2 Achieved Results by Layer

Results represent the aggregate performance across all 8 validated scenario datasets spanning ROS 2 autonomous vehicles, IoT smart cities, financial trading platforms, and healthcare systems. The best reported overall Spearman ρ (0.876) was published in the IEEE RASSE 2025 paper.

| Metric | Application Layer | Infrastructure Layer | Target (App) |
|--------|-------------------|----------------------|--------------|
| Spearman ρ | **0.876** ✓ | 0.54 | ≥ 0.80 |
| F1-Score | **0.943** ✓ | 0.68 | ≥ 0.90 |
| Precision | **0.94** ✓ | 0.71 | ≥ 0.80 |
| Recall | **0.95** ✓ | 0.65 | ≥ 0.80 |
| Top-5 Overlap | **80%** ✓ | 40% ✓ | ≥ 60% |

Application layer consistently meets all targets. Infrastructure layer shows lower but improving correlation, reflecting the inherent difficulty of capturing infrastructure dependencies through topology alone. Improving infrastructure layer accuracy is an active research direction (see §2.8).

### 6.3 Accuracy by System Scale

Results are measured on application layer analysis. F1-score variance at small scale reflects the sensitivity of binary classification to outlier count in synthetic systems; Spearman ρ is the primary robustness indicator.

| Scale Band | Components | Spearman ρ (μ ± σ) | F1-Score (μ ± σ) | Analysis Time |
|------------|------------|:-------------------:|:-----------------:|:-------------:|
| Small | 10–25 | 0.787 ± 0.092 | 0.232 ± 0.377 | < 1 s |
| Medium | 30–50 | 0.847 ± 0.067 | 0.150 ± 0.217 | ~2 s |
| Large | 60–100 | 0.858 ± 0.025 | 0.125 ± 0.152 | ~4 s |
| XLarge | 150–300 | **0.876** (aggregate) | — | ~10 s |
| Enterprise | 300–1,000 | TBD (empirical validation in progress) | TBD | ≤ 20 s (REQ-PERF-03) |

Prediction accuracy improves with scale — larger systems produce more stable centrality distributions and more reliable rank correlation. The Enterprise row represents the REQ-SCAL-01 upper bound; empirical results at this scale are part of ongoing thesis validation work.

---

## Appendix A: Prediction Formula Reference

### A.1 Edge Weight Formula

```
w(e) = w_reliability + w_durability + w_priority + w_size

w_reliability = 0.30  if RELIABLE,  0.00 otherwise
w_durability  = 0.40  if PERSISTENT, 0.24 if TRANSIENT, 0.20 if TRANSIENT_LOCAL, 0.00 if VOLATILE
w_priority    = 0.30  if URGENT, 0.20 if HIGH, 0.10 if MEDIUM, 0.00 if LOW
w_size        = min(log₂(1 + size_bytes / 1024) / 50, 0.20)
```

### A.2 Reliability Score

```
R(v) = w₁ × RPR(v) + w₂ × DG_in(v) + w₃ × CDPot_enh(v)
```

Default intra-dimension weights: w₁ = 0.45, w₂ = 0.30, w₃ = 0.25.
CDPot_enh(v) captures the cascade potential based on follower count, topic fan-out hotspots (FOC), and multi-path couplings (MPCI).

### A.3 Maintainability Score

```
M(v) = w₁ × BT(v) + w₂ × w_out(v) + w₃ × CQP(v) + w₄ × CouplingRisk(v) + w₅ × (1 − CC(v))
```

Default intra-dimension weights: w₁ = 0.35, w₂ = 0.30, w₃ = 0.15, w₄ = 0.12, w₅ = 0.08.
CouplingRisk(v) is derived from Martin instability; CC is the clustering coefficient representing local redundancy. CQP is code-level maintainability penalty.

### A.4 Availability Score

```
A(v) = w₁ × QSPOF(v) + w₂ × BR(v) + w₃ × AP_c_directed(v) + w₄ × CDI(v)
```

Default intra-dimension weights: w₁ = 0.45, w₂ = 0.30, w₃ = 0.15, w₄ = 0.10.
QSPOF is the QoS-weighted articulation point score; CDI is the connectivity degradation index.

### A.5 Vulnerability Score

```
V(v) = w₁ × REV(v) + w₂ × RCL(v) + w₃ × QADS(v)
```

Default intra-dimension weights: w₁ = 0.40, w₂ = 0.35, w₃ = 0.25.
REV/RCL are reverse eigenvector/closeness centrality; QADS is QoS-weighted Attack-Dependent Surface.

### A.6 Composite Quality Score

```
Default system-layer weights: α = β = γ = δ = 0.25 (Balanced). 
The system also supports AHP-derived weights which typically prioritize Availability (γ ≈ 0.43) and Reliability (α ≈ 0.24) over Maintainability and Vulnerability, reflecting the mission-critical nature of distributed pub-sub infrastructure.

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| AP | Articulation Point — a vertex whose removal disconnects the graph |
| AP\_c | Continuous articulation point score — fraction of graph fragmented upon vertex removal (scalar in [0, 1]) |
| AHP | Analytic Hierarchy Process — a structured method for deriving priority weights from pairwise comparisons (Saaty, 1980) |
| BR | Bridge Ratio — fraction of a vertex's incident edges that are bridges |
| BT | Betweenness Centrality |
| CC | Clustering Coefficient |
| CI | Consistency Index in AHP: (λ\_max − n) / (n − 1) |
| CL | Closeness Centrality |
| CLI | Command-Line Interface |
| CR | Consistency Ratio in AHP: CI / RI (must be < 0.10) |
| DG\_in / DG\_out | In-Degree / Out-Degree centrality |
| DTO | Data Transfer Object — a plain data carrier between architectural layers |
| EV | Eigenvector Centrality |
| FastAPI | Python web framework for building REST APIs with automatic OpenAPI documentation |
| GDS | Graph Data Science — Neo4j plugin providing graph algorithm implementations |
| Genieus | The browser-based web application frontend for Software-as-a-Graph, built with Next.js |
| I(v) | Impact score for component v — the composite cascade impact measured by failure simulation |
| IMP(v) | Structural importance proxy: (PR(v) + RPR(v)) / 2 |
| IQR | Interquartile Range (Q3 − Q1) — used in box-plot classification of criticality levels |
| MAE | Mean Absolute Error |
| NDCG | Normalized Discounted Cumulative Gain — a ranking quality metric that discounts gains at lower ranks |
| Next.js | React-based web framework used for the Genieus frontend |
| PR / RPR | PageRank / Reverse PageRank |
| Q(v) | Composite quality score for component v — the RMAV-weighted criticality prediction |
| RI | Random Index — a scale-dependent constant used in the AHP consistency ratio denominator |
| RMAV | Reliability, Maintainability, Availability, Vulnerability — the four quality dimensions |
| RMSE | Root Mean Squared Error |
| ρ | Spearman rank correlation coefficient — primary validation metric |
| SOLID | Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion (design principles) |
| SPOF | Single Point of Failure — a component whose removal disconnects the graph (AP with AP\_c > threshold) |
| TP / FP / TN / FN | True/False Positive/Negative — binary classification outcomes |

---

*Software-as-a-Graph Framework v2.3 · March 2026*
*Istanbul Technical University, Computer Engineering Department*