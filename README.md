# Software-as-a-Graph

## Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems

A comprehensive framework for modeling distributed publish-subscribe systems as multi-layer graphs and analyzing their structural vulnerabilities using graph theory and centrality metrics.

**Author**: Ibrahim Onuralp Yigit  
**Research**: PhD Thesis - Graph-Based Modeling and Analysis of Distributed Pub-Sub Systems  
**Publication**: IEEE RASSE 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Research Motivation](#research-motivation)
3. [Multi-Layer Graph Model](#multi-layer-graph-model)
4. [Methodology](#methodology)
5. [Quick Start](#quick-start)
6. [Installation](#installation)
7. [Pipeline Steps](#pipeline-steps)
8. [Validation Approach](#validation-approach)
9. [Project Structure](#project-structure)
10. [API Reference](#api-reference)
11. [Publications](#publications)

---

## Overview

This project provides a **six-step methodology** for identifying critical components in complex distributed publish-subscribe systems using graph-based analysis:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. MODEL   │───▶│  2. ANALYZE │───▶│ 3. SIMULATE │
│  Build Graph│    │  Centrality │    │   Failures  │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌─────────────┐    ┌───────▼─────┐
│ 6. DEPLOY   │◀───│ 5. VISUALIZE│◀───│ 4. VALIDATE │
│ Digital Twin│    │  Dashboard  │    │ Correlation │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Key Features

- **Multi-Layer Graph Modeling**: Represents systems across Application, Topic, Broker, and Infrastructure layers
- **Composite Criticality Scoring**: Combines betweenness centrality, articulation points, PageRank, and degree metrics
- **Failure Simulation**: Measures actual impact through exhaustive failure campaigns with cascade propagation
- **Statistical Validation**: Validates predictions using Spearman correlation (≥0.70) and F1-score (≥0.90)
- **Interactive Visualization**: Generates multi-layer dashboards with Chart.js and vis.js

### Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Spearman ρ | ≥ 0.70 | Rank correlation between predicted and actual criticality |
| F1-Score | ≥ 0.90 | Classification accuracy for critical component identification |
| Precision | ≥ 0.80 | Fraction of predicted critical that are actually critical |
| Recall | ≥ 0.80 | Fraction of actual critical components correctly identified |
| Top-5 Overlap | ≥ 60% | Agreement on most critical components |

---

## Research Motivation

### The Challenge

Modern distributed systems (autonomous vehicles, IoT deployments, financial trading platforms) rely on publish-subscribe messaging for communication. These systems present unique challenges:

1. **Complex Dependencies**: Hundreds of applications, topics, and brokers with intricate relationships
2. **Hidden Bottlenecks**: Critical components not obvious from architecture diagrams
3. **Cascade Failures**: Single component failures can propagate through the system
4. **Qualitative Assessment**: Traditional approaches rely on expert judgment

### Our Solution

We model these systems as **directed multi-layer graphs** where:
- **Nodes** represent system components (applications, topics, brokers, infrastructure)
- **Edges** represent dependencies (publishes, subscribes, runs-on, connects-to)
- **Weights** capture dependency strength (message frequency, QoS requirements)

Graph algorithms then **quantify** what was previously qualitative:
- *"This broker seems important"* → **Betweenness Centrality: 0.847**
- *"Failure here would be bad"* → **Impact Score: 0.92**
- *"We should add redundancy"* → **Articulation Point: Yes**

---

## Multi-Layer Graph Model

### Layer Architecture

Our model represents pub-sub systems as four interconnected layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Sensor  │  │  Data   │  │Dashboard│  │ Alert   │  │ Logger  │        │
│  │   App   │  │Processor│  │   App   │  │ Service │  │   App   │        │
│  └────┬────┘  └────┬────┘  └────▲────┘  └────▲────┘  └────▲────┘        │
│       │            │            │            │            │             │
│       │ publishes  │ pub/sub    │ subscribes │            │             │
├───────▼────────────▼────────────┴────────────┴────────────┴─────────────┤
│                          TOPIC LAYER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ /sensors/   │  │ /processed/ │  │  /alerts/   │  │   /logs/    │     │
│  │ temperature │  │    data     │  │  critical   │  │   system    │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │                │            │
│         │ routes         │                │                │            │
├─────────▼────────────────▼────────────────▼────────────────▼────────────┤
│                          BROKER LAYER                                   │
│         ┌────────────────────┐       ┌────────────────────┐             │
│         │     Broker 0       │       │     Broker 1       │             │
│         │  (Primary, QoS-2)  │       │  (Secondary, QoS-1)│             │
│         └─────────┬──────────┘       └─────────┬──────────┘             │
│                   │ runs_on                    │ runs_on                │
├───────────────────▼────────────────────────────▼────────────────────────┤
│                      INFRASTRUCTURE LAYER                               │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│    │  Node 0  │───▶│  Node 1  │───▶│  Node 2  │───▶│  Node 3  │         │
│    │ (Master) │    │ (Worker) │    │ (Worker) │    │ (Backup) │         │
│    └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Node Types

| Type | Description | Example Properties |
|------|-------------|-------------------|
| **Application** | Software components that publish/subscribe to messages | criticality, qos_requirements |
| **Topic** | Message channels for pub/sub communication | message_rate, qos_level |
| **Broker** | Message routing infrastructure | max_connections, throughput |
| **Node** | Physical/virtual infrastructure | cpu, memory, availability_zone |

### Edge Types (Dependencies)

| Type | From | To | Description |
|------|------|-----|-------------|
| `publishes` | Application | Topic | App publishes messages to topic |
| `subscribes` | Application | Topic | App subscribes to topic |
| `app_to_app` | Application | Application | Direct application dependency |
| `app_to_broker` | Application | Broker | App connected to broker |
| `node_to_broker` | Broker | Node | Broker runs on infrastructure |
| `node_to_node` | Node | Node | Infrastructure dependency |

### Automatic DEPENDS_ON Derivation

Cross-layer dependencies are automatically derived during analysis:

```
If App_A publishes to Topic_T AND App_B subscribes to Topic_T
Then: App_A → DEPENDS_ON → App_B (weight based on QoS)
```

---

## Methodology

### Step 1: Graph Model Construction

Transform system architecture into a directed weighted multi-layer graph:

```python
# Input: System configuration (YAML, JSON, or live discovery)
# Output: G = (V, E) where V = nodes, E = weighted edges

graph = build_graph(system_config)
# Nodes: Applications, Topics, Brokers, Infrastructure
# Edges: Dependencies with weights based on QoS, frequency, criticality
```

### Step 2: Structural Analysis

Calculate criticality using multiple graph algorithms:

```python
# Betweenness Centrality - Information flow bottlenecks
bc = nx.betweenness_centrality(G, weight='weight')

# Articulation Points - Single points of failure
aps = nx.articulation_points(G.to_undirected())

# PageRank - Overall importance via incoming dependencies
pr = nx.pagerank(G, weight='weight')

# Degree Centrality - Direct connectivity
dc = nx.degree_centrality(G)
```

### Step 3: Composite Criticality Scoring

Combine metrics into a single criticality score:

```
C_score(v) = α·BC_norm(v) + β·AP(v) + γ·I(v) + δ·DC_norm(v) + ε·PR_norm(v)
```

Where:
- **BC_norm** = Normalized betweenness centrality
- **AP** = 1 if articulation point, 0 otherwise
- **I** = Downstream impact score (reachability-based)
- **DC_norm** = Normalized degree centrality
- **PR_norm** = Normalized PageRank

**Default Weights**:

| Parameter | Weight | Rationale |
|-----------|--------|-----------|
| α (BC) | 0.25 | Information flow bottlenecks |
| β (AP) | 0.30 | Single points of failure (highest weight) |
| γ (I) | 0.25 | Downstream cascade impact |
| δ (DC) | 0.10 | Direct connectivity |
| ε (PR) | 0.10 | Overall system importance |

### Step 4: Failure Simulation

Validate predictions through exhaustive failure simulation:

```python
# For each component:
#   1. Remove component from graph
#   2. Calculate reachability loss
#   3. Simulate cascade propagation
#   4. Measure total system impact

for component in graph.nodes:
    result = simulate_failure(graph, component, enable_cascade=True)
    actual_impact[component] = result.impact_score
```

**Impact Score Formula**:
```
Impact = 0.5 × (reachability_loss%) + 0.3 × (fragmentation) + 0.2 × (cascade_extent)
```

### Step 5: Statistical Validation

Compare predicted criticality against actual impact:

```python
# Correlation Analysis
spearman_rho = spearman_correlation(predicted_scores, actual_impacts)
# Target: ρ ≥ 0.70

# Classification Metrics (using 80th percentile threshold)
precision, recall, f1 = calculate_classification_metrics(
    predicted_critical, actual_critical
)
# Targets: F1 ≥ 0.90, Precision ≥ 0.80, Recall ≥ 0.80

# Ranking Analysis
top_k_overlap = calculate_top_k_overlap(predicted_ranking, actual_ranking, k=5)
# Target: Top-5 ≥ 60%
```

### Step 6: Visualization & Reporting

Generate interactive dashboards and multi-layer architecture views:

- **Dashboard**: Metrics, charts, sortable component tables
- **Multi-Layer View**: Vertical layer separation with dependency lines
- **Criticality Heatmap**: Color-coded by criticality level

---

## Quick Start

### Option 1: End-to-End Demo Script

```bash
# Clone repository
git clone https://github.com/onuralpyigit/software-as-a-graph.git
cd software-as-a-graph

# Make script executable
chmod +x run.sh

# Run full demo (requires Neo4j)
./run.sh

# Or quick demo without Neo4j
./run.sh --skip-neo4j --quick

# View results
open demo_output/dashboard.html
```

### Option 2: Step-by-Step Python

```python
from src.analysis import GDSClient
from src.analysis.criticality_classifier import GDSCriticalityClassifier
from src.simulation import Neo4jGraphLoader, FailureSimulator
from src.validation import GraphValidator
from src.visualization import Neo4jVisualizer

# 1. Connect to Neo4j
gds = GDSClient(uri="bolt://localhost:7687", user="neo4j", password="password")

# 2. Analyze criticality
projection = gds.create_depends_on_projection("analysis")
classifier = GDSCriticalityClassifier(gds)
analysis = classifier.classify_by_composite_score("analysis")

# 3. Simulate failures
loader = Neo4jGraphLoader(uri="bolt://localhost:7687", user="neo4j", password="password")
graph = loader.load_graph()
simulator = FailureSimulator()
simulation = simulator.simulate_all_single_failures(graph)

# 4. Validate
predicted = {item.item_id: item.score for item in analysis.items}
actual = {r.simulation_id.split('_fail_')[0]: r.impact.impact_score 
          for r in simulation.results}

validator = GraphValidator()
result = validator.validate(predicted, actual)
print(f"Spearman: {result.correlation.spearman_coefficient:.3f}")
print(f"F1-Score: {result.classification.f1_score:.3f}")

# 5. Visualize
visualizer = Neo4jVisualizer(uri="bolt://localhost:7687", user="neo4j", password="password")
html = visualizer.generate_dashboard(validation_results=result.to_dict())
with open('dashboard.html', 'w') as f:
    f.write(html)

# Cleanup
gds.close()
loader.close()
visualizer.close()
```

---

## Installation

### Prerequisites

- Python 3.10+
- Neo4j 5.x with Graph Data Science (GDS) plugin (optional but recommended)

### Install Dependencies

```bash
# Core dependencies
pip install networkx

# Neo4j integration (recommended)
pip install neo4j

# All dependencies
pip install networkx neo4j
```

### Neo4j Setup (Optional)

```bash
# Using Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["graph-data-science"]' \
  neo4j:5.15.0-enterprise

# Wait for startup
sleep 30

# Verify connection
python -c "from neo4j import GraphDatabase; GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')).verify_connectivity()"
```

---

## Pipeline Steps

### Step 1: Generate Graph

Create a realistic pub-sub system topology:

```bash
# Using run.sh (generates and loads into Neo4j)
./run.sh --scenario iot --scale medium
```

Supported scenarios:
- `iot`: IoT sensor network (temperature, humidity, alerts)
- `financial`: Financial trading platform (market data, orders, risk)
- `microservices`: Cloud microservices architecture
- `ros2`: ROS 2 robotic system with DDS

Supported scales:
- `small`: 10 apps, 2 brokers, 4 nodes
- `medium`: 30 apps, 4 brokers, 8 nodes
- `large`: 100 apps, 8 brokers, 20 nodes

### Step 2: Analyze Graph

Calculate criticality scores using GDS algorithms:

```bash
python analyze_graph.py \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password password \
  --method composite \
  --output results/
```

Analysis methods:
- `composite`: Combined BC + AP + PR + DC (recommended)
- `betweenness`: Betweenness centrality only
- `pagerank`: PageRank only
- `degree`: Degree centrality only

### Step 3: Simulate Failures

Run exhaustive failure campaign:

```bash
python simulate_graph.py \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password password \
  --campaign \
  --cascade \
  --output results/
```

Simulation modes:
- `--component ID`: Single component failure
- `--campaign`: All components (exhaustive)
- `--attack --strategy highest_betweenness`: Targeted attack

### Step 4: Validate Results

Compare predictions with actuals:

```bash
python validate_graph.py \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password password \
  --method composite \
  --output results/
```

Options:
- `--compare`: Compare all analysis methods
- `--bootstrap`: Calculate confidence intervals
- `--spearman-target 0.75`: Custom target

### Step 5: Visualize

Generate interactive dashboards:

```bash
python visualize_graph.py \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password password \
  --dashboard \
  --output dashboard.html
```

Visualization modes:
- `--dashboard`: Full dashboard with charts
- `--multi-layer`: Layer-separated architecture view
- `--layer application`: Single layer view

---

## Validation Approach

### Why Validation Matters

Graph metrics (betweenness, PageRank) measure **structural** importance. But do they predict **actual** system impact? Validation answers this by:

1. **Predicting** criticality using topological metrics
2. **Measuring** actual impact through failure simulation
3. **Comparing** predictions vs actuals using statistical methods

### Metrics Explained

**Spearman Rank Correlation (ρ)**
- Measures if predicted ranking matches actual ranking
- ρ = 1: Perfect agreement
- ρ = 0: No correlation
- ρ = -1: Perfect disagreement
- **Target: ρ ≥ 0.70** (strong positive correlation)

**F1-Score**
- Harmonic mean of precision and recall
- Measures classification accuracy for critical vs non-critical
- **Target: F1 ≥ 0.90**

**Top-k Overlap**
- Fraction of predicted top-k that appear in actual top-k
- Critical for prioritizing remediation
- **Target: Top-5 ≥ 60%**

### Interpretation

| Spearman | F1 | Status | Interpretation |
|----------|-----|--------|----------------|
| ≥ 0.70 | ≥ 0.90 | ✅ PASSED | Predictions highly reliable |
| 0.50-0.70 | 0.70-0.90 | ⚠️ PARTIAL | Useful but verify critical cases |
| < 0.50 | < 0.70 | ❌ FAILED | Consider different methodology |

---

## Project Structure

```
software-as-a-graph/
├── run.sh                          # End-to-end demo script
├── README.md                       # This file
│
├── analyze_graph.py                # Analysis CLI
├── simulate_graph.py               # Simulation CLI
├── validate_graph.py               # Validation CLI
├── visualize_graph.py              # Visualization CLI
│
└── src/
    ├── __init__.py
    │
    ├── analysis/                   # Graph analysis module
    │   ├── __init__.py
    │   ├── gds_client.py           # Neo4j GDS integration
    │   ├── criticality_classifier.py  # Box-plot classification
    │   └── README.md
    │
    ├── simulation/                 # Failure simulation module
    │   ├── __init__.py
    │   ├── neo4j_loader.py         # Graph loading from Neo4j
    │   ├── failure_simulator.py    # Failure impact simulation
    │   ├── event_simulator.py      # Event-driven simulation
    │   └── README.md
    │
    ├── validation/                 # Validation module
    │   ├── __init__.py
    │   ├── graph_validator.py      # Statistical validation
    │   ├── integrated_validator.py # End-to-end pipeline
    │   └── README.md
    │
    └── visualization/              # Visualization module
        ├── __init__.py
        ├── graph_visualizer.py     # Multi-layer vis.js rendering
        ├── dashboard_generator.py  # Chart.js dashboards
        ├── neo4j_visualizer.py     # Neo4j integration
        └── README.md
```

---

## API Reference

### Analysis Module

```python
from src.analysis import GDSClient
from src.analysis.criticality_classifier import GDSCriticalityClassifier

# GDS Client
gds = GDSClient(uri, user, password, database)
projection = gds.create_depends_on_projection("name", include_weights=True)
bc_results = gds.betweenness_centrality("name", weighted=True)
pr_results = gds.pagerank("name", weighted=True)
gds.cleanup_projections()
gds.close()

# Criticality Classifier
classifier = GDSCriticalityClassifier(gds)
result = classifier.classify_by_composite_score("projection_name")
result = classifier.classify_by_betweenness("projection_name")
# result.items: List[ClassifiedItem] with score, level, metrics
```

### Simulation Module

```python
from src.simulation import Neo4jGraphLoader, FailureSimulator

# Load graph
loader = Neo4jGraphLoader(uri, user, password, database)
graph = loader.load_graph()  # Returns SimulationGraph
loader.close()

# Simulate failures
simulator = FailureSimulator(cascade_threshold=0.5, cascade_probability=0.7)
result = simulator.simulate_single_failure(graph, "component_id")
batch = simulator.simulate_all_single_failures(graph, enable_cascade=True)
# result.impact.impact_score, batch.critical_components
```

### Validation Module

```python
from src.validation import GraphValidator, IntegratedValidator

# Manual validation
validator = GraphValidator(targets=ValidationTargets(spearman_correlation=0.70))
result = validator.validate(predicted_scores, actual_impacts)
result = validator.validate_with_bootstrap(predicted, actual, n_iterations=1000)
# result.correlation.spearman_coefficient, result.classification.f1_score

# Integrated pipeline
integrated = IntegratedValidator(uri, user, password)
result = integrated.run_validation(analysis_method='composite', enable_cascade=True)
result.print_summary()
```

### Visualization Module

```python
from src.visualization import Neo4jVisualizer, DashboardGenerator

# Neo4j-based visualization
viz = Neo4jVisualizer(uri, user, password)
html = viz.visualize_html()
html = viz.visualize_multi_layer()
html = viz.generate_dashboard(validation_results=result.to_dict())
viz.close()

# Manual dashboard
from src.visualization import DashboardGenerator
generator = DashboardGenerator()
html = generator.generate(nodes, edges, criticality, validation_results)
```

---

## Publications

### Primary Publication

**Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems**  
Ibrahim Onuralp Yigit  
IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE) 2025

### Cite This Work

```bibtex
@inproceedings{yigit2025graph,
  title={Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems},
  author={Yigit, Ibrahim Onuralp},
  booktitle={IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE)},
  year={2025},
  organization={IEEE}
}
```

---

## License

This project is developed as part of PhD research at Istanbul Technical University. See LICENSE for details.

---

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

---

## Acknowledgments

- Neo4j and the Graph Data Science team
- NetworkX developers
- vis.js and Chart.js communities