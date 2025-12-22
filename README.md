# Software-as-a-Graph

## Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![IEEE RASSE 2025](https://img.shields.io/badge/IEEE-RASSE%202025-green.svg)](https://rasse2025.ieee.org/)

A comprehensive framework for modeling distributed publish-subscribe systems as graphs and analyzing them to identify critical components, single points of failure, and architectural anti-patterns.

**Author:** Ibrahim Onuralp Yigit  
**Publication:** IEEE RASSE 2025 - *Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems*

---

## Table of Contents

1. [Overview](#overview)
2. [Research Methodology](#research-methodology)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Pipeline Steps](#pipeline-steps)
   - [Step 1: Graph Generation](#step-1-graph-generation)
   - [Step 2: Graph Analysis](#step-2-graph-analysis)
   - [Step 3: Failure Simulation](#step-3-failure-simulation)
   - [Step 4: Validation](#step-4-validation)
   - [Step 5: Visualization](#step-5-visualization)
6. [Multi-Layer Graph Model](#multi-layer-graph-model)
7. [Criticality Scoring](#criticality-scoring)
8. [Validation Metrics](#validation-metrics)
9. [Project Structure](#project-structure)
10. [API Reference](#api-reference)
11. [Examples](#examples)
12. [Contributing](#contributing)
13. [License](#license)

---

## Overview

Modern distributed systems built on publish-subscribe (pub-sub) patterns—such as IoT platforms, financial trading systems, autonomous vehicles (ROS 2), and microservices architectures—are complex networks of interconnected components. Understanding which components are critical to system reliability is essential for:

- **Reliability Engineering**: Identifying single points of failure (SPOFs)
- **Risk Assessment**: Prioritizing components for redundancy and monitoring
- **Architecture Optimization**: Detecting anti-patterns and bottlenecks
- **Maintenance Planning**: Focusing testing and maintenance efforts

This framework models pub-sub systems as **multi-layer directed graphs** and applies **graph algorithms** to systematically identify critical components.

### Key Features

- **Multi-layer graph modeling** (Infrastructure → Broker → Topic → Application)
- **Composite criticality scoring** using multiple graph metrics
- **Failure simulation** with cascading effects
- **Statistical validation** (Spearman ρ ≥ 0.70, F1 ≥ 0.90)
- **Anti-pattern detection** (God Topics, SPOFs, circular dependencies)
- **Interactive visualizations** with vis.js and dashboards

---

## Research Methodology

Our six-step methodology provides a rigorous approach to identifying critical components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GRAPH-BASED ANALYSIS METHODOLOGY                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│   │   STEP 1    │    │   STEP 2    │    │   STEP 3    │                    │
│   │  GENERATE   │───▶│   ANALYZE   │───▶│  SIMULATE   │                    │
│   │ Graph Model │    │ Criticality │    │  Failures   │                    │
│   └─────────────┘    └─────────────┘    └──────┬──────┘                    │
│                                                │                            │
│   ┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐                    │
│   │   STEP 6    │    │   STEP 5    │    │   STEP 4    │                    │
│   │   DEPLOY    │◀───│  VISUALIZE  │◀───│  VALIDATE   │                    │
│   │ Digital Twin│    │   Results   │    │   Results   │                    │
│   └─────────────┘    └─────────────┘    └─────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Step | Purpose | Output |
|------|---------|--------|
| 1. Generate | Create graph model from system topology | JSON graph file |
| 2. Analyze | Calculate criticality using graph algorithms | Criticality scores |
| 3. Simulate | Run failure scenarios to measure actual impact | Impact scores |
| 4. Validate | Compare predictions with simulation results | Correlation metrics |
| 5. Visualize | Generate interactive multi-layer visualizations | HTML dashboards |
| 6. Deploy | Implement as digital twin (future work) | Real-time monitoring |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/onuralpyigit/software-as-a-graph.git
cd software-as-a-graph

# Install required packages
pip install networkx scipy matplotlib

# Optional: Install Neo4j driver for database storage
pip install neo4j
```

### Verify Installation

```bash
# Check available scenarios
python generate_graph.py --list-scenarios

# Check available scales
python generate_graph.py --list-scales
```

---

## Quick Start

### Option 1: Run the Complete Demo

```bash
# Make the script executable
chmod +x run.sh

# Run the full end-to-end demo
./run.sh

# Quick demo (small scale for fast testing)
./run.sh --quick

# Specific scenario
./run.sh --scenario financial --scale large
```

### Option 2: Step-by-Step Commands

```bash
# Step 1: Generate graph
python generate_graph.py --scenario iot --scale medium --output system.json

# Step 2: Analyze graph
python analyze_graph.py --input system.json --output results/ --format json html --full

# Step 3: Simulate failures
python simulate_graph.py --input system.json --campaign --export-json simulation.json

# Step 4: Validate results
python validate_graph.py --input system.json --output results/ --format json html

# Step 5: Visualize
python visualize_graph.py --input system.json --output dashboard.html --dashboard
```

### Option 3: Python API

```python
import networkx as nx
from src.analysis import GraphAnalyzer
from src.simulation import FailureSimulator
from src.validation import GraphValidator
from src.visualization import GraphVisualizer

# Load or create graph
graph = nx.DiGraph()
# ... add nodes and edges ...

# Analyze
analyzer = GraphAnalyzer()
criticality = analyzer.analyze(graph)

# Simulate
simulator = FailureSimulator()
impacts = simulator.simulate_all_single_failures(graph)

# Validate
validator = GraphValidator()
result = validator.validate(graph, criticality, impacts)
print(f"Spearman: {result.correlation.spearman_coefficient:.3f}")

# Visualize
visualizer = GraphVisualizer()
html = visualizer.render_html(graph, criticality)
```

---

## Pipeline Steps

### Step 1: Graph Generation

Generate realistic pub-sub system topologies for analysis and testing.

```bash
python generate_graph.py --scenario iot --scale medium --output system.json
```

#### Supported Scenarios

| Scenario | Description | Example Components |
|----------|-------------|-------------------|
| `iot` | IoT Smart City | Sensors, gateways, MQTT brokers, analytics |
| `financial` | Financial Trading | Order routers, market data, risk engines |
| `microservices` | Microservices | Services, RabbitMQ, Kafka, event queues |
| `ros2` | Autonomous Vehicles | Perception, planning, control, DDS |

#### Scale Parameters

| Scale | Applications | Topics | Brokers | Infrastructure |
|-------|-------------|--------|---------|----------------|
| `small` | ~8 | ~6 | 1 | 2 |
| `medium` | ~20 | ~15 | 3 | 4 |
| `large` | ~50 | ~40 | 5 | 8 |

#### Graph JSON Format

```json
{
  "metadata": {
    "scenario": "iot",
    "scale": "medium",
    "generated_at": "2025-01-15T10:30:00"
  },
  "nodes": [
    {"id": "sensor_0", "name": "Temperature Sensor", "type": "Application", "layer": "application"},
    {"id": "topic_temp", "name": "temperature/readings", "type": "Topic", "layer": "topic"},
    {"id": "mqtt_broker", "name": "MQTT Broker", "type": "Broker", "layer": "broker"},
    {"id": "edge_node_0", "name": "Edge Node", "type": "Node", "layer": "infrastructure"}
  ],
  "edges": [
    {"source": "sensor_0", "target": "topic_temp", "type": "PUBLISHES_TO"},
    {"source": "topic_temp", "target": "aggregator_0", "type": "SUBSCRIBES_TO"},
    {"source": "topic_temp", "target": "mqtt_broker", "type": "DEPENDS_ON"},
    {"source": "mqtt_broker", "target": "edge_node_0", "type": "RUNS_ON"}
  ]
}
```

---

### Step 2: Graph Analysis

Calculate criticality scores using structural graph algorithms.

```bash
python analyze_graph.py --input system.json --output results/ --format json html --full
```

#### Quality Attributes Analyzed

**Reliability Analysis:**
- Single Points of Failure (SPOFs) via articulation point detection
- Redundancy assessment via k-connectivity
- Fault tolerance via component reachability

**Maintainability Analysis:**
- Coupling metrics (afferent/efferent coupling)
- Cohesion analysis
- Modularity detection via community algorithms

**Availability Analysis:**
- Network connectivity
- Reachability analysis
- Service dependency chains

#### Anti-Pattern Detection

| Anti-Pattern | Detection Method | Impact |
|--------------|------------------|--------|
| God Topic | High fan-in/fan-out (>10 connections) | Bottleneck, SPOF |
| Single Point of Failure | Articulation point detection | Reliability risk |
| Circular Dependency | Cycle detection in dependency graph | Maintainability issue |
| Chatty Communication | High message frequency analysis | Performance degradation |
| Broker Bottleneck | Betweenness centrality analysis | Scalability limit |

---

### Step 3: Failure Simulation

Simulate component failures to measure actual system impact.

```bash
# Test all components
python simulate_graph.py --input system.json --campaign --export-json simulation.json

# Single component failure
python simulate_graph.py --input system.json --component broker_0

# With cascading failures
python simulate_graph.py --input system.json --campaign --cascade
```

#### Simulation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `--component` | Single failure | Targeted analysis |
| `--campaign` | Test all components | Exhaustive analysis |
| `--attack` | Targeted attack simulation | Security assessment |
| `--chaos` | Random failure injection | Chaos engineering |
| `--event-sim` | Event-driven simulation | Performance testing |

#### Impact Metrics

The simulation calculates:
- **Reachability Loss**: % of nodes no longer reachable
- **Service Disruption**: Number of applications affected
- **Cascade Depth**: How far failures propagate
- **Impact Score**: Normalized 0-1 severity measure

---

### Step 4: Validation

Compare predicted criticality with actual simulation impact.

```bash
python validate_graph.py --input system.json --output results/ --format json html
```

#### Validation Approach

1. **Calculate Predicted Scores**: Use composite criticality formula
2. **Run Failure Simulation**: Get actual impact for each component
3. **Compare Rankings**: Correlate predicted vs actual rankings
4. **Classify Accuracy**: Measure critical/non-critical classification

#### Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Spearman ρ | ≥ 0.70 | Rank correlation between predicted and actual |
| F1-Score | ≥ 0.90 | Classification accuracy for critical components |
| Precision | ≥ 0.80 | True Positives / (True Positives + False Positives) |
| Recall | ≥ 0.80 | True Positives / (True Positives + False Negatives) |
| Top-5 Overlap | ≥ 60% | Overlap of top-5 predicted vs actual |
| Top-10 Overlap | ≥ 70% | Overlap of top-10 predicted vs actual |

#### Advanced Validation

```bash
# Full analysis with sensitivity, bootstrap, and cross-validation
python validate_graph.py --input system.json --full-analysis

# Custom thresholds
python validate_graph.py --input system.json --target-spearman 0.8 --target-f1 0.85
```

---

### Step 5: Visualization

Generate interactive multi-layer visualizations.

```bash
# Interactive network graph
python visualize_graph.py --input system.json --output graph.html

# Multi-layer view
python visualize_graph.py --input system.json --output layers.html --multi-layer

# Criticality coloring
python visualize_graph.py --input system.json --output crit.html --color-by criticality

# Comprehensive dashboard
python visualize_graph.py --input system.json --output dashboard.html --dashboard
```

#### Visualization Features

- **Interactive Network**: Vis.js-based with physics simulation
- **Multi-Layer View**: Separated by system layer
- **Criticality Heatmap**: Color-coded by severity
- **Dashboard**: Metrics, charts, and component tables

#### Layout Algorithms

| Layout | Description | Best For |
|--------|-------------|----------|
| `spring` | Force-directed | General graphs |
| `hierarchical` | Layer-based tree | Dependency chains |
| `circular` | Circular arrangement | Small graphs |
| `shell` | Concentric by type | Clustered systems |

---

## Multi-Layer Graph Model

Our approach models pub-sub systems as **four-layer directed graphs**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                           │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│   │ Sensor  │  │Aggregator│  │Dashboard│  │ Alert   │          │
│   │   App   │  │   App   │  │   App   │  │ Service │          │
│   └────┬────┘  └────┬────┘  └────▲────┘  └────▲────┘          │
│        │            │            │            │                 │
│        │ PUBLISHES  │ SUBSCRIBES │            │                 │
├────────▼────────────▼────────────┴────────────┴─────────────────┤
│                       TOPIC LAYER                               │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │ temperature │  │  humidity   │  │   alerts    │            │
│   │   /readings │  │  /readings  │  │  /critical  │            │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│          │ DEPENDS_ON     │                │                    │
├──────────▼────────────────▼────────────────▼────────────────────┤
│                      BROKER LAYER                               │
│   ┌─────────────────────────────────────────────────┐          │
│   │              MQTT / Kafka Broker                │          │
│   └────────────────────────┬────────────────────────┘          │
│                            │ RUNS_ON                            │
├────────────────────────────▼────────────────────────────────────┤
│                   INFRASTRUCTURE LAYER                          │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │ Edge Node 1 │  │ Edge Node 2 │  │   Gateway   │            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Node Types

| Type | Layer | Description |
|------|-------|-------------|
| Application | Application | Publishers, subscribers, services |
| Topic | Topic | Message channels, queues |
| Broker | Broker | Message brokers (MQTT, Kafka, RabbitMQ) |
| Node | Infrastructure | Servers, gateways, edge devices |

### Edge Types

| Type | Direction | Meaning |
|------|-----------|---------|
| PUBLISHES_TO | App → Topic | Application publishes to topic |
| SUBSCRIBES_TO | Topic → App | Application subscribes to topic |
| DEPENDS_ON | Topic → Broker | Topic managed by broker |
| RUNS_ON | Broker → Node | Broker runs on infrastructure |
| CONNECTS_TO | Node → Node | Infrastructure connectivity |

---

## Criticality Scoring

### Composite Criticality Score Formula

```
C_score = α × BC + β × AP + γ × I + δ × DC + ε × PR
```

Where:
- **BC** = Betweenness Centrality (normalized)
- **AP** = Articulation Point (1.0 if SPOF, 0.0 otherwise)
- **I** = Impact Score (based on reachability)
- **DC** = Degree Centrality (normalized)
- **PR** = PageRank (normalized)

### Default Weights

| Parameter | Weight | Rationale |
|-----------|--------|-----------|
| α (BC) | 0.25 | Information flow bottlenecks |
| β (AP) | 0.30 | Single points of failure (highest weight) |
| γ (I) | 0.25 | Downstream impact |
| δ (DC) | 0.10 | Direct connectivity |
| ε (PR) | 0.10 | Overall importance |

### Criticality Levels

| Level | Score Range | Description |
|-------|-------------|-------------|
| Critical | ≥ 0.70 | Immediate attention required |
| High | 0.50 - 0.69 | High priority for redundancy |
| Medium | 0.30 - 0.49 | Moderate risk |
| Low | 0.10 - 0.29 | Low risk |
| Minimal | < 0.10 | Negligible impact |

---

## Validation Metrics

### Correlation Analysis

**Spearman Rank Correlation (ρ)**
- Measures monotonic relationship between predicted and actual rankings
- Target: ρ ≥ 0.70
- Range: -1 to +1 (1 = perfect positive correlation)

**Pearson Correlation (r)**
- Measures linear relationship between scores
- Useful for assessing score magnitude accuracy

**Kendall's Tau (τ)**
- Alternative rank correlation, more robust to ties

### Classification Metrics

**Confusion Matrix:**
```
                    Predicted
                Critical  Non-Critical
Actual  Critical    TP         FN
        Non-Crit    FP         TN
```

**Derived Metrics:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
- Accuracy = (TP + TN) / Total

### Ranking Metrics

- **Top-k Overlap**: Intersection of top-k predicted vs actual
- **Mean Rank Difference**: Average difference in rankings
- **Max Rank Difference**: Worst-case ranking error

---

## Project Structure

```
software-as-a-graph/
├── run.sh                      # End-to-end demo script
├── generate_graph.py           # Step 1: Graph generation CLI
├── analyze_graph.py            # Step 2: Analysis CLI
├── simulate_graph.py           # Step 3: Simulation CLI
├── validate_graph.py           # Step 4: Validation CLI
├── visualize_graph.py          # Step 5: Visualization CLI
│
├── src/
│   ├── analysis/               # Graph analysis modules
│   │   ├── graph_analyzer.py   # Main analyzer class
│   │   ├── quality_analyzer.py # Quality attribute analysis
│   │   └── antipattern_detector.py
│   │
│   ├── simulation/             # Failure simulation
│   │   ├── graph_simulator.py  # Failure simulator
│   │   └── event_simulator.py  # Event-driven simulation
│   │
│   ├── validation/             # Validation module
│   │   └── graph_validator.py  # Statistical validation
│   │
│   └── visualization/          # Visualization
│       ├── graph_visualizer.py # Vis.js renderer
│       └── dashboard_generator.py
│
├── tests/                      # Test suites
│   ├── test_simulation.py
│   ├── test_validation.py
│   └── test_visualization.py
│
├── examples/                   # Example scripts
│   ├── quick_start.py
│   ├── simulation_examples.py
│   ├── validation_examples.py
│   └── visualization_examples.py
│
└── output/                     # Generated outputs (gitignored)
```

---

## API Reference

### GraphAnalyzer

```python
from src.analysis import GraphAnalyzer

analyzer = GraphAnalyzer()

# Full analysis
results = analyzer.analyze(graph)

# Specific analysis
reliability = analyzer.analyze_reliability(graph)
maintainability = analyzer.analyze_maintainability(graph)
availability = analyzer.analyze_availability(graph)
antipatterns = analyzer.detect_antipatterns(graph)
```

### FailureSimulator

```python
from src.simulation import FailureSimulator

simulator = FailureSimulator(seed=42)

# Single failure
result = simulator.simulate_single_failure(graph, "broker_0")

# All components
batch_result = simulator.simulate_all_single_failures(graph)

# With cascade
result = simulator.simulate_single_failure(graph, "broker_0", enable_cascade=True)
```

### GraphValidator

```python
from src.validation import GraphValidator

validator = GraphValidator(seed=42)

# Basic validation
result = validator.validate(graph, predicted_scores, actual_impacts)

# With simulation
result = validator.validate_with_simulation(graph, predicted_scores)

# Advanced analysis
sensitivity = validator.run_sensitivity_analysis(graph, scores, impacts)
bootstrap = validator.run_bootstrap_analysis(graph, scores, impacts, n_iterations=1000)
cv_result = validator.run_cross_validation(graph, scores, impacts, n_folds=5)
```

### GraphVisualizer

```python
from src.visualization import GraphVisualizer, DashboardGenerator

# Basic visualization
visualizer = GraphVisualizer()
html = visualizer.render_html(graph, criticality)

# Multi-layer view
html = visualizer.render_multi_layer_html(graph, criticality)

# Dashboard
generator = DashboardGenerator()
dashboard = generator.generate(graph, criticality, validation, simulation)
```

---

## Examples

### Example 1: IoT Smart City Analysis

```bash
./run.sh --scenario iot --scale medium
```

### Example 2: Financial Trading System

```bash
./run.sh --scenario financial --scale large --full-validation
```

### Example 3: Custom Analysis

```python
import json
import networkx as nx
from src.analysis import GraphAnalyzer
from src.validation import GraphValidator

# Load your system
with open('my_system.json') as f:
    data = json.load(f)

# Build graph
G = nx.DiGraph()
for node in data['nodes']:
    G.add_node(node['id'], **node)
for edge in data['edges']:
    G.add_edge(edge['source'], edge['target'], **edge)

# Analyze
analyzer = GraphAnalyzer()
results = analyzer.analyze(G)

# Validate
validator = GraphValidator()
validation = validator.validate_with_simulation(G, results['criticality'])

print(f"Spearman: {validation.correlation.spearman_coefficient:.3f}")
print(f"F1-Score: {validation.classification.overall.f1_score:.3f}")
```

---

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

### Development Setup

```bash
# Clone and install
git clone https://github.com/onuralpyigit/software-as-a-graph.git
cd software-as-a-graph
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use this framework in your research, please cite:

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

## Acknowledgments

- NetworkX team for the excellent graph analysis library
- Neo4j for the powerful graph database
- Vis.js for interactive network visualizations

---

*Generated by Software-as-a-Graph Research Framework v2.0*