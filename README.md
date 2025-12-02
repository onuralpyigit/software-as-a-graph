# 🔬 Software-as-a-Graph

## Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-green.svg)](https://networkx.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-orange.svg)](https://neo4j.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

This project implements a **comprehensive methodology** for modeling and analyzing distributed publish-subscribe (pub-sub) systems using graph-based techniques. The approach transforms complex distributed architectures into analyzable graph structures, enabling:

- 🎯 **Predictive Analysis**: Identify critical components *before* failures occur
- 🔍 **Structural Vulnerability Detection**: Discover single points of failure and anti-patterns
- 📊 **Quantitative Assessment**: Transform qualitative architectural attributes into measurable metrics
- ✅ **Validation Framework**: Correlate predictions with simulation outcomes

### Research Publication

> **IEEE RASSE 2025** (Accepted): "Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems"

---

## 🎯 Research Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Spearman ρ** | ≥ 0.7 | Correlation between predicted criticality and actual failure impact |
| **F1 Score** | ≥ 0.9 | Harmonic mean of precision and recall |
| **Precision** | ≥ 0.9 | Correctly identified critical / Total identified |
| **Recall** | ≥ 0.85 | Correctly identified critical / Actual critical |

---

## 📐 Mathematical Foundation

### Composite Criticality Scoring Formula

The core innovation is the **Composite Criticality Score**:

```
C_score(v) = α · C_B^norm(v) + β · AP(v) + γ · I(v)
```

| Symbol | Range | Description |
|--------|-------|-------------|
| `C_B^norm(v)` | [0, 1] | **Normalized Betweenness Centrality** - Measures information flow importance |
| `AP(v)` | {0, 1} | **Articulation Point Indicator** - 1 if removing node disconnects graph |
| `I(v)` | [0, 1] | **Impact Score** - Measures reachability loss when node fails |
| `α, β, γ` | [0, 1] | **Tunable Weights** (default: 0.4, 0.3, 0.3) |

### Criticality Level Classification

| Score Range | Level | Action Required |
|-------------|-------|-----------------|
| ≥ 0.8 | 🔴 **CRITICAL** | Immediate redundancy required |
| ≥ 0.6 | 🟠 **HIGH** | Enhanced monitoring, plan redundancy |
| ≥ 0.4 | 🟡 **MEDIUM** | Standard monitoring |
| ≥ 0.2 | 🟢 **LOW** | Regular review |
| < 0.2 | ⚪ **MINIMAL** | No special attention |

---

## 🏗️ Five-Step Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SOFTWARE-AS-A-GRAPH METHODOLOGY                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │   STEP 1    │    │   STEP 2    │    │   STEP 3    │                     │
│  │  GENERATE   │───▶│   IMPORT    │───▶│   ANALYZE   │                     │
│  │ Graph Data  │    │  to Neo4j   │    │ Criticality │                     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                     │
│        │                  │                   │                             │
│        │ JSON             │ Cypher            │ Scores                      │
│        ▼                  ▼                   ▼                             │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │                   DATA FLOW                          │                   │
│  └─────────────────────────────────────────────────────┘                   │
│                                               │                             │
│                                               ▼                             │
│                     ┌─────────────┐    ┌─────────────┐                     │
│                     │   STEP 5    │    │   STEP 4    │                     │
│                     │  VISUALIZE  │◀───│  SIMULATE   │                     │
│                     │   Results   │    │ & VALIDATE  │                     │
│                     └─────────────┘    └─────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Step | Tool | Purpose |
|------|------|---------|
| 1 | `generate_graph.py` | Generate realistic pub-sub system graph data |
| 2 | `import_graph.py` | Import graph data into Neo4j database |
| 3 | `analyze_graph.py` | Analyze graph and calculate criticality scores |
| 4 | `simulate_graph.py` | Simulate failures and validate predictions |
| 5 | `visualize_graph.py` | Visualize multi-layer graph and results |

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- Neo4j 5.0+ (optional, for database storage)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/software-as-a-graph.git
cd software-as-a-graph

# Install required packages
pip install networkx scipy matplotlib

# Optional: Install Neo4j driver
pip install neo4j
```

### Verify Installation

```bash
python generate_graph.py --list-scales
python generate_graph.py --list-scenarios
```

---

## 🚀 Quick Start

### Option 1: Complete Pipeline (5 Steps)

```bash
# Step 1: Generate graph data
python generate_graph.py --scale medium --scenario iot --output system.json

# Step 2: Import to Neo4j
python import_graph.py --input system.json --uri bolt://localhost:7687 \
    --user neo4j --password your_password --clear --analytics

# Step 3: Analyze graph
python analyze_graph.py --input system.json --detect-antipatterns \
    --export-json analysis.json

# Step 4: Simulate failures
python simulate_graph.py --input system.json --campaign \
    --export-json simulation.json

# Step 5: Visualize results
python visualize_graph.py --input system.json --output dashboard.html \
    --dashboard --analysis analysis.json
```

### Option 2: E2E Pipeline Script

```bash
# Quick demo mode
python e2e_pipeline.py --demo

# Full pipeline with Neo4j
python e2e_pipeline.py --scenario financial --scale medium \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j --neo4j-password password \
    --output-dir ./results

# JSON-only mode (no Neo4j required)
python e2e_pipeline.py --scenario iot --scale small --no-neo4j
```

---

## 📖 Step-by-Step Guide

### Step 1: Generate Graph Data (`generate_graph.py`)

Generate realistic pub-sub system topologies with domain-specific configurations.

#### Usage

```bash
python generate_graph.py [OPTIONS]
```

#### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--scale` | Graph scale preset | `medium` |
| `--scenario` | Domain scenario | `generic` |
| `--antipatterns` | Anti-patterns to inject | None |
| `--output` | Output file path | `pub_sub_system.json` |
| `--seed` | Random seed | `42` |

#### Scale Presets

| Scale | Nodes | Apps | Topics | Brokers |
|-------|-------|------|--------|---------|
| `tiny` | 2 | 6 | 4 | 1 |
| `small` | 4 | 12 | 8 | 2 |
| `medium` | 8 | 25 | 15 | 3 |
| `large` | 15 | 50 | 30 | 5 |
| `xlarge` | 30 | 100 | 60 | 8 |
| `extreme` | 60 | 200 | 120 | 15 |

#### Domain Scenarios

| Scenario | Description | Example Applications |
|----------|-------------|---------------------|
| `iot` | IoT/Smart City | TrafficSensor, ParkingSensor, AirQualityMonitor |
| `financial` | Financial Trading | MarketDataFeed, OrderProcessor, RiskEngine |
| `healthcare` | Healthcare Systems | VitalSignsMonitor, PatientTracker, AlertDispatcher |
| `ecommerce` | E-commerce | OrderService, InventoryManager, PaymentProcessor |
| `autonomous_vehicle` | Autonomous Vehicles | LidarProcessor, CameraFusion, PathPlanner |
| `gaming` | Online Gaming | GameStateManager, PlayerController, PhysicsEngine |

#### Anti-Patterns

| Pattern | Description |
|---------|-------------|
| `spof` | Single Point of Failure |
| `god_topic` | Topic with excessive connections |
| `circular` | Circular dependencies |
| `broker_overload` | Overloaded broker |
| `bottleneck` | System bottleneck |
| `chatty` | Chatty communication pattern |

#### Examples

```bash
# Generate medium IoT system
python generate_graph.py --scale medium --scenario iot --output iot_system.json

# Generate large financial system with anti-patterns
python generate_graph.py --scale large --scenario financial \
    --antipatterns spof god_topic --output financial_system.json

# Generate with high-availability patterns
python generate_graph.py --scale medium --scenario healthcare \
    --ha --multi-zone --num-zones 3 --output ha_system.json

# Preview without generating
python generate_graph.py --scale xlarge --scenario gaming --preview
```

#### Output Format

```json
{
  "metadata": {
    "scenario": "iot",
    "scale": "medium",
    "seed": 42,
    "generated_at": "2025-01-01T00:00:00"
  },
  "nodes": [...],
  "brokers": [...],
  "topics": [...],
  "applications": [...],
  "relationships": {
    "runs_on": [...],
    "publishes_to": [...],
    "subscribes_to": [...],
    "routes": [...]
  }
}
```

---

### Step 2: Import to Neo4j (`import_graph.py`)

Import generated graph data into Neo4j for persistent storage and advanced queries.

#### Usage

```bash
python import_graph.py [OPTIONS]
```

#### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--uri` | Neo4j connection URI | `bolt://localhost:7687` |
| `--user` | Neo4j username | `neo4j` |
| `--password` | Neo4j password | `password` |
| `--input` | Input JSON file | Required |
| `--clear` | Clear database first | False |
| `--analytics` | Run analytics after import | False |

#### Examples

```bash
# Basic import
python import_graph.py --input system.json --uri bolt://localhost:7687 \
    --user neo4j --password mypassword

# Import with database clear and analytics
python import_graph.py --input system.json --clear --analytics

# Import with validation and progress
python import_graph.py --input system.json --validate --progress

# Export useful Cypher queries
python import_graph.py --input system.json --export-queries queries.cypher
```

#### Neo4j Schema

**Nodes:**
- `(:Application)` - Publisher/subscriber applications
- `(:Topic)` - Message topics
- `(:Broker)` - Message brokers
- `(:Node)` - Infrastructure nodes

**Relationships:**
- `[:PUBLISHES_TO]` - Application publishes to Topic
- `[:SUBSCRIBES_TO]` - Application subscribes to Topic
- `[:ROUTES]` - Broker routes Topic
- `[:RUNS_ON]` - Application/Broker runs on Node
- `[:DEPENDS_ON]` - **Derived** dependency relationship

#### Key Cypher Queries

```cypher
-- Find all critical dependencies
MATCH (sub:Application)-[:SUBSCRIBES_TO]->(t:Topic)<-[:PUBLISHES_TO]-(pub:Application)
WHERE sub <> pub
RETURN sub.name, t.name, pub.name

-- Find Single Points of Failure
MATCH (a:Application)
WHERE size((a)-[:PUBLISHES_TO]->()) > 5
RETURN a.name, size((a)-[:PUBLISHES_TO]->()) as pub_count

-- Topic connectivity analysis
MATCH (t:Topic)
RETURN t.name, 
       size(()-[:PUBLISHES_TO]->(t)) as publishers,
       size(()-[:SUBSCRIBES_TO]->(t)) as subscribers
ORDER BY publishers + subscribers DESC
```

---

### Step 3: Analyze Graph (`analyze_graph.py`)

Perform comprehensive analysis including criticality scoring, structural analysis, and anti-pattern detection.

#### Usage

```bash
python analyze_graph.py [OPTIONS]
```

#### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input JSON file | Required |
| `--neo4j` | Load from Neo4j | False |
| `--alpha` | Betweenness centrality weight | `0.4` |
| `--beta` | Articulation point weight | `0.3` |
| `--gamma` | Impact score weight | `0.3` |
| `--detect-antipatterns` | Detect anti-patterns | False |
| `--simulate` | Run failure simulations | False |
| `--export-json` | Export results to JSON | None |

#### Examples

```bash
# Basic analysis
python analyze_graph.py --input system.json

# Full analysis with anti-pattern detection
python analyze_graph.py --input system.json --detect-antipatterns

# Analysis with custom weights
python analyze_graph.py --input system.json --alpha 0.5 --beta 0.25 --gamma 0.25

# Analysis with failure simulations
python analyze_graph.py --input system.json --simulate --top-n 10

# Export results
python analyze_graph.py --input system.json --detect-antipatterns \
    --export-json analysis_results.json --export-csv scores.csv

# Load from Neo4j
python analyze_graph.py --neo4j --uri bolt://localhost:7687 \
    --user neo4j --password mypassword
```

#### Output

```
================================================================================
                    GRAPH ANALYSIS RESULTS
================================================================================

📈 GRAPH SUMMARY
   Total Nodes:     57
   Total Edges:     301
   Density:         0.0943
   Connected:       Yes
   Components:      1

🔧 STRUCTURAL ANALYSIS
   Articulation Points (SPOFs): 2
   Bridges:                     3
   Cycles Detected:             45

⚠️ CRITICALITY DISTRIBUTION
   🔴 CRITICAL    0
   🟠 HIGH        3
   🟡 MEDIUM      10
   🟢 LOW         44
   ⚪ MINIMAL     0

🎯 TOP 10 CRITICAL COMPONENTS
   # Type         Component                Score    Level      AP
   ─────────────────────────────────────────────────────────────
   1 Application  app_10                   0.789    HIGH       ★
   2 Application  app_8                    0.695    HIGH
   3 Application  app_16                   0.602    HIGH
   ...
```

---

### Step 4: Simulate Failures (`simulate_graph.py`)

Validate analysis predictions through comprehensive failure simulations.

#### Usage

```bash
python simulate_graph.py [OPTIONS]
```

#### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input JSON file | Required |
| `--component` | Single component to fail | None |
| `--components` | Multiple components to fail | None |
| `--cascade` | Enable cascading failures | True |
| `--campaign` | Test all components | False |
| `--event-sim` | Event-driven simulation | False |
| `--chaos` | Chaos engineering mode | False |

#### Simulation Modes

| Mode | Description |
|------|-------------|
| **Single Failure** | Fail a specific component |
| **Multi-Failure** | Fail multiple components simultaneously |
| **Campaign** | Systematically test each component |
| **Attack** | Targeted attack based on strategy |
| **Random** | Random failure injection |
| **Event-Driven** | Time-based message simulation |
| **Load Test** | Stress testing with ramp-up |
| **Chaos** | Random failures with recovery |

#### Examples

```bash
# Single component failure
python simulate_graph.py --input system.json --component app_0

# Multiple component failure with cascade
python simulate_graph.py --input system.json \
    --components app_0 app_1 broker_0 --cascade

# Failure campaign (test all applications)
python simulate_graph.py --input system.json --campaign \
    --component-types Application

# Targeted attack simulation
python simulate_graph.py --input system.json --attack \
    --strategy criticality --count 5

# Event-driven simulation
python simulate_graph.py --input system.json --event-sim \
    --duration 60000 --failure-at 30000 --message-rate 100

# Load testing
python simulate_graph.py --input system.json --load-test \
    --initial-rate 10 --peak-rate 1000 --ramp-time 10000

# Chaos engineering
python simulate_graph.py --input system.json --chaos \
    --failure-prob 0.1 --recovery-prob 0.3

# Export results
python simulate_graph.py --input system.json --campaign \
    --export-json simulation_results.json
```

#### Output

```
================================================================================
                    FAILURE SIMULATION RESULTS
================================================================================

💥 FAILURE: app_0 (Application)
   ─────────────────────────────────────────────────────
   Components Affected:  12
   Cascade Depth:        2
   Topics Disrupted:     5
   Messages at Risk:     2,450/sec

   📊 IMPACT METRICS
      Connectivity Loss:   15.3%
      Reachability Loss:   22.1%
      Service Disruption:  8 applications
      
   🔗 CASCADE PATH
      app_0 → topic_3 → app_5 → topic_7 → app_12
```

---

### Step 5: Visualize Results (`visualize_graph.py`)

Generate interactive visualizations and comprehensive dashboards.

#### Usage

```bash
python visualize_graph.py [OPTIONS]
```

#### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input JSON file | Required |
| `--output` | Output file | Required |
| `--format` | Output format (html/png/svg/pdf) | `html` |
| `--layer` | Layer to visualize | `all` |
| `--layout` | Layout algorithm | `spring` |
| `--color-by` | Color scheme | `type` |
| `--dashboard` | Generate dashboard | False |
| `--analysis` | Include analysis results | None |

#### Layout Algorithms

| Layout | Description | Best For |
|--------|-------------|----------|
| `spring` | Force-directed | General graphs |
| `hierarchical` | Tree-like | Dependency graphs |
| `circular` | Circular arrangement | Small graphs |
| `layered` | Layer-based | Multi-layer systems |
| `kamada_kawai` | Energy minimization | Moderate graphs |
| `shell` | Concentric circles | Clustered graphs |

#### Color Schemes

| Scheme | Description |
|--------|-------------|
| `type` | Color by component type (App=Blue, Topic=Green, Broker=Red) |
| `criticality` | Color by criticality level (Critical=Red → Minimal=Gray) |
| `layer` | Color by system layer |
| `qos` | Color by QoS policy |

#### Examples

```bash
# Basic HTML visualization
python visualize_graph.py --input system.json --output graph.html

# Criticality-colored visualization
python visualize_graph.py --input system.json --output criticality.html \
    --color-by criticality

# Application layer only
python visualize_graph.py --input system.json --output apps.html \
    --layer application --layout hierarchical

# Comprehensive dashboard
python visualize_graph.py --input system.json --output dashboard.html \
    --dashboard --analysis analysis.json

# Static image export
python visualize_graph.py --input system.json --output graph.png \
    --format png --dpi 300

# Multi-layer visualization
python visualize_graph.py --input system.json --output layers.html \
    --layout layered --color-by layer
```

#### Dashboard Features

The `--dashboard` option generates a comprehensive HTML dashboard with:

- **Interactive Network Graph**: Vis.js-based visualization with physics simulation
- **Overview Statistics**: Node/edge counts, critical components, SPOFs
- **Validation Results**: Precision, recall, F1, Spearman metrics with target badges
- **Simulation Impact**: Before/after failure comparison
- **Criticality Distribution**: Bar chart showing component distribution
- **Top Critical Components**: Ranked table with scores
- **Layer Analysis**: Breakdown by system layer

---

## 📁 Project Structure

```
software-as-a-graph/
├── generate_graph.py          # Step 1: Graph generation CLI
├── import_graph.py            # Step 2: Neo4j import CLI
├── analyze_graph.py           # Step 3: Analysis CLI
├── simulate_graph.py          # Step 4: Simulation CLI
├── visualize_graph.py         # Step 5: Visualization CLI
├── e2e_pipeline.py            # Complete E2E pipeline script
├── e2e_pipeline_notebook.ipynb    # Interactive notebook
├── graph_based_methodology.ipynb  # Methodology documentation
├── README.md                  # This file
├── src/
│   ├── core/                  # Core graph models and builders
│   │   ├── models.py          # Data models
│   │   ├── graph_builder.py   # Graph construction
│   │   └── ...
│   ├── analysis/              # Analysis modules
│   │   ├── criticality.py     # Criticality scoring
│   │   ├── structural.py      # Structural analysis
│   │   └── antipatterns.py    # Anti-pattern detection
│   ├── simulation/            # Simulation modules
│   │   ├── failure.py         # Failure simulation
│   │   ├── cascade.py         # Cascade propagation
│   │   └── event_driven.py    # Event-driven simulation
│   └── visualization/         # Visualization modules
│       ├── layers.py          # Layer rendering
│       ├── dashboard.py       # Dashboard generation
│       └── ...
└── tests/                     # Test suite
    └── test_*.py
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Neo4j connection
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"

# Default analysis weights
export CRITICALITY_ALPHA=0.4
export CRITICALITY_BETA=0.3
export CRITICALITY_GAMMA=0.3
```

### Neo4j Setup

```bash
# Using Docker
docker run -d \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest

# Verify connection
python import_graph.py --connection-help
```

---

## 📊 Example Workflow

### Complete Analysis of Financial Trading System

```bash
# 1. Generate a large financial trading system with anti-patterns
python generate_graph.py \
    --scale large \
    --scenario financial \
    --antipatterns spof god_topic circular \
    --output financial_system.json \
    --seed 42

# 2. Import to Neo4j with analytics
python import_graph.py \
    --input financial_system.json \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password \
    --clear \
    --analytics \
    --export-queries useful_queries.cypher

# 3. Run comprehensive analysis
python analyze_graph.py \
    --input financial_system.json \
    --detect-antipatterns \
    --simulate \
    --top-n 15 \
    --export-json analysis_results.json \
    --export-csv criticality_scores.csv

# 4. Run failure simulations
python simulate_graph.py \
    --input financial_system.json \
    --campaign \
    --component-types Application Broker \
    --export-json simulation_results.json

# 5. Generate comprehensive dashboard
python visualize_graph.py \
    --input financial_system.json \
    --output financial_dashboard.html \
    --dashboard \
    --analysis analysis_results.json

# Open the dashboard
open financial_dashboard.html  # macOS
xdg-open financial_dashboard.html  # Linux
```

---

## 📓 Jupyter Notebooks

Interactive notebooks are provided for exploration and documentation:

| Notebook | Description |
|----------|-------------|
| `e2e_pipeline_notebook.ipynb` | Complete E2E pipeline with all 5 steps |
| `graph_based_methodology.ipynb` | Methodology explanation and theory |

```bash
# Launch Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_unified_depends_on.py

# Run with coverage
python -m pytest --cov=src tests/
```

---

## 📚 References

- NetworkX Documentation: https://networkx.org/
- Neo4j Graph Database: https://neo4j.com/
- Vis.js Network: https://visjs.github.io/vis-network/

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Onuralp Sezer**

- Research: Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems
- Publication: IEEE RASSE 2025

---

## 🙏 Acknowledgments

- NetworkX team for the excellent graph analysis library
- Neo4j for the powerful graph database
- Vis.js for interactive visualizations

---

*Generated by Software-as-a-Graph Research Framework v2.0*
