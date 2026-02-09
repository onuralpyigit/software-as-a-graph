# Software-as-a-Graph

**Predict which components in a distributed system will cause the most damage when they fail — using only the system's architecture.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-green.svg)](https://neo4j.com/)
[![IEEE RASSE 2025](https://img.shields.io/badge/IEEE-RASSE%202025-orange.svg)](#citation)

## The Problem

In distributed publish-subscribe systems (ROS 2, Kafka, MQTT, etc.), some components are far more critical than others. When they fail, failures cascade through the system. Traditionally, finding these weak points requires expensive runtime monitoring or waiting for production incidents.

## Our Approach

We treat your system architecture as a graph and apply topological analysis to predict critical components *before deployment*. The key insight is simple: **a component's position in the dependency graph reliably predicts its real-world failure impact.**

This has been validated empirically with Spearman correlation > 0.87 and F1-scores > 0.90 across multiple system scales.

## How It Works — The 6-Step Pipeline

```
Architecture  →  Graph  →  Metrics  →  Scores  →  Simulation  →  Validation  →  Dashboard
   (input)      Step 1     Step 2     Step 3      Step 4         Step 5        Step 6
```

Each step builds on the previous one:

| Step | What It Does | Output |
|------|-------------|--------|
| **1. [Graph Model](docs/graph-model.md)** | Converts your system topology into a weighted directed graph with derived dependencies | Graph G(V, E) |
| **2. [Structural Analysis](docs/structural-analysis.md)** | Computes centrality metrics (PageRank, Betweenness, etc.) for every component | Metric vectors M(v) |
| **3. [Quality Scoring](docs/quality-scoring.md)** | Maps metrics to four quality dimensions (Reliability, Maintainability, Availability, Vulnerability) using AHP weights | Quality scores Q(v) |
| **4. [Failure Simulation](docs/failure-simulation.md)** | Injects actual faults and measures cascade impact to establish ground truth | Impact scores I(v) |
| **5. [Validation](docs/validation.md)** | Statistically compares predicted Q(v) against actual I(v) | Spearman ρ, F1, etc. |
| **6. [Visualization](docs/visualization.md)** | Generates interactive dashboards for decision-making | HTML dashboard |

Steps 1–3 give you predictions. Step 4 gives you ground truth. Step 5 proves the predictions work. Step 6 makes it all actionable.

## Quick Start

### Prerequisites

- Python 3.9+
- Neo4j 5.x (locally or via Docker)

### Install

```bash
git clone https://github.com/your-org/software-as-a-graph.git
cd software-as-a-graph
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python bin/run.py --all --layer system
```

This generates a synthetic system, analyzes it, simulates failures, validates predictions, and opens a dashboard — all in one command.

### Run Step by Step

```bash
# 1. Generate a synthetic system and import it into Neo4j
python bin/generate_graph.py --config config/medium_scale.yaml --output data/system.json
python bin/import_graph.py --input data/system.json --clear

# 2. Analyze structural metrics
python bin/analyze_graph.py --layer system

# 3. Simulate failures to get ground truth
python bin/simulate_graph.py failure --layer system --exhaustive

# 4. Validate predictions against simulation
python bin/validate_graph.py --layer system

# 5. Generate an interactive dashboard
python bin/visualize_graph.py --layer system --output dashboard.html --open
```

### Layer Options

Analysis can target specific architectural layers or the full system:

| Layer | What It Covers |
|-------|---------------|
| `app` | Application-to-application dependencies |
| `infra` | Infrastructure (node-to-node) dependencies |
| `mw` | Middleware (broker) dependencies |
| `system` | All layers combined |

## Project Structure

```
software-as-a-graph/
├── bin/                    # CLI tools (one per pipeline step)
│   ├── generate_graph.py   #   Generate synthetic topology
│   ├── import_graph.py     #   Import into Neo4j
│   ├── analyze_graph.py    #   Structural analysis + quality scoring
│   ├── simulate_graph.py   #   Failure simulation
│   ├── validate_graph.py   #   Statistical validation
│   ├── visualize_graph.py  #   Dashboard generation
│   ├── run.py              #   Pipeline orchestrator
│   └── benchmark.py        #   Performance benchmarks
├── src/
│   ├── domain/             # Core models, services, and business logic
│   ├── application/        # Use cases and service orchestration
│   └── infrastructure/     # Neo4j adapters, file I/O
├── config/                 # Scale presets and configuration templates
├── docs/                   # Step-by-step methodology documentation
└── tests/                  # Unit and integration tests
```

## Citation

```bibtex
@INPROCEEDINGS{11315354,
  author={Yigit, Ibrahim Onuralp and Buzluca, Feza},
  booktitle={2025 IEEE International Conference on Recent Advances in
             Systems Science and Engineering (RASSE)},
  title={A Graph-Based Dependency Analysis Method for Identifying
         Critical Components in Distributed Publish-Subscribe Systems},
  year={2025},
  pages={1-9},
  doi={10.1109/RASSE64831.2025.11315354}}
```

## License

MIT License — see [LICENSE](LICENSE) for details.