# Software-as-a-Graph

**Predict which components in a distributed system will cause the most damage when they fail — using only the system's architecture.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black)](https://nextjs.org/)
[![Docker](https://img.shields.io/badge/docker-compose-blue)](https://www.docker.com/)
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-green.svg)](https://neo4j.com/)
[![IEEE RASSE 2025](https://img.shields.io/badge/IEEE-RASSE%202025-orange.svg)](#citation)

## The Problem

In distributed publish-subscribe systems (ROS 2, Kafka, MQTT, etc.), some components are far more critical than others. When they fail, failures cascade through the system. Traditionally, finding these weak points requires expensive runtime monitoring or waiting for production incidents.

## Our Approach

We treat your system architecture as a graph and apply topological analysis to predict critical components *before deployment*. The key insight is simple: **a component's position in the dependency graph reliably predicts its real-world failure impact.**

This has been validated empirically with Spearman correlation > 0.87 and F1-scores > 0.90 across multiple system scales.

## Features

- **Web Dashboard (Genieus):** Interactive visualization of system topology and analysis results.
- **Graph Modeling:** Automatic conversion of topology JSON to Neo4j graph.
- **Validation:** Statistical validation of predictions against ground-truth simulations.
- **Simulation:** Cascade failure simulation to test resilience.
- **REST API:** Fully featured API for integration.

## Quick Start (Docker)

The easiest way to run the full system (Frontend + Backend + Database) is via Docker Compose.

### Prerequisites
- Docker & Docker Compose
- 4GB+ RAM available

### Run
```bash
git clone https://github.com/your-org/software-as-a-graph.git
cd software-as-a-graph

# Start the full stack
docker compose up --build
```

### Access
- **Web Dashboard:** [http://localhost:7000](http://localhost:7000)
- **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Neo4j Browser:** [http://localhost:7474](http://localhost:7474) (User: `neo4j`, Password: `password`)

## Web Interface (Genieus)

The **Genieus** dashboard provides a modern interface for the analysis pipeline:

1.  **Dashboard:** High-level metrics, criticality distribution, and top-list.
2.  **Graph Explorer:** Interactive 2D/3D force-directed graph. Filter by layer, search components, and view details.
3.  **Analysis:** Run structural analysis and quality scoring on specific layers.
4.  **Simulation:** Simulates component failures and visualizes cascade paths.
5.  **Settings:** Configure Neo4j connection.

## Development Setup (CLI & Local)

If you prefer running components individually or using the CLI tools directly:

### 1. Backend & CLI
```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI pipeline
python bin/run.py --all --layer system
```

### 2. Frontend
```bash
cd genieus
npm install
npm run dev
# Runs on http://localhost:3000
```

### 3. CLI Tools Usage

The `bin/` directory contains individual scripts for each step of the pipeline:

```bash
# 1. Generate & Import
python bin/generate_graph.py --config config/medium_scale.yaml --output data/system.json
python bin/import_graph.py --input data/system.json --clear

# 2. Analyze
python bin/analyze_graph.py --layer system

# 3. Simulate (Ground Truth)
python bin/simulate_graph.py failure --layer system --exhaustive

# 4. Validate
python bin/validate_graph.py --layer system

# 5. Visualize (Static HTML)
python bin/visualize_graph.py --layer system --output dashboard.html
```

## How It Works — The 6-Step Pipeline

```
Architecture  →  Graph  →  Metrics  →  Scores  →  Simulation  →  Validation  →  Dashboard
   (input)      Step 1     Step 2     Step 3      Step 4         Step 5        Step 6
```

| Step | What It Does |
|------|-------------|
| **1. [Graph Model](docs/graph-model.md)** | Converts system topology into a weighted directed graph |
| **2. [Structural Analysis](docs/structural-analysis.md)** | Computes centrality metrics (PageRank, Betweenness, etc.) |
| **3. [Quality Scoring](docs/quality-scoring.md)** | Maps metrics to quality dimensions (RMAV) using AHP weights |
| **4. [Failure Simulation](docs/failure-simulation.md)** | Injects faults and measures cascade impact for ground truth |
| **5. [Validation](docs/validation.md)** | Statistically compares predictions against simulation impact |
| **6. [Visualization](docs/visualization.md)** | Generates interactive dashboards |

## Project Structure

```
software-as-a-graph/
├── genieus/                # Next.js Frontend (Web Dashboard)
├── api.py                  # FastAPI Backend
├── docker-compose.yml      # Full stack orchestration
├── bin/                    # CLI tools
│   ├── generate_graph.py   # Synthetic generation
│   ├── analyze_graph.py    # Analysis engine
│   └── ...                 # Other CLI scripts
├── src/                    # Python Source Code
│   ├── core/               # Core entities and utilities
│   ├── analysis/           # Analysis logic (Structural, Quality)
│   ├── simulation/         # Simulation logic (Event, Failure)
│   ├── validation/         # Validation logic
│   ├── visualization/      # Visualization logic
│   ├── generation/         # Graph generation
│   └── cli/                # CLI utilities
├── config/                 # Configuration templates
└── tests/                  # Unit tests
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