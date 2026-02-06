# Software-as-a-Graph

**Graph-Based Critical Component Prediction for Distributed Pub-Sub Systems**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-green.svg)](https://neo4j.com/)
[![IEEE RASSE 2025](https://img.shields.io/badge/IEEE-RASSE%202025-orange.svg)](#citation)

## Overview

This framework accurately predicts **critical components**—those whose failure results in the greatest system impact—by modeling your distributed software architecture as a graph.

**Key Insight:** Topological graph metrics (like PageRank and Betweenness) computed on a dependency graph can reliably predict runtime failure impact ($I(v)$) without needing expensive failure simulations.

## The 6-Step Methodology

Our approach follows a systematic pipeline:

1.  **[Generation & Model Construction](docs/graph-model.md)**
    *   Transform system topology (Apps, Topics, Brokers, Nodes) into a weighted directed graph.
    *   Derive logical dependencies (e.g., `App` $\to$ `Topic` $\to$ `App` becomes `App` $\to$ `App`).
2.  **[Structural Analysis](docs/structural-analysis.md)**
    *   Compute topological metrics ($PR$, $BT$, $CC$) to identify structural bottlenecks and super-spreaders.
3.  **[Quality Scoring](docs/quality-scoring.md)**
    *   Synthesize metrics into quality dimensions: Reliability ($R$), Maintainability ($M$), Availability ($A$), and Vulnerability ($V$).
    *   Calculate a composite Quality Score $Q(v)$.
4.  **[Failure Simulation](docs/failure-simulation.md)**
    *   (Ground Truth) Inject actual faults to measure Reachability Loss, Fragmentation, and Throughput Drop.
    *   This produces the "actual" Impact Score $I(v)$.
5.  **[Validation](docs/validation.md)**
    *   Compare predicted $Q(v)$ against simulated $I(v)$ using Spearman's rank correlation ($\rho$).
    *   *Result:* High correlation ($\rho > 0.8$) proves the graph model is a valid predictor.
6.  **[Visualization](docs/visualization.md)**
    *   Explore the graph interactively and view analysis dashboards.

---

## Quick Start

### Prerequisites
*   Python 3.9+
*   Neo4j 5.x (running locally or via Docker)

### Installation
```bash
git clone https://github.com/your-org/software-as-a-graph.git
cd software-as-a-graph
pip install -r requirements.txt
```

### Running the Pipeline
You can run the entire end-to-end pipeline with a single command:

```bash
# Run everything for the complete 'system' layer
python bin/run.py --all --layer system
```

### Step-by-Step Execution
Alternatively, run each step individually to understand the process:

**1. Generate Data & Import**
```bash
# Generate synthetic system design
python bin/generate_graph.py --config config/medium_scale.yaml --output data/system.json

# Import into Neo4j
python bin/import_graph.py --input data/system.json --clear
```

**2. Analyze & Score**
```bash
# specific layer: app, infra, mw, or system
python bin/analyze_graph.py --layer system --use-ahp
```

**3. Simulate Failures (Ground Truth)**
```bash
# Run exhaustive simulation to get actual impact scores
python bin/simulate_graph.py --layer system --exhaustive
```

**4. Validate Results**
```bash
# Compare Analysis vs. Simulation
python bin/validate_graph.py --layer system
```

**5. Visualize**
```bash
# Generate HTML dashboard
python bin/visualize_graph.py --layer system --output dashboard.html --open
```

---

## Documentation

| Step | Topic | Description |
|------|-------|-------------|
| 1 | [Graph Model](docs/graph-model.md) | Nodes, Edges, Weights, and Derivation Rules |
| 2 | [Structural Analysis](docs/structural-analysis.md) | PageRank, Betweenness, and other metrics |
| 3 | [Quality Scoring](docs/quality-scoring.md) | AHP-based quality attributes ($R, M, A, V$) |
| 4 | [Failure Simulation](docs/failure-simulation.md) | Cascading failure logic and impact metrics |
| 5 | [Validation](docs/validation.md) | Statistical correlation results |
| 6 | [Visualization](docs/visualization.md) | Dashboard & Interactive Graph details |

## Project Structure

```text
software-as-a-graph/
├── bin/                       # CLI Entry Points
│   ├── generate_graph.py      # Generate synthetic data
│   ├── import_graph.py        # Import to Neo4j
│   ├── analyze_graph.py       # Run analysis
│   ├── simulate_graph.py      # Failure simulation
│   ├── validate_graph.py      # Statistical validation
│   ├── visualize_graph.py     # Generate dashboards
│   ├── run.py                 # Pipeline orchestrator
│   └── benchmark.py           # Benchmark suite
├── src/                       # Source Code
│   ├── domain/                # Core models & logic
│   ├── application/           # Services & Use cases
│   └── infrastructure/        # Adapters (Neo4j, File System)
├── config/                    # Configuration templates
├── docs/                      # Step-by-step documentation
└── tests/                     # Unit & Integration tests
```

## Citation

```bibtex
@INPROCEEDINGS{11315354,
  author={Yigit, Ibrahim Onuralp and Buzluca, Feza},
  booktitle={2025 IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE)}, 
  title={A Graph-Based Dependency Analysis Method for Identifying Critical Components in Distributed Publish-Subscribe Systems}, 
  year={2025},
  volume={},
  number={},
  pages={1-9},
  keywords={Fault tolerance;Architecture;Fault tolerant systems;Microservice architectures;Publish-subscribe;Computer architecture;Solids;Loss measurement;Complexity theory;Topology;Distributed Systems;Publish-Subscribe Architecture;Graph Modeling;Dependency Analysis;Critical Components;Failure Impact},
  doi={10.1109/RASSE64831.2025.11315354}}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
