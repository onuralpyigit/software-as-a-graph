# Software-as-a-Graph

**Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-green.svg)](https://neo4j.com/)
[![IEEE RASSE 2025](https://img.shields.io/badge/IEEE-RASSE%202025-orange.svg)](#citation)

## Overview

This framework predicts which components in a distributed system are most critical—those whose failure would cause the greatest impact—using only the system's architectural structure.

**Key insight**: Graph topological metrics can reliably predict component criticality *before* deployment, without expensive runtime monitoring.

### Validation Results

| Metric | Application Layer | Infrastructure Layer |
|--------|-------------------|----------------------|
| Spearman ρ | **0.85** | 0.54 |
| F1-Score | **0.83** | 0.68 |
| Top-5 Overlap | **62%** | 40% |
| Speedup | **2.2×** faster | 1.2× |

---

## The Six-Step Methodology

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Step 1: GRAPH MODEL           System → Weighted directed graph │
│          CONSTRUCTION          with derived dependencies        │
│                 ↓                                               │
│  Step 2: STRUCTURAL            Compute PageRank, Betweenness,   │
│          ANALYSIS              Articulation Points, etc.        │
│                 ↓                                               │
│  Step 3: QUALITY               Calculate R, M, A, V → Q(v)      │
│          SCORING               using AHP-weighted formulas      │
│                 ↓                                               │
│  Step 4: FAILURE               Simulate failures, measure       │
│          SIMULATION            actual impact I(v)               │
│                 ↓                                               │
│  Step 5: VALIDATION            Compare predicted Q(v) vs I(v)   │
│                 ↓                                               │
│  Step 6: VISUALIZATION         Generate interactive dashboards  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Prerequisites: Python 3.9+, Neo4j 5.x

# Setup
git clone https://github.com/your-org/software-as-a-graph.git
cd software-as-a-graph
pip install -r requirements.txt
docker-compose up -d neo4j

# Run complete pipeline
python run.py --all --layer system
```

Or step by step:

```bash
python generate_graph.py --scale medium --output data/system.json
python import_graph.py --input data/system.json --clear
python analyze_graph.py --layer system --use-ahp
python simulate_graph.py --exhaustive --layer system
python validate_graph.py --layer system
python visualize_graph.py --layer system --output dashboard.html --open
```

---

## Documentation

| Step | Document | Description |
|------|----------|-------------|
| 1 | [Graph Model Construction](docs/graph-model.md) | Transform system topology into a weighted graph |
| 2 | [Structural Analysis](docs/structural-analysis.md) | Compute topological metrics |
| 3 | [Quality Scoring](docs/quality-scoring.md) | Calculate R, M, A, V scores with AHP weights |
| 4 | [Failure Simulation](docs/failure-simulation.md) | Measure actual failure impact |
| 5 | [Validation](docs/validation.md) | Compare predictions against ground truth |
| 6 | [Visualization](docs/visualization.md) | Generate interactive dashboards |

---

## Project Structure

```
software-as-a-graph/
├── src/
│   ├── core/                  # Graph model, import/export
│   ├── analysis/              # Metrics, quality scoring, AHP
│   ├── simulation/            # Failure simulation
│   ├── validation/            # Statistical validation
│   └── visualization/         # Dashboard generation
├── docs/                      # Step-by-step documentation
├── generate_graph.py          # Generate synthetic data
├── import_graph.py            # Import to Neo4j
├── analyze_graph.py           # Run analysis
├── simulate_graph.py          # Failure simulation
├── validate_graph.py          # Statistical validation
├── visualize_graph.py         # Generate dashboards
├── run.py                     # Pipeline orchestrator
└── benchmark.py               # Benchmark suite
```

---

## Citation

```bibtex
@inproceedings{software-as-a-graph-2025,
  title={Graph-Based Critical Component Prediction for Distributed Pub-Sub Systems},
  booktitle={IEEE RASSE 2025},
  year={2025}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
