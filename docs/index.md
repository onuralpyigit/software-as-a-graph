# Software-as-a-Graph Documentation

## Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems

Welcome to the documentation for the Software-as-a-Graph research project. This toolkit implements a six-step methodology for identifying critical components in distributed publish-subscribe systems using graph theory and structural analysis.

---

## Documentation Structure

| Document | Description |
|----------|-------------|
| [Methodology Overview](methodology.md) | Complete six-step methodology explanation |
| [Graph Model](graph-model.md) | Multi-layer graph representation |
| [Structural Analysis](analysis.md) | Centrality metrics and criticality scoring |
| [Failure Simulation](simulation.md) | Impact measurement through simulation |
| [Statistical Validation](validation.md) | Comparing predictions with actuals |
| [Visualization](visualization.md) | Dashboards and graph rendering |
| [API Reference](api-reference.md) | Python module documentation |
| [CLI Reference](cli-reference.md) | Command-line tool usage |

---

## Quick Overview

### The Problem

Modern distributed systems use publish-subscribe (pub-sub) messaging for communication. These systems can have:
- Hundreds of applications
- Thousands of topics
- Complex multi-layer dependencies

Traditional approaches to identifying critical components rely on expert intuition or runtime monitoring—both have significant limitations.

### Our Solution

We model pub-sub systems as **multi-layer graphs** and apply **graph algorithms** to predict component criticality:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   STEP 1     │────▶│   STEP 2     │────▶│   STEP 3     │
│ Graph Model  │     │  Structural  │     │ Criticality  │
│ Construction │     │  Analysis    │     │   Scoring    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│   STEP 6     │◀────│   STEP 5     │◀────│   STEP 4     │
│Visualization │     │  Statistical │     │   Failure    │
│              │     │  Validation  │     │  Simulation  │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Key Insight

> **Topological structure predicts system behavior.** Components on many shortest paths (high betweenness centrality) or whose removal disconnects the graph (articulation points) are structurally critical—and this correlates with actual failure impact.

---

## Research Targets

| Metric | Target | Description |
|--------|--------|-------------|
| **Spearman ρ** | ≥ 0.70 | Rank correlation between predicted and actual criticality |
| **F1-Score** | ≥ 0.90 | Classification accuracy for critical components |
| **Precision** | ≥ 0.80 | Fraction of predicted critical that are actually critical |
| **Recall** | ≥ 0.80 | Fraction of actual critical correctly identified |
| **Top-5 Overlap** | ≥ 60% | Agreement on most critical components |

---

## Getting Started

### Installation

```bash
git clone https://github.com/onuralpyigit/software-as-a-graph.git
cd software-as-a-graph
pip install networkx
```

### Quick Demo

```bash
python run.py --quick
open output/dashboard.html
```

### Python API

```python
from src.core import generate_graph
from src.simulation import SimulationGraph
from src.validation import ValidationPipeline

# Generate graph
data = generate_graph(scale="small", scenario="iot", seed=42)
graph = SimulationGraph.from_dict(data)

# Run validation pipeline
pipeline = ValidationPipeline(seed=42)
result = pipeline.run(graph)

print(f"Spearman: {result.validation.correlation.spearman:.4f}")
print(f"F1-Score: {result.validation.classification.f1:.4f}")
```

---

## Publication

**Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems**  
Ibrahim Onuralp Yigit  
IEEE RASSE 2025

```bibtex
@inproceedings{yigit2025graph,
  title={Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems},
  author={Yigit, Ibrahim Onuralp},
  booktitle={IEEE RASSE},
  year={2025}
}
```

---

## Navigation

- **Next:** [Methodology Overview →](methodology.md)
