# Software-as-a-Graph Examples

This directory contains a progressive tutorial for using the Software-as-a-Graph (SAAG) framework. The examples transition from basic topological intuition to complex simulation-based validation.

## Prerequisites

- **Python 3.9+**
- **Neo4j 5.x** (default: `bolt://localhost:7687`)
- **Dependencies**: `pip install -r requirements.txt`

> [!NOTE]
> Most examples assume data has been imported into Neo4j. If you are just starting, run the `example_import.py` script first.

## Recommended Run Order

| Order | Script | Question Answered | Run Time |
| :--- | :--- | :--- | :--- |
| 0 | `example_introduction.py` | Why does topology predict risk? | 2s (No Neo4j) |
| 1 | `example_end_to_end.py` | How does the full 6-step pipeline work? | ~30s |
| 2 | `example_analysis.py` | Which components are my critical SPOFs? | ~5s |
| 3 | `example_antipatterns.py` | Can I block unsafe deployments in CI/CD? | ~5s |
| 4 | `example_simulation.py` | What is the real-world impact of a failure? | ~15s |
| 5 | `example_validation.py` | Are the predictions actually accurate? | ~10s |
| 6 | `example_compare.py` | Which architectural design is safer? | ~20s |
| 7 | `example_visualization.py` | How do I produce an executive report? | ~10s |

## Reading the Output

### Spearman Correlation (ρ)
- **ρ ≥ 0.80**: Excellent alignment. The architectural structure is the primary driver of system risk.
- **0.60 ≤ ρ < 0.80**: Good alignment, but non-topological factors (QoS, individual node logic) significantly influence impact.
- **ρ < 0.60**: The model may need domain-specific tuning for your topology.

### Criticality Levels
- **CRITICAL**: Immediate attention required. Structural Single Points of Failure (SPOFs) or massive cascade amplifiers.
- **HIGH**: Significant risk. Hub components with high betweenness or exposure.
- **MEDIUM/LOW**: Normal operational components.

## Domain-Specific Topologies
The `topologies/` directory contains hand-authored examples for specific domains:
- `ros2_autonomous_vehicle.json`: A realistic robotics system with mixed QoS.
- `kafka_event_pipeline.json`: A data pipeline with schema-registry dependencies.
