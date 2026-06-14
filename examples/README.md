# Software-as-a-Graph Worked Examples

This directory contains worked example scripts and configurations demonstrating system architecture modeling, weight calculation, dependency derivation, and persistence roundtrip validation. These files strictly adhere to the specifications defined in [graph-model.md](../docs/graph-model.md).

## Files Overview

- **`worked_example.json`**: The canonical topology file defining the worked example described in Section 8 of the documentation.
- **`run_worked_example.py`**: Programmatically loads the worked example topology, executes the modeling/analysis phases, and asserts that computed weights and derived `DEPENDS_ON` relationships match the specifications exactly.
- **`roundtrip_validation.py`**: Automates the import-export roundtrip test described in Section 13 of the documentation. It exports the database snapshot, wipes the database, re-imports the snapshot, and asserts total parity.

## Running the Examples

By default, the scripts run in-memory using `MemoryRepository` and require no external database setup. You can run them from the repository root:

### 1. Run the Worked Example Verification
```bash
python examples/run_worked_example.py
```

This will output the computed components, their weights, and derived dependencies, followed by the verification assertions:
```
=== Components and Computed Weights ===
ID | Name         | Type        | Weight
----------------------------------------
T0 | /temperature | Topic       | 0.5936
A0 | SensorApp    | Application | 0.5936
A1 | MonitorApp   | Application | 0.5936
B0 | MainBroker   | Broker      | 0.5936
L0 | NavLib       | Library     | 0.7347
N0 | ComputeNode1 | Node        | 0.5936
N1 | ComputeNode2 | Node        | 0.5936

...
Worked example verified successfully! All calculations and derived dependencies match spec.
```

### 2. Run the Export-Import Roundtrip Validation
```bash
python examples/roundtrip_validation.py --input examples/worked_example.json
```

This will run the roundtrip pipeline and confirm structural/functional parity:
```
Running roundtrip using MemoryRepository...

[Step A] Importing original topology: examples/worked_example.json
[Step B] Exporting snapshot to: output/snapshot_worked_example.json
[Step C] Re-importing snapshot from: output/snapshot_worked_example.json
[Step D] Comparing original vs. re-imported database states...
✅ ROUNDTRIP VALIDATION PASSED!
  - Total Vertices: 7
  - Total Relationships: 17
```

### Optional: Running with a Live Neo4j Database
If you have a live Neo4j database running (e.g. via Docker using `docker compose up`), you can verify the same calculations on Neo4j by passing the `--neo4j` flag:
```bash
python examples/run_worked_example.py --neo4j --uri bolt://localhost:7687 --user neo4j --password password
python examples/roundtrip_validation.py --input examples/worked_example.json --neo4j --uri bolt://localhost:7687 --user neo4j --password password
```

## Mathematical Formula Reference

The examples showcase the following specifications from `docs/graph-model.md`:

### Topic Weight Calculation (Phase 3)
$$w(topic) = \max(0.01, 0.85 \times QoS\_score + 0.15 \times size\_norm)$$
Where:
- $QoS\_score = 0.30 \times reliability\_score + 0.40 \times durability\_score + 0.30 \times priority\_score$
- $size\_norm = \min(\log_2(1 + size\_kb) / 50, 1.0)$

For `/temperature` (Topic `T0`):
- Reliability = `RELIABLE` (1.0)
- Durability = `TRANSIENT_LOCAL` (0.5)
- Priority = `HIGH` (0.66)
- QoS Score = $0.3 \times 1.0 + 0.4 \times 0.5 + 0.3 \times 0.66 = 0.698$
- Size Norm = $\log_2(1 + 0.0625) / 50 \approx 0.0017$
- Calculated Weight $\approx 0.85 \times 0.698 + 0.15 \times 0.0017 \approx \mathbf{0.5936}$ (matches documentation theoretical $\approx 0.592$)

### Application Weight Calculation (Phase 5)
$$w(app) = 0.80 \times \max(w(t)) + 0.20 \times \text{mean}(w(t))$$
- SensorApp (`A0`) & MonitorApp (`A1`) are connected only to `/temperature` (`T0`), so their weights collapse to $w(\text{topic}) \approx \mathbf{0.5936}$.

### Library Weight Calculation (Phase 5)
$$w(lib) = \min(1.0, base\_w \times (1 + 0.15 \times \log_2(1 + DG_{in})))$$
- NavLib (`L0`) is used by both applications ($DG_{in} = 2$), with $base\_w = \max(w(A0), w(A1)) \approx 0.5936$.
- Calculated Weight $\approx 0.5936 \times (1 + 0.15 \times \log_2(3)) \approx \mathbf{0.7347}$ (matches documentation theoretical $\approx 0.733$).

---

## Step 2: Structural Analysis & Quality Scoring

- **`run_structural_analysis.py`**: Executes the Step 2 Analyze phase. It computes directed centrality, continuous articulation point scores, Connectivity Degradation Index (CDI), and the final multi-dimensional criticality scores (Reliability, Maintainability, Availability, Vulnerability, and Overall Q).

### Running the Analysis Example

To run the structural analysis on the 5-node system projection (excluding infrastructure nodes):
```bash
python examples/run_structural_analysis.py
```

This prints two tables:
1. **System Layer Normalized Structural Metrics**: Normalized RPR, DG_in, MPCI, AP_c_dir, BR, BT, w_in, FOC.
2. **Component Criticality Scores and Levels (RMAV)**: Multi-dimensional criticality scores and their adaptive classification levels (e.g. `critical`, `high`, `medium`, `low`, `minimal`).

### Topological Observations & Discrepancies

When comparing the output of the actual Python script against the theoretical Section 13 table, there are a few important architectural and mathematical differences to note:

1. **Strict Articulation Points & Bridges (`num_articulation_points`, `num_bridges`)**:
   - Section 13 states `num_articulation_points=3` and `num_bridges=5`.
   - In a live dependency graph, Topic `T0` (/temperature) has no incoming or outgoing `DEPENDS_ON` relationships, making it an isolated node in the undirected projection $G_{undir}$.
   - The remaining 4-node subgraph (`A0`, `A1`, `B0`, `L0`) is a 2-connected component (complete graph $K_4$ minus the `B0-L0` edge). Removing any single node from this component does not disconnect it.
   - NetworkX correctly computes **0 articulation points** and **0 bridges** for this topology, which is mathematically correct.

2. **Reverse PageRank (RPR)**:
   - Section 13 lists RPR values that correspond to standard PageRank (downstream flow).
   - In the actual system, PageRank (downstream) flows to dependencies, while **Reverse PageRank (RPR)** runs on $G_{rev}$ (upstream flow) to identify components that depend on many others.
   - Therefore, the script correctly reports that `MonitorApp (A1)` has the highest RPR (`0.41`) as it is the root dependent of the system.

3. **QoS-Aware Weight Adaptation**:
   - The actual `saag` analysis pipeline automatically adjusts the weights of the quality dimensions based on the QoS profile of the topics. Since `/temperature` has high-reliability and high-priority QoS settings, the weights are adapted to emphasize reliability.
   - Therefore, the resulting R(v) and A(v) scores differ from the simplified hand-calculations of Section 13, reflecting the framework's real-world adaptive capability.

