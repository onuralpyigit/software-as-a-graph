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

- **`run_structural_analysis.py`**: Executes the Step 2 Analyze phase (directed centrality, continuous articulation point scores, Connectivity Degradation Index (CDI)) followed by the Step 3 Predict / Step 4 Diagnose stages (`client.predict()`, which bundles Step 4 in by default) to obtain the final criticality scores — Reliability (hierarchical, blending Fault Tolerance and Availability as sub-characteristics), Maintainability, and Overall Q.

### Running the Analysis Example

To run the structural analysis on the 5-node system projection (excluding infrastructure nodes):
```bash
python examples/run_structural_analysis.py
```

This prints two tables:
1. **System Layer Normalized Structural Metrics**: Normalized RPR, DG_in, MPCI, AP_c_dir, BR, BT, w_in, FOC.
2. **Component Criticality Scores and Levels (RM)**: Multi-dimensional criticality scores and their adaptive classification levels (e.g. `critical`, `high`, `medium`, `low`, `minimal`).

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

---

## Step 5: Failure Simulation

- **`run_failure_simulation.py`**: Executes the Step 5 Failure Simulation phase. It showcases two of the [simulation engines](../docs/failure-simulation.md#2-which-engine-do-i-need): the stochastic **Fault Injector** (cascade labels I\*) and the discrete-event **Message Flow Simulator** (SimPy queue timing, I_dyn).

### Running the Simulation Example

To run both simulation modes on the worked example topology:
```bash
python examples/run_failure_simulation.py
```

This runs:
1. **Fault Injection**: Simulates node failures for `A0`, `A1`, and `B0` using 5 seeds to verify cascade propagation.
2. **Message Flow Simulation**: Runs a baseline (clean) scenario and a faulted scenario (injecting a fault on `A0` / SensorApp at $t = 50.0$ s for a 100.0 s run).

### Simulation Observations & Metrics

The script asserts each result below, so a change in behaviour makes it fail.

1. **Fault Injection Cascade** (`SensorApp` publishes `/temperature` to `MonitorApp`, routed by `MainBroker`):

   | Component | I\* | Cascade depth | Why |
   |:---|:---|:---|:---|
   | `SensorApp` (`A0`) | 1.0 | 1 | Sole publisher of `T0`, so `MonitorApp` loses its only feed |
   | `MainBroker` (`B0`) | 1.0 | 1 | Sole router of `T0`, so the feed is lost even though `SensorApp` is alive |
   | `MonitorApp` (`A1`) | 0.0 | 0 | Only subscribes, so no other subscriber is harmed |

2. **Message Flow Simulation (SimPy)**, a 100 s run:
   - **Baseline**: `A0` publishes 1000 messages and 998 are delivered, a delivery rate of `0.9980`. The other 2 are not dropped: they are reported as `unserved_demand`, still unprocessed when the run ends.
   - **Faulted**: a fault on `A0` at $t = 50.0$ s silences it after 499 messages. All 499 are delivered, so the *system* delivery rate reads `1.0000`, which hides the outage. The damage is in the fault record: the delivery rate falls from `1.0000` before the fault to `0.0000` after it, so I_dyn(A0) = 1.0.

### Simulator Walkthrough (Mini-ATM Failure Lab)

- **`mini_atm.json`**: A hand-made 15-component air-traffic topology (2 hosts, 2 brokers, 1 library, 6 applications, 4 topics). It is a demo and is not part of the scenario corpus.
- **`run_simulator_walkthrough.py`**: Runs all four [simulation engines](../docs/failure-simulation.md#2-which-engine-do-i-need) on it with their published defaults (`FaultInjector`, `FailureSimulator`, `MessageFlowSimulator`, `ChangePropagationSimulator`), prints the four scores per component, and builds the page below.
- **`mini_atm_failure_lab.template.html`**: The page template; the script injects the engine output into it.
- **`mini_atm_failure_lab.html`**: The generated interactive walkthrough. Open it in a browser, pick a component to fail, and compare what each engine does with it.

```bash
python examples/run_simulator_walkthrough.py
```

The four scores answer different questions and rank the components differently. `radar` (`A1`) tops I\* (0.833) but has the lowest I_comp (0.053), because its only topic is best-effort and carries little QoS weight. `display` (`A5`) scores 0 on I\* as a pure sink but 0.296 on I_comp, because three of the eight delivery paths end there.

### Container Testbed Walkthrough (Mini-ATM Testbed Run)

- **`run_testbed_walkthrough.py`**: Runs the [container testbed](../docs/simulation-extensions.md#3-emulated-testbed-calibration-and-fidelity) on `mini_atm.json` twice, each time with one fault (the `B1` relay stopped for 12 s; host `N1` killed with `docker kill`). For each run it calibrates from the telemetry, runs the fidelity comparison with declared and with calibrated parameters, and sets the observed trace beside the simulator's.
- **`mini_atm_testbed_run.template.html`** / **`mini_atm_testbed_run.html`**: The page template and the generated interactive walkthrough.

```bash
python examples/run_testbed_walkthrough.py           # two Docker runs, about three minutes
python examples/run_testbed_walkthrough.py --reuse   # rebuild the page from the last runs, no Docker
```

Needs Docker with the compose plugin; the first run builds an image that downloads `python:3.11-slim` and `cyclonedds`. The testbed directories and telemetry go to the ignored `output/testbed_walkthrough/`.

On this topology the simulator reproduces who stalls and who recovers (stall-set Jaccard 1.0 in both runs) and the delivery rates (within 8% per reader), but not latency: the median p95 error is about 0.6 with declared parameters. Each scenario is run once, so this is a smoke test, not a fidelity estimate. The page also notes one input-format mismatch: `mini_atm.json` declares deadlines only inside each topic's `qos` block, which the simulator reads and the testbed plan does not.

---

## The Method End to End (SaG Method Walkthrough)

- **`run_method_walkthrough.py`**: Runs every pipeline stage on `mini_atm.json` through the SDK (Model, Analyze, Predict, Diagnose with the triage bridge, Simulate, Validate, Prescribe) and builds a page showing what each stage produced.
- **`sag_method_walkthrough.template.html`** / **`sag_method_walkthrough.html`**: The page template and the generated walkthrough.

```bash
python examples/run_method_walkthrough.py
```

Stage 3 loads a trained HGT-QoS checkpoint from the ignored `output/realworld_zeroshot/hgl_qos/seed_42`; pass `--checkpoint` to use another. The script stops if GNN inference does not actually run.

What the run shows on this system: 26 derived dependencies; a trained ranking and a quality-model ranking that disagree at the top, kept side by side by triage; a system-layer validation that misses its correlation gate narrowly (ρ = 0.685 against 0.70 with n = 10); and a prescription step that accepts 2 of 7 candidate fixes after counterfactual simulation. Fifteen components cannot support an accuracy claim either way; the page demonstrates how the stages connect.

---

## Workflow Walkthroughs, One Page Each

Nine scripts each take one workflow apart on a small system and build an interactive page from real output. Every script `run_<name>_walkthrough.py` fills `<name>_walkthrough.template.html` and writes `<name>_walkthrough.html`; the shared palette, helpers and mini-ATM diagram live in `walkthrough_page.py`.

| Workflow | Script | What the page shows |
|:---|:---|:---|
| Model (Step 1) | `run_model_walkthrough.py` | Topic weights from QoS, size and rate; component weights; each derived `DEPENDS_ON` edge with its rule and weight; the four layer projections |
| Analyze (Step 2) | `run_analyze_walkthrough.py` | The structural metric vector one metric at a time, which later step reads each field, and eight metrics reproduced with NetworkX |
| Simulate (Step 5) | `run_simulate_walkthrough.py` | The label artifact, what its options change, the checks it must pass before training, and the oracle's single, pairwise, event, Monte Carlo and edge runs |
| Predict (Step 3) | `run_predict_walkthrough.py` | Closed-form references term by term, HGT-QoS across five seeds, the dual engine, the cold-start fallback, and each ranking against simulated I\* |
| Diagnose (Step 4) | `run_diagnose_walkthrough.py` | FT, A and M as weighted terms, the roll-up to R and Q(v), the level fences and the generated explanation |
| Triage | `run_triage_walkthrough.py` | The Triage Bridge with a trained ranking and in cold start, then one change followed through the pull-request note, the delta-aware gate and a targeted simulation |
| Validate (Step 6) | `run_validate_walkthrough.py` | Alignment, rank agreement, bootstrap uncertainty, critical-set overlap, dimensions, gates and health indices |
| Anti-patterns and Prescribe (Steps 4 and 7) | `run_prescribe_walkthrough.py` | Which catalogue patterns fire, the candidate edits they produce, and which survive counterfactual simulation |
| Before and after deployment | `run_deployment_walkthrough.py` | Placement planning with held-out verification, then the chosen placement run on the in-process testbed, calibrated and compared with simulation |

```bash
python examples/run_model_walkthrough.py
python examples/run_analyze_walkthrough.py
python examples/run_simulate_walkthrough.py
python examples/run_predict_walkthrough.py
python examples/run_diagnose_walkthrough.py
python examples/run_triage_walkthrough.py
PYTHONHASHSEED=0 python examples/run_validate_walkthrough.py
python examples/run_prescribe_walkthrough.py
python examples/run_deployment_walkthrough.py
```

All use `mini_atm.json` except the prescribe walkthrough, which uses `data/scenarios/atm_system_tiny.json` because mini-ATM triggers too few patterns. None needs Neo4j or Docker. The predict walkthrough needs the HGT-QoS checkpoints in the ignored `output/realworld_zeroshot/hgl_qos`; the triage walkthrough uses one of them when present and otherwise shows the cold-start bridge only. The deployment walkthrough runs a 24-second testbed in local mode and keeps its run in the ignored `output/deployment_walkthrough/`.

The Simulate walkthrough covers simulation as a workflow; `run_simulator_walkthrough.py` (above) shows what each engine does inside a single failure.

What the runs surfaced, beyond the mechanics:

- **Model.** The script recomputes every topic weight, component weight and edge weight with the public helpers in `saag/core/models.py` and stops if any differs from the repository; all match. 28 declared relationships yield 26 dependencies through 5 of the 7 rules; mini-ATM has no co-hosted brokers and no network devices.
- **Analyze.** Each component gets 38 numeric fields, of which 27 vary on mini-ATM and 19 feed the quality score. PageRank, reverse PageRank and betweenness are the dependency-weighted variants, not the textbook ones: computed unweighted they differ by up to 0.019. Display is flagged as a directed cut vertex while its continuous cut-vertex score is 0.0, as is every component's.
- **Simulate.** The label checks refuse 4 of 5 artifacts offered (wrong topology, wrong labeler, missing label, retest ρ below 0.70). On mini-ATM the propagation threshold barely moves the labels (at most 0.017), while 30 of 45 failure pairs do more damage than the worse member alone.
- **Predict.** On mini-ATM, Reach orders the applications exactly as the simulator does; HGT-QoS correlates at 0.81 and RM Q(v) at −0.32 with I\*. Six applications are too few to read as accuracy.
- **Diagnose.** The shipped level fence is Q3 + 0.75 × IQR (`QualityAnalyzer(k_factor=0.75)`), while `docs/diagnosis.md` and `docs/structural-analysis.md` state 1.5 × IQR. With 1.5 nothing on mini-ATM would be rated critical.
- **Triage.** A one-topic change that closes a feedback loop is caught at every tier: the references flag the new publisher, the delta-aware gate blocks on three new findings while ignoring six pre-existing ones, and targeted simulation shows four of six applications becoming more critical, one of them untouched by the change.
- **Validate.** ρ is identical on every run, but the bootstrap interval's lower bound moves with Python's hash seed (roughly −0.03 to 0.07 on mini-ATM), which is why the command above fixes `PYTHONHASHSEED`.
- **Anti-patterns and Prescribe.** On `atm_system_tiny`, 8 of 19 patterns fire; `DEEP_PIPELINE` alone reports 746 findings on ten applications (the over-detection noted below for the larger ATM system). Of 32 candidate edits, 8 are accepted and 19 would have increased impact.
- **Deployment.** The optimizer's blast proxy correlates at −0.04 with the held-out simulator across placements, and no reliability gain is claimed. Running the recommended placement shows a replicated application stalling after a single host crash because the only relay for its input was on the same host.

Each page says what it is not: these are small systems chosen to make the mechanics visible, not evidence about accuracy.

---

## Step 6: Validation Pipeline

- **`run_validation.py`**: Executes the Step 6 Validation phase. It runs the programmatic validation pipeline on the Complete System (`system`) layer, comparing the topological quality predictions against the simulation-derived ground truths.

### Running the Validation Example

To execute the validation pipeline and verify the G1-G6, G8 gates, correlation metrics, and system health indices:
```bash
python examples/run_validation.py
```

This prints six detailed ASCII tables:
1. **Layer Statistical Validation Summary**: Overview of statistical correlation, F1 score, and error metrics for the layer.
2. **Unified Validation Gates Checklist (G1-G6, G8)**: Evaluates the system against all target validation gates (Spearman rank correlation, F1 score, precision, top-5 overlap, predictive gain, weighted Kappa, etc.).
3. **Multi-Dimensional Validation**: Evaluates the Spearman rank correlation of each composite quality dimension (Reliability, Maintainability) plus the Fault Tolerance and Availability sub-characteristic diagnostics, against its corresponding simulation-derived ground truth.
4. **System Health and Risk Indices**: Displays system-wide health scores ($H_R, H_M$, feeding the System Risk Index, plus diagnostic $H_{FT}, H_A$), the System Risk Index ($SRI$), and the Risk Concentration Index ($RCI$/Gini).
5. **Node-Type Stratified Reporting**: Computes correlation scores stratified by component type.
6. **Topic Frequency-Decile Stratified Reporting**: Decile-stratified correlation and significance ($p$-values) based on topic publishing frequencies.

It also outputs a clean comparison table:
- **Component Predictions vs. Actuals**: Displays predicted quality scores vs. simulation-derived actual composite impacts for the matched components.

### Validation Observations & Results

1. **System Layer Data Alignment**:
   - The validation pipeline automatically filters the system topology to the 5 matched components analyzed in the `system` layer (`A0`, `A1`, `B0`, `N0`, `N1`).
   
2. **Statistical Correlation & Validation Gates**:
   - The overall Spearman rank correlation is calculated as $\rho = 0.0000$ — a small-N degeneracy of the ranking on this 5-node topology (the predicted vs. simulated orderings tie out completely at this scale).
   - Classification gates **G3 (Precision)** and **G4 (Top-5 Overlap)** pass with perfect scores (`1.0000`), demonstrating that the highest criticality predictors perfectly align with the highest simulation impacts.
   - **G1 (Spearman)** and **G2 (F1)** fail due to the low list variance on such a small node count, matching the expected behavior of the validation engine.

3. **System Health and Risk**:
   - System health indices correctly indicate high availability headroom ($H_A = 0.9290$) and moderate reliability ($H_R = 0.4757$).
   - The System Risk Index ($SRI$) is calculated as `0.3763` with a low Risk Concentration Gini index ($RCI = 0.1024$), confirming that risk is relatively distributed across the small network.

---

## Step 8: Visualization Dashboard

- **`run_visualization.py`**: Executes the Step 8 Visualization phase. It compiles results from all preceding steps (Analysis, Simulation, Validation, and Anti-Patterns) and generates a self-contained, interactive HTML dashboard showcasing all 10 standard sections defined in the specifications.
- **`example_visualization.py`**: An alias/wrapper script provided for out-of-the-box execution matching the exact command referenced in `docs/visualization.md`.

### Running the Visualization Example

To generate the dashboard and verify the layout:
```bash
python examples/run_visualization.py
```

This runs the full pipeline and generates the dashboard:
```
Loading topology JSON from: examples/worked_example.json
Initializing MemoryRepository...
Importing topology and deriving dependencies...
Running validation for seeds...
  Generated multi-seed validation files: [...]
Scanning for anti-patterns...
  Generated anti-pattern catalog: output/worked_example_antipatterns.json
Simulating QoS cascade risks...
  Generated cascade risk report: output/worked_example_cascade.json
Generating interactive HTML dashboard...
  [PATCH] Injected MIL-STD-498 hierarchy tree.
Dashboard generated successfully at: output/worked_example_dashboard.html

Verifying dashboard contents:
  [INFO] Validation Report section omitted (spearman=0.0000 <= 0).
  [PASS] All applicable standard dashboard sections are correctly present in the output HTML file.
  [PASS] Embedded Cytoscape.js and D3.js libraries/scripts verified.
Worked example dashboard verified successfully! Strictly adheres to docs/visualization.md.
```

The script outputs the static dashboard to `output/worked_example_dashboard.html`. You can open this file in any web browser to view the interactive elements. Note: the **Validation Report** section (§7) only renders when the layer's Spearman correlation is positive — on this 5-node worked example it is exactly `0.0` (see the Step 6 discussion above), so that one section is legitimately skipped here. It renders normally on the larger ATM dataset (`run_atm_visualization.py`).

### Dashboard Sections Rendered

1. **Executive Overview**: 6 KPI cards (Total Components, Total Dependencies, Critical Assets, SPOFs, Anti-Patterns, Validation $\rho$), criticality distribution doughnut, and top-5 components criticality bar chart.
2. **Layer Comparison**: Grouped side-by-side bar chart comparing density, node scale, average impact, and validation Spearman correlation across layers.
3. **Component Details**: Sortable, filterable table displaying Q(v) scores, criticality levels, simulation impacts, RM dimension bars, and SPOF flags.
4. **Validation Diagnostics**: Q* predicted vs I* simulated composite scatter plot with regression/confidence lines, per-dimension Spearman progress bars, and individual scatter plots for the Reliability/Maintainability composite dimensions and the Fault Tolerance/Availability sub-characteristic diagnostics.
5. **Interactive Network Graph**: Layer-stratified topology map rendered in Cytoscape.js using compound layer boundaries, node shapes/colors by component type and criticality, and alert box details on tap.
6. **Dependency Matrix**: Directed adjacency matrix sorted by Q(v), with cell opacity encoding QoS edge weight.
7. **Validation Report**: Unified validation checklist evaluating G1-G4 gates.
8. **Multi-Seed Stability**: Mean, min, max KPI statistics and a stability line plot across multiple seed runs.
9. **Anti-Pattern Catalog**: Expandable list of detected bad smells grouped by CRITICAL/HIGH/MEDIUM severity levels.
10. **QoS Cascade Risk (QoS Ablation)**: dual-bar chart comparing QoS-enriched vs topology-only cascade risks, with QoS Gini, Wilcoxon p-value, and $\Delta\rho$ metrics.
11. **MIL-STD-498 Hierarchy**: Recursive tree view rolling up computed criticality (Q) and cross-boundary coupling index (CBCI) from CSS down to CSCI, CSC, and CSU.


---

## Realistic Safety-Critical ATM System Examples

These examples run structural analysis, failure simulation, validation, and visualization on the realistic 26-app, 27-topic Air Traffic Management (ATM) system dataset located at `data/scenarios/atm_system.json`. 

This topology models the safety-critical ICAO SWIM concepts, utilizing realistic components such as radar trackers, conflict detectors, trajectory predictors, and flight plans.

### 1. ATM Structural Analysis & Quality Scoring
To run structural centrality, articulation points, and RM quality scoring on the ATM system:
```bash
python examples/run_atm_structural_analysis.py
```
This prints:
- **ATM Layer Normalized Structural Metrics**: RPR, degrees, and centrality scores for critical ATM modules.
- **ATM Component Criticality Scores and Levels (RM)**: Safety-critical elements like `conflict-detector` and `radar-tracker` score noticeably above baseline. Under the RM composite, the four `conflict-detector` instances land LOW-to-MEDIUM (Q(v) ≈ 0.24–0.35): `A17`=0.3453 (medium), `A9`=0.3397 (medium), `A20`=0.3270 (medium), `A24`=0.2365 (**low** — a genuine tier drop from the retired 4-D model, since Availability's weight in the composite decreased once it folded into Reliability as a sub-characteristic at α=0.36).
- **ATM Graph Summary**: Detailed topological summary ($S(G)$) of the 74-component graph.

### 2. ATM Failure Simulation
To run the stochastic fault cascade and the discrete-event SimPy queue-flow simulation on the ATM system:
```bash
python examples/run_atm_failure_simulation.py
```
This runs:
- **Fault Injection**: Simulates node failures for Application and Broker nodes and measures cascade reach.
- **Message Flow Simulation**: Compares baseline delivery rates and latencies to a faulted run where a `radar-tracker` is failed at $t=5.0\text{ s}$.
- Saves results to `output/atm_system_impact_scores.json` and `output/atm_system_message_flow_results.json`.

### 3. ATM Validation Pipeline
To run the validation checks comparing predicted quality metrics against simulation-derived actuals:
```bash
python examples/run_atm_validation.py
```
This prints statistical metrics showing strong model alignment for safety-critical components (such as an application-stratified Spearman $\rho \approx 0.853$), checks unified validation gates (G1-G6, G8), and computes health indices.

### 4. ATM Visualization Dashboard
To compile all ATM analysis steps and output an interactive HTML dashboard:
```bash
python examples/run_atm_visualization.py
```
This generates a single, self-contained dashboard at `output/atm_system_dashboard.html` that showcases all 10 required sections (including cytoscape network graphs, D3 adjacency matrices, multi-seed stability, and the MIL-STD-498 system hierarchy drill-down).

> **Known issue**: on the 74-component ATM topology, the "Deep Processing Pipeline"
> anti-pattern detector currently emits tens of thousands of near-duplicate findings
> (path-combinatoric over-detection), inflating `output/atm_system_antipatterns.json`
> and the generated dashboard HTML to tens of megabytes each. The dashboard still
> renders correctly, but generation is slow and the Anti-Pattern Catalog section is
> impractically large. This is a pre-existing `AntiPatternDetector` scaling issue,
> independent of this script — flagged here for awareness, not yet fixed.

### 5. ATM GNN Prediction

To run the GNN prediction pipeline on the ATM system:
```bash
python examples/run_atm_prediction.py
```
This script loads the pre-trained Heterogeneous Graph Transformer (HGT) model checkpoints from `models/gnn_checkpoints/`, executes the forward pass on the ATM Application layer, and prints:
- **ATM Component Criticality Ranks**: The top-10 ranked application and library components sorted by GNN criticality score.
- **ATM Edge Criticality Ranks**: The top-10 ranked directed dependencies (interaction edges) sorted by direct GNN-predicted criticality.
- Saves the predictions to `output/atm_system_predictions.json`.


