# CLI Pipeline Execution Guide

## Prerequisites & Setup

### Neo4j

Start a local Neo4j instance (default Bolt URI `bolt://localhost:7687`, user `neo4j`, password `password`). Ensure the database is reachable before running stages that require it (Steps 1–6 unless `--input` is provided for file-mode utilities).

### Python Environment

All CLI scripts must be invoked from the project root with the project root on `PYTHONPATH`:

```bash
PYTHONPATH=. python cli/<script>.py [args]
```

Some scripts (e.g. `generate_graph.py`) add the project root to `sys.path` when run directly, but `PYTHONPATH=.` is required for consistent imports across every stage.

### Dependencies

Install the project's Python dependencies (including `torch`, `torch-geometric`, `networkx`, `numpy`, `scipy`, `pyyaml`, `saag`).

### Environment Variables

Neo4j credentials can be set via environment variables to avoid repeating flags:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

All CLI scripts read these via `cli/common/arguments.py`.

---

## Quick Start

First-time runs should train a GNN checkpoint before using `--all`:

```bash
# 1. Generate + import + analyze + simulate
PYTHONPATH=. python cli/run.py --all --layer system

# 2. Train GNN checkpoint
PYTHONPATH=. python cli/train_graph.py --layer system --output output/gnn_checkpoints/best_model

# 3. Full pipeline with predictions
PYTHONPATH=. python cli/run.py --all --layer system --gnn-model output/gnn_checkpoints/best_model
```

The master orchestrator `cli/run.py --all` emits a first-run guard warning if `--gnn-model` is absent and no default checkpoint exists at `output/gnn_checkpoints/best_model`.

---

## Stage 0: Offline Prep — Generate

**Script:** `cli/generate_graph.py`  
**Purpose:** Produce synthetic pub-sub topology JSON files.

```bash
PYTHONPATH=. python cli/generate_graph.py --scale medium --output output/graph.json
```

or with a domain:

```bash
PYTHONPATH=. python cli/generate_graph.py --scale small --domain atm --output data/atm_system.json
```

### Arguments

| Flag | Default | Choices / Type | Description |
|------|---------|----------------|-------------|
| `--scale` | `None` (falls back to `medium`) | `tiny`, `small`, `medium`, `large`, `jumbo`, `xlarge` | Scale preset |
| `--config` | `None` | Path | YAML configuration file (mutually exclusive with `--scale`) |
| `--output` | `output/graph.json` | Path | Output JSON path |
| `--seed` | `42` | int | Random seed |
| `--domain` | `None` | `av`, `iot`, `finance`, `healthcare`, `hub-and-spoke`, `microservices`, `enterprise`, `atm` | Realistic naming domain |
| `--scenario` | `None` | same as `--domain` | Scenario mapping for QoS generation |
| `--connection-density` | `0.3` | float | Probability of `connects_to` edges |
| `--verbose` / `-v` | `False` | flag | Debug logging |
| `--quiet` / `-q` | `False` | flag | Suppress console output |

### Subcommands

- **`batch`** — generate multiple scenario datasets.
  ```bash
  PYTHONPATH=. python cli/generate_graph.py batch --input-dir data/ --output-dir output/ --seeds 42,123,456 --manifest batch_manifest.json --report batch_report.json
  ```
- **`validate`** — topology-class validation for scenarios.
  ```bash
  PYTHONPATH=. python cli/generate_graph.py validate --input-dir output/ --report validation_report.json
  ```

### Output

- `output/graph.json` (or path given by `--output`)

---

## Step 1: Model — Import & Export

**Scripts:** `cli/import_graph.py`, `cli/export_graph.py`  
**Purpose:** Load JSON topology into Neo4j, derive `DEPENDS_ON` relationships, and optionally export back.

### Import

```bash
PYTHONPATH=. python cli/import_graph.py --input output/graph.json --clear --dry-run
```

**Arguments** (shared Neo4j + runtime flags apply):

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Input JSON file |
| `--clear` | `False` | Clear existing DB before import |
| `--dry-run` | `False` | Validate input without importing |
| `--output` / `-o` | `None` | Save import statistics JSON |

Default Neo4j connection flags: `--uri bolt://localhost:7687`, `--user neo4j`, `--password password`.

### Export

```bash
PYTHONPATH=. python cli/export_graph.py --output output/exported_graph.json --format analysis --layer system --include-structural
```

**Arguments:**

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--output` / `-o` | required | Path | Output JSON path |
| `--format` | `persistence` | `persistence`, `analysis` | Export format |
| `--include-structural` | `False` | flag | Include `RUNS_ON`, `ROUTES`, etc. in `analysis` format |
| `--layer` | `system` | `app`, `infra`, `mw`, `system` | Layer for `analysis` format |

---

## Step 2: Analyze

**Script:** `cli/analyze_graph.py`  
**Purpose:** Compute structural metrics, RMAV/Q quality scores, and detect anti-patterns.

```bash
PYTHONPATH=. python cli/analyze_graph.py --layer system
```

### Arguments

| Flag | Default | Choices / Type | Description |
|------|---------|----------------|-------------|
| `--layer` | `system` | `app`, `infra`, `mw`, `system`, `all`, comma-separated | Analysis layer(s). `all` runs `app`, `infra`, `mw`, `system` sequentially. |
| `--use-ahp` | `False` | flag | Use AHP-derived weights |
| `--equal-weights` | `False` | flag | Equal 0.25 weights per dimension (baseline) |
| `--ahp-shrinkage` | `0.7` | float | Shrinkage factor λ ∈ [0,1] |
| `--norm` | `robust` | `robust`, `minmax`, `zscore`, `rank` | Normalization method |
| `--winsorize` | `False` | flag | Cap extreme outliers |
| `--sensitivity` | `False` | flag | Run weight sensitivity analysis |
| `--output` / `-o` | `None` | Path | Save analysis results JSON |

Multi-layer output filenames append the layer (e.g. `analysis_app.json`, `analysis_system.json`).

---

## Step 3: Predict

**Script:** `cli/predict_graph.py`  
**Purpose:** Unified prediction — RMAV scoring, optional GNN inference, and anti-pattern detection.

```bash
PYTHONPATH=. python cli/predict_graph.py --layer system --gnn-model output/gnn_checkpoints/best_model
```

### Arguments

**Weighting:**

| Flag | Description |
|------|-------------|
| `--use-ahp` | AHP-derived dimension weights |
| `--equal-weights` | Equal 0.25 per dimension |
| `--ahp-shrinkage` | λ blending factor (default `0.7`) |

**GNN inference:**

| Flag | Description |
|------|-------------|
| `--gnn-model` | Path to trained checkpoint directory |

**Anti-pattern detection:**

| Flag | Default | Description |
|------|---------|-------------|
| `--no-antipatterns` | `False` | Skip detection; exit code always 0 |
| `--severity` | `None` | Comma-separated filter: `critical`, `high`, `medium` |
| `--pattern` | `None` | Comma-separated pattern IDs (e.g. `SPOF,FAILURE_HUB,GOD_COMPONENT`) |
| `--catalog` | `False` | Print full catalog and exit |

**Output:**

| Flag | Description |
|------|-------------|
| `--output-antipatterns` | Write anti-pattern report JSON (feeds `visualize_graph.py`) |
| `--no-exit-code` | Always exit 0 (disable CI/CD gating) |

### Exit Codes (CI/CD Gate)

- `0` — clean (no anti-patterns, or `--no-antipatterns`)
- `1` — MEDIUM anti-patterns detected
- `2` — HIGH or CRITICAL anti-patterns detected → blocks deployment

---

## Step 3b: Train GNN

**Script:** `cli/train_graph.py`  
**Purpose:** Train a Heterogeneous Graph Attention Network (HeteroGAT) to predict component criticality.

```bash
PYTHONPATH=. python cli/train_graph.py --layer system --epochs 500 --hidden 128 --heads 8 --checkpoint output/gnn_checkpoints/
```

### Arguments

| Flag | Default | Type / Choices | Description |
|------|---------|----------------|-------------|
| `--layer` | `app` | `app`, `infra`, `mw`, `system` | System layer |
| `--structural` | `None` | Path | Skip Step 2, load pre-computed metrics JSON |
| `--simulated` | `None` | Path | Skip Step 4, load simulation results JSON |
| `--rmav` | `None` | Path | Skip Step 3, load RMAV scores JSON |
| `--hidden` | `64` | int | Hidden dimension |
| `--heads` | `4` | int | Attention heads |
| `--layers` | `3` | int | GNN layers |
| `--dropout` | `0.2` | float | Dropout rate |
| `--epochs` | `300` | int | Max epochs |
| `--lr` | `3e-4` | float | Learning rate |
| `--patience` | `30` | int | Early stopping patience |
| `--train-ratio` | `0.6` | float | Train split |
| `--val-ratio` | `0.2` | float | Validation split |
| `--no-edge-model` | `False` | flag | Skip edge model |
| `--seeds` | `None` | ints | Seed list for stability validation |
| `--multi-scenario` | `False` | flag | Inductive training on all domain scenarios |
| `--mode` | `gnn` | `rmav`, `gnn` | Evaluation path for final summary |
| `--variant` | `hetero_qos` | `hetero_qos`, `homo_unweighted`, `homo_scalar`, `topology_rmav` | Model architecture variant |
| `--checkpoint` | `output/gnn_checkpoints` | Path | Checkpoint directory |
| `--output` | `None` | Path | Save result JSON |

---

## Step 4: Simulate

**Script:** `cli/simulate_graph.py`  
**Purpose:** Fault injection and discrete-event message-flow simulation.

```bash
PYTHONPATH=. python cli/simulate_graph.py fault-inject --input data/atm_system.json --output output/simulation/ --seeds 42,123,456,789,2024 --export-json
```

### Subcommands

#### `fault-inject`

Systematic BFS cascade fault injection → `impact_scores.json`.

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `None` | Graph JSON file |
| `--layer` | `None` | Resolves to `data/<layer>.json` if `--input` missing |
| `--output` | `output/simulation/` | Output directory |
| `--nodes` | `None` | Comma-separated node IDs |
| `--node-types` | `Application,Broker,Library` | Comma-separated node types. Do **not** add `Topic` or `Node` — the cascade cannot express their failure and every instance scores `I(v)=0`. |
| `--seeds` | `42,123,456,789,2024` | Comma-separated seeds. ≥ 2 required for the artifact's `label_stability` block to be measurable. |
| `--cascade-depth` | `0` | Max depth (0 = unlimited) |
| `--propagation-threshold` | `0.2` | Fraction of feed loss before cascade |
| `--export-json` | `False` | Write JSON result files |

The emitted `impact_scores.json` is schema 2.1: it names its `labeler`, declares
`labeled_node_types` / `labeled_dimensions` / `unlabeled_node_ids`, and carries a
`label_stability` block giving the ceiling on any correlation computed against it. See
[failure-simulation.md §6.1](failure-simulation.md#61-impact_scoresjson).

#### `message-flow`

Discrete-event SimPy simulation.

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `None` | Graph JSON file |
| `--layer` | `None` | Resolves to `data/<layer>.json` |
| `--output` | `output/simulation/` | Output directory |
| `--duration` | `100.0` | Simulation duration (seconds) |
| `--fault-node` | `None` | Node ID to fault |
| `--fault-time` | `None` | Simulated fault time (default: `duration / 2`) |
| `--seed` | `42` | Random seed |
| `--default-rate` | `10.0` | Fallback publish rate (Hz) |
| `--default-queue-size` | `100` | Fallback broker queue capacity |
| `--export-json` | `False` | Write JSON result files |

#### `combined`

Run `fault-inject` then `message-flow` in sequence. Accepts all flags from both subcommands.

### Output Files

- `fault-inject` → `output/simulation/impact_scores.json`, `output/simulation/impact_scores_summary.txt`
- `message-flow` → `output/simulation/message_flow_results.json`, `output/simulation/message_flow_summary.txt`
- `combined` → all of the above

---

## Step 5: Validate

**Script:** `cli/validate_graph.py`  
**Purpose:** Statistically prove that topology-based Q(v) predictions agree with simulation-derived I(v).

### Subcommands

#### `single`

One-seed validation run.

```bash
PYTHONPATH=. python cli/validate_graph.py single --input data/system.json --qos
```

#### `sweep`

Multi-seed stability sweep.

```bash
PYTHONPATH=. python cli/validate_graph.py sweep --input data/system.json --qos
```

#### `report`

Full sweep + topology-class gates + node-type strata + JSON report.

```bash
PYTHONPATH=. python cli/validate_graph.py report --input data/system.json --output output/validation_report.json --qos
```

#### `compare`

Ablation study: topology-only vs QoS-enriched side-by-side. Produces a `--latex` table.

```bash
PYTHONPATH=. python cli/validate_graph.py compare --input data/system.json --latex --output output/ablation.json
```

#### `harness`

Methodological-guard validation on pre-computed JSON artifacts.

```bash
PYTHONPATH=. python cli/validate_graph.py harness \
    --predictions output/predictions.json \
    --ground-truth cascade=output/impact_scores.json \
    --ground-truth latency=output/latency_delta.json \
    --out output/harness_report.json
```

Append `:qos` to a ground-truth source to mark it QoS-coupled (triggers independence caveat).

### Common Arguments (all subcommands)

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Path to `system.json` |
| `--qos` | `False` | Enable QoS-weighted scoring |
| `--gnn-model` | `None` | Path to GNN checkpoint |
| `--top-k` | `20%` of nodes | K for classification metrics |
| `--seeds` | `42,123,456,789,2024` | Comma-separated seed list |
| `--cascade` | `5` | Cascade depth limit |
| `--bootstrap` | `2000` | Bootstrap resamples for CI |
| `--alpha` | `0.05` | Significance level |
| `--output` | `None` | Write JSON report |
| `--csv` | `False` | Write per-node CSV |
| `--latex` | `False` | Write LaTeX ablation table |
| `--verbose` | `False` | Print per-node scores |

### Output

- JSON report at `--output` path
- Optional CSV at `<output>_nodes.csv`
- Optional LaTeX at `<output>_table.tex`

---

## Step 6: Visualize

**Script:** `cli/visualize_graph.py`  
**Purpose:** Generate multi-layer HTML dashboards.

```bash
PYTHONPATH=. python cli/visualize_graph.py --layer system -o output/dashboard.html
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--demo` | `False` | Generate demo dashboard (no Neo4j) |
| `--open` / `-b` | `False` | Open in browser after generation |
| `--layer` | `system` | Comma-separated layers |
| `--layers` | `None` | Explicit multi-layer flag (takes precedence over `--layer`) |
| `--no-network` | `False` | Exclude interactive network graphs |
| `--no-matrix` | `False` | Exclude dependency matrices |
| `--no-validation` | `False` | Exclude validation metrics |
| `--antipatterns` | `None` | Path to pre-calculated anti-pattern JSON report |
| `--multi-seed` | `[]` | Paths to per-seed validation JSON files |
| `--cascade-file` | `None` | QoS ablation experiment JSON |

---

## Utilities

### Benchmark

**Script:** `cli/benchmark.py`  
Runs the pipeline across scales and layers, collecting timing and accuracy metrics.

```bash
PYTHONPATH=. python cli/benchmark.py --scales tiny,small,medium --layers app,system --runs 3 --output results/benchmark
```

| Flag | Default | Description |
|------|---------|-------------|
| `--scales` | `None` | Comma-separated scales |
| `--config` | `None` | YAML configuration file |
| `--full-suite` | `False` | Run `tiny + small + medium` |
| `--layers` | `app,infra,mw,system` | Layers to benchmark |
| `--runs` | `1` | Runs per scenario |
| `--output` / `-o` | `results/benchmark` | Output directory |
| `--ndcg-k` | `10` | K for NDCG@K |
| `--spearman-target` | `0.70` | Pass threshold |
| `--f1-target` | `0.80` | Pass threshold |
| `--dry-run` | `False` | Print plan without executing |

### Detect Anti-Patterns

**Script:** `cli/detect_antipatterns.py`  
Standalone anti-pattern detector.

```bash
PYTHONPATH=. python cli/detect_antipatterns.py --layer system --output results/ap.json
```

| Flag | Description |
|------|-------------|
| `--catalog` | Print full pattern catalog and exit |
| `--pattern` | Filter by pattern ID (comma-separated) |
| `--severity` | Filter by severity (comma-separated) |
| `--use-ahp` | Use AHP weights |
| `--ahp-shrinkage` | λ factor (default `0.7`) |

Exit codes mirror `predict_graph.py`: `0` clean, `1` MEDIUM, `2` HIGH/CRITICAL.

### Statistics

**Script:** `cli/statistics_graph.py`  
Cross-cutting topology statistics (live Neo4j or standalone file mode).

```bash
PYTHONPATH=. python cli/statistics_graph.py --input output/dataset.json --chart topic_bandwidth app_balance --format minimal
```

Available charts: `topic_bandwidth`, `app_balance`, `topic_fanout`, `cross_node_heatmap`, `node_comm_load`, `domain_comm`, `criticality_io`, `lib_dependency`, `node_critical_density`, `domain_diversity`.

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--input` / `-i` | `None` | Path | Standalone file mode (skips Neo4j) |
| `--chart` | all | registry IDs | Specific charts to compute |
| `--format` | `table` | `table`, `minimal`, `json` | Output format |
| `--output` / `-o` | `None` | Path | Save JSON results |

### Master Orchestrator

**Script:** `cli/run.py`  
Execute the full pipeline via the `saag` SDK.

```bash
PYTHONPATH=. python cli/run.py --all --layer system --gnn-model output/gnn_checkpoints/best_model --output output/result.json
```

| Flag | Description |
|------|-------------|
| `--all` | Run all stages sequentially |
| `--generate` | Run graph generation |
| `--analyze` | Run analysis |
| `--predict` | Run prediction (requires GNN) |
| `--simulate` | Run simulation |
| `--validate` | Run validation |
| `--visualize` | Run visualization |
| `--input` / `-i` | Input JSON |
| `--config` | Generation YAML config |
| `--scale` | Generation scale preset |
| `--output-dir` | Directory for outputs |
| `--clear` | Clear Neo4j before import |
| `--use-ahp` | AHP weights |
| `--gnn-model` | GNN checkpoint path |
| `--sim-mode` | `exhaustive` (default), `monte_carlo` |
| `--no-network`, `--no-matrix`, `--no-validation` | Visualization exclusions |

---

## First-Run Sequencing

Correct dependency order for a fresh project:

```
Generate → Import → Analyze → Simulate → Train → Predict → Validate → Visualize
```

Corresponding CLI commands:

```bash
# 0. Generate
PYTHONPATH=. python cli/generate_graph.py --scale medium --domain atm --output data/atm_system.json

# 1. Import
PYTHONPATH=. python cli/import_graph.py --input data/atm_system.json --clear

# 2. Analyze
PYTHONPATH=. python cli/analyze_graph.py --layer system

# 3. Simulate (fault-inject for ground-truth, then optional message-flow)
PYTHONPATH=. python cli/simulate_graph.py fault-inject --input data/atm_system.json --output output/simulation/ --export-json

# 3b. Train GNN (requires simulation results)
PYTHONPATH=. python cli/train_graph.py --layer system --output output/gnn_checkpoints/best_model

# 4. Predict
PYTHONPATH=. python cli/predict_graph.py --layer system --gnn-model output/gnn_checkpoints/best_model

# 5. Validate
PYTHONPATH=. python cli/validate_graph.py report --input data/atm_system.json --output output/validation_report.json --qos

# 6. Visualize
PYTHONPATH=. python cli/visualize_graph.py --layer system -o output/dashboard.html --antipatterns results/ap.json
```

For repeated runs after GNN training, use the master orchestrator:

```bash
PYTHONPATH=. python cli/run.py --all --layer system --gnn-model output/gnn_checkpoints/best_model
```

---

## Layer Projections

The `--layer` flag accepts:

| Value | Description |
|-------|-------------|
| `app` | Application layer |
| `infra` | Infrastructure layer |
| `mw` | Middleware layer |
| `system` | Full system (default) |
| `all` | Run `app`, `infra`, `mw`, `system` sequentially |

Multiple layers can be specified as a comma-separated list: `--layer app,infra,system`.

Layer values are case-insensitive in most scripts.

---

## Advanced Workflows

### Batch Scenario Generation

```bash
PYTHONPATH=. python cli/generate_graph.py batch \
    --input-dir data/ \
    --output-dir output/ \
    --seeds 42,123,456,789,2024 \
    --manifest batch_manifest.json \
    --report batch_report.json
```

### Multi-Seed Stability Sweeps

Run validation across multiple random seeds:

```bash
PYTHONPATH=. python cli/validate_graph.py sweep \
    --input data/system.json \
    --seeds 42,123,456,789,2024 \
    --qos \
    --output output/sweep_report.json
```

### LOSO Cross-Validation

Use `cli/loso_evaluate.py` for leave-one-scenario-out evaluation across domain datasets.

### Ablation Studies (Topology-only vs QoS-enriched)

```bash
PYTHONPATH=. python cli/validate_graph.py compare \
    --input data/system.json \
    --output output/ablation.json \
    --latex
```

The `compare` subcommand runs both QoS-off and QoS-on sweeps, computes Δρ, and produces a LaTeX table suitable for publication.

### CI/CD Anti-Pattern Gating

```bash
PYTHONPATH=. python cli/predict_graph.py --layer system \
    --severity critical,high \
    --output-antipatterns results/ap.json
```

- Exit `0`: no anti-patterns
- Exit `1`: MEDIUM patterns
- Exit `2`: HIGH / CRITICAL → blocks pipeline

### Benchmarking Across Scales

```bash
PYTHONPATH=. python cli/benchmark.py --scales tiny,small,medium,large,jumbo,xlarge \
    --layers app,infra,mw,system \
    --runs 3 \
    --spearman-target 0.80 \
    --f1-target 0.75
```

---

## Output Reference

| Stage | Default Output Path | Notes |
|-------|---------------------|-------|
| Generate | `output/graph.json` | Configurable via `--output` |
| Import | (Neo4j DB) | Stats via `--output` |
| Export | user-specified | `persistence` or `analysis` format |
| Analyze | propagated via console | Save with `--output` |
| Train | `output/gnn_checkpoints/` | Checkpoint directory |
| Predict | user-specified via `--output` | Anti-patterns via `--output-antipatterns` |
| Simulate | `output/simulation/` | JSON + text summaries |
| Validate | user-specified via `--output` | Optional CSV / LaTeX |
| Visualize | `dashboard.html` | Configurable via `-o` |
| Batch | `output/<domain>_results/` | Per-scenario output directories |

---

## Troubleshooting

### `PYTHONPATH=.` Requirement

Direct invocation requires the project root on `PYTHONPATH`. Omitting it causes `ModuleNotFoundError` for `saag` and `cli` modules.

### Missing GNN Checkpoint for `--all`

`cli/run.py --all` (or `--predict`) requires a trained GNN checkpoint. If `--gnn-model` is not provided and `output/gnn_checkpoints/best_model` does not exist, the predict stage is skipped with a warning. Train first:

```bash
PYTHONPATH=. python cli/train_graph.py --layer system --output output/gnn_checkpoints/best_model
```

### Neo4j Connection Issues

- Verify `bolt://localhost:7687` is reachable.
- Default credentials `neo4j` / `password` apply unless overridden with `--user` / `--password` or `NEO4J_USER` / `NEO4J_PASSWORD`.
- Use `--dry-run` with `import_graph.py` to validate JSON without touching the database.

### Case-Sensitive QoS Values

QoS scenario names (`--domain`, `--scenario`) must match exactly: `av`, `iot`, `finance`, `healthcare`, `hub-and-spoke`, `microservices`, `enterprise`, `atm`.

### First-Run Guard Warning

The orchestrator warns when `--all` is used without a GNN checkpoint. Follow the printed three-step sequence to generate data, train, and then run the full pipeline.

### Small Graphs and Unreliable Spearman ρ

`validate_graph.py` warns when fewer than 10 Application nodes are present; Spearman ρ has high variance on small `n`.

---

## Cross-References

- Graph generation details: [graph-generation.md](graph-generation.md)
- Graph model and topology: [graph-model.md](graph-model.md)
- Structural analysis: [structural-analysis.md](structural-analysis.md)
- Prediction and GNN: [prediction.md](prediction.md)
- Failure simulation: [failure-simulation.md](failure-simulation.md)
- Validation methodology: [validation.md](validation.md)
- Visualization: [visualization.md](visualization.md)
- Scenario management: [scenario.md](scenario.md)
- Statistics: [statistics.md](statistics.md)
- Anti-patterns: [antipatterns.md](antipatterns.md)
