# Graph Generation

**Offline synthetic graph generation, configuration presets, and command-line reference.**

[README](../README.md) | → [Step 1: Model (Import)](graph-model.md)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture of the Generation Stack](#2-architecture-of-the-generation-stack)
3. [Graph Model — Node and Edge Types](#3-graph-model--node-and-edge-types)
4. [Generation Modes](#4-generation-modes)
5. [The `generate_graph.py` CLI](#5-the-generate_graphpy-cli)
6. [The `generate_graph.py batch` Subcommand](#6-the-generate_graphpy-batch-subcommand)
7. [Scenario Configuration Files](#7-scenario-configuration-files)
8. [The `StatisticalGraphGenerator`](#8-the-statisticalgraphgenerator)
9. [Output Format](#9-output-format)
10. [Programmatic API](#10-programmatic-api)
11. [Known Limitations and Open Issues](#11-known-limitations-and-open-issues)

---

## 1. Overview

Synthetic graph generation is an offline input preparation stage that supports the core 6-step Software-as-a-Graph (SaG) analytical pipeline:

```
Offline Input Prep (Generate) → Model / Import (Step 1) → Analyze (Step 2) → Predict (Step 3) → Simulate (Step 4) → Validate (Step 5) → Visualize (Step 6)
```

Its role is to produce a **synthetic publish-subscribe system topology** in JSON format that can be loaded into Neo4j and subsequently subjected to structural analysis and failure simulation. The generator is self-contained: it requires no running database, no external service, and no runtime monitoring data. A single deterministic seed produces an identical dataset on every invocation.

The generator is used for two distinct purposes in the project:
- **Validation** — producing the synthetic datasets over which Spearman $\rho$ and F1 scores are measured. Reproducibility via seed is essential here.
- **Benchmarking** — producing datasets of controlled scale (tiny through xlarge) to measure pipeline throughput and algorithmic complexity.

---

## 2. Architecture of the Generation Stack

The generation functionality is organized in four layers, each with a single responsibility:

```
cli/generate_graph.py              ← CLI entry point (single-graph, batch, and validate subcommands)
cli/common/dispatcher.py           ← dispatch_generate() — bridges CLI args to service
cli/common/batch_generation.py     ← run_batch_generation() — batch dataset generation logic
cli/common/dataset_validation.py   ← run_dataset_validation() — topology-class validation
tools/generation/service.py        ← GenerationService / generate_graph() convenience fn
tools/generation/generator.py      ← StatisticalGraphGenerator (core logic)
tools/generation/models.py         ← GraphConfig, SCALE_PRESETS, statistical structs
tools/generation/datasets.py       ← Domain name pools, QoS scenario mappings
```

The same service layer is exposed via the FastAPI router at `POST /api/v1/graph/generate`, so the web UI and the CLI share identical generation logic with no duplication.

---

## 3. Graph Model — Node and Edge Types

A generated graph contains five node types and six structural edge types.

### 3.1 Node Types

| Type | ID Prefix | Description |
|:---|:---|:---|
| `Application` (CSU) | `A{n}` | Software component that publishes and/or subscribes to topics |
| `Library` | `L{n}` | Shared software library used by one or more applications |
| `Broker` | `B{n}` | Message broker / DDS participant that routes topics |
| `Node` (Infrastructure) | `N{n}` | Physical or virtual host running applications and brokers |
| `Topic` | `T{n}` | Named communication channel carrying typed messages; includes size, QoS policies, frequency (Hz), and ground-truth criticality |

### 3.2 Structural Edge Types

| Edge | From → To | Meaning |
|:---|:---|:---|
| `PUBLISHES_TO` | Application $\rightarrow$ Topic | App produces messages on this topic |
| `SUBSCRIBES_TO` | Application $\rightarrow$ Topic | App consumes messages from this topic |
| `ROUTES` | Broker $\rightarrow$ Topic | Broker is responsible for routing this topic |
| `RUNS_ON` | Application/Broker $\rightarrow$ Node | Component is deployed on this infrastructure node |
| `USES` | Application / Library $\rightarrow$ Library | Component or library depends on this shared library (transitive dependencies are also generated with 30% probability) |
| `CONNECTS_TO` | Node $\rightarrow$ Node | Network link between infrastructure nodes (30% probability by default) |

These six edge types constitute the **structural graph $G_{\text{structural}}$**, which is used by the simulation stage (Step 4) to trace failure propagation. A separate **analysis graph $G_{\text{analysis}}$** is derived from $G_{\text{structural}}$ by computing `DEPENDS_ON` edges, which are used exclusively by Steps 2 and 3 (analysis and prediction). The separation ensures that prediction and simulation remain independent.

---

## 4. Generation Modes

### 4.1 Scale-Preset Mode

The fastest way to generate a graph. Six named presets are built in:

| Preset | Applications | Topics | Brokers | Nodes | Libraries | Total Nodes | Typical Use |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `tiny` | 5 | 5 | 1 | 2 | 2 | 15 | Unit tests |
| `small` | 15 | 10 | 2 | 4 | 5 | 36 | Quick checks |
| `medium` | 50 | 30 | 3 | 8 | 10 | 101 | Development |
| `large` | 150 | 100 | 6 | 20 | 30 | 306 | Integration tests |
| `jumbo` | 300 | 120 | 10 | 40 | 50 | 520 | Large-scale benchmarks |
| `xlarge` | 500 | 300 | 10 | 50 | 100 | 960 | Performance benchmarks |

In scale-preset mode, QoS values for each topic are sampled uniformly from the full option space. Node placement, publish/subscribe wiring, and library usage are assigned using simple random selection without statistical distributions. This mode is suitable for benchmarking and quick smoke tests. Total edge counts are seed-dependent; for scale-preset graphs expect roughly 4–10$\times$ the total node count in edges.

```bash
PYTHONPATH=. python cli/generate_graph.py --scale medium --seed 42 --output output/graph.json
```

### 4.2 Statistical-Config Mode (YAML)

The primary mode for validation work. A YAML configuration file fully specifies:
- exact component counts
- statistical distributions for node loading, publish counts, subscribe counts, and topic fan-in/fan-out
- categorical distributions for QoS durability, reliability, and transport priority
- role and criticality distributions for applications
- library usage distributions

When a YAML config is loaded, the generator uses `StatisticalMetric` sampling (clamped Gaussian via `random.gauss`) for continuous quantities and weighted-list sampling for categorical quantities. This produces topologies whose structural properties closely match the declared distributions, enabling repeatable validation experiments.

```bash
PYTHONPATH=. python cli/generate_graph.py --config data/scenarios/scenario_01_autonomous_vehicle.yaml \
       --output output/av_system.json
```

`--scale` and `--config` are mutually exclusive.

### 4.3 Domain and Scenario Enrichment

Two optional flags add realistic naming and domain-matched QoS mappings to either generation mode.

`--domain` selects a domain name pool (`av`, `iot`, `finance`, `healthcare`, `enterprise`, `robotics`, `e-commerce`). When active, application names are drawn from domain-specific name lists (e.g. `PlanningEngine`, `RiskManager`, `VitalSignsMonitor`) rather than generic `App-{n}` identifiers. Library names, broker names, and system hierarchy labels are similarly drawn from domain pools.

`--scenario` selects a QoS mapping table that overrides the statistical distribution for each topic's QoS settings. For instance, `--scenario av` ensures that topics whose names contain navigation or safety keywords receive `RELIABLE / TRANSIENT_LOCAL / HIGH` QoS regardless of the general distribution. The available scenario values are `av`, `iot`, `finance`, `healthcare`, `hub-and-spoke`, `microservices`, and `enterprise`.

Domain and scenario can be combined with either `--scale` or `--config`. When a YAML config is used and specifies `domain:` and `scenario:` fields at the top level, those values are used automatically without needing to pass them on the CLI.

---

## 5. The `generate_graph.py` CLI

Located at `cli/generate_graph.py`. This is the primary generation entry point. In its default mode (no subcommand) it generates a single graph file. It also exposes two subcommands: `batch` (batch dataset generation — see Section 6) and `validate` (topology-class validation — see Section 5.3).

### 5.1 Arguments Reference

| Argument | Type | Default | Description |
|:---|:---|:---|:---|
| `--scale` | choice | — | Named scale preset: `tiny`, `small`, `medium`, `large`, `jumbo`, `xlarge`. Mutually exclusive with `--config`. |
| `--config` | path | — | Path to a YAML scenario configuration file. Mutually exclusive with `--scale`. |
| `--output` | path | `output/graph.json` | Destination path for the generated JSON file. Parent directories are created automatically. |
| `--seed` | int | `42` | Random seed. Any integer value gives a deterministic, reproducible output. |
| `--domain` | str | — | Domain name pool: `av`, `iot`, `finance`, `healthcare`, `hub-and-spoke`, `microservices`, `enterprise`, `atm`. Silently falls back to generic naming on unsupported strings. |
| `--scenario` | choice | — | QoS scenario mapping: `av`, `iot`, `finance`, `healthcare`, `hub-and-spoke`, `microservices`, `enterprise`, `atm`. |
| `--connection-density` | float | `0.3` | Connection density (probability of `connects_to` edges between physical infrastructure nodes). |
| `--verbose` / `-v` | flag | off | Print full tracebacks on error. |

When `--config` is provided, the seed embedded in the YAML (`graph.seed`) is used and the `--seed` CLI argument is ignored. When `--scale` is provided without `--seed`, the default seed of 42 is used.

### 5.2 Usage Examples

```bash
# Minimal: medium graph, default seed
PYTHONPATH=. python cli/generate_graph.py --scale medium --output output/graph.json

# Reproducible small graph for a unit test
PYTHONPATH=. python cli/generate_graph.py --scale small --seed 123 --output output/small_123.json

# Autonomous vehicle scenario from YAML config
PYTHONPATH=. python cli/generate_graph.py \
    --config data/scenarios/scenario_01_autonomous_vehicle.yaml \
    --output output/av_system.json

# IoT scenario with domain naming
PYTHONPATH=. python cli/generate_graph.py \
    --config data/scenarios/scenario_02_iot_smart_city.yaml \
    --domain iot \
    --output output/iot_system.json

# Quick large graph for benchmarking
PYTHONPATH=. python cli/generate_graph.py --scale large --seed 2024 --output output/large_bench.json

# Followed immediately by the full pipeline
PYTHONPATH=. python cli/run.py --all --input output/av_system.json

# Generation as stage 1 of run.py orchestrator
PYTHONPATH=. python cli/run.py --all --config data/scenarios/scenario_01_autonomous_vehicle.yaml \
    --input output/av_system.json --output-dir output/av_results
```

### 5.3 Subcommands: batch and validate

`generate_graph.py` exposes two named subcommands beyond single-graph generation:
- **`batch`** — Generates all scenario datasets in a single invocation. Accepts all the arguments documented in Section 6.
  ```bash
  PYTHONPATH=. python cli/generate_graph.py batch [OPTIONS]
  ```
- **`validate`** — Runs topology-class validation across the generated datasets, checking that each scenario falls into the correct expected class (fan-out dominated, dense pubsub, anti-pattern / SPOF, or sparse). The implementation lives in `cli/common/dataset_validation.py`.
  ```bash
  PYTHONPATH=. python cli/generate_graph.py validate [OPTIONS]
  ```

---

## 6. The `generate_graph.py batch` Subcommand

Invoked as `PYTHONPATH=. python cli/generate_graph.py batch`. Generates all scenario datasets in a single invocation, with optional multi-seed variants and legacy dataset refresh. The implementation lives in `cli/common/batch_generation.py`.

### 6.1 Arguments Reference

| Argument | Default | Description |
|:---|:---|:---|
| `--input-dir` | `data/scenarios/` | Directory containing `scenario_*.yaml` files. |
| `--output-dir` | `output/` | Output directory for generated JSON files. |
| `--refresh-legacy` | off | Regenerate `data/system.json` (medium, seed=42) and `data/dataset.json` (small, seed=42) using the current generator, adding `code_metrics` and `system_hierarchy`. |
| `--multi-seed` | off | Generate per-seed variants for scenarios 01–06 using all seeds in `--seeds`. Scenarios 07–09 are excluded. |
| `--seeds` | `42,123,456,789,2024` | Comma-separated seed list for `--multi-seed` mode. |
| `--scenario` | — | Substring filter — only generate matching scenario names (e.g. `scenario_03`). |
| `--force` | off | Overwrite existing output files. |
| `--manifest` | `output/dataset_manifest.json` | Path for the JSON manifest recording metadata for every generated file. |
| `--dry-run` | off | Print the execution plan without writing any files. |
| `--verbose` / `-v` | off | Print topology breakdown (node counts, edge counts, QoS distribution) per scenario. |

### 6.2 Usage Examples

```bash
# Generate all scenario datasets (skips existing by default)
PYTHONPATH=. python cli/generate_graph.py batch

# Full preparation run before a validation session
PYTHONPATH=. python cli/generate_graph.py batch --multi-seed --refresh-legacy

# Preview the plan without writing anything
PYTHONPATH=. python cli/generate_graph.py batch --multi-seed --refresh-legacy --dry-run

# Regenerate only the financial and healthcare scenarios
PYTHONPATH=. python cli/generate_graph.py batch --scenario scenario_03 --force
PYTHONPATH=. python cli/generate_graph.py batch --scenario scenario_04 --force

# Generate all scenarios with verbose topology output
PYTHONPATH=. python cli/generate_graph.py batch --verbose

# Custom seeds for multi-seed stability sweep
PYTHONPATH=. python cli/generate_graph.py batch --multi-seed --seeds 42,100,200,300,400
```

### 6.3 Output Layout and Manifest

```
output/
├── scenario_08_tiny_regression.json         ← smoke test generated first
├── scenario_01_autonomous_vehicle.json
├── scenario_02_iot_smart_city.json
├── scenario_03_financial_trading.json
├── scenario_04_healthcare.json
├── scenario_05_hub_and_spoke.json
├── scenario_06_microservices.json
├── scenario_07_enterprise_xlarge.json
├── scenario_09_xlarge_stress.json
│
├── multi_seed/                               ← only with --multi-seed
│   ├── scenario_01_autonomous_vehicle/
│   │   ├── scenario_01_autonomous_vehicle_seed42.json
│   │   ├── scenario_01_autonomous_vehicle_seed123.json
│   │   ├── scenario_01_autonomous_vehicle_seed456.json
│   │   ├── scenario_01_autonomous_vehicle_seed789.json
│   │   └── scenario_01_autonomous_vehicle_seed2024.json
│   └── ... (scenarios 02–06)
│
└── dataset_manifest.json                     ← metadata for every generated file
```

The manifest JSON records metadata details (`scenario_name`, `source_config`, `output_path`, `seed`, `counts`, `edge_counts`, `qos_distribution`, `criticality_distribution`, etc.) and validation status checks (`"ok"`, `"skipped"`, or `"error"`).

---

## 7. Scenario Configuration Files

Each `data/scenarios/scenario_*.yaml` configuration file is a self-contained specification for one validation scenario. The file is passed directly to the generator via `cli/generate_graph.py --config`.

> [!NOTE]
> For a full list of available validation scenarios, details on their stress parameters, seed presets, and topology configurations, refer to the central scenario guide: [scenario.md](scenario.md).

### 7.1 YAML Schema Reference

Every scenario YAML follows this structure:

```yaml
graph:
  seed: <integer>           # Unique per scenario; determines full reproducibility
  domain: <string>          # Optional: av | iot | finance | healthcare | enterprise
  scenario: <string>        # Optional: matches --scenario QoS mapping table
  connection_density: <float> # Optional: probability of physical node reachability (default: 0.3)

  counts:
    nodes: <int>            # Infrastructure nodes
    applications: <int>     # Application (CSU) components
    libraries: <int>        # Shared libraries
    topics: <int>           # Message topics
    brokers: <int>          # Message brokers

  # Statistical distribution for node loading
  node_stats:
    applications_per_node:
      mean: <float>
      median: <float>
      std: <float>
      min: <int>
      max: <int>
      q1: <float>
      q3: <float>
      iqr: <float>

  # Statistical distributions for application connectivity
  application_stats:
    direct_publish_count: { mean, median, std, min, max, q1, q3, iqr }
    direct_subscribe_count: { mean, median, std, min, max, q1, q3, iqr }
    total_publish_count_including_libraries: { ... }
    total_subscribe_count_including_libraries: { ... }
    app_role_distribution:
      total_count: <int>
      category_counts: { pub: <int>, sub: <int>, pubsub: <int> }
      mode: <string>
      mode_count: <int>
      mode_percentage: <float>
    app_criticality_distribution:
      total_count: <int>
      category_counts: { critical: <int>, non_critical: <int> }
      mode: <string>
      mode_count: <int>
      mode_percentage: <float>

  # Statistical distributions for library connectivity
  library_stats:
    applications_using_this_library: { mean, median, std, min, max, q1, q3, iqr }
    direct_publish_count: { ... }
    direct_subscribe_count: { ... }
    total_publish_count_including_libraries: { ... }
    total_subscribe_count_including_libraries: { ... }

  # Statistical distribution for topic sizing and fanin/fanout
  topic_stats:
    topic_size_bytes: { mean, median, std, min, max, q1, q3, iqr }
    applications_publishing_to_this_topic: { mean, median, std, min, max, q1, q3, iqr }
    applications_subscribing_to_this_topic: { mean, median, std, min, max, q1, q3, iqr }

  # Categorical QoS distributions (counts must equal topic count)
  qos_stats:
    qos_durability_distribution:
      total_count: <int>
      category_counts:
        volatile: <int>
        transient_local: <int>
        transient: <int>
        persistent: <int>
      mode: <string>
      mode_count: <int>
      mode_percentage: <float>
    qos_reliability_distribution:
      total_count: <int>
      category_counts:
        best_effort: <int>
        reliable: <int>
      mode: <string>
      mode_count: <int>
      mode_percentage: <float>
    qos_transport_priority_distribution:
      total_count: <int>
      category_counts:
        low: <int>
        medium: <int>
        high: <int>
        critical: <int>
      mode: <string>
      mode_count: <int>
      mode_percentage: <float>
```

### 7.2 Writing a New Scenario

- **Copy Template** — Copy the closest existing scenario file and rename it `scenario_NN_<name>.yaml`.
- **Set Unique Seed** — Set `graph.seed` to a unique value (e.g. `1010` for scenario 10).
- **Scale Component Counts** — Adjust `graph.counts` to the desired scale using the presets table in Section 4.1.
- **Tune Statistical Distributions** — Configure distributions to reflect specific topological properties (e.g. increase subscriber fan-out to test Reliability, or reduce broker count to force SPOFs).
- **Configure QoS Distributions** — Set QoS durability and reliability variables to match the target domain profile.
- **Document expected outcomes** — Write a clear comment block at the top of the file describing the scenario's validation objectives.

---

## 8. The `StatisticalGraphGenerator`

Located at `tools/generation/generator.py`. This class performs the actual graph construction. It is instantiated by `GenerationService` and should not normally be called directly.

### 8.1 Generation Pipeline

The generator runs in two passes.

**Pass 1 — Nodes and Attribute Initialization.** All five node types are created first, before any edges, in the following order: infrastructure nodes, brokers, topics, applications, libraries.
- **Topic Frequency** — Sampled via `_sample_topic_frequency()` from a per-domain log-uniform distribution, capturing differences in message frequencies (e.g. HFT tick feeds vs. environmental sensors).
- **Topic Criticality** — Derived from QoS scores. The generator injects controlled label noise (~17% flip rate) to prevent GNN prediction leakage, forcing models to rely on structural context rather than simple QoS parameter lookups.
- **Hierarchy Cluster Pre-assignment** — Pre-assigns applications and topics to hierarchy clusters (`css_name` CSC boundaries). This ensures subsequent wiring preferentially routes communication within clusters, making system hierarchies structurally meaningful.

**Pass 2 — Relationships.** Edges are constructed in strict dependency order to resolve inherited library publish/subscribe counts:
1. `RUNS_ON` — Places applications and brokers on hardware hosts nodes. Uses 70% cluster-affine node placement to co-locate related applications on shared infrastructure.
2. `ROUTES` — Assigns routing brokers to topics. Adjusts broker placement to co-locate brokers with the applications they serve.
3. `USES` — Resolves library dependencies (app-to-lib and lib-to-lib), establishing transitive chains.
4. `PUBLISHES_TO` / `SUBSCRIBES_TO` — Wires pub/sub relationships. Implements QoS topic partition affinity, steering gateway/controller components to high-priority reliable topics and sensors to best-effort topics.
5. **Post-Topology Quality Passes** — Enforces role constraints (e.g. ensuring `pub`-only apps do not subscribe) and assigns application criticality based on structural degree hubs.

### 8.2 Code-Metrics Generation

Every `Application` and `Library` node carries a `code_metrics` block feeding the RMAV Maintainability $M(v)$ penalty. Metrics are generated from type-specific archetypes in `_CODE_METRICS_PARAMS` and `_LIB_CODE_METRICS_PARAMS`, outputting the following schema:
- **Size** — `total_loc`, `total_classes`, `total_methods`, `total_fields`.
- **Complexity** — `total_wmc`, `avg_wmc`, `max_wmc`.
- **Cohesion** — `avg_lcom`, `max_lcom`.
- **Coupling** — `avg_cbo`, `max_cbo`, `avg_rfc`, `max_rfc`, `avg_fanin`, `max_fanin`, `avg_fanout`, `max_fanout`.

### 8.3 System-Hierarchy Assignment

Every application and library is assigned a `system_hierarchy` block representing its position in the MIL-STD-498 decomposition hierarchy (CSCI, CSC, CSU cluster, and system label), drawing from `SYSTEM_HIERARCHY_POOLS` based on the configured domain.

### 8.4 QoS Assignment

Topics receive one categorical value for durability, reliability, and transport priority. The QoS weight for a topic is computed from `QoSPolicy` score mappings. Topics carrying `PERSISTENT + RELIABLE + CRITICAL` traffic receive the maximum QoS weight ($1.0$), which multiplies the directed articulation point score to produce the full QSPOF Availability $A(v)$ term.

---

## 9. Output Format

The generator produces a single JSON file structured as follows:

```json
{
  "metadata": {
    "scale": {
      "apps": 50,
      "topics": 30,
      "brokers": 3,
      "nodes": 8,
      "libs": 10
    },
    "seed": 42,
    "generation_mode": "statistical",
    "domain": "av",
    "scenario": "av"
  },
  "nodes": [
    { "id": "N0", "name": "ComputeNode-Alpha", "cpu_cores": 8, "memory_gb": 32 }
  ],
  "brokers": [
    { "id": "B0", "name": "DDS-Participant-0" }
  ],
  "topics": [
    {
      "id": "T0",
      "name": "/lidar/point_cloud",
      "size": 4096,
      "qos": {
        "durability": "TRANSIENT_LOCAL",
        "reliability": "RELIABLE",
        "transport_priority": "HIGH"
      },
      "frequency": 10.0,
      "criticality": "high"
    }
  ],
  "applications": [
    {
      "id": "A0",
      "name": "SensorFusion",
      "role": "pubsub",
      "app_type": "processor",
      "criticality": true,
      "version": "2.3.1",
      "system_hierarchy": {
        "csc_name": "Autonomous Vehicle Platform",
        "csci_name": "Perception Software",
        "css_name": "Sensor Fusion",
        "csms_name": "Point Cloud Processing"
      },
      "code_metrics": {
        "size":       { "total_loc": 1850, "total_classes": 24, "total_methods": 187, "total_fields": 48 },
        "complexity": { "total_wmc": 437, "avg_wmc": 18.2, "max_wmc": 49 },
        "cohesion":   { "avg_lcom": 42.5, "max_lcom": 170.0 },
        "coupling":   { "avg_cbo": 11.3, "max_cbo": 24,
                        "avg_rfc": 34.7, "max_rfc": 74,
                        "avg_fanin": 5.0, "max_fanin": 18,
                        "avg_fanout": 9.0, "max_fanout": 24 }
      }
    }
  ],
  "libraries": [
    {
      "id": "L0",
      "name": "NavCore",
      "version": "1.4.2",
      "system_hierarchy": {},
      "code_metrics": {}
    }
  ],
  "relationships": {
    "runs_on":      [{ "from": "A0", "to": "N2" }],
    "routes":       [{ "from": "B0", "to": "T0" }],
    "publishes_to": [{ "from": "A0", "to": "T0" }],
    "subscribes_to":[{ "from": "A3", "to": "T0" }],
    "connects_to":  [{ "from": "N0", "to": "N1" }],
    "uses":         [{ "from": "A0", "to": "L0" }]
  }
}
```

---

## 10. Programmatic API

The generation service can be called directly from Python:

```python
from tools.generation import GenerationService, load_config, generate_graph
from tools.generation.models import GraphConfig, SCALE_PRESETS

# Generate using a scale preset with custom connection density
data = generate_graph(scale="medium", seed=42, connection_density=0.15)

# Load configuration from a YAML file
config = load_config(Path("data/scenarios/scenario_01_autonomous_vehicle.yaml"))
service = GenerationService(config=config)
data = service.generate()

# Inspect available presets
for name, preset in SCALE_PRESETS.items():
    print(f"{name}: {preset}")
```

---

## 11. Known Limitations and Open Issues

*Document last updated: June 2026. Maintained alongside `tools/generation/` and `cli/generate_graph.py`.*

| # | Area | Description | Status |
|:---|:---|:---|:---|
| 1 | `--domain` validation | Invalid domain strings are silently accepted and fall back to generic naming with no warning. | Open |
| 2 | `generation_mode` field | The `metadata.generation_mode` field is `"statistical"` only when a YAML config with at least one `*_stats` section is provided. A YAML file that specifies only `counts` and no stats sections will produce `generation_mode: "random"`. | Open |
| 3 | Pub/sub duplicate edges | The pub/sub wiring strategies do not deduplicate edges: the same `(app, topic)` pair can appear more than once in `publishes_to` / `subscribes_to` if both topic-driven and app-driven wiring write the same edge. The import pipeline deduplicates on ingest. | Open |
| 4 | Broker guard semantics | The unrouted-broker guard (round-robin assignment) is deterministic but can assign a stranded broker to an already over-routed topic, skewing broker betweenness scores in topologies with many brokers and few topics. | Open |
| 5 | Hierarchy clustering | Hierarchy labels were formerly assigned independently from a flat pool with no structural effect. As of April 2026 the generator performs a seeded cluster pre-assignment and uses `_sample_biased()` (p_intra = 0.65) to make intra-cluster pub/sub edges more probable. | **Resolved** |
| 6 | Infrastructure placement | `RUNS_ON` edges were previously assigned uniformly at random (or sequentially by count distribution), with no cluster awareness. As of April 2026 `_assign_apps_to_nodes()` applies 70 % cluster-affine placement, co-locating functionally related apps on the same infrastructure node. | **Resolved** |
| 7 | Broker placement | Brokers were previously placed on nodes via round-robin index, ignoring which cluster of topics they route. As of April 2026 `_rewrite_broker_placement()` reads the completed routes list and places each broker on a node in its plurality cluster. | **Resolved** |
| 8 | Criticality coherence | Criticality was previously sampled before topology was built, so critical apps could be structural leaf nodes. As of April 2026 `_assign_criticality_two_pass()` assigns `critical=True` after topology is built, biased toward high-degree (hub) applications. | **Resolved** |
| 9 | Role constraint drift | The `topic_stats` pub/sub wiring path could assign subscribe edges to `pub`-only apps. As of April 2026 `_validate_role_constraints()` removes such violations before deduplication. | **Resolved** |
| 10 | QoS–topology coherence | Topic selection during pub/sub wiring was cluster-biased but QoS-agnostic. As of April 2026 `_partition_topics_by_qos_affinity()` steers gateway/controller apps toward `RELIABLE`/`HIGH` topics and sensor apps toward `BEST_EFFORT`/`LOW` topics. | **Resolved** |
| 11 | Topic attribute leakage | Deterministic QoS-to-criticality mappings allowed GNNs to bypass structural learning. The generator now samples topic frequency from log-uniform domain distributions and injects ~17% label noise into topic criticality, forcing GNNs to exploit multi-hop context. | **Resolved** |
