# Graph Generation — Process and CLI Reference

**Software-as-a-Graph (SaG) — Methodology Documentation**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture of the Generation Stack](#2-architecture-of-the-generation-stack)
3. [Graph Model — Node and Edge Types](#3-graph-model--node-and-edge-types)
4. [Generation Modes](#4-generation-modes)
   - 4.1 [Scale-preset mode](#41-scale-preset-mode)
   - 4.2 [Statistical-config mode (YAML)](#42-statistical-config-mode-yaml)
   - 4.3 [Domain and scenario enrichment](#43-domain-and-scenario-enrichment)
5. [The `generate_graph.py` CLI](#5-the-generate_graphpy-cli)
   - 5.1 [Arguments reference](#51-arguments-reference)
   - 5.2 [Usage examples](#52-usage-examples)
   - 5.3 [Subcommands: batch and validate](#53-subcommands-batch-and-validate)
6. [The `generate_graph.py batch` Subcommand](#6-the-generate_graphpy-batch-subcommand)
   - 6.1 [Arguments reference](#61-arguments-reference)
   - 6.2 [Usage examples](#62-usage-examples)
   - 6.3 [Output layout and manifest](#63-output-layout-and-manifest)
7. [Scenario Configuration Files](#7-scenario-configuration-files)
   - 7.1 [Scenario catalogue](#71-scenario-catalogue)
   - 7.2 [YAML schema reference](#72-yaml-schema-reference)
   - 7.3 [Writing a new scenario](#73-writing-a-new-scenario)
8. [The `StatisticalGraphGenerator`](#8-the-statisticalgraphgenerator)
   - 8.1 [Generation pipeline](#81-generation-pipeline)
   - 8.2 [Code-metrics generation](#82-code-metrics-generation)
   - 8.3 [System-hierarchy assignment](#83-system-hierarchy-assignment)
   - 8.4 [QoS assignment](#84-qos-assignment)
9. [Output Format](#9-output-format)
10. [Programmatic API](#10-programmatic-api)
11. [REST API Export Formats](#11-rest-api-export-formats)
12. [Known Limitations and Open Issues](#12-known-limitations-and-open-issues)

---

## 1. Overview

Graph generation is Step 1 of the six-step SaG methodology pipeline:

```
Generate → Import → Analyze → Simulate → Validate → Visualize
```

Its role is to produce a **synthetic publish-subscribe system topology** in JSON format that can be loaded into Neo4j and subsequently subjected to structural analysis and failure simulation. The generator is self-contained: it requires no running database, no external service, and no runtime monitoring data. A single deterministic seed produces an identical dataset on every invocation.

The generator is used for two distinct purposes in the project:

- **Validation** — producing the synthetic datasets over which Spearman ρ and F1 scores are measured. Reproducibility via seed is essential here.
- **Benchmarking** — producing datasets of controlled scale (tiny through xlarge) to measure pipeline throughput and algorithmic complexity.

---

## 2. Architecture of the Generation Stack

The generation functionality is organised in four layers, each with a single responsibility.

```
bin/generate_graph.py              ← CLI entry point (single-graph, batch, and validate subcommands)
bin/common/dispatcher.py           ← dispatch_generate() — bridges CLI args to service
bin/common/batch_generation.py     ← run_batch_generation() — batch dataset generation logic
bin/common/dataset_validation.py   ← run_dataset_validation() — topology-class validation
tools/generation/service.py        ← GenerationService / generate_graph() convenience fn
tools/generation/generator.py      ← StatisticalGraphGenerator (core logic)
tools/generation/models.py         ← GraphConfig, SCALE_PRESETS, statistical structs
tools/generation/datasets.py       ← Domain name pools, QoS scenario mappings
```

The same service layer is exposed via the FastAPI router at `POST /api/v1/graph/generate`, so the web UI and the CLI share identical generation logic with no duplication.

---

## 3. Graph Model — Node and Edge Types

A generated graph contains five node types and six structural edge types.

### Node types

| Type | ID prefix | Description |
|------|-----------|-------------|
| `Application` (CSU) | `A{n}` | Software component that publishes and/or subscribes to topics |
| `Broker` | `B{n}` | Message broker / DDS participant that routes topics |
| `Topic` | `T{n}` | Named communication channel carrying typed messages |
| `Node` (Infrastructure) | `N{n}` | Physical or virtual host running applications and brokers |
| `Library` | `L{n}` | Shared software library used by one or more applications |

### Structural edge types

| Edge | From → To | Meaning |
|------|-----------|---------|
| `PUBLISHES_TO` | Application → Topic | App produces messages on this topic |
| `SUBSCRIBES_TO` | Application → Topic | App consumes messages from this topic |
| `ROUTES` | Broker → Topic | Broker is responsible for routing this topic |
| `RUNS_ON` | Application/Broker → Node | Component is deployed on this infrastructure node |
| `USES` | Application / Library → Library | Component or library depends on this shared library; library-to-library transitive dependencies are also generated (30 % probability per library) |
| `CONNECTS_TO` | Node → Node | Network link between infrastructure nodes (30 % probability) |

These six edge types constitute the **structural graph G_structural**, which is used by the simulation stage (Step 4) to trace failure propagation. A separate **analysis graph G_analysis** is derived from G_structural by computing `DEPENDS_ON` edges, which are used exclusively by Steps 2 and 3 (analysis and prediction). The separation ensures that prediction and simulation remain independent.

---

## 4. Generation Modes

### 4.1 Scale-preset mode

The fastest way to generate a graph. Six named presets are built in:

| Preset | Applications | Topics | Brokers | Nodes | Libraries | Total nodes |
|--------|-------------|--------|---------|-------|-----------|-------------|
| `tiny` | 5 | 5 | 1 | 2 | 2 | 15 |
| `small` | 15 | 10 | 2 | 4 | 5 | 36 |
| `medium` | 50 | 30 | 3 | 8 | 10 | 101 |
| `large` | 150 | 100 | 6 | 20 | 30 | 306 |
| `jumbo` | 300 | 120 | 10 | 40 | 50 | 520 |
| `xlarge` | 500 | 300 | 10 | 50 | 100 | 960 |

In scale-preset mode, QoS values for each topic are sampled uniformly from the full option space. Node placement, publish/subscribe wiring, and library usage are assigned using simple random selection without statistical distributions. This mode is suitable for benchmarking and quick smoke tests. Total edge counts are seed-dependent; for scale-preset graphs expect roughly 4–10× the total node count in edges.

```bash
python bin/generate_graph.py --scale medium --seed 42 --output output/graph.json
```

### 4.2 Statistical-config mode (YAML)

The primary mode for validation work. A YAML configuration file fully specifies:

- exact component counts
- statistical distributions for node loading, publish counts, subscribe counts, and topic fan-in/fan-out
- categorical distributions for QoS durability, reliability, and transport priority
- role and criticality distributions for applications
- library usage distributions

When a YAML config is loaded, the generator uses `StatisticalMetric` sampling (clamped Gaussian via `random.gauss`) for continuous quantities and weighted-list sampling for categorical quantities. This produces topologies whose structural properties closely match the declared distributions, enabling repeatable validation experiments.

```bash
python bin/generate_graph.py --config input/scenario_01_autonomous_vehicle.yaml \
       --output output/av_system.json
```

`--scale` and `--config` are mutually exclusive.

### 4.3 Domain and scenario enrichment

Two optional flags add realistic naming and domain-matched QoS mappings to either generation mode.

`--domain` selects a domain name pool (`av`, `iot`, `finance`, `healthcare`, `enterprise`, `robotics`, `e-commerce`). When active, application names are drawn from domain-specific name lists (e.g. `PlanningEngine`, `RiskManager`, `VitalSignsMonitor`) rather than generic `App-{n}` identifiers. Library names, broker names, and system hierarchy labels are similarly drawn from domain pools.

`--scenario` selects a QoS mapping table that overrides the statistical distribution for each topic's QoS settings. For instance, `--scenario av` ensures that topics whose names contain navigation or safety keywords receive `RELIABLE / TRANSIENT_LOCAL / HIGH` QoS regardless of the general distribution. The available scenario values are `av`, `iot`, `finance`, `healthcare`, `hub-and-spoke`, `microservices`, and `enterprise`.

Domain and scenario can be combined with either `--scale` or `--config`. When a YAML config is used and specifies `domain:` and `scenario:` fields at the top level, those values are used automatically without needing to pass them on the CLI.

---

## 5. The `generate_graph.py` CLI

Located at `bin/generate_graph.py`. This is the primary generation entry point. In its default mode (no subcommand) it generates a single graph file. It also exposes two subcommands: `batch` (batch dataset generation — see Section 6) and `validate` (topology-class validation — see Section 5.3).

### 5.1 Arguments reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--scale` | choice | — | Named scale preset: `tiny`, `small`, `medium`, `large`, `jumbo`, `xlarge`. Mutually exclusive with `--config`. |
| `--config` | path | — | Path to a YAML scenario configuration file. Mutually exclusive with `--scale`. |
| `--output` | path | `output/graph.json` | Destination path for the generated JSON file. Parent directories are created automatically. |
| `--seed` | int | `42` | Random seed. Any integer value gives a deterministic, reproducible output. |
| `--domain` | str | — | Domain name pool: `av`, `iot`, `finance`, `healthcare`, `hub-and-spoke`, `microservices`, `enterprise`, `atm`. **No validation is performed** — an unrecognised string silently falls back to generic naming with no error. |
| `--scenario` | choice | — | QoS scenario mapping: `av`, `iot`, `finance`, `healthcare`, `hub-and-spoke`, `microservices`, `enterprise`, `atm`. |
| `--verbose` / `-v` | flag | off | Print full tracebacks on error. |

When `--config` is provided, the seed embedded in the YAML (`graph.seed`) is used and the `--seed` CLI argument is ignored. When `--scale` is provided without `--seed`, the default seed of 42 is used.

### 5.2 Usage examples

```bash
# Minimal: medium graph, default seed
python bin/generate_graph.py --scale medium --output output/graph.json

# Reproducible small graph for a unit test
python bin/generate_graph.py --scale small --seed 123 --output output/small_123.json

# Autonomous vehicle scenario from YAML config
python bin/generate_graph.py \
    --config input/scenario_01_autonomous_vehicle.yaml \
    --output output/av_system.json

# IoT scenario with domain naming
python bin/generate_graph.py \
    --config input/scenario_02_iot_smart_city.yaml \
    --domain iot \
    --output output/iot_system.json

# Quick large graph for benchmarking
python bin/generate_graph.py --scale large --seed 2024 --output output/large_bench.json

# Followed immediately by the full pipeline
python bin/run.py --all --input output/av_system.json

# Generation as stage 1 of run.py orchestrator
python bin/run.py --all --config input/scenario_01_autonomous_vehicle.yaml \
    --input output/av_system.json --output-dir output/av_results
```

### 5.3 Subcommands: batch and validate

`generate_graph.py` exposes two named subcommands beyond single-graph generation.

**`batch`** — Generates all scenario datasets in a single invocation (equivalent to the former `generate_datasets.py` standalone script, now consolidated here). Accepts all the arguments documented in Section 6.

```bash
python bin/generate_graph.py batch [OPTIONS]
```

**`validate`** — Runs topology-class validation across the generated datasets, checking that each scenario falls into the correct expected class (fan-out dominated, dense pubsub, anti-pattern / SPOF, or sparse). The implementation lives in `bin/common/dataset_validation.py`.

```bash
python bin/generate_graph.py validate [OPTIONS]
```

---

## 6. The `generate_graph.py batch` Subcommand

Invoked as `python bin/generate_graph.py batch`. Generates all scenario datasets in a single invocation, with optional multi-seed variants and legacy dataset refresh. Intended to be run once before a full validation or benchmarking session. The implementation lives in `bin/common/batch_generation.py`.

### 6.1 Arguments reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | `input/` | Directory containing `scenario_*.yaml` files. |
| `--output-dir` | `output/` | Output directory for generated JSON files. |
| `--refresh-legacy` | off | Regenerate `input/system.json` (medium, seed=42) and `input/dataset.json` (small, seed=42) using the current generator, adding `code_metrics` and `system_hierarchy`. |
| `--multi-seed` | off | Generate per-seed variants for scenarios 01–06 using all seeds in `--seeds`. Scenarios 07–09 are excluded to manage runtime. |
| `--seeds` | `42,123,456,789,2024` | Comma-separated seed list for `--multi-seed` mode. These five seeds are the standard multi-seed stability set for the thesis. |
| `--scenario` | — | Substring filter — only generate matching scenario names (e.g. `scenario_03`). |
| `--force` | off | Overwrite existing output files. Without this flag, existing files are skipped. |
| `--manifest` | `output/dataset_manifest.json` | Path for the JSON manifest recording metadata for every generated file. |
| `--dry-run` | off | Print the execution plan without writing any files. |
| `--verbose` / `-v` | off | Print topology breakdown (node counts, edge counts, QoS distribution) per scenario. |

### 6.2 Usage examples

```bash
# Generate all 9 scenario datasets (skips existing by default)
python bin/generate_graph.py batch

# Full preparation run before a validation session
python bin/generate_graph.py batch --multi-seed --refresh-legacy

# Preview the plan without writing anything
python bin/generate_graph.py batch --multi-seed --refresh-legacy --dry-run

# Regenerate only the financial and healthcare scenarios
python bin/generate_graph.py batch --scenario scenario_03 --force
python bin/generate_graph.py batch --scenario scenario_04 --force

# Generate all scenarios with verbose topology output
python bin/generate_graph.py batch --verbose

# Custom seeds for multi-seed stability sweep
python bin/generate_graph.py batch --multi-seed --seeds 42,100,200,300,400
```

### 6.3 Output layout and manifest

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

The manifest JSON records the following fields for each dataset: `scenario_name`, `source_config`, `output_path`, `seed`, `generation_mode`, `counts` (node counts by type), `edge_counts` (by edge type), `qos_distribution` (reliability/durability/priority breakdown), `criticality_distribution`, `generated_at`, `elapsed_s`, `status` (`"ok"` | `"skipped"` | `"error"`), and `error` (exception message string, populated only when `status` is `"error"`). CI pipelines that parse the manifest programmatically should check the `error` field whenever `status != "ok"`.

---

## 7. Scenario Configuration Files

Located in `input/scenario_*.yaml`. Each file is a self-contained specification for one validation scenario. The file is passed directly to `bin/generate_graph.py --config`.

### 7.1 Scenario catalogue

| File | Domain | Scale | Key topology stress | Seed |
|------|--------|-------|---------------------|------|
| `scenario_01_autonomous_vehicle.yaml` | ROS 2 / AV | Medium (80 apps) | Sensor fan-out, RELIABLE + TRANSIENT_LOCAL QoS | 1001 |
| `scenario_02_iot_smart_city.yaml` | IoT | Large (200 apps) | Massive edge-node count, VOLATILE / BEST_EFFORT flood | 2002 |
| `scenario_03_financial_trading.yaml` | HFT / Finance | Medium (60 apps) | PERSISTENT + CRITICAL priority, dense pubsub | 3003 |
| `scenario_04_healthcare.yaml` | Clinical / HIS | Medium | PERSISTENT clinical data, PHI-scoped fan-out | 4004 |
| `scenario_05_hub_and_spoke.yaml` | Anti-pattern | Medium (70 apps) | Only 2 brokers — deliberate SPOF | 5005 |
| `scenario_06_microservices.yaml` | Cloud-native | Medium | Sparse topology, low coupling, precision test | 6006 |
| `scenario_07_enterprise_xlarge.yaml` | Enterprise ESB | Jumbo (300 apps) | Scalability and performance benchmark | 7007 |
| `scenario_08_tiny_regression.yaml` | Smoke test | Tiny (12 apps) | CI regression, fully deterministic | 8008 |
| `scenario_09_xlarge_stress.yaml` | Cloud Platform | XLarge (500 apps) | True xlarge validation, thesis coverage | 9009 |
| `scenario_10_atm_system.yaml` | ATM / Aviation | Medium (26 apps) | Ultra-high reliability, safety-critical surveillance | 0042 |
| `scenario_11_broker_redundancy.yaml` | Enterprise Clearing | Small-medium (40 apps) | 12 brokers / 15 topics (ratio 0.8) — validates non-SPOF redundant brokers | 1111 |

The eleven scenarios collectively cover five topology classes: fan-out dominated (01, 02), dense pubsub (03, 04), anti-pattern / SPOF (05), sparse / well-distributed (06), and over-provisioned redundancy (11). Scenarios 07 and 09 are scalability benchmarks; scenario 08 is the CI smoke test and should always be the first to run. Scenario 10 (ATM) tests ultra-high criticality and scenario 11 tests the Availability dimension under broker redundancy.

### 7.2 YAML schema reference

Every scenario YAML follows this structure:

```yaml
graph:
  seed: <integer>           # Unique per scenario; determines full reproducibility
  domain: <string>          # Optional: av | iot | finance | healthcare | enterprise
  scenario: <string>        # Optional: matches --scenario QoS mapping table

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

All `*_stats` sections are optional. If a section is absent, the corresponding values are assigned uniformly at random from the full option space.

### 7.3 Writing a new scenario

1. Copy the closest existing scenario file and rename it `scenario_NN_<name>.yaml`.
2. Set `graph.seed` to a unique value (convention: use `NNNN` where `NN` is the scenario number, e.g. `seed: 1010` for scenario 10).
3. Adjust `graph.counts` to the desired scale. Refer to the scale presets table in Section 4.1 as a guide.
4. Set the statistical distributions to reflect the topology you want to stress. For a fan-out scenario, increase `applications_subscribing_to_this_topic.mean`; for a dense pubsub scenario, increase `pubsub` share in `app_role_distribution`; for an anti-pattern scenario, reduce `brokers` to 2 or 3.
5. Set the QoS distributions to match your domain. The QoS weight in RMAV A(v) is driven by durability and reliability; `PERSISTENT + RELIABLE` produces the highest QSPOF boost.
6. Document the **expected analysis outcomes** in a comment block at the top of the file, following the pattern of the existing scenarios. This makes the scenario self-documenting as a validation fixture.
7. Add a row to the quick-reference table in `input/scenario.md`.

---

## 8. The `StatisticalGraphGenerator`

Located at `tools/generation/generator.py`. This class performs the actual graph construction. It is instantiated by `GenerationService` and should not normally be called directly.

### 8.1 Generation pipeline

The generator runs in two passes.

**Pass 1 — Nodes.** All five node types are created first, before any edges. The order is: infrastructure nodes, brokers, topics, applications, libraries.

For each topic, QoS values (durability, reliability, transport priority) and message size are assigned. If a `domain` and `scenario` are configured, the domain-specific `get_qos_for_topic()` function maps topic names to QoS settings using keyword matching. Otherwise, values are drawn from the `qos_stats` categorical distributions.

For each application, role, app type, system hierarchy, and code metrics are assigned. **Criticality is not assigned here** — it is deferred to the two-pass assignment after topology is built (see Pass 2, step 5 below). If a `domain` dataset is active, the app name is drawn from the domain's name pool and the app type is inferred from the name using `get_app_type_for_name()`. Otherwise, names are generic (`App-{n}`) and the app type is sampled uniformly from `APP_TYPE_OPTIONS`.

**Pass 2 — Relationships.** Edges are constructed in strict dependency order because library pub/sub counts must be known before inherited application counts can be computed:

1. `RUNS_ON` — infrastructure placement for applications and brokers
2. `ROUTES` — broker-to-topic routing
3. `USES` — library dependencies (app→lib and lib→lib), required before step 4
4. `PUBLISHES_TO` / `SUBSCRIBES_TO` — publish/subscribe wiring
5. **Post-topology quality passes** — role constraint enforcement, criticality assignment

**`RUNS_ON` — cluster-affine placement.** A shared cluster→node partition is built first by randomly assigning each infrastructure node to one hierarchy cluster (`_build_cluster_to_nodes()`). Applications are then placed on nodes using `_assign_apps_to_nodes()`: with probability 0.70 an application lands on a node in its own cluster's subset; with probability 0.30 it lands on any node. This co-locates functionally related applications on shared infrastructure, making node-level structural metrics (betweenness, SPOF detection) realistic. The same cluster→node partition is reused for broker placement (see `ROUTES` below).

**`ROUTES` — cluster-affine broker placement.** Broker-to-topic routing is assigned first (each topic gets one primary broker plus a 30 % chance of a secondary). A stranded-broker guard then ensures every broker routes at least one topic. After the routes list is complete, `_rewrite_broker_placement()` reads each broker's topic assignments, determines the plurality cluster of its routed topics, and places the broker on a node from that cluster's subset. This ensures brokers are co-located with the applications they serve, improving infra-layer SPOF and betweenness accuracy.

**`USES` edges** include both application-to-library and library-to-library dependencies. Each library has a 30 % chance of depending on up to two other libraries, creating transitive dependency chains. The resulting `_uses_graph` is traversed recursively in Pass 2 step 4 to compute inherited pub/sub counts.

**Pub/sub wiring priority order.** Four strategies are tried in order; the first one whose relevant stat has a non-zero mean is used:

1. `total_publish_count_including_libraries` from `application_stats` — subtracts the inherited library contribution to derive the required direct count per application.
2. `direct_publish_count` from `application_stats` — uses per-application direct counts without library inheritance.
3. `applications_publishing_to_this_topic` from `topic_stats` — drives wiring from the topic side (fan-in perspective).
4. **Random fallback** — 1–5 publishers and 1–8 subscribers sampled per topic when no stat section defines the distribution.

All four strategies use `_sample_biased()` (p_intra = 0.65) so that applications in the same hierarchy cluster preferentially share topics. In strategies 1 and 2, the preferred pool is further filtered by `_partition_topics_by_qos_affinity()`: gateway and controller applications are steered toward `RELIABLE` / `HIGH`-priority topics; sensor applications are steered toward `BEST_EFFORT` / `LOW`-priority topics. This makes QoS semantics structurally coherent — high-priority topics attract structurally central components.

**Post-topology quality passes (step 5).**

- *Role constraint enforcement* — `_validate_role_constraints()` removes any edges that violate declared app roles: a `pub`-only application cannot appear in `subscribes_to`; a `sub`-only application cannot appear in `publishes_to`. Library edges are not affected.
- *Two-pass criticality assignment* — `_assign_criticality_two_pass()` computes a structural degree proxy (direct publish count + direct subscribe count) for each application. The top-N applications by degree proxy receive `criticality=True`, where N is the target count from the scenario's `app_criticality_distribution`. Ties are broken with a seeded random jitter so the result is deterministic. This ensures that structurally central components (hubs, articulation-point candidates) are the ones labelled critical, rather than randomly selected applications that may be structural leaf nodes.

### 8.2 Code-metrics generation

Every `Application` and `Library` node carries a `code_metrics` block. This block feeds the code-quality penalty term in RMAV M(v).

For applications, code metrics are generated from type-specific parameter tables defined in `_CODE_METRICS_PARAMS`. The six application types (`sensor`, `actuator`, `controller`, `monitor`, `gateway`, `processor`) have calibrated ranges for LOC, classes per KLOC, methods per class, WMC, LCOM, CBO, RFC, and fan-in/fan-out. Each type produces the following output schema:

```
size:       total_loc, total_classes, total_methods, total_fields
complexity: total_wmc, avg_wmc, max_wmc
cohesion:   avg_lcom, max_lcom
coupling:   avg_cbo, max_cbo, avg_rfc, max_rfc, avg_fanin, max_fanin, avg_fanout, max_fanout
```

Note that `avg_fanin` and `avg_fanout` are nested under `coupling`, not in a separate `fan` section, and there are no `coupling_afferent` / `coupling_efferent` fields.

For libraries, code metrics are generated from archetype-specific parameter tables in `_LIB_CODE_METRICS_PARAMS`. Five archetypes are used (`utility`, `framework`, `driver`, `middleware`, `protocol`), with utility and driver archetypes weighted more heavily in random selection. Framework libraries have the widest LOC range (500–6000) and the highest coupling metrics. The schema is identical to the application schema above.

### 8.3 System-hierarchy assignment

Every application and library is assigned a `system_hierarchy` block representing its position in the MIL-STD-498 decomposition hierarchy: component (CSCI), configuration item (CSC), domain (CSU cluster), and system label.

When a `domain` dataset is active, hierarchy values are drawn from `SYSTEM_HIERARCHY_POOLS[domain]`, which contains domain-specific labels (e.g. `"Navigation Software"`, `"Path Planning"` for the AV domain). When no domain is set, generic labels are drawn from `GENERIC_HIERARCHY_POOL`.

> **Resolved (April 2026):** The generator now performs a cluster pre-assignment pass before edges are created. Applications are assigned to clusters via `rng.choices()` (seeded, not strict round-robin) and topics are partitioned across clusters in round-robin index order. The `_sample_biased()` helper (p_intra = 0.65) then ensures that applications in the same cluster preferentially publish and subscribe to topics in the same cluster, making the hierarchy signal structurally meaningful for coupling analysis. Infrastructure nodes and brokers are also co-located with their cluster (see §8.1).

### 8.4 QoS assignment

Topics receive one value for each of three QoS dimensions:

- **Durability**: `VOLATILE`, `TRANSIENT_LOCAL`, `TRANSIENT`, `PERSISTENT` (in increasing persistence order)
- **Reliability**: `BEST_EFFORT`, `RELIABLE`
- **Transport priority**: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`

These values are used by the RMAV framework to compute the QSPOF term in A(v). The QoS weight for a topic is computed from `QoSPolicy.DURABILITY_SCORES`, `QoSPolicy.RELIABILITY_SCORES`, and `QoSPolicy.PRIORITY_SCORES`. Topics carrying `PERSISTENT + RELIABLE + CRITICAL` traffic receive the maximum QoS weight (1.0), which multiplies the AP_c_directed articulation-point score to produce the full A(v) term.

The QoS Gini coefficient across all topic weights measures the heterogeneity of a generated dataset. Homogeneous datasets (all topics have similar QoS) are expected to show Δρ ≈ 0 when comparing weighted vs. unweighted RMAV, while heterogeneous datasets (high Gini coefficient) are expected to show statistically significant positive Δρ. This is the core hypothesis tested in the Middleware 2026 QoS ablation experiment.

---

## 9. Output Format

The generator produces a single JSON file with the following top-level structure:

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
      }
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
      "system_hierarchy": { ... },
      "code_metrics": { ... }
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

`generation_mode` is `"statistical"` when a YAML config with `*_stats` sections was used and `"random"` for scale-preset mode. This field is used downstream by `GraphBuilder` to choose the appropriate import path.

---

## 10. Programmatic API

The generation service can be called directly from Python without going through the CLI.

```python
from tools.generation import GenerationService, load_config, generate_graph  # service layer
from tools.generation.models import GraphConfig, SCALE_PRESETS               # data models (separate import required)

# Scale-preset mode — quickest
data = generate_graph(scale="medium", seed=42)

# Statistical-config mode from a YAML file
config = load_config(Path("input/scenario_01_autonomous_vehicle.yaml"))
service = GenerationService(config=config)
data = service.generate()

# Scale-preset mode with domain enrichment
service = GenerationService(scale="medium", seed=42, domain="finance", scenario="finance")
data = service.generate()

# Inspect available presets
for name, preset in SCALE_PRESETS.items():
    print(f"{name}: {preset}")

# Feed directly into the pipeline without writing to disk
from saag import Pipeline
pipeline = Pipeline.from_data(data)
result = pipeline.analyze().simulate().validate().run()
```

---

## 11. REST API Export Formats

The Software-as-a-Graph REST API provides several ways to export graph data. These exports are categorized into two distinct views:

### 11.1 Persistence View (Re-importable)

This view provides a high-fidelity representation of the graph structure that can be used for backup, migration, or as input to the `import_graph.py` tool.

- **Endpoints**: `POST /api/v1/graph/export-neo4j-data`, `POST /api/v1/graph/export-persistence`
- **Response Label**: `"export_format": "persistence"`
- **Shape**: Nested structure containing keys like `nodes`, `brokers`, `topics`, `applications`, `libraries`, and `relationships`.
- **Best for**: Backing up the database, migrating between instances, or programmatically modifying the graph and re-importing it.

### 11.2 Analysis View (Visualization only)

This view provides a flattened, derived representation of the graph optimized for visualization and external graph analysis tools.

- **Endpoints**: `POST /api/v1/graph/export`, `POST /api/v1/graph/export-limited`
- **Response Label**: `"export_format": "analysis"`
- **Warning**: This format **CANNOT** be re-imported into Software-as-a-Graph. It lacks the internal nesting required by the import pipeline.
- **Shape**: Flat lists of `components` and `edges`.
- **Best for**: D3.js visualization, NetworkX analysis, or CSV exports for spreadsheets.

---

## 12. Known Limitations and Open Issues

*Document last updated: April 2026. Maintained alongside `tools/generation/` and `bin/generate_graph.py`.*

| # | Area | Description | Status |
|---|------|-------------|--------|
| 1 | `--domain` validation | Invalid domain strings are silently accepted and fall back to generic naming with no warning. Passing an unsupported domain name is a silent no-op. | Open |
| 2 | `generation_mode` field | The `metadata.generation_mode` field is `"statistical"` only when a YAML config with at least one `*_stats` section is provided. A YAML file that specifies only `counts` and no stats sections will produce `generation_mode: "random"`, even though it was loaded from a YAML config. | Open |
| 3 | Pub/sub duplicate edges | The pub/sub wiring strategies do not deduplicate edges: the same `(app, topic)` pair can appear more than once in `publishes_to` / `subscribes_to` if both topic-driven and app-driven wiring write the same edge. The import pipeline deduplicates on ingest; downstream consumers reading the raw JSON should not assume uniqueness. | Open |
| 4 | Broker guard semantics | The unrouted-broker guard (round-robin assignment) is deterministic but can assign a stranded broker to an already over-routed topic, skewing broker betweenness scores in topologies with many brokers and few topics. | Open |
| 5 | Hierarchy clustering | Hierarchy labels were formerly assigned independently from a flat pool with no structural effect. As of April 2026 the generator performs a seeded cluster pre-assignment and uses `_sample_biased()` (p_intra = 0.65) to make intra-cluster pub/sub edges more probable. | **Resolved** |
| 6 | Infrastructure placement | `RUNS_ON` edges were previously assigned uniformly at random (or sequentially by count distribution), with no cluster awareness. As of April 2026 `_assign_apps_to_nodes()` applies 70 % cluster-affine placement, co-locating functionally related apps on the same infrastructure node. | **Resolved** |
| 7 | Broker placement | Brokers were previously placed on nodes via round-robin index, ignoring which cluster of topics they route. As of April 2026 `_rewrite_broker_placement()` reads the completed routes list and places each broker on a node in its plurality cluster. | **Resolved** |
| 8 | Criticality coherence | Criticality was previously sampled before topology was built, so critical apps could be structural leaf nodes. As of April 2026 `_assign_criticality_two_pass()` assigns `critical=True` after topology is built, biased toward high-degree (hub) applications. | **Resolved** |
| 9 | Role constraint drift | The `topic_stats` pub/sub wiring path could assign subscribe edges to `pub`-only apps. As of April 2026 `_validate_role_constraints()` removes such violations before deduplication. | **Resolved** |
| 10 | QoS–topology coherence | Topic selection during pub/sub wiring was cluster-biased but QoS-agnostic. As of April 2026 `_partition_topics_by_qos_affinity()` steers gateway/controller apps toward `RELIABLE`/`HIGH` topics and sensor apps toward `BEST_EFFORT`/`LOW` topics. | **Resolved** |
