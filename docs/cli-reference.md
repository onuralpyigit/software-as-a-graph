# CLI Reference

This document provides a comprehensive reference for command-line tools.

---

## Table of Contents

1. [Overview](#overview)
2. [run.py - End-to-End Pipeline](#runpy---end-to-end-pipeline)
3. [generate_graph.py - Graph Generation](#generate_graphpy---graph-generation)
4. [validate_graph.py - Validation](#validate_graphpy---validation)
5. [simulate_graph.py - Simulation](#simulate_graphpy---simulation)
6. [visualize_graph.py - Visualization](#visualize_graphpy---visualization)
7. [import_graph.py - Neo4j Import](#import_graphpy---neo4j-import)
8. [analyze_graph.py - GDS Analysis](#analyze_graphpy---gds-analysis)

---

## Overview

The toolkit provides several CLI tools for different stages of the methodology:

| Tool | Purpose | Neo4j Required |
|------|---------|----------------|
| `run.py` | End-to-end pipeline | No |
| `generate_graph.py` | Generate test graphs | No |
| `validate_graph.py` | Statistical validation | No |
| `simulate_graph.py` | Failure simulation | No |
| `visualize_graph.py` | Generate visualizations | No |
| `import_graph.py` | Import to Neo4j | Yes |
| `analyze_graph.py` | GDS-based analysis | Yes |

---

## run.py - End-to-End Pipeline

Complete methodology pipeline in one command.

### Usage

```bash
python run.py [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--scenario` | Domain scenario | `iot` |
| `--scale` | System scale | `medium` |
| `--quick` | Quick demo (small scale) | False |
| `--input FILE` | Use existing graph file | None |
| `--output DIR` | Output directory | `output/` |
| `--skip-generate` | Skip graph generation | False |
| `--skip-validate` | Skip validation | False |
| `--skip-visualize` | Skip visualization | False |
| `--spearman-target` | Spearman target | 0.70 |
| `--f1-target` | F1 target | 0.90 |
| `--cascade` | Enable cascade simulation | False |
| `--seed` | Random seed | None |
| `--verbose, -v` | Verbose output | False |
| `--quiet, -q` | Minimal output | False |

### Examples

```bash
# Quick demo
python run.py --quick

# Full analysis with IoT scenario
python run.py --scenario iot --scale medium --cascade

# Use existing graph
python run.py --input my_graph.json --skip-generate

# Custom targets
python run.py --spearman-target 0.75 --f1-target 0.85

# Save to custom directory
python run.py --output results/experiment1/
```

### Output Files

```
output/
├── graph.json              # Generated graph
├── validation_results.json # Validation metrics
├── simulation_results.json # Simulation data
├── dashboard.html          # Interactive dashboard
└── summary.txt             # Text summary
```

---

## generate_graph.py - Graph Generation

Generate synthetic pub-sub system graphs.

### Usage

```bash
python generate_graph.py [OPTIONS]
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--scale` | `-s` | Scale preset | `medium` |
| `--scenario` | `-c` | Domain scenario | `generic` |
| `--seed` | | Random seed | None |
| `--antipatterns` | `-a` | Anti-patterns to inject | None |
| `--output` | `-o` | Output file path | None |
| `--preview` | `-p` | Preview without saving | False |
| `--list-options` | `-l` | List available options | False |
| `--quiet` | `-q` | Minimal output | False |

### Scale Presets

| Scale | Applications | Brokers | Topics | Nodes |
|-------|-------------|---------|--------|-------|
| `tiny` | 5 | 1 | 8 | 2 |
| `small` | 10 | 2 | 20 | 4 |
| `medium` | 30 | 4 | 60 | 8 |
| `large` | 100 | 8 | 200 | 20 |
| `xlarge` | 300 | 16 | 600 | 50 |

### Scenarios

| Scenario | Description |
|----------|-------------|
| `generic` | General-purpose system |
| `iot` | IoT sensor network |
| `financial` | Trading platform |
| `healthcare` | Medical monitoring |
| `autonomous_vehicle` | ROS 2 system |
| `smart_city` | Urban infrastructure |

### Anti-Patterns

| Pattern | Description |
|---------|-------------|
| `god_topic` | Topic with many connections |
| `spof` | Single point of failure |
| `chatty` | Over-connected application |
| `bottleneck` | Overloaded broker |

### Examples

```bash
# List available options
python generate_graph.py --list-options

# Generate medium IoT system
python generate_graph.py --scale medium --scenario iot --output system.json

# Preview without saving
python generate_graph.py --scale small --preview

# With anti-patterns
python generate_graph.py --scale medium --antipatterns god_topic spof -o graph.json

# Reproducible generation
python generate_graph.py --scale large --seed 42 --output graph.json
```

---

## validate_graph.py - Validation

Run statistical validation on graphs.

### Usage

```bash
python validate_graph.py [OPTIONS]
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input graph JSON (required) | None |
| `--output` | `-o` | Output file/directory | None |
| `--method` | `-m` | Analysis method | `composite` |
| `--spearman` | | Spearman target | 0.70 |
| `--f1` | | F1 target | 0.90 |
| `--cascade` | | Enable cascade | False |
| `--cascade-threshold` | | Cascade threshold | 0.5 |
| `--cascade-prob` | | Cascade probability | 0.7 |
| `--compare` | | Compare methods | False |
| `--methods` | | Methods to compare | All |
| `--format` | | Output formats | `json` |
| `--seed` | | Random seed | None |
| `--verbose` | `-v` | Verbose output | False |
| `--quiet` | `-q` | Minimal output | False |

### Analysis Methods

| Method | Description |
|--------|-------------|
| `composite` | Weighted combination (default) |
| `betweenness` | Betweenness centrality only |
| `degree` | Degree centrality only |
| `pagerank` | PageRank only |
| `message_path` | Message path centrality only |

### Examples

```bash
# Basic validation
python validate_graph.py --input graph.json

# With cascade simulation
python validate_graph.py --input graph.json --cascade

# Custom targets
python validate_graph.py --input graph.json --spearman 0.75 --f1 0.85

# Compare methods
python validate_graph.py --input graph.json --compare

# Specific methods
python validate_graph.py --input graph.json --compare \
    --methods betweenness degree composite

# Export results
python validate_graph.py --input graph.json --output results/ --format json csv

# Verbose output
python validate_graph.py --input graph.json --verbose
```

### Output

```
================================================================================
                        VALIDATION RESULTS
================================================================================

STATUS: PASSED
Components Validated: 77
Analysis Method: composite

CORRELATION METRICS
  Spearman ρ:  0.8081 (p < 0.001) ✓
  Pearson r:   0.7842 (p < 0.001)
  Kendall τ:   0.6523

CLASSIFICATION METRICS
  Precision:   0.8750 ✓
  Recall:      0.8750 ✓
  F1-Score:    0.8750 ⚠ (target: 0.90)

RANKING METRICS
  Top-3 Overlap:  100.0%
  Top-5 Overlap:   80.0% ✓
  Top-10 Overlap:  70.0%

================================================================================
```

---

## simulate_graph.py - Simulation

Run failure and event simulations.

### Usage

```bash
python simulate_graph.py [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input graph JSON | Required |
| `--output` | Output file | None |
| `--component` | Simulate single component | None |
| `--attack` | Run attack simulation | False |
| `--strategy` | Attack strategy | `highest_betweenness` |
| `--count` | Attack count | 3 |
| `--campaign` | Exhaustive campaign | False |
| `--cascade` | Enable cascade | False |
| `--cascade-threshold` | Cascade threshold | 0.5 |
| `--cascade-prob` | Cascade probability | 0.7 |
| `--max-cascade` | Max cascade depth | 5 |
| `--event` | Event simulation | False |
| `--duration` | Simulation duration (ms) | 10000 |
| `--message-rate` | Messages per second | 100 |
| `--load-test` | Load testing mode | False |
| `--chaos` | Chaos engineering mode | False |
| `--seed` | Random seed | None |
| `--json` | JSON output | False |
| `--verbose` | Verbose output | False |

### Simulation Modes

| Mode | Option | Description |
|------|--------|-------------|
| Single Failure | `--component B1` | Simulate one failure |
| Attack | `--attack` | Targeted multi-failure |
| Campaign | `--campaign` | Exhaustive all-component |
| Event | `--event` | Message flow simulation |
| Load Test | `--load-test` | Ramping rate test |
| Chaos | `--chaos` | Random failures |

### Attack Strategies

| Strategy | Description |
|----------|-------------|
| `highest_betweenness` | Target highest BC |
| `highest_degree` | Target highest degree |
| `highest_pagerank` | Target highest PR |
| `articulation_points` | Target APs first |
| `random` | Random selection |

### Examples

```bash
# Single failure
python simulate_graph.py --input graph.json --component B1

# With cascade
python simulate_graph.py --input graph.json --component B1 --cascade

# Attack simulation
python simulate_graph.py --input graph.json --attack --count 5

# Exhaustive campaign
python simulate_graph.py --input graph.json --campaign --cascade

# Event simulation
python simulate_graph.py --input graph.json --event --duration 30000

# Load testing
python simulate_graph.py --input graph.json --load-test \
    --initial-rate 10 --peak-rate 500

# Chaos engineering
python simulate_graph.py --input graph.json --chaos \
    --failure-probability 0.01 --recovery-probability 0.1

# Export results
python simulate_graph.py --input graph.json --campaign --output results.json
```

---

## visualize_graph.py - Visualization

Generate visualizations and dashboards.

### Usage

```bash
python visualize_graph.py [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input graph JSON | Required |
| `--output` | Output HTML file | `visualization.html` |
| `--dashboard` | Generate dashboard | False |
| `--multi-layer` | Multi-layer view | False |
| `--run-analysis` | Compute criticality | False |
| `--validation` | Validation results JSON | None |
| `--simulation` | Simulation results JSON | None |
| `--layout` | Graph layout | `physics` |
| `--title` | Dashboard title | Auto |
| `--verbose` | Verbose output | False |

### Visualization Types

| Type | Option | Description |
|------|--------|-------------|
| Network | (default) | Interactive graph |
| Multi-layer | `--multi-layer` | Layer-separated view |
| Dashboard | `--dashboard` | Full dashboard |

### Examples

```bash
# Basic network visualization
python visualize_graph.py --input graph.json --output network.html

# Multi-layer view
python visualize_graph.py --input graph.json --multi-layer --output layers.html

# Dashboard with analysis
python visualize_graph.py --input graph.json --dashboard --run-analysis

# Dashboard with validation results
python visualize_graph.py --input graph.json --dashboard \
    --validation validation.json --simulation simulation.json

# Custom layout
python visualize_graph.py --input graph.json --layout hierarchical

# Custom title
python visualize_graph.py --input graph.json --dashboard \
    --title "Production System Analysis"
```

---

## import_graph.py - Neo4j Import

Import graphs into Neo4j database.

### Usage

```bash
python import_graph.py [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input graph JSON | Required |
| `--uri` | Neo4j URI | `bolt://localhost:7687` |
| `--user` | Neo4j username | `neo4j` |
| `--password` | Neo4j password | `password` |
| `--database` | Database name | `neo4j` |
| `--clear` | Clear database first | False |
| `--batch-size` | Import batch size | 100 |
| `--no-depends-on` | Skip DEPENDS_ON | False |
| `--analytics` | Show analytics | False |
| `--export-queries` | Export Cypher file | None |
| `--export-stats` | Export stats JSON | None |
| `--export-graph` | Export from Neo4j | None |
| `--verbose` | Verbose output | False |
| `--quiet` | Minimal output | False |

### Examples

```bash
# Basic import
python import_graph.py --input graph.json

# With credentials
python import_graph.py --input graph.json \
    --uri bolt://neo4j:7687 \
    --user neo4j \
    --password secret

# Clear and reimport
python import_graph.py --input graph.json --clear

# Show analytics after import
python import_graph.py --input graph.json --analytics

# Export queries for review
python import_graph.py --input graph.json --export-queries queries.cypher

# Export statistics
python import_graph.py --input graph.json --export-stats stats.json
```

---

## analyze_graph.py - GDS Analysis

Neo4j Graph Data Science analysis (requires Neo4j).

### Usage

```bash
python analyze_graph.py [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--uri` | Neo4j URI | `bolt://localhost:7687` |
| `--user` | Neo4j username | `neo4j` |
| `--password` | Neo4j password | `password` |
| `--output` | Output directory | `output/` |
| `--method` | Analysis method | `composite` |
| `--export` | Export format | `json` |
| `--classify` | Run classification | False |
| `--k-factor` | Box-plot k factor | 1.5 |
| `--verbose` | Verbose output | False |

### Examples

```bash
# Basic analysis
python analyze_graph.py

# With credentials
python analyze_graph.py \
    --uri bolt://neo4j:7687 \
    --user neo4j \
    --password secret

# Specific method
python analyze_graph.py --method betweenness

# With classification
python analyze_graph.py --classify --k-factor 1.5

# Export results
python analyze_graph.py --output results/ --export json csv
```

---

## Navigation

- **Previous:** [← API Reference](api-reference.md)
- **Back to:** [Documentation Index](index.md)
