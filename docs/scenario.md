# Graph Configuration Scenarios
# Test Suite for Graph-Based Criticality Prediction Methodology

Each YAML file in this directory is a `--config` argument for
`bin/generate_graph.py` and targets a distinct topological and
domain scenario for validating the six-step methodology.

---

## Quick Reference

| File | Domain | Scale | Key Stress | Seed |
|------|--------|-------|-----------|------|
| `scenario_01_autonomous_vehicle.yaml`  | ROS 2 / AV     | Medium  | Sensor fan-out, RELIABLE+TRANSIENT_LOCAL QoS        | 1001 |
| `scenario_02_iot_smart_city.yaml`      | IoT             | Large   | Massive node count, VOLATILE/BEST_EFFORT flood      | 2002 |
| `scenario_03_financial_trading.yaml`   | HFT / Finance   | Medium  | PERSISTENT+CRITICAL priority, dense pubsub          | 3003 |
| `scenario_04_healthcare.yaml`          | Clinical / HIS  | Medium  | PERSISTENT clinical data, PHI-scoped fan-out        | 4004 |
| `scenario_05_hub_and_spoke.yaml`       | Anti-pattern    | Medium  | Only 2 brokers → deliberate SPOF                    | 5005 |
| `scenario_06_microservices.yaml`       | Cloud-native    | Medium  | Sparse topology, low coupling, precision check      | 6006 |
| `scenario_07_enterprise_xlarge.yaml`   | Enterprise ESB  | **Jumbo** (300 apps) | Scalability + performance benchmark  | 7007 |
| `scenario_08_tiny_regression.yaml`     | Smoke test      | Tiny    | CI regression, fully deterministic, fast            | 8008 |
| `scenario_09_xlarge_stress.yaml`       | Cloud Platform  | XLarge (500 apps) | True xlarge validation, thesis coverage gap | 9009 |
| `scenario_10_atm_system.yaml`          | ATM / Aviation  | Medium  | Critical surveillance, high reliability focus       | 0042 |

> **Scale note — scenario_07:** The enterprise scenario uses 300 applications, which sits between
> the `large` preset (150 apps) and the `xlarge` preset (500 apps).  It must be run with
> `--config` (not `--scale`).  The `jumbo` preset (`--scale jumbo`) targets the same counts
> (300 apps / 120 topics / 10 brokers / 40 nodes / 50 libs) and can be used for quick
> ad-hoc runs at this scale without a YAML config file.

---

## Usage

```bash
# Generate a single scenario
python bin/generate_graph.py \
  --config input/scenario_01_autonomous_vehicle.yaml \
  --output output/av_system.json

# Run the full pipeline on a generated dataset
python bin/run.py --all --input output/av_system.json

# Run all scenarios in sequence (bash)
for cfg in input/scenario_*.yaml; do
  name=$(basename "$cfg" .yaml)
  python bin/generate_graph.py --config "$cfg" --output "output/${name}.json"
  python bin/run.py --all --input "output/${name}.json" --output-dir "output/${name}_results"
done
```

---

## Scale Presets Reference

The named `--scale` presets and their component counts:

| Preset   | Apps | Topics | Brokers | Nodes | Libs | Matches scenario |
|----------|------|--------|---------|-------|------|-----------------|
| `tiny`   | 5    | 5      | 1       | 2     | 2    | 08 (smoke test) |
| `small`  | 15   | 10     | 2       | 4     | 5    | —               |
| `medium` | 50   | 30     | 3       | 8     | 10   | 01, 03, 04, 05, 06 |
| `large`  | 150  | 100    | 6       | 20    | 30   | 02              |
| `jumbo`  | 300  | 120    | 10      | 40    | 50   | 07 (enterprise) |
| `xlarge` | 500  | 300    | 10      | 50    | 100  | 09 (stress)     |

`--scale jumbo` and `--scale xlarge` require `--config` to reproduce the full statistical
distributions of scenarios 07 and 09 respectively; the preset alone gives the same counts
with uniform random QoS and topology.

---

## Design Rationale

### Topology Coverage

The ten scenarios collectively cover the five major topology classes
identified in the thesis:

1. **Fan-out dominated** (AV, IoT) — many subscribers per topic;
   broker and topic betweenness centrality are the primary criticality driver.

2. **Dense pubsub** (Finance, Healthcare) — most apps are both publishers
   and subscribers; articulation-point detection and QoS weight are decisive.

3. **Anti-pattern / SPOF** (Hub-and-Spoke) — structural vulnerability is
   deliberately encoded; validates that the methodology catches what a
   human architect would flag in a review.

4. **Sparse / well-distributed** (Microservices) — challenges the
   box-plot classifier to avoid over-flagging; validates precision.

5. **Safety-Critical Real-time** (ATM) — ultra-reliable, high-priority
   surveillance feeds; validates criticality modulation for transport priority.

### QoS Weight Variation

| Scenario | Dominant Durability | Dominant Reliability | Dominant Priority |
|----------|--------------------|--------------------|------------------|
| 01 AV    | TRANSIENT_LOCAL    | RELIABLE           | HIGH             |
| 02 IoT   | VOLATILE           | BEST_EFFORT        | LOW              |
| 03 Finance | PERSISTENT       | RELIABLE           | HIGH/CRITICAL    |
| 04 Healthcare | PERSISTENT    | RELIABLE           | HIGH             |
| 05 Hub   | TRANSIENT_LOCAL    | RELIABLE           | MEDIUM           |
| 06 µSvc  | TRANSIENT_LOCAL    | RELIABLE           | MEDIUM           |
| 07 Enterprise | mixed         | RELIABLE           | MEDIUM           |
| 08 Tiny  | balanced           | balanced           | balanced         |
| 10 ATM   | VOLATILE           | RELIABLE           | HIGH/CRITICAL    |

### Expected Validation Thresholds

All scenarios should satisfy the core thesis validation targets:

- Spearman ρ (predicted criticality vs. simulated impact) ≥ **0.7**
- F1-score (critical classification) ≥ **0.9**
- Precision ≥ **0.85**, Recall ≥ **0.80**

Scenario 06 (microservices) is the hardest precision test.
Scenario 07 (enterprise) is the primary scalability benchmark.
Scenario 08 (tiny) is the CI smoke test and should always pass first.

---

## Adding New Scenarios

Copy any existing file and adjust:

1. `graph.seed` — use a unique value to avoid dataset collision
2. `graph.counts` — set the component counts for your scale
3. Adjust `*_stats` distributions to reflect your domain's topology
4. Document the **expected analysis outcomes** in the header comment
5. Add a row to the quick-reference table above
