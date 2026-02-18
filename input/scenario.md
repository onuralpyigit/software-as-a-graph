# Graph Configuration Scenarios
# Test Suite for Graph-Based Criticality Prediction Methodology

Each YAML file in this directory is a `--config` argument for
`bin/generate_graph.py` and targets a distinct topological and
domain scenario for validating the six-step methodology.

---

## Quick Reference

| File | Domain | Scale | Key Stress | Seed |
|------|--------|-------|-----------|------|
| `scenario_01_autonomous_vehicle.yaml`  | ROS 2 / AV     | Medium  | Sensor fan-out, RELIABLE+TRANSIENT_LOCAL QoS  | 1001 |
| `scenario_02_iot_smart_city.yaml`      | IoT             | Large   | Massive node count, VOLATILE/BEST_EFFORT flood  | 2002 |
| `scenario_03_financial_trading.yaml`   | HFT / Finance   | Medium  | PERSISTENT+CRITICAL priority, dense pubsub      | 3003 |
| `scenario_04_healthcare.yaml`          | Clinical / HIS  | Medium  | PERSISTENT clinical data, PHI-scoped fan-out    | 4004 |
| `scenario_05_hub_and_spoke.yaml`       | Anti-pattern    | Medium  | Only 2 brokers → deliberate SPOF                | 5005 |
| `scenario_06_microservices.yaml`       | Cloud-native    | Medium  | Sparse topology, low coupling, precision check  | 6006 |
| `scenario_07_enterprise_xlarge.yaml`   | Enterprise ESB  | XLarge  | 300 apps — scalability + performance benchmark  | 7007 |
| `scenario_08_tiny_regression.yaml`     | Smoke test      | Tiny    | CI regression, fully deterministic, fast        | 8008 |

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

## Design Rationale

### Topology Coverage

The eight scenarios collectively cover the four major topology classes
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
