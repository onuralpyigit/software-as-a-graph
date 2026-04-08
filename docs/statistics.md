# Statistics Calculator Module

Computes descriptive and categorical statistics for distributed publish-subscribe system topologies. Operates on raw aggregated JSON data (e.g. `dataset.json`) and produces per-chart statistics used by the visualization layer.

## Quick Start

```python
import json
from statistics import extract_cross_cutting_data, compute_all_extras_statistics

with open("dataset.json") as f:
    data = json.load(f)

cc = extract_cross_cutting_data(data)
stats = compute_all_extras_statistics(cc)
```

## Module Structure

### Data Classes

| Class | Purpose |
|---|---|
| `DescriptiveStats` | Numeric statistics: count, mean, median, std, min/max, quartiles, IQR, outlier fences |
| `CategoricalStats` | Distribution statistics: total count, category count, mode, mode percentage |

### Core Functions

| Function | Description |
|---|---|
| `calculate_descriptive_stats(values)` | Computes `DescriptiveStats` for a list of floats |
| `calculate_categorical_stats(category_counts)` | Computes `CategoricalStats` from a `{category: count}` dict |
| `sort_entities_by_metric(entities, metric_key)` | Sorts entity dicts by a numeric field, returns `(id, name, value)` tuples |

### Outlier Detection

| Function | Description |
|---|---|
| `find_1d_outliers_iqr(values)` | Returns `(lower_fence, upper_fence, iqr)` using the 1.5×IQR method |
| `calculate_outliers(ranked_list, outlier_stats)` | Identifies outlier entries from a ranked list given precomputed `DescriptiveStats` |

### Data Extraction

| Function | Description |
|---|---|
| `extract_cross_cutting_data(raw_data)` | Preprocesses raw JSON into lookup dictionaries (node/app/topic maps, pub/sub counts, criticality, domain, library relations) |

### Per-Chart Statistics

Each function takes the cross-cutting data dict (`cc`) and returns a dict with chart-specific arrays, outlier lists, and a `summary` sub-dict.

| Function | Chart | Key Outputs |
|---|---|---|
| `compute_topic_bandwidth_stats` | Topic Size × Subscribers | bandwidth per topic, IQR outliers |
| `compute_qos_risk_stats` | QoS Risk Scatter | risk score per topic (durability × reliability × priority × log₂(size+1)), outliers |
| `compute_app_balance_stats` | App Pub/Sub Balance | per-app I/O load, quadrant classification (high-I/O, consumer, producer, low) |
| `compute_topic_fanout_stats` | Topic Fanout | publisher × subscriber fanout, pattern counts (1:N, N:1, N:M, orphan) |
| `compute_cross_node_heatmap_stats` | Cross-Node Heatmap | NxN communication matrix, intra/inter-node traffic, outlier pairs |
| `compute_node_comm_load_stats` | Node Communication Load | per-node pub/sub totals, coefficient of variation |
| `compute_domain_comm_stats` | Domain Communication | domain-to-domain matrix, cross-domain pair counts |
| `compute_criticality_io_stats` | Criticality × I/O | critical vs. normal app I/O comparison, critical/normal ratio |
| `compute_lib_dependency_stats` | Library Dependency Density | in/out degree for apps and libraries in the USES graph |
| `compute_node_critical_density_stats` | Node Critical Density | critical vs. normal app counts per node |
| `compute_domain_diversity_stats` | Domain Diversity | app count, topic count, and I/O per domain |

### Top-Level Entry Point

```python
compute_all_extras_statistics(cc, risk_weight_fn=None, w2name=None)
```

Runs all per-chart statistics in one call. The `qos_risk` chart is included only when `risk_weight_fn` is provided.

**Parameters:**
- `cc` — Cross-cutting data from `extract_cross_cutting_data`
- `risk_weight_fn` — `Callable(dimension, value) -> float` for QoS risk weighting (optional)
- `w2name` — `Dict[str, Dict[float, str]]` mapping weights to display names per QoS dimension (optional)

## Expected Input Format

`extract_cross_cutting_data` expects a JSON dict with:

```
{
  "nodes": [{"id": ..., "name": ...}, ...],
  "applications": [{"id": ..., "name": ..., "criticality": bool, "role": ..., "system_hierarchy": {"css_name": ...}}, ...],
  "topics": [{"id": ..., "name": ..., "size": int, "qos": {"durability": ..., "reliability": ..., "transport_priority": ...}}, ...],
  "libraries": [{"id": ..., "name": ...}, ...],
  "relationships": {
    "runs_on": [{"from": app_id, "to": node_id}, ...],
    "publishes_to": [{"from": app_id, "to": topic_id}, ...],
    "subscribes_to": [{"from": app_id, "to": topic_id}, ...],
    "uses": [{"from": app_id, "to": lib_id}, ...]
  }
}
```

## Dependencies

- `numpy` — array operations, percentiles
- Python stdlib: `math`, `statistics`, `dataclasses`, `typing`
