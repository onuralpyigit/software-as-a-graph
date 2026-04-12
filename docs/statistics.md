# Statistics Calculator Module

Comprises two distinct layers for analyzing system topology and communication patterns:
1. **Visualization Statistics** (`api.statistics`): Operates on JSON exports to feed "Extras" charts.
2. **Core Structural Statistics** (`src.analysis.statistics`): Operates on `GraphData` in the hexagonal core for topological analysis.

---

## Quick Start (Visualization Layer)

```python
import json
from backend.api.statistics import extract_cross_cutting_data, compute_all_extras_statistics

with open("dataset.json") as f:
    data = json.load(f)

cc = extract_cross_cutting_data(data)
stats = compute_all_extras_statistics(cc)
```

## Module Ownership Policy

| Module | Scope | Input | Typical Use Case |
|---|---|---|---|
| `backend/api/statistics.py` | Visualization | Raw JSON (`dataset.json`) | Populating scatter plots, heatmaps, and distribution charts in the dashboard. |
| `backend/src/analysis/statistics.py` | Core Analysis | `GraphData` (Domain Models) | Identifying architectural smells, SPOFs, and computing graph-theoretic metrics. |

---

## 1. Visualization Statistics (API Layer)

### Data Classes

| Class | Purpose |
|---|---|
| `DescriptiveStats` | Numeric statistics: count, mean, median, std, min/max, quartiles, IQR, upper fence |
| `CategoricalStats` | Distribution statistics: total count, category count, mode, mode percentage |

### Per-Chart Statistics

| Function | Chart | Key Outputs |
|---|---|---|
| `compute_topic_bandwidth` | Topic Size × Subscribers | bandwidth per topic, IQR outliers |
| `compute_qos_risk_stats` | QoS Risk Scatter | **(Optional)** risk score per topic, outliers |
| `compute_app_balance` | App Pub/Sub Balance | per-app I/O load, quadrant classification |
| `compute_topic_fanout` | Topic Fanout | publisher × subscriber fanout, pattern counts |
| `compute_cross_node_heatmap` | Cross-Node Heatmap | NxN communication matrix, intra/inter-node traffic |
| `compute_node_comm_load` | Node Communication Load | per-node pub/sub totals, coefficient of variation |
| `compute_domain_comm` | Domain Communication | domain-to-domain matrix, cross-domain pair counts |
| `compute_criticality_io` | Criticality × I/O | critical vs. normal app I/O comparison |
| `compute_lib_dependency` | Library Dependency | in/out degree for apps and libraries |
| `compute_node_critical_density` | Node Critical Density| critical vs. normal app counts per node |
| `compute_domain_diversity` | Domain Diversity | app count, topic count, and I/O per domain |

---

## 2. Core Structural Statistics (Analysis Layer)

Accessed via `src.analysis.statistics_service.StatisticsService`. These metrics operate on the live graph structure.

### Topological Metrics

| Metric | Function | Description |
|---|---|---|
| **Degree Distribution** | `get_degree_distribution` | Identifies Hubs (mean + 2σ) and isolated nodes. |
| **Connectivity Density** | `get_connectivity_density` | System coupling factor (Sparse < 0.05 to Very Dense > 0.30). |
| **Clustering Coefficient** | `get_clustering_coefficient`| Measures "triangle" formation (local interconnectedness). |
| **Dependency Depth** | `get_dependency_depth` | BFS hierarchy analysis to find deepest components and roots/leaves. |
| **Component Isolation** | `get_component_isolation`| Classifies components as Source, Sink, Bidirectional, or Isolated. |
| **Redundancy & SPOF** | `get_component_redundancy`| Identifies Single Points of Failure and bridge components. |

---

## 3. REST API Reference

All endpoints return a `{"success": bool, "stats": {...}, "computation_time_ms": float}` envelope.

| Endpoint | Method | Response Model |
|---|---|---|
| `/api/v1/stats/` | `POST` | Full Visualization Statistics ("Extras") |
| `/api/v1/stats/summary` | `POST` | Overall graph node/edge counts |
| `/api/v1/stats/degree-distribution`| `POST` | `DegreeDistributionResponse` |
| `/api/v1/stats/connectivity-density`| `POST` | `ConnectivityDensityResponse` |
| `/api/v1/stats/dependency-depth` | `POST` | `DependencyDepthResponse` |
| `/api/v1/stats/component-redundancy`| `POST` | SPOFs and Resilience Score |

---

## Expected Input Format (Visualization)

`extract_cross_cutting_data` expects a JSON dict with:

```json
{
  "nodes": [{"id": "node-1", "name": "Worker-A"}, ...],
  "applications": [
    {
      "id": "app-1", 
      "name": "Processor", 
      "criticality": true, 
      "role": "Publisher",
      "system_hierarchy": {"css_name": "Domain-X"}
    }, ...
  ],
  "topics": [
    {
      "id": "topic-1", 
      "size": 1024, 
      "qos": {"durability": "High", "reliability": "Reliable", "transport_priority": "High"}
    }, ...
  ],
  "relationships": {
    "runs_on": [{"from": "app-1", "to": "node-1"}, ...],
    "publishes_to": [{"from": "app-1", "to": "topic-1"}, ...],
    "subscribes_to": [{"from": "app-2", "to": "topic-1"}, ...],
    "uses": [{"from": "app-1", "to": "lib-1"}, ...]
  }
}
```

## Dependencies

- `numpy` — array operations, percentiles
- Python stdlib: `math`, `statistics`, `dataclasses`, `collections`
