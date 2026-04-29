# Statistics Calculator Module

Comprises two distinct layers for analyzing system topology and communication patterns:
1. **Visualization Statistics** (`api.statistics`): Operates on JSON exports to feed "Extras" charts.
2. **Core Structural Statistics** (`saag.analysis.statistics`): Operates on `GraphData` in the hexagonal core for topological analysis.

---

## Quick Start (Visualization Layer)

```python
import json
from api.statistics import extract_cross_cutting_data, compute_all_extras_statistics

with open("dataset.json") as f:
    data = json.load(f)

cc = extract_cross_cutting_data(data)
stats = compute_all_extras_statistics(cc)
```

## Module Ownership Policy

| Module | Scope | Input | Typical Use Case |
|---|---|---|---|
| `api/statistics.py` | Visualization | Raw JSON (`dataset.json`) | Populating scatter plots, heatmaps, and distribution charts in the dashboard. |
| `saag/analysis/statistics.py` | Core Analysis | `GraphData` (Domain Models) | Identifying architectural smells, SPOFs, and computing graph-theoretic metrics. |

> [!WARNING]
> Do not mix the outputs or import paths of these two modules. The API layer relies on `dataset.json` extracts and outputs JSON-safe dictionaries, while the Core Analysis layer operates directly on live `GraphData` objects within the hexagonal architecture.

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
| `compute_topic_bandwidth_stats` | Topic Size × Subscribers | bandwidth per topic, IQR outliers |
| `compute_qos_risk_stats` | QoS Risk Scatter | **(Optional)** risk score per topic, outliers *(Note: REST API omits this by default)* |
| `compute_app_balance_stats` | App Pub/Sub Balance | per-app I/O load, quadrant classification |
| `compute_topic_fanout_stats` | Topic Fanout | publisher × subscriber fanout, pattern counts |
| `compute_cross_node_heatmap_stats` | Cross-Node Heatmap | NxN communication matrix, intra/inter-node traffic |
| `compute_node_comm_load_stats` | Node Communication Load | per-node pub/sub totals, coefficient of variation |
| `compute_segment_comm_stats` | Segment Communication | segment-to-segment matrix, cross-segment pair counts |
| `compute_criticality_io_stats` | Criticality × I/O | critical vs. normal app I/O comparison |
| `compute_lib_dependency_stats` | Library Dependency | in/out degree for apps and libraries |
| `compute_node_critical_density_stats` | Node Critical Density| critical vs. normal app counts per node |
| `compute_segment_diversity_stats` | Segment Diversity | app count, topic count, and I/O per segment |

### Architectural Interpretations & Insights

The metrics computed by the API layer provide actionable insights into the system's architectural health, helping identify anti-patterns, bottlenecks, and deployment risks.

- **Topic Bandwidth**: Measures the network strain per topic by calculating the theoretical data volume pushed to all consumers per publish event. High bandwidth outliers indicate topics that could saturate network links, while zero-subscriber topics indicate wasted producer resources or incomplete system integration.
- **App Pub/Sub Balance**: Categorizes applications into quadrants based on their I/O activity. **High I/O hubs** are critical routing nodes whose failure disrupts both upstream and downstream flows. **Producer-only** apps are data sources whose failure cascades down, while **Consumer-only** apps represent system endpoints.
- **Topic Fanout**: Evaluates message multiplication factors. A high **Max Fanout** amplifies the blast radius of bad data (e.g., if a publisher sends malformed data, it simultaneously crashes many subscribers). It also highlights 1→N (broadcast) and N→1 (aggregation) communication patterns.
- **Cross-Node Heatmap & Node Load**: Analyzes traffic distribution across physical infrastructure. High inter-node traffic relative to intra-node traffic suggests poor deployment strategies (tightly coupled apps placed on different servers). High coefficient of variation in Node Load indicates severe resource imbalances.
- **Segment Communication**: Evaluates logical coupling between business domains/segments. High cross-segment traffic indicates tight coupling and bleeding of segment responsibilities, which violates Domain-Driven Design (DDD) principles.
- **Criticality × I/O Load**: Compares the I/O load of critical vs. non-critical applications. If critical applications handle significantly higher I/O loads, they represent severe bottlenecks and potential system vulnerabilities.
- **Library Dependency**: Identifies tightly coupled shared libraries. Libraries with high "In-Degree" are foundational components; changes to these libraries carry a high risk of regression across the system.
- **Node Critical Density**: Identifies infrastructure concentration risk. Nodes hosting a disproportionately high percentage of critical applications represent severe Single Points of Failure (SPOF) at the hardware level.
- **Segment Diversity**: Evaluates the logical composition of segments. Low diversity (few apps/topics) may indicate a fragmented architecture, while excessively high diversity might point to a monolithic segment that has absorbed too many responsibilities.

---

## 2. Core Structural Statistics (Analysis Layer)

Accessed via `saag.analysis.statistics_service.StatisticsService` which delegates to functions inside `saag/analysis/statistics.py`. These metrics operate on the live graph structure.

### Topological Metrics

| Metric | Function | Description |
|---|---|---|
| **Degree Distribution** | `get_degree_distribution` | Identifies Hubs (mean + 2σ) and isolated nodes. |
| **Connectivity Density** | `get_connectivity_density` | System coupling factor (Sparse < 0.05 to Very Dense > 0.30). |
| **Clustering Coefficient** | `get_clustering_coefficient`| Measures "triangle" formation (local interconnectedness). |
| **Dependency Depth** | `get_dependency_depth` | BFS hierarchy analysis to find deepest components and roots/leaves. |
| **Component Isolation** | `get_component_isolation`| Classifies components as Source, Sink, Bidirectional, or Isolated. |
| **Redundancy & SPOF** | `get_component_redundancy`| Identifies Single Points of Failure and bridge components. |

### Topological Interpretations & Insights

- **Degree Distribution**: Identifies structural hubs. Nodes with degree > mean + 2σ are structural bottlenecks.
- **Connectivity Density**: A measure of overall system coupling. Sparse graphs (< 0.05) are highly decoupled, while very dense graphs (> 0.30) indicate a "big ball of mud" architecture where everything is connected to everything.
- **Clustering Coefficient**: High clustering indicates modular, localized groups of components.
- **Component Isolation**: Identifies detached sub-graphs. Isolated components are often deprecated features or incomplete integrations.
- **Redundancy & SPOF**: Identifies Articulation Points (nodes whose removal splits the graph into disconnected components) and Bridges (critical edges between subsystems).

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
---

## 4. CLI Usage

The `cli/statistics_graph.py` script provides a command-line interface for both live and file-based analysis.

### Examples

| Usage | Command |
|---|---|
| **Live Analysis** | `PYTHONPATH=. python cli/statistics_graph.py` |
| **File Analysis** | `PYTHONPATH=. python cli/statistics_graph.py --input output/dataset.json` |
| **Filtered Charts** | `PYTHONPATH=. python cli/statistics_graph.py --chart topic_fanout qos_risk` |
| **JSON Export** | `PYTHONPATH=. python cli/statistics_graph.py --format json --output stats.json` |

### Key Flags
- `--chart`: Select specific chart IDs (e.g., `topic_bandwidth`, `qos_risk`).
- `--format`: Choose output format: `table` (rich), `minimal` (compact), or `json`.
- `--input`: Read from a pre-exported JSON file instead of connecting to Neo4j.
- `--output`: Save the resulting statistics as a JSON file.

---

## 5. Expected Input Format (Visualization)

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
      "system_hierarchy": {"css_name": "Segment-X"}
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

> [!NOTE]
> The `applications` objects must include the `role` and `system_hierarchy.css_name` keys to ensure that segment-based charts like segment communication and diversity are generated correctly and do not silently fall back to `NOT_FOUND`.

## Dependencies

- `numpy` — array operations, percentiles
- Python stdlib: `math`, `statistics`, `dataclasses`, `collections`
