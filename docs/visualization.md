# Step 6: Visualization

**Generate interactive dashboards to communicate analysis results**

---

## Overview

Visualization creates comprehensive HTML dashboards that combine analysis results, simulation outcomes, and validation metrics into an interactive presentation.

```
┌─────────────────────┐          ┌─────────────────────┐
│  Analysis Results   │          │  HTML Dashboard     │
│  Simulation Results │    →     │                     │
│  Validation Metrics │          │  - KPIs             │
│                     │          │  - Charts           │
│                     │          │  - Network Graph    │
│                     │          │  - Tables           │
└─────────────────────┘          └─────────────────────┘
```

---

## Dashboard Components

| Component | Purpose |
|-----------|---------|
| **KPI Cards** | High-level metrics at a glance |
| **Pie Charts** | Distribution visualizations |
| **Bar Charts** | Comparisons and rankings |
| **Network Graph** | Interactive topology (vis.js) |
| **Data Tables** | Detailed component information |
| **Validation Box** | Pass/fail status with metrics |

---

## Commands

### Generate Dashboard

```bash
# Single layer
python bin/visualize_graph.py --layer system --output dashboard.html

# Multiple layers
python bin/visualize_graph.py --layers app,infra,system --output dashboard.html

# Open in browser automatically
python bin/visualize_graph.py --layer system --output dashboard.html --open
```

### Options

| Option | Description |
|--------|-------------|
| `--layers` | Layers to include (app, infra, mw, system) |
| `--all` | Include all layers |
| `--output` | Output HTML file path |
| `--no-network` | Exclude interactive network graph |
| `--no-validation` | Exclude validation metrics |
| `--open` | Open in browser after generation |

---

## Dashboard Sections

### 1. Overview

High-level system summary:

```
┌─────────────────────────────────────────────────────────────┐
│  📊 OVERVIEW                                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐ │
│  │   48    │ │   127   │ │    5    │ │    3    │ │   2   │ │
│  │  Nodes  │ │  Edges  │ │Critical │ │  SPOFs  │ │Problems│ │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───────┘ │
│                                                             │
│  [Criticality Distribution]    [Component Types]           │
│       (Pie Chart)                  (Pie Chart)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. Layer Comparison

Compare metrics across analysis layers:

```
┌─────────────────────────────────────────────────────────────┐
│  📈 LAYER COMPARISON                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer         Nodes  Edges  Density  Critical  SPOFs      │
│  ─────────────────────────────────────────────────────────  │
│  Application     25     42    0.070      3        2        │
│  Infrastructure   8     15    0.268      1        1        │
│  System          48    127    0.056      5        3        │
│                                                             │
│  [Criticality by Layer]    [Validation by Layer]           │
│       (Grouped Bar)            (Grouped Bar)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. Layer Details

Per-layer deep dive:

```
┌─────────────────────────────────────────────────────────────┐
│  🌐 SYSTEM LAYER                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Graph Statistics          Criticality Summary             │
│  ─────────────────         ─────────────────               │
│  Nodes: 48                 CRITICAL: 5                     │
│  Edges: 127                HIGH: 8                         │
│  Density: 0.056            MEDIUM: 15                      │
│  Connected: Yes            LOW: 12                         │
│                            MINIMAL: 8                      │
│                                                             │
│  Top Components by Q(v):                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Component      Type        Q(v)   Level            │    │
│  ├────────────────────────────────────────────────────┤    │
│  │ sensor_fusion  Application 0.892  CRITICAL         │    │
│  │ main_broker    Broker      0.856  CRITICAL         │    │
│  │ planning_node  Application 0.789  HIGH             │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  Validation Metrics:                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Spearman ρ:  0.876  ✓    F1-Score:  0.923  ✓      │    │
│  │ Precision:   0.912  ✓    Recall:    0.857  ✓      │    │
│  │ Status: PASSED                                     │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  Interactive Network:                                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │                                                    │    │
│  │              [vis.js Network Graph]               │    │
│  │                                                    │    │
│  │   ○ Application  ○ Broker  ○ Node  ○ Topic       │    │
│  │                                                    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Network Visualization

The interactive network uses **vis.js** for pan, zoom, and hover interactions.

### Color Coding

**By Component Type:**
| Type | Color |
|------|-------|
| Application | Blue |
| Broker | Purple |
| Node | Green |
| Topic | Yellow |

**By Criticality Level:**
| Level | Color |
|-------|-------|
| CRITICAL | Red |
| HIGH | Orange |
| MEDIUM | Yellow |
| LOW | Green |
| MINIMAL | Gray |

### Interactions

- **Hover**: Show component details
- **Click**: Highlight connections
- **Drag**: Reposition nodes
- **Scroll**: Zoom in/out
- **Double-click**: Focus on node

---

## Charts

### Criticality Distribution (Pie)

Shows breakdown of components by criticality level.

### Component Types (Pie)

Shows distribution of Applications, Brokers, Nodes, Topics.

### Impact Ranking (Bar)

Top components ranked by simulation impact I(v).

### Validation Comparison (Scatter)

Q(v) vs I(v) scatter plot showing correlation.

### Layer Comparison (Grouped Bar)

Side-by-side comparison of metrics across layers.

---

## Output Files

```bash
python visualize_graph.py --layer system --output dashboard.html
```

Generates:
- `dashboard.html` — Self-contained HTML file with embedded CSS/JS

Optional with `--visualize`:
- `scatter_plot.png` — Correlation scatter plot
- `confusion_matrix.png` — Classification confusion matrix
- `ranking_comparison.png` — Side-by-side rankings

---

## Demo Mode

Generate a demo dashboard without Neo4j:

```bash
python visualize_graph.py --demo --output demo_dashboard.html
```

Uses sample data to demonstrate dashboard features.

---

## Programmatic Usage

```python
from src.visualization import GraphVisualizer

with GraphVisualizer(uri="bolt://localhost:7687") as viz:
    viz.generate_dashboard(
        output_file="dashboard.html",
        layers=["app", "infra", "system"],
        include_network=True,
        include_validation=True
    )
```

---

## Dashboard Features

| Feature | Description |
|---------|-------------|
| **Responsive** | Works on desktop, tablet, mobile |
| **Self-contained** | Single HTML file, no external dependencies |
| **Print-friendly** | Clean print layout |
| **Navigation** | Sidebar with section links |
| **Collapsible** | Expandable/collapsible sections |
| **Interactive** | vis.js network graph |

---

## Example Output

```
═══════════════════════════════════════════════════════════════
  VISUALIZATION
═══════════════════════════════════════════════════════════════

  [1/4] Initializing visualization pipeline...
        ✓ Analysis module connected
        ✓ Simulation module connected
        ✓ Validation module connected
        
  [2/4] Processing 📱 Application Layer...
  [3/4] Processing 🖥️ Infrastructure Layer...
  [4/4] Processing 🌐 Complete System...
  
  Generating HTML dashboard...
  
  ✓ Dashboard generated: dashboard.html
```

---

## Summary

The visualization dashboard provides:

1. **Executive Summary**: KPIs and distributions at a glance
2. **Layer Comparison**: Cross-layer analysis
3. **Detailed Tables**: Component-level data
4. **Interactive Network**: Topology exploration
5. **Validation Status**: Pass/fail with metrics

This completes the six-step Software-as-a-Graph methodology.

---

## Navigation

← [Step 5: Validation](validation.md) | [README](../README.md)
