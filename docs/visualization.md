# Visualization

This document explains the visualization capabilities for exploring and communicating analysis results.

---

## Table of Contents

1. [Overview](#overview)
2. [Visualization Types](#visualization-types)
3. [Network Graph](#network-graph)
4. [Multi-Layer View](#multi-layer-view)
5. [Dashboard](#dashboard)
6. [Implementation](#implementation)

---

## Overview

Visualization transforms analysis results into actionable insights through interactive web-based interfaces.

### Purpose

| Goal | Visualization |
|------|---------------|
| **Explore** topology | Interactive network graph |
| **Understand** architecture | Multi-layer view |
| **Analyze** results | Comprehensive dashboard |
| **Communicate** findings | Exportable HTML reports |

### Technology Stack

| Component | Technology |
|-----------|------------|
| Graph Rendering | vis.js Network |
| Charts | Chart.js |
| Layout | HTML5 + CSS3 |
| Interactivity | JavaScript |
| Export | Standalone HTML |

---

## Visualization Types

### Quick Reference

| Type | Best For | Key Features |
|------|----------|--------------|
| **Network Graph** | Topology exploration | Pan, zoom, select, filter |
| **Multi-Layer** | Architecture understanding | Layer separation, dependency lines |
| **Dashboard** | Comprehensive analysis | Metrics, charts, tables combined |

---

## Network Graph

Interactive node-link diagram for exploring system topology.

### Features

- **Pan and Zoom**: Navigate large graphs
- **Node Selection**: Click for details
- **Filtering**: Show/hide by type or level
- **Layout**: Physics-based or hierarchical
- **Color Coding**: By criticality level

### Color Scheme

| Level | Color | Hex |
|-------|-------|-----|
| CRITICAL | Red | #FF4444 |
| HIGH | Orange | #FFA500 |
| MEDIUM | Yellow | #FFD700 |
| LOW | Light Green | #90EE90 |
| MINIMAL | Gray | #D3D3D3 |

### Node Shapes

| Component Type | Shape |
|----------------|-------|
| Application | Circle |
| Topic | Square |
| Broker | Diamond |
| Node | Triangle |

### Usage

```python
from src.visualization import GraphRenderer

renderer = GraphRenderer()

# Basic network view
html = renderer.render(graph, criticality_scores)
Path("network.html").write_text(html)

# With options
html = renderer.render(
    graph,
    criticality_scores,
    layout="hierarchical",      # or "physics"
    show_labels=True,
    edge_arrows=True,
    physics_enabled=False       # Disable physics after layout
)
```

### Interactive Controls

```
┌─────────────────────────────────────────────────────────────────────┐
│ [🔍 Zoom+] [🔍 Zoom-] [📍 Fit] [🔄 Reset]                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Filter by Type:  [✓] Applications  [✓] Topics  [✓] Brokers       │
│                   [✓] Nodes                                        │
│                                                                     │
│  Filter by Level: [✓] Critical  [✓] High  [✓] Medium              │
│                   [✓] Low  [✓] Minimal                             │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│              ┌─────────────────────────────────┐                   │
│              │                                 │                   │
│              │     [Interactive Graph]         │                   │
│              │                                 │                   │
│              └─────────────────────────────────┘                   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Selected: B1 (Broker) | Score: 0.82 | Level: CRITICAL              │
│ Connections: 12 in, 8 out | Articulation Point: Yes                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Layer View

Vertical layer separation showing architectural hierarchy.

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MULTI-LAYER VIEW                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INFRASTRUCTURE LAYER                                              │
│  ┌──────┐         ┌──────┐         ┌──────┐                       │
│  │  N1  │─────────│  N2  │─────────│  N3  │                       │
│  └──┬───┘         └──┬───┘         └──┬───┘                       │
│     │                │                │                            │
│ ════│════════════════│════════════════│════════════════════════════│
│     │                │                │                            │
│  BROKER LAYER                                                      │
│  ┌──▼───┐         ┌──▼───┐                                        │
│  │  B1  │─────────│  B2  │                                        │
│  │ CRIT │         │ HIGH │                                        │
│  └──┬───┘         └──┬───┘                                        │
│    /│\              /│\                                            │
│ ══/═│═\════════════/═│═\═══════════════════════════════════════════│
│  /  │  \          /  │  \                                          │
│  TOPIC LAYER                                                       │
│ ┌▼┐ ┌▼┐ ┌▼┐    ┌▼┐ ┌▼┐ ┌▼┐                                       │
│ │T1│ │T2│ │T3│  │T4│ │T5│ │T6│                                    │
│ └┬┘ └┬┘ └┬┘    └┬┘ └┬┘ └┬┘                                       │
│  │   │   │      │   │   │                                          │
│ ═│═══│═══│══════│═══│═══│══════════════════════════════════════════│
│  │   │   │      │   │   │                                          │
│  APPLICATION LAYER                                                 │
│ ┌▼┐ ┌▼┐ ┌▼┐    ┌▼┐ ┌▼┐ ┌▼┐                                       │
│ │A1│ │A2│ │A3│  │A4│ │A5│ │A6│                                    │
│ └──┘ └──┘ └──┘  └──┘ └──┘ └──┘                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Features

- **Layer Separation**: Clear visual hierarchy
- **Dependency Lines**: Cross-layer relationships
- **Criticality Colors**: Nodes colored by level
- **Hover Details**: Component information on hover

### Usage

```python
html = renderer.render_multi_layer(
    graph,
    criticality_scores,
    layer_spacing=150,      # Pixels between layers
    show_dependencies=True,
    show_labels=True
)
```

---

## Dashboard

Comprehensive analysis report combining all visualizations and metrics.

### Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  GRAPH-BASED CRITICALITY ANALYSIS DASHBOARD                                 │
│  Generated: 2025-12-28 14:30:00 | Components: 77 | Validation: PASSED      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐│
│  │   Components    │ │    Spearman     │ │    F1-Score     │ │   Status    ││
│  │      77         │ │     0.808       │ │     0.875       │ │   PASSED    ││
│  │                 │ │     ✓ MET       │ │     ⚠ CLOSE     │ │             ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────┘│
│                                                                             │
├──────────────────────────────────────┬──────────────────────────────────────┤
│  CRITICALITY DISTRIBUTION            │  PREDICTED VS ACTUAL                 │
│  ┌──────────────────────────────┐   │  ┌──────────────────────────────┐   │
│  │ ████████████ CRITICAL: 3    │   │  │     ×                        │   │
│  │ ██████████████████ HIGH: 8  │   │  │   ×   ×    ×                 │   │
│  │ ████████████████ MEDIUM: 12 │   │  │  ×  ×  ×    ×  ×             │   │
│  │ ██████████████████ LOW: 25  │   │  │ × ×  × ×  ×   ×              │   │
│  │ ███████████████ MINIMAL: 29 │   │  │  ×   × ×                     │   │
│  └──────────────────────────────┘   │  └──────────────────────────────┘   │
│                                      │  ρ = 0.808, p < 0.001              │
├──────────────────────────────────────┴──────────────────────────────────────┤
│  TOP CRITICAL COMPONENTS                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Rank │ Component │ Type   │ Score  │ Level    │ Impact │ AP │ BC    │  │
│  ├──────┼───────────┼────────┼────────┼──────────┼────────┼────┼───────┤  │
│  │  1   │ B1        │ Broker │ 0.823  │ CRITICAL │ 0.781  │ ✓  │ 0.452 │  │
│  │  2   │ B2        │ Broker │ 0.756  │ CRITICAL │ 0.712  │ ✓  │ 0.398 │  │
│  │  3   │ N2        │ Node   │ 0.698  │ CRITICAL │ 0.654  │ ✓  │ 0.356 │  │
│  │  4   │ A12       │ App    │ 0.612  │ HIGH     │ 0.589  │    │ 0.312 │  │
│  │  5   │ T5        │ Topic  │ 0.598  │ HIGH     │ 0.567  │    │ 0.289 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  NETWORK VISUALIZATION                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │                    [Interactive Graph]                              │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Sections

| Section | Content |
|---------|---------|
| **Header** | Title, timestamp, summary stats |
| **Metric Cards** | Key metrics with status indicators |
| **Distribution Chart** | Bar chart of criticality levels |
| **Scatter Plot** | Predicted vs actual scores |
| **Component Table** | Sortable, searchable details |
| **Network Graph** | Interactive visualization |

### Usage

```python
from src.visualization import DashboardGenerator

generator = DashboardGenerator()

html = generator.generate(
    graph,
    criticality=criticality_scores,
    validation=validation_result.to_dict(),
    simulation=batch_result.to_dict(),
    title="System Analysis Dashboard"
)

Path("dashboard.html").write_text(html)
```

### Customization

```python
html = generator.generate(
    graph,
    criticality=scores,
    validation=validation,
    
    # Appearance
    title="Custom Dashboard",
    theme="dark",                    # "light" or "dark"
    
    # Sections
    show_network=True,
    show_table=True,
    show_charts=True,
    
    # Table options
    table_page_size=20,
    sortable_columns=True,
    searchable=True
)
```

---

## Implementation

### GraphRenderer Class

```python
from src.visualization import GraphRenderer

renderer = GraphRenderer()

# Network visualization
html = renderer.render(graph, criticality)

# Multi-layer view
html = renderer.render_multi_layer(graph, criticality)

# With all options
html = renderer.render(
    graph,
    criticality,
    layout="hierarchical",
    show_labels=True,
    edge_arrows=True,
    physics_enabled=False,
    width="100%",
    height="600px"
)
```

### DashboardGenerator Class

```python
from src.visualization import DashboardGenerator

generator = DashboardGenerator()

# Full dashboard
html = generator.generate(
    graph,
    criticality=scores,
    validation=validation.to_dict(),
    simulation=simulation.to_dict()
)

# Minimal dashboard
html = generator.generate(
    graph,
    criticality=scores
)
```

### Criticality Data Format

```python
criticality = {
    "B1": {"score": 0.82, "level": "CRITICAL"},
    "B2": {"score": 0.75, "level": "HIGH"},
    "A1": {"score": 0.45, "level": "MEDIUM"},
    ...
}
```

Or simplified:
```python
criticality = {
    "B1": 0.82,
    "B2": 0.75,
    "A1": 0.45,
    ...
}
```

### CLI Usage

```bash
# Basic visualization
python visualize_graph.py --input graph.json --output network.html

# Dashboard
python visualize_graph.py \
    --input graph.json \
    --dashboard \
    --run-analysis \
    --output dashboard.html

# Multi-layer view
python visualize_graph.py \
    --input graph.json \
    --multi-layer \
    --output layers.html
```

### Export Options

| Format | Usage |
|--------|-------|
| **HTML** | Standalone web page |
| **PNG** | Static image (via browser) |
| **PDF** | Print-ready (via browser) |

```python
# HTML is default
html = generator.generate(graph, criticality)
Path("report.html").write_text(html)

# For PNG/PDF: open in browser and use print/screenshot
```

---

## Navigation

- **Previous:** [← Statistical Validation](validation.md)
- **Next:** [API Reference →](api-reference.md)
