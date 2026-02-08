# Step 6: Visualization

**Transform multi-layer analysis results into interactive decision-support dashboards that enable practitioners to identify, prioritize, and communicate critical component risks.**

---

## 6.1 Overview

Visualization is the final step of the six-step methodology. It synthesizes the outputs
of all preceding steps — structural metrics (Step 2), quality scores (Step 3), failure
impact scores (Step 4), and validation results (Step 5) — into a unified interactive
dashboard that supports architectural decision-making.

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  Step 2 Output      │     │                     │     │  Decision-Support   │
│  M(v) metric vectors│     │                     │     │  Dashboard          │
├─────────────────────┤     │                     │     │                     │
│  Step 3 Output      │     │   Visualization     │     │  - Executive KPIs   │
│  Q(v), R/M/A/V      │────▶│   Pipeline          │────▶│  - Topology Explorer│
│  classifications     │     │                     │     │  - Dependency Matrix│
├─────────────────────┤     │                     │     │  - Validation Report│
│  Step 4 Output      │     │                     │     │  - Layer Comparison │
│  I(v), cascades     │     │                     │     │  - Actionable Tables│
├─────────────────────┤     │                     │     │                     │
│  Step 5 Output      │     │                     │     │                     │
│  ρ, F1, Precision   │     │                     │     │                     │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

Unlike Steps 1–5, which compute quantitative results, Visualization focuses on
**information presentation**: translating numerical outputs into visual encodings
that expose patterns, anomalies, and actionable insights to different stakeholders —
from software architects performing root-cause triage to engineering managers evaluating
system-wide risk posture.

---

## 6.2 Formal Definition

### Definition 9: Visualization Input

Visualization consumes the complete output set from preceding steps:

```
Input:  For each layer l ∈ L_selected ⊆ { app, infra, mw, system }:

        S_l = (G_l, M_l, Q_l, F_l, V_l)

        where:
          G_l = G_analysis(l) = (V_l, E_l, w)              from Step 1
          M_l = { M(v) : v ∈ V_l }                          from Step 2
          Q_l = { (R(v), M(v), A(v), V(v), Q(v)) : v ∈ V_l } from Step 3
          F_l = { (I(v), cascade(v)) : v ∈ V_l }            from Step 4
          V_l = (ρ_l, F1_l, P_l, R_l, TopK_l, pass/fail)   from Step 5
```

### Definition 10: Dashboard Output

The visualization pipeline produces a self-contained HTML document:

```
Output: D = (sections, navigation, scripts)

        where:
          sections = { overview, layer_comparison, layer_detail_1, ..., layer_detail_n }
          Each section s ∈ sections contains:
            s.kpis     ⊆ KPI_CATALOGUE
            s.charts   ⊆ CHART_CATALOGUE
            s.tables   ⊆ TABLE_CATALOGUE
            s.network  ∈ { interactive_graph, dependency_matrix, ∅ }
```

### Definition 11: Visual Encoding Functions

Each visualization element maps analysis data to visual properties:

```
Color encoding:
  κ_type : T_V → Color        (component type → color)
  κ_crit : Level → Color      (criticality level → color)

Size encoding:
  σ : Q(v) → ℝ⁺              (quality score → node radius)

Position encoding:
  π : G_l → ℝ² per vertex     (graph layout algorithm)

Chart encoding:
  χ_dist  : {Level → count} → PieChart        (criticality distribution)
  χ_rank  : {v → I(v)} → BarChart             (impact ranking)
  χ_corr  : {(Q(v), I(v))} → ScatterPlot      (prediction correlation)
  χ_layer : {l → metrics} → GroupedBarChart    (cross-layer comparison)
  χ_dep   : E_l → AdjacencyMatrix             (dependency heatmap)
```

---

## 6.3 Design Principles

### 6.3.1 Progressive Disclosure

The dashboard follows a progressive disclosure hierarchy, from executive summary
to component-level detail:

```
Level 1: KPI Cards            → "Is our system healthy?"        (5 seconds)
Level 2: Distribution Charts  → "Where are the risks?"          (30 seconds)
Level 3: Ranking Tables       → "Which components need action?" (2 minutes)
Level 4: Network Explorer     → "How are risks connected?"      (5+ minutes)
Level 5: Dependency Matrix    → "What are the coupling patterns?"(5+ minutes)
```

This design respects Shneiderman's Visual Information-Seeking Mantra: _Overview first,
zoom and filter, then details-on-demand._

### 6.3.2 Multi-Audience Design

Different stakeholders extract different insights from the same dashboard:

| Audience | Primary Sections | Key Questions |
|----------|-----------------|---------------|
| Engineering Manager | KPIs, Distribution | Overall risk posture? Budget for remediation? |
| Software Architect | Network, Matrix, Tables | Which components to refactor? What are the coupling hotspots? |
| Reliability Engineer | Validation, Impact Ranking | Are predictions trustworthy? What fails most severely? |
| DevOps Engineer | Layer Comparison, SPOFs | Infrastructure vs. application risk? Deployment priorities? |

### 6.3.3 Analytical Traceability

Every visual element traces back to a specific methodology output. The dashboard
never presents derived or post-hoc metrics that were not computed in Steps 1–5.
This ensures reproducibility: any number shown in the dashboard can be verified
by re-running the corresponding pipeline step.

| Dashboard Element | Source Step | Source Data |
|-------------------|-----------|-------------|
| KPI: Total Nodes/Edges | Step 1 | \|V_l\|, \|E_l\| |
| KPI: Critical Count | Step 3 | \|{v : class(v) = CRITICAL}\| |
| KPI: SPOF Count | Step 2 | \|{v : AP(v) = 1}\| |
| Criticality Pie Chart | Step 3 | Classification distribution |
| Impact Bar Chart | Step 4 | Top-K by I(v) |
| Correlation Scatter | Steps 3+4 | Q(v) vs I(v) pairs |
| Validation Box | Step 5 | ρ, F1, Precision, Recall |
| Network Graph | Step 1 | G_analysis(l) topology |
| Dependency Matrix | Step 1+3 | E_l edges with Q(v) coloring |
| Component Table | Steps 2+3+4 | M(v), Q(v), I(v) per component |

### 6.3.4 Self-Contained Output

The dashboard is generated as a single HTML file with all CSS, JavaScript, and data
embedded inline. This ensures portability — the file can be shared via email, stored
alongside analysis artifacts, or archived with no external dependencies.

---

## 6.4 Visualization Taxonomy

The dashboard employs six distinct visualization types, each chosen for its perceptual
effectiveness at communicating a specific class of analytical insight.

### 6.4.1 KPI Cards — Categorical Assessment

**Purpose**: Provide instant system health assessment via aggregate counts.

**Encoding**: Number + label + optional highlight color.

```
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│   48    │ │   127   │ │    5    │ │    3    │ │    2    │
│  Nodes  │ │  Edges  │ │Critical │ │  SPOFs  │ │Anti-Pat.│
└─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

**Design rationale**: KPIs exploit preattentive numerical processing. A practitioner
can assess system-level risk in under 5 seconds. Critical and SPOF counts are
highlighted in red/orange when they exceed configurable thresholds.

### 6.4.2 Distribution Charts (Pie) — Part-to-Whole Relationships

**Purpose**: Show how components distribute across criticality levels and types.

**Encoding**: Arc length proportional to count, color-coded by level.

| Chart | Data Source | Insight |
|-------|-----------|---------|
| Criticality Distribution | Step 3 classifications | Risk concentration |
| Component Types | Step 1 vertex types | Architectural composition |

**Design rationale**: Pie charts are appropriate here because the categories are
mutually exclusive (each component has exactly one criticality level and one type)
and the total (100%) is meaningful — practitioners want to know _what fraction_
of their system is critical.

### 6.4.3 Ranking Charts (Bar) — Ordered Comparison

**Purpose**: Identify the top-N most impactful or most critical components.

**Encoding**: Bar length proportional to score, color-coded by criticality level.

| Chart | Data Source | Insight |
|-------|-----------|---------|
| Impact Ranking | Step 4: I(v) | Which components cause the most damage when failing |
| Quality Ranking | Step 3: Q(v) | Which components are predicted most critical |
| RMAV Breakdown | Step 3: R, M, A, V | Which quality dimension drives criticality for each component |

**Design rationale**: Bar charts with sorted ordering support the primary practitioner
task: _prioritization_. The ranked list directly answers "what should I fix first?"

### 6.4.4 Correlation Charts (Scatter) — Prediction Validity

**Purpose**: Visually confirm that predicted Q(v) aligns with simulated I(v).

**Encoding**: Each point represents one component at coordinates (Q(v), I(v)).
Points near the diagonal indicate good prediction; outliers warrant investigation.

```
I(v) │            ·
     │          · ·
     │        · ·
     │      ·  ·
     │    · ·
     │  · ·
     │ ·
     └─────────────── Q(v)
```

**Design rationale**: Scatter plots expose the core claim of the methodology —
that topology predicts impact. The Spearman ρ statistic summarizes the correlation
numerically, but the scatter plot reveals _where_ predictions fail (e.g., components
with high Q but low I suggest over-prediction; the reverse suggests hidden criticality).

### 6.4.5 Network Graph (Interactive) — Topological Exploration

**Purpose**: Enable spatial exploration of the dependency graph, revealing clusters,
bottlenecks, and structural patterns not visible in tabular data.

**Encoding**:
- Node color → component type (κ_type) or criticality level (κ_crit)
- Node size → Q(v) score (σ)
- Edge direction → dependency direction
- Edge thickness → dependency weight w(e)
- Spatial layout → force-directed or hierarchical

**Interactions**:

| Action | Effect |
|--------|--------|
| Hover | Show component details (type, Q(v), I(v), RMAV scores) |
| Click | Highlight direct dependencies (1-hop neighborhood) |
| Drag | Reposition nodes for manual layout |
| Scroll | Zoom in/out |
| Double-click | Focus and center on the selected node |

**Design rationale**: Force-directed graph layouts naturally position highly connected
nodes centrally and push peripheral nodes outward, creating a visual analog of
centrality. Critical components — which tend to have high betweenness and degree —
are visually prominent without explicit emphasis.

### 6.4.6 Dependency Matrix (Heatmap) — Coupling Analysis

**Purpose**: Expose dense coupling regions, dependency clusters, and asymmetric
relationships that are difficult to discern in a node-link diagram at scale.

**Encoding**: Rows and columns represent components; cell color intensity represents
dependency weight. Components are sorted by criticality score to cluster high-risk
regions in the top-left corner.

**Interactions**:

| Action | Effect |
|--------|--------|
| Hover | Show source, target, weight, and dependency type |
| Sort buttons | Reorder by criticality, alphabetical, or cluster |
| Export | Download matrix as PNG for reports |

**Design rationale**: Adjacency matrices scale better than node-link diagrams for
dense graphs (|E| > 3|V|). They make coupling patterns immediately visible as visual
blocks, and are particularly effective for identifying:
- **Dense clusters**: Square blocks along the diagonal
- **Hub components**: Full rows or columns
- **Asymmetric dependencies**: Off-diagonal patterns

---

## 6.5 Dashboard Structure

### Section 1: Overview

High-level system summary aggregated across all selected layers.

```
┌─────────────────────────────────────────────────────────────┐
│  📊 OVERVIEW                                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐│
│  │   48    │ │   127   │ │    5    │ │    3    │ │   2   ││
│  │  Nodes  │ │  Edges  │ │Critical │ │  SPOFs  │ │A-Pats ││
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───────┘│
│                                                             │
│  [Criticality Distribution]    [Component Types]           │
│       (Pie Chart)                  (Pie Chart)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Section 2: Layer Comparison

Side-by-side comparison of key metrics across layers, enabling practitioners to
identify which architectural layer carries the most risk.

```
┌─────────────────────────────────────────────────────────────┐
│  📊 LAYER COMPARISON                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Grouped Bar Chart: Spearman ρ by layer]                  │
│  [Grouped Bar Chart: Critical count by layer]              │
│  [Grouped Bar Chart: Mean Q(v) by layer]                   │
│                                                             │
│  Cross-Layer Summary Table:                                │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Layer │ Nodes │ Critical │ ρ     │ F1    │ Top-5     │ │
│  ├───────────────────────────────────────────────────────┤ │
│  │ App   │  22   │    3     │ 0.912 │ 0.943 │ 80%      │ │
│  │ Infra │  18   │    2     │ 0.789 │ 0.867 │ 60%      │ │
│  │ Sys   │  48   │    5     │ 0.876 │ 0.923 │ 80%      │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Section 3: Layer Detail (per layer)

Deep dive into each selected layer with full analysis results.

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
│  ┌────────────────────────────────────────────────────────┐│
│  │ Top Components by Q(v)                                 ││
│  ├────────────────────────────────────────────────────────┤│
│  │ Component      Type        R     M     A     V     Q  ││
│  │ sensor_fusion  Application 0.82  0.88  0.90  0.75 0.84││
│  │ main_broker    Broker      0.78  0.65  0.95  0.80 0.80││
│  │ planning_node  Application 0.71  0.73  0.45  0.68 0.64││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌────────────────────────────────────────────────────────┐│
│  │ Validation: PASSED                                     ││
│  │ Spearman ρ: 0.876 ✓   F1-Score:  0.923 ✓             ││
│  │ Precision:  0.912 ✓   Recall:    0.857 ✓             ││
│  │ Top-5 Overlap: 80% ✓  Top-10 Overlap: 70% ✓         ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌────────────────────────────────────────────────────────┐│
│  │           [Interactive Network Graph]                  ││
│  │                                                        ││
│  │  ○ Application  ○ Broker  ○ Node  ○ Topic            ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌────────────────────────────────────────────────────────┐│
│  │           [Dependency Matrix Heatmap]                  ││
│  │                                                        ││
│  │  [Sort: Criticality] [Sort: Name] [Sort: Cluster]    ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6.6 Visual Encoding Specification

### Color Coding

**By Component Type (κ_type):**

| Type | Color | Hex | Rationale |
|------|-------|-----|-----------|
| Application | Blue | `#4A90D9` | Primary analysis targets |
| Broker | Purple | `#9B59B6` | Middleware intermediaries |
| Node | Green | `#27AE60` | Infrastructure/physical |
| Topic | Yellow/Amber | `#F39C12` | Communication channels |

**By Criticality Level (κ_crit):**

| Level | Color | Hex | Q(v) Range |
|-------|-------|-----|------------|
| CRITICAL | Red | `#E74C3C` | Upper outliers (box-plot) |
| HIGH | Orange | `#E67E22` | Q3 – upper fence |
| MEDIUM | Yellow | `#F1C40F` | Q1 – Q3 |
| LOW | Green | `#2ECC71` | Lower fence – Q1 |
| MINIMAL | Gray | `#95A5A6` | Lower outliers |

**Design note**: The criticality color palette follows a traffic-light metaphor
(red → green) that aligns with universal risk communication conventions. Color
choices are WCAG 2.1 AA compliant for contrast against white backgrounds.

### Node Sizing

Node radius in the network graph scales linearly with Q(v):

```
radius(v) = r_min + (r_max - r_min) × Q_norm(v)

where:
  r_min = 8px   (minimum visible size)
  r_max = 32px  (maximum size cap)
  Q_norm(v) = (Q(v) - Q_min) / (Q_max - Q_min)
```

This ensures even minimal-criticality nodes remain visible while
critical nodes are visually prominent.

---

## 6.7 Decision Support Workflows

The dashboard is designed to support specific practitioner workflows:

### Workflow 1: Risk Triage

_"Which components should we address first?"_

1. **Start** at Overview KPIs → check Critical count and SPOF count
2. **Navigate** to Layer Detail → examine ranked component table sorted by Q(v)
3. **Cross-reference** Impact Ranking bar chart → verify that high-Q(v) components
   also have high I(v) (check validation scatter for outliers)
4. **Explore** Network Graph → click on critical components to understand their
   dependency neighborhood

### Workflow 2: Validation Review

_"Can we trust these predictions?"_

1. **Start** at Validation Box → check ρ, F1, pass/fail status
2. **If PASSED**: Use predicted rankings confidently for prioritization
3. **If FAILED**: Examine scatter plot for systematic bias:
   - Points above the diagonal → methodology under-predicts (hidden criticality)
   - Points below the diagonal → methodology over-predicts (false alarms)
4. **Compare** across layers → application layer typically validates stronger

### Workflow 3: Architecture Assessment

_"What are the structural weaknesses in our system?"_

1. **Start** at Layer Comparison → identify which layer has highest risk
2. **Navigate** to Dependency Matrix → look for dense coupling blocks
3. **Examine** RMAV breakdown in component tables → determine which quality
   dimension (R, M, A, V) drives criticality for top components
4. **Use** Network Graph → trace dependency chains from critical components
   to understand failure propagation paths

---

## 6.8 Commands

### Generate Dashboard

```bash
# Single layer
python bin/visualize_graph.py --layer system --output dashboard.html

# Multiple layers
python bin/visualize_graph.py --layers app,infra,system --output dashboard.html

# All layers
python bin/visualize_graph.py --all --output dashboard.html

# Open in browser automatically
python bin/visualize_graph.py --layer system --output dashboard.html --open
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--layers` | Comma-separated layers (app, infra, mw, system) | — |
| `--layer` | Single layer shorthand | — |
| `--all` | Include all layers | false |
| `--output` / `-o` | Output HTML file path | `dashboard.html` |
| `--no-network` | Exclude interactive network graph | false |
| `--no-matrix` | Exclude dependency matrix | false |
| `--no-validation` | Exclude validation metrics | false |
| `--open` | Open in browser after generation | false |
| `--uri` | Neo4j connection URI | `bolt://localhost:7687` |
| `--user` / `-u` | Neo4j username | `neo4j` |
| `--password` / `-p` | Neo4j password | `password` |

### Demo Mode

Generate a dashboard with sample data (no Neo4j required):

```bash
python bin/visualize_graph.py --demo --output demo_dashboard.html
```

---

## 6.9 Programmatic Usage

```python
from src.visualization import GraphVisualizer

# Context manager handles connection lifecycle
with GraphVisualizer(uri="bolt://localhost:7687") as viz:
    viz.generate_dashboard(
        output_file="dashboard.html",
        layers=["app", "infra", "system"],
        include_network=True,
        include_matrix=True,
        include_validation=True
    )
```

### Using the Service Layer (Hexagonal Architecture)

```python
from src.application.container import Container

container = Container(uri="bolt://localhost:7687", user="neo4j", password="password")

viz_service = container.visualization_service()
path = viz_service.generate_dashboard(
    output_file="dashboard.html",
    layers=["app", "infra", "system"],
    include_network=True,
    include_matrix=True,
    include_validation=True
)

container.close()
```

---

## 6.10 Output Artifacts

### Primary Output

| File | Format | Content |
|------|--------|---------|
| `dashboard.html` | Self-contained HTML | Complete interactive dashboard with embedded CSS, JS, and data |

### Optional Static Exports

When using `--visualize` flags with analysis or validation scripts:

| File | Format | Purpose |
|------|--------|---------|
| `scatter_plot.png` | PNG | Q(v) vs I(v) correlation (publication-quality) |
| `confusion_matrix.png` | PNG | Classification confusion matrix |
| `ranking_comparison.png` | PNG | Side-by-side Q(v) vs I(v) rankings |

### Dashboard Properties

| Property | Description |
|----------|-------------|
| **Responsive** | Adapts to desktop, tablet, and mobile viewports |
| **Self-contained** | Single HTML file with zero external dependencies |
| **Print-friendly** | Clean layout with hidden navigation for printing |
| **Navigable** | Sidebar with section links and collapsible sections |
| **Exportable** | Matrix and chart views support PNG export |
| **Interactive** | Network graph and dependency matrix support hover, click, drag, zoom |

---

## 6.11 Scalability Considerations

Dashboard generation and rendering performance varies with system size:

| Scale | Components | Dashboard Size | Generation Time | Render Note |
|-------|-----------|----------------|-----------------|-------------|
| Small | < 50 | ~500 KB | < 5s | All visualizations smooth |
| Medium | 50–200 | ~2 MB | 5–15s | Network graph may need stabilization time |
| Large | 200–1000 | ~10 MB | 15–60s | Consider `--no-network` for faster load |
| XLarge | > 1000 | ~50 MB+ | 60s+ | Use dependency matrix only; network graph impractical |

**Recommendation for large systems**: Use the dependency matrix (6.4.6) rather than
the interactive network graph (6.4.5) for coupling analysis. Adjacency matrices scale
O(|V|²) in visual elements while maintaining readability, whereas node-link diagrams
degrade rapidly beyond ~200 nodes due to visual clutter and layout instability.

---

## 6.12 Example Output

```
═══════════════════════════════════════════════════════════════
  VISUALIZATION
═══════════════════════════════════════════════════════════════

  [1/4] Initializing visualization pipeline...
        ✓ Analysis module connected
        ✓ Simulation module connected
        ✓ Validation module connected

  [2/4] Processing 📱 Application Layer...
        ✓ 22 components analyzed
        ✓ Network graph: 22 nodes, 47 edges
        ✓ Dependency matrix: 22×22

  [3/4] Processing 🖥️ Infrastructure Layer...
        ✓ 18 components analyzed
        ✓ Network graph: 18 nodes, 31 edges
        ✓ Dependency matrix: 18×18

  [4/4] Processing 🌐 Complete System...
        ✓ 48 components analyzed
        ✓ Network graph: 48 nodes, 127 edges
        ✓ Dependency matrix: 48×48

  Generating HTML dashboard...

  ✓ Dashboard generated: dashboard.html (2.3 MB)
  ✓ Contains: 3 layers, 6 charts, 3 network graphs, 3 matrices
```

---

## 6.13 Limitations and Future Work

### Current Limitations

1. **Static snapshot**: The dashboard represents a single point-in-time analysis.
   It does not support temporal comparison (e.g., "how has criticality changed
   since last sprint?"). Future temporal analysis (see §6.13 Future Work) will
   address this gap.

2. **Browser rendering limits**: Interactive network graphs become impractical
   above ~500 nodes due to SVG/Canvas rendering performance. The dependency
   matrix partially mitigates this (§6.11).

3. **Single-user design**: The dashboard is a static HTML file with no
   collaborative annotation, commenting, or shared state capabilities.

### Future Work

1. **Temporal Evolution Dashboard**: Side-by-side comparison of analysis results
   across multiple time points, enabling architectural drift detection and
   trend analysis of criticality scores.

2. **GNN Prediction Overlay**: When Graph Neural Network prediction becomes
   available, the dashboard will display GNN-predicted criticality alongside
   topology-predicted Q(v) for comparison.

3. **Digital Twin Integration**: Real-time dashboard updates fed by continuous
   calibration against runtime telemetry, bridging the gap between pre-deployment
   prediction and operational reality.

4. **Cascade Visualization**: Interactive cascade tree exploration showing
   failure propagation paths from `FailureResult.cascade_to_graph()` output,
   enabling visual root-cause analysis.

5. **What-If Scenario Comparison**: Dashboard mode showing analysis results
   before and after proposed architectural changes (e.g., adding a redundant
   broker), enabling quantitative evaluation of refactoring proposals.

---

## Summary

The visualization dashboard completes the six-step methodology by transforming
quantitative analysis outputs into actionable, interactive decision-support tools:

1. **Executive KPIs** — system health at a glance
2. **Layer Comparison** — cross-layer risk assessment
3. **Detailed Tables** — component-level prioritization with RMAV breakdown
4. **Interactive Network** — topological exploration and dependency tracing
5. **Dependency Matrix** — coupling analysis and cluster identification
6. **Validation Report** — prediction trustworthiness with pass/fail status

Together, these elements enable practitioners to move from analysis results
to informed architectural decisions — the ultimate purpose of the Software-as-a-Graph
methodology.

---

## Navigation

← [Step 5: Validation](validation.md) | [README](../README.md)