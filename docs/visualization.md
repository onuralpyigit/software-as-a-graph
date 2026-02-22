# Step 6: Visualization

**Translate quantitative analysis into interactive dashboards for architectural decision-making.**

← [Step 5: Validation](validation.md) | [README](../README.md)

---

## Table of Contents

1. [What This Step Does](#what-this-step-does)
2. [Two Visualization Surfaces](#two-visualization-surfaces)
3. [Genieus: Live Web Application](#genieus-live-web-application)
   - [Dashboard Tab](#dashboard-tab)
   - [Graph Explorer Tab](#graph-explorer-tab)
   - [Analysis Tab](#analysis-tab)
   - [Simulation Tab](#simulation-tab)
   - [Settings Tab](#settings-tab)
4. [Static HTML Dashboard](#static-html-dashboard)
   - [Section 1 — Executive Overview](#section-1--executive-overview)
   - [Section 2 — Layer Comparison](#section-2--layer-comparison)
   - [Section 3 — Component Details Table](#section-3--component-details-table)
   - [Section 4 — Correlation Scatter Plot](#section-4--correlation-scatter-plot)
   - [Section 5 — Interactive Network Graph](#section-5--interactive-network-graph)
   - [Section 6 — Dependency Matrix](#section-6--dependency-matrix)
   - [Section 7 — Validation Report](#section-7--validation-report)
5. [Visualization Design Rationale](#visualization-design-rationale)
6. [Visual Encoding Reference](#visual-encoding-reference)
7. [Anti-Pattern Detection](#anti-pattern-detection)
8. [From Dashboard to Decisions](#from-dashboard-to-decisions)
9. [Performance](#performance)
10. [Commands](#commands)
11. [Programmatic API](#programmatic-api)
12. [Methodology Summary](#methodology-summary)

---

## What This Step Does

Visualization is the final step. It takes all outputs from Steps 2–5 — structural metrics, RMAV quality scores, impact scores, validation metrics — and synthesizes them into interactive dashboards. The goal is to move from numbers to decisions.

```
Steps 2–5 Outputs                Visualization               Output
──────────────────               ─────────────               ──────
M(v)  — metric vectors           Pipeline         →  Genieus live web app
Q(v)  — quality scores              │                (always-on, interactive)
I(v)  — impact scores               │             →  Static HTML dashboard
Spearman ρ, F1, …             ──────┘                (one-shot export)
```

The dashboard design follows one principle: every visual element should reduce time-to-decision for a specific architectural concern. Each view answers a different stakeholder question; the [From Dashboard to Decisions](#from-dashboard-to-decisions) section maps questions to views.

---

## Two Visualization Surfaces

The framework provides two distinct output surfaces that serve different use cases:

| Surface | How to Launch | Best For | Output |
|---------|--------------|----------|--------|
| **Genieus web app** | `docker compose up` → http://localhost:7000 | Interactive exploration, running analysis live, filtering and searching components | Live web app backed by FastAPI + Neo4j |
| **Static HTML dashboard** | `python bin/visualize_graph.py` | Sharing reports, archiving results, offline review, thesis figures | Self-contained `.html` file (~1–3 MB) |

Both surfaces render the same underlying data and use the same [visual encoding](#visual-encoding-reference). Choose the web app for exploratory analysis and the static dashboard for documentation and sharing.

---

## Genieus: Live Web Application

The Genieus frontend is a Next.js 15 application that communicates with the FastAPI backend (port 8000) and Neo4j database (port 7687). All analysis operations can be triggered and viewed directly from the browser without running any CLI commands.

### Launching

```bash
# Start the full stack (Neo4j + API + Frontend)
docker compose up --build

# Access points:
#   Web dashboard:      http://localhost:7000
#   API documentation:  http://localhost:8000/docs
#   Neo4j browser:      http://localhost:7474  (neo4j / password)
```

### Dashboard Tab

Provides a high-level overview equivalent to Sections 1–4 of the static dashboard:

- **KPI cards:** Component counts, criticality distribution, SPOF count, anti-pattern count, validation pass/fail
- **Criticality distribution pie chart:** Proportion of CRITICAL / HIGH / MEDIUM / LOW / MINIMAL components
- **Top-10 critical components list:** Ranked by Q(v) with RMAV sparklines
- **Validation summary:** Spearman ρ and F1 with pass/fail badges

### Graph Explorer Tab

The most analytically powerful view — an interactive 2D/3D force-directed graph of the dependency network:

- **Filter by layer:** Switch between app, infra, mw, and system layers
- **Search components:** Type to highlight and centre on specific components
- **Criticality overlay:** Nodes coloured and sized by criticality level
- **Component type overlay:** Nodes shaped and coloured by type (Application / Broker / Node / Topic / Library)
- **Click a node:** Opens a side panel showing all RMAV scores, I(v), criticality level, cascade count, and direct dependencies
- **2D / 3D toggle:** Three-dimensional layout gives better visual separation for dense graphs (> 100 components)

The Graph Explorer provides the visual proof of the methodology: the most central, largest, reddest nodes in the force-directed layout are exactly the components the analysis predicts as critical — and Step 5 confirms they are.

### Analysis Tab

Run structural analysis and quality scoring on demand from the browser:

- Select a layer and weight mode (equal / AHP)
- Trigger analysis and watch results update in real time
- Export results as JSON for use with `validate_graph.py`

### Simulation Tab

Run failure simulation and visualize cascade paths interactively:

- Select a target component and failure mode (CRASH / DEGRADED / PARTITION / OVERLOAD)
- View the cascade as an animated propagation on the network graph
- Compare I(v) scores against Q(v) predictions in a live scatter plot

### Settings Tab

Configure the Neo4j connection URI, username, and password. Settings are persisted in browser local storage and used by all API calls.

---

## Static HTML Dashboard

The static HTML dashboard is a single self-contained file generated by `visualize_graph.py`. It uses vis.js for the network graph and Chart.js for charts. The file requires no server — open it directly in any browser.

The dashboard uses a responsive single-page layout with tabbed layer navigation:

```
┌──────────────────────────────────────────────────────────────────┐
│  Software-as-a-Graph Dashboard        [Overview][App][Infra][Sys] │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │  48    │ │  127   │ │   5    │ │   3    │ │   2    │        │
│  │ Nodes  │ │ Edges  │ │Critical│ │ SPOFs  │ │Problems│        │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐             │
│  │ Criticality Dist.    │  │ RMAV Breakdown        │             │
│  │ [Pie Chart]          │  │ [Bar Chart]           │             │
│  └──────────────────────┘  └──────────────────────┘             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Q(v) vs I(v) Scatter Plot                                 │   │
│  │ Each point = one component · diagonal = perfect prediction│   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            Interactive Network Graph (vis.js)             │   │
│  │    Hover: details · Click: neighbours · Drag: reposition  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Component   │ Type   │  R   │  M   │  A   │  Q  │Level  │   │
│  │──────────────┼────────┼──────┼──────┼──────┼─────┼───────│   │
│  │ sensor_fusion│ App    │ 0.82 │ 0.75 │ 0.90 │ 0.84│ CRIT  │   │
│  │ main_broker  │ Broker │ 0.78 │ 0.65 │ 0.95 │ 0.80│ CRIT  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────┐                                   │
│  │ Validation: PASSED  ✓    │                                   │
│  │ Spearman ρ: 0.876 ✓     │                                   │
│  │ F1-Score:   0.893 ✓     │                                   │
│  └──────────────────────────┘                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Section 1 — Executive Overview

KPI cards providing the highest-level system health summary:

| KPI | Source | High Value Indicates |
|-----|--------|---------------------|
| **Nodes** | Step 1 | System scale |
| **Edges** | Step 1 | Dependency density |
| **Critical** | Step 3 Q(v) | Number of CRITICAL-classified components |
| **SPOFs** | Step 2 AP_c > 0 | Structural single points of failure |
| **Problems** | Step 3 | Detected architectural anti-patterns (see [Anti-Pattern Detection](#anti-pattern-detection)) |

Color-coded indicators: Critical and SPOFs use warning/danger colors when non-zero.

### Section 2 — Layer Comparison

Side-by-side RMAV summary metrics across all analyzed layers (app, infra, mw, system). Useful for identifying whether risk is concentrated at the application layer or the infrastructure layer. A system where the infrastructure layer has far more SPOFs than the application layer has a different remediation priority than one where the application layer dominates.

### Section 3 — Component Details Table

A sortable, filterable table showing every component with its complete analytical profile:

| Column | Source | Notes |
|--------|--------|-------|
| Component | Step 1 | ID and type |
| R, M, A, V | Step 3 | Individual RMAV dimension scores (hover for interpretation) |
| Q(v) | Step 3 | Composite quality score |
| I(v) | Step 4 | Simulated impact score (ground truth) |
| Level | Step 3 | CRITICAL / HIGH / MEDIUM / LOW / MINIMAL |
| SPOF | Step 2 | ✓ if AP_c > 0 |

Sort by Q(v) to prioritise by predicted criticality. Sort by I(v) to prioritise by confirmed impact. Filter by Level = CRITICAL and A(v) > 0.7 to identify the specific components most in need of redundancy. Filter by M(v) > 0.7 to find refactoring targets.

### Section 4 — Correlation Scatter Plot

Plots Q(v) (horizontal axis) against I(v) (vertical axis) for every component. This is the visual proof of the methodology's central claim.

**Reading the scatter plot:**

- **Points near the diagonal:** Accurate predictions — topology correctly identifies which components are most critical
- **Points above the diagonal:** Under-predicted components — the model under-estimated their impact; they have higher actual impact than predicted. Often caused by redundancy that the topology captures but that was not used during the cascade (e.g., a standby broker that was also failing)
- **Points below the diagonal:** Over-predicted components — structurally important but resilient in practice; they have built-in redundancy that absorbed the failure
- **Cluster of points at (low Q, low I):** MINIMAL components — the model correctly identifies the bulk of non-critical components; this cluster should be dense and near the origin

The Spearman ρ value is displayed on the plot. A tight cluster around the diagonal with ρ ≥ 0.70 is the visual target.

### Section 5 — Interactive Network Graph

A force-directed graph rendered with vis.js. The force-directed layout organises components such that highly connected (high centrality) components migrate toward the centre and low-degree components settle at the periphery — the layout itself becomes a qualitative criticality indicator.

**Node interactions:**

| Action | Effect |
|--------|--------|
| Hover | Tooltip showing component type, Q(v), I(v), criticality level, SPOF flag |
| Click | Highlights all direct DEPENDS_ON neighbours; dims the rest |
| Double-click | Centres and zooms to the selected node |
| Drag | Repositions the node (layout adapts) |
| Scroll | Zoom in/out |

**When to use vs. the Dependency Matrix:**

Use the network graph for systems up to ~80 components. Above that threshold, edge crossings dominate and the graph becomes difficult to read. For denser systems, switch to the Dependency Matrix (Section 6), which scales to hundreds of components without visual saturation.

### Section 6 — Dependency Matrix

The Dependency Matrix is a directed adjacency matrix that provides a condensed view of system-wide coupling. While the network graph is best for exploring local neighbourhoods, the matrix scales to hundreds of components without visual saturation.

**Technical Specifications:**
- **Matrix Type:** Directed adjacency matrix $A$, where $A_{ij} = w(e)$ if a dependency exists from component $i$ to component $j$.
- **Bandwidth Minimization:** Components are ordered using the **Reverse Cuthill-McKee (RCM)** algorithm (Cuthill & McKee, 1969). RCM reorders nodes to minimize the matrix bandwidth, bringing non-zero cells (dependencies) as close to the diagonal as possible.
- **Cluster Identification:** In this topological ordering, tightly coupled component clusters emerge as dense blocks along the diagonal. Unlike random or alphabetical ordering, RCM ensures that these blocks represent genuine structural communities rather than ordering artifacts.

**Reading the matrix:**

- **Blocks along the diagonal:** Tightly coupled component clusters — these components depend heavily on each other and should be assessed as a unit for redundancy planning.
- **Full rows:** A component with high out-degree — it depends on many others (high efferent coupling, M(v) risk).
- **Full columns:** A component with high in-degree — many others depend on it (high reliability risk, R(v)).

#### Worked Example: Topological Reordering

The power of the matrix lies in the reordering. Consider a system with two isolated clusters.

````carousel
```text
Unordered Adjacency Matrix
(Alphabetical)

      A B C D E F
    A . . X . . .
    B . . . X . X
    C X . . . . .
    D . X . . . .
    E . . . . . X
    F . X . . X .
```
<!-- slide -->
```text
Ordered Adjacency Matrix
(Topological / RCM)

      A C B D F E
    A . X . . . .
    C X . . . . .
    B . . . X X .
    D . . X . . .
    F . . X . . X
    E . . . . X .
```
````

In the **Unordered** view, dependencies appear scattered, and the two functional clusters $\{A, C\}$ and $\{B, D, E, F\}$ are invisible. In the **Topological** view, the clusters emerge as distinct 2x2 and 4x4 blocks. The dashboard defaults to this Topological view to ensure that architectural "gravity" is immediately apparent.

### Section 7 — Validation Report

Displays all eleven validation metrics from Step 5 with pass/fail badges:

- **Primary gates** (Spearman ρ, p-value, F1-Score, Top-5 Overlap): shown with ✓/✗ indicators
- **Secondary gate** (RMSE): shown with ✓/✗
- **Reported metrics** (Kendall τ, Pearson r, Precision, Recall, Cohen's κ, Top-10 Overlap, NDCG@K, MAE): shown without gates

The validation report answers the question: "Can I trust the Q(v) predictions in this dashboard?" If all primary gates pass, the scatter plot and component table should be used with confidence for architectural decision-making. If any primary gate fails, review the [Interpreting Results](validation.md#interpreting-results) section in the validation docs before acting on the predictions.

---

## Visualization Design Rationale

This section formally justifies the visual encoding decisions used in the dashboards, aligning with graph visualization literature and the specific needs of software architectural analysis.

| Encoding Decision | Theoretical Backing | Domain-Specific Motivation |
|-------------------|---------------------|---------------------------|
| **Force-Directed Layout** | Encodes centrality implicitly through emergent positioning (Eades, 1984), where high-degree nodes migrate to the center. | Aligns with the goal of making structural importance visually self-evident without needing explicit annotation or complex navigation. |
| **Color for Criticality** | Color is a pre-attentive visual feature (Ware, 2012) that allows for rapid detection and categorization during visual search. | Enables engineers to instantly identify high-risk components (e.g., "red nodes") across a dense system graph at a single glance. |
| **Size for Quality Score** | Human perception of area is highly effective for judging relative magnitude and importance (Cleveland & McGill, 1984). | Provides a quantitative indicator of $Q(v)$, ensuring that the most impactful components dominate the visual field. |
| **Shape for Type** | Shape is an effective channel for categorical separation without interfering with color or size (Munzner, 2014). | Distinguishes between different architectural roles (e.g., Broker vs. Application) without confounding the risk-based priority. |
| **Edge Thickness** | Line width is a standard channel for encoding quantitative weights in networks (Gibson et al., 2002). | Surfaces the strength of dependency (QoS-derived weights), highlighting the "heaviest" paths of failure propagation. |
| **Dependency Matrix** | Heatmap encoding with Reverse Cuthill-McKee (RCM) ordering minimizes matrix bandwidth (Cuthill & McKee, 1969). | Objectively surfaces tightly coupled component clusters as diagonal blocks, eliminating ordering artifacts. |

These choices ensure that the visualization is not merely an implementation detail but a rigorous tool for architectural decision-making, where aesthetics serve the goal of information density and cognitive efficiency.

---

## Visual Encoding Reference

All visual elements use a consistent encoding across both the Genieus web app and the static HTML dashboard.

### Criticality Level Colours

| Level | Colour | Hex | Meaning |
|-------|--------|-----|---------|
| CRITICAL | Red | `#E74C3C` | Statistical outlier — Q(v) > Q3 + 1.5×IQR |
| HIGH | Orange | `#E67E22` | Above 75th percentile |
| MEDIUM | Yellow | `#F1C40F` | Above 50th percentile |
| LOW | Green | `#2ECC71` | Above 25th percentile |
| MINIMAL | Blue | `#3498DB` | Bottom 25th percentile |

### Network Graph Encoding

| Visual Property | Maps To | Notes |
|----------------|---------|-------|
| **Node size** | Q(v) quality score | Larger = higher predicted criticality |
| **Node colour** | Criticality level | Uses colour table above |
| **Node shape** | Component type | Circle = Application · Diamond = Broker · Square = Node · Triangle = Topic · Star = Library |
| **Edge thickness** | Dependency weight w(e) | Thicker = higher QoS-derived weight |
| **Edge arrow** | Dependency direction | Points from dependent to dependency (DEPENDS_ON direction) |
| **Spatial position** | Force-directed layout | Central = high centrality; peripheral = low degree |

### KPI Card Colours

| State | Colour | When |
|-------|--------|------|
| Normal | Grey/neutral | Value is expected or low-concern |
| Warning | Orange | SPOFs > 0 or anti-patterns detected |
| Danger | Red | Critical count > 0 or validation failed |
| Success | Green | Validation passed |

---

## Anti-Pattern Detection

The "Problems" KPI counts detected architectural anti-patterns — structural configurations that empirically increase failure risk. The dashboard highlights which components participate in each detected pattern.

| Anti-Pattern | Definition | Risk |
|-------------|------------|------|
| **God Component** | A single Application with both in-degree and out-degree in the top 10% of the system | Combines high Reliability risk and high Maintainability risk — failure impact is very broad and it is difficult to change safely |
| **Circular Dependency** | A directed cycle of length ≥ 2 in the DEPENDS_ON graph | Prevents independent deployment; failure in one component can cascade around the loop |
| **Chain Topology** | A linear sequence of ≥ 4 components with no branching (maximum in-degree = out-degree = 1) | High fragility — any link in the chain is a SPOF; no redundancy is possible without architectural change |
| **Isolated Cluster** | A connected subgraph with no edges to the rest of the system other than through a single bridge component | The bridge component is an extreme SPOF; the entire cluster is unreachable if the bridge fails |

Anti-patterns are detected during graph model construction (Step 1) and stored as properties on the relevant components. The dashboard surfaces them as overlays on the network graph (toggle with the "Anti-Patterns" layer button in the Genieus Graph Explorer) and as highlighted rows in the component table.

---

## From Dashboard to Decisions

The dashboard is designed to support specific architectural decisions by different stakeholder roles. This table maps the most common questions to the relevant dashboard view and the action it supports.

| Stakeholder | Question | Dashboard View | Action |
|-------------|----------|----------------|--------|
| **Reliability Engineer** | Which components need redundancy most urgently? | Component table: filter CRITICAL + high A(v); network graph: red nodes | Add redundant instance or failover path for top-A(v) CRITICAL components |
| **Software Architect** | Where is our coupling complexity worst? | Component table: sort by M(v) descending; dependency matrix: full rows | Refactor high-M(v) components to reduce betweenness and out-degree |
| **DevOps / SRE** | Which components should we monitor most closely? | Component table: filter by SPOF flag; Step 4 I(v) column | Configure health checks, circuit breakers, and alerting for high-I(v) SPOFs |
| **Security Engineer** | Where should we prioritise hardening? | Component table: sort by V(v) descending | Harden and isolate top-V(v) components; add access controls and anomaly detection |
| **Engineering Manager** | What is our overall system risk posture? | Executive overview KPI cards; layer comparison | Use Critical/SPOF counts as sprint planning input; target zero CRITICAL at next milestone |
| **Researcher / Architect** | Are the predictions trustworthy? | Validation report; correlation scatter plot | Accept predictions if primary gates pass; review outliers if any gate fails |
| **Any stakeholder** | Where are hidden coupling clusters? | Dependency matrix: diagonal blocks | Refactor tightly coupled clusters to improve independent deployability |

---

## Performance

Dashboard generation time depends on which components are included. The dominant cost is serialising the vis.js network graph JSON for large systems.

| Content | Small (< 30) | Medium (30–100) | Large (100–300) |
|---------|:------------:|:---------------:|:---------------:|
| Full dashboard (all sections) | < 2 s | 3–5 s | 8–10 s |
| Without network graph (`--no-network`) | < 1 s | 1–2 s | 2–3 s |
| Without matrix + network (`--no-network --no-matrix`) | < 1 s | < 1 s | < 2 s |

For large systems in automated pipelines (CI/CD), use `--no-network` to keep generation within the 10-second target (SRS REQ-PERF-04). The network graph is most useful for interactive exploration — the Genieus Graph Explorer serves this use case better for large graphs anyway (using lazy loading and WebGL rendering).

---

## Commands

```bash
# ─── Basic dashboard generation ───────────────────────────────────────────────
# Single layer
python bin/visualize_graph.py --layer app --output dashboards/app.html

# Multiple layers (side-by-side layer comparison)
python bin/visualize_graph.py --layers app,infra,system --output dashboards/multi.html

# All layers (equivalent to --layers app,infra,mw,system)
python bin/visualize_graph.py --all --output dashboards/full.html

# Generate and open immediately in browser
python bin/visualize_graph.py --all --output dashboard.html --open

# ─── Demo mode (no Neo4j required) ────────────────────────────────────────────
# Generates a dashboard from synthetic sample data.
# Use this to explore the dashboard layout and visual encoding without
# importing a real system topology.
python bin/visualize_graph.py --demo --output demo.html --open

# ─── Performance tuning flags ─────────────────────────────────────────────────
# Skip network graph (fastest; recommended for large systems or CI pipelines)
python bin/visualize_graph.py --all --no-network --output report.html

# Skip dependency matrix (saves memory for very large edge counts)
python bin/visualize_graph.py --all --no-matrix --output report.html

# Skip validation report (if validation was not run)
python bin/visualize_graph.py --all --no-validation --output report.html

# Minimal dashboard: KPIs, charts, and component table only
python bin/visualize_graph.py --all --no-network --no-matrix --output minimal.html

# ─── Full pipeline: generate → import → analyze → simulate → validate → visualize
python bin/run.py --all --layer system --open

# ─── List available layers ────────────────────────────────────────────────────
python bin/visualize_graph.py --list-layers

# ─── Docker stack (Genieus web app) ───────────────────────────────────────────
docker compose up --build
# → http://localhost:7000   (Genieus dashboard)
# → http://localhost:8000   (FastAPI with Swagger UI)
# → http://localhost:7474   (Neo4j Browser)
```

---

## Programmatic API

The `VisualizationService` can be used directly in Python scripts for integration into CI/CD pipelines, automated reporting workflows, or custom dashboards.

```python
from src.core import create_repository
from src.analysis import AnalysisService
from src.simulation import SimulationService
from src.validation import ValidationService
from src.visualization import VisualizationService

# Initialize services
repo      = create_repository()  # uses env vars for Neo4j connection
analysis  = AnalysisService(repo)
simulation = SimulationService(repo)
validation = ValidationService(analysis, simulation)

viz = VisualizationService(
    analysis_service=analysis,
    simulation_service=simulation,
    validation_service=validation,
    repository=repo
)

# Generate dashboard for specific layers
output_path = viz.generate_dashboard(
    output_file="output/dashboard.html",
    layers=["app", "system"],
    include_network=True,    # set False for large systems
    include_matrix=True,
    include_validation=True,
)

print(f"Dashboard saved to: {output_path}")
repo.close()
```

See `examples/example_visualization.py` for a complete runnable example.

---

## Methodology Summary

Step 6 closes the methodology loop by making all six steps' outputs accessible to the humans who act on them. Together the pipeline moves from raw architecture description to informed, quantified decisions:

| Step | Output | Answers |
|------|--------|---------|
| 1. Graph Model | G(V, E, w) | What does the system look like as a graph? |
| 2. Structural Analysis | M(v) — 13 metrics per component | Where is each component positioned topologically? |
| 3. Quality Scoring | Q(v), RMAV, criticality level | Which components are structurally at-risk, and why? |
| 4. Failure Simulation | I(v) — empirical impact per component | Which components actually cause the most damage when they fail? |
| 5. Validation | Spearman ρ, F1, pass/fail | Can we trust the topology-based predictions? |
| **6. Visualization** | **Dashboard** | **Where should we act, and what should we do?** |

---

← [Step 5: Validation](validation.md) | [README](../README.md)