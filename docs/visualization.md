# Step 8: Visualize

**Translate quantitative analysis into an interactive dashboard for architectural decision-making.**

← [Step 7: Prescribe](prescription.md) | [README](../README.md)

---

## Table of Contents

1. [What This Step Does](#1-what-this-step-does)
2. [Two Visualization Surfaces](#2-two-visualization-surfaces)
3. [Dashboard Tabs](#3-dashboard-tabs)
4. [Visual Encoding Reference](#4-visual-encoding-reference)
5. [Anti-Pattern Reports and CI/CD Integration](#5-anti-pattern-reports-and-cicd-integration)
6. [From Dashboard to Decisions](#6-from-dashboard-to-decisions)
7. [Scale Limits](#7-scale-limits)
8. [Commands](#8-commands)
9. [Programmatic API](#9-programmatic-api)

---

## 1. What This Step Does

Visualization is the final step. It takes the outputs of Steps 2–5 — structural metric vectors $M(v)$, dual-pathway prediction outputs (diagnostic RM $Q^*(v)$, learned GNN $\hat{I}^*(v)$, Top-K Triage profiles), simulation impact scores $I^*(v)$, and validation metrics — and synthesizes them into one interactive dashboard. The goal is to move from numbers to decisions: every view answers a specific stakeholder question, tabulated in [From Dashboard to Decisions](#6-from-dashboard-to-decisions).

```
Steps 2–5 Outputs                    Visualization              Output
─────────────────────────────        ─────────────              ──────
M(v)  — Tier 1 structural metrics    Pipeline         →   HTML dashboard
Q*(v) — Diagnostic RM profiles       │                    (archivable research artifact)
Î*(v) — Learned GNN forecasts        │                →   SMART live web app
Triage — Scoped stakeholder actions  │                    (operational practitioner tool)
I*(v), IR, IM, IA — ground truths    │
ρ, F1, PG, specialist metrics        │
Anti-pattern report (19 smells) ─────┘
```

### Implementation Map

| Concern | Module |
|:---|:---|
| Stage orchestration (`collect` → `build_html` → write) | [saag/visualization/service.py](../saag/visualization/service.py) |
| Pulling per-layer data out of the other services | [saag/visualization/collector.py](../saag/visualization/collector.py) |
| HTML shell, tabs, tables, network, matrix | [saag/visualization/dashboard.py](../saag/visualization/dashboard.py) |
| Chart.js snippets | [saag/visualization/charts.py](../saag/visualization/charts.py) |
| Every colour the dashboard renders | [saag/visualization/palette.py](../saag/visualization/palette.py) |
| The `LayerData` aggregate both stages pass around | [saag/visualization/models.py](../saag/visualization/models.py) |

`build_html()` is pure — it reads only the `LayerData` list it is handed, touching neither the repository nor the filesystem. That is what lets `--demo` render the real dashboard from a fixture with no database, so demo mode doubles as a smoke test of the production assembly path.

---

## 2. Two Visualization Surfaces

| Surface | Use Case | Output |
|---------|---------|--------|
| **HTML dashboard** | Reproducible research artifact; sharing with stakeholders who have no infrastructure; archiving validation results | Single `.html` file, typically 50 KB–2 MB |
| **SMART web application** | Interactive real-time exploration; triggering pipeline steps from a browser; collaborative review | Next.js frontend at `http://localhost:7000` against the FastAPI backend at `:8000` — see [smart/README.md](../smart/README.md) |

Both surfaces share the same data source (Neo4j + pipeline outputs) and the same anti-pattern results.

> **The HTML file embeds its data, not its libraries.** All analysis results are inlined, but Chart.js, Cytoscape.js, D3 and the Inter webfont are loaded from CDNs (`unpkg`, `jsdelivr`, `d3js.org`, `fonts.googleapis.com`). The file renders text and tables offline; charts and the network graph need network access. Vendor the libraries locally if you need a genuinely air-gapped artifact.

---

## 3. Dashboard Tabs

The dashboard renders six tabs, switched client-side. Optional tabs display a placeholder when their input data is absent rather than disappearing, so the layout is stable across runs.

Sections render for the **primary layer** — `system` when it was collected, otherwise the first layer requested.

### 3.1 Overview

Six KPI cards aggregated across every collected layer:

| Card | Value | Colour rule |
|------|-------|-------------|
| Total Components | Sum of nodes across layers | Neutral |
| Total Dependencies | Sum of edges across layers | Neutral |
| Critical Assets | Count of CRITICAL components | Red if > 0, else green |
| SPOFs Detected | Count of articulation points | Amber if > 0, else green |
| Anti-Patterns | Count of detected instances | Amber if > 0, else green |
| Validation ρ | Spearman ρ for the primary layer | Green if > 0.7, else amber |

Below the cards: a **criticality doughnut** (CRITICAL/HIGH/MEDIUM/LOW/MINIMAL) and an **RM dimension breakdown stacked bar** ("RM dimension comparison — top 6") for the top 6 components, with Fault Tolerance / Availability / Maintainability segments sized by each dimension's effective contribution to $Q(v)$ — Reliability itself is not a segment; it is the sum of the Fault Tolerance and Availability segments.

When more than one layer was collected, a **layer comparison** grouped bar chart compares density, nodes/100, average impact and validation ρ across them — useful for judging whether a reliability concern lives in application topology or in infrastructure.

### 3.2 Component Table

A sortable, filterable table, one row per component, capped at the **first 100 components** by descending $Q(v)$.

| Column | Content |
|--------|---------|
| ID | Component identifier |
| Name | Human-readable name |
| Type | Application / Broker / Topic / Node / Library |
| Q(v) | Composite criticality score |
| Level | Coloured criticality badge |
| Impact | Simulation-derived $I(v)$ |
| R, M, FT, A | Reliability, Maintainability, Fault Tolerance, Availability scores (FT and A are Reliability sub-characteristics, reported alongside it) |
| RM | Segmented bar; 3 unweighted segments (Fault Tolerance / Availability / Maintainability), each capped at 33 % of the width |
| SPOF | Badge when the component is an articulation point |

**Controls:** type dropdown, criticality-level dropdown, free-text search across the whole row, and click-to-sort on any column header (numeric where the column parses as a number, lexicographic otherwise).

Below the table, the "RM quality dimension breakdown" chart shows Fault Tolerance / Availability / Maintainability segments — each weighted by its effective contribution to $Q(v)$, not AHP-derived at this composite level — for the top 10 components.

MPCI and FOC are **not** table columns — they appear in the network graph tooltip.

**Architectural Explanations** follow the table when the analysis service produced them (e.g. `analyze_graph.py --explain`): one card per component with an automated risk narrative and triage guidance.

### 3.3 Validation

Renders unless `--no-validation` is given. KPI cards for Spearman ρ, F1 (top-k), precision and recall lead the tab.

**Composite scatter — $Q^*(v)$ vs $I^*(v)$.** The central visual proof of the methodology's claim. Each point is a component: horizontal axis predicted $Q(v)$, vertical axis simulated $I(v)$. Points near the dashed diagonal indicate good prediction; upper-left points are false negatives (impactful components the model underrated), lower-right are false positives. Points are coloured by criticality level, and the Spearman ρ appears in the chart title.

**Per-dimension ρ bars.** A horizontal progress-bar panel for R, M and A (Reliability, Maintainability, and the Availability sub-characteristic, reported as a diagnostic) using the RM semantic colours. Negative ρ is valid but cannot be a CSS width, so it renders as 0 % width with a red value label. When multi-seed data is loaded, the stability line chart sits beside it.

**Per-dimension scatter plots** (when `include_per_dim_scatter` and the dimensional ground truths are present) — three plots against the Step 5 per-dimension ground truths:

| Plot | Horizontal | Vertical | Answers |
|------|-----------|----------|---------|
| Reliability | R(v) | IR(v) | Do cascade-propagation predictions match cascade dynamics? |
| Maintainability | M(v) | IM(v) | Do coupling predictions match change propagation? |
| Availability | A(v) | IA(v) | Do SPOF predictions match connectivity disruption? |

These are the most diagnostic view for understanding which RM characteristic drives the overall correlation and which carries systematic bias.

**Validation gates.** A metrics box scoring four gates:

| Gate | Metric | Threshold |
|------|--------|-----------|
| G1 | Spearman ρ(Q, I) | > 0.7 |
| G2 | F1-score | > 0.6 |
| G3 | Top-K precision | > 0.5 |
| G4 | Top-5 overlap | > 0.6 |

**Multi-seed stability** renders when `--multi-seed` supplied validation JSONs: KPI cards for mean/min/max ρ and seed count, plus a line chart of ρ (solid purple) and F1 (dashed green) across seeds. A tight range indicates the prediction is robust to graph topology variation.

### 3.4 Cascade Risk

Renders when `--cascade-file` supplied QoS ablation results; otherwise shows a placeholder. Three stat cards — QoS Gini coefficient, Wilcoxon p-value (green below 0.05), and Δρ enrichment (green above 0.03) — above a dual horizontal bar chart of the top 12 components by cascade risk: grey for the topology-only baseline, purple for the QoS-enriched score. Components downstream of `RELIABLE` or tight-deadline topics show a larger purple-to-grey ratio, identifying where QoS topology amplifies blast radius beyond the structural prediction.

**Expected JSON schema.** No script in this repository currently emits this shape — the file comes from an external QoS ablation experiment. `reproduce/run_experiment.py` reports the same statistics but under different keys (`wilcoxon_pvalue`) and without a per-component array, so its output is **not** a drop-in for `--cascade-file`.

```json
{
  "components": [
    {"id": "...", "name": "...", "type": "...", "level": "...",
     "cascade_risk": 0.81, "cascade_risk_topo": 0.71, "cascade_depth": 3}
  ],
  "qos_gini":   0.347,
  "wilcoxon_p": 0.031,
  "delta_rho":  0.052
}
```

Missing keys degrade gracefully: `cascade_risk_topo` defaults to 88 % of `cascade_risk`, the scalars to 0.0 / 1.0 / 0.0.

### 3.5 Topology

**Interactive network graph** (unless `--no-network`), rendered with Cytoscape.js using the `cose-bilkent` layout, with components grouped into compound boundaries for the Application, Middleware and Infrastructure layers. Drag to inspect dense subgraphs; zoom and pan with the wheel or trackpad; click a node for its metadata. Above 500 nodes the graph is replaced by a note and only the matrix renders.

**Dependency matrix** (unless `--no-matrix`): a directed adjacency matrix where $A_{ij}$ is the weight of the edge from component $i$ to $j$, ordered by descending $Q(v)$ so the most critical components sit at the top-left. Cell opacity scales with edge weight.

| Pattern | Meaning |
|---------|---------|
| Dense top-left block | Inter-coupling between high-criticality components — high failure coordination risk |
| Full row | High out-degree: many efferent couplings → high $M(v)$ maintenance risk |
| Full column | High in-degree: many dependents → high $R(v)$ blast risk |

**Anti-pattern catalog** closes the tab when any were detected — one card per instance with a severity badge, description and the components involved. See [antipatterns.md](antipatterns.md) for the full pattern catalog, detection signals and remediations.

### 3.6 MIL-STD-498

Renders when the analysis service produced hierarchy data; otherwise a placeholder. An indented CSS → CSCI → CSC → CSU tree with BPA$_\beta$ rollup scores:

```
CSS  (system)   BPA_β rollup
├── CSCI A       CBCI: 0.42   Q = 0.731
│   ├── CSC A1                 Q = 0.821
│   │   └── CSU sensor_fusion  Q = 0.840
│   └── CSC A2                 Q = 0.642
└── CSCI B       CBCI: 0.18   Q = 0.581
```

**CBCI (Cross-Boundary Coupling Index)** at CSCI level quantifies how tightly a subsystem couples to its neighbours. High CBCI (> 0.5) signals an architectural modularity violation — the subsystem boundary is not providing effective isolation. Relevant to MIL-STD-498 compliance reviews and to demonstrating subsystem independence to a certification authority.

---

## 4. Visual Encoding Reference

All colours come from one place — [saag/visualization/palette.py](../saag/visualization/palette.py) — so chart series, network nodes, matrix cells and badges cannot drift apart.

### Criticality (nodes, badges, scatter points, matrix labels)

| Level | Colour | Hex |
|-------|--------|-----|
| CRITICAL | Red | `#A32D2D` |
| HIGH | Amber | `#854F0B` |
| MEDIUM | Blue | `#185FA5` |
| LOW | Green | `#3B6D11` |
| MINIMAL | Grey | `#5F5E5A` |

### RM dimensions (stacked bars, per-dimension ρ bars)

| Characteristic | Colour | Hex | Effective weight in stacked bar |
|-----------|--------|-----|-----------|
| Reliability | Purple | `#534AB7` | — (sum of the Fault Tolerance and Availability segments below; not itself a bar segment) |
| Fault Tolerance (Reliability sub-characteristic) | Indigo | `#3B4FA1` | 0.29 |
| Availability (Reliability sub-characteristic) | Coral | `#993C1D` | 0.51 |
| Maintainability | Teal | `#0F6E56` | 0.20 |

Weights read from `saag/visualization/palette.py`'s `EFFECTIVE_WEIGHTS` (declared constants, not AHP-derived at this composite level — see [criticality.md](criticality.md)).

### Node size

Scaled from $Q(v)$ as `Q × 30 + 10`, so CRITICAL components are visually largest before any label is read. The dependency matrix reuses this value for its ordering.

### Edge thickness

Frequency-weighted with logarithmic scaling: $\text{thickness} = 1.5 + 2.5 \times \log_{10}(1 + w)$.

- **Topic edges** — weighted by the messaging frequency of the Topic (`frequency` or `topic_frequency`, in Hz).
- **Library edges (`USES`)** — weighted by the aggregate messaging frequency of the dependent Application (sum over all its publish/subscribe topics).
- **Everything else** — baseline weight 1.0.

### Node tooltip (click)

The dashboard opens a plain browser dialog with the fields the collector emitted:

```
[Component ID]
Name:           Sensor Fusion
Type:           Application
Score:          0.840 (CRITICAL)
MPCI:           0.080
FOC:            0.000
Impact:         0.790          (only when simulation has been run)
Cascade:        3 layers       (only when simulation has been run)
Anti-Patterns:  SPOF, GOD_COMPONENT   (only when any were detected)
```

The SMART Graph Explorer provides a richer right-side detail panel, 2D/3D layout switching, live search and overlay selection — see [smart/README.md](../smart/README.md).

---

## 5. Anti-Pattern Reports and CI/CD Integration

`cli/detect_antipatterns.py` runs the pattern catalog against any analyzed system and exits with a code suitable for a deployment gate:

| Code | Meaning | CI/CD action |
|------|---------|-------------|
| 0 | No anti-patterns detected | Allow deployment |
| 1 | Only MEDIUM patterns detected | Allow with warning |
| 2 | HIGH or CRITICAL patterns detected | **Block deployment** |

```yaml
- name: Run anti-pattern detection
  run: |
    PYTHONPATH=. python cli/detect_antipatterns.py \
        --layer system --severity critical,high --use-ahp \
        --output results/antipatterns.json
  # Exit code 2 fails this step and blocks deployment
```

Pass that report to the dashboard with `--antipatterns` to render the catalog from the saved file, filtered to the layer being visualized:

```bash
PYTHONPATH=. python cli/detect_antipatterns.py --layer system --output results/antipatterns.json
PYTHONPATH=. python cli/visualize_graph.py --layers app,system \
    --antipatterns results/antipatterns.json --output output/dashboard.html
```

Without `--antipatterns` the dashboard runs detection live against the analysis it already computed, so the catalog is populated either way. Supply the file when you want the dashboard to reflect exactly what the CI gate saw.

Full catalog, detection signals and remediations: [antipatterns.md](antipatterns.md).

---

## 6. From Dashboard to Decisions

| Stakeholder Question | Primary View | Secondary View |
|---------------------|-------------|----------------|
| What are the most critical components I must protect? | Overview KPI cards + Top 5 | Component table (sort by Q(v)) |
| Which components are structural SPOFs? | Component table (SPOF column) | Overview SPOF KPI card |
| Which application pairs are dangerously multi-channel coupled? | Topology → CHATTY_PAIR cards | Network tooltip (MPCI) |
| Which library failure has the largest simultaneous blast radius? | Component table (Type = Library, sort R) | Network graph neighbourhood |
| Is our reliability concern in app topology or infrastructure? | Overview layer comparison | Per-dimension scatter R vs IR |
| Do our topology predictions actually match failure impact? | Composite scatter Q(v) vs I(v) | Validation gates G1–G4 |
| Which RM dimension best predicts this system's failures? | Per-dimension ρ bars | Per-dimension scatter plots |
| Are there hidden cyclic dependencies? | Topology → CYCLIC_DEPENDENCY cards | Dependency matrix (off-diagonal symmetric blocks) |
| Does QoS topology amplify blast radius beyond structure? | Cascade risk dual-bar chart | Δρ and Wilcoxon stat cards |
| Are my predictions stable across graph seeds? | Multi-seed stability chart | Mean/min/max ρ KPI cards |
| Are we blocking deployment correctly? | Topology → CRITICAL anti-patterns | Exit code of `detect_antipatterns.py` |

---

## 7. Scale Limits

One limit is enforced in code: **above 500 nodes the network graph is skipped** and replaced with a note, because `cose-bilkent` layout time grows sharply and a hairball of that size is not readable anyway. The dependency matrix, scatter plots and component table all still render, and the matrix is the more informative view for dense graphs.

`--no-network` skips the graph at any size — the right choice for batch dashboard generation, where layout is the dominant cost. `--no-matrix` and `--no-validation` similarly trim their sections.

The component table caps at 100 rows regardless of graph size; use the CSV/JSON exports from Step 3 (Predict, with Step 4 Diagnose bundled by default) when you need the full ranking.

---

## 8. Commands

```bash
# ─── Standard dashboard generation ───────────────────────────────────────────
# --layer accepts comma-separated values; --layers is an explicit alias
PYTHONPATH=. python cli/visualize_graph.py --layer app,system --output output/dashboard.html
PYTHONPATH=. python cli/visualize_graph.py --layers app,system --output output/dashboard.html

# ─── With an anti-pattern report ──────────────────────────────────────────────
PYTHONPATH=. python cli/detect_antipatterns.py --layer system --use-ahp --output results/antipatterns.json
PYTHONPATH=. python cli/visualize_graph.py --layers app,system \
    --antipatterns results/antipatterns.json --output output/dashboard.html

# ─── With QoS cascade risk (see §3.4 for the expected schema) ─────────────────
PYTHONPATH=. python cli/visualize_graph.py --layer system \
    --cascade-file results/cascade.json --output output/dashboard_cascade.html

# ─── Open immediately in a browser (-b, since -o is --output) ────────────────
PYTHONPATH=. python cli/visualize_graph.py --layer app --open

# ─── Skip the network graph (large systems) ──────────────────────────────────
PYTHONPATH=. python cli/visualize_graph.py --layers system --no-network --output output/dashboard.html

# ─── Full pipeline in one command ────────────────────────────────────────────
PYTHONPATH=. python cli/run.py --all --layer app --open

# ─── Demo mode (no Neo4j) — renders the production dashboard from a fixture ──
PYTHONPATH=. python cli/visualize_graph.py --demo --open

# ─── Multi-seed validation + stability panel ─────────────────────────────────
for seed in 42 123 456 789 2024; do
    PYTHONPATH=. python cli/generate_graph.py --scale medium --seed $seed --output data/s${seed}.json
    PYTHONPATH=. python cli/import_graph.py --input data/s${seed}.json --clear
    PYTHONPATH=. python cli/analyze_graph.py  --layer app --use-ahp --output results/pred_s${seed}.json
    PYTHONPATH=. python cli/simulate_graph.py event --all --messages 50 --layer app
    PYTHONPATH=. python cli/simulate_graph.py failure --exhaustive --layer app \
                                  --output results/sim_s${seed}.json
    PYTHONPATH=. python cli/validate_graph.py results/pred_s${seed}.json results/sim_s${seed}.json \
                           --output results/val_s${seed}.json
done
PYTHONPATH=. python cli/multi_seed_summary.py results/val_s*.json
# Pass the expanded glob to --multi-seed
PYTHONPATH=. python cli/visualize_graph.py --layers app \
    --multi-seed results/val_s*.json \
    --output output/dashboard_multiseed.html
```

Full flag table: [cli-pipeline-guide.md §Step 8](cli-pipeline-guide.md).

---

## 9. Programmatic API

```python
from saag.infrastructure import create_repository
from saag.analysis import AnalysisService
from saag.prediction import PredictionService
from saag.simulation import SimulationService
from saag.validation import ValidationService
from saag.visualization import VisualizationService

repo       = create_repository()
analysis   = AnalysisService(repo)
prediction = PredictionService()
simulation = SimulationService(repo)
validation = ValidationService(analysis, prediction, simulation)

viz = VisualizationService(
    analysis_service=analysis,
    prediction_service=prediction,
    simulation_service=simulation,
    validation_service=validation,
    repository=repo,
)

output_path = viz.generate_dashboard(
    output_file="output/dashboard.html",
    layers=["app", "system"],
    include_network=True,           # skipped automatically above 500 nodes
    include_matrix=True,
    include_validation=True,
    include_per_dim_scatter=True,   # R/M/A/V scatter plots
    antipatterns_file=None,         # JSON path for the anti-pattern catalog
    cascade_file=None,              # JSON path for the Cascade Risk tab
)

print(f"Dashboard: {output_path}")
repo.close()
```

The three stages are separately callable when you need to inspect or synthesize the data in between:

```python
layers = viz.collect(layers=["system"])          # services + files → List[LayerData]
html   = viz.build_html(layers)                  # pure: no repository, no file I/O
Path("output/dashboard.html").write_text(html)
```

Because `build_html()` is pure, it renders equally well from hand-built `LayerData` — which is exactly how `--demo` works (see `_demo_layer_data()` in [cli/visualize_graph.py](../cli/visualize_graph.py)).

See [examples/example_visualization.py](../examples/example_visualization.py) for a complete runnable example.

---

← [Step 7: Prescribe](prescription.md) | [README](../README.md)
