# Step 6: Visualization

**Translate quantitative analysis into interactive dashboards for architectural decision-making.**

← [Step 5: Validation](validation.md) | [README](../README.md)

---

## Table of Contents

1. [What This Step Does](#what-this-step-does)
2. [Two Visualization Surfaces](#two-visualization-surfaces)
3. [Static HTML Dashboard](#static-html-dashboard)
   - [Section 1 — Executive Overview](#section-1--executive-overview)
   - [Section 2 — Layer Comparison](#section-2--layer-comparison)
   - [Section 3 — Component Details Table](#section-3--component-details-table)
   - [Section 3.5 — Architectural Explanations](#section-35--architectural-explanations)
   - [Section 4 — Validation Diagnostics](#section-4--validation-diagnostics)
   - [Section 5 — Interactive Network Graph](#section-5--interactive-network-graph)
   - [Section 6 — Dependency Matrix](#section-6--dependency-matrix)
   - [Section 7 — Validation Report](#section-7--validation-report)
   - [Section 8 — Multi-Seed Stability](#section-8--multi-seed-stability)
   - [Section 9 — Anti-Pattern Catalog](#section-9--anti-pattern-catalog)
   - [Section 9a — Cascade Risk / QoS Ablation](#section-9a--cascade-risk--qos-ablation)
   - [Section 10 — MIL-STD-498 Hierarchy](#section-10--mil-std-498-hierarchy)
4. [Visual Encoding Reference](#visual-encoding-reference)
5. [Genieus: Live Web Application](#genieus-live-web-application)
6. [Anti-Pattern Detection and CI/CD Integration](#anti-pattern-detection-and-cicd-integration)
7. [From Dashboard to Decisions](#from-dashboard-to-decisions)
8. [Performance](#performance)
9. [Commands](#commands)
10. [Programmatic API](#programmatic-api)
11. [What Comes Next](#what-comes-next)

---

## What This Step Does

Visualization is the final step. It takes all outputs from Steps 2–5 — structural metric vectors M(v), RMAV prediction scores Q(v), simulation impact scores I(v), and validation metrics — and synthesizes them into interactive dashboards. The goal is to move from numbers to decisions.

```
Steps 2–5 Outputs                    Visualization              Output
─────────────────────────────        ─────────────              ──────
M(v)  — 13 Tier 1 metrics            Pipeline         →   Static HTML dashboard
Q(v)  — R, M, A, V, composite            │                (archivable research artifact)
I(v), IR, IM, IA, IV — ground truths     │            →   Genieus live web app
ρ, F1, PG, specialist metrics    ────────┘                (operational practitioner tool)
Anti-pattern report (12 patterns)
```

The dashboard design follows one principle: every visual element should answer a specific stakeholder question. Each view corresponds to a row in the [From Dashboard to Decisions](#from-dashboard-to-decisions) table.

---

## Two Visualization Surfaces

| Surface | Use Case | Output |
|---------|---------|--------|
| **Static HTML dashboard** | Reproducible research artifact; sharing with stakeholders who have no infrastructure; archiving validation results for thesis or paper submission | Single self-contained `.html` file (~1–4 MB), embeds all data and charts |
| **Genieus web application** | Interactive real-time exploration; triggering pipeline steps from a browser; collaborative review sessions | Next.js frontend at `http://localhost:7000` communicating with FastAPI backend at `:8000` |

Both surfaces share the same visual encoding, the same data source (Neo4j + pipeline outputs), and the same anti-pattern detection results.

---

## Static HTML Dashboard

The dashboard is structured as ten sections (plus one conditional sub-section). All sections except Section 5 (network graph) render quickly even at xlarge scale. Sections are navigable via a fixed top navbar.

### Section 1 — Executive Overview

Six KPI cards summarizing the system at a glance:

| Card | Value | Colour |
|------|-------|--------|
| Total components | Count in the analysis layer | Neutral |
| Total dependencies | DEPENDS_ON edge count | Neutral |
| CRITICAL components | Q(v) = CRITICAL | Red |
| Structural SPOFs | AP_c_directed > 0 | Red |
| Anti-patterns detected | Total across all 12 patterns | Orange (CRITICAL) / Yellow (HIGH) / Blue (MEDIUM) |
| Validation status | PASS / FAIL | Green / Red |

Below the KPI cards: a criticality distribution pie chart (CRITICAL / HIGH / MEDIUM / LOW / MINIMAL counts) and an RMAV dimension bar chart showing mean R, M, A, V across all components in the current layer. The RMAV bar chart reveals which quality dimension is most compromised system-wide — a useful first signal for determining which remediation dimension to prioritize.

### Section 2 — Layer Comparison

A side-by-side bar chart comparing mean R(v), M(v), A(v), V(v) across all four analysis layers (`app`, `infra`, `mw`, `system`). This section answers the question: "Is our primary reliability concern in the application topology or the infrastructure topology?" A Reliability bar that is high in `infra` but low in `app` signals that the failure risk is concentrated at infrastructure nodes rather than pub-sub logic.

### Section 3 — Component Details Table

A sortable, filterable table with one row per component in the selected layer. Columns:

| Column | Content |
|--------|---------|
| ID | Component identifier |
| Name | Human-readable name |
| Type | Application / Broker / Topic / Node / Library |
| Q(v) | Composite criticality score |
| Level | CRITICAL / HIGH / MEDIUM / LOW / MINIMAL (coloured badge) |
| Impact | Simulation-derived I(v) |
| R | Reliability score |
| M | Maintainability score |
| A | Availability score |
| V | Vulnerability score |
| RMAV | AHP-weighted dimension bar |
| SPOF | SPOF badge if AP_c_directed > 0 |

**Filter controls:** Level dropdown (show only CRITICAL/HIGH), Type filter, free-text search. Default sort: descending Q(v).

Below the table, an **AHP-weighted RMAV stacked bar chart** shows the per-dimension contribution to Q(v) for the top-10 components.

**MPCI column guidance:** A non-zero MPCI identifies the "Multi-path Sink" pattern introduced in Step 3. These components have multiple independent failure vectors from the same dependents. Sorting by MPCI descending surfaces the highest multi-channel coupling risk. Components with MPCI > 0.10 warrant investigation of their topic sharing structure.

### Section 3.5 — Architectural Explanations

Rendered only when the analysis service produces a system-level explanation (e.g. when `--explain` is passed to `analyze_graph.py`). Shows a card per component with automated risk narrative and triage guidance derived from the RMAV pattern match.


### Section 4 — Validation Diagnostics

**Composite scatter: Q*(v) vs I*(v)**

The central visual proof of the methodology's claim. Each point is a component; horizontal axis = Q(v), vertical axis = I(v). Points near the diagonal indicate good prediction. Points in the upper-left quadrant (high I, low Q) are false negatives — critically impactful components the model underrated. Points in the lower-right (high Q, low I) are false positives.

Components are colour-coded by their simulation-derived I-criticality level. The Spearman ρ, its bootstrap 95% CI band, and an optional CI ribbon around the diagonal are displayed on the chart.

**Per-dimension ρ bars:**

A horizontal progress-bar panel immediately below the main scatter. Shows Spearman ρ for each of A, R, M, V and optionally the Infrastructure dimension. Bars use RMAV semantic colours (A=coral, R=purple, M=teal, V=pink). Bars are clamped to [0, 100 %] — negative ρ is displayed as 0 % width with a red value label.

**Per-dimension scatter plots (when simulation has been run):**

Four additional scatter plots, one per RMAV dimension, using the dimension-specific ground truths from Step 4:

| Plot | Horizontal | Vertical | Shows |
|------|-----------|----------|-------|
| Reliability | R(v) | IR(v) | Do cascade-propagation predictions match cascade dynamics? |
| Maintainability | M(v) | IM(v) | Do coupling predictions match change propagation? |
| Availability | A(v) | IA(v) | Do SPOF predictions match connectivity disruption? |
| Vulnerability | V(v) | IV(v) | Do security exposure predictions match compromise reach? |

The per-dimension scatter plots are the most diagnostic view for understanding which RMAV dimension is driving the overall correlation and which dimensions have systematic bias.

### Section 5 — Interactive Network Graph

A force-directed graph rendered with vis.js. The layout places high-betweenness components near the centre automatically — the layout itself is a qualitative criticality indicator before any colour coding is applied.

**Node interactions:**

| Action | Effect |
|--------|--------|
| Hover | Tooltip (see [Visual Encoding Reference](#visual-encoding-reference)) |
| Click | Highlights all direct DEPENDS_ON neighbours; dims the rest; opens component panel |
| Double-click | Centres and zooms to selected node |
| Drag | Repositions node (layout adapts) |
| Scroll | Zoom in/out |

**Component panel** (right-side drawer on click): full RMAV scores, I(v), criticality level, SPOF flag, MPCI value, FOC value (for Topics), cascade count/depth from simulation, direct dependency list with weights, any detected anti-pattern IDs with one-sentence descriptions.

**Overlay selector** (toolbar above graph): switch between *Criticality* overlay (default — colour by Q level), *Type* overlay (colour by node type), *RMAV:R* / *RMAV:M* / *RMAV:A* / *RMAV:V* overlays (colour by individual dimension score).

Use the network graph for systems up to ~80 components. Above that threshold switch to the Dependency Matrix, which scales without visual saturation.

### Section 6 — Dependency Matrix

A directed adjacency matrix A where A_{ij} = w(e) if a DEPENDS_ON edge exists from component i to component j. Components are ordered using the **Reverse Cuthill-McKee (RCM)** algorithm, which minimizes matrix bandwidth and brings tightly coupled clusters onto the diagonal.

**Reading the matrix:**

| Pattern | Meaning |
|---------|---------|
| Dense diagonal block | Tightly coupled cluster — assess as a unit for redundancy planning |
| Full row | High out-degree: many efferent couplings → M(v) risk |
| Full column | High in-degree: many dependents → R(v) risk |
| Off-diagonal dense block | Cross-cluster dependency — bridge component; inspect for SPOF |

The colour intensity of each cell encodes the QoS-derived edge weight w(e): dark cells represent high-priority, reliable, persistent flows; light cells represent low-priority best-effort flows. This lets an architect immediately see not just which dependencies exist but how critical each one is.

### Section 7 — Validation Report

The validation report answers: "Can I trust the Q(v) predictions in this dashboard?" It is organized in one metrics box.

**Methodology validation gates (G1–G4):**

| Gate | Metric | Threshold | Result |
|------|--------|-----------|--------|
| G1 | Spearman ρ(Q, I) | > 0.7 | ✓/✗ |
| G2 | F1-Score | > 0.6 | ✓/✗ |
| G3 | Top-K precision | > 0.5 | ✓/✗ |
| G4 | Top-5 Overlap | > 0.6 | ✓/✗ |

If any primary gate fails, each sub-panel provides an interpretation hint (e.g., "ρ(A, IA) below target — check AP_c_directed storage in Step 2").

### Section 8 — Multi-Seed Stability

Rendered when `--multi-seed` is given (with one or more validation JSON paths). Shows:

- **KPI cards**: Mean ρ, Min ρ, Max ρ, and seed count.
- **Stability line chart**: Spearman ρ (solid purple) and optionally F1 (dashed green) over the seed labels. A tight range indicates the prediction is robust to graph topology variation.

```bash
# Generate stability panel from five pre-validated seeds
python bin/visualize_graph.py --layer app \
    --multi-seed results/val_s42.json results/val_s123.json results/val_s456.json \
    --output output/dashboard_stability.html
```

> This section is the primary evidence for the multi-seed reproducibility claim in Definition G5 / §6.2 Section 8 of the thesis.

### Section 9 — Anti-Pattern Catalog

A dedicated dashboard section surfacing the results of `detect_antipatterns.py`. Organized in three expandable severity tiers.

**CRITICAL tier (block deployment):**

| Pattern ID | Name | Detection Signal | Risk |
|-----------|------|-----------------|------|
| SPOF | Single Point of Failure | AP_c_directed > 0 | Component removal partitions graph |
| SYSTEMIC_RISK | Systemic Risk Cluster | ≥ 3 CRITICAL components with mutual DEPENDS_ON clique | Correlated failure — multiple critical components fail together |
| CYCLIC_DEPENDENCY | Cyclic Dependency | Strongly connected component with ≥ 2 non-trivial members | Bi-directional blast: cascade in both directions |

**HIGH tier (address in current sprint):**

| Pattern ID | Name | Detection Signal | Risk |
|-----------|------|-----------------|------|
| GOD_COMPONENT | God Component | Q(v) > upper_fence AND (DG_in + DG_out) > 75th percentile | Single component concentrates coupling and blast radius |
| BOTTLENECK_EDGE | Bottleneck Edge | Edge betweenness > upper_fence of edge betweenness distribution | Single link carries disproportionate dependency traffic |
| BROKER_OVERLOAD | Broker Overload | A(v) for a broker > upper_fence of A distribution | Broker is saturated with routing responsibility |
| DEEP_PIPELINE | Deep Pipeline | Longest dependency path ≥ 5 hops | Failures cascade through many sequential components |

**MEDIUM tier (track as architectural debt):**

| Pattern ID | Name | Detection Signal | Risk |
|-----------|------|-----------------|------|
| TOPIC_FANOUT | Topic Fan-Out | FOC(t) > 75th percentile of FOC across all Topics | Topic is a distribution relay; its loss blasts many subscribers simultaneously |
| CHATTY_PAIR | Chatty Pair | Two components share ≥ 3 DEPENDS_ON edge paths (MPCI-derived) | Over-coupling via multiple shared topics; each is a separate failure vector |
| QOS_MISMATCH | QoS Mismatch | Publisher w_out(v) < subscriber w_in(v) by > threshold | High-reliability subscriber receives data from low-reliability publisher |
| ORPHANED_TOPIC | Orphaned Topic | Topic has SUBSCRIBES_TO but no active PUBLISHES_TO | Subscriber permanently starved — data source removed |
| UNSTABLE_INTERFACE | Unstable Interface | CouplingRisk_enh(v) > upper_fence | Component deeply embedded on both afferent and efferent sides; hardest to change safely |

For each detected instance the section shows: pattern name and severity badge, the component(s) involved (clickable to highlight in network graph), the specific metric evidence (e.g., "AP_c_directed = 0.62, w(v) = 0.71"), and the recommended remediation.

> **Note on TOPIC_FANOUT:** This pattern is newly visible after Step 1's fan-out augmentation (subscriber_count on Topic nodes) and Step 2's FOC metric. Topics that were previously invisible in the DEPENDS_ON graph now carry a measurable blast relay signal. A TOPIC_FANOUT detection means the topic has more subscribers than its 75th-percentile peer — it is a structurally exceptional distribution point.

> **Note on CHATTY_PAIR:** This pattern uses the `path_count` attribute on DEPENDS_ON edges introduced in Step 1. Two applications that share three or more topics have MPCI > 0 on each other's in-degree, identifying them as multi-channel coupled. The detection threshold is path_count ≥ 3 on any single DEPENDS_ON edge.

### Section 9a — Cascade Risk / QoS Ablation

Rendered when `--cascade-file` is given. Shows the QoS-enriched cascade risk contribution — the primary novel Middleware 2026 claim.

**Stat cards:**

| Card | Content |
|------|---------|
| QoS Gini coefficient | Heterogeneity of QoS reliability across all topics (higher = more diverse) |
| Wilcoxon p-value | Statistical significance of QoS enrichment vs topology-only (p < 0.05 = significant) |
| Δρ (enrichment) | Spearman ρ gain from adding QoS weighting to the cascade scorer |

**Dual horizontal bar chart:** For each of the top-12 components, two bars side by side:
- **Grey**: topology-only cascade risk baseline.
- **Purple**: QoS-enriched cascade risk score.

Components downstream of `RELIABLE` / tight-deadline topics appear with a larger purple-to-grey ratio, visually identifying where QoS topology amplifies blast radius beyond the structural prediction.

```bash
# Generate cascade ablation results then embed in dashboard
python tools/qos_ablation_experiment.py --layer mw --output results/cascade.json
python bin/visualize_graph.py --layer system \
    --cascade-file results/cascade.json \
    --output output/dashboard_cascade.html
```

### Section 10 — MIL-STD-498 Hierarchy

Rendered when the analysis service produces hierarchy data (requires structurally grounded hierarchy assignment — not random pool selection). Shows a recursive tree:

```
CSS  (system)   BPA_β rollup
├── CSCI A       CBCI: 0.42   Q = 0.731
│   ├── CSC A1                 Q = 0.821
│   │   └── CSU sensor_fusion  Q = 0.840
│   └── CSC A2                 Q = 0.642
└── CSCI B       CBCI: 0.18   Q = 0.581
```

**CBCI (Cross-Boundary Coupling Index)** at CSCI level quantifies how tightly coupled a subsystem is to its neighbours. High CBCI (> 0.5) signals an architectural modularity violation — the subsystem boundary does not provide effective isolation.

This section is relevant for MIL-STD-498 compliance reviews and for projects that need to demonstrate subsystem independence to an airworthiness or certification authority.

---

## Visual Encoding Reference

### Node Shape (by vertex type)

| Shape | Type | Failure Semantics |
|-------|------|-------------------|
| Circle | Application | Sequential pub-sub cascade (Rules 1–3) |
| Diamond | Library | Simultaneous blast to all USES consumers (Rule 4) |
| Hexagon | Broker | Logical cascade to exclusively-routed topics (Rule 2) |
| Rectangle | Node (infrastructure) | Physical cascade to all hosted components (Rule 1) |
| Octagon | Topic | Distribution relay — loss of publisher starves all subscribers |

The Library diamond shape is intentionally distinct because Library failures produce simultaneous blasts rather than sequential cascades. In a mixed-type network graph, diamonds that are connected to many circles (Applications) are high-priority MPCI candidates.

### Node Colour (by criticality level)

| Colour | Level | Q(v) range (typical) |
|--------|-------|---------------------|
| Red (dark) | CRITICAL | > Q3 + 1.5 × IQR |
| Orange | HIGH | Q3 < Q(v) ≤ upper_fence |
| Yellow | MEDIUM | Median < Q(v) ≤ Q3 |
| Light blue | LOW | Q1 < Q(v) ≤ Median |
| Grey | MINIMAL | Q(v) ≤ Q1 |

### Node Size

Proportional to Q(v). CRITICAL components are visually largest, making them immediately apparent before reading any labels.

### Node Border

| Style | Meaning |
|-------|---------|
| Solid thin | Normal component |
| Dashed thick | Structural SPOF (AP_c_directed > 0) |
| Solid thick | Multi-path sink (MPCI > 0.10) |

### Edge Encoding

| Property | Encoding |
|----------|---------|
| Thickness | Proportional to DEPENDS_ON edge weight w(e) |
| Colour | Edge type: structural (grey); highlighted on component click (orange) |
| Dashed edge | Bridge edge (part of Bridge Ratio) |

### Tooltip Fields (on node hover)

```
[Node Name]
Type:           Application / Broker / Topic / Node / Library
Criticality:    CRITICAL ████████████ 0.84
R(v):          0.63   M(v): 0.71   A(v): 0.91   V(v): 0.55
SPOF:          ✓ AP_c_directed = 0.62
MPCI:          0.08   (multi-channel coupling detected)
FOC:           n/a    (Topic nodes only)
I(v):          0.79   (simulation-derived; when available)
Cascade:       depth=2, count=7   (when simulation has been run)
Anti-patterns: SPOF, GOD_COMPONENT
```

For Topic nodes specifically, the tooltip replaces MPCI with FOC:
```
FOC:           0.83   subscriber_count=10
```

---

## Genieus: Live Web Application

A Next.js 16 / React 19 application at `http://localhost:7000`, backed by FastAPI at `:8000` and Neo4j at `:7687`.

**Five-tab structure:**

| Tab | Primary View | Function |
|-----|-------------|----------|
| Dashboard | KPI cards, criticality pie, top-10 list, validation badges, anti-pattern summary | System health at a glance |
| Graph Explorer | Interactive 2D/3D force-directed dependency graph | Topology exploration |
| Analysis | Layer selector, weight mode, result tables | Trigger and view Steps 2 & 3 |
| Simulation | Component selector, failure mode picker, cascade animation | Trigger and view Step 4 |
| Settings | Neo4j URI, credentials | Connection configuration |

### Graph Explorer Tab

The Graph Explorer is the primary architectural review interface.

- **Layer filter**: app / infra / mw / system
- **Search**: highlight and centre on a named component
- **Overlay selector**: Criticality / Type / RMAV:R / RMAV:M / RMAV:A / RMAV:V — changes node colour encoding
- **Anti-pattern filter**: show only components with a specific pattern (SPOF, GOD_COMPONENT, etc.)
- **Component panel** (click): full RMAV scores, I(v), criticality level, SPOF flag, MPCI, FOC, cascade count/depth, direct dependency list with weights and path_count, anti-pattern instances with remediation text
- **2D / 3D toggle**: three-dimensional layout for dense graphs (> 100 components)

### Analysis Tab

Triggers Steps 2 and 3 directly from the browser:
- Layer and weight mode selection
- Results table with full RMAV breakdown, anti-pattern annotations, MPCI and FOC columns
- Node-type stratified ρ panel (when validation data is present): shows ρ separately for Application, Library, Broker, Node
- Export results as JSON

### Simulation Tab

Triggers Step 4:
- Target component from dropdown (includes Library nodes for Rule 4 blast demonstration)
- Failure mode: CRASH / DEGRADED / PARTITION / OVERLOAD
- Animated cascade overlay distinguishing sequential cascades (Rules 1–3) from simultaneous blasts (Rule 4 Library failures)
- Per-dimension impact metrics (IR, IM, IA, IV) alongside composite I(v)

---

## Anti-Pattern Detection and CI/CD Integration

The `detect_antipatterns.py` tool runs the full 12-pattern catalog against any analyzed system. It is designed to integrate directly into CI/CD pipelines as a deployment gate.

### Exit Codes

| Code | Meaning | CI/CD Action |
|------|---------|-------------|
| 0 | No anti-patterns detected | Allow deployment |
| 1 | Only MEDIUM patterns detected | Allow with warning |
| 2 | HIGH or CRITICAL patterns detected | **Block deployment** |

### Usage

```bash
# Full detection across all layers
python bin/detect_antipatterns.py --layer system --output results/antipatterns.json

# Print human-readable catalog
python bin/detect_antipatterns.py --catalog

# Filter to CRITICAL only (strict gate)
python bin/detect_antipatterns.py --layer app --severity critical

# Run specific patterns
python bin/detect_antipatterns.py --layer system --pattern SPOF,SYSTEMIC_RISK,CHATTY_PAIR

# Use AHP-derived RMAV weights (recommended)
python bin/detect_antipatterns.py --layer system --use-ahp --output results/antipatterns.json
```

### CI/CD Pipeline Integration (GitHub Actions example)

```yaml
- name: Run anti-pattern detection
  run: |
    python bin/detect_antipatterns.py \
        --layer system \
        --severity critical,high \
        --use-ahp \
        --output results/antipatterns.json
  # Exit code 2 automatically fails this step and blocks deployment
```

### Dashboard Integration

The anti-pattern report (from `--output`) feeds Section 8 of the static HTML dashboard. To include it:

```bash
python bin/detect_antipatterns.py --layer system --output results/antipatterns.json
python bin/visualize_graph.py \
    --layers app system \
    --antipatterns results/antipatterns.json \
    --output output/dashboard.html
```

Without `--antipatterns`, Section 8 shows a "No anti-pattern report available" placeholder.

---

## From Dashboard to Decisions

| Stakeholder Question | Primary View | Secondary View |
|---------------------|-------------|----------------|
| What are the most critical components I must protect? | Executive Overview KPI cards | Component Table (sort by Q(v)) |
| Which components are structural SPOFs? | Component Table (SPOF filter = ✓) | Network Graph (dashed border nodes) |
| Which topics are blast relays for many subscribers? | Section 8 TOPIC_FANOUT patterns | Component Table (FOC column) |
| Which application pairs have dangerous multi-channel coupling? | Section 8 CHATTY_PAIR patterns | Component Table (MPCI column, sort descending) |
| Which library failure has the largest simultaneous blast radius? | Component Table (Type = Library, sort R(v)) | Network Graph (diamond nodes with red fill) |
| Is our reliability concern in app topology or infrastructure? | Layer Comparison R(v) bars | Per-dimension scatter R(v) vs IR(v) |
| Do our topology predictions actually match failure impact? | Correlation Scatter Q(v) vs I(v) | Validation Report (primary gates) |
| Does combining all four RMAV dimensions add value? | Validation Report (Predictive Gain PG) | Composite scatter Q*(v) vs I*(v) |
| Which RMAV dimension best predicts this system's failures? | Per-dimension ρ table | Per-dimension scatter plots |
| What happens if NavLib fails? | Simulation Tab → select NavLib → CRASH (Rule 4 blast) | Network Graph → highlight all diamond-connected circles |
| Are there hidden cyclic dependencies? | Section 8 CYCLIC_DEPENDENCY | Dependency Matrix (off-diagonal symmetric blocks) |
| Are we blocking deployment correctly? | Section 8 CRITICAL patterns | Exit code of detect_antipatterns.py in CI/CD |
| Is the architecture getting worse across commits? | Compare dashboards from successive CI runs | Layer Comparison trend (if stored) |
| Are my predictions stable across different graph seeds? | Validation Report (multi-seed sub-panel, if run) | — |

---

## Performance

| Scale | Dashboard Generation Time | Recommended Settings |
|-------|--------------------------|---------------------|
| tiny (≤ 10 components) | < 1s | All sections enabled |
| small (10–25) | ~2s | All sections enabled |
| medium (30–50) | ~5s | All sections enabled |
| large (80–100) | ~12s | Consider `--no-network` |
| xlarge (150–300) | ~40s | `--no-network` — use matrix instead |

`--no-network` skips the vis.js network graph (the dominant rendering cost). The dependency matrix, scatter plots, and component table still render quickly and are more informative for dense graphs anyway.

For system-layer analysis with all five node types (Application, Library, Broker, Node, Topic), allow ~25% additional time compared to app-layer-only analysis. The extra cost comes from computing FOC across all Topic nodes and rendering the larger per-dimension scatter plots.

---

## Commands

```bash
# ─── Standard dashboard generation ───────────────────────────────────────────
# --layer accepts comma-separated values; --layers is an explicit alias
python bin/visualize_graph.py --layer app,system --output output/dashboard.html
python bin/visualize_graph.py --layers app,system --output output/dashboard.html

# ─── With anti-pattern report ─────────────────────────────────────────────────
python bin/detect_antipatterns.py --layer system --use-ahp --output results/antipatterns.json
python bin/visualize_graph.py \
    --layers app,system \
    --antipatterns results/antipatterns.json \
    --output output/dashboard.html

# ─── With QoS cascade risk (§9a) ──────────────────────────────────────────────
python tools/qos_ablation_experiment.py --layer mw --output results/cascade.json
python bin/visualize_graph.py --layer system \
    --cascade-file results/cascade.json \
    --output output/dashboard_cascade.html

# ─── Open immediately in browser (-b, not -o which is --output) ───────────────
python bin/visualize_graph.py --layer app --open

# ─── Skip network graph (large systems, > 80 components) ─────────────────────
python bin/visualize_graph.py --layers system --no-network --output output/dashboard.html

# ─── Full pipeline in one command ─────────────────────────────────────────────
python bin/run.py --all --layer app --open

# ─── CI/CD gate (fail build if CRITICAL or HIGH anti-patterns detected) ───────
python bin/detect_antipatterns.py --layer system --severity critical,high
# exit code 2 → CI step fails → deployment blocked

# ─── Multi-seed validation + dashboard (§8 stability panel) ───────────────────
for seed in 42 123 456 789 2024; do
    python bin/generate_graph.py --scale medium --seed $seed --output input/s${seed}.json
    python bin/import_graph.py --input input/s${seed}.json --clear
    python bin/analyze_graph.py  --layer app --use-ahp --output results/pred_s${seed}.json
    python bin/simulate_graph.py event --all --messages 50 --layer app
    python bin/simulate_graph.py failure --exhaustive --layer app \
                                  --output results/sim_s${seed}.json
    python bin/validate_graph.py results/pred_s${seed}.json results/sim_s${seed}.json \
                           --output results/val_s${seed}.json
done
python bin/multi_seed_summary.py results/val_s*.json
# Pass expanded glob paths to --multi-seed
python bin/visualize_graph.py --layers app \
    --multi-seed results/val_s42.json results/val_s123.json results/val_s456.json results/val_s789.json results/val_s2024.json \
    --output output/dashboard_multiseed.html

# ─── Demo mode (no Neo4j) — smoke tests all chart paths ──────────────────────
python bin/visualize_graph.py --demo --open
```

---

## Programmatic API

```python
from src.core import create_repository
from src.analysis import AnalysisService
from src.simulation import SimulationService
from src.validation import ValidationService
from src.visualization import VisualizationService

repo       = create_repository()
analysis   = AnalysisService(repo)
simulation = SimulationService(repo)
validation = ValidationService(analysis, simulation, ndcg_k=10)

viz = VisualizationService(
    analysis_service=analysis,
    simulation_service=simulation,
    validation_service=validation,
    repository=repo,
)

output_path = viz.generate_dashboard(
    output_file="output/dashboard.html",
    layers=["app", "system"],
    include_network=True,      # set False for > 80 components
    include_matrix=True,
    include_validation=True,
    include_per_dim_scatter=True,   # R/M/A/V scatter plots
    antipatterns_report=None,       # pass SmellReport for Section 8
)

print(f"Dashboard: {output_path}")
repo.close()
```

See `examples/example_visualization.py` for a complete runnable example.

---

## What Comes Next

Step 6 completes the six-step methodology loop.

**For thesis defense:** The static dashboard is the primary research artifact. Use it to demonstrate the methodology end-to-end: topology input → prediction Q(v) → simulation I(v) → validation ρ = 0.876 → decision-ready output. The per-dimension scatter plots in Section 4 and the specialist metric sub-panel in Section 7 provide the evidence for each RMAV dimension's validity claim independently.

**For ICSA 2026 submission:** Generate dashboards for all eight validated scenarios and include them as supplementary material. The anti-pattern catalog (Section 8) and the Predictive Gain PG metric in the validation report are the two novel claims that distinguish this submission from the RASSE 2025 paper.

**For production deployment:** Use Genieus as the operational interface. The Dashboard tab provides the real-time health view; the Simulation tab enables pre-deployment what-if analysis; the `detect_antipatterns.py` CI/CD integration ensures that CRITICAL patterns block deployment automatically.

---

← [Step 5: Validation](validation.md) | [README](../README.md)