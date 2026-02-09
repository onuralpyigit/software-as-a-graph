# Step 6: Visualization

**Turn analysis results into interactive dashboards for architectural decision-making.**

← [Step 5: Validation](validation.md) | [README](../README.md)

---

## What This Step Does

Visualization is the final step. It takes the outputs of all preceding steps — metrics, quality scores, impact scores, and validation results — and synthesizes them into an interactive HTML dashboard. The goal is to move from numbers to decisions.

```
Steps 2–5 Outputs          Visualization          Interactive Dashboard
(metrics, scores,     →    Pipeline        →     (KPIs, graphs, tables,
 impact, validation)                               network explorer)
```

## Dashboard Sections

### 1. Executive Overview

High-level KPIs at a glance: total components, edges, number of critical/high-risk components, SPOFs detected, and anti-patterns found. Color-coded indicators highlight areas needing attention.

### 2. Layer Comparison

Side-by-side comparison of analysis results across architectural layers (app, infra, mw, system). This reveals where risk is concentrated — for example, whether application-level or infrastructure-level components pose greater systemic risk.

### 3. Component Details Table

A sortable, filterable table showing every component with its RMAV scores, overall quality score Q(v), impact score I(v), and criticality classification. This is the primary tool for component-level prioritization.

### 4. Correlation Scatter Plot

Plots predicted Q(v) against actual I(v) for every component. Points near the diagonal indicate accurate predictions. Outliers reveal where the model over- or under-predicts, helping identify components with hidden criticality or unexpected resilience.

### 5. Interactive Network Graph

A force-directed graph visualization of the dependency network. Node size reflects quality score, node color reflects component type or criticality level, and edge thickness reflects dependency weight. You can hover for details, click to highlight neighborhoods, and drag to reposition nodes.

| Action | Effect |
|--------|--------|
| Hover | Show component details (type, Q(v), I(v), RMAV) |
| Click | Highlight direct dependencies |
| Drag | Reposition nodes |
| Scroll | Zoom in/out |
| Double-click | Center on selected node |

### 6. Dependency Matrix

A heatmap showing component-to-component dependency weights. Sorted by criticality score, dense blocks along the diagonal reveal tightly coupled clusters. Full rows/columns indicate hub components. This view scales better than the network graph for dense systems.

### 7. Validation Report

Displays all validation metrics with pass/fail indicators. Includes the Spearman ρ value, F1-score, precision, recall, and other metrics from Step 5, along with confidence intervals and statistical significance.

## Commands

```bash
# Generate dashboard for a specific layer
python bin/visualize_graph.py --layer system --output dashboard.html

# Generate and open in browser
python bin/visualize_graph.py --layer system --output dashboard.html --open

# Generate a demo dashboard (no Neo4j needed)
python bin/visualize_graph.py --demo --output demo_dashboard.html --open
```

## From Analysis to Decisions

The dashboard is designed to support specific architectural decisions:

| Question | Where to Look |
|----------|--------------|
| Which components need redundancy? | Component table filtered by CRITICAL + high A(v) |
| Where should we focus refactoring? | Components with high M(v) (maintainability risk) |
| What's our overall system risk posture? | Executive overview KPIs |
| Are there hidden coupling clusters? | Dependency matrix diagonal blocks |
| Can we trust these predictions? | Validation report pass/fail status |
| Which layer needs the most attention? | Layer comparison section |

## Summary

The visualization dashboard completes the six-step methodology by translating quantitative analysis into actionable insights. Together, the six steps form a pipeline from raw architecture to informed decisions:

1. **Graph Model** — capture the architecture
2. **Structural Analysis** — measure component importance
3. **Quality Scoring** — interpret importance as quality risk
4. **Failure Simulation** — establish ground truth
5. **Validation** — prove predictions work
6. **Visualization** — make it actionable

---

← [Step 5: Validation](validation.md) | [README](../README.md)