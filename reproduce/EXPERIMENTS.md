# Experimental Harness & Evaluation Suite

This document provides a technical deep-dive into the reproducibility infrastructure for the Middleware 2026 paper: **"QoS-Aware Heterogeneous Graph Learning for Architectural Criticality Prediction"**.

---

## 1. The Experimental Harness (`middleware26_main_table.py`)

The primary harness orchestrates an **8 × 4 × 5 training matrix** (8 scenarios, 4 model variants, 5 seeds), totaling 160 independent training and evaluation cycles.

### A. Topology Refinement (`DEPENDS_ON` Edges)
Raw pub-sub graphs often exhibit "feature degeneracy" where Application nodes lack structural centrality because they only possess high-level logical connections. The harness implements a custom edge derivation rule (Rule 1 & 5) before training:
- **Rule 1**: If Application $A$ publishes a topic $T$ consumed by $B$, add a `DEPENDS_ON` edge $A \to B$.
- **Rule 5**: If Application $A$ uses Library $L$, add a `DEPENDS_ON` edge $A \to L$.
Structural metrics (betweenness, bridge ratio, etc.) are computed on this refined subgraph, ensuring a meaningful feature signal for the GNN.

### B. Remapping & Normalization
- **Node ID Alignment**: Handles inconsistent naming across simulation logs (e.g., remapping `A1` to `A01` to match architectural JSONs).
- **RMAV Label Substitution**: In scenarios where failure simulation results are extremely sparse (density < 20%), the harness automatically substitutes ground-truth labels with **RMAV quality scores**. This provides a consistent training target derived from the same `DEPENDS_ON` features, resolving "cold-start" training issues.

### C. Resilience & Resumption
- **Incremental Saving**: Results are saved to `results/main_table.json` after every single cell (seed-variant-scenario) completion.
- **Resume Support**: Using the `--resume` flag allows the harness to skip already-calculated cells, making it resilient to hardware interruptions or timeouts in CPU-only environments.

---

## 2. The Evaluation Suite

The evaluation suite (implemented in `saag/prediction/trainer.py` and aggregated in the harness) uses a multi-dimensional metric battery to validate the predictions.

### A. Ranking Performance (Spearman ρ)
The primary metric is the **Spearman Rank Correlation Coefficient (ρ)**.
- It measures the monotonic relationship between the predicted criticality $Q^*(v)$ and the ground-truth impact $I^*(v)$.
- A high ρ indicates that the system correctly identifies the relative priority of components for architectural hardening.

### B. Identification Performance (F1, Precision, Recall)
While Spearman measures ordering, identification metrics measure the quality of the binary classification of "Critical" vs. "Safe" nodes at a threshold of 0.5.
- **Recall**: Critical for safety-critical systems; measures the fraction of truly critical components that were correctly identified.
- **Precision**: Measures the "False Top Rate"; ensures that architect resources aren't wasted on components that are actually safe.
- **F1 Score**: The harmonic mean, used as the primary gate for "Identification Quality" (Target ≥ 0.70).

### C. Top-K Overlap (Top-5, Top-10)
Measures the intersection between the top $K$ most critical components in the ground truth vs. the top $K$ in the predictions.
- $\text{Overlap}@K = \frac{| \text{Top}_K(\text{Pred}) \cap \text{Top}_K(\text{Truth}) |}{K}$
- This metric is particularly useful for manual architectural reviews where only a handful of components can be refactored at a time.

### D. Statistical Rigor
- **Bootstrap 95% Confidence Intervals**: Computed using $B=2,000$ resamples for each mean Spearman ρ.
- **Paired Wilcoxon Signed-Rank Test**: A non-parametric test used to prove that **Q-HGL** is statistically superior to the baselines (`RMAV`, `Homo-S`, `Homo-U`) across different seeds and scenarios ($p < 0.05$).

---

## 3. Model Variants

| Variant | Logic |
|---|---|
| `hetero_qos` | **Q-HGL (Proposed)**: Heterogeneous GAT with 16-dimensional edge features and node-type-specific attention heads. |
| `homo_scalar` | **Baseline**: Homogeneous GAT that reduces QoS metadata to a single scalar weight per edge. |
| `homo_unweighted` | **Baseline**: Homogeneous GAT that ignores QoS metadata entirely (pure topology). |
| `topology_rmav` | **Baseline**: Non-learning centrality-based heuristic (RMAV). |

---

## 4. Reproducing the Table

To reproduce the full Table 3 with all identification metrics:

```bash
# Run the harness
python tools/middleware26_main_table.py --epochs 300 --seeds 42 123 456 789 2024

# Render the report
python tools/render_table.py --table3 results/main_table.json
```

The resulting `results/table3_id_metrics.md` will contain the F1, Precision, Recall, and Top-K breakdown for each scenario.
