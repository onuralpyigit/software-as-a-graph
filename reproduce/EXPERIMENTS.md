# Experimental Harness & Evaluation Suite

This document provides a technical deep-dive into the reproducibility infrastructure for the paper
**"Heterogeneous Graph Learning for Cascade Impact Prediction in Distributed Publish-Subscribe
Middleware"** (originally submitted to Middleware 2026; now extended for the JSS special issue
VSI:AI4MSS — see `docs/research/jss/si_middleware_extension.md`).

---

## 1. The Experimental Harness (`middleware26_main_table.py`)

The primary harness orchestrates the **Attributable GNN Evaluation Ladder** over a **7 × 6 × 5
evaluation matrix** (7 scenarios, 6 variants, 5 seeds), totaling 210 cells (140 GNN training runs
plus 70 closed-form structural baseline computations) — matching the paper's own "210 evaluation
cells: 140 trained GNN models and 70 structural-baseline computations."

### A. Topology Refinement (`DEPENDS_ON` Edges)
Raw pub-sub graphs often exhibit "feature degeneracy" where Application nodes lack structural centrality because they only possess high-level logical connections. The harness implements a custom edge derivation rule (Rule 1 & 5) before training:
- **Rule 1**: If Application $A$ publishes a topic $T$ consumed by $B$, add a `DEPENDS_ON` edge $B \to A$ — the subscriber depends on the publisher, the reverse of data flow (see the paper's §3.1 "Formal Definitions": dependent → dependency).
- **Rule 5**: If Application $A$ uses Library $L$, add a `DEPENDS_ON` edge $A \to L$ — the user depends on the library.
Structural metrics (betweenness, bridge ratio, etc.) are computed on this refined subgraph, ensuring a meaningful feature signal for the GNN. (§5.5 of the paper empirically validates this direction: inverting it flips the structural predictor's correlation with ground truth from ρ≈+0.84 to ρ≈−0.79.)

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
While Spearman measures ordering, identification:
- **Spearman ρ**: Measures global ranking quality.
- **Accuracy Score**: Overall fraction of correct predictions (Threshold = 0.5).
- **Precision / Recall / F1**: Binary classification quality metrics.
- **Top-5/10 Overlap**: Top-K identification quality.
- **NDCG@10**: Normalized Discounted Cumulative Gain for ranking stability.

### C. Top-K Overlap (Top-5, Top-10)
Measures the intersection between the top $K$ most critical components in the ground truth vs. the top $K$ in the predictions.
- $\text{Overlap}@K = \frac{| \text{Top}_K(\text{Pred}) \cap \text{Top}_K(\text{Truth}) |}{K}$
- This metric is particularly useful for manual architectural reviews where only a handful of components can be refactored at a time.

### D. Statistical Rigor
- **Bootstrap 95% Confidence Intervals**: Computed using $B=2,000$ resamples for each mean Spearman ρ.
- **Paired Wilcoxon Signed-Rank Test**: A non-parametric test used to prove that **HGL-QoS** is statistically superior to the baselines (`Topo-BL`, `Topo-QoS`, `GL`, `GL-QoS`, `HGL`) across different seeds and scenarios ($p < 0.05$).

---

## 3. Model Variants

| Variant | Logic |
|---|---|
| `topo_baseline` (`Topo-BL`) | **Baseline**: Structural centrality (Betweenness + Articulation Point) on unweighted `DEPENDS_ON` projection. |
| `topo_qos` (`Topo-QoS`) | **Baseline**: Structural centrality weighted by local QoS edge features on `DEPENDS_ON` projection. |
| `gl` (`GL`) | **Baseline**: Homogeneous GAT on unweighted `DEPENDS_ON` projection. |
| `gl_qos` (`GL-QoS`) | **Baseline**: Homogeneous GAT over QoS-weighted `DEPENDS_ON` projection (edge weight = QoS-derived weight). |
| `hgl` (`HGL`) | **Baseline/Ablation**: Heterogeneous GAT over native pub-sub graph substrate with QoS attributes masked (isolates heterogeneous structure). |
| `hgl_qos` (`HGL-QoS`) | **Proposed Variant**: QoS-aware Heterogeneous GAT over native pub-sub graph substrate. |

---

## 4. Reproducing the Table

To reproduce the full Table 3 with all identification metrics:

```bash
# Run the harness
python reproduce/middleware26_main_table.py --epochs 300 --seeds 42 123 456 789 2024

# Render the report
python reproduce/render_table.py --table3 results/main_table.json
```

The resulting `results/table3_id_metrics.md` will contain the F1, Precision, Recall, and Top-K breakdown for each scenario.
