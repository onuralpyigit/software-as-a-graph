# Experimental Harness & Evaluation Suite

> **Per-experiment protocols, commands and artifacts for the JSS paper** are in
> [`docs/research/jss/experiments/`](../docs/research/jss/experiments/README.md). This page covers harness internals.

This document provides a technical deep-dive into the reproducibility infrastructure for the paper
**"Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in Publish–Subscribe Systems?"**
(Ibrahim Onuralp Yigit & Feza Buzluca, submitted to the *Journal of Systems and Software* Special Issue **VSI:AI4MSS** — see
`docs/research/jss/latex/`).

---

## 1. The Experimental Harness (`main_table.py`)

The primary harness orchestrates the **Attributable GNN Evaluation Ladder** over a **7 × 6 × 5
evaluation matrix** (7 scenarios, 6 variants, 5 seeds), totaling 210 cells (140 GNN training runs
plus 70 closed-form structural baseline computations) — matching the paper's own "210 evaluation
cells: 140 trained GNN models and 70 structural-baseline computations."

### A. Topology Refinement (`DEPENDS_ON` Edges)
Raw pub-sub graphs often exhibit "feature degeneracy" where Application nodes lack structural centrality because they only possess high-level logical connections. The harness implements custom edge derivation rules (Rule 1 & 5) before training:
- **Rule 1**: If Application $A$ publishes a topic $T$ consumed by $B$, add a `DEPENDS_ON` edge $B \to A$ — the subscriber depends on the publisher, the reverse of data flow (see the paper's §3.1 "Formal Definitions": dependent → dependency).
- **Rule 5**: If Application $A$ uses Library $L$, add a `DEPENDS_ON` edge $A \to L$ — the user depends on the library.
Structural metrics (betweenness, bridge ratio, etc.) are computed on this refined subgraph, ensuring a meaningful feature signal for the GNN. (§5.5 of the paper empirically validates this direction: inverting it flips the structural predictor's correlation with ground truth from ρ≈+0.84 to ρ≈−0.79.)

### B. Remapping & Normalization
- **Node ID Alignment**: Handles inconsistent naming across simulation logs (e.g., remapping `A1` to `A01` to match architectural JSONs).
- **RM Label Substitution (disabled by default)**: When failure-simulation labels are sparse (< 20% non-zero composite), the harness *used to* substitute **RM quality scores** as the training target. It no longer does: RM is computed from the same structural metrics that form the GNN's input features, so substituting it makes the labels a function of the features and invalidates every correlation metric. `_load_cache_dicts` now raises instead; `--allow-rm-substitution` opts back in and tags the affected results `RM-sub` rather than `Sim`. Sparse labels are a signal to fix the labeler, not to swap in a proxy.

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

### B. Identification Performance (Overlap@K / F1, Top-5 Overlap)
While Spearman measures ordering monotonicity, identification assesses operational critical-set detection:
- **Spearman ρ**: Global rank-order monotonicity against ground-truth cascade impact $I^*(v)$.
- **Active-Stratum Spearman ρ (ρ_{>0})**: Rank correlation restricted to components with non-zero true impact ($I^*(v) > 0$), isolating ranking quality from inertness detection.
- **Critical-Set Overlap@K (reported as F1@K)**: Set agreement on the top 20% critical services ($K = \text{round}(0.20 \cdot |V_{\text{app}}|)$). At equal set cardinality $K$, Precision, Recall, and $F_1$ are mathematically identical:
  $$\text{Overlap}@K = \frac{| \text{Top}_K(\text{Pred}) \cap \text{Top}_K(\text{Truth}) |}{K}$$
- **Top-5 Overlap**: Direct capture rate of the top 5 most catastrophic system components.
- **NDCG@10**: Normalized Discounted Cumulative Gain with logarithmic position discounting for the top 10 ranked entities.

### D. Statistical Rigor
- **Bootstrap 95% Confidence Intervals**: Computed using $B=2,000$ resamples for each mean Spearman ρ.
- **Paired Wilcoxon Signed-Rank Test**: A non-parametric test evaluated across the 12 LOSO folds ($n=12$ avoiding small-sample floor effects, with Holm step-down correction within registered families).
- **Confirmatory Contrast Outcome**: The registered primary contrast between $\text{HGT-QoS}$ and $\text{Topo-QoS}$ on raw multigraph $G_{\text{raw}}$ was **null** ($\Delta\rho = +0.069, p = 0.266$). On projected $G_{\text{dep}}$, closed-form references ($\text{Analytic } I^*$ at $\rho = \mathbf{0.808}$ and afferent coupling $\text{InDeg}$ at $\rho = \mathbf{0.764}$) match or outperform learned GNNs ($\text{GAT-P-QoS}$ at $0.748$ single seed, $0.772$ ensemble).

---

## 3. Model Variants & Closed-Form References

| Variant / Reference | Substrate | Nature / Role |
|---|---|---|
| `analytic_i_star` (`Analytic I*`) | Projected $G_{\text{dep}}$ | **Closed-Form Reference ($T_\infty(O)$)**: Exact transitive reachability expectation under uniform edge severity ($\rho = 0.808$ on $I^*$). |
| `in_degree` (`InDeg`) | Projected $G_{\text{dep}}$ | **Closed-Form Reference ($T_1(O)$)**: First-order direct subscriber count / afferent coupling $C_a$ ($\rho = 0.764$ on $I^*$). |
| `reach` (`Reach`) | Projected $G_{\text{dep}}$ | **Closed-Form Reference**: Transitive dependent count over $G_{\text{dep}}$ ($\rho = 0.732$ on $I^*$). |
| `rate_expansion` ($I_{\text{dyn}}^{(1)}$) | Declared pub/sub rates | **Closed-Form Reference**: Rate-Weighted First-Order Expansion Eq. 7 ($\rho = \mathbf{0.830}$ on $I_{\text{dyn}}$ in $<1\text{ ms}$). |
| `topo_baseline` (`Topo`) | Flow Projection | **Baseline**: Structural centrality (Betweenness) on unweighted projection ($\rho = 0.349$). |
| `topo_qos` (`Topo-QoS`) | Flow Projection | **Registered Comparator**: QoS-weighted betweenness ($\rho = 0.553$; corrected with articulation term: $0.533$). |
| `gat_p_qos` (`GAT-P-QoS`) | Projected $G_{\text{dep}}$ | **Learned Ranker**: Homogeneous GAT with QoS edge attributes on $G_{\text{dep}}$ ($\rho = 0.748$ single seed, $0.772$ ensemble). |
| `gin_p_qos` (`GIN-P-QoS`) | Projected $G_{\text{dep}}$ | **Learned Control**: Sum-aggregation GIN on $G_{\text{dep}}$ ($\rho = 0.716$). |
| `gl_qos` (`GAT-QoS`) | Native Multigraph $G_{\text{raw}}$ | **Learned Control**: Homogeneous GAT on raw multigraph ($\rho = 0.635$). |
| `hgl_qos` (`HGT-QoS`) | Native Multigraph $G_{\text{raw}}$ | **Primary Confirmatory Learner**: QoS-aware Heterogeneous Graph Transformer ($\rho = 0.622$; contrast vs. baseline is null, $p = 0.266$). |
| `hybrid_hgt` (`SaG-Hybrid`) | $G_{\text{raw}}$ + Prior | **Hybrid Ranker**: Residual correction over rank-normalized `Topo-QoS` ($\rho = 0.657$). |
| `hybrid_gat` (`SaG-Hybrid-GAT`) | $G_{\text{raw}}$ + Prior | **Hybrid Ranker**: Residual correction over rank-normalized `Topo-QoS` ($\rho = 0.683$). |

---

## 4. Reproducing the Tables (JSS Tables 6, 7, 8, 9, 10)

To reproduce the main LOSO benchmark on $I^*$ (JSS Table 6) and the three-oracle comparison (JSS Table 7):

```bash
# Run the inductive LOSO cross-validation sweep
python reproduce/main_table.py --epochs 300 --seeds 42 123 456 789 2024
make -f reproduce/Makefile table4

# Run zero-shot transfer to 5 open-source systems (JSS Table 9)
python reproduce/realworld_zeroshot.py

# Run scalability sweep (JSS Table 10)
python reproduce/main_table.py --scale-sweep
```


---

## 5. Oracle Agreement (JSS Table 13, `convergent_validity.py`)

The harness above scores predictors against $I^*(v)$. The project has **three** simulation oracles,
and this script measures how far they agree — a construct-validity check, not a predictor
evaluation. It never scores $Q(v)$ or a trained model.

| Oracle | Engine | Quantity |
|---|---|---|
| $I^*(v)$ | `FaultInjector` | Mean subscriber feed-loss fraction under a BFS cascade |
| $I_{\text{comp}}(v)$ | `FailureSimulator` | Weighted composite of reachability, fragmentation, throughput, and flow terms |
| $I_{\text{dyn}}(v)$ | `MessageFlowSimulator` | Drop in delivered message rate that *surviving* consumers experience, by SimPy discrete-event simulation of actual traffic |

For each unordered pair the script reports Spearman ρ (with $p$), Kendall τ, and top-20% Jaccard
**over the node set the two oracles share** — the three differ in coverage, so `n_common` is
reported per pair and is not the scenario size. Scales differ, so only rank agreement is meaningful.

$I^*$ and $I_{\text{comp}}$ are both topological cascade engines over the same substrate, so their
agreement cannot rule out a shared construction artifact. $I_{\text{dyn}}$ is the one that can:
it reaches the same ranking by simulating traffic rather than by traversing edges. It is
delivery-based and QoS-agnostic on this corpus, produces no training labels, and gates nothing.


```bash
make -f reproduce/Makefile convergent-validity
# or, bounding the expensive oracle on large scenarios:
python reproduce/convergent_validity.py --max-candidates 100
```

$I_{\text{dyn}}$ costs one discrete-event run per candidate component, so runtime scales with
corpus size rather than with epochs; `enterprise_system` dominates. `--skip-message-flow` falls
back to the two topological oracles. Output is `results/convergent_validity.json`.

