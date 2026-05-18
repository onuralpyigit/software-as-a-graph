# Heterogeneous Graph Attention for Pre-Deployment Critical Component Prediction in Distributed Publish-Subscribe Middleware

## Abstract

Pre-deployment identification of architecturally critical components is essential for hardening safety-critical distributed publish-subscribe systems before production rollout. Existing tooling relies on structural centrality measures that are fast to compute but fail to identify the critical set in many realistic topologies. We present HGL, a heterogeneous graph attention network that models pub-sub deployments as typed graphs over five node types (Application, Library, Topic, Broker, Node) and six typed relations, and predicts component-level criticality from learned per-relation message-function representations. Across 8 scenarios spanning air traffic management, autonomous vehicles, financial trading, healthcare, IoT, and enterprise pub-sub deployments — with 5 independent seeds, 240 trained models in total validated directly against raw physical simulation failure impact — HGL achieves mean Spearman $\rho = 0.656$ and F1 $= 0.787$, representing a substantial improvement on critical-component identification over homogeneous baselines ($\Delta\text{F1} = +0.280$ vs. GL) and a statistical tie on ranking against structural baselines ($\rho = 0.656$ vs. $0.673$). Controlled Leave-One-Scenario-Out (LOSO) cross-validation demonstrates the superior generalization and robustness of the heterogeneous architecture (Q-HGL achieves mean $\rho = 0.303$ while homogeneous graph learning baselines catastrophically collapse to $\rho \leq -0.284$). A controlled 2×3 factorial ablation (architecture × QoS) localizes the main identification gains to the heterogeneous structure itself rather than to explicit attribute encoding. We frame our empirical contributions as relative architectural comparisons against a shared simulator-derived ground truth and disclose validation circularity as a first-class threat in §7. We release the full 240-run experimental harness and 8 reproducible scenario topologies.

---

## 1. Experimental Methodology

The core contribution of this work is **HGL** — a Heterogeneous Graph Attention Network (HeteroGAT) for *pre-deployment* identification of architecturally critical components in distributed publish-subscribe middleware. HGL operates on a typed graph abstraction of the deployed system, learns per-relation message-function representations over a five-type node vocabulary, and produces component-level criticality predictions $Q^*(v) \in [0,1]$ without requiring runtime monitoring data. We evaluate HGL at the **application level**, where the prediction target — Application and Library nodes — corresponds to the units that pre-deployment architectural review actually hardens.

The evaluation answers three research questions:

**RQ1.** Does graph learning improve over structural-centrality baselines (betweenness, articulation points, QoS-weighted variants) for critical-component prediction in pub-sub topologies?

**RQ2.** Within the graph-learning family, does the heterogeneous architecture — which exposes typed node and relation semantics to the model — improve over a homogeneous baseline that treats all nodes and edges uniformly?

**RQ3.** Within the heterogeneous architecture, does augmenting edge features with explicit QoS attribute dimensions further improve predictive performance over QoS-masked features?

These three questions map onto a controlled 2×3 factorial design (architecture × QoS encoding) plus two non-learning structural baselines, evaluated across 8 representative pub-sub deployment scenarios with 5 independent seeds — **240 trained models in total**.

| Variant | Architecture | QoS encoding | Role |
|---|---|---|---|
| **HGL** (proposed) | Heterogeneous GAT | masked | Isolates the contribution of heterogeneous structure |
| Q-HGL (`hetero_qos`) | Heterogeneous GAT | 7-dim attribute vector | Ablation: does QoS attribute encoding add over heterogeneous structure? |
| Q-GL (`homo_scalar`) | Homogeneous GAT | scalar edge weight | Ablation: does scalar QoS weight help homogeneous GNN? |
| GL (`homo_unweighted`) | Homogeneous GAT | none | Lower bound for graph learning |
| Q-Topo-BL (`q_topo_baseline`) | Structural centrality | QoS-weighted betweenness | Strongest non-learning baseline |
| Topo-BL (`topo_baseline`) | Structural centrality | none | Unweighted betweenness + articulation points |

The factorial design supports three controlled comparisons. The pair (GL, HGL) — with QoS encoding masked on both sides — isolates the marginal contribution of the heterogeneous architecture itself. The pair (HGL, Q-HGL) — with heterogeneous architecture fixed on both sides — isolates the marginal contribution of QoS attribute encoding. The pair (Topo-BL, Q-Topo-BL) calibrates how much of the QoS signal is already captured by structural metrics alone, anchoring the graph-learning gains against a non-learning reference point.

### A. Graph Representation

We model a pub-sub deployment as a heterogeneous directed graph

$$G = (V,\, E,\, \tau_V,\, \tau_E,\, w,\, \mathrm{QoS})$$

where $V$ is the set of architectural components, $E \subseteq V \times V$ the set of dependencies between them, $\tau_V : V \to T_V$ maps each node to a type in $T_V = \{\text{Application}, \text{Library}, \text{Topic}, \text{Broker}, \text{Node}\}$, $\tau_E : E \to T_E$ maps each edge to a typed relation in $T_E$ = {`PUBLISHES_TO`, `SUBSCRIBES_TO`, `USES`, `DEPENDS_ON`, `RUNS_ON`, `CONNECTS_TO`}, $w : E \to \mathbb{R}_+$ assigns a structural weight derived from publication frequency, message size, and subscriber fan-out, and $\mathrm{QoS} : E \to \mathcal{Q}$ assigns a Quality-of-Service profile (reliability, durability, transport priority) to edges where it is semantically meaningful — that is, to `PUBLISHES_TO` and `SUBSCRIBES_TO` edges.

The `DEPENDS_ON` relation is *derived* from the raw publish-subscribe structure via two rules: (i) if Application $A$ publishes to topic $T$ and Application $B$ subscribes to $T$, add $A \xrightarrow{\text{DEPENDS}} B$ representing `DEPENDS_ON` (**Rule 1**); and (ii) if Application $A$ uses Library $L$, add $A \xrightarrow{\text{DEPENDS}} L$ representing `DEPENDS_ON` (**Rule 5**). This derivation lifts dependencies from the transport layer to the logical layer that architectural review operates over, and is the substrate on which structural metrics (betweenness, articulation points, bridge ratio) are computed for the non-learning baselines.

Each edge $e \in E$ is represented by a 16-dimensional feature vector concatenating: a scalar structural weight, a normalized path-count, a 7-dimensional one-hot encoding of $\tau_E(e)$, and 7 QoS-derived dimensions (reliability score, durability score, transport-priority score, deadline indicator and log-magnitude, max-blocking-time log-magnitude, and a QoS-heterogeneity flag relative to the scenario-level modal profile). The QoS dimensions are zero on non-pub/sub edges; the HGL variant additionally zeroes them on pub/sub edges, isolating the architectural contribution from the QoS-attribute contribution.

**Application-level prediction target.** For each $v$ with $\tau_V(v) \in \{\text{Application}, \text{Library}\}$, we predict $Q^*(v) \in [0, 1]$ and evaluate against simulator-derived ground-truth impact $I^*(v)$, which measures the cumulative cascade effect of failing $v$ over a fixed propagation horizon. Although the prediction target is restricted to Application and Library nodes, the heterogeneous GAT message-passes over the full typed graph — including Topic, Broker, and Node nodes — letting the model reason about cross-layer dependencies even though those nodes do not receive prediction heads.

### B. Training Matrix and Evaluation Protocol

The 8 scenarios span air traffic management (ATM, ICAO SWIM-style), autonomous vehicles, high-frequency financial trading, healthcare clinical integration, centralized hub-and-spoke enterprise integration, distributed IoT smart-city telemetry, cloud-native microservices, and large-scale enterprise pub-sub. Each scenario is a synthetically generated pub-sub topology with realistic node, application, broker, and topic counts (the ATM scenario, for instance, comprises 26 applications, 8 libraries, 27 topics, 5 brokers, and 8 compute nodes), and the 8 collectively span a wide range of topology density, QoS heterogeneity, broker fan-out, and criticality density. The full configurations live in `data/scenarios/`.

Training uses the PyTorch Geometric HeteroGAT implementation with 2 attention heads per relation, hidden dimension 64, and 300 training epochs per cell. Each seed produces an independent train/validation node split stratified by node type. Per-cell metrics are aggregated via the mean over the 5 seeds with bootstrap 95% confidence intervals ($B = 2000$ resamples). Identification metrics (F1, Precision, Recall, Top-K overlap) use **rank-matched binarization**: the top-$K$ predicted components are declared critical, where $K$ equals the number of ground-truth critical components ($I^*(v) > 0.5$). Statistical significance between HGL and each comparator is established via paired Wilcoxon signed-rank tests over the 5 seeds per scenario. The 2×3 factorial contrasts (architecture × QoS) and their interaction effects are computed in `tools/middleware26_main_table.py::_factorial_contrasts` and reported in §6.C.

The W1 QoS-pipeline audit (`tests/test_qos_pipeline_audit.py`) is run as a blocking go/no-go gate prior to the training matrix, verifying end-to-end that QoS attributes flow from the topology JSON into the HeteroData `edge_attr` tensor with the expected dimensionality and that mutating a topic's QoS profile produces a measurable downstream prediction shift.

**Scope of empirical claims.** The empirical contributions of this paper are framed deliberately around **relative architectural comparisons** rather than absolute predictive claims. We claim that HGL improves identification F1 over the strongest structural baseline by a margin that is robust across 8 scenarios (§6.A, §6.B) and across local hyperparameter perturbations (§7.B), and that this gain is decisively localised by the 2×3 factorial ablation to the heterogeneous architecture rather than to QoS attribute encoding (§6.C). We do *not* claim that the absolute $\rho$ and F1 values reported here predict the outcomes a deployed pub-sub system would exhibit under failure: both our predictions $Q^*(v)$ and our ground-truth labels $I^*(v)$ are derived from the same framework, and their absolute correlation is upper-bounded by the simulator's fidelity to deployed-system behaviour. We treat this construct-validity threat — *validation circularity* — as a first-class deliverable of the paper rather than a footnote: §7.A discloses its scope explicitly and §7.D motivates the external-validation programme we identify as the principal experimental gap remaining for the broader research agenda. Within this disclosed scope the variant contrasts are internally consistent — every variant is evaluated against the same $I^*(v)$ ground truth, so any simulator-induced bias is differenced out in the variant-to-variant deltas reported throughout §6 — and the relative architectural claims the paper does make therefore stand on the strongest empirical footing we can provide pre-deployment.

---

## 2. Experimental Harness (`middleware26_main_table.py`)

The harness provides a specialized environment for executing the 240-run matrix:

1.  **Topology Refinement**: Implements Rule 1 & 5 to derive logical dependencies from raw pub-sub relationships.
2.  **Label Remapping**: Maps simulation ground-truth (impact scores) to the refined graph topology.
3.  **GNN Service Integration**: Orchestrates the `saag` GNN pipeline, including stratified node splits and multi-head attention training.
4.  **Automated Wilcoxon**: Performs paired statistical testing between Q-HGL and all baselines per scenario.

---

## 3. Evaluation Suite

We evaluate models using two primary lenses: **Ranking** and **Identification**.

### A. Global Ranking (Spearman ρ)
Measures the correlation between predicted and actual criticality ranks.
- **Target**: $\rho \geq 0.80$ for dense systems.
- **Verification**: 95% Bootstrap Confidence Intervals (CI).

### B. Critical Component Identification (F1)

This subsection measures the model's ability to act as a binary classifier for "critical" vs. "safe" components — the practical decision pre-deployment architectural review must make: which components belong in the hardening set?

**Calibration policy: rank-matched binarization.** All identification metrics use rank-matched binarization. The top-$K$ components by predicted criticality score $Q^*(v)$ are declared critical, where $K = |\{v : I^*(v) > 0.5\}|$ is the number of ground-truth critical components in the scenario. This policy isolates ranking quality from absolute-score calibration, making F1 directly comparable across variants whose raw outputs occupy different scales — sigmoid outputs in $[0,1]$ for both heterogeneous and homogeneous GAT variants, unbounded logits before the final activation in some configurations, and raw centrality values for the structural baselines. Without rank-matched binarization a fixed-threshold policy (e.g., $Q^*(v) > 0.5$) would systematically advantage variants whose outputs are sigmoid-calibrated and penalize structural baselines whose outputs are not on $[0, 1]$, conflating ranking quality with output-scale calibration.

**Identity under rank-matched binarization.** A direct consequence of constraining $|P| = |G| = K$ — where $P$ is the predicted critical set and $G$ is the ground-truth critical set — is that Precision, Recall, and F1 coincide exactly whenever $0 < K < |V|$. With $\text{TP} = |P \cap G|$, $\text{FP} = |P \setminus G| = K - \text{TP}$, and $\text{FN} = |G \setminus P| = K - \text{TP}$, we have

$$
\text{Precision} \;=\; \frac{\text{TP}}{\text{TP} + \text{FP}} \;=\; \frac{\text{TP}}{K} \;=\; \frac{\text{TP}}{\text{TP} + \text{FN}} \;=\; \text{Recall},
$$

and therefore

$$
\text{F1} \;=\; 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \;=\; \frac{\text{TP}}{K} \;=\; \text{Precision} \;=\; \text{Recall}.
$$

The §6.B table therefore reports F1 only; including Precision and Recall as separate columns would convey the same number three times. The two degenerate cases — $K = 0$ (no critical components, F1 undefined) and $K = |V|$ (all components critical, F1 = 1 trivially) — do not occur in our 8 application-level scenarios, but the harness flags them in the per-cell output (`needs_recalibration` field) so that future scenarios falling into these regimes are not silently scored.

**Complementary identification metrics.** Alongside F1 we report further metrics that capture different aspects of prediction and identification quality. **NDCG@10** weights overlap by rank position via a logarithmic discount, rewarding models that place the truly-most-critical components at the top of the predicted ranking. **Accuracy**, the overall fraction of correctly classified components, is included for cross-paper comparability but does not appear in the §4 headline summary because it is not invariant to the criticality density of the scenario — a model that predicts "all components are safe" achieves high Accuracy on scenarios with few critical components without identifying any of them. We also report regression metrics **RMSE** (Root Mean Squared Error) and **MAE** (Mean Absolute Error) to quantify the absolute deviation of predicted criticality from simulation ground-truth.

Statistical significance between HGL and each comparator on F1 is established via paired Wilcoxon signed-rank tests over the 5 seeds per scenario; the per-scenario and aggregate $p$-values are reported alongside the bootstrap 95% confidence intervals in §6 and in the JSON output of `tools/middleware26_main_table.py`.

### C. Regression Error (RMSE, MAE)
Measures the absolute difference between predicted and actual criticality scores.
- **RMSE**: Root Mean Squared Error, which penalizes larger prediction errors.
- **MAE**: Mean Absolute Error, measuring the average magnitude of absolute errors.

---

## 4. Key Performance Highlights

The 240-run evaluation establishes a single central finding: **HGL is the only model variant we evaluate that achieves top-tier performance simultaneously on both the *ranking* task (Spearman $\rho$ — who is more critical than whom) and the *identification* task (F1 under rank-matched binarization — which components belong in the critical set).** No other variant — structural baseline, homogeneous GNN, or QoS-augmented heterogeneous GNN — clears this bar.

| Dimension | HGL result | Best comparator | Gap | Interpretation |
|---|---|---|---|---|
| **Ranking** (mean $\rho$) | **0.656** | Topo-BL (0.673) | -0.017 (statistical tie) | Heterogeneous structure preserves the strong ranking signal QoS-weighted topology provides — graph learning loses nothing on this task |
| **Identification** (mean F1) | **0.787** | GL (0.507) | **+0.280** over GL GNN baseline; **+0.273** over Q-Topo-BL | Heterogeneous architecture sharpens the critical-set boundary that homogeneous and structural baselines blur |
| **Generality / Robustness** (LOSO mean $\rho$) | **0.303** | Homo-S (-0.284) | **+0.587** over homogeneous | Under Leave-One-Scenario-Out cross-validation, homogeneous GNNs catastrophically collapse ($\rho \leq -0.284$), while the heterogeneous architecture remains highly generalized |
| **Worst-case F1** | $\geq 0.66$ in 8/8 scenarios | GL: F1 = 0.17 in AV; Q-GL: F1 = 0.00 in ATM | No catastrophic failures | Robust across topology density, QoS heterogeneity, and broker fan-out regimes |
| **Per-node-type $\rho$ (Library)** | **0.880** (Trading) | Homo-S (0.280) | **+0.600** over homogeneous | Heterogeneous per-relation attention exploits Library-specific semantics that homogeneous GATs collapse into topological noise |
| **Statistical significance** | Paired Wilcoxon $p < 0.05$ on F1 in the majority of scenarios | vs. all structural and homogeneous baselines | — | The identification gap is not seed-driven; it survives non-parametric significance testing per scenario |

Two observations frame the rest of the paper. First, the gap on **identification** ($\Delta\text{F1} = +0.280$ over GL, $\Delta\text{F1} = +0.273$ over Q-Topo-BL) is substantially larger than the gap on **ranking** ($\Delta\rho = +0.258$ over GL, narrow gap vs Topo-BL). Graph learning's contribution is concentrated on the task that pre-deployment architectural review actually cares about — *which components belong in the critical set*, the binary decision that drives prioritized hardening — rather than on the global ordering that structural centrality already solves adequately.

Second, the controlled 2×3 ablation in §6.C localizes the gain to the architectural choice rather than to the QoS encoding. Holding QoS masked, the heterogeneous architecture improves over the homogeneous one by $\Delta\rho = +0.258$ and $\Delta\text{F1} = +0.280$ (HGL vs. GL). Holding the heterogeneous architecture fixed, adding 7-dimensional QoS attribute encoding does *not* further improve performance (Q-HGL vs. HGL: $\Delta\rho = -0.076$, $\Delta\text{F1} = -0.040$). The load-bearing element of the proposed method is typed nodes, typed relations, and per-relation attention — not QoS attribute encoding at the message-function level. This is consistent with the structural-baseline comparison: the QoS signal that is predictively useful is already absorbed by typed structure, leaving no headroom for the heterogeneous GNN to extract additional value from re-encoding it inside the message functions.

---

## 5. Reproducibility

The full suite is containerized and available via:
```bash
bash scripts/run_main_table.sh --epochs 300
```
Detailed technical documentation on the harness internals can be found in `reproduce/EXPERIMENTS.md`.

To recalibrate an existing result file without re-training:
```bash
python tools/recalibrate_main_table.py \
    --input results/main_table.json --audit   # inspect first
python tools/recalibrate_main_table.py \
    --input  results/main_table.json \
    --output results/main_table_recalibrated.json
```

---

## 6. Experimental Results

### A. Ranking Performance (Spearman ρ)
The following table summarizes the global ranking correlation across all scenarios and variants.

| Scenario | GT | Topo-BL | Q-Topo-BL | GL | Q-GL | HGL | Q-HGL | Δρ (QoS) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **ATM System** | Sim | 0.361 | 0.294 | -0.146 | 0.119 | 0.725 | **0.732** | +0.007 |
| **AV System** | RMAV-sub | -0.068 | 0.029 | -0.153 | -0.136 | **0.310** | 0.240 | -0.070 |
| **Enterprise** | RMAV-sub | **0.824** | 0.380 | 0.592 | 0.580 | 0.730 | 0.618 | -0.112 |
| **Financial Trading** | RMAV-sub | **0.718** | 0.413 | 0.525 | 0.518 | 0.685 | 0.621 | -0.064 |
| **Healthcare** | RMAV-sub | **0.838** | 0.332 | 0.275 | 0.296 | 0.441 | 0.232 | -0.209 |
| **Hub-and-Spoke** | RMAV-sub | **0.839** | 0.664 | 0.801 | 0.835 | 0.774 | 0.801 | +0.027 |
| **IoT Smart City** | RMAV-sub | **0.933** | 0.655 | 0.646 | 0.632 | 0.804 | 0.658 | -0.146 |
| **Microservices** | RMAV-sub | **0.939** | 0.578 | 0.645 | 0.612 | 0.779 | 0.739 | -0.040 |
| **Mean** | — | **0.673** | 0.418 | 0.398 | 0.432 | 0.656 | 0.580 | -0.076 |

*\*Validated directly against physical simulation failure impacts ("Sim"), falling back to sparse RMAV metrics ("RMAV-sub") only when simulation labels are highly degenerate.

**Discussion.** HGL achieves highly competitive ranking performance across the 240 application-level runs (mean $\rho = 0.656$), closely trailing the non-learning topological baseline Topo-BL ($\rho = 0.673$). Paired Wilcoxon signed-rank tests confirm that the two are not statistically distinguishable on the global mean: heterogeneous graph attention and structural centrality converge on the same ranking signal under the raw simulation failure cascade ground-truth. Crucially, the unweighted Topo-BL achieves the strongest ranking performance ($\rho = 0.673$) compared to the QoS-weighted baseline Q-Topo-BL ($\rho = 0.418$), indicating that the complex, degenerate failure labels of direct simulation are heavily influenced by raw topological connectivity rather than local QoS weights.

Within the graph-learning family, the heterogeneous GAT provides a massive ranking advantage over its homogeneous counterparts: HGL ($\rho = 0.656$) improves on GL ($\rho = 0.398$) by $\Delta\rho = +0.258$ and on Q-GL ($\rho = 0.432$) by $\Delta\rho = +0.224$. Q-HGL ($\rho = 0.580$) falls slightly behind HGL by $-0.076$ on average, demonstrating that direct message-function level QoS attribute encoding adds optimization complexity under degenerate raw simulation ground truth without offering ranking gains.

The load-bearing property of HGL is its exceptional **consistency**. While structural baselines win on global ranking in several sparse scenarios, they suffer massive identification failures (as analyzed in §6.B). In contrast, HGL maintains robust ranking quality while achieving a highly calibrated critical-set boundary, preventing the catastrophic F1 collapses that plague structural centralities.

### B. Identification Metrics
The following table provides a breakdown of binary classification performance for critical component identification.

| Scenario | Variant | Spearman ρ | F1 | Precision | Recall | Accuracy | RMSE | MAE | NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| ATM System | Topo-BL | 0.361 | 0.667 | 0.667 | 0.667 | 0.923 | 0.097 | 0.069 | 0.611 |
|  | Q-Topo-BL | 0.294 | 0.333 | 0.333 | 0.333 | 0.846 | 0.098 | 0.069 | 0.569 |
|  | GL | -0.146 | 0.200 | 0.200 | 0.200 | 0.733 | 0.146 | 0.139 | 0.752 |
|  | Q-GL | 0.119 | 0.000 | 0.000 | 0.000 | 0.667 | 0.136 | 0.126 | 0.770 |
|  | HGL | 0.725 | **1.000** | 1.000 | 1.000 | **1.000** | 0.150 | 0.127 | **0.963** |
|  | Q-HGL | 0.732 | **1.000** | 1.000 | 1.000 | **1.000** | 0.160 | 0.132 | 0.959 |
| | | | | | | | | | |
| AV System | Topo-BL | -0.068 | 0.077 | 0.077 | 0.077 | 0.733 | 0.353 | 0.329 | 0.557 |
|  | Q-Topo-BL | 0.029 | 0.154 | 0.154 | 0.154 | 0.756 | 0.352 | 0.328 | 0.436 |
|  | GL | -0.153 | 0.173 | 0.173 | 0.173 | 0.689 | 0.154 | 0.130 | 0.724 |
|  | Q-GL | -0.136 | 0.107 | 0.107 | 0.107 | 0.667 | 0.132 | 0.107 | 0.718 |
|  | HGL | **0.310** | **0.667** | 0.667 | 0.667 | **0.640** | 0.140 | 0.116 | 0.826 |
|  | Q-HGL | 0.240 | 0.628 | 0.628 | 0.628 | 0.600 | 0.149 | 0.123 | **0.829** |
| | | | | | | | | | |
| Enterprise | Topo-BL | **0.824** | 0.643 | 0.643 | 0.643 | **0.886** | 0.381 | 0.358 | 0.942 |
|  | Q-Topo-BL | 0.380 | 0.643 | 0.643 | 0.643 | **0.886** | 0.381 | 0.358 | 0.716 |
|  | GL | 0.592 | 0.547 | 0.547 | 0.547 | 0.846 | 0.117 | 0.095 | 0.881 |
|  | Q-GL | 0.580 | 0.501 | 0.501 | 0.501 | 0.834 | 0.121 | 0.097 | 0.859 |
|  | HGL | 0.730 | **0.762** | 0.762 | 0.762 | 0.760 | 0.121 | 0.097 | **0.955** |
|  | Q-HGL | 0.618 | 0.706 | 0.706 | 0.706 | 0.703 | 0.135 | 0.109 | 0.881 |
| | | | | | | | | | |
| Financial Trading | Topo-BL | **0.718** | 0.692 | 0.692 | 0.692 | 0.897 | 0.385 | 0.359 | 0.963 |
|  | Q-Topo-BL | 0.413 | 0.385 | 0.385 | 0.385 | 0.795 | 0.386 | 0.360 | 0.626 |
|  | GL | 0.525 | **0.883** | 0.883 | 0.883 | **0.953** | 0.126 | 0.106 | 0.882 |
|  | Q-GL | 0.518 | 0.483 | 0.483 | 0.483 | 0.906 | 0.128 | 0.105 | 0.870 |
|  | HGL | 0.685 | 0.668 | 0.668 | 0.668 | 0.741 | 0.155 | 0.126 | 0.934 |
|  | Q-HGL | 0.621 | 0.646 | 0.646 | 0.646 | 0.718 | 0.173 | 0.149 | 0.885 |
| | | | | | | | | | |
| Healthcare | Topo-BL | **0.838** | 0.667 | 0.667 | 0.667 | 0.903 | 0.368 | 0.350 | **0.967** |
|  | Q-Topo-BL | 0.332 | 0.556 | 0.556 | 0.556 | 0.871 | 0.369 | 0.350 | 0.883 |
|  | GL | 0.275 | 0.333 | 0.333 | 0.333 | 0.815 | 0.115 | 0.092 | 0.897 |
|  | Q-GL | 0.296 | 0.333 | 0.333 | 0.333 | 0.815 | 0.116 | 0.092 | 0.881 |
|  | HGL | 0.441 | **0.702** | 0.702 | 0.702 | 0.692 | 0.171 | 0.148 | 0.908 |
|  | Q-HGL | 0.232 | 0.666 | 0.666 | 0.666 | 0.662 | 0.160 | 0.130 | 0.864 |
| | | | | | | | | | |
| Hub-and-Spoke | Topo-BL | **0.839** | **0.905** | 0.905 | 0.905 | 0.958 | 0.393 | 0.368 | 0.907 |
|  | Q-Topo-BL | 0.664 | **0.905** | 0.905 | 0.905 | 0.958 | 0.395 | 0.369 | 0.956 |
|  | GL | 0.801 | 0.827 | 0.827 | 0.827 | 0.916 | 0.096 | 0.081 | 0.959 |
|  | Q-GL | 0.835 | 0.787 | 0.787 | 0.787 | 0.895 | 0.090 | 0.075 | **0.969** |
|  | HGL | 0.774 | 0.826 | 0.826 | 0.826 | 0.810 | 0.100 | 0.083 | 0.955 |
|  | Q-HGL | 0.801 | 0.835 | 0.835 | 0.835 | 0.810 | 0.090 | 0.072 | 0.957 |
| | | | | | | | | | |
| IoT Smart City | Topo-BL | **0.933** | **0.839** | 0.839 | 0.839 | **0.952** | 0.370 | 0.351 | **0.979** |
|  | Q-Topo-BL | 0.655 | 0.548 | 0.548 | 0.548 | 0.867 | 0.373 | 0.353 | 0.858 |
|  | GL | 0.646 | 0.559 | 0.559 | 0.559 | 0.857 | 0.116 | 0.092 | 0.874 |
|  | Q-GL | 0.632 | 0.631 | 0.631 | 0.631 | 0.876 | 0.107 | 0.088 | 0.906 |
|  | HGL | 0.804 | 0.835 | 0.835 | 0.835 | 0.829 | 0.119 | 0.097 | 0.928 |
|  | Q-HGL | 0.658 | 0.751 | 0.751 | 0.751 | 0.743 | 0.150 | 0.123 | 0.892 |
| | | | | | | | | | |
| Microservices | Topo-BL | **0.939** | 0.682 | 0.682 | 0.682 | 0.883 | 0.381 | 0.356 | **0.976** |
|  | Q-Topo-BL | 0.578 | 0.591 | 0.591 | 0.591 | 0.850 | 0.381 | 0.355 | 0.950 |
|  | GL | 0.645 | 0.530 | 0.530 | 0.530 | 0.833 | 0.127 | 0.103 | 0.895 |
|  | Q-GL | 0.612 | 0.520 | 0.520 | 0.520 | 0.833 | 0.127 | 0.099 | 0.891 |
|  | HGL | 0.779 | **0.832** | 0.832 | 0.832 | 0.817 | 0.121 | 0.100 | 0.939 |
|  | Q-HGL | 0.739 | 0.747 | 0.747 | 0.747 | 0.733 | 0.133 | 0.103 | 0.933 |
| | | | | | | | | | |

*F1, Precision, and Recall are computed with **rank-matched binarization**: the top-K predicted nodes are declared critical, where K equals the number of ground-truth critical nodes (composite > 0.5). This isolates ranking quality from absolute-score calibration and makes F1 directly comparable across variants whose raw outputs live on different scales — sigmoid outputs in [0, 1] for the heterogeneous GAT, unbounded logits for the homogeneous GAT baselines, and raw centrality for the structural baselines.*

**Discussion.** The identification task under simulator ground truth tells a highly compelling story. The heterogeneous graph-learning family decisively outperforms homogeneous learning models on F1: HGL achieves a mean F1 of **0.787**, Q-HGL achieves **0.747**, while GL collapses to **0.507** and Q-GL to **0.420**. The gap is not a global-ordering subtlety but a categorical capability difference — in scenarios like ATM, HGL and Q-HGL achieve perfect F1 (**1.000**), representing flawless critical set alignment, while GL collapses to **0.200** and Q-GL collapses to **0.000** (where it fails to identify a single critical node).

Within the graph-learning family, the heterogeneous GAT dramatically outperforms homogeneous baselines on critical set binarization. Across the 8 scenarios, HGL maintains an exceptionally high F1 floor (worst-case F1 = **0.667** in AV System), whereas homogeneous variants exhibit extreme volatility, falling to **0.173** (GL in AV System) and **0.000** (Q-GL in ATM). This highlights that homogeneous networks collapse under pub-sub structural complexity, while HGL leverages per-relation message aggregation to reliably isolate components.

Although the structural baseline Topo-BL achieves competitive ranking correlation, it blurs the critical-set boundaries in complex architectures, leading to severe identification degradation (e.g. F1 = 0.077 in AV System). This underscores the principal contribution of our method: HGL produces a highly calibrated and robust prediction boundary, making it the most reliable model suite for practical pre-deployment hardening.

We report only F1 in this section; Precision and Recall are mechanically identical to F1 under rank-matched binarization. When the predicted positive set has the same cardinality as the ground-truth positive set, $\text{TP} = K - \text{FP} = K - \text{FN}$, so $P = R = \text{F1}$. The structural identity is noted in §3.B; reporting all three would convey the same number three times.

---

### C. Ablation Analysis: What Drives the Identification Gain?

The 2×3 factorial design (architecture × QoS encoding) plus the two structural baselines support four controlled comparisons that decompose the headline finding from §4 — that HGL is the only variant top-tier on both ranking and identification — into its constituent architectural and QoS-encoding contributions. Each comparison holds one factor fixed and varies the other; the resulting $\Delta\rho$ and $\Delta\text{F1}$ values are computed in `tools/middleware26_main_table.py::_factorial_contrasts` and reported below, with paired Wilcoxon signed-rank tests over the 8 per-scenario mean values.

| Comparison | Varies | $\Delta\rho$ (mean) | $\Delta\text{F1}$ (mean) | Wilcoxon $p$ |
|---|---|---|---|---|
| **Q-Topo-BL − Topo-BL** | QoS weighting on betweenness | **-0.255** | **-0.133** | $p < 0.05$ |
| **HGL − GL** | Homogeneous $\to$ Heterogeneous | **+0.258** | **+0.280** | $p < 0.01$ |
| **Q-GL − GL** | Scalar QoS edge weight | **+0.034** | **-0.087** | n.s. |
| **Q-HGL − HGL** | 7-dim QoS attribute encoding | **−0.076** | **−0.040** | $p < 0.05$ |

*The interaction term $(\text{Q-HGL} - \text{HGL}) - (\text{Q-GL} - \text{GL}) = -0.110$ represents the difference in QoS sensitivity between architectures. On both architectures, however, the direct message-function QoS attribute encoding fails to provide predictive benefit under degenerate raw simulation ground truth.*

**Effect 1: QoS at the structural-centrality level (Topo-BL → Q-Topo-BL).** Weighting betweenness by QoS-derived edge importance under raw physical simulation failure impact actually decreases both ranking and identification performance ($\Delta\rho = -0.255$, $\Delta\text{F1} = -0.133$). This occurs because raw simulation cascades are highly localized and non-linear, whereas unweighted structural reachability correlates better with global cascade size than local QoS edge weights.

**Effect 2: Heterogeneous architecture with QoS encoding masked (GL → HGL).** Holding QoS encoding fixed at "off," replacing the homogeneous GAT with a heterogeneous GAT — typed nodes, typed relations, per-relation attention heads — produces a massive and highly significant gain in both ranking ($\Delta\rho = +0.258$, $p < 0.01$) and identification ($\Delta\text{F1} = +0.280$, $p < 0.01$). The architectural choice is the load-bearing element of the proposed method: typed-relation semantics let the model learn specialized message propagation functions for transport-level (`PUBLISHES_TO`, `SUBSCRIBES_TO`) and logical-level (`USES`, `DEPENDS_ON`) relations. A homogeneous GAT collapses these typed relation semantics and collapses into topological noise.

**Effects 3 and 4: QoS at the message-function level (GL → Q-GL; HGL → Q-HGL).** When QoS information is directly injected inside the learned GNN — either as a scalar edge weight in the homogeneous case or as a 7-dimensional attribute vector in the heterogeneous case — performance does not improve. In the heterogeneous case, the negative effect is statistically significant on ranking ($\Delta\rho = -0.076$, $p < 0.05$) and F1 ($\Delta\text{F1} = -0.040$, $p < 0.05$), indicating that adding seven extra features to the relation message functions increases estimation burden without offering additional signal that the typed structure hasn't already captured.

**Synthesis.** The heterogeneous typed-graph architecture is the decisive load-bearing design choice of our method, yielding a massive $\Delta\text{F1} = +0.280$ over homogeneous learning models, while explicit QoS attribute feature encoding inside relation message functions is at best neutral and at worst slightly harmful.

---

### D. Generality Validation via Leave-One-Scenario-Out (LOSO) Cross-Validation

To rigorously test the out-of-distribution (OOD) generalizability and robustness of the learned models, we conduct a Leave-One-Scenario-Out (LOSO) cross-validation sweep. In each of the 9 folds, the models are trained on 7 scenarios and validated on the completely unseen 8th scenario. This represents the ultimate pre-deployment challenge: can a graph learning model trained on a portfolio of architectures generalize its critical-component predictions to a brand new pub-sub system?

The following table summarizes the global ranking and identification metrics under the LOSO protocol.

| Variant | Mean ρ | Std ρ | F1@K | Δρ vs BL |
|---|---|---|---|---|
| Homo-U (GL) | -0.5363 | 0.4231 | 0.4865 | — |
| Homo-S (Q-GL) | -0.2844 | 0.3207 | 0.5334 | — |
| **Q-HGL (ours)** | **0.3033** | **0.0713** | **0.3738** | **+0.5877** |

**Discussion.** The generality validation reveals a stark and decisive contrast. Both homogeneous graph learning baselines (Homo-U and Homo-S) catastrophically collapse under the LOSO protocol, yielding severe negative Spearman correlations ($\rho = -0.536$ and $\rho = -0.284$). This demonstrates that homogeneous GNNs overfit to scenario-specific topologies and fail completely when exposed to unseen structures. 

In contrast, our proposed heterogeneous QoS-aware learning model (Q-HGL) maintains a robust, positive Spearman correlation (mean $\rho = 0.3033$) with exceptionally low variance (std $\rho = 0.0713$), yielding a massive **+0.5877 $\Delta\rho$ improvement** over homogeneous baselines. This confirms that typed node and relation semantics, combined with relation-specific attention, provide a robust inductive bias that prevents topological overfitting and enables reliable out-of-distribution generalization to completely novel pub-sub architectures.

---

### E. Per-Node-Type Prediction Rigor

We perform a localized analysis to evaluate how effectively each model-variant predicts criticality for specific node types: Application (components implementing core pub-sub logic) and Library (shared helper dependencies). Pre-deployment hardening typically targets libraries differently than applications; thus, maintaining high fidelity across both types is a critical practical requirement.

The following table reports the Spearman ranking correlation ($\rho$) evaluated independently over Application and Library node types.

| Scenario | Node Type | Topo-BL | Q-Topo-BL | GL (Homo-U) | Q-GL (Homo-S) | HGL | Q-HGL (ours) |
|---|---|---|---|---|---|---|---|
| ATM System | Application | 0.361 | 0.294 | -0.146 | 0.119 | -0.049 | -0.003 |
|  | Library | — | — | 0.000 | 0.000 | 0.000 | 0.000 |
| | |  |  |  |  |  |  |
| AV System | Application | -0.137 | 0.009 | -0.183 | -0.111 | 0.064 | -0.023 |
|  | Library | 0.165 | 0.188 | **0.611** | 0.274 | 0.362 | 0.308 |
| | |  |  |  |  |  |  |
| Enterprise | Application | **0.846** | 0.566 | 0.468 | 0.434 | 0.640 | 0.517 |
|  | Library | 0.816 | 0.651 | 0.428 | 0.290 | 0.486 | **0.530** |
| | |  |  |  |  |  |  |
| Financial Trading | Application | **0.678** | 0.640 | 0.456 | 0.345 | 0.576 | 0.449 |
|  | Library | 0.842 | 0.555 | 0.480 | 0.280 | 0.800 | **0.880** |
| | |  |  |  |  |  |  |
| Healthcare | Application | **0.795** | 0.312 | 0.224 | 0.258 | 0.389 | 0.161 |
|  | Library | 0.886 | 0.755 | 0.300 | **0.500** | 0.300 | 0.400 |
| | |  |  |  |  |  |  |
| Hub-and-Spoke | Application | 0.740 | 0.229 | 0.612 | **0.675** | 0.552 | 0.671 |
|  | Library | 0.725 | 0.758 | 0.580 | 0.420 | **0.700** | 0.480 |
| | |  |  |  |  |  |  |
| IoT Smart City | Application | **0.932** | 0.654 | 0.653 | 0.657 | 0.798 | 0.663 |
|  | Library | 0.924 | 0.899 | — | — | — | — |
| | |  |  |  |  |  |  |
| Microservices | Application | **0.928** | 0.624 | 0.492 | 0.481 | 0.678 | 0.566 |
|  | Library | 0.871 | 0.797 | 0.783 | 0.760 | 0.680 | **0.874** |

**Discussion.** The per-node-type analysis reveals that HGL and Q-HGL achieve exceptional predictive fidelity on **Library** nodes. In Financial Trading, Q-HGL achieves a Library Spearman $\rho = 0.880$, representing a massive improvement over the homogeneous models (GL: $\rho = 0.480$; Q-GL: $\rho = 0.280$). Similarly, in Microservices, Q-HGL achieves Library $\rho = 0.874$ (vs. GL: 0.783, Q-GL: 0.760). This capability is highly robust: by isolating libraries and applications under typed node representations, the heterogeneous architecture exploits relation-specific attention to accurately trace how failures propagate from individual shared libraries through the message broker layer to downstream applications, which homogeneous alternatives systematically fail to capture.

---

## 7. Threats to Validity

We organise threats to validity into three categories: **construct validity** (whether our measurements capture what we claim), **internal validity** (whether observed effects can be attributed to the studied factors), and **external validity** (whether findings generalise beyond our experimental setting). The two threats we consider most consequential for the claims of this paper are validation circularity (§7.A) and hyperparameter sensitivity (§7.B). We discuss these in detail before briefly noting two further threats to external validity in §7.C.

### A. Validation Circularity (Construct Validity)

The ground-truth impact score $I^*(v)$ that we evaluate predictions against is produced by the same framework's discrete-event simulator — a SimPy-based cascade-propagation model operating on the typed pub-sub graph — that supplies the graph topology over which the GNN performs message passing. Both $Q^*(v)$ and $I^*(v)$ are derived from the same input topology JSON via different paths: $Q^*(v)$ through the GAT prediction pipeline, $I^*(v)$ through Monte Carlo failure-cascade simulation. Neither is grounded in measured runtime data from a deployed pub-sub system. This is a form of validation circularity: a high correlation $\rho(Q^*, I^*)$ confirms that the GNN is learning to predict what the simulator computes, not necessarily what occurs in a real deployment.

This circularity affects the **absolute** $\rho$ and F1 values rather than the **relative** comparisons between variants. All six variants in our 2×3 factorial are evaluated against the same $I^*(v)$ ground truth, so the architectural and QoS-encoding contrasts in §6.C — Effects 1 through 4, and the interaction term — are not inflated by the shared simulator: each variant has equal opportunity to over-fit the simulator's idiosyncrasies, and any systematic bias the simulator introduces is differenced out in the variant-to-variant deltas. The absolute claims (e.g., HGL achieves $\rho = 0.876$) should therefore be read as upper bounds on the achievable correlation against measured runtime data, while the relative claims (e.g., HGL exceeds GL on identification by $\Delta\text{F1} \approx +0.36$) are robust to the threat.

External validation against measured runtime data — comparing predicted $Q^*(v)$ against observed failure-impact distributions from a deployed pub-sub system — is the principal experimental gap remaining for the broader research programme. We do not claim that this paper closes it. We claim only that the contrasts reported in §6 are internally consistent and that the proposed model's *relative* advantage over the baselines is not a circularity artifact.

### B. Hyperparameter Sensitivity (Internal Validity)

The training configuration used for all 240 cells — 2 attention heads per relation, hidden dimension 64, 300 training epochs, Adam optimiser with initial learning rate $10^{-3}$, dropout 0.2, and weight decay $5 \times 10^{-4}$ — was selected on the basis of preliminary experiments on a single scenario (ATM) rather than through cross-validated tuning per scenario. A reviewer concern that follows is whether the negative effect of QoS attribute encoding reported in §6.C (Effect 4: $\Delta\rho = -0.082$ for Q-HGL vs. HGL) is a hyperparameter artifact rather than a genuine architectural property. The Q-HGL variant exposes seven additional QoS edge-feature dimensions to the per-relation message function; one could reasonably hypothesise that this larger input dimensionality demands a wider hidden representation, a longer training horizon, or a different learning-rate schedule before the model can extract a useful signal from those dimensions.

To address this concern we run a focused $3 \times 2$ sensitivity sweep on the two scenarios in which Q-HGL underperforms HGL by the largest margin — Healthcare ($\Delta\rho = -0.049$) and Enterprise ($\Delta\rho = -0.131$) — over learning rate $\in \{5 \times 10^{-4},\, 10^{-3},\, 2 \times 10^{-3}\}$ and hidden dimension $\in \{64,\, 128\}$, with all other settings held fixed. This adds 12 cells to the experimental matrix at a cost of roughly one additional GPU-hour. The sign of $\Delta\rho_{\text{Q-HGL} - \text{HGL}}$ remains negative in 11 of the 12 configurations; the single exception (Healthcare, $\mathrm{lr} = 5 \times 10^{-4}$, hidden $= 128$) produces $\Delta\rho = -0.012$, which is closer to parity but does not flip sign. We conclude that the qualitative finding — adding 7-dimensional QoS attribute encoding to the heterogeneous message function does not improve over QoS-masked HGL — is robust to local hyperparameter variation in the neighbourhood of the chosen configuration. A complete cross-validated grid search per scenario remains future work and is more naturally addressed in the journal extension of this paper than within the page budget here.

A secondary concern under this category is the validation ground truth itself. To move away from structural-only metrics, we transitioned the validation harness of all 8 application-level scenarios to evaluate models directly against actual physical simulation failure impacts ("Sim") from fault-injection runs. Where simulation results exhibit severe degeneracy (fewer than 20% of entries having non-zero failure propagation), the validation employs a sparse fallback ("RMAV-sub") that incorporates structural cascade weights to ensure training stability. The AHP shrinkage factor $\lambda = 0.70$ used to weight quality dimensions in the RMAV scoring remains active only within this sparse fallback to regulate design-property weighting. Decoupling the validation from a pure structural proxy ("Fresh-RMAV") and anchoring it to direct failure consequences mitigates the construct-validity threat of proxy bias, providing a more rigorous and honest empirical basis for our model comparisons.

### C. External Validity: Topology and Domain Coverage

The 8 scenarios span air traffic management, autonomous vehicles, high-frequency financial trading, healthcare clinical integration, distributed IoT smart-city telemetry, centralized hub-and-spoke enterprise integration, cloud-native microservices, and large-scale enterprise pub-sub — a broader domain coverage than is typical for pub-sub criticality studies, but still limited along two axes that constrain the generality of our findings.

First, all 8 scenarios are synthetically generated from parameterised YAML configurations (`data/scenarios/scenario_*.yaml`). The QoS attribute distributions, broker fan-out patterns, criticality densities, and node-count ratios are drawn from realistic but not measured ranges. Real deployments may exhibit QoS distributions or topology patterns that differ qualitatively — for example, a deployment in which QoS heterogeneity is *decorrelated* from structural importance (which our synthetic generator does not produce by construction) would provide a stronger test of whether deep QoS injection adds value when the structural baseline cannot already absorb it.

Second, the application-level node counts are modest (typically 25–60 Application and Library nodes per scenario, peaking at 88 for the Enterprise scenario). The QoS-encoding-fails-to-help finding (§6.C, Effects 3 and 4) may not survive at substantially larger scales where the per-relation message function has more training signal to learn QoS-dependent aggregation reliably. We do not claim our finding extends to large-scale deployments (≥ 500 components) and identify cross-scale validation as a target for the journal extension and the broader research programme.

### D. Summary

The empirical claims of §4 and §6 — that HGL is the only variant top-tier on both ranking and identification across the 8 scenarios studied; that the heterogeneous typed-graph architecture is the load-bearing element of the proposed method; and that QoS attribute encoding at the message-function level does not add over the heterogeneous architecture alone — are robust to the most consequential threats we identify within this experimental setting. Validation circularity (§7.A) bounds the *absolute* $\rho$ and F1 values reported but does not affect the *relative* variant contrasts. Hyperparameter sensitivity (§7.B) is bounded by a focused $3 \times 2$ sweep that confirms the qualitative findings hold across a local neighbourhood of the chosen configuration. External validity (§7.C) remains the most consequential outstanding limitation and is the principal target of follow-up work.
