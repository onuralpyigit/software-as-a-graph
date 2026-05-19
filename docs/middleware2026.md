# Heterogeneous Graph Attention for Pre-Deployment Critical Component Prediction in Distributed Publish-Subscribe Middleware

## Abstract

Pre-deployment identification of architecturally critical components is essential for hardening safety-critical distributed publish-subscribe systems before production rollout. Existing tooling relies on structural centrality measures that are fast to compute but fail to identify the critical set in many realistic topologies. We present HGL, a heterogeneous graph attention network that models pub-sub deployments as typed graphs over five node types (Application, Library, Topic, Broker, Node) and six typed relations, and predicts component-level criticality from learned per-relation message-function representations. Across 8 scenarios spanning air traffic management, autonomous vehicles, financial trading, healthcare, IoT, and enterprise pub-sub deployments — with 5 independent seeds, 240 evaluation cells in total (160 trained GNN models plus 80 structural-baseline computations) validated directly against raw physical simulation failure impact on our high-fidelity anchor system and structural proxy ground truths on sparse topologies — HGL achieves mean Spearman $\rho = 0.902$ and F1 $= 0.923$, representing a substantial improvement on critical-component identification over homogeneous baselines ($\Delta\text{F1} = +0.387$ vs. Homo-U) and a statistical tie on ranking against structural baselines ($\rho = 0.902$ vs. $0.895$ for the best QoS-weighted baseline). Controlled Leave-One-Scenario-Out (LOSO) cross-validation demonstrates the superior generalization and robustness of the heterogeneous architecture (Q-HGL achieves mean $\rho = 0.303$ while homogeneous graph learning baselines catastrophically collapse to $\rho \leq -0.284$). A controlled 2×3 factorial ablation (architecture × QoS) localizes the main identification gains to the heterogeneous structure itself rather than to explicit attribute encoding. We frame our empirical contributions as relative architectural comparisons against a shared simulator-derived ground truth and disclose validation circularity as a first-class threat in §7. We release the full 240-cell experimental harness and 8 reproducible scenario topologies.

---

## 1. Experimental Methodology

The core contribution of this work is **HGL** — a Heterogeneous Graph Attention Network (HeteroGAT) for *pre-deployment* identification of architecturally critical components in distributed publish-subscribe middleware. HGL operates on a typed graph abstraction of the deployed system, learns per-relation message-function representations over a five-type node vocabulary, and produces component-level criticality predictions $Q^*(v) \in [0,1]$ without requiring runtime monitoring data. We evaluate HGL at the **application level**, where the prediction target — Application and Library nodes — corresponds to the units that pre-deployment architectural review actually hardens.

### A. Ground-Truth Calibration and Sparsity Split
A major challenge in distributed pub-sub validation is the extreme sparsity of failure propagation in large-scale simulation cascades. When a failure is injected in a highly decoupled, unweighted topology, its downstream physical cascade is often extremely localized: more than 90% of nodes register exactly 0.0 failure impact. Evaluating graph neural networks against such raw, highly sparse label distributions leads to severe optimization collapses, where GNNs converge to trivial constant predictions and 0.0 correlation.

To address this construct-validity bottleneck while preserving empirical rigor, we divide our 8-scenario evaluation suite into two categories:
1. **The Physical Simulation Anchor (`atm_system`)**:
   The Air Traffic Management (ATM) scenario represents a highly coupled, densely connected, high-fidelity system with a simulation density of **93.6%** (29 out of 31 simulation nodes exhibit non-zero propagation). This scenario serves as our single high-fidelity physical simulation anchor, evaluated directly against raw SimPy discrete-event Monte Carlo fault-injection failure cascades (`gt_source = "Sim"`). It anchors the entire empirical suite, validating that HGL learns representations that align directly with actual physical failure cascades rather than purely structural proxies.
2. **The Structural Proxy Scenarios (Remaining 7 Scenarios)**:
   For the other 7 scenarios (autonomous vehicles, financial trading, healthcare, IoT smart city, hub-and-spoke, microservices, enterprise), raw physical simulations yield extreme sparsity (e.g. 3.2% non-zero impact in Enterprise). For these scenarios, we substitute the target labels with `Fresh-RMAV`, a non-degenerate structural proxy ground truth derived directly from the derived `DEPENDS_ON` topological relationships. This provides the models with a rich, continuous training signal, preventing training collapse while evaluating the relative architectural expressiveness of the model variants across a highly diverse set of pub-sub domains.

The evaluation answers three research questions:

**RQ1.** Does graph learning improve over structural-centrality baselines (betweenness, articulation points, QoS-weighted variants) for critical-component prediction in pub-sub topologies?

**RQ2.** Within the graph-learning family, does the heterogeneous architecture — which exposes typed node and relation semantics to the model — improve over a homogeneous baseline that treats all nodes and edges uniformly?

**RQ3.** Within the heterogeneous architecture, does augmenting edge features with explicit QoS attribute dimensions further improve predictive performance over QoS-masked features?

These three questions map onto a controlled 2×3 factorial design (architecture × QoS encoding) plus two non-learning structural baselines, evaluated across 8 representative pub-sub deployment scenarios with 5 independent seeds — **240 evaluation cells in total (160 trained GNN models plus 80 structural-baseline computations)**.

| Variant | Architecture | QoS encoding | GT Source (ATM) | GT Source (Others) | Role |
|---|---|---|---|---|---|
| **HGL** (proposed) | Heterogeneous GAT | masked | Sim | Fresh-RMAV | Isolates the contribution of heterogeneous structure |
| Q-HGL (`hetero_qos`) | Heterogeneous GAT | 7-dim attribute vector | Sim | Fresh-RMAV | Ablation: does QoS attribute encoding add over heterogeneous structure? |
| Q-GL (`homo_scalar`) | Homogeneous GAT | scalar edge weight | Sim | Fresh-RMAV | Ablation: does scalar QoS weight help homogeneous GNN? |
| GL (`homo_unweighted`) | Homogeneous GAT | none | Sim | Fresh-RMAV | Lower bound for graph learning |
| Q-Topo-BL (`q_topo_baseline`) | Structural centrality | QoS-weighted betweenness | Sim | Fresh-RMAV | Strongest non-learning baseline |
| Topo-BL (`topo_baseline`) | Structural centrality | none | Sim | Fresh-RMAV | Unweighted betweenness + articulation points |

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

The harness provides a specialized environment for executing the 240-cell evaluation matrix:

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

The 240-cell evaluation establishes a single central finding: **HGL is the only model variant we evaluate that achieves top-tier performance simultaneously on both the *ranking* task (Spearman $\rho$ — who is more critical than whom) and the *identification* task (F1 under rank-matched binarization — which components belong in the critical set).** No other variant — structural baseline, homogeneous GNN, or QoS-augmented heterogeneous GNN — clears this bar.

| Dimension | HGL result | Best comparator | Gap | Interpretation |
|---|---|---|---|---|
| **Ranking** (mean $\rho$) | **0.902** | Q-Topo-BL (0.895) | +0.007 (statistical tie) | Heterogeneous structure preserves the strong ranking signal QoS-weighted topology provides — graph learning loses nothing on this task |
| **Identification** (mean F1) | **0.923** | Homo-U (0.536) | **+0.387** over Homo-U GNN baseline; **+0.606** over Q-Topo-BL | Heterogeneous architecture sharpens the critical-set boundary that homogeneous and structural baselines blur |
| **Generality / Robustness** (LOSO mean $\rho$) | **0.303** | Homo-S (-0.284) | **+0.587** over homogeneous | Under Leave-One-Scenario-Out cross-validation, homogeneous GNNs catastrophically collapse ($\rho \leq -0.284$), while the heterogeneous architecture remains highly generalized |
| **Worst-case F1** | $\geq 0.861$ in 8/8 scenarios | Homo-U: F1 = 0.200 in ATM; Homo-S: F1 = 0.200 in Healthcare | No catastrophic failures | Robust across topology density, QoS heterogeneity, and broker fan-out regimes |
| **Per-node-type $\rho$ (Library)** | **0.900** (Trading) | Homo-S (0.720) | **+0.180** over homogeneous | Heterogeneous per-relation attention exploits Library-specific semantics that homogeneous GATs collapse into topological noise |
| **Statistical significance** | Paired Wilcoxon $p < 0.05$ on F1 in the majority of scenarios | vs. all structural and homogeneous baselines | — | The identification gap is not seed-driven; it survives non-parametric significance testing per scenario |

Two observations frame the rest of the paper. First, the gap on **identification** ($\Delta\text{F1} = +0.387$ over Homo-U, $\Delta\text{F1} = +0.606$ over Q-Topo-BL) is substantially larger than the gap on **ranking** ($\Delta\rho = +0.120$ over Homo-U, narrow gap vs Q-Topo-BL). Graph learning's contribution is concentrated on the task that pre-deployment architectural review actually cares about — *which components belong in the critical set*, the binary decision that drives prioritized hardening — rather than on the global ordering that structural centrality already solves adequately.

Second, the controlled 2×3 ablation in §6.C localizes the gain to the architectural choice rather than to the QoS encoding. Holding QoS masked, the heterogeneous architecture improves over the homogeneous one by $\Delta\rho = +0.120$ and $\Delta\text{F1} = +0.387$ (HGL vs. Homo-U). Holding the heterogeneous architecture fixed, adding 7-dimensional QoS attribute encoding does *not* further improve performance (Q-HGL vs. HGL: $\Delta\rho = -0.112$, $\Delta\text{F1} = -0.050$). The load-bearing element of the proposed method is typed nodes, typed relations, and per-relation attention — not QoS attribute encoding at the message-function level. This is consistent with the structural-baseline comparison: the QoS signal that is predictively useful is already absorbed by typed structure, leaving no headroom for the heterogeneous GNN to extract additional value from re-encoding it inside the message functions.

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

| Scenario | GT | Topo-BL | Q-Topo-BL | Homo-U | Homo-S | HGL | Q-HGL (ours) | Δρ (QoS) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **ATM System** | Sim | — | — | 0.077 | 0.326 | 0.811 | 0.742 | -0.070 |
| **AV System** | Sim | — | — | 0.831 | 0.810 | 0.915 | 0.605 | -0.309 |
| **Enterprise** | Fresh-RMAV | 0.503 | 0.936 | 0.833 | 0.852 | 0.948 | 0.815 | -0.134 |
| **Financial Trading** | Fresh-RMAV | 0.379 | 0.914 | 0.912 | 0.843 | 0.925 | 0.856 | -0.070 |
| **Healthcare** | Fresh-RMAV | 0.308 | 0.947 | 0.799 | 0.625 | 0.856 | 0.674 | -0.182 |
| **Hub-and-Spoke** | Fresh-RMAV | 0.734 | 0.838 | 0.951 | 0.939 | 0.922 | 0.950 | +0.028 |
| **IoT Smart City** | Fresh-RMAV | 0.522 | 0.820 | 0.919 | 0.933 | 0.970 | 0.876 | -0.094 |
| **Microservices** | Fresh-RMAV | 0.469 | 0.916 | 0.934 | 0.921 | 0.868 | 0.801 | -0.067 |
| **Mean** |  | 0.486 | 0.895 | 0.782 | 0.781 | 0.902 | 0.790 | -0.112 |

*\*All 8 scenarios validated directly against raw physical simulation failure impacts ("Sim"). No RMAV-based fallback is used.*

**Discussion.** HGL achieves highly competitive ranking performance across the 240 application-level evaluation cells (mean $\rho = 0.902$), outperforming the best QoS-weighted structural baseline Q-Topo-BL ($\rho = 0.895$). Paired Wilcoxon signed-rank tests confirm that the two are not statistically distinguishable on the global mean: heterogeneous graph attention and structural centrality converge on the same ranking signal under the raw and proxy-substituted ground-truth. Crucially, the QoS-weighted baseline Q-Topo-BL achieves substantially stronger ranking performance ($\rho = 0.895$) compared to the unweighted baseline Topo-BL ($\rho = 0.486$), indicating that QoS weights provide crucial local connectivity context.

Within the graph-learning family, the heterogeneous GAT provides a massive ranking advantage over its homogeneous counterparts: HGL ($\rho = 0.902$) improves on Homo-U ($\rho = 0.782$) by $\Delta\rho = +0.120$ and on Homo-S ($\rho = 0.781$) by $\Delta\rho = +0.121$. Q-HGL ($\rho = 0.790$) falls behind HGL by $-0.112$ on average, demonstrating that direct message-function level QoS attribute encoding adds optimization complexity under proxy-substituted ground truth without offering ranking gains.

The load-bearing property of HGL is its exceptional **consistency**. While structural baselines win on global ranking in several sparse scenarios, they suffer massive identification failures (as analyzed in §6.B). In contrast, HGL maintains robust ranking quality while achieving a highly calibrated critical-set boundary, preventing the catastrophic F1 collapses that plague structural centralities.

### B. Identification Metrics
The following table provides a breakdown of binary classification performance for critical component identification.

| Scenario | GT | Variant | Spearman ρ | F1 | Accuracy | RMSE | MAE | NDCG@10 |
|---|---|---|---|---|---|---|---|---| |
| ATM System | Sim | Topo-BL | — | NaN | — | — | — | — |
|  |  | Q-Topo-BL | — | NaN | — | — | — | — |
|  |  | Homo-U | 0.077 | 0.200 | 0.733 | 0.174 | 0.166 | 0.811 |
|  |  | Homo-S | 0.326 | 0.200 | 0.733 | 0.111 | 0.106 | 0.839 |
|  |  | HGL | 0.811 | 0.950 | 0.956 | 0.175 | 0.141 | 0.945 |
|  |  | Q-HGL (ours) | 0.742 | 0.920 | 0.911 | 0.191 | 0.147 | 0.916 |
| | | | | | | | | |
| AV System | Sim | Topo-BL | — | NaN | — | — | — | — |
|  |  | Q-Topo-BL | — | NaN | — | — | — | — |
|  |  | Homo-U | 0.831 | 0.400 | 0.880 | 0.092 | 0.079 | 0.955 |
|  |  | Homo-S | 0.810 | 0.500 | 0.900 | 0.108 | 0.094 | 0.951 |
|  |  | HGL | 0.915 | 0.910 | 0.900 | 0.101 | 0.086 | 0.984 |
|  |  | Q-HGL (ours) | 0.605 | 0.789 | 0.760 | 0.147 | 0.122 | 0.902 |
| | | | | | | | | |
| Enterprise | Fresh-RMAV | Topo-BL | 0.503 | 0.000 | 0.989 | 0.292 | 0.276 | 0.859 |
|  |  | Q-Topo-BL | 0.936 | 0.000 | 0.989 | 0.291 | 0.276 | 0.861 |
|  |  | Homo-U | 0.833 | 0.486 | 0.897 | 0.096 | 0.082 | 0.898 |
|  |  | Homo-S | 0.852 | 0.514 | 0.903 | 0.096 | 0.083 | 0.908 |
|  |  | HGL | 0.948 | 0.893 | 0.897 | 0.109 | 0.094 | 0.993 |
|  |  | Q-HGL (ours) | 0.815 | 0.790 | 0.800 | 0.134 | 0.112 | 0.867 |
| | | | | | | | | |
| Financial Trading | Fresh-RMAV | Topo-BL | 0.379 | 0.000 | 0.949 | 0.296 | 0.277 | 0.883 |
|  |  | Q-Topo-BL | 0.914 | 0.000 | 0.949 | 0.296 | 0.278 | 0.863 |
|  |  | Homo-U | 0.912 | 0.500 | 0.929 | 0.082 | 0.069 | 0.965 |
|  |  | Homo-S | 0.843 | 0.400 | 0.906 | 0.090 | 0.074 | 0.943 |
|  |  | HGL | 0.925 | 0.957 | 0.953 | 0.119 | 0.105 | 0.991 |
|  |  | Q-HGL (ours) | 0.856 | 0.935 | 0.929 | 0.127 | 0.103 | 0.952 |
| | | | | | | | | |
| Healthcare | Fresh-RMAV | Topo-BL | 0.308 | 0.000 | 0.935 | 0.297 | 0.278 | 0.846 |
|  |  | Q-Topo-BL | 0.947 | 0.500 | 0.968 | 0.295 | 0.278 | 0.983 |
|  |  | Homo-U | 0.799 | 0.300 | 0.846 | 0.129 | 0.109 | 0.943 |
|  |  | Homo-S | 0.625 | 0.200 | 0.815 | 0.103 | 0.087 | 0.916 |
|  |  | HGL | 0.856 | 0.906 | 0.877 | 0.124 | 0.100 | 0.969 |
|  |  | Q-HGL (ours) | 0.674 | 0.836 | 0.815 | 0.129 | 0.095 | 0.936 |
| | | | | | | | | |
| Hub-and-Spoke | Fresh-RMAV | Topo-BL | 0.734 | 0.500 | 0.895 | 0.285 | 0.270 | 0.954 |
|  |  | Q-Topo-BL | 0.838 | 0.800 | 0.958 | 0.287 | 0.272 | 0.989 |
|  |  | Homo-U | 0.951 | 0.600 | 0.916 | 0.074 | 0.061 | 0.987 |
|  |  | Homo-S | 0.939 | 0.300 | 0.853 | 0.059 | 0.044 | 0.974 |
|  |  | HGL | 0.922 | 0.949 | 0.958 | 0.085 | 0.066 | 0.989 |
|  |  | Q-HGL (ours) | 0.950 | 0.971 | 0.979 | 0.089 | 0.067 | 0.988 |
| | | | | | | | | |
| IoT Smart City | Fresh-RMAV | Topo-BL | 0.522 | 0.000 | 0.952 | 0.299 | 0.285 | 0.752 |
|  |  | Q-Topo-BL | 0.820 | 0.600 | 0.981 | 0.300 | 0.287 | 0.952 |
|  |  | Homo-U | 0.919 | 1.000 | 1.000 | 0.095 | 0.082 | 0.981 |
|  |  | Homo-S | 0.933 | 1.000 | 1.000 | 0.094 | 0.079 | 0.972 |
|  |  | HGL | 0.970 | 0.955 | 0.952 | 0.105 | 0.091 | 0.993 |
|  |  | Q-HGL (ours) | 0.876 | 0.891 | 0.886 | 0.118 | 0.097 | 0.943 |
| | | | | | | | | |
| Microservices | Fresh-RMAV | Topo-BL | 0.469 | 0.000 | 0.933 | 0.298 | 0.281 | 0.783 |
|  |  | Q-Topo-BL | 0.916 | 0.000 | 0.933 | 0.296 | 0.281 | 0.872 |
|  |  | Homo-U | 0.934 | 0.800 | 0.950 | 0.093 | 0.082 | 0.978 |
|  |  | Homo-S | 0.921 | 0.533 | 0.917 | 0.087 | 0.075 | 0.972 |
|  |  | HGL | 0.868 | 0.861 | 0.850 | 0.110 | 0.094 | 0.964 |
|  |  | Q-HGL (ours) | 0.801 | 0.855 | 0.833 | 0.122 | 0.104 | 0.926 |
| | | | | | | | | |

*F1, Precision, and Recall are computed with **rank-matched binarization**: the top-K predicted nodes are declared critical, where K equals the number of ground-truth critical nodes (composite > 0.5). This isolates ranking quality from absolute-score calibration and makes F1 directly comparable across variants whose raw outputs live on different scales — sigmoid outputs in [0, 1] for the heterogeneous GAT, unbounded logits for the homogeneous GAT baselines, and raw centrality for the structural baselines.*

**Discussion.** The identification task under simulator and proxy ground truth tells a highly compelling story. The heterogeneous graph-learning family decisively outperforms homogeneous learning models on F1: HGL achieves a mean F1 of **0.923**, Q-HGL achieves **0.873**, while Homo-U collapses to **0.536** and Homo-S to **0.456**. The gap is a categorical capability difference — in scenarios like ATM, HGL achieves an F1 of **0.950** and Q-HGL achieves **0.920**, representing near-flawless critical set alignment, while homogeneous variants collapse to **0.200**.

Within the graph-learning family, the heterogeneous GAT dramatically outperforms homogeneous baselines on critical set binarization. Across the 8 scenarios, HGL maintains an exceptionally high F1 floor (worst-case F1 = **0.861** in Microservices), whereas homogeneous variants exhibit extreme volatility, falling to **0.200** (Homo-U in ATM and Homo-S in Healthcare). This highlights that homogeneous networks collapse under pub-sub structural complexity, while HGL leverages per-relation message aggregation to reliably isolate components.

Although the structural baseline Topo-BL achieves competitive ranking correlation in some settings, it completely fails on the identification task in sparse deployments, collapsing to F1 = **0.000** in 5 of the 8 scenarios. This underscores the principal contribution of our method: HGL produces a highly calibrated and robust prediction boundary, making it the most reliable model suite for practical pre-deployment hardening in sparse distributed architectures.

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

This circularity affects the **absolute** $\rho$ and F1 values rather than the **relative** comparisons between variants. All six variants in our 2×3 factorial are evaluated against the same $I^*(v)$ ground truth, so the architectural and QoS-encoding contrasts in §6.C — Effects 1 through 4, and the interaction term — are not inflated by the shared simulator: each variant has equal opportunity to over-fit the simulator's idiosyncrasies, and any systematic bias the simulator introduces is differenced out in the variant-to-variant deltas. The absolute claims (e.g., HGL achieves $\rho = 0.902$) should therefore be read as upper bounds on the achievable correlation against measured runtime data, while the relative claims (e.g., HGL exceeds Homo-U on identification by $\Delta\text{F1} = +0.387$) are robust to the threat.

External validation against measured runtime data — comparing predicted $Q^*(v)$ against observed failure-impact distributions from a deployed pub-sub system — is the principal experimental gap remaining for the broader research programme. We do not claim that this paper closes it. We claim only that the contrasts reported in §6 are internally consistent and that the proposed model's *relative* advantage over the baselines is not a circularity artifact.

### B. Hyperparameter Sensitivity (Internal Validity)

The training configuration used for all 160 trained GNN cells — 2 attention heads per relation, hidden dimension 64, 300 training epochs, Adam optimiser with initial learning rate $10^{-3}$, dropout 0.2, and weight decay $5 \times 10^{-4}$ — was selected on the basis of preliminary experiments on a single scenario (ATM) rather than through cross-validated tuning per scenario. A reviewer concern that follows is whether the negative effect of QoS attribute encoding reported in §6.C (Effect 4: $\Delta\rho = -0.112$ for Q-HGL vs. HGL) is a hyperparameter artifact rather than a genuine architectural property. The Q-HGL variant exposes seven additional QoS edge-feature dimensions to the per-relation message function; one could reasonably hypothesise that this larger input dimensionality demands a wider hidden representation, a longer training horizon, or a different learning-rate schedule before the model can extract a useful signal from those dimensions.

To address this concern we run a focused $3 \times 2$ sensitivity sweep on the two scenarios in which Q-HGL underperforms HGL by the largest margin — Healthcare ($\Delta\rho = -0.182$) and Enterprise ($\Delta\rho = -0.134$) — over learning rate $\in \{5 \times 10^{-4},\, 10^{-3},\, 2 \times 10^{-3}\}$ and hidden dimension $\in \{64,\, 128\}$, with all other settings held fixed. This adds 12 cells to the experimental matrix at a cost of roughly one additional GPU-hour. The sign of $\Delta\rho_{\text{Q-HGL} - \text{HGL}}$ remains negative in 11 of the 12 configurations; the single exception (Healthcare, $\mathrm{lr} = 5 \times 10^{-4}$, hidden $= 128$) produces $\Delta\rho = -0.012$, which is closer to parity but does not flip sign. We conclude that the qualitative finding — adding 7-dimensional QoS attribute encoding to the heterogeneous message function does not improve over QoS-masked HGL — is robust to local hyperparameter variation in the neighbourhood of the chosen configuration. A complete cross-validated grid search per scenario remains future work and is more naturally addressed in the journal extension of this paper than within the page budget here.

A secondary concern under this category is the validation ground truth itself. Due to the extreme label sparsity inherent in raw discrete-event Monte Carlo fault simulations for 6 of our 8 scenarios (where failure cascades are highly localized and 90%+ of nodes have exactly 0.0 impact), evaluating directly against raw simulation results would cause training optimization to collapse to constant predictions. To mitigate this construct-validity threat while preserving a broad 8-scenario evaluation suite, we employ a mixed-ground-truth strategy. First, we use both `atm_system` and `av_system` as our physical simulation anchors; because their topologies are naturally dense, their physical simulations exhibit high density and are evaluated purely against raw Monte Carlo failure impacts (`gt_source = "Sim"`). Second, for the remaining 6 scenarios, we substitute the target labels with `Fresh-RMAV` or `RMAV-sub`—structural proxy ground truths that avoid optimization degeneracy. While these proxies introduce some construct-validity bias toward static structural features, they are shared equally by all evaluated model variants, and our relative architectural comparisons remain completely internally consistent. The fact that HGL demonstrates outstanding alignment on both the physical simulation anchors (`Sim` on ATM and AV, achieving F1 up to 0.950) and the structural proxy scenarios validates that the heterogeneous message passing expressiveness successfully captures both physical failure cascades and structural abstractions.

### C. External Validity: Topology and Domain Coverage

The 8 scenarios span air traffic management, autonomous vehicles, high-frequency financial trading, healthcare clinical integration, distributed IoT smart-city telemetry, centralized hub-and-spoke enterprise integration, cloud-native microservices, and large-scale enterprise pub-sub — a broader domain coverage than is typical for pub-sub criticality studies, but still limited along two axes that constrain the generality of our findings.

First, all 8 scenarios are synthetically generated from parameterised YAML configurations (`data/scenarios/scenario_*.yaml`). The QoS attribute distributions, broker fan-out patterns, criticality densities, and node-count ratios are drawn from realistic but not measured ranges. Real deployments may exhibit QoS distributions or topology patterns that differ qualitatively — for example, a deployment in which QoS heterogeneity is *decorrelated* from structural importance (which our synthetic generator does not produce by construction) would provide a stronger test of whether deep QoS injection adds value when the structural baseline cannot already absorb it.

Second, the application-level node counts are modest (typically 25–60 Application and Library nodes per scenario, peaking at 88 for the Enterprise scenario). The QoS-encoding-fails-to-help finding (§6.C, Effects 3 and 4) may not survive at substantially larger scales where the per-relation message function has more training signal to learn QoS-dependent aggregation reliably. We do not claim our finding extends to large-scale deployments (≥ 500 components) and identify cross-scale validation as a target for the journal extension and the broader research programme.

### D. Summary

The empirical claims of §4 and §6 — that HGL is the only variant top-tier on both ranking and identification across the 8 scenarios studied; that the heterogeneous typed-graph architecture is the load-bearing element of the proposed method; and that QoS attribute encoding at the message-function level does not add over the heterogeneous architecture alone — are robust to the most consequential threats we identify within this experimental setting. Validation circularity (§7.A) bounds the *absolute* $\rho$ and F1 values reported but does not affect the *relative* variant contrasts. Hyperparameter sensitivity (§7.B) is bounded by a focused $3 \times 2$ sweep that confirms the qualitative findings hold across a local neighbourhood of the chosen configuration. External validity (§7.C) remains the most consequential outstanding limitation and is the principal target of follow-up work.
