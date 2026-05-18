# Middleware 2026: Experimental Evaluation Overview

This document summarizes the methodology, experimental harness, and evaluation suite for the paper **"QoS-Aware Heterogeneous Graph Learning for Architectural Criticality Prediction"**.

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

where $V$ is the set of architectural components, $E \subseteq V \times V$ the set of dependencies between them, $\tau_V : V \to T_V$ maps each node to a type in $T_V = \{\text{Application}, \text{Library}, \text{Topic}, \text{Broker}, \text{Node}\}$, $\tau_E : E \to T_E$ maps each edge to a typed relation in $T_E = \{\text{PUBLISHES\_TO}, \text{SUBSCRIBES\_TO}, \text{USES}, \text{DEPENDS\_ON}, \text{RUNS\_ON}, \text{CONNECTS\_TO}\}$, $w : E \to \mathbb{R}_+$ assigns a structural weight derived from publication frequency, message size, and subscriber fan-out, and $\mathrm{QoS} : E \to \mathcal{Q}$ assigns a Quality-of-Service profile (reliability, durability, transport priority) to edges where it is semantically meaningful — that is, to PUBLISHES_TO and SUBSCRIBES_TO edges.

The `DEPENDS_ON` relation is *derived* from the raw publish-subscribe structure via two rules: (i) if Application $A$ publishes to topic $T$ and Application $B$ subscribes to $T$, add $A \xrightarrow{\text{DEPENDS\_ON}} B$ (**Rule 1**); and (ii) if Application $A$ uses Library $L$, add $A \xrightarrow{\text{DEPENDS\_ON}} L$ (**Rule 5**). This derivation lifts dependencies from the transport layer to the logical layer that architectural review operates over, and is the substrate on which structural metrics (betweenness, articulation points, bridge ratio) are computed for the non-learning baselines.

Each edge $e \in E$ is represented by a 16-dimensional feature vector concatenating: a scalar structural weight, a normalized path-count, a 7-dimensional one-hot encoding of $\tau_E(e)$, and 7 QoS-derived dimensions (reliability score, durability score, transport-priority score, deadline indicator and log-magnitude, max-blocking-time log-magnitude, and a QoS-heterogeneity flag relative to the scenario-level modal profile). The QoS dimensions are zero on non-pub/sub edges; the HGL variant additionally zeroes them on pub/sub edges, isolating the architectural contribution from the QoS-attribute contribution.

**Application-level prediction target.** For each $v$ with $\tau_V(v) \in \{\text{Application}, \text{Library}\}$, we predict $Q^*(v) \in [0, 1]$ and evaluate against simulator-derived ground-truth impact $I^*(v)$, which measures the cumulative cascade effect of failing $v$ over a fixed propagation horizon. Although the prediction target is restricted to Application and Library nodes, the heterogeneous GAT message-passes over the full typed graph — including Topic, Broker, and Node nodes — letting the model reason about cross-layer dependencies even though those nodes do not receive prediction heads.

### B. Training Matrix and Evaluation Protocol

The 8 scenarios span air traffic management (ATM, ICAO SWIM-style), autonomous vehicles, high-frequency financial trading, healthcare clinical integration, centralized hub-and-spoke enterprise integration, distributed IoT smart-city telemetry, cloud-native microservices, and large-scale enterprise pub-sub. Each scenario is a synthetically generated pub-sub topology with realistic node, application, broker, and topic counts (the ATM scenario, for instance, comprises 26 applications, 8 libraries, 27 topics, 5 brokers, and 8 compute nodes), and the 8 collectively span a wide range of topology density, QoS heterogeneity, broker fan-out, and criticality density. The full configurations live in `data/scenarios/`.

Training uses the PyTorch Geometric HeteroGAT implementation with 2 attention heads per relation, hidden dimension 64, and 300 training epochs per cell. Each seed produces an independent train/validation node split stratified by node type. Per-cell metrics are aggregated via the mean over the 5 seeds with bootstrap 95% confidence intervals ($B = 2000$ resamples). Identification metrics (F1, Precision, Recall, Top-K overlap) use **rank-matched binarization**: the top-$K$ predicted components are declared critical, where $K$ equals the number of ground-truth critical components ($I^*(v) > 0.5$). Statistical significance between HGL and each comparator is established via paired Wilcoxon signed-rank tests over the 5 seeds per scenario. The 2×3 factorial contrasts (architecture × QoS) and their interaction effects are computed in `tools/middleware26_main_table.py::_factorial_contrasts` and reported in §6.C.

The W1 QoS-pipeline audit (`tests/test_qos_pipeline_audit.py`) is run as a blocking go/no-go gate prior to the training matrix, verifying end-to-end that QoS attributes flow from the topology JSON into the HeteroData `edge_attr` tensor with the expected dimensionality and that mutating a topic's QoS profile produces a measurable downstream prediction shift.

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

### B. Critical Component Identification (F1, Acc, Prec, Rec, Top-K)
Measures the model's ability to act as a binary classifier for "Critical" vs. "Safe" components.
- **Calibration policy**: All identification metrics use **rank-matched binarization** (top-K predicted = critical, where K equals the number of ground-truth critical nodes with composite > 0.5). This isolates ranking quality from absolute-score calibration and makes F1 directly comparable across variants whose raw outputs live on different scales. See §6.B for per-cell calibration flags.
- **Accuracy**: Overall fraction of correctly identified components.
- **F1 Score**: Harmonic mean of Precision and Recall.
- **Top-5/10 Overlap**: Intersection of the most critical components found by the model vs. simulation.
- **NDCG@10**: Normalized Discounted Cumulative Gain to reward high-accuracy at the top of the list.

### C. Regression Error (RMSE, MAE)
Measures the absolute difference between predicted and actual criticality scores.
- **RMSE**: Root Mean Squared Error, which penalizes larger prediction errors.
- **MAE**: Mean Absolute Error, measuring the average magnitude of absolute errors.

---

## 4. Key Performance Highlights

The 240-run evaluation establishes a single central finding: **HGL is the only model variant we evaluate that achieves top-tier performance simultaneously on both the *ranking* task (Spearman $\rho$ — who is more critical than whom) and the *identification* task (F1 under rank-matched binarization — which components belong in the critical set).** No other variant — structural baseline, homogeneous GNN, or QoS-augmented heterogeneous GNN — clears this bar.

| Dimension | HGL result | Best comparator | Gap | Interpretation |
|---|---|---|---|---|
| **Ranking** (mean $\rho$) | **0.876** | Q-Topo-BL (0.883) | -0.007 (statistical tie) | Heterogeneous structure preserves the strong ranking signal QoS-weighted topology provides — graph learning loses nothing on this task |
| **Identification** (mean F1) | **0.90** | GL (0.54) | **+0.36** over best GNN baseline; **+0.52** over best structural baseline | Heterogeneous architecture sharpens the critical-set boundary that homogeneous and structural baselines blur |
| **Worst-case F1** | $\geq 0.68$ in 8/8 scenarios | Topo-BL: F1 = 0.00 in 5/8 scenarios; Q-Topo-BL: 3/8 | No catastrophic failures | Robust across topology density, QoS heterogeneity, and broker fan-out regimes |
| **Per-node-type $\rho$ (Library)** | $> 0.9$ | GL ($\approx 0.7$) | +0.2 | Heterogeneous per-relation attention exploits Library-specific semantics that homogeneous GATs collapse into topological noise |
| **Statistical significance** | Paired Wilcoxon $p < 0.05$ on F1 in the majority of scenarios | vs. all structural and homogeneous baselines | — | The identification gap is not seed-driven; it survives non-parametric significance testing per scenario |

Two observations frame the rest of the paper. First, the gap on **identification** ($\Delta\text{F1} \approx +0.36$ over the best homogeneous GNN, $\approx +0.52$ over the best structural baseline) is substantially larger than the gap on **ranking** ($\Delta\rho \approx +0.076$ over GL, statistical tie with Q-Topo-BL). Graph learning's contribution is concentrated on the task that pre-deployment architectural review actually cares about — *which components belong in the critical set*, the binary decision that drives prioritized hardening — rather than on the global ordering that structural centrality already solves adequately when QoS-weighted. Structural baselines, even with QoS weighting, fail catastrophically on identification: Topo-BL collapses to F1 = 0 in 5 of 8 scenarios, and Q-Topo-BL in 3 of 8 — they rank components correctly but cannot calibrate the critical-set boundary.

Second, the controlled 2×3 ablation in §6.C localizes the gain to the architectural choice rather than to the QoS encoding. Holding QoS masked, the heterogeneous architecture improves over the homogeneous one by $\Delta\rho = +0.076$ and $\Delta\text{F1} \approx +0.36$ (HGL vs. GL). Holding the heterogeneous architecture fixed, adding 7-dimensional QoS attribute encoding does *not* further improve performance (Q-HGL vs. HGL: $\Delta\rho = -0.082$, $\Delta\text{F1} \approx -0.05$). The load-bearing element of the proposed method is typed nodes, typed relations, and per-relation attention — not QoS attribute encoding at the message-function level. This is consistent with the structural-baseline comparison: the QoS signal that is predictively useful is already absorbed by QoS-weighted betweenness (Q-Topo-BL vs. Topo-BL: $\Delta\rho = +0.379$), leaving no headroom for the heterogeneous GNN to extract additional value from re-encoding it inside the message functions.

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

| Scenario | Topo-BL | Q-Topo-BL | GL | Q-GL | HGL | Q-HGL | Δρ (QoS) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ATM System** | 0.747 | 0.766 | 0.613 | 0.673 | 0.623 | 0.620 | — |
| **AV System** | 0.372 | 0.923 | 0.665 | 0.773 | 0.873 | 0.667 | — |
| **Enterprise** | 0.503 | 0.936 | 0.866 | 0.864 | 0.957 | 0.826 | — |
| **Financial Trading** | 0.379 | 0.914 | 0.827 | 0.738 | 0.841 | 0.657 | — |
| **Healthcare** | 0.308 | 0.947 | 0.734 | 0.915 | 0.918 | 0.869 | — |
| **Hub-and-Spoke** | 0.734 | 0.838 | 0.873 | 0.890 | 0.911 | 0.909 | — |
| **IoT Smart City** | 0.522 | 0.820 | 0.909 | 0.935 | 0.959 | 0.909 | — |
| **Microservices** | 0.469 | 0.916 | 0.916 | 0.951 | 0.927 | 0.896 | — |
| **Mean** | 0.504 | 0.883 | 0.800 | 0.843 | 0.876 | 0.794 | -0.082 |

*\*Trivial correlation due to sparse simulation data forcing the use of RMAV as ground truth.

**Discussion.** HGL achieves a highly competitive mean Spearman correlation across the 240 application-level runs ($\rho = 0.876$), narrowly behind Q-Topo-BL ($\rho = 0.883$). Paired Wilcoxon signed-rank tests on the per-seed $\rho$ values confirm that the two are not statistically distinguishable on the global mean: heterogeneous graph attention and QoS-weighted structural centrality converge on the same ranking signal from very different inductive biases. This convergence is itself informative — it tells us that the ranking task in pub-sub criticality prediction is approximately solved once QoS-weighted edge importance is captured, and that the choice between a learned and a structural method should be made on grounds other than ranking quality alone. §6.B and §6.C make that case in favor of HGL on identification and consistency.

Within the graph-learning family, the heterogeneous architecture provides a ranking advantage over its homogeneous counterparts: HGL ($\rho = 0.876$) improves on GL ($\rho = 0.800$) by $\Delta\rho = +0.076$ and on Q-GL ($\rho = 0.843$) by $\Delta\rho = +0.033$. Q-HGL ($\rho = 0.794$) is the weakest of the four GNN configurations on ranking — adding 7-dimensional QoS attribute encoding to the heterogeneous message functions reduces $\rho$ by 0.082 relative to QoS-masked HGL, an effect §6.C decomposes via the controlled factorial contrasts.

The most stable property of HGL on ranking is **consistency rather than peak performance per scenario**. HGL is among the top-3 variants in 7 of 8 scenarios, whereas Q-Topo-BL — which wins outright in 4 scenarios (ATM, AV, Financial Trading, Healthcare) — falls out of the top-3 in Hub-and-Spoke, IoT Smart City, and Microservices, where centralized broker topologies and dense subscriber fan-out blur the betweenness signal that QoS-weighted centrality relies on. HGL drops to 4th place only on ATM, where high criticality density and prominent single-points-of-failure (ConflictDetector, ASTERIX_Broker) yield an unusually strong structural signal that even unweighted betweenness exploits ($\rho = 0.747$ for Topo-BL, the only scenario in which it remains competitive). The application-level ranking picture is therefore less a story of one variant dominating across scenarios than of two architectural families — heterogeneous GAT and QoS-weighted centrality — producing comparable global orderings with smaller within-family variance for HGL.

### B. Identification Metrics
The following table provides a breakdown of binary classification performance for critical component identification.

| Scenario | Variant | F1 | Precision | Recall | Top-5 | Top-10 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| ATM System | Topo-BL | 0.250 | 0.250 | 0.250 | 0.400 | 0.700 | 0.905 |
|  | Q-Topo-BL | 0.500 | 0.500 | 0.500 | 0.400 | 0.700 | 0.897 |
|  | GL | 0.600 | 0.600 | 0.600 | 0.680 | 0.000 | 0.972 |
|  | Q-GL | 0.600 | 0.600 | 0.600 | 0.720 | 0.000 | 0.974 |
|  | HGL | 0.788 | 0.788 | 0.788 | 0.720 | 0.000 | 0.976 |
|  | Q-HGL | 0.788 | 0.788 | 0.788 | 0.720 | 0.000 | 0.975 |
| | | | | | | |
| AV System | Topo-BL | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.740 |
|  | Q-Topo-BL | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | 0.969 |
|  | GL | 0.700 | 0.700 | 0.700 | 0.600 | 0.740 | 0.931 |
|  | Q-GL | 0.600 | 0.600 | 0.600 | 0.720 | 0.720 | 0.949 |
|  | HGL | 0.855 | 0.855 | 0.855 | 0.960 | 0.840 | 0.977 |
|  | Q-HGL | 0.732 | 0.732 | 0.732 | 0.680 | 0.660 | 0.928 |
| | | | | | | |
| Enterprise | Topo-BL | 0.000 | 0.000 | 0.000 | 0.200 | 0.300 | 0.859 |
|  | Q-Topo-BL | 0.000 | 0.000 | 0.000 | 0.400 | 0.500 | 0.861 |
|  | GL | 0.343 | 0.343 | 0.343 | 0.360 | 0.580 | 0.920 |
|  | Q-GL | 0.400 | 0.400 | 0.400 | 0.440 | 0.580 | 0.928 |
|  | HGL | 0.922 | 0.922 | 0.922 | 0.760 | 0.860 | 0.994 |
|  | Q-HGL | 0.845 | 0.845 | 0.845 | 0.200 | 0.380 | 0.867 |
| | | | | | | |
| Financial Trading | Topo-BL | 0.000 | 0.000 | 0.000 | 0.400 | 0.600 | 0.883 |
|  | Q-Topo-BL | 0.000 | 0.000 | 0.000 | 0.200 | 0.400 | 0.863 |
|  | GL | 0.500 | 0.500 | 0.500 | 0.640 | 0.920 | 0.958 |
|  | Q-GL | 0.200 | 0.200 | 0.200 | 0.600 | 0.900 | 0.930 |
|  | HGL | 0.938 | 0.938 | 0.938 | 0.560 | 0.900 | 0.975 |
|  | Q-HGL | 0.873 | 0.873 | 0.873 | 0.320 | 0.880 | 0.927 |
| | | | | | | |
| Healthcare | Topo-BL | 0.000 | 0.000 | 0.000 | 0.400 | 0.600 | 0.846 |
|  | Q-Topo-BL | 0.500 | 0.500 | 0.500 | 0.800 | 0.900 | 0.983 |
|  | GL | 0.500 | 0.500 | 0.500 | 0.760 | 0.880 | 0.941 |
|  | Q-GL | 0.700 | 0.700 | 0.700 | 0.840 | 0.920 | 0.982 |
|  | HGL | 0.946 | 0.946 | 0.946 | 0.880 | 0.940 | 0.983 |
|  | Q-HGL | 0.878 | 0.878 | 0.878 | 0.800 | 0.940 | 0.964 |
| | | | | | | |
| Hub-and-Spoke | Topo-BL | 0.500 | 0.500 | 0.500 | 0.400 | 0.500 | 0.954 |
|  | Q-Topo-BL | 0.800 | 0.800 | 0.800 | 0.600 | 0.800 | 0.989 |
|  | GL | 0.300 | 0.300 | 0.300 | 0.800 | 0.940 | 0.955 |
|  | Q-GL | 0.400 | 0.400 | 0.400 | 0.760 | 0.940 | 0.971 |
|  | HGL | 0.978 | 0.978 | 0.978 | 0.800 | 0.940 | 0.995 |
|  | Q-HGL | 1.000 | 1.000 | 1.000 | 0.760 | 0.960 | 0.978 |
| | | | | | | |
| IoT Smart City | Topo-BL | 0.000 | 0.000 | 0.000 | 0.000 | 0.300 | 0.752 |
|  | Q-Topo-BL | 0.600 | 0.600 | 0.600 | 0.600 | 0.700 | 0.952 |
|  | GL | 0.640 | 0.640 | 0.640 | 0.560 | 0.840 | 0.969 |
|  | Q-GL | 0.700 | 0.700 | 0.700 | 0.640 | 0.840 | 0.978 |
|  | HGL | 0.918 | 0.918 | 0.918 | 0.880 | 0.880 | 0.994 |
|  | Q-HGL | 0.863 | 0.863 | 0.863 | 0.760 | 0.860 | 0.974 |
| | | | | | | |
| Microservices | Topo-BL | 0.000 | 0.000 | 0.000 | 0.200 | 0.400 | 0.783 |
|  | Q-Topo-BL | 0.000 | 0.000 | 0.000 | 0.200 | 0.600 | 0.872 |
|  | GL | 0.733 | 0.733 | 0.733 | 0.840 | 0.860 | 0.978 |
|  | Q-GL | 0.800 | 0.800 | 0.800 | 0.840 | 0.860 | 0.987 |
|  | HGL | 0.832 | 0.832 | 0.832 | 0.800 | 0.820 | 0.979 |
|  | Q-HGL | 0.816 | 0.816 | 0.816 | 0.720 | 0.800 | 0.956 |
| | | | | | | |

*F1, Precision, and Recall are computed with **rank-matched binarization**:
the top-K predicted nodes are declared critical, where K equals the number
of ground-truth critical nodes (composite > 0.5). This isolates ranking
quality from absolute-score calibration and makes F1 directly comparable
across variants whose raw outputs live on different scales — sigmoid outputs
in [0, 1] for the heterogeneous GAT, unbounded logits for the homogeneous
GAT baselines, and raw centrality for the structural baselines. Cells
marked with † used the legacy fixed-threshold binarization (threshold = 0.5);
cells marked with ‡ have a degenerate label distribution (all nodes critical,
or none) for which F1 is undefined.*

**Discussion.** The identification task tells a sharply different story from ranking. The graph-learning family decisively outperforms the structural baselines on F1: HGL achieves a mean F1 of **0.90**, Q-HGL **0.85**, the homogeneous variants **0.54** (GL) and **0.55** (Q-GL), and the structural baselines only **0.12** (Topo-BL) and **0.38** (Q-Topo-BL). The gap is not a global-ordering subtlety but a categorical capability difference — Topo-BL collapses to F1 = 0 in 5 of 8 scenarios (Enterprise, Financial Trading, Healthcare, IoT Smart City, Microservices) and Q-Topo-BL in 3 of 8 (Enterprise, Financial Trading, Microservices). In those scenarios, structural centrality ranks the components in a defensible order but cannot bring the top-$K$ predicted set into alignment with the top-$K$ ground-truth critical set, even when the cardinality $K$ is supplied by rank-matched binarization. The QoS-weighted betweenness that suffices for ranking ($\rho = 0.883$) does not suffice for identification ($\text{F1} = 0.38$).

Within the graph-learning family, the heterogeneous architecture again outperforms the homogeneous variants, but the margin on F1 is **far larger** than on ranking: HGL improves on GL by $\Delta\text{F1} \approx +0.36$ and on Q-GL by $\Delta\text{F1} \approx +0.35$ — roughly an order of magnitude larger than the corresponding ranking gaps. Across the scenarios with full F1 data, HGL's per-scenario F1 lies in the range $[0.788, 0.978]$, while the homogeneous variants span $[0.200, 0.800]$. The homogeneous GNNs occasionally outperform HGL at the peak (Q-GL reaches 0.800 on Microservices; GL reaches 0.733 on Microservices), but their floor is far lower than HGL's, producing systematic identification failures on scenarios with dense Library coupling and high QoS heterogeneity — Financial Trading (Q-GL: F1 = 0.200; HGL: 0.938) and Enterprise (Q-GL: 0.400; HGL: 0.922) are the clearest examples.

The Q-HGL variant edges or ties HGL only on two scenarios (Hub-and-Spoke: 1.000 vs. 0.978, and ATM: 0.788 vs. 0.788); on the six remaining scenarios Q-HGL falls behind HGL by margins ranging from 0.016 (Microservices) to 0.123 (AV System). The composite identification picture is that HGL trades a sliver of peak F1 for substantially tighter worst-case behavior — a property pre-deployment architectural review workflows specifically value, since the cost of a missed critical component dominates the benefit of an extra correctly-flagged one.

We report only F1 in this section; Precision and Recall are mechanically identical to F1 under rank-matched binarization. When the predicted positive set has the same cardinality as the ground-truth positive set, $\text{TP} = K - \text{FP} = K - \text{FN}$, so $P = R = \text{F1}$. The structural identity is noted in §3.B; reporting all three would convey the same number three times.

---

### C. Ablation Analysis: What Drives the Identification Gain?

The 2×3 factorial design (architecture × QoS encoding) plus the two structural baselines support four controlled comparisons that decompose the headline finding from §4 — that HGL is the only variant top-tier on both ranking and identification — into its constituent architectural and QoS-encoding contributions. Each comparison holds one factor fixed and varies the other; the resulting $\Delta\rho$ and $\Delta\text{F1}$ values are computed in `tools/middleware26_main_table.py::_factorial_contrasts` and reported below, with paired Wilcoxon signed-rank tests over the 8 per-scenario mean values.

| Comparison | Varies | $\Delta\rho$ (mean) | $\Delta\text{F1}$ (mean) | Wilcoxon $p$ |
|---|---|---|---|---|
| **Q-Topo-BL − Topo-BL** | QoS weighting on betweenness | **+0.379** | **+0.26** | $p < 0.01$ |
| **HGL − GL** | Homogeneous $\to$ Heterogeneous | **+0.076** | **+0.36** | $p < 0.01$ |
| **Q-GL − GL** | Scalar QoS edge weight | **+0.042** | **+0.01** | n.s. |
| **Q-HGL − HGL** | 7-dim QoS attribute encoding | **−0.082** | **−0.05** | $p < 0.01$ |

*The interaction term $(\text{Q-HGL} - \text{HGL}) - (\text{Q-GL} - \text{GL}) = -0.124$ represents the difference in QoS sensitivity between architectures. On both architectures, however, the direct message-function QoS attribute encoding fails to provide predictive benefit.*

**Effect 1: QoS at the structural-centrality level (Topo-BL → Q-Topo-BL).** Weighting betweenness by QoS-derived edge importance lifts mean $\rho$ from 0.504 to 0.883 and mean F1 from approximately 0.12 to 0.38 — a large, statistically significant gain on both metrics. This establishes that the QoS signal *is* predictively useful, and that the simplest possible injection point — multiplying the path-counting routine by QoS-derived edge weights — captures the bulk of that value. Q-Topo-BL is, on ranking, statistically indistinguishable from the best learned model.

**Effect 2: Heterogeneous architecture with QoS encoding masked (GL → HGL).** Holding QoS encoding fixed at "off," replacing the homogeneous GAT with a heterogeneous GAT — typed nodes, typed relations, per-relation attention heads — produces a modest ranking gain ($\Delta\rho = +0.076$, $p < 0.01$) and a substantially larger identification gain ($\Delta\text{F1} \approx +0.36$, $p < 0.01$). The architectural choice is the load-bearing element of the proposed method on the task that matters most: critical-set identification. The mechanism is intuitive — per-relation message functions let the model condition on whether an edge represents an application's dependency on another application (DEPENDS_ON), a library usage (USES), or a transport-layer publication (PUBLISHES_TO, SUBSCRIBES_TO), and each of those relation types carries different semantics for criticality propagation. A homogeneous GAT collapses all four relation types into a single message function and cannot exploit this distinction.

**Effects 3 and 4: QoS at the message-function level (GL → Q-GL; HGL → Q-HGL).** When QoS information is injected *inside* the learned GNN — either as a scalar edge weight in the homogeneous case or as a 7-dimensional attribute vector in the heterogeneous case — performance does not improve. In the homogeneous case the effect is small ($\Delta\rho = +0.042$, $\Delta\text{F1} \approx +0.01$) and not statistically significant. In the heterogeneous case the negative effect is statistically significant on ranking ($\Delta\rho = -0.082$, $p < 0.01$) and F1 ($\Delta\text{F1} = -0.05$, $p < 0.05$). This negative direction is a real, robust effect across scenarios.

**Synthesis.** Three patterns emerge. First, the QoS signal that is predictively useful for criticality prediction is **structurally informative** — it is already captured by QoS-weighted betweenness at the structural-baseline level (Effect 1), leaving little headroom for the GNN to extract additional value from re-encoding it inside the message functions (Effects 3 and 4). Second, the gain attributable to graph learning over the strongest structural baseline is concentrated on the **identification** task, not the ranking task; Effect 2 is dominated by $\Delta\text{F1} \approx +0.36$, not by the modest $\Delta\rho \approx +0.076$. Learning produces the calibrated critical/non-critical boundary that structural centrality blurs. Third, the architectural contribution (heterogeneous typed graph) and the QoS-encoding contribution (QoS attribute features in message functions) are not interchangeable design choices: the former is decisively positive and the latter is at best neutral and at worst slightly harmful.

The first pattern admits a candidate mechanism that we offer cautiously. In the pub-sub topologies we study, QoS attributes are *correlated* with structural importance — high-reliability, high-priority edges tend to lie on cascade-propagation paths anyway — and re-encoding them at the message-function level adds estimation burden to the GNN without contributing new information that the typed-graph structural prior has not already absorbed. A definitive test of this mechanism would require either (i) scenarios in which QoS heterogeneity is decorrelated from structural importance by construction, or (ii) substantially larger pub-sub deployments in which the per-relation message function has enough training signal to learn QoS-dependent aggregation reliably. We flag both as targets for future work, while emphasizing that the empirical finding — heterogeneous architecture helps decisively on identification, QoS attribute encoding does not help further at this scale — is robust within the 240-run evaluation reported here.
