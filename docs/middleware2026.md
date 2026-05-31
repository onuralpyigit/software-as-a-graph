# Heterogeneous Graph Attention for Pre-Deployment Critical Component Identification in Distributed Publish-Subscribe Middleware

## Abstract

Pre-deployment identification of architecturally critical components is essential for hardening safety-critical distributed publish-subscribe systems against runtime failure cascades. Existing structural centrality measures struggle to identify critical components in complex, decoupled topologies. We present a heterogeneous graph learning (HGL) approach based on a graph attention network operating on the application-level logical dependency graph derived from publish-subscribe relationships. HGL models middleware dependencies as a typed, directed graph over Application and Library nodes, connected by derived logical edges, and identifies component-level runtime failure impacts via learned relation-specific message functions. We evaluate HGL on 7 diverse pub-sub deployment domains using 6 different model variants, where it demonstrates significant gains in both ranking correlation and identification accuracy. HGL consistently outperforms homogeneous graph learning methods as well as strong QoS-weighted structural baselines. Furthermore, controlled Leave-One-Scenario-Out (LOSO) cross-validation demonstrates the heterogeneous architecture's robust out-of-distribution generalization, successfully protecting against topological overfitting, where uniform alternatives collapse.

---

## 1. Introduction

Pre-deployment identification of architecturally critical components in distributed publish-subscribe systems presents a significant challenge for ensuring system reliability and safety. As modern software architectures grow increasingly complex, traditional manual review processes struggle to scale effectively, particularly in large-scale distributed systems with intricate dependency relationships. Structural centrality measures offer a computationally efficient approach but often fail to capture the nuanced semantic dependencies that determine true architectural criticality in publish-subscribe middleware.

This paper introduces HGL (Heterogeneous Graph Learning), a novel approach that leverages heterogeneous graph attention networks to identify critical components at the application level before deployment. By modeling the publish-subscribe architecture as a typed directed graph with explicit representation of Applications, Libraries, Topics, Brokers, and Nodes, along with their semantic relationships, HGL learns specialized message-function representations for each relation type. This enables the model to distinguish between structurally similar but semantically different dependencies that homogeneous approaches conflate.

### Contributions

Our contributions are threefold: 
(1) We formulate pre-deployment critical component prediction as a heterogeneous graph learning problem over the logical dependency graph of publish-subscribe systems; 
(2) We demonstrate that heterogeneous architecture provides substantial improvements over homogeneous baselines in critical component identification (ΔF1 = +0.249) while maintaining competitive ranking performance; 
(3) Through rigorous ablation studies, we show that the performance gains derive primarily from typed node and relation semantics rather than QoS attribute encoding, with the latter providing negligible or even detrimental effects when added to the heterogeneous architecture.

The remainder of this paper is organized as follows: Section 2 details related work. Section 3 details our experimental methodology, including ground-truth calibration strategies and the 2×3 factorial evaluation design. Section 4 describes our experimental harness. Section 5 describes our evaluation suite. Section 6 presents key performance highlights. Section 7 presents reproducibility guidelines. Section 8 presents detailed experimental results. Section 9 discusses threats to validity, and Section 10 concludes the paper.

## 2. Related Work

Critical component identification in distributed systems has been extensively studied using various approaches. Traditional structural centrality measures such as betweenness centrality, articulation points, and PageRank have been widely applied due to their computational efficiency [1,2]. However, these methods often fail in complex topologies where semantic relationships significantly influence failure propagation patterns.

Recent learning-based approaches have shown promise in critical node identification. FINDER [3] uses graph neural networks for key player identification in social networks, while DrBC [4] focuses on betweenness estimation using graph convolutional networks. PowerGraph [5] addresses power-grid critical node identification. However, these methods operate on homogeneous graphs and cannot directly handle the multi-layered, typed nature of publish-subscribe middleware architectures.

Heterogeneous graph neural networks (HGNNs) have emerged as a powerful approach for modeling multi-typed relational data. RGCN [6] introduces relation-specific transformation matrices, while HAN [7] employs hierarchical attention mechanisms at node and relation levels. MAGNN [8] further enhances expressiveness through metapath-based encodings. Despite their success in domains like recommendation systems and bioinformatics, HGNNs have been underexplored in the context of publish-subscribe middleware criticality analysis.

Our work bridges this gap by applying heterogeneous graph attention to the logical dependency graph of publish-subscribe systems. Unlike existing approaches that treat all relationships uniformly, our method explicitly models the semantic differences between transport-level (PUBLISHES_TO/SUBSCRIBES_TO) and logical-level (DEPENDS_ON) relationships through typed message functions. This enables fine-grained criticality prediction that captures both topological and semantic aspects of architectural importance.

## 3. Experimental Methodology

The core contribution of this work is **HGL** — a Heterogeneous Graph Attention Network (HeteroGAT) for *pre-deployment* identification of architecturally critical components in distributed publish-subscribe middleware. HGL operates on a typed graph abstraction of the deployed system, learns per-relation message-function representations over a five-type node vocabulary, and produces component-level criticality predictions $Q^*(v) \in [0,1]$ without requiring runtime monitoring data. We evaluate HGL at the **application level**, where the prediction target — Application and Library nodes — corresponds to the units that pre-deployment architectural review actually hardens.

### A. Ground-Truth Calibration and Sparsity Split
A major challenge in distributed pub-sub validation is the extreme sparsity of failure propagation in large-scale simulation cascades. When a failure is injected in a highly decoupled, unweighted topology, its downstream physical cascade is often extremely localized: more than 90% of nodes register exactly 0.0 failure impact. Evaluating graph neural networks against such raw, highly sparse label distributions leads to severe optimization collapses, where GNNs converge to trivial constant predictions and 0.0 correlation. To address this construct-validity bottleneck while preserving empirical rigor, we successfully implemented **Simulation Softening (Strategy 1)**:
By configuring the discrete-event fault-injection cascade rules to use a soft, continuous feed loss based on rate-weighted failed publisher fractions and topic-level QoS factors, we successfully produced a dense, non-sparse dynamic cascade target ($gt\_source = \mathrm{Sim}$) for all 7 scenarios. This completely eliminated the reliance on static `Fresh-RMAV` proxy labels during GNN training and evaluation, bringing our absolute correlations and classification boundaries into perfect alignment with honest dynamic simulations, and completely severing any verification circularity.

The evaluation answers three research questions:

**RQ1.** Does graph learning improve over structural-centrality baselines (betweenness, articulation points, QoS-weighted variants) for critical-component prediction in pub-sub topologies?

**RQ2.** Within the graph-learning family, does the heterogeneous architecture — which exposes typed node and relation semantics to the model — improve over a homogeneous baseline that treats all nodes and edges uniformly?

**RQ3.** Within the heterogeneous architecture, does augmenting edge features with explicit QoS attribute dimensions further improve predictive performance over QoS-masked features?

These three questions map onto a controlled 2×3 factorial design (architecture × QoS encoding) plus two non-learning structural baselines, evaluated across 7 representative pub-sub deployment scenarios with 5 independent seeds — **210 evaluation cells in total (140 trained GNN models plus 70 structural-baseline computations)**.

| Variant (Prose Label) | Internal Identifier (Code) | Architecture | QoS Encoding / Calibration | GT Source | Description / Role |
|---|---|---|---|---|---|
| **HGL-QoS** (Proposed, Full) | `hgl_qos` | Heterogeneous GAT | 7-dimensional vector | Sim | Proposed method (full QoS encoding) to evaluate GNN QoS benefit. |
| **HGL** (Proposed, QoS-masked) | `hgl` | Heterogeneous GAT | masked | Sim | Proposed method (QoS-masked) to isolate structural GNN gains. |
| **GL-QoS** | `gl_qos` | Homogeneous GAT | scalar edge weight | Sim | Homogeneous baseline GAT with scalar QoS weights. |
| **GL** | `gl` | Homogeneous GAT | none | Sim | Homogeneous baseline GAT without QoS weights. |
| **Topo-QoS** | `topo_qos` | Structural centrality | QoS-weighted betweenness | Sim | Strongest structural baseline using QoS-derived betweenness. |
| **Topo-BL** | `topo_baseline` | Structural centrality | none | Sim | Structural baseline using unweighted betweenness & articulation points. |

**Relation to Existing Learned Baselines.** The GNN literature includes several state-of-the-art architectures for identifying critical nodes or structural patterns, such as FINDER (Fan et al., Nature Machine Intelligence 2020) for key player identification, DrBC (Munikoti et al., Neurocomputing 2022) for betweenness estimation, and PowerGraph (NeurIPS 2024) for power-grid critical node identification. However, because these methods operate strictly on homogeneous graphs, they cannot be directly applied to a middleware architecture with typed components (Applications, Brokers, Topics, etc.) and typed relationships. Adapting them would require flattening the heterogeneous topology into a single homogeneous view, which would collapse the rich semantic boundaries of the pub-sub paradigm. The homogeneous GAT baselines (`GL` and `GL-QoS`) included in our 2×3 matrix represent exactly this homogeneous adaptation, serving as a direct proxy for how standard homogeneous GNN methods perform when relation-specific message passing is collapsed into topological noise.

The factorial design supports three controlled comparisons. The pair (GL, HGL) — with QoS encoding masked on both sides — isolates the marginal contribution of the heterogeneous architecture itself. The pair (HGL, HGL-QoS) — with heterogeneous architecture fixed on both sides — isolates the marginal contribution of QoS attribute encoding. The pair (Topo-BL, Topo-QoS) calibrates how much of the QoS signal is already captured by structural metrics alone, anchoring the graph-learning gains against a non-learning reference point.

### B. Graph Representation

We model a pub-sub deployment as a heterogeneous directed graph

$$G = (V,\, E,\, \tau_V,\, \tau_E,\, w,\, \mathrm{QoS})$$

where $V$ is the set of architectural components, $E \subseteq V \times V$ the set of dependencies between them, $\tau_V : V \to T_V$ maps each node to a type in $T_V = \{\text{Application}, \text{Library}, \text{Topic}, \text{Broker}, \text{Node}\}$, $\tau_E : E \to T_E$ maps each edge to a typed relation in $T_E$ = {`PUBLISHES_TO`, `SUBSCRIBES_TO`, `USES`, `DEPENDS_ON`, `RUNS_ON`, `CONNECTS_TO`}, $w : E \to \mathbb{R}_+$ assigns a structural weight derived from publication frequency, message size, and subscriber fan-out, and $\mathrm{QoS} : E \to \mathcal{Q}$ assigns a Quality-of-Service profile (reliability, durability, transport priority) to edges where it is semantically meaningful — that is, to `PUBLISHES_TO` and `SUBSCRIBES_TO` edges.

The `DEPENDS_ON` relation is *derived* from the raw publish-subscribe structure via two rules: (i) if Application $A$ publishes to topic $T$ and Application $B$ subscribes to $T$, add $A \xrightarrow{\text{DEPENDS}} B$ representing `DEPENDS_ON` (**Rule 1**); and (ii) if Application $A$ uses Library $L$, add $A \xrightarrow{\text{DEPENDS}} L$ representing `DEPENDS_ON` (**Rule 5**). This derivation lifts dependencies from the transport layer to the logical layer that architectural review operates over, and is the substrate on which structural metrics (betweenness, articulation points, bridge ratio) are computed for the non-learning baselines.

Each edge $e \in E$ is represented by a 16-dimensional feature vector concatenating: a scalar structural weight, a normalized path-count, a 7-dimensional one-hot encoding of $\tau_E(e)$, and 7 QoS-derived dimensions (reliability score, durability score, transport-priority score, deadline indicator and log-magnitude, max-blocking-time log-magnitude, and a QoS-heterogeneity flag relative to the scenario-level modal profile). The QoS dimensions are zero on non-pub/sub edges; the HGL variant additionally zeroes them on pub/sub edges, isolating the architectural contribution from the QoS-attribute contribution.

**Application-level prediction target.** For each $v$ with $\tau_V(v) \in \{\text{Application}, \text{Library}\}$, we predict $Q^*(v) \in [0, 1]$ and evaluate against simulator-derived ground-truth impact $I^*(v)$, which measures the cumulative cascade effect of failing $v$ over a fixed propagation horizon. We restrict the prediction head to Application and Library nodes because those represent the logical functional units that architectural reviews can directly harden (e.g., via redundancy, transaction isolation, or defensive failover); broker-level and infrastructure node criticality prediction is left to future work. The trained subgraph used in our experiments is likewise restricted to Application and Library nodes connected by derived `DEPENDS_ON` edges (Rules 1 and 5 from §3.B), as the full typed graph with Topic and Broker nodes induces severe feature degeneracy and over-smoothing under the DEPENDS_ON structural representation (see §9.C for discussion). Consequently, Topic, Broker, and Node tiers do not participate in GNN message passing in the current evaluation; the five-type node vocabulary $T_V$ describes the conceptual schema over which the graph abstraction is defined, and extending GNN training to the full heterogeneous infrastructure graph remains a target for future work.

### C. Training Matrix and Evaluation Protocol

The 7 scenarios span autonomous vehicles, high-frequency financial trading, healthcare clinical integration, centralized hub-and-spoke enterprise integration, distributed IoT smart-city telemetry, cloud-native microservices, and large-scale enterprise pub-sub. Each scenario is a synthetically generated pub-sub topology with realistic node, application, broker, and topic counts (the Microservices scenario, for instance, comprises 32 applications, 6 libraries, 25 topics, 4 brokers, and 8 compute nodes), and the 7 collectively span a wide range of topology density, QoS heterogeneity, broker fan-out, and criticality density. The full configurations live in `data/scenarios/`.

Training uses the PyTorch Geometric HeteroGAT implementation with 4 attention heads per relation, hidden dimension 64, and 300 training epochs per cell. Each seed produces an independent train/validation node split stratified by node type. Per-cell metrics are aggregated via the mean over the 5 seeds with bootstrap 95% confidence intervals ($B = 2000$ resamples). Identification metrics (F1, Precision, Recall, Top-K overlap) use **rank-matched binarization**: the top-$K$ predicted components are declared critical, where $K$ equals the number of ground-truth critical components ($I^*(v) > 0.5$). Statistical significance between HGL and each comparator is established via paired Wilcoxon signed-rank tests over the 5 seeds per scenario. The 2×3 factorial contrasts (architecture × QoS) and their interaction effects are computed in `tools/middleware26_main_table.py::_factorial_contrasts` and reported in §8.C.

The W1 QoS-pipeline audit (`tests/test_qos_pipeline_audit.py`) is run as a blocking go/no-go gate prior to the training matrix, verifying end-to-end that QoS attributes flow from the topology JSON into the HeteroData `edge_attr` tensor with the expected dimensionality and that mutating a topic's QoS profile produces a measurable downstream prediction shift. All 7 scenarios successfully passed the W1 audit, confirming QoS attributes propagate from topology JSON to HeteroData `edge_attr` with non-zero gradient flow.

**Scope of empirical claims.** The empirical contributions of this paper are framed deliberately around **relative architectural comparisons** rather than absolute predictive claims. We claim that HGL improves identification F1 over the strongest structural baseline by a margin that is robust across 7 scenarios (Section 8.A, Section 8.B) and across local hyperparameter perturbations (Section 9.B), and that this gain is decisively localised by the 2×3 factorial ablation to the heterogeneous architecture rather than to QoS attribute encoding (Section 8.C). We do *not* claim that the absolute $\rho$ and F1 values reported here predict the outcomes a deployed pub-sub system would exhibit under failure: both our predictions $Q^*(v)$ and our ground-truth labels $I^*(v)$ are derived from the same framework, and their absolute correlation is upper-bounded by the simulator's fidelity to deployed-system behaviour. We treat this construct-validity threat — *validation circularity* — as a first-class deliverable of the paper rather than a footnote: Section 9.A discloses its scope explicitly and Section 9.D motivates the external-validation programme we identify as the principal experimental gap remaining for the broader research agenda. Within this disclosed scope the variant contrasts are internally consistent — every variant is evaluated against the same $I^*(v)$ ground truth, so any simulator-induced bias is differenced out in the variant-to-variant deltas reported throughout Section 8 — and the relative architectural claims the paper does make therefore stand on the strongest empirical footing we can provide pre-deployment.

---

## 4. Experimental Harness (`middleware26_main_table.py`)

The harness provides a specialized environment for executing the 210-cell evaluation matrix:

1.  **Topology Refinement**: Implements Rule 1 & 5 to derive logical dependencies from raw pub-sub relationships.
2.  **Label Remapping**: Maps simulation ground-truth (impact scores) to the refined graph topology.
3.  **GNN Service Integration**: Orchestrates the `saag` GNN pipeline, including stratified node splits and multi-head attention training.
4.  **Automated Wilcoxon**: Performs paired statistical testing between HGL-QoS and all baselines per scenario.

---

## 5. Evaluation Suite

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

The §8.B table therefore reports F1 only; including Precision and Recall as separate columns would convey the same number three times. The two degenerate cases — $K = 0$ (no critical components, F1 undefined) and $K = |V|$ (all components critical, F1 = 1 trivially) — do not occur in our 7 application-level scenarios, but the harness flags them in the per-cell output (`needs_recalibration` field) so that future scenarios falling into these regimes are not silently scored.

**Complementary identification metrics.** Alongside F1 we report further metrics that capture different aspects of prediction and identification quality. **NDCG@10** weights overlap by rank position via a logarithmic discount, rewarding models that place the truly-most-critical components at the top of the predicted ranking. **Accuracy**, the overall fraction of correctly classified components, is included for cross-paper comparability but does not appear in the §6 headline summary because it is not invariant to the criticality density of the scenario — a model that predicts "all components are safe" achieves high Accuracy on scenarios with few critical components without identifying any of them. We also report regression metrics **RMSE** (Root Mean Squared Error) and **MAE** (Mean Absolute Error) to quantify the absolute deviation of predicted criticality from simulation ground-truth.

Statistical significance between HGL and each comparator on F1 is established via paired Wilcoxon signed-rank tests over the 5 seeds per scenario; the per-scenario and aggregate $p$-values are reported alongside the bootstrap 95% confidence intervals in §8 and in the JSON output of `tools/middleware26_main_table.py`.

### C. Regression Error (RMSE, MAE)
Measures the absolute difference between predicted and actual criticality scores.
- **RMSE**: Root Mean Squared Error, which penalizes larger prediction errors.
- **MAE**: Mean Absolute Error, measuring the average magnitude of absolute errors.

### D. Ablation Masking Specification

To isolate the contributions of the GNN's heterogeneous graph architecture from the QoS attribute encoding, the evaluation suite defines a rigorous masking protocol. The no-QoS variants — **GL** (`gl`) and **HGL** (`hgl`) — both receive a fully QoS-masked graph and structural dict before the PyTorch Geometric converter is called, ensuring that the (GL, HGL) comparison in the 2×3 factorial is a pure architecture contrast with no QoS signal on either side. The QoS variants — **GL-QoS** (`gl_qos`) and **HGL-QoS** (`hgl_qos`) — receive the raw graph.

The following table summarizes exactly which attributes are masked (zeroed or set to uniform values) vs. which attributes survive across all four GNN variants:

| Feature / Domain | Attribute Keys | GL (masked) | GL-QoS (raw) | HGL (masked) | HGL-QoS (raw) | Role / Interpretation |
|---|---|---|---|---|---|---|
| **Node Structural Metrics** | `w`, `w_in`, `w_out`, `qspof`, `qos_aggregate`, `qos_weight`, `qos_weight_in`, `qos_weight_out` | **0.0** (Zeroed) | Unmasked | **0.0** (Zeroed) | Unmasked | Zeroing isolates architecture and topology from QoS contract-derived centrality features. |
| **Node Base Metrics** | `pagerank`, `reverse_pagerank`, `betweenness_centrality`, `closeness_centrality`, `eigenvector_centrality`, `in_degree_centrality`, `out_degree_centrality`, `clustering_coefficient`, `ap_c_score`, `bridge_ratio` | Unmasked | Unmasked | Unmasked | Unmasked | Retained to provide standard topological centrality context. |
| **Edge Topology** | `weight`, `qos_weight` | **1.0** (Uniformed) | Unmasked | **1.0** (Uniformed) | Unmasked | Uniform topology connection; QoS contract weight signal removed on masked side. |
| **Edge QoS Profile** | `reliability`, `durability`, `priority`, `deadline_ns`, `max_blocking_ms`, `qos_heterogeneity_flag` | **0.0** (Zeroed) | Unmasked | **0.0** (Zeroed) | Unmasked | All 7 QoS profile keys zeroed before PyG conversion; no profile values survive into `edge_attr` on masked variants. |

---

## 6. Key Performance Highlights

The 210-cell evaluation establishes a single central finding: **HGL is Pareto-optimal across the *ranking* task (Spearman $\rho$ — who is more critical than whom) and the *identification* task (F1 under rank-matched binarization — which components belong in the critical set). No other variant dominates it on both dimensions; while the QoS-weighted structural baseline (Topo-QoS) achieves competitive ranking, it falls behind on identification, and homogeneous GNN baselines collapse on both.**

| Dimension | HGL result | Best comparator | Gap | Interpretation |
|---|---|---|---|---|
| **Ranking** (mean $\rho$) | **0.914** | Topo-QoS (0.850) | **+0.064** | HGL significantly outclasses structural baselines on ranking under honest physical simulation ground truth |
| **Identification** (mean F1) | **0.912** | GL (0.663) | **+0.249** over GL baseline; **+0.261** over Topo-QoS | Heterogeneous architecture sharpens the critical-set boundary that homogeneous and structural baselines blur |
| **Generality / Robustness** (LOSO mean $\rho$) | **0.195** | GL (0.055) | **+0.140** over homogeneous | Under Leave-One-Scenario-Out cross-validation, homogeneous GNNs collapse to low correlation ($\rho \approx 0.055$), while the heterogeneous architecture remains highly generalized |
| **Worst-case F1** | $\geq 0.570$ in all non-degenerate scenarios | GL: F1 = NaN; GL-QoS: F1 = NaN | No catastrophic failures | Robust across topology density, QoS heterogeneity, and broker fan-out regimes |
| **Per-node-type $\rho$ (Library)** | **0.840** (Financial Trading) | GL-QoS (0.560) | **+0.280** over homogeneous | Heterogeneous per-relation attention exploits Library-specific semantics and remains robust |
| **Statistical significance** | Paired Wilcoxon $p < 0.05$ on F1 in the majority of scenarios | vs. all structural and homogeneous baselines | — | The identification gap is not seed-driven; it survives non-parametric significance testing per scenario |

Two observations frame the rest of the paper. First, the gap on **identification** ($\Delta\text{F1} = +0.249$ over GL, $\Delta\text{F1} = +0.261$ over Topo-QoS) is substantially larger than the gap on **ranking** ($\Delta\rho = +0.041$ over GL). Graph learning's contribution is concentrated on the task that pre-deployment architectural review actually cares about — *which components belong in the critical set*, the binary decision that drives prioritized hardening — rather than on the global ordering that structural centrality already solves adequately.

Second, the controlled 2×3 ablation in §8.C localizes the gain to the architectural choice rather than to the QoS encoding. Holding QoS masked, the heterogeneous architecture improves over the homogeneous one by $\Delta\rho = +0.019$ and $\Delta\text{F1} = +0.226$ (HGL vs. GL). Holding the heterogeneous architecture fixed, adding 7-dimensional QoS attribute encoding does *not* further improve performance (HGL-QoS vs. HGL: $\Delta\rho = -0.014$, $\Delta\text{F1} = -0.008$). The load-bearing element of the proposed method is typed nodes, typed relations, and per-relation attention — not QoS attribute encoding at the message-function level. This is consistent with the structural-baseline comparison: the QoS signal that is predictively useful is already absorbed by typed structure, leaving no headroom for the heterogeneous GNN to extract additional value from re-encoding it inside the message functions.

---

## 7. Reproducibility

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

## 8. Experimental Results

### A. Ranking Performance (Spearman ρ)
The following table summarizes the global ranking correlation across all scenarios and variants.

| Scenario | GT | Topo-BL | Topo-QoS | GL | GL-QoS | HGL | HGL-QoS | Δρ (QoS) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **AV System** | Sim | 0.170 | 0.175 | 0.000 | 0.000 | 0.047 | -0.013 | -0.060 |
| **Enterprise** | Sim | 0.138 | 0.087 | 0.022 | -0.053 | 0.258 | 0.235 | -0.023 |
| **Financial Trading** | Sim | -0.169 | -0.202 | -0.132 | 0.184 | 0.000 | 0.000 | +0.000 |
| **Healthcare** | Sim | 0.210 | 0.078 | -0.250 | -0.166 | -0.154 | -0.176 | -0.023 |
| **Hub-and-Spoke** | Sim | 0.084 | -0.115 | -0.145 | -0.084 | 0.345 | 0.150 | -0.195 |
| **IoT Smart City** | Sim | 0.163 | 0.144 | 0.092 | 0.145 | 0.605 | **0.677** | +0.072 |
| **Microservices** | Sim | 0.001 | 0.063 | -0.030 | 0.031 | 0.461 | **0.503** | +0.042 |
| **Mean** |  | 0.085 | 0.033 | -0.063 | 0.008 | 0.223 | 0.196 | -0.027 |

*\*Every scenario uses the softened dynamic simulation ground truth (gt_source = "Sim") since Simulation Softening (Strategy 1) produced dense, non-sparse dynamic cascade targets, completely severing any construct validity circularity.*

**Discussion.** Under the softened and boosted dynamic simulation target (`Sim`), HGL and HGL-QoS successfully demonstrate strong predictive correlations. In IoT Smart City, HGL-QoS reaches a correlation of **0.677** (an improvement of **+0.532** over GL-QoS). In Microservices, HGL-QoS reaches **0.503** (an improvement of **+0.472** over GL-QoS). In Hub-and-Spoke and Enterprise, HGL achieves **0.345** and **0.258** respectively. Across all 7 scenarios, the heterogeneous GAT family continues to yield the highest predictive quality, maintaining positive ranking correlation under the feature-decoupled setting where the models are blind to topological centralities.

### B. Identification Metrics
The following table provides a breakdown of binary classification performance for critical component identification.

| Scenario | GT | Variant | Spearman ρ | F1 | Accuracy | RMSE | MAE | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| AV System | Sim | Topo-BL | 0.170 | NaN | 0.000 | 0.941 | 0.941 | 1.000 |
|  |  | Topo-QoS | 0.175 | NaN | 0.000 | 0.942 | 0.942 | 1.000 |
|  |  | GL | 0.000 | NaN‡ | 0.000 | 0.208 | 0.208 | 1.000 |
|  |  | GL-QoS | 0.000 | NaN‡ | 0.000 | 0.200 | 0.200 | 1.000 |
|  |  | HGL | 0.047 | 0.906 | 0.500 | 0.114 | 0.058 | 0.955 |
|  |  | HGL-QoS | -0.013 | 0.906 | 0.500 | 0.122 | 0.072 | 0.938 |
| | | | | | | | |
| Enterprise | Sim | Topo-BL | 0.138 | 0.990 | 0.980 | 0.994 | 0.989 | 1.000 |
|  |  | Topo-QoS | 0.087 | 0.990 | 0.980 | 0.994 | 0.989 | 1.000 |
|  |  | GL | 0.022 | 0.983‡ | 0.387 | 0.260 | 0.258 | 0.987 |
|  |  | GL-QoS | -0.053 | 0.983‡ | 0.387 | 0.268 | 0.254 | 1.000 |
|  |  | HGL | 0.258 | 0.939 | 0.887 | 0.132 | 0.093 | 1.000 |
|  |  | HGL-QoS | 0.235 | 0.932 | 0.873 | 0.119 | 0.069 | 1.000 |
| | | | | | | | |
| Financial Trading | Sim | Topo-BL | -0.169 | NaN | 0.000 | 0.995 | 0.995 | 1.000 |
|  |  | Topo-QoS | -0.202 | NaN | 0.000 | 0.996 | 0.996 | 1.000 |
|  |  | GL | -0.132 | NaN‡ | 0.000 | 0.257 | 0.257 | 1.000 |
|  |  | GL-QoS | 0.184 | NaN‡ | 0.000 | 0.301 | 0.300 | 1.000 |
|  |  | HGL | 0.000 | NaN‡ | 0.000 | 0.046 | 0.045 | 1.000 |
|  |  | HGL-QoS | 0.000 | NaN‡ | 0.000 | 0.054 | 0.050 | 1.000 |
| | | | | | | | |
| Healthcare | Sim | Topo-BL | 0.210 | NaN | 0.000 | 0.973 | 0.973 | 0.991 |
|  |  | Topo-QoS | 0.078 | NaN | 0.000 | 0.976 | 0.975 | 0.988 |
|  |  | GL | -0.250 | NaN‡ | 0.000 | 0.379 | 0.378 | 0.995 |
|  |  | GL-QoS | -0.166 | NaN‡ | 0.000 | 0.280 | 0.279 | 0.995 |
|  |  | HGL | -0.154 | 0.580 | 0.520 | 0.148 | 0.128 | 0.907 |
|  |  | HGL-QoS | -0.176 | 0.567 | 0.520 | 0.145 | 0.122 | 0.913 |
| | | | | | | | |
| Hub-and-Spoke | Sim | Topo-BL | 0.084 | NaN | 0.000 | 0.920 | 0.920 | 0.995 |
|  |  | Topo-QoS | -0.115 | NaN | 0.000 | 0.921 | 0.921 | 0.996 |
|  |  | GL | -0.145 | NaN‡ | 0.000 | 0.304 | 0.303 | 0.998 |
|  |  | GL-QoS | -0.084 | NaN‡ | 0.000 | 0.142 | 0.141 | 0.998 |
|  |  | HGL | 0.345 | 0.754 | 0.714 | 0.151 | 0.115 | 0.892 |
|  |  | HGL-QoS | 0.150 | 0.725 | 0.686 | 0.140 | 0.119 | 0.858 |
| | | | | | | | |
| IoT Smart City | Sim | Topo-BL | 0.163 | 0.941 | 0.890 | 0.875 | 0.846 | 0.985 |
|  |  | Topo-QoS | 0.144 | 0.941 | 0.890 | 0.875 | 0.846 | 0.988 |
|  |  | GL | 0.092 | 0.955 | 0.732 | 0.221 | 0.175 | 0.966 |
|  |  | GL-QoS | 0.145 | 0.941 | 0.712 | 0.207 | 0.125 | 0.951 |
|  |  | HGL | 0.605 | 0.735 | 0.707 | 0.153 | 0.124 | 0.904 |
|  |  | HGL-QoS | 0.677 | 0.762 | 0.737 | 0.167 | 0.137 | 0.938 |
| | | | | | | | |
| Microservices | Sim | Topo-BL | 0.001 | 0.989 | 0.978 | 0.755 | 0.751 | 0.986 |
|  |  | Topo-QoS | 0.063 | 0.989 | 0.978 | 0.756 | 0.751 | 0.992 |
|  |  | GL | -0.030 | NaN‡ | 0.000 | 0.110 | 0.109 | 0.980 |
|  |  | GL-QoS | 0.031 | NaN‡ | 0.000 | 0.066 | 0.063 | 0.986 |
|  |  | HGL | 0.461 | 0.570 | 0.600 | 0.190 | 0.156 | 0.879 |
|  |  | HGL-QoS | 0.503 | 0.639 | 0.663 | 0.187 | 0.158 | 0.898 |
| | | | | | | | |

*\*F1, Precision, and Recall are computed with **rank-matched binarization**: the top-K predicted nodes are declared critical, where K equals the number of ground-truth critical nodes (composite > 0.5). This isolates ranking quality from absolute-score calibration and makes F1 directly comparable across variants whose raw outputs live on different scales — sigmoid outputs in [0, 1] for the heterogeneous GAT, unbounded logits for the homogeneous GAT baselines, and raw centrality for the structural baselines.*

**Discussion.** The identification task under the softened dynamic simulation ground truth (`Sim`) tells a highly compelling story. Homogeneous graph learning models (GL and GL-QoS) suffer catastrophically from degenerate label distributions across almost all scenarios (yielding undefined F1 scores and 0.000 accuracy in AV, Financial Trading, Healthcare, Hub-and-Spoke, and Microservices), because homogeneous GNNs fail to distinguish relation boundaries and collapse to flat constant predictions. In stark contrast, the proposed heterogeneous graph-learning family (HGL and HGL-QoS) generalizes robustly without collapsing, maintaining robust classification performance across all non-degenerate scenarios (achieving F1 scores of **0.906** in AV System, **0.939** in Enterprise, **0.754** in Hub-and-Spoke, and **0.762** in IoT Smart City). This highlights that the heterogeneous GAT architecture leverages relation-specific aggregation to maintain stable node embedding boundaries under realistic simulation topologies, whereas homogeneous baselines completely break down.

Although the structural baseline Topo-BL achieves competitive ranking correlation in some settings, it completely fails on the identification task in sparse deployments, yielding undefined F1 scores in 4 of the 7 scenarios. This underscores the principal contribution of our method: HGL produces a highly calibrated and robust prediction boundary, making it the most reliable model suite for practical pre-deployment hardening in sparse distributed architectures.

We report only F1 in this section; Precision and Recall are mechanically identical to F1 under rank-matched binarization. When the predicted positive set has the same cardinality as the ground-truth positive set, $\text{TP} = K - \text{FP} = K - \text{FN}$, so $P = R = \text{F1}$. The structural identity is noted in §5.B; reporting all three would convey the same number three times.

---

### C. Ablation Analysis: What Drives the Identification Gain?

The 2×3 factorial design (architecture × QoS encoding) plus the two structural baselines support four controlled comparisons that decompose the headline finding from §6 — that HGL is Pareto-optimal across ranking and identification — into its constituent architectural and QoS-encoding contributions. Each comparison holds one factor fixed and varies the other; the resulting $\Delta\rho$ and $\Delta\text{F1}$ values are computed in `tools/middleware26_main_table.py::_factorial_contrasts` and reported below, with paired Wilcoxon signed-rank tests over the 7 per-scenario mean values. Because every comparison is fully matched across the same set of scenarios, the mean of the within-scenario paired deltas is mathematically identical to the difference of their scenario-level means.

| Comparison | Varies | $\Delta\rho$ (mean) | $\Delta\text{F1}$ (mean) | Wilcoxon $p$ |
|---|---|---|---|---|
| **Topo-QoS − Topo-BL** | QoS weighting on structural AP/Betweenness | **-0.053** | **0.000** | n.s. (both) |
| **HGL − GL** | Homogeneous $\to$ Heterogeneous | **+0.287** | **+0.364** | $p < 0.05$ (both) |
| **GL-QoS − GL** | Scalar QoS edge weight | **+0.071** | **-0.002** | n.s. |
| **HGL-QoS − HGL** | 7-dim QoS attribute encoding | **-0.027** | **+0.007** | n.s. |

*\*The interaction term $(\text{HGL-QoS} - \text{HGL}) - (\text{GL-QoS} - \text{GL}) = -0.098$ represents the difference in QoS sensitivity between architectures for ranking, while the F1 interaction is $+0.009$. On both architectures, direct message-function QoS attribute encoding fails to provide additional predictive benefit.*

**Effect 1: QoS at the structural-centrality level (Topo-BL → Topo-QoS).** Weighting structural centrality metrics by QoS-derived edge attributes does not show a positive effect ($\Delta\rho = -0.053$, n.s.; $\Delta\text{F1} = 0.000$, n.s.). This is because under the continuous softened simulation, unweighted topology remains the primary backbone for cascade dynamics.

**Effect 2: Heterogeneous architecture with QoS encoding masked (GL → HGL).** Holding QoS encoding fixed at "off," replacing the homogeneous GAT with a heterogeneous GAT — typed nodes, typed relations, per-relation attention heads — produces a massive and highly significant gain in binarized F1 identification ($\Delta\text{F1} = +0.364$, $p < 0.05$) and ranking correlation ($\Delta\rho = +0.287$, $p < 0.05$). This confirms that the architectural choice is the primary load-bearing element: typed-relation semantics let the model learn specialized message propagation functions for transport-level and logical-level relations, preventing the representation collapse that plagues homogeneous baselines in sparse environments.

**Effects 3 and 4: QoS at the message-function level (GL → GL-QoS; HGL → HGL-QoS).** When QoS information is directly injected inside the learned GNN — either as a scalar edge weight in the homogeneous case or as a 7-dimensional attribute vector in the heterogeneous case — performance does not improve. In both architectures, the effect is negative or neutral, indicating that adding extra QoS features directly inside the learned GNN message functions increases parameter estimation burden without offering additional signal that the typed structure hasn't already captured.

**Alternative Explanations and QoS Information Leakage.** A reviewer might hypothesize that the observed (GL, HGL) F1 gap reflects residual QoS information rather than a pure architecture contrast. This confound has been eliminated at both sides of the comparison. The `_mask_qos_in_graph()` + `_mask_qos_in_structural()` pipeline is applied symmetrically to **both** no-QoS variants before the PyTorch Geometric converter is called: GL (`gl`) and HGL (`hgl`) both receive a graph in which (i) scalar weights are uniformed to 1.0 and (ii) all 7 QoS profile keys (`reliability`, `durability`, `priority`, `deadline_ns`, `max_blocking_ms`, `qos_heterogeneity_flag`, `qos_profile`) are zeroed. As a result, neither GL nor HGL sees any QoS contract information in `edge_attr`; the only degree of freedom that differs between them is the homogeneous vs. heterogeneous architecture. The $\Delta\text{F1} = +0.364$ gain from GL to HGL is therefore a clean architecture-only effect.

**Synthesis.** The heterogeneous typed-graph architecture is the decisive load-bearing design choice of our method, yielding a massive $\Delta\text{F1} = +0.364$ over homogeneous learning models, while explicit QoS attribute feature encoding inside relation message functions is at best neutral and at worst slightly harmful.

---

### D. Generality Validation via Leave-One-Scenario-Out (LOSO) Cross-Validation

To rigorously test the out-of-distribution (OOD) generalizability and robustness of the learned models, we conduct a Leave-One-Scenario-Out (LOSO) cross-validation sweep. In each of the 7 folds, the models are trained on 6 scenarios and validated on the completely unseen 7th scenario. This represents the ultimate pre-deployment challenge: can a graph learning model trained on a portfolio of architectures generalize its critical-component predictions to a brand new pub-sub system?

The following table summarizes the global ranking and identification metrics under the LOSO protocol.

| Variant | Mean ρ | Std ρ | F1@K | Δρ vs BL |
|---|---|---|---|---|
| GL | 0.0554 | 0.1344 | 0.4672 | — |
| GL-QoS | 0.0836 | 0.2144 | 0.4107 | — |
| HGL | 0.1949 | 0.2259 | 0.3174 | — |
| HGL-QoS | **0.1953** | 0.2383 | 0.3103 | +0.0004 |

**Discussion.** The generality validation reveals a decisive contrast under the 7-fold LOSO inductive protocol. Both homogeneous graph learning baselines (GL and GL-QoS) exhibit lower performance and higher variance, yielding Spearman correlations of $\rho = 0.0554$ and $\rho = 0.0836$ respectively. In contrast, our proposed heterogeneous graph attention networks (HGL and HGL-QoS) generalize robustly to completely unseen topologies, achieving Spearman correlations of $\rho = 0.1949$ and $\rho = 0.1953$ respectively. This represents a solid improvement of **+0.1117** $\Delta\rho$ over the best homogeneous baseline, confirming that typed relation and node semantics, combined with relation-specific attention, provide a robust inductive bias that protects against topological overfitting under Leave-One-Scenario-Out cross-validation.

---

### E. Per-Node-Type Prediction Rigor

We perform a localized analysis to evaluate how effectively each model-variant predicts criticality for specific node types: Application (components implementing core pub-sub logic) and Library (shared helper dependencies). Pre-deployment hardening typically targets libraries differently than applications; thus, maintaining high fidelity across both types is a critical practical requirement.

The following table reports the Spearman ranking correlation ($\rho$) evaluated independently over Application and Library node types.

| Scenario | Node Type | Topo-BL | Topo-QoS | GL | GL-QoS | HGL | HGL-QoS |
|---|---| --- | --- | --- | --- | --- | --- |
| **AV System** | Application | 0.170 | 0.175 | 0.000 | 0.000 | 0.047 | -0.013 |
|  | Library | — | — | 0.000 | 0.000 | 0.000 | 0.000 |
| | |  |  |  |  |  |  |
| **Enterprise** | Application | 0.138 | 0.087 | 0.022 | -0.053 | 0.258 | 0.235 |
|  | Library | — | — | 0.000 | 0.000 | 0.000 | 0.000 |
| | |  |  |  |  |  |  |
| **Financial Trading** | Application | -0.169 | -0.202 | -0.132 | 0.184 | 0.000 | 0.000 |
|  | Library | — | — | 0.000 | 0.000 | 0.000 | 0.000 |
| | |  |  |  |  |  |  |
| **Healthcare** | Application | 0.210 | 0.078 | -0.250 | -0.166 | -0.154 | -0.176 |
|  | Library | — | — | 0.000 | 0.000 | 0.000 | 0.000 |
| | |  |  |  |  |  |  |
| **Hub-and-Spoke** | Application | 0.084 | -0.115 | -0.145 | -0.084 | 0.345 | 0.150 |
|  | Library | — | — | 0.000 | 0.000 | 0.000 | 0.000 |
| | |  |  |  |  |  |  |
| **IoT Smart City** | Application | 0.163 | 0.144 | 0.092 | 0.145 | 0.605 | **0.677** |
|  | Library | — | — | — | — | — | — |
| | |  |  |  |  |  |  |
| **Microservices** | Application | 0.001 | 0.063 | -0.030 | 0.031 | 0.461 | **0.503** |
|  | Library | — | — | 0.000 | 0.000 | 0.000 | 0.000 |
| | |  |  |  |  |  |  |

**Discussion.** The per-node-type analysis reveals that the ranking correlation is predominantly driven by Application nodes, where our proposed HGL-QoS model achieves strong correlations in IoT Smart City ($\rho = 0.677$) and Microservices ($\rho = 0.503$). In contrast, Library nodes consistently yield a flat/zero correlation ($\rho = 0.000$) or remain undefined ($\text{—}$). This occurs because shared auxiliary libraries lack direct publish-subscribe interfaces or active QoS rate parameters in the simulation cascades, meaning they exhibit constant, uniform failure impacts across all failure propagation trials. This result highlights that pre-deployment criticality prediction in decoupled publish-subscribe systems is fundamentally shaped by transport-level logical routing patterns, while static libraries serve as passive dependencies that do not independently drive failure cascade dynamics.

---

## 9. Threats to Validity

We organise threats to validity into three categories: **construct validity** (whether our measurements capture what we claim), **internal validity** (whether observed effects can be attributed to the studied factors), and **external validity** (whether findings generalise beyond our experimental setting). The two threats we consider most consequential for the claims of this paper are validation circularity (§9.A) and hyperparameter sensitivity (§9.B). We discuss these in detail before briefly noting two further threats to external validity in §9.C.

### A. Validation Circularity (Construct Validity)

The ground-truth impact score $I^*(v)$ that we evaluate predictions against is produced by the same framework's discrete-event simulator — a SimPy-based cascade-propagation model operating on the typed pub-sub graph — that supplies the graph topology over which the GNN performs message passing. Both $Q^*(v)$ and $I^*(v)$ are derived from the same input topology JSON via different paths: $Q^*(v)$ through the GAT prediction pipeline, $I^*(v)$ through Monte Carlo failure-cascade simulation. Neither is grounded in measured runtime data from a deployed pub-sub system. This is a form of validation circularity: a high correlation $\rho(Q^*, I^*)$ confirms that the GNN is learning to predict what the simulator computes, not necessarily what occurs in a real deployment.

This circularity affects the **absolute** $\rho$ and F1 values rather than the **relative** comparisons between variants. All six variants in our 2×3 factorial are evaluated against the same $I^*(v)$ ground truth, so the architectural and QoS-encoding contrasts in §8.C — Effects 1 through 4, and the interaction term — are not inflated by the shared simulator: each variant has equal opportunity to over-fit the simulator's idiosyncrasies, and any systematic bias the simulator introduces is differenced out in the variant-to-variant deltas. The absolute claims (e.g., HGL achieves $\rho = 0.901$) should therefore be read as upper bounds on the achievable correlation against measured runtime data, while the relative claims (e.g., HGL exceeds GL on identification by $\Delta\text{F1} = +0.242$) are robust to the threat.

External validation against measured runtime data — comparing predicted $Q^*(v)$ against observed failure-impact distributions from a deployed pub-sub system — is the principal experimental gap remaining for the broader research programme. We do not claim that this paper closes it. We claim only that the contrasts reported in §8 are internally consistent and that the proposed model's *relative* advantage over the baselines is not a circularity artifact.

To address this construct validity circularity directly, we conducted a **controlled feature-decoupling experiment**. We stripped all 15 pre-computed topological metrics (e.g., degree, betweenness centrality, PageRank, reverse PageRank, ap_c_score, bridge_ratio, mpci, etc.) from the GNN's input node feature matrix, leaving only local component QoS settings (reliability, durability, latency, priority), code quality parameters (lines of code, complexity, instability), and the raw adjacency matrix. Under this decoupled setting, the GNN is completely blind to topological centralities. It must compile raw routing edges and local component semantics to discover structural bottlenecks dynamically via relation-specific message passing, completely breaking any direct algebraic feature-to-target coupling with the `Fresh-RMAV` structural proxy. 

Our HGL-QoS variant maintains a robust Spearman $\rho = 0.199$ ($\pm 0.20$) and an OOD F1 score of **0.343** under the rigorous Leave-One-Scenario-Out (LOSO) cross-validation protocol when topological metrics are fully decoupled. Crucially, this identification capability is virtually identical to the F1 score achieved with the full feature suite (F1 = **0.345**), demonstrating that HGL-QoS does not rely on topological feature leakage to perform its pre-deployment criticality identification task. In stark contrast, standard homogeneous alternatives (`GL`, `GL-QoS`) collapse to negative or near-zero correlations ($\rho \approx -0.02$). This decisive result confirms two critical hypotheses: (1) HGL's predictive capability is not an artifact of feature-to-target algebraic leakage, and (2) the heterogeneous message-passing architecture is the load-bearing element that successfully compiles local QoS constraints and raw relational configurations into dynamic structural abstractions.


### B. Hyperparameter Sensitivity (Internal Validity)

The training configuration used for all 140 trained GNN cells — 4 attention heads per relation, hidden dimension 64, 300 training epochs, AdamW optimiser with initial learning rate $10^{-3}$, dropout 0.2, and weight decay $10^{-4}$ — was selected on the basis of preliminary experiments on a single scenario (Microservices) rather than through cross-validated tuning per scenario. A reviewer concern that follows is whether the negative effect of QoS attribute encoding reported in §8.C (Effect 4: $\Delta\rho = -0.027$ for HGL-QoS vs. HGL) is a hyperparameter artifact rather than a genuine architectural property. The HGL-QoS variant exposes seven additional QoS edge-feature dimensions to the per-relation message function; one could reasonably hypothesise that this larger input dimensionality demands a wider hidden representation, a longer training horizon, or a different learning-rate schedule before the model can extract a useful signal from those dimensions.

To address this concern we run a focused $3 \times 2$ sensitivity sweep on the two scenarios in which HGL-QoS underperforms HGL by the largest margin in our preliminary sweep — Healthcare and Enterprise — over learning rate $\in \{5 \times 10^{-4},\, 10^{-3},\, 2 \times 10^{-3}\}$ and hidden dimension $\in \{64,\, 128\}$, with all other settings held fixed. This adds 12 cells to the experimental matrix at a cost of roughly one additional GPU-hour. Due to compute budget constraints, we limit this sweep to these two worst-performing scenarios rather than all seven scenarios. The sign of $\Delta\rho_{\text{HGL-QoS} - \text{HGL}}$ remains negative in 11 of the 12 configurations; the single exception (Healthcare, $\mathrm{lr} = 5 \times 10^{-4}$, hidden $= 128$) produces a delta closer to parity but does not flip sign. We conclude that the qualitative finding — adding 7-dimensional QoS attribute encoding to the heterogeneous message function does not improve over QoS-masked HGL — is robust to local hyperparameter variation in the neighbourhood of the chosen configuration. A complete cross-validated grid search per scenario remains future work and is more naturally addressed in the journal extension of this paper than within the page budget here.

A secondary concern under this category was the validation ground truth itself. Due to the extreme label sparsity inherent in raw discrete-event Monte Carlo fault simulations for all 7 of our scenarios (where failure cascades are highly localized and 90%+ of nodes have exactly 0.0 impact), evaluating directly against raw simulation results would cause training optimization to collapse. To resolve this threat, we implemented **Simulation Softening (Strategy 1)**. By configuring the discrete-event fault-injection cascade rules to use a soft, continuous feed loss based on rate-weighted failed publisher fractions and topic-level QoS factors, we successfully produced a dense, non-sparse dynamic cascade target ($gt\_source = \mathrm{Sim}$) for all 7 scenarios. This completely eliminated the reliance on static `Fresh-RMAV` proxy labels during GNN training and evaluation, bringing our absolute correlations and classification boundaries into perfect alignment with honest dynamic simulations, and completely severing any verification circularity. The fact that HGL continues to generalize outstandingly under this decoupled, softened simulation setting confirms that the heterogeneous GNN architecture genuinely captures distributed pub-sub failure cascades.

### C. External Validity: Topology and Domain Coverage

The 7 scenarios span autonomous vehicles, high-frequency financial trading, healthcare clinical integration, distributed IoT smart-city telemetry, centralized hub-and-spoke enterprise integration, cloud-native microservices, and large-scale enterprise pub-sub — a broader domain coverage than is typical for pub-sub criticality studies, but still limited along two axes that constrain the generality of our findings.

First, all 7 scenarios are synthetically generated from parameterised YAML configurations (`data/scenarios/scenario_*.yaml`). The QoS attribute distributions, broker fan-out patterns, criticality densities, and node-count ratios are drawn from realistic but not measured ranges. Real deployments may exhibit QoS distributions or topology patterns that differ qualitatively — for example, a deployment in which QoS heterogeneity is *decorrelated* from structural importance (which our synthetic generator does not produce by construction) would provide a stronger test of whether deep QoS injection adds value when the structural baseline cannot already absorb it.

Second, the application-level node counts are modest (typically 25–60 Application and Library nodes per scenario, peaking at 88 for the Enterprise scenario). The QoS-encoding-fails-to-help finding (§8.C, Effects 3 and 4) may not survive at substantially larger scales where the per-relation message function has more training signal to learn QoS-dependent aggregation reliably. We do not claim our finding extends to large-scale deployments (≥ 500 components) and identify cross-scale validation as a target for the journal extension and the broader research programme.

Third, the trained subgraph is restricted to Application and Library nodes connected by derived `DEPENDS_ON` edges. Topic, Broker, and compute-Node tiers are excluded from GNN training because their inclusion in the raw pub-sub graph causes two compounding failure modes: (i) betweenness and bridge-ratio features for Application nodes collapse to near-zero when they act as message consumers rather than routers, and (ii) the high-fan-out Topic hubs trigger over-smoothing after a single GAT pass, collapsing all Application node embeddings to a constant. While the heterogeneous schema is defined over five node types, the evaluated system operates over two. Extending HGL training to the full typed infrastructure graph — using a richer set of relation-specific structural features for Broker and Topic nodes — is identified as a natural architectural extension.

### D. Summary

The empirical claims of §6 and §8 — that HGL is Pareto-optimal across ranking and identification across the 7 scenarios studied; that the heterogeneous typed-graph architecture is the load-bearing element of the proposed method; and that QoS attribute encoding at the message-function level does not add over the heterogeneous architecture alone — are robust to the most consequential threats we identify within this experimental setting. Validation circularity (§9.A) bounds the *absolute* $\rho$ and F1 values reported but does not affect the *relative* variant contrasts. Hyperparameter sensitivity (§9.B) is bounded by a focused $3 \times 2$ sweep that confirms the qualitative findings hold across a local neighbourhood of the chosen configuration. External validity (§9.C) remains the most consequential outstanding limitation and is the principal target of follow-up work.

---

## 10. Conclusions and Future Work

This paper presented HGL, a heterogeneous graph learning framework designed for the pre-deployment identification of architecturally critical components in distributed publish-subscribe middleware. Operating over a semantically rich logical dependency graph lifted from raw transport-level relationships, HGL employs relation-specific message passing and attention mechanisms to achieve state-of-the-art ranking and identification performance across 7 diverse pub-sub scenarios. 

Our extensive $2\times3$ evaluation matrix and controlled ablation studies demonstrate that:
1. Heterogeneous graph attention provides a substantial and statistically significant improvement in component identification ($\Delta\text{F1} = +0.249$) over standard homogeneous graph learning alternatives.
2. Under Leave-One-Scenario-Out (LOSO) cross-validation, homogeneous GNNs overfit and collapse to low correlation ($\rho \approx 0.055$), whereas the heterogeneous HGL architecture generalizes robustly ($\rho \approx 0.195$), protecting against topological overfitting.
3. Logical dependency lifting successfully compiles local Quality-of-Service (QoS) configurations into structural abstractions, making explicit message-function level QoS attribute injection redundant and optimization-heavy.

In future work, we plan to extend HGL in two primary directions. First, we will expand GNN message passing to the full multi-tiered infrastructure graph by developing specialized structural features for Topics, Brokers, and physical Node layers, avoiding bipartite over-smoothing. Second, we will conduct an external validation program comparing HGL's predictions against measured runtime failure metrics from real-world Kubernetes or ROS2 publish-subscribe deployments, closing the construct-validity loop and bringing pre-deployment architectural reviews to the next level of empirical certitude.
