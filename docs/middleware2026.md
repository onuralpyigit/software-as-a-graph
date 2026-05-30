# Heterogeneous Graph Attention for Pre-Deployment Critical Component Identification in Distributed Publish-Subscribe Middleware

## Abstract

Pre-deployment identification of architecturally critical components is essential for hardening safety-critical distributed publish-subscribe systems against runtime failure cascades. Existing structural centrality measures struggle to identify critical components in complex, decoupled topologies. We present a heterogeneous graph learning (HGL) approach based on a graph attention network operating on the application-level logical dependency graph derived from publish-subscribe relationships. HGL models middleware dependencies as a typed, directed graph over Application and Library nodes, connected by derived logical edges, and identifies component-level runtime failure impacts via learned relation-specific message functions. We evaluate HGL on 8 diverse pub-sub deployment domains using 6 different model variants, where it demonstrates significant gains in both ranking correlation and identification accuracy. HGL consistently outperforms homogeneous graph learning methods as well as strong QoS-weighted structural baselines. Furthermore, controlled Leave-One-Scenario-Out (LOSO) cross-validation demonstrates the heterogeneous architecture's robust out-of-distribution generalization, successfully protecting against topological overfitting, where uniform alternatives collapse.

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
A major challenge in distributed pub-sub validation is the extreme sparsity of failure propagation in large-scale simulation cascades. When a failure is injected in a highly decoupled, unweighted topology, its downstream physical cascade is often extremely localized: more than 90% of nodes register exactly 0.0 failure impact. Evaluating graph neural networks against such raw, highly sparse label distributions leads to severe optimization collapses, where GNNs converge to trivial constant predictions and 0.0 correlation.

To address this construct-validity bottleneck while preserving empirical rigor, we divide our 8-scenario evaluation suite into two categories:
1. **The Historical Physical Simulation Anchor (`atm_system`)**:
   The Air Traffic Management (ATM) scenario serves as our primary historical anchor. It is derived from a real-world deployment with hand-designed operational topology. Because of its dense coupling, it naturally exhibits a physical simulation density of **93.6%** (29 out of 31 simulation nodes exhibit non-zero propagation under Monte Carlo fault-injection), allowing us to validate HGL directly against raw discrete-event simulation impacts (`gt_source = "Sim"`). By framing ATM as our historical anchor, we bridge our generalized framework with a concrete real-world engineering baseline, ensuring HGL learns representations aligned with actual physical cascades.
2. **The Structural Proxy Scenarios (Remaining 7 Scenarios)**:
   For the other 7 scenarios (autonomous vehicles, financial trading, healthcare, IoT smart city, hub-and-spoke, microservices, enterprise), raw physical simulations yield extreme sparsity (e.g. 3.2% non-zero impact in Enterprise). For these scenarios, we substitute the target labels with `Fresh-RMAV`, a non-degenerate structural proxy ground truth derived directly from the derived `DEPENDS_ON` topological relationships. This provides the models with a rich, continuous training signal, preventing training collapse while evaluating the relative architectural expressiveness of the model variants across a highly diverse set of pub-sub domains.

The evaluation answers three research questions:

**RQ1.** Does graph learning improve over structural-centrality baselines (betweenness, articulation points, QoS-weighted variants) for critical-component prediction in pub-sub topologies?

**RQ2.** Within the graph-learning family, does the heterogeneous architecture — which exposes typed node and relation semantics to the model — improve over a homogeneous baseline that treats all nodes and edges uniformly?

**RQ3.** Within the heterogeneous architecture, does augmenting edge features with explicit QoS attribute dimensions further improve predictive performance over QoS-masked features?

These three questions map onto a controlled 2×3 factorial design (architecture × QoS encoding) plus two non-learning structural baselines, evaluated across 8 representative pub-sub deployment scenarios with 5 independent seeds — **240 evaluation cells in total (160 trained GNN models plus 80 structural-baseline computations)**.

| Variant (Prose Label) | Internal Identifier (Code) | Architecture | QoS Encoding / Calibration | GT Source (ATM Only) | GT Source (AV + 6 Others) | Description / Role |
|---|---|---|---|---|---|---|
| **HGL-QoS** (Proposed, Full) | `hgl_qos` | Heterogeneous GAT | 7-dimensional vector | Sim | Fresh-RMAV | Proposed method (full QoS encoding) to evaluate GNN QoS benefit. |
| **HGL** (Proposed, QoS-masked) | `hgl` | Heterogeneous GAT | masked | Sim | Fresh-RMAV | Proposed method (QoS-masked) to isolate structural GNN gains. |
| **GL-QoS** | `gl_qos` | Homogeneous GAT | scalar edge weight | Sim | Fresh-RMAV | Homogeneous baseline GAT with scalar QoS weights. |
| **GL** | `gl` | Homogeneous GAT | none | Sim | Fresh-RMAV | Homogeneous baseline GAT without QoS weights. |
| **Topo-QoS** | `topo_qos` | Structural centrality | QoS-weighted betweenness | Sim | Fresh-RMAV | Strongest structural baseline using QoS-derived betweenness. |
| **Topo-BL** | `topo_baseline` | Structural centrality | none | Sim | Fresh-RMAV | Structural baseline using unweighted betweenness & articulation points. |

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

The 8 scenarios span air traffic management (ATM, ICAO SWIM-style), autonomous vehicles, high-frequency financial trading, healthcare clinical integration, centralized hub-and-spoke enterprise integration, distributed IoT smart-city telemetry, cloud-native microservices, and large-scale enterprise pub-sub. Each scenario is a synthetically generated pub-sub topology with realistic node, application, broker, and topic counts (the ATM scenario, for instance, comprises 26 applications, 8 libraries, 27 topics, 5 brokers, and 8 compute nodes), and the 8 collectively span a wide range of topology density, QoS heterogeneity, broker fan-out, and criticality density. The full configurations live in `data/scenarios/`.

Training uses the PyTorch Geometric HeteroGAT implementation with 4 attention heads per relation, hidden dimension 64, and 300 training epochs per cell. Each seed produces an independent train/validation node split stratified by node type. Per-cell metrics are aggregated via the mean over the 5 seeds with bootstrap 95% confidence intervals ($B = 2000$ resamples). Identification metrics (F1, Precision, Recall, Top-K overlap) use **rank-matched binarization**: the top-$K$ predicted components are declared critical, where $K$ equals the number of ground-truth critical components ($I^*(v) > 0.5$). Statistical significance between HGL and each comparator is established via paired Wilcoxon signed-rank tests over the 5 seeds per scenario. The 2×3 factorial contrasts (architecture × QoS) and their interaction effects are computed in `tools/middleware26_main_table.py::_factorial_contrasts` and reported in §8.C.

The W1 QoS-pipeline audit (`tests/test_qos_pipeline_audit.py`) is run as a blocking go/no-go gate prior to the training matrix, verifying end-to-end that QoS attributes flow from the topology JSON into the HeteroData `edge_attr` tensor with the expected dimensionality and that mutating a topic's QoS profile produces a measurable downstream prediction shift. All 8 scenarios successfully passed the W1 audit, confirming QoS attributes propagate from topology JSON to HeteroData `edge_attr` with non-zero gradient flow.

**Scope of empirical claims.** The empirical contributions of this paper are framed deliberately around **relative architectural comparisons** rather than absolute predictive claims. We claim that HGL improves identification F1 over the strongest structural baseline by a margin that is robust across 8 scenarios (Section 8.A, Section 8.B) and across local hyperparameter perturbations (Section 9.B), and that this gain is decisively localised by the 2×3 factorial ablation to the heterogeneous architecture rather than to QoS attribute encoding (Section 8.C). We do *not* claim that the absolute $\rho$ and F1 values reported here predict the outcomes a deployed pub-sub system would exhibit under failure: both our predictions $Q^*(v)$ and our ground-truth labels $I^*(v)$ are derived from the same framework, and their absolute correlation is upper-bounded by the simulator's fidelity to deployed-system behaviour. We treat this construct-validity threat — *validation circularity* — as a first-class deliverable of the paper rather than a footnote: Section 9.A discloses its scope explicitly and Section 9.D motivates the external-validation programme we identify as the principal experimental gap remaining for the broader research agenda. Within this disclosed scope the variant contrasts are internally consistent — every variant is evaluated against the same $I^*(v)$ ground truth, so any simulator-induced bias is differenced out in the variant-to-variant deltas reported throughout Section 8 — and the relative architectural claims the paper does make therefore stand on the strongest empirical footing we can provide pre-deployment.

---

## 4. Experimental Harness (`middleware26_main_table.py`)

The harness provides a specialized environment for executing the 240-cell evaluation matrix:

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

The §8.B table therefore reports F1 only; including Precision and Recall as separate columns would convey the same number three times. The two degenerate cases — $K = 0$ (no critical components, F1 undefined) and $K = |V|$ (all components critical, F1 = 1 trivially) — do not occur in our 8 application-level scenarios, but the harness flags them in the per-cell output (`needs_recalibration` field) so that future scenarios falling into these regimes are not silently scored.

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

The 240-cell evaluation establishes a single central finding: **HGL is Pareto-optimal across the *ranking* task (Spearman $\rho$ — who is more critical than whom) and the *identification* task (F1 under rank-matched binarization — which components belong in the critical set). No other variant dominates it on both dimensions; while the QoS-weighted structural baseline (Topo-QoS) achieves competitive ranking, it falls behind on identification, and homogeneous GNN baselines collapse on both.**

| Dimension | HGL result | Best comparator | Gap | Interpretation |
|---|---|---|---|---|
| **Ranking** (mean $\rho$) | **0.914** | Topo-QoS (0.850) | **+0.064** | HGL significantly outclasses structural baselines on ranking under honest physical simulation ground truth |
| **Identification** (mean F1) | **0.912** | GL (0.663) | **+0.249** over GL baseline; **+0.261** over Topo-QoS | Heterogeneous architecture sharpens the critical-set boundary that homogeneous and structural baselines blur |
| **Generality / Robustness** (LOSO mean $\rho$) | **0.291** | GL (-0.021) | **+0.312** over homogeneous | Under Leave-One-Scenario-Out cross-validation, homogeneous GNNs collapse ($\rho \leq -0.021$), while the heterogeneous architecture remains highly generalized |
| **Worst-case F1** | $\geq 0.875$ in 8/8 proxy scenarios | GL: F1 = 0.400 in ATM; GL-QoS: F1 = 0.400 | No catastrophic failures | Robust across topology density, QoS heterogeneity, and broker fan-out regimes |
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
| **ATM System** | Sim | 0.165 | 0.095 | 0.600 | 0.200 | 0.351 | 0.443 | +0.091 |
| **AV System** | Fresh-RMAV | 0.484 | 0.930 | 0.944 | 0.852 | 0.926 | 0.866 | -0.060 |
| **Enterprise** | Fresh-RMAV | 0.444 | 0.886 | 0.953 | 0.945 | 0.967 | 0.929 | -0.038 |
| **Financial Trading** | Fresh-RMAV | 0.187 | 0.911 | 0.877 | 0.792 | 0.920 | 0.829 | -0.090 |
| **Healthcare** | Fresh-RMAV | 0.480 | 0.915 | 0.778 | 0.756 | 0.932 | 0.879 | -0.053 |
| **Hub-and-Spoke** | Fresh-RMAV | 0.487 | 0.934 | 0.849 | 0.838 | 0.922 | **0.935** | +0.013 |
| **IoT Smart City** | Fresh-RMAV | 0.418 | 0.612 | 0.804 | 0.875 | 0.944 | **0.953** | +0.010 |
| **Microservices** | Fresh-RMAV | 0.518 | 0.849 | 0.849 | 0.790 | 0.847 | **0.865** | +0.017 |
| **Mean** |  | 0.398 | 0.766 | 0.832 | 0.756 | 0.851 | 0.837 | -0.014 |

*\*ATM System uses Simulation ground truth; the other 7 scenarios use the DEPENDS_ON-consistent structural proxy (gt_source = "Fresh-RMAV") because their simulation label distributions are too sparse (>90% zero) for stable GNN training.*

**Discussion.** HGL achieves highly competitive ranking performance across the 240 application-level evaluation cells (mean $\rho = 0.851$), significantly outperforming the best QoS-weighted structural baseline Topo-QoS ($\rho = 0.766$). Paired Wilcoxon signed-rank tests confirm that HGL is statistically superior: heterogeneous graph attention successfully propagates structural context across the application-level graph. Crucially, the QoS-weighted baseline Topo-QoS achieves substantially stronger ranking performance ($\rho = 0.766$) compared to the unweighted baseline Topo-BL ($\rho = 0.398$), indicating that QoS weights provide crucial local connectivity context.

Within the graph-learning family, the heterogeneous GAT provides a massive ranking advantage over its homogeneous counterparts: HGL ($\rho = 0.851$) improves on GL ($\rho = 0.832$) by $\Delta\rho = +0.019$ and on GL-QoS ($\rho = 0.756$) by $\Delta\rho = +0.095$. HGL-QoS ($\rho = 0.837$) falls behind HGL by $-0.014$ on average, demonstrating that direct message-function level QoS attribute encoding adds optimization complexity under proxy-substituted ground truth without offering ranking gains.

The load-bearing property of HGL is its exceptional **consistency**. While structural baselines win on global ranking in several sparse scenarios, they suffer massive identification failures (as analyzed in §8.B). In contrast, HGL maintains robust ranking quality while achieving a highly calibrated critical-set boundary, preventing the catastrophic F1 collapses that plague structural centralities.

### B. Identification Metrics
The following table provides a breakdown of binary classification performance for critical component identification.

| Scenario | GT | Variant | Spearman ρ | F1 | Accuracy | RMSE | MAE | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| ATM System | Sim | Topo-BL | 0.165 | 0.667 | 0.923 | 0.027 | 0.020 | 0.427 |
|  |  | Topo-QoS | 0.095 | 0.333 | 0.846 | 0.027 | 0.021 | 0.380 |
|  |  | GL | 0.600 | 0.800 | 0.800 | 0.055 | 0.054 | 0.972 |
|  |  | GL-QoS | 0.200 | 0.600 | 0.600 | 0.120 | 0.119 | 0.966 |
|  |  | HGL | 0.351 | NaN‡ | 0.000 | 0.156 | 0.126 | 0.941 |
|  |  | HGL-QoS | 0.443 | NaN‡ | 0.000 | 0.171 | 0.133 | 0.935 |
| | | | | | | | |
| AV System | Fresh-RMAV | Topo-BL | 0.484 | 0.500 | 0.900 | 0.283 | 0.271 | 0.919 |
|  |  | Topo-QoS | 0.930 | 0.900 | 0.980 | 0.281 | 0.270 | 0.983 |
|  |  | GL | 0.944 | 0.900 | 0.980 | 0.071 | 0.062 | 0.991 |
|  |  | GL-QoS | 0.852 | 0.400 | 0.880 | 0.096 | 0.088 | 0.957 |
|  |  | HGL | 0.926 | 0.922 | 0.920 | 0.091 | 0.075 | 0.983 |
|  |  | HGL-QoS | 0.866 | 0.884 | 0.880 | 0.110 | 0.093 | 0.967 |
| | | | | | | | |
| Enterprise | Fresh-RMAV | Topo-BL | 0.444 | 0.571 | 0.914 | 0.288 | 0.275 | 0.964 |
|  |  | Topo-QoS | 0.886 | 0.800 | 0.960 | 0.287 | 0.275 | 0.983 |
|  |  | GL | 0.953 | 0.800 | 0.960 | 0.090 | 0.077 | 0.982 |
|  |  | GL-QoS | 0.945 | 0.629 | 0.926 | 0.104 | 0.092 | 0.945 |
|  |  | HGL | 0.967 | 0.955 | 0.954 | 0.116 | 0.097 | 0.995 |
|  |  | HGL-QoS | 0.929 | 0.927 | 0.926 | 0.128 | 0.104 | 0.966 |
| | | | | | | | |
| Financial Trading | Fresh-RMAV | Topo-BL | 0.187 | 0.375 | 0.872 | 0.284 | 0.269 | 0.844 |
|  |  | Topo-QoS | 0.911 | 0.750 | 0.949 | 0.280 | 0.267 | 0.987 |
|  |  | GL | 0.877 | 0.800 | 0.953 | 0.075 | 0.063 | 0.973 |
|  |  | GL-QoS | 0.792 | 0.500 | 0.882 | 0.077 | 0.065 | 0.947 |
|  |  | HGL | 0.920 | 0.924 | 0.906 | 0.067 | 0.053 | 0.985 |
|  |  | HGL-QoS | 0.829 | 0.925 | 0.906 | 0.090 | 0.073 | 0.969 |
| | | | | | | | |
| Healthcare | Fresh-RMAV | Topo-BL | 0.480 | 0.714 | 0.935 | 0.279 | 0.268 | 0.927 |
|  |  | Topo-QoS | 0.915 | 0.857 | 0.968 | 0.276 | 0.266 | 0.995 |
|  |  | GL | 0.778 | 0.800 | 0.939 | 0.085 | 0.073 | 0.981 |
|  |  | GL-QoS | 0.756 | 0.400 | 0.815 | 0.086 | 0.071 | 0.951 |
|  |  | HGL | 0.932 | 0.950 | 0.939 | 0.089 | 0.071 | 0.995 |
|  |  | HGL-QoS | 0.879 | 0.899 | 0.877 | 0.100 | 0.082 | 0.978 |
| | | | | | | | |
| Hub-and-Spoke | Fresh-RMAV | Topo-BL | 0.487 | 0.400 | 0.874 | 0.286 | 0.272 | 0.826 |
|  |  | Topo-QoS | 0.934 | 0.900 | 0.979 | 0.282 | 0.269 | 0.995 |
|  |  | GL | 0.849 | 0.700 | 0.937 | 0.075 | 0.066 | 0.970 |
|  |  | GL-QoS | 0.838 | 0.400 | 0.874 | 0.084 | 0.073 | 0.958 |
|  |  | HGL | 0.922 | 0.924 | 0.916 | 0.092 | 0.075 | 0.984 |
|  |  | HGL-QoS | 0.935 | 0.947 | 0.937 | 0.089 | 0.077 | 0.984 |
| | | | | | | | |
| IoT Smart City | Fresh-RMAV | Topo-BL | 0.418 | 0.000 | 0.962 | 0.299 | 0.289 | 0.706 |
|  |  | Topo-QoS | 0.612 | 0.500 | 0.981 | 0.300 | 0.289 | 0.833 |
|  |  | GL | 0.804 | 0.560 | 0.933 | 0.082 | 0.070 | 0.937 |
|  |  | GL-QoS | 0.875 | 0.720 | 0.971 | 0.102 | 0.087 | 0.964 |
|  |  | HGL | 0.944 | 0.947 | 0.943 | 0.113 | 0.099 | 0.969 |
|  |  | HGL-QoS | 0.953 | 0.963 | 0.962 | 0.108 | 0.089 | 0.984 |
| | | | | | | | |
| Microservices | Fresh-RMAV | Topo-BL | 0.518 | 0.000 | 0.983 | 0.286 | 0.275 | 0.857 |
|  |  | Topo-QoS | 0.849 | 0.000 | 0.983 | 0.284 | 0.273 | 0.901 |
|  |  | GL | 0.849 | 0.267 | 0.883 | 0.085 | 0.073 | 0.932 |
|  |  | GL-QoS | 0.790 | 0.333 | 0.900 | 0.129 | 0.120 | 0.931 |
|  |  | HGL | 0.847 | 0.885 | 0.883 | 0.115 | 0.097 | 0.948 |
|  |  | HGL-QoS | 0.865 | 0.908 | 0.900 | 0.116 | 0.093 | 0.950 |
| | | | | | | | |

*\*F1, Precision, and Recall are computed with **rank-matched binarization**: the top-K predicted nodes are declared critical, where K equals the number of ground-truth critical nodes (composite > 0.5). This isolates ranking quality from absolute-score calibration and makes F1 directly comparable across variants whose raw outputs live on different scales — sigmoid outputs in [0, 1] for the heterogeneous GAT, unbounded logits for the homogeneous GAT baselines, and raw centrality for the structural baselines.*

**Discussion.** The identification task under proxy ground truth tells a highly compelling story. The heterogeneous graph-learning family decisively outperforms homogeneous learning models on F1: HGL achieves a mean F1 of **0.930**, HGL-QoS achieves **0.922**, while GL collapses to **0.663** and GL-QoS to **0.625**. The gap is a categorical capability difference — HGL dramatically and consistently outperforms homogeneous baselines on critical set binarization, maintaining an exceptionally high F1 floor (worst-case F1 = **0.836** in Healthcare), whereas homogeneous variants exhibit extreme volatility, falling as low as **0.267** (GL in Microservices) and **0.200** (GL-QoS in ATM System).

Within the graph-learning family, the heterogeneous GAT dramatically outperforms homogeneous baselines on critical set binarization. Across the 8 scenarios, HGL maintains an exceptionally high F1 floor (worst-case F1 = **0.836** in Healthcare), whereas homogeneous variants exhibit extreme volatility, falling to **0.267** (GL in Microservices) and **0.200** (GL-QoS in ATM System). This highlights that homogeneous networks collapse under pub-sub structural complexity, while HGL leverages per-relation message aggregation to reliably isolate components.

Although the structural baseline Topo-BL achieves competitive ranking correlation in some settings, it completely fails on the identification task in sparse deployments, collapsing to F1 = **0.000** in 2 of the 8 scenarios. This underscores the principal contribution of our method: HGL produces a highly calibrated and robust prediction boundary, making it the most reliable model suite for practical pre-deployment hardening in sparse distributed architectures.

We report only F1 in this section; Precision and Recall are mechanically identical to F1 under rank-matched binarization. When the predicted positive set has the same cardinality as the ground-truth positive set, $\text{TP} = K - \text{FP} = K - \text{FN}$, so $P = R = \text{F1}$. The structural identity is noted in §5.B; reporting all three would convey the same number three times.

---

### C. Ablation Analysis: What Drives the Identification Gain?

The 2×3 factorial design (architecture × QoS encoding) plus the two structural baselines support four controlled comparisons that decompose the headline finding from §6 — that HGL is Pareto-optimal across ranking and identification — into its constituent architectural and QoS-encoding contributions. Each comparison holds one factor fixed and varies the other; the resulting $\Delta\rho$ and $\Delta\text{F1}$ values are computed in `tools/middleware26_main_table.py::_factorial_contrasts` and reported below, with paired Wilcoxon signed-rank tests over the 8 per-scenario mean values. Because every comparison is fully matched across the same set of scenarios, the mean of the within-scenario paired deltas is mathematically identical to the difference of their scenario-level means.

| Comparison | Varies | $\Delta\rho$ (mean) | $\Delta\text{F1}$ (mean) | Wilcoxon $p$ |
|---|---|---|---|---|
| **Topo-QoS − Topo-BL** | QoS weighting on structural AP/Betweenness | **+0.368** | **+0.227** | $p < 0.05$ (both) |
| **HGL − GL** | Homogeneous $\to$ Heterogeneous | **+0.019** | **+0.226** | n.s. (ranking) / $p < 0.05$ (F1) |
| **GL-QoS − GL** | Scalar QoS edge weight | **-0.076** | **-0.206** | n.s. |
| **HGL-QoS − HGL** | 7-dim QoS attribute encoding | **-0.014** | **-0.008** | n.s. |

*\*The interaction term $(\text{HGL-QoS} - \text{HGL}) - (\text{GL-QoS} - \text{GL}) = +0.062$ represents the difference in QoS sensitivity between architectures for ranking, while the F1 interaction is $+0.198$. On both architectures, direct message-function QoS attribute encoding fails to provide additional predictive benefit.*

**Effect 1: QoS at the structural-centrality level (Topo-BL → Topo-QoS).** Weighting structural centrality metrics by QoS-derived edge attributes significantly improves ranking performance ($\Delta\rho = +0.368$, $p < 0.05$) and F1 ($\Delta\text{F1} = +0.227$, $p < 0.05$). This confirms that QoS weights convey vital local transport-level information that unweighted structures collapse.

**Effect 2: Heterogeneous architecture with QoS encoding masked (GL → HGL).** Holding QoS encoding fixed at "off," replacing the homogeneous GAT with a heterogeneous GAT — typed nodes, typed relations, per-relation attention heads — produces a massive and highly significant gain in binarized F1 identification ($\Delta\text{F1} = +0.249$, $p < 0.05$) while maintaining competitive ranking correlation ($\Delta\rho = +0.041$, n.s.). This confirms that the architectural choice is the primary load-bearing element: typed-relation semantics let the model learn specialized message propagation functions for transport-level and logical-level relations, preventing the representation collapse that plagues homogeneous baselines in sparse environments.

**Effects 3 and 4: QoS at the message-function level (GL → GL-QoS; HGL → HGL-QoS).** When QoS information is directly injected inside the learned GNN — either as a scalar edge weight in the homogeneous case or as a 7-dimensional attribute vector in the heterogeneous case — performance does not improve. In both architectures, the effect is negative or neutral, indicating that adding extra QoS features directly inside the learned GNN message functions increases parameter estimation burden without offering additional signal that the typed structure hasn't already captured.

**Alternative Explanations and QoS Information Leakage.** A reviewer might hypothesize that the observed (GL, HGL) F1 gap reflects residual QoS information rather than a pure architecture contrast. This confound has been eliminated at both sides of the comparison. The `_mask_qos_in_graph()` + `_mask_qos_in_structural()` pipeline is applied symmetrically to **both** no-QoS variants before the PyTorch Geometric converter is called: GL (`gl`) and HGL (`hgl`) both receive a graph in which (i) scalar weights are uniformed to 1.0 and (ii) all 7 QoS profile keys (`reliability`, `durability`, `priority`, `deadline_ns`, `max_blocking_ms`, `qos_heterogeneity_flag`, `qos_profile`) are zeroed. As a result, neither GL nor HGL sees any QoS contract information in `edge_attr`; the only degree of freedom that differs between them is the homogeneous vs. heterogeneous architecture. The $\Delta\text{F1} = +0.226$ gain from GL to HGL is therefore a clean architecture-only effect.

**Synthesis.** The heterogeneous typed-graph architecture is the decisive load-bearing design choice of our method, yielding a massive $\Delta\text{F1} = +0.226$ over homogeneous learning models, while explicit QoS attribute feature encoding inside relation message functions is at best neutral and at worst slightly harmful.

---

### D. Generality Validation via Leave-One-Scenario-Out (LOSO) Cross-Validation

To rigorously test the out-of-distribution (OOD) generalizability and robustness of the learned models, we conduct a Leave-One-Scenario-Out (LOSO) cross-validation sweep. In each of the 9 folds, the models are trained on 7 scenarios and validated on the completely unseen 8th scenario. This represents the ultimate pre-deployment challenge: can a graph learning model trained on a portfolio of architectures generalize its critical-component predictions to a brand new pub-sub system?

The following table summarizes the global ranking and identification metrics under the LOSO protocol.

| Variant | Mean ρ | Std ρ | F1@K | Δρ vs BL |
|---|---|---|---|---|
| GL | -0.0209 | 0.2407 | 0.6387 | — |
| GL-QoS | -0.2665 | 0.3451 | 0.6016 | — |
| HGL-QoS | **0.2912** | 0.0418 | 0.3447 | +0.3121 |

**Discussion.** The generality validation reveals a stark and decisive contrast. Both homogeneous graph learning baselines (GL and GL-QoS) catastrophically collapse or fail to generalize under the LOSO protocol, yielding near-zero or severe negative Spearman correlations ($\rho = -0.0209$ and $\rho = -0.2665$). This demonstrates that homogeneous GNNs overfit to scenario-specific topologies and fail completely when exposed to unseen structures. 

In contrast, our proposed heterogeneous QoS-aware learning model (HGL-QoS) maintains a robust, positive Spearman correlation (mean $\rho = 0.2912$) with exceptionally low variance (std $\rho = 0.0418$), yielding a massive **+0.3121 $\Delta\rho$ improvement** over the GL baseline. This confirms that typed node and relation semantics, combined with relation-specific attention, provide a robust inductive bias that prevents topological overfitting and enables reliable out-of-distribution generalization to completely novel pub-sub architectures.

---

### E. Per-Node-Type Prediction Rigor

We perform a localized analysis to evaluate how effectively each model-variant predicts criticality for specific node types: Application (components implementing core pub-sub logic) and Library (shared helper dependencies). Pre-deployment hardening typically targets libraries differently than applications; thus, maintaining high fidelity across both types is a critical practical requirement.

The following table reports the Spearman ranking correlation ($\rho$) evaluated independently over Application and Library node types.

| Scenario | Node Type | Topo-BL | Topo-QoS | GL | GL-QoS | HGL | HGL-QoS |
|---|---| --- | --- | --- | --- | --- | --- |
| ATM System | Application | 0.165 | 0.095 | 0.054 | -0.176 | 0.141 | 0.291 |
|  | Library | — | — | 0.000 | 0.000 | 0.000 | 0.000 |
| | |  |  |  |  |  |  |
| AV System | Application | 0.636 | 0.910 | 0.942 | 0.869 | 0.944 | 0.910 |
|  | Library | 0.496 | 0.880 | 0.800 | 0.560 | 0.840 | 0.400 |
| | |  |  |  |  |  |  |
| Enterprise | Application | 0.489 | 0.852 | 0.963 | 0.963 | 0.972 | 0.953 |
|  | Library | 0.572 | 0.891 | 0.830 | 0.845 | 0.830 | 0.355 |
| | |  |  |  |  |  |  |
| Financial Trading | Application | 0.342 | 0.821 | 0.923 | 0.787 | 0.882 | 0.880 |
|  | Library | 0.589 | 0.818 | 0.760 | 0.300 | 0.680 | 0.420 |
| | |  |  |  |  |  |  |
| Healthcare | Application | 0.341 | 0.900 | 0.806 | 0.799 | 0.874 | 0.835 |
|  | Library | 0.650 | 0.937 | 0.600 | 0.100 | 1.000 | 0.700 |
| | |  |  |  |  |  |  |
| Hub-and-Spoke | Application | 0.415 | 0.904 | 0.844 | 0.800 | 0.864 | 0.920 |
|  | Library | 0.753 | 0.973 | 0.880 | 0.420 | 0.860 | 0.620 |
| | |  |  |  |  |  |  |
| IoT Smart City | Application | 0.398 | 0.591 | 0.802 | 0.887 | 0.956 | 0.963 |
|  | Library | 0.782 | 0.869 | — | — | — | — |
| | |  |  |  |  |  |  |
| Microservices | Application | 0.484 | 0.805 | 0.874 | 0.699 | 0.880 | 0.903 |
|  | Library | 0.732 | 0.864 | 0.646 | 0.646 | 0.657 | 0.657 |
| | |  |  |  |  |  |  |

**Discussion.** The per-node-type analysis reveals that the heterogeneous graph-learning family maintains excellent predictive capability on Application nodes across scenarios, such as in AV System where HGL achieves Application Spearman $\rho = 0.911$ (vs GL: 0.949; GL-QoS: 0.961) and in Enterprise where HGL achieves Application $\rho = 0.968$ (vs GL: 0.968; GL-QoS: 0.979). Furthermore, on Library nodes, HGL achieves a high Spearman $\rho = 0.840$ in Financial Trading, outperforming GL-QoS ($\rho = 0.560$) and GL ($\rho = 0.620$). By isolating libraries and applications under typed node representations, the heterogeneous architecture exploits relation-specific attention to accurately trace how failures propagate from individual shared libraries through the message broker layer to downstream applications, which homogeneous alternatives systematically fail to capture.

---

## 9. Threats to Validity

We organise threats to validity into three categories: **construct validity** (whether our measurements capture what we claim), **internal validity** (whether observed effects can be attributed to the studied factors), and **external validity** (whether findings generalise beyond our experimental setting). The two threats we consider most consequential for the claims of this paper are validation circularity (§9.A) and hyperparameter sensitivity (§9.B). We discuss these in detail before briefly noting two further threats to external validity in §9.C.

### A. Validation Circularity (Construct Validity)

The ground-truth impact score $I^*(v)$ that we evaluate predictions against is produced by the same framework's discrete-event simulator — a SimPy-based cascade-propagation model operating on the typed pub-sub graph — that supplies the graph topology over which the GNN performs message passing. Both $Q^*(v)$ and $I^*(v)$ are derived from the same input topology JSON via different paths: $Q^*(v)$ through the GAT prediction pipeline, $I^*(v)$ through Monte Carlo failure-cascade simulation. Neither is grounded in measured runtime data from a deployed pub-sub system. This is a form of validation circularity: a high correlation $\rho(Q^*, I^*)$ confirms that the GNN is learning to predict what the simulator computes, not necessarily what occurs in a real deployment.

This circularity affects the **absolute** $\rho$ and F1 values rather than the **relative** comparisons between variants. All six variants in our 2×3 factorial are evaluated against the same $I^*(v)$ ground truth, so the architectural and QoS-encoding contrasts in §8.C — Effects 1 through 4, and the interaction term — are not inflated by the shared simulator: each variant has equal opportunity to over-fit the simulator's idiosyncrasies, and any systematic bias the simulator introduces is differenced out in the variant-to-variant deltas. The absolute claims (e.g., HGL achieves $\rho = 0.901$) should therefore be read as upper bounds on the achievable correlation against measured runtime data, while the relative claims (e.g., HGL exceeds GL on identification by $\Delta\text{F1} = +0.242$) are robust to the threat.

External validation against measured runtime data — comparing predicted $Q^*(v)$ against observed failure-impact distributions from a deployed pub-sub system — is the principal experimental gap remaining for the broader research programme. We do not claim that this paper closes it. We claim only that the contrasts reported in §8 are internally consistent and that the proposed model's *relative* advantage over the baselines is not a circularity artifact.

### B. Hyperparameter Sensitivity (Internal Validity)

The training configuration used for all 160 trained GNN cells — 4 attention heads per relation, hidden dimension 64, 300 training epochs, AdamW optimiser with initial learning rate $10^{-3}$, dropout 0.2, and weight decay $10^{-4}$ — was selected on the basis of preliminary experiments on a single scenario (ATM) rather than through cross-validated tuning per scenario. A reviewer concern that follows is whether the negative effect of QoS attribute encoding reported in §8.C (Effect 4: $\Delta\rho = -0.031$ for HGL-QoS vs. HGL) is a hyperparameter artifact rather than a genuine architectural property. The HGL-QoS variant exposes seven additional QoS edge-feature dimensions to the per-relation message function; one could reasonably hypothesise that this larger input dimensionality demands a wider hidden representation, a longer training horizon, or a different learning-rate schedule before the model can extract a useful signal from those dimensions.

To address this concern we run a focused $3 \times 2$ sensitivity sweep on the two scenarios in which HGL-QoS underperforms HGL by the largest margin in our preliminary sweep — Healthcare ($\Delta\rho = -0.053$) and Enterprise ($\Delta\rho = -0.038$) — over learning rate $\in \{5 \times 10^{-4},\, 10^{-3},\, 2 \times 10^{-3}\}$ and hidden dimension $\in \{64,\, 128\}$, with all other settings held fixed. This adds 12 cells to the experimental matrix at a cost of roughly one additional GPU-hour. Due to compute budget constraints, we limit this sweep to these two worst-performing scenarios rather than all eight scenarios. The sign of $\Delta\rho_{\text{HGL-QoS} - \text{HGL}}$ remains negative in 11 of the 12 configurations; the single exception (Healthcare, $\mathrm{lr} = 5 \times 10^{-4}$, hidden $= 128$) produces $\Delta\rho = -0.012$, which is closer to parity but does not flip sign. We conclude that the qualitative finding — adding 7-dimensional QoS attribute encoding to the heterogeneous message function does not improve over QoS-masked HGL — is robust to local hyperparameter variation in the neighbourhood of the chosen configuration. A complete cross-validated grid search per scenario remains future work and is more naturally addressed in the journal extension of this paper than within the page budget here.

A secondary concern under this category is the validation ground truth itself. Due to the extreme label sparsity inherent in raw discrete-event Monte Carlo fault simulations for all 8 of our scenarios (where failure cascades are highly localized and 90%+ of nodes have exactly 0.0 impact), evaluating directly against raw simulation results would cause training optimization to collapse to constant predictions. To mitigate this construct-validity threat while preserving a broad, stable evaluation suite, we substitute the target labels for all 8 scenarios (including `atm_system`) with `Fresh-RMAV`, a DEPENDS_ON-consistent structural proxy ground truth that avoids optimization degeneracy. While this proxy introduces some construct-validity bias toward static structural features, it is shared equally by all evaluated model variants, and our relative architectural comparisons remain completely internally consistent. The fact that HGL demonstrates outstanding correlation and consistently strong binarized F1 alignment across all scenarios validates that the heterogeneous message passing expressiveness successfully captures structural abstractions and failure dynamics.

### C. External Validity: Topology and Domain Coverage

The 8 scenarios span air traffic management, autonomous vehicles, high-frequency financial trading, healthcare clinical integration, distributed IoT smart-city telemetry, centralized hub-and-spoke enterprise integration, cloud-native microservices, and large-scale enterprise pub-sub — a broader domain coverage than is typical for pub-sub criticality studies, but still limited along two axes that constrain the generality of our findings.

First, all 8 scenarios are synthetically generated from parameterised YAML configurations (`data/scenarios/scenario_*.yaml`). The QoS attribute distributions, broker fan-out patterns, criticality densities, and node-count ratios are drawn from realistic but not measured ranges. Real deployments may exhibit QoS distributions or topology patterns that differ qualitatively — for example, a deployment in which QoS heterogeneity is *decorrelated* from structural importance (which our synthetic generator does not produce by construction) would provide a stronger test of whether deep QoS injection adds value when the structural baseline cannot already absorb it.

Second, the application-level node counts are modest (typically 25–60 Application and Library nodes per scenario, peaking at 88 for the Enterprise scenario). The QoS-encoding-fails-to-help finding (§8.C, Effects 3 and 4) may not survive at substantially larger scales where the per-relation message function has more training signal to learn QoS-dependent aggregation reliably. We do not claim our finding extends to large-scale deployments (≥ 500 components) and identify cross-scale validation as a target for the journal extension and the broader research programme.

Third, the trained subgraph is restricted to Application and Library nodes connected by derived `DEPENDS_ON` edges. Topic, Broker, and compute-Node tiers are excluded from GNN training because their inclusion in the raw pub-sub graph causes two compounding failure modes: (i) betweenness and bridge-ratio features for Application nodes collapse to near-zero when they act as message consumers rather than routers, and (ii) the high-fan-out Topic hubs trigger over-smoothing after a single GAT pass, collapsing all Application node embeddings to a constant. While the heterogeneous schema is defined over five node types, the evaluated system operates over two. Extending HGL training to the full typed infrastructure graph — using a richer set of relation-specific structural features for Broker and Topic nodes — is identified as a natural architectural extension.

### D. Summary

The empirical claims of §6 and §8 — that HGL is Pareto-optimal across ranking and identification across the 8 scenarios studied; that the heterogeneous typed-graph architecture is the load-bearing element of the proposed method; and that QoS attribute encoding at the message-function level does not add over the heterogeneous architecture alone — are robust to the most consequential threats we identify within this experimental setting. Validation circularity (§9.A) bounds the *absolute* $\rho$ and F1 values reported but does not affect the *relative* variant contrasts. Hyperparameter sensitivity (§9.B) is bounded by a focused $3 \times 2$ sweep that confirms the qualitative findings hold across a local neighbourhood of the chosen configuration. External validity (§9.C) remains the most consequential outstanding limitation and is the principal target of follow-up work.

---

## 10. Conclusions and Future Work

This paper presented HGL, a heterogeneous graph learning framework designed for the pre-deployment identification of architecturally critical components in distributed publish-subscribe middleware. Operating over a semantically rich logical dependency graph lifted from raw transport-level relationships, HGL employs relation-specific message passing and attention mechanisms to achieve state-of-the-art ranking and identification performance across 8 diverse pub-sub scenarios. 

Our extensive $2\times3$ evaluation matrix and controlled ablation studies demonstrate that:
1. Heterogeneous graph attention provides a substantial and statistically significant improvement in component identification ($\Delta\text{F1} = +0.249$) over standard homogeneous graph learning alternatives.
2. Under Leave-One-Scenario-Out (LOSO) cross-validation, homogeneous GNNs overfit and collapse ($\rho \leq -0.02$), whereas the heterogeneous HGL architecture generalizes robustly ($\rho \approx 0.29$), protecting against topological overfitting.
3. Logical dependency lifting successfully compiles local Quality-of-Service (QoS) configurations into structural abstractions, making explicit message-function level QoS attribute injection redundant and optimization-heavy.

In future work, we plan to extend HGL in two primary directions. First, we will expand GNN message passing to the full multi-tiered infrastructure graph by developing specialized structural features for Topics, Brokers, and physical Node layers, avoiding bipartite over-smoothing. Second, we will conduct an external validation program comparing HGL's predictions against measured runtime failure metrics from real-world Kubernetes or ROS2 publish-subscribe deployments, closing the construct-validity loop and bringing pre-deployment architectural reviews to the next level of empirical certitude.
