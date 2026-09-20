# 6. Experimental Setup

## 6.1 Datasets and System Corpus

The evaluation corpus comprises 2,812 components across seventeen system architectures, including twelve synthetic topologies that form the inductive cross-validation folds and five real-world reference systems withheld from all training procedures, as detailed in Table 5. To establish broad external validity across modern distributed computing paradigms, the benchmark deliberately spans six architectural domains: robotics (autonomous vehicles / ROS 2), smart cities, healthcare integration, financial trading, enterprise application integration (centralized ESB broker hubs and microservices), and industrial/edge IoT (SCADA, EdgeX Foundry, Home Assistant; detailed in Table S11 of the Supplementary Material).

**Table 5.** Overview of the evaluation corpus. The twelve synthetic topologies correspond to the inductive Leave-One-Scenario-Out folds described in Table 8, and the five real-world systems are excluded from all training folds and used exclusively for zero-shot transfer (§7.4). Per-scenario entity and edge counts are obtained from the committed topology files and verified through continuous integration.

| **Dataset**                            | **$|V|$** | **$|V_{\text{app}}|$** | **Topics** | **Brokers** | **Hosts** | **Libs** |  **$|E|$** |
|:---------------------------------------|----------:|-----------------------:|-----------:|------------:|----------:|---------:|-----------:|
| Synthetic evaluation scenarios (11)    |     2,387 |                  1,295 |        588 |          60 |       194 |      250 |     10,657 |
| Synthetic case study — ATM (1)         |        74 |                     26 |         27 |           5 |         8 |        8 |        261 |
| **Synthetic subtotal (12 LOSO folds)** | **2,461** |              **1,321** |    **615** |      **65** |   **202** |  **258** | **10,918** |
| **Real-world subtotal (5 systems)**    |   **351** |                **141** |    **120** |      **16** |    **32** |   **42** |    **700** |
| **Total**                              | **2,812** |              **1,462** |    **735** |      **81** |   **234** |  **300** | **11,618** |

### 6.1.1 Reproducibility of the Corpus

The benchmark corpus is designed to be fully regenerable rather than statically archived. Each dataset is deterministically generated from its configuration file using the following procedure:

> `python cli/generate_graph.py batch –input-dir data/scenarios –output-dir <path>`

A companion manifest records the random seed, entity counts, git commit hash, and a SHA-256 cryptographic digest for each emitted topology. Continuous integration regression tests verify that every committed dataset regenerates byte-identically from its configuration and that all disk digests match the manifest. This procedure makes sure that third parties can reproduce the exact graphs used in these experiments, rather than sampling from similar distributions.

## 6.2 Baselines and Evaluated Predictors

Four primary predictor configurations, drawn from three distinct families, are evaluated alongside reference baselines. Table 6 provides a structured taxonomy of the evaluated predictors, their underlying graph representations, edge feature encodings, and empirical roles. Predictor names indicate the model family and substrate: an `-N` infix denotes a model trained on the complete native multigraph, its absence denotes the obtained Application–Library flow projection, and a `-QoS` suffix denotes a configuration that consumes declared QoS contracts. SaG throughout denotes the overall framework, not an individual predictor.

**Table 6.** Taxonomy of evaluated predictors and reference baselines. The `-N` infix denotes execution on the complete native multigraph; `-QoS` indicates inclusion of middleware QoS attributes.

| **Predictor**   | **Evaluation Substrate**                |   **Typing**   | **Edge Features** | **Parameters** | **Trained?** | **Empirical Role**                                 |
|:----------------|:----------------------------------------|:--------------:|:-----------------:|:--------------:|:------------:|:---------------------------------------------------|
| **Topo**        | $G_{\text{analysis}}$ (Flow Projection) |       No       |       None        |       0        |      No      | Unweighted topological baseline                    |
| **Topo-QoS**    | $G_{\text{analysis}}$ (Flow Projection) |       No       |   Scalar $w(e)$   |       0        |      No      | QoS-weighted closed-form benchmark                 |
| **GAT**         | $G_{\text{analysis}}$ (Flow Projection) |  Homogeneous   |       None        |     28,168     |     Yes      | In-distribution counterpart of GAT-N (Table 7)     |
| **GAT-QoS**     | $G_{\text{analysis}}$ (Flow Projection) |  Homogeneous   |   Scalar $w(e)$   |     28,168     |     Yes      | In-distribution counterpart of GAT-N-QoS (Table 7) |
| **GAT-N**       | Native Multigraph                       |  Homogeneous   |       None        |     28,168     |     Yes      | Untyped, unweighted GNN floor                      |
| **GAT-N-QoS**   | Native Multigraph                       |  Homogeneous   |   Scalar $w(e)$   |     28,168     |     Yes      | Isolates QoS channel without typing                |
| **HGT**         | Native Multigraph                       | Heterogeneous  |  Relation 1-hot   |    434,620     |     Yes      | Isolates relational typing without QoS             |
| **HGT-QoS**     | Native Multigraph                       | Heterogeneous  |  16-D QoS Vector  |    434,620     |     Yes      | Proposed full learned model                        |
| **RM / $Q(v)$** | $G_{\text{analysis}}$ (Analysis Graph)  | Per-type rules |   Scalar $w(e)$   |       0        |      No      | Diagnostic attribution reference                   |

Table 6 summarizes the evaluated predictors, graph representations, and empirical roles across three families: heterogeneous graph learning (`HGT-QoS` with 16-D QoS edge vectors, and its ablation `HGT`), homogeneous graph learning (`GAT-N-QoS` with scalar $w(e)$, and unweighted `GAT-N`), and training-free structural baselines (`Topo-QoS` and unweighted `Topo` on the `DEPENDS_ON` flow projection). The `-N` infix denotes the native multigraph substrate; on flow projections, homogeneous variants are denoted `GAT`/`GAT-QoS` (Table 7). The out-of-distribution evaluation (Table 8) additionally reports **RM** ($Q(v)$, §5) as a diagnostic reference baseline. Deterministic RM scoring also drives sensitivity sweeps in §7.3.

### 6.2.1 Evaluation Substrates and Parity Guarantee

Predictors in this study operate on matched substrates within their respective families:

**Graph Learning Models (`GAT-N-QoS`, `HGT-QoS`):** Under Leave-One-Scenario-Out (Table 8), both learned predictors ingest the complete native typed multigraph across all five entity types (recorded using the shared `-N` infix). In-distribution (Table 7), GAT/GAT-QoS consume the Application–Library `DEPENDS_ON` projection, confounding typing with multi-entity visibility. `GAT-N-QoS` uses per-type projection with untyped GATConv across edges, whereas `HGT-QoS` uses relation-specific `HGTConv` weights. The edge channel also differs: `GAT-N-QoS` consumes a scalar $w(e)$, while `HGT-QoS` consumes all 16 dimensions. The parameter budget ($434{,}620$ vs. $28{,}168$) and directionality also remain open confounds (§8.4).

**Training-Free Structural Baselines (Topo, `Topo-QoS`):** We evaluate topological baselines on the obtained Application–Library `DEPENDS_ON` projection (§3.2), since raw multigraphs route messages via topics/brokers, leaving Application nodes with near-zero betweenness.

**Ground-Truth Independence Guarantee:** Crucially, **no predictor in either group accesses $G_{\text{structural}}$**: that raw topology is strictly reserved for simulation oracles (§4.4), verified by `tests/test_independence_guarantee.py`. Regardless of substrate, all variants are scored on the same independently resolved Application node set ($V_{\text{app}}$, §6.3).

## 6.3 Evaluation Metrics and Protocols

**Figure accessibility.** All figures employ the Okabe–Ito palette with distinct markers and hatchings to ensure monochrome legibility.

**Ranking Precision:** Evaluated via Spearman $\rho$ and Kendall $\tau$ against the simulated impact $I^*(v)$ from the primary oracle (§4.3). **Critical-Set Identification:** Measured via $F_1@K$, Precision@$K$, and Recall@$K$ for top-$K$ components ($K = \text{round}(0.20 \cdot |V_{\text{app}}|)$). Because the predicted and reference sets are both of size $K$, the three coincide identically and reduce to top-$K$ set overlap. **Statistical Significance:** Paired Wilcoxon signed-rank tests [85] ($p < 0.05$) and bootstrap 95% CIs ($B = 2{,}000$) over folds [86, 87]. In the 12-fold LOSO design, the power floor is $p = 0.00049$. Applying Holm’s step-down correction across ten full-population rank contrasts (§§7.1–7.3.1), three survive: `Topo-QoS` over Topo ($p = 0.0010$), Topo over RM ($p = 0.0005$), and unweighted typing HGT over `GAT-N` ($p = 0.0005$). The two contrasts that carry the QoS channel into an already-typed model do not: typing with the QoS channel present reaches only $p = 0.1294$ (9/12 folds) and the QoS edge ablation under typing $p = 0.2036$ (10/12), as Table 10 reports and §7.3.1 discusses.

**Pre-registration.** The primary out-of-distribution contrast (`HGT-QoS` vs. `Topo-QoS` under LOSO, 5 fixed seeds, fold as unit of analysis) was pre-registered in the replication package prior to obtaining results. As reported in §7.1, the margin did not reach statistical significance.

### Evaluation Population and Protocols

Each predictor within an evaluation table is scored on an identical node population, resolved strictly from scenario topology and ground truth, specifically the **Application** set ($V_{\text{app}}$) unless otherwise noted. Pooling node types conflates distinct base rates and can trigger Simpson’s paradox (§7.3).

**In-Distribution Evaluation:** Stratified 60% train / 20% val / 20% test node splits over five seeds $\{42, 123, 456, 789, 2024\}$, redrawing partitions and initializations. **Inductive LOSO Cross-Validation:** Models are trained on eleven scenarios and test zero-shot on the held-out twelfth across all 12 folds under equal 3-layer depth and inner-split early stopping (§8.4). **Real-World Architectural Transfer:** Synthetic-trained models are evaluated zero-shot on five open-source systems without fine-tuning.
