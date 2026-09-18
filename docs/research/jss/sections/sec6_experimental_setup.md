# 6. Experimental Setup

## 6.1 Datasets and System Corpus

The evaluation corpus comprises 2,812 components across seventeen system architectures, including twelve synthetic topologies that form the inductive cross-validation folds and five real-world reference systems withheld from all training procedures, as detailed in Table 3.

**Table 3.** Overview of the evaluation corpus. The twelve synthetic topologies correspond to the inductive Leave-One-Scenario-Out folds described in Table 5, and the five real-world systems are excluded from all training folds and used exclusively for zero-shot transfer (§7.4). Per-scenario entity and edge counts are obtained from the committed topology files and verified through continuous integration.

| **Dataset**                             | **$|V|$** | **$|V_{\text{app}}|$** | **Topics** | **Brokers** | **Hosts** | **Libs** |  **$|E|$** |
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

Four primary predictor configurations, drawn from three families, are evaluated. Predictor names indicate the family and substrate: an -N suffix denotes a model trained on the native multigraph, its absence denotes the obtained Application–Library flow projection, and a -QoS suffix denotes a configuration that consumes declared QoS contracts. SaG throughout denotes the framework, not an individual predictor.

1.  **Heterogeneous graph learning (typed `HGT`).** **HGT-QoS** (proposed): relation-specific Heterogeneous Graph Transformer (§4) ingesting the complete native multigraph with 16-dimensional continuous-categorical edge features that encode middleware QoS contracts. We report its ablation, HGT, which masks those QoS dimensions, in §7.3.1.

2.  **Homogeneous graph learning (untyped `GAT`).** **GAT-N-QoS**: homogeneous Graph Attention Network [65] trained on the identical native multigraph substrate with per-type input projections, but untyped, single-relation message passing. Its edge channel carries the scalar QoS aggregate $w(e)$ — dimension $0$ of the same 16-D encoding `HGT-QoS` consumes — rather than the per-dimension decomposition; no homogeneous architecture in our suite ingests the full 16-D vector. The `HGT-QoS`–`GAT-N-QoS` contrast therefore isolates the joint contribution of relational typing and per-dimension QoS encoding. §7.3.1 separates the second factor within the typed architecture, and the corresponding unweighted ablation is `GAT-N`. The -N suffix denotes the native substrate and is load-bearing: the same homogeneous architecture run on the `DEPENDS_ON` projection is reported as GAT / GAT-QoS, and Table 4 reports that pair.

3.  **QoS-weighted structural baseline (training-free).** **Topo-QoS**: QoS-weighted topological centrality evaluated on the obtained application flow projection.

4.  **Unweighted structural baseline (training-free).** **Topo**: structural centrality combining unweighted betweenness centrality and articulation point scoring on the flow projection.

Additionally, the out-of-distribution evaluation (Table 5) reports RM ($Q(v)$, the deterministic hierarchical quality attribution model of §5) as a diagnostic reference baseline. RM is not fitted to rank failure impact; its inclusion shows the added value of learned relational prediction compared with static structural attribution (§1.2). Deterministic RM scoring also drives every sensitivity sweep in §7.3, where closed-form formulations isolate parameter effects from neural training stochasticity.

### 6.2.1 Evaluation Substrates and Parity Guarantee

Predictors in this study operate on matched substrates within their respective families:

**Graph Learning Models (`GAT-N-QoS`, `HGT-QoS`):** Under Leave-One-Scenario-Out (Table 5), both learned predictors ingest the complete native typed multigraph across all five entity types (recorded using the shared -N suffix). In-distribution (Table 4), GAT/GAT-QoS consume the Application–Library `DEPENDS_ON` projection, confounding typing with multi-entity visibility. `GAT-N-QoS` uses per-type projection with untyped GATConv across edges, whereas `HGT-QoS` uses relation-specific `HGTConv` weights. The edge channel also differs: `GAT-N-QoS` consumes a scalar $w(e)$, while `HGT-QoS` consumes all 16 dimensions. The parameter budget ($434{,}620$ vs. $28{,}168$) and directionality also remain open confounds (§8.4).

**Training-Free Structural Baselines (Topo, `Topo-QoS`):** We evaluate topological baselines on the obtained Application–Library `DEPENDS_ON` projection (§3.2), since raw multigraphs route messages via topics/brokers, leaving Application nodes with near-zero betweenness.

**Ground-Truth Independence Guarantee:** Crucially, **no predictor in either group accesses $G_{\text{structural}}$**: that raw topology is strictly reserved for simulation oracles (§4.4), verified by . Regardless of substrate, all variants are scored on the same independently resolved Application node set ($V_{\text{app}}$, §6.3).

## 6.3 Evaluation Metrics and Protocols

**Figure accessibility.** All figures employ the Okabe–Ito palette with distinct markers and hatchings to ensure monochrome legibility.

**Ranking Precision:** Evaluated via Spearman $\rho$ and Kendall $\tau$ against the simulated impact $I^*(v)$ from the primary oracle (§4.3). **Critical-Set Identification:** Measured via $F_1@K$, Precision@$K$, and Recall@$K$ for top-$K$ components ($K = \text{round}(0.20 \cdot |V_{\text{app}}|)$). Because the predicted and reference sets are both of size $K$, the three coincide identically and reduce to top-$K$ set overlap. **Statistical Significance:** Paired Wilcoxon signed-rank tests [80] ($p < 0.05$) and bootstrap 95% CIs ($B = 2{,}000$) over folds [81, 82]. In the 12-fold LOSO design, the power floor is $p = 0.00049$. Applying Holm’s step-down correction across ten full-population rank contrasts (§§7.1–7.3.1), three survive: `Topo-QoS` over Topo ($p = 0.0010$), Topo over RM ($p = 0.0005$), and unweighted typing HGT over `GAT-N` ($p = 0.0005$). The two contrasts that carry the QoS channel into an already-typed model do not: typing with the QoS channel present reaches only $p = 0.1294$ (9/12 folds) and the QoS edge ablation under typing $p = 0.2036$ (10/12), as Table 7 reports and §7.3.1 discusses.
**Pre-registration.** The primary out-of-distribution contrast (`HGT-QoS` vs. `Topo-QoS` under LOSO, 5 fixed seeds, fold as unit of analysis) was pre-registered in the replication package prior to obtaining results. As reported in §7.1, the margin did not reach statistical significance.

### Evaluation Population and Protocols

Each predictor within an evaluation table is scored on an identical node population, resolved strictly from scenario topology and ground truth, specifically the **Application** set ($V_{\text{app}}$) unless otherwise noted. Pooling node types conflates distinct base rates and can trigger Simpson’s paradox (§7.3).

**In-Distribution Evaluation:** Stratified 60% train / 20% val / 20% test node splits over five seeds $\\{42, 123, 456, 789, 2024\\}$, redrawing partitions and initializations. **Inductive LOSO Cross-Validation:** Models are trained on eleven scenarios and test zero-shot on the held-out twelfth across all 12 folds under equal 3-layer depth and inner-split early stopping (§8.4). **Real-World Architectural Transfer:** Synthetic-trained models are evaluated zero-shot on five open-source systems without fine-tuning.