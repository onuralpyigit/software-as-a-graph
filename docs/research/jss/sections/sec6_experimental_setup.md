# 6. Experimental Setup

## 6.1 Datasets and System Corpus

The evaluation corpus comprises 2,812 components across seventeen system architectures, including twelve synthetic topologies that form the inductive cross-validation folds and five real-world reference systems withheld from all training procedures, as detailed in Table 4. The synthetic scenarios span diverse operational domains (autonomous vehicles, financial trading, healthcare integration, industrial SCADA, smart-city IoT, telecom RAN, cloud microservices, and enterprise application integration via centralized broker hubs/ESB; detailed in Supplementary Table S12).

**Table 4.** Overview of the evaluation corpus. The twelve synthetic topologies correspond to the inductive Leave-One-Scenario-Out folds described in Table 6, and the five real-world systems are excluded from all training folds and used exclusively for zero-shot transfer (§7.4). Per-scenario entity and edge counts are obtained from the committed topology files and verified through continuous integration.

| **Dataset**                            | **$|V|$** | **$|V_{\text{app}}|$** | **Topics** | **Brokers** | **Hosts** | **Libs** |  **$|E|$** |
|:---------------------------------------|----------:|-----------------------:|-----------:|------------:|----------:|---------:|-----------:|
| Synthetic evaluation scenarios (11)    |     2,387 |                  1,295 |        588 |          60 |       194 |      250 |     10,657 |
| Synthetic case study — ATM (1)         |        74 |                     26 |         27 |           5 |         8 |        8 |        261 |
| **Synthetic subtotal (12 LOSO folds)** | **2,461** |              **1,321** |    **615** |      **65** |   **202** |  **258** | **10,918** |
| **Real-world subtotal (5 systems)**    |   **351** |                **141** |    **120** |      **16** |    **32** |   **42** |    **700** |
| **Total**                              | **2,812** |              **1,462** |    **735** |      **81** |   **234** |  **300** | **11,618** |

### 6.1.1 Reproducibility of the Corpus

The benchmark corpus is designed to be fully regenerable rather than statically archived. Each dataset is deterministically generated from its configuration file via `python cli/generate_graph.py batch` `–input-dir data/scenarios` `–output-dir <path>`. A companion manifest records the random seed, entity counts, git commit hash, and a SHA-256 cryptographic digest for each emitted topology. Continuous integration regression tests verify that every committed dataset regenerates byte-identically from its configuration and that all disk digests match the manifest. This procedure makes sure that third parties can reproduce the exact graphs used in these experiments, rather than sampling from similar distributions.

## 6.2 Baselines and Evaluated Predictors

Four primary predictor configurations, drawn from three distinct families, are evaluated alongside reference baselines. Table 5 provides a structured taxonomy of the evaluated predictors, their underlying graph representations, edge feature encodings, and empirical roles. Predictor names indicate the model family and substrate: an `-N` infix denotes a model trained on the complete native multigraph, its absence denotes the obtained Application–Library flow projection, and a `-QoS` suffix denotes a configuration that consumes declared QoS contracts. SaG throughout denotes the overall framework, not an individual predictor.

**Table 5.** Taxonomy of evaluated predictors and reference baselines. The `-N` infix denotes execution on the complete native multigraph; `-QoS` indicates inclusion of middleware QoS attributes.

| **Predictor**   | **Evaluation Substrate**                |   **Typing**   | **Edge Features** | **Parameters** | **Trained?** | **Empirical Role**                                         |
|:----------------|:----------------------------------------|:--------------:|:-----------------:|:--------------:|:------------:|:-----------------------------------------------------------|
| **Topo**        | $G_{\text{analysis}}$ (Flow Projection) |       No       |       None        |       0        |      No      | Unweighted topological baseline                            |
| **Topo-QoS**    | $G_{\text{analysis}}$ (Flow Projection) |       No       |   Scalar $w(e)$   |       0        |      No      | QoS-weighted closed-form benchmark                         |
| **GAT**         | $G_{\text{analysis}}$ (Flow Projection) |  Homogeneous   |       None        |     28,168     |     Yes      | In-distribution counterpart of GAT-N (Supp. Table S15)     |
| **GAT-QoS**     | $G_{\text{analysis}}$ (Flow Projection) |  Homogeneous   |   Scalar $w(e)$   |     28,168     |     Yes      | In-distribution counterpart of GAT-N-QoS (Supp. Table S15) |
| **GAT-N**       | Native Multigraph                       |  Homogeneous   |       None        |     28,168     |     Yes      | Untyped, unweighted GNN floor                              |
| **GAT-N-QoS**   | Native Multigraph                       |  Homogeneous   |   Scalar $w(e)$   |     28,168     |     Yes      | Isolates QoS channel without typing                        |
| **HGT**         | Native Multigraph                       | Heterogeneous  |  Relation 1-hot   |    434,620     |     Yes      | Isolates relational typing without QoS                     |
| **HGT-QoS**     | Native Multigraph                       | Heterogeneous  |  16-D QoS Vector  |    434,620     |     Yes      | Proposed full learned model                                |
| **RM / $Q(v)$** | $G_{\text{analysis}}$ (Analysis Graph)  | Per-type rules |   Scalar $w(e)$   |       0        |      No      | Diagnostic attribution reference                           |

Table 5 summarizes the evaluated predictors, graph representations, and empirical roles across three families: heterogeneous graph learning (`HGT-QoS` with 16-D QoS edge vectors, and its ablation `HGT`), homogeneous graph learning (`GAT-N-QoS` with scalar $w(e)$, and unweighted `GAT-N`), and training-free structural baselines (`Topo-QoS` and unweighted `Topo` on the `DEPENDS_ON` flow projection). The `-N` infix denotes the native multigraph substrate; on flow projections, homogeneous variants are denoted `GAT`/`GAT-QoS` (Supplementary Table S15). The out-of-distribution evaluation (Table 6) additionally reports **RM** ($Q(v)$, §5) as a diagnostic reference baseline. Deterministic RM scoring also drives sensitivity sweeps in §7.3.

### 6.2.1 Evaluation Substrates and Parity Guarantee

Predictors in this study operate on matched substrates within their respective families:

**Graph Learning Models (`GAT-N-QoS`, `HGT-QoS`):** Under Leave-One-Scenario-Out (Table 6), both learned predictors ingest the complete native typed multigraph across all five entity types (recorded using the shared `-N` infix). In-distribution (Supplementary Table S15), GAT/GAT-QoS consume the Application–Library `DEPENDS_ON` projection, confounding typing with multi-entity visibility. `GAT-N-QoS` uses per-type projection with untyped GATConv across edges, whereas `HGT-QoS` uses relation-specific `HGTConv` weights. The edge channel also differs: `GAT-N-QoS` consumes a scalar $w(e)$, while `HGT-QoS` consumes all 16 dimensions. The parameter budget ($434{,}620$ vs. $28{,}168$) and message directionality ($103{,}725$ reverse parameters) also remain open confounds: four control arms (`GAT-N-C`, `GAT-N-QoS-C`, `GAT-N-QoS16-C`, and `HGT-QoS-U`) are formally specified in Amendment 2 of our registered analysis plan to isolate these factors in future sweeps (§8.4).


**Training-Free Structural Baselines (Topo, `Topo-QoS`):** We evaluate topological baselines on the obtained Application–Library `DEPENDS_ON` projection (§3.2), since raw multigraphs route messages via topics/brokers, leaving Application nodes with near-zero betweenness. Because these two baselines carry every headline contrast in §7, we state them in closed form rather than by name. Both are convex combinations of a path-traversal term and a cut-vertex term:

$$\text{Topo}(v) = 0.6 \cdot \text{BT}(v) + 0.4 \cdot \text{AP}(v),
\qquad
\text{Topo-QoS}(v) = 0.6 \cdot \text{BT}_{w}(v) + 0.4 \cdot \text{AP}(v)$$

where $\text{BT}(v)$ is normalized betweenness centrality on the projection, $\text{AP}(v) \in \{0, 1\}$ indicates whether $v$ is an articulation point of the projection’s undirected form, and $\text{BT}_{w}(v)$ is betweenness computed over edge *distances* $d(e) = 1/(w(e) + \varepsilon)$ with $\varepsilon = 10^{-6}$, so that strongly coupled edges are short and attract shortest paths. The projection itself carries two of the six rules of Table 2: Rule 1 joins a subscriber to each publisher of a topic it consumes, at $w = 1 - \prod_{t \in T}(1 - w(t))$, and Rule 5 joins an application to each library it uses, at the median topic weight.

Two properties of this pair matter for how §7 should be read. First, *only the traversal term is QoS-weighted*: the articulation term is identical in both, so the entire `Topo-QoS` margin is carried by re-weighting shortest paths. Second, the weighting *degenerates gracefully*: when no edge of a graph carries a non-unit weight, $\text{BT}_{w}$ reduces to $\text{BT}$ and `Topo-QoS` coincides with Topo on that graph. The $0.6/0.4$ split is a declared convention, not a fitted parameter, and is not tuned per scenario.

**Ground-Truth Independence Guarantee:** Crucially, **no predictor in either group accesses $G_{\text{structural}}$**: that raw topology is strictly reserved for simulation oracles (§4.4), verified by `tests/test_independence_guarantee.py`. Regardless of substrate, all variants are scored on the same independently resolved Application node set ($V_{\text{app}}$, §6.3).

## 6.3 Evaluation Metrics and Protocols

**Figure accessibility.** All figures employ the Okabe–Ito palette with distinct markers and hatchings to ensure monochrome legibility.

**Ranking Precision:** Evaluated via Spearman $\rho$ against the simulated impact $I^*(v)$ from the primary oracle (§4.3). **Critical-Set Identification:** Measured via $F_1@K$ for top-$K$ components ($K = \text{round}(0.20 \cdot |V_{\text{app}}|)$). Because both the predicted and reference sets have exactly $K$ members, precision, recall and $F_1@K$ coincide identically: the column is top-$K$ set overlap, not an $F_1$ score, and it is reported under that reading everywhere it appears. Identification at operating points where precision and recall are free to differ — $F_1@\tau$ against the labels’ own critical set, threshold-free PR-AUC, and rank-weighted nDCG@10 — is reported separately in Supplementary Table S13, because those quantities answer a question $F_1@K$ structurally cannot. **Statistical Significance:** Paired Wilcoxon signed-rank tests [85] ($p < 0.05$) and bootstrap 95% CIs ($B = 2{,}000$) over folds [86, 87]. In the 12-fold LOSO design, the power floor is $p = 0.00049$.

**One confirmatory family, and everything else exploratory.** The confirmatory family is the pre-registered one and contains two contrasts: `HGT-QoS` against `Topo-QoS`, and `HGT` against `Topo-QoS`, Holm-corrected across those two alone. Neither reaches $\alpha = 0.05$ (§7.1). Every other contrast in this paper — the remaining full-population comparisons against `Topo-QoS`, the three orthogonal quantities of the $2\times2$ (Table 8), and the four simple effects — was formulated after the results existed and is reported as exploratory. Earlier versions of this manuscript corrected two overlapping families, which made the same contrast survive under one correction and not the other; the $2\times2$ quantities are still Holm-corrected within their own block, and that block is labelled post-hoc where it appears. For orientation, the largest exploratory effects are `Topo-QoS` over Topo ($+0.204$, 12/12, $p = 0.0005$) and unweighted typing, `HGT` over `GAT-N` ($+0.234$, 12/12, $p = 0.0005$), while the two contrasts that carry the QoS channel into an already-typed model do not separate: typing with the QoS channel present reaches $p = 0.1294$ (9/12 folds) and the QoS edge ablation under typing $p = 0.2036$ (10/12).

**Registered analysis plan.** The primary out-of-distribution contrast (`HGT-QoS` vs. `Topo-QoS` under LOSO, 5 fixed seeds, fold as unit of analysis) was registered in the replication package, with its protocol, statistic, unit of analysis and reporting commitment fixed, before the revised harness produced any result. We describe it as registered rather than pre-registered, and the distinction is not cosmetic: the plan is a file in our own repository with no third-party timestamp, and a prior eight-fold run of the same contrast — since withdrawn, because it reproduced from no commit — predates it and is what motivated writing the plan down. What the registration establishes is that the analysis was not selected after seeing the twelve-fold result; what it cannot establish is that the question was asked in ignorance of any earlier estimate. As reported in §7.1, the margin did not reach statistical significance, which is the outcome the plan committed to reporting.

### Evaluation Population and Protocols

Each predictor within an evaluation table is scored on an identical node population, resolved strictly from scenario topology and ground truth, specifically the **Application** set ($V_{\text{app}}$) unless otherwise noted. Pooling node types conflates distinct base rates and can trigger Simpson’s paradox (§7.3).

**In-Distribution Evaluation:** Stratified 60% train / 20% val / 20% test node splits over five seeds $\{42, 123, 456, 789, 2024\}$, redrawing partitions and initializations. **Inductive LOSO Cross-Validation:** Models are trained on eleven scenarios and test zero-shot on the held-out twelfth across all 12 folds under equal 3-layer depth and inner-split early stopping (§8.4). **Real-World Architectural Transfer:** Synthetic-trained models are evaluated zero-shot on five open-source systems without fine-tuning.
