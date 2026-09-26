# 6. Experimental Setup

## 6.1 Corpus and Replication Package

The corpus comprises 2,812 components across seventeen architectures (Table 5). Twelve synthetic topologies form the LOSO folds. They span autonomous vehicles, financial trading, healthcare, industrial SCADA, smart-city IoT, telecom RAN, logistics, gaming, microservices, enterprise integration and air-traffic management. All twelve come from one generator family, as do their code metrics, so LOSO measures transfer across configurations of that generator. Each regenerates byte-identically from a committed configuration, and CI verifies this against a SHA-256 manifest.

**Table 5.** Evaluation corpus. The twelve synthetic topologies are the LOSO folds; the five open-source system models are excluded from all training and used only for zero-shot transfer (§7.3). Counts are read from the committed topology files and verified in CI; per-scenario composition: Supplementary §S13.

| **Dataset**                            | **$|V|$** | **$|V_{\text{app}}|$** | **Topics** | **Brokers** | **Hosts** | **Libs** |  **$|E|$** |
|:---------------------------------------|----------:|-----------------------:|-----------:|------------:|----------:|---------:|-----------:|
| Synthetic evaluation scenarios (11)    |     2,387 |                  1,295 |        588 |          60 |       194 |      250 |     10,657 |
| Synthetic case study — ATM (1)         |        74 |                     26 |         27 |           5 |         8 |        8 |        261 |
| **Synthetic subtotal (12 LOSO folds)** | **2,461** |              **1,321** |    **615** |      **65** |   **202** |  **258** | **10,918** |
| **Open-source system models (5)**      |   **351** |                **141** |    **120** |      **16** |    **32** |   **42** |    **700** |
| **Total**                              | **2,812** |              **1,462** |    **735** |      **81** |   **234** |  **300** | **11,618** |

**The five open-source systems are hand-authored models.** Autoware.universe (ROS 2), EdgeX Foundry, Home Assistant, and two meshes modelled after Online Boutique and Train-Ticket were each written by one author as typed multigraphs from public documentation (`saag/adapters/realworld_adapter.py`). They are not mechanical extractions. Brokers, QoS profiles, code metrics and host specifications are partly assumed. Two models depart materially from their originals: the Online Boutique model is a 22-application pub-sub mesh with four brokers, whereas the original is about eleven gRPC services with no broker, and the Train-Ticket model represents its service-discovery server as a broker. Neither contains a synchronous call edge. RQ3 therefore tests transfer to independently authored architecture models, not to deployed systems.

**Replication package.** Datasets, harnesses, checkpoints and result artifacts are archived on Zenodo (see Data Availability). The public repository documents each experiment: its protocol, hyperparameters, `make` target, artifacts and the supplementary section holding its extended results (<https://github.com/onuralpyigit/software-as-a-graph/tree/jss-submission-v4/docs/research/jss/experiments>).

## 6.2 Predictors

SaG’s closed-form engine (`Topo-QoS`), learned engines (`HGT-QoS`, `GAT-QoS`) and hybrid engines are compared against unweighted centrality (Topo) and untyped GNNs (Table 6). On a GNN, `-QoS` means the 16-D QoS edge vector (§4.1.1) together with three QoS node columns ($w$, $w_{\text{in}}$, $w_{\text{out}}$), and `Hybrid-X` is engine X corrected by the closed-form prior. `GAT` and `GAT-QoS` are untyped GATs matched to HGT in parameter budget, and `GAT-QoS` reads the same 16-D edge vector as `HGT-QoS`, so the four learned models form a $2\times2$ over typing (GAT vs. HGT) and the QoS channel. Smaller and projection-based variants that the study also ran are listed in Supplementary §S29. All learned predictors ingest the native typed multigraph under LOSO. In QoS-off arms every edge weight is 1 and the QoS node columns are zeroed, though four centralities still carry QoS (§3.4).

**Table 6.** Predictors reported in this paper. `-QoS`: QoS-weighted distances (Topo) or the 16-D QoS edge vector (GNNs); `Hybrid-X`: engine X corrected by the `Topo-QoS` prior. Every predictor run in the study, with the earlier names used by the registered plan and the artifacts: Supplementary §S29.

| **Predictor**                                                        | **Evaluation Substrate**                |  **Typing**   |     **Edge Features**      | **Parameters** | **Trained?** | **Empirical Role**           |
|:---------------------------------------------------------------------|:----------------------------------------|:-------------:|:--------------------------:|:--------------:|:------------:|:-----------------------------|
| *Training-free*                                                      |                                         |               |                            |                |              |                              |
| **Topo**                                                             | $G_{\text{analysis}}$ (Flow Projection) |      No       |            None            |       0        |      No      | Standard centrality baseline |
| **Topo-QoS**                                                         | $G_{\text{analysis}}$ (Flow Projection) |      No       |       Scalar $w(e)$        |       0        |      No      | SaG closed-form engine       |
| *Learned: typing $\times$ QoS channel at matched parameter budget*   |                                         |               |                            |                |              |                              |
| **GAT**                                                              | Native Multigraph                       |  Homogeneous  |            None            |    437,496     |     Yes      | Untyped, no QoS channel      |
| **GAT-QoS**                                                          | Native Multigraph                       |  Homogeneous  |      16-D QoS Vector       |    429,992     |     Yes      | Untyped SaG learned engine   |
| **HGT**                                                              | Native Multigraph                       | Heterogeneous | Relation 1-hot; $w(e){=}1$ |    434,620     |     Yes      | Typed, no QoS channel        |
| **HGT-QoS**                                                          | Native Multigraph                       | Heterogeneous |      16-D QoS Vector       |    434,620     |     Yes      | Typed SaG learned engine     |
| *Hybrid engines: a learned engine corrected by the `Topo-QoS` prior* |                                         |               |                            |                |              |                              |
| **Hybrid-HGT**                                                       | Native Multigraph                       | Heterogeneous |      16-D QoS Vector       |    434,941     |     Yes      | `HGT-QoS` + prior            |
| **Hybrid-GAT**                                                       | Native Multigraph                       |  Homogeneous  |      16-D QoS Vector       |    431,433     |     Yes      | `GAT-QoS` + prior            |

**Closed-form scores.** Topo and `Topo-QoS` are computed on the Application–Library `DEPENDS_ON` projection (Rules 1 and 5), because on the raw multigraph messages route through topics and brokers and Application betweenness vanishes: $$\text{Topo}(v) = 0.6 \cdot \text{BT}(v) + 0.4 \cdot \text{AP}(v),
\qquad
\text{Topo-QoS}(v) = 0.6 \cdot \text{BT}_{w}(v) + 0.4 \cdot \text{AP}(v),$$ where BT is normalized betweenness, $\text{BT}_{w}$ is betweenness over edge distances $d(e) = 1/(w(e) + 10^{-6})$, so strongly coupled edges attract shortest paths, and AP flags articulation points. In the evaluated implementation the articulation term reads zero for every node, so the reported scores rank exactly as betweenness and QoS-weighted betweenness. Restoring it lowers both (Supplementary §S22), so the evaluated form is the stronger reference and is kept as registered. The whole `Topo-QoS` gain over Topo therefore comes from QoS weighting of shortest paths. No predictor reads $G_{\text{structural}}$ (§4.4).

## 6.3 Metrics, Protocols and Statistics

**Population.** Every predictor in a table is scored on the same node population, the Application set $V_{\text{app}}$. Pooling entity types conflates distinct base rates. Against $I_{\text{comp}}$, RM correlates at $\rho = 0.597$ on Applications but only $0.217$ pooled over all types (Supplementary §S6).

**Metrics.** Ranking is measured by Spearman $\rho$ against $I^*(v)$. Critical-set identification is measured by Overlap@$K$, the fraction of the true top-$K$ recovered by the predicted top-$K$ with $K = \text{round}(0.20\,|V_{\text{app}}|)$, at which top-$K$ precision, recall and $F_1$ coincide. PR-AUC, $F_1@\tau$ and nDCG@10 are reported in Supplementary §S15.

**Protocols.** Under *LOSO*, models train on eleven scenarios and are tested zero-shot on the twelfth, over all 12 folds and five seeds, with equal 3-layer depth and inner-split early stopping. Under *zero-shot transfer*, models trained on all twelve scenarios are evaluated without fine-tuning on the five system models. In-distribution node-split results are in Supplementary §S17.

**Statistics.** We use paired Wilcoxon signed-rank tests [84] and bootstrap 95% CIs ($B = 2{,}000$) over folds [85, 86]; with 12 folds, the smallest attainable two-sided $p$ is $0.00049$. Folds share ten of eleven training scenarios, so the tests are anti-conservative [87], and we read all $p$-values as nominal.

**Registered analysis plan.** Before the twelve-fold harness produced any result, we registered the primary contrast, `HGT-QoS` vs. `Topo-QoS`, together with `HGT` vs. `Topo-QoS` and Holm correction across the two. We call it *registered* rather than pre-registered, because the plan is a file in our repository with no third-party timestamp. Three amendments registered further contrasts, each before its run and each Holm-corrected within its own family: the matched $2\times2$ (Amendment 2), Hybrid-HGT (Amendment 5) and Hybrid-GAT (Amendment 6). Every other contrast is exploratory. Because the sequence was adaptive (Amendment 6 followed Amendment 2’s result), we also pool all eleven registered contrasts under one Holm correction (`reproduce/omnibus_holm.py`). Both hybrid primaries remain significant (Hybrid-GAT $p_{\text{omni}} = 0.016$, Hybrid-HGT $p_{\text{omni}} = 0.034$), and no other registered contrast reaches $\alpha = 0.05$ ($p_{\text{omni}} \ge 0.38$). Supplementary §S24 lists all six amendments with dates and outcomes.
