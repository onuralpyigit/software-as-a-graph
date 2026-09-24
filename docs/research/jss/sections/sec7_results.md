# 7. Results

All results are reported on the Application population ($V_{\text{app}}$) against the primary oracle $I^*(v)$, under the input–label independence guarantee (§4.4). Per-fold results, secondary strata and extended protocol notes are in the Supplementary Material and the experiment pages of the replication repository (§6.1).

## 7.1 RQ1: SaG’s Engines Against Structural Baselines

#### Summary

*SaG’s QoS-weighted closed-form engine (`Topo-QoS`, $\rho = 0.553$) outperforms unweighted centrality ($\rho = 0.349$) on all twelve held-out architectures ($+0.204$, $p = 0.0005$). The learned engine `HGT-QoS` has the highest mean of the single engines ($\rho = 0.638$) but is statistically on par with the closed-form engine ($+0.085$, $p = 0.151$, Holm $0.303$). The hybrid engines, in which a learned engine corrects the closed-form score, significantly outperform it ($+0.103$ and $+0.130$, each on 11/12 folds, Holm $p \le 0.0068$).*

### 7.1.1 Single Engines Under LOSO

Each of the twelve folds holds out one scenario and trains on the remaining eleven; all variants are scored on the same Application node set (26 to 300 nodes, $K$ between 5 and 60). Table 7 gives the cross-fold summary; per-fold values for the CPU re-runs of the SaG engines are in Supplementary §S23.

**Table 7.** Inductive LOSO evaluation, Application population, twelve folds. Learned variants share training set, the native multigraph substrate (`-N`), depth and selection rule (§6.2). Fold score = mean over five seeds; **Fold $\sigma$** = spread of the twelve fold means; **Seed $\sigma$** = median within-fold spread (zero for deterministic baselines); CI = bootstrap over folds ($B = 2{,}000$). **$\Delta\rho$** is paired by fold against the registered comparator `Topo-QoS`, with its own bootstrap interval; for every learned variant the interval spans zero.

| **Predictor / Reference**                                            | **Mean LOSO $\rho$** |    **95% CI**    |   **$\Delta\rho$ vs `Topo-QoS`**    | **Fold $\sigma$** | **Seed $\sigma$** | **Overlap@$K$** | **Requires Training** |
|:---------------------------------------------------------------------|:--------------------:|:----------------:|:-----------------------------------:|:-----------------:|:-----------------:|:---------------:|:---------------------:|
| *Training-free structural baselines*                                 |                      |                  |                                     |                   |                   |                 |                       |
| **Topo**                                                             |        0.349         | $[0.254, 0.452]$ | -0.204 $[-0.286, -0.122]$ |       0.173       |         —         |      0.366      |          No           |
| **Topo-QoS**                                                         |        0.553         | $[0.443, 0.657]$ |            — (reference)            |       0.192       |         —         |      0.388      |          No           |
| *Learned predictors (shared native substrate, matched training set)* |                      |                  |                                     |                   |                   |                 |                       |
| **GAT-N**                                                            |        0.317         | $[0.254, 0.381]$ | -0.236 $[-0.342, -0.125]$ |     **0.111**     |       0.298       |      0.328      |          Yes          |
| **GAT-N-QoS**                                                        |        0.604         | $[0.538, 0.665]$ | $+$0.051 $[-0.067, +0.169]$ |       0.112       |     **0.024**     |    **0.431**    |          Yes          |
| **HGT**                                                              |        0.551         | $[0.474, 0.617]$ | -0.002 $[-0.066, +0.072]$ |       0.124       |       0.114       |      0.427      |          Yes          |
| **HGT-QoS**                                                          |      **0.638**       | $[0.561, 0.710]$ | $+$0.085 $[-0.029, +0.194]$ |       0.133       |       0.052       |      0.424      |          Yes          |
| *Diagnostic reference — not a ranking model*                         |                      |                  |                                     |                   |                   |                 |                       |
| **RM / $Q(v)$**                                                      |        0.205         | $[0.092, 0.320]$ | -0.348 $[-0.432, -0.265]$ |       0.195       |         —         |      0.322      |          No           |

**The QoS-aware projection is the largest single gain.** Re-weighting shortest paths by declared QoS contracts lifts closed-form ranking from $\rho = 0.349$ to $0.553$ on every held-out architecture. Learned engines without the QoS channel reach only about this level (`HGT` $0.551$; capacity-matched untyped GAT $0.563$, §7.2), so the representation carries much of the signal.

**The learned engine leads numerically but does not significantly outperform the closed-form engine.** `HGT-QoS` wins 9 of 12 folds against `Topo-QoS` ($+0.085$, CI $[-0.029, +0.194]$), but the registered confirmatory contrast is not significant ($p = 0.151$, Holm $0.303$; `HGT`: $-0.002$, $p = 0.470$). On Overlap@$K$ the learned engines lead ($0.424$–$0.431$ against $0.388$), but the fold-level differences are not significant either ($\Delta = +0.037$, $p = 0.470$ for `HGT-QoS`). RM/$Q(v)$ is listed for reference only; its role is attribution (§5).

**Label noise and inert components.** Re-running the oracle across five seeds gives a test–retest rank correlation of $0.811$–$1.000$ (median $0.982$), so `HGT-QoS` recovers roughly $65\%$ of the attainable signal. Between $21\%$ and $52\%$ of each held-out population carries zero simulated impact. Restricting the evaluation to components with positive impact halves every predictor’s correlation, learned or not, and leaves the method ordering unchanged (Supplementary §S25).

**The engines are complementary.** `HGT-QoS` loses substantively on only two folds, Enterprise ($0.461$ vs. $0.795$) and Telecom RAN ($0.407$ vs. $0.576$). Both are folds where `Topo-QoS` is at or above its mean. Conversely, on the two folds where `Topo-QoS` is weakest (Microservices $0.265$, ATM $0.311$), the learned engine wins by $+0.229$ and $+0.210$. Enterprise is the largest graph ($520$ nodes) and has by far the densest derived projection ($26{,}276$ edges). However, neither graph size ($\rho = -0.434$ with the margin, $p = 0.159$) nor density predicts in advance which engine wins on an unseen architecture. This complementarity motivates the hybrids.

### 7.1.2 Hybrid Engines: Learning a Correction to the Closed-Form Score

Each hybrid takes a learned engine and gives it one extra input per Application and Library: the `Topo-QoS` score, rank-normalized within the graph. Its output adds a learned correction to that score on the logit scale, $\hat{I}^*(v) = \sigma\big(z(v) + \alpha\,\operatorname{logit}(p(v))\big)$, with a single learnable $\alpha$. Architecture, loss, epochs, early stopping and seeds are otherwise identical to the underlying engine. **SaG-Hybrid** is built on `HGT-QoS` (Amendment 5; $321$ extra parameters). **SaG-Hybrid-GAT** is built on `GAT-N-QoS16-C`, the capacity-matched untyped GAT with the QoS channel (Amendment 6; $1{,}441$ extra parameters). Each was registered with its contrasts and decision rule before any run, with no setting tuned. Each was evaluated in its own CPU sweep, with its comparators re-run in the same invocation. The comparator rows are bit-identical across these sweeps, and `HGT-QoS` reaches $0.622$ on CPU against $0.638$ on GPU.

**Table 8.** Hybrid engines under the LOSO protocol of Table 7 (twelve folds, five seeds, Application population, CPU sweeps), and zero-shot on the five open-source system models under the protocol of Table 10. $\Delta\rho$ is paired by fold against `Topo-QoS`, with a bootstrap 95% CI and a two-sided Wilcoxon test. Holm correction is within each hybrid’s registered family: vs. `Topo-QoS` and vs. its own underlying engine. Per-fold values: Supplementary §S23.

|                    |                                          |                                           |           |                             |                 |                            |            |
|:-------------------|:----------------------------------------:|:-----------------------------------------:|:---------:|:---------------------------:|:---------------:|:--------------------------:|:----------:|
|                    | **LOSO, twelve synthetic architectures** |                                           |           |                             |                 |   **Five system models**   |            |
| **Predictor**      |        **Mean $\rho$ [95% CI]**        | **$\Delta\rho$ vs `Topo-QoS` [95% CI]** |  **Won**  | **$p$ ($p_{\text{Holm}}$)** | **Overlap@$K$** |   **$\rho$ [95% CI]**    | **PR-AUC** |
| **Topo**           |          0.349 $[0.254, 0.452]$          |        $-0.204$ $[-0.286, -0.122]$        |   0/12    |           0.0005            |      0.366      |   0.511 $[0.346, 0.703]$   |   0.474    |
| **Topo-QoS**       |          0.553 $[0.443, 0.657]$          |                     —                     |     —     |              —              |      0.388      |   0.526 $[0.357, 0.699]$   |   0.474    |
| **HGT-QoS**        |          0.622 $[0.547, 0.690]$          |        $+0.069$ $[-0.046, +0.174]$        |   8/12    |            0.266            |      0.426      |   0.760 $[0.714, 0.819]$   |   0.713    |
| **GAT-N-QoS16-C**  |          0.635 $[0.567, 0.696]$          |        $+0.082$ $[-0.046, +0.201]$        |   7/12    |            0.233            |      0.438      | **0.805** $[0.759, 0.868]$ | **0.790**  |
| **SaG-Hybrid**     |          0.657 $[0.572, 0.733]$          |        $+0.103$ $[+0.055, +0.152]$        |   11/12   |       0.0034 (0.0068)       |      0.435      |   0.695 $[0.643, 0.730]$   |   0.602    |
| **SaG-Hybrid-GAT** |        **0.683** $[0.603, 0.753]$        |   $\mathbf{+0.130}$ $[+0.075, +0.190]$    | **11/12** |   **0.0015** (**0.0029**)   |    **0.450**    |   0.662 $[0.597, 0.727]$   |   0.600    |

**Both hybrids significantly outperform the closed-form engine.** SaG-Hybrid reaches $\rho = 0.657$ ($+0.103$, Holm $p = 0.0068$) and SaG-Hybrid-GAT reaches $\rho = 0.683$ ($+0.130$, CI $[+0.075, +0.190]$, Holm $p = 0.0029$), each on 11 of 12 folds. Each meets the decision rule registered before its run. Both remain significant under one Holm correction pooled over all eleven registered contrasts of the study ($p_{\text{omni}} = 0.034$ and $0.016$; §6.3). They are the only engines in this study that significantly beat closed-form ranking. SaG-Hybrid-GAT also has the highest Overlap@$K$ ($0.450$) under LOSO.

**The prior removes the learned engines’ failure mode.** On Enterprise, `HGT-QoS` and `GAT-N-QoS16-C` score $0.426$ and $0.407$ against `Topo-QoS`’s $0.795$; with the prior they reach $0.735$ and $0.768$. Telecom RAN turns from a loss into a win for both.

**Anchoring trades transfer for in-distribution accuracy.** On the five independently authored system models, both hybrids stay well above every training-free score ($0.695$ and $0.662$ vs. $0.511$–$0.526$) but below the pure learned engines ($0.760$ and $0.805$; §7.3). By the rule registered in Amendment 6, SaG-Hybrid-GAT therefore does not replace SaG-Hybrid as the recommended hybrid ($0.662 < 0.695$).

## 7.2 RQ2: What Learned Engines Need

#### Summary

*With model capacity and edge-channel width matched, the QoS edge channel is what improves learned ranking ($+0.073$ main effect, 10 of 12 folds), and relation-specific weights add nothing beyond it (typing main effect $-0.014$, interaction $+0.001$). A capacity-matched untyped GAT with the QoS channel ($\rho = 0.635$) performs as well as `HGT-QoS` ($0.622$).*

In Table 7, the untyped arms have $28{,}168$ parameters against $434{,}620$ for HGT, and they read a 1-dimensional edge channel against HGT-QoS’s 16 dimensions. A $2\times2$ over those four arms therefore credits typing with a large gain ($+0.234$ without QoS). That gain is an effect of capacity and channel width (Supplementary §S26). The control registered in Amendment 2 removes both differences. `GAT-N-C` and `GAT-N-QoS16-C` are untyped GATs widened to HGT’s parameter budget ($437{,}496$ and $429{,}992$). The latter reads the same 16-D edge channel as `HGT-QoS`, including the relation one-hot as an edge feature. All four matched arms ran in one CPU sweep, and the decision rule was fixed before any control result existed (Table 9).

**Table 9.** The $2\times2$ with capacity and edge-channel width matched (Amendment 2): `GAT-N-C` ($\neg$T$\neg$Q, $437{,}496$ parameters), `HGT` (T$\neg$Q, $434{,}620$), `GAT-N-QoS16-C` ($\neg$TQ, $429{,}992$), `HGT-QoS` (TQ, $434{,}620$). One CPU sweep, twelve LOSO folds, five seeds, Application population. Holm correction across the three orthogonal quantities; simple effects are descriptive. Cell means: $0.563$, $0.548$, $0.635$, $0.622$.

| **Quantity**                                                     | **Contrast**              |  **$\Delta\rho$** |     **95% CI**     | **Won** | **$W$** | **$p$** | **$p_{\text{Holm}}$** |
|:-----------------------------------------------------------------|:--------------------------|------------------:|:------------------:|:-------:|:-------:|:-------:|:----------------------|
| *Three orthogonal quantities, Holm-corrected across these three* |                           |                   |                    |         |         |         |                       |
| **Typing (main effect)**                                         | averaged over Q           |          $-0.014$ | $[-0.052, +0.023]$ |  4/12   |  29.0   |  0.470  | 0.940                 |
| **QoS channel (main effect)**                                    | averaged over T           | $\mathbf{+0.073}$ | $[+0.013, +0.120]$ |  10/12  |  13.0   |  0.043  | 0.127                 |
| **Typing $\times$ QoS interaction**                              | difference of differences |          $+0.001$ | $[-0.050, +0.042]$ |  6/12   |  34.0   |  0.733  | 0.940                 |
| *Simple effects — descriptive, not separately corrected*         |                           |                   |                    |         |         |         |                       |
| **Typing, QoS absent**                                           | HGT vs. GAT-N-C           |          $-0.015$ | $[-0.064, +0.033]$ |  5/12   |  31.0   |  0.569  | —                     |
| **Typing, QoS present**                                          | HGT-QoS vs. GAT-N-QoS16-C |          $-0.013$ | $[-0.054, +0.026]$ |  4/12   |  27.0   |  0.380  | —                     |
| **QoS channel, typing absent**                                   | GAT-N-QoS16-C vs. GAT-N-C | $\mathbf{+0.072}$ | $[+0.028, +0.109]$ |  10/12  |   9.0   |  0.016  | —                     |
| **QoS channel, typing present**                                  | HGT-QoS vs. HGT           |          $+0.073$ | $[-0.002, +0.136]$ |  10/12  |  19.0   |  0.129  | —                     |

**The QoS edge channel is the working ingredient.** At matched capacity, adding the 16-D QoS channel raises ranking by $+0.073$ with or without typing, on 10 of 12 folds each time, and significantly for the untyped pair ($+0.072$, CI $[+0.028, +0.109]$, $p = 0.016$). The channel also stabilizes training: the median within-fold seed spread falls from $0.083$ to $0.010$ for the matched untyped pair, from $0.298$ to $0.024$ for the small GAT, and from $0.114$ to $0.052$ for HGT.

**Relation-specific weights add nothing once capacity is matched.** `HGT` and `HGT-QoS` are within $0.015$ of their capacity-matched untyped counterparts and win only 4–5 of 12 folds against them. Because `GAT-N-QoS16-C` receives each edge’s relation type as a feature, the precise finding is that relation-typed *parameters* add nothing beyond relation-typed *inputs*. Message directionality (the `HGT-QoS-U` control) remains unmatched.

**How much QoS the target can reward.** $I^*(v)$ is a near-topological target: a topology-only relabeling recovers its ordering at mean $\rho = 0.965$, and QoS acts mainly at its top-$K$ boundary (§4.3). On this target the QoS edge channel therefore acts largely as a relation-identity and coupling-strength signal. Oracles that express deadline misses, durability replay or priority inversion would let the encodings contribute contract semantics as well.

## 7.3 RQ3: Zero-Shot Transfer to Models of Open-Source Systems

#### Summary

*Learned engines trained only on synthetic scenarios transfer to independently authored system models: `HGT-QoS` reaches $\rho = 0.760$ $[0.714, 0.819]$ and the capacity-matched untyped GAT with the same QoS channel reaches $0.805$ $[0.759, 0.868]$, against $0.511$–$0.526$ for every training-free score, and both nearly double top-$K$ critical-set overlap. On components that actually propagate failures, the differences remain unresolved at five systems.*

The five systems are hand-authored models of Autoware.universe (ROS 2), EdgeX Foundry and Home Assistant, plus meshes modelled after Online Boutique and Train-Ticket (§6.1). They were written independently of the scenario generator but are models rather than extractions, and they carry labels from the same oracles. The test is therefore transfer to independently authored topologies under simulated reachability. `HGT-QoS` was trained on all twelve synthetic scenarios and evaluated zero-shot at the same 3-layer, 300-epoch budget as every other learned result. No system model contributed gradients or checkpoint selection. Where a system declares no QoS manifest, standard middleware defaults (ROS 2 Best-Effort/Reliable, MQTT QoS 0/1) are applied uniformly across all predictors.

**Table 10.** Zero-shot transfer to hand-authored models of five open-source systems, scored against $I^*(v)$ on the Application population (3 layers, 300 epochs; five seeds, $\pm$ = spread over seeds). Training-free references are deterministic and scored on identical labels and node sets. All five models are encoded as publish–subscribe graphs; the grouping follows the paradigm of the *original* system. Identification metrics: Supplementary Table S14; bootstrap intervals and the active stratum for every predictor: Supplementary §S27.

| **System model**                                  | **$|V_{\text{app}}|$** | **$n_{>0}$** | **RM ($\rho$)** | **Topo ($\rho$)** | **Topo-QoS ($\rho$)** | **HGT-QoS ($\rho$)**  | **HGT-QoS ($\rho_{>0}$)** |
|:--------------------------------------------------|-----------------------:|-------------:|----------------:|------------------:|----------------------:|:---------------------:|--------------------------:|
| *Originals are publish–subscribe systems*         |                        |              |                 |                   |                       |                       |                           |
| **Autoware.universe (ROS 2)**                     |                     32 |           19 |           0.357 |             0.307 |                 0.378 | **0.716 $\pm$ 0.081** |          $+$0.517 |
| **EdgeX Foundry (Industrial IoT)**                |                     22 |           10 |           0.470 |             0.534 |                 0.534 | **0.793 $\pm$ 0.037** |          $+$0.183 |
| **Home Assistant (Smart Home)**                   |                     24 |           17 |           0.265 |             0.297 |                 0.289 | **0.864 $\pm$ 0.063** |          $+$0.702 |
| *Originals are RPC systems (modelled as pub-sub)* |                        |              |                 |                   |                       |                       |                           |
| **Online Boutique (pub-sub model)**               |                     22 |            8 |           0.777 |         **0.891** |                 0.888 |   0.710 $\pm$ 0.070   |          -0.031 |
| **Train-Ticket Booking Mesh**                     |                     41 |           14 |           0.713 |             0.528 |                 0.541 | **0.717 $\pm$ 0.096** |          -0.192 |
| **Mean**                                          |                      — |            — |           0.516 |             0.511 |                 0.526 |       **0.760**       |          $+$0.236 |

**Strong full-population transfer that does not depend on typing.** `HGT-QoS` leads on 4 of 5 systems, and its interval does not overlap those of the training-free scores. The capacity-matched untyped `GAT-N-QoS16-C` does better still on all five systems ($\rho = 0.805$). It also scores higher on Overlap@$K$ ($0.519$ vs. $0.470$) and PR-AUC ($0.790$ vs. $0.713$). Consistent with RQ2, what transfers is learning over SaG’s QoS-annotated graph, not relation-specific parameters. Identification separates the engines most clearly: top-$K$ overlap averages $0.470$ against $0.248$ for the closed-form scores, and PR-AUC $0.713$ against $0.474$–$0.521$. On EdgeX, symmetric adapter-to-broker stars create betweenness ties that collapse closed-form triage entirely (Overlap@$K = 0.000$).

**Active components.** Restricted to components that propagate failures, the learned engine keeps a positive mean correlation ($\rho_{>0} = +0.236$) where every training-free score turns negative. Every interval spans zero at five systems, so this comparison is unresolved. $\rho_{>0}$ is positive on the three models of publish–subscribe systems and non-positive on the two modelled after RPC systems. Because both RPC-derived models are encoded as publish–subscribe graphs, this split cannot be attributed to call-tree semantics (§8.3). An earlier, non-blind 2-layer configuration leaves every conclusion unchanged (Supplementary §S29).

## 7.4 RQ4: Analysis Cost

#### Summary

*Neural inference is effectively free ($56\,\text{ms}$ for a 2,000-node architecture, $0.02\%$ of pipeline time). Cost is dominated by deterministic feature extraction, specifically the $O(|V|^2 + |V||E|)$ Connectivity Degradation Index, and it tracks the density of the derived dependency projection. Cold extraction takes $2$–$18\times$ (median $5.6\times$) as long as the in-process cascade simulation, so SaG’s advantage is avoiding staging infrastructure rather than saving CPU time.*

**Table 11.** Per-stage latency of the inference pipeline across graph sizes (CPU, median of 3 runs; 5 for the forward pass). The analysis stage is stable across repeats (p10–p90 within $1\%$ of the median), while the forward pass is dominated by interpreter and dispatch overhead; the 249-node row still carries first-call warm-up.

| **$|V|$** | **$|E|$** | **Analyze (s)** | **Graph $\to$ Tensor (s)** | **HGT Forward (ms)** | **Analyze : Forward** | **Forward p10–p90 (ms)** |
|:---------:|:---------:|:---------------:|:--------------------------:|:--------------------:|:---------------------:|:------------------------:|
|    249    |   1,127   |      1.74       |           0.010            |         26.5         |          66×          |        13.0–34.4         |
|    499    |   2,402   |      8.32       |           0.022            |         16.4         |         509×          |        15.6–16.4         |
|    999    |   6,422   |      44.54      |           0.056            |         21.1         |        2,108×         |        19.1–36.5         |
|   1,998   |  19,301   |     239.34      |           0.157            |         56.2         |      **4,259×**       |        43.8–57.8         |

At 2,000 components the HGT forward pass takes $56\,\text{ms}$ against $239\,\text{s}$ for structural analysis (Table 11). Because the analysis stage produces the node features the forward pass consumes, end-to-end evaluation of an unseen 2,000-component architecture takes about four minutes, of which the learned model is $0.02\%$. The $56\,\text{ms}$ is the marginal cost of re-scoring an already-analysed graph. The dominant term is the Connectivity Degradation Index, computed for every node in the main connected component. That is a correctness requirement: restricting CDI to articulation points leaves it identically zero in redundant multi-publisher topologies and makes $A(v)$ near-constant.

Across the corpus, the complete analysis gate (structural analysis plus 18 anti-pattern detectors) runs in $0.16$–$79.3\,\text{s}$ per scenario. That is $2.0$–$17.7\times$ (median $5.6\times$) the five-seed cascade labeling sweep measured in the same session, and the gate is more expensive on all twelve scenarios. The premium tracks the size of the derived projection rather than component count; Enterprise, with 300 applications on 120 topics, is the maximum (Supplementary §S28). On raw CPU time, direct simulation is therefore faster wherever its parameters are available. SaG additionally scores infrastructure components and dependency edges and avoids staging infrastructure. Training the four learned arms once took $7.7$ CPU-hours, amortized over every later evaluation.
