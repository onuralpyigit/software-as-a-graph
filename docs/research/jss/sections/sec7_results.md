# 7. Results

All results are reported on the Application population ($V_{\text{app}}$) against the primary oracle $I^*(v)$, with additional convergent validation against the independent dynamic queue-flow oracle $I_{\text{dyn}}(v)$ and composite oracle $I_{\text{comp}}(v)$, under the input–label independence guarantee (§4.4). Per-fold results, secondary strata and extended protocol notes are in the Supplementary Material and public experiment pages (§6.1). Figure 5 summarizes the main findings.

![Figure 5](../latex/figures/Figure_5.png)

*Figure 5. Main results at a glance, Application population. (A) Mean Spearman ρ with 95% bootstrap intervals under LOSO (filled circles; Table 6) and zero-shot on five system models (open diamonds). Counting dependents on the derived dependency graph (InDeg, Reach) achieves top performance on both unseen synthetic architectures and system models; neural models reading the dependency graph reach parity with InDeg. (B) Per held-out fold, the gain of HGT-QoS and of Hybrid-HGT over Topo-QoS. (C) Cell means of the capacity- and channel-matched 2 × 2 (Table 8): QoS inputs raise both models by about 0.07, while typed and untyped configurations remain virtually indistinguishable.*

## 7.1 RQ1: SaG’s Engines Against Structural Baselines

<span id="sec:rq1-loso" label="sec:rq1-loso">[sec:rq1-loso]</span>

#### Summary

*On SaG’s derived dependency graph, counting a component’s dependents ranks simulated cascade impact as well as any engine in the study: `InDeg` reaches $\rho = 0.764$ ($\rho_{>0} = 0.516$ on active components) and beats closed-form centrality (`Topo-QoS`, $0.553$) on all twelve held-out architectures ($+0.211$, Holm $p = 0.002$). For Applications, `InDeg` is mathematically identical to counting distinct subscribers across published topics (publish–subscribe afferent coupling / AIS). A graph neural network that reads the dependency graph reaches the same level (`GAT-P-QoS`, $\rho = 0.748$), while an analytic first-order approximation of the simulator achieves $\rho = 0.808$ ($\rho_{>0} = 0.631$). Both hybrid engines significantly outperform closed-form centrality under their registered rules ($+0.103$ and $+0.130$, Holm $p \le 0.0068$).*

Each of the twelve folds holds out one scenario and trains on the remaining eleven, scored on the Application node set (26 to 300 nodes, $K$ between 5 and 60). Table 6 reports the training-free baselines, the raw-multigraph learned and hybrid engines, and the dependency-graph learners.

**Table 6.** Main results under LOSO across twelve synthetic architectures (Application population; learned engines: five seeds, CPU sweeps). $\Delta\rho$ is paired by fold against `Topo-QoS`, with a bootstrap 95% CI ($B = 2{,}000$) and a two-sided Wilcoxon signed-rank test. $\rho_{>0}$ denotes Spearman correlation restricted to active components ($I^* > 0$). $p_{\text{Holm}}$ is given within each registered family: hybrids (Amendments 5 and 6) and dependency counts (Amendment 7). $^\ddagger$Exploratory: Amendment 9 registered contrasts against `InDeg`, `Reach` and raw counterparts. The reference ceiling gives the post hoc closed-form analytic first-order $I^*$ approximation ($^*$outside registered confirmatory families). Per-fold values: Supplementary §§S23, S32 and S33.

| **Predictor**                                                       | **LOSO $\rho$ [95% CI]** | **Active $\rho_{>0}$** | **$\Delta\rho$ vs `Topo-QoS` [95% CI]** |  **Won**  | **$p$ ($p_{\text{Holm}}$)** | **Overlap@$K$** |
|:--------------------------------------------------------------------|:--------------------------:|:----------------------:|:-----------------------------------------:|:---------:|:---------------------------:|:---------------:|
| *Reference ceiling (first-order simulator approximation, post hoc)* |                            |                        |                                           |           |                             |                 |
| **Analytic $I^*$$^*$**                                              |   0.808 $[0.755, 0.853]$   |         0.631          |        $+0.255$ $[+0.170, +0.344]$        |   12/12   |         0.0005$^*$          |      0.536      |
| *Training-free, application layer*                                  |                            |                        |                                           |           |                             |                 |
| **Topo**                                                            |   0.349 $[0.254, 0.452]$   |         0.174          |        $-0.204$ $[-0.286, -0.122]$        |   0/12    |           0.0005            |      0.366      |
| *Training-free, dependency graph*                                   |                            |                        |                                           |           |                             |                 |
| **Topo-QoS**                                                        |   0.553 $[0.443, 0.657]$   |         0.280          |                     —                     |     —     |              —              |      0.388      |
| **Reach**                                                           |   0.732 $[0.674, 0.782]$   |         0.286          |        $+0.178$ $[+0.088, +0.268]$        |   11/12   |       0.0034 (0.0068)       |      0.344      |
| **InDeg** (pub-sub fan-in)                                          | **0.764** $[0.674, 0.840]$ |       **0.516**        |   $\mathbf{+0.211}$ $[+0.132, +0.299]$    | **12/12** |   **0.0005** (**0.0020**)   |    **0.504**    |
| *Learned and hybrid, on the raw multigraph*                         |                            |                        |                                           |           |                             |                 |
| **HGT-QoS**                                                         |   0.622 $[0.547, 0.690]$   |         0.312          |        $+0.069$ $[-0.046, +0.174]$        |   8/12    |            0.266            |      0.426      |
| **GAT-QoS**                                                         |   0.635 $[0.567, 0.696]$   |         0.338          |        $+0.082$ $[-0.046, +0.201]$        |   7/12    |            0.233            |      0.438      |
| **Hybrid-HGT**                                                      |   0.657 $[0.572, 0.733]$   |         0.345          |        $+0.103$ $[+0.055, +0.152]$        |   11/12   |       0.0034 (0.0068)       |      0.435      |
| **Hybrid-GAT**                                                      |   0.683 $[0.603, 0.753]$   |         0.362          |        $+0.130$ $[+0.075, +0.190]$        |   11/12   |       0.0015 (0.0029)       |      0.450      |
| *Learned, on the dependency graph (Amendment 9)*                    |                            |                        |                                           |           |                             |                 |
| **GAT-P-QoS**                                                       |   0.748 $[0.704, 0.789]$   |         0.440          |        $+0.195$ $[+0.100, +0.294]$        |   10/12   |      0.0068$^\ddagger$      |      0.454      |
| **Hybrid-GAT-P**                                                    |   0.758 $[0.710, 0.801]$   |         0.537          |        $+0.205$ $[+0.112, +0.298]$        |   11/12   |      0.0034$^\ddagger$      |      0.468      |
| **HGT-P-QoS**                                                       |   0.514 $[0.392, 0.636]$   |         0.237          |        $-0.039$ $[-0.155, +0.077]$        |   4/12    |      0.622$^\ddagger$       |      0.380      |

**Counting dependents formalizes afferent coupling.** `InDeg`, the number of direct dependents of $v$ on the derived graph, reaches $\rho = 0.764$ ($\rho_{>0} = 0.516$) and beats `Topo-QoS` on every fold; `Reach`, the number of transitive dependents, reaches $0.732$ (11/12). For an Application, `InDeg` is identically the number of distinct subscribers across all topics it publishes: publish–subscribe afferent coupling [57, 58, 59]. Deriving topic-mediated dependencies is what makes this metric computable from architecture manifests without running code. Furthermore, transitive reachability profits from the derived library rule: without Rule 5, `Reach` falls by $0.058$ (9/12 folds, Holm $p = 0.0068$; Amendment 10).

**Learners on the dependency graph match the count.** The same graph neural networks, trained on the dependency graph instead of the raw multigraph, reach $\rho = 0.748$ (`GAT-P-QoS`), level with `InDeg` ($-0.017$, Holm $p = 1.000$). Anchoring the learner on the `InDeg` score yields $\rho = 0.758$ (Hybrid-GAT-P), level with both. On the active stratum, `Hybrid-GAT-P` achieves the highest correlation ($\rho_{>0} = 0.537$, above `InDeg`’s $0.516$), though this margin is within statistical noise. In contrast, heterogeneous transformers (`HGT-P-QoS`) fail to train stably on the sparse projection in standard untuned configurations (mean $\rho = 0.514$, seed spread $0.208$), showing that parameter-heavy relational typing impairs optimization on sparse dependency graphs.

**Hybrids outperform closed-form centrality.** Hybrid-HGT reaches $\rho = 0.657$ ($+0.103$, Holm $p = 0.0068$) and Hybrid-GAT reaches $\rho = 0.683$ ($+0.130$, Holm $p = 0.0029$), each on 11 of 12 folds. Both survive omnibus Holm correction across all thirteen confirmatory contrasts ($p_{\text{omni}} = 0.041$ and $0.019$). However, on their own, the raw-multigraph learned engines are statistically on par with `Topo-QoS` (`HGT-QoS` $+0.069$, $p = 0.266$; `GAT-QoS` $+0.082$, $p = 0.233$).

**Where the closed-form gain comes from.** `Topo` reads betweenness from the application-layer graph, whereas `Topo-QoS` computes QoS-weighted betweenness on the Application–Library dependency graph. Three controls registered in Amendment 7 separate substrate from weighting: unweighted betweenness on the same graph scores $0.591$; constant topic weights score $0.595$; permuting QoS profiles scores $0.559$ ($-0.006$, $p = 0.73$). The gain is therefore driven by the Application–Library graph structure, not QoS contract content.

**The analytic reference ceiling.** To quantify how much of `InDeg`’s accuracy is driven by the simulator’s construction, Table 6 includes the post hoc closed-form analytic first-order approximation: $$\tag{5}
\hat{I}^*_1(v) = \sum_{t \in \text{pub}(v)} \frac{|\text{sub}(t)|}{|\text{pub}(t)|},$$ where $|\text{pub}(t)| \ge 1$ holds for all published topics $t \in \text{pub}(v)$ by construction ($v$ publishes to $t$), preventing any division by zero. This first-order expression achieves mean $\rho = 0.808$ $[0.755, 0.853]$ ($\rho_{>0} = 0.631$), demonstrating that direct topological subscriber loss forms the core mechanism of the reachability oracle. However, as a ceiling, it bounds the mean rather than every fold: on four folds (AV, Financial Trading, Healthcare, Industrial SCADA), `InDeg` slightly exceeds the analytic approximation (by $0.005\text{--}0.038$), while on the remaining eight folds the analytic expression leads.

**Table 7.** Agreement of architectural rankers across independent simulation paradigms over the twelve LOSO folds (Application population; exploratory evaluation). $I^*$ denotes the topological reachability cascade oracle (exhaustive over 1,321 applications); $I_{\text{dyn}}$ denotes the independent discrete-event queue-flow simulator (SimPy, evaluated on an $n = 30$ candidate application sample per fold); and $I_{\text{comp}}$ denotes the genuine multi-criteria failure simulator (exhaustive over 1,321 applications). $\rho_{>0}$ denotes correlation on the active stratum ($I > 0$).

|                    |                          |                 |                                     |                 |                                        |                 |
|:-------------------|:------------------------:|:---------------:|:-----------------------------------:|:---------------:|:--------------------------------------:|:---------------:|
|                    | **$I^*$ (Reachability)** |                 | **$I_{\text{dyn}}$ (Dynamic Flow)** |                 | **$I_{\text{comp}}$ (Multi-Criteria)** |                 |
| **Ranker**         |     **Mean $\rho$**      | **$\rho_{>0}$** |           **Mean $\rho$**           | **$\rho_{>0}$** |            **Mean $\rho$**             | **$\rho_{>0}$** |
| **Analytic $I^*$** |          0.808           |      0.631      |              **0.636**              |    **0.631**    |                 0.636                  |      0.636      |
| **InDeg**          |        **0.764**         |    **0.516**    |                0.610                |      0.589      |                 0.650                  |      0.650      |
| **Reach**          |          0.732           |      0.286      |                0.504                |      0.452      |                 0.302                  |      0.302      |
| **Topo-QoS**       |          0.553           |      0.280      |                0.393                |      0.343      |               **0.702**                |    **0.702**    |

**Validation against independent oracles.** Table 7 reports ranking agreement when the same training-free rankers are evaluated across independent simulation paradigms: the dynamic queue-flow oracle $I_{\text{dyn}}$ and the multi-criteria composite oracle $I_{\text{comp}}$. On $I_{\text{dyn}}$ ($n = 30$ candidates per fold), `InDeg` achieves $\rho = 0.610$ $[0.475, 0.727]$ ($\rho_{>0} = 0.589$), significantly outperforming closed-form centrality (`Topo-QoS`, $\rho = 0.393$; $\Delta = +0.217$, 11/12 fold wins, Wilcoxon $p = 0.0034$), while the analytic approximation reaches $\rho = 0.636$ ($\Delta = +0.243$, 11/12, $p = 0.0015$). Because $I_{\text{dyn}}$ measures continuous message delivery degradation under stochastic queueing load rather than graph reachability, this confirms that `InDeg`’s predictive utility is not an artifact of reachability cascade simulation. On genuine $I_{\text{comp}}$ (exhaustive multi-criteria failure simulation across all 1,321 applications), `Topo-QoS` achieves $\rho = 0.702$ $[0.649, 0.753]$, aligning with its QoS-weighted betweenness paths, while `InDeg` achieves $\rho = 0.650$ ($-0.052$, $p = 0.110$) and `Analytic I^*` achieves $\rho = 0.636$ ($-0.066$, $p = 0.064$). Because $I_{\text{comp}}$ evaluates multi-criteria failure dimensions including continuous network fragmentation and flow disruption across operational tiers, virtually every component suffers non-zero impact ($I_{\text{comp}} > 0$), rendering the active stratum $\rho_{>0}$ identical to the full-population $\rho$. Conversely, `Reach` falls to $0.504$ on $I_{\text{dyn}}$ and $0.302$ on $I_{\text{comp}}$ ($-0.400$, $p = 0.0005$), indicating that transitive reachability is specialized to topological cascade reachability.

## 7.2 RQ2: What Learned Engines Need

#### Summary

*Learned engines need two things: (1) they need to read the dependency graph, where directed messages reach scored applications ($+0.08$ to $+0.11$ gain, 11–12 of 12 folds); and (2) they need node-level degree features. When capacity is matched, relation typing adds nothing (typing main effect $-0.014$). In the QoS ablation, node-level coupling features ($w_{\text{in}}$) act as a weighted in-degree, confounding QoS contract content with structural dependent counting.*

**Table 8.** The $2\times2$ with capacity and edge-channel width matched (Amendment 2): `GAT` ($\neg$T$\neg$Q, $437{,}496$ parameters), `HGT` (T$\neg$Q, $434{,}620$), `GAT-QoS` ($\neg$TQ, $429{,}992$), `HGT-QoS` (TQ, $434{,}620$). One CPU sweep, twelve LOSO folds, five seeds, Application population. Holm correction across the three orthogonal quantities. Cell means: $0.563$, $0.548$, $0.635$, $0.622$.

| **Quantity**                                                     | **Contrast**              |  **$\Delta\rho$** |     **95% CI**     | **Won** | **$W$** | **$p$** | **$p_{\text{Holm}}$** |
|:-----------------------------------------------------------------|:--------------------------|------------------:|:------------------:|:-------:|:-------:|:-------:|:----------------------|
| *Three orthogonal quantities, Holm-corrected across these three* |                           |                   |                    |         |         |         |                       |
| **Typing (main effect)**                                         | averaged over Q           |          $-0.014$ | $[-0.052, +0.023]$ |  4/12   |  29.0   |  0.470  | 0.940                 |
| **QoS channel (main effect)**                                    | averaged over T           | $\mathbf{+0.073}$ | $[+0.013, +0.120]$ |  10/12  |  13.0   |  0.043  | 0.127                 |
| **Typing $\times$ QoS interaction**                              | difference of differences |          $+0.001$ | $[-0.050, +0.042]$ |  6/12   |  34.0   |  0.733  | 0.940                 |
| *Simple effects — descriptive, not separately corrected*         |                           |                   |                    |         |         |         |                       |
| **Typing, QoS absent**                                           | HGT vs. GAT               |          $-0.015$ | $[-0.064, +0.033]$ |  5/12   |  31.0   |  0.569  | —                     |
| **Typing, QoS present**                                          | HGT-QoS vs. GAT-QoS       |          $-0.013$ | $[-0.054, +0.026]$ |  4/12   |  27.0   |  0.380  | —                     |
| **QoS channel, typing absent**                                   | GAT-QoS vs. GAT           | $\mathbf{+0.072}$ | $[+0.028, +0.109]$ |  10/12  |   9.0   |  0.016  | —                     |
| **QoS channel, typing present**                                  | HGT-QoS vs. HGT           |          $+0.073$ | $[-0.002, +0.136]$ |  10/12  |  19.0   |  0.129  | —                     |

**QoS feature attribution and confounding.** Table 8 shows that the QoS main effect ($+0.073$) is not statistically significant after Holm correction ($p_{\text{Holm}} = 0.127$). In the node feature encoding, $w_{\text{in}}$ (`qos_weight_in`) is the sum of incoming edge weights on the analysis graph—a weighted in-degree. The QoS ablation zeroed this column, thereby removing the dependent count itself. As demonstrated by the registered closed-form controls in Amendment 7 (`results/qos_attribution_controls.json`), unweighted betweenness ($0.591$) and constant topic weights ($0.595$) match or exceed QoS-weighted betweenness ($0.553$), and permuting QoS profiles across topics yields no significant difference ($\Delta = -0.006$ $[-0.025, +0.015]$, $p = 0.733$), confirming that the observed gain is driven by dependency graph structure rather than QoS contract parameters.

**Relation typing under matched capacity.** `HGT-QoS` remains within $0.013$ of `GAT-QoS` (Table 8). In the single untuned configuration evaluated (width 100, default hyperparameters), `HGT-P-QoS` failed to train stably on the dependency projection (mean $\rho = 0.514$, seed spread $0.208$), whereas `GAT-P-QoS` converged smoothly ($\rho = 0.748$; Table 6). Relational typing remains untested in tuned hyperparameter configurations or architectures where message passing reaches scored nodes.

## 7.3 RQ3: Zero-Shot Transfer to Models of Open-Source Systems

#### Summary

*The dependency structure SaG derives transfers zero-shot: on hand-authored models inspired by five open-source systems, counting transitive dependents reaches $\rho = 0.938$ $[0.879, 0.991]$ ($\rho_{>0} = 0.871$ on active components) and `InDeg` reaches $0.863$ without training. Learned engines transfer far above closed-form centrality (`GAT-QoS` $\rho = 0.805$ vs. $0.511$–$0.526$).*

<span id="tab:9b" label="tab:9b">[tab:9b]</span>

**Table 9.** Zero-shot transfer to hand-authored models inspired by five open-source systems (Application population). Models are evaluated out-of-distribution without fine-tuning. $\rho_{>0}$ denotes rank correlation restricted to active components ($I^* > 0$). PR-AUC measures critical-set identification quality. $^\dagger$Scored by the Amendment 7 evaluation harness, where `Topo-QoS` scores $0.582$ (compared to $0.526$ on the learned engine pipeline; see Supplementary §S32).

| **Predictor**       | **Evaluation Substrate** | **Mean $\rho$ [95% CI]** | **Active $\rho_{>0}$** | **PR-AUC** |
|:--------------------|:-------------------------|:--------------------------:|:----------------------:|:----------:|
| **Topo**            | Application layer        |   0.511 $[0.346, 0.703]$   |        $-0.104$        |   0.474    |
| **Topo-QoS**        | Dependency graph         |   0.526 $[0.357, 0.699]$   |        $-0.088$        |   0.474    |
| **Reach**$^\dagger$ | Dependency graph         | **0.938** $[0.879, 0.991]$ |       **0.871**        | **0.933**  |
| **InDeg**$^\dagger$ | Dependency graph         |   0.863 $[0.734, 0.952]$   |         0.321          |   0.752    |
| **HGT-QoS**         | Raw multigraph           |   0.760 $[0.714, 0.819]$   |         0.236          |   0.713    |
| **GAT-QoS**         | Raw multigraph           |   0.805 $[0.759, 0.868]$   |         0.319          |   0.790    |
| **Hybrid-HGT**      | Raw multigraph           |   0.695 $[0.643, 0.730]$   |         0.210          |   0.602    |
| **Hybrid-GAT**      | Raw multigraph           |   0.662 $[0.597, 0.727]$   |         0.185          |   0.600    |
| **GAT-P-QoS**       | Dependency graph         |   0.806 $[0.785, 0.829]$   |         0.342          |   0.838    |

Table 9 reports transfer performance across all five systems. `Reach` achieves $\rho = 0.938$ and $\rho_{>0} = 0.871$, providing robust zero-shot ranking on unseen systems. Transitive reachability is especially strong on the three publish–subscribe models ($\rho = 0.836\text{--}0.997$, $\rho_{>0} = 0.674\text{--}0.971$) and on the two RPC-derived models ($\rho = 0.966\text{--}0.998$, $\rho_{>0} = 0.813\text{--}0.976$; full per-system breakdown in Supplementary Table S36). Because inert components with zero impact are separated by construction on these small models (22 to 41 applications), the active stratum $\rho_{>0}$ provides the critical differentiator.

## 7.4 RQ4: Analysis Cost

#### Summary

*Neural inference takes $56\,\text{ms}$ on a 2,000-node graph ($0.02\%$ of pipeline time), but cold feature extraction takes median $5.6\times$ longer than in-process direct simulation ($I^*$). Dependency counting executes in milliseconds and requires no feature extraction.*

**Table 10.** Per-stage latency of the inference pipeline across graph sizes (CPU, median of 3 runs; 5 for the forward pass). The analysis stage is stable across repeats (p10–p90 within $1\%$ of the median), while the forward pass is dominated by interpreter and dispatch overhead; the 249-node row carries first-call warm-up.

| **$|V|$** | **$|E|$** | **Analyze (s)** | **Graph $\to$ Tensor (s)** | **HGT Forward (ms)** | **Analyze : Forward** | **Forward p10–p90 (ms)** |
|:---------:|:---------:|:---------------:|:--------------------------:|:--------------------:|:---------------------:|:------------------------:|
|    249    |   1,127   |      1.74       |           0.010            |         26.5         |      66$\times$       |        13.0–34.4         |
|    499    |   2,402   |      8.32       |           0.022            |         16.4         |      509$\times$      |        15.6–16.4         |
|    999    |   6,422   |      44.54      |           0.056            |         21.1         |     2,108$\times$     |        19.1–36.5         |
|   1,998   |  19,301   |     239.34      |           0.157            |         56.2         |   **4,259$\times$**   |        43.8–57.8         |

Across the corpus, complete analysis takes $2.0$–$17.7\times$ (median $5.6\times$) the time of running the cascade simulation directly. Direct simulation is faster on raw CPU time wherever simulation parameters are available. SaG’s practical advantage lies in enabling sub-millisecond, training-free dependency counting for instantaneous CI/CD quality gates directly from manifests, while providing explainable ISO/IEC 25010 remediation profiles to guide architectural decisions.
