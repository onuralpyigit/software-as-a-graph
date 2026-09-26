# 7. Results

All results are reported on the Application population ($V_{\text{app}}$) against the primary oracle $I^*(v)$, under the input–label independence guarantee (§4.4). Per-fold results, secondary strata and extended protocol notes are in the Supplementary Material and the experiment pages of the replication repository (§6.1). Figure 5 summarizes the three main findings.

![Figure 5](../latex/figures/Figure_5.png)

*Figure 5. Main results at a glance, Application population. (A) Mean Spearman ρ with 95% bootstrap intervals under LOSO (filled circles; CPU sweeps of Table 7) and zero-shot on the five system models (open diamonds). The hybrids lead on unseen synthetic architectures; the pure learned engines transfer best. (B) Per held-out fold, the gain of HGT-QoS and of Hybrid-HGT over Topo-QoS; the arrow shows what the closed-form prior changes. It removes the learned engine’s losses where the closed-form engine is strongest (Enterprise, Telecom RAN) and trims its largest gains where it is weakest. (C) Cell means of the capacity- and channel-matched 2 × 2 (Table 8): the QoS inputs raise both models by about 0.07, while the typed and untyped lines stay together.*

## 7.1 RQ1: SaG’s Engines Against Structural Baselines

#### Summary

*SaG’s QoS-weighted closed-form engine (`Topo-QoS`, $\rho = 0.553$) outperforms unweighted centrality ($\rho = 0.349$) on all twelve held-out architectures ($+0.204$, $p = 0.0005$). On their own, the learned engines are statistically on par with it (`HGT-QoS` $0.622$, `GAT-QoS` $0.635$). The hybrid engines, in which a learned engine corrects the closed-form score, significantly outperform it ($+0.103$ and $+0.130$, each on 11/12 folds, Holm $p \le 0.0068$).*

Each of the twelve folds holds out one scenario and trains on the remaining eleven, and every predictor is scored on the same Application node set (26 to 300 nodes, $K$ between 5 and 60). Table 7 reports SaG’s six engines and baselines from one family of CPU runs, in which the shared comparator rows are bit-identical.

**Table 7.** Main results. SaG’s engines against the structural baselines under LOSO (twelve synthetic architectures, five seeds, Application population, CPU runs) and zero-shot on the five open-source system models (protocol of Table 9). $\Delta\rho$ is paired by fold against the registered comparator `Topo-QoS`, with a bootstrap 95% CI ($B = 2{,}000$) and a two-sided Wilcoxon test. $p_{\text{Holm}}$ is given for the hybrids, within each one’s registered family (vs. `Topo-QoS` and vs. its own learned engine). The registered GPU sweep of `HGT-QoS`: Supplementary §S30. Per-fold values: Supplementary §S23.

|                |                                          |                                           |           |                             |                 |                            |            |
|:---------------|:----------------------------------------:|:-----------------------------------------:|:---------:|:---------------------------:|:---------------:|:--------------------------:|:----------:|
|                | **LOSO, twelve synthetic architectures** |                                           |           |                             |                 |   **Five system models**   |            |
| **Predictor**  |        **Mean $\rho$ [95% CI]**        | **$\Delta\rho$ vs `Topo-QoS` [95% CI]** |  **Won**  | **$p$ ($p_{\text{Holm}}$)** | **Overlap@$K$** |   **$\rho$ [95% CI]**    | **PR-AUC** |
| **Topo**       |          0.349 $[0.254, 0.452]$          |        $-0.204$ $[-0.286, -0.122]$        |   0/12    |           0.0005            |      0.366      |   0.511 $[0.346, 0.703]$   |   0.474    |
| **Topo-QoS**   |          0.553 $[0.443, 0.657]$          |                     —                     |     —     |              —              |      0.388      |   0.526 $[0.357, 0.699]$   |   0.474    |
| **HGT-QoS**    |          0.622 $[0.547, 0.690]$          |        $+0.069$ $[-0.046, +0.174]$        |   8/12    |            0.266            |      0.426      |   0.760 $[0.714, 0.819]$   |   0.713    |
| **GAT-QoS**    |          0.635 $[0.567, 0.696]$          |        $+0.082$ $[-0.046, +0.201]$        |   7/12    |            0.233            |      0.438      | **0.805** $[0.759, 0.868]$ | **0.790**  |
| **Hybrid-HGT** |          0.657 $[0.572, 0.733]$          |        $+0.103$ $[+0.055, +0.152]$        |   11/12   |       0.0034 (0.0068)       |      0.435      |   0.695 $[0.643, 0.730]$   |   0.602    |
| **Hybrid-GAT** |        **0.683** $[0.603, 0.753]$        |   $\mathbf{+0.130}$ $[+0.075, +0.190]$    | **11/12** |   **0.0015** (**0.0029**)   |    **0.450**    |   0.662 $[0.597, 0.727]$   |   0.600    |

**The QoS-aware projection is the largest single gain.** Re-weighting shortest paths by declared QoS contracts lifts closed-form ranking from $\rho = 0.349$ to $0.553$ on every held-out architecture. Learned engines without the QoS channel reach only about this level (`HGT` $0.548$, `GAT` $0.563$; §7.2), so the representation carries much of the signal.

**On their own, learned engines are on par with the closed-form engine.** `HGT-QoS` ($+0.069$, $p = 0.266$) and `GAT-QoS` ($+0.082$, $p = 0.233$) lead `Topo-QoS` numerically, with intervals spanning zero. The registered confirmatory contrast, run on the GPU sweep the plan specified, agrees ($+0.085$, $p = 0.151$, Holm $0.303$; Supplementary §S30).

**Both hybrids significantly outperform the closed-form engine.** Hybrid-HGT reaches $\rho = 0.657$ ($+0.103$, Holm $p = 0.0068$) and Hybrid-GAT reaches $\rho = 0.683$ ($+0.130$, CI $[+0.075, +0.190]$, Holm $p = 0.0029$), each on 11 of 12 folds. Each meets the decision rule registered before its run. Both remain significant under one Holm correction pooled over all eleven registered contrasts of the study ($p_{\text{omni}} = 0.034$ and $0.016$; §6.3). They are the only engines in this study that significantly beat closed-form ranking, and Hybrid-GAT also has the highest Overlap@$K$ ($0.450$).

**Why the hybrids work: the engines are complementary.** `HGT-QoS` loses to `Topo-QoS` on four folds, all where the closed-form engine is strongest: Real-Time Gaming, Enterprise ($0.426$ vs. $0.795$), AV System and Telecom RAN ($0.427$ vs. $0.576$). Its largest gains come where the closed-form engine is weakest: Healthcare, IoT Smart City, ATM and Microservices, $+0.210$ to $+0.362$ (Figure 5B). The prior removes the failure mode: on Enterprise, Hybrid-HGT and Hybrid-GAT reach $0.735$ and $0.768$, and Telecom RAN turns from a loss into a win for both. The cost is a smaller gain on the weakest folds.

**Anchoring trades transfer for in-distribution accuracy.** On the five independently authored system models, both hybrids stay well above every training-free score ($0.695$ and $0.662$ vs. $0.511$–$0.526$) but below the pure learned engines ($0.760$ and $0.805$; §7.3). By the rule registered in Amendment 6, Hybrid-GAT therefore does not replace Hybrid-HGT as the recommended hybrid ($0.662 < 0.695$).

**Label noise and inert components.** Re-running the oracle across five seeds gives a test–retest rank correlation of $0.811$–$1.000$ (median $0.982$), well above every engine’s mean $\rho$. Between $21\%$ and $52\%$ of each held-out population carries zero simulated impact. In the registered sweep, restricting evaluation to components with positive impact halves every predictor’s correlation, learned or not, and leaves the method ordering unchanged (Supplementary §S25).

### 7.1.1 How the Hybrids Are Built

Each hybrid gives a learned engine the rank-normalized `Topo-QoS` score as one extra input per Application and Library, and adds a learned correction to that score on the logit scale, $\hat{I}^*(v) = \sigma\big(z(v) + \alpha\,\operatorname{logit}(p(v))\big)$, with one learnable $\alpha$ (Figure 3a). Everything else matches the underlying engine. Hybrid-HGT is built on `HGT-QoS` (Amendment 5; $321$ extra parameters) and Hybrid-GAT on `GAT-QoS` (Amendment 6; $1{,}441$ extra parameters). Each was registered with its contrasts and decision rule before any run, with no setting tuned, and evaluated in its own CPU sweep with its comparators re-run in the same invocation.

## 7.2 RQ2: What Learned Engines Need

#### Summary

*With model capacity and edge-channel width matched, the QoS inputs are what improve learned ranking ($+0.073$ main effect, 10 of 12 folds), and relation-specific weights add nothing beyond it (typing main effect $-0.014$, interaction $+0.001$). The untyped `GAT-QoS` ($\rho = 0.635$) performs as well as the typed `HGT-QoS` ($0.622$).*

A first comparison against small untyped GATs ($28{,}168$ parameters against HGT’s $434{,}620$, reading at most a scalar edge weight) credited typing with a large gain ($+0.234$ without QoS). That gain is an effect of capacity and channel width (Supplementary §§S26 and S30), and the control registered in Amendment 2 removes both differences. `GAT` and `GAT-QoS` are untyped GATs at HGT’s parameter budget ($437{,}496$ and $429{,}992$), and `GAT-QoS` reads the same 16-D edge vector as `HGT-QoS`, including the relation one-hot. All four matched arms ran in one CPU sweep, and the decision rule was fixed before any control result existed (Table 8).

**Table 8.** The $2\times2$ with capacity and edge-channel width matched (Amendment 2): `GAT` ($\neg$T$\neg$Q, $437{,}496$ parameters), `HGT` (T$\neg$Q, $434{,}620$), `GAT-QoS` ($\neg$TQ, $429{,}992$), `HGT-QoS` (TQ, $434{,}620$). One CPU sweep, twelve LOSO folds, five seeds, Application population. Holm correction across the three orthogonal quantities; simple effects are descriptive. Cell means: $0.563$, $0.548$, $0.635$, $0.622$.

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

**QoS inputs are the working ingredient.** At matched capacity, adding the QoS inputs raises ranking by $+0.073$ with or without typing, on 10 of 12 folds each time, and significantly for the untyped pair ($+0.072$, CI $[+0.028, +0.109]$, $p = 0.016$). The QoS inputs also stabilize training: the median within-fold seed spread of the untyped pair falls from $0.083$ to $0.010$ (the small GATs and HGT show the same pattern; Supplementary §S30). The Q factor switches two inputs together: the 16-D edge vector and three node columns holding each component’s declared coupling weights ($w$, $w_{\text{in}}$, $w_{\text{out}}$). An exploratory follow-up that separates them locates the gain in the node columns: removing them from `GAT-QoS` costs $0.095$ (11 of 12 folds, Holm $p = 0.024$), whereas adding the edge channel alone to `GAT` gives $-0.023$. The seed stabilization follows the same columns (median spread $0.010$ with them, $0.136$ without).

**Relation-specific weights add nothing once capacity is matched.** `HGT` and `HGT-QoS` are within $0.015$ of their untyped counterparts `GAT` and `GAT-QoS` and win only 4–5 of 12 folds against them. Because `GAT-QoS` receives each edge’s relation type as a feature, the precise finding is that relation-typed *parameters* add nothing beyond relation-typed *inputs*. Message directionality (the `HGT-QoS-U` control) remains unmatched.

**How much QoS the target can reward.** $I^*(v)$ is a near-topological target: a topology-only relabeling recovers its ordering at mean $\rho = 0.965$, and QoS acts mainly at its top-$K$ boundary (§4.3). On this target the QoS inputs therefore act largely as a coupling-strength signal. Oracles that express deadline misses, durability replay or priority inversion would let the encodings contribute contract semantics as well.

## 7.3 RQ3: Zero-Shot Transfer to Models of Open-Source Systems

#### Summary

*Learned engines trained only on synthetic scenarios transfer to independently authored system models: `HGT-QoS` reaches $\rho = 0.760$ $[0.714, 0.819]$ and its untyped counterpart `GAT-QoS` reaches $0.805$ $[0.759, 0.868]$, against $0.511$–$0.526$ for every training-free score, and both nearly double top-$K$ critical-set overlap. On components that actually propagate failures, the differences remain unresolved at five systems.*

The five systems are hand-authored models of Autoware.universe (ROS 2), EdgeX Foundry and Home Assistant, plus meshes modelled after Online Boutique and Train-Ticket (§6.1). They were written independently of the scenario generator but are models rather than extractions, and they carry labels from the same oracles. The test is therefore transfer to independently authored topologies under simulated reachability. `HGT-QoS` and `GAT-QoS` were trained on all twelve synthetic scenarios and evaluated zero-shot at the same 3-layer, 300-epoch budget as every other learned result. No system model contributed gradients or checkpoint selection. Where a system declares no QoS manifest, standard middleware defaults (ROS 2 Best-Effort/Reliable, MQTT QoS 0/1) are applied uniformly across all predictors.

**Table 9.** Zero-shot transfer to hand-authored models of five open-source systems: Spearman $\rho$ against $I^*(v)$ on the Application population. Learned engines were trained on all twelve synthetic scenarios (3 layers, 300 epochs; five seeds, $\pm$ = spread over seeds). Training-free scores are deterministic and scored on identical labels and node sets. The last column restricts `HGT-QoS` to components with positive impact. All five models are encoded as publish–subscribe graphs; the grouping follows the paradigm of the *original* system. Identification metrics: Supplementary Table S14; bootstrap intervals and the active stratum for every predictor: Supplementary §S27.

| **System model**                                  | **$|V_{\text{app}}|$** | **$n_{>0}$** |  **Topo** | **Topo-QoS** |    **HGT-QoS**    |      **GAT-QoS**      | **HGT-QoS $\rho_{>0}$** |
|:--------------------------------------------------|-----------------------:|-------------:|----------:|-------------:|:-----------------:|:---------------------:|------------------------:|
| *Originals are publish–subscribe systems*         |                        |              |           |              |                   |                       |                         |
| **Autoware.universe (ROS 2)**                     |                     32 |           19 |     0.307 |        0.378 | 0.716 $\pm$ 0.081 | **0.758 $\pm$ 0.019** |        $+$0.517 |
| **EdgeX Foundry (Industrial IoT)**                |                     22 |           10 |     0.534 |        0.534 | 0.793 $\pm$ 0.037 | **0.815 $\pm$ 0.055** |        $+$0.183 |
| **Home Assistant (Smart Home)**                   |                     24 |           17 |     0.297 |        0.289 | 0.864 $\pm$ 0.063 | **0.925 $\pm$ 0.023** |        $+$0.702 |
| *Originals are RPC systems (modelled as pub-sub)* |                        |              |           |              |                   |                       |                         |
| **Online Boutique (pub-sub model)**               |                     22 |            8 | **0.891** |        0.888 | 0.710 $\pm$ 0.070 |   0.750 $\pm$ 0.119   |        -0.031 |
| **Train-Ticket Booking Mesh**                     |                     41 |           14 |     0.528 |        0.541 | 0.717 $\pm$ 0.096 | **0.777 $\pm$ 0.007** |        -0.192 |
| **Mean**                                          |                      — |            — |     0.511 |        0.526 |       0.760       |       **0.805**       |        $+$0.236 |

**Strong full-population transfer that does not depend on typing.** Both learned engines lead on 4 of 5 systems, with intervals that do not overlap those of the training-free scores (Table 7). The untyped `GAT-QoS` beats `HGT-QoS` on all five systems ($\rho = 0.805$ vs. $0.760$). It also scores higher on Overlap@$K$ ($0.519$ vs. $0.470$) and PR-AUC ($0.790$ vs. $0.713$). Consistent with RQ2, what transfers is learning over SaG’s QoS-annotated graph, not relation-specific parameters. Identification separates the engines most clearly: top-$K$ overlap averages $0.470$ against $0.248$ for the closed-form scores, and PR-AUC $0.713$ against $0.474$–$0.521$. On EdgeX, symmetric adapter-to-broker stars create betweenness ties that collapse closed-form triage entirely (Overlap@$K = 0.000$).

**Active components.** Restricted to components that propagate failures, `HGT-QoS` keeps a positive mean correlation ($\rho_{>0} = +0.236$) where every training-free score turns negative. Every interval spans zero at five systems, so this comparison is unresolved. $\rho_{>0}$ is positive on the three models of publish–subscribe systems and non-positive on the two modelled after RPC systems. Because both RPC-derived models are encoded as publish–subscribe graphs, this split cannot be attributed to call-tree semantics (§8.3). An earlier, non-blind 2-layer configuration leaves every conclusion unchanged (Supplementary §S29).

## 7.4 RQ4: Analysis Cost

#### Summary

*Neural inference is effectively free ($56\,\text{ms}$ for a 2,000-node architecture, $0.02\%$ of pipeline time). Cost is dominated by deterministic feature extraction, specifically the $O(|V|^2 + |V||E|)$ Connectivity Degradation Index, and it tracks the density of the derived dependency projection. Cold extraction takes $2$–$18\times$ (median $5.6\times$) as long as the in-process cascade simulation, so SaG’s advantage is avoiding staging infrastructure rather than saving CPU time.*

**Table 10.** Per-stage latency of the inference pipeline across graph sizes (CPU, median of 3 runs; 5 for the forward pass). The analysis stage is stable across repeats (p10–p90 within $1\%$ of the median), while the forward pass is dominated by interpreter and dispatch overhead; the 249-node row still carries first-call warm-up.

| **$|V|$** | **$|E|$** | **Analyze (s)** | **Graph $\to$ Tensor (s)** | **HGT Forward (ms)** | **Analyze : Forward** | **Forward p10–p90 (ms)** |
|:---------:|:---------:|:---------------:|:--------------------------:|:--------------------:|:---------------------:|:------------------------:|
|    249    |   1,127   |      1.74       |           0.010            |         26.5         |          66×          |        13.0–34.4         |
|    499    |   2,402   |      8.32       |           0.022            |         16.4         |         509×          |        15.6–16.4         |
|    999    |   6,422   |      44.54      |           0.056            |         21.1         |        2,108×         |        19.1–36.5         |
|   1,998   |  19,301   |     239.34      |           0.157            |         56.2         |      **4,259×**       |        43.8–57.8         |

At 2,000 components the HGT forward pass takes $56\,\text{ms}$ against $239\,\text{s}$ for structural analysis (Table 10). Because the analysis stage produces the node features the forward pass consumes, end-to-end evaluation of an unseen 2,000-component architecture takes about four minutes, of which the learned model is $0.02\%$. The $56\,\text{ms}$ is the marginal cost of re-scoring an already-analysed graph. The dominant term is the Connectivity Degradation Index, computed for every node in the main connected component. That is a correctness requirement: restricting CDI to articulation points leaves it identically zero in redundant multi-publisher topologies and makes $A(v)$ near-constant.

Across the corpus, the complete analysis gate (structural analysis plus 18 anti-pattern detectors) runs in $0.16$–$79.3\,\text{s}$ per scenario. That is $2.0$–$17.7\times$ (median $5.6\times$) the five-seed cascade labeling sweep measured in the same session, and the gate is more expensive on all twelve scenarios. The premium tracks the size of the derived projection rather than component count; Enterprise, with 300 applications on 120 topics, is the maximum (Supplementary §S28). On raw CPU time, direct simulation is therefore faster wherever its parameters are available. SaG additionally scores infrastructure components and dependency edges and avoids staging infrastructure. Training the four learned arms once took $7.7$ CPU-hours, amortized over every later evaluation.
