# 7. Results and Empirical Analysis

Empirical results for RQ1–RQ5 are presented across the twelve-fold inductive benchmark and five architecture models of open-source systems. The evaluated populations are strictly stratified on the Application service set ($V_{\text{app}}$) under the input–label independence guarantee (§4.4).

## 7.1 RQ1: Graph Learning vs. Structural Baselines

#### RQ1 Summary (Predictive Efficacy):

*SaG’s QoS-weighted closed-form engine (`Topo-QoS`, $\rho = 0.553$) outperforms unweighted centrality ($\rho = 0.349$) on all twelve held-out architectures ($+0.204$, $p = 0.0005$). The learned heterogeneous engine (`HGT-QoS`) attains the highest mean correlation in Table 8 ($\rho = 0.638$) but does not outperform the closed-form engine: on their own, the two SaG engines are statistically on par ($+0.085$, $p = 0.151$, Holm $0.303$); combined in the hybrid engines they significantly outperform the closed-form engine ($+0.103$ and $+0.130$, each on 11/12 folds, Holm $p \le 0.0068$; §7.5).*

**What the in-distribution results can and cannot support.** Per-scenario in-distribution figures are reported in Supplementary Table S15 rather than here, because no comparison can be drawn down their columns: `GAT`/`GAT-QoS` consume the Application–Library projection while `HGT`/`HGT-QoS` consume the native multigraph (§6.2.1), so a difference between the families confounds message passing with multi-entity visibility. Within a predictor the cells are still informative. In Healthcare, `Topo-QoS` achieves $\rho = 0.399$ but fails at critical triage (Overlap@$K = 0.000$); in the synthetic Microservices fold, `HGT` degrades ($\rho = 0.141$, Overlap@$K = 0.300$) where `HGT-QoS` does not ($\rho = 0.664$, Overlap@$K = 0.600$).

### 7.1.1 Out-of-Distribution (LOSO) Generalization

Inductive Leave-One-Scenario-Out cross-validation asks each model to predict cascading criticality on an entirely unseen topology, and the twelve folds are this paper’s primary anchor for generalization across architectural archetypes. Per-fold breakdowns are in the Supplementary Material (Table S12, §S10); the main text reports the cross-fold summary (Table 8) and the active-stratum contrast (Table 9):

**Table 8.** Inductive LOSO evaluation, Application population, twelve folds. Learned variants share training set, the native multigraph substrate (hence the `-N` infix; §6.2), depth, and selection rule (§6.3), differing in typing and edge channel; parameter budget, message-passing directionality and edge-channel width remain unmatched, and neither this table nor Table 10 controls for them (§8.3). Fold score = mean over five seeds; **Fold $\sigma$** = spread of the twelve fold means; **Seed $\sigma$** = median within-fold spread (zero by construction for deterministic baselines); CI = bootstrap over folds ($B = 2{,}000$). **$\Delta\rho$** is paired by fold against `Topo-QoS`, the registered comparator (§6.3), with its own bootstrap interval; an interval spanning zero means the contrast is not resolved at twelve folds, which is the case for every learned variant.

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

Each of the twelve folds holds out one scenario for zero-shot testing and trains on the remaining eleven, with all variants scored on the same Application node set (26 to 300 nodes, giving $K$ between 5 and 60) and paired Wilcoxon tests over folds. For Overlap@$K$, `HGT-QoS` achieved $0.424$ compared to `Topo-QoS`’s $0.388$ ($\Delta = +0.037$, prevailing in 7 of 12 folds, $W = 29.0$, $p = 0.470$). The untyped `GAT-N-QoS` attained $0.431$ ($\Delta = -0.006$, prevailing in 5 of 12 folds, 1 tie, $W = 27.5$, $p = 0.653$), indicating that critical-set identification does not statistically distinguish the typed model from either untyped learning or the QoS-weighted baseline.

**Label-noise ceiling.** No predictor can exceed the reproducibility of its own labels. Re-running the oracle across the five seeds gives a test–retest rank correlation of $0.811$–$1.000$ (median $0.982$), so `HGT-QoS`’s $\rho = 0.638$ recovers roughly $65\%$ of the attainable signal. Top-$K$ sets are far noisier — cross-seed Jaccard median $0.847$, falling to $0.370$ on Logistics Fleet — which is why the Overlap@$K$ margins are less stable than the ranking ones. The least reproducible fold (Microservices, $0.811$) is not one the typed model loses; on this corpus the low-ceiling folds and the lost folds are disjoint.

**Key Insights concerning RQ1:**

1.  **SaG’s QoS-aware projection is the largest single gain.** Re-weighting shortest paths by declared QoS contracts lifts closed-form ranking from $\rho = 0.349$ to $0.553$ and improves every one of the twelve held-out architectures ($+0.204$, $p = 0.0005$). Learned engines without the QoS channel reach only about this level (capacity-matched untyped GAT $0.563$, `HGT` $0.551$; §7.2), so the QoS-aware representation carries much of the signal.

2.  **The learned engine leads, and matches the closed-form engine.** `HGT-QoS` has the highest mean correlation in Table 8 ($\rho = 0.638$) and wins 9 of 12 folds against `Topo-QoS` ($+0.085$, CI $[-0.029, +0.194]$). The registered confirmatory contrast does not reach significance ($p = 0.151$, Holm $0.303$; un-augmented `HGT`: $-0.002$, $p = 0.470$), so we treat the two engines as comparable on this corpus. §7.5 evaluates a hybrid that combines them.

3.  **Learned engines identify critical sets best.** On Overlap@$K$, `HGT-QoS` ($0.424$), `HGT` ($0.427$) and `GAT-N-QoS` ($0.431$) all lead `Topo-QoS` ($0.388$) and Topo ($0.366$), although the fold-level differences are not significant ($\Delta = +0.037$, $p = 0.470$ for `HGT-QoS`).

4.  **Where the engines differ.** `HGT-QoS` wins most clearly where the closed-form engine is weakest (Microservices $+0.229$, ATM $+0.210$) and loses where it is strongest (Enterprise $-0.335$, Telecom RAN $-0.169$; §7.2.1). The two engines are therefore complementary, which motivates the hybrid of §7.5.

5.  **The explanation layer is an attribution instrument.** RM/$Q(v)$ is listed for reference ($\rho = 0.205$, interval above zero); its role is to explain flagged components, not to rank them (§5).

### 7.1.2 The Active Stratum: Ranking Quality Separated From Inertness Detection

Between $21\%$ (Microservices) and $52\%$ (Healthcare) of each held-out Application population carries exactly zero simulated impact, so a predictor can score well by separating components that can propagate a failure from those that cannot, without ordering the propagating ones correctly. These are different capabilities, so we re-score all twelve folds on the *active stratum* — the $n_{>0}$ components with strictly positive impact — using the same predictions, folds and seeds. Table 9 reports both.

**Table 9.** LOSO means over twelve folds on the full Application population ($\rho$) and restricted to components with strictly positive ground-truth impact ($\rho_{>0}$). Same predictions, folds, and seeds; only the evaluated subset differs. Retained is defined as the active-stratum ratio $\rho_{>0}/\rho$ (percentage of full-population correlation preserved).

| **Predictor**   | **$\rho$ (full)** | **$\rho_{>0}$ (active)** | **Retained** |
|:----------------|:-----------------:|:------------------------:|:------------:|
| **RM / $Q(v)$** |      $0.205$      |         $0.102$          |    $49\%$    |
| **Topo**        |      $0.349$      |         $0.181$          |    $52\%$    |
| **Topo-QoS**    |      $0.553$      |         $0.280$          |    $51\%$    |
| **GAT-N**       |      $0.317$      |         $0.159$          |    $50\%$    |
| **GAT-N-QoS**   |      $0.604$      |         $0.328$          |    $54\%$    |
| **HGT**         |      $0.551$      |         $0.299$          |    $54\%$    |
| **HGT-QoS**     | $\mathbf{0.638}$  |     $\mathbf{0.356}$     |    $56\%$    |

Two consequences follow.

1.  **The restriction costs every predictor roughly half its correlation, learned or not.** Retained fractions run from $49\%$ (RM) to $56\%$ (`HGT-QoS`) with no systematic separation between the training-free and learned families (`Topo-QoS` $51\%$, `GAT-N` $50\%$, HGT $54\%$). Roughly half of every predictor’s full-population correlation reflects separating inert components from active ones, a property of the label distribution rather than a discriminator between methods.

2.  **The method ordering is unchanged.** On the active stratum `HGT-QoS` still leads numerically ($\rho_{>0} = 0.356$), ahead of `GAT-N-QoS` ($0.328$) and `Topo-QoS` ($0.280$). No significance test is reported on this stratum, so it supports the full-population ordering descriptively and adds no new verdict.

## 7.2 RQ2: What the Learned Engine Needs

#### RQ2 Summary (Typing vs. QoS):

*With model capacity and edge-channel width matched, the QoS edge channel is what improves learned ranking ($+0.073$ main effect, 10 of 12 folds; $+0.072$ for untyped models, $p = 0.016$), and relation-specific weights add nothing beyond it (typing main effect $-0.014$, interaction $+0.001$). A capacity-matched untyped GAT with the QoS channel ($\rho = 0.635$) performs as well as `HGT-QoS` ($0.622$). The large typing gains of the unmatched comparison ($+0.234$) came from a $15\times$ capacity gap, not from typing.*

Table 10 gives the $2\times2$ over relation typing (T) and the QoS edge channel (Q) for the four reported learned arms. The untyped arms there have $28{,}168$ parameters against $434{,}620$ for HGT, and read a 1-dimensional edge channel against HGT-QoS’s 16 dimensions. The control arms registered in Amendment 2 remove both differences. `GAT-N-C` and `GAT-N-QoS16-C` are untyped GATs widened to HGT’s parameter budget ($437{,}496$ and $429{,}992$), and the latter reads the same 16-D edge channel as `HGT-QoS`, including the relation one-hot as an edge feature. All four matched arms ran in one CPU sweep (Table 11), and the decision rule was fixed before any control result existed.

**Table 10.** *Naive* $2 \times 2$ over relation typing (T) and the QoS edge channel (Q), unmatched in capacity ($15.4\times$) and edge-channel width; superseded for all typing claims by Table 11. Cells are the four reported learned arms: GAT-N ($\neg$T$\neg$Q), HGT (T$\neg$Q), GAT-N-QoS ($\neg$TQ), HGT-QoS (TQ). **Holm correction is applied across the three orthogonal quantities in the upper block only**; the four simple effects below are algebraically determined by those three and are reported descriptively (§7.3.1). Main effects average over the other factor’s levels; the typing effect reflects the joint transition to HGT and is confounded (§8.4). **Won** counts folds with $\Delta > 0$; the interaction is *negative* on all twelve. All quantities are post-hoc, and none were registered in advance. Fold overlap makes the reported $p$-values nominal (§8.3).

| **Quantity**                                                                     | **Contrast**              |  **$\Delta\rho$** |  **Won**  | **$W$** |  **$p$**   | **$p_{\text{Holm}}$** |
|:---------------------------------------------------------------------------------|:--------------------------|------------------:|:---------:|:-------:|:----------:|:----------------------|
| *The $2\times2$: three orthogonal quantities, Holm-corrected across these three* |                           |                   |           |         |            |                       |
| **Typing (main effect)**                                                         | averaged over Q           | $\mathbf{+0.134}$ | **12/12** |   0.0   | **0.0005** | **0.0015**            |
| **QoS channel (main effect)**                                                    | averaged over T           | $\mathbf{+0.187}$ |   11/12   |   2.0   | **0.0015** | **0.0015**            |
| **Typing $\times$ QoS interaction**                                              | difference of differences | $\mathbf{-0.199}$ |   0/12    |   0.0   | **0.0005** | **0.0015**            |
| *Simple effects — descriptive, not separately corrected*                         |                           |                   |           |         |            |                       |
| **Typing, QoS absent**                                                           | HGT vs. GAT-N             |          $+0.234$ |   12/12   |   0.0   |   0.0005   | —                     |
| **Typing, QoS present**                                                          | HGT-QoS vs. GAT-N-QoS     |          $+0.035$ |   9/12    |  19.0   |   0.1294   | —                     |
| **QoS channel, typing absent**                                                   | GAT-N-QoS vs. GAT-N       |          $+0.287$ |   11/12   |   1.0   |   0.0010   | —                     |
| **QoS channel, typing present**                                                  | HGT-QoS vs. HGT           |          $+0.087$ |   10/12   |  22.0   |   0.2036   | —                     |

**Table 11.** The $2\times2$ with capacity and edge-channel width matched (PREREGISTRATION Amendment 2): `GAT-N-C` ($\neg$T$\neg$Q, $437{,}496$ parameters), `HGT` (T$\neg$Q, $434{,}620$), `GAT-N-QoS16-C` ($\neg$TQ, $429{,}992$), `HGT-QoS` (TQ, $434{,}620$). One CPU sweep, twelve LOSO folds, five seeds, Application population. Holm correction across the three orthogonal quantities; simple effects are descriptive. Cell means: $0.563$, $0.548$, $0.635$, $0.622$.

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

**Key insights concerning RQ2:**

1.  **The QoS edge channel is the learned engine’s working ingredient.** At matched capacity, adding the 16-D QoS channel raises ranking by $+0.073$ with or without typing, on 10 of 12 folds each time, and significantly for the untyped pair ($+0.072$, CI $[+0.028, +0.109]$, $p = 0.016$). The channel also stabilizes training: the seed spread of the untyped model falls from $0.083$ to $0.010$.

2.  **Relation-specific weights add nothing once capacity is matched.** `HGT` and `HGT-QoS` are within $0.015$ of their capacity-matched untyped counterparts and win only 4–5 of 12 folds against them. Because `GAT-N-QoS16-C` receives each edge’s relation type as a feature, the result is specifically that relation-typed *parameters* add nothing beyond relation-typed *inputs*. Message directionality (the `HGT-QoS-U` control) remains unmatched.

3.  **The unmatched comparison overstated typing.** The $28{,}168$-parameter `GAT-N` of Table 10 reaches $\rho = 0.317$; at HGT’s capacity the same untyped design reaches $0.563$. Its $+0.234$ typing gain and $-0.199$ interaction are therefore effects of capacity and channel width. Table 10 is kept as the naive comparison it is, and it is not used for any claim about typing.

Neither the Fisher-$z$ transform nor robust seed aggregation (Supplementary §§S14 and S21), which left the unmatched interaction intact, could detect this confound: both hold the four unmatched arms fixed.

### 7.2.1 Where the Two SaG Engines Differ

`HGT-QoS` wins nine of twelve folds against `Topo-QoS`. Of its three losses, two are substantive — Enterprise ($\rho = 0.461$ vs. $0.795$) and Telecom RAN ($0.407$ vs. $0.576$) — and AV System is a near-tie ($-0.030$). All three losses fall where `Topo-QoS` is at its own strongest. Enterprise, AV and Telecom RAN are its 2nd, 3rd and 7th best folds of twelve ($0.795$, $0.753$, $0.576$), each at or above its mean of $0.553$. Conversely, on the two folds where `Topo-QoS` is weakest (Microservices $0.265$, ATM $0.311$) the learned engine wins by $+0.229$ and $+0.210$. The engines are thus complementary: the learned model gains most where closed-form structure is least informative, and gives up ground where it is most informative. §7.5 exploits this directly.

Two properties of Enterprise may explain its deficit: it is the largest graph ($520$ nodes), so three rounds of message passing cover less of its diameter, and its derived projection is by far the densest in the corpus ($26{,}276$ edges, Table 16). Neither graph size ($\rho = -0.434$ with the margin, $p = 0.159$), connection density, nor prediction dispersion predicts in advance which engine will win on an unseen architecture.

## 7.3 RQ3: Ablations and Sensitivity Analysis

#### RQ3 Summary (Ablations and Sensitivity):

*The reported orderings are robust. No configuration of the topic-weight or QoS sub-weight constants changes any comparison, only two of ten declared constants matter under Morris screening, and the behavioral queue-flow oracle agrees substantially with the cascade oracle ($\rho = 0.627$), which supports $I^*(v)$ as a valid ranking target.*

This section presents ablations relevant to the primary claims, including the QoS edge encoding, cross-oracle agreement, and per-type stratification that informs the interpretation of the results that follow. Of the ten constants, only the AHP shrinkage $\lambda$ and the Fault-Tolerance/Availability blend $r_{\text{FT}}$ exhibit appreciable influence on $\rho$ ($\mu^* = 0.134$ and $0.132$ under Morris screening, compared to $\le 0.025$ for the remaining eight). No configuration of the topic-weight or QoS sub-weight constants alters any comparison reported above. For the explanation layer, a uniform intra-dimension prior ranks better than the elicited AHP weights ($0.319$ vs. $0.200$) and is recommended when $Q(v)$ is used to rank (Supplementary §S1).

### 7.3.1 QoS Feature Ablation

To isolate the contribution of the continuous-categorical QoS edge features (§4.1.1) we evaluated **HGT**, an ablation whose edge features carry a constant unit weight and the relation one-hot. In the main sweep, `HGT-QoS` leads `HGT` by $+0.087$ (10/12 folds, $p = 0.204$). In the capacity-matched design the QoS channel adds $+0.073$ to typed and $+0.072$ to untyped models, each on 10 of 12 folds, with no interaction (Table 11). This is a consistent gain of $0.073$ regardless of architecture. The larger $+0.287$ of the unmatched untyped pair mostly reflects the small model’s instability, which the channel repairs.

The encodings also stabilise optimization, and there the asymmetry runs the other way: the median within-fold standard deviation over five seeds is $0.024$ for `GAT-N-QoS` against $0.298$ for `GAT-N`, and $0.052$ for `HGT-QoS` against $0.114$ for `HGT`. At matched capacity the same holds: $0.083$ for `GAT-N-C` against $0.010$ for `GAT-N-QoS16-C`. The QoS channel is the most reliable stabilizer of learned training.

#### How much QoS the target expresses.

These gains are earned against $I^*(v)$, whose ordering a topology-only relabeling recovers at mean $\rho = 0.965$ across the same twelve folds with no QoS term in the labeler (§4.3). On this target the QoS edge channel acts mainly as a relation-identity signal; QoS changes the label chiefly at its top-$K$ boundary (Jaccard $0.678$ between labels with and without QoS). Oracles that express QoS-driven impact in their ordering — deadline misses, durability replay, priority inversion — would let the encodings contribute contract semantics as well.

#### QoS Parameter Variance

Modal QoS shares range from 29% to 89% across the twelve scenarios, so every fold carries genuine variation in declared reliability, durability, and priority. As noted in §4.1.1, one schema dimension (`max_blocking_ms_log`) remains zero throughout the corpus as a reserved extension point, while the declared deadline populates the other two (`has_deadline`, `deadline_ns_log`) on $75\%$ of topics. Reported gains therefore stem from six active dimensions.

### 7.3.2 Convergent Validity Over Simulation Oracles

The three reliability-facing oracles measure distinct constructs, so we checked whether they agree before treating any as ground truth. Over the twelve inductive folds on the Application population, the behavioural queue-flow oracle and the topological cascade injector agree at mean Spearman $\rho = 0.627$ (top-$K$ Jaccard $0.370$ against $0.111$ expected by chance), against $I^*$’s own seed-to-seed test–retest of $0.811$–$1.000$. The agreement is therefore substantial but distinctly below label noise, which is the reading we want: an oracle reproducing another to within its own reproducibility would be re-measuring the same topology rather than corroborating it. Two boundaries qualify this — a large share of the agreement is the two oracles concurring on which components are *harmless*, and $I_{\text{dyn}}$ has a measured noise floor of its own that the headline does not correct for.

The Four Golden Signals captured during execution show the mechanism behind the separation: crashing a critical publisher degrades delivery to surviving consumers ($I_{\text{dyn}} > 0$) while *reducing* their queue waits through contention relief ($\rho = -0.499$ against tail-latency delta). Since $I^*$ is a deterministic breadth-first reachability computation and $I_{\text{dyn}}$ a stochastic queue simulation, their moderate agreement is convergent validity between two independent formulations rather than re-measurement, and it places the learned predictors as surrogates for topological cascade reach, not detectors of queue dynamics (§8.3).

### 7.3.3 Node-Type Stratification and Attention

One result governs how every other number in this paper is read. Measured against $I_{\text{comp}}(v)$ over the twelve scenarios of the corpus, stratified RM rank correlations are $\rho = 0.597$ (Application), $0.317$ (Broker), and $0.138$ (Execution Host), while pooling all types gives $\rho = 0.217$ — less than two fifths of the Application figure it is supposed to summarise. This is why every evaluation is reported on a single stratum, and why pooled figures should not be read as summaries of any one entity type. Pooled correlation sits above the Execution Host stratum ($0.138$), so this is aggregation bias rather than a strict Simpson reversal. A global sensitivity sweep of $I_{\text{comp}}$’s four severity weights ($N = 1{,}000$ Dirichlet draws; Supplementary §S1.2) shows Application correlation exceeding pooled on *all* draws ($\rho \in [0.441, 0.599]$ vs. $[0.096, 0.351]$), while the strict condition holds on only $13.9\%$ of the simplex. The stratification argument rests on the former, which is weight-invariant. A rule-based anti-pattern catalog on the same benchmark flags $93.4\%$ of scored components and so does not discriminate; critical-set identification is delegated to the continuous rankers of §§7.1–7.2.

Aggregated by relation type over the ATM case study, first-layer mean HGT attention orders `USES` into libraries ($0.227$) above publish–subscribe channels ($0.163$–$0.176$). However, the spread across all seven relation types is narrow ($0.15$–$0.23$) and driven substantially by destination in-degree artifacts. Typed attention remains active across relation types without establishing a statistically distinct ordering.

## 7.4 RQ4: Zero-Shot Transfer to Models of Open-Source Systems

#### RQ4 Summary (Out-of-Generator Transfer):

*Learned engines trained only on synthetic scenarios transfer to systems they have never seen. On five independently authored models of open-source systems, `HGT-QoS` reaches $\rho = 0.760$ $[0.714, 0.819]$ and the capacity-matched untyped GAT with the same QoS channel reaches $0.805$ $[0.759, 0.868]$, against $0.511$–$0.526$ for every training-free score. Both nearly double top-$K$ critical-set overlap. On components that actually propagate failures, both learned engines keep positive correlation where every training-free score turns negative; with five systems these active-stratum differences remain unresolved.*

We evaluated the framework on hand-authored architecture models of five open-source systems: Autoware.universe (ROS 2), EdgeX Foundry, Home Assistant, and meshes modelled after Online Boutique and Train-Ticket (§6.1). They were written independently of the scenario generator, which is what makes them a transfer test, but they are models rather than extractions, and all carry labels from the same simulation oracles used throughout. The test is therefore transfer to independently authored topologies under simulated reachability, not agreement with field incident telemetry.

Two evaluations are conducted. The first is an exploratory evaluation of the closed-form explanation layer $Q(v)$ against the composite oracle $I_{\text{comp}}(v)$ ($\rho = 0.514$–$0.800$); as noted in §4.3, $I_{\text{comp}}$’s four severity weights are unswept heuristics. The second, §7.4.1, evaluates learned predictors zero-shot against $I^*(v)$ (where RM achieves a lower mean $\rho = 0.516$). `Topo-QoS` is scored here on the flow projection, as everywhere else, because on the raw multigraph Application betweenness vanishes (§6.2.1). Standard middleware default contract profiles (e.g., ROS 2 Best-Effort/Reliable, MQTT QoS 0/1) were applied uniformly across edges lacking explicit manifests to ensure identical graph representations across baselines.

### 7.4.1 Zero-Shot Transfer of the Learned Model

To assess generalization outside the generator, `HGT-QoS` was trained on all twelve synthetic scenarios and evaluated zero-shot across five open-source systems versus $I^*(v)$ (five seeds; no open-source graph contributed training gradients or checkpoint selection). Tables 12–13 report this arm at the 3-layer, 300-epoch budget used for every other learned result in this paper. An earlier configuration used 2 layers and 150 epochs, chosen to limit over-smoothing across the smaller diameters of these meshes ($|V_{\text{app}}| \le 41$); that reasoning appeals to a property of the evaluation targets, so although no target label or gradient reached the model, the configuration was not blind to the test systems and we do not report it as the primary result. It is uniformly slightly stronger ($\rho = 0.792$ against $0.760$; $\rho_{>0} = +0.281$ against $+0.236$; Overlap@$K = 0.533$ against $0.470$) and leaves every qualitative conclusion unchanged, which is the sensitivity we draw from it.

**Table 12.** Zero-shot transfer to hand-authored models of five open-source systems, scored against $I^*(v)$ on the Application population under the protocol used throughout this paper (3 layers, 300 epochs; five seeds, $\pm$ = spread over seeds). No open-source graph contributed gradients or checkpoint selection. Training-free references are deterministic and are scored on identical labels, populations and node sets. Systems are grouped by the communication paradigm of the *original* system. All five models, including the two in the second group, are encoded as publish–subscribe graphs (§6.1). Where a system declares no explicit QoS manifest, default middleware contract profiles (ROS 2 Best-Effort/Reliable, MQTT QoS 0/1) are applied uniformly across baselines. Identification metrics for the same runs are in Supplementary Table S13.

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

**Table 13.** Means over the five system models on the full Application population ($\rho$) and restricted to components with strictly positive ground-truth impact ($\rho_{>0}$), with percentile bootstrap intervals over the five systems ($B = 2{,}000$). All four predictors are scored on identical labels, populations and node sets from one run, so the columns are commensurable. At $n = 5$ the intervals are descriptive and carry no significance claim. **The full-population intervals separate the learned model from all three baselines; every active-stratum interval spans zero**, which is why RQ4 is reported as established on the full population and unresolved on the active one.

| **Predictor**   |     **$\rho$ (full), 95% CI**     |   **$\rho_{>0}$ (active), 95% CI**   |
|:----------------|:---------------------------------:|:------------------------------------:|
| **RM / $Q(v)$** |     $0.516$ $[0.343, 0.680]$      |     $-0.055$ $[-0.292, +0.213]$      |
| **Topo**        |     $0.511$ $[0.346, 0.703]$      |     $-0.083$ $[-0.269, +0.108]$      |
| **Topo-QoS**    |     $0.526$ $[0.357, 0.699]$      |     $-0.092$ $[-0.268, +0.094]$      |
| **HGT-QoS**     | $\mathbf{0.760}$ $[0.714, 0.819]$ | $\mathbf{+0.236}$ $[-0.053, +0.525]$ |

**Key insights for out-of-generator transfer:**

1.  **Strong full-population transfer.** On all Applications, `HGT-QoS` reaches $\rho = 0.760$ $[0.714, 0.819]$ against $0.511$ (Topo), $0.526$ (`Topo-QoS`) and $0.516$ (RM), leading on 4 of 5 systems with a non-overlapping interval. Between $29\%$ and $66\%$ of Applications carry zero simulated impact, so part of this reflects correctly separating inert from active components — itself a useful triage property.

2.  **Transfer does not depend on relation typing.** Trained and scored under the same zero-shot protocol, the capacity-matched untyped GAT with the 16-D QoS channel (`GAT-N-QoS16-C`) outperforms `HGT-QoS` on all five systems ($\rho = 0.805$ $[0.759, 0.868]$ vs. $0.760$; mean paired difference $-0.045$ for `HGT-QoS`). It also scores higher on Overlap@$K$ ($0.519$ vs. $0.470$), PR-AUC ($0.790$ vs. $0.713$) and active-stratum correlation ($+0.319$ $[0.001, 0.638]$ vs. $+0.236$). Consistent with §7.2, what transfers is learning over SaG’s QoS-annotated graph, not relation-specific parameters.

3.  **Active components.** Restricted to components that propagate failures, the learned model keeps a positive mean correlation ($\rho_{>0} = +0.236$) where every training-free score turns negative ($-0.055$ to $-0.092$). Its interval $[-0.053, +0.525]$ spans zero, as do the baselines’, so this comparison is unresolved at five systems. $\rho_{>0}$ is positive on the three models of publish–subscribe systems (Home Assistant $+0.702$, Autoware $+0.517$, EdgeX $+0.183$) and non-positive on the two modelled after RPC systems (Online Boutique $-0.031$, Train-Ticket $-0.192$).

4.  **Identification is where the learned engine separates most clearly.** On top-$K$ overlap it averages $0.470$ $[0.410, 0.540]$ against $0.248$ $[0.09, 0.46]$ for the closed-form scores; on threshold-free measures the gap is wider ($F_1@\tau$ $0.473$ vs. $0.287$–$0.329$; PR-AUC $0.713$ vs. $0.474$–$0.521$; Supplementary Table S13). On EdgeX, symmetric adapter-to-broker star connections create betweenness ties that collapse closed-form triage entirely (Overlap@$K = 0.000$), while the learned engine still separates components.

5.  **Scope of the 3–2 split.** Both RPC-derived models are encoded with topics and brokers and labelled by the same forward-reachability oracle, so the split cannot be attributed to call-tree semantics; testing that hypothesis requires synchronous edges in the schema and a backward-propagating oracle (§8.4). On the Online Boutique model the closed-form scores lead only on the full population (`Topo-QoS` $0.888$) and are negative on its active components ($-0.072$).

## 7.5 SaG-Hybrid: Learning a Correction to the Closed-Form Engine

§7.2.1 showed that the learned and closed-form engines are complementary: the learned engine gains most where closed-form structure is least informative, and gives up ground where it is most informative. The hybrid engines combine them directly. Each takes a learned engine and gives it one extra input per Application and Library — the closed-form `Topo-QoS` score, rank-normalized within the graph — together with an output that adds a learned correction to that score on the logit scale, $\hat{I}^*(v) = \sigma\big(z(v) + \alpha\,\operatorname{logit}(p(v))\big)$, with a single learnable $\alpha$. Everything else (architecture, loss, epochs, early stopping, seeds) is identical to the underlying engine. Two hybrids were evaluated, each registered with its contrasts and decision rule before any run, with no setting tuned:

-   **SaG-Hybrid** is built on `HGT-QoS` (Amendment 5; $434{,}941$ parameters, $321$ more than `HGT-QoS`).

-   **SaG-Hybrid-GAT** is built on `GAT-N-QoS16-C`, the capacity-matched untyped GAT with the QoS channel (Amendment 6; $431{,}433$ parameters, $1{,}441$ more). It was added after the matched control of §7.2 showed that relation-typed weights add nothing at matched capacity.

Each hybrid was evaluated in its own CPU sweep, with its comparators re-run in the same invocation and never mixed with the GPU rows of Table 8. The `Topo-QoS`, `HGT-QoS` and `GAT-N-QoS16-C` rows are bit-identical across the CPU sweeps that contain them, so the rows of Table 14 are directly comparable. `HGT-QoS` reaches $\rho = 0.622$ on CPU against $0.638$ on GPU, within the run-to-run displacement of §8.3.

**Table 14.** Hybrid engines under the LOSO protocol of Table 8 (twelve folds, five seeds, Application population, CPU sweeps), and zero-shot on the five open-source system models under the protocol of Table 12. $\Delta\rho$ is paired by fold against `Topo-QoS`, with a bootstrap 95% CI and a two-sided Wilcoxon test. Holm correction is within each hybrid’s registered family: vs. `Topo-QoS` and vs. its own underlying engine.

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

**Key insights for the hybrid engines:**

1.  **Both hybrids significantly outperform the closed-form engine out of distribution.** SaG-Hybrid reaches $\rho = 0.657$ ($+0.103$, 11/12 folds, Holm $p = 0.0068$) and SaG-Hybrid-GAT $\rho = 0.683$ ($+0.130$, CI $[+0.075, +0.190]$, 11/12 folds, Holm $p = 0.0029$), against $0.553$ for `Topo-QoS`. Each meets the decision rule registered before its run, and both remain significant under one Holm correction pooled over all eleven registered contrasts of the study ($p_{\text{omni}} = 0.034$ and $0.016$; §6.3). All of these $p$-values are nominal, because the folds share training scenarios. The hybrids are the only engines in this study that significantly beat closed-form ranking, and they do so on architectures from the training generator (see the fourth insight for transfer).

2.  **SaG-Hybrid-GAT is the most accurate engine on unseen synthetic architectures.** It has the highest mean correlation, Overlap@$K$ ($0.450$), active-stratum correlation ($\rho_{>0} = 0.398$) and PR-AUC ($0.525$) of any engine under LOSO. It leads SaG-Hybrid on 11 of 12 folds ($+0.027$, CI $[+0.013, +0.041]$; a descriptive comparison, not a registered test), and it leads its own untyped engine by $+0.048$ (7/12, Holm $p = 0.30$).

3.  **The prior removes the learned engines’ failure mode.** The folds that held the pure learned engines back were those where closed-form structure is most informative. On Enterprise, `HGT-QoS` scores $0.426$ and `GAT-N-QoS16-C` $0.407$ against `Topo-QoS`’s $0.795$; with the prior they reach $0.735$ and $0.768$. Enterprise is each hybrid’s only loss to the closed-form engine ($-0.061$ and $-0.027$). Telecom RAN turns from a loss into a win for both.

4.  **Anchoring trades transfer for in-distribution accuracy.** On the folds where the closed-form engine is weakest (Healthcare, IoT Smart City, ATM, Microservices), both hybrids give up some of the learned engines’ gains. On the five independently authored system models, both stay well above every training-free score ($0.695$ and $0.662$ vs. $0.511$–$0.526$) but below the pure learned engines ($0.760$ and $0.805$). By the rule registered in Amendment 6, SaG-Hybrid-GAT therefore does not replace SaG-Hybrid as the recommended hybrid, since it transfers less well ($0.662 < 0.695$). For systems unlike the training corpus, the pure untyped engine with the QoS channel remains the best choice.

## 7.6 RQ5: Analysis Cost and Its Comparison Against Simulation

#### RQ5 Summary (Analysis Cost):

*Neural inference is effectively free ($56\,\text{ms}$ for a 2,000-node architecture, $0.02\%$ of pipeline time). Cost is dominated by deterministic feature extraction, specifically the $O(|V|^2 + |V||E|)$ Connectivity Degradation Index, and tracks the density of the derived dependency projection rather than component count. Cold extraction takes $0.16$–$79.3\,\text{s}$ per scenario, $2$–$18\times$ (median $5.6\times$) the in-process cascade simulation, so SaG’s advantage is avoiding staging infrastructure rather than CPU time.*

RQ5 quantifies computing overhead and sustainability during CI/CD evaluation, with per-stage latencies reported in Table 15:

**Table 15.** Per-stage latency of the inference pipeline across scaling graph sizes (CPU, median of 3 runs; 5 for the forward pass). The analysis stage is stable across repeats (p10–p90 within $1\%$ of the median everywhere) while the forward pass is not, which is why its column carries a spread: at $56\,\text{ms}$ the measurement is dominated by interpreter and dispatch overhead rather than by the graph. The 249-node row is slower than the 499-node row for the same reason: it is measured first, and its median still carries first-call allocation and dispatch warm-up.

| **$|V|$** | **$|E|$** | **Analyze (s)** | **Graph $\to$ Tensor (s)** | **HGT Forward (ms)** | **Analyze : Forward** | **Forward p10–p90 (ms)** |
|:---------:|:---------:|:---------------:|:--------------------------:|:--------------------:|:---------------------:|:------------------------:|
|    249    |   1,127   |      1.74       |           0.010            |         26.5         |          66×          |        13.0–34.4         |
|    499    |   2,402   |      8.32       |           0.022            |         16.4         |         509×          |        15.6–16.4         |
|    999    |   6,422   |      44.54      |           0.056            |         21.1         |        2,108×         |        19.1–36.5         |
|   1,998   |  19,301   |     239.34      |           0.157            |         56.2         |      **4,259×**       |        43.8–57.8         |

The neural stage is the cheapest by a wide margin and the deterministic one is not. At 2,000 components the HGT forward pass takes $56\,\text{ms}$ against $239\,\text{s}$ for structural analysis, a ratio of $4{,}259\times$ — but that is a ratio between pipeline stages, not a cost of evaluation, because indices 0–17 of every node feature vector are produced by the analysis stage the forward pass depends on. End-to-end evaluation of an unseen 2,000-component architecture takes about four minutes, of which the learned model is $0.02\%$; the $56\,\text{ms}$ is the marginal cost of re-scoring an already-analysed graph. Across the corpus the complete gate (structural analysis plus 18 anti-pattern detectors, whose share is $\le 0.19\,\text{s}$) executes in $0.16$–$79.3\,\text{s}$ (Table 16).

**One metric dominates cost.** Across the endpoints of Table 15 the measured cost is consistent with the stage’s $O(|V|^2 + |V||E|)$ bound: from 249 to 1,998 components wall-clock rises $138\times$ against a $137\times$ growth in $|V||E|$. That is an endpoint coincidence rather than a tracked curve: between the middle rows the component count doubles ($499 \to 999$) while wall-clock rises $5.4\times$, faster than the bound requires, and the series varies $|V|$ and $|E|$ together so it cannot separate them — Table 16 does that on the corpus. The dominant term is the Connectivity Degradation Index, computed for every node in the main connected component rather than for articulation points alone. That is a correctness requirement, not an oversight: gating CDI to articulation points leaves it identically zero wherever removal does not literally disconnect the graph, driving $A(v)$ to a near-constant in exactly the redundant multi-publisher topologies this system targets. The cost is the price of a non-degenerate Availability score.

**Table 16.** Static analysis gate against the cascade-reachability oracle, per scenario, from one paired measurement session. **Gate** is structural analysis plus the 18 anti-pattern detectors, whose own share is negligible ($\le 0.19\,\text{s}$ everywhere). **Oracle** is the full five-seed ground-truth labelling sweep over Application, Broker and Library nodes. $|E_{\text{proj}}|$ is the number of derived `DEPENDS_ON` edges in the Application–Library projection the analysis stage traverses, read from the committed cache. The ratio is $2.0$–$17.7\times$ with a median of $5.6\times$: the eighteen-fold figure quoted elsewhere in this section is the maximum, not the typical case. Ordering by $|E_{\text{proj}}|$ rather than by $|V|$ is what makes the column monotone — Enterprise carries 300 applications over only 120 topics, so its Rule-1 projection is near-complete, while IoT Smart City has more components and a sixth of the edges at a third of the cost.

| **Scenario**                     | **$|E_{\text{proj}}|$** | **Gate (s)** | **Oracle (s)** |       **Ratio** |
|:---------------------------------|------------------------:|-------------:|---------------:|----------------:|
| **Enterprise**                   |                  26,276 |        79.27 |          4.487 |    17.7$\times$ |
| **Enterprise Integration (ESB)** |                   4,641 |         3.17 |          0.245 |    12.9$\times$ |
| **AV System**                    |                   3,073 |         2.61 |          0.355 |     7.4$\times$ |
| **Financial Trading**            |                   2,657 |         1.71 |          0.239 |     7.1$\times$ |
| **Real-Time Gaming**             |                   2,318 |         2.43 |          0.379 |     6.4$\times$ |
| **Healthcare**                   |                   1,590 |         0.92 |          0.149 |     6.2$\times$ |
| **IoT Smart City**               |                   1,835 |         5.48 |          1.104 |     5.0$\times$ |
| **Telecom RAN**                  |                   2,116 |         3.70 |          1.065 |     3.5$\times$ |
| **Logistics Fleet**              |                   1,660 |         2.74 |          0.939 |     2.9$\times$ |
| **Microservices (synthetic)**    |                   1,524 |         2.29 |          0.828 |     2.8$\times$ |
| **Industrial SCADA**             |                     897 |         2.43 |          1.191 |     2.0$\times$ |
| **ATM System**                   |                     155 |         0.16 |          0.080 |     2.0$\times$ |
| **Median**                       |                       — |            — |              — | **5.6$\times$** |

### 7.6.1 Comparison With Direct Simulation

Timing the cascade reachability labeling sweep (five seeds, node types Application/Broker/Library, the full ground-truth run) over all twelve scenarios, on the same machine, at the same commit and in the same measurement session as the gate, gives $0.08$–$4.49\,\text{s}$ per scenario against $0.16$–$79.3\,\text{s}$ for the analysis gate. Both maxima occur in the 520-component Enterprise mesh, so the largest scenario compares $4.5\,\text{s}$ of simulation against $79.3\,\text{s}$ of static analysis: there the gate costs roughly eighteen times as much as the simulation. Pairing the sweeps scenario by scenario (Table 16) gives $2.0$–$17.7\times$, median $5.6\times$. The typical premium is therefore far below the maximum, but the gate is more expensive on *all twelve* scenarios. What predicts the premium is the derived projection’s size, not the component count — the ratio correlates with $|E_{\text{proj}}|$ at $\rho = 0.951$ against $0.792$ for cost against $|V|$. Enterprise is the outlier because its 300 applications share only 120 topics, so Rule 1 derives a near-complete graph of $26{,}276$ edges, while IoT Smart City has more components, an eighth of the edges and a third of the cost. These are wall-clock figures on one commodity CPU, read to one significant figure: across sessions the gate maximum ranges over $77$–$83\,\text{s}$ and the oracle maximum over $4.5$–$4.8\,\text{s}$, giving $16.7\times$ and $17.7\times$ on two independently paired sessions. Both halves of Table 16 share a commit and corpus digest.

Breadth-first cascade traversal is cheaper than all-pairs connectivity degradation ($O(|V|^2 + |V||E|)$), so on raw CPU time direct simulation is faster wherever its parameters are available. SaG’s analysis additionally scores infrastructure components and dependency edges that node-level message-flow simulation leaves unscored, and its metrics are amenable to caching across commits so that only the $k$-hop neighbourhood of a change is recomputed (not implemented here; Table 15 times full recomputation).

**Training cost.** The figures above are inference costs. Training is a separate, one-off cost per model version: the last sweep with recorded per-arm wall-clock (CPU, sequential, 12 folds $\times$ 5 seeds $= 60$ fits per arm) took $0.6\,\text{h}$ for `GAT-N`, $0.9\,\text{h}$ for `GAT-N-QoS`, $1.4\,\text{h}$ for `HGT` and $4.9\,\text{h}$ for `HGT-QoS`, $7.7$ CPU-hours in all. The GPU sweep behind Table 8 did not record per-fit durations.
