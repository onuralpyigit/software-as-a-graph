# 7. Results and Empirical Analysis

Empirical results for RQ1–RQ5 are presented across the twelve-fold inductive benchmark and five authentic open-source distributed systems. The evaluated populations are strictly stratified on the Application service set ($V_{ ext{app}}$) under the input–label independence guarantee (§4.4).

## 7.1 RQ1: Graph Learning vs. Structural Baselines

Table 4 presents in-distribution held-out Spearman rank correlation ($\rho$) versus simulated cascade impact $I^*(v)$ across all twelve distributed architecture scenarios ($n = 12$).

**Table 4.** In-distribution held-out Spearman $\rho$ versus simulated cascade impact $I^*(v)$: mean over five seeds with bootstrap 95% CI in brackets; $n$ = held-out Application count. Substrates differ, and the comparison is confounded in-distribution: HGT/HGT-QoS consume the native typed multigraph. In contrast, GAT/GAT-QoS and the topological baselines consume the Application–Library `DEPENDS_ON` flow projection (§6.2). The typed–untyped contrast below therefore mixes typing with multi-entity visibility here; the substrate-matched comparison is the LOSO one in Table 5, where both architectures read the same graph. Each seed redraws the 60/20/20 split and the initialization. Supplementary § S10 reports the paired significance tests for these scenarios.

| **Scenario**          | **$n$** | **Topo** | **Topo-QoS** |     **GAT**      |   **GAT-QoS**    | **HGT** | **HGT-QoS** |
|:----------------------|--------:|:--------:|:------------:|:----------------:|:----------------:|:-------:|:-----------:|
| **ATM System**        |       5 |  0.538   |  **0.557**   | -0.393 | -0.080 |  0.492  |    0.348    |
| **AV System**         |      16 |  0.187   |    0.797     |    **0.816**     |      0.465       |  0.637  |    0.558    |
| **Enterprise**        |      60 |  0.443   |    0.793     |      0.779       |      0.481       |  0.861  |  **0.878**  |
| **Financial Trading** |      12 |  0.387   |    0.512     |      0.565       |      0.666       |  0.693  |  **0.730**  |
| **Healthcare**        |      10 |  0.291   |    0.399     |    **0.725**     |      0.575       |  0.575  |    0.607    |
| **Hub-and-Spoke**     |      14 |  0.179   |    0.429     |      0.363       | -0.156 |  0.421  |  **0.476**  |
| **Industrial SCADA**  |      28 |  0.601   |    0.710     |      0.656       |      0.478       |  0.787  |  **0.839**  |
| **IoT Smart City**    |      40 |  0.320   |    0.397     |      0.580       |      0.538       |  0.849  |  **0.850**  |
| **Logistics Fleet**   |      22 |  0.511   |    0.652     |      0.746       |      0.780       |  0.796  |  **0.815**  |
| **Microservices**     |      18 |  0.219   |    0.344     |      0.351       |      0.363       |  0.141  |  **0.664**  |
| **Real-Time Gaming**  |      15 |  0.360   |  **0.802**   |      0.464       |      0.471       |  0.651  |    0.641    |
| **Telecom RAN**       |      24 |  0.402   |    0.422     |    **0.608**     |      0.350       |  0.591  |    0.526    |
| **Mean**              |       — |  0.370   |    0.568     |      0.522       |      0.411       |  0.624  |  **0.661**  |

### Out-of-Distribution (LOSO) Generalization

In inductive Leave-One-Scenario-Out (LOSO) cross-validation, models are assessed based on their ability to predict cascading criticality across entirely unseen system topologies:

**Table 5.** Inductive LOSO evaluation, Application population, twelve folds. Learned variants share training set, substrate, depth, and selection rule (§6.3), differing in typing and edge channel, and also – as published – regarding parameter budget and message-passing directionality, which Table 7 controls for. Fold score = mean over five seeds; **Fold $\sigma$** = spread of the twelve fold means; **Seed $\sigma$** = median within-fold spread (zero by construction for deterministic baselines); CI = bootstrap over folds ($B = 2{,}000$).

| **Predictor / Reference**                                            | **Mean LOSO $\rho$** |    **95% CI**    | **Fold $\sigma$** | **Seed $\sigma$** | **Critical-Set $F_1@K$** | **Requires Training** |     |
|:---------------------------------------------------------------------|:--------------------:|:----------------:|:-----------------:|:-----------------:|:------------------------:|:---------------------:|:---:|
| *Training-free structural baselines*                                 |                      |                  |                   |                   |                          |                       |     |
| **Topo**                                                             |        0.349         | $[0.254, 0.452]$ |       0.173       |         —         |          0.366           |          No           |     |
| **Topo-QoS**                                                         |        0.553         | $[0.443, 0.657]$ |       0.192       |         —         |          0.388           |          No           |     |
| *Learned predictors (shared native substrate, matched training set)* |                      |                  |                   |                   |                          |                       |     |
| **GAT-N**                                                            |        0.317         | $[0.254, 0.381]$ |     **0.111**     |       0.298       |          0.328           |          Yes          |     |
| **GAT-N-QoS**                                                        |        0.604         | $[0.538, 0.665]$ |       0.112       |     **0.024**     |        **0.431**         |          Yes          |     |
| **HGT**                                                              |        0.551         | $[0.474, 0.617]$ |       0.124       |       0.114       |          0.427           |          Yes          |     |
| **HGT-QoS**                                                          |      **0.638**       | $[0.561, 0.710]$ |       0.133       |       0.052       |          0.424           |          Yes          |     |
| *Diagnostic reference — not a ranking model*                         |                      |                  |                   |                   |                          |                       |     |
| **RM / $Q(v)$**                                                      |        0.205         | $[0.092, 0.320]$ |       0.195       |         —         |          0.322           |          No           |     |

We evaluated twelve LOSO folds, encompassing eleven synthetic scenarios and the ATM case study, as detailed in Table 3 (§6.1). In each fold, we held out one scenario for zero-shot testing and trained the model exclusively on the remaining eleven. We assessed all model variants on the same Application node set per fold (§6.3) and ran paired Wilcoxon tests across the twelve folds, yielding a minimum attainable two-sided $p$-value of $0.00049$. The evaluated populations per fold ranged from 26 to 300 Application nodes, resulting in $K = \ text {round}(0.20,|V_{ ext{app}}|)$ values between 5 and 60. For $F_1@K$, `HGT-QoS` achieved $0.424$ compared to `Topo-QoS`’s $0.388$ ($
abla = +0.037$, prevailing in 7 of 12 folds, $W = 29.0$, $p = 0.470$). The untyped `GAT-N-QoS` attained $0.431$ ($
$ \ beta = -0.006$, prevailing in 5 of 12 folds, 1 tie, $W = 27.5$, $p = 0.653$), indicating that critical-set identification does not statistically distinguish the typed model from either untyped learning or the QoS-weighted baseline.

**Label-noise ceiling.** These correlations are bounded by the reproducibility of the target they are scored against. Re-running the ground-truth oracle across the five seeds gives a test–retest rank correlation between $0.811$ and $1.000$ across the twelve folds (median $0.982$; nine of twelve at or above $0.95$), with Microservices the least reproducible at $0.811$. `HGT-QoS`’s $\rho = 0.638$ therefore recovers roughly $65\%$ of the attainable signal against the median ceiling, and no predictor in Table 5 can exceed the reproducibility of its own labels. Top-$K$ critical sets are a much noisier construct: their cross-seed Jaccard has a median of $0.847$ and falls to $0.370$ (Logistics Fleet), $0.500$ (Industrial SCADA), and $0.500$ (Telecom RAN). That instability is the main reason the $F_1@K$ margins are less stable than the ranking margins, and it bounds how much weight any single critical-set comparison can carry. Microservices is the least reproducible fold, but it is not one the typed model loses (§7.2.1): `HGT-QoS` wins it by $+0.229$. On this corpus, the folds with the lowest label ceiling and the folds where the model is beaten are disjoint.

Figure 3 summarizes these results alongside critical-set identification and inter-oracle agreement.

**Key Insights concerning RQ1:**

1.  **Typed learning is the best configuration, but not demonstrably better than the QoS baseline.** `HGT-QoS` leads all predictors out-of-distribution ($\rho = 0.638$). Against training-free `Topo-QoS` it is $+0.085$ (9/12, $W = 20.0$, $p = 0.151$, CI $[-0.029, +0.194]$), an interval that includes zero; un-augmented HGT is indistinguishable from the baseline outright ($-0.002$, 3/12, $p = 0.470$). Neither pre-registered contrast reaches significance, and we report that as the answer rather than as a near miss.

2.  **A QoS-weighted structural score is a genuinely strong baseline — and untyped learning is worse than it.** `Topo-QoS` reaches $\rho = 0.553$ zero-shot, beating unweighted Topo on all twelve folds ($+0.204$, $p = 0.0005$). More pointedly, the untyped, unweighted learned model loses to it decisively (`GAT-N`, $-0.236$, 2/12, $p = 0.0024$): on this task, a homogeneous graph network trained on eleven architectures does not reach what a closed-form centrality score achieves with no training at all. Any claim that graph learning is required must be made against this baseline.

3.  **Critical-set identification does not favor the typed model.** On $F_1@K$, `HGT-QoS` scores $0.424$ against `GAT-N-QoS`’s $0.431$ and HGT’s $0.427$ — a three-way tie within noise — while all three numerically lead `Topo-QoS` ($0.388$), though without statistical significance throughout folds ($\Delta = +0.037$, $p = 0.470$). The margin over untyped learning claimed in earlier versions does not hold on the reconciled 12-fold corpus.

4.  **Power is not the limiting factor.** At $n = 12$, the design tolerates four lost folds and still reaches $\alpha = 0.05$, provided the losses are smallest in magnitude. `HGT-QoS`’s are not: it loses Enterprise ($-0.335$) and Telecom RAN ($-0.169$) to `Topo-QoS` by the two largest margins in the set, which is what holds $W$ at $20.0$. Enlarging the corpus will not resolve this; we must understand the inversions instead (§7.2.1).

5.  **The explanation layer is weakly predictive, not noise.** RM/$Q(v)$ reaches $\rho = 0.205$, losing to unweighted Topo on every fold ($-0.144$), so no ranking claim is made for it. Its interval $[0.092, 0.320]$ stays above zero, and it supplies interpretable diagnostics without training (§5). Table 5 lists it as a reference point, not a competitor.

![Figure 3](latex/figures/Figure_3.png)

*Figure 3. Results at a glance: Application population. (A) Out-of-distribution rank correlation per predictor across the twelve LOSO folds. (B) Critical-set identification at K = 20%, where the typed model does not separate from untyped learning. (C) Pairwise rank agreement between the three simulation oracles, against the chance baseline. (D) The typing × QoS interaction per fold — how much relation typing buys when the QoS edge channel is present, minus how much it buys when it is absent. Every one of the twelve folds is negative, as the substitution claim of §7.2 predicts and the evidence it rests on; the dashed line is the mean and the band is its bootstrap 95% CI. Panels A and B are read from the same artifact as Table 5, C from the convergent-validity artifact, and D from the significance artifact behind Table 7.*

### 7.1.2 The Active Stratum: Ranking Quality Separated From Inertness Detection

Every LOSO figure above shows correlation over the full held-out Application population, and between $21\%$ (Microservices) and $52\%$ (Healthcare) of that population carries exactly zero simulated impact, depending on the fold. A predictor can therefore score well by separating components that can propagate a failure from those that cannot, without ordering the propagating ones correctly. Because these are different capabilities with different functional value, we re-score all twelve folds on the *active stratum* — the $n_{>0}$ components with strictly positive ground-truth impact — using the same predictions, folds, and seeds. Table 6 reports both.

**Table 6.** LOSO means over twelve folds on the full Application population ($\rho$) and restricted to components with strictly positive ground-truth impact ($\rho_{>0}$). Same predictions, folds, and seeds; only the evaluated subset differs. Retained is defined as the active-stratum ratio $\rho_{>0}/\rho$ (percentage of full-population correlation preserved). Across all predictors, this restriction retains roughly half the correlation.

| **Predictor**   | **$\rho$ (full)** | **$\rho_{>0}$ (active)** | **Retained** |
|:----------------|:-----------------:|:------------------------:|:------------:|
| **RM / $Q(v)$** |      $0.205$      |         $0.102$          |    $49\%$    |
| **Topo**        |      $0.349$      |         $0.181$          |    $52\%$    |
| **Topo-QoS**    |      $0.553$      |         $0.280$          |    $51\%$    |
| **GAT-N**       |      $0.317$      |         $0.159$          |    $50\%$    |
| **GAT-N-QoS**   |      $0.604$      |         $0.328$          |    $54\%$    |
| **HGT**         |      $0.551$      |         $0.299$          |    $54\%$    |
| **HGT-QoS**     | $\mathbf{0.638}$  |     $\mathbf{0.356}$     |    $56\%$    |

Two consequences follow, the first of which revises a claim made in earlier versions of this paper.

1.  **The restriction costs every predictor roughly half its correlation, learned or not.** Retained fractions run from $49\%$ (RM) to $56\%$ (`HGT-QoS`) with no systematic separation between the training-free and learned families (`Topo-QoS` $51\%$, `GAT-N` $50\%$, HGT $54\%$). Roughly half of every predictor’s full-population correlation reflects separating inert components from active ones, a property of the label distribution rather than a discriminator between methods.

2.  **The method ordering is unchanged, and so are the verdicts.** On the active stratum `HGT-QoS` still leads ($\rho_{>0} = 0.356$), ahead of `Topo-QoS` ($0.280$) and `GAT-N-QoS` ($0.328$). Nothing in §7.1 or §7.2 turns on whether zero-impact components are included; we report both columns because they answer alternative operational questions.

## 7.2 RQ2: Value of Typed Heterogeneity

RQ2 investigates whether relation typing boosts prediction compared to homogeneous message passing. Addressing this question requires ablating typing against a matched comparator, and the outcome depends entirely on the comparator selected:

-   **Both factors have a real main effect.** Averaged over the other factor’s levels, relation typing is worth $\Delta\rho = +0.134$ (12 of 12 folds, $W = 0.0$, $p = 0.0005$, Holm $0.0015$) and the QoS edge channel $+0.187$ (11 of 12, $p = 0.0015$, Holm $0.0015$). We qualify this: because the HGT and GAT architectures differ in number of parameters ($15.4\times$) and message directionality alongside typing, this $+0.134$ margin reflects the joint architectural transition to HGT instead of isolated relational typing alone.

-   **But they interact, strongly and sub-additively.** The difference of differences — how much typing buys with the QoS channel present, minus how much it buys without it — is $-0.199$, negative on all twelve folds ($W = 0.0$, $p = 0.0005$, Holm $0.0015$, 95% CI $[-0.258, -0.147]$).

-   **The simple effects show the same pattern on both sides.** Typing is worth $+0.234$ when the QoS channel is absent (12/12, $p = 0.0005$) and $+0.035$ when it is present ($p = 0.1294$); the QoS channel is worth $+0.287$ without typing ($p = 0.0010$) and $+0.087$ with it ($p = 0.2036$).

-   **In-distribution fitting (Table 4) is not evidence either way.** The homogeneous pair reads the Application–Library projection while the typed pair reads the native multigraph, confounding message passing with multi-entity visibility.

**Typing and QoS encoding are substitutes, not complements.** Each mechanism, alone, lifts the plain baseline from $\rho = 0.317$ to roughly $0.55$–$0.60$; together they reach $0.638$, barely more than either achieves by itself. Both supply the model with the same underlying information: which relation a message crosses. Relation-specific parameters encode it in the weight matrices; the QoS edge vector encodes it in the edge features. Furthermore, $I^*(v)$’s ordering is recovered at mean $\rho = 0.965$ by a topology-only relabeling dropping the QoS term entirely (§4.3), confirming that neither channel tracks QoS-driven impact the oracle does not itself express.

**What the reference arm is, and why it matters for the effect sizes.** Both large simple effects are measured against GAT-N, a floor rather than a competitor ($\rho = 0.317$, losing to Topo-QoS by $-0.236$, $p = 0.0024$). Its score carries high seed instability ($\sigma = 0.298$ vs. mean $0.317$). The interaction is robust across folds, but read simple effects as recoveries from a deficit rather than absolute gains.

**Critical Confounders in the Typing Comparison.** While substrate, training set, depth, and selection rules are held constant, two architectural factors remain unmatched: (1) **Parameter Capacity:** HGT-QoS carries $434{,}620$ parameters on the primary training graph vs. $28{,}168$ for GAT-N-QoS ($15.4\times$); and (2) **Message Directionality:** HGT-QoS executes bidirectional message passing ($103{,}725$ parameters) enabling downstream nodes to aggregate upstream representations, whereas GAT-N-QoS propagates signals strictly forward. Because $I^*(v)$ measures downstream cascade starvation, bidirectional visibility confers a built-in topological advantage. The observed $+0.134$ typing main effect is thus consistent with relational inductive bias, but equally consistent with capacity or backward edge advantages (§8.4). On cost grounds, untyped QoS-weighted GNNs reach $\rho = 0.604$ at $28{,}168$ parameters vs. HGT-QoS’s $0.638$ at $434{,}620$ ($15.4\times$ capacity gap for a non-significant margin).

**Table 7.** The $2 \times 2$ over relation typing (T) and the QoS edge channel (Q), whose four cells are the four reported learned arms: GAT-N ($\neg$T$\neg$Q), HGT (T$\neg$Q), GAT-N-QoS ($\neg$TQ), HGT-QoS (TQ). **Holm correction is applied across the three orthogonal quantities in the upper block only.** The four simple effects below are algebraically linked to those three — given the cell means, any three determine the fourth — so correcting across them would treat one structural fact as four questions; they are reported descriptively because they carry the narrative, and the claim that they differ from one another rests on the interaction row above, not on the difference between their $p$-values. Main effects average over the other factor’s levels. **Won** counts folds with $\Delta > 0$; the interaction is *negative* on all twelve, which is the direction the substitution claim predicts. All quantities are post-hoc, and none were pre-registered.

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

### 7.2.1 Where Typed Learning Fails, and How It Can Be Detected

HGT-QoS loses to Topo-QoS on three folds, but only two of them are substantive: Enterprise ($\rho = 0.461$ vs. $0.795$) and Telecom RAN ($0.407$ vs. $0.576$). The third, AV System ($0.722$ vs. $0.753$), is a near-tie at $-0.030$ and carries little interpretive weight. The two substantive inversions are what keep the RQ1 comparison below significance. All three losses fall where `Topo-QoS` is at its own strongest: Enterprise, AV and Telecom RAN are its 2nd, 3rd and 7th best folds of twelve ($0.795$, $0.753$, $0.576$), each at or above its mean of $0.553$. This supports reading the inversions as the learned model discarding structural signal the baseline retains, rather than as folds that are intrinsically hard — on the two folds where `Topo-QoS` is weakest (Microservices $0.265$, ATM $0.311$) the typed model wins by $+0.229$ and $+0.210$.

#### Absence of a Label-Free Confidence Signature

We evaluated whether the standard deviation of predicted scores $\hat{\sigma}$ over held-out applications may indicate model trustworthiness at inference time without labels. It does not. Low dispersion does not mark the folds the model loses, and the lowest-dispersion folds include ones carrying large positive margins, so no threshold on $\hat{\sigma}$ separates the two. We report the direction of this result rather than its coefficients: the dispersion diagnostic was computed against a superseded prediction export, and none exists at the fidelity of the twelve-fold artifact behind Tables 5–7, so the per-fold $\hat{\sigma}$ values are withheld pending a re-run. Graph size is measurable on the current artifact and also fails to flag difficulty in advance (rank correlation with margin $-0.434$, $p = 0.159$), as does connection density.

Feature scale drift across scenarios (documented in ) remains the primary explanation for the Enterprise deficit, as Enterprise is the largest graph ($520$ nodes) and feature-scaling disparities are most acute in this case. Because the protocol strictly holds model hyperparameters constant across folds, there is currently no automated, label-free signal to predetermine whether an unseen architecture will favor the learned model or the training-free baseline (§8.4).

## 7.3 RQ3: Ablations and Sensitivity Analysis

This section presents ablations relevant to the primary claims, including the QoS edge encoding, cross-oracle agreement, and per-type stratification that informs the interpretation of the results that follow. Parameter-sensitivity sweeps over the explanation layer’s ten declared weight constants are provided in the supplementary material (Supplementary §§S1–S2) to illustrate robustness rather than to establish new findings. Of the ten constants, only the AHP shrinkage $\lambda$ and the Fault-Tolerance/Availability blend $r_\alpha$ exhibit appreciable influence on $\rho$ ($\mu^* = 0.134$ and $0.132$ under Morris screening, compared to $\le 0.025$ for the remaining eight). No configuration of the topic-weight or QoS sub-weight constants alters any comparison reported above. The elicited AHP weights are, notably, *anti*-predictive: rank correlation decreases monotonically from $0.319$ under a uniform setting to $0.200$ under raw AHP judgment. We retain these weights because RM serves as an attribution instrument rather than a ranking model, a trade-off discussed in §8.4.

### 7.3.1 QoS Feature Ablation

To isolate the specific empirical contribution of the continuous-categorical QoS edge features (§4.1.1), we evaluated **HGT**, an unaugmented ablation of HGT-QoS whose edge features contain only scalar coupling and relation one-hot encodings.

Under the inductive LOSO evaluation, the QoS edge encoding’s value depends on whether relation typing is already present. This is not a second finding but the same one seen from the other side: the interaction tested in §7.2 ($-0.199$, negative on all twelve folds, $p = 0.0005$) is symmetric in the two factors. Hence, a conditional effect of typing on the QoS channel is necessarily also a conditional effect of the QoS channel on typing. The figures below are the simple effects of Table 7 restated for the ablation reader; the significance of their *difference* rests on that interaction, not on the difference between their $p$-values.

*Without typing, the encoding is decisive.* GAT-N-QoS reaches $\rho = 0.604$ against GAT-N’s $0.317$: $\Delta\rho = +0.287$, won in 11 of 12 folds, $W = 1.0$, $p = 0.0010$, CI $[+0.207, +0.365]$. *With typing, it is not significant.* HGT-QoS reaches $0.638$ against HGT’s $0.551$: $+0.087$, 10 of 12 folds, $W = 22.0$, $p = 0.2036$. An earlier version of this manuscript reported the typed gain as $+0.054$ at $p = 0.0093$ from a superseded artifact and treated it as an independent contribution on top of typing; the effect does not reproduce at that significance, and the independence claim is withdrawn.

The encodings also improve optimization reproducibility, and here the asymmetry runs the other way. The median within-fold standard deviation across five seeds is $0.024$ for `GAT-N-QoS` against $0.298$ for `GAT-N` — more than a tenfold reduction — and $0.052$ for `HGT-QoS` against $0.114$ for HGT. An untyped model without the QoS channel is the least stable configuration by a wide margin, and either mechanism stabilizes it. This is consistent with the ranking result: both channels tell the model which relation an edge belongs to, and a model given neither must infer it from topology alone.

#### The target is nearly QoS-free, which bounds what either channel can credit.

The gains above are earned against $I^*(v)$, whose ordering a topology-only relabeling recovers at mean $\rho = 0.965$ across the same twelve folds, with no QoS term in the labeler at all (§4.3). The QoS edge channel therefore cannot help the model track QoS-driven impact that the oracle does not itself express, which is the strongest evidence we have for reading it as a relation-identity channel rather than a contract-semantics one. The label does move under QoS, but only at its top-$K$ boundary (Jaccard $0.678$ against the topology-only arm), not in its ranking. A corpus whose oracle expressed QoS-driven impact in its ordering — deadline misses, durability replay, priority inversion under load, none of which $I^*$ observes — would be a stronger test of the encodings than the one we report.

#### QoS Parameter Variance

Modal QoS shares range from 29% to 89% across the twelve scenarios (Supplementary §S5), making sure that every fold carries genuine variation in declared reliability, durability, and priority. As noted in §4.1.1, one schema dimension (`max_blocking_ms_log`) remains zero throughout the corpus as a reserved extension point, while the declared deadline populates the other two (`has_deadline`, `deadline_ns_log`) on $75\%$ of topics; reported gains therefore stem from six active dimensions.

### 7.3.2 Convergent Validity Over Simulation Oracles

The three reliability-facing oracles measure distinct constructs, so we checked whether they agree before treating any as ground truth. Over the twelve inductive folds on the Application population, the behavioural queue-flow oracle and the topological cascade injector agree at mean Spearman $\rho = 0.620$ (top-$K$ Jaccard $0.365$ against $0.111$ expected by chance), against $I^*$’s own seed-to-seed test–retest of $0.811$–$1.000$. The agreement is therefore substantial but distinctly below label noise, which is the reading we want: an oracle reproducing another to within its own reproducibility would be re-measuring the same topology rather than corroborating it. Two boundaries qualify this — a large share of the agreement is the two oracles concurring on which components are *harmless*, and $I_{\text{dyn}}$ has a measured noise floor of its own that the headline does not correct for. Supplementary §S9 reports the full pairwise table, the zero-excluded correlations, and the multi-seed floors for both non-primary oracles.

### 7.3.3 Node-Type Stratification and Attention

One result governs how every other number in this paper is read. Measured against $I_{\text{comp}}(v)$ over the eight scenarios of the detection benchmark, stratified RM rank correlations are $\rho = 0.566$ (Application), $0.119$ (Broker), and $0.244$ (Node), while pooling all types collapses the correlation to $\rho = 0.098$ — below every per-type value it aggregates, which is Simpson’s paradox in its textbook form. This is why we report every evaluation on a single stratum, and why pooled critical-set figures should be read as inflated wherever they appear. Supplementary §S6 reports the rule-based anti-pattern catalog evaluated on the same benchmark; it summarizes that the catalog flags $93.8\%$ of scored components and therefore does not discriminate, so critical-set identification is delegated to the continuous rankers of §§7.1–7.2.

Aggregated by relation type over the ATM case study, first-layer mean HGT attention orders `USES` into libraries ($0.227$) above publish–subscribe channels ($0.163$–$0.176$), but the spread across all eight relation types is narrow ($0.15$–$0.23$) and driven substantially by destination in-degree artifacts. Supplementary §S8 gives the layer-wise distribution and heatmap, and confirms that typed attention remains active across relation types without establishing a statistically distinct ordering.

## 7.4 RQ4: Real-World Distributed Architecture Validation

We evaluated the framework on five open-source distributed systems transcribed from public repositories: Online Boutique, Train-Ticket, Home Assistant, Autoware Universe (ROS 2), and EdgeX Foundry. All carry labels from the same simulation oracles used throughout, testing topological transfer rather than agreement with field failures.

Two distinct real-world evaluations are conducted. First, the closed-form explanation layer $Q(v)$ is evaluated against $I_{ ext{comp}}(v)$ in Supplementary §S7, achieving strong correlation across all five systems ($
ho = 0.514$–$0.800$) and outperforming degree centrality. We score both training-free references in this context. Second, §7.4.1 evaluates the learned predictor zero-shot against $I^*(v)$ on the same five systems. `Topo-QoS` was absent from earlier versions of Table 8 because the open-source adapters carried no QoS contracts; this was a defect in appraising its betweenness rather than a property of the data, and §8.4 records the correction.

### 7.4.1 Zero-Shot Transfer of the Learned Model

To assess generalization to architectures outside the generator, `HGT-QoS` was trained on all twelve synthetic scenarios and evaluated zero-shot across the five open-source systems versus $I^*(v)$ (five seeds). To reduce cross-scenario feature-scale drift, this transfer evaluation applies within-graph rank normalization, two message-passing layers, and 150 epochs. No real-world system contributed training gradients as well as was used to select checkpoints. Table 8 presents the resulting transfer performance.

**Table 8.** Zero-shot transfer to five open-source systems, scored against $I^*(v)$ on the Application population. `HGT-QoS` trains on all twelve synthetic scenarios ($\pm$ = spread over five seeds); RM, Topo, and `Topo-QoS` are training-free and are scored on identical labels and nodes. **$\rho_{>0}$ restricts the correlation to the $n_{>0}$ components that actually propagate a failure and is the column to read for ranking quality**; full-population $\rho$ conflates that with separating active from inert. Both training-free references are scored on the same labels and nodes (§8.4).

| **Real-World Architecture**        | **$|V_{\text{app}}|$** | **RM $\rho$** | **Topo $\rho$**  | **Topo-QoS $\rho$** |         **HGT-QoS $\rho$**          | **HGT-QoS $\rho_{>0}$** | **$n_{>0}$** | **$F_1@K$** |
|:-----------------------------------|-----------------------:|:-------------:|:----------------:|:-------------------:|:-----------------------------------:|:-----------------------:|:------------:|:-----------:|
| **Cloud Microservices Mesh**       |                     22 |    $0.777$    | $\mathbf{0.891}$ |       $0.888$       |     $0.649$ $\pm$0.131      |    $\mathbf{-0.029}$    |      18      |   $0.400$   |
| **Train-Ticket Booking Mesh**      |                     41 |    $0.713$    |     $0.528$      |       $0.541$       | $\mathbf{0.776}$ $\pm$0.004 |    $\mathbf{-0.213}$    |      22      |   $0.450$   |
| **Autoware.universe (ROS 2)**      |                     32 |    $0.357$    |     $0.307$      |       $0.378$       | $\mathbf{0.734}$ $\pm$0.054 |        $+0.559$         |      28      |   $0.633$   |
| **EdgeX Foundry (Industrial IoT)** |                     22 |    $0.470$    |     $0.534$      |       $0.534$       | $\mathbf{0.804}$ $\pm$0.040 |        $+0.304$         |      19      |   $0.500$   |
| **Home Assistant (Smart Home)**    |                     24 |    $0.265$    |     $0.297$      |       $0.289$       | $\mathbf{0.872}$ $\pm$0.035 |        $+0.704$         |      23      |   $0.600$   |
| **Mean**                           |                      — |    $0.516$    |     $0.511$      |       $0.526$       |          $\mathbf{0.767}$           |        $+0.265$         |      —       |   $0.517$   |

**Key Insights for Real-World Transfer:**

1.  **The full-population figure is not a ranking result.** On all Applications, `HGT-QoS` reaches $\rho = 0.767$ vs. $0.511$ (Topo), $0.526$ (`Topo-QoS`), and $0.516$ (RM), leading on 4/5 systems. However, between $4\%$ (Home Assistant) and $46\%$ (Train-Ticket) of Applications carry zero simulated impact; full correlation heavily rewards separating inert from active components rather than ranking active ones.

2.  **Restricted to components that actually propagate failures, transfer is not established.** On the active stratum the mean drops to $+0.265$, and both microservice call trees invert: Cloud Microservices ($\rho_{>0} = -0.029$, $n = 18$) and Train-Ticket ($-0.213$, $n = 22$). The three pub-sub systems hold up ($+0.559$ Autoware, $+0.704$ Home Assistant, $+0.304$ EdgeX). **We therefore report RQ4 as a negative result:** learned relational transfer to authentic open-source architectures is not established.

3.  **The QoS-weighted baseline is now scored, and it sharpens one comparison.** `Topo-QoS` reaches $\rho = 0.888$ on Cloud Microservices, where the learned model scores worst ($0.649$) and inverts on active components, sharpening the deficit on synchronous call trees.

4.  **What the full-population number does support.** Separating propagating components from non-propagating ones narrows the review surface effectively, reflected in the $F_1@K$ column ($0.517$ mean, $0.633$ on Autoware).

5.  **Architectural Ingestion Boundary.** The active-stratum inversion on microservices emphasizes a fundamental domain mismatch: synchronous RPC architectures propagate chain failures backward along invocation trees via timeout accumulation and thread pool starvation [45], whereas asynchronous pub-sub architectures cascade forward via queue saturation and topic starvation. Since `HGT-QoS` was trained only on pub-sub communication semantics, its relational inductive bias inverts when applied to synchronous call trees. A strict ingestion boundary is therefore recommended: SaG’s learned GNN pipeline should be deployed on asynchronous and event-driven architectures (ROS 2, Kafka, DDS, MQTT), while closed-form structural baselines (`Topo-QoS`, $
ho = 0.888$) or dedicated static call-graph analyzers should be used for synchronous RPC/REST microservice meshes.

## 7.5 RQ5: Analysis Cost and Its Comparison Against Simulation

RQ5 quantifies computing overhead and sustainability during CI/CD evaluation, with per-stage latencies reported in Table 9:

**Table 9.** Per-stage latency of the inference pipeline across scaling graph sizes (CPU, median of 3 runs).

| **$|V|$** | **$|E|$** | **Analyze (s)** | **Graph$\to$tensor (s)** | **HGT forward (ms)** | **Analyze : forward** |
|----------:|----------:|----------------:|-------------------------:|---------------------:|----------------------:|
|       249 |     1,127 |            1.74 |                    0.010 |                 26.5 |            66$\times$ |
|       499 |     2,402 |            8.32 |                    0.022 |                 16.4 |           509$\times$ |
|       999 |     6,422 |           44.54 |                    0.056 |                 21.1 |         2,108$\times$ |
|     1,998 |    19,301 |          239.34 |                    0.157 |                 56.2 |     **4,259$\times$** |

The neural model constitutes the least expensive stage, whereas the deterministic model incurs considerable computational cost. For 2,000 components, the HGT forward pass requires $56\,\text{ms}$, compared to $239\,\text{s}$ for deterministic structural analysis, yielding a ratio of $4{,}259\times$. This ratio reflects pipeline cost rather than the total cost of architectural evaluation. Indices 0–17 of each node feature vector (§3.4)—including betweenness, closeness, reverse PageRank, articulation, and bridge scores—are generated during this analysis stage, making the forward pass dependent on its completion. End-to-end evaluation of an unseen 2,000-component architecture takes about four minutes, with the learned model accounting for only $0.02\%$ of that time; the $56\,\text{ms}$ figure represents the marginal cost of re-scoring an already-analyzed graph. Across the eight-scenario detection benchmark, the complete gate (structural analysis plus 18 anti-pattern detectors) executes in $0.04$–$82.7\,\text{s}$, with the upper bound corresponding to the 520-component Enterprise mesh.

**One metric dominates cost, and it grew.** Measured cost now tracks the stage’s $O(|V|^2 + |V||E|)$ bound closely: from 249 to 1,998 components wall-clock rises $138\times$ against a $137\times$ growth in $|V||E|$. The dominant term is the Connectivity Degradation Index, which is computed for every node in the main connected component rather than for articulation points alone. That choice is deliberate and is a correctness requirement rather than an oversight: gating CDI to articulation points leaves it identically zero for every node whose removal does not literally disconnect the graph, which drives $A(v)$ to a near-constant in the redundant multi-publisher topologies this system targets. The cost is the price of a non-degenerate Availability score, and we report it rather than the cheaper gated variant we could have measured.

### 7.5.1 The Gate Is Not Cheaper Than the Simulation It Replaces

The framing that motivated this analysis — static gating as a low-cost substitute for dynamic simulation — does not survive measurement against our own oracle. Timing the `FaultInjector` labeling sweep (five seeds, node types Application/Broker/Library, the full ground-truth run) on the same corpus and the same idle hardware gives $0.14$–$7.2\,\text{s}$ per scenario, against $0.04$–$82.7\,\text{s}$ for the analysis gate. Both maxima occur in the 520-component Enterprise mesh, so the largest scenario compares $7.2\,\text{s}$ of simulation against $82.7\,\text{s}$ from static analysis: the gate costs roughly eleven times as much as the simulation it is meant to replace.

This finding refutes the assumption that static analysis is computationally cheaper than in-process simulation: breadth-first cascade traversal is simpler than computing all-pairs connectivity degradation ($O(|V|^2 + |V||E|)$). However, in practical continuous integration workflows, static SSA provides a key deployment trade-off: (1) it scores components (e.g., shared libraries, hosts) and dependency edges that node-level simulation passes do not evaluate; and (2) deterministic graph metrics can be incrementally cached across git commits, recomputing only the $k$-hop neighborhood touched by an architectural pull request. We clarify that the benchmark timings in Table 9 reflect full from-scratch recomputation without caching; once cached, GNN scoring executes in $56\,\text{ms}$, whereas repeating full simulation sweeps requires re-running stochastic traversals globally. Without such caching, direct simulation is strictly faster.